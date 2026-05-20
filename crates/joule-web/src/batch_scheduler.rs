//! Batch job scheduler — job definitions, cron scheduling, job chaining
//! (job A triggers job B), failure handling (retry/skip/halt), execution log,
//! job priority, resource locking, and concurrent job limits.
//!
//! Replaces Node.js batch schedulers (Agenda, BullMQ, node-cron) with a
//! pure-Rust batch scheduler that tracks every job execution from start to finish.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

// ── Errors ──────────────────────────────────────────────────────

/// Batch scheduler domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    /// Job not found.
    JobNotFound(String),
    /// Duplicate job ID.
    DuplicateJob(String),
    /// Resource already locked.
    ResourceLocked { resource: String, holder: String },
    /// Resource not locked.
    ResourceNotLocked(String),
    /// Concurrent job limit reached.
    ConcurrencyLimitReached(usize),
    /// Invalid cron expression.
    InvalidCron(String),
    /// Job chain creates a cycle.
    CyclicChain { from: String, to: String },
    /// Max retries exceeded.
    MaxRetriesExceeded { job_id: String, attempts: u32 },
    /// Job already running.
    AlreadyRunning(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobNotFound(id) => write!(f, "job not found: {id}"),
            Self::DuplicateJob(id) => write!(f, "duplicate job: {id}"),
            Self::ResourceLocked { resource, holder } => {
                write!(f, "resource {resource} locked by {holder}")
            }
            Self::ResourceNotLocked(r) => write!(f, "resource not locked: {r}"),
            Self::ConcurrencyLimitReached(limit) => {
                write!(f, "concurrency limit reached: {limit}")
            }
            Self::InvalidCron(expr) => write!(f, "invalid cron expression: {expr}"),
            Self::CyclicChain { from, to } => {
                write!(f, "cyclic chain detected: {from} -> {to}")
            }
            Self::MaxRetriesExceeded { job_id, attempts } => {
                write!(f, "job {job_id} exceeded max retries ({attempts})")
            }
            Self::AlreadyRunning(id) => write!(f, "job already running: {id}"),
        }
    }
}

impl std::error::Error for BatchError {}

// ── Enums ───────────────────────────────────────────────────────

/// Job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobStatus {
    Defined,
    Scheduled,
    Waiting,
    Running,
    Completed,
    Failed,
    Retrying,
    Skipped,
    Halted,
    Cancelled,
}

/// How to handle a failed job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FailureAction {
    /// Retry the job up to max_retries times.
    Retry,
    /// Skip this job and continue the chain.
    Skip,
    /// Halt the entire chain.
    Halt,
}

impl Default for FailureAction {
    fn default() -> Self {
        Self::Retry
    }
}

/// Job priority (higher value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl JobPriority {
    fn as_u32(self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

impl Default for JobPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl PartialOrd for JobPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for JobPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_u32().cmp(&other.as_u32())
    }
}

/// Schedule type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScheduleType {
    /// Run once immediately.
    Immediate,
    /// Run once at a specific time.
    OneShot { run_at: DateTime<Utc> },
    /// Run at regular intervals.
    Interval { seconds: u64 },
    /// Cron expression (simple: "minute hour day_of_month month day_of_week").
    Cron { expression: String },
}

// ── Data Structures ─────────────────────────────────────────────

/// Execution log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionLog {
    pub job_id: String,
    pub execution_id: u64,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub status: JobStatus,
    pub duration_ms: Option<u64>,
    pub error_message: Option<String>,
    pub attempt: u32,
}

/// Resource lock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLock {
    pub resource: String,
    pub holder_job_id: String,
    pub acquired_at: DateTime<Utc>,
}

/// Retry policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub delay_seconds: u64,
    pub backoff_multiplier: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            delay_seconds: 10,
            backoff_multiplier: 2,
        }
    }
}

/// Chain trigger: when this job completes, trigger another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainTrigger {
    pub on_job_id: String,
    pub trigger_job_id: String,
    /// Only trigger on successful completion.
    pub on_success_only: bool,
}

/// Job definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobDef {
    pub id: String,
    pub name: String,
    pub schedule: ScheduleType,
    pub priority: JobPriority,
    pub failure_action: FailureAction,
    pub retry_policy: RetryPolicy,
    pub required_resources: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub enabled: bool,
}

/// Runtime state of a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobState {
    pub def: JobDef,
    pub status: JobStatus,
    pub current_attempt: u32,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// Prioritized job entry for the ready queue.
#[derive(Debug, Clone, Eq, PartialEq)]
struct PrioritizedJob {
    job_id: String,
    priority: JobPriority,
    scheduled_at: DateTime<Utc>,
}

impl Ord for PrioritizedJob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.scheduled_at.cmp(&self.scheduled_at))
    }
}

impl PartialOrd for PrioritizedJob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Engine ──────────────────────────────────────────────────────

/// Batch job scheduler.
pub struct BatchScheduler {
    jobs: HashMap<String, JobState>,
    chains: Vec<ChainTrigger>,
    ready_queue: BinaryHeap<PrioritizedJob>,
    running_jobs: HashSet<String>,
    locks: HashMap<String, ResourceLock>,
    execution_log: Vec<ExecutionLog>,
    max_concurrent: usize,
    next_execution_id: u64,
}

impl BatchScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            jobs: HashMap::new(),
            chains: Vec::new(),
            ready_queue: BinaryHeap::new(),
            running_jobs: HashSet::new(),
            locks: HashMap::new(),
            execution_log: Vec::new(),
            max_concurrent,
            next_execution_id: 1,
        }
    }

    // ── Job Management ──────────────────────────────────────────

    /// Register a job definition.
    pub fn register_job(&mut self, def: JobDef) -> Result<(), BatchError> {
        if self.jobs.contains_key(&def.id) {
            return Err(BatchError::DuplicateJob(def.id.clone()));
        }
        let now = Utc::now();
        let next_run = compute_next_run(&def.schedule, now);
        let state = JobState {
            def,
            status: JobStatus::Defined,
            current_attempt: 0,
            last_run: None,
            next_run,
            created_at: now,
        };
        let id = state.def.id.clone();
        self.jobs.insert(id, state);
        Ok(())
    }

    /// Remove a job.
    pub fn remove_job(&mut self, job_id: &str) -> Result<JobDef, BatchError> {
        let state = self
            .jobs
            .remove(job_id)
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;
        self.running_jobs.remove(job_id);
        // Remove chains.
        self.chains.retain(|c| c.on_job_id != job_id && c.trigger_job_id != job_id);
        Ok(state.def)
    }

    /// Get job state.
    pub fn get_job(&self, job_id: &str) -> Option<&JobState> {
        self.jobs.get(job_id)
    }

    /// Enable/disable a job.
    pub fn set_enabled(&mut self, job_id: &str, enabled: bool) -> Result<(), BatchError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;
        job.def.enabled = enabled;
        Ok(())
    }

    // ── Chain Management ────────────────────────────────────────

    /// Add a chain trigger: when `on_job_id` finishes, trigger `trigger_job_id`.
    pub fn add_chain(
        &mut self,
        on_job_id: &str,
        trigger_job_id: &str,
        on_success_only: bool,
    ) -> Result<(), BatchError> {
        if !self.jobs.contains_key(on_job_id) {
            return Err(BatchError::JobNotFound(on_job_id.to_string()));
        }
        if !self.jobs.contains_key(trigger_job_id) {
            return Err(BatchError::JobNotFound(trigger_job_id.to_string()));
        }

        // Check for cycles.
        if self.would_create_cycle(trigger_job_id, on_job_id) {
            return Err(BatchError::CyclicChain {
                from: on_job_id.to_string(),
                to: trigger_job_id.to_string(),
            });
        }

        self.chains.push(ChainTrigger {
            on_job_id: on_job_id.to_string(),
            trigger_job_id: trigger_job_id.to_string(),
            on_success_only,
        });
        Ok(())
    }

    /// Check if adding an edge from -> to would create a cycle.
    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        // BFS from `from` to see if we can reach `to`.
        let mut visited = HashSet::new();
        let mut queue = vec![from.to_string()];
        while let Some(current) = queue.pop() {
            if current == to {
                return true;
            }
            if !visited.insert(current.clone()) {
                continue;
            }
            for chain in &self.chains {
                if chain.on_job_id == current {
                    queue.push(chain.trigger_job_id.clone());
                }
            }
        }
        false
    }

    // ── Scheduling ──────────────────────────────────────────────

    /// Schedule all jobs whose next_run is at or before `now`.
    pub fn tick(&mut self, now: DateTime<Utc>) -> Vec<String> {
        let mut scheduled = Vec::new();
        let job_ids: Vec<String> = self.jobs.keys().cloned().collect();

        for id in job_ids {
            let should_schedule = {
                let job = &self.jobs[&id];
                job.def.enabled
                    && job.status != JobStatus::Running
                    && job.status != JobStatus::Halted
                    && job.status != JobStatus::Cancelled
                    && job.next_run.map_or(false, |nr| nr <= now)
            };
            if should_schedule {
                let priority = self.jobs[&id].def.priority;
                self.ready_queue.push(PrioritizedJob {
                    job_id: id.clone(),
                    priority,
                    scheduled_at: now,
                });
                let job = self.jobs.get_mut(&id).unwrap();
                job.status = JobStatus::Scheduled;
                scheduled.push(id);
            }
        }
        scheduled
    }

    /// Dequeue the next job to run (respecting concurrency limit).
    pub fn dequeue_next(&mut self) -> Result<Option<String>, BatchError> {
        if self.running_jobs.len() >= self.max_concurrent {
            return Err(BatchError::ConcurrencyLimitReached(self.max_concurrent));
        }

        while let Some(entry) = self.ready_queue.pop() {
            // Check if the job is still in a schedulable state.
            if let Some(job) = self.jobs.get(&entry.job_id) {
                if job.status == JobStatus::Scheduled || job.status == JobStatus::Waiting {
                    return Ok(Some(entry.job_id));
                }
            }
        }
        Ok(None)
    }

    // ── Job Execution ───────────────────────────────────────────

    /// Start executing a job.
    pub fn start_job(&mut self, job_id: &str) -> Result<u64, BatchError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;

        if job.status == JobStatus::Running {
            return Err(BatchError::AlreadyRunning(job_id.to_string()));
        }

        // Check resource locks.
        for resource in &job.def.required_resources {
            if let Some(lock) = self.locks.get(resource) {
                if lock.holder_job_id != job_id {
                    return Err(BatchError::ResourceLocked {
                        resource: resource.clone(),
                        holder: lock.holder_job_id.clone(),
                    });
                }
            }
        }

        // Acquire resource locks.
        let now = Utc::now();
        let resources = job.def.required_resources.clone();
        let jid = job_id.to_string();
        for resource in &resources {
            self.locks.insert(
                resource.clone(),
                ResourceLock {
                    resource: resource.clone(),
                    holder_job_id: jid.clone(),
                    acquired_at: now,
                },
            );
        }

        let job = self.jobs.get_mut(job_id).unwrap();
        job.status = JobStatus::Running;
        job.current_attempt += 1;
        job.last_run = Some(now);
        self.running_jobs.insert(job_id.to_string());

        let exec_id = self.next_execution_id;
        self.next_execution_id += 1;

        self.execution_log.push(ExecutionLog {
            job_id: job_id.to_string(),
            execution_id: exec_id,
            started_at: now,
            finished_at: None,
            status: JobStatus::Running,
            duration_ms: None,
            error_message: None,
            attempt: job.current_attempt,
        });

        Ok(exec_id)
    }

    /// Complete a job successfully.
    pub fn complete_job(
        &mut self,
        job_id: &str,
        execution_id: u64,
    ) -> Result<Vec<String>, BatchError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;

        let now = Utc::now();
        job.status = JobStatus::Completed;
        job.current_attempt = 0;
        job.next_run = compute_next_run(&job.def.schedule, now);
        self.running_jobs.remove(job_id);

        // Release locks.
        let resources = job.def.required_resources.clone();
        for resource in &resources {
            self.locks.remove(resource);
        }

        // Update execution log.
        if let Some(log) = self
            .execution_log
            .iter_mut()
            .find(|l| l.execution_id == execution_id)
        {
            log.finished_at = Some(now);
            log.status = JobStatus::Completed;
            let started = log.started_at;
            log.duration_ms = Some((now - started).num_milliseconds().max(0) as u64);
        }

        // Trigger chains.
        let triggered = self.trigger_chains(job_id, true);
        Ok(triggered)
    }

    /// Fail a job.
    pub fn fail_job(
        &mut self,
        job_id: &str,
        execution_id: u64,
        error: &str,
    ) -> Result<JobStatus, BatchError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;

        let now = Utc::now();
        let failure_action = job.def.failure_action;
        let max_retries = job.def.retry_policy.max_retries;
        let attempt = job.current_attempt;

        self.running_jobs.remove(job_id);

        // Release locks.
        let resources = job.def.required_resources.clone();
        for resource in &resources {
            self.locks.remove(resource);
        }

        // Update execution log.
        if let Some(log) = self
            .execution_log
            .iter_mut()
            .find(|l| l.execution_id == execution_id)
        {
            log.finished_at = Some(now);
            log.status = JobStatus::Failed;
            log.error_message = Some(error.to_string());
            let started = log.started_at;
            log.duration_ms = Some((now - started).num_milliseconds().max(0) as u64);
        }

        let new_status = match failure_action {
            FailureAction::Retry if attempt < max_retries => {
                let delay = job.def.retry_policy.delay_seconds
                    * job.def.retry_policy.backoff_multiplier.pow(attempt.saturating_sub(1)) as u64;
                job.next_run = Some(now + Duration::seconds(delay as i64));
                job.status = JobStatus::Retrying;
                JobStatus::Retrying
            }
            FailureAction::Retry => {
                job.status = JobStatus::Failed;
                job.current_attempt = 0;
                JobStatus::Failed
            }
            FailureAction::Skip => {
                job.status = JobStatus::Skipped;
                job.current_attempt = 0;
                job.next_run = compute_next_run(&job.def.schedule, now);
                // Still trigger chains.
                self.trigger_chains(job_id, false);
                JobStatus::Skipped
            }
            FailureAction::Halt => {
                job.status = JobStatus::Halted;
                JobStatus::Halted
            }
        };

        Ok(new_status)
    }

    /// Cancel a job.
    pub fn cancel_job(&mut self, job_id: &str) -> Result<(), BatchError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))?;
        job.status = JobStatus::Cancelled;
        self.running_jobs.remove(job_id);

        let resources = job.def.required_resources.clone();
        for resource in &resources {
            self.locks.remove(resource);
        }
        Ok(())
    }

    // ── Chain Triggering ────────────────────────────────────────

    fn trigger_chains(&mut self, completed_job_id: &str, success: bool) -> Vec<String> {
        let triggers: Vec<ChainTrigger> = self
            .chains
            .iter()
            .filter(|c| c.on_job_id == completed_job_id)
            .filter(|c| !c.on_success_only || success)
            .cloned()
            .collect();

        let mut triggered = Vec::new();
        let now = Utc::now();
        for chain in &triggers {
            if let Some(job) = self.jobs.get_mut(&chain.trigger_job_id) {
                if job.def.enabled
                    && job.status != JobStatus::Running
                    && job.status != JobStatus::Halted
                {
                    job.status = JobStatus::Scheduled;
                    job.next_run = Some(now);
                    self.ready_queue.push(PrioritizedJob {
                        job_id: chain.trigger_job_id.clone(),
                        priority: job.def.priority,
                        scheduled_at: now,
                    });
                    triggered.push(chain.trigger_job_id.clone());
                }
            }
        }
        triggered
    }

    // ── Resource Locks ──────────────────────────────────────────

    /// Manually acquire a resource lock.
    pub fn acquire_lock(
        &mut self,
        resource: &str,
        job_id: &str,
    ) -> Result<(), BatchError> {
        if let Some(lock) = self.locks.get(resource) {
            return Err(BatchError::ResourceLocked {
                resource: resource.to_string(),
                holder: lock.holder_job_id.clone(),
            });
        }
        self.locks.insert(
            resource.to_string(),
            ResourceLock {
                resource: resource.to_string(),
                holder_job_id: job_id.to_string(),
                acquired_at: Utc::now(),
            },
        );
        Ok(())
    }

    /// Release a resource lock.
    pub fn release_lock(&mut self, resource: &str) -> Result<(), BatchError> {
        self.locks
            .remove(resource)
            .ok_or_else(|| BatchError::ResourceNotLocked(resource.to_string()))?;
        Ok(())
    }

    /// Check if a resource is locked.
    pub fn is_locked(&self, resource: &str) -> bool {
        self.locks.contains_key(resource)
    }

    // ── Querying ────────────────────────────────────────────────

    /// Get execution log for a job.
    pub fn execution_log_for(&self, job_id: &str) -> Vec<&ExecutionLog> {
        self.execution_log
            .iter()
            .filter(|l| l.job_id == job_id)
            .collect()
    }

    /// Get all execution logs.
    pub fn all_execution_logs(&self) -> &[ExecutionLog] {
        &self.execution_log
    }

    /// Get currently running job count.
    pub fn running_count(&self) -> usize {
        self.running_jobs.len()
    }

    /// List all job IDs.
    pub fn list_jobs(&self) -> Vec<&str> {
        self.jobs.keys().map(|k| k.as_str()).collect()
    }

    /// Get jobs by status.
    pub fn jobs_by_status(&self, status: JobStatus) -> Vec<&JobState> {
        self.jobs.values().filter(|j| j.status == status).collect()
    }

    /// Get max concurrent limit.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Get chain triggers for a job.
    pub fn chains_for_job(&self, job_id: &str) -> Vec<&ChainTrigger> {
        self.chains
            .iter()
            .filter(|c| c.on_job_id == job_id)
            .collect()
    }
}

// ── Schedule Helpers ────────────────────────────────────────────

fn compute_next_run(schedule: &ScheduleType, now: DateTime<Utc>) -> Option<DateTime<Utc>> {
    match schedule {
        ScheduleType::Immediate => Some(now),
        ScheduleType::OneShot { run_at } => {
            if *run_at > now {
                Some(*run_at)
            } else {
                None
            }
        }
        ScheduleType::Interval { seconds } => Some(now + Duration::seconds(*seconds as i64)),
        ScheduleType::Cron { .. } => {
            // Simplified: next minute.
            Some(now + Duration::seconds(60))
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str) -> JobDef {
        JobDef {
            id: id.to_string(),
            name: format!("Job {id}"),
            schedule: ScheduleType::Immediate,
            priority: JobPriority::Normal,
            failure_action: FailureAction::Retry,
            retry_policy: RetryPolicy::default(),
            required_resources: vec![],
            metadata: HashMap::new(),
            enabled: true,
        }
    }

    fn setup() -> BatchScheduler {
        let mut sched = BatchScheduler::new(4);
        sched.register_job(make_job("j1")).unwrap();
        sched.register_job(make_job("j2")).unwrap();
        sched
    }

    #[test]
    fn test_register_job() {
        let sched = setup();
        assert!(sched.get_job("j1").is_some());
    }

    #[test]
    fn test_duplicate_job() {
        let mut sched = setup();
        let err = sched.register_job(make_job("j1")).unwrap_err();
        assert!(matches!(err, BatchError::DuplicateJob(_)));
    }

    #[test]
    fn test_tick_schedules_jobs() {
        let mut sched = setup();
        let now = Utc::now();
        let scheduled = sched.tick(now);
        assert_eq!(scheduled.len(), 2);
    }

    #[test]
    fn test_dequeue_respects_priority() {
        let mut sched = BatchScheduler::new(10);
        let mut low = make_job("low");
        low.priority = JobPriority::Low;
        let mut high = make_job("high");
        high.priority = JobPriority::High;
        sched.register_job(low).unwrap();
        sched.register_job(high).unwrap();
        sched.tick(Utc::now());

        let first = sched.dequeue_next().unwrap().unwrap();
        assert_eq!(first, "high");
    }

    #[test]
    fn test_start_and_complete_job() {
        let mut sched = setup();
        sched.tick(Utc::now());
        let _id = sched.dequeue_next().unwrap();
        let exec_id = sched.start_job("j1").unwrap();
        assert_eq!(sched.running_count(), 1);
        sched.complete_job("j1", exec_id).unwrap();
        assert_eq!(sched.running_count(), 0);
        let job = sched.get_job("j1").unwrap();
        assert_eq!(job.status, JobStatus::Completed);
    }

    #[test]
    fn test_fail_job_retries() {
        let mut sched = setup();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let exec_id = sched.start_job("j1").unwrap();
        let status = sched.fail_job("j1", exec_id, "oops").unwrap();
        assert_eq!(status, JobStatus::Retrying);
    }

    #[test]
    fn test_fail_job_halts() {
        let mut sched = BatchScheduler::new(4);
        let mut job = make_job("j1");
        job.failure_action = FailureAction::Halt;
        sched.register_job(job).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let exec_id = sched.start_job("j1").unwrap();
        let status = sched.fail_job("j1", exec_id, "fatal").unwrap();
        assert_eq!(status, JobStatus::Halted);
    }

    #[test]
    fn test_fail_job_skips() {
        let mut sched = BatchScheduler::new(4);
        let mut job = make_job("j1");
        job.failure_action = FailureAction::Skip;
        sched.register_job(job).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let exec_id = sched.start_job("j1").unwrap();
        let status = sched.fail_job("j1", exec_id, "skip me").unwrap();
        assert_eq!(status, JobStatus::Skipped);
    }

    #[test]
    fn test_concurrency_limit() {
        let mut sched = BatchScheduler::new(1);
        sched.register_job(make_job("j1")).unwrap();
        sched.register_job(make_job("j2")).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        sched.start_job("j1").unwrap();
        let err = sched.dequeue_next().unwrap_err();
        assert!(matches!(err, BatchError::ConcurrencyLimitReached(1)));
    }

    #[test]
    fn test_resource_locking() {
        let mut sched = BatchScheduler::new(4);
        let mut job = make_job("j1");
        job.required_resources = vec!["db".to_string()];
        sched.register_job(job).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        sched.start_job("j1").unwrap();
        assert!(sched.is_locked("db"));
    }

    #[test]
    fn test_resource_conflict() {
        let mut sched = BatchScheduler::new(4);
        sched.acquire_lock("db", "j1").unwrap();
        let mut job = make_job("j2");
        job.required_resources = vec!["db".to_string()];
        sched.register_job(job).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let err = sched.start_job("j2").unwrap_err();
        assert!(matches!(err, BatchError::ResourceLocked { .. }));
    }

    #[test]
    fn test_chain_trigger() {
        let mut sched = setup();
        sched.add_chain("j1", "j2", true).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let exec_id = sched.start_job("j1").unwrap();
        let triggered = sched.complete_job("j1", exec_id).unwrap();
        assert!(triggered.contains(&"j2".to_string()));
    }

    #[test]
    fn test_chain_cycle_detection() {
        let mut sched = setup();
        sched.add_chain("j1", "j2", false).unwrap();
        let err = sched.add_chain("j2", "j1", false).unwrap_err();
        assert!(matches!(err, BatchError::CyclicChain { .. }));
    }

    #[test]
    fn test_cancel_job() {
        let mut sched = setup();
        sched.cancel_job("j1").unwrap();
        let job = sched.get_job("j1").unwrap();
        assert_eq!(job.status, JobStatus::Cancelled);
    }

    #[test]
    fn test_execution_log() {
        let mut sched = setup();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let exec_id = sched.start_job("j1").unwrap();
        sched.complete_job("j1", exec_id).unwrap();
        let log = sched.execution_log_for("j1");
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].status, JobStatus::Completed);
        assert!(log[0].finished_at.is_some());
    }

    #[test]
    fn test_set_enabled() {
        let mut sched = setup();
        sched.set_enabled("j1", false).unwrap();
        let job = sched.get_job("j1").unwrap();
        assert!(!job.def.enabled);

        // Disabled job should not be scheduled.
        sched.tick(Utc::now());
        // Only j2 should be scheduled.
        let scheduled = sched.jobs_by_status(JobStatus::Scheduled);
        assert!(scheduled.iter().all(|j| j.def.id != "j1"));
    }

    #[test]
    fn test_remove_job() {
        let mut sched = setup();
        sched.remove_job("j1").unwrap();
        assert!(sched.get_job("j1").is_none());
    }

    #[test]
    fn test_already_running() {
        let mut sched = setup();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        sched.start_job("j1").unwrap();
        let err = sched.start_job("j1").unwrap_err();
        assert!(matches!(err, BatchError::AlreadyRunning(_)));
    }

    #[test]
    fn test_release_lock() {
        let mut sched = BatchScheduler::new(4);
        sched.acquire_lock("db", "j1").unwrap();
        assert!(sched.is_locked("db"));
        sched.release_lock("db").unwrap();
        assert!(!sched.is_locked("db"));
    }

    #[test]
    fn test_release_nonexistent_lock() {
        let mut sched = BatchScheduler::new(4);
        let err = sched.release_lock("nope").unwrap_err();
        assert!(matches!(err, BatchError::ResourceNotLocked(_)));
    }

    #[test]
    fn test_interval_schedule_next_run() {
        let now = Utc::now();
        let next = compute_next_run(&ScheduleType::Interval { seconds: 300 }, now);
        assert!(next.is_some());
        let expected = now + Duration::seconds(300);
        assert_eq!(next.unwrap(), expected);
    }

    #[test]
    fn test_oneshot_past_returns_none() {
        let past = Utc::now() - Duration::seconds(1000);
        let next = compute_next_run(&ScheduleType::OneShot { run_at: past }, Utc::now());
        assert!(next.is_none());
    }

    #[test]
    fn test_chains_for_job() {
        let mut sched = setup();
        sched.add_chain("j1", "j2", true).unwrap();
        let chains = sched.chains_for_job("j1");
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].trigger_job_id, "j2");
    }

    #[test]
    fn test_jobs_by_status() {
        let sched = setup();
        let defined = sched.jobs_by_status(JobStatus::Defined);
        assert_eq!(defined.len(), 2);
    }

    #[test]
    fn test_max_retries_exhausted() {
        let mut sched = BatchScheduler::new(4);
        let mut job = make_job("j1");
        job.retry_policy.max_retries = 1;
        sched.register_job(job).unwrap();
        sched.tick(Utc::now());
        sched.dequeue_next().unwrap();
        let exec1 = sched.start_job("j1").unwrap();
        sched.fail_job("j1", exec1, "err1").unwrap(); // Attempt 1 -> retrying

        // Simulate next tick and retry.
        sched.tick(Utc::now() + Duration::seconds(100));
        sched.dequeue_next().ok();
        let exec2 = sched.start_job("j1").unwrap();
        let status = sched.fail_job("j1", exec2, "err2").unwrap(); // Attempt 2 -> failed (max=1)
        assert_eq!(status, JobStatus::Failed);
    }
}
