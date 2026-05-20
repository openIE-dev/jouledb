//! Job scheduler — cron-like scheduling, one-shot delayed jobs, recurring jobs,
//! job priority, job queue with workers, status tracking, retry with backoff,
//! and job history/audit.
//!
//! Replaces Node.js schedulers (node-cron, Agenda, BullMQ) with a pure-Rust
//! job scheduler that tracks every job from creation to completion.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// ── Errors ──────────────────────────────────────────────────────

/// Scheduler domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// Job not found.
    JobNotFound(String),
    /// Duplicate job ID.
    DuplicateJob(String),
    /// Invalid cron expression.
    InvalidCron(String),
    /// Job already completed.
    AlreadyCompleted(String),
    /// Queue is full.
    QueueFull { capacity: usize },
    /// Worker not found.
    WorkerNotFound(String),
    /// Max retries exceeded.
    MaxRetriesExceeded { job_id: String, attempts: u32 },
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobNotFound(id) => write!(f, "job not found: {id}"),
            Self::DuplicateJob(id) => write!(f, "duplicate job: {id}"),
            Self::InvalidCron(expr) => write!(f, "invalid cron expression: {expr}"),
            Self::AlreadyCompleted(id) => write!(f, "job already completed: {id}"),
            Self::QueueFull { capacity } => write!(f, "queue full (capacity {capacity})"),
            Self::WorkerNotFound(id) => write!(f, "worker not found: {id}"),
            Self::MaxRetriesExceeded { job_id, attempts } => {
                write!(f, "job {job_id} exceeded max retries ({attempts} attempts)")
            }
        }
    }
}

impl std::error::Error for SchedulerError {}

// ── Enums ───────────────────────────────────────────────────────

/// Job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Retrying,
    Cancelled,
}

/// Job schedule type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Schedule {
    /// Run once at a specific time.
    OneShot { run_at: DateTime<Utc> },
    /// Run once after a delay.
    Delayed { delay_seconds: u64 },
    /// Recurring with interval in seconds.
    Recurring { interval_seconds: u64 },
    /// Cron-like schedule (simplified: minute, hour, day_of_month, month, day_of_week).
    Cron { expression: String },
    /// Run immediately.
    Immediate,
}

/// Retry policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub backoff_multiplier: u32,
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2,
            max_delay_ms: 60_000,
        }
    }
}

impl RetryPolicy {
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let mut delay = self.initial_delay_ms;
        for _ in 0..attempt {
            delay = delay.saturating_mul(self.backoff_multiplier as u64);
        }
        delay.min(self.max_delay_ms)
    }
}

// ── Simplified Cron Parser ──────────────────────────────────────

/// Simplified cron schedule (5-field: min hour dom month dow).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CronSchedule {
    pub minute: CronField,
    pub hour: CronField,
    pub day_of_month: CronField,
    pub month: CronField,
    pub day_of_week: CronField,
}

/// A cron field value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CronField {
    Any,
    Value(u32),
    Range(u32, u32),
    Step(u32),
}

impl CronSchedule {
    /// Parse a simplified cron expression.
    pub fn parse(expr: &str) -> Result<Self, SchedulerError> {
        let parts: Vec<&str> = expr.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(SchedulerError::InvalidCron(expr.to_string()));
        }
        Ok(Self {
            minute: Self::parse_field(parts[0])?,
            hour: Self::parse_field(parts[1])?,
            day_of_month: Self::parse_field(parts[2])?,
            month: Self::parse_field(parts[3])?,
            day_of_week: Self::parse_field(parts[4])?,
        })
    }

    fn parse_field(field: &str) -> Result<CronField, SchedulerError> {
        if field == "*" {
            return Ok(CronField::Any);
        }
        if let Some(step) = field.strip_prefix("*/") {
            return step.parse::<u32>()
                .map(CronField::Step)
                .map_err(|_| SchedulerError::InvalidCron(field.to_string()));
        }
        if let Some((start, end)) = field.split_once('-') {
            let s = start.parse::<u32>().map_err(|_| SchedulerError::InvalidCron(field.to_string()))?;
            let e = end.parse::<u32>().map_err(|_| SchedulerError::InvalidCron(field.to_string()))?;
            return Ok(CronField::Range(s, e));
        }
        field.parse::<u32>()
            .map(CronField::Value)
            .map_err(|_| SchedulerError::InvalidCron(field.to_string()))
    }

    /// Check if a specific value matches a cron field.
    pub fn matches_field(field: &CronField, value: u32) -> bool {
        match field {
            CronField::Any => true,
            CronField::Value(v) => value == *v,
            CronField::Range(s, e) => value >= *s && value <= *e,
            CronField::Step(s) => *s > 0 && value % *s == 0,
        }
    }
}

// ── Job Definition ──────────────────────────────────────────────

/// A job definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub description: String,
    pub schedule: Schedule,
    pub priority: i32,
    pub status: JobStatus,
    pub retry_policy: Option<RetryPolicy>,
    pub attempt: u32,
    pub payload: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub next_run: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub worker_id: Option<String>,
}

impl Job {
    pub fn new(id: impl Into<String>, name: impl Into<String>, schedule: Schedule) -> Self {
        let now = Utc::now();
        let next = match &schedule {
            Schedule::Immediate => Some(now),
            Schedule::OneShot { run_at } => Some(*run_at),
            Schedule::Delayed { delay_seconds } => {
                Some(now + Duration::seconds(*delay_seconds as i64))
            }
            Schedule::Recurring { interval_seconds } => {
                Some(now + Duration::seconds(*interval_seconds as i64))
            }
            Schedule::Cron { .. } => None, // computed separately
        };
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            schedule,
            priority: 0,
            status: JobStatus::Pending,
            retry_policy: None,
            attempt: 0,
            payload: HashMap::new(),
            created_at: now,
            scheduled_at: next,
            started_at: None,
            completed_at: None,
            next_run: next,
            error: None,
            worker_id: None,
        }
    }

    pub fn with_priority(mut self, p: i32) -> Self {
        self.priority = p;
        self
    }

    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    pub fn with_payload(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.payload.insert(key.into(), val.into());
        self
    }
}

// ── Priority Queue Entry ────────────────────────────────────────

#[derive(Debug, Clone, Eq, PartialEq)]
struct PriorityEntry {
    job_id: String,
    priority: i32,
    scheduled_at: DateTime<Utc>,
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier scheduled time.
        self.priority.cmp(&other.priority)
            .then_with(|| other.scheduled_at.cmp(&self.scheduled_at))
    }
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Job History Entry ───────────────────────────────────────────

/// A history entry for job execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobHistoryEntry {
    pub job_id: String,
    pub attempt: u32,
    pub status: JobStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
    pub worker_id: Option<String>,
}

// ── Scheduler ───────────────────────────────────────────────────

/// The job scheduler.
#[derive(Debug)]
pub struct Scheduler {
    pub jobs: HashMap<String, Job>,
    queue: BinaryHeap<PriorityEntry>,
    pub history: Vec<JobHistoryEntry>,
    pub workers: HashMap<String, WorkerInfo>,
    pub max_queue_size: usize,
}

/// Worker information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub name: String,
    pub active_jobs: Vec<String>,
    pub max_concurrent: usize,
}

impl WorkerInfo {
    pub fn new(id: impl Into<String>, name: impl Into<String>, max_concurrent: usize) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            active_jobs: Vec::new(),
            max_concurrent,
        }
    }

    pub fn is_available(&self) -> bool {
        self.active_jobs.len() < self.max_concurrent
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            queue: BinaryHeap::new(),
            history: Vec::new(),
            workers: HashMap::new(),
            max_queue_size: 10_000,
        }
    }

    /// Register a worker.
    pub fn register_worker(&mut self, worker: WorkerInfo) {
        self.workers.insert(worker.id.clone(), worker);
    }

    /// Submit a job.
    pub fn submit(&mut self, job: Job) -> Result<(), SchedulerError> {
        if self.jobs.contains_key(&job.id) {
            return Err(SchedulerError::DuplicateJob(job.id));
        }
        if self.queue.len() >= self.max_queue_size {
            return Err(SchedulerError::QueueFull { capacity: self.max_queue_size });
        }
        let entry = PriorityEntry {
            job_id: job.id.clone(),
            priority: job.priority,
            scheduled_at: job.scheduled_at.unwrap_or_else(Utc::now),
        };
        self.jobs.insert(job.id.clone(), job);
        self.queue.push(entry);
        Ok(())
    }

    /// Get the next job from the queue.
    pub fn next_job(&mut self) -> Option<String> {
        while let Some(entry) = self.queue.pop() {
            if let Some(job) = self.jobs.get(&entry.job_id) {
                if job.status == JobStatus::Pending || job.status == JobStatus::Scheduled || job.status == JobStatus::Retrying {
                    return Some(entry.job_id);
                }
            }
        }
        None
    }

    /// Start a job (assign to worker).
    pub fn start_job(&mut self, job_id: &str, worker_id: &str) -> Result<(), SchedulerError> {
        let job = self.jobs.get_mut(job_id)
            .ok_or_else(|| SchedulerError::JobNotFound(job_id.to_string()))?;
        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());
        job.attempt += 1;
        job.worker_id = Some(worker_id.to_string());

        if let Some(worker) = self.workers.get_mut(worker_id) {
            worker.active_jobs.push(job_id.to_string());
        }
        Ok(())
    }

    /// Complete a job successfully.
    pub fn complete_job(&mut self, job_id: &str) -> Result<(), SchedulerError> {
        let job = self.jobs.get_mut(job_id)
            .ok_or_else(|| SchedulerError::JobNotFound(job_id.to_string()))?;
        let now = Utc::now();
        job.status = JobStatus::Completed;
        job.completed_at = Some(now);

        self.history.push(JobHistoryEntry {
            job_id: job_id.to_string(),
            attempt: job.attempt,
            status: JobStatus::Completed,
            started_at: job.started_at.unwrap_or(now),
            completed_at: Some(now),
            error: None,
            worker_id: job.worker_id.clone(),
        });

        // Remove from worker.
        if let Some(wid) = &job.worker_id {
            if let Some(worker) = self.workers.get_mut(wid.as_str()) {
                worker.active_jobs.retain(|j| j != job_id);
            }
        }

        // Handle recurring: re-enqueue.
        if let Schedule::Recurring { interval_seconds } = &job.schedule {
            let next = now + Duration::seconds(*interval_seconds as i64);
            job.status = JobStatus::Pending;
            job.completed_at = None;
            job.started_at = None;
            job.next_run = Some(next);
            job.scheduled_at = Some(next);
            job.worker_id = None;
            self.queue.push(PriorityEntry {
                job_id: job_id.to_string(),
                priority: job.priority,
                scheduled_at: next,
            });
        }
        Ok(())
    }

    /// Fail a job.
    pub fn fail_job(&mut self, job_id: &str, error: &str) -> Result<(), SchedulerError> {
        let job = self.jobs.get_mut(job_id)
            .ok_or_else(|| SchedulerError::JobNotFound(job_id.to_string()))?;
        let now = Utc::now();

        self.history.push(JobHistoryEntry {
            job_id: job_id.to_string(),
            attempt: job.attempt,
            status: JobStatus::Failed,
            started_at: job.started_at.unwrap_or(now),
            completed_at: Some(now),
            error: Some(error.to_string()),
            worker_id: job.worker_id.clone(),
        });

        // Remove from worker.
        if let Some(wid) = &job.worker_id {
            if let Some(worker) = self.workers.get_mut(wid.as_str()) {
                worker.active_jobs.retain(|j| j != job_id);
            }
        }

        // Check retry.
        if let Some(policy) = &job.retry_policy {
            if job.attempt < policy.max_attempts {
                let delay = policy.delay_for_attempt(job.attempt);
                let next = now + Duration::milliseconds(delay as i64);
                job.status = JobStatus::Retrying;
                job.error = Some(error.to_string());
                job.next_run = Some(next);
                job.scheduled_at = Some(next);
                job.worker_id = None;
                self.queue.push(PriorityEntry {
                    job_id: job_id.to_string(),
                    priority: job.priority,
                    scheduled_at: next,
                });
                return Ok(());
            }
        }

        job.status = JobStatus::Failed;
        job.error = Some(error.to_string());
        job.completed_at = Some(now);
        Ok(())
    }

    /// Cancel a job.
    pub fn cancel_job(&mut self, job_id: &str) -> Result<(), SchedulerError> {
        let job = self.jobs.get_mut(job_id)
            .ok_or_else(|| SchedulerError::JobNotFound(job_id.to_string()))?;
        if job.status == JobStatus::Completed {
            return Err(SchedulerError::AlreadyCompleted(job_id.to_string()));
        }
        job.status = JobStatus::Cancelled;
        job.completed_at = Some(Utc::now());
        Ok(())
    }

    /// Get jobs by status.
    pub fn jobs_by_status(&self, status: JobStatus) -> Vec<&Job> {
        self.jobs.values().filter(|j| j.status == status).collect()
    }

    /// Get job count.
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Get history for a specific job.
    pub fn job_history(&self, job_id: &str) -> Vec<&JobHistoryEntry> {
        self.history.iter().filter(|h| h.job_id == job_id).collect()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_and_next() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("j1", "Job 1", Schedule::Immediate)).unwrap();
        let next = sched.next_job();
        assert_eq!(next, Some("j1".to_string()));
    }

    #[test]
    fn test_duplicate_job() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("j1", "J", Schedule::Immediate)).unwrap();
        assert!(matches!(
            sched.submit(Job::new("j1", "J2", Schedule::Immediate)),
            Err(SchedulerError::DuplicateJob(_))
        ));
    }

    #[test]
    fn test_priority_ordering() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("low", "Low", Schedule::Immediate).with_priority(1)).unwrap();
        sched.submit(Job::new("high", "High", Schedule::Immediate).with_priority(10)).unwrap();
        let next = sched.next_job().unwrap();
        assert_eq!(next, "high");
    }

    #[test]
    fn test_job_lifecycle() {
        let mut sched = Scheduler::new();
        sched.register_worker(WorkerInfo::new("w1", "Worker 1", 5));
        sched.submit(Job::new("j1", "J1", Schedule::Immediate)).unwrap();

        let jid = sched.next_job().unwrap();
        sched.start_job(&jid, "w1").unwrap();
        assert_eq!(sched.jobs["j1"].status, JobStatus::Running);

        sched.complete_job("j1").unwrap();
        assert_eq!(sched.jobs["j1"].status, JobStatus::Completed);
        assert_eq!(sched.history.len(), 1);
    }

    #[test]
    fn test_retry_on_failure() {
        let mut sched = Scheduler::new();
        sched.submit(
            Job::new("j1", "J1", Schedule::Immediate)
                .with_retry(RetryPolicy { max_attempts: 3, ..Default::default() }),
        ).unwrap();

        sched.next_job().unwrap();
        sched.start_job("j1", "w").unwrap();
        sched.fail_job("j1", "transient error").unwrap();

        assert_eq!(sched.jobs["j1"].status, JobStatus::Retrying);
        // Should be re-queued.
        assert!(sched.next_job().is_some());
    }

    #[test]
    fn test_max_retries_exceeded() {
        let mut sched = Scheduler::new();
        sched.submit(
            Job::new("j1", "J1", Schedule::Immediate)
                .with_retry(RetryPolicy { max_attempts: 1, ..Default::default() }),
        ).unwrap();

        sched.next_job().unwrap();
        sched.start_job("j1", "w").unwrap();
        sched.fail_job("j1", "error").unwrap();
        // attempt = 1, max = 1 → no more retries
        assert_eq!(sched.jobs["j1"].status, JobStatus::Failed);
    }

    #[test]
    fn test_cancel_job() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("j1", "J1", Schedule::Immediate)).unwrap();
        sched.cancel_job("j1").unwrap();
        assert_eq!(sched.jobs["j1"].status, JobStatus::Cancelled);
    }

    #[test]
    fn test_cancel_completed_fails() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("j1", "J1", Schedule::Immediate)).unwrap();
        sched.next_job().unwrap();
        sched.start_job("j1", "w").unwrap();
        sched.complete_job("j1").unwrap();
        assert!(matches!(sched.cancel_job("j1"), Err(SchedulerError::AlreadyCompleted(_))));
    }

    #[test]
    fn test_recurring_job() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("j1", "J1", Schedule::Recurring { interval_seconds: 60 })).unwrap();

        sched.next_job().unwrap();
        sched.start_job("j1", "w").unwrap();
        sched.complete_job("j1").unwrap();

        // Should be re-queued as pending.
        assert_eq!(sched.jobs["j1"].status, JobStatus::Pending);
        assert!(sched.next_job().is_some());
    }

    #[test]
    fn test_cron_parse() {
        let cron = CronSchedule::parse("*/5 * * * *").unwrap();
        assert_eq!(cron.minute, CronField::Step(5));
        assert_eq!(cron.hour, CronField::Any);
    }

    #[test]
    fn test_cron_parse_range() {
        let cron = CronSchedule::parse("0 9-17 * * *").unwrap();
        assert_eq!(cron.hour, CronField::Range(9, 17));
    }

    #[test]
    fn test_cron_parse_invalid() {
        assert!(CronSchedule::parse("bad").is_err());
    }

    #[test]
    fn test_cron_field_matching() {
        assert!(CronSchedule::matches_field(&CronField::Any, 42));
        assert!(CronSchedule::matches_field(&CronField::Value(5), 5));
        assert!(!CronSchedule::matches_field(&CronField::Value(5), 6));
        assert!(CronSchedule::matches_field(&CronField::Range(1, 10), 5));
        assert!(!CronSchedule::matches_field(&CronField::Range(1, 10), 11));
        assert!(CronSchedule::matches_field(&CronField::Step(5), 15));
    }

    #[test]
    fn test_worker_availability() {
        let mut worker = WorkerInfo::new("w1", "W1", 2);
        assert!(worker.is_available());
        worker.active_jobs.push("j1".into());
        worker.active_jobs.push("j2".into());
        assert!(!worker.is_available());
    }

    #[test]
    fn test_jobs_by_status() {
        let mut sched = Scheduler::new();
        sched.submit(Job::new("j1", "J1", Schedule::Immediate)).unwrap();
        sched.submit(Job::new("j2", "J2", Schedule::Immediate)).unwrap();
        sched.next_job().unwrap();
        sched.start_job("j1", "w").unwrap();

        assert_eq!(sched.jobs_by_status(JobStatus::Running).len(), 1);
        assert_eq!(sched.jobs_by_status(JobStatus::Pending).len(), 1);
    }

    #[test]
    fn test_retry_backoff() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 100,
            backoff_multiplier: 2,
            max_delay_ms: 1000,
        };
        assert_eq!(policy.delay_for_attempt(0), 100);
        assert_eq!(policy.delay_for_attempt(1), 200);
        assert_eq!(policy.delay_for_attempt(2), 400);
        assert_eq!(policy.delay_for_attempt(4), 1000); // capped
    }
}
