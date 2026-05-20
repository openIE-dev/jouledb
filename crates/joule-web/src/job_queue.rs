//! Background job queue — scheduling with priorities, delayed execution, retry
//! with exponential backoff, status tracking, concurrency limits, cancellation.
//!
//! Replaces Sidekiq/BullMQ/Celery job queue libraries with a pure-Rust job
//! queue that models the full lifecycle: pending -> running -> completed/failed,
//! with exponential backoff, priority ordering, and concurrency control.

use std::collections::{BTreeMap, HashMap, VecDeque};

// ── Errors ─────────────────────────────────────────────────────

/// Job queue domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobError {
    /// Job not found.
    JobNotFound(String),
    /// Concurrency limit reached.
    ConcurrencyLimitReached { limit: usize, running: usize },
    /// Job already exists.
    DuplicateJob(String),
    /// Job is not in a cancellable state.
    NotCancellable { job_id: String, status: JobStatus },
    /// Queue is full.
    QueueFull { limit: usize },
    /// Invalid retry config.
    InvalidRetry(String),
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobNotFound(id) => write!(f, "job not found: {id}"),
            Self::ConcurrencyLimitReached { limit, running } => {
                write!(f, "concurrency limit {limit} reached ({running} running)")
            }
            Self::DuplicateJob(id) => write!(f, "duplicate job: {id}"),
            Self::NotCancellable { job_id, status } => {
                write!(f, "job {job_id} not cancellable in state {status:?}")
            }
            Self::QueueFull { limit } => write!(f, "queue full (limit {limit})"),
            Self::InvalidRetry(msg) => write!(f, "invalid retry config: {msg}"),
        }
    }
}

impl std::error::Error for JobError {}

// ── Job Status ────────────────────────────────────────────────

/// Lifecycle status of a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobStatus {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Cancelled,
    Retrying,
}

// ── Priority ──────────────────────────────────────────────────

/// Job priority (lower numeric value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobPriority(pub u32);

impl JobPriority {
    pub const CRITICAL: Self = Self(0);
    pub const HIGH: Self = Self(10);
    pub const NORMAL: Self = Self(50);
    pub const LOW: Self = Self(100);
}

impl Default for JobPriority {
    fn default() -> Self {
        Self::NORMAL
    }
}

// ── Retry Policy ──────────────────────────────────────────────

/// Exponential backoff retry configuration.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 1000,
            max_delay_ms: 60_000,
            multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Compute delay for the given attempt (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let delay = self.base_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        (delay as u64).min(self.max_delay_ms)
    }
}

// ── Job ───────────────────────────────────────────────────────

/// A background job.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub payload: String,
    pub priority: JobPriority,
    pub status: JobStatus,
    pub attempt: u32,
    pub retry_policy: RetryPolicy,
    pub created_at_ms: u64,
    pub started_at_ms: Option<u64>,
    pub completed_at_ms: Option<u64>,
    /// Delay before becoming eligible (ms from creation).
    pub delay_ms: u64,
    /// Next retry eligible time (absolute ms).
    pub next_retry_at_ms: Option<u64>,
    /// Error message if failed.
    pub error: Option<String>,
    /// Tags for filtering.
    pub tags: Vec<String>,
}

impl Job {
    pub fn new(id: impl Into<String>, name: impl Into<String>, payload: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            payload: payload.into(),
            priority: JobPriority::default(),
            status: JobStatus::Pending,
            attempt: 0,
            retry_policy: RetryPolicy::default(),
            created_at_ms: 0,
            started_at_ms: None,
            completed_at_ms: None,
            delay_ms: 0,
            next_retry_at_ms: None,
            error: None,
            tags: Vec::new(),
        }
    }

    pub fn with_priority(mut self, p: JobPriority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_delay(mut self, ms: u64) -> Self {
        self.delay_ms = ms;
        self.status = JobStatus::Scheduled;
        self
    }

    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.retry_policy.max_retries = n;
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Whether the job is eligible for execution at the given time.
    fn is_eligible(&self, now_ms: u64) -> bool {
        if self.status == JobStatus::Scheduled {
            return now_ms.saturating_sub(self.created_at_ms) >= self.delay_ms;
        }
        if self.status == JobStatus::Retrying {
            return self
                .next_retry_at_ms
                .map_or(true, |t| now_ms >= t);
        }
        self.status == JobStatus::Pending
    }
}

// ── Queue Stats ───────────────────────────────────────────────

/// Job queue statistics.
#[derive(Debug, Clone, Default)]
pub struct JobQueueStats {
    pub total_submitted: u64,
    pub total_completed: u64,
    pub total_failed: u64,
    pub total_cancelled: u64,
    pub total_retries: u64,
    pub current_running: usize,
    pub current_pending: usize,
}

// ── Job Queue ─────────────────────────────────────────────────

/// Background job queue with priority scheduling and concurrency control.
#[derive(Debug)]
pub struct JobQueue {
    /// All jobs by ID.
    jobs: HashMap<String, Job>,
    /// Priority-ordered queue of job IDs.
    ready_queue: BTreeMap<JobPriority, VecDeque<String>>,
    /// Maximum concurrent running jobs.
    max_concurrency: usize,
    /// Currently running job IDs.
    running: Vec<String>,
    /// Simulated clock.
    clock_ms: u64,
    /// Max queue size (None = unlimited).
    max_queue_size: Option<usize>,
    /// Stats.
    stats: JobQueueStats,
}

impl Default for JobQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl JobQueue {
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            ready_queue: BTreeMap::new(),
            max_concurrency: 4,
            running: Vec::new(),
            clock_ms: 0,
            max_queue_size: None,
            stats: JobQueueStats::default(),
        }
    }

    pub fn with_concurrency(mut self, limit: usize) -> Self {
        self.max_concurrency = limit.max(1);
        self
    }

    pub fn with_max_queue_size(mut self, limit: usize) -> Self {
        self.max_queue_size = Some(limit);
        self
    }

    pub fn advance_time(&mut self, ms: u64) {
        self.clock_ms += ms;
    }

    pub fn set_clock(&mut self, ms: u64) {
        self.clock_ms = ms;
    }

    // ── Submit ────────────────────────────────────────────────

    /// Submit a job to the queue.
    pub fn submit(&mut self, mut job: Job) -> Result<(), JobError> {
        if self.jobs.contains_key(&job.id) {
            return Err(JobError::DuplicateJob(job.id));
        }
        if let Some(limit) = self.max_queue_size {
            let pending = self
                .jobs
                .values()
                .filter(|j| {
                    matches!(
                        j.status,
                        JobStatus::Pending | JobStatus::Scheduled | JobStatus::Retrying
                    )
                })
                .count();
            if pending >= limit {
                return Err(JobError::QueueFull { limit });
            }
        }
        job.created_at_ms = self.clock_ms;
        let id = job.id.clone();
        let priority = job.priority;
        self.jobs.insert(id.clone(), job);
        self.ready_queue
            .entry(priority)
            .or_insert_with(VecDeque::new)
            .push_back(id);
        self.stats.total_submitted += 1;
        self.stats.current_pending += 1;
        Ok(())
    }

    // ── Promote scheduled/retrying jobs ──────────────────────

    /// Check and promote scheduled and retrying jobs that are now eligible.
    pub fn promote_eligible(&mut self) -> Vec<String> {
        let now = self.clock_ms;
        let mut promoted = Vec::new();
        for job in self.jobs.values_mut() {
            if (job.status == JobStatus::Scheduled || job.status == JobStatus::Retrying)
                && job.is_eligible(now)
            {
                job.status = JobStatus::Pending;
                promoted.push((job.id.clone(), job.priority));
            }
        }
        let ids: Vec<String> = promoted.iter().map(|(id, _)| id.clone()).collect();
        for (id, priority) in promoted {
            self.ready_queue
                .entry(priority)
                .or_insert_with(VecDeque::new)
                .push_back(id);
        }
        ids
    }

    // ── Pick next ─────────────────────────────────────────────

    /// Pick the next eligible job, respecting concurrency limits.
    pub fn pick_next(&mut self) -> Result<Option<String>, JobError> {
        self.promote_eligible();

        if self.running.len() >= self.max_concurrency {
            return Err(JobError::ConcurrencyLimitReached {
                limit: self.max_concurrency,
                running: self.running.len(),
            });
        }

        let now = self.clock_ms;
        for (_prio, queue) in self.ready_queue.iter_mut() {
            while let Some(id) = queue.front() {
                let id_clone = id.clone();
                if let Some(job) = self.jobs.get(&id_clone) {
                    if job.status != JobStatus::Pending {
                        queue.pop_front();
                        continue;
                    }
                    if !job.is_eligible(now) {
                        queue.pop_front();
                        continue;
                    }
                } else {
                    queue.pop_front();
                    continue;
                }
                queue.pop_front();
                if let Some(job) = self.jobs.get_mut(&id_clone) {
                    job.status = JobStatus::Running;
                    job.attempt += 1;
                    job.started_at_ms = Some(now);
                    self.running.push(id_clone.clone());
                    self.stats.current_pending = self.stats.current_pending.saturating_sub(1);
                    self.stats.current_running += 1;
                    return Ok(Some(id_clone));
                }
            }
        }
        Ok(None)
    }

    // ── Complete ──────────────────────────────────────────────

    /// Mark a job as completed.
    pub fn complete(&mut self, job_id: &str) -> Result<(), JobError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| JobError::JobNotFound(job_id.to_string()))?;
        job.status = JobStatus::Completed;
        job.completed_at_ms = Some(self.clock_ms);
        self.running.retain(|id| id != job_id);
        self.stats.total_completed += 1;
        self.stats.current_running = self.stats.current_running.saturating_sub(1);
        Ok(())
    }

    // ── Fail ──────────────────────────────────────────────────

    /// Mark a job as failed. If retries remain, schedule retry with backoff.
    pub fn fail(&mut self, job_id: &str, error: impl Into<String>) -> Result<bool, JobError> {
        let error_str = error.into();
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| JobError::JobNotFound(job_id.to_string()))?;
        self.running.retain(|id| id != job_id);
        self.stats.current_running = self.stats.current_running.saturating_sub(1);

        if job.attempt < job.retry_policy.max_retries {
            // Schedule retry.
            let delay = job.retry_policy.delay_for_attempt(job.attempt);
            job.status = JobStatus::Retrying;
            job.next_retry_at_ms = Some(self.clock_ms + delay);
            job.error = Some(error_str);
            let priority = job.priority;
            let id = job.id.clone();
            self.ready_queue
                .entry(priority)
                .or_insert_with(VecDeque::new)
                .push_back(id);
            self.stats.total_retries += 1;
            self.stats.current_pending += 1;
            Ok(true)
        } else {
            job.status = JobStatus::Failed;
            job.error = Some(error_str);
            job.completed_at_ms = Some(self.clock_ms);
            self.stats.total_failed += 1;
            Ok(false)
        }
    }

    // ── Cancel ────────────────────────────────────────────────

    /// Cancel a pending/scheduled job.
    pub fn cancel(&mut self, job_id: &str) -> Result<(), JobError> {
        let job = self
            .jobs
            .get_mut(job_id)
            .ok_or_else(|| JobError::JobNotFound(job_id.to_string()))?;
        if !matches!(
            job.status,
            JobStatus::Pending | JobStatus::Scheduled | JobStatus::Retrying
        ) {
            let status = job.status;
            return Err(JobError::NotCancellable {
                job_id: job_id.to_string(),
                status,
            });
        }
        job.status = JobStatus::Cancelled;
        self.stats.total_cancelled += 1;
        self.stats.current_pending = self.stats.current_pending.saturating_sub(1);
        Ok(())
    }

    // ── Queries ──────────────────────────────────────────────

    /// Get a job by ID.
    pub fn get_job(&self, id: &str) -> Option<&Job> {
        self.jobs.get(id)
    }

    /// Get jobs by status.
    pub fn jobs_by_status(&self, status: JobStatus) -> Vec<&Job> {
        self.jobs.values().filter(|j| j.status == status).collect()
    }

    /// Get jobs by tag.
    pub fn jobs_by_tag(&self, tag: &str) -> Vec<&Job> {
        self.jobs
            .values()
            .filter(|j| j.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Queue stats.
    pub fn stats(&self) -> &JobQueueStats {
        &self.stats
    }

    /// Number of running jobs.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Total jobs tracked.
    pub fn total_jobs(&self) -> usize {
        self.jobs.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_queue() -> JobQueue {
        JobQueue::new().with_concurrency(2)
    }

    #[test]
    fn test_submit_and_pick() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "email", "send")).unwrap();
        let id = q.pick_next().unwrap().unwrap();
        assert_eq!(id, "j1");
        assert_eq!(q.get_job("j1").unwrap().status, JobStatus::Running);
    }

    #[test]
    fn test_priority_ordering() {
        let mut q = make_queue();
        q.submit(Job::new("low", "task", "a").with_priority(JobPriority::LOW)).unwrap();
        q.submit(Job::new("high", "task", "b").with_priority(JobPriority::HIGH)).unwrap();
        q.submit(Job::new("crit", "task", "c").with_priority(JobPriority::CRITICAL)).unwrap();
        assert_eq!(q.pick_next().unwrap().unwrap(), "crit");
        assert_eq!(q.pick_next().unwrap().unwrap(), "high");
    }

    #[test]
    fn test_concurrency_limit() {
        let mut q = JobQueue::new().with_concurrency(1);
        q.submit(Job::new("j1", "task", "a")).unwrap();
        q.submit(Job::new("j2", "task", "b")).unwrap();
        q.pick_next().unwrap(); // j1 running
        assert!(matches!(
            q.pick_next(),
            Err(JobError::ConcurrencyLimitReached { .. })
        ));
    }

    #[test]
    fn test_complete_job() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "a")).unwrap();
        q.pick_next().unwrap();
        q.complete("j1").unwrap();
        assert_eq!(q.get_job("j1").unwrap().status, JobStatus::Completed);
        assert_eq!(q.running_count(), 0);
    }

    #[test]
    fn test_fail_with_retry() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "a").with_max_retries(3)).unwrap();
        q.pick_next().unwrap();
        let retried = q.fail("j1", "transient error").unwrap();
        assert!(retried);
        assert_eq!(q.get_job("j1").unwrap().status, JobStatus::Retrying);
    }

    #[test]
    fn test_fail_exhausts_retries() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "a").with_max_retries(1)).unwrap();
        q.pick_next().unwrap(); // attempt 1
        q.fail("j1", "error").unwrap(); // retrying
        q.advance_time(10000);
        q.pick_next().unwrap(); // attempt 2 (attempt == max_retries now)
        let retried = q.fail("j1", "error again").unwrap();
        assert!(!retried); // permanently failed
        assert_eq!(q.get_job("j1").unwrap().status, JobStatus::Failed);
    }

    #[test]
    fn test_exponential_backoff() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            multiplier: 2.0,
        };
        assert_eq!(policy.delay_for_attempt(0), 100);
        assert_eq!(policy.delay_for_attempt(1), 200);
        assert_eq!(policy.delay_for_attempt(2), 400);
        assert_eq!(policy.delay_for_attempt(3), 800);
        // Capped at max.
        assert_eq!(policy.delay_for_attempt(10), 10_000);
    }

    #[test]
    fn test_delayed_job() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "data").with_delay(500)).unwrap();
        // Not eligible yet.
        assert!(q.pick_next().unwrap().is_none());
        q.advance_time(500);
        let id = q.pick_next().unwrap().unwrap();
        assert_eq!(id, "j1");
    }

    #[test]
    fn test_cancel_pending() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "a")).unwrap();
        q.cancel("j1").unwrap();
        assert_eq!(q.get_job("j1").unwrap().status, JobStatus::Cancelled);
    }

    #[test]
    fn test_cancel_running_fails() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "a")).unwrap();
        q.pick_next().unwrap();
        assert!(matches!(
            q.cancel("j1"),
            Err(JobError::NotCancellable { .. })
        ));
    }

    #[test]
    fn test_duplicate_job() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "task", "a")).unwrap();
        assert!(matches!(
            q.submit(Job::new("j1", "task", "b")),
            Err(JobError::DuplicateJob(_))
        ));
    }

    #[test]
    fn test_queue_full() {
        let mut q = JobQueue::new().with_max_queue_size(2);
        q.submit(Job::new("j1", "t", "a")).unwrap();
        q.submit(Job::new("j2", "t", "b")).unwrap();
        assert!(matches!(
            q.submit(Job::new("j3", "t", "c")),
            Err(JobError::QueueFull { .. })
        ));
    }

    #[test]
    fn test_jobs_by_status() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "t", "a")).unwrap();
        q.submit(Job::new("j2", "t", "b")).unwrap();
        q.pick_next().unwrap();
        let pending = q.jobs_by_status(JobStatus::Pending);
        assert_eq!(pending.len(), 1);
        let running = q.jobs_by_status(JobStatus::Running);
        assert_eq!(running.len(), 1);
    }

    #[test]
    fn test_jobs_by_tag() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "t", "a").with_tag("email")).unwrap();
        q.submit(Job::new("j2", "t", "b").with_tag("sms")).unwrap();
        q.submit(Job::new("j3", "t", "c").with_tag("email")).unwrap();
        let email_jobs = q.jobs_by_tag("email");
        assert_eq!(email_jobs.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut q = make_queue();
        q.submit(Job::new("j1", "t", "a")).unwrap();
        q.submit(Job::new("j2", "t", "b")).unwrap();
        assert_eq!(q.stats().total_submitted, 2);
        q.pick_next().unwrap();
        q.complete("j1").unwrap();
        assert_eq!(q.stats().total_completed, 1);
    }

    #[test]
    fn test_job_not_found() {
        let mut q = make_queue();
        assert!(matches!(
            q.complete("nope"),
            Err(JobError::JobNotFound(_))
        ));
        assert!(matches!(
            q.fail("nope", "err"),
            Err(JobError::JobNotFound(_))
        ));
        assert!(matches!(
            q.cancel("nope"),
            Err(JobError::JobNotFound(_))
        ));
    }

    #[test]
    fn test_retry_backoff_delay() {
        let mut q = make_queue();
        let policy = RetryPolicy {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            multiplier: 2.0,
        };
        q.submit(Job::new("j1", "t", "a").with_retry(policy)).unwrap();
        q.pick_next().unwrap();
        q.fail("j1", "err").unwrap();
        // Retry at clock + 200 (attempt 1, delay = 100 * 2^1 = 200).
        let job = q.get_job("j1").unwrap();
        assert_eq!(job.next_retry_at_ms, Some(200));
    }
}
