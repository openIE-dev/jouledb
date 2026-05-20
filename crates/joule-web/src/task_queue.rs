//! Task queue — FIFO queue with priority support, retry, dependencies, timeouts.
//!
//! Replaces Bull/BeeQueue/Celery with a pure Rust task queue.
//! Models task lifecycle: pending → running → completed/failed, with retry
//! and exponential backoff, max concurrency, task dependencies, and pause/resume.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

// ── Task State ─────────────────────────────────────────────────

/// Lifecycle state of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskState {
    Pending,
    Running,
    Completed,
    Failed,
}

// ── Task Priority ──────────────────────────────────────────────

/// Priority level (lower numeric value = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u32);

impl Priority {
    pub const CRITICAL: Priority = Priority(0);
    pub const HIGH: Priority = Priority(10);
    pub const NORMAL: Priority = Priority(50);
    pub const LOW: Priority = Priority(100);
}

impl Default for Priority {
    fn default() -> Self {
        Priority::NORMAL
    }
}

// ── Retry Policy ───────────────────────────────────────────────

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
            base_delay_ms: 100,
            max_delay_ms: 30_000,
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

// ── Task ───────────────────────────────────────────────────────

/// A task in the queue.
#[derive(Debug, Clone)]
pub struct Task {
    pub id: u64,
    pub name: String,
    pub priority: Priority,
    pub state: TaskState,
    pub attempt: u32,
    pub retry_policy: RetryPolicy,
    pub timeout_ms: Option<u64>,
    pub dependencies: HashSet<u64>,
    pub result: Option<String>,
    pub error: Option<String>,
    created_at_tick: u64,
}

impl Task {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            priority: Priority::default(),
            state: TaskState::Pending,
            attempt: 0,
            retry_policy: RetryPolicy::default(),
            timeout_ms: None,
            dependencies: HashSet::new(),
            result: None,
            error: None,
            created_at_tick: 0,
        }
    }

    pub fn with_priority(mut self, p: Priority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    pub fn depends_on(mut self, dep_id: u64) -> Self {
        self.dependencies.insert(dep_id);
        self
    }

    /// Whether all dependencies are satisfied (completed).
    fn deps_satisfied(&self, completed: &HashSet<u64>) -> bool {
        self.dependencies.is_subset(completed)
    }
}

// ── Queue State ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueState {
    Running,
    Paused,
    Draining,
}

// ── TaskQueue ──────────────────────────────────────────────────

/// Priority-aware FIFO task queue with concurrency limits.
#[derive(Debug)]
pub struct TaskQueue {
    /// All tasks by id.
    tasks: HashMap<u64, Task>,
    /// Pending task ids grouped by priority, FIFO within each priority.
    pending: BTreeMap<Priority, VecDeque<u64>>,
    /// Currently running task ids.
    running: HashSet<u64>,
    /// Completed task ids.
    completed: HashSet<u64>,
    /// Failed task ids.
    failed: HashSet<u64>,
    /// Maximum number of concurrently running tasks.
    max_concurrency: usize,
    /// Queue state.
    state: QueueState,
    /// Monotonic tick counter.
    tick: u64,
    /// Next task id.
    next_id: u64,
}

impl TaskQueue {
    pub fn new(max_concurrency: usize) -> Self {
        Self {
            tasks: HashMap::new(),
            pending: BTreeMap::new(),
            running: HashSet::new(),
            completed: HashSet::new(),
            failed: HashSet::new(),
            max_concurrency: max_concurrency.max(1),
            state: QueueState::Running,
            tick: 0,
            next_id: 1,
        }
    }

    /// Generate a unique task id.
    pub fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Enqueue a task. Returns the task id.
    pub fn enqueue(&mut self, mut task: Task) -> u64 {
        task.created_at_tick = self.tick;
        let id = task.id;
        let prio = task.priority;
        self.pending.entry(prio).or_default().push_back(id);
        self.tasks.insert(id, task);
        id
    }

    /// Number of pending tasks.
    pub fn pending_count(&self) -> usize {
        self.pending.values().map(|q| q.len()).sum()
    }

    /// Number of running tasks.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of completed tasks.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Number of failed tasks.
    pub fn failed_count(&self) -> usize {
        self.failed.len()
    }

    pub fn queue_state(&self) -> QueueState {
        self.state
    }

    pub fn pause(&mut self) {
        self.state = QueueState::Paused;
    }

    pub fn resume(&mut self) {
        self.state = QueueState::Running;
    }

    /// Drain: finish running tasks but don't start new ones. When running
    /// count hits zero the queue moves to Paused.
    pub fn drain(&mut self) {
        self.state = QueueState::Draining;
    }

    /// Try to start the next eligible pending task(s). Returns ids of newly started tasks.
    pub fn poll(&mut self) -> Vec<u64> {
        self.tick += 1;
        if self.state == QueueState::Paused {
            return Vec::new();
        }
        if self.state == QueueState::Draining {
            if self.running.is_empty() {
                self.state = QueueState::Paused;
            }
            return Vec::new();
        }

        let mut started = Vec::new();
        // Iterate priorities low-to-high (BTreeMap is ascending, Priority(0) is highest).
        let priorities: Vec<Priority> = self.pending.keys().copied().collect();
        for prio in priorities {
            if self.running.len() >= self.max_concurrency {
                break;
            }
            let queue = match self.pending.get_mut(&prio) {
                Some(q) => q,
                None => continue,
            };
            let mut skipped: VecDeque<u64> = VecDeque::new();
            while self.running.len() < self.max_concurrency {
                let id = match queue.pop_front() {
                    Some(id) => id,
                    None => break,
                };
                let deps_ok = self
                    .tasks
                    .get(&id)
                    .map(|t| t.deps_satisfied(&self.completed))
                    .unwrap_or(false);
                if deps_ok {
                    if let Some(t) = self.tasks.get_mut(&id) {
                        t.state = TaskState::Running;
                        t.attempt += 1;
                    }
                    self.running.insert(id);
                    started.push(id);
                } else {
                    skipped.push_back(id);
                }
            }
            // Put skipped tasks back.
            if let Some(q) = self.pending.get_mut(&prio) {
                for s in skipped.into_iter().rev() {
                    q.push_front(s);
                }
            }
        }
        // Clean empty priority buckets.
        self.pending.retain(|_, q| !q.is_empty());
        started
    }

    /// Mark a task as completed.
    pub fn complete(&mut self, id: u64, result: Option<String>) {
        if self.running.remove(&id) {
            if let Some(t) = self.tasks.get_mut(&id) {
                t.state = TaskState::Completed;
                t.result = result;
            }
            self.completed.insert(id);
        }
    }

    /// Mark a task as failed. If retries remain, re-enqueue it.
    pub fn fail(&mut self, id: u64, error: impl Into<String>) -> bool {
        let error = error.into();
        if !self.running.remove(&id) {
            return false;
        }
        let should_retry = self
            .tasks
            .get(&id)
            .map(|t| t.attempt < t.retry_policy.max_retries)
            .unwrap_or(false);

        if should_retry {
            if let Some(t) = self.tasks.get_mut(&id) {
                t.state = TaskState::Pending;
                t.error = Some(error);
                let prio = t.priority;
                self.pending.entry(prio).or_default().push_back(id);
            }
            true
        } else {
            if let Some(t) = self.tasks.get_mut(&id) {
                t.state = TaskState::Failed;
                t.error = Some(error);
            }
            self.failed.insert(id);
            false
        }
    }

    /// Get a task by id.
    pub fn get(&self, id: u64) -> Option<&Task> {
        self.tasks.get(&id)
    }

    /// Check if a task has timed out given elapsed_ms since it started running.
    pub fn check_timeout(&self, id: u64, elapsed_ms: u64) -> bool {
        self.tasks
            .get(&id)
            .and_then(|t| t.timeout_ms)
            .map(|timeout| elapsed_ms >= timeout)
            .unwrap_or(false)
    }

    /// Get the retry delay for a task's current attempt.
    pub fn retry_delay(&self, id: u64) -> Option<u64> {
        self.tasks.get(&id).map(|t| {
            t.retry_policy
                .delay_for_attempt(t.attempt.saturating_sub(1))
        })
    }

    /// Total number of tasks ever enqueued.
    pub fn total_tasks(&self) -> usize {
        self.tasks.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_and_poll() {
        let mut q = TaskQueue::new(2);
        let id1 = q.next_id();
        let id2 = q.next_id();
        q.enqueue(Task::new(id1, "a"));
        q.enqueue(Task::new(id2, "b"));
        assert_eq!(q.pending_count(), 2);
        let started = q.poll();
        assert_eq!(started.len(), 2);
        assert_eq!(q.running_count(), 2);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn test_concurrency_limit() {
        let mut q = TaskQueue::new(1);
        for i in 0..3 {
            q.enqueue(Task::new(i + 1, format!("t{i}")));
        }
        let started = q.poll();
        assert_eq!(started.len(), 1);
        assert_eq!(q.running_count(), 1);
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_priority_ordering() {
        let mut q = TaskQueue::new(1);
        let lo = q.next_id();
        let hi = q.next_id();
        q.enqueue(Task::new(lo, "low").with_priority(Priority::LOW));
        q.enqueue(Task::new(hi, "high").with_priority(Priority::HIGH));
        let started = q.poll();
        assert_eq!(started, vec![hi]); // Higher priority (lower value) first.
    }

    #[test]
    fn test_complete_and_state() {
        let mut q = TaskQueue::new(2);
        let id = q.next_id();
        q.enqueue(Task::new(id, "x"));
        q.poll();
        q.complete(id, Some("done".into()));
        assert_eq!(q.completed_count(), 1);
        assert_eq!(q.get(id).unwrap().state, TaskState::Completed);
        assert_eq!(q.get(id).unwrap().result.as_deref(), Some("done"));
    }

    #[test]
    fn test_fail_with_retry() {
        let mut q = TaskQueue::new(1);
        let id = q.next_id();
        q.enqueue(Task::new(id, "flaky").with_retry(RetryPolicy {
            max_retries: 2,
            ..Default::default()
        }));
        q.poll(); // attempt 1
        let retried = q.fail(id, "oops");
        assert!(retried);
        assert_eq!(q.pending_count(), 1);
        assert_eq!(q.running_count(), 0);
        // Second attempt.
        q.poll();
        let retried = q.fail(id, "oops again");
        assert!(!retried); // max_retries=2, attempt is now 2
        assert_eq!(q.failed_count(), 1);
    }

    #[test]
    fn test_exponential_backoff() {
        let policy = RetryPolicy {
            max_retries: 5,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            multiplier: 2.0,
        };
        assert_eq!(policy.delay_for_attempt(0), 100);
        assert_eq!(policy.delay_for_attempt(1), 200);
        assert_eq!(policy.delay_for_attempt(2), 400);
        assert_eq!(policy.delay_for_attempt(10), 5000); // capped
    }

    #[test]
    fn test_task_dependencies() {
        let mut q = TaskQueue::new(2);
        let a = q.next_id();
        let b = q.next_id();
        q.enqueue(Task::new(a, "parent"));
        q.enqueue(Task::new(b, "child").depends_on(a));
        let started = q.poll();
        // Only 'a' should start — 'b' depends on 'a'.
        assert_eq!(started, vec![a]);
        assert_eq!(q.running_count(), 1);
        // Complete 'a', then 'b' becomes eligible.
        q.complete(a, None);
        let started = q.poll();
        assert_eq!(started, vec![b]);
    }

    #[test]
    fn test_timeout_check() {
        let mut q = TaskQueue::new(1);
        let id = q.next_id();
        q.enqueue(Task::new(id, "slow").with_timeout(500));
        q.poll();
        assert!(!q.check_timeout(id, 300));
        assert!(q.check_timeout(id, 500));
        assert!(q.check_timeout(id, 1000));
    }

    #[test]
    fn test_pause_resume() {
        let mut q = TaskQueue::new(2);
        let id = q.next_id();
        q.enqueue(Task::new(id, "a"));
        q.pause();
        let started = q.poll();
        assert!(started.is_empty());
        q.resume();
        let started = q.poll();
        assert_eq!(started.len(), 1);
    }

    #[test]
    fn test_drain() {
        let mut q = TaskQueue::new(2);
        let id1 = q.next_id();
        let id2 = q.next_id();
        q.enqueue(Task::new(id1, "a"));
        q.enqueue(Task::new(id2, "b"));
        q.poll(); // starts both
        q.drain();
        // No new tasks start while draining.
        let id3 = q.next_id();
        q.enqueue(Task::new(id3, "c"));
        let started = q.poll();
        assert!(started.is_empty());
        // Complete running tasks.
        q.complete(id1, None);
        q.complete(id2, None);
        q.poll(); // transitions to Paused
        assert_eq!(q.queue_state(), QueueState::Paused);
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut q = TaskQueue::new(10);
        let ids: Vec<u64> = (0..5).map(|_| q.next_id()).collect();
        for id in &ids {
            q.enqueue(Task::new(*id, format!("t{id}")));
        }
        let started = q.poll();
        assert_eq!(started, ids); // FIFO order preserved.
    }

    #[test]
    fn test_retry_delay() {
        let mut q = TaskQueue::new(1);
        let id = q.next_id();
        q.enqueue(Task::new(id, "x").with_retry(RetryPolicy {
            max_retries: 3,
            base_delay_ms: 50,
            max_delay_ms: 1000,
            multiplier: 3.0,
        }));
        q.poll();
        q.fail(id, "err");
        // After first failure, attempt=1, delay_for_attempt(0) = 50.
        assert_eq!(q.retry_delay(id), Some(50));
    }
}
