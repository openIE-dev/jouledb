//! Worker pool — logical worker pool with task dispatch, health tracking, and scaling.
//!
//! Replaces Node.js `workerpool`, `piscina`, `threads.js` with a pure-Rust
//! worker pool abstraction. Supports round-robin and least-loaded dispatch,
//! worker health monitoring, pool scaling, task timeouts, and pool statistics.

use std::collections::{HashMap, VecDeque};

// ── Errors ──────────────────────────────────────────────────────

/// Worker pool domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerPoolError {
    /// Worker not found.
    WorkerNotFound(u64),
    /// Task not found.
    TaskNotFound(u64),
    /// Pool is shut down.
    PoolShutDown,
    /// No healthy workers available.
    NoHealthyWorkers,
    /// Worker already exists.
    DuplicateWorker(u64),
    /// Pool capacity reached.
    CapacityReached { max: usize },
    /// Task timed out.
    TaskTimeout { task_id: u64, timeout_ms: u64 },
}

impl std::fmt::Display for WorkerPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkerNotFound(id) => write!(f, "worker not found: {id}"),
            Self::TaskNotFound(id) => write!(f, "task not found: {id}"),
            Self::PoolShutDown => write!(f, "pool is shut down"),
            Self::NoHealthyWorkers => write!(f, "no healthy workers available"),
            Self::DuplicateWorker(id) => write!(f, "worker already exists: {id}"),
            Self::CapacityReached { max } => write!(f, "pool capacity reached: {max}"),
            Self::TaskTimeout { task_id, timeout_ms } => {
                write!(f, "task {task_id} timed out after {timeout_ms}ms")
            }
        }
    }
}

impl std::error::Error for WorkerPoolError {}

// ── Dispatch Strategy ───────────────────────────────────────────

/// How tasks are dispatched to workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchStrategy {
    /// Round-robin across healthy workers.
    RoundRobin,
    /// Send to the worker with the fewest active tasks.
    LeastLoaded,
}

impl Default for DispatchStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

// ── Worker Health ───────────────────────────────────────────────

/// Health status of a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    Healthy,
    Degraded,
    Unhealthy,
}

// ── Worker State ────────────────────────────────────────────────

/// Whether a worker is idle or busy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    Idle,
    Busy,
}

// ── Task State ──────────────────────────────────────────────────

/// Lifecycle state of a pool task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolTaskState {
    Queued,
    Running,
    Completed,
    Failed,
    TimedOut,
}

// ── Pool Task ───────────────────────────────────────────────────

/// A task submitted to the worker pool.
#[derive(Debug, Clone)]
pub struct PoolTask {
    pub id: u64,
    pub name: String,
    pub state: PoolTaskState,
    pub assigned_worker: Option<u64>,
    pub timeout_ms: Option<u64>,
    pub elapsed_ms: u64,
    pub result: Option<String>,
    pub error: Option<String>,
    pub submitted_at_tick: u64,
}

impl PoolTask {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            state: PoolTaskState::Queued,
            assigned_worker: None,
            timeout_ms: None,
            elapsed_ms: 0,
            result: None,
            error: None,
            submitted_at_tick: 0,
        }
    }

    /// Set a timeout for this task.
    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }
}

// ── Worker ──────────────────────────────────────────────────────

/// A logical worker in the pool.
#[derive(Debug, Clone)]
pub struct Worker {
    pub id: u64,
    pub name: String,
    pub state: WorkerState,
    pub health: WorkerHealth,
    pub active_tasks: Vec<u64>,
    pub completed_count: u64,
    pub failed_count: u64,
    pub total_time_ms: u64,
    pub max_concurrent: usize,
}

impl Worker {
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            state: WorkerState::Idle,
            health: WorkerHealth::Healthy,
            active_tasks: Vec::new(),
            completed_count: 0,
            failed_count: 0,
            total_time_ms: 0,
            max_concurrent: 4,
        }
    }

    /// Set max concurrent tasks this worker can handle.
    pub fn with_max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n;
        self
    }

    /// Whether the worker can accept more tasks.
    pub fn can_accept(&self) -> bool {
        self.health != WorkerHealth::Unhealthy && self.active_tasks.len() < self.max_concurrent
    }

    /// Current load as number of active tasks.
    pub fn load(&self) -> usize {
        self.active_tasks.len()
    }
}

// ── Pool Statistics ─────────────────────────────────────────────

/// Aggregate statistics for the pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_submitted: u64,
    pub total_completed: u64,
    pub total_failed: u64,
    pub total_timed_out: u64,
    pub total_workers: usize,
    pub healthy_workers: usize,
    pub queued_tasks: usize,
    pub running_tasks: usize,
}

// ── Worker Pool ─────────────────────────────────────────────────

/// Pool of logical workers with task dispatch and management.
pub struct WorkerPool {
    workers: HashMap<u64, Worker>,
    tasks: HashMap<u64, PoolTask>,
    queue: VecDeque<u64>,
    strategy: DispatchStrategy,
    round_robin_index: usize,
    /// Ordered list of worker IDs for round-robin.
    worker_order: Vec<u64>,
    next_task_id: u64,
    max_pool_size: Option<usize>,
    shut_down: bool,
    current_tick: u64,
}

impl WorkerPool {
    /// Create a new empty worker pool with the given dispatch strategy.
    pub fn new(strategy: DispatchStrategy) -> Self {
        Self {
            workers: HashMap::new(),
            tasks: HashMap::new(),
            queue: VecDeque::new(),
            strategy,
            round_robin_index: 0,
            worker_order: Vec::new(),
            next_task_id: 1,
            max_pool_size: None,
            shut_down: false,
            current_tick: 0,
        }
    }

    /// Set maximum pool size.
    pub fn with_max_pool_size(mut self, max: usize) -> Self {
        self.max_pool_size = Some(max);
        self
    }

    /// Advance the simulated clock.
    pub fn tick(&mut self, ms: u64) {
        self.current_tick += ms;
    }

    /// Add a worker to the pool.
    pub fn add_worker(&mut self, worker: Worker) -> Result<(), WorkerPoolError> {
        if self.shut_down {
            return Err(WorkerPoolError::PoolShutDown);
        }
        if self.workers.contains_key(&worker.id) {
            return Err(WorkerPoolError::DuplicateWorker(worker.id));
        }
        if let Some(max) = self.max_pool_size {
            if self.workers.len() >= max {
                return Err(WorkerPoolError::CapacityReached { max });
            }
        }
        let id = worker.id;
        self.workers.insert(id, worker);
        self.worker_order.push(id);
        Ok(())
    }

    /// Remove a worker from the pool.
    pub fn remove_worker(&mut self, worker_id: u64) -> Result<Worker, WorkerPoolError> {
        let worker = self
            .workers
            .remove(&worker_id)
            .ok_or(WorkerPoolError::WorkerNotFound(worker_id))?;
        self.worker_order.retain(|id| *id != worker_id);
        // Requeue active tasks from the removed worker
        for task_id in &worker.active_tasks {
            if let Some(task) = self.tasks.get_mut(task_id) {
                task.state = PoolTaskState::Queued;
                task.assigned_worker = None;
                self.queue.push_back(*task_id);
            }
        }
        // Adjust round-robin index
        if !self.worker_order.is_empty() {
            self.round_robin_index %= self.worker_order.len();
        } else {
            self.round_robin_index = 0;
        }
        Ok(worker)
    }

    /// Submit a task to the pool. Returns the assigned task ID.
    pub fn submit(&mut self, mut task: PoolTask) -> Result<u64, WorkerPoolError> {
        if self.shut_down {
            return Err(WorkerPoolError::PoolShutDown);
        }
        let id = self.next_task_id;
        self.next_task_id += 1;
        task.id = id;
        task.submitted_at_tick = self.current_tick;
        self.tasks.insert(id, task);
        self.queue.push_back(id);
        Ok(id)
    }

    /// Dispatch queued tasks to available workers.
    pub fn dispatch(&mut self) -> Vec<(u64, u64)> {
        let mut assignments = Vec::new();
        let mut undispatchable = VecDeque::new();

        while let Some(task_id) = self.queue.pop_front() {
            match self.select_worker() {
                Some(worker_id) => {
                    // Apply assignment immediately so subsequent select_worker
                    // calls see the updated load/capacity.
                    if let Some(task) = self.tasks.get_mut(&task_id) {
                        task.state = PoolTaskState::Running;
                        task.assigned_worker = Some(worker_id);
                    }
                    if let Some(worker) = self.workers.get_mut(&worker_id) {
                        worker.active_tasks.push(task_id);
                        worker.state = WorkerState::Busy;
                    }
                    assignments.push((task_id, worker_id));
                }
                None => {
                    undispatchable.push_back(task_id);
                }
            }
        }

        // Put back undispatchable tasks
        self.queue = undispatchable;

        assignments
    }

    /// Select a worker using the configured strategy.
    fn select_worker(&mut self) -> Option<u64> {
        match self.strategy {
            DispatchStrategy::RoundRobin => self.select_round_robin(),
            DispatchStrategy::LeastLoaded => self.select_least_loaded(),
        }
    }

    fn select_round_robin(&mut self) -> Option<u64> {
        let len = self.worker_order.len();
        if len == 0 {
            return None;
        }
        // Try all workers starting from current index
        for i in 0..len {
            let idx = (self.round_robin_index + i) % len;
            let wid = self.worker_order[idx];
            if let Some(w) = self.workers.get(&wid) {
                if w.can_accept() {
                    self.round_robin_index = (idx + 1) % len;
                    return Some(wid);
                }
            }
        }
        None
    }

    fn select_least_loaded(&self) -> Option<u64> {
        self.workers
            .values()
            .filter(|w| w.can_accept())
            .min_by_key(|w| w.load())
            .map(|w| w.id)
    }

    /// Mark a task as completed.
    pub fn complete_task(
        &mut self,
        task_id: u64,
        result: impl Into<String>,
        duration_ms: u64,
    ) -> Result<(), WorkerPoolError> {
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or(WorkerPoolError::TaskNotFound(task_id))?;
        task.state = PoolTaskState::Completed;
        task.result = Some(result.into());
        task.elapsed_ms = duration_ms;
        let worker_id = task.assigned_worker;

        if let Some(wid) = worker_id {
            if let Some(worker) = self.workers.get_mut(&wid) {
                worker.active_tasks.retain(|id| *id != task_id);
                worker.completed_count += 1;
                worker.total_time_ms += duration_ms;
                if worker.active_tasks.is_empty() {
                    worker.state = WorkerState::Idle;
                }
            }
        }
        Ok(())
    }

    /// Mark a task as failed.
    pub fn fail_task(
        &mut self,
        task_id: u64,
        error: impl Into<String>,
        duration_ms: u64,
    ) -> Result<(), WorkerPoolError> {
        let task = self
            .tasks
            .get_mut(&task_id)
            .ok_or(WorkerPoolError::TaskNotFound(task_id))?;
        task.state = PoolTaskState::Failed;
        task.error = Some(error.into());
        task.elapsed_ms = duration_ms;
        let worker_id = task.assigned_worker;

        if let Some(wid) = worker_id {
            if let Some(worker) = self.workers.get_mut(&wid) {
                worker.active_tasks.retain(|id| *id != task_id);
                worker.failed_count += 1;
                worker.total_time_ms += duration_ms;
                if worker.active_tasks.is_empty() {
                    worker.state = WorkerState::Idle;
                }
            }
        }
        Ok(())
    }

    /// Check and time out tasks that exceeded their timeout.
    pub fn check_timeouts(&mut self) -> Vec<u64> {
        let mut timed_out = Vec::new();
        let tick = self.current_tick;
        for task in self.tasks.values_mut() {
            if task.state == PoolTaskState::Running {
                if let Some(timeout) = task.timeout_ms {
                    let elapsed = tick.saturating_sub(task.submitted_at_tick);
                    if elapsed >= timeout {
                        task.state = PoolTaskState::TimedOut;
                        task.elapsed_ms = elapsed;
                        timed_out.push(task.id);
                    }
                }
            }
        }
        // Clean up worker state for timed out tasks
        for task_id in &timed_out {
            if let Some(task) = self.tasks.get(task_id) {
                let worker_id = task.assigned_worker;
                if let Some(wid) = worker_id {
                    if let Some(worker) = self.workers.get_mut(&wid) {
                        worker.active_tasks.retain(|id| id != task_id);
                        if worker.active_tasks.is_empty() {
                            worker.state = WorkerState::Idle;
                        }
                    }
                }
            }
        }
        timed_out
    }

    /// Update a worker's health status.
    pub fn set_worker_health(
        &mut self,
        worker_id: u64,
        health: WorkerHealth,
    ) -> Result<(), WorkerPoolError> {
        let worker = self
            .workers
            .get_mut(&worker_id)
            .ok_or(WorkerPoolError::WorkerNotFound(worker_id))?;
        worker.health = health;
        Ok(())
    }

    /// Shut down the pool — no new submissions accepted.
    pub fn shut_down(&mut self) {
        self.shut_down = true;
    }

    /// Whether the pool is shut down.
    pub fn is_shut_down(&self) -> bool {
        self.shut_down
    }

    /// Get a reference to a task.
    pub fn get_task(&self, task_id: u64) -> Option<&PoolTask> {
        self.tasks.get(&task_id)
    }

    /// Get a reference to a worker.
    pub fn get_worker(&self, worker_id: u64) -> Option<&Worker> {
        self.workers.get(&worker_id)
    }

    /// Number of workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Number of healthy workers.
    pub fn healthy_worker_count(&self) -> usize {
        self.workers
            .values()
            .filter(|w| w.health != WorkerHealth::Unhealthy)
            .count()
    }

    /// Number of queued tasks.
    pub fn queued_count(&self) -> usize {
        self.queue.len()
    }

    /// Collect pool statistics.
    pub fn stats(&self) -> PoolStats {
        let total_completed = self
            .tasks
            .values()
            .filter(|t| t.state == PoolTaskState::Completed)
            .count() as u64;
        let total_failed = self
            .tasks
            .values()
            .filter(|t| t.state == PoolTaskState::Failed)
            .count() as u64;
        let total_timed_out = self
            .tasks
            .values()
            .filter(|t| t.state == PoolTaskState::TimedOut)
            .count() as u64;
        let running_tasks = self
            .tasks
            .values()
            .filter(|t| t.state == PoolTaskState::Running)
            .count();

        PoolStats {
            total_submitted: self.tasks.len() as u64,
            total_completed,
            total_failed,
            total_timed_out,
            total_workers: self.workers.len(),
            healthy_workers: self.healthy_worker_count(),
            queued_tasks: self.queue.len(),
            running_tasks,
        }
    }

    /// Get list of all worker IDs.
    pub fn worker_ids(&self) -> Vec<u64> {
        self.worker_order.clone()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(strategy: DispatchStrategy) -> WorkerPool {
        let mut pool = WorkerPool::new(strategy);
        pool.add_worker(Worker::new(1, "w1")).unwrap();
        pool.add_worker(Worker::new(2, "w2")).unwrap();
        pool.add_worker(Worker::new(3, "w3")).unwrap();
        pool
    }

    #[test]
    fn test_add_and_count_workers() {
        let pool = make_pool(DispatchStrategy::RoundRobin);
        assert_eq!(pool.worker_count(), 3);
        assert_eq!(pool.healthy_worker_count(), 3);
    }

    #[test]
    fn test_duplicate_worker() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let err = pool.add_worker(Worker::new(1, "dup")).unwrap_err();
        assert_eq!(err, WorkerPoolError::DuplicateWorker(1));
    }

    #[test]
    fn test_remove_worker() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let removed = pool.remove_worker(2).unwrap();
        assert_eq!(removed.id, 2);
        assert_eq!(pool.worker_count(), 2);
    }

    #[test]
    fn test_remove_nonexistent_worker() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let err = pool.remove_worker(99).unwrap_err();
        assert_eq!(err, WorkerPoolError::WorkerNotFound(99));
    }

    #[test]
    fn test_submit_and_dispatch_round_robin() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let t1 = pool.submit(PoolTask::new(0, "task1")).unwrap();
        let t2 = pool.submit(PoolTask::new(0, "task2")).unwrap();
        let t3 = pool.submit(PoolTask::new(0, "task3")).unwrap();

        let assignments = pool.dispatch();
        assert_eq!(assignments.len(), 3);

        // Round-robin should cycle through workers
        let worker_ids: Vec<u64> = assignments.iter().map(|(_, wid)| *wid).collect();
        assert_eq!(worker_ids[0], 1);
        assert_eq!(worker_ids[1], 2);
        assert_eq!(worker_ids[2], 3);

        // Tasks should be running
        assert_eq!(pool.get_task(t1).unwrap().state, PoolTaskState::Running);
        assert_eq!(pool.get_task(t2).unwrap().state, PoolTaskState::Running);
        assert_eq!(pool.get_task(t3).unwrap().state, PoolTaskState::Running);
    }

    #[test]
    fn test_submit_and_dispatch_least_loaded() {
        let mut pool = WorkerPool::new(DispatchStrategy::LeastLoaded);
        pool.add_worker(Worker::new(1, "w1").with_max_concurrent(2))
            .unwrap();
        pool.add_worker(Worker::new(2, "w2").with_max_concurrent(2))
            .unwrap();

        // Submit 3 tasks
        pool.submit(PoolTask::new(0, "task1")).unwrap();
        pool.submit(PoolTask::new(0, "task2")).unwrap();
        pool.submit(PoolTask::new(0, "task3")).unwrap();

        let assignments = pool.dispatch();
        assert_eq!(assignments.len(), 3);

        // After dispatch, verify load distribution
        let w1 = pool.get_worker(1).unwrap();
        let w2 = pool.get_worker(2).unwrap();
        let total = w1.load() + w2.load();
        assert_eq!(total, 3);
        // One worker has 2, the other has 1 (least-loaded balancing)
        assert!(w1.load() <= 2 && w2.load() <= 2);
    }

    #[test]
    fn test_complete_task() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let tid = pool.submit(PoolTask::new(0, "task1")).unwrap();
        pool.dispatch();
        pool.complete_task(tid, "done", 100).unwrap();

        let task = pool.get_task(tid).unwrap();
        assert_eq!(task.state, PoolTaskState::Completed);
        assert_eq!(task.result.as_deref(), Some("done"));
        assert_eq!(task.elapsed_ms, 100);
    }

    #[test]
    fn test_fail_task() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let tid = pool.submit(PoolTask::new(0, "task1")).unwrap();
        pool.dispatch();
        pool.fail_task(tid, "crash", 50).unwrap();

        let task = pool.get_task(tid).unwrap();
        assert_eq!(task.state, PoolTaskState::Failed);
        assert_eq!(task.error.as_deref(), Some("crash"));
    }

    #[test]
    fn test_task_timeout() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let tid = pool
            .submit(PoolTask::new(0, "slow").with_timeout(500))
            .unwrap();
        pool.dispatch();
        pool.tick(600);
        let timed_out = pool.check_timeouts();
        assert_eq!(timed_out, vec![tid]);
        assert_eq!(pool.get_task(tid).unwrap().state, PoolTaskState::TimedOut);
    }

    #[test]
    fn test_task_not_timed_out() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let tid = pool
            .submit(PoolTask::new(0, "fast").with_timeout(500))
            .unwrap();
        pool.dispatch();
        pool.tick(200);
        let timed_out = pool.check_timeouts();
        assert!(timed_out.is_empty());
        assert_eq!(pool.get_task(tid).unwrap().state, PoolTaskState::Running);
    }

    #[test]
    fn test_worker_health_tracking() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        pool.set_worker_health(2, WorkerHealth::Unhealthy).unwrap();
        assert_eq!(pool.healthy_worker_count(), 2);
    }

    #[test]
    fn test_unhealthy_worker_skipped() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        pool.set_worker_health(1, WorkerHealth::Unhealthy).unwrap();

        pool.submit(PoolTask::new(0, "task1")).unwrap();
        let assignments = pool.dispatch();
        assert_eq!(assignments.len(), 1);
        // Worker 1 is unhealthy, so task goes to worker 2
        assert_eq!(assignments[0].1, 2);
    }

    #[test]
    fn test_pool_shutdown_rejects_submissions() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        pool.shut_down();
        let err = pool.submit(PoolTask::new(0, "nope")).unwrap_err();
        assert_eq!(err, WorkerPoolError::PoolShutDown);
        assert!(pool.is_shut_down());
    }

    #[test]
    fn test_max_pool_size() {
        let mut pool = WorkerPool::new(DispatchStrategy::RoundRobin).with_max_pool_size(2);
        pool.add_worker(Worker::new(1, "w1")).unwrap();
        pool.add_worker(Worker::new(2, "w2")).unwrap();
        let err = pool.add_worker(Worker::new(3, "w3")).unwrap_err();
        assert_eq!(err, WorkerPoolError::CapacityReached { max: 2 });
    }

    #[test]
    fn test_pool_stats() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let t1 = pool.submit(PoolTask::new(0, "a")).unwrap();
        let t2 = pool.submit(PoolTask::new(0, "b")).unwrap();
        pool.dispatch();
        pool.complete_task(t1, "ok", 10).unwrap();
        pool.fail_task(t2, "err", 5).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.total_submitted, 2);
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.total_failed, 1);
        assert_eq!(stats.total_workers, 3);
    }

    #[test]
    fn test_worker_becomes_idle_after_completion() {
        let mut pool = WorkerPool::new(DispatchStrategy::RoundRobin);
        pool.add_worker(Worker::new(1, "w1").with_max_concurrent(1))
            .unwrap();
        let tid = pool.submit(PoolTask::new(0, "task1")).unwrap();
        pool.dispatch();
        assert_eq!(pool.get_worker(1).unwrap().state, WorkerState::Busy);
        pool.complete_task(tid, "ok", 10).unwrap();
        assert_eq!(pool.get_worker(1).unwrap().state, WorkerState::Idle);
    }

    #[test]
    fn test_remove_worker_requeues_tasks() {
        let mut pool = WorkerPool::new(DispatchStrategy::RoundRobin);
        pool.add_worker(Worker::new(1, "w1")).unwrap();
        pool.add_worker(Worker::new(2, "w2")).unwrap();

        let tid = pool.submit(PoolTask::new(0, "task1")).unwrap();
        pool.dispatch();
        // Task is running on worker 1
        let assigned = pool.get_task(tid).unwrap().assigned_worker.unwrap();
        pool.remove_worker(assigned).unwrap();

        // Task should be requeued
        assert_eq!(pool.get_task(tid).unwrap().state, PoolTaskState::Queued);
        assert_eq!(pool.queued_count(), 1);
    }

    #[test]
    fn test_no_healthy_workers_leaves_tasks_queued() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        pool.set_worker_health(1, WorkerHealth::Unhealthy).unwrap();
        pool.set_worker_health(2, WorkerHealth::Unhealthy).unwrap();
        pool.set_worker_health(3, WorkerHealth::Unhealthy).unwrap();

        pool.submit(PoolTask::new(0, "task1")).unwrap();
        let assignments = pool.dispatch();
        assert!(assignments.is_empty());
        assert_eq!(pool.queued_count(), 1);
    }

    #[test]
    fn test_degraded_worker_accepts_tasks() {
        let mut pool = WorkerPool::new(DispatchStrategy::RoundRobin);
        pool.add_worker(Worker::new(1, "w1")).unwrap();
        pool.set_worker_health(1, WorkerHealth::Degraded).unwrap();

        pool.submit(PoolTask::new(0, "task1")).unwrap();
        let assignments = pool.dispatch();
        assert_eq!(assignments.len(), 1);
    }

    #[test]
    fn test_round_robin_wraps_around() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        // Submit 6 tasks -> should wrap around the 3 workers twice
        for i in 0..6 {
            pool.submit(PoolTask::new(0, format!("t{i}"))).unwrap();
        }
        let assignments = pool.dispatch();
        assert_eq!(assignments.len(), 6);
        // First cycle: 1,2,3; second cycle: 1,2,3
        let wids: Vec<u64> = assignments.iter().map(|(_, w)| *w).collect();
        assert_eq!(wids, vec![1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_worker_stats_accumulate() {
        let mut pool = WorkerPool::new(DispatchStrategy::RoundRobin);
        pool.add_worker(Worker::new(1, "w1")).unwrap();

        let t1 = pool.submit(PoolTask::new(0, "a")).unwrap();
        pool.dispatch();
        pool.complete_task(t1, "ok", 50).unwrap();

        let t2 = pool.submit(PoolTask::new(0, "b")).unwrap();
        pool.dispatch();
        pool.complete_task(t2, "ok", 30).unwrap();

        let worker = pool.get_worker(1).unwrap();
        assert_eq!(worker.completed_count, 2);
        assert_eq!(worker.total_time_ms, 80);
    }

    #[test]
    fn test_full_worker_not_selected() {
        let mut pool = WorkerPool::new(DispatchStrategy::RoundRobin);
        pool.add_worker(Worker::new(1, "w1").with_max_concurrent(1))
            .unwrap();
        pool.add_worker(Worker::new(2, "w2").with_max_concurrent(1))
            .unwrap();

        pool.submit(PoolTask::new(0, "t1")).unwrap();
        pool.submit(PoolTask::new(0, "t2")).unwrap();
        pool.submit(PoolTask::new(0, "t3")).unwrap();
        let assignments = pool.dispatch();
        // Only 2 workers with capacity 1 each
        assert_eq!(assignments.len(), 2);
        assert_eq!(pool.queued_count(), 1);
    }

    #[test]
    fn test_complete_unknown_task() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        let err = pool.complete_task(999, "ok", 10).unwrap_err();
        assert_eq!(err, WorkerPoolError::TaskNotFound(999));
    }

    #[test]
    fn test_shutdown_rejects_workers() {
        let mut pool = make_pool(DispatchStrategy::RoundRobin);
        pool.shut_down();
        let err = pool.add_worker(Worker::new(4, "w4")).unwrap_err();
        assert_eq!(err, WorkerPoolError::PoolShutDown);
    }
}
