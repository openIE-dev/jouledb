//! Work-stealing multi-threaded executor for Joule
//!
//! This module provides a high-performance work-stealing executor that
//! distributes tasks across multiple worker threads. Each worker has a
//! local queue and can steal work from other workers when idle.
//!
//! # Design
//!
//! - Each worker thread has a local deque (double-ended queue)
//! - Workers push new tasks to their local queue
//! - Workers pop tasks from the front of their local queue (LIFO for cache locality)
//! - Idle workers steal from the back of other workers' queues (FIFO to reduce contention)
//! - A global injector queue accepts tasks from external spawns

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::future::Future;
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle as ThreadJoinHandle};

use crate::executor::next_task_id;

// ============================================================================
// Work-Stealing Deque
// ============================================================================

/// A lock-free work-stealing deque.
///
/// The owner can push and pop from one end (LIFO),
/// while stealers can steal from the other end (FIFO).
pub struct WorkStealingDeque<T> {
    /// The underlying buffer (power of 2 size).
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Mask for wrapping indices (buffer_size - 1).
    mask: usize,
    /// Bottom index (owner's end).
    bottom: AtomicUsize,
    /// Top index (stealer's end).
    top: AtomicUsize,
}

unsafe impl<T: Send> Send for WorkStealingDeque<T> {}
unsafe impl<T: Send> Sync for WorkStealingDeque<T> {}

impl<T> WorkStealingDeque<T> {
    /// Create a new deque with the given capacity (rounded up to power of 2).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(16);
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(UnsafeCell::new(MaybeUninit::uninit()));
        }

        Self {
            buffer: buffer.into_boxed_slice(),
            mask: capacity - 1,
            bottom: AtomicUsize::new(0),
            top: AtomicUsize::new(0),
        }
    }

    /// Push an item to the bottom (owner only).
    ///
    /// Returns `Err(item)` if the deque is full.
    pub fn push(&self, item: T) -> Result<(), T> {
        let bottom = self.bottom.load(Ordering::Relaxed);
        let top = self.top.load(Ordering::Acquire);

        // Check if full
        if bottom.wrapping_sub(top) >= self.buffer.len() {
            return Err(item);
        }

        // Write the item
        let idx = bottom & self.mask;
        unsafe {
            (*self.buffer[idx].get()).write(item);
        }

        // Make the write visible before updating bottom
        std::sync::atomic::fence(Ordering::Release);
        self.bottom.store(bottom.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Pop an item from the bottom (owner only).
    pub fn pop(&self) -> Option<T> {
        let bottom = self.bottom.load(Ordering::Relaxed);
        if bottom == 0 {
            return None;
        }

        let new_bottom = bottom.wrapping_sub(1);
        self.bottom.store(new_bottom, Ordering::Relaxed);

        // Synchronize with stealers
        std::sync::atomic::fence(Ordering::SeqCst);

        let top = self.top.load(Ordering::Relaxed);

        if top > new_bottom {
            // Deque is empty, restore bottom
            self.bottom.store(bottom, Ordering::Relaxed);
            return None;
        }

        let idx = new_bottom & self.mask;
        let item = unsafe { (*self.buffer[idx].get()).assume_init_read() };

        if top == new_bottom {
            // Last item, race with stealers
            if self
                .top
                .compare_exchange(
                    top,
                    top.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_err()
            {
                // Lost the race, item was stolen
                self.bottom.store(bottom, Ordering::Relaxed);
                return None;
            }
            self.bottom.store(bottom, Ordering::Relaxed);
        }

        Some(item)
    }

    /// Steal an item from the top (any thread).
    pub fn steal(&self) -> Option<T> {
        loop {
            let top = self.top.load(Ordering::Acquire);
            std::sync::atomic::fence(Ordering::SeqCst);
            let bottom = self.bottom.load(Ordering::Acquire);

            if top >= bottom {
                // Deque is empty
                return None;
            }

            let idx = top & self.mask;
            let item = unsafe { (*self.buffer[idx].get()).assume_init_read() };

            if self
                .top
                .compare_exchange(
                    top,
                    top.wrapping_add(1),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return Some(item);
            }
            // CAS failed, retry
        }
    }

    /// Check if the deque is empty.
    pub fn is_empty(&self) -> bool {
        let top = self.top.load(Ordering::Acquire);
        let bottom = self.bottom.load(Ordering::Acquire);
        top >= bottom
    }

    /// Get the approximate length.
    pub fn len(&self) -> usize {
        let top = self.top.load(Ordering::Relaxed);
        let bottom = self.bottom.load(Ordering::Relaxed);
        bottom.saturating_sub(top)
    }
}

impl<T> Drop for WorkStealingDeque<T> {
    fn drop(&mut self) {
        // Drop any remaining items
        while self.pop().is_some() {}
    }
}

// ============================================================================
// Global Injector Queue
// ============================================================================

/// A thread-safe queue for injecting tasks from external threads.
struct InjectorQueue<T> {
    queue: Mutex<VecDeque<T>>,
    condvar: Condvar,
}

impl<T> InjectorQueue<T> {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
        }
    }

    fn push(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(item);
        self.condvar.notify_one();
    }

    fn pop(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }

    #[allow(dead_code)]
    fn steal_batch(&self, max: usize) -> Vec<T> {
        let mut queue = self.queue.lock().unwrap();
        let count = max.min(queue.len());
        queue.drain(..count).collect()
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }
}

// ============================================================================
// Task Representation
// ============================================================================

/// A spawned task for the work-stealing executor.
struct Task {
    /// Unique identifier.
    id: u64,
    /// The future to poll.
    future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

// ============================================================================
// Worker Thread
// ============================================================================

/// State shared between all workers.
struct SharedState {
    /// The global injector queue.
    injector: InjectorQueue<Task>,
    /// All worker queues (for stealing).
    workers: Vec<Arc<WorkStealingDeque<Task>>>,
    /// Shutdown flag.
    shutdown: AtomicBool,
    /// Number of active workers.
    active_workers: AtomicUsize,
    /// Total task count (for knowing when we're done).
    task_count: AtomicUsize,
    /// Wake signal for idle workers.
    wake_signal: Mutex<()>,
    wake_condvar: Condvar,
}

impl SharedState {
    fn wake_all(&self) {
        let _guard = self.wake_signal.lock().unwrap();
        self.wake_condvar.notify_all();
    }

    fn wait_for_work(&self, timeout: std::time::Duration) {
        let guard = self.wake_signal.lock().unwrap();
        let _ = self.wake_condvar.wait_timeout(guard, timeout).unwrap();
    }
}

/// Waker that re-queues a task.
struct TaskWaker {
    #[allow(dead_code)]
    task_id: u64,
    #[allow(dead_code)]
    worker_id: usize,
    state: Arc<SharedState>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.state.wake_all();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.state.wake_all();
    }
}

/// A worker thread in the executor.
struct Worker {
    /// Worker ID.
    id: usize,
    /// Local deque.
    local: Arc<WorkStealingDeque<Task>>,
    /// Shared state.
    state: Arc<SharedState>,
    /// Random state for victim selection.
    rng_state: u64,
}

impl Worker {
    fn new(id: usize, local: Arc<WorkStealingDeque<Task>>, state: Arc<SharedState>) -> Self {
        Self {
            id,
            local,
            state,
            rng_state: id as u64 ^ 0x517cc1b727220a95,
        }
    }

    /// Run the worker loop.
    fn run(&mut self) {
        while !self.state.shutdown.load(Ordering::Acquire) {
            if let Some(task) = self.find_task() {
                self.execute_task(task);
            } else {
                // No work found, wait briefly
                self.state
                    .wait_for_work(std::time::Duration::from_millis(1));
            }
        }

        // Drain remaining local tasks
        while let Some(task) = self.local.pop() {
            self.execute_task(task);
        }
    }

    /// Find a task to execute.
    fn find_task(&mut self) -> Option<Task> {
        // 1. Try local queue first
        if let Some(task) = self.local.pop() {
            return Some(task);
        }

        // 2. Try global injector
        if let Some(task) = self.state.injector.pop() {
            return Some(task);
        }

        // 3. Try stealing from other workers
        self.steal_from_others()
    }

    /// Steal work from other workers.
    fn steal_from_others(&mut self) -> Option<Task> {
        let num_workers = self.state.workers.len();
        if num_workers <= 1 {
            return None;
        }

        // Try stealing from random workers
        for _ in 0..num_workers {
            let victim = self.random_victim();
            if victim != self.id {
                if let Some(task) = self.state.workers[victim].steal() {
                    return Some(task);
                }
            }
        }

        None
    }

    /// Select a random victim for stealing.
    fn random_victim(&mut self) -> usize {
        // Simple xorshift PRNG
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;

        (x as usize) % self.state.workers.len()
    }

    /// Execute a single task.
    fn execute_task(&mut self, mut task: Task) {
        // Check for cancellation
        if task.cancelled.load(Ordering::Acquire) {
            self.state.task_count.fetch_sub(1, Ordering::AcqRel);
            return;
        }

        // Create waker
        let waker = Waker::from(Arc::new(TaskWaker {
            task_id: task.id,
            worker_id: self.id,
            state: self.state.clone(),
        }));
        let mut cx = Context::from_waker(&waker);

        // Poll the task
        match task.future.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                // Task completed
                self.state.task_count.fetch_sub(1, Ordering::AcqRel);
                self.state.wake_all();
            }
            Poll::Pending => {
                // Re-queue the task
                if self.local.push(task).is_err() {
                    // Local queue full, shouldn't happen with reasonable capacity
                    // For now, just drop the task (in production, we'd grow or use overflow)
                    self.state.task_count.fetch_sub(1, Ordering::AcqRel);
                }
            }
        }
    }
}

// ============================================================================
// Work-Stealing Executor
// ============================================================================

/// A multi-threaded work-stealing executor.
///
/// This executor distributes work across multiple worker threads,
/// with idle workers stealing from busy ones to maintain balance.
///
/// # Example
///
/// ```ignore
/// let executor = WorkStealingExecutor::new(4); // 4 worker threads
///
/// executor.spawn(async {
///     println!("Hello from work-stealing executor!");
/// });
///
/// executor.run();
/// ```
pub struct WorkStealingExecutor {
    /// Shared state.
    state: Arc<SharedState>,
    /// Worker thread handles.
    threads: Vec<ThreadJoinHandle<()>>,
    /// Whether the executor has been started.
    started: bool,
}

impl WorkStealingExecutor {
    /// Create a new executor with the specified number of worker threads.
    ///
    /// If `num_threads` is 0, uses the number of available CPU cores.
    pub fn new(num_threads: usize) -> Self {
        let num_threads = if num_threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        } else {
            num_threads
        };

        // Create worker queues
        let workers: Vec<_> = (0..num_threads)
            .map(|_| Arc::new(WorkStealingDeque::new(1024)))
            .collect();

        let state = Arc::new(SharedState {
            injector: InjectorQueue::new(),
            workers: workers.clone(),
            shutdown: AtomicBool::new(false),
            active_workers: AtomicUsize::new(0),
            task_count: AtomicUsize::new(0),
            wake_signal: Mutex::new(()),
            wake_condvar: Condvar::new(),
        });

        Self {
            state,
            threads: Vec::with_capacity(num_threads),
            started: false,
        }
    }

    /// Spawn a future onto the executor.
    ///
    /// Returns a handle that can be used to cancel the task.
    pub fn spawn<F>(&self, future: F) -> WorkStealingTaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task_id = next_task_id();
        let cancelled = Arc::new(AtomicBool::new(false));

        let task = Task {
            id: task_id,
            future: Box::pin(future),
            cancelled: cancelled.clone(),
        };

        self.state.task_count.fetch_add(1, Ordering::AcqRel);
        self.state.injector.push(task);
        self.state.wake_all();

        WorkStealingTaskHandle { task_id, cancelled }
    }

    /// Spawn a future that returns a value.
    pub fn spawn_with_result<F, T>(&self, future: F) -> WorkStealingJoinHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = next_task_id();
        let cancelled = Arc::new(AtomicBool::new(false));
        let result = Arc::new(Mutex::new(None));
        let result_clone = result.clone();
        let state = self.state.clone();

        let wrapped = async move {
            let output = future.await;
            *result_clone.lock().unwrap() = Some(output);
            state.wake_all();
        };

        let task = Task {
            id: task_id,
            future: Box::pin(wrapped),
            cancelled: cancelled.clone(),
        };

        self.state.task_count.fetch_add(1, Ordering::AcqRel);
        self.state.injector.push(task);
        self.state.wake_all();

        WorkStealingJoinHandle {
            task_id,
            cancelled,
            result,
            state: self.state.clone(),
        }
    }

    /// Start the worker threads.
    pub fn start(&mut self) {
        if self.started {
            return;
        }
        self.started = true;

        let num_workers = self.state.workers.len();
        for id in 0..num_workers {
            let local = self.state.workers[id].clone();
            let state = self.state.clone();

            let handle = thread::Builder::new()
                .name(format!("joule-worker-{}", id))
                .spawn(move || {
                    let mut worker = Worker::new(id, local, state.clone());
                    state.active_workers.fetch_add(1, Ordering::AcqRel);
                    worker.run();
                    state.active_workers.fetch_sub(1, Ordering::AcqRel);
                })
                .expect("Failed to spawn worker thread");

            self.threads.push(handle);
        }
    }

    /// Run the executor until all tasks complete.
    pub fn run(&mut self) {
        self.start();

        // Wait for all tasks to complete
        while self.state.task_count.load(Ordering::Acquire) > 0 {
            self.state
                .wait_for_work(std::time::Duration::from_millis(10));
        }

        self.shutdown();
    }

    /// Shutdown the executor.
    pub fn shutdown(&mut self) {
        self.state.shutdown.store(true, Ordering::Release);
        self.state.wake_all();

        // Wait for all workers to finish
        for handle in self.threads.drain(..) {
            let _ = handle.join();
        }

        self.started = false;
    }

    /// Get the number of pending tasks.
    pub fn task_count(&self) -> usize {
        self.state.task_count.load(Ordering::Acquire)
    }

    /// Get the number of worker threads.
    pub fn num_workers(&self) -> usize {
        self.state.workers.len()
    }

    /// Check if shutdown has been requested.
    pub fn is_shutdown(&self) -> bool {
        self.state.shutdown.load(Ordering::Acquire)
    }
}

impl Drop for WorkStealingExecutor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// A handle to a task for cancellation.
#[derive(Clone)]
pub struct WorkStealingTaskHandle {
    pub(crate) task_id: u64,
    pub(crate) cancelled: Arc<AtomicBool>,
}

impl WorkStealingTaskHandle {
    /// Cancel the task.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Get the task ID.
    pub fn task_id(&self) -> u64 {
        self.task_id
    }
}

/// A handle to await a task's result.
pub struct WorkStealingJoinHandle<T> {
    task_id: u64,
    cancelled: Arc<AtomicBool>,
    result: Arc<Mutex<Option<T>>>,
    state: Arc<SharedState>,
}

impl<T> WorkStealingJoinHandle<T> {
    /// Cancel the task.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.state.wake_all();
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Try to get the result without blocking.
    pub fn try_get(&self) -> Option<T> {
        self.result.lock().unwrap().take()
    }

    /// Get the task ID.
    pub fn task_id(&self) -> u64 {
        self.task_id
    }

    /// Block until the result is available.
    pub fn blocking_get(self) -> Option<T> {
        loop {
            if let Some(result) = self.result.lock().unwrap().take() {
                return Some(result);
            }
            if self.is_cancelled() {
                return None;
            }
            self.state
                .wait_for_work(std::time::Duration::from_millis(1));
        }
    }
}

impl<T> Future for WorkStealingJoinHandle<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_cancelled() {
            return Poll::Ready(None);
        }

        if let Some(result) = self.result.lock().unwrap().take() {
            Poll::Ready(Some(result))
        } else {
            Poll::Pending
        }
    }
}

// ============================================================================
// Global Executor
// ============================================================================

/// Global executor state.
static GLOBAL_EXECUTOR: std::sync::OnceLock<WorkStealingExecutor> = std::sync::OnceLock::new();

/// Initialize the global executor.
///
/// This should be called once at program startup. If not called,
/// the global executor will be lazily initialized with default settings.
pub fn init_global_executor(num_threads: usize) {
    let _ = GLOBAL_EXECUTOR.get_or_init(|| {
        let mut executor = WorkStealingExecutor::new(num_threads);
        executor.start();
        executor
    });
}

/// Get the global executor.
fn global_executor() -> &'static WorkStealingExecutor {
    GLOBAL_EXECUTOR.get_or_init(|| {
        let mut executor = WorkStealingExecutor::new(0);
        executor.start();
        executor
    })
}

/// Spawn a task on the global executor.
pub fn spawn_global<F>(future: F) -> WorkStealingTaskHandle
where
    F: Future<Output = ()> + Send + 'static,
{
    global_executor().spawn(future)
}

/// Spawn a task with a result on the global executor.
pub fn spawn_global_with_result<F, T>(future: F) -> WorkStealingJoinHandle<T>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    global_executor().spawn_with_result(future)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_deque_push_pop() {
        let deque = WorkStealingDeque::new(16);

        assert!(deque.is_empty());
        deque.push(1).unwrap();
        deque.push(2).unwrap();
        deque.push(3).unwrap();

        assert_eq!(deque.len(), 3);
        assert_eq!(deque.pop(), Some(3)); // LIFO
        assert_eq!(deque.pop(), Some(2));
        assert_eq!(deque.pop(), Some(1));
        assert!(deque.is_empty());
    }

    #[test]
    fn test_deque_steal() {
        let deque = WorkStealingDeque::new(16);

        deque.push(1).unwrap();
        deque.push(2).unwrap();
        deque.push(3).unwrap();

        assert_eq!(deque.steal(), Some(1)); // FIFO stealing
        assert_eq!(deque.steal(), Some(2));
        assert_eq!(deque.pop(), Some(3));
    }

    #[test]
    fn test_executor_basic() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut executor = WorkStealingExecutor::new(2);

        for _ in 0..10 {
            let counter_clone = counter.clone();
            executor.spawn(async move {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        executor.run();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_executor_with_result() {
        let mut executor = WorkStealingExecutor::new(2);

        let handle = executor.spawn_with_result(async { 42 });

        executor.start();

        // Wait for result
        let result = handle.blocking_get();
        assert_eq!(result, Some(42));

        executor.shutdown();
    }

    #[test]
    fn test_task_cancellation() {
        let started = Arc::new(AtomicBool::new(false));
        let started_clone = started.clone();

        let mut executor = WorkStealingExecutor::new(2);

        let handle = executor.spawn(async move {
            started_clone.store(true, Ordering::SeqCst);
            // This would do more work
        });

        // Cancel immediately
        handle.cancel();
        assert!(handle.is_cancelled());

        executor.run();
    }

    #[test]
    fn test_work_stealing() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut executor = WorkStealingExecutor::new(4);

        // Spawn many tasks to trigger work stealing
        for _ in 0..100 {
            let counter_clone = counter.clone();
            executor.spawn(async move {
                // Simulate some work
                std::thread::yield_now();
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        executor.run();
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_injector_queue() {
        let queue = InjectorQueue::new();

        assert!(queue.is_empty());
        queue.push(1);
        queue.push(2);
        queue.push(3);

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert!(queue.is_empty());
    }

    #[test]
    fn test_injector_batch_steal() {
        let queue = InjectorQueue::new();

        for i in 0..10 {
            queue.push(i);
        }

        let batch = queue.steal_batch(5);
        assert_eq!(batch, vec![0, 1, 2, 3, 4]);
        assert_eq!(queue.len(), 5);
    }

    #[test]
    fn test_deque_capacity() {
        let deque = WorkStealingDeque::new(4);

        // Fill to capacity
        for i in 0..16 {
            // Actually 16 due to next_power_of_two
            assert!(deque.push(i).is_ok());
        }

        // Should be full
        assert!(deque.push(99).is_err());
    }

    #[test]
    fn test_executor_num_workers() {
        let executor = WorkStealingExecutor::new(8);
        assert_eq!(executor.num_workers(), 8);

        let auto_executor = WorkStealingExecutor::new(0);
        assert!(auto_executor.num_workers() >= 1);
    }
}
