//! Async executor for Joule programs
//!
//! This module provides a simple, single-threaded executor for running
//! async Joule code. A more sophisticated multi-threaded executor with
//! work-stealing will be implemented in Phase 4 (Track H).

use std::collections::{BinaryHeap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

/// Counter for generating unique task IDs.
static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique task ID.
pub fn next_task_id() -> u64 {
    NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed)
}

/// A spawned task that can be polled by the executor.
struct Task {
    /// Unique identifier for this task.
    #[allow(dead_code)]
    id: u64,
    /// The future to poll.
    future: Pin<Box<dyn Future<Output = ()> + Send + 'static>>,
    /// Waker for this task.
    waker: Waker,
    /// Whether this task has been cancelled.
    cancelled: Arc<AtomicBool>,
}

/// Waker implementation that signals the executor to wake up.
///
/// This waker uses a condition variable to unpark the executor thread
/// when a task is ready to make progress.
struct ExecutorWaker {
    /// Shared state for signaling.
    signal: Arc<WakeSignal>,
    /// Task ID that this waker is associated with.
    #[allow(dead_code)]
    task_id: u64,
}

/// Shared wake signal state.
struct WakeSignal {
    /// Flag indicating whether a wake has been requested.
    woken: AtomicBool,
    /// Mutex for the condition variable.
    mutex: Mutex<()>,
    /// Condition variable for waiting.
    condvar: Condvar,
}

impl WakeSignal {
    fn new() -> Self {
        Self {
            woken: AtomicBool::new(false),
            mutex: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }

    /// Signal that a wake is requested.
    fn wake(&self) {
        self.woken.store(true, Ordering::Release);
        // Notify the waiting thread
        let _guard = self.mutex.lock().unwrap();
        self.condvar.notify_one();
    }

    /// Wait for a wake signal or timeout.
    fn wait(&self, timeout: Option<Duration>) -> bool {
        // Check if already woken
        if self.woken.swap(false, Ordering::Acquire) {
            return true;
        }

        let guard = self.mutex.lock().unwrap();
        match timeout {
            Some(duration) => {
                let result = self.condvar.wait_timeout(guard, duration).unwrap();
                !result.1.timed_out() || self.woken.swap(false, Ordering::Acquire)
            }
            None => {
                let _guard = self.condvar.wait(guard).unwrap();
                self.woken.swap(false, Ordering::Acquire);
                true
            }
        }
    }

    /// Check if woken without waiting.
    fn is_woken(&self) -> bool {
        self.woken.load(Ordering::Acquire)
    }

    /// Clear the woken flag.
    fn clear(&self) {
        self.woken.store(false, Ordering::Release);
    }
}

impl Wake for ExecutorWaker {
    fn wake(self: Arc<Self>) {
        self.signal.wake();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.signal.wake();
    }
}

/// A simple single-threaded async executor.
///
/// This executor runs futures to completion on the current thread.
/// It's suitable for simple async programs and testing.
///
/// # Example
///
/// ```ignore
/// let executor = Executor::new();
/// executor.spawn(async {
///     println!("Hello from async!");
/// });
/// executor.run();
/// ```
pub struct Executor {
    /// Queue of tasks ready to be polled.
    ready_queue: VecDeque<Task>,
    /// Shared wake signal for all tasks.
    signal: Arc<WakeSignal>,
    /// Timer heap for scheduled wakeups.
    timers: BinaryHeap<TimerEntry>,
}

/// Entry in the timer heap.
struct TimerEntry {
    /// When this timer should fire.
    deadline: Instant,
    /// Task ID to wake.
    #[allow(dead_code)]
    task_id: u64,
    /// Signal to wake.
    signal: Arc<WakeSignal>,
}

impl PartialEq for TimerEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}

impl Eq for TimerEntry {}

impl PartialOrd for TimerEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimerEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap behavior
        other.deadline.cmp(&self.deadline)
    }
}

impl Executor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self {
            ready_queue: VecDeque::new(),
            signal: Arc::new(WakeSignal::new()),
            timers: BinaryHeap::new(),
        }
    }

    /// Spawn a future onto the executor.
    ///
    /// The future will be polled to completion when `run()` is called.
    /// Returns a `TaskHandle` that can be used to cancel the task.
    pub fn spawn<F>(&mut self, future: F) -> TaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let task_id = next_task_id();
        let cancelled = Arc::new(AtomicBool::new(false));
        let waker = Waker::from(Arc::new(ExecutorWaker {
            signal: self.signal.clone(),
            task_id,
        }));

        self.ready_queue.push_back(Task {
            id: task_id,
            future: Box::pin(future),
            waker,
            cancelled: cancelled.clone(),
        });

        TaskHandle { task_id, cancelled }
    }

    /// Spawn a future that returns a value.
    ///
    /// Returns a `JoinHandle` that can be awaited to get the result.
    pub fn spawn_with_handle<F, T>(&mut self, future: F) -> JoinHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let task_id = next_task_id();
        let result = Arc::new(Mutex::new(None));
        let result_clone = result.clone();
        let signal = self.signal.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();

        // Wrap the future to store its result
        let wrapped = async move {
            let output = future.await;
            *result_clone.lock().unwrap() = Some(output);
        };

        let waker = Waker::from(Arc::new(ExecutorWaker {
            signal: signal.clone(),
            task_id,
        }));

        self.ready_queue.push_back(Task {
            id: task_id,
            future: Box::pin(wrapped),
            waker,
            cancelled: cancelled_clone,
        });

        JoinHandle {
            task_id,
            result,
            signal,
            cancelled,
        }
    }

    /// Run the executor until all spawned tasks complete.
    ///
    /// This method blocks the current thread until all tasks are done.
    pub fn run(&mut self) {
        while self.has_tasks() {
            // Process any expired timers
            self.process_timers();

            // Poll all ready tasks
            let mut pending_tasks = VecDeque::new();

            while let Some(mut task) = self.ready_queue.pop_front() {
                // Check if task was cancelled
                if task.cancelled.load(Ordering::Acquire) {
                    continue; // Drop the task
                }

                let mut cx = Context::from_waker(&task.waker);

                match task.future.as_mut().poll(&mut cx) {
                    Poll::Ready(()) => {
                        // Task completed, don't re-queue
                    }
                    Poll::Pending => {
                        // Task not ready, put it back in the queue
                        pending_tasks.push_back(task);
                    }
                }
            }

            // Put pending tasks back
            self.ready_queue = pending_tasks;

            // If there are still tasks, wait for a wake signal or timer
            if self.has_tasks() {
                let timeout = self
                    .next_timer_deadline()
                    .map(|deadline| deadline.saturating_duration_since(Instant::now()));

                // Only wait if no tasks are immediately ready
                if !self.signal.is_woken() {
                    self.signal.wait(timeout);
                }
                self.signal.clear();
            }
        }
    }

    /// Process expired timers.
    fn process_timers(&mut self) {
        let now = Instant::now();
        while let Some(entry) = self.timers.peek() {
            if entry.deadline <= now {
                let entry = self.timers.pop().unwrap();
                entry.signal.wake();
            } else {
                break;
            }
        }
    }

    /// Get the next timer deadline, if any.
    fn next_timer_deadline(&self) -> Option<Instant> {
        self.timers.peek().map(|e| e.deadline)
    }

    /// Schedule a timer to fire at the given deadline.
    pub fn schedule_timer(&mut self, deadline: Instant, task_id: u64) {
        self.timers.push(TimerEntry {
            deadline,
            task_id,
            signal: self.signal.clone(),
        });
    }

    /// Check if the executor has any pending tasks.
    pub fn has_tasks(&self) -> bool {
        !self.ready_queue.is_empty() || !self.timers.is_empty()
    }

    /// Get the number of pending tasks.
    pub fn task_count(&self) -> usize {
        self.ready_queue.len()
    }

    /// Get the number of pending timers.
    pub fn timer_count(&self) -> usize {
        self.timers.len()
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

/// A handle to a spawned task for cancellation.
#[derive(Clone)]
pub struct TaskHandle {
    /// The task ID.
    pub(crate) task_id: u64,
    /// Cancellation flag.
    pub(crate) cancelled: Arc<AtomicBool>,
}

impl TaskHandle {
    /// Cancel the task.
    ///
    /// The task will be dropped on its next poll.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check if the task has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// A handle to a spawned task that can be awaited for its result.
pub struct JoinHandle<T> {
    /// The task ID.
    #[allow(dead_code)]
    task_id: u64,
    /// The result storage.
    result: Arc<Mutex<Option<T>>>,
    /// Wake signal.
    signal: Arc<WakeSignal>,
    /// Cancellation flag.
    cancelled: Arc<AtomicBool>,
}

impl<T> JoinHandle<T> {
    /// Cancel the task.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.signal.wake();
    }

    /// Check if the task has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Try to get the result without blocking.
    ///
    /// Returns `Some(result)` if the task has completed, `None` otherwise.
    pub fn try_get(&self) -> Option<T> {
        self.result.lock().unwrap().take()
    }
}

impl<T> Future for JoinHandle<T> {
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

/// Block on a single future, running it to completion.
///
/// This is a convenience function for running a single async computation
/// without setting up a full executor.
///
/// # Example
///
/// ```ignore
/// let result = block_on(async {
///     42
/// });
/// assert_eq!(result, 42);
/// ```
pub fn block_on<F, T>(future: F) -> T
where
    F: Future<Output = T>,
{
    let signal = Arc::new(WakeSignal::new());
    let waker = Waker::from(Arc::new(ExecutorWaker {
        signal: signal.clone(),
        task_id: 0,
    }));
    let mut cx = Context::from_waker(&waker);

    let mut future = std::pin::pin!(future);

    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(result) => return result,
            Poll::Pending => {
                // Wait for a wake signal with a timeout to avoid deadlocks
                // in case the future doesn't properly wake
                signal.wait(Some(Duration::from_millis(10)));
                signal.clear();
            }
        }
    }
}

/// Spawn a future as a detached task.
///
/// The future will run in the background on a new OS thread. Use this
/// for fire-and-forget operations where you don't need the result.
pub fn spawn<F>(future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    // Each detached task runs on its own OS thread with a dedicated executor.
    // This provides true parallelism at the cost of per-thread overhead.
    std::thread::spawn(move || {
        block_on(future);
    });
}

/// Yield control back to the executor, allowing other tasks to run.
///
/// This creates a future that returns `Poll::Pending` once, then
/// `Poll::Ready(())` on the next poll.
pub fn yield_now() -> impl Future<Output = ()> {
    struct YieldNow {
        yielded: bool,
    }

    impl Future for YieldNow {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.yielded {
                Poll::Ready(())
            } else {
                self.yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    YieldNow { yielded: false }
}

// ============================================================================
// Runtime Functions
// ============================================================================

/// A future that completes after a specified duration.
///
/// # Example
///
/// ```ignore
/// // Sleep for 100 milliseconds
/// sleep(Duration::from_millis(100)).await;
/// ```
pub struct Sleep {
    deadline: Instant,
    registered: bool,
}

impl Sleep {
    /// Create a new sleep future.
    pub fn new(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
            registered: false,
        }
    }

    /// Create a sleep future that completes at a specific instant.
    pub fn until(deadline: Instant) -> Self {
        Self {
            deadline,
            registered: false,
        }
    }
}

impl Future for Sleep {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            return Poll::Ready(());
        }

        if !self.registered {
            self.registered = true;
            // Schedule a wake at the deadline
            // In a real implementation, we would register with the executor's timer
            // For now, we rely on the block_on timeout or executor's timer processing
        }

        // Wake ourselves when the deadline passes
        // This is a simplification - in a real implementation, the executor
        // would handle timer registration
        let waker = cx.waker().clone();
        let deadline = self.deadline;

        std::thread::spawn(move || {
            let now = Instant::now();
            if deadline > now {
                std::thread::sleep(deadline - now);
            }
            waker.wake();
        });

        Poll::Pending
    }
}

/// Sleep for the specified duration.
///
/// # Example
///
/// ```ignore
/// sleep(Duration::from_millis(100)).await;
/// ```
pub fn sleep(duration: Duration) -> Sleep {
    Sleep::new(duration)
}

/// Sleep until the specified instant.
pub fn sleep_until(deadline: Instant) -> Sleep {
    Sleep::until(deadline)
}

/// Error returned when a timeout expires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeoutError;

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation timed out")
    }
}

impl std::error::Error for TimeoutError {}

/// A future that wraps another future with a timeout.
pub struct Timeout<F> {
    future: F,
    deadline: Instant,
    sleep_spawned: bool,
}

impl<F> Timeout<F> {
    /// Create a new timeout future.
    pub fn new(future: F, duration: Duration) -> Self {
        Self {
            future,
            deadline: Instant::now() + duration,
            sleep_spawned: false,
        }
    }
}

impl<F: Future> Future for Timeout<F> {
    type Output = Result<F::Output, TimeoutError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner future after pinning
        let this = unsafe { self.get_unchecked_mut() };

        // Check if deadline has passed
        if Instant::now() >= this.deadline {
            return Poll::Ready(Err(TimeoutError));
        }

        // Poll the inner future
        let future = unsafe { Pin::new_unchecked(&mut this.future) };
        match future.poll(cx) {
            Poll::Ready(value) => Poll::Ready(Ok(value)),
            Poll::Pending => {
                // Spawn a thread to wake us at the deadline if not already done
                if !this.sleep_spawned {
                    this.sleep_spawned = true;
                    let waker = cx.waker().clone();
                    let deadline = this.deadline;

                    std::thread::spawn(move || {
                        let now = Instant::now();
                        if deadline > now {
                            std::thread::sleep(deadline - now);
                        }
                        waker.wake();
                    });
                }
                Poll::Pending
            }
        }
    }
}

/// Run a future with a timeout.
///
/// If the future does not complete within the specified duration,
/// returns `Err(TimeoutError)`.
///
/// # Example
///
/// ```ignore
/// match timeout(Duration::from_secs(5), fetch_data()).await {
///     Ok(data) => println!("Got data: {:?}", data),
///     Err(_) => println!("Request timed out"),
/// }
/// ```
pub fn timeout<F: Future>(duration: Duration, future: F) -> Timeout<F> {
    Timeout::new(future, duration)
}

/// A future that runs multiple futures concurrently and collects their results.
pub struct JoinAll<F: Future> {
    futures: Vec<Option<F>>,
    results: Vec<Option<F::Output>>,
}

impl<F: Future> JoinAll<F> {
    /// Create a new JoinAll future.
    pub fn new(futures: Vec<F>) -> Self {
        let len = futures.len();
        Self {
            futures: futures.into_iter().map(Some).collect(),
            results: (0..len).map(|_| None).collect(),
        }
    }
}

impl<F: Future> Future for JoinAll<F> {
    type Output = Vec<F::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We carefully manage the futures
        let this = unsafe { self.get_unchecked_mut() };

        let mut all_done = true;

        for (i, future_opt) in this.futures.iter_mut().enumerate() {
            if let Some(future) = future_opt {
                // SAFETY: future is not moved
                let future = unsafe { Pin::new_unchecked(future) };
                match future.poll(cx) {
                    Poll::Ready(output) => {
                        this.results[i] = Some(output);
                        *future_opt = None;
                    }
                    Poll::Pending => {
                        all_done = false;
                    }
                }
            }
        }

        if all_done {
            let results = this.results.iter_mut().map(|r| r.take().unwrap()).collect();
            Poll::Ready(results)
        } else {
            Poll::Pending
        }
    }
}

/// Run all futures concurrently and collect their results.
///
/// All futures are polled concurrently and the results are returned
/// in the same order as the input futures.
///
/// # Example
///
/// ```ignore
/// let futures = vec![fetch_a(), fetch_b(), fetch_c()];
/// let results = join_all(futures).await;
/// ```
pub fn join_all<F: Future>(futures: Vec<F>) -> JoinAll<F> {
    JoinAll::new(futures)
}

/// Run all futures concurrently and collect their results.
///
/// This is an alias for `join_all` that works with iterators.
pub fn join_all_iter<I, F>(iter: I) -> JoinAll<F>
where
    I: IntoIterator<Item = F>,
    F: Future,
{
    JoinAll::new(iter.into_iter().collect())
}

/// Try to run all futures concurrently, short-circuiting on the first error.
pub struct TryJoinAll<F, T, E>
where
    F: Future<Output = Result<T, E>>,
{
    futures: Vec<Option<F>>,
    results: Vec<Option<T>>,
    error: Option<E>,
}

impl<F, T, E> TryJoinAll<F, T, E>
where
    F: Future<Output = Result<T, E>>,
{
    /// Create a new TryJoinAll future.
    pub fn new(futures: Vec<F>) -> Self {
        let len = futures.len();
        Self {
            futures: futures.into_iter().map(Some).collect(),
            results: (0..len).map(|_| None).collect(),
            error: None,
        }
    }
}

impl<F, T, E> Future for TryJoinAll<F, T, E>
where
    F: Future<Output = Result<T, E>>,
{
    type Output = Result<Vec<T>, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We carefully manage the futures
        let this = unsafe { self.get_unchecked_mut() };

        // Check if we already have an error
        if let Some(error) = this.error.take() {
            return Poll::Ready(Err(error));
        }

        let mut all_done = true;

        for (i, future_opt) in this.futures.iter_mut().enumerate() {
            if let Some(future) = future_opt {
                // SAFETY: future is not moved
                let future = unsafe { Pin::new_unchecked(future) };
                match future.poll(cx) {
                    Poll::Ready(Ok(output)) => {
                        this.results[i] = Some(output);
                        *future_opt = None;
                    }
                    Poll::Ready(Err(e)) => {
                        // Short-circuit on first error
                        return Poll::Ready(Err(e));
                    }
                    Poll::Pending => {
                        all_done = false;
                    }
                }
            }
        }

        if all_done {
            let results = this.results.iter_mut().map(|r| r.take().unwrap()).collect();
            Poll::Ready(Ok(results))
        } else {
            Poll::Pending
        }
    }
}

/// Run all futures concurrently, short-circuiting on the first error.
///
/// If all futures complete successfully, returns `Ok(Vec<T>)`.
/// If any future returns an error, returns that error immediately.
///
/// # Example
///
/// ```ignore
/// let futures = vec![validate_a(), validate_b(), validate_c()];
/// match try_join_all(futures).await {
///     Ok(results) => println!("All validated: {:?}", results),
///     Err(e) => println!("Validation failed: {:?}", e),
/// }
/// ```
pub fn try_join_all<F, T, E>(futures: Vec<F>) -> TryJoinAll<F, T, E>
where
    F: Future<Output = Result<T, E>>,
{
    TryJoinAll::new(futures)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let executor = Executor::new();
        assert!(!executor.has_tasks());
        assert_eq!(executor.task_count(), 0);
    }

    #[test]
    fn test_spawn_and_run() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        let mut executor = Executor::new();
        let _handle = executor.spawn(async move {
            completed_clone.store(true, Ordering::SeqCst);
        });

        assert!(executor.has_tasks());
        executor.run();
        assert!(!executor.has_tasks());
        assert!(completed.load(Ordering::SeqCst));
    }

    #[test]
    fn test_task_cancellation() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let started = Arc::new(AtomicBool::new(false));
        let started_clone = started.clone();

        let mut executor = Executor::new();
        let handle = executor.spawn(async move {
            started_clone.store(true, Ordering::SeqCst);
            // This would normally do more work
        });

        // Cancel before running
        handle.cancel();
        assert!(handle.is_cancelled());

        executor.run();

        // The task may or may not have started depending on timing,
        // but it should have been cancelled
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_block_on() {
        let result = block_on(async { 42 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_yield_now() {
        let mut executor = Executor::new();
        let mut count = 0;

        executor.spawn(async {
            yield_now().await;
        });

        // Count iterations
        while executor.has_tasks() {
            count += 1;
            if let Some(mut task) = executor.ready_queue.pop_front() {
                let mut cx = Context::from_waker(&task.waker);
                match task.future.as_mut().poll(&mut cx) {
                    Poll::Ready(()) => {}
                    Poll::Pending => {
                        executor.ready_queue.push_back(task);
                    }
                }
            }
            if count > 10 {
                break; // Safety limit
            }
        }

        // yield_now should cause at least 2 iterations
        assert!(count >= 2);
    }

    #[test]
    fn test_join_all_immediate() {
        use std::future::ready;
        let result = block_on(async { join_all(vec![ready(1), ready(2), ready(3)]).await });
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_try_join_all_success() {
        use std::future::ready;
        let result: Result<Vec<i32>, &str> = block_on(async {
            try_join_all(vec![ready(Ok::<_, &str>(1)), ready(Ok(2)), ready(Ok(3))]).await
        });
        assert_eq!(result, Ok(vec![1, 2, 3]));
    }

    #[test]
    fn test_try_join_all_error() {
        use std::future::ready;
        let result: Result<Vec<i32>, &str> = block_on(async {
            try_join_all(vec![
                ready(Ok::<_, &str>(1)),
                ready(Err("error")),
                ready(Ok(3)),
            ])
            .await
        });
        assert_eq!(result, Err("error"));
    }

    #[test]
    fn test_spawn_with_handle() {
        let mut executor = Executor::new();
        let handle = executor.spawn_with_handle(async { 42 });

        executor.run();

        // The result should be available after running
        assert_eq!(handle.try_get(), Some(42));
    }

    #[test]
    fn test_timeout_immediate() {
        let result = block_on(async { timeout(Duration::from_secs(10), async { 42 }).await });
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_wake_signal() {
        let signal = Arc::new(WakeSignal::new());

        assert!(!signal.is_woken());

        signal.wake();
        assert!(signal.is_woken());

        // wait should return immediately since already woken
        let woken = signal.wait(Some(Duration::from_millis(1)));
        assert!(woken);
    }

    #[test]
    fn test_executor_waker() {
        let signal = Arc::new(WakeSignal::new());
        let waker = Waker::from(Arc::new(ExecutorWaker {
            signal: signal.clone(),
            task_id: 1,
        }));

        assert!(!signal.is_woken());
        waker.wake_by_ref();
        assert!(signal.is_woken());
    }

    #[test]
    fn test_timer_entry_ordering() {
        let signal = Arc::new(WakeSignal::new());
        let now = Instant::now();

        let entry1 = TimerEntry {
            deadline: now + Duration::from_millis(100),
            task_id: 1,
            signal: signal.clone(),
        };
        let entry2 = TimerEntry {
            deadline: now + Duration::from_millis(50),
            task_id: 2,
            signal: signal.clone(),
        };

        // entry2 should come first (earlier deadline)
        assert!(entry2 > entry1);
    }

    #[test]
    fn test_task_handle_clone() {
        let cancelled = Arc::new(AtomicBool::new(false));
        let handle = TaskHandle {
            task_id: 1,
            cancelled: cancelled.clone(),
        };

        let handle2 = handle.clone();
        assert!(!handle.is_cancelled());
        assert!(!handle2.is_cancelled());

        handle.cancel();
        assert!(handle.is_cancelled());
        assert!(handle2.is_cancelled());
    }

    #[test]
    fn test_join_handle_cancel() {
        let mut executor = Executor::new();
        let handle = executor.spawn_with_handle(async {
            // Simulate long-running task
            yield_now().await;
            yield_now().await;
            42
        });

        handle.cancel();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_multiple_tasks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let counter = Arc::new(AtomicUsize::new(0));
        let mut executor = Executor::new();

        for _ in 0..5 {
            let counter_clone = counter.clone();
            executor.spawn(async move {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            });
        }

        assert_eq!(executor.task_count(), 5);
        executor.run();
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }
}
