//! Task groups for structured concurrency
//!
//! Task groups provide structured concurrency - a way to spawn multiple tasks
//! that are guaranteed to complete before the parent continues. This prevents
//! "orphan" tasks and makes concurrent code easier to reason about.
//!
//! # Design
//!
//! - Child tasks are scoped to their parent task group
//! - The task group waits for all children before completing
//! - Cancellation propagates from parent to children
//! - Errors can optionally cancel sibling tasks
//!
//! # Integration with Work-Stealing Executor
//!
//! When using `WorkStealingTaskGroup`, tasks are spawned onto the global
//! work-stealing executor for efficient multi-threaded execution. Otherwise,
//! the basic `TaskGroup` spawns OS threads for simplicity.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::executor::TaskHandle;
use crate::work_stealing::{WorkStealingTaskHandle, spawn_global};

/// A task group for structured concurrency.
///
/// Task groups allow spawning multiple concurrent tasks and waiting for
/// all of them to complete. The task group ensures that all child tasks
/// complete before the group completes.
///
/// # Example
///
/// ```ignore
/// let mut group = TaskGroup::new();
///
/// group.spawn(async {
///     // Task 1
/// });
///
/// group.spawn(async {
///     // Task 2
/// });
///
/// // Wait for all tasks to complete
/// group.join().await;
/// ```
pub struct TaskGroup {
    /// Shared state for the task group.
    inner: Arc<TaskGroupInner>,
}

struct TaskGroupInner {
    /// Number of active tasks (including pending spawns).
    active_count: AtomicUsize,
    /// Whether the group has been cancelled.
    cancelled: AtomicBool,
    /// Waker to notify when all tasks complete.
    waker: Mutex<Option<Waker>>,
    /// Task handles for cancellation.
    handles: Mutex<Vec<TaskHandle>>,
}

impl TaskGroup {
    /// Create a new task group.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TaskGroupInner {
                active_count: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                waker: Mutex::new(None),
                handles: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Spawn a task into this group.
    ///
    /// The spawned task will be tracked by the group and must complete
    /// before the group's `join()` method returns.
    ///
    /// Returns a handle that can be used to cancel this specific task.
    pub fn spawn<F>(&self, future: F) -> TaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        // Check if cancelled
        if self.inner.cancelled.load(Ordering::Acquire) {
            // Return a dummy cancelled handle
            return TaskHandle::new_cancelled();
        }

        self.inner.active_count.fetch_add(1, Ordering::AcqRel);

        let inner = self.inner.clone();

        // Wrap the future to decrement count when done
        let wrapped = async move {
            future.await;

            // Decrement active count
            let prev = inner.active_count.fetch_sub(1, Ordering::AcqRel);

            // If this was the last task, wake the joiner
            if prev == 1 {
                if let Some(waker) = inner.waker.lock().unwrap().take() {
                    waker.wake();
                }
            }
        };

        let handle = TaskHandle::new_placeholder();

        // Store handle for cancellation
        self.inner.handles.lock().unwrap().push(handle.clone());

        // Spawn the future on a new OS thread with its own executor.
        // Each spawned task gets its own thread for true parallelism.
        std::thread::spawn(move || {
            crate::executor::block_on(wrapped);
        });

        handle
    }

    /// Spawn a task that returns a value.
    ///
    /// Returns a `JoinHandle` that can be awaited to get the result.
    pub fn spawn_with_result<F, T>(&self, future: F) -> TaskJoinHandle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        // Check if cancelled
        if self.inner.cancelled.load(Ordering::Acquire) {
            return TaskJoinHandle::new_cancelled();
        }

        self.inner.active_count.fetch_add(1, Ordering::AcqRel);

        let result = Arc::new(Mutex::new(None));
        let result_clone = result.clone();
        let inner = self.inner.clone();

        // Wrap the future to store result and decrement count
        let wrapped = async move {
            let output = future.await;
            *result_clone.lock().unwrap() = Some(output);

            // Decrement active count
            let prev = inner.active_count.fetch_sub(1, Ordering::AcqRel);

            if prev == 1 {
                if let Some(waker) = inner.waker.lock().unwrap().take() {
                    waker.wake();
                }
            }
        };

        let cancelled = Arc::new(AtomicBool::new(false));

        // Spawn the wrapped future
        std::thread::spawn(move || {
            crate::executor::block_on(wrapped);
        });

        TaskJoinHandle {
            result,
            cancelled,
            waker: Arc::new(Mutex::new(None)),
        }
    }

    /// Wait for all tasks in the group to complete.
    ///
    /// This method blocks until all spawned tasks have finished.
    pub fn join(&self) -> Join<'_> {
        Join { group: self }
    }

    /// Cancel all tasks in the group.
    ///
    /// This sets the cancelled flag and cancels all task handles.
    /// Tasks should check for cancellation periodically.
    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::Release);

        // Cancel all handles
        for handle in self.inner.handles.lock().unwrap().iter() {
            handle.cancel();
        }

        // Wake the joiner
        if let Some(waker) = self.inner.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    /// Check if the group has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Get the number of active tasks.
    pub fn active_count(&self) -> usize {
        self.inner.active_count.load(Ordering::Acquire)
    }

    /// Check if all tasks have completed.
    pub fn is_complete(&self) -> bool {
        self.inner.active_count.load(Ordering::Acquire) == 0
    }
}

impl Default for TaskGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Future returned by [`TaskGroup::join`].
pub struct Join<'a> {
    group: &'a TaskGroup,
}

impl Future for Join<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Check if all tasks are complete
        if self.group.inner.active_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(());
        }

        // Check if cancelled
        if self.group.inner.cancelled.load(Ordering::Acquire) {
            // Even if cancelled, wait for tasks to acknowledge
            if self.group.inner.active_count.load(Ordering::Acquire) == 0 {
                return Poll::Ready(());
            }
        }

        // Register waker and wait
        *self.group.inner.waker.lock().unwrap() = Some(cx.waker().clone());

        // Double-check after registering waker
        if self.group.inner.active_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(());
        }

        Poll::Pending
    }
}

/// A handle to a task spawned with a result.
pub struct TaskJoinHandle<T> {
    result: Arc<Mutex<Option<T>>>,
    cancelled: Arc<AtomicBool>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl<T> TaskJoinHandle<T> {
    fn new_cancelled() -> Self {
        Self {
            result: Arc::new(Mutex::new(None)),
            cancelled: Arc::new(AtomicBool::new(true)),
            waker: Arc::new(Mutex::new(None)),
        }
    }

    /// Cancel the task.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(waker) = self.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    /// Check if the task has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Try to get the result without blocking.
    pub fn try_get(&self) -> Option<T> {
        self.result.lock().unwrap().take()
    }
}

impl<T> Future for TaskJoinHandle<T> {
    type Output = Option<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_cancelled() {
            return Poll::Ready(None);
        }

        if let Some(result) = self.result.lock().unwrap().take() {
            return Poll::Ready(Some(result));
        }

        *self.waker.lock().unwrap() = Some(cx.waker().clone());

        // Double-check
        if let Some(result) = self.result.lock().unwrap().take() {
            return Poll::Ready(Some(result));
        }

        Poll::Pending
    }
}

// Add placeholder methods to TaskHandle
impl TaskHandle {
    /// Create a cancelled handle.
    pub(crate) fn new_cancelled() -> Self {
        use std::sync::Arc;
        let cancelled = Arc::new(AtomicBool::new(true));
        Self {
            task_id: 0,
            cancelled,
        }
    }

    /// Create a placeholder handle.
    pub(crate) fn new_placeholder() -> Self {
        use std::sync::Arc;
        let cancelled = Arc::new(AtomicBool::new(false));
        Self {
            task_id: crate::executor::next_task_id(),
            cancelled,
        }
    }

    /// Get the task ID.
    pub fn task_id(&self) -> u64 {
        self.task_id
    }
}

/// Scoped task group that ensures all tasks complete before dropping.
///
/// Unlike `TaskGroup`, `ScopedTaskGroup` blocks on drop until all tasks
/// complete. This ensures structured concurrency even if `join()` is not
/// explicitly called.
pub struct ScopedTaskGroup {
    group: TaskGroup,
}

impl ScopedTaskGroup {
    /// Create a new scoped task group.
    pub fn new() -> Self {
        Self {
            group: TaskGroup::new(),
        }
    }

    /// Spawn a task into this group.
    pub fn spawn<F>(&self, future: F) -> TaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.group.spawn(future)
    }

    /// Cancel all tasks in the group.
    pub fn cancel(&self) {
        self.group.cancel();
    }

    /// Check if the group has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.group.is_cancelled()
    }

    /// Get the number of active tasks.
    pub fn active_count(&self) -> usize {
        self.group.active_count()
    }
}

impl Default for ScopedTaskGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ScopedTaskGroup {
    fn drop(&mut self) {
        // Block until all tasks complete
        while self.group.active_count() > 0 {
            std::thread::yield_now();
        }
    }
}

// ============================================================================
// Work-Stealing Task Group
// ============================================================================

/// Shared state for work-stealing task group.
struct WorkStealingTaskGroupInner {
    /// Number of active tasks.
    active_count: AtomicUsize,
    /// Whether the group has been cancelled.
    cancelled: AtomicBool,
    /// Waker to notify when all tasks complete.
    waker: Mutex<Option<Waker>>,
    /// Task handles for cancellation.
    handles: Mutex<Vec<WorkStealingTaskHandle>>,
    /// Error from first failed task (for cancel-on-error mode).
    first_error: Mutex<Option<Box<dyn std::any::Any + Send>>>,
    /// Whether to cancel siblings on error.
    cancel_on_error: AtomicBool,
}

/// A task group that uses the work-stealing executor.
///
/// This is the recommended task group for production use as it efficiently
/// distributes work across multiple threads.
///
/// # Example
///
/// ```ignore
/// let group = WorkStealingTaskGroup::new();
///
/// group.spawn(async {
///     // Task 1 runs on work-stealing executor
/// });
///
/// group.spawn(async {
///     // Task 2 runs on work-stealing executor
/// });
///
/// // Wait for all tasks
/// group.join().await;
/// ```
pub struct WorkStealingTaskGroup {
    inner: Arc<WorkStealingTaskGroupInner>,
}

impl WorkStealingTaskGroup {
    /// Create a new work-stealing task group.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(WorkStealingTaskGroupInner {
                active_count: AtomicUsize::new(0),
                cancelled: AtomicBool::new(false),
                waker: Mutex::new(None),
                handles: Mutex::new(Vec::new()),
                first_error: Mutex::new(None),
                cancel_on_error: AtomicBool::new(false),
            }),
        }
    }

    /// Create a task group that cancels all tasks when one fails.
    pub fn with_cancel_on_error() -> Self {
        let group = Self::new();
        group.inner.cancel_on_error.store(true, Ordering::Release);
        group
    }

    /// Spawn a task into the group.
    ///
    /// The task will run on the global work-stealing executor.
    pub fn spawn<F>(&self, future: F) -> WorkStealingTaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        if self.inner.cancelled.load(Ordering::Acquire) {
            return WorkStealingTaskHandle::cancelled();
        }

        self.inner.active_count.fetch_add(1, Ordering::AcqRel);

        let inner = self.inner.clone();

        let wrapped = async move {
            future.await;

            let prev = inner.active_count.fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                if let Some(waker) = inner.waker.lock().unwrap().take() {
                    waker.wake();
                }
            }
        };

        let handle = spawn_global(wrapped);
        self.inner.handles.lock().unwrap().push(handle.clone());
        handle
    }

    /// Spawn a fallible task that can cancel siblings on error.
    ///
    /// If the task returns an error and `cancel_on_error` is enabled,
    /// all other tasks in the group will be cancelled.
    pub fn spawn_fallible<F, E>(&self, future: F) -> WorkStealingTaskHandle
    where
        F: Future<Output = Result<(), E>> + Send + 'static,
        E: Send + 'static,
    {
        if self.inner.cancelled.load(Ordering::Acquire) {
            return WorkStealingTaskHandle::cancelled();
        }

        self.inner.active_count.fetch_add(1, Ordering::AcqRel);

        let inner = self.inner.clone();

        let wrapped = async move {
            match future.await {
                Ok(()) => {}
                Err(e) => {
                    // Store the error
                    let mut first_error = inner.first_error.lock().unwrap();
                    if first_error.is_none() {
                        *first_error = Some(Box::new(e));
                    }

                    // Cancel siblings if configured
                    if inner.cancel_on_error.load(Ordering::Acquire) {
                        inner.cancelled.store(true, Ordering::Release);
                        for handle in inner.handles.lock().unwrap().iter() {
                            handle.cancel();
                        }
                    }
                }
            }

            let prev = inner.active_count.fetch_sub(1, Ordering::AcqRel);
            if prev == 1 {
                if let Some(waker) = inner.waker.lock().unwrap().take() {
                    waker.wake();
                }
            }
        };

        let handle = spawn_global(wrapped);
        self.inner.handles.lock().unwrap().push(handle.clone());
        handle
    }

    /// Wait for all tasks to complete.
    pub fn join(&self) -> WorkStealingJoin<'_> {
        WorkStealingJoin { group: self }
    }

    /// Cancel all tasks in the group.
    pub fn cancel(&self) {
        self.inner.cancelled.store(true, Ordering::Release);

        for handle in self.inner.handles.lock().unwrap().iter() {
            handle.cancel();
        }

        if let Some(waker) = self.inner.waker.lock().unwrap().take() {
            waker.wake();
        }
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Get the number of active tasks.
    pub fn active_count(&self) -> usize {
        self.inner.active_count.load(Ordering::Acquire)
    }

    /// Check if all tasks have completed.
    pub fn is_complete(&self) -> bool {
        self.inner.active_count.load(Ordering::Acquire) == 0
    }

    /// Take the first error if any task failed.
    pub fn take_error(&self) -> Option<Box<dyn std::any::Any + Send>> {
        self.inner.first_error.lock().unwrap().take()
    }
}

impl Default for WorkStealingTaskGroup {
    fn default() -> Self {
        Self::new()
    }
}

/// Future for joining a work-stealing task group.
pub struct WorkStealingJoin<'a> {
    group: &'a WorkStealingTaskGroup,
}

impl Future for WorkStealingJoin<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.group.inner.active_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(());
        }

        if self.group.inner.cancelled.load(Ordering::Acquire) {
            if self.group.inner.active_count.load(Ordering::Acquire) == 0 {
                return Poll::Ready(());
            }
        }

        *self.group.inner.waker.lock().unwrap() = Some(cx.waker().clone());

        if self.group.inner.active_count.load(Ordering::Acquire) == 0 {
            return Poll::Ready(());
        }

        Poll::Pending
    }
}

/// A scoped work-stealing task group that blocks on drop.
pub struct ScopedWorkStealingTaskGroup {
    group: WorkStealingTaskGroup,
}

impl ScopedWorkStealingTaskGroup {
    /// Create a new scoped work-stealing task group.
    pub fn new() -> Self {
        Self {
            group: WorkStealingTaskGroup::new(),
        }
    }

    /// Spawn a task into the group.
    pub fn spawn<F>(&self, future: F) -> WorkStealingTaskHandle
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.group.spawn(future)
    }

    /// Cancel all tasks.
    pub fn cancel(&self) {
        self.group.cancel();
    }

    /// Check if cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.group.is_cancelled()
    }

    /// Get active task count.
    pub fn active_count(&self) -> usize {
        self.group.active_count()
    }
}

impl Default for ScopedWorkStealingTaskGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ScopedWorkStealingTaskGroup {
    fn drop(&mut self) {
        // Busy-wait for all tasks to complete
        while self.group.active_count() > 0 {
            std::thread::yield_now();
        }
    }
}

// ============================================================================
// WorkStealingTaskHandle extensions
// ============================================================================

impl WorkStealingTaskHandle {
    /// Create a cancelled handle.
    pub(crate) fn cancelled() -> Self {
        let cancelled = Arc::new(AtomicBool::new(true));
        Self {
            task_id: 0,
            cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_group_creation() {
        let group = TaskGroup::new();
        assert_eq!(group.active_count(), 0);
        assert!(group.is_complete());
        assert!(!group.is_cancelled());
    }

    #[test]
    fn test_task_group_cancel() {
        let group = TaskGroup::new();
        assert!(!group.is_cancelled());
        group.cancel();
        assert!(group.is_cancelled());
    }

    #[test]
    fn test_scoped_task_group_creation() {
        let group = ScopedTaskGroup::new();
        assert_eq!(group.active_count(), 0);
        assert!(!group.is_cancelled());
    }

    #[test]
    fn test_scoped_task_group_cancel() {
        let group = ScopedTaskGroup::new();
        group.cancel();
        assert!(group.is_cancelled());
    }

    #[test]
    fn test_task_handle_cancelled() {
        let handle = TaskHandle::new_cancelled();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_task_handle_placeholder() {
        let handle = TaskHandle::new_placeholder();
        assert!(!handle.is_cancelled());
        handle.cancel();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_task_join_handle_cancelled() {
        let handle: TaskJoinHandle<i32> = TaskJoinHandle::new_cancelled();
        assert!(handle.is_cancelled());
    }

    // Work-stealing task group tests

    #[test]
    fn test_work_stealing_task_group_creation() {
        let group = WorkStealingTaskGroup::new();
        assert_eq!(group.active_count(), 0);
        assert!(group.is_complete());
        assert!(!group.is_cancelled());
    }

    #[test]
    fn test_work_stealing_task_group_cancel() {
        let group = WorkStealingTaskGroup::new();
        assert!(!group.is_cancelled());
        group.cancel();
        assert!(group.is_cancelled());
    }

    #[test]
    fn test_work_stealing_task_handle_cancelled() {
        let handle = WorkStealingTaskHandle::cancelled();
        assert!(handle.is_cancelled());
        assert_eq!(handle.task_id(), 0);
    }

    #[test]
    fn test_scoped_work_stealing_task_group_creation() {
        let group = ScopedWorkStealingTaskGroup::new();
        assert_eq!(group.active_count(), 0);
        assert!(!group.is_cancelled());
    }

    #[test]
    fn test_scoped_work_stealing_task_group_cancel() {
        let group = ScopedWorkStealingTaskGroup::new();
        group.cancel();
        assert!(group.is_cancelled());
    }

    #[test]
    fn test_work_stealing_cancel_on_error() {
        let group = WorkStealingTaskGroup::with_cancel_on_error();
        assert!(!group.is_cancelled());
        assert_eq!(group.active_count(), 0);
    }
}
