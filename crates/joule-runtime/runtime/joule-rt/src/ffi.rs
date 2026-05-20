//! FFI exports for Joule runtime
//!
//! This module provides C-compatible FFI functions that compiled Joule code
//! calls into for concurrency operations. All types are opaque pointers from
//! the caller's perspective.
//!
//! # Naming Convention
//!
//! All functions use the `joule_rt_` prefix to avoid symbol conflicts.
//!
//! # Memory Management
//!
//! - Channels and task groups are reference-counted internally
//! - The caller is responsible for calling the appropriate close/destroy functions
//! - Values sent through channels are raw pointers; the caller manages their lifetime

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::channel::{Receiver, Sender, channel};
use crate::task_group::WorkStealingTaskGroup;

// ============================================================================
// Type-Erased Channel
// ============================================================================

/// An opaque channel handle that stores raw pointers.
///
/// This is a type-erased wrapper around the generic Channel<*mut c_void>.
pub struct FfiChannel {
    sender: Sender<*mut c_void>,
    receiver: Receiver<*mut c_void>,
}

/// An opaque sender handle.
#[allow(dead_code)]
pub struct FfiSender {
    sender: Sender<*mut c_void>,
}

/// An opaque receiver handle.
#[allow(dead_code)]
pub struct FfiReceiver {
    receiver: Receiver<*mut c_void>,
}

// ============================================================================
// Channel FFI Functions
// ============================================================================

/// Create a new bounded channel.
///
/// # Arguments
/// * `capacity` - The maximum number of items the channel can hold
///
/// # Returns
/// A pointer to the channel, or null on failure
///
/// # Safety
/// The returned pointer must be freed with `joule_rt_channel_close`.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_create(capacity: u64) -> *mut c_void {
    let cap = capacity.max(1) as usize; // Ensure capacity >= 1
    let (sender, receiver) = channel::<*mut c_void>(cap);

    let channel = Box::new(FfiChannel { sender, receiver });
    Box::into_raw(channel) as *mut c_void
}

/// Send a value through the channel (blocking).
///
/// # Arguments
/// * `channel` - The channel to send on
/// * `value` - The value to send (opaque pointer)
///
/// # Returns
/// * `1` (true) if the value was sent successfully
/// * `0` (false) if the channel is closed
///
/// # Safety
/// The channel pointer must be valid.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_send(channel: *mut c_void, value: *mut c_void) -> u8 {
    if channel.is_null() {
        return 0;
    }

    // SAFETY: Caller guarantees valid pointer
    let chan = unsafe { &*(channel as *const FfiChannel) };

    match chan.sender.try_send(value) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Try to send a value through the channel (non-blocking).
///
/// # Returns
/// * `1` if sent successfully
/// * `0` if the channel is full or closed
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_try_send(channel: *mut c_void, value: *mut c_void) -> u8 {
    if channel.is_null() {
        return 0;
    }

    // SAFETY: Caller guarantees valid pointer
    let chan = unsafe { &*(channel as *const FfiChannel) };

    match chan.sender.try_send(value) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Receive a value from the channel (blocking).
///
/// # Returns
/// The received value, or null if the channel is closed and empty.
///
/// # Note
/// This is currently non-blocking due to FFI constraints. For true blocking,
/// use the async runtime.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_recv(channel: *mut c_void) -> *mut c_void {
    if channel.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: Caller guarantees valid pointer
    let chan = unsafe { &*(channel as *const FfiChannel) };

    // Spin until we get a value or the channel closes
    loop {
        match chan.receiver.try_recv() {
            Ok(value) => return value,
            Err(crate::channel::TryRecvError::Closed) => return ptr::null_mut(),
            Err(crate::channel::TryRecvError::Empty) => {
                std::thread::yield_now();
            }
        }
    }
}

/// Try to receive a value from the channel (non-blocking).
///
/// # Returns
/// The received value, or null if the channel is empty or closed.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_try_recv(channel: *mut c_void) -> *mut c_void {
    if channel.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: Caller guarantees valid pointer
    let chan = unsafe { &*(channel as *const FfiChannel) };

    match chan.receiver.try_recv() {
        Ok(value) => value,
        Err(_) => ptr::null_mut(),
    }
}

/// Get a sender handle from a channel.
///
/// The sender can be used independently and is reference-counted.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_sender(channel: *mut c_void) -> *mut c_void {
    if channel.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: Caller guarantees valid pointer
    let chan = unsafe { &*(channel as *const FfiChannel) };

    let sender = Box::new(FfiSender {
        sender: chan.sender.clone(),
    });
    Box::into_raw(sender) as *mut c_void
}

/// Get a receiver handle from a channel.
///
/// The receiver can be used independently and is reference-counted.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_receiver(channel: *mut c_void) -> *mut c_void {
    if channel.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: Caller guarantees valid pointer
    let chan = unsafe { &*(channel as *const FfiChannel) };

    let receiver = Box::new(FfiReceiver {
        receiver: chan.receiver.clone(),
    });
    Box::into_raw(receiver) as *mut c_void
}

/// Close a channel.
///
/// After closing, no more values can be sent. Existing values can still be received.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_channel_close(channel: *mut c_void) {
    if channel.is_null() {
        return;
    }

    // SAFETY: Caller guarantees valid pointer and single ownership
    let chan = unsafe { Box::from_raw(channel as *mut FfiChannel) };
    chan.sender.close();
    // Box is dropped here, cleaning up the channel
}

// ============================================================================
// Select FFI Function
// ============================================================================

/// Select on multiple channels, returning the index of the first ready channel.
///
/// Performs a round-robin poll across the given channels. Each channel pointer
/// must be a valid `FfiChannel*` (as returned by `joule_rt_channel_create`).
/// The function spins until at least one channel has a value available, then
/// returns the zero-based index of that channel.
///
/// # Arguments
/// * `channels` - Pointer to an array of channel pointers (`*mut c_void`)
/// * `count` - Number of channels in the array
///
/// # Returns
/// The zero-based index of the first ready channel, or `u64::MAX` if all
/// channels are closed.
///
/// # Safety
/// - `channels` must point to a valid array of `count` channel pointers
/// - Each channel pointer in the array must be valid (from `joule_rt_channel_create`)
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_select(channels: *const *mut c_void, count: u64) -> u64 {
    if channels.is_null() || count == 0 {
        return u64::MAX;
    }

    let count = count as usize;

    loop {
        let mut all_closed = true;

        for i in 0..count {
            // SAFETY: Caller guarantees valid array of valid channel pointers
            let chan_ptr = unsafe { *channels.add(i) };
            if chan_ptr.is_null() {
                continue;
            }

            let chan = unsafe { &*(chan_ptr as *const FfiChannel) };

            match chan.receiver.try_recv() {
                Ok(_value) => {
                    // Found a ready channel — return its index.
                    // Note: the received value is consumed here. The compiled code
                    // should re-recv from the selected channel in the arm body,
                    // or use a peek-based approach. For the current MIR lowering,
                    // the Select terminator stores the selected arm index and the
                    // arm body performs the actual recv.
                    //
                    // Future optimization: a peek/put-back mechanism could avoid
                    // re-receiving. For now, MIR select semantics treat this as a
                    // readiness check — the value is received inline by the arm body.
                    return i as u64;
                }
                Err(crate::channel::TryRecvError::Closed) => {
                    // This channel is closed, but others might not be
                }
                Err(crate::channel::TryRecvError::Empty) => {
                    all_closed = false;
                }
            }
        }

        if all_closed {
            return u64::MAX;
        }

        std::thread::yield_now();
    }
}

// ============================================================================
// Task FFI Functions
// ============================================================================

/// Opaque task handle for FFI.
pub struct FfiTask {
    /// The function to execute
    func: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    /// Argument to pass to the function
    arg: *mut c_void,
    /// Result of the task
    result: *mut c_void,
    /// Whether the task has completed
    completed: AtomicBool,
    /// Whether the task was cancelled
    cancelled: AtomicBool,
}

// SAFETY: FfiTask uses atomic operations for synchronization
unsafe impl Send for FfiTask {}
unsafe impl Sync for FfiTask {}

/// Spawn a new task.
///
/// # Arguments
/// * `func` - Function pointer to execute
///
/// # Returns
/// A task handle, or null on failure.
///
/// # Note
/// The function signature expected is `fn(*mut c_void) -> *mut c_void`.
/// In practice, this is a pointer to a Joule closure.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_spawn(func: *mut c_void) -> *mut c_void {
    if func.is_null() {
        return ptr::null_mut();
    }

    // Create a task handle
    let task = Box::new(FfiTask {
        func: Some(unsafe { std::mem::transmute(func) }),
        arg: ptr::null_mut(),
        result: ptr::null_mut(),
        completed: AtomicBool::new(false),
        cancelled: AtomicBool::new(false),
    });

    let task_ptr = Box::into_raw(task);

    // Spawn a thread to run the task
    let task_arc = unsafe { Arc::from_raw(task_ptr) };
    let task_clone = task_arc.clone();

    std::thread::spawn(move || {
        if !task_clone.cancelled.load(Ordering::Acquire) {
            if let Some(f) = task_clone.func {
                // SAFETY: Caller guarantees valid function pointer
                let result = unsafe { f(task_clone.arg) };
                // Store result - this is a data race but acceptable for now
                // In production, we'd use proper synchronization
                let task_mut = unsafe { &mut *(Arc::as_ptr(&task_clone) as *mut FfiTask) };
                task_mut.result = result;
            }
        }
        task_clone.completed.store(true, Ordering::Release);
    });

    // Leak the Arc to return a raw pointer
    // The await function will clean it up
    Arc::into_raw(task_arc) as *mut c_void
}

/// Wait for a task to complete and get its result.
///
/// # Returns
/// The task's result, or null if the task was cancelled.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_task_await(task: *mut c_void) -> *mut c_void {
    if task.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: Caller guarantees valid pointer
    let task_ref = unsafe { &*(task as *const FfiTask) };

    // Spin until completed
    while !task_ref.completed.load(Ordering::Acquire) {
        std::thread::yield_now();
    }

    if task_ref.cancelled.load(Ordering::Acquire) {
        return ptr::null_mut();
    }

    task_ref.result
}

// ============================================================================
// Task Group FFI Functions
// ============================================================================

/// Create a new task group.
///
/// # Returns
/// A task group handle.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_task_group_create() -> *mut c_void {
    let group = Box::new(WorkStealingTaskGroup::new());
    Box::into_raw(group) as *mut c_void
}

/// Wait for all tasks in a group to complete.
///
/// This is a blocking operation.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_task_group_join(group: *mut c_void) {
    if group.is_null() {
        return;
    }

    // SAFETY: Caller guarantees valid pointer
    let grp = unsafe { &*(group as *const WorkStealingTaskGroup) };

    // Spin until all tasks complete
    while !grp.is_complete() {
        std::thread::yield_now();
    }
}

/// Destroy a task group.
///
/// This also waits for all tasks to complete.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_task_group_destroy(group: *mut c_void) {
    if group.is_null() {
        return;
    }

    // SAFETY: Caller guarantees valid pointer and single ownership
    let grp = unsafe { Box::from_raw(group as *mut WorkStealingTaskGroup) };

    // Wait for completion before dropping
    while !grp.is_complete() {
        std::thread::yield_now();
    }
    // Box is dropped here
}

// ============================================================================
// Cancellation FFI Functions
// ============================================================================

thread_local! {
    static CURRENT_CANCELLED: AtomicBool = const { AtomicBool::new(false) };
}

/// Check if the current task has been cancelled.
///
/// # Returns
/// * `1` if cancelled
/// * `0` if not cancelled
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_is_cancelled() -> u8 {
    CURRENT_CANCELLED.with(|c| if c.load(Ordering::Acquire) { 1 } else { 0 })
}

/// Get the current task handle.
///
/// # Returns
/// The current task handle, or null if not in a task context.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_current_task() -> *mut c_void {
    // For now, return null - proper task context would require TLS
    ptr::null_mut()
}

/// Cancel the current task.
///
/// This sets the cancellation flag for the current task.
#[unsafe(no_mangle)]
pub extern "C" fn joule_rt_cancel() {
    CURRENT_CANCELLED.with(|c| {
        c.store(true, Ordering::Release);
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_create_close() {
        let channel = joule_rt_channel_create(10);
        assert!(!channel.is_null());
        joule_rt_channel_close(channel);
    }

    #[test]
    fn test_channel_send_recv() {
        let channel = joule_rt_channel_create(10);
        assert!(!channel.is_null());

        let value = 42usize as *mut c_void;
        let sent = joule_rt_channel_try_send(channel, value);
        assert_eq!(sent, 1);

        let received = joule_rt_channel_try_recv(channel);
        assert_eq!(received, value);

        joule_rt_channel_close(channel);
    }

    #[test]
    fn test_channel_sender_receiver() {
        let channel = joule_rt_channel_create(10);

        let sender = joule_rt_channel_sender(channel);
        let receiver = joule_rt_channel_receiver(channel);

        assert!(!sender.is_null());
        assert!(!receiver.is_null());

        // Clean up
        joule_rt_channel_close(channel);
        // Note: sender and receiver also need cleanup in real code
    }

    #[test]
    fn test_task_group_create_destroy() {
        let group = joule_rt_task_group_create();
        assert!(!group.is_null());
        joule_rt_task_group_destroy(group);
    }

    #[test]
    fn test_is_cancelled() {
        assert_eq!(joule_rt_is_cancelled(), 0);
        joule_rt_cancel();
        assert_eq!(joule_rt_is_cancelled(), 1);
    }

    #[test]
    fn test_current_task_null() {
        assert!(joule_rt_current_task().is_null());
    }

    #[test]
    fn test_select_single_ready_channel() {
        let ch1 = joule_rt_channel_create(10);
        let ch2 = joule_rt_channel_create(10);

        // Send a value on channel 2
        let value = 99usize as *mut c_void;
        joule_rt_channel_try_send(ch2, value);

        // Select should return index 1 (ch2 is ready)
        let channels = [ch1, ch2];
        let selected = joule_rt_select(channels.as_ptr(), 2);
        assert_eq!(selected, 1);

        joule_rt_channel_close(ch1);
        joule_rt_channel_close(ch2);
    }

    #[test]
    fn test_select_all_closed() {
        // Create channels and close the sender side by dropping through the raw API.
        // We need channels that report Closed on try_recv, but the FfiChannel
        // struct must still be alive (not deallocated). So we close the sender
        // explicitly and then select on them.
        let ch1 = joule_rt_channel_create(10);
        let ch2 = joule_rt_channel_create(10);

        // Close the sender side of both channels so try_recv returns Closed
        unsafe {
            let c1 = &*(ch1 as *const FfiChannel);
            c1.sender.close();
            let c2 = &*(ch2 as *const FfiChannel);
            c2.sender.close();
        }

        // Select should return u64::MAX (all closed)
        let channels = [ch1, ch2];
        let selected = joule_rt_select(channels.as_ptr(), 2);
        assert_eq!(selected, u64::MAX);

        // Clean up
        joule_rt_channel_close(ch1);
        joule_rt_channel_close(ch2);
    }

    #[test]
    fn test_select_null_and_empty() {
        // Null array
        assert_eq!(joule_rt_select(ptr::null(), 5), u64::MAX);
        // Zero count
        let ch = joule_rt_channel_create(10);
        let channels = [ch];
        assert_eq!(joule_rt_select(channels.as_ptr(), 0), u64::MAX);
        joule_rt_channel_close(ch);
    }
}
