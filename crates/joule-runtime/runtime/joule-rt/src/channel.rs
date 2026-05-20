//! Bounded channel implementation for Joule concurrency
//!
//! This module provides bounded MPMC (multi-producer, multi-consumer) channels
//! that integrate with the Joule async executor. Channels provide backpressure
//! through bounded capacity.
//!
//! # Design
//!
//! - Bounded by default (prevents unbounded memory growth)
//! - Supports multiple senders and receivers
//! - Async send/recv that integrate with the executor
//! - Non-blocking try_send/try_recv for polling
//! - Graceful close semantics

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};

/// Error returned when trying to send on a closed channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendError<T>(pub T);

impl<T> std::fmt::Display for SendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "channel closed")
    }
}

impl<T: std::fmt::Debug> std::error::Error for SendError<T> {}

/// Error returned when trying to receive on a closed, empty channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvError;

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "channel closed")
    }
}

impl std::error::Error for RecvError {}

/// Result of a non-blocking try_send operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrySendError<T> {
    /// Channel is full.
    Full(T),
    /// Channel is closed.
    Closed(T),
}

impl<T> TrySendError<T> {
    /// Get the value that couldn't be sent.
    pub fn into_inner(self) -> T {
        match self {
            TrySendError::Full(t) | TrySendError::Closed(t) => t,
        }
    }
}

impl<T> std::fmt::Display for TrySendError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrySendError::Full(_) => write!(f, "channel full"),
            TrySendError::Closed(_) => write!(f, "channel closed"),
        }
    }
}

impl<T: std::fmt::Debug> std::error::Error for TrySendError<T> {}

/// Result of a non-blocking try_recv operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TryRecvError {
    /// Channel is empty.
    Empty,
    /// Channel is closed and empty.
    Closed,
}

impl std::fmt::Display for TryRecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TryRecvError::Empty => write!(f, "channel empty"),
            TryRecvError::Closed => write!(f, "channel closed"),
        }
    }
}

impl std::error::Error for TryRecvError {}

/// Shared channel state.
struct ChannelInner<T> {
    /// The bounded buffer.
    buffer: VecDeque<T>,
    /// Maximum capacity.
    capacity: usize,
    /// Whether the channel is closed.
    closed: bool,
    /// Wakers for blocked senders.
    send_wakers: Vec<Waker>,
    /// Wakers for blocked receivers.
    recv_wakers: Vec<Waker>,
}

impl<T> ChannelInner<T> {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            closed: false,
            send_wakers: Vec::new(),
            recv_wakers: Vec::new(),
        }
    }

    fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    fn wake_one_sender(&mut self) {
        if let Some(waker) = self.send_wakers.pop() {
            waker.wake();
        }
    }

    fn wake_one_receiver(&mut self) {
        if let Some(waker) = self.recv_wakers.pop() {
            waker.wake();
        }
    }

    fn wake_all_senders(&mut self) {
        for waker in self.send_wakers.drain(..) {
            waker.wake();
        }
    }

    fn wake_all_receivers(&mut self) {
        for waker in self.recv_wakers.drain(..) {
            waker.wake();
        }
    }
}

/// Shared channel handle.
struct Channel<T> {
    inner: Mutex<ChannelInner<T>>,
    /// Condition variable for blocking operations.
    condvar: Condvar,
    /// Number of senders.
    sender_count: AtomicUsize,
    /// Number of receivers.
    receiver_count: AtomicUsize,
    /// Fast path: closed flag for quick checks.
    closed: AtomicBool,
}

impl<T> Channel<T> {
    fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(ChannelInner::new(capacity)),
            condvar: Condvar::new(),
            sender_count: AtomicUsize::new(1),
            receiver_count: AtomicUsize::new(1),
            closed: AtomicBool::new(false),
        })
    }
}

/// The sending half of a channel.
///
/// Messages can be sent through this channel with [`send`] or [`try_send`].
///
/// [`send`]: Sender::send
/// [`try_send`]: Sender::try_send
pub struct Sender<T> {
    channel: Arc<Channel<T>>,
}

impl<T> Sender<T> {
    /// Send a value into the channel.
    ///
    /// If the channel is full, this method will wait until there is capacity.
    /// Returns an error if the channel is closed.
    pub fn send(&self, value: T) -> Send<'_, T> {
        Send {
            sender: self,
            value: Some(value),
        }
    }

    /// Try to send a value without blocking.
    ///
    /// Returns `Ok(())` if the value was sent, or an error if the channel
    /// is full or closed.
    pub fn try_send(&self, value: T) -> Result<(), TrySendError<T>> {
        if self.channel.closed.load(Ordering::Acquire) {
            return Err(TrySendError::Closed(value));
        }

        let mut inner = self.channel.inner.lock().unwrap();

        if inner.closed {
            return Err(TrySendError::Closed(value));
        }

        if inner.is_full() {
            return Err(TrySendError::Full(value));
        }

        inner.buffer.push_back(value);
        inner.wake_one_receiver();
        self.channel.condvar.notify_one();

        Ok(())
    }

    /// Close the channel.
    ///
    /// After closing, no more values can be sent. Receivers will continue
    /// to receive values until the buffer is empty.
    pub fn close(&self) {
        self.channel.closed.store(true, Ordering::Release);
        let mut inner = self.channel.inner.lock().unwrap();
        inner.closed = true;
        inner.wake_all_receivers();
        self.channel.condvar.notify_all();
    }

    /// Check if the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.channel.closed.load(Ordering::Acquire)
    }

    /// Get the capacity of the channel.
    pub fn capacity(&self) -> usize {
        self.channel.inner.lock().unwrap().capacity
    }

    /// Get the current number of items in the channel.
    pub fn len(&self) -> usize {
        self.channel.inner.lock().unwrap().buffer.len()
    }

    /// Check if the channel is empty.
    pub fn is_empty(&self) -> bool {
        self.channel.inner.lock().unwrap().is_empty()
    }

    /// Check if the channel is full.
    pub fn is_full(&self) -> bool {
        self.channel.inner.lock().unwrap().is_full()
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        self.channel.sender_count.fetch_add(1, Ordering::AcqRel);
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if self.channel.sender_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last sender dropped, close the channel
            self.close();
        }
    }
}

/// The receiving half of a channel.
///
/// Messages sent to the channel can be retrieved with [`recv`] or [`try_recv`].
///
/// [`recv`]: Receiver::recv
/// [`try_recv`]: Receiver::try_recv
pub struct Receiver<T> {
    channel: Arc<Channel<T>>,
}

impl<T> Receiver<T> {
    /// Receive a value from the channel.
    ///
    /// If the channel is empty, this method will wait until a value is available.
    /// Returns an error if the channel is closed and empty.
    pub fn recv(&self) -> Recv<'_, T> {
        Recv { receiver: self }
    }

    /// Try to receive a value without blocking.
    ///
    /// Returns `Ok(value)` if a value was received, or an error if the channel
    /// is empty or closed.
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        let mut inner = self.channel.inner.lock().unwrap();

        if let Some(value) = inner.buffer.pop_front() {
            inner.wake_one_sender();
            self.channel.condvar.notify_one();
            return Ok(value);
        }

        if inner.closed {
            Err(TryRecvError::Closed)
        } else {
            Err(TryRecvError::Empty)
        }
    }

    /// Register a waker that will be notified when new data arrives.
    ///
    /// Used by `ReceiverStream` for async notification when `try_recv` returns Empty.
    pub fn register_waker(&self, waker: std::task::Waker) {
        let mut inner = self.channel.inner.lock().unwrap();
        inner.recv_wakers.push(waker);
    }

    /// Close the channel.
    ///
    /// After closing, no more values can be sent. Any remaining values in the
    /// buffer can still be received.
    pub fn close(&self) {
        self.channel.closed.store(true, Ordering::Release);
        let mut inner = self.channel.inner.lock().unwrap();
        inner.closed = true;
        inner.wake_all_senders();
        self.channel.condvar.notify_all();
    }

    /// Check if the channel is closed.
    pub fn is_closed(&self) -> bool {
        self.channel.closed.load(Ordering::Acquire)
    }

    /// Get the capacity of the channel.
    pub fn capacity(&self) -> usize {
        self.channel.inner.lock().unwrap().capacity
    }

    /// Get the current number of items in the channel.
    pub fn len(&self) -> usize {
        self.channel.inner.lock().unwrap().buffer.len()
    }

    /// Check if the channel is empty.
    pub fn is_empty(&self) -> bool {
        self.channel.inner.lock().unwrap().is_empty()
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        self.channel.receiver_count.fetch_add(1, Ordering::AcqRel);
        Self {
            channel: self.channel.clone(),
        }
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        if self.channel.receiver_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last receiver dropped, close the channel
            self.close();
        }
    }
}

/// Future returned by [`Sender::send`].
pub struct Send<'a, T> {
    sender: &'a Sender<T>,
    value: Option<T>,
}

impl<T> Future for Send<'_, T> {
    type Output = Result<(), SendError<T>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We never move the inner data, we just modify the value field
        let this = unsafe { self.get_unchecked_mut() };

        let value = match this.value.take() {
            Some(v) => v,
            None => return Poll::Ready(Ok(())), // Already sent
        };

        // Fast path: check if closed
        if this.sender.channel.closed.load(Ordering::Acquire) {
            return Poll::Ready(Err(SendError(value)));
        }

        let mut inner = this.sender.channel.inner.lock().unwrap();

        if inner.closed {
            return Poll::Ready(Err(SendError(value)));
        }

        if !inner.is_full() {
            inner.buffer.push_back(value);
            inner.wake_one_receiver();
            this.sender.channel.condvar.notify_one();
            return Poll::Ready(Ok(()));
        }

        // Channel is full, register waker and wait
        this.value = Some(value);
        inner.send_wakers.push(cx.waker().clone());
        Poll::Pending
    }
}

/// Future returned by [`Receiver::recv`].
pub struct Recv<'a, T> {
    receiver: &'a Receiver<T>,
}

impl<T> Future for Recv<'_, T> {
    type Output = Result<T, RecvError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut inner = self.receiver.channel.inner.lock().unwrap();

        if let Some(value) = inner.buffer.pop_front() {
            inner.wake_one_sender();
            self.receiver.channel.condvar.notify_one();
            return Poll::Ready(Ok(value));
        }

        if inner.closed {
            return Poll::Ready(Err(RecvError));
        }

        // Channel is empty, register waker and wait
        inner.recv_wakers.push(cx.waker().clone());
        Poll::Pending
    }
}

/// Create a bounded channel with the specified capacity.
///
/// Returns a `(Sender, Receiver)` pair. Messages can be sent through the
/// sender and received through the receiver.
///
/// # Panics
///
/// Panics if capacity is 0.
///
/// # Example
///
/// ```
/// use joule_rt::channel::channel;
///
/// let (tx, rx) = channel::<i32>(10);
///
/// tx.try_send(42).unwrap();
/// assert_eq!(rx.try_recv().unwrap(), 42);
/// ```
pub fn channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "channel capacity must be greater than 0");

    let channel = Channel::new(capacity);
    let sender = Sender {
        channel: channel.clone(),
    };
    let receiver = Receiver { channel };

    (sender, receiver)
}

/// Create a bounded channel with capacity of 1 (rendezvous channel).
///
/// A rendezvous channel requires the sender and receiver to "meet" - the
/// sender blocks until the receiver is ready, and vice versa.
pub fn rendezvous<T>() -> (Sender<T>, Receiver<T>) {
    channel(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_creation() {
        let (tx, rx) = channel::<i32>(10);
        assert!(!tx.is_closed());
        assert!(!rx.is_closed());
        assert_eq!(tx.capacity(), 10);
        assert_eq!(rx.capacity(), 10);
        assert!(tx.is_empty());
        assert!(rx.is_empty());
    }

    #[test]
    fn test_try_send_recv() {
        let (tx, rx) = channel::<i32>(2);

        // Send values
        assert!(tx.try_send(1).is_ok());
        assert!(tx.try_send(2).is_ok());
        assert!(matches!(tx.try_send(3), Err(TrySendError::Full(3))));

        // Receive values
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn test_channel_close_sender() {
        let (tx, rx) = channel::<i32>(10);

        tx.try_send(42).unwrap();
        tx.close();

        assert!(tx.is_closed());
        assert!(rx.is_closed());

        // Can still receive buffered values
        assert_eq!(rx.try_recv().unwrap(), 42);

        // But then get closed error
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Closed)));
    }

    #[test]
    fn test_channel_close_receiver() {
        let (tx, rx) = channel::<i32>(10);

        rx.close();

        assert!(tx.is_closed());
        assert!(rx.is_closed());

        assert!(matches!(tx.try_send(42), Err(TrySendError::Closed(42))));
    }

    #[test]
    fn test_sender_drop_closes() {
        let (tx, rx) = channel::<i32>(10);

        tx.try_send(42).unwrap();
        drop(tx);

        assert!(rx.is_closed());
        assert_eq!(rx.try_recv().unwrap(), 42);
        assert!(matches!(rx.try_recv(), Err(TryRecvError::Closed)));
    }

    #[test]
    fn test_receiver_drop_closes() {
        let (tx, rx) = channel::<i32>(10);

        drop(rx);

        assert!(tx.is_closed());
        assert!(matches!(tx.try_send(42), Err(TrySendError::Closed(42))));
    }

    #[test]
    fn test_sender_clone() {
        let (tx, rx) = channel::<i32>(10);

        let tx2 = tx.clone();

        tx.try_send(1).unwrap();
        tx2.try_send(2).unwrap();

        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx.try_recv().unwrap(), 2);

        // Dropping one sender shouldn't close the channel
        drop(tx);
        assert!(!rx.is_closed());

        // Dropping all senders should close
        drop(tx2);
        assert!(rx.is_closed());
    }

    #[test]
    fn test_receiver_clone() {
        let (tx, rx) = channel::<i32>(10);

        let rx2 = rx.clone();

        tx.try_send(1).unwrap();
        tx.try_send(2).unwrap();

        // Both receivers can receive (MPMC)
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert_eq!(rx2.try_recv().unwrap(), 2);

        // Dropping one receiver shouldn't close the channel
        drop(rx);
        assert!(!tx.is_closed());

        // Dropping all receivers should close
        drop(rx2);
        assert!(tx.is_closed());
    }

    #[test]
    fn test_len_and_full() {
        let (tx, rx) = channel::<i32>(2);

        assert_eq!(tx.len(), 0);
        assert!(!tx.is_full());

        tx.try_send(1).unwrap();
        assert_eq!(tx.len(), 1);
        assert!(!tx.is_full());

        tx.try_send(2).unwrap();
        assert_eq!(tx.len(), 2);
        assert!(tx.is_full());

        rx.try_recv().unwrap();
        assert_eq!(tx.len(), 1);
        assert!(!tx.is_full());
    }

    #[test]
    fn test_rendezvous() {
        let (tx, rx) = rendezvous::<i32>();

        assert_eq!(tx.capacity(), 1);
        assert!(tx.try_send(1).is_ok());
        assert!(matches!(tx.try_send(2), Err(TrySendError::Full(2))));

        assert_eq!(rx.try_recv().unwrap(), 1);
    }

    #[test]
    #[should_panic(expected = "channel capacity must be greater than 0")]
    fn test_zero_capacity_panics() {
        channel::<i32>(0);
    }

    #[test]
    fn test_send_error_display() {
        let err = SendError(42);
        assert_eq!(format!("{}", err), "channel closed");
    }

    #[test]
    fn test_recv_error_display() {
        let err = RecvError;
        assert_eq!(format!("{}", err), "channel closed");
    }

    #[test]
    fn test_try_send_error_into_inner() {
        let err = TrySendError::Full(42);
        assert_eq!(err.into_inner(), 42);

        let err = TrySendError::Closed(42);
        assert_eq!(err.into_inner(), 42);
    }
}
