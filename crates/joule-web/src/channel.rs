//! Message channels — MPSC, bounded, oneshot, and broadcast channels.
//!
//! Pure Rust channel implementations for synchronous message passing.
//! No external dependencies — models channel semantics including
//! backpressure, closing, and ordering guarantees.

use std::collections::VecDeque;

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SendError<T> {
    /// Channel is closed.
    Closed(T),
    /// Bounded channel is full.
    Full(T),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecvError {
    /// Channel is empty.
    Empty,
    /// Channel is closed and empty.
    Closed,
}

// ── Unbounded MPSC Channel ─────────────────────────────────────

/// Unbounded multi-producer single-consumer channel.
#[derive(Debug)]
pub struct UnboundedChannel<T> {
    buffer: VecDeque<T>,
    closed: bool,
    send_count: u64,
    recv_count: u64,
}

impl<T> UnboundedChannel<T> {
    pub fn new() -> Self {
        Self {
            buffer: VecDeque::new(),
            closed: false,
            send_count: 0,
            recv_count: 0,
        }
    }

    pub fn send(&mut self, msg: T) -> Result<(), SendError<T>> {
        if self.closed {
            return Err(SendError::Closed(msg));
        }
        self.buffer.push_back(msg);
        self.send_count += 1;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<T, RecvError> {
        match self.buffer.pop_front() {
            Some(msg) => {
                self.recv_count += 1;
                Ok(msg)
            }
            None if self.closed => Err(RecvError::Closed),
            None => Err(RecvError::Empty),
        }
    }

    pub fn try_recv(&mut self) -> Option<T> {
        let msg = self.buffer.pop_front()?;
        self.recv_count += 1;
        Some(msg)
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn send_count(&self) -> u64 {
        self.send_count
    }

    pub fn recv_count(&self) -> u64 {
        self.recv_count
    }
}

impl<T> Default for UnboundedChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Bounded Channel ────────────────────────────────────────────

/// Bounded channel with backpressure (try_send returns Full when at capacity).
#[derive(Debug)]
pub struct BoundedChannel<T> {
    buffer: VecDeque<T>,
    capacity: usize,
    closed: bool,
    send_count: u64,
    recv_count: u64,
}

impl<T> BoundedChannel<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            closed: false,
            send_count: 0,
            recv_count: 0,
        }
    }

    pub fn send(&mut self, msg: T) -> Result<(), SendError<T>> {
        if self.closed {
            return Err(SendError::Closed(msg));
        }
        if self.buffer.len() >= self.capacity {
            return Err(SendError::Full(msg));
        }
        self.buffer.push_back(msg);
        self.send_count += 1;
        Ok(())
    }

    pub fn try_send(&mut self, msg: T) -> Result<(), SendError<T>> {
        self.send(msg)
    }

    pub fn recv(&mut self) -> Result<T, RecvError> {
        match self.buffer.pop_front() {
            Some(msg) => {
                self.recv_count += 1;
                Ok(msg)
            }
            None if self.closed => Err(RecvError::Closed),
            None => Err(RecvError::Empty),
        }
    }

    pub fn try_recv(&mut self) -> Option<T> {
        let msg = self.buffer.pop_front()?;
        self.recv_count += 1;
        Some(msg)
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    pub fn send_count(&self) -> u64 {
        self.send_count
    }

    pub fn recv_count(&self) -> u64 {
        self.recv_count
    }
}

// ── Oneshot Channel ────────────────────────────────────────────

/// Single-use channel: one send, one recv.
#[derive(Debug)]
pub struct OneshotChannel<T> {
    value: Option<T>,
    sent: bool,
    received: bool,
    closed: bool,
}

impl<T> OneshotChannel<T> {
    pub fn new() -> Self {
        Self {
            value: None,
            sent: false,
            received: false,
            closed: false,
        }
    }

    pub fn send(&mut self, msg: T) -> Result<(), SendError<T>> {
        if self.closed {
            return Err(SendError::Closed(msg));
        }
        if self.sent {
            return Err(SendError::Full(msg));
        }
        self.value = Some(msg);
        self.sent = true;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<T, RecvError> {
        if self.received {
            return Err(RecvError::Closed);
        }
        match self.value.take() {
            Some(v) => {
                self.received = true;
                Ok(v)
            }
            None if self.closed => Err(RecvError::Closed),
            None => Err(RecvError::Empty),
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_sent(&self) -> bool {
        self.sent
    }

    pub fn is_received(&self) -> bool {
        self.received
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl<T> Default for OneshotChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Broadcast Channel ──────────────────────────────────────────

/// One-to-many broadcast channel. Each subscriber has its own read cursor.
#[derive(Debug)]
pub struct BroadcastChannel<T: Clone> {
    /// Ring buffer of messages.
    messages: Vec<T>,
    /// Total messages ever sent (monotonic).
    total_sent: u64,
    /// Subscriber read cursors: subscriber_id → next message index to read.
    subscribers: std::collections::HashMap<u64, u64>,
    next_sub_id: u64,
    closed: bool,
}

impl<T: Clone> BroadcastChannel<T> {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            total_sent: 0,
            subscribers: std::collections::HashMap::new(),
            next_sub_id: 1,
            closed: false,
        }
    }

    /// Subscribe and get a subscriber id.
    pub fn subscribe(&mut self) -> u64 {
        let id = self.next_sub_id;
        self.next_sub_id += 1;
        self.subscribers.insert(id, self.total_sent);
        id
    }

    /// Unsubscribe.
    pub fn unsubscribe(&mut self, id: u64) -> bool {
        self.subscribers.remove(&id).is_some()
    }

    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Send a message to all subscribers.
    pub fn send(&mut self, msg: T) -> Result<(), SendError<T>> {
        if self.closed {
            return Err(SendError::Closed(msg));
        }
        self.messages.push(msg);
        self.total_sent += 1;
        Ok(())
    }

    /// Receive the next message for a subscriber.
    pub fn recv(&mut self, sub_id: u64) -> Result<T, RecvError> {
        let cursor = match self.subscribers.get_mut(&sub_id) {
            Some(c) => c,
            None => return Err(RecvError::Closed),
        };
        let idx = *cursor as usize;
        if idx < self.messages.len() {
            let msg = self.messages[idx].clone();
            *cursor += 1;
            Ok(msg)
        } else if self.closed {
            Err(RecvError::Closed)
        } else {
            Err(RecvError::Empty)
        }
    }

    /// Number of pending messages for a subscriber.
    pub fn pending_for(&self, sub_id: u64) -> usize {
        self.subscribers
            .get(&sub_id)
            .map(|cursor| self.messages.len().saturating_sub(*cursor as usize))
            .unwrap_or(0)
    }

    pub fn close(&mut self) {
        self.closed = true;
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn total_sent(&self) -> u64 {
        self.total_sent
    }
}

impl<T: Clone> Default for BroadcastChannel<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unbounded_send_recv() {
        let mut ch = UnboundedChannel::new();
        ch.send(1).unwrap();
        ch.send(2).unwrap();
        assert_eq!(ch.recv().unwrap(), 1);
        assert_eq!(ch.recv().unwrap(), 2);
        assert_eq!(ch.recv(), Err(RecvError::Empty));
    }

    #[test]
    fn test_unbounded_ordering() {
        let mut ch = UnboundedChannel::new();
        for i in 0..100 {
            ch.send(i).unwrap();
        }
        for i in 0..100 {
            assert_eq!(ch.recv().unwrap(), i);
        }
    }

    #[test]
    fn test_unbounded_close() {
        let mut ch = UnboundedChannel::new();
        ch.send(1).unwrap();
        ch.close();
        assert!(matches!(ch.send(2), Err(SendError::Closed(2))));
        assert_eq!(ch.recv().unwrap(), 1); // Buffered message still available.
        assert_eq!(ch.recv(), Err(RecvError::Closed));
    }

    #[test]
    fn test_bounded_backpressure() {
        let mut ch = BoundedChannel::new(2);
        ch.send("a").unwrap();
        ch.send("b").unwrap();
        assert!(ch.is_full());
        assert!(matches!(ch.send("c"), Err(SendError::Full("c"))));
        ch.recv().unwrap();
        ch.send("c").unwrap(); // Now there's room.
    }

    #[test]
    fn test_bounded_try_send() {
        let mut ch = BoundedChannel::new(1);
        assert!(ch.try_send(1).is_ok());
        assert!(ch.try_send(2).is_err());
    }

    #[test]
    fn test_oneshot_channel() {
        let mut ch = OneshotChannel::new();
        ch.send(42).unwrap();
        assert!(ch.is_sent());
        // Second send fails.
        assert!(matches!(ch.send(99), Err(SendError::Full(99))));
        assert_eq!(ch.recv().unwrap(), 42);
        assert!(ch.is_received());
        // Second recv fails.
        assert_eq!(ch.recv(), Err(RecvError::Closed));
    }

    #[test]
    fn test_oneshot_close_before_send() {
        let mut ch: OneshotChannel<i32> = OneshotChannel::new();
        ch.close();
        assert!(matches!(ch.send(1), Err(SendError::Closed(1))));
    }

    #[test]
    fn test_broadcast_multi_subscriber() {
        let mut ch = BroadcastChannel::new();
        let sub1 = ch.subscribe();
        let sub2 = ch.subscribe();
        ch.send("hello").unwrap();
        ch.send("world").unwrap();
        assert_eq!(ch.recv(sub1).unwrap(), "hello");
        assert_eq!(ch.recv(sub1).unwrap(), "world");
        assert_eq!(ch.recv(sub2).unwrap(), "hello");
        assert_eq!(ch.recv(sub2).unwrap(), "world");
    }

    #[test]
    fn test_broadcast_late_subscriber() {
        let mut ch = BroadcastChannel::new();
        ch.send("before").unwrap();
        let sub = ch.subscribe();
        ch.send("after").unwrap();
        // Late subscriber only sees messages after subscribing.
        assert_eq!(ch.recv(sub).unwrap(), "after");
    }

    #[test]
    fn test_broadcast_pending() {
        let mut ch = BroadcastChannel::new();
        let sub = ch.subscribe();
        ch.send(1).unwrap();
        ch.send(2).unwrap();
        assert_eq!(ch.pending_for(sub), 2);
        ch.recv(sub).unwrap();
        assert_eq!(ch.pending_for(sub), 1);
    }

    #[test]
    fn test_broadcast_unsubscribe() {
        let mut ch: BroadcastChannel<i32> = BroadcastChannel::new();
        let sub = ch.subscribe();
        assert_eq!(ch.subscriber_count(), 1);
        ch.unsubscribe(sub);
        assert_eq!(ch.subscriber_count(), 0);
        assert_eq!(ch.recv(sub), Err(RecvError::Closed));
    }

    #[test]
    fn test_send_recv_counts() {
        let mut ch = UnboundedChannel::new();
        ch.send(1).unwrap();
        ch.send(2).unwrap();
        ch.recv().unwrap();
        assert_eq!(ch.send_count(), 2);
        assert_eq!(ch.recv_count(), 1);
    }
}
