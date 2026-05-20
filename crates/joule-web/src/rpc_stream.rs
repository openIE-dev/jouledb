//! Streaming RPC support — bidirectional streams with flow control.
//!
//! Provides [`StreamMessage`] for framing individual stream data, [`StreamManager`]
//! for tracking open streams, flow control via receive windows, bidirectional
//! stream support, cancellation, and per-stream statistics. Max concurrent
//! streams can be limited.

use std::collections::HashMap;
use std::fmt;

// ── Stream State ───────────────────────────────────────────────

/// The lifecycle state of a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamState {
    /// Stream is open and accepting messages.
    Open,
    /// Local side has sent final message, waiting for remote.
    HalfClosedLocal,
    /// Remote side has sent final message, local can still send.
    HalfClosedRemote,
    /// Stream is fully closed.
    Closed,
    /// Stream was cancelled.
    Cancelled,
}

impl fmt::Display for StreamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => write!(f, "Open"),
            Self::HalfClosedLocal => write!(f, "HalfClosedLocal"),
            Self::HalfClosedRemote => write!(f, "HalfClosedRemote"),
            Self::Closed => write!(f, "Closed"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

// ── Stream Message ─────────────────────────────────────────────

/// A single message within an RPC stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamMessage {
    pub stream_id: u64,
    pub sequence: u64,
    pub payload: Vec<u8>,
    pub is_final: bool,
}

impl StreamMessage {
    pub fn new(stream_id: u64, sequence: u64, payload: Vec<u8>) -> Self {
        Self { stream_id, sequence, payload, is_final: false }
    }

    pub fn final_msg(stream_id: u64, sequence: u64, payload: Vec<u8>) -> Self {
        Self { stream_id, sequence, payload, is_final: true }
    }

    /// Payload size in bytes.
    pub fn payload_size(&self) -> usize { self.payload.len() }
}

impl fmt::Display for StreamMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stream[id={}, seq={}, {}B{}]",
            self.stream_id, self.sequence, self.payload.len(),
            if self.is_final { ", FINAL" } else { "" })
    }
}

// ── Stream Direction ───────────────────────────────────────────

/// Whether a stream is unidirectional or bidirectional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDirection {
    ClientToServer,
    ServerToClient,
    Bidirectional,
}

impl fmt::Display for StreamDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClientToServer => write!(f, "C->S"),
            Self::ServerToClient => write!(f, "S->C"),
            Self::Bidirectional => write!(f, "BiDi"),
        }
    }
}

// ── Stream Error ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    StreamNotFound(u64),
    StreamClosed(u64),
    StreamCancelled(u64),
    MaxStreamsExceeded { limit: usize },
    WindowExhausted { stream_id: u64, window: u64 },
    SequenceGap { stream_id: u64, expected: u64, got: u64 },
    InvalidState { stream_id: u64, state: StreamState, action: String },
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamNotFound(id) => write!(f, "stream {id} not found"),
            Self::StreamClosed(id) => write!(f, "stream {id} is closed"),
            Self::StreamCancelled(id) => write!(f, "stream {id} was cancelled"),
            Self::MaxStreamsExceeded { limit } =>
                write!(f, "max concurrent streams exceeded ({limit})"),
            Self::WindowExhausted { stream_id, window } =>
                write!(f, "stream {stream_id} flow control window exhausted ({window})"),
            Self::SequenceGap { stream_id, expected, got } =>
                write!(f, "stream {stream_id} sequence gap: expected {expected}, got {got}"),
            Self::InvalidState { stream_id, state, action } =>
                write!(f, "stream {stream_id} invalid state {state} for {action}"),
        }
    }
}

// ── Stream Statistics ──────────────────────────────────────────

/// Per-stream statistics.
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

impl StreamStats {
    pub fn total_messages(&self) -> u64 { self.messages_sent + self.messages_received }
    pub fn total_bytes(&self) -> u64 { self.bytes_sent + self.bytes_received }
}

// ── Stream Entry ───────────────────────────────────────────────

/// Internal tracking structure for a single stream.
#[derive(Debug, Clone)]
struct StreamEntry {
    state: StreamState,
    direction: StreamDirection,
    next_send_seq: u64,
    next_recv_seq: u64,
    send_window: u64,
    recv_window: u64,
    initial_window: u64,
    stats: StreamStats,
    created_at_ms: u64,
}

// ── Stream Manager ─────────────────────────────────────────────

/// Manages multiple concurrent RPC streams with flow control.
#[derive(Debug)]
pub struct StreamManager {
    streams: HashMap<u64, StreamEntry>,
    next_stream_id: u64,
    max_concurrent: usize,
    default_window: u64,
    total_streams_created: u64,
    total_streams_completed: u64,
    total_streams_cancelled: u64,
}

impl StreamManager {
    pub fn new(max_concurrent: usize, default_window: u64) -> Self {
        Self {
            streams: HashMap::new(),
            next_stream_id: 1,
            max_concurrent,
            default_window,
            total_streams_created: 0,
            total_streams_completed: 0,
            total_streams_cancelled: 0,
        }
    }

    /// Open a new stream. Returns the stream ID.
    pub fn open(&mut self, direction: StreamDirection, now_ms: u64) -> Result<u64, StreamError> {
        let active = self.active_count();
        if active >= self.max_concurrent {
            return Err(StreamError::MaxStreamsExceeded { limit: self.max_concurrent });
        }
        let id = self.next_stream_id;
        self.next_stream_id += 1;
        self.streams.insert(id, StreamEntry {
            state: StreamState::Open,
            direction,
            next_send_seq: 0,
            next_recv_seq: 0,
            send_window: self.default_window,
            recv_window: self.default_window,
            initial_window: self.default_window,
            stats: StreamStats::default(),
            created_at_ms: now_ms,
        });
        self.total_streams_created += 1;
        Ok(id)
    }

    /// Send a message on a stream. Returns the produced [`StreamMessage`].
    pub fn send(&mut self, stream_id: u64, payload: Vec<u8>, is_final: bool) -> Result<StreamMessage, StreamError> {
        let entry = self.streams.get_mut(&stream_id)
            .ok_or(StreamError::StreamNotFound(stream_id))?;
        match entry.state {
            StreamState::Open | StreamState::HalfClosedRemote => {}
            StreamState::HalfClosedLocal | StreamState::Closed =>
                return Err(StreamError::StreamClosed(stream_id)),
            StreamState::Cancelled =>
                return Err(StreamError::StreamCancelled(stream_id)),
        }
        let payload_len = payload.len() as u64;
        if payload_len > entry.send_window {
            return Err(StreamError::WindowExhausted {
                stream_id,
                window: entry.send_window,
            });
        }
        entry.send_window -= payload_len;
        let seq = entry.next_send_seq;
        entry.next_send_seq += 1;
        entry.stats.messages_sent += 1;
        entry.stats.bytes_sent += payload_len;
        if is_final {
            match entry.state {
                StreamState::Open => entry.state = StreamState::HalfClosedLocal,
                StreamState::HalfClosedRemote => entry.state = StreamState::Closed,
                _ => {}
            }
            if entry.state == StreamState::Closed {
                self.total_streams_completed += 1;
            }
        }
        Ok(if is_final {
            StreamMessage::final_msg(stream_id, seq, payload)
        } else {
            StreamMessage::new(stream_id, seq, payload)
        })
    }

    /// Receive (process) an incoming stream message.
    pub fn receive(&mut self, msg: &StreamMessage) -> Result<(), StreamError> {
        let entry = self.streams.get_mut(&msg.stream_id)
            .ok_or(StreamError::StreamNotFound(msg.stream_id))?;
        match entry.state {
            StreamState::Open | StreamState::HalfClosedLocal => {}
            StreamState::HalfClosedRemote | StreamState::Closed =>
                return Err(StreamError::StreamClosed(msg.stream_id)),
            StreamState::Cancelled =>
                return Err(StreamError::StreamCancelled(msg.stream_id)),
        }
        if msg.sequence != entry.next_recv_seq {
            return Err(StreamError::SequenceGap {
                stream_id: msg.stream_id,
                expected: entry.next_recv_seq,
                got: msg.sequence,
            });
        }
        entry.next_recv_seq += 1;
        entry.recv_window = entry.recv_window.saturating_sub(msg.payload.len() as u64);
        entry.stats.messages_received += 1;
        entry.stats.bytes_received += msg.payload.len() as u64;
        if msg.is_final {
            match entry.state {
                StreamState::Open => entry.state = StreamState::HalfClosedRemote,
                StreamState::HalfClosedLocal => entry.state = StreamState::Closed,
                _ => {}
            }
            if entry.state == StreamState::Closed {
                self.total_streams_completed += 1;
            }
        }
        Ok(())
    }

    /// Cancel a stream.
    pub fn cancel(&mut self, stream_id: u64) -> Result<(), StreamError> {
        let entry = self.streams.get_mut(&stream_id)
            .ok_or(StreamError::StreamNotFound(stream_id))?;
        if entry.state == StreamState::Closed || entry.state == StreamState::Cancelled {
            return Err(StreamError::StreamClosed(stream_id));
        }
        entry.state = StreamState::Cancelled;
        self.total_streams_cancelled += 1;
        Ok(())
    }

    /// Replenish the send window for a stream (e.g., after receiving a WINDOW_UPDATE).
    pub fn update_send_window(&mut self, stream_id: u64, increment: u64) -> Result<(), StreamError> {
        let entry = self.streams.get_mut(&stream_id)
            .ok_or(StreamError::StreamNotFound(stream_id))?;
        entry.send_window += increment;
        Ok(())
    }

    /// Get the state of a stream.
    pub fn state(&self, stream_id: u64) -> Option<StreamState> {
        self.streams.get(&stream_id).map(|e| e.state)
    }

    /// Get statistics for a stream.
    pub fn stats(&self, stream_id: u64) -> Option<&StreamStats> {
        self.streams.get(&stream_id).map(|e| &e.stats)
    }

    /// Number of active (non-closed, non-cancelled) streams.
    pub fn active_count(&self) -> usize {
        self.streams.values().filter(|e| {
            e.state != StreamState::Closed && e.state != StreamState::Cancelled
        }).count()
    }

    /// Total streams ever created.
    pub fn total_created(&self) -> u64 { self.total_streams_created }
    pub fn total_completed(&self) -> u64 { self.total_streams_completed }
    pub fn total_cancelled(&self) -> u64 { self.total_streams_cancelled }

    /// List all active stream IDs.
    pub fn active_stream_ids(&self) -> Vec<u64> {
        self.streams.iter()
            .filter(|(_, e)| e.state != StreamState::Closed && e.state != StreamState::Cancelled)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Remove all closed/cancelled streams from tracking.
    pub fn gc(&mut self) -> usize {
        let before = self.streams.len();
        self.streams.retain(|_, e| {
            e.state != StreamState::Closed && e.state != StreamState::Cancelled
        });
        before - self.streams.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_stream() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        assert_eq!(id, 1);
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.state(id), Some(StreamState::Open));
    }

    #[test]
    fn max_concurrent_streams_enforced() {
        let mut mgr = StreamManager::new(2, 1024);
        mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        assert!(matches!(
            mgr.open(StreamDirection::Bidirectional, 0),
            Err(StreamError::MaxStreamsExceeded { .. })
        ));
    }

    #[test]
    fn send_produces_message() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        let msg = mgr.send(id, vec![1, 2, 3], false).unwrap();
        assert_eq!(msg.stream_id, id);
        assert_eq!(msg.sequence, 0);
        assert_eq!(msg.payload, vec![1, 2, 3]);
        assert!(!msg.is_final);
    }

    #[test]
    fn send_increments_sequence() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.send(id, vec![1], false).unwrap();
        let msg = mgr.send(id, vec![2], false).unwrap();
        assert_eq!(msg.sequence, 1);
    }

    #[test]
    fn send_final_closes_local() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.send(id, vec![1], true).unwrap();
        assert_eq!(mgr.state(id), Some(StreamState::HalfClosedLocal));
    }

    #[test]
    fn receive_final_closes_remote() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        let msg = StreamMessage::final_msg(id, 0, vec![1]);
        mgr.receive(&msg).unwrap();
        assert_eq!(mgr.state(id), Some(StreamState::HalfClosedRemote));
    }

    #[test]
    fn both_sides_final_fully_closes() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.send(id, vec![1], true).unwrap(); // half-closed local
        let msg = StreamMessage::final_msg(id, 0, vec![2]);
        mgr.receive(&msg).unwrap();
        assert_eq!(mgr.state(id), Some(StreamState::Closed));
        assert_eq!(mgr.total_completed(), 1);
    }

    #[test]
    fn cancel_stream() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.cancel(id).unwrap();
        assert_eq!(mgr.state(id), Some(StreamState::Cancelled));
        assert_eq!(mgr.total_cancelled(), 1);
    }

    #[test]
    fn send_on_cancelled_fails() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.cancel(id).unwrap();
        assert!(matches!(
            mgr.send(id, vec![1], false),
            Err(StreamError::StreamCancelled(_))
        ));
    }

    #[test]
    fn flow_control_window_exhaustion() {
        let mut mgr = StreamManager::new(10, 10); // window=10 bytes
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.send(id, vec![0; 8], false).unwrap();
        assert!(matches!(
            mgr.send(id, vec![0; 5], false),
            Err(StreamError::WindowExhausted { .. })
        ));
    }

    #[test]
    fn window_update_replenishes() {
        let mut mgr = StreamManager::new(10, 10);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.send(id, vec![0; 10], false).unwrap();
        mgr.update_send_window(id, 20).unwrap();
        mgr.send(id, vec![0; 15], false).unwrap(); // now has window
    }

    #[test]
    fn sequence_gap_detected() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        let msg = StreamMessage::new(id, 5, vec![1]); // expect seq 0
        assert!(matches!(
            mgr.receive(&msg),
            Err(StreamError::SequenceGap { .. })
        ));
    }

    #[test]
    fn stream_stats_tracking() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.send(id, vec![0; 100], false).unwrap();
        let msg = StreamMessage::new(id, 0, vec![0; 50]);
        mgr.receive(&msg).unwrap();
        let stats = mgr.stats(id).unwrap();
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.bytes_sent, 100);
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.bytes_received, 50);
        assert_eq!(stats.total_messages(), 2);
        assert_eq!(stats.total_bytes(), 150);
    }

    #[test]
    fn gc_removes_closed_streams() {
        let mut mgr = StreamManager::new(10, 65536);
        let id = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.cancel(id).unwrap();
        let removed = mgr.gc();
        assert_eq!(removed, 1);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn active_stream_ids_filters() {
        let mut mgr = StreamManager::new(10, 65536);
        let id1 = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        let id2 = mgr.open(StreamDirection::Bidirectional, 0).unwrap();
        mgr.cancel(id1).unwrap();
        let ids = mgr.active_stream_ids();
        assert_eq!(ids, vec![id2]);
    }

    #[test]
    fn stream_message_display() {
        let msg = StreamMessage::final_msg(5, 10, vec![0; 20]);
        let s = format!("{msg}");
        assert!(s.contains("id=5"));
        assert!(s.contains("seq=10"));
        assert!(s.contains("FINAL"));
    }

    #[test]
    fn total_created_counter() {
        let mut mgr = StreamManager::new(10, 65536);
        mgr.open(StreamDirection::ClientToServer, 0).unwrap();
        mgr.open(StreamDirection::ServerToClient, 0).unwrap();
        assert_eq!(mgr.total_created(), 2);
    }

    #[test]
    fn send_on_nonexistent_stream() {
        let mut mgr = StreamManager::new(10, 65536);
        assert!(matches!(
            mgr.send(999, vec![], false),
            Err(StreamError::StreamNotFound(999))
        ));
    }
}
