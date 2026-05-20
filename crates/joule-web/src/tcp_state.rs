//! TCP state machine simulation — connection states, transitions, sequence tracking.
//!
//! Pure Rust implementation of the TCP state machine (RFC 793). Models all 11
//! connection states, validates transitions on events, tracks sequence/ack
//! numbers, manages send/receive windows, and handles RST and timeouts.

use std::fmt;

// ── TCP States ────────────────────────────────────────────────

/// The 11 states of a TCP connection per RFC 793.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TcpState {
    Closed,
    Listen,
    SynSent,
    SynReceived,
    Established,
    FinWait1,
    FinWait2,
    CloseWait,
    Closing,
    LastAck,
    TimeWait,
}

impl fmt::Display for TcpState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Closed => "CLOSED",
            Self::Listen => "LISTEN",
            Self::SynSent => "SYN_SENT",
            Self::SynReceived => "SYN_RECEIVED",
            Self::Established => "ESTABLISHED",
            Self::FinWait1 => "FIN_WAIT_1",
            Self::FinWait2 => "FIN_WAIT_2",
            Self::CloseWait => "CLOSE_WAIT",
            Self::Closing => "CLOSING",
            Self::LastAck => "LAST_ACK",
            Self::TimeWait => "TIME_WAIT",
        };
        f.write_str(s)
    }
}

// ── Events ────────────────────────────────────────────────────

/// Events that drive TCP state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpEvent {
    /// Application calls OPEN (passive).
    PassiveOpen,
    /// Application calls OPEN (active).
    ActiveOpen,
    /// Received SYN segment.
    RecvSyn,
    /// Received SYN+ACK segment.
    RecvSynAck,
    /// Received ACK segment.
    RecvAck,
    /// Received FIN segment.
    RecvFin,
    /// Received FIN+ACK segment.
    RecvFinAck,
    /// Received RST segment.
    RecvRst,
    /// Application calls SEND (triggers SYN for passive open).
    Send,
    /// Application calls CLOSE.
    Close,
    /// Timeout expired (retransmission or TIME_WAIT).
    Timeout,
}

// ── Transition Error ──────────────────────────────────────────

/// Error when a transition is invalid for the current state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionError {
    pub state: TcpState,
    pub event: TcpEvent,
}

impl fmt::Display for TransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid event {:?} in state {}", self.event, self.state)
    }
}

// ── Segment Flags ─────────────────────────────────────────────

/// TCP segment flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TcpFlags {
    pub syn: bool,
    pub ack: bool,
    pub fin: bool,
    pub rst: bool,
    pub psh: bool,
    pub urg: bool,
}

impl TcpFlags {
    pub fn syn() -> Self {
        Self { syn: true, ..Default::default() }
    }

    pub fn syn_ack() -> Self {
        Self { syn: true, ack: true, ..Default::default() }
    }

    pub fn ack() -> Self {
        Self { ack: true, ..Default::default() }
    }

    pub fn fin() -> Self {
        Self { fin: true, ..Default::default() }
    }

    pub fn fin_ack() -> Self {
        Self { fin: true, ack: true, ..Default::default() }
    }

    pub fn rst() -> Self {
        Self { rst: true, ..Default::default() }
    }
}

// ── Segment ───────────────────────────────────────────────────

/// A simplified TCP segment for simulation purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcpSegment {
    pub src_port: u16,
    pub dst_port: u16,
    pub seq_num: u32,
    pub ack_num: u32,
    pub flags: TcpFlags,
    pub window: u16,
    pub payload_len: u32,
}

impl TcpSegment {
    /// Compute the sequence length consumed by this segment.
    /// SYN and FIN each consume one sequence number.
    pub fn seq_len(&self) -> u32 {
        let mut len = self.payload_len;
        if self.flags.syn {
            len = len.saturating_add(1);
        }
        if self.flags.fin {
            len = len.saturating_add(1);
        }
        len
    }
}

// ── Window Tracker ────────────────────────────────────────────

/// Tracks send/receive windows for flow control.
#[derive(Debug, Clone)]
pub struct WindowTracker {
    /// Send window size (advertised by remote).
    pub send_window: u32,
    /// Receive window size (our capacity).
    pub recv_window: u32,
    /// Bytes sent but not yet acknowledged.
    pub bytes_in_flight: u32,
}

impl WindowTracker {
    pub fn new(send_window: u32, recv_window: u32) -> Self {
        Self {
            send_window,
            recv_window,
            bytes_in_flight: 0,
        }
    }

    /// How many bytes we can still send.
    pub fn available_send(&self) -> u32 {
        self.send_window.saturating_sub(self.bytes_in_flight)
    }

    /// Record sending `n` bytes.
    pub fn record_send(&mut self, n: u32) -> bool {
        if n > self.available_send() {
            return false;
        }
        self.bytes_in_flight = self.bytes_in_flight.saturating_add(n);
        true
    }

    /// Record acknowledgement of `n` bytes.
    pub fn record_ack(&mut self, n: u32) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(n);
    }

    /// Update send window from remote advertisement.
    pub fn update_send_window(&mut self, new_window: u32) {
        self.send_window = new_window;
    }

    /// Update receive window (our capacity).
    pub fn update_recv_window(&mut self, new_window: u32) {
        self.recv_window = new_window;
    }

    /// Consume `n` bytes from receive window.
    pub fn consume_recv(&mut self, n: u32) -> bool {
        if n > self.recv_window {
            return false;
        }
        self.recv_window = self.recv_window.saturating_sub(n);
        true
    }

    /// Free `n` bytes in receive window (application consumed data).
    pub fn free_recv(&mut self, n: u32) {
        self.recv_window = self.recv_window.saturating_add(n);
    }
}

// ── Timeout Tracker ───────────────────────────────────────────

/// Tracks connection-level timeouts.
#[derive(Debug, Clone)]
pub struct TimeoutTracker {
    /// Retransmission timeout in milliseconds.
    pub rto_ms: u64,
    /// Current retransmission count.
    pub retransmit_count: u32,
    /// Maximum retransmissions before abort.
    pub max_retransmits: u32,
    /// TIME_WAIT duration in milliseconds (2*MSL).
    pub time_wait_ms: u64,
    /// Connection idle timeout in milliseconds.
    pub idle_timeout_ms: u64,
    /// Timestamp when last activity occurred (ms since epoch).
    pub last_activity_ms: u64,
}

impl TimeoutTracker {
    pub fn new() -> Self {
        Self {
            rto_ms: 1000,
            retransmit_count: 0,
            max_retransmits: 5,
            time_wait_ms: 60_000,
            idle_timeout_ms: 120_000,
            last_activity_ms: 0,
        }
    }

    /// Record activity at the given timestamp.
    pub fn touch(&mut self, now_ms: u64) {
        self.last_activity_ms = now_ms;
        self.retransmit_count = 0;
    }

    /// Check if retransmission is allowed and bump counter.
    pub fn retransmit(&mut self) -> bool {
        if self.retransmit_count >= self.max_retransmits {
            return false;
        }
        self.retransmit_count += 1;
        // Exponential backoff: double RTO each time, cap at 60s.
        self.rto_ms = (self.rto_ms.saturating_mul(2)).min(60_000);
        true
    }

    /// Check if idle timeout has expired.
    pub fn is_idle_expired(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.last_activity_ms) >= self.idle_timeout_ms
    }

    /// Reset retransmission state.
    pub fn reset_rto(&mut self) {
        self.rto_ms = 1000;
        self.retransmit_count = 0;
    }
}

impl Default for TimeoutTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Connection State Machine ──────────────────────────────────

/// A TCP connection state machine with sequence tracking and windows.
#[derive(Debug, Clone)]
pub struct TcpConnection {
    pub state: TcpState,
    pub local_port: u16,
    pub remote_port: u16,
    /// Our next sequence number to send.
    pub send_next: u32,
    /// Our initial sequence number.
    pub send_isn: u32,
    /// Highest sequence number acknowledged by remote.
    pub send_una: u32,
    /// Remote's next expected sequence number.
    pub recv_next: u32,
    /// Remote's initial sequence number.
    pub recv_isn: u32,
    pub window: WindowTracker,
    pub timeout: TimeoutTracker,
    /// Transition history for debugging.
    history: Vec<(TcpState, TcpEvent, TcpState)>,
}

impl TcpConnection {
    /// Create a new connection in CLOSED state.
    pub fn new(local_port: u16, remote_port: u16, isn: u32) -> Self {
        Self {
            state: TcpState::Closed,
            local_port,
            remote_port,
            send_next: isn,
            send_isn: isn,
            send_una: isn,
            recv_next: 0,
            recv_isn: 0,
            window: WindowTracker::new(65535, 65535),
            timeout: TimeoutTracker::new(),
            history: Vec::new(),
        }
    }

    /// Apply an event, transitioning to the next state.
    pub fn apply(&mut self, event: TcpEvent) -> Result<TcpState, TransitionError> {
        let old = self.state;

        // RST always goes to CLOSED (except from CLOSED itself).
        if event == TcpEvent::RecvRst && self.state != TcpState::Closed {
            self.state = TcpState::Closed;
            self.history.push((old, event, self.state));
            return Ok(self.state);
        }

        let next = Self::next_state(self.state, event)?;
        self.state = next;
        self.update_seq_on_transition(event);
        self.history.push((old, event, next));
        Ok(next)
    }

    /// The core RFC 793 state transition table.
    fn next_state(state: TcpState, event: TcpEvent) -> Result<TcpState, TransitionError> {
        let err = || TransitionError { state, event };
        match (state, event) {
            // From CLOSED
            (TcpState::Closed, TcpEvent::PassiveOpen) => Ok(TcpState::Listen),
            (TcpState::Closed, TcpEvent::ActiveOpen) => Ok(TcpState::SynSent),

            // From LISTEN
            (TcpState::Listen, TcpEvent::RecvSyn) => Ok(TcpState::SynReceived),
            (TcpState::Listen, TcpEvent::Send) => Ok(TcpState::SynSent),
            (TcpState::Listen, TcpEvent::Close) => Ok(TcpState::Closed),

            // From SYN_SENT
            (TcpState::SynSent, TcpEvent::RecvSynAck) => Ok(TcpState::Established),
            (TcpState::SynSent, TcpEvent::RecvSyn) => Ok(TcpState::SynReceived),
            (TcpState::SynSent, TcpEvent::Close) => Ok(TcpState::Closed),
            (TcpState::SynSent, TcpEvent::Timeout) => Ok(TcpState::Closed),

            // From SYN_RECEIVED
            (TcpState::SynReceived, TcpEvent::RecvAck) => Ok(TcpState::Established),
            (TcpState::SynReceived, TcpEvent::Close) => Ok(TcpState::FinWait1),

            // From ESTABLISHED
            (TcpState::Established, TcpEvent::RecvFin) => Ok(TcpState::CloseWait),
            (TcpState::Established, TcpEvent::Close) => Ok(TcpState::FinWait1),

            // From FIN_WAIT_1
            (TcpState::FinWait1, TcpEvent::RecvAck) => Ok(TcpState::FinWait2),
            (TcpState::FinWait1, TcpEvent::RecvFin) => Ok(TcpState::Closing),
            (TcpState::FinWait1, TcpEvent::RecvFinAck) => Ok(TcpState::TimeWait),

            // From FIN_WAIT_2
            (TcpState::FinWait2, TcpEvent::RecvFin) => Ok(TcpState::TimeWait),

            // From CLOSE_WAIT
            (TcpState::CloseWait, TcpEvent::Close) => Ok(TcpState::LastAck),

            // From CLOSING
            (TcpState::Closing, TcpEvent::RecvAck) => Ok(TcpState::TimeWait),

            // From LAST_ACK
            (TcpState::LastAck, TcpEvent::RecvAck) => Ok(TcpState::Closed),

            // From TIME_WAIT
            (TcpState::TimeWait, TcpEvent::Timeout) => Ok(TcpState::Closed),

            _ => Err(err()),
        }
    }

    fn update_seq_on_transition(&mut self, event: TcpEvent) {
        match event {
            TcpEvent::ActiveOpen => {
                // SYN consumes one seq number.
                self.send_next = self.send_next.wrapping_add(1);
            }
            TcpEvent::RecvSynAck => {
                // Record remote ISN and advance recv_next past SYN.
                self.send_una = self.send_next;
            }
            TcpEvent::Close => {
                // FIN consumes one seq number.
                self.send_next = self.send_next.wrapping_add(1);
            }
            _ => {}
        }
    }

    /// Process an incoming segment.
    pub fn recv_segment(&mut self, seg: &TcpSegment) -> Result<TcpState, TransitionError> {
        // Map segment flags to an event.
        let event = if seg.flags.rst {
            TcpEvent::RecvRst
        } else if seg.flags.syn && seg.flags.ack {
            TcpEvent::RecvSynAck
        } else if seg.flags.syn {
            TcpEvent::RecvSyn
        } else if seg.flags.fin && seg.flags.ack {
            TcpEvent::RecvFinAck
        } else if seg.flags.fin {
            TcpEvent::RecvFin
        } else if seg.flags.ack {
            TcpEvent::RecvAck
        } else {
            return Err(TransitionError { state: self.state, event: TcpEvent::RecvAck });
        };

        // Update remote sequence tracking.
        if seg.flags.syn {
            self.recv_isn = seg.seq_num;
            self.recv_next = seg.seq_num.wrapping_add(1);
        }
        if seg.flags.ack {
            self.send_una = seg.ack_num;
        }
        if seg.payload_len > 0 {
            self.recv_next = self.recv_next.wrapping_add(seg.payload_len);
        }
        if seg.flags.fin {
            self.recv_next = self.recv_next.wrapping_add(1);
        }

        self.window.update_send_window(seg.window as u32);
        self.apply(event)
    }

    /// Send data: advance send_next by payload length.
    pub fn send_data(&mut self, len: u32) -> bool {
        if self.state != TcpState::Established {
            return false;
        }
        if !self.window.record_send(len) {
            return false;
        }
        self.send_next = self.send_next.wrapping_add(len);
        true
    }

    /// Get transition history.
    pub fn history(&self) -> &[(TcpState, TcpEvent, TcpState)] {
        &self.history
    }

    /// Whether the connection is in a state that can transmit data.
    pub fn can_send(&self) -> bool {
        self.state == TcpState::Established || self.state == TcpState::CloseWait
    }

    /// Whether the connection is in a state that can receive data.
    pub fn can_recv(&self) -> bool {
        self.state == TcpState::Established
            || self.state == TcpState::FinWait1
            || self.state == TcpState::FinWait2
    }

    /// Whether the connection is fully closed.
    pub fn is_closed(&self) -> bool {
        self.state == TcpState::Closed
    }
}

// ── Connection Table ──────────────────────────────────────────

/// A table tracking multiple TCP connections by (local_port, remote_port).
#[derive(Debug, Default)]
pub struct ConnectionTable {
    connections: Vec<TcpConnection>,
}

impl ConnectionTable {
    pub fn new() -> Self {
        Self { connections: Vec::new() }
    }

    /// Add a connection to the table.
    pub fn add(&mut self, conn: TcpConnection) {
        self.connections.push(conn);
    }

    /// Find a connection by port pair.
    pub fn find(&self, local_port: u16, remote_port: u16) -> Option<&TcpConnection> {
        self.connections
            .iter()
            .find(|c| c.local_port == local_port && c.remote_port == remote_port)
    }

    /// Find a mutable connection by port pair.
    pub fn find_mut(&mut self, local_port: u16, remote_port: u16) -> Option<&mut TcpConnection> {
        self.connections
            .iter_mut()
            .find(|c| c.local_port == local_port && c.remote_port == remote_port)
    }

    /// Remove closed connections.
    pub fn prune_closed(&mut self) -> usize {
        let before = self.connections.len();
        self.connections.retain(|c| c.state != TcpState::Closed);
        before - self.connections.len()
    }

    /// Count connections in a given state.
    pub fn count_in_state(&self, state: TcpState) -> usize {
        self.connections.iter().filter(|c| c.state == state).count()
    }

    /// Total number of tracked connections.
    pub fn len(&self) -> usize {
        self.connections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }

    /// Get all connections in ESTABLISHED state.
    pub fn established(&self) -> Vec<&TcpConnection> {
        self.connections
            .iter()
            .filter(|c| c.state == TcpState::Established)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passive_open_to_listen() {
        let mut conn = TcpConnection::new(80, 0, 1000);
        assert_eq!(conn.apply(TcpEvent::PassiveOpen).unwrap(), TcpState::Listen);
    }

    #[test]
    fn test_active_open_to_syn_sent() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        assert_eq!(conn.apply(TcpEvent::ActiveOpen).unwrap(), TcpState::SynSent);
        // SYN consumes one sequence number.
        assert_eq!(conn.send_next, 1001);
    }

    #[test]
    fn test_three_way_handshake() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        assert_eq!(conn.state, TcpState::Established);
    }

    #[test]
    fn test_passive_handshake() {
        let mut conn = TcpConnection::new(80, 0, 2000);
        conn.apply(TcpEvent::PassiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSyn).unwrap();
        assert_eq!(conn.state, TcpState::SynReceived);
        conn.apply(TcpEvent::RecvAck).unwrap();
        assert_eq!(conn.state, TcpState::Established);
    }

    #[test]
    fn test_active_close() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        conn.apply(TcpEvent::Close).unwrap();
        assert_eq!(conn.state, TcpState::FinWait1);
        conn.apply(TcpEvent::RecvAck).unwrap();
        assert_eq!(conn.state, TcpState::FinWait2);
        conn.apply(TcpEvent::RecvFin).unwrap();
        assert_eq!(conn.state, TcpState::TimeWait);
        conn.apply(TcpEvent::Timeout).unwrap();
        assert_eq!(conn.state, TcpState::Closed);
    }

    #[test]
    fn test_passive_close() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        conn.apply(TcpEvent::RecvFin).unwrap();
        assert_eq!(conn.state, TcpState::CloseWait);
        conn.apply(TcpEvent::Close).unwrap();
        assert_eq!(conn.state, TcpState::LastAck);
        conn.apply(TcpEvent::RecvAck).unwrap();
        assert_eq!(conn.state, TcpState::Closed);
    }

    #[test]
    fn test_simultaneous_close() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        conn.apply(TcpEvent::Close).unwrap();
        assert_eq!(conn.state, TcpState::FinWait1);
        conn.apply(TcpEvent::RecvFin).unwrap();
        assert_eq!(conn.state, TcpState::Closing);
        conn.apply(TcpEvent::RecvAck).unwrap();
        assert_eq!(conn.state, TcpState::TimeWait);
    }

    #[test]
    fn test_rst_resets_to_closed() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvRst).unwrap();
        assert_eq!(conn.state, TcpState::Closed);
    }

    #[test]
    fn test_rst_from_established() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        conn.apply(TcpEvent::RecvRst).unwrap();
        assert_eq!(conn.state, TcpState::Closed);
    }

    #[test]
    fn test_invalid_transition() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        assert!(conn.apply(TcpEvent::RecvAck).is_err());
    }

    #[test]
    fn test_syn_sent_timeout() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::Timeout).unwrap();
        assert_eq!(conn.state, TcpState::Closed);
    }

    #[test]
    fn test_recv_segment_syn() {
        let mut conn = TcpConnection::new(80, 12345, 2000);
        conn.apply(TcpEvent::PassiveOpen).unwrap();

        let seg = TcpSegment {
            src_port: 12345,
            dst_port: 80,
            seq_num: 5000,
            ack_num: 0,
            flags: TcpFlags::syn(),
            window: 32768,
            payload_len: 0,
        };
        conn.recv_segment(&seg).unwrap();
        assert_eq!(conn.state, TcpState::SynReceived);
        assert_eq!(conn.recv_isn, 5000);
        assert_eq!(conn.recv_next, 5001);
    }

    #[test]
    fn test_send_data_established() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        assert!(conn.send_data(100));
        assert_eq!(conn.send_next, 1101); // ISN + 1 (SYN) + 100
    }

    #[test]
    fn test_send_data_not_established() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        assert!(!conn.send_data(100));
    }

    #[test]
    fn test_window_tracker_basic() {
        let mut w = WindowTracker::new(1000, 2000);
        assert_eq!(w.available_send(), 1000);
        assert!(w.record_send(500));
        assert_eq!(w.available_send(), 500);
        assert_eq!(w.bytes_in_flight, 500);
        w.record_ack(200);
        assert_eq!(w.bytes_in_flight, 300);
        assert_eq!(w.available_send(), 700);
    }

    #[test]
    fn test_window_tracker_overflow_protection() {
        let mut w = WindowTracker::new(100, 100);
        assert!(!w.record_send(200));
        assert_eq!(w.bytes_in_flight, 0);
    }

    #[test]
    fn test_window_recv() {
        let mut w = WindowTracker::new(1000, 500);
        assert!(w.consume_recv(200));
        assert_eq!(w.recv_window, 300);
        assert!(!w.consume_recv(400));
        w.free_recv(200);
        assert_eq!(w.recv_window, 500);
    }

    #[test]
    fn test_timeout_tracker_retransmit() {
        let mut t = TimeoutTracker::new();
        assert!(t.retransmit());
        assert_eq!(t.rto_ms, 2000);
        assert!(t.retransmit());
        assert_eq!(t.rto_ms, 4000);
    }

    #[test]
    fn test_timeout_tracker_max_retransmits() {
        let mut t = TimeoutTracker::new();
        t.max_retransmits = 2;
        assert!(t.retransmit());
        assert!(t.retransmit());
        assert!(!t.retransmit());
    }

    #[test]
    fn test_timeout_idle_expired() {
        let mut t = TimeoutTracker::new();
        t.touch(1000);
        assert!(!t.is_idle_expired(50_000));
        assert!(t.is_idle_expired(200_000));
    }

    #[test]
    fn test_connection_history() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        let hist = conn.history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], (TcpState::Closed, TcpEvent::ActiveOpen, TcpState::SynSent));
        assert_eq!(hist[1], (TcpState::SynSent, TcpEvent::RecvSynAck, TcpState::Established));
    }

    #[test]
    fn test_can_send_recv() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        assert!(!conn.can_send());
        assert!(!conn.can_recv());
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        assert!(conn.can_send());
        assert!(conn.can_recv());
    }

    #[test]
    fn test_connection_table() {
        let mut table = ConnectionTable::new();
        assert!(table.is_empty());

        let mut c1 = TcpConnection::new(12345, 80, 1000);
        c1.apply(TcpEvent::ActiveOpen).unwrap();
        c1.apply(TcpEvent::RecvSynAck).unwrap();

        let c2 = TcpConnection::new(12346, 80, 2000);

        table.add(c1);
        table.add(c2);
        assert_eq!(table.len(), 2);
        assert_eq!(table.count_in_state(TcpState::Established), 1);
        assert_eq!(table.count_in_state(TcpState::Closed), 1);
        assert_eq!(table.established().len(), 1);
    }

    #[test]
    fn test_connection_table_prune() {
        let mut table = ConnectionTable::new();
        table.add(TcpConnection::new(1, 80, 100));
        table.add(TcpConnection::new(2, 80, 200));
        assert_eq!(table.prune_closed(), 2);
        assert!(table.is_empty());
    }

    #[test]
    fn test_connection_table_find() {
        let mut table = ConnectionTable::new();
        let mut c = TcpConnection::new(12345, 80, 1000);
        c.apply(TcpEvent::ActiveOpen).unwrap();
        table.add(c);

        assert!(table.find(12345, 80).is_some());
        assert!(table.find(99, 99).is_none());
    }

    #[test]
    fn test_segment_seq_len() {
        let seg = TcpSegment {
            src_port: 1, dst_port: 2, seq_num: 0, ack_num: 0,
            flags: TcpFlags::syn(),
            window: 1000, payload_len: 0,
        };
        assert_eq!(seg.seq_len(), 1);

        let seg2 = TcpSegment {
            src_port: 1, dst_port: 2, seq_num: 0, ack_num: 0,
            flags: TcpFlags::ack(),
            window: 1000, payload_len: 500,
        };
        assert_eq!(seg2.seq_len(), 500);

        let seg3 = TcpSegment {
            src_port: 1, dst_port: 2, seq_num: 0, ack_num: 0,
            flags: TcpFlags::fin_ack(),
            window: 1000, payload_len: 100,
        };
        assert_eq!(seg3.seq_len(), 101);
    }

    #[test]
    fn test_tcp_state_display() {
        assert_eq!(format!("{}", TcpState::Established), "ESTABLISHED");
        assert_eq!(format!("{}", TcpState::SynSent), "SYN_SENT");
        assert_eq!(format!("{}", TcpState::TimeWait), "TIME_WAIT");
    }

    #[test]
    fn test_fin_wait1_to_time_wait_via_fin_ack() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        conn.apply(TcpEvent::RecvSynAck).unwrap();
        conn.apply(TcpEvent::Close).unwrap();
        conn.apply(TcpEvent::RecvFinAck).unwrap();
        assert_eq!(conn.state, TcpState::TimeWait);
    }

    #[test]
    fn test_simultaneous_open() {
        let mut conn = TcpConnection::new(12345, 80, 1000);
        conn.apply(TcpEvent::ActiveOpen).unwrap();
        // Both sides send SYN simultaneously.
        conn.apply(TcpEvent::RecvSyn).unwrap();
        assert_eq!(conn.state, TcpState::SynReceived);
        conn.apply(TcpEvent::RecvAck).unwrap();
        assert_eq!(conn.state, TcpState::Established);
    }

    #[test]
    fn test_window_update_send_window() {
        let mut w = WindowTracker::new(1000, 1000);
        w.record_send(800);
        assert_eq!(w.available_send(), 200);
        w.update_send_window(2000);
        assert_eq!(w.available_send(), 1200);
    }

    #[test]
    fn test_timeout_touch_resets_count() {
        let mut t = TimeoutTracker::new();
        t.retransmit();
        t.retransmit();
        assert_eq!(t.retransmit_count, 2);
        t.touch(5000);
        assert_eq!(t.retransmit_count, 0);
        assert_eq!(t.last_activity_ms, 5000);
    }
}
