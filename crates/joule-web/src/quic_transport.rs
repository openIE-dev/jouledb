//! QUIC-like transport protocol simulation.
//!
//! Models a [`QuicConnection`] with stream multiplexing, a connection
//! state machine (Initial -> Handshake -> Connected -> Closing -> Closed),
//! per-stream and per-connection flow control, stream prioritization,
//! 0-RTT data support, and connection migration.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// QUIC transport domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuicError {
    /// Connection is not in a valid state for this operation.
    InvalidState { current: ConnectionState, expected: &'static str },
    /// Stream not found.
    StreamNotFound(u64),
    /// Duplicate stream ID.
    DuplicateStream(u64),
    /// Maximum streams reached.
    MaxStreamsReached { max: usize },
    /// Flow control limit exceeded.
    FlowControlExceeded { stream_id: u64, limit: u64 },
    /// Connection-level flow control exceeded.
    ConnectionFlowExceeded { limit: u64 },
    /// Connection is closed.
    ConnectionClosed,
    /// Migration failed.
    MigrationFailed(String),
}

impl fmt::Display for QuicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidState { current, expected } => {
                write!(f, "invalid state: {current:?}, expected {expected}")
            }
            Self::StreamNotFound(id) => write!(f, "stream not found: {id}"),
            Self::DuplicateStream(id) => write!(f, "duplicate stream: {id}"),
            Self::MaxStreamsReached { max } => write!(f, "max streams reached: {max}"),
            Self::FlowControlExceeded { stream_id, limit } => {
                write!(f, "flow control exceeded on stream {stream_id} (limit={limit})")
            }
            Self::ConnectionFlowExceeded { limit } => {
                write!(f, "connection flow control exceeded (limit={limit})")
            }
            Self::ConnectionClosed => write!(f, "connection closed"),
            Self::MigrationFailed(reason) => write!(f, "migration failed: {reason}"),
        }
    }
}

impl std::error::Error for QuicError {}

// ── Connection State ────────────────────────────────────────────

/// State machine for a QUIC connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectionState {
    Initial,
    Handshake,
    Connected,
    Closing,
    Closed,
}

impl fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initial => write!(f, "Initial"),
            Self::Handshake => write!(f, "Handshake"),
            Self::Connected => write!(f, "Connected"),
            Self::Closing => write!(f, "Closing"),
            Self::Closed => write!(f, "Closed"),
        }
    }
}

// ── Stream Priority ─────────────────────────────────────────────

/// Priority level for a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for StreamPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ── Stream ──────────────────────────────────────────────────────

/// A multiplexed stream within a QUIC connection.
#[derive(Debug, Clone)]
pub struct QuicStream {
    pub id: u64,
    pub priority: StreamPriority,
    pub send_buffer: VecDeque<Vec<u8>>,
    pub recv_buffer: VecDeque<Vec<u8>>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub flow_limit: u64,
    pub open: bool,
}

impl QuicStream {
    pub fn new(id: u64, flow_limit: u64) -> Self {
        Self {
            id,
            priority: StreamPriority::default(),
            send_buffer: VecDeque::new(),
            recv_buffer: VecDeque::new(),
            bytes_sent: 0,
            bytes_received: 0,
            flow_limit,
            open: true,
        }
    }

    pub fn with_priority(mut self, priority: StreamPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Remaining flow control budget for this stream.
    pub fn remaining_budget(&self) -> u64 {
        self.flow_limit.saturating_sub(self.bytes_sent)
    }
}

// ── Connection Config ───────────────────────────────────────────

/// Configuration for a QUIC connection.
#[derive(Debug, Clone)]
pub struct QuicConfig {
    pub max_streams: usize,
    pub initial_stream_flow_limit: u64,
    pub connection_flow_limit: u64,
    pub enable_0rtt: bool,
    pub max_send_buffer: usize,
}

impl Default for QuicConfig {
    fn default() -> Self {
        Self {
            max_streams: 100,
            initial_stream_flow_limit: 65536,
            connection_flow_limit: 1_048_576,
            enable_0rtt: false,
            max_send_buffer: 1024,
        }
    }
}

impl QuicConfig {
    pub fn with_max_streams(mut self, max: usize) -> Self {
        self.max_streams = max;
        self
    }

    pub fn with_0rtt(mut self, enable: bool) -> Self {
        self.enable_0rtt = enable;
        self
    }

    pub fn with_connection_flow_limit(mut self, limit: u64) -> Self {
        self.connection_flow_limit = limit;
        self
    }
}

// ── Connection Statistics ───────────────────────────────────────

/// Cumulative connection statistics.
#[derive(Debug, Clone, Default)]
pub struct ConnectionStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub streams_opened: u64,
    pub streams_closed: u64,
    pub migrations: u64,
    pub zero_rtt_data: u64,
}

impl fmt::Display for ConnectionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sent={}B recv={}B streams={}/{} migrations={}",
            self.bytes_sent,
            self.bytes_received,
            self.streams_opened,
            self.streams_closed,
            self.migrations,
        )
    }
}

// ── Address ─────────────────────────────────────────────────────

/// A simulated network address.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Address {
    pub ip: String,
    pub port: u16,
}

impl Address {
    pub fn new(ip: impl Into<String>, port: u16) -> Self {
        Self { ip: ip.into(), port }
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.ip, self.port)
    }
}

// ── QUIC Connection ─────────────────────────────────────────────

/// A QUIC-like connection with stream multiplexing.
pub struct QuicConnection {
    pub connection_id: u64,
    config: QuicConfig,
    state: ConnectionState,
    streams: BTreeMap<u64, QuicStream>,
    next_stream_id: u64,
    total_bytes_sent: u64,
    local_address: Address,
    remote_address: Address,
    zero_rtt_buffer: VecDeque<Vec<u8>>,
    stats: ConnectionStats,
}

impl QuicConnection {
    pub fn new(
        connection_id: u64,
        local: Address,
        remote: Address,
        config: QuicConfig,
    ) -> Self {
        Self {
            connection_id,
            config,
            state: ConnectionState::Initial,
            streams: BTreeMap::new(),
            next_stream_id: 0,
            total_bytes_sent: 0,
            local_address: local,
            remote_address: remote,
            zero_rtt_buffer: VecDeque::new(),
            stats: ConnectionStats::default(),
        }
    }

    /// Current connection state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Advance to Handshake state.
    pub fn initiate_handshake(&mut self) -> Result<(), QuicError> {
        if self.state != ConnectionState::Initial {
            return Err(QuicError::InvalidState {
                current: self.state,
                expected: "Initial",
            });
        }
        self.state = ConnectionState::Handshake;
        Ok(())
    }

    /// Complete handshake and move to Connected.
    pub fn complete_handshake(&mut self) -> Result<(), QuicError> {
        if self.state != ConnectionState::Handshake {
            return Err(QuicError::InvalidState {
                current: self.state,
                expected: "Handshake",
            });
        }
        self.state = ConnectionState::Connected;

        // Deliver any 0-RTT data.
        if self.config.enable_0rtt && !self.zero_rtt_buffer.is_empty() {
            let stream_id = self.open_stream_internal()?;
            let stream = self.streams.get_mut(&stream_id).unwrap();
            while let Some(data) = self.zero_rtt_buffer.pop_front() {
                stream.recv_buffer.push_back(data);
            }
        }

        Ok(())
    }

    /// Start closing the connection.
    pub fn initiate_close(&mut self) -> Result<(), QuicError> {
        if self.state != ConnectionState::Connected {
            return Err(QuicError::InvalidState {
                current: self.state,
                expected: "Connected",
            });
        }
        self.state = ConnectionState::Closing;
        Ok(())
    }

    /// Finalize the close.
    pub fn finalize_close(&mut self) -> Result<(), QuicError> {
        if self.state != ConnectionState::Closing {
            return Err(QuicError::InvalidState {
                current: self.state,
                expected: "Closing",
            });
        }
        // Close all remaining streams.
        for stream in self.streams.values_mut() {
            if stream.open {
                stream.open = false;
                self.stats.streams_closed += 1;
            }
        }
        self.state = ConnectionState::Closed;
        Ok(())
    }

    /// Open a new stream.
    pub fn open_stream(&mut self) -> Result<u64, QuicError> {
        self.require_connected()?;
        self.open_stream_internal()
    }

    /// Open a stream with a specific priority.
    pub fn open_stream_with_priority(&mut self, priority: StreamPriority) -> Result<u64, QuicError> {
        let id = self.open_stream()?;
        if let Some(s) = self.streams.get_mut(&id) {
            s.priority = priority;
        }
        Ok(id)
    }

    /// Close a stream.
    pub fn close_stream(&mut self, stream_id: u64) -> Result<(), QuicError> {
        let stream = self.streams.get_mut(&stream_id)
            .ok_or(QuicError::StreamNotFound(stream_id))?;
        if !stream.open {
            return Err(QuicError::StreamNotFound(stream_id));
        }
        stream.open = false;
        self.stats.streams_closed += 1;
        Ok(())
    }

    /// Send data on a stream.
    pub fn send(&mut self, stream_id: u64, data: Vec<u8>) -> Result<(), QuicError> {
        self.require_connected()?;

        let data_len = data.len() as u64;

        // Connection-level flow control.
        if self.total_bytes_sent + data_len > self.config.connection_flow_limit {
            return Err(QuicError::ConnectionFlowExceeded {
                limit: self.config.connection_flow_limit,
            });
        }

        let stream = self.streams.get_mut(&stream_id)
            .ok_or(QuicError::StreamNotFound(stream_id))?;
        if !stream.open {
            return Err(QuicError::StreamNotFound(stream_id));
        }

        // Stream-level flow control.
        if stream.bytes_sent + data_len > stream.flow_limit {
            return Err(QuicError::FlowControlExceeded {
                stream_id,
                limit: stream.flow_limit,
            });
        }

        stream.bytes_sent += data_len;
        stream.send_buffer.push_back(data);
        self.total_bytes_sent += data_len;
        self.stats.bytes_sent += data_len;

        Ok(())
    }

    /// Receive data for a stream (simulated: place into recv buffer).
    pub fn receive(&mut self, stream_id: u64, data: Vec<u8>) -> Result<(), QuicError> {
        self.require_connected()?;
        let data_len = data.len() as u64;
        let stream = self.streams.get_mut(&stream_id)
            .ok_or(QuicError::StreamNotFound(stream_id))?;
        stream.bytes_received += data_len;
        stream.recv_buffer.push_back(data);
        self.stats.bytes_received += data_len;
        Ok(())
    }

    /// Drain received data from a stream.
    pub fn drain_recv(&mut self, stream_id: u64) -> Result<Vec<Vec<u8>>, QuicError> {
        let stream = self.streams.get_mut(&stream_id)
            .ok_or(QuicError::StreamNotFound(stream_id))?;
        Ok(stream.recv_buffer.drain(..).collect())
    }

    /// Send 0-RTT data before handshake completes.
    pub fn send_0rtt(&mut self, data: Vec<u8>) -> Result<(), QuicError> {
        if !self.config.enable_0rtt {
            return Err(QuicError::InvalidState {
                current: self.state,
                expected: "0-RTT enabled",
            });
        }
        self.zero_rtt_buffer.push_back(data.clone());
        self.stats.zero_rtt_data += data.len() as u64;
        Ok(())
    }

    /// Migrate connection to a new local address.
    pub fn migrate(&mut self, new_local: Address) -> Result<(), QuicError> {
        self.require_connected()?;
        self.local_address = new_local;
        self.stats.migrations += 1;
        Ok(())
    }

    /// Get streams ordered by priority (highest first).
    pub fn streams_by_priority(&self) -> Vec<u64> {
        let mut ids: Vec<_> = self.streams.iter()
            .filter(|(_, s)| s.open)
            .map(|(&id, s)| (s.priority, id))
            .collect();
        ids.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// Number of open streams.
    pub fn stream_count(&self) -> usize {
        self.streams.values().filter(|s| s.open).count()
    }

    /// Get a reference to a stream.
    pub fn stream(&self, stream_id: u64) -> Option<&QuicStream> {
        self.streams.get(&stream_id)
    }

    /// Get connection statistics.
    pub fn stats(&self) -> &ConnectionStats {
        &self.stats
    }

    /// Local address.
    pub fn local_address(&self) -> &Address {
        &self.local_address
    }

    /// Remote address.
    pub fn remote_address(&self) -> &Address {
        &self.remote_address
    }

    // ── Internal ────────────────────────────────────────────────

    fn require_connected(&self) -> Result<(), QuicError> {
        if self.state == ConnectionState::Closed {
            return Err(QuicError::ConnectionClosed);
        }
        if self.state != ConnectionState::Connected {
            return Err(QuicError::InvalidState {
                current: self.state,
                expected: "Connected",
            });
        }
        Ok(())
    }

    fn open_stream_internal(&mut self) -> Result<u64, QuicError> {
        let open_count = self.streams.values().filter(|s| s.open).count();
        if open_count >= self.config.max_streams {
            return Err(QuicError::MaxStreamsReached { max: self.config.max_streams });
        }
        let id = self.next_stream_id;
        self.next_stream_id += 1;
        let stream = QuicStream::new(id, self.config.initial_stream_flow_limit);
        self.streams.insert(id, stream);
        self.stats.streams_opened += 1;
        Ok(id)
    }
}

impl fmt::Display for QuicConnection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuicConnection(id={}, state={}, streams={}, {} -> {})",
            self.connection_id,
            self.state,
            self.stream_count(),
            self.local_address,
            self.remote_address,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn connected_conn() -> QuicConnection {
        let local = Address::new("192.168.1.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, QuicConfig::default());
        conn.initiate_handshake().unwrap();
        conn.complete_handshake().unwrap();
        conn
    }

    #[test]
    fn state_machine_normal_flow() {
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, QuicConfig::default());
        assert_eq!(conn.state(), ConnectionState::Initial);
        conn.initiate_handshake().unwrap();
        assert_eq!(conn.state(), ConnectionState::Handshake);
        conn.complete_handshake().unwrap();
        assert_eq!(conn.state(), ConnectionState::Connected);
        conn.initiate_close().unwrap();
        assert_eq!(conn.state(), ConnectionState::Closing);
        conn.finalize_close().unwrap();
        assert_eq!(conn.state(), ConnectionState::Closed);
    }

    #[test]
    fn invalid_state_transition() {
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, QuicConfig::default());
        let err = conn.complete_handshake().unwrap_err();
        assert!(matches!(err, QuicError::InvalidState { .. }));
    }

    #[test]
    fn open_and_close_stream() {
        let mut conn = connected_conn();
        let id = conn.open_stream().unwrap();
        assert_eq!(conn.stream_count(), 1);
        conn.close_stream(id).unwrap();
        assert_eq!(conn.stream_count(), 0);
    }

    #[test]
    fn max_streams_enforced() {
        let config = QuicConfig::default().with_max_streams(2);
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, config);
        conn.initiate_handshake().unwrap();
        conn.complete_handshake().unwrap();
        conn.open_stream().unwrap();
        conn.open_stream().unwrap();
        let err = conn.open_stream().unwrap_err();
        assert!(matches!(err, QuicError::MaxStreamsReached { max: 2 }));
    }

    #[test]
    fn send_receive_on_stream() {
        let mut conn = connected_conn();
        let sid = conn.open_stream().unwrap();
        conn.send(sid, vec![1, 2, 3]).unwrap();
        conn.receive(sid, vec![4, 5]).unwrap();
        let data = conn.drain_recv(sid).unwrap();
        assert_eq!(data, vec![vec![4, 5]]);
    }

    #[test]
    fn stream_flow_control() {
        let mut config = QuicConfig::default();
        config.initial_stream_flow_limit = 10;
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, config);
        conn.initiate_handshake().unwrap();
        conn.complete_handshake().unwrap();
        let sid = conn.open_stream().unwrap();
        conn.send(sid, vec![0; 8]).unwrap();
        let err = conn.send(sid, vec![0; 5]).unwrap_err();
        assert!(matches!(err, QuicError::FlowControlExceeded { .. }));
    }

    #[test]
    fn connection_flow_control() {
        let config = QuicConfig::default().with_connection_flow_limit(20);
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, config);
        conn.initiate_handshake().unwrap();
        conn.complete_handshake().unwrap();
        let s1 = conn.open_stream().unwrap();
        conn.send(s1, vec![0; 15]).unwrap();
        let err = conn.send(s1, vec![0; 10]).unwrap_err();
        assert!(matches!(err, QuicError::ConnectionFlowExceeded { .. }));
    }

    #[test]
    fn stream_priority_ordering() {
        let mut conn = connected_conn();
        let low = conn.open_stream_with_priority(StreamPriority::Low).unwrap();
        let high = conn.open_stream_with_priority(StreamPriority::High).unwrap();
        let normal = conn.open_stream_with_priority(StreamPriority::Normal).unwrap();
        let ordered = conn.streams_by_priority();
        assert_eq!(ordered[0], high);
        assert_eq!(ordered[1], normal);
        assert_eq!(ordered[2], low);
    }

    #[test]
    fn zero_rtt_data() {
        let config = QuicConfig::default().with_0rtt(true);
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, config);
        conn.initiate_handshake().unwrap();
        conn.send_0rtt(vec![42, 43]).unwrap();
        conn.complete_handshake().unwrap();
        // 0-RTT data should be in the auto-opened stream.
        assert!(conn.stream_count() >= 1);
        assert_eq!(conn.stats().zero_rtt_data, 2);
    }

    #[test]
    fn zero_rtt_disabled_error() {
        let local = Address::new("127.0.0.1", 4433);
        let remote = Address::new("10.0.0.1", 443);
        let mut conn = QuicConnection::new(1, local, remote, QuicConfig::default());
        let err = conn.send_0rtt(vec![1]).unwrap_err();
        assert!(matches!(err, QuicError::InvalidState { .. }));
    }

    #[test]
    fn connection_migration() {
        let mut conn = connected_conn();
        let new_addr = Address::new("192.168.1.100", 5000);
        conn.migrate(new_addr.clone()).unwrap();
        assert_eq!(conn.local_address().ip, "192.168.1.100");
        assert_eq!(conn.stats().migrations, 1);
    }

    #[test]
    fn send_on_nonexistent_stream() {
        let mut conn = connected_conn();
        let err = conn.send(999, vec![1]).unwrap_err();
        assert!(matches!(err, QuicError::StreamNotFound(999)));
    }

    #[test]
    fn send_after_close_errors() {
        let mut conn = connected_conn();
        conn.initiate_close().unwrap();
        conn.finalize_close().unwrap();
        let err = conn.open_stream().unwrap_err();
        assert!(matches!(err, QuicError::ConnectionClosed));
    }

    #[test]
    fn stats_tracking() {
        let mut conn = connected_conn();
        let sid = conn.open_stream().unwrap();
        conn.send(sid, vec![0; 100]).unwrap();
        conn.receive(sid, vec![0; 50]).unwrap();
        assert_eq!(conn.stats().bytes_sent, 100);
        assert_eq!(conn.stats().bytes_received, 50);
        assert_eq!(conn.stats().streams_opened, 1);
    }

    #[test]
    fn finalize_close_closes_all_streams() {
        let mut conn = connected_conn();
        conn.open_stream().unwrap();
        conn.open_stream().unwrap();
        assert_eq!(conn.stream_count(), 2);
        conn.initiate_close().unwrap();
        conn.finalize_close().unwrap();
        assert_eq!(conn.stream_count(), 0);
    }

    #[test]
    fn connection_display() {
        let conn = connected_conn();
        let s = format!("{conn}");
        assert!(s.contains("QuicConnection"));
        assert!(s.contains("Connected"));
    }

    #[test]
    fn address_display() {
        let addr = Address::new("10.0.0.1", 443);
        assert_eq!(format!("{addr}"), "10.0.0.1:443");
    }

    #[test]
    fn config_builder() {
        let config = QuicConfig::default()
            .with_max_streams(50)
            .with_0rtt(true)
            .with_connection_flow_limit(500_000);
        assert_eq!(config.max_streams, 50);
        assert!(config.enable_0rtt);
        assert_eq!(config.connection_flow_limit, 500_000);
    }

    #[test]
    fn stream_remaining_budget() {
        let stream = QuicStream::new(0, 1000);
        assert_eq!(stream.remaining_budget(), 1000);
    }
}
