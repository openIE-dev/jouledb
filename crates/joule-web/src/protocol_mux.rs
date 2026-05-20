//! Protocol multiplexer — stream multiplexing, flow control, priority, framing.
//!
//! Pure Rust implementation of a multiplexing protocol layer (HTTP/2-style).
//! Manages multiple logical streams over a single connection with stream IDs,
//! per-stream flow control windows, priority levels, typed frames (data,
//! headers, RST, ping, window_update), and demultiplexing of incoming frames.

use std::collections::HashMap;
use std::fmt;

// ── Frame Types ───────────────────────────────────────────────

/// Types of protocol frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    Data,
    Headers,
    Priority,
    RstStream,
    Settings,
    Ping,
    GoAway,
    WindowUpdate,
}

impl FrameType {
    pub fn type_id(&self) -> u8 {
        match self {
            Self::Data => 0,
            Self::Headers => 1,
            Self::Priority => 2,
            Self::RstStream => 3,
            Self::Settings => 4,
            Self::Ping => 6,
            Self::GoAway => 7,
            Self::WindowUpdate => 8,
        }
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Data),
            1 => Some(Self::Headers),
            2 => Some(Self::Priority),
            3 => Some(Self::RstStream),
            4 => Some(Self::Settings),
            6 => Some(Self::Ping),
            7 => Some(Self::GoAway),
            8 => Some(Self::WindowUpdate),
            _ => None,
        }
    }
}

impl fmt::Display for FrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Data => "DATA",
            Self::Headers => "HEADERS",
            Self::Priority => "PRIORITY",
            Self::RstStream => "RST_STREAM",
            Self::Settings => "SETTINGS",
            Self::Ping => "PING",
            Self::GoAway => "GOAWAY",
            Self::WindowUpdate => "WINDOW_UPDATE",
        };
        f.write_str(s)
    }
}

// ── Frame Flags ───────────────────────────────────────────────

/// Frame flags bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameFlags(pub u8);

impl FrameFlags {
    pub const END_STREAM: u8 = 0x01;
    pub const END_HEADERS: u8 = 0x04;
    pub const PADDED: u8 = 0x08;
    pub const ACK: u8 = 0x01; // For SETTINGS and PING.

    pub fn has(&self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

// ── Frame ─────────────────────────────────────────────────────

/// A protocol frame.
#[derive(Debug, Clone)]
pub struct Frame {
    pub frame_type: FrameType,
    pub stream_id: u32,
    pub flags: FrameFlags,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(frame_type: FrameType, stream_id: u32, payload: Vec<u8>) -> Self {
        Self {
            frame_type,
            stream_id,
            flags: FrameFlags::default(),
            payload,
        }
    }

    pub fn with_flags(mut self, flags: FrameFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Create a DATA frame.
    pub fn data(stream_id: u32, payload: Vec<u8>, end_stream: bool) -> Self {
        let mut flags = FrameFlags::default();
        if end_stream {
            flags.set(FrameFlags::END_STREAM);
        }
        Self { frame_type: FrameType::Data, stream_id, flags, payload }
    }

    /// Create a HEADERS frame.
    pub fn headers(stream_id: u32, header_block: Vec<u8>, end_stream: bool, end_headers: bool) -> Self {
        let mut flags = FrameFlags::default();
        if end_stream {
            flags.set(FrameFlags::END_STREAM);
        }
        if end_headers {
            flags.set(FrameFlags::END_HEADERS);
        }
        Self { frame_type: FrameType::Headers, stream_id, flags, payload: header_block }
    }

    /// Create a RST_STREAM frame.
    pub fn rst_stream(stream_id: u32, error_code: u32) -> Self {
        Self {
            frame_type: FrameType::RstStream,
            stream_id,
            flags: FrameFlags::default(),
            payload: error_code.to_be_bytes().to_vec(),
        }
    }

    /// Create a PING frame.
    pub fn ping(data: [u8; 8], ack: bool) -> Self {
        let mut flags = FrameFlags::default();
        if ack {
            flags.set(FrameFlags::ACK);
        }
        Self {
            frame_type: FrameType::Ping,
            stream_id: 0,
            flags,
            payload: data.to_vec(),
        }
    }

    /// Create a WINDOW_UPDATE frame.
    pub fn window_update(stream_id: u32, increment: u32) -> Self {
        Self {
            frame_type: FrameType::WindowUpdate,
            stream_id,
            flags: FrameFlags::default(),
            payload: increment.to_be_bytes().to_vec(),
        }
    }

    /// Create a GOAWAY frame.
    pub fn goaway(last_stream_id: u32, error_code: u32) -> Self {
        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&last_stream_id.to_be_bytes());
        payload.extend_from_slice(&error_code.to_be_bytes());
        Self {
            frame_type: FrameType::GoAway,
            stream_id: 0,
            flags: FrameFlags::default(),
            payload,
        }
    }

    /// Create a SETTINGS frame.
    pub fn settings(pairs: &[(u16, u32)], ack: bool) -> Self {
        let mut payload = Vec::new();
        for (id, val) in pairs {
            payload.extend_from_slice(&id.to_be_bytes());
            payload.extend_from_slice(&val.to_be_bytes());
        }
        let mut flags = FrameFlags::default();
        if ack {
            flags.set(FrameFlags::ACK);
        }
        Self {
            frame_type: FrameType::Settings,
            stream_id: 0,
            flags,
            payload,
        }
    }

    /// Whether this frame ends the stream.
    pub fn is_end_stream(&self) -> bool {
        self.flags.has(FrameFlags::END_STREAM)
    }

    /// Payload length.
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }

    /// Extract error code from RST_STREAM or GOAWAY payload.
    pub fn error_code(&self) -> Option<u32> {
        match self.frame_type {
            FrameType::RstStream if self.payload.len() >= 4 => {
                Some(u32::from_be_bytes([
                    self.payload[0], self.payload[1],
                    self.payload[2], self.payload[3],
                ]))
            }
            FrameType::GoAway if self.payload.len() >= 8 => {
                Some(u32::from_be_bytes([
                    self.payload[4], self.payload[5],
                    self.payload[6], self.payload[7],
                ]))
            }
            _ => None,
        }
    }

    /// Extract window increment from WINDOW_UPDATE payload.
    pub fn window_increment(&self) -> Option<u32> {
        if self.frame_type == FrameType::WindowUpdate && self.payload.len() >= 4 {
            Some(u32::from_be_bytes([
                self.payload[0], self.payload[1],
                self.payload[2], self.payload[3],
            ]) & 0x7FFF_FFFF)
        } else {
            None
        }
    }
}

// ── Stream State ──────────────────────────────────────────────

/// State of a multiplexed stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Idle,
    Open,
    ReservedLocal,
    ReservedRemote,
    HalfClosedLocal,
    HalfClosedRemote,
    Closed,
}

impl fmt::Display for StreamState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Idle => "idle",
            Self::Open => "open",
            Self::ReservedLocal => "reserved(local)",
            Self::ReservedRemote => "reserved(remote)",
            Self::HalfClosedLocal => "half-closed(local)",
            Self::HalfClosedRemote => "half-closed(remote)",
            Self::Closed => "closed",
        };
        f.write_str(s)
    }
}

// ── Stream ────────────────────────────────────────────────────

/// A multiplexed stream.
#[derive(Debug, Clone)]
pub struct MuxStream {
    pub id: u32,
    pub state: StreamState,
    pub priority: u8,
    /// Flow control window (send direction).
    pub send_window: i64,
    /// Flow control window (receive direction).
    pub recv_window: i64,
    /// Accumulated received data.
    pub recv_buffer: Vec<u8>,
    /// Accumulated header data.
    pub recv_headers: Vec<u8>,
    /// Total bytes sent on this stream.
    pub bytes_sent: u64,
    /// Total bytes received on this stream.
    pub bytes_received: u64,
}

impl MuxStream {
    pub fn new(id: u32, initial_window: i64) -> Self {
        Self {
            id,
            state: StreamState::Idle,
            priority: 128,
            send_window: initial_window,
            recv_window: initial_window,
            recv_buffer: Vec::new(),
            recv_headers: Vec::new(),
            bytes_sent: 0,
            bytes_received: 0,
        }
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Whether the stream can send data.
    pub fn can_send(&self) -> bool {
        matches!(self.state, StreamState::Open | StreamState::HalfClosedRemote)
            && self.send_window > 0
    }

    /// Whether the stream can receive data.
    pub fn can_recv(&self) -> bool {
        matches!(self.state, StreamState::Open | StreamState::HalfClosedLocal)
    }

    /// Consume send window.
    pub fn consume_send_window(&mut self, n: u32) -> bool {
        if (n as i64) > self.send_window {
            return false;
        }
        self.send_window -= n as i64;
        self.bytes_sent += n as u64;
        true
    }

    /// Consume receive window.
    pub fn consume_recv_window(&mut self, n: u32) -> bool {
        if (n as i64) > self.recv_window {
            return false;
        }
        self.recv_window -= n as i64;
        self.bytes_received += n as u64;
        true
    }

    /// Increase send window (from WINDOW_UPDATE).
    pub fn increase_send_window(&mut self, increment: u32) {
        self.send_window += increment as i64;
    }

    /// Increase receive window (application consumed data).
    pub fn increase_recv_window(&mut self, increment: u32) {
        self.recv_window += increment as i64;
    }
}

// ── Mux Error ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MuxError {
    StreamNotFound { stream_id: u32 },
    StreamClosed { stream_id: u32 },
    FlowControlError { stream_id: u32, window: i64, requested: u32 },
    InvalidStreamId,
    MaxStreamsExceeded { max: u32 },
    ProtocolError(String),
}

impl fmt::Display for MuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StreamNotFound { stream_id } =>
                write!(f, "stream {} not found", stream_id),
            Self::StreamClosed { stream_id } =>
                write!(f, "stream {} is closed", stream_id),
            Self::FlowControlError { stream_id, window, requested } =>
                write!(f, "flow control: stream {} window={}, requested={}", stream_id, window, requested),
            Self::InvalidStreamId =>
                write!(f, "invalid stream ID"),
            Self::MaxStreamsExceeded { max } =>
                write!(f, "maximum streams exceeded: {}", max),
            Self::ProtocolError(msg) =>
                write!(f, "protocol error: {}", msg),
        }
    }
}

// ── Protocol Multiplexer ──────────────────────────────────────

/// The multiplexer managing all streams over a single connection.
pub struct ProtocolMux {
    streams: HashMap<u32, MuxStream>,
    /// Next stream ID to allocate (client-initiated: odd, server: even).
    next_stream_id: u32,
    /// Whether this is the client side (odd IDs) or server side (even IDs).
    is_client: bool,
    /// Connection-level send window.
    connection_send_window: i64,
    /// Connection-level receive window.
    connection_recv_window: i64,
    /// Initial window size for new streams.
    initial_window_size: i64,
    /// Maximum concurrent streams.
    max_concurrent_streams: u32,
    /// Outbound frame queue.
    outbound: Vec<Frame>,
    /// Received ping data awaiting ack.
    pending_pings: Vec<[u8; 8]>,
    /// Whether GOAWAY has been sent.
    goaway_sent: bool,
    /// Last stream ID in GOAWAY.
    goaway_last_stream_id: u32,
}

impl ProtocolMux {
    pub fn new(is_client: bool) -> Self {
        Self {
            streams: HashMap::new(),
            next_stream_id: if is_client { 1 } else { 2 },
            is_client,
            connection_send_window: 65535,
            connection_recv_window: 65535,
            initial_window_size: 65535,
            max_concurrent_streams: 100,
            outbound: Vec::new(),
            pending_pings: Vec::new(),
            goaway_sent: false,
            goaway_last_stream_id: 0,
        }
    }

    /// Create a new stream and return its ID.
    pub fn open_stream(&mut self) -> Result<u32, MuxError> {
        if self.goaway_sent {
            return Err(MuxError::ProtocolError("GOAWAY sent".into()));
        }
        let active = self.streams.values()
            .filter(|s| !matches!(s.state, StreamState::Closed | StreamState::Idle))
            .count() as u32;
        if active >= self.max_concurrent_streams {
            return Err(MuxError::MaxStreamsExceeded { max: self.max_concurrent_streams });
        }

        let id = self.next_stream_id;
        self.next_stream_id += 2;

        let mut stream = MuxStream::new(id, self.initial_window_size);
        stream.state = StreamState::Open;
        self.streams.insert(id, stream);
        Ok(id)
    }

    /// Get a stream by ID.
    pub fn stream(&self, id: u32) -> Option<&MuxStream> {
        self.streams.get(&id)
    }

    /// Get a mutable stream by ID.
    pub fn stream_mut(&mut self, id: u32) -> Option<&mut MuxStream> {
        self.streams.get_mut(&id)
    }

    /// Send data on a stream.
    pub fn send_data(&mut self, stream_id: u32, data: Vec<u8>, end_stream: bool) -> Result<(), MuxError> {
        let data_len = data.len() as u32;

        // Check connection-level window.
        if (data_len as i64) > self.connection_send_window {
            return Err(MuxError::FlowControlError {
                stream_id: 0,
                window: self.connection_send_window,
                requested: data_len,
            });
        }

        let stream = self.streams.get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound { stream_id })?;

        if !stream.can_send() {
            return Err(MuxError::StreamClosed { stream_id });
        }

        if !stream.consume_send_window(data_len) {
            return Err(MuxError::FlowControlError {
                stream_id,
                window: stream.send_window,
                requested: data_len,
            });
        }

        self.connection_send_window -= data_len as i64;

        if end_stream {
            let stream = self.streams.get_mut(&stream_id).unwrap();
            stream.state = match stream.state {
                StreamState::Open => StreamState::HalfClosedLocal,
                StreamState::HalfClosedRemote => StreamState::Closed,
                other => other,
            };
        }

        self.outbound.push(Frame::data(stream_id, data, end_stream));
        Ok(())
    }

    /// Send headers on a stream.
    pub fn send_headers(&mut self, stream_id: u32, headers: Vec<u8>, end_stream: bool) -> Result<(), MuxError> {
        let stream = self.streams.get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound { stream_id })?;

        if matches!(stream.state, StreamState::Closed) {
            return Err(MuxError::StreamClosed { stream_id });
        }

        if end_stream {
            stream.state = match stream.state {
                StreamState::Open => StreamState::HalfClosedLocal,
                StreamState::HalfClosedRemote => StreamState::Closed,
                other => other,
            };
        }

        self.outbound.push(Frame::headers(stream_id, headers, end_stream, true));
        Ok(())
    }

    /// Reset a stream.
    pub fn reset_stream(&mut self, stream_id: u32, error_code: u32) -> Result<(), MuxError> {
        let stream = self.streams.get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound { stream_id })?;
        stream.state = StreamState::Closed;
        self.outbound.push(Frame::rst_stream(stream_id, error_code));
        Ok(())
    }

    /// Send a ping.
    pub fn send_ping(&mut self, data: [u8; 8]) {
        self.outbound.push(Frame::ping(data, false));
    }

    /// Initiate graceful shutdown.
    pub fn send_goaway(&mut self, error_code: u32) {
        let last_id = self.streams.keys().max().copied().unwrap_or(0);
        self.goaway_sent = true;
        self.goaway_last_stream_id = last_id;
        self.outbound.push(Frame::goaway(last_id, error_code));
    }

    /// Process an incoming frame.
    pub fn recv_frame(&mut self, frame: Frame) -> Result<(), MuxError> {
        match frame.frame_type {
            FrameType::Data => self.handle_data(frame),
            FrameType::Headers => self.handle_headers(frame),
            FrameType::RstStream => self.handle_rst(frame),
            FrameType::Ping => self.handle_ping(frame),
            FrameType::WindowUpdate => self.handle_window_update(frame),
            FrameType::GoAway => self.handle_goaway(frame),
            FrameType::Settings => Ok(()), // Simplified — just accept.
            FrameType::Priority => self.handle_priority(frame),
        }
    }

    fn handle_data(&mut self, frame: Frame) -> Result<(), MuxError> {
        let stream_id = frame.stream_id;
        let data_len = frame.payload.len() as u32;

        self.connection_recv_window -= data_len as i64;

        let stream = self.streams.get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound { stream_id })?;

        if !stream.can_recv() {
            return Err(MuxError::StreamClosed { stream_id });
        }

        stream.consume_recv_window(data_len);
        stream.recv_buffer.extend_from_slice(&frame.payload);

        if frame.is_end_stream() {
            stream.state = match stream.state {
                StreamState::Open => StreamState::HalfClosedRemote,
                StreamState::HalfClosedLocal => StreamState::Closed,
                other => other,
            };
        }

        Ok(())
    }

    fn handle_headers(&mut self, frame: Frame) -> Result<(), MuxError> {
        let stream_id = frame.stream_id;

        // If stream doesn't exist, create it (remote-initiated).
        if !self.streams.contains_key(&stream_id) {
            let mut stream = MuxStream::new(stream_id, self.initial_window_size);
            stream.state = StreamState::Open;
            self.streams.insert(stream_id, stream);
        }

        let stream = self.streams.get_mut(&stream_id).unwrap();
        stream.recv_headers.extend_from_slice(&frame.payload);

        if frame.is_end_stream() {
            stream.state = match stream.state {
                StreamState::Open => StreamState::HalfClosedRemote,
                StreamState::HalfClosedLocal => StreamState::Closed,
                other => other,
            };
        }

        Ok(())
    }

    fn handle_rst(&mut self, frame: Frame) -> Result<(), MuxError> {
        let stream_id = frame.stream_id;
        let stream = self.streams.get_mut(&stream_id)
            .ok_or(MuxError::StreamNotFound { stream_id })?;
        stream.state = StreamState::Closed;
        Ok(())
    }

    fn handle_ping(&mut self, frame: Frame) -> Result<(), MuxError> {
        if !frame.flags.has(FrameFlags::ACK) {
            // Send PING ACK.
            let mut data = [0u8; 8];
            let copy_len = frame.payload.len().min(8);
            data[..copy_len].copy_from_slice(&frame.payload[..copy_len]);
            self.pending_pings.push(data);
            self.outbound.push(Frame::ping(data, true));
        }
        Ok(())
    }

    fn handle_window_update(&mut self, frame: Frame) -> Result<(), MuxError> {
        let increment = frame.window_increment().unwrap_or(0);
        if increment == 0 {
            return Err(MuxError::ProtocolError("zero window increment".into()));
        }

        if frame.stream_id == 0 {
            self.connection_send_window += increment as i64;
        } else {
            let stream = self.streams.get_mut(&frame.stream_id)
                .ok_or(MuxError::StreamNotFound { stream_id: frame.stream_id })?;
            stream.increase_send_window(increment);
        }
        Ok(())
    }

    fn handle_goaway(&mut self, frame: Frame) -> Result<(), MuxError> {
        if frame.payload.len() >= 4 {
            let last_id = u32::from_be_bytes([
                frame.payload[0], frame.payload[1],
                frame.payload[2], frame.payload[3],
            ]);
            // Close all streams with ID > last_id.
            for (id, stream) in &mut self.streams {
                if *id > last_id {
                    stream.state = StreamState::Closed;
                }
            }
        }
        Ok(())
    }

    fn handle_priority(&mut self, frame: Frame) -> Result<(), MuxError> {
        if frame.payload.is_empty() {
            return Ok(());
        }
        if let Some(stream) = self.streams.get_mut(&frame.stream_id) {
            stream.priority = frame.payload[0];
        }
        Ok(())
    }

    /// Drain outbound frames.
    pub fn drain_outbound(&mut self) -> Vec<Frame> {
        std::mem::take(&mut self.outbound)
    }

    /// Number of active (non-closed) streams.
    pub fn active_streams(&self) -> usize {
        self.streams.values()
            .filter(|s| !matches!(s.state, StreamState::Closed | StreamState::Idle))
            .count()
    }

    /// Total streams (including closed).
    pub fn total_streams(&self) -> usize {
        self.streams.len()
    }

    /// Prune closed streams.
    pub fn prune_closed(&mut self) -> usize {
        let before = self.streams.len();
        self.streams.retain(|_, s| s.state != StreamState::Closed);
        before - self.streams.len()
    }

    /// Connection send window.
    pub fn connection_send_window(&self) -> i64 {
        self.connection_send_window
    }

    /// Connection receive window.
    pub fn connection_recv_window(&self) -> i64 {
        self.connection_recv_window
    }

    /// Set maximum concurrent streams.
    pub fn set_max_concurrent_streams(&mut self, max: u32) {
        self.max_concurrent_streams = max;
    }

    /// Set initial window size for new streams.
    pub fn set_initial_window_size(&mut self, size: i64) {
        self.initial_window_size = size;
    }

    /// Whether GOAWAY has been sent.
    pub fn is_goaway_sent(&self) -> bool {
        self.goaway_sent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_stream() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        assert_eq!(id, 1); // Client: odd IDs.
        let id2 = mux.open_stream().unwrap();
        assert_eq!(id2, 3);
    }

    #[test]
    fn test_server_stream_ids() {
        let mut mux = ProtocolMux::new(false);
        let id = mux.open_stream().unwrap();
        assert_eq!(id, 2); // Server: even IDs.
    }

    #[test]
    fn test_send_data() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        mux.send_data(id, vec![1, 2, 3], false).unwrap();
        let frames = mux.drain_outbound();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_type, FrameType::Data);
        assert_eq!(frames[0].payload, vec![1, 2, 3]);
    }

    #[test]
    fn test_send_data_end_stream() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        mux.send_data(id, vec![1, 2, 3], true).unwrap();

        let stream = mux.stream(id).unwrap();
        assert_eq!(stream.state, StreamState::HalfClosedLocal);
    }

    #[test]
    fn test_send_headers() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        mux.send_headers(id, b"headers".to_vec(), false).unwrap();
        let frames = mux.drain_outbound();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_type, FrameType::Headers);
    }

    #[test]
    fn test_recv_data() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        let frame = Frame::data(id, vec![10, 20, 30], false);
        mux.recv_frame(frame).unwrap();

        let stream = mux.stream(id).unwrap();
        assert_eq!(stream.recv_buffer, vec![10, 20, 30]);
    }

    #[test]
    fn test_recv_data_end_stream() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        let frame = Frame::data(id, vec![10], true);
        mux.recv_frame(frame).unwrap();

        let stream = mux.stream(id).unwrap();
        assert_eq!(stream.state, StreamState::HalfClosedRemote);
    }

    #[test]
    fn test_recv_headers_creates_stream() {
        let mut mux = ProtocolMux::new(false);
        // Remote-initiated stream (odd ID from client).
        let frame = Frame::headers(1, b"request".to_vec(), false, true);
        mux.recv_frame(frame).unwrap();

        let stream = mux.stream(1).unwrap();
        assert_eq!(stream.recv_headers, b"request");
        assert_eq!(stream.state, StreamState::Open);
    }

    #[test]
    fn test_rst_stream() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        mux.reset_stream(id, 0).unwrap();

        let stream = mux.stream(id).unwrap();
        assert_eq!(stream.state, StreamState::Closed);
    }

    #[test]
    fn test_recv_rst() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        mux.recv_frame(Frame::rst_stream(id, 2)).unwrap();

        let stream = mux.stream(id).unwrap();
        assert_eq!(stream.state, StreamState::Closed);
    }

    #[test]
    fn test_ping_pong() {
        let mut mux = ProtocolMux::new(true);
        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        let frame = Frame::ping(data, false);
        mux.recv_frame(frame).unwrap();

        let outbound = mux.drain_outbound();
        assert_eq!(outbound.len(), 1);
        assert_eq!(outbound[0].frame_type, FrameType::Ping);
        assert!(outbound[0].flags.has(FrameFlags::ACK));
    }

    #[test]
    fn test_window_update_connection() {
        let mut mux = ProtocolMux::new(true);
        let initial = mux.connection_send_window();
        mux.recv_frame(Frame::window_update(0, 1000)).unwrap();
        assert_eq!(mux.connection_send_window(), initial + 1000);
    }

    #[test]
    fn test_window_update_stream() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        let initial = mux.stream(id).unwrap().send_window;
        mux.recv_frame(Frame::window_update(id, 500)).unwrap();
        assert_eq!(mux.stream(id).unwrap().send_window, initial + 500);
    }

    #[test]
    fn test_flow_control_exhaustion() {
        let mut mux = ProtocolMux::new(true);
        mux.set_initial_window_size(100);
        let id = mux.open_stream().unwrap();

        // Stream's pre-existing window is 65535 (set before config change).
        // Let's manually set it.
        mux.stream_mut(id).unwrap().send_window = 100;

        mux.send_data(id, vec![0; 50], false).unwrap();
        mux.send_data(id, vec![0; 50], false).unwrap();
        let result = mux.send_data(id, vec![0; 1], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_concurrent_streams() {
        let mut mux = ProtocolMux::new(true);
        mux.set_max_concurrent_streams(2);
        mux.open_stream().unwrap();
        mux.open_stream().unwrap();
        let result = mux.open_stream();
        assert!(matches!(result, Err(MuxError::MaxStreamsExceeded { .. })));
    }

    #[test]
    fn test_goaway() {
        let mut mux = ProtocolMux::new(true);
        let id1 = mux.open_stream().unwrap();
        let _id2 = mux.open_stream().unwrap();
        mux.send_goaway(0);
        assert!(mux.is_goaway_sent());

        // No new streams after GOAWAY.
        assert!(mux.open_stream().is_err());

        // Existing streams still work.
        assert!(mux.stream(id1).is_some());
    }

    #[test]
    fn test_recv_goaway_closes_higher_streams() {
        let mut mux = ProtocolMux::new(false);
        // Receive headers for stream 1 and 3.
        mux.recv_frame(Frame::headers(1, vec![], false, true)).unwrap();
        mux.recv_frame(Frame::headers(3, vec![], false, true)).unwrap();

        // GOAWAY with last_stream_id=1 closes stream 3.
        mux.recv_frame(Frame::goaway(1, 0)).unwrap();
        assert_eq!(mux.stream(3).unwrap().state, StreamState::Closed);
        assert_ne!(mux.stream(1).unwrap().state, StreamState::Closed);
    }

    #[test]
    fn test_prune_closed() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();
        mux.reset_stream(id, 0).unwrap();
        assert_eq!(mux.total_streams(), 1);
        let pruned = mux.prune_closed();
        assert_eq!(pruned, 1);
        assert_eq!(mux.total_streams(), 0);
    }

    #[test]
    fn test_active_streams() {
        let mut mux = ProtocolMux::new(true);
        let id1 = mux.open_stream().unwrap();
        mux.open_stream().unwrap();
        assert_eq!(mux.active_streams(), 2);
        mux.reset_stream(id1, 0).unwrap();
        assert_eq!(mux.active_streams(), 1);
    }

    #[test]
    fn test_frame_type_roundtrip() {
        for ft in [FrameType::Data, FrameType::Headers, FrameType::RstStream,
                   FrameType::Settings, FrameType::Ping, FrameType::GoAway,
                   FrameType::WindowUpdate, FrameType::Priority] {
            let id = ft.type_id();
            assert_eq!(FrameType::from_id(id), Some(ft));
        }
    }

    #[test]
    fn test_frame_error_code() {
        let rst = Frame::rst_stream(1, 42);
        assert_eq!(rst.error_code(), Some(42));

        let goaway = Frame::goaway(5, 99);
        assert_eq!(goaway.error_code(), Some(99));
    }

    #[test]
    fn test_stream_state_display() {
        assert_eq!(format!("{}", StreamState::Open), "open");
        assert_eq!(format!("{}", StreamState::Closed), "closed");
    }

    #[test]
    fn test_frame_flags() {
        let mut flags = FrameFlags::default();
        assert!(!flags.has(FrameFlags::END_STREAM));
        flags.set(FrameFlags::END_STREAM);
        assert!(flags.has(FrameFlags::END_STREAM));
        flags.clear(FrameFlags::END_STREAM);
        assert!(!flags.has(FrameFlags::END_STREAM));
    }

    #[test]
    fn test_bidirectional_close() {
        let mut mux = ProtocolMux::new(true);
        let id = mux.open_stream().unwrap();

        // Local sends END_STREAM.
        mux.send_data(id, vec![1], true).unwrap();
        assert_eq!(mux.stream(id).unwrap().state, StreamState::HalfClosedLocal);

        // Remote sends END_STREAM.
        mux.recv_frame(Frame::data(id, vec![2], true)).unwrap();
        assert_eq!(mux.stream(id).unwrap().state, StreamState::Closed);
    }

    #[test]
    fn test_settings_frame() {
        let frame = Frame::settings(&[(1, 4096), (3, 100)], false);
        assert_eq!(frame.frame_type, FrameType::Settings);
        assert_eq!(frame.stream_id, 0);
        assert!(!frame.flags.has(FrameFlags::ACK));

        let ack = Frame::settings(&[], true);
        assert!(ack.flags.has(FrameFlags::ACK));
    }
}
