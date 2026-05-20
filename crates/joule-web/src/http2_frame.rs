//! HTTP/2 framing — frame types (DATA, HEADERS, SETTINGS, PING, GOAWAY,
//! WINDOW_UPDATE, PRIORITY, RST_STREAM, PUSH_PROMISE, CONTINUATION),
//! HPACK header compression basics, stream multiplexing, flow control.
//!
//! Pure-Rust replacement for h2, nghttp2, etc.

use std::collections::HashMap;
use std::fmt;

// ── Frame types ───────────────────────────────────────────────────

/// HTTP/2 frame type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FrameType {
    Data = 0x0,
    Headers = 0x1,
    Priority = 0x2,
    RstStream = 0x3,
    Settings = 0x4,
    PushPromise = 0x5,
    Ping = 0x6,
    Goaway = 0x7,
    WindowUpdate = 0x8,
    Continuation = 0x9,
}

impl FrameType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x0 => Some(Self::Data),
            0x1 => Some(Self::Headers),
            0x2 => Some(Self::Priority),
            0x3 => Some(Self::RstStream),
            0x4 => Some(Self::Settings),
            0x5 => Some(Self::PushPromise),
            0x6 => Some(Self::Ping),
            0x7 => Some(Self::Goaway),
            0x8 => Some(Self::WindowUpdate),
            0x9 => Some(Self::Continuation),
            _ => None,
        }
    }
}

impl fmt::Display for FrameType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data => write!(f, "DATA"),
            Self::Headers => write!(f, "HEADERS"),
            Self::Priority => write!(f, "PRIORITY"),
            Self::RstStream => write!(f, "RST_STREAM"),
            Self::Settings => write!(f, "SETTINGS"),
            Self::PushPromise => write!(f, "PUSH_PROMISE"),
            Self::Ping => write!(f, "PING"),
            Self::Goaway => write!(f, "GOAWAY"),
            Self::WindowUpdate => write!(f, "WINDOW_UPDATE"),
            Self::Continuation => write!(f, "CONTINUATION"),
        }
    }
}

// ── Frame flags ───────────────────────────────────────────────────

/// Common frame flags.
pub mod flags {
    pub const END_STREAM: u8 = 0x1;
    pub const END_HEADERS: u8 = 0x4;
    pub const PADDED: u8 = 0x8;
    pub const PRIORITY_FLAG: u8 = 0x20;
    pub const ACK: u8 = 0x1;
}

// ── Error codes ───────────────────────────────────────────────────

/// HTTP/2 error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ErrorCode {
    NoError = 0x0,
    ProtocolError = 0x1,
    InternalError = 0x2,
    FlowControlError = 0x3,
    SettingsTimeout = 0x4,
    StreamClosed = 0x5,
    FrameSizeError = 0x6,
    RefusedStream = 0x7,
    Cancel = 0x8,
    CompressionError = 0x9,
    ConnectError = 0xa,
    EnhanceYourCalm = 0xb,
    InadequateSecurity = 0xc,
    Http11Required = 0xd,
}

impl ErrorCode {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0x0 => Some(Self::NoError),
            0x1 => Some(Self::ProtocolError),
            0x2 => Some(Self::InternalError),
            0x3 => Some(Self::FlowControlError),
            0x4 => Some(Self::SettingsTimeout),
            0x5 => Some(Self::StreamClosed),
            0x6 => Some(Self::FrameSizeError),
            0x7 => Some(Self::RefusedStream),
            0x8 => Some(Self::Cancel),
            0x9 => Some(Self::CompressionError),
            0xa => Some(Self::ConnectError),
            0xb => Some(Self::EnhanceYourCalm),
            0xc => Some(Self::InadequateSecurity),
            0xd => Some(Self::Http11Required),
            _ => None,
        }
    }
}

// ── Frame ─────────────────────────────────────────────────────────

/// A raw HTTP/2 frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub frame_type: FrameType,
    pub flags: u8,
    pub stream_id: u32,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(frame_type: FrameType, stream_id: u32, payload: Vec<u8>) -> Self {
        Self { frame_type, flags: 0, stream_id, payload }
    }

    pub fn with_flags(mut self, flags: u8) -> Self {
        self.flags = flags;
        self
    }

    /// 9-byte frame header: length (3) + type (1) + flags (1) + stream_id (4).
    pub fn encode_header(&self) -> [u8; 9] {
        let len = self.payload.len() as u32;
        let mut hdr = [0u8; 9];
        hdr[0] = ((len >> 16) & 0xFF) as u8;
        hdr[1] = ((len >> 8) & 0xFF) as u8;
        hdr[2] = (len & 0xFF) as u8;
        hdr[3] = self.frame_type as u8;
        hdr[4] = self.flags;
        let sid = self.stream_id & 0x7FFF_FFFF; // clear reserved bit
        hdr[5] = ((sid >> 24) & 0xFF) as u8;
        hdr[6] = ((sid >> 16) & 0xFF) as u8;
        hdr[7] = ((sid >> 8) & 0xFF) as u8;
        hdr[8] = (sid & 0xFF) as u8;
        hdr
    }

    /// Encode the full frame (header + payload).
    pub fn encode(&self) -> Vec<u8> {
        let hdr = self.encode_header();
        let mut out = Vec::with_capacity(9 + self.payload.len());
        out.extend_from_slice(&hdr);
        out.extend_from_slice(&self.payload);
        out
    }

    /// Decode a frame from bytes. Returns (frame, bytes_consumed) or None.
    pub fn decode(data: &[u8]) -> Option<(Self, usize)> {
        if data.len() < 9 { return None; }
        let len = ((data[0] as u32) << 16) | ((data[1] as u32) << 8) | (data[2] as u32);
        let total = 9 + len as usize;
        if data.len() < total { return None; }

        let frame_type = FrameType::from_u8(data[3])?;
        let flags = data[4];
        let stream_id = ((data[5] as u32 & 0x7F) << 24)
            | ((data[6] as u32) << 16)
            | ((data[7] as u32) << 8)
            | (data[8] as u32);
        let payload = data[9..total].to_vec();

        Some((Self { frame_type, flags, stream_id, payload }, total))
    }

    pub fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }
}

// ── Typed frame constructors ──────────────────────────────────────

/// Construct a DATA frame.
pub fn data_frame(stream_id: u32, data: &[u8], end_stream: bool) -> Frame {
    let mut f = Frame::new(FrameType::Data, stream_id, data.to_vec());
    if end_stream { f.flags |= flags::END_STREAM; }
    f
}

/// Construct a HEADERS frame (with raw header block).
pub fn headers_frame(stream_id: u32, header_block: &[u8], end_stream: bool, end_headers: bool) -> Frame {
    let mut f = Frame::new(FrameType::Headers, stream_id, header_block.to_vec());
    if end_stream { f.flags |= flags::END_STREAM; }
    if end_headers { f.flags |= flags::END_HEADERS; }
    f
}

/// Construct a SETTINGS frame.
pub fn settings_frame(settings: &[(u16, u32)]) -> Frame {
    let mut payload = Vec::with_capacity(settings.len() * 6);
    for &(id, val) in settings {
        payload.push((id >> 8) as u8);
        payload.push(id as u8);
        payload.push((val >> 24) as u8);
        payload.push((val >> 16) as u8);
        payload.push((val >> 8) as u8);
        payload.push(val as u8);
    }
    Frame::new(FrameType::Settings, 0, payload)
}

/// Construct a SETTINGS ACK frame.
pub fn settings_ack() -> Frame {
    Frame::new(FrameType::Settings, 0, Vec::new()).with_flags(flags::ACK)
}

/// Construct a PING frame.
pub fn ping_frame(opaque_data: [u8; 8]) -> Frame {
    Frame::new(FrameType::Ping, 0, opaque_data.to_vec())
}

/// Construct a PING ACK.
pub fn ping_ack(opaque_data: [u8; 8]) -> Frame {
    Frame::new(FrameType::Ping, 0, opaque_data.to_vec()).with_flags(flags::ACK)
}

/// Construct a GOAWAY frame.
pub fn goaway_frame(last_stream_id: u32, error_code: ErrorCode, debug_data: &[u8]) -> Frame {
    let mut payload = Vec::with_capacity(8 + debug_data.len());
    let sid = last_stream_id & 0x7FFF_FFFF;
    payload.push((sid >> 24) as u8);
    payload.push((sid >> 16) as u8);
    payload.push((sid >> 8) as u8);
    payload.push(sid as u8);
    let ec = error_code as u32;
    payload.push((ec >> 24) as u8);
    payload.push((ec >> 16) as u8);
    payload.push((ec >> 8) as u8);
    payload.push(ec as u8);
    payload.extend_from_slice(debug_data);
    Frame::new(FrameType::Goaway, 0, payload)
}

/// Construct a WINDOW_UPDATE frame.
pub fn window_update_frame(stream_id: u32, increment: u32) -> Frame {
    let inc = increment & 0x7FFF_FFFF;
    let payload = vec![
        (inc >> 24) as u8, (inc >> 16) as u8,
        (inc >> 8) as u8, inc as u8,
    ];
    Frame::new(FrameType::WindowUpdate, stream_id, payload)
}

/// Construct a RST_STREAM frame.
pub fn rst_stream_frame(stream_id: u32, error_code: ErrorCode) -> Frame {
    let ec = error_code as u32;
    let payload = vec![
        (ec >> 24) as u8, (ec >> 16) as u8,
        (ec >> 8) as u8, ec as u8,
    ];
    Frame::new(FrameType::RstStream, stream_id, payload)
}

// ── Settings ──────────────────────────────────────────────────────

/// Well-known settings identifiers.
pub mod setting_id {
    pub const HEADER_TABLE_SIZE: u16 = 0x1;
    pub const ENABLE_PUSH: u16 = 0x2;
    pub const MAX_CONCURRENT_STREAMS: u16 = 0x3;
    pub const INITIAL_WINDOW_SIZE: u16 = 0x4;
    pub const MAX_FRAME_SIZE: u16 = 0x5;
    pub const MAX_HEADER_LIST_SIZE: u16 = 0x6;
}

/// Parse a SETTINGS frame payload into (id, value) pairs.
pub fn parse_settings(payload: &[u8]) -> Vec<(u16, u32)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i + 6 <= payload.len() {
        let id = ((payload[i] as u16) << 8) | (payload[i + 1] as u16);
        let val = ((payload[i + 2] as u32) << 24)
            | ((payload[i + 3] as u32) << 16)
            | ((payload[i + 4] as u32) << 8)
            | (payload[i + 5] as u32);
        out.push((id, val));
        i += 6;
    }
    out
}

// ── Flow control ──────────────────────────────────────────────────

/// Per-stream or connection flow control window.
#[derive(Debug, Clone)]
pub struct FlowWindow {
    window_size: i64,
}

impl FlowWindow {
    /// Default initial window size per RFC 9113: 65535.
    pub fn new() -> Self { Self { window_size: 65535 } }

    pub fn with_size(size: i64) -> Self { Self { window_size: size } }

    pub fn available(&self) -> i64 { self.window_size }

    /// Consume `n` bytes from the window. Returns false if insufficient.
    pub fn consume(&mut self, n: u32) -> bool {
        let n = n as i64;
        if self.window_size < n { return false; }
        self.window_size -= n;
        true
    }

    /// Add `increment` bytes to the window (from WINDOW_UPDATE).
    pub fn replenish(&mut self, increment: u32) -> Result<(), &'static str> {
        let new_size = self.window_size + increment as i64;
        if new_size > 0x7FFF_FFFF {
            return Err("flow control window overflow");
        }
        self.window_size = new_size;
        Ok(())
    }

    /// Update window size when SETTINGS changes initial window size.
    pub fn adjust(&mut self, delta: i64) {
        self.window_size += delta;
    }
}

impl Default for FlowWindow {
    fn default() -> Self { Self::new() }
}

// ── Stream state ──────────────────────────────────────────────────

/// HTTP/2 stream state per RFC 9113 section 5.1.
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

/// A single HTTP/2 stream.
#[derive(Debug, Clone)]
pub struct Stream {
    pub id: u32,
    pub state: StreamState,
    pub send_window: FlowWindow,
    pub recv_window: FlowWindow,
    pub weight: u8,
    pub dependency: u32,
    pub exclusive: bool,
}

impl Stream {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            state: StreamState::Idle,
            send_window: FlowWindow::new(),
            recv_window: FlowWindow::new(),
            weight: 16,
            dependency: 0,
            exclusive: false,
        }
    }

    pub fn is_client_initiated(&self) -> bool { self.id % 2 == 1 }
    pub fn is_server_initiated(&self) -> bool { self.id % 2 == 0 && self.id > 0 }
}

// ── Stream multiplexer ───────────────────────────────────────────

/// Manages multiple HTTP/2 streams on a connection.
#[derive(Debug)]
pub struct StreamMultiplexer {
    streams: HashMap<u32, Stream>,
    next_client_id: u32,
    next_server_id: u32,
    max_concurrent: u32,
    conn_send_window: FlowWindow,
    conn_recv_window: FlowWindow,
}

impl StreamMultiplexer {
    pub fn new_client() -> Self {
        Self {
            streams: HashMap::new(),
            next_client_id: 1,
            next_server_id: 2,
            max_concurrent: 100,
            conn_send_window: FlowWindow::new(),
            conn_recv_window: FlowWindow::new(),
        }
    }

    pub fn new_server() -> Self {
        Self {
            streams: HashMap::new(),
            next_client_id: 1,
            next_server_id: 2,
            max_concurrent: 100,
            conn_send_window: FlowWindow::new(),
            conn_recv_window: FlowWindow::new(),
        }
    }

    pub fn set_max_concurrent(&mut self, max: u32) { self.max_concurrent = max; }

    /// Open a new client-initiated stream.
    pub fn open_client_stream(&mut self) -> Option<u32> {
        let active = self.active_count();
        if active >= self.max_concurrent { return None; }
        let id = self.next_client_id;
        self.next_client_id += 2;
        let mut stream = Stream::new(id);
        stream.state = StreamState::Open;
        self.streams.insert(id, stream);
        Some(id)
    }

    /// Open a new server-initiated stream.
    pub fn open_server_stream(&mut self) -> Option<u32> {
        let active = self.active_count();
        if active >= self.max_concurrent { return None; }
        let id = self.next_server_id;
        self.next_server_id += 2;
        let mut stream = Stream::new(id);
        stream.state = StreamState::Open;
        self.streams.insert(id, stream);
        Some(id)
    }

    pub fn get_stream(&self, id: u32) -> Option<&Stream> {
        self.streams.get(&id)
    }

    pub fn get_stream_mut(&mut self, id: u32) -> Option<&mut Stream> {
        self.streams.get_mut(&id)
    }

    /// Count of streams in open/half-closed states.
    pub fn active_count(&self) -> u32 {
        self.streams.values().filter(|s| matches!(
            s.state,
            StreamState::Open | StreamState::HalfClosedLocal | StreamState::HalfClosedRemote
        )).count() as u32
    }

    /// Close a stream.
    pub fn close_stream(&mut self, id: u32) {
        if let Some(s) = self.streams.get_mut(&id) {
            s.state = StreamState::Closed;
        }
    }

    /// Half-close the local side.
    pub fn half_close_local(&mut self, id: u32) {
        if let Some(s) = self.streams.get_mut(&id) {
            match s.state {
                StreamState::Open => s.state = StreamState::HalfClosedLocal,
                StreamState::HalfClosedRemote => s.state = StreamState::Closed,
                _ => {}
            }
        }
    }

    /// Half-close the remote side.
    pub fn half_close_remote(&mut self, id: u32) {
        if let Some(s) = self.streams.get_mut(&id) {
            match s.state {
                StreamState::Open => s.state = StreamState::HalfClosedRemote,
                StreamState::HalfClosedLocal => s.state = StreamState::Closed,
                _ => {}
            }
        }
    }

    pub fn connection_send_window(&self) -> &FlowWindow { &self.conn_send_window }
    pub fn connection_recv_window(&self) -> &FlowWindow { &self.conn_recv_window }
    pub fn connection_send_window_mut(&mut self) -> &mut FlowWindow { &mut self.conn_send_window }
    pub fn connection_recv_window_mut(&mut self) -> &mut FlowWindow { &mut self.conn_recv_window }

    /// Number of tracked streams (including closed).
    pub fn total_streams(&self) -> usize { self.streams.len() }

    /// Remove closed streams to free memory.
    pub fn prune_closed(&mut self) {
        self.streams.retain(|_, s| s.state != StreamState::Closed);
    }
}

// ── HPACK basics ──────────────────────────────────────────────────

/// Simplified HPACK static table (first 15 entries per RFC 7541).
static STATIC_TABLE: &[(&str, &str)] = &[
    (":authority", ""),
    (":method", "GET"),
    (":method", "POST"),
    (":path", "/"),
    (":path", "/index.html"),
    (":scheme", "http"),
    (":scheme", "https"),
    (":status", "200"),
    (":status", "204"),
    (":status", "206"),
    (":status", "304"),
    (":status", "400"),
    (":status", "404"),
    (":status", "500"),
    ("accept-charset", ""),
];

/// Look up a header in the HPACK static table. Returns (index, has_value).
pub fn hpack_static_lookup(name: &str, value: &str) -> Option<(usize, bool)> {
    let mut name_match = None;
    for (i, &(n, v)) in STATIC_TABLE.iter().enumerate() {
        if n == name {
            if v == value {
                return Some((i + 1, true)); // 1-indexed
            }
            if name_match.is_none() {
                name_match = Some(i + 1);
            }
        }
    }
    name_match.map(|idx| (idx, false))
}

/// Encode an integer with the HPACK integer encoding.
pub fn hpack_encode_integer(value: u64, prefix_bits: u8) -> Vec<u8> {
    let max_prefix = (1u64 << prefix_bits) - 1;
    if value < max_prefix {
        return vec![value as u8];
    }
    let mut out = vec![max_prefix as u8];
    let mut remaining = value - max_prefix;
    while remaining >= 128 {
        out.push((remaining & 0x7F | 0x80) as u8);
        remaining >>= 7;
    }
    out.push(remaining as u8);
    out
}

/// Decode an HPACK integer from a byte slice. Returns (value, bytes_consumed).
pub fn hpack_decode_integer(data: &[u8], prefix_bits: u8) -> Option<(u64, usize)> {
    if data.is_empty() { return None; }
    let max_prefix = (1u64 << prefix_bits) - 1;
    let first = (data[0] as u64) & max_prefix;
    if first < max_prefix {
        return Some((first, 1));
    }
    let mut value = max_prefix;
    let mut shift = 0u64;
    let mut i = 1;
    loop {
        if i >= data.len() { return None; }
        let byte = data[i] as u64;
        value += (byte & 0x7F) << shift;
        shift += 7;
        i += 1;
        if byte & 0x80 == 0 { break; }
        if shift > 63 { return None; } // overflow protection
    }
    Some((value, i))
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_type_from_u8() {
        assert_eq!(FrameType::from_u8(0x0), Some(FrameType::Data));
        assert_eq!(FrameType::from_u8(0x1), Some(FrameType::Headers));
        assert_eq!(FrameType::from_u8(0x7), Some(FrameType::Goaway));
        assert_eq!(FrameType::from_u8(0xFF), None);
    }

    #[test]
    fn frame_type_display() {
        assert_eq!(FrameType::Data.to_string(), "DATA");
        assert_eq!(FrameType::Settings.to_string(), "SETTINGS");
        assert_eq!(FrameType::WindowUpdate.to_string(), "WINDOW_UPDATE");
    }

    #[test]
    fn error_code_roundtrip() {
        for ec in [ErrorCode::NoError, ErrorCode::ProtocolError, ErrorCode::Cancel,
                    ErrorCode::Http11Required] {
            let val = ec as u32;
            assert_eq!(ErrorCode::from_u32(val), Some(ec));
        }
        assert_eq!(ErrorCode::from_u32(999), None);
    }

    #[test]
    fn frame_encode_decode_data() {
        let f = data_frame(1, b"Hello", true);
        let encoded = f.encode();
        assert_eq!(encoded.len(), 9 + 5);

        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(consumed, 14);
        assert_eq!(decoded.frame_type, FrameType::Data);
        assert_eq!(decoded.stream_id, 1);
        assert_eq!(decoded.payload, b"Hello");
        assert!(decoded.has_flag(flags::END_STREAM));
    }

    #[test]
    fn frame_encode_decode_headers() {
        let f = headers_frame(3, b"header-block", false, true);
        let encoded = f.encode();
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, FrameType::Headers);
        assert_eq!(decoded.stream_id, 3);
        assert!(decoded.has_flag(flags::END_HEADERS));
        assert!(!decoded.has_flag(flags::END_STREAM));
    }

    #[test]
    fn settings_frame_encode_decode() {
        let f = settings_frame(&[
            (setting_id::MAX_CONCURRENT_STREAMS, 128),
            (setting_id::INITIAL_WINDOW_SIZE, 1048576),
        ]);
        assert_eq!(f.payload.len(), 12);
        let settings = parse_settings(&f.payload);
        assert_eq!(settings.len(), 2);
        assert_eq!(settings[0], (setting_id::MAX_CONCURRENT_STREAMS, 128));
        assert_eq!(settings[1], (setting_id::INITIAL_WINDOW_SIZE, 1048576));
    }

    #[test]
    fn settings_ack_frame() {
        let f = settings_ack();
        assert_eq!(f.frame_type, FrameType::Settings);
        assert!(f.has_flag(flags::ACK));
        assert!(f.payload.is_empty());
    }

    #[test]
    fn ping_frame_roundtrip() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        let f = ping_frame(data);
        let encoded = f.encode();
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, FrameType::Ping);
        assert_eq!(decoded.payload.len(), 8);
        assert!(!decoded.has_flag(flags::ACK));
    }

    #[test]
    fn ping_ack_frame() {
        let data = [0; 8];
        let f = ping_ack(data);
        assert!(f.has_flag(flags::ACK));
    }

    #[test]
    fn goaway_frame_fields() {
        let f = goaway_frame(5, ErrorCode::NoError, b"bye");
        let encoded = f.encode();
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, FrameType::Goaway);
        assert_eq!(decoded.stream_id, 0);
        // Parse last_stream_id from payload
        let last = ((decoded.payload[0] as u32 & 0x7F) << 24)
            | ((decoded.payload[1] as u32) << 16)
            | ((decoded.payload[2] as u32) << 8)
            | (decoded.payload[3] as u32);
        assert_eq!(last, 5);
    }

    #[test]
    fn window_update_frame_test() {
        let f = window_update_frame(1, 32768);
        let encoded = f.encode();
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, FrameType::WindowUpdate);
        assert_eq!(decoded.payload.len(), 4);
        let inc = ((decoded.payload[0] as u32 & 0x7F) << 24)
            | ((decoded.payload[1] as u32) << 16)
            | ((decoded.payload[2] as u32) << 8)
            | (decoded.payload[3] as u32);
        assert_eq!(inc, 32768);
    }

    #[test]
    fn rst_stream_frame_test() {
        let f = rst_stream_frame(3, ErrorCode::Cancel);
        let encoded = f.encode();
        let (decoded, _) = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.frame_type, FrameType::RstStream);
        assert_eq!(decoded.stream_id, 3);
    }

    #[test]
    fn frame_decode_insufficient_data() {
        assert!(Frame::decode(b"short").is_none());
        assert!(Frame::decode(&[0, 0, 10, 0, 0, 0, 0, 0, 0]).is_none()); // claims 10 bytes but only header
    }

    #[test]
    fn flow_window_basic() {
        let mut w = FlowWindow::new();
        assert_eq!(w.available(), 65535);
        assert!(w.consume(1000));
        assert_eq!(w.available(), 64535);
        w.replenish(500).unwrap();
        assert_eq!(w.available(), 65035);
    }

    #[test]
    fn flow_window_insufficient() {
        let mut w = FlowWindow::with_size(100);
        assert!(!w.consume(200));
        assert_eq!(w.available(), 100); // unchanged
    }

    #[test]
    fn flow_window_overflow() {
        let mut w = FlowWindow::with_size(0x7FFF_FFFE);
        let result = w.replenish(2);
        assert!(result.is_err());
    }

    #[test]
    fn flow_window_adjust() {
        let mut w = FlowWindow::new();
        w.adjust(-100);
        assert_eq!(w.available(), 65435);
        w.adjust(200);
        assert_eq!(w.available(), 65635);
    }

    #[test]
    fn stream_basic() {
        let s = Stream::new(1);
        assert!(s.is_client_initiated());
        assert!(!s.is_server_initiated());
        assert_eq!(s.state, StreamState::Idle);
        assert_eq!(s.weight, 16);
    }

    #[test]
    fn stream_server_initiated() {
        let s = Stream::new(2);
        assert!(s.is_server_initiated());
        assert!(!s.is_client_initiated());
    }

    #[test]
    fn multiplexer_client() {
        let mut mux = StreamMultiplexer::new_client();
        let id1 = mux.open_client_stream().unwrap();
        let id2 = mux.open_client_stream().unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 3);
        assert_eq!(mux.active_count(), 2);
    }

    #[test]
    fn multiplexer_server() {
        let mut mux = StreamMultiplexer::new_server();
        let id1 = mux.open_server_stream().unwrap();
        let id2 = mux.open_server_stream().unwrap();
        assert_eq!(id1, 2);
        assert_eq!(id2, 4);
    }

    #[test]
    fn multiplexer_max_concurrent() {
        let mut mux = StreamMultiplexer::new_client();
        mux.set_max_concurrent(2);
        assert!(mux.open_client_stream().is_some());
        assert!(mux.open_client_stream().is_some());
        assert!(mux.open_client_stream().is_none());
    }

    #[test]
    fn multiplexer_close_stream() {
        let mut mux = StreamMultiplexer::new_client();
        let id = mux.open_client_stream().unwrap();
        assert_eq!(mux.active_count(), 1);
        mux.close_stream(id);
        assert_eq!(mux.active_count(), 0);
        assert_eq!(mux.get_stream(id).unwrap().state, StreamState::Closed);
    }

    #[test]
    fn multiplexer_half_close() {
        let mut mux = StreamMultiplexer::new_client();
        let id = mux.open_client_stream().unwrap();
        mux.half_close_local(id);
        assert_eq!(mux.get_stream(id).unwrap().state, StreamState::HalfClosedLocal);
        assert_eq!(mux.active_count(), 1);
        mux.half_close_remote(id);
        assert_eq!(mux.get_stream(id).unwrap().state, StreamState::Closed);
        assert_eq!(mux.active_count(), 0);
    }

    #[test]
    fn multiplexer_prune_closed() {
        let mut mux = StreamMultiplexer::new_client();
        let id = mux.open_client_stream().unwrap();
        mux.open_client_stream().unwrap();
        mux.close_stream(id);
        assert_eq!(mux.total_streams(), 2);
        mux.prune_closed();
        assert_eq!(mux.total_streams(), 1);
    }

    #[test]
    fn hpack_static_lookup_exact() {
        let (idx, has_val) = hpack_static_lookup(":method", "GET").unwrap();
        assert_eq!(idx, 2);
        assert!(has_val);
    }

    #[test]
    fn hpack_static_lookup_name_only() {
        let (idx, has_val) = hpack_static_lookup(":method", "DELETE").unwrap();
        assert_eq!(idx, 2); // name match
        assert!(!has_val);
    }

    #[test]
    fn hpack_static_lookup_miss() {
        assert!(hpack_static_lookup("x-custom", "value").is_none());
    }

    #[test]
    fn hpack_integer_small() {
        let encoded = hpack_encode_integer(10, 5);
        assert_eq!(encoded, vec![10]);
        let (val, consumed) = hpack_decode_integer(&encoded, 5).unwrap();
        assert_eq!(val, 10);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn hpack_integer_large() {
        let encoded = hpack_encode_integer(1337, 5);
        // 31 in first byte, then multi-byte
        assert_eq!(encoded[0], 31);
        let (val, consumed) = hpack_decode_integer(&encoded, 5).unwrap();
        assert_eq!(val, 1337);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn hpack_integer_exact_prefix() {
        // Value exactly at prefix boundary
        let encoded = hpack_encode_integer(31, 5);
        let (val, _) = hpack_decode_integer(&encoded, 5).unwrap();
        assert_eq!(val, 31);
    }

    #[test]
    fn hpack_integer_zero() {
        let encoded = hpack_encode_integer(0, 7);
        assert_eq!(encoded, vec![0]);
        let (val, consumed) = hpack_decode_integer(&encoded, 7).unwrap();
        assert_eq!(val, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn decode_empty_returns_none() {
        assert!(hpack_decode_integer(&[], 5).is_none());
    }

    #[test]
    fn frame_stream_id_clears_reserved_bit() {
        let f = Frame::new(FrameType::Data, 0x8000_0001, Vec::new());
        let hdr = f.encode_header();
        // Top bit of stream_id must be clear
        assert_eq!(hdr[5] & 0x80, 0);
    }

    #[test]
    fn connection_flow_window() {
        let mut mux = StreamMultiplexer::new_client();
        assert_eq!(mux.connection_send_window().available(), 65535);
        assert!(mux.connection_send_window_mut().consume(100));
        assert_eq!(mux.connection_recv_window().available(), 65535);
    }

    #[test]
    fn empty_settings_frame() {
        let f = settings_frame(&[]);
        assert!(f.payload.is_empty());
        let settings = parse_settings(&f.payload);
        assert!(settings.is_empty());
    }

    #[test]
    fn multiple_frames_in_buffer() {
        let f1 = data_frame(1, b"A", false);
        let f2 = data_frame(1, b"B", true);
        let mut buf = f1.encode();
        buf.extend_from_slice(&f2.encode());

        let (d1, c1) = Frame::decode(&buf).unwrap();
        assert_eq!(d1.payload, b"A");
        let (d2, _) = Frame::decode(&buf[c1..]).unwrap();
        assert_eq!(d2.payload, b"B");
        assert!(d2.has_flag(flags::END_STREAM));
    }
}
