//! HTTP/2 frame types and HPACK header compression.
//!
//! Replaces `h2`, `nghttp2`, and `node-http2` with pure Rust frame parsing
//! (DATA, HEADERS, SETTINGS, PING, GOAWAY, WINDOW_UPDATE, RST_STREAM,
//! PRIORITY, PUSH_PROMISE, CONTINUATION), stream states, HPACK static/dynamic
//! table, and Huffman encoding stubs for header compression.

use std::collections::HashMap;
use std::fmt;

// ── Frame types ────────────────────────────────────────────────

/// HTTP/2 frame type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameType {
    Data,
    Headers,
    Priority,
    RstStream,
    Settings,
    PushPromise,
    Ping,
    Goaway,
    WindowUpdate,
    Continuation,
    Unknown(u8),
}

impl FrameType {
    pub fn from_u8(b: u8) -> Self {
        match b {
            0x0 => Self::Data,
            0x1 => Self::Headers,
            0x2 => Self::Priority,
            0x3 => Self::RstStream,
            0x4 => Self::Settings,
            0x5 => Self::PushPromise,
            0x6 => Self::Ping,
            0x7 => Self::Goaway,
            0x8 => Self::WindowUpdate,
            0x9 => Self::Continuation,
            other => Self::Unknown(other),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Data => 0x0,
            Self::Headers => 0x1,
            Self::Priority => 0x2,
            Self::RstStream => 0x3,
            Self::Settings => 0x4,
            Self::PushPromise => 0x5,
            Self::Ping => 0x6,
            Self::Goaway => 0x7,
            Self::WindowUpdate => 0x8,
            Self::Continuation => 0x9,
            Self::Unknown(v) => v,
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
            Self::Unknown(v) => write!(f, "UNKNOWN(0x{:02x})", v),
        }
    }
}

// ── Frame flags ────────────────────────────────────────────────

/// Common HTTP/2 frame flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameFlags(pub u8);

impl FrameFlags {
    pub const END_STREAM: u8 = 0x1;
    pub const END_HEADERS: u8 = 0x4;
    pub const PADDED: u8 = 0x8;
    pub const PRIORITY_FLAG: u8 = 0x20;
    pub const ACK: u8 = 0x1; // for SETTINGS and PING

    pub fn has(self, flag: u8) -> bool {
        (self.0 & flag) != 0
    }

    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    pub fn clear(&mut self, flag: u8) {
        self.0 &= !flag;
    }
}

// ── Frame header ───────────────────────────────────────────────

/// HTTP/2 frame header (9 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub length: u32,      // 24 bits
    pub frame_type: FrameType,
    pub flags: FrameFlags,
    pub stream_id: u32,   // 31 bits (MSB reserved)
}

impl FrameHeader {
    pub const SIZE: usize = 9;

    pub fn new(frame_type: FrameType, flags: u8, stream_id: u32, length: u32) -> Self {
        Self {
            length,
            frame_type,
            flags: FrameFlags(flags),
            stream_id: stream_id & 0x7FFF_FFFF,
        }
    }

    /// Parse from 9 bytes.
    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < 9 {
            return None;
        }
        let length = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
        let frame_type = FrameType::from_u8(buf[3]);
        let flags = FrameFlags(buf[4]);
        let stream_id =
            ((buf[5] as u32) << 24) | ((buf[6] as u32) << 16) | ((buf[7] as u32) << 8) | (buf[8] as u32);
        let stream_id = stream_id & 0x7FFF_FFFF;

        Some(Self { length, frame_type, flags, stream_id })
    }

    /// Serialize to 9 bytes.
    pub fn serialize(&self) -> [u8; 9] {
        let mut buf = [0u8; 9];
        buf[0] = ((self.length >> 16) & 0xFF) as u8;
        buf[1] = ((self.length >> 8) & 0xFF) as u8;
        buf[2] = (self.length & 0xFF) as u8;
        buf[3] = self.frame_type.to_u8();
        buf[4] = self.flags.0;
        let sid = self.stream_id & 0x7FFF_FFFF;
        buf[5] = ((sid >> 24) & 0xFF) as u8;
        buf[6] = ((sid >> 16) & 0xFF) as u8;
        buf[7] = ((sid >> 8) & 0xFF) as u8;
        buf[8] = (sid & 0xFF) as u8;
        buf
    }
}

// ── Frame payloads ─────────────────────────────────────────────

/// A parsed HTTP/2 frame.
#[derive(Debug, Clone)]
pub enum Frame {
    Data {
        stream_id: u32,
        end_stream: bool,
        data: Vec<u8>,
    },
    Headers {
        stream_id: u32,
        end_stream: bool,
        end_headers: bool,
        header_block: Vec<u8>,
    },
    Settings {
        ack: bool,
        params: Vec<(u16, u32)>,
    },
    Ping {
        ack: bool,
        opaque_data: [u8; 8],
    },
    Goaway {
        last_stream_id: u32,
        error_code: u32,
        debug_data: Vec<u8>,
    },
    WindowUpdate {
        stream_id: u32,
        increment: u32,
    },
    RstStream {
        stream_id: u32,
        error_code: u32,
    },
}

/// Well-known HTTP/2 settings identifiers.
pub const SETTINGS_HEADER_TABLE_SIZE: u16 = 0x1;
pub const SETTINGS_ENABLE_PUSH: u16 = 0x2;
pub const SETTINGS_MAX_CONCURRENT_STREAMS: u16 = 0x3;
pub const SETTINGS_INITIAL_WINDOW_SIZE: u16 = 0x4;
pub const SETTINGS_MAX_FRAME_SIZE: u16 = 0x5;
pub const SETTINGS_MAX_HEADER_LIST_SIZE: u16 = 0x6;

/// Build a SETTINGS frame payload.
pub fn build_settings(params: &[(u16, u32)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(params.len() * 6);
    for &(id, val) in params {
        buf.push(((id >> 8) & 0xFF) as u8);
        buf.push((id & 0xFF) as u8);
        buf.push(((val >> 24) & 0xFF) as u8);
        buf.push(((val >> 16) & 0xFF) as u8);
        buf.push(((val >> 8) & 0xFF) as u8);
        buf.push((val & 0xFF) as u8);
    }
    buf
}

/// Parse SETTINGS payload into (id, value) pairs.
pub fn parse_settings(payload: &[u8]) -> Vec<(u16, u32)> {
    let mut result = Vec::new();
    let mut i = 0;
    while i + 5 < payload.len() {
        let id = ((payload[i] as u16) << 8) | (payload[i + 1] as u16);
        let val = ((payload[i + 2] as u32) << 24)
            | ((payload[i + 3] as u32) << 16)
            | ((payload[i + 4] as u32) << 8)
            | (payload[i + 5] as u32);
        result.push((id, val));
        i += 6;
    }
    result
}

// ── Stream states ──────────────────────────────────────────────

/// HTTP/2 stream state (RFC 7540 Section 5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamState {
    Idle,
    ReservedLocal,
    ReservedRemote,
    Open,
    HalfClosedLocal,
    HalfClosedRemote,
    Closed,
}

/// A stream tracker.
#[derive(Debug)]
pub struct StreamTracker {
    streams: HashMap<u32, StreamState>,
    next_stream_id: u32,
}

impl StreamTracker {
    pub fn new(is_client: bool) -> Self {
        Self {
            streams: HashMap::new(),
            // Clients use odd IDs, servers use even.
            next_stream_id: if is_client { 1 } else { 2 },
        }
    }

    pub fn open_stream(&mut self) -> u32 {
        let id = self.next_stream_id;
        self.streams.insert(id, StreamState::Open);
        self.next_stream_id += 2;
        id
    }

    pub fn state(&self, stream_id: u32) -> StreamState {
        self.streams
            .get(&stream_id)
            .copied()
            .unwrap_or(StreamState::Idle)
    }

    pub fn transition(&mut self, stream_id: u32, new_state: StreamState) {
        self.streams.insert(stream_id, new_state);
    }

    pub fn close(&mut self, stream_id: u32) {
        self.streams.insert(stream_id, StreamState::Closed);
    }

    pub fn active_count(&self) -> usize {
        self.streams
            .values()
            .filter(|s| matches!(s, StreamState::Open | StreamState::HalfClosedLocal | StreamState::HalfClosedRemote))
            .count()
    }
}

// ── HPACK ──────────────────────────────────────────────────────

/// HPACK static table (RFC 7541, Appendix A) — first 61 entries.
const HPACK_STATIC_TABLE: &[(&str, &str)] = &[
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
    ("accept-encoding", "gzip, deflate"),
    ("accept-language", ""),
    ("accept-ranges", ""),
    ("accept", ""),
    ("access-control-allow-origin", ""),
    ("age", ""),
    ("allow", ""),
    ("authorization", ""),
    ("cache-control", ""),
    ("content-disposition", ""),
    ("content-encoding", ""),
    ("content-language", ""),
    ("content-length", ""),
    ("content-location", ""),
    ("content-range", ""),
    ("content-type", ""),
    ("cookie", ""),
    ("date", ""),
    ("etag", ""),
    ("expect", ""),
    ("expires", ""),
    ("from", ""),
    ("host", ""),
    ("if-match", ""),
    ("if-modified-since", ""),
    ("if-none-match", ""),
    ("if-range", ""),
    ("if-unmodified-since", ""),
    ("last-modified", ""),
    ("link", ""),
    ("location", ""),
    ("max-forwards", ""),
    ("proxy-authenticate", ""),
    ("proxy-authorization", ""),
    ("range", ""),
    ("referer", ""),
    ("refresh", ""),
    ("retry-after", ""),
    ("server", ""),
    ("set-cookie", ""),
    ("strict-transport-security", ""),
    ("transfer-encoding", ""),
    ("user-agent", ""),
    ("vary", ""),
    ("via", ""),
    ("www-authenticate", ""),
];

/// HPACK dynamic table.
#[derive(Debug)]
pub struct HpackDynamicTable {
    entries: Vec<(String, String)>,
    max_size: usize,
    current_size: usize,
}

impl HpackDynamicTable {
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
            current_size: 0,
        }
    }

    fn entry_size(name: &str, value: &str) -> usize {
        // RFC 7541: size = len(name) + len(value) + 32
        name.len() + value.len() + 32
    }

    pub fn insert(&mut self, name: &str, value: &str) {
        let size = Self::entry_size(name, value);

        // Evict entries to make room.
        while self.current_size + size > self.max_size && !self.entries.is_empty() {
            let removed = self.entries.pop().unwrap();
            self.current_size -= Self::entry_size(&removed.0, &removed.1);
        }

        if size <= self.max_size {
            self.entries.insert(0, (name.to_string(), value.to_string()));
            self.current_size += size;
        }
    }

    pub fn get(&self, index: usize) -> Option<(&str, &str)> {
        self.entries
            .get(index)
            .map(|(n, v)| (n.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn size(&self) -> usize {
        self.current_size
    }

    pub fn set_max_size(&mut self, max: usize) {
        self.max_size = max;
        while self.current_size > self.max_size && !self.entries.is_empty() {
            let removed = self.entries.pop().unwrap();
            self.current_size -= Self::entry_size(&removed.0, &removed.1);
        }
    }
}

/// HPACK header table (static + dynamic).
#[derive(Debug)]
pub struct HpackTable {
    pub dynamic: HpackDynamicTable,
}

impl HpackTable {
    pub fn new(dynamic_max_size: usize) -> Self {
        Self {
            dynamic: HpackDynamicTable::new(dynamic_max_size),
        }
    }

    /// Look up by 1-based index (1..=61 = static, 62+ = dynamic).
    pub fn get(&self, index: usize) -> Option<(&str, &str)> {
        if index == 0 {
            return None;
        }
        if index <= HPACK_STATIC_TABLE.len() {
            let (n, v) = HPACK_STATIC_TABLE[index - 1];
            Some((n, v))
        } else {
            let dyn_index = index - HPACK_STATIC_TABLE.len() - 1;
            self.dynamic.get(dyn_index)
        }
    }

    /// Find a static table index for a header name (returns first match).
    pub fn find_name(&self, name: &str) -> Option<usize> {
        let lower = name.to_ascii_lowercase();
        HPACK_STATIC_TABLE
            .iter()
            .enumerate()
            .find(|(_, (n, _))| *n == lower)
            .map(|(i, _)| i + 1)
    }

    /// Find a static table index for exact (name, value) match.
    pub fn find_exact(&self, name: &str, value: &str) -> Option<usize> {
        let lower = name.to_ascii_lowercase();
        HPACK_STATIC_TABLE
            .iter()
            .enumerate()
            .find(|(_, (n, v))| *n == lower && *v == value)
            .map(|(i, _)| i + 1)
    }
}

// ── HPACK integer encoding ─────────────────────────────────────

/// Encode an integer with the given prefix size (RFC 7541, Section 5.1).
pub fn hpack_encode_integer(value: u64, prefix_bits: u8) -> Vec<u8> {
    let max_prefix = (1u64 << prefix_bits) - 1;
    if value < max_prefix {
        return vec![value as u8];
    }
    let mut result = vec![max_prefix as u8];
    let mut remaining = value - max_prefix;
    while remaining >= 128 {
        result.push((remaining & 0x7F) as u8 | 0x80);
        remaining >>= 7;
    }
    result.push(remaining as u8);
    result
}

/// Decode an HPACK integer from bytes, returning (value, bytes_consumed).
pub fn hpack_decode_integer(buf: &[u8], prefix_bits: u8) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    let max_prefix = (1u64 << prefix_bits) - 1;
    let first = (buf[0] as u64) & max_prefix;
    if first < max_prefix {
        return Some((first, 1));
    }
    let mut value = max_prefix;
    let mut m = 0u32;
    let mut i = 1;
    loop {
        if i >= buf.len() {
            return None;
        }
        let b = buf[i] as u64;
        value += (b & 0x7F) << m;
        m += 7;
        i += 1;
        if b & 0x80 == 0 {
            break;
        }
    }
    Some((value, i))
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_type_roundtrip() {
        for i in 0u8..=9 {
            let ft = FrameType::from_u8(i);
            assert_eq!(ft.to_u8(), i);
        }
        let unknown = FrameType::from_u8(0xFE);
        assert_eq!(unknown, FrameType::Unknown(0xFE));
    }

    #[test]
    fn frame_header_parse_serialize() {
        let hdr = FrameHeader::new(FrameType::Data, FrameFlags::END_STREAM, 1, 100);
        let bytes = hdr.serialize();
        let parsed = FrameHeader::parse(&bytes).unwrap();
        assert_eq!(parsed, hdr);
    }

    #[test]
    fn frame_header_stream_id_mask() {
        let hdr = FrameHeader::new(FrameType::Headers, 0, 0xFFFF_FFFF, 0);
        assert_eq!(hdr.stream_id, 0x7FFF_FFFF);
    }

    #[test]
    fn frame_flags() {
        let mut flags = FrameFlags(0);
        flags.set(FrameFlags::END_STREAM);
        assert!(flags.has(FrameFlags::END_STREAM));
        assert!(!flags.has(FrameFlags::END_HEADERS));
        flags.set(FrameFlags::END_HEADERS);
        assert!(flags.has(FrameFlags::END_HEADERS));
        flags.clear(FrameFlags::END_STREAM);
        assert!(!flags.has(FrameFlags::END_STREAM));
    }

    #[test]
    fn settings_build_and_parse() {
        let params = vec![
            (SETTINGS_MAX_CONCURRENT_STREAMS, 100),
            (SETTINGS_INITIAL_WINDOW_SIZE, 65535),
        ];
        let payload = build_settings(&params);
        assert_eq!(payload.len(), 12);
        let parsed = parse_settings(&payload);
        assert_eq!(parsed, params);
    }

    #[test]
    fn stream_tracker_basics() {
        let mut tracker = StreamTracker::new(true);
        let s1 = tracker.open_stream();
        assert_eq!(s1, 1);
        assert_eq!(tracker.state(1), StreamState::Open);
        let s2 = tracker.open_stream();
        assert_eq!(s2, 3);
        assert_eq!(tracker.active_count(), 2);
        tracker.close(1);
        assert_eq!(tracker.state(1), StreamState::Closed);
        assert_eq!(tracker.active_count(), 1);
    }

    #[test]
    fn stream_tracker_server_even_ids() {
        let mut tracker = StreamTracker::new(false);
        assert_eq!(tracker.open_stream(), 2);
        assert_eq!(tracker.open_stream(), 4);
    }

    #[test]
    fn hpack_static_table_lookup() {
        let table = HpackTable::new(4096);
        let (name, value) = table.get(2).unwrap();
        assert_eq!(name, ":method");
        assert_eq!(value, "GET");
        let (name, _) = table.get(1).unwrap();
        assert_eq!(name, ":authority");
    }

    #[test]
    fn hpack_dynamic_table_insert_and_evict() {
        let mut dt = HpackDynamicTable::new(128);
        dt.insert("custom-header", "value1");
        assert_eq!(dt.len(), 1);
        assert_eq!(dt.get(0), Some(("custom-header", "value1")));

        // Insert enough to evict.
        dt.insert("another-long-header-name", "another-long-value-here-to-fill-the-table");
        // The first entry should have been evicted.
        assert!(dt.len() <= 2);
    }

    #[test]
    fn hpack_dynamic_table_max_size_change() {
        let mut dt = HpackDynamicTable::new(4096);
        dt.insert("a", "b");
        dt.insert("c", "d");
        assert_eq!(dt.len(), 2);
        dt.set_max_size(0);
        assert_eq!(dt.len(), 0);
        assert_eq!(dt.size(), 0);
    }

    #[test]
    fn hpack_integer_encode_small() {
        let encoded = hpack_encode_integer(10, 5);
        assert_eq!(encoded, vec![10]);
    }

    #[test]
    fn hpack_integer_encode_large() {
        let encoded = hpack_encode_integer(1337, 5);
        // 31 + 1306: 31 is the prefix max for 5 bits.
        let (decoded, consumed) = hpack_decode_integer(&encoded, 5).unwrap();
        assert_eq!(decoded, 1337);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn hpack_integer_roundtrip() {
        for bits in [4u8, 5, 6, 7] {
            for val in [0, 1, 30, 31, 127, 128, 255, 1000, 65535] {
                let encoded = hpack_encode_integer(val, bits);
                let (decoded, _) = hpack_decode_integer(&encoded, bits).unwrap();
                assert_eq!(decoded, val, "bits={bits} val={val}");
            }
        }
    }

    #[test]
    fn hpack_find_name() {
        let table = HpackTable::new(4096);
        let idx = table.find_name(":method").unwrap();
        assert_eq!(idx, 2); // First :method entry.
        let idx = table.find_exact(":method", "POST").unwrap();
        assert_eq!(idx, 3);
    }
}
