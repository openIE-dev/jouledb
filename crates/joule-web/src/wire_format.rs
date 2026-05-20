//! Wire format protocol — message framing, versioning, checksums, fragmentation.
//!
//! Defines the low-level [`WireMessage`] format with header fields (version, flags,
//! message type, length), serialization/deserialization, version negotiation,
//! CRC32-like checksum validation, compression/encryption flags, and message
//! fragmentation for large payloads with reassembly.

use std::collections::HashMap;
use std::fmt;

// ── Constants ──────────────────────────────────────────────────

const WIRE_MAGIC: u16 = 0x4A57; // 'JW' for Joule Wire
const HEADER_SIZE: usize = 16;
const MAX_FRAGMENT_PAYLOAD: usize = 65536;

// ── Message Type ───────────────────────────────────────────────

/// The type of wire message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageType {
    Handshake = 0x01,
    Data = 0x02,
    Heartbeat = 0x03,
    Close = 0x04,
    Error = 0x05,
    Fragment = 0x06,
}

impl MessageType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x01 => Some(Self::Handshake),
            0x02 => Some(Self::Data),
            0x03 => Some(Self::Heartbeat),
            0x04 => Some(Self::Close),
            0x05 => Some(Self::Error),
            0x06 => Some(Self::Fragment),
            _ => None,
        }
    }
}

impl fmt::Display for MessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake => write!(f, "Handshake"),
            Self::Data => write!(f, "Data"),
            Self::Heartbeat => write!(f, "Heartbeat"),
            Self::Close => write!(f, "Close"),
            Self::Error => write!(f, "Error"),
            Self::Fragment => write!(f, "Fragment"),
        }
    }
}

// ── Wire Flags ─────────────────────────────────────────────────

/// Bitflags for wire message options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireFlags(u8);

impl WireFlags {
    pub const NONE: Self = Self(0);
    pub const COMPRESSED: Self = Self(0x01);
    pub const ENCRYPTED: Self = Self(0x02);
    pub const CHECKSUM: Self = Self(0x04);
    pub const FRAGMENTED: Self = Self(0x08);

    pub fn new(bits: u8) -> Self { Self(bits) }
    pub fn bits(self) -> u8 { self.0 }
    pub fn has(self, flag: WireFlags) -> bool { self.0 & flag.0 != 0 }
    pub fn set(self, flag: WireFlags) -> Self { Self(self.0 | flag.0) }
    pub fn clear(self, flag: WireFlags) -> Self { Self(self.0 & !flag.0) }
}

impl fmt::Display for WireFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.has(Self::COMPRESSED) { parts.push("COMPRESSED"); }
        if self.has(Self::ENCRYPTED) { parts.push("ENCRYPTED"); }
        if self.has(Self::CHECKSUM) { parts.push("CHECKSUM"); }
        if self.has(Self::FRAGMENTED) { parts.push("FRAGMENTED"); }
        if parts.is_empty() { write!(f, "NONE") } else { write!(f, "{}", parts.join("|")) }
    }
}

// ── Wire Header ────────────────────────────────────────────────

/// Fixed-size header for every wire message (16 bytes).
///
/// ```text
/// [2B magic][1B version][1B flags][1B type][3B reserved][4B length][4B checksum]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireHeader {
    pub version: u8,
    pub flags: WireFlags,
    pub msg_type: MessageType,
    pub payload_length: u32,
    pub checksum: u32,
}

impl WireHeader {
    pub fn new(version: u8, msg_type: MessageType, payload_length: u32) -> Self {
        Self {
            version,
            flags: WireFlags::NONE,
            msg_type,
            payload_length,
            checksum: 0,
        }
    }

    pub fn with_flags(mut self, flags: WireFlags) -> Self {
        self.flags = flags; self
    }

    pub fn with_checksum(mut self, checksum: u32) -> Self {
        self.checksum = checksum;
        self.flags = self.flags.set(WireFlags::CHECKSUM);
        self
    }

    /// Serialize header to 16 bytes.
    pub fn serialize(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..2].copy_from_slice(&WIRE_MAGIC.to_be_bytes());
        buf[2] = self.version;
        buf[3] = self.flags.bits();
        buf[4] = self.msg_type as u8;
        // bytes 5..8 reserved
        buf[8..12].copy_from_slice(&self.payload_length.to_be_bytes());
        buf[12..16].copy_from_slice(&self.checksum.to_be_bytes());
        buf
    }

    /// Deserialize header from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self, WireError> {
        if data.len() < HEADER_SIZE {
            return Err(WireError::IncompleteHeader { needed: HEADER_SIZE, available: data.len() });
        }
        let magic = u16::from_be_bytes([data[0], data[1]]);
        if magic != WIRE_MAGIC {
            return Err(WireError::InvalidMagic(magic));
        }
        let version = data[2];
        let flags = WireFlags::new(data[3]);
        let msg_type = MessageType::from_u8(data[4])
            .ok_or(WireError::InvalidMessageType(data[4]))?;
        let payload_length = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let checksum = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);
        Ok(Self { version, flags, msg_type, payload_length, checksum })
    }
}

// ── Wire Message ───────────────────────────────────────────────

/// A complete wire message: header + payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireMessage {
    pub header: WireHeader,
    pub payload: Vec<u8>,
}

impl WireMessage {
    pub fn new(version: u8, msg_type: MessageType, payload: Vec<u8>) -> Self {
        let header = WireHeader::new(version, msg_type, payload.len() as u32);
        Self { header, payload }
    }

    /// Create a handshake message containing supported version range.
    pub fn handshake(version: u8, min_version: u8, max_version: u8) -> Self {
        Self::new(version, MessageType::Handshake, vec![min_version, max_version])
    }

    /// Create a heartbeat message.
    pub fn heartbeat(version: u8) -> Self {
        Self::new(version, MessageType::Heartbeat, Vec::new())
    }

    /// Create a close message.
    pub fn close(version: u8, reason: &str) -> Self {
        Self::new(version, MessageType::Close, reason.as_bytes().to_vec())
    }

    /// Create an error message.
    pub fn error(version: u8, code: u16, detail: &str) -> Self {
        let mut payload = code.to_be_bytes().to_vec();
        payload.extend_from_slice(detail.as_bytes());
        Self::new(version, MessageType::Error, payload)
    }

    /// Serialize the full message (header + payload) to bytes.
    pub fn serialize(&self, compute_checksum: bool) -> Vec<u8> {
        let mut header = self.header;
        if compute_checksum {
            header.checksum = crc32_simple(&self.payload);
            header.flags = header.flags.set(WireFlags::CHECKSUM);
        }
        let hdr_bytes = header.serialize();
        let mut buf = Vec::with_capacity(HEADER_SIZE + self.payload.len());
        buf.extend_from_slice(&hdr_bytes);
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Deserialize a wire message from bytes. Returns message and bytes consumed.
    pub fn deserialize(data: &[u8]) -> Result<(Self, usize), WireError> {
        let header = WireHeader::deserialize(data)?;
        let total = HEADER_SIZE + header.payload_length as usize;
        if data.len() < total {
            return Err(WireError::IncompletePayload {
                needed: total,
                available: data.len(),
            });
        }
        let payload = data[HEADER_SIZE..total].to_vec();
        // Validate checksum if present
        if header.flags.has(WireFlags::CHECKSUM) {
            let computed = crc32_simple(&payload);
            if computed != header.checksum {
                return Err(WireError::ChecksumMismatch {
                    expected: header.checksum,
                    computed,
                });
            }
        }
        Ok((Self { header, payload }, total))
    }

    /// Total wire size.
    pub fn wire_size(&self) -> usize { HEADER_SIZE + self.payload.len() }
}

impl fmt::Display for WireMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wire[v{} {} flags={} {}B]",
            self.header.version, self.header.msg_type, self.header.flags, self.payload.len())
    }
}

// ── Wire Error ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireError {
    IncompleteHeader { needed: usize, available: usize },
    IncompletePayload { needed: usize, available: usize },
    InvalidMagic(u16),
    InvalidMessageType(u8),
    ChecksumMismatch { expected: u32, computed: u32 },
    VersionMismatch { local: u8, remote: u8 },
    FragmentError(String),
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteHeader { needed, available } =>
                write!(f, "incomplete header: need {needed}, have {available}"),
            Self::IncompletePayload { needed, available } =>
                write!(f, "incomplete payload: need {needed}, have {available}"),
            Self::InvalidMagic(m) => write!(f, "invalid magic: 0x{m:04x}"),
            Self::InvalidMessageType(t) => write!(f, "invalid message type: 0x{t:02x}"),
            Self::ChecksumMismatch { expected, computed } =>
                write!(f, "checksum mismatch: expected 0x{expected:08x}, computed 0x{computed:08x}"),
            Self::VersionMismatch { local, remote } =>
                write!(f, "version mismatch: local={local}, remote={remote}"),
            Self::FragmentError(msg) => write!(f, "fragment error: {msg}"),
        }
    }
}

// ── CRC32 (simple) ─────────────────────────────────────────────

/// Simple CRC32-like checksum (not standards-compliant, but deterministic).
pub fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFF_FFFF
}

// ── Version Negotiation ────────────────────────────────────────

/// Negotiate a common version between two peers.
pub fn negotiate_version(
    local_min: u8, local_max: u8,
    remote_min: u8, remote_max: u8,
) -> Option<u8> {
    let low = local_min.max(remote_min);
    let high = local_max.min(remote_max);
    if low <= high { Some(high) } else { None }
}

// ── Fragmenter ─────────────────────────────────────────────────

/// Fragment header embedded at the start of each fragment payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentHeader {
    pub message_id: u32,
    pub fragment_index: u16,
    pub total_fragments: u16,
}

impl FragmentHeader {
    const SIZE: usize = 8;

    pub fn serialize(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.message_id.to_be_bytes());
        buf[4..6].copy_from_slice(&self.fragment_index.to_be_bytes());
        buf[6..8].copy_from_slice(&self.total_fragments.to_be_bytes());
        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, WireError> {
        if data.len() < Self::SIZE {
            return Err(WireError::FragmentError("fragment header too short".into()));
        }
        Ok(Self {
            message_id: u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
            fragment_index: u16::from_be_bytes([data[4], data[5]]),
            total_fragments: u16::from_be_bytes([data[6], data[7]]),
        })
    }
}

/// Fragment a large payload into multiple wire messages.
pub fn fragment_message(
    version: u8,
    message_id: u32,
    payload: &[u8],
    max_fragment_size: usize,
) -> Vec<WireMessage> {
    let effective_max = max_fragment_size.max(1);
    let chunk_size = effective_max.saturating_sub(FragmentHeader::SIZE).max(1);
    let total_fragments = (payload.len() + chunk_size - 1) / chunk_size;
    let total_fragments = total_fragments.max(1) as u16;

    let mut fragments = Vec::new();
    let mut offset = 0;

    for i in 0..total_fragments {
        let end = (offset + chunk_size).min(payload.len());
        let chunk = &payload[offset..end];
        let frag_header = FragmentHeader {
            message_id,
            fragment_index: i,
            total_fragments,
        };
        let mut frag_payload = frag_header.serialize().to_vec();
        frag_payload.extend_from_slice(chunk);
        let mut msg = WireMessage::new(version, MessageType::Fragment, frag_payload);
        msg.header.flags = msg.header.flags.set(WireFlags::FRAGMENTED);
        fragments.push(msg);
        offset = end;
    }
    fragments
}

// ── Reassembler ────────────────────────────────────────────────

/// Collects fragments and reassembles the original payload.
#[derive(Debug)]
pub struct Reassembler {
    pending: HashMap<u32, FragmentBuffer>,
}

#[derive(Debug)]
struct FragmentBuffer {
    total: u16,
    received: HashMap<u16, Vec<u8>>,
}

impl Reassembler {
    pub fn new() -> Self {
        Self { pending: HashMap::new() }
    }

    /// Feed a fragment wire message. Returns the reassembled payload when all
    /// fragments are received.
    pub fn feed(&mut self, msg: &WireMessage) -> Result<Option<Vec<u8>>, WireError> {
        if msg.header.msg_type != MessageType::Fragment {
            return Err(WireError::FragmentError("not a fragment message".into()));
        }
        let frag_hdr = FragmentHeader::deserialize(&msg.payload)?;
        let data = msg.payload[FragmentHeader::SIZE..].to_vec();

        let buf = self.pending.entry(frag_hdr.message_id)
            .or_insert_with(|| FragmentBuffer {
                total: frag_hdr.total_fragments,
                received: HashMap::new(),
            });
        buf.received.insert(frag_hdr.fragment_index, data);

        if buf.received.len() == buf.total as usize {
            let buf = self.pending.remove(&frag_hdr.message_id).unwrap();
            let mut result = Vec::new();
            for i in 0..buf.total {
                if let Some(chunk) = buf.received.get(&i) {
                    result.extend_from_slice(chunk);
                } else {
                    return Err(WireError::FragmentError(format!("missing fragment {i}")));
                }
            }
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    pub fn pending_count(&self) -> usize { self.pending.len() }
}

impl Default for Reassembler {
    fn default() -> Self { Self::new() }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_serialize_deserialize() {
        let hdr = WireHeader::new(1, MessageType::Data, 42);
        let bytes = hdr.serialize();
        let decoded = WireHeader::deserialize(&bytes).unwrap();
        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.msg_type, MessageType::Data);
        assert_eq!(decoded.payload_length, 42);
    }

    #[test]
    fn message_roundtrip() {
        let msg = WireMessage::new(1, MessageType::Data, vec![1, 2, 3, 4]);
        let bytes = msg.serialize(false);
        let (decoded, consumed) = WireMessage::deserialize(&bytes).unwrap();
        assert_eq!(decoded, msg);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn message_roundtrip_with_checksum() {
        let msg = WireMessage::new(1, MessageType::Data, vec![10, 20, 30]);
        let bytes = msg.serialize(true);
        let (decoded, _) = WireMessage::deserialize(&bytes).unwrap();
        assert_eq!(decoded.payload, vec![10, 20, 30]);
    }

    #[test]
    fn checksum_mismatch_detected() {
        let msg = WireMessage::new(1, MessageType::Data, vec![1, 2, 3]);
        let mut bytes = msg.serialize(true);
        // Corrupt one payload byte
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert!(matches!(
            WireMessage::deserialize(&bytes),
            Err(WireError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn invalid_magic_rejected() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0] = 0xFF;
        bytes[1] = 0xFF;
        assert!(matches!(
            WireHeader::deserialize(&bytes),
            Err(WireError::InvalidMagic(0xFFFF))
        ));
    }

    #[test]
    fn incomplete_header() {
        assert!(matches!(
            WireHeader::deserialize(&[0; 4]),
            Err(WireError::IncompleteHeader { .. })
        ));
    }

    #[test]
    fn handshake_message() {
        let msg = WireMessage::handshake(1, 1, 3);
        assert_eq!(msg.header.msg_type, MessageType::Handshake);
        assert_eq!(msg.payload, vec![1, 3]);
    }

    #[test]
    fn heartbeat_message() {
        let msg = WireMessage::heartbeat(1);
        assert_eq!(msg.header.msg_type, MessageType::Heartbeat);
        assert!(msg.payload.is_empty());
    }

    #[test]
    fn close_message() {
        let msg = WireMessage::close(1, "done");
        assert_eq!(msg.header.msg_type, MessageType::Close);
        assert_eq!(msg.payload, b"done");
    }

    #[test]
    fn error_message() {
        let msg = WireMessage::error(1, 500, "fail");
        assert_eq!(msg.header.msg_type, MessageType::Error);
        assert!(msg.payload.len() >= 2);
    }

    #[test]
    fn version_negotiation_compatible() {
        assert_eq!(negotiate_version(1, 3, 2, 5), Some(3));
    }

    #[test]
    fn version_negotiation_incompatible() {
        assert_eq!(negotiate_version(1, 2, 3, 5), None);
    }

    #[test]
    fn wire_flags_operations() {
        let f = WireFlags::NONE.set(WireFlags::COMPRESSED).set(WireFlags::CHECKSUM);
        assert!(f.has(WireFlags::COMPRESSED));
        assert!(f.has(WireFlags::CHECKSUM));
        assert!(!f.has(WireFlags::ENCRYPTED));
        let f2 = f.clear(WireFlags::COMPRESSED);
        assert!(!f2.has(WireFlags::COMPRESSED));
    }

    #[test]
    fn fragmentation_and_reassembly() {
        let payload = vec![0xAB; 200];
        let fragments = fragment_message(1, 42, &payload, 100);
        assert!(fragments.len() > 1);
        let mut reassembler = Reassembler::new();
        let mut result = None;
        for frag in &fragments {
            result = reassembler.feed(frag).unwrap();
        }
        assert_eq!(result.unwrap(), payload);
    }

    #[test]
    fn fragment_single_small_payload() {
        let payload = vec![1, 2, 3];
        let fragments = fragment_message(1, 1, &payload, MAX_FRAGMENT_PAYLOAD);
        assert_eq!(fragments.len(), 1);
        let mut reassembler = Reassembler::new();
        let result = reassembler.feed(&fragments[0]).unwrap();
        assert_eq!(result.unwrap(), payload);
    }

    #[test]
    fn crc32_deterministic() {
        let a = crc32_simple(b"hello");
        let b = crc32_simple(b"hello");
        assert_eq!(a, b);
        let c = crc32_simple(b"world");
        assert_ne!(a, c);
    }

    #[test]
    fn wire_message_display() {
        let msg = WireMessage::new(1, MessageType::Data, vec![0; 50]);
        let s = format!("{msg}");
        assert!(s.contains("Data"));
        assert!(s.contains("50B"));
    }

    #[test]
    fn flags_display() {
        assert_eq!(format!("{}", WireFlags::NONE), "NONE");
        let f = WireFlags::COMPRESSED.set(WireFlags::ENCRYPTED);
        let s = format!("{f}");
        assert!(s.contains("COMPRESSED"));
        assert!(s.contains("ENCRYPTED"));
    }

    #[test]
    fn message_wire_size() {
        let msg = WireMessage::new(1, MessageType::Data, vec![0; 100]);
        assert_eq!(msg.wire_size(), HEADER_SIZE + 100);
    }
}
