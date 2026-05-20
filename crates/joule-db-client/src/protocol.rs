//! Binary wire protocol encoder/decoder for JouleDB.
//!
//! This module implements the exact same wire format as the server's
//! `binary_protocol.rs`, ensuring full interoperability.
//!
//! ## Wire Format
//!
//! ```text
//! +--------+--------+-------+----------+-----------+-------------+
//! | Magic  | Version| Flags | Msg Type | Request ID| Payload Len |
//! | 4 bytes| 1 byte | 1 byte| 2 bytes  | 4 bytes   | 4 bytes     |
//! +--------+--------+-------+----------+-----------+-------------+
//! |                      Payload                                 |
//! |                   (variable length)                          |
//! +--------------------------------------------------------------+
//! ```
//!
//! Payload encoding uses LEB128 varint length-prefixed fields for bytes and
//! strings.

use crate::error::ClientError;

// ============================================================================
// Constants
// ============================================================================

/// Protocol magic bytes: "WVDB" (0x57 0x56 0x44 0x42).
pub const MAGIC: [u8; 4] = [0x57, 0x56, 0x44, 0x42];

/// Current protocol version.
pub const VERSION: u8 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 16;

/// Maximum allowed payload size (16 MiB).
pub const MAX_PAYLOAD_SIZE: u32 = 16 * 1024 * 1024;

// ============================================================================
// Message Types
// ============================================================================

/// Message type identifier -- values match the server exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MessageType {
    // Requests (0x0000 - 0x7FFF)
    Get = 0x0001,
    Set = 0x0002,
    Delete = 0x0003,
    Query = 0x0004,
    Batch = 0x0005,
    Ping = 0x0006,
    Subscribe = 0x0007,
    Unsubscribe = 0x0008,
    Auth = 0x0009,
    BeginTx = 0x0010,
    Commit = 0x0011,
    Rollback = 0x0012,
    Savepoint = 0x0013,
    Prepare = 0x0014,
    Execute = 0x0015,

    // Responses (0x8000 - 0xFFFF)
    GetResponse = 0x8001,
    SetResponse = 0x8002,
    DeleteResponse = 0x8003,
    QueryResponse = 0x8004,
    BatchResponse = 0x8005,
    Pong = 0x8006,
    Notification = 0x8007,
    AuthResponse = 0x8008,
    BeginTxResponse = 0x8010,
    CommitResponse = 0x8011,
    RollbackResponse = 0x8012,
    SavepointResponse = 0x8013,
    PrepareResponse = 0x8014,
    ExecuteResponse = 0x8015,
    Error = 0x80FF,
}

impl MessageType {
    /// Construct from a raw `u16` on the wire. Returns `None` for unknown
    /// types.
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::Get),
            0x0002 => Some(Self::Set),
            0x0003 => Some(Self::Delete),
            0x0004 => Some(Self::Query),
            0x0005 => Some(Self::Batch),
            0x0006 => Some(Self::Ping),
            0x0007 => Some(Self::Subscribe),
            0x0008 => Some(Self::Unsubscribe),
            0x0009 => Some(Self::Auth),
            0x0010 => Some(Self::BeginTx),
            0x0011 => Some(Self::Commit),
            0x0012 => Some(Self::Rollback),
            0x0013 => Some(Self::Savepoint),
            0x0014 => Some(Self::Prepare),
            0x0015 => Some(Self::Execute),
            0x8001 => Some(Self::GetResponse),
            0x8002 => Some(Self::SetResponse),
            0x8003 => Some(Self::DeleteResponse),
            0x8004 => Some(Self::QueryResponse),
            0x8005 => Some(Self::BatchResponse),
            0x8006 => Some(Self::Pong),
            0x8007 => Some(Self::Notification),
            0x8008 => Some(Self::AuthResponse),
            0x8010 => Some(Self::BeginTxResponse),
            0x8011 => Some(Self::CommitResponse),
            0x8012 => Some(Self::RollbackResponse),
            0x8013 => Some(Self::SavepointResponse),
            0x8014 => Some(Self::PrepareResponse),
            0x8015 => Some(Self::ExecuteResponse),
            0x80FF => Some(Self::Error),
            _ => None,
        }
    }

    /// Returns `true` for request message types (< 0x8000).
    pub fn is_request(self) -> bool {
        (self as u16) < 0x8000
    }

    /// Returns `true` for response message types (>= 0x8000).
    pub fn is_response(self) -> bool {
        (self as u16) >= 0x8000
    }
}

// ============================================================================
// Flags
// ============================================================================

/// Protocol flags byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags(u8);

impl Flags {
    /// No flags set.
    pub const NONE: Flags = Flags(0);
    /// Bit 0: payload is LZ4-compressed.
    pub const COMPRESSED: Flags = Flags(1 << 0);
    /// Bit 1: request expects a response.
    pub const EXPECT_RESPONSE: Flags = Flags(1 << 1);
    /// Bit 2: this is the final message in a stream.
    pub const FINAL: Flags = Flags(1 << 2);

    /// Create flags with no bits set.
    pub fn new() -> Self {
        Self::NONE
    }

    /// Check whether the given flag is set.
    pub fn has(self, flag: Flags) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Set a flag bit.
    pub fn set(&mut self, flag: Flags) {
        self.0 |= flag.0;
    }

    /// Clear a flag bit.
    pub fn clear(&mut self, flag: Flags) {
        self.0 &= !flag.0;
    }

    /// Return the raw byte value.
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Construct from a raw byte value.
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
}

// ============================================================================
// Message
// ============================================================================

/// A binary protocol message consisting of a type, request ID, flags, and an
/// opaque payload.
#[derive(Debug, Clone)]
pub struct Message {
    /// The message type discriminant.
    pub msg_type: MessageType,
    /// Correlation ID so responses can be matched to requests.
    pub request_id: u32,
    /// Protocol flags.
    pub flags: Flags,
    /// The raw payload bytes. Layout depends on `msg_type`.
    pub payload: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Message constructors
// ---------------------------------------------------------------------------

impl Message {
    /// Create a raw message.
    pub fn new(msg_type: MessageType, request_id: u32, payload: Vec<u8>) -> Self {
        Self {
            msg_type,
            request_id,
            flags: Flags::new(),
            payload,
        }
    }

    // -- Request constructors -----------------------------------------------

    /// Create a **Get** request.
    ///
    /// Payload: `bytes(key)`.
    pub fn get(request_id: u32, key: &[u8]) -> Self {
        let mut payload = Vec::with_capacity(key.len() + 5);
        write_bytes(&mut payload, key);
        Self::new(MessageType::Get, request_id, payload)
    }

    /// Create a **Set** request.
    ///
    /// Payload: `bytes(key) + bytes(value) + has_ttl:u8 [+ ttl:u64 LE]`.
    pub fn set(request_id: u32, key: &[u8], value: &[u8], ttl: Option<u64>) -> Self {
        let mut payload = Vec::with_capacity(key.len() + value.len() + 16);
        write_bytes(&mut payload, key);
        write_bytes(&mut payload, value);
        match ttl {
            Some(t) => {
                payload.push(1);
                payload.extend_from_slice(&t.to_le_bytes());
            }
            None => {
                payload.push(0);
            }
        }
        Self::new(MessageType::Set, request_id, payload)
    }

    /// Create a **Delete** request.
    ///
    /// Payload: `bytes(key)`.
    pub fn delete(request_id: u32, key: &[u8]) -> Self {
        let mut payload = Vec::with_capacity(key.len() + 5);
        write_bytes(&mut payload, key);
        Self::new(MessageType::Delete, request_id, payload)
    }

    /// Create a **Query** request.
    ///
    /// Payload: `string(sql) + has_params:u8 [+ bytes(params_json)]`.
    pub fn query(request_id: u32, sql: &str, params_json: Option<&[u8]>) -> Self {
        let mut payload = Vec::new();
        write_string(&mut payload, sql);
        match params_json {
            Some(p) => {
                payload.push(1);
                write_bytes(&mut payload, p);
            }
            None => {
                payload.push(0);
            }
        }
        Self::new(MessageType::Query, request_id, payload)
    }

    /// Create a **Ping** request (empty payload).
    pub fn ping(request_id: u32) -> Self {
        Self::new(MessageType::Ping, request_id, Vec::new())
    }

    /// Create a **BeginTx** request (empty payload).
    pub fn begin_tx(request_id: u32) -> Self {
        Self::new(MessageType::BeginTx, request_id, Vec::new())
    }

    /// Create a **Commit** request (empty payload).
    pub fn commit(request_id: u32) -> Self {
        Self::new(MessageType::Commit, request_id, Vec::new())
    }

    /// Create a **Rollback** request (empty payload).
    pub fn rollback(request_id: u32) -> Self {
        Self::new(MessageType::Rollback, request_id, Vec::new())
    }

    // -- Response parsers ---------------------------------------------------

    /// Parse a **GetResponse** payload.
    ///
    /// Returns `Some(value)` if the key was found, or `None` if it was not.
    pub fn parse_get_response(&self) -> Result<Option<Vec<u8>>, ClientError> {
        if self.payload.is_empty() {
            return Err(ClientError::protocol("GetResponse: empty payload"));
        }
        if self.payload[0] == 0 {
            Ok(None)
        } else {
            let mut cursor = 1;
            let value = read_bytes(&self.payload, &mut cursor)
                .ok_or_else(|| ClientError::protocol("GetResponse: missing value bytes"))?;
            Ok(Some(value))
        }
    }

    /// Parse a **SetResponse** payload.
    ///
    /// Returns `true` if the set succeeded.
    pub fn parse_set_response(&self) -> Result<bool, ClientError> {
        if self.payload.is_empty() {
            return Err(ClientError::protocol("SetResponse: empty payload"));
        }
        Ok(self.payload[0] != 0)
    }

    /// Parse a **DeleteResponse** payload.
    ///
    /// Returns `true` if the key existed prior to deletion.
    pub fn parse_delete_response(&self) -> Result<bool, ClientError> {
        if self.payload.is_empty() {
            return Err(ClientError::protocol("DeleteResponse: empty payload"));
        }
        Ok(self.payload[0] != 0)
    }

    /// Parse a **QueryResponse** payload (raw JSON bytes).
    pub fn parse_query_response(&self) -> Result<Vec<u8>, ClientError> {
        // The QueryResponse payload is the full JSON result.
        Ok(self.payload.clone())
    }

    /// Parse an **Error** response payload into `(code, message)`.
    pub fn parse_error(&self) -> Result<(String, String), ClientError> {
        let mut cursor = 0;
        let code = read_string(&self.payload, &mut cursor)
            .ok_or_else(|| ClientError::protocol("Error response: missing code"))?;
        let message = read_string(&self.payload, &mut cursor)
            .ok_or_else(|| ClientError::protocol("Error response: missing message"))?;
        Ok((code, message))
    }
}

// ============================================================================
// Encode / Decode
// ============================================================================

/// Encode a `Message` into a byte buffer ready for the wire.
pub fn encode(msg: &Message) -> Result<Vec<u8>, ClientError> {
    let payload_len = msg.payload.len() as u32;
    if payload_len > MAX_PAYLOAD_SIZE {
        return Err(ClientError::protocol(format!(
            "payload too large: {} bytes (max {})",
            payload_len, MAX_PAYLOAD_SIZE
        )));
    }

    let mut buf = Vec::with_capacity(HEADER_SIZE + msg.payload.len());

    // Header
    buf.extend_from_slice(&MAGIC);
    buf.push(VERSION);
    buf.push(msg.flags.bits());
    buf.extend_from_slice(&(msg.msg_type as u16).to_le_bytes());
    buf.extend_from_slice(&msg.request_id.to_le_bytes());
    buf.extend_from_slice(&payload_len.to_le_bytes());

    // Payload
    buf.extend_from_slice(&msg.payload);

    Ok(buf)
}

/// Decode a complete message (header + payload) from a byte slice.
pub fn decode(data: &[u8]) -> Result<Message, ClientError> {
    if data.len() < HEADER_SIZE {
        return Err(ClientError::protocol("truncated message (< 16 bytes)"));
    }

    // Magic
    if data[0..4] != MAGIC {
        return Err(ClientError::protocol("invalid magic bytes"));
    }

    // Version
    let version = data[4];
    if version != VERSION {
        return Err(ClientError::protocol(format!(
            "unsupported protocol version: {}",
            version
        )));
    }

    // Header fields
    let flags = Flags::from_bits(data[5]);
    let msg_type_raw = u16::from_le_bytes([data[6], data[7]]);
    let request_id = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let payload_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

    let msg_type = MessageType::from_u16(msg_type_raw).ok_or_else(|| {
        ClientError::protocol(format!("unknown message type: 0x{:04X}", msg_type_raw))
    })?;

    if payload_len > MAX_PAYLOAD_SIZE {
        return Err(ClientError::protocol(format!(
            "payload too large: {} bytes",
            payload_len
        )));
    }

    let expected_len = HEADER_SIZE + payload_len as usize;
    if data.len() < expected_len {
        return Err(ClientError::protocol("truncated payload"));
    }

    let payload = data[HEADER_SIZE..expected_len].to_vec();

    Ok(Message {
        msg_type,
        request_id,
        flags,
        payload,
    })
}

/// Parse only the 16-byte header. Returns `(msg_type, request_id, payload_len)`.
pub fn decode_header(data: &[u8]) -> Result<(MessageType, u32, Flags, u32), ClientError> {
    if data.len() < HEADER_SIZE {
        return Err(ClientError::protocol("truncated header"));
    }
    if data[0..4] != MAGIC {
        return Err(ClientError::protocol("invalid magic bytes"));
    }
    let version = data[4];
    if version != VERSION {
        return Err(ClientError::protocol(format!(
            "unsupported protocol version: {}",
            version
        )));
    }
    let flags = Flags::from_bits(data[5]);
    let msg_type_raw = u16::from_le_bytes([data[6], data[7]]);
    let request_id = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let payload_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);

    let msg_type = MessageType::from_u16(msg_type_raw).ok_or_else(|| {
        ClientError::protocol(format!("unknown message type: 0x{:04X}", msg_type_raw))
    })?;

    Ok((msg_type, request_id, flags, payload_len))
}

// ============================================================================
// Varint / Bytes / String helpers (public for advanced use)
// ============================================================================

/// Encode a `u64` as an unsigned LEB128 varint and append it to `buf`.
pub fn write_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Read an unsigned LEB128 varint from `data` at the given `cursor`.
/// Advances `cursor` past the varint. Returns `None` on truncation or
/// overflow.
pub fn read_varint(data: &[u8], cursor: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift = 0;

    loop {
        if *cursor >= data.len() {
            return None;
        }

        let byte = data[*cursor];
        *cursor += 1;

        result |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            break;
        }

        shift += 7;
        if shift >= 64 {
            return None; // overflow
        }
    }

    Some(result)
}

/// Write a LEB128 length-prefixed byte slice.
pub fn write_bytes(buf: &mut Vec<u8>, data: &[u8]) {
    write_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

/// Read a LEB128 length-prefixed byte slice from `data` at `cursor`.
pub fn read_bytes(data: &[u8], cursor: &mut usize) -> Option<Vec<u8>> {
    let len = read_varint(data, cursor)? as usize;
    if *cursor + len > data.len() {
        return None;
    }
    let result = data[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Some(result)
}

/// Write a LEB128 length-prefixed UTF-8 string.
pub fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_bytes(buf, s.as_bytes());
}

/// Read a LEB128 length-prefixed UTF-8 string from `data` at `cursor`.
pub fn read_string(data: &[u8], cursor: &mut usize) -> Option<String> {
    let bytes = read_bytes(data, cursor)?;
    String::from_utf8(bytes).ok()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- MessageType --------------------------------------------------------

    #[test]
    fn test_message_type_roundtrip() {
        let types: &[u16] = &[
            0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x0010, 0x0011,
            0x0012, 0x0013, 0x0014, 0x0015, 0x8001, 0x8002, 0x8003, 0x8004, 0x8005, 0x8006, 0x8007,
            0x8008, 0x8010, 0x8011, 0x8012, 0x8013, 0x8014, 0x8015, 0x80FF,
        ];
        for &v in types {
            let mt = MessageType::from_u16(v);
            assert!(mt.is_some(), "should parse 0x{:04X}", v);
            assert_eq!(mt.unwrap() as u16, v);
        }
    }

    #[test]
    fn test_unknown_message_type() {
        assert!(MessageType::from_u16(0x0000).is_none());
        assert!(MessageType::from_u16(0xFFFF).is_none());
    }

    #[test]
    fn test_request_response_classification() {
        assert!(MessageType::Get.is_request());
        assert!(!MessageType::Get.is_response());
        assert!(MessageType::GetResponse.is_response());
        assert!(!MessageType::GetResponse.is_request());
    }

    // -- Flags --------------------------------------------------------------

    #[test]
    fn test_flags_set_clear_has() {
        let mut flags = Flags::new();
        assert!(!flags.has(Flags::COMPRESSED));
        assert!(!flags.has(Flags::EXPECT_RESPONSE));
        assert!(!flags.has(Flags::FINAL));

        flags.set(Flags::COMPRESSED);
        assert!(flags.has(Flags::COMPRESSED));
        assert!(!flags.has(Flags::EXPECT_RESPONSE));

        flags.set(Flags::EXPECT_RESPONSE);
        assert!(flags.has(Flags::COMPRESSED));
        assert!(flags.has(Flags::EXPECT_RESPONSE));

        flags.clear(Flags::COMPRESSED);
        assert!(!flags.has(Flags::COMPRESSED));
        assert!(flags.has(Flags::EXPECT_RESPONSE));
    }

    #[test]
    fn test_flags_from_bits() {
        let flags = Flags::from_bits(0b00000101);
        assert!(flags.has(Flags::COMPRESSED));
        assert!(flags.has(Flags::FINAL));
        assert!(!flags.has(Flags::EXPECT_RESPONSE));
    }

    // -- Varint -------------------------------------------------------------

    #[test]
    fn test_varint_small() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 0);
        assert_eq!(buf, vec![0]);

        buf.clear();
        write_varint(&mut buf, 1);
        assert_eq!(buf, vec![1]);

        buf.clear();
        write_varint(&mut buf, 127);
        assert_eq!(buf, vec![127]);
    }

    #[test]
    fn test_varint_multibyte() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);

        buf.clear();
        write_varint(&mut buf, 16384);
        assert_eq!(buf, vec![0x80, 0x80, 0x01]);
    }

    #[test]
    fn test_varint_roundtrip() {
        let values = [
            0,
            1,
            127,
            128,
            255,
            256,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
        ];
        for value in values {
            let mut buf = Vec::new();
            write_varint(&mut buf, value);

            let mut cursor = 0;
            let decoded = read_varint(&buf, &mut cursor).unwrap_or_else(|| {
                panic!("failed to decode varint for value {}", value);
            });
            assert_eq!(decoded, value, "varint roundtrip failed for {}", value);
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn test_read_varint_truncated() {
        // A continuation byte with no following byte.
        let data = [0x80];
        let mut cursor = 0;
        assert!(read_varint(&data, &mut cursor).is_none());
    }

    #[test]
    fn test_read_varint_empty() {
        let data: &[u8] = &[];
        let mut cursor = 0;
        assert!(read_varint(data, &mut cursor).is_none());
    }

    // -- Bytes / String helpers ---------------------------------------------

    #[test]
    fn test_bytes_roundtrip() {
        let original = b"hello world";
        let mut buf = Vec::new();
        write_bytes(&mut buf, original);

        let mut cursor = 0;
        let decoded = read_bytes(&buf, &mut cursor).unwrap();
        assert_eq!(decoded, original);
        assert_eq!(cursor, buf.len());
    }

    #[test]
    fn test_bytes_empty() {
        let mut buf = Vec::new();
        write_bytes(&mut buf, b"");

        let mut cursor = 0;
        let decoded = read_bytes(&buf, &mut cursor).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_string_roundtrip() {
        let original = "JouleDB rocks!";
        let mut buf = Vec::new();
        write_string(&mut buf, original);

        let mut cursor = 0;
        let decoded = read_string(&buf, &mut cursor).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_read_bytes_truncated() {
        // Varint says 10 bytes but only 2 are present.
        let mut buf = Vec::new();
        write_varint(&mut buf, 10);
        buf.extend_from_slice(b"ab");

        let mut cursor = 0;
        assert!(read_bytes(&buf, &mut cursor).is_none());
    }

    // -- Encode / Decode full messages --------------------------------------

    #[test]
    fn test_encode_decode_get() {
        let msg = Message::get(42, b"mykey");
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MessageType::Get);
        assert_eq!(decoded.request_id, 42);
        assert_eq!(decoded.payload, msg.payload);

        // Verify payload contents
        let mut cursor = 0;
        let key = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(key, b"mykey");
    }

    #[test]
    fn test_encode_decode_set_with_ttl() {
        let msg = Message::set(123, b"key1", b"value1", Some(3600));
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MessageType::Set);
        assert_eq!(decoded.request_id, 123);

        // Parse the payload manually like the server does.
        let mut cursor = 0;
        let key = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(key, b"key1");
        let value = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(value, b"value1");
        assert_eq!(decoded.payload[cursor], 1); // has_ttl
        cursor += 1;
        let ttl = u64::from_le_bytes(decoded.payload[cursor..cursor + 8].try_into().unwrap());
        assert_eq!(ttl, 3600);
    }

    #[test]
    fn test_encode_decode_set_no_ttl() {
        let msg = Message::set(1, b"k", b"v", None);
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        let mut cursor = 0;
        let key = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(key, b"k");
        let value = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(value, b"v");
        assert_eq!(decoded.payload[cursor], 0); // no ttl
    }

    #[test]
    fn test_encode_decode_delete() {
        let msg = Message::delete(99, b"delkey");
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MessageType::Delete);
        let mut cursor = 0;
        let key = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(key, b"delkey");
    }

    #[test]
    fn test_encode_decode_ping() {
        let msg = Message::ping(7);
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MessageType::Ping);
        assert_eq!(decoded.request_id, 7);
        assert!(decoded.payload.is_empty());
    }

    #[test]
    fn test_encode_decode_query() {
        let sql = "SELECT * FROM users WHERE id = ?";
        let params = serde_json::to_vec(&serde_json::json!([42])).unwrap();
        let msg = Message::query(10, sql, Some(&params));
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MessageType::Query);
        let mut cursor = 0;
        let decoded_sql = read_string(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(decoded_sql, sql);
        assert_eq!(decoded.payload[cursor], 1); // has params
        cursor += 1;
        let decoded_params = read_bytes(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(decoded_params, params);
    }

    #[test]
    fn test_encode_decode_query_no_params() {
        let msg = Message::query(11, "SELECT 1", None);
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();

        let mut cursor = 0;
        let decoded_sql = read_string(&decoded.payload, &mut cursor).unwrap();
        assert_eq!(decoded_sql, "SELECT 1");
        assert_eq!(decoded.payload[cursor], 0); // no params
    }

    #[test]
    fn test_encode_decode_begin_commit_rollback() {
        let cases: [(fn(u32) -> Message, MessageType); 3] = [
            (Message::begin_tx, MessageType::BeginTx),
            (Message::commit, MessageType::Commit),
            (Message::rollback, MessageType::Rollback),
        ];
        for (constructor, expected_type) in cases {
            let msg = constructor(55);
            let encoded = encode(&msg).unwrap();
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded.msg_type, expected_type);
            assert_eq!(decoded.request_id, 55);
            assert!(decoded.payload.is_empty());
        }
    }

    // -- Response parsers ---------------------------------------------------

    #[test]
    fn test_parse_get_response_found() {
        let mut payload = vec![1u8]; // found
        write_bytes(&mut payload, b"the-value");
        let msg = Message::new(MessageType::GetResponse, 1, payload);
        let result = msg.parse_get_response().unwrap();
        assert_eq!(result, Some(b"the-value".to_vec()));
    }

    #[test]
    fn test_parse_get_response_not_found() {
        let payload = vec![0u8]; // not found
        let msg = Message::new(MessageType::GetResponse, 1, payload);
        let result = msg.parse_get_response().unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_set_response() {
        let msg = Message::new(MessageType::SetResponse, 1, vec![1]);
        assert!(msg.parse_set_response().unwrap());

        let msg = Message::new(MessageType::SetResponse, 1, vec![0]);
        assert!(!msg.parse_set_response().unwrap());
    }

    #[test]
    fn test_parse_delete_response() {
        let msg = Message::new(MessageType::DeleteResponse, 1, vec![1]);
        assert!(msg.parse_delete_response().unwrap());

        let msg = Message::new(MessageType::DeleteResponse, 1, vec![0]);
        assert!(!msg.parse_delete_response().unwrap());
    }

    #[test]
    fn test_parse_error_response() {
        let mut payload = Vec::new();
        write_string(&mut payload, "NOT_FOUND");
        write_string(&mut payload, "Key does not exist");
        let msg = Message::new(MessageType::Error, 1, payload);
        let (code, message) = msg.parse_error().unwrap();
        assert_eq!(code, "NOT_FOUND");
        assert_eq!(message, "Key does not exist");
    }

    // -- Decode errors ------------------------------------------------------

    #[test]
    fn test_decode_truncated_header() {
        let data = [0x57, 0x56, 0x44]; // 3 bytes < 16
        assert!(decode(&data).is_err());
    }

    #[test]
    fn test_decode_invalid_magic() {
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        data[4] = VERSION;
        data[6..8].copy_from_slice(&(MessageType::Ping as u16).to_le_bytes());
        assert!(decode(&data).is_err());
    }

    #[test]
    fn test_decode_unsupported_version() {
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = 99; // bad version
        data[6..8].copy_from_slice(&(MessageType::Ping as u16).to_le_bytes());
        assert!(decode(&data).is_err());
    }

    #[test]
    fn test_decode_unknown_msg_type() {
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&MAGIC);
        data[4] = VERSION;
        data[6..8].copy_from_slice(&0xFFFFu16.to_le_bytes());
        assert!(decode(&data).is_err());
    }

    #[test]
    fn test_decode_truncated_payload() {
        let msg = Message::get(1, b"test");
        let mut encoded = encode(&msg).unwrap();
        // Chop off the last byte of payload.
        encoded.pop();
        assert!(decode(&encoded).is_err());
    }

    // -- Header-only decode -------------------------------------------------

    #[test]
    fn test_decode_header() {
        let msg = Message::get(42, b"test");
        let encoded = encode(&msg).unwrap();
        let (msg_type, request_id, _flags, payload_len) = decode_header(&encoded).unwrap();
        assert_eq!(msg_type, MessageType::Get);
        assert_eq!(request_id, 42);
        assert_eq!(payload_len as usize, encoded.len() - HEADER_SIZE);
    }

    // -- Wire compatibility with server encoding ----------------------------

    #[test]
    fn test_wire_compat_header_layout() {
        let msg = Message {
            msg_type: MessageType::Set,
            request_id: 0x12345678,
            flags: Flags::from_bits(0x03),
            payload: vec![0xAA, 0xBB],
        };
        let encoded = encode(&msg).unwrap();

        // Magic (4 bytes)
        assert_eq!(&encoded[0..4], &MAGIC);
        // Version (1 byte)
        assert_eq!(encoded[4], VERSION);
        // Flags (1 byte)
        assert_eq!(encoded[5], 0x03);
        // MsgType (2 bytes LE)
        assert_eq!(&encoded[6..8], &(0x0002u16).to_le_bytes());
        // RequestID (4 bytes LE)
        assert_eq!(&encoded[8..12], &(0x12345678u32).to_le_bytes());
        // PayloadLen (4 bytes LE)
        assert_eq!(&encoded[12..16], &(2u32).to_le_bytes());
        // Payload
        assert_eq!(&encoded[16..], &[0xAA, 0xBB]);
    }
}
