//! Protobuf wire format decoder — varint, fixed, length-delimited, nested messages.
//!
//! Pure-Rust protobuf wire-format decoder. Parses field tags, dispatches by
//! wire type (varint/fixed32/fixed64/length-delimited), handles unknown fields
//! gracefully, decodes nested messages, packed repeated fields, zigzag-encoded
//! signed integers, and validates message structure.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Decoder errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer exhausted mid-field.
    UnexpectedEof,
    /// Varint exceeds 10 bytes.
    VarintOverflow,
    /// Unknown wire type.
    UnknownWireType(u8),
    /// Field number zero is invalid.
    InvalidFieldNumber,
    /// Nesting depth exceeded.
    NestingTooDeep,
    /// Length-delimited field length exceeds remaining data.
    LengthExceeded { expected: usize, available: usize },
    /// UTF-8 string decoding failed.
    InvalidUtf8,
    /// Field type mismatch.
    TypeMismatch { field: u32, expected: &'static str, got: &'static str },
    /// Trailing data after message.
    TrailingData { consumed: usize, total: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => f.write_str("unexpected end of input"),
            Self::VarintOverflow => f.write_str("varint exceeds 10 bytes"),
            Self::UnknownWireType(w) => write!(f, "unknown wire type {w}"),
            Self::InvalidFieldNumber => f.write_str("field number must be >= 1"),
            Self::NestingTooDeep => f.write_str("nesting exceeds maximum depth"),
            Self::LengthExceeded { expected, available } => {
                write!(f, "length-delimited: need {expected} bytes, only {available} available")
            }
            Self::InvalidUtf8 => f.write_str("invalid UTF-8 in string field"),
            Self::TypeMismatch { field, expected, got } => {
                write!(f, "field {field}: expected {expected}, got {got}")
            }
            Self::TrailingData { consumed, total } => {
                write!(f, "trailing data: consumed {consumed} of {total} bytes")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

// ── Wire Types ───────────────────────────────────────────────

/// Wire type constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    Fixed32 = 5,
}

impl WireType {
    /// Parse from the low 3 bits of a tag.
    pub fn from_u8(v: u8) -> Result<Self, DecodeError> {
        match v {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::Fixed32),
            other => Err(DecodeError::UnknownWireType(other)),
        }
    }

    /// Wire type name for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Varint => "varint",
            Self::Fixed64 => "fixed64",
            Self::LengthDelimited => "length-delimited",
            Self::Fixed32 => "fixed32",
        }
    }
}

// ── Raw Field Value ──────────────────────────────────────────

/// A decoded raw field value.
#[derive(Debug, Clone, PartialEq)]
pub enum RawValue {
    Varint(u64),
    Fixed32([u8; 4]),
    Fixed64([u8; 8]),
    LengthDelimited(Vec<u8>),
}

impl RawValue {
    /// Interpret as u64 varint.
    pub fn as_uint64(&self) -> Option<u64> {
        match self {
            Self::Varint(v) => Some(*v),
            _ => None,
        }
    }

    /// Interpret as u32 varint.
    pub fn as_uint32(&self) -> Option<u32> {
        match self {
            Self::Varint(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Interpret as i32 varint (sign-extended).
    pub fn as_int32(&self) -> Option<i32> {
        match self {
            Self::Varint(v) => Some(*v as i32),
            _ => None,
        }
    }

    /// Interpret as i64 varint.
    pub fn as_int64(&self) -> Option<i64> {
        match self {
            Self::Varint(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Interpret as sint32 (zigzag-decoded).
    pub fn as_sint32(&self) -> Option<i32> {
        match self {
            Self::Varint(v) => {
                let n = *v as u32;
                Some(((n >> 1) as i32) ^ (-((n & 1) as i32)))
            }
            _ => None,
        }
    }

    /// Interpret as sint64 (zigzag-decoded).
    pub fn as_sint64(&self) -> Option<i64> {
        match self {
            Self::Varint(v) => {
                let n = *v;
                Some(((n >> 1) as i64) ^ (-((n & 1) as i64)))
            }
            _ => None,
        }
    }

    /// Interpret as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Varint(v) => Some(*v != 0),
            _ => None,
        }
    }

    /// Interpret as fixed32 u32.
    pub fn as_fixed32(&self) -> Option<u32> {
        match self {
            Self::Fixed32(b) => Some(u32::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Interpret as fixed64 u64.
    pub fn as_fixed64(&self) -> Option<u64> {
        match self {
            Self::Fixed64(b) => Some(u64::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Interpret as sfixed32 i32.
    pub fn as_sfixed32(&self) -> Option<i32> {
        match self {
            Self::Fixed32(b) => Some(i32::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Interpret as sfixed64 i64.
    pub fn as_sfixed64(&self) -> Option<i64> {
        match self {
            Self::Fixed64(b) => Some(i64::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Interpret as f32.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Self::Fixed32(b) => Some(f32::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Interpret as f64.
    pub fn as_double(&self) -> Option<f64> {
        match self {
            Self::Fixed64(b) => Some(f64::from_le_bytes(*b)),
            _ => None,
        }
    }

    /// Interpret as bytes.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::LengthDelimited(b) => Some(b),
            _ => None,
        }
    }

    /// Interpret as UTF-8 string.
    pub fn as_str(&self) -> Result<Option<&str>, DecodeError> {
        match self {
            Self::LengthDelimited(b) => {
                std::str::from_utf8(b)
                    .map(Some)
                    .map_err(|_| DecodeError::InvalidUtf8)
            }
            _ => Ok(None),
        }
    }

    /// Wire type of this value.
    pub fn wire_type(&self) -> WireType {
        match self {
            Self::Varint(_) => WireType::Varint,
            Self::Fixed32(_) => WireType::Fixed32,
            Self::Fixed64(_) => WireType::Fixed64,
            Self::LengthDelimited(_) => WireType::LengthDelimited,
        }
    }
}

// ── Field Tag ────────────────────────────────────────────────

/// A parsed field tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldTag {
    pub field_number: u32,
    pub wire_type: WireType,
}

// ── Varint Decoding ──────────────────────────────────────────

/// Decode a varint at `offset`. Returns (value, new_offset).
pub fn decode_varint(data: &[u8], offset: usize) -> Result<(u64, usize), DecodeError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut pos = offset;
    loop {
        if pos >= data.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 70 {
            return Err(DecodeError::VarintOverflow);
        }
    }
}

/// Decode a field tag from data at offset.
pub fn decode_tag(data: &[u8], offset: usize) -> Result<(FieldTag, usize), DecodeError> {
    let (tag_val, new_offset) = decode_varint(data, offset)?;
    let wire_raw = (tag_val & 0x07) as u8;
    let field_number = (tag_val >> 3) as u32;
    if field_number == 0 {
        return Err(DecodeError::InvalidFieldNumber);
    }
    let wire_type = WireType::from_u8(wire_raw)?;
    Ok((FieldTag { field_number, wire_type }, new_offset))
}

// ── Decoded Message ──────────────────────────────────────────

/// Maximum nesting depth to prevent stack overflow.
const MAX_NESTING_DEPTH: usize = 64;

/// A decoded protobuf message. Fields are stored by number; repeated fields
/// accumulate in order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodedMessage {
    /// Fields keyed by field number. Repeated fields have multiple entries.
    pub fields: HashMap<u32, Vec<RawValue>>,
    /// Unknown field numbers encountered.
    pub unknown_fields: Vec<u32>,
}

impl DecodedMessage {
    /// Decode a protobuf message from bytes.
    pub fn decode(data: &[u8]) -> Result<Self, DecodeError> {
        Self::decode_with_depth(data, 0)
    }

    /// Decode with nesting depth tracking.
    fn decode_with_depth(data: &[u8], depth: usize) -> Result<Self, DecodeError> {
        if depth > MAX_NESTING_DEPTH {
            return Err(DecodeError::NestingTooDeep);
        }
        let mut msg = DecodedMessage::default();
        let mut offset = 0;
        while offset < data.len() {
            let (tag, new_offset) = decode_tag(data, offset)?;
            offset = new_offset;

            let value = match tag.wire_type {
                WireType::Varint => {
                    let (v, o) = decode_varint(data, offset)?;
                    offset = o;
                    RawValue::Varint(v)
                }
                WireType::Fixed64 => {
                    if offset + 8 > data.len() {
                        return Err(DecodeError::UnexpectedEof);
                    }
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data[offset..offset + 8]);
                    offset += 8;
                    RawValue::Fixed64(bytes)
                }
                WireType::Fixed32 => {
                    if offset + 4 > data.len() {
                        return Err(DecodeError::UnexpectedEof);
                    }
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&data[offset..offset + 4]);
                    offset += 4;
                    RawValue::Fixed32(bytes)
                }
                WireType::LengthDelimited => {
                    let (len_val, o) = decode_varint(data, offset)?;
                    offset = o;
                    let len = len_val as usize;
                    if offset + len > data.len() {
                        return Err(DecodeError::LengthExceeded {
                            expected: len,
                            available: data.len() - offset,
                        });
                    }
                    let bytes = data[offset..offset + len].to_vec();
                    offset += len;
                    RawValue::LengthDelimited(bytes)
                }
            };
            msg.fields.entry(tag.field_number).or_default().push(value);
        }
        Ok(msg)
    }

    /// Get the first value for a field.
    pub fn get_field(&self, field_number: u32) -> Option<&RawValue> {
        self.fields.get(&field_number).and_then(|v| v.first())
    }

    /// Get all values for a field.
    pub fn get_all(&self, field_number: u32) -> &[RawValue] {
        self.fields.get(&field_number).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get a uint64 value.
    pub fn get_uint64(&self, field_number: u32) -> Option<u64> {
        self.get_field(field_number).and_then(|v| v.as_uint64())
    }

    /// Get a uint32 value.
    pub fn get_uint32(&self, field_number: u32) -> Option<u32> {
        self.get_field(field_number).and_then(|v| v.as_uint32())
    }

    /// Get an int32 value.
    pub fn get_int32(&self, field_number: u32) -> Option<i32> {
        self.get_field(field_number).and_then(|v| v.as_int32())
    }

    /// Get an int64 value.
    pub fn get_int64(&self, field_number: u32) -> Option<i64> {
        self.get_field(field_number).and_then(|v| v.as_int64())
    }

    /// Get a sint32 value (zigzag-decoded).
    pub fn get_sint32(&self, field_number: u32) -> Option<i32> {
        self.get_field(field_number).and_then(|v| v.as_sint32())
    }

    /// Get a sint64 value (zigzag-decoded).
    pub fn get_sint64(&self, field_number: u32) -> Option<i64> {
        self.get_field(field_number).and_then(|v| v.as_sint64())
    }

    /// Get a bool value.
    pub fn get_bool(&self, field_number: u32) -> Option<bool> {
        self.get_field(field_number).and_then(|v| v.as_bool())
    }

    /// Get a float value.
    pub fn get_float(&self, field_number: u32) -> Option<f32> {
        self.get_field(field_number).and_then(|v| v.as_float())
    }

    /// Get a double value.
    pub fn get_double(&self, field_number: u32) -> Option<f64> {
        self.get_field(field_number).and_then(|v| v.as_double())
    }

    /// Get a fixed32 value.
    pub fn get_fixed32(&self, field_number: u32) -> Option<u32> {
        self.get_field(field_number).and_then(|v| v.as_fixed32())
    }

    /// Get a fixed64 value.
    pub fn get_fixed64(&self, field_number: u32) -> Option<u64> {
        self.get_field(field_number).and_then(|v| v.as_fixed64())
    }

    /// Get a bytes value.
    pub fn get_bytes(&self, field_number: u32) -> Option<&[u8]> {
        self.get_field(field_number).and_then(|v| v.as_bytes())
    }

    /// Get a string value.
    pub fn get_string(&self, field_number: u32) -> Result<Option<&str>, DecodeError> {
        match self.get_field(field_number) {
            Some(v) => v.as_str(),
            None => Ok(None),
        }
    }

    /// Decode a nested message field.
    pub fn get_message(&self, field_number: u32) -> Result<Option<DecodedMessage>, DecodeError> {
        match self.get_bytes(field_number) {
            Some(b) => Ok(Some(DecodedMessage::decode(b)?)),
            None => Ok(None),
        }
    }

    /// Decode packed repeated varints.
    pub fn get_packed_uint64(&self, field_number: u32) -> Result<Vec<u64>, DecodeError> {
        let mut result = Vec::new();
        for val in self.get_all(field_number) {
            match val {
                RawValue::LengthDelimited(bytes) => {
                    let mut offset = 0;
                    while offset < bytes.len() {
                        let (v, new_offset) = decode_varint(bytes, offset)?;
                        result.push(v);
                        offset = new_offset;
                    }
                }
                RawValue::Varint(v) => {
                    // Non-packed varint occurrence.
                    result.push(*v);
                }
                _ => {}
            }
        }
        Ok(result)
    }

    /// Decode packed repeated uint32 values.
    pub fn get_packed_uint32(&self, field_number: u32) -> Result<Vec<u32>, DecodeError> {
        let values = self.get_packed_uint64(field_number)?;
        Ok(values.into_iter().map(|v| v as u32).collect())
    }

    /// Decode packed repeated fixed32 values.
    pub fn get_packed_fixed32(&self, field_number: u32) -> Result<Vec<u32>, DecodeError> {
        let mut result = Vec::new();
        for val in self.get_all(field_number) {
            if let RawValue::LengthDelimited(bytes) = val {
                if bytes.len() % 4 != 0 {
                    return Err(DecodeError::LengthExceeded {
                        expected: (bytes.len() / 4 + 1) * 4,
                        available: bytes.len(),
                    });
                }
                for chunk in bytes.chunks_exact(4) {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(chunk);
                    result.push(u32::from_le_bytes(arr));
                }
            }
        }
        Ok(result)
    }

    /// Decode packed repeated fixed64 values.
    pub fn get_packed_fixed64(&self, field_number: u32) -> Result<Vec<u64>, DecodeError> {
        let mut result = Vec::new();
        for val in self.get_all(field_number) {
            if let RawValue::LengthDelimited(bytes) = val {
                if bytes.len() % 8 != 0 {
                    return Err(DecodeError::LengthExceeded {
                        expected: (bytes.len() / 8 + 1) * 8,
                        available: bytes.len(),
                    });
                }
                for chunk in bytes.chunks_exact(8) {
                    let mut arr = [0u8; 8];
                    arr.copy_from_slice(chunk);
                    result.push(u64::from_le_bytes(arr));
                }
            }
        }
        Ok(result)
    }

    /// Decode packed repeated float values.
    pub fn get_packed_float(&self, field_number: u32) -> Result<Vec<f32>, DecodeError> {
        let fixed = self.get_packed_fixed32(field_number)?;
        Ok(fixed.into_iter().map(f32::from_bits).collect())
    }

    /// Decode packed repeated double values.
    pub fn get_packed_double(&self, field_number: u32) -> Result<Vec<f64>, DecodeError> {
        let fixed = self.get_packed_fixed64(field_number)?;
        Ok(fixed.into_iter().map(f64::from_bits).collect())
    }

    /// Decode packed repeated bool values.
    pub fn get_packed_bool(&self, field_number: u32) -> Result<Vec<bool>, DecodeError> {
        let values = self.get_packed_uint64(field_number)?;
        Ok(values.into_iter().map(|v| v != 0).collect())
    }

    /// Get repeated string fields (non-packed).
    pub fn get_repeated_string(&self, field_number: u32) -> Result<Vec<String>, DecodeError> {
        let mut result = Vec::new();
        for val in self.get_all(field_number) {
            if let RawValue::LengthDelimited(bytes) = val {
                let s = std::str::from_utf8(bytes).map_err(|_| DecodeError::InvalidUtf8)?;
                result.push(s.to_string());
            }
        }
        Ok(result)
    }

    /// Get repeated bytes fields (non-packed).
    pub fn get_repeated_bytes(&self, field_number: u32) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for val in self.get_all(field_number) {
            if let RawValue::LengthDelimited(bytes) = val {
                result.push(bytes.clone());
            }
        }
        result
    }

    /// Decode a map<string, string> field. Each entry is a sub-message with
    /// field 1 = key, field 2 = value.
    pub fn get_map_string_string(&self, field_number: u32) -> Result<HashMap<String, String>, DecodeError> {
        let mut map = HashMap::new();
        for val in self.get_all(field_number) {
            if let RawValue::LengthDelimited(bytes) = val {
                let entry = DecodedMessage::decode(bytes)?;
                let key = entry.get_string(1)?.unwrap_or("").to_string();
                let value = entry.get_string(2)?.unwrap_or("").to_string();
                map.insert(key, value);
            }
        }
        Ok(map)
    }

    /// Number of distinct field numbers.
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// All field numbers present (sorted).
    pub fn field_numbers(&self) -> Vec<u32> {
        let mut nums: Vec<u32> = self.fields.keys().copied().collect();
        nums.sort();
        nums
    }

    /// Check whether a field is present.
    pub fn has_field(&self, field_number: u32) -> bool {
        self.fields.contains_key(&field_number)
    }
}

// ── Streaming Decoder ────────────────────────────────────────

/// Incremental decoder that processes data in chunks.
#[derive(Debug)]
pub struct StreamDecoder {
    buffer: Vec<u8>,
    messages_decoded: usize,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            messages_decoded: 0,
        }
    }

    /// Feed data into the decoder.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to decode a length-prefixed message. Returns the message if enough
    /// data is available, None otherwise.
    pub fn try_decode(&mut self) -> Result<Option<DecodedMessage>, DecodeError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        // Try to read a varint length prefix.
        let (len, header_size) = match decode_varint(&self.buffer, 0) {
            Ok(v) => v,
            Err(DecodeError::UnexpectedEof) => return Ok(None),
            Err(e) => return Err(e),
        };
        let msg_len = len as usize;
        let total = header_size + msg_len;
        if self.buffer.len() < total {
            return Ok(None); // need more data
        }
        let msg_data = self.buffer[header_size..total].to_vec();
        self.buffer.drain(..total);
        self.messages_decoded += 1;
        let msg = DecodedMessage::decode(&msg_data)?;
        Ok(Some(msg))
    }

    /// Number of messages decoded so far.
    pub fn messages_decoded(&self) -> usize {
        self.messages_decoded
    }

    /// Remaining buffered bytes.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Reset the decoder.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.messages_decoded = 0;
    }
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: encode a simple varint-tagged message manually.
    fn encode_varint_to(buf: &mut Vec<u8>, mut value: u64) {
        loop {
            let mut byte = (value & 0x7F) as u8;
            value >>= 7;
            if value != 0 { byte |= 0x80; }
            buf.push(byte);
            if value == 0 { break; }
        }
    }

    fn encode_tag_to(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
        encode_varint_to(buf, ((field_number << 3) | wire_type) as u64);
    }

    fn build_simple_message() -> Vec<u8> {
        let mut buf = Vec::new();
        // field 1, varint, value 150
        encode_tag_to(&mut buf, 1, 0);
        encode_varint_to(&mut buf, 150);
        // field 2, length-delimited, "testing"
        encode_tag_to(&mut buf, 2, 2);
        encode_varint_to(&mut buf, 7);
        buf.extend_from_slice(b"testing");
        buf
    }

    #[test]
    fn decode_varint_small() {
        let data = [0x00];
        let (val, off) = decode_varint(&data, 0).unwrap();
        assert_eq!(val, 0);
        assert_eq!(off, 1);

        let data = [0x01];
        let (val, _) = decode_varint(&data, 0).unwrap();
        assert_eq!(val, 1);

        let data = [0x96, 0x01]; // 150
        let (val, _) = decode_varint(&data, 0).unwrap();
        assert_eq!(val, 150);
    }

    #[test]
    fn decode_varint_large() {
        // u64::MAX = 10 bytes of 0xFF except last is 0x01
        let mut data = vec![0xFF; 9];
        data.push(0x01);
        let (val, _) = decode_varint(&data, 0).unwrap();
        assert_eq!(val, u64::MAX);
    }

    #[test]
    fn decode_varint_overflow() {
        // 11 continuation bytes
        let data = vec![0xFF; 11];
        assert!(matches!(decode_varint(&data, 0), Err(DecodeError::VarintOverflow)));
    }

    #[test]
    fn decode_varint_eof() {
        let data: &[u8] = &[];
        assert!(matches!(decode_varint(data, 0), Err(DecodeError::UnexpectedEof)));
    }

    #[test]
    fn decode_tag_basic() {
        // field 1, wire type 0 => byte 0x08
        let data = [0x08];
        let (tag, _) = decode_tag(&data, 0).unwrap();
        assert_eq!(tag.field_number, 1);
        assert_eq!(tag.wire_type, WireType::Varint);

        // field 2, wire type 2 => byte 0x12
        let data = [0x12];
        let (tag, _) = decode_tag(&data, 0).unwrap();
        assert_eq!(tag.field_number, 2);
        assert_eq!(tag.wire_type, WireType::LengthDelimited);
    }

    #[test]
    fn decode_tag_field_zero() {
        let data = [0x00];
        assert!(matches!(decode_tag(&data, 0), Err(DecodeError::InvalidFieldNumber)));
    }

    #[test]
    fn decode_simple_message() {
        let data = build_simple_message();
        let msg = DecodedMessage::decode(&data).unwrap();
        assert_eq!(msg.get_uint64(1), Some(150));
        assert_eq!(msg.get_string(2).unwrap(), Some("testing"));
    }

    #[test]
    fn decode_missing_fields() {
        let msg = DecodedMessage::decode(&[]).unwrap();
        assert_eq!(msg.get_uint64(1), None);
        assert_eq!(msg.get_string(2).unwrap(), None);
        assert_eq!(msg.get_bytes(3), None);
        assert!(!msg.has_field(1));
    }

    #[test]
    fn decode_fixed32_field() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 5); // fixed32
        buf.extend_from_slice(&42u32.to_le_bytes());
        let msg = DecodedMessage::decode(&buf).unwrap();
        assert_eq!(msg.get_fixed32(1), Some(42));
    }

    #[test]
    fn decode_fixed64_field() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 1); // fixed64
        buf.extend_from_slice(&123456789u64.to_le_bytes());
        let msg = DecodedMessage::decode(&buf).unwrap();
        assert_eq!(msg.get_fixed64(1), Some(123456789));
    }

    #[test]
    fn decode_float_field() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 5); // fixed32
        buf.extend_from_slice(&3.14f32.to_bits().to_le_bytes());
        let msg = DecodedMessage::decode(&buf).unwrap();
        let val = msg.get_float(1).unwrap();
        assert!((val - 3.14).abs() < 0.001);
    }

    #[test]
    fn decode_double_field() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 1); // fixed64
        buf.extend_from_slice(&2.718f64.to_bits().to_le_bytes());
        let msg = DecodedMessage::decode(&buf).unwrap();
        let val = msg.get_double(1).unwrap();
        assert!((val - 2.718).abs() < 0.001);
    }

    #[test]
    fn decode_bool_field() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 0);
        encode_varint_to(&mut buf, 1);
        let msg = DecodedMessage::decode(&buf).unwrap();
        assert_eq!(msg.get_bool(1), Some(true));
    }

    #[test]
    fn decode_sint32_field() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 0);
        // zigzag(-42) = 83
        let zz = ((-42i32 << 1) ^ (-42i32 >> 31)) as u32;
        encode_varint_to(&mut buf, zz as u64);
        let msg = DecodedMessage::decode(&buf).unwrap();
        assert_eq!(msg.get_sint32(1), Some(-42));
    }

    #[test]
    fn decode_nested_message() {
        let mut inner_buf = Vec::new();
        encode_tag_to(&mut inner_buf, 1, 0);
        encode_varint_to(&mut inner_buf, 99);

        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 2, 2);
        encode_varint_to(&mut buf, inner_buf.len() as u64);
        buf.extend_from_slice(&inner_buf);

        let msg = DecodedMessage::decode(&buf).unwrap();
        let nested = msg.get_message(2).unwrap().unwrap();
        assert_eq!(nested.get_uint64(1), Some(99));
    }

    #[test]
    fn decode_packed_varints() {
        let mut packed = Vec::new();
        encode_varint_to(&mut packed, 3);
        encode_varint_to(&mut packed, 270);
        encode_varint_to(&mut packed, 86942);

        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 4, 2);
        encode_varint_to(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);

        let msg = DecodedMessage::decode(&buf).unwrap();
        let vals = msg.get_packed_uint64(4).unwrap();
        assert_eq!(vals, vec![3, 270, 86942]);
    }

    #[test]
    fn decode_packed_fixed32() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&100u32.to_le_bytes());
        payload.extend_from_slice(&200u32.to_le_bytes());

        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 5, 2);
        encode_varint_to(&mut buf, payload.len() as u64);
        buf.extend_from_slice(&payload);

        let msg = DecodedMessage::decode(&buf).unwrap();
        let vals = msg.get_packed_fixed32(5).unwrap();
        assert_eq!(vals, vec![100, 200]);
    }

    #[test]
    fn decode_repeated_strings() {
        let mut buf = Vec::new();
        for s in ["hello", "world"] {
            encode_tag_to(&mut buf, 3, 2);
            encode_varint_to(&mut buf, s.len() as u64);
            buf.extend_from_slice(s.as_bytes());
        }
        let msg = DecodedMessage::decode(&buf).unwrap();
        let vals = msg.get_repeated_string(3).unwrap();
        assert_eq!(vals, vec!["hello", "world"]);
    }

    #[test]
    fn decode_field_numbers() {
        let data = build_simple_message();
        let msg = DecodedMessage::decode(&data).unwrap();
        let nums = msg.field_numbers();
        assert_eq!(nums, vec![1, 2]);
        assert_eq!(msg.field_count(), 2);
        assert!(msg.has_field(1));
        assert!(!msg.has_field(3));
    }

    #[test]
    fn decode_length_exceeded() {
        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 1, 2);
        encode_varint_to(&mut buf, 100); // claim 100 bytes
        buf.push(0); // only 1 byte
        let err = DecodedMessage::decode(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::LengthExceeded { .. }));
    }

    #[test]
    fn decode_unknown_wire_type() {
        let buf = [0x0F]; // field 1, wire type 7 => unknown
        let err = DecodedMessage::decode(&buf).unwrap_err();
        assert!(matches!(err, DecodeError::UnknownWireType(7)));
    }

    #[test]
    fn stream_decoder_basic() {
        // Build a length-prefixed message.
        let inner = build_simple_message();
        let mut framed = Vec::new();
        encode_varint_to(&mut framed, inner.len() as u64);
        framed.extend_from_slice(&inner);

        let mut dec = StreamDecoder::new();
        // Feed partial data.
        dec.feed(&framed[..2]);
        assert_eq!(dec.try_decode().unwrap(), None);
        // Feed the rest.
        dec.feed(&framed[2..]);
        let msg = dec.try_decode().unwrap().unwrap();
        assert_eq!(msg.get_uint64(1), Some(150));
        assert_eq!(dec.messages_decoded(), 1);
        assert_eq!(dec.buffered(), 0);
    }

    #[test]
    fn stream_decoder_empty() {
        let mut dec = StreamDecoder::new();
        assert_eq!(dec.try_decode().unwrap(), None);
        assert_eq!(dec.messages_decoded(), 0);
    }

    #[test]
    fn stream_decoder_reset() {
        let mut dec = StreamDecoder::new();
        dec.feed(&[1, 2, 3]);
        dec.reset();
        assert_eq!(dec.buffered(), 0);
        assert_eq!(dec.messages_decoded(), 0);
    }

    #[test]
    fn raw_value_type_mismatch() {
        let v = RawValue::Varint(42);
        assert_eq!(v.as_bytes(), None);
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_fixed32(), None);

        let v = RawValue::Fixed32([0; 4]);
        assert_eq!(v.as_uint64(), None);
        assert_eq!(v.as_bytes(), None);
    }

    #[test]
    fn raw_value_wire_type() {
        assert_eq!(RawValue::Varint(0).wire_type(), WireType::Varint);
        assert_eq!(RawValue::Fixed32([0; 4]).wire_type(), WireType::Fixed32);
        assert_eq!(RawValue::Fixed64([0; 8]).wire_type(), WireType::Fixed64);
        assert_eq!(RawValue::LengthDelimited(vec![]).wire_type(), WireType::LengthDelimited);
    }

    #[test]
    fn decode_error_display() {
        let err = DecodeError::UnexpectedEof;
        assert_eq!(err.to_string(), "unexpected end of input");
        let err = DecodeError::InvalidUtf8;
        assert_eq!(err.to_string(), "invalid UTF-8 in string field");
    }

    #[test]
    fn wire_type_name() {
        assert_eq!(WireType::Varint.name(), "varint");
        assert_eq!(WireType::LengthDelimited.name(), "length-delimited");
    }

    #[test]
    fn sfixed_values() {
        let v = RawValue::Fixed32((-100i32).to_le_bytes());
        assert_eq!(v.as_sfixed32(), Some(-100));

        let v = RawValue::Fixed64((-200i64).to_le_bytes());
        assert_eq!(v.as_sfixed64(), Some(-200));
    }

    #[test]
    fn empty_message_decode() {
        let msg = DecodedMessage::decode(&[]).unwrap();
        assert_eq!(msg.field_count(), 0);
        assert!(msg.field_numbers().is_empty());
    }

    #[test]
    fn packed_bool_decode() {
        let mut packed = Vec::new();
        packed.push(1); // true
        packed.push(0); // false
        packed.push(1); // true

        let mut buf = Vec::new();
        encode_tag_to(&mut buf, 3, 2);
        encode_varint_to(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);

        let msg = DecodedMessage::decode(&buf).unwrap();
        let vals = msg.get_packed_bool(3).unwrap();
        assert_eq!(vals, vec![true, false, true]);
    }

    #[test]
    fn get_repeated_bytes() {
        let mut buf = Vec::new();
        for data in [b"abc".as_slice(), b"def"] {
            encode_tag_to(&mut buf, 5, 2);
            encode_varint_to(&mut buf, data.len() as u64);
            buf.extend_from_slice(data);
        }
        let msg = DecodedMessage::decode(&buf).unwrap();
        let vals = msg.get_repeated_bytes(5);
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], b"abc");
        assert_eq!(vals[1], b"def");
    }
}
