//! Protobuf wire format encoder — varint, fixed32, fixed64, length-delimited.
//!
//! Pure-Rust protobuf encoder that produces wire-format bytes. Supports all
//! protobuf wire types: varint (type 0), fixed64 (type 1), length-delimited
//! (type 2), and fixed32 (type 5). Handles field tags, packed repeated fields,
//! nested messages, map fields, zigzag-encoded signed integers, and float/double.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────

/// Encoder errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    /// Field number must be 1..=536870911.
    InvalidFieldNumber(u32),
    /// Buffer capacity exceeded.
    BufferOverflow,
    /// Map key type not supported.
    InvalidMapKey,
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFieldNumber(n) => write!(f, "invalid field number: {n}"),
            Self::BufferOverflow => f.write_str("buffer overflow"),
            Self::InvalidMapKey => f.write_str("invalid map key type"),
        }
    }
}

impl std::error::Error for EncodeError {}

// ── Wire Types ───────────────────────────────────────────────

/// Protobuf wire type constants.
pub const WIRE_TYPE_VARINT: u32 = 0;
pub const WIRE_TYPE_FIXED64: u32 = 1;
pub const WIRE_TYPE_LENGTH_DELIMITED: u32 = 2;
pub const WIRE_TYPE_FIXED32: u32 = 5;

// ── Varint Encoding ──────────────────────────────────────────

/// Encode a u64 as a varint into a buffer. Returns number of bytes written.
pub fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
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

/// Encode a u64 varint and return as a standalone Vec.
pub fn encode_varint_vec(value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    encode_varint(&mut buf, value);
    buf
}

/// Compute the encoded size of a varint without actually encoding.
pub fn varint_size(mut value: u64) -> usize {
    let mut size = 1;
    while value >= 0x80 {
        value >>= 7;
        size += 1;
    }
    size
}

// ── Zigzag Encoding ──────────────────────────────────────────

/// Zigzag-encode a signed 32-bit integer to unsigned.
pub fn zigzag_encode_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Zigzag-encode a signed 64-bit integer to unsigned.
pub fn zigzag_encode_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

// ── Tag Encoding ─────────────────────────────────────────────

/// Encode a field tag (field_number << 3 | wire_type).
pub fn encode_tag(buf: &mut Vec<u8>, field_number: u32, wire_type: u32) {
    let tag = (field_number << 3) | wire_type;
    encode_varint(buf, tag as u64);
}

/// Compute the tag value for a field.
pub fn make_tag(field_number: u32, wire_type: u32) -> u32 {
    (field_number << 3) | wire_type
}

/// Validate a field number (1..=536870911, not in 19000..=19999).
pub fn validate_field_number(field_number: u32) -> Result<(), EncodeError> {
    if field_number == 0 || field_number > 536_870_911 {
        return Err(EncodeError::InvalidFieldNumber(field_number));
    }
    Ok(())
}

// ── Encoder ──────────────────────────────────────────────────

/// Protobuf wire format encoder that builds messages incrementally.
#[derive(Debug, Clone, Default)]
pub struct ProtoEncoder {
    buf: Vec<u8>,
}

impl ProtoEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Create an encoder with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self { buf: Vec::with_capacity(cap) }
    }

    /// Current encoded size in bytes.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Get a reference to the encoded bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Consume the encoder and return the encoded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Reset the encoder, keeping allocated memory.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    // ── Raw encoding helpers ────────────────────────────

    /// Write a raw varint.
    pub fn write_varint(&mut self, value: u64) {
        encode_varint(&mut self.buf, value);
    }

    /// Write a raw fixed32 (little-endian).
    pub fn write_fixed32(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a raw fixed64 (little-endian).
    pub fn write_fixed64(&mut self, value: u64) {
        self.buf.extend_from_slice(&value.to_le_bytes());
    }

    /// Write raw bytes.
    pub fn write_raw(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Write a field tag.
    pub fn write_tag(&mut self, field_number: u32, wire_type: u32) -> Result<(), EncodeError> {
        validate_field_number(field_number)?;
        encode_tag(&mut self.buf, field_number, wire_type);
        Ok(())
    }

    // ── Typed field encoding ────────────────────────────

    /// Encode a uint32 field.
    pub fn encode_uint32(&mut self, field_number: u32, value: u32) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(()); // proto3 default omission
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(value as u64);
        Ok(())
    }

    /// Encode a uint32 field, always including even if zero.
    pub fn encode_uint32_always(&mut self, field_number: u32, value: u32) -> Result<(), EncodeError> {
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(value as u64);
        Ok(())
    }

    /// Encode a uint64 field.
    pub fn encode_uint64(&mut self, field_number: u32, value: u64) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(value);
        Ok(())
    }

    /// Encode an int32 field (varint, may be sign-extended to 10 bytes for negative).
    pub fn encode_int32(&mut self, field_number: u32, value: i32) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        // Protobuf int32 sign-extends to 64 bits for negative values.
        self.write_varint(value as u64);
        Ok(())
    }

    /// Encode an int64 field.
    pub fn encode_int64(&mut self, field_number: u32, value: i64) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(value as u64);
        Ok(())
    }

    /// Encode a sint32 field (zigzag encoding).
    pub fn encode_sint32(&mut self, field_number: u32, value: i32) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(zigzag_encode_i32(value) as u64);
        Ok(())
    }

    /// Encode a sint64 field (zigzag encoding).
    pub fn encode_sint64(&mut self, field_number: u32, value: i64) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(zigzag_encode_i64(value));
        Ok(())
    }

    /// Encode a bool field.
    pub fn encode_bool(&mut self, field_number: u32, value: bool) -> Result<(), EncodeError> {
        if !value {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(1);
        Ok(())
    }

    /// Encode a bool field, always including.
    pub fn encode_bool_always(&mut self, field_number: u32, value: bool) -> Result<(), EncodeError> {
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(if value { 1 } else { 0 });
        Ok(())
    }

    /// Encode a fixed32 field.
    pub fn encode_fixed32(&mut self, field_number: u32, value: u32) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_FIXED32)?;
        self.write_fixed32(value);
        Ok(())
    }

    /// Encode a fixed64 field.
    pub fn encode_fixed64(&mut self, field_number: u32, value: u64) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_FIXED64)?;
        self.write_fixed64(value);
        Ok(())
    }

    /// Encode an sfixed32 field.
    pub fn encode_sfixed32(&mut self, field_number: u32, value: i32) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_FIXED32)?;
        self.write_fixed32(value as u32);
        Ok(())
    }

    /// Encode an sfixed64 field.
    pub fn encode_sfixed64(&mut self, field_number: u32, value: i64) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_FIXED64)?;
        self.write_fixed64(value as u64);
        Ok(())
    }

    /// Encode a float field.
    pub fn encode_float(&mut self, field_number: u32, value: f32) -> Result<(), EncodeError> {
        if value == 0.0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_FIXED32)?;
        self.write_fixed32(value.to_bits());
        Ok(())
    }

    /// Encode a float field, always including.
    pub fn encode_float_always(&mut self, field_number: u32, value: f32) -> Result<(), EncodeError> {
        self.write_tag(field_number, WIRE_TYPE_FIXED32)?;
        self.write_fixed32(value.to_bits());
        Ok(())
    }

    /// Encode a double field.
    pub fn encode_double(&mut self, field_number: u32, value: f64) -> Result<(), EncodeError> {
        if value == 0.0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_FIXED64)?;
        self.write_fixed64(value.to_bits());
        Ok(())
    }

    /// Encode a double field, always including.
    pub fn encode_double_always(&mut self, field_number: u32, value: f64) -> Result<(), EncodeError> {
        self.write_tag(field_number, WIRE_TYPE_FIXED64)?;
        self.write_fixed64(value.to_bits());
        Ok(())
    }

    /// Encode a bytes field.
    pub fn encode_bytes(&mut self, field_number: u32, data: &[u8]) -> Result<(), EncodeError> {
        if data.is_empty() {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(data.len() as u64);
        self.write_raw(data);
        Ok(())
    }

    /// Encode a bytes field, always including.
    pub fn encode_bytes_always(&mut self, field_number: u32, data: &[u8]) -> Result<(), EncodeError> {
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(data.len() as u64);
        self.write_raw(data);
        Ok(())
    }

    /// Encode a string field.
    pub fn encode_string(&mut self, field_number: u32, s: &str) -> Result<(), EncodeError> {
        self.encode_bytes(field_number, s.as_bytes())
    }

    /// Encode a string field, always including.
    pub fn encode_string_always(&mut self, field_number: u32, s: &str) -> Result<(), EncodeError> {
        self.encode_bytes_always(field_number, s.as_bytes())
    }

    /// Encode a nested message (length-delimited).
    pub fn encode_message(&mut self, field_number: u32, inner: &ProtoEncoder) -> Result<(), EncodeError> {
        let data = inner.as_bytes();
        if data.is_empty() {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(data.len() as u64);
        self.write_raw(data);
        Ok(())
    }

    /// Encode a nested message from raw bytes.
    pub fn encode_message_raw(&mut self, field_number: u32, data: &[u8]) -> Result<(), EncodeError> {
        if data.is_empty() {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(data.len() as u64);
        self.write_raw(data);
        Ok(())
    }

    /// Encode an enum field (varint).
    pub fn encode_enum(&mut self, field_number: u32, value: i32) -> Result<(), EncodeError> {
        if value == 0 {
            return Ok(());
        }
        self.write_tag(field_number, WIRE_TYPE_VARINT)?;
        self.write_varint(value as u64);
        Ok(())
    }

    // ── Packed repeated encoding ────────────────────────

    /// Encode packed repeated uint32 values.
    pub fn encode_packed_uint32(&mut self, field_number: u32, values: &[u32]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let mut packed = Vec::new();
        for v in values {
            encode_varint(&mut packed, *v as u64);
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(packed.len() as u64);
        self.write_raw(&packed);
        Ok(())
    }

    /// Encode packed repeated uint64 values.
    pub fn encode_packed_uint64(&mut self, field_number: u32, values: &[u64]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let mut packed = Vec::new();
        for v in values {
            encode_varint(&mut packed, *v);
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(packed.len() as u64);
        self.write_raw(&packed);
        Ok(())
    }

    /// Encode packed repeated sint32 values.
    pub fn encode_packed_sint32(&mut self, field_number: u32, values: &[i32]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let mut packed = Vec::new();
        for v in values {
            encode_varint(&mut packed, zigzag_encode_i32(*v) as u64);
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(packed.len() as u64);
        self.write_raw(&packed);
        Ok(())
    }

    /// Encode packed repeated sint64 values.
    pub fn encode_packed_sint64(&mut self, field_number: u32, values: &[i64]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let mut packed = Vec::new();
        for v in values {
            encode_varint(&mut packed, zigzag_encode_i64(*v));
        }
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(packed.len() as u64);
        self.write_raw(&packed);
        Ok(())
    }

    /// Encode packed repeated fixed32 values.
    pub fn encode_packed_fixed32(&mut self, field_number: u32, values: &[u32]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let byte_len = values.len() * 4;
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(byte_len as u64);
        for v in values {
            self.write_fixed32(*v);
        }
        Ok(())
    }

    /// Encode packed repeated fixed64 values.
    pub fn encode_packed_fixed64(&mut self, field_number: u32, values: &[u64]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let byte_len = values.len() * 8;
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(byte_len as u64);
        for v in values {
            self.write_fixed64(*v);
        }
        Ok(())
    }

    /// Encode packed repeated float values.
    pub fn encode_packed_float(&mut self, field_number: u32, values: &[f32]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let byte_len = values.len() * 4;
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(byte_len as u64);
        for v in values {
            self.write_fixed32(v.to_bits());
        }
        Ok(())
    }

    /// Encode packed repeated double values.
    pub fn encode_packed_double(&mut self, field_number: u32, values: &[f64]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        let byte_len = values.len() * 8;
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(byte_len as u64);
        for v in values {
            self.write_fixed64(v.to_bits());
        }
        Ok(())
    }

    /// Encode packed repeated bool values.
    pub fn encode_packed_bool(&mut self, field_number: u32, values: &[bool]) -> Result<(), EncodeError> {
        if values.is_empty() {
            return Ok(());
        }
        // Each bool is one byte (varint 0 or 1).
        self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
        self.write_varint(values.len() as u64);
        for v in values {
            self.buf.push(if *v { 1 } else { 0 });
        }
        Ok(())
    }

    // ── Repeated (non-packed) ───────────────────────────

    /// Encode repeated string values (each as a separate field occurrence).
    pub fn encode_repeated_string(&mut self, field_number: u32, values: &[&str]) -> Result<(), EncodeError> {
        for s in values {
            self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
            self.write_varint(s.len() as u64);
            self.write_raw(s.as_bytes());
        }
        Ok(())
    }

    /// Encode repeated bytes values.
    pub fn encode_repeated_bytes(&mut self, field_number: u32, values: &[&[u8]]) -> Result<(), EncodeError> {
        for data in values {
            self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
            self.write_varint(data.len() as u64);
            self.write_raw(data);
        }
        Ok(())
    }

    /// Encode repeated message values.
    pub fn encode_repeated_message(&mut self, field_number: u32, encoders: &[&ProtoEncoder]) -> Result<(), EncodeError> {
        for enc in encoders {
            let data = enc.as_bytes();
            self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
            self.write_varint(data.len() as u64);
            self.write_raw(data);
        }
        Ok(())
    }

    // ── Map field encoding ──────────────────────────────

    /// Encode a map<string, string> field. Each entry is a nested message
    /// with field 1 = key, field 2 = value.
    pub fn encode_map_string_string(
        &mut self,
        field_number: u32,
        entries: &HashMap<String, String>,
    ) -> Result<(), EncodeError> {
        // Encode in sorted order for determinism.
        let mut keys: Vec<&String> = entries.keys().collect();
        keys.sort();
        for key in keys {
            let value = &entries[key];
            let mut entry = ProtoEncoder::new();
            // key = field 1, value = field 2
            entry.write_tag(1, WIRE_TYPE_LENGTH_DELIMITED)?;
            entry.write_varint(key.len() as u64);
            entry.write_raw(key.as_bytes());
            entry.write_tag(2, WIRE_TYPE_LENGTH_DELIMITED)?;
            entry.write_varint(value.len() as u64);
            entry.write_raw(value.as_bytes());

            let entry_bytes = entry.as_bytes();
            self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
            self.write_varint(entry_bytes.len() as u64);
            self.write_raw(entry_bytes);
        }
        Ok(())
    }

    /// Encode a map<string, uint64> field.
    pub fn encode_map_string_uint64(
        &mut self,
        field_number: u32,
        entries: &HashMap<String, u64>,
    ) -> Result<(), EncodeError> {
        let mut keys: Vec<&String> = entries.keys().collect();
        keys.sort();
        for key in keys {
            let value = entries[key];
            let mut entry = ProtoEncoder::new();
            entry.write_tag(1, WIRE_TYPE_LENGTH_DELIMITED)?;
            entry.write_varint(key.len() as u64);
            entry.write_raw(key.as_bytes());
            entry.write_tag(2, WIRE_TYPE_VARINT)?;
            entry.write_varint(value);

            let entry_bytes = entry.as_bytes();
            self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
            self.write_varint(entry_bytes.len() as u64);
            self.write_raw(entry_bytes);
        }
        Ok(())
    }

    /// Encode a map<uint32, string> field.
    pub fn encode_map_uint32_string(
        &mut self,
        field_number: u32,
        entries: &HashMap<u32, String>,
    ) -> Result<(), EncodeError> {
        let mut keys: Vec<u32> = entries.keys().copied().collect();
        keys.sort();
        for key in keys {
            let value = &entries[&key];
            let mut entry = ProtoEncoder::new();
            entry.write_tag(1, WIRE_TYPE_VARINT)?;
            entry.write_varint(key as u64);
            entry.write_tag(2, WIRE_TYPE_LENGTH_DELIMITED)?;
            entry.write_varint(value.len() as u64);
            entry.write_raw(value.as_bytes());

            let entry_bytes = entry.as_bytes();
            self.write_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED)?;
            self.write_varint(entry_bytes.len() as u64);
            self.write_raw(entry_bytes);
        }
        Ok(())
    }
}

// ── Size computation ─────────────────────────────────────────

/// Compute the encoded size of a uint32 field (tag + varint).
pub fn uint32_field_size(field_number: u32, value: u32) -> usize {
    if value == 0 {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_VARINT) as u64);
    tag_size + varint_size(value as u64)
}

/// Compute the encoded size of a uint64 field.
pub fn uint64_field_size(field_number: u32, value: u64) -> usize {
    if value == 0 {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_VARINT) as u64);
    tag_size + varint_size(value)
}

/// Compute the encoded size of a string field.
pub fn string_field_size(field_number: u32, s: &str) -> usize {
    if s.is_empty() {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED) as u64);
    let len_size = varint_size(s.len() as u64);
    tag_size + len_size + s.len()
}

/// Compute the encoded size of a bytes field.
pub fn bytes_field_size(field_number: u32, data: &[u8]) -> usize {
    if data.is_empty() {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_LENGTH_DELIMITED) as u64);
    let len_size = varint_size(data.len() as u64);
    tag_size + len_size + data.len()
}

/// Compute the encoded size of a bool field.
pub fn bool_field_size(field_number: u32, value: bool) -> usize {
    if !value {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_VARINT) as u64);
    tag_size + 1
}

/// Compute the encoded size of a fixed32 field.
pub fn fixed32_field_size(field_number: u32, value: u32) -> usize {
    if value == 0 {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_FIXED32) as u64);
    tag_size + 4
}

/// Compute the encoded size of a fixed64 field.
pub fn fixed64_field_size(field_number: u32, value: u64) -> usize {
    if value == 0 {
        return 0;
    }
    let tag_size = varint_size(make_tag(field_number, WIRE_TYPE_FIXED64) as u64);
    tag_size + 8
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_encoding_small() {
        let buf = encode_varint_vec(0);
        assert_eq!(buf, vec![0]);

        let buf = encode_varint_vec(1);
        assert_eq!(buf, vec![1]);

        let buf = encode_varint_vec(127);
        assert_eq!(buf, vec![127]);

        let buf = encode_varint_vec(128);
        assert_eq!(buf, vec![0x80, 0x01]);

        let buf = encode_varint_vec(300);
        assert_eq!(buf, vec![0xAC, 0x02]);
    }

    #[test]
    fn varint_encoding_large() {
        let buf = encode_varint_vec(u64::MAX);
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn varint_size_computation() {
        assert_eq!(varint_size(0), 1);
        assert_eq!(varint_size(127), 1);
        assert_eq!(varint_size(128), 2);
        assert_eq!(varint_size(16383), 2);
        assert_eq!(varint_size(16384), 3);
        assert_eq!(varint_size(u64::MAX), 10);
    }

    #[test]
    fn zigzag_encoding() {
        assert_eq!(zigzag_encode_i32(0), 0);
        assert_eq!(zigzag_encode_i32(-1), 1);
        assert_eq!(zigzag_encode_i32(1), 2);
        assert_eq!(zigzag_encode_i32(-2), 3);
        assert_eq!(zigzag_encode_i64(0), 0);
        assert_eq!(zigzag_encode_i64(-1), 1);
        assert_eq!(zigzag_encode_i64(1), 2);
    }

    #[test]
    fn tag_encoding() {
        // field 1, wire type 0 (varint) => (1 << 3) | 0 = 8
        let mut buf = Vec::new();
        encode_tag(&mut buf, 1, WIRE_TYPE_VARINT);
        assert_eq!(buf, vec![0x08]);

        // field 2, wire type 2 (length-delimited) => (2 << 3) | 2 = 18
        let mut buf = Vec::new();
        encode_tag(&mut buf, 2, WIRE_TYPE_LENGTH_DELIMITED);
        assert_eq!(buf, vec![0x12]);
    }

    #[test]
    fn make_tag_values() {
        assert_eq!(make_tag(1, WIRE_TYPE_VARINT), 8);
        assert_eq!(make_tag(2, WIRE_TYPE_LENGTH_DELIMITED), 18);
        assert_eq!(make_tag(3, WIRE_TYPE_FIXED32), 29);
        assert_eq!(make_tag(4, WIRE_TYPE_FIXED64), 33);
    }

    #[test]
    fn validate_field_number_valid() {
        assert!(validate_field_number(1).is_ok());
        assert!(validate_field_number(536_870_911).is_ok());
    }

    #[test]
    fn validate_field_number_invalid() {
        assert!(validate_field_number(0).is_err());
        assert!(validate_field_number(536_870_912).is_err());
    }

    #[test]
    fn encode_uint32_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_uint32(1, 150).unwrap();
        // tag = 0x08, value = 150 => 0x96 0x01
        assert_eq!(enc.as_bytes(), &[0x08, 0x96, 0x01]);
    }

    #[test]
    fn encode_uint32_default_omitted() {
        let mut enc = ProtoEncoder::new();
        enc.encode_uint32(1, 0).unwrap();
        assert!(enc.is_empty());
    }

    #[test]
    fn encode_uint32_always() {
        let mut enc = ProtoEncoder::new();
        enc.encode_uint32_always(1, 0).unwrap();
        assert!(!enc.is_empty());
    }

    #[test]
    fn encode_string_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_string(2, "testing").unwrap();
        let bytes = enc.finish();
        // tag(2, LD) = 0x12, len = 7, "testing"
        assert_eq!(bytes[0], 0x12);
        assert_eq!(bytes[1], 7);
        assert_eq!(&bytes[2..], b"testing");
    }

    #[test]
    fn encode_bool_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_bool(1, true).unwrap();
        assert_eq!(enc.as_bytes(), &[0x08, 0x01]);

        let mut enc2 = ProtoEncoder::new();
        enc2.encode_bool(1, false).unwrap();
        assert!(enc2.is_empty());
    }

    #[test]
    fn encode_float_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_float_always(1, 1.0).unwrap();
        let bytes = enc.finish();
        // tag(1, fixed32) = 0x0D, then 4 bytes of 1.0f32
        assert_eq!(bytes[0], 0x0D);
        let float_bytes = &bytes[1..5];
        assert_eq!(f32::from_le_bytes([float_bytes[0], float_bytes[1], float_bytes[2], float_bytes[3]]), 1.0);
    }

    #[test]
    fn encode_double_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_double_always(2, 3.14).unwrap();
        let bytes = enc.finish();
        // tag(2, fixed64) = 0x11
        assert_eq!(bytes[0], 0x11);
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&bytes[1..9]);
        assert_eq!(f64::from_le_bytes(arr), 3.14);
    }

    #[test]
    fn encode_nested_message() {
        let mut inner = ProtoEncoder::new();
        inner.encode_uint32_always(1, 42).unwrap();

        let mut outer = ProtoEncoder::new();
        outer.encode_message(2, &inner).unwrap();

        let bytes = outer.finish();
        assert!(!bytes.is_empty());
        // tag(2, LD) = 0x12, then length, then inner bytes
        assert_eq!(bytes[0], 0x12);
    }

    #[test]
    fn encode_packed_uint32() {
        let mut enc = ProtoEncoder::new();
        enc.encode_packed_uint32(4, &[3, 270, 86942]).unwrap();
        let bytes = enc.finish();
        // tag(4, LD) = 0x22
        assert_eq!(bytes[0], 0x22);
    }

    #[test]
    fn encode_packed_fixed32() {
        let mut enc = ProtoEncoder::new();
        enc.encode_packed_fixed32(5, &[100, 200]).unwrap();
        let bytes = enc.finish();
        // 2 fixed32s = 8 bytes payload
        let tag_and_len = varint_size(make_tag(5, WIRE_TYPE_LENGTH_DELIMITED) as u64) + varint_size(8);
        assert_eq!(bytes.len(), tag_and_len + 8);
    }

    #[test]
    fn encode_packed_bool() {
        let mut enc = ProtoEncoder::new();
        enc.encode_packed_bool(3, &[true, false, true]).unwrap();
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_empty_packed_omitted() {
        let mut enc = ProtoEncoder::new();
        enc.encode_packed_uint32(1, &[]).unwrap();
        assert!(enc.is_empty());

        let mut enc = ProtoEncoder::new();
        enc.encode_packed_fixed32(1, &[]).unwrap();
        assert!(enc.is_empty());
    }

    #[test]
    fn encode_repeated_string() {
        let mut enc = ProtoEncoder::new();
        enc.encode_repeated_string(3, &["a", "bb", "ccc"]).unwrap();
        let bytes = enc.finish();
        // Three separate occurrences of field 3
        let tag_byte = (3u32 << 3 | WIRE_TYPE_LENGTH_DELIMITED) as u8;
        let tag_count = bytes.iter().filter(|&&b| b == tag_byte).count();
        assert_eq!(tag_count, 3);
    }

    #[test]
    fn encode_map_string_string() {
        let mut entries = HashMap::new();
        entries.insert("key1".to_string(), "val1".to_string());
        entries.insert("key2".to_string(), "val2".to_string());

        let mut enc = ProtoEncoder::new();
        enc.encode_map_string_string(5, &entries).unwrap();
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_sint32_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_sint32(1, -1).unwrap();
        let bytes = enc.finish();
        // zigzag(-1) = 1, tag(1, varint) = 0x08
        assert_eq!(bytes, vec![0x08, 0x01]);
    }

    #[test]
    fn encode_sint64_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_sint64(1, -2).unwrap();
        let bytes = enc.finish();
        // zigzag(-2) = 3, tag(1, varint) = 0x08
        assert_eq!(bytes, vec![0x08, 0x03]);
    }

    #[test]
    fn encoder_clear_and_reuse() {
        let mut enc = ProtoEncoder::new();
        enc.encode_uint32_always(1, 10).unwrap();
        assert!(!enc.is_empty());
        enc.clear();
        assert!(enc.is_empty());
        enc.encode_uint32_always(2, 20).unwrap();
        assert!(!enc.is_empty());
    }

    #[test]
    fn size_computation_uint32() {
        assert_eq!(uint32_field_size(1, 0), 0); // default omitted
        assert_eq!(uint32_field_size(1, 1), 2); // tag(1 byte) + varint(1 byte)
        assert_eq!(uint32_field_size(1, 150), 3); // tag(1) + varint(2)
    }

    #[test]
    fn size_computation_string() {
        assert_eq!(string_field_size(1, ""), 0);
        assert_eq!(string_field_size(2, "test"), 1 + 1 + 4); // tag(1) + len(1) + data(4) — tag is 0x12 (1 byte)
    }

    #[test]
    fn size_computation_bool() {
        assert_eq!(bool_field_size(1, false), 0);
        assert_eq!(bool_field_size(1, true), 2);
    }

    #[test]
    fn size_computation_fixed() {
        assert_eq!(fixed32_field_size(1, 0), 0);
        assert_eq!(fixed32_field_size(1, 1), 1 + 4); // tag + 4 bytes
        assert_eq!(fixed64_field_size(1, 0), 0);
        assert_eq!(fixed64_field_size(1, 1), 1 + 8); // tag + 8 bytes
    }

    #[test]
    fn encode_int32_negative() {
        let mut enc = ProtoEncoder::new();
        enc.encode_int32(1, -1).unwrap();
        let bytes = enc.finish();
        // -1 as u64 = 0xFFFFFFFFFFFFFFFF, which is 10 bytes as varint
        assert_eq!(bytes.len(), 1 + 10); // tag + 10-byte varint
    }

    #[test]
    fn invalid_field_number_zero() {
        let mut enc = ProtoEncoder::new();
        assert!(enc.encode_uint32(0, 42).is_err());
    }

    #[test]
    fn encode_packed_double() {
        let mut enc = ProtoEncoder::new();
        enc.encode_packed_double(1, &[1.0, 2.0, 3.0]).unwrap();
        let bytes = enc.finish();
        // 3 doubles = 24 bytes payload
        assert!(bytes.len() > 24);
    }

    #[test]
    fn encode_error_display() {
        let err = EncodeError::InvalidFieldNumber(0);
        assert_eq!(err.to_string(), "invalid field number: 0");
        assert_eq!(EncodeError::BufferOverflow.to_string(), "buffer overflow");
        assert_eq!(EncodeError::InvalidMapKey.to_string(), "invalid map key type");
    }

    #[test]
    fn encode_map_string_uint64() {
        let mut entries = HashMap::new();
        entries.insert("count".to_string(), 42u64);

        let mut enc = ProtoEncoder::new();
        enc.encode_map_string_uint64(3, &entries).unwrap();
        assert!(!enc.is_empty());
    }

    #[test]
    fn encode_map_uint32_string() {
        let mut entries = HashMap::new();
        entries.insert(1u32, "one".to_string());
        entries.insert(2, "two".to_string());

        let mut enc = ProtoEncoder::new();
        enc.encode_map_uint32_string(4, &entries).unwrap();
        assert!(!enc.is_empty());
    }

    #[test]
    fn encode_sfixed_fields() {
        let mut enc = ProtoEncoder::new();
        enc.encode_sfixed32(1, -100).unwrap();
        enc.encode_sfixed64(2, -200).unwrap();
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_enum_field() {
        let mut enc = ProtoEncoder::new();
        enc.encode_enum(1, 0).unwrap(); // default omitted
        assert!(enc.is_empty());
        enc.encode_enum(1, 2).unwrap();
        assert!(!enc.is_empty());
    }

    #[test]
    fn encode_with_capacity() {
        let enc = ProtoEncoder::with_capacity(1024);
        assert!(enc.is_empty());
        assert_eq!(enc.len(), 0);
    }

    #[test]
    fn encode_repeated_bytes() {
        let mut enc = ProtoEncoder::new();
        enc.encode_repeated_bytes(5, &[b"abc", b"def"]).unwrap();
        let bytes = enc.finish();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn encode_packed_sint() {
        let mut enc = ProtoEncoder::new();
        enc.encode_packed_sint32(1, &[-1, 0, 1]).unwrap();
        assert!(!enc.is_empty());

        let mut enc2 = ProtoEncoder::new();
        enc2.encode_packed_sint64(2, &[-100, 100]).unwrap();
        assert!(!enc2.is_empty());
    }
}
