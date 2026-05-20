//! Protocol Buffers wire format — varint, zigzag, length-delimited fields.
//!
//! Pure-Rust protobuf wire-format codec.  No `.proto` compiler needed — builds
//! and parses raw wire bytes using a message builder and field-tag API.

use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Protobuf codec errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoError {
    /// Buffer exhausted while reading.
    UnexpectedEof,
    /// Varint exceeds 10 bytes.
    VarintOverflow,
    /// Unknown wire type.
    UnknownWireType(u8),
    /// Field number zero is reserved.
    InvalidFieldNumber,
    /// Nested depth exceeded.
    NestingTooDeep,
}

impl std::fmt::Display for ProtoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::VarintOverflow => write!(f, "varint exceeds 10 bytes"),
            Self::UnknownWireType(w) => write!(f, "unknown wire type {w}"),
            Self::InvalidFieldNumber => write!(f, "field number must be >= 1"),
            Self::NestingTooDeep => write!(f, "nesting exceeds maximum depth"),
        }
    }
}

impl std::error::Error for ProtoError {}

// ── Wire types ──────────────────────────────────────────────────

/// Protobuf wire types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    Fixed32 = 5,
}

impl WireType {
    fn from_u8(v: u8) -> Result<Self, ProtoError> {
        match v {
            0 => Ok(Self::Varint),
            1 => Ok(Self::Fixed64),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::Fixed32),
            other => Err(ProtoError::UnknownWireType(other)),
        }
    }
}

// ── Field tag ───────────────────────────────────────────────────

/// A parsed field tag (field_number, wire_type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldTag {
    pub field_number: u32,
    pub wire_type: WireType,
}

impl FieldTag {
    pub fn new(field_number: u32, wire_type: WireType) -> Result<Self, ProtoError> {
        if field_number == 0 {
            return Err(ProtoError::InvalidFieldNumber);
        }
        Ok(Self { field_number, wire_type })
    }

    fn encode_u32(&self) -> u32 {
        (self.field_number << 3) | (self.wire_type as u32)
    }
}

// ── Varint encoding/decoding ────────────────────────────────────

/// Encode a `u64` as a protobuf varint, returning the bytes.
pub fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
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
    buf
}

/// Decode a varint from `data` starting at `offset`.  Returns `(value, new_offset)`.
pub fn decode_varint(data: &[u8], offset: usize) -> Result<(u64, usize), ProtoError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut pos = offset;
    loop {
        if pos >= data.len() {
            return Err(ProtoError::UnexpectedEof);
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 70 {
            return Err(ProtoError::VarintOverflow);
        }
    }
}

// ── Zigzag encoding (signed integers) ───────────────────────────

/// Zigzag-encode a signed 32-bit integer.
pub fn zigzag_encode_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Zigzag-decode back to signed 32-bit.
pub fn zigzag_decode_i32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Zigzag-encode a signed 64-bit integer.
pub fn zigzag_encode_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Zigzag-decode back to signed 64-bit.
pub fn zigzag_decode_i64(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

// ── Field value ─────────────────────────────────────────────────

/// A decoded protobuf field value.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Varint(u64),
    Fixed32([u8; 4]),
    Fixed64([u8; 8]),
    LengthDelimited(Vec<u8>),
}

// ── Message builder ─────────────────────────────────────────────

/// Builds a protobuf message by appending fields.
#[derive(Debug, Clone, Default)]
pub struct MessageBuilder {
    buf: Vec<u8>,
}

impl MessageBuilder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Write a tag to the internal buffer.
    fn write_tag(&mut self, tag: FieldTag) {
        self.buf.extend_from_slice(&encode_varint(tag.encode_u32() as u64));
    }

    /// Add a varint field.
    pub fn add_varint(&mut self, field_number: u32, value: u64) -> Result<(), ProtoError> {
        let tag = FieldTag::new(field_number, WireType::Varint)?;
        self.write_tag(tag);
        self.buf.extend_from_slice(&encode_varint(value));
        Ok(())
    }

    /// Add a signed varint (zigzag) field.
    pub fn add_sint64(&mut self, field_number: u32, value: i64) -> Result<(), ProtoError> {
        self.add_varint(field_number, zigzag_encode_i64(value))
    }

    /// Add a sint32 field.
    pub fn add_sint32(&mut self, field_number: u32, value: i32) -> Result<(), ProtoError> {
        self.add_varint(field_number, zigzag_encode_i32(value) as u64)
    }

    /// Add a fixed32 field.
    pub fn add_fixed32(&mut self, field_number: u32, value: u32) -> Result<(), ProtoError> {
        let tag = FieldTag::new(field_number, WireType::Fixed32)?;
        self.write_tag(tag);
        self.buf.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Add a fixed64 field.
    pub fn add_fixed64(&mut self, field_number: u32, value: u64) -> Result<(), ProtoError> {
        let tag = FieldTag::new(field_number, WireType::Fixed64)?;
        self.write_tag(tag);
        self.buf.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    /// Add a length-delimited field (bytes).
    pub fn add_bytes(&mut self, field_number: u32, data: &[u8]) -> Result<(), ProtoError> {
        let tag = FieldTag::new(field_number, WireType::LengthDelimited)?;
        self.write_tag(tag);
        self.buf.extend_from_slice(&encode_varint(data.len() as u64));
        self.buf.extend_from_slice(data);
        Ok(())
    }

    /// Add a string field.
    pub fn add_string(&mut self, field_number: u32, s: &str) -> Result<(), ProtoError> {
        self.add_bytes(field_number, s.as_bytes())
    }

    /// Add a nested message field.
    pub fn add_message(&mut self, field_number: u32, inner: &MessageBuilder) -> Result<(), ProtoError> {
        self.add_bytes(field_number, &inner.buf)
    }

    /// Add a repeated varint field (packed encoding).
    pub fn add_packed_varints(&mut self, field_number: u32, values: &[u64]) -> Result<(), ProtoError> {
        let mut packed = Vec::new();
        for v in values {
            packed.extend_from_slice(&encode_varint(*v));
        }
        self.add_bytes(field_number, &packed)
    }

    /// Consume the builder and return the encoded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    /// Current byte length.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

// ── Message decoder ─────────────────────────────────────────────

/// Decoded fields from a protobuf message, keyed by field number.
/// Repeated fields accumulate into a `Vec`.
#[derive(Debug, Clone, Default)]
pub struct DecodedMessage {
    pub fields: HashMap<u32, Vec<FieldValue>>,
}

impl DecodedMessage {
    /// Decode a protobuf message from bytes.
    pub fn decode(data: &[u8]) -> Result<Self, ProtoError> {
        Self::decode_depth(data, 0)
    }

    fn decode_depth(data: &[u8], depth: usize) -> Result<Self, ProtoError> {
        if depth > 64 {
            return Err(ProtoError::NestingTooDeep);
        }
        let mut msg = DecodedMessage::default();
        let mut offset = 0;
        while offset < data.len() {
            let (tag_val, new_offset) = decode_varint(data, offset)?;
            offset = new_offset;
            let wire_type_raw = (tag_val & 0x07) as u8;
            let field_number = (tag_val >> 3) as u32;
            if field_number == 0 {
                return Err(ProtoError::InvalidFieldNumber);
            }
            let wire_type = WireType::from_u8(wire_type_raw)?;
            let value = match wire_type {
                WireType::Varint => {
                    let (v, o) = decode_varint(data, offset)?;
                    offset = o;
                    FieldValue::Varint(v)
                }
                WireType::Fixed64 => {
                    if offset + 8 > data.len() {
                        return Err(ProtoError::UnexpectedEof);
                    }
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data[offset..offset + 8]);
                    offset += 8;
                    FieldValue::Fixed64(bytes)
                }
                WireType::Fixed32 => {
                    if offset + 4 > data.len() {
                        return Err(ProtoError::UnexpectedEof);
                    }
                    let mut bytes = [0u8; 4];
                    bytes.copy_from_slice(&data[offset..offset + 4]);
                    offset += 4;
                    FieldValue::Fixed32(bytes)
                }
                WireType::LengthDelimited => {
                    let (len, o) = decode_varint(data, offset)?;
                    offset = o;
                    let len = len as usize;
                    if offset + len > data.len() {
                        return Err(ProtoError::UnexpectedEof);
                    }
                    let bytes = data[offset..offset + len].to_vec();
                    offset += len;
                    FieldValue::LengthDelimited(bytes)
                }
            };
            msg.fields.entry(field_number).or_default().push(value);
        }
        Ok(msg)
    }

    /// Get the first varint value for a field.
    pub fn get_varint(&self, field_number: u32) -> Option<u64> {
        self.fields.get(&field_number).and_then(|vals| {
            vals.iter().find_map(|v| match v {
                FieldValue::Varint(n) => Some(*n),
                _ => None,
            })
        })
    }

    /// Get the first length-delimited value as a string.
    pub fn get_string(&self, field_number: u32) -> Option<String> {
        self.fields.get(&field_number).and_then(|vals| {
            vals.iter().find_map(|v| match v {
                FieldValue::LengthDelimited(b) => String::from_utf8(b.clone()).ok(),
                _ => None,
            })
        })
    }

    /// Get the first length-delimited value as bytes.
    pub fn get_bytes(&self, field_number: u32) -> Option<&[u8]> {
        self.fields.get(&field_number).and_then(|vals| {
            vals.iter().find_map(|v| match v {
                FieldValue::LengthDelimited(b) => Some(b.as_slice()),
                _ => None,
            })
        })
    }

    /// Get the first fixed32 value.
    pub fn get_fixed32(&self, field_number: u32) -> Option<u32> {
        self.fields.get(&field_number).and_then(|vals| {
            vals.iter().find_map(|v| match v {
                FieldValue::Fixed32(b) => Some(u32::from_le_bytes(*b)),
                _ => None,
            })
        })
    }

    /// Get the first fixed64 value.
    pub fn get_fixed64(&self, field_number: u32) -> Option<u64> {
        self.fields.get(&field_number).and_then(|vals| {
            vals.iter().find_map(|v| match v {
                FieldValue::Fixed64(b) => Some(u64::from_le_bytes(*b)),
                _ => None,
            })
        })
    }

    /// Decode a nested message from a length-delimited field.
    pub fn get_message(&self, field_number: u32) -> Result<Option<DecodedMessage>, ProtoError> {
        match self.get_bytes(field_number) {
            Some(b) => Ok(Some(DecodedMessage::decode(b)?)),
            None => Ok(None),
        }
    }

    /// Decode packed repeated varints.
    pub fn get_packed_varints(&self, field_number: u32) -> Result<Vec<u64>, ProtoError> {
        let mut result = Vec::new();
        if let Some(vals) = self.fields.get(&field_number) {
            for v in vals {
                if let FieldValue::LengthDelimited(bytes) = v {
                    let mut offset = 0;
                    while offset < bytes.len() {
                        let (val, new_offset) = decode_varint(bytes, offset)?;
                        result.push(val);
                        offset = new_offset;
                    }
                }
            }
        }
        Ok(result)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip_small() {
        for v in [0u64, 1, 127, 128, 300, 16383, 16384] {
            let encoded = encode_varint(v);
            let (decoded, _) = decode_varint(&encoded, 0).unwrap();
            assert_eq!(decoded, v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn varint_roundtrip_large() {
        let v = u64::MAX;
        let encoded = encode_varint(v);
        assert_eq!(encoded.len(), 10);
        let (decoded, _) = decode_varint(&encoded, 0).unwrap();
        assert_eq!(decoded, v);
    }

    #[test]
    fn zigzag_i32_roundtrip() {
        for n in [0i32, -1, 1, -2, 2, i32::MIN, i32::MAX] {
            assert_eq!(zigzag_decode_i32(zigzag_encode_i32(n)), n);
        }
    }

    #[test]
    fn zigzag_i64_roundtrip() {
        for n in [0i64, -1, 1, -2, 2, i64::MIN, i64::MAX] {
            assert_eq!(zigzag_decode_i64(zigzag_encode_i64(n)), n);
        }
    }

    #[test]
    fn message_builder_varint_field() {
        let mut msg = MessageBuilder::new();
        msg.add_varint(1, 150).unwrap();
        let data = msg.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        assert_eq!(decoded.get_varint(1), Some(150));
    }

    #[test]
    fn message_builder_string_field() {
        let mut msg = MessageBuilder::new();
        msg.add_string(2, "testing").unwrap();
        let data = msg.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        assert_eq!(decoded.get_string(2).as_deref(), Some("testing"));
    }

    #[test]
    fn message_fixed32_fixed64() {
        let mut msg = MessageBuilder::new();
        msg.add_fixed32(1, 0xDEAD_BEEF).unwrap();
        msg.add_fixed64(2, 0xCAFE_BABE_DEAD_BEEFu64).unwrap();
        let data = msg.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        assert_eq!(decoded.get_fixed32(1), Some(0xDEAD_BEEF));
        assert_eq!(decoded.get_fixed64(2), Some(0xCAFE_BABE_DEAD_BEEFu64));
    }

    #[test]
    fn nested_message() {
        let mut inner = MessageBuilder::new();
        inner.add_varint(1, 42).unwrap();
        inner.add_string(2, "inner").unwrap();

        let mut outer = MessageBuilder::new();
        outer.add_string(1, "outer").unwrap();
        outer.add_message(2, &inner).unwrap();

        let data = outer.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        assert_eq!(decoded.get_string(1).as_deref(), Some("outer"));
        let nested = decoded.get_message(2).unwrap().unwrap();
        assert_eq!(nested.get_varint(1), Some(42));
        assert_eq!(nested.get_string(2).as_deref(), Some("inner"));
    }

    #[test]
    fn packed_repeated_varints() {
        let mut msg = MessageBuilder::new();
        msg.add_packed_varints(4, &[3, 270, 86942]).unwrap();
        let data = msg.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        let vals = decoded.get_packed_varints(4).unwrap();
        assert_eq!(vals, vec![3, 270, 86942]);
    }

    #[test]
    fn field_number_zero_rejected() {
        let mut msg = MessageBuilder::new();
        assert!(msg.add_varint(0, 1).is_err());
    }

    #[test]
    fn multiple_fields_same_number() {
        let mut msg = MessageBuilder::new();
        msg.add_varint(1, 10).unwrap();
        msg.add_varint(1, 20).unwrap();
        let data = msg.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        let vals = decoded.fields.get(&1).unwrap();
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn sint32_roundtrip() {
        let mut msg = MessageBuilder::new();
        msg.add_sint32(1, -42).unwrap();
        let data = msg.finish();
        let decoded = DecodedMessage::decode(&data).unwrap();
        let raw = decoded.get_varint(1).unwrap();
        assert_eq!(zigzag_decode_i32(raw as u32), -42);
    }

    #[test]
    fn empty_message() {
        let msg = MessageBuilder::new();
        assert!(msg.is_empty());
        let data = msg.finish();
        assert!(data.is_empty());
        let decoded = DecodedMessage::decode(&data).unwrap();
        assert!(decoded.fields.is_empty());
    }
}
