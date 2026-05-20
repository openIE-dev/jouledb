//! CBOR (RFC 7049) codec — Concise Binary Object Representation.
//!
//! Supports all major types, indefinite-length encoding, canonical CBOR
//! (deterministic map key sorting), and diagnostic notation output.

use std::collections::BTreeMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// CBOR codec errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CborError {
    UnexpectedEof,
    InvalidAdditionalInfo(u8),
    InvalidMajorType(u8),
    InvalidUtf8,
    NestingTooDeep,
    UnexpectedBreak,
    IndefiniteLengthNotSupported,
}

impl fmt::Display for CborError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::InvalidAdditionalInfo(v) => write!(f, "invalid additional info: {v}"),
            Self::InvalidMajorType(v) => write!(f, "invalid major type: {v}"),
            Self::InvalidUtf8 => write!(f, "text string is not valid UTF-8"),
            Self::NestingTooDeep => write!(f, "nesting too deep"),
            Self::UnexpectedBreak => write!(f, "unexpected break code"),
            Self::IndefiniteLengthNotSupported => write!(f, "indefinite length not supported here"),
        }
    }
}

impl std::error::Error for CborError {}

// ── Major types ─────────────────────────────────────────────────

/// CBOR major types (0–7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MajorType {
    UnsignedInt = 0,
    NegativeInt = 1,
    ByteString = 2,
    TextString = 3,
    Array = 4,
    Map = 5,
    Tag = 6,
    SimpleOrFloat = 7,
}

// ── Value ───────────────────────────────────────────────────────

/// A CBOR value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    UnsignedInt(u64),
    NegativeInt(i64),
    ByteString(Vec<u8>),
    TextString(String),
    Array(Vec<Value>),
    Map(Vec<(Value, Value)>),
    Tag(u64, Box<Value>),
    Bool(bool),
    Null,
    Undefined,
    Float16(f32),
    Float32(f32),
    Float64(f64),
    Simple(u8),
}

// ── Encoder ─────────────────────────────────────────────────────

/// CBOR encoder.
#[derive(Debug, Clone, Default)]
pub struct Encoder {
    buf: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_type_value(&mut self, major: u8, value: u64) {
        let mt = major << 5;
        if value <= 23 {
            self.buf.push(mt | value as u8);
        } else if value <= u8::MAX as u64 {
            self.buf.push(mt | 24);
            self.buf.push(value as u8);
        } else if value <= u16::MAX as u64 {
            self.buf.push(mt | 25);
            self.buf.extend_from_slice(&(value as u16).to_be_bytes());
        } else if value <= u32::MAX as u64 {
            self.buf.push(mt | 26);
            self.buf.extend_from_slice(&(value as u32).to_be_bytes());
        } else {
            self.buf.push(mt | 27);
            self.buf.extend_from_slice(&value.to_be_bytes());
        }
    }

    /// Encode a value.
    pub fn encode(&mut self, val: &Value) {
        match val {
            Value::UnsignedInt(n) => self.write_type_value(0, *n),
            Value::NegativeInt(n) => {
                // CBOR: major type 1, value = -1 - n
                let v = (-1i128 - (*n as i128)) as u64;
                self.write_type_value(1, v);
            }
            Value::ByteString(b) => {
                self.write_type_value(2, b.len() as u64);
                self.buf.extend_from_slice(b);
            }
            Value::TextString(s) => {
                self.write_type_value(3, s.len() as u64);
                self.buf.extend_from_slice(s.as_bytes());
            }
            Value::Array(arr) => {
                self.write_type_value(4, arr.len() as u64);
                for v in arr {
                    self.encode(v);
                }
            }
            Value::Map(entries) => {
                self.write_type_value(5, entries.len() as u64);
                for (k, v) in entries {
                    self.encode(k);
                    self.encode(v);
                }
            }
            Value::Tag(tag, inner) => {
                self.write_type_value(6, *tag);
                self.encode(inner);
            }
            Value::Bool(true) => self.buf.push(0xf5),
            Value::Bool(false) => self.buf.push(0xf4),
            Value::Null => self.buf.push(0xf6),
            Value::Undefined => self.buf.push(0xf7),
            Value::Float16(_v) => {
                // Encode as float32 for simplicity
                self.buf.push(0xfa);
                self.buf.extend_from_slice(&_v.to_be_bytes());
            }
            Value::Float32(v) => {
                self.buf.push(0xfa);
                self.buf.extend_from_slice(&v.to_be_bytes());
            }
            Value::Float64(v) => {
                self.buf.push(0xfb);
                self.buf.extend_from_slice(&v.to_be_bytes());
            }
            Value::Simple(s) => {
                if *s <= 23 {
                    self.buf.push(0xe0 | s);
                } else {
                    self.buf.push(0xf8);
                    self.buf.push(*s);
                }
            }
        }
    }

    /// Encode with indefinite-length byte string.
    pub fn encode_indefinite_bytes(&mut self, chunks: &[&[u8]]) {
        self.buf.push(0x5f); // major 2, additional 31
        for chunk in chunks {
            self.write_type_value(2, chunk.len() as u64);
            self.buf.extend_from_slice(chunk);
        }
        self.buf.push(0xff); // break
    }

    /// Encode with indefinite-length text string.
    pub fn encode_indefinite_text(&mut self, chunks: &[&str]) {
        self.buf.push(0x7f); // major 3, additional 31
        for chunk in chunks {
            self.write_type_value(3, chunk.len() as u64);
            self.buf.extend_from_slice(chunk.as_bytes());
        }
        self.buf.push(0xff);
    }

    /// Encode with indefinite-length array.
    pub fn encode_indefinite_array(&mut self, items: &[Value]) {
        self.buf.push(0x9f); // major 4, additional 31
        for item in items {
            self.encode(item);
        }
        self.buf.push(0xff);
    }

    /// Consume and return bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

// ── Decoder ─────────────────────────────────────────────────────

/// CBOR decoder.
pub struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, depth: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, CborError> {
        if self.pos >= self.data.len() {
            return Err(CborError::UnexpectedEof);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], CborError> {
        if self.pos + n > self.data.len() {
            return Err(CborError::UnexpectedEof);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_argument(&mut self, additional: u8) -> Result<u64, CborError> {
        match additional {
            0..=23 => Ok(additional as u64),
            24 => Ok(self.read_u8()? as u64),
            25 => {
                let b = self.read_bytes(2)?;
                Ok(u16::from_be_bytes([b[0], b[1]]) as u64)
            }
            26 => {
                let b = self.read_bytes(4)?;
                Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as u64)
            }
            27 => {
                let b = self.read_bytes(8)?;
                Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
            }
            _ => Err(CborError::InvalidAdditionalInfo(additional)),
        }
    }

    /// Decode the next CBOR value.
    pub fn decode(&mut self) -> Result<Value, CborError> {
        if self.depth > 128 {
            return Err(CborError::NestingTooDeep);
        }
        let initial = self.read_u8()?;
        let major = initial >> 5;
        let additional = initial & 0x1f;

        match major {
            0 => {
                let v = self.read_argument(additional)?;
                Ok(Value::UnsignedInt(v))
            }
            1 => {
                let v = self.read_argument(additional)?;
                Ok(Value::NegativeInt(-1 - v as i64))
            }
            2 => {
                if additional == 31 {
                    // indefinite-length byte string
                    let mut result = Vec::new();
                    loop {
                        if self.pos >= self.data.len() {
                            return Err(CborError::UnexpectedEof);
                        }
                        if self.data[self.pos] == 0xff {
                            self.pos += 1;
                            break;
                        }
                        let chunk = self.decode()?;
                        if let Value::ByteString(b) = chunk {
                            result.extend_from_slice(&b);
                        } else {
                            return Err(CborError::InvalidMajorType(2));
                        }
                    }
                    Ok(Value::ByteString(result))
                } else {
                    let len = self.read_argument(additional)? as usize;
                    let bytes = self.read_bytes(len)?;
                    Ok(Value::ByteString(bytes.to_vec()))
                }
            }
            3 => {
                if additional == 31 {
                    let mut result = String::new();
                    loop {
                        if self.pos >= self.data.len() {
                            return Err(CborError::UnexpectedEof);
                        }
                        if self.data[self.pos] == 0xff {
                            self.pos += 1;
                            break;
                        }
                        let chunk = self.decode()?;
                        if let Value::TextString(s) = chunk {
                            result.push_str(&s);
                        } else {
                            return Err(CborError::InvalidMajorType(3));
                        }
                    }
                    Ok(Value::TextString(result))
                } else {
                    let len = self.read_argument(additional)? as usize;
                    let bytes = self.read_bytes(len)?;
                    let s = std::str::from_utf8(bytes).map_err(|_| CborError::InvalidUtf8)?;
                    Ok(Value::TextString(s.to_string()))
                }
            }
            4 => {
                self.depth += 1;
                let arr = if additional == 31 {
                    let mut items = Vec::new();
                    loop {
                        if self.pos >= self.data.len() {
                            return Err(CborError::UnexpectedEof);
                        }
                        if self.data[self.pos] == 0xff {
                            self.pos += 1;
                            break;
                        }
                        items.push(self.decode()?);
                    }
                    items
                } else {
                    let len = self.read_argument(additional)? as usize;
                    let mut items = Vec::with_capacity(len);
                    for _ in 0..len {
                        items.push(self.decode()?);
                    }
                    items
                };
                self.depth -= 1;
                Ok(Value::Array(arr))
            }
            5 => {
                self.depth += 1;
                let entries = if additional == 31 {
                    let mut entries = Vec::new();
                    loop {
                        if self.pos >= self.data.len() {
                            return Err(CborError::UnexpectedEof);
                        }
                        if self.data[self.pos] == 0xff {
                            self.pos += 1;
                            break;
                        }
                        let k = self.decode()?;
                        let v = self.decode()?;
                        entries.push((k, v));
                    }
                    entries
                } else {
                    let len = self.read_argument(additional)? as usize;
                    let mut entries = Vec::with_capacity(len);
                    for _ in 0..len {
                        let k = self.decode()?;
                        let v = self.decode()?;
                        entries.push((k, v));
                    }
                    entries
                };
                self.depth -= 1;
                Ok(Value::Map(entries))
            }
            6 => {
                let tag = self.read_argument(additional)?;
                let inner = self.decode()?;
                Ok(Value::Tag(tag, Box::new(inner)))
            }
            7 => {
                match additional {
                    20 => Ok(Value::Bool(false)),
                    21 => Ok(Value::Bool(true)),
                    22 => Ok(Value::Null),
                    23 => Ok(Value::Undefined),
                    24 => {
                        let v = self.read_u8()?;
                        Ok(Value::Simple(v))
                    }
                    25 => {
                        // half-precision float — decode to f32
                        let b = self.read_bytes(2)?;
                        let half = u16::from_be_bytes([b[0], b[1]]);
                        Ok(Value::Float16(half_to_f32(half)))
                    }
                    26 => {
                        let b = self.read_bytes(4)?;
                        Ok(Value::Float32(f32::from_be_bytes([b[0], b[1], b[2], b[3]])))
                    }
                    27 => {
                        let b = self.read_bytes(8)?;
                        Ok(Value::Float64(f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])))
                    }
                    31 => Err(CborError::UnexpectedBreak),
                    v if v <= 23 => Ok(Value::Simple(v)),
                    other => Err(CborError::InvalidAdditionalInfo(other)),
                }
            }
            other => Err(CborError::InvalidMajorType(other)),
        }
    }

    pub fn position(&self) -> usize {
        self.pos
    }
}

/// Decode IEEE 754 half-precision float to f32.
fn half_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exp = ((half >> 10) & 0x1f) as u32;
    let mant = (half & 0x3ff) as u32;

    if exp == 0 {
        // subnormal or zero
        let val = (sign << 31) | 0;
        let f = f32::from_bits(val);
        if mant == 0 {
            f
        } else {
            let scale = 2.0f32.powi(-24);
            let v = mant as f32 * scale;
            if sign == 1 { -v } else { v }
        }
    } else if exp == 31 {
        // inf or nan
        if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let new_exp = exp as i32 - 15 + 127;
        let bits = (sign << 31) | ((new_exp as u32) << 23) | (mant << 13);
        f32::from_bits(bits)
    }
}

// ── Convenience ─────────────────────────────────────────────────

/// Encode a value to CBOR bytes.
pub fn encode(val: &Value) -> Vec<u8> {
    let mut enc = Encoder::new();
    enc.encode(val);
    enc.finish()
}

/// Decode CBOR bytes to a value.
pub fn decode(data: &[u8]) -> Result<Value, CborError> {
    let mut dec = Decoder::new(data);
    dec.decode()
}

// ── Canonical CBOR ──────────────────────────────────────────────

/// Encode a value in canonical CBOR (RFC 7049 Section 3.9):
/// map keys sorted by their encoded bytes (shortest first, then lexicographic).
pub fn encode_canonical(val: &Value) -> Vec<u8> {
    let mut enc = Encoder::new();
    encode_canonical_inner(&mut enc, val);
    enc.finish()
}

fn encode_canonical_inner(enc: &mut Encoder, val: &Value) {
    match val {
        Value::Map(entries) => {
            let mut sorted: Vec<(Vec<u8>, &Value)> = entries
                .iter()
                .map(|(k, v)| {
                    let kb = encode_canonical(k);
                    (kb, v)
                })
                .collect();
            // Sort by length first, then lexicographic
            sorted.sort_by(|a, b| {
                a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0))
            });
            enc.write_type_value(5, sorted.len() as u64);
            for (kb, v) in &sorted {
                enc.buf.extend_from_slice(kb);
                encode_canonical_inner(enc, v);
            }
        }
        Value::Array(arr) => {
            enc.write_type_value(4, arr.len() as u64);
            for v in arr {
                encode_canonical_inner(enc, v);
            }
        }
        other => enc.encode(other),
    }
}

// ── Diagnostic notation ─────────────────────────────────────────

/// Produce diagnostic notation string (RFC 7049 Section 6).
pub fn diagnostic(val: &Value) -> String {
    match val {
        Value::UnsignedInt(n) => n.to_string(),
        Value::NegativeInt(n) => n.to_string(),
        Value::ByteString(b) => {
            let hex: Vec<String> = b.iter().map(|byte| format!("{byte:02x}")).collect();
            format!("h'{}'", hex.join(""))
        }
        Value::TextString(s) => format!("\"{s}\""),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(diagnostic).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(entries) => {
            let pairs: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}: {}", diagnostic(k), diagnostic(v)))
                .collect();
            format!("{{{}}}", pairs.join(", "))
        }
        Value::Tag(tag, inner) => format!("{tag}({})", diagnostic(inner)),
        Value::Bool(true) => "true".into(),
        Value::Bool(false) => "false".into(),
        Value::Null => "null".into(),
        Value::Undefined => "undefined".into(),
        Value::Float16(v) | Value::Float32(v) => format!("{v}"),
        Value::Float64(v) => format!("{v}"),
        Value::Simple(s) => format!("simple({s})"),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(val: &Value) -> Value {
        let bytes = encode(val);
        decode(&bytes).unwrap()
    }

    #[test]
    fn unsigned_int_roundtrip() {
        for n in [0u64, 1, 23, 24, 255, 256, 65535, 65536, u32::MAX as u64, u64::MAX] {
            assert_eq!(roundtrip(&Value::UnsignedInt(n)), Value::UnsignedInt(n));
        }
    }

    #[test]
    fn unsigned_int_compact() {
        // 0..23 should be single byte
        assert_eq!(encode(&Value::UnsignedInt(0)).len(), 1);
        assert_eq!(encode(&Value::UnsignedInt(23)).len(), 1);
        // 24 should be 2 bytes
        assert_eq!(encode(&Value::UnsignedInt(24)).len(), 2);
    }

    #[test]
    fn negative_int_roundtrip() {
        for n in [-1i64, -10, -100, -1000, -1_000_000] {
            assert_eq!(roundtrip(&Value::NegativeInt(n)), Value::NegativeInt(n));
        }
    }

    #[test]
    fn byte_string_roundtrip() {
        let val = Value::ByteString(vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn text_string_roundtrip() {
        let val = Value::TextString("hello world".into());
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn array_roundtrip() {
        let val = Value::Array(vec![
            Value::UnsignedInt(1),
            Value::TextString("two".into()),
            Value::Bool(true),
        ]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn map_roundtrip() {
        let val = Value::Map(vec![
            (Value::TextString("key".into()), Value::UnsignedInt(42)),
            (Value::UnsignedInt(1), Value::Bool(false)),
        ]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn tag_roundtrip() {
        let val = Value::Tag(1, Box::new(Value::UnsignedInt(1363896240)));
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn simple_values() {
        assert_eq!(roundtrip(&Value::Bool(true)), Value::Bool(true));
        assert_eq!(roundtrip(&Value::Bool(false)), Value::Bool(false));
        assert_eq!(roundtrip(&Value::Null), Value::Null);
        assert_eq!(roundtrip(&Value::Undefined), Value::Undefined);
    }

    #[test]
    fn float_roundtrip() {
        let val = Value::Float64(3.4028234663852886e+38);
        assert_eq!(roundtrip(&val), val);
        let val32 = Value::Float32(100000.0);
        assert_eq!(roundtrip(&val32), val32);
    }

    #[test]
    fn indefinite_bytes() {
        let mut enc = Encoder::new();
        enc.encode_indefinite_bytes(&[&[0x01, 0x02], &[0x03, 0x04]]);
        let bytes = enc.finish();
        let val = decode(&bytes).unwrap();
        assert_eq!(val, Value::ByteString(vec![0x01, 0x02, 0x03, 0x04]));
    }

    #[test]
    fn indefinite_text() {
        let mut enc = Encoder::new();
        enc.encode_indefinite_text(&["strea", "ming"]);
        let bytes = enc.finish();
        let val = decode(&bytes).unwrap();
        assert_eq!(val, Value::TextString("streaming".into()));
    }

    #[test]
    fn indefinite_array() {
        let mut enc = Encoder::new();
        enc.encode_indefinite_array(&[Value::UnsignedInt(1), Value::UnsignedInt(2)]);
        let bytes = enc.finish();
        let val = decode(&bytes).unwrap();
        assert_eq!(val, Value::Array(vec![Value::UnsignedInt(1), Value::UnsignedInt(2)]));
    }

    #[test]
    fn canonical_sorting() {
        let val = Value::Map(vec![
            (Value::TextString("bb".into()), Value::UnsignedInt(2)),
            (Value::TextString("a".into()), Value::UnsignedInt(1)),
            (Value::TextString("ccc".into()), Value::UnsignedInt(3)),
        ]);
        let canonical = encode_canonical(&val);
        let decoded = decode(&canonical).unwrap();
        if let Value::Map(entries) = decoded {
            // shortest key first
            assert_eq!(entries[0].0, Value::TextString("a".into()));
            assert_eq!(entries[1].0, Value::TextString("bb".into()));
            assert_eq!(entries[2].0, Value::TextString("ccc".into()));
        } else {
            panic!("expected map");
        }
    }

    #[test]
    fn diagnostic_notation() {
        let val = Value::Map(vec![
            (Value::TextString("a".into()), Value::UnsignedInt(1)),
            (Value::TextString("b".into()), Value::Array(vec![Value::UnsignedInt(2), Value::UnsignedInt(3)])),
        ]);
        let diag = diagnostic(&val);
        assert!(diag.contains("\"a\": 1"));
        assert!(diag.contains("\"b\": [2, 3]"));
    }

    #[test]
    fn nested_structures() {
        let val = Value::Array(vec![
            Value::Map(vec![
                (Value::TextString("inner".into()), Value::Array(vec![
                    Value::UnsignedInt(1),
                    Value::NegativeInt(-2),
                ])),
            ]),
        ]);
        assert_eq!(roundtrip(&val), val);
    }
}
