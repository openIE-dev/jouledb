// SPDX-License-Identifier: MIT
//! MessagePack codec -- compact binary serialization.
//!
//! Full spec: nil, bool, int (positive/negative fixint, 8/16/32/64),
//! float 32/64, str (fix/8/16/32), bin (8/16/32), array (fix/16/32),
//! map (fix/16/32), ext (fix1/2/4/8/16, ext8/16/32).
//! Streaming encoder/decoder with serde_json conversion.

use serde_json::{self, Map};

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MsgPackError {
    UnexpectedEof,
    InvalidFormat(u8),
    InvalidUtf8,
    NonStringMapKey,
    NestingTooDeep,
}

impl std::fmt::Display for MsgPackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::InvalidFormat(b) => write!(f, "invalid format byte: 0x{b:02x}"),
            Self::InvalidUtf8 => write!(f, "string is not valid UTF-8"),
            Self::NonStringMapKey => write!(f, "map key must be string for JSON"),
            Self::NestingTooDeep => write!(f, "nesting exceeds max depth"),
        }
    }
}

// ── Value ───────────────────────────────────────────────────────────────────

/// A MessagePack value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    UInt(u64),
    Int(i64),
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<Value>),
    MsgMap(Vec<(Value, Value)>),
    Ext(i8, Vec<u8>),
}

// ── Encoder ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Encoder {
    buf: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self { Self { buf: Vec::new() } }

    pub fn encode(&mut self, val: &Value) {
        match val {
            Value::Nil => self.buf.push(0xc0),
            Value::Bool(false) => self.buf.push(0xc2),
            Value::Bool(true) => self.buf.push(0xc3),
            Value::UInt(n) => self.encode_uint(*n),
            Value::Int(n) => self.encode_int(*n),
            Value::F32(v) => { self.buf.push(0xca); self.buf.extend_from_slice(&v.to_be_bytes()); }
            Value::F64(v) => { self.buf.push(0xcb); self.buf.extend_from_slice(&v.to_be_bytes()); }
            Value::Str(s) => self.encode_str(s),
            Value::Bin(b) => self.encode_bin(b),
            Value::Array(arr) => {
                self.encode_array_hdr(arr.len());
                for v in arr { self.encode(v); }
            }
            Value::MsgMap(entries) => {
                self.encode_map_hdr(entries.len());
                for (k, v) in entries { self.encode(k); self.encode(v); }
            }
            Value::Ext(type_id, data) => self.encode_ext(*type_id, data),
        }
    }

    fn encode_uint(&mut self, n: u64) {
        if n <= 127 {
            self.buf.push(n as u8);
        } else if n <= u8::MAX as u64 {
            self.buf.push(0xcc); self.buf.push(n as u8);
        } else if n <= u16::MAX as u64 {
            self.buf.push(0xcd); self.buf.extend_from_slice(&(n as u16).to_be_bytes());
        } else if n <= u32::MAX as u64 {
            self.buf.push(0xce); self.buf.extend_from_slice(&(n as u32).to_be_bytes());
        } else {
            self.buf.push(0xcf); self.buf.extend_from_slice(&n.to_be_bytes());
        }
    }

    fn encode_int(&mut self, n: i64) {
        if n >= 0 {
            self.encode_uint(n as u64);
        } else if n >= -32 {
            self.buf.push(n as u8); // negative fixint
        } else if n >= i8::MIN as i64 {
            self.buf.push(0xd0); self.buf.push(n as i8 as u8);
        } else if n >= i16::MIN as i64 {
            self.buf.push(0xd1); self.buf.extend_from_slice(&(n as i16).to_be_bytes());
        } else if n >= i32::MIN as i64 {
            self.buf.push(0xd2); self.buf.extend_from_slice(&(n as i32).to_be_bytes());
        } else {
            self.buf.push(0xd3); self.buf.extend_from_slice(&n.to_be_bytes());
        }
    }

    fn encode_str(&mut self, s: &str) {
        let len = s.len();
        if len <= 31 {
            self.buf.push(0xa0 | len as u8);
        } else if len <= u8::MAX as usize {
            self.buf.push(0xd9); self.buf.push(len as u8);
        } else if len <= u16::MAX as usize {
            self.buf.push(0xda); self.buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buf.push(0xdb); self.buf.extend_from_slice(&(len as u32).to_be_bytes());
        }
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn encode_bin(&mut self, b: &[u8]) {
        let len = b.len();
        if len <= u8::MAX as usize {
            self.buf.push(0xc4); self.buf.push(len as u8);
        } else if len <= u16::MAX as usize {
            self.buf.push(0xc5); self.buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buf.push(0xc6); self.buf.extend_from_slice(&(len as u32).to_be_bytes());
        }
        self.buf.extend_from_slice(b);
    }

    fn encode_array_hdr(&mut self, len: usize) {
        if len <= 15 {
            self.buf.push(0x90 | len as u8);
        } else if len <= u16::MAX as usize {
            self.buf.push(0xdc); self.buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buf.push(0xdd); self.buf.extend_from_slice(&(len as u32).to_be_bytes());
        }
    }

    fn encode_map_hdr(&mut self, len: usize) {
        if len <= 15 {
            self.buf.push(0x80 | len as u8);
        } else if len <= u16::MAX as usize {
            self.buf.push(0xde); self.buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            self.buf.push(0xdf); self.buf.extend_from_slice(&(len as u32).to_be_bytes());
        }
    }

    fn encode_ext(&mut self, type_id: i8, data: &[u8]) {
        let len = data.len();
        match len {
            1 => self.buf.push(0xd4),
            2 => self.buf.push(0xd5),
            4 => self.buf.push(0xd6),
            8 => self.buf.push(0xd7),
            16 => self.buf.push(0xd8),
            _ if len <= u8::MAX as usize => { self.buf.push(0xc7); self.buf.push(len as u8); }
            _ if len <= u16::MAX as usize => { self.buf.push(0xc8); self.buf.extend_from_slice(&(len as u16).to_be_bytes()); }
            _ => { self.buf.push(0xc9); self.buf.extend_from_slice(&(len as u32).to_be_bytes()); }
        }
        self.buf.push(type_id as u8);
        self.buf.extend_from_slice(data);
    }

    pub fn finish(self) -> Vec<u8> { self.buf }
}

// ── Decoder ─────────────────────────────────────────────────────────────────

pub struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self { Self { data, pos: 0, depth: 0 } }

    fn read_u8(&mut self) -> Result<u8, MsgPackError> {
        if self.pos >= self.data.len() { return Err(MsgPackError::UnexpectedEof); }
        let b = self.data[self.pos]; self.pos += 1; Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], MsgPackError> {
        if self.pos + n > self.data.len() { return Err(MsgPackError::UnexpectedEof); }
        let s = &self.data[self.pos..self.pos + n]; self.pos += n; Ok(s)
    }

    fn read_u16(&mut self) -> Result<u16, MsgPackError> {
        let b = self.read_bytes(2)?; Ok(u16::from_be_bytes([b[0], b[1]]))
    }
    fn read_u32(&mut self) -> Result<u32, MsgPackError> {
        let b = self.read_bytes(4)?; Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64, MsgPackError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub fn decode(&mut self) -> Result<Value, MsgPackError> {
        if self.depth > 128 { return Err(MsgPackError::NestingTooDeep); }
        let byte = self.read_u8()?;
        match byte {
            0x00..=0x7f => Ok(Value::UInt(byte as u64)),
            0x80..=0x8f => self.decode_map((byte & 0x0f) as usize),
            0x90..=0x9f => self.decode_array((byte & 0x0f) as usize),
            0xa0..=0xbf => self.decode_str((byte & 0x1f) as usize),
            0xc0 => Ok(Value::Nil),
            0xc2 => Ok(Value::Bool(false)),
            0xc3 => Ok(Value::Bool(true)),
            0xc4 => { let n = self.read_u8()? as usize; Ok(Value::Bin(self.read_bytes(n)?.to_vec())) }
            0xc5 => { let n = self.read_u16()? as usize; Ok(Value::Bin(self.read_bytes(n)?.to_vec())) }
            0xc6 => { let n = self.read_u32()? as usize; Ok(Value::Bin(self.read_bytes(n)?.to_vec())) }
            0xc7 => { let n = self.read_u8()? as usize; let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(n)?.to_vec())) }
            0xc8 => { let n = self.read_u16()? as usize; let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(n)?.to_vec())) }
            0xc9 => { let n = self.read_u32()? as usize; let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(n)?.to_vec())) }
            0xca => {
                let b = self.read_bytes(4)?;
                Ok(Value::F32(f32::from_be_bytes([b[0], b[1], b[2], b[3]])))
            }
            0xcb => {
                let b = self.read_bytes(8)?;
                Ok(Value::F64(f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])))
            }
            0xcc => Ok(Value::UInt(self.read_u8()? as u64)),
            0xcd => Ok(Value::UInt(self.read_u16()? as u64)),
            0xce => Ok(Value::UInt(self.read_u32()? as u64)),
            0xcf => Ok(Value::UInt(self.read_u64()?)),
            0xd0 => Ok(Value::Int(self.read_u8()? as i8 as i64)),
            0xd1 => Ok(Value::Int(self.read_u16()? as i16 as i64)),
            0xd2 => Ok(Value::Int(self.read_u32()? as i32 as i64)),
            0xd3 => Ok(Value::Int(self.read_u64()? as i64)),
            0xd4 => { let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(1)?.to_vec())) }
            0xd5 => { let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(2)?.to_vec())) }
            0xd6 => { let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(4)?.to_vec())) }
            0xd7 => { let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(8)?.to_vec())) }
            0xd8 => { let t = self.read_u8()? as i8; Ok(Value::Ext(t, self.read_bytes(16)?.to_vec())) }
            0xd9 => { let n = self.read_u8()? as usize; self.decode_str(n) }
            0xda => { let n = self.read_u16()? as usize; self.decode_str(n) }
            0xdb => { let n = self.read_u32()? as usize; self.decode_str(n) }
            0xdc => { let n = self.read_u16()? as usize; self.decode_array(n) }
            0xdd => { let n = self.read_u32()? as usize; self.decode_array(n) }
            0xde => { let n = self.read_u16()? as usize; self.decode_map(n) }
            0xdf => { let n = self.read_u32()? as usize; self.decode_map(n) }
            0xe0..=0xff => Ok(Value::Int(byte as i8 as i64)),
            other => Err(MsgPackError::InvalidFormat(other)),
        }
    }

    fn decode_str(&mut self, len: usize) -> Result<Value, MsgPackError> {
        let bytes = self.read_bytes(len)?;
        let s = std::str::from_utf8(bytes).map_err(|_| MsgPackError::InvalidUtf8)?;
        Ok(Value::Str(s.to_string()))
    }

    fn decode_array(&mut self, len: usize) -> Result<Value, MsgPackError> {
        self.depth += 1;
        let mut arr = Vec::with_capacity(len);
        for _ in 0..len { arr.push(self.decode()?); }
        self.depth -= 1;
        Ok(Value::Array(arr))
    }

    fn decode_map(&mut self, len: usize) -> Result<Value, MsgPackError> {
        self.depth += 1;
        let mut entries = Vec::with_capacity(len);
        for _ in 0..len { let k = self.decode()?; let v = self.decode()?; entries.push((k, v)); }
        self.depth -= 1;
        Ok(Value::MsgMap(entries))
    }

    pub fn position(&self) -> usize { self.pos }
}

// ── Convenience ─────────────────────────────────────────────────────────────

pub fn encode(val: &Value) -> Vec<u8> {
    let mut enc = Encoder::new(); enc.encode(val); enc.finish()
}

pub fn decode(data: &[u8]) -> Result<Value, MsgPackError> {
    Decoder::new(data).decode()
}

// ── serde_json conversion ───────────────────────────────────────────────────

pub fn from_json(jv: &serde_json::Value) -> Value {
    match jv {
        serde_json::Value::Null => Value::Nil,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= 0 { Value::UInt(i as u64) } else { Value::Int(i) }
            } else if let Some(u) = n.as_u64() {
                Value::UInt(u)
            } else {
                Value::F64(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(arr) => Value::Array(arr.iter().map(from_json).collect()),
        serde_json::Value::Object(map) => {
            Value::MsgMap(map.iter().map(|(k, v)| (Value::Str(k.clone()), from_json(v))).collect())
        }
    }
}

pub fn to_json(val: &Value) -> Result<serde_json::Value, MsgPackError> {
    Ok(match val {
        Value::Nil => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::UInt(n) => serde_json::json!(*n),
        Value::Int(n) => serde_json::json!(*n),
        Value::F32(v) => serde_json::json!(*v as f64),
        Value::F64(v) => serde_json::json!(*v),
        Value::Str(s) => serde_json::Value::String(s.clone()),
        Value::Bin(b) => serde_json::json!(b),
        Value::Array(arr) => {
            let items: Result<Vec<_>, _> = arr.iter().map(to_json).collect();
            serde_json::Value::Array(items?)
        }
        Value::MsgMap(entries) => {
            let mut map = Map::new();
            for (k, v) in entries {
                let key = match k {
                    Value::Str(s) => s.clone(),
                    _ => return Err(MsgPackError::NonStringMapKey),
                };
                map.insert(key, to_json(v)?);
            }
            serde_json::Value::Object(map)
        }
        Value::Ext(type_id, data) => {
            serde_json::json!({"__ext_type": *type_id, "__ext_data": data})
        }
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(val: &Value) -> Value {
        let bytes = encode(val);
        decode(&bytes).unwrap()
    }

    #[test]
    fn nil_roundtrip() { assert_eq!(roundtrip(&Value::Nil), Value::Nil); }

    #[test]
    fn bool_roundtrip() {
        assert_eq!(roundtrip(&Value::Bool(true)), Value::Bool(true));
        assert_eq!(roundtrip(&Value::Bool(false)), Value::Bool(false));
    }

    #[test]
    fn uint_compact() {
        let bytes = encode(&Value::UInt(42));
        assert_eq!(bytes.len(), 1); // fixint
        assert_eq!(roundtrip(&Value::UInt(42)), Value::UInt(42));
        assert_eq!(roundtrip(&Value::UInt(200)), Value::UInt(200)); // uint8
        assert_eq!(roundtrip(&Value::UInt(1000)), Value::UInt(1000)); // uint16
        assert_eq!(roundtrip(&Value::UInt(100_000)), Value::UInt(100_000)); // uint32
        assert_eq!(roundtrip(&Value::UInt(u64::MAX)), Value::UInt(u64::MAX)); // uint64
    }

    #[test]
    fn negative_int_compact() {
        let bytes = encode(&Value::Int(-1));
        assert_eq!(bytes.len(), 1); // negative fixint
        assert_eq!(roundtrip(&Value::Int(-1)), Value::Int(-1));
        assert_eq!(roundtrip(&Value::Int(-32)), Value::Int(-32));
        assert_eq!(roundtrip(&Value::Int(-100)), Value::Int(-100)); // int8
        assert_eq!(roundtrip(&Value::Int(-1000)), Value::Int(-1000)); // int16
        assert_eq!(roundtrip(&Value::Int(-100_000)), Value::Int(-100_000)); // int32
        assert_eq!(roundtrip(&Value::Int(i64::MIN)), Value::Int(i64::MIN)); // int64
    }

    #[test]
    fn float_roundtrip() {
        let f32_val = Value::F32(3.14);
        match roundtrip(&f32_val) {
            Value::F32(v) => assert!((v - 3.14).abs() < 0.001),
            other => panic!("expected F32, got {other:?}"),
        }
        let f64_val = Value::F64(std::f64::consts::PI);
        assert_eq!(roundtrip(&f64_val), f64_val);
    }

    #[test]
    fn string_roundtrip() {
        let short = Value::Str("hello".into());
        assert_eq!(roundtrip(&short), short);
        assert_eq!(encode(&short).len(), 6); // fixstr header + 5 bytes
        let medium = Value::Str("x".repeat(200));
        assert_eq!(roundtrip(&medium), medium); // str8
    }

    #[test]
    fn bin_roundtrip() {
        let val = Value::Bin(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn array_roundtrip() {
        let val = Value::Array(vec![Value::UInt(1), Value::Str("two".into()), Value::Bool(true), Value::Nil]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn map_roundtrip() {
        let val = Value::MsgMap(vec![
            (Value::Str("name".into()), Value::Str("alice".into())),
            (Value::Str("age".into()), Value::UInt(30)),
        ]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn ext_roundtrip() {
        let val = Value::Ext(42, vec![1, 2, 3, 4]);
        assert_eq!(roundtrip(&val), val);
        let val1 = Value::Ext(1, vec![0xFF]);
        assert_eq!(roundtrip(&val1), val1);
    }

    #[test]
    fn ext_fixext_sizes() {
        // fixext 2
        let v2 = Value::Ext(2, vec![0xAA, 0xBB]);
        assert_eq!(roundtrip(&v2), v2);
        // fixext 8
        let v8 = Value::Ext(3, vec![0; 8]);
        assert_eq!(roundtrip(&v8), v8);
        // fixext 16
        let v16 = Value::Ext(4, vec![0; 16]);
        assert_eq!(roundtrip(&v16), v16);
    }

    #[test]
    fn json_conversion_roundtrip() {
        let json = serde_json::json!({"name": "test", "count": 42, "active": true, "items": [1, 2, 3], "empty": null});
        let mp = from_json(&json);
        let back = to_json(&mp).unwrap();
        assert_eq!(json, back);
    }

    #[test]
    fn nested_structures() {
        let val = Value::MsgMap(vec![
            (Value::Str("outer".into()), Value::MsgMap(vec![
                (Value::Str("inner".into()), Value::Array(vec![Value::UInt(1), Value::UInt(2)])),
            ])),
        ]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn empty_containers() {
        assert_eq!(roundtrip(&Value::Array(vec![])), Value::Array(vec![]));
        assert_eq!(roundtrip(&Value::MsgMap(vec![])), Value::MsgMap(vec![]));
        assert_eq!(roundtrip(&Value::Str(String::new())), Value::Str(String::new()));
        assert_eq!(roundtrip(&Value::Bin(vec![])), Value::Bin(vec![]));
    }

    #[test]
    fn unexpected_eof() {
        assert!(decode(&[]).is_err());
        assert!(decode(&[0xcc]).is_err()); // uint8 missing data
    }

    #[test]
    fn non_string_key_error() {
        let val = Value::MsgMap(vec![(Value::UInt(1), Value::Nil)]);
        assert!(matches!(to_json(&val), Err(MsgPackError::NonStringMapKey)));
    }

    #[test]
    fn large_array() {
        // array 16
        let arr: Vec<Value> = (0..20).map(|i| Value::UInt(i)).collect();
        let val = Value::Array(arr);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn error_display() {
        assert_eq!(format!("{}", MsgPackError::UnexpectedEof), "unexpected end of input");
        assert_eq!(format!("{}", MsgPackError::InvalidFormat(0xFF)), "invalid format byte: 0xff");
    }

    #[test]
    fn positive_int_as_uint() {
        // Positive i64 should roundtrip through UInt encoding
        let val = Value::Int(42);
        let rt = roundtrip(&val);
        // encode_int for positive uses encode_uint, so it comes back as UInt
        assert_eq!(rt, Value::UInt(42));
    }

    #[test]
    fn json_negative_number() {
        let json = serde_json::json!({"x": -5});
        let mp = from_json(&json);
        let back = to_json(&mp).unwrap();
        assert_eq!(back["x"], serde_json::json!(-5));
    }
}
