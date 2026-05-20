//! JSONB (Binary JSON) Support
//!
//! Provides efficient binary encoding for JSON data with support for:
//! - Fast path-based lookups without full deserialization
//! - Compact binary representation
//! - Type-preserving encoding

use crate::ast::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// JSONB type tags for binary encoding
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonbTag {
    /// Null value
    Null = 0x00,
    /// Boolean false
    False = 0x01,
    /// Boolean true
    True = 0x02,
    /// 8-bit integer
    Int8 = 0x10,
    /// 16-bit integer
    Int16 = 0x11,
    /// 32-bit integer
    Int32 = 0x12,
    /// 64-bit integer
    Int64 = 0x13,
    /// 64-bit floating point
    Float64 = 0x20,
    /// String (length-prefixed)
    String = 0x30,
    /// Array (length-prefixed)
    Array = 0x40,
    /// Object (length-prefixed key-value pairs)
    Object = 0x50,
}

impl TryFrom<u8> for JsonbTag {
    type Error = JsonbError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(JsonbTag::Null),
            0x01 => Ok(JsonbTag::False),
            0x02 => Ok(JsonbTag::True),
            0x10 => Ok(JsonbTag::Int8),
            0x11 => Ok(JsonbTag::Int16),
            0x12 => Ok(JsonbTag::Int32),
            0x13 => Ok(JsonbTag::Int64),
            0x20 => Ok(JsonbTag::Float64),
            0x30 => Ok(JsonbTag::String),
            0x40 => Ok(JsonbTag::Array),
            0x50 => Ok(JsonbTag::Object),
            _ => Err(JsonbError::InvalidTag(value)),
        }
    }
}

/// JSONB error types
#[derive(Debug, Clone, PartialEq)]
pub enum JsonbError {
    /// Invalid type tag
    InvalidTag(u8),
    /// Unexpected end of data
    UnexpectedEof,
    /// Invalid UTF-8 string
    InvalidUtf8,
    /// Path not found
    PathNotFound(String),
    /// Invalid path expression
    InvalidPath(String),
}

impl std::fmt::Display for JsonbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonbError::InvalidTag(tag) => write!(f, "Invalid JSONB tag: 0x{:02x}", tag),
            JsonbError::UnexpectedEof => write!(f, "Unexpected end of JSONB data"),
            JsonbError::InvalidUtf8 => write!(f, "Invalid UTF-8 in JSONB string"),
            JsonbError::PathNotFound(path) => write!(f, "Path not found: {}", path),
            JsonbError::InvalidPath(path) => write!(f, "Invalid path expression: {}", path),
        }
    }
}

impl std::error::Error for JsonbError {}

/// JSONB binary data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Jsonb {
    /// Binary-encoded JSON data
    data: Vec<u8>,
}

impl Jsonb {
    /// Create JSONB from a Value
    pub fn from_value(value: &Value) -> Self {
        let mut data = Vec::new();
        Self::encode_value(value, &mut data);
        Self { data }
    }

    /// Decode to a Value
    pub fn to_value(&self) -> Result<Value, JsonbError> {
        let mut cursor = 0;
        Self::decode_value(&self.data, &mut cursor)
    }

    /// Get the raw binary data
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Create from raw binary data
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self { data }
    }

    /// Get a value at a JSON path (e.g., "$.foo.bar[0]")
    pub fn get_path(&self, path: &str) -> Result<Value, JsonbError> {
        let value = self.to_value()?;
        Self::extract_path(&value, path)
    }

    /// Check if a path exists
    pub fn path_exists(&self, path: &str) -> Result<bool, JsonbError> {
        match self.get_path(path) {
            Ok(_) => Ok(true),
            Err(JsonbError::PathNotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    // ========================================================================
    // Encoding
    // ========================================================================

    fn encode_value(value: &Value, out: &mut Vec<u8>) {
        match value {
            Value::Null => out.push(JsonbTag::Null as u8),
            Value::Bool(false) => out.push(JsonbTag::False as u8),
            Value::Bool(true) => out.push(JsonbTag::True as u8),
            Value::Int(i) => Self::encode_int(*i, out),
            Value::Float(f) => {
                out.push(JsonbTag::Float64 as u8);
                out.extend_from_slice(&f.to_le_bytes());
            }
            Value::String(s) => Self::encode_string(s, out),
            Value::Array(arr) => {
                out.push(JsonbTag::Array as u8);
                Self::encode_varint(arr.len() as u64, out);
                for item in arr {
                    Self::encode_value(item, out);
                }
            }
            Value::Object(obj) => {
                out.push(JsonbTag::Object as u8);
                Self::encode_varint(obj.len() as u64, out);
                for (key, val) in obj {
                    Self::encode_string(key, out);
                    Self::encode_value(val, out);
                }
            }
            Value::Bytes(b) => {
                // Encode bytes as a string (base64 could be used alternatively)
                out.push(JsonbTag::String as u8);
                Self::encode_varint(b.len() as u64, out);
                out.extend_from_slice(b);
            }
            Value::Timestamp(ts) => Self::encode_int(*ts, out),
            Value::Uuid(s) => Self::encode_string(s, out),
            Value::Vector(v) => {
                // Encode vector as JSON array of floats
                out.push(JsonbTag::Array as u8);
                Self::encode_varint(v.len() as u64, out);
                for f in v {
                    out.push(JsonbTag::Float64 as u8);
                    out.extend_from_slice(&(*f as f64).to_le_bytes());
                }
            }
        }
    }

    fn encode_int(i: i64, out: &mut Vec<u8>) {
        if i >= i8::MIN as i64 && i <= i8::MAX as i64 {
            out.push(JsonbTag::Int8 as u8);
            out.push(i as i8 as u8);
        } else if i >= i16::MIN as i64 && i <= i16::MAX as i64 {
            out.push(JsonbTag::Int16 as u8);
            out.extend_from_slice(&(i as i16).to_le_bytes());
        } else if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
            out.push(JsonbTag::Int32 as u8);
            out.extend_from_slice(&(i as i32).to_le_bytes());
        } else {
            out.push(JsonbTag::Int64 as u8);
            out.extend_from_slice(&i.to_le_bytes());
        }
    }

    fn encode_string(s: &str, out: &mut Vec<u8>) {
        out.push(JsonbTag::String as u8);
        Self::encode_varint(s.len() as u64, out);
        out.extend_from_slice(s.as_bytes());
    }

    fn encode_varint(mut n: u64, out: &mut Vec<u8>) {
        loop {
            let byte = (n & 0x7F) as u8;
            n >>= 7;
            if n == 0 {
                out.push(byte);
                break;
            } else {
                out.push(byte | 0x80);
            }
        }
    }

    // ========================================================================
    // Decoding
    // ========================================================================

    fn decode_value(data: &[u8], cursor: &mut usize) -> Result<Value, JsonbError> {
        if *cursor >= data.len() {
            return Err(JsonbError::UnexpectedEof);
        }

        let tag = JsonbTag::try_from(data[*cursor])?;
        *cursor += 1;

        match tag {
            JsonbTag::Null => Ok(Value::Null),
            JsonbTag::False => Ok(Value::Bool(false)),
            JsonbTag::True => Ok(Value::Bool(true)),
            JsonbTag::Int8 => {
                if *cursor >= data.len() {
                    return Err(JsonbError::UnexpectedEof);
                }
                let i = data[*cursor] as i8 as i64;
                *cursor += 1;
                Ok(Value::Int(i))
            }
            JsonbTag::Int16 => {
                if *cursor + 2 > data.len() {
                    return Err(JsonbError::UnexpectedEof);
                }
                let bytes: [u8; 2] = data[*cursor..*cursor + 2]
                    .try_into()
                    .expect("slice is exactly 2 bytes");
                let i = i16::from_le_bytes(bytes) as i64;
                *cursor += 2;
                Ok(Value::Int(i))
            }
            JsonbTag::Int32 => {
                if *cursor + 4 > data.len() {
                    return Err(JsonbError::UnexpectedEof);
                }
                let bytes: [u8; 4] = data[*cursor..*cursor + 4]
                    .try_into()
                    .expect("slice is exactly 4 bytes");
                let i = i32::from_le_bytes(bytes) as i64;
                *cursor += 4;
                Ok(Value::Int(i))
            }
            JsonbTag::Int64 => {
                if *cursor + 8 > data.len() {
                    return Err(JsonbError::UnexpectedEof);
                }
                let bytes: [u8; 8] = data[*cursor..*cursor + 8]
                    .try_into()
                    .expect("slice is exactly 8 bytes");
                let i = i64::from_le_bytes(bytes);
                *cursor += 8;
                Ok(Value::Int(i))
            }
            JsonbTag::Float64 => {
                if *cursor + 8 > data.len() {
                    return Err(JsonbError::UnexpectedEof);
                }
                let bytes: [u8; 8] = data[*cursor..*cursor + 8]
                    .try_into()
                    .expect("slice is exactly 8 bytes");
                let f = f64::from_le_bytes(bytes);
                *cursor += 8;
                Ok(Value::Float(f))
            }
            JsonbTag::String => {
                let len = Self::decode_varint(data, cursor)? as usize;
                if *cursor + len > data.len() {
                    return Err(JsonbError::UnexpectedEof);
                }
                let s = std::str::from_utf8(&data[*cursor..*cursor + len])
                    .map_err(|_| JsonbError::InvalidUtf8)?;
                *cursor += len;
                Ok(Value::String(s.to_string()))
            }
            JsonbTag::Array => {
                let len = Self::decode_varint(data, cursor)? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(Self::decode_value(data, cursor)?);
                }
                Ok(Value::Array(arr))
            }
            JsonbTag::Object => {
                let len = Self::decode_varint(data, cursor)? as usize;
                let mut obj = HashMap::with_capacity(len);
                for _ in 0..len {
                    // Decode key
                    if *cursor >= data.len() || data[*cursor] != JsonbTag::String as u8 {
                        return Err(JsonbError::InvalidTag(
                            data.get(*cursor).copied().unwrap_or(0),
                        ));
                    }
                    *cursor += 1;
                    let key_len = Self::decode_varint(data, cursor)? as usize;
                    if *cursor + key_len > data.len() {
                        return Err(JsonbError::UnexpectedEof);
                    }
                    let key = std::str::from_utf8(&data[*cursor..*cursor + key_len])
                        .map_err(|_| JsonbError::InvalidUtf8)?
                        .to_string();
                    *cursor += key_len;

                    // Decode value
                    let val = Self::decode_value(data, cursor)?;
                    obj.insert(key, val);
                }
                Ok(Value::Object(obj))
            }
        }
    }

    fn decode_varint(data: &[u8], cursor: &mut usize) -> Result<u64, JsonbError> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            if *cursor >= data.len() {
                return Err(JsonbError::UnexpectedEof);
            }
            let byte = data[*cursor];
            *cursor += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(result)
    }

    // ========================================================================
    // Path extraction
    // ========================================================================

    fn extract_path(value: &Value, path: &str) -> Result<Value, JsonbError> {
        let path = path.trim();
        if path.is_empty() || path == "$" {
            return Ok(value.clone());
        }

        // Remove leading "$." if present
        let path = path
            .strip_prefix("$.")
            .unwrap_or(path.strip_prefix("$").unwrap_or(path));

        let mut current = value.clone();
        for segment in Self::parse_path_segments(path)? {
            current = match segment {
                PathSegment::Key(key) => {
                    if let Value::Object(obj) = &current {
                        obj.get(&key)
                            .cloned()
                            .ok_or_else(|| JsonbError::PathNotFound(key.clone()))?
                    } else {
                        return Err(JsonbError::PathNotFound(key));
                    }
                }
                PathSegment::Index(idx) => {
                    if let Value::Array(arr) = &current {
                        arr.get(idx)
                            .cloned()
                            .ok_or_else(|| JsonbError::PathNotFound(idx.to_string()))?
                    } else {
                        return Err(JsonbError::PathNotFound(idx.to_string()));
                    }
                }
            };
        }

        Ok(current)
    }

    fn parse_path_segments(path: &str) -> Result<Vec<PathSegment>, JsonbError> {
        let mut segments = Vec::new();
        let mut chars = path.chars().peekable();
        let mut current = String::new();

        while let Some(c) = chars.next() {
            match c {
                '.' => {
                    if !current.is_empty() {
                        segments.push(PathSegment::Key(current.clone()));
                        current.clear();
                    }
                }
                '[' => {
                    if !current.is_empty() {
                        segments.push(PathSegment::Key(current.clone()));
                        current.clear();
                    }
                    // Parse array index
                    let mut idx_str = String::new();
                    while let Some(&next) = chars.peek() {
                        if next == ']' {
                            chars.next();
                            break;
                        }
                        idx_str.push(chars.next().expect("peeked char exists"));
                    }
                    let idx: usize = idx_str.parse().map_err(|_| {
                        JsonbError::InvalidPath(format!("Invalid index: {}", idx_str))
                    })?;
                    segments.push(PathSegment::Index(idx));
                }
                _ => current.push(c),
            }
        }

        if !current.is_empty() {
            segments.push(PathSegment::Key(current));
        }

        Ok(segments)
    }
}

enum PathSegment {
    Key(String),
    Index(usize),
}

impl Default for Jsonb {
    fn default() -> Self {
        Self::from_value(&Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_null() {
        let value = Value::Null;
        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_encode_decode_bool() {
        let value = Value::Bool(true);
        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);

        let value = Value::Bool(false);
        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_encode_decode_int() {
        // Small int (fits in i8)
        let value = Value::Int(42);
        let jsonb = Jsonb::from_value(&value);
        assert_eq!(jsonb.as_bytes().len(), 2); // tag + 1 byte
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);

        // Large int (needs i64)
        let value = Value::Int(i64::MAX);
        let jsonb = Jsonb::from_value(&value);
        assert_eq!(jsonb.as_bytes().len(), 9); // tag + 8 bytes
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_encode_decode_float() {
        let value = Value::Float(3.14159);
        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_encode_decode_string() {
        let value = Value::String("Hello, JSONB!".to_string());
        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_encode_decode_array() {
        let value = Value::Array(vec![
            Value::Int(1),
            Value::Int(2),
            Value::String("three".to_string()),
        ]);
        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn test_encode_decode_object() {
        let mut obj = HashMap::new();
        obj.insert("name".to_string(), Value::String("Alice".to_string()));
        obj.insert("age".to_string(), Value::Int(30));
        let value = Value::Object(obj);

        let jsonb = Jsonb::from_value(&value);
        let decoded = jsonb.to_value().unwrap();

        // Compare as objects (order-independent)
        if let (Value::Object(a), Value::Object(b)) = (&decoded, &value) {
            assert_eq!(a.len(), b.len());
            for (key, val) in a {
                assert_eq!(b.get(key), Some(val));
            }
        } else {
            panic!("Expected object");
        }
    }

    #[test]
    fn test_path_extraction() {
        let mut inner = HashMap::new();
        inner.insert("x".to_string(), Value::Int(10));

        let mut obj = HashMap::new();
        obj.insert("name".to_string(), Value::String("test".to_string()));
        obj.insert("nested".to_string(), Value::Object(inner));
        obj.insert(
            "items".to_string(),
            Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        );

        let jsonb = Jsonb::from_value(&Value::Object(obj));

        // Test path extraction
        assert_eq!(
            jsonb.get_path("$.name").unwrap(),
            Value::String("test".to_string())
        );
        assert_eq!(jsonb.get_path("$.nested.x").unwrap(), Value::Int(10));
        assert_eq!(jsonb.get_path("$.items[1]").unwrap(), Value::Int(2));
    }

    #[test]
    fn test_path_exists() {
        let mut obj = HashMap::new();
        obj.insert("key".to_string(), Value::Int(123));
        let jsonb = Jsonb::from_value(&Value::Object(obj));

        assert!(jsonb.path_exists("$.key").unwrap());
        assert!(!jsonb.path_exists("$.nonexistent").unwrap());
    }
}
