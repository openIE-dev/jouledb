//! Database value types

use crate::error::CodecError;
use crate::types::spatial::Spatial3dValue;
use std::collections::BTreeMap;

/// Type tags for binary encoding
mod tags {
    pub const NULL: u8 = 0;
    pub const BOOL_FALSE: u8 = 1;
    pub const BOOL_TRUE: u8 = 2;
    pub const INT: u8 = 3;
    pub const FLOAT: u8 = 4;
    pub const STRING: u8 = 5;
    pub const BYTES: u8 = 6;
    pub const ARRAY: u8 = 7;
    pub const MAP: u8 = 8;
    pub const TIMESTAMP: u8 = 9;
    pub const VECTOR: u8 = 10;
    /// 3D spatial value (Point3 / Quat / Pose6 / Bbox3). Inner tag follows.
    pub const SPATIAL3D: u8 = 11;
}

/// Database value type
///
/// Represents all possible values that can be stored in JouleDB.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Value {
    /// Null/missing value
    Null,
    /// Boolean
    Bool(bool),
    /// 64-bit signed integer
    Int(i64),
    /// 64-bit floating point
    Float(f64),
    /// UTF-8 string
    String(String),
    /// Raw bytes
    Bytes(Vec<u8>),
    /// Array of values
    Array(Vec<Value>),
    /// Map of string keys to values
    Map(BTreeMap<String, Value>),
    /// Timestamp (microseconds since Unix epoch)
    Timestamp(i64),
    /// Fixed-dimension vector of f32 values (for similarity search)
    Vector(Vec<f32>),
    /// 3D spatial value — Point3, Quat, Pose6, or Bbox3.
    /// First-class so spatial indexes (R-tree / kd-tree / octree) and the
    /// query planner can push spatial predicates without serializing through
    /// JSON or Bytes.
    Spatial3d(Spatial3dValue),
}

impl Value {
    /// Get the type name as a string
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::Array(_) => "array",
            Value::Map(_) => "map",
            Value::Timestamp(_) => "timestamp",
            Value::Vector(_) => "vector",
            Value::Spatial3d(s) => s.type_name(),
        }
    }

    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get as f64
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as string slice
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as byte slice
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Try to get as array
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(arr) => Some(arr),
            _ => None,
        }
    }

    /// Try to get as map
    pub fn as_map(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Map(map) => Some(map),
            _ => None,
        }
    }

    /// Try to get as vector
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Value::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Encode value to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.encode_into(&mut buf);
        buf
    }

    /// Encode value into an existing buffer
    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        match self {
            Value::Null => {
                buf.push(tags::NULL);
            }
            Value::Bool(false) => {
                buf.push(tags::BOOL_FALSE);
            }
            Value::Bool(true) => {
                buf.push(tags::BOOL_TRUE);
            }
            Value::Int(i) => {
                buf.push(tags::INT);
                buf.extend_from_slice(&i.to_le_bytes());
            }
            Value::Float(f) => {
                buf.push(tags::FLOAT);
                buf.extend_from_slice(&f.to_le_bytes());
            }
            Value::String(s) => {
                buf.push(tags::STRING);
                let bytes = s.as_bytes();
                buf.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(bytes);
            }
            Value::Bytes(b) => {
                buf.push(tags::BYTES);
                buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
                buf.extend_from_slice(b);
            }
            Value::Array(arr) => {
                buf.push(tags::ARRAY);
                buf.extend_from_slice(&(arr.len() as u32).to_le_bytes());
                for item in arr {
                    item.encode_into(buf);
                }
            }
            Value::Map(map) => {
                buf.push(tags::MAP);
                buf.extend_from_slice(&(map.len() as u32).to_le_bytes());
                for (key, value) in map {
                    let key_bytes = key.as_bytes();
                    buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
                    buf.extend_from_slice(key_bytes);
                    value.encode_into(buf);
                }
            }
            Value::Timestamp(ts) => {
                buf.push(tags::TIMESTAMP);
                buf.extend_from_slice(&ts.to_le_bytes());
            }
            Value::Vector(v) => {
                buf.push(tags::VECTOR);
                buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for f in v {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            Value::Spatial3d(s) => {
                buf.push(tags::SPATIAL3D);
                s.encode_into(buf);
            }
        }
    }

    /// Decode value from bytes
    pub fn decode(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut cursor = 0;
        Self::decode_at(bytes, &mut cursor)
    }

    /// Decode value from bytes at a given position
    fn decode_at(bytes: &[u8], cursor: &mut usize) -> Result<Self, CodecError> {
        if *cursor >= bytes.len() {
            return Err(CodecError::UnexpectedEof {
                expected: 1,
                actual: 0,
            });
        }

        let tag = bytes[*cursor];
        *cursor += 1;

        match tag {
            tags::NULL => Ok(Value::Null),
            tags::BOOL_FALSE => Ok(Value::Bool(false)),
            tags::BOOL_TRUE => Ok(Value::Bool(true)),
            tags::INT => {
                let i = read_i64(bytes, cursor)?;
                Ok(Value::Int(i))
            }
            tags::FLOAT => {
                let f = read_f64(bytes, cursor)?;
                Ok(Value::Float(f))
            }
            tags::STRING => {
                let len = read_u32(bytes, cursor)? as usize;
                let s = read_string(bytes, cursor, len)?;
                Ok(Value::String(s))
            }
            tags::BYTES => {
                let len = read_u32(bytes, cursor)? as usize;
                let b = read_bytes(bytes, cursor, len)?;
                Ok(Value::Bytes(b))
            }
            tags::ARRAY => {
                let len = read_u32(bytes, cursor)? as usize;
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(Self::decode_at(bytes, cursor)?);
                }
                Ok(Value::Array(arr))
            }
            tags::MAP => {
                let len = read_u32(bytes, cursor)? as usize;
                let mut map = BTreeMap::new();
                for _ in 0..len {
                    let key_len = read_u32(bytes, cursor)? as usize;
                    let key = read_string(bytes, cursor, key_len)?;
                    let value = Self::decode_at(bytes, cursor)?;
                    map.insert(key, value);
                }
                Ok(Value::Map(map))
            }
            tags::TIMESTAMP => {
                let ts = read_i64(bytes, cursor)?;
                Ok(Value::Timestamp(ts))
            }
            tags::VECTOR => {
                let len = read_u32(bytes, cursor)? as usize;
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    if *cursor + 4 > bytes.len() {
                        return Err(CodecError::UnexpectedEof {
                            expected: 4,
                            actual: bytes.len() - *cursor,
                        });
                    }
                    let arr: [u8; 4] = bytes[*cursor..*cursor + 4]
                        .try_into()
                        .expect("exact 4-byte slice");
                    *cursor += 4;
                    v.push(f32::from_le_bytes(arr));
                }
                Ok(Value::Vector(v))
            }
            tags::SPATIAL3D => {
                let s = Spatial3dValue::decode_at(bytes, cursor)?;
                Ok(Value::Spatial3d(s))
            }
            _ => Err(CodecError::UnknownType { tag }),
        }
    }
}

// Helper functions for reading primitive types

fn read_i64(bytes: &[u8], cursor: &mut usize) -> Result<i64, CodecError> {
    if *cursor + 8 > bytes.len() {
        return Err(CodecError::UnexpectedEof {
            expected: 8,
            actual: bytes.len() - *cursor,
        });
    }
    let arr: [u8; 8] = bytes[*cursor..*cursor + 8]
        .try_into()
        .expect("exact 8-byte slice");
    *cursor += 8;
    Ok(i64::from_le_bytes(arr))
}

fn read_f64(bytes: &[u8], cursor: &mut usize) -> Result<f64, CodecError> {
    if *cursor + 8 > bytes.len() {
        return Err(CodecError::UnexpectedEof {
            expected: 8,
            actual: bytes.len() - *cursor,
        });
    }
    let arr: [u8; 8] = bytes[*cursor..*cursor + 8]
        .try_into()
        .expect("exact 8-byte slice");
    *cursor += 8;
    Ok(f64::from_le_bytes(arr))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, CodecError> {
    if *cursor + 4 > bytes.len() {
        return Err(CodecError::UnexpectedEof {
            expected: 4,
            actual: bytes.len() - *cursor,
        });
    }
    let arr: [u8; 4] = bytes[*cursor..*cursor + 4]
        .try_into()
        .expect("exact 4-byte slice");
    *cursor += 4;
    Ok(u32::from_le_bytes(arr))
}

fn read_bytes(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<u8>, CodecError> {
    if *cursor + len > bytes.len() {
        return Err(CodecError::UnexpectedEof {
            expected: len,
            actual: bytes.len() - *cursor,
        });
    }
    let result = bytes[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Ok(result)
}

fn read_string(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<String, CodecError> {
    let raw = read_bytes(bytes, cursor, len)?;
    String::from_utf8(raw).map_err(|_| CodecError::InvalidUtf8)
}

// Conversion traits

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Int(i as i64)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<Vec<u8>> for Value {
    fn from(b: Vec<u8>) -> Self {
        Value::Bytes(b)
    }
}

impl From<Vec<Value>> for Value {
    fn from(arr: Vec<Value>) -> Self {
        Value::Array(arr)
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_primitives() {
        let values = vec![
            Value::Null,
            Value::Bool(true),
            Value::Bool(false),
            Value::Int(42),
            Value::Int(-1000),
            Value::Float(3.14159),
            Value::String("hello world".to_string()),
            Value::Bytes(vec![1, 2, 3, 4, 5]),
            Value::Timestamp(1234567890),
        ];

        for value in values {
            let encoded = value.encode();
            let decoded = Value::decode(&encoded).unwrap();
            assert_eq!(value, decoded, "Failed for {:?}", value);
        }
    }

    #[test]
    fn test_encode_decode_array() {
        let value = Value::Array(vec![
            Value::Int(1),
            Value::String("two".to_string()),
            Value::Bool(true),
        ]);

        let encoded = value.encode();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_encode_decode_map() {
        let mut map = BTreeMap::new();
        map.insert("name".to_string(), Value::String("Alice".to_string()));
        map.insert("age".to_string(), Value::Int(30));
        let value = Value::Map(map);

        let encoded = value.encode();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_nested_structures() {
        let mut inner_map = BTreeMap::new();
        inner_map.insert("x".to_string(), Value::Int(10));
        inner_map.insert("y".to_string(), Value::Int(20));

        let value = Value::Array(vec![
            Value::Map(inner_map),
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
        ]);

        let encoded = value.encode();
        let decoded = Value::decode(&encoded).unwrap();
        assert_eq!(value, decoded);
    }

    #[test]
    fn test_type_name() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Bool(true).type_name(), "bool");
        assert_eq!(Value::Int(0).type_name(), "int");
    }

    #[test]
    fn test_accessors() {
        assert_eq!(Value::Bool(true).as_bool(), Some(true));
        assert_eq!(Value::Int(42).as_int(), Some(42));
        assert_eq!(Value::Float(3.14).as_float(), Some(3.14));
        assert_eq!(Value::Int(42).as_float(), Some(42.0));
        assert_eq!(Value::String("hello".into()).as_str(), Some("hello"));
    }
}
