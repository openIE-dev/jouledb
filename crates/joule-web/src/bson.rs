//! BSON codec — Binary JSON as used by MongoDB.
//!
//! Supports document encoding/decoding, all standard element types (double,
//! string, document, array, binary, ObjectId, boolean, datetime, null,
//! regex, int32, int64), ObjectId generation, and a document builder.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BsonError {
    UnexpectedEof,
    InvalidElementType(u8),
    InvalidUtf8,
    InvalidDocument,
    InvalidObjectId,
    DocumentTooSmall,
    MissingNullTerminator,
}

impl fmt::Display for BsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::InvalidElementType(t) => write!(f, "invalid element type: 0x{t:02x}"),
            Self::InvalidUtf8 => write!(f, "string is not valid UTF-8"),
            Self::InvalidDocument => write!(f, "invalid document structure"),
            Self::InvalidObjectId => write!(f, "invalid ObjectId"),
            Self::DocumentTooSmall => write!(f, "document size < 5 bytes"),
            Self::MissingNullTerminator => write!(f, "missing null terminator"),
        }
    }
}

impl std::error::Error for BsonError {}

// ── Element types ───────────────────────────────────────────────

/// BSON element type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ElementType {
    Double = 0x01,
    String = 0x02,
    Document = 0x03,
    Array = 0x04,
    Binary = 0x05,
    ObjectId = 0x07,
    Boolean = 0x08,
    DateTime = 0x09,
    Null = 0x0A,
    Regex = 0x0B,
    Int32 = 0x10,
    Int64 = 0x12,
}

impl ElementType {
    fn from_u8(v: u8) -> Result<Self, BsonError> {
        match v {
            0x01 => Ok(Self::Double),
            0x02 => Ok(Self::String),
            0x03 => Ok(Self::Document),
            0x04 => Ok(Self::Array),
            0x05 => Ok(Self::Binary),
            0x07 => Ok(Self::ObjectId),
            0x08 => Ok(Self::Boolean),
            0x09 => Ok(Self::DateTime),
            0x0A => Ok(Self::Null),
            0x0B => Ok(Self::Regex),
            0x10 => Ok(Self::Int32),
            0x12 => Ok(Self::Int64),
            other => Err(BsonError::InvalidElementType(other)),
        }
    }
}

// ── ObjectId ────────────────────────────────────────────────────

static OBJECT_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

/// A 12-byte BSON ObjectId.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ObjectId {
    bytes: [u8; 12],
}

impl ObjectId {
    /// Generate a new ObjectId.
    pub fn new() -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as u32;

        // 4 bytes timestamp + 5 bytes random + 3 bytes counter
        let mut bytes = [0u8; 12];
        bytes[0..4].copy_from_slice(&timestamp.to_be_bytes());

        // Use a simple hash of thread id for the "random" portion
        let thread_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            hasher.finish()
        };
        bytes[4] = (thread_hash >> 0) as u8;
        bytes[5] = (thread_hash >> 8) as u8;
        bytes[6] = (thread_hash >> 16) as u8;
        bytes[7] = (thread_hash >> 24) as u8;
        bytes[8] = (thread_hash >> 32) as u8;

        let counter = OBJECT_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        bytes[9] = (counter >> 16) as u8;
        bytes[10] = (counter >> 8) as u8;
        bytes[11] = counter as u8;

        Self { bytes }
    }

    /// Create from 12 bytes.
    pub fn from_bytes(bytes: [u8; 12]) -> Self {
        Self { bytes }
    }

    /// Create from hex string.
    pub fn from_hex(hex: &str) -> Result<Self, BsonError> {
        if hex.len() != 24 {
            return Err(BsonError::InvalidObjectId);
        }
        let mut bytes = [0u8; 12];
        for i in 0..12 {
            bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
                .map_err(|_| BsonError::InvalidObjectId)?;
        }
        Ok(Self { bytes })
    }

    /// Get the 4-byte timestamp.
    pub fn timestamp(&self) -> u32 {
        u32::from_be_bytes([self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]])
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 12] {
        &self.bytes
    }

    /// Format as hex string.
    pub fn to_hex(&self) -> String {
        self.bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectId(\"{}\")", self.to_hex())
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

impl Default for ObjectId {
    fn default() -> Self {
        Self::new()
    }
}

// ── Value ───────────────────────────────────────────────────────

/// A BSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Double(f64),
    String(String),
    Document(Document),
    Array(Vec<Value>),
    Binary(u8, Vec<u8>),  // subtype, data
    ObjectId(ObjectId),
    Boolean(bool),
    DateTime(i64),  // milliseconds since epoch
    Null,
    Regex(String, String),  // pattern, options
    Int32(i32),
    Int64(i64),
}

// ── Document ────────────────────────────────────────────────────

/// An ordered BSON document.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Document {
    entries: Vec<(String, Value)>,
}

impl Document {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Insert a key-value pair (appends; keeps order).
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.entries.push((key.into(), value));
    }

    /// Get value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the document is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over entries.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Value)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }
}

// ── Document builder ────────────────────────────────────────────

/// Fluent document builder.
pub struct DocumentBuilder {
    doc: Document,
}

impl DocumentBuilder {
    pub fn new() -> Self {
        Self { doc: Document::new() }
    }

    pub fn double(mut self, key: &str, val: f64) -> Self {
        self.doc.insert(key, Value::Double(val));
        self
    }

    pub fn string(mut self, key: &str, val: &str) -> Self {
        self.doc.insert(key, Value::String(val.to_string()));
        self
    }

    pub fn document(mut self, key: &str, val: Document) -> Self {
        self.doc.insert(key, Value::Document(val));
        self
    }

    pub fn array(mut self, key: &str, val: Vec<Value>) -> Self {
        self.doc.insert(key, Value::Array(val));
        self
    }

    pub fn binary(mut self, key: &str, subtype: u8, data: Vec<u8>) -> Self {
        self.doc.insert(key, Value::Binary(subtype, data));
        self
    }

    pub fn object_id(mut self, key: &str, oid: ObjectId) -> Self {
        self.doc.insert(key, Value::ObjectId(oid));
        self
    }

    pub fn boolean(mut self, key: &str, val: bool) -> Self {
        self.doc.insert(key, Value::Boolean(val));
        self
    }

    pub fn datetime(mut self, key: &str, millis: i64) -> Self {
        self.doc.insert(key, Value::DateTime(millis));
        self
    }

    pub fn null(mut self, key: &str) -> Self {
        self.doc.insert(key, Value::Null);
        self
    }

    pub fn regex(mut self, key: &str, pattern: &str, options: &str) -> Self {
        self.doc.insert(key, Value::Regex(pattern.to_string(), options.to_string()));
        self
    }

    pub fn int32(mut self, key: &str, val: i32) -> Self {
        self.doc.insert(key, Value::Int32(val));
        self
    }

    pub fn int64(mut self, key: &str, val: i64) -> Self {
        self.doc.insert(key, Value::Int64(val));
        self
    }

    pub fn build(self) -> Document {
        self.doc
    }
}

impl Default for DocumentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Encoder ─────────────────────────────────────────────────────

/// Encode a document to BSON bytes.
pub fn encode(doc: &Document) -> Vec<u8> {
    let mut buf = Vec::new();
    // Placeholder for document size (4 bytes)
    buf.extend_from_slice(&[0u8; 4]);

    for (key, val) in &doc.entries {
        encode_element(&mut buf, key, val);
    }

    // Null terminator
    buf.push(0x00);

    // Write actual size
    let size = buf.len() as i32;
    buf[0..4].copy_from_slice(&size.to_le_bytes());
    buf
}

fn encode_element(buf: &mut Vec<u8>, key: &str, val: &Value) {
    match val {
        Value::Double(v) => {
            buf.push(ElementType::Double as u8);
            write_cstring(buf, key);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Value::String(s) => {
            buf.push(ElementType::String as u8);
            write_cstring(buf, key);
            write_string(buf, s);
        }
        Value::Document(doc) => {
            buf.push(ElementType::Document as u8);
            write_cstring(buf, key);
            let encoded = encode(doc);
            buf.extend_from_slice(&encoded);
        }
        Value::Array(arr) => {
            buf.push(ElementType::Array as u8);
            write_cstring(buf, key);
            // Encode array as document with "0", "1", "2"... keys
            let mut array_doc = Document::new();
            for (i, v) in arr.iter().enumerate() {
                array_doc.insert(i.to_string(), v.clone());
            }
            let encoded = encode(&array_doc);
            buf.extend_from_slice(&encoded);
        }
        Value::Binary(subtype, data) => {
            buf.push(ElementType::Binary as u8);
            write_cstring(buf, key);
            buf.extend_from_slice(&(data.len() as i32).to_le_bytes());
            buf.push(*subtype);
            buf.extend_from_slice(data);
        }
        Value::ObjectId(oid) => {
            buf.push(ElementType::ObjectId as u8);
            write_cstring(buf, key);
            buf.extend_from_slice(oid.as_bytes());
        }
        Value::Boolean(b) => {
            buf.push(ElementType::Boolean as u8);
            write_cstring(buf, key);
            buf.push(if *b { 0x01 } else { 0x00 });
        }
        Value::DateTime(millis) => {
            buf.push(ElementType::DateTime as u8);
            write_cstring(buf, key);
            buf.extend_from_slice(&millis.to_le_bytes());
        }
        Value::Null => {
            buf.push(ElementType::Null as u8);
            write_cstring(buf, key);
        }
        Value::Regex(pattern, options) => {
            buf.push(ElementType::Regex as u8);
            write_cstring(buf, key);
            write_cstring(buf, pattern);
            write_cstring(buf, options);
        }
        Value::Int32(v) => {
            buf.push(ElementType::Int32 as u8);
            write_cstring(buf, key);
            buf.extend_from_slice(&v.to_le_bytes());
        }
        Value::Int64(v) => {
            buf.push(ElementType::Int64 as u8);
            write_cstring(buf, key);
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn write_cstring(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(s.as_bytes());
    buf.push(0x00);
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let len = s.len() as i32 + 1; // +1 for null terminator
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
    buf.push(0x00);
}

// ── Decoder ─────────────────────────────────────────────────────

struct Decoder<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], BsonError> {
        if self.pos + n > self.data.len() {
            return Err(BsonError::UnexpectedEof);
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, BsonError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i32(&mut self) -> Result<i32, BsonError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i64(&mut self) -> Result<i64, BsonError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_f64(&mut self) -> Result<f64, BsonError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn read_cstring(&mut self) -> Result<String, BsonError> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != 0 {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(BsonError::MissingNullTerminator);
        }
        let s = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| BsonError::InvalidUtf8)?;
        self.pos += 1; // skip null
        Ok(s.to_string())
    }

    fn read_string(&mut self) -> Result<String, BsonError> {
        let len = self.read_i32()? as usize;
        if len < 1 {
            return Err(BsonError::InvalidDocument);
        }
        let bytes = self.read_bytes(len)?;
        if bytes[len - 1] != 0 {
            return Err(BsonError::MissingNullTerminator);
        }
        let s = std::str::from_utf8(&bytes[..len - 1])
            .map_err(|_| BsonError::InvalidUtf8)?;
        Ok(s.to_string())
    }

    fn decode_document(&mut self) -> Result<Document, BsonError> {
        let size = self.read_i32()? as usize;
        if size < 5 {
            return Err(BsonError::DocumentTooSmall);
        }
        let doc_end = self.pos + size - 4; // -4 because we already read the size

        let mut doc = Document::new();
        while self.pos < doc_end - 1 {
            let type_byte = self.read_u8()?;
            if type_byte == 0 {
                break;
            }
            let key = self.read_cstring()?;
            let elem_type = ElementType::from_u8(type_byte)?;
            let value = self.decode_value(elem_type)?;
            doc.insert(key, value);
        }

        // Read null terminator if not already consumed
        if self.pos < doc_end && self.data.get(self.pos) == Some(&0) {
            self.pos += 1;
        }

        Ok(doc)
    }

    fn decode_value(&mut self, elem_type: ElementType) -> Result<Value, BsonError> {
        match elem_type {
            ElementType::Double => Ok(Value::Double(self.read_f64()?)),
            ElementType::String => Ok(Value::String(self.read_string()?)),
            ElementType::Document => Ok(Value::Document(self.decode_document()?)),
            ElementType::Array => {
                let arr_doc = self.decode_document()?;
                let values: Vec<Value> = arr_doc.entries.into_iter().map(|(_, v)| v).collect();
                Ok(Value::Array(values))
            }
            ElementType::Binary => {
                let len = self.read_i32()? as usize;
                let subtype = self.read_u8()?;
                let data = self.read_bytes(len)?.to_vec();
                Ok(Value::Binary(subtype, data))
            }
            ElementType::ObjectId => {
                let bytes = self.read_bytes(12)?;
                let mut arr = [0u8; 12];
                arr.copy_from_slice(bytes);
                Ok(Value::ObjectId(ObjectId::from_bytes(arr)))
            }
            ElementType::Boolean => {
                let b = self.read_u8()?;
                Ok(Value::Boolean(b != 0))
            }
            ElementType::DateTime => Ok(Value::DateTime(self.read_i64()?)),
            ElementType::Null => Ok(Value::Null),
            ElementType::Regex => {
                let pattern = self.read_cstring()?;
                let options = self.read_cstring()?;
                Ok(Value::Regex(pattern, options))
            }
            ElementType::Int32 => Ok(Value::Int32(self.read_i32()?)),
            ElementType::Int64 => Ok(Value::Int64(self.read_i64()?)),
        }
    }
}

/// Decode BSON bytes into a document.
pub fn decode(data: &[u8]) -> Result<Document, BsonError> {
    let mut dec = Decoder::new(data);
    dec.decode_document()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(doc: &Document) -> Document {
        let bytes = encode(doc);
        decode(&bytes).unwrap()
    }

    #[test]
    fn empty_document() {
        let doc = Document::new();
        let bytes = encode(&doc);
        // Minimum BSON document: 4 byte size + 1 byte null = 5 bytes
        assert_eq!(bytes.len(), 5);
        let decoded = decode(&bytes).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn string_field() {
        let doc = DocumentBuilder::new()
            .string("name", "Alice")
            .build();
        let decoded = roundtrip(&doc);
        match decoded.get("name") {
            Some(Value::String(s)) => assert_eq!(s, "Alice"),
            other => panic!("expected string, got {other:?}"),
        }
    }

    #[test]
    fn int32_field() {
        let doc = DocumentBuilder::new().int32("count", 42).build();
        let decoded = roundtrip(&doc);
        assert_eq!(decoded.get("count"), Some(&Value::Int32(42)));
    }

    #[test]
    fn int64_field() {
        let doc = DocumentBuilder::new().int64("big", i64::MAX).build();
        let decoded = roundtrip(&doc);
        assert_eq!(decoded.get("big"), Some(&Value::Int64(i64::MAX)));
    }

    #[test]
    fn double_field() {
        let doc = DocumentBuilder::new().double("pi", std::f64::consts::PI).build();
        let decoded = roundtrip(&doc);
        match decoded.get("pi") {
            Some(Value::Double(v)) => assert!((v - std::f64::consts::PI).abs() < f64::EPSILON),
            other => panic!("expected double, got {other:?}"),
        }
    }

    #[test]
    fn boolean_field() {
        let doc = DocumentBuilder::new()
            .boolean("active", true)
            .boolean("deleted", false)
            .build();
        let decoded = roundtrip(&doc);
        assert_eq!(decoded.get("active"), Some(&Value::Boolean(true)));
        assert_eq!(decoded.get("deleted"), Some(&Value::Boolean(false)));
    }

    #[test]
    fn null_field() {
        let doc = DocumentBuilder::new().null("nothing").build();
        let decoded = roundtrip(&doc);
        assert_eq!(decoded.get("nothing"), Some(&Value::Null));
    }

    #[test]
    fn datetime_field() {
        let doc = DocumentBuilder::new().datetime("created", 1709900000000).build();
        let decoded = roundtrip(&doc);
        assert_eq!(decoded.get("created"), Some(&Value::DateTime(1709900000000)));
    }

    #[test]
    fn array_field() {
        let doc = DocumentBuilder::new()
            .array("tags", vec![
                Value::String("rust".into()),
                Value::String("bson".into()),
            ])
            .build();
        let decoded = roundtrip(&doc);
        match decoded.get("tags") {
            Some(Value::Array(arr)) => {
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0], Value::String("rust".into()));
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn nested_document() {
        let inner = DocumentBuilder::new()
            .string("city", "Sarasota")
            .int32("zip", 34236)
            .build();
        let doc = DocumentBuilder::new()
            .string("name", "HQ")
            .document("address", inner)
            .build();
        let decoded = roundtrip(&doc);
        match decoded.get("address") {
            Some(Value::Document(d)) => {
                assert_eq!(d.get("city"), Some(&Value::String("Sarasota".into())));
                assert_eq!(d.get("zip"), Some(&Value::Int32(34236)));
            }
            other => panic!("expected document, got {other:?}"),
        }
    }

    #[test]
    fn objectid_generation() {
        let id1 = ObjectId::new();
        let id2 = ObjectId::new();
        assert_ne!(id1, id2);
        assert_eq!(id1.to_hex().len(), 24);
    }

    #[test]
    fn objectid_hex_roundtrip() {
        let hex = "507f1f77bcf86cd799439011";
        let oid = ObjectId::from_hex(hex).unwrap();
        assert_eq!(oid.to_hex(), hex);
    }

    #[test]
    fn objectid_field() {
        let oid = ObjectId::from_hex("507f1f77bcf86cd799439011").unwrap();
        let doc = DocumentBuilder::new().object_id("_id", oid.clone()).build();
        let decoded = roundtrip(&doc);
        match decoded.get("_id") {
            Some(Value::ObjectId(id)) => assert_eq!(id.to_hex(), "507f1f77bcf86cd799439011"),
            other => panic!("expected ObjectId, got {other:?}"),
        }
    }

    #[test]
    fn binary_field() {
        let doc = DocumentBuilder::new()
            .binary("data", 0x00, vec![0xDE, 0xAD, 0xBE, 0xEF])
            .build();
        let decoded = roundtrip(&doc);
        match decoded.get("data") {
            Some(Value::Binary(st, d)) => {
                assert_eq!(*st, 0x00);
                assert_eq!(d, &[0xDE, 0xAD, 0xBE, 0xEF]);
            }
            other => panic!("expected binary, got {other:?}"),
        }
    }

    #[test]
    fn regex_field() {
        let doc = DocumentBuilder::new()
            .regex("pattern", "^test.*$", "im")
            .build();
        let decoded = roundtrip(&doc);
        match decoded.get("pattern") {
            Some(Value::Regex(p, o)) => {
                assert_eq!(p, "^test.*$");
                assert_eq!(o, "im");
            }
            other => panic!("expected regex, got {other:?}"),
        }
    }

    #[test]
    fn multiple_fields() {
        let doc = DocumentBuilder::new()
            .string("name", "test")
            .int32("count", 42)
            .boolean("active", true)
            .null("deleted_at")
            .double("score", 99.5)
            .build();
        let decoded = roundtrip(&doc);
        assert_eq!(decoded.len(), 5);
        assert_eq!(decoded.get("name"), Some(&Value::String("test".into())));
        assert_eq!(decoded.get("count"), Some(&Value::Int32(42)));
    }

    #[test]
    fn document_key_order_preserved() {
        let doc = DocumentBuilder::new()
            .string("z", "last")
            .string("a", "first")
            .string("m", "middle")
            .build();
        let decoded = roundtrip(&doc);
        let keys: Vec<&str> = decoded.keys().collect();
        assert_eq!(keys, vec!["z", "a", "m"]);
    }
}
