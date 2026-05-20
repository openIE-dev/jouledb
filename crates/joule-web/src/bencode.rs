//! Bencode codec — BitTorrent's serialization format.
//!
//! Supports integers (`i42e`), strings (`4:spam`), lists (`l...e`),
//! dictionaries (`d...e` with sorted keys), info_hash computation,
//! and torrent file metadata parsing.

use std::collections::BTreeMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BencodeError {
    UnexpectedEof,
    UnexpectedChar(u8, usize),
    InvalidInteger(String),
    InvalidUtf8,
    ExpectedEnd(usize),
    UnsortedKeys,
    MissingInfoDict,
}

impl fmt::Display for BencodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::UnexpectedChar(c, pos) => write!(f, "unexpected byte 0x{c:02x} at position {pos}"),
            Self::InvalidInteger(s) => write!(f, "invalid integer: {s}"),
            Self::InvalidUtf8 => write!(f, "string is not valid UTF-8"),
            Self::ExpectedEnd(pos) => write!(f, "expected 'e' at position {pos}"),
            Self::UnsortedKeys => write!(f, "dictionary keys not sorted"),
            Self::MissingInfoDict => write!(f, "missing 'info' dictionary"),
        }
    }
}

impl std::error::Error for BencodeError {}

// ── Value ───────────────────────────────────────────────────────

/// A bencoded value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// Integer: `i42e`
    Int(i64),
    /// Byte string: `4:spam`
    Bytes(Vec<u8>),
    /// List: `l...e`
    List(Vec<Value>),
    /// Dictionary: `d...e` (keys are byte strings, sorted)
    Dict(BTreeMap<Vec<u8>, Value>),
}

impl Value {
    /// Get as i64.
    pub fn as_int(&self) -> Option<i64> {
        match self { Self::Int(n) => Some(*n), _ => None }
    }

    /// Get as UTF-8 string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Bytes(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// Get as byte slice.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self { Self::Bytes(b) => Some(b), _ => None }
    }

    /// Get as list.
    pub fn as_list(&self) -> Option<&Vec<Value>> {
        match self { Self::List(l) => Some(l), _ => None }
    }

    /// Get as dict.
    pub fn as_dict(&self) -> Option<&BTreeMap<Vec<u8>, Value>> {
        match self { Self::Dict(d) => Some(d), _ => None }
    }

    /// Look up a key by string in a dict.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_dict()?.get(key.as_bytes())
    }
}

// ── Encoder ─────────────────────────────────────────────────────

/// Encode a value to bencoded bytes.
pub fn encode(val: &Value) -> Vec<u8> {
    let mut buf = Vec::new();
    encode_into(&mut buf, val);
    buf
}

fn encode_into(buf: &mut Vec<u8>, val: &Value) {
    match val {
        Value::Int(n) => {
            buf.push(b'i');
            buf.extend_from_slice(n.to_string().as_bytes());
            buf.push(b'e');
        }
        Value::Bytes(b) => {
            buf.extend_from_slice(b.len().to_string().as_bytes());
            buf.push(b':');
            buf.extend_from_slice(b);
        }
        Value::List(items) => {
            buf.push(b'l');
            for item in items {
                encode_into(buf, item);
            }
            buf.push(b'e');
        }
        Value::Dict(entries) => {
            buf.push(b'd');
            // BTreeMap iterates in sorted order
            for (k, v) in entries {
                buf.extend_from_slice(k.len().to_string().as_bytes());
                buf.push(b':');
                buf.extend_from_slice(k);
                encode_into(buf, v);
            }
            buf.push(b'e');
        }
    }
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

    fn peek(&self) -> Result<u8, BencodeError> {
        self.data.get(self.pos).copied().ok_or(BencodeError::UnexpectedEof)
    }

    fn decode(&mut self) -> Result<Value, BencodeError> {
        match self.peek()? {
            b'i' => self.decode_int(),
            b'l' => self.decode_list(),
            b'd' => self.decode_dict(),
            b'0'..=b'9' => self.decode_bytes(),
            other => Err(BencodeError::UnexpectedChar(other, self.pos)),
        }
    }

    fn decode_int(&mut self) -> Result<Value, BencodeError> {
        self.pos += 1; // skip 'i'
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'e' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(BencodeError::UnexpectedEof);
        }
        let num_str = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| BencodeError::InvalidUtf8)?;
        // Validate: no leading zeros (except i0e)
        if num_str.len() > 1 && num_str.starts_with('0') {
            return Err(BencodeError::InvalidInteger(num_str.into()));
        }
        if num_str.len() > 2 && num_str.starts_with("-0") {
            return Err(BencodeError::InvalidInteger(num_str.into()));
        }
        if num_str == "-0" {
            return Err(BencodeError::InvalidInteger(num_str.into()));
        }
        let n: i64 = num_str.parse()
            .map_err(|_| BencodeError::InvalidInteger(num_str.into()))?;
        self.pos += 1; // skip 'e'
        Ok(Value::Int(n))
    }

    fn decode_bytes(&mut self) -> Result<Value, BencodeError> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b':' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(BencodeError::UnexpectedEof);
        }
        let len_str = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|_| BencodeError::InvalidUtf8)?;
        let len: usize = len_str.parse()
            .map_err(|_| BencodeError::InvalidInteger(len_str.into()))?;
        self.pos += 1; // skip ':'
        if self.pos + len > self.data.len() {
            return Err(BencodeError::UnexpectedEof);
        }
        let bytes = self.data[self.pos..self.pos + len].to_vec();
        self.pos += len;
        Ok(Value::Bytes(bytes))
    }

    fn decode_list(&mut self) -> Result<Value, BencodeError> {
        self.pos += 1; // skip 'l'
        let mut items = Vec::new();
        loop {
            if self.peek()? == b'e' {
                self.pos += 1;
                return Ok(Value::List(items));
            }
            items.push(self.decode()?);
        }
    }

    fn decode_dict(&mut self) -> Result<Value, BencodeError> {
        self.pos += 1; // skip 'd'
        let mut entries = BTreeMap::new();
        let mut last_key: Option<Vec<u8>> = None;
        loop {
            if self.peek()? == b'e' {
                self.pos += 1;
                return Ok(Value::Dict(entries));
            }
            let key = self.decode()?;
            let key_bytes = match key {
                Value::Bytes(b) => b,
                _ => return Err(BencodeError::InvalidInteger("dict key must be string".into())),
            };
            // Validate sorted order
            if let Some(prev) = &last_key {
                if key_bytes <= *prev {
                    return Err(BencodeError::UnsortedKeys);
                }
            }
            last_key = Some(key_bytes.clone());
            let val = self.decode()?;
            entries.insert(key_bytes, val);
        }
    }
}

/// Decode bencoded bytes into a value.
pub fn decode(data: &[u8]) -> Result<Value, BencodeError> {
    let mut dec = Decoder::new(data);
    dec.decode()
}

// ── Info hash ───────────────────────────────────────────────────

/// Compute the SHA-1 info_hash of a bencoded torrent (the "info" dictionary).
/// Returns the 20-byte hash.
///
/// Uses a minimal SHA-1 implementation (no external deps).
pub fn info_hash(torrent: &Value) -> Result<[u8; 20], BencodeError> {
    let info = torrent.get("info").ok_or(BencodeError::MissingInfoDict)?;
    let encoded = encode(info);
    Ok(sha1(&encoded))
}

// ── Torrent metadata ────────────────────────────────────────────

/// Parsed torrent metadata.
#[derive(Debug, Clone)]
pub struct TorrentMetadata {
    pub announce: Option<String>,
    pub name: String,
    pub piece_length: i64,
    pub pieces: Vec<u8>,
    pub length: Option<i64>,
    pub files: Vec<TorrentFile>,
    pub info_hash: [u8; 20],
}

/// A file entry in a multi-file torrent.
#[derive(Debug, Clone)]
pub struct TorrentFile {
    pub length: i64,
    pub path: Vec<String>,
}

/// Parse torrent metadata from a bencoded value.
pub fn parse_torrent(val: &Value) -> Result<TorrentMetadata, BencodeError> {
    let dict = val.as_dict().ok_or(BencodeError::MissingInfoDict)?;

    let announce = dict.get(b"announce".as_slice())
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let info = val.get("info").ok_or(BencodeError::MissingInfoDict)?;
    let info_dict = info.as_dict().ok_or(BencodeError::MissingInfoDict)?;

    let name = info.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let piece_length = info.get("piece length")
        .and_then(|v| v.as_int())
        .unwrap_or(0);

    let pieces = info.get("pieces")
        .and_then(|v| v.as_bytes())
        .unwrap_or(&[])
        .to_vec();

    let length = info.get("length").and_then(|v| v.as_int());

    let mut files = Vec::new();
    if let Some(file_list) = info.get("files") {
        if let Some(list) = file_list.as_list() {
            for file in list {
                let fl = file.get("length").and_then(|v| v.as_int()).unwrap_or(0);
                let path_list = file.get("path")
                    .and_then(|v| v.as_list())
                    .cloned()
                    .unwrap_or_default();
                let path: Vec<String> = path_list.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                files.push(TorrentFile { length: fl, path });
            }
        }
    }

    let hash = info_hash(val)?;

    Ok(TorrentMetadata {
        announce,
        name,
        piece_length,
        pieces,
        length,
        files,
        info_hash: hash,
    })
}

// ── Minimal SHA-1 ───────────────────────────────────────────────

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);

        for i in 0..80 {
            let (f_val, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let temp = a.rotate_left(5)
                .wrapping_add(f_val)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    result[0..4].copy_from_slice(&h0.to_be_bytes());
    result[4..8].copy_from_slice(&h1.to_be_bytes());
    result[8..12].copy_from_slice(&h2.to_be_bytes());
    result[12..16].copy_from_slice(&h3.to_be_bytes());
    result[16..20].copy_from_slice(&h4.to_be_bytes());
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(val: &Value) -> Value {
        let encoded = encode(val);
        decode(&encoded).unwrap()
    }

    #[test]
    fn integer_roundtrip() {
        for n in [0i64, 1, -1, 42, -42, 1000, i64::MAX, i64::MIN] {
            assert_eq!(roundtrip(&Value::Int(n)), Value::Int(n));
        }
    }

    #[test]
    fn integer_encoding() {
        assert_eq!(encode(&Value::Int(42)), b"i42e");
        assert_eq!(encode(&Value::Int(0)), b"i0e");
        assert_eq!(encode(&Value::Int(-7)), b"i-7e");
    }

    #[test]
    fn string_roundtrip() {
        let val = Value::Bytes(b"spam".to_vec());
        assert_eq!(roundtrip(&val), val);
        assert_eq!(encode(&val), b"4:spam");
    }

    #[test]
    fn empty_string() {
        let val = Value::Bytes(vec![]);
        assert_eq!(encode(&val), b"0:");
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn list_roundtrip() {
        let val = Value::List(vec![
            Value::Bytes(b"spam".to_vec()),
            Value::Bytes(b"eggs".to_vec()),
        ]);
        assert_eq!(roundtrip(&val), val);
        assert_eq!(encode(&val), b"l4:spam4:eggse");
    }

    #[test]
    fn dict_roundtrip() {
        let mut dict = BTreeMap::new();
        dict.insert(b"cow".to_vec(), Value::Bytes(b"moo".to_vec()));
        dict.insert(b"spam".to_vec(), Value::Bytes(b"eggs".to_vec()));
        let val = Value::Dict(dict);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn dict_keys_sorted() {
        let mut dict = BTreeMap::new();
        dict.insert(b"z".to_vec(), Value::Int(1));
        dict.insert(b"a".to_vec(), Value::Int(2));
        let encoded = encode(&Value::Dict(dict));
        // 'a' should come before 'z' in encoding
        let a_pos = encoded.windows(3).position(|w| w == b"1:a").unwrap();
        let z_pos = encoded.windows(3).position(|w| w == b"1:z").unwrap();
        assert!(a_pos < z_pos);
    }

    #[test]
    fn nested_structures() {
        let mut inner = BTreeMap::new();
        inner.insert(b"key".to_vec(), Value::Int(42));
        let val = Value::List(vec![
            Value::Dict(inner),
            Value::List(vec![Value::Int(1), Value::Int(2)]),
        ]);
        assert_eq!(roundtrip(&val), val);
    }

    #[test]
    fn leading_zero_rejected() {
        assert!(decode(b"i03e").is_err());
    }

    #[test]
    fn negative_zero_rejected() {
        assert!(decode(b"i-0e").is_err());
    }

    #[test]
    fn sha1_known_vector() {
        // SHA-1("abc") = a9993e36 4706816a ba3e2571 7850c26c 9cd0d89d
        let hash = sha1(b"abc");
        assert_eq!(
            hash,
            [0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a,
             0xba, 0x3e, 0x25, 0x71, 0x78, 0x50, 0xc2, 0x6c,
             0x9c, 0xd0, 0xd8, 0x9d]
        );
    }

    #[test]
    fn torrent_metadata() {
        let mut info = BTreeMap::new();
        info.insert(b"name".to_vec(), Value::Bytes(b"test.txt".to_vec()));
        info.insert(b"piece length".to_vec(), Value::Int(262144));
        info.insert(b"pieces".to_vec(), Value::Bytes(vec![0u8; 20]));
        info.insert(b"length".to_vec(), Value::Int(1024));

        let mut torrent = BTreeMap::new();
        torrent.insert(b"announce".to_vec(), Value::Bytes(b"http://tracker.example.com/announce".to_vec()));
        torrent.insert(b"info".to_vec(), Value::Dict(info));

        let val = Value::Dict(torrent);
        let meta = parse_torrent(&val).unwrap();
        assert_eq!(meta.name, "test.txt");
        assert_eq!(meta.piece_length, 262144);
        assert_eq!(meta.length, Some(1024));
        assert_eq!(meta.announce.as_deref(), Some("http://tracker.example.com/announce"));
        assert_eq!(meta.info_hash.len(), 20);
    }

    #[test]
    fn info_hash_deterministic() {
        let mut info = BTreeMap::new();
        info.insert(b"name".to_vec(), Value::Bytes(b"test".to_vec()));
        info.insert(b"piece length".to_vec(), Value::Int(256));
        info.insert(b"pieces".to_vec(), Value::Bytes(vec![0; 20]));
        info.insert(b"length".to_vec(), Value::Int(100));

        let mut torrent = BTreeMap::new();
        torrent.insert(b"info".to_vec(), Value::Dict(info));
        let val = Value::Dict(torrent);

        let h1 = info_hash(&val).unwrap();
        let h2 = info_hash(&val).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn binary_data() {
        let val = Value::Bytes(vec![0x00, 0xFF, 0x01, 0xFE]);
        assert_eq!(roundtrip(&val), val);
    }
}
