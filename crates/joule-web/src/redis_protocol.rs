//! RESP (Redis Serialization Protocol) codec.
//!
//! Replaces `redis-rs` with a pure-Rust RESP parser and serializer.
//! Supports RESP2 types (simple string, error, integer, bulk string, array),
//! null bulk string, nested arrays, inline commands, and RESP3 extensions
//! (map, set, double, boolean, big number).

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// RESP protocol errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RespError {
    /// Not enough data to parse a complete message.
    Incomplete,
    /// Invalid type prefix byte.
    InvalidPrefix(u8),
    /// Invalid integer.
    InvalidInteger(String),
    /// Invalid bulk string length.
    InvalidBulkLength(i64),
    /// Invalid line ending (expected \r\n).
    InvalidLineEnding,
    /// Unexpected end of data.
    UnexpectedEof,
    /// Invalid double value.
    InvalidDouble(String),
    /// Inline command parse error.
    InvalidInline(String),
}

impl fmt::Display for RespError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete => write!(f, "incomplete RESP data"),
            Self::InvalidPrefix(b) => write!(f, "invalid RESP prefix: {b:#x}"),
            Self::InvalidInteger(s) => write!(f, "invalid integer: {s}"),
            Self::InvalidBulkLength(n) => write!(f, "invalid bulk length: {n}"),
            Self::InvalidLineEnding => write!(f, "expected \\r\\n line ending"),
            Self::UnexpectedEof => write!(f, "unexpected end of data"),
            Self::InvalidDouble(s) => write!(f, "invalid double: {s}"),
            Self::InvalidInline(s) => write!(f, "invalid inline command: {s}"),
        }
    }
}

impl std::error::Error for RespError {}

// ── RESP Value ──────────────────────────────────────────────

/// A RESP protocol value.
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    /// RESP2: Simple string (+)
    SimpleString(String),
    /// RESP2: Error (-)
    Error(String),
    /// RESP2: Integer (:)
    Integer(i64),
    /// RESP2: Bulk string ($) — None = null bulk string
    BulkString(Option<Vec<u8>>),
    /// RESP2: Array (*) — None = null array
    Array(Option<Vec<RespValue>>),
    /// RESP3: Boolean (#)
    Boolean(bool),
    /// RESP3: Double (,)
    Double(f64),
    /// RESP3: Big number (()
    BigNumber(String),
    /// RESP3: Map (%)
    Map(Vec<(RespValue, RespValue)>),
    /// RESP3: Set (~)
    Set(Vec<RespValue>),
    /// RESP3: Null (_)
    Null,
    /// RESP3: Verbatim string (=)
    VerbatimString { encoding: String, data: String },
}

impl RespValue {
    /// Convenience: create a bulk string from a str.
    pub fn bulk(s: &str) -> Self {
        Self::BulkString(Some(s.as_bytes().to_vec()))
    }

    /// Convenience: create a null bulk string.
    pub fn null_bulk() -> Self {
        Self::BulkString(None)
    }

    /// Convenience: create a command array.
    pub fn command(parts: &[&str]) -> Self {
        Self::Array(Some(
            parts.iter().map(|s| Self::bulk(s)).collect(),
        ))
    }

    /// Try to extract as a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::SimpleString(s) => Some(s),
            Self::BulkString(Some(b)) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// Try to extract as an integer.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(*n),
            _ => None,
        }
    }

    /// Try to extract as an error message.
    pub fn as_error(&self) -> Option<&str> {
        match self {
            Self::Error(e) => Some(e),
            _ => None,
        }
    }

    /// Whether this is a null value.
    pub fn is_null(&self) -> bool {
        matches!(
            self,
            Self::BulkString(None) | Self::Array(None) | Self::Null
        )
    }
}

// ── Serialization ───────────────────────────────────────────

impl RespValue {
    /// Serialize to RESP wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf);
        buf
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        match self {
            Self::SimpleString(s) => {
                buf.push(b'+');
                buf.extend_from_slice(s.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            Self::Error(s) => {
                buf.push(b'-');
                buf.extend_from_slice(s.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            Self::Integer(n) => {
                buf.push(b':');
                buf.extend_from_slice(n.to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            Self::BulkString(None) => {
                buf.extend_from_slice(b"$-1\r\n");
            }
            Self::BulkString(Some(data)) => {
                buf.push(b'$');
                buf.extend_from_slice(data.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                buf.extend_from_slice(data);
                buf.extend_from_slice(b"\r\n");
            }
            Self::Array(None) => {
                buf.extend_from_slice(b"*-1\r\n");
            }
            Self::Array(Some(items)) => {
                buf.push(b'*');
                buf.extend_from_slice(items.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                for item in items {
                    item.write_to(buf);
                }
            }
            Self::Boolean(b) => {
                buf.push(b'#');
                buf.push(if *b { b't' } else { b'f' });
                buf.extend_from_slice(b"\r\n");
            }
            Self::Double(d) => {
                buf.push(b',');
                buf.extend_from_slice(d.to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            Self::BigNumber(n) => {
                buf.push(b'(');
                buf.extend_from_slice(n.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            Self::Map(entries) => {
                buf.push(b'%');
                buf.extend_from_slice(entries.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                for (key, val) in entries {
                    key.write_to(buf);
                    val.write_to(buf);
                }
            }
            Self::Set(items) => {
                buf.push(b'~');
                buf.extend_from_slice(items.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                for item in items {
                    item.write_to(buf);
                }
            }
            Self::Null => {
                buf.extend_from_slice(b"_\r\n");
            }
            Self::VerbatimString { encoding, data } => {
                let content = format!("{encoding}:{data}");
                buf.push(b'=');
                buf.extend_from_slice(content.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                buf.extend_from_slice(content.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
        }
    }
}

// ── Parsing ─────────────────────────────────────────────────

/// Parse a RESP value from bytes. Returns (value, bytes_consumed).
pub fn parse_resp(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    if data.is_empty() {
        return Err(RespError::Incomplete);
    }
    match data[0] {
        b'+' => parse_simple_string(data),
        b'-' => parse_error(data),
        b':' => parse_integer(data),
        b'$' => parse_bulk_string(data),
        b'*' => parse_array(data),
        b'#' => parse_boolean(data),
        b',' => parse_double(data),
        b'(' => parse_big_number(data),
        b'%' => parse_map(data),
        b'~' => parse_set(data),
        b'_' => parse_null(data),
        b'=' => parse_verbatim(data),
        other => Err(RespError::InvalidPrefix(other)),
    }
}

fn find_crlf(data: &[u8], start: usize) -> Option<usize> {
    data[start..]
        .windows(2)
        .position(|w| w == b"\r\n")
        .map(|p| start + p)
}

fn read_line(data: &[u8], start: usize) -> Result<(&[u8], usize), RespError> {
    let crlf = find_crlf(data, start).ok_or(RespError::Incomplete)?;
    Ok((&data[start..crlf], crlf + 2))
}

fn parse_simple_string(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, end) = read_line(data, 1)?;
    let s = String::from_utf8_lossy(line).to_string();
    Ok((RespValue::SimpleString(s), end))
}

fn parse_error(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, end) = read_line(data, 1)?;
    let s = String::from_utf8_lossy(line).to_string();
    Ok((RespValue::Error(s), end))
}

fn parse_integer(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, end) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidInteger("not utf8".into()))?;
    let n: i64 = s.parse().map_err(|_| RespError::InvalidInteger(s.to_string()))?;
    Ok((RespValue::Integer(n), end))
}

fn parse_bulk_string(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, after_line) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidInteger("not utf8".into()))?;
    let len: i64 = s.parse().map_err(|_| RespError::InvalidBulkLength(0))?;
    if len < 0 {
        return Ok((RespValue::BulkString(None), after_line));
    }
    let len = len as usize;
    if after_line + len + 2 > data.len() {
        return Err(RespError::Incomplete);
    }
    let payload = data[after_line..after_line + len].to_vec();
    // Check \r\n after payload
    if data[after_line + len] != b'\r' || data[after_line + len + 1] != b'\n' {
        return Err(RespError::InvalidLineEnding);
    }
    Ok((RespValue::BulkString(Some(payload)), after_line + len + 2))
}

fn parse_array(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, mut offset) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidInteger("not utf8".into()))?;
    let count: i64 = s.parse().map_err(|_| RespError::InvalidInteger(s.to_string()))?;
    if count < 0 {
        return Ok((RespValue::Array(None), offset));
    }
    let count = count as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        let (val, consumed) = parse_resp(&data[offset..])?;
        items.push(val);
        offset += consumed;
    }
    Ok((RespValue::Array(Some(items)), offset))
}

fn parse_boolean(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, end) = read_line(data, 1)?;
    let val = match line {
        [b't'] => true,
        [b'f'] => false,
        _ => return Err(RespError::InvalidPrefix(b'#')),
    };
    Ok((RespValue::Boolean(val), end))
}

fn parse_double(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, end) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidDouble("not utf8".into()))?;
    let d: f64 = s.parse().map_err(|_| RespError::InvalidDouble(s.to_string()))?;
    Ok((RespValue::Double(d), end))
}

fn parse_big_number(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, end) = read_line(data, 1)?;
    let s = String::from_utf8_lossy(line).to_string();
    Ok((RespValue::BigNumber(s), end))
}

fn parse_map(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, mut offset) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidInteger("not utf8".into()))?;
    let count: usize = s.parse().map_err(|_| RespError::InvalidInteger(s.to_string()))?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let (key, kc) = parse_resp(&data[offset..])?;
        offset += kc;
        let (val, vc) = parse_resp(&data[offset..])?;
        offset += vc;
        entries.push((key, val));
    }
    Ok((RespValue::Map(entries), offset))
}

fn parse_set(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, mut offset) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidInteger("not utf8".into()))?;
    let count: usize = s.parse().map_err(|_| RespError::InvalidInteger(s.to_string()))?;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        let (val, consumed) = parse_resp(&data[offset..])?;
        items.push(val);
        offset += consumed;
    }
    Ok((RespValue::Set(items), offset))
}

fn parse_null(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (_, end) = read_line(data, 1)?;
    Ok((RespValue::Null, end))
}

fn parse_verbatim(data: &[u8]) -> Result<(RespValue, usize), RespError> {
    let (line, after_line) = read_line(data, 1)?;
    let s = std::str::from_utf8(line).map_err(|_| RespError::InvalidInteger("not utf8".into()))?;
    let len: usize = s.parse().map_err(|_| RespError::InvalidBulkLength(0))?;
    if after_line + len + 2 > data.len() {
        return Err(RespError::Incomplete);
    }
    let content = std::str::from_utf8(&data[after_line..after_line + len])
        .map_err(|_| RespError::InvalidDouble("not utf8".into()))?;
    // Format is "enc:data" where enc is 3 chars
    if content.len() < 4 || content.as_bytes()[3] != b':' {
        return Err(RespError::InvalidDouble("bad verbatim format".into()));
    }
    let encoding = content[..3].to_string();
    let text = content[4..].to_string();
    Ok((RespValue::VerbatimString { encoding, data: text }, after_line + len + 2))
}

// ── Inline Command Parser ───────────────────────────────────

/// Parse an inline command (space-separated, no type prefix).
pub fn parse_inline(line: &str) -> Result<Vec<String>, RespError> {
    let line = line.trim();
    if line.is_empty() {
        return Err(RespError::InvalidInline("empty command".into()));
    }
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                in_quotes = false;
            } else if ch == '\\' {
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                }
            } else {
                current.push(ch);
            }
        } else if ch == '"' {
            in_quotes = true;
        } else if ch == ' ' {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    Ok(parts)
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_string_roundtrip() {
        let val = RespValue::SimpleString("OK".into());
        let bytes = val.to_bytes();
        assert_eq!(bytes, b"+OK\r\n");
        let (parsed, consumed) = parse_resp(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert_eq!(parsed, val);
    }

    #[test]
    fn error_roundtrip() {
        let val = RespValue::Error("ERR unknown command".into());
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed.as_error(), Some("ERR unknown command"));
    }

    #[test]
    fn integer_roundtrip() {
        let val = RespValue::Integer(-42);
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed.as_integer(), Some(-42));
    }

    #[test]
    fn bulk_string_roundtrip() {
        let val = RespValue::bulk("hello world");
        let bytes = val.to_bytes();
        assert_eq!(bytes, b"$11\r\nhello world\r\n");
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed.as_str(), Some("hello world"));
    }

    #[test]
    fn null_bulk_string() {
        let val = RespValue::null_bulk();
        let bytes = val.to_bytes();
        assert_eq!(bytes, b"$-1\r\n");
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert!(parsed.is_null());
    }

    #[test]
    fn array_roundtrip() {
        let val = RespValue::command(&["SET", "key", "value"]);
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        if let RespValue::Array(Some(items)) = parsed {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0].as_str(), Some("SET"));
            assert_eq!(items[1].as_str(), Some("key"));
            assert_eq!(items[2].as_str(), Some("value"));
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn null_array() {
        let val = RespValue::Array(None);
        let bytes = val.to_bytes();
        assert_eq!(bytes, b"*-1\r\n");
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert!(parsed.is_null());
    }

    #[test]
    fn nested_array() {
        let val = RespValue::Array(Some(vec![
            RespValue::Integer(1),
            RespValue::Array(Some(vec![
                RespValue::Integer(2),
                RespValue::Integer(3),
            ])),
        ]));
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed, val);
    }

    #[test]
    fn resp3_boolean() {
        let val = RespValue::Boolean(true);
        let bytes = val.to_bytes();
        assert_eq!(bytes, b"#t\r\n");
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed, RespValue::Boolean(true));
    }

    #[test]
    fn resp3_double() {
        let val = RespValue::Double(3.14);
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        if let RespValue::Double(d) = parsed {
            assert!((d - 3.14).abs() < 1e-10);
        } else {
            panic!("expected double");
        }
    }

    #[test]
    fn resp3_map() {
        let val = RespValue::Map(vec![
            (RespValue::bulk("key1"), RespValue::Integer(1)),
            (RespValue::bulk("key2"), RespValue::Integer(2)),
        ]);
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed, val);
    }

    #[test]
    fn resp3_set() {
        let val = RespValue::Set(vec![
            RespValue::Integer(1),
            RespValue::Integer(2),
            RespValue::Integer(3),
        ]);
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed, val);
    }

    #[test]
    fn resp3_big_number() {
        let val = RespValue::BigNumber("123456789012345678901234567890".into());
        let bytes = val.to_bytes();
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert_eq!(parsed, val);
    }

    #[test]
    fn resp3_null() {
        let val = RespValue::Null;
        let bytes = val.to_bytes();
        assert_eq!(bytes, b"_\r\n");
        let (parsed, _) = parse_resp(&bytes).unwrap();
        assert!(parsed.is_null());
    }

    #[test]
    fn inline_command_parsing() {
        let parts = parse_inline("SET key value").unwrap();
        assert_eq!(parts, vec!["SET", "key", "value"]);

        let parts = parse_inline("GET \"hello world\"").unwrap();
        assert_eq!(parts, vec!["GET", "hello world"]);
    }

    #[test]
    fn incomplete_data() {
        assert_eq!(parse_resp(b"+OK").unwrap_err(), RespError::Incomplete);
        assert_eq!(parse_resp(b"$5\r\nhel").unwrap_err(), RespError::Incomplete);
    }
}
