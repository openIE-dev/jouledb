//! HTTP/1.1 parser.
//!
//! Replaces `httparse` / `hyper` parser with a pure-Rust HTTP/1.1 codec.
//! Supports request line parsing, response status line, header parsing
//! (including folded headers), chunked transfer encoding, content-length
//! body reading, trailer headers, and keep-alive detection.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// HTTP parse errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpParseError {
    /// Incomplete data — need more bytes.
    Incomplete,
    /// Invalid request line.
    InvalidRequestLine(String),
    /// Invalid status line.
    InvalidStatusLine(String),
    /// Invalid header.
    InvalidHeader(String),
    /// Invalid chunk size.
    InvalidChunkSize(String),
    /// Content-Length mismatch.
    ContentLengthMismatch { expected: usize, got: usize },
    /// Multiple Content-Length headers with different values.
    ConflictingContentLength,
    /// Invalid method.
    InvalidMethod(String),
    /// Header value not valid UTF-8.
    InvalidHeaderValue,
    /// Body too large.
    BodyTooLarge(usize),
}

impl fmt::Display for HttpParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete => write!(f, "incomplete HTTP data"),
            Self::InvalidRequestLine(s) => write!(f, "invalid request line: {s}"),
            Self::InvalidStatusLine(s) => write!(f, "invalid status line: {s}"),
            Self::InvalidHeader(s) => write!(f, "invalid header: {s}"),
            Self::InvalidChunkSize(s) => write!(f, "invalid chunk size: {s}"),
            Self::ContentLengthMismatch { expected, got } => {
                write!(f, "content-length mismatch: expected {expected}, got {got}")
            }
            Self::ConflictingContentLength => write!(f, "conflicting content-length values"),
            Self::InvalidMethod(m) => write!(f, "invalid method: {m}"),
            Self::InvalidHeaderValue => write!(f, "invalid header value encoding"),
            Self::BodyTooLarge(n) => write!(f, "body too large: {n} bytes"),
        }
    }
}

impl std::error::Error for HttpParseError {}

// ── Method ──────────────────────────────────────────────────

/// HTTP method.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Method {
    Get,
    Head,
    Post,
    Put,
    Delete,
    Connect,
    Options,
    Trace,
    Patch,
    Other(String),
}

impl Method {
    pub fn from_str(s: &str) -> Self {
        match s {
            "GET" => Self::Get,
            "HEAD" => Self::Head,
            "POST" => Self::Post,
            "PUT" => Self::Put,
            "DELETE" => Self::Delete,
            "CONNECT" => Self::Connect,
            "OPTIONS" => Self::Options,
            "TRACE" => Self::Trace,
            "PATCH" => Self::Patch,
            other => Self::Other(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Get => "GET",
            Self::Head => "HEAD",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Delete => "DELETE",
            Self::Connect => "CONNECT",
            Self::Options => "OPTIONS",
            Self::Trace => "TRACE",
            Self::Patch => "PATCH",
            Self::Other(s) => s,
        }
    }
}

// ── HTTP Version ────────────────────────────────────────────

/// HTTP version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpVersion {
    Http10,
    Http11,
}

impl HttpVersion {
    pub fn from_str(s: &str) -> Result<Self, HttpParseError> {
        match s {
            "HTTP/1.0" => Ok(Self::Http10),
            "HTTP/1.1" => Ok(Self::Http11),
            _ => Err(HttpParseError::InvalidRequestLine(format!("unknown version: {s}"))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http10 => "HTTP/1.0",
            Self::Http11 => "HTTP/1.1",
        }
    }
}

// ── Headers ─────────────────────────────────────────────────

/// Case-insensitive HTTP headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Headers {
    entries: Vec<(String, String)>,
}

impl Headers {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a header (appends; does not replace).
    pub fn add(&mut self, name: &str, value: &str) {
        self.entries.push((name.to_string(), value.to_string()));
    }

    /// Get the first value for a header name (case-insensitive).
    pub fn get(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .find(|(n, _)| n.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }

    /// Get all values for a header name.
    pub fn get_all(&self, name: &str) -> Vec<&str> {
        let lower = name.to_lowercase();
        self.entries
            .iter()
            .filter(|(n, _)| n.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    /// Number of header entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no headers.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all (name, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(n, v)| (n.as_str(), v.as_str()))
    }

    /// Serialize headers to HTTP wire format.
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        for (name, value) in &self.entries {
            out.push_str(name);
            out.push_str(": ");
            out.push_str(value);
            out.push_str("\r\n");
        }
        out
    }
}

// ── Request ─────────────────────────────────────────────────

/// A parsed HTTP request.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: Method,
    pub uri: String,
    pub version: HttpVersion,
    pub headers: Headers,
    pub body: Vec<u8>,
}

impl HttpRequest {
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = format!(
            "{} {} {}\r\n{}\r\n",
            self.method.as_str(),
            self.uri,
            self.version.as_str(),
            self.headers.to_string(),
        );
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

// ── Response ────────────────────────────────────────────────

/// A parsed HTTP response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub version: HttpVersion,
    pub status_code: u16,
    pub reason: String,
    pub headers: Headers,
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let out = format!(
            "{} {} {}\r\n{}\r\n",
            self.version.as_str(),
            self.status_code,
            self.reason,
            self.headers.to_string(),
        );
        let mut bytes = out.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}

// ── Parser ──────────────────────────────────────────────────

/// Find the end of the header block (\r\n\r\n).
fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|pos| pos + 4)
}

/// Parse headers from a header block string (lines after request/status line).
fn parse_headers(header_text: &str) -> Result<Headers, HttpParseError> {
    let mut headers = Headers::new();
    let mut lines = header_text.lines().peekable();

    while let Some(line) = lines.next() {
        if line.is_empty() {
            continue;
        }
        let colon = line
            .find(':')
            .ok_or_else(|| HttpParseError::InvalidHeader(line.to_string()))?;
        let name = line[..colon].trim();
        let mut value = line[colon + 1..].trim().to_string();

        // Handle folded headers (continuation lines starting with space/tab)
        while let Some(next) = lines.peek() {
            if next.starts_with(' ') || next.starts_with('\t') {
                value.push(' ');
                value.push_str(next.trim());
                lines.next();
            } else {
                break;
            }
        }
        headers.add(name, &value);
    }
    Ok(headers)
}

/// Parse an HTTP request from raw bytes.
pub fn parse_request(data: &[u8]) -> Result<(HttpRequest, usize), HttpParseError> {
    let header_end = find_header_end(data).ok_or(HttpParseError::Incomplete)?;
    let header_block = std::str::from_utf8(&data[..header_end])
        .map_err(|_| HttpParseError::InvalidHeaderValue)?;

    let first_line_end = header_block
        .find("\r\n")
        .ok_or(HttpParseError::Incomplete)?;
    let request_line = &header_block[..first_line_end];

    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() != 3 {
        return Err(HttpParseError::InvalidRequestLine(request_line.to_string()));
    }
    let method = Method::from_str(parts[0]);
    let uri = parts[1].to_string();
    let version = HttpVersion::from_str(parts[2])?;

    let header_text = &header_block[first_line_end + 2..header_end - 4];
    let headers = parse_headers(header_text)?;

    let (body, total) = read_body(data, header_end, &headers)?;

    Ok((
        HttpRequest { method, uri, version, headers, body },
        total,
    ))
}

/// Parse an HTTP response from raw bytes.
pub fn parse_response(data: &[u8]) -> Result<(HttpResponse, usize), HttpParseError> {
    let header_end = find_header_end(data).ok_or(HttpParseError::Incomplete)?;
    let header_block = std::str::from_utf8(&data[..header_end])
        .map_err(|_| HttpParseError::InvalidHeaderValue)?;

    let first_line_end = header_block
        .find("\r\n")
        .ok_or(HttpParseError::Incomplete)?;
    let status_line = &header_block[..first_line_end];

    let parts: Vec<&str> = status_line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err(HttpParseError::InvalidStatusLine(status_line.to_string()));
    }
    let version = HttpVersion::from_str(parts[0])
        .map_err(|_| HttpParseError::InvalidStatusLine(status_line.to_string()))?;
    let status_code: u16 = parts[1]
        .parse()
        .map_err(|_| HttpParseError::InvalidStatusLine(status_line.to_string()))?;
    let reason = if parts.len() == 3 { parts[2].to_string() } else { String::new() };

    let header_text = &header_block[first_line_end + 2..header_end - 4];
    let headers = parse_headers(header_text)?;

    let (body, total) = read_body(data, header_end, &headers)?;

    Ok((
        HttpResponse { version, status_code, reason, headers, body },
        total,
    ))
}

/// Read the body based on Transfer-Encoding or Content-Length.
fn read_body(
    data: &[u8],
    body_start: usize,
    headers: &Headers,
) -> Result<(Vec<u8>, usize), HttpParseError> {
    // Check for chunked transfer encoding
    if let Some(te) = headers.get("Transfer-Encoding") {
        if te.to_lowercase().contains("chunked") {
            return decode_chunked(&data[body_start..]).map(|(body, consumed)| {
                (body, body_start + consumed)
            });
        }
    }

    // Check Content-Length
    if let Some(cl) = headers.get("Content-Length") {
        let len: usize = cl
            .trim()
            .parse()
            .map_err(|_| HttpParseError::InvalidHeader(format!("bad content-length: {cl}")))?;
        if body_start + len > data.len() {
            return Err(HttpParseError::Incomplete);
        }
        let body = data[body_start..body_start + len].to_vec();
        return Ok((body, body_start + len));
    }

    // No body indication — return empty body
    Ok((Vec::new(), body_start))
}

// ── Chunked Transfer Encoding ───────────────────────────────

/// Decode chunked transfer encoding. Returns (body, bytes_consumed).
pub fn decode_chunked(data: &[u8]) -> Result<(Vec<u8>, usize), HttpParseError> {
    let mut body = Vec::new();
    let mut pos = 0;

    loop {
        // Find chunk size line
        let line_end = find_crlf(data, pos).ok_or(HttpParseError::Incomplete)?;
        let size_str = std::str::from_utf8(&data[pos..line_end])
            .map_err(|_| HttpParseError::InvalidChunkSize("not utf8".into()))?
            .trim();
        // Chunk size may have extensions after semicolon
        let size_hex = size_str.split(';').next().unwrap_or("").trim();
        let chunk_size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| HttpParseError::InvalidChunkSize(size_hex.to_string()))?;

        pos = line_end + 2; // skip \r\n

        if chunk_size == 0 {
            // Read optional trailer headers (skip until \r\n)
            if pos + 2 <= data.len() && data[pos] == b'\r' && data[pos + 1] == b'\n' {
                pos += 2;
            }
            break;
        }

        if pos + chunk_size + 2 > data.len() {
            return Err(HttpParseError::Incomplete);
        }
        body.extend_from_slice(&data[pos..pos + chunk_size]);
        pos += chunk_size + 2; // skip chunk data + \r\n
    }

    Ok((body, pos))
}

/// Encode data as chunked transfer encoding.
pub fn encode_chunked(data: &[u8], chunk_size: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in data.chunks(chunk_size) {
        out.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
        out.extend_from_slice(chunk);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n\r\n");
    out
}

fn find_crlf(data: &[u8], start: usize) -> Option<usize> {
    data[start..]
        .windows(2)
        .position(|w| w == b"\r\n")
        .map(|p| start + p)
}

// ── Keep-Alive Detection ────────────────────────────────────

/// Determine if the connection should be kept alive.
pub fn is_keep_alive(version: HttpVersion, headers: &Headers) -> bool {
    if let Some(conn) = headers.get("Connection") {
        let lower = conn.to_lowercase();
        if lower.contains("close") {
            return false;
        }
        if lower.contains("keep-alive") {
            return true;
        }
    }
    // HTTP/1.1 defaults to keep-alive
    version == HttpVersion::Http11
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_get_request() {
        let raw = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n";
        let (req, consumed) = parse_request(raw).unwrap();
        assert_eq!(req.method, Method::Get);
        assert_eq!(req.uri, "/index.html");
        assert_eq!(req.version, HttpVersion::Http11);
        assert_eq!(req.headers.get("Host"), Some("example.com"));
        assert!(req.body.is_empty());
        assert_eq!(consumed, raw.len());
    }

    #[test]
    fn parse_post_with_content_length() {
        let body = b"name=test&value=42";
        let raw = format!(
            "POST /api HTTP/1.1\r\nContent-Length: {}\r\nContent-Type: application/x-www-form-urlencoded\r\n\r\n{}",
            body.len(),
            std::str::from_utf8(body).unwrap()
        );
        let (req, _) = parse_request(raw.as_bytes()).unwrap();
        assert_eq!(req.method, Method::Post);
        assert_eq!(req.body, body);
    }

    #[test]
    fn parse_response_200() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let (resp, _) = parse_response(raw).unwrap();
        assert_eq!(resp.status_code, 200);
        assert_eq!(resp.reason, "OK");
        assert_eq!(resp.body, b"hello");
    }

    #[test]
    fn parse_response_404() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        let (resp, _) = parse_response(raw).unwrap();
        assert_eq!(resp.status_code, 404);
        assert_eq!(resp.reason, "Not Found");
    }

    #[test]
    fn chunked_encoding_decode() {
        let chunked = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let (body, _) = decode_chunked(chunked).unwrap();
        assert_eq!(body, b"hello world");
    }

    #[test]
    fn chunked_encoding_encode() {
        let data = b"hello world";
        let encoded = encode_chunked(data, 5);
        let (decoded, _) = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn chunked_response() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n7\r\npedia i\r\n0\r\n\r\n";
        let (resp, _) = parse_response(raw).unwrap();
        assert_eq!(resp.body, b"Wikipedia i");
    }

    #[test]
    fn folded_header() {
        let raw = b"GET / HTTP/1.1\r\nX-Long: first\r\n second\r\nHost: test\r\n\r\n";
        let (req, _) = parse_request(raw).unwrap();
        assert_eq!(req.headers.get("X-Long"), Some("first second"));
    }

    #[test]
    fn keep_alive_detection() {
        let mut headers = Headers::new();
        assert!(is_keep_alive(HttpVersion::Http11, &headers));
        assert!(!is_keep_alive(HttpVersion::Http10, &headers));

        headers.add("Connection", "keep-alive");
        assert!(is_keep_alive(HttpVersion::Http10, &headers));

        let mut close_headers = Headers::new();
        close_headers.add("Connection", "close");
        assert!(!is_keep_alive(HttpVersion::Http11, &close_headers));
    }

    #[test]
    fn request_roundtrip() {
        let mut headers = Headers::new();
        headers.add("Host", "example.com");
        headers.add("Content-Length", "3");
        let req = HttpRequest {
            method: Method::Post,
            uri: "/test".to_string(),
            version: HttpVersion::Http11,
            headers,
            body: b"abc".to_vec(),
        };
        let bytes = req.to_bytes();
        let (parsed, _) = parse_request(&bytes).unwrap();
        assert_eq!(parsed.method, Method::Post);
        assert_eq!(parsed.uri, "/test");
        assert_eq!(parsed.body, b"abc");
    }

    #[test]
    fn multiple_headers_same_name() {
        let raw = b"GET / HTTP/1.1\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2\r\n\r\n";
        let (req, _) = parse_request(raw).unwrap();
        let cookies = req.headers.get_all("Set-Cookie");
        assert_eq!(cookies.len(), 2);
        assert_eq!(cookies[0], "a=1");
        assert_eq!(cookies[1], "b=2");
    }

    #[test]
    fn incomplete_request() {
        let raw = b"GET / HTTP/1.1\r\nHost: test";
        assert_eq!(parse_request(raw).unwrap_err(), HttpParseError::Incomplete);
    }

    #[test]
    fn method_variants() {
        assert_eq!(Method::from_str("GET"), Method::Get);
        assert_eq!(Method::from_str("PATCH"), Method::Patch);
        assert_eq!(Method::from_str("CUSTOM"), Method::Other("CUSTOM".into()));
        assert_eq!(Method::Get.as_str(), "GET");
    }
}
