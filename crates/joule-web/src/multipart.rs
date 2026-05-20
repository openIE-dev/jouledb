//! Multipart form data — boundary generation, part headers, content-disposition,
//! file upload modeling, streaming parser, mixed content types.
//!
//! Pure-Rust replacement for multer, busboy, form-data, etc.

use std::collections::BTreeMap;
use std::fmt;

// ── Boundary generation ───────────────────────────────────────────

/// Generate a multipart boundary string from a seed value.
/// Uses a deterministic approach suitable for testing; callers should
/// supply a random seed in production.
pub fn generate_boundary(seed: u64) -> String {
    // Mix the seed to produce a hex-like boundary
    let mut h = seed;
    let mut chars = Vec::with_capacity(24);
    for _ in 0..24 {
        h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = ((h >> 33) % 36) as u8;
        let c = if idx < 10 { b'0' + idx } else { b'a' + idx - 10 };
        chars.push(c);
    }
    let boundary = String::from_utf8(chars).unwrap_or_else(|_| "boundary000".into());
    format!("----joule{boundary}")
}

/// Validate that a boundary string is safe for use in multipart messages.
pub fn validate_boundary(boundary: &str) -> bool {
    if boundary.is_empty() || boundary.len() > 70 {
        return false;
    }
    boundary.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_'.+:=()/, ".contains(&b))
}

// ── Content-Disposition ───────────────────────────────────────────

/// Parsed Content-Disposition header.
#[derive(Debug, Clone, PartialEq)]
pub struct ContentDisposition {
    pub disposition_type: String,
    pub name: Option<String>,
    pub filename: Option<String>,
}

impl ContentDisposition {
    pub fn form_data(name: &str) -> Self {
        Self {
            disposition_type: "form-data".into(),
            name: Some(name.into()),
            filename: None,
        }
    }

    pub fn file(name: &str, filename: &str) -> Self {
        Self {
            disposition_type: "form-data".into(),
            name: Some(name.into()),
            filename: Some(filename.into()),
        }
    }

    /// Parse from header value string.
    pub fn parse(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split(';').map(|s| s.trim()).collect();
        if parts.is_empty() { return None; }
        let disposition_type = parts[0].to_lowercase();
        let mut name = None;
        let mut filename = None;
        for part in &parts[1..] {
            if let Some(val) = part.strip_prefix("name=") {
                name = Some(unquote(val));
            } else if let Some(val) = part.strip_prefix("filename=") {
                filename = Some(unquote(val));
            }
        }
        Some(Self { disposition_type, name, filename })
    }

    /// Format as header value.
    pub fn to_header(&self) -> String {
        let mut out = self.disposition_type.clone();
        if let Some(ref n) = self.name {
            out.push_str(&format!("; name=\"{n}\""));
        }
        if let Some(ref f) = self.filename {
            out.push_str(&format!("; filename=\"{f}\""));
        }
        out
    }
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

// ── Part ──────────────────────────────────────────────────────────

/// A single part in a multipart message.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl Part {
    pub fn new() -> Self {
        Self { headers: BTreeMap::new(), body: Vec::new() }
    }

    /// Create a text field part.
    pub fn text_field(name: &str, value: &str) -> Self {
        let mut headers = BTreeMap::new();
        headers.insert(
            "content-disposition".into(),
            ContentDisposition::form_data(name).to_header(),
        );
        Self { headers, body: value.as_bytes().to_vec() }
    }

    /// Create a file part.
    pub fn file_field(name: &str, filename: &str, content_type: &str, data: &[u8]) -> Self {
        let mut headers = BTreeMap::new();
        headers.insert(
            "content-disposition".into(),
            ContentDisposition::file(name, filename).to_header(),
        );
        headers.insert("content-type".into(), content_type.into());
        Self { headers, body: data.to_vec() }
    }

    /// Get the Content-Disposition, if present.
    pub fn content_disposition(&self) -> Option<ContentDisposition> {
        self.headers.get("content-disposition")
            .and_then(|v| ContentDisposition::parse(v))
    }

    /// Get the field name from Content-Disposition.
    pub fn field_name(&self) -> Option<String> {
        self.content_disposition().and_then(|cd| cd.name)
    }

    /// Get the filename from Content-Disposition.
    pub fn filename(&self) -> Option<String> {
        self.content_disposition().and_then(|cd| cd.filename)
    }

    /// Get the content type header value.
    pub fn content_type(&self) -> Option<&str> {
        self.headers.get("content-type").map(|s| s.as_str())
    }

    /// Get body as a UTF-8 string (if valid).
    pub fn body_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.body).ok()
    }

    /// Body length.
    pub fn body_len(&self) -> usize {
        self.body.len()
    }
}

impl Default for Part {
    fn default() -> Self { Self::new() }
}

// ── Encoder ───────────────────────────────────────────────────────

/// Encode parts into a multipart body.
pub struct MultipartEncoder {
    boundary: String,
    parts: Vec<Part>,
}

impl MultipartEncoder {
    pub fn new(boundary: &str) -> Self {
        Self { boundary: boundary.into(), parts: Vec::new() }
    }

    pub fn add_part(&mut self, part: Part) {
        self.parts.push(part);
    }

    /// Get the Content-Type header value (includes boundary).
    pub fn content_type(&self) -> String {
        format!("multipart/form-data; boundary={}", self.boundary)
    }

    /// Encode to bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for part in &self.parts {
            out.extend_from_slice(b"--");
            out.extend_from_slice(self.boundary.as_bytes());
            out.extend_from_slice(b"\r\n");
            for (k, v) in &part.headers {
                out.extend_from_slice(k.as_bytes());
                out.extend_from_slice(b": ");
                out.extend_from_slice(v.as_bytes());
                out.extend_from_slice(b"\r\n");
            }
            out.extend_from_slice(b"\r\n");
            out.extend_from_slice(&part.body);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(b"--");
        out.extend_from_slice(self.boundary.as_bytes());
        out.extend_from_slice(b"--\r\n");
        out
    }

    /// Encode to a String (panics if body contains non-UTF-8).
    pub fn encode_string(&self) -> String {
        String::from_utf8(self.encode()).unwrap_or_default()
    }
}

// ── Parser ────────────────────────────────────────────────────────

/// Parse state for streaming multipart parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    Preamble,
    Headers,
    Body,
    Done,
}

/// Streaming multipart parser.
pub struct MultipartParser {
    boundary: String,
    delimiter: Vec<u8>,
    end_delimiter: Vec<u8>,
    buffer: Vec<u8>,
    state: ParseState,
    current_headers: BTreeMap<String, String>,
    current_body: Vec<u8>,
    parts: Vec<Part>,
}

impl MultipartParser {
    pub fn new(boundary: &str) -> Self {
        let delimiter = format!("--{boundary}").into_bytes();
        let end_delimiter = format!("--{boundary}--").into_bytes();
        Self {
            boundary: boundary.into(),
            delimiter,
            end_delimiter,
            buffer: Vec::new(),
            state: ParseState::Preamble,
            current_headers: BTreeMap::new(),
            current_body: Vec::new(),
            parts: Vec::new(),
        }
    }

    /// Feed data into the parser. Call `finish()` when done.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.process();
    }

    fn process(&mut self) {
        loop {
            match self.state {
                ParseState::Preamble => {
                    if let Some(pos) = find_subsequence(&self.buffer, &self.delimiter) {
                        let after = pos + self.delimiter.len();
                        // Skip past \r\n after delimiter
                        let start = skip_crlf(&self.buffer, after);
                        self.buffer = self.buffer[start..].to_vec();
                        self.state = ParseState::Headers;
                    } else {
                        break;
                    }
                }
                ParseState::Headers => {
                    // Look for the blank line (\r\n\r\n) separating headers from body
                    if let Some(pos) = find_subsequence(&self.buffer, b"\r\n\r\n") {
                        let header_bytes = self.buffer[..pos].to_vec();
                        self.parse_headers(&header_bytes);
                        self.buffer = self.buffer[pos + 4..].to_vec();
                        self.state = ParseState::Body;
                    } else {
                        break;
                    }
                }
                ParseState::Body => {
                    // Look for next delimiter
                    let delim_with_crlf = [b"\r\n".as_slice(), &self.delimiter].concat();
                    if let Some(pos) = find_subsequence(&self.buffer, &delim_with_crlf) {
                        self.current_body.extend_from_slice(&self.buffer[..pos]);
                        let after = pos + delim_with_crlf.len();

                        // Check if this is the end delimiter
                        let remaining = &self.buffer[after..];
                        if remaining.starts_with(b"--") {
                            // Final part
                            self.emit_part();
                            self.state = ParseState::Done;
                            self.buffer.clear();
                        } else {
                            self.emit_part();
                            let start = skip_crlf(&self.buffer, after);
                            self.buffer = self.buffer[start..].to_vec();
                            self.state = ParseState::Headers;
                        }
                    } else {
                        // Check for end delimiter directly (no preceding data)
                        if let Some(pos) = find_subsequence(&self.buffer, &self.end_delimiter) {
                            self.current_body.extend_from_slice(&self.buffer[..pos]);
                            // Trim trailing \r\n from body
                            if self.current_body.ends_with(b"\r\n") {
                                let new_len = self.current_body.len() - 2;
                                self.current_body.truncate(new_len);
                            }
                            self.emit_part();
                            self.state = ParseState::Done;
                            self.buffer.clear();
                        } else {
                            break;
                        }
                    }
                }
                ParseState::Done => break,
            }
        }
    }

    fn parse_headers(&mut self, header_bytes: &[u8]) {
        let text = String::from_utf8_lossy(header_bytes);
        for line in text.split("\r\n") {
            if let Some(idx) = line.find(':') {
                let name = line[..idx].trim().to_lowercase();
                let value = line[idx + 1..].trim().to_string();
                self.current_headers.insert(name, value);
            }
        }
    }

    fn emit_part(&mut self) {
        let part = Part {
            headers: std::mem::take(&mut self.current_headers),
            body: std::mem::take(&mut self.current_body),
        };
        self.parts.push(part);
    }

    /// Finish parsing and return all parts.
    pub fn finish(mut self) -> Vec<Part> {
        // If there's data left in body state, emit it
        if self.state == ParseState::Body && !self.current_body.is_empty() {
            self.emit_part();
        }
        self.parts
    }

    /// Get parts parsed so far.
    pub fn parts_so_far(&self) -> &[Part] {
        &self.parts
    }

    /// The boundary string.
    pub fn boundary(&self) -> &str {
        &self.boundary
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() { return Some(0); }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn skip_crlf(data: &[u8], pos: usize) -> usize {
    if data.len() > pos + 1 && data[pos] == b'\r' && data[pos + 1] == b'\n' {
        pos + 2
    } else if data.len() > pos && data[pos] == b'\n' {
        pos + 1
    } else {
        pos
    }
}

// ── Extract boundary from Content-Type ────────────────────────────

/// Extract the boundary from a Content-Type header value.
pub fn extract_boundary(content_type: &str) -> Option<String> {
    for part in content_type.split(';') {
        let part = part.trim();
        if let Some(val) = part.strip_prefix("boundary=") {
            return Some(unquote(val));
        }
    }
    None
}

// ── Helper: find field by name ────────────────────────────────────

/// Find a part by field name in a list of parts.
pub fn find_field<'a>(parts: &'a [Part], name: &str) -> Option<&'a Part> {
    parts.iter().find(|p| p.field_name().as_deref() == Some(name))
}

/// Extract all file parts (those with a filename).
pub fn file_parts(parts: &[Part]) -> Vec<&Part> {
    parts.iter().filter(|p| p.filename().is_some()).collect()
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_generation() {
        let b1 = generate_boundary(42);
        let b2 = generate_boundary(43);
        assert_ne!(b1, b2);
        assert!(b1.starts_with("----joule"));
        assert!(validate_boundary(&b1));
    }

    #[test]
    fn boundary_validation() {
        assert!(validate_boundary("simple-boundary"));
        assert!(validate_boundary("abc123"));
        assert!(!validate_boundary("")); // empty
        assert!(!validate_boundary(&"a".repeat(71))); // too long
    }

    #[test]
    fn content_disposition_form_data() {
        let cd = ContentDisposition::form_data("username");
        assert_eq!(cd.to_header(), r#"form-data; name="username""#);
    }

    #[test]
    fn content_disposition_file() {
        let cd = ContentDisposition::file("avatar", "photo.jpg");
        assert_eq!(cd.to_header(), r#"form-data; name="avatar"; filename="photo.jpg""#);
    }

    #[test]
    fn content_disposition_parse() {
        let cd = ContentDisposition::parse(r#"form-data; name="field1""#).unwrap();
        assert_eq!(cd.disposition_type, "form-data");
        assert_eq!(cd.name.as_deref(), Some("field1"));
        assert!(cd.filename.is_none());
    }

    #[test]
    fn content_disposition_parse_file() {
        let cd = ContentDisposition::parse(r#"form-data; name="file"; filename="test.txt""#).unwrap();
        assert_eq!(cd.name.as_deref(), Some("file"));
        assert_eq!(cd.filename.as_deref(), Some("test.txt"));
    }

    #[test]
    fn part_text_field() {
        let p = Part::text_field("username", "alice");
        assert_eq!(p.field_name().as_deref(), Some("username"));
        assert_eq!(p.body_str(), Some("alice"));
        assert!(p.filename().is_none());
    }

    #[test]
    fn part_file_field() {
        let p = Part::file_field("doc", "readme.txt", "text/plain", b"Hello World");
        assert_eq!(p.field_name().as_deref(), Some("doc"));
        assert_eq!(p.filename().as_deref(), Some("readme.txt"));
        assert_eq!(p.content_type(), Some("text/plain"));
        assert_eq!(p.body_len(), 11);
    }

    #[test]
    fn encoder_basic() {
        let mut enc = MultipartEncoder::new("testboundary");
        enc.add_part(Part::text_field("name", "alice"));
        enc.add_part(Part::text_field("age", "30"));

        let ct = enc.content_type();
        assert!(ct.contains("boundary=testboundary"));

        let body = enc.encode_string();
        assert!(body.contains("--testboundary\r\n"));
        assert!(body.contains("--testboundary--\r\n"));
        assert!(body.contains("alice"));
        assert!(body.contains("30"));
    }

    #[test]
    fn encoder_with_file() {
        let mut enc = MultipartEncoder::new("myboundary");
        enc.add_part(Part::text_field("title", "My Doc"));
        enc.add_part(Part::file_field("file", "data.bin", "application/octet-stream", &[0xDE, 0xAD]));

        let bytes = enc.encode();
        assert!(!bytes.is_empty());
        // Contains the file content bytes
        assert!(bytes.windows(2).any(|w| w == [0xDE, 0xAD]));
    }

    #[test]
    fn parser_text_fields() {
        let mut enc = MultipartEncoder::new("BOUNDARY");
        enc.add_part(Part::text_field("field1", "value1"));
        enc.add_part(Part::text_field("field2", "value2"));
        let encoded = enc.encode();

        let mut parser = MultipartParser::new("BOUNDARY");
        parser.feed(&encoded);
        let parts = parser.finish();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].field_name().as_deref(), Some("field1"));
        assert_eq!(parts[0].body_str(), Some("value1"));
        assert_eq!(parts[1].field_name().as_deref(), Some("field2"));
        assert_eq!(parts[1].body_str(), Some("value2"));
    }

    #[test]
    fn parser_with_file() {
        let mut enc = MultipartEncoder::new("FILEBND");
        enc.add_part(Part::text_field("name", "test"));
        enc.add_part(Part::file_field("upload", "test.txt", "text/plain", b"file content here"));
        let encoded = enc.encode();

        let mut parser = MultipartParser::new("FILEBND");
        parser.feed(&encoded);
        let parts = parser.finish();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1].filename().as_deref(), Some("test.txt"));
        assert_eq!(parts[1].body_str(), Some("file content here"));
    }

    #[test]
    fn parser_chunked_feed() {
        let mut enc = MultipartEncoder::new("CHUNK");
        enc.add_part(Part::text_field("data", "hello world"));
        let encoded = enc.encode();

        let mut parser = MultipartParser::new("CHUNK");
        // Feed in small chunks
        for chunk in encoded.chunks(10) {
            parser.feed(chunk);
        }
        let parts = parser.finish();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].body_str(), Some("hello world"));
    }

    #[test]
    fn extract_boundary_basic() {
        let ct = "multipart/form-data; boundary=abc123";
        assert_eq!(extract_boundary(ct), Some("abc123".into()));
    }

    #[test]
    fn extract_boundary_quoted() {
        let ct = r#"multipart/form-data; boundary="my-boundary""#;
        assert_eq!(extract_boundary(ct), Some("my-boundary".into()));
    }

    #[test]
    fn extract_boundary_missing() {
        assert_eq!(extract_boundary("application/json"), None);
    }

    #[test]
    fn find_field_helper() {
        let parts = vec![
            Part::text_field("a", "1"),
            Part::text_field("b", "2"),
            Part::text_field("c", "3"),
        ];
        let found = find_field(&parts, "b").unwrap();
        assert_eq!(found.body_str(), Some("2"));
        assert!(find_field(&parts, "d").is_none());
    }

    #[test]
    fn file_parts_helper() {
        let parts = vec![
            Part::text_field("name", "alice"),
            Part::file_field("photo", "pic.jpg", "image/jpeg", b"jpeg data"),
            Part::text_field("bio", "hello"),
            Part::file_field("doc", "file.pdf", "application/pdf", b"pdf data"),
        ];
        let files = file_parts(&parts);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].filename().as_deref(), Some("pic.jpg"));
        assert_eq!(files[1].filename().as_deref(), Some("file.pdf"));
    }

    #[test]
    fn part_default() {
        let p = Part::default();
        assert!(p.headers.is_empty());
        assert!(p.body.is_empty());
        assert_eq!(p.body_len(), 0);
    }

    #[test]
    fn content_type_header() {
        let enc = MultipartEncoder::new("xyz");
        assert_eq!(enc.content_type(), "multipart/form-data; boundary=xyz");
    }

    #[test]
    fn parser_boundary_accessor() {
        let parser = MultipartParser::new("test");
        assert_eq!(parser.boundary(), "test");
    }

    #[test]
    fn parts_so_far() {
        let mut enc = MultipartEncoder::new("BND");
        enc.add_part(Part::text_field("x", "y"));
        let encoded = enc.encode();

        let mut parser = MultipartParser::new("BND");
        parser.feed(&encoded);
        assert!(!parser.parts_so_far().is_empty() || parser.finish().len() == 1);
    }

    #[test]
    fn unquote_handles_various() {
        assert_eq!(unquote(r#""hello""#), "hello");
        assert_eq!(unquote("bare"), "bare");
        assert_eq!(unquote(r#""""#), "");
    }

    #[test]
    fn roundtrip_multiple_parts() {
        let boundary = "ROUNDTRIP";
        let mut enc = MultipartEncoder::new(boundary);
        enc.add_part(Part::text_field("first", "alpha"));
        enc.add_part(Part::text_field("second", "beta"));
        enc.add_part(Part::file_field("file", "data.bin", "application/octet-stream", &[1, 2, 3, 4]));
        let encoded = enc.encode();

        let mut parser = MultipartParser::new(boundary);
        parser.feed(&encoded);
        let parts = parser.finish();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].body_str(), Some("alpha"));
        assert_eq!(parts[1].body_str(), Some("beta"));
        assert_eq!(parts[2].body, vec![1, 2, 3, 4]);
    }

    #[test]
    fn binary_body_preserved() {
        let binary_data: Vec<u8> = (0u8..=255).collect();
        let boundary = "BINTEST";
        let mut enc = MultipartEncoder::new(boundary);
        enc.add_part(Part::file_field("bin", "data.bin", "application/octet-stream", &binary_data));
        let encoded = enc.encode();

        let mut parser = MultipartParser::new(boundary);
        parser.feed(&encoded);
        let parts = parser.finish();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].body, binary_data);
    }

    #[test]
    fn content_disposition_roundtrip() {
        let cd = ContentDisposition::file("upload", "my file.txt");
        let header = cd.to_header();
        let parsed = ContentDisposition::parse(&header).unwrap();
        assert_eq!(parsed.name.as_deref(), Some("upload"));
        assert_eq!(parsed.filename.as_deref(), Some("my file.txt"));
    }
}
