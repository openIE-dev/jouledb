//! gRPC-Web protocol — message framing, base64 encoding, trailers, status codes.
//!
//! Replaces `grpc-web`, `@improbable-eng/grpc-web`, and `grpcwebproxy` with
//! pure Rust.  Length-prefixed message framing, base64 encoding for text mode,
//! trailer parsing, gRPC status codes, metadata, unary/server-streaming
//! serialization.

use std::collections::HashMap;
use std::fmt;

// ── gRPC status codes ──────────────────────────────────────────

/// gRPC status code (per spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatusCode {
    Ok,
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
}

impl StatusCode {
    pub fn code(&self) -> u32 {
        match self {
            Self::Ok => 0,
            Self::Cancelled => 1,
            Self::Unknown => 2,
            Self::InvalidArgument => 3,
            Self::DeadlineExceeded => 4,
            Self::NotFound => 5,
            Self::AlreadyExists => 6,
            Self::PermissionDenied => 7,
            Self::ResourceExhausted => 8,
            Self::FailedPrecondition => 9,
            Self::Aborted => 10,
            Self::OutOfRange => 11,
            Self::Unimplemented => 12,
            Self::Internal => 13,
            Self::Unavailable => 14,
            Self::DataLoss => 15,
            Self::Unauthenticated => 16,
        }
    }

    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::Ok,
            1 => Self::Cancelled,
            2 => Self::Unknown,
            3 => Self::InvalidArgument,
            4 => Self::DeadlineExceeded,
            5 => Self::NotFound,
            6 => Self::AlreadyExists,
            7 => Self::PermissionDenied,
            8 => Self::ResourceExhausted,
            9 => Self::FailedPrecondition,
            10 => Self::Aborted,
            11 => Self::OutOfRange,
            12 => Self::Unimplemented,
            13 => Self::Internal,
            14 => Self::Unavailable,
            15 => Self::DataLoss,
            16 => Self::Unauthenticated,
            _ => Self::Unknown,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Ok => "OK",
            Self::Cancelled => "CANCELLED",
            Self::Unknown => "UNKNOWN",
            Self::InvalidArgument => "INVALID_ARGUMENT",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::NotFound => "NOT_FOUND",
            Self::AlreadyExists => "ALREADY_EXISTS",
            Self::PermissionDenied => "PERMISSION_DENIED",
            Self::ResourceExhausted => "RESOURCE_EXHAUSTED",
            Self::FailedPrecondition => "FAILED_PRECONDITION",
            Self::Aborted => "ABORTED",
            Self::OutOfRange => "OUT_OF_RANGE",
            Self::Unimplemented => "UNIMPLEMENTED",
            Self::Internal => "INTERNAL",
            Self::Unavailable => "UNAVAILABLE",
            Self::DataLoss => "DATA_LOSS",
            Self::Unauthenticated => "UNAUTHENTICATED",
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.as_str(), self.code())
    }
}

// ── Metadata ───────────────────────────────────────────────────

/// gRPC metadata (headers/trailers).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<(String, String)>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: &str, value: &str) {
        self.entries
            .push((key.to_ascii_lowercase(), value.to_string()));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        let lower = key.to_ascii_lowercase();
        self.entries
            .iter()
            .find(|(k, _)| *k == lower)
            .map(|(_, v)| v.as_str())
    }

    pub fn get_all(&self, key: &str) -> Vec<&str> {
        let lower = key.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|(k, _)| *k == lower)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Is this a binary metadata key? (ends with `-bin`)
    pub fn is_binary_key(key: &str) -> bool {
        key.to_ascii_lowercase().ends_with("-bin")
    }
}

// ── gRPC-Web framing ───────────────────────────────────────────

/// gRPC-Web frame type flag.
const DATA_FRAME: u8 = 0x00;
const TRAILER_FRAME: u8 = 0x80;

/// A gRPC-Web frame (data or trailer).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrpcFrame {
    Data(Vec<u8>),
    Trailer(Vec<u8>),
}

/// Encode a data message into a gRPC-Web length-prefixed frame.
///
/// Format: [1 byte flag][4 bytes length][payload]
pub fn encode_data_frame(payload: &[u8]) -> Vec<u8> {
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(5 + payload.len());
    buf.push(DATA_FRAME);
    buf.push(((len >> 24) & 0xFF) as u8);
    buf.push(((len >> 16) & 0xFF) as u8);
    buf.push(((len >> 8) & 0xFF) as u8);
    buf.push((len & 0xFF) as u8);
    buf.extend_from_slice(payload);
    buf
}

/// Encode trailer metadata into a gRPC-Web trailer frame.
pub fn encode_trailer_frame(trailers: &Metadata) -> Vec<u8> {
    let mut trailer_text = String::new();
    for (k, v) in trailers.iter() {
        trailer_text.push_str(k);
        trailer_text.push_str(": ");
        trailer_text.push_str(v);
        trailer_text.push_str("\r\n");
    }
    let payload = trailer_text.as_bytes();
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(5 + payload.len());
    buf.push(TRAILER_FRAME);
    buf.push(((len >> 24) & 0xFF) as u8);
    buf.push(((len >> 16) & 0xFF) as u8);
    buf.push(((len >> 8) & 0xFF) as u8);
    buf.push((len & 0xFF) as u8);
    buf.extend_from_slice(payload);
    buf
}

/// Decode a single gRPC-Web frame from a buffer.
/// Returns (frame, bytes_consumed).
pub fn decode_frame(buf: &[u8]) -> Option<(GrpcFrame, usize)> {
    if buf.len() < 5 {
        return None;
    }
    let flag = buf[0];
    let len = ((buf[1] as u32) << 24)
        | ((buf[2] as u32) << 16)
        | ((buf[3] as u32) << 8)
        | (buf[4] as u32);
    let total = 5 + len as usize;
    if buf.len() < total {
        return None;
    }
    let payload = buf[5..total].to_vec();
    let frame = if flag & TRAILER_FRAME != 0 {
        GrpcFrame::Trailer(payload)
    } else {
        GrpcFrame::Data(payload)
    };
    Some((frame, total))
}

/// Decode all frames from a buffer.
pub fn decode_all_frames(buf: &[u8]) -> Vec<GrpcFrame> {
    let mut frames = Vec::new();
    let mut offset = 0;
    while offset < buf.len() {
        if let Some((frame, consumed)) = decode_frame(&buf[offset..]) {
            frames.push(frame);
            offset += consumed;
        } else {
            break;
        }
    }
    frames
}

// ── Base64 (for gRPC-Web text mode) ────────────────────────────

const B64_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Base64 encode.
pub fn base64_encode(data: &[u8]) -> String {
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    let chunks = data.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };

        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);

        if chunk.len() > 1 {
            result.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn b64_decode_char(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Base64 decode.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let bytes: Vec<u8> = s.bytes().filter(|b| *b != b'\n' && *b != b'\r').collect();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut result = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let a = b64_decode_char(chunk[0])?;
        let b = b64_decode_char(chunk[1])?;
        result.push((a << 2) | (b >> 4));

        if chunk[2] != b'=' {
            let c = b64_decode_char(chunk[2])?;
            result.push(((b & 0x0F) << 4) | (c >> 2));
            if chunk[3] != b'=' {
                let d = b64_decode_char(chunk[3])?;
                result.push(((c & 0x03) << 6) | d);
            }
        }
    }
    Some(result)
}

// ── Trailer parsing ────────────────────────────────────────────

/// Parse gRPC trailers from raw bytes.
pub fn parse_trailers(data: &[u8]) -> Metadata {
    let text = String::from_utf8_lossy(data);
    let mut meta = Metadata::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(colon) = line.find(':') {
            let key = line[..colon].trim();
            let value = line[colon + 1..].trim();
            meta.insert(key, value);
        }
    }
    meta
}

/// Extract grpc-status and grpc-message from trailers.
pub fn extract_status(trailers: &Metadata) -> (StatusCode, Option<String>) {
    let code = trailers
        .get("grpc-status")
        .and_then(|s| s.parse::<u32>().ok())
        .map(StatusCode::from_code)
        .unwrap_or(StatusCode::Unknown);

    let message = trailers.get("grpc-message").map(|s| s.to_string());

    (code, message)
}

// ── gRPC-Web request/response helpers ──────────────────────────

/// Content type for gRPC-Web binary mode.
pub const CONTENT_TYPE_GRPC_WEB: &str = "application/grpc-web";
/// Content type for gRPC-Web text mode (base64).
pub const CONTENT_TYPE_GRPC_WEB_TEXT: &str = "application/grpc-web-text";

/// Build a unary gRPC-Web request body.
pub fn build_unary_request(message: &[u8]) -> Vec<u8> {
    encode_data_frame(message)
}

/// Build a unary gRPC-Web response body (data + trailers).
pub fn build_unary_response(message: &[u8], status: StatusCode, message_text: Option<&str>) -> Vec<u8> {
    let mut buf = encode_data_frame(message);

    let mut trailers = Metadata::new();
    trailers.insert("grpc-status", &status.code().to_string());
    if let Some(msg) = message_text {
        trailers.insert("grpc-message", msg);
    }
    buf.extend(encode_trailer_frame(&trailers));
    buf
}

/// Encode a gRPC-Web body for text mode (base64).
pub fn encode_text_mode(binary_body: &[u8]) -> String {
    base64_encode(binary_body)
}

/// Decode a gRPC-Web text mode body.
pub fn decode_text_mode(text_body: &str) -> Option<Vec<u8>> {
    base64_decode(text_body)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_code_roundtrip() {
        for code in 0u32..=16 {
            let sc = StatusCode::from_code(code);
            assert_eq!(sc.code(), code);
        }
    }

    #[test]
    fn status_code_unknown_value() {
        let sc = StatusCode::from_code(999);
        assert_eq!(sc, StatusCode::Unknown);
    }

    #[test]
    fn metadata_basic() {
        let mut m = Metadata::new();
        m.insert("Content-Type", "application/grpc");
        m.insert("X-Custom", "value");
        assert_eq!(m.get("content-type"), Some("application/grpc"));
        assert_eq!(m.get("x-custom"), Some("value"));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn metadata_binary_key() {
        assert!(Metadata::is_binary_key("grpc-payload-bin"));
        assert!(!Metadata::is_binary_key("content-type"));
    }

    #[test]
    fn encode_decode_data_frame() {
        let payload = b"hello grpc";
        let encoded = encode_data_frame(payload);
        assert_eq!(encoded[0], DATA_FRAME);
        assert_eq!(encoded.len(), 5 + payload.len());

        let (frame, consumed) = decode_frame(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(frame, GrpcFrame::Data(payload.to_vec()));
    }

    #[test]
    fn encode_decode_trailer_frame() {
        let mut trailers = Metadata::new();
        trailers.insert("grpc-status", "0");
        trailers.insert("grpc-message", "ok");
        let encoded = encode_trailer_frame(&trailers);
        assert_eq!(encoded[0] & TRAILER_FRAME, TRAILER_FRAME);

        let (frame, _) = decode_frame(&encoded).unwrap();
        if let GrpcFrame::Trailer(data) = frame {
            let parsed = parse_trailers(&data);
            assert_eq!(parsed.get("grpc-status"), Some("0"));
            assert_eq!(parsed.get("grpc-message"), Some("ok"));
        } else {
            panic!("expected trailer frame");
        }
    }

    #[test]
    fn decode_multiple_frames() {
        let mut buf = encode_data_frame(b"msg1");
        buf.extend(encode_data_frame(b"msg2"));
        let frames = decode_all_frames(&buf);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn base64_roundtrip() {
        let data = b"Hello, gRPC-Web!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn base64_padding() {
        // 1 byte -> 4 chars (2 padding)
        assert_eq!(base64_encode(b"A"), "QQ==");
        // 2 bytes -> 4 chars (1 padding)
        assert_eq!(base64_encode(b"AB"), "QUI=");
        // 3 bytes -> 4 chars (no padding)
        assert_eq!(base64_encode(b"ABC"), "QUJD");
    }

    #[test]
    fn base64_decode_invalid() {
        assert!(base64_decode("!!!").is_none());
    }

    #[test]
    fn build_unary_request_body() {
        let body = build_unary_request(b"request-data");
        let frames = decode_all_frames(&body);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], GrpcFrame::Data(b"request-data".to_vec()));
    }

    #[test]
    fn build_unary_response_body() {
        let body = build_unary_response(b"response", StatusCode::Ok, Some("success"));
        let frames = decode_all_frames(&body);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], GrpcFrame::Data(b"response".to_vec()));
        if let GrpcFrame::Trailer(data) = &frames[1] {
            let trailers = parse_trailers(data);
            let (code, msg) = extract_status(&trailers);
            assert_eq!(code, StatusCode::Ok);
            assert_eq!(msg.as_deref(), Some("success"));
        } else {
            panic!("expected trailer");
        }
    }

    #[test]
    fn text_mode_roundtrip() {
        let binary = encode_data_frame(b"test data");
        let text = encode_text_mode(&binary);
        let decoded = decode_text_mode(&text).unwrap();
        assert_eq!(decoded, binary);
    }

    #[test]
    fn extract_status_from_trailers() {
        let mut meta = Metadata::new();
        meta.insert("grpc-status", "5");
        meta.insert("grpc-message", "not found");
        let (code, msg) = extract_status(&meta);
        assert_eq!(code, StatusCode::NotFound);
        assert_eq!(msg.as_deref(), Some("not found"));
    }

    #[test]
    fn content_types() {
        assert_eq!(CONTENT_TYPE_GRPC_WEB, "application/grpc-web");
        assert_eq!(CONTENT_TYPE_GRPC_WEB_TEXT, "application/grpc-web-text");
    }
}
