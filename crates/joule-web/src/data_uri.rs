//! Data URI (RFC 2397) parsing and generation.
//!
//! Replaces datauri, data-urls, and mini-svg-data-uri npm packages with a
//! pure-Rust implementation supporting base64 encoding, mediatype extraction,
//! size limits, and text/base64 modes.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────

/// Default maximum data URI size (2 MB).
pub const DEFAULT_MAX_SIZE: usize = 2 * 1024 * 1024;

// ── DataUri ─────────────────────────────────────────────────────

/// Encoding mode for the data payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Percent-encoded text (default per RFC 2397).
    Text,
    /// Base64-encoded binary.
    Base64,
}

/// A parsed or constructed data URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataUri {
    /// Media type string (e.g. `text/plain`, `image/png`).
    pub media_type: String,
    /// Optional charset parameter.
    pub charset: Option<String>,
    /// Encoding mode.
    pub encoding: Encoding,
    /// Raw decoded data.
    pub data: Vec<u8>,
}

impl DataUri {
    /// Create a data URI from raw bytes with a media type.
    pub fn from_bytes(media_type: &str, data: &[u8]) -> Self {
        Self {
            media_type: media_type.to_string(),
            charset: None,
            encoding: Encoding::Base64,
            data: data.to_vec(),
        }
    }

    /// Create a data URI from a UTF-8 text string.
    pub fn from_text(media_type: &str, text: &str) -> Self {
        Self {
            media_type: media_type.to_string(),
            charset: Some("utf-8".to_string()),
            encoding: Encoding::Text,
            data: text.as_bytes().to_vec(),
        }
    }

    /// Create a text/plain data URI.
    pub fn text_plain(text: &str) -> Self {
        Self::from_text("text/plain", text)
    }

    /// Create an SVG data URI.
    pub fn svg(svg_content: &str) -> Self {
        Self::from_text("image/svg+xml", svg_content)
    }

    /// Create a PNG data URI from raw PNG bytes.
    pub fn png(data: &[u8]) -> Self {
        Self::from_bytes("image/png", data)
    }

    /// Create a JPEG data URI from raw JPEG bytes.
    pub fn jpeg(data: &[u8]) -> Self {
        Self::from_bytes("image/jpeg", data)
    }

    /// Create a JSON data URI.
    pub fn json(json_str: &str) -> Self {
        Self::from_text("application/json", json_str)
    }

    /// Parse a data URI string.
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        let rest = trimmed.strip_prefix("data:")?;

        // Split at the comma separating metadata from data
        let comma_pos = rest.find(',')?;
        let metadata = &rest[..comma_pos];
        let data_str = &rest[comma_pos + 1..];

        // Parse metadata: [mediatype][;charset=xxx][;base64]
        let is_base64 = metadata.ends_with(";base64");
        let meta_without_base64 = if is_base64 {
            &metadata[..metadata.len() - 7]
        } else {
            metadata
        };

        let mut media_type = String::new();
        let mut charset = None;

        if !meta_without_base64.is_empty() {
            let parts: Vec<&str> = meta_without_base64.split(';').collect();
            if !parts.is_empty() && !parts[0].is_empty() {
                media_type = parts[0].to_string();
            }
            for part in &parts[1..] {
                let part = part.trim();
                if let Some(val) = part.strip_prefix("charset=") {
                    charset = Some(val.to_string());
                }
            }
        }

        if media_type.is_empty() {
            media_type = "text/plain".to_string();
        }

        let data = if is_base64 {
            base64_decode(data_str)?
        } else {
            percent_decode_bytes(data_str)
        };

        let encoding = if is_base64 {
            Encoding::Base64
        } else {
            Encoding::Text
        };

        Some(Self {
            media_type,
            charset,
            encoding,
            data,
        })
    }

    /// Parse with a size limit. Returns None if the decoded data exceeds the limit.
    pub fn parse_with_limit(input: &str, max_bytes: usize) -> Option<Self> {
        let uri = Self::parse(input)?;
        if uri.data.len() > max_bytes {
            None
        } else {
            Some(uri)
        }
    }

    /// Get the data as a UTF-8 string if possible.
    pub fn as_text(&self) -> Option<&str> {
        std::str::from_utf8(&self.data).ok()
    }

    /// Get the data size in bytes.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Get the approximate encoded URI length.
    pub fn encoded_size(&self) -> usize {
        // Approximate: "data:" + mediatype + metadata + encoded data
        let meta_len = 5 + self.media_type.len() + 10; // overhead
        match self.encoding {
            Encoding::Base64 => meta_len + (self.data.len() + 2) / 3 * 4,
            Encoding::Text => meta_len + self.data.len() * 3, // worst case percent-encoded
        }
    }

    /// Check if this is a text-based data URI.
    pub fn is_text(&self) -> bool {
        self.encoding == Encoding::Text
    }

    /// Check if this is base64-encoded.
    pub fn is_base64(&self) -> bool {
        self.encoding == Encoding::Base64
    }

    /// Get the main media type (before the slash).
    pub fn main_type(&self) -> &str {
        self.media_type
            .split('/')
            .next()
            .unwrap_or("application")
    }

    /// Get the sub type (after the slash).
    pub fn sub_type(&self) -> &str {
        self.media_type.split('/').nth(1).unwrap_or("octet-stream")
    }

    /// Convert to use base64 encoding.
    pub fn to_base64(mut self) -> Self {
        self.encoding = Encoding::Base64;
        self
    }

    /// Convert to use text encoding.
    pub fn to_text_encoding(mut self) -> Self {
        self.encoding = Encoding::Text;
        self
    }

    /// Render as a complete data URI string.
    pub fn to_uri(&self) -> String {
        self.to_string()
    }
}

// ── Display ─────────────────────────────────────────────────────

impl fmt::Display for DataUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "data:{}", self.media_type)?;

        if let Some(charset) = &self.charset {
            write!(f, ";charset={}", charset)?;
        }

        match self.encoding {
            Encoding::Base64 => {
                write!(f, ";base64,{}", base64_encode(&self.data))?;
            }
            Encoding::Text => {
                write!(f, ",{}", percent_encode_data(&self.data))?;
            }
        }

        Ok(())
    }
}

// ── Base64 (built-in, no deps) ──────────────────────────────────

const B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 2 < data.len() {
        let b0 = data[i] as u32;
        let b1 = data[i + 1] as u32;
        let b2 = data[i + 2] as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        out.push(B64_CHARS[(triple & 0x3F) as usize] as char);
        i += 3;
    }
    let remaining = data.len() - i;
    if remaining == 1 {
        let b0 = data[i] as u32;
        out.push(B64_CHARS[((b0 >> 2) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((b0 << 4) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if remaining == 2 {
        let b0 = data[i] as u32;
        let b1 = data[i + 1] as u32;
        out.push(B64_CHARS[((b0 >> 2) & 0x3F) as usize] as char);
        out.push(B64_CHARS[(((b0 << 4) | (b1 >> 4)) & 0x3F) as usize] as char);
        out.push(B64_CHARS[((b1 << 2) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let clean: String = input
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();

    if clean.is_empty() {
        return Some(Vec::new());
    }

    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    let mut buf = [0u8; 4];
    let mut buf_len = 0;
    let mut pad_count = 0;

    for c in clean.chars() {
        let val = if c == '=' {
            pad_count += 1;
            0
        } else {
            b64_char_value(c)?
        };
        buf[buf_len] = val;
        buf_len += 1;

        if buf_len == 4 {
            let triple = ((buf[0] as u32) << 18)
                | ((buf[1] as u32) << 12)
                | ((buf[2] as u32) << 6)
                | (buf[3] as u32);
            out.push((triple >> 16) as u8);
            if pad_count < 2 {
                out.push((triple >> 8) as u8);
            }
            if pad_count < 1 {
                out.push(triple as u8);
            }
            buf_len = 0;
            pad_count = 0;
        }
    }

    Some(out)
}

fn b64_char_value(c: char) -> Option<u8> {
    match c {
        'A'..='Z' => Some(c as u8 - b'A'),
        'a'..='z' => Some(c as u8 - b'a' + 26),
        '0'..='9' => Some(c as u8 - b'0' + 52),
        '+' => Some(62),
        '/' => Some(63),
        _ => None,
    }
}

// ── Percent encoding ────────────────────────────────────────────

fn percent_encode_data(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len() * 3);
    for &b in data {
        if b.is_ascii_alphanumeric()
            || b == b'-'
            || b == b'_'
            || b == b'.'
            || b == b'~'
            || b == b' '
        {
            if b == b' ' {
                out.push_str("%20");
            } else {
                out.push(b as char);
            }
        } else {
            out.push('%');
            out.push(hex_char(b >> 4));
            out.push(hex_char(b & 0x0F));
        }
    }
    out
}

fn percent_decode_bytes(s: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let h1 = chars.next().unwrap_or('0');
            let h2 = chars.next().unwrap_or('0');
            let hex: String = [h1, h2].iter().collect();
            if let Ok(b) = u8::from_str_radix(&hex, 16) {
                bytes.push(b);
            }
        } else {
            // For data URIs, chars map to their byte representation
            for b in c.to_string().bytes() {
                bytes.push(b);
            }
        }
    }
    bytes
}

fn hex_char(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + n - 10) as char,
        _ => '0',
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_text_plain() {
        let uri = DataUri::parse("data:text/plain,Hello%20World").unwrap();
        assert_eq!(uri.media_type, "text/plain");
        assert_eq!(uri.encoding, Encoding::Text);
        assert_eq!(uri.as_text(), Some("Hello World"));
    }

    #[test]
    fn parse_default_mediatype() {
        let uri = DataUri::parse("data:,Hello").unwrap();
        assert_eq!(uri.media_type, "text/plain");
        assert_eq!(uri.as_text(), Some("Hello"));
    }

    #[test]
    fn parse_base64() {
        let uri = DataUri::parse("data:text/plain;base64,SGVsbG8=").unwrap();
        assert_eq!(uri.encoding, Encoding::Base64);
        assert_eq!(uri.as_text(), Some("Hello"));
    }

    #[test]
    fn parse_with_charset() {
        let uri = DataUri::parse("data:text/plain;charset=utf-8,test").unwrap();
        assert_eq!(uri.charset, Some("utf-8".to_string()));
    }

    #[test]
    fn parse_binary_base64() {
        // Small PNG-like header bytes
        let bytes = [0x89, 0x50, 0x4E, 0x47];
        let encoded = base64_encode(&bytes);
        let uri_str = format!("data:image/png;base64,{}", encoded);
        let uri = DataUri::parse(&uri_str).unwrap();
        assert_eq!(uri.media_type, "image/png");
        assert_eq!(uri.data, bytes);
    }

    #[test]
    fn parse_invalid() {
        assert!(DataUri::parse("").is_none());
        assert!(DataUri::parse("http://example.com").is_none());
        assert!(DataUri::parse("data:").is_none());
    }

    #[test]
    fn parse_with_limit_ok() {
        let uri = DataUri::parse_with_limit("data:text/plain,Hello", 100).unwrap();
        assert_eq!(uri.as_text(), Some("Hello"));
    }

    #[test]
    fn parse_with_limit_exceeded() {
        let result = DataUri::parse_with_limit("data:text/plain,Hello", 3);
        assert!(result.is_none());
    }

    #[test]
    fn from_text_constructor() {
        let uri = DataUri::text_plain("Hello World");
        assert_eq!(uri.media_type, "text/plain");
        assert_eq!(uri.charset, Some("utf-8".to_string()));
        assert_eq!(uri.as_text(), Some("Hello World"));
    }

    #[test]
    fn from_bytes_constructor() {
        let uri = DataUri::png(&[1, 2, 3, 4]);
        assert_eq!(uri.media_type, "image/png");
        assert_eq!(uri.encoding, Encoding::Base64);
        assert_eq!(uri.data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn svg_constructor() {
        let uri = DataUri::svg("<svg></svg>");
        assert_eq!(uri.media_type, "image/svg+xml");
        assert!(uri.is_text());
    }

    #[test]
    fn json_constructor() {
        let uri = DataUri::json(r#"{"key":"value"}"#);
        assert_eq!(uri.media_type, "application/json");
    }

    #[test]
    fn jpeg_constructor() {
        let uri = DataUri::jpeg(&[0xFF, 0xD8, 0xFF]);
        assert_eq!(uri.media_type, "image/jpeg");
        assert!(uri.is_base64());
    }

    #[test]
    fn data_size() {
        let uri = DataUri::text_plain("Hello");
        assert_eq!(uri.data_size(), 5);
    }

    #[test]
    fn main_and_sub_type() {
        let uri = DataUri::png(&[]);
        assert_eq!(uri.main_type(), "image");
        assert_eq!(uri.sub_type(), "png");
    }

    #[test]
    fn display_text() {
        let uri = DataUri::text_plain("Hi");
        let s = uri.to_string();
        assert!(s.starts_with("data:text/plain;charset=utf-8,"));
    }

    #[test]
    fn display_base64() {
        let uri = DataUri::from_bytes("image/png", &[1, 2, 3]);
        let s = uri.to_string();
        assert!(s.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn roundtrip_text() {
        let original = DataUri::text_plain("Hello, World!");
        let uri_str = original.to_string();
        let parsed = DataUri::parse(&uri_str).unwrap();
        assert_eq!(parsed.as_text(), Some("Hello, World!"));
    }

    #[test]
    fn roundtrip_base64() {
        let original_data = vec![0, 1, 2, 255, 128, 64];
        let original = DataUri::from_bytes("application/octet-stream", &original_data);
        let uri_str = original.to_string();
        let parsed = DataUri::parse(&uri_str).unwrap();
        assert_eq!(parsed.data, original_data);
    }

    #[test]
    fn base64_encode_empty() {
        assert_eq!(base64_encode(&[]), "");
    }

    #[test]
    fn base64_encode_one_byte() {
        assert_eq!(base64_encode(&[0x4D]), "TQ==");
    }

    #[test]
    fn base64_encode_two_bytes() {
        assert_eq!(base64_encode(&[0x4D, 0x61]), "TWE=");
    }

    #[test]
    fn base64_encode_three_bytes() {
        assert_eq!(base64_encode(&[0x4D, 0x61, 0x6E]), "TWFu");
    }

    #[test]
    fn base64_decode_empty() {
        assert_eq!(base64_decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn base64_roundtrip() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = base64_encode(&data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn base64_decode_invalid_char() {
        assert!(base64_decode("!!!").is_none());
    }

    #[test]
    fn to_base64_conversion() {
        let uri = DataUri::text_plain("Hello").to_base64();
        assert_eq!(uri.encoding, Encoding::Base64);
    }

    #[test]
    fn to_text_conversion() {
        let uri = DataUri::from_bytes("text/plain", b"Hello").to_text_encoding();
        assert_eq!(uri.encoding, Encoding::Text);
    }

    #[test]
    fn to_uri_method() {
        let uri = DataUri::text_plain("x");
        assert_eq!(uri.to_uri(), uri.to_string());
    }

    #[test]
    fn encoded_size_estimate() {
        let uri = DataUri::from_bytes("image/png", &[0u8; 100]);
        let est = uri.encoded_size();
        let actual = uri.to_string().len();
        // Estimate should be in the right ballpark
        assert!(est >= actual / 2);
    }

    #[test]
    fn parse_empty_data() {
        let uri = DataUri::parse("data:text/plain,").unwrap();
        assert!(uri.data.is_empty());
    }

    #[test]
    fn parse_empty_base64() {
        let uri = DataUri::parse("data:text/plain;base64,").unwrap();
        assert!(uri.data.is_empty());
    }

    #[test]
    fn percent_encoding_special_chars() {
        let uri = DataUri::text_plain("a=b&c=d");
        let s = uri.to_string();
        let parsed = DataUri::parse(&s).unwrap();
        assert_eq!(parsed.as_text(), Some("a=b&c=d"));
    }
}
