//! Low-level percent encoding (RFC 3986).
//!
//! Provides configurable encode sets for different URL components,
//! streaming encoding, double-encoding detection, and normalization.
//! Replaces the percent-encoding npm package with pure Rust.

// ── Encode Sets ──────────────────────────────────────────────────────

/// Predefined percent-encode sets for different URL positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeSet {
    /// RFC 3986 component encoding — encodes everything except unreserved.
    Component,
    /// Path encoding — preserves `/`, `:`, `@`, sub-delims.
    Path,
    /// Query encoding — preserves path-safe chars plus `?` and `/`.
    Query,
    /// Fragment encoding — same as query.
    Fragment,
    /// Userinfo encoding — preserves unreserved and sub-delims and `:`.
    UserInfo,
}

/// A custom encode set defined by a predicate function.
pub struct CustomEncodeSet {
    /// Returns `true` if the byte should be encoded (i.e., is NOT safe).
    pub should_encode: fn(u8) -> bool,
}

// ── RFC 3986 Character Classes ───────────────────────────────────────

/// RFC 3986 unreserved characters: ALPHA / DIGIT / "-" / "." / "_" / "~"
fn is_unreserved(b: u8) -> bool {
    matches!(b,
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~'
    )
}

/// Sub-delimiters: "!" / "$" / "&" / "'" / "(" / ")" / "*" / "+" / "," / ";" / "="
fn is_sub_delim(b: u8) -> bool {
    matches!(b,
        b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b';' | b'='
    )
}

/// Determine if a byte should be encoded based on the encode set.
fn needs_encoding(b: u8, set: EncodeSet) -> bool {
    match set {
        EncodeSet::Component => !is_unreserved(b),
        EncodeSet::Path => {
            !(is_unreserved(b) || is_sub_delim(b) || matches!(b, b':' | b'@' | b'/'))
        }
        EncodeSet::Query => {
            !(is_unreserved(b)
                || is_sub_delim(b)
                || matches!(b, b':' | b'@' | b'/' | b'?'))
        }
        EncodeSet::Fragment => {
            !(is_unreserved(b)
                || is_sub_delim(b)
                || matches!(b, b':' | b'@' | b'/' | b'?'))
        }
        EncodeSet::UserInfo => {
            !(is_unreserved(b) || is_sub_delim(b) || b == b':')
        }
    }
}

// ── Hex Helpers ──────────────────────────────────────────────────────

const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

fn is_hex(c: u8) -> bool {
    matches!(c, b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F')
}

// ── Errors ───────────────────────────────────────────────────────────

/// Errors from percent encoding/decoding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PercentEncodeError {
    #[error("invalid percent sequence at position {0}")]
    InvalidSequence(usize),
    #[error("incomplete percent sequence at position {0}")]
    IncompleteSequence(usize),
    #[error("invalid UTF-8 in decoded bytes")]
    InvalidUtf8,
}

// ── Encode ───────────────────────────────────────────────────────────

/// Percent-encode bytes according to the specified encode set.
pub fn encode(input: &[u8], set: EncodeSet) -> String {
    let mut result = String::with_capacity(input.len());
    for &b in input {
        if needs_encoding(b, set) {
            result.push('%');
            result.push(HEX_UPPER[(b >> 4) as usize] as char);
            result.push(HEX_UPPER[(b & 0x0F) as usize] as char);
        } else {
            result.push(b as char);
        }
    }
    result
}

/// Percent-encode a string according to the specified encode set.
pub fn encode_str(input: &str, set: EncodeSet) -> String {
    encode(input.as_bytes(), set)
}

/// Percent-encode bytes with a custom encode set.
pub fn encode_custom(input: &[u8], custom: &CustomEncodeSet) -> String {
    let mut result = String::with_capacity(input.len());
    for &b in input {
        if (custom.should_encode)(b) {
            result.push('%');
            result.push(HEX_UPPER[(b >> 4) as usize] as char);
            result.push(HEX_UPPER[(b & 0x0F) as usize] as char);
        } else {
            result.push(b as char);
        }
    }
    result
}

// ── Decode ───────────────────────────────────────────────────────────

/// Decode percent-encoded bytes.
pub fn decode_bytes(input: &[u8]) -> Result<Vec<u8>, PercentEncodeError> {
    let mut result = Vec::with_capacity(input.len());
    let mut i = 0;

    while i < input.len() {
        if input[i] == b'%' {
            if i + 2 >= input.len() {
                return Err(PercentEncodeError::IncompleteSequence(i));
            }
            let hi = hex_val(input[i + 1])
                .ok_or(PercentEncodeError::InvalidSequence(i))?;
            let lo = hex_val(input[i + 2])
                .ok_or(PercentEncodeError::InvalidSequence(i))?;
            result.push((hi << 4) | lo);
            i += 3;
        } else {
            result.push(input[i]);
            i += 1;
        }
    }

    Ok(result)
}

/// Decode a percent-encoded string.
pub fn decode_str(input: &str) -> Result<String, PercentEncodeError> {
    let bytes = decode_bytes(input.as_bytes())?;
    String::from_utf8(bytes).map_err(|_| PercentEncodeError::InvalidUtf8)
}

// ── Streaming Encoder ────────────────────────────────────────────────

/// Streaming percent encoder that processes one byte at a time.
pub struct StreamingEncoder {
    set: EncodeSet,
    output: String,
}

impl StreamingEncoder {
    /// Create a new streaming encoder.
    pub fn new(set: EncodeSet) -> Self {
        Self {
            set,
            output: String::new(),
        }
    }

    /// Feed a single byte.
    pub fn write_byte(&mut self, b: u8) {
        if needs_encoding(b, self.set) {
            self.output.push('%');
            self.output.push(HEX_UPPER[(b >> 4) as usize] as char);
            self.output.push(HEX_UPPER[(b & 0x0F) as usize] as char);
        } else {
            self.output.push(b as char);
        }
    }

    /// Feed a slice of bytes.
    pub fn write(&mut self, data: &[u8]) {
        for &b in data {
            self.write_byte(b);
        }
    }

    /// Finish and return the encoded string.
    pub fn finish(self) -> String {
        self.output
    }
}

// ── Double-Encoding Detection ────────────────────────────────────────

/// Detect whether a string appears to be double-encoded.
///
/// Returns `true` if the string contains sequences like `%25XX` where
/// XX are hex digits (i.e., a percent sign that was itself percent-encoded).
pub fn is_double_encoded(input: &str) -> bool {
    let bytes = input.as_bytes();
    let mut i = 0;
    while i + 4 < bytes.len() {
        if bytes[i] == b'%'
            && bytes[i + 1] == b'2'
            && bytes[i + 2] == b'5'
            && i + 5 <= bytes.len()
            && is_hex(bytes[i + 3])
            && is_hex(bytes[i + 4])
        {
            return true;
        }
        i += 1;
    }
    false
}

/// Count the number of encoding layers. Returns 0 for a plain string,
/// 1 for single encoding, 2+ for double/multiple encoding.
pub fn encoding_depth(input: &str) -> usize {
    let mut depth = 0;
    let mut current = input.to_string();
    loop {
        if !current.contains('%') {
            break;
        }
        match decode_str(&current) {
            Ok(decoded) => {
                if decoded == current {
                    break;
                }
                depth += 1;
                current = decoded;
            }
            Err(_) => break,
        }
    }
    depth
}

// ── Normalization ────────────────────────────────────────────────────

/// Normalize a percent-encoded string: decode then re-encode to canonical form.
///
/// Ensures uppercase hex digits and only necessary characters are encoded.
pub fn normalize(input: &str, set: EncodeSet) -> Result<String, PercentEncodeError> {
    let decoded = decode_bytes(input.as_bytes())?;
    Ok(encode(&decoded, set))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_encoding() {
        let input = b"hello world";
        let encoded = encode(input, EncodeSet::Component);
        assert_eq!(encoded, "hello%20world");
    }

    #[test]
    fn path_preserves_slashes() {
        let input = b"/path/to/resource";
        let encoded = encode(input, EncodeSet::Path);
        assert_eq!(encoded, "/path/to/resource");
    }

    #[test]
    fn path_encodes_spaces() {
        let input = b"/path/to my/file";
        let encoded = encode(input, EncodeSet::Path);
        assert_eq!(encoded, "/path/to%20my/file");
    }

    #[test]
    fn query_preserves_question_marks() {
        let input = b"key=value&other=?yes";
        let encoded = encode(input, EncodeSet::Query);
        assert_eq!(encoded, "key=value&other=?yes");
    }

    #[test]
    fn userinfo_encoding() {
        let input = b"user:pass@host";
        let encoded = encode(input, EncodeSet::UserInfo);
        assert_eq!(encoded, "user:pass%40host");
    }

    #[test]
    fn decode_basic() {
        assert_eq!(decode_str("hello%20world").unwrap(), "hello world");
        assert_eq!(decode_str("%2F%2F").unwrap(), "//");
    }

    #[test]
    fn decode_invalid() {
        assert!(decode_str("hello%ZZ").is_err());
        assert!(decode_str("hello%2").is_err());
    }

    #[test]
    fn roundtrip_component() {
        let input = "hello world & foo=bar / baz?qux";
        let encoded = encode_str(input, EncodeSet::Component);
        let decoded = decode_str(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn custom_encode_set() {
        let custom = CustomEncodeSet {
            should_encode: |b| !b.is_ascii_alphanumeric(),
        };
        let encoded = encode_custom(b"hello-world_123", &custom);
        assert_eq!(encoded, "hello%2Dworld%5F123");
    }

    #[test]
    fn streaming_encoder() {
        let mut encoder = StreamingEncoder::new(EncodeSet::Component);
        encoder.write(b"hello ");
        encoder.write(b"world");
        assert_eq!(encoder.finish(), "hello%20world");
    }

    #[test]
    fn streaming_encoder_byte_by_byte() {
        let mut encoder = StreamingEncoder::new(EncodeSet::Component);
        for &b in b"a b" {
            encoder.write_byte(b);
        }
        assert_eq!(encoder.finish(), "a%20b");
    }

    #[test]
    fn double_encoding_detection() {
        assert!(!is_double_encoded("hello%20world"));
        assert!(is_double_encoded("hello%2520world"));
        assert!(!is_double_encoded("plain text"));
    }

    #[test]
    fn encoding_depth_values() {
        assert_eq!(encoding_depth("hello"), 0);
        assert_eq!(encoding_depth("hello%20world"), 1);
        assert_eq!(encoding_depth("hello%2520world"), 2);
    }

    #[test]
    fn normalization() {
        // Lowercase hex should become uppercase.
        let normalized = normalize("hello%20world", EncodeSet::Component).unwrap();
        assert_eq!(normalized, "hello%20world");

        // Unnecessarily encoded unreserved chars should be decoded.
        let normalized = normalize("hello%2Dworld", EncodeSet::Component).unwrap();
        assert_eq!(normalized, "hello-world");
    }

    #[test]
    fn unreserved_chars_passthrough() {
        let unreserved = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
        let encoded = encode(unreserved, EncodeSet::Component);
        assert_eq!(
            encoded,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
        );
    }

    #[test]
    fn utf8_encoding() {
        let input = "caf\u{00E9}";
        let encoded = encode_str(input, EncodeSet::Component);
        assert_eq!(encoded, "caf%C3%A9");
        let decoded = decode_str(&encoded).unwrap();
        assert_eq!(decoded, input);
    }
}
