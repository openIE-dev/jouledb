//! URL encoding and decoding (RFC 3986).
//!
//! Percent-encodes and decodes URL components, parses and builds
//! query strings. Replaces encodeURIComponent / decodeURIComponent
//! and querystring npm packages with pure Rust.

use std::collections::HashMap;

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during URL encoding/decoding.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UrlEncodeError {
    #[error("invalid percent-encoded sequence at position {0}")]
    InvalidSequence(usize),
    #[error("invalid UTF-8 in decoded data")]
    InvalidUtf8,
    #[error("incomplete percent sequence at position {0}")]
    IncompleteSequence(usize),
}

// ── Unreserved Characters ────────────────────────────────────────────

/// RFC 3986 unreserved characters: A-Z a-z 0-9 - _ . ~
fn is_unreserved(b: u8) -> bool {
    matches!(b,
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~'
    )
}

/// Characters safe in URL paths (unreserved + sub-delims + : @ /).
fn is_path_safe(b: u8) -> bool {
    is_unreserved(b)
        || matches!(b,
            b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b';' | b'='
            | b':' | b'@' | b'/'
        )
}

/// Characters safe in query strings.
fn is_query_safe(b: u8) -> bool {
    is_path_safe(b) || b == b'?' || b == b'/'
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

// ── Component Encoding ───────────────────────────────────────────────

/// Percent-encode a URL component (encodes everything except unreserved).
///
/// Equivalent to JavaScript's `encodeURIComponent`.
pub fn encode_component(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for &b in input.as_bytes() {
        if is_unreserved(b) {
            result.push(b as char);
        } else {
            result.push('%');
            result.push(HEX_UPPER[(b >> 4) as usize] as char);
            result.push(HEX_UPPER[(b & 0x0F) as usize] as char);
        }
    }
    result
}

/// Percent-encode a full URL, preserving structure (scheme, host, path, query).
///
/// Preserves characters that are valid in their respective URL positions.
pub fn encode_url(input: &str) -> String {
    // Find scheme (e.g., "https://").
    if let Some(scheme_end) = input.find("://") {
        let scheme = &input[..scheme_end + 3];
        let rest = &input[scheme_end + 3..];

        let mut result = scheme.to_string();

        // Split at first '?' for query string.
        if let Some(q_pos) = rest.find('?') {
            let path_part = &rest[..q_pos];
            let query_part = &rest[q_pos + 1..];

            // Find host vs path boundary.
            if let Some(slash_pos) = path_part.find('/') {
                let host = &path_part[..slash_pos];
                let path = &path_part[slash_pos..];
                result.push_str(host);
                encode_path_into(path, &mut result);
            } else {
                result.push_str(path_part);
            }

            result.push('?');
            encode_query_into(query_part, &mut result);
        } else {
            // No query string.
            if let Some(slash_pos) = rest.find('/') {
                let host = &rest[..slash_pos];
                let path = &rest[slash_pos..];
                result.push_str(host);
                encode_path_into(path, &mut result);
            } else {
                result.push_str(rest);
            }
        }

        result
    } else {
        // No scheme — treat as a path.
        let mut result = String::with_capacity(input.len());
        encode_path_into(input, &mut result);
        result
    }
}

fn encode_path_into(path: &str, out: &mut String) {
    for &b in path.as_bytes() {
        if is_path_safe(b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX_UPPER[(b >> 4) as usize] as char);
            out.push(HEX_UPPER[(b & 0x0F) as usize] as char);
        }
    }
}

fn encode_query_into(query: &str, out: &mut String) {
    for &b in query.as_bytes() {
        if is_query_safe(b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX_UPPER[(b >> 4) as usize] as char);
            out.push(HEX_UPPER[(b & 0x0F) as usize] as char);
        }
    }
}

// ── Decoding ─────────────────────────────────────────────────────────

/// Decode a percent-encoded string.
///
/// Handles `+` as space (form encoding) when `plus_as_space` is true.
pub fn decode(input: &str, plus_as_space: bool) -> Result<String, UrlEncodeError> {
    let bytes = decode_bytes(input, plus_as_space)?;
    String::from_utf8(bytes).map_err(|_| UrlEncodeError::InvalidUtf8)
}

/// Decode a percent-encoded string to bytes.
pub fn decode_bytes(input: &str, plus_as_space: bool) -> Result<Vec<u8>, UrlEncodeError> {
    let bytes = input.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(UrlEncodeError::IncompleteSequence(i));
            }
            let hi = hex_val(bytes[i + 1])
                .ok_or(UrlEncodeError::InvalidSequence(i))?;
            let lo = hex_val(bytes[i + 2])
                .ok_or(UrlEncodeError::InvalidSequence(i))?;
            result.push((hi << 4) | lo);
            i += 3;
        } else if plus_as_space && bytes[i] == b'+' {
            result.push(b' ');
            i += 1;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }

    Ok(result)
}

/// Decode a percent-encoded string (no plus-as-space).
pub fn decode_component(input: &str) -> Result<String, UrlEncodeError> {
    decode(input, false)
}

// ── Query String ─────────────────────────────────────────────────────

/// Parse a query string into key-value pairs.
///
/// Handles `+` as space. Supports repeated keys (last value wins in
/// the returned HashMap).
pub fn parse_query(query: &str) -> Result<HashMap<String, String>, UrlEncodeError> {
    let query = query.strip_prefix('?').unwrap_or(query);
    let mut map = HashMap::new();

    if query.is_empty() {
        return Ok(map);
    }

    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = if let Some(eq_pos) = pair.find('=') {
            (&pair[..eq_pos], &pair[eq_pos + 1..])
        } else {
            (pair, "")
        };

        let decoded_key = decode(key, true)?;
        let decoded_value = decode(value, true)?;
        map.insert(decoded_key, decoded_value);
    }

    Ok(map)
}

/// Parse a query string preserving all values for repeated keys.
pub fn parse_query_multi(query: &str) -> Result<HashMap<String, Vec<String>>, UrlEncodeError> {
    let query = query.strip_prefix('?').unwrap_or(query);
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    if query.is_empty() {
        return Ok(map);
    }

    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = if let Some(eq_pos) = pair.find('=') {
            (&pair[..eq_pos], &pair[eq_pos + 1..])
        } else {
            (pair, "")
        };

        let decoded_key = decode(key, true)?;
        let decoded_value = decode(value, true)?;
        map.entry(decoded_key).or_default().push(decoded_value);
    }

    Ok(map)
}

/// Build a query string from key-value pairs.
pub fn build_query(params: &HashMap<String, String>) -> String {
    let mut pairs: Vec<String> = params
        .iter()
        .map(|(k, v)| {
            format!("{}={}", encode_component(k), encode_component(v))
        })
        .collect();
    pairs.sort(); // Deterministic output.
    pairs.join("&")
}

/// Build a query string from key-value pairs (preserving order via slice).
pub fn build_query_ordered(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", encode_component(k), encode_component(v)))
        .collect::<Vec<_>>()
        .join("&")
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_component_basic() {
        assert_eq!(encode_component("hello world"), "hello%20world");
        assert_eq!(encode_component("foo@bar.com"), "foo%40bar.com");
        assert_eq!(encode_component("100%"), "100%25");
    }

    #[test]
    fn encode_component_unreserved_passthrough() {
        assert_eq!(encode_component("ABCxyz019-_.~"), "ABCxyz019-_.~");
    }

    #[test]
    fn encode_component_utf8() {
        let encoded = encode_component("caf\u{00E9}");
        assert_eq!(encoded, "caf%C3%A9");
    }

    #[test]
    fn decode_component_basic() {
        assert_eq!(decode_component("hello%20world").unwrap(), "hello world");
        assert_eq!(decode_component("foo%40bar.com").unwrap(), "foo@bar.com");
    }

    #[test]
    fn decode_plus_as_space() {
        assert_eq!(decode("hello+world", true).unwrap(), "hello world");
        assert_eq!(decode("hello+world", false).unwrap(), "hello+world");
    }

    #[test]
    fn roundtrip_encode_decode() {
        let input = "hello world & foo=bar / baz?qux";
        let encoded = encode_component(input);
        let decoded = decode_component(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn encode_url_preserves_structure() {
        let url = "https://example.com/path with spaces/file?q=hello world";
        let encoded = encode_url(url);
        assert!(encoded.starts_with("https://example.com/"));
        assert!(encoded.contains("%20"));
    }

    #[test]
    fn parse_query_basic() {
        let map = parse_query("key1=value1&key2=value2").unwrap();
        assert_eq!(map.get("key1").unwrap(), "value1");
        assert_eq!(map.get("key2").unwrap(), "value2");
    }

    #[test]
    fn parse_query_with_leading_question_mark() {
        let map = parse_query("?foo=bar&baz=qux").unwrap();
        assert_eq!(map.get("foo").unwrap(), "bar");
    }

    #[test]
    fn parse_query_encoded_values() {
        let map = parse_query("name=John+Doe&city=New+York").unwrap();
        assert_eq!(map.get("name").unwrap(), "John Doe");
        assert_eq!(map.get("city").unwrap(), "New York");
    }

    #[test]
    fn parse_query_multi_values() {
        let map = parse_query_multi("tag=a&tag=b&tag=c").unwrap();
        let tags = map.get("tag").unwrap();
        assert_eq!(tags.len(), 3);
        assert!(tags.contains(&"a".to_string()));
        assert!(tags.contains(&"b".to_string()));
        assert!(tags.contains(&"c".to_string()));
    }

    #[test]
    fn build_query_basic() {
        let mut params = HashMap::new();
        params.insert("key".to_string(), "value".to_string());
        params.insert("foo".to_string(), "bar baz".to_string());
        let query = build_query(&params);
        assert!(query.contains("key=value"));
        assert!(query.contains("foo=bar%20baz"));
    }

    #[test]
    fn test_build_query_ordered() {
        let params = [("b", "2"), ("a", "1")];
        let query = super::build_query_ordered(&params);
        assert_eq!(query, "b=2&a=1");
    }

    #[test]
    fn decode_invalid_sequence() {
        assert!(decode_component("hello%ZZ").is_err());
        assert!(decode_component("hello%2").is_err());
    }

    #[test]
    fn empty_inputs() {
        assert_eq!(encode_component(""), "");
        assert_eq!(decode_component("").unwrap(), "");
        assert!(parse_query("").unwrap().is_empty());
    }

    #[test]
    fn encode_url_no_scheme() {
        let encoded = encode_url("/path/to file");
        assert_eq!(encoded, "/path/to%20file");
    }
}
