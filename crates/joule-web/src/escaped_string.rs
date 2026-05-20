//! String escaping and unescaping — JSON, URL, HTML, XML, C-style, shell, CSV.
//!
//! Provides roundtrip-safe encoding/decoding across all common string escaping
//! formats. Replaces JavaScript encodeURIComponent, he, escape-html, and
//! similar npm packages with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during escape/unescape operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EscapeError {
    #[error("invalid escape sequence at position {0}")]
    InvalidEscape(usize),
    #[error("unexpected end of escape sequence")]
    UnexpectedEof,
    #[error("invalid unicode escape: {0}")]
    InvalidUnicode(String),
    #[error("invalid percent-encoded byte at position {0}")]
    InvalidPercentEncoding(usize),
}

// ── JSON Escaping ───────────────────────────────────────────────────

/// Escape a string for safe inclusion in a JSON string value.
/// Escapes: `"`, `\`, control characters (U+0000..U+001F), and
/// characters U+2028/U+2029 (line/paragraph separators).
pub fn json_escape(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + input.len() / 8);
    for ch in input.chars() {
        match ch {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\u{0008}' => result.push_str("\\b"),
            '\u{000C}' => result.push_str("\\f"),
            c if c < '\u{0020}' => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            '\u{2028}' => result.push_str("\\u2028"),
            '\u{2029}' => result.push_str("\\u2029"),
            c => result.push(c),
        }
    }
    result
}

/// Unescape a JSON-escaped string. Handles \\, \", \n, \r, \t, \b, \f,
/// \/, and \uXXXX (including surrogate pairs).
pub fn json_unescape(input: &str) -> Result<String, EscapeError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().enumerate().peekable();

    while let Some((pos, ch)) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some((_, '"')) => result.push('"'),
            Some((_, '\\')) => result.push('\\'),
            Some((_, '/')) => result.push('/'),
            Some((_, 'n')) => result.push('\n'),
            Some((_, 'r')) => result.push('\r'),
            Some((_, 't')) => result.push('\t'),
            Some((_, 'b')) => result.push('\u{0008}'),
            Some((_, 'f')) => result.push('\u{000C}'),
            Some((_, 'u')) => {
                let cp = parse_json_unicode_escape(&mut chars, pos)?;
                result.push(cp);
            }
            Some(_) => return Err(EscapeError::InvalidEscape(pos)),
            None => return Err(EscapeError::UnexpectedEof),
        }
    }
    Ok(result)
}

fn parse_json_unicode_escape(
    chars: &mut std::iter::Peekable<std::iter::Enumerate<std::str::Chars<'_>>>,
    pos: usize,
) -> Result<char, EscapeError> {
    let hex = read_hex_digits(chars, 4, pos)?;
    let code = u32::from_str_radix(&hex, 16)
        .map_err(|_| EscapeError::InvalidUnicode(hex.clone()))?;

    // Handle surrogate pairs.
    if (0xD800..=0xDBFF).contains(&code) {
        // High surrogate — expect \uDCxx low surrogate.
        match (chars.next(), chars.next()) {
            (Some((_, '\\')), Some((_, 'u'))) => {}
            _ => return Err(EscapeError::InvalidUnicode(format!("unpaired high surrogate U+{code:04X}"))),
        }
        let low_hex = read_hex_digits(chars, 4, pos)?;
        let low = u32::from_str_radix(&low_hex, 16)
            .map_err(|_| EscapeError::InvalidUnicode(low_hex.clone()))?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return Err(EscapeError::InvalidUnicode(format!("expected low surrogate, got U+{low:04X}")));
        }
        let combined = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        char::from_u32(combined)
            .ok_or_else(|| EscapeError::InvalidUnicode(format!("invalid codepoint U+{combined:X}")))
    } else if (0xDC00..=0xDFFF).contains(&code) {
        Err(EscapeError::InvalidUnicode(format!("unpaired low surrogate U+{code:04X}")))
    } else {
        char::from_u32(code)
            .ok_or_else(|| EscapeError::InvalidUnicode(format!("invalid codepoint U+{code:X}")))
    }
}

fn read_hex_digits(
    chars: &mut std::iter::Peekable<std::iter::Enumerate<std::str::Chars<'_>>>,
    count: usize,
    pos: usize,
) -> Result<String, EscapeError> {
    let mut hex = String::with_capacity(count);
    for _ in 0..count {
        match chars.next() {
            Some((_, c)) if c.is_ascii_hexdigit() => hex.push(c),
            _ => return Err(EscapeError::InvalidEscape(pos)),
        }
    }
    Ok(hex)
}

// ── URL Percent-Encoding ────────────────────────────────────────────

/// Percent-encode a string (RFC 3986 unreserved characters are NOT encoded).
/// Unreserved: A-Z a-z 0-9 - _ . ~
pub fn url_encode(input: &str) -> String {
    let mut result = String::with_capacity(input.len() * 3);
    for byte in input.bytes() {
        if is_url_unreserved(byte) {
            result.push(byte as char);
        } else {
            result.push('%');
            result.push(HEX_UPPER[(byte >> 4) as usize] as char);
            result.push(HEX_UPPER[(byte & 0x0F) as usize] as char);
        }
    }
    result
}

/// Decode a percent-encoded string.
pub fn url_decode(input: &str) -> Result<String, EscapeError> {
    let bytes = url_decode_bytes(input)?;
    String::from_utf8(bytes).map_err(|_| EscapeError::InvalidPercentEncoding(0))
}

/// Decode a percent-encoded string to bytes.
pub fn url_decode_bytes(input: &str) -> Result<Vec<u8>, EscapeError> {
    let mut result = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(EscapeError::InvalidPercentEncoding(i));
            }
            let hi = hex_val(bytes[i + 1]).ok_or(EscapeError::InvalidPercentEncoding(i + 1))?;
            let lo = hex_val(bytes[i + 2]).ok_or(EscapeError::InvalidPercentEncoding(i + 2))?;
            result.push((hi << 4) | lo);
            i += 3;
        } else if bytes[i] == b'+' {
            result.push(b' ');
            i += 1;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    Ok(result)
}

fn is_url_unreserved(byte: u8) -> bool {
    matches!(byte,
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~'
    )
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

// ── HTML Entity Encoding ────────────────────────────────────────────

/// Escape a string for safe inclusion in HTML text content.
/// Escapes: &, <, >, ", '
pub fn html_escape(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + input.len() / 8);
    for ch in input.chars() {
        match ch {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&#x27;"),
            c => result.push(c),
        }
    }
    result
}

/// Unescape HTML entities back to their original characters.
/// Handles named entities (&amp; &lt; &gt; &quot; &apos;) and
/// numeric entities (&#NNN; &#xHH;).
pub fn html_unescape(input: &str) -> Result<String, EscapeError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut pos = 0usize;

    while let Some(ch) = chars.next() {
        if ch != '&' {
            result.push(ch);
            pos += ch.len_utf8();
            continue;
        }
        let start_pos = pos;
        pos += 1; // skip '&'

        // Collect entity content up to ';'.
        let mut entity = String::new();
        let mut found_semi = false;
        while let Some(&c) = chars.peek() {
            if c == ';' {
                chars.next();
                pos += 1;
                found_semi = true;
                break;
            }
            entity.push(c);
            chars.next();
            pos += c.len_utf8();
            if entity.len() > 10 {
                break;
            }
        }

        if !found_semi {
            // Not a valid entity — emit literally.
            result.push('&');
            result.push_str(&entity);
            continue;
        }

        match entity.as_str() {
            "amp" => result.push('&'),
            "lt" => result.push('<'),
            "gt" => result.push('>'),
            "quot" => result.push('"'),
            "apos" => result.push('\''),
            "nbsp" => result.push('\u{00A0}'),
            s if s.starts_with('#') => {
                let numeric = &s[1..];
                let code = if let Some(hex) = numeric.strip_prefix('x').or_else(|| numeric.strip_prefix('X')) {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    numeric.parse::<u32>().ok()
                };
                match code.and_then(char::from_u32) {
                    Some(c) => result.push(c),
                    None => return Err(EscapeError::InvalidEscape(start_pos)),
                }
            }
            _ => {
                // Unknown entity — emit literally.
                result.push('&');
                result.push_str(&entity);
                result.push(';');
            }
        }
    }
    Ok(result)
}

// ── XML Escaping ────────────────────────────────────────────────────

/// Escape a string for safe inclusion in XML text content.
/// Escapes: &, <, >, ", '
pub fn xml_escape(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + input.len() / 8);
    for ch in input.chars() {
        match ch {
            '&' => result.push_str("&amp;"),
            '<' => result.push_str("&lt;"),
            '>' => result.push_str("&gt;"),
            '"' => result.push_str("&quot;"),
            '\'' => result.push_str("&apos;"),
            c => result.push(c),
        }
    }
    result
}

/// Unescape XML entities.
pub fn xml_unescape(input: &str) -> Result<String, EscapeError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut pos = 0usize;

    while let Some(ch) = chars.next() {
        if ch != '&' {
            result.push(ch);
            pos += ch.len_utf8();
            continue;
        }
        let start_pos = pos;
        pos += 1;

        let mut entity = String::new();
        let mut found_semi = false;
        while let Some(&c) = chars.peek() {
            if c == ';' {
                chars.next();
                pos += 1;
                found_semi = true;
                break;
            }
            entity.push(c);
            chars.next();
            pos += c.len_utf8();
            if entity.len() > 10 {
                break;
            }
        }

        if !found_semi {
            result.push('&');
            result.push_str(&entity);
            continue;
        }

        match entity.as_str() {
            "amp" => result.push('&'),
            "lt" => result.push('<'),
            "gt" => result.push('>'),
            "quot" => result.push('"'),
            "apos" => result.push('\''),
            s if s.starts_with('#') => {
                let numeric = &s[1..];
                let code = if let Some(hex) = numeric.strip_prefix('x').or_else(|| numeric.strip_prefix('X')) {
                    u32::from_str_radix(hex, 16).ok()
                } else {
                    numeric.parse::<u32>().ok()
                };
                match code.and_then(char::from_u32) {
                    Some(c) => result.push(c),
                    None => return Err(EscapeError::InvalidEscape(start_pos)),
                }
            }
            _ => {
                result.push('&');
                result.push_str(&entity);
                result.push(';');
            }
        }
    }
    Ok(result)
}

// ── C-Style Escaping ────────────────────────────────────────────────

/// Escape a string using C-style escape sequences.
/// Escapes: \n, \r, \t, \\, \", and non-printable bytes as \xHH.
pub fn c_escape(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + input.len() / 8);
    for ch in input.chars() {
        match ch {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\0' => result.push_str("\\0"),
            '\u{0007}' => result.push_str("\\a"),
            '\u{0008}' => result.push_str("\\b"),
            '\u{000C}' => result.push_str("\\f"),
            '\u{001B}' => result.push_str("\\e"),
            c if c < '\u{0020}' || c == '\u{007F}' => {
                for byte in c.to_string().bytes() {
                    result.push_str(&format!("\\x{byte:02x}"));
                }
            }
            c => result.push(c),
        }
    }
    result
}

/// Unescape a C-style escaped string.
pub fn c_unescape(input: &str) -> Result<String, EscapeError> {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().enumerate().peekable();

    while let Some((pos, ch)) = chars.next() {
        if ch != '\\' {
            result.push(ch);
            continue;
        }
        match chars.next() {
            Some((_, '\\')) => result.push('\\'),
            Some((_, '"')) => result.push('"'),
            Some((_, '\'')) => result.push('\''),
            Some((_, 'n')) => result.push('\n'),
            Some((_, 'r')) => result.push('\r'),
            Some((_, 't')) => result.push('\t'),
            Some((_, '0')) => result.push('\0'),
            Some((_, 'a')) => result.push('\u{0007}'),
            Some((_, 'b')) => result.push('\u{0008}'),
            Some((_, 'f')) => result.push('\u{000C}'),
            Some((_, 'e')) => result.push('\u{001B}'),
            Some((_, 'x')) => {
                let hex = read_c_hex(&mut chars, pos)?;
                result.push(hex as char);
            }
            Some(_) => return Err(EscapeError::InvalidEscape(pos)),
            None => return Err(EscapeError::UnexpectedEof),
        }
    }
    Ok(result)
}

fn read_c_hex(
    chars: &mut std::iter::Peekable<std::iter::Enumerate<std::str::Chars<'_>>>,
    pos: usize,
) -> Result<u8, EscapeError> {
    let mut hex = String::with_capacity(2);
    for _ in 0..2 {
        match chars.next() {
            Some((_, c)) if c.is_ascii_hexdigit() => hex.push(c),
            _ => return Err(EscapeError::InvalidEscape(pos)),
        }
    }
    u8::from_str_radix(&hex, 16).map_err(|_| EscapeError::InvalidEscape(pos))
}

// ── Shell Escaping ──────────────────────────────────────────────────

/// Escape a string for safe use in a POSIX shell command.
/// Wraps the string in single quotes if needed, escaping embedded single quotes.
pub fn shell_escape(input: &str) -> String {
    if input.is_empty() {
        return "''".to_string();
    }
    // If the string contains only safe characters, return as-is.
    let safe = input.bytes().all(|b| matches!(b,
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'
        | b'-' | b'_' | b'.' | b'/' | b':' | b',' | b'+' | b'='
    ));
    if safe {
        return input.to_string();
    }
    // Wrap in single quotes, escaping any embedded single quotes as '\''
    let mut result = String::with_capacity(input.len() + 4);
    result.push('\'');
    for ch in input.chars() {
        if ch == '\'' {
            result.push_str("'\\''");
        } else {
            result.push(ch);
        }
    }
    result.push('\'');
    result
}

/// Unescape a shell-escaped string (handles single-quoted and double-quoted).
pub fn shell_unescape(input: &str) -> Result<String, EscapeError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(String::new());
    }

    // Simple single-quoted string.
    if trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        // In single-quoted shell strings, no escaping happens — but we handle '\''
        return Ok(inner.replace("'\\''", "'"));
    }

    // Double-quoted string.
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        let mut result = String::with_capacity(inner.len());
        let mut chars = inner.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some(c @ ('"' | '\\' | '$' | '`')) => result.push(c),
                    Some('\n') => {} // line continuation
                    Some(c) => {
                        result.push('\\');
                        result.push(c);
                    }
                    None => return Err(EscapeError::UnexpectedEof),
                }
            } else {
                result.push(ch);
            }
        }
        return Ok(result);
    }

    // Unquoted — return as-is.
    Ok(trimmed.to_string())
}

// ── CSV Escaping ────────────────────────────────────────────────────

/// Escape a string for inclusion in a CSV field (RFC 4180).
/// Wraps in double quotes if the field contains commas, quotes, or newlines.
pub fn csv_escape(input: &str) -> String {
    let needs_quoting = input.contains(',')
        || input.contains('"')
        || input.contains('\n')
        || input.contains('\r');

    if !needs_quoting {
        return input.to_string();
    }

    let mut result = String::with_capacity(input.len() + 4);
    result.push('"');
    for ch in input.chars() {
        if ch == '"' {
            result.push_str("\"\"");
        } else {
            result.push(ch);
        }
    }
    result.push('"');
    result
}

/// Unescape a CSV field (RFC 4180). Removes surrounding quotes and
/// un-doubles escaped quotes.
pub fn csv_unescape(input: &str) -> Result<String, EscapeError> {
    let trimmed = input.trim();
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        let inner = &trimmed[1..trimmed.len() - 1];
        Ok(inner.replace("\"\"", "\""))
    } else {
        Ok(trimmed.to_string())
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── JSON ────────────────────────────────────────────────────────

    #[test]
    fn json_escape_basic() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape("he\"llo"), "he\\\"llo");
        assert_eq!(json_escape("back\\slash"), "back\\\\slash");
    }

    #[test]
    fn json_escape_control_chars() {
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("a\rb"), "a\\rb");
        assert_eq!(json_escape("a\tb"), "a\\tb");
        assert_eq!(json_escape("\u{0000}"), "\\u0000");
    }

    #[test]
    fn json_roundtrip() {
        let original = "Hello \"world\"\nnew\tline\u{0000}\u{001F}";
        let escaped = json_escape(original);
        let unescaped = json_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn json_unescape_unicode() {
        assert_eq!(json_unescape("\\u0041").unwrap(), "A");
        // Surrogate pair for U+1F600 (grinning face)
        assert_eq!(json_unescape("\\uD83D\\uDE00").unwrap(), "\u{1F600}");
    }

    #[test]
    fn json_unescape_invalid() {
        assert!(json_unescape("\\q").is_err());
        assert!(json_unescape("\\").is_err());
    }

    // ── URL ─────────────────────────────────────────────────────────

    #[test]
    fn url_encode_basic() {
        assert_eq!(url_encode("hello"), "hello");
        assert_eq!(url_encode("hello world"), "hello%20world");
        assert_eq!(url_encode("a&b=c"), "a%26b%3Dc");
    }

    #[test]
    fn url_roundtrip() {
        let original = "Hello World! @#$%^&*()";
        let encoded = url_encode(original);
        let decoded = url_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn url_decode_plus_as_space() {
        assert_eq!(url_decode("hello+world").unwrap(), "hello world");
    }

    #[test]
    fn url_encode_utf8() {
        let encoded = url_encode("\u{00E9}"); // e-acute
        assert_eq!(encoded, "%C3%A9");
        let decoded = url_decode(&encoded).unwrap();
        assert_eq!(decoded, "\u{00E9}");
    }

    #[test]
    fn url_decode_invalid() {
        assert!(url_decode("%GG").is_err());
        assert!(url_decode("%A").is_err());
    }

    // ── HTML ────────────────────────────────────────────────────────

    #[test]
    fn html_escape_basic() {
        assert_eq!(html_escape("<script>alert('xss')</script>"),
                   "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;");
    }

    #[test]
    fn html_escape_amp() {
        assert_eq!(html_escape("a & b"), "a &amp; b");
    }

    #[test]
    fn html_roundtrip() {
        let original = "<div class=\"test\">a & b's</div>";
        let escaped = html_escape(original);
        let unescaped = html_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn html_unescape_numeric() {
        assert_eq!(html_unescape("&#65;").unwrap(), "A");
        assert_eq!(html_unescape("&#x41;").unwrap(), "A");
        assert_eq!(html_unescape("&nbsp;").unwrap(), "\u{00A0}");
    }

    // ── XML ─────────────────────────────────────────────────────────

    #[test]
    fn xml_escape_basic() {
        assert_eq!(xml_escape("a < b & c > d"), "a &lt; b &amp; c &gt; d");
        assert_eq!(xml_escape("say \"hello\""), "say &quot;hello&quot;");
        assert_eq!(xml_escape("it's"), "it&apos;s");
    }

    #[test]
    fn xml_roundtrip() {
        let original = "<tag attr=\"val\" other='val2'>content & more</tag>";
        let escaped = xml_escape(original);
        let unescaped = xml_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    // ── C-Style ─────────────────────────────────────────────────────

    #[test]
    fn c_escape_basic() {
        assert_eq!(c_escape("hello\nworld"), "hello\\nworld");
        assert_eq!(c_escape("tab\there"), "tab\\there");
        assert_eq!(c_escape("null\0byte"), "null\\0byte");
    }

    #[test]
    fn c_escape_control() {
        assert_eq!(c_escape("\x01"), "\\x01");
        assert_eq!(c_escape("\x1B"), "\\e");
    }

    #[test]
    fn c_roundtrip() {
        let original = "hello\n\t\r\0world\\end\"quote";
        let escaped = c_escape(original);
        let unescaped = c_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn c_unescape_hex() {
        assert_eq!(c_unescape("\\x41").unwrap(), "A");
        assert_eq!(c_unescape("\\x00").unwrap(), "\0");
    }

    // ── Shell ───────────────────────────────────────────────────────

    #[test]
    fn shell_escape_safe() {
        assert_eq!(shell_escape("hello"), "hello");
        assert_eq!(shell_escape("/usr/bin/ls"), "/usr/bin/ls");
    }

    #[test]
    fn shell_escape_spaces() {
        assert_eq!(shell_escape("hello world"), "'hello world'");
    }

    #[test]
    fn shell_escape_single_quote() {
        assert_eq!(shell_escape("it's"), "'it'\\''s'");
    }

    #[test]
    fn shell_escape_empty() {
        assert_eq!(shell_escape(""), "''");
    }

    #[test]
    fn shell_unescape_single() {
        assert_eq!(shell_unescape("'hello world'").unwrap(), "hello world");
    }

    #[test]
    fn shell_unescape_double() {
        assert_eq!(shell_unescape("\"hello \\\"world\\\"\"").unwrap(), "hello \"world\"");
    }

    // ── CSV ─────────────────────────────────────────────────────────

    #[test]
    fn csv_escape_no_special() {
        assert_eq!(csv_escape("hello"), "hello");
    }

    #[test]
    fn csv_escape_comma() {
        assert_eq!(csv_escape("hello,world"), "\"hello,world\"");
    }

    #[test]
    fn csv_escape_quotes() {
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn csv_roundtrip() {
        let original = "field with, comma and \"quotes\"";
        let escaped = csv_escape(original);
        let unescaped = csv_unescape(&escaped).unwrap();
        assert_eq!(unescaped, original);
    }

    #[test]
    fn csv_escape_newline() {
        assert_eq!(csv_escape("line1\nline2"), "\"line1\nline2\"");
    }

    // ── Cross-Format ────────────────────────────────────────────────

    #[test]
    fn all_formats_handle_empty() {
        assert_eq!(json_escape(""), "");
        assert_eq!(url_encode(""), "");
        assert_eq!(html_escape(""), "");
        assert_eq!(xml_escape(""), "");
        assert_eq!(c_escape(""), "");
        assert_eq!(csv_escape(""), "");
    }

    #[test]
    fn json_unescape_slash() {
        assert_eq!(json_unescape("\\/").unwrap(), "/");
    }

    #[test]
    fn url_encode_preserves_unreserved() {
        let unreserved = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~";
        assert_eq!(url_encode(unreserved), unreserved);
    }
}
