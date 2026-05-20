//! Hexadecimal encoding, decoding, and formatting.
//!
//! Provides hex encode/decode, pretty-print hex dumps, hex with
//! separators, color parsing, and constant-time comparison.
//! Replaces npm hex / buffer.toString('hex') with pure Rust.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors from hex operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HexError {
    #[error("invalid hex character '{0}' at position {1}")]
    InvalidCharacter(char, usize),
    #[error("odd-length hex string")]
    OddLength,
    #[error("invalid hex color format")]
    InvalidColor,
}

// ── Constants ────────────────────────────────────────────────────────

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
const HEX_UPPER_TABLE: &[u8; 16] = b"0123456789ABCDEF";

// ── Encode ───────────────────────────────────────────────────────────

/// Encode bytes to lowercase hex string.
pub fn encode(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 2);
    for &b in data {
        result.push(HEX_LOWER[(b >> 4) as usize] as char);
        result.push(HEX_LOWER[(b & 0x0F) as usize] as char);
    }
    result
}

/// Encode bytes to uppercase hex string.
pub fn encode_upper(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 2);
    for &b in data {
        result.push(HEX_UPPER_TABLE[(b >> 4) as usize] as char);
        result.push(HEX_UPPER_TABLE[(b & 0x0F) as usize] as char);
    }
    result
}

// ── Decode ───────────────────────────────────────────────────────────

fn hex_val(c: u8, pos: usize) -> Result<u8, HexError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(HexError::InvalidCharacter(c as char, pos)),
    }
}

/// Decode hex string to bytes.
pub fn decode(input: &str) -> Result<Vec<u8>, HexError> {
    let bytes = input.as_bytes();
    if bytes.len() % 2 != 0 {
        return Err(HexError::OddLength);
    }
    let mut result = Vec::with_capacity(bytes.len() / 2);
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_val(bytes[i], i)?;
        let lo = hex_val(bytes[i + 1], i + 1)?;
        result.push((hi << 4) | lo);
    }
    Ok(result)
}

// ── Validation ───────────────────────────────────────────────────────

/// Validate that a string is a valid hex encoding.
pub fn is_valid(input: &str) -> bool {
    input.len() % 2 == 0
        && input.bytes().all(|b| matches!(b,
            b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F'
        ))
}

// ── Hex Dump ─────────────────────────────────────────────────────────

/// Pretty-print a hex dump like `xxd` or `hexdump -C`.
///
/// Format:
/// ```text
/// 00000000  48 65 6c 6c 6f 2c 20 57  6f 72 6c 64 21           |Hello, World!|
/// ```
pub fn hex_dump(data: &[u8]) -> String {
    let mut result = String::new();
    for (line_offset, chunk) in data.chunks(16).enumerate() {
        let offset = line_offset * 16;
        // Offset.
        result.push_str(&format!("{offset:08x}  "));
        // Hex bytes.
        for (i, &b) in chunk.iter().enumerate() {
            if i == 8 {
                result.push(' ');
            }
            result.push(HEX_LOWER[(b >> 4) as usize] as char);
            result.push(HEX_LOWER[(b & 0x0F) as usize] as char);
            result.push(' ');
        }
        // Pad if short line.
        let padding = 16 - chunk.len();
        for i in 0..padding {
            result.push_str("   ");
            if chunk.len() + i == 7 {
                result.push(' ');
            }
        }
        if chunk.len() <= 8 {
            result.push(' ');
        }
        // ASCII.
        result.push('|');
        for &b in chunk {
            if b.is_ascii_graphic() || b == b' ' {
                result.push(b as char);
            } else {
                result.push('.');
            }
        }
        result.push('|');
        result.push('\n');
    }
    result
}

// ── Hex with Separator ───────────────────────────────────────────────

/// Encode bytes to hex with a separator (e.g., "AA:BB:CC").
pub fn encode_with_separator(data: &[u8], sep: &str) -> String {
    let hex_bytes: Vec<String> = data
        .iter()
        .map(|b| {
            let mut s = String::with_capacity(2);
            s.push(HEX_UPPER_TABLE[(b >> 4) as usize] as char);
            s.push(HEX_UPPER_TABLE[(b & 0x0F) as usize] as char);
            s
        })
        .collect();
    hex_bytes.join(sep)
}

/// Decode hex with separator.
pub fn decode_with_separator(input: &str, sep: &str) -> Result<Vec<u8>, HexError> {
    let cleaned: String = if sep.is_empty() {
        input.to_string()
    } else {
        input.split(sep).collect()
    };
    decode(&cleaned)
}

// ── Hex Color ────────────────────────────────────────────────────────

/// A parsed RGB color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Parse a hex color string (#RRGGBB or #RGB shorthand).
pub fn parse_color(input: &str) -> Result<RgbColor, HexError> {
    let s = input.strip_prefix('#').unwrap_or(input);
    match s.len() {
        6 => {
            let bytes = decode(s)?;
            Ok(RgbColor {
                r: bytes[0],
                g: bytes[1],
                b: bytes[2],
            })
        }
        3 => {
            let bytes = s.as_bytes();
            let r = hex_val(bytes[0], 0)?;
            let g = hex_val(bytes[1], 1)?;
            let b = hex_val(bytes[2], 2)?;
            Ok(RgbColor {
                r: (r << 4) | r,
                g: (g << 4) | g,
                b: (b << 4) | b,
            })
        }
        _ => Err(HexError::InvalidColor),
    }
}

/// Format an RGB color as #RRGGBB.
pub fn format_color(color: &RgbColor) -> String {
    format!(
        "#{:02x}{:02x}{:02x}",
        color.r, color.g, color.b
    )
}

// ── Hex to Integer ───────────────────────────────────────────────────

/// Parse a hex string to u32.
pub fn to_u32(input: &str) -> Result<u32, HexError> {
    let s = input.strip_prefix("0x").or(input.strip_prefix("0X")).unwrap_or(input);
    let mut result = 0u32;
    for (i, &b) in s.as_bytes().iter().enumerate() {
        let val = hex_val(b, i)?;
        result = result
            .checked_shl(4)
            .ok_or(HexError::InvalidCharacter(b as char, i))?
            | val as u32;
    }
    Ok(result)
}

/// Parse a hex string to u64.
pub fn to_u64(input: &str) -> Result<u64, HexError> {
    let s = input.strip_prefix("0x").or(input.strip_prefix("0X")).unwrap_or(input);
    let mut result = 0u64;
    for (i, &b) in s.as_bytes().iter().enumerate() {
        let val = hex_val(b, i)?;
        result = result
            .checked_shl(4)
            .ok_or(HexError::InvalidCharacter(b as char, i))?
            | val as u64;
    }
    Ok(result)
}

/// Format u32 as hex string.
pub fn from_u32(value: u32) -> String {
    format!("{value:08x}")
}

/// Format u64 as hex string.
pub fn from_u64(value: u64) -> String {
    format!("{value:016x}")
}

// ── Constant-Time Comparison ─────────────────────────────────────────

/// Compare two hex strings in constant time (to prevent timing attacks).
///
/// Both strings must be the same length; returns `false` otherwise.
pub fn constant_time_eq(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.len() != b_bytes.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a_bytes.iter().zip(b_bytes.iter()) {
        // Normalize case: uppercase both.
        let xa = x.to_ascii_lowercase();
        let ya = y.to_ascii_lowercase();
        diff |= xa ^ ya;
    }
    diff == 0
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_basic() {
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"\xDE\xAD\xBE\xEF"), "deadbeef");
        assert_eq!(encode(b"Hello"), "48656c6c6f");
    }

    #[test]
    fn encode_upper_basic() {
        assert_eq!(encode_upper(b"\xDE\xAD"), "DEAD");
    }

    #[test]
    fn decode_basic() {
        assert_eq!(decode("deadbeef").unwrap(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(decode("DEADBEEF").unwrap(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn decode_odd_length() {
        assert_eq!(decode("abc"), Err(HexError::OddLength));
    }

    #[test]
    fn decode_invalid_char() {
        assert!(decode("zz").is_err());
    }

    #[test]
    fn roundtrip() {
        let data: Vec<u8> = (0..=255).collect();
        let encoded = encode(&data);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn validation() {
        assert!(is_valid("deadbeef"));
        assert!(is_valid("DEADBEEF"));
        assert!(is_valid("0123456789abcdef"));
        assert!(!is_valid("xyz"));
        assert!(!is_valid("abc")); // odd length
    }

    #[test]
    fn hex_dump_output() {
        let data = b"Hello, World!";
        let dump = hex_dump(data);
        assert!(dump.contains("48 65 6c 6c"));
        assert!(dump.contains("|Hello, World!|"));
        assert!(dump.starts_with("00000000"));
    }

    #[test]
    fn hex_dump_multiline() {
        let data: Vec<u8> = (0..32).collect();
        let dump = hex_dump(&data);
        let lines: Vec<&str> = dump.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("00000000"));
        assert!(lines[1].starts_with("00000010"));
    }

    #[test]
    fn separator_encoding() {
        let data = vec![0xAA, 0xBB, 0xCC];
        assert_eq!(encode_with_separator(&data, ":"), "AA:BB:CC");
        assert_eq!(encode_with_separator(&data, "-"), "AA-BB-CC");
    }

    #[test]
    fn separator_decoding() {
        let decoded = decode_with_separator("AA:BB:CC", ":").unwrap();
        assert_eq!(decoded, vec![0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn parse_color_rrggbb() {
        let color = parse_color("#FF8040").unwrap();
        assert_eq!(color, RgbColor { r: 255, g: 128, b: 64 });
    }

    #[test]
    fn parse_color_rgb_shorthand() {
        let color = parse_color("#FAB").unwrap();
        assert_eq!(color, RgbColor { r: 0xFF, g: 0xAA, b: 0xBB });
    }

    #[test]
    fn parse_color_no_hash() {
        let color = parse_color("FF0000").unwrap();
        assert_eq!(color, RgbColor { r: 255, g: 0, b: 0 });
    }

    #[test]
    fn format_color_roundtrip() {
        let color = RgbColor { r: 255, g: 128, b: 0 };
        let formatted = format_color(&color);
        assert_eq!(formatted, "#ff8000");
        let parsed = parse_color(&formatted).unwrap();
        assert_eq!(parsed, color);
    }

    #[test]
    fn hex_to_u32() {
        assert_eq!(to_u32("FF").unwrap(), 255);
        assert_eq!(to_u32("0xFF").unwrap(), 255);
        assert_eq!(to_u32("DEADBEEF").unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn hex_to_u64() {
        assert_eq!(to_u64("DEADBEEFCAFE0123").unwrap(), 0xDEADBEEFCAFE0123);
    }

    #[test]
    fn u32_u64_formatting() {
        assert_eq!(from_u32(0xDEADBEEF), "deadbeef");
        assert_eq!(from_u64(0x0123456789ABCDEF), "0123456789abcdef");
    }

    #[test]
    fn constant_time_comparison() {
        assert!(constant_time_eq("deadbeef", "DEADBEEF"));
        assert!(constant_time_eq("abc123", "ABC123"));
        assert!(!constant_time_eq("abc123", "abc124"));
        assert!(!constant_time_eq("abc", "abcd"));
    }

    #[test]
    fn parse_color_invalid() {
        assert!(parse_color("#GGHHII").is_err());
        assert!(parse_color("#12345").is_err());
    }
}
