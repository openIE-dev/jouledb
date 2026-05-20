//! Hex dump formatter — xxd-style hex+ASCII display with configurable layout.
//!
//! Generates human-readable hex dumps for binary data inspection, with
//! canonical format, grouping, binary mode, and reverse hex dump parsing.
//! Replaces JavaScript hex dump libraries with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during hex dump operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HexDumpError {
    #[error("invalid hex digit '{0}' at line {1}")]
    InvalidHexDigit(char, usize),
    #[error("invalid format at line {0}")]
    InvalidFormat(usize),
    #[error("line too short at line {0}")]
    LineTooShort(usize),
}

// ── Configuration ───────────────────────────────────────────────────

/// Grouping of bytes in the hex display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteGrouping {
    /// Each byte separate (xx xx xx).
    Single,
    /// Pairs of bytes (xxxx xxxx).
    Pair,
    /// Groups of 4 bytes (xxxxxxxx xxxxxxxx).
    Quad,
    /// Groups of 8 bytes (xxxxxxxxxxxxxxxx).
    Octet,
}

/// Display mode for the dump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    /// Hexadecimal (default).
    Hex,
    /// Binary (0s and 1s).
    Binary,
}

/// Configuration for hex dump formatting.
#[derive(Debug, Clone)]
pub struct HexDumpConfig {
    /// Bytes per line (default: 16).
    pub bytes_per_line: usize,
    /// Byte grouping in hex display.
    pub grouping: ByteGrouping,
    /// Whether to show the offset column.
    pub show_offset: bool,
    /// Whether to show the ASCII column.
    pub show_ascii: bool,
    /// Whether to use uppercase hex digits.
    pub uppercase: bool,
    /// Starting offset for display (added to line offsets).
    pub start_offset: usize,
    /// Display mode.
    pub mode: DisplayMode,
    /// Whether to include ANSI color annotations.
    pub colorize: bool,
}

impl Default for HexDumpConfig {
    fn default() -> Self {
        Self {
            bytes_per_line: 16,
            grouping: ByteGrouping::Single,
            show_offset: true,
            show_ascii: true,
            uppercase: false,
            start_offset: 0,
            mode: DisplayMode::Hex,
            colorize: false,
        }
    }
}

impl HexDumpConfig {
    /// Create a canonical xxd-style configuration.
    pub fn canonical() -> Self {
        Self::default()
    }

    /// Create a compact configuration (no offset, no ASCII).
    pub fn compact() -> Self {
        Self {
            show_offset: false,
            show_ascii: false,
            ..Self::default()
        }
    }

    /// Create a binary display configuration.
    pub fn binary() -> Self {
        Self {
            bytes_per_line: 8,
            mode: DisplayMode::Binary,
            grouping: ByteGrouping::Single,
            ..Self::default()
        }
    }

    /// Set bytes per line.
    pub fn with_bytes_per_line(mut self, n: usize) -> Self {
        self.bytes_per_line = n.max(1);
        self
    }

    /// Set byte grouping.
    pub fn with_grouping(mut self, g: ByteGrouping) -> Self {
        self.grouping = g;
        self
    }

    /// Set uppercase hex.
    pub fn with_uppercase(mut self, u: bool) -> Self {
        self.uppercase = u;
        self
    }

    /// Set color annotations.
    pub fn with_color(mut self, c: bool) -> Self {
        self.colorize = c;
        self
    }

    /// Set start offset.
    pub fn with_start_offset(mut self, offset: usize) -> Self {
        self.start_offset = offset;
        self
    }
}

// ── ANSI Color Codes ────────────────────────────────────────────────

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_DIM: &str = "\x1b[2m";
const ANSI_CYAN: &str = "\x1b[36m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_YELLOW: &str = "\x1b[33m";
const ANSI_RED: &str = "\x1b[31m";

fn color_for_byte(byte: u8) -> &'static str {
    match byte {
        0 => ANSI_DIM,
        0x20..=0x7E => ANSI_GREEN,
        0xFF => ANSI_RED,
        _ => ANSI_YELLOW,
    }
}

// ── Hex Dump Formatting ─────────────────────────────────────────────

/// Format binary data as a hex dump string using the given configuration.
pub fn format_hex_dump(data: &[u8], config: &HexDumpConfig) -> String {
    if data.is_empty() {
        return String::new();
    }
    let bpl = config.bytes_per_line.max(1);
    let mut lines = Vec::new();

    for (chunk_idx, chunk) in data.chunks(bpl).enumerate() {
        let offset = config.start_offset + chunk_idx * bpl;
        let mut line = String::with_capacity(80);

        // Offset column.
        if config.show_offset {
            if config.colorize {
                line.push_str(ANSI_CYAN);
            }
            line.push_str(&format!("{offset:08x}"));
            if config.colorize {
                line.push_str(ANSI_RESET);
            }
            line.push_str("  ");
        }

        // Data columns.
        match config.mode {
            DisplayMode::Hex => format_hex_line(chunk, bpl, config, &mut line),
            DisplayMode::Binary => format_binary_line(chunk, bpl, config, &mut line),
        }

        // ASCII column.
        if config.show_ascii {
            // Pad hex column for short last line so ASCII column aligns.
            line.push_str(" |");
            for &byte in chunk {
                if config.colorize {
                    line.push_str(color_for_byte(byte));
                }
                if (0x20..=0x7E).contains(&byte) {
                    line.push(byte as char);
                } else {
                    line.push('.');
                }
                if config.colorize {
                    line.push_str(ANSI_RESET);
                }
            }
            line.push('|');
        }

        lines.push(line);
    }
    lines.join("\n")
}

fn format_hex_line(chunk: &[u8], bpl: usize, config: &HexDumpConfig, line: &mut String) {
    let group_size = match config.grouping {
        ByteGrouping::Single => 1,
        ByteGrouping::Pair => 2,
        ByteGrouping::Quad => 4,
        ByteGrouping::Octet => 8,
    };

    for (i, &byte) in chunk.iter().enumerate() {
        if config.colorize {
            line.push_str(color_for_byte(byte));
        }
        if config.uppercase {
            line.push_str(&format!("{byte:02X}"));
        } else {
            line.push_str(&format!("{byte:02x}"));
        }
        if config.colorize {
            line.push_str(ANSI_RESET);
        }
        // Add spacing between groups.
        if (i + 1) % group_size == 0 && i + 1 < bpl {
            line.push(' ');
        }
    }

    // Pad for short last line.
    for i in chunk.len()..bpl {
        line.push_str("  ");
        if (i + 1) % group_size == 0 && i + 1 < bpl {
            line.push(' ');
        }
    }
}

fn format_binary_line(chunk: &[u8], bpl: usize, config: &HexDumpConfig, line: &mut String) {
    for (i, &byte) in chunk.iter().enumerate() {
        if config.colorize {
            line.push_str(color_for_byte(byte));
        }
        line.push_str(&format!("{byte:08b}"));
        if config.colorize {
            line.push_str(ANSI_RESET);
        }
        if i + 1 < bpl {
            line.push(' ');
        }
    }
    // Pad for short last line.
    for i in chunk.len()..bpl {
        line.push_str("        ");
        if i + 1 < bpl {
            line.push(' ');
        }
    }
}

// ── Convenience Functions ───────────────────────────────────────────

/// Format binary data as a canonical hex dump (xxd-style).
pub fn hex_dump(data: &[u8]) -> String {
    format_hex_dump(data, &HexDumpConfig::canonical())
}

/// Format binary data as a compact hex dump (no offset, no ASCII).
pub fn hex_dump_compact(data: &[u8]) -> String {
    format_hex_dump(data, &HexDumpConfig::compact())
}

/// Format binary data as a binary dump (bits).
pub fn binary_dump(data: &[u8]) -> String {
    format_hex_dump(data, &HexDumpConfig::binary())
}

/// Format a single line of hex for a small byte slice.
pub fn hex_line(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 3);
    for (i, &byte) in data.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(&format!("{byte:02x}"));
    }
    result
}

/// Format bytes as a continuous hex string (no spaces).
pub fn hex_string(data: &[u8]) -> String {
    let mut result = String::with_capacity(data.len() * 2);
    for &byte in data {
        result.push_str(&format!("{byte:02x}"));
    }
    result
}

// ── Reverse Hex Dump ────────────────────────────────────────────────

/// Parse a hex string (continuous, no spaces) back to bytes.
pub fn from_hex_string(input: &str) -> Result<Vec<u8>, HexDumpError> {
    let input = input.trim();
    if input.len() % 2 != 0 {
        return Err(HexDumpError::InvalidFormat(0));
    }
    let mut result = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = parse_hex_digit(bytes[i], 0)?;
        let lo = parse_hex_digit(bytes[i + 1], 0)?;
        result.push((hi << 4) | lo);
    }
    Ok(result)
}

/// Parse a space-separated hex line back to bytes.
/// Example: "48 65 6c 6c 6f" -> b"Hello"
pub fn from_hex_line(input: &str) -> Result<Vec<u8>, HexDumpError> {
    let mut result = Vec::new();
    for (idx, token) in input.split_whitespace().enumerate() {
        let bytes = token.as_bytes();
        if bytes.len() != 2 {
            return Err(HexDumpError::InvalidFormat(idx));
        }
        let hi = parse_hex_digit(bytes[0], idx)?;
        let lo = parse_hex_digit(bytes[1], idx)?;
        result.push((hi << 4) | lo);
    }
    Ok(result)
}

/// Parse a canonical hex dump back to bytes.
/// Accepts xxd-style format with offset, hex, and ASCII columns.
pub fn reverse_hex_dump(input: &str) -> Result<Vec<u8>, HexDumpError> {
    let mut result = Vec::new();
    for (line_num, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Try to find the hex data portion.
        // Canonical format: "OFFSET  HEX DATA  |ASCII|"
        let hex_part = extract_hex_portion(line, line_num)?;
        for token in hex_part.split_whitespace() {
            // Skip tokens that aren't valid hex pairs.
            if token.len() == 2
                && token.bytes().all(|b| b.is_ascii_hexdigit())
            {
                let hi = parse_hex_digit(token.as_bytes()[0], line_num)?;
                let lo = parse_hex_digit(token.as_bytes()[1], line_num)?;
                result.push((hi << 4) | lo);
            }
        }
    }
    Ok(result)
}

fn extract_hex_portion(line: &str, _line_num: usize) -> Result<&str, HexDumpError> {
    // Look for the offset separator "  " and ASCII separator " |".
    let start = if let Some(pos) = line.find("  ") {
        pos + 2
    } else {
        0
    };
    let end = if let Some(pos) = line.rfind(" |") {
        pos
    } else {
        line.len()
    };
    if start > end {
        return Ok(line);
    }
    Ok(&line[start..end])
}

fn parse_hex_digit(c: u8, line: usize) -> Result<u8, HexDumpError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(HexDumpError::InvalidHexDigit(c as char, line)),
    }
}

// ── Comparison & Analysis ───────────────────────────────────────────

/// Produce a side-by-side hex diff of two byte slices.
/// Lines that differ are marked with a `*` prefix.
pub fn hex_diff(a: &[u8], b: &[u8], bytes_per_line: usize) -> String {
    let bpl = bytes_per_line.max(1);
    let max_len = a.len().max(b.len());
    let mut lines = Vec::new();

    let line_count = (max_len + bpl - 1) / bpl;
    for i in 0..line_count {
        let start = i * bpl;
        let a_chunk = if start < a.len() {
            &a[start..(start + bpl).min(a.len())]
        } else {
            &[]
        };
        let b_chunk = if start < b.len() {
            &b[start..(start + bpl).min(b.len())]
        } else {
            &[]
        };

        let differs = a_chunk != b_chunk;
        let marker = if differs { "*" } else { " " };

        let a_hex = format_chunk_hex(a_chunk, bpl);
        let b_hex = format_chunk_hex(b_chunk, bpl);

        lines.push(format!(
            "{marker}{:08x}  {a_hex}  |  {b_hex}",
            start,
        ));
    }
    lines.join("\n")
}

fn format_chunk_hex(chunk: &[u8], bpl: usize) -> String {
    let mut s = String::with_capacity(bpl * 3);
    for (i, &byte) in chunk.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{byte:02x}"));
    }
    for i in chunk.len()..bpl {
        if i > 0 {
            s.push(' ');
        }
        s.push_str("  ");
    }
    s
}

/// Count the number of differing bytes between two slices.
pub fn byte_diff_count(a: &[u8], b: &[u8]) -> usize {
    let common = a.len().min(b.len());
    let mut count = 0;
    for i in 0..common {
        if a[i] != b[i] {
            count += 1;
        }
    }
    // Bytes present in one but not the other are all different.
    count + a.len().abs_diff(b.len())
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_dump_empty() {
        assert_eq!(hex_dump(&[]), "");
    }

    #[test]
    fn canonical_dump_hello() {
        let dump = hex_dump(b"Hello, World!");
        assert!(dump.contains("00000000"));
        assert!(dump.contains("48"));  // 'H'
        assert!(dump.contains("|Hello, World!|"));
    }

    #[test]
    fn canonical_dump_multi_line() {
        let data: Vec<u8> = (0..32).collect();
        let dump = hex_dump(&data);
        let line_count = dump.lines().count();
        assert_eq!(line_count, 2);
        assert!(dump.contains("00000000"));
        assert!(dump.contains("00000010"));
    }

    #[test]
    fn compact_dump() {
        let dump = hex_dump_compact(b"AB");
        // No offset, no ASCII
        assert!(!dump.contains("00000000"));
        assert!(!dump.contains('|'));
        assert!(dump.contains("41"));
        assert!(dump.contains("42"));
    }

    #[test]
    fn binary_dump_display() {
        let dump = binary_dump(&[0b10101010]);
        assert!(dump.contains("10101010"));
    }

    #[test]
    fn hex_line_format() {
        assert_eq!(hex_line(&[0x48, 0x65, 0x6c]), "48 65 6c");
    }

    #[test]
    fn hex_string_format() {
        assert_eq!(hex_string(&[0xDE, 0xAD, 0xBE, 0xEF]), "deadbeef");
    }

    #[test]
    fn uppercase_hex() {
        let config = HexDumpConfig::default().with_uppercase(true);
        let dump = format_hex_dump(&[0xAB, 0xCD], &config);
        assert!(dump.contains("AB"));
        assert!(dump.contains("CD"));
    }

    #[test]
    fn pair_grouping() {
        let config = HexDumpConfig::default()
            .with_grouping(ByteGrouping::Pair)
            .with_bytes_per_line(4);
        let dump = format_hex_dump(&[0x01, 0x02, 0x03, 0x04], &config);
        // Should group pairs: "0102 0304"
        assert!(dump.contains("0102 0304"));
    }

    #[test]
    fn quad_grouping() {
        let config = HexDumpConfig::default()
            .with_grouping(ByteGrouping::Quad)
            .with_bytes_per_line(8);
        let dump = format_hex_dump(&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], &config);
        assert!(dump.contains("01020304 05060708"));
    }

    #[test]
    fn start_offset() {
        let config = HexDumpConfig::default().with_start_offset(0x1000);
        let dump = format_hex_dump(&[0xFF], &config);
        assert!(dump.contains("00001000"));
    }

    #[test]
    fn from_hex_string_roundtrip() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let hex = hex_string(&data);
        let parsed = from_hex_string(&hex).unwrap();
        assert_eq!(parsed, data);
    }

    #[test]
    fn from_hex_line_roundtrip() {
        let data = [0x48, 0x65, 0x6c, 0x6c, 0x6f];
        let line = hex_line(&data);
        let parsed = from_hex_line(&line).unwrap();
        assert_eq!(parsed, data);
    }

    #[test]
    fn reverse_hex_dump_canonical() {
        let original = b"Hello!";
        let dump = hex_dump(original);
        let recovered = reverse_hex_dump(&dump).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn from_hex_string_invalid() {
        assert!(from_hex_string("GG").is_err());
        assert!(from_hex_string("ABC").is_err()); // odd length
    }

    #[test]
    fn hex_diff_identical() {
        let a = b"Hello";
        let diff = hex_diff(a, a, 16);
        // No lines should have the differ marker.
        for line in diff.lines() {
            assert!(line.starts_with(' '));
        }
    }

    #[test]
    fn hex_diff_different() {
        let a = b"Hello";
        let b = b"World";
        let diff = hex_diff(a, b, 16);
        assert!(diff.contains('*'));
    }

    #[test]
    fn hex_diff_different_lengths() {
        let a = b"Hi";
        let b = b"Hello";
        let diff = hex_diff(a, b, 16);
        assert!(diff.contains('*'));
    }

    #[test]
    fn byte_diff_count_same() {
        assert_eq!(byte_diff_count(b"hello", b"hello"), 0);
    }

    #[test]
    fn byte_diff_count_different() {
        assert_eq!(byte_diff_count(b"hello", b"hallo"), 1);
        assert_eq!(byte_diff_count(b"ab", b"abcd"), 2);
    }

    #[test]
    fn non_printable_ascii_as_dot() {
        let data = [0x00, 0x01, 0x41, 0x7F, 0xFF];
        let dump = hex_dump(&data);
        // 0x41 = 'A', rest should be dots in the ASCII column
        assert!(dump.contains("|..A..|"));
    }

    #[test]
    fn color_output_contains_ansi() {
        let config = HexDumpConfig::default().with_color(true);
        let dump = format_hex_dump(&[0x00, 0x41, 0xFF], &config);
        assert!(dump.contains("\x1b["));
    }

    #[test]
    fn short_last_line_padded() {
        let data: Vec<u8> = (0..20).collect();
        let dump = hex_dump(&data);
        let lines: Vec<&str> = dump.lines().collect();
        assert_eq!(lines.len(), 2);
        // The ASCII column of the last line should still have |...|
        assert!(lines[1].contains('|'));
    }
}
