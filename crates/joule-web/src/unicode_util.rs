//! Unicode utilities — UTF-8 encoding, codepoint handling, width estimation.
//!
//! Provides Unicode text analysis, normalization helpers, script detection,
//! case folding, and East Asian width estimation. Replaces JavaScript
//! unicode libraries with a pure Rust implementation.

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during Unicode operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnicodeError {
    #[error("invalid UTF-8 sequence at byte offset {0}")]
    InvalidUtf8(usize),
    #[error("invalid codepoint: U+{0:04X}")]
    InvalidCodepoint(u32),
    #[error("surrogate codepoint: U+{0:04X}")]
    SurrogateCodepoint(u32),
    #[error("codepoint out of range: {0}")]
    OutOfRange(u32),
}

// ── UTF-8 Encode/Decode ─────────────────────────────────────────────

/// Encode a single Unicode codepoint to UTF-8 bytes.
pub fn encode_codepoint(cp: u32) -> Result<Vec<u8>, UnicodeError> {
    validate_codepoint(cp)?;
    let ch = char::from_u32(cp).ok_or(UnicodeError::InvalidCodepoint(cp))?;
    let mut buf = [0u8; 4];
    let s = ch.encode_utf8(&mut buf);
    Ok(s.as_bytes().to_vec())
}

/// Decode the first UTF-8 codepoint from a byte slice.
/// Returns (codepoint, bytes_consumed).
pub fn decode_codepoint(data: &[u8]) -> Result<(u32, usize), UnicodeError> {
    if data.is_empty() {
        return Err(UnicodeError::InvalidUtf8(0));
    }
    let first = data[0];
    let (expected_len, initial_mask) = if first < 0x80 {
        (1, first as u32)
    } else if first & 0xE0 == 0xC0 {
        (2, (first & 0x1F) as u32)
    } else if first & 0xF0 == 0xE0 {
        (3, (first & 0x0F) as u32)
    } else if first & 0xF8 == 0xF0 {
        (4, (first & 0x07) as u32)
    } else {
        return Err(UnicodeError::InvalidUtf8(0));
    };

    if data.len() < expected_len {
        return Err(UnicodeError::InvalidUtf8(0));
    }

    let mut cp = initial_mask;
    for i in 1..expected_len {
        if data[i] & 0xC0 != 0x80 {
            return Err(UnicodeError::InvalidUtf8(i));
        }
        cp = (cp << 6) | (data[i] & 0x3F) as u32;
    }

    // Reject overlong encodings.
    let min_cp = match expected_len {
        1 => 0,
        2 => 0x80,
        3 => 0x800,
        4 => 0x10000,
        _ => unreachable!(),
    };
    if cp < min_cp {
        return Err(UnicodeError::InvalidUtf8(0));
    }
    // Reject surrogates.
    if (0xD800..=0xDFFF).contains(&cp) {
        return Err(UnicodeError::SurrogateCodepoint(cp));
    }
    if cp > 0x10FFFF {
        return Err(UnicodeError::OutOfRange(cp));
    }

    Ok((cp, expected_len))
}

/// Decode all codepoints from a UTF-8 byte slice.
pub fn decode_all_codepoints(data: &[u8]) -> Result<Vec<u32>, UnicodeError> {
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let (cp, len) = decode_codepoint(&data[offset..])?;
        result.push(cp);
        offset += len;
    }
    Ok(result)
}

/// Calculate the UTF-8 byte length for a codepoint.
pub fn utf8_byte_length(cp: u32) -> Result<usize, UnicodeError> {
    validate_codepoint(cp)?;
    Ok(if cp < 0x80 {
        1
    } else if cp < 0x800 {
        2
    } else if cp < 0x10000 {
        3
    } else {
        4
    })
}

// ── Codepoint Validation ────────────────────────────────────────────

/// Validate that a u32 is a valid Unicode codepoint.
pub fn validate_codepoint(cp: u32) -> Result<(), UnicodeError> {
    if cp > 0x10FFFF {
        return Err(UnicodeError::OutOfRange(cp));
    }
    if (0xD800..=0xDFFF).contains(&cp) {
        return Err(UnicodeError::SurrogateCodepoint(cp));
    }
    Ok(())
}

/// Check if a codepoint is a valid Unicode scalar value.
pub fn is_valid_codepoint(cp: u32) -> bool {
    validate_codepoint(cp).is_ok()
}

/// Check if a byte sequence is valid UTF-8.
pub fn is_valid_utf8(data: &[u8]) -> bool {
    std::str::from_utf8(data).is_ok()
}

// ── Grapheme Cluster Approximation ──────────────────────────────────

/// Approximate grapheme cluster boundaries in a string.
/// This is a simplified heuristic — it handles common cases like
/// combining marks following base characters but does not implement
/// the full UAX #29 algorithm.
///
/// Returns a list of (byte_offset, byte_length) for each cluster.
pub fn approx_grapheme_clusters(s: &str) -> Vec<(usize, usize)> {
    if s.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut cluster_start = 0;
    let mut chars = s.char_indices().peekable();

    while let Some((idx, ch)) = chars.next() {
        let _ = idx;
        // Peek at the next character.
        let next_is_combining = chars.peek().is_some_and(|&(_, next_ch)| {
            is_combining_mark(next_ch) || is_emoji_modifier(next_ch)
        });

        if !next_is_combining {
            // End of cluster.
            let char_end = idx + ch.len_utf8();
            clusters.push((cluster_start, char_end - cluster_start));
            if let Some(&(next_idx, _)) = chars.peek() {
                cluster_start = next_idx;
            }
        }
    }
    clusters
}

/// Count the approximate number of grapheme clusters in a string.
pub fn approx_grapheme_count(s: &str) -> usize {
    approx_grapheme_clusters(s).len()
}

fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    // Combining Diacritical Marks (0300-036F)
    // Combining Diacritical Marks Extended (1AB0-1AFF)
    // Combining Diacritical Marks Supplement (1DC0-1DFF)
    // Combining Half Marks (FE20-FE2F)
    // General Category Mn, Mc, Me ranges (simplified)
    (0x0300..=0x036F).contains(&cp)
        || (0x0483..=0x0489).contains(&cp)
        || (0x0591..=0x05BD).contains(&cp)
        || (0x05BF..=0x05BF).contains(&cp)
        || (0x05C1..=0x05C2).contains(&cp)
        || (0x05C4..=0x05C5).contains(&cp)
        || (0x05C7..=0x05C7).contains(&cp)
        || (0x0610..=0x061A).contains(&cp)
        || (0x064B..=0x065F).contains(&cp)
        || (0x0670..=0x0670).contains(&cp)
        || (0x06D6..=0x06DC).contains(&cp)
        || (0x06DF..=0x06E4).contains(&cp)
        || (0x06E7..=0x06E8).contains(&cp)
        || (0x06EA..=0x06ED).contains(&cp)
        || (0x0711..=0x0711).contains(&cp)
        || (0x0730..=0x074A).contains(&cp)
        || (0x0900..=0x0903).contains(&cp)
        || (0x093A..=0x094F).contains(&cp)
        || (0x0951..=0x0957).contains(&cp)
        || (0x0962..=0x0963).contains(&cp)
        || (0x0981..=0x0983).contains(&cp)
        || (0x09BC..=0x09BC).contains(&cp)
        || (0x09BE..=0x09CD).contains(&cp)
        || (0x0A01..=0x0A03).contains(&cp)
        || (0x0A3C..=0x0A51).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE00..=0xFE0F).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

fn is_emoji_modifier(ch: char) -> bool {
    let cp = ch as u32;
    // Emoji skin tone modifiers and ZWJ
    (0x1F3FB..=0x1F3FF).contains(&cp)
        || cp == 0x200D  // ZWJ
        || cp == 0xFE0F  // Variation Selector-16 (emoji presentation)
}

// ── Simplified Normalization ────────────────────────────────────────
//
// Full NFC/NFD requires the full Unicode decomposition tables.
// Here we provide a simplified form for the most common Latin diacritics.

/// Canonical decomposition table for common Latin characters.
/// Maps composed character -> (base, combining mark).
const DECOMPOSITIONS: &[(char, char, char)] = &[
    ('\u{00C0}', 'A', '\u{0300}'), // A grave
    ('\u{00C1}', 'A', '\u{0301}'), // A acute
    ('\u{00C2}', 'A', '\u{0302}'), // A circumflex
    ('\u{00C3}', 'A', '\u{0303}'), // A tilde
    ('\u{00C4}', 'A', '\u{0308}'), // A diaeresis
    ('\u{00C5}', 'A', '\u{030A}'), // A ring above
    ('\u{00C7}', 'C', '\u{0327}'), // C cedilla
    ('\u{00C8}', 'E', '\u{0300}'), // E grave
    ('\u{00C9}', 'E', '\u{0301}'), // E acute
    ('\u{00CA}', 'E', '\u{0302}'), // E circumflex
    ('\u{00CB}', 'E', '\u{0308}'), // E diaeresis
    ('\u{00CC}', 'I', '\u{0300}'), // I grave
    ('\u{00CD}', 'I', '\u{0301}'), // I acute
    ('\u{00CE}', 'I', '\u{0302}'), // I circumflex
    ('\u{00CF}', 'I', '\u{0308}'), // I diaeresis
    ('\u{00D1}', 'N', '\u{0303}'), // N tilde
    ('\u{00D2}', 'O', '\u{0300}'), // O grave
    ('\u{00D3}', 'O', '\u{0301}'), // O acute
    ('\u{00D4}', 'O', '\u{0302}'), // O circumflex
    ('\u{00D5}', 'O', '\u{0303}'), // O tilde
    ('\u{00D6}', 'O', '\u{0308}'), // O diaeresis
    ('\u{00D9}', 'U', '\u{0300}'), // U grave
    ('\u{00DA}', 'U', '\u{0301}'), // U acute
    ('\u{00DB}', 'U', '\u{0302}'), // U circumflex
    ('\u{00DC}', 'U', '\u{0308}'), // U diaeresis
    ('\u{00DD}', 'Y', '\u{0301}'), // Y acute
    ('\u{00E0}', 'a', '\u{0300}'), // a grave
    ('\u{00E1}', 'a', '\u{0301}'), // a acute
    ('\u{00E2}', 'a', '\u{0302}'), // a circumflex
    ('\u{00E3}', 'a', '\u{0303}'), // a tilde
    ('\u{00E4}', 'a', '\u{0308}'), // a diaeresis
    ('\u{00E5}', 'a', '\u{030A}'), // a ring above
    ('\u{00E7}', 'c', '\u{0327}'), // c cedilla
    ('\u{00E8}', 'e', '\u{0300}'), // e grave
    ('\u{00E9}', 'e', '\u{0301}'), // e acute
    ('\u{00EA}', 'e', '\u{0302}'), // e circumflex
    ('\u{00EB}', 'e', '\u{0308}'), // e diaeresis
    ('\u{00EC}', 'i', '\u{0300}'), // i grave
    ('\u{00ED}', 'i', '\u{0301}'), // i acute
    ('\u{00EE}', 'i', '\u{0302}'), // i circumflex
    ('\u{00EF}', 'i', '\u{0308}'), // i diaeresis
    ('\u{00F1}', 'n', '\u{0303}'), // n tilde
    ('\u{00F2}', 'o', '\u{0300}'), // o grave
    ('\u{00F3}', 'o', '\u{0301}'), // o acute
    ('\u{00F4}', 'o', '\u{0302}'), // o circumflex
    ('\u{00F5}', 'o', '\u{0303}'), // o tilde
    ('\u{00F6}', 'o', '\u{0308}'), // o diaeresis
    ('\u{00F9}', 'u', '\u{0300}'), // u grave
    ('\u{00FA}', 'u', '\u{0301}'), // u acute
    ('\u{00FB}', 'u', '\u{0302}'), // u circumflex
    ('\u{00FC}', 'u', '\u{0308}'), // u diaeresis
    ('\u{00FD}', 'y', '\u{0301}'), // y acute
    ('\u{00FF}', 'y', '\u{0308}'), // y diaeresis
];

/// Simplified NFD-like decomposition for common Latin diacritics.
/// Decomposes precomposed characters into base + combining mark.
pub fn decompose_simple(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2);
    for ch in s.chars() {
        let mut found = false;
        for &(composed, base, combining) in DECOMPOSITIONS {
            if ch == composed {
                result.push(base);
                result.push(combining);
                found = true;
                break;
            }
        }
        if !found {
            result.push(ch);
        }
    }
    result
}

/// Simplified NFC-like composition for common Latin diacritics.
/// Composes base + combining mark sequences back to precomposed form.
pub fn compose_simple(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut result = String::with_capacity(s.len());
    let mut i = 0;

    while i < chars.len() {
        if i + 1 < chars.len() {
            let base = chars[i];
            let combining = chars[i + 1];
            let mut composed_found = false;
            for &(composed, b, c) in DECOMPOSITIONS {
                if base == b && combining == c {
                    result.push(composed);
                    composed_found = true;
                    i += 2;
                    break;
                }
            }
            if composed_found {
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

// ── Case Folding ────────────────────────────────────────────────────

/// Simple case folding — maps characters to their lowercase form.
/// This handles basic Latin, extended Latin, Greek, and Cyrillic.
pub fn case_fold(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for ch in s.chars() {
        // Special cases first.
        match ch {
            '\u{00DF}' => result.push_str("ss"), // German sharp s
            '\u{0130}' => {
                result.push('i'); // Turkish capital I with dot
                result.push('\u{0307}');
            }
            _ => {
                // Use standard lowercase conversion.
                for c in ch.to_lowercase() {
                    result.push(c);
                }
            }
        }
    }
    result
}

/// Case-insensitive string comparison using case folding.
pub fn case_fold_eq(a: &str, b: &str) -> bool {
    case_fold(a) == case_fold(b)
}

// ── Script Detection ────────────────────────────────────────────────

/// Unicode script categories (simplified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Latin,
    Greek,
    Cyrillic,
    Arabic,
    Hebrew,
    Devanagari,
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Thai,
    Common,
    Unknown,
}

/// Detect the primary script of a codepoint.
pub fn detect_script(cp: u32) -> Script {
    match cp {
        0x0041..=0x005A | 0x0061..=0x007A => Script::Latin,
        0x00C0..=0x024F => Script::Latin,  // Latin Extended
        0x1E00..=0x1EFF => Script::Latin,  // Latin Extended Additional
        0x0370..=0x03FF => Script::Greek,
        0x1F00..=0x1FFF => Script::Greek,  // Greek Extended
        0x0400..=0x04FF => Script::Cyrillic,
        0x0500..=0x052F => Script::Cyrillic, // Cyrillic Supplement
        0x0600..=0x06FF => Script::Arabic,
        0x0750..=0x077F => Script::Arabic,  // Arabic Supplement
        0xFB50..=0xFDFF => Script::Arabic,  // Arabic Presentation Forms
        0x0590..=0x05FF => Script::Hebrew,
        0xFB1D..=0xFB4F => Script::Hebrew,  // Hebrew Presentation Forms
        0x0900..=0x097F => Script::Devanagari,
        0x4E00..=0x9FFF => Script::Han,     // CJK Unified Ideographs
        0x3400..=0x4DBF => Script::Han,     // CJK Extension A
        0x20000..=0x2A6DF => Script::Han,   // CJK Extension B
        0x2A700..=0x2B73F => Script::Han,   // CJK Extension C
        0xF900..=0xFAFF => Script::Han,     // CJK Compatibility
        0x3040..=0x309F => Script::Hiragana,
        0x30A0..=0x30FF => Script::Katakana,
        0x31F0..=0x31FF => Script::Katakana, // Katakana Phonetic Extensions
        0xAC00..=0xD7AF => Script::Hangul,   // Hangul Syllables
        0x1100..=0x11FF => Script::Hangul,   // Hangul Jamo
        0x3130..=0x318F => Script::Hangul,   // Hangul Compat Jamo
        0x0E00..=0x0E7F => Script::Thai,
        0x0020..=0x0040 | 0x005B..=0x0060 | 0x007B..=0x007E => Script::Common,
        0x2000..=0x206F => Script::Common, // General Punctuation
        0x3000..=0x303F => Script::Common, // CJK Symbols
        _ => Script::Unknown,
    }
}

/// Detect the dominant script of a string.
pub fn detect_dominant_script(s: &str) -> Script {
    let mut counts = [0u32; 13]; // One per Script variant

    for ch in s.chars() {
        let script = detect_script(ch as u32);
        let idx = match script {
            Script::Latin => 0,
            Script::Greek => 1,
            Script::Cyrillic => 2,
            Script::Arabic => 3,
            Script::Hebrew => 4,
            Script::Devanagari => 5,
            Script::Han => 6,
            Script::Hiragana => 7,
            Script::Katakana => 8,
            Script::Hangul => 9,
            Script::Thai => 10,
            Script::Common => 11,
            Script::Unknown => 12,
        };
        counts[idx] += 1;
    }

    // Find the most common non-Common, non-Unknown script.
    let mut best_idx = 12; // Unknown
    let mut best_count = 0;
    for (i, &count) in counts.iter().enumerate() {
        if i == 11 || i == 12 {
            continue; // Skip Common and Unknown
        }
        if count > best_count {
            best_count = count;
            best_idx = i;
        }
    }

    if best_count == 0 {
        return Script::Common;
    }

    match best_idx {
        0 => Script::Latin,
        1 => Script::Greek,
        2 => Script::Cyrillic,
        3 => Script::Arabic,
        4 => Script::Hebrew,
        5 => Script::Devanagari,
        6 => Script::Han,
        7 => Script::Hiragana,
        8 => Script::Katakana,
        9 => Script::Hangul,
        10 => Script::Thai,
        _ => Script::Unknown,
    }
}

// ── Bidirectional Text Basics ───────────────────────────────────────

/// Bidirectional character category (simplified).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BidiClass {
    /// Left-to-right (Latin, Greek, CJK, etc.)
    LeftToRight,
    /// Right-to-left (Arabic, Hebrew)
    RightToLeft,
    /// Neutral/weak (numbers, punctuation, whitespace)
    Neutral,
}

/// Get the simplified bidirectional class of a character.
pub fn bidi_class(ch: char) -> BidiClass {
    let cp = ch as u32;
    match cp {
        // Arabic
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => {
            BidiClass::RightToLeft
        }
        // Hebrew
        0x0590..=0x05FF | 0xFB1D..=0xFB4F => BidiClass::RightToLeft,
        // RTL markers
        0x200F | 0x202B | 0x202E | 0x2067 => BidiClass::RightToLeft,
        // LTR Latin, Greek, Cyrillic, CJK, etc.
        0x0041..=0x005A | 0x0061..=0x007A => BidiClass::LeftToRight,
        0x00C0..=0x024F => BidiClass::LeftToRight,
        0x0370..=0x03FF => BidiClass::LeftToRight,
        0x0400..=0x04FF => BidiClass::LeftToRight,
        0x4E00..=0x9FFF => BidiClass::LeftToRight,
        0x3040..=0x30FF => BidiClass::LeftToRight,
        0xAC00..=0xD7AF => BidiClass::LeftToRight,
        // LTR marker
        0x200E | 0x202A | 0x202D | 0x2066 => BidiClass::LeftToRight,
        _ => BidiClass::Neutral,
    }
}

/// Check if a string contains any right-to-left characters.
pub fn has_rtl(s: &str) -> bool {
    s.chars().any(|ch| bidi_class(ch) == BidiClass::RightToLeft)
}

/// Determine the base direction of a string (first strong character).
pub fn base_direction(s: &str) -> BidiClass {
    for ch in s.chars() {
        let cls = bidi_class(ch);
        if cls != BidiClass::Neutral {
            return cls;
        }
    }
    BidiClass::LeftToRight
}

// ── Width Estimation (East Asian) ───────────────────────────────────

/// Estimate the display width of a character in a monospace terminal.
/// Returns 0 for combining marks, 2 for fullwidth/wide characters, 1 otherwise.
pub fn char_width(ch: char) -> usize {
    let cp = ch as u32;

    // Zero-width characters.
    if is_combining_mark(ch)
        || cp == 0x200B  // Zero Width Space
        || cp == 0x200C  // ZWNJ
        || cp == 0x200D  // ZWJ
        || cp == 0xFEFF  // BOM
        || (0xFE00..=0xFE0F).contains(&cp)  // Variation Selectors
        || (0xE0100..=0xE01EF).contains(&cp) // Variation Selectors Supplement
    {
        return 0;
    }

    // Fullwidth and wide characters.
    if is_east_asian_wide(cp) {
        return 2;
    }

    1
}

/// Estimate the display width of a string in a monospace terminal.
pub fn string_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

fn is_east_asian_wide(cp: u32) -> bool {
    // CJK Unified Ideographs
    (0x4E00..=0x9FFF).contains(&cp)
    // CJK Extension A
    || (0x3400..=0x4DBF).contains(&cp)
    // CJK Extension B
    || (0x20000..=0x2A6DF).contains(&cp)
    // CJK Compatibility Ideographs
    || (0xF900..=0xFAFF).contains(&cp)
    // Hangul Syllables
    || (0xAC00..=0xD7AF).contains(&cp)
    // Fullwidth Forms
    || (0xFF01..=0xFF60).contains(&cp)
    || (0xFFE0..=0xFFE6).contains(&cp)
    // CJK Symbols and Punctuation
    || (0x3000..=0x303F).contains(&cp)
    // Hiragana and Katakana
    || (0x3040..=0x309F).contains(&cp)
    || (0x30A0..=0x30FF).contains(&cp)
    // Enclosed CJK
    || (0x3200..=0x32FF).contains(&cp)
    // CJK Compatibility
    || (0x3300..=0x33FF).contains(&cp)
    // Enclosed Ideographic Supplement
    || (0x1F200..=0x1F2FF).contains(&cp)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── UTF-8 Encode/Decode ─────────────────────────────────────────

    #[test]
    fn encode_ascii() {
        assert_eq!(encode_codepoint(0x41).unwrap(), [0x41]); // 'A'
    }

    #[test]
    fn encode_two_byte() {
        assert_eq!(encode_codepoint(0xE9).unwrap(), [0xC3, 0xA9]); // e-acute
    }

    #[test]
    fn encode_three_byte() {
        assert_eq!(encode_codepoint(0x4E16).unwrap(), [0xE4, 0xB8, 0x96]); // CJK character
    }

    #[test]
    fn encode_four_byte() {
        let bytes = encode_codepoint(0x1F600).unwrap(); // grinning face emoji
        assert_eq!(bytes.len(), 4);
    }

    #[test]
    fn decode_roundtrip() {
        for cp in [0x41, 0xE9, 0x4E16, 0x1F600] {
            let encoded = encode_codepoint(cp).unwrap();
            let (decoded, len) = decode_codepoint(&encoded).unwrap();
            assert_eq!(decoded, cp);
            assert_eq!(len, encoded.len());
        }
    }

    #[test]
    fn decode_all_hello() {
        let codepoints = decode_all_codepoints("Hello".as_bytes()).unwrap();
        assert_eq!(codepoints, [0x48, 0x65, 0x6C, 0x6C, 0x6F]);
    }

    #[test]
    fn utf8_byte_length_check() {
        assert_eq!(utf8_byte_length(0x41).unwrap(), 1);
        assert_eq!(utf8_byte_length(0xE9).unwrap(), 2);
        assert_eq!(utf8_byte_length(0x4E16).unwrap(), 3);
        assert_eq!(utf8_byte_length(0x1F600).unwrap(), 4);
    }

    // ── Validation ──────────────────────────────────────────────────

    #[test]
    fn validate_valid_codepoints() {
        assert!(is_valid_codepoint(0));
        assert!(is_valid_codepoint(0x41));
        assert!(is_valid_codepoint(0x10FFFF));
    }

    #[test]
    fn validate_invalid_codepoints() {
        assert!(!is_valid_codepoint(0xD800));  // surrogate
        assert!(!is_valid_codepoint(0xDFFF));  // surrogate
        assert!(!is_valid_codepoint(0x110000)); // out of range
    }

    #[test]
    fn encode_surrogate_fails() {
        assert!(matches!(
            encode_codepoint(0xD800),
            Err(UnicodeError::SurrogateCodepoint(0xD800))
        ));
    }

    #[test]
    fn decode_overlong_rejected() {
        // Overlong encoding of U+0041 as 2 bytes: C1 81
        assert!(decode_codepoint(&[0xC1, 0x81]).is_err());
    }

    #[test]
    fn valid_utf8_check() {
        assert!(is_valid_utf8(b"Hello"));
        assert!(is_valid_utf8("\u{00E9}".as_bytes()));
        assert!(!is_valid_utf8(&[0xFF, 0xFE]));
    }

    // ── Grapheme Clusters ───────────────────────────────────────────

    #[test]
    fn grapheme_ascii() {
        let clusters = approx_grapheme_clusters("abc");
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn grapheme_combining() {
        // e + combining acute = one cluster
        let s = "e\u{0301}";
        let clusters = approx_grapheme_clusters(s);
        assert_eq!(clusters.len(), 1);
    }

    #[test]
    fn grapheme_count_basic() {
        assert_eq!(approx_grapheme_count("hello"), 5);
        assert_eq!(approx_grapheme_count(""), 0);
    }

    // ── Normalization ───────────────────────────────────────────────

    #[test]
    fn decompose_e_acute() {
        let composed = "\u{00E9}"; // e-acute precomposed
        let decomposed = decompose_simple(composed);
        assert_eq!(decomposed, "e\u{0301}");
    }

    #[test]
    fn compose_e_acute() {
        let decomposed = "e\u{0301}";
        let composed = compose_simple(decomposed);
        assert_eq!(composed, "\u{00E9}");
    }

    #[test]
    fn decompose_compose_roundtrip() {
        let original = "\u{00C9}\u{00F1}\u{00FC}"; // ENu
        let decomposed = decompose_simple(original);
        let recomposed = compose_simple(&decomposed);
        assert_eq!(recomposed, original);
    }

    #[test]
    fn decompose_plain_ascii_unchanged() {
        assert_eq!(decompose_simple("Hello"), "Hello");
    }

    // ── Case Folding ────────────────────────────────────────────────

    #[test]
    fn case_fold_basic() {
        assert_eq!(case_fold("Hello"), "hello");
        assert_eq!(case_fold("ABC"), "abc");
    }

    #[test]
    fn case_fold_german_sharp_s() {
        assert_eq!(case_fold("\u{00DF}"), "ss");
        assert!(case_fold_eq("stra\u{00DF}e", "STRASSE"));
    }

    #[test]
    fn case_fold_eq_basic() {
        assert!(case_fold_eq("Hello", "hello"));
        assert!(case_fold_eq("HELLO", "hello"));
        assert!(!case_fold_eq("hello", "world"));
    }

    // ── Script Detection ────────────────────────────────────────────

    #[test]
    fn detect_latin() {
        assert_eq!(detect_script(0x41), Script::Latin); // 'A'
        assert_eq!(detect_script(0x00E9), Script::Latin); // e-acute
    }

    #[test]
    fn detect_cjk() {
        assert_eq!(detect_script(0x4E16), Script::Han);
    }

    #[test]
    fn detect_arabic() {
        assert_eq!(detect_script(0x0627), Script::Arabic);
    }

    #[test]
    fn detect_dominant_script_english() {
        assert_eq!(detect_dominant_script("Hello, World!"), Script::Latin);
    }

    #[test]
    fn detect_dominant_script_japanese() {
        assert_eq!(detect_dominant_script("\u{3053}\u{3093}\u{306B}\u{3061}\u{306F}"), Script::Hiragana);
    }

    // ── Bidi ────────────────────────────────────────────────────────

    #[test]
    fn bidi_class_check() {
        assert_eq!(bidi_class('A'), BidiClass::LeftToRight);
        assert_eq!(bidi_class('\u{0627}'), BidiClass::RightToLeft); // Arabic Alef
        assert_eq!(bidi_class(' '), BidiClass::Neutral);
    }

    #[test]
    fn has_rtl_check() {
        assert!(!has_rtl("Hello"));
        assert!(has_rtl("\u{0627}\u{0644}\u{0639}")); // Arabic text
    }

    #[test]
    fn base_direction_check() {
        assert_eq!(base_direction("Hello"), BidiClass::LeftToRight);
        assert_eq!(base_direction("\u{0627}test"), BidiClass::RightToLeft);
        assert_eq!(base_direction("   "), BidiClass::LeftToRight); // default
    }

    // ── Width Estimation ────────────────────────────────────────────

    #[test]
    fn char_width_ascii() {
        assert_eq!(char_width('A'), 1);
        assert_eq!(char_width(' '), 1);
    }

    #[test]
    fn char_width_cjk() {
        assert_eq!(char_width('\u{4E16}'), 2); // CJK character
        assert_eq!(char_width('\u{3042}'), 2); // Hiragana
    }

    #[test]
    fn char_width_combining() {
        assert_eq!(char_width('\u{0301}'), 0); // Combining acute
    }

    #[test]
    fn string_width_mixed() {
        // "Hello" = 5 * 1 = 5
        assert_eq!(string_width("Hello"), 5);
        // Two CJK chars = 2 * 2 = 4
        assert_eq!(string_width("\u{4E16}\u{754C}"), 4);
    }

    #[test]
    fn string_width_empty() {
        assert_eq!(string_width(""), 0);
    }

    #[test]
    fn char_width_zwj() {
        assert_eq!(char_width('\u{200D}'), 0); // ZWJ
        assert_eq!(char_width('\u{200B}'), 0); // Zero Width Space
    }
}
