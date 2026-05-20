//! Input sanitization beyond XSS — SQL injection pattern detection, path traversal
//! prevention, null byte stripping, unicode normalization for security, homoglyph
//! detection, URL sanitization.
//!
//! Replaces validator, sanitize-html, and DOMPurify's non-HTML features with a
//! pure-Rust input security toolkit.

use std::collections::HashSet;

// ── Errors ─────────────────────────────────────────────────────

/// Input sanitization errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SanitizeError {
    /// SQL injection pattern detected.
    SqlInjection(String),
    /// Path traversal detected.
    PathTraversal(String),
    /// Null byte found.
    NullByte,
    /// Homoglyph attack detected.
    HomoglyphDetected { original: String, suspicious: String },
    /// Invalid URL.
    InvalidUrl(String),
    /// Input too long.
    InputTooLong { max: usize, got: usize },
}

impl std::fmt::Display for SanitizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SqlInjection(s) => write!(f, "SQL injection pattern detected: {s}"),
            Self::PathTraversal(s) => write!(f, "path traversal detected: {s}"),
            Self::NullByte => write!(f, "null byte in input"),
            Self::HomoglyphDetected { original, suspicious } => {
                write!(
                    f,
                    "homoglyph detected: '{suspicious}' in '{original}'"
                )
            }
            Self::InvalidUrl(s) => write!(f, "invalid URL: {s}"),
            Self::InputTooLong { max, got } => {
                write!(f, "input too long: {got} chars (max {max})")
            }
        }
    }
}

impl std::error::Error for SanitizeError {}

// ── Null Byte Stripping ────────────────────────────────────────

/// Strip null bytes from input.
pub fn strip_null_bytes(input: &str) -> String {
    input.replace('\0', "")
}

/// Check for null bytes in input.
pub fn has_null_bytes(input: &str) -> bool {
    input.contains('\0')
}

/// Validate and strip null bytes, returning error if found.
pub fn reject_null_bytes(input: &str) -> Result<&str, SanitizeError> {
    if has_null_bytes(input) {
        Err(SanitizeError::NullByte)
    } else {
        Ok(input)
    }
}

// ── SQL Injection Detection ────────────────────────────────────

/// SQL injection patterns to detect (case-insensitive).
const SQL_PATTERNS: &[&str] = &[
    "' or ",
    "' and ",
    "'; drop ",
    "'; delete ",
    "'; insert ",
    "'; update ",
    "'; select ",
    "union select",
    "union all select",
    "1=1",
    "1' or '1'='1",
    "' or 1=1",
    "'; exec ",
    "'; execute ",
    "--",
    "/*",
    "*/",
    "xp_",
    "sp_executesql",
    "char(",
    "nchar(",
    "varchar(",
    "alter table",
    "drop table",
    "exec(",
    "execute(",
    "information_schema",
    "sysobjects",
    "syscolumns",
    "load_file",
    "into outfile",
    "into dumpfile",
    "benchmark(",
    "sleep(",
    "waitfor delay",
];

/// Detect SQL injection patterns in input.
pub fn detect_sql_injection(input: &str) -> Option<String> {
    let lower = input.to_lowercase();
    for pattern in SQL_PATTERNS {
        if lower.contains(pattern) {
            return Some(pattern.to_string());
        }
    }
    None
}

/// Check for SQL injection, returning an error if detected.
pub fn reject_sql_injection(input: &str) -> Result<&str, SanitizeError> {
    if let Some(pattern) = detect_sql_injection(input) {
        Err(SanitizeError::SqlInjection(pattern))
    } else {
        Ok(input)
    }
}

/// Escape SQL special characters (for display, not for query building).
pub fn escape_sql_chars(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\'' => out.push_str("''"),
            '\\' => out.push_str("\\\\"),
            '\0' => {} // strip null
            _ => out.push(ch),
        }
    }
    out
}

// ── Path Traversal Prevention ──────────────────────────────────

/// Detect path traversal patterns.
pub fn detect_path_traversal(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    // Check for directory traversal.
    if normalized.contains("../") || normalized.contains("/..") {
        return true;
    }
    // Check for absolute paths when relative expected.
    if normalized.starts_with('/') {
        return true;
    }
    // Check for Windows drive letters.
    if path.len() >= 2 && path.as_bytes()[1] == b':' && path.as_bytes()[0].is_ascii_alphabetic() {
        return true;
    }
    // URL-encoded traversal.
    let decoded = percent_decode_simple(path);
    if decoded.contains("../") || decoded.contains("/..") {
        return true;
    }
    // Double-encoded.
    let double_decoded = percent_decode_simple(&decoded);
    if double_decoded.contains("../") || double_decoded.contains("/..") {
        return true;
    }
    false
}

/// Reject path traversal, returning error if detected.
pub fn reject_path_traversal(path: &str) -> Result<&str, SanitizeError> {
    if detect_path_traversal(path) {
        Err(SanitizeError::PathTraversal(path.to_string()))
    } else {
        Ok(path)
    }
}

/// Sanitize a file path by removing traversal components.
pub fn sanitize_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let parts: Vec<&str> = normalized.split('/').collect();
    let mut safe_parts = Vec::new();
    for part in parts {
        if part == ".." || part == "." || part.is_empty() {
            continue;
        }
        // Strip null bytes.
        let clean = part.replace('\0', "");
        if !clean.is_empty() {
            safe_parts.push(clean);
        }
    }
    safe_parts.join("/")
}

/// Simple percent-decode (handles %XX).
fn percent_decode_simple(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = &input[i + 1..i + 3];
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                result.push(byte as char);
                i += 3;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

// ── Unicode Security ───────────────────────────────────────────

/// Common homoglyph mappings (confusable characters).
/// Maps suspicious Unicode characters to their ASCII lookalikes.
const HOMOGLYPHS: &[(char, char)] = &[
    ('\u{0430}', 'a'), // Cyrillic а → Latin a
    ('\u{0435}', 'e'), // Cyrillic е → Latin e
    ('\u{043E}', 'o'), // Cyrillic о → Latin o
    ('\u{0440}', 'p'), // Cyrillic р → Latin p
    ('\u{0441}', 'c'), // Cyrillic с → Latin c
    ('\u{0443}', 'y'), // Cyrillic у → Latin y
    ('\u{0445}', 'x'), // Cyrillic х → Latin x
    ('\u{0455}', 's'), // Cyrillic ѕ → Latin s
    ('\u{0456}', 'i'), // Cyrillic і → Latin i
    ('\u{0458}', 'j'), // Cyrillic ј → Latin j
    ('\u{04BB}', 'h'), // Cyrillic һ → Latin h
    ('\u{0501}', 'd'), // Cyrillic ԁ → Latin d
    ('\u{FF41}', 'a'), // Fullwidth a → Latin a
    ('\u{FF45}', 'e'), // Fullwidth e → Latin e
    ('\u{FF4F}', 'o'), // Fullwidth o → Latin o
    ('\u{2010}', '-'), // Hyphen → ASCII hyphen
    ('\u{2011}', '-'), // Non-breaking hyphen
    ('\u{2012}', '-'), // Figure dash
    ('\u{2013}', '-'), // En dash
    ('\u{2014}', '-'), // Em dash
    ('\u{00A0}', ' '), // Non-breaking space
    ('\u{2000}', ' '), // En quad
    ('\u{2001}', ' '), // Em quad
    ('\u{2002}', ' '), // En space
    ('\u{2003}', ' '), // Em space
    ('\u{200B}', ' '), // Zero-width space (mapped to space for detection)
];

/// Detect homoglyphs in input, returning the suspicious characters found.
pub fn detect_homoglyphs(input: &str) -> Vec<(char, char)> {
    let mut found = Vec::new();
    for ch in input.chars() {
        for &(homoglyph, ascii) in HOMOGLYPHS {
            if ch == homoglyph {
                found.push((ch, ascii));
            }
        }
    }
    found
}

/// Check if input contains homoglyphs.
pub fn has_homoglyphs(input: &str) -> bool {
    input
        .chars()
        .any(|ch| HOMOGLYPHS.iter().any(|&(h, _)| ch == h))
}

/// Normalize homoglyphs to their ASCII equivalents.
pub fn normalize_homoglyphs(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    for ch in input.chars() {
        let mut replaced = false;
        for &(homoglyph, ascii) in HOMOGLYPHS {
            if ch == homoglyph {
                result.push(ascii);
                replaced = true;
                break;
            }
        }
        if !replaced {
            result.push(ch);
        }
    }
    result
}

/// Detect mixed scripts in a string (e.g., Latin + Cyrillic).
pub fn has_mixed_scripts(input: &str) -> bool {
    let mut has_latin = false;
    let mut has_cyrillic = false;
    let mut has_cjk = false;

    for ch in input.chars() {
        if ch.is_ascii_alphabetic() || ('\u{00C0}'..='\u{024F}').contains(&ch) {
            has_latin = true;
        } else if ('\u{0400}'..='\u{04FF}').contains(&ch) {
            has_cyrillic = true;
        } else if ('\u{4E00}'..='\u{9FFF}').contains(&ch) {
            has_cjk = true;
        }
    }

    let script_count =
        has_latin as u32 + has_cyrillic as u32 + has_cjk as u32;
    script_count > 1
}

/// Strip zero-width and invisible Unicode characters.
pub fn strip_invisible_chars(input: &str) -> String {
    input
        .chars()
        .filter(|ch| {
            !matches!(
                *ch,
                '\u{200B}' // Zero-width space
                | '\u{200C}' // Zero-width non-joiner
                | '\u{200D}' // Zero-width joiner
                | '\u{FEFF}' // BOM / zero-width no-break space
                | '\u{2060}' // Word joiner
                | '\u{00AD}' // Soft hyphen
                | '\u{200E}' // LTR mark
                | '\u{200F}' // RTL mark
                | '\u{202A}' // LTR embedding
                | '\u{202B}' // RTL embedding
                | '\u{202C}' // Pop directional formatting
                | '\u{202D}' // LTR override
                | '\u{202E}' // RTL override
            )
        })
        .collect()
}

// ── URL Sanitization ───────────────────────────────────────────

/// Dangerous URL schemes.
const DANGEROUS_SCHEMES: &[&str] = &[
    "javascript:",
    "data:",
    "vbscript:",
    "file:",
    "blob:",
];

/// Allowed URL schemes.
const SAFE_SCHEMES: &[&str] = &[
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "ftp://",
];

/// Sanitize a URL, rejecting dangerous schemes.
pub fn sanitize_url(url: &str) -> Result<String, SanitizeError> {
    let trimmed = url.trim();
    let lower = trimmed.to_lowercase();

    // Strip invisible chars.
    let clean = strip_invisible_chars(trimmed);
    let clean_lower = clean.to_lowercase();

    // Check for dangerous schemes.
    for scheme in DANGEROUS_SCHEMES {
        if clean_lower.starts_with(scheme) {
            return Err(SanitizeError::InvalidUrl(format!(
                "dangerous scheme: {scheme}"
            )));
        }
    }

    // Check for encoded dangerous schemes.
    let decoded = percent_decode_simple(&clean);
    let decoded_lower = decoded.to_lowercase();
    for scheme in DANGEROUS_SCHEMES {
        if decoded_lower.starts_with(scheme) {
            return Err(SanitizeError::InvalidUrl(format!(
                "encoded dangerous scheme: {scheme}"
            )));
        }
    }

    // Allow relative URLs.
    if clean.starts_with('/') || clean.starts_with('#') || clean.starts_with('?') {
        return Ok(clean);
    }

    // Check for safe schemes.
    let is_safe = SAFE_SCHEMES.iter().any(|s| clean_lower.starts_with(s));
    if !is_safe && clean.contains(':') {
        return Err(SanitizeError::InvalidUrl(format!(
            "unknown scheme in URL: {}",
            &clean[..clean.find(':').unwrap_or(0)]
        )));
    }

    Ok(clean)
}

/// Check if a URL uses a safe scheme.
pub fn is_safe_url(url: &str) -> bool {
    sanitize_url(url).is_ok()
}

// ── Input Validator ────────────────────────────────────────────

/// Comprehensive input sanitizer combining multiple checks.
pub struct InputSanitizer {
    /// Maximum input length.
    pub max_length: usize,
    /// Whether to check for SQL injection.
    pub check_sql: bool,
    /// Whether to check for path traversal.
    pub check_path_traversal: bool,
    /// Whether to check for homoglyphs.
    pub check_homoglyphs: bool,
    /// Whether to strip null bytes.
    pub strip_nulls: bool,
    /// Whether to strip invisible characters.
    pub strip_invisible: bool,
    /// Custom blocked patterns.
    pub blocked_patterns: Vec<String>,
}

impl Default for InputSanitizer {
    fn default() -> Self {
        Self {
            max_length: 10_000,
            check_sql: true,
            check_path_traversal: true,
            check_homoglyphs: false,
            strip_nulls: true,
            strip_invisible: true,
            blocked_patterns: Vec::new(),
        }
    }
}

impl InputSanitizer {
    /// Create a new sanitizer with all checks enabled.
    pub fn strict() -> Self {
        Self {
            check_homoglyphs: true,
            ..Default::default()
        }
    }

    /// Add a custom blocked pattern.
    pub fn block_pattern(mut self, pattern: &str) -> Self {
        self.blocked_patterns.push(pattern.to_lowercase());
        self
    }

    /// Sanitize input, returning the cleaned string or an error.
    pub fn sanitize(&self, input: &str) -> Result<String, SanitizeError> {
        // Length check.
        if input.len() > self.max_length {
            return Err(SanitizeError::InputTooLong {
                max: self.max_length,
                got: input.len(),
            });
        }

        let mut clean = input.to_string();

        // Null byte handling.
        if self.strip_nulls {
            clean = strip_null_bytes(&clean);
        } else {
            reject_null_bytes(&clean)?;
        }

        // Invisible character stripping.
        if self.strip_invisible {
            clean = strip_invisible_chars(&clean);
        }

        // SQL injection check.
        if self.check_sql {
            reject_sql_injection(&clean)?;
        }

        // Path traversal check.
        if self.check_path_traversal && detect_path_traversal(&clean) {
            return Err(SanitizeError::PathTraversal(clean));
        }

        // Homoglyph check.
        if self.check_homoglyphs && has_homoglyphs(&clean) {
            let found = detect_homoglyphs(&clean);
            if let Some((homo, _)) = found.first() {
                return Err(SanitizeError::HomoglyphDetected {
                    original: clean,
                    suspicious: homo.to_string(),
                });
            }
        }

        // Custom blocked patterns.
        let lower = clean.to_lowercase();
        for pattern in &self.blocked_patterns {
            if lower.contains(pattern) {
                return Err(SanitizeError::SqlInjection(pattern.clone()));
            }
        }

        Ok(clean)
    }
}

// ── Utility: Normalize whitespace ──────────────────────────────

/// Normalize whitespace: collapse runs, trim, replace special whitespace.
pub fn normalize_whitespace(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut last_was_space = true; // trim leading
    for ch in input.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(ch);
            last_was_space = false;
        }
    }
    // Trim trailing space.
    if result.ends_with(' ') {
        result.pop();
    }
    result
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Null Bytes ──

    #[test]
    fn test_strip_null_bytes() {
        assert_eq!(strip_null_bytes("hel\0lo"), "hello");
        assert_eq!(strip_null_bytes("no nulls"), "no nulls");
    }

    #[test]
    fn test_has_null_bytes() {
        assert!(has_null_bytes("abc\0def"));
        assert!(!has_null_bytes("abcdef"));
    }

    #[test]
    fn test_reject_null_bytes() {
        assert!(reject_null_bytes("clean").is_ok());
        assert_eq!(reject_null_bytes("nul\0l").unwrap_err(), SanitizeError::NullByte);
    }

    // ── SQL Injection ──

    #[test]
    fn test_detect_sql_injection_basic() {
        assert!(detect_sql_injection("admin' or 1=1--").is_some());
        assert!(detect_sql_injection("'; DROP TABLE users;").is_some());
        assert!(detect_sql_injection("hello world").is_none());
    }

    #[test]
    fn test_detect_sql_injection_union() {
        assert!(detect_sql_injection("1 UNION SELECT * FROM users").is_some());
    }

    #[test]
    fn test_reject_sql_injection() {
        assert!(reject_sql_injection("normal input").is_ok());
        assert!(reject_sql_injection("'; delete from x").is_err());
    }

    #[test]
    fn test_escape_sql_chars() {
        assert_eq!(escape_sql_chars("it's"), "it''s");
        assert_eq!(escape_sql_chars("a\\b"), "a\\\\b");
        assert_eq!(escape_sql_chars("nul\0l"), "null");
    }

    // ── Path Traversal ──

    #[test]
    fn test_detect_path_traversal() {
        assert!(detect_path_traversal("../etc/passwd"));
        assert!(detect_path_traversal("..\\..\\windows\\system32"));
        assert!(detect_path_traversal("/etc/passwd"));
        assert!(detect_path_traversal("C:\\windows"));
        assert!(!detect_path_traversal("safe/path/file.txt"));
    }

    #[test]
    fn test_encoded_path_traversal() {
        assert!(detect_path_traversal("%2e%2e/etc/passwd"));
        assert!(detect_path_traversal("..%2Fetc%2Fpasswd"));
    }

    #[test]
    fn test_sanitize_path() {
        assert_eq!(sanitize_path("../etc/passwd"), "etc/passwd");
        assert_eq!(sanitize_path("./file.txt"), "file.txt");
        assert_eq!(sanitize_path("a/b/../c"), "a/b/c");
        assert_eq!(sanitize_path("safe/path"), "safe/path");
    }

    // ── Homoglyphs ──

    #[test]
    fn test_detect_homoglyphs() {
        // Cyrillic 'а' (U+0430) looks like Latin 'a'.
        let input = "p\u{0430}ypal.com";
        let found = detect_homoglyphs(input);
        assert!(!found.is_empty());
        assert_eq!(found[0].1, 'a');
    }

    #[test]
    fn test_normalize_homoglyphs() {
        let input = "p\u{0430}ypal";
        let normalized = normalize_homoglyphs(input);
        assert_eq!(normalized, "paypal");
    }

    #[test]
    fn test_no_homoglyphs_in_ascii() {
        assert!(!has_homoglyphs("hello world"));
        assert!(detect_homoglyphs("hello world").is_empty());
    }

    // ── Mixed Scripts ──

    #[test]
    fn test_mixed_scripts() {
        // Latin + Cyrillic.
        assert!(has_mixed_scripts("hello\u{0410}"));
        // Pure ASCII.
        assert!(!has_mixed_scripts("hello world"));
    }

    // ── Invisible Characters ──

    #[test]
    fn test_strip_invisible() {
        let input = "he\u{200B}ll\u{FEFF}o";
        assert_eq!(strip_invisible_chars(input), "hello");
    }

    #[test]
    fn test_strip_bidi_override() {
        let input = "normal\u{202E}esrever";
        let clean = strip_invisible_chars(input);
        assert!(!clean.contains('\u{202E}'));
    }

    // ── URL Sanitization ──

    #[test]
    fn test_safe_url() {
        assert!(sanitize_url("https://example.com").is_ok());
        assert!(sanitize_url("http://example.com/path").is_ok());
        assert!(sanitize_url("/relative/path").is_ok());
        assert!(sanitize_url("mailto:user@example.com").is_ok());
    }

    #[test]
    fn test_dangerous_url() {
        assert!(sanitize_url("javascript:alert(1)").is_err());
        assert!(sanitize_url("data:text/html,<script>").is_err());
        assert!(sanitize_url("vbscript:msgbox").is_err());
    }

    #[test]
    fn test_encoded_dangerous_url() {
        assert!(sanitize_url("java%73cript:alert(1)").is_err());
    }

    #[test]
    fn test_is_safe_url() {
        assert!(is_safe_url("https://example.com"));
        assert!(!is_safe_url("javascript:void(0)"));
    }

    // ── Whitespace ──

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("a\t\nb"), "a b");
    }

    // ── InputSanitizer ──

    #[test]
    fn test_sanitizer_default_pass() {
        let s = InputSanitizer::default();
        assert!(s.sanitize("normal input").is_ok());
    }

    #[test]
    fn test_sanitizer_rejects_sql() {
        let s = InputSanitizer::default();
        assert!(s.sanitize("'; drop table users").is_err());
    }

    #[test]
    fn test_sanitizer_rejects_too_long() {
        let s = InputSanitizer {
            max_length: 10,
            ..Default::default()
        };
        assert!(s.sanitize("this is way too long").is_err());
    }

    #[test]
    fn test_sanitizer_strips_nulls() {
        let s = InputSanitizer::default();
        let result = s.sanitize("hel\0lo").unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_sanitizer_custom_pattern() {
        let s = InputSanitizer::default().block_pattern("badword");
        assert!(s.sanitize("this has a BADWORD in it").is_err());
    }

    #[test]
    fn test_sanitizer_strict_homoglyphs() {
        let s = InputSanitizer::strict();
        let input = "p\u{0430}ypal";
        assert!(s.sanitize(input).is_err());
    }

    #[test]
    fn test_error_display() {
        let e = SanitizeError::SqlInjection("1=1".to_string());
        assert!(e.to_string().contains("SQL injection"));
        let e2 = SanitizeError::NullByte;
        assert!(e2.to_string().contains("null byte"));
    }
}
