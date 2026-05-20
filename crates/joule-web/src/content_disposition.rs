//! Content-Disposition header parsing and generation.
//!
//! Replaces content-disposition npm package with a pure-Rust implementation
//! supporting inline/attachment, filename/filename* (RFC 5987), percent-encoding
//! for UTF-8 filenames, and browser compatibility helpers.

use std::fmt;

// ── DispositionType ─────────────────────────────────────────────

/// The disposition type: inline or attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispositionType {
    Inline,
    Attachment,
    FormData,
}

impl fmt::Display for DispositionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline => write!(f, "inline"),
            Self::Attachment => write!(f, "attachment"),
            Self::FormData => write!(f, "form-data"),
        }
    }
}

// ── ContentDisposition ──────────────────────────────────────────

/// A parsed Content-Disposition header value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentDisposition {
    pub disposition: DispositionType,
    pub filename: Option<String>,
    pub filename_star: Option<String>,
    pub name: Option<String>,
    pub params: Vec<(String, String)>,
}

impl ContentDisposition {
    /// Create a simple inline disposition.
    pub fn inline() -> Self {
        Self {
            disposition: DispositionType::Inline,
            filename: None,
            filename_star: None,
            name: None,
            params: Vec::new(),
        }
    }

    /// Create an attachment disposition with optional filename.
    pub fn attachment(filename: Option<&str>) -> Self {
        let mut cd = Self {
            disposition: DispositionType::Attachment,
            filename: None,
            filename_star: None,
            name: None,
            params: Vec::new(),
        };
        if let Some(f) = filename {
            cd.set_filename(f);
        }
        cd
    }

    /// Create a form-data disposition with a field name.
    pub fn form_data(name: &str, filename: Option<&str>) -> Self {
        let mut cd = Self {
            disposition: DispositionType::FormData,
            filename: None,
            filename_star: None,
            name: Some(name.to_string()),
            params: Vec::new(),
        };
        if let Some(f) = filename {
            cd.set_filename(f);
        }
        cd
    }

    /// Set the filename, automatically choosing RFC 5987 encoding if non-ASCII.
    pub fn set_filename(&mut self, filename: &str) {
        if is_ascii_safe(filename) {
            self.filename = Some(filename.to_string());
            self.filename_star = None;
        } else {
            // ASCII fallback: strip non-ASCII chars
            let ascii_fallback: String = filename
                .chars()
                .filter(|c| c.is_ascii() && *c != '"' && *c != '\\')
                .collect();
            if ascii_fallback.is_empty() {
                self.filename = Some("download".to_string());
            } else {
                self.filename = Some(ascii_fallback);
            }
            self.filename_star = Some(filename.to_string());
        }
    }

    /// Get the best filename: prefer filename* (RFC 5987) over filename.
    pub fn best_filename(&self) -> Option<&str> {
        self.filename_star
            .as_deref()
            .or(self.filename.as_deref())
    }

    /// Parse a Content-Disposition header value.
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut parts = trimmed.splitn(2, ';');
        let disposition_str = parts.next()?.trim().to_ascii_lowercase();

        let disposition = match disposition_str.as_str() {
            "inline" => DispositionType::Inline,
            "attachment" => DispositionType::Attachment,
            "form-data" => DispositionType::FormData,
            _ => return None,
        };

        let mut cd = Self {
            disposition,
            filename: None,
            filename_star: None,
            name: None,
            params: Vec::new(),
        };

        if let Some(param_str) = parts.next() {
            for param in param_str.split(';') {
                let param = param.trim();
                if param.is_empty() {
                    continue;
                }
                if let Some((key, value)) = param.split_once('=') {
                    let key = key.trim().to_ascii_lowercase();
                    let value = value.trim();
                    let unquoted = unquote(value);

                    match key.as_str() {
                        "filename" => {
                            cd.filename = Some(unquoted);
                        }
                        "filename*" => {
                            cd.filename_star = Some(decode_rfc5987(&unquoted));
                        }
                        "name" => {
                            cd.name = Some(unquoted);
                        }
                        _ => {
                            cd.params.push((key, unquoted));
                        }
                    }
                }
            }
        }

        Some(cd)
    }

    /// Set a custom parameter.
    pub fn set_param(&mut self, key: &str, value: &str) {
        let key_lower = key.to_ascii_lowercase();
        if let Some(existing) = self.params.iter_mut().find(|(k, _)| *k == key_lower) {
            existing.1 = value.to_string();
        } else {
            self.params.push((key_lower, value.to_string()));
        }
    }

    /// Get a custom parameter.
    pub fn param(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_ascii_lowercase();
        self.params
            .iter()
            .find(|(k, _)| *k == key_lower)
            .map(|(_, v)| v.as_str())
    }

    /// Check whether the filename has a specific extension (case-insensitive).
    pub fn has_extension(&self, ext: &str) -> bool {
        if let Some(f) = self.best_filename() {
            f.to_ascii_lowercase()
                .ends_with(&format!(".{}", ext.to_ascii_lowercase()))
        } else {
            false
        }
    }

    /// Compute the file extension from the filename, if any.
    pub fn extension(&self) -> Option<String> {
        let f = self.best_filename()?;
        let dot_pos = f.rfind('.')?;
        if dot_pos + 1 < f.len() {
            Some(f[dot_pos + 1..].to_ascii_lowercase())
        } else {
            None
        }
    }
}

// ── Display ─────────────────────────────────────────────────────

impl fmt::Display for ContentDisposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disposition)?;

        if let Some(name) = &self.name {
            write!(f, "; name=\"{}\"", escape_quotes(name))?;
        }

        if let Some(filename) = &self.filename {
            write!(f, "; filename=\"{}\"", escape_quotes(filename))?;
        }

        if let Some(filename_star) = &self.filename_star {
            write!(f, "; filename*=UTF-8''{}", percent_encode(filename_star))?;
        }

        for (k, v) in &self.params {
            write!(f, "; {}=\"{}\"", k, escape_quotes(v))?;
        }

        Ok(())
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Check if a filename contains only ASCII-safe characters.
fn is_ascii_safe(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii() && c != '\0')
}

/// Remove surrounding quotes from a value.
fn unquote(s: &str) -> String {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].replace("\\\"", "\"").replace("\\\\", "\\")
    } else {
        s.to_string()
    }
}

/// Escape quotes in a string for use in header values.
fn escape_quotes(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Percent-encode a UTF-8 string for RFC 5987 filename*.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b == b'-' || b == b'.' || b == b'_' || b == b'~' {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_digit(b >> 4));
            out.push(hex_digit(b & 0x0F));
        }
    }
    out
}

fn hex_digit(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'A' + n - 10) as char,
        _ => '0',
    }
}

/// Decode an RFC 5987 encoded value like `UTF-8''hello%20world`.
fn decode_rfc5987(s: &str) -> String {
    // Format: charset'language'value
    if let Some(value_start) = s.find("''") {
        let encoded = &s[value_start + 2..];
        percent_decode(encoded)
    } else {
        // No encoding prefix, just try to percent-decode
        percent_decode(s)
    }
}

/// Decode percent-encoded bytes into a UTF-8 string.
fn percent_decode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let h1 = chars.next().unwrap_or('0');
            let h2 = chars.next().unwrap_or('0');
            let hex_str: String = [h1, h2].iter().collect();
            if let Ok(b) = u8::from_str_radix(&hex_str, 16) {
                bytes.push(b);
            }
        } else {
            bytes.push(c as u8);
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|_| s.to_string())
}

/// Sanitize a filename for safe filesystem use.
pub fn sanitize_filename(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for c in name.chars() {
        match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '\0' => {
                result.push('_');
            }
            c if c.is_control() => {
                result.push('_');
            }
            _ => {
                result.push(c);
            }
        }
    }

    // Trim leading/trailing dots and spaces (Windows compatibility)
    let trimmed = result.trim_matches(|c: char| c == '.' || c == ' ');
    if trimmed.is_empty() {
        "download".to_string()
    } else {
        trimmed.to_string()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_inline() {
        let cd = ContentDisposition::parse("inline").unwrap();
        assert_eq!(cd.disposition, DispositionType::Inline);
        assert!(cd.filename.is_none());
    }

    #[test]
    fn parse_attachment_no_filename() {
        let cd = ContentDisposition::parse("attachment").unwrap();
        assert_eq!(cd.disposition, DispositionType::Attachment);
        assert!(cd.filename.is_none());
    }

    #[test]
    fn parse_attachment_with_filename() {
        let cd = ContentDisposition::parse("attachment; filename=\"report.pdf\"").unwrap();
        assert_eq!(cd.disposition, DispositionType::Attachment);
        assert_eq!(cd.filename.as_deref(), Some("report.pdf"));
    }

    #[test]
    fn parse_attachment_unquoted_filename() {
        let cd = ContentDisposition::parse("attachment; filename=report.pdf").unwrap();
        assert_eq!(cd.filename.as_deref(), Some("report.pdf"));
    }

    #[test]
    fn parse_form_data() {
        let cd = ContentDisposition::parse("form-data; name=\"file\"; filename=\"photo.jpg\"").unwrap();
        assert_eq!(cd.disposition, DispositionType::FormData);
        assert_eq!(cd.name.as_deref(), Some("file"));
        assert_eq!(cd.filename.as_deref(), Some("photo.jpg"));
    }

    #[test]
    fn parse_filename_star() {
        let cd = ContentDisposition::parse(
            "attachment; filename*=UTF-8''%E4%B8%AD%E6%96%87.pdf"
        ).unwrap();
        assert_eq!(cd.filename_star.as_deref(), Some("\u{4e2d}\u{6587}.pdf"));
    }

    #[test]
    fn parse_both_filenames() {
        let cd = ContentDisposition::parse(
            "attachment; filename=\"fallback.pdf\"; filename*=UTF-8''%C3%BCber.pdf"
        ).unwrap();
        assert_eq!(cd.filename.as_deref(), Some("fallback.pdf"));
        assert_eq!(cd.filename_star.as_deref(), Some("\u{00fc}ber.pdf"));
        assert_eq!(cd.best_filename(), Some("\u{00fc}ber.pdf"));
    }

    #[test]
    fn parse_invalid() {
        assert!(ContentDisposition::parse("").is_none());
        assert!(ContentDisposition::parse("unknown").is_none());
    }

    #[test]
    fn parse_case_insensitive() {
        let cd = ContentDisposition::parse("ATTACHMENT; FILENAME=\"test.txt\"").unwrap();
        assert_eq!(cd.disposition, DispositionType::Attachment);
        assert_eq!(cd.filename.as_deref(), Some("test.txt"));
    }

    #[test]
    fn display_inline() {
        let cd = ContentDisposition::inline();
        assert_eq!(cd.to_string(), "inline");
    }

    #[test]
    fn display_attachment_with_filename() {
        let cd = ContentDisposition::attachment(Some("report.pdf"));
        assert_eq!(cd.to_string(), "attachment; filename=\"report.pdf\"");
    }

    #[test]
    fn display_attachment_utf8() {
        let cd = ContentDisposition::attachment(Some("\u{00fc}ber.pdf"));
        let s = cd.to_string();
        assert!(s.contains("filename=\"ber.pdf\""));
        assert!(s.contains("filename*=UTF-8''%C3%BCber.pdf"));
    }

    #[test]
    fn display_form_data() {
        let cd = ContentDisposition::form_data("field", Some("data.csv"));
        let s = cd.to_string();
        assert!(s.starts_with("form-data"));
        assert!(s.contains("name=\"field\""));
        assert!(s.contains("filename=\"data.csv\""));
    }

    #[test]
    fn best_filename_prefers_star() {
        let mut cd = ContentDisposition::attachment(Some("test.pdf"));
        cd.filename_star = Some("special.pdf".to_string());
        assert_eq!(cd.best_filename(), Some("special.pdf"));
    }

    #[test]
    fn best_filename_fallback() {
        let cd = ContentDisposition::attachment(Some("test.pdf"));
        assert_eq!(cd.best_filename(), Some("test.pdf"));
    }

    #[test]
    fn has_extension_check() {
        let cd = ContentDisposition::attachment(Some("report.PDF"));
        assert!(cd.has_extension("pdf"));
        assert!(cd.has_extension("PDF"));
        assert!(!cd.has_extension("doc"));
    }

    #[test]
    fn extension_extraction() {
        let cd = ContentDisposition::attachment(Some("archive.tar.gz"));
        assert_eq!(cd.extension(), Some("gz".to_string()));
    }

    #[test]
    fn extension_none() {
        let cd = ContentDisposition::attachment(Some("README"));
        assert!(cd.extension().is_none());
    }

    #[test]
    fn custom_params() {
        let mut cd = ContentDisposition::inline();
        cd.set_param("creation-date", "2024-01-15");
        assert_eq!(cd.param("creation-date"), Some("2024-01-15"));
    }

    #[test]
    fn sanitize_basic() {
        assert_eq!(sanitize_filename("hello.txt"), "hello.txt");
    }

    #[test]
    fn sanitize_dangerous_chars() {
        assert_eq!(sanitize_filename("file/name.txt"), "file_name.txt");
        assert_eq!(sanitize_filename("file\\name.txt"), "file_name.txt");
        assert_eq!(sanitize_filename("file:name.txt"), "file_name.txt");
        assert_eq!(sanitize_filename("file*name.txt"), "file_name.txt");
        assert_eq!(sanitize_filename("file?name.txt"), "file_name.txt");
    }

    #[test]
    fn sanitize_empty_result() {
        assert_eq!(sanitize_filename("..."), "download");
        assert_eq!(sanitize_filename("   "), "download");
    }

    #[test]
    fn sanitize_control_chars() {
        assert_eq!(sanitize_filename("file\x01name.txt"), "file_name.txt");
    }

    #[test]
    fn percent_encode_roundtrip() {
        let original = "hello world\u{00e9}.pdf";
        let encoded = percent_encode(original);
        let decoded = percent_decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn percent_encode_ascii_safe() {
        let encoded = percent_encode("hello.pdf");
        assert_eq!(encoded, "hello.pdf");
    }

    #[test]
    fn rfc5987_decode() {
        let decoded = decode_rfc5987("UTF-8''hello%20world.pdf");
        assert_eq!(decoded, "hello world.pdf");
    }

    #[test]
    fn display_roundtrip() {
        let cd = ContentDisposition::attachment(Some("test file.pdf"));
        let s = cd.to_string();
        let parsed = ContentDisposition::parse(&s).unwrap();
        assert_eq!(parsed.disposition, DispositionType::Attachment);
        // The filename should be recoverable
        assert!(parsed.best_filename().is_some());
    }

    #[test]
    fn set_filename_ascii() {
        let mut cd = ContentDisposition::attachment(None);
        cd.set_filename("simple.txt");
        assert_eq!(cd.filename.as_deref(), Some("simple.txt"));
        assert!(cd.filename_star.is_none());
    }

    #[test]
    fn set_filename_unicode() {
        let mut cd = ContentDisposition::attachment(None);
        cd.set_filename("\u{6587}\u{4ef6}.pdf");
        assert!(cd.filename.is_some());
        assert!(cd.filename_star.is_some());
        assert_eq!(cd.filename_star.as_deref(), Some("\u{6587}\u{4ef6}.pdf"));
    }

    #[test]
    fn escaped_quotes_in_filename() {
        let cd = ContentDisposition::attachment(Some("file\"name.txt"));
        let s = cd.to_string();
        assert!(s.contains("\\\""));
    }
}
