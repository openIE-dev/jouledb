//! MIME / media type parsing and matching.
//!
//! Replaces mime, accept-header, and content-type npm packages with a pure-Rust
//! implementation supporting type/subtype, parameters, quality values, media
//! range matching, charset, boundary, and content-type building.

use std::fmt;

// ── MediaType ───────────────────────────────────────────────────

/// A parsed MIME media type (e.g. `text/html; charset=utf-8`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaType {
    /// Top-level type (e.g. `text`, `application`, `image`).
    pub main_type: String,
    /// Sub-type (e.g. `html`, `json`, `png`).
    pub sub_type: String,
    /// Optional parameters (e.g. `charset=utf-8`, `boundary=---`).
    pub params: Vec<(String, String)>,
}

/// A media range with an optional quality value for content negotiation.
#[derive(Debug, Clone)]
pub struct MediaRange {
    pub main_type: String,
    pub sub_type: String,
    pub quality: f64,
    pub params: Vec<(String, String)>,
}

impl PartialEq for MediaRange {
    fn eq(&self, other: &Self) -> bool {
        self.main_type == other.main_type
            && self.sub_type == other.sub_type
            && (self.quality - other.quality).abs() < 1e-6
            && self.params == other.params
    }
}

/// Common media types.
impl MediaType {
    pub fn text_plain() -> Self {
        Self::new("text", "plain")
    }

    pub fn text_html() -> Self {
        Self::new("text", "html")
    }

    pub fn text_css() -> Self {
        Self::new("text", "css")
    }

    pub fn application_json() -> Self {
        Self::new("application", "json")
    }

    pub fn application_octet_stream() -> Self {
        Self::new("application", "octet-stream")
    }

    pub fn application_xml() -> Self {
        Self::new("application", "xml")
    }

    pub fn image_png() -> Self {
        Self::new("image", "png")
    }

    pub fn image_jpeg() -> Self {
        Self::new("image", "jpeg")
    }

    pub fn image_svg() -> Self {
        Self::new("image", "svg+xml")
    }

    pub fn multipart_form_data() -> Self {
        Self::new("multipart", "form-data")
    }

    pub fn application_form_urlencoded() -> Self {
        Self::new("application", "x-www-form-urlencoded")
    }
}

// ── Constructors & Parsing ──────────────────────────────────────

impl MediaType {
    /// Create a media type without parameters.
    pub fn new(main_type: &str, sub_type: &str) -> Self {
        Self {
            main_type: main_type.to_ascii_lowercase(),
            sub_type: sub_type.to_ascii_lowercase(),
            params: Vec::new(),
        }
    }

    /// Parse a media type string like `text/html; charset=utf-8`.
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }

        let mut parts = trimmed.splitn(2, ';');
        let type_part = parts.next()?.trim();

        let (main, sub) = type_part.split_once('/')?;
        let main = main.trim();
        let sub = sub.trim();

        if main.is_empty() || sub.is_empty() {
            return None;
        }

        let mut mt = Self::new(main, sub);

        if let Some(param_str) = parts.next() {
            for param in param_str.split(';') {
                let param = param.trim();
                if param.is_empty() {
                    continue;
                }
                if let Some((k, v)) = param.split_once('=') {
                    let key = k.trim().to_ascii_lowercase();
                    let val = v.trim().trim_matches('"').to_string();
                    mt.params.push((key, val));
                }
            }
        }

        Some(mt)
    }

    /// Get the `charset` parameter if present.
    pub fn charset(&self) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == "charset")
            .map(|(_, v)| v.as_str())
    }

    /// Get the `boundary` parameter if present (for multipart types).
    pub fn boundary(&self) -> Option<&str> {
        self.params
            .iter()
            .find(|(k, _)| k == "boundary")
            .map(|(_, v)| v.as_str())
    }

    /// Get the full type string without parameters (e.g. `text/html`).
    pub fn essence(&self) -> String {
        format!("{}/{}", self.main_type, self.sub_type)
    }

    /// Check if this media type is text-based.
    pub fn is_text(&self) -> bool {
        self.main_type == "text"
            || (self.main_type == "application"
                && (self.sub_type == "json"
                    || self.sub_type == "xml"
                    || self.sub_type == "javascript"
                    || self.sub_type.ends_with("+json")
                    || self.sub_type.ends_with("+xml")))
    }

    /// Check whether this type matches a wildcard-capable range.
    pub fn matches_range(&self, range: &MediaRange) -> bool {
        if range.main_type == "*" && range.sub_type == "*" {
            return true;
        }
        if range.main_type == self.main_type && range.sub_type == "*" {
            return true;
        }
        range.main_type == self.main_type && range.sub_type == self.sub_type
    }

    /// Set or replace a parameter.
    pub fn set_param(&mut self, key: &str, value: &str) {
        let key_lower = key.to_ascii_lowercase();
        if let Some(existing) = self.params.iter_mut().find(|(k, _)| *k == key_lower) {
            existing.1 = value.to_string();
        } else {
            self.params.push((key_lower, value.to_string()));
        }
    }

    /// Remove a parameter by key.
    pub fn remove_param(&mut self, key: &str) {
        let key_lower = key.to_ascii_lowercase();
        self.params.retain(|(k, _)| *k != key_lower);
    }

    /// Get a parameter by key.
    pub fn param(&self, key: &str) -> Option<&str> {
        let key_lower = key.to_ascii_lowercase();
        self.params
            .iter()
            .find(|(k, _)| *k == key_lower)
            .map(|(_, v)| v.as_str())
    }
}

// ── MediaRange ──────────────────────────────────────────────────

impl MediaRange {
    /// Parse a single media range like `text/*;q=0.8`.
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        let mut parts = trimmed.splitn(2, ';');
        let type_part = parts.next()?.trim();

        let (main, sub) = type_part.split_once('/')?;
        let main = main.trim();
        let sub = sub.trim();

        if main.is_empty() || sub.is_empty() {
            return None;
        }

        let mut quality = 1.0_f64;
        let mut params = Vec::new();

        if let Some(param_str) = parts.next() {
            for param in param_str.split(';') {
                let param = param.trim();
                if param.is_empty() {
                    continue;
                }
                if let Some((k, v)) = param.split_once('=') {
                    let key = k.trim().to_ascii_lowercase();
                    let val = v.trim().trim_matches('"').to_string();
                    if key == "q" {
                        quality = val.parse::<f64>().unwrap_or(1.0).clamp(0.0, 1.0);
                    } else {
                        params.push((key, val));
                    }
                }
            }
        }

        Some(Self {
            main_type: main.to_ascii_lowercase(),
            sub_type: sub.to_ascii_lowercase(),
            quality,
            params,
        })
    }

    /// Parse a full Accept header into sorted media ranges.
    pub fn parse_accept(header: &str) -> Vec<Self> {
        let mut ranges: Vec<Self> = header
            .split(',')
            .filter_map(|part| Self::parse(part))
            .collect();

        // Sort by quality descending, then by specificity descending
        ranges.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| specificity(b).cmp(&specificity(a)))
        });

        ranges
    }

    /// Compute the specificity: 0 = */*, 1 = type/*, 2 = type/subtype, +1 per param.
    pub fn specificity(&self) -> usize {
        specificity(self)
    }
}

fn specificity(range: &MediaRange) -> usize {
    let base = if range.main_type == "*" {
        0
    } else if range.sub_type == "*" {
        1
    } else {
        2
    };
    base + range.params.len()
}

// ── ContentTypeBuilder ──────────────────────────────────────────

/// Builder for Content-Type header values.
pub struct ContentTypeBuilder {
    media_type: MediaType,
}

impl ContentTypeBuilder {
    pub fn new(main_type: &str, sub_type: &str) -> Self {
        Self {
            media_type: MediaType::new(main_type, sub_type),
        }
    }

    pub fn from_media_type(mt: MediaType) -> Self {
        Self { media_type: mt }
    }

    pub fn charset(mut self, charset: &str) -> Self {
        self.media_type.set_param("charset", charset);
        self
    }

    pub fn boundary(mut self, boundary: &str) -> Self {
        self.media_type.set_param("boundary", boundary);
        self
    }

    pub fn param(mut self, key: &str, value: &str) -> Self {
        self.media_type.set_param(key, value);
        self
    }

    pub fn build(self) -> MediaType {
        self.media_type
    }

    pub fn build_string(self) -> String {
        self.media_type.to_string()
    }
}

// ── Content negotiation ─────────────────────────────────────────

/// Select the best media type from `available` that matches the Accept header.
pub fn negotiate(accept_header: &str, available: &[MediaType]) -> Option<MediaType> {
    let ranges = MediaRange::parse_accept(accept_header);

    for range in &ranges {
        if range.quality <= 0.0 {
            continue;
        }
        for mt in available {
            if mt.matches_range(range) {
                return Some(mt.clone());
            }
        }
    }

    None
}

/// Guess a media type from a file extension.
pub fn from_extension(ext: &str) -> MediaType {
    match ext.to_ascii_lowercase().as_str() {
        "html" | "htm" => MediaType::text_html(),
        "css" => MediaType::text_css(),
        "js" | "mjs" => MediaType::new("application", "javascript"),
        "json" => MediaType::application_json(),
        "xml" => MediaType::application_xml(),
        "txt" => MediaType::text_plain(),
        "png" => MediaType::image_png(),
        "jpg" | "jpeg" => MediaType::image_jpeg(),
        "gif" => MediaType::new("image", "gif"),
        "svg" => MediaType::image_svg(),
        "webp" => MediaType::new("image", "webp"),
        "ico" => MediaType::new("image", "x-icon"),
        "pdf" => MediaType::new("application", "pdf"),
        "wasm" => MediaType::new("application", "wasm"),
        "mp3" => MediaType::new("audio", "mpeg"),
        "mp4" => MediaType::new("video", "mp4"),
        "webm" => MediaType::new("video", "webm"),
        "woff" => MediaType::new("font", "woff"),
        "woff2" => MediaType::new("font", "woff2"),
        "ttf" => MediaType::new("font", "ttf"),
        "otf" => MediaType::new("font", "otf"),
        "csv" => MediaType::new("text", "csv"),
        "zip" => MediaType::new("application", "zip"),
        "gz" | "gzip" => MediaType::new("application", "gzip"),
        "tar" => MediaType::new("application", "x-tar"),
        _ => MediaType::application_octet_stream(),
    }
}

// ── Display ─────────────────────────────────────────────────────

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.main_type, self.sub_type)?;
        for (k, v) in &self.params {
            if v.contains(' ') || v.contains(';') {
                write!(f, "; {}=\"{}\"", k, v)?;
            } else {
                write!(f, "; {}={}", k, v)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for MediaRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.main_type, self.sub_type)?;
        for (k, v) in &self.params {
            write!(f, "; {}={}", k, v)?;
        }
        if (self.quality - 1.0).abs() > 1e-6 {
            write!(f, "; q={:.1}", self.quality)?;
        }
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple() {
        let mt = MediaType::parse("text/html").unwrap();
        assert_eq!(mt.main_type, "text");
        assert_eq!(mt.sub_type, "html");
        assert!(mt.params.is_empty());
    }

    #[test]
    fn parse_with_charset() {
        let mt = MediaType::parse("text/html; charset=utf-8").unwrap();
        assert_eq!(mt.charset(), Some("utf-8"));
    }

    #[test]
    fn parse_with_boundary() {
        let mt = MediaType::parse("multipart/form-data; boundary=----WebKitFormBoundary").unwrap();
        assert_eq!(mt.boundary(), Some("----WebKitFormBoundary"));
    }

    #[test]
    fn parse_multiple_params() {
        let mt = MediaType::parse("text/plain; charset=utf-8; format=flowed").unwrap();
        assert_eq!(mt.charset(), Some("utf-8"));
        assert_eq!(mt.param("format"), Some("flowed"));
    }

    #[test]
    fn parse_case_insensitive() {
        let mt = MediaType::parse("Application/JSON").unwrap();
        assert_eq!(mt.main_type, "application");
        assert_eq!(mt.sub_type, "json");
    }

    #[test]
    fn parse_quoted_value() {
        let mt = MediaType::parse(r#"text/plain; charset="utf-8""#).unwrap();
        assert_eq!(mt.charset(), Some("utf-8"));
    }

    #[test]
    fn parse_invalid_returns_none() {
        assert!(MediaType::parse("").is_none());
        assert!(MediaType::parse("text").is_none());
        assert!(MediaType::parse("/html").is_none());
        assert!(MediaType::parse("text/").is_none());
    }

    #[test]
    fn essence() {
        let mt = MediaType::parse("text/html; charset=utf-8").unwrap();
        assert_eq!(mt.essence(), "text/html");
    }

    #[test]
    fn is_text_types() {
        assert!(MediaType::text_plain().is_text());
        assert!(MediaType::text_html().is_text());
        assert!(MediaType::application_json().is_text());
        assert!(MediaType::application_xml().is_text());
        assert!(MediaType::new("application", "ld+json").is_text());
        assert!(!MediaType::image_png().is_text());
        assert!(!MediaType::application_octet_stream().is_text());
    }

    #[test]
    fn display_simple() {
        let mt = MediaType::text_html();
        assert_eq!(mt.to_string(), "text/html");
    }

    #[test]
    fn display_with_params() {
        let mt = MediaType::parse("text/html; charset=utf-8").unwrap();
        assert_eq!(mt.to_string(), "text/html; charset=utf-8");
    }

    #[test]
    fn display_quoted_param_with_space() {
        let mut mt = MediaType::text_plain();
        mt.set_param("name", "hello world");
        assert_eq!(mt.to_string(), r#"text/plain; name="hello world""#);
    }

    #[test]
    fn set_and_remove_param() {
        let mut mt = MediaType::text_plain();
        mt.set_param("charset", "ascii");
        assert_eq!(mt.charset(), Some("ascii"));
        mt.set_param("charset", "utf-8");
        assert_eq!(mt.charset(), Some("utf-8"));
        mt.remove_param("charset");
        assert!(mt.charset().is_none());
    }

    #[test]
    fn media_range_parse() {
        let r = MediaRange::parse("text/html;q=0.9").unwrap();
        assert_eq!(r.main_type, "text");
        assert_eq!(r.sub_type, "html");
        assert!((r.quality - 0.9).abs() < 1e-6);
    }

    #[test]
    fn media_range_wildcard() {
        let r = MediaRange::parse("*/*").unwrap();
        assert_eq!(r.main_type, "*");
        assert_eq!(r.sub_type, "*");
        assert!((r.quality - 1.0).abs() < 1e-6);
    }

    #[test]
    fn media_range_specificity() {
        let all = MediaRange::parse("*/*").unwrap();
        let typed = MediaRange::parse("text/*").unwrap();
        let exact = MediaRange::parse("text/html").unwrap();
        assert_eq!(all.specificity(), 0);
        assert_eq!(typed.specificity(), 1);
        assert_eq!(exact.specificity(), 2);
    }

    #[test]
    fn parse_accept_header() {
        let ranges = MediaRange::parse_accept("text/html, application/json;q=0.9, */*;q=0.1");
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].sub_type, "html");
        assert_eq!(ranges[1].sub_type, "json");
        assert_eq!(ranges[2].sub_type, "*");
    }

    #[test]
    fn matches_range_exact() {
        let mt = MediaType::text_html();
        let range = MediaRange::parse("text/html").unwrap();
        assert!(mt.matches_range(&range));
    }

    #[test]
    fn matches_range_subtype_wildcard() {
        let mt = MediaType::text_html();
        let range = MediaRange::parse("text/*").unwrap();
        assert!(mt.matches_range(&range));
    }

    #[test]
    fn matches_range_full_wildcard() {
        let mt = MediaType::image_png();
        let range = MediaRange::parse("*/*").unwrap();
        assert!(mt.matches_range(&range));
    }

    #[test]
    fn matches_range_mismatch() {
        let mt = MediaType::image_png();
        let range = MediaRange::parse("text/*").unwrap();
        assert!(!mt.matches_range(&range));
    }

    #[test]
    fn negotiate_selects_best() {
        let available = vec![
            MediaType::application_json(),
            MediaType::text_html(),
        ];
        let result = negotiate("text/html, application/json;q=0.9", &available).unwrap();
        assert_eq!(result.essence(), "text/html");
    }

    #[test]
    fn negotiate_no_match() {
        let available = vec![MediaType::image_png()];
        let result = negotiate("text/html", &available);
        assert!(result.is_none());
    }

    #[test]
    fn negotiate_wildcard_fallback() {
        let available = vec![MediaType::application_json()];
        let result = negotiate("text/html, */*;q=0.1", &available).unwrap();
        assert_eq!(result.essence(), "application/json");
    }

    #[test]
    fn from_extension_known() {
        assert_eq!(from_extension("html").essence(), "text/html");
        assert_eq!(from_extension("json").essence(), "application/json");
        assert_eq!(from_extension("png").essence(), "image/png");
        assert_eq!(from_extension("wasm").essence(), "application/wasm");
        assert_eq!(from_extension("csv").essence(), "text/csv");
        assert_eq!(from_extension("mp4").essence(), "video/mp4");
    }

    #[test]
    fn from_extension_unknown() {
        assert_eq!(from_extension("xyz").essence(), "application/octet-stream");
    }

    #[test]
    fn from_extension_case_insensitive() {
        assert_eq!(from_extension("HTML").essence(), "text/html");
        assert_eq!(from_extension("JSON").essence(), "application/json");
    }

    #[test]
    fn content_type_builder() {
        let ct = ContentTypeBuilder::new("text", "html")
            .charset("utf-8")
            .build();
        assert_eq!(ct.to_string(), "text/html; charset=utf-8");
    }

    #[test]
    fn content_type_builder_multipart() {
        let ct = ContentTypeBuilder::new("multipart", "form-data")
            .boundary("----abc123")
            .build();
        assert_eq!(ct.boundary(), Some("----abc123"));
    }

    #[test]
    fn content_type_builder_string() {
        let s = ContentTypeBuilder::new("application", "json")
            .charset("utf-8")
            .build_string();
        assert_eq!(s, "application/json; charset=utf-8");
    }

    #[test]
    fn content_type_builder_from_media_type() {
        let mt = MediaType::text_plain();
        let ct = ContentTypeBuilder::from_media_type(mt)
            .charset("ascii")
            .build();
        assert_eq!(ct.to_string(), "text/plain; charset=ascii");
    }

    #[test]
    fn common_types() {
        assert_eq!(MediaType::text_plain().essence(), "text/plain");
        assert_eq!(MediaType::text_css().essence(), "text/css");
        assert_eq!(MediaType::image_svg().essence(), "image/svg+xml");
        assert_eq!(
            MediaType::application_form_urlencoded().essence(),
            "application/x-www-form-urlencoded"
        );
        assert_eq!(
            MediaType::multipart_form_data().essence(),
            "multipart/form-data"
        );
    }

    #[test]
    fn media_range_display() {
        let r = MediaRange::parse("text/html;q=0.8").unwrap();
        assert_eq!(r.to_string(), "text/html; q=0.8");

        let r2 = MediaRange::parse("text/html").unwrap();
        assert_eq!(r2.to_string(), "text/html");
    }

    #[test]
    fn quality_clamping() {
        let r = MediaRange::parse("text/html;q=1.5").unwrap();
        assert!((r.quality - 1.0).abs() < 1e-6);

        let r2 = MediaRange::parse("text/html;q=-0.5").unwrap();
        assert!((r2.quality - 0.0).abs() < 1e-6);
    }

    #[test]
    fn negotiate_respects_zero_quality() {
        let available = vec![MediaType::text_html(), MediaType::application_json()];
        let result = negotiate("text/html;q=0, application/json", &available).unwrap();
        assert_eq!(result.essence(), "application/json");
    }
}
