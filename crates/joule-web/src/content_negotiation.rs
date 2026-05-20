//! HTTP content negotiation — Accept header parsing (media types, quality
//! values), best match selection, Accept-Language negotiation, Accept-Encoding
//! negotiation, Vary header generation, and type-specific serialization dispatch.
//!
//! Replaces `negotiator`, `accepts`, and similar JS content negotiation
//! libraries with a pure-Rust implementation following RFC 7231.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Content negotiation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NegotiationError {
    /// No acceptable content type found.
    NotAcceptable { available: Vec<String> },
    /// Invalid Accept header syntax.
    InvalidHeader(String),
    /// No acceptable language.
    NoAcceptableLanguage,
    /// No acceptable encoding.
    NoAcceptableEncoding,
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAcceptable { available } => {
                write!(f, "not acceptable; available: {}", available.join(", "))
            }
            Self::InvalidHeader(h) => write!(f, "invalid header: {h}"),
            Self::NoAcceptableLanguage => write!(f, "no acceptable language"),
            Self::NoAcceptableEncoding => write!(f, "no acceptable encoding"),
        }
    }
}

impl std::error::Error for NegotiationError {}

// ── Media Type ───────────────────────────────────────────────────

/// A parsed media type with optional parameters and quality value.
#[derive(Debug, Clone)]
pub struct MediaType {
    /// Main type (e.g., "text", "application", "*").
    pub main_type: String,
    /// Subtype (e.g., "html", "json", "*").
    pub subtype: String,
    /// Quality value (0.0 to 1.0).
    pub quality: f64,
    /// Additional parameters (e.g., charset=utf-8).
    pub params: HashMap<String, String>,
}

impl MediaType {
    /// Parse a single media type string like "text/html;q=0.9;charset=utf-8".
    pub fn parse(input: &str) -> Result<Self, NegotiationError> {
        let trimmed = input.trim();
        let parts: Vec<&str> = trimmed.split(';').collect();

        let type_part = parts[0].trim();
        let slash_pos = type_part
            .find('/')
            .ok_or_else(|| NegotiationError::InvalidHeader(format!("missing '/' in: {type_part}")))?;

        let main_type = type_part[..slash_pos].trim().to_lowercase();
        let subtype = type_part[slash_pos + 1..].trim().to_lowercase();

        let mut quality = 1.0;
        let mut params = HashMap::new();

        for part in &parts[1..] {
            let param = part.trim();
            if let Some(eq_pos) = param.find('=') {
                let key = param[..eq_pos].trim().to_lowercase();
                let val = param[eq_pos + 1..].trim().to_string();
                if key == "q" {
                    quality = val.parse::<f64>().unwrap_or(1.0).clamp(0.0, 1.0);
                } else {
                    params.insert(key, val);
                }
            }
        }

        Ok(Self {
            main_type,
            subtype,
            quality,
            params,
        })
    }

    /// Full media type string (e.g., "text/html").
    pub fn mime(&self) -> String {
        format!("{}/{}", self.main_type, self.subtype)
    }

    /// Whether this is a wildcard type (*/*).
    pub fn is_wildcard(&self) -> bool {
        self.main_type == "*" && self.subtype == "*"
    }

    /// Whether the subtype is a wildcard (e.g., text/*).
    pub fn is_subtype_wildcard(&self) -> bool {
        self.subtype == "*"
    }

    /// Specificity score: more specific = higher.
    fn specificity(&self) -> u32 {
        let mut score = 0;
        if self.main_type != "*" {
            score += 100;
        }
        if self.subtype != "*" {
            score += 10;
        }
        score += self.params.len() as u32;
        score
    }

    /// Check if this media type matches another (considering wildcards).
    pub fn matches(&self, other: &str) -> bool {
        if self.is_wildcard() {
            return true;
        }
        if let Ok(other_mt) = MediaType::parse(other) {
            if self.main_type != other_mt.main_type && self.main_type != "*" {
                return false;
            }
            if self.is_subtype_wildcard() {
                return true;
            }
            self.subtype == other_mt.subtype
        } else {
            false
        }
    }
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.main_type, self.subtype)?;
        if (self.quality - 1.0).abs() > f64::EPSILON {
            write!(f, ";q={:.1}", self.quality)?;
        }
        for (k, v) in &self.params {
            write!(f, ";{k}={v}")?;
        }
        Ok(())
    }
}

impl PartialEq for MediaType {
    fn eq(&self, other: &Self) -> bool {
        self.main_type == other.main_type && self.subtype == other.subtype
    }
}

impl Eq for MediaType {}

// ── Accept Header Parsing ────────────────────────────────────────

/// Parse an Accept header value into a sorted list of media types.
/// Sorted by quality (descending) then specificity (descending).
pub fn parse_accept(header: &str) -> Result<Vec<MediaType>, NegotiationError> {
    let mut types = Vec::new();
    for part in header.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            types.push(MediaType::parse(trimmed)?);
        }
    }
    types.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.specificity().cmp(&a.specificity()))
    });
    Ok(types)
}

/// Select the best matching content type from the available types
/// given the client's Accept header.
pub fn negotiate_content_type(
    accept_header: &str,
    available: &[&str],
) -> Result<String, NegotiationError> {
    let accepted = parse_accept(accept_header)?;

    if accepted.is_empty() && !available.is_empty() {
        // No Accept header means accept anything
        return Ok(available[0].to_string());
    }

    for accept_mt in &accepted {
        if accept_mt.quality == 0.0 {
            continue;
        }
        for avail in available {
            if accept_mt.matches(avail) {
                return Ok(avail.to_string());
            }
        }
    }

    Err(NegotiationError::NotAcceptable {
        available: available.iter().map(|s| s.to_string()).collect(),
    })
}

// ── Accept-Language ──────────────────────────────────────────────

/// A parsed language tag with quality value.
#[derive(Debug, Clone)]
pub struct LanguageTag {
    pub tag: String,
    pub quality: f64,
}

/// Parse an Accept-Language header.
pub fn parse_accept_language(header: &str) -> Vec<LanguageTag> {
    let mut tags = Vec::new();
    for part in header.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let segments: Vec<&str> = trimmed.split(';').collect();
        let tag = segments[0].trim().to_lowercase();
        let mut quality = 1.0;
        for seg in &segments[1..] {
            let s = seg.trim();
            if let Some(qval) = s.strip_prefix("q=") {
                quality = qval.parse::<f64>().unwrap_or(1.0).clamp(0.0, 1.0);
            }
        }
        tags.push(LanguageTag { tag, quality });
    }
    tags.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    tags
}

/// Negotiate the best language from available options.
pub fn negotiate_language(
    accept_language: &str,
    available: &[&str],
) -> Result<String, NegotiationError> {
    let tags = parse_accept_language(accept_language);

    for lang_tag in &tags {
        if lang_tag.quality == 0.0 {
            continue;
        }
        // Exact match
        for avail in available {
            if avail.to_lowercase() == lang_tag.tag {
                return Ok(avail.to_string());
            }
        }
        // Prefix match (e.g., "en" matches "en-US")
        for avail in available {
            let avail_lower = avail.to_lowercase();
            if avail_lower.starts_with(&lang_tag.tag)
                || lang_tag.tag.starts_with(&avail_lower)
            {
                return Ok(avail.to_string());
            }
        }
        // Wildcard
        if lang_tag.tag == "*" && !available.is_empty() {
            return Ok(available[0].to_string());
        }
    }

    Err(NegotiationError::NoAcceptableLanguage)
}

// ── Accept-Encoding ──────────────────────────────────────────────

/// A parsed encoding with quality value.
#[derive(Debug, Clone)]
pub struct EncodingTag {
    pub encoding: String,
    pub quality: f64,
}

/// Parse an Accept-Encoding header.
pub fn parse_accept_encoding(header: &str) -> Vec<EncodingTag> {
    let mut tags = Vec::new();
    for part in header.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let segments: Vec<&str> = trimmed.split(';').collect();
        let encoding = segments[0].trim().to_lowercase();
        let mut quality = 1.0;
        for seg in &segments[1..] {
            let s = seg.trim();
            if let Some(qval) = s.strip_prefix("q=") {
                quality = qval.parse::<f64>().unwrap_or(1.0).clamp(0.0, 1.0);
            }
        }
        tags.push(EncodingTag { encoding, quality });
    }
    tags.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    tags
}

/// Negotiate the best encoding.
pub fn negotiate_encoding(
    accept_encoding: &str,
    available: &[&str],
) -> Result<String, NegotiationError> {
    let tags = parse_accept_encoding(accept_encoding);

    for enc_tag in &tags {
        if enc_tag.quality == 0.0 {
            continue;
        }
        if enc_tag.encoding == "*" && !available.is_empty() {
            return Ok(available[0].to_string());
        }
        for avail in available {
            if avail.to_lowercase() == enc_tag.encoding {
                return Ok(avail.to_string());
            }
        }
    }

    // Always allow "identity" if nothing matches
    if available.contains(&"identity") {
        return Ok("identity".to_string());
    }

    Err(NegotiationError::NoAcceptableEncoding)
}

// ── Vary Header ──────────────────────────────────────────────────

/// Collect the Vary header value based on which negotiation was performed.
pub struct VaryBuilder {
    headers: Vec<String>,
}

impl VaryBuilder {
    pub fn new() -> Self {
        Self {
            headers: Vec::new(),
        }
    }

    pub fn add(&mut self, header: &str) -> &mut Self {
        let lower = header.to_string();
        if !self.headers.contains(&lower) {
            self.headers.push(lower);
        }
        self
    }

    pub fn content_type(&mut self) -> &mut Self {
        self.add("Accept")
    }

    pub fn language(&mut self) -> &mut Self {
        self.add("Accept-Language")
    }

    pub fn encoding(&mut self) -> &mut Self {
        self.add("Accept-Encoding")
    }

    /// Build the Vary header value.
    pub fn build(&self) -> String {
        if self.headers.is_empty() {
            "*".to_string()
        } else {
            self.headers.join(", ")
        }
    }
}

impl Default for VaryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Serialization Dispatch ───────────────────────────────────────

/// Serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    Json,
    Xml,
    Yaml,
    Csv,
    Html,
    PlainText,
    MessagePack,
}

impl SerializationFormat {
    /// Get the media type for this format.
    pub fn media_type(&self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Xml => "application/xml",
            Self::Yaml => "application/yaml",
            Self::Csv => "text/csv",
            Self::Html => "text/html",
            Self::PlainText => "text/plain",
            Self::MessagePack => "application/msgpack",
        }
    }

    /// Resolve a media type string to a serialization format.
    pub fn from_media_type(mime: &str) -> Option<Self> {
        let lower = mime.to_lowercase();
        match lower.as_str() {
            "application/json" | "text/json" => Some(Self::Json),
            "application/xml" | "text/xml" => Some(Self::Xml),
            "application/yaml" | "application/x-yaml" | "text/yaml" => Some(Self::Yaml),
            "text/csv" => Some(Self::Csv),
            "text/html" => Some(Self::Html),
            "text/plain" => Some(Self::PlainText),
            "application/msgpack" | "application/x-msgpack" => Some(Self::MessagePack),
            _ => None,
        }
    }
}

/// Dispatch serialization based on content negotiation.
pub fn serialize_by_type(
    data: &serde_json::Value,
    format: SerializationFormat,
) -> Result<String, NegotiationError> {
    match format {
        SerializationFormat::Json => Ok(serde_json::to_string_pretty(data)
            .unwrap_or_else(|_| "null".to_string())),
        SerializationFormat::PlainText => Ok(format_as_text(data)),
        SerializationFormat::Html => Ok(format_as_html(data)),
        SerializationFormat::Csv => Ok(format_as_csv(data)),
        SerializationFormat::Xml => Ok(format_as_xml(data)),
        SerializationFormat::Yaml => Ok(format_as_yaml(data)),
        SerializationFormat::MessagePack => {
            // Represent as hex-encoded JSON for demonstration
            Ok(format!("msgpack:{}", serde_json::to_string(data).unwrap_or_default()))
        }
    }
}

fn format_as_text(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Array(arr) => {
            arr.iter()
                .map(|v| format_as_text(v))
                .collect::<Vec<_>>()
                .join("\n")
        }
        serde_json::Value::Object(obj) => {
            obj.iter()
                .map(|(k, v)| format!("{k}: {}", format_as_text(v)))
                .collect::<Vec<_>>()
                .join("\n")
        }
    }
}

fn format_as_html(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::Object(obj) => {
            let rows: String = obj
                .iter()
                .map(|(k, v)| format!("<tr><td>{k}</td><td>{v}</td></tr>"))
                .collect();
            format!("<table>{rows}</table>")
        }
        serde_json::Value::Array(arr) => {
            let items: String = arr
                .iter()
                .map(|v| format!("<li>{v}</li>"))
                .collect();
            format!("<ul>{items}</ul>")
        }
        other => format!("<p>{other}</p>"),
    }
}

fn format_as_csv(data: &serde_json::Value) -> String {
    match data {
        serde_json::Value::Array(arr) => {
            let mut lines = Vec::new();
            // Header from first object
            if let Some(first) = arr.first() {
                if let serde_json::Value::Object(obj) = first {
                    let header: Vec<&String> = obj.keys().collect();
                    lines.push(
                        header.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(","),
                    );
                    for item in arr {
                        if let serde_json::Value::Object(row) = item {
                            let vals: Vec<String> = header
                                .iter()
                                .map(|k| {
                                    row.get(k.as_str())
                                        .map(|v| csv_escape(v))
                                        .unwrap_or_default()
                                })
                                .collect();
                            lines.push(vals.join(","));
                        }
                    }
                }
            }
            lines.join("\n")
        }
        _ => format_as_text(data),
    }
}

fn csv_escape(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => {
            if s.contains(',') || s.contains('"') || s.contains('\n') {
                format!("\"{}\"", s.replace('"', "\"\""))
            } else {
                s.clone()
            }
        }
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn format_as_xml(data: &serde_json::Value) -> String {
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<root>\n");
    append_xml(&mut xml, data, "  ");
    xml.push_str("</root>");
    xml
}

fn append_xml(xml: &mut String, data: &serde_json::Value, indent: &str) {
    match data {
        serde_json::Value::Object(obj) => {
            for (k, v) in obj {
                xml.push_str(&format!("{indent}<{k}>"));
                match v {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        xml.push('\n');
                        let next_indent = format!("{indent}  ");
                        append_xml(xml, v, &next_indent);
                        xml.push_str(&format!("{indent}</{k}>\n"));
                    }
                    _ => {
                        xml.push_str(&xml_escape_text(v));
                        xml.push_str(&format!("</{k}>\n"));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                xml.push_str(&format!("{indent}<item>"));
                match v {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        xml.push('\n');
                        let next_indent = format!("{indent}  ");
                        append_xml(xml, v, &next_indent);
                        xml.push_str(&format!("{indent}</item>\n"));
                    }
                    _ => {
                        xml.push_str(&xml_escape_text(v));
                        xml.push_str("</item>\n");
                    }
                }
            }
        }
        other => {
            xml.push_str(&format!("{indent}{}\n", xml_escape_text(other)));
        }
    }
}

fn xml_escape_text(val: &serde_json::Value) -> String {
    let s = match val {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        other => other.to_string(),
    };
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn format_as_yaml(data: &serde_json::Value) -> String {
    let mut yaml = String::new();
    append_yaml(&mut yaml, data, 0);
    yaml
}

fn append_yaml(yaml: &mut String, data: &serde_json::Value, depth: usize) {
    let indent = "  ".repeat(depth);
    match data {
        serde_json::Value::Object(obj) => {
            for (k, v) in obj {
                match v {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        yaml.push_str(&format!("{indent}{k}:\n"));
                        append_yaml(yaml, v, depth + 1);
                    }
                    _ => {
                        yaml.push_str(&format!("{indent}{k}: {}\n", yaml_value(v)));
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                match v {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        yaml.push_str(&format!("{indent}-\n"));
                        append_yaml(yaml, v, depth + 1);
                    }
                    _ => {
                        yaml.push_str(&format!("{indent}- {}\n", yaml_value(v)));
                    }
                }
            }
        }
        _ => {
            yaml.push_str(&format!("{indent}{}\n", yaml_value(data)));
        }
    }
}

fn yaml_value(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::String(s) => {
            if s.contains(':') || s.contains('#') || s.is_empty() {
                format!("\"{s}\"")
            } else {
                s.clone()
            }
        }
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        _ => format!("{val}"),
    }
}

/// Full negotiation: parse Accept, find format, serialize, generate Vary.
pub fn negotiate_and_serialize(
    accept_header: &str,
    data: &serde_json::Value,
    available_types: &[&str],
) -> Result<NegotiatedResponse, NegotiationError> {
    let content_type = negotiate_content_type(accept_header, available_types)?;
    let format = SerializationFormat::from_media_type(&content_type)
        .ok_or_else(|| NegotiationError::NotAcceptable {
            available: available_types.iter().map(|s| s.to_string()).collect(),
        })?;
    let body = serialize_by_type(data, format)?;

    let mut vary = VaryBuilder::new();
    vary.content_type();

    Ok(NegotiatedResponse {
        content_type,
        body,
        vary: vary.build(),
        format,
    })
}

/// Result of content negotiation + serialization.
#[derive(Debug, Clone)]
pub struct NegotiatedResponse {
    pub content_type: String,
    pub body: String,
    pub vary: String,
    pub format: SerializationFormat,
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_media_type() {
        let mt = MediaType::parse("text/html").unwrap();
        assert_eq!(mt.main_type, "text");
        assert_eq!(mt.subtype, "html");
        assert!((mt.quality - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_media_type_with_quality() {
        let mt = MediaType::parse("text/html;q=0.9").unwrap();
        assert!((mt.quality - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_media_type_with_params() {
        let mt = MediaType::parse("text/html;charset=utf-8;q=0.8").unwrap();
        assert_eq!(mt.params.get("charset").unwrap(), "utf-8");
        assert!((mt.quality - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn media_type_wildcard() {
        let mt = MediaType::parse("*/*").unwrap();
        assert!(mt.is_wildcard());
    }

    #[test]
    fn media_type_subtype_wildcard() {
        let mt = MediaType::parse("text/*").unwrap();
        assert!(mt.is_subtype_wildcard());
        assert!(mt.matches("text/html"));
        assert!(!mt.matches("application/json"));
    }

    #[test]
    fn media_type_matching() {
        let mt = MediaType::parse("application/json").unwrap();
        assert!(mt.matches("application/json"));
        assert!(!mt.matches("text/html"));
    }

    #[test]
    fn parse_accept_header() {
        let types = parse_accept("text/html, application/json;q=0.9, */*;q=0.1").unwrap();
        assert_eq!(types.len(), 3);
        assert_eq!(types[0].mime(), "text/html");
        assert_eq!(types[1].mime(), "application/json");
        assert_eq!(types[2].mime(), "*/*");
    }

    #[test]
    fn negotiate_json() {
        let result = negotiate_content_type(
            "application/json, text/html;q=0.9",
            &["text/html", "application/json"],
        )
        .unwrap();
        assert_eq!(result, "application/json");
    }

    #[test]
    fn negotiate_html_preferred() {
        let result = negotiate_content_type(
            "text/html, application/json;q=0.5",
            &["application/json", "text/html"],
        )
        .unwrap();
        assert_eq!(result, "text/html");
    }

    #[test]
    fn negotiate_wildcard() {
        let result = negotiate_content_type(
            "*/*",
            &["application/json", "text/html"],
        )
        .unwrap();
        assert_eq!(result, "application/json");
    }

    #[test]
    fn negotiate_not_acceptable() {
        let result = negotiate_content_type(
            "application/xml",
            &["application/json", "text/html"],
        );
        assert!(matches!(result, Err(NegotiationError::NotAcceptable { .. })));
    }

    #[test]
    fn negotiate_quality_zero_excluded() {
        let result = negotiate_content_type(
            "text/html;q=0, application/json",
            &["text/html", "application/json"],
        )
        .unwrap();
        assert_eq!(result, "application/json");
    }

    #[test]
    fn parse_accept_language_tags() {
        let tags = parse_accept_language("en-US, en;q=0.9, fr;q=0.8");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0].tag, "en-us");
        assert!((tags[1].quality - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn negotiate_language_exact() {
        let result = negotiate_language("en-US, fr;q=0.9", &["en-US", "fr"]).unwrap();
        assert_eq!(result, "en-US");
    }

    #[test]
    fn negotiate_language_prefix() {
        let result = negotiate_language("en", &["en-US", "fr"]).unwrap();
        assert_eq!(result, "en-US");
    }

    #[test]
    fn negotiate_language_wildcard() {
        let result = negotiate_language("*", &["ja", "fr"]).unwrap();
        assert_eq!(result, "ja");
    }

    #[test]
    fn negotiate_language_none() {
        let result = negotiate_language("de", &["en", "fr"]);
        assert!(matches!(result, Err(NegotiationError::NoAcceptableLanguage)));
    }

    #[test]
    fn parse_accept_encoding_tags() {
        let encs = parse_accept_encoding("gzip, deflate;q=0.5, br;q=0.9");
        assert_eq!(encs.len(), 3);
        assert_eq!(encs[0].encoding, "gzip");
    }

    #[test]
    fn negotiate_encoding_match() {
        let result =
            negotiate_encoding("gzip, br;q=0.9", &["identity", "gzip", "br"]).unwrap();
        assert_eq!(result, "gzip");
    }

    #[test]
    fn negotiate_encoding_fallback_identity() {
        let result = negotiate_encoding("br", &["identity", "gzip"]).unwrap();
        assert_eq!(result, "identity");
    }

    #[test]
    fn vary_builder() {
        let mut vary = VaryBuilder::new();
        vary.content_type().language().encoding();
        let header = vary.build();
        assert!(header.contains("Accept"));
        assert!(header.contains("Accept-Language"));
        assert!(header.contains("Accept-Encoding"));
    }

    #[test]
    fn vary_builder_no_duplicates() {
        let mut vary = VaryBuilder::new();
        vary.add("Accept").add("Accept");
        // Should only contain one "Accept"
        let built = vary.build();
        let parts: Vec<&str> = built.split(", ").collect();
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn serialization_format_roundtrip() {
        let fmt = SerializationFormat::from_media_type("application/json").unwrap();
        assert_eq!(fmt, SerializationFormat::Json);
        assert_eq!(fmt.media_type(), "application/json");
    }

    #[test]
    fn serialize_json() {
        let data = serde_json::json!({"name": "Alice"});
        let result = serialize_by_type(&data, SerializationFormat::Json).unwrap();
        assert!(result.contains("Alice"));
    }

    #[test]
    fn serialize_plain_text() {
        let data = serde_json::json!({"name": "Bob", "age": 30});
        let result = serialize_by_type(&data, SerializationFormat::PlainText).unwrap();
        assert!(result.contains("name: Bob"));
    }

    #[test]
    fn serialize_html() {
        let data = serde_json::json!({"name": "Carol"});
        let result = serialize_by_type(&data, SerializationFormat::Html).unwrap();
        assert!(result.contains("<table>"));
        assert!(result.contains("Carol"));
    }

    #[test]
    fn serialize_csv() {
        let data = serde_json::json!([
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]);
        let result = serialize_by_type(&data, SerializationFormat::Csv).unwrap();
        assert!(result.contains("Alice"));
        assert!(result.contains("Bob"));
    }

    #[test]
    fn serialize_xml() {
        let data = serde_json::json!({"name": "Dave"});
        let result = serialize_by_type(&data, SerializationFormat::Xml).unwrap();
        assert!(result.contains("<name>"));
        assert!(result.contains("Dave"));
    }

    #[test]
    fn full_negotiation() {
        let data = serde_json::json!({"key": "value"});
        let result = negotiate_and_serialize(
            "application/json",
            &data,
            &["application/json", "text/html"],
        )
        .unwrap();
        assert_eq!(result.content_type, "application/json");
        assert_eq!(result.format, SerializationFormat::Json);
        assert!(result.body.contains("value"));
        assert!(result.vary.contains("Accept"));
    }

    #[test]
    fn media_type_display() {
        let mt = MediaType::parse("text/html;q=0.9;charset=utf-8").unwrap();
        let s = mt.to_string();
        assert!(s.contains("text/html"));
        assert!(s.contains("q=0.9"));
    }

    #[test]
    fn error_display() {
        let err = NegotiationError::NotAcceptable {
            available: vec!["application/json".to_string()],
        };
        assert!(err.to_string().contains("not acceptable"));
    }

    #[test]
    fn invalid_media_type() {
        let result = MediaType::parse("invalid");
        assert!(result.is_err());
    }
}
