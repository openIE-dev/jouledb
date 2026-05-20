//! XSS protection — HTML entity encoding, attribute escaping, DOM purifier.
//!
//! Replaces DOMPurify, xss-filters, and he.js with a pure-Rust XSS
//! prevention toolkit.  Includes HTML entity encoding, attribute escaping,
//! JavaScript string escaping, URL sanitization, CSS value escaping,
//! allowlist-based DOM purifier, and template auto-escaping.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Errors ─────────────────────────────────────────────────────

/// XSS protection errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XssError {
    /// URL contains a dangerous scheme.
    DangerousScheme(String),
    /// Tag not in allowlist.
    DisallowedTag(String),
    /// Attribute not in allowlist.
    DisallowedAttribute { tag: String, attr: String },
}

impl std::fmt::Display for XssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DangerousScheme(s) => write!(f, "dangerous URL scheme: {s}"),
            Self::DisallowedTag(t) => write!(f, "disallowed tag: {t}"),
            Self::DisallowedAttribute { tag, attr } => {
                write!(f, "disallowed attribute '{attr}' on <{tag}>")
            }
        }
    }
}

impl std::error::Error for XssError {}

// ── HTML entity encoding ───────────────────────────────────────

/// Encode HTML special characters (&, <, >, ", ').
pub fn html_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Decode HTML entities back to characters.
pub fn html_decode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '&' {
            let mut entity = String::new();
            for c in chars.by_ref() {
                if c == ';' {
                    break;
                }
                entity.push(c);
            }
            match entity.as_str() {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                s if s.starts_with("#x") || s.starts_with("#X") => {
                    if let Ok(cp) = u32::from_str_radix(&s[2..], 16) {
                        if let Some(c) = char::from_u32(cp) {
                            out.push(c);
                        }
                    }
                }
                s if s.starts_with('#') => {
                    if let Ok(cp) = s[1..].parse::<u32>() {
                        if let Some(c) = char::from_u32(cp) {
                            out.push(c);
                        }
                    }
                }
                _ => {
                    out.push('&');
                    out.push_str(&entity);
                    out.push(';');
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

// ── Attribute value escaping ───────────────────────────────────

/// Escape a value for use inside an HTML attribute (double-quoted context).
pub fn escape_attribute(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\'' => out.push_str("&#x27;"),
            '/' => out.push_str("&#x2F;"),
            '`' => out.push_str("&#96;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── JavaScript string escaping ─────────────────────────────────

/// Escape a string for safe embedding inside a JavaScript string literal.
pub fn escape_js_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003C"),
            '>' => out.push_str("\\u003E"),
            '/' => out.push_str("\\/"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            _ => out.push(ch),
        }
    }
    out
}

// ── CSS value escaping ─────────────────────────────────────────

/// Escape a value for safe use in a CSS property value.
pub fn escape_css_value(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\'' => out.push_str("\\'"),
            '<' => out.push_str("\\3C "),
            '>' => out.push_str("\\3E "),
            '&' => out.push_str("\\26 "),
            '/' => out.push_str("\\2F "),
            '(' => out.push_str("\\28 "),
            ')' => out.push_str("\\29 "),
            '{' => out.push_str("\\7B "),
            '}' => out.push_str("\\7D "),
            ';' => out.push_str("\\3B "),
            _ => out.push(ch),
        }
    }
    out
}

// ── URL sanitization ───────────────────────────────────────────

/// Safe URL schemes.
const SAFE_SCHEMES: &[&str] = &["http", "https", "mailto", "tel", "ftp", "ftps"];

/// Sanitize a URL: reject dangerous schemes (javascript:, data:, vbscript:).
pub fn sanitize_url(url: &str) -> Result<String, XssError> {
    let trimmed = url.trim();
    let lower = trimmed.to_lowercase();

    // Strip leading whitespace and control chars.
    let clean: String = lower
        .chars()
        .filter(|c| !c.is_control())
        .collect();

    // Check for dangerous schemes.
    if clean.starts_with("javascript:") || clean.starts_with("vbscript:")
        || clean.starts_with("data:text/html")
    {
        return Err(XssError::DangerousScheme(trimmed.to_string()));
    }

    // If has a scheme, verify it's safe.
    if let Some(colon_pos) = clean.find(':') {
        let scheme = &clean[..colon_pos];
        if !scheme.is_empty()
            && scheme.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        {
            if !SAFE_SCHEMES.contains(&scheme) && !clean.starts_with("data:image/") {
                return Err(XssError::DangerousScheme(scheme.to_string()));
            }
        }
    }

    Ok(trimmed.to_string())
}

// ── DOM Purifier ───────────────────────────────────────────────

/// Allowlist-based HTML purifier configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurifierConfig {
    /// Allowed tag names (lowercase).
    pub allowed_tags: HashSet<String>,
    /// Allowed attributes per tag.  "*" key means allowed on all tags.
    pub allowed_attrs: HashMap<String, HashSet<String>>,
    /// Allow data: URIs in img src.
    pub allow_data_images: bool,
}

impl Default for PurifierConfig {
    fn default() -> Self {
        let mut tags = HashSet::new();
        for t in &["p", "br", "b", "i", "em", "strong", "a", "ul", "ol", "li",
                    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "code",
                    "pre", "span", "div", "img", "table", "thead", "tbody",
                    "tr", "th", "td", "dl", "dt", "dd", "sub", "sup"] {
            tags.insert(t.to_string());
        }

        let mut attrs = HashMap::new();
        // Global attrs.
        let mut global = HashSet::new();
        for a in &["class", "id", "title", "lang", "dir"] {
            global.insert(a.to_string());
        }
        attrs.insert("*".to_string(), global);

        let mut a_attrs = HashSet::new();
        for a in &["href", "rel", "target"] {
            a_attrs.insert(a.to_string());
        }
        attrs.insert("a".to_string(), a_attrs);

        let mut img_attrs = HashSet::new();
        for a in &["src", "alt", "width", "height"] {
            img_attrs.insert(a.to_string());
        }
        attrs.insert("img".to_string(), img_attrs);

        Self {
            allowed_tags: tags,
            allowed_attrs: attrs,
            allow_data_images: false,
        }
    }
}

impl PurifierConfig {
    /// Check if a tag is allowed.
    pub fn is_tag_allowed(&self, tag: &str) -> bool {
        self.allowed_tags.contains(&tag.to_lowercase())
    }

    /// Check if an attribute is allowed on a tag.
    pub fn is_attr_allowed(&self, tag: &str, attr: &str) -> bool {
        let tag_lower = tag.to_lowercase();
        let attr_lower = attr.to_lowercase();

        // Check global attrs.
        if let Some(global) = self.allowed_attrs.get("*") {
            if global.contains(&attr_lower) {
                return true;
            }
        }
        // Check tag-specific attrs.
        if let Some(tag_attrs) = self.allowed_attrs.get(&tag_lower) {
            return tag_attrs.contains(&attr_lower);
        }
        false
    }
}

/// Purify HTML by stripping disallowed tags and attributes.
///
/// This is a simplified parser — strips all tags not in the allowlist.
/// For production use, a full HTML parser is recommended.
pub fn purify_html(input: &str, config: &PurifierConfig) -> String {
    let mut output = String::with_capacity(input.len());
    let mut source = input.to_string();

    loop {
        let mut chars = source.chars().peekable();
        let mut new_output = String::new();
        let mut restart_with: Option<String> = None;

        while let Some(ch) = chars.next() {
            if ch == '<' {
                // Parse tag.
                let mut tag_content = String::new();
                let mut in_tag = true;
                for c in chars.by_ref() {
                    if c == '>' {
                        in_tag = false;
                        break;
                    }
                    tag_content.push(c);
                }
                if in_tag {
                    // Unclosed tag — strip it.
                    continue;
                }

                let trimmed = tag_content.trim();
                let is_closing = trimmed.starts_with('/');
                let tag_str = if is_closing { &trimmed[1..] } else { trimmed };

                // Extract tag name.
                let tag_name = tag_str
                    .split(|c: char| c.is_whitespace() || c == '/')
                    .next()
                    .unwrap_or("")
                    .to_lowercase();

                if tag_name.is_empty() {
                    continue;
                }

                if !config.is_tag_allowed(&tag_name) {
                    // Strip disallowed tag (but keep content for non-script tags).
                    if tag_name == "script" || tag_name == "style" || tag_name == "iframe"
                        || tag_name == "object" || tag_name == "embed" || tag_name == "form"
                    {
                        // Also strip content until closing tag.
                        if !is_closing {
                            let close_tag = format!("</{tag_name}>");
                            let remaining: String = chars.collect();
                            let after = if let Some(pos) = remaining.to_lowercase().find(&close_tag) {
                                remaining[pos + close_tag.len()..].to_string()
                            } else {
                                remaining
                            };
                            // Restart processing with the remaining content
                            output.push_str(&new_output);
                            restart_with = Some(after);
                            break;
                        }
                    }
                    continue;
                }

                // Filter attributes.
                if is_closing {
                    new_output.push_str(&format!("</{tag_name}>"));
                } else {
                    new_output.push('<');
                    new_output.push_str(&tag_name);

                    // Parse attributes (simplified).
                    let attr_str = &tag_str[tag_name.len()..];
                    let attrs = parse_attributes(attr_str);
                    for (name, value) in &attrs {
                        if config.is_attr_allowed(&tag_name, name) {
                            // Sanitize href/src attributes.
                            if (name == "href" || name == "src") && sanitize_url(value).is_err() {
                                continue;
                            }
                            new_output.push(' ');
                            new_output.push_str(name);
                            new_output.push_str("=\"");
                            new_output.push_str(&escape_attribute(value));
                            new_output.push('"');
                        }
                    }

                    let self_closing = trimmed.ends_with('/');
                    if self_closing {
                        new_output.push_str(" /");
                    }
                    new_output.push('>');
                }
            } else {
                new_output.push(ch);
            }
        }

        output.push_str(&new_output);
        match restart_with {
            Some(remaining) => source = remaining,
            None => break,
        }
    }

    output
}

/// Simple attribute parser.
fn parse_attributes(attr_str: &str) -> Vec<(String, String)> {
    let mut attrs = Vec::new();
    let trimmed = attr_str.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return attrs;
    }

    let mut remaining = trimmed;
    while !remaining.is_empty() {
        remaining = remaining.trim_start();
        if remaining.is_empty() {
            break;
        }

        // Find attribute name.
        let name_end = remaining
            .find(|c: char| c == '=' || c.is_whitespace())
            .unwrap_or(remaining.len());
        let name = remaining[..name_end].to_lowercase();
        remaining = &remaining[name_end..];
        remaining = remaining.trim_start();

        if name.is_empty() {
            break;
        }

        if remaining.starts_with('=') {
            remaining = remaining[1..].trim_start();
            // Parse value.
            if remaining.starts_with('"') {
                remaining = &remaining[1..];
                let end = remaining.find('"').unwrap_or(remaining.len());
                let value = remaining[..end].to_string();
                remaining = if end < remaining.len() { &remaining[end + 1..] } else { "" };
                attrs.push((name, value));
            } else if remaining.starts_with('\'') {
                remaining = &remaining[1..];
                let end = remaining.find('\'').unwrap_or(remaining.len());
                let value = remaining[..end].to_string();
                remaining = if end < remaining.len() { &remaining[end + 1..] } else { "" };
                attrs.push((name, value));
            } else {
                let end = remaining
                    .find(|c: char| c.is_whitespace())
                    .unwrap_or(remaining.len());
                let value = remaining[..end].to_string();
                remaining = &remaining[end..];
                attrs.push((name, value));
            }
        } else {
            attrs.push((name, String::new()));
        }
    }
    attrs
}

// ── Template auto-escaping ─────────────────────────────────────

/// Context for template auto-escaping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscapeContext {
    Html,
    Attribute,
    JavaScript,
    Css,
    Url,
}

/// Auto-escape a value based on its template context.
pub fn auto_escape(value: &str, context: EscapeContext) -> String {
    match context {
        EscapeContext::Html => html_encode(value),
        EscapeContext::Attribute => escape_attribute(value),
        EscapeContext::JavaScript => escape_js_string(value),
        EscapeContext::Css => escape_css_value(value),
        EscapeContext::Url => {
            // Percent-encode special chars.
            let mut out = String::with_capacity(value.len());
            for ch in value.chars() {
                if ch.is_ascii_alphanumeric() || "-._~".contains(ch) {
                    out.push(ch);
                } else {
                    for b in ch.to_string().as_bytes() {
                        out.push_str(&format!("%{b:02X}"));
                    }
                }
            }
            out
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_encode() {
        assert_eq!(html_encode("<script>alert('xss')</script>"),
                   "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;");
    }

    #[test]
    fn test_html_encode_ampersand() {
        assert_eq!(html_encode("a & b"), "a &amp; b");
    }

    #[test]
    fn test_html_decode() {
        assert_eq!(html_decode("&lt;b&gt;hi&lt;/b&gt;"), "<b>hi</b>");
        assert_eq!(html_decode("&#x27;"), "'");
        assert_eq!(html_decode("&#39;"), "'");
    }

    #[test]
    fn test_html_encode_decode_roundtrip() {
        let input = "<div class=\"test\">'hello' & goodbye</div>";
        let encoded = html_encode(input);
        let decoded = html_decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_escape_attribute() {
        assert_eq!(
            escape_attribute("\" onclick=\"alert(1)"),
            "&quot; onclick=&quot;alert(1)"
        );
    }

    #[test]
    fn test_escape_js_string() {
        assert_eq!(escape_js_string("'; alert('xss');//"), "\\'; alert(\\'xss\\');\\/\\/");
    }

    #[test]
    fn test_escape_js_html_context() {
        assert_eq!(escape_js_string("</script>"), "\\u003C\\/script\\u003E");
    }

    #[test]
    fn test_escape_css_value() {
        assert_eq!(escape_css_value("expression(alert(1))"),
                   "expression\\28 alert\\28 1\\29 \\29 ");
    }

    #[test]
    fn test_sanitize_url_safe() {
        assert!(sanitize_url("https://example.com").is_ok());
        assert!(sanitize_url("mailto:test@example.com").is_ok());
        assert!(sanitize_url("/relative/path").is_ok());
    }

    #[test]
    fn test_sanitize_url_dangerous() {
        assert!(sanitize_url("javascript:alert(1)").is_err());
        assert!(sanitize_url("vbscript:msgbox(1)").is_err());
    }

    #[test]
    fn test_purify_html_strips_script() {
        let config = PurifierConfig::default();
        let input = "<p>Hello</p><script>alert('xss')</script><p>World</p>";
        let result = purify_html(input, &config);
        assert!(result.contains("<p>Hello</p>"));
        assert!(!result.contains("script"));
        assert!(result.contains("<p>World</p>"));
    }

    #[test]
    fn test_purify_html_allows_safe_tags() {
        let config = PurifierConfig::default();
        let input = "<p>Hello <b>world</b></p>";
        let result = purify_html(input, &config);
        assert_eq!(result, "<p>Hello <b>world</b></p>");
    }

    #[test]
    fn test_purify_html_strips_onclick() {
        let config = PurifierConfig::default();
        let input = r#"<a href="https://example.com" onclick="alert(1)">link</a>"#;
        let result = purify_html(input, &config);
        assert!(result.contains("href"));
        assert!(!result.contains("onclick"));
    }

    #[test]
    fn test_purify_html_strips_javascript_href() {
        let config = PurifierConfig::default();
        let input = r#"<a href="javascript:alert(1)">click</a>"#;
        let result = purify_html(input, &config);
        assert!(!result.contains("javascript"));
    }

    #[test]
    fn test_auto_escape_html() {
        assert_eq!(auto_escape("<script>", EscapeContext::Html), "&lt;script&gt;");
    }

    #[test]
    fn test_auto_escape_url() {
        assert_eq!(auto_escape("hello world", EscapeContext::Url), "hello%20world");
    }

    #[test]
    fn test_purifier_config_tag_check() {
        let config = PurifierConfig::default();
        assert!(config.is_tag_allowed("p"));
        assert!(config.is_tag_allowed("P")); // case insensitive
        assert!(!config.is_tag_allowed("script"));
    }

    #[test]
    fn test_purifier_config_attr_check() {
        let config = PurifierConfig::default();
        assert!(config.is_attr_allowed("p", "class"));
        assert!(config.is_attr_allowed("a", "href"));
        assert!(!config.is_attr_allowed("p", "onclick"));
    }
}
