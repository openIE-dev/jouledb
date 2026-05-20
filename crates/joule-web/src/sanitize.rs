//! HTML sanitization: whitelist-based tag/attribute filtering to prevent XSS.
//!
//! Replaces DOMPurify / sanitize-html. Character-by-character scanner with
//! configurable allow-lists for tags, attributes, and URL protocols.

use std::collections::{HashMap, HashSet};

// ── Configuration ───────────────────────────────────────────────

/// Whitelist-based sanitization configuration.
#[derive(Debug, Clone)]
pub struct SanitizeConfig {
    pub allowed_tags: HashSet<String>,
    pub allowed_attrs: HashMap<String, HashSet<String>>,
    pub allowed_protocols: HashSet<String>,
    pub strip_comments: bool,
    pub strip_empty: bool,
}

impl Default for SanitizeConfig {
    fn default() -> Self {
        let tags: &[&str] = &[
            "p", "br", "b", "i", "u", "em", "strong", "a", "ul", "ol", "li",
            "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "code", "pre",
            "span", "div", "img", "table", "thead", "tbody", "tr", "td", "th",
            "hr", "sub", "sup", "small", "mark", "del", "ins", "abbr",
            "details", "summary",
        ];

        let mut allowed_attrs: HashMap<String, HashSet<String>> = HashMap::new();
        allowed_attrs.insert(
            "a".into(),
            ["href", "title", "target", "rel"].iter().map(|s| (*s).into()).collect(),
        );
        allowed_attrs.insert(
            "img".into(),
            ["src", "alt", "title", "width", "height"].iter().map(|s| (*s).into()).collect(),
        );
        allowed_attrs.insert(
            "td".into(),
            ["colspan", "rowspan"].iter().map(|s| (*s).into()).collect(),
        );
        allowed_attrs.insert(
            "th".into(),
            ["colspan", "rowspan"].iter().map(|s| (*s).into()).collect(),
        );
        allowed_attrs.insert(
            "*".into(),
            ["class", "id"].iter().map(|s| (*s).into()).collect(),
        );

        let protocols: HashSet<String> =
            ["http", "https", "mailto"].iter().map(|s| (*s).into()).collect();

        Self {
            allowed_tags: tags.iter().map(|s| (*s).into()).collect(),
            allowed_attrs,
            allowed_protocols: protocols,
            strip_comments: true,
            strip_empty: false,
        }
    }
}

impl SanitizeConfig {
    /// Strict mode: only inline text elements.
    pub fn strict() -> Self {
        let tags: &[&str] = &["p", "br", "b", "i", "em", "strong", "span", "a"];
        let mut cfg = Self::default();
        cfg.allowed_tags = tags.iter().map(|s| (*s).into()).collect();
        cfg
    }

    /// Strip all HTML tags.
    pub fn none() -> Self {
        Self {
            allowed_tags: HashSet::new(),
            allowed_attrs: HashMap::new(),
            allowed_protocols: HashSet::new(),
            strip_comments: true,
            strip_empty: false,
        }
    }

    /// Allow an additional tag.
    pub fn allow_tag(&mut self, tag: &str) -> &mut Self {
        self.allowed_tags.insert(tag.to_lowercase());
        self
    }

    /// Allow an attribute on a specific tag.
    pub fn allow_attr(&mut self, tag: &str, attr: &str) -> &mut Self {
        self.allowed_attrs
            .entry(tag.to_lowercase())
            .or_default()
            .insert(attr.to_lowercase());
        self
    }

    /// Allow an attribute on all tags.
    pub fn allow_global_attr(&mut self, attr: &str) -> &mut Self {
        self.allowed_attrs
            .entry("*".into())
            .or_default()
            .insert(attr.to_lowercase());
        self
    }

    /// Remove a tag from the allow-list.
    pub fn deny_tag(&mut self, tag: &str) -> &mut Self {
        self.allowed_tags.remove(&tag.to_lowercase());
        self
    }

    /// Allow a URL protocol for href/src attributes.
    pub fn allow_protocol(&mut self, proto: &str) -> &mut Self {
        self.allowed_protocols.insert(proto.to_lowercase());
        self
    }
}

// ── Self-closing tags ───────────────────────────────────────────

const VOID_ELEMENTS: &[&str] = &[
    "br", "hr", "img", "input", "meta", "link", "area", "base", "col",
    "embed", "source", "track", "wbr",
];

fn is_void_element(tag: &str) -> bool {
    VOID_ELEMENTS.contains(&tag)
}

// ── Attribute checking ──────────────────────────────────────────

fn is_attr_allowed(config: &SanitizeConfig, tag: &str, attr: &str) -> bool {
    let attr_lower = attr.to_lowercase();
    // Event handlers are never allowed.
    if attr_lower.starts_with("on") {
        return false;
    }
    if let Some(set) = config.allowed_attrs.get(tag) {
        if set.contains(&attr_lower) {
            return true;
        }
    }
    if let Some(global) = config.allowed_attrs.get("*") {
        if global.contains(&attr_lower) {
            return true;
        }
    }
    false
}

fn is_protocol_allowed(config: &SanitizeConfig, value: &str) -> bool {
    let trimmed = value.trim();
    // Check for javascript: and other dangerous protocols.
    if let Some(colon_pos) = trimmed.find(':') {
        let proto = trimmed[..colon_pos].trim().to_lowercase();
        // Relative URLs and fragment-only URLs are fine.
        if proto.contains('/') || proto.contains('.') || proto.is_empty() {
            return true;
        }
        config.allowed_protocols.contains(&proto)
    } else {
        // No protocol — relative URL, always allowed.
        true
    }
}

// ── Parsing helpers ─────────────────────────────────────────────

/// Parse an HTML tag from position (after the `<`). Returns (tag_name, attributes_string, is_closing, is_self_closing, consumed_len).
fn parse_tag(html: &str) -> Option<(String, Vec<(String, String)>, bool, bool, usize)> {
    let bytes = html.as_bytes();
    if bytes.is_empty() {
        return None;
    }

    let mut pos = 0;
    let is_closing = bytes.get(pos) == Some(&b'/');
    if is_closing {
        pos += 1;
    }

    // Skip whitespace.
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }

    // Read tag name.
    let tag_start = pos;
    while pos < bytes.len()
        && bytes[pos] != b'>'
        && bytes[pos] != b'/'
        && !bytes[pos].is_ascii_whitespace()
    {
        pos += 1;
    }
    if pos == tag_start {
        return None;
    }
    let tag_name = html[tag_start..pos].to_lowercase();

    // Parse attributes.
    let mut attrs = Vec::new();
    loop {
        // Skip whitespace.
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] == b'>' || bytes[pos] == b'/' {
            break;
        }
        // Read attribute name.
        let attr_start = pos;
        while pos < bytes.len()
            && bytes[pos] != b'='
            && bytes[pos] != b'>'
            && bytes[pos] != b'/'
            && !bytes[pos].is_ascii_whitespace()
        {
            pos += 1;
        }
        let attr_name = html[attr_start..pos].to_lowercase();
        // Skip whitespace.
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let attr_value;
        if pos < bytes.len() && bytes[pos] == b'=' {
            pos += 1;
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && (bytes[pos] == b'"' || bytes[pos] == b'\'') {
                let quote = bytes[pos];
                pos += 1;
                let val_start = pos;
                while pos < bytes.len() && bytes[pos] != quote {
                    pos += 1;
                }
                attr_value = html[val_start..pos].to_string();
                if pos < bytes.len() {
                    pos += 1; // skip closing quote
                }
            } else {
                let val_start = pos;
                while pos < bytes.len()
                    && bytes[pos] != b'>'
                    && bytes[pos] != b'/'
                    && !bytes[pos].is_ascii_whitespace()
                {
                    pos += 1;
                }
                attr_value = html[val_start..pos].to_string();
            }
        } else {
            attr_value = String::new();
        }
        if !attr_name.is_empty() {
            attrs.push((attr_name, attr_value));
        }
    }

    let mut is_self_closing = false;
    if pos < bytes.len() && bytes[pos] == b'/' {
        is_self_closing = true;
        pos += 1;
    }
    // Skip to closing >.
    while pos < bytes.len() && bytes[pos] != b'>' {
        pos += 1;
    }
    if pos < bytes.len() {
        pos += 1; // skip >
    }

    Some((tag_name, attrs, is_closing, is_self_closing, pos))
}

// ── Sanitize ────────────────────────────────────────────────────

/// Sanitize HTML according to the given configuration.
pub fn sanitize(html: &str, config: &SanitizeConfig) -> String {
    let mut result = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'<' {
            // Check for comment.
            if html[i..].starts_with("<!--") {
                if let Some(end) = html[i..].find("-->") {
                    if !config.strip_comments {
                        result.push_str(&html[i..i + end + 3]);
                    }
                    i += end + 3;
                    continue;
                }
            }

            // Parse the tag.
            if let Some((tag_name, attrs, is_closing, is_self_closing, consumed)) =
                parse_tag(&html[i + 1..])
            {
                if config.allowed_tags.contains(&tag_name) {
                    if is_closing {
                        result.push_str("</");
                        result.push_str(&tag_name);
                        result.push('>');
                    } else {
                        result.push('<');
                        result.push_str(&tag_name);
                        // Filter attributes.
                        for (name, value) in &attrs {
                            if !is_attr_allowed(config, &tag_name, name) {
                                continue;
                            }
                            // Check URL protocols for href/src.
                            if (name == "href" || name == "src")
                                && !is_protocol_allowed(config, value)
                            {
                                continue;
                            }
                            result.push(' ');
                            result.push_str(name);
                            result.push_str("=\"");
                            result.push_str(&value.replace('"', "&quot;"));
                            result.push('"');
                        }
                        if is_self_closing || is_void_element(&tag_name) {
                            result.push_str(" />");
                        } else {
                            result.push('>');
                        }
                    }
                }
                // Disallowed tag: just skip the tag markup, inner content flows through.
                i += 1 + consumed;
            } else {
                // Not a valid tag — escape it.
                result.push_str("&lt;");
                i += 1;
            }
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }

    result
}

/// Escape all HTML-special characters in plain text.
pub fn sanitize_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_disallowed_tags() {
        let config = SanitizeConfig::default();
        let out = sanitize("<script>alert('xss')</script>", &config);
        assert!(!out.contains("<script"));
        assert!(out.contains("alert('xss')"));
    }

    #[test]
    fn keeps_allowed_tags() {
        let config = SanitizeConfig::default();
        let out = sanitize("<p>Hello <strong>world</strong></p>", &config);
        assert!(out.contains("<p>"));
        assert!(out.contains("<strong>"));
        assert!(out.contains("</strong>"));
        assert!(out.contains("</p>"));
    }

    #[test]
    fn strips_disallowed_attributes() {
        let config = SanitizeConfig::default();
        let out = sanitize("<p style=\"color:red\" class=\"ok\">hi</p>", &config);
        assert!(!out.contains("style"));
        assert!(out.contains("class=\"ok\""));
    }

    #[test]
    fn strips_javascript_urls() {
        let config = SanitizeConfig::default();
        let out = sanitize("<a href=\"javascript:alert(1)\">click</a>", &config);
        assert!(!out.contains("javascript"));
        assert!(out.contains("<a>"));
    }

    #[test]
    fn keeps_https_urls() {
        let config = SanitizeConfig::default();
        let out = sanitize("<a href=\"https://example.com\">link</a>", &config);
        assert!(out.contains("href=\"https://example.com\""));
    }

    #[test]
    fn preserves_text_content() {
        let config = SanitizeConfig::default();
        let out = sanitize("<div>Hello &amp; goodbye</div>", &config);
        assert!(out.contains("Hello &amp; goodbye"));
    }

    #[test]
    fn handles_self_closing_tags() {
        let config = SanitizeConfig::default();
        let out = sanitize("<p>Line 1<br>Line 2</p>", &config);
        assert!(out.contains("<br"));
        assert!(out.contains("Line 1"));
        assert!(out.contains("Line 2"));
    }

    #[test]
    fn strips_comments() {
        let config = SanitizeConfig::default();
        let out = sanitize("<p>Hello<!-- secret -->world</p>", &config);
        assert!(!out.contains("secret"));
        assert!(out.contains("Helloworld"));
    }

    #[test]
    fn entities_preserved() {
        let config = SanitizeConfig::default();
        let out = sanitize("<p>&lt;script&gt;</p>", &config);
        assert!(out.contains("&lt;script&gt;"));
    }

    #[test]
    fn nested_tags() {
        let config = SanitizeConfig::default();
        let out = sanitize(
            "<div><ul><li><strong>Item</strong></li></ul></div>",
            &config,
        );
        assert!(out.contains("<ul>"));
        assert!(out.contains("<li>"));
        assert!(out.contains("<strong>Item</strong>"));
    }

    #[test]
    fn strict_mode() {
        let config = SanitizeConfig::strict();
        let out = sanitize("<table><tr><td>data</td></tr></table>", &config);
        assert!(!out.contains("<table"));
        assert!(out.contains("data"));
    }

    #[test]
    fn sanitize_text_escapes_all() {
        let out = sanitize_text("<script>alert('xss') & \"more\"</script>");
        assert!(!out.contains('<'));
        assert!(!out.contains('>'));
        assert!(out.contains("&lt;"));
        assert!(out.contains("&gt;"));
        assert!(out.contains("&amp;"));
        assert!(out.contains("&quot;"));
        assert!(out.contains("&#x27;"));
    }

    #[test]
    fn custom_config() {
        let mut config = SanitizeConfig::none();
        config.allow_tag("b");
        let out = sanitize("<b>bold</b><i>italic</i>", &config);
        assert!(out.contains("<b>bold</b>"));
        assert!(!out.contains("<i>"));
        assert!(out.contains("italic"));
    }

    #[test]
    fn empty_input() {
        let config = SanitizeConfig::default();
        assert_eq!(sanitize("", &config), "");
    }

    #[test]
    fn script_tag_removed() {
        let config = SanitizeConfig::default();
        let out = sanitize(
            "<p>Safe</p><script>document.cookie</script><p>Also safe</p>",
            &config,
        );
        assert!(!out.contains("<script"));
        assert!(out.contains("<p>Safe</p>"));
        assert!(out.contains("<p>Also safe</p>"));
    }

    #[test]
    fn inline_event_handlers_removed() {
        let config = SanitizeConfig::default();
        let out = sanitize("<p onclick=\"alert(1)\" class=\"ok\">hi</p>", &config);
        assert!(!out.contains("onclick"));
        assert!(out.contains("class=\"ok\""));
    }

    #[test]
    fn img_allowed_with_src() {
        let config = SanitizeConfig::default();
        let out = sanitize("<img src=\"https://img.png\" alt=\"pic\">", &config);
        assert!(out.contains("<img"));
        assert!(out.contains("src=\"https://img.png\""));
        assert!(out.contains("alt=\"pic\""));
    }
}
