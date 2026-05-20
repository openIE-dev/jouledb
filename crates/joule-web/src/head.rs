//! Document head management — title, meta, link, script, and structured data.
//!
//! Replaces `react-helmet` and `next/head` with a pure-Rust manager suitable
//! for SSR and SPA navigation.

use serde_json::Value as JsonValue;

// ── Tag types ───────────────────────────────────────────────────────────────

/// A single tag that belongs in `<head>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadTag {
    Title(String),
    Meta {
        name: Option<String>,
        property: Option<String>,
        content: String,
        charset: Option<String>,
        http_equiv: Option<String>,
    },
    Link {
        rel: String,
        href: String,
        type_: Option<String>,
        sizes: Option<String>,
        crossorigin: Option<String>,
    },
    Script {
        src: Option<String>,
        content: Option<String>,
        type_: Option<String>,
        defer: bool,
        async_: bool,
    },
    Style(String),
    Base {
        href: String,
    },
}

// ── SEO config ──────────────────────────────────────────────────────────────

/// Convenience struct for applying common SEO tags in one call.
#[derive(Debug, Clone, Default)]
pub struct SeoConfig {
    pub title: Option<String>,
    pub description: Option<String>,
    pub canonical_url: Option<String>,
    pub og_title: Option<String>,
    pub og_description: Option<String>,
    pub og_image: Option<String>,
    pub og_type: Option<String>,
    pub twitter_card: Option<String>,
    pub twitter_site: Option<String>,
    pub robots: Option<String>,
}

// ── HeadManager ─────────────────────────────────────────────────────────────

/// Manages the set of `<head>` tags for the current page, with a scope stack
/// for nested route overrides.
pub struct HeadManager {
    tags: Vec<HeadTag>,
    stack: Vec<Vec<HeadTag>>,
}

impl HeadManager {
    pub fn new() -> Self {
        Self {
            tags: Vec::new(),
            stack: Vec::new(),
        }
    }

    // ── Title ───────────────────────────────────────────────────────────

    pub fn set_title(&mut self, title: &str) {
        // Replace existing title if present.
        self.tags.retain(|t| !matches!(t, HeadTag::Title(_)));
        self.tags.push(HeadTag::Title(title.to_string()));
    }

    pub fn title(&self) -> Option<&str> {
        self.tags.iter().rev().find_map(|t| {
            if let HeadTag::Title(s) = t {
                Some(s.as_str())
            } else {
                None
            }
        })
    }

    // ── Meta ────────────────────────────────────────────────────────────

    pub fn add_meta(&mut self, name: &str, content: &str) {
        self.tags.push(HeadTag::Meta {
            name: Some(name.to_string()),
            property: None,
            content: content.to_string(),
            charset: None,
            http_equiv: None,
        });
    }

    pub fn add_meta_property(&mut self, property: &str, content: &str) {
        self.tags.push(HeadTag::Meta {
            name: None,
            property: Some(property.to_string()),
            content: content.to_string(),
            charset: None,
            http_equiv: None,
        });
    }

    // ── Link ────────────────────────────────────────────────────────────

    pub fn add_link(&mut self, rel: &str, href: &str) {
        self.tags.push(HeadTag::Link {
            rel: rel.to_string(),
            href: href.to_string(),
            type_: None,
            sizes: None,
            crossorigin: None,
        });
    }

    // ── Script ──────────────────────────────────────────────────────────

    pub fn add_script_src(&mut self, src: &str) {
        self.tags.push(HeadTag::Script {
            src: Some(src.to_string()),
            content: None,
            type_: None,
            defer: false,
            async_: false,
        });
    }

    // ── Style ───────────────────────────────────────────────────────────

    pub fn add_inline_style(&mut self, css: &str) {
        self.tags.push(HeadTag::Style(css.to_string()));
    }

    // ── Scope stack (nested routes) ─────────────────────────────────────

    pub fn push_scope(&mut self) {
        self.stack.push(self.tags.clone());
    }

    pub fn pop_scope(&mut self) {
        if let Some(saved) = self.stack.pop() {
            self.tags = saved;
        }
    }

    pub fn clear(&mut self) {
        self.tags.clear();
    }

    // ── Accessors ───────────────────────────────────────────────────────

    pub fn tags(&self) -> &[HeadTag] {
        &self.tags
    }

    // ── Render to HTML (SSR) ────────────────────────────────────────────

    pub fn render_html(&self) -> String {
        let mut out = String::new();
        for tag in &self.tags {
            match tag {
                HeadTag::Title(t) => {
                    out.push_str(&format!("<title>{t}</title>\n"));
                }
                HeadTag::Meta {
                    name,
                    property,
                    content,
                    charset,
                    http_equiv,
                } => {
                    out.push_str("<meta");
                    if let Some(n) = name {
                        out.push_str(&format!(" name=\"{n}\""));
                    }
                    if let Some(p) = property {
                        out.push_str(&format!(" property=\"{p}\""));
                    }
                    out.push_str(&format!(" content=\"{content}\""));
                    if let Some(c) = charset {
                        out.push_str(&format!(" charset=\"{c}\""));
                    }
                    if let Some(h) = http_equiv {
                        out.push_str(&format!(" http-equiv=\"{h}\""));
                    }
                    out.push_str(">\n");
                }
                HeadTag::Link {
                    rel,
                    href,
                    type_,
                    sizes,
                    crossorigin,
                } => {
                    out.push_str(&format!("<link rel=\"{rel}\" href=\"{href}\""));
                    if let Some(t) = type_ {
                        out.push_str(&format!(" type=\"{t}\""));
                    }
                    if let Some(s) = sizes {
                        out.push_str(&format!(" sizes=\"{s}\""));
                    }
                    if let Some(c) = crossorigin {
                        out.push_str(&format!(" crossorigin=\"{c}\""));
                    }
                    out.push_str(">\n");
                }
                HeadTag::Script {
                    src,
                    content,
                    type_,
                    defer,
                    async_,
                } => {
                    out.push_str("<script");
                    if let Some(s) = src {
                        out.push_str(&format!(" src=\"{s}\""));
                    }
                    if let Some(t) = type_ {
                        out.push_str(&format!(" type=\"{t}\""));
                    }
                    if *defer {
                        out.push_str(" defer");
                    }
                    if *async_ {
                        out.push_str(" async");
                    }
                    out.push('>');
                    if let Some(c) = content {
                        out.push_str(c);
                    }
                    out.push_str("</script>\n");
                }
                HeadTag::Style(css) => {
                    out.push_str(&format!("<style>{css}</style>\n"));
                }
                HeadTag::Base { href } => {
                    out.push_str(&format!("<base href=\"{href}\">\n"));
                }
            }
        }
        out
    }

    // ── SEO ─────────────────────────────────────────────────────────────

    pub fn apply_seo(&mut self, config: &SeoConfig) {
        if let Some(ref title) = config.title {
            self.set_title(title);
        }
        if let Some(ref desc) = config.description {
            self.add_meta("description", desc);
        }
        if let Some(ref url) = config.canonical_url {
            self.add_link("canonical", url);
        }
        if let Some(ref t) = config.og_title {
            self.add_meta_property("og:title", t);
        }
        if let Some(ref d) = config.og_description {
            self.add_meta_property("og:description", d);
        }
        if let Some(ref img) = config.og_image {
            self.add_meta_property("og:image", img);
        }
        if let Some(ref ot) = config.og_type {
            self.add_meta_property("og:type", ot);
        }
        if let Some(ref tc) = config.twitter_card {
            self.add_meta("twitter:card", tc);
        }
        if let Some(ref ts) = config.twitter_site {
            self.add_meta("twitter:site", ts);
        }
        if let Some(ref robots) = config.robots {
            self.add_meta("robots", robots);
        }
    }

    // ── Structured data (JSON-LD) ───────────────────────────────────────

    pub fn add_json_ld(&mut self, data: &JsonValue) {
        let json = serde_json::to_string(data).unwrap_or_default();
        self.tags.push(HeadTag::Script {
            src: None,
            content: Some(json),
            type_: Some("application/ld+json".to_string()),
            defer: false,
            async_: false,
        });
    }

    // ── Favicon ─────────────────────────────────────────────────────────

    pub fn set_favicon(&mut self, href: &str) {
        self.tags.push(HeadTag::Link {
            rel: "icon".to_string(),
            href: href.to_string(),
            type_: None,
            sizes: None,
            crossorigin: None,
        });
    }

    pub fn set_favicon_svg(&mut self, href: &str) {
        self.tags.push(HeadTag::Link {
            rel: "icon".to_string(),
            href: href.to_string(),
            type_: Some("image/svg+xml".to_string()),
            sizes: None,
            crossorigin: None,
        });
    }
}

impl Default for HeadManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_title() {
        let mut hm = HeadManager::new();
        assert!(hm.title().is_none());
        hm.set_title("Hello");
        assert_eq!(hm.title(), Some("Hello"));
    }

    #[test]
    fn add_meta_renders() {
        let mut hm = HeadManager::new();
        hm.add_meta("description", "A page");
        let html = hm.render_html();
        assert!(html.contains("name=\"description\""));
        assert!(html.contains("content=\"A page\""));
    }

    #[test]
    fn opengraph_tags() {
        let mut hm = HeadManager::new();
        hm.add_meta_property("og:title", "My Title");
        let html = hm.render_html();
        assert!(html.contains("property=\"og:title\""));
        assert!(html.contains("content=\"My Title\""));
    }

    #[test]
    fn push_pop_scope_restores() {
        let mut hm = HeadManager::new();
        hm.set_title("Parent");
        hm.push_scope();
        hm.set_title("Child");
        assert_eq!(hm.title(), Some("Child"));
        hm.pop_scope();
        assert_eq!(hm.title(), Some("Parent"));
    }

    #[test]
    fn render_html_output() {
        let mut hm = HeadManager::new();
        hm.set_title("Test");
        hm.add_inline_style("body { margin: 0; }");
        let html = hm.render_html();
        assert!(html.contains("<title>Test</title>"));
        assert!(html.contains("<style>body { margin: 0; }</style>"));
    }

    #[test]
    fn seo_config_applies_all_tags() {
        let mut hm = HeadManager::new();
        let seo = SeoConfig {
            title: Some("SEO Title".into()),
            description: Some("desc".into()),
            canonical_url: Some("https://example.com".into()),
            og_title: Some("OG".into()),
            twitter_card: Some("summary".into()),
            robots: Some("noindex".into()),
            ..Default::default()
        };
        hm.apply_seo(&seo);

        assert_eq!(hm.title(), Some("SEO Title"));
        let html = hm.render_html();
        assert!(html.contains("name=\"description\""));
        assert!(html.contains("rel=\"canonical\""));
        assert!(html.contains("property=\"og:title\""));
        assert!(html.contains("twitter:card"));
        assert!(html.contains("robots"));
    }

    #[test]
    fn json_ld_structured_data() {
        let mut hm = HeadManager::new();
        let data = serde_json::json!({
            "@type": "Organization",
            "name": "Test"
        });
        hm.add_json_ld(&data);
        let html = hm.render_html();
        assert!(html.contains("application/ld+json"));
        assert!(html.contains("Organization"));
    }

    #[test]
    fn favicon_link_tag() {
        let mut hm = HeadManager::new();
        hm.set_favicon("/favicon.ico");
        let html = hm.render_html();
        assert!(html.contains("rel=\"icon\""));
        assert!(html.contains("href=\"/favicon.ico\""));
    }

    #[test]
    fn clear_empties() {
        let mut hm = HeadManager::new();
        hm.set_title("Gone");
        hm.add_meta("a", "b");
        hm.clear();
        assert!(hm.tags().is_empty());
        assert!(hm.title().is_none());
    }

    #[test]
    fn multiple_scopes_nested() {
        let mut hm = HeadManager::new();
        hm.set_title("Root");

        hm.push_scope();
        hm.set_title("Level 1");

        hm.push_scope();
        hm.set_title("Level 2");
        assert_eq!(hm.title(), Some("Level 2"));

        hm.pop_scope();
        assert_eq!(hm.title(), Some("Level 1"));

        hm.pop_scope();
        assert_eq!(hm.title(), Some("Root"));
    }

    #[test]
    fn favicon_svg() {
        let mut hm = HeadManager::new();
        hm.set_favicon_svg("/icon.svg");
        let html = hm.render_html();
        assert!(html.contains("image/svg+xml"));
    }
}
