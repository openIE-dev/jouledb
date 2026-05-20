//! SEO utilities — meta tags, Open Graph, Twitter Card, JSON-LD structured data,
//! canonical URLs, hreflang, title/description validation.
//!
//! Pure Rust SEO toolkit. Generates HTML meta tags and structured data.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Meta Tag ─────────────────────────────────────────────────────

/// A single HTML meta tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetaTag {
    /// `<meta name="..." content="...">`
    Name { name: String, content: String },
    /// `<meta property="..." content="...">`
    Property { property: String, content: String },
    /// `<meta http-equiv="..." content="...">`
    HttpEquiv { http_equiv: String, content: String },
    /// `<meta charset="...">`
    Charset(String),
}

impl MetaTag {
    pub fn name(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Name { name: name.into(), content: content.into() }
    }

    pub fn property(property: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Property { property: property.into(), content: content.into() }
    }

    pub fn http_equiv(equiv: impl Into<String>, content: impl Into<String>) -> Self {
        Self::HttpEquiv { http_equiv: equiv.into(), content: content.into() }
    }

    /// Render as HTML string.
    pub fn to_html(&self) -> String {
        match self {
            Self::Name { name, content } => format!(r#"<meta name="{name}" content="{content}">"#),
            Self::Property { property, content } => format!(r#"<meta property="{property}" content="{content}">"#),
            Self::HttpEquiv { http_equiv, content } => format!(r#"<meta http-equiv="{http_equiv}" content="{content}">"#),
            Self::Charset(cs) => format!(r#"<meta charset="{cs}">"#),
        }
    }
}

impl fmt::Display for MetaTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_html())
    }
}

// ── Open Graph ───────────────────────────────────────────────────

/// Open Graph metadata.
#[derive(Debug, Clone, Default)]
pub struct OpenGraph {
    pub og_type: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub url: Option<String>,
    pub image: Option<String>,
    pub image_width: Option<u32>,
    pub image_height: Option<u32>,
    pub image_alt: Option<String>,
    pub site_name: Option<String>,
    pub locale: Option<String>,
    pub article_author: Option<String>,
    pub article_published: Option<String>,
    pub article_section: Option<String>,
}

impl OpenGraph {
    pub fn new() -> Self { Self::default() }

    pub fn website() -> Self { Self { og_type: Some("website".into()), ..Default::default() } }
    pub fn article() -> Self { Self { og_type: Some("article".into()), ..Default::default() } }

    pub fn title(mut self, t: impl Into<String>) -> Self { self.title = Some(t.into()); self }
    pub fn description(mut self, d: impl Into<String>) -> Self { self.description = Some(d.into()); self }
    pub fn url(mut self, u: impl Into<String>) -> Self { self.url = Some(u.into()); self }
    pub fn image(mut self, i: impl Into<String>) -> Self { self.image = Some(i.into()); self }
    pub fn image_dimensions(mut self, w: u32, h: u32) -> Self {
        self.image_width = Some(w);
        self.image_height = Some(h);
        self
    }
    pub fn image_alt(mut self, a: impl Into<String>) -> Self { self.image_alt = Some(a.into()); self }
    pub fn site_name(mut self, s: impl Into<String>) -> Self { self.site_name = Some(s.into()); self }
    pub fn locale(mut self, l: impl Into<String>) -> Self { self.locale = Some(l.into()); self }

    /// Generate Open Graph meta tags.
    pub fn to_meta_tags(&self) -> Vec<MetaTag> {
        let mut tags = Vec::new();
        if let Some(ref t) = self.og_type { tags.push(MetaTag::property("og:type", t)); }
        if let Some(ref t) = self.title { tags.push(MetaTag::property("og:title", t)); }
        if let Some(ref d) = self.description { tags.push(MetaTag::property("og:description", d)); }
        if let Some(ref u) = self.url { tags.push(MetaTag::property("og:url", u)); }
        if let Some(ref i) = self.image { tags.push(MetaTag::property("og:image", i)); }
        if let Some(w) = self.image_width { tags.push(MetaTag::property("og:image:width", w.to_string())); }
        if let Some(h) = self.image_height { tags.push(MetaTag::property("og:image:height", h.to_string())); }
        if let Some(ref a) = self.image_alt { tags.push(MetaTag::property("og:image:alt", a)); }
        if let Some(ref s) = self.site_name { tags.push(MetaTag::property("og:site_name", s)); }
        if let Some(ref l) = self.locale { tags.push(MetaTag::property("og:locale", l)); }
        if let Some(ref a) = self.article_author { tags.push(MetaTag::property("article:author", a)); }
        if let Some(ref p) = self.article_published { tags.push(MetaTag::property("article:published_time", p)); }
        if let Some(ref s) = self.article_section { tags.push(MetaTag::property("article:section", s)); }
        tags
    }
}

// ── Twitter Card ─────────────────────────────────────────────────

/// Twitter Card type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwitterCardType {
    Summary,
    SummaryLargeImage,
    App,
    Player,
}

impl fmt::Display for TwitterCardType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Summary => write!(f, "summary"),
            Self::SummaryLargeImage => write!(f, "summary_large_image"),
            Self::App => write!(f, "app"),
            Self::Player => write!(f, "player"),
        }
    }
}

/// Twitter Card metadata.
#[derive(Debug, Clone)]
pub struct TwitterCard {
    pub card_type: TwitterCardType,
    pub title: Option<String>,
    pub description: Option<String>,
    pub image: Option<String>,
    pub image_alt: Option<String>,
    pub site: Option<String>,
    pub creator: Option<String>,
}

impl TwitterCard {
    pub fn summary() -> Self {
        Self { card_type: TwitterCardType::Summary, title: None, description: None, image: None, image_alt: None, site: None, creator: None }
    }

    pub fn large_image() -> Self {
        Self { card_type: TwitterCardType::SummaryLargeImage, title: None, description: None, image: None, image_alt: None, site: None, creator: None }
    }

    pub fn title(mut self, t: impl Into<String>) -> Self { self.title = Some(t.into()); self }
    pub fn description(mut self, d: impl Into<String>) -> Self { self.description = Some(d.into()); self }
    pub fn image(mut self, i: impl Into<String>) -> Self { self.image = Some(i.into()); self }
    pub fn image_alt(mut self, a: impl Into<String>) -> Self { self.image_alt = Some(a.into()); self }
    pub fn site(mut self, s: impl Into<String>) -> Self { self.site = Some(s.into()); self }
    pub fn creator(mut self, c: impl Into<String>) -> Self { self.creator = Some(c.into()); self }

    /// Generate Twitter Card meta tags.
    pub fn to_meta_tags(&self) -> Vec<MetaTag> {
        let mut tags = vec![MetaTag::name("twitter:card", self.card_type.to_string())];
        if let Some(ref t) = self.title { tags.push(MetaTag::name("twitter:title", t)); }
        if let Some(ref d) = self.description { tags.push(MetaTag::name("twitter:description", d)); }
        if let Some(ref i) = self.image { tags.push(MetaTag::name("twitter:image", i)); }
        if let Some(ref a) = self.image_alt { tags.push(MetaTag::name("twitter:image:alt", a)); }
        if let Some(ref s) = self.site { tags.push(MetaTag::name("twitter:site", s)); }
        if let Some(ref c) = self.creator { tags.push(MetaTag::name("twitter:creator", c)); }
        tags
    }
}

// ── JSON-LD Structured Data ──────────────────────────────────────

/// JSON-LD structured data builder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonLd {
    #[serde(rename = "@context")]
    pub context: String,
    #[serde(rename = "@type")]
    pub ld_type: String,
    #[serde(flatten)]
    pub properties: serde_json::Map<String, serde_json::Value>,
}

impl JsonLd {
    pub fn new(ld_type: impl Into<String>) -> Self {
        Self {
            context: "https://schema.org".into(),
            ld_type: ld_type.into(),
            properties: serde_json::Map::new(),
        }
    }

    pub fn set_str(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), serde_json::Value::String(value.into()));
        self
    }

    pub fn set_number(mut self, key: impl Into<String>, value: f64) -> Self {
        self.properties.insert(key.into(), serde_json::json!(value));
        self
    }

    pub fn set_bool(mut self, key: impl Into<String>, value: bool) -> Self {
        self.properties.insert(key.into(), serde_json::Value::Bool(value));
        self
    }

    pub fn set_object(mut self, key: impl Into<String>, obj: JsonLd) -> Self {
        let val = serde_json::to_value(obj).unwrap_or(serde_json::Value::Null);
        self.properties.insert(key.into(), val);
        self
    }

    pub fn set_array(mut self, key: impl Into<String>, arr: Vec<serde_json::Value>) -> Self {
        self.properties.insert(key.into(), serde_json::Value::Array(arr));
        self
    }

    /// Render as `<script type="application/ld+json">...</script>`.
    pub fn to_script_tag(&self) -> String {
        let json = serde_json::to_string_pretty(self).unwrap_or_default();
        format!("<script type=\"application/ld+json\">\n{json}\n</script>")
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Create a common Organization JSON-LD.
pub fn organization_ld(name: &str, url: &str, logo: &str) -> JsonLd {
    JsonLd::new("Organization")
        .set_str("name", name)
        .set_str("url", url)
        .set_str("logo", logo)
}

/// Create a WebPage JSON-LD.
pub fn webpage_ld(name: &str, url: &str, description: &str) -> JsonLd {
    JsonLd::new("WebPage")
        .set_str("name", name)
        .set_str("url", url)
        .set_str("description", description)
}

/// Create a BreadcrumbList JSON-LD.
pub fn breadcrumb_ld(items: &[(&str, &str)]) -> JsonLd {
    let list_items: Vec<serde_json::Value> = items.iter().enumerate().map(|(i, (name, url))| {
        serde_json::json!({
            "@type": "ListItem",
            "position": i + 1,
            "name": name,
            "item": url,
        })
    }).collect();

    JsonLd::new("BreadcrumbList")
        .set_array("itemListElement", list_items)
}

// ── Canonical & Hreflang ─────────────────────────────────────────

/// Generate canonical URL link tag.
pub fn canonical_tag(url: &str) -> String {
    format!(r#"<link rel="canonical" href="{url}">"#)
}

/// Hreflang entry for international SEO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hreflang {
    pub lang: String,
    pub url: String,
}

impl Hreflang {
    pub fn new(lang: impl Into<String>, url: impl Into<String>) -> Self {
        Self { lang: lang.into(), url: url.into() }
    }

    /// Generate `<link rel="alternate" hreflang="..." href="...">`.
    pub fn to_html(&self) -> String {
        format!(r#"<link rel="alternate" hreflang="{}" href="{}">"#, self.lang, self.url)
    }
}

/// Generate hreflang tags for a set of language/url pairs.
pub fn hreflang_tags(entries: &[Hreflang]) -> Vec<String> {
    entries.iter().map(|e| e.to_html()).collect()
}

// ── Title & Description Validation ───────────────────────────────

/// SEO validation result.
#[derive(Debug, Clone)]
pub struct SeoValidation {
    pub issues: Vec<SeoIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueSeverity { Error, Warning, Info }

impl fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SeoIssue {
    pub severity: IssueSeverity,
    pub field: String,
    pub message: String,
}

impl fmt::Display for SeoIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.field, self.message)
    }
}

/// Validate title tag for SEO best practices.
pub fn validate_title(title: &str) -> Vec<SeoIssue> {
    let mut issues = Vec::new();
    if title.is_empty() {
        issues.push(SeoIssue { severity: IssueSeverity::Error, field: "title".into(), message: "title is empty".into() });
    } else if title.len() < 10 {
        issues.push(SeoIssue { severity: IssueSeverity::Warning, field: "title".into(), message: format!("title is very short ({} chars, recommended 30-60)", title.len()) });
    } else if title.len() > 60 {
        issues.push(SeoIssue { severity: IssueSeverity::Warning, field: "title".into(), message: format!("title may be truncated in SERPs ({} chars, recommended max 60)", title.len()) });
    }
    issues
}

/// Validate meta description for SEO best practices.
pub fn validate_description(desc: &str) -> Vec<SeoIssue> {
    let mut issues = Vec::new();
    if desc.is_empty() {
        issues.push(SeoIssue { severity: IssueSeverity::Error, field: "description".into(), message: "description is empty".into() });
    } else if desc.len() < 50 {
        issues.push(SeoIssue { severity: IssueSeverity::Warning, field: "description".into(), message: format!("description is short ({} chars, recommended 120-160)", desc.len()) });
    } else if desc.len() > 160 {
        issues.push(SeoIssue { severity: IssueSeverity::Warning, field: "description".into(), message: format!("description may be truncated ({} chars, recommended max 160)", desc.len()) });
    }
    issues
}

// ── Robots Meta ──────────────────────────────────────────────────

/// Robots directives.
#[derive(Debug, Clone)]
pub struct RobotsDirectives {
    pub index: bool,
    pub follow: bool,
    pub no_archive: bool,
    pub no_snippet: bool,
    pub max_snippet: Option<i32>,
    pub max_image_preview: Option<String>,
}

impl RobotsDirectives {
    pub fn default_allow() -> Self {
        Self { index: true, follow: true, no_archive: false, no_snippet: false, max_snippet: None, max_image_preview: None }
    }

    pub fn noindex() -> Self {
        Self { index: false, ..Self::default_allow() }
    }

    pub fn noindex_nofollow() -> Self {
        Self { index: false, follow: false, ..Self::default_allow() }
    }

    /// Generate meta robots tag.
    pub fn to_meta_tag(&self) -> MetaTag {
        let mut parts = Vec::new();
        if self.index { parts.push("index"); } else { parts.push("noindex"); }
        if self.follow { parts.push("follow"); } else { parts.push("nofollow"); }
        if self.no_archive { parts.push("noarchive"); }
        if self.no_snippet { parts.push("nosnippet"); }
        MetaTag::name("robots", parts.join(", "))
    }
}

// ── Page SEO Builder ─────────────────────────────────────────────

/// Complete SEO configuration for a page.
#[derive(Debug, Clone)]
pub struct PageSeo {
    pub title: String,
    pub description: Option<String>,
    pub canonical: Option<String>,
    pub robots: Option<RobotsDirectives>,
    pub og: Option<OpenGraph>,
    pub twitter: Option<TwitterCard>,
    pub json_ld: Vec<JsonLd>,
    pub hreflang: Vec<Hreflang>,
    pub extra_meta: Vec<MetaTag>,
}

impl PageSeo {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            description: None,
            canonical: None,
            robots: None,
            og: None,
            twitter: None,
            json_ld: Vec::new(),
            hreflang: Vec::new(),
            extra_meta: Vec::new(),
        }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self { self.description = Some(d.into()); self }
    pub fn canonical(mut self, c: impl Into<String>) -> Self { self.canonical = Some(c.into()); self }
    pub fn robots(mut self, r: RobotsDirectives) -> Self { self.robots = Some(r); self }
    pub fn open_graph(mut self, og: OpenGraph) -> Self { self.og = Some(og); self }
    pub fn twitter_card(mut self, tc: TwitterCard) -> Self { self.twitter = Some(tc); self }
    pub fn add_json_ld(mut self, ld: JsonLd) -> Self { self.json_ld.push(ld); self }
    pub fn add_hreflang(mut self, h: Hreflang) -> Self { self.hreflang.push(h); self }
    pub fn add_meta(mut self, m: MetaTag) -> Self { self.extra_meta.push(m); self }

    /// Generate all HTML tags for this page's SEO.
    pub fn to_html(&self) -> String {
        let mut parts = Vec::new();

        parts.push(format!("<title>{}</title>", self.title));

        if let Some(ref desc) = self.description {
            parts.push(MetaTag::name("description", desc).to_html());
        }

        if let Some(ref url) = self.canonical {
            parts.push(canonical_tag(url));
        }

        if let Some(ref robots) = self.robots {
            parts.push(robots.to_meta_tag().to_html());
        }

        if let Some(ref og) = self.og {
            for tag in og.to_meta_tags() {
                parts.push(tag.to_html());
            }
        }

        if let Some(ref tc) = self.twitter {
            for tag in tc.to_meta_tags() {
                parts.push(tag.to_html());
            }
        }

        for ld in &self.json_ld {
            parts.push(ld.to_script_tag());
        }

        for h in &self.hreflang {
            parts.push(h.to_html());
        }

        for m in &self.extra_meta {
            parts.push(m.to_html());
        }

        parts.join("\n")
    }

    /// Validate this page's SEO configuration.
    pub fn validate(&self) -> Vec<SeoIssue> {
        let mut issues = validate_title(&self.title);
        if let Some(ref desc) = self.description {
            issues.extend(validate_description(desc));
        } else {
            issues.push(SeoIssue {
                severity: IssueSeverity::Warning,
                field: "description".into(),
                message: "meta description not set".into(),
            });
        }
        if self.canonical.is_none() {
            issues.push(SeoIssue {
                severity: IssueSeverity::Info,
                field: "canonical".into(),
                message: "canonical URL not set".into(),
            });
        }
        if self.og.is_none() {
            issues.push(SeoIssue {
                severity: IssueSeverity::Info,
                field: "og".into(),
                message: "Open Graph tags not set".into(),
            });
        }
        issues
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_tag_name() {
        let tag = MetaTag::name("description", "A test page");
        assert_eq!(tag.to_html(), r#"<meta name="description" content="A test page">"#);
    }

    #[test]
    fn test_meta_tag_property() {
        let tag = MetaTag::property("og:title", "Test");
        assert_eq!(tag.to_html(), r#"<meta property="og:title" content="Test">"#);
    }

    #[test]
    fn test_meta_tag_http_equiv() {
        let tag = MetaTag::http_equiv("refresh", "5");
        assert_eq!(tag.to_html(), r#"<meta http-equiv="refresh" content="5">"#);
    }

    #[test]
    fn test_meta_tag_charset() {
        let tag = MetaTag::Charset("utf-8".into());
        assert_eq!(tag.to_html(), r#"<meta charset="utf-8">"#);
    }

    #[test]
    fn test_open_graph_tags() {
        let og = OpenGraph::website()
            .title("My Site")
            .description("A great site")
            .url("https://example.com")
            .image("https://example.com/img.jpg")
            .image_dimensions(1200, 630);

        let tags = og.to_meta_tags();
        assert!(tags.iter().any(|t| matches!(t, MetaTag::Property { property, .. } if property == "og:type")));
        assert!(tags.iter().any(|t| matches!(t, MetaTag::Property { property, .. } if property == "og:title")));
        assert!(tags.iter().any(|t| matches!(t, MetaTag::Property { property, .. } if property == "og:image:width")));
    }

    #[test]
    fn test_open_graph_article() {
        let og = OpenGraph::article().title("Blog Post");
        let tags = og.to_meta_tags();
        let type_tag = tags.iter().find(|t| matches!(t, MetaTag::Property { property, .. } if property == "og:type")).unwrap();
        if let MetaTag::Property { content, .. } = type_tag {
            assert_eq!(content, "article");
        }
    }

    #[test]
    fn test_twitter_card_summary() {
        let tc = TwitterCard::summary()
            .title("Test")
            .site("@example")
            .creator("@author");

        let tags = tc.to_meta_tags();
        assert!(tags.iter().any(|t| matches!(t, MetaTag::Name { name, content } if name == "twitter:card" && content == "summary")));
        assert!(tags.iter().any(|t| matches!(t, MetaTag::Name { name, .. } if name == "twitter:site")));
    }

    #[test]
    fn test_twitter_card_large_image() {
        let tc = TwitterCard::large_image();
        let tags = tc.to_meta_tags();
        assert!(tags.iter().any(|t| matches!(t, MetaTag::Name { content, .. } if content == "summary_large_image")));
    }

    #[test]
    fn test_json_ld_basic() {
        let ld = JsonLd::new("Article")
            .set_str("headline", "Test Article")
            .set_str("author", "John Doe");

        let json = ld.to_json();
        assert!(json.contains("schema.org"));
        assert!(json.contains("Article"));
        assert!(json.contains("Test Article"));
    }

    #[test]
    fn test_json_ld_script_tag() {
        let ld = JsonLd::new("WebPage").set_str("name", "Home");
        let script = ld.to_script_tag();
        assert!(script.starts_with("<script type=\"application/ld+json\">"));
        assert!(script.ends_with("</script>"));
    }

    #[test]
    fn test_json_ld_nested() {
        let author = JsonLd::new("Person").set_str("name", "Jane");
        let article = JsonLd::new("Article")
            .set_str("headline", "Test")
            .set_object("author", author);

        let json = article.to_json();
        assert!(json.contains("Person"));
        assert!(json.contains("Jane"));
    }

    #[test]
    fn test_organization_ld() {
        let org = organization_ld("ACME", "https://acme.com", "https://acme.com/logo.png");
        let json = org.to_json();
        assert!(json.contains("Organization"));
        assert!(json.contains("ACME"));
    }

    #[test]
    fn test_breadcrumb_ld() {
        let bc = breadcrumb_ld(&[("Home", "/"), ("Blog", "/blog"), ("Post", "/blog/post")]);
        let json = bc.to_json();
        assert!(json.contains("BreadcrumbList"));
        assert!(json.contains("ListItem"));
    }

    #[test]
    fn test_canonical_tag() {
        assert_eq!(
            canonical_tag("https://example.com/page"),
            r#"<link rel="canonical" href="https://example.com/page">"#
        );
    }

    #[test]
    fn test_hreflang() {
        let h = Hreflang::new("en", "https://example.com/en/");
        assert!(h.to_html().contains(r#"hreflang="en""#));
        assert!(h.to_html().contains(r#"href="https://example.com/en/""#));
    }

    #[test]
    fn test_hreflang_tags() {
        let entries = vec![
            Hreflang::new("en", "https://example.com/en/"),
            Hreflang::new("fr", "https://example.com/fr/"),
            Hreflang::new("x-default", "https://example.com/"),
        ];
        let tags = hreflang_tags(&entries);
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn test_validate_title_empty() {
        let issues = validate_title("");
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Error));
    }

    #[test]
    fn test_validate_title_short() {
        let issues = validate_title("Hi");
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Warning));
    }

    #[test]
    fn test_validate_title_long() {
        let issues = validate_title(&"a".repeat(70));
        assert!(issues.iter().any(|i| i.message.contains("truncated")));
    }

    #[test]
    fn test_validate_title_good() {
        let issues = validate_title("A Good Page Title for Testing SEO");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validate_description_empty() {
        let issues = validate_description("");
        assert!(issues.iter().any(|i| i.severity == IssueSeverity::Error));
    }

    #[test]
    fn test_validate_description_good() {
        let desc = "This is a well-crafted meta description that provides useful information about the page content for search engines and users alike, of appropriate length.";
        let issues = validate_description(desc);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_robots_default() {
        let r = RobotsDirectives::default_allow();
        let tag = r.to_meta_tag();
        if let MetaTag::Name { content, .. } = tag {
            assert!(content.contains("index"));
            assert!(content.contains("follow"));
            assert!(!content.contains("noindex"));
        }
    }

    #[test]
    fn test_robots_noindex_nofollow() {
        let r = RobotsDirectives::noindex_nofollow();
        let tag = r.to_meta_tag();
        if let MetaTag::Name { content, .. } = tag {
            assert!(content.contains("noindex"));
            assert!(content.contains("nofollow"));
        }
    }

    #[test]
    fn test_page_seo_full() {
        let seo = PageSeo::new("My Page Title - Example Site")
            .description("A comprehensive description of the page content for search engines")
            .canonical("https://example.com/page")
            .robots(RobotsDirectives::default_allow())
            .open_graph(OpenGraph::website().title("My Page").url("https://example.com/page"))
            .twitter_card(TwitterCard::summary().title("My Page"))
            .add_json_ld(webpage_ld("My Page", "https://example.com/page", "Description"))
            .add_hreflang(Hreflang::new("en", "https://example.com/en/page"));

        let html = seo.to_html();
        assert!(html.contains("<title>My Page Title - Example Site</title>"));
        assert!(html.contains("canonical"));
        assert!(html.contains("og:title"));
        assert!(html.contains("twitter:card"));
        assert!(html.contains("application/ld+json"));
        assert!(html.contains("hreflang"));
    }

    #[test]
    fn test_page_seo_validate() {
        let seo = PageSeo::new("OK Title for This Page");
        let issues = seo.validate();
        assert!(issues.iter().any(|i| i.field == "description"));
    }

    #[test]
    fn test_page_seo_validate_good() {
        let seo = PageSeo::new("Good Title for SEO Testing")
            .description("A well-written meta description providing useful summary of the page content for both search engines and users browsing results pages")
            .canonical("https://example.com/page")
            .open_graph(OpenGraph::website());

        let issues = seo.validate();
        let errors: Vec<_> = issues.iter().filter(|i| i.severity == IssueSeverity::Error).collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_twitter_card_type_display() {
        assert_eq!(TwitterCardType::Summary.to_string(), "summary");
        assert_eq!(TwitterCardType::SummaryLargeImage.to_string(), "summary_large_image");
        assert_eq!(TwitterCardType::App.to_string(), "app");
        assert_eq!(TwitterCardType::Player.to_string(), "player");
    }

    #[test]
    fn test_json_ld_number_and_bool() {
        let ld = JsonLd::new("Product")
            .set_number("price", 29.99)
            .set_bool("inStock", true);

        let json = ld.to_json();
        assert!(json.contains("29.99"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_meta_tag_display_trait() {
        let tag = MetaTag::name("viewport", "width=device-width");
        assert_eq!(format!("{tag}"), tag.to_html());
    }

    #[test]
    fn test_issue_severity_display() {
        assert_eq!(IssueSeverity::Error.to_string(), "ERROR");
        assert_eq!(IssueSeverity::Warning.to_string(), "WARNING");
        assert_eq!(IssueSeverity::Info.to_string(), "INFO");
    }

    #[test]
    fn test_seo_issue_display() {
        let issue = SeoIssue {
            severity: IssueSeverity::Warning,
            field: "title".into(),
            message: "too short".into(),
        };
        let s = issue.to_string();
        assert!(s.contains("[WARNING]"));
        assert!(s.contains("title"));
    }
}
