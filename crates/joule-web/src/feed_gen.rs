//! RSS/Atom feed generation — RSS 2.0, Atom 1.0, entries with titles/links/
//! dates/content, categories, enclosures, feed validation.
//!
//! Pure-Rust replacement for RSS/Atom feed generation libraries.

use std::fmt;

// ── Feed Errors ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FeedError {
    MissingTitle,
    MissingLink,
    MissingId,
    InvalidDate(String),
}

impl fmt::Display for FeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FeedError::MissingTitle => write!(f, "feed title is required"),
            FeedError::MissingLink => write!(f, "feed link is required"),
            FeedError::MissingId => write!(f, "atom feed id is required"),
            FeedError::InvalidDate(d) => write!(f, "invalid date format: {d}"),
        }
    }
}

// ── Common Types ────────────────────────────────────────────────

/// A person (author/contributor).
#[derive(Debug, Clone)]
pub struct Person {
    pub name: String,
    pub email: Option<String>,
    pub uri: Option<String>,
}

impl Person {
    pub fn new(name: &str) -> Self {
        Self { name: name.into(), email: None, uri: None }
    }
    pub fn email(mut self, email: &str) -> Self { self.email = Some(email.into()); self }
    pub fn uri(mut self, uri: &str) -> Self { self.uri = Some(uri.into()); self }
}

/// An enclosure (media attachment).
#[derive(Debug, Clone)]
pub struct Enclosure {
    pub url: String,
    pub length: u64,
    pub mime_type: String,
}

impl Enclosure {
    pub fn new(url: &str, length: u64, mime_type: &str) -> Self {
        Self { url: url.into(), length, mime_type: mime_type.into() }
    }
}

/// Content type for feed entries.
#[derive(Debug, Clone)]
pub enum Content {
    Text(String),
    Html(String),
    Xhtml(String),
}

impl Content {
    pub fn text(s: &str) -> Self { Content::Text(s.into()) }
    pub fn html(s: &str) -> Self { Content::Html(s.into()) }
}

// ── Feed Entry ──────────────────────────────────────────────────

/// A single entry/item in a feed.
#[derive(Debug, Clone)]
pub struct FeedEntry {
    pub title: String,
    pub link: String,
    pub id: Option<String>,
    pub published: Option<String>,
    pub updated: Option<String>,
    pub summary: Option<String>,
    pub content: Option<Content>,
    pub author: Option<Person>,
    pub categories: Vec<String>,
    pub enclosure: Option<Enclosure>,
}

impl FeedEntry {
    pub fn new(title: &str, link: &str) -> Self {
        Self {
            title: title.into(), link: link.into(), id: None,
            published: None, updated: None, summary: None,
            content: None, author: None, categories: Vec::new(),
            enclosure: None,
        }
    }

    pub fn id(mut self, id: &str) -> Self { self.id = Some(id.into()); self }
    pub fn published(mut self, date: &str) -> Self { self.published = Some(date.into()); self }
    pub fn updated(mut self, date: &str) -> Self { self.updated = Some(date.into()); self }
    pub fn summary(mut self, s: &str) -> Self { self.summary = Some(s.into()); self }
    pub fn content(mut self, c: Content) -> Self { self.content = Some(c); self }
    pub fn author(mut self, a: Person) -> Self { self.author = Some(a); self }
    pub fn category(mut self, cat: &str) -> Self { self.categories.push(cat.into()); self }
    pub fn enclosure(mut self, e: Enclosure) -> Self { self.enclosure = Some(e); self }
}

// ── RSS 2.0 Builder ─────────────────────────────────────────────

/// Builder for RSS 2.0 feeds.
#[derive(Debug, Clone)]
pub struct RssBuilder {
    pub title: String,
    pub link: String,
    pub description: String,
    pub language: Option<String>,
    pub copyright: Option<String>,
    pub managing_editor: Option<String>,
    pub pub_date: Option<String>,
    pub last_build_date: Option<String>,
    pub generator: Option<String>,
    pub ttl: Option<u32>,
    pub image_url: Option<String>,
    pub image_title: Option<String>,
    pub image_link: Option<String>,
    pub items: Vec<FeedEntry>,
}

impl RssBuilder {
    pub fn new(title: &str, link: &str, description: &str) -> Self {
        Self {
            title: title.into(), link: link.into(), description: description.into(),
            language: None, copyright: None, managing_editor: None,
            pub_date: None, last_build_date: None, generator: None,
            ttl: None, image_url: None, image_title: None, image_link: None,
            items: Vec::new(),
        }
    }

    pub fn language(mut self, lang: &str) -> Self { self.language = Some(lang.into()); self }
    pub fn copyright(mut self, c: &str) -> Self { self.copyright = Some(c.into()); self }
    pub fn managing_editor(mut self, e: &str) -> Self { self.managing_editor = Some(e.into()); self }
    pub fn pub_date(mut self, d: &str) -> Self { self.pub_date = Some(d.into()); self }
    pub fn last_build_date(mut self, d: &str) -> Self { self.last_build_date = Some(d.into()); self }
    pub fn generator(mut self, g: &str) -> Self { self.generator = Some(g.into()); self }
    pub fn ttl(mut self, minutes: u32) -> Self { self.ttl = Some(minutes); self }

    pub fn image(mut self, url: &str, title: &str, link: &str) -> Self {
        self.image_url = Some(url.into());
        self.image_title = Some(title.into());
        self.image_link = Some(link.into());
        self
    }

    pub fn item(mut self, entry: FeedEntry) -> Self { self.items.push(entry); self }

    /// Build the RSS 2.0 XML string.
    pub fn build(&self) -> Result<String, FeedError> {
        if self.title.is_empty() { return Err(FeedError::MissingTitle); }
        if self.link.is_empty() { return Err(FeedError::MissingLink); }

        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<rss version=\"2.0\">\n");
        xml.push_str("  <channel>\n");
        xml.push_str(&format!("    <title>{}</title>\n", xml_escape(&self.title)));
        xml.push_str(&format!("    <link>{}</link>\n", xml_escape(&self.link)));
        xml.push_str(&format!("    <description>{}</description>\n", xml_escape(&self.description)));

        if let Some(ref lang) = self.language {
            xml.push_str(&format!("    <language>{lang}</language>\n"));
        }
        if let Some(ref cr) = self.copyright {
            xml.push_str(&format!("    <copyright>{}</copyright>\n", xml_escape(cr)));
        }
        if let Some(ref me) = self.managing_editor {
            xml.push_str(&format!("    <managingEditor>{}</managingEditor>\n", xml_escape(me)));
        }
        if let Some(ref pd) = self.pub_date {
            xml.push_str(&format!("    <pubDate>{pd}</pubDate>\n"));
        }
        if let Some(ref lbd) = self.last_build_date {
            xml.push_str(&format!("    <lastBuildDate>{lbd}</lastBuildDate>\n"));
        }
        if let Some(ref gn) = self.generator {
            xml.push_str(&format!("    <generator>{}</generator>\n", xml_escape(gn)));
        }
        if let Some(ttl) = self.ttl {
            xml.push_str(&format!("    <ttl>{ttl}</ttl>\n"));
        }
        if let (Some(url), Some(title), Some(link)) = (&self.image_url, &self.image_title, &self.image_link) {
            xml.push_str("    <image>\n");
            xml.push_str(&format!("      <url>{}</url>\n", xml_escape(url)));
            xml.push_str(&format!("      <title>{}</title>\n", xml_escape(title)));
            xml.push_str(&format!("      <link>{}</link>\n", xml_escape(link)));
            xml.push_str("    </image>\n");
        }

        for item in &self.items {
            xml.push_str("    <item>\n");
            xml.push_str(&format!("      <title>{}</title>\n", xml_escape(&item.title)));
            xml.push_str(&format!("      <link>{}</link>\n", xml_escape(&item.link)));
            if let Some(ref id) = item.id {
                xml.push_str(&format!("      <guid>{}</guid>\n", xml_escape(id)));
            }
            if let Some(ref pd) = item.published {
                xml.push_str(&format!("      <pubDate>{pd}</pubDate>\n"));
            }
            if let Some(ref summary) = item.summary {
                xml.push_str(&format!("      <description>{}</description>\n", xml_escape(summary)));
            }
            if let Some(ref content) = item.content {
                match content {
                    Content::Text(t) | Content::Html(t) | Content::Xhtml(t) => {
                        xml.push_str(&format!("      <content:encoded><![CDATA[{}]]></content:encoded>\n", t));
                    }
                }
            }
            if let Some(ref author) = item.author {
                if let Some(ref email) = author.email {
                    xml.push_str(&format!("      <author>{} ({})</author>\n", email, xml_escape(&author.name)));
                }
            }
            for cat in &item.categories {
                xml.push_str(&format!("      <category>{}</category>\n", xml_escape(cat)));
            }
            if let Some(ref enc) = item.enclosure {
                xml.push_str(&format!(
                    "      <enclosure url=\"{}\" length=\"{}\" type=\"{}\" />\n",
                    xml_escape(&enc.url), enc.length, xml_escape(&enc.mime_type)
                ));
            }
            xml.push_str("    </item>\n");
        }

        xml.push_str("  </channel>\n");
        xml.push_str("</rss>\n");
        Ok(xml)
    }
}

// ── Atom 1.0 Builder ────────────────────────────────────────────

/// Builder for Atom 1.0 feeds.
#[derive(Debug, Clone)]
pub struct AtomBuilder {
    pub id: String,
    pub title: String,
    pub updated: String,
    pub subtitle: Option<String>,
    pub link_self: Option<String>,
    pub link_alternate: Option<String>,
    pub author: Option<Person>,
    pub generator: Option<String>,
    pub icon: Option<String>,
    pub logo: Option<String>,
    pub rights: Option<String>,
    pub entries: Vec<FeedEntry>,
}

impl AtomBuilder {
    pub fn new(id: &str, title: &str, updated: &str) -> Self {
        Self {
            id: id.into(), title: title.into(), updated: updated.into(),
            subtitle: None, link_self: None, link_alternate: None,
            author: None, generator: None, icon: None, logo: None,
            rights: None, entries: Vec::new(),
        }
    }

    pub fn subtitle(mut self, s: &str) -> Self { self.subtitle = Some(s.into()); self }
    pub fn link_self(mut self, url: &str) -> Self { self.link_self = Some(url.into()); self }
    pub fn link_alternate(mut self, url: &str) -> Self { self.link_alternate = Some(url.into()); self }
    pub fn author(mut self, a: Person) -> Self { self.author = Some(a); self }
    pub fn generator(mut self, g: &str) -> Self { self.generator = Some(g.into()); self }
    pub fn icon(mut self, url: &str) -> Self { self.icon = Some(url.into()); self }
    pub fn logo(mut self, url: &str) -> Self { self.logo = Some(url.into()); self }
    pub fn rights(mut self, r: &str) -> Self { self.rights = Some(r.into()); self }
    pub fn entry(mut self, e: FeedEntry) -> Self { self.entries.push(e); self }

    /// Build the Atom 1.0 XML string.
    pub fn build(&self) -> Result<String, FeedError> {
        if self.title.is_empty() { return Err(FeedError::MissingTitle); }
        if self.id.is_empty() { return Err(FeedError::MissingId); }

        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<feed xmlns=\"http://www.w3.org/2005/Atom\">\n");
        xml.push_str(&format!("  <id>{}</id>\n", xml_escape(&self.id)));
        xml.push_str(&format!("  <title>{}</title>\n", xml_escape(&self.title)));
        xml.push_str(&format!("  <updated>{}</updated>\n", &self.updated));

        if let Some(ref st) = self.subtitle {
            xml.push_str(&format!("  <subtitle>{}</subtitle>\n", xml_escape(st)));
        }
        if let Some(ref ls) = self.link_self {
            xml.push_str(&format!("  <link rel=\"self\" href=\"{}\" />\n", xml_escape(ls)));
        }
        if let Some(ref la) = self.link_alternate {
            xml.push_str(&format!("  <link rel=\"alternate\" href=\"{}\" />\n", xml_escape(la)));
        }
        if let Some(ref a) = self.author {
            write_atom_person(&mut xml, "author", a, 2);
        }
        if let Some(ref g) = self.generator {
            xml.push_str(&format!("  <generator>{}</generator>\n", xml_escape(g)));
        }
        if let Some(ref icon) = self.icon {
            xml.push_str(&format!("  <icon>{}</icon>\n", xml_escape(icon)));
        }
        if let Some(ref logo) = self.logo {
            xml.push_str(&format!("  <logo>{}</logo>\n", xml_escape(logo)));
        }
        if let Some(ref rights) = self.rights {
            xml.push_str(&format!("  <rights>{}</rights>\n", xml_escape(rights)));
        }

        for entry in &self.entries {
            xml.push_str("  <entry>\n");
            let entry_id = entry.id.as_deref().unwrap_or(&entry.link);
            xml.push_str(&format!("    <id>{}</id>\n", xml_escape(entry_id)));
            xml.push_str(&format!("    <title>{}</title>\n", xml_escape(&entry.title)));
            xml.push_str(&format!("    <link href=\"{}\" />\n", xml_escape(&entry.link)));

            if let Some(ref updated) = entry.updated {
                xml.push_str(&format!("    <updated>{updated}</updated>\n"));
            }
            if let Some(ref published) = entry.published {
                xml.push_str(&format!("    <published>{published}</published>\n"));
            }
            if let Some(ref summary) = entry.summary {
                xml.push_str(&format!("    <summary>{}</summary>\n", xml_escape(summary)));
            }
            if let Some(ref content) = entry.content {
                match content {
                    Content::Text(t) => {
                        xml.push_str(&format!("    <content type=\"text\">{}</content>\n", xml_escape(t)));
                    }
                    Content::Html(h) => {
                        xml.push_str(&format!("    <content type=\"html\">{}</content>\n", xml_escape(h)));
                    }
                    Content::Xhtml(x) => {
                        xml.push_str(&format!("    <content type=\"xhtml\"><div xmlns=\"http://www.w3.org/1999/xhtml\">{x}</div></content>\n"));
                    }
                }
            }
            if let Some(ref author) = entry.author {
                write_atom_person(&mut xml, "author", author, 4);
            }
            for cat in &entry.categories {
                xml.push_str(&format!("    <category term=\"{}\" />\n", xml_escape(cat)));
            }
            xml.push_str("  </entry>\n");
        }

        xml.push_str("</feed>\n");
        Ok(xml)
    }
}

fn write_atom_person(xml: &mut String, tag: &str, person: &Person, indent: usize) {
    let pad = " ".repeat(indent);
    xml.push_str(&format!("{pad}<{tag}>\n"));
    xml.push_str(&format!("{pad}  <name>{}</name>\n", xml_escape(&person.name)));
    if let Some(ref email) = person.email {
        xml.push_str(&format!("{pad}  <email>{email}</email>\n"));
    }
    if let Some(ref uri) = person.uri {
        xml.push_str(&format!("{pad}  <uri>{}</uri>\n", xml_escape(uri)));
    }
    xml.push_str(&format!("{pad}</{tag}>\n"));
}

// ── Feed Validation ─────────────────────────────────────────────

/// Validation issues found in a feed.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationIssue {
    pub level: ValidationLevel,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel { Error, Warning }

/// Validate an RSS feed builder.
pub fn validate_rss(builder: &RssBuilder) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if builder.title.is_empty() {
        issues.push(ValidationIssue { level: ValidationLevel::Error, message: "title is required".into() });
    }
    if builder.link.is_empty() {
        issues.push(ValidationIssue { level: ValidationLevel::Error, message: "link is required".into() });
    }
    if builder.description.is_empty() {
        issues.push(ValidationIssue { level: ValidationLevel::Warning, message: "description is recommended".into() });
    }
    for (i, item) in builder.items.iter().enumerate() {
        if item.title.is_empty() {
            issues.push(ValidationIssue {
                level: ValidationLevel::Warning,
                message: format!("item {} has no title", i + 1),
            });
        }
        if item.link.is_empty() {
            issues.push(ValidationIssue {
                level: ValidationLevel::Warning,
                message: format!("item {} has no link", i + 1),
            });
        }
    }
    issues
}

/// Validate an Atom feed builder.
pub fn validate_atom(builder: &AtomBuilder) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    if builder.id.is_empty() {
        issues.push(ValidationIssue { level: ValidationLevel::Error, message: "id is required".into() });
    }
    if builder.title.is_empty() {
        issues.push(ValidationIssue { level: ValidationLevel::Error, message: "title is required".into() });
    }
    if builder.updated.is_empty() {
        issues.push(ValidationIssue { level: ValidationLevel::Error, message: "updated is required".into() });
    }
    if builder.author.is_none() {
        issues.push(ValidationIssue { level: ValidationLevel::Warning, message: "author is recommended".into() });
    }
    for (i, entry) in builder.entries.iter().enumerate() {
        if entry.title.is_empty() {
            issues.push(ValidationIssue {
                level: ValidationLevel::Warning,
                message: format!("entry {} has no title", i + 1),
            });
        }
        if entry.id.is_none() && entry.link.is_empty() {
            issues.push(ValidationIssue {
                level: ValidationLevel::Error,
                message: format!("entry {} has no id or link", i + 1),
            });
        }
    }
    issues
}

// ── Helpers ─────────────────────────────────────────────────────

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
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
    fn rss_basic() {
        let rss = RssBuilder::new("My Feed", "https://example.com", "A test feed")
            .item(FeedEntry::new("Post 1", "https://example.com/1"))
            .build()
            .unwrap();
        assert!(rss.contains("<rss version=\"2.0\">"));
        assert!(rss.contains("<title>My Feed</title>"));
        assert!(rss.contains("<link>https://example.com</link>"));
        assert!(rss.contains("<title>Post 1</title>"));
        assert!(rss.contains("</rss>"));
    }

    #[test]
    fn rss_with_metadata() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .language("en-us")
            .copyright("2026")
            .generator("joule-web")
            .ttl(60)
            .pub_date("Mon, 09 Mar 2026 00:00:00 GMT")
            .build()
            .unwrap();
        assert!(rss.contains("<language>en-us</language>"));
        assert!(rss.contains("<copyright>2026</copyright>"));
        assert!(rss.contains("<generator>joule-web</generator>"));
        assert!(rss.contains("<ttl>60</ttl>"));
    }

    #[test]
    fn rss_with_image() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .image("https://example.com/logo.png", "Logo", "https://example.com")
            .build()
            .unwrap();
        assert!(rss.contains("<image>"));
        assert!(rss.contains("<url>https://example.com/logo.png</url>"));
    }

    #[test]
    fn rss_item_with_categories() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .item(
                FeedEntry::new("Post", "https://example.com/1")
                    .category("tech")
                    .category("rust")
            )
            .build()
            .unwrap();
        assert!(rss.contains("<category>tech</category>"));
        assert!(rss.contains("<category>rust</category>"));
    }

    #[test]
    fn rss_item_with_enclosure() {
        let rss = RssBuilder::new("Podcast", "https://example.com", "A podcast")
            .item(
                FeedEntry::new("Episode 1", "https://example.com/ep1")
                    .enclosure(Enclosure::new("https://example.com/ep1.mp3", 12345678, "audio/mpeg"))
            )
            .build()
            .unwrap();
        assert!(rss.contains("enclosure"));
        assert!(rss.contains("audio/mpeg"));
        assert!(rss.contains("12345678"));
    }

    #[test]
    fn rss_item_with_author() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .item(
                FeedEntry::new("Post", "https://example.com/1")
                    .author(Person::new("Alice").email("alice@example.com"))
            )
            .build()
            .unwrap();
        assert!(rss.contains("<author>alice@example.com (Alice)</author>"));
    }

    #[test]
    fn rss_item_with_content() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .item(
                FeedEntry::new("Post", "https://example.com/1")
                    .content(Content::html("<p>Hello</p>"))
            )
            .build()
            .unwrap();
        assert!(rss.contains("content:encoded"));
        assert!(rss.contains("<![CDATA[<p>Hello</p>]]>"));
    }

    #[test]
    fn rss_missing_title_error() {
        let result = RssBuilder::new("", "https://example.com", "Desc").build();
        assert_eq!(result, Err(FeedError::MissingTitle));
    }

    #[test]
    fn rss_missing_link_error() {
        let result = RssBuilder::new("Title", "", "Desc").build();
        assert_eq!(result, Err(FeedError::MissingLink));
    }

    #[test]
    fn atom_basic() {
        let atom = AtomBuilder::new(
            "urn:uuid:feed-id",
            "My Atom Feed",
            "2026-03-09T00:00:00Z"
        )
        .entry(
            FeedEntry::new("Entry 1", "https://example.com/1")
                .id("urn:uuid:entry-1")
                .updated("2026-03-09T00:00:00Z")
        )
        .build()
        .unwrap();
        assert!(atom.contains("<feed xmlns=\"http://www.w3.org/2005/Atom\">"));
        assert!(atom.contains("<id>urn:uuid:feed-id</id>"));
        assert!(atom.contains("<title>My Atom Feed</title>"));
        assert!(atom.contains("</feed>"));
    }

    #[test]
    fn atom_with_author() {
        let atom = AtomBuilder::new("urn:uuid:1", "Feed", "2026-03-09T00:00:00Z")
            .author(Person::new("Alice").email("alice@example.com").uri("https://alice.example.com"))
            .build()
            .unwrap();
        assert!(atom.contains("<name>Alice</name>"));
        assert!(atom.contains("<email>alice@example.com</email>"));
        assert!(atom.contains("<uri>https://alice.example.com</uri>"));
    }

    #[test]
    fn atom_with_links() {
        let atom = AtomBuilder::new("urn:uuid:1", "Feed", "2026-03-09T00:00:00Z")
            .link_self("https://example.com/feed.xml")
            .link_alternate("https://example.com")
            .build()
            .unwrap();
        assert!(atom.contains("rel=\"self\""));
        assert!(atom.contains("rel=\"alternate\""));
    }

    #[test]
    fn atom_with_metadata() {
        let atom = AtomBuilder::new("urn:uuid:1", "Feed", "2026-03-09T00:00:00Z")
            .subtitle("A subtitle")
            .generator("joule-web")
            .icon("https://example.com/icon.png")
            .logo("https://example.com/logo.png")
            .rights("Copyright 2026")
            .build()
            .unwrap();
        assert!(atom.contains("<subtitle>A subtitle</subtitle>"));
        assert!(atom.contains("<generator>joule-web</generator>"));
        assert!(atom.contains("<icon>"));
        assert!(atom.contains("<logo>"));
        assert!(atom.contains("<rights>"));
    }

    #[test]
    fn atom_entry_with_content_text() {
        let atom = AtomBuilder::new("urn:uuid:1", "Feed", "2026-03-09T00:00:00Z")
            .entry(
                FeedEntry::new("Entry", "https://example.com/1")
                    .id("urn:uuid:e1")
                    .content(Content::text("Plain text content"))
            )
            .build()
            .unwrap();
        assert!(atom.contains("type=\"text\""));
        assert!(atom.contains("Plain text content"));
    }

    #[test]
    fn atom_entry_with_content_html() {
        let atom = AtomBuilder::new("urn:uuid:1", "Feed", "2026-03-09T00:00:00Z")
            .entry(
                FeedEntry::new("Entry", "https://example.com/1")
                    .id("urn:uuid:e1")
                    .content(Content::html("<p>HTML</p>"))
            )
            .build()
            .unwrap();
        assert!(atom.contains("type=\"html\""));
    }

    #[test]
    fn atom_entry_categories() {
        let atom = AtomBuilder::new("urn:uuid:1", "Feed", "2026-03-09T00:00:00Z")
            .entry(
                FeedEntry::new("Entry", "https://example.com/1")
                    .id("urn:uuid:e1")
                    .category("rust")
            )
            .build()
            .unwrap();
        assert!(atom.contains("category term=\"rust\""));
    }

    #[test]
    fn atom_missing_title_error() {
        let result = AtomBuilder::new("urn:id", "", "2026-03-09").build();
        assert_eq!(result, Err(FeedError::MissingTitle));
    }

    #[test]
    fn atom_missing_id_error() {
        let result = AtomBuilder::new("", "Title", "2026-03-09").build();
        assert_eq!(result, Err(FeedError::MissingId));
    }

    #[test]
    fn validate_rss_valid() {
        let builder = RssBuilder::new("Title", "https://example.com", "Description")
            .item(FeedEntry::new("Item", "https://example.com/1"));
        let issues = validate_rss(&builder);
        assert!(issues.iter().all(|i| i.level != ValidationLevel::Error));
    }

    #[test]
    fn validate_rss_missing_fields() {
        let builder = RssBuilder::new("", "", "");
        let issues = validate_rss(&builder);
        let errors: Vec<_> = issues.iter().filter(|i| i.level == ValidationLevel::Error).collect();
        assert!(errors.len() >= 2);
    }

    #[test]
    fn validate_atom_valid() {
        let builder = AtomBuilder::new("urn:id", "Title", "2026-03-09")
            .author(Person::new("Author"))
            .entry(FeedEntry::new("Entry", "https://example.com").id("urn:e1"));
        let issues = validate_atom(&builder);
        assert!(issues.iter().all(|i| i.level != ValidationLevel::Error));
    }

    #[test]
    fn validate_atom_missing_author_warning() {
        let builder = AtomBuilder::new("urn:id", "Title", "2026-03-09");
        let issues = validate_atom(&builder);
        assert!(issues.iter().any(|i| i.message.contains("author")));
    }

    #[test]
    fn xml_escaping() {
        let rss = RssBuilder::new("A & B", "https://example.com", "< > \"")
            .build()
            .unwrap();
        assert!(rss.contains("A &amp; B"));
        assert!(rss.contains("&lt; &gt; &quot;"));
    }

    #[test]
    fn person_builder() {
        let p = Person::new("Alice").email("a@example.com").uri("https://alice.com");
        assert_eq!(p.name, "Alice");
        assert_eq!(p.email.as_deref(), Some("a@example.com"));
        assert_eq!(p.uri.as_deref(), Some("https://alice.com"));
    }

    #[test]
    fn enclosure_builder() {
        let e = Enclosure::new("https://example.com/file.mp3", 9999, "audio/mpeg");
        assert_eq!(e.url, "https://example.com/file.mp3");
        assert_eq!(e.length, 9999);
    }

    #[test]
    fn feed_entry_builder() {
        let entry = FeedEntry::new("Title", "https://example.com")
            .id("id123")
            .published("2026-03-09")
            .updated("2026-03-09")
            .summary("Summary text")
            .category("cat1")
            .category("cat2");
        assert_eq!(entry.id.as_deref(), Some("id123"));
        assert_eq!(entry.categories.len(), 2);
    }

    #[test]
    fn error_display() {
        assert_eq!(format!("{}", FeedError::MissingTitle), "feed title is required");
        assert_eq!(format!("{}", FeedError::MissingLink), "feed link is required");
        assert_eq!(format!("{}", FeedError::MissingId), "atom feed id is required");
    }

    #[test]
    fn rss_managing_editor() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .managing_editor("editor@example.com")
            .build()
            .unwrap();
        assert!(rss.contains("<managingEditor>editor@example.com</managingEditor>"));
    }

    #[test]
    fn rss_last_build_date() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .last_build_date("Mon, 09 Mar 2026 12:00:00 GMT")
            .build()
            .unwrap();
        assert!(rss.contains("<lastBuildDate>"));
    }

    #[test]
    fn rss_item_guid() {
        let rss = RssBuilder::new("Feed", "https://example.com", "Desc")
            .item(FeedEntry::new("Post", "https://example.com/1").id("unique-id-123"))
            .build()
            .unwrap();
        assert!(rss.contains("<guid>unique-id-123</guid>"));
    }
}
