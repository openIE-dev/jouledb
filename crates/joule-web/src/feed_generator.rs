//! RSS/Atom feed generator — feed metadata, entry items, RFC 822 date
//! formatting, Atom XML output, RSS 2.0 XML output, pagination, and
//! category/tag support.
//!
//! Pure-Rust replacement for feed, rss, atom, and similar Node.js/JS
//! feed generation libraries.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from feed generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedError {
    MissingRequired(String),
    InvalidDate(String),
}

impl fmt::Display for FeedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequired(field) => write!(f, "missing required field: {field}"),
            Self::InvalidDate(msg) => write!(f, "invalid date: {msg}"),
        }
    }
}

impl std::error::Error for FeedError {}

// ── Feed Format ─────────────────────────────────────────────────

/// Output format for a feed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedFormat {
    Rss20,
    Atom,
}

// ── Author ──────────────────────────────────────────────────────

/// A feed/entry author.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Author {
    pub name: String,
    pub email: Option<String>,
    pub uri: Option<String>,
}

impl Author {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            email: None,
            uri: None,
        }
    }

    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }

    pub fn with_uri(mut self, uri: impl Into<String>) -> Self {
        self.uri = Some(uri.into());
        self
    }
}

// ── Category ────────────────────────────────────────────────────

/// A feed category/tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Category {
    pub term: String,
    pub label: Option<String>,
    pub scheme: Option<String>,
}

impl Category {
    pub fn new(term: impl Into<String>) -> Self {
        Self {
            term: term.into(),
            label: None,
            scheme: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ── Feed Entry ──────────────────────────────────────────────────

/// A single entry/item in a feed.
#[derive(Debug, Clone)]
pub struct FeedEntry {
    pub id: String,
    pub title: String,
    pub link: String,
    pub summary: Option<String>,
    pub content: Option<String>,
    pub published: Option<String>,
    pub updated: Option<String>,
    pub authors: Vec<Author>,
    pub categories: Vec<Category>,
}

impl FeedEntry {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        link: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            link: link.into(),
            summary: None,
            content: None,
            published: None,
            updated: None,
            authors: Vec::new(),
            categories: Vec::new(),
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    pub fn with_published(mut self, date: impl Into<String>) -> Self {
        self.published = Some(date.into());
        self
    }

    pub fn with_updated(mut self, date: impl Into<String>) -> Self {
        self.updated = Some(date.into());
        self
    }

    pub fn add_author(mut self, author: Author) -> Self {
        self.authors.push(author);
        self
    }

    pub fn add_category(mut self, category: Category) -> Self {
        self.categories.push(category);
        self
    }
}

// ── Pagination ──────────────────────────────────────────────────

/// Pagination links for a feed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeedPagination {
    pub first: Option<String>,
    pub last: Option<String>,
    pub next: Option<String>,
    pub previous: Option<String>,
}

impl FeedPagination {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_next(mut self, url: impl Into<String>) -> Self {
        self.next = Some(url.into());
        self
    }

    pub fn with_previous(mut self, url: impl Into<String>) -> Self {
        self.previous = Some(url.into());
        self
    }

    pub fn with_first(mut self, url: impl Into<String>) -> Self {
        self.first = Some(url.into());
        self
    }

    pub fn with_last(mut self, url: impl Into<String>) -> Self {
        self.last = Some(url.into());
        self
    }
}

// ── Feed ────────────────────────────────────────────────────────

/// A complete feed (RSS or Atom).
#[derive(Debug, Clone)]
pub struct Feed {
    pub title: String,
    pub link: String,
    pub description: String,
    pub language: Option<String>,
    pub updated: Option<String>,
    pub id: Option<String>,
    pub authors: Vec<Author>,
    pub categories: Vec<Category>,
    pub entries: Vec<FeedEntry>,
    pub pagination: FeedPagination,
    pub generator: Option<String>,
    pub icon: Option<String>,
    pub logo: Option<String>,
}

impl Feed {
    pub fn new(
        title: impl Into<String>,
        link: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            link: link.into(),
            description: description.into(),
            language: None,
            updated: None,
            id: None,
            authors: Vec::new(),
            categories: Vec::new(),
            entries: Vec::new(),
            pagination: FeedPagination::new(),
            generator: None,
            icon: None,
            logo: None,
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    pub fn with_updated(mut self, date: impl Into<String>) -> Self {
        self.updated = Some(date.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_generator(mut self, gen_name: impl Into<String>) -> Self {
        self.generator = Some(gen_name.into());
        self
    }

    pub fn with_icon(mut self, icon_url: impl Into<String>) -> Self {
        self.icon = Some(icon_url.into());
        self
    }

    pub fn with_logo(mut self, logo_url: impl Into<String>) -> Self {
        self.logo = Some(logo_url.into());
        self
    }

    pub fn add_author(mut self, author: Author) -> Self {
        self.authors.push(author);
        self
    }

    pub fn add_category(mut self, category: Category) -> Self {
        self.categories.push(category);
        self
    }

    pub fn add_entry(mut self, entry: FeedEntry) -> Self {
        self.entries.push(entry);
        self
    }

    pub fn with_pagination(mut self, pagination: FeedPagination) -> Self {
        self.pagination = pagination;
        self
    }

    /// Render as RSS 2.0 XML.
    pub fn to_rss(&self) -> Result<String, FeedError> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<rss version=\"2.0\" xmlns:atom=\"http://www.w3.org/2005/Atom\">\n");
        xml.push_str("<channel>\n");

        let _ = write!(xml, "  <title>{}</title>\n", xml_escape(&self.title));
        let _ = write!(xml, "  <link>{}</link>\n", xml_escape(&self.link));
        let _ = write!(
            xml,
            "  <description>{}</description>\n",
            xml_escape(&self.description)
        );

        if let Some(lang) = &self.language {
            let _ = write!(xml, "  <language>{}</language>\n", xml_escape(lang));
        }

        if let Some(updated) = &self.updated {
            let _ = write!(xml, "  <lastBuildDate>{}</lastBuildDate>\n", xml_escape(updated));
        }

        if let Some(gen_name) = &self.generator {
            let _ = write!(xml, "  <generator>{}</generator>\n", xml_escape(gen_name));
        }

        // Atom self link
        let _ = write!(
            xml,
            "  <atom:link href=\"{}\" rel=\"self\" type=\"application/rss+xml\"/>\n",
            xml_escape(&self.link)
        );

        // Pagination links via atom namespace
        if let Some(next) = &self.pagination.next {
            let _ = write!(
                xml,
                "  <atom:link href=\"{}\" rel=\"next\" type=\"application/rss+xml\"/>\n",
                xml_escape(next)
            );
        }
        if let Some(prev) = &self.pagination.previous {
            let _ = write!(
                xml,
                "  <atom:link href=\"{}\" rel=\"previous\" type=\"application/rss+xml\"/>\n",
                xml_escape(prev)
            );
        }

        for cat in &self.categories {
            let _ = write!(xml, "  <category>{}</category>\n", xml_escape(&cat.term));
        }

        // Items
        for entry in &self.entries {
            xml.push_str("  <item>\n");
            let _ = write!(xml, "    <title>{}</title>\n", xml_escape(&entry.title));
            let _ = write!(xml, "    <link>{}</link>\n", xml_escape(&entry.link));
            let _ = write!(xml, "    <guid>{}</guid>\n", xml_escape(&entry.id));

            if let Some(summary) = &entry.summary {
                let _ = write!(
                    xml,
                    "    <description>{}</description>\n",
                    xml_escape(summary)
                );
            }

            if let Some(content) = &entry.content {
                let _ = write!(
                    xml,
                    "    <content:encoded><![CDATA[{}]]></content:encoded>\n",
                    content
                );
            }

            if let Some(pub_date) = &entry.published {
                let _ = write!(xml, "    <pubDate>{}</pubDate>\n", xml_escape(pub_date));
            }

            for author in &entry.authors {
                if let Some(email) = &author.email {
                    let _ = write!(
                        xml,
                        "    <author>{} ({})</author>\n",
                        xml_escape(email),
                        xml_escape(&author.name)
                    );
                }
            }

            for cat in &entry.categories {
                let _ = write!(xml, "    <category>{}</category>\n", xml_escape(&cat.term));
            }

            xml.push_str("  </item>\n");
        }

        xml.push_str("</channel>\n</rss>");
        Ok(xml)
    }

    /// Render as Atom XML.
    pub fn to_atom(&self) -> Result<String, FeedError> {
        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<feed xmlns=\"http://www.w3.org/2005/Atom\">\n");

        let _ = write!(xml, "  <title>{}</title>\n", xml_escape(&self.title));
        let _ = write!(xml, "  <subtitle>{}</subtitle>\n", xml_escape(&self.description));
        let _ = write!(
            xml,
            "  <link href=\"{}\" rel=\"alternate\"/>\n",
            xml_escape(&self.link)
        );

        let feed_id = self.id.as_deref().unwrap_or(&self.link);
        let _ = write!(xml, "  <id>{}</id>\n", xml_escape(feed_id));

        if let Some(updated) = &self.updated {
            let _ = write!(xml, "  <updated>{}</updated>\n", xml_escape(updated));
        }

        if let Some(gen_name) = &self.generator {
            let _ = write!(xml, "  <generator>{}</generator>\n", xml_escape(gen_name));
        }

        if let Some(icon) = &self.icon {
            let _ = write!(xml, "  <icon>{}</icon>\n", xml_escape(icon));
        }

        if let Some(logo) = &self.logo {
            let _ = write!(xml, "  <logo>{}</logo>\n", xml_escape(logo));
        }

        for author in &self.authors {
            xml.push_str("  <author>\n");
            let _ = write!(xml, "    <name>{}</name>\n", xml_escape(&author.name));
            if let Some(email) = &author.email {
                let _ = write!(xml, "    <email>{}</email>\n", xml_escape(email));
            }
            if let Some(uri) = &author.uri {
                let _ = write!(xml, "    <uri>{}</uri>\n", xml_escape(uri));
            }
            xml.push_str("  </author>\n");
        }

        // Pagination
        if let Some(next) = &self.pagination.next {
            let _ = write!(xml, "  <link href=\"{}\" rel=\"next\"/>\n", xml_escape(next));
        }
        if let Some(prev) = &self.pagination.previous {
            let _ = write!(xml, "  <link href=\"{}\" rel=\"previous\"/>\n", xml_escape(prev));
        }
        if let Some(first) = &self.pagination.first {
            let _ = write!(xml, "  <link href=\"{}\" rel=\"first\"/>\n", xml_escape(first));
        }
        if let Some(last) = &self.pagination.last {
            let _ = write!(xml, "  <link href=\"{}\" rel=\"last\"/>\n", xml_escape(last));
        }

        for cat in &self.categories {
            let label = cat.label.as_deref().unwrap_or(&cat.term);
            let _ = write!(
                xml,
                "  <category term=\"{}\" label=\"{}\"/>\n",
                xml_escape(&cat.term),
                xml_escape(label)
            );
        }

        // Entries
        for entry in &self.entries {
            xml.push_str("  <entry>\n");
            let _ = write!(xml, "    <id>{}</id>\n", xml_escape(&entry.id));
            let _ = write!(xml, "    <title>{}</title>\n", xml_escape(&entry.title));
            let _ = write!(
                xml,
                "    <link href=\"{}\" rel=\"alternate\"/>\n",
                xml_escape(&entry.link)
            );

            if let Some(summary) = &entry.summary {
                let _ = write!(
                    xml,
                    "    <summary>{}</summary>\n",
                    xml_escape(summary)
                );
            }

            if let Some(content) = &entry.content {
                let _ = write!(
                    xml,
                    "    <content type=\"html\">{}</content>\n",
                    xml_escape(content)
                );
            }

            if let Some(published) = &entry.published {
                let _ = write!(xml, "    <published>{}</published>\n", xml_escape(published));
            }

            if let Some(updated) = &entry.updated {
                let _ = write!(xml, "    <updated>{}</updated>\n", xml_escape(updated));
            }

            for author in &entry.authors {
                xml.push_str("    <author>\n");
                let _ = write!(xml, "      <name>{}</name>\n", xml_escape(&author.name));
                if let Some(email) = &author.email {
                    let _ = write!(xml, "      <email>{}</email>\n", xml_escape(email));
                }
                xml.push_str("    </author>\n");
            }

            for cat in &entry.categories {
                let _ = write!(
                    xml,
                    "    <category term=\"{}\"/>\n",
                    xml_escape(&cat.term)
                );
            }

            xml.push_str("  </entry>\n");
        }

        xml.push_str("</feed>");
        Ok(xml)
    }

    /// Render to the given format.
    pub fn render(&self, format: FeedFormat) -> Result<String, FeedError> {
        match format {
            FeedFormat::Rss20 => self.to_rss(),
            FeedFormat::Atom => self.to_atom(),
        }
    }

    /// Total number of entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

// ── RFC 822 Date Formatting ─────────────────────────────────────

/// Day-of-week names for RFC 822.
const RFC822_DOW: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
/// Month names for RFC 822.
const RFC822_MON: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Format a date as RFC 822 (used in RSS).
///
/// Accepts year, month (1-12), day (1-31), hour, minute, second, and a
/// day-of-week index (0=Sun).
pub fn format_rfc822(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    dow: usize,
) -> String {
    let dow_str = RFC822_DOW.get(dow % 7).unwrap_or(&"Mon");
    let mon_str = RFC822_MON
        .get((month.saturating_sub(1)) as usize % 12)
        .unwrap_or(&"Jan");
    format!(
        "{}, {:02} {} {} {:02}:{:02}:{:02} +0000",
        dow_str, day, mon_str, year, hour, minute, second
    )
}

/// Format a date as RFC 3339 / ISO 8601 (used in Atom).
pub fn format_rfc3339(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, minute, second
    )
}

// ── XML Utilities ───────────────────────────────────────────────

/// Escape XML special characters.
fn xml_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
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

    fn sample_feed() -> Feed {
        Feed::new("Test Blog", "https://example.com", "A test blog feed")
            .with_language("en")
            .with_updated("2026-03-09T12:00:00Z")
            .with_id("https://example.com/feed")
            .with_generator("JoulesPerBit")
            .add_author(Author::new("Alice").with_email("alice@example.com"))
    }

    fn sample_entry() -> FeedEntry {
        FeedEntry::new(
            "https://example.com/posts/1",
            "First Post",
            "https://example.com/posts/1",
        )
        .with_summary("This is the first post.")
        .with_content("<p>Full post content here.</p>")
        .with_published("Sat, 09 Mar 2026 12:00:00 +0000")
        .add_author(Author::new("Alice"))
        .add_category(Category::new("tech"))
    }

    #[test]
    fn test_rss_basic() {
        let feed = sample_feed().add_entry(sample_entry());
        let xml = feed.to_rss().unwrap();
        assert!(xml.contains("<rss version=\"2.0\""));
        assert!(xml.contains("<title>Test Blog</title>"));
        assert!(xml.contains("<item>"));
        assert!(xml.contains("First Post"));
    }

    #[test]
    fn test_atom_basic() {
        let feed = sample_feed().add_entry(sample_entry());
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("<feed xmlns="));
        assert!(xml.contains("<title>Test Blog</title>"));
        assert!(xml.contains("<entry>"));
    }

    #[test]
    fn test_rss_contains_channel_metadata() {
        let feed = sample_feed();
        let xml = feed.to_rss().unwrap();
        assert!(xml.contains("<language>en</language>"));
        assert!(xml.contains("<generator>JoulesPerBit</generator>"));
        assert!(xml.contains("<lastBuildDate>"));
    }

    #[test]
    fn test_atom_contains_author() {
        let feed = sample_feed();
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("<author>"));
        assert!(xml.contains("<name>Alice</name>"));
        assert!(xml.contains("<email>alice@example.com</email>"));
    }

    #[test]
    fn test_rss_entry_categories() {
        let entry = sample_entry()
            .add_category(Category::new("rust"))
            .add_category(Category::new("web"));
        let feed = sample_feed().add_entry(entry);
        let xml = feed.to_rss().unwrap();
        assert!(xml.contains("<category>tech</category>"));
        assert!(xml.contains("<category>rust</category>"));
    }

    #[test]
    fn test_atom_categories() {
        let feed = sample_feed()
            .add_category(Category::new("tech").with_label("Technology"))
            .add_entry(sample_entry());
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("term=\"tech\""));
        assert!(xml.contains("label=\"Technology\""));
    }

    #[test]
    fn test_pagination_rss() {
        let feed = sample_feed().with_pagination(
            FeedPagination::new()
                .with_next("https://example.com/feed?page=2")
                .with_previous("https://example.com/feed?page=0"),
        );
        let xml = feed.to_rss().unwrap();
        assert!(xml.contains("rel=\"next\""));
        assert!(xml.contains("rel=\"previous\""));
    }

    #[test]
    fn test_pagination_atom() {
        let feed = sample_feed().with_pagination(
            FeedPagination::new()
                .with_first("https://example.com/feed?page=1")
                .with_last("https://example.com/feed?page=10"),
        );
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("rel=\"first\""));
        assert!(xml.contains("rel=\"last\""));
    }

    #[test]
    fn test_rfc822_format() {
        let date = format_rfc822(2026, 3, 9, 12, 0, 0, 6); // 6 = Sat
        assert_eq!(date, "Sat, 09 Mar 2026 12:00:00 +0000");
    }

    #[test]
    fn test_rfc3339_format() {
        let date = format_rfc3339(2026, 3, 9, 12, 0, 0);
        assert_eq!(date, "2026-03-09T12:00:00Z");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a & b"), "a &amp; b");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(xml_escape("it's"), "it&apos;s");
    }

    #[test]
    fn test_entry_count() {
        let feed = sample_feed()
            .add_entry(sample_entry())
            .add_entry(sample_entry());
        assert_eq!(feed.entry_count(), 2);
    }

    #[test]
    fn test_empty_feed_rss() {
        let feed = Feed::new("Empty", "https://example.com", "Empty feed");
        let xml = feed.to_rss().unwrap();
        assert!(xml.contains("<channel>"));
        assert!(!xml.contains("<item>"));
    }

    #[test]
    fn test_empty_feed_atom() {
        let feed = Feed::new("Empty", "https://example.com", "Empty feed");
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("<feed"));
        assert!(!xml.contains("<entry>"));
    }

    #[test]
    fn test_feed_render_rss() {
        let feed = sample_feed();
        let xml = feed.render(FeedFormat::Rss20).unwrap();
        assert!(xml.contains("<rss"));
    }

    #[test]
    fn test_feed_render_atom() {
        let feed = sample_feed();
        let xml = feed.render(FeedFormat::Atom).unwrap();
        assert!(xml.contains("<feed"));
    }

    #[test]
    fn test_rss_content_cdata() {
        let entry = sample_entry();
        let feed = sample_feed().add_entry(entry);
        let xml = feed.to_rss().unwrap();
        assert!(xml.contains("<![CDATA["));
    }

    #[test]
    fn test_atom_entry_with_updated() {
        let entry = sample_entry().with_updated("2026-03-09T14:00:00Z");
        let feed = sample_feed().add_entry(entry);
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("<updated>2026-03-09T14:00:00Z</updated>"));
    }

    #[test]
    fn test_author_builder() {
        let author = Author::new("Bob")
            .with_email("bob@test.com")
            .with_uri("https://bob.test");
        assert_eq!(author.name, "Bob");
        assert_eq!(author.email.as_deref(), Some("bob@test.com"));
        assert_eq!(author.uri.as_deref(), Some("https://bob.test"));
    }

    #[test]
    fn test_feed_icon_and_logo() {
        let feed = sample_feed()
            .with_icon("https://example.com/icon.png")
            .with_logo("https://example.com/logo.png");
        let xml = feed.to_atom().unwrap();
        assert!(xml.contains("<icon>https://example.com/icon.png</icon>"));
        assert!(xml.contains("<logo>https://example.com/logo.png</logo>"));
    }
}
