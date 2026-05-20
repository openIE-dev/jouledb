//! Sitemap XML generator — URL entries with lastmod/changefreq/priority,
//! sitemap index for large sites, alternate language links (hreflang),
//! image/video sitemap extensions, XML output, and URL count limits.
//!
//! Pure-Rust replacement for sitemap.js, next-sitemap, and similar Node.js
//! sitemap generation libraries.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from sitemap generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SitemapError {
    UrlLimitExceeded(usize),
    InvalidPriority(String),
    InvalidUrl(String),
}

impl fmt::Display for SitemapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UrlLimitExceeded(n) => write!(f, "URL limit exceeded: {n} > 50000"),
            Self::InvalidPriority(msg) => write!(f, "invalid priority: {msg}"),
            Self::InvalidUrl(msg) => write!(f, "invalid URL: {msg}"),
        }
    }
}

impl std::error::Error for SitemapError {}

// ── Change Frequency ────────────────────────────────────────────

/// How frequently the page is likely to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeFreq {
    Always,
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Yearly,
    Never,
}

impl ChangeFreq {
    fn as_str(&self) -> &str {
        match self {
            Self::Always => "always",
            Self::Hourly => "hourly",
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
            Self::Yearly => "yearly",
            Self::Never => "never",
        }
    }
}

impl fmt::Display for ChangeFreq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Alternate Link ──────────────────────────────────────────────

/// An alternate language link (hreflang).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlternateLink {
    pub hreflang: String,
    pub href: String,
}

impl AlternateLink {
    pub fn new(hreflang: impl Into<String>, href: impl Into<String>) -> Self {
        Self {
            hreflang: hreflang.into(),
            href: href.into(),
        }
    }
}

// ── Image Entry ─────────────────────────────────────────────────

/// An image entry for image sitemaps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageEntry {
    pub loc: String,
    pub caption: Option<String>,
    pub title: Option<String>,
    pub geo_location: Option<String>,
    pub license: Option<String>,
}

impl ImageEntry {
    pub fn new(loc: impl Into<String>) -> Self {
        Self {
            loc: loc.into(),
            caption: None,
            title: None,
            geo_location: None,
            license: None,
        }
    }

    pub fn with_caption(mut self, caption: impl Into<String>) -> Self {
        self.caption = Some(caption.into());
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_geo_location(mut self, geo: impl Into<String>) -> Self {
        self.geo_location = Some(geo.into());
        self
    }
}

// ── Video Entry ─────────────────────────────────────────────────

/// A video entry for video sitemaps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoEntry {
    pub thumbnail_loc: String,
    pub title: String,
    pub description: String,
    pub content_loc: Option<String>,
    pub player_loc: Option<String>,
    pub duration_secs: Option<u32>,
}

impl VideoEntry {
    pub fn new(
        thumbnail_loc: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            thumbnail_loc: thumbnail_loc.into(),
            title: title.into(),
            description: description.into(),
            content_loc: None,
            player_loc: None,
            duration_secs: None,
        }
    }

    pub fn with_content_loc(mut self, loc: impl Into<String>) -> Self {
        self.content_loc = Some(loc.into());
        self
    }

    pub fn with_player_loc(mut self, loc: impl Into<String>) -> Self {
        self.player_loc = Some(loc.into());
        self
    }

    pub fn with_duration(mut self, secs: u32) -> Self {
        self.duration_secs = Some(secs);
        self
    }
}

// ── URL Entry ───────────────────────────────────────────────────

/// A URL entry in a sitemap.
#[derive(Debug, Clone)]
pub struct UrlEntry {
    pub loc: String,
    pub lastmod: Option<String>,
    pub changefreq: Option<ChangeFreq>,
    pub priority: Option<f64>,
    pub alternates: Vec<AlternateLink>,
    pub images: Vec<ImageEntry>,
    pub videos: Vec<VideoEntry>,
}

impl UrlEntry {
    pub fn new(loc: impl Into<String>) -> Self {
        Self {
            loc: loc.into(),
            lastmod: None,
            changefreq: None,
            priority: None,
            alternates: Vec::new(),
            images: Vec::new(),
            videos: Vec::new(),
        }
    }

    pub fn with_lastmod(mut self, lastmod: impl Into<String>) -> Self {
        self.lastmod = Some(lastmod.into());
        self
    }

    pub fn with_changefreq(mut self, freq: ChangeFreq) -> Self {
        self.changefreq = Some(freq);
        self
    }

    pub fn with_priority(mut self, priority: f64) -> Result<Self, SitemapError> {
        if !(0.0..=1.0).contains(&priority) {
            return Err(SitemapError::InvalidPriority(format!(
                "{priority} is not between 0.0 and 1.0"
            )));
        }
        self.priority = Some(priority);
        Ok(self)
    }

    pub fn add_alternate(mut self, alt: AlternateLink) -> Self {
        self.alternates.push(alt);
        self
    }

    pub fn add_image(mut self, img: ImageEntry) -> Self {
        self.images.push(img);
        self
    }

    pub fn add_video(mut self, vid: VideoEntry) -> Self {
        self.videos.push(vid);
        self
    }
}

// ── Sitemap Builder ─────────────────────────────────────────────

/// Maximum URLs per sitemap file (per sitemap protocol spec).
pub const MAX_URLS_PER_SITEMAP: usize = 50_000;

/// Builder for sitemap XML.
#[derive(Debug, Clone)]
pub struct SitemapBuilder {
    urls: Vec<UrlEntry>,
}

impl SitemapBuilder {
    pub fn new() -> Self {
        Self { urls: Vec::new() }
    }

    /// Add a URL entry.
    pub fn add_url(&mut self, url: UrlEntry) {
        self.urls.push(url);
    }

    /// Number of URLs.
    pub fn url_count(&self) -> usize {
        self.urls.len()
    }

    /// Whether this sitemap exceeds the URL limit and needs splitting.
    pub fn needs_index(&self) -> bool {
        self.urls.len() > MAX_URLS_PER_SITEMAP
    }

    /// Render to sitemap XML. Returns an error if the URL count exceeds 50K.
    pub fn to_xml(&self) -> Result<String, SitemapError> {
        if self.urls.len() > MAX_URLS_PER_SITEMAP {
            return Err(SitemapError::UrlLimitExceeded(self.urls.len()));
        }
        Ok(self.render_urlset(&self.urls))
    }

    /// Split into multiple sitemaps (each <=50K URLs) and render them.
    pub fn to_xml_chunks(&self) -> Vec<String> {
        self.urls
            .chunks(MAX_URLS_PER_SITEMAP)
            .map(|chunk| self.render_urlset(chunk))
            .collect()
    }

    fn render_urlset(&self, urls: &[UrlEntry]) -> String {
        let has_images = urls.iter().any(|u| !u.images.is_empty());
        let has_videos = urls.iter().any(|u| !u.videos.is_empty());
        let has_alternates = urls.iter().any(|u| !u.alternates.is_empty());

        let mut xml = String::new();
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        let _ = write!(xml, "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\"");
        if has_alternates {
            let _ = write!(xml, "\n  xmlns:xhtml=\"http://www.w3.org/1999/xhtml\"");
        }
        if has_images {
            let _ = write!(
                xml,
                "\n  xmlns:image=\"http://www.google.com/schemas/sitemap-image/1.1\""
            );
        }
        if has_videos {
            let _ = write!(
                xml,
                "\n  xmlns:video=\"http://www.google.com/schemas/sitemap-video/1.1\""
            );
        }
        xml.push_str(">\n");

        for url in urls {
            xml.push_str("  <url>\n");
            let _ = write!(xml, "    <loc>{}</loc>\n", xml_escape(&url.loc));

            if let Some(lastmod) = &url.lastmod {
                let _ = write!(xml, "    <lastmod>{}</lastmod>\n", xml_escape(lastmod));
            }

            if let Some(freq) = &url.changefreq {
                let _ = write!(xml, "    <changefreq>{}</changefreq>\n", freq.as_str());
            }

            if let Some(priority) = url.priority {
                let _ = write!(xml, "    <priority>{:.1}</priority>\n", priority);
            }

            // Alternate links (hreflang)
            for alt in &url.alternates {
                let _ = write!(
                    xml,
                    "    <xhtml:link rel=\"alternate\" hreflang=\"{}\" href=\"{}\"/>\n",
                    xml_escape(&alt.hreflang),
                    xml_escape(&alt.href)
                );
            }

            // Images
            for img in &url.images {
                xml.push_str("    <image:image>\n");
                let _ = write!(xml, "      <image:loc>{}</image:loc>\n", xml_escape(&img.loc));
                if let Some(caption) = &img.caption {
                    let _ = write!(
                        xml,
                        "      <image:caption>{}</image:caption>\n",
                        xml_escape(caption)
                    );
                }
                if let Some(title) = &img.title {
                    let _ = write!(
                        xml,
                        "      <image:title>{}</image:title>\n",
                        xml_escape(title)
                    );
                }
                if let Some(geo) = &img.geo_location {
                    let _ = write!(
                        xml,
                        "      <image:geo_location>{}</image:geo_location>\n",
                        xml_escape(geo)
                    );
                }
                xml.push_str("    </image:image>\n");
            }

            // Videos
            for vid in &url.videos {
                xml.push_str("    <video:video>\n");
                let _ = write!(
                    xml,
                    "      <video:thumbnail_loc>{}</video:thumbnail_loc>\n",
                    xml_escape(&vid.thumbnail_loc)
                );
                let _ = write!(
                    xml,
                    "      <video:title>{}</video:title>\n",
                    xml_escape(&vid.title)
                );
                let _ = write!(
                    xml,
                    "      <video:description>{}</video:description>\n",
                    xml_escape(&vid.description)
                );
                if let Some(content_loc) = &vid.content_loc {
                    let _ = write!(
                        xml,
                        "      <video:content_loc>{}</video:content_loc>\n",
                        xml_escape(content_loc)
                    );
                }
                if let Some(player_loc) = &vid.player_loc {
                    let _ = write!(
                        xml,
                        "      <video:player_loc>{}</video:player_loc>\n",
                        xml_escape(player_loc)
                    );
                }
                if let Some(dur) = vid.duration_secs {
                    let _ = write!(xml, "      <video:duration>{}</video:duration>\n", dur);
                }
                xml.push_str("    </video:video>\n");
            }

            xml.push_str("  </url>\n");
        }

        xml.push_str("</urlset>");
        xml
    }
}

impl Default for SitemapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sitemap Index ───────────────────────────────────────────────

/// An entry in a sitemap index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SitemapIndexEntry {
    pub loc: String,
    pub lastmod: Option<String>,
}

impl SitemapIndexEntry {
    pub fn new(loc: impl Into<String>) -> Self {
        Self {
            loc: loc.into(),
            lastmod: None,
        }
    }

    pub fn with_lastmod(mut self, lastmod: impl Into<String>) -> Self {
        self.lastmod = Some(lastmod.into());
        self
    }
}

/// Generate a sitemap index XML from a list of sitemap entries.
pub fn render_sitemap_index(entries: &[SitemapIndexEntry]) -> String {
    let mut xml = String::new();
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<sitemapindex xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n");

    for entry in entries {
        xml.push_str("  <sitemap>\n");
        let _ = write!(xml, "    <loc>{}</loc>\n", xml_escape(&entry.loc));
        if let Some(lastmod) = &entry.lastmod {
            let _ = write!(xml, "    <lastmod>{}</lastmod>\n", xml_escape(lastmod));
        }
        xml.push_str("  </sitemap>\n");
    }

    xml.push_str("</sitemapindex>");
    xml
}

/// Build a sitemap index from a large URL set, splitting into multiple
/// sitemap files and returning (sitemap_index_xml, Vec<sitemap_xml>).
pub fn build_sitemap_with_index(
    urls: &[UrlEntry],
    base_url: &str,
    sitemap_prefix: &str,
) -> (String, Vec<String>) {
    let chunks: Vec<&[UrlEntry]> = urls.chunks(MAX_URLS_PER_SITEMAP).collect();
    let builder = SitemapBuilder::new();

    let mut index_entries = Vec::new();
    let mut sitemap_xmls = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        let sitemap_url = format!("{}/{}{}.xml", base_url, sitemap_prefix, i + 1);
        index_entries.push(SitemapIndexEntry::new(sitemap_url));
        sitemap_xmls.push(builder.render_urlset(chunk));
    }

    let index_xml = render_sitemap_index(&index_entries);
    (index_xml, sitemap_xmls)
}

// ── XML Utilities ───────────────────────────────────────────────

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

    #[test]
    fn test_basic_sitemap() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(UrlEntry::new("https://example.com/"));
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("<urlset"));
        assert!(xml.contains("<loc>https://example.com/</loc>"));
    }

    #[test]
    fn test_url_with_all_fields() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(
            UrlEntry::new("https://example.com/page")
                .with_lastmod("2026-03-09")
                .with_changefreq(ChangeFreq::Weekly)
                .with_priority(0.8)
                .unwrap(),
        );
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("<lastmod>2026-03-09</lastmod>"));
        assert!(xml.contains("<changefreq>weekly</changefreq>"));
        assert!(xml.contains("<priority>0.8</priority>"));
    }

    #[test]
    fn test_invalid_priority_low() {
        let result = UrlEntry::new("https://example.com").with_priority(-0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_priority_high() {
        let result = UrlEntry::new("https://example.com").with_priority(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_alternate_links() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(
            UrlEntry::new("https://example.com/en")
                .add_alternate(AlternateLink::new("de", "https://example.com/de"))
                .add_alternate(AlternateLink::new("fr", "https://example.com/fr")),
        );
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("xhtml:link"));
        assert!(xml.contains("hreflang=\"de\""));
        assert!(xml.contains("hreflang=\"fr\""));
    }

    #[test]
    fn test_image_sitemap() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(UrlEntry::new("https://example.com/gallery").add_image(
            ImageEntry::new("https://example.com/img/photo.jpg")
                .with_caption("A nice photo")
                .with_title("Photo 1"),
        ));
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("image:image"));
        assert!(xml.contains("image:loc"));
        assert!(xml.contains("A nice photo"));
    }

    #[test]
    fn test_video_sitemap() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(UrlEntry::new("https://example.com/video").add_video(
            VideoEntry::new(
                "https://example.com/thumb.jpg",
                "My Video",
                "A great video",
            )
            .with_content_loc("https://example.com/video.mp4")
            .with_duration(120),
        ));
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("video:video"));
        assert!(xml.contains("video:thumbnail_loc"));
        assert!(xml.contains("<video:duration>120</video:duration>"));
    }

    #[test]
    fn test_url_count() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(UrlEntry::new("https://a.com"));
        builder.add_url(UrlEntry::new("https://b.com"));
        assert_eq!(builder.url_count(), 2);
    }

    #[test]
    fn test_needs_index() {
        let builder = SitemapBuilder::new();
        assert!(!builder.needs_index());
    }

    #[test]
    fn test_sitemap_index_render() {
        let entries = vec![
            SitemapIndexEntry::new("https://example.com/sitemap1.xml")
                .with_lastmod("2026-03-09"),
            SitemapIndexEntry::new("https://example.com/sitemap2.xml"),
        ];
        let xml = render_sitemap_index(&entries);
        assert!(xml.contains("<sitemapindex"));
        assert!(xml.contains("sitemap1.xml"));
        assert!(xml.contains("sitemap2.xml"));
        assert!(xml.contains("<lastmod>"));
    }

    #[test]
    fn test_build_sitemap_with_index() {
        let urls: Vec<UrlEntry> = (0..5)
            .map(|i| UrlEntry::new(format!("https://example.com/page{}", i)))
            .collect();
        let (index, sitemaps) = build_sitemap_with_index(&urls, "https://example.com", "sitemap");
        assert!(index.contains("<sitemapindex"));
        assert_eq!(sitemaps.len(), 1);
        assert!(sitemaps[0].contains("page0"));
    }

    #[test]
    fn test_changefreq_display() {
        assert_eq!(ChangeFreq::Daily.to_string(), "daily");
        assert_eq!(ChangeFreq::Always.to_string(), "always");
        assert_eq!(ChangeFreq::Never.to_string(), "never");
    }

    #[test]
    fn test_xml_escape_in_sitemap() {
        let mut builder = SitemapBuilder::new();
        builder.add_url(UrlEntry::new("https://example.com/?a=1&b=2"));
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("a=1&amp;b=2"));
    }

    #[test]
    fn test_empty_sitemap() {
        let builder = SitemapBuilder::new();
        let xml = builder.to_xml().unwrap();
        assert!(xml.contains("<urlset"));
        assert!(xml.contains("</urlset>"));
    }

    #[test]
    fn test_image_with_geo() {
        let img = ImageEntry::new("https://example.com/img.jpg")
            .with_geo_location("New York, USA");
        assert_eq!(img.geo_location.as_deref(), Some("New York, USA"));
    }

    #[test]
    fn test_video_with_player() {
        let vid = VideoEntry::new("https://t.jpg", "Title", "Desc")
            .with_player_loc("https://example.com/player");
        assert_eq!(vid.player_loc.as_deref(), Some("https://example.com/player"));
    }

    #[test]
    fn test_to_xml_chunks() {
        let mut builder = SitemapBuilder::new();
        for i in 0..10 {
            builder.add_url(UrlEntry::new(format!("https://example.com/{}", i)));
        }
        let chunks = builder.to_xml_chunks();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("example.com/0"));
    }

    #[test]
    fn test_valid_priority_bounds() {
        let entry = UrlEntry::new("https://example.com")
            .with_priority(0.0)
            .unwrap();
        assert_eq!(entry.priority, Some(0.0));
        let entry = UrlEntry::new("https://example.com")
            .with_priority(1.0)
            .unwrap();
        assert_eq!(entry.priority, Some(1.0));
    }
}
