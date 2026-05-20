//! Content processing pipeline — plugin-based transformation chain,
//! content types, frontmatter extraction, slug generation, excerpt extraction,
//! reading time estimation, and word count.
//!
//! Pure-Rust replacement for gray-matter, reading-time, remark-slug,
//! remark-excerpt, and similar Node.js content processing tools.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from content pipeline processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    InvalidContent(String),
    PluginFailed(String),
    InvalidFrontmatter(String),
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContent(msg) => write!(f, "invalid content: {msg}"),
            Self::PluginFailed(msg) => write!(f, "plugin failed: {msg}"),
            Self::InvalidFrontmatter(msg) => write!(f, "invalid frontmatter: {msg}"),
        }
    }
}

impl std::error::Error for PipelineError {}

// ── Content Type ────────────────────────────────────────────────

/// The type of content being processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Markdown,
    Html,
    PlainText,
}

impl ContentType {
    /// Detect content type from file extension.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "md" | "markdown" | "mdx" => Self::Markdown,
            "html" | "htm" => Self::Html,
            _ => Self::PlainText,
        }
    }
}

// ── Frontmatter ─────────────────────────────────────────────────

/// Simple key-value frontmatter extracted from YAML front matter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Frontmatter {
    fields: Vec<(String, String)>,
}

impl Frontmatter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        if let Some(entry) = self.fields.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = value.into();
        } else {
            self.fields.push((key, value.into()));
        }
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|(k, _)| k.as_str())
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }
}

/// Extract YAML front matter from content.
pub fn extract_frontmatter(input: &str) -> Result<(Frontmatter, String), PipelineError> {
    let trimmed = input.trim_start();
    if !trimmed.starts_with("---") {
        return Ok((Frontmatter::new(), input.to_string()));
    }

    let after_delim = &trimmed[3..];
    let after_delim = after_delim.strip_prefix('\n').unwrap_or(after_delim);

    match after_delim.find("\n---") {
        None => Err(PipelineError::InvalidFrontmatter(
            "unclosed front matter".into(),
        )),
        Some(pos) => {
            let yaml = &after_delim[..pos];
            let rest_start = pos + 4;
            let rest = if rest_start < after_delim.len() {
                let r = &after_delim[rest_start..];
                r.strip_prefix('\n').unwrap_or(r).to_string()
            } else {
                String::new()
            };

            let mut fm = Frontmatter::new();
            for line in yaml.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Some((key, val)) = line.split_once(':') {
                    fm.set(key.trim(), val.trim());
                }
            }

            Ok((fm, rest))
        }
    }
}

// ── Slug Generation ─────────────────────────────────────────────

/// Generate a URL-friendly slug from a string.
pub fn generate_slug(input: &str) -> String {
    let mut slug = String::new();
    let mut last_was_hyphen = false;

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_hyphen = false;
        } else if ch == ' ' || ch == '-' || ch == '_' {
            if !last_was_hyphen && !slug.is_empty() {
                slug.push('-');
                last_was_hyphen = true;
            }
        }
    }

    // Remove trailing hyphen
    if slug.ends_with('-') {
        slug.pop();
    }

    slug
}

// ── Word Count ──────────────────────────────────────────────────

/// Count words in text, stripping HTML tags if present.
pub fn word_count(text: &str) -> usize {
    let stripped = strip_html_tags(text);
    stripped.split_whitespace().count()
}

/// Strip HTML tags from text.
fn strip_html_tags(input: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in input.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
            out.push(' ');
        } else if !in_tag {
            out.push(ch);
        }
    }
    out
}

// ── Reading Time ────────────────────────────────────────────────

/// Estimated reading time result.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadingTime {
    /// Estimated minutes to read.
    pub minutes: usize,
    /// Total words in the content.
    pub words: usize,
    /// Words per minute used for the estimate.
    pub wpm: usize,
}

/// Estimate reading time. Default 200 WPM.
pub fn reading_time(text: &str) -> ReadingTime {
    reading_time_with_wpm(text, 200)
}

/// Estimate reading time with a custom WPM.
pub fn reading_time_with_wpm(text: &str, wpm: usize) -> ReadingTime {
    let words = word_count(text);
    let wpm = if wpm == 0 { 200 } else { wpm };
    let minutes = if words == 0 {
        0
    } else {
        (words + wpm - 1) / wpm // ceiling division
    };
    ReadingTime {
        minutes,
        words,
        wpm,
    }
}

// ── Excerpt Extraction ──────────────────────────────────────────

/// Extract an excerpt from content.
pub fn extract_excerpt(text: &str, max_chars: usize) -> String {
    let stripped = strip_html_tags(text);
    let trimmed = stripped.trim();

    if trimmed.len() <= max_chars {
        return trimmed.to_string();
    }

    // Find word boundary near max_chars
    let mut end = max_chars;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    while end > 0 && !trimmed[..end].ends_with(char::is_whitespace) {
        end -= 1;
    }
    if end == 0 {
        end = max_chars.min(trimmed.len());
        while end > 0 && !trimmed.is_char_boundary(end) {
            end -= 1;
        }
    }

    let mut excerpt = trimmed[..end].trim_end().to_string();
    excerpt.push_str("...");
    excerpt
}

/// Extract excerpt by separator (e.g., `<!-- more -->`).
pub fn extract_excerpt_by_separator(text: &str, separator: &str) -> Option<String> {
    text.find(separator)
        .map(|pos| text[..pos].trim().to_string())
}

// ── Content Item ────────────────────────────────────────────────

/// A content item flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct ContentItem {
    pub source: String,
    pub content_type: ContentType,
    pub frontmatter: Frontmatter,
    pub body: String,
    pub slug: String,
    pub excerpt: Option<String>,
    pub word_count: usize,
    pub reading_time: ReadingTime,
    pub metadata: HashMap<String, String>,
}

impl ContentItem {
    /// Create a content item from raw source text.
    pub fn from_source(source: &str, content_type: ContentType) -> Result<Self, PipelineError> {
        let (frontmatter, body) = extract_frontmatter(source)?;

        let title = frontmatter.get("title").unwrap_or("untitled");
        let slug_source = frontmatter.get("slug").unwrap_or(title);
        let slug = generate_slug(slug_source);

        let words = word_count(&body);
        let rt = reading_time(&body);
        let excerpt = Some(extract_excerpt(&body, 160));

        Ok(Self {
            source: source.to_string(),
            content_type,
            frontmatter,
            body,
            slug,
            excerpt,
            word_count: words,
            reading_time: rt,
            metadata: HashMap::new(),
        })
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

// ── Pipeline Plugin ─────────────────────────────────────────────

/// A transformation applied to a content item.
pub trait ContentPlugin {
    /// Name of this plugin.
    fn name(&self) -> &str;

    /// Transform a content item, returning the modified item.
    fn transform(&self, item: ContentItem) -> Result<ContentItem, PipelineError>;
}

// ── Built-in Plugins ────────────────────────────────────────────

/// Plugin that converts body text to uppercase (for testing/demo).
pub struct UppercasePlugin;

impl ContentPlugin for UppercasePlugin {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn transform(&self, mut item: ContentItem) -> Result<ContentItem, PipelineError> {
        item.body = item.body.to_uppercase();
        Ok(item)
    }
}

/// Plugin that strips HTML tags from the body.
pub struct StripHtmlPlugin;

impl ContentPlugin for StripHtmlPlugin {
    fn name(&self) -> &str {
        "strip_html"
    }

    fn transform(&self, mut item: ContentItem) -> Result<ContentItem, PipelineError> {
        item.body = strip_html_tags(&item.body);
        Ok(item)
    }
}

/// Plugin that adds a metadata entry with the word count.
pub struct WordCountPlugin;

impl ContentPlugin for WordCountPlugin {
    fn name(&self) -> &str {
        "word_count"
    }

    fn transform(&self, mut item: ContentItem) -> Result<ContentItem, PipelineError> {
        let wc = word_count(&item.body);
        item.word_count = wc;
        item.set_metadata("word_count", wc.to_string());
        Ok(item)
    }
}

/// Plugin that regenerates the slug from frontmatter title.
pub struct SlugPlugin;

impl ContentPlugin for SlugPlugin {
    fn name(&self) -> &str {
        "slug"
    }

    fn transform(&self, mut item: ContentItem) -> Result<ContentItem, PipelineError> {
        if let Some(title) = item.frontmatter.get("title") {
            let title_owned = title.to_string();
            item.slug = generate_slug(&title_owned);
        }
        Ok(item)
    }
}

/// Plugin that regenerates the excerpt.
pub struct ExcerptPlugin {
    pub max_chars: usize,
}

impl ExcerptPlugin {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl ContentPlugin for ExcerptPlugin {
    fn name(&self) -> &str {
        "excerpt"
    }

    fn transform(&self, mut item: ContentItem) -> Result<ContentItem, PipelineError> {
        item.excerpt = Some(extract_excerpt(&item.body, self.max_chars));
        Ok(item)
    }
}

// ── Pipeline ────────────────────────────────────────────────────

/// A content processing pipeline.
pub struct ContentPipeline {
    plugins: Vec<Box<dyn ContentPlugin>>,
}

impl ContentPipeline {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Add a plugin to the pipeline.
    pub fn add_plugin(&mut self, plugin: Box<dyn ContentPlugin>) {
        self.plugins.push(plugin);
    }

    /// Process a content item through all plugins in order.
    pub fn process(&self, item: ContentItem) -> Result<ContentItem, PipelineError> {
        let mut current = item;
        for plugin in &self.plugins {
            current = plugin.transform(current).map_err(|e| {
                PipelineError::PluginFailed(format!("{}: {}", plugin.name(), e))
            })?;
        }
        Ok(current)
    }

    /// Process raw content from source text.
    pub fn process_source(
        &self,
        source: &str,
        content_type: ContentType,
    ) -> Result<ContentItem, PipelineError> {
        let item = ContentItem::from_source(source, content_type)?;
        self.process(item)
    }

    /// Number of plugins in the pipeline.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }
}

impl Default for ContentPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Render content summary as a debug-friendly string.
pub fn content_summary(item: &ContentItem) -> String {
    let mut out = String::new();
    let _ = write!(out, "slug: {}", item.slug);
    let _ = write!(out, ", type: {:?}", item.content_type);
    let _ = write!(out, ", words: {}", item.word_count);
    let _ = write!(out, ", reading_time: {} min", item.reading_time.minutes);
    if let Some(excerpt) = &item.excerpt {
        let _ = write!(out, ", excerpt: \"{}\"", excerpt);
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_frontmatter() {
        let input = "---\ntitle: Hello\nauthor: Alice\n---\nBody text here.";
        let (fm, body) = extract_frontmatter(input).unwrap();
        assert_eq!(fm.get("title"), Some("Hello"));
        assert_eq!(fm.get("author"), Some("Alice"));
        assert_eq!(body, "Body text here.");
    }

    #[test]
    fn test_extract_frontmatter_none() {
        let input = "No frontmatter here.";
        let (fm, body) = extract_frontmatter(input).unwrap();
        assert!(fm.is_empty());
        assert_eq!(body, input);
    }

    #[test]
    fn test_extract_frontmatter_unclosed() {
        let input = "---\ntitle: Hello\nNo closing";
        let result = extract_frontmatter(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_slug() {
        assert_eq!(generate_slug("Hello World"), "hello-world");
        assert_eq!(generate_slug("My  Great   Post"), "my-great-post");
        assert_eq!(generate_slug("Hello!@World"), "helloworld");
        assert_eq!(generate_slug("trailing- "), "trailing");
    }

    #[test]
    fn test_word_count_plain() {
        assert_eq!(word_count("Hello world"), 2);
        assert_eq!(word_count("One two three four five"), 5);
        assert_eq!(word_count(""), 0);
    }

    #[test]
    fn test_word_count_with_html() {
        assert_eq!(word_count("<p>Hello <b>world</b></p>"), 2);
        assert_eq!(word_count("<div>one two</div><p>three</p>"), 3);
    }

    #[test]
    fn test_reading_time_short() {
        let rt = reading_time("hello world");
        assert_eq!(rt.words, 2);
        assert_eq!(rt.minutes, 1);
    }

    #[test]
    fn test_reading_time_longer() {
        // 400 words at 200 WPM = 2 minutes
        let text = "word ".repeat(400);
        let rt = reading_time(&text);
        assert_eq!(rt.words, 400);
        assert_eq!(rt.minutes, 2);
    }

    #[test]
    fn test_reading_time_custom_wpm() {
        let text = "word ".repeat(100);
        let rt = reading_time_with_wpm(&text, 100);
        assert_eq!(rt.minutes, 1);
        assert_eq!(rt.wpm, 100);
    }

    #[test]
    fn test_extract_excerpt() {
        let text = "This is a longer piece of text that should be truncated to produce an excerpt for the content summary.";
        let excerpt = extract_excerpt(text, 30);
        assert!(excerpt.ends_with("..."));
        assert!(excerpt.len() <= 40); // 30 + some word boundary slack + "..."
    }

    #[test]
    fn test_extract_excerpt_short_text() {
        let text = "Short";
        let excerpt = extract_excerpt(text, 100);
        assert_eq!(excerpt, "Short");
    }

    #[test]
    fn test_extract_excerpt_by_separator() {
        let text = "Introduction paragraph.\n<!-- more -->\nRest of content.";
        let excerpt = extract_excerpt_by_separator(text, "<!-- more -->");
        assert_eq!(excerpt, Some("Introduction paragraph.".into()));
    }

    #[test]
    fn test_extract_excerpt_no_separator() {
        let text = "No separator here.";
        let excerpt = extract_excerpt_by_separator(text, "<!-- more -->");
        assert!(excerpt.is_none());
    }

    #[test]
    fn test_content_item_from_source() {
        let source = "---\ntitle: My Post\n---\nThis is the body of the post with some content.";
        let item = ContentItem::from_source(source, ContentType::Markdown).unwrap();
        assert_eq!(item.slug, "my-post");
        assert!(item.word_count > 0);
        assert!(item.excerpt.is_some());
    }

    #[test]
    fn test_content_item_metadata() {
        let source = "Hello world";
        let mut item = ContentItem::from_source(source, ContentType::PlainText).unwrap();
        item.set_metadata("key", "value");
        assert_eq!(item.get_metadata("key"), Some("value"));
        assert_eq!(item.get_metadata("missing"), None);
    }

    #[test]
    fn test_content_type_from_extension() {
        assert_eq!(ContentType::from_extension("md"), ContentType::Markdown);
        assert_eq!(ContentType::from_extension("html"), ContentType::Html);
        assert_eq!(ContentType::from_extension("txt"), ContentType::PlainText);
        assert_eq!(ContentType::from_extension("mdx"), ContentType::Markdown);
    }

    #[test]
    fn test_pipeline_with_plugins() {
        let mut pipeline = ContentPipeline::new();
        pipeline.add_plugin(Box::new(WordCountPlugin));
        pipeline.add_plugin(Box::new(UppercasePlugin));

        let source = "---\ntitle: Test\n---\nHello world";
        let item = ContentItem::from_source(source, ContentType::Markdown).unwrap();
        let result = pipeline.process(item).unwrap();
        assert_eq!(result.body, "HELLO WORLD");
        assert_eq!(result.get_metadata("word_count"), Some("2"));
    }

    #[test]
    fn test_pipeline_strip_html() {
        let mut pipeline = ContentPipeline::new();
        pipeline.add_plugin(Box::new(StripHtmlPlugin));

        let source = "<p>Hello</p>";
        let item = ContentItem::from_source(source, ContentType::Html).unwrap();
        let result = pipeline.process(item).unwrap();
        assert!(!result.body.contains('<'));
    }

    #[test]
    fn test_pipeline_slug_plugin() {
        let mut pipeline = ContentPipeline::new();
        pipeline.add_plugin(Box::new(SlugPlugin));

        let source = "---\ntitle: A New Title\n---\nBody";
        let item = ContentItem::from_source(source, ContentType::Markdown).unwrap();
        let result = pipeline.process(item).unwrap();
        assert_eq!(result.slug, "a-new-title");
    }

    #[test]
    fn test_pipeline_excerpt_plugin() {
        let mut pipeline = ContentPipeline::new();
        pipeline.add_plugin(Box::new(ExcerptPlugin::new(10)));

        let source = "This is a longer body text for testing purposes.";
        let item = ContentItem::from_source(source, ContentType::PlainText).unwrap();
        let result = pipeline.process(item).unwrap();
        assert!(result.excerpt.is_some());
    }

    #[test]
    fn test_empty_pipeline() {
        let pipeline = ContentPipeline::new();
        assert_eq!(pipeline.plugin_count(), 0);
        let source = "Hello";
        let result = pipeline.process_source(source, ContentType::PlainText).unwrap();
        assert_eq!(result.body, "Hello");
    }

    #[test]
    fn test_content_summary() {
        let source = "---\ntitle: Test\n---\nHello world in the body.";
        let item = ContentItem::from_source(source, ContentType::Markdown).unwrap();
        let summary = content_summary(&item);
        assert!(summary.contains("slug:"));
        assert!(summary.contains("words:"));
    }

    #[test]
    fn test_frontmatter_set_overwrite() {
        let mut fm = Frontmatter::new();
        fm.set("a", "1");
        fm.set("a", "2");
        assert_eq!(fm.get("a"), Some("2"));
        assert_eq!(fm.len(), 1);
    }

    #[test]
    fn test_reading_time_empty() {
        let rt = reading_time("");
        assert_eq!(rt.words, 0);
        assert_eq!(rt.minutes, 0);
    }
}
