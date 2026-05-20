//! Extended markdown processor — footnotes, definition lists, abbreviations,
//! math blocks, admonitions, table of contents, cross-references, and YAML
//! front matter metadata.
//!
//! Pure-Rust replacement for markdown-it plugins, remark-gfm, remark-math,
//! remark-frontmatter, and similar Node.js extensions.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Errors ──────────────────────────────────────────────────────

/// Errors that can arise during extended markdown processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkdownExtError {
    InvalidFrontmatter(String),
    InvalidFootnote(String),
    MissingReference(String),
}

impl fmt::Display for MarkdownExtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrontmatter(msg) => write!(f, "invalid frontmatter: {msg}"),
            Self::InvalidFootnote(msg) => write!(f, "invalid footnote: {msg}"),
            Self::MissingReference(msg) => write!(f, "missing reference: {msg}"),
        }
    }
}

impl std::error::Error for MarkdownExtError {}

// ── Front Matter ────────────────────────────────────────────────

/// Simple YAML front matter (key: value pairs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FrontMatter {
    pub fields: Vec<(String, String)>,
}

impl FrontMatter {
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
}

/// Extract YAML front matter delimited by `---`.
pub fn extract_frontmatter(input: &str) -> Result<(FrontMatter, &str), MarkdownExtError> {
    let trimmed = input.trim_start();
    if !trimmed.starts_with("---") {
        return Ok((FrontMatter::new(), input));
    }

    // Find closing ---
    let after_first = &trimmed[3..];
    let after_first = after_first.strip_prefix('\n').unwrap_or(after_first);
    let close_pos = after_first.find("\n---");
    match close_pos {
        None => Err(MarkdownExtError::InvalidFrontmatter(
            "unclosed front matter block".into(),
        )),
        Some(pos) => {
            let yaml_block = &after_first[..pos];
            let remainder_start = pos + 4; // "\n---"
            let remainder = if remainder_start < after_first.len() {
                let rest = &after_first[remainder_start..];
                rest.strip_prefix('\n').unwrap_or(rest)
            } else {
                ""
            };

            let mut fm = FrontMatter::new();
            for line in yaml_block.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Some((key, val)) = line.split_once(':') {
                    fm.set(key.trim(), val.trim());
                }
            }
            Ok((fm, remainder))
        }
    }
}

// ── Footnotes ───────────────────────────────────────────────────

/// A footnote definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Footnote {
    pub label: String,
    pub content: String,
}

/// Extract footnote definitions from text. Returns (cleaned text, footnotes).
///
/// Footnote definitions: `[^label]: content`
/// Footnote references: `[^label]`
pub fn extract_footnotes(input: &str) -> (String, Vec<Footnote>) {
    let mut footnotes = Vec::new();
    let mut output_lines = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[^") {
            if let Some(close) = trimmed.find("]:") {
                let label = trimmed[2..close].to_string();
                let content = trimmed[close + 2..].trim().to_string();
                footnotes.push(Footnote { label, content });
                continue;
            }
        }
        output_lines.push(line);
    }

    (output_lines.join("\n"), footnotes)
}

/// Render footnote references in text as HTML superscript links and append
/// a footnotes section.
pub fn render_footnotes(text: &str, footnotes: &[Footnote]) -> String {
    let mut result = text.to_string();

    // Replace references [^label] with superscript links
    for (i, fn_def) in footnotes.iter().enumerate() {
        let ref_marker = format!("[^{}]", fn_def.label);
        let sup_html = format!(
            "<sup class=\"footnote-ref\"><a href=\"#fn-{}\" id=\"fnref-{}\">{}</a></sup>",
            fn_def.label,
            fn_def.label,
            i + 1
        );
        result = result.replace(&ref_marker, &sup_html);
    }

    // Append footnote section
    if !footnotes.is_empty() {
        result.push_str("\n<section class=\"footnotes\">\n<ol>\n");
        for fn_def in footnotes {
            let _ = write!(
                result,
                "<li id=\"fn-{}\"><p>{} <a href=\"#fnref-{}\">↩</a></p></li>\n",
                fn_def.label, fn_def.content, fn_def.label
            );
        }
        result.push_str("</ol>\n</section>");
    }

    result
}

// ── Definition Lists ────────────────────────────────────────────

/// A definition list entry.
#[derive(Debug, Clone, PartialEq)]
pub struct DefinitionItem {
    pub term: String,
    pub definitions: Vec<String>,
}

/// Parse definition list blocks.
///
/// Format:
/// ```text
/// Term
/// : Definition 1
/// : Definition 2
/// ```
pub fn parse_definition_list(input: &str) -> Vec<DefinitionItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        if line.is_empty() {
            i += 1;
            continue;
        }

        // Check if next line(s) start with `: ` (definition)
        let mut definitions = Vec::new();
        let mut j = i + 1;
        while j < lines.len() {
            let def_line = lines[j].trim();
            if let Some(stripped) = def_line.strip_prefix(": ") {
                definitions.push(stripped.to_string());
                j += 1;
            } else {
                break;
            }
        }

        if !definitions.is_empty() {
            items.push(DefinitionItem {
                term: line.to_string(),
                definitions,
            });
            i = j;
        } else {
            i += 1;
        }
    }

    items
}

/// Render a definition list to HTML.
pub fn render_definition_list(items: &[DefinitionItem]) -> String {
    let mut out = String::from("<dl>\n");
    for item in items {
        let _ = write!(out, "  <dt>{}</dt>\n", item.term);
        for def in &item.definitions {
            let _ = write!(out, "  <dd>{}</dd>\n", def);
        }
    }
    out.push_str("</dl>");
    out
}

// ── Abbreviations ───────────────────────────────────────────────

/// An abbreviation definition.
#[derive(Debug, Clone, PartialEq)]
pub struct Abbreviation {
    pub abbr: String,
    pub full: String,
}

/// Extract abbreviation definitions from text.
///
/// Format: `*[ABBR]: Full text`
pub fn extract_abbreviations(input: &str) -> (String, Vec<Abbreviation>) {
    let mut abbreviations = Vec::new();
    let mut output_lines = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("*[") {
            if let Some(close) = trimmed.find("]:") {
                let abbr = trimmed[2..close].to_string();
                let full = trimmed[close + 2..].trim().to_string();
                abbreviations.push(Abbreviation { abbr, full });
                continue;
            }
        }
        output_lines.push(line);
    }

    (output_lines.join("\n"), abbreviations)
}

/// Wrap abbreviations in `<abbr>` tags.
pub fn apply_abbreviations(text: &str, abbreviations: &[Abbreviation]) -> String {
    let mut result = text.to_string();
    for ab in abbreviations {
        let replacement = format!("<abbr title=\"{}\">{}</abbr>", ab.full, ab.abbr);
        result = result.replace(&ab.abbr, &replacement);
    }
    result
}

// ── Math Blocks ─────────────────────────────────────────────────

/// A math block extracted from markdown.
#[derive(Debug, Clone, PartialEq)]
pub struct MathBlock {
    /// Whether this is display math ($$..$$) vs inline ($...$).
    pub display: bool,
    /// The raw math content.
    pub content: String,
}

/// Extract and replace math blocks with HTML containers.
pub fn process_math(input: &str) -> (String, Vec<MathBlock>) {
    let mut blocks = Vec::new();
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Display math: $$...$$
        if i + 1 < len && chars[i] == '$' && chars[i + 1] == '$' {
            let start = i + 2;
            let mut end = start;
            while end + 1 < len && !(chars[end] == '$' && chars[end + 1] == '$') {
                end += 1;
            }
            if end + 1 < len {
                let content: String = chars[start..end].iter().collect();
                let idx = blocks.len();
                blocks.push(MathBlock {
                    display: true,
                    content: content.clone(),
                });
                let _ = write!(
                    result,
                    "<div class=\"math-display\" data-math-index=\"{}\">\\[{}\\]</div>",
                    idx, content
                );
                i = end + 2;
                continue;
            }
        }

        // Inline math: $...$
        if chars[i] == '$' {
            let start = i + 1;
            let mut end = start;
            while end < len && chars[end] != '$' {
                end += 1;
            }
            if end < len && end > start {
                let content: String = chars[start..end].iter().collect();
                let idx = blocks.len();
                blocks.push(MathBlock {
                    display: false,
                    content: content.clone(),
                });
                let _ = write!(
                    result,
                    "<span class=\"math-inline\" data-math-index=\"{}\">\\({}\\)</span>",
                    idx, content
                );
                i = end + 1;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    (result, blocks)
}

// ── Admonitions ─────────────────────────────────────────────────

/// An admonition (callout) type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmonitionKind {
    Note,
    Tip,
    Warning,
    Danger,
    Info,
    Custom,
}

impl AdmonitionKind {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "note" => Self::Note,
            "tip" => Self::Tip,
            "warning" => Self::Warning,
            "danger" => Self::Danger,
            "info" => Self::Info,
            _ => Self::Custom,
        }
    }

    fn css_class(&self) -> &str {
        match self {
            Self::Note => "admonition-note",
            Self::Tip => "admonition-tip",
            Self::Warning => "admonition-warning",
            Self::Danger => "admonition-danger",
            Self::Info => "admonition-info",
            Self::Custom => "admonition-custom",
        }
    }
}

/// A parsed admonition block.
#[derive(Debug, Clone, PartialEq)]
pub struct Admonition {
    pub kind: AdmonitionKind,
    pub title: Option<String>,
    pub content: String,
}

/// Parse admonition blocks from text.
///
/// Format:
/// ```text
/// :::note Title
/// Content here
/// :::
/// ```
pub fn parse_admonitions(input: &str) -> (String, Vec<Admonition>) {
    let mut admonitions = Vec::new();
    let mut result = String::new();
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        if let Some(rest) = trimmed.strip_prefix(":::") {
            let rest = rest.trim();
            if rest.is_empty() {
                // Stray closing fence
                result.push_str(lines[i]);
                result.push('\n');
                i += 1;
                continue;
            }

            // Parse kind and optional title
            let (kind_str, title) = if let Some(space_pos) = rest.find(' ') {
                (
                    &rest[..space_pos],
                    Some(rest[space_pos + 1..].trim().to_string()),
                )
            } else {
                (rest, None)
            };

            let kind = AdmonitionKind::from_str(kind_str);

            // Collect content until closing :::
            let mut content_lines = Vec::new();
            i += 1;
            while i < lines.len() {
                let inner = lines[i].trim();
                if inner == ":::" {
                    i += 1;
                    break;
                }
                content_lines.push(lines[i]);
                i += 1;
            }

            let content = content_lines.join("\n");
            let idx = admonitions.len();
            let title_ref = title.clone();
            admonitions.push(Admonition {
                kind,
                title,
                content: content.clone(),
            });

            let _ = write!(
                result,
                "<div class=\"admonition {}\" data-admonition-index=\"{}\">",
                kind.css_class(),
                idx
            );
            if let Some(t) = title_ref {
                let _ = write!(result, "<p class=\"admonition-title\">{}</p>", t);
            }
            let _ = write!(result, "<p>{}</p></div>\n", content);
        } else {
            result.push_str(lines[i]);
            result.push('\n');
            i += 1;
        }
    }

    (result, admonitions)
}

// ── Table of Contents ───────────────────────────────────────────

/// A heading entry in the table of contents.
#[derive(Debug, Clone, PartialEq)]
pub struct TocEntry {
    pub level: u8,
    pub text: String,
    pub slug: String,
}

/// Generate a slug from heading text.
fn heading_slug(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c
            } else if c == ' ' || c == '-' {
                '-'
            } else {
                '_'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Extract headings from markdown text and produce a table of contents.
pub fn generate_toc(input: &str) -> Vec<TocEntry> {
    let mut entries = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') {
            let hashes = trimmed.chars().take_while(|c| *c == '#').count();
            if hashes >= 1 && hashes <= 6 {
                let text = trimmed[hashes..].trim().to_string();
                if !text.is_empty() {
                    let slug = heading_slug(&text);
                    entries.push(TocEntry {
                        level: hashes as u8,
                        text,
                        slug,
                    });
                }
            }
        }
    }

    entries
}

/// Render a table of contents to HTML.
pub fn render_toc(entries: &[TocEntry]) -> String {
    if entries.is_empty() {
        return String::new();
    }

    let mut out = String::from("<nav class=\"toc\">\n<ul>\n");
    for entry in entries {
        let indent = "  ".repeat(entry.level as usize);
        let _ = write!(
            out,
            "{}<li><a href=\"#{}\">{}</a></li>\n",
            indent, entry.slug, entry.text
        );
    }
    out.push_str("</ul>\n</nav>");
    out
}

// ── Cross-references ────────────────────────────────────────────

/// A cross-reference target.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossRef {
    pub label: String,
    pub target_slug: String,
    pub display_text: Option<String>,
}

/// Parse cross-references in the format `{ref:label}` or `{ref:label|display text}`.
pub fn parse_cross_references(input: &str) -> (String, Vec<CrossRef>) {
    let mut refs = Vec::new();
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        if chars[i] == '{' {
            // Look for {ref:...}
            let remaining: String = chars[i..].iter().collect();
            if remaining.starts_with("{ref:") {
                if let Some(close) = remaining.find('}') {
                    let inner = &remaining[5..close];
                    let (label, display) = if let Some(pipe) = inner.find('|') {
                        (
                            inner[..pipe].to_string(),
                            Some(inner[pipe + 1..].to_string()),
                        )
                    } else {
                        (inner.to_string(), None)
                    };

                    let slug = heading_slug(&label);
                    let link_text = display.clone().unwrap_or_else(|| label.clone());
                    let _ = write!(
                        result,
                        "<a href=\"#{}\" class=\"cross-ref\">{}</a>",
                        slug, link_text
                    );
                    refs.push(CrossRef {
                        label,
                        target_slug: slug,
                        display_text: display,
                    });
                    i += close + 1;
                    continue;
                }
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    (result, refs)
}

// ── Extended Processor ──────────────────────────────────────────

/// Configuration for the extended markdown processor.
#[derive(Debug, Clone)]
pub struct ExtendedMarkdownConfig {
    pub enable_footnotes: bool,
    pub enable_definition_lists: bool,
    pub enable_abbreviations: bool,
    pub enable_math: bool,
    pub enable_admonitions: bool,
    pub enable_toc: bool,
    pub enable_cross_refs: bool,
    pub enable_frontmatter: bool,
}

impl Default for ExtendedMarkdownConfig {
    fn default() -> Self {
        Self {
            enable_footnotes: true,
            enable_definition_lists: true,
            enable_abbreviations: true,
            enable_math: true,
            enable_admonitions: true,
            enable_toc: true,
            enable_cross_refs: true,
            enable_frontmatter: true,
        }
    }
}

/// Result of processing extended markdown.
#[derive(Debug, Clone)]
pub struct ExtendedMarkdownResult {
    pub html: String,
    pub frontmatter: FrontMatter,
    pub toc: Vec<TocEntry>,
    pub footnotes: Vec<Footnote>,
    pub math_blocks: Vec<MathBlock>,
    pub abbreviations: Vec<Abbreviation>,
    pub admonitions: Vec<Admonition>,
    pub cross_refs: Vec<CrossRef>,
}

/// Process extended markdown with all extensions.
pub fn process_extended_markdown(
    input: &str,
    config: &ExtendedMarkdownConfig,
) -> Result<ExtendedMarkdownResult, MarkdownExtError> {
    let mut text = input.to_string();
    let mut frontmatter = FrontMatter::new();
    let mut toc = Vec::new();
    let mut footnotes_out = Vec::new();
    let mut math_blocks = Vec::new();
    let mut abbreviations_out = Vec::new();
    let mut admonitions_out = Vec::new();
    let mut cross_refs = Vec::new();

    // 1. Front matter
    if config.enable_frontmatter {
        let (fm, rest) = extract_frontmatter(&text)?;
        frontmatter = fm;
        text = rest.to_string();
    }

    // 2. Abbreviations (extract before rendering)
    if config.enable_abbreviations {
        let (cleaned, abbrs) = extract_abbreviations(&text);
        text = cleaned;
        abbreviations_out = abbrs;
    }

    // 3. Footnotes
    if config.enable_footnotes {
        let (cleaned, fns) = extract_footnotes(&text);
        text = render_footnotes(&cleaned, &fns);
        footnotes_out = fns;
    }

    // 4. Math
    if config.enable_math {
        let (processed, blocks) = process_math(&text);
        text = processed;
        math_blocks = blocks;
    }

    // 5. Admonitions
    if config.enable_admonitions {
        let (processed, adm) = parse_admonitions(&text);
        text = processed;
        admonitions_out = adm;
    }

    // 6. TOC (extract from remaining headings)
    if config.enable_toc {
        toc = generate_toc(&text);
    }

    // 7. Cross-references
    if config.enable_cross_refs {
        let (processed, refs) = parse_cross_references(&text);
        text = processed;
        cross_refs = refs;
    }

    // 8. Apply abbreviations last
    if config.enable_abbreviations {
        text = apply_abbreviations(&text, &abbreviations_out);
    }

    Ok(ExtendedMarkdownResult {
        html: text,
        frontmatter,
        toc,
        footnotes: footnotes_out,
        math_blocks,
        abbreviations: abbreviations_out,
        admonitions: admonitions_out,
        cross_refs,
    })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_frontmatter_basic() {
        let input = "---\ntitle: Hello World\nauthor: Alice\n---\nContent here";
        let (fm, rest) = extract_frontmatter(input).unwrap();
        assert_eq!(fm.get("title"), Some("Hello World"));
        assert_eq!(fm.get("author"), Some("Alice"));
        assert_eq!(rest, "Content here");
    }

    #[test]
    fn test_extract_frontmatter_no_frontmatter() {
        let input = "Just some content";
        let (fm, rest) = extract_frontmatter(input).unwrap();
        assert!(fm.fields.is_empty());
        assert_eq!(rest, input);
    }

    #[test]
    fn test_extract_frontmatter_unclosed() {
        let input = "---\ntitle: Hello\nNo closing";
        let result = extract_frontmatter(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_frontmatter_set_and_get() {
        let mut fm = FrontMatter::new();
        fm.set("key", "value");
        assert_eq!(fm.get("key"), Some("value"));
        fm.set("key", "new_value");
        assert_eq!(fm.get("key"), Some("new_value"));
        assert_eq!(fm.fields.len(), 1);
    }

    #[test]
    fn test_extract_footnotes() {
        let input = "Hello[^1] world[^2].\n\n[^1]: First note\n[^2]: Second note";
        let (cleaned, fns) = extract_footnotes(input);
        assert_eq!(fns.len(), 2);
        assert_eq!(fns[0].label, "1");
        assert_eq!(fns[0].content, "First note");
        assert!(!cleaned.contains("[^1]:"));
    }

    #[test]
    fn test_render_footnotes() {
        let footnotes = vec![Footnote {
            label: "1".into(),
            content: "A footnote".into(),
        }];
        let text = "Text[^1] here";
        let rendered = render_footnotes(text, &footnotes);
        assert!(rendered.contains("footnote-ref"));
        assert!(rendered.contains("fn-1"));
        assert!(rendered.contains("A footnote"));
    }

    #[test]
    fn test_definition_list_parse() {
        let input = "Rust\n: A systems programming language\n: Fast and safe\n\nPython\n: A scripting language";
        let items = parse_definition_list(input);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].term, "Rust");
        assert_eq!(items[0].definitions.len(), 2);
        assert_eq!(items[1].term, "Python");
    }

    #[test]
    fn test_definition_list_render() {
        let items = vec![DefinitionItem {
            term: "Term".into(),
            definitions: vec!["Def1".into(), "Def2".into()],
        }];
        let html = render_definition_list(&items);
        assert!(html.contains("<dl>"));
        assert!(html.contains("<dt>Term</dt>"));
        assert!(html.contains("<dd>Def1</dd>"));
        assert!(html.contains("<dd>Def2</dd>"));
    }

    #[test]
    fn test_extract_abbreviations() {
        let input = "HTML is great.\n*[HTML]: Hyper Text Markup Language";
        let (cleaned, abbrs) = extract_abbreviations(input);
        assert_eq!(abbrs.len(), 1);
        assert_eq!(abbrs[0].abbr, "HTML");
        assert_eq!(abbrs[0].full, "Hyper Text Markup Language");
        assert!(!cleaned.contains("*[HTML]"));
    }

    #[test]
    fn test_apply_abbreviations() {
        let text = "The HTML spec is great.";
        let abbrs = vec![Abbreviation {
            abbr: "HTML".into(),
            full: "Hyper Text Markup Language".into(),
        }];
        let result = apply_abbreviations(text, &abbrs);
        assert!(result.contains("<abbr title=\"Hyper Text Markup Language\">HTML</abbr>"));
    }

    #[test]
    fn test_math_inline() {
        let input = "Euler's $e^{i\\pi} + 1 = 0$ is beautiful.";
        let (html, blocks) = process_math(input);
        assert_eq!(blocks.len(), 1);
        assert!(!blocks[0].display);
        assert!(html.contains("math-inline"));
    }

    #[test]
    fn test_math_display() {
        let input = "The equation:\n$$\\sum_{i=0}^{n} i$$\nis well known.";
        let (html, blocks) = process_math(input);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].display);
        assert!(html.contains("math-display"));
    }

    #[test]
    fn test_admonition_parse() {
        let input = ":::note Important\nDon't forget this.\n:::\n\nRegular text.";
        let (html, adm) = parse_admonitions(input);
        assert_eq!(adm.len(), 1);
        assert_eq!(adm[0].kind, AdmonitionKind::Note);
        assert_eq!(adm[0].title.as_deref(), Some("Important"));
        assert!(html.contains("admonition-note"));
    }

    #[test]
    fn test_admonition_kinds() {
        assert_eq!(AdmonitionKind::from_str("note"), AdmonitionKind::Note);
        assert_eq!(AdmonitionKind::from_str("tip"), AdmonitionKind::Tip);
        assert_eq!(AdmonitionKind::from_str("WARNING"), AdmonitionKind::Warning);
        assert_eq!(AdmonitionKind::from_str("danger"), AdmonitionKind::Danger);
        assert_eq!(AdmonitionKind::from_str("info"), AdmonitionKind::Info);
        assert_eq!(AdmonitionKind::from_str("other"), AdmonitionKind::Custom);
    }

    #[test]
    fn test_generate_toc() {
        let input = "# Introduction\n\nSome text.\n\n## Background\n\n### Details\n\n## Conclusion";
        let toc = generate_toc(input);
        assert_eq!(toc.len(), 4);
        assert_eq!(toc[0].level, 1);
        assert_eq!(toc[0].text, "Introduction");
        assert_eq!(toc[1].level, 2);
        assert_eq!(toc[2].level, 3);
    }

    #[test]
    fn test_render_toc() {
        let entries = vec![
            TocEntry {
                level: 1,
                text: "Intro".into(),
                slug: "intro".into(),
            },
            TocEntry {
                level: 2,
                text: "Sub".into(),
                slug: "sub".into(),
            },
        ];
        let html = render_toc(&entries);
        assert!(html.contains("<nav class=\"toc\">"));
        assert!(html.contains("href=\"#intro\""));
        assert!(html.contains("href=\"#sub\""));
    }

    #[test]
    fn test_toc_empty() {
        let html = render_toc(&[]);
        assert!(html.is_empty());
    }

    #[test]
    fn test_cross_reference_basic() {
        let input = "See {ref:intro} for details.";
        let (html, refs) = parse_cross_references(input);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].label, "intro");
        assert!(html.contains("cross-ref"));
    }

    #[test]
    fn test_cross_reference_with_display_text() {
        let input = "See {ref:intro|the introduction} for details.";
        let (html, refs) = parse_cross_references(input);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].display_text.as_deref(), Some("the introduction"));
        assert!(html.contains("the introduction</a>"));
    }

    #[test]
    fn test_heading_slug() {
        assert_eq!(heading_slug("Hello World"), "hello-world");
        assert_eq!(heading_slug("API Reference"), "api-reference");
        assert_eq!(heading_slug("It's a Test!"), "it_s-a-test_");
    }

    #[test]
    fn test_full_pipeline() {
        let input = "---\ntitle: Test\n---\n\n# Heading\n\n:::note Note\nContent\n:::\n\nSome $math$ here.\n\n[^1]: A footnote";
        let config = ExtendedMarkdownConfig::default();
        let result = process_extended_markdown(input, &config).unwrap();
        assert_eq!(result.frontmatter.get("title"), Some("Test"));
        assert!(!result.toc.is_empty());
        assert_eq!(result.math_blocks.len(), 1);
    }

    #[test]
    fn test_pipeline_features_disabled() {
        let config = ExtendedMarkdownConfig {
            enable_footnotes: false,
            enable_definition_lists: false,
            enable_abbreviations: false,
            enable_math: false,
            enable_admonitions: false,
            enable_toc: false,
            enable_cross_refs: false,
            enable_frontmatter: false,
        };
        let input = "# Heading\nContent";
        let result = process_extended_markdown(input, &config).unwrap();
        assert!(result.toc.is_empty());
        assert!(result.frontmatter.fields.is_empty());
    }

    #[test]
    fn test_admonition_without_title() {
        let input = ":::warning\nBe careful!\n:::";
        let (html, adm) = parse_admonitions(input);
        assert_eq!(adm.len(), 1);
        assert_eq!(adm[0].kind, AdmonitionKind::Warning);
        assert!(adm[0].title.is_none());
        assert!(html.contains("admonition-warning"));
    }

    #[test]
    fn test_multiple_math_blocks() {
        let input = "Inline $a$ and display $$b$$ and another $c$.";
        let (_, blocks) = process_math(input);
        assert_eq!(blocks.len(), 3);
        assert!(!blocks[0].display);
        assert!(blocks[1].display);
        assert!(!blocks[2].display);
    }

    #[test]
    fn test_frontmatter_missing_key() {
        let fm = FrontMatter::new();
        assert_eq!(fm.get("nonexistent"), None);
    }

    #[test]
    fn test_definition_list_empty_input() {
        let items = parse_definition_list("");
        assert!(items.is_empty());
    }
}
