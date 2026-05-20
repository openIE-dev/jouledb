//! Documentation generator.
//!
//! Parses doc comments, generates module/function/struct documentation,
//! cross-references, table of contents, markdown output, code example
//! extraction, and API surface enumeration. Pure Rust — no external doc
//! generator dependencies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from doc generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocGenError {
    /// Failed to parse doc comments.
    ParseError(String),
    /// Item not found for cross-reference.
    ItemNotFound(String),
    /// Empty input.
    EmptyInput,
}

impl fmt::Display for DocGenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "doc parse error: {msg}"),
            Self::ItemNotFound(name) => write!(f, "item not found: {name}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for DocGenError {}

// ── Visibility ──────────────────────────────────────────────────

/// Visibility of a documented item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Crate,
    Private,
}

impl fmt::Display for Visibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => write!(f, "pub"),
            Self::Crate => write!(f, "pub(crate)"),
            Self::Private => write!(f, "(private)"),
        }
    }
}

// ── Item Kind ───────────────────────────────────────────────────

/// Kind of documented item.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ItemKind {
    Module,
    Struct,
    Enum,
    Trait,
    Function,
    Method,
    Constant,
    TypeAlias,
    Macro,
    Impl,
}

impl fmt::Display for ItemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module => write!(f, "Module"),
            Self::Struct => write!(f, "Struct"),
            Self::Enum => write!(f, "Enum"),
            Self::Trait => write!(f, "Trait"),
            Self::Function => write!(f, "Function"),
            Self::Method => write!(f, "Method"),
            Self::Constant => write!(f, "Constant"),
            Self::TypeAlias => write!(f, "Type Alias"),
            Self::Macro => write!(f, "Macro"),
            Self::Impl => write!(f, "Impl"),
        }
    }
}

// ── Parameter Doc ───────────────────────────────────────────────

/// Documentation for a function/method parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDoc {
    pub name: String,
    pub type_name: String,
    pub description: String,
}

// ── Code Example ────────────────────────────────────────────────

/// A code example extracted from documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// The code text.
    pub code: String,
    /// Language hint (e.g., "rust", "text").
    pub language: String,
    /// Whether this example should be tested.
    pub testable: bool,
    /// Caption/description.
    pub caption: Option<String>,
}

// ── Doc Comment ─────────────────────────────────────────────────

/// Parsed documentation comment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocComment {
    /// Summary line (first paragraph).
    pub summary: String,
    /// Full description (all paragraphs after summary).
    pub description: String,
    /// Parameter documentation (from `# Arguments` or `@param`).
    pub params: Vec<ParamDoc>,
    /// Return value documentation.
    pub returns: Option<String>,
    /// Code examples.
    pub examples: Vec<CodeExample>,
    /// Panics section.
    pub panics: Option<String>,
    /// Errors section.
    pub errors: Option<String>,
    /// Safety section (for unsafe fns).
    pub safety: Option<String>,
    /// See-also cross-references.
    pub see_also: Vec<String>,
    /// Tags/annotations.
    pub tags: Vec<String>,
}

/// Parse a doc comment block into structured documentation.
///
/// Expects lines from `///` or `//!` comments, with the prefix stripped.
pub fn parse_doc_comment(lines: &[&str]) -> DocComment {
    let mut doc = DocComment::default();
    let mut current_section: Option<String> = None;
    let mut section_content = String::new();
    let mut in_code_block = false;
    let mut code_block_lang = String::new();
    let mut code_block_content = String::new();
    let mut summary_done = false;
    let mut description_lines = Vec::new();

    for line in lines {
        let trimmed = line.trim();

        // Handle code blocks.
        if trimmed.starts_with("```") {
            if in_code_block {
                // End code block.
                doc.examples.push(CodeExample {
                    code: code_block_content.trim_end().to_string(),
                    language: if code_block_lang.is_empty() {
                        "text".to_string()
                    } else {
                        code_block_lang.clone()
                    },
                    testable: code_block_lang == "rust"
                        || code_block_lang.is_empty()
                        || !code_block_lang.contains("no_run"),
                    caption: None,
                });
                code_block_content.clear();
                code_block_lang.clear();
                in_code_block = false;
            } else {
                // Start code block.
                in_code_block = true;
                code_block_lang = trimmed[3..].trim().to_string();
                // Strip modifiers like "rust,no_run".
                if let Some(comma) = code_block_lang.find(',') {
                    code_block_lang = code_block_lang[..comma].to_string();
                }
            }
            continue;
        }

        if in_code_block {
            code_block_content.push_str(line);
            code_block_content.push('\n');
            continue;
        }

        // Handle section headers (# Arguments, # Returns, etc).
        if trimmed.starts_with("# ") {
            // Save previous section.
            if let Some(sec) = current_section.take() {
                apply_section(&mut doc, &sec, &section_content);
                section_content.clear();
            }
            current_section = Some(trimmed[2..].trim().to_string());
            continue;
        }

        // Handle @-style tags.
        if trimmed.starts_with("@param ") {
            let rest = &trimmed[7..];
            if let Some(space) = rest.find(' ') {
                let name = &rest[..space];
                let desc = &rest[space + 1..];
                doc.params.push(ParamDoc {
                    name: name.to_string(),
                    type_name: String::new(),
                    description: desc.to_string(),
                });
            }
            continue;
        }
        if trimmed.starts_with("@returns ") || trimmed.starts_with("@return ") {
            let at_len = if trimmed.starts_with("@returns ") {
                9
            } else {
                8
            };
            doc.returns = Some(trimmed[at_len..].to_string());
            continue;
        }
        if trimmed.starts_with("@see ") {
            doc.see_also.push(trimmed[5..].trim().to_string());
            continue;
        }
        if trimmed.starts_with("@tag ") {
            doc.tags.push(trimmed[5..].trim().to_string());
            continue;
        }

        if let Some(_) = &current_section {
            section_content.push_str(trimmed);
            section_content.push('\n');
            continue;
        }

        // Summary and description.
        if !summary_done {
            if trimmed.is_empty() {
                if !doc.summary.is_empty() {
                    summary_done = true;
                }
            } else {
                if !doc.summary.is_empty() {
                    doc.summary.push(' ');
                }
                doc.summary.push_str(trimmed);
            }
        } else {
            description_lines.push(trimmed.to_string());
        }
    }

    // Save last section.
    if let Some(sec) = current_section {
        apply_section(&mut doc, &sec, &section_content);
    }

    doc.description = description_lines.join("\n").trim().to_string();

    doc
}

fn apply_section(doc: &mut DocComment, section: &str, content: &str) {
    let content_trimmed = content.trim();
    match section.to_lowercase().as_str() {
        "arguments" | "params" | "parameters" => {
            // Parse `* `name` — description` lines.
            for line in content_trimmed.lines() {
                let trimmed = line.trim().trim_start_matches('*').trim_start_matches('-').trim();
                if let Some(rest) = trimmed.strip_prefix('`') {
                    if let Some(end_tick) = rest.find('`') {
                        let name = &rest[..end_tick];
                        let desc = rest[end_tick + 1..]
                            .trim_start_matches(|c: char| c == ' ' || c == '-' || c == ':')
                            .trim();
                        doc.params.push(ParamDoc {
                            name: name.to_string(),
                            type_name: String::new(),
                            description: desc.to_string(),
                        });
                    }
                }
            }
        }
        "returns" | "return" | "return value" => {
            doc.returns = Some(content_trimmed.to_string());
        }
        "panics" => {
            doc.panics = Some(content_trimmed.to_string());
        }
        "errors" => {
            doc.errors = Some(content_trimmed.to_string());
        }
        "safety" => {
            doc.safety = Some(content_trimmed.to_string());
        }
        "examples" | "example" => {
            // Examples section — examples are already captured by code blocks.
        }
        "see also" => {
            for line in content_trimmed.lines() {
                let item = line.trim().trim_start_matches('*').trim_start_matches('-').trim();
                if !item.is_empty() {
                    doc.see_also.push(item.to_string());
                }
            }
        }
        _ => {
            // Unknown section, ignore.
        }
    }
}

// ── Documented Item ─────────────────────────────────────────────

/// A documented item (module, struct, function, etc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocItem {
    /// Fully qualified path (e.g., "crate::module::Struct").
    pub path: String,
    /// Short name.
    pub name: String,
    /// Kind of item.
    pub kind: ItemKind,
    /// Visibility.
    pub visibility: Visibility,
    /// Documentation.
    pub doc: DocComment,
    /// Signature (for functions/methods).
    pub signature: Option<String>,
    /// Child items (for modules, structs, enums).
    pub children: Vec<String>,
    /// Source file path.
    pub source_file: Option<String>,
    /// Source line number.
    pub source_line: Option<usize>,
}

impl DocItem {
    /// Create a new documented item.
    pub fn new(path: &str, name: &str, kind: ItemKind) -> Self {
        Self {
            path: path.to_string(),
            name: name.to_string(),
            kind,
            visibility: Visibility::Public,
            doc: DocComment::default(),
            signature: None,
            children: Vec::new(),
            source_file: None,
            source_line: None,
        }
    }
}

// ── Cross-Reference ─────────────────────────────────────────────

/// A cross-reference link.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRef {
    /// The text that triggered the cross-reference (e.g., "[`Foo`]").
    pub text: String,
    /// Resolved target path.
    pub target: String,
    /// Whether the target was successfully resolved.
    pub resolved: bool,
}

/// Resolve cross-references in documentation text.
pub fn resolve_cross_refs(text: &str, items: &HashMap<String, DocItem>) -> Vec<CrossRef> {
    let mut refs = Vec::new();
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();

    while i < chars.len() {
        // Look for [`...`] patterns.
        if i + 2 < chars.len() && chars[i] == '[' && chars[i + 1] == '`' {
            let start = i + 2;
            if let Some(end) = text[start..].find("`]") {
                let ref_text = &text[start..start + end];
                let resolved = items.contains_key(ref_text)
                    || items
                        .values()
                        .any(|item| item.name == ref_text);
                refs.push(CrossRef {
                    text: format!("[`{ref_text}`]"),
                    target: ref_text.to_string(),
                    resolved,
                });
                i = start + end + 2;
                continue;
            }
        }
        i += 1;
    }

    refs
}

// ── Table of Contents ───────────────────────────────────────────

/// A table of contents entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    pub title: String,
    pub anchor: String,
    pub level: usize,
    pub children: Vec<TocEntry>,
}

/// Build a table of contents from documented items.
pub fn build_toc(items: &[DocItem]) -> Vec<TocEntry> {
    let mut toc = Vec::new();

    // Group by kind for the top-level TOC.
    let kinds = [
        ItemKind::Module,
        ItemKind::Struct,
        ItemKind::Enum,
        ItemKind::Trait,
        ItemKind::Function,
        ItemKind::Constant,
        ItemKind::TypeAlias,
        ItemKind::Macro,
    ];

    for kind in &kinds {
        let matching: Vec<&DocItem> = items.iter().filter(|i| &i.kind == kind).collect();
        if matching.is_empty() {
            continue;
        }
        let section_title = format!("{kind}s");
        let anchor = section_title.to_lowercase().replace(' ', "-");
        let children: Vec<TocEntry> = matching
            .iter()
            .map(|item| TocEntry {
                title: item.name.clone(),
                anchor: item.name.to_lowercase().replace("::", "-"),
                level: 2,
                children: Vec::new(),
            })
            .collect();
        toc.push(TocEntry {
            title: section_title,
            anchor,
            level: 1,
            children,
        });
    }

    toc
}

// ── API Surface ─────────────────────────────────────────────────

/// Summary of the public API surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSurface {
    pub modules: usize,
    pub structs: usize,
    pub enums: usize,
    pub traits: usize,
    pub functions: usize,
    pub methods: usize,
    pub constants: usize,
    pub type_aliases: usize,
    pub total: usize,
    pub documented_pct: f64,
}

/// Enumerate the API surface from a set of doc items.
pub fn enumerate_api(items: &[DocItem]) -> ApiSurface {
    let public: Vec<&DocItem> = items
        .iter()
        .filter(|i| i.visibility == Visibility::Public)
        .collect();

    let count = |kind: &ItemKind| public.iter().filter(|i| &i.kind == kind).count();

    let total = public.len();
    let documented = public
        .iter()
        .filter(|i| !i.doc.summary.is_empty())
        .count();

    ApiSurface {
        modules: count(&ItemKind::Module),
        structs: count(&ItemKind::Struct),
        enums: count(&ItemKind::Enum),
        traits: count(&ItemKind::Trait),
        functions: count(&ItemKind::Function),
        methods: count(&ItemKind::Method),
        constants: count(&ItemKind::Constant),
        type_aliases: count(&ItemKind::TypeAlias),
        total,
        documented_pct: if total == 0 {
            100.0
        } else {
            documented as f64 / total as f64 * 100.0
        },
    }
}

// ── Markdown Output ─────────────────────────────────────────────

/// Generate markdown documentation for a set of items.
pub fn generate_markdown(items: &[DocItem], title: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {title}\n\n"));

    // Table of contents.
    let toc = build_toc(items);
    if !toc.is_empty() {
        out.push_str("## Table of Contents\n\n");
        for entry in &toc {
            out.push_str(&format!("- [{}](#{})\n", entry.title, entry.anchor));
            for child in &entry.children {
                out.push_str(&format!("  - [{}](#{})\n", child.title, child.anchor));
            }
        }
        out.push('\n');
    }

    // Group items by kind.
    let kinds = [
        ItemKind::Module,
        ItemKind::Struct,
        ItemKind::Enum,
        ItemKind::Trait,
        ItemKind::Function,
        ItemKind::Constant,
        ItemKind::TypeAlias,
    ];

    for kind in &kinds {
        let matching: Vec<&DocItem> = items
            .iter()
            .filter(|i| &i.kind == kind && i.visibility == Visibility::Public)
            .collect();
        if matching.is_empty() {
            continue;
        }

        out.push_str(&format!("## {kind}s\n\n"));

        for item in &matching {
            out.push_str(&format!("### {}\n\n", item.name));

            if let Some(sig) = &item.signature {
                out.push_str(&format!("```rust\n{sig}\n```\n\n"));
            }

            if !item.doc.summary.is_empty() {
                out.push_str(&format!("{}\n\n", item.doc.summary));
            }

            if !item.doc.description.is_empty() {
                out.push_str(&format!("{}\n\n", item.doc.description));
            }

            if !item.doc.params.is_empty() {
                out.push_str("**Parameters:**\n\n");
                for p in &item.doc.params {
                    out.push_str(&format!("- `{}` — {}\n", p.name, p.description));
                }
                out.push('\n');
            }

            if let Some(ret) = &item.doc.returns {
                out.push_str(&format!("**Returns:** {ret}\n\n"));
            }

            if !item.doc.examples.is_empty() {
                out.push_str("**Examples:**\n\n");
                for ex in &item.doc.examples {
                    out.push_str(&format!("```{}\n{}\n```\n\n", ex.language, ex.code));
                }
            }

            if let Some(panics) = &item.doc.panics {
                out.push_str(&format!("**Panics:** {panics}\n\n"));
            }

            if !item.doc.see_also.is_empty() {
                out.push_str("**See also:** ");
                out.push_str(&item.doc.see_also.join(", "));
                out.push_str("\n\n");
            }
        }
    }

    out
}

// ── Source Parser (simplified) ──────────────────────────────────

/// Extract doc items from a simplified Rust-like source.
///
/// This is a simplified parser that recognizes:
/// - `/// comment` doc comments
/// - `//! module-level` doc comments
/// - `pub fn name(...)` functions
/// - `pub struct Name` structs
/// - `pub enum Name` enums
/// - `pub trait Name` traits
/// - `pub const NAME` constants
/// - `pub type Name` type aliases
pub fn extract_doc_items(source: &str, module_path: &str) -> Vec<DocItem> {
    let mut items = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut doc_lines: Vec<&str> = Vec::new();
    let mut module_doc_lines: Vec<&str> = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Module-level doc comments.
        if let Some(rest) = trimmed.strip_prefix("//!") {
            module_doc_lines.push(rest.trim_start_matches(' '));
            continue;
        }

        // Item doc comments.
        if let Some(rest) = trimmed.strip_prefix("///") {
            doc_lines.push(rest.trim_start_matches(' '));
            continue;
        }

        // Skip indented lines (inside trait/impl/struct bodies).
        if line.starts_with(' ') || line.starts_with('\t') {
            if !trimmed.is_empty() && !trimmed.starts_with("//") {
                doc_lines.clear();
            }
            continue;
        }

        // Check for item definitions.
        let (vis, rest_of_line) = if let Some(r) = trimmed.strip_prefix("pub(crate) ") {
            (Visibility::Crate, r)
        } else if let Some(r) = trimmed.strip_prefix("pub ") {
            (Visibility::Public, r)
        } else {
            (Visibility::Private, trimmed)
        };

        let mut created_item = false;

        if let Some(after_fn) = rest_of_line.strip_prefix("fn ") {
            let name = after_fn
                .split(|c: char| c == '(' || c == '<' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                let path = format!("{module_path}::{name}");
                let mut item = DocItem::new(&path, name, ItemKind::Function);
                item.visibility = vis;
                item.doc = parse_doc_comment(&doc_lines);
                item.signature = Some(trimmed.trim_end_matches('{').trim().to_string());
                item.source_line = Some(i + 1);
                items.push(item);
                created_item = true;
            }
        } else if let Some(after) = rest_of_line.strip_prefix("struct ") {
            let name = after
                .split(|c: char| c == '{' || c == '(' || c == '<' || c.is_whitespace() || c == ';')
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                let path = format!("{module_path}::{name}");
                let mut item = DocItem::new(&path, name, ItemKind::Struct);
                item.visibility = vis;
                item.doc = parse_doc_comment(&doc_lines);
                item.source_line = Some(i + 1);
                items.push(item);
                created_item = true;
            }
        } else if let Some(after) = rest_of_line.strip_prefix("enum ") {
            let name = after
                .split(|c: char| c == '{' || c == '<' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                let path = format!("{module_path}::{name}");
                let mut item = DocItem::new(&path, name, ItemKind::Enum);
                item.visibility = vis;
                item.doc = parse_doc_comment(&doc_lines);
                item.source_line = Some(i + 1);
                items.push(item);
                created_item = true;
            }
        } else if let Some(after) = rest_of_line.strip_prefix("trait ") {
            let name = after
                .split(|c: char| c == '{' || c == ':' || c == '<' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                let path = format!("{module_path}::{name}");
                let mut item = DocItem::new(&path, name, ItemKind::Trait);
                item.visibility = vis;
                item.doc = parse_doc_comment(&doc_lines);
                item.source_line = Some(i + 1);
                items.push(item);
                created_item = true;
            }
        } else if let Some(after) = rest_of_line.strip_prefix("const ") {
            let name = after
                .split(|c: char| c == ':' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                let path = format!("{module_path}::{name}");
                let mut item = DocItem::new(&path, name, ItemKind::Constant);
                item.visibility = vis;
                item.doc = parse_doc_comment(&doc_lines);
                item.source_line = Some(i + 1);
                items.push(item);
                created_item = true;
            }
        } else if let Some(after) = rest_of_line.strip_prefix("type ") {
            let name = after
                .split(|c: char| c == '=' || c == '<' || c.is_whitespace())
                .next()
                .unwrap_or("");
            if !name.is_empty() {
                let path = format!("{module_path}::{name}");
                let mut item = DocItem::new(&path, name, ItemKind::TypeAlias);
                item.visibility = vis;
                item.doc = parse_doc_comment(&doc_lines);
                item.source_line = Some(i + 1);
                items.push(item);
                created_item = true;
            }
        }

        if created_item || (!trimmed.is_empty() && !trimmed.starts_with("//")) {
            doc_lines.clear();
        }
    }

    // Add module doc if present.
    if !module_doc_lines.is_empty() {
        let mut mod_item = DocItem::new(module_path, module_path, ItemKind::Module);
        mod_item.doc = parse_doc_comment(&module_doc_lines);
        mod_item.visibility = Visibility::Public;
        items.insert(0, mod_item);
    }

    items
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_doc_comment() {
        let lines = vec!["Summary line.", "", "More details here."];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.summary, "Summary line.");
        assert_eq!(doc.description, "More details here.");
    }

    #[test]
    fn parse_doc_with_params() {
        let lines = vec![
            "Do something.",
            "",
            "# Arguments",
            "* `name` - The name to use",
            "* `count` - How many times",
        ];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.params.len(), 2);
        assert_eq!(doc.params[0].name, "name");
        assert_eq!(doc.params[1].name, "count");
    }

    #[test]
    fn parse_doc_with_returns() {
        let lines = vec!["Compute a value.", "", "# Returns", "The computed result."];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.returns, Some("The computed result.".to_string()));
    }

    #[test]
    fn parse_doc_with_examples() {
        let lines = vec!["A function.", "", "```rust", "let x = 42;", "```"];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.examples[0].language, "rust");
        assert!(doc.examples[0].code.contains("let x = 42;"));
        assert!(doc.examples[0].testable);
    }

    #[test]
    fn parse_doc_with_panics() {
        let lines = vec!["Run something.", "", "# Panics", "If the input is empty."];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.panics, Some("If the input is empty.".to_string()));
    }

    #[test]
    fn parse_doc_with_at_tags() {
        let lines = vec![
            "Do work.",
            "@param name The user name",
            "@returns The result",
            "@see OtherThing",
        ];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.params.len(), 1);
        assert_eq!(doc.params[0].name, "name");
        assert!(doc.returns.is_some());
        assert_eq!(doc.see_also, vec!["OtherThing"]);
    }

    #[test]
    fn extract_items_from_source() {
        let source = "\
//! Module docs.

/// A struct.
pub struct Foo {
    x: i32,
}

/// Do something.
pub fn bar() {}

/// An enum.
pub enum Color {
    Red,
    Blue,
}
";
        let items = extract_doc_items(source, "crate::mymod");
        // Module + Foo + bar + Color = 4 items.
        assert_eq!(items.len(), 4);
        assert_eq!(items[0].kind, ItemKind::Module);
        assert_eq!(items[1].kind, ItemKind::Struct);
        assert_eq!(items[1].name, "Foo");
        assert_eq!(items[2].kind, ItemKind::Function);
        assert_eq!(items[2].name, "bar");
        assert_eq!(items[3].kind, ItemKind::Enum);
    }

    #[test]
    fn extract_visibility() {
        let source = "\
pub fn public_fn() {}
pub(crate) fn crate_fn() {}
fn private_fn() {}
";
        let items = extract_doc_items(source, "crate");
        assert_eq!(items[0].visibility, Visibility::Public);
        assert_eq!(items[1].visibility, Visibility::Crate);
        assert_eq!(items[2].visibility, Visibility::Private);
    }

    #[test]
    fn generate_markdown_basic() {
        let mut item = DocItem::new("crate::foo", "foo", ItemKind::Function);
        item.visibility = Visibility::Public;
        item.doc.summary = "Does something useful.".to_string();
        item.signature = Some("pub fn foo(x: i32) -> bool".to_string());

        let md = generate_markdown(&[item], "My Crate");
        assert!(md.contains("# My Crate"));
        assert!(md.contains("### foo"));
        assert!(md.contains("Does something useful."));
        assert!(md.contains("pub fn foo"));
    }

    #[test]
    fn toc_generation() {
        let items = vec![
            DocItem::new("c::Foo", "Foo", ItemKind::Struct),
            DocItem::new("c::Bar", "Bar", ItemKind::Struct),
            DocItem::new("c::baz", "baz", ItemKind::Function),
        ];
        let toc = build_toc(&items);
        assert!(toc.len() >= 2); // Structs, Functions
        let struct_entry = toc.iter().find(|e| e.title.contains("Struct")).unwrap();
        assert_eq!(struct_entry.children.len(), 2);
    }

    #[test]
    fn api_surface_enumeration() {
        let items = vec![
            {
                let mut i = DocItem::new("c::Foo", "Foo", ItemKind::Struct);
                i.visibility = Visibility::Public;
                i.doc.summary = "A struct".to_string();
                i
            },
            {
                let mut i = DocItem::new("c::bar", "bar", ItemKind::Function);
                i.visibility = Visibility::Public;
                i
            },
            {
                let mut i = DocItem::new("c::priv_fn", "priv_fn", ItemKind::Function);
                i.visibility = Visibility::Private;
                i
            },
        ];
        let api = enumerate_api(&items);
        assert_eq!(api.total, 2); // only public items
        assert_eq!(api.structs, 1);
        assert_eq!(api.functions, 1);
        assert_eq!(api.documented_pct, 50.0); // Foo has summary, bar doesn't
    }

    #[test]
    fn cross_ref_resolution() {
        let mut items = HashMap::new();
        items.insert(
            "Foo".to_string(),
            DocItem::new("crate::Foo", "Foo", ItemKind::Struct),
        );

        let refs = resolve_cross_refs("See [`Foo`] and [`Bar`].", &items);
        assert_eq!(refs.len(), 2);
        assert!(refs[0].resolved); // Foo exists
        assert!(!refs[1].resolved); // Bar doesn't exist
    }

    #[test]
    fn cross_ref_no_refs() {
        let items = HashMap::new();
        let refs = resolve_cross_refs("No references here.", &items);
        assert!(refs.is_empty());
    }

    #[test]
    fn item_kind_display() {
        assert_eq!(format!("{}", ItemKind::Struct), "Struct");
        assert_eq!(format!("{}", ItemKind::Function), "Function");
    }

    #[test]
    fn visibility_display() {
        assert_eq!(format!("{}", Visibility::Public), "pub");
        assert_eq!(format!("{}", Visibility::Private), "(private)");
    }

    #[test]
    fn doc_error_display() {
        let e = DocGenError::ItemNotFound("Foo".to_string());
        assert!(format!("{e}").contains("Foo"));
    }

    #[test]
    fn doc_with_safety_section() {
        let lines = vec!["Unsafe fn.", "", "# Safety", "Caller must ensure alignment."];
        let doc = parse_doc_comment(&lines);
        assert!(doc.safety.is_some());
        assert!(doc.safety.unwrap().contains("alignment"));
    }

    #[test]
    fn extract_trait() {
        let source = "/// A trait.\npub trait MyTrait {\n    fn method(&self);\n}";
        let items = extract_doc_items(source, "crate");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, ItemKind::Trait);
        assert_eq!(items[0].name, "MyTrait");
    }

    #[test]
    fn extract_const() {
        let source = "/// A constant.\npub const MAX_SIZE: usize = 1024;";
        let items = extract_doc_items(source, "crate");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, ItemKind::Constant);
        assert_eq!(items[0].name, "MAX_SIZE");
    }

    #[test]
    fn extract_type_alias() {
        let source = "/// A type alias.\npub type Result<T> = std::result::Result<T, Error>;";
        let items = extract_doc_items(source, "crate");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, ItemKind::TypeAlias);
    }

    #[test]
    fn empty_doc() {
        let doc = parse_doc_comment(&[]);
        assert!(doc.summary.is_empty());
        assert!(doc.description.is_empty());
        assert!(doc.examples.is_empty());
    }

    #[test]
    fn code_example_no_run() {
        let lines = vec!["Example.", "", "```text", "not testable", "```"];
        let doc = parse_doc_comment(&lines);
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.examples[0].language, "text");
    }

    #[test]
    fn markdown_with_params_and_returns() {
        let mut item = DocItem::new("c::calc", "calc", ItemKind::Function);
        item.visibility = Visibility::Public;
        item.doc.summary = "Calculate something.".to_string();
        item.doc.params.push(ParamDoc {
            name: "x".to_string(),
            type_name: "i32".to_string(),
            description: "The input value".to_string(),
        });
        item.doc.returns = Some("The result".to_string());

        let md = generate_markdown(&[item], "Test");
        assert!(md.contains("`x`"));
        assert!(md.contains("The result"));
    }
}
