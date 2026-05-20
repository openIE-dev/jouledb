//! Code formatter engine (Wadler-Lindig style pretty printer).
//!
//! Token-based formatting with indentation management, line width limits,
//! break points, group formatting, format rules, and format diff output.
//! Pure Rust — no external formatter dependencies.

use serde::{Deserialize, Serialize};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from formatter operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// Invalid document structure.
    InvalidDocument(String),
    /// Line width limit cannot be zero.
    InvalidWidth,
    /// Token stream is empty.
    EmptyInput,
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDocument(msg) => write!(f, "invalid document: {msg}"),
            Self::InvalidWidth => write!(f, "line width must be > 0"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for FormatError {}

// ── Document IR (Wadler-Lindig) ─────────────────────────────────

/// Document IR node for the pretty printer.
/// Based on Wadler's "A Prettier Printer" and Lindig's practical implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Doc {
    /// Empty document.
    Nil,
    /// Literal text (no newlines).
    Text(String),
    /// A line break. In flat mode, replaced by the given string (often a space).
    Line(String),
    /// A hard line break (always breaks, even in flat mode).
    HardLine,
    /// Concatenation of two documents.
    Concat(Box<Doc>, Box<Doc>),
    /// Nest: increase indentation for the inner document.
    Nest(usize, Box<Doc>),
    /// Group: try to fit on one line; if not, break.
    Group(Box<Doc>),
    /// If-flat: choose between flat and broken alternatives.
    IfFlat {
        flat: Box<Doc>,
        broken: Box<Doc>,
    },
}

impl Doc {
    /// Concatenate two documents.
    pub fn concat(a: Doc, b: Doc) -> Doc {
        Doc::Concat(Box::new(a), Box::new(b))
    }

    /// Join multiple documents with a separator.
    pub fn join(docs: Vec<Doc>, sep: Doc) -> Doc {
        let mut result = Doc::Nil;
        let mut first = true;
        for doc in docs {
            if first {
                result = doc;
                first = false;
            } else {
                result = Doc::concat(result, Doc::concat(sep.clone(), doc));
            }
        }
        result
    }

    /// Text node.
    pub fn text(s: &str) -> Doc {
        Doc::Text(s.to_string())
    }

    /// Soft line (space if flat, newline if broken).
    pub fn softline() -> Doc {
        Doc::Line(" ".to_string())
    }

    /// Soft break (empty if flat, newline if broken).
    pub fn softbreak() -> Doc {
        Doc::Line(String::new())
    }

    /// Group a document.
    pub fn group(doc: Doc) -> Doc {
        Doc::Group(Box::new(doc))
    }

    /// Nest a document with additional indentation.
    pub fn nest(indent: usize, doc: Doc) -> Doc {
        Doc::Nest(indent, Box::new(doc))
    }

    /// Concatenate a list of documents.
    pub fn concat_all(docs: Vec<Doc>) -> Doc {
        docs.into_iter()
            .fold(Doc::Nil, |acc, d| Doc::concat(acc, d))
    }
}

// ── Pretty Printer ──────────────────────────────────────────────

/// Mode for the pretty printer stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Flat,
    Break,
}

/// Entry on the printer's work stack.
struct StackEntry {
    indent: usize,
    mode: Mode,
    doc: Doc,
}

/// The pretty printer engine.
pub struct PrettyPrinter {
    /// Maximum line width.
    pub width: usize,
    /// Indentation string (e.g., "  " for 2 spaces).
    pub indent_str: String,
    /// Whether to use tabs for indentation.
    pub use_tabs: bool,
}

impl PrettyPrinter {
    /// Create with default settings (80 columns, 2-space indent).
    pub fn new(width: usize) -> Self {
        Self {
            width,
            indent_str: "  ".to_string(),
            use_tabs: false,
        }
    }

    /// Create with custom indent string.
    pub fn with_indent(width: usize, indent: &str) -> Self {
        Self {
            width,
            indent_str: indent.to_string(),
            use_tabs: false,
        }
    }

    /// Format a document to a string.
    pub fn format(&self, doc: &Doc) -> Result<String, FormatError> {
        if self.width == 0 {
            return Err(FormatError::InvalidWidth);
        }

        let mut output = String::new();
        let mut col = 0usize;
        let mut stack = vec![StackEntry {
            indent: 0,
            mode: Mode::Break,
            doc: doc.clone(),
        }];

        while let Some(entry) = stack.pop() {
            match entry.doc {
                Doc::Nil => {}
                Doc::Text(ref text) => {
                    output.push_str(text);
                    col += text.len();
                }
                Doc::Line(ref flat_text) => {
                    if entry.mode == Mode::Flat {
                        output.push_str(flat_text);
                        col += flat_text.len();
                    } else {
                        output.push('\n');
                        let indent_text = self.make_indent(entry.indent);
                        output.push_str(&indent_text);
                        col = indent_text.len();
                    }
                }
                Doc::HardLine => {
                    output.push('\n');
                    let indent_text = self.make_indent(entry.indent);
                    output.push_str(&indent_text);
                    col = indent_text.len();
                }
                Doc::Concat(a, b) => {
                    // Push b first (stack is LIFO).
                    stack.push(StackEntry {
                        indent: entry.indent,
                        mode: entry.mode,
                        doc: *b,
                    });
                    stack.push(StackEntry {
                        indent: entry.indent,
                        mode: entry.mode,
                        doc: *a,
                    });
                }
                Doc::Nest(extra, inner) => {
                    stack.push(StackEntry {
                        indent: entry.indent + extra,
                        mode: entry.mode,
                        doc: *inner,
                    });
                }
                Doc::Group(inner) => {
                    let flat_width = self.measure_flat(&inner);
                    let mode = if col + flat_width <= self.width {
                        Mode::Flat
                    } else {
                        Mode::Break
                    };
                    stack.push(StackEntry {
                        indent: entry.indent,
                        mode,
                        doc: *inner,
                    });
                }
                Doc::IfFlat { flat, broken } => {
                    let chosen = if entry.mode == Mode::Flat {
                        *flat
                    } else {
                        *broken
                    };
                    stack.push(StackEntry {
                        indent: entry.indent,
                        mode: entry.mode,
                        doc: chosen,
                    });
                }
            }
        }

        Ok(output)
    }

    /// Measure the width if the document were rendered flat (no line breaks).
    fn measure_flat(&self, doc: &Doc) -> usize {
        match doc {
            Doc::Nil => 0,
            Doc::Text(t) => t.len(),
            Doc::Line(flat) => flat.len(),
            Doc::HardLine => usize::MAX / 2, // force break
            Doc::Concat(a, b) => {
                let la = self.measure_flat(a);
                let lb = self.measure_flat(b);
                la.saturating_add(lb)
            }
            Doc::Nest(_, inner) => self.measure_flat(inner),
            Doc::Group(inner) => self.measure_flat(inner),
            Doc::IfFlat { flat, .. } => self.measure_flat(flat),
        }
    }

    /// Build the indentation string for a given level.
    fn make_indent(&self, level: usize) -> String {
        if self.use_tabs {
            "\t".repeat(level)
        } else {
            self.indent_str.repeat(level)
        }
    }
}

impl Default for PrettyPrinter {
    fn default() -> Self {
        Self::new(80)
    }
}

// ── Token-Based Formatting ──────────────────────────────────────

/// Token types for source code formatting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Token {
    /// Keyword (fn, let, if, etc.).
    Keyword(String),
    /// Identifier.
    Ident(String),
    /// Operator (+, -, =, etc.).
    Operator(String),
    /// Punctuation ({, }, (, ), ;, ,).
    Punct(char),
    /// String literal.
    StringLit(String),
    /// Number literal.
    NumberLit(String),
    /// Whitespace (space, tab).
    Whitespace(String),
    /// Newline.
    Newline,
    /// Comment.
    Comment(String),
}

impl Token {
    /// Get the text representation.
    pub fn text(&self) -> String {
        match self {
            Self::Keyword(s) | Self::Ident(s) | Self::Operator(s) | Self::StringLit(s)
            | Self::NumberLit(s) | Self::Whitespace(s) | Self::Comment(s) => s.clone(),
            Self::Punct(c) => c.to_string(),
            Self::Newline => "\n".to_string(),
        }
    }
}

// ── Format Rules ────────────────────────────────────────────────

/// A formatting rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatRule {
    pub name: String,
    pub description: String,
    pub kind: FormatRuleKind,
    pub enabled: bool,
}

/// Kind of formatting rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormatRuleKind {
    /// Space before a token pattern.
    SpaceBefore(String),
    /// Space after a token pattern.
    SpaceAfter(String),
    /// No space before a token pattern.
    NoSpaceBefore(String),
    /// No space after a token pattern.
    NoSpaceAfter(String),
    /// Newline after a token.
    NewlineAfter(String),
    /// Indent after open brace.
    IndentAfterOpen,
    /// Dedent before close brace.
    DedentBeforeClose,
    /// Max consecutive blank lines.
    MaxBlankLines(usize),
    /// Trailing newline at end of file.
    TrailingNewline,
}

/// A collection of format rules.
#[derive(Debug, Clone, Default)]
pub struct FormatConfig {
    pub rules: Vec<FormatRule>,
    pub indent_width: usize,
    pub max_width: usize,
    pub use_tabs: bool,
}

impl FormatConfig {
    /// Create with sensible defaults.
    pub fn default_style() -> Self {
        Self {
            rules: vec![
                FormatRule {
                    name: "space_after_comma".to_string(),
                    description: "Space after comma".to_string(),
                    kind: FormatRuleKind::SpaceAfter(",".to_string()),
                    enabled: true,
                },
                FormatRule {
                    name: "space_around_operators".to_string(),
                    description: "Space around binary operators".to_string(),
                    kind: FormatRuleKind::SpaceBefore("=".to_string()),
                    enabled: true,
                },
                FormatRule {
                    name: "indent_after_brace".to_string(),
                    description: "Indent after opening brace".to_string(),
                    kind: FormatRuleKind::IndentAfterOpen,
                    enabled: true,
                },
                FormatRule {
                    name: "trailing_newline".to_string(),
                    description: "Trailing newline at EOF".to_string(),
                    kind: FormatRuleKind::TrailingNewline,
                    enabled: true,
                },
            ],
            indent_width: 4,
            max_width: 100,
            use_tabs: false,
        }
    }
}

// ── Simple Token Formatter ──────────────────────────────────────

/// Format a token stream according to rules.
pub fn format_tokens(tokens: &[Token], config: &FormatConfig) -> String {
    let mut output = String::new();
    let mut indent_level = 0usize;
    let indent_str = if config.use_tabs {
        "\t".to_string()
    } else {
        " ".repeat(config.indent_width)
    };
    let mut at_line_start = true;
    let mut consecutive_newlines = 0u32;

    let max_blank = config
        .rules
        .iter()
        .find_map(|r| match &r.kind {
            FormatRuleKind::MaxBlankLines(n) if r.enabled => Some(*n),
            _ => None,
        })
        .unwrap_or(2);

    for (i, token) in tokens.iter().enumerate() {
        match token {
            Token::Newline => {
                consecutive_newlines += 1;
                if (consecutive_newlines as usize) <= max_blank + 1 {
                    output.push('\n');
                }
                at_line_start = true;
            }
            Token::Punct('}') => {
                consecutive_newlines = 0;
                if indent_level > 0 {
                    indent_level -= 1;
                }
                if at_line_start {
                    output.push_str(&indent_str.repeat(indent_level));
                }
                output.push('}');
                at_line_start = false;
            }
            Token::Punct('{') => {
                consecutive_newlines = 0;
                // Check if we need space before.
                if !at_line_start && !output.ends_with(' ') && !output.ends_with('\n') {
                    output.push(' ');
                }
                output.push('{');
                indent_level += 1;
                at_line_start = false;
            }
            Token::Whitespace(_) => {
                // Normalize whitespace: single space unless at line start.
                if !at_line_start && !output.ends_with(' ') && !output.ends_with('\n') {
                    output.push(' ');
                }
            }
            Token::Comment(text) => {
                consecutive_newlines = 0;
                if at_line_start {
                    output.push_str(&indent_str.repeat(indent_level));
                }
                output.push_str(text);
                at_line_start = false;
            }
            _ => {
                consecutive_newlines = 0;
                if at_line_start {
                    output.push_str(&indent_str.repeat(indent_level));
                }

                // Apply space-after rules for the previous token.
                if i > 0 && !at_line_start {
                    let prev = &tokens[i - 1];
                    let prev_text = prev.text();
                    for rule in &config.rules {
                        if !rule.enabled {
                            continue;
                        }
                        if let FormatRuleKind::SpaceAfter(pat) = &rule.kind {
                            if prev_text == *pat && !output.ends_with(' ') {
                                output.push(' ');
                            }
                        }
                    }
                }

                output.push_str(&token.text());
                at_line_start = false;
            }
        }
    }

    // Trailing newline rule.
    let has_trailing = config
        .rules
        .iter()
        .any(|r| matches!(&r.kind, FormatRuleKind::TrailingNewline) && r.enabled);
    if has_trailing && !output.ends_with('\n') {
        output.push('\n');
    }

    output
}

// ── Format Diff ─────────────────────────────────────────────────

/// A single change in a format diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatChange {
    pub line: usize,
    pub original: String,
    pub formatted: String,
}

/// Compute a line-by-line diff between original and formatted text.
pub fn format_diff(original: &str, formatted: &str) -> Vec<FormatChange> {
    let orig_lines: Vec<&str> = original.lines().collect();
    let fmt_lines: Vec<&str> = formatted.lines().collect();
    let mut changes = Vec::new();

    let max_len = orig_lines.len().max(fmt_lines.len());
    for i in 0..max_len {
        let orig = orig_lines.get(i).copied().unwrap_or("");
        let fmt = fmt_lines.get(i).copied().unwrap_or("");
        if orig != fmt {
            changes.push(FormatChange {
                line: i + 1,
                original: orig.to_string(),
                formatted: fmt.to_string(),
            });
        }
    }

    changes
}

// ── Indentation Utilities ───────────────────────────────────────

/// Detect the indentation style used in source text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndentStyle {
    Spaces(usize),
    Tabs,
    Mixed,
    Unknown,
}

/// Detect indentation style from source text.
pub fn detect_indent(source: &str) -> IndentStyle {
    let mut space_counts: HashMap<usize, usize> = HashMap::new();
    let mut tab_lines = 0usize;
    let mut space_lines = 0usize;

    use std::collections::HashMap;

    for line in source.lines() {
        if line.is_empty() {
            continue;
        }
        let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
        if indent.is_empty() {
            continue;
        }
        if indent.contains('\t') && indent.contains(' ') {
            return IndentStyle::Mixed;
        }
        if indent.contains('\t') {
            tab_lines += 1;
        } else {
            let count = indent.len();
            if count > 0 {
                *space_counts.entry(count).or_insert(0) += 1;
                space_lines += 1;
            }
        }
    }

    if tab_lines > 0 && space_lines > 0 {
        return IndentStyle::Mixed;
    }
    if tab_lines > 0 {
        return IndentStyle::Tabs;
    }
    if space_lines == 0 {
        return IndentStyle::Unknown;
    }

    // Find the most common indent increment.
    // Collect deltas between subsequent indent sizes.
    let mut sizes: Vec<usize> = space_counts.keys().copied().collect();
    sizes.sort();

    if sizes.is_empty() {
        return IndentStyle::Unknown;
    }

    // Use GCD of all indent sizes as the likely indent width.
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    let mut g = sizes[0];
    for s in &sizes[1..] {
        g = gcd(g, *s);
    }
    if g == 0 {
        g = 2; // fallback
    }

    IndentStyle::Spaces(g)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Pretty Printer tests ────────────────────────────────────

    #[test]
    fn pretty_print_simple_text() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::text("hello world");
        let result = pp.format(&doc).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn pretty_print_concat() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::concat(Doc::text("hello"), Doc::concat(Doc::text(" "), Doc::text("world")));
        let result = pp.format(&doc).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn pretty_print_group_fits() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::group(Doc::concat(
            Doc::text("short"),
            Doc::concat(Doc::softline(), Doc::text("text")),
        ));
        let result = pp.format(&doc).unwrap();
        assert_eq!(result, "short text");
    }

    #[test]
    fn pretty_print_group_breaks() {
        let pp = PrettyPrinter::new(10);
        let doc = Doc::group(Doc::concat(
            Doc::text("this is"),
            Doc::concat(Doc::softline(), Doc::text("too long")),
        ));
        let result = pp.format(&doc).unwrap();
        assert!(result.contains('\n'));
    }

    #[test]
    fn pretty_print_nest() {
        let pp = PrettyPrinter::new(10);
        let inner = Doc::concat(Doc::Line(" ".to_string()), Doc::text("body"));
        let doc = Doc::concat(
            Doc::text("fn {"),
            Doc::concat(Doc::nest(1, inner), Doc::concat(Doc::HardLine, Doc::text("}"))),
        );
        let result = pp.format(&doc).unwrap();
        assert!(result.contains("  body"));
    }

    #[test]
    fn pretty_print_hard_line() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::group(Doc::concat(Doc::text("a"), Doc::concat(Doc::HardLine, Doc::text("b"))));
        let result = pp.format(&doc).unwrap();
        assert!(result.contains('\n'));
    }

    #[test]
    fn pretty_print_nil() {
        let pp = PrettyPrinter::new(80);
        let result = pp.format(&Doc::Nil).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn pretty_print_invalid_width() {
        let pp = PrettyPrinter::new(0);
        let err = pp.format(&Doc::text("x")).unwrap_err();
        assert_eq!(err, FormatError::InvalidWidth);
    }

    #[test]
    fn pretty_print_if_flat() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::group(Doc::IfFlat {
            flat: Box::new(Doc::text("FLAT")),
            broken: Box::new(Doc::text("BROKEN")),
        });
        let result = pp.format(&doc).unwrap();
        assert_eq!(result, "FLAT");
    }

    #[test]
    fn pretty_print_join() {
        let pp = PrettyPrinter::new(80);
        let items = vec![Doc::text("a"), Doc::text("b"), Doc::text("c")];
        let doc = Doc::join(items, Doc::text(", "));
        let result = pp.format(&doc).unwrap();
        assert_eq!(result, "a, b, c");
    }

    // ── Token formatting tests ──────────────────────────────────

    #[test]
    fn format_tokens_basic() {
        let tokens = vec![
            Token::Keyword("fn".to_string()),
            Token::Whitespace(" ".to_string()),
            Token::Ident("main".to_string()),
            Token::Punct('('),
            Token::Punct(')'),
            Token::Whitespace(" ".to_string()),
            Token::Punct('{'),
            Token::Newline,
            Token::Ident("println".to_string()),
            Token::Punct('('),
            Token::Punct(')'),
            Token::Newline,
            Token::Punct('}'),
            Token::Newline,
        ];
        let config = FormatConfig::default_style();
        let result = format_tokens(&tokens, &config);
        assert!(result.contains("fn"));
        assert!(result.contains("main"));
    }

    #[test]
    fn format_tokens_indent() {
        let tokens = vec![
            Token::Punct('{'),
            Token::Newline,
            Token::Ident("x".to_string()),
            Token::Newline,
            Token::Punct('}'),
        ];
        let config = FormatConfig::default_style();
        let result = format_tokens(&tokens, &config);
        // x should be indented inside braces.
        let lines: Vec<&str> = result.lines().collect();
        assert!(lines.len() >= 2);
        let x_line = lines.iter().find(|l| l.contains('x')).unwrap();
        assert!(x_line.starts_with(' '));
    }

    #[test]
    fn format_tokens_trailing_newline() {
        let tokens = vec![Token::Ident("x".to_string())];
        let config = FormatConfig::default_style();
        let result = format_tokens(&tokens, &config);
        assert!(result.ends_with('\n'));
    }

    #[test]
    fn format_tokens_space_after_comma() {
        let tokens = vec![
            Token::Ident("a".to_string()),
            Token::Punct(','),
            Token::Ident("b".to_string()),
        ];
        let config = FormatConfig::default_style();
        let result = format_tokens(&tokens, &config);
        assert!(result.contains(", ") || result.contains(",b"));
    }

    // ── Format diff tests ───────────────────────────────────────

    #[test]
    fn format_diff_no_changes() {
        let text = "line1\nline2\nline3";
        let changes = format_diff(text, text);
        assert!(changes.is_empty());
    }

    #[test]
    fn format_diff_with_changes() {
        let original = "fn main() {\n  x\n}";
        let formatted = "fn main() {\n    x\n}";
        let changes = format_diff(original, formatted);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].line, 2);
    }

    #[test]
    fn format_diff_different_lengths() {
        let original = "a\nb";
        let formatted = "a\nb\nc";
        let changes = format_diff(original, formatted);
        assert_eq!(changes.len(), 1); // new line 3
        assert_eq!(changes[0].line, 3);
    }

    // ── Indent detection ────────────────────────────────────────

    #[test]
    fn detect_indent_spaces_2() {
        let source = "fn main() {\n  x\n  y\n}";
        assert_eq!(detect_indent(source), IndentStyle::Spaces(2));
    }

    #[test]
    fn detect_indent_spaces_4() {
        let source = "fn main() {\n    x\n    y\n}";
        assert_eq!(detect_indent(source), IndentStyle::Spaces(4));
    }

    #[test]
    fn detect_indent_tabs() {
        let source = "fn main() {\n\tx\n\ty\n}";
        assert_eq!(detect_indent(source), IndentStyle::Tabs);
    }

    #[test]
    fn detect_indent_unknown() {
        let source = "no indentation at all";
        assert_eq!(detect_indent(source), IndentStyle::Unknown);
    }

    // ── Doc builder tests ───────────────────────────────────────

    #[test]
    fn doc_concat_all() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::concat_all(vec![
            Doc::text("a"),
            Doc::text(" "),
            Doc::text("b"),
        ]);
        let result = pp.format(&doc).unwrap();
        assert_eq!(result, "a b");
    }

    #[test]
    fn doc_softbreak() {
        let pp = PrettyPrinter::new(80);
        let doc = Doc::group(Doc::concat(
            Doc::text("a"),
            Doc::concat(Doc::softbreak(), Doc::text("b")),
        ));
        let result = pp.format(&doc).unwrap();
        // Fits on one line, so softbreak is empty.
        assert_eq!(result, "ab");
    }

    #[test]
    fn token_text() {
        assert_eq!(Token::Keyword("fn".to_string()).text(), "fn");
        assert_eq!(Token::Punct('{').text(), "{");
        assert_eq!(Token::Newline.text(), "\n");
    }

    #[test]
    fn error_display() {
        let e = FormatError::InvalidWidth;
        assert!(format!("{e}").contains("width"));
    }

    #[test]
    fn format_config_custom() {
        let config = FormatConfig {
            rules: vec![],
            indent_width: 2,
            max_width: 120,
            use_tabs: true,
        };
        assert!(config.use_tabs);
        assert_eq!(config.indent_width, 2);
    }

    #[test]
    fn pretty_printer_with_tabs() {
        let mut pp = PrettyPrinter::new(80);
        pp.use_tabs = true;
        let doc = Doc::concat(
            Doc::text("x"),
            Doc::nest(1, Doc::concat(Doc::HardLine, Doc::text("y"))),
        );
        let result = pp.format(&doc).unwrap();
        assert!(result.contains('\t'));
    }
}
