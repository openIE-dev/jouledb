//! Syntax highlighting — language definitions, token classification,
//! HTML span output with CSS classes, theme support, line numbers,
//! and highlight ranges.
//!
//! Pure-Rust replacement for highlight.js, Prism, and Shiki with zero
//! runtime dependencies.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Token Types ─────────────────────────────────────────────────

/// Classification of a syntax token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Keyword,
    Operator,
    StringLiteral,
    NumberLiteral,
    Comment,
    Identifier,
    Type,
    Function,
    Punctuation,
    Whitespace,
    LineBreak,
    Preprocessor,
    Attribute,
    Plain,
}

impl TokenKind {
    /// CSS class name for this token kind.
    pub fn css_class(&self) -> &str {
        match self {
            Self::Keyword => "hl-keyword",
            Self::Operator => "hl-operator",
            Self::StringLiteral => "hl-string",
            Self::NumberLiteral => "hl-number",
            Self::Comment => "hl-comment",
            Self::Identifier => "hl-ident",
            Self::Type => "hl-type",
            Self::Function => "hl-function",
            Self::Punctuation => "hl-punct",
            Self::Whitespace => "hl-ws",
            Self::LineBreak => "hl-break",
            Self::Preprocessor => "hl-preproc",
            Self::Attribute => "hl-attr",
            Self::Plain => "hl-plain",
        }
    }
}

// ── Token ───────────────────────────────────────────────────────

/// A classified token from source code.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub line: usize,
    pub column: usize,
}

// ── Language Definition ─────────────────────────────────────────

/// A language definition for the syntax highlighter.
#[derive(Debug, Clone)]
pub struct LanguageDefinition {
    pub name: String,
    pub keywords: Vec<String>,
    pub types: Vec<String>,
    pub operators: Vec<String>,
    pub single_line_comment: Option<String>,
    pub multi_line_comment: Option<(String, String)>,
    pub string_delimiters: Vec<char>,
    pub preprocessor_prefix: Option<String>,
}

impl LanguageDefinition {
    /// Create a Rust language definition.
    pub fn rust() -> Self {
        Self {
            name: "rust".into(),
            keywords: vec![
                "fn", "let", "mut", "const", "if", "else", "match", "while", "for", "loop",
                "return", "break", "continue", "struct", "enum", "impl", "trait", "pub", "use",
                "mod", "crate", "self", "super", "as", "in", "where", "async", "await", "move",
                "unsafe", "extern", "type", "static", "dyn", "true", "false",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            types: vec![
                "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128",
                "isize", "f32", "f64", "bool", "char", "str", "String", "Vec", "Option",
                "Result", "Box", "Rc", "Arc", "HashMap", "HashSet", "Self",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            operators: vec![
                "=>", "->", "::", "&&", "||", "==", "!=", "<=", ">=", "<<", ">>", "+=", "-=",
                "*=", "/=", "+", "-", "*", "/", "%", "&", "|", "^", "!", "<", ">", "=",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            single_line_comment: Some("//".into()),
            multi_line_comment: Some(("/*".into(), "*/".into())),
            string_delimiters: vec!['"'],
            preprocessor_prefix: None,
        }
    }

    /// Create a JavaScript language definition.
    pub fn javascript() -> Self {
        Self {
            name: "javascript".into(),
            keywords: vec![
                "function", "var", "let", "const", "if", "else", "return", "while", "for", "do",
                "switch", "case", "break", "continue", "new", "this", "class", "extends",
                "import", "export", "default", "from", "try", "catch", "finally", "throw",
                "async", "await", "yield", "typeof", "instanceof", "in", "of", "true", "false",
                "null", "undefined",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            types: vec![
                "Array", "Object", "String", "Number", "Boolean", "Symbol", "Map", "Set",
                "Promise", "Date", "RegExp", "Error", "JSON", "Math", "console",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            operators: vec![
                "===", "!==", "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "+=",
                "-=", "*=", "/=", "**", "+", "-", "*", "/", "%", "&", "|", "^", "!", "<",
                ">", "=", "~",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            single_line_comment: Some("//".into()),
            multi_line_comment: Some(("/*".into(), "*/".into())),
            string_delimiters: vec!['"', '\'', '`'],
            preprocessor_prefix: None,
        }
    }

    /// Create a Python language definition.
    pub fn python() -> Self {
        Self {
            name: "python".into(),
            keywords: vec![
                "def", "class", "if", "elif", "else", "while", "for", "return", "import",
                "from", "as", "with", "try", "except", "finally", "raise", "pass", "break",
                "continue", "lambda", "yield", "global", "nonlocal", "assert", "and", "or",
                "not", "in", "is", "True", "False", "None", "async", "await",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            types: vec![
                "int", "float", "str", "bool", "list", "dict", "tuple", "set", "frozenset",
                "bytes", "bytearray", "object", "type", "range", "complex",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            operators: vec![
                "**", "//", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "//=", "**=",
                "->", "+", "-", "*", "/", "%", "&", "|", "^", "~", "<", ">", "=",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            single_line_comment: Some("#".into()),
            multi_line_comment: None,
            string_delimiters: vec!['"', '\''],
            preprocessor_prefix: None,
        }
    }

    /// Create a C language definition.
    pub fn c() -> Self {
        Self {
            name: "c".into(),
            keywords: vec![
                "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
                "else", "enum", "extern", "float", "for", "goto", "if", "int", "long",
                "register", "return", "short", "signed", "sizeof", "static", "struct", "switch",
                "typedef", "union", "unsigned", "void", "volatile", "while",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            types: vec![
                "size_t", "ptrdiff_t", "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t",
                "uint16_t", "uint32_t", "uint64_t", "bool", "FILE",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            operators: vec![
                "->", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+=", "-=", "*=", "/=",
                "+", "-", "*", "/", "%", "&", "|", "^", "!", "<", ">", "=", "~",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            single_line_comment: Some("//".into()),
            multi_line_comment: Some(("/*".into(), "*/".into())),
            string_delimiters: vec!['"', '\''],
            preprocessor_prefix: Some("#".into()),
        }
    }

    /// Look up a language definition by name.
    pub fn by_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "rust" | "rs" => Some(Self::rust()),
            "javascript" | "js" => Some(Self::javascript()),
            "python" | "py" => Some(Self::python()),
            "c" => Some(Self::c()),
            _ => None,
        }
    }
}

// ── Tokenizer ───────────────────────────────────────────────────

/// Tokenize source code using a language definition.
pub fn tokenize(source: &str, lang: &LanguageDefinition) -> Vec<Token> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;

    // Sort operators by length descending for longest-match
    let mut sorted_ops = lang.operators.clone();
    sorted_ops.sort_by(|a, b| b.len().cmp(&a.len()));

    while i < len {
        // Newline
        if chars[i] == '\n' {
            tokens.push(Token {
                kind: TokenKind::LineBreak,
                text: "\n".into(),
                line,
                column: col,
            });
            line += 1;
            col = 1;
            i += 1;
            continue;
        }

        // Whitespace
        if chars[i].is_whitespace() {
            let start = i;
            let start_col = col;
            while i < len && chars[i].is_whitespace() && chars[i] != '\n' {
                i += 1;
                col += 1;
            }
            let text: String = chars[start..i].iter().collect();
            tokens.push(Token {
                kind: TokenKind::Whitespace,
                text,
                line,
                column: start_col,
            });
            continue;
        }

        // Single-line comment
        if let Some(prefix) = &lang.single_line_comment {
            let remaining: String = chars[i..].iter().collect();
            if remaining.starts_with(prefix.as_str()) {
                // Check it's not a preprocessor line when preprocessor uses same prefix
                let is_preproc = lang
                    .preprocessor_prefix
                    .as_ref()
                    .is_some_and(|pp| pp == prefix);

                if !is_preproc {
                    let start_col = col;
                    let start = i;
                    while i < len && chars[i] != '\n' {
                        i += 1;
                        col += 1;
                    }
                    let text: String = chars[start..i].iter().collect();
                    tokens.push(Token {
                        kind: TokenKind::Comment,
                        text,
                        line,
                        column: start_col,
                    });
                    continue;
                }
            }
        }

        // Multi-line comment
        if let Some((open, close)) = &lang.multi_line_comment {
            let remaining: String = chars[i..].iter().collect();
            if remaining.starts_with(open.as_str()) {
                let start_col = col;
                let start = i;
                i += open.len();
                col += open.len();
                loop {
                    if i >= len {
                        break;
                    }
                    let rest: String = chars[i..].iter().collect();
                    if rest.starts_with(close.as_str()) {
                        i += close.len();
                        col += close.len();
                        break;
                    }
                    if chars[i] == '\n' {
                        line += 1;
                        col = 1;
                    } else {
                        col += 1;
                    }
                    i += 1;
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(Token {
                    kind: TokenKind::Comment,
                    text,
                    line,
                    column: start_col,
                });
                continue;
            }
        }

        // Preprocessor
        if let Some(prefix) = &lang.preprocessor_prefix {
            let remaining: String = chars[i..].iter().collect();
            if remaining.starts_with(prefix.as_str()) && col == 1 {
                let start_col = col;
                let start = i;
                while i < len && chars[i] != '\n' {
                    i += 1;
                    col += 1;
                }
                let text: String = chars[start..i].iter().collect();
                tokens.push(Token {
                    kind: TokenKind::Preprocessor,
                    text,
                    line,
                    column: start_col,
                });
                continue;
            }
        }

        // String literals
        if lang.string_delimiters.contains(&chars[i]) {
            let delim = chars[i];
            let start_col = col;
            let start = i;
            i += 1;
            col += 1;
            while i < len && chars[i] != delim {
                if chars[i] == '\\' && i + 1 < len {
                    i += 1;
                    col += 1;
                }
                if chars[i] == '\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
                i += 1;
            }
            if i < len {
                i += 1;
                col += 1;
            }
            let text: String = chars[start..i].iter().collect();
            tokens.push(Token {
                kind: TokenKind::StringLiteral,
                text,
                line,
                column: start_col,
            });
            continue;
        }

        // Number literals
        if chars[i].is_ascii_digit() {
            let start = i;
            let start_col = col;
            while i < len && (chars[i].is_ascii_alphanumeric() || chars[i] == '.' || chars[i] == '_') {
                i += 1;
                col += 1;
            }
            let text: String = chars[start..i].iter().collect();
            tokens.push(Token {
                kind: TokenKind::NumberLiteral,
                text,
                line,
                column: start_col,
            });
            continue;
        }

        // Operators (longest match first)
        {
            let remaining: String = chars[i..].iter().collect();
            let mut matched = false;
            for op in &sorted_ops {
                if remaining.starts_with(op.as_str()) {
                    tokens.push(Token {
                        kind: TokenKind::Operator,
                        text: op.clone(),
                        line,
                        column: col,
                    });
                    i += op.len();
                    col += op.len();
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
        }

        // Identifiers / keywords / types
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            let start_col = col;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
                col += 1;
            }
            let text: String = chars[start..i].iter().collect();
            let kind = if lang.keywords.iter().any(|k| k == &text) {
                TokenKind::Keyword
            } else if lang.types.iter().any(|t| t == &text) {
                TokenKind::Type
            } else if i < len && chars[i] == '(' {
                TokenKind::Function
            } else {
                TokenKind::Identifier
            };
            tokens.push(Token {
                kind,
                text,
                line,
                column: start_col,
            });
            continue;
        }

        // Punctuation
        let ch = chars[i];
        tokens.push(Token {
            kind: TokenKind::Punctuation,
            text: ch.to_string(),
            line,
            column: col,
        });
        i += 1;
        col += 1;
    }

    tokens
}

// ── Theme ───────────────────────────────────────────────────────

/// A color theme for syntax highlighting.
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub colors: HashMap<TokenKind, String>,
    pub background: String,
    pub foreground: String,
}

impl Theme {
    /// A dark theme inspired by popular dark editor themes.
    pub fn dark() -> Self {
        let mut colors = HashMap::new();
        colors.insert(TokenKind::Keyword, "color: #c678dd".into());
        colors.insert(TokenKind::Operator, "color: #56b6c2".into());
        colors.insert(TokenKind::StringLiteral, "color: #98c379".into());
        colors.insert(TokenKind::NumberLiteral, "color: #d19a66".into());
        colors.insert(TokenKind::Comment, "color: #5c6370; font-style: italic".into());
        colors.insert(TokenKind::Identifier, "color: #e06c75".into());
        colors.insert(TokenKind::Type, "color: #e5c07b".into());
        colors.insert(TokenKind::Function, "color: #61afef".into());
        colors.insert(TokenKind::Punctuation, "color: #abb2bf".into());
        colors.insert(TokenKind::Preprocessor, "color: #c678dd".into());
        colors.insert(TokenKind::Attribute, "color: #d19a66".into());
        colors.insert(TokenKind::Plain, "color: #abb2bf".into());
        Self {
            name: "dark".into(),
            colors,
            background: "background: #282c34".into(),
            foreground: "color: #abb2bf".into(),
        }
    }

    /// A light theme for syntax highlighting.
    pub fn light() -> Self {
        let mut colors = HashMap::new();
        colors.insert(TokenKind::Keyword, "color: #a626a4".into());
        colors.insert(TokenKind::Operator, "color: #0184bc".into());
        colors.insert(TokenKind::StringLiteral, "color: #50a14f".into());
        colors.insert(TokenKind::NumberLiteral, "color: #986801".into());
        colors.insert(TokenKind::Comment, "color: #a0a1a7; font-style: italic".into());
        colors.insert(TokenKind::Identifier, "color: #e45649".into());
        colors.insert(TokenKind::Type, "color: #c18401".into());
        colors.insert(TokenKind::Function, "color: #4078f2".into());
        colors.insert(TokenKind::Punctuation, "color: #383a42".into());
        colors.insert(TokenKind::Preprocessor, "color: #a626a4".into());
        colors.insert(TokenKind::Attribute, "color: #986801".into());
        colors.insert(TokenKind::Plain, "color: #383a42".into());
        Self {
            name: "light".into(),
            colors,
            background: "background: #fafafa".into(),
            foreground: "color: #383a42".into(),
        }
    }

    /// Get inline CSS for a token kind.
    pub fn style_for(&self, kind: TokenKind) -> &str {
        self.colors
            .get(&kind)
            .map(|s| s.as_str())
            .unwrap_or("")
    }
}

// ── Highlight Range ─────────────────────────────────────────────

/// A range of lines to highlight specially.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HighlightRange {
    pub start_line: usize,
    pub end_line: usize,
    pub css_class: String,
}

impl HighlightRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start_line: start,
            end_line: end,
            css_class: "hl-highlight".into(),
        }
    }

    pub fn with_class(start: usize, end: usize, class: impl Into<String>) -> Self {
        Self {
            start_line: start,
            end_line: end,
            css_class: class.into(),
        }
    }

    pub fn contains_line(&self, line: usize) -> bool {
        line >= self.start_line && line <= self.end_line
    }
}

// ── Rendering Config ────────────────────────────────────────────

/// Configuration for rendering highlighted code.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    pub show_line_numbers: bool,
    pub use_inline_styles: bool,
    pub highlight_ranges: Vec<HighlightRange>,
    pub starting_line: usize,
    pub wrap_in_pre: bool,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            show_line_numbers: false,
            use_inline_styles: false,
            highlight_ranges: Vec::new(),
            starting_line: 1,
            wrap_in_pre: true,
        }
    }
}

// ── HTML Renderer ───────────────────────────────────────────────

/// Escape HTML special characters.
fn html_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Render tokens to HTML with CSS classes.
pub fn render_html(tokens: &[Token], config: &RenderConfig, theme: Option<&Theme>) -> String {
    let mut out = String::new();

    if config.wrap_in_pre {
        if config.use_inline_styles {
            if let Some(t) = theme {
                let _ = write!(
                    out,
                    "<pre style=\"{}; {}\"><code>",
                    t.background, t.foreground
                );
            } else {
                out.push_str("<pre><code>");
            }
        } else {
            out.push_str("<pre><code>");
        }
    }

    let mut current_line = config.starting_line;

    // Emit first line number if needed
    if config.show_line_numbers {
        let line_class = if config
            .highlight_ranges
            .iter()
            .any(|r| r.contains_line(current_line))
        {
            let range = config
                .highlight_ranges
                .iter()
                .find(|r| r.contains_line(current_line))
                .unwrap();
            format!(" {}", range.css_class)
        } else {
            String::new()
        };
        let _ = write!(
            out,
            "<span class=\"hl-line-number{}\">{:>4} </span>",
            line_class, current_line
        );
    }

    for token in tokens {
        if token.kind == TokenKind::LineBreak {
            out.push('\n');
            current_line += 1;
            if config.show_line_numbers {
                let line_class = if config
                    .highlight_ranges
                    .iter()
                    .any(|r| r.contains_line(current_line))
                {
                    let range = config
                        .highlight_ranges
                        .iter()
                        .find(|r| r.contains_line(current_line))
                        .unwrap();
                    format!(" {}", range.css_class)
                } else {
                    String::new()
                };
                let _ = write!(
                    out,
                    "<span class=\"hl-line-number{}\">{:>4} </span>",
                    line_class, current_line
                );
            }
            continue;
        }

        if token.kind == TokenKind::Whitespace {
            out.push_str(&token.text);
            continue;
        }

        let escaped = html_escape(&token.text);

        if config.use_inline_styles {
            if let Some(t) = theme {
                let style = t.style_for(token.kind);
                if style.is_empty() {
                    out.push_str(&escaped);
                } else {
                    let _ = write!(out, "<span style=\"{}\">{}</span>", style, escaped);
                }
            } else {
                let _ = write!(
                    out,
                    "<span class=\"{}\">{}</span>",
                    token.kind.css_class(),
                    escaped
                );
            }
        } else {
            let _ = write!(
                out,
                "<span class=\"{}\">{}</span>",
                token.kind.css_class(),
                escaped
            );
        }
    }

    if config.wrap_in_pre {
        out.push_str("</code></pre>");
    }

    out
}

/// Convenience: tokenize and render in one step.
pub fn highlight(source: &str, language: &str, config: &RenderConfig, theme: Option<&Theme>) -> String {
    let lang = LanguageDefinition::by_name(language);
    match lang {
        Some(def) => {
            let tokens = tokenize(source, &def);
            render_html(&tokens, config, theme)
        }
        None => {
            // Unknown language — plain text
            let escaped = html_escape(source);
            if config.wrap_in_pre {
                format!("<pre><code>{}</code></pre>", escaped)
            } else {
                escaped
            }
        }
    }
}

/// Generate CSS for a theme.
pub fn generate_theme_css(theme: &Theme) -> String {
    let mut css = String::new();
    let _ = write!(css, "pre.hl-{} {{ {}; {} }}\n", theme.name, theme.background, theme.foreground);
    for (kind, style) in &theme.colors {
        let _ = write!(css, ".{} {{ {} }}\n", kind.css_class(), style);
    }
    css
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_rust_keyword() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("fn main", &lang);
        let keywords: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Keyword).collect();
        assert_eq!(keywords.len(), 1);
        assert_eq!(keywords[0].text, "fn");
    }

    #[test]
    fn test_tokenize_rust_string() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("let s = \"hello\"", &lang);
        let strings: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::StringLiteral)
            .collect();
        assert_eq!(strings.len(), 1);
        assert_eq!(strings[0].text, "\"hello\"");
    }

    #[test]
    fn test_tokenize_rust_comment() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("// this is a comment", &lang);
        let comments: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Comment).collect();
        assert_eq!(comments.len(), 1);
        assert!(comments[0].text.contains("comment"));
    }

    #[test]
    fn test_tokenize_multiline_comment() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("/* multi\nline */", &lang);
        let comments: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Comment).collect();
        assert_eq!(comments.len(), 1);
        assert!(comments[0].text.contains("multi"));
    }

    #[test]
    fn test_tokenize_number() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("42", &lang);
        let nums: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::NumberLiteral)
            .collect();
        assert_eq!(nums.len(), 1);
        assert_eq!(nums[0].text, "42");
    }

    #[test]
    fn test_tokenize_operator() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("a => b", &lang);
        let ops: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Operator)
            .collect();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].text, "=>");
    }

    #[test]
    fn test_tokenize_type() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("let x: u32", &lang);
        let types: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Type).collect();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].text, "u32");
    }

    #[test]
    fn test_tokenize_function_call() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("foo()", &lang);
        let funcs: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Function)
            .collect();
        assert_eq!(funcs.len(), 1);
        assert_eq!(funcs[0].text, "foo");
    }

    #[test]
    fn test_render_html_css_classes() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("fn main", &lang);
        let config = RenderConfig::default();
        let html = render_html(&tokens, &config, None);
        assert!(html.contains("hl-keyword"));
        assert!(html.contains("<pre>"));
    }

    #[test]
    fn test_render_with_line_numbers() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("let a = 1\nlet b = 2", &lang);
        let config = RenderConfig {
            show_line_numbers: true,
            ..Default::default()
        };
        let html = render_html(&tokens, &config, None);
        assert!(html.contains("hl-line-number"));
    }

    #[test]
    fn test_render_with_inline_styles() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("fn test", &lang);
        let theme = Theme::dark();
        let config = RenderConfig {
            use_inline_styles: true,
            ..Default::default()
        };
        let html = render_html(&tokens, &config, Some(&theme));
        assert!(html.contains("style="));
    }

    #[test]
    fn test_highlight_range() {
        let range = HighlightRange::new(2, 5);
        assert!(range.contains_line(2));
        assert!(range.contains_line(5));
        assert!(!range.contains_line(1));
        assert!(!range.contains_line(6));
    }

    #[test]
    fn test_highlight_convenience() {
        let html = highlight("fn main() {}", "rust", &RenderConfig::default(), None);
        assert!(html.contains("hl-keyword"));
        assert!(html.contains("<pre>"));
    }

    #[test]
    fn test_highlight_unknown_language() {
        let html = highlight("hello", "brainfuck", &RenderConfig::default(), None);
        assert!(html.contains("<pre>"));
        assert!(html.contains("hello"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<div>"), "&lt;div&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
    }

    #[test]
    fn test_language_by_name() {
        assert!(LanguageDefinition::by_name("rust").is_some());
        assert!(LanguageDefinition::by_name("rs").is_some());
        assert!(LanguageDefinition::by_name("js").is_some());
        assert!(LanguageDefinition::by_name("py").is_some());
        assert!(LanguageDefinition::by_name("unknown").is_none());
    }

    #[test]
    fn test_token_kind_css_class() {
        assert_eq!(TokenKind::Keyword.css_class(), "hl-keyword");
        assert_eq!(TokenKind::Comment.css_class(), "hl-comment");
        assert_eq!(TokenKind::StringLiteral.css_class(), "hl-string");
    }

    #[test]
    fn test_generate_theme_css() {
        let theme = Theme::dark();
        let css = generate_theme_css(&theme);
        assert!(css.contains("hl-keyword"));
        assert!(css.contains("hl-dark"));
    }

    #[test]
    fn test_python_tokenizer() {
        let lang = LanguageDefinition::python();
        let tokens = tokenize("def hello():\n    pass", &lang);
        let keywords: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Keyword).collect();
        assert!(keywords.len() >= 2);
        assert_eq!(keywords[0].text, "def");
    }

    #[test]
    fn test_javascript_tokenizer() {
        let lang = LanguageDefinition::javascript();
        let tokens = tokenize("const x = 42", &lang);
        let keywords: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Keyword).collect();
        assert_eq!(keywords.len(), 1);
        assert_eq!(keywords[0].text, "const");
    }

    #[test]
    fn test_c_preprocessor() {
        let lang = LanguageDefinition::c();
        let tokens = tokenize("#include <stdio.h>", &lang);
        let preprocs: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::Preprocessor)
            .collect();
        assert_eq!(preprocs.len(), 1);
    }

    #[test]
    fn test_token_position_tracking() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("fn\nmain", &lang);
        // first token should be line 1
        assert_eq!(tokens[0].line, 1);
        // after newline, next identifier should be line 2
        let main_tok = tokens.iter().find(|t| t.text == "main").unwrap();
        assert_eq!(main_tok.line, 2);
    }

    #[test]
    fn test_dark_and_light_themes() {
        let dark = Theme::dark();
        let light = Theme::light();
        assert_eq!(dark.name, "dark");
        assert_eq!(light.name, "light");
        // Both should have entries for keywords
        assert!(dark.colors.contains_key(&TokenKind::Keyword));
        assert!(light.colors.contains_key(&TokenKind::Keyword));
    }

    #[test]
    fn test_escaped_string_in_token() {
        let lang = LanguageDefinition::rust();
        let tokens = tokenize("\"hello\\nworld\"", &lang);
        let strings: Vec<_> = tokens
            .iter()
            .filter(|t| t.kind == TokenKind::StringLiteral)
            .collect();
        assert_eq!(strings.len(), 1);
        assert!(strings[0].text.contains("\\n"));
    }
}
