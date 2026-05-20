//! Syntax highlighting: token-based highlighter producing styled HTML spans.
//!
//! Replaces Prism.js / highlight.js. Scans source code left-to-right,
//! applies language-specific rules, and emits `<span class="hl-...">` markup.

use std::fmt;

// ── Token types ─────────────────────────────────────────────────

/// Classification of a syntax token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Keyword,
    String_,
    Number,
    Comment,
    Operator,
    Punctuation,
    Function,
    Type,
    Variable,
    Property,
    Attribute,
    Tag,
    Builtin,
    Constant,
    Regex,
    Preprocessor,
    Annotation,
    Plain,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            TokenKind::Keyword => "keyword",
            TokenKind::String_ => "string",
            TokenKind::Number => "number",
            TokenKind::Comment => "comment",
            TokenKind::Operator => "operator",
            TokenKind::Punctuation => "punctuation",
            TokenKind::Function => "function",
            TokenKind::Type => "type",
            TokenKind::Variable => "variable",
            TokenKind::Property => "property",
            TokenKind::Attribute => "attribute",
            TokenKind::Tag => "tag",
            TokenKind::Builtin => "builtin",
            TokenKind::Constant => "constant",
            TokenKind::Regex => "regex",
            TokenKind::Preprocessor => "preprocessor",
            TokenKind::Annotation => "annotation",
            TokenKind::Plain => "plain",
        };
        write!(f, "{name}")
    }
}

// ── Token ───────────────────────────────────────────────────────

/// A single highlighted token.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

// ── Language Rules ──────────────────────────────────────────────

/// A highlighting rule: a pattern (keyword/prefix) mapped to a token kind.
#[derive(Debug, Clone)]
pub struct HighlightRule {
    pub pattern: String,
    pub kind: TokenKind,
}

/// Language definition with ordered rules.
#[derive(Debug, Clone)]
pub struct Language {
    pub name: String,
    pub rules: Vec<HighlightRule>,
}

impl Language {
    pub fn rust() -> Self {
        let keywords = [
            "fn", "let", "mut", "pub", "struct", "enum", "impl", "use", "mod",
            "if", "else", "for", "while", "match", "return", "self", "Self",
            "crate", "super", "where", "async", "await", "trait", "type",
            "const", "static", "ref", "move", "loop", "break", "continue",
            "unsafe", "extern", "dyn", "as", "in",
        ];
        let mut rules: Vec<HighlightRule> = keywords
            .iter()
            .map(|kw| HighlightRule {
                pattern: kw.to_string(),
                kind: TokenKind::Keyword,
            })
            .collect();
        for c in &["true", "false"] {
            rules.push(HighlightRule {
                pattern: c.to_string(),
                kind: TokenKind::Constant,
            });
        }
        Self { name: "rust".into(), rules }
    }

    pub fn javascript() -> Self {
        let keywords = [
            "function", "var", "let", "const", "if", "else", "for", "while",
            "return", "class", "import", "export", "default", "new", "this",
            "typeof", "instanceof", "try", "catch", "finally", "throw",
            "async", "await", "yield", "of", "in",
        ];
        let mut rules: Vec<HighlightRule> = keywords
            .iter()
            .map(|kw| HighlightRule {
                pattern: kw.to_string(),
                kind: TokenKind::Keyword,
            })
            .collect();
        for c in &["true", "false", "null", "undefined", "NaN", "Infinity"] {
            rules.push(HighlightRule {
                pattern: c.to_string(),
                kind: TokenKind::Constant,
            });
        }
        Self { name: "javascript".into(), rules }
    }

    pub fn html() -> Self {
        Self {
            name: "html".into(),
            rules: Vec::new(),
        }
    }

    pub fn css() -> Self {
        let at_rules = ["@media", "@import", "@keyframes", "@font-face", "@charset"];
        let rules = at_rules
            .iter()
            .map(|r| HighlightRule {
                pattern: r.to_string(),
                kind: TokenKind::Preprocessor,
            })
            .collect();
        Self { name: "css".into(), rules }
    }

    pub fn json() -> Self {
        let mut rules = Vec::new();
        for c in &["true", "false", "null"] {
            rules.push(HighlightRule {
                pattern: c.to_string(),
                kind: TokenKind::Constant,
            });
        }
        Self { name: "json".into(), rules }
    }

    pub fn python() -> Self {
        let keywords = [
            "def", "class", "if", "elif", "else", "for", "while", "return",
            "import", "from", "as", "with", "try", "except", "finally",
            "raise", "yield", "lambda", "pass", "break", "continue", "and",
            "or", "not", "is", "in",
        ];
        let mut rules: Vec<HighlightRule> = keywords
            .iter()
            .map(|kw| HighlightRule {
                pattern: kw.to_string(),
                kind: TokenKind::Keyword,
            })
            .collect();
        for c in &["True", "False", "None"] {
            rules.push(HighlightRule {
                pattern: c.to_string(),
                kind: TokenKind::Constant,
            });
        }
        Self { name: "python".into(), rules }
    }
}

// ── Tokenizer ───────────────────────────────────────────────────

/// Tokenize source code using the given language definition.
pub fn tokenize(source: &str, language: &Language) -> Vec<Token> {
    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::new();
    let mut pos = 0;

    while pos < len {
        // Whitespace as Plain.
        if bytes[pos].is_ascii_whitespace() {
            let start = pos;
            while pos < len && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Plain,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Line comments: //
        if pos + 1 < len && bytes[pos] == b'/' && bytes[pos + 1] == b'/' {
            let start = pos;
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Comment,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Python line comments: #
        if bytes[pos] == b'#' && language.name == "python" {
            let start = pos;
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Comment,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Block comments: /* ... */
        if pos + 1 < len && bytes[pos] == b'/' && bytes[pos + 1] == b'*' {
            let start = pos;
            pos += 2;
            while pos + 1 < len && !(bytes[pos] == b'*' && bytes[pos + 1] == b'/') {
                pos += 1;
            }
            if pos + 1 < len {
                pos += 2;
            }
            tokens.push(Token {
                kind: TokenKind::Comment,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // HTML comments: <!-- ... -->
        if language.name == "html" && source[pos..].starts_with("<!--") {
            let start = pos;
            if let Some(end_offset) = source[pos..].find("-->") {
                pos += end_offset + 3;
            } else {
                pos = len;
            }
            tokens.push(Token {
                kind: TokenKind::Comment,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Rust attributes: #[...]
        if language.name == "rust" && bytes[pos] == b'#' && pos + 1 < len && bytes[pos + 1] == b'[' {
            let start = pos;
            pos += 2;
            let mut depth = 1;
            while pos < len && depth > 0 {
                if bytes[pos] == b'[' {
                    depth += 1;
                } else if bytes[pos] == b']' {
                    depth -= 1;
                }
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Annotation,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Python decorators: @...
        if language.name == "python" && bytes[pos] == b'@' {
            let start = pos;
            pos += 1;
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_' || bytes[pos] == b'.') {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Annotation,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // String literals.
        if bytes[pos] == b'"' || bytes[pos] == b'\'' {
            let quote = bytes[pos];
            let start = pos;
            pos += 1;
            // Triple-quote strings (Python).
            if language.name == "python"
                && pos + 1 < len
                && bytes[pos] == quote
                && bytes.get(pos + 1) == Some(&quote)
            {
                pos += 2;
                let end_pat = [quote, quote, quote];
                loop {
                    if pos + 2 >= len {
                        pos = len;
                        break;
                    }
                    if bytes[pos] == end_pat[0]
                        && bytes[pos + 1] == end_pat[1]
                        && bytes[pos + 2] == end_pat[2]
                    {
                        pos += 3;
                        break;
                    }
                    pos += 1;
                }
            } else {
                while pos < len && bytes[pos] != quote {
                    if bytes[pos] == b'\\' {
                        pos += 1;
                    }
                    pos += 1;
                }
                if pos < len {
                    pos += 1;
                }
            }
            tokens.push(Token {
                kind: TokenKind::String_,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Template literals (JavaScript).
        if language.name == "javascript" && bytes[pos] == b'`' {
            let start = pos;
            pos += 1;
            while pos < len && bytes[pos] != b'`' {
                if bytes[pos] == b'\\' {
                    pos += 1;
                }
                pos += 1;
            }
            if pos < len {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::String_,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Numbers.
        if bytes[pos].is_ascii_digit()
            || (bytes[pos] == b'.' && pos + 1 < len && bytes[pos + 1].is_ascii_digit())
        {
            let start = pos;
            if bytes[pos] == b'0' && pos + 1 < len && (bytes[pos + 1] == b'x' || bytes[pos + 1] == b'X') {
                pos += 2;
                while pos < len && (bytes[pos].is_ascii_hexdigit() || bytes[pos] == b'_') {
                    pos += 1;
                }
            } else {
                while pos < len && (bytes[pos].is_ascii_digit() || bytes[pos] == b'.' || bytes[pos] == b'_' || bytes[pos] == b'e' || bytes[pos] == b'E') {
                    pos += 1;
                }
            }
            tokens.push(Token {
                kind: TokenKind::Number,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // HTML tags.
        if language.name == "html" && bytes[pos] == b'<' {
            let start = pos;
            while pos < len && bytes[pos] != b'>' {
                pos += 1;
            }
            if pos < len {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Tag,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // CSS hex colors.
        if language.name == "css" && bytes[pos] == b'#' {
            let start = pos;
            pos += 1;
            while pos < len && bytes[pos].is_ascii_hexdigit() {
                pos += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Number,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // JSON structural characters.
        if language.name == "json" && matches!(bytes[pos], b'{' | b'}' | b'[' | b']' | b':' | b',') {
            let start = pos;
            pos += 1;
            tokens.push(Token {
                kind: TokenKind::Punctuation,
                text: source[start..pos].to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Identifiers / keywords.
        if bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_' {
            let start = pos;
            while pos < len && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            let word = &source[start..pos];

            let mut kind = TokenKind::Plain;
            for rule in &language.rules {
                if rule.pattern == word {
                    kind = rule.kind;
                    break;
                }
            }

            tokens.push(Token {
                kind,
                text: word.to_string(),
                start,
                end: pos,
            });
            continue;
        }

        // Operators and punctuation.
        let start = pos;
        let ch = bytes[pos];
        pos += 1;
        let kind = match ch {
            b'+' | b'-' | b'*' | b'/' | b'%' | b'=' | b'!' | b'<' | b'>'
            | b'&' | b'|' | b'^' | b'~' => TokenKind::Operator,
            b'(' | b')' | b'{' | b'}' | b'[' | b']' | b';' | b',' | b'.' | b':' => {
                TokenKind::Punctuation
            }
            _ => TokenKind::Plain,
        };
        tokens.push(Token {
            kind,
            text: source[start..pos].to_string(),
            start,
            end: pos,
        });
    }

    tokens
}

// ── Rendering ───────────────────────────────────────────────────

fn escape_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Render tokens as HTML with `<span class="hl-{kind}">` wrappers.
pub fn render_html(tokens: &[Token]) -> String {
    let mut out = String::from("<pre class=\"highlight\"><code>");
    for token in tokens {
        let escaped = escape_html(&token.text);
        if token.kind == TokenKind::Plain {
            out.push_str(&escaped);
        } else {
            out.push_str(&format!("<span class=\"hl-{}\">", token.kind));
            out.push_str(&escaped);
            out.push_str("</span>");
        }
    }
    out.push_str("</code></pre>");
    out
}

/// Return a default CSS stylesheet for all token kinds.
pub fn render_classes() -> String {
    r#".hl-keyword { color: #c678dd; font-weight: bold; }
.hl-string { color: #98c379; }
.hl-number { color: #d19a66; }
.hl-comment { color: #5c6370; font-style: italic; }
.hl-operator { color: #56b6c2; }
.hl-punctuation { color: #abb2bf; }
.hl-function { color: #61afef; }
.hl-type { color: #e5c07b; }
.hl-variable { color: #e06c75; }
.hl-property { color: #e06c75; }
.hl-attribute { color: #d19a66; }
.hl-tag { color: #e06c75; }
.hl-builtin { color: #56b6c2; }
.hl-constant { color: #d19a66; }
.hl-regex { color: #98c379; }
.hl-preprocessor { color: #c678dd; }
.hl-annotation { color: #d19a66; }
.hl-plain { color: #abb2bf; }"#
        .to_string()
}

// ── High-level API ──────────────────────────────────────────────

/// Highlighting configuration.
pub struct HighlightConfig {
    pub language: Language,
    pub line_numbers: bool,
    pub wrap_lines: bool,
}

/// Highlight source code and return HTML.
pub fn highlight(source: &str, config: &HighlightConfig) -> String {
    let tokens = tokenize(source, &config.language);
    let mut html = render_html(&tokens);

    if config.line_numbers {
        let code_start = "<pre class=\"highlight\"><code>";
        let code_end = "</code></pre>";
        if let Some(start_idx) = html.find(code_start) {
            if let Some(end_idx) = html.rfind(code_end) {
                let inner = &html[start_idx + code_start.len()..end_idx];
                let lines: Vec<&str> = inner.split('\n').collect();
                let mut numbered = String::from(code_start);
                for (i, line) in lines.iter().enumerate() {
                    numbered.push_str(&format!(
                        "<span class=\"line-number\">{:>4}</span> {}\n",
                        i + 1,
                        line
                    ));
                }
                if numbered.ends_with('\n') {
                    numbered.pop();
                }
                numbered.push_str(code_end);
                html = numbered;
            }
        }
    }

    html
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_rust_fn_keyword() {
        let tokens = tokenize("fn main", &Language::rust());
        let kw = tokens.iter().find(|t| t.text == "fn").unwrap();
        assert_eq!(kw.kind, TokenKind::Keyword);
    }

    #[test]
    fn tokenize_string_literal() {
        let tokens = tokenize("let s = \"hello\";", &Language::rust());
        let s = tokens.iter().find(|t| t.kind == TokenKind::String_).unwrap();
        assert_eq!(s.text, "\"hello\"");
    }

    #[test]
    fn tokenize_comment() {
        let tokens = tokenize("// comment\nlet x = 1;", &Language::rust());
        let c = tokens.iter().find(|t| t.kind == TokenKind::Comment).unwrap();
        assert!(c.text.starts_with("//"));
    }

    #[test]
    fn javascript_const_keyword() {
        let tokens = tokenize("const x = 42;", &Language::javascript());
        let kw = tokens.iter().find(|t| t.text == "const").unwrap();
        assert_eq!(kw.kind, TokenKind::Keyword);
    }

    #[test]
    fn html_tag_detection() {
        let tokens = tokenize("<div class=\"main\">Hello</div>", &Language::html());
        let tag_tokens: Vec<_> = tokens.iter().filter(|t| t.kind == TokenKind::Tag).collect();
        assert!(!tag_tokens.is_empty());
        assert!(tag_tokens[0].text.starts_with('<'));
    }

    #[test]
    fn css_property_tokenization() {
        let tokens = tokenize("color: #ff0000;", &Language::css());
        let hex = tokens.iter().find(|t| t.text.starts_with('#')).unwrap();
        assert_eq!(hex.kind, TokenKind::Number);
    }

    #[test]
    fn json_string_number() {
        let tokens = tokenize("{\"key\": 42}", &Language::json());
        let s = tokens.iter().find(|t| t.kind == TokenKind::String_).unwrap();
        assert_eq!(s.text, "\"key\"");
        let n = tokens.iter().find(|t| t.kind == TokenKind::Number).unwrap();
        assert_eq!(n.text, "42");
    }

    #[test]
    fn render_html_wraps_in_spans() {
        let tokens = vec![Token {
            kind: TokenKind::Keyword,
            text: "fn".into(),
            start: 0,
            end: 2,
        }];
        let html = render_html(&tokens);
        assert!(html.contains("<span class=\"hl-keyword\">fn</span>"));
        assert!(html.starts_with("<pre class=\"highlight\"><code>"));
        assert!(html.ends_with("</code></pre>"));
    }

    #[test]
    fn line_numbers_added_when_configured() {
        let config = HighlightConfig {
            language: Language::rust(),
            line_numbers: true,
            wrap_lines: false,
        };
        let html = highlight("fn main() {\n    println!(\"hi\");\n}", &config);
        assert!(html.contains("class=\"line-number\""));
        assert!(html.contains("   1"));
    }

    #[test]
    fn python_triple_quote_string() {
        let tokens = tokenize("x = \"\"\"multi\nline\"\"\"", &Language::python());
        let s = tokens.iter().find(|t| t.kind == TokenKind::String_).unwrap();
        assert!(s.text.contains("multi"));
        assert!(s.text.contains("line"));
    }

    #[test]
    fn multiple_tokens_in_sequence() {
        let tokens = tokenize("let x = 42;", &Language::rust());
        assert!(tokens.len() >= 4);
        let kinds: Vec<_> = tokens.iter().filter(|t| t.kind != TokenKind::Plain).map(|t| t.kind).collect();
        assert!(kinds.contains(&TokenKind::Keyword));
        assert!(kinds.contains(&TokenKind::Number));
    }

    #[test]
    fn empty_input() {
        let tokens = tokenize("", &Language::rust());
        assert!(tokens.is_empty());
    }

    #[test]
    fn rust_block_comment() {
        let tokens = tokenize("/* block */", &Language::rust());
        let c = tokens.iter().find(|t| t.kind == TokenKind::Comment).unwrap();
        assert_eq!(c.text, "/* block */");
    }
}
