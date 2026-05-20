//! Generic lexer / scanner — character stream with configurable keywords,
//! span tracking (line:col), token types (ident, number, string, operator,
//! keyword), peek/advance, string escape handling, and comment skipping.

use std::collections::HashSet;
use std::fmt;

// ── Span ────────────────────────────────────────────────────────────────────

/// Source location tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Byte offset of the start of the span.
    pub start: usize,
    /// Byte offset past the end of the span.
    pub end: usize,
    /// 1-based line number where the span starts.
    pub line: usize,
    /// 1-based column number where the span starts.
    pub col: usize,
}

impl Span {
    pub fn new(start: usize, end: usize, line: usize, col: usize) -> Self {
        Self { start, end, line, col }
    }

    /// Return the length of the span in bytes.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// True if the span covers zero bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

// ── Token types ─────────────────────────────────────────────────────────────

/// The kind of a lexed token.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    /// An identifier (variable name, function name, etc.).
    Ident(String),
    /// A keyword (recognized from the configured keyword set).
    Keyword(String),
    /// An integer literal.
    Integer(i64),
    /// A floating-point literal.
    Float(f64),
    /// A string literal (contents after escape processing).
    StringLit(String),
    /// A character literal.
    CharLit(char),
    /// An operator or punctuation symbol (e.g. `+`, `==`, `{`).
    Operator(String),
    /// A boolean literal.
    Bool(bool),
    /// End of input.
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ident(s) => write!(f, "ident({s})"),
            Self::Keyword(s) => write!(f, "keyword({s})"),
            Self::Integer(n) => write!(f, "int({n})"),
            Self::Float(n) => write!(f, "float({n})"),
            Self::StringLit(s) => write!(f, "string({s})"),
            Self::CharLit(c) => write!(f, "char({c})"),
            Self::Operator(s) => write!(f, "op({s})"),
            Self::Bool(b) => write!(f, "bool({b})"),
            Self::Eof => write!(f, "EOF"),
        }
    }
}

/// A token with its span.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// The literal source text that produced this token.
    pub lexeme: String,
}

// ── Lexer errors ────────────────────────────────────────────────────────────

/// Errors produced during lexing.
#[derive(Debug, Clone, PartialEq)]
pub enum LexError {
    /// Unexpected character at position.
    UnexpectedChar(char, Span),
    /// Unterminated string literal.
    UnterminatedString(Span),
    /// Unterminated character literal.
    UnterminatedChar(Span),
    /// Invalid escape sequence.
    InvalidEscape(char, Span),
    /// Invalid number literal.
    InvalidNumber(String, Span),
    /// Unterminated block comment.
    UnterminatedBlockComment(Span),
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedChar(c, sp) => write!(f, "unexpected char '{c}' at {sp}"),
            Self::UnterminatedString(sp) => write!(f, "unterminated string at {sp}"),
            Self::UnterminatedChar(sp) => write!(f, "unterminated char at {sp}"),
            Self::InvalidEscape(c, sp) => write!(f, "invalid escape '\\{c}' at {sp}"),
            Self::InvalidNumber(s, sp) => write!(f, "invalid number '{s}' at {sp}"),
            Self::UnterminatedBlockComment(sp) => {
                write!(f, "unterminated block comment at {sp}")
            }
        }
    }
}

// ── Comment style ───────────────────────────────────────────────────────────

/// How line comments start.
#[derive(Debug, Clone)]
pub enum LineCommentStyle {
    /// `//`
    DoubleSlash,
    /// `#`
    Hash,
    /// `--`
    DoubleDash,
    /// Custom prefix.
    Custom(String),
}

/// Block comment delimiters.
#[derive(Debug, Clone)]
pub struct BlockCommentStyle {
    pub open: String,
    pub close: String,
}

impl BlockCommentStyle {
    pub fn c_style() -> Self {
        Self {
            open: "/*".into(),
            close: "*/".into(),
        }
    }
}

// ── Lexer config ────────────────────────────────────────────────────────────

/// Configuration for the lexer.
#[derive(Debug, Clone)]
pub struct LexerConfig {
    /// Set of reserved keywords (turned into `TokenKind::Keyword`).
    pub keywords: HashSet<String>,
    /// Multi-character operators, sorted longest first for greedy matching.
    pub operators: Vec<String>,
    /// Single-character operators / punctuation.
    pub single_operators: HashSet<char>,
    /// Optional line comment style.
    pub line_comment: Option<LineCommentStyle>,
    /// Optional block comment style.
    pub block_comment: Option<BlockCommentStyle>,
    /// Whether to recognise `true` / `false` as boolean literals.
    pub bool_literals: bool,
}

impl Default for LexerConfig {
    fn default() -> Self {
        let keywords: HashSet<String> = [
            "let", "fn", "if", "else", "while", "for", "return", "true", "false",
            "struct", "enum", "match", "break", "continue", "const", "mut",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        let mut ops = vec![
            "==", "!=", "<=", ">=", "&&", "||", "->", "=>", "::", "..", "<<", ">>",
            "+=", "-=", "*=", "/=",
        ];
        ops.sort_by(|a, b| b.len().cmp(&a.len()));
        let operators: Vec<String> = ops.into_iter().map(String::from).collect();

        let single_operators: HashSet<char> =
            "+-*/%=<>!&|^~?.,:;(){}[]@".chars().collect();

        Self {
            keywords,
            operators,
            single_operators,
            line_comment: Some(LineCommentStyle::DoubleSlash),
            block_comment: Some(BlockCommentStyle::c_style()),
            bool_literals: true,
        }
    }
}

// ── Lexer ───────────────────────────────────────────────────────────────────

/// A configurable lexer that converts source text into tokens.
pub struct Lexer {
    /// Source characters.
    chars: Vec<char>,
    /// Current byte-offset into `chars`.
    pos: usize,
    /// Current 1-based line.
    line: usize,
    /// Current 1-based column.
    col: usize,
    /// Configuration.
    config: LexerConfig,
    /// One-token lookahead buffer.
    peeked: Option<Token>,
}

impl Lexer {
    /// Create a new lexer over `source` with the given config.
    pub fn new(source: &str, config: LexerConfig) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
            config,
            peeked: None,
        }
    }

    /// Create a lexer with the default configuration.
    pub fn with_defaults(source: &str) -> Self {
        Self::new(source, LexerConfig::default())
    }

    // ── Character-level helpers ─────────────────────────────────────────

    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn current(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_char(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn advance_char(&mut self) -> Option<char> {
        let c = self.current()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn starts_with_at(&self, pos: usize, prefix: &str) -> bool {
        let pchars: Vec<char> = prefix.chars().collect();
        if pos + pchars.len() > self.chars.len() {
            return false;
        }
        for (i, pc) in pchars.iter().enumerate() {
            if self.chars[pos + i] != *pc {
                return false;
            }
        }
        true
    }

    fn make_span(&self, start: usize, start_line: usize, start_col: usize) -> Span {
        Span::new(start, self.pos, start_line, start_col)
    }

    // ── Whitespace & comments ───────────────────────────────────────────

    fn skip_whitespace_and_comments(&mut self) -> Result<(), LexError> {
        loop {
            // Skip whitespace
            while let Some(c) = self.current() {
                if c.is_whitespace() {
                    self.advance_char();
                } else {
                    break;
                }
            }
            if self.at_end() {
                return Ok(());
            }

            // Try line comment
            if let Some(ref style) = self.config.line_comment.clone() {
                let prefix = match style {
                    LineCommentStyle::DoubleSlash => "//",
                    LineCommentStyle::Hash => "#",
                    LineCommentStyle::DoubleDash => "--",
                    LineCommentStyle::Custom(s) => s.as_str(),
                };
                if self.starts_with_at(self.pos, prefix) {
                    while let Some(c) = self.current() {
                        if c == '\n' {
                            break;
                        }
                        self.advance_char();
                    }
                    continue;
                }
            }

            // Try block comment
            if let Some(ref style) = self.config.block_comment.clone() {
                let open = style.open.clone();
                let close = style.close.clone();
                if self.starts_with_at(self.pos, &open) {
                    let start_line = self.line;
                    let start_col = self.col;
                    let start_pos = self.pos;
                    for _ in 0..open.len() {
                        self.advance_char();
                    }
                    let mut depth = 1u32;
                    while !self.at_end() && depth > 0 {
                        if self.starts_with_at(self.pos, &close) {
                            for _ in 0..close.len() {
                                self.advance_char();
                            }
                            depth -= 1;
                        } else if self.starts_with_at(self.pos, &open) {
                            for _ in 0..open.len() {
                                self.advance_char();
                            }
                            depth += 1;
                        } else {
                            self.advance_char();
                        }
                    }
                    if depth > 0 {
                        return Err(LexError::UnterminatedBlockComment(Span::new(
                            start_pos, self.pos, start_line, start_col,
                        )));
                    }
                    continue;
                }
            }

            break;
        }
        Ok(())
    }

    // ── Number ──────────────────────────────────────────────────────────

    fn lex_number(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.col;
        let mut buf = String::new();
        let mut is_float = false;

        // Hex: 0x...
        if self.current() == Some('0')
            && (self.peek_char() == Some('x') || self.peek_char() == Some('X'))
        {
            buf.push(self.advance_char().unwrap());
            buf.push(self.advance_char().unwrap());
            while let Some(c) = self.current() {
                if c.is_ascii_hexdigit() || c == '_' {
                    buf.push(self.advance_char().unwrap());
                } else {
                    break;
                }
            }
            let clean: String = buf[2..].chars().filter(|c| *c != '_').collect();
            let span = self.make_span(start, start_line, start_col);
            let val = i64::from_str_radix(&clean, 16)
                .map_err(|_| LexError::InvalidNumber(buf.clone(), span))?;
            return Ok(Token {
                kind: TokenKind::Integer(val),
                span,
                lexeme: buf,
            });
        }

        // Binary: 0b...
        if self.current() == Some('0')
            && (self.peek_char() == Some('b') || self.peek_char() == Some('B'))
        {
            buf.push(self.advance_char().unwrap());
            buf.push(self.advance_char().unwrap());
            while let Some(c) = self.current() {
                if c == '0' || c == '1' || c == '_' {
                    buf.push(self.advance_char().unwrap());
                } else {
                    break;
                }
            }
            let clean: String = buf[2..].chars().filter(|c| *c != '_').collect();
            let span = self.make_span(start, start_line, start_col);
            let val = i64::from_str_radix(&clean, 2)
                .map_err(|_| LexError::InvalidNumber(buf.clone(), span))?;
            return Ok(Token {
                kind: TokenKind::Integer(val),
                span,
                lexeme: buf,
            });
        }

        while let Some(c) = self.current() {
            if c.is_ascii_digit() || c == '_' {
                buf.push(self.advance_char().unwrap());
            } else if c == '.' && !is_float {
                // Check the char after the dot — if it's a digit, it's a float
                if self.peek_char().is_some_and(|nc| nc.is_ascii_digit()) {
                    is_float = true;
                    buf.push(self.advance_char().unwrap()); // '.'
                } else {
                    break;
                }
            } else if (c == 'e' || c == 'E') && !buf.is_empty() {
                is_float = true;
                buf.push(self.advance_char().unwrap());
                if let Some(sign) = self.current() {
                    if sign == '+' || sign == '-' {
                        buf.push(self.advance_char().unwrap());
                    }
                }
            } else {
                break;
            }
        }

        let span = self.make_span(start, start_line, start_col);
        let clean: String = buf.chars().filter(|c| *c != '_').collect();
        if is_float {
            let val: f64 = clean
                .parse()
                .map_err(|_| LexError::InvalidNumber(buf.clone(), span))?;
            Ok(Token {
                kind: TokenKind::Float(val),
                span,
                lexeme: buf,
            })
        } else {
            let val: i64 = clean
                .parse()
                .map_err(|_| LexError::InvalidNumber(buf.clone(), span))?;
            Ok(Token {
                kind: TokenKind::Integer(val),
                span,
                lexeme: buf,
            })
        }
    }

    // ── String ──────────────────────────────────────────────────────────

    fn lex_string(&mut self, quote: char) -> Result<Token, LexError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.col;
        self.advance_char(); // consume opening quote
        let mut val = String::new();
        let mut raw = String::new();
        raw.push(quote);

        loop {
            match self.current() {
                None => {
                    return Err(LexError::UnterminatedString(self.make_span(
                        start, start_line, start_col,
                    )));
                }
                Some(c) if c == quote => {
                    raw.push(c);
                    self.advance_char();
                    break;
                }
                Some('\\') => {
                    raw.push('\\');
                    self.advance_char();
                    match self.current() {
                        None => {
                            return Err(LexError::UnterminatedString(self.make_span(
                                start, start_line, start_col,
                            )));
                        }
                        Some(esc) => {
                            raw.push(esc);
                            self.advance_char();
                            match esc {
                                'n' => val.push('\n'),
                                't' => val.push('\t'),
                                'r' => val.push('\r'),
                                '\\' => val.push('\\'),
                                '0' => val.push('\0'),
                                c if c == quote => val.push(c),
                                'u' => {
                                    // \uXXXX
                                    let mut hex = String::new();
                                    if self.current() == Some('{') {
                                        raw.push('{');
                                        self.advance_char();
                                        while let Some(hc) = self.current() {
                                            if hc == '}' {
                                                raw.push('}');
                                                self.advance_char();
                                                break;
                                            }
                                            hex.push(hc);
                                            raw.push(hc);
                                            self.advance_char();
                                        }
                                    } else {
                                        for _ in 0..4 {
                                            if let Some(hc) = self.current() {
                                                hex.push(hc);
                                                raw.push(hc);
                                                self.advance_char();
                                            }
                                        }
                                    }
                                    let cp = u32::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
                                    val.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                }
                                other => {
                                    return Err(LexError::InvalidEscape(
                                        other,
                                        self.make_span(start, start_line, start_col),
                                    ));
                                }
                            }
                        }
                    }
                }
                Some(c) => {
                    raw.push(c);
                    val.push(c);
                    self.advance_char();
                }
            }
        }

        let span = self.make_span(start, start_line, start_col);
        Ok(Token {
            kind: TokenKind::StringLit(val),
            span,
            lexeme: raw,
        })
    }

    // ── Char literal ────────────────────────────────────────────────────

    fn lex_char(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.col;
        self.advance_char(); // consume '
        let ch = match self.current() {
            None => {
                return Err(LexError::UnterminatedChar(self.make_span(
                    start, start_line, start_col,
                )));
            }
            Some('\\') => {
                self.advance_char();
                match self.current() {
                    Some('n') => { self.advance_char(); '\n' }
                    Some('t') => { self.advance_char(); '\t' }
                    Some('r') => { self.advance_char(); '\r' }
                    Some('\\') => { self.advance_char(); '\\' }
                    Some('0') => { self.advance_char(); '\0' }
                    Some('\'') => { self.advance_char(); '\'' }
                    _ => {
                        return Err(LexError::UnterminatedChar(self.make_span(
                            start, start_line, start_col,
                        )));
                    }
                }
            }
            Some(c) => {
                self.advance_char();
                c
            }
        };
        if self.current() != Some('\'') {
            return Err(LexError::UnterminatedChar(self.make_span(
                start, start_line, start_col,
            )));
        }
        self.advance_char(); // consume closing '
        let span = self.make_span(start, start_line, start_col);
        Ok(Token {
            kind: TokenKind::CharLit(ch),
            span,
            lexeme: format!("'{ch}'"),
        })
    }

    // ── Ident / keyword ─────────────────────────────────────────────────

    fn lex_ident(&mut self) -> Token {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.col;
        let mut buf = String::new();
        while let Some(c) = self.current() {
            if c.is_alphanumeric() || c == '_' {
                buf.push(c);
                self.advance_char();
            } else {
                break;
            }
        }
        let span = self.make_span(start, start_line, start_col);

        // Check booleans
        if self.config.bool_literals {
            if buf == "true" {
                return Token { kind: TokenKind::Bool(true), span, lexeme: buf };
            }
            if buf == "false" {
                return Token { kind: TokenKind::Bool(false), span, lexeme: buf };
            }
        }

        if self.config.keywords.contains(&buf) {
            Token { kind: TokenKind::Keyword(buf.clone()), span, lexeme: buf }
        } else {
            Token { kind: TokenKind::Ident(buf.clone()), span, lexeme: buf }
        }
    }

    // ── Operator ────────────────────────────────────────────────────────

    fn lex_operator(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.col;

        // Try multi-char operators (longest first)
        let ops = self.config.operators.clone();
        for op in &ops {
            if self.starts_with_at(self.pos, op) {
                for _ in 0..op.len() {
                    self.advance_char();
                }
                let span = self.make_span(start, start_line, start_col);
                return Ok(Token {
                    kind: TokenKind::Operator(op.clone()),
                    span,
                    lexeme: op.clone(),
                });
            }
        }

        // Single-char operator
        if let Some(c) = self.current() {
            if self.config.single_operators.contains(&c) {
                self.advance_char();
                let s = c.to_string();
                let span = self.make_span(start, start_line, start_col);
                return Ok(Token {
                    kind: TokenKind::Operator(s.clone()),
                    span,
                    lexeme: s,
                });
            }
        }

        let c = self.current().unwrap_or('\0');
        Err(LexError::UnexpectedChar(c, self.make_span(start, start_line, start_col)))
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Peek at the next token without consuming it.
    pub fn peek(&mut self) -> Result<&Token, LexError> {
        if self.peeked.is_none() {
            let tok = self.next_token_inner()?;
            self.peeked = Some(tok);
        }
        Ok(self.peeked.as_ref().unwrap())
    }

    /// Advance and return the next token.
    pub fn next_token(&mut self) -> Result<Token, LexError> {
        if let Some(tok) = self.peeked.take() {
            return Ok(tok);
        }
        self.next_token_inner()
    }

    fn next_token_inner(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace_and_comments()?;
        if self.at_end() {
            let span = Span::new(self.pos, self.pos, self.line, self.col);
            return Ok(Token {
                kind: TokenKind::Eof,
                span,
                lexeme: String::new(),
            });
        }

        let c = self.current().unwrap();

        // String
        if c == '"' {
            return self.lex_string('"');
        }

        // Char literal — only if next char exists and it's not an ident start following '
        if c == '\'' {
            // Peek: if after the quote we see a char then another quote, it's a char literal
            let remaining = self.chars.len() - self.pos;
            if remaining >= 3 {
                let c1 = self.chars[self.pos + 1];
                if c1 == '\\' || self.chars.get(self.pos + 2) == Some(&'\'') {
                    return self.lex_char();
                }
            }
            // Fall through to operator
        }

        // Number
        if c.is_ascii_digit() {
            return self.lex_number();
        }

        // Ident / keyword
        if c.is_alphabetic() || c == '_' {
            return Ok(self.lex_ident());
        }

        // Operator / punctuation
        self.lex_operator()
    }

    /// Consume all remaining tokens and return them.
    pub fn tokenize_all(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            if tok.kind == TokenKind::Eof {
                tokens.push(tok);
                break;
            }
            tokens.push(tok);
        }
        Ok(tokens)
    }

    /// Current 1-based line number.
    pub fn current_line(&self) -> usize {
        self.line
    }

    /// Current 1-based column number.
    pub fn current_col(&self) -> usize {
        self.col
    }

    /// Returns true if the lexer has reached end of input.
    pub fn is_at_end(&self) -> bool {
        self.peeked.is_none() && self.at_end()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Vec<Token> {
        Lexer::with_defaults(src).tokenize_all().unwrap()
    }

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty_input() {
        let toks = lex("");
        assert_eq!(toks.len(), 1);
        assert_eq!(toks[0].kind, TokenKind::Eof);
    }

    #[test]
    fn test_integer_literals() {
        let k = kinds("42 0 100_000");
        assert_eq!(k[0], TokenKind::Integer(42));
        assert_eq!(k[1], TokenKind::Integer(0));
        assert_eq!(k[2], TokenKind::Integer(100_000));
    }

    #[test]
    fn test_hex_literal() {
        let k = kinds("0xFF 0x10");
        assert_eq!(k[0], TokenKind::Integer(255));
        assert_eq!(k[1], TokenKind::Integer(16));
    }

    #[test]
    fn test_binary_literal() {
        let k = kinds("0b1010 0b0001");
        assert_eq!(k[0], TokenKind::Integer(10));
        assert_eq!(k[1], TokenKind::Integer(1));
    }

    #[test]
    fn test_float_literals() {
        let k = kinds("3.14 1e10 2.5e-3");
        assert!(matches!(k[0], TokenKind::Float(v) if (v - 3.14).abs() < 1e-10));
        assert!(matches!(k[1], TokenKind::Float(v) if (v - 1e10).abs() < 1.0));
        assert!(matches!(k[2], TokenKind::Float(v) if (v - 2.5e-3).abs() < 1e-10));
    }

    #[test]
    fn test_string_literal() {
        let k = kinds("\"hello world\"");
        assert_eq!(k[0], TokenKind::StringLit("hello world".into()));
    }

    #[test]
    fn test_string_escapes() {
        let k = kinds("\"line\\nnewline\" \"tab\\there\"");
        assert_eq!(k[0], TokenKind::StringLit("line\nnewline".into()));
        assert_eq!(k[1], TokenKind::StringLit("tab\there".into()));
    }

    #[test]
    fn test_unicode_escape() {
        let k = kinds("\"\\u{1F600}\"");
        // U+1F600 is a smiley emoji
        assert!(matches!(&k[0], TokenKind::StringLit(s) if s.contains('\u{1F600}')));
    }

    #[test]
    fn test_char_literal() {
        let k = kinds("'a' 'Z'");
        assert_eq!(k[0], TokenKind::CharLit('a'));
        assert_eq!(k[1], TokenKind::CharLit('Z'));
    }

    #[test]
    fn test_identifiers() {
        let k = kinds("foo bar_baz _priv x1");
        assert_eq!(k[0], TokenKind::Ident("foo".into()));
        assert_eq!(k[1], TokenKind::Ident("bar_baz".into()));
        assert_eq!(k[2], TokenKind::Ident("_priv".into()));
        assert_eq!(k[3], TokenKind::Ident("x1".into()));
    }

    #[test]
    fn test_keywords() {
        let k = kinds("let fn if else return");
        assert_eq!(k[0], TokenKind::Keyword("let".into()));
        assert_eq!(k[1], TokenKind::Keyword("fn".into()));
        assert_eq!(k[2], TokenKind::Keyword("if".into()));
        assert_eq!(k[3], TokenKind::Keyword("else".into()));
        assert_eq!(k[4], TokenKind::Keyword("return".into()));
    }

    #[test]
    fn test_boolean_literals() {
        let k = kinds("true false");
        assert_eq!(k[0], TokenKind::Bool(true));
        assert_eq!(k[1], TokenKind::Bool(false));
    }

    #[test]
    fn test_operators() {
        let k = kinds("+ - * == != <=");
        assert_eq!(k[0], TokenKind::Operator("+".into()));
        assert_eq!(k[1], TokenKind::Operator("-".into()));
        assert_eq!(k[2], TokenKind::Operator("*".into()));
        assert_eq!(k[3], TokenKind::Operator("==".into()));
        assert_eq!(k[4], TokenKind::Operator("!=".into()));
        assert_eq!(k[5], TokenKind::Operator("<=".into()));
    }

    #[test]
    fn test_punctuation() {
        let k = kinds("( ) { } [ ] ; ,");
        assert_eq!(k[0], TokenKind::Operator("(".into()));
        assert_eq!(k[1], TokenKind::Operator(")".into()));
        assert_eq!(k[2], TokenKind::Operator("{".into()));
        assert_eq!(k[3], TokenKind::Operator("}".into()));
    }

    #[test]
    fn test_line_comments() {
        let k = kinds("42 // this is a comment\n99");
        assert_eq!(k[0], TokenKind::Integer(42));
        assert_eq!(k[1], TokenKind::Integer(99));
    }

    #[test]
    fn test_block_comments() {
        let k = kinds("1 /* block */ 2");
        assert_eq!(k[0], TokenKind::Integer(1));
        assert_eq!(k[1], TokenKind::Integer(2));
    }

    #[test]
    fn test_nested_block_comments() {
        let k = kinds("a /* outer /* inner */ still comment */ b");
        assert_eq!(k[0], TokenKind::Ident("a".into()));
        assert_eq!(k[1], TokenKind::Ident("b".into()));
    }

    #[test]
    fn test_span_tracking() {
        let toks = lex("let x = 42");
        assert_eq!(toks[0].span.line, 1);
        assert_eq!(toks[0].span.col, 1);
        assert_eq!(toks[1].span.col, 5); // "x" starts at col 5
    }

    #[test]
    fn test_multiline_span() {
        let toks = lex("a\nb\nc");
        assert_eq!(toks[0].span.line, 1);
        assert_eq!(toks[1].span.line, 2);
        assert_eq!(toks[2].span.line, 3);
    }

    #[test]
    fn test_peek_does_not_consume() {
        let mut lexer = Lexer::with_defaults("abc 123");
        let p = lexer.peek().unwrap().clone();
        let n = lexer.next_token().unwrap();
        assert_eq!(p.kind, n.kind);
        assert_eq!(p.kind, TokenKind::Ident("abc".into()));
    }

    #[test]
    fn test_unterminated_string_error() {
        let mut lexer = Lexer::with_defaults("\"hello");
        let err = lexer.next_token().unwrap_err();
        assert!(matches!(err, LexError::UnterminatedString(_)));
    }

    #[test]
    fn test_invalid_escape_error() {
        let mut lexer = Lexer::with_defaults("\"\\q\"");
        let err = lexer.next_token().unwrap_err();
        assert!(matches!(err, LexError::InvalidEscape('q', _)));
    }

    #[test]
    fn test_unterminated_block_comment() {
        let mut lexer = Lexer::with_defaults("/* unterminated");
        let err = lexer.next_token().unwrap_err();
        assert!(matches!(err, LexError::UnterminatedBlockComment(_)));
    }

    #[test]
    fn test_custom_keywords() {
        let mut cfg = LexerConfig::default();
        cfg.keywords.clear();
        cfg.keywords.insert("SELECT".into());
        cfg.keywords.insert("FROM".into());
        cfg.bool_literals = false;
        let mut lexer = Lexer::new("SELECT name FROM users", cfg);
        let toks = lexer.tokenize_all().unwrap();
        assert_eq!(toks[0].kind, TokenKind::Keyword("SELECT".into()));
        assert_eq!(toks[1].kind, TokenKind::Ident("name".into()));
        assert_eq!(toks[2].kind, TokenKind::Keyword("FROM".into()));
        assert_eq!(toks[3].kind, TokenKind::Ident("users".into()));
    }

    #[test]
    fn test_hash_comment_style() {
        let mut cfg = LexerConfig::default();
        cfg.line_comment = Some(LineCommentStyle::Hash);
        let mut lexer = Lexer::new("42 # comment\n99", cfg);
        let toks = lexer.tokenize_all().unwrap();
        assert_eq!(toks[0].kind, TokenKind::Integer(42));
        assert_eq!(toks[1].kind, TokenKind::Integer(99));
    }

    #[test]
    fn test_complete_expression() {
        let k = kinds("fn add(a, b) { return a + b; }");
        assert_eq!(k[0], TokenKind::Keyword("fn".into()));
        assert_eq!(k[1], TokenKind::Ident("add".into()));
        assert_eq!(k[2], TokenKind::Operator("(".into()));
        // Verify we get all the expected tokens
        assert!(k.last() == Some(&TokenKind::Eof));
    }

    #[test]
    fn test_arrow_operators() {
        let k = kinds("-> =>");
        assert_eq!(k[0], TokenKind::Operator("->".into()));
        assert_eq!(k[1], TokenKind::Operator("=>".into()));
    }

    #[test]
    fn test_lexeme_preserved() {
        let toks = lex("100_000");
        assert_eq!(toks[0].lexeme, "100_000");
        assert_eq!(toks[0].kind, TokenKind::Integer(100000));
    }
}
