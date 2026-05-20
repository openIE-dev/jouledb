//! Glob pattern matcher for file paths and strings.
//!
//! Supports `*` (any chars except separator), `**` (any including separator),
//! `?` (single char), `[abc]` character classes, `{a,b,c}` alternation,
//! negation `!`, and case-insensitive mode.

use std::fmt;

// ── Types ────────────────────────────────────────────────────────

/// A compiled glob pattern.
#[derive(Debug, Clone)]
pub struct Glob {
    tokens: Vec<Token>,
    negated: bool,
    case_insensitive: bool,
    separator: char,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Literal(String),
    AnyExceptSep,      // *
    AnyIncludingSep,    // **
    SingleChar,         // ?
    CharClass { chars: Vec<(char, char)>, negated: bool },
    Alternation(Vec<Vec<Token>>),
}

/// Options for glob compilation.
#[derive(Debug, Clone)]
pub struct GlobOptions {
    pub case_insensitive: bool,
    pub separator: char,
}

impl Default for GlobOptions {
    fn default() -> Self {
        Self { case_insensitive: false, separator: '/' }
    }
}

// ── Parser ───────────────────────────────────────────────────────

struct GlobParser {
    chars: Vec<char>,
    pos: usize,
    separator: char,
    case_insensitive: bool,
}

impl GlobParser {
    fn new(pattern: &str, separator: char, case_insensitive: bool) -> Self {
        Self { chars: pattern.chars().collect(), pos: 0, separator, case_insensitive }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() { self.pos += 1; }
        c
    }

    fn parse_tokens(&mut self) -> Vec<Token> {
        let mut tokens = vec![];
        while let Some(c) = self.peek() {
            match c {
                '*' => {
                    self.advance();
                    if self.peek() == Some('*') {
                        self.advance();
                        // Consume trailing separator if present
                        if self.peek() == Some(self.separator) {
                            self.advance();
                        }
                        tokens.push(Token::AnyIncludingSep);
                    } else {
                        tokens.push(Token::AnyExceptSep);
                    }
                }
                '?' => { self.advance(); tokens.push(Token::SingleChar); }
                '[' => tokens.push(self.parse_class()),
                '{' => tokens.push(self.parse_alternation()),
                '}' | ',' => break, // End of alternation branch
                '\\' => {
                    self.advance();
                    if let Some(nc) = self.advance() {
                        let s = if self.case_insensitive {
                            nc.to_lowercase().collect()
                        } else {
                            nc.to_string()
                        };
                        tokens.push(Token::Literal(s));
                    }
                }
                _ => {
                    self.advance();
                    let s = if self.case_insensitive {
                        c.to_lowercase().collect()
                    } else {
                        c.to_string()
                    };
                    tokens.push(Token::Literal(s));
                }
            }
        }
        tokens
    }

    fn parse_class(&mut self) -> Token {
        self.advance(); // eat '['
        let negated = if self.peek() == Some('!') || self.peek() == Some('^') {
            self.advance(); true
        } else {
            false
        };
        let mut ranges = vec![];
        while let Some(c) = self.peek() {
            if c == ']' { self.advance(); break; }
            self.advance();
            let ch = if c == '\\' { self.advance().unwrap_or('\\') } else { c };
            if self.peek() == Some('-') {
                self.advance();
                if let Some(end) = self.advance() {
                    ranges.push((ch, end));
                }
            } else {
                ranges.push((ch, ch));
            }
        }
        Token::CharClass { chars: ranges, negated }
    }

    fn parse_alternation(&mut self) -> Token {
        self.advance(); // eat '{'
        let mut branches = vec![];
        loop {
            let branch = self.parse_tokens();
            branches.push(branch);
            match self.peek() {
                Some(',') => { self.advance(); }
                Some('}') => { self.advance(); break; }
                _ => break,
            }
        }
        Token::Alternation(branches)
    }
}

// ── Glob ─────────────────────────────────────────────────────────

impl Glob {
    /// Compile a glob pattern with default options (case-sensitive, `/` separator).
    pub fn new(pattern: &str) -> Self {
        Self::with_options(pattern, GlobOptions::default())
    }

    /// Compile a glob with custom options.
    pub fn with_options(pattern: &str, opts: GlobOptions) -> Self {
        let (pat, negated) = if pattern.starts_with('!') {
            (&pattern[1..], true)
        } else {
            (pattern, false)
        };
        let mut parser = GlobParser::new(pat, opts.separator, opts.case_insensitive);
        let tokens = parser.parse_tokens();
        Self { tokens, negated, case_insensitive: opts.case_insensitive, separator: opts.separator }
    }

    /// Check if the input matches this glob pattern.
    pub fn is_match(&self, input: &str) -> bool {
        let text: Vec<char> = if self.case_insensitive {
            input.to_lowercase().chars().collect()
        } else {
            input.chars().collect()
        };
        let result = match_tokens(&self.tokens, &text, 0, self.separator);
        if self.negated { !result } else { result }
    }
}

fn match_tokens(tokens: &[Token], text: &[char], pos: usize, sep: char) -> bool {
    if tokens.is_empty() {
        return pos == text.len();
    }

    let token = &tokens[0];
    let rest = &tokens[1..];

    match token {
        Token::Literal(s) => {
            let chars: Vec<char> = s.chars().collect();
            if pos + chars.len() > text.len() { return false; }
            for (i, c) in chars.iter().enumerate() {
                if text[pos + i] != *c { return false; }
            }
            match_tokens(rest, text, pos + chars.len(), sep)
        }
        Token::SingleChar => {
            if pos >= text.len() { return false; }
            if text[pos] == sep { return false; }
            match_tokens(rest, text, pos + 1, sep)
        }
        Token::AnyExceptSep => {
            // Try matching 0 to N chars (not separator)
            for end in pos..=text.len() {
                if end > pos && text[end - 1] == sep { break; }
                if match_tokens(rest, text, end, sep) { return true; }
            }
            false
        }
        Token::AnyIncludingSep => {
            // Try matching 0 to N chars (including separator)
            for end in pos..=text.len() {
                if match_tokens(rest, text, end, sep) { return true; }
            }
            false
        }
        Token::CharClass { chars, negated } => {
            if pos >= text.len() { return false; }
            let c = text[pos];
            let in_class = chars.iter().any(|(lo, hi)| c >= *lo && c <= *hi);
            let matched = if *negated { !in_class } else { in_class };
            if !matched { return false; }
            match_tokens(rest, text, pos + 1, sep)
        }
        Token::Alternation(branches) => {
            for branch in branches {
                let mut combined = branch.clone();
                combined.extend_from_slice(rest);
                if match_tokens(&combined, text, pos, sep) { return true; }
            }
            false
        }
    }
}

impl fmt::Display for Glob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Glob({} tokens{})", self.tokens.len(), if self.negated { ", negated" } else { "" })
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal() {
        let g = Glob::new("hello.txt");
        assert!(g.is_match("hello.txt"));
        assert!(!g.is_match("hello.rs"));
    }

    #[test]
    fn test_star_wildcard() {
        let g = Glob::new("*.txt");
        assert!(g.is_match("hello.txt"));
        assert!(g.is_match("world.txt"));
        assert!(!g.is_match("hello.rs"));
        assert!(!g.is_match("dir/hello.txt"));
    }

    #[test]
    fn test_double_star() {
        let g = Glob::new("**/*.txt");
        assert!(g.is_match("hello.txt"));
        assert!(g.is_match("dir/hello.txt"));
        assert!(g.is_match("a/b/c/hello.txt"));
    }

    #[test]
    fn test_question_mark() {
        let g = Glob::new("file?.txt");
        assert!(g.is_match("file1.txt"));
        assert!(g.is_match("fileA.txt"));
        assert!(!g.is_match("file10.txt"));
        assert!(!g.is_match("file.txt"));
    }

    #[test]
    fn test_char_class() {
        let g = Glob::new("file[0-9].txt");
        assert!(g.is_match("file5.txt"));
        assert!(!g.is_match("fileA.txt"));
    }

    #[test]
    fn test_negated_class() {
        let g = Glob::new("file[!0-9].txt");
        assert!(!g.is_match("file5.txt"));
        assert!(g.is_match("fileA.txt"));
    }

    #[test]
    fn test_alternation_braces() {
        let g = Glob::new("*.{rs,toml}");
        assert!(g.is_match("Cargo.toml"));
        assert!(g.is_match("main.rs"));
        assert!(!g.is_match("main.py"));
    }

    #[test]
    fn test_negation() {
        let g = Glob::new("!*.txt");
        assert!(!g.is_match("hello.txt"));
        assert!(g.is_match("hello.rs"));
    }

    #[test]
    fn test_case_insensitive() {
        let g = Glob::with_options("*.TXT", GlobOptions { case_insensitive: true, separator: '/' });
        assert!(g.is_match("hello.txt"));
        assert!(g.is_match("hello.TXT"));
    }

    #[test]
    fn test_path_matching() {
        let g = Glob::new("src/**/*.rs");
        assert!(g.is_match("src/main.rs"));
        assert!(g.is_match("src/lib/util.rs"));
        assert!(!g.is_match("test/main.rs"));
    }

    #[test]
    fn test_complex_pattern() {
        let g = Glob::new("src/{main,lib}/**/*.{rs,toml}");
        assert!(g.is_match("src/main/mod.rs"));
        assert!(g.is_match("src/lib/Cargo.toml"));
        assert!(!g.is_match("src/test/mod.rs"));
    }

    #[test]
    fn test_escaped_chars() {
        let g = Glob::new(r"hello\*.txt");
        assert!(g.is_match("hello*.txt"));
        assert!(!g.is_match("helloX.txt"));
    }
}
