//! ICU MessageFormat parser and formatter.
//!
//! Handles simple substitution `{name}`, number formatting `{count, number}`,
//! plural rules `{count, plural, one{…} other{…}}`, select patterns
//! `{gender, select, male{…} female{…} other{…}}`, nested messages,
//! and escaping with single quotes — pure Rust, no ICU dependency.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from parsing or formatting ICU messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IcuMessageError {
    /// Unexpected end of input while parsing.
    UnexpectedEnd,
    /// Unexpected character at the given position.
    UnexpectedChar { pos: usize, ch: char },
    /// Missing closing brace for a placeholder.
    UnclosedBrace { open_pos: usize },
    /// Unknown format type (not `number`, `plural`, `select`).
    UnknownType(String),
    /// Missing required `other` clause in plural/select.
    MissingOther,
    /// Missing argument in the values map.
    MissingArgument(String),
}

impl fmt::Display for IcuMessageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::UnexpectedChar { pos, ch } => {
                write!(f, "unexpected character '{ch}' at position {pos}")
            }
            Self::UnclosedBrace { open_pos } => {
                write!(f, "unclosed brace opened at position {open_pos}")
            }
            Self::UnknownType(t) => write!(f, "unknown format type: {t}"),
            Self::MissingOther => write!(f, "missing required 'other' clause"),
            Self::MissingArgument(name) => write!(f, "missing argument: {name}"),
        }
    }
}

impl std::error::Error for IcuMessageError {}

// ── Value ───────────────────────────────────────────────────────

/// A value that can be substituted into a message.
#[derive(Debug, Clone)]
pub enum Value {
    /// A string value.
    String(String),
    /// An integer value.
    Integer(i64),
    /// A floating-point value.
    Float(f64),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => {
                if n.fract() == 0.0 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{n}")
                }
            }
        }
    }
}

impl Value {
    /// Return the numeric value as f64, if applicable.
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Integer(n) => Some(*n as f64),
            Self::Float(n) => Some(*n),
            Self::String(_) => None,
        }
    }

    /// Return the integer value if applicable.
    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(*n),
            Self::Float(n) => Some(*n as i64),
            Self::String(_) => None,
        }
    }

    /// Return the string representation for select matching.
    fn as_str_match(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Integer(n) => n.to_string(),
            Self::Float(n) => n.to_string(),
        }
    }
}

// ── AST ─────────────────────────────────────────────────────────

/// A parsed ICU message AST node.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    /// Literal text.
    Text(String),
    /// Simple argument substitution: `{name}`.
    Argument(String),
    /// Number format: `{name, number}`.
    NumberFormat(String),
    /// Plural: `{name, plural, one{…} other{…}}`.
    Plural {
        arg: String,
        clauses: Vec<(PluralSelector, Vec<Node>)>,
    },
    /// Select: `{name, select, val{…} other{…}}`.
    Select {
        arg: String,
        clauses: Vec<(String, Vec<Node>)>,
    },
}

/// Plural selectors.
#[derive(Debug, Clone, PartialEq)]
pub enum PluralSelector {
    /// Exact match: `=0`, `=1`, etc.
    Exact(i64),
    /// Named category.
    Zero,
    One,
    Two,
    Few,
    Many,
    Other,
}

// ── Parser ──────────────────────────────────────────────────────

/// Parser state.
struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn parse_message(&mut self, stop_at_brace: bool) -> Result<Vec<Node>, IcuMessageError> {
        let mut nodes = Vec::new();
        let mut text = String::new();

        while let Some(ch) = self.peek() {
            match ch {
                '}' if stop_at_brace => {
                    if !text.is_empty() {
                        nodes.push(Node::Text(text));
                    }
                    return Ok(nodes);
                }
                '}' if !stop_at_brace => {
                    // Stray close brace in top-level, treat as text
                    text.push(ch);
                    self.advance();
                }
                '{' => {
                    if !text.is_empty() {
                        nodes.push(Node::Text(text));
                        text = String::new();
                    }
                    let open_pos = self.pos;
                    self.advance(); // skip '{'
                    nodes.push(self.parse_placeholder(open_pos)?);
                }
                '\'' => {
                    self.advance();
                    // Quoting: '' → literal ', '{' → literal {, etc.
                    if let Some(next) = self.peek() {
                        if next == '\'' {
                            text.push('\'');
                            self.advance();
                        } else if next == '{' || next == '}' {
                            text.push(next);
                            self.advance();
                            // Read until closing quote
                            while let Some(c) = self.peek() {
                                if c == '\'' {
                                    self.advance();
                                    break;
                                }
                                text.push(c);
                                self.advance();
                            }
                        } else {
                            text.push('\'');
                        }
                    } else {
                        text.push('\'');
                    }
                }
                '#' => {
                    // # is replaced with the plural value during formatting
                    text.push('#');
                    self.advance();
                }
                _ => {
                    text.push(ch);
                    self.advance();
                }
            }
        }
        if !text.is_empty() {
            nodes.push(Node::Text(text));
        }
        Ok(nodes)
    }

    fn read_ident(&mut self) -> String {
        self.skip_ws();
        let mut ident = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        ident
    }

    fn parse_placeholder(&mut self, open_pos: usize) -> Result<Node, IcuMessageError> {
        let arg_name = self.read_ident();
        self.skip_ws();

        match self.peek() {
            Some('}') => {
                self.advance();
                Ok(Node::Argument(arg_name))
            }
            Some(',') => {
                self.advance();
                let format_type = self.read_ident();
                self.skip_ws();
                match format_type.as_str() {
                    "number" => {
                        self.expect_char('}')?;
                        Ok(Node::NumberFormat(arg_name))
                    }
                    "plural" => {
                        self.expect_char(',')?;
                        self.parse_plural(arg_name)
                    }
                    "select" => {
                        self.expect_char(',')?;
                        self.parse_select(arg_name)
                    }
                    _ => Err(IcuMessageError::UnknownType(format_type)),
                }
            }
            Some(_) => Err(IcuMessageError::UnexpectedChar {
                pos: self.pos,
                ch: self.chars[self.pos],
            }),
            None => Err(IcuMessageError::UnclosedBrace { open_pos }),
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), IcuMessageError> {
        self.skip_ws();
        match self.peek() {
            Some(ch) if ch == expected => {
                self.advance();
                Ok(())
            }
            Some(ch) => Err(IcuMessageError::UnexpectedChar {
                pos: self.pos,
                ch,
            }),
            None => Err(IcuMessageError::UnexpectedEnd),
        }
    }

    fn parse_plural(&mut self, arg: String) -> Result<Node, IcuMessageError> {
        let mut clauses = Vec::new();
        self.skip_ws();

        while let Some(ch) = self.peek() {
            if ch == '}' {
                self.advance();
                if !clauses.iter().any(|(s, _): &(PluralSelector, _)| *s == PluralSelector::Other) {
                    return Err(IcuMessageError::MissingOther);
                }
                return Ok(Node::Plural { arg, clauses });
            }

            let selector = self.read_plural_selector()?;
            self.skip_ws();
            self.expect_char('{')?;
            let body = self.parse_message(true)?;
            self.expect_char('}')?;
            clauses.push((selector, body));
            self.skip_ws();
        }
        Err(IcuMessageError::UnexpectedEnd)
    }

    fn read_plural_selector(&mut self) -> Result<PluralSelector, IcuMessageError> {
        self.skip_ws();
        if let Some('=') = self.peek() {
            self.advance();
            let num_str = self.read_ident();
            let n: i64 = num_str.parse().unwrap_or(0);
            return Ok(PluralSelector::Exact(n));
        }
        let kw = self.read_ident();
        match kw.as_str() {
            "zero" => Ok(PluralSelector::Zero),
            "one" => Ok(PluralSelector::One),
            "two" => Ok(PluralSelector::Two),
            "few" => Ok(PluralSelector::Few),
            "many" => Ok(PluralSelector::Many),
            "other" => Ok(PluralSelector::Other),
            _ => Err(IcuMessageError::UnexpectedChar {
                pos: self.pos,
                ch: ' ',
            }),
        }
    }

    fn parse_select(&mut self, arg: String) -> Result<Node, IcuMessageError> {
        let mut clauses = Vec::new();
        self.skip_ws();

        while let Some(ch) = self.peek() {
            if ch == '}' {
                self.advance();
                if !clauses.iter().any(|(s, _): &(String, _)| s == "other") {
                    return Err(IcuMessageError::MissingOther);
                }
                return Ok(Node::Select { arg, clauses });
            }

            let key = self.read_ident();
            self.skip_ws();
            self.expect_char('{')?;
            let body = self.parse_message(true)?;
            self.expect_char('}')?;
            clauses.push((key, body));
            self.skip_ws();
        }
        Err(IcuMessageError::UnexpectedEnd)
    }
}

// ── Public API ──────────────────────────────────────────────────

/// A compiled ICU message ready for formatting.
#[derive(Debug, Clone)]
pub struct IcuMessage {
    nodes: Vec<Node>,
    source: String,
}

impl IcuMessage {
    /// Parse an ICU MessageFormat string.
    pub fn parse(input: &str) -> Result<Self, IcuMessageError> {
        let mut parser = Parser::new(input);
        let nodes = parser.parse_message(false)?;
        Ok(Self {
            nodes,
            source: input.to_string(),
        })
    }

    /// Format the message with the given values.
    pub fn format(&self, values: &HashMap<String, Value>) -> Result<String, IcuMessageError> {
        format_nodes(&self.nodes, values, None)
    }

    /// Return the original source string.
    pub fn source(&self) -> &str {
        &self.source
    }
}

fn format_nodes(
    nodes: &[Node],
    values: &HashMap<String, Value>,
    pound_value: Option<i64>,
) -> Result<String, IcuMessageError> {
    let mut out = String::new();
    for node in nodes {
        match node {
            Node::Text(t) => {
                // Replace '#' with the pound value if inside a plural
                if let Some(pv) = pound_value {
                    for ch in t.chars() {
                        if ch == '#' {
                            out.push_str(&pv.to_string());
                        } else {
                            out.push(ch);
                        }
                    }
                } else {
                    out.push_str(t);
                }
            }
            Node::Argument(name) => {
                let val = values
                    .get(name)
                    .ok_or_else(|| IcuMessageError::MissingArgument(name.clone()))?;
                out.push_str(&val.to_string());
            }
            Node::NumberFormat(name) => {
                let val = values
                    .get(name)
                    .ok_or_else(|| IcuMessageError::MissingArgument(name.clone()))?;
                // Format with grouping separators
                if let Some(n) = val.as_f64() {
                    out.push_str(&format_number_grouped(n));
                } else {
                    out.push_str(&val.to_string());
                }
            }
            Node::Plural { arg, clauses } => {
                let val = values
                    .get(arg)
                    .ok_or_else(|| IcuMessageError::MissingArgument(arg.clone()))?;
                let n = val.as_i64().unwrap_or(0);
                let category = english_plural_category(n);

                // First try exact match, then category, then other
                let body = clauses
                    .iter()
                    .find(|(s, _)| *s == PluralSelector::Exact(n))
                    .or_else(|| clauses.iter().find(|(s, _)| *s == category))
                    .or_else(|| {
                        clauses
                            .iter()
                            .find(|(s, _)| *s == PluralSelector::Other)
                    })
                    .map(|(_, body)| body)
                    .ok_or(IcuMessageError::MissingOther)?;
                out.push_str(&format_nodes(body, values, Some(n))?);
            }
            Node::Select { arg, clauses } => {
                let val = values
                    .get(arg)
                    .ok_or_else(|| IcuMessageError::MissingArgument(arg.clone()))?;
                let key = val.as_str_match();
                let body = clauses
                    .iter()
                    .find(|(k, _)| *k == key)
                    .or_else(|| clauses.iter().find(|(k, _)| k == "other"))
                    .map(|(_, body)| body)
                    .ok_or(IcuMessageError::MissingOther)?;
                out.push_str(&format_nodes(body, values, pound_value)?);
            }
        }
    }
    Ok(out)
}

fn english_plural_category(n: i64) -> PluralSelector {
    let abs = n.unsigned_abs();
    if abs == 1 {
        PluralSelector::One
    } else {
        PluralSelector::Other
    }
}

fn format_number_grouped(n: f64) -> String {
    let int_part = n.abs() as u64;
    let negative = n < 0.0;
    let frac = n.fract().abs();

    let int_str = int_part.to_string();
    let mut grouped = String::new();
    for (i, ch) in int_str.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    let grouped: String = grouped.chars().rev().collect();

    let mut result = String::new();
    if negative {
        result.push('-');
    }
    result.push_str(&grouped);

    if frac > 0.0 {
        let frac_str = format!("{:.10}", frac);
        let frac_digits = frac_str.trim_start_matches("0.").trim_end_matches('0');
        if !frac_digits.is_empty() {
            result.push('.');
            result.push_str(frac_digits);
        }
    }
    result
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vals(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn simple_substitution() {
        let msg = IcuMessage::parse("Hello, {name}!").unwrap();
        let v = vals(&[("name", Value::String("World".into()))]);
        assert_eq!(msg.format(&v).unwrap(), "Hello, World!");
    }

    #[test]
    fn multiple_substitutions() {
        let msg = IcuMessage::parse("{greeting}, {name}!").unwrap();
        let v = vals(&[
            ("greeting", Value::String("Hi".into())),
            ("name", Value::String("Alice".into())),
        ]);
        assert_eq!(msg.format(&v).unwrap(), "Hi, Alice!");
    }

    #[test]
    fn number_format() {
        let msg = IcuMessage::parse("You have {count, number} items.").unwrap();
        let v = vals(&[("count", Value::Integer(1234567))]);
        assert_eq!(msg.format(&v).unwrap(), "You have 1,234,567 items.");
    }

    #[test]
    fn plural_one_other() {
        let msg =
            IcuMessage::parse("{count, plural, one{# item} other{# items}}").unwrap();
        let v1 = vals(&[("count", Value::Integer(1))]);
        assert_eq!(msg.format(&v1).unwrap(), "1 item");

        let v5 = vals(&[("count", Value::Integer(5))]);
        assert_eq!(msg.format(&v5).unwrap(), "5 items");
    }

    #[test]
    fn plural_exact_match() {
        let msg = IcuMessage::parse(
            "{count, plural, =0{no items} one{# item} other{# items}}",
        )
        .unwrap();
        let v0 = vals(&[("count", Value::Integer(0))]);
        assert_eq!(msg.format(&v0).unwrap(), "no items");
    }

    #[test]
    fn select_gender() {
        let msg = IcuMessage::parse(
            "{gender, select, male{He} female{She} other{They}} liked this.",
        )
        .unwrap();
        let vm = vals(&[("gender", Value::String("male".into()))]);
        assert_eq!(msg.format(&vm).unwrap(), "He liked this.");

        let vf = vals(&[("gender", Value::String("female".into()))]);
        assert_eq!(msg.format(&vf).unwrap(), "She liked this.");

        let vo = vals(&[("gender", Value::String("nonbinary".into()))]);
        assert_eq!(msg.format(&vo).unwrap(), "They liked this.");
    }

    #[test]
    fn nested_plural_in_select() {
        let msg = IcuMessage::parse(
            "{gender, select, male{He has {count, plural, one{# cat} other{# cats}}} other{They have {count, plural, one{# cat} other{# cats}}}}",
        ).unwrap();
        let v = vals(&[
            ("gender", Value::String("male".into())),
            ("count", Value::Integer(3)),
        ]);
        assert_eq!(msg.format(&v).unwrap(), "He has 3 cats");
    }

    #[test]
    fn escaped_brace() {
        let msg = IcuMessage::parse("Use '{' and '}'").unwrap();
        let v = HashMap::new();
        assert_eq!(msg.format(&v).unwrap(), "Use { and }");
    }

    #[test]
    fn escaped_single_quote() {
        let msg = IcuMessage::parse("It''s great").unwrap();
        let v = HashMap::new();
        assert_eq!(msg.format(&v).unwrap(), "It's great");
    }

    #[test]
    fn missing_argument_error() {
        let msg = IcuMessage::parse("{missing}").unwrap();
        let v = HashMap::new();
        assert!(matches!(
            msg.format(&v),
            Err(IcuMessageError::MissingArgument(_))
        ));
    }

    #[test]
    fn missing_other_clause() {
        let result = IcuMessage::parse("{x, select, male{He}}");
        assert!(matches!(result, Err(IcuMessageError::MissingOther)));
    }

    #[test]
    fn plain_text_passthrough() {
        let msg = IcuMessage::parse("No placeholders here.").unwrap();
        let v = HashMap::new();
        assert_eq!(msg.format(&v).unwrap(), "No placeholders here.");
    }

    #[test]
    fn source_preserved() {
        let src = "Hello {name}!";
        let msg = IcuMessage::parse(src).unwrap();
        assert_eq!(msg.source(), src);
    }
}
