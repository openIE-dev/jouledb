//! Search query parser.
//!
//! Quoted phrases, `field:value`, boolean operators (AND / OR / NOT / + / -),
//! parenthetical grouping, wildcards, range queries (`field:[a TO z]`),
//! query AST, and query optimization.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Parse errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("unmatched parenthesis")]
    UnmatchedParen,
    #[error("unmatched quote")]
    UnmatchedQuote,
    #[error("invalid range syntax")]
    InvalidRange,
    #[error("empty query")]
    EmptyQuery,
}

// ── AST Nodes ───────────────────────────────────────────────────

/// The abstract syntax tree for a parsed search query.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryNode {
    /// A single term, e.g. `hello`.
    Term(String),
    /// A quoted phrase, e.g. `"hello world"`.
    Phrase(String),
    /// A field-scoped term, e.g. `title:hello`.
    FieldTerm {
        field: String,
        value: Box<QueryNode>,
    },
    /// A wildcard term, e.g. `hel*` or `h?llo`.
    Wildcard(String),
    /// A range query, e.g. `field:[a TO z]`.
    Range {
        field: String,
        lower: String,
        upper: String,
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
    /// Boolean AND.
    And(Box<QueryNode>, Box<QueryNode>),
    /// Boolean OR.
    Or(Box<QueryNode>, Box<QueryNode>),
    /// Boolean NOT (must-not).
    Not(Box<QueryNode>),
    /// Required (+ prefix).
    Must(Box<QueryNode>),
    /// A group (parenthesized sub-expression).
    Group(Box<QueryNode>),
}

impl fmt::Display for QueryNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryNode::Term(t) => write!(f, "{}", t),
            QueryNode::Phrase(p) => write!(f, "\"{}\"", p),
            QueryNode::FieldTerm { field, value } => write!(f, "{}:{}", field, value),
            QueryNode::Wildcard(w) => write!(f, "{}", w),
            QueryNode::Range {
                field,
                lower,
                upper,
                lower_inclusive,
                upper_inclusive,
            } => {
                let lb = if *lower_inclusive { "[" } else { "{" };
                let rb = if *upper_inclusive { "]" } else { "}" };
                write!(f, "{}:{}{} TO {}{}", field, lb, lower, upper, rb)
            }
            QueryNode::And(l, r) => write!(f, "({} AND {})", l, r),
            QueryNode::Or(l, r) => write!(f, "({} OR {})", l, r),
            QueryNode::Not(n) => write!(f, "NOT {}", n),
            QueryNode::Must(n) => write!(f, "+{}", n),
            QueryNode::Group(g) => write!(f, "({})", g),
        }
    }
}

// ── Tokenizer ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Word(String),
    Quoted(String),
    LParen,
    RParen,
    And,
    Or,
    Not,
    Plus,
    Minus,
    Colon,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    To,
}

fn tokenize_query(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let ch = chars[i];

        // Skip whitespace.
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        match ch {
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '[' => {
                tokens.push(Token::LBracket);
                i += 1;
            }
            ']' => {
                tokens.push(Token::RBracket);
                i += 1;
            }
            '{' => {
                tokens.push(Token::LBrace);
                i += 1;
            }
            '}' => {
                tokens.push(Token::RBrace);
                i += 1;
            }
            ':' => {
                tokens.push(Token::Colon);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '"' => {
                // Quoted string.
                i += 1;
                let start = i;
                while i < len && chars[i] != '"' {
                    i += 1;
                }
                if i >= len {
                    return Err(ParseError::UnmatchedQuote);
                }
                let phrase: String = chars[start..i].iter().collect();
                tokens.push(Token::Quoted(phrase));
                i += 1; // skip closing quote
            }
            _ => {
                // Word.
                let start = i;
                while i < len
                    && !chars[i].is_whitespace()
                    && !matches!(
                        chars[i],
                        '(' | ')' | '[' | ']' | '{' | '}' | ':' | '"'
                    )
                {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                match word.as_str() {
                    "AND" => tokens.push(Token::And),
                    "OR" => tokens.push(Token::Or),
                    "NOT" => tokens.push(Token::Not),
                    "TO" => tokens.push(Token::To),
                    _ => tokens.push(Token::Word(word)),
                }
            }
        }
    }
    Ok(tokens)
}

// ── Parser ──────────────────────────────────────────────────────

/// Recursive descent parser for search queries.
struct QueryParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl QueryParser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect_token(&mut self, expected: &Token) -> Result<(), ParseError> {
        match self.advance() {
            Some(t) if t == *expected => Ok(()),
            _ => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse the full query: or_expr.
    fn parse(&mut self) -> Result<QueryNode, ParseError> {
        if self.tokens.is_empty() {
            return Err(ParseError::EmptyQuery);
        }
        self.parse_or()
    }

    /// or_expr = and_expr (OR and_expr)*
    fn parse_or(&mut self) -> Result<QueryNode, ParseError> {
        let mut left = self.parse_and()?;
        while self.peek() == Some(&Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = QueryNode::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    /// and_expr = unary_expr (AND? unary_expr)*
    fn parse_and(&mut self) -> Result<QueryNode, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            // Explicit AND.
            if self.peek() == Some(&Token::And) {
                self.advance();
                let right = self.parse_unary()?;
                left = QueryNode::And(Box::new(left), Box::new(right));
                continue;
            }
            // Implicit AND: next token is a term/phrase/paren/etc.
            if matches!(
                self.peek(),
                Some(Token::Word(_))
                    | Some(Token::Quoted(_))
                    | Some(Token::LParen)
                    | Some(Token::Not)
                    | Some(Token::Plus)
                    | Some(Token::Minus)
            ) {
                let right = self.parse_unary()?;
                left = QueryNode::And(Box::new(left), Box::new(right));
                continue;
            }
            break;
        }
        Ok(left)
    }

    /// unary_expr = (NOT | + | -) primary | primary
    fn parse_unary(&mut self) -> Result<QueryNode, ParseError> {
        match self.peek() {
            Some(Token::Not) => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(QueryNode::Not(Box::new(inner)))
            }
            Some(Token::Plus) => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(QueryNode::Must(Box::new(inner)))
            }
            Some(Token::Minus) => {
                self.advance();
                let inner = self.parse_primary()?;
                Ok(QueryNode::Not(Box::new(inner)))
            }
            _ => self.parse_primary(),
        }
    }

    /// primary = group | field_term | quoted | term
    fn parse_primary(&mut self) -> Result<QueryNode, ParseError> {
        match self.peek().cloned() {
            Some(Token::LParen) => {
                self.advance();
                let inner = self.parse_or()?;
                if self.peek() != Some(&Token::RParen) {
                    return Err(ParseError::UnmatchedParen);
                }
                self.advance();
                Ok(QueryNode::Group(Box::new(inner)))
            }
            Some(Token::Quoted(phrase)) => {
                self.advance();
                Ok(QueryNode::Phrase(phrase))
            }
            Some(Token::Word(word)) => {
                self.advance();
                // Check for field:value or field:[range].
                if self.peek() == Some(&Token::Colon) {
                    self.advance();
                    let field = word;
                    // Check for range query.
                    if matches!(self.peek(), Some(Token::LBracket) | Some(Token::LBrace)) {
                        return self.parse_range(&field);
                    }
                    let value = self.parse_field_value()?;
                    Ok(QueryNode::FieldTerm {
                        field,
                        value: Box::new(value),
                    })
                } else if word.contains('*') || word.contains('?') {
                    Ok(QueryNode::Wildcard(word))
                } else {
                    Ok(QueryNode::Term(word))
                }
            }
            _ => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse the value part of a field:value expression.
    fn parse_field_value(&mut self) -> Result<QueryNode, ParseError> {
        match self.peek().cloned() {
            Some(Token::Quoted(phrase)) => {
                self.advance();
                Ok(QueryNode::Phrase(phrase))
            }
            Some(Token::Word(word)) => {
                self.advance();
                if word.contains('*') || word.contains('?') {
                    Ok(QueryNode::Wildcard(word))
                } else {
                    Ok(QueryNode::Term(word))
                }
            }
            _ => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse a range query: `[lower TO upper]` or `{lower TO upper}`.
    fn parse_range(&mut self, field: &str) -> Result<QueryNode, ParseError> {
        let lower_inclusive = match self.peek() {
            Some(Token::LBracket) => true,
            Some(Token::LBrace) => false,
            _ => return Err(ParseError::InvalidRange),
        };
        self.advance();

        let lower = match self.advance() {
            Some(Token::Word(w)) => w,
            _ => return Err(ParseError::InvalidRange),
        };

        self.expect_token(&Token::To)
            .map_err(|_| ParseError::InvalidRange)?;

        let upper = match self.advance() {
            Some(Token::Word(w)) => w,
            _ => return Err(ParseError::InvalidRange),
        };

        let upper_inclusive = match self.peek() {
            Some(Token::RBracket) => true,
            Some(Token::RBrace) => false,
            _ => return Err(ParseError::InvalidRange),
        };
        self.advance();

        Ok(QueryNode::Range {
            field: field.to_string(),
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        })
    }
}

// ── Public API ──────────────────────────────────────────────────

/// Parse a search query string into an AST.
pub fn parse_query(input: &str) -> Result<QueryNode, ParseError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParseError::EmptyQuery);
    }
    let tokens = tokenize_query(trimmed)?;
    if tokens.is_empty() {
        return Err(ParseError::EmptyQuery);
    }
    let mut parser = QueryParser::new(tokens);
    parser.parse()
}

// ── Query optimization ──────────────────────────────────────────

/// Flatten nested AND/OR chains and remove double negation.
pub fn optimize(node: QueryNode) -> QueryNode {
    match node {
        QueryNode::And(l, r) => {
            let ol = optimize(*l);
            let or = optimize(*r);
            // Flatten AND(AND(a,b), c) -> we just keep the tree form, but
            // optimize children.
            QueryNode::And(Box::new(ol), Box::new(or))
        }
        QueryNode::Or(l, r) => {
            let ol = optimize(*l);
            let or = optimize(*r);
            QueryNode::Or(Box::new(ol), Box::new(or))
        }
        QueryNode::Not(inner) => {
            let oi = optimize(*inner);
            // Double negation elimination.
            if let QueryNode::Not(double_inner) = oi {
                *double_inner
            } else {
                QueryNode::Not(Box::new(oi))
            }
        }
        QueryNode::Must(inner) => QueryNode::Must(Box::new(optimize(*inner))),
        QueryNode::Group(inner) => {
            let oi = optimize(*inner);
            // Unwrap unnecessary groups around single terms.
            match oi {
                QueryNode::Term(_) | QueryNode::Phrase(_) | QueryNode::Wildcard(_) => oi,
                other => QueryNode::Group(Box::new(other)),
            }
        }
        QueryNode::FieldTerm { field, value } => QueryNode::FieldTerm {
            field,
            value: Box::new(optimize(*value)),
        },
        other => other,
    }
}

/// Extract all leaf terms from a query AST.
pub fn extract_terms(node: &QueryNode) -> Vec<String> {
    let mut terms = Vec::new();
    collect_terms(node, &mut terms);
    terms
}

fn collect_terms(node: &QueryNode, out: &mut Vec<String>) {
    match node {
        QueryNode::Term(t) => out.push(t.clone()),
        QueryNode::Phrase(p) => out.push(p.clone()),
        QueryNode::Wildcard(w) => out.push(w.clone()),
        QueryNode::FieldTerm { value, .. } => collect_terms(value, out),
        QueryNode::And(l, r) | QueryNode::Or(l, r) => {
            collect_terms(l, out);
            collect_terms(r, out);
        }
        QueryNode::Not(n) | QueryNode::Must(n) | QueryNode::Group(n) => {
            collect_terms(n, out);
        }
        QueryNode::Range { .. } => {}
    }
}

/// Count the total number of nodes in the AST.
pub fn node_count(node: &QueryNode) -> usize {
    match node {
        QueryNode::Term(_)
        | QueryNode::Phrase(_)
        | QueryNode::Wildcard(_)
        | QueryNode::Range { .. } => 1,
        QueryNode::FieldTerm { value, .. } => 1 + node_count(value),
        QueryNode::And(l, r) | QueryNode::Or(l, r) => 1 + node_count(l) + node_count(r),
        QueryNode::Not(n) | QueryNode::Must(n) | QueryNode::Group(n) => 1 + node_count(n),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_term() {
        let q = parse_query("hello").unwrap();
        assert_eq!(q, QueryNode::Term("hello".to_string()));
    }

    #[test]
    fn test_quoted_phrase() {
        let q = parse_query("\"hello world\"").unwrap();
        assert_eq!(q, QueryNode::Phrase("hello world".to_string()));
    }

    #[test]
    fn test_field_term() {
        let q = parse_query("title:hello").unwrap();
        assert!(matches!(q, QueryNode::FieldTerm { field, .. } if field == "title"));
    }

    #[test]
    fn test_field_phrase() {
        let q = parse_query("title:\"hello world\"").unwrap();
        if let QueryNode::FieldTerm { field, value } = q {
            assert_eq!(field, "title");
            assert_eq!(*value, QueryNode::Phrase("hello world".to_string()));
        } else {
            panic!("expected FieldTerm");
        }
    }

    #[test]
    fn test_boolean_and() {
        let q = parse_query("foo AND bar").unwrap();
        assert!(matches!(q, QueryNode::And(_, _)));
    }

    #[test]
    fn test_boolean_or() {
        let q = parse_query("foo OR bar").unwrap();
        assert!(matches!(q, QueryNode::Or(_, _)));
    }

    #[test]
    fn test_boolean_not() {
        let q = parse_query("NOT foo").unwrap();
        assert!(matches!(q, QueryNode::Not(_)));
    }

    #[test]
    fn test_plus_prefix() {
        let q = parse_query("+required").unwrap();
        assert!(matches!(q, QueryNode::Must(_)));
    }

    #[test]
    fn test_minus_prefix() {
        let q = parse_query("-excluded").unwrap();
        assert!(matches!(q, QueryNode::Not(_)));
    }

    #[test]
    fn test_parenthetical_grouping() {
        let q = parse_query("(foo OR bar) AND baz").unwrap();
        assert!(matches!(q, QueryNode::And(_, _)));
    }

    #[test]
    fn test_wildcard() {
        let q = parse_query("hel*").unwrap();
        assert_eq!(q, QueryNode::Wildcard("hel*".to_string()));
    }

    #[test]
    fn test_question_wildcard() {
        let q = parse_query("h?llo").unwrap();
        assert_eq!(q, QueryNode::Wildcard("h?llo".to_string()));
    }

    #[test]
    fn test_range_inclusive() {
        let q = parse_query("price:[10 TO 100]").unwrap();
        if let QueryNode::Range {
            field,
            lower,
            upper,
            lower_inclusive,
            upper_inclusive,
        } = q
        {
            assert_eq!(field, "price");
            assert_eq!(lower, "10");
            assert_eq!(upper, "100");
            assert!(lower_inclusive);
            assert!(upper_inclusive);
        } else {
            panic!("expected Range");
        }
    }

    #[test]
    fn test_range_exclusive() {
        let q = parse_query("date:{2020 TO 2025}").unwrap();
        if let QueryNode::Range {
            lower_inclusive,
            upper_inclusive,
            ..
        } = q
        {
            assert!(!lower_inclusive);
            assert!(!upper_inclusive);
        } else {
            panic!("expected Range");
        }
    }

    #[test]
    fn test_implicit_and() {
        let q = parse_query("foo bar").unwrap();
        assert!(matches!(q, QueryNode::And(_, _)));
    }

    #[test]
    fn test_complex_query() {
        let q = parse_query("title:rust AND (fast OR efficient) NOT slow").unwrap();
        assert!(matches!(q, QueryNode::And(_, _)));
    }

    #[test]
    fn test_empty_query_error() {
        let err = parse_query("").unwrap_err();
        assert!(matches!(err, ParseError::EmptyQuery));
    }

    #[test]
    fn test_unmatched_quote() {
        let err = parse_query("\"hello").unwrap_err();
        assert!(matches!(err, ParseError::UnmatchedQuote));
    }

    #[test]
    fn test_unmatched_paren() {
        let err = parse_query("(hello").unwrap_err();
        assert!(matches!(err, ParseError::UnmatchedParen));
    }

    #[test]
    fn test_display() {
        let q = parse_query("hello").unwrap();
        assert_eq!(format!("{}", q), "hello");

        let q2 = parse_query("\"hello world\"").unwrap();
        assert_eq!(format!("{}", q2), "\"hello world\"");
    }

    #[test]
    fn test_extract_terms() {
        let q = parse_query("foo AND bar OR baz").unwrap();
        let terms = extract_terms(&q);
        assert!(terms.contains(&"foo".to_string()));
        assert!(terms.contains(&"bar".to_string()));
        assert!(terms.contains(&"baz".to_string()));
    }

    #[test]
    fn test_node_count() {
        let q = parse_query("foo AND bar").unwrap();
        assert_eq!(node_count(&q), 3); // AND + 2 terms
    }

    #[test]
    fn test_optimize_double_negation() {
        let node = QueryNode::Not(Box::new(QueryNode::Not(Box::new(QueryNode::Term(
            "hello".to_string(),
        )))));
        let optimized = optimize(node);
        assert_eq!(optimized, QueryNode::Term("hello".to_string()));
    }

    #[test]
    fn test_optimize_group_unwrap() {
        let node = QueryNode::Group(Box::new(QueryNode::Term("hello".to_string())));
        let optimized = optimize(node);
        assert_eq!(optimized, QueryNode::Term("hello".to_string()));
    }

    #[test]
    fn test_field_wildcard() {
        let q = parse_query("title:hel*").unwrap();
        if let QueryNode::FieldTerm { field, value } = q {
            assert_eq!(field, "title");
            assert!(matches!(*value, QueryNode::Wildcard(_)));
        } else {
            panic!("expected FieldTerm with Wildcard");
        }
    }
}
