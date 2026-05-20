//! Pratt parser for expression parsing — prefix/infix/postfix operators,
//! configurable precedence/associativity, recursive descent, AST node
//! construction, error recovery, operator registration.

use std::collections::HashMap;
use std::fmt;

// ── AST ─────────────────────────────────────────────────────────────────────

/// A node in the expression AST.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Integer literal.
    Integer(i64),
    /// Float literal.
    Float(f64),
    /// Boolean literal.
    Bool(bool),
    /// String literal.
    StringLit(String),
    /// Variable / identifier reference.
    Ident(String),
    /// Prefix (unary) operation: operator, operand.
    Prefix {
        op: String,
        operand: Box<Expr>,
    },
    /// Infix (binary) operation: left, operator, right.
    Infix {
        left: Box<Expr>,
        op: String,
        right: Box<Expr>,
    },
    /// Postfix (unary) operation: operand, operator.
    Postfix {
        operand: Box<Expr>,
        op: String,
    },
    /// Function call: callee, arguments.
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// Index operation: base[index].
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    /// Ternary conditional: cond ? then_expr : else_expr.
    Ternary {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
    /// Grouping (parenthesised expression).
    Group(Box<Expr>),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::StringLit(s) => write!(f, "\"{s}\""),
            Self::Ident(s) => write!(f, "{s}"),
            Self::Prefix { op, operand } => write!(f, "({op}{operand})"),
            Self::Infix { left, op, right } => write!(f, "({left} {op} {right})"),
            Self::Postfix { operand, op } => write!(f, "({operand}{op})"),
            Self::Call { callee, args } => {
                write!(f, "{callee}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            Self::Index { base, index } => write!(f, "{base}[{index}]"),
            Self::Ternary { cond, then_expr, else_expr } => {
                write!(f, "({cond} ? {then_expr} : {else_expr})")
            }
            Self::Group(inner) => write!(f, "({inner})"),
        }
    }
}

// ── Tokens ──────────────────────────────────────────────────────────────────

/// A minimal token type for the Pratt parser's input stream.
#[derive(Debug, Clone, PartialEq)]
pub enum PToken {
    Integer(i64),
    Float(f64),
    Bool(bool),
    StringLit(String),
    Ident(String),
    Op(String),
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Question,
    Colon,
    Eof,
}

impl fmt::Display for PToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::StringLit(s) => write!(f, "\"{s}\""),
            Self::Ident(s) => write!(f, "{s}"),
            Self::Op(s) => write!(f, "{s}"),
            Self::LParen => write!(f, "("),
            Self::RParen => write!(f, ")"),
            Self::LBracket => write!(f, "["),
            Self::RBracket => write!(f, "]"),
            Self::Comma => write!(f, ","),
            Self::Question => write!(f, "?"),
            Self::Colon => write!(f, ":"),
            Self::Eof => write!(f, "EOF"),
        }
    }
}

// ── Associativity ───────────────────────────────────────────────────────────

/// Operator associativity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assoc {
    Left,
    Right,
    None,
}

// ── Operator info ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PrefixOp {
    precedence: u32,
}

#[derive(Debug, Clone)]
struct InfixOp {
    precedence: u32,
    assoc: Assoc,
}

#[derive(Debug, Clone)]
struct PostfixOp {
    precedence: u32,
}

// ── Parse error ─────────────────────────────────────────────────────────────

/// Errors during parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Expected a certain token, got another.
    Expected(String, String),
    /// Unexpected token.
    Unexpected(String),
    /// Unexpected end of input.
    UnexpectedEof,
    /// Multiple errors collected during recovery.
    Multiple(Vec<ParseError>),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expected(exp, got) => write!(f, "expected {exp}, got {got}"),
            Self::Unexpected(tok) => write!(f, "unexpected token: {tok}"),
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::Multiple(errs) => {
                for (i, e) in errs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{e}")?;
                }
                Ok(())
            }
        }
    }
}

// ── Pratt parser ────────────────────────────────────────────────────────────

/// A Pratt parser with configurable operator precedence and associativity.
pub struct PrattParser {
    prefix_ops: HashMap<String, PrefixOp>,
    infix_ops: HashMap<String, InfixOp>,
    postfix_ops: HashMap<String, PostfixOp>,
    /// Precedence for function call (default 90).
    call_precedence: u32,
    /// Precedence for index (default 90).
    index_precedence: u32,
    /// Precedence for ternary (default 5).
    ternary_precedence: u32,
    /// Enable ternary `? :` operator.
    enable_ternary: bool,
    /// Enable call syntax `f(args)`.
    enable_call: bool,
    /// Enable index syntax `a[i]`.
    enable_index: bool,
}

impl PrattParser {
    /// Create a new empty parser.
    pub fn new() -> Self {
        Self {
            prefix_ops: HashMap::new(),
            infix_ops: HashMap::new(),
            postfix_ops: HashMap::new(),
            call_precedence: 90,
            index_precedence: 90,
            ternary_precedence: 5,
            enable_ternary: false,
            enable_call: false,
            enable_index: false,
        }
    }

    /// Create a parser pre-configured with C-like arithmetic operators.
    pub fn arithmetic() -> Self {
        let mut p = Self::new();
        p.prefix("-", 80);
        p.prefix("!", 80);
        p.prefix("+", 80);
        p.infix("+", 40, Assoc::Left);
        p.infix("-", 40, Assoc::Left);
        p.infix("*", 50, Assoc::Left);
        p.infix("/", 50, Assoc::Left);
        p.infix("%", 50, Assoc::Left);
        p.infix("**", 60, Assoc::Right);
        p.infix("==", 20, Assoc::Left);
        p.infix("!=", 20, Assoc::Left);
        p.infix("<", 25, Assoc::Left);
        p.infix(">", 25, Assoc::Left);
        p.infix("<=", 25, Assoc::Left);
        p.infix(">=", 25, Assoc::Left);
        p.infix("&&", 10, Assoc::Left);
        p.infix("||", 8, Assoc::Left);
        p.infix("=", 2, Assoc::Right);
        p.enable_call = true;
        p.enable_index = true;
        p.enable_ternary = true;
        p
    }

    /// Register a prefix operator.
    pub fn prefix(&mut self, op: &str, precedence: u32) -> &mut Self {
        self.prefix_ops.insert(op.to_string(), PrefixOp { precedence });
        self
    }

    /// Register an infix operator.
    pub fn infix(&mut self, op: &str, precedence: u32, assoc: Assoc) -> &mut Self {
        self.infix_ops.insert(op.to_string(), InfixOp { precedence, assoc });
        self
    }

    /// Register a postfix operator.
    pub fn postfix(&mut self, op: &str, precedence: u32) -> &mut Self {
        self.postfix_ops.insert(op.to_string(), PostfixOp { precedence });
        self
    }

    /// Enable call syntax: `expr(args...)`.
    pub fn with_calls(mut self) -> Self {
        self.enable_call = true;
        self
    }

    /// Enable index syntax: `expr[index]`.
    pub fn with_indexing(mut self) -> Self {
        self.enable_index = true;
        self
    }

    /// Enable ternary syntax: `cond ? a : b`.
    pub fn with_ternary(mut self) -> Self {
        self.enable_ternary = true;
        self
    }

    // ── Parsing ─────────────────────────────────────────────────────────

    /// Parse a token stream into an expression AST.
    pub fn parse(&self, tokens: &[PToken]) -> Result<Expr, ParseError> {
        let mut cursor = Cursor::new(tokens);
        let expr = self.parse_expr(&mut cursor, 0)?;
        if !cursor.is_at_end() {
            return Err(ParseError::Unexpected(format!("{}", cursor.current())));
        }
        Ok(expr)
    }

    /// Parse a stream of tokens and collect all errors (for recovery).
    pub fn parse_recovering(&self, tokens: &[PToken]) -> (Option<Expr>, Vec<ParseError>) {
        let mut cursor = Cursor::new(tokens);
        let mut errors = Vec::new();
        match self.parse_expr(&mut cursor, 0) {
            Ok(expr) => {
                if !cursor.is_at_end() {
                    errors.push(ParseError::Unexpected(format!("{}", cursor.current())));
                }
                (Some(expr), errors)
            }
            Err(e) => {
                errors.push(e);
                (None, errors)
            }
        }
    }

    fn parse_expr(&self, cursor: &mut Cursor<'_>, min_bp: u32) -> Result<Expr, ParseError> {
        // NUD — prefix / atom
        let mut lhs = self.nud(cursor)?;

        // LED — infix / postfix
        loop {
            if cursor.is_at_end() {
                break;
            }
            let tok = cursor.current();

            // Postfix
            if let PToken::Op(ref op) = tok {
                if let Some(pf) = self.postfix_ops.get(op) {
                    if pf.precedence >= min_bp {
                        let op_str = op.clone();
                        cursor.advance();
                        lhs = Expr::Postfix {
                            operand: Box::new(lhs),
                            op: op_str,
                        };
                        continue;
                    }
                }
            }

            // Ternary
            if self.enable_ternary {
                if let PToken::Question = tok {
                    if self.ternary_precedence >= min_bp {
                        cursor.advance(); // consume ?
                        let then_expr = self.parse_expr(cursor, 0)?;
                        self.expect_colon(cursor)?;
                        let else_expr = self.parse_expr(cursor, self.ternary_precedence)?;
                        lhs = Expr::Ternary {
                            cond: Box::new(lhs),
                            then_expr: Box::new(then_expr),
                            else_expr: Box::new(else_expr),
                        };
                        continue;
                    }
                }
            }

            // Call
            if self.enable_call {
                if let PToken::LParen = tok {
                    if self.call_precedence >= min_bp {
                        cursor.advance(); // consume (
                        let args = self.parse_args(cursor)?;
                        self.expect_rparen(cursor)?;
                        lhs = Expr::Call {
                            callee: Box::new(lhs),
                            args,
                        };
                        continue;
                    }
                }
            }

            // Index
            if self.enable_index {
                if let PToken::LBracket = tok {
                    if self.index_precedence >= min_bp {
                        cursor.advance(); // consume [
                        let index = self.parse_expr(cursor, 0)?;
                        self.expect_rbracket(cursor)?;
                        lhs = Expr::Index {
                            base: Box::new(lhs),
                            index: Box::new(index),
                        };
                        continue;
                    }
                }
            }

            // Infix
            if let PToken::Op(ref op) = tok {
                if let Some(inf) = self.infix_ops.get(op) {
                    if inf.precedence >= min_bp {
                        let op_str = op.clone();
                        let next_bp = match inf.assoc {
                            Assoc::Left => inf.precedence + 1,
                            Assoc::Right => inf.precedence,
                            Assoc::None => inf.precedence + 1,
                        };
                        cursor.advance();
                        let rhs = self.parse_expr(cursor, next_bp)?;
                        lhs = Expr::Infix {
                            left: Box::new(lhs),
                            op: op_str,
                            right: Box::new(rhs),
                        };
                        continue;
                    }
                }
            }

            break;
        }

        Ok(lhs)
    }

    fn nud(&self, cursor: &mut Cursor<'_>) -> Result<Expr, ParseError> {
        let tok = cursor.current();
        match tok {
            PToken::Integer(n) => {
                let v = n;
                cursor.advance();
                Ok(Expr::Integer(v))
            }
            PToken::Float(n) => {
                let v = n;
                cursor.advance();
                Ok(Expr::Float(v))
            }
            PToken::Bool(b) => {
                let v = b;
                cursor.advance();
                Ok(Expr::Bool(v))
            }
            PToken::StringLit(ref s) => {
                let v = s.clone();
                cursor.advance();
                Ok(Expr::StringLit(v))
            }
            PToken::Ident(ref s) => {
                let v = s.clone();
                cursor.advance();
                Ok(Expr::Ident(v))
            }
            PToken::LParen => {
                cursor.advance();
                let inner = self.parse_expr(cursor, 0)?;
                self.expect_rparen(cursor)?;
                Ok(Expr::Group(Box::new(inner)))
            }
            PToken::Op(ref op) => {
                if let Some(pre) = self.prefix_ops.get(op) {
                    let op_str = op.clone();
                    let prec = pre.precedence;
                    cursor.advance();
                    let operand = self.parse_expr(cursor, prec)?;
                    Ok(Expr::Prefix {
                        op: op_str,
                        operand: Box::new(operand),
                    })
                } else {
                    Err(ParseError::Unexpected(format!("{tok}")))
                }
            }
            PToken::Eof => Err(ParseError::UnexpectedEof),
            _ => Err(ParseError::Unexpected(format!("{tok}"))),
        }
    }

    fn parse_args(&self, cursor: &mut Cursor<'_>) -> Result<Vec<Expr>, ParseError> {
        let mut args = Vec::new();
        if let PToken::RParen = cursor.current() {
            return Ok(args);
        }
        args.push(self.parse_expr(cursor, 0)?);
        while let PToken::Comma = cursor.current() {
            cursor.advance();
            args.push(self.parse_expr(cursor, 0)?);
        }
        Ok(args)
    }

    fn expect_rparen(&self, cursor: &mut Cursor<'_>) -> Result<(), ParseError> {
        if let PToken::RParen = cursor.current() {
            cursor.advance();
            Ok(())
        } else {
            Err(ParseError::Expected(")".into(), format!("{}", cursor.current())))
        }
    }

    fn expect_rbracket(&self, cursor: &mut Cursor<'_>) -> Result<(), ParseError> {
        if let PToken::RBracket = cursor.current() {
            cursor.advance();
            Ok(())
        } else {
            Err(ParseError::Expected("]".into(), format!("{}", cursor.current())))
        }
    }

    fn expect_colon(&self, cursor: &mut Cursor<'_>) -> Result<(), ParseError> {
        if let PToken::Colon = cursor.current() {
            cursor.advance();
            Ok(())
        } else {
            Err(ParseError::Expected(":".into(), format!("{}", cursor.current())))
        }
    }
}

impl Default for PrattParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Cursor ──────────────────────────────────────────────────────────────────

struct Cursor<'a> {
    tokens: &'a [PToken],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(tokens: &'a [PToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn current(&self) -> PToken {
        self.tokens.get(self.pos).cloned().unwrap_or(PToken::Eof)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn is_at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.current(), PToken::Eof)
    }
}

// ── Convenience tokenizer ───────────────────────────────────────────────────

/// A very simple tokenizer that turns a mathematical expression string into
/// `PToken`s suitable for the `PrattParser`.
pub fn tokenize_math(input: &str) -> Vec<PToken> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let len = chars.len();

    while i < len {
        let c = chars[i];

        if c.is_whitespace() {
            i += 1;
            continue;
        }

        if c.is_ascii_digit() {
            let start = i;
            let mut is_float = false;
            while i < len && (chars[i].is_ascii_digit() || chars[i] == '.') {
                if chars[i] == '.' {
                    is_float = true;
                }
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            if is_float {
                tokens.push(PToken::Float(s.parse().unwrap_or(0.0)));
            } else {
                tokens.push(PToken::Integer(s.parse().unwrap_or(0)));
            }
            continue;
        }

        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            match s.as_str() {
                "true" => tokens.push(PToken::Bool(true)),
                "false" => tokens.push(PToken::Bool(false)),
                _ => tokens.push(PToken::Ident(s)),
            }
            continue;
        }

        match c {
            '(' => { tokens.push(PToken::LParen); i += 1; }
            ')' => { tokens.push(PToken::RParen); i += 1; }
            '[' => { tokens.push(PToken::LBracket); i += 1; }
            ']' => { tokens.push(PToken::RBracket); i += 1; }
            ',' => { tokens.push(PToken::Comma); i += 1; }
            '?' => { tokens.push(PToken::Question); i += 1; }
            ':' => { tokens.push(PToken::Colon); i += 1; }
            '"' => {
                i += 1; // skip opening "
                let start = i;
                while i < len && chars[i] != '"' {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(PToken::StringLit(s));
                if i < len {
                    i += 1; // skip closing "
                }
            }
            _ => {
                // Multi-char operators: **, ==, !=, <=, >=, &&, ||
                let two: String = chars[i..std::cmp::min(i + 2, len)].iter().collect();
                match two.as_str() {
                    "**" | "==" | "!=" | "<=" | ">=" | "&&" | "||" => {
                        tokens.push(PToken::Op(two));
                        i += 2;
                    }
                    _ => {
                        tokens.push(PToken::Op(c.to_string()));
                        i += 1;
                    }
                }
            }
        }
    }

    tokens.push(PToken::Eof);
    tokens
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> Expr {
        let tokens = tokenize_math(input);
        PrattParser::arithmetic().parse(&tokens).unwrap()
    }

    #[test]
    fn test_integer_literal() {
        assert_eq!(parse("42"), Expr::Integer(42));
    }

    #[test]
    fn test_float_literal() {
        assert_eq!(parse("3.14"), Expr::Float(3.14));
    }

    #[test]
    fn test_bool_literal() {
        assert_eq!(parse("true"), Expr::Bool(true));
    }

    #[test]
    fn test_ident() {
        assert_eq!(parse("x"), Expr::Ident("x".into()));
    }

    #[test]
    fn test_prefix_negation() {
        let e = parse("-5");
        assert!(matches!(e, Expr::Prefix { ref op, .. } if op == "-"));
    }

    #[test]
    fn test_prefix_not() {
        let e = parse("!true");
        assert!(matches!(e, Expr::Prefix { ref op, .. } if op == "!"));
    }

    #[test]
    fn test_simple_addition() {
        let e = parse("1 + 2");
        assert!(matches!(e, Expr::Infix { ref op, .. } if op == "+"));
    }

    #[test]
    fn test_precedence_mul_over_add() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let e = parse("1 + 2 * 3");
        if let Expr::Infix { op, right, .. } = e {
            assert_eq!(op, "+");
            assert!(matches!(*right, Expr::Infix { ref op, .. } if op == "*"));
        } else {
            panic!("expected infix");
        }
    }

    #[test]
    fn test_left_associativity() {
        // 1 - 2 - 3 should parse as (1 - 2) - 3
        let e = parse("1 - 2 - 3");
        if let Expr::Infix { left, op, .. } = e {
            assert_eq!(op, "-");
            assert!(matches!(*left, Expr::Infix { ref op, .. } if op == "-"));
        } else {
            panic!("expected infix");
        }
    }

    #[test]
    fn test_right_associativity() {
        // 2 ** 3 ** 4 should parse as 2 ** (3 ** 4) since ** is right-assoc
        let e = parse("2 ** 3 ** 4");
        if let Expr::Infix { right, op, .. } = e {
            assert_eq!(op, "**");
            assert!(matches!(*right, Expr::Infix { ref op, .. } if op == "**"));
        } else {
            panic!("expected infix");
        }
    }

    #[test]
    fn test_parenthesised_grouping() {
        let e = parse("(1 + 2) * 3");
        if let Expr::Infix { left, op, .. } = e {
            assert_eq!(op, "*");
            assert!(matches!(*left, Expr::Group(_)));
        } else {
            panic!("expected infix");
        }
    }

    #[test]
    fn test_function_call_no_args() {
        let e = parse("f()");
        assert!(matches!(e, Expr::Call { ref args, .. } if args.is_empty()));
    }

    #[test]
    fn test_function_call_with_args() {
        let e = parse("add(1, 2)");
        if let Expr::Call { args, .. } = e {
            assert_eq!(args.len(), 2);
        } else {
            panic!("expected call");
        }
    }

    #[test]
    fn test_index_access() {
        let e = parse("a[0]");
        assert!(matches!(e, Expr::Index { .. }));
    }

    #[test]
    fn test_ternary() {
        let e = parse("x ? 1 : 2");
        assert!(matches!(e, Expr::Ternary { .. }));
    }

    #[test]
    fn test_complex_expression() {
        let e = parse("a + b * c - d / e");
        // Just verify it parses without error
        assert!(matches!(e, Expr::Infix { .. }));
    }

    #[test]
    fn test_postfix_operator() {
        let mut p = PrattParser::arithmetic();
        p.postfix("++", 85);
        let tokens = vec![PToken::Ident("x".into()), PToken::Op("++".into()), PToken::Eof];
        let e = p.parse(&tokens).unwrap();
        assert!(matches!(e, Expr::Postfix { ref op, .. } if op == "++"));
    }

    #[test]
    fn test_error_unexpected_eof() {
        let tokens = vec![PToken::Eof];
        let p = PrattParser::arithmetic();
        let err = p.parse(&tokens).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEof));
    }

    #[test]
    fn test_error_unexpected_token() {
        let tokens = vec![PToken::RParen, PToken::Eof];
        let p = PrattParser::arithmetic();
        let err = p.parse(&tokens).unwrap_err();
        assert!(matches!(err, ParseError::Unexpected(_)));
    }

    #[test]
    fn test_error_missing_rparen() {
        let tokens = vec![
            PToken::LParen,
            PToken::Integer(1),
            PToken::Eof,
        ];
        let p = PrattParser::arithmetic();
        let err = p.parse(&tokens).unwrap_err();
        assert!(matches!(err, ParseError::Expected(..)));
    }

    #[test]
    fn test_nested_calls() {
        let e = parse("f(g(x))");
        if let Expr::Call { args, .. } = e {
            assert!(matches!(args[0], Expr::Call { .. }));
        } else {
            panic!("expected nested call");
        }
    }

    #[test]
    fn test_display() {
        let e = parse("1 + 2");
        let s = format!("{e}");
        assert!(s.contains("+"));
    }

    #[test]
    fn test_parse_recovering_success() {
        let tokens = tokenize_math("1 + 2");
        let p = PrattParser::arithmetic();
        let (expr, errs) = p.parse_recovering(&tokens);
        assert!(expr.is_some());
        assert!(errs.is_empty());
    }

    #[test]
    fn test_parse_recovering_error() {
        let tokens = vec![PToken::Eof];
        let p = PrattParser::arithmetic();
        let (expr, errs) = p.parse_recovering(&tokens);
        assert!(expr.is_none());
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_comparison_operators() {
        let e = parse("a == b");
        assert!(matches!(e, Expr::Infix { ref op, .. } if op == "=="));
        let e2 = parse("a != b");
        assert!(matches!(e2, Expr::Infix { ref op, .. } if op == "!="));
    }

    #[test]
    fn test_logical_operators() {
        let e = parse("a && b || c");
        // || has lower precedence so should be root
        assert!(matches!(e, Expr::Infix { ref op, .. } if op == "||"));
    }

    #[test]
    fn test_string_literal_parse() {
        let tokens = vec![PToken::StringLit("hello".into()), PToken::Eof];
        let p = PrattParser::arithmetic();
        let e = p.parse(&tokens).unwrap();
        assert_eq!(e, Expr::StringLit("hello".into()));
    }
}
