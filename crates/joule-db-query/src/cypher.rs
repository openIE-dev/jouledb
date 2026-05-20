//! Cypher Query Language Parser
//!
//! Cypher is a graph query language used by Neo4j and other graph databases.

use crate::ast::{Expression, Operator, Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;

/// Cypher query
#[derive(Debug, Clone)]
pub struct CypherQuery {
    pub clauses: Vec<CypherClause>,
}

impl CypherQuery {
    /// Rewrite every `AsOf(ts)` clause into the equivalent conjunctive
    /// temporal predicate `valid_from <= ts AND ts < valid_to`, folded
    /// into an existing `Where` (or inserted as a new `Where` right
    /// after the last MATCH). After this runs no `AsOf` clause remains,
    /// so the executor — which already evaluates `Where` against
    /// `valid_from`/`valid_to` — enforces the time-travel pin with zero
    /// executor changes. Idempotent.
    fn desugar_temporal(&mut self) {
        // Collect + drop the AsOf clauses, remembering where the last
        // MATCH/OPTIONAL MATCH sat so the synthetic WHERE lands there.
        let mut pins: Vec<Expression> = Vec::new();
        let mut last_match_idx: Option<usize> = None;
        let mut kept: Vec<CypherClause> = Vec::with_capacity(self.clauses.len());
        for c in std::mem::take(&mut self.clauses) {
            match c {
                CypherClause::AsOf(ts) => pins.push(ts),
                other => {
                    if matches!(other, CypherClause::Match(_) | CypherClause::OptionalMatch(_)) {
                        last_match_idx = Some(kept.len());
                    }
                    kept.push(other);
                }
            }
        }
        self.clauses = kept;
        if pins.is_empty() {
            return;
        }

        // pin(ts) := (valid_from <= ts) AND (ts < valid_to)
        let pin_pred = |ts: Expression| -> Expression {
            Expression::Binary {
                left: Box::new(Expression::Binary {
                    left: Box::new(Expression::Column("valid_from".to_string())),
                    op: Operator::Le,
                    right: Box::new(ts.clone()),
                }),
                op: Operator::And,
                right: Box::new(Expression::Binary {
                    left: Box::new(ts),
                    op: Operator::Lt,
                    right: Box::new(Expression::Column("valid_to".to_string())),
                }),
            }
        };

        // Conjoin all pins (multiple AS OF is unusual but well-defined).
        let Some(temporal) = pins.into_iter().map(pin_pred).reduce(|a, b| {
            Expression::Binary { left: Box::new(a), op: Operator::And, right: Box::new(b) }
        }) else {
            return;
        };

        // Fold into the first existing WHERE if there is one.
        for c in &mut self.clauses {
            if let CypherClause::Where(existing) = c {
                let merged = Expression::Binary {
                    left: Box::new(existing.clone()),
                    op: Operator::And,
                    right: Box::new(temporal),
                };
                *existing = merged;
                return;
            }
        }

        // No WHERE — insert one right after the last MATCH (or at front).
        let insert_at = last_match_idx.map(|i| i + 1).unwrap_or(0);
        self.clauses
            .insert(insert_at, CypherClause::Where(temporal));
    }

    /// Convert to generic Query
    pub fn to_query(&self) -> Query {
        Query {
            query_type: QueryType::Traverse,
            source: None,
            columns: Vec::new(),
            filter: None,
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: None,
            limit: None,
            offset: None,
            joins: Vec::new(),
            values: Vec::new(),
            returning: Vec::new(),
            ctes: Vec::new(),
            derived_columns: HashMap::new(),
            distinct: false, source_alias: None,
        }
    }
}

/// Cypher clause
#[derive(Debug, Clone)]
pub enum CypherClause {
    Match(CypherMatch),
    OptionalMatch(CypherMatch),
    Where(Expression),
    Create(Vec<CypherPattern>),
    Merge(CypherPattern),
    Delete(Vec<String>, bool), // variables, detach
    Set(Vec<CypherSet>),
    Remove(Vec<CypherRemove>),
    Return(CypherReturn),
    With(CypherWith),
    OrderBy(Vec<(String, bool)>), // (expr, descending)
    Skip(usize),
    Limit(usize),
    Union(bool),                   // all
    Unwind(Expression, String),    // expr, alias
    Call(String, Vec<Expression>), // procedure, args
    /// System-versioned time-travel pin: `AS OF <ts>` or
    /// `FOR SYSTEM_TIME AS OF <ts>`. Desugared transparently in
    /// `parse()` into a conjunctive `Where (valid_from <= ts AND ts <
    /// valid_to)` — the executor never sees this variant.
    AsOf(Expression),
}

/// Cypher MATCH clause
#[derive(Debug, Clone)]
pub struct CypherMatch {
    pub patterns: Vec<CypherPattern>,
}

/// Cypher pattern (nodes and relationships)
#[derive(Debug, Clone)]
pub struct CypherPattern {
    pub elements: Vec<CypherPatternElement>,
}

/// Pattern element (node or relationship)
#[derive(Debug, Clone)]
pub enum CypherPatternElement {
    Node(CypherNode),
    Relationship(CypherRelationship),
}

/// Cypher node
#[derive(Debug, Clone)]
pub struct CypherNode {
    pub variable: Option<String>,
    pub labels: Vec<String>,
    pub properties: HashMap<String, Expression>,
}

impl CypherNode {
    pub fn new() -> Self {
        Self {
            variable: None,
            labels: Vec::new(),
            properties: HashMap::new(),
        }
    }

    pub fn with_variable(mut self, var: &str) -> Self {
        self.variable = Some(var.to_string());
        self
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.labels.push(label.to_string());
        self
    }
}

impl Default for CypherNode {
    fn default() -> Self {
        Self::new()
    }
}

/// Cypher relationship
#[derive(Debug, Clone)]
pub struct CypherRelationship {
    pub variable: Option<String>,
    pub rel_types: Vec<String>,
    pub properties: HashMap<String, Expression>,
    pub direction: RelationshipDirection,
    pub range: Option<(Option<usize>, Option<usize>)>, // min, max hops
}

/// Relationship direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationshipDirection {
    Outgoing, // ->
    Incoming, // <-
    Both,     // --
}

/// Cypher SET clause item
#[derive(Debug, Clone)]
pub enum CypherSet {
    Property(String, String, Expression), // variable, property, value
    Labels(String, Vec<String>),          // variable, labels
    AllProperties(String, Expression),    // variable, map
    MergeProperties(String, Expression),  // variable, map
}

/// Cypher REMOVE clause item
#[derive(Debug, Clone)]
pub enum CypherRemove {
    Property(String, String),    // variable, property
    Labels(String, Vec<String>), // variable, labels
}

/// Cypher RETURN clause
#[derive(Debug, Clone)]
pub struct CypherReturn {
    pub distinct: bool,
    pub items: Vec<CypherReturnItem>,
}

/// Cypher RETURN item
#[derive(Debug, Clone)]
pub struct CypherReturnItem {
    pub expression: Expression,
    pub alias: Option<String>,
}

/// Cypher WITH clause
#[derive(Debug, Clone)]
pub struct CypherWith {
    pub distinct: bool,
    pub items: Vec<CypherReturnItem>,
    pub where_clause: Option<Expression>,
}

/// Cypher token
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Match,
    Optional,
    Where,
    Create,
    Merge,
    Delete,
    Detach,
    Set,
    Remove,
    Return,
    With,
    Order,
    By,
    Skip,
    Limit,
    Union,
    All,
    Unwind,
    As,
    Of,
    For,
    SystemTime,
    And,
    Or,
    Not,
    In,
    Is,
    Null,
    True,
    False,
    Call,
    Yield,
    Distinct,

    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Dot,
    Pipe,
    Star,

    // Relationship arrows
    ArrowLeft,  // <-
    ArrowRight, // ->
    Dash,       // -

    // Operators
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Slash,
    Percent,
    Caret,
    Contains,
    StartsWith,
    EndsWith,
    RegexMatch,

    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Identifier(String),
    Parameter(String),

    Eof,
}

/// Cypher Parser
/// Maximum expression nesting depth to prevent stack overflow from crafted inputs.
/// The Cypher parser has an 8-function recursive chain per nesting level, so we
/// use a lower limit than other parsers to stay within default stack sizes.
const MAX_EXPRESSION_DEPTH: usize = 50;

/// Maximum query length in bytes (1 MB).
const MAX_QUERY_LENGTH: usize = 1_048_576;

pub struct CypherParser {
    tokens: Vec<Token>,
    pos: usize,
    /// Current expression nesting depth (prevents stack overflow).
    expression_depth: usize,
}

impl CypherParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            pos: 0,
            expression_depth: 0,
        }
    }

    /// Parse Cypher query
    pub fn parse(&mut self, cypher: &str) -> QueryResult<CypherQuery> {
        if cypher.len() > MAX_QUERY_LENGTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Query too long: {} bytes exceeds maximum of {} bytes",
                cypher.len(),
                MAX_QUERY_LENGTH
            )));
        }
        self.expression_depth = 0;
        self.tokenize(cypher)?;
        self.pos = 0;
        self.parse_query()
    }

    fn tokenize(&mut self, input: &str) -> QueryResult<()> {
        self.tokens.clear();
        let mut chars = input.chars().peekable();

        while let Some(&c) = chars.peek() {
            match c {
                ' ' | '\t' | '\n' | '\r' => {
                    chars.next();
                }
                '(' => {
                    chars.next();
                    self.tokens.push(Token::LParen);
                }
                ')' => {
                    chars.next();
                    self.tokens.push(Token::RParen);
                }
                '[' => {
                    chars.next();
                    self.tokens.push(Token::LBracket);
                }
                ']' => {
                    chars.next();
                    self.tokens.push(Token::RBracket);
                }
                '{' => {
                    chars.next();
                    self.tokens.push(Token::LBrace);
                }
                '}' => {
                    chars.next();
                    self.tokens.push(Token::RBrace);
                }
                ':' => {
                    chars.next();
                    self.tokens.push(Token::Colon);
                }
                ',' => {
                    chars.next();
                    self.tokens.push(Token::Comma);
                }
                '.' => {
                    chars.next();
                    self.tokens.push(Token::Dot);
                }
                '|' => {
                    chars.next();
                    self.tokens.push(Token::Pipe);
                }
                '*' => {
                    chars.next();
                    self.tokens.push(Token::Star);
                }
                '+' => {
                    chars.next();
                    self.tokens.push(Token::Plus);
                }
                '/' => {
                    chars.next();
                    if chars.peek() == Some(&'/') {
                        // Comment
                        while chars.peek().map(|&c| c != '\n').unwrap_or(false) {
                            chars.next();
                        }
                    } else {
                        self.tokens.push(Token::Slash);
                    }
                }
                '%' => {
                    chars.next();
                    self.tokens.push(Token::Percent);
                }
                '^' => {
                    chars.next();
                    self.tokens.push(Token::Caret);
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'~') {
                        chars.next();
                        self.tokens.push(Token::RegexMatch);
                    } else {
                        self.tokens.push(Token::Eq);
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'-') {
                        chars.next();
                        self.tokens.push(Token::ArrowLeft);
                    } else if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Le);
                    } else if chars.peek() == Some(&'>') {
                        chars.next();
                        self.tokens.push(Token::Ne);
                    } else {
                        self.tokens.push(Token::Lt);
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        self.tokens.push(Token::Ge);
                    } else {
                        self.tokens.push(Token::Gt);
                    }
                }
                '-' => {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        self.tokens.push(Token::ArrowRight);
                    } else {
                        self.tokens.push(Token::Dash);
                    }
                }
                '$' => {
                    chars.next();
                    let mut name = String::new();
                    while chars
                        .peek()
                        .map(|&c| c.is_alphanumeric() || c == '_')
                        .unwrap_or(false)
                    {
                        name.push(chars.next().expect("peeked char exists"));
                    }
                    self.tokens.push(Token::Parameter(name));
                }
                '\'' | '"' => {
                    let quote = chars.next().expect("quote char exists");
                    let mut s = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == quote {
                            chars.next();
                            if chars.peek() == Some(&quote) {
                                s.push(chars.next().expect("peeked char exists"));
                            } else {
                                break;
                            }
                        } else if c == '\\' {
                            chars.next();
                            if let Some(&escaped) = chars.peek() {
                                chars.next();
                                match escaped {
                                    'n' => s.push('\n'),
                                    't' => s.push('\t'),
                                    'r' => s.push('\r'),
                                    _ => s.push(escaped),
                                }
                            }
                        } else {
                            s.push(chars.next().expect("peeked char exists"));
                        }
                    }
                    self.tokens.push(Token::String(s));
                }
                '`' => {
                    chars.next();
                    let mut s = String::new();
                    while let Some(&c) = chars.peek() {
                        if c == '`' {
                            chars.next();
                            break;
                        }
                        s.push(chars.next().expect("peeked char exists"));
                    }
                    self.tokens.push(Token::Identifier(s));
                }
                _ if c.is_ascii_digit() => {
                    let mut num = String::new();
                    let mut is_float = false;
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() {
                            num.push(chars.next().expect("peeked char exists"));
                        } else if c == '.' && !is_float {
                            // Look ahead: if next char is also '.', don't consume (it's a range like 1..3)
                            let mut peek_chars = chars.clone();
                            peek_chars.next(); // skip the first dot
                            if peek_chars.peek() == Some(&'.') {
                                // This is a range, don't consume the dot
                                break;
                            }
                            is_float = true;
                            num.push(chars.next().expect("peeked char exists"));
                        } else {
                            break;
                        }
                    }
                    if is_float {
                        self.tokens.push(Token::Float(num.parse().unwrap_or(0.0)));
                    } else {
                        self.tokens.push(Token::Integer(num.parse().unwrap_or(0)));
                    }
                }
                _ if c.is_alphabetic() || c == '_' => {
                    let mut ident = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_alphanumeric() || c == '_' {
                            ident.push(chars.next().expect("peeked char exists"));
                        } else {
                            break;
                        }
                    }
                    let token = match ident.to_uppercase().as_str() {
                        "MATCH" => Token::Match,
                        "OPTIONAL" => Token::Optional,
                        "WHERE" => Token::Where,
                        "CREATE" => Token::Create,
                        "MERGE" => Token::Merge,
                        "DELETE" => Token::Delete,
                        "DETACH" => Token::Detach,
                        "SET" => Token::Set,
                        "REMOVE" => Token::Remove,
                        "RETURN" => Token::Return,
                        "WITH" => Token::With,
                        "ORDER" => Token::Order,
                        "BY" => Token::By,
                        "SKIP" => Token::Skip,
                        "LIMIT" => Token::Limit,
                        "UNION" => Token::Union,
                        "ALL" => Token::All,
                        "UNWIND" => Token::Unwind,
                        "AS" => Token::As,
                        "OF" => Token::Of,
                        "FOR" => Token::For,
                        "SYSTEM_TIME" => Token::SystemTime,
                        "AND" => Token::And,
                        "OR" => Token::Or,
                        "NOT" => Token::Not,
                        "IN" => Token::In,
                        "IS" => Token::Is,
                        "NULL" => Token::Null,
                        "TRUE" => Token::True,
                        "FALSE" => Token::False,
                        "CALL" => Token::Call,
                        "YIELD" => Token::Yield,
                        "DISTINCT" => Token::Distinct,
                        "CONTAINS" => Token::Contains,
                        "STARTS" => Token::StartsWith,
                        "ENDS" => Token::EndsWith,
                        _ => Token::Identifier(ident),
                    };
                    self.tokens.push(token);
                }
                _ => {
                    return Err(QueryError::SyntaxError {
                        message: format!("Unexpected character: {}", c),
                        line: 1,
                        column: 1,
                    });
                }
            }
        }

        self.tokens.push(Token::Eof);
        Ok(())
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: &Token) -> QueryResult<()> {
        let token = self.advance();
        if &token == expected {
            Ok(())
        } else {
            Err(QueryError::ParseError(format!(
                "Expected {:?}, got {:?}",
                expected, token
            )))
        }
    }

    fn parse_query(&mut self) -> QueryResult<CypherQuery> {
        let mut clauses = Vec::new();

        while *self.peek() != Token::Eof {
            let clause = match self.peek() {
                Token::Match => self.parse_match(false)?,
                Token::Optional => {
                    self.advance();
                    self.expect(&Token::Match)?;
                    CypherClause::OptionalMatch(self.parse_match_inner()?)
                }
                Token::Where => {
                    self.advance();
                    CypherClause::Where(self.parse_expression()?)
                }
                Token::Create => {
                    self.advance();
                    CypherClause::Create(self.parse_patterns()?)
                }
                Token::Merge => {
                    self.advance();
                    CypherClause::Merge(self.parse_pattern()?)
                }
                Token::Delete => {
                    self.advance();
                    let vars = self.parse_identifier_list()?;
                    CypherClause::Delete(vars, false)
                }
                Token::Detach => {
                    self.advance();
                    self.expect(&Token::Delete)?;
                    let vars = self.parse_identifier_list()?;
                    CypherClause::Delete(vars, true)
                }
                Token::Set => {
                    self.advance();
                    CypherClause::Set(self.parse_set_items()?)
                }
                Token::Remove => {
                    self.advance();
                    CypherClause::Remove(self.parse_remove_items()?)
                }
                Token::Return => {
                    self.advance();
                    CypherClause::Return(self.parse_return()?)
                }
                Token::With => {
                    self.advance();
                    CypherClause::With(self.parse_with()?)
                }
                Token::Order => {
                    self.advance();
                    self.expect(&Token::By)?;
                    CypherClause::OrderBy(self.parse_order_by()?)
                }
                Token::Skip => {
                    self.advance();
                    match self.advance() {
                        Token::Integer(n) => CypherClause::Skip(n as usize),
                        t => {
                            return Err(QueryError::ParseError(format!(
                                "Expected integer, got {:?}",
                                t
                            )));
                        }
                    }
                }
                Token::Limit => {
                    self.advance();
                    match self.advance() {
                        Token::Integer(n) => CypherClause::Limit(n as usize),
                        t => {
                            return Err(QueryError::ParseError(format!(
                                "Expected integer, got {:?}",
                                t
                            )));
                        }
                    }
                }
                Token::Union => {
                    self.advance();
                    let all = *self.peek() == Token::All;
                    if all {
                        self.advance();
                    }
                    CypherClause::Union(all)
                }
                Token::Unwind => {
                    self.advance();
                    let expr = self.parse_expression()?;
                    self.expect(&Token::As)?;
                    let alias = self.parse_identifier()?;
                    CypherClause::Unwind(expr, alias)
                }
                Token::Call => {
                    self.advance();
                    // Parse dotted procedure name: db.labels, db.schema.nodeTypeProperties, etc.
                    let mut proc = self.parse_identifier()?;
                    while *self.peek() == Token::Dot {
                        self.advance(); // consume '.'
                        let part = self.parse_identifier()?;
                        proc = format!("{}.{}", proc, part);
                    }
                    self.expect(&Token::LParen)?;
                    let args = if *self.peek() != Token::RParen {
                        self.parse_expression_list()?
                    } else {
                        Vec::new()
                    };
                    self.expect(&Token::RParen)?;
                    CypherClause::Call(proc, args)
                }
                Token::As => {
                    // Top-level `AS OF <ts>` time-travel pin. (At clause
                    // position `AS` is unambiguous — aliasing `AS` only
                    // occurs inside RETURN/WITH/UNWIND parsing.)
                    self.advance();
                    self.expect(&Token::Of)?;
                    let ts = self.parse_expression()?;
                    CypherClause::AsOf(ts)
                }
                Token::For => {
                    // SQL:2011-style `FOR SYSTEM_TIME AS OF <ts>`.
                    self.advance();
                    self.expect(&Token::SystemTime)?;
                    self.expect(&Token::As)?;
                    self.expect(&Token::Of)?;
                    let ts = self.parse_expression()?;
                    CypherClause::AsOf(ts)
                }
                t => return Err(QueryError::ParseError(format!("Unexpected token: {:?}", t))),
            };
            clauses.push(clause);
        }

        let mut q = CypherQuery { clauses };
        q.desugar_temporal();
        Ok(q)
    }

    fn parse_match(&mut self, optional: bool) -> QueryResult<CypherClause> {
        self.expect(&Token::Match)?;
        let m = self.parse_match_inner()?;
        if optional {
            Ok(CypherClause::OptionalMatch(m))
        } else {
            Ok(CypherClause::Match(m))
        }
    }

    fn parse_match_inner(&mut self) -> QueryResult<CypherMatch> {
        let patterns = self.parse_patterns()?;
        Ok(CypherMatch { patterns })
    }

    fn parse_patterns(&mut self) -> QueryResult<Vec<CypherPattern>> {
        let mut patterns = Vec::new();
        patterns.push(self.parse_pattern()?);

        while *self.peek() == Token::Comma {
            self.advance();
            patterns.push(self.parse_pattern()?);
        }

        Ok(patterns)
    }

    fn parse_pattern(&mut self) -> QueryResult<CypherPattern> {
        let mut elements = Vec::new();

        // Must start with a node
        elements.push(CypherPatternElement::Node(self.parse_node()?));

        // Parse relationship-node pairs
        while matches!(self.peek(), Token::Dash | Token::ArrowLeft) {
            elements.push(CypherPatternElement::Relationship(
                self.parse_relationship()?,
            ));
            elements.push(CypherPatternElement::Node(self.parse_node()?));
        }

        Ok(CypherPattern { elements })
    }

    fn parse_node(&mut self) -> QueryResult<CypherNode> {
        self.expect(&Token::LParen)?;

        let mut node = CypherNode::new();

        // Variable name
        if let Token::Identifier(name) = self.peek().clone() {
            self.advance();
            node.variable = Some(name);
        }

        // Labels
        while *self.peek() == Token::Colon {
            self.advance();
            node.labels.push(self.parse_identifier()?);
        }

        // Properties
        if *self.peek() == Token::LBrace {
            node.properties = self.parse_map()?;
        }

        self.expect(&Token::RParen)?;
        Ok(node)
    }

    fn parse_relationship(&mut self) -> QueryResult<CypherRelationship> {
        let mut direction = RelationshipDirection::Both;

        // Check for incoming arrow
        if *self.peek() == Token::ArrowLeft {
            self.advance();
            direction = RelationshipDirection::Incoming;
        } else {
            self.expect(&Token::Dash)?;
        }

        let mut rel = CypherRelationship {
            variable: None,
            rel_types: Vec::new(),
            properties: HashMap::new(),
            direction,
            range: None,
        };

        // Optional bracket for details
        if *self.peek() == Token::LBracket {
            self.advance();

            // Variable
            if let Token::Identifier(name) = self.peek().clone() {
                self.advance();
                rel.variable = Some(name);
            }

            // Relationship types
            while *self.peek() == Token::Colon {
                self.advance();
                rel.rel_types.push(self.parse_identifier()?);
                if *self.peek() == Token::Pipe {
                    self.advance();
                }
            }

            // Variable length
            if *self.peek() == Token::Star {
                self.advance();
                let min = if let Token::Integer(n) = self.peek() {
                    let n = *n;
                    self.advance();
                    Some(n as usize)
                } else {
                    None
                };
                let max = if *self.peek() == Token::Dot {
                    self.advance();
                    self.expect(&Token::Dot)?;
                    if let Token::Integer(n) = self.peek() {
                        let n = *n;
                        self.advance();
                        Some(n as usize)
                    } else {
                        None
                    }
                } else {
                    min
                };
                rel.range = Some((min, max));
            }

            // Properties
            if *self.peek() == Token::LBrace {
                rel.properties = self.parse_map()?;
            }

            self.expect(&Token::RBracket)?;
        }

        // Check for outgoing arrow
        if *self.peek() == Token::ArrowRight {
            self.advance();
            if direction == RelationshipDirection::Incoming {
                return Err(QueryError::ParseError(
                    "Invalid relationship direction".to_string(),
                ));
            }
            rel.direction = RelationshipDirection::Outgoing;
        } else {
            self.expect(&Token::Dash)?;
        }

        Ok(rel)
    }

    fn parse_map(&mut self) -> QueryResult<HashMap<String, Expression>> {
        self.expect(&Token::LBrace)?;
        let mut map = HashMap::new();

        if *self.peek() != Token::RBrace {
            loop {
                let key = self.parse_identifier()?;
                self.expect(&Token::Colon)?;
                let value = self.parse_expression()?;
                map.insert(key, value);

                if *self.peek() != Token::Comma {
                    break;
                }
                self.advance();
            }
        }

        self.expect(&Token::RBrace)?;
        Ok(map)
    }

    fn parse_set_items(&mut self) -> QueryResult<Vec<CypherSet>> {
        let mut items = Vec::new();

        loop {
            let var = self.parse_identifier()?;

            if *self.peek() == Token::Dot {
                self.advance();
                let prop = self.parse_identifier()?;
                self.expect(&Token::Eq)?;
                let value = self.parse_expression()?;
                items.push(CypherSet::Property(var, prop, value));
            } else if *self.peek() == Token::Colon {
                let mut labels = Vec::new();
                while *self.peek() == Token::Colon {
                    self.advance();
                    labels.push(self.parse_identifier()?);
                }
                items.push(CypherSet::Labels(var, labels));
            } else if *self.peek() == Token::Eq {
                self.advance();
                let value = self.parse_expression()?;
                items.push(CypherSet::AllProperties(var, value));
            } else if *self.peek() == Token::Plus {
                self.advance();
                self.expect(&Token::Eq)?;
                let value = self.parse_expression()?;
                items.push(CypherSet::MergeProperties(var, value));
            }

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(items)
    }

    fn parse_remove_items(&mut self) -> QueryResult<Vec<CypherRemove>> {
        let mut items = Vec::new();

        loop {
            let var = self.parse_identifier()?;

            if *self.peek() == Token::Dot {
                self.advance();
                let prop = self.parse_identifier()?;
                items.push(CypherRemove::Property(var, prop));
            } else {
                let mut labels = Vec::new();
                while *self.peek() == Token::Colon {
                    self.advance();
                    labels.push(self.parse_identifier()?);
                }
                items.push(CypherRemove::Labels(var, labels));
            }

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(items)
    }

    fn parse_return(&mut self) -> QueryResult<CypherReturn> {
        let distinct = if *self.peek() == Token::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let items = self.parse_return_items()?;
        Ok(CypherReturn { distinct, items })
    }

    fn parse_with(&mut self) -> QueryResult<CypherWith> {
        let distinct = if *self.peek() == Token::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let items = self.parse_return_items()?;

        let where_clause = if *self.peek() == Token::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(CypherWith {
            distinct,
            items,
            where_clause,
        })
    }

    fn parse_return_items(&mut self) -> QueryResult<Vec<CypherReturnItem>> {
        let mut items = Vec::new();

        loop {
            let expression = self.parse_expression()?;
            let alias = if *self.peek() == Token::As {
                self.advance();
                Some(self.parse_identifier()?)
            } else {
                None
            };

            items.push(CypherReturnItem { expression, alias });

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(items)
    }

    fn parse_order_by(&mut self) -> QueryResult<Vec<(String, bool)>> {
        let mut orders = Vec::new();

        loop {
            // Parse the expression (which can be "n" or "n.name")
            let base = self.parse_identifier()?;
            let expr = if *self.peek() == Token::Dot {
                self.advance();
                let prop = self.parse_identifier()?;
                format!("{}.{}", base, prop)
            } else {
                base
            };

            let desc = if let Token::Identifier(s) = self.peek() {
                if s.to_uppercase() == "DESC" {
                    self.advance();
                    true
                } else if s.to_uppercase() == "ASC" {
                    self.advance();
                    false
                } else {
                    false
                }
            } else {
                false
            };

            orders.push((expr, desc));

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(orders)
    }

    fn parse_identifier_list(&mut self) -> QueryResult<Vec<String>> {
        let mut list = Vec::new();

        loop {
            list.push(self.parse_identifier()?);
            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(list)
    }

    fn parse_identifier(&mut self) -> QueryResult<String> {
        match self.advance() {
            Token::Identifier(s) => Ok(s),
            t => Err(QueryError::ParseError(format!(
                "Expected identifier, got {:?}",
                t
            ))),
        }
    }

    fn parse_expression_list(&mut self) -> QueryResult<Vec<Expression>> {
        let mut list = Vec::new();

        loop {
            list.push(self.parse_expression()?);
            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(list)
    }

    fn parse_expression(&mut self) -> QueryResult<Expression> {
        self.expression_depth += 1;
        if self.expression_depth > MAX_EXPRESSION_DEPTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Expression nesting too deep: exceeds maximum depth of {}",
                MAX_EXPRESSION_DEPTH
            )));
        }
        let result = self.parse_or();
        self.expression_depth -= 1;
        result
    }

    fn parse_or(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_and()?;

        while *self.peek() == Token::Or {
            self.advance();
            let right = self.parse_and()?;
            left = Expression::or(left, right);
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_not()?;

        while *self.peek() == Token::And {
            self.advance();
            let right = self.parse_not()?;
            left = Expression::and(left, right);
        }

        Ok(left)
    }

    fn parse_not(&mut self) -> QueryResult<Expression> {
        if *self.peek() == Token::Not {
            self.advance();
            let expr = self.parse_not()?;
            Ok(Expression::Unary {
                op: crate::ast::UnaryOperator::Not,
                expr: Box::new(expr),
            })
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> QueryResult<Expression> {
        let left = self.parse_additive()?;

        let op = match self.peek() {
            Token::Eq => Some(Operator::Eq),
            Token::Ne => Some(Operator::Ne),
            Token::Lt => Some(Operator::Lt),
            Token::Le => Some(Operator::Le),
            Token::Gt => Some(Operator::Gt),
            Token::Ge => Some(Operator::Ge),
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let right = self.parse_additive()?;
            Ok(Expression::binary(left, op, right))
        } else if *self.peek() == Token::Contains {
            self.advance();
            let right = self.parse_additive()?;
            // CONTAINS 'str' → LIKE '%str%'
            let pattern = match &right {
                Expression::Literal(Value::String(s)) => format!("%{}%", s),
                _ => "%".to_string(),
            };
            Ok(Expression::Like {
                expr: Box::new(left),
                pattern,
                negated: false,
                case_insensitive: false,
            })
        } else if *self.peek() == Token::StartsWith {
            self.advance();
            // Consume WITH token (tokenized as Token::With, not Identifier)
            if *self.peek() == Token::With {
                self.advance();
            }
            let right = self.parse_additive()?;
            let pattern = match &right {
                Expression::Literal(Value::String(s)) => format!("{}%", s),
                _ => "%".to_string(),
            };
            Ok(Expression::Like {
                expr: Box::new(left),
                pattern,
                negated: false,
                case_insensitive: false,
            })
        } else if *self.peek() == Token::EndsWith {
            self.advance();
            if *self.peek() == Token::With {
                self.advance();
            }
            let right = self.parse_additive()?;
            let pattern = match &right {
                Expression::Literal(Value::String(s)) => format!("%{}", s),
                _ => "%".to_string(),
            };
            Ok(Expression::Like {
                expr: Box::new(left),
                pattern,
                negated: false,
                case_insensitive: false,
            })
        } else if *self.peek() == Token::RegexMatch {
            self.advance();
            let right = self.parse_additive()?;
            let pattern = match &right {
                Expression::Literal(Value::String(s)) => s.clone(),
                _ => ".*".to_string(),
            };
            Ok(Expression::RegexMatch {
                expr: Box::new(left),
                pattern,
                negated: false,
            })
        } else if *self.peek() == Token::Is {
            // IS NULL / IS NOT NULL
            self.advance(); // skip IS
            let negated = if *self.peek() == Token::Not {
                self.advance(); // skip NOT
                true
            } else {
                false
            };
            self.expect(&Token::Null)?;
            Ok(Expression::IsNull {
                expr: Box::new(left),
                negated,
            })
        } else if *self.peek() == Token::In {
            // IN [list] → Expression::In { expr, list, negated: false }
            self.advance();
            self.expect(&Token::LBracket)?;
            let list = if *self.peek() != Token::RBracket {
                self.parse_expression_list()?
            } else {
                Vec::new()
            };
            self.expect(&Token::RBracket)?;
            Ok(Expression::In {
                expr: Box::new(left),
                list,
                negated: false,
            })
        } else if *self.peek() == Token::Not {
            // NOT IN [list] → Expression::In { expr, list, negated: true }
            if self.tokens.get(self.pos + 1) == Some(&Token::In) {
                self.advance(); // skip NOT
                self.advance(); // skip IN
                self.expect(&Token::LBracket)?;
                let list = if *self.peek() != Token::RBracket {
                    self.parse_expression_list()?
                } else {
                    Vec::new()
                };
                self.expect(&Token::RBracket)?;
                Ok(Expression::In {
                    expr: Box::new(left),
                    list,
                    negated: true,
                })
            } else {
                Ok(left)
            }
        } else {
            Ok(left)
        }
    }

    fn parse_additive(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_multiplicative()?;

        loop {
            let op = match self.peek() {
                Token::Plus => Operator::Add,
                Token::Minus | Token::Dash => Operator::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expression::binary(left, op, right);
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_primary()?;

        loop {
            let op = match self.peek() {
                Token::Star => Operator::Mul,
                Token::Slash => Operator::Div,
                Token::Percent => Operator::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_primary()?;
            left = Expression::binary(left, op, right);
        }

        Ok(left)
    }

    fn parse_primary(&mut self) -> QueryResult<Expression> {
        match self.advance() {
            Token::Integer(n) => Ok(Expression::Literal(Value::Int(n))),
            Token::Float(n) => Ok(Expression::Literal(Value::Float(n))),
            Token::String(s) => Ok(Expression::Literal(Value::String(s))),
            Token::True => Ok(Expression::Literal(Value::Bool(true))),
            Token::False => Ok(Expression::Literal(Value::Bool(false))),
            Token::Null => Ok(Expression::Literal(Value::Null)),
            Token::Parameter(name) => Ok(Expression::NamedParameter(name)),
            Token::Identifier(name) => {
                if *self.peek() == Token::LParen {
                    // Function call
                    self.advance();
                    let args = if *self.peek() != Token::RParen {
                        self.parse_expression_list()?
                    } else {
                        Vec::new()
                    };
                    self.expect(&Token::RParen)?;
                    Ok(Expression::Function { name, args })
                } else if *self.peek() == Token::Dot {
                    // Property access
                    self.advance();
                    let prop = self.parse_identifier()?;
                    Ok(Expression::QualifiedColumn {
                        table: name,
                        column: prop,
                    })
                } else {
                    Ok(Expression::Column(name))
                }
            }
            Token::LParen => {
                let expr = self.parse_expression()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBracket => {
                // List literal
                let items = if *self.peek() != Token::RBracket {
                    self.parse_expression_list()?
                } else {
                    Vec::new()
                };
                self.expect(&Token::RBracket)?;
                let values: Vec<Value> = items
                    .into_iter()
                    .filter_map(|e| {
                        if let Expression::Literal(v) = e {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(Expression::Literal(Value::Array(values)))
            }
            t => Err(QueryError::ParseError(format!("Unexpected token: {:?}", t))),
        }
    }
}

impl Default for CypherParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_match() {
        let mut parser = CypherParser::new();
        let query = parser.parse("MATCH (n) RETURN n").unwrap();

        assert_eq!(query.clauses.len(), 2);
        assert!(matches!(query.clauses[0], CypherClause::Match(_)));
        assert!(matches!(query.clauses[1], CypherClause::Return(_)));
    }

    #[test]
    fn test_match_with_label() {
        let mut parser = CypherParser::new();
        let query = parser.parse("MATCH (n:Person) RETURN n.name").unwrap();

        if let CypherClause::Match(m) = &query.clauses[0] {
            if let CypherPatternElement::Node(node) = &m.patterns[0].elements[0] {
                assert_eq!(node.labels, vec!["Person"]);
            }
        }
    }

    #[test]
    fn test_match_relationship() {
        let mut parser = CypherParser::new();
        let query = parser
            .parse("MATCH (a)-[r:KNOWS]->(b) RETURN a, b")
            .unwrap();

        if let CypherClause::Match(m) = &query.clauses[0] {
            assert_eq!(m.patterns[0].elements.len(), 3);
        }
    }

    #[test]
    fn test_match_with_where() {
        let mut parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n:Person) WHERE n.age > 18 RETURN n")
            .unwrap();

        assert!(matches!(query.clauses[1], CypherClause::Where(_)));
    }

    #[test]
    fn test_create() {
        let mut parser = CypherParser::new();
        let query = parser.parse("CREATE (n:Person {name: 'Alice'})").unwrap();

        assert!(matches!(query.clauses[0], CypherClause::Create(_)));
    }

    #[test]
    fn test_with_order_limit() {
        let mut parser = CypherParser::new();
        let query = parser
            .parse("MATCH (n) RETURN n ORDER BY n.name SKIP 10 LIMIT 5")
            .unwrap();

        assert!(matches!(query.clauses[2], CypherClause::OrderBy(_)));
        assert!(matches!(query.clauses[3], CypherClause::Skip(10)));
        assert!(matches!(query.clauses[4], CypherClause::Limit(5)));
    }

    #[test]
    fn test_variable_length_path() {
        let mut parser = CypherParser::new();
        let query = parser.parse("MATCH (a)-[*1..3]->(b) RETURN a, b").unwrap();

        if let CypherClause::Match(m) = &query.clauses[0] {
            if let CypherPatternElement::Relationship(rel) = &m.patterns[0].elements[1] {
                assert_eq!(rel.range, Some((Some(1), Some(3))));
            }
        }
    }

    // ── Cypher AS OF / FOR SYSTEM_TIME time-travel ──────────────────────

    /// The desugared temporal predicate must mention both validity
    /// columns and the pinned timestamp. Walk the expression tree.
    fn mentions(expr: &Expression, col: &str) -> bool {
        match expr {
            Expression::Column(c) => c == col,
            Expression::QualifiedColumn { column, .. } => column == col,
            Expression::Binary { left, right, .. } => {
                mentions(left, col) || mentions(right, col)
            }
            _ => false,
        }
    }

    #[test]
    fn as_of_desugars_to_where_with_validity_bounds() {
        let mut parser = CypherParser::new();
        let q = parser
            .parse("MATCH (n:Account) AS OF datetime('2026-01-01') RETURN n.balance")
            .unwrap();

        // No AsOf survives — the executor never sees it.
        assert!(
            !q.clauses.iter().any(|c| matches!(c, CypherClause::AsOf(_))),
            "AsOf clause leaked past desugaring"
        );
        // A synthetic WHERE was inserted right after the MATCH.
        assert!(matches!(q.clauses[0], CypherClause::Match(_)));
        let CypherClause::Where(w) = &q.clauses[1] else {
            panic!("expected a synthetic WHERE after MATCH, got {:?}", q.clauses[1]);
        };
        assert!(mentions(w, "valid_from"), "missing valid_from bound");
        assert!(mentions(w, "valid_to"), "missing valid_to bound");
    }

    #[test]
    fn for_system_time_as_of_is_equivalent() {
        let mut parser = CypherParser::new();
        let q = parser
            .parse("MATCH (n) FOR SYSTEM_TIME AS OF datetime('2026-05-01') RETURN n")
            .unwrap();
        assert!(!q.clauses.iter().any(|c| matches!(c, CypherClause::AsOf(_))));
        assert!(matches!(q.clauses[1], CypherClause::Where(_)));
        if let CypherClause::Where(w) = &q.clauses[1] {
            assert!(mentions(w, "valid_from") && mentions(w, "valid_to"));
        }
    }

    #[test]
    fn as_of_conjoins_into_existing_where() {
        let mut parser = CypherParser::new();
        let q = parser
            .parse("MATCH (n:Account) WHERE n.id = 42 AS OF datetime('2026-01-01') RETURN n")
            .unwrap();

        // Exactly one WHERE — the user predicate AND the temporal pin
        // folded together, not two separate WHERE clauses.
        let where_count = q
            .clauses
            .iter()
            .filter(|c| matches!(c, CypherClause::Where(_)))
            .count();
        assert_eq!(where_count, 1, "AS OF should fold into the existing WHERE");
        if let Some(CypherClause::Where(w)) =
            q.clauses.iter().find(|c| matches!(c, CypherClause::Where(_)))
        {
            assert!(mentions(w, "id"), "user predicate dropped");
            assert!(mentions(w, "valid_from") && mentions(w, "valid_to"));
        }
    }

    #[test]
    fn no_as_of_leaves_clauses_untouched() {
        let mut parser = CypherParser::new();
        let q = parser.parse("MATCH (n) WHERE n.x = 1 RETURN n").unwrap();
        // Desugar is a no-op when there's no AS OF.
        assert_eq!(q.clauses.len(), 3);
        assert!(matches!(q.clauses[1], CypherClause::Where(_)));
    }
}
