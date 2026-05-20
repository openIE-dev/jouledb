//! SQL Parser and Executor for JouleDB
//!
//! This module provides SQL query parsing and execution for analytical workloads.
//!
//! ## Supported SQL Features
//!
//! - **SELECT**: Column projection with aliases
//! - **FROM**: Table/column references
//! - **WHERE**: Filtering with comparisons and AND/OR
//! - **JOIN**: INNER, LEFT OUTER, RIGHT OUTER joins
//! - **GROUP BY**: Aggregation grouping
//! - **ORDER BY**: Sorting (ASC/DESC)
//! - **LIMIT**: Result limiting
//! - **Aggregates**: SUM, COUNT, AVG, MIN, MAX
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_amorphic::sql::{SqlParser, SqlExecutor};
//!
//! let query = "SELECT customer, SUM(amount) as total
//!              FROM orders
//!              WHERE date >= 20240101
//!              GROUP BY customer
//!              ORDER BY total DESC
//!              LIMIT 10";
//!
//! let ast = SqlParser::parse(query)?;
//! let executor = SqlExecutor::new(&store);
//! let result = executor.execute(&ast)?;
//! ```

use crate::columnar::ColumnarStore;
use crate::optimizer::{
    AggregateFunc, JoinType, LogicalPlan, PhysicalPlan, Predicate, QueryOptimizer, SortOrder,
};
use crate::{AmorphicError, AmorphicResult, RecordId};

// =============================================================================
// SQL TOKEN TYPES
// =============================================================================

/// SQL token types
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    From,
    Where,
    Join,
    Inner,
    Left,
    Right,
    Outer,
    On,
    And,
    Or,
    Not,
    GroupBy,
    OrderBy,
    Asc,
    Desc,
    Limit,
    As,
    Over,
    PartitionBy,

    // Aggregates
    Sum,
    Count,
    Avg,
    Min,
    Max,

    // Window functions
    RowNumber,
    Rank,
    DenseRank,
    Ntile,
    Lead,
    Lag,
    FirstValue,
    LastValue,

    // Operators
    Eq, // =
    Ne, // <> or !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
    Between,
    In,
    Like,

    // Punctuation
    Comma,
    Dot,
    LParen,
    RParen,
    Star,

    // Literals
    Identifier(String),
    Number(f64),
    String(String),

    // End
    Eof,
}

// =============================================================================
// SQL LEXER
// =============================================================================

/// SQL tokenizer
pub struct SqlLexer {
    input: Vec<char>,
    pos: usize,
}

impl SqlLexer {
    /// Create a new lexer for the given SQL string
    pub fn new(sql: &str) -> Self {
        Self {
            input: sql.chars().collect(),
            pos: 0,
        }
    }

    /// Tokenize the entire input
    pub fn tokenize(&mut self) -> AmorphicResult<Vec<Token>> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = token == Token::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    fn next_token(&mut self) -> AmorphicResult<Token> {
        self.skip_whitespace();

        if self.pos >= self.input.len() {
            return Ok(Token::Eof);
        }

        let ch = self.input[self.pos];

        // Single character tokens
        match ch {
            ',' => {
                self.pos += 1;
                return Ok(Token::Comma);
            }
            '.' => {
                self.pos += 1;
                return Ok(Token::Dot);
            }
            '(' => {
                self.pos += 1;
                return Ok(Token::LParen);
            }
            ')' => {
                self.pos += 1;
                return Ok(Token::RParen);
            }
            '*' => {
                self.pos += 1;
                return Ok(Token::Star);
            }
            '=' => {
                self.pos += 1;
                return Ok(Token::Eq);
            }
            _ => {}
        }

        // Two-character operators
        if ch == '<' {
            self.pos += 1;
            if self.pos < self.input.len() {
                match self.input[self.pos] {
                    '=' => {
                        self.pos += 1;
                        return Ok(Token::Le);
                    }
                    '>' => {
                        self.pos += 1;
                        return Ok(Token::Ne);
                    }
                    _ => return Ok(Token::Lt),
                }
            }
            return Ok(Token::Lt);
        }

        if ch == '>' {
            self.pos += 1;
            if self.pos < self.input.len() && self.input[self.pos] == '=' {
                self.pos += 1;
                return Ok(Token::Ge);
            }
            return Ok(Token::Gt);
        }

        if ch == '!' {
            self.pos += 1;
            if self.pos < self.input.len() && self.input[self.pos] == '=' {
                self.pos += 1;
                return Ok(Token::Ne);
            }
            return Err(AmorphicError::InvalidQuery("Unexpected '!'".to_string()));
        }

        // String literal
        if ch == '\'' || ch == '"' {
            return self.read_string(ch);
        }

        // Number
        if ch.is_ascii_digit()
            || (ch == '-'
                && self
                    .peek_ahead()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false))
        {
            return self.read_number();
        }

        // Identifier or keyword
        if ch.is_alphabetic() || ch == '_' {
            return self.read_identifier();
        }

        Err(AmorphicError::InvalidQuery(format!(
            "Unexpected character: '{}'",
            ch
        )))
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() && self.input[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek_ahead(&self) -> Option<char> {
        if self.pos + 1 < self.input.len() {
            Some(self.input[self.pos + 1])
        } else {
            None
        }
    }

    fn read_string(&mut self, quote: char) -> AmorphicResult<Token> {
        self.pos += 1; // Skip opening quote
        let start = self.pos;

        while self.pos < self.input.len() && self.input[self.pos] != quote {
            self.pos += 1;
        }

        if self.pos >= self.input.len() {
            return Err(AmorphicError::InvalidQuery(
                "Unterminated string".to_string(),
            ));
        }

        let s: String = self.input[start..self.pos].iter().collect();
        self.pos += 1; // Skip closing quote

        Ok(Token::String(s))
    }

    fn read_number(&mut self) -> AmorphicResult<Token> {
        let start = self.pos;

        if self.input[self.pos] == '-' {
            self.pos += 1;
        }

        while self.pos < self.input.len()
            && (self.input[self.pos].is_ascii_digit() || self.input[self.pos] == '.')
        {
            self.pos += 1;
        }

        let s: String = self.input[start..self.pos].iter().collect();
        let num: f64 = s
            .parse()
            .map_err(|_| AmorphicError::InvalidQuery(format!("Invalid number: {}", s)))?;

        Ok(Token::Number(num))
    }

    fn read_identifier(&mut self) -> AmorphicResult<Token> {
        let start = self.pos;

        while self.pos < self.input.len()
            && (self.input[self.pos].is_alphanumeric() || self.input[self.pos] == '_')
        {
            self.pos += 1;
        }

        let s: String = self.input[start..self.pos].iter().collect();
        let upper = s.to_uppercase();

        // Check for keywords
        let token = match upper.as_str() {
            "SELECT" => Token::Select,
            "FROM" => Token::From,
            "WHERE" => Token::Where,
            "JOIN" => Token::Join,
            "INNER" => Token::Inner,
            "LEFT" => Token::Left,
            "RIGHT" => Token::Right,
            "OUTER" => Token::Outer,
            "ON" => Token::On,
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "GROUP" => Token::GroupBy,         // Will need to check for BY
            "ORDER" => Token::OrderBy,         // Will need to check for BY
            "PARTITION" => Token::PartitionBy, // Will need to check for BY
            "BY" => Token::Identifier("BY".to_string()), // Handle separately
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "LIMIT" => Token::Limit,
            "AS" => Token::As,
            "OVER" => Token::Over,
            // Aggregate functions
            "SUM" => Token::Sum,
            "COUNT" => Token::Count,
            "AVG" => Token::Avg,
            "MIN" => Token::Min,
            "MAX" => Token::Max,
            // Window functions
            "ROW_NUMBER" => Token::RowNumber,
            "RANK" => Token::Rank,
            "DENSE_RANK" => Token::DenseRank,
            "NTILE" => Token::Ntile,
            "LEAD" => Token::Lead,
            "LAG" => Token::Lag,
            "FIRST_VALUE" => Token::FirstValue,
            "LAST_VALUE" => Token::LastValue,
            // Other
            "BETWEEN" => Token::Between,
            "IN" => Token::In,
            "LIKE" => Token::Like,
            _ => Token::Identifier(s),
        };

        Ok(token)
    }
}

// =============================================================================
// SQL AST
// =============================================================================

/// Parsed SQL statement
#[derive(Debug, Clone)]
pub enum SqlStatement {
    Select(SelectStatement),
}

/// SELECT statement
#[derive(Debug, Clone)]
pub struct SelectStatement {
    /// Columns to select
    pub columns: Vec<SelectColumn>,
    /// FROM clause
    pub from: Option<FromClause>,
    /// WHERE clause
    pub where_clause: Option<WhereClause>,
    /// JOIN clauses
    pub joins: Vec<JoinClause>,
    /// GROUP BY columns
    pub group_by: Vec<String>,
    /// ORDER BY clause
    pub order_by: Vec<OrderByItem>,
    /// LIMIT value
    pub limit: Option<usize>,
}

/// Selected column
#[derive(Debug, Clone)]
pub enum SelectColumn {
    /// All columns (*)
    Star,
    /// Named column
    Column { name: String, alias: Option<String> },
    /// Aggregate function
    Aggregate {
        func: AggregateFunc,
        column: String,
        alias: Option<String>,
    },
    /// Window function
    WindowFunction {
        func: WindowFunc,
        partition_by: Vec<String>,
        order_by: Vec<(String, SortOrder)>,
        alias: Option<String>,
    },
}

/// Window function types
#[derive(Debug, Clone, PartialEq)]
pub enum WindowFunc {
    /// ROW_NUMBER() - sequential row numbers
    RowNumber,
    /// RANK() - ranking with gaps for ties
    Rank,
    /// DENSE_RANK() - ranking without gaps
    DenseRank,
    /// NTILE(n) - divide into n buckets
    Ntile(usize),
    /// LEAD(col, offset) - value from following row
    Lead(String, usize),
    /// LAG(col, offset) - value from preceding row
    Lag(String, usize),
    /// FIRST_VALUE(col) - first value in partition
    FirstValue(String),
    /// LAST_VALUE(col) - last value in partition
    LastValue(String),
    /// SUM(col) OVER - running sum
    RunningSum(String),
    /// AVG(col) OVER - running average
    RunningAvg(String),
    /// COUNT(*) OVER - running count
    RunningCount,
}

/// FROM clause
#[derive(Debug, Clone)]
pub struct FromClause {
    pub table: String,
    pub alias: Option<String>,
}

/// WHERE clause
#[derive(Debug, Clone)]
pub enum WhereClause {
    /// Comparison: column op value
    Comparison {
        column: String,
        op: CompareOp,
        value: SqlValue,
    },
    /// BETWEEN: column BETWEEN min AND max
    Between {
        column: String,
        min: SqlValue,
        max: SqlValue,
    },
    /// IN: column IN (values)
    In {
        column: String,
        values: Vec<SqlValue>,
    },
    /// AND of clauses
    And(Vec<WhereClause>),
    /// OR of clauses
    Or(Vec<WhereClause>),
    /// NOT clause
    Not(Box<WhereClause>),
}

/// Comparison operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// SQL value
#[derive(Debug, Clone)]
pub enum SqlValue {
    Number(f64),
    String(String),
    Null,
}

impl SqlValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SqlValue::Number(n) => Some(*n),
            SqlValue::String(s) => s.parse().ok(),
            SqlValue::Null => None,
        }
    }
}

/// JOIN clause
#[derive(Debug, Clone)]
pub struct JoinClause {
    pub join_type: JoinType,
    pub table: String,
    pub alias: Option<String>,
    pub on_left: String,
    pub on_right: String,
}

/// ORDER BY item
#[derive(Debug, Clone)]
pub struct OrderByItem {
    pub column: String,
    pub order: SortOrder,
}

// =============================================================================
// SQL PARSER
// =============================================================================

/// SQL parser
pub struct SqlParser {
    tokens: Vec<Token>,
    pos: usize,
}

impl SqlParser {
    /// Parse a SQL query string
    pub fn parse(sql: &str) -> AmorphicResult<SqlStatement> {
        let mut lexer = SqlLexer::new(sql);
        let tokens = lexer.tokenize()?;

        let mut parser = Self { tokens, pos: 0 };
        parser.parse_statement()
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: Token) -> AmorphicResult<()> {
        if self.current() == &expected {
            self.advance();
            Ok(())
        } else {
            Err(AmorphicError::InvalidQuery(format!(
                "Expected {:?}, got {:?}",
                expected,
                self.current()
            )))
        }
    }

    fn parse_statement(&mut self) -> AmorphicResult<SqlStatement> {
        match self.current() {
            Token::Select => self.parse_select(),
            other => Err(AmorphicError::InvalidQuery(format!(
                "Expected SELECT, got {:?}",
                other
            ))),
        }
    }

    fn parse_select(&mut self) -> AmorphicResult<SqlStatement> {
        self.expect(Token::Select)?;

        // Parse columns
        let columns = self.parse_select_columns()?;

        // Parse FROM
        let from = if self.current() == &Token::From {
            self.advance();
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        // Parse JOINs
        let mut joins = Vec::new();
        while matches!(
            self.current(),
            Token::Join | Token::Inner | Token::Left | Token::Right
        ) {
            joins.push(self.parse_join_clause()?);
        }

        // Parse WHERE
        let where_clause = if self.current() == &Token::Where {
            self.advance();
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Parse GROUP BY
        let group_by = if self.current() == &Token::GroupBy {
            self.advance();
            // Skip "BY" if present
            if let Token::Identifier(s) = self.current() {
                if s.to_uppercase() == "BY" {
                    self.advance();
                }
            }
            self.parse_column_list()?
        } else {
            Vec::new()
        };

        // Parse ORDER BY
        let order_by = if self.current() == &Token::OrderBy {
            self.advance();
            // Skip "BY" if present
            if let Token::Identifier(s) = self.current() {
                if s.to_uppercase() == "BY" {
                    self.advance();
                }
            }
            self.parse_order_by_list()?
        } else {
            Vec::new()
        };

        // Parse LIMIT
        let limit = if self.current() == &Token::Limit {
            self.advance();
            if let Token::Number(n) = self.current() {
                let l = *n as usize;
                self.advance();
                Some(l)
            } else {
                return Err(AmorphicError::InvalidQuery(
                    "Expected number after LIMIT".to_string(),
                ));
            }
        } else {
            None
        };

        Ok(SqlStatement::Select(SelectStatement {
            columns,
            from,
            where_clause,
            joins,
            group_by,
            order_by,
            limit,
        }))
    }

    fn parse_select_columns(&mut self) -> AmorphicResult<Vec<SelectColumn>> {
        let mut columns = Vec::new();

        loop {
            let col = self.parse_select_column()?;
            columns.push(col);

            if self.current() == &Token::Comma {
                self.advance();
            } else {
                break;
            }
        }

        Ok(columns)
    }

    fn parse_select_column(&mut self) -> AmorphicResult<SelectColumn> {
        // Check for *
        if self.current() == &Token::Star {
            self.advance();
            return Ok(SelectColumn::Star);
        }

        // Check for window functions (ROW_NUMBER, RANK, etc.)
        if let Some(window_func) = self.try_parse_window_function()? {
            return Ok(window_func);
        }

        // Check for aggregate - might be window function if followed by OVER
        let agg_func = match self.current() {
            Token::Sum => Some(AggregateFunc::Sum),
            Token::Count => Some(AggregateFunc::Count),
            Token::Avg => Some(AggregateFunc::Avg),
            Token::Min => Some(AggregateFunc::Min),
            Token::Max => Some(AggregateFunc::Max),
            _ => None,
        };

        if let Some(func) = agg_func {
            self.advance();
            self.expect(Token::LParen)?;

            let column = if self.current() == &Token::Star {
                self.advance();
                "*".to_string()
            } else {
                self.parse_qualified_name()?
            };

            self.expect(Token::RParen)?;

            // Check if this is a window function (aggregate OVER)
            if self.current() == &Token::Over {
                let window_func = match func {
                    AggregateFunc::Sum => WindowFunc::RunningSum(column),
                    AggregateFunc::Avg => WindowFunc::RunningAvg(column),
                    AggregateFunc::Count => WindowFunc::RunningCount,
                    _ => {
                        return Err(AmorphicError::InvalidQuery(format!(
                            "{:?} is not supported as a window function",
                            func
                        )));
                    }
                };
                let (partition_by, order_by) = self.parse_over_clause()?;
                let alias = self.parse_alias()?;
                return Ok(SelectColumn::WindowFunction {
                    func: window_func,
                    partition_by,
                    order_by,
                    alias,
                });
            }

            let alias = self.parse_alias()?;

            return Ok(SelectColumn::Aggregate {
                func,
                column,
                alias,
            });
        }

        // Regular column (possibly qualified like table.column)
        let name = self.parse_qualified_name()?;
        let alias = self.parse_alias()?;
        Ok(SelectColumn::Column { name, alias })
    }

    /// Try to parse a window function (ROW_NUMBER, RANK, etc.)
    fn try_parse_window_function(&mut self) -> AmorphicResult<Option<SelectColumn>> {
        let window_func = match self.current() {
            Token::RowNumber => {
                self.advance();
                self.expect(Token::LParen)?;
                self.expect(Token::RParen)?;
                WindowFunc::RowNumber
            }
            Token::Rank => {
                self.advance();
                self.expect(Token::LParen)?;
                self.expect(Token::RParen)?;
                WindowFunc::Rank
            }
            Token::DenseRank => {
                self.advance();
                self.expect(Token::LParen)?;
                self.expect(Token::RParen)?;
                WindowFunc::DenseRank
            }
            Token::Ntile => {
                self.advance();
                self.expect(Token::LParen)?;
                let n = if let Token::Number(n) = self.current().clone() {
                    self.advance();
                    n as usize
                } else {
                    return Err(AmorphicError::InvalidQuery(
                        "NTILE requires a number argument".to_string(),
                    ));
                };
                self.expect(Token::RParen)?;
                WindowFunc::Ntile(n)
            }
            Token::Lead => {
                self.advance();
                self.expect(Token::LParen)?;
                let column = self.parse_qualified_name()?;
                let offset = if self.current() == &Token::Comma {
                    self.advance();
                    if let Token::Number(n) = self.current().clone() {
                        self.advance();
                        n as usize
                    } else {
                        1
                    }
                } else {
                    1
                };
                self.expect(Token::RParen)?;
                WindowFunc::Lead(column, offset)
            }
            Token::Lag => {
                self.advance();
                self.expect(Token::LParen)?;
                let column = self.parse_qualified_name()?;
                let offset = if self.current() == &Token::Comma {
                    self.advance();
                    if let Token::Number(n) = self.current().clone() {
                        self.advance();
                        n as usize
                    } else {
                        1
                    }
                } else {
                    1
                };
                self.expect(Token::RParen)?;
                WindowFunc::Lag(column, offset)
            }
            Token::FirstValue => {
                self.advance();
                self.expect(Token::LParen)?;
                let column = self.parse_qualified_name()?;
                self.expect(Token::RParen)?;
                WindowFunc::FirstValue(column)
            }
            Token::LastValue => {
                self.advance();
                self.expect(Token::LParen)?;
                let column = self.parse_qualified_name()?;
                self.expect(Token::RParen)?;
                WindowFunc::LastValue(column)
            }
            _ => return Ok(None),
        };

        // Must have OVER clause
        let (partition_by, order_by) = self.parse_over_clause()?;
        let alias = self.parse_alias()?;

        Ok(Some(SelectColumn::WindowFunction {
            func: window_func,
            partition_by,
            order_by,
            alias,
        }))
    }

    /// Parse OVER (PARTITION BY col ORDER BY col ASC/DESC)
    fn parse_over_clause(&mut self) -> AmorphicResult<(Vec<String>, Vec<(String, SortOrder)>)> {
        self.expect(Token::Over)?;
        self.expect(Token::LParen)?;

        let mut partition_by = Vec::new();
        let mut order_by = Vec::new();

        // Parse PARTITION BY (optional)
        // Note: lexer tokenizes "PARTITION" as PartitionBy, "BY" as separate identifier
        if self.current() == &Token::PartitionBy {
            self.advance();
            // Skip the "BY" token if present
            if let Token::Identifier(s) = self.current() {
                if s.to_uppercase() == "BY" {
                    self.advance();
                }
            }
            loop {
                let col = self.parse_qualified_name()?;
                partition_by.push(col);
                if self.current() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        // Parse ORDER BY (optional but usually present)
        // Note: lexer tokenizes "ORDER" as OrderBy, "BY" as separate identifier
        if self.current() == &Token::OrderBy {
            self.advance();
            // Skip the "BY" token if present
            if let Token::Identifier(s) = self.current() {
                if s.to_uppercase() == "BY" {
                    self.advance();
                }
            }
            loop {
                let col = self.parse_qualified_name()?;
                let order = match self.current() {
                    Token::Desc => {
                        self.advance();
                        SortOrder::Descending
                    }
                    Token::Asc => {
                        self.advance();
                        SortOrder::Ascending
                    }
                    _ => SortOrder::Ascending,
                };
                order_by.push((col, order));
                if self.current() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(Token::RParen)?;

        Ok((partition_by, order_by))
    }

    /// Parse a qualified name like "table.column" or just "column"
    fn parse_qualified_name(&mut self) -> AmorphicResult<String> {
        if let Token::Identifier(name) = self.current().clone() {
            self.advance();

            // Check for dot (qualified name)
            if self.current() == &Token::Dot {
                self.advance();
                if let Token::Identifier(col_name) = self.current().clone() {
                    self.advance();
                    return Ok(format!("{}.{}", name, col_name));
                }
            }

            return Ok(name);
        }

        Err(AmorphicError::InvalidQuery(
            "Expected identifier".to_string(),
        ))
    }

    fn parse_alias(&mut self) -> AmorphicResult<Option<String>> {
        if self.current() == &Token::As {
            self.advance();
        }

        if let Token::Identifier(name) = self.current().clone() {
            self.advance();
            return Ok(Some(name));
        }

        Ok(None)
    }

    fn parse_from_clause(&mut self) -> AmorphicResult<FromClause> {
        if let Token::Identifier(table) = self.current().clone() {
            self.advance();
            let alias = self.parse_alias()?;
            return Ok(FromClause { table, alias });
        }

        Err(AmorphicError::InvalidQuery(
            "Expected table name".to_string(),
        ))
    }

    fn parse_join_clause(&mut self) -> AmorphicResult<JoinClause> {
        let join_type = match self.current() {
            Token::Inner => {
                self.advance();
                self.expect(Token::Join)?;
                JoinType::Inner
            }
            Token::Left => {
                self.advance();
                if self.current() == &Token::Outer {
                    self.advance();
                }
                self.expect(Token::Join)?;
                JoinType::LeftOuter
            }
            Token::Right => {
                self.advance();
                if self.current() == &Token::Outer {
                    self.advance();
                }
                self.expect(Token::Join)?;
                JoinType::RightOuter
            }
            Token::Join => {
                self.advance();
                JoinType::Inner
            }
            other => {
                return Err(AmorphicError::InvalidQuery(format!(
                    "Unexpected {:?} in JOIN",
                    other
                )));
            }
        };

        // Table name
        let table = if let Token::Identifier(name) = self.current().clone() {
            self.advance();
            name
        } else {
            return Err(AmorphicError::InvalidQuery(
                "Expected table name in JOIN".to_string(),
            ));
        };

        let alias = self.parse_alias()?;

        // ON clause
        self.expect(Token::On)?;

        let on_left = self.parse_qualified_name()?;

        self.expect(Token::Eq)?;

        let on_right = self.parse_qualified_name()?;

        Ok(JoinClause {
            join_type,
            table,
            alias,
            on_left,
            on_right,
        })
    }

    fn parse_where_clause(&mut self) -> AmorphicResult<WhereClause> {
        self.parse_where_or()
    }

    fn parse_where_or(&mut self) -> AmorphicResult<WhereClause> {
        let mut left = self.parse_where_and()?;

        while self.current() == &Token::Or {
            self.advance();
            let right = self.parse_where_and()?;
            left = WhereClause::Or(vec![left, right]);
        }

        Ok(left)
    }

    fn parse_where_and(&mut self) -> AmorphicResult<WhereClause> {
        let mut left = self.parse_where_primary()?;

        while self.current() == &Token::And {
            self.advance();
            let right = self.parse_where_primary()?;
            left = WhereClause::And(vec![left, right]);
        }

        Ok(left)
    }

    fn parse_where_primary(&mut self) -> AmorphicResult<WhereClause> {
        // NOT
        if self.current() == &Token::Not {
            self.advance();
            let inner = self.parse_where_primary()?;
            return Ok(WhereClause::Not(Box::new(inner)));
        }

        // Parenthesized expression
        if self.current() == &Token::LParen {
            self.advance();
            let inner = self.parse_where_clause()?;
            self.expect(Token::RParen)?;
            return Ok(inner);
        }

        // Column comparison (possibly qualified name)
        let column = self.parse_qualified_name()?;

        // BETWEEN
        if self.current() == &Token::Between {
            self.advance();
            let min = self.parse_value()?;
            self.expect(Token::And)?;
            let max = self.parse_value()?;
            return Ok(WhereClause::Between { column, min, max });
        }

        // IN
        if self.current() == &Token::In {
            self.advance();
            self.expect(Token::LParen)?;
            let values = self.parse_value_list()?;
            self.expect(Token::RParen)?;
            return Ok(WhereClause::In { column, values });
        }

        // Regular comparison
        let op = match self.current() {
            Token::Eq => CompareOp::Eq,
            Token::Ne => CompareOp::Ne,
            Token::Lt => CompareOp::Lt,
            Token::Le => CompareOp::Le,
            Token::Gt => CompareOp::Gt,
            Token::Ge => CompareOp::Ge,
            other => {
                return Err(AmorphicError::InvalidQuery(format!(
                    "Expected comparison operator, got {:?}",
                    other
                )));
            }
        };
        self.advance();

        let value = self.parse_value()?;

        Ok(WhereClause::Comparison { column, op, value })
    }

    fn parse_value(&mut self) -> AmorphicResult<SqlValue> {
        match self.current().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(SqlValue::Number(n))
            }
            Token::String(s) => {
                self.advance();
                Ok(SqlValue::String(s))
            }
            Token::Identifier(s) if s.to_uppercase() == "NULL" => {
                self.advance();
                Ok(SqlValue::Null)
            }
            other => Err(AmorphicError::InvalidQuery(format!(
                "Expected value, got {:?}",
                other
            ))),
        }
    }

    fn parse_value_list(&mut self) -> AmorphicResult<Vec<SqlValue>> {
        let mut values = Vec::new();

        loop {
            values.push(self.parse_value()?);

            if self.current() == &Token::Comma {
                self.advance();
            } else {
                break;
            }
        }

        Ok(values)
    }

    fn parse_column_list(&mut self) -> AmorphicResult<Vec<String>> {
        let mut columns = Vec::new();

        loop {
            if let Token::Identifier(name) = self.current().clone() {
                self.advance();
                columns.push(name);

                if self.current() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(columns)
    }

    fn parse_order_by_list(&mut self) -> AmorphicResult<Vec<OrderByItem>> {
        let mut items = Vec::new();

        loop {
            if let Token::Identifier(column) = self.current().clone() {
                self.advance();

                let order = match self.current() {
                    Token::Asc => {
                        self.advance();
                        SortOrder::Ascending
                    }
                    Token::Desc => {
                        self.advance();
                        SortOrder::Descending
                    }
                    _ => SortOrder::Ascending,
                };

                items.push(OrderByItem { column, order });

                if self.current() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(items)
    }
}

// =============================================================================
// SQL TO LOGICAL PLAN CONVERSION
// =============================================================================

impl SelectStatement {
    /// Convert to a logical plan
    pub fn to_logical_plan(&self) -> AmorphicResult<LogicalPlan> {
        // Start with FROM clause
        let table_name = self
            .from
            .as_ref()
            .map(|f| f.table.clone())
            .unwrap_or_else(|| "default".to_string());

        let mut builder = LogicalPlan::scan(&table_name);

        // Add columns from SELECT
        let column_names: Vec<&str> = self
            .columns
            .iter()
            .filter_map(|c| match c {
                SelectColumn::Column { name, .. } => Some(name.as_str()),
                SelectColumn::Aggregate { column, .. } => Some(column.as_str()),
                SelectColumn::WindowFunction { .. } => None, // Window functions handled separately
                SelectColumn::Star => None,
            })
            .collect();

        if !column_names.is_empty() {
            builder = builder.columns(column_names);
        }

        // Add JOINs
        for join in &self.joins {
            let right_plan = LogicalPlan::scan(&join.table).build();

            builder = builder.join(right_plan, &join.on_left, &join.on_right, join.join_type);
        }

        // Add WHERE clause
        if let Some(where_clause) = &self.where_clause {
            let predicate = where_clause.to_predicate()?;
            builder = builder.filter(predicate);
        }

        // Add GROUP BY
        if !self.group_by.is_empty() {
            let group_keys: Vec<&str> = self.group_by.iter().map(|s| s.as_str()).collect();
            let aggregates: Vec<(&str, AggregateFunc, &str)> = self
                .columns
                .iter()
                .filter_map(|c| match c {
                    SelectColumn::Aggregate {
                        func,
                        column,
                        alias,
                    } => {
                        let out_name = alias.as_ref().unwrap_or(column);
                        Some((out_name.as_str(), *func, column.as_str()))
                    }
                    _ => None,
                })
                .collect();

            builder = builder.group_by(group_keys, aggregates);
        }

        // Add ORDER BY
        if !self.order_by.is_empty() {
            let sort_keys: Vec<(&str, SortOrder)> = self
                .order_by
                .iter()
                .map(|o| (o.column.as_str(), o.order))
                .collect();
            builder = builder.sort(sort_keys);
        }

        // Add LIMIT
        if let Some(limit) = self.limit {
            builder = builder.limit(limit);
        }

        Ok(builder.build())
    }
}

impl WhereClause {
    /// Convert to optimizer predicate
    fn to_predicate(&self) -> AmorphicResult<Predicate> {
        match self {
            WhereClause::Comparison { column, op, value } => {
                let num = value.as_f64().ok_or_else(|| {
                    AmorphicError::InvalidQuery("Non-numeric comparison value".to_string())
                })?;

                match op {
                    CompareOp::Eq => Ok(Predicate::Equals {
                        field: column.clone(),
                        value: num,
                    }),
                    CompareOp::Gt => Ok(Predicate::Range {
                        field: column.clone(),
                        min: num + f64::EPSILON,
                        max: f64::MAX,
                    }),
                    CompareOp::Ge => Ok(Predicate::Range {
                        field: column.clone(),
                        min: num,
                        max: f64::MAX,
                    }),
                    CompareOp::Lt => Ok(Predicate::Range {
                        field: column.clone(),
                        min: f64::MIN,
                        max: num - f64::EPSILON,
                    }),
                    CompareOp::Le => Ok(Predicate::Range {
                        field: column.clone(),
                        min: f64::MIN,
                        max: num,
                    }),
                    CompareOp::Ne => Ok(Predicate::Not(Box::new(Predicate::Equals {
                        field: column.clone(),
                        value: num,
                    }))),
                }
            }
            WhereClause::Between { column, min, max } => {
                let min_num = min.as_f64().ok_or_else(|| {
                    AmorphicError::InvalidQuery("Non-numeric BETWEEN value".to_string())
                })?;
                let max_num = max.as_f64().ok_or_else(|| {
                    AmorphicError::InvalidQuery("Non-numeric BETWEEN value".to_string())
                })?;

                Ok(Predicate::Range {
                    field: column.clone(),
                    min: min_num,
                    max: max_num,
                })
            }
            WhereClause::In { column, values } => {
                let nums: Result<Vec<f64>, _> = values
                    .iter()
                    .map(|v| {
                        v.as_f64().ok_or_else(|| {
                            AmorphicError::InvalidQuery("Non-numeric IN value".to_string())
                        })
                    })
                    .collect();

                Ok(Predicate::In {
                    field: column.clone(),
                    values: nums?,
                })
            }
            WhereClause::And(clauses) => {
                let preds: Result<Vec<Predicate>, _> =
                    clauses.iter().map(|c| c.to_predicate()).collect();
                Ok(Predicate::And(preds?))
            }
            WhereClause::Or(clauses) => {
                let preds: Result<Vec<Predicate>, _> =
                    clauses.iter().map(|c| c.to_predicate()).collect();
                Ok(Predicate::Or(preds?))
            }
            WhereClause::Not(inner) => Ok(Predicate::Not(Box::new(inner.to_predicate()?))),
        }
    }
}

// =============================================================================
// SQL EXECUTOR
// =============================================================================

/// SQL query result
#[derive(Debug)]
pub struct SqlResult {
    /// Column names
    pub columns: Vec<String>,
    /// Rows of values
    pub rows: Vec<Vec<SqlValue>>,
    /// Number of rows
    pub row_count: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

impl SqlResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            row_count: 0,
            execution_time_ms: 0.0,
        }
    }

    /// Format as a table string
    pub fn to_table(&self) -> String {
        if self.columns.is_empty() {
            return "Empty result".to_string();
        }

        // Calculate column widths
        let mut widths: Vec<usize> = self.columns.iter().map(|c| c.len()).collect();

        for row in &self.rows {
            for (i, val) in row.iter().enumerate() {
                let s = format!("{:?}", val);
                if i < widths.len() {
                    widths[i] = widths[i].max(s.len());
                }
            }
        }

        let mut output = String::new();

        // Header
        for (i, col) in self.columns.iter().enumerate() {
            if i > 0 {
                output.push_str(" | ");
            }
            output.push_str(&format!("{:width$}", col, width = widths[i]));
        }
        output.push('\n');

        // Separator
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                output.push_str("-+-");
            }
            output.push_str(&"-".repeat(*w));
        }
        output.push('\n');

        // Rows
        for row in &self.rows {
            for (i, val) in row.iter().enumerate() {
                if i > 0 {
                    output.push_str(" | ");
                }
                let s = match val {
                    SqlValue::Number(n) => format!("{:.2}", n),
                    SqlValue::String(s) => s.clone(),
                    SqlValue::Null => "NULL".to_string(),
                };
                if i < widths.len() {
                    output.push_str(&format!("{:width$}", s, width = widths[i]));
                }
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "\n({} rows, {:.2}ms)",
            self.row_count, self.execution_time_ms
        ));

        output
    }
}

/// SQL executor
pub struct SqlExecutor<'a> {
    columnar: &'a ColumnarStore,
}

impl<'a> SqlExecutor<'a> {
    /// Create a new executor
    pub fn new(columnar: &'a ColumnarStore) -> Self {
        Self { columnar }
    }

    /// Execute a SQL query string
    pub fn execute_sql(&self, sql: &str) -> AmorphicResult<SqlResult> {
        let start = std::time::Instant::now();

        // Parse
        let statement = SqlParser::parse(sql)?;

        // Execute
        let mut result = self.execute(&statement)?;

        result.execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(result)
    }

    /// Execute a parsed statement
    pub fn execute(&self, statement: &SqlStatement) -> AmorphicResult<SqlResult> {
        match statement {
            SqlStatement::Select(select) => self.execute_select(select),
        }
    }

    fn execute_select(&self, select: &SelectStatement) -> AmorphicResult<SqlResult> {
        // Convert to logical plan
        let logical = select.to_logical_plan()?;

        // Optimize
        let optimizer = QueryOptimizer::new(self.columnar);
        let physical = optimizer.plan(logical);

        // Execute physical plan
        self.execute_physical(&physical, select)
    }

    fn execute_physical(
        &self,
        plan: &PhysicalPlan,
        select: &SelectStatement,
    ) -> AmorphicResult<SqlResult> {
        // Build column names from SELECT
        let columns: Vec<String> = select
            .columns
            .iter()
            .map(|c| match c {
                SelectColumn::Star => "*".to_string(),
                SelectColumn::Column { name, alias } => {
                    alias.clone().unwrap_or_else(|| name.clone())
                }
                SelectColumn::Aggregate {
                    func,
                    column,
                    alias,
                } => alias
                    .clone()
                    .unwrap_or_else(|| format!("{:?}({})", func, column)),
                SelectColumn::WindowFunction { func, alias, .. } => {
                    alias.clone().unwrap_or_else(|| format!("{:?}", func))
                }
            })
            .collect();

        // For simple queries, use columnar store directly
        let rows = self.execute_simple(select)?;
        let row_count = rows.len();

        Ok(SqlResult {
            columns,
            rows,
            row_count,
            execution_time_ms: 0.0,
        })
    }

    fn execute_simple(&self, select: &SelectStatement) -> AmorphicResult<Vec<Vec<SqlValue>>> {
        // Check for window functions
        let has_window = select
            .columns
            .iter()
            .any(|c| matches!(c, SelectColumn::WindowFunction { .. }));

        if has_window {
            return self.execute_window(select);
        }

        // Check for aggregates without GROUP BY (scalar aggregation)
        let has_agg = select
            .columns
            .iter()
            .any(|c| matches!(c, SelectColumn::Aggregate { .. }));

        if has_agg && select.group_by.is_empty() {
            return self.execute_scalar_aggregation(select);
        }

        // Check for GROUP BY
        if !select.group_by.is_empty() {
            return self.execute_group_by(select);
        }

        // Simple scan/filter
        self.execute_scan(select)
    }

    fn execute_scalar_aggregation(
        &self,
        select: &SelectStatement,
    ) -> AmorphicResult<Vec<Vec<SqlValue>>> {
        let mut row = Vec::new();

        for col in &select.columns {
            if let SelectColumn::Aggregate { func, column, .. } = col {
                let value = match func {
                    AggregateFunc::Count => self
                        .columnar
                        .count(column)
                        .map(|c| SqlValue::Number(c as f64))
                        .unwrap_or(SqlValue::Null),
                    AggregateFunc::Sum => self
                        .columnar
                        .sum(column)
                        .map(SqlValue::Number)
                        .unwrap_or(SqlValue::Null),
                    AggregateFunc::Avg => self
                        .columnar
                        .avg(column)
                        .map(SqlValue::Number)
                        .unwrap_or(SqlValue::Null),
                    AggregateFunc::Min => self
                        .columnar
                        .min(column)
                        .map(SqlValue::Number)
                        .unwrap_or(SqlValue::Null),
                    AggregateFunc::Max => self
                        .columnar
                        .max(column)
                        .map(SqlValue::Number)
                        .unwrap_or(SqlValue::Null),
                };
                row.push(value);
            }
        }

        Ok(vec![row])
    }

    fn execute_group_by(&self, select: &SelectStatement) -> AmorphicResult<Vec<Vec<SqlValue>>> {
        // Get the first aggregate
        let agg = select.columns.iter().find_map(|c| match c {
            SelectColumn::Aggregate { func, column, .. } => Some((func, column)),
            _ => None,
        });

        let group_field = select.group_by.first().ok_or_else(|| {
            AmorphicError::InvalidQuery("GROUP BY requires at least one column".to_string())
        })?;

        if let Some((func, agg_field)) = agg {
            let groups = match func {
                AggregateFunc::Sum => self.columnar.group_by_sum(group_field, agg_field),
                AggregateFunc::Count => self
                    .columnar
                    .group_by_count(group_field)
                    .map(|m| m.into_iter().map(|(k, v)| (k, v as f64)).collect()),
                AggregateFunc::Avg => self.columnar.group_by_avg(group_field, agg_field),
                AggregateFunc::Min => self.columnar.group_by_min(group_field, agg_field),
                AggregateFunc::Max => self.columnar.group_by_max(group_field, agg_field),
            };

            if let Some(groups) = groups {
                let mut rows: Vec<Vec<SqlValue>> = groups
                    .into_iter()
                    .map(|(key, value)| vec![SqlValue::Number(key as f64), SqlValue::Number(value)])
                    .collect();

                // Apply LIMIT
                if let Some(limit) = select.limit {
                    rows.truncate(limit);
                }

                return Ok(rows);
            }
        }

        Ok(Vec::new())
    }

    fn execute_scan(&self, select: &SelectStatement) -> AmorphicResult<Vec<Vec<SqlValue>>> {
        // Get column to scan
        let column_name = select
            .columns
            .first()
            .and_then(|c| match c {
                SelectColumn::Column { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .or_else(|| select.from.as_ref().map(|f| f.table.as_str()))
            .unwrap_or("*");

        // Apply WHERE filter if present
        let values: Vec<(RecordId, f64)> = if let Some(where_clause) = &select.where_clause {
            self.apply_where_filter(column_name, where_clause)?
        } else if let Some(col) = self.columnar.get_column(column_name) {
            col.scan().collect()
        } else {
            Vec::new()
        };

        // Convert to rows
        let mut rows: Vec<Vec<SqlValue>> = values
            .into_iter()
            .map(|(_, v)| vec![SqlValue::Number(v)])
            .collect();

        // Apply LIMIT
        if let Some(limit) = select.limit {
            rows.truncate(limit);
        }

        Ok(rows)
    }

    fn apply_where_filter(
        &self,
        column: &str,
        where_clause: &WhereClause,
    ) -> AmorphicResult<Vec<(RecordId, f64)>> {
        let col = self
            .columnar
            .get_column(column)
            .ok_or_else(|| AmorphicError::QueryError(format!("Column not found: {}", column)))?;

        match where_clause {
            WhereClause::Comparison {
                column: filter_col,
                op,
                value,
            } => {
                let filter = self.columnar.get_column(filter_col);
                let num = value.as_f64().unwrap_or(0.0);

                if let Some(filter_col_data) = filter {
                    let filtered: Vec<(RecordId, f64)> = col
                        .scan()
                        .filter(|(id, _)| {
                            if let Some(filter_val) = filter_col_data.get_value(*id) {
                                match op {
                                    CompareOp::Eq => (filter_val - num).abs() < f64::EPSILON,
                                    CompareOp::Ne => (filter_val - num).abs() >= f64::EPSILON,
                                    CompareOp::Lt => filter_val < num,
                                    CompareOp::Le => filter_val <= num,
                                    CompareOp::Gt => filter_val > num,
                                    CompareOp::Ge => filter_val >= num,
                                }
                            } else {
                                false
                            }
                        })
                        .collect();
                    return Ok(filtered);
                }
            }
            WhereClause::Between {
                column: filter_col,
                min,
                max,
            } => {
                let filter = self.columnar.get_column(filter_col);
                let min_num = min.as_f64().unwrap_or(f64::MIN);
                let max_num = max.as_f64().unwrap_or(f64::MAX);

                if let Some(filter_col_data) = filter {
                    let filtered: Vec<(RecordId, f64)> = col
                        .scan()
                        .filter(|(id, _)| {
                            if let Some(filter_val) = filter_col_data.get_value(*id) {
                                filter_val >= min_num && filter_val <= max_num
                            } else {
                                false
                            }
                        })
                        .collect();
                    return Ok(filtered);
                }
            }
            _ => {}
        }

        // Fallback: return all
        Ok(col.scan().collect())
    }

    /// Execute a query with window functions
    fn execute_window(&self, select: &SelectStatement) -> AmorphicResult<Vec<Vec<SqlValue>>> {
        use std::collections::HashMap;

        // Get the order by column (required for most window functions)
        let (order_col, order_asc) = select
            .order_by
            .first()
            .map(|item| (item.column.as_str(), item.order == SortOrder::Ascending))
            .unwrap_or_else(|| {
                // Try to find an order column from the first window function
                for col in &select.columns {
                    if let SelectColumn::WindowFunction { order_by, .. } = col {
                        if let Some((col, order)) = order_by.first() {
                            return (col.as_str(), *order == SortOrder::Ascending);
                        }
                    }
                }
                // Default to first column if no order specified
                ("id", true)
            });

        // Get partition column if any
        let partition_col: Option<&str> = select.columns.iter().find_map(|c| {
            if let SelectColumn::WindowFunction { partition_by, .. } = c {
                partition_by.first().map(|s| s.as_str())
            } else {
                None
            }
        });

        // Get base column for scanning (to get all record IDs)
        let base_col_name = if let Some(col) = self.columnar.get_column(order_col) {
            order_col
        } else {
            // Find any column to use as base
            self.columnar
                .column_names()
                .next()
                .map(|s| s.as_str())
                .unwrap_or("id")
        };

        let base_col = match self.columnar.get_column(base_col_name) {
            Some(col) => col,
            None => return Ok(Vec::new()),
        };

        // Get all record IDs in order
        let mut record_values: Vec<(RecordId, f64)> = base_col.scan().collect();

        // Sort by order column
        if order_asc {
            record_values
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            record_values
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Compute window function values for each column
        let mut window_results: Vec<HashMap<RecordId, f64>> = Vec::new();
        let mut regular_columns: Vec<String> = Vec::new();

        for col in &select.columns {
            match col {
                SelectColumn::WindowFunction {
                    func,
                    partition_by,
                    order_by,
                    ..
                } => {
                    // Get order column and direction from window function
                    let (wf_order_col, wf_order_asc) = order_by
                        .first()
                        .map(|(col, order)| (col.as_str(), *order == SortOrder::Ascending))
                        .unwrap_or((order_col, order_asc));

                    let wf_partition_col = partition_by.first().map(|s| s.as_str());

                    let result = match func {
                        WindowFunc::RowNumber => self.columnar.compute_row_number(
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::Rank => {
                            self.columnar
                                .compute_rank(wf_order_col, wf_order_asc, wf_partition_col)
                        }
                        WindowFunc::DenseRank => self.columnar.compute_dense_rank(
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::Ntile(n) => self.columnar.compute_ntile(
                            *n,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::Lead(value_col, offset) => self.columnar.compute_lead(
                            value_col,
                            *offset,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::Lag(value_col, offset) => self.columnar.compute_lag(
                            value_col,
                            *offset,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::FirstValue(value_col) => self.columnar.compute_first_value(
                            value_col,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::LastValue(value_col) => self.columnar.compute_last_value(
                            value_col,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::RunningSum(value_col) => self.columnar.compute_running_sum(
                            value_col,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::RunningAvg(value_col) => self.columnar.compute_running_avg(
                            value_col,
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                        WindowFunc::RunningCount => self.columnar.compute_running_count(
                            wf_order_col,
                            wf_order_asc,
                            wf_partition_col,
                        ),
                    };
                    window_results.push(result);
                    regular_columns.push(String::new()); // Placeholder
                }
                SelectColumn::Column { name, .. } => {
                    window_results.push(HashMap::new());
                    regular_columns.push(name.clone());
                }
                SelectColumn::Star | SelectColumn::Aggregate { .. } => {
                    window_results.push(HashMap::new());
                    regular_columns.push(String::new());
                }
            }
        }

        // Build rows
        let mut rows: Vec<Vec<SqlValue>> = Vec::new();

        for (record_id, _) in &record_values {
            let mut row = Vec::new();

            for (i, col) in select.columns.iter().enumerate() {
                match col {
                    SelectColumn::WindowFunction { .. } => {
                        let value = window_results[i]
                            .get(record_id)
                            .copied()
                            .unwrap_or(f64::NAN);
                        row.push(SqlValue::Number(value));
                    }
                    SelectColumn::Column { name, .. } => {
                        let value = self
                            .columnar
                            .get_column(name)
                            .and_then(|c| c.get_value(*record_id))
                            .unwrap_or(f64::NAN);
                        row.push(SqlValue::Number(value));
                    }
                    SelectColumn::Star => {
                        // For now, just include the order column value
                        let value = base_col.get_value(*record_id).unwrap_or(f64::NAN);
                        row.push(SqlValue::Number(value));
                    }
                    SelectColumn::Aggregate { .. } => {
                        // Aggregates not supported in window queries
                        row.push(SqlValue::Null);
                    }
                }
            }

            rows.push(row);
        }

        // Apply LIMIT
        if let Some(limit) = select.limit {
            rows.truncate(limit);
        }

        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Value;

    #[test]
    fn test_lexer_basic() {
        let mut lexer = SqlLexer::new("SELECT * FROM users WHERE age > 25");
        let tokens = lexer.tokenize().unwrap();

        assert!(matches!(tokens[0], Token::Select));
        assert!(matches!(tokens[1], Token::Star));
        assert!(matches!(tokens[2], Token::From));
        assert!(matches!(tokens[3], Token::Identifier(_)));
        assert!(matches!(tokens[4], Token::Where));
    }

    #[test]
    fn test_lexer_numbers_and_strings() {
        let mut lexer =
            SqlLexer::new("SELECT name, age FROM users WHERE age = 25 AND city = 'NYC'");
        let tokens = lexer.tokenize().unwrap();

        assert!(
            tokens
                .iter()
                .any(|t| matches!(t, Token::Number(n) if *n == 25.0))
        );
        assert!(
            tokens
                .iter()
                .any(|t| matches!(t, Token::String(s) if s == "NYC"))
        );
    }

    #[test]
    fn test_parser_simple_select() {
        let stmt = SqlParser::parse("SELECT name, age FROM users").unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 2);
            assert!(select.from.is_some());
            assert_eq!(select.from.unwrap().table, "users");
        } else {
            panic!("Expected SELECT statement");
        }
    }

    #[test]
    fn test_parser_select_with_where() {
        let stmt = SqlParser::parse("SELECT * FROM orders WHERE amount > 100").unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert!(select.where_clause.is_some());
            if let Some(WhereClause::Comparison { column, op, value }) = select.where_clause {
                assert_eq!(column, "amount");
                assert_eq!(op, CompareOp::Gt);
                assert!(matches!(value, SqlValue::Number(n) if n == 100.0));
            }
        }
    }

    #[test]
    fn test_parser_select_with_aggregates() {
        let stmt = SqlParser::parse("SELECT SUM(amount), COUNT(*) FROM orders").unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 2);
            assert!(matches!(
                &select.columns[0],
                SelectColumn::Aggregate {
                    func: AggregateFunc::Sum,
                    ..
                }
            ));
            assert!(matches!(
                &select.columns[1],
                SelectColumn::Aggregate {
                    func: AggregateFunc::Count,
                    ..
                }
            ));
        }
    }

    #[test]
    fn test_parser_select_with_group_by() {
        let stmt =
            SqlParser::parse("SELECT customer, SUM(amount) FROM orders GROUP BY customer").unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert_eq!(select.group_by.len(), 1);
            assert_eq!(select.group_by[0], "customer");
        }
    }

    #[test]
    fn test_parser_select_with_order_by_limit() {
        let stmt = SqlParser::parse("SELECT * FROM users ORDER BY age DESC LIMIT 10").unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert_eq!(select.order_by.len(), 1);
            assert_eq!(select.order_by[0].order, SortOrder::Descending);
            assert_eq!(select.limit, Some(10));
        }
    }

    #[test]
    fn test_parser_select_with_join() {
        let stmt = SqlParser::parse(
            "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id",
        )
        .unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert_eq!(select.joins.len(), 1);
            assert_eq!(select.joins[0].join_type, JoinType::Inner);
            assert_eq!(select.joins[0].table, "customers");
        }
    }

    #[test]
    fn test_to_logical_plan() {
        let stmt = SqlParser::parse(
            "SELECT customer, SUM(amount) as total FROM orders WHERE date >= 20240101 GROUP BY customer LIMIT 10"
        ).unwrap();

        if let SqlStatement::Select(select) = stmt {
            let plan = select.to_logical_plan().unwrap();
            // Should have Limit at the top
            assert!(matches!(plan, LogicalPlan::Limit { .. }));
        }
    }

    #[test]
    fn test_executor_scalar_aggregation() {
        let mut store = ColumnarStore::new();

        // Add test data
        for i in 0..100 {
            store.record_value("amount", i as u64, &Value::Float(i as f64 * 10.0));
        }

        let executor = SqlExecutor::new(&store);
        let result = executor
            .execute_sql("SELECT SUM(amount), COUNT(amount), AVG(amount) FROM orders")
            .unwrap();

        assert_eq!(result.row_count, 1);
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_executor_group_by() {
        let mut store = ColumnarStore::new();

        // Add test data with groups
        for i in 0..100 {
            let group = (i % 5) as f64; // 5 groups
            store.record_value("category", i as u64, &Value::Float(group));
            store.record_value("amount", i as u64, &Value::Float(i as f64));
        }

        let executor = SqlExecutor::new(&store);
        let result = executor
            .execute_sql("SELECT category, SUM(amount) FROM orders GROUP BY category")
            .unwrap();

        assert!(result.row_count <= 5); // At most 5 groups
    }

    #[test]
    fn test_result_to_table() {
        let result = SqlResult {
            columns: vec!["name".to_string(), "age".to_string()],
            rows: vec![
                vec![
                    SqlValue::String("Alice".to_string()),
                    SqlValue::Number(30.0),
                ],
                vec![SqlValue::String("Bob".to_string()), SqlValue::Number(25.0)],
            ],
            row_count: 2,
            execution_time_ms: 1.5,
        };

        let table = result.to_table();
        assert!(table.contains("name"));
        assert!(table.contains("Alice"));
        assert!(table.contains("2 rows"));
    }

    // ==========================================================================
    // Window Function Tests
    // ==========================================================================

    #[test]
    fn test_lexer_window_functions() {
        let sql =
            "SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) FROM employees";
        let mut lexer = SqlLexer::new(sql);
        let tokens = lexer.tokenize().unwrap();

        assert!(tokens.iter().any(|t| matches!(t, Token::RowNumber)));
        assert!(tokens.iter().any(|t| matches!(t, Token::Over)));
        assert!(tokens.iter().any(|t| matches!(t, Token::PartitionBy)));
        assert!(tokens.iter().any(|t| matches!(t, Token::OrderBy)));
    }

    #[test]
    fn test_parser_window_row_number() {
        let stmt = SqlParser::parse(
            "SELECT ROW_NUMBER() OVER (ORDER BY salary DESC) as rn FROM employees",
        )
        .unwrap();

        if let SqlStatement::Select(select) = stmt {
            assert_eq!(select.columns.len(), 1);
            if let SelectColumn::WindowFunction {
                func,
                order_by,
                alias,
                ..
            } = &select.columns[0]
            {
                assert!(matches!(func, WindowFunc::RowNumber));
                assert_eq!(order_by.len(), 1);
                assert_eq!(order_by[0].0, "salary");
                assert_eq!(order_by[0].1, SortOrder::Descending);
                assert_eq!(alias.as_deref(), Some("rn"));
            } else {
                panic!("Expected WindowFunction");
            }
        }
    }

    #[test]
    fn test_parser_window_rank_with_partition() {
        let stmt = SqlParser::parse(
            "SELECT RANK() OVER (PARTITION BY dept ORDER BY salary) FROM employees",
        )
        .unwrap();

        if let SqlStatement::Select(select) = stmt {
            if let SelectColumn::WindowFunction {
                func,
                partition_by,
                order_by,
                ..
            } = &select.columns[0]
            {
                assert!(matches!(func, WindowFunc::Rank));
                assert_eq!(partition_by.len(), 1);
                assert_eq!(partition_by[0], "dept");
                assert_eq!(order_by.len(), 1);
            } else {
                panic!("Expected WindowFunction");
            }
        }
    }

    #[test]
    fn test_parser_window_ntile() {
        let stmt = SqlParser::parse("SELECT NTILE(4) OVER (ORDER BY score) FROM students").unwrap();

        if let SqlStatement::Select(select) = stmt {
            if let SelectColumn::WindowFunction { func, .. } = &select.columns[0] {
                assert!(matches!(func, WindowFunc::Ntile(4)));
            } else {
                panic!("Expected WindowFunction");
            }
        }
    }

    #[test]
    fn test_parser_window_lead_lag() {
        let stmt =
            SqlParser::parse("SELECT LEAD(price, 1) OVER (ORDER BY date) FROM stock").unwrap();

        if let SqlStatement::Select(select) = stmt {
            if let SelectColumn::WindowFunction { func, .. } = &select.columns[0] {
                assert!(matches!(func, WindowFunc::Lead(_, 1)));
            } else {
                panic!("Expected WindowFunction");
            }
        }

        let stmt2 =
            SqlParser::parse("SELECT LAG(price, 2) OVER (ORDER BY date) FROM stock").unwrap();

        if let SqlStatement::Select(select) = stmt2 {
            if let SelectColumn::WindowFunction { func, .. } = &select.columns[0] {
                assert!(matches!(func, WindowFunc::Lag(_, 2)));
            } else {
                panic!("Expected WindowFunction");
            }
        }
    }

    #[test]
    fn test_parser_window_first_last_value() {
        let stmt =
            SqlParser::parse("SELECT FIRST_VALUE(name) OVER (ORDER BY id) FROM users").unwrap();

        if let SqlStatement::Select(select) = stmt {
            if let SelectColumn::WindowFunction { func, .. } = &select.columns[0] {
                if let WindowFunc::FirstValue(col) = func {
                    assert_eq!(col, "name");
                } else {
                    panic!("Expected FirstValue");
                }
            } else {
                panic!("Expected WindowFunction");
            }
        }
    }

    #[test]
    fn test_parser_running_sum() {
        let stmt = SqlParser::parse(
            "SELECT SUM(amount) OVER (ORDER BY date) as running_total FROM orders",
        )
        .unwrap();

        if let SqlStatement::Select(select) = stmt {
            if let SelectColumn::WindowFunction { func, alias, .. } = &select.columns[0] {
                assert!(matches!(func, WindowFunc::RunningSum(_)));
                assert_eq!(alias.as_deref(), Some("running_total"));
            } else {
                panic!("Expected WindowFunction");
            }
        }
    }

    #[test]
    fn test_executor_window_row_number() {
        let mut store = ColumnarStore::new();

        // Add test data
        for i in 0..10 {
            store.record_value("salary", i as u64, &Value::Float((10 - i) as f64 * 1000.0));
            store.record_value("dept", i as u64, &Value::Float((i % 2) as f64));
        }

        // Test ROW_NUMBER
        let result = store.compute_row_number("salary", false, None);
        assert_eq!(result.len(), 10);

        // Values should be 1 through 10
        let mut values: Vec<f64> = result.values().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            values,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        );
    }

    #[test]
    fn test_executor_window_rank() {
        let mut store = ColumnarStore::new();

        // Add test data with ties
        store.record_value("score", 0, &Value::Float(100.0));
        store.record_value("score", 1, &Value::Float(100.0)); // tie
        store.record_value("score", 2, &Value::Float(90.0));
        store.record_value("score", 3, &Value::Float(80.0));

        let result = store.compute_rank("score", false, None);

        // Descending order: 100, 100, 90, 80
        // Ranks should be: 1, 1, 3, 4 (gap after ties)
        assert_eq!(result.get(&0), Some(&1.0));
        assert_eq!(result.get(&1), Some(&1.0));
        assert_eq!(result.get(&2), Some(&3.0));
        assert_eq!(result.get(&3), Some(&4.0));
    }

    #[test]
    fn test_executor_window_dense_rank() {
        let mut store = ColumnarStore::new();

        // Add test data with ties
        store.record_value("score", 0, &Value::Float(100.0));
        store.record_value("score", 1, &Value::Float(100.0)); // tie
        store.record_value("score", 2, &Value::Float(90.0));
        store.record_value("score", 3, &Value::Float(80.0));

        let result = store.compute_dense_rank("score", false, None);

        // Descending order: 100, 100, 90, 80
        // Dense ranks should be: 1, 1, 2, 3 (no gap after ties)
        assert_eq!(result.get(&0), Some(&1.0));
        assert_eq!(result.get(&1), Some(&1.0));
        assert_eq!(result.get(&2), Some(&2.0));
        assert_eq!(result.get(&3), Some(&3.0));
    }

    #[test]
    fn test_executor_window_ntile() {
        let mut store = ColumnarStore::new();

        // Add 10 records
        for i in 0..10 {
            store.record_value("value", i as u64, &Value::Float(i as f64));
        }

        // Divide into 4 buckets: 3, 3, 2, 2
        let result = store.compute_ntile(4, "value", true, None);
        assert_eq!(result.len(), 10);

        // Count how many in each bucket
        let mut bucket_counts = [0; 4];
        for v in result.values() {
            bucket_counts[(*v as usize) - 1] += 1;
        }
        // Should be roughly equal distribution
        assert!(bucket_counts.iter().all(|&c| c >= 2 && c <= 3));
    }

    #[test]
    fn test_executor_window_lead_lag() {
        let mut store = ColumnarStore::new();

        // Add sequential values
        for i in 0..5 {
            store.record_value("value", i as u64, &Value::Float(i as f64 * 10.0));
        }

        // LEAD(value, 1)
        let lead_result = store.compute_lead("value", 1, "value", true, None);
        // Record 0 (value=0) should have lead = 10 (next record's value)
        assert_eq!(lead_result.get(&0), Some(&10.0));
        // Record 4 (last) should have lead = NaN
        assert!(lead_result.get(&4).unwrap().is_nan());

        // LAG(value, 1)
        let lag_result = store.compute_lag("value", 1, "value", true, None);
        // Record 0 (first) should have lag = NaN
        assert!(lag_result.get(&0).unwrap().is_nan());
        // Record 1 should have lag = 0 (previous record's value)
        assert_eq!(lag_result.get(&1), Some(&0.0));
    }

    #[test]
    fn test_executor_running_sum() {
        let mut store = ColumnarStore::new();

        // Add values: 1, 2, 3, 4, 5
        for i in 0..5 {
            store.record_value("amount", i as u64, &Value::Float((i + 1) as f64));
        }

        let result = store.compute_running_sum("amount", "amount", true, None);

        // Running sum should be: 1, 3, 6, 10, 15
        assert_eq!(result.get(&0), Some(&1.0));
        assert_eq!(result.get(&1), Some(&3.0));
        assert_eq!(result.get(&2), Some(&6.0));
        assert_eq!(result.get(&3), Some(&10.0));
        assert_eq!(result.get(&4), Some(&15.0));
    }

    #[test]
    fn test_executor_running_avg() {
        let mut store = ColumnarStore::new();

        // Add values: 10, 20, 30
        for i in 0..3 {
            store.record_value("value", i as u64, &Value::Float((i + 1) as f64 * 10.0));
        }

        let result = store.compute_running_avg("value", "value", true, None);

        // Running avg should be: 10, 15, 20
        assert!((result.get(&0).unwrap() - 10.0).abs() < 0.001);
        assert!((result.get(&1).unwrap() - 15.0).abs() < 0.001);
        assert!((result.get(&2).unwrap() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_window_with_partition() {
        let mut store = ColumnarStore::new();

        // Add data with two departments
        // Dept 0: salaries 100, 200, 300
        // Dept 1: salaries 150, 250
        store.record_value("dept", 0, &Value::Float(0.0));
        store.record_value("salary", 0, &Value::Float(100.0));

        store.record_value("dept", 1, &Value::Float(0.0));
        store.record_value("salary", 1, &Value::Float(200.0));

        store.record_value("dept", 2, &Value::Float(0.0));
        store.record_value("salary", 2, &Value::Float(300.0));

        store.record_value("dept", 3, &Value::Float(1.0));
        store.record_value("salary", 3, &Value::Float(150.0));

        store.record_value("dept", 4, &Value::Float(1.0));
        store.record_value("salary", 4, &Value::Float(250.0));

        // ROW_NUMBER with partition by dept
        let result = store.compute_row_number("salary", true, Some("dept"));

        // Within dept 0: 100->1, 200->2, 300->3
        // Within dept 1: 150->1, 250->2
        assert_eq!(result.get(&0), Some(&1.0)); // dept 0, salary 100
        assert_eq!(result.get(&1), Some(&2.0)); // dept 0, salary 200
        assert_eq!(result.get(&2), Some(&3.0)); // dept 0, salary 300
        assert_eq!(result.get(&3), Some(&1.0)); // dept 1, salary 150
        assert_eq!(result.get(&4), Some(&2.0)); // dept 1, salary 250
    }
}
