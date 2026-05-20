//! Time Series Query Languages
//!
//! Supports InfluxQL and PromQL for time series data.

use crate::ast::{Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// Common Types
// ============================================================================

/// Time series query
#[derive(Debug, Clone)]
pub struct TimeSeriesQuery {
    pub query_type: TimeSeriesQueryType,
}

/// Query type
#[derive(Debug, Clone)]
pub enum TimeSeriesQueryType {
    InfluxQL(InfluxQuery),
    PromQL(PromQuery),
}

impl TimeSeriesQuery {
    /// Convert to generic Query
    pub fn to_query(&self) -> Query {
        Query {
            query_type: QueryType::Select,
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

/// Time range
#[derive(Debug, Clone)]
pub struct TimeRange {
    pub start: Option<TimeSpec>,
    pub end: Option<TimeSpec>,
}

/// Time specification
#[derive(Debug, Clone)]
pub enum TimeSpec {
    Now,
    Relative(Duration, bool), // duration, is_negative
    Absolute(i64),            // Unix timestamp
    Rfc3339(String),
}

/// Aggregation function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFn {
    Mean,
    Sum,
    Count,
    Min,
    Max,
    First,
    Last,
    Median,
    Mode,
    Stddev,
    Spread,
    Percentile(u8),
    Rate,
    Irate,
    Increase,
    Delta,
    Derivative,
    NonNegativeDerivative,
    Integral,
    MovingAverage,
    CumulativeSum,
    Difference,
    Elapsed,
    Histogram,
}

// ============================================================================
// InfluxQL
// ============================================================================

/// InfluxQL query
#[derive(Debug, Clone)]
pub struct InfluxQuery {
    pub statement: InfluxStatement,
}

/// InfluxQL statement
#[derive(Debug, Clone)]
pub enum InfluxStatement {
    Select(InfluxSelect),
    ShowDatabases,
    ShowMeasurements(Option<String>),
    ShowTagKeys(String),
    ShowTagValues(String, String),
    ShowFieldKeys(String),
    ShowRetentionPolicies(String),
    CreateDatabase(String),
    DropDatabase(String),
    CreateRetentionPolicy(InfluxRetentionPolicy),
    DropRetentionPolicy(String, String),
    Insert(InfluxInsert),
    Delete(InfluxDelete),
}

/// InfluxQL SELECT statement
#[derive(Debug, Clone)]
pub struct InfluxSelect {
    pub columns: Vec<InfluxColumn>,
    pub from: Vec<InfluxMeasurement>,
    pub where_clause: Option<InfluxWhere>,
    pub group_by: Option<InfluxGroupBy>,
    pub order_by: Option<InfluxOrderBy>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub slimit: Option<usize>,
    pub soffset: Option<usize>,
    pub fill: Option<InfluxFill>,
    pub tz: Option<String>,
}

/// InfluxQL column/field selection
#[derive(Debug, Clone)]
pub struct InfluxColumn {
    pub expr: InfluxExpr,
    pub alias: Option<String>,
}

/// InfluxQL expression
#[derive(Debug, Clone)]
pub enum InfluxExpr {
    Wildcard,
    Field(String),
    Tag(String),
    Literal(Value),
    Function(String, Vec<InfluxExpr>),
    Binary(Box<InfluxExpr>, String, Box<InfluxExpr>),
    Subquery(Box<InfluxSelect>),
}

/// InfluxQL measurement reference
#[derive(Debug, Clone)]
pub struct InfluxMeasurement {
    pub database: Option<String>,
    pub retention_policy: Option<String>,
    pub name: String,
    pub regex: bool,
}

/// InfluxQL WHERE clause
#[derive(Debug, Clone)]
pub struct InfluxWhere {
    pub condition: InfluxCondition,
}

/// InfluxQL condition
#[derive(Debug, Clone)]
pub enum InfluxCondition {
    Comparison(String, String, Value), // field, op, value
    TimeRange(TimeRange),
    And(Box<InfluxCondition>, Box<InfluxCondition>),
    Or(Box<InfluxCondition>, Box<InfluxCondition>),
    Not(Box<InfluxCondition>),
    Regex(String, String), // field, pattern
}

/// InfluxQL GROUP BY clause
#[derive(Debug, Clone)]
pub struct InfluxGroupBy {
    pub tags: Vec<String>,
    pub time_interval: Option<Duration>,
    pub fill: Option<InfluxFill>,
}

/// InfluxQL fill option
#[derive(Debug, Clone)]
pub enum InfluxFill {
    None,
    Null,
    Previous,
    Linear,
    Value(f64),
}

/// InfluxQL ORDER BY clause
#[derive(Debug, Clone)]
pub struct InfluxOrderBy {
    pub time_desc: bool,
}

/// InfluxQL retention policy
#[derive(Debug, Clone)]
pub struct InfluxRetentionPolicy {
    pub name: String,
    pub database: String,
    pub duration: Duration,
    pub replication: usize,
    pub shard_duration: Option<Duration>,
    pub is_default: bool,
}

/// InfluxQL INSERT statement
#[derive(Debug, Clone)]
pub struct InfluxInsert {
    pub measurement: String,
    pub tags: HashMap<String, String>,
    pub fields: HashMap<String, Value>,
    pub timestamp: Option<i64>,
}

/// InfluxQL DELETE statement
#[derive(Debug, Clone)]
pub struct InfluxDelete {
    pub measurement: String,
    pub where_clause: Option<InfluxWhere>,
}

/// InfluxQL Parser
pub struct InfluxParser {
    input: String,
    pos: usize,
}

impl InfluxParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            input: String::new(),
            pos: 0,
        }
    }

    /// Parse InfluxQL query
    pub fn parse(&mut self, influx: &str) -> QueryResult<InfluxQuery> {
        self.input = influx.to_string();
        self.pos = 0;
        self.skip_whitespace();

        let statement = self.parse_statement()?;
        Ok(InfluxQuery { statement })
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_keyword(&self) -> Option<String> {
        let mut end = self.pos;
        while end < self.input.len() {
            let c = self.input[end..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_alphanumeric() || c == '_' {
                end += c.len_utf8();
            } else {
                break;
            }
        }
        if end > self.pos {
            Some(self.input[self.pos..end].to_uppercase())
        } else {
            None
        }
    }

    fn try_consume_keyword(&mut self, keyword: &str) -> bool {
        self.skip_whitespace();
        if let Some(kw) = self.peek_keyword() {
            if kw == keyword.to_uppercase() {
                self.pos += keyword.len();
                self.skip_whitespace();
                return true;
            }
        }
        false
    }

    fn consume_keyword(&mut self, keyword: &str) -> QueryResult<()> {
        if self.try_consume_keyword(keyword) {
            Ok(())
        } else {
            Err(QueryError::ParseError(format!("Expected {}", keyword)))
        }
    }

    fn parse_identifier(&mut self) -> QueryResult<String> {
        self.skip_whitespace();

        // Handle quoted identifier
        if self.input[self.pos..].starts_with('"') {
            self.pos += 1;
            let start = self.pos;
            while self.pos < self.input.len() && !self.input[self.pos..].starts_with('"') {
                self.pos += 1;
            }
            let result = self.input[start..self.pos].to_string();
            self.pos += 1;
            self.skip_whitespace();
            return Ok(result);
        }

        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_alphanumeric() || c == '_' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }

        if self.pos > start {
            let result = self.input[start..self.pos].to_string();
            self.skip_whitespace();
            Ok(result)
        } else {
            Err(QueryError::ParseError("Expected identifier".to_string()))
        }
    }

    fn try_consume_char(&mut self, c: char) -> bool {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(c) {
            self.pos += c.len_utf8();
            self.skip_whitespace();
            true
        } else {
            false
        }
    }

    fn parse_statement(&mut self) -> QueryResult<InfluxStatement> {
        match self.peek_keyword().as_deref() {
            Some("SELECT") => self.parse_select(),
            Some("SHOW") => self.parse_show(),
            Some("CREATE") => self.parse_create(),
            Some("DROP") => self.parse_drop(),
            Some("INSERT") => self.parse_insert(),
            Some("DELETE") => self.parse_delete(),
            _ => Err(QueryError::ParseError("Unknown statement".to_string())),
        }
    }

    fn parse_select(&mut self) -> QueryResult<InfluxStatement> {
        self.consume_keyword("SELECT")?;

        let columns = self.parse_columns()?;

        self.consume_keyword("FROM")?;
        let from = self.parse_measurements()?;

        let where_clause = if self.try_consume_keyword("WHERE") {
            Some(self.parse_where()?)
        } else {
            None
        };

        let group_by = if self.try_consume_keyword("GROUP") {
            self.consume_keyword("BY")?;
            Some(self.parse_group_by()?)
        } else {
            None
        };

        let order_by = if self.try_consume_keyword("ORDER") {
            self.consume_keyword("BY")?;
            Some(self.parse_order_by()?)
        } else {
            None
        };

        let limit = if self.try_consume_keyword("LIMIT") {
            Some(self.parse_integer()? as usize)
        } else {
            None
        };

        let offset = if self.try_consume_keyword("OFFSET") {
            Some(self.parse_integer()? as usize)
        } else {
            None
        };

        let fill = if self.try_consume_keyword("FILL") {
            self.try_consume_char('(');
            let f = self.parse_fill()?;
            self.try_consume_char(')');
            Some(f)
        } else {
            None
        };

        Ok(InfluxStatement::Select(InfluxSelect {
            columns,
            from,
            where_clause,
            group_by,
            order_by,
            limit,
            offset,
            slimit: None,
            soffset: None,
            fill,
            tz: None,
        }))
    }

    fn parse_columns(&mut self) -> QueryResult<Vec<InfluxColumn>> {
        let mut columns = Vec::new();

        loop {
            let expr = self.parse_expr()?;
            let alias = if self.try_consume_keyword("AS") {
                Some(self.parse_identifier()?)
            } else {
                None
            };

            columns.push(InfluxColumn { expr, alias });

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(columns)
    }

    fn parse_expr(&mut self) -> QueryResult<InfluxExpr> {
        self.skip_whitespace();

        if self.try_consume_char('*') {
            return Ok(InfluxExpr::Wildcard);
        }

        // Try to parse function
        let start_pos = self.pos;
        if let Ok(name) = self.parse_identifier() {
            if self.try_consume_char('(') {
                let mut args = Vec::new();
                if !self.input[self.pos..].trim_start().starts_with(')') {
                    loop {
                        args.push(self.parse_expr()?);
                        if !self.try_consume_char(',') {
                            break;
                        }
                    }
                }
                self.try_consume_char(')');
                return Ok(InfluxExpr::Function(name, args));
            }
            return Ok(InfluxExpr::Field(name));
        }

        self.pos = start_pos;
        Err(QueryError::ParseError("Expected expression".to_string()))
    }

    fn parse_measurements(&mut self) -> QueryResult<Vec<InfluxMeasurement>> {
        let mut measurements = Vec::new();

        loop {
            let name = self.parse_identifier()?;
            measurements.push(InfluxMeasurement {
                database: None,
                retention_policy: None,
                name,
                regex: false,
            });

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(measurements)
    }

    fn parse_where(&mut self) -> QueryResult<InfluxWhere> {
        let condition = self.parse_condition()?;
        Ok(InfluxWhere { condition })
    }

    fn parse_condition(&mut self) -> QueryResult<InfluxCondition> {
        let left = self.parse_comparison()?;

        if self.try_consume_keyword("AND") {
            let right = self.parse_condition()?;
            Ok(InfluxCondition::And(Box::new(left), Box::new(right)))
        } else if self.try_consume_keyword("OR") {
            let right = self.parse_condition()?;
            Ok(InfluxCondition::Or(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_comparison(&mut self) -> QueryResult<InfluxCondition> {
        let field = self.parse_identifier()?;

        self.skip_whitespace();
        let op = if self.input[self.pos..].starts_with(">=") {
            self.pos += 2;
            ">="
        } else if self.input[self.pos..].starts_with("<=") {
            self.pos += 2;
            "<="
        } else if self.input[self.pos..].starts_with("<>") {
            self.pos += 2;
            "<>"
        } else if self.input[self.pos..].starts_with("!=") {
            self.pos += 2;
            "!="
        } else if self.input[self.pos..].starts_with('=') {
            self.pos += 1;
            "="
        } else if self.input[self.pos..].starts_with('<') {
            self.pos += 1;
            "<"
        } else if self.input[self.pos..].starts_with('>') {
            self.pos += 1;
            ">"
        } else {
            return Err(QueryError::ParseError("Expected operator".to_string()));
        };

        self.skip_whitespace();
        let value = self.parse_value()?;

        Ok(InfluxCondition::Comparison(field, op.to_string(), value))
    }

    fn parse_value(&mut self) -> QueryResult<Value> {
        self.skip_whitespace();

        // String
        if self.input[self.pos..].starts_with('\'') {
            self.pos += 1;
            let start = self.pos;
            while self.pos < self.input.len() && !self.input[self.pos..].starts_with('\'') {
                self.pos += 1;
            }
            let s = self.input[start..self.pos].to_string();
            self.pos += 1;
            return Ok(Value::String(s));
        }

        // Number or duration
        let start = self.pos;
        let mut has_dot = false;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == '.' && !has_dot {
                has_dot = true;
                self.pos += 1;
            } else if c == '-' && self.pos == start {
                self.pos += 1;
            } else {
                break;
            }
        }

        if self.pos > start {
            let num_str = self.input[start..self.pos].to_string();
            self.skip_whitespace();
            if has_dot {
                return Ok(Value::Float(num_str.parse().unwrap_or(0.0)));
            } else {
                return Ok(Value::Int(num_str.parse().unwrap_or(0)));
            }
        }

        // Boolean or identifier
        let ident = self.parse_identifier()?;
        match ident.to_uppercase().as_str() {
            "TRUE" => Ok(Value::Bool(true)),
            "FALSE" => Ok(Value::Bool(false)),
            "NOW" => Ok(Value::Timestamp(0)), // Special case
            _ => Ok(Value::String(ident)),
        }
    }

    fn parse_group_by(&mut self) -> QueryResult<InfluxGroupBy> {
        let mut tags = Vec::new();
        let mut time_interval = None;

        loop {
            self.skip_whitespace();

            // Check for time()
            if self.try_consume_keyword("TIME") {
                self.try_consume_char('(');
                time_interval = Some(self.parse_duration()?);
                self.try_consume_char(')');
            } else {
                tags.push(self.parse_identifier()?);
            }

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(InfluxGroupBy {
            tags,
            time_interval,
            fill: None,
        })
    }

    fn parse_duration(&mut self) -> QueryResult<Duration> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let num: u64 = self.input[start..self.pos].parse().unwrap_or(0);

        let unit = self.parse_identifier()?;
        let multiplier = match unit.as_str() {
            "ns" | "n" => 1,
            "us" | "u" | "µ" => 1_000,
            "ms" => 1_000_000,
            "s" => 1_000_000_000,
            "m" => 60_000_000_000,
            "h" => 3_600_000_000_000u64,
            "d" => 86_400_000_000_000u64,
            "w" => 604_800_000_000_000u64,
            _ => {
                return Err(QueryError::ParseError(format!(
                    "Unknown duration unit: {}",
                    unit
                )));
            }
        };

        Ok(Duration::from_nanos(num * multiplier))
    }

    fn parse_order_by(&mut self) -> QueryResult<InfluxOrderBy> {
        self.consume_keyword("TIME")?;
        let time_desc = self.try_consume_keyword("DESC");
        if !time_desc {
            self.try_consume_keyword("ASC");
        }
        Ok(InfluxOrderBy { time_desc })
    }

    fn parse_fill(&mut self) -> QueryResult<InfluxFill> {
        let ident = self.parse_identifier()?;
        match ident.to_uppercase().as_str() {
            "NONE" => Ok(InfluxFill::None),
            "NULL" => Ok(InfluxFill::Null),
            "PREVIOUS" => Ok(InfluxFill::Previous),
            "LINEAR" => Ok(InfluxFill::Linear),
            _ => {
                // Try to parse as number
                if let Ok(n) = ident.parse::<f64>() {
                    Ok(InfluxFill::Value(n))
                } else {
                    Err(QueryError::ParseError(format!("Unknown fill: {}", ident)))
                }
            }
        }
    }

    fn parse_show(&mut self) -> QueryResult<InfluxStatement> {
        self.consume_keyword("SHOW")?;

        if self.try_consume_keyword("DATABASES") {
            Ok(InfluxStatement::ShowDatabases)
        } else if self.try_consume_keyword("MEASUREMENTS") {
            let db = if self.try_consume_keyword("ON") {
                Some(self.parse_identifier()?)
            } else {
                None
            };
            Ok(InfluxStatement::ShowMeasurements(db))
        } else if self.try_consume_keyword("TAG") {
            if self.try_consume_keyword("KEYS") {
                self.consume_keyword("FROM")?;
                let measurement = self.parse_identifier()?;
                Ok(InfluxStatement::ShowTagKeys(measurement))
            } else if self.try_consume_keyword("VALUES") {
                self.consume_keyword("FROM")?;
                let measurement = self.parse_identifier()?;
                self.consume_keyword("WITH")?;
                self.consume_keyword("KEY")?;
                self.try_consume_char('=');
                let key = self.parse_identifier()?;
                Ok(InfluxStatement::ShowTagValues(measurement, key))
            } else {
                Err(QueryError::ParseError(
                    "Expected KEYS or VALUES".to_string(),
                ))
            }
        } else if self.try_consume_keyword("FIELD") {
            self.consume_keyword("KEYS")?;
            self.consume_keyword("FROM")?;
            let measurement = self.parse_identifier()?;
            Ok(InfluxStatement::ShowFieldKeys(measurement))
        } else if self.try_consume_keyword("RETENTION") {
            self.consume_keyword("POLICIES")?;
            self.consume_keyword("ON")?;
            let db = self.parse_identifier()?;
            Ok(InfluxStatement::ShowRetentionPolicies(db))
        } else {
            Err(QueryError::ParseError("Unknown SHOW command".to_string()))
        }
    }

    fn parse_create(&mut self) -> QueryResult<InfluxStatement> {
        self.consume_keyword("CREATE")?;

        if self.try_consume_keyword("DATABASE") {
            let name = self.parse_identifier()?;
            Ok(InfluxStatement::CreateDatabase(name))
        } else {
            Err(QueryError::ParseError("Expected DATABASE".to_string()))
        }
    }

    fn parse_drop(&mut self) -> QueryResult<InfluxStatement> {
        self.consume_keyword("DROP")?;

        if self.try_consume_keyword("DATABASE") {
            let name = self.parse_identifier()?;
            Ok(InfluxStatement::DropDatabase(name))
        } else {
            Err(QueryError::ParseError("Expected DATABASE".to_string()))
        }
    }

    fn parse_insert(&mut self) -> QueryResult<InfluxStatement> {
        self.consume_keyword("INSERT")?;
        let measurement = self.parse_identifier()?;

        Ok(InfluxStatement::Insert(InfluxInsert {
            measurement,
            tags: HashMap::new(),
            fields: HashMap::new(),
            timestamp: None,
        }))
    }

    fn parse_delete(&mut self) -> QueryResult<InfluxStatement> {
        self.consume_keyword("DELETE")?;
        self.consume_keyword("FROM")?;
        let measurement = self.parse_identifier()?;

        let where_clause = if self.try_consume_keyword("WHERE") {
            Some(self.parse_where()?)
        } else {
            None
        };

        Ok(InfluxStatement::Delete(InfluxDelete {
            measurement,
            where_clause,
        }))
    }

    fn parse_integer(&mut self) -> QueryResult<i64> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_ascii_digit() || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        self.input[start..self.pos]
            .parse()
            .map_err(|_| QueryError::ParseError("Expected integer".to_string()))
    }
}

impl Default for InfluxParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PromQL
// ============================================================================

/// PromQL query
#[derive(Debug, Clone)]
pub struct PromQuery {
    pub expr: PromExpr,
}

/// PromQL expression
#[derive(Debug, Clone)]
pub enum PromExpr {
    /// Metric name with optional labels
    Metric {
        name: String,
        labels: Vec<(String, String, LabelMatch)>,
    },
    /// Binary operation
    Binary {
        left: Box<PromExpr>,
        op: PromBinaryOp,
        right: Box<PromExpr>,
        modifier: Option<PromBinaryModifier>,
    },
    /// Unary operation
    Unary {
        op: PromUnaryOp,
        expr: Box<PromExpr>,
    },
    /// Function call
    Function { name: String, args: Vec<PromExpr> },
    /// Aggregation
    Aggregation {
        op: PromAggOp,
        expr: Box<PromExpr>,
        by: Option<Vec<String>>,
        without: Option<Vec<String>>,
        parameter: Option<f64>,
    },
    /// Subquery
    Subquery {
        expr: Box<PromExpr>,
        range: Duration,
        step: Option<Duration>,
    },
    /// Number literal
    Number(f64),
    /// String literal
    String(String),
    /// Range vector
    Range {
        expr: Box<PromExpr>,
        range: Duration,
        offset: Option<Duration>,
    },
    /// Offset modifier
    Offset {
        expr: Box<PromExpr>,
        offset: Duration,
    },
    /// @ modifier (timestamp)
    At { expr: Box<PromExpr>, timestamp: i64 },
}

/// Label match type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelMatch {
    Eq,  // =
    Ne,  // !=
    Re,  // =~
    Nre, // !~
}

/// PromQL binary operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromBinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    // Comparison
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
    // Logical
    And,
    Or,
    Unless,
}

/// Binary operation modifier
#[derive(Debug, Clone)]
pub struct PromBinaryModifier {
    pub matching: Option<PromVectorMatching>,
    pub group: Option<PromGroupModifier>,
    pub bool_modifier: bool,
}

/// Vector matching for binary ops
#[derive(Debug, Clone)]
pub struct PromVectorMatching {
    pub on: Option<Vec<String>>,
    pub ignoring: Option<Vec<String>>,
}

/// Group modifier for binary ops
#[derive(Debug, Clone)]
pub struct PromGroupModifier {
    pub left: bool,
    pub right: bool,
    pub labels: Vec<String>,
}

/// PromQL unary operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromUnaryOp {
    Plus,
    Minus,
}

/// PromQL aggregation operator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromAggOp {
    Sum,
    Min,
    Max,
    Avg,
    Group,
    Stddev,
    Stdvar,
    Count,
    CountValues,
    Bottomk,
    Topk,
    Quantile,
}

/// PromQL Parser
pub struct PromqlParser {
    input: String,
    pos: usize,
}

impl PromqlParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            input: String::new(),
            pos: 0,
        }
    }

    /// Parse PromQL query
    pub fn parse(&mut self, promql: &str) -> QueryResult<PromQuery> {
        self.input = promql.to_string();
        self.pos = 0;
        self.skip_whitespace();

        let expr = self.parse_expr()?;
        Ok(PromQuery { expr })
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else if self.input[self.pos..].starts_with('#') {
                while self.pos < self.input.len() && !self.input[self.pos..].starts_with('\n') {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn try_consume(&mut self, s: &str) -> bool {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    fn parse_identifier(&mut self) -> QueryResult<String> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_alphanumeric() || c == '_' || c == ':' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        if self.pos > start {
            Ok(self.input[start..self.pos].to_string())
        } else {
            Err(QueryError::ParseError("Expected identifier".to_string()))
        }
    }

    fn parse_expr(&mut self) -> QueryResult<PromExpr> {
        self.parse_or_expr()
    }

    fn parse_or_expr(&mut self) -> QueryResult<PromExpr> {
        let mut left = self.parse_and_expr()?;

        while self.try_consume("or") {
            let right = self.parse_and_expr()?;
            left = PromExpr::Binary {
                left: Box::new(left),
                op: PromBinaryOp::Or,
                right: Box::new(right),
                modifier: None,
            };
        }

        Ok(left)
    }

    fn parse_and_expr(&mut self) -> QueryResult<PromExpr> {
        let mut left = self.parse_comparison_expr()?;

        while self.try_consume("and") || self.try_consume("unless") {
            let op = if self.input[self.pos - 3..self.pos].ends_with("and") {
                PromBinaryOp::And
            } else {
                PromBinaryOp::Unless
            };
            let right = self.parse_comparison_expr()?;
            left = PromExpr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                modifier: None,
            };
        }

        Ok(left)
    }

    fn parse_comparison_expr(&mut self) -> QueryResult<PromExpr> {
        let mut left = self.parse_additive_expr()?;

        loop {
            let op = if self.try_consume("==") {
                PromBinaryOp::Eq
            } else if self.try_consume("!=") {
                PromBinaryOp::Ne
            } else if self.try_consume(">=") {
                PromBinaryOp::Ge
            } else if self.try_consume("<=") {
                PromBinaryOp::Le
            } else if self.try_consume(">") {
                PromBinaryOp::Gt
            } else if self.try_consume("<") {
                PromBinaryOp::Lt
            } else {
                break;
            };

            let right = self.parse_additive_expr()?;
            left = PromExpr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                modifier: None,
            };
        }

        Ok(left)
    }

    fn parse_additive_expr(&mut self) -> QueryResult<PromExpr> {
        let mut left = self.parse_multiplicative_expr()?;

        loop {
            let op = if self.try_consume("+") {
                PromBinaryOp::Add
            } else if self.try_consume("-") {
                PromBinaryOp::Sub
            } else {
                break;
            };

            let right = self.parse_multiplicative_expr()?;
            left = PromExpr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                modifier: None,
            };
        }

        Ok(left)
    }

    fn parse_multiplicative_expr(&mut self) -> QueryResult<PromExpr> {
        let mut left = self.parse_power_expr()?;

        loop {
            let op = if self.try_consume("*") {
                PromBinaryOp::Mul
            } else if self.try_consume("/") {
                PromBinaryOp::Div
            } else if self.try_consume("%") {
                PromBinaryOp::Mod
            } else {
                break;
            };

            let right = self.parse_power_expr()?;
            left = PromExpr::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
                modifier: None,
            };
        }

        Ok(left)
    }

    fn parse_power_expr(&mut self) -> QueryResult<PromExpr> {
        let left = self.parse_unary_expr()?;

        if self.try_consume("^") {
            let right = self.parse_power_expr()?;
            Ok(PromExpr::Binary {
                left: Box::new(left),
                op: PromBinaryOp::Pow,
                right: Box::new(right),
                modifier: None,
            })
        } else {
            Ok(left)
        }
    }

    fn parse_unary_expr(&mut self) -> QueryResult<PromExpr> {
        self.skip_whitespace();

        if self.try_consume("-") {
            let expr = self.parse_unary_expr()?;
            Ok(PromExpr::Unary {
                op: PromUnaryOp::Minus,
                expr: Box::new(expr),
            })
        } else if self.try_consume("+") {
            let expr = self.parse_unary_expr()?;
            Ok(PromExpr::Unary {
                op: PromUnaryOp::Plus,
                expr: Box::new(expr),
            })
        } else {
            self.parse_postfix_expr()
        }
    }

    fn parse_postfix_expr(&mut self) -> QueryResult<PromExpr> {
        let mut expr = self.parse_primary_expr()?;

        // Check for range vector [5m]
        if self.try_consume("[") {
            let range = self.parse_duration()?;
            self.skip_whitespace();
            if !self.try_consume("]") {
                return Err(QueryError::ParseError("Expected ']'".to_string()));
            }
            expr = PromExpr::Range {
                expr: Box::new(expr),
                range,
                offset: None,
            };
        }

        // Check for offset modifier
        if self.try_consume("offset") {
            let offset = self.parse_duration()?;
            expr = PromExpr::Offset {
                expr: Box::new(expr),
                offset,
            };
        }

        Ok(expr)
    }

    fn parse_primary_expr(&mut self) -> QueryResult<PromExpr> {
        self.skip_whitespace();

        // Check for aggregation
        let agg_op = self.try_parse_agg_op();
        if let Some(op) = agg_op {
            return self.parse_aggregation(op);
        }

        // Parenthesized expression
        if self.try_consume("(") {
            let expr = self.parse_expr()?;
            if !self.try_consume(")") {
                return Err(QueryError::ParseError("Expected ')'".to_string()));
            }
            return Ok(expr);
        }

        // Number
        if let Some(n) = self.try_parse_number() {
            return Ok(PromExpr::Number(n));
        }

        // String
        if self.input[self.pos..].starts_with('"') || self.input[self.pos..].starts_with('\'') {
            let s = self.parse_string()?;
            return Ok(PromExpr::String(s));
        }

        // Metric or function
        let name = self.parse_identifier()?;

        // Check for function call
        if self.try_consume("(") {
            let mut args = Vec::new();
            if !self.input[self.pos..].trim_start().starts_with(')') {
                loop {
                    args.push(self.parse_expr()?);
                    if !self.try_consume(",") {
                        break;
                    }
                }
            }
            if !self.try_consume(")") {
                return Err(QueryError::ParseError("Expected ')'".to_string()));
            }
            return Ok(PromExpr::Function { name, args });
        }

        // Metric with labels
        let labels = if self.try_consume("{") {
            let l = self.parse_labels()?;
            if !self.try_consume("}") {
                return Err(QueryError::ParseError("Expected '}'".to_string()));
            }
            l
        } else {
            Vec::new()
        };

        Ok(PromExpr::Metric { name, labels })
    }

    fn try_parse_agg_op(&mut self) -> Option<PromAggOp> {
        let ops = [
            ("sum", PromAggOp::Sum),
            ("min", PromAggOp::Min),
            ("max", PromAggOp::Max),
            ("avg", PromAggOp::Avg),
            ("group", PromAggOp::Group),
            ("stddev", PromAggOp::Stddev),
            ("stdvar", PromAggOp::Stdvar),
            ("count", PromAggOp::Count),
            ("count_values", PromAggOp::CountValues),
            ("bottomk", PromAggOp::Bottomk),
            ("topk", PromAggOp::Topk),
            ("quantile", PromAggOp::Quantile),
        ];

        for (name, op) in ops {
            if self.input[self.pos..].to_lowercase().starts_with(name) {
                let after = self.input[self.pos + name.len()..].chars().next();
                if after
                    .map(|c| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(true)
                {
                    self.pos += name.len();
                    return Some(op);
                }
            }
        }
        None
    }

    fn parse_aggregation(&mut self, op: PromAggOp) -> QueryResult<PromExpr> {
        let mut by = None;
        let mut without = None;
        let mut parameter = None;

        // Check for by/without before arguments
        if self.try_consume("by") {
            if !self.try_consume("(") {
                return Err(QueryError::ParseError("Expected '('".to_string()));
            }
            by = Some(self.parse_label_list()?);
            if !self.try_consume(")") {
                return Err(QueryError::ParseError("Expected ')'".to_string()));
            }
        } else if self.try_consume("without") {
            if !self.try_consume("(") {
                return Err(QueryError::ParseError("Expected '('".to_string()));
            }
            without = Some(self.parse_label_list()?);
            if !self.try_consume(")") {
                return Err(QueryError::ParseError("Expected ')'".to_string()));
            }
        }

        // Parse arguments
        if !self.try_consume("(") {
            return Err(QueryError::ParseError("Expected '('".to_string()));
        }

        // For topk/bottomk/quantile, first arg is parameter
        if matches!(
            op,
            PromAggOp::Topk | PromAggOp::Bottomk | PromAggOp::Quantile
        ) {
            if let Some(n) = self.try_parse_number() {
                parameter = Some(n);
                self.try_consume(",");
            }
        }

        let expr = self.parse_expr()?;

        if !self.try_consume(")") {
            return Err(QueryError::ParseError("Expected ')'".to_string()));
        }

        // Check for by/without after arguments
        if by.is_none() && without.is_none() {
            if self.try_consume("by") {
                if !self.try_consume("(") {
                    return Err(QueryError::ParseError("Expected '('".to_string()));
                }
                by = Some(self.parse_label_list()?);
                if !self.try_consume(")") {
                    return Err(QueryError::ParseError("Expected ')'".to_string()));
                }
            } else if self.try_consume("without") {
                if !self.try_consume("(") {
                    return Err(QueryError::ParseError("Expected '('".to_string()));
                }
                without = Some(self.parse_label_list()?);
                if !self.try_consume(")") {
                    return Err(QueryError::ParseError("Expected ')'".to_string()));
                }
            }
        }

        Ok(PromExpr::Aggregation {
            op,
            expr: Box::new(expr),
            by,
            without,
            parameter,
        })
    }

    fn parse_labels(&mut self) -> QueryResult<Vec<(String, String, LabelMatch)>> {
        let mut labels = Vec::new();

        loop {
            self.skip_whitespace();
            if self.input[self.pos..].starts_with('}') {
                break;
            }

            let name = self.parse_identifier()?;
            self.skip_whitespace();

            let match_type = if self.try_consume("=~") {
                LabelMatch::Re
            } else if self.try_consume("!~") {
                LabelMatch::Nre
            } else if self.try_consume("!=") {
                LabelMatch::Ne
            } else if self.try_consume("=") {
                LabelMatch::Eq
            } else {
                return Err(QueryError::ParseError("Expected label matcher".to_string()));
            };

            let value = self.parse_string()?;
            labels.push((name, value, match_type));

            self.skip_whitespace();
            if !self.try_consume(",") {
                break;
            }
        }

        Ok(labels)
    }

    fn parse_label_list(&mut self) -> QueryResult<Vec<String>> {
        let mut labels = Vec::new();

        loop {
            self.skip_whitespace();
            if self.input[self.pos..].starts_with(')') {
                break;
            }

            labels.push(self.parse_identifier()?);

            self.skip_whitespace();
            if !self.try_consume(",") {
                break;
            }
        }

        Ok(labels)
    }

    fn parse_string(&mut self) -> QueryResult<String> {
        self.skip_whitespace();
        let quote = if self.input[self.pos..].starts_with('"') {
            '"'
        } else if self.input[self.pos..].starts_with('\'') {
            '\''
        } else {
            return Err(QueryError::ParseError("Expected string".to_string()));
        };

        self.pos += 1;
        let mut s = String::new();
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c == quote {
                self.pos += 1;
                return Ok(s);
            } else if c == '\\' {
                self.pos += 1;
                if let Some(escaped) = self.input[self.pos..].chars().next() {
                    self.pos += escaped.len_utf8();
                    match escaped {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        '\\' => s.push('\\'),
                        _ => s.push(escaped),
                    }
                }
            } else {
                s.push(c);
                self.pos += c.len_utf8();
            }
        }
        Err(QueryError::ParseError("Unterminated string".to_string()))
    }

    fn try_parse_number(&mut self) -> Option<f64> {
        self.skip_whitespace();
        let start = self.pos;
        let mut has_dot = false;
        let mut has_exp = false;

        if self.input[self.pos..].starts_with('-') || self.input[self.pos..].starts_with('+') {
            self.pos += 1;
        }

        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos/end < len guarantees char exists");
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == '.' && !has_dot {
                has_dot = true;
                self.pos += 1;
            } else if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                self.pos += 1;
                if self.input[self.pos..].starts_with('-')
                    || self.input[self.pos..].starts_with('+')
                {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }

        if self.pos > start {
            let num_str = &self.input[start..self.pos];
            if let Ok(n) = num_str.parse() {
                return Some(n);
            }
        }

        self.pos = start;
        None
    }

    fn parse_duration(&mut self) -> QueryResult<Duration> {
        self.skip_whitespace();
        let mut total_nanos: u64 = 0;

        loop {
            let start = self.pos;
            while self.pos < self.input.len() {
                let c = self.input[self.pos..]
                    .chars()
                    .next()
                    .expect("pos/end < len guarantees char exists");
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }

            if self.pos == start {
                break;
            }

            let num: u64 = self.input[start..self.pos].parse().unwrap_or(0);

            let unit_start = self.pos;
            while self.pos < self.input.len() {
                let c = self.input[self.pos..]
                    .chars()
                    .next()
                    .expect("pos/end < len guarantees char exists");
                if c.is_alphabetic() {
                    self.pos += c.len_utf8();
                } else {
                    break;
                }
            }

            let unit = &self.input[unit_start..self.pos];
            let multiplier = match unit {
                "ms" => 1_000_000,
                "s" => 1_000_000_000,
                "m" => 60_000_000_000,
                "h" => 3_600_000_000_000u64,
                "d" => 86_400_000_000_000u64,
                "w" => 604_800_000_000_000u64,
                "y" => 31_536_000_000_000_000u64,
                _ => {
                    return Err(QueryError::ParseError(format!(
                        "Unknown duration unit: {}",
                        unit
                    )));
                }
            };

            total_nanos += num * multiplier;
        }

        if total_nanos == 0 {
            return Err(QueryError::ParseError("Expected duration".to_string()));
        }

        Ok(Duration::from_nanos(total_nanos))
    }
}

impl Default for PromqlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // InfluxQL tests
    #[test]
    fn test_influx_simple_select() {
        let mut parser = InfluxParser::new();
        let query = parser.parse("SELECT * FROM cpu").unwrap();

        match query.statement {
            InfluxStatement::Select(s) => {
                assert_eq!(s.from[0].name, "cpu");
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_influx_select_with_function() {
        let mut parser = InfluxParser::new();
        let query = parser.parse("SELECT MEAN(value) FROM cpu").unwrap();

        match query.statement {
            InfluxStatement::Select(s) => {
                assert_eq!(s.columns.len(), 1);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_influx_select_with_where() {
        let mut parser = InfluxParser::new();
        let query = parser
            .parse("SELECT * FROM cpu WHERE host = 'server01'")
            .unwrap();

        match query.statement {
            InfluxStatement::Select(s) => {
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_influx_select_with_group_by() {
        let mut parser = InfluxParser::new();
        let query = parser
            .parse("SELECT MEAN(value) FROM cpu GROUP BY TIME(1h), host")
            .unwrap();

        match query.statement {
            InfluxStatement::Select(s) => {
                let group = s.group_by.unwrap();
                assert!(group.time_interval.is_some());
                assert!(group.tags.contains(&"host".to_string()));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_influx_show_databases() {
        let mut parser = InfluxParser::new();
        let query = parser.parse("SHOW DATABASES").unwrap();

        assert!(matches!(query.statement, InfluxStatement::ShowDatabases));
    }

    // PromQL tests
    #[test]
    fn test_prom_simple_metric() {
        let mut parser = PromqlParser::new();
        let query = parser.parse("http_requests_total").unwrap();

        match query.expr {
            PromExpr::Metric { name, .. } => {
                assert_eq!(name, "http_requests_total");
            }
            _ => panic!("Expected metric"),
        }
    }

    #[test]
    fn test_prom_metric_with_labels() {
        let mut parser = PromqlParser::new();
        let query = parser
            .parse("http_requests_total{method=\"GET\", status=\"200\"}")
            .unwrap();

        match query.expr {
            PromExpr::Metric { labels, .. } => {
                assert_eq!(labels.len(), 2);
            }
            _ => panic!("Expected metric"),
        }
    }

    #[test]
    fn test_prom_range_vector() {
        let mut parser = PromqlParser::new();
        let query = parser.parse("http_requests_total[5m]").unwrap();

        match query.expr {
            PromExpr::Range { range, .. } => {
                assert_eq!(range, Duration::from_secs(300));
            }
            _ => panic!("Expected range"),
        }
    }

    #[test]
    fn test_prom_aggregation() {
        let mut parser = PromqlParser::new();
        let query = parser.parse("sum by (job) (http_requests_total)").unwrap();

        match query.expr {
            PromExpr::Aggregation { op, by, .. } => {
                assert_eq!(op, PromAggOp::Sum);
                assert_eq!(by, Some(vec!["job".to_string()]));
            }
            _ => panic!("Expected aggregation"),
        }
    }

    #[test]
    fn test_prom_binary_expr() {
        let mut parser = PromqlParser::new();
        let query = parser.parse("http_requests_total / 1000").unwrap();

        match query.expr {
            PromExpr::Binary {
                op: PromBinaryOp::Div,
                ..
            } => {}
            _ => panic!("Expected binary"),
        }
    }

    #[test]
    fn test_prom_function() {
        let mut parser = PromqlParser::new();
        let query = parser.parse("rate(http_requests_total[5m])").unwrap();

        match query.expr {
            PromExpr::Function { name, args } => {
                assert_eq!(name, "rate");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected function"),
        }
    }

    #[test]
    fn test_prom_offset() {
        let mut parser = PromqlParser::new();
        let query = parser.parse("http_requests_total offset 1h").unwrap();

        match query.expr {
            PromExpr::Offset { offset, .. } => {
                assert_eq!(offset, Duration::from_secs(3600));
            }
            _ => panic!("Expected offset"),
        }
    }
}
