//! Abstract Syntax Tree for Query Languages
//!
//! Common AST types shared across different query languages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Select/Read query
    Select,
    /// Insert query
    Insert,
    /// Update query
    Update,
    /// Delete query
    Delete,
    /// Create (table, index, etc)
    Create,
    /// Drop (table, index, etc)
    Drop,
    /// Alter (table, etc)
    Alter,
    /// Graph traversal
    Traverse,
    /// Aggregation
    Aggregate,
    /// Transaction control
    Transaction,
}

/// Generic query representation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Query {
    /// Query type
    pub query_type: QueryType,
    /// Source (table, collection, etc)
    pub source: Option<String>,
    /// Target columns/fields
    pub columns: Vec<String>,
    /// Filter condition
    pub filter: Option<Expression>,
    /// Order by clauses
    pub order_by: Vec<OrderBy>,
    /// Group by clauses (expressions, not just column names)
    pub group_by: Vec<Expression>,
    /// Having clause
    pub having: Option<Expression>,
    /// Limit
    pub limit: Option<usize>,
    /// Offset
    pub offset: Option<usize>,
    /// Joins
    pub joins: Vec<Join>,
    /// Values (for insert/update)
    pub values: Vec<HashMap<String, Value>>,
    /// Returning clause
    pub returning: Vec<String>,
    /// Common Table Expressions
    pub ctes: Vec<Cte>,
    /// Derived columns (expressions)
    pub derived_columns: HashMap<String, Expression>,
    /// Whether to deduplicate result rows (SELECT DISTINCT)
    pub distinct: bool,
    /// Source alias (e.g., "e" in "FROM employees e")
    pub source_alias: Option<String>,
}

/// Common Table Expression
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Cte {
    pub name: String,
    pub columns: Vec<String>,
    pub query: Box<Query>,
    pub recursive: bool,
}

impl Query {
    /// Create a new select query
    pub fn select(table: &str) -> Self {
        Self {
            query_type: QueryType::Select,
            source: Some(table.to_string()),
            columns: vec!["*".to_string()],
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
            distinct: false,
            source_alias: None,
        }
    }

    /// Create a new insert query
    pub fn insert(table: &str) -> Self {
        Self {
            query_type: QueryType::Insert,
            source: Some(table.to_string()),
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
            distinct: false,
            source_alias: None,
        }
    }

    /// Add columns to select
    pub fn columns(mut self, cols: Vec<&str>) -> Self {
        self.columns = cols.into_iter().map(String::from).collect();
        self
    }

    /// Add filter condition
    pub fn filter(mut self, expr: Expression) -> Self {
        self.filter = Some(expr);
        self
    }

    /// Add limit
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Add offset
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    /// Add order by
    pub fn order_by(mut self, column: &str, descending: bool) -> Self {
        self.order_by.push(OrderBy {
            expr: Expression::Column(column.to_string()),
            descending,
            nulls_first: None,
        });
        self
    }
}

/// Expression node
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Expression {
    /// Literal value
    Literal(Value),
    /// Column reference
    Column(String),
    /// Qualified column (table.column)
    QualifiedColumn { table: String, column: String },
    /// Binary operation
    Binary {
        left: Box<Expression>,
        op: Operator,
        right: Box<Expression>,
    },
    /// Unary operation
    Unary {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    /// Function call
    Function { name: String, args: Vec<Expression> },
    /// Subquery
    Subquery(Box<Query>),
    /// CASE expression
    Case {
        operand: Option<Box<Expression>>,
        when_clauses: Vec<(Expression, Expression)>,
        else_clause: Option<Box<Expression>>,
    },
    /// IN expression
    In {
        expr: Box<Expression>,
        list: Vec<Expression>,
        negated: bool,
    },
    /// BETWEEN expression
    Between {
        expr: Box<Expression>,
        low: Box<Expression>,
        high: Box<Expression>,
        negated: bool,
    },
    /// IS NULL expression
    IsNull {
        expr: Box<Expression>,
        negated: bool,
    },
    /// LIKE / ILIKE expression
    Like {
        expr: Box<Expression>,
        pattern: String,
        negated: bool,
        case_insensitive: bool,
    },
    /// Regex match (Cypher =~ operator)
    RegexMatch {
        expr: Box<Expression>,
        pattern: String,
        negated: bool,
    },
    /// EXISTS expression
    Exists(Box<Query>),
    /// Parameter placeholder
    Parameter(usize),
    /// Named parameter
    NamedParameter(String),
    /// Wildcard (*)
    Wildcard,
    /// Qualified wildcard (table.*)
    QualifiedWildcard(String),
    /// Window function
    WindowFunction {
        function: String,
        args: Vec<Expression>,
        window: WindowSpec,
    },
    /// SIMILAR TO expression (fuzzy matching with threshold)
    /// Syntax: column SIMILAR TO 'pattern' [THRESHOLD 0.8]
    SimilarTo {
        expr: Box<Expression>,
        pattern: String,
        threshold: Option<f64>,
        negated: bool,
    },
    /// LIKE MEANING expression (semantic similarity search)
    /// Syntax: column LIKE MEANING 'concept'
    LikeMeaning {
        expr: Box<Expression>,
        concept: String,
        negated: bool,
    },
    /// CAST expression
    /// Syntax: CAST(expr AS type)
    Cast {
        expr: Box<Expression>,
        target_type: String,
    },
    /// Reverse reference traversal (~>reference_name)
    /// Syntax: ~>author_books
    ReverseReference { reference_name: String },
}

impl Expression {
    /// Create a column reference
    pub fn column(name: &str) -> Self {
        Self::Column(name.to_string())
    }

    /// Create a literal value
    pub fn literal(value: Value) -> Self {
        Self::Literal(value)
    }

    /// Create a binary expression
    pub fn binary(left: Expression, op: Operator, right: Expression) -> Self {
        Self::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// Create an AND expression
    pub fn and(left: Expression, right: Expression) -> Self {
        Self::binary(left, Operator::And, right)
    }

    /// Create an OR expression
    pub fn or(left: Expression, right: Expression) -> Self {
        Self::binary(left, Operator::Or, right)
    }

    /// Create an equals expression
    pub fn eq(left: Expression, right: Expression) -> Self {
        Self::binary(left, Operator::Eq, right)
    }

    /// Create a function call
    pub fn function(name: &str, args: Vec<Expression>) -> Self {
        Self::Function {
            name: name.to_string(),
            args,
        }
    }
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operator {
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical
    And,
    Or,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // String
    Concat,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    // JSON
    JsonArrow,           // ->  (extract JSON object field as JSON)
    JsonDoubleArrow,     // ->> (extract JSON object field as text)
    JsonHashArrow,       // #>  (extract JSON path as JSON)
    JsonHashDoubleArrow, // #>> (extract JSON path as text)
    JsonContains,        // @>  (left contains right)
    JsonContainedBy,     // <@  (left is contained by right)
    JsonExists,          // ?   (does key exist)
    // Vector distance (pgvector-compatible)
    VectorL2Distance,     // <->  (L2/Euclidean distance)
    VectorIPDistance,     // <#>  (negative inner product)
    VectorCosineDistance, // <=>  (cosine distance)
}

impl Operator {
    /// Get operator precedence (higher = binds tighter)
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Or => 1,
            Self::And => 2,
            Self::Eq | Self::Ne | Self::Lt | Self::Le | Self::Gt | Self::Ge => 3,
            Self::BitOr => 4,
            Self::BitXor => 5,
            Self::BitAnd => 6,
            Self::Add | Self::Sub | Self::Concat => 7,
            Self::Mul | Self::Div | Self::Mod => 8,
            // JSON operators bind tighter than comparison but lower than arithmetic
            Self::JsonArrow
            | Self::JsonDoubleArrow
            | Self::JsonHashArrow
            | Self::JsonHashDoubleArrow => 9,
            Self::JsonContains | Self::JsonContainedBy | Self::JsonExists => 3,
            Self::VectorL2Distance | Self::VectorIPDistance | Self::VectorCosineDistance => 3,
        }
    }
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Neg,
    BitNot,
}

/// Value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
    Timestamp(i64),
    Uuid(String),
    Vector(Vec<f32>),
}

impl Value {
    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to get as i64
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get as f64
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Try to get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

/// Order by clause
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OrderBy {
    pub expr: Expression,
    pub descending: bool,
    pub nulls_first: Option<bool>,
}

/// Join clause
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Join {
    pub join_type: JoinType,
    pub table: String,
    pub alias: Option<String>,
    pub condition: Option<Expression>,
    /// Columns for USING clause (e.g., JOIN t2 USING (id, name))
    pub using_columns: Vec<String>,
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// Aggregate function
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    First,
    Last,
    ArrayAgg,
    StringAgg,
}

/// Window specification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WindowSpec {
    pub partition_by: Vec<Expression>,
    pub order_by: Vec<OrderBy>,
    pub frame: Option<WindowFrame>,
}

/// Window frame definition
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WindowFrame {
    pub units: WindowFrameUnits,
    pub start_bound: WindowFrameBound,
    pub end_bound: WindowFrameBound,
}

/// Window frame units
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WindowFrameUnits {
    Rows,
    Range,
    Groups,
}

/// Window frame bound
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum WindowFrameBound {
    CurrentRow,
    Preceding(u64),
    Following(u64),
    UnboundedPreceding,
    UnboundedFollowing,
}

impl WindowFrame {
    /// Compute the (start, end) index range (inclusive) within a sorted partition
    /// for a row at position `pos` in a partition of length `partition_len`.
    /// Returns indices into the sorted partition array.
    pub fn frame_range(&self, pos: usize, partition_len: usize) -> (usize, usize) {
        if partition_len == 0 {
            return (0, 0);
        }
        let start = self.resolve_bound(&self.start_bound, pos, partition_len, true);
        let end = self.resolve_bound(&self.end_bound, pos, partition_len, false);
        // Ensure valid range
        let end = end.max(start);
        (start, end.min(partition_len - 1))
    }

    fn resolve_bound(
        &self,
        bound: &WindowFrameBound,
        pos: usize,
        partition_len: usize,
        _is_start: bool,
    ) -> usize {
        match bound {
            WindowFrameBound::UnboundedPreceding => 0,
            WindowFrameBound::Preceding(n) => pos.saturating_sub((*n).max(0) as usize),
            WindowFrameBound::CurrentRow => pos,
            WindowFrameBound::Following(n) => (pos + (*n).max(0) as usize).min(partition_len - 1),
            WindowFrameBound::UnboundedFollowing => partition_len - 1,
        }
    }
}

// --- JSON operator helpers ---

/// Convert an ast::Value to serde_json::Value for JSON manipulation.
/// String values are attempted to be parsed as JSON first.
pub fn value_to_serde_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::json!(*i),
        Value::Float(f) => serde_json::json!(*f),
        Value::String(s) => {
            // Try parsing as JSON first (for stored JSON strings)
            if (s.starts_with('{') && s.ends_with('}'))
                || (s.starts_with('[') && s.ends_with(']'))
                || s == "true"
                || s == "false"
                || s == "null"
            {
                serde_json::from_str(s).unwrap_or_else(|_| serde_json::Value::String(s.clone()))
            } else {
                serde_json::Value::String(s.clone())
            }
        }
        Value::Array(arr) => {
            let items: Vec<serde_json::Value> = arr.iter().map(value_to_serde_json).collect();
            serde_json::Value::Array(items)
        }
        Value::Object(map) => {
            let obj: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), value_to_serde_json(v)))
                .collect();
            serde_json::Value::Object(obj)
        }
        Value::Timestamp(t) => serde_json::json!(*t),
        Value::Uuid(u) => serde_json::Value::String(u.clone()),
        Value::Bytes(b) => serde_json::json!(b),
        Value::Vector(v) => serde_json::json!(v),
    }
}

/// Convert a serde_json::Value back to ast::Value.
pub fn serde_json_to_value(v: serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => Value::String(serde_json::Value::Array(arr).to_string()),
        serde_json::Value::Object(obj) => Value::String(serde_json::Value::Object(obj).to_string()),
    }
}

/// Parse a PostgreSQL-style path literal like '{a,b,c}' into a vec of keys.
fn parse_pg_path(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => {
            let s = s.trim();
            if s.starts_with('{') && s.ends_with('}') {
                s[1..s.len() - 1]
                    .split(',')
                    .map(|p| p.trim().to_string())
                    .collect()
            } else {
                vec![s.to_string()]
            }
        }
        _ => vec![],
    }
}

/// Check if `container` JSON value contains `contained` (PostgreSQL @> semantics).
pub fn json_contains(container: &serde_json::Value, contained: &serde_json::Value) -> bool {
    match (container, contained) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => b
            .iter()
            .all(|(k, v)| a.get(k).is_some_and(|av| json_contains(av, v))),
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            b.iter().all(|bv| a.iter().any(|av| json_contains(av, bv)))
        }
        (a, b) => a == b,
    }
}

/// Extract a value from a JSON value by key (string) or index (int).
fn json_extract(json: &serde_json::Value, key: &Value) -> serde_json::Value {
    match key {
        Value::String(k) => {
            if let serde_json::Value::Object(map) = json {
                map.get(k).cloned().unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
        Value::Int(idx) => {
            if let serde_json::Value::Array(arr) = json {
                let i = if *idx < 0 {
                    arr.len().saturating_sub((-*idx) as usize)
                } else {
                    *idx as usize
                };
                arr.get(i).cloned().unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
        _ => serde_json::Value::Null,
    }
}

/// Traverse a JSON value by a path (vec of keys).
fn json_traverse(json: &serde_json::Value, path: &[String]) -> serde_json::Value {
    let mut current = json.clone();
    for key in path {
        current = match &current {
            serde_json::Value::Object(map) => {
                map.get(key).cloned().unwrap_or(serde_json::Value::Null)
            }
            serde_json::Value::Array(arr) => {
                if let Ok(idx) = key.parse::<usize>() {
                    arr.get(idx).cloned().unwrap_or(serde_json::Value::Null)
                } else {
                    serde_json::Value::Null
                }
            }
            _ => serde_json::Value::Null,
        };
    }
    current
}

/// Convert a serde_json::Value to its text representation (for ->> and #>> operators).
fn json_to_text(v: serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::String(s) => Value::String(s),
        other => Value::String(other.to_string()),
    }
}

/// Evaluate a JSON binary operator on ast::Value operands.
/// Used by executor.rs, execution.rs, and storage_executor.rs.
pub fn eval_json_operator(left: &Value, op: &Operator, right: &Value) -> Value {
    let left_json = value_to_serde_json(left);

    match op {
        Operator::JsonArrow => serde_json_to_value(json_extract(&left_json, right)),
        Operator::JsonDoubleArrow => json_to_text(json_extract(&left_json, right)),
        Operator::JsonHashArrow => {
            let path = parse_pg_path(right);
            serde_json_to_value(json_traverse(&left_json, &path))
        }
        Operator::JsonHashDoubleArrow => {
            let path = parse_pg_path(right);
            json_to_text(json_traverse(&left_json, &path))
        }
        Operator::JsonContains => {
            let right_json = value_to_serde_json(right);
            Value::Bool(json_contains(&left_json, &right_json))
        }
        Operator::JsonContainedBy => {
            let right_json = value_to_serde_json(right);
            Value::Bool(json_contains(&right_json, &left_json))
        }
        Operator::JsonExists => match right {
            Value::String(key) => match &left_json {
                serde_json::Value::Object(map) => Value::Bool(map.contains_key(key)),
                serde_json::Value::Array(arr) => {
                    let key_json = serde_json::Value::String(key.clone());
                    Value::Bool(arr.contains(&key_json))
                }
                _ => Value::Bool(false),
            },
            _ => Value::Bool(false),
        },
        _ => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder() {
        let query = Query::select("users")
            .columns(vec!["id", "name"])
            .filter(Expression::eq(
                Expression::column("id"),
                Expression::literal(Value::Int(1)),
            ))
            .limit(10);

        assert_eq!(query.query_type, QueryType::Select);
        assert_eq!(query.source, Some("users".to_string()));
        assert_eq!(query.columns, vec!["id", "name"]);
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_expression_builders() {
        let expr = Expression::and(
            Expression::eq(Expression::column("a"), Expression::literal(1.into())),
            Expression::eq(Expression::column("b"), Expression::literal(2.into())),
        );

        match expr {
            Expression::Binary {
                op: Operator::And, ..
            } => {}
            _ => panic!("Expected AND expression"),
        }
    }

    #[test]
    fn test_value_conversions() {
        let v: Value = 42i64.into();
        assert_eq!(v.as_int(), Some(42));

        let v: Value = "hello".into();
        assert_eq!(v.as_str(), Some("hello"));

        let v: Value = true.into();
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_operator_precedence() {
        assert!(Operator::Mul.precedence() > Operator::Add.precedence());
        assert!(Operator::And.precedence() > Operator::Or.precedence());
    }
}
