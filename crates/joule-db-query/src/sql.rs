//! SQL Parser and AST
//!
//! Parses a subset of SQL compatible with most databases.

use crate::ast::{
    Cte, Expression, Join, JoinType, Operator, OrderBy, Query, QueryType, Value, WindowFrame,
    WindowFrameBound, WindowFrameUnits, WindowSpec,
};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Chars;

/// Convert a token back to its SQL text representation.
fn token_to_sql(token: &Token) -> String {
    match token {
        Token::Select => "SELECT".to_string(),
        Token::From => "FROM".to_string(),
        Token::Where => "WHERE".to_string(),
        Token::And => "AND".to_string(),
        Token::Or => "OR".to_string(),
        Token::Not => "NOT".to_string(),
        Token::Null => "NULL".to_string(),
        Token::As => "AS".to_string(),
        Token::On => "ON".to_string(),
        Token::In => "IN".to_string(),
        Token::Is => "IS".to_string(),
        Token::Like => "LIKE".to_string(),
        Token::ILike => "ILIKE".to_string(),
        Token::Between => "BETWEEN".to_string(),
        Token::Join => "JOIN".to_string(),
        Token::Inner => "INNER".to_string(),
        Token::Left => "LEFT".to_string(),
        Token::Right => "RIGHT".to_string(),
        Token::Full => "FULL".to_string(),
        Token::Cross => "CROSS".to_string(),
        Token::Outer => "OUTER".to_string(),
        Token::Group => "GROUP".to_string(),
        Token::By => "BY".to_string(),
        Token::Having => "HAVING".to_string(),
        Token::Order => "ORDER".to_string(),
        Token::Asc => "ASC".to_string(),
        Token::Desc => "DESC".to_string(),
        Token::Limit => "LIMIT".to_string(),
        Token::Offset => "OFFSET".to_string(),
        Token::Insert => "INSERT".to_string(),
        Token::Into => "INTO".to_string(),
        Token::Values => "VALUES".to_string(),
        Token::Update => "UPDATE".to_string(),
        Token::Set => "SET".to_string(),
        Token::Delete => "DELETE".to_string(),
        Token::Create => "CREATE".to_string(),
        Token::Drop => "DROP".to_string(),
        Token::Table => "TABLE".to_string(),
        Token::Index => "INDEX".to_string(),
        Token::If => "IF".to_string(),
        Token::Exists => "EXISTS".to_string(),
        Token::Unique => "UNIQUE".to_string(),
        Token::Primary => "PRIMARY".to_string(),
        Token::Key => "KEY".to_string(),
        Token::Default => "DEFAULT".to_string(),
        Token::Returning => "RETURNING".to_string(),
        Token::Begin => "BEGIN".to_string(),
        Token::Commit => "COMMIT".to_string(),
        Token::Rollback => "ROLLBACK".to_string(),
        Token::Distinct => "DISTINCT".to_string(),
        Token::All => "ALL".to_string(),
        Token::Union => "UNION".to_string(),
        Token::Except => "EXCEPT".to_string(),
        Token::Intersect => "INTERSECT".to_string(),
        Token::Case => "CASE".to_string(),
        Token::When => "WHEN".to_string(),
        Token::Then => "THEN".to_string(),
        Token::Else => "ELSE".to_string(),
        Token::End => "END".to_string(),
        Token::True => "TRUE".to_string(),
        Token::False => "FALSE".to_string(),
        Token::With => "WITH".to_string(),
        Token::Recursive => "RECURSIVE".to_string(),
        Token::Date => "DATE".to_string(),
        Token::Interval => "INTERVAL".to_string(),
        Token::Extract => "EXTRACT".to_string(),
        Token::Year => "YEAR".to_string(),
        Token::Month => "MONTH".to_string(),
        Token::Day => "DAY".to_string(),
        Token::Hour => "HOUR".to_string(),
        Token::Minute => "MINUTE".to_string(),
        Token::Second => "SECOND".to_string(),
        Token::Over => "OVER".to_string(),
        Token::Partition => "PARTITION".to_string(),
        Token::Window => "WINDOW".to_string(),
        Token::Rows => "ROWS".to_string(),
        Token::Row => "ROW".to_string(),
        Token::Range => "RANGE".to_string(),
        Token::Groups => "GROUPS".to_string(),
        Token::Current => "CURRENT".to_string(),
        Token::Preceding => "PRECEDING".to_string(),
        Token::Following => "FOLLOWING".to_string(),
        Token::Unbounded => "UNBOUNDED".to_string(),
        Token::Alter => "ALTER".to_string(),
        Token::Add => "ADD".to_string(),
        Token::Column => "COLUMN".to_string(),
        Token::Rename => "RENAME".to_string(),
        Token::To => "TO".to_string(),
        Token::View => "VIEW".to_string(),
        Token::Replace => "REPLACE".to_string(),
        Token::Truncate => "TRUNCATE".to_string(),
        Token::Check => "CHECK".to_string(),
        Token::Foreign => "FOREIGN".to_string(),
        Token::References => "REFERENCES".to_string(),
        Token::Cascade => "CASCADE".to_string(),
        Token::Restrict => "RESTRICT".to_string(),
        Token::Natural => "NATURAL".to_string(),
        Token::Using => "USING".to_string(),
        Token::Show => "SHOW".to_string(),
        Token::Conflict => "CONFLICT".to_string(),
        Token::Do => "DO".to_string(),
        Token::Nothing => "NOTHING".to_string(),
        Token::Cast => "CAST".to_string(),
        Token::Materialized => "MATERIALIZED".to_string(),
        Token::Refresh => "REFRESH".to_string(),
        Token::Family => "FAMILY".to_string(),
        Token::Spatial => "SPATIAL".to_string(),
        Token::Vector => "VECTOR".to_string(),
        Token::Fulltext => "FULLTEXT".to_string(),
        Token::Match => "MATCH".to_string(),
        Token::Against => "AGAINST".to_string(),
        Token::Similar => "SIMILAR".to_string(),
        Token::Meaning => "MEANING".to_string(),
        Token::Threshold => "THRESHOLD".to_string(),
        Token::Nearest => "NEAREST".to_string(),
        Token::Explain => "EXPLAIN".to_string(),
        Token::Analyze => "ANALYZE".to_string(),
        Token::Savepoint => "SAVEPOINT".to_string(),
        Token::Release => "RELEASE".to_string(),
        Token::Lateral => "LATERAL".to_string(),
        Token::Fetch => "FETCH".to_string(),
        Token::Include => "INCLUDE".to_string(),
        Token::Trigger => "TRIGGER".to_string(),
        Token::Before => "BEFORE".to_string(),
        Token::After => "AFTER".to_string(),
        Token::Each => "EACH".to_string(),
        Token::Execute => "EXECUTE".to_string(),
        Token::Grant => "GRANT".to_string(),
        Token::Revoke => "REVOKE".to_string(),
        Token::User => "USER".to_string(),
        Token::Role => "ROLE".to_string(),
        Token::Password => "PASSWORD".to_string(),
        Token::Shard => "SHARD".to_string(),
        Token::Computed => "COMPUTED".to_string(),
        Token::Define => "DEFINE".to_string(),
        Token::Reference => "REFERENCE".to_string(),
        Token::TildeArrow => "~>".to_string(),
        Token::Integer(n) => n.to_string(),
        Token::Float(f) => f.to_string(),
        Token::String(s) => format!("'{}'", s.replace('\'', "''")),
        Token::Identifier(s) => s.clone(),
        Token::QuotedIdentifier(s) => format!("\"{}\"", s),
        Token::Star => "*".to_string(),
        Token::Plus => "+".to_string(),
        Token::Minus => "-".to_string(),
        Token::Slash => "/".to_string(),
        Token::Percent => "%".to_string(),
        Token::Eq => "=".to_string(),
        Token::Ne => "!=".to_string(),
        Token::Lt => "<".to_string(),
        Token::Le => "<=".to_string(),
        Token::Gt => ">".to_string(),
        Token::Ge => ">=".to_string(),
        Token::LParen => "(".to_string(),
        Token::RParen => ")".to_string(),
        Token::LBracket => "[".to_string(),
        Token::RBracket => "]".to_string(),
        Token::Comma => ",".to_string(),
        Token::Dot => ".".to_string(),
        Token::Semicolon => ";".to_string(),
        Token::Concat => "||".to_string(),
        Token::Ampersand => "&".to_string(),
        Token::Pipe => "|".to_string(),
        Token::Caret => "^".to_string(),
        Token::Tilde => "~".to_string(),
        Token::Arrow => "->".to_string(),
        Token::DoubleArrow => "->>".to_string(),
        Token::HashArrow => "#>".to_string(),
        Token::HashDoubleArrow => "#>>".to_string(),
        Token::AtGt => "@>".to_string(),
        Token::LtAt => "<@".to_string(),
        Token::Question => "?".to_string(),
        Token::VectorL2 => "<->".to_string(),
        Token::VectorIP => "<#>".to_string(),
        Token::VectorCosine => "<=>".to_string(),
        Token::Colon => ":".to_string(),
        Token::Parameter(n) => format!("${}", n),
        Token::NamedParam(s) => format!(":{}", s),
        Token::Eof => String::new(),
    }
}

/// SQL statement types
#[derive(Debug, Clone, PartialEq)]
pub enum SqlStatement {
    Select(SqlQuery),
    Insert(SqlInsert),
    Update(SqlUpdate),
    Delete(SqlDelete),
    CreateTable(SqlCreateTable),
    DropTable {
        name: String,
        if_exists: bool,
    },
    CreateIndex(SqlCreateIndex),
    DropIndex(String),
    AlterTable(SqlAlterTable),
    CreateView(SqlCreateView),
    DropView {
        name: String,
        if_exists: bool,
    },
    TruncateTable(String),
    CreateTableAs {
        name: String,
        query: Box<SqlQuery>,
        if_not_exists: bool,
    },
    CreateFulltextIndex {
        name: String,
        table: String,
        columns: Vec<String>,
        options: std::collections::HashMap<String, String>,
    },
    DropFulltextIndex(String),
    CreateSpatialIndex {
        name: String,
        table: String,
        column: String,
    },
    DropSpatialIndex(String),
    CreateVectorIndex {
        name: String,
        table: String,
        column: String,
        method: String,
        options: std::collections::HashMap<String, String>,
    },
    DropVectorIndex(String),
    CreateTrigger(SqlCreateTrigger),
    DropTrigger {
        name: String,
        if_exists: bool,
    },
    ShowTables,
    ShowColumns(String),
    CreateMaterializedView {
        name: String,
        query_sql: String,
        query: Box<SqlQuery>,
    },
    RefreshMaterializedView(String),
    DropMaterializedView {
        name: String,
        if_exists: bool,
    },
    Begin,
    Commit,
    Rollback,
    // Auth/RBAC statements
    Grant {
        permissions: Vec<String>,
        resource: String,
        grantee: String,
    },
    Revoke {
        permissions: Vec<String>,
        resource: String,
        grantee: String,
    },
    CreateUser {
        name: String,
        password: Option<String>,
    },
    DropUser(String),
    AlterUser {
        name: String,
        password: String,
    },
    CreateRole(String),
    DropRole(String),
    /// DEFINE REFERENCE name ON table.column REFERENCES ref_table(ref_column)
    DefineReference {
        name: String,
        table: String,
        column: String,
        ref_table: String,
        ref_column: String,
    },
    /// DEFINE BUCKET name [MAX_SIZE size_bytes]
    DefineBucket {
        name: String,
        max_size: Option<u64>,
    },
    /// DEFINE API path METHOD handler_expr
    DefineApi {
        path: String,
        method: String,
        handler_sql: String,
    },
    /// EXPLAIN [ANALYZE] statement
    Explain {
        analyze: bool,
        statement: Box<SqlStatement>,
    },
    /// SAVEPOINT name
    Savepoint(String),
    /// RELEASE SAVEPOINT name
    ReleaseSavepoint(String),
    /// ROLLBACK TO SAVEPOINT name
    RollbackToSavepoint(String),
}

/// SQL CREATE TRIGGER statement
/// Syntax: CREATE TRIGGER name BEFORE|AFTER INSERT|UPDATE|DELETE ON table
///         FOR EACH ROW EXECUTE sql_statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlCreateTrigger {
    pub name: String,
    pub timing: TriggerTiming,
    pub event: TriggerEvent,
    pub table: String,
    pub body: String,
    pub or_replace: bool,
}

/// Trigger timing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
}

/// Trigger event
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerEvent {
    Insert,
    Update,
    Delete,
}

/// SQL CREATE VIEW statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlCreateView {
    pub name: String,
    pub columns: Option<Vec<String>>,
    pub query: String,
    pub or_replace: bool,
}

/// ALTER TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlAlterTable {
    /// Table to alter
    pub table: String,
    /// Alteration to perform
    pub action: AlterTableAction,
}

/// Individual ALTER TABLE operation
#[derive(Debug, Clone, PartialEq)]
pub enum AlterTableAction {
    /// ADD COLUMN <name> <type>
    AddColumn { name: String, data_type: String },
    /// DROP COLUMN <name>
    DropColumn { name: String },
    /// RENAME COLUMN <old> TO <new>
    RenameColumn { old_name: String, new_name: String },
}

/// Common Table Expression (CTE)
#[derive(Debug, Clone, PartialEq)]
pub struct SqlCTE {
    /// CTE name
    pub name: String,
    /// Optional column names
    pub columns: Vec<String>,
    /// CTE query
    pub query: Box<SqlQuery>,
    /// Whether this CTE is recursive
    pub recursive: bool,
}

/// Type of set operation (UNION, EXCEPT, INTERSECT)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SetOperationType {
    Union,
    UnionAll,
    Except,
    ExceptAll,
    Intersect,
    IntersectAll,
}

/// A set operation combining two queries
#[derive(Debug, Clone, PartialEq)]
pub struct SetOp {
    pub op_type: SetOperationType,
    pub query: Box<SqlQuery>,
}

/// SQL SELECT query
#[derive(Debug, Clone, PartialEq)]
pub struct SqlQuery {
    /// Common Table Expressions (WITH clause)
    pub ctes: Vec<SqlCTE>,
    pub distinct: bool,
    pub columns: Vec<SqlColumn>,
    pub from: Option<SqlFrom>,
    pub joins: Vec<SqlJoin>,
    pub where_clause: Option<Expression>,
    pub group_by: Vec<Expression>,
    pub having: Option<Expression>,
    pub order_by: Vec<OrderBy>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub set_op: Option<SetOp>,
    /// Time-travel: SELECT ... AS OF TIMESTAMP <epoch_ms>
    pub as_of_timestamp: Option<u64>,
}

impl SqlQuery {
    /// Convert to generic Query
    pub fn to_query(&self) -> Query {
        let mut columns = Vec::new();
        let mut derived_columns = HashMap::new();

        for (i, col) in self.columns.iter().enumerate() {
            match &col.expr {
                Expression::Column(name) if col.alias.is_none() => {
                    columns.push(name.clone());
                }
                Expression::QualifiedColumn { table: _, column } if col.alias.is_none() => {
                    // For now, simplify qualified columns to just column name if no conflict
                    // In a full implementation, we'd handle fully qualified names
                    columns.push(column.clone());
                }
                Expression::Wildcard => {
                    columns.push("*".to_string());
                }
                _ => {
                    // Complex expression
                    let alias = col.alias.clone().unwrap_or_else(|| format!("col_{}", i));
                    columns.push(alias.clone());
                    derived_columns.insert(alias, col.expr.clone());
                }
            }
        }

        Query {
            query_type: QueryType::Select,
            source: self
                .from
                .as_ref()
                .and_then(|f| f.table_name().map(|s| s.to_string())),
            columns,
            filter: self.where_clause.clone(),
            order_by: self.order_by.clone(),
            group_by: self.group_by.clone(),
            having: self.having.clone(),
            limit: self.limit,
            offset: self.offset,
            joins: self.joins.iter().map(|j| j.to_join()).collect(),
            values: Vec::new(),
            returning: Vec::new(),
            ctes: self.ctes.iter().map(|c| c.to_cte()).collect(),
            derived_columns,
            distinct: self.distinct,
            source_alias: self.from.as_ref().and_then(|f| f.alias.clone()),
        }
    }
}

impl SqlCTE {
    pub fn to_cte(&self) -> Cte {
        Cte {
            name: self.name.clone(),
            columns: self.columns.clone(),
            query: Box::new(self.query.to_query()),
            recursive: self.recursive,
        }
    }
}

/// SQL column reference
#[derive(Debug, Clone, PartialEq)]
pub struct SqlColumn {
    pub expr: Expression,
    pub alias: Option<String>,
}

impl SqlColumn {
    pub fn to_string(&self) -> String {
        if let Some(alias) = &self.alias {
            alias.clone()
        } else {
            match &self.expr {
                Expression::Column(name) => name.clone(),
                Expression::Wildcard => "*".to_string(),
                _ => "expr".to_string(),
            }
        }
    }
}

/// Source for FROM clause - either a table name or a subquery
#[derive(Debug, Clone, PartialEq)]
pub enum FromSource {
    Table(String),
    Subquery(Box<SqlQuery>),
}

/// SQL FROM clause
#[derive(Debug, Clone, PartialEq)]
pub struct SqlFrom {
    pub source: FromSource,
    pub alias: Option<String>,
    /// Whether LATERAL was specified (allows subquery to reference earlier FROM items).
    pub lateral: bool,
}

impl SqlFrom {
    /// Get table name if this is a simple table reference
    pub fn table_name(&self) -> Option<&str> {
        match &self.source {
            FromSource::Table(name) => Some(name),
            FromSource::Subquery(_) => None,
        }
    }

    /// Check if this FROM clause is a subquery
    pub fn is_subquery(&self) -> bool {
        matches!(self.source, FromSource::Subquery(_))
    }

    /// Get the alias or table name
    pub fn effective_name(&self) -> &str {
        self.alias.as_deref().unwrap_or_else(|| match &self.source {
            FromSource::Table(name) => name,
            FromSource::Subquery(_) => "subquery",
        })
    }
}

/// SQL JOIN clause
#[derive(Debug, Clone, PartialEq)]
pub struct SqlJoin {
    pub join_type: JoinType,
    pub table: String,
    pub alias: Option<String>,
    pub condition: Option<Expression>,
    /// Columns for USING clause (e.g., JOIN t2 USING (id, name))
    pub using_columns: Vec<String>,
    /// Derived table: JOIN (SELECT ...) alias ON ...
    pub source_subquery: Option<Box<SqlQuery>>,
    /// Whether LATERAL was specified on this join source.
    pub lateral: bool,
}

impl SqlJoin {
    fn to_join(&self) -> Join {
        Join {
            join_type: self.join_type,
            table: self.table.clone(),
            alias: self.alias.clone(),
            condition: self.condition.clone(),
            using_columns: self.using_columns.clone(),
        }
    }
}

/// Source for INSERT statement data
#[derive(Debug, Clone, PartialEq)]
pub enum InsertSource {
    /// INSERT ... VALUES (expr, ...), (expr, ...)
    Values(Vec<Vec<Expression>>),
    /// INSERT ... SELECT ...
    Select(Box<SqlQuery>),
}

/// ON CONFLICT action for UPSERT
#[derive(Debug, Clone, PartialEq)]
pub enum OnConflictAction {
    /// DO NOTHING — skip conflicting rows
    DoNothing,
    /// DO UPDATE SET col = expr, ... — update conflicting rows
    DoUpdate(Vec<(String, Expression)>),
}

/// ON CONFLICT clause
#[derive(Debug, Clone, PartialEq)]
pub struct OnConflict {
    /// Conflict target columns (optional)
    pub columns: Vec<String>,
    /// Action to take on conflict
    pub action: OnConflictAction,
}

/// SQL INSERT statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlInsert {
    pub table: String,
    pub columns: Vec<String>,
    pub source: InsertSource,
    pub returning: Vec<String>,
    pub on_conflict: Option<OnConflict>,
}

/// SQL UPDATE statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlUpdate {
    pub table: String,
    pub assignments: Vec<(String, Expression)>,
    pub where_clause: Option<Expression>,
    pub returning: Vec<String>,
}

/// SQL DELETE statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlDelete {
    pub table: String,
    pub where_clause: Option<Expression>,
    pub returning: Vec<String>,
}

/// SQL CREATE TABLE statement
#[derive(Debug, Clone, PartialEq)]
pub struct SqlCreateTable {
    pub name: String,
    pub columns: Vec<SqlColumnDef>,
    pub if_not_exists: bool,
    /// Column families for wide-column storage
    pub column_families: Vec<String>,
    /// Shard key column (SHARD BY (col))
    pub shard_key: Option<String>,
}

/// Foreign key referential action
#[derive(Debug, Clone, PartialEq)]
pub enum ReferentialAction {
    Cascade,
    Restrict,
    SetNull,
    NoAction,
}

/// Foreign key definition
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignKeyDef {
    pub ref_table: String,
    pub ref_column: String,
    pub on_delete: Option<ReferentialAction>,
    pub on_update: Option<ReferentialAction>,
}

/// SQL column definition
#[derive(Debug, Clone, PartialEq)]
pub struct SqlColumnDef {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub default: Option<Expression>,
    pub check: Option<Expression>,
    pub auto_increment: bool,
    pub foreign_key: Option<ForeignKeyDef>,
    /// Column family assignment (for wide-column storage)
    pub column_family: Option<String>,
    /// Computed (virtual) column expression, evaluated at query time
    pub computed: Option<Expression>,
}

/// Index column — either a simple name or an expression (expression index).
#[derive(Debug, Clone, PartialEq)]
pub enum IndexColumn {
    Name(String),
    Expression(Expression),
}

impl IndexColumn {
    /// Extract the column name, or format the expression as a string for storage.
    pub fn to_column_name(&self) -> String {
        match self {
            IndexColumn::Name(n) => n.clone(),
            IndexColumn::Expression(e) => format!("{e:?}"),
        }
    }
}

impl Default for IndexColumn {
    fn default() -> Self {
        IndexColumn::Name(String::new())
    }
}

/// SQL CREATE INDEX statement
///
/// Supports `CREATE INDEX ... USING method WITH (key=val, ...)` syntax
/// for all index types (B-tree, HNSW, IVF, LSH, GIN, GIST).
#[derive(Debug, Clone, PartialEq)]
pub struct SqlCreateIndex {
    pub name: String,
    pub table: String,
    pub columns: Vec<IndexColumn>,
    pub unique: bool,
    pub if_not_exists: bool,
    /// Index method (e.g., "BTREE", "HNSW", "IVF", "LSH", "GIN", "GIST").
    /// None defaults to B-tree.
    pub method: Option<String>,
    /// Key-value options (e.g., m=16, ef_construction=200, metric=cosine).
    pub options: std::collections::HashMap<String, String>,
    /// INCLUDE columns for covering indexes.
    pub include_columns: Vec<String>,
    /// Optional WHERE clause for partial indexes.
    pub where_clause: Option<Expression>,
}

/// SQL Token
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    In,
    Between,
    Like,
    ILike,
    Is,
    Null,
    As,
    Join,
    Inner,
    Left,
    Right,
    Full,
    Outer,
    Cross,
    On,
    Group,
    By,
    Having,
    Order,
    Asc,
    Desc,
    Limit,
    Offset,
    Insert,
    Into,
    Values,
    Update,
    Set,
    Delete,
    Create,
    Drop,
    Table,
    Index,
    If,
    Exists,
    Unique,
    Primary,
    Key,
    Default,
    Returning,
    Begin,
    Commit,
    Rollback,
    Distinct,
    All,
    Union,
    Except,
    Intersect,
    Case,
    When,
    Then,
    Else,
    End,
    True,
    False,
    With,
    Recursive,
    Date,
    Interval,
    Extract,
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    Over,
    Partition,
    Window,
    Rows,
    Row,
    Range,
    Groups,
    Current,
    Preceding,
    Following,
    Unbounded,

    // ALTER TABLE keywords
    Alter,
    Add,
    Column,
    Rename,
    To,

    // Views
    View,
    Replace,

    // TRUNCATE
    Truncate,

    // CHECK constraint
    Check,

    // FOREIGN KEY
    Foreign,
    References,
    Cascade,
    Restrict,
    // NATURAL JOIN / USING
    Natural,
    Using,

    // SHOW commands
    Show,

    // UPSERT (INSERT ON CONFLICT)
    Conflict,
    Do,
    Nothing,

    // Type casting
    Cast, // CAST

    // Full-text search
    Fulltext, // FULLTEXT
    Match,    // MATCH
    Against,  // AGAINST

    // Spatial
    Spatial,
    Vector,

    // Triggers
    Trigger,
    Before,
    After,
    Each,
    Execute,

    // Auth/RBAC
    Grant,    // GRANT
    Revoke,   // REVOKE
    User,     // USER
    Role,     // ROLE
    Password, // PASSWORD

    // Sharding
    Shard, // SHARD

    // Materialized views
    Materialized, // MATERIALIZED
    Refresh,      // REFRESH

    // Wide-column
    Family, // FAMILY (for COLUMN FAMILY / COLUMN_FAMILIES)

    // Schema-level keywords
    Computed,  // COMPUTED (virtual column)
    Define,    // DEFINE (schema DDL)
    Reference, // REFERENCE (bidirectional ref)

    // Semantic query extensions (JouleDB specific)
    Similar,   // SIMILAR TO
    Meaning,   // LIKE MEANING
    Threshold, // THRESHOLD
    Nearest,   // NEAREST TO

    // EXPLAIN
    Explain, // EXPLAIN
    Analyze, // ANALYZE

    // Savepoints
    Savepoint, // SAVEPOINT
    Release,   // RELEASE

    // LATERAL
    Lateral, // LATERAL

    // SQL:2008 FETCH FIRST
    Fetch, // FETCH

    // Index extensions
    Include, // INCLUDE (covering indexes)

    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Identifier(String),
    QuotedIdentifier(String),

    // Operators
    Eq,              // =
    Ne,              // <> or !=
    Lt,              // <
    Le,              // <=
    Gt,              // >
    Ge,              // >=
    Plus,            // +
    Minus,           // -
    Star,            // *
    Slash,           // /
    Percent,         // %
    Concat,          // ||
    Ampersand,       // &
    Pipe,            // |
    Caret,           // ^
    Tilde,           // ~
    Arrow,           // ->
    DoubleArrow,     // ->>
    TildeArrow,      // ~> (reverse reference traversal)
    HashArrow,       // #>
    HashDoubleArrow, // #>>
    AtGt,            // @>
    LtAt,            // <@
    Question,        // ?
    VectorL2,        // <->
    VectorIP,        // <#>
    VectorCosine,    // <=>

    // Punctuation
    LParen,    // (
    RParen,    // )
    LBracket,  // [
    RBracket,  // ]
    Comma,     // ,
    Semicolon, // ;
    Dot,       // .
    Colon,     // :

    // Special
    Parameter(usize),   // $1, $2, etc
    NamedParam(String), // :name

    Eof,
}

/// SQL Lexer
pub struct SqlLexer<'a> {
    chars: Peekable<Chars<'a>>,
    line: usize,
    column: usize,
}

impl<'a> SqlLexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            chars: input.chars().peekable(),
            line: 1,
            column: 1,
        }
    }

    fn peek(&mut self) -> Option<&char> {
        self.chars.peek()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next();
        if let Some(ch) = c {
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(&c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else if c == '-' {
                // Peek ahead to check for -- comment (don't consume yet)
                let mut chars_copy = self.chars.clone();
                chars_copy.next(); // skip the first '-'
                if chars_copy.peek() == Some(&'-') {
                    // It's a comment, consume and skip to end of line
                    self.advance(); // consume first '-'
                    self.advance(); // consume second '-'
                    while let Some(&c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                } else {
                    // Not a comment, leave the '-' for the lexer
                    break;
                }
            } else {
                break;
            }
        }
    }

    pub fn next_token(&mut self) -> QueryResult<Token> {
        self.skip_whitespace();

        let c = match self.advance() {
            Some(c) => c,
            None => return Ok(Token::Eof),
        };

        match c {
            '(' => Ok(Token::LParen),
            ')' => Ok(Token::RParen),
            '[' => Ok(Token::LBracket),
            ']' => Ok(Token::RBracket),
            ',' => Ok(Token::Comma),
            ';' => Ok(Token::Semicolon),
            '.' => Ok(Token::Dot),
            ':' => {
                if let Some(&c) = self.peek() {
                    if c.is_alphabetic() {
                        let name = self.read_identifier();
                        return Ok(Token::NamedParam(name));
                    }
                }
                Ok(Token::Colon)
            }
            '+' => Ok(Token::Plus),
            '-' => {
                if self.peek() == Some(&'>') {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::DoubleArrow)
                    } else {
                        Ok(Token::Arrow)
                    }
                } else {
                    Ok(Token::Minus)
                }
            }
            '*' => Ok(Token::Star),
            '/' => Ok(Token::Slash),
            '%' => Ok(Token::Percent),
            '=' => Ok(Token::Eq),
            '<' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::VectorCosine) // <=>
                    } else {
                        Ok(Token::Le)
                    }
                } else if self.peek() == Some(&'-') {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::VectorL2) // <->
                    } else {
                        // Put back: we consumed '<' and '-', but '-' is not '>'
                        // We can't un-consume, so treat '<' as Lt and we'll
                        // need to handle the '-' on the next call.
                        // Actually, since we already consumed '-', we need to
                        // return Lt and then the '-' is lost. Instead, store it.
                        // For a simpler approach: this is an error in SQL context.
                        Err(QueryError::SyntaxError {
                            message: "Expected '>' after '<-'".to_string(),
                            line: self.line,
                            column: self.column,
                        })
                    }
                } else if self.peek() == Some(&'#') {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::VectorIP) // <#>
                    } else {
                        Err(QueryError::SyntaxError {
                            message: "Expected '>' after '<#'".to_string(),
                            line: self.line,
                            column: self.column,
                        })
                    }
                } else if self.peek() == Some(&'>') {
                    self.advance();
                    Ok(Token::Ne)
                } else if self.peek() == Some(&'@') {
                    self.advance();
                    Ok(Token::LtAt)
                } else {
                    Ok(Token::Lt)
                }
            }
            '>' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            '!' => {
                if self.peek() == Some(&'=') {
                    self.advance();
                    Ok(Token::Ne)
                } else {
                    Err(QueryError::SyntaxError {
                        message: "Expected '=' after '!'".to_string(),
                        line: self.line,
                        column: self.column,
                    })
                }
            }
            '|' => {
                if self.peek() == Some(&'|') {
                    self.advance();
                    Ok(Token::Concat)
                } else {
                    Ok(Token::Pipe)
                }
            }
            '&' => Ok(Token::Ampersand),
            '^' => Ok(Token::Caret),
            '~' => {
                if self.peek() == Some(&'>') {
                    self.advance();
                    Ok(Token::TildeArrow)
                } else {
                    Ok(Token::Tilde)
                }
            }
            '@' => {
                if self.peek() == Some(&'>') {
                    self.advance();
                    Ok(Token::AtGt)
                } else {
                    Err(QueryError::SyntaxError {
                        message: "Unexpected character: @".to_string(),
                        line: self.line,
                        column: self.column,
                    })
                }
            }
            '#' => {
                if self.peek() == Some(&'>') {
                    self.advance();
                    if self.peek() == Some(&'>') {
                        self.advance();
                        Ok(Token::HashDoubleArrow)
                    } else {
                        Ok(Token::HashArrow)
                    }
                } else {
                    Err(QueryError::SyntaxError {
                        message: "Unexpected character: #".to_string(),
                        line: self.line,
                        column: self.column,
                    })
                }
            }
            '?' => Ok(Token::Question),
            '$' => {
                let num = self.read_number();
                Ok(Token::Parameter(num as usize))
            }
            '\'' => self.read_string(),
            '"' => self.read_quoted_identifier(),
            _ if c.is_ascii_digit() => self.read_numeric(c),
            _ if c.is_alphabetic() || c == '_' => self.read_keyword_or_identifier(c),
            _ => Err(QueryError::SyntaxError {
                message: format!("Unexpected character: {}", c),
                line: self.line,
                column: self.column,
            }),
        }
    }

    fn read_string(&mut self) -> QueryResult<Token> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('\'') => {
                    if self.peek() == Some(&'\'') {
                        self.advance();
                        s.push('\'');
                    } else {
                        break;
                    }
                }
                Some(c) => s.push(c),
                None => {
                    return Err(QueryError::SyntaxError {
                        message: "Unterminated string".to_string(),
                        line: self.line,
                        column: self.column,
                    });
                }
            }
        }
        Ok(Token::String(s))
    }

    fn read_quoted_identifier(&mut self) -> QueryResult<Token> {
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => {
                    if self.peek() == Some(&'"') {
                        self.advance();
                        s.push('"');
                    } else {
                        break;
                    }
                }
                Some(c) => s.push(c),
                None => {
                    return Err(QueryError::SyntaxError {
                        message: "Unterminated identifier".to_string(),
                        line: self.line,
                        column: self.column,
                    });
                }
            }
        }
        Ok(Token::QuotedIdentifier(s))
    }

    fn read_numeric(&mut self, first: char) -> QueryResult<Token> {
        let mut s = String::from(first);
        let mut is_float = false;

        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else if c == '.' && !is_float {
                is_float = true;
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            s.parse::<f64>()
                .map(Token::Float)
                .map_err(|_| QueryError::ParseError(format!("Invalid float: {}", s)))
        } else {
            s.parse::<i64>()
                .map(Token::Integer)
                .map_err(|_| QueryError::ParseError(format!("Invalid integer: {}", s)))
        }
    }

    fn read_number(&mut self) -> i64 {
        let mut s = String::new();
        while let Some(&c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        s.parse().unwrap_or(0)
    }

    fn read_identifier(&mut self) -> String {
        let mut s = String::new();
        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        s
    }

    fn read_keyword_or_identifier(&mut self, first: char) -> QueryResult<Token> {
        let mut s = String::from(first);
        while let Some(&c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        Ok(match s.to_uppercase().as_str() {
            "SELECT" => Token::Select,
            "FROM" => Token::From,
            "WHERE" => Token::Where,
            "AND" => Token::And,
            "OR" => Token::Or,
            "NOT" => Token::Not,
            "IN" => Token::In,
            "BETWEEN" => Token::Between,
            "LIKE" => Token::Like,
            "ILIKE" => Token::ILike,
            "IS" => Token::Is,
            "NULL" => Token::Null,
            "AS" => Token::As,
            "JOIN" => Token::Join,
            "INNER" => Token::Inner,
            "LEFT" => Token::Left,
            "RIGHT" => Token::Right,
            "FULL" => Token::Full,
            "OUTER" => Token::Outer,
            "CROSS" => Token::Cross,
            "ON" => Token::On,
            "GROUP" => Token::Group,
            "BY" => Token::By,
            "HAVING" => Token::Having,
            "ORDER" => Token::Order,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "LIMIT" => Token::Limit,
            "OFFSET" => Token::Offset,
            "INSERT" => Token::Insert,
            "INTO" => Token::Into,
            "VALUES" => Token::Values,
            "UPDATE" => Token::Update,
            "SET" => Token::Set,
            "DELETE" => Token::Delete,
            "CREATE" => Token::Create,
            "DROP" => Token::Drop,
            "TABLE" => Token::Table,
            "INDEX" => Token::Index,
            "IF" => Token::If,
            "EXISTS" => Token::Exists,
            "UNIQUE" => Token::Unique,
            "PRIMARY" => Token::Primary,
            "KEY" => Token::Key,
            "DEFAULT" => Token::Default,
            "RETURNING" => Token::Returning,
            "BEGIN" => Token::Begin,
            "COMMIT" => Token::Commit,
            "ROLLBACK" => Token::Rollback,
            "ALTER" => Token::Alter,
            "ADD" => Token::Add,
            "COLUMN" => Token::Column,
            "RENAME" => Token::Rename,
            "TO" => Token::To,
            "DISTINCT" => Token::Distinct,
            "ALL" => Token::All,
            "UNION" => Token::Union,
            "EXCEPT" => Token::Except,
            "INTERSECT" => Token::Intersect,
            "CASE" => Token::Case,
            "WHEN" => Token::When,
            "THEN" => Token::Then,
            "ELSE" => Token::Else,
            "END" => Token::End,
            "TRUE" => Token::True,
            "FALSE" => Token::False,
            "WITH" => Token::With,
            "RECURSIVE" => Token::Recursive,
            "DATE" => Token::Date,
            "INTERVAL" => Token::Interval,
            "EXTRACT" => Token::Extract,
            "YEAR" => Token::Year,
            "MONTH" => Token::Month,
            "DAY" => Token::Day,
            "HOUR" => Token::Hour,
            "MINUTE" => Token::Minute,
            "SECOND" => Token::Second,
            "OVER" => Token::Over,
            "PARTITION" => Token::Partition,
            "WINDOW" => Token::Window,
            "ROWS" => Token::Rows,
            "ROW" => Token::Row,
            "RANGE" => Token::Range,
            "GROUPS" => Token::Groups,
            "CURRENT" => Token::Current,
            "PRECEDING" => Token::Preceding,
            "FOLLOWING" => Token::Following,
            "UNBOUNDED" => Token::Unbounded,
            "CAST" => Token::Cast,
            "VIEW" => Token::View,
            "REPLACE" => Token::Replace,
            "TRUNCATE" => Token::Truncate,
            "CHECK" => Token::Check,
            "FOREIGN" => Token::Foreign,
            "REFERENCES" => Token::References,
            "CASCADE" => Token::Cascade,
            "RESTRICT" => Token::Restrict,
            "NATURAL" => Token::Natural,
            "USING" => Token::Using,
            "SHOW" => Token::Show,
            "CONFLICT" => Token::Conflict,
            "DO" => Token::Do,
            "NOTHING" => Token::Nothing,
            // Full-text search
            "FULLTEXT" => Token::Fulltext,
            "MATCH" => Token::Match,
            "AGAINST" => Token::Against,
            // JouleDB semantic query extensions
            "SIMILAR" => Token::Similar,
            "MEANING" => Token::Meaning,
            "THRESHOLD" => Token::Threshold,
            "NEAREST" => Token::Nearest,
            "MATERIALIZED" => Token::Materialized,
            "REFRESH" => Token::Refresh,
            "FAMILY" => Token::Family,
            "SPATIAL" => Token::Spatial,
            "VECTOR" => Token::Vector,
            "TRIGGER" => Token::Trigger,
            "BEFORE" => Token::Before,
            "AFTER" => Token::After,
            "EACH" => Token::Each,
            "EXECUTE" => Token::Execute,
            "GRANT" => Token::Grant,
            "REVOKE" => Token::Revoke,
            "USER" => Token::User,
            "ROLE" => Token::Role,
            "PASSWORD" => Token::Password,
            "SHARD" => Token::Shard,
            "COMPUTED" => Token::Computed,
            "DEFINE" => Token::Define,
            "REFERENCE" => Token::Reference,
            "EXPLAIN" => Token::Explain,
            "ANALYZE" => Token::Analyze,
            "SAVEPOINT" => Token::Savepoint,
            "RELEASE" => Token::Release,
            "LATERAL" => Token::Lateral,
            "FETCH" => Token::Fetch,
            "INCLUDE" => Token::Include,
            _ => Token::Identifier(s),
        })
    }
}

/// SQL Parser
/// Maximum expression nesting depth to prevent stack overflow from crafted inputs.
/// The SQL parser has a 9-function recursive chain per nesting level with a
/// particularly large `parse_comparison` frame, so we use a conservative limit
/// to stay within default stack sizes in debug builds.
const MAX_EXPRESSION_DEPTH: usize = 27;

/// Maximum query length in bytes (1 MB).
const MAX_QUERY_LENGTH: usize = 1_048_576;

pub struct SqlParser {
    tokens: Vec<Token>,
    pos: usize,
    /// Current expression nesting depth (prevents stack overflow).
    expression_depth: usize,
    /// Counter for generating unique derived table aliases.
    derived_counter: usize,
}

impl SqlParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            pos: 0,
            expression_depth: 0,
            derived_counter: 0,
        }
    }

    /// Parse SQL string
    pub fn parse(&mut self, sql: &str) -> QueryResult<SqlStatement> {
        if sql.len() > MAX_QUERY_LENGTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Query too long: {} bytes exceeds maximum of {} bytes",
                sql.len(),
                MAX_QUERY_LENGTH
            )));
        }
        self.expression_depth = 0;
        // Tokenize
        let mut lexer = SqlLexer::new(sql);
        self.tokens.clear();
        loop {
            let token = lexer.next_token()?;
            if token == Token::Eof {
                self.tokens.push(token);
                break;
            }
            self.tokens.push(token);
        }
        self.pos = 0;

        self.parse_statement()
    }

    /// Parse a standalone SQL expression from a string.
    /// Used for evaluating stored computed column expressions.
    pub fn parse_expression_str(&mut self, sql: &str) -> QueryResult<Expression> {
        self.expression_depth = 0;
        let mut lexer = SqlLexer::new(sql);
        self.tokens.clear();
        loop {
            let token = lexer.next_token()?;
            if token == Token::Eof {
                self.tokens.push(token);
                break;
            }
            self.tokens.push(token);
        }
        self.pos = 0;
        self.parse_expression()
    }

    /// Parse to generic Query
    pub fn parse_query(&mut self, sql: &str) -> QueryResult<Query> {
        let stmt = self.parse(sql)?;
        match stmt {
            SqlStatement::Select(q) => Ok(q.to_query()),
            _ => Err(QueryError::Unsupported(
                "Only SELECT statements can be converted to Query".to_string(),
            )),
        }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    /// Check if peeked identifier is a SQL keyword (ON, WHERE, etc.)
    fn peek_is_keyword(&self) -> bool {
        match self.peek() {
            Token::Identifier(s) => {
                matches!(
                    s.to_uppercase().as_str(),
                    "ON" | "WHERE"
                        | "AND"
                        | "OR"
                        | "ORDER"
                        | "GROUP"
                        | "HAVING"
                        | "LIMIT"
                        | "OFFSET"
                        | "JOIN"
                        | "LEFT"
                        | "RIGHT"
                        | "INNER"
                        | "OUTER"
                        | "FULL"
                        | "CROSS"
                        | "UNION"
                        | "NATURAL"
                        | "USING"
                )
            }
            // Any non-identifier token is treated as keyword-like
            Token::On
            | Token::Where
            | Token::And
            | Token::Or
            | Token::Order
            | Token::Group
            | Token::Having
            | Token::Limit
            | Token::Join
            | Token::Left
            | Token::Right
            | Token::Inner
            | Token::Outer
            | Token::Full
            | Token::Cross
            | Token::Natural
            | Token::Using
            | Token::Union
            | Token::Except
            | Token::Intersect => true,
            _ => false,
        }
    }

    fn advance(&mut self) -> Token {
        let token = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: Token) -> QueryResult<()> {
        let token = self.advance();
        if token == expected {
            Ok(())
        } else {
            Err(QueryError::ParseError(format!(
                "Expected {:?}, got {:?}",
                expected, token
            )))
        }
    }

    fn parse_statement(&mut self) -> QueryResult<SqlStatement> {
        match self.peek() {
            Token::With => self.parse_with_query().map(SqlStatement::Select),
            Token::Select => self
                .parse_select_with_ctes(Vec::new())
                .map(SqlStatement::Select),
            Token::Values => {
                // Standalone VALUES (1,'a'), (2,'b') → synthetic SELECT
                self.advance(); // consume VALUES
                self.parse_values_as_select()
            }
            Token::Insert => self.parse_insert().map(SqlStatement::Insert),
            Token::Update => self.parse_update().map(SqlStatement::Update),
            Token::Delete => self.parse_delete().map(SqlStatement::Delete),
            Token::Create => self.parse_create(),
            Token::Drop => self.parse_drop(),
            Token::Alter => self.parse_alter(),
            Token::Truncate => self.parse_truncate(),
            Token::Refresh => {
                // REFRESH MATERIALIZED VIEW name
                self.advance(); // consume REFRESH
                self.expect(Token::Materialized)?;
                self.expect(Token::View)?;
                let name = self.parse_identifier()?;
                Ok(SqlStatement::RefreshMaterializedView(name))
            }
            Token::Show => self.parse_show(),
            Token::Begin => {
                self.advance();
                Ok(SqlStatement::Begin)
            }
            Token::Commit => {
                self.advance();
                Ok(SqlStatement::Commit)
            }
            Token::Rollback => {
                self.advance();
                // Check for ROLLBACK TO [SAVEPOINT] name
                if *self.peek() == Token::To {
                    self.advance(); // consume TO
                    if *self.peek() == Token::Savepoint {
                        self.advance(); // consume optional SAVEPOINT
                    }
                    let name = self.parse_identifier()?;
                    Ok(SqlStatement::RollbackToSavepoint(name))
                } else {
                    Ok(SqlStatement::Rollback)
                }
            }
            Token::Grant => self.parse_grant(),
            Token::Revoke => self.parse_revoke(),
            Token::Define => self.parse_define(),
            Token::Explain => {
                self.advance();
                let analyze = if *self.peek() == Token::Analyze {
                    self.advance();
                    true
                } else {
                    false
                };
                let statement = self.parse_statement()?;
                Ok(SqlStatement::Explain {
                    analyze,
                    statement: Box::new(statement),
                })
            }
            Token::Savepoint => {
                self.advance();
                let name = self.parse_identifier()?;
                Ok(SqlStatement::Savepoint(name))
            }
            Token::Release => {
                self.advance();
                // RELEASE [SAVEPOINT] name
                if *self.peek() == Token::Savepoint {
                    self.advance();
                }
                let name = self.parse_identifier()?;
                Ok(SqlStatement::ReleaseSavepoint(name))
            }
            Token::LParen => {
                // Parenthesized SELECT: (SELECT ...) UNION/EXCEPT/INTERSECT (SELECT ...)
                self.advance(); // consume (
                let lhs = self.parse_select_with_ctes(Vec::new())?;
                self.expect(Token::RParen)?;

                // Check for set operation
                let set_op = match self.peek() {
                    Token::Union => {
                        self.advance();
                        let all = *self.peek() == Token::All;
                        if all { self.advance(); }
                        // RHS may also be parenthesized
                        let rhs = if *self.peek() == Token::LParen {
                            self.advance();
                            let q = self.parse_select_with_ctes(Vec::new())?;
                            self.expect(Token::RParen)?;
                            q
                        } else {
                            self.parse_select_with_ctes(Vec::new())?
                        };
                        Some(SetOp {
                            op_type: if all { SetOperationType::UnionAll } else { SetOperationType::Union },
                            query: Box::new(rhs),
                        })
                    }
                    Token::Except => {
                        self.advance();
                        let all = *self.peek() == Token::All;
                        if all { self.advance(); }
                        let rhs = if *self.peek() == Token::LParen {
                            self.advance();
                            let q = self.parse_select_with_ctes(Vec::new())?;
                            self.expect(Token::RParen)?;
                            q
                        } else {
                            self.parse_select_with_ctes(Vec::new())?
                        };
                        Some(SetOp {
                            op_type: if all { SetOperationType::ExceptAll } else { SetOperationType::Except },
                            query: Box::new(rhs),
                        })
                    }
                    Token::Intersect => {
                        self.advance();
                        let all = *self.peek() == Token::All;
                        if all { self.advance(); }
                        let rhs = if *self.peek() == Token::LParen {
                            self.advance();
                            let q = self.parse_select_with_ctes(Vec::new())?;
                            self.expect(Token::RParen)?;
                            q
                        } else {
                            self.parse_select_with_ctes(Vec::new())?
                        };
                        Some(SetOp {
                            op_type: if all { SetOperationType::IntersectAll } else { SetOperationType::Intersect },
                            query: Box::new(rhs),
                        })
                    }
                    _ => None,
                };

                // Apply set_op if present
                let mut query = lhs;
                query.set_op = set_op;

                // Parse trailing ORDER BY / LIMIT / OFFSET that apply to combined result
                if *self.peek() == Token::Order {
                    self.advance();
                    self.expect(Token::By)?;
                    query.order_by = self.parse_order_by()?;
                }
                if *self.peek() == Token::Limit {
                    self.advance();
                    match self.advance() {
                        Token::Integer(n) => query.limit = Some(n as usize),
                        t => return Err(QueryError::ParseError(format!("Expected integer after LIMIT, got {:?}", t))),
                    }
                }
                if *self.peek() == Token::Offset {
                    self.advance();
                    match self.advance() {
                        Token::Integer(n) => query.offset = Some(n as usize),
                        t => return Err(QueryError::ParseError(format!("Expected integer after OFFSET, got {:?}", t))),
                    }
                }

                Ok(SqlStatement::Select(query))
            }
            t => Err(QueryError::ParseError(format!("Unexpected token: {:?}", t))),
        }
    }

    /// Parse a permission name — accepts identifiers AND SQL keywords like SELECT, INSERT, etc.
    fn parse_permission_name(&mut self) -> QueryResult<String> {
        let tok = self.advance().clone();
        match tok {
            Token::Identifier(s) | Token::QuotedIdentifier(s) => Ok(s),
            Token::Select => Ok("SELECT".to_string()),
            Token::Insert => Ok("INSERT".to_string()),
            Token::Update => Ok("UPDATE".to_string()),
            Token::Delete => Ok("DELETE".to_string()),
            Token::Create => Ok("CREATE".to_string()),
            Token::Drop => Ok("DROP".to_string()),
            Token::Alter => Ok("ALTER".to_string()),
            _ => {
                // Fall back to token_to_sql for other keywords
                let s = token_to_sql(&tok);
                if s.chars().all(|c| c.is_alphabetic() || c == '_') {
                    Ok(s)
                } else {
                    Err(QueryError::ParseError(format!(
                        "Expected permission name, got {:?}",
                        tok
                    )))
                }
            }
        }
    }

    /// Parse GRANT permissions ON resource TO grantee
    /// Parse DEFINE REFERENCE|BUCKET|API
    fn parse_define(&mut self) -> QueryResult<SqlStatement> {
        self.advance(); // consume DEFINE
        match self.peek() {
            Token::Reference => self.parse_define_reference(),
            Token::Identifier(s) if s.eq_ignore_ascii_case("BUCKET") => self.parse_define_bucket(),
            Token::Identifier(s) if s.eq_ignore_ascii_case("API") => self.parse_define_api(),
            t => Err(QueryError::ParseError(format!(
                "Expected REFERENCE, BUCKET, or API after DEFINE, got {:?}",
                t
            ))),
        }
    }

    /// Parse DEFINE REFERENCE name ON table.column REFERENCES ref_table(ref_column)
    fn parse_define_reference(&mut self) -> QueryResult<SqlStatement> {
        self.advance(); // consume REFERENCE
        let name = self.parse_identifier()?;

        // Expect ON
        self.expect(Token::On)?;

        // Parse table.column
        let table = self.parse_identifier()?;
        self.expect(Token::Dot)?;
        let column = self.parse_identifier()?;

        // Expect REFERENCES
        self.expect(Token::References)?;

        // Parse ref_table(ref_column)
        let ref_table = self.parse_identifier()?;
        self.expect(Token::LParen)?;
        let ref_column = self.parse_identifier()?;
        self.expect(Token::RParen)?;

        Ok(SqlStatement::DefineReference {
            name,
            table,
            column,
            ref_table,
            ref_column,
        })
    }

    /// Parse DEFINE BUCKET name [MAX_SIZE <bytes>]
    fn parse_define_bucket(&mut self) -> QueryResult<SqlStatement> {
        self.advance(); // consume BUCKET
        let name = self.parse_identifier()?;

        // Optional MAX_SIZE
        let max_size = if let Token::Identifier(ref s) = self.peek().clone() {
            if s.to_uppercase() == "MAX_SIZE" {
                self.advance(); // consume MAX_SIZE
                match self.advance() {
                    Token::Integer(n) => Some(n as u64),
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected integer after MAX_SIZE, got {:?}",
                            t
                        )));
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(SqlStatement::DefineBucket { name, max_size })
    }

    /// Parse DEFINE API '/path' METHOD 'handler_sql'
    fn parse_define_api(&mut self) -> QueryResult<SqlStatement> {
        self.advance(); // consume API
        let path = match self.advance() {
            Token::String(s) => s,
            Token::Identifier(s) => s,
            t => {
                return Err(QueryError::ParseError(format!(
                    "Expected path string after DEFINE API, got {:?}",
                    t
                )));
            }
        };

        // Expect HTTP method keyword (GET, POST, PUT, DELETE, PATCH)
        let method = self.parse_identifier()?.to_uppercase();
        if !matches!(method.as_str(), "GET" | "POST" | "PUT" | "DELETE" | "PATCH") {
            return Err(QueryError::ParseError(format!(
                "Expected HTTP method (GET/POST/PUT/DELETE/PATCH), got '{}'",
                method
            )));
        }

        // Handler SQL as a string literal
        let handler_sql = match self.advance() {
            Token::String(s) => s,
            t => {
                return Err(QueryError::ParseError(format!(
                    "Expected handler SQL string, got {:?}",
                    t
                )));
            }
        };

        Ok(SqlStatement::DefineApi {
            path,
            method,
            handler_sql,
        })
    }

    fn parse_grant(&mut self) -> QueryResult<SqlStatement> {
        self.advance(); // consume GRANT
        let mut permissions = Vec::new();
        loop {
            let perm = self.parse_permission_name()?;
            permissions.push(perm.to_uppercase());
            if self.peek() != &Token::Comma {
                break;
            }
            self.advance(); // consume comma
        }
        self.expect(Token::On)?;
        let resource = self.parse_identifier()?;
        self.expect(Token::To)?;
        let grantee = self.parse_identifier()?;
        Ok(SqlStatement::Grant {
            permissions,
            resource,
            grantee,
        })
    }

    /// Parse REVOKE permissions ON resource FROM grantee
    fn parse_revoke(&mut self) -> QueryResult<SqlStatement> {
        self.advance(); // consume REVOKE
        let mut permissions = Vec::new();
        loop {
            let perm = self.parse_permission_name()?;
            permissions.push(perm.to_uppercase());
            if self.peek() != &Token::Comma {
                break;
            }
            self.advance(); // consume comma
        }
        self.expect(Token::On)?;
        let resource = self.parse_identifier()?;
        self.expect(Token::From)?;
        let grantee = self.parse_identifier()?;
        Ok(SqlStatement::Revoke {
            permissions,
            resource,
            grantee,
        })
    }

    /// Parse WITH clause (CTEs) followed by SELECT
    fn parse_with_query(&mut self) -> QueryResult<SqlQuery> {
        self.expect(Token::With)?;

        // Check for RECURSIVE
        let is_recursive = if *self.peek() == Token::Recursive {
            self.advance();
            true
        } else {
            false
        };

        let mut ctes = Vec::new();

        loop {
            let name = self.parse_identifier()?;

            // Optional column list
            let columns = if *self.peek() == Token::LParen {
                self.advance();
                let cols = self.parse_identifier_list()?;
                self.expect(Token::RParen)?;
                cols
            } else {
                Vec::new()
            };

            self.expect(Token::As)?;
            self.expect(Token::LParen)?;

            // Parse the CTE query — allow SELECT or VALUES
            let query = if *self.peek() == Token::Values {
                self.advance(); // consume VALUES
                match self.parse_values_as_select()? {
                    SqlStatement::Select(q) => q,
                    _ => unreachable!(),
                }
            } else {
                self.expect(Token::Select)?;
                self.parse_select_body(Vec::new())?
            };

            self.expect(Token::RParen)?;

            ctes.push(SqlCTE {
                name,
                columns,
                query: Box::new(query),
                recursive: is_recursive,
            });

            // Check for more CTEs
            if *self.peek() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }

        // Now parse the main SELECT
        self.parse_select_with_ctes(ctes)
    }

    /// Parse SELECT with pre-parsed CTEs
    /// Parse standalone VALUES (1,'a'), (2,'b') → synthetic SELECT with UNION ALL
    fn parse_values_as_select(&mut self) -> QueryResult<SqlStatement> {
        // Parse all value rows
        let mut all_rows: Vec<Vec<Expression>> = Vec::new();
        loop {
            self.expect(Token::LParen)?;
            let row = self.parse_expression_list()?;
            self.expect(Token::RParen)?;
            all_rows.push(row);
            if *self.peek() != Token::Comma {
                break;
            }
            self.advance(); // consume comma
        }

        if all_rows.is_empty() {
            return Err(QueryError::ParseError(
                "VALUES requires at least one row".to_string(),
            ));
        }

        let ncols = all_rows[0].len();

        // Build the first SELECT with column aliases
        let first_row = all_rows.remove(0);
        let columns: Vec<SqlColumn> = first_row
            .into_iter()
            .enumerate()
            .map(|(i, expr)| SqlColumn {
                expr,
                alias: Some(format!("column{}", i + 1)),
            })
            .collect();

        let mut query = SqlQuery {
            ctes: Vec::new(),
            distinct: false,
            columns,
            from: None,
            joins: Vec::new(),
            where_clause: None,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
            offset: None,
            set_op: None,
            as_of_timestamp: None,
        };

        // Chain remaining rows as UNION ALL
        for row in all_rows {
            if row.len() != ncols {
                return Err(QueryError::ParseError(format!(
                    "VALUES rows must all have {} columns, got {}",
                    ncols,
                    row.len()
                )));
            }
            let rhs_columns: Vec<SqlColumn> = row
                .into_iter()
                .map(|expr| SqlColumn { expr, alias: None })
                .collect();
            let rhs = SqlQuery {
                ctes: Vec::new(),
                distinct: false,
                columns: rhs_columns,
                from: None,
                joins: Vec::new(),
                where_clause: None,
                group_by: Vec::new(),
                having: None,
                order_by: Vec::new(),
                limit: None,
                offset: None,
                set_op: None,
                as_of_timestamp: None,
            };
            // Walk to the deepest set_op to chain
            let mut target = &mut query;
            while target.set_op.is_some() {
                target = &mut target.set_op.as_mut().unwrap().query;
            }
            target.set_op = Some(SetOp {
                op_type: SetOperationType::UnionAll,
                query: Box::new(rhs),
            });
        }

        // Handle optional ORDER BY / LIMIT / OFFSET after VALUES
        if *self.peek() == Token::Order {
            self.advance();
            self.expect(Token::By)?;
            query.order_by = self.parse_order_by()?;
        }
        if *self.peek() == Token::Limit {
            self.advance();
            match self.advance() {
                Token::Integer(n) => query.limit = Some(n as usize),
                t => {
                    return Err(QueryError::ParseError(format!(
                        "Expected integer after LIMIT, got {:?}",
                        t
                    )));
                }
            }
        }
        if *self.peek() == Token::Offset {
            self.advance();
            match self.advance() {
                Token::Integer(n) => query.offset = Some(n as usize),
                t => {
                    return Err(QueryError::ParseError(format!(
                        "Expected integer after OFFSET, got {:?}",
                        t
                    )));
                }
            }
        }

        Ok(SqlStatement::Select(query))
    }

    fn parse_select_with_ctes(&mut self, ctes: Vec<SqlCTE>) -> QueryResult<SqlQuery> {
        self.expect(Token::Select)?;
        self.parse_select_body(ctes)
    }

    /// Parse the body of a SELECT (after SELECT keyword)
    fn parse_select_body(&mut self, ctes: Vec<SqlCTE>) -> QueryResult<SqlQuery> {
        let distinct = if *self.peek() == Token::Distinct {
            self.advance();
            true
        } else {
            false
        };

        let columns = self.parse_select_columns()?;

        let from = if *self.peek() == Token::From {
            self.advance();
            Some(self.parse_from()?)
        } else {
            None
        };

        // Parse optional AS OF TIMESTAMP <epoch_ms>
        let as_of_timestamp = if *self.peek() == Token::As {
            let save = self.pos;
            self.advance(); // consume AS
            if let Token::Identifier(ref s) = self.peek().clone() {
                if s.to_uppercase() == "OF" {
                    self.advance(); // consume OF
                    // Expect a TIMESTAMP-like identifier
                    if let Token::Identifier(ref ts) = self.peek().clone() {
                        if ts.to_uppercase() == "TIMESTAMP" {
                            self.advance(); // consume TIMESTAMP
                            match self.advance() {
                                Token::Integer(n) => Some(n as u64),
                                t => {
                                    return Err(QueryError::ParseError(format!(
                                        "Expected integer epoch after AS OF TIMESTAMP, got {:?}",
                                        t
                                    )));
                                }
                            }
                        } else {
                            // Not AS OF TIMESTAMP, rewind
                            self.pos = save;
                            None
                        }
                    } else {
                        self.pos = save;
                        None
                    }
                } else {
                    // Just AS (alias), rewind
                    self.pos = save;
                    None
                }
            } else {
                self.pos = save;
                None
            }
        } else {
            None
        };

        let joins = self.parse_joins()?;

        let where_clause = if *self.peek() == Token::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let group_by = if *self.peek() == Token::Group {
            self.advance();
            self.expect(Token::By)?;
            let mut exprs = vec![self.parse_expression()?];
            while *self.peek() == Token::Comma {
                self.advance();
                exprs.push(self.parse_expression()?);
            }
            exprs
        } else {
            Vec::new()
        };

        let having = if *self.peek() == Token::Having {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let order_by = if *self.peek() == Token::Order {
            self.advance();
            self.expect(Token::By)?;
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        let limit = if *self.peek() == Token::Limit {
            self.advance();
            match self.advance() {
                Token::Integer(n) => Some(n as usize),
                t => {
                    return Err(QueryError::ParseError(format!(
                        "Expected integer, got {:?}",
                        t
                    )));
                }
            }
        } else if *self.peek() == Token::Fetch {
            // SQL:2008 FETCH FIRST|NEXT N ROWS ONLY → maps to LIMIT N
            self.advance(); // consume FETCH
            // Accept FIRST or NEXT (both mean the same thing)
            match self.peek() {
                Token::Identifier(s) if s.eq_ignore_ascii_case("FIRST") || s.eq_ignore_ascii_case("NEXT") => {
                    self.advance();
                }
                _ => {} // FETCH N ROWS ONLY is also acceptable
            }
            let n = match self.advance() {
                Token::Integer(n) => n as usize,
                t => {
                    return Err(QueryError::ParseError(format!(
                        "Expected integer after FETCH FIRST/NEXT, got {:?}",
                        t
                    )));
                }
            };
            // Consume optional ROWS/ROW
            if *self.peek() == Token::Rows {
                self.advance();
            } else if let Token::Identifier(ref s) = *self.peek() {
                if s.eq_ignore_ascii_case("ROW") {
                    self.advance();
                }
            }
            // Consume optional ONLY
            if let Token::Identifier(ref s) = *self.peek() {
                if s.eq_ignore_ascii_case("ONLY") {
                    self.advance();
                }
            }
            Some(n)
        } else {
            None
        };

        let offset = if *self.peek() == Token::Offset {
            self.advance();
            match self.advance() {
                Token::Integer(n) => Some(n as usize),
                t => {
                    return Err(QueryError::ParseError(format!(
                        "Expected integer, got {:?}",
                        t
                    )));
                }
            }
        } else {
            None
        };

        // Parse UNION / UNION ALL / EXCEPT / EXCEPT ALL / INTERSECT / INTERSECT ALL
        let mut set_op = match self.peek() {
            Token::Union => {
                self.advance();
                let all = *self.peek() == Token::All;
                if all {
                    self.advance();
                }
                let rhs = self.parse_select_with_ctes(Vec::new())?;
                Some(SetOp {
                    op_type: if all {
                        SetOperationType::UnionAll
                    } else {
                        SetOperationType::Union
                    },
                    query: Box::new(rhs),
                })
            }
            Token::Except => {
                self.advance();
                let all = *self.peek() == Token::All;
                if all {
                    self.advance();
                }
                let rhs = self.parse_select_with_ctes(Vec::new())?;
                Some(SetOp {
                    op_type: if all {
                        SetOperationType::ExceptAll
                    } else {
                        SetOperationType::Except
                    },
                    query: Box::new(rhs),
                })
            }
            Token::Intersect => {
                self.advance();
                let all = *self.peek() == Token::All;
                if all {
                    self.advance();
                }
                let rhs = self.parse_select_with_ctes(Vec::new())?;
                Some(SetOp {
                    op_type: if all {
                        SetOperationType::IntersectAll
                    } else {
                        SetOperationType::Intersect
                    },
                    query: Box::new(rhs),
                })
            }
            _ => None,
        };

        // SQL standard: ORDER BY / LIMIT / OFFSET after a set operation apply to the
        // combined result, not just the RHS.  The recursive parse_select_with_ctes call
        // above may have consumed these tokens as part of the RHS query.  If the LHS
        // (this query) has no ORDER BY / LIMIT but the RHS does, lift them here.
        let (order_by, limit, offset) = if let Some(ref mut sop) = set_op {
            let rhs = &mut sop.query;
            let ob = if order_by.is_empty() && !rhs.order_by.is_empty() {
                std::mem::take(&mut rhs.order_by)
            } else {
                order_by
            };
            let lim = if limit.is_none() && rhs.limit.is_some() {
                rhs.limit.take()
            } else {
                limit
            };
            let off = if offset.is_none() && rhs.offset.is_some() {
                rhs.offset.take()
            } else {
                offset
            };
            (ob, lim, off)
        } else {
            (order_by, limit, offset)
        };

        Ok(SqlQuery {
            ctes,
            distinct,
            columns,
            from,
            joins,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
            set_op,
            as_of_timestamp,
        })
    }

    fn parse_select_columns(&mut self) -> QueryResult<Vec<SqlColumn>> {
        let mut columns = Vec::new();

        loop {
            let expr = if *self.peek() == Token::Star {
                self.advance();
                Expression::Wildcard
            } else {
                self.parse_expression()?
            };

            let alias = if *self.peek() == Token::As {
                self.advance();
                Some(self.parse_identifier()?)
            } else if let Token::Identifier(_) = self.peek() {
                // Implicit alias
                Some(self.parse_identifier()?)
            } else {
                None
            };

            columns.push(SqlColumn { expr, alias });

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(columns)
    }

    fn parse_from(&mut self) -> QueryResult<SqlFrom> {
        // Check for LATERAL prefix
        let is_lateral = if *self.peek() == Token::Lateral {
            self.advance();
            true
        } else {
            false
        };

        // Check for subquery (SELECT in parentheses)
        if *self.peek() == Token::LParen {
            self.advance(); // consume '('

            // Check if this is a subquery
            if *self.peek() == Token::Select {
                let subquery = self.parse_select_with_ctes(Vec::new())?;
                self.expect(Token::RParen)?;

                // Subqueries must have an alias
                let alias = if *self.peek() == Token::As {
                    self.advance();
                    Some(self.parse_identifier()?)
                } else if let Token::Identifier(_) = self.peek() {
                    Some(self.parse_identifier()?)
                } else {
                    self.derived_counter += 1;
                    Some(format!("derived_{}", self.derived_counter))
                };

                return Ok(SqlFrom {
                    source: FromSource::Subquery(Box::new(subquery)),
                    alias,
                    lateral: is_lateral,
                });
            } else {
                // Not a subquery, might be parenthesized table reference
                // For now, treat as error
                return Err(QueryError::ParseError(
                    "Expected SELECT in subquery".to_string(),
                ));
            }
        }

        let mut table = self.parse_identifier()?;
        // Handle dotted names like information_schema.tables
        if *self.peek() == Token::Dot {
            self.advance(); // consume '.'
            let suffix = self.parse_identifier()?;
            table = format!("{}.{}", table, suffix);
        }
        let alias = if *self.peek() == Token::As {
            self.advance();
            Some(self.parse_identifier()?)
        } else if let Token::Identifier(_) = self.peek() {
            Some(self.parse_identifier()?)
        } else {
            None
        };

        Ok(SqlFrom {
            source: FromSource::Table(table),
            alias,
            lateral: is_lateral,
        })
    }

    fn parse_joins(&mut self) -> QueryResult<Vec<SqlJoin>> {
        let mut joins = Vec::new();

        loop {
            let mut is_natural = false;
            // Check for NATURAL prefix
            if *self.peek() == Token::Natural {
                is_natural = true;
                self.advance();
            }

            let join_type = match self.peek() {
                // Handle comma-separated tables as implicit cross joins
                Token::Comma if !is_natural => {
                    self.advance();
                    JoinType::Cross
                }
                Token::Join | Token::Inner => {
                    if *self.peek() == Token::Inner {
                        self.advance();
                    }
                    self.expect(Token::Join)?;
                    JoinType::Inner
                }
                Token::Left => {
                    self.advance();
                    if *self.peek() == Token::Outer {
                        self.advance();
                    }
                    self.expect(Token::Join)?;
                    JoinType::Left
                }
                Token::Right => {
                    self.advance();
                    if *self.peek() == Token::Outer {
                        self.advance();
                    }
                    self.expect(Token::Join)?;
                    JoinType::Right
                }
                Token::Full => {
                    self.advance();
                    if *self.peek() == Token::Outer {
                        self.advance();
                    }
                    self.expect(Token::Join)?;
                    JoinType::Full
                }
                Token::Cross => {
                    self.advance();
                    self.expect(Token::Join)?;
                    JoinType::Cross
                }
                _ => break,
            };

            // Check for LATERAL prefix on join source
            let is_lateral = if *self.peek() == Token::Lateral {
                self.advance();
                true
            } else {
                false
            };

            // Check for derived table: JOIN (SELECT ...) alias ON ...
            let (table, alias, source_subquery) = if *self.peek() == Token::LParen {
                self.advance(); // consume '('
                if *self.peek() == Token::Select {
                    let subquery = self.parse_select_with_ctes(Vec::new())?;
                    self.expect(Token::RParen)?;
                    // Parse required alias
                    let alias = if *self.peek() == Token::As {
                        self.advance();
                        self.parse_identifier()?
                    } else if let Token::Identifier(_) = self.peek() {
                        self.parse_identifier()?
                    } else {
                        self.derived_counter += 1;
                        format!("derived_{}", self.derived_counter)
                    };
                    (alias.clone(), Some(alias), Some(Box::new(subquery)))
                } else {
                    return Err(QueryError::ParseError(
                        "Expected SELECT in subquery".to_string(),
                    ));
                }
            } else {
                let table = self.parse_identifier()?;
                // Parse optional alias (AS keyword is optional)
                let alias = if *self.peek() == Token::As {
                    self.advance();
                    Some(self.parse_identifier()?)
                } else if let Token::Identifier(_) = self.peek() {
                    // Allow alias without AS keyword, but only if it's not ON/USING
                    if self.peek_is_keyword() {
                        None
                    } else {
                        Some(self.parse_identifier()?)
                    }
                } else {
                    None
                };
                (table, alias, None)
            };

            let mut condition = None;
            let mut using_columns = Vec::new();

            if is_natural {
                // NATURAL JOIN: condition will be resolved at execution time
                // (auto-join on common column names)
            } else if *self.peek() == Token::On {
                self.advance();
                condition = Some(self.parse_expression()?);
            } else if *self.peek() == Token::Using {
                self.advance();
                self.expect(Token::LParen)?;
                using_columns = self.parse_identifier_list()?;
                self.expect(Token::RParen)?;
            }

            joins.push(SqlJoin {
                join_type,
                table,
                alias,
                condition,
                using_columns,
                source_subquery,
                lateral: is_lateral,
            });

            // Store natural flag: for NATURAL joins, we set a special using_columns marker
            if is_natural {
                // Mark as natural by setting a sentinel empty vec (already empty)
                // The executor checks: using_columns.is_empty() && condition.is_none() && join_type != Cross
                // → that means NATURAL join. Let's use a more explicit approach.
                // We'll repurpose the first element to signal NATURAL.
                joins.last_mut().unwrap().using_columns = vec!["*".to_string()];
            }
        }

        Ok(joins)
    }

    fn parse_order_by(&mut self) -> QueryResult<Vec<OrderBy>> {
        let mut orders = Vec::new();

        loop {
            let expr = self.parse_expression()?;
            let descending = if *self.peek() == Token::Desc {
                self.advance();
                true
            } else {
                if *self.peek() == Token::Asc {
                    self.advance();
                }
                false
            };

            // Parse optional NULLS FIRST / NULLS LAST
            let nulls_first = if *self.peek() == Token::Identifier("".to_string())
                || matches!(self.peek(), Token::Identifier(s) if s.to_uppercase() == "NULLS")
            {
                if let Token::Identifier(s) = self.peek().clone() {
                    if s.to_uppercase() == "NULLS" {
                        self.advance();
                        if let Token::Identifier(s2) = self.peek().clone() {
                            match s2.to_uppercase().as_str() {
                                "FIRST" => {
                                    self.advance();
                                    Some(true)
                                }
                                "LAST" => {
                                    self.advance();
                                    Some(false)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            orders.push(OrderBy {
                expr,
                descending,
                nulls_first,
            });

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        Ok(orders)
    }

    fn parse_returning_list(&mut self) -> QueryResult<Vec<String>> {
        if *self.peek() == Token::Star {
            self.advance();
            return Ok(vec!["*".to_string()]);
        }
        self.parse_identifier_list()
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
            Token::QuotedIdentifier(s) => Ok(s),
            // Allow date/time keywords as identifiers (they're commonly used as column aliases)
            Token::Year => Ok("year".to_string()),
            Token::Month => Ok("month".to_string()),
            Token::Day => Ok("day".to_string()),
            Token::Hour => Ok("hour".to_string()),
            Token::Minute => Ok("minute".to_string()),
            Token::Second => Ok("second".to_string()),
            Token::Date => Ok("date".to_string()),
            Token::Interval => Ok("interval".to_string()),
            // Allow ALTER TABLE keywords as identifiers
            Token::Column => Ok("column".to_string()),
            Token::Add => Ok("add".to_string()),
            Token::Rename => Ok("rename".to_string()),
            Token::To => Ok("to".to_string()),
            Token::Alter => Ok("alter".to_string()),
            Token::View => Ok("view".to_string()),
            Token::Replace => Ok("replace".to_string()),
            Token::Conflict => Ok("conflict".to_string()),
            Token::Do => Ok("do".to_string()),
            Token::Nothing => Ok("nothing".to_string()),
            Token::Truncate => Ok("truncate".to_string()),
            Token::Check => Ok("check".to_string()),
            Token::Foreign => Ok("foreign".to_string()),
            Token::References => Ok("references".to_string()),
            Token::Cascade => Ok("cascade".to_string()),
            Token::Restrict => Ok("restrict".to_string()),
            Token::Natural => Ok("natural".to_string()),
            Token::Using => Ok("using".to_string()),
            Token::Show => Ok("show".to_string()),
            Token::Trigger => Ok("trigger".to_string()),
            Token::Before => Ok("before".to_string()),
            Token::After => Ok("after".to_string()),
            Token::Each => Ok("each".to_string()),
            Token::Execute => Ok("execute".to_string()),
            Token::Materialized => Ok("materialized".to_string()),
            Token::Refresh => Ok("refresh".to_string()),
            Token::Family => Ok("family".to_string()),
            Token::Spatial => Ok("spatial".to_string()),
            Token::Vector => Ok("vector".to_string()),
            Token::Fulltext => Ok("fulltext".to_string()),
            Token::Match => Ok("match".to_string()),
            Token::Against => Ok("against".to_string()),
            Token::Rows => Ok("rows".to_string()),
            Token::Row => Ok("row".to_string()),
            Token::Range => Ok("range".to_string()),
            Token::Groups => Ok("groups".to_string()),
            Token::Current => Ok("current".to_string()),
            Token::Preceding => Ok("preceding".to_string()),
            Token::Following => Ok("following".to_string()),
            Token::Unbounded => Ok("unbounded".to_string()),
            Token::Role => Ok("role".to_string()),
            Token::User => Ok("user".to_string()),
            Token::Password => Ok("password".to_string()),
            Token::Grant => Ok("grant".to_string()),
            Token::Revoke => Ok("revoke".to_string()),
            Token::Shard => Ok("shard".to_string()),
            Token::Key => Ok("key".to_string()),
            Token::Threshold => Ok("threshold".to_string()),
            Token::Similar => Ok("similar".to_string()),
            Token::Meaning => Ok("meaning".to_string()),
            Token::Nearest => Ok("nearest".to_string()),
            Token::Primary => Ok("primary".to_string()),
            Token::Explain => Ok("explain".to_string()),
            Token::Analyze => Ok("analyze".to_string()),
            Token::Savepoint => Ok("savepoint".to_string()),
            Token::Release => Ok("release".to_string()),
            Token::Lateral => Ok("lateral".to_string()),
            Token::Fetch => Ok("fetch".to_string()),
            Token::Include => Ok("include".to_string()),
            t => Err(QueryError::ParseError(format!(
                "Expected identifier, got {:?}",
                t
            ))),
        }
    }

    /// Parse a SQL type name, including optional parenthesized parameters.
    ///
    /// Handles types like `INTEGER`, `VARCHAR(255)`, `VECTOR(384)`, `DECIMAL(10,2)`.
    /// Returns the type as a string, e.g. `"VECTOR(384)"` or `"VARCHAR(255)"`.
    fn parse_type_with_params(&mut self) -> QueryResult<String> {
        let base = self.parse_identifier()?;
        if *self.peek() == Token::LParen {
            self.advance(); // consume '('
            let mut params = String::new();
            params.push_str(&base.to_uppercase());
            params.push('(');
            // Read first parameter
            match self.advance() {
                Token::Integer(n) => params.push_str(&n.to_string()),
                Token::Identifier(s) => params.push_str(&s),
                other => {
                    return Err(QueryError::ParseError(format!(
                        "Expected type parameter, got {:?}",
                        other
                    )));
                }
            }
            // Optional additional parameters (e.g., DECIMAL(10,2))
            while *self.peek() == Token::Comma {
                self.advance(); // consume ','
                params.push(',');
                match self.advance() {
                    Token::Integer(n) => params.push_str(&n.to_string()),
                    Token::Identifier(s) => params.push_str(&s),
                    other => {
                        return Err(QueryError::ParseError(format!(
                            "Expected type parameter, got {:?}",
                            other
                        )));
                    }
                }
            }
            self.expect(Token::RParen)?;
            params.push(')');
            Ok(params)
        } else {
            Ok(base)
        }
    }

    /// Parse optional ON DELETE/ON UPDATE referential action.
    fn parse_referential_action(&mut self, kind: &str) -> QueryResult<Option<ReferentialAction>> {
        if *self.peek() != Token::On {
            return Ok(None);
        }
        // Peek further — is the next identifier "DELETE" or "UPDATE"?
        let saved = self.pos;
        self.advance(); // consume ON
        match self.peek() {
            Token::Delete if kind == "DELETE" => {
                self.advance();
                Ok(Some(self.parse_action_keyword()?))
            }
            Token::Update if kind == "UPDATE" => {
                self.advance();
                Ok(Some(self.parse_action_keyword()?))
            }
            _ => {
                // Not our action — rewind
                self.pos = saved;
                Ok(None)
            }
        }
    }

    /// Parse CASCADE | RESTRICT | SET NULL | NO ACTION.
    fn parse_action_keyword(&mut self) -> QueryResult<ReferentialAction> {
        match self.peek() {
            Token::Cascade => {
                self.advance();
                Ok(ReferentialAction::Cascade)
            }
            Token::Restrict => {
                self.advance();
                Ok(ReferentialAction::Restrict)
            }
            Token::Set => {
                self.advance();
                self.expect(Token::Null)?;
                Ok(ReferentialAction::SetNull)
            }
            _ => {
                // Check for "NO ACTION" as identifier
                if let Token::Identifier(s) = self.peek() {
                    if s.to_uppercase() == "NO" {
                        self.advance();
                        // Expect "ACTION" identifier
                        if let Token::Identifier(s2) = self.peek() {
                            if s2.to_uppercase() == "ACTION" {
                                self.advance();
                                return Ok(ReferentialAction::NoAction);
                            }
                        }
                    }
                }
                Ok(ReferentialAction::NoAction)
            }
        }
    }

    fn parse_expression(&mut self) -> QueryResult<Expression> {
        self.expression_depth += 1;
        if self.expression_depth > MAX_EXPRESSION_DEPTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Expression nesting too deep: exceeds maximum depth of {}",
                MAX_EXPRESSION_DEPTH
            )));
        }
        let result = self.parse_or_expression();
        self.expression_depth -= 1;
        result
    }

    fn parse_or_expression(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_and_expression()?;

        while *self.peek() == Token::Or {
            self.advance();
            let right = self.parse_and_expression()?;
            left = Expression::or(left, right);
        }

        Ok(left)
    }

    fn parse_and_expression(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_not_expression()?;

        while *self.peek() == Token::And {
            self.advance();
            let right = self.parse_not_expression()?;
            left = Expression::and(left, right);
        }

        Ok(left)
    }

    fn parse_not_expression(&mut self) -> QueryResult<Expression> {
        if *self.peek() == Token::Not {
            self.advance();
            let expr = self.parse_not_expression()?;
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
            Token::VectorL2 => Some(Operator::VectorL2Distance),
            Token::VectorIP => Some(Operator::VectorIPDistance),
            Token::VectorCosine => Some(Operator::VectorCosineDistance),
            Token::Is => {
                self.advance();
                let negated = if *self.peek() == Token::Not {
                    self.advance();
                    true
                } else {
                    false
                };
                self.expect(Token::Null)?;
                return Ok(Expression::IsNull {
                    expr: Box::new(left),
                    negated,
                });
            }
            Token::Not => {
                // Peek ahead for NOT IN, NOT LIKE, NOT BETWEEN, NOT EXISTS
                let next = self.tokens.get(self.pos + 1).cloned().unwrap_or(Token::Eof);
                match next {
                    Token::In => {
                        self.advance(); // consume NOT
                        self.advance(); // consume IN
                        self.expect(Token::LParen)?;
                        let list = if *self.peek() == Token::Select || *self.peek() == Token::With {
                            let subquery = if *self.peek() == Token::With {
                                self.parse_with_query()?
                            } else {
                                self.parse_select_with_ctes(Vec::new())?
                            };
                            vec![Expression::Subquery(Box::new(subquery.to_query()))]
                        } else {
                            self.parse_expression_list()?
                        };
                        self.expect(Token::RParen)?;
                        return Ok(Expression::In {
                            expr: Box::new(left),
                            list,
                            negated: true,
                        });
                    }
                    Token::Like => {
                        self.advance(); // consume NOT
                        self.advance(); // consume LIKE
                        let pattern = match self.advance() {
                            Token::String(s) => s,
                            t => {
                                return Err(QueryError::ParseError(format!(
                                    "Expected string pattern after NOT LIKE, got {:?}",
                                    t
                                )));
                            }
                        };
                        return Ok(Expression::Like {
                            expr: Box::new(left),
                            pattern,
                            negated: true,
                            case_insensitive: false,
                        });
                    }
                    Token::ILike => {
                        self.advance(); // consume NOT
                        self.advance(); // consume ILIKE
                        let pattern = match self.advance() {
                            Token::String(s) => s,
                            t => {
                                return Err(QueryError::ParseError(format!(
                                    "Expected string pattern after NOT ILIKE, got {:?}",
                                    t
                                )));
                            }
                        };
                        return Ok(Expression::Like {
                            expr: Box::new(left),
                            pattern,
                            negated: true,
                            case_insensitive: true,
                        });
                    }
                    Token::Between => {
                        self.advance(); // consume NOT
                        self.advance(); // consume BETWEEN
                        let low = self.parse_additive()?;
                        self.expect(Token::And)?;
                        let high = self.parse_additive()?;
                        return Ok(Expression::Between {
                            expr: Box::new(left),
                            low: Box::new(low),
                            high: Box::new(high),
                            negated: true,
                        });
                    }
                    Token::Similar => {
                        self.advance(); // consume NOT
                        self.advance(); // consume SIMILAR
                        // Expect TO keyword
                        match self.peek() {
                            Token::Identifier(s) if s.to_uppercase() == "TO" => {
                                self.advance();
                            }
                            _ => {
                                return Err(QueryError::ParseError(
                                    "Expected TO after NOT SIMILAR".to_string(),
                                ));
                            }
                        }
                        let pattern = match self.advance() {
                            Token::String(s) => s,
                            t => {
                                return Err(QueryError::ParseError(format!(
                                    "Expected string after NOT SIMILAR TO, got {:?}",
                                    t
                                )));
                            }
                        };
                        let threshold = if *self.peek() == Token::Threshold {
                            self.advance();
                            match self.advance() {
                                Token::Float(f) => Some(f),
                                Token::Integer(i) => Some(i as f64),
                                t => {
                                    return Err(QueryError::ParseError(format!(
                                        "Expected number after THRESHOLD, got {:?}",
                                        t
                                    )));
                                }
                            }
                        } else {
                            None
                        };
                        return Ok(Expression::SimilarTo {
                            expr: Box::new(left),
                            pattern,
                            threshold,
                            negated: true,
                        });
                    }
                    Token::Exists => {
                        self.advance(); // consume NOT
                        self.advance(); // consume EXISTS
                        self.expect(Token::LParen)?;
                        let subquery = if *self.peek() == Token::With {
                            self.parse_with_query()?
                        } else {
                            self.parse_select_with_ctes(Vec::new())?
                        };
                        self.expect(Token::RParen)?;
                        return Ok(Expression::Unary {
                            op: crate::ast::UnaryOperator::Not,
                            expr: Box::new(Expression::Exists(Box::new(subquery.to_query()))),
                        });
                    }
                    _ => None,
                }
            }
            Token::In => {
                self.advance();
                self.expect(Token::LParen)?;

                // Check if this is a subquery or a list of values
                let list = if *self.peek() == Token::Select || *self.peek() == Token::With {
                    // Subquery in IN clause
                    let subquery = if *self.peek() == Token::With {
                        self.parse_with_query()?
                    } else {
                        self.parse_select_with_ctes(Vec::new())?
                    };
                    vec![Expression::Subquery(Box::new(subquery.to_query()))]
                } else {
                    self.parse_expression_list()?
                };

                self.expect(Token::RParen)?;
                return Ok(Expression::In {
                    expr: Box::new(left),
                    list,
                    negated: false,
                });
            }
            Token::Like => {
                self.advance();
                // Check for LIKE MEANING (semantic search)
                if *self.peek() == Token::Meaning {
                    self.advance(); // consume MEANING
                    let concept = match self.advance() {
                        Token::String(s) => s,
                        t => {
                            return Err(QueryError::ParseError(format!(
                                "Expected string after LIKE MEANING, got {:?}",
                                t
                            )));
                        }
                    };
                    return Ok(Expression::LikeMeaning {
                        expr: Box::new(left),
                        concept,
                        negated: false,
                    });
                }
                // Standard LIKE pattern matching
                let pattern = match self.advance() {
                    Token::String(s) => s,
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected string, got {:?}",
                            t
                        )));
                    }
                };
                return Ok(Expression::Like {
                    expr: Box::new(left),
                    pattern,
                    negated: false,
                    case_insensitive: false,
                });
            }
            Token::ILike => {
                self.advance();
                let pattern = match self.advance() {
                    Token::String(s) => s,
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected string after ILIKE, got {:?}",
                            t
                        )));
                    }
                };
                return Ok(Expression::Like {
                    expr: Box::new(left),
                    pattern,
                    negated: false,
                    case_insensitive: true,
                });
            }
            Token::Similar => {
                self.advance();
                // Expect TO keyword (as identifier)
                match self.peek() {
                    Token::Identifier(s) if s.to_uppercase() == "TO" => {
                        self.advance();
                    }
                    _ => {
                        return Err(QueryError::ParseError(
                            "Expected TO after SIMILAR".to_string(),
                        ));
                    }
                }

                let pattern = match self.advance() {
                    Token::String(s) => s,
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected string after SIMILAR TO, got {:?}",
                            t
                        )));
                    }
                };

                // Check for optional THRESHOLD
                let threshold = if *self.peek() == Token::Threshold {
                    self.advance();
                    match self.advance() {
                        Token::Float(f) => Some(f),
                        Token::Integer(i) => Some(i as f64),
                        t => {
                            return Err(QueryError::ParseError(format!(
                                "Expected number after THRESHOLD, got {:?}",
                                t
                            )));
                        }
                    }
                } else {
                    None
                };

                return Ok(Expression::SimilarTo {
                    expr: Box::new(left),
                    pattern,
                    threshold,
                    negated: false,
                });
            }
            Token::Between => {
                self.advance();
                let low = self.parse_additive()?;
                self.expect(Token::And)?;
                let high = self.parse_additive()?;
                return Ok(Expression::Between {
                    expr: Box::new(left),
                    low: Box::new(low),
                    high: Box::new(high),
                    negated: false,
                });
            }
            _ => None,
        };

        if let Some(op) = op {
            self.advance();
            let right = self.parse_additive()?;
            Ok(Expression::binary(left, op, right))
        } else {
            Ok(left)
        }
    }

    fn parse_additive(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_multiplicative()?;

        loop {
            let op = match self.peek() {
                Token::Plus => Operator::Add,
                Token::Minus => Operator::Sub,
                Token::Concat => Operator::Concat,
                Token::Ampersand => Operator::BitAnd,
                Token::Pipe => Operator::BitOr,
                Token::Caret => Operator::BitXor,
                // JSON operators
                Token::Arrow => Operator::JsonArrow,
                Token::DoubleArrow => Operator::JsonDoubleArrow,
                Token::HashArrow => Operator::JsonHashArrow,
                Token::HashDoubleArrow => Operator::JsonHashDoubleArrow,
                Token::AtGt => Operator::JsonContains,
                Token::LtAt => Operator::JsonContainedBy,
                Token::Question => Operator::JsonExists,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expression::binary(left, op, right);
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match self.peek() {
                Token::Star => Operator::Mul,
                Token::Slash => Operator::Div,
                Token::Percent => Operator::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expression::binary(left, op, right);
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> QueryResult<Expression> {
        if *self.peek() == Token::Minus {
            self.advance();
            let expr = self.parse_unary()?;
            Ok(Expression::Unary {
                op: crate::ast::UnaryOperator::Neg,
                expr: Box::new(expr),
            })
        } else if *self.peek() == Token::Tilde {
            self.advance();
            let expr = self.parse_unary()?;
            Ok(Expression::Unary {
                op: crate::ast::UnaryOperator::BitNot,
                expr: Box::new(expr),
            })
        } else if *self.peek() == Token::Exists {
            self.advance();
            self.expect(Token::LParen)?;
            let subquery = if *self.peek() == Token::With {
                self.parse_with_query()?
            } else {
                self.parse_select_with_ctes(Vec::new())?
            };
            self.expect(Token::RParen)?;
            Ok(Expression::Exists(Box::new(subquery.to_query())))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> QueryResult<Expression> {
        match self.advance() {
            Token::Integer(n) => Ok(Expression::Literal(Value::Int(n))),
            Token::Float(n) => Ok(Expression::Literal(Value::Float(n))),
            Token::String(s) => Ok(Expression::Literal(Value::String(s))),
            Token::True => Ok(Expression::Literal(Value::Bool(true))),
            Token::False => Ok(Expression::Literal(Value::Bool(false))),
            Token::Null => Ok(Expression::Literal(Value::Null)),
            Token::Parameter(n) => Ok(Expression::Parameter(n)),
            Token::NamedParam(name) => Ok(Expression::NamedParameter(name)),
            Token::Identifier(name) if name.eq_ignore_ascii_case("ARRAY") && *self.peek() == Token::LBracket => {
                // ARRAY[expr, expr, ...] constructor
                self.advance(); // consume [
                let mut elements = Vec::new();
                if *self.peek() != Token::RBracket {
                    loop {
                        elements.push(self.parse_expression()?);
                        if *self.peek() != Token::Comma {
                            break;
                        }
                        self.advance(); // consume comma
                    }
                }
                self.expect(Token::RBracket)?;
                // Build as a literal array of the evaluated expressions
                Ok(Expression::Function {
                    name: "ARRAY".to_string(),
                    args: elements,
                })
            }
            Token::Identifier(name) => {
                // Check for function call
                if *self.peek() == Token::LParen {
                    self.advance();

                    // Check for DISTINCT keyword in aggregate functions
                    let has_distinct = if *self.peek() == Token::Distinct {
                        self.advance();
                        true
                    } else {
                        false
                    };

                    let args = if *self.peek() == Token::RParen {
                        Vec::new()
                    } else {
                        self.parse_expression_list()?
                    };
                    self.expect(Token::RParen)?;

                    // SQL:2003 FILTER (WHERE cond) on aggregates
                    // Rewrites AGG(x) FILTER (WHERE c) → AGG(CASE WHEN c THEN x END)
                    // Special: COUNT(*) FILTER (WHERE c) → SUM(CASE WHEN c THEN 1 ELSE 0 END)
                    let (args, name) = if let Token::Identifier(ref s) = *self.peek() {
                        if s.eq_ignore_ascii_case("FILTER") {
                            self.advance(); // consume FILTER
                            self.expect(Token::LParen)?;
                            self.expect(Token::Where)?;
                            let filter_cond = self.parse_expression()?;
                            self.expect(Token::RParen)?;
                            let is_count_star = args.is_empty()
                                || (args.len() == 1
                                    && matches!(args[0], Expression::Wildcard));
                            if is_count_star {
                                // COUNT(*) FILTER → SUM(CASE WHEN c THEN 1 ELSE 0 END)
                                let case_expr = Expression::Case {
                                    operand: None,
                                    when_clauses: vec![(
                                        filter_cond,
                                        Expression::Literal(Value::Int(1)),
                                    )],
                                    else_clause: Some(Box::new(Expression::Literal(
                                        Value::Int(0),
                                    ))),
                                };
                                (vec![case_expr], "SUM".to_string())
                            } else {
                                let filtered: Vec<Expression> = args
                                    .into_iter()
                                    .map(|arg| Expression::Case {
                                        operand: None,
                                        when_clauses: vec![(filter_cond.clone(), arg)],
                                        else_clause: None,
                                    })
                                    .collect();
                                (filtered, name)
                            }
                        } else {
                            (args, name)
                        }
                    } else {
                        (args, name)
                    };

                    // Encode DISTINCT in function name for aggregate functions
                    let func_name = if has_distinct {
                        format!("{}_DISTINCT", name.to_uppercase())
                    } else {
                        name
                    };

                    if *self.peek() == Token::Over {
                        self.advance();
                        let window = self.parse_window_spec()?;
                        Ok(Expression::WindowFunction {
                            function: func_name,
                            args,
                            window,
                        })
                    } else {
                        Ok(Expression::Function {
                            name: func_name,
                            args,
                        })
                    }
                } else if *self.peek() == Token::Dot {
                    // Qualified column or qualified wildcard (table.*)
                    self.advance();
                    if *self.peek() == Token::Star {
                        self.advance();
                        Ok(Expression::QualifiedWildcard(name))
                    } else {
                        let column = self.parse_identifier()?;
                        Ok(Expression::QualifiedColumn {
                            table: name,
                            column,
                        })
                    }
                } else {
                    Ok(Expression::Column(name))
                }
            }
            Token::LParen => {
                // Check if this is a subquery or parenthesized expression
                if *self.peek() == Token::Select || *self.peek() == Token::With {
                    // Subquery
                    let subquery = if *self.peek() == Token::With {
                        self.parse_with_query()?
                    } else {
                        self.parse_select_with_ctes(Vec::new())?
                    };
                    self.expect(Token::RParen)?;
                    Ok(Expression::Subquery(Box::new(subquery.to_query())))
                } else {
                    // Regular parenthesized expression
                    let expr = self.parse_expression()?;
                    self.expect(Token::RParen)?;
                    Ok(expr)
                }
            }
            Token::Left | Token::Right | Token::Replace if *self.peek() == Token::LParen => {
                let name = match &self.tokens[self.pos - 1] {
                    Token::Left => "LEFT".to_string(),
                    Token::Right => "RIGHT".to_string(),
                    Token::Replace => "REPLACE".to_string(),
                    _ => unreachable!(),
                };
                self.advance(); // consume LParen
                let args = if *self.peek() == Token::RParen {
                    Vec::new()
                } else {
                    self.parse_expression_list()?
                };
                self.expect(Token::RParen)?;
                Ok(Expression::Function { name, args })
            }
            Token::Match if *self.peek() == Token::LParen => {
                // MATCH(col1, col2) AGAINST('search text')
                self.advance(); // consume LParen
                let match_cols = self.parse_expression_list()?;
                self.expect(Token::RParen)?;
                self.expect(Token::Against)?;
                self.expect(Token::LParen)?;
                let search_expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                // Encode as Function: MATCH_AGAINST(search_text, col1, col2, ...)
                let mut args = vec![search_expr];
                args.extend(match_cols);
                Ok(Expression::Function {
                    name: "MATCH_AGAINST".to_string(),
                    args,
                })
            }
            Token::Star => Ok(Expression::Wildcard),
            Token::Cast => {
                // CAST(expr AS type)
                self.expect(Token::LParen)?;
                let expr = self.parse_expression()?;
                self.expect(Token::As)?;
                let target_type = self.parse_identifier()?.to_uppercase();
                self.expect(Token::RParen)?;
                Ok(Expression::Cast {
                    expr: Box::new(expr),
                    target_type,
                })
            }
            Token::Case => self.parse_case_expression(),
            Token::Date => {
                // DATE 'YYYY-MM-DD' - parse as a timestamp
                if let Token::String(date_str) = self.advance() {
                    // Parse date string to unix timestamp
                    if let Ok(parsed) = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d") {
                        let timestamp = parsed
                            .and_hms_opt(0, 0, 0)
                            .map(|dt| dt.and_utc().timestamp())
                            .unwrap_or(0);
                        Ok(Expression::Literal(Value::Timestamp(timestamp)))
                    } else {
                        Err(QueryError::ParseError(format!(
                            "Invalid date format: {}",
                            date_str
                        )))
                    }
                } else {
                    Err(QueryError::ParseError(
                        "Expected date string after DATE".to_string(),
                    ))
                }
            }
            Token::Interval => {
                // INTERVAL 'N' DAY/MONTH/YEAR - parse as seconds
                if let Token::String(interval_str) = self.advance() {
                    let value: i64 = interval_str.parse().map_err(|_| {
                        QueryError::ParseError(format!("Invalid interval value: {}", interval_str))
                    })?;

                    // Get the unit
                    let seconds = match self.advance() {
                        Token::Day => value * 86400,     // seconds in a day
                        Token::Month => value * 2592000, // ~30 days
                        Token::Year => value * 31536000, // 365 days
                        Token::Hour => value * 3600,
                        Token::Minute => value * 60,
                        Token::Second => value,
                        t => {
                            return Err(QueryError::ParseError(format!(
                                "Expected time unit after INTERVAL, got {:?}",
                                t
                            )));
                        }
                    };

                    Ok(Expression::Literal(Value::Int(seconds)))
                } else {
                    Err(QueryError::ParseError(
                        "Expected interval string after INTERVAL".to_string(),
                    ))
                }
            }
            Token::Extract => {
                // EXTRACT(YEAR FROM date_column) - parse as a function call
                self.expect(Token::LParen)?;

                // Get the field (YEAR, MONTH, DAY, etc.)
                let field = match self.advance() {
                    Token::Year => "YEAR".to_string(),
                    Token::Month => "MONTH".to_string(),
                    Token::Day => "DAY".to_string(),
                    Token::Hour => "HOUR".to_string(),
                    Token::Minute => "MINUTE".to_string(),
                    Token::Second => "SECOND".to_string(),
                    Token::Identifier(s) => s.to_uppercase(),
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected time field in EXTRACT, got {:?}",
                            t
                        )));
                    }
                };

                // Expect FROM (as keyword token, not identifier)
                match self.peek() {
                    Token::From => {
                        self.advance();
                    }
                    Token::Identifier(s) if s.to_uppercase() == "FROM" => {
                        self.advance();
                    }
                    _ => {
                        return Err(QueryError::ParseError(
                            "Expected FROM in EXTRACT".to_string(),
                        ));
                    }
                }

                // Get the date expression
                let date_expr = self.parse_expression()?;

                self.expect(Token::RParen)?;

                // Return as a function call: EXTRACT with field as first arg (as string literal)
                Ok(Expression::Function {
                    name: "EXTRACT".to_string(),
                    args: vec![Expression::Literal(Value::String(field)), date_expr],
                })
            }
            // Keywords that can be used as identifiers in expressions
            Token::Spatial => Ok(Expression::Column("spatial".to_string())),
            Token::Vector => Ok(Expression::Column("vector".to_string())),
            Token::Fulltext => Ok(Expression::Column("fulltext".to_string())),
            Token::Against => Ok(Expression::Column("against".to_string())),
            // MATCH without LPAREN is treated as an identifier
            Token::Match => Ok(Expression::Column("match".to_string())),
            Token::Foreign => Ok(Expression::Column("foreign".to_string())),
            Token::References => Ok(Expression::Column("references".to_string())),
            Token::Cascade => Ok(Expression::Column("cascade".to_string())),
            Token::Restrict => Ok(Expression::Column("restrict".to_string())),
            Token::Natural => Ok(Expression::Column("natural".to_string())),
            Token::Using => Ok(Expression::Column("using".to_string())),
            Token::Trigger => Ok(Expression::Column("trigger".to_string())),
            Token::Before => Ok(Expression::Column("before".to_string())),
            Token::After => Ok(Expression::Column("after".to_string())),
            Token::Each => Ok(Expression::Column("each".to_string())),
            Token::Execute => Ok(Expression::Column("execute".to_string())),
            Token::Materialized => Ok(Expression::Column("materialized".to_string())),
            Token::Refresh => Ok(Expression::Column("refresh".to_string())),
            Token::Family => Ok(Expression::Column("family".to_string())),
            Token::Role => Ok(Expression::Column("role".to_string())),
            Token::User => Ok(Expression::Column("user".to_string())),
            Token::Password => Ok(Expression::Column("password".to_string())),
            Token::Grant => Ok(Expression::Column("grant".to_string())),
            Token::Revoke => Ok(Expression::Column("revoke".to_string())),
            Token::Shard => Ok(Expression::Column("shard".to_string())),
            Token::Define => Ok(Expression::Column("define".to_string())),
            Token::Reference => Ok(Expression::Column("reference".to_string())),
            Token::Computed => Ok(Expression::Column("computed".to_string())),
            Token::Threshold => Ok(Expression::Column("threshold".to_string())),
            Token::Key => Ok(Expression::Column("key".to_string())),
            Token::Similar => Ok(Expression::Column("similar".to_string())),
            Token::Meaning => Ok(Expression::Column("meaning".to_string())),
            Token::Nearest => Ok(Expression::Column("nearest".to_string())),
            Token::Primary => Ok(Expression::Column("primary".to_string())),
            Token::Explain => Ok(Expression::Column("explain".to_string())),
            Token::Analyze => Ok(Expression::Column("analyze".to_string())),
            Token::Savepoint => Ok(Expression::Column("savepoint".to_string())),
            Token::Release => Ok(Expression::Column("release".to_string())),
            Token::Lateral => Ok(Expression::Column("lateral".to_string())),
            Token::Fetch => Ok(Expression::Column("fetch".to_string())),
            Token::Include => Ok(Expression::Column("include".to_string())),
            Token::Year => Ok(Expression::Column("year".to_string())),
            Token::Month => Ok(Expression::Column("month".to_string())),
            Token::Day => Ok(Expression::Column("day".to_string())),
            Token::Hour => Ok(Expression::Column("hour".to_string())),
            Token::Minute => Ok(Expression::Column("minute".to_string())),
            Token::Second => Ok(Expression::Column("second".to_string())),
            Token::TildeArrow => {
                // ~>reference_name — reverse reference traversal
                let ref_name = self.parse_identifier()?;
                Ok(Expression::ReverseReference {
                    reference_name: ref_name,
                })
            }
            t => Err(QueryError::ParseError(format!(
                "Unexpected token in expression: {:?}",
                t
            ))),
        }
    }

    fn parse_case_expression(&mut self) -> QueryResult<Expression> {
        // CASE [expr] WHEN expr THEN expr [WHEN expr THEN expr ...] [ELSE expr] END
        let operand = if *self.peek() != Token::When {
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        let mut when_clauses = Vec::new();
        while *self.peek() == Token::When {
            self.advance(); // consume WHEN
            let when_expr = self.parse_expression()?;
            self.expect(Token::Then)?;
            let then_expr = self.parse_expression()?;
            when_clauses.push((when_expr, then_expr));
        }

        let else_clause = if *self.peek() == Token::Else {
            self.advance();
            Some(Box::new(self.parse_expression()?))
        } else {
            None
        };

        self.expect(Token::End)?;

        Ok(Expression::Case {
            operand,
            when_clauses,
            else_clause,
        })
    }

    fn parse_window_spec(&mut self) -> QueryResult<WindowSpec> {
        self.expect(Token::LParen)?;

        let partition_by = if *self.peek() == Token::Partition {
            self.advance();
            self.expect(Token::By)?;
            self.parse_expression_list()?
        } else {
            Vec::new()
        };

        let order_by = if *self.peek() == Token::Order {
            self.advance();
            self.expect(Token::By)?;
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        // Frame parsing: ROWS|RANGE|GROUPS BETWEEN ... AND ... or ROWS|RANGE|GROUPS bound
        let frame = if matches!(self.peek(), Token::Rows | Token::Range | Token::Groups) {
            let units = match self.advance() {
                Token::Rows => WindowFrameUnits::Rows,
                Token::Range => WindowFrameUnits::Range,
                Token::Groups => WindowFrameUnits::Groups,
                _ => unreachable!(),
            };

            if *self.peek() == Token::Between {
                self.advance(); // consume BETWEEN
                let start_bound = self.parse_frame_bound()?;
                self.expect(Token::And)?;
                let end_bound = self.parse_frame_bound()?;
                Some(WindowFrame {
                    units,
                    start_bound,
                    end_bound,
                })
            } else {
                // Shorthand: just a start bound, end defaults to CURRENT ROW
                let start_bound = self.parse_frame_bound()?;
                Some(WindowFrame {
                    units,
                    start_bound,
                    end_bound: WindowFrameBound::CurrentRow,
                })
            }
        } else {
            None
        };

        self.expect(Token::RParen)?;

        Ok(WindowSpec {
            partition_by,
            order_by,
            frame,
        })
    }

    fn parse_frame_bound(&mut self) -> QueryResult<WindowFrameBound> {
        match self.peek().clone() {
            Token::Unbounded => {
                self.advance();
                match self.peek().clone() {
                    Token::Preceding => {
                        self.advance();
                        Ok(WindowFrameBound::UnboundedPreceding)
                    }
                    Token::Following => {
                        self.advance();
                        Ok(WindowFrameBound::UnboundedFollowing)
                    }
                    t => Err(QueryError::ParseError(format!(
                        "Expected PRECEDING or FOLLOWING after UNBOUNDED, got {:?}",
                        t
                    ))),
                }
            }
            Token::Current => {
                self.advance();
                // Expect ROW or ROWS
                match self.peek().clone() {
                    Token::Row | Token::Rows => {
                        self.advance();
                    }
                    _ => {}
                }
                Ok(WindowFrameBound::CurrentRow)
            }
            Token::Integer(n) => {
                let n = n as u64;
                self.advance();
                match self.peek().clone() {
                    Token::Preceding => {
                        self.advance();
                        Ok(WindowFrameBound::Preceding(n))
                    }
                    Token::Following => {
                        self.advance();
                        Ok(WindowFrameBound::Following(n))
                    }
                    t => Err(QueryError::ParseError(format!(
                        "Expected PRECEDING or FOLLOWING after number, got {:?}",
                        t
                    ))),
                }
            }
            t => Err(QueryError::ParseError(format!(
                "Expected frame bound (UNBOUNDED PRECEDING/FOLLOWING, CURRENT ROW, or N PRECEDING/FOLLOWING), got {:?}",
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

    fn parse_insert(&mut self) -> QueryResult<SqlInsert> {
        self.expect(Token::Insert)?;
        self.expect(Token::Into)?;

        let table = self.parse_identifier()?;

        let columns = if *self.peek() == Token::LParen {
            // Could be column list or VALUES subexpression — peek ahead
            // If next token after LParen is an identifier followed by comma/RParen, it's columns
            // If next token is SELECT, it's a subquery in parens (rare but valid)
            self.advance();
            let cols = self.parse_identifier_list()?;
            self.expect(Token::RParen)?;
            cols
        } else {
            Vec::new()
        };

        // Check for duplicate column names in INSERT column list
        if !columns.is_empty() {
            let mut seen = std::collections::HashSet::new();
            for col in &columns {
                let lower = col.to_lowercase();
                if !seen.insert(lower) {
                    return Err(QueryError::ParseError(format!(
                        "duplicate column name '{}' in INSERT column list",
                        col
                    )));
                }
            }
        }

        // Check if next token is SELECT/WITH (INSERT...SELECT) or VALUES
        let source = if *self.peek() == Token::Select || *self.peek() == Token::With {
            let select_query = if *self.peek() == Token::With {
                self.parse_with_query()?
            } else {
                self.parse_select_with_ctes(Vec::new())?
            };
            InsertSource::Select(Box::new(select_query))
        } else {
            self.expect(Token::Values)?;

            let mut values = Vec::new();
            loop {
                self.expect(Token::LParen)?;
                values.push(self.parse_expression_list()?);
                self.expect(Token::RParen)?;

                if *self.peek() != Token::Comma {
                    break;
                }
                self.advance();
            }
            InsertSource::Values(values)
        };

        // Parse ON CONFLICT clause (before or after RETURNING)
        let on_conflict = if *self.peek() == Token::On {
            self.advance(); // consume ON
            self.expect(Token::Conflict)?;

            // Optional conflict target columns
            let conflict_columns = if *self.peek() == Token::LParen {
                self.advance();
                let cols = self.parse_identifier_list()?;
                self.expect(Token::RParen)?;
                cols
            } else {
                Vec::new()
            };

            self.expect(Token::Do)?;

            let action = if *self.peek() == Token::Nothing {
                self.advance();
                OnConflictAction::DoNothing
            } else {
                // DO UPDATE SET ...
                self.expect(Token::Update)?;
                self.expect(Token::Set)?;

                let mut assignments = Vec::new();
                loop {
                    let col = self.parse_identifier()?;
                    self.expect(Token::Eq)?;
                    let expr = self.parse_expression()?;
                    assignments.push((col, expr));
                    if *self.peek() != Token::Comma {
                        break;
                    }
                    self.advance();
                }
                OnConflictAction::DoUpdate(assignments)
            };

            Some(OnConflict {
                columns: conflict_columns,
                action,
            })
        } else {
            None
        };

        let returning = if *self.peek() == Token::Returning {
            self.advance();
            self.parse_returning_list()?
        } else {
            Vec::new()
        };

        Ok(SqlInsert {
            table,
            columns,
            source,
            returning,
            on_conflict,
        })
    }

    fn parse_update(&mut self) -> QueryResult<SqlUpdate> {
        self.expect(Token::Update)?;
        let table = self.parse_identifier()?;
        self.expect(Token::Set)?;

        let mut assignments = Vec::new();
        loop {
            let column = self.parse_identifier()?;
            self.expect(Token::Eq)?;
            let value = self.parse_expression()?;
            assignments.push((column, value));

            if *self.peek() != Token::Comma {
                break;
            }
            self.advance();
        }

        // Detect UPDATE ... FROM (not yet supported — give clear error)
        if *self.peek() == Token::From {
            return Err(QueryError::ParseError(
                "UPDATE ... FROM is not yet supported. Use a correlated subquery in SET instead: \
                 UPDATE t SET col = (SELECT val FROM t2 WHERE t2.id = t.id) WHERE ..."
                    .to_string(),
            ));
        }

        let where_clause = if *self.peek() == Token::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let returning = if *self.peek() == Token::Returning {
            self.advance();
            self.parse_returning_list()?
        } else {
            Vec::new()
        };

        Ok(SqlUpdate {
            table,
            assignments,
            where_clause,
            returning,
        })
    }

    fn parse_delete(&mut self) -> QueryResult<SqlDelete> {
        self.expect(Token::Delete)?;
        self.expect(Token::From)?;
        let table = self.parse_identifier()?;

        // Detect DELETE ... USING (not yet supported — give clear error)
        if *self.peek() == Token::Using {
            return Err(QueryError::ParseError(
                "DELETE ... USING is not yet supported. Use WHERE EXISTS (SELECT ...) instead."
                    .to_string(),
            ));
        }

        let where_clause = if *self.peek() == Token::Where {
            self.advance();
            Some(self.parse_expression()?)
        } else {
            None
        };

        let returning = if *self.peek() == Token::Returning {
            self.advance();
            self.parse_returning_list()?
        } else {
            Vec::new()
        };

        Ok(SqlDelete {
            table,
            where_clause,
            returning,
        })
    }

    fn parse_create(&mut self) -> QueryResult<SqlStatement> {
        self.expect(Token::Create)?;

        match self.peek() {
            Token::Table => {
                self.advance();
                let if_not_exists = if *self.peek() == Token::If {
                    self.advance();
                    self.expect(Token::Not)?;
                    self.expect(Token::Exists)?;
                    true
                } else {
                    false
                };

                let name = self.parse_identifier()?;

                // Check for CREATE TABLE ... AS SELECT (CTAS)
                if *self.peek() == Token::As {
                    self.advance(); // consume AS
                    let query = self.parse_select_with_ctes(Vec::new())?;
                    return Ok(SqlStatement::CreateTableAs {
                        name,
                        query: Box::new(query),
                        if_not_exists,
                    });
                }

                self.expect(Token::LParen)?;

                let mut columns: Vec<SqlColumnDef> = Vec::new();
                loop {
                    // Table-level PRIMARY KEY (col1, col2, ...) constraint
                    if *self.peek() == Token::Primary {
                        if self.tokens.get(self.pos + 1) == Some(&Token::Key) {
                            self.advance(); // consume PRIMARY
                            self.advance(); // consume KEY
                            self.expect(Token::LParen)?;
                            loop {
                                let pk_col = self.parse_identifier()?;
                                for col in &mut columns {
                                    if col.name.eq_ignore_ascii_case(&pk_col) {
                                        col.primary_key = true;
                                    }
                                }
                                if *self.peek() != Token::Comma {
                                    break;
                                }
                                self.advance();
                            }
                            self.expect(Token::RParen)?;
                            if *self.peek() != Token::Comma {
                                break;
                            }
                            self.advance();
                            continue;
                        }
                    }

                    let col_name = self.parse_identifier()?;
                    let data_type = self.parse_type_with_params()?;

                    // SERIAL/BIGSERIAL implies INTEGER + auto_increment + NOT NULL
                    let auto_increment =
                        matches!(data_type.to_uppercase().as_str(), "SERIAL" | "BIGSERIAL");
                    let mut nullable = !auto_increment; // SERIAL is NOT NULL by default
                    let mut primary_key = false;
                    let mut unique = false;
                    let mut default = None;
                    let mut check = None;
                    let mut foreign_key = None;
                    let mut column_family = None;
                    let mut computed = None;

                    while matches!(
                        self.peek(),
                        Token::Not
                            | Token::Primary
                            | Token::Default
                            | Token::Unique
                            | Token::Check
                            | Token::References
                            | Token::Column
                            | Token::Family
                            | Token::Computed
                    ) {
                        match self.peek() {
                            Token::Not => {
                                self.advance();
                                self.expect(Token::Null)?;
                                nullable = false;
                            }
                            Token::Primary => {
                                self.advance();
                                self.expect(Token::Key)?;
                                primary_key = true;
                            }
                            Token::Unique => {
                                self.advance();
                                unique = true;
                            }
                            Token::Default => {
                                self.advance();
                                default = Some(self.parse_expression()?);
                            }
                            Token::Check => {
                                self.advance();
                                self.expect(Token::LParen)?;
                                check = Some(self.parse_expression()?);
                                self.expect(Token::RParen)?;
                            }
                            Token::References => {
                                self.advance();
                                let ref_table = self.parse_identifier()?;
                                self.expect(Token::LParen)?;
                                let ref_column = self.parse_identifier()?;
                                self.expect(Token::RParen)?;
                                let on_delete = self.parse_referential_action("DELETE")?;
                                let on_update = self.parse_referential_action("UPDATE")?;
                                foreign_key = Some(ForeignKeyDef {
                                    ref_table,
                                    ref_column,
                                    on_delete,
                                    on_update,
                                });
                            }
                            Token::Column => {
                                // COLUMN FAMILY cf_name
                                self.advance(); // consume COLUMN
                                self.expect(Token::Family)?;
                                column_family = Some(self.parse_identifier()?);
                            }
                            Token::Family => {
                                // FAMILY cf_name (shorthand)
                                self.advance();
                                column_family = Some(self.parse_identifier()?);
                            }
                            Token::Computed => {
                                // COMPUTED AS (expr)
                                self.advance();
                                self.expect(Token::As)?;
                                self.expect(Token::LParen)?;
                                let expr = self.parse_expression()?;
                                self.expect(Token::RParen)?;
                                // Reject conflicting constraints
                                if default.is_some() {
                                    return Err(QueryError::ParseError(
                                        "COMPUTED column cannot have a DEFAULT".to_string(),
                                    ));
                                }
                                if primary_key {
                                    return Err(QueryError::ParseError(
                                        "COMPUTED column cannot be a PRIMARY KEY".to_string(),
                                    ));
                                }
                                if auto_increment {
                                    return Err(QueryError::ParseError(
                                        "COMPUTED column cannot be AUTO_INCREMENT".to_string(),
                                    ));
                                }
                                computed = Some(expr);
                            }
                            _ => break,
                        }
                    }

                    columns.push(SqlColumnDef {
                        name: col_name,
                        data_type,
                        nullable,
                        primary_key,
                        unique,
                        default,
                        check,
                        auto_increment,
                        foreign_key,
                        column_family,
                        computed,
                    });

                    if *self.peek() != Token::Comma {
                        break;
                    }
                    self.advance();
                }

                self.expect(Token::RParen)?;

                // Parse optional WITH COLUMN_FAMILIES (cf1, cf2, ...)
                let mut column_families = Vec::new();
                if *self.peek() == Token::With {
                    let save = self.pos;
                    self.advance(); // consume WITH
                    // Check for COLUMN FAMILY / COLUMN_FAMILIES
                    let is_cf = match self.peek() {
                        Token::Column => {
                            // WITH COLUMN FAMILIES (...)
                            self.advance(); // consume COLUMN
                            if *self.peek() == Token::Family {
                                self.advance(); // consume FAMILY
                                true
                            } else {
                                false
                            }
                        }
                        Token::Identifier(s) if s.eq_ignore_ascii_case("COLUMN_FAMILIES") => {
                            self.advance();
                            true
                        }
                        _ => false,
                    };
                    if is_cf {
                        self.expect(Token::LParen)?;
                        loop {
                            let cf_name = self.parse_identifier()?;
                            column_families.push(cf_name);
                            if *self.peek() != Token::Comma {
                                break;
                            }
                            self.advance();
                        }
                        self.expect(Token::RParen)?;
                    } else {
                        self.pos = save; // restore, WITH was not for column families
                    }
                }

                // Optional SHARD BY (column)
                let shard_key = if *self.peek() == Token::Shard {
                    self.advance(); // consume SHARD
                    self.expect(Token::By)?;
                    self.expect(Token::LParen)?;
                    let col = self.parse_identifier()?;
                    self.expect(Token::RParen)?;
                    Some(col)
                } else {
                    None
                };

                Ok(SqlStatement::CreateTable(SqlCreateTable {
                    name,
                    columns,
                    if_not_exists,
                    column_families,
                    shard_key,
                }))
            }
            Token::Spatial => {
                // CREATE SPATIAL INDEX name ON table (column)
                self.advance(); // consume SPATIAL
                self.expect(Token::Index)?;
                let name = self.parse_identifier()?;
                self.expect(Token::On)?;
                let table = self.parse_identifier()?;
                self.expect(Token::LParen)?;
                let column = self.parse_identifier()?;
                self.expect(Token::RParen)?;
                Ok(SqlStatement::CreateSpatialIndex {
                    name,
                    table,
                    column,
                })
            }
            Token::Vector => {
                // CREATE VECTOR INDEX name ON table (column) [USING HNSW|LSH] [WITH (key=val, ...)]
                self.advance(); // consume VECTOR
                self.expect(Token::Index)?;
                let name = self.parse_identifier()?;
                self.expect(Token::On)?;
                let table = self.parse_identifier()?;
                self.expect(Token::LParen)?;
                let column = self.parse_identifier()?;
                self.expect(Token::RParen)?;

                // Optional USING clause
                let method = if *self.peek() == Token::Using {
                    self.advance(); // consume USING
                    self.parse_identifier()?.to_uppercase()
                } else {
                    "HNSW".to_string()
                };

                // Optional WITH (key=val, ...) clause
                let mut options = std::collections::HashMap::new();
                if *self.peek() == Token::With {
                    self.advance(); // consume WITH
                    self.expect(Token::LParen)?;
                    loop {
                        let key = self.parse_identifier()?;
                        self.expect(Token::Eq)?;
                        let val = match self.advance() {
                            Token::Integer(n) => n.to_string(),
                            Token::Float(f) => f.to_string(),
                            Token::String(s) => s.clone(),
                            Token::Identifier(s) => s.clone(),
                            other => {
                                return Err(crate::error::QueryError::ParseError(format!(
                                    "Expected value, got {:?}",
                                    other
                                )));
                            }
                        };
                        options.insert(key.to_lowercase(), val);
                        if *self.peek() != Token::Comma {
                            break;
                        }
                        self.advance(); // consume comma
                    }
                    self.expect(Token::RParen)?;
                }

                Ok(SqlStatement::CreateVectorIndex {
                    name,
                    table,
                    column,
                    method,
                    options,
                })
            }
            Token::Fulltext => {
                // CREATE FULLTEXT INDEX name ON table (col1, col2, ...) [WITH (key=val, ...)]
                self.advance(); // consume FULLTEXT
                self.expect(Token::Index)?;
                let name = self.parse_identifier()?;
                self.expect(Token::On)?;
                let table = self.parse_identifier()?;
                self.expect(Token::LParen)?;
                let columns = self.parse_identifier_list()?;
                self.expect(Token::RParen)?;

                let mut options = std::collections::HashMap::new();
                if self.peek() == &Token::With {
                    self.advance(); // consume WITH
                    self.expect(Token::LParen)?;
                    loop {
                        let key = self.parse_identifier()?;
                        self.expect(Token::Eq)?;
                        let val = match self.peek().clone() {
                            Token::String(s) => {
                                let v = s.clone();
                                self.advance();
                                v
                            }
                            Token::Integer(n) => {
                                let v = n.to_string();
                                self.advance();
                                v
                            }
                            _ => self.parse_identifier()?,
                        };
                        options.insert(key.to_lowercase(), val);
                        if self.peek() != &Token::Comma {
                            break;
                        }
                        self.advance(); // consume comma
                    }
                    self.expect(Token::RParen)?;
                }

                Ok(SqlStatement::CreateFulltextIndex {
                    name,
                    table,
                    columns,
                    options,
                })
            }
            Token::Index | Token::Unique => {
                let unique = if *self.peek() == Token::Unique {
                    self.advance();
                    true
                } else {
                    false
                };
                self.expect(Token::Index)?;

                let if_not_exists = if *self.peek() == Token::If {
                    self.advance();
                    self.expect(Token::Not)?;
                    self.expect(Token::Exists)?;
                    true
                } else {
                    false
                };

                let name = self.parse_identifier()?;
                self.expect(Token::On)?;
                let table = self.parse_identifier()?;
                self.expect(Token::LParen)?;

                // Parse index columns (names or expressions)
                let mut columns = Vec::new();
                loop {
                    let col = if *self.peek() == Token::LParen {
                        // Expression index: CREATE INDEX ON t ((col1 + col2))
                        self.advance(); // consume (
                        let expr = self.parse_expression()?;
                        self.expect(Token::RParen)?;
                        IndexColumn::Expression(expr)
                    } else {
                        IndexColumn::Name(self.parse_identifier()?)
                    };
                    columns.push(col);
                    if *self.peek() != Token::Comma {
                        break;
                    }
                    self.advance(); // consume comma
                }
                self.expect(Token::RParen)?;

                // Optional USING clause: USING BTREE | HNSW | IVF | LSH | GIN | GIST
                let method = if *self.peek() == Token::Using {
                    self.advance(); // consume USING
                    Some(self.parse_identifier()?.to_uppercase())
                } else {
                    None
                };

                // Optional WITH (key=val, ...) clause
                let mut options = std::collections::HashMap::new();
                if *self.peek() == Token::With {
                    self.advance(); // consume WITH
                    self.expect(Token::LParen)?;
                    loop {
                        let key = self.parse_identifier()?;
                        self.expect(Token::Eq)?;
                        let val = match self.advance() {
                            Token::Integer(n) => n.to_string(),
                            Token::Float(f) => f.to_string(),
                            Token::String(s) => s.clone(),
                            Token::Identifier(s) => s.clone(),
                            other => {
                                return Err(crate::error::QueryError::ParseError(format!(
                                    "Expected value in WITH clause, got {:?}",
                                    other
                                )));
                            }
                        };
                        options.insert(key.to_lowercase(), val);
                        if *self.peek() != Token::Comma {
                            break;
                        }
                        self.advance(); // consume comma
                    }
                    self.expect(Token::RParen)?;
                }

                // Optional INCLUDE (col1, col2) for covering indexes
                let include_columns = if *self.peek() == Token::Include {
                    self.advance();
                    self.expect(Token::LParen)?;
                    let mut cols = vec![self.parse_identifier()?];
                    while *self.peek() == Token::Comma {
                        self.advance();
                        cols.push(self.parse_identifier()?);
                    }
                    self.expect(Token::RParen)?;
                    cols
                } else {
                    vec![]
                };

                // Optional WHERE clause for partial indexes
                let where_clause = if *self.peek() == Token::Where {
                    self.advance();
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                Ok(SqlStatement::CreateIndex(SqlCreateIndex {
                    name,
                    table,
                    columns,
                    unique,
                    if_not_exists,
                    method,
                    options,
                    include_columns,
                    where_clause,
                }))
            }
            Token::Materialized => {
                // CREATE MATERIALIZED VIEW name AS SELECT ...
                self.advance(); // consume MATERIALIZED
                self.expect(Token::View)?;
                let name = self.parse_identifier()?;
                self.expect(Token::As)?;
                // Capture the remaining tokens as SQL text for storage
                let query_sql = self.remaining_tokens_as_sql();
                // Re-parse the query SQL to get the AST
                let mut inner_parser = SqlParser::new();
                let inner_stmt = inner_parser.parse(&query_sql)?;
                let query = match inner_stmt {
                    SqlStatement::Select(q) => q,
                    _ => {
                        return Err(QueryError::ParseError(
                            "Materialized view query must be a SELECT statement".to_string(),
                        ));
                    }
                };
                Ok(SqlStatement::CreateMaterializedView {
                    name,
                    query_sql,
                    query: Box::new(query),
                })
            }
            Token::View => {
                self.advance();
                self.parse_create_view(false)
            }
            Token::Trigger => {
                self.advance();
                self.parse_create_trigger(false)
            }
            Token::Or => {
                // CREATE OR REPLACE VIEW or CREATE OR REPLACE TRIGGER
                self.advance(); // consume OR
                self.expect(Token::Replace)?;
                match self.peek() {
                    Token::View => {
                        self.advance();
                        self.parse_create_view(true)
                    }
                    Token::Trigger => {
                        self.advance();
                        self.parse_create_trigger(true)
                    }
                    t => Err(QueryError::ParseError(format!(
                        "Expected VIEW or TRIGGER after OR REPLACE, got {:?}",
                        t
                    ))),
                }
            }
            Token::User => {
                self.advance(); // consume USER
                let name = self.parse_identifier()?;
                let password = if *self.peek() == Token::With {
                    self.advance(); // consume WITH
                    self.expect(Token::Password)?;
                    match self.advance() {
                        Token::String(s) => Some(s.clone()),
                        t => {
                            return Err(QueryError::ParseError(format!(
                                "Expected string for password, got {:?}",
                                t
                            )));
                        }
                    }
                } else {
                    None
                };
                Ok(SqlStatement::CreateUser { name, password })
            }
            Token::Role => {
                self.advance(); // consume ROLE
                let name = self.parse_identifier()?;
                Ok(SqlStatement::CreateRole(name))
            }
            t => Err(QueryError::ParseError(format!(
                "Expected TABLE, FULLTEXT INDEX, INDEX, VIEW, TRIGGER, USER, or ROLE, got {:?}",
                t
            ))),
        }
    }

    fn parse_create_view(&mut self, or_replace: bool) -> QueryResult<SqlStatement> {
        let name = self.parse_identifier()?;

        // Optional column list: CREATE VIEW v (a, b) AS ...
        let columns = if *self.peek() == Token::LParen {
            self.advance();
            let cols = self.parse_identifier_list()?;
            self.expect(Token::RParen)?;
            Some(cols)
        } else {
            None
        };

        // Expect AS keyword
        self.expect(Token::As)?;

        // Capture the remaining tokens as the view query SQL
        // We re-serialize the tokens back to SQL text
        let query = self.remaining_tokens_as_sql();

        Ok(SqlStatement::CreateView(SqlCreateView {
            name,
            columns,
            query,
            or_replace,
        }))
    }

    /// Parse CREATE TRIGGER syntax:
    /// CREATE [OR REPLACE] TRIGGER name BEFORE|AFTER INSERT|UPDATE|DELETE ON table
    /// FOR EACH ROW EXECUTE sql_statement
    fn parse_create_trigger(&mut self, or_replace: bool) -> QueryResult<SqlStatement> {
        let name = self.parse_identifier()?;

        // Parse timing: BEFORE | AFTER
        let timing = match self.advance() {
            Token::Before => TriggerTiming::Before,
            Token::After => TriggerTiming::After,
            t => {
                return Err(QueryError::ParseError(format!(
                    "Expected BEFORE or AFTER in CREATE TRIGGER, got {:?}",
                    t
                )));
            }
        };

        // Parse event: INSERT | UPDATE | DELETE
        let event = match self.advance() {
            Token::Insert => TriggerEvent::Insert,
            Token::Update => TriggerEvent::Update,
            Token::Delete => TriggerEvent::Delete,
            t => {
                return Err(QueryError::ParseError(format!(
                    "Expected INSERT, UPDATE, or DELETE in CREATE TRIGGER, got {:?}",
                    t
                )));
            }
        };

        // ON table_name
        self.expect(Token::On)?;
        let table = self.parse_identifier()?;

        // Optional: FOR EACH ROW
        if matches!(self.peek(), Token::Identifier(s) if s.to_uppercase() == "FOR") {
            self.advance(); // FOR
            self.expect(Token::Each)?;
            self.expect(Token::Row)?;
        }

        // EXECUTE sql_body
        self.expect(Token::Execute)?;

        // Collect the rest as the trigger body SQL
        let body = self.remaining_tokens_as_sql();

        Ok(SqlStatement::CreateTrigger(SqlCreateTrigger {
            name,
            timing,
            event,
            table,
            body,
            or_replace,
        }))
    }

    /// Consume all remaining tokens and reconstruct them as SQL text.
    fn remaining_tokens_as_sql(&mut self) -> String {
        let mut parts = Vec::new();
        while *self.peek() != Token::Eof && *self.peek() != Token::Semicolon {
            let token = self.advance();
            parts.push(token_to_sql(&token));
        }
        parts.join(" ")
    }

    fn parse_drop(&mut self) -> QueryResult<SqlStatement> {
        self.expect(Token::Drop)?;

        match self.advance() {
            Token::Table => {
                let if_exists = if *self.peek() == Token::If {
                    self.advance(); // IF
                    self.expect(Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropTable { name, if_exists })
            }
            Token::Spatial => {
                // DROP SPATIAL INDEX name
                self.expect(Token::Index)?;
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropSpatialIndex(name))
            }
            Token::Vector => {
                // DROP VECTOR INDEX [IF EXISTS] name
                self.expect(Token::Index)?;
                let name = if *self.peek() == Token::If {
                    self.advance(); // consume IF
                    self.expect(Token::Exists)?;
                    self.parse_identifier()?
                } else {
                    self.parse_identifier()?
                };
                Ok(SqlStatement::DropVectorIndex(name))
            }
            Token::Fulltext => {
                // DROP FULLTEXT INDEX name
                self.expect(Token::Index)?;
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropFulltextIndex(name))
            }
            Token::Index => {
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropIndex(name))
            }
            Token::Materialized => {
                // DROP MATERIALIZED VIEW [IF EXISTS] name
                self.expect(Token::View)?;
                let if_exists = if *self.peek() == Token::If {
                    self.advance();
                    self.expect(Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropMaterializedView { name, if_exists })
            }
            Token::View => {
                let if_exists = if *self.peek() == Token::If {
                    self.advance();
                    self.expect(Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropView { name, if_exists })
            }
            Token::Trigger => {
                let if_exists = if *self.peek() == Token::If {
                    self.advance();
                    self.expect(Token::Exists)?;
                    true
                } else {
                    false
                };
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropTrigger { name, if_exists })
            }
            Token::User => {
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropUser(name))
            }
            Token::Role => {
                let name = self.parse_identifier()?;
                Ok(SqlStatement::DropRole(name))
            }
            t => Err(QueryError::ParseError(format!(
                "Expected TABLE, INDEX, VIEW, TRIGGER, USER, or ROLE, got {:?}",
                t
            ))),
        }
    }

    fn parse_alter(&mut self) -> QueryResult<SqlStatement> {
        self.expect(Token::Alter)?;

        match self.peek() {
            Token::User => {
                self.advance(); // consume USER
                let name = self.parse_identifier()?;
                // Accept: ALTER USER x WITH PASSWORD 'y' or ALTER USER x PASSWORD 'y'
                if *self.peek() == Token::With {
                    self.advance(); // consume optional WITH
                }
                self.expect(Token::Password)?;
                let password = match self.advance() {
                    Token::String(s) => s.clone(),
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected string for password, got {:?}",
                            t
                        )));
                    }
                };
                Ok(SqlStatement::AlterUser { name, password })
            }
            _ => {
                // ALTER TABLE path
                self.expect(Token::Table)?;
                let table = self.parse_identifier()?;

                let action = match self.peek() {
                    Token::Add => {
                        self.advance();
                        // Optional COLUMN keyword
                        if *self.peek() == Token::Column {
                            self.advance();
                        }
                        let name = self.parse_identifier()?;
                        let data_type = self.parse_type_with_params()?;
                        AlterTableAction::AddColumn { name, data_type }
                    }
                    Token::Drop => {
                        self.advance();
                        // Optional COLUMN keyword
                        if *self.peek() == Token::Column {
                            self.advance();
                        }
                        let name = self.parse_identifier()?;
                        AlterTableAction::DropColumn { name }
                    }
                    Token::Rename => {
                        self.advance();
                        // Optional COLUMN keyword
                        if *self.peek() == Token::Column {
                            self.advance();
                        }
                        let old_name = self.parse_identifier()?;
                        self.expect(Token::To)?;
                        let new_name = self.parse_identifier()?;
                        AlterTableAction::RenameColumn { old_name, new_name }
                    }
                    t => {
                        return Err(QueryError::ParseError(format!(
                            "Expected ADD, DROP, or RENAME after ALTER TABLE, got {:?}",
                            t
                        )));
                    }
                };

                Ok(SqlStatement::AlterTable(SqlAlterTable { table, action }))
            }
        }
    }

    fn parse_truncate(&mut self) -> QueryResult<SqlStatement> {
        self.expect(Token::Truncate)?;
        // Optional TABLE keyword
        if *self.peek() == Token::Table {
            self.advance();
        }
        let name = self.parse_identifier()?;
        Ok(SqlStatement::TruncateTable(name))
    }

    fn parse_show(&mut self) -> QueryResult<SqlStatement> {
        self.expect(Token::Show)?;
        match self.peek().clone() {
            Token::Identifier(ref s) if s.to_uppercase() == "TABLES" => {
                self.advance();
                Ok(SqlStatement::ShowTables)
            }
            Token::Identifier(ref s) if s.to_uppercase() == "COLUMNS" => {
                self.advance();
                // SHOW COLUMNS FROM <table>
                if *self.peek() == Token::From {
                    self.advance();
                }
                let table = self.parse_identifier()?;
                Ok(SqlStatement::ShowColumns(table))
            }
            Token::Column => {
                // Token::Column is a keyword that could match "COLUMNS" too
                self.advance();
                // Check if next is "S" part... no, "COLUMN" is already a keyword.
                // SHOW COLUMNS — "COLUMNS" would be parsed as Identifier("COLUMNS") since
                // only "COLUMN" (singular) is a keyword. So this branch handles SHOW COLUMN FROM table
                if *self.peek() == Token::From {
                    self.advance();
                }
                let table = self.parse_identifier()?;
                Ok(SqlStatement::ShowColumns(table))
            }
            t => Err(QueryError::ParseError(format!(
                "Expected TABLES or COLUMNS after SHOW, got {:?}",
                t
            ))),
        }
    }
}

impl Default for SqlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("SELECT * FROM users").unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.from.unwrap().table_name(), Some("users"));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_select_with_columns() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("SELECT id, name FROM users").unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 2);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_select_with_where() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_select_with_join() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users u JOIN orders o ON u.id = o.user_id")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Inner);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_select_with_order_limit() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users ORDER BY name DESC LIMIT 10 OFFSET 5")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.order_by.len(), 1);
                assert!(q.order_by[0].descending);
                assert_eq!(q.limit, Some(10));
                assert_eq!(q.offset, Some(5));
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_filter_clause_on_aggregate() {
        let mut parser = SqlParser::new();

        // COUNT(*) FILTER (WHERE x > 0) → SUM(CASE WHEN x > 0 THEN 1 ELSE 0 END)
        let stmt = parser
            .parse("SELECT COUNT(*) FILTER (WHERE x > 0) FROM t")
            .unwrap();
        match &stmt {
            SqlStatement::Select(q) => {
                // Should be rewritten to SUM with CASE
                match &q.columns[0].expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "SUM");
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expression::Case { .. }));
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            _ => panic!("Expected SELECT"),
        }

        // SUM(x) FILTER (WHERE active) — arg gets wrapped in CASE
        let stmt = parser
            .parse("SELECT SUM(amount) FILTER (WHERE active = true) FROM t")
            .unwrap();
        match &stmt {
            SqlStatement::Select(q) => {
                match &q.columns[0].expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "SUM");
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expression::Case { .. }));
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_values_standalone() {
        let mut parser = SqlParser::new();

        // Single row
        let stmt = parser.parse("VALUES (1, 'hello')").unwrap();
        match &stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 2);
                assert_eq!(q.columns[0].alias, Some("column1".to_string()));
                assert_eq!(q.columns[1].alias, Some("column2".to_string()));
            }
            _ => panic!("Expected SELECT"),
        }

        // Multiple rows
        let stmt = parser
            .parse("VALUES (1, 'a'), (2, 'b'), (3, 'c')")
            .unwrap();
        match &stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 2);
                assert!(q.set_op.is_some()); // UNION ALL for row 2+
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_fetch_first_n_rows_only() {
        let mut parser = SqlParser::new();

        // FETCH FIRST N ROWS ONLY
        let stmt = parser
            .parse("SELECT * FROM users ORDER BY name FETCH FIRST 10 ROWS ONLY")
            .unwrap();
        match &stmt {
            SqlStatement::Select(q) => assert_eq!(q.limit, Some(10)),
            _ => panic!("Expected SELECT"),
        }

        // FETCH NEXT N ROWS ONLY
        let stmt = parser
            .parse("SELECT * FROM users FETCH NEXT 5 ROWS ONLY")
            .unwrap();
        match &stmt {
            SqlStatement::Select(q) => assert_eq!(q.limit, Some(5)),
            _ => panic!("Expected SELECT"),
        }

        // FETCH FIRST N ROW ONLY (singular)
        let stmt = parser
            .parse("SELECT * FROM users FETCH FIRST 1 ROW ONLY")
            .unwrap();
        match &stmt {
            SqlStatement::Select(q) => assert_eq!(q.limit, Some(1)),
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_insert() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();

        match stmt {
            SqlStatement::Insert(i) => {
                assert_eq!(i.table, "users");
                assert_eq!(i.columns, vec!["id", "name"]);
                match &i.source {
                    InsertSource::Values(values) => assert_eq!(values.len(), 1),
                    _ => panic!("Expected VALUES source"),
                }
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_insert_select_basic() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("INSERT INTO archive (id, name) SELECT id, name FROM users")
            .unwrap();

        match stmt {
            SqlStatement::Insert(i) => {
                assert_eq!(i.table, "archive");
                assert_eq!(i.columns, vec!["id", "name"]);
                match &i.source {
                    InsertSource::Select(query) => {
                        assert_eq!(query.columns.len(), 2);
                        assert_eq!(
                            query.from.as_ref().and_then(|f| f.table_name()).as_deref(),
                            Some("users")
                        );
                    }
                    _ => panic!("Expected SELECT source"),
                }
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_insert_select_with_where() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse(
                "INSERT INTO active_users (id, name) SELECT id, name FROM users WHERE active = 1",
            )
            .unwrap();

        match stmt {
            SqlStatement::Insert(i) => {
                assert_eq!(i.table, "active_users");
                match &i.source {
                    InsertSource::Select(query) => {
                        assert!(query.where_clause.is_some());
                    }
                    _ => panic!("Expected SELECT source"),
                }
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_insert_select_column_mapping() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("INSERT INTO summary (total_count) SELECT COUNT(*) FROM orders")
            .unwrap();

        match stmt {
            SqlStatement::Insert(i) => {
                assert_eq!(i.table, "summary");
                assert_eq!(i.columns, vec!["total_count"]);
                match &i.source {
                    InsertSource::Select(query) => {
                        assert_eq!(query.columns.len(), 1);
                    }
                    _ => panic!("Expected SELECT source"),
                }
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_insert_select_with_cte() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("INSERT INTO results (id, val) WITH src AS (SELECT id, value FROM data WHERE value > 10) SELECT id, val FROM src")
            .unwrap();

        match stmt {
            SqlStatement::Insert(i) => {
                assert_eq!(i.table, "results");
                assert_eq!(i.columns, vec!["id", "val"]);
                match &i.source {
                    InsertSource::Select(query) => {
                        assert_eq!(query.ctes.len(), 1);
                        assert_eq!(query.ctes[0].name, "src");
                    }
                    _ => panic!("Expected SELECT source"),
                }
            }
            _ => panic!("Expected INSERT statement"),
        }
    }

    #[test]
    fn test_update() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("UPDATE users SET name = 'Bob' WHERE id = 1")
            .unwrap();

        match stmt {
            SqlStatement::Update(u) => {
                assert_eq!(u.table, "users");
                assert_eq!(u.assignments.len(), 1);
                assert!(u.where_clause.is_some());
            }
            _ => panic!("Expected UPDATE statement"),
        }
    }

    #[test]
    fn test_delete() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("DELETE FROM users WHERE id = 1").unwrap();

        match stmt {
            SqlStatement::Delete(d) => {
                assert_eq!(d.table, "users");
                assert!(d.where_clause.is_some());
            }
            _ => panic!("Expected DELETE statement"),
        }
    }

    #[test]
    fn test_create_table() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
            .unwrap();

        match stmt {
            SqlStatement::CreateTable(c) => {
                assert_eq!(c.name, "users");
                assert_eq!(c.columns.len(), 2);
                assert!(c.columns[0].primary_key);
                assert!(!c.columns[1].nullable);
            }
            _ => panic!("Expected CREATE TABLE statement"),
        }
    }

    #[test]
    fn test_parameters() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users WHERE id = $1 AND name = :name")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_function_call() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT COUNT(*), MAX(age) FROM users")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 2);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_complex_expression() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users WHERE age >= 18 AND (status = 'active' OR role = 'admin')")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_simple_cte() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("WITH active_users AS (SELECT * FROM users WHERE active = true) SELECT * FROM active_users")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.ctes.len(), 1);
                assert_eq!(q.ctes[0].name, "active_users");
                assert!(!q.ctes[0].recursive);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_multiple_ctes() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("WITH a AS (SELECT 1), b AS (SELECT 2) SELECT * FROM a JOIN b ON true")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.ctes.len(), 2);
                assert_eq!(q.ctes[0].name, "a");
                assert_eq!(q.ctes[1].name, "b");
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_cte_with_columns() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("WITH temp(x, y) AS (SELECT a, b FROM t) SELECT * FROM temp")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.ctes.len(), 1);
                assert_eq!(q.ctes[0].columns, vec!["x", "y"]);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_recursive_cte() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("WITH RECURSIVE nums AS (SELECT 1) SELECT * FROM nums")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.ctes.len(), 1);
                assert!(q.ctes[0].recursive);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_subquery_in_where() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
                // Check that the IN expression contains a subquery
                if let Some(Expression::In { list, .. }) = q.where_clause {
                    assert!(!list.is_empty());
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_scalar_subquery() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT (SELECT MAX(id) FROM users) as max_id FROM dual")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                match &q.columns[0].expr {
                    Expression::Subquery(_) => {}
                    _ => panic!("Expected subquery expression"),
                }
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_date_literal() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM t WHERE d >= DATE '1994-01-01'")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_interval() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM t WHERE d <= DATE '1998-12-01' - INTERVAL '90' DAY")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_extract() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT EXTRACT(YEAR FROM o_orderdate) as year FROM orders")
            .unwrap();

        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
            }
            _ => panic!("Expected SELECT statement"),
        }
    }

    #[test]
    fn test_parse_cast() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("SELECT CAST(age AS TEXT) FROM users").unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                match &q.columns[0].expr {
                    Expression::Cast { target_type, .. } => {
                        assert_eq!(target_type, "TEXT");
                    }
                    other => panic!("Expected Cast, got {:?}", other),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_cast_in_where() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users WHERE CAST(id AS TEXT) = '1'")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_union() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT id FROM users UNION SELECT id FROM orders")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                let set_op = q.set_op.as_ref().unwrap();
                assert_eq!(set_op.op_type, SetOperationType::Union);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_union_all() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT id FROM users UNION ALL SELECT id FROM orders")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                let set_op = q.set_op.as_ref().unwrap();
                assert_eq!(set_op.op_type, SetOperationType::UnionAll);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ======================== ALTER TABLE Parser Tests ========================

    #[test]
    fn test_alter_table_add_column() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("ALTER TABLE users ADD COLUMN email TEXT")
            .unwrap();
        match stmt {
            SqlStatement::AlterTable(alter) => {
                assert_eq!(alter.table, "users");
                match alter.action {
                    AlterTableAction::AddColumn { name, data_type } => {
                        assert_eq!(name, "email");
                        assert_eq!(data_type, "TEXT");
                    }
                    _ => panic!("Expected AddColumn"),
                }
            }
            _ => panic!("Expected AlterTable"),
        }
    }

    #[test]
    fn test_alter_table_add_without_column_keyword() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("ALTER TABLE users ADD email TEXT").unwrap();
        match stmt {
            SqlStatement::AlterTable(alter) => {
                assert_eq!(alter.table, "users");
                match alter.action {
                    AlterTableAction::AddColumn { name, data_type } => {
                        assert_eq!(name, "email");
                        assert_eq!(data_type, "TEXT");
                    }
                    _ => panic!("Expected AddColumn"),
                }
            }
            _ => panic!("Expected AlterTable"),
        }
    }

    #[test]
    fn test_alter_table_drop_column() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("ALTER TABLE users DROP COLUMN age").unwrap();
        match stmt {
            SqlStatement::AlterTable(alter) => {
                assert_eq!(alter.table, "users");
                match alter.action {
                    AlterTableAction::DropColumn { name } => {
                        assert_eq!(name, "age");
                    }
                    _ => panic!("Expected DropColumn"),
                }
            }
            _ => panic!("Expected AlterTable"),
        }
    }

    #[test]
    fn test_alter_table_drop_without_column_keyword() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("ALTER TABLE users DROP age").unwrap();
        match stmt {
            SqlStatement::AlterTable(alter) => {
                assert_eq!(alter.table, "users");
                match alter.action {
                    AlterTableAction::DropColumn { name } => {
                        assert_eq!(name, "age");
                    }
                    _ => panic!("Expected DropColumn"),
                }
            }
            _ => panic!("Expected AlterTable"),
        }
    }

    #[test]
    fn test_alter_table_rename_column() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("ALTER TABLE users RENAME COLUMN name TO full_name")
            .unwrap();
        match stmt {
            SqlStatement::AlterTable(alter) => {
                assert_eq!(alter.table, "users");
                match alter.action {
                    AlterTableAction::RenameColumn { old_name, new_name } => {
                        assert_eq!(old_name, "name");
                        assert_eq!(new_name, "full_name");
                    }
                    _ => panic!("Expected RenameColumn"),
                }
            }
            _ => panic!("Expected AlterTable"),
        }
    }

    #[test]
    fn test_alter_table_rename_without_column_keyword() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("ALTER TABLE users RENAME name TO full_name")
            .unwrap();
        match stmt {
            SqlStatement::AlterTable(alter) => {
                assert_eq!(alter.table, "users");
                match alter.action {
                    AlterTableAction::RenameColumn { old_name, new_name } => {
                        assert_eq!(old_name, "name");
                        assert_eq!(new_name, "full_name");
                    }
                    _ => panic!("Expected RenameColumn"),
                }
            }
            _ => panic!("Expected AlterTable"),
        }
    }

    #[test]
    fn test_alter_table_missing_action_errors() {
        let mut parser = SqlParser::new();
        let result = parser.parse("ALTER TABLE users");
        assert!(result.is_err());
    }

    #[test]
    fn test_alter_table_add_various_types() {
        for (type_name, expected) in &[
            ("INTEGER", "INTEGER"),
            ("VARCHAR", "VARCHAR"),
            ("BOOLEAN", "BOOLEAN"),
        ] {
            let mut parser = SqlParser::new();
            let stmt = parser
                .parse(&format!("ALTER TABLE t ADD COLUMN c {}", type_name))
                .unwrap();
            match stmt {
                SqlStatement::AlterTable(alter) => match alter.action {
                    AlterTableAction::AddColumn { data_type, .. } => {
                        assert_eq!(data_type, *expected);
                    }
                    _ => panic!("Expected AddColumn"),
                },
                _ => panic!("Expected AlterTable"),
            }
        }
    }

    // ==================== ORDER BY expression tests ====================

    #[test]
    fn test_parse_order_by_expression() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM products ORDER BY price * quantity")
            .unwrap();
        if let SqlStatement::Select(query) = stmt {
            assert_eq!(query.order_by.len(), 1);
            assert!(!query.order_by[0].descending);
            match &query.order_by[0].expr {
                Expression::Binary { .. } => {} // Expected
                other => panic!("Expected Binary expression, got {:?}", other),
            }
        } else {
            panic!("Expected Select");
        }
    }

    #[test]
    fn test_parse_order_by_function() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM users ORDER BY LOWER(name)")
            .unwrap();
        if let SqlStatement::Select(query) = stmt {
            assert_eq!(query.order_by.len(), 1);
            match &query.order_by[0].expr {
                Expression::Function { name, args } => {
                    assert_eq!(name, "LOWER");
                    assert_eq!(args.len(), 1);
                }
                other => panic!("Expected Function expression, got {:?}", other),
            }
        } else {
            panic!("Expected Select");
        }
    }

    #[test]
    fn test_parse_order_by_nulls_first() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM t ORDER BY name NULLS FIRST")
            .unwrap();
        if let SqlStatement::Select(query) = stmt {
            assert_eq!(query.order_by.len(), 1);
            assert!(!query.order_by[0].descending);
            assert_eq!(query.order_by[0].nulls_first, Some(true));
        } else {
            panic!("Expected Select");
        }
    }

    #[test]
    fn test_parse_order_by_nulls_last() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM t ORDER BY name DESC NULLS LAST")
            .unwrap();
        if let SqlStatement::Select(query) = stmt {
            assert_eq!(query.order_by.len(), 1);
            assert!(query.order_by[0].descending);
            assert_eq!(query.order_by[0].nulls_first, Some(false));
        } else {
            panic!("Expected Select");
        }
    }

    // ==================== View parser tests ====================

    #[test]
    fn test_parse_create_view() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("CREATE VIEW active_users AS SELECT * FROM users WHERE active = 1")
            .unwrap();
        match stmt {
            SqlStatement::CreateView(v) => {
                assert_eq!(v.name, "active_users");
                assert!(v.columns.is_none());
                assert!(!v.or_replace);
                assert!(v.query.contains("SELECT"));
                assert!(v.query.contains("users"));
            }
            _ => panic!("Expected CreateView, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_view_with_columns() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("CREATE VIEW v (a, b) AS SELECT id, name FROM t")
            .unwrap();
        match stmt {
            SqlStatement::CreateView(v) => {
                assert_eq!(v.name, "v");
                assert_eq!(v.columns, Some(vec!["a".to_string(), "b".to_string()]));
                assert!(!v.or_replace);
            }
            _ => panic!("Expected CreateView"),
        }
    }

    #[test]
    fn test_parse_create_or_replace_view() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("CREATE OR REPLACE VIEW myview AS SELECT id FROM t")
            .unwrap();
        match stmt {
            SqlStatement::CreateView(v) => {
                assert_eq!(v.name, "myview");
                assert!(v.or_replace);
            }
            _ => panic!("Expected CreateView"),
        }
    }

    #[test]
    fn test_parse_drop_view() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("DROP VIEW myview").unwrap();
        match stmt {
            SqlStatement::DropView { name, if_exists } => {
                assert_eq!(name, "myview");
                assert!(!if_exists);
            }
            _ => panic!("Expected DropView"),
        }
    }

    #[test]
    fn test_parse_drop_view_if_exists() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("DROP VIEW IF EXISTS myview").unwrap();
        match stmt {
            SqlStatement::DropView { name, if_exists } => {
                assert_eq!(name, "myview");
                assert!(if_exists);
            }
            _ => panic!("Expected DropView"),
        }
    }

    #[test]
    fn test_parse_group_by_expression() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT UPPER(name), COUNT(*) FROM t GROUP BY UPPER(name)")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.group_by.len(), 1);
                match &q.group_by[0] {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "UPPER");
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_group_by_function() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT SUBSTR(name, 1, 1), COUNT(*) FROM t GROUP BY SUBSTR(name, 1, 1)")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.group_by.len(), 1);
                match &q.group_by[0] {
                    Expression::Function { name, .. } => assert_eq!(name, "SUBSTR"),
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_group_by_multiple_expressions() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT a, b + c, COUNT(*) FROM t GROUP BY a, b + c")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.group_by.len(), 2);
                match &q.group_by[0] {
                    Expression::Column(name) => assert_eq!(name, "a"),
                    other => panic!("Expected Column, got {:?}", other),
                }
                match &q.group_by[1] {
                    Expression::Binary { .. } => {}
                    other => panic!("Expected Binary, got {:?}", other),
                }
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_except() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT id FROM users EXCEPT SELECT id FROM banned")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                let set_op = q.set_op.as_ref().unwrap();
                assert_eq!(set_op.op_type, SetOperationType::Except);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_intersect() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT id FROM users INTERSECT SELECT id FROM active")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                let set_op = q.set_op.as_ref().unwrap();
                assert_eq!(set_op.op_type, SetOperationType::Intersect);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_except_all() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT id FROM users EXCEPT ALL SELECT id FROM banned")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                let set_op = q.set_op.as_ref().unwrap();
                assert_eq!(set_op.op_type, SetOperationType::ExceptAll);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_subquery_in_from() {
        let mut parser = SqlParser::new();
        // Use simple column (not qualified) to isolate the issue
        let stmt = parser.parse("SELECT name FROM (SELECT id, name FROM users) AS t");
        assert!(stmt.is_ok(), "Parse failed: {:?}", stmt.err());
        match stmt.unwrap() {
            SqlStatement::Select(q) => {
                let from = q.from.as_ref().unwrap();
                assert!(from.is_subquery());
                assert_eq!(from.alias, Some("t".to_string()));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_left_join() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM a LEFT JOIN b ON a.id = b.id")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Left);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_full_outer_join() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Full);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_create_table_as_select() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse(
                "CREATE TABLE summary AS SELECT dept, COUNT(*) AS cnt FROM employees GROUP BY dept",
            )
            .unwrap();
        match stmt {
            SqlStatement::CreateTableAs {
                name,
                query,
                if_not_exists,
            } => {
                assert_eq!(name, "summary");
                assert!(!if_not_exists);
                assert_eq!(query.columns.len(), 2);
            }
            other => panic!("Expected CreateTableAs, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_truncate() {
        let mut parser = SqlParser::new();
        let stmt = parser.parse("TRUNCATE TABLE users").unwrap();
        match stmt {
            SqlStatement::TruncateTable(name) => {
                assert_eq!(name, "users");
            }
            other => panic!("Expected TruncateTable, got {:?}", other),
        }

        // Without TABLE keyword
        let mut parser2 = SqlParser::new();
        let stmt2 = parser2.parse("TRUNCATE users").unwrap();
        match stmt2 {
            SqlStatement::TruncateTable(name) => {
                assert_eq!(name, "users");
            }
            other => panic!("Expected TruncateTable, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_count_distinct() {
        let mut parser = SqlParser::new();
        let stmt = parser
            .parse("SELECT COUNT(DISTINCT category) FROM items")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                match &q.columns[0].expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "COUNT_DISTINCT");
                        assert_eq!(args.len(), 1);
                        match &args[0] {
                            Expression::Column(c) => assert_eq!(c, "category"),
                            other => panic!("Expected column, got {:?}", other),
                        }
                    }
                    other => panic!("Expected function, got {:?}", other),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_insert_on_conflict() {
        let mut parser = SqlParser::new();

        // DO NOTHING
        let stmt = parser
            .parse("INSERT INTO t (id, name) VALUES (1, 'a') ON CONFLICT (id) DO NOTHING")
            .unwrap();
        match stmt {
            SqlStatement::Insert(ins) => {
                let oc = ins.on_conflict.unwrap();
                assert_eq!(oc.columns, vec!["id".to_string()]);
                assert!(matches!(oc.action, OnConflictAction::DoNothing));
            }
            _ => panic!("Expected INSERT"),
        }

        // DO UPDATE SET
        let stmt2 = parser.parse("INSERT INTO t (id, name) VALUES (1, 'a') ON CONFLICT (id) DO UPDATE SET name = excluded.name").unwrap();
        match stmt2 {
            SqlStatement::Insert(ins) => {
                let oc = ins.on_conflict.unwrap();
                assert_eq!(oc.columns, vec!["id".to_string()]);
                match &oc.action {
                    OnConflictAction::DoUpdate(assignments) => {
                        assert_eq!(assignments.len(), 1);
                        assert_eq!(assignments[0].0, "name");
                        match &assignments[0].1 {
                            Expression::QualifiedColumn { table, column } => {
                                assert_eq!(table, "excluded");
                                assert_eq!(column, "name");
                            }
                            other => panic!("Expected qualified column, got {:?}", other),
                        }
                    }
                    _ => panic!("Expected DoUpdate"),
                }
            }
            _ => panic!("Expected INSERT"),
        }

        // No target columns
        let stmt3 = parser
            .parse("INSERT INTO t (id) VALUES (1) ON CONFLICT DO NOTHING")
            .unwrap();
        match stmt3 {
            SqlStatement::Insert(ins) => {
                let oc = ins.on_conflict.unwrap();
                assert!(oc.columns.is_empty());
                assert!(matches!(oc.action, OnConflictAction::DoNothing));
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_parse_create_fulltext_index() {
        let stmt = SqlParser::new()
            .parse("CREATE FULLTEXT INDEX ft_articles ON articles (title, body)")
            .unwrap();
        match stmt {
            SqlStatement::CreateFulltextIndex {
                name,
                table,
                columns,
                options,
            } => {
                assert_eq!(name, "ft_articles");
                assert_eq!(table, "articles");
                assert_eq!(columns, vec!["title", "body"]);
                assert!(options.is_empty());
            }
            _ => panic!("Expected CreateFulltextIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_fulltext_index_with_options() {
        let stmt = SqlParser::new().parse("CREATE FULLTEXT INDEX ft_articles ON articles (title, body) WITH (analyzer = 'standard')").unwrap();
        match stmt {
            SqlStatement::CreateFulltextIndex {
                name,
                table,
                columns,
                options,
            } => {
                assert_eq!(name, "ft_articles");
                assert_eq!(table, "articles");
                assert_eq!(columns, vec!["title", "body"]);
                assert_eq!(options.get("analyzer").unwrap(), "standard");
            }
            _ => panic!("Expected CreateFulltextIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_drop_fulltext_index() {
        let stmt = SqlParser::new()
            .parse("DROP FULLTEXT INDEX ft_articles")
            .unwrap();
        match stmt {
            SqlStatement::DropFulltextIndex(name) => {
                assert_eq!(name, "ft_articles");
            }
            _ => panic!("Expected DropFulltextIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_match_against() {
        let stmt = SqlParser::new()
            .parse("SELECT * FROM articles WHERE MATCH(title, body) AGAINST ('database search')")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                let where_clause = q.where_clause.unwrap();
                match where_clause {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "MATCH_AGAINST");
                        assert_eq!(args.len(), 3); // search_text + 2 columns
                        assert!(
                            matches!(&args[0], Expression::Literal(Value::String(s)) if s == "database search")
                        );
                        assert!(matches!(&args[1], Expression::Column(s) if s == "title"));
                        assert!(matches!(&args[2], Expression::Column(s) if s == "body"));
                    }
                    _ => panic!("Expected MATCH_AGAINST function, got {:?}", where_clause),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_match_against_in_select() {
        let stmt = SqlParser::new()
            .parse("SELECT title, MATCH(body) AGAINST ('rust') AS relevance FROM articles")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 2);
                let col2 = &q.columns[1];
                assert_eq!(col2.alias.as_deref(), Some("relevance"));
                match &col2.expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "MATCH_AGAINST");
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("Expected MATCH_AGAINST function"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_ts_rank_function() {
        let stmt = SqlParser::new()
            .parse("SELECT TS_RANK(body, 'search') FROM docs")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => match &q.columns[0].expr {
                Expression::Function { name, args } => {
                    assert_eq!(name, "TS_RANK");
                    assert_eq!(args.len(), 2);
                }
                _ => panic!("Expected TS_RANK function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_ts_headline_function() {
        let stmt = SqlParser::new()
            .parse("SELECT TS_HEADLINE(body, 'search terms') FROM docs")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => match &q.columns[0].expr {
                Expression::Function { name, args } => {
                    assert_eq!(name, "TS_HEADLINE");
                    assert_eq!(args.len(), 2);
                }
                _ => panic!("Expected TS_HEADLINE function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_to_tsvector() {
        let stmt = SqlParser::new()
            .parse("SELECT TO_TSVECTOR('the quick brown fox')")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => match &q.columns[0].expr {
                Expression::Function { name, args } => {
                    assert_eq!(name, "TO_TSVECTOR");
                    assert_eq!(args.len(), 1);
                }
                _ => panic!("Expected TO_TSVECTOR function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_to_tsquery() {
        let stmt = SqlParser::new()
            .parse("SELECT TO_TSQUERY('search AND query')")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => match &q.columns[0].expr {
                Expression::Function { name, args } => {
                    assert_eq!(name, "TO_TSQUERY");
                    assert_eq!(args.len(), 1);
                }
                _ => panic!("Expected TO_TSQUERY function"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_fulltext_keywords_as_identifiers() {
        // FULLTEXT, MATCH, AGAINST should work as identifiers (column/table names)
        let stmt = SqlParser::new()
            .parse("SELECT fulltext, match, against FROM t")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 3);
                assert!(matches!(&q.columns[0].expr, Expression::Column(s) if s == "fulltext"));
                assert!(matches!(&q.columns[1].expr, Expression::Column(s) if s == "match"));
                assert!(matches!(&q.columns[2].expr, Expression::Column(s) if s == "against"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ==================== Foreign Key parser tests ====================

    #[test]
    fn test_parse_foreign_key_references() {
        let stmt = SqlParser::new().parse(
            "CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT REFERENCES customers(id))"
        ).unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                let fk_col = &ct.columns[1];
                assert_eq!(fk_col.name, "customer_id");
                let fk = fk_col.foreign_key.as_ref().unwrap();
                assert_eq!(fk.ref_table, "customers");
                assert_eq!(fk.ref_column, "id");
                assert!(fk.on_delete.is_none());
                assert!(fk.on_update.is_none());
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_foreign_key_with_actions() {
        let stmt = SqlParser::new().parse(
            "CREATE TABLE orders (id INT, cust_id INT REFERENCES customers(id) ON DELETE CASCADE ON UPDATE RESTRICT)"
        ).unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                let fk = ct.columns[1].foreign_key.as_ref().unwrap();
                assert_eq!(fk.ref_table, "customers");
                assert_eq!(fk.ref_column, "id");
                assert_eq!(fk.on_delete, Some(ReferentialAction::Cascade));
                assert_eq!(fk.on_update, Some(ReferentialAction::Restrict));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_foreign_key_set_null() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE child (id INT, pid INT REFERENCES parent(id) ON DELETE SET NULL)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                let fk = ct.columns[1].foreign_key.as_ref().unwrap();
                assert_eq!(fk.on_delete, Some(ReferentialAction::SetNull));
                assert!(fk.on_update.is_none());
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_foreign_key_no_action() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE child (id INT, pid INT REFERENCES parent(id) ON DELETE NO ACTION)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                let fk = ct.columns[1].foreign_key.as_ref().unwrap();
                assert_eq!(fk.on_delete, Some(ReferentialAction::NoAction));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    // ==================== NATURAL JOIN / USING parser tests ====================

    #[test]
    fn test_parse_natural_join() {
        let stmt = SqlParser::new()
            .parse("SELECT * FROM t1 NATURAL JOIN t2")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Inner);
                assert_eq!(q.joins[0].table, "t2");
                assert_eq!(q.joins[0].using_columns, vec!["*"]); // sentinel
                assert!(q.joins[0].condition.is_none());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_natural_left_join() {
        let stmt = SqlParser::new()
            .parse("SELECT * FROM t1 NATURAL LEFT JOIN t2")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Left);
                assert_eq!(q.joins[0].using_columns, vec!["*"]);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_join_using_single() {
        let stmt = SqlParser::new()
            .parse("SELECT * FROM t1 JOIN t2 USING (id)")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Inner);
                assert_eq!(q.joins[0].table, "t2");
                assert_eq!(q.joins[0].using_columns, vec!["id"]);
                assert!(q.joins[0].condition.is_none());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_join_using_multiple() {
        let stmt = SqlParser::new()
            .parse("SELECT * FROM t1 LEFT JOIN t2 USING (a, b, c)")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.joins.len(), 1);
                assert_eq!(q.joins[0].join_type, JoinType::Left);
                assert_eq!(q.joins[0].using_columns, vec!["a", "b", "c"]);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_fk_keywords_as_identifiers() {
        // FOREIGN, REFERENCES, CASCADE, RESTRICT, NATURAL, USING should work as identifiers
        let stmt = SqlParser::new()
            .parse("SELECT foreign, references, cascade, restrict, natural, using FROM t")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 6);
                assert!(matches!(&q.columns[0].expr, Expression::Column(s) if s == "foreign"));
                assert!(matches!(&q.columns[1].expr, Expression::Column(s) if s == "references"));
                assert!(matches!(&q.columns[2].expr, Expression::Column(s) if s == "cascade"));
                assert!(matches!(&q.columns[3].expr, Expression::Column(s) if s == "restrict"));
                assert!(matches!(&q.columns[4].expr, Expression::Column(s) if s == "natural"));
                assert!(matches!(&q.columns[5].expr, Expression::Column(s) if s == "using"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ==================== Trigger parser tests ====================

    #[test]
    fn test_parse_create_trigger_after_insert() {
        let stmt = SqlParser::new().parse(
            "CREATE TRIGGER audit_insert AFTER INSERT ON users FOR EACH ROW EXECUTE INSERT INTO audit (msg) VALUES ('insert')"
        ).unwrap();
        match stmt {
            SqlStatement::CreateTrigger(t) => {
                assert_eq!(t.name, "audit_insert");
                assert_eq!(t.timing, TriggerTiming::After);
                assert_eq!(t.event, TriggerEvent::Insert);
                assert_eq!(t.table, "users");
                assert!(t.body.contains("INSERT INTO audit"));
                assert!(!t.or_replace);
            }
            _ => panic!("Expected CREATE TRIGGER"),
        }
    }

    #[test]
    fn test_parse_create_trigger_before_update() {
        let stmt = SqlParser::new().parse(
            "CREATE TRIGGER check_update BEFORE UPDATE ON items EXECUTE UPDATE log SET count = 1"
        ).unwrap();
        match stmt {
            SqlStatement::CreateTrigger(t) => {
                assert_eq!(t.name, "check_update");
                assert_eq!(t.timing, TriggerTiming::Before);
                assert_eq!(t.event, TriggerEvent::Update);
                assert_eq!(t.table, "items");
                assert!(t.body.contains("UPDATE log"));
            }
            _ => panic!("Expected CREATE TRIGGER"),
        }
    }

    #[test]
    fn test_parse_create_or_replace_trigger() {
        let stmt = SqlParser::new().parse(
            "CREATE OR REPLACE TRIGGER trg1 AFTER DELETE ON items FOR EACH ROW EXECUTE SELECT 1"
        ).unwrap();
        match stmt {
            SqlStatement::CreateTrigger(t) => {
                assert_eq!(t.name, "trg1");
                assert_eq!(t.timing, TriggerTiming::After);
                assert_eq!(t.event, TriggerEvent::Delete);
                assert!(t.or_replace);
            }
            _ => panic!("Expected CREATE TRIGGER"),
        }
    }

    #[test]
    fn test_parse_drop_trigger() {
        let stmt = SqlParser::new().parse("DROP TRIGGER my_trigger").unwrap();
        match stmt {
            SqlStatement::DropTrigger { name, if_exists } => {
                assert_eq!(name, "my_trigger");
                assert!(!if_exists);
            }
            _ => panic!("Expected DROP TRIGGER"),
        }
    }

    #[test]
    fn test_parse_drop_trigger_if_exists() {
        let stmt = SqlParser::new()
            .parse("DROP TRIGGER IF EXISTS my_trigger")
            .unwrap();
        match stmt {
            SqlStatement::DropTrigger { name, if_exists } => {
                assert_eq!(name, "my_trigger");
                assert!(if_exists);
            }
            _ => panic!("Expected DROP TRIGGER IF EXISTS"),
        }
    }

    #[test]
    fn test_trigger_keywords_as_identifiers() {
        let stmt = SqlParser::new()
            .parse("SELECT trigger, before, after, each, execute FROM t")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 5);
                assert!(matches!(&q.columns[0].expr, Expression::Column(s) if s == "trigger"));
                assert!(matches!(&q.columns[1].expr, Expression::Column(s) if s == "before"));
                assert!(matches!(&q.columns[2].expr, Expression::Column(s) if s == "after"));
                assert!(matches!(&q.columns[3].expr, Expression::Column(s) if s == "each"));
                assert!(matches!(&q.columns[4].expr, Expression::Column(s) if s == "execute"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ============================================================
    // Spatial SQL Parser Tests
    // ============================================================

    #[test]
    fn test_parse_create_spatial_index() {
        let stmt = SqlParser::new()
            .parse("CREATE SPATIAL INDEX idx_geom ON places (geom)")
            .unwrap();
        match stmt {
            SqlStatement::CreateSpatialIndex {
                name,
                table,
                column,
            } => {
                assert_eq!(name, "idx_geom");
                assert_eq!(table, "places");
                assert_eq!(column, "geom");
            }
            _ => panic!("Expected CreateSpatialIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_drop_spatial_index() {
        let stmt = SqlParser::new()
            .parse("DROP SPATIAL INDEX idx_geom")
            .unwrap();
        match stmt {
            SqlStatement::DropSpatialIndex(name) => {
                assert_eq!(name, "idx_geom");
            }
            _ => panic!("Expected DropSpatialIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_st_point_function() {
        let stmt = SqlParser::new()
            .parse("SELECT ST_POINT(10.5, 20.3)")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                match &q.columns[0].expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "ST_POINT");
                        assert_eq!(args.len(), 2);
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_st_distance_function() {
        let stmt = SqlParser::new()
            .parse("SELECT ST_DISTANCE(geom1, geom2) FROM t")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                match &q.columns[0].expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "ST_DISTANCE");
                        assert_eq!(args.len(), 2);
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_st_contains_where() {
        let stmt = SqlParser::new()
            .parse("SELECT * FROM places WHERE ST_CONTAINS(boundary, ST_POINT(1, 2))")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert!(q.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_spatial_keyword_as_identifier() {
        let stmt = SqlParser::new().parse("SELECT spatial FROM t").unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                assert!(matches!(&q.columns[0].expr, Expression::Column(s) if s == "spatial"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ==================== Column Family parser tests ====================

    #[test]
    fn test_parse_column_family_inline() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE t (id INT PRIMARY KEY, name TEXT COLUMN FAMILY meta)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(ct.columns.len(), 2);
                assert_eq!(ct.columns[0].column_family, None);
                assert_eq!(ct.columns[1].column_family, Some("meta".to_string()));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_column_family_shorthand() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE t (id INT, val TEXT FAMILY readings)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(ct.columns[1].column_family, Some("readings".to_string()));
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_with_column_families() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE t (id INT PRIMARY KEY, val TEXT) WITH COLUMN FAMILY (meta, data)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(
                    ct.column_families,
                    vec!["meta".to_string(), "data".to_string()]
                );
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_column_families_combined() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE t (id INT PRIMARY KEY, a TEXT FAMILY cf1, b TEXT FAMILY cf2) WITH COLUMN FAMILY (cf1, cf2)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(ct.columns[1].column_family, Some("cf1".to_string()));
                assert_eq!(ct.columns[2].column_family, Some("cf2".to_string()));
                assert_eq!(ct.column_families, vec!["cf1", "cf2"]);
            }
            _ => panic!("Expected CREATE TABLE"),
        }
    }

    #[test]
    fn test_parse_family_keyword_as_identifier() {
        let stmt = SqlParser::new().parse("SELECT family FROM t").unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                assert!(matches!(&q.columns[0].expr, Expression::Column(s) if s == "family"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ============================================================
    // Vector SQL Parser Tests
    // ============================================================

    #[test]
    fn test_parse_create_vector_index_basic() {
        let stmt = SqlParser::new()
            .parse("CREATE VECTOR INDEX idx_embed ON items (embedding)")
            .unwrap();
        match stmt {
            SqlStatement::CreateVectorIndex {
                name,
                table,
                column,
                method,
                options,
            } => {
                assert_eq!(name, "idx_embed");
                assert_eq!(table, "items");
                assert_eq!(column, "embedding");
                assert_eq!(method, "HNSW");
                assert!(options.is_empty());
            }
            _ => panic!("Expected CreateVectorIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_vector_index_with_method_and_options() {
        let stmt = SqlParser::new()
            .parse("CREATE VECTOR INDEX idx_v ON docs (vec) USING LSH WITH (num_tables = 8, num_bits = 128)")
            .unwrap();
        match stmt {
            SqlStatement::CreateVectorIndex {
                name,
                table,
                column,
                method,
                options,
            } => {
                assert_eq!(name, "idx_v");
                assert_eq!(table, "docs");
                assert_eq!(column, "vec");
                assert_eq!(method, "LSH");
                assert_eq!(options.get("num_tables"), Some(&"8".to_string()));
                assert_eq!(options.get("num_bits"), Some(&"128".to_string()));
            }
            _ => panic!("Expected CreateVectorIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_drop_vector_index() {
        let stmt = SqlParser::new()
            .parse("DROP VECTOR INDEX idx_embed")
            .unwrap();
        match stmt {
            SqlStatement::DropVectorIndex(name) => {
                assert_eq!(name, "idx_embed");
            }
            _ => panic!("Expected DropVectorIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_drop_vector_index_if_exists() {
        let stmt = SqlParser::new()
            .parse("DROP VECTOR INDEX IF EXISTS idx_embed")
            .unwrap();
        match stmt {
            SqlStatement::DropVectorIndex(name) => {
                assert_eq!(name, "idx_embed");
            }
            _ => panic!("Expected DropVectorIndex, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_vector_keyword_as_identifier() {
        let stmt = SqlParser::new().parse("SELECT vector FROM t").unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                assert!(matches!(&q.columns[0].expr, Expression::Column(s) if s == "vector"));
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_vector_distance_function() {
        let stmt = SqlParser::new()
            .parse("SELECT L2_DISTANCE(a, b) FROM embeddings")
            .unwrap();
        match stmt {
            SqlStatement::Select(q) => {
                assert_eq!(q.columns.len(), 1);
                match &q.columns[0].expr {
                    Expression::Function { name, args } => {
                        assert_eq!(name, "L2_DISTANCE");
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("Expected Function"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    // ========== Auth/RBAC Parser Tests ==========

    #[test]
    fn test_parse_grant() {
        let stmt = SqlParser::new()
            .parse("GRANT SELECT, INSERT ON users TO alice")
            .unwrap();
        match stmt {
            SqlStatement::Grant {
                permissions,
                resource,
                grantee,
            } => {
                assert_eq!(permissions, vec!["SELECT", "INSERT"]);
                assert_eq!(resource, "users");
                assert_eq!(grantee, "alice");
            }
            _ => panic!("Expected Grant, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_revoke() {
        let stmt = SqlParser::new()
            .parse("REVOKE DELETE ON orders FROM bob")
            .unwrap();
        match stmt {
            SqlStatement::Revoke {
                permissions,
                resource,
                grantee,
            } => {
                assert_eq!(permissions, vec!["DELETE"]);
                assert_eq!(resource, "orders");
                assert_eq!(grantee, "bob");
            }
            _ => panic!("Expected Revoke, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_user_with_password() {
        let stmt = SqlParser::new()
            .parse("CREATE USER alice WITH PASSWORD 'secret123'")
            .unwrap();
        match stmt {
            SqlStatement::CreateUser { name, password } => {
                assert_eq!(name, "alice");
                assert_eq!(password, Some("secret123".to_string()));
            }
            _ => panic!("Expected CreateUser, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_user_no_password() {
        let stmt = SqlParser::new().parse("CREATE USER bob").unwrap();
        match stmt {
            SqlStatement::CreateUser { name, password } => {
                assert_eq!(name, "bob");
                assert_eq!(password, None);
            }
            _ => panic!("Expected CreateUser, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_drop_user() {
        let stmt = SqlParser::new().parse("DROP USER alice").unwrap();
        match stmt {
            SqlStatement::DropUser(name) => assert_eq!(name, "alice"),
            _ => panic!("Expected DropUser, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_role() {
        let stmt = SqlParser::new().parse("CREATE ROLE analyst").unwrap();
        match stmt {
            SqlStatement::CreateRole(name) => assert_eq!(name, "analyst"),
            _ => panic!("Expected CreateRole, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_drop_role() {
        let stmt = SqlParser::new().parse("DROP ROLE analyst").unwrap();
        match stmt {
            SqlStatement::DropRole(name) => assert_eq!(name, "analyst"),
            _ => panic!("Expected DropRole, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_alter_user_with_password() {
        let stmt = SqlParser::new()
            .parse("ALTER USER alice WITH PASSWORD 'newpass123'")
            .unwrap();
        match stmt {
            SqlStatement::AlterUser { name, password } => {
                assert_eq!(name, "alice");
                assert_eq!(password, "newpass123");
            }
            _ => panic!("Expected AlterUser, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_alter_user_password_shorthand() {
        let stmt = SqlParser::new()
            .parse("ALTER USER bob PASSWORD 'secret'")
            .unwrap();
        match stmt {
            SqlStatement::AlterUser { name, password } => {
                assert_eq!(name, "bob");
                assert_eq!(password, "secret");
            }
            _ => panic!("Expected AlterUser, got {:?}", stmt),
        }
    }

    // --- Shard BY Tests (Session 60) ---

    #[test]
    fn test_parse_create_table_shard_by() {
        let stmt = SqlParser::new()
            .parse(
                "CREATE TABLE orders (id INT, customer_id INT, amount REAL) SHARD BY (customer_id)",
            )
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(ct.name, "orders");
                assert_eq!(ct.columns.len(), 3);
                assert_eq!(ct.shard_key, Some("customer_id".to_string()));
            }
            _ => panic!("Expected CreateTable, got {:?}", stmt),
        }
    }

    #[test]
    fn test_parse_create_table_no_shard() {
        let stmt = SqlParser::new()
            .parse("CREATE TABLE users (id INT, name TEXT)")
            .unwrap();
        match stmt {
            SqlStatement::CreateTable(ct) => {
                assert_eq!(ct.name, "users");
                assert_eq!(ct.shard_key, None);
            }
            _ => panic!("Expected CreateTable, got {:?}", stmt),
        }
    }

    #[test]
    fn test_shard_as_column_name() {
        let stmt = SqlParser::new()
            .parse("SELECT shard FROM routing_table WHERE shard = 'shard_001'")
            .unwrap();
        match stmt {
            SqlStatement::Select(_) => {} // Just need it to parse
            _ => panic!("Expected Select, got {:?}", stmt),
        }
    }
}
