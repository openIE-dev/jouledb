//! CQL (Cassandra Query Language) Parser
//!
//! CQL is the query language for Apache Cassandra and compatible wide-column stores.

use crate::ast::{Expression, Operator, Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;

/// CQL query
#[derive(Debug, Clone)]
pub struct CqlQuery {
    pub statement: CqlStatement,
}

impl CqlQuery {
    /// Convert to generic Query
    pub fn to_query(&self) -> Query {
        match &self.statement {
            CqlStatement::Select(s) => Query {
                query_type: QueryType::Select,
                source: Some(format!(
                    "{}.{}",
                    s.keyspace.as_deref().unwrap_or(""),
                    s.table
                )),
                columns: s.columns.clone(),
                filter: s.where_clause.clone(),
                order_by: Vec::new(),
                group_by: Vec::new(),
                having: None,
                limit: s.limit,
                offset: None,
                joins: Vec::new(),
                values: Vec::new(),
                returning: Vec::new(),
                ctes: Vec::new(),
                derived_columns: HashMap::new(),
                distinct: false, source_alias: None,
            },
            _ => Query {
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
            },
        }
    }
}

/// CQL statement types
#[derive(Debug, Clone)]
pub enum CqlStatement {
    Select(CqlSelect),
    Insert(CqlInsert),
    Update(CqlUpdate),
    Delete(CqlDelete),
    CreateKeyspace(CqlCreateKeyspace),
    CreateTable(CqlCreateTable),
    CreateIndex(CqlCreateIndex),
    DropKeyspace(String),
    DropTable(CqlDropTable),
    DropIndex(String),
    Truncate(CqlDropTable),
    Use(String),
    Batch(Vec<CqlStatement>),
}

/// CQL SELECT statement
#[derive(Debug, Clone)]
pub struct CqlSelect {
    pub columns: Vec<String>,
    pub keyspace: Option<String>,
    pub table: String,
    pub where_clause: Option<Expression>,
    pub order_by: Vec<(String, bool)>,
    pub limit: Option<usize>,
    pub allow_filtering: bool,
    pub distinct: bool,
}

/// CQL INSERT statement
#[derive(Debug, Clone)]
pub struct CqlInsert {
    pub keyspace: Option<String>,
    pub table: String,
    pub columns: Vec<String>,
    pub values: Vec<Expression>,
    pub if_not_exists: bool,
    pub ttl: Option<u64>,
    pub timestamp: Option<u64>,
}

/// CQL UPDATE statement
#[derive(Debug, Clone)]
pub struct CqlUpdate {
    pub keyspace: Option<String>,
    pub table: String,
    pub assignments: Vec<CqlAssignment>,
    pub where_clause: Expression,
    pub if_clause: Option<Vec<CqlCondition>>,
    pub ttl: Option<u64>,
    pub timestamp: Option<u64>,
}

/// CQL assignment in UPDATE
#[derive(Debug, Clone)]
pub enum CqlAssignment {
    Set(String, Expression),
    Increment(String, Expression),
    Decrement(String, Expression),
    Append(String, Expression),
    Prepend(String, Expression),
    RemoveFromList(String, Expression),
    AddToSet(String, Expression),
    RemoveFromSet(String, Expression),
    PutInMap(String, Expression, Expression),
    RemoveFromMap(String, Expression),
}

/// CQL condition
#[derive(Debug, Clone)]
pub struct CqlCondition {
    pub column: String,
    pub operator: String,
    pub value: Expression,
}

/// CQL DELETE statement
#[derive(Debug, Clone)]
pub struct CqlDelete {
    pub columns: Vec<String>,
    pub keyspace: Option<String>,
    pub table: String,
    pub where_clause: Expression,
    pub if_clause: Option<Vec<CqlCondition>>,
    pub timestamp: Option<u64>,
}

/// CQL CREATE KEYSPACE statement
#[derive(Debug, Clone)]
pub struct CqlCreateKeyspace {
    pub name: String,
    pub if_not_exists: bool,
    pub replication: HashMap<String, String>,
    pub durable_writes: bool,
}

/// CQL CREATE TABLE statement
#[derive(Debug, Clone)]
pub struct CqlCreateTable {
    pub keyspace: Option<String>,
    pub name: String,
    pub columns: Vec<CqlColumnDef>,
    pub primary_key: CqlPrimaryKey,
    pub if_not_exists: bool,
    pub options: HashMap<String, String>,
}

/// CQL column definition
#[derive(Debug, Clone)]
pub struct CqlColumnDef {
    pub name: String,
    pub data_type: CqlDataType,
    pub static_column: bool,
}

/// CQL data types
#[derive(Debug, Clone, PartialEq)]
pub enum CqlDataType {
    Ascii,
    Bigint,
    Blob,
    Boolean,
    Counter,
    Date,
    Decimal,
    Double,
    Duration,
    Float,
    Inet,
    Int,
    Smallint,
    Text,
    Time,
    Timestamp,
    Timeuuid,
    Tinyint,
    Uuid,
    Varchar,
    Varint,
    List(Box<CqlDataType>),
    Set(Box<CqlDataType>),
    Map(Box<CqlDataType>, Box<CqlDataType>),
    Tuple(Vec<CqlDataType>),
    Frozen(Box<CqlDataType>),
    UserDefined(String),
}

/// CQL primary key definition
#[derive(Debug, Clone)]
pub struct CqlPrimaryKey {
    pub partition_key: Vec<String>,
    pub clustering_columns: Vec<String>,
}

/// CQL CREATE INDEX statement
#[derive(Debug, Clone)]
pub struct CqlCreateIndex {
    pub name: Option<String>,
    pub keyspace: Option<String>,
    pub table: String,
    pub column: String,
    pub if_not_exists: bool,
    pub using: Option<String>,
}

/// CQL DROP TABLE statement
#[derive(Debug, Clone)]
pub struct CqlDropTable {
    pub keyspace: Option<String>,
    pub table: String,
    pub if_exists: bool,
}

/// CQL Parser
/// Maximum expression nesting depth to prevent stack overflow from crafted inputs.
/// Conservative limit to stay within default stack sizes in debug builds.
const MAX_EXPRESSION_DEPTH: usize = 50;

/// Maximum query length in bytes (1 MB).
const MAX_QUERY_LENGTH: usize = 1_048_576;

pub struct CqlParser {
    input: String,
    pos: usize,
    /// Current expression nesting depth (prevents stack overflow).
    expression_depth: usize,
}

impl CqlParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            input: String::new(),
            pos: 0,
            expression_depth: 0,
        }
    }

    /// Parse CQL query
    pub fn parse(&mut self, cql: &str) -> QueryResult<CqlQuery> {
        if cql.len() > MAX_QUERY_LENGTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Query too long: {} bytes exceeds maximum of {} bytes",
                cql.len(),
                MAX_QUERY_LENGTH
            )));
        }
        self.input = cql.to_string();
        self.pos = 0;
        self.expression_depth = 0;
        self.skip_whitespace();

        let statement = self.parse_statement()?;
        Ok(CqlQuery { statement })
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else if self.input[self.pos..].starts_with("--") {
                while self.pos < self.input.len() && !self.input[self.pos..].starts_with('\n') {
                    let ch = self.input[self.pos..].chars().next().expect("pos < len");
                    self.pos += ch.len_utf8();
                }
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
                .expect("end < len guarantees char exists");
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

    fn consume_keyword(&mut self, expected: &str) -> QueryResult<()> {
        self.skip_whitespace();
        if let Some(kw) = self.peek_keyword() {
            if kw == expected.to_uppercase() {
                self.pos += expected.len();
                self.skip_whitespace();
                return Ok(());
            }
        }
        Err(QueryError::ParseError(format!("Expected {}", expected)))
    }

    fn try_consume_keyword(&mut self, expected: &str) -> bool {
        self.skip_whitespace();
        if let Some(kw) = self.peek_keyword() {
            if kw == expected.to_uppercase() {
                self.pos += expected.len();
                self.skip_whitespace();
                return true;
            }
        }
        false
    }

    fn parse_identifier(&mut self) -> QueryResult<String> {
        self.skip_whitespace();
        let start = self.pos;

        // Handle quoted identifier
        if self.input[self.pos..].starts_with('"') {
            self.pos += 1;
            let inner_start = self.pos;
            while self.pos < self.input.len() && !self.input[self.pos..].starts_with('"') {
                self.pos += 1;
            }
            let result = self.input[inner_start..self.pos].to_string();
            self.pos += 1;
            self.skip_whitespace();
            return Ok(result);
        }

        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
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

    fn parse_table_name(&mut self) -> QueryResult<(Option<String>, String)> {
        let first = self.parse_identifier()?;
        if self.try_consume_char('.') {
            let second = self.parse_identifier()?;
            Ok((Some(first), second))
        } else {
            Ok((None, first))
        }
    }

    fn try_consume_char(&mut self, expected: char) -> bool {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(expected) {
            self.pos += expected.len_utf8();
            self.skip_whitespace();
            true
        } else {
            false
        }
    }

    fn try_consume_str(&mut self, expected: &str) -> bool {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(expected) {
            self.pos += expected.len();
            self.skip_whitespace();
            true
        } else {
            false
        }
    }

    fn consume_char(&mut self, expected: char) -> QueryResult<()> {
        if self.try_consume_char(expected) {
            Ok(())
        } else {
            Err(QueryError::ParseError(format!("Expected '{}'", expected)))
        }
    }

    fn parse_statement(&mut self) -> QueryResult<CqlStatement> {
        match self.peek_keyword().as_deref() {
            Some("SELECT") => self.parse_select(),
            Some("INSERT") => self.parse_insert(),
            Some("UPDATE") => self.parse_update(),
            Some("DELETE") => self.parse_delete(),
            Some("CREATE") => self.parse_create(),
            Some("DROP") => self.parse_drop(),
            Some("TRUNCATE") => self.parse_truncate(),
            Some("USE") => self.parse_use(),
            Some("BEGIN") => self.parse_batch(),
            _ => Err(QueryError::ParseError("Unknown statement".to_string())),
        }
    }

    fn parse_select(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("SELECT")?;

        let distinct = self.try_consume_keyword("DISTINCT");

        let columns = self.parse_select_columns()?;

        self.consume_keyword("FROM")?;
        let (keyspace, table) = self.parse_table_name()?;

        let where_clause = if self.try_consume_keyword("WHERE") {
            Some(self.parse_expression()?)
        } else {
            None
        };

        let order_by = if self.try_consume_keyword("ORDER") {
            self.consume_keyword("BY")?;
            self.parse_order_by()?
        } else {
            Vec::new()
        };

        let limit = if self.try_consume_keyword("LIMIT") {
            Some(self.parse_integer()? as usize)
        } else {
            None
        };

        let allow_filtering =
            self.try_consume_keyword("ALLOW") && self.try_consume_keyword("FILTERING");

        Ok(CqlStatement::Select(CqlSelect {
            columns,
            keyspace,
            table,
            where_clause,
            order_by,
            limit,
            allow_filtering,
            distinct,
        }))
    }

    fn parse_select_columns(&mut self) -> QueryResult<Vec<String>> {
        let mut columns = Vec::new();

        loop {
            if self.try_consume_char('*') {
                columns.push("*".to_string());
            } else {
                let ident = self.parse_identifier()?;
                // Check for function call syntax: ident(...)
                if self.try_consume_char('(') {
                    let mut inner = String::new();
                    let mut depth = 1;
                    while depth > 0 && self.pos < self.input.len() {
                        let ch = self.input.as_bytes()[self.pos] as char;
                        if ch == '(' {
                            depth += 1;
                        } else if ch == ')' {
                            depth -= 1;
                            if depth == 0 {
                                self.pos += 1;
                                break;
                            }
                        }
                        inner.push(ch);
                        self.pos += 1;
                    }
                    self.skip_whitespace();
                    columns.push(format!("{}({})", ident.to_uppercase(), inner.trim()));
                } else {
                    columns.push(ident);
                }
            }

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(columns)
    }

    fn parse_order_by(&mut self) -> QueryResult<Vec<(String, bool)>> {
        let mut orders = Vec::new();

        loop {
            let column = self.parse_identifier()?;
            let desc = self.try_consume_keyword("DESC");
            if !desc {
                self.try_consume_keyword("ASC");
            }
            orders.push((column, desc));

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(orders)
    }

    fn parse_insert(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("INSERT")?;
        self.consume_keyword("INTO")?;

        let (keyspace, table) = self.parse_table_name()?;

        self.consume_char('(')?;
        let columns = self.parse_identifier_list()?;
        self.consume_char(')')?;

        self.consume_keyword("VALUES")?;
        self.consume_char('(')?;
        let values = self.parse_expression_list()?;
        self.consume_char(')')?;

        let if_not_exists = self.try_consume_keyword("IF")
            && self.try_consume_keyword("NOT")
            && self.try_consume_keyword("EXISTS");

        let ttl = if self.try_consume_keyword("USING") && self.try_consume_keyword("TTL") {
            Some(self.parse_integer()? as u64)
        } else {
            None
        };

        Ok(CqlStatement::Insert(CqlInsert {
            keyspace,
            table,
            columns,
            values,
            if_not_exists,
            ttl,
            timestamp: None,
        }))
    }

    fn parse_update(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("UPDATE")?;

        let (keyspace, table) = self.parse_table_name()?;

        let ttl = if self.try_consume_keyword("USING") && self.try_consume_keyword("TTL") {
            Some(self.parse_integer()? as u64)
        } else {
            None
        };

        self.consume_keyword("SET")?;
        let assignments = self.parse_assignments()?;

        self.consume_keyword("WHERE")?;
        let where_clause = self.parse_expression()?;

        let if_clause = if self.try_consume_keyword("IF") {
            Some(self.parse_conditions()?)
        } else {
            None
        };

        Ok(CqlStatement::Update(CqlUpdate {
            keyspace,
            table,
            assignments,
            where_clause,
            if_clause,
            ttl,
            timestamp: None,
        }))
    }

    fn parse_assignments(&mut self) -> QueryResult<Vec<CqlAssignment>> {
        let mut assignments = Vec::new();

        loop {
            let column = self.parse_identifier()?;
            self.consume_char('=')?;

            // Try to detect counter pattern: col = col +/- value
            let saved_pos = self.pos;
            self.skip_whitespace();
            if let Some(rhs_ident) = self.peek_keyword() {
                if rhs_ident.to_lowercase() == column.to_lowercase() {
                    // Speculatively consume the identifier
                    let ident_end = self.pos + rhs_ident.len();
                    let after_ident = &self.input[ident_end..].trim_start();
                    if after_ident.starts_with('+') || after_ident.starts_with('-') {
                        let is_add = after_ident.starts_with('+');
                        // Commit: consume identifier and operator
                        self.pos = ident_end;
                        self.skip_whitespace();
                        self.pos += 1; // consume + or -
                        self.skip_whitespace();
                        let value = self.parse_expression()?;
                        if is_add {
                            assignments.push(CqlAssignment::Increment(column, value));
                        } else {
                            assignments.push(CqlAssignment::Decrement(column, value));
                        }
                        if !self.try_consume_char(',') {
                            break;
                        }
                        continue;
                    }
                }
            }
            // Not a counter pattern — restore and parse normally
            self.pos = saved_pos;
            let value = self.parse_expression()?;
            assignments.push(CqlAssignment::Set(column, value));

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(assignments)
    }

    fn parse_conditions(&mut self) -> QueryResult<Vec<CqlCondition>> {
        let mut conditions = Vec::new();

        loop {
            let column = self.parse_identifier()?;
            self.consume_char('=')?;
            let value = self.parse_expression()?;
            conditions.push(CqlCondition {
                column,
                operator: "=".to_string(),
                value,
            });

            if !self.try_consume_keyword("AND") {
                break;
            }
        }

        Ok(conditions)
    }

    fn parse_delete(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("DELETE")?;

        let columns = if !self.try_consume_keyword("FROM") {
            let cols = self.parse_identifier_list()?;
            self.consume_keyword("FROM")?;
            cols
        } else {
            Vec::new()
        };

        let (keyspace, table) = self.parse_table_name()?;

        self.consume_keyword("WHERE")?;
        let where_clause = self.parse_expression()?;

        let if_clause = if self.try_consume_keyword("IF") {
            Some(self.parse_conditions()?)
        } else {
            None
        };

        Ok(CqlStatement::Delete(CqlDelete {
            columns,
            keyspace,
            table,
            where_clause,
            if_clause,
            timestamp: None,
        }))
    }

    fn parse_create(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("CREATE")?;

        if self.try_consume_keyword("KEYSPACE") {
            let if_not_exists = self.try_consume_keyword("IF")
                && self.try_consume_keyword("NOT")
                && self.try_consume_keyword("EXISTS");
            let name = self.parse_identifier()?;

            self.consume_keyword("WITH")?;
            self.consume_keyword("REPLICATION")?;
            self.consume_char('=')?;
            let replication = self.parse_map_literal()?;

            let durable_writes =
                if self.try_consume_keyword("AND") && self.try_consume_keyword("DURABLE_WRITES") {
                    self.consume_char('=')?;
                    self.try_consume_keyword("TRUE")
                } else {
                    true
                };

            Ok(CqlStatement::CreateKeyspace(CqlCreateKeyspace {
                name,
                if_not_exists,
                replication,
                durable_writes,
            }))
        } else if self.try_consume_keyword("TABLE") {
            let if_not_exists = self.try_consume_keyword("IF")
                && self.try_consume_keyword("NOT")
                && self.try_consume_keyword("EXISTS");
            let (keyspace, name) = self.parse_table_name()?;

            self.consume_char('(')?;
            let (columns, primary_key) = self.parse_table_columns()?;
            self.consume_char(')')?;

            let options = if self.try_consume_keyword("WITH") {
                self.parse_table_options()?
            } else {
                HashMap::new()
            };

            Ok(CqlStatement::CreateTable(CqlCreateTable {
                keyspace,
                name,
                columns,
                primary_key,
                if_not_exists,
                options,
            }))
        } else if self.try_consume_keyword("INDEX") {
            let if_not_exists = self.try_consume_keyword("IF")
                && self.try_consume_keyword("NOT")
                && self.try_consume_keyword("EXISTS");

            let name = if !self.try_consume_keyword("ON") {
                let n = self.parse_identifier()?;
                self.consume_keyword("ON")?;
                Some(n)
            } else {
                None
            };

            let (keyspace, table) = self.parse_table_name()?;
            self.consume_char('(')?;
            let column = self.parse_identifier()?;
            self.consume_char(')')?;

            let using = if self.try_consume_keyword("USING") {
                Some(self.parse_identifier()?)
            } else {
                None
            };

            Ok(CqlStatement::CreateIndex(CqlCreateIndex {
                name,
                keyspace,
                table,
                column,
                if_not_exists,
                using,
            }))
        } else {
            Err(QueryError::ParseError(
                "Expected KEYSPACE, TABLE, or INDEX".to_string(),
            ))
        }
    }

    fn parse_table_columns(&mut self) -> QueryResult<(Vec<CqlColumnDef>, CqlPrimaryKey)> {
        let mut columns = Vec::new();
        let mut primary_key = CqlPrimaryKey {
            partition_key: Vec::new(),
            clustering_columns: Vec::new(),
        };

        loop {
            if self.try_consume_keyword("PRIMARY") {
                self.consume_keyword("KEY")?;
                self.consume_char('(')?;

                // Partition key
                if self.try_consume_char('(') {
                    primary_key.partition_key = self.parse_identifier_list()?;
                    self.consume_char(')')?;
                } else {
                    primary_key.partition_key.push(self.parse_identifier()?);
                }

                // Clustering columns
                if self.try_consume_char(',') {
                    primary_key.clustering_columns = self.parse_identifier_list()?;
                }

                self.consume_char(')')?;
            } else {
                let name = self.parse_identifier()?;
                let data_type = self.parse_data_type()?;
                let static_column = self.try_consume_keyword("STATIC");

                if self.try_consume_keyword("PRIMARY") {
                    self.consume_keyword("KEY")?;
                    primary_key.partition_key.push(name.clone());
                }

                columns.push(CqlColumnDef {
                    name,
                    data_type,
                    static_column,
                });
            }

            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok((columns, primary_key))
    }

    fn parse_data_type(&mut self) -> QueryResult<CqlDataType> {
        let type_name = self.parse_identifier()?.to_uppercase();

        match type_name.as_str() {
            "ASCII" => Ok(CqlDataType::Ascii),
            "BIGINT" => Ok(CqlDataType::Bigint),
            "BLOB" => Ok(CqlDataType::Blob),
            "BOOLEAN" => Ok(CqlDataType::Boolean),
            "COUNTER" => Ok(CqlDataType::Counter),
            "DATE" => Ok(CqlDataType::Date),
            "DECIMAL" => Ok(CqlDataType::Decimal),
            "DOUBLE" => Ok(CqlDataType::Double),
            "DURATION" => Ok(CqlDataType::Duration),
            "FLOAT" => Ok(CqlDataType::Float),
            "INET" => Ok(CqlDataType::Inet),
            "INT" => Ok(CqlDataType::Int),
            "SMALLINT" => Ok(CqlDataType::Smallint),
            "TEXT" => Ok(CqlDataType::Text),
            "TIME" => Ok(CqlDataType::Time),
            "TIMESTAMP" => Ok(CqlDataType::Timestamp),
            "TIMEUUID" => Ok(CqlDataType::Timeuuid),
            "TINYINT" => Ok(CqlDataType::Tinyint),
            "UUID" => Ok(CqlDataType::Uuid),
            "VARCHAR" => Ok(CqlDataType::Varchar),
            "VARINT" => Ok(CqlDataType::Varint),
            "LIST" => {
                self.consume_char('<')?;
                let inner = self.parse_data_type()?;
                self.consume_char('>')?;
                Ok(CqlDataType::List(Box::new(inner)))
            }
            "SET" => {
                self.consume_char('<')?;
                let inner = self.parse_data_type()?;
                self.consume_char('>')?;
                Ok(CqlDataType::Set(Box::new(inner)))
            }
            "MAP" => {
                self.consume_char('<')?;
                let key = self.parse_data_type()?;
                self.consume_char(',')?;
                let value = self.parse_data_type()?;
                self.consume_char('>')?;
                Ok(CqlDataType::Map(Box::new(key), Box::new(value)))
            }
            "FROZEN" => {
                self.consume_char('<')?;
                let inner = self.parse_data_type()?;
                self.consume_char('>')?;
                Ok(CqlDataType::Frozen(Box::new(inner)))
            }
            _ => Ok(CqlDataType::UserDefined(type_name)),
        }
    }

    fn parse_table_options(&mut self) -> QueryResult<HashMap<String, String>> {
        let mut options = HashMap::new();

        loop {
            let key = self.parse_identifier()?;
            self.consume_char('=')?;
            let value = self.parse_identifier()?;
            options.insert(key, value);

            if !self.try_consume_keyword("AND") {
                break;
            }
        }

        Ok(options)
    }

    fn parse_drop(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("DROP")?;

        if self.try_consume_keyword("KEYSPACE") {
            let name = self.parse_identifier()?;
            Ok(CqlStatement::DropKeyspace(name))
        } else if self.try_consume_keyword("TABLE") {
            let if_exists = self.try_consume_keyword("IF") && self.try_consume_keyword("EXISTS");
            let (keyspace, table) = self.parse_table_name()?;
            Ok(CqlStatement::DropTable(CqlDropTable {
                keyspace,
                table,
                if_exists,
            }))
        } else if self.try_consume_keyword("INDEX") {
            let name = self.parse_identifier()?;
            Ok(CqlStatement::DropIndex(name))
        } else {
            Err(QueryError::ParseError(
                "Expected KEYSPACE, TABLE, or INDEX".to_string(),
            ))
        }
    }

    fn parse_truncate(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("TRUNCATE")?;
        self.try_consume_keyword("TABLE");
        let (keyspace, table) = self.parse_table_name()?;
        Ok(CqlStatement::Truncate(CqlDropTable {
            keyspace,
            table,
            if_exists: false,
        }))
    }

    fn parse_use(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("USE")?;
        let keyspace = self.parse_identifier()?;
        Ok(CqlStatement::Use(keyspace))
    }

    fn parse_batch(&mut self) -> QueryResult<CqlStatement> {
        self.consume_keyword("BEGIN")?;
        self.try_consume_keyword("UNLOGGED");
        self.try_consume_keyword("COUNTER");
        self.consume_keyword("BATCH")?;

        let mut statements = Vec::new();
        while !self.try_consume_keyword("APPLY") {
            statements.push(self.parse_statement()?);
            self.try_consume_char(';');
        }

        self.consume_keyword("BATCH")?;
        Ok(CqlStatement::Batch(statements))
    }

    fn parse_identifier_list(&mut self) -> QueryResult<Vec<String>> {
        let mut list = Vec::new();

        loop {
            list.push(self.parse_identifier()?);
            if !self.try_consume_char(',') {
                break;
            }
        }

        Ok(list)
    }

    fn parse_expression_list(&mut self) -> QueryResult<Vec<Expression>> {
        let mut list = Vec::new();

        loop {
            list.push(self.parse_expression()?);
            if !self.try_consume_char(',') {
                break;
            }
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
        let result = self.parse_or_expression();
        self.expression_depth -= 1;
        result
    }

    fn parse_or_expression(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_and_expression()?;

        while self.try_consume_keyword("OR") {
            let right = self.parse_and_expression()?;
            left = Expression::or(left, right);
        }

        Ok(left)
    }

    fn parse_and_expression(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_comparison()?;

        while self.try_consume_keyword("AND") {
            let right = self.parse_comparison()?;
            left = Expression::and(left, right);
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> QueryResult<Expression> {
        let left = self.parse_primary()?;

        self.skip_whitespace();
        if self.try_consume_str("!=") {
            let right = self.parse_primary()?;
            Ok(Expression::binary(left, Operator::Ne, right))
        } else if self.try_consume_str(">=") {
            let right = self.parse_primary()?;
            Ok(Expression::binary(left, Operator::Ge, right))
        } else if self.try_consume_str("<=") {
            let right = self.parse_primary()?;
            Ok(Expression::binary(left, Operator::Le, right))
        } else if self.try_consume_char('>') {
            let right = self.parse_primary()?;
            Ok(Expression::binary(left, Operator::Gt, right))
        } else if self.try_consume_char('<') {
            let right = self.parse_primary()?;
            Ok(Expression::binary(left, Operator::Lt, right))
        } else if self.try_consume_char('=') {
            let right = self.parse_primary()?;
            Ok(Expression::eq(left, right))
        } else if self.try_consume_keyword("IN") {
            self.consume_char('(')?;
            let list = self.parse_expression_list()?;
            self.consume_char(')')?;
            Ok(Expression::In {
                expr: Box::new(left),
                list,
                negated: false,
            })
        } else {
            Ok(left)
        }
    }

    fn parse_primary(&mut self) -> QueryResult<Expression> {
        self.skip_whitespace();

        // String literal
        if self.input[self.pos..].starts_with('\'') {
            self.pos += 1;
            let start = self.pos;
            while self.pos < self.input.len() && !self.input[self.pos..].starts_with('\'') {
                self.pos += 1;
            }
            let s = self.input[start..self.pos].to_string();
            self.pos += 1;
            self.skip_whitespace();
            return Ok(Expression::Literal(Value::String(s)));
        }

        // Number
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c.is_ascii_digit() || c == '.' || c == '-' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        if self.pos > start {
            let num_str = self.input[start..self.pos].to_string();
            self.skip_whitespace();
            if num_str.contains('.') {
                return Ok(Expression::Literal(Value::Float(
                    num_str.parse().unwrap_or(0.0),
                )));
            } else {
                return Ok(Expression::Literal(Value::Int(
                    num_str.parse().unwrap_or(0),
                )));
            }
        }

        // Identifier
        let ident = self.parse_identifier()?;
        if ident.to_uppercase() == "TRUE" {
            return Ok(Expression::Literal(Value::Bool(true)));
        }
        if ident.to_uppercase() == "FALSE" {
            return Ok(Expression::Literal(Value::Bool(false)));
        }
        if ident.to_uppercase() == "NULL" {
            return Ok(Expression::Literal(Value::Null));
        }

        Ok(Expression::Column(ident))
    }

    fn parse_integer(&mut self) -> QueryResult<i64> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c.is_ascii_digit() || c == '-' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        let num_str = self.input[start..self.pos].to_string();
        self.skip_whitespace();
        num_str
            .parse()
            .map_err(|_| QueryError::ParseError("Expected integer".to_string()))
    }

    fn parse_map_literal(&mut self) -> QueryResult<HashMap<String, String>> {
        self.consume_char('{')?;
        let mut map = HashMap::new();

        if !self.try_consume_char('}') {
            loop {
                self.consume_char('\'')?;
                let key = self.parse_until('\'')?;
                self.consume_char('\'')?;
                self.consume_char(':')?;
                self.consume_char('\'')?;
                let value = self.parse_until('\'')?;
                self.consume_char('\'')?;
                map.insert(key, value);

                if !self.try_consume_char(',') {
                    break;
                }
            }
            self.consume_char('}')?;
        }

        Ok(map)
    }

    fn parse_until(&mut self, c: char) -> QueryResult<String> {
        let start = self.pos;
        while self.pos < self.input.len() && !self.input[self.pos..].starts_with(c) {
            self.pos += 1;
        }
        Ok(self.input[start..self.pos].to_string())
    }
}

impl Default for CqlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_select() {
        let mut parser = CqlParser::new();
        let query = parser.parse("SELECT * FROM users").unwrap();

        match query.statement {
            CqlStatement::Select(s) => {
                assert_eq!(s.table, "users");
                assert_eq!(s.columns, vec!["*"]);
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_with_keyspace() {
        let mut parser = CqlParser::new();
        let query = parser.parse("SELECT id, name FROM myks.users").unwrap();

        match query.statement {
            CqlStatement::Select(s) => {
                assert_eq!(s.keyspace, Some("myks".to_string()));
                assert_eq!(s.table, "users");
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_select_with_where() {
        let mut parser = CqlParser::new();
        let query = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();

        match query.statement {
            CqlStatement::Select(s) => {
                assert!(s.where_clause.is_some());
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_insert() {
        let mut parser = CqlParser::new();
        let query = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();

        match query.statement {
            CqlStatement::Insert(i) => {
                assert_eq!(i.table, "users");
                assert_eq!(i.columns, vec!["id", "name"]);
            }
            _ => panic!("Expected INSERT"),
        }
    }

    #[test]
    fn test_update() {
        let mut parser = CqlParser::new();
        let query = parser
            .parse("UPDATE users SET name = 'Bob' WHERE id = 1")
            .unwrap();

        match query.statement {
            CqlStatement::Update(u) => {
                assert_eq!(u.table, "users");
                assert_eq!(u.assignments.len(), 1);
            }
            _ => panic!("Expected UPDATE"),
        }
    }

    #[test]
    fn test_create_keyspace() {
        let mut parser = CqlParser::new();
        let query = parser
            .parse("CREATE KEYSPACE myks WITH REPLICATION = {'class': 'SimpleStrategy'}")
            .unwrap();

        match query.statement {
            CqlStatement::CreateKeyspace(c) => {
                assert_eq!(c.name, "myks");
            }
            _ => panic!("Expected CREATE KEYSPACE"),
        }
    }

    #[test]
    fn test_use() {
        let mut parser = CqlParser::new();
        let query = parser.parse("USE myks").unwrap();

        match query.statement {
            CqlStatement::Use(ks) => {
                assert_eq!(ks, "myks");
            }
            _ => panic!("Expected USE"),
        }
    }
}
