//! SQL query builder — select, insert, update, delete with parameterized queries.
//!
//! Replaces Knex query builder, Kysely, and Drizzle ORM with pure Rust.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Query builder errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryBuilderError {
    /// Missing required clause (e.g. table name).
    MissingClause(String),
    /// Empty column list.
    EmptyColumns,
}

impl fmt::Display for QueryBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingClause(clause) => write!(f, "missing required clause: {clause}"),
            Self::EmptyColumns => write!(f, "column list is empty"),
        }
    }
}

impl std::error::Error for QueryBuilderError {}

// ── Param Value ─────────────────────────────────────────────────

/// A bound parameter value.
#[derive(Debug, Clone, PartialEq)]
pub enum Param {
    Text(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Null,
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "'{s}'"),
            Self::Integer(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Null => write!(f, "NULL"),
        }
    }
}

// ── Sort Direction ──────────────────────────────────────────────

/// Sort direction for ORDER BY.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDir {
    Asc,
    Desc,
}

impl fmt::Display for SortDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asc => write!(f, "ASC"),
            Self::Desc => write!(f, "DESC"),
        }
    }
}

// ── Join Type ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

impl fmt::Display for JoinType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inner => write!(f, "JOIN"),
            Self::Left => write!(f, "LEFT JOIN"),
            Self::Right => write!(f, "RIGHT JOIN"),
            Self::Full => write!(f, "FULL JOIN"),
        }
    }
}

// ── Join Clause ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct JoinClause {
    join_type: JoinType,
    table: String,
    on_condition: String,
}

// ── SELECT Builder ──────────────────────────────────────────────

/// Builds a SELECT query.
#[derive(Debug, Clone, Default)]
pub struct SelectBuilder {
    columns: Vec<String>,
    table: Option<String>,
    joins: Vec<JoinClause>,
    where_clauses: Vec<String>,
    group_by_cols: Vec<String>,
    having_clause: Option<String>,
    order_by_cols: Vec<(String, SortDir)>,
    limit_val: Option<u64>,
    offset_val: Option<u64>,
    params: Vec<Param>,
    union_queries: Vec<String>,
}

impl SelectBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select(mut self, columns: &[&str]) -> Self {
        self.columns = columns.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn select_all(mut self) -> Self {
        self.columns = vec!["*".to_string()];
        self
    }

    pub fn from(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    pub fn where_clause(mut self, condition: &str) -> Self {
        self.where_clauses.push(condition.to_string());
        self
    }

    pub fn where_param(mut self, condition: &str, param: Param) -> Self {
        self.where_clauses.push(condition.to_string());
        self.params.push(param);
        self
    }

    pub fn join(mut self, table: &str, on: &str) -> Self {
        self.joins.push(JoinClause {
            join_type: JoinType::Inner,
            table: table.to_string(),
            on_condition: on.to_string(),
        });
        self
    }

    pub fn left_join(mut self, table: &str, on: &str) -> Self {
        self.joins.push(JoinClause {
            join_type: JoinType::Left,
            table: table.to_string(),
            on_condition: on.to_string(),
        });
        self
    }

    pub fn group_by(mut self, col: &str) -> Self {
        self.group_by_cols.push(col.to_string());
        self
    }

    pub fn having(mut self, condition: &str) -> Self {
        self.having_clause = Some(condition.to_string());
        self
    }

    pub fn order_by(mut self, col: &str, dir: SortDir) -> Self {
        self.order_by_cols.push((col.to_string(), dir));
        self
    }

    pub fn limit(mut self, n: u64) -> Self {
        self.limit_val = Some(n);
        self
    }

    pub fn offset(mut self, n: u64) -> Self {
        self.offset_val = Some(n);
        self
    }

    pub fn union(mut self, sql: &str) -> Self {
        self.union_queries.push(sql.to_string());
        self
    }

    /// Build to (sql, params).
    pub fn build(self) -> Result<(String, Vec<Param>), QueryBuilderError> {
        let table = self
            .table
            .as_deref()
            .ok_or_else(|| QueryBuilderError::MissingClause("FROM".into()))?;

        let cols = if self.columns.is_empty() {
            "*".to_string()
        } else {
            self.columns.join(", ")
        };

        let mut sql = format!("SELECT {cols} FROM {table}");

        for j in &self.joins {
            sql.push_str(&format!(" {} {} ON {}", j.join_type, j.table, j.on_condition));
        }

        if !self.where_clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&self.where_clauses.join(" AND "));
        }

        if !self.group_by_cols.is_empty() {
            sql.push_str(" GROUP BY ");
            sql.push_str(&self.group_by_cols.join(", "));
        }

        if let Some(having) = &self.having_clause {
            sql.push_str(&format!(" HAVING {having}"));
        }

        if !self.order_by_cols.is_empty() {
            sql.push_str(" ORDER BY ");
            let parts: Vec<String> = self
                .order_by_cols
                .iter()
                .map(|(col, dir)| format!("{col} {dir}"))
                .collect();
            sql.push_str(&parts.join(", "));
        }

        if let Some(limit) = self.limit_val {
            sql.push_str(&format!(" LIMIT {limit}"));
        }

        if let Some(offset) = self.offset_val {
            sql.push_str(&format!(" OFFSET {offset}"));
        }

        for union_sql in &self.union_queries {
            sql.push_str(&format!(" UNION {union_sql}"));
        }

        Ok((sql, self.params))
    }
}

// ── INSERT Builder ──────────────────────────────────────────────

/// Builds an INSERT query.
#[derive(Debug, Clone, Default)]
pub struct InsertBuilder {
    table: Option<String>,
    columns: Vec<String>,
    rows: Vec<Vec<Param>>,
}

impl InsertBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn into_table(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = cols.iter().map(|s| (*s).to_string()).collect();
        self
    }

    pub fn values(mut self, vals: Vec<Param>) -> Self {
        self.rows.push(vals);
        self
    }

    pub fn build(self) -> Result<(String, Vec<Param>), QueryBuilderError> {
        let table = self
            .table
            .as_deref()
            .ok_or_else(|| QueryBuilderError::MissingClause("INTO".into()))?;

        if self.columns.is_empty() {
            return Err(QueryBuilderError::EmptyColumns);
        }

        let cols = self.columns.join(", ");
        let mut all_params = Vec::new();
        let mut value_groups = Vec::new();

        for row in &self.rows {
            let placeholders: Vec<&str> = row.iter().map(|_| "?").collect();
            value_groups.push(format!("({})", placeholders.join(", ")));
            all_params.extend(row.iter().cloned());
        }

        let sql = format!(
            "INSERT INTO {table} ({cols}) VALUES {}",
            value_groups.join(", ")
        );
        Ok((sql, all_params))
    }
}

// ── UPDATE Builder ──────────────────────────────────────────────

/// Builds an UPDATE query.
#[derive(Debug, Clone, Default)]
pub struct UpdateBuilder {
    table: Option<String>,
    sets: Vec<String>,
    where_clauses: Vec<String>,
    params: Vec<Param>,
}

impl UpdateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn table(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    pub fn set(mut self, col: &str, value: Param) -> Self {
        self.sets.push(format!("{col} = ?"));
        self.params.push(value);
        self
    }

    pub fn where_clause(mut self, condition: &str) -> Self {
        self.where_clauses.push(condition.to_string());
        self
    }

    pub fn where_param(mut self, condition: &str, param: Param) -> Self {
        self.where_clauses.push(condition.to_string());
        self.params.push(param);
        self
    }

    pub fn build(self) -> Result<(String, Vec<Param>), QueryBuilderError> {
        let table = self
            .table
            .as_deref()
            .ok_or_else(|| QueryBuilderError::MissingClause("TABLE".into()))?;

        if self.sets.is_empty() {
            return Err(QueryBuilderError::EmptyColumns);
        }

        let mut sql = format!("UPDATE {table} SET {}", self.sets.join(", "));

        if !self.where_clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&self.where_clauses.join(" AND "));
        }

        Ok((sql, self.params))
    }
}

// ── DELETE Builder ──────────────────────────────────────────────

/// Builds a DELETE query.
#[derive(Debug, Clone, Default)]
pub struct DeleteBuilder {
    table: Option<String>,
    where_clauses: Vec<String>,
    params: Vec<Param>,
}

impl DeleteBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    pub fn where_clause(mut self, condition: &str) -> Self {
        self.where_clauses.push(condition.to_string());
        self
    }

    pub fn where_param(mut self, condition: &str, param: Param) -> Self {
        self.where_clauses.push(condition.to_string());
        self.params.push(param);
        self
    }

    pub fn build(self) -> Result<(String, Vec<Param>), QueryBuilderError> {
        let table = self
            .table
            .as_deref()
            .ok_or_else(|| QueryBuilderError::MissingClause("FROM".into()))?;

        let mut sql = format!("DELETE FROM {table}");

        if !self.where_clauses.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&self.where_clauses.join(" AND "));
        }

        Ok((sql, self.params))
    }
}

// ── Subquery helper ─────────────────────────────────────────────

/// Wrap a SELECT as a subquery for use in WHERE ... IN (subquery).
pub fn subquery(builder: SelectBuilder) -> Result<String, QueryBuilderError> {
    let (sql, _) = builder.build()?;
    Ok(format!("({sql})"))
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_basic() {
        let (sql, params) = SelectBuilder::new()
            .select(&["id", "name"])
            .from("users")
            .build()
            .unwrap();
        assert_eq!(sql, "SELECT id, name FROM users");
        assert!(params.is_empty());
    }

    #[test]
    fn select_star() {
        let (sql, _) = SelectBuilder::new()
            .select_all()
            .from("users")
            .build()
            .unwrap();
        assert_eq!(sql, "SELECT * FROM users");
    }

    #[test]
    fn select_with_where() {
        let (sql, params) = SelectBuilder::new()
            .select(&["*"])
            .from("users")
            .where_param("age > ?", Param::Integer(18))
            .build()
            .unwrap();
        assert_eq!(sql, "SELECT * FROM users WHERE age > ?");
        assert_eq!(params, vec![Param::Integer(18)]);
    }

    #[test]
    fn select_with_join() {
        let (sql, _) = SelectBuilder::new()
            .select(&["u.name", "p.title"])
            .from("users u")
            .join("posts p", "p.user_id = u.id")
            .build()
            .unwrap();
        assert!(sql.contains("JOIN posts p ON p.user_id = u.id"));
    }

    #[test]
    fn select_left_join() {
        let (sql, _) = SelectBuilder::new()
            .select(&["*"])
            .from("users u")
            .left_join("orders o", "o.user_id = u.id")
            .build()
            .unwrap();
        assert!(sql.contains("LEFT JOIN orders o ON o.user_id = u.id"));
    }

    #[test]
    fn select_group_by_having() {
        let (sql, _) = SelectBuilder::new()
            .select(&["status", "COUNT(*)"])
            .from("orders")
            .group_by("status")
            .having("COUNT(*) > 5")
            .build()
            .unwrap();
        assert!(sql.contains("GROUP BY status"));
        assert!(sql.contains("HAVING COUNT(*) > 5"));
    }

    #[test]
    fn select_order_limit_offset() {
        let (sql, _) = SelectBuilder::new()
            .select(&["*"])
            .from("users")
            .order_by("name", SortDir::Asc)
            .order_by("age", SortDir::Desc)
            .limit(10)
            .offset(20)
            .build()
            .unwrap();
        assert!(sql.contains("ORDER BY name ASC, age DESC"));
        assert!(sql.contains("LIMIT 10"));
        assert!(sql.contains("OFFSET 20"));
    }

    #[test]
    fn select_union() {
        let (sql, _) = SelectBuilder::new()
            .select(&["name"])
            .from("users")
            .union("SELECT name FROM admins")
            .build()
            .unwrap();
        assert!(sql.contains("UNION SELECT name FROM admins"));
    }

    #[test]
    fn select_missing_from() {
        let err = SelectBuilder::new().select(&["*"]).build().unwrap_err();
        assert_eq!(err, QueryBuilderError::MissingClause("FROM".into()));
    }

    #[test]
    fn insert_basic() {
        let (sql, params) = InsertBuilder::new()
            .into_table("users")
            .columns(&["name", "age"])
            .values(vec![Param::Text("Alice".into()), Param::Integer(30)])
            .build()
            .unwrap();
        assert_eq!(sql, "INSERT INTO users (name, age) VALUES (?, ?)");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn insert_multiple_rows() {
        let (sql, params) = InsertBuilder::new()
            .into_table("users")
            .columns(&["name"])
            .values(vec![Param::Text("Alice".into())])
            .values(vec![Param::Text("Bob".into())])
            .build()
            .unwrap();
        assert_eq!(sql, "INSERT INTO users (name) VALUES (?), (?)");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn update_basic() {
        let (sql, params) = UpdateBuilder::new()
            .table("users")
            .set("name", Param::Text("Bob".into()))
            .set("age", Param::Integer(25))
            .where_param("id = ?", Param::Integer(1))
            .build()
            .unwrap();
        assert_eq!(sql, "UPDATE users SET name = ?, age = ? WHERE id = ?");
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn delete_basic() {
        let (sql, params) = DeleteBuilder::new()
            .from("users")
            .where_param("id = ?", Param::Integer(42))
            .build()
            .unwrap();
        assert_eq!(sql, "DELETE FROM users WHERE id = ?");
        assert_eq!(params, vec![Param::Integer(42)]);
    }

    #[test]
    fn delete_no_where() {
        let (sql, params) = DeleteBuilder::new().from("temp").build().unwrap();
        assert_eq!(sql, "DELETE FROM temp");
        assert!(params.is_empty());
    }

    #[test]
    fn subquery_in_where() {
        let inner = SelectBuilder::new()
            .select(&["id"])
            .from("admins");
        let sub = subquery(inner).unwrap();
        let (sql, _) = SelectBuilder::new()
            .select(&["*"])
            .from("users")
            .where_clause(&format!("id IN {sub}"))
            .build()
            .unwrap();
        assert!(sql.contains("WHERE id IN (SELECT id FROM admins)"));
    }

    #[test]
    fn param_display() {
        assert_eq!(Param::Text("hi".into()).to_string(), "'hi'");
        assert_eq!(Param::Integer(42).to_string(), "42");
        assert_eq!(Param::Null.to_string(), "NULL");
    }
}
