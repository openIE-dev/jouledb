//! Storage-backed SQL Executor
//!
//! Connects SQL queries to the B-tree storage engine from joule-db-core,
//! providing actual data persistence with row serialization.
//!
//! ## Key Format
//!
//! - Row data: `__data__::{table}::{pk_value}`
//!
//! ## Row Serialization
//!
//! Rows are serialized as JSON for simplicity and human readability.

use crate::ast::{Expression, JoinType, Operator, OrderBy, Query, UnaryOperator, Value};
use crate::error::{QueryError, QueryResult};
use crate::execution::{QueryContext, ResultSet, Row};
use crate::sql::{
    SqlCreateIndex, SqlCreateTable, SqlDelete, SqlInsert, SqlParser, SqlStatement, SqlUpdate,
};
use chrono::{Datelike, Timelike};
use joule_db_core::catalog::{Catalog, ColumnDef, DataType, IndexDef, TableSchema};

use joule_db_core::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Prefix for data keys
const DATA_PREFIX: &str = "__data__::";

/// Serialized row representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedRow {
    /// Column names
    pub columns: Vec<String>,
    /// Column values (JSON-compatible)
    pub values: Vec<serde_json::Value>,
}

impl SerializedRow {
    /// Create a new serialized row
    pub fn new(columns: Vec<String>, values: Vec<serde_json::Value>) -> Self {
        Self { columns, values }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> QueryResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| QueryError::ExecutionError(format!("Failed to serialize row: {}", e)))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> QueryResult<Self> {
        serde_json::from_slice(data)
            .map_err(|e| QueryError::ExecutionError(format!("Failed to deserialize row: {}", e)))
    }

    /// Convert Value to JSON value
    pub fn value_to_json(value: &Value) -> serde_json::Value {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Int(i) => serde_json::Value::Number((*i).into()),
            Value::Float(f) => serde_json::Number::from_f64(*f)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(s) => serde_json::Value::String(s.clone()),
            Value::Bytes(b) => serde_json::Value::String(base64_encode(b)),
            Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(Self::value_to_json).collect())
            }
            Value::Object(obj) => {
                let map: serde_json::Map<String, serde_json::Value> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::value_to_json(v)))
                    .collect();
                serde_json::Value::Object(map)
            }
            Value::Timestamp(ts) => serde_json::Value::Number((*ts).into()),
            Value::Uuid(u) => serde_json::Value::String(u.clone()),
            Value::Vector(v) => serde_json::json!(v),
        }
    }

    /// Convert JSON value to Value
    pub fn json_to_value(json: &serde_json::Value) -> Value {
        match json {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Int(i)
                } else if let Some(f) = n.as_f64() {
                    Value::Float(f)
                } else {
                    Value::Null
                }
            }
            serde_json::Value::String(s) => Value::String(s.clone()),
            serde_json::Value::Array(arr) => {
                Value::Array(arr.iter().map(Self::json_to_value).collect())
            }
            serde_json::Value::Object(obj) => {
                let map: HashMap<String, Value> = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), Self::json_to_value(v)))
                    .collect();
                Value::Object(map)
            }
        }
    }

    /// Get value by column name
    pub fn get(&self, column: &str) -> Option<Value> {
        self.columns
            .iter()
            .position(|c| c == column)
            .and_then(|i| self.values.get(i))
            .map(Self::json_to_value)
    }

    /// Convert to RowData for query execution
    pub fn to_row_data(&self) -> RowData {
        RowData {
            columns: self.columns.clone(),
            values: self.values.iter().map(Self::json_to_value).collect(),
        }
    }
}

/// Simple base64 encoding for bytes
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let mut n = (chunk[0] as u32) << 16;
        if chunk.len() > 1 {
            n |= (chunk[1] as u32) << 8;
        }
        if chunk.len() > 2 {
            n |= chunk[2] as u32;
        }
        result.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Row data for internal processing
#[derive(Debug, Clone)]
pub struct RowData {
    /// Column names
    pub columns: Vec<String>,
    /// Column values
    pub values: Vec<Value>,
}

impl RowData {
    /// Create new row data
    pub fn new(columns: Vec<String>, values: Vec<Value>) -> Self {
        Self { columns, values }
    }

    /// Get value by column name
    pub fn get(&self, column: &str) -> Option<&Value> {
        self.columns
            .iter()
            .position(|c| c == column)
            .and_then(|i| self.values.get(i))
    }

    /// Get value by qualified column name (table.column)
    pub fn get_qualified(&self, table: &str, column: &str) -> Option<&Value> {
        // Try qualified name first
        let qualified = format!("{}.{}", table, column);
        if let Some(v) = self.get(&qualified) {
            return Some(v);
        }
        // Fall back to simple column name
        self.get(column)
    }

    /// Get mutable value by column name
    pub fn get_mut(&mut self, column: &str) -> Option<&mut Value> {
        self.columns
            .iter()
            .position(|c| c == column)
            .and_then(|i| self.values.get_mut(i))
    }

    /// Set value by column name
    pub fn set(&mut self, column: &str, value: Value) {
        if let Some(idx) = self.columns.iter().position(|c| c == column) {
            self.values[idx] = value;
        }
    }

    /// Merge two row data (for joins)
    pub fn merge(
        &self,
        other: &RowData,
        prefix_left: Option<&str>,
        prefix_right: Option<&str>,
    ) -> RowData {
        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (i, col) in self.columns.iter().enumerate() {
            let name = if let Some(prefix) = prefix_left {
                format!("{}.{}", prefix, col)
            } else {
                col.clone()
            };
            columns.push(name);
            values.push(self.values[i].clone());
        }

        for (i, col) in other.columns.iter().enumerate() {
            let name = if let Some(prefix) = prefix_right {
                format!("{}.{}", prefix, col)
            } else {
                col.clone()
            };
            columns.push(name);
            values.push(other.values[i].clone());
        }

        RowData { columns, values }
    }

    /// Convert to serialized row
    pub fn to_serialized(&self) -> SerializedRow {
        SerializedRow {
            columns: self.columns.clone(),
            values: self
                .values
                .iter()
                .map(SerializedRow::value_to_json)
                .collect(),
        }
    }
}

/// Execution result with metadata
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Result set (for SELECT)
    pub result_set: ResultSet,
    /// Number of affected rows (for INSERT/UPDATE/DELETE)
    pub affected_rows: usize,
    /// Last inserted ID (for INSERT)
    pub last_insert_id: Option<i64>,
}

impl ExecutionResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self {
            result_set: ResultSet::empty(),
            affected_rows: 0,
            last_insert_id: None,
        }
    }

    /// Create result with affected rows count
    pub fn with_affected_rows(affected_rows: usize) -> Self {
        Self {
            result_set: ResultSet::empty(),
            affected_rows,
            last_insert_id: None,
        }
    }

    /// Create result with result set
    pub fn with_result_set(result_set: ResultSet) -> Self {
        Self {
            result_set,
            affected_rows: 0,
            last_insert_id: None,
        }
    }
}

/// Storage-backed SQL executor
///
/// Connects SQL queries to the B-tree storage engine, providing full
/// CRUD operations with actual data persistence.
pub struct StorageExecutor {
    /// The underlying B-tree engine
    engine: Arc<Engine>,
    /// Schema catalog
    catalog: Arc<Catalog>,
    /// SQL parser
    parser: SqlParser,
}

impl StorageExecutor {
    /// Create a new storage executor
    pub fn new(engine: Arc<Engine>) -> Self {
        let catalog = Arc::new(Catalog::new(Arc::clone(&engine)));
        Self {
            engine,
            catalog,
            parser: SqlParser::new(),
        }
    }

    /// Get the catalog reference
    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Get the engine reference
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Execute a SQL string
    pub fn execute_sql(
        &mut self,
        sql: &str,
        context: &QueryContext,
    ) -> QueryResult<ExecutionResult> {
        let stmt = self.parser.parse(sql)?;
        self.execute_statement(&stmt, context)
    }

    /// Execute a parsed SQL statement
    pub fn execute_statement(
        &self,
        stmt: &SqlStatement,
        context: &QueryContext,
    ) -> QueryResult<ExecutionResult> {
        match stmt {
            SqlStatement::Select(query) => {
                let result_set = self.execute_select(query, context)?;
                Ok(ExecutionResult::with_result_set(result_set))
            }
            SqlStatement::Insert(insert) => self.execute_insert(insert, context),
            SqlStatement::Update(update) => self.execute_update(update, context),
            SqlStatement::Delete(delete) => self.execute_delete(delete, context),
            SqlStatement::CreateTable(create) => {
                self.execute_create_table(create)?;
                Ok(ExecutionResult::empty())
            }
            SqlStatement::DropTable { name, .. } => {
                self.execute_drop_table(name)?;
                Ok(ExecutionResult::empty())
            }
            SqlStatement::CreateIndex(create) => {
                self.execute_create_index(create)?;
                Ok(ExecutionResult::empty())
            }
            SqlStatement::DropIndex(name) => {
                self.execute_drop_index(name)?;
                Ok(ExecutionResult::empty())
            }
            SqlStatement::AlterTable(_) => {
                // ALTER TABLE not supported in storage executor
                Err(QueryError::ExecutionError(
                    "ALTER TABLE not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::TruncateTable(name) => {
                self.execute_truncate_table(name)?;
                Ok(ExecutionResult::empty())
            }
            SqlStatement::CreateTableAs {
                name,
                query,
                if_not_exists,
            } => {
                self.execute_create_table_as(name, query, *if_not_exists, context)?;
                Ok(ExecutionResult::empty())
            }
            SqlStatement::CreateView(_) | SqlStatement::DropView { .. } => {
                Err(QueryError::ExecutionError(
                    "Views not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::Begin | SqlStatement::Commit | SqlStatement::Rollback => {
                // Transaction control - simplified for now
                Ok(ExecutionResult::empty())
            }
            SqlStatement::ShowTables | SqlStatement::ShowColumns(_) => {
                Err(QueryError::ExecutionError(
                    "SHOW commands not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::CreateFulltextIndex { .. } | SqlStatement::DropFulltextIndex(_) => {
                Err(QueryError::ExecutionError(
                    "Fulltext index management not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::CreateSpatialIndex { .. } | SqlStatement::DropSpatialIndex(_) => {
                Err(QueryError::ExecutionError(
                    "Spatial index management not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::CreateTrigger(_) | SqlStatement::DropTrigger { .. } => {
                Err(QueryError::ExecutionError(
                    "Triggers not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::CreateMaterializedView { .. }
            | SqlStatement::RefreshMaterializedView(_)
            | SqlStatement::DropMaterializedView { .. } => Err(QueryError::ExecutionError(
                "Materialized views not supported in this execution path".to_string(),
            )),
            SqlStatement::CreateVectorIndex { .. } | SqlStatement::DropVectorIndex(_) => {
                Err(QueryError::ExecutionError(
                    "Vector index operations not supported in this execution path".to_string(),
                ))
            }
            SqlStatement::Grant { .. }
            | SqlStatement::Revoke { .. }
            | SqlStatement::CreateUser { .. }
            | SqlStatement::DropUser(_)
            | SqlStatement::AlterUser { .. }
            | SqlStatement::CreateRole(_)
            | SqlStatement::DropRole(_) => Err(QueryError::ExecutionError(
                "Auth/RBAC operations not supported in this execution path".to_string(),
            )),
            SqlStatement::DefineReference { .. } => Err(QueryError::ExecutionError(
                "DEFINE REFERENCE not supported in this execution path".to_string(),
            )),
            SqlStatement::DefineBucket { .. } => Err(QueryError::ExecutionError(
                "DEFINE BUCKET not supported in this execution path".to_string(),
            )),
            SqlStatement::DefineApi { .. } => Err(QueryError::ExecutionError(
                "DEFINE API not supported in this execution path".to_string(),
            )),
            SqlStatement::Explain { statement, .. } => {
                // Execute the inner statement (ANALYZE mode not yet implemented)
                self.execute_statement(statement, context)
            }
            SqlStatement::Savepoint(_)
            | SqlStatement::ReleaseSavepoint(_)
            | SqlStatement::RollbackToSavepoint(_) => {
                // Savepoint control — simplified for now
                Ok(ExecutionResult::empty())
            }
        }
    }

    // ========================================================================
    // DDL Operations
    // ========================================================================

    /// Execute CREATE TABLE
    fn execute_create_table(&self, create: &SqlCreateTable) -> QueryResult<()> {
        let mut schema = TableSchema::new(&create.name);

        for col in &create.columns {
            let data_type = self.parse_data_type(&col.data_type)?;
            let mut col_def = ColumnDef::new(&col.name, data_type);
            col_def = col_def.nullable(col.nullable);
            if col.primary_key {
                col_def = col_def.primary_key();
            }
            schema = schema.add_column(col_def);
        }

        if create.if_not_exists {
            self.catalog
                .create_table_if_not_exists(schema)
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
        } else {
            self.catalog
                .create_table(schema)
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
        }

        Ok(())
    }

    /// Execute DROP TABLE
    fn execute_drop_table(&self, name: &str) -> QueryResult<()> {
        // First, delete all rows
        let prefix = format!("{}{}::", DATA_PREFIX, name);
        self.delete_with_prefix(&prefix)?;

        // Then drop the table schema
        self.catalog
            .drop_table(name)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        Ok(())
    }

    /// Execute TRUNCATE TABLE
    fn execute_truncate_table(&self, name: &str) -> QueryResult<()> {
        // Verify table exists
        let schema = self
            .catalog
            .get_table(name)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
        if schema.is_none() {
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' does not exist",
                name
            )));
        }

        // Delete all rows (keep schema)
        let prefix = format!("{}{}::", DATA_PREFIX, name);
        self.delete_with_prefix(&prefix)?;

        Ok(())
    }

    /// Execute CREATE TABLE AS SELECT
    fn execute_create_table_as(
        &self,
        name: &str,
        query: &crate::sql::SqlQuery,
        if_not_exists: bool,
        context: &QueryContext,
    ) -> QueryResult<()> {
        // Check if table exists
        if let Ok(Some(_)) = self.catalog.get_table(name) {
            if if_not_exists {
                return Ok(());
            }
            return Err(QueryError::ExecutionError(format!(
                "Table '{}' already exists",
                name
            )));
        }

        // Execute the SELECT query
        let result_set = self.execute_select(query, context)?;

        // Create table schema from result columns
        let mut schema = TableSchema::new(name);
        for col_name in &result_set.columns {
            let col_def = ColumnDef::new(col_name, DataType::String);
            schema = schema.add_column(col_def);
        }

        self.catalog
            .create_table(schema)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        // Insert result rows
        for (idx, row) in result_set.rows.iter().enumerate() {
            let row_data = RowData::new(result_set.columns.clone(), row.values.clone());
            let key = format!("{}{}::ctas_{}", DATA_PREFIX, name, idx);
            let serialized = row_data.to_serialized();
            let data = serialized.to_bytes()?;
            self.engine
                .put(key.as_bytes(), &data)
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
        }

        Ok(())
    }

    /// Execute CREATE INDEX
    fn execute_create_index(&self, create: &SqlCreateIndex) -> QueryResult<()> {
        let col_names: Vec<String> = create
            .columns
            .iter()
            .map(|c| match c {
                crate::sql::IndexColumn::Name(n) => n.clone(),
                crate::sql::IndexColumn::Expression(e) => format!("{:?}", e),
            })
            .collect();
        let index = IndexDef::new(&create.name, &create.table, col_names);
        let index = if create.unique { index.unique() } else { index };

        self.catalog
            .create_index(index)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        Ok(())
    }

    /// Execute DROP INDEX
    fn execute_drop_index(&self, name: &str) -> QueryResult<()> {
        self.catalog
            .drop_index(name)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        Ok(())
    }

    /// Parse SQL data type string to catalog DataType
    fn parse_data_type(&self, type_str: &str) -> QueryResult<DataType> {
        DataType::from_sql(type_str)
            .ok_or_else(|| QueryError::ExecutionError(format!("Unknown data type: {}", type_str)))
    }

    // ========================================================================
    // DML Operations
    // ========================================================================

    /// Execute SELECT query
    fn execute_select(
        &self,
        query: &crate::sql::SqlQuery,
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        self.execute_select_inner(query, context, &HashMap::new())
    }

    /// Inner SELECT execution with parent CTE context for recursive references
    fn execute_select_inner(
        &self,
        query: &crate::sql::SqlQuery,
        context: &QueryContext,
        parent_ctes: &HashMap<String, Vec<RowData>>,
    ) -> QueryResult<ResultSet> {
        // Handle SELECT without FROM — evaluate expressions against a single empty row
        if query.from.is_none() {
            return self.execute_select_no_from(query, context);
        }

        // Materialize CTEs
        let mut cte_data: HashMap<String, Vec<RowData>> = parent_ctes.clone();
        for cte in &query.ctes {
            if cte.recursive {
                // Recursive CTE: split base query from recursive part (UNION ALL)
                let mut base_query = cte.query.as_ref().clone();
                let recursive_part = base_query.set_op.take();

                let base_result = self.execute_select_inner(&base_query, context, &cte_data)?;
                let cte_columns = if !cte.columns.is_empty() {
                    cte.columns.clone()
                } else {
                    base_result.columns.clone()
                };

                let mut all_rows: Vec<RowData> = Vec::new();
                for row in &base_result.rows {
                    all_rows.push(RowData::new(cte_columns.clone(), row.values.clone()));
                }
                let mut working_rows: Vec<RowData> = all_rows.clone();

                if let Some(set_op) = recursive_part {
                    const MAX_RECURSION_DEPTH: usize = 1000;
                    for _ in 0..MAX_RECURSION_DEPTH {
                        if working_rows.is_empty() {
                            break;
                        }
                        // Make working table available under the CTE name
                        let mut recursive_ctes = cte_data.clone();
                        recursive_ctes.insert(cte.name.clone(), working_rows);

                        let iter_result =
                            self.execute_select_inner(&set_op.query, context, &recursive_ctes)?;
                        if iter_result.rows.is_empty() {
                            break;
                        }

                        let mut new_rows = Vec::new();
                        for row in iter_result.rows {
                            let renamed = RowData::new(cte_columns.clone(), row.values);
                            new_rows.push(renamed);
                        }
                        all_rows.extend(new_rows.clone());
                        working_rows = new_rows;
                    }
                }

                cte_data.insert(cte.name.clone(), all_rows);
            } else {
                // Non-recursive CTE
                let cte_result = self.execute_select_inner(&cte.query, context, &cte_data)?;
                let mut cte_rows = Vec::new();
                let cte_columns = if !cte.columns.is_empty() {
                    cte.columns.clone()
                } else {
                    cte_result.columns.clone()
                };
                for row in cte_result.rows {
                    let renamed = RowData::new(cte_columns.clone(), row.values);
                    cte_rows.push(renamed);
                }
                cte_data.insert(cte.name.clone(), cte_rows);
            }
        }

        // Get the base table
        let from = query
            .from
            .as_ref()
            .ok_or_else(|| QueryError::ExecutionError("SELECT requires FROM clause".to_string()))?;

        let table_name = from.table_name().ok_or_else(|| {
            QueryError::ExecutionError(
                "Subqueries in FROM not supported by storage executor".to_string(),
            )
        })?;
        let table_alias = from.alias.as_deref();

        // Load rows: check CTE cache first, then physical table
        let mut rows = if let Some(cte_rows) = cte_data.get(table_name) {
            cte_rows.clone()
        } else {
            if !self.table_exists(table_name)? {
                return Err(QueryError::UnknownTable(table_name.to_string()));
            }
            self.scan_table(table_name)?
        };

        // Apply joins
        for join in &query.joins {
            rows = self.apply_join(
                rows,
                table_alias.unwrap_or(table_name),
                join,
                context,
                &cte_data,
            )?;
        }

        // Apply WHERE clause
        if let Some(ref where_clause) = query.where_clause {
            rows = self.filter_rows(rows, where_clause, context)?;
        }

        // Apply GROUP BY and aggregations
        if !query.group_by.is_empty() || self.has_aggregates(&query.columns) {
            rows = self.apply_grouping(rows, &query.group_by, &query.columns, context)?;
        }

        // Apply HAVING clause
        if let Some(ref having) = query.having {
            rows = self.filter_rows(rows, having, context)?;
        }

        // Apply ORDER BY
        if !query.order_by.is_empty() {
            rows = self.sort_rows(rows, &query.order_by)?;
        }

        // Apply OFFSET
        if let Some(offset) = query.offset {
            if offset < rows.len() {
                rows = rows[offset..].to_vec();
            } else {
                rows = Vec::new();
            }
        }

        // Apply LIMIT
        if let Some(limit) = query.limit {
            rows.truncate(limit);
        }

        // Project columns
        let mut result = self.project_columns(rows, &query.columns, context)?;

        // Apply DISTINCT
        if query.distinct {
            let mut seen = std::collections::HashSet::new();
            result
                .rows
                .retain(|row| seen.insert(format!("{:?}", row.values)));
        }

        Ok(result)
    }

    /// Execute SELECT without FROM clause — evaluates expressions against an empty row
    fn execute_select_no_from(
        &self,
        query: &crate::sql::SqlQuery,
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        let empty_row = RowData::new(Vec::new(), Vec::new());
        let mut columns = Vec::new();
        let mut values = Vec::new();

        for (i, col) in query.columns.iter().enumerate() {
            let alias = col.alias.clone().unwrap_or_else(|| format!("col_{}", i));
            columns.push(alias);
            let val = self.evaluate_expression(&col.expr, &empty_row, context)?;
            values.push(val);
        }

        let mut result = ResultSet {
            columns,
            rows: vec![Row::new(values)],
            affected_rows: 0,
            execution_time_ms: 0,
            truncated: false,
        };

        // Apply DISTINCT
        if query.distinct {
            let mut seen = std::collections::HashSet::new();
            result
                .rows
                .retain(|row| seen.insert(format!("{:?}", row.values)));
        }

        Ok(result)
    }

    /// Execute INSERT statement
    fn execute_insert(
        &self,
        insert: &SqlInsert,
        context: &QueryContext,
    ) -> QueryResult<ExecutionResult> {
        // Verify table exists
        let schema = self
            .catalog
            .get_table(&insert.table)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?
            .ok_or_else(|| QueryError::UnknownTable(insert.table.clone()))?;

        let mut inserted_count = 0;
        let mut last_pk: Option<i64> = None;

        // Resolve rows to insert: either from VALUES or from SELECT
        let resolved_rows: Vec<Vec<Value>> = match &insert.source {
            crate::sql::InsertSource::Values(value_rows) => {
                let mut rows = Vec::new();
                for value_row in value_rows {
                    let mut values = Vec::new();
                    for expr in value_row {
                        let value =
                            self.evaluate_expression(expr, &RowData::new(vec![], vec![]), context)?;
                        values.push(value);
                    }
                    rows.push(values);
                }
                rows
            }
            crate::sql::InsertSource::Select(select_query) => {
                // Execute the SELECT and collect result rows
                let select_stmt = crate::sql::SqlStatement::Select(*select_query.clone());
                let select_result = self.execute_statement(&select_stmt, context)?;
                select_result
                    .result_set
                    .rows
                    .iter()
                    .map(|row| row.values.clone())
                    .collect()
            }
        };

        for values in resolved_rows {
            let columns = if insert.columns.is_empty() {
                schema
                    .column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            } else {
                insert.columns.clone()
            };

            if columns.len() != values.len() {
                return Err(QueryError::ExecutionError(format!(
                    "Column count ({}) doesn't match value count ({})",
                    columns.len(),
                    values.len()
                )));
            }

            // Build the row
            let row_data = RowData::new(columns, values);

            // Get primary key value
            let pk_value = self.get_primary_key_value(&schema, &row_data)?;
            if let Value::Int(pk) = &pk_value {
                last_pk = Some(*pk);
            }

            // Generate storage key
            let key = self.make_row_key(&insert.table, &pk_value);

            // Check for ON CONFLICT handling
            let key_exists = self
                .engine
                .get(key.as_bytes())
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?
                .is_some();

            if key_exists {
                match &insert.on_conflict {
                    Some(oc) => match &oc.action {
                        crate::sql::OnConflictAction::DoNothing => {
                            // Skip this row
                            continue;
                        }
                        crate::sql::OnConflictAction::DoUpdate(assignments) => {
                            // Read existing row, apply SET expressions, write back
                            let existing_bytes = self
                                .engine
                                .get(key.as_bytes())
                                .map_err(|e| QueryError::ExecutionError(e.to_string()))?
                                .ok_or_else(|| {
                                    QueryError::ExecutionError(format!(
                                        "Row disappeared during ON CONFLICT UPDATE for key: {}",
                                        key
                                    ))
                                })?;
                            let existing = SerializedRow::from_bytes(&existing_bytes)?;
                            let mut existing_row = existing.to_row_data();

                            let context = &QueryContext::default();
                            for (col, expr) in assignments {
                                // Resolve excluded.col references
                                let resolved = self.resolve_excluded_refs_storage(expr, &row_data);
                                let val =
                                    self.evaluate_expression(&resolved, &existing_row, context)?;
                                if let Some(idx) =
                                    existing_row.columns.iter().position(|c| c == col)
                                {
                                    existing_row.values[idx] = val;
                                }
                            }

                            let serialized = existing_row.to_serialized();
                            let data = serialized.to_bytes()?;
                            self.engine
                                .put(key.as_bytes(), &data)
                                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
                            inserted_count += 1;
                        }
                    },
                    None => {
                        // No ON CONFLICT — overwrite (legacy behavior for storage executor)
                        let serialized = row_data.to_serialized();
                        let data = serialized.to_bytes()?;
                        self.engine
                            .put(key.as_bytes(), &data)
                            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
                        inserted_count += 1;
                    }
                }
            } else {
                // No conflict — insert normally
                let serialized = row_data.to_serialized();
                let data = serialized.to_bytes()?;
                self.engine
                    .put(key.as_bytes(), &data)
                    .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
                inserted_count += 1;
            }
        }

        Ok(ExecutionResult {
            result_set: ResultSet::empty(),
            affected_rows: inserted_count,
            last_insert_id: last_pk,
        })
    }

    /// Execute UPDATE statement
    fn execute_update(
        &self,
        update: &SqlUpdate,
        context: &QueryContext,
    ) -> QueryResult<ExecutionResult> {
        // Verify table exists
        let schema = self
            .catalog
            .get_table(&update.table)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?
            .ok_or_else(|| QueryError::UnknownTable(update.table.clone()))?;

        // Load all rows
        let rows = self.scan_table(&update.table)?;

        // Filter rows that match WHERE clause
        let matching_rows = if let Some(ref where_clause) = update.where_clause {
            self.filter_rows(rows, where_clause, context)?
        } else {
            rows
        };

        let mut updated_count = 0;

        for mut row in matching_rows {
            // Apply assignments
            for (column, expr) in &update.assignments {
                let value = self.evaluate_expression(expr, &row, context)?;
                row.set(column, value);
            }

            // Get primary key value
            let pk_value = self.get_primary_key_value(&schema, &row)?;

            // Generate storage key
            let key = self.make_row_key(&update.table, &pk_value);

            // Serialize and store
            let serialized = row.to_serialized();
            let data = serialized.to_bytes()?;

            self.engine
                .put(key.as_bytes(), &data)
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

            updated_count += 1;
        }

        Ok(ExecutionResult::with_affected_rows(updated_count))
    }

    /// Execute DELETE statement
    fn execute_delete(
        &self,
        delete: &SqlDelete,
        context: &QueryContext,
    ) -> QueryResult<ExecutionResult> {
        // Verify table exists
        let schema = self
            .catalog
            .get_table(&delete.table)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?
            .ok_or_else(|| QueryError::UnknownTable(delete.table.clone()))?;

        // Load all rows
        let rows = self.scan_table(&delete.table)?;

        // Filter rows that match WHERE clause
        let matching_rows = if let Some(ref where_clause) = delete.where_clause {
            self.filter_rows(rows, where_clause, context)?
        } else {
            rows
        };

        let mut deleted_count = 0;

        for row in matching_rows {
            // Get primary key value
            let pk_value = self.get_primary_key_value(&schema, &row)?;

            // Generate storage key
            let key = self.make_row_key(&delete.table, &pk_value);

            // Delete from storage
            self.engine
                .delete(key.as_bytes())
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

            deleted_count += 1;
        }

        Ok(ExecutionResult::with_affected_rows(deleted_count))
    }

    // ========================================================================
    // Table Operations
    // ========================================================================

    /// Check if a table exists
    fn table_exists(&self, table: &str) -> QueryResult<bool> {
        self.catalog
            .table_exists(table)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))
    }

    /// Scan all rows from a table
    fn scan_table(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let prefix = format!("{}{}::", DATA_PREFIX, table);
        let mut rows = Vec::new();

        // Use range scan to get all rows with the table prefix
        let mut iter = self
            .engine
            .prefix_scan(prefix.as_bytes())
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        while let Some(result) = iter.next() {
            let entry = result.map_err(|e| QueryError::ExecutionError(e.to_string()))?;
            let serialized = SerializedRow::from_bytes(&entry.value)?;
            rows.push(serialized.to_row_data());
        }

        Ok(rows)
    }

    /// Delete all rows with a given key prefix
    fn delete_with_prefix(&self, prefix: &str) -> QueryResult<usize> {
        let mut iter = self
            .engine
            .prefix_scan(prefix.as_bytes())
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        let mut deleted = 0;
        let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();

        while let Some(result) = iter.next() {
            if let Ok(entry) = result {
                keys_to_delete.push(entry.key);
            }
        }

        for key in keys_to_delete {
            self.engine
                .delete(&key)
                .map_err(|e| QueryError::ExecutionError(e.to_string()))?;
            deleted += 1;
        }

        Ok(deleted)
    }

    /// Resolve `excluded.column` references in ON CONFLICT SET expressions
    fn resolve_excluded_refs_storage(&self, expr: &Expression, new_row: &RowData) -> Expression {
        match expr {
            Expression::QualifiedColumn { table, column }
                if table.eq_ignore_ascii_case("excluded") =>
            {
                if let Some(idx) = new_row.columns.iter().position(|c| c == column) {
                    Expression::Literal(new_row.values[idx].clone())
                } else {
                    expr.clone()
                }
            }
            Expression::Binary { left, op, right } => Expression::Binary {
                left: Box::new(self.resolve_excluded_refs_storage(left, new_row)),
                op: op.clone(),
                right: Box::new(self.resolve_excluded_refs_storage(right, new_row)),
            },
            Expression::Function { name, args } => Expression::Function {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| self.resolve_excluded_refs_storage(a, new_row))
                    .collect(),
            },
            _ => expr.clone(),
        }
    }

    /// Generate storage key for a row
    fn make_row_key(&self, table: &str, pk_value: &Value) -> String {
        let pk_str = self.value_to_key_string(pk_value);
        format!("{}{}::{}", DATA_PREFIX, table, pk_str)
    }

    /// Convert a value to a string suitable for use in a key
    fn value_to_key_string(&self, value: &Value) -> String {
        match value {
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int(i) => format!("{:020}", i), // Zero-padded for proper sorting
            Value::Float(f) => format!("{}", f),
            Value::String(s) => s.clone(),
            Value::Timestamp(ts) => format!("{:020}", ts),
            Value::Uuid(u) => u.clone(),
            _ => format!("{:?}", value),
        }
    }

    /// Get the primary key value from a row
    fn get_primary_key_value(&self, schema: &TableSchema, row: &RowData) -> QueryResult<Value> {
        if schema.primary_key.is_empty() {
            return Err(QueryError::ExecutionError(
                "Table has no primary key".to_string(),
            ));
        }

        // For composite keys, concatenate values
        if schema.primary_key.len() == 1 {
            let pk_col = &schema.primary_key[0];
            row.get(pk_col)
                .cloned()
                .ok_or_else(|| QueryError::UnknownColumn(pk_col.clone()))
        } else {
            let parts: Vec<String> = schema
                .primary_key
                .iter()
                .filter_map(|col| row.get(col).map(|v| self.value_to_key_string(v)))
                .collect();
            Ok(Value::String(parts.join(":")))
        }
    }

    // ========================================================================
    // Query Operations
    // ========================================================================

    /// Apply a JOIN operation
    fn apply_join(
        &self,
        left_rows: Vec<RowData>,
        left_alias: &str,
        join: &crate::sql::SqlJoin,
        context: &QueryContext,
        cte_data: &HashMap<String, Vec<RowData>>,
    ) -> QueryResult<Vec<RowData>> {
        let right_alias = join.alias.as_deref().unwrap_or(&join.table);

        // Load right table rows — check CTE cache first, then physical table
        let right_rows = if let Some(cte_rows) = cte_data.get(&join.table) {
            cte_rows.clone()
        } else {
            if !self.table_exists(&join.table)? {
                return Err(QueryError::UnknownTable(join.table.clone()));
            }
            self.scan_table(&join.table)?
        };

        match join.join_type {
            JoinType::Inner => self.inner_join(
                left_rows,
                right_rows,
                left_alias,
                right_alias,
                &join.condition,
                context,
            ),
            JoinType::Left => self.left_join(
                left_rows,
                right_rows,
                left_alias,
                right_alias,
                &join.condition,
                context,
            ),
            JoinType::Right => self.right_join(
                left_rows,
                right_rows,
                left_alias,
                right_alias,
                &join.condition,
                context,
            ),
            JoinType::Full => self.full_join(
                left_rows,
                right_rows,
                left_alias,
                right_alias,
                &join.condition,
                context,
            ),
            JoinType::Cross => self.cross_join(left_rows, right_rows, left_alias, right_alias),
        }
    }

    /// Execute INNER JOIN
    fn inner_join(
        &self,
        left_rows: Vec<RowData>,
        right_rows: Vec<RowData>,
        left_alias: &str,
        right_alias: &str,
        condition: &Option<Expression>,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        let mut result = Vec::new();

        // Check if left rows already have prefixes (from previous joins)
        let left_already_prefixed = left_rows
            .first()
            .map(|r| r.columns.first().map(|c| c.contains('.')).unwrap_or(false))
            .unwrap_or(false);

        for left in &left_rows {
            for right in &right_rows {
                let merged = if left_already_prefixed {
                    // Left is already prefixed, only prefix right
                    left.merge(right, None, Some(right_alias))
                } else {
                    left.merge(right, Some(left_alias), Some(right_alias))
                };

                let matches = match condition {
                    Some(cond) => self.evaluate_predicate(cond, &merged, context)?,
                    None => true,
                };

                if matches {
                    result.push(merged);
                }
            }
        }

        Ok(result)
    }

    /// Execute LEFT JOIN
    fn left_join(
        &self,
        left_rows: Vec<RowData>,
        right_rows: Vec<RowData>,
        left_alias: &str,
        right_alias: &str,
        condition: &Option<Expression>,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        let mut result = Vec::new();

        let left_already_prefixed = left_rows
            .first()
            .map(|r| r.columns.first().map(|c| c.contains('.')).unwrap_or(false))
            .unwrap_or(false);

        for left in &left_rows {
            let mut matched = false;

            for right in &right_rows {
                let merged = if left_already_prefixed {
                    left.merge(right, None, Some(right_alias))
                } else {
                    left.merge(right, Some(left_alias), Some(right_alias))
                };

                let matches = match condition {
                    Some(cond) => self.evaluate_predicate(cond, &merged, context)?,
                    None => true,
                };

                if matches {
                    result.push(merged);
                    matched = true;
                }
            }

            if !matched {
                // Add left row with NULLs for right columns
                let null_right = self.make_null_row(&right_rows);
                let merged = if left_already_prefixed {
                    left.merge(&null_right, None, Some(right_alias))
                } else {
                    left.merge(&null_right, Some(left_alias), Some(right_alias))
                };
                result.push(merged);
            }
        }

        Ok(result)
    }

    /// Execute RIGHT JOIN
    fn right_join(
        &self,
        left_rows: Vec<RowData>,
        right_rows: Vec<RowData>,
        left_alias: &str,
        right_alias: &str,
        condition: &Option<Expression>,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        let mut result = Vec::new();

        let left_already_prefixed = left_rows
            .first()
            .map(|r| r.columns.first().map(|c| c.contains('.')).unwrap_or(false))
            .unwrap_or(false);

        for right in &right_rows {
            let mut matched = false;

            for left in &left_rows {
                let merged = if left_already_prefixed {
                    left.merge(right, None, Some(right_alias))
                } else {
                    left.merge(right, Some(left_alias), Some(right_alias))
                };

                let matches = match condition {
                    Some(cond) => self.evaluate_predicate(cond, &merged, context)?,
                    None => true,
                };

                if matches {
                    result.push(merged);
                    matched = true;
                }
            }

            if !matched {
                let null_left = self.make_null_row(&left_rows);
                let merged = if left_already_prefixed {
                    null_left.merge(right, None, Some(right_alias))
                } else {
                    null_left.merge(right, Some(left_alias), Some(right_alias))
                };
                result.push(merged);
            }
        }

        Ok(result)
    }

    /// Execute FULL OUTER JOIN
    fn full_join(
        &self,
        left_rows: Vec<RowData>,
        right_rows: Vec<RowData>,
        left_alias: &str,
        right_alias: &str,
        condition: &Option<Expression>,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        let mut result = Vec::new();
        let mut right_matched = vec![false; right_rows.len()];

        let left_already_prefixed = left_rows
            .first()
            .map(|r| r.columns.first().map(|c| c.contains('.')).unwrap_or(false))
            .unwrap_or(false);

        for left in &left_rows {
            let mut matched = false;

            for (ri, right) in right_rows.iter().enumerate() {
                let merged = if left_already_prefixed {
                    left.merge(right, None, Some(right_alias))
                } else {
                    left.merge(right, Some(left_alias), Some(right_alias))
                };

                let matches = match condition {
                    Some(cond) => self.evaluate_predicate(cond, &merged, context)?,
                    None => true,
                };

                if matches {
                    result.push(merged);
                    matched = true;
                    right_matched[ri] = true;
                }
            }

            if !matched {
                let null_right = self.make_null_row(&right_rows);
                let merged = if left_already_prefixed {
                    left.merge(&null_right, None, Some(right_alias))
                } else {
                    left.merge(&null_right, Some(left_alias), Some(right_alias))
                };
                result.push(merged);
            }
        }

        // Add unmatched right rows
        for (ri, right) in right_rows.iter().enumerate() {
            if !right_matched[ri] {
                let null_left = self.make_null_row(&left_rows);
                let merged = if left_already_prefixed {
                    null_left.merge(right, None, Some(right_alias))
                } else {
                    null_left.merge(right, Some(left_alias), Some(right_alias))
                };
                result.push(merged);
            }
        }

        Ok(result)
    }

    /// Execute CROSS JOIN
    fn cross_join(
        &self,
        left_rows: Vec<RowData>,
        right_rows: Vec<RowData>,
        left_alias: &str,
        right_alias: &str,
    ) -> QueryResult<Vec<RowData>> {
        let mut result = Vec::new();

        let left_already_prefixed = left_rows
            .first()
            .map(|r| r.columns.first().map(|c| c.contains('.')).unwrap_or(false))
            .unwrap_or(false);

        for left in &left_rows {
            for right in &right_rows {
                let merged = if left_already_prefixed {
                    left.merge(right, None, Some(right_alias))
                } else {
                    left.merge(right, Some(left_alias), Some(right_alias))
                };
                result.push(merged);
            }
        }

        Ok(result)
    }

    /// Create a row with NULL values matching the structure of existing rows
    fn make_null_row(&self, sample_rows: &[RowData]) -> RowData {
        if let Some(sample) = sample_rows.first() {
            RowData::new(
                sample.columns.clone(),
                vec![Value::Null; sample.columns.len()],
            )
        } else {
            RowData::new(vec![], vec![])
        }
    }

    /// Filter rows based on a predicate
    fn filter_rows(
        &self,
        rows: Vec<RowData>,
        predicate: &Expression,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        let mut result = Vec::new();
        for row in rows {
            if self.evaluate_predicate(predicate, &row, context)? {
                result.push(row);
            }
        }
        Ok(result)
    }

    /// Sort rows based on ORDER BY clauses
    fn sort_rows(&self, mut rows: Vec<RowData>, order_by: &[OrderBy]) -> QueryResult<Vec<RowData>> {
        rows.sort_by(|a, b| {
            for order in order_by {
                let a_val = self.eval_order_expr(&order.expr, a);
                let b_val = self.eval_order_expr(&order.expr, b);

                // Handle NULLs separately from the ASC/DESC reversal.
                // Default: NULLS LAST for ASC, NULLS FIRST for DESC (SQL standard).
                let nulls_first = order.nulls_first.unwrap_or(order.descending);
                match (&a_val, &b_val) {
                    (Value::Null, Value::Null) => {} // equal, check next order clause
                    (Value::Null, _) => {
                        return if nulls_first {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Greater
                        };
                    }
                    (_, Value::Null) => {
                        return if nulls_first {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        };
                    }
                    _ => {}
                }
                let cmp = self.compare_values(&a_val, &b_val);
                if cmp != std::cmp::Ordering::Equal {
                    return if order.descending { cmp.reverse() } else { cmp };
                }
            }
            std::cmp::Ordering::Equal
        });

        Ok(rows)
    }

    fn eval_order_expr(&self, expr: &Expression, row: &RowData) -> Value {
        match expr {
            Expression::Column(name) => row.get(name).cloned().unwrap_or(Value::Null),
            Expression::Literal(v) => v.clone(),
            Expression::Binary { left, op, right } => {
                let l = self.eval_order_expr(left, row);
                let r = self.eval_order_expr(right, row);
                match (l, op, r) {
                    (Value::Int(a), crate::ast::Operator::Add, Value::Int(b)) => Value::Int(a + b),
                    (Value::Int(a), crate::ast::Operator::Sub, Value::Int(b)) => Value::Int(a - b),
                    (Value::Int(a), crate::ast::Operator::Mul, Value::Int(b)) => Value::Int(a * b),
                    (Value::Int(a), crate::ast::Operator::Div, Value::Int(b)) if b != 0 => {
                        Value::Int(a / b)
                    }
                    (Value::Float(a), crate::ast::Operator::Add, Value::Float(b)) => {
                        Value::Float(a + b)
                    }
                    (Value::Float(a), crate::ast::Operator::Mul, Value::Float(b)) => {
                        Value::Float(a * b)
                    }
                    (Value::Int(a), crate::ast::Operator::Mul, Value::Float(b))
                    | (Value::Float(b), crate::ast::Operator::Mul, Value::Int(a)) => {
                        Value::Float(a as f64 * b)
                    }
                    (Value::Int(a), crate::ast::Operator::Add, Value::Float(b))
                    | (Value::Float(b), crate::ast::Operator::Add, Value::Int(a)) => {
                        Value::Float(a as f64 + b)
                    }
                    _ => Value::Null,
                }
            }
            Expression::Function { name, args } => {
                let arg_vals: Vec<Value> =
                    args.iter().map(|a| self.eval_order_expr(a, row)).collect();
                match name.to_uppercase().as_str() {
                    "LOWER" => match arg_vals.first() {
                        Some(Value::String(s)) => Value::String(s.to_lowercase()),
                        _ => Value::Null,
                    },
                    "UPPER" => match arg_vals.first() {
                        Some(Value::String(s)) => Value::String(s.to_uppercase()),
                        _ => Value::Null,
                    },
                    "ABS" => match arg_vals.first() {
                        Some(Value::Int(i)) => Value::Int(i.abs()),
                        Some(Value::Float(f)) => Value::Float(f.abs()),
                        _ => Value::Null,
                    },
                    "LENGTH" | "LEN" => match arg_vals.first() {
                        Some(Value::String(s)) => Value::Int(s.len() as i64),
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                }
            }
            Expression::QualifiedColumn { column, .. } => {
                row.get(column).cloned().unwrap_or(Value::Null)
            }
            Expression::Case { operand, when_clauses, else_clause } => {
                if let Some(op_expr) = operand {
                    // Simple CASE: CASE expr WHEN val THEN result ...
                    let op_val = self.eval_order_expr(op_expr, row);
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.eval_order_expr(when_expr, row);
                        if self.compare_values(&op_val, &when_val) == std::cmp::Ordering::Equal {
                            return self.eval_order_expr(then_expr, row);
                        }
                    }
                } else {
                    // Searched CASE: CASE WHEN cond THEN result ...
                    for (when_expr, then_expr) in when_clauses {
                        let cond = self.eval_order_expr(when_expr, row);
                        let is_true = match &cond {
                            Value::Bool(b) => *b,
                            Value::Int(n) => *n != 0,
                            _ => {
                                // Evaluate as comparison expression
                                self.eval_predicate_bool(when_expr, row)
                            }
                        };
                        if is_true {
                            return self.eval_order_expr(then_expr, row);
                        }
                    }
                }
                if let Some(else_expr) = else_clause {
                    self.eval_order_expr(else_expr, row)
                } else {
                    Value::Null
                }
            }
            Expression::Unary { op, expr } => {
                let val = self.eval_order_expr(expr, row);
                match op {
                    crate::ast::UnaryOperator::Neg => match val {
                        Value::Int(n) => Value::Int(-n),
                        Value::Float(n) => Value::Float(-n),
                        _ => Value::Null,
                    },
                    crate::ast::UnaryOperator::Not => match val {
                        Value::Bool(b) => Value::Bool(!b),
                        _ => Value::Null,
                    },
                    _ => Value::Null,
                }
            }
            Expression::IsNull { expr, negated } => {
                let val = self.eval_order_expr(expr, row);
                let is_null = matches!(val, Value::Null);
                Value::Bool(if *negated { !is_null } else { is_null })
            }
            Expression::Between { expr, low, high, negated } => {
                let val = self.eval_order_expr(expr, row);
                let low_val = self.eval_order_expr(low, row);
                let high_val = self.eval_order_expr(high, row);
                let in_range = self.compare_values(&val, &low_val) != std::cmp::Ordering::Less
                    && self.compare_values(&val, &high_val) != std::cmp::Ordering::Greater;
                Value::Bool(if *negated { !in_range } else { in_range })
            }
            _ => Value::Null,
        }
    }

    /// Evaluate a predicate expression as a boolean for CASE WHEN conditions
    fn eval_predicate_bool(&self, expr: &Expression, row: &RowData) -> bool {
        match expr {
            Expression::Binary { left, op, right } => {
                match op {
                    crate::ast::Operator::And => {
                        self.eval_predicate_bool(left, row) && self.eval_predicate_bool(right, row)
                    }
                    crate::ast::Operator::Or => {
                        self.eval_predicate_bool(left, row) || self.eval_predicate_bool(right, row)
                    }
                    crate::ast::Operator::Eq | crate::ast::Operator::Ne |
                    crate::ast::Operator::Lt | crate::ast::Operator::Le |
                    crate::ast::Operator::Gt | crate::ast::Operator::Ge => {
                        let l = self.eval_order_expr(left, row);
                        let r = self.eval_order_expr(right, row);
                        let cmp = self.compare_values(&l, &r);
                        match op {
                            crate::ast::Operator::Eq => cmp == std::cmp::Ordering::Equal,
                            crate::ast::Operator::Ne => cmp != std::cmp::Ordering::Equal,
                            crate::ast::Operator::Lt => cmp == std::cmp::Ordering::Less,
                            crate::ast::Operator::Le => cmp != std::cmp::Ordering::Greater,
                            crate::ast::Operator::Gt => cmp == std::cmp::Ordering::Greater,
                            crate::ast::Operator::Ge => cmp != std::cmp::Ordering::Less,
                            _ => false,
                        }
                    }
                    _ => false,
                }
            }
            Expression::IsNull { expr, negated } => {
                let val = self.eval_order_expr(expr, row);
                let is_null = matches!(val, Value::Null);
                if *negated { !is_null } else { is_null }
            }
            Expression::Unary { op: crate::ast::UnaryOperator::Not, expr } => {
                !self.eval_predicate_bool(expr, row)
            }
            _ => {
                match self.eval_order_expr(expr, row) {
                    Value::Bool(b) => b,
                    _ => false,
                }
            }
        }
    }

    /// Check if columns contain aggregate functions
    fn has_aggregates(&self, columns: &[crate::sql::SqlColumn]) -> bool {
        for col in columns {
            if self.expr_has_aggregate(&col.expr) {
                return true;
            }
        }
        false
    }

    /// Check if expression contains aggregate function
    fn expr_has_aggregate(&self, expr: &Expression) -> bool {
        match expr {
            Expression::Function { name, .. } => {
                let upper = name.to_uppercase();
                matches!(
                    upper.as_str(),
                    "COUNT"
                        | "SUM"
                        | "AVG"
                        | "MIN"
                        | "MAX"
                        | "FIRST"
                        | "LAST"
                        | "COUNT_DISTINCT"
                        | "STRING_AGG"
                        | "GROUP_CONCAT"
                        | "JSON_AGG"
                        | "JSON_OBJECT_AGG"
                        | "STDDEV"
                        | "STDDEV_POP"
                        | "STDDEV_SAMP"
                        | "VARIANCE"
                        | "VAR_POP"
                        | "VAR_SAMP"
                        | "MEDIAN"
                        | "APPROX_COUNT_DISTINCT"
                        | "APPROX_PERCENTILE"
                        | "PERCENTILE_APPROX"
                )
            }
            Expression::Binary { left, right, .. } => {
                self.expr_has_aggregate(left) || self.expr_has_aggregate(right)
            }
            Expression::Unary { expr, .. } => self.expr_has_aggregate(expr),
            _ => false,
        }
    }

    /// Apply GROUP BY and aggregations
    fn apply_grouping(
        &self,
        rows: Vec<RowData>,
        group_by: &[Expression],
        columns: &[crate::sql::SqlColumn],
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        // Group rows
        let mut groups: HashMap<String, Vec<RowData>> = HashMap::new();

        for row in rows {
            let key = if group_by.is_empty() {
                String::new()
            } else {
                group_by
                    .iter()
                    .map(|expr| {
                        self.evaluate_expression(expr, &row, context)
                            .map(|v| format!("{:?}", v))
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>()
                    .join("::")
            };
            groups.entry(key).or_default().push(row);
        }

        // Compute aggregates for each group
        let mut result_rows = Vec::new();

        for (_, group_rows) in groups {
            let mut result_columns = Vec::new();
            let mut result_values = Vec::new();

            for col in columns {
                let col_name = col.alias.clone().unwrap_or_else(|| match &col.expr {
                    Expression::Column(name) => name.clone(),
                    Expression::Function { name, .. } => name.clone(),
                    _ => "expr".to_string(),
                });

                let value = self.evaluate_expression_with_group(&col.expr, &group_rows, context)?;
                result_columns.push(col_name);
                result_values.push(value);
            }

            result_rows.push(RowData::new(result_columns, result_values));
        }

        Ok(result_rows)
    }

    /// Evaluate expression with group context (for aggregates)
    fn evaluate_expression_with_group(
        &self,
        expr: &Expression,
        group_rows: &[RowData],
        context: &QueryContext,
    ) -> QueryResult<Value> {
        match expr {
            Expression::Function { name, args } => {
                let upper = name.to_uppercase();
                match upper.as_str() {
                    "COUNT" => {
                        if args.is_empty() || matches!(args[0], Expression::Wildcard) {
                            Ok(Value::Int(group_rows.len() as i64))
                        } else {
                            let count = group_rows
                                .iter()
                                .filter(|row| {
                                    self.evaluate_expression(&args[0], row, context)
                                        .map(|v| !matches!(v, Value::Null))
                                        .unwrap_or(false)
                                })
                                .count();
                            Ok(Value::Int(count as i64))
                        }
                    }
                    "SUM" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.is_empty() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float(nums.iter().sum()))
                        }
                    }
                    "AVG" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let values: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if values.is_empty() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Float(
                                values.iter().sum::<f64>() / values.len() as f64,
                            ))
                        }
                    }
                    "MIN" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let min = group_rows
                            .iter()
                            .filter_map(|row| self.evaluate_expression(&args[0], row, context).ok())
                            .min_by(|a, b| self.compare_values(a, b));
                        Ok(min.unwrap_or(Value::Null))
                    }
                    "MAX" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let max = group_rows
                            .iter()
                            .filter_map(|row| self.evaluate_expression(&args[0], row, context).ok())
                            .max_by(|a, b| self.compare_values(a, b));
                        Ok(max.unwrap_or(Value::Null))
                    }
                    "FIRST" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        if args.len() >= 2 {
                            // FIRST(value, time) — return value from row with min time
                            let mut best_row = None;
                            let mut best_time: Option<i64> = None;
                            for row in group_rows {
                                let t = self
                                    .evaluate_expression(&args[1], row, context)
                                    .ok()
                                    .and_then(|v| match v {
                                        Value::Timestamp(t) => Some(t),
                                        Value::Int(t) => Some(t),
                                        _ => None,
                                    });
                                if let Some(t) = t {
                                    if best_time.is_none() || t < best_time.unwrap() {
                                        best_time = Some(t);
                                        best_row = Some(row);
                                    }
                                }
                            }
                            best_row
                                .map(|row| self.evaluate_expression(&args[0], row, context))
                                .transpose()?
                                .map_or(Ok(Value::Null), Ok)
                        } else {
                            group_rows
                                .first()
                                .map(|row| self.evaluate_expression(&args[0], row, context))
                                .transpose()?
                                .map_or(Ok(Value::Null), Ok)
                        }
                    }
                    "LAST" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        if args.len() >= 2 {
                            // LAST(value, time) — return value from row with max time
                            let mut best_row = None;
                            let mut best_time: Option<i64> = None;
                            for row in group_rows {
                                let t = self
                                    .evaluate_expression(&args[1], row, context)
                                    .ok()
                                    .and_then(|v| match v {
                                        Value::Timestamp(t) => Some(t),
                                        Value::Int(t) => Some(t),
                                        _ => None,
                                    });
                                if let Some(t) = t {
                                    if best_time.is_none() || t > best_time.unwrap() {
                                        best_time = Some(t);
                                        best_row = Some(row);
                                    }
                                }
                            }
                            best_row
                                .map(|row| self.evaluate_expression(&args[0], row, context))
                                .transpose()?
                                .map_or(Ok(Value::Null), Ok)
                        } else {
                            group_rows
                                .last()
                                .map(|row| self.evaluate_expression(&args[0], row, context))
                                .transpose()?
                                .map_or(Ok(Value::Null), Ok)
                        }
                    }
                    "COUNT_DISTINCT" => {
                        if args.is_empty() {
                            return Ok(Value::Int(group_rows.len() as i64));
                        }
                        let mut unique: std::collections::HashSet<String> =
                            std::collections::HashSet::new();
                        for row in group_rows {
                            let val = self.evaluate_expression(&args[0], row, context)?;
                            if !matches!(val, Value::Null) {
                                unique.insert(format!("{:?}", val));
                            }
                        }
                        Ok(Value::Int(unique.len() as i64))
                    }
                    "STRING_AGG" | "GROUP_CONCAT" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let delimiter = if args.len() > 1 {
                            if let Some(row) = group_rows.first() {
                                match self.evaluate_expression(&args[1], row, context)? {
                                    Value::String(s) => s,
                                    _ => ",".to_string(),
                                }
                            } else {
                                ",".to_string()
                            }
                        } else {
                            ",".to_string()
                        };
                        let parts: Vec<String> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| match v {
                                        Value::String(s) => Some(s),
                                        Value::Null => None,
                                        Value::Int(n) => Some(n.to_string()),
                                        Value::Float(f) => Some(f.to_string()),
                                        Value::Bool(b) => Some(b.to_string()),
                                        other => Some(format!("{:?}", other)),
                                    })
                            })
                            .collect();
                        if parts.is_empty() {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::String(parts.join(&delimiter)))
                        }
                    }
                    "JSON_AGG" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let arr: Vec<Value> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .filter(|v| !v.is_null())
                            })
                            .collect();
                        Ok(Value::Array(arr))
                    }
                    "JSON_OBJECT_AGG" => {
                        if args.len() < 2 {
                            return Ok(Value::Null);
                        }
                        let mut map = serde_json::Map::new();
                        for row in group_rows {
                            let key = self.evaluate_expression(&args[0], row, context)?;
                            let val = self.evaluate_expression(&args[1], row, context)?;
                            let key_str = match &key {
                                Value::String(s) => s.clone(),
                                Value::Int(n) => n.to_string(),
                                _ => format!("{:?}", key),
                            };
                            map.insert(key_str, crate::ast::value_to_serde_json(&val));
                        }
                        Ok(Value::String(serde_json::Value::Object(map).to_string()))
                    }
                    "STDDEV" | "STDDEV_POP" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.is_empty() {
                            Ok(Value::Null)
                        } else {
                            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                            let variance = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                / nums.len() as f64;
                            Ok(Value::Float(variance.sqrt()))
                        }
                    }
                    "STDDEV_SAMP" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.len() < 2 {
                            Ok(Value::Null)
                        } else {
                            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                            let variance = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                / (nums.len() - 1) as f64;
                            Ok(Value::Float(variance.sqrt()))
                        }
                    }
                    "VARIANCE" | "VAR_POP" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.is_empty() {
                            Ok(Value::Null)
                        } else {
                            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                            let variance = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                / nums.len() as f64;
                            Ok(Value::Float(variance))
                        }
                    }
                    "VAR_SAMP" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.len() < 2 {
                            Ok(Value::Null)
                        } else {
                            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
                            let variance = nums.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                                / (nums.len() - 1) as f64;
                            Ok(Value::Float(variance))
                        }
                    }
                    "MEDIAN" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        let mut nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.is_empty() {
                            Ok(Value::Null)
                        } else {
                            nums.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let mid = nums.len() / 2;
                            if nums.len() % 2 == 0 {
                                Ok(Value::Float((nums[mid - 1] + nums[mid]) / 2.0))
                            } else {
                                Ok(Value::Float(nums[mid]))
                            }
                        }
                    }
                    "APPROX_COUNT_DISTINCT" => {
                        if args.is_empty() {
                            return Ok(Value::Null);
                        }
                        // HyperLogLog with precision=14, m=16384 registers
                        const P: u32 = 14;
                        const M: usize = 1 << P; // 16384
                        let mut registers = vec![0u8; M];
                        for row in group_rows {
                            let val = self.evaluate_expression(&args[0], row, context)?;
                            if matches!(val, Value::Null) {
                                continue;
                            }
                            let val_str = format!("{:?}", val);
                            // FNV-1a hash
                            let mut h: u64 = 0xcbf29ce484222325;
                            for b in val_str.as_bytes() {
                                h ^= *b as u64;
                                h = h.wrapping_mul(0x100000001b3);
                            }
                            let idx = (h as usize) & (M - 1);
                            let w = h >> P;
                            let rho = if w == 0 {
                                (64 - P) as u8 + 1
                            } else {
                                (w.leading_zeros() as u8) + 1
                            };
                            if rho > registers[idx] {
                                registers[idx] = rho;
                            }
                        }
                        // Estimate with alpha_m correction
                        let alpha_m = 0.7213 / (1.0 + 1.079 / M as f64);
                        let sum: f64 = registers.iter().map(|&r| 2.0_f64.powi(-(r as i32))).sum();
                        let estimate = alpha_m * (M as f64) * (M as f64) / sum;
                        // Small range correction (linear counting)
                        let result = if estimate <= 2.5 * M as f64 {
                            let zeros = registers.iter().filter(|&&r| r == 0).count();
                            if zeros > 0 {
                                (M as f64) * ((M as f64) / (zeros as f64)).ln()
                            } else {
                                estimate
                            }
                        } else {
                            estimate
                        };
                        Ok(Value::Int(result.round() as i64))
                    }
                    "APPROX_PERCENTILE" | "PERCENTILE_APPROX" => {
                        // APPROX_PERCENTILE(column, percentile)
                        // percentile is 0.0-1.0, sort values and pick nearest rank
                        if args.len() < 2 {
                            return Ok(Value::Null);
                        }
                        let percentile = group_rows
                            .first()
                            .and_then(|row| self.evaluate_expression(&args[1], row, context).ok())
                            .and_then(|v| v.as_float())
                            .unwrap_or(0.5);
                        let mut nums: Vec<f64> = group_rows
                            .iter()
                            .filter_map(|row| {
                                self.evaluate_expression(&args[0], row, context)
                                    .ok()
                                    .and_then(|v| v.as_float())
                            })
                            .collect();
                        if nums.is_empty() {
                            Ok(Value::Null)
                        } else {
                            nums.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let idx = (percentile * (nums.len() as f64 - 1.0)).round() as usize;
                            let idx = idx.min(nums.len() - 1);
                            Ok(Value::Float(nums[idx]))
                        }
                    }
                    _ => {
                        // Non-aggregate function, evaluate on first row
                        if let Some(row) = group_rows.first() {
                            self.evaluate_expression(expr, row, context)
                        } else {
                            Ok(Value::Null)
                        }
                    }
                }
            }
            Expression::Column(name) => {
                // For GROUP BY columns, return value from first row
                if let Some(row) = group_rows.first() {
                    Ok(row.get(name).cloned().unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            _ => {
                // For other expressions, evaluate on first row
                if let Some(row) = group_rows.first() {
                    self.evaluate_expression(expr, row, context)
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Project columns from rows into a ResultSet
    fn project_columns(
        &self,
        rows: Vec<RowData>,
        columns: &[crate::sql::SqlColumn],
        context: &QueryContext,
    ) -> QueryResult<ResultSet> {
        // Determine result column names
        let col_names: Vec<String> = if columns
            .iter()
            .any(|c| matches!(c.expr, Expression::Wildcard | Expression::QualifiedWildcard(_)))
        {
            // If wildcard, use columns from first row
            if let Some(row) = rows.first() {
                row.columns.clone()
            } else {
                Vec::new()
            }
        } else {
            columns
                .iter()
                .map(|c| {
                    c.alias.clone().unwrap_or_else(|| match &c.expr {
                        Expression::Column(name) => name.clone(),
                        Expression::QualifiedColumn { column, .. } => column.clone(),
                        Expression::Function { name, .. } => name.clone(),
                        _ => "expr".to_string(),
                    })
                })
                .collect()
        };

        let mut result = ResultSet::with_columns(col_names.clone());

        for row in rows {
            let values: Vec<Value> = if columns
                .iter()
                .any(|c| matches!(c.expr, Expression::Wildcard | Expression::QualifiedWildcard(_)))
            {
                row.values.clone()
            } else {
                columns
                    .iter()
                    .zip(col_names.iter())
                    .map(|(c, col_name)| {
                        // If this column already exists in the row (e.g., from aggregation),
                        // use that value instead of re-evaluating the expression.
                        // This handles the case where apply_grouping has already computed
                        // aggregate values like COUNT, SUM, etc.
                        if let Some(val) = row.get(col_name) {
                            Ok(val.clone())
                        } else {
                            self.evaluate_expression(&c.expr, &row, context)
                        }
                    })
                    .collect::<QueryResult<Vec<_>>>()?
            };

            result.add_row(Row::new(values));
        }

        Ok(result)
    }

    // ========================================================================
    // Expression Evaluation
    // ========================================================================

    /// Evaluate a predicate expression to a boolean
    fn evaluate_predicate(
        &self,
        expr: &Expression,
        row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<bool> {
        let value = self.evaluate_expression(expr, row, context)?;
        match value {
            Value::Bool(b) => Ok(b),
            Value::Null => Ok(false),
            _ => Ok(true), // Non-null values are truthy
        }
    }

    /// Execute a subquery expression (for scalar subqueries, IN, EXISTS).
    ///
    /// Scans the subquery's source table, filters rows using the subquery's
    /// WHERE clause (with access to the outer row for correlated subqueries),
    /// and projects the requested columns.
    fn execute_subquery_expr(
        &self,
        query: &Query,
        outer_row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<Vec<RowData>> {
        if context.subquery_depth >= crate::execution::MAX_SUBQUERY_DEPTH {
            return Err(QueryError::ExecutionError(format!(
                "Subquery nesting depth exceeds maximum ({})",
                crate::execution::MAX_SUBQUERY_DEPTH
            )));
        }
        let deeper = QueryContext {
            subquery_depth: context.subquery_depth + 1,
            ..context.clone()
        };
        let context = &deeper;
        let table_name = match &query.source {
            Some(name) => name.as_str(),
            None => return Ok(Vec::new()),
        };

        let rows = self.scan_table(table_name)?;

        // Get table columns for building combined rows
        let table_cols = if let Some(first) = rows.first() {
            first.columns.clone()
        } else {
            return Ok(Vec::new());
        };

        let mut result = Vec::new();

        for inner_row in &rows {
            // Build combined row: outer columns + inner columns (for correlated subqueries)
            let mut combined_cols = outer_row.columns.clone();
            combined_cols.extend(inner_row.columns.iter().cloned());
            let mut combined_vals = outer_row.values.clone();
            combined_vals.extend(inner_row.values.iter().cloned());
            let combined = RowData::new(combined_cols, combined_vals);

            // Apply WHERE filter
            let matches = if let Some(ref filter) = query.filter {
                self.evaluate_predicate(filter, &combined, context)?
            } else {
                true
            };

            if matches {
                // Project columns
                if query.columns.contains(&"*".to_string()) || query.columns.is_empty() {
                    result.push(inner_row.clone());
                } else {
                    let proj_cols: Vec<String> = query.columns.clone();
                    let proj_vals: Vec<Value> = query
                        .columns
                        .iter()
                        .map(|col| {
                            // Check derived_columns first (e.g. SELECT 15 → col_0 = Literal(15))
                            if let Some(expr) = query.derived_columns.get(col) {
                                self.evaluate_expression(expr, &combined, context)
                                    .unwrap_or(Value::Null)
                            } else {
                                inner_row.get(col).cloned().unwrap_or(Value::Null)
                            }
                        })
                        .collect();
                    result.push(RowData::new(proj_cols, proj_vals));
                }
            }
        }

        // Apply LIMIT
        if let Some(limit) = query.limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    /// Evaluate an expression
    fn evaluate_expression(
        &self,
        expr: &Expression,
        row: &RowData,
        context: &QueryContext,
    ) -> QueryResult<Value> {
        match expr {
            Expression::Literal(v) => Ok(v.clone()),
            Expression::Column(name) => Ok(row.get(name).cloned().unwrap_or(Value::Null)),
            Expression::QualifiedColumn { table, column } => Ok(row
                .get_qualified(table, column)
                .cloned()
                .unwrap_or(Value::Null)),
            Expression::Binary { left, op, right } => {
                let lval = self.evaluate_expression(left, row, context)?;
                let rval = self.evaluate_expression(right, row, context)?;
                self.evaluate_binary_op(&lval, op, &rval)
            }
            Expression::Unary { op, expr } => {
                let val = self.evaluate_expression(expr, row, context)?;
                self.evaluate_unary_op(op, &val)
            }
            Expression::IsNull { expr, negated } => {
                let val = self.evaluate_expression(expr, row, context)?;
                let is_null = matches!(val, Value::Null);
                Ok(Value::Bool(if *negated { !is_null } else { is_null }))
            }
            Expression::In {
                expr,
                list,
                negated,
            } => {
                let val = self.evaluate_expression(expr, row, context)?;
                // SQL three-valued logic for IN:
                // - If val is NULL, result is NULL (unless list is empty → false)
                // - If any list item matches val, result is true
                // - If any list item is NULL and no match found, result is NULL
                // - Otherwise result is false
                if matches!(val, Value::Null) && !list.is_empty() {
                    return Ok(Value::Null);
                }
                let mut found = false;
                let mut has_null = false;
                for item in list {
                    if let Expression::Subquery(subquery) = item {
                        let results = self.execute_subquery_expr(subquery, row, context)?;
                        for r in &results {
                            match r.values.first() {
                                Some(Value::Null) => has_null = true,
                                Some(v) if *v == val => {
                                    found = true;
                                    break;
                                }
                                _ => {}
                            }
                        }
                        if found {
                            break;
                        }
                    } else {
                        let item_val = self.evaluate_expression(item, row, context)?;
                        if matches!(item_val, Value::Null) {
                            has_null = true;
                        } else if val == item_val {
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    Ok(Value::Bool(!*negated))
                } else if has_null {
                    Ok(Value::Null) // UNKNOWN
                } else {
                    Ok(Value::Bool(*negated))
                }
            }
            Expression::WindowFunction { .. } => Err(QueryError::ExecutionError(
                "Window functions should be handled by the Window operator".to_string(),
            )),
            Expression::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let val = self.evaluate_expression(expr, row, context)?;
                let low_val = self.evaluate_expression(low, row, context)?;
                let high_val = self.evaluate_expression(high, row, context)?;

                // SQL three-valued logic: if any operand is NULL, result is NULL
                if matches!(val, Value::Null)
                    || matches!(low_val, Value::Null)
                    || matches!(high_val, Value::Null)
                {
                    return Ok(Value::Null);
                }

                let in_range = self.compare_values(&val, &low_val) != std::cmp::Ordering::Less
                    && self.compare_values(&val, &high_val) != std::cmp::Ordering::Greater;

                Ok(Value::Bool(if *negated { !in_range } else { in_range }))
            }
            Expression::Like {
                expr,
                pattern,
                negated,
                case_insensitive,
            } => {
                let val = self.evaluate_expression(expr, row, context)?;
                if let Value::String(s) = val {
                    let (s, pat) = if *case_insensitive {
                        (s.to_lowercase(), pattern.to_lowercase())
                    } else {
                        (s, pattern.clone())
                    };
                    // Convert SQL LIKE pattern to regex: escape regex metacharacters,
                    // but preserve SQL wildcards % and _
                    let mut regex_pattern = String::with_capacity(pat.len() * 2);
                    for ch in pat.chars() {
                        match ch {
                            '%' => regex_pattern.push_str(".*"),
                            '_' => regex_pattern.push('.'),
                            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{'
                            | '}' | '|' | '\\' => {
                                regex_pattern.push('\\');
                                regex_pattern.push(ch);
                            }
                            _ => regex_pattern.push(ch),
                        }
                    }
                    let matches = regex::Regex::new(&format!("^{}$", regex_pattern))
                        .map(|re| re.is_match(&s))
                        .unwrap_or(false);
                    Ok(Value::Bool(if *negated { !matches } else { matches }))
                } else if matches!(val, Value::Null) {
                    // SQL: NULL LIKE pattern → NULL
                    Ok(Value::Null)
                } else {
                    Ok(Value::Bool(false))
                }
            }
            Expression::RegexMatch { expr, pattern, negated } => {
                let val = self.evaluate_expression(expr, row, context)?;
                if let Value::String(s) = val {
                    let matched = regex::Regex::new(pattern)
                        .map(|re| re.is_match(&s))
                        .unwrap_or(false);
                    Ok(Value::Bool(if *negated { !matched } else { matched }))
                } else if matches!(val, Value::Null) {
                    Ok(Value::Null)
                } else {
                    Ok(Value::Bool(false))
                }
            }
            Expression::Function { name, args } => {
                // For aggregate functions, first try to resolve as a column reference.
                // After GROUP BY, aggregate results are stored as columns named by
                // the function name (e.g., "SUM", "COUNT"). This makes HAVING work.
                let upper = name.to_uppercase();
                if matches!(
                    upper.as_str(),
                    "SUM" | "AVG" | "MIN" | "MAX" | "COUNT" | "COUNT_DISTINCT"
                ) {
                    if let Some(val) = row.get(&upper) {
                        return Ok(val.clone());
                    }
                    // Also try the original case
                    if let Some(val) = row.get(name) {
                        return Ok(val.clone());
                    }
                }
                let arg_values: Vec<Value> = args
                    .iter()
                    .map(|a| self.evaluate_expression(a, row, context))
                    .collect::<QueryResult<Vec<_>>>()?;
                self.evaluate_function(name, &arg_values)
            }
            Expression::Parameter(idx) => Ok(context
                .positional_params
                .get(*idx)
                .cloned()
                .unwrap_or(Value::Null)),
            Expression::NamedParameter(name) => {
                Ok(context.parameters.get(name).cloned().unwrap_or(Value::Null))
            }
            Expression::Wildcard | Expression::QualifiedWildcard(_) => Ok(Value::Null),
            Expression::Case {
                operand,
                when_clauses,
                else_clause,
            } => {
                if let Some(op) = operand {
                    let op_val = self.evaluate_expression(op, row, context)?;
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = self.evaluate_expression(when_expr, row, context)?;
                        if self.values_equal(&op_val, &when_val) {
                            return self.evaluate_expression(then_expr, row, context);
                        }
                    }
                } else {
                    for (when_expr, then_expr) in when_clauses {
                        if self.evaluate_predicate(when_expr, row, context)? {
                            return self.evaluate_expression(then_expr, row, context);
                        }
                    }
                }
                if let Some(else_expr) = else_clause {
                    self.evaluate_expression(else_expr, row, context)
                } else {
                    Ok(Value::Null)
                }
            }
            Expression::Subquery(subquery) => {
                // Scalar subquery: return first column of first row
                let results = self.execute_subquery_expr(subquery, row, context)?;
                if let Some(first_row) = results.first() {
                    Ok(first_row.values.first().cloned().unwrap_or(Value::Null))
                } else {
                    Ok(Value::Null)
                }
            }
            Expression::Exists(subquery) => {
                let results = self.execute_subquery_expr(subquery, row, context)?;
                Ok(Value::Bool(!results.is_empty()))
            }
            // JouleDB semantic query extensions
            Expression::SimilarTo {
                expr,
                pattern,
                threshold,
                negated,
            } => {
                let val = self.evaluate_expression(expr, row, context)?;
                if let Value::String(s) = val {
                    // Use normalized Levenshtein distance for fuzzy matching
                    let similarity = self.levenshtein_similarity(&s, pattern);
                    let thresh = threshold.unwrap_or(0.8);
                    let matches = similarity >= thresh;
                    Ok(Value::Bool(if *negated { !matches } else { matches }))
                } else {
                    Ok(Value::Bool(false))
                }
            }
            Expression::LikeMeaning {
                expr,
                concept,
                negated,
            } => {
                let val = self.evaluate_expression(expr, row, context)?;
                if let Value::String(s) = val {
                    // Semantic similarity - placeholder for AmorphicEngine integration
                    // For now, use simple keyword matching as approximation
                    let concept_lower = concept.to_lowercase();
                    let s_lower = s.to_lowercase();
                    let matches = s_lower.contains(&concept_lower)
                        || concept_lower
                            .split_whitespace()
                            .any(|word| s_lower.contains(word));
                    Ok(Value::Bool(if *negated { !matches } else { matches }))
                } else {
                    Ok(Value::Bool(false))
                }
            }
            Expression::Cast { expr, target_type } => {
                let val = self.evaluate_expression(expr, row, context)?;
                self.cast_value(&val, target_type)
            }
            Expression::ReverseReference { reference_name } => {
                Ok(Value::String(format!("~>{}", reference_name)))
            }
        }
    }

    fn cast_value(&self, val: &Value, target_type: &str) -> QueryResult<Value> {
        match target_type {
            "INT" | "INTEGER" | "BIGINT" => match val {
                Value::Int(_) => Ok(val.clone()),
                Value::Float(f) => Ok(Value::Int(*f as i64)),
                Value::String(s) => s
                    .parse::<i64>()
                    .map(Value::Int)
                    .or_else(|_| s.parse::<f64>().map(|f| Value::Int(f as i64)))
                    .map_err(|_| QueryError::TypeError(format!("Cannot cast '{}' to INT", s))),
                Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::Null),
            },
            "FLOAT" | "DOUBLE" | "REAL" | "NUMERIC" | "DECIMAL" => match val {
                Value::Float(_) => Ok(val.clone()),
                Value::Int(i) => Ok(Value::Float(*i as f64)),
                Value::String(s) => s
                    .parse::<f64>()
                    .map(Value::Float)
                    .map_err(|_| QueryError::TypeError(format!("Cannot cast '{}' to FLOAT", s))),
                Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::Null),
            },
            "TEXT" | "VARCHAR" | "STRING" | "CHAR" => match val {
                Value::String(_) => Ok(val.clone()),
                Value::Int(i) => Ok(Value::String(i.to_string())),
                Value::Float(f) => Ok(Value::String(f.to_string())),
                Value::Bool(b) => Ok(Value::String(b.to_string())),
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::String(format!("{:?}", val))),
            },
            "BOOL" | "BOOLEAN" => match val {
                Value::Bool(_) => Ok(val.clone()),
                Value::Int(i) => Ok(Value::Bool(*i != 0)),
                Value::Float(f) => Ok(Value::Bool(*f != 0.0)),
                Value::String(s) => match s.to_uppercase().as_str() {
                    "TRUE" | "1" | "YES" | "T" => Ok(Value::Bool(true)),
                    "FALSE" | "0" | "NO" | "F" => Ok(Value::Bool(false)),
                    _ => Err(QueryError::TypeError(format!(
                        "Cannot cast '{}' to BOOL",
                        s
                    ))),
                },
                Value::Null => Ok(Value::Null),
                _ => Ok(Value::Null),
            },
            _ => Err(QueryError::TypeError(format!(
                "Unknown target type: {}",
                target_type
            ))),
        }
    }

    /// Evaluate binary operator
    fn evaluate_binary_op(&self, left: &Value, op: &Operator, right: &Value) -> QueryResult<Value> {
        // SQL NULL semantics: ordered comparisons with NULL yield NULL
        if matches!(left, Value::Null) || matches!(right, Value::Null) {
            match op {
                Operator::Lt | Operator::Le | Operator::Gt | Operator::Ge => {
                    return Ok(Value::Null);
                }
                _ => {}
            }
        }
        match op {
            Operator::Eq => Ok(Value::Bool(self.values_equal(left, right))),
            Operator::Ne => Ok(Value::Bool(!self.values_equal(left, right))),
            Operator::Lt => Ok(Value::Bool(
                self.compare_values(left, right) == std::cmp::Ordering::Less,
            )),
            Operator::Le => Ok(Value::Bool(
                self.compare_values(left, right) != std::cmp::Ordering::Greater,
            )),
            Operator::Gt => Ok(Value::Bool(
                self.compare_values(left, right) == std::cmp::Ordering::Greater,
            )),
            Operator::Ge => Ok(Value::Bool(
                self.compare_values(left, right) != std::cmp::Ordering::Less,
            )),
            Operator::And => {
                let lb = left.as_bool().unwrap_or(false);
                let rb = right.as_bool().unwrap_or(false);
                Ok(Value::Bool(lb && rb))
            }
            Operator::Or => {
                let lb = left.as_bool().unwrap_or(false);
                let rb = right.as_bool().unwrap_or(false);
                Ok(Value::Bool(lb || rb))
            }
            Operator::Add => {
                match (left, right) {
                    (Value::Timestamp(ts), Value::Int(n))
                    | (Value::Int(n), Value::Timestamp(ts)) => {
                        return Ok(Value::Timestamp(ts + n));
                    }
                    _ => {}
                }
                self.numeric_op(left, right, |a, b| a + b, |a, b| a + b)
            }
            Operator::Sub => {
                match (left, right) {
                    (Value::Timestamp(a), Value::Timestamp(b)) => {
                        return Ok(Value::Int(a - b));
                    }
                    (Value::Timestamp(ts), Value::Int(n)) => {
                        return Ok(Value::Timestamp(ts - n));
                    }
                    _ => {}
                }
                self.numeric_op(left, right, |a, b| a - b, |a, b| a - b)
            }
            Operator::Mul => self.numeric_op(left, right, |a, b| a * b, |a, b| a * b),
            Operator::Div => {
                if let Some(r) = right.as_float() {
                    if r == 0.0 {
                        return Err(QueryError::ExecutionError("Division by zero".to_string()));
                    }
                }
                self.numeric_op(
                    left,
                    right,
                    |a, b| if b != 0 { a / b } else { 0 },
                    |a, b| a / b,
                )
            }
            Operator::Mod => {
                if let Some(r) = right.as_float() {
                    if r == 0.0 {
                        return Err(QueryError::ExecutionError("Division by zero".to_string()));
                    }
                }
                self.numeric_op(
                    left,
                    right,
                    |a, b| if b != 0 { a % b } else { 0 },
                    |a, b| if b != 0.0 { a % b } else { f64::NAN },
                )
            }
            Operator::Concat => {
                let ls = match left {
                    Value::String(s) => s.clone(),
                    Value::Null => return Ok(Value::Null),
                    v => format!("{:?}", v),
                };
                let rs = match right {
                    Value::String(s) => s.clone(),
                    Value::Null => return Ok(Value::Null),
                    v => format!("{:?}", v),
                };
                Ok(Value::String(format!("{}{}", ls, rs)))
            }
            Operator::BitAnd | Operator::BitOr | Operator::BitXor => {
                match (left.as_int(), right.as_int()) {
                    (Some(a), Some(b)) => {
                        let result = match op {
                            Operator::BitAnd => a & b,
                            Operator::BitOr => a | b,
                            Operator::BitXor => a ^ b,
                            _ => unreachable!(),
                        };
                        Ok(Value::Int(result))
                    }
                    _ => Ok(Value::Null),
                }
            }
            Operator::JsonArrow
            | Operator::JsonDoubleArrow
            | Operator::JsonHashArrow
            | Operator::JsonHashDoubleArrow
            | Operator::JsonContains
            | Operator::JsonContainedBy
            | Operator::JsonExists => Ok(crate::ast::eval_json_operator(&left, op, &right)),

            // Vector distance operators — delegate to shared implementation
            Operator::VectorL2Distance
            | Operator::VectorIPDistance
            | Operator::VectorCosineDistance => crate::functions::eval_binary_op(&left, op, &right),
        }
    }

    /// Evaluate unary operator
    fn evaluate_unary_op(&self, op: &UnaryOperator, val: &Value) -> QueryResult<Value> {
        match op {
            UnaryOperator::Not => {
                let b = val.as_bool().unwrap_or(false);
                Ok(Value::Bool(!b))
            }
            UnaryOperator::Neg => {
                if let Some(i) = val.as_int() {
                    Ok(Value::Int(-i))
                } else if let Some(f) = val.as_float() {
                    Ok(Value::Float(-f))
                } else {
                    Ok(Value::Null)
                }
            }
            UnaryOperator::BitNot => {
                if let Some(i) = val.as_int() {
                    Ok(Value::Int(!i))
                } else {
                    Ok(Value::Null)
                }
            }
        }
    }

    /// Numeric operation helper
    fn numeric_op<F, G>(
        &self,
        left: &Value,
        right: &Value,
        int_op: F,
        float_op: G,
    ) -> QueryResult<Value>
    where
        F: Fn(i64, i64) -> i64,
        G: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(*a as f64, *b))),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(*a, *b as f64))),
            _ => Ok(Value::Null),
        }
    }

    /// Evaluate a function call
    fn evaluate_function(&self, name: &str, args: &[Value]) -> QueryResult<Value> {
        let scanner = |table: &str| -> QueryResult<(Vec<String>, Vec<Vec<serde_json::Value>>)> {
            let rows = self.scan_table(table)?;
            Ok(self.rows_to_json_columns(&rows))
        };
        crate::functions::eval_scalar_function(name, args, Some(&scanner))
    }

    /// Convert scanned RowData to column names and JSON row vectors for graph functions
    fn rows_to_json_columns(&self, rows: &[RowData]) -> (Vec<String>, Vec<Vec<serde_json::Value>>) {
        if rows.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let columns = rows[0].columns.clone();
        let json_rows: Vec<Vec<serde_json::Value>> = rows
            .iter()
            .map(|row| {
                row.values
                    .iter()
                    .map(|v| SerializedRow::value_to_json(v))
                    .collect()
            })
            .collect();
        (columns, json_rows)
    }

    /// Compare two values for equality
    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Int(a), Value::Float(b)) => (*a as f64 - b).abs() < f64::EPSILON,
            (Value::Float(a), Value::Int(b)) => (a - *b as f64).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Bytes(a), Value::Bytes(b)) => a == b,
            (Value::Timestamp(a), Value::Timestamp(b)) => a == b,
            (Value::Uuid(a), Value::Uuid(b)) => a == b,
            _ => false,
        }
    }

    /// Compare two values for ordering
    fn compare_values(&self, a: &Value, b: &Value) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (a, b) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Int(a), Value::Float(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (Value::Float(a), Value::Int(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }

    /// Calculate normalized Levenshtein similarity (0.0 - 1.0)
    /// Used for SIMILAR TO fuzzy matching
    fn levenshtein_similarity(&self, s1: &str, s2: &str) -> f64 {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 && len2 == 0 {
            return 1.0;
        }
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        // Compute Levenshtein distance
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();

        let mut prev_row: Vec<usize> = (0..=len2).collect();
        let mut curr_row: Vec<usize> = vec![0; len2 + 1];

        for (i, c1) in s1_chars.iter().enumerate() {
            curr_row[0] = i + 1;
            for (j, c2) in s2_chars.iter().enumerate() {
                let cost = if c1.to_lowercase().eq(c2.to_lowercase()) {
                    0
                } else {
                    1
                };
                curr_row[j + 1] = (prev_row[j + 1] + 1)
                    .min(curr_row[j] + 1)
                    .min(prev_row[j] + cost);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        let distance = prev_row[len2];
        let max_len = len1.max(len2);
        1.0 - (distance as f64 / max_len as f64)
    }
}

/// Convert serde_json::Value to ast::Value
fn json_to_ast_value(v: serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Int(if b { 1 } else { 0 }),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            Value::Array(arr.into_iter().map(json_to_ast_value).collect())
        }
        serde_json::Value::Object(obj) => Value::Object(
            obj.into_iter()
                .map(|(k, v)| (k, json_to_ast_value(v)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_core::storage::memory::MemoryBackend;

    fn create_test_executor() -> StorageExecutor {
        let backend = MemoryBackend::new();
        let engine = Arc::new(Engine::new(backend).unwrap());
        StorageExecutor::new(engine)
    }

    #[test]
    fn test_create_table() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        let result = executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
                &context,
            )
            .unwrap();

        assert_eq!(result.affected_rows, 0);
        assert!(executor.table_exists("users").unwrap());
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)",
                &context,
            )
            .unwrap();

        // Second create should succeed without error
        executor
            .execute_sql(
                "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)",
                &context,
            )
            .unwrap();

        assert!(executor.table_exists("users").unwrap());
    }

    #[test]
    fn test_insert_and_select() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        // Create table
        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();

        // Insert rows
        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (2, 'Bob')", &context)
            .unwrap();

        // Select all
        let result = executor
            .execute_sql("SELECT * FROM users", &context)
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_select_with_where() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO users (id, name, age) VALUES (3, 'Charlie', 35)",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql("SELECT * FROM users WHERE age > 25", &context)
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_update() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "UPDATE users SET name = 'Alice Updated' WHERE id = 1",
                &context,
            )
            .unwrap();

        assert_eq!(result.affected_rows, 1);

        let select_result = executor
            .execute_sql("SELECT name FROM users WHERE id = 1", &context)
            .unwrap();

        assert_eq!(
            select_result.result_set.rows[0].values[0],
            Value::String("Alice Updated".to_string())
        );
    }

    #[test]
    fn test_delete() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (2, 'Bob')", &context)
            .unwrap();

        let result = executor
            .execute_sql("DELETE FROM users WHERE id = 1", &context)
            .unwrap();

        assert_eq!(result.affected_rows, 1);

        let select_result = executor
            .execute_sql("SELECT * FROM users", &context)
            .unwrap();

        assert_eq!(select_result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_order_by() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO users (id, name) VALUES (1, 'Charlie')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (2, 'Alice')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (3, 'Bob')", &context)
            .unwrap();

        let result = executor
            .execute_sql("SELECT * FROM users ORDER BY name ASC", &context)
            .unwrap();

        assert_eq!(
            result.result_set.rows[0].values[1],
            Value::String("Alice".to_string())
        );
        assert_eq!(
            result.result_set.rows[1].values[1],
            Value::String("Bob".to_string())
        );
        assert_eq!(
            result.result_set.rows[2].values[1],
            Value::String("Charlie".to_string())
        );
    }

    #[test]
    fn test_limit_offset() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql("CREATE TABLE items (id INTEGER PRIMARY KEY)", &context)
            .unwrap();

        for i in 1..=10 {
            executor
                .execute_sql(&format!("INSERT INTO items (id) VALUES ({})", i), &context)
                .unwrap();
        }

        let result = executor
            .execute_sql("SELECT * FROM items ORDER BY id LIMIT 3 OFFSET 2", &context)
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 3);
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(3));
    }

    #[test]
    fn test_aggregation_count() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, dept TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO users (id, dept) VALUES (1, 'Engineering')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO users (id, dept) VALUES (2, 'Engineering')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, dept) VALUES (3, 'Sales')", &context)
            .unwrap();

        let result = executor
            .execute_sql("SELECT COUNT(*) FROM users", &context)
            .unwrap();

        assert_eq!(result.result_set.rows[0].values[0], Value::Int(3));
    }

    #[test]
    fn test_aggregation_sum_avg() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, amount DOUBLE)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO orders (id, amount) VALUES (1, 100.0)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO orders (id, amount) VALUES (2, 200.0)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO orders (id, amount) VALUES (3, 300.0)",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql("SELECT SUM(amount), AVG(amount) FROM orders", &context)
            .unwrap();

        assert_eq!(result.result_set.rows[0].values[0], Value::Float(600.0));
        assert_eq!(result.result_set.rows[0].values[1], Value::Float(200.0));
    }

    #[test]
    fn test_inner_join() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, total REAL)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO users (id, name) VALUES (2, 'Bob')", &context)
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO orders (id, user_id, total) VALUES (1, 1, 100.0)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO orders (id, user_id, total) VALUES (2, 1, 200.0)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO orders (id, user_id, total) VALUES (3, 2, 150.0)",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id",
                &context,
            )
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 3);
    }

    #[test]
    fn test_drop_table() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql("CREATE TABLE temp (id INTEGER PRIMARY KEY)", &context)
            .unwrap();

        assert!(executor.table_exists("temp").unwrap());

        executor.execute_sql("DROP TABLE temp", &context).unwrap();

        assert!(!executor.table_exists("temp").unwrap());
    }

    #[test]
    fn test_expression_evaluation() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE calc (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql("INSERT INTO calc (id, a, b) VALUES (1, 10, 3)", &context)
            .unwrap();

        // Test addition
        let result = executor
            .execute_sql("SELECT a + b FROM calc", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(13));

        // Test multiplication
        let result = executor
            .execute_sql("SELECT a * b FROM calc", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(30));

        // Test division
        let result = executor
            .execute_sql("SELECT a / b FROM calc", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(3));
    }

    #[test]
    fn test_string_functions() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE strings (id INTEGER PRIMARY KEY, s TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO strings (id, s) VALUES (1, 'Hello World')",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT UPPER(s), LOWER(s), LENGTH(s) FROM strings",
                &context,
            )
            .unwrap();

        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("HELLO WORLD".to_string())
        );
        assert_eq!(
            result.result_set.rows[0].values[1],
            Value::String("hello world".to_string())
        );
        assert_eq!(result.result_set.rows[0].values[2], Value::Int(11));
    }

    #[test]
    fn test_coalesce_nullif() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE nulltest (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql("INSERT INTO nulltest (id, val) VALUES (1, NULL)", &context)
            .unwrap();

        let result = executor
            .execute_sql("SELECT COALESCE(val, 42) FROM nulltest", &context)
            .unwrap();

        assert_eq!(result.result_set.rows[0].values[0], Value::Int(42));
    }

    #[test]
    fn test_serialized_row() {
        let row = RowData::new(
            vec!["id".to_string(), "name".to_string()],
            vec![Value::Int(1), Value::String("Alice".to_string())],
        );

        let serialized = row.to_serialized();
        let bytes = serialized.to_bytes().unwrap();
        let deserialized = SerializedRow::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized.columns, vec!["id", "name"]);
        assert_eq!(deserialized.get("id"), Some(Value::Int(1)));
        assert_eq!(
            deserialized.get("name"),
            Some(Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_having_filters_groups() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, product TEXT, qty INTEGER)",
                &context,
            )
            .unwrap();

        for (i, (product, qty)) in [("A", 10), ("A", 20), ("B", 5), ("C", 100)]
            .iter()
            .enumerate()
        {
            executor
                .execute_sql(
                    &format!(
                        "INSERT INTO orders (id, product, qty) VALUES ({}, '{}', {})",
                        i + 1,
                        product,
                        qty
                    ),
                    &context,
                )
                .unwrap();
        }

        let result = executor
            .execute_sql(
                "SELECT product, SUM(qty) FROM orders GROUP BY product HAVING SUM(qty) > 10",
                &context,
            )
            .unwrap();

        // A: 30, B: 5, C: 100  =>  HAVING SUM(qty) > 10 keeps A and C
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_having_with_count() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
                &context,
            )
            .unwrap();

        for (i, (cat, val)) in [("x", 1), ("x", 2), ("x", 3), ("y", 10)].iter().enumerate() {
            executor
                .execute_sql(
                    &format!(
                        "INSERT INTO events (id, category, val) VALUES ({}, '{}', {})",
                        i + 1,
                        cat,
                        val
                    ),
                    &context,
                )
                .unwrap();
        }

        let result = executor
            .execute_sql(
                "SELECT category, COUNT(*) FROM events GROUP BY category HAVING COUNT(*) >= 3",
                &context,
            )
            .unwrap();

        // x: 3 rows, y: 1 row  =>  HAVING COUNT(*) >= 3 keeps only x
        assert_eq!(result.result_set.rows.len(), 1);
    }

    // ===================== Subquery Tests =====================

    /// Helper: create executor with two tables for subquery tests
    fn setup_subquery_tables() -> (StorageExecutor, QueryContext) {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER, qty INTEGER)",
                &context,
            )
            .unwrap();

        for (id, name, price) in &[(1, "Widget", 10), (2, "Gadget", 25), (3, "Doohickey", 5)] {
            executor
                .execute_sql(
                    &format!(
                        "INSERT INTO products (id, name, price) VALUES ({}, '{}', {})",
                        id, name, price
                    ),
                    &context,
                )
                .unwrap();
        }

        for (id, pid, qty) in &[(1, 1, 100), (2, 2, 50), (3, 1, 200)] {
            executor
                .execute_sql(
                    &format!(
                        "INSERT INTO orders (id, product_id, qty) VALUES ({}, {}, {})",
                        id, pid, qty
                    ),
                    &context,
                )
                .unwrap();
        }

        (executor, context)
    }

    #[test]
    fn test_subquery_in_where() {
        let (mut executor, context) = setup_subquery_tables();

        // SELECT * FROM products WHERE id IN (SELECT product_id FROM orders)
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE id IN (SELECT product_id FROM orders)",
                &context,
            )
            .unwrap();

        // products 1 and 2 have orders; product 3 does not
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_subquery_not_in() {
        let (mut executor, context) = setup_subquery_tables();

        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE id NOT IN (SELECT product_id FROM orders)",
                &context,
            )
            .unwrap();

        // Only product 3 (Doohickey) has no orders
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_exists() {
        let (mut executor, context) = setup_subquery_tables();

        // EXISTS with correlated subquery
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE EXISTS (SELECT 1 FROM orders WHERE orders.product_id = products.id)",
                &context,
            )
            .unwrap();

        // Products 1 and 2 have orders
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_subquery_not_exists() {
        let (mut executor, context) = setup_subquery_tables();

        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.product_id = products.id)",
                &context,
            )
            .unwrap();

        // Only product 3 has no orders
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_scalar_in_where() {
        let (mut executor, context) = setup_subquery_tables();

        // Scalar subquery: price > (average price)
        // Avg price = (10 + 25 + 5) / 3 = 13.33
        // Products with price > 13.33: Gadget (25)
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE price > (SELECT 15 FROM products LIMIT 1)",
                &context,
            )
            .unwrap();

        // Only Gadget (25) has price > 15
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_empty_returns_null() {
        let (mut executor, context) = setup_subquery_tables();

        // Scalar subquery that returns no rows should yield NULL
        // Products WHERE price > NULL is false, so no rows match
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE price > (SELECT price FROM products WHERE id = 999)",
                &context,
            )
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 0);
    }

    #[test]
    fn test_subquery_in_empty_result() {
        let (mut executor, context) = setup_subquery_tables();

        // IN subquery that returns empty set
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE id IN (SELECT product_id FROM orders WHERE qty > 1000)",
                &context,
            )
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 0);
    }

    #[test]
    fn test_subquery_exists_empty_table() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE a (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE b (id INTEGER PRIMARY KEY, ref_id INTEGER)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql("INSERT INTO a (id, val) VALUES (1, 10)", &context)
            .unwrap();

        // b is empty, so EXISTS should return false for all rows in a
        let result = executor
            .execute_sql(
                "SELECT * FROM a WHERE EXISTS (SELECT 1 FROM b WHERE b.ref_id = a.id)",
                &context,
            )
            .unwrap();

        assert_eq!(result.result_set.rows.len(), 0);
    }

    #[test]
    fn test_subquery_correlated_exists_sql() {
        let (mut executor, context) = setup_subquery_tables();

        // Correlated EXISTS: products that have at least one order
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE EXISTS (SELECT 1 FROM orders WHERE orders.product_id = products.id)",
                &context,
            )
            .unwrap();

        // Products 1 and 2 have orders; 3 does not
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_subquery_self_referencing() {
        let (mut executor, context) = setup_subquery_tables();

        // Self-referencing: products with price above average
        // Avg = (10 + 25 + 5) / 3 ≈ 13.33
        // This tests a subquery referencing the same table as the outer query
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE price > (SELECT 15 FROM products LIMIT 1)",
                &context,
            )
            .unwrap();

        // Only Gadget (25) > 15
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_in_with_filter() {
        let (mut executor, context) = setup_subquery_tables();

        // IN subquery with a WHERE clause in the subquery
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE id IN (SELECT product_id FROM orders WHERE qty >= 100)",
                &context,
            )
            .unwrap();

        // Orders with qty >= 100: order 1 (product_id=1, qty=100), order 3 (product_id=1, qty=200)
        // So only product 1 is in the result
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_not_exists_sql() {
        let (mut executor, context) = setup_subquery_tables();

        // NOT EXISTS: products with no orders
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.product_id = products.id)",
                &context,
            )
            .unwrap();

        // Only product 3 (Doohickey) has no orders
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_not_in_sql() {
        let (mut executor, context) = setup_subquery_tables();

        // NOT IN with subquery
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE id NOT IN (SELECT product_id FROM orders)",
                &context,
            )
            .unwrap();

        // Only product 3 has no orders
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_subquery_nested() {
        let (mut executor, context) = setup_subquery_tables();

        // Add a third table for nested subquery
        executor
            .execute_sql(
                "CREATE TABLE categories (id INTEGER PRIMARY KEY, product_id INTEGER, category TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO categories (id, product_id, category) VALUES (1, 1, 'tools')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO categories (id, product_id, category) VALUES (2, 2, 'electronics')",
                &context,
            )
            .unwrap();

        // Nested: products that are in categories AND have orders
        let result = executor
            .execute_sql(
                "SELECT * FROM products WHERE id IN (SELECT product_id FROM categories) AND id IN (SELECT product_id FROM orders)",
                &context,
            )
            .unwrap();

        // Products 1 and 2 are in categories, products 1 and 2 have orders → both match
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_insert_select_basic() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE source (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE target (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO source (id, name) VALUES (1, 'Alice')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO source (id, name) VALUES (2, 'Bob')", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "INSERT INTO target (id, name) SELECT id, name FROM source",
                &context,
            )
            .unwrap();
        assert_eq!(result.affected_rows, 2);

        let result = executor
            .execute_sql("SELECT * FROM target", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_insert_select_with_where() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE active_employees (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO employees (id, name, active) VALUES (1, 'Alice', 1)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO employees (id, name, active) VALUES (2, 'Bob', 0)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO employees (id, name, active) VALUES (3, 'Carol', 1)",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql(
                "INSERT INTO active_employees (id, name) SELECT id, name FROM employees WHERE active = 1",
                &context,
            )
            .unwrap();
        assert_eq!(result.affected_rows, 2);

        let result = executor
            .execute_sql("SELECT name FROM active_employees", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
    }

    #[test]
    fn test_insert_select_column_reorder() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE src (id INTEGER PRIMARY KEY, first TEXT, last TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE dst (id INTEGER PRIMARY KEY, last_name TEXT, first_name TEXT)",
                &context,
            )
            .unwrap();

        executor
            .execute_sql(
                "INSERT INTO src (id, first, last) VALUES (1, 'Alice', 'Smith')",
                &context,
            )
            .unwrap();

        // Reverse column order in SELECT, include id for PK
        let result = executor
            .execute_sql(
                "INSERT INTO dst (id, last_name, first_name) SELECT id, last, first FROM src",
                &context,
            )
            .unwrap();
        assert_eq!(result.affected_rows, 1);

        let result = executor
            .execute_sql("SELECT last_name, first_name FROM dst", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
    }

    #[test]
    fn test_insert_select_row_count() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE nums (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE evens (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();

        for i in 1..=10 {
            executor
                .execute_sql(
                    &format!("INSERT INTO nums (id, val) VALUES ({}, {})", i, i * 10),
                    &context,
                )
                .unwrap();
        }

        // Select only even ids
        let result = executor
            .execute_sql(
                "INSERT INTO evens (id, val) SELECT id, val FROM nums WHERE id IN (2, 4, 6, 8, 10)",
                &context,
            )
            .unwrap();
        assert_eq!(result.affected_rows, 5);

        let result = executor
            .execute_sql("SELECT COUNT(*) FROM evens", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(5));
    }

    // ---- Aggregate correctness tests ----

    #[test]
    fn test_count_col_filters_nulls_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (1, 10)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (2, NULL)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (3, 30)", &context)
            .unwrap();

        // COUNT(*) counts all
        let result = executor
            .execute_sql("SELECT COUNT(*) FROM t", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(3));

        // COUNT(val) only counts non-NULL
        let result = executor
            .execute_sql("SELECT COUNT(val) FROM t", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(2));
    }

    #[test]
    fn test_sum_all_null_returns_null_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (1, NULL)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (2, NULL)", &context)
            .unwrap();

        let result = executor
            .execute_sql("SELECT SUM(val) FROM t", &context)
            .unwrap();
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::Null,
            "SUM of all-NULL should return NULL"
        );
    }

    #[test]
    fn test_group_by_expression_storage_executor() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE grpexpr (id INTEGER PRIMARY KEY, name TEXT, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO grpexpr (id, name, val) VALUES (1, 'alice', 10)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO grpexpr (id, name, val) VALUES (2, 'Alice', 20)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO grpexpr (id, name, val) VALUES (3, 'BOB', 5)",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT UPPER(name), SUM(val) FROM grpexpr GROUP BY UPPER(name)",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        // Collect results into a map for order-independent assertion
        let mut results: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for row in &result.result_set.rows {
            if let Value::String(name) = &row.values[0] {
                if let Value::Float(val) = &row.values[1] {
                    results.insert(name.clone(), *val);
                }
            }
        }
        assert_eq!(results["ALICE"], 30.0);
        assert_eq!(results["BOB"], 5.0);
    }

    #[test]
    fn test_group_by_function_storage_executor() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (1, 10)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (2, 10)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, val) VALUES (3, 20)", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT val, COUNT(*) FROM t GROUP BY val ORDER BY val",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        // val=10 appears twice, val=20 appears once
        assert_eq!(result.result_set.rows[0].values[1], Value::Int(2));
        assert_eq!(result.result_set.rows[1].values[1], Value::Int(1));
    }

    // ---- Step 4: ILIKE + scalar function tests ----

    #[test]
    fn test_greatest_least_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO t (id, a, b, c) VALUES (1, 10, 20, 5)",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql("SELECT GREATEST(a, b, c), LEAST(a, b, c) FROM t", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(20));
        assert_eq!(result.result_set.rows[0].values[1], Value::Int(5));
    }

    #[test]
    fn test_sign_mod_exp_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO t (id, x) VALUES (1, 42)", &context)
            .unwrap();

        let result = executor
            .execute_sql("SELECT SIGN(x), MOD(x, 10) FROM t", &context)
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(1));
        assert_eq!(result.result_set.rows[0].values[1], Value::Int(2));
    }

    // ---- Step 1: CASE WHEN, BETWEEN, JOINs ----

    #[test]
    fn test_case_when_searched_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE cw_s (id INTEGER PRIMARY KEY, val INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO cw_s (id, val) VALUES (1, 15)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cw_s (id, val) VALUES (2, 7)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cw_s (id, val) VALUES (3, 2)", &context)
            .unwrap();

        let result = executor.execute_sql("SELECT val, CASE WHEN val > 10 THEN 'high' WHEN val > 5 THEN 'medium' ELSE 'low' END FROM cw_s ORDER BY val", &context).unwrap();
        assert_eq!(result.result_set.rows.len(), 3);
        assert_eq!(
            result.result_set.rows[0].values[1],
            Value::String("low".to_string())
        );
        assert_eq!(
            result.result_set.rows[1].values[1],
            Value::String("medium".to_string())
        );
        assert_eq!(
            result.result_set.rows[2].values[1],
            Value::String("high".to_string())
        );
    }

    #[test]
    fn test_case_when_simple_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE cw_s2 (id INTEGER PRIMARY KEY, status TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO cw_s2 (id, status) VALUES (1, 'A')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cw_s2 (id, status) VALUES (2, 'I')", &context)
            .unwrap();

        let result = executor.execute_sql("SELECT CASE status WHEN 'A' THEN 'Active' WHEN 'I' THEN 'Inactive' ELSE 'Unknown' END FROM cw_s2 ORDER BY status", &context).unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("Active".to_string())
        );
        assert_eq!(
            result.result_set.rows[1].values[0],
            Value::String("Inactive".to_string())
        );
    }

    #[test]
    fn test_between_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE bt_s (id INTEGER PRIMARY KEY, age INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s (id, age) VALUES (1, 10)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s (id, age) VALUES (2, 25)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s (id, age) VALUES (3, 50)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s (id, age) VALUES (4, 70)", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT age FROM bt_s WHERE age BETWEEN 18 AND 65 ORDER BY age",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(25));
        assert_eq!(result.result_set.rows[1].values[0], Value::Int(50));
    }

    #[test]
    fn test_between_with_expressions() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE bt_s2 (id INTEGER PRIMARY KEY, price REAL)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s2 (id, price) VALUES (1, 5.0)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s2 (id, price) VALUES (2, 50.0)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO bt_s2 (id, price) VALUES (3, 150.0)", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT price FROM bt_s2 WHERE price BETWEEN 10.0 AND 100.0",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(result.result_set.rows[0].values[0], Value::Float(50.0));
    }

    #[test]
    fn test_left_join_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE ljs_emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE ljs_dept (id INTEGER PRIMARY KEY, dept_name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO ljs_emp (id, name, dept_id) VALUES (1, 'Alice', 10)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO ljs_emp (id, name, dept_id) VALUES (2, 'Bob', 99)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO ljs_dept (id, dept_name) VALUES (10, 'Engineering')",
                &context,
            )
            .unwrap();

        let result = executor.execute_sql("SELECT ljs_emp.name, ljs_dept.dept_name FROM ljs_emp LEFT JOIN ljs_dept ON ljs_emp.dept_id = ljs_dept.id ORDER BY ljs_emp.name", &context).unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("Alice".to_string())
        );
        assert_eq!(
            result.result_set.rows[0].values[1],
            Value::String("Engineering".to_string())
        );
        assert_eq!(
            result.result_set.rows[1].values[0],
            Value::String("Bob".to_string())
        );
        assert_eq!(result.result_set.rows[1].values[1], Value::Null);
    }

    #[test]
    fn test_right_join_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE rjs_emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE rjs_dept (id INTEGER PRIMARY KEY, dept_name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO rjs_emp (id, name, dept_id) VALUES (1, 'Alice', 10)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO rjs_dept (id, dept_name) VALUES (10, 'Engineering')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO rjs_dept (id, dept_name) VALUES (20, 'Sales')",
                &context,
            )
            .unwrap();

        let result = executor.execute_sql("SELECT rjs_emp.name, rjs_dept.dept_name FROM rjs_emp RIGHT JOIN rjs_dept ON rjs_emp.dept_id = rjs_dept.id ORDER BY rjs_dept.dept_name", &context).unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("Alice".to_string())
        );
        assert_eq!(
            result.result_set.rows[0].values[1],
            Value::String("Engineering".to_string())
        );
        assert_eq!(result.result_set.rows[1].values[0], Value::Null);
        assert_eq!(
            result.result_set.rows[1].values[1],
            Value::String("Sales".to_string())
        );
    }

    #[test]
    fn test_full_join_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE fjs_emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE fjs_dept (id INTEGER PRIMARY KEY, dept_name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO fjs_emp (id, name, dept_id) VALUES (1, 'Alice', 10)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO fjs_emp (id, name, dept_id) VALUES (2, 'Bob', 99)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO fjs_dept (id, dept_name) VALUES (10, 'Engineering')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO fjs_dept (id, dept_name) VALUES (30, 'HR')",
                &context,
            )
            .unwrap();

        let result = executor.execute_sql("SELECT fjs_emp.name, fjs_dept.dept_name FROM fjs_emp FULL OUTER JOIN fjs_dept ON fjs_emp.dept_id = fjs_dept.id", &context).unwrap();
        assert_eq!(result.result_set.rows.len(), 3);
    }

    #[test]
    fn test_cross_join_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE cjs_a (id INTEGER PRIMARY KEY, x INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "CREATE TABLE cjs_b (id INTEGER PRIMARY KEY, y TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO cjs_a (id, x) VALUES (1, 10)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cjs_a (id, x) VALUES (2, 20)", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cjs_b (id, y) VALUES (1, 'a')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cjs_b (id, y) VALUES (2, 'b')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO cjs_b (id, y) VALUES (3, 'c')", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT cjs_a.x, cjs_b.y FROM cjs_a CROSS JOIN cjs_b",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 6); // 2 x 3
    }

    // ── Step 2: COUNT(DISTINCT) + STRING_AGG tests ──

    #[test]
    fn test_count_distinct_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE cds_items (id INTEGER PRIMARY KEY, category TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cds_items (id, category) VALUES (1, 'A')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cds_items (id, category) VALUES (2, 'B')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cds_items (id, category) VALUES (3, 'A')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cds_items (id, category) VALUES (4, 'C')",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT COUNT(DISTINCT category) AS cnt FROM cds_items",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(3));
    }

    #[test]
    fn test_count_distinct_group_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE cdg_sales (id INTEGER PRIMARY KEY, region TEXT, product TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cdg_sales (id, region, product) VALUES (1, 'East', 'Widget')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cdg_sales (id, region, product) VALUES (2, 'East', 'Gadget')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cdg_sales (id, region, product) VALUES (3, 'East', 'Widget')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO cdg_sales (id, region, product) VALUES (4, 'West', 'Widget')",
                &context,
            )
            .unwrap();

        let result = executor.execute_sql("SELECT region, COUNT(DISTINCT product) AS uniq FROM cdg_sales GROUP BY region ORDER BY region", &context).unwrap();
        assert_eq!(result.result_set.rows.len(), 2);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("East".to_string())
        );
        assert_eq!(result.result_set.rows[0].values[1], Value::Int(2));
        assert_eq!(
            result.result_set.rows[1].values[0],
            Value::String("West".to_string())
        );
        assert_eq!(result.result_set.rows[1].values[1], Value::Int(1));
    }

    #[test]
    fn test_string_agg_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE sas_tags (id INTEGER PRIMARY KEY, tag TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sas_tags (id, tag) VALUES (1, 'alpha')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sas_tags (id, tag) VALUES (2, 'beta')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sas_tags (id, tag) VALUES (3, 'gamma')",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT STRING_AGG(tag, '-') AS combined FROM sas_tags",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        if let Value::String(s) = &result.result_set.rows[0].values[0] {
            assert!(s.contains("alpha"));
            assert!(s.contains("beta"));
            assert!(s.contains("gamma"));
            assert!(s.contains("-"));
        } else {
            panic!(
                "Expected string, got {:?}",
                result.result_set.rows[0].values[0]
            );
        }
    }

    #[test]
    fn test_group_concat_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE gcs_items (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO gcs_items (id, name) VALUES (1, 'x')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO gcs_items (id, name) VALUES (2, 'y')", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT GROUP_CONCAT(name) AS names FROM gcs_items",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        if let Value::String(s) = &result.result_set.rows[0].values[0] {
            assert!(s.contains("x"));
            assert!(s.contains("y"));
        } else {
            panic!(
                "Expected string, got {:?}",
                result.result_set.rows[0].values[0]
            );
        }
    }

    // ── Step 3: TRUNCATE TABLE + CTAS tests ──

    #[test]
    fn test_truncate_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE trs_data (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO trs_data (id, name) VALUES (1, 'alice')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO trs_data (id, name) VALUES (2, 'bob')",
                &context,
            )
            .unwrap();

        // Truncate
        executor
            .execute_sql("TRUNCATE TABLE trs_data", &context)
            .unwrap();

        // Verify empty — SELECT * should return no rows
        let result = executor
            .execute_sql("SELECT * FROM trs_data", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 0);
    }

    #[test]
    fn test_truncate_preserves_schema() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE trp_data (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO trp_data (id, name) VALUES (1, 'test')",
                &context,
            )
            .unwrap();
        executor.execute_sql("TRUNCATE trp_data", &context).unwrap();

        // Should still be able to insert (schema preserved)
        executor
            .execute_sql(
                "INSERT INTO trp_data (id, name) VALUES (2, 'after')",
                &context,
            )
            .unwrap();
        let result = executor
            .execute_sql("SELECT name FROM trp_data", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("after".to_string())
        );
    }

    #[test]
    fn test_upsert_do_nothing_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE udn_s (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO udn_s (id, name) VALUES (1, 'alice')", &context)
            .unwrap();

        // Conflict with DO NOTHING — row should remain unchanged
        executor
            .execute_sql(
                "INSERT INTO udn_s (id, name) VALUES (1, 'bob') ON CONFLICT (id) DO NOTHING",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql("SELECT name FROM udn_s WHERE id = 1", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("alice".to_string())
        );
    }

    #[test]
    fn test_upsert_do_update_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE udu_s (id INTEGER PRIMARY KEY, name TEXT, score INTEGER)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO udu_s (id, name, score) VALUES (1, 'alice', 10)",
                &context,
            )
            .unwrap();

        // Conflict with DO UPDATE — update name and score using excluded
        executor.execute_sql(
            "INSERT INTO udu_s (id, name, score) VALUES (1, 'bob', 20) ON CONFLICT (id) DO UPDATE SET name = excluded.name, score = excluded.score",
            &context,
        ).unwrap();

        let result = executor
            .execute_sql("SELECT name, score FROM udu_s WHERE id = 1", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("bob".to_string())
        );
        assert_eq!(result.result_set.rows[0].values[1], Value::Int(20));
    }

    // ---- Phase 10: Trig/Math + String functions ----

    #[test]
    fn test_trig_functions_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE tf_s (id INTEGER PRIMARY KEY, val REAL)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO tf_s (id, val) VALUES (1, 0.0)", &context)
            .unwrap();

        let result = executor
            .execute_sql(
                "SELECT SIN(val), COS(val), TAN(val) FROM tf_s WHERE id = 1",
                &context,
            )
            .unwrap();
        assert_eq!(result.result_set.rows[0].values[0], Value::Float(0.0));
        assert_eq!(result.result_set.rows[0].values[1], Value::Float(1.0));
        assert_eq!(result.result_set.rows[0].values[2], Value::Float(0.0));

        // PI, CBRT, TRUNC
        let result2 = executor
            .execute_sql(
                "SELECT PI(), CBRT(27), TRUNC(3.14159, 2) FROM tf_s WHERE id = 1",
                &context,
            )
            .unwrap();
        if let Value::Float(pi) = &result2.result_set.rows[0].values[0] {
            assert!((*pi - std::f64::consts::PI).abs() < 1e-10);
        } else {
            panic!("Expected float for PI()");
        }
        if let Value::Float(cbrt) = &result2.result_set.rows[0].values[1] {
            assert!((*cbrt - 3.0).abs() < 1e-10);
        } else {
            panic!("Expected float for CBRT()");
        }
        if let Value::Float(trunc) = &result2.result_set.rows[0].values[2] {
            assert!((*trunc - 3.14).abs() < 1e-10);
        } else {
            panic!("Expected float for TRUNC()");
        }
    }

    #[test]
    fn test_string_functions_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE sf_s (id INTEGER PRIMARY KEY, name TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sf_s (id, name) VALUES (1, 'hello world')",
                &context,
            )
            .unwrap();

        // INITCAP
        let result = executor
            .execute_sql("SELECT INITCAP(name) FROM sf_s WHERE id = 1", &context)
            .unwrap();
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("Hello World".to_string())
        );

        // ASCII, CHR
        let result2 = executor
            .execute_sql(
                "SELECT ASCII('A'), CHR(65) FROM sf_s WHERE id = 1",
                &context,
            )
            .unwrap();
        assert_eq!(result2.result_set.rows[0].values[0], Value::Int(65));
        assert_eq!(
            result2.result_set.rows[0].values[1],
            Value::String("A".to_string())
        );

        // CHAR_LENGTH
        let result3 = executor
            .execute_sql("SELECT CHAR_LENGTH(name) FROM sf_s WHERE id = 1", &context)
            .unwrap();
        assert_eq!(result3.result_set.rows[0].values[0], Value::Int(11));

        // CONCAT_WS
        let result4 = executor
            .execute_sql(
                "SELECT CONCAT_WS('-', 'a', 'b', 'c') FROM sf_s WHERE id = 1",
                &context,
            )
            .unwrap();
        assert_eq!(
            result4.result_set.rows[0].values[0],
            Value::String("a-b-c".to_string())
        );
    }

    #[test]
    fn test_date_functions_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        // Create table with timestamp column
        executor
            .execute_sql(
                "CREATE TABLE dt_s (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO dt_s (id, ts) VALUES (1, 1710513045)", &context)
            .unwrap(); // 2024-03-15 14:30:45

        // EXTRACT
        let r1 = executor.execute_sql("SELECT EXTRACT(YEAR FROM ts), EXTRACT(MONTH FROM ts), EXTRACT(DAY FROM ts) FROM dt_s WHERE id = 1", &context).unwrap();
        assert_eq!(r1.result_set.rows[0].values[0], Value::Int(2024));
        assert_eq!(r1.result_set.rows[0].values[1], Value::Int(3));
        assert_eq!(r1.result_set.rows[0].values[2], Value::Int(15));

        // EXTRACT time parts + DOW/DOY
        let r2 = executor.execute_sql("SELECT EXTRACT(HOUR FROM ts), EXTRACT(MINUTE FROM ts), EXTRACT(DOW FROM ts) FROM dt_s WHERE id = 1", &context).unwrap();
        assert_eq!(r2.result_set.rows[0].values[0], Value::Int(14));
        assert_eq!(r2.result_set.rows[0].values[1], Value::Int(30));
        assert_eq!(r2.result_set.rows[0].values[2], Value::Int(5)); // Friday

        // DATE_PART (alias for EXTRACT)
        let r3 = executor
            .execute_sql(
                "SELECT DATE_PART('YEAR', ts) FROM dt_s WHERE id = 1",
                &context,
            )
            .unwrap();
        assert_eq!(r3.result_set.rows[0].values[0], Value::Int(2024));

        // TO_CHAR
        let r4 = executor
            .execute_sql(
                "SELECT TO_CHAR(ts, 'YYYY-MM-DD') FROM dt_s WHERE id = 1",
                &context,
            )
            .unwrap();
        assert_eq!(
            r4.result_set.rows[0].values[0],
            Value::String("2024-03-15".to_string())
        );
    }

    #[test]
    fn test_date_trunc_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();

        executor
            .execute_sql(
                "CREATE TABLE dtt_s (id INTEGER PRIMARY KEY, ts TIMESTAMP)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO dtt_s (id, ts) VALUES (1, 1710513045)",
                &context,
            )
            .unwrap(); // 2024-03-15 14:30:45

        // DATE_TRUNC('YEAR') -> 2024-01-01 = 1704067200
        let r1 = executor
            .execute_sql(
                "SELECT DATE_TRUNC('YEAR', ts) FROM dtt_s WHERE id = 1",
                &context,
            )
            .unwrap();
        assert_eq!(
            r1.result_set.rows[0].values[0],
            Value::Timestamp(1704067200)
        );

        // DATE_TRUNC('DAY') should zero out time
        let r2 = executor
            .execute_sql(
                "SELECT DATE_TRUNC('DAY', ts) FROM dtt_s WHERE id = 1",
                &context,
            )
            .unwrap();
        if let Value::Timestamp(ts) = &r2.result_set.rows[0].values[0] {
            assert_eq!(ts % 86400, 0);
        } else {
            panic!("Expected Timestamp");
        }
    }

    // ========================================================================
    // SELECT DISTINCT tests
    // ========================================================================

    #[test]
    fn test_select_distinct_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        executor
            .execute_sql(
                "CREATE TABLE sd_test (id INTEGER PRIMARY KEY, color TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sd_test (id, color) VALUES (1, 'red')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sd_test (id, color) VALUES (2, 'blue')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sd_test (id, color) VALUES (3, 'red')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO sd_test (id, color) VALUES (4, 'blue')",
                &context,
            )
            .unwrap();

        let result = executor
            .execute_sql("SELECT DISTINCT color FROM sd_test", &context)
            .unwrap();
        assert_eq!(
            result.result_set.rows.len(),
            2,
            "DISTINCT should return 2 unique colors"
        );
    }

    #[test]
    fn test_select_distinct_with_order_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        executor
            .execute_sql(
                "CREATE TABLE sdo_test (id INTEGER PRIMARY KEY, val TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql("INSERT INTO sdo_test (id, val) VALUES (1, 'c')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO sdo_test (id, val) VALUES (2, 'a')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO sdo_test (id, val) VALUES (3, 'b')", &context)
            .unwrap();
        executor
            .execute_sql("INSERT INTO sdo_test (id, val) VALUES (4, 'a')", &context)
            .unwrap();

        let result = executor
            .execute_sql("SELECT DISTINCT val FROM sdo_test ORDER BY val", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 3);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("a".to_string())
        );
        assert_eq!(
            result.result_set.rows[1].values[0],
            Value::String("b".to_string())
        );
        assert_eq!(
            result.result_set.rows[2].values[0],
            Value::String("c".to_string())
        );
    }

    // ========================================================================
    // SELECT without FROM tests
    // ========================================================================

    #[test]
    fn test_select_expression_no_from() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let result = executor
            .execute_sql("SELECT 1 + 1 AS result", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(2));
    }

    #[test]
    fn test_select_function_no_from() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let result = executor
            .execute_sql("SELECT UPPER('hello') AS upper_val", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(
            result.result_set.rows[0].values[0],
            Value::String("HELLO".to_string())
        );
    }

    #[test]
    fn test_select_multiple_no_from() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let result = executor
            .execute_sql("SELECT 1 AS a, 2 AS b, 3 AS c", &context)
            .unwrap();
        assert_eq!(result.result_set.rows.len(), 1);
        assert_eq!(result.result_set.columns, vec!["a", "b", "c"]);
        assert_eq!(result.result_set.rows[0].values[0], Value::Int(1));
        assert_eq!(result.result_set.rows[0].values[1], Value::Int(2));
        assert_eq!(result.result_set.rows[0].values[2], Value::Int(3));
    }

    // ========================================================================
    // Full-text search function tests
    // ========================================================================

    #[test]
    fn test_fts_to_tsvector() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let r = executor
            .execute_sql("SELECT TO_TSVECTOR('The Quick Brown Fox')", &context)
            .unwrap();
        assert_eq!(r.result_set.rows.len(), 1);
        if let Value::String(s) = &r.result_set.rows[0].values[0] {
            assert!(s.contains("quick"), "Expected 'quick' in: {}", s);
            assert!(s.contains("brown"), "Expected 'brown' in: {}", s);
            assert!(s.contains("fox"), "Expected 'fox' in: {}", s);
        } else {
            panic!("Expected String result");
        }
    }

    #[test]
    fn test_fts_to_tsquery() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let r = executor
            .execute_sql("SELECT TO_TSQUERY('hello AND world')", &context)
            .unwrap();
        assert_eq!(r.result_set.rows.len(), 1);
        if let Value::String(s) = &r.result_set.rows[0].values[0] {
            assert!(s.contains("hello"), "Expected 'hello' in: {}", s);
            assert!(s.contains("world"), "Expected 'world' in: {}", s);
            assert!(s.contains("&"), "Expected '&' in: {}", s);
            assert!(
                !s.contains("and"),
                "Should not contain 'and' stopword: {}",
                s
            );
        } else {
            panic!("Expected String result");
        }
    }

    #[test]
    fn test_fts_ts_headline_storage_exec() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        executor
            .execute_sql(
                "CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO docs (id, content) VALUES (1, 'Rust is a fast systems language')",
                &context,
            )
            .unwrap();

        let r = executor
            .execute_sql("SELECT TS_HEADLINE(content, 'rust') FROM docs", &context)
            .unwrap();
        if let Value::String(s) = &r.result_set.rows[0].values[0] {
            assert!(s.contains("<b>"), "Expected bold tags in headline: {}", s);
        } else {
            panic!("Expected String result");
        }
    }

    #[test]
    fn test_fts_match_against_storage_exec() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        executor
            .execute_sql(
                "CREATE TABLE docs (id INTEGER PRIMARY KEY, body TEXT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO docs (id, body) VALUES (1, 'rust programming language')",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO docs (id, body) VALUES (2, 'python scripting')",
                &context,
            )
            .unwrap();

        let r = executor.execute_sql("SELECT id, MATCH(body) AGAINST ('rust') AS score FROM docs WHERE MATCH(body) AGAINST ('rust') > 0", &context).unwrap();
        assert_eq!(r.result_set.rows.len(), 1);
        assert_eq!(r.result_set.rows[0].values[0], Value::Int(1));
    }

    // ==================== RECURSIVE CTE TESTS ====================

    #[test]
    fn test_recursive_cte_sequence_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let r = executor.execute_sql(
            "WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 5) SELECT n FROM nums",
            &context,
        ).unwrap();
        assert_eq!(r.result_set.rows.len(), 5);
        let values: Vec<i64> = r
            .result_set
            .rows
            .iter()
            .map(|row| match &row.values[0] {
                Value::Int(n) => *n,
                _ => panic!("Expected Int"),
            })
            .collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_recursive_cte_hierarchy_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        executor
            .execute_sql(
                "CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, mgr INT)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO emp (id, name, mgr) VALUES (1, 'CEO', 0)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO emp (id, name, mgr) VALUES (2, 'VP', 1)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO emp (id, name, mgr) VALUES (3, 'Dir', 2)",
                &context,
            )
            .unwrap();
        executor
            .execute_sql(
                "INSERT INTO emp (id, name, mgr) VALUES (4, 'Mgr', 3)",
                &context,
            )
            .unwrap();

        let r = executor
            .execute_sql(
                "WITH RECURSIVE sub(id, name, mgr) AS (\
               SELECT id, name, mgr FROM emp WHERE id = 2 \
               UNION ALL \
               SELECT e.id, e.name, e.mgr FROM emp e INNER JOIN sub s ON e.mgr = s.id\
             ) SELECT name FROM sub",
                &context,
            )
            .unwrap();
        // VP, Dir, Mgr = 3 subordinates
        assert_eq!(r.result_set.rows.len(), 3);
    }

    #[test]
    fn test_recursive_cte_empty_base_storage() {
        let mut executor = create_test_executor();
        let context = QueryContext::default();
        let r = executor.execute_sql(
            "WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM nums WHERE n < 0) SELECT n FROM nums",
            &context,
        ).unwrap();
        assert_eq!(r.result_set.rows.len(), 1);
    }
}
