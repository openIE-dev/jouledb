//! AmorphicTableStorage - bridges SQL queries to the Amorphic store
//!
//! Uses a `__table__` field convention to map SQL tables onto amorphic records.
//! Schema metadata is stored as amorphic records with `__table__ = "__schemas__"`.
//!
//! This lets the query executor treat `DurableAmorphicStore` as a collection
//! of SQL tables, while the amorphic store handles HDC encoding, indexing,
//! and graph edges transparently.

mod constraints;
mod conversions;
mod crud;
mod expressions;
mod fulltext;
mod indexes;
mod schema;
#[cfg(test)]
mod tests;
mod triggers;
mod views;

use std::collections::HashMap;
use std::sync::RwLock;

use joule_db_amorphic::{AmorphicRecord, DurableAmorphicStore, RecordId, Value as AmorphicValue};
use joule_db_query::ast::{Expression, Operator, UnaryOperator, Value as AstValue};
use joule_db_query::error::{QueryError, QueryResult};
use joule_db_query::executor::{RowData, TableStorage};
use joule_db_query::planner::{
    ColumnStats as PlannerColumnStats, IndexInfo, IndexType, Statistics as PlannerStatistics,
    TableStats as PlannerTableStats,
};
use joule_db_query::sql::SqlColumnDef;

// Re-export public items that external code references
pub use conversions::{ast_to_amorphic_value, ast_to_json, expression_to_sql};

/// Reserved field name that identifies which SQL table a record belongs to
pub(super) const TABLE_FIELD: &str = "__table__";
/// Reserved table name for sequence metadata (auto-increment counters)
pub(super) const SEQUENCE_TABLE: &str = "__sequences__";
/// Field for sequence name (table.column)
pub(super) const SEQUENCE_NAME_FIELD: &str = "__sequence_name__";
/// Field for current sequence value
pub(super) const SEQUENCE_VALUE_FIELD: &str = "__sequence_value__";

/// Reserved table name for schema metadata
pub(super) const SCHEMA_TABLE: &str = "__schemas__";
/// Field in schema records that stores the table name
pub(super) const SCHEMA_NAME_FIELD: &str = "__schema_name__";
/// Field in schema records that stores column definitions (JSON array)
pub(super) const SCHEMA_COLUMNS_FIELD: &str = "__columns__";
/// Reserved table name for index metadata
pub(super) const INDEX_TABLE: &str = "__indexes__";
/// Field in schema records that stores full column definitions (JSON array of objects)
pub(super) const SCHEMA_COLUMN_DEFS_FIELD: &str = "__column_defs__";
/// Field in index records that stores the index name
pub(super) const INDEX_NAME_FIELD: &str = "__index_name__";
/// Field in index records that stores the target table
pub(super) const INDEX_TABLE_REF: &str = "__index_table__";
/// Field in index records that stores indexed columns
pub(super) const INDEX_COLUMNS_FIELD: &str = "__index_columns__";
/// Field in index records that stores unique flag
pub(super) const INDEX_UNIQUE_FIELD: &str = "__index_unique__";
/// Reserved table name for trigger metadata
pub(super) const TRIGGER_TABLE: &str = "__triggers__";
/// Field in trigger records that stores the trigger name
pub(super) const TRIGGER_NAME_FIELD: &str = "__trigger_name__";
/// Field in trigger records that stores the target table
pub(super) const TRIGGER_TABLE_REF: &str = "__trigger_table__";
/// Field in trigger records that stores the timing (BEFORE/AFTER)
pub(super) const TRIGGER_TIMING_FIELD: &str = "__trigger_timing__";
/// Field in trigger records that stores the event (INSERT/UPDATE/DELETE)
pub(super) const TRIGGER_EVENT_FIELD: &str = "__trigger_event__";
/// Field in trigger records that stores the body SQL
pub(super) const TRIGGER_BODY_FIELD: &str = "__trigger_body__";
/// Field in schema records that stores column families
pub(super) const SCHEMA_COLUMN_FAMILIES_FIELD: &str = "__column_families__";
/// Reserved table name for materialized view metadata
pub(super) const MATVIEW_TABLE: &str = "__matviews__";
/// Field in matview records that stores the view name
pub(super) const MATVIEW_NAME_FIELD: &str = "__matview_name__";
/// Field in matview records that stores the query SQL
pub(super) const MATVIEW_QUERY_FIELD: &str = "__matview_query__";
/// Field in matview records that stores the backing table name
pub(super) const MATVIEW_BACKING_TABLE_FIELD: &str = "__matview_backing_table__";
/// Reserved table name for reference metadata
pub(super) const REFERENCE_TABLE: &str = "__references__";
/// Field in reference records that stores the reference name
pub(super) const REFERENCE_NAME_FIELD: &str = "__reference_name__";
/// Field in reference records that stores the source table
pub(super) const REFERENCE_SRC_TABLE_FIELD: &str = "__reference_src_table__";
/// Field in reference records that stores the source column
pub(super) const REFERENCE_SRC_COLUMN_FIELD: &str = "__reference_src_column__";
/// Field in reference records that stores the target table
pub(super) const REFERENCE_TGT_TABLE_FIELD: &str = "__reference_tgt_table__";
/// Field in reference records that stores the target column
pub(super) const REFERENCE_TGT_COLUMN_FIELD: &str = "__reference_tgt_column__";

/// Reserved table name for bucket metadata
pub(super) const BUCKET_TABLE: &str = "__buckets__";
/// Field in bucket records that stores the bucket name
pub(super) const BUCKET_NAME_FIELD: &str = "__bucket_name__";
/// Field in bucket records that stores optional max size in bytes
pub(super) const BUCKET_MAX_SIZE_FIELD: &str = "__bucket_max_size__";

/// Reserved table name for API endpoint metadata
pub(super) const API_TABLE: &str = "__apis__";
/// Field in API records that stores the endpoint path
pub(super) const API_PATH_FIELD: &str = "__api_path__";
/// Field in API records that stores the HTTP method
pub(super) const API_METHOD_FIELD: &str = "__api_method__";
/// Field in API records that stores the handler SQL
pub(super) const API_HANDLER_FIELD: &str = "__api_handler__";

/// Default collection name for schema-less ingest
pub(super) const DEFAULT_COLLECTION: &str = "__default__";
/// Field name for specifying collection in unified ingest
pub(super) const COLLECTION_FIELD: &str = "_collection";

/// Trigger info retrieved from metadata
#[derive(Debug, Clone)]
pub struct TriggerInfo {
    pub name: String,
    pub table: String,
    pub timing: String, // "BEFORE" or "AFTER"
    pub event: String,  // "INSERT", "UPDATE", or "DELETE"
    pub body: String,   // SQL statement to execute
}

/// Reference definition info retrieved from metadata
#[derive(Debug, Clone)]
pub struct ReferenceInfo {
    pub name: String,
    pub src_table: String,
    pub src_column: String,
    pub tgt_table: String,
    pub tgt_column: String,
}

/// Bucket definition info retrieved from metadata
#[derive(Debug, Clone)]
pub struct BucketInfo {
    pub name: String,
    pub max_size: Option<u64>,
}

/// API endpoint definition info retrieved from metadata
#[derive(Debug, Clone)]
pub struct ApiEndpointInfo {
    pub path: String,
    pub method: String,
    pub handler_sql: String,
}

/// Foreign key info stored in schema metadata
#[derive(Debug, Clone)]
pub struct ForeignKeyInfo {
    pub ref_table: String,
    pub ref_column: String,
    pub on_delete: Option<String>,
    pub on_update: Option<String>,
}

/// Column definition info retrieved from schema metadata
#[derive(Debug, Clone)]
pub struct ColumnDefInfo {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    /// Default value as a JSON-serializable value (if any)
    pub default_value: Option<serde_json::Value>,
    /// CHECK constraint expression as serialized string (if any)
    pub check_expr: Option<String>,
    /// Whether this column auto-increments (SERIAL/BIGSERIAL)
    pub auto_increment: bool,
    /// Foreign key constraint (if any)
    pub foreign_key: Option<ForeignKeyInfo>,
    /// Column family assignment (for wide-column storage)
    pub column_family: Option<String>,
    /// Computed column expression as serialized SQL string (if any)
    pub computed_expr: Option<String>,
}

/// Collected statistics for a table
#[derive(Debug, Clone)]
pub struct TableStatistics {
    pub row_count: usize,
    pub columns: HashMap<String, ColumnStatistics>,
    /// (index_name, columns, unique, index_type)
    pub indexes: Vec<(String, Vec<String>, bool, String)>,
}

/// Statistics for a single column
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    pub distinct_count: usize,
    pub null_fraction: f64,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

/// Adapter that exposes a `DurableAmorphicStore` as SQL-queryable tables.
///
/// Each SQL table is represented as amorphic records sharing the same
/// `__table__` field value. Column schemas are stored as dedicated
/// `__schemas__` records for `CREATE TABLE` / `DROP TABLE` support.
pub struct AmorphicTableStorage {
    store: RwLock<DurableAmorphicStore>,
}

impl AmorphicTableStorage {
    pub fn new(store: DurableAmorphicStore) -> Self {
        Self {
            store: RwLock::new(store),
        }
    }

    /// List all user-created table names (from schema metadata).
    pub fn list_tables(&self) -> Vec<String> {
        let store = match self.store.read() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let schema_result = store.query_equals(
            TABLE_FIELD,
            &AmorphicValue::String(SCHEMA_TABLE.to_string()),
        );
        let mut names = Vec::new();
        for record in schema_result.records() {
            if let Some(AmorphicValue::String(name)) = record.get(SCHEMA_NAME_FIELD) {
                names.push(name.clone());
            }
        }
        names
    }

    /// Convert an AmorphicRecord to a RowData given column names
    pub(super) fn record_to_row(record: &AmorphicRecord, columns: &[String]) -> RowData {
        let values: Vec<AstValue> = columns
            .iter()
            .map(|col| {
                record
                    .get(col)
                    .map(conversions::amorphic_to_ast)
                    .unwrap_or(AstValue::Null)
            })
            .collect();
        RowData::new(columns.to_vec(), values)
    }

    /// Get schema columns for a table
    pub(super) fn get_schema_columns(
        &self,
        store: &DurableAmorphicStore,
        table: &str,
    ) -> QueryResult<Vec<String>> {
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        for record in result.records() {
            if record.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string())) {
                if let Some(AmorphicValue::Array(cols)) = record.get(SCHEMA_COLUMNS_FIELD) {
                    return Ok(cols
                        .iter()
                        .filter_map(|v| match v {
                            AmorphicValue::String(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect());
                }
            }
        }
        Err(QueryError::ExecutionError(format!(
            "Table '{}' does not exist",
            table
        )))
    }
}

// ============ TableStorage implementation ============

impl TableStorage for AmorphicTableStorage {
    fn scan(&self, table: &str) -> QueryResult<Vec<RowData>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let rows = result
            .records()
            .iter()
            .map(|r| Self::record_to_row(r, &columns))
            .collect();

        Ok(rows)
    }

    fn columns(&self, table: &str) -> QueryResult<Vec<String>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        self.get_schema_columns(&store, table)
    }

    fn insert(&self, table: &str, row: &RowData) -> QueryResult<()> {
        self.insert_returning_id(table, row)?;
        Ok(())
    }

    fn update(
        &self,
        table: &str,
        assignments: &HashMap<String, AstValue>,
        predicate: Option<&Expression>,
    ) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        // Find matching records
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut updated = 0;

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| expressions::AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                let updates: HashMap<String, AmorphicValue> = assignments
                    .iter()
                    .map(|(k, v)| (k.clone(), conversions::ast_to_amorphic(v)))
                    .collect();

                store
                    .update_fields(record.id, updates)
                    .map_err(conversions::amorphic_error)?;
                updated += 1;
            }
        }

        Ok(updated)
    }

    fn delete(&self, table: &str, predicate: Option<&Expression>) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        // Find matching records
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut to_delete = Vec::new();

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| expressions::AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                to_delete.push(record.id);
            }
        }

        let count = to_delete.len();
        for id in &to_delete {
            store.delete(*id).map_err(conversions::amorphic_error)?;
        }
        drop(store);

        // Clean up fulltext index entries for deleted records
        for id in &to_delete {
            let _ = self.update_fulltext_indexes_on_delete(table, *id);
        }

        Ok(count)
    }

    fn table_exists(&self, table: &str) -> QueryResult<bool> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let result =
            store.query_equals(SCHEMA_NAME_FIELD, &AmorphicValue::String(table.to_string()));
        Ok(result
            .records()
            .iter()
            .any(|r| r.get(TABLE_FIELD) == Some(&AmorphicValue::String(SCHEMA_TABLE.to_string()))))
    }

    fn index_scan(
        &self,
        table: &str,
        index: &str,
        predicate: &Expression,
    ) -> QueryResult<Vec<RowData>> {
        let store = self.store.read().map_err(conversions::lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        // Try to extract equality predicate for the indexed field
        if let Some((field, value)) = conversions::extract_eq_predicate(predicate) {
            if field == index {
                let amorphic_val = conversions::ast_to_amorphic(&value);
                let result = store.query_equals(&field, &amorphic_val);
                let rows = result
                    .records()
                    .iter()
                    .filter(|r| {
                        r.get(TABLE_FIELD) == Some(&AmorphicValue::String(table.to_string()))
                    })
                    .map(|r| Self::record_to_row(r, &columns))
                    .collect();

                return Ok(rows);
            }
        }

        // Try to extract range predicate for the indexed field
        if let Some((field, min, max)) = conversions::extract_range_predicate(predicate) {
            if field == index {
                let result = store.query_range(&field, min, max);
                let rows = result
                    .records()
                    .iter()
                    .filter(|r| {
                        r.get(TABLE_FIELD) == Some(&AmorphicValue::String(table.to_string()))
                    })
                    .map(|r| Self::record_to_row(r, &columns))
                    .collect();

                return Ok(rows);
            }
        }

        // Fallback: full scan with predicate filter
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let rows = result
            .records()
            .iter()
            .map(|r| Self::record_to_row(r, &columns))
            .filter(|row| expressions::AmorphicExprEval::evaluate_predicate(row, predicate))
            .collect();

        Ok(rows)
    }
}
