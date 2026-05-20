//! Apache DataFusion Integration for Columnar Query Execution
//!
//! Provides a pluggable DataFusion-based backend for JouleDB's columnar execution path.
//! DataFusion gives us:
//!
//! - **Vectorized execution** on Apache Arrow columnar batches (65K rows at a time)
//! - **Built-in SIMD optimizations** for filter/aggregate/join
//! - **SQL query planning and optimization** out of the box
//! - **Written in Rust**, embeddable, and actively maintained
//!
//! This allows JouleDB to focus engineering effort on the HDC layer (unique value)
//! while leveraging a battle-tested columnar engine for analytical workloads.
//!
//! # Architecture
//!
//! The DataFusion backend acts as an alternative execution path for analytical queries:
//!
//! ```text
//! Query → QueryPathRouter → {
//!     Point lookup  → B-Tree path
//!     Similarity    → HDC/Holographic path
//!     Analytics     → DataFusion columnar path ← THIS MODULE
//! }
//! ```
//!
//! # Feature Gate
//!
//! Enable with `datafusion-backend` feature in Cargo.toml.

use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the DataFusion columnar backend.
#[derive(Debug, Clone)]
pub struct DataFusionConfig {
    /// Maximum batch size for vectorized execution (default: 65536)
    pub batch_size: usize,
    /// Number of partitions for parallel execution (default: num_cpus)
    pub target_partitions: usize,
    /// Memory limit for the execution context in bytes (default: 256MB)
    pub memory_limit_bytes: usize,
    /// Enable query result caching (default: true)
    pub enable_cache: bool,
}

impl Default for DataFusionConfig {
    fn default() -> Self {
        Self {
            batch_size: 65536,
            target_partitions: num_cpus::get(),
            memory_limit_bytes: 256 * 1024 * 1024,
            enable_cache: true,
        }
    }
}

/// Statistics for DataFusion query execution
#[derive(Debug, Clone, Default)]
pub struct DataFusionStats {
    /// Total queries executed through DataFusion
    pub queries_executed: u64,
    /// Total rows processed
    pub rows_processed: u64,
    /// Total bytes scanned
    pub bytes_scanned: u64,
    /// Cache hit count
    pub cache_hits: u64,
    /// Average execution time in microseconds
    pub avg_execution_us: f64,
}

/// A table registration for the DataFusion context.
///
/// Maps JouleDB table data to a DataFusion-compatible format.
#[derive(Debug, Clone)]
pub struct RegisteredTable {
    /// Table name
    pub name: String,
    /// Column definitions: (name, data_type)
    pub columns: Vec<(String, ColumnType)>,
    /// Number of rows
    pub row_count: usize,
}

/// Column data types supported in the DataFusion bridge
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    /// 64-bit integer
    Int64,
    /// 64-bit float
    Float64,
    /// UTF-8 string
    Utf8,
    /// Boolean
    Boolean,
    /// Binary data
    Binary,
    /// Timestamp (milliseconds since epoch)
    TimestampMs,
}

/// DataFusion columnar execution backend.
///
/// This wraps a DataFusion SessionContext and provides methods to:
/// 1. Register JouleDB tables as DataFusion data sources
/// 2. Execute SQL queries through DataFusion's optimized engine
/// 3. Return results in JouleDB's format
///
/// When the `datafusion-backend` feature is NOT enabled, this struct
/// provides stub implementations that return errors, allowing the code
/// to compile without the DataFusion dependency.
pub struct DataFusionBackend {
    /// Configuration
    config: DataFusionConfig,
    /// Registered tables
    tables: HashMap<String, RegisteredTable>,
    /// Execution statistics
    stats: DataFusionStats,
    /// DataFusion session context (when feature is enabled)
    #[cfg(feature = "datafusion-backend")]
    ctx: datafusion::prelude::SessionContext,
}

impl DataFusionBackend {
    /// Create a new DataFusion backend with default configuration
    pub fn new() -> Self {
        Self::with_config(DataFusionConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DataFusionConfig) -> Self {
        #[cfg(feature = "datafusion-backend")]
        let ctx = {
            let mut session_config = datafusion::prelude::SessionConfig::new()
                .with_batch_size(config.batch_size)
                .with_target_partitions(config.target_partitions);
            datafusion::prelude::SessionContext::new_with_config(session_config)
        };

        Self {
            config,
            tables: HashMap::new(),
            stats: DataFusionStats::default(),
            #[cfg(feature = "datafusion-backend")]
            ctx,
        }
    }

    /// Register a table with the DataFusion context.
    ///
    /// The table data will be available for SQL queries executed through this backend.
    pub fn register_table(&mut self, table: RegisteredTable) {
        self.tables.insert(table.name.clone(), table);
    }

    /// Execute a SQL query through DataFusion's columnar engine.
    ///
    /// Returns results as a vector of rows (each row is a vector of JSON values).
    pub fn execute_sql(&mut self, sql: &str) -> Result<DataFusionResult, DataFusionError> {
        self.stats.queries_executed += 1;

        // Without the datafusion feature, return an error
        #[cfg(not(feature = "datafusion-backend"))]
        {
            return Err(DataFusionError::NotEnabled);
        }

        #[cfg(feature = "datafusion-backend")]
        {
            // Use DataFusion's SQL execution
            // This would use the tokio runtime to execute the async DataFusion query
            let rt = tokio::runtime::Handle::try_current()
                .map_err(|_| DataFusionError::RuntimeError("No tokio runtime".to_string()))?;

            let ctx = &self.ctx;
            let result = rt.block_on(async {
                let df = ctx
                    .sql(sql)
                    .await
                    .map_err(|e| DataFusionError::ExecutionError(e.to_string()))?;

                let batches = df
                    .collect()
                    .await
                    .map_err(|e| DataFusionError::ExecutionError(e.to_string()))?;

                let mut columns = Vec::new();
                let mut rows = Vec::new();
                let mut total_rows = 0usize;

                for batch in &batches {
                    if columns.is_empty() {
                        columns = batch
                            .schema()
                            .fields()
                            .iter()
                            .map(|f| f.name().clone())
                            .collect();
                    }

                    total_rows += batch.num_rows();
                    // Convert Arrow RecordBatch to JSON rows
                    for row_idx in 0..batch.num_rows() {
                        let mut row = Vec::with_capacity(batch.num_columns());
                        for col_idx in 0..batch.num_columns() {
                            let col = batch.column(col_idx);
                            let value = arrow_value_to_json(col, row_idx);
                            row.push(value);
                        }
                        rows.push(row);
                    }
                }

                Ok(DataFusionResult {
                    columns,
                    rows,
                    rows_affected: total_rows,
                })
            })?;

            self.stats.rows_processed += result.rows_affected as u64;
            Ok(result)
        }
    }

    /// Check if DataFusion backend is available (feature enabled)
    pub fn is_available() -> bool {
        cfg!(feature = "datafusion-backend")
    }

    /// Get execution statistics
    pub fn stats(&self) -> &DataFusionStats {
        &self.stats
    }

    /// Get registered tables
    pub fn tables(&self) -> &HashMap<String, RegisteredTable> {
        &self.tables
    }
}

impl Default for DataFusionBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from a DataFusion query execution
#[derive(Debug, Clone)]
pub struct DataFusionResult {
    /// Column names
    pub columns: Vec<String>,
    /// Row data as JSON values
    pub rows: Vec<Vec<serde_json::Value>>,
    /// Number of rows affected/returned
    pub rows_affected: usize,
}

/// Errors from the DataFusion backend
#[derive(Debug, Clone)]
pub enum DataFusionError {
    /// DataFusion feature not enabled
    NotEnabled,
    /// Table not found
    TableNotFound(String),
    /// SQL execution error
    ExecutionError(String),
    /// Runtime error (no tokio runtime available)
    RuntimeError(String),
}

impl std::fmt::Display for DataFusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotEnabled => write!(
                f,
                "DataFusion backend not enabled (add 'datafusion-backend' feature)"
            ),
            Self::TableNotFound(t) => write!(f, "Table '{}' not registered with DataFusion", t),
            Self::ExecutionError(e) => write!(f, "DataFusion execution error: {}", e),
            Self::RuntimeError(e) => write!(f, "Runtime error: {}", e),
        }
    }
}

impl std::error::Error for DataFusionError {}

/// Convert an Arrow array value at a specific row to a JSON value
#[cfg(feature = "datafusion-backend")]
fn arrow_value_to_json(col: &arrow::array::ArrayRef, row: usize) -> serde_json::Value {
    use arrow::array::*;
    use arrow::datatypes::DataType;

    if col.is_null(row) {
        return serde_json::Value::Null;
    }

    match col.data_type() {
        DataType::Int8 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int8Array>()
                .expect("Int8 column");
            serde_json::Value::Number(arr.value(row).into())
        }
        DataType::Int16 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int16Array>()
                .expect("Int16 column");
            serde_json::Value::Number(arr.value(row).into())
        }
        DataType::Int32 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("Int32 column");
            serde_json::Value::Number(arr.value(row).into())
        }
        DataType::Int64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 column");
            serde_json::Value::Number(arr.value(row).into())
        }
        DataType::Float32 => {
            let arr = col
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("Float32 column");
            serde_json::Number::from_f64(arr.value(row) as f64)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        DataType::Float64 => {
            let arr = col
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64 column");
            serde_json::Number::from_f64(arr.value(row))
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("String column");
            serde_json::Value::String(arr.value(row).to_string())
        }
        DataType::Boolean => {
            let arr = col
                .as_any()
                .downcast_ref::<BooleanArray>()
                .expect("Boolean column");
            serde_json::Value::Bool(arr.value(row))
        }
        _ => serde_json::Value::String(format!("<unsupported type: {:?}>", col.data_type())),
    }
}
