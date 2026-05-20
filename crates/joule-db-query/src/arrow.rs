//! Arrow-based Vectorized Query Execution
//!
//! This module provides columnar, vectorized query execution using Apache Arrow.
//! Arrow's columnar format enables SIMD-optimized operations for filters,
//! projections, and aggregations.
//!
//! ## Benefits
//!
//! - **CPU Cache Efficiency**: Columnar data fits better in CPU caches
//! - **SIMD Vectorization**: Arrow kernels use SIMD instructions automatically
//! - **Zero-Copy**: Arrow uses reference-counted buffers for zero-copy operations
//! - **Interoperability**: Arrow format is compatible with many data tools
//!
//! ## Example
//!
//! ```ignore
//! use joule_db_query::arrow::{ArrowExecutor, RecordBatchBuilder};
//!
//! // Build a batch of data
//! let batch = RecordBatchBuilder::new()
//!     .add_i64_column("id", vec![1, 2, 3, 4, 5])
//!     .add_string_column("name", vec!["Alice", "Bob", "Carol", "Dave", "Eve"])
//!     .add_f64_column("score", vec![95.0, 87.5, 92.0, 78.0, 88.5])
//!     .build()?;
//!
//! // Execute vectorized filter
//! let executor = ArrowExecutor::new();
//! let filtered = executor.filter(&batch, "score", FilterOp::Gt, 85.0)?;
//! ```

use crate::ast::{Operator, Value};
use crate::error::{QueryError, QueryResult};
use crate::execution::{ResultSet, Row};

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, PrimitiveBuilder,
    StringArray, StringBuilder,
};
use arrow::compute::{self, SortOptions, kernels::cmp};
use arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use arrow::record_batch::RecordBatch;

use std::sync::Arc;

/// Filter operation for vectorized filtering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl From<Operator> for FilterOp {
    fn from(op: Operator) -> Self {
        match op {
            Operator::Eq => FilterOp::Eq,
            Operator::Ne => FilterOp::Ne,
            Operator::Lt => FilterOp::Lt,
            Operator::Le => FilterOp::Le,
            Operator::Gt => FilterOp::Gt,
            Operator::Ge => FilterOp::Ge,
            _ => FilterOp::Eq, // Default for non-comparison ops
        }
    }
}

/// Aggregate function type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregateOp {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Builder for creating RecordBatches from row data
pub struct RecordBatchBuilder {
    columns: Vec<(String, ColumnData)>,
    row_count: Option<usize>,
    error: Option<QueryError>,
}

enum ColumnData {
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    String(Vec<Option<String>>),
    Boolean(Vec<Option<bool>>),
}

impl RecordBatchBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            row_count: None,
            error: None,
        }
    }

    /// Add an i64 column
    pub fn add_i64_column(mut self, name: &str, values: Vec<i64>) -> Self {
        self.validate_row_count(values.len());
        self.columns.push((
            name.to_string(),
            ColumnData::Int64(values.into_iter().map(Some).collect()),
        ));
        self
    }

    /// Add an i64 column with nulls
    pub fn add_i64_column_nullable(mut self, name: &str, values: Vec<Option<i64>>) -> Self {
        self.validate_row_count(values.len());
        self.columns
            .push((name.to_string(), ColumnData::Int64(values)));
        self
    }

    /// Add a f64 column
    pub fn add_f64_column(mut self, name: &str, values: Vec<f64>) -> Self {
        self.validate_row_count(values.len());
        self.columns.push((
            name.to_string(),
            ColumnData::Float64(values.into_iter().map(Some).collect()),
        ));
        self
    }

    /// Add a f64 column with nulls
    pub fn add_f64_column_nullable(mut self, name: &str, values: Vec<Option<f64>>) -> Self {
        self.validate_row_count(values.len());
        self.columns
            .push((name.to_string(), ColumnData::Float64(values)));
        self
    }

    /// Add a string column
    pub fn add_string_column(mut self, name: &str, values: Vec<&str>) -> Self {
        self.validate_row_count(values.len());
        self.columns.push((
            name.to_string(),
            ColumnData::String(values.into_iter().map(|s| Some(s.to_string())).collect()),
        ));
        self
    }

    /// Add a string column with nulls
    pub fn add_string_column_nullable(mut self, name: &str, values: Vec<Option<String>>) -> Self {
        self.validate_row_count(values.len());
        self.columns
            .push((name.to_string(), ColumnData::String(values)));
        self
    }

    /// Add a boolean column
    pub fn add_bool_column(mut self, name: &str, values: Vec<bool>) -> Self {
        self.validate_row_count(values.len());
        self.columns.push((
            name.to_string(),
            ColumnData::Boolean(values.into_iter().map(Some).collect()),
        ));
        self
    }

    fn validate_row_count(&mut self, count: usize) {
        if let Some(existing) = self.row_count {
            if existing != count && self.error.is_none() {
                self.error = Some(QueryError::ParseError(format!(
                    "Column row count mismatch: expected {}, got {}",
                    existing, count
                )));
            }
        } else {
            self.row_count = Some(count);
        }
    }

    /// Build the RecordBatch
    pub fn build(self) -> QueryResult<RecordBatch> {
        if let Some(err) = self.error {
            return Err(err);
        }
        if self.columns.is_empty() {
            return Err(QueryError::ParseError("No columns in batch".to_string()));
        }

        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();

        for (name, data) in self.columns {
            match data {
                ColumnData::Int64(values) => {
                    fields.push(Field::new(&name, DataType::Int64, true));
                    let array: Int64Array = values.into_iter().collect();
                    arrays.push(Arc::new(array));
                }
                ColumnData::Float64(values) => {
                    fields.push(Field::new(&name, DataType::Float64, true));
                    let array: Float64Array = values.into_iter().collect();
                    arrays.push(Arc::new(array));
                }
                ColumnData::String(values) => {
                    fields.push(Field::new(&name, DataType::Utf8, true));
                    let array: StringArray = values.iter().map(|s| s.as_deref()).collect();
                    arrays.push(Arc::new(array));
                }
                ColumnData::Boolean(values) => {
                    fields.push(Field::new(&name, DataType::Boolean, true));
                    let array: BooleanArray = values.into_iter().collect();
                    arrays.push(Arc::new(array));
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays).map_err(|e| QueryError::ExecutionError(e.to_string()))
    }
}

impl Default for RecordBatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Arrow-based vectorized query executor
pub struct ArrowExecutor {
    /// Batch size for processing (default 8192)
    batch_size: usize,
}

impl ArrowExecutor {
    /// Create a new executor with default settings
    pub fn new() -> Self {
        Self { batch_size: 8192 }
    }

    /// Create executor with custom batch size
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Vectorized filter operation on a column with i64 value
    pub fn filter_i64(
        &self,
        batch: &RecordBatch,
        column: &str,
        op: FilterOp,
        value: i64,
    ) -> QueryResult<RecordBatch> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let int_array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
            QueryError::TypeMismatch {
                expected: "Int64".to_string(),
                found: format!("{:?}", array.data_type()),
            }
        })?;

        // Create scalar array for comparison
        let scalar = Int64Array::new_scalar(value);

        let mask = match op {
            FilterOp::Eq => cmp::eq(int_array, &scalar)?,
            FilterOp::Ne => cmp::neq(int_array, &scalar)?,
            FilterOp::Lt => cmp::lt(int_array, &scalar)?,
            FilterOp::Le => cmp::lt_eq(int_array, &scalar)?,
            FilterOp::Gt => cmp::gt(int_array, &scalar)?,
            FilterOp::Ge => cmp::gt_eq(int_array, &scalar)?,
        };

        self.apply_filter(batch, &mask)
    }

    /// Vectorized filter operation on a column with f64 value
    pub fn filter_f64(
        &self,
        batch: &RecordBatch,
        column: &str,
        op: FilterOp,
        value: f64,
    ) -> QueryResult<RecordBatch> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let float_array = array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| QueryError::TypeMismatch {
                expected: "Float64".to_string(),
                found: format!("{:?}", array.data_type()),
            })?;

        let scalar = Float64Array::new_scalar(value);

        let mask = match op {
            FilterOp::Eq => cmp::eq(float_array, &scalar)?,
            FilterOp::Ne => cmp::neq(float_array, &scalar)?,
            FilterOp::Lt => cmp::lt(float_array, &scalar)?,
            FilterOp::Le => cmp::lt_eq(float_array, &scalar)?,
            FilterOp::Gt => cmp::gt(float_array, &scalar)?,
            FilterOp::Ge => cmp::gt_eq(float_array, &scalar)?,
        };

        self.apply_filter(batch, &mask)
    }

    /// Vectorized filter on string column
    pub fn filter_string(
        &self,
        batch: &RecordBatch,
        column: &str,
        op: FilterOp,
        value: &str,
    ) -> QueryResult<RecordBatch> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let str_array = array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| QueryError::TypeMismatch {
                expected: "String".to_string(),
                found: format!("{:?}", array.data_type()),
            })?;

        let scalar = StringArray::new_scalar(value);

        let mask = match op {
            FilterOp::Eq => cmp::eq(str_array, &scalar)?,
            FilterOp::Ne => cmp::neq(str_array, &scalar)?,
            FilterOp::Lt => cmp::lt(str_array, &scalar)?,
            FilterOp::Le => cmp::lt_eq(str_array, &scalar)?,
            FilterOp::Gt => cmp::gt(str_array, &scalar)?,
            FilterOp::Ge => cmp::gt_eq(str_array, &scalar)?,
        };

        self.apply_filter(batch, &mask)
    }

    /// Apply a boolean mask to filter a batch
    fn apply_filter(&self, batch: &RecordBatch, mask: &BooleanArray) -> QueryResult<RecordBatch> {
        compute::filter_record_batch(batch, mask)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))
    }

    /// Vectorized projection - select specific columns
    pub fn project(&self, batch: &RecordBatch, columns: &[&str]) -> QueryResult<RecordBatch> {
        let indices: Vec<usize> = columns
            .iter()
            .map(|c| self.get_column_index(batch, c))
            .collect::<QueryResult<Vec<_>>>()?;

        let projected_columns: Vec<ArrayRef> =
            indices.iter().map(|&i| batch.column(i).clone()).collect();

        let projected_fields: Vec<Field> = indices
            .iter()
            .map(|&i| batch.schema().field(i).clone())
            .collect();

        let schema = Arc::new(Schema::new(projected_fields));
        RecordBatch::try_new(schema, projected_columns)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))
    }

    /// Vectorized sort
    pub fn sort(
        &self,
        batch: &RecordBatch,
        column: &str,
        descending: bool,
    ) -> QueryResult<RecordBatch> {
        let col_idx = self.get_column_index(batch, column)?;

        let options = SortOptions {
            descending,
            nulls_first: false,
        };

        let indices = compute::sort_to_indices(batch.column(col_idx).as_ref(), Some(options), None)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        let sorted_columns: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|col| compute::take(col.as_ref(), &indices, None))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| QueryError::ExecutionError(e.to_string()))?;

        RecordBatch::try_new(batch.schema().clone(), sorted_columns)
            .map_err(|e| QueryError::ExecutionError(e.to_string()))
    }

    /// Vectorized limit
    pub fn limit(&self, batch: &RecordBatch, n: usize) -> QueryResult<RecordBatch> {
        if n >= batch.num_rows() {
            return Ok(batch.clone());
        }
        Ok(batch.slice(0, n))
    }

    /// Vectorized offset
    pub fn offset(&self, batch: &RecordBatch, n: usize) -> QueryResult<RecordBatch> {
        if n >= batch.num_rows() {
            return self.empty_batch(batch.schema());
        }
        Ok(batch.slice(n, batch.num_rows() - n))
    }

    /// Create empty batch with same schema
    fn empty_batch(&self, schema: Arc<Schema>) -> QueryResult<RecordBatch> {
        Ok(RecordBatch::new_empty(schema))
    }

    /// Vectorized aggregation - count
    pub fn count(&self, batch: &RecordBatch) -> usize {
        batch.num_rows()
    }

    /// Vectorized aggregation - count non-null in column
    pub fn count_column(&self, batch: &RecordBatch, column: &str) -> QueryResult<usize> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);
        Ok(array.len() - array.null_count())
    }

    /// Vectorized aggregation - sum of i64 column
    pub fn sum_i64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<i64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let int_array = array.as_primitive::<Int64Type>();
        Ok(compute::sum(int_array))
    }

    /// Vectorized aggregation - sum of f64 column
    pub fn sum_f64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<f64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let float_array = array.as_primitive::<Float64Type>();
        Ok(compute::sum(float_array))
    }

    /// Vectorized aggregation - average of f64 column
    pub fn avg_f64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<f64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let float_array = array.as_primitive::<Float64Type>();
        let sum = compute::sum(float_array);
        let count = float_array.len() - float_array.null_count();

        Ok(sum.map(|s| s / count as f64))
    }

    /// Vectorized aggregation - min of i64 column
    pub fn min_i64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<i64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let int_array = array.as_primitive::<Int64Type>();
        Ok(compute::min(int_array))
    }

    /// Vectorized aggregation - max of i64 column
    pub fn max_i64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<i64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let int_array = array.as_primitive::<Int64Type>();
        Ok(compute::max(int_array))
    }

    /// Vectorized aggregation - min of f64 column
    pub fn min_f64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<f64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let float_array = array.as_primitive::<Float64Type>();
        Ok(compute::min(float_array))
    }

    /// Vectorized aggregation - max of f64 column
    pub fn max_f64(&self, batch: &RecordBatch, column: &str) -> QueryResult<Option<f64>> {
        let col_idx = self.get_column_index(batch, column)?;
        let array = batch.column(col_idx);

        let float_array = array.as_primitive::<Float64Type>();
        Ok(compute::max(float_array))
    }

    /// Convert RecordBatch to ResultSet (for compatibility with existing code)
    pub fn to_result_set(&self, batch: &RecordBatch) -> QueryResult<ResultSet> {
        let columns: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        let mut rows = Vec::with_capacity(batch.num_rows());

        for row_idx in 0..batch.num_rows() {
            let mut values = Vec::with_capacity(batch.num_columns());

            for col in batch.columns() {
                let value = self.array_value_to_value(col.as_ref(), row_idx)?;
                values.push(value);
            }

            rows.push(Row::new(values));
        }

        Ok(ResultSet {
            columns,
            rows,
            affected_rows: 0,
            execution_time_ms: 0,
            truncated: false,
        })
    }

    /// Convert ResultSet to RecordBatch (for vectorized processing)
    pub fn from_result_set(&self, result: &ResultSet) -> QueryResult<RecordBatch> {
        if result.columns.is_empty() || result.rows.is_empty() {
            let schema = Arc::new(Schema::new(
                result
                    .columns
                    .iter()
                    .map(|c| Field::new(c, DataType::Utf8, true))
                    .collect::<Vec<_>>(),
            ));
            return Ok(RecordBatch::new_empty(schema));
        }

        // Infer column types from first row
        let first_row = &result.rows[0];
        let mut builders: Vec<ColumnBuilder> = first_row
            .values
            .iter()
            .map(|v| ColumnBuilder::from_value(v, result.rows.len()))
            .collect();

        // Add all rows
        for row in &result.rows {
            for (i, value) in row.values.iter().enumerate() {
                if i < builders.len() {
                    builders[i].append(value);
                }
            }
        }

        // Build arrays
        let arrays: Vec<ArrayRef> = builders.into_iter().map(|b| b.finish()).collect();

        let fields: Vec<Field> = result
            .columns
            .iter()
            .zip(arrays.iter())
            .map(|(name, arr)| Field::new(name, arr.data_type().clone(), true))
            .collect();

        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, arrays).map_err(|e| QueryError::ExecutionError(e.to_string()))
    }

    fn get_column_index(&self, batch: &RecordBatch, column: &str) -> QueryResult<usize> {
        batch
            .schema()
            .index_of(column)
            .map_err(|_| QueryError::UnknownColumn(column.to_string()))
    }

    fn array_value_to_value(&self, array: &dyn Array, idx: usize) -> QueryResult<Value> {
        if array.is_null(idx) {
            return Ok(Value::Null);
        }

        match array.data_type() {
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64 array");
                Ok(Value::Int(arr.value(idx)))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("Float64 array");
                Ok(Value::Float(arr.value(idx)))
            }
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("String array");
                Ok(Value::String(arr.value(idx).to_string()))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .expect("Boolean array");
                Ok(Value::Bool(arr.value(idx)))
            }
            dt => Err(QueryError::Unsupported(format!(
                "Unsupported data type: {:?}",
                dt
            ))),
        }
    }
}

impl Default for ArrowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for building columns dynamically
enum ColumnBuilder {
    Int64(PrimitiveBuilder<Int64Type>),
    Float64(PrimitiveBuilder<Float64Type>),
    String(StringBuilder),
    Boolean(arrow::array::BooleanBuilder),
}

impl ColumnBuilder {
    fn from_value(value: &Value, capacity: usize) -> Self {
        match value {
            Value::Int(_) => Self::Int64(PrimitiveBuilder::with_capacity(capacity)),
            Value::Float(_) => Self::Float64(PrimitiveBuilder::with_capacity(capacity)),
            Value::Bool(_) => Self::Boolean(arrow::array::BooleanBuilder::with_capacity(capacity)),
            _ => Self::String(StringBuilder::with_capacity(capacity, capacity * 32)),
        }
    }

    fn append(&mut self, value: &Value) {
        match (self, value) {
            (Self::Int64(b), Value::Int(v)) => b.append_value(*v),
            (Self::Int64(b), Value::Null) => b.append_null(),
            (Self::Float64(b), Value::Float(v)) => b.append_value(*v),
            (Self::Float64(b), Value::Int(v)) => b.append_value(*v as f64),
            (Self::Float64(b), Value::Null) => b.append_null(),
            (Self::String(b), Value::String(v)) => b.append_value(v),
            (Self::String(b), Value::Null) => b.append_null(),
            (Self::Boolean(b), Value::Bool(v)) => b.append_value(*v),
            (Self::Boolean(b), Value::Null) => b.append_null(),
            // Type mismatch - append null or convert
            (Self::String(b), v) => b.append_value(format!("{:?}", v)),
            _ => {}
        }
    }

    fn finish(self) -> ArrayRef {
        match self {
            Self::Int64(mut b) => Arc::new(b.finish()),
            Self::Float64(mut b) => Arc::new(b.finish()),
            Self::String(mut b) => Arc::new(b.finish()),
            Self::Boolean(mut b) => Arc::new(b.finish()),
        }
    }
}

/// Combine multiple boolean masks with AND
pub fn and_masks(masks: &[&BooleanArray]) -> QueryResult<BooleanArray> {
    if masks.is_empty() {
        return Err(QueryError::ParseError("No masks to combine".to_string()));
    }

    let mut result = masks[0].clone();
    for mask in &masks[1..] {
        result =
            compute::and(&result, mask).map_err(|e| QueryError::ExecutionError(e.to_string()))?;
    }
    Ok(result)
}

/// Combine multiple boolean masks with OR
pub fn or_masks(masks: &[&BooleanArray]) -> QueryResult<BooleanArray> {
    if masks.is_empty() {
        return Err(QueryError::ParseError("No masks to combine".to_string()));
    }

    let mut result = masks[0].clone();
    for mask in &masks[1..] {
        result =
            compute::or(&result, mask).map_err(|e| QueryError::ExecutionError(e.to_string()))?;
    }
    Ok(result)
}

/// Negate a boolean mask
pub fn not_mask(mask: &BooleanArray) -> QueryResult<BooleanArray> {
    compute::not(mask).map_err(|e| QueryError::ExecutionError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_batch() -> RecordBatch {
        RecordBatchBuilder::new()
            .add_i64_column("id", vec![1, 2, 3, 4, 5])
            .add_string_column("name", vec!["Alice", "Bob", "Carol", "Dave", "Eve"])
            .add_f64_column("score", vec![95.0, 87.5, 92.0, 78.0, 88.5])
            .add_bool_column("active", vec![true, true, false, true, false])
            .build()
            .unwrap()
    }

    #[test]
    fn test_record_batch_builder() {
        let batch = sample_batch();

        assert_eq!(batch.num_rows(), 5);
        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.schema().field(0).name(), "id");
        assert_eq!(batch.schema().field(1).name(), "name");
    }

    #[test]
    fn test_filter_i64() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        // Filter id > 2
        let filtered = executor.filter_i64(&batch, "id", FilterOp::Gt, 2).unwrap();
        assert_eq!(filtered.num_rows(), 3); // 3, 4, 5

        // Filter id = 1
        let filtered = executor.filter_i64(&batch, "id", FilterOp::Eq, 1).unwrap();
        assert_eq!(filtered.num_rows(), 1);
    }

    #[test]
    fn test_filter_f64() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        // Filter score >= 90.0
        let filtered = executor
            .filter_f64(&batch, "score", FilterOp::Ge, 90.0)
            .unwrap();
        assert_eq!(filtered.num_rows(), 2); // 95.0, 92.0

        // Filter score < 85.0
        let filtered = executor
            .filter_f64(&batch, "score", FilterOp::Lt, 85.0)
            .unwrap();
        assert_eq!(filtered.num_rows(), 1); // 78.0
    }

    #[test]
    fn test_filter_string() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        // Filter name = "Bob"
        let filtered = executor
            .filter_string(&batch, "name", FilterOp::Eq, "Bob")
            .unwrap();
        assert_eq!(filtered.num_rows(), 1);
    }

    #[test]
    fn test_project() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        let projected = executor.project(&batch, &["id", "name"]).unwrap();

        assert_eq!(projected.num_columns(), 2);
        assert_eq!(projected.num_rows(), 5);
        assert_eq!(projected.schema().field(0).name(), "id");
        assert_eq!(projected.schema().field(1).name(), "name");
    }

    #[test]
    fn test_sort() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        // Sort by score descending
        let sorted = executor.sort(&batch, "score", true).unwrap();

        let score_col = sorted
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(score_col.value(0), 95.0); // Highest
        assert_eq!(score_col.value(4), 78.0); // Lowest
    }

    #[test]
    fn test_limit_offset() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        let limited = executor.limit(&batch, 3).unwrap();
        assert_eq!(limited.num_rows(), 3);

        let offset = executor.offset(&batch, 2).unwrap();
        assert_eq!(offset.num_rows(), 3);
    }

    #[test]
    fn test_aggregations() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        assert_eq!(executor.count(&batch), 5);
        assert_eq!(executor.sum_i64(&batch, "id").unwrap(), Some(15)); // 1+2+3+4+5
        assert_eq!(executor.min_i64(&batch, "id").unwrap(), Some(1));
        assert_eq!(executor.max_i64(&batch, "id").unwrap(), Some(5));

        let sum = executor.sum_f64(&batch, "score").unwrap().unwrap();
        assert!((sum - 441.0).abs() < 0.001); // 95+87.5+92+78+88.5

        let avg = executor.avg_f64(&batch, "score").unwrap().unwrap();
        assert!((avg - 88.2).abs() < 0.001);
    }

    #[test]
    fn test_to_result_set() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        let result = executor.to_result_set(&batch).unwrap();

        assert_eq!(result.columns, vec!["id", "name", "score", "active"]);
        assert_eq!(result.rows.len(), 5);
        assert_eq!(result.rows[0].values[0], Value::Int(1));
        assert_eq!(result.rows[0].values[1], Value::String("Alice".to_string()));
    }

    #[test]
    fn test_from_result_set() {
        let original = sample_batch();
        let executor = ArrowExecutor::new();

        // Convert to ResultSet and back
        let result_set = executor.to_result_set(&original).unwrap();
        let batch = executor.from_result_set(&result_set).unwrap();

        assert_eq!(batch.num_rows(), original.num_rows());
        assert_eq!(batch.num_columns(), original.num_columns());
    }

    #[test]
    fn test_chained_operations() {
        let batch = sample_batch();
        let executor = ArrowExecutor::new();

        // Filter -> Sort -> Limit pipeline
        let result = executor
            .filter_f64(&batch, "score", FilterOp::Ge, 85.0)
            .and_then(|b| executor.sort(&b, "score", true))
            .and_then(|b| executor.limit(&b, 2))
            .unwrap();

        assert_eq!(result.num_rows(), 2);

        // Check scores are sorted descending
        let scores = result
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(scores.value(0), 95.0);
        assert_eq!(scores.value(1), 92.0);
    }

    #[test]
    fn test_mask_operations() {
        let batch = sample_batch();

        let id_col = batch.column(0).as_primitive::<Int64Type>();
        let score_col = batch.column(2).as_primitive::<Float64Type>();

        let scalar1 = Int64Array::new_scalar(2);
        let scalar2 = Float64Array::new_scalar(85.0);

        let mask1 = cmp::gt(id_col, &scalar1).unwrap();
        let mask2 = cmp::gt(score_col, &scalar2).unwrap();

        // AND: id > 2 AND score > 85.0
        let combined = and_masks(&[&mask1, &mask2]).unwrap();
        let filtered = compute::filter_record_batch(&batch, &combined).unwrap();
        assert_eq!(filtered.num_rows(), 2); // Carol (92.0) and Eve (88.5)

        // OR: id > 2 OR score > 85.0
        let combined = or_masks(&[&mask1, &mask2]).unwrap();
        let filtered = compute::filter_record_batch(&batch, &combined).unwrap();
        assert_eq!(filtered.num_rows(), 5); // All match
    }
}
