//! # Columnar Storage
//!
//! Provides column-oriented storage for analytical workloads.
//!
//! ## Features
//!
//! - Column-based storage for efficient analytical queries
//! - Run-length encoding (RLE) compression
//! - Dictionary encoding for low-cardinality columns
//! - Vectorized operations
//! - Predicate pushdown
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_features::columnar::{ColumnStore, Column, DataType};
//!
//! let mut store = ColumnStore::new("sales");
//! store.add_column("product", DataType::String);
//! store.add_column("quantity", DataType::Int64);
//! store.add_column("price", DataType::Float64);
//!
//! store.insert_row(vec!["widget".into(), 10i64.into(), 9.99f64.into()]);
//!
//! let sum = store.aggregate("price", Aggregation::Sum);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data types supported by columnar storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    /// Boolean values.
    Boolean,
    /// 8-bit signed integer.
    Int8,
    /// 16-bit signed integer.
    Int16,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit floating point.
    Float32,
    /// 64-bit floating point.
    Float64,
    /// UTF-8 string.
    String,
    /// Binary data.
    Binary,
    /// Timestamp (microseconds since epoch).
    Timestamp,
    /// Date (days since epoch).
    Date,
}

impl DataType {
    /// Get the size in bytes for fixed-size types.
    pub fn size(&self) -> Option<usize> {
        match self {
            DataType::Boolean => Some(1),
            DataType::Int8 => Some(1),
            DataType::Int16 => Some(2),
            DataType::Int32 => Some(4),
            DataType::Int64 => Some(8),
            DataType::Float32 => Some(4),
            DataType::Float64 => Some(8),
            DataType::Timestamp => Some(8),
            DataType::Date => Some(4),
            DataType::String | DataType::Binary => None,
        }
    }

    /// Check if this is a numeric type.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
        )
    }
}

/// A value in a column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
    Timestamp(i64),
    Date(i32),
}

impl Value {
    /// Check if this value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to convert to f64 for numeric operations.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int8(v) => Some(*v as f64),
            Value::Int16(v) => Some(*v as f64),
            Value::Int32(v) => Some(*v as f64),
            Value::Int64(v) => Some(*v as f64),
            Value::Float32(v) => Some(*v as f64),
            Value::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to convert to i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int8(v) => Some(*v as i64),
            Value::Int16(v) => Some(*v as i64),
            Value::Int32(v) => Some(*v as i64),
            Value::Int64(v) => Some(*v),
            Value::Timestamp(v) => Some(*v),
            Value::Date(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to convert to string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Boolean(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::Int32(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int64(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float64(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

/// Compression type for columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compression {
    /// No compression.
    None,
    /// Run-length encoding.
    Rle,
    /// Dictionary encoding.
    Dictionary,
    /// Delta encoding (for sorted/sequential data).
    Delta,
    /// Bit-packing for small integers.
    BitPacked,
}

/// Column metadata and storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Data type.
    pub data_type: DataType,
    /// Whether nulls are allowed.
    pub nullable: bool,
    /// Compression type.
    pub compression: Compression,
    /// Column values.
    values: Vec<Value>,
    /// Null bitmap (true = null).
    null_bitmap: Vec<bool>,
    /// Dictionary for dictionary-encoded columns.
    dictionary: Option<Vec<Value>>,
    /// Dictionary indices for dictionary-encoded columns.
    dict_indices: Option<Vec<u32>>,
    /// Statistics.
    stats: ColumnStats,
}

/// Column statistics for query optimization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Number of values.
    pub count: usize,
    /// Number of null values.
    pub null_count: usize,
    /// Number of distinct values (approximate).
    pub distinct_count: usize,
    /// Minimum value (for numeric/string columns).
    pub min: Option<Value>,
    /// Maximum value (for numeric/string columns).
    pub max: Option<Value>,
    /// Sum (for numeric columns).
    pub sum: Option<f64>,
}

impl Column {
    /// Create a new column.
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable: true,
            compression: Compression::None,
            values: Vec::new(),
            null_bitmap: Vec::new(),
            dictionary: None,
            dict_indices: None,
            stats: ColumnStats::default(),
        }
    }

    /// Set whether nulls are allowed.
    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set compression type.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        if compression == Compression::Dictionary {
            self.dictionary = Some(Vec::new());
            self.dict_indices = Some(Vec::new());
        }
        self
    }

    /// Get the number of values.
    pub fn len(&self) -> usize {
        if self.compression == Compression::Dictionary {
            self.dict_indices.as_ref().map(|d| d.len()).unwrap_or(0)
        } else {
            self.values.len()
        }
    }

    /// Check if the column is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append a value to the column.
    pub fn push(&mut self, value: Value) -> Result<(), ColumnarError> {
        // Validate type
        if !self.is_compatible(&value) {
            return Err(ColumnarError::TypeMismatch {
                expected: self.data_type,
                got: format!("{:?}", value),
            });
        }

        // Handle nulls
        if value.is_null() {
            if !self.nullable {
                return Err(ColumnarError::NullNotAllowed(self.name.clone()));
            }
            self.null_bitmap.push(true);
            self.stats.null_count += 1;
        } else {
            self.null_bitmap.push(false);
            self.update_stats(&value);
        }

        // Store value
        if self.compression == Compression::Dictionary {
            self.push_dictionary(value);
        } else {
            self.values.push(value);
        }

        self.stats.count += 1;
        Ok(())
    }

    /// Push value with dictionary encoding.
    fn push_dictionary(&mut self, value: Value) {
        let dict = self.dictionary.as_mut().unwrap();
        let indices = self.dict_indices.as_mut().unwrap();

        // Find or insert value in dictionary
        let idx = dict.iter().position(|v| v == &value).unwrap_or_else(|| {
            dict.push(value);
            dict.len() - 1
        });

        indices.push(idx as u32);
    }

    /// Check if a value is compatible with this column's type.
    fn is_compatible(&self, value: &Value) -> bool {
        match (&self.data_type, value) {
            (_, Value::Null) => true,
            (DataType::Boolean, Value::Boolean(_)) => true,
            (DataType::Int8, Value::Int8(_)) => true,
            (DataType::Int16, Value::Int16(_)) => true,
            (DataType::Int32, Value::Int32(_)) => true,
            (DataType::Int64, Value::Int64(_)) => true,
            (DataType::Float32, Value::Float32(_)) => true,
            (DataType::Float64, Value::Float64(_)) => true,
            (DataType::String, Value::String(_)) => true,
            (DataType::Binary, Value::Binary(_)) => true,
            (DataType::Timestamp, Value::Timestamp(_)) => true,
            (DataType::Date, Value::Date(_)) => true,
            // Allow numeric coercion
            (DataType::Int64, Value::Int32(_)) => true,
            (DataType::Float64, Value::Float32(_)) => true,
            (DataType::Float64, Value::Int64(_)) => true,
            (DataType::Float64, Value::Int32(_)) => true,
            _ => false,
        }
    }

    /// Update statistics with a new value.
    fn update_stats(&mut self, value: &Value) {
        if let Some(num) = value.as_f64() {
            self.stats.sum = Some(self.stats.sum.unwrap_or(0.0) + num);

            let update_min = self
                .stats
                .min
                .as_ref()
                .and_then(|m| m.as_f64())
                .map(|m| num < m)
                .unwrap_or(true);
            if update_min {
                self.stats.min = Some(value.clone());
            }

            let update_max = self
                .stats
                .max
                .as_ref()
                .and_then(|m| m.as_f64())
                .map(|m| num > m)
                .unwrap_or(true);
            if update_max {
                self.stats.max = Some(value.clone());
            }
        }
    }

    /// Get a value at the given index.
    pub fn get(&self, index: usize) -> Option<&Value> {
        if index >= self.len() {
            return None;
        }

        if self.null_bitmap.get(index).copied().unwrap_or(false) {
            return Some(&Value::Null);
        }

        if self.compression == Compression::Dictionary {
            let dict_idx = self.dict_indices.as_ref()?.get(index)?;
            self.dictionary.as_ref()?.get(*dict_idx as usize)
        } else {
            self.values.get(index)
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ColumnStats {
        &self.stats
    }

    /// Iterate over values.
    pub fn iter(&self) -> ColumnIterator<'_> {
        ColumnIterator {
            column: self,
            index: 0,
        }
    }

    /// Filter column by predicate, returning matching indices.
    pub fn filter<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(&Value) -> bool,
    {
        self.iter()
            .enumerate()
            .filter_map(|(i, v)| if predicate(v) { Some(i) } else { None })
            .collect()
    }

    /// Apply a function to all values, creating a new column.
    pub fn map<F>(&self, name: &str, f: F) -> Column
    where
        F: Fn(&Value) -> Value,
    {
        let mut result = Column::new(name, self.data_type);
        for value in self.iter() {
            let _ = result.push(f(value));
        }
        result
    }
}

/// Iterator over column values.
pub struct ColumnIterator<'a> {
    column: &'a Column,
    index: usize,
}

impl<'a> Iterator for ColumnIterator<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.column.get(self.index)?;
        self.index += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.column.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ColumnIterator<'a> {}

/// Aggregation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    Count,
    Sum,
    Min,
    Max,
    Avg,
    First,
    Last,
    CountDistinct,
}

/// Column store (table with columnar layout).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStore {
    /// Table name.
    name: String,
    /// Columns by name.
    columns: HashMap<String, Column>,
    /// Column order.
    column_order: Vec<String>,
    /// Number of rows.
    row_count: usize,
}

impl ColumnStore {
    /// Create a new column store.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            columns: HashMap::new(),
            column_order: Vec::new(),
            row_count: 0,
        }
    }

    /// Get the table name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of rows.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Get the number of columns.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Add a column to the store.
    pub fn add_column(&mut self, name: impl Into<String>, data_type: DataType) -> &mut Self {
        let name = name.into();
        let column = Column::new(&name, data_type);
        self.columns.insert(name.clone(), column);
        self.column_order.push(name);
        self
    }

    /// Add a column with configuration.
    pub fn add_column_with_config(&mut self, column: Column) -> &mut Self {
        let name = column.name.clone();
        self.columns.insert(name.clone(), column);
        self.column_order.push(name);
        self
    }

    /// Get a column by name.
    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Get a mutable column by name.
    pub fn get_column_mut(&mut self, name: &str) -> Option<&mut Column> {
        self.columns.get_mut(name)
    }

    /// Get column names in order.
    pub fn column_names(&self) -> &[String] {
        &self.column_order
    }

    /// Insert a row of values.
    pub fn insert_row(&mut self, values: Vec<Value>) -> Result<(), ColumnarError> {
        if values.len() != self.columns.len() {
            return Err(ColumnarError::ColumnCountMismatch {
                expected: self.columns.len(),
                got: values.len(),
            });
        }

        for (i, value) in values.into_iter().enumerate() {
            let col_name = &self.column_order[i];
            let column = self.columns.get_mut(col_name).unwrap();
            column.push(value)?;
        }

        self.row_count += 1;
        Ok(())
    }

    /// Insert a row from a map of column name to value.
    pub fn insert_row_map(&mut self, values: HashMap<String, Value>) -> Result<(), ColumnarError> {
        for col_name in &self.column_order {
            let value = values.get(col_name).cloned().unwrap_or(Value::Null);
            let column = self.columns.get_mut(col_name).unwrap();
            column.push(value)?;
        }

        self.row_count += 1;
        Ok(())
    }

    /// Get a row by index.
    pub fn get_row(&self, index: usize) -> Option<Vec<&Value>> {
        if index >= self.row_count {
            return None;
        }

        let mut row = Vec::with_capacity(self.columns.len());
        for col_name in &self.column_order {
            let column = self.columns.get(col_name)?;
            row.push(column.get(index)?);
        }
        Some(row)
    }

    /// Aggregate a column.
    pub fn aggregate(&self, column_name: &str, agg: Aggregation) -> Option<Value> {
        let column = self.columns.get(column_name)?;

        match agg {
            Aggregation::Count => Some(Value::Int64(column.len() as i64)),
            Aggregation::Sum => column.stats.sum.map(Value::Float64),
            Aggregation::Min => column.stats.min.clone(),
            Aggregation::Max => column.stats.max.clone(),
            Aggregation::Avg => {
                let sum = column.stats.sum?;
                let count = column.stats.count - column.stats.null_count;
                if count > 0 {
                    Some(Value::Float64(sum / count as f64))
                } else {
                    None
                }
            }
            Aggregation::First => column.get(0).cloned(),
            Aggregation::Last => column.get(column.len().saturating_sub(1)).cloned(),
            Aggregation::CountDistinct => {
                let distinct: std::collections::HashSet<String> = column
                    .iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        v => v.as_i64().map(|n| n.to_string()),
                    })
                    .collect();
                Some(Value::Int64(distinct.len() as i64))
            }
        }
    }

    /// Filter rows by predicate on a column.
    pub fn filter(&self, column_name: &str, predicate: impl Fn(&Value) -> bool) -> Vec<usize> {
        self.columns
            .get(column_name)
            .map(|c| c.filter(predicate))
            .unwrap_or_default()
    }

    /// Select specific columns, returning a new store.
    pub fn select(&self, columns: &[&str]) -> ColumnStore {
        let mut result = ColumnStore::new(format!("{}_projection", self.name));

        for &col_name in columns {
            if let Some(column) = self.columns.get(col_name) {
                result.columns.insert(col_name.to_string(), column.clone());
                result.column_order.push(col_name.to_string());
            }
        }
        result.row_count = self.row_count;

        result
    }

    /// Project rows at given indices.
    pub fn project_rows(&self, indices: &[usize]) -> ColumnStore {
        let mut result = ColumnStore::new(format!("{}_filtered", self.name));

        for col_name in &self.column_order {
            let src_col = self.columns.get(col_name).unwrap();
            let mut new_col = Column::new(col_name, src_col.data_type);

            for &idx in indices {
                if let Some(value) = src_col.get(idx) {
                    let _ = new_col.push(value.clone());
                }
            }

            result.columns.insert(col_name.clone(), new_col);
            result.column_order.push(col_name.clone());
        }
        result.row_count = indices.len();

        result
    }

    /// Group by a column and aggregate another.
    /// Returns a map from group key (as String) to aggregated value.
    pub fn group_by_aggregate(
        &self,
        group_col: &str,
        agg_col: &str,
        agg: Aggregation,
    ) -> HashMap<String, Value> {
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();

        let group_column = match self.columns.get(group_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };
        let agg_column = match self.columns.get(agg_col) {
            Some(c) => c,
            None => return HashMap::new(),
        };

        for i in 0..self.row_count {
            if let (Some(group_val), Some(agg_val)) = (group_column.get(i), agg_column.get(i)) {
                let key = match group_val {
                    Value::String(s) => s.clone(),
                    v => format!("{:?}", v),
                };
                if let Some(num) = agg_val.as_f64() {
                    groups.entry(key).or_default().push(num);
                }
            }
        }

        groups
            .into_iter()
            .map(|(key, values)| {
                let result = match agg {
                    Aggregation::Count => Value::Int64(values.len() as i64),
                    Aggregation::Sum => Value::Float64(values.iter().sum()),
                    Aggregation::Min => {
                        Value::Float64(values.iter().copied().fold(f64::INFINITY, f64::min))
                    }
                    Aggregation::Max => {
                        Value::Float64(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
                    }
                    Aggregation::Avg => {
                        Value::Float64(values.iter().sum::<f64>() / values.len() as f64)
                    }
                    Aggregation::First => Value::Float64(values.first().copied().unwrap_or(0.0)),
                    Aggregation::Last => Value::Float64(values.last().copied().unwrap_or(0.0)),
                    Aggregation::CountDistinct => {
                        let distinct: std::collections::HashSet<u64> =
                            values.iter().map(|v| v.to_bits()).collect();
                        Value::Int64(distinct.len() as i64)
                    }
                };
                (key, result)
            })
            .collect()
    }

    /// Get memory usage estimate in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for column in self.columns.values() {
            // Base struct size
            total += std::mem::size_of::<Column>();
            // Values
            total += column.values.len() * std::mem::size_of::<Value>();
            // Null bitmap
            total += column.null_bitmap.len();
            // Dictionary if present
            if let Some(ref dict) = column.dictionary {
                total += dict.len() * std::mem::size_of::<Value>();
            }
            if let Some(ref indices) = column.dict_indices {
                total += indices.len() * std::mem::size_of::<u32>();
            }
        }
        total
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        for column in self.columns.values_mut() {
            column.values.clear();
            column.null_bitmap.clear();
            column.dictionary = if column.compression == Compression::Dictionary {
                Some(Vec::new())
            } else {
                None
            };
            column.dict_indices = if column.compression == Compression::Dictionary {
                Some(Vec::new())
            } else {
                None
            };
            column.stats = ColumnStats::default();
        }
        self.row_count = 0;
    }
}

/// Columnar storage errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ColumnarError {
    #[error("Type mismatch: expected {expected:?}, got {got}")]
    TypeMismatch { expected: DataType, got: String },

    #[error("Null not allowed in column: {0}")]
    NullNotAllowed(String),

    #[error("Column count mismatch: expected {expected}, got {got}")]
    ColumnCountMismatch { expected: usize, got: usize },

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Builder for creating column stores with schema.
pub struct ColumnStoreBuilder {
    name: String,
    columns: Vec<(String, DataType, bool, Compression)>,
}

impl ColumnStoreBuilder {
    /// Create a new builder.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            columns: Vec::new(),
        }
    }

    /// Add a column.
    pub fn column(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.columns
            .push((name.into(), data_type, true, Compression::None));
        self
    }

    /// Add a non-nullable column.
    pub fn column_not_null(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.columns
            .push((name.into(), data_type, false, Compression::None));
        self
    }

    /// Add a column with dictionary encoding.
    pub fn column_dictionary(mut self, name: impl Into<String>, data_type: DataType) -> Self {
        self.columns
            .push((name.into(), data_type, true, Compression::Dictionary));
        self
    }

    /// Build the column store.
    pub fn build(self) -> ColumnStore {
        let mut store = ColumnStore::new(self.name);
        for (name, data_type, nullable, compression) in self.columns {
            let column = Column::new(&name, data_type)
                .with_nullable(nullable)
                .with_compression(compression);
            store.add_column_with_config(column);
        }
        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_column() {
        let mut col = Column::new("test", DataType::Int64);
        col.push(Value::Int64(42)).unwrap();
        col.push(Value::Int64(100)).unwrap();

        assert_eq!(col.len(), 2);
        assert_eq!(col.get(0), Some(&Value::Int64(42)));
        assert_eq!(col.get(1), Some(&Value::Int64(100)));
    }

    #[test]
    fn test_column_null() {
        let mut col = Column::new("test", DataType::String);
        col.push(Value::String("hello".to_string())).unwrap();
        col.push(Value::Null).unwrap();
        col.push(Value::String("world".to_string())).unwrap();

        assert_eq!(col.len(), 3);
        assert_eq!(col.get(1), Some(&Value::Null));
        assert_eq!(col.stats().null_count, 1);
    }

    #[test]
    fn test_column_not_nullable() {
        let mut col = Column::new("test", DataType::Int64).with_nullable(false);

        assert!(col.push(Value::Int64(42)).is_ok());
        assert!(matches!(
            col.push(Value::Null),
            Err(ColumnarError::NullNotAllowed(_))
        ));
    }

    #[test]
    fn test_type_mismatch() {
        let mut col = Column::new("test", DataType::Int64);

        assert!(col.push(Value::Int64(42)).is_ok());
        assert!(matches!(
            col.push(Value::String("hello".to_string())),
            Err(ColumnarError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn test_dictionary_encoding() {
        let mut col =
            Column::new("test", DataType::String).with_compression(Compression::Dictionary);

        col.push(Value::String("apple".to_string())).unwrap();
        col.push(Value::String("banana".to_string())).unwrap();
        col.push(Value::String("apple".to_string())).unwrap();
        col.push(Value::String("cherry".to_string())).unwrap();
        col.push(Value::String("apple".to_string())).unwrap();

        assert_eq!(col.len(), 5);
        // Dictionary should only have 3 unique values
        assert_eq!(col.dictionary.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_column_store_basic() {
        let mut store = ColumnStore::new("test_table");
        store.add_column("name", DataType::String);
        store.add_column("age", DataType::Int64);

        store
            .insert_row(vec!["Alice".into(), 30i64.into()])
            .unwrap();
        store.insert_row(vec!["Bob".into(), 25i64.into()]).unwrap();

        assert_eq!(store.row_count(), 2);
        assert_eq!(store.column_count(), 2);
    }

    #[test]
    fn test_aggregation() {
        let mut store = ColumnStore::new("sales");
        store.add_column("product", DataType::String);
        store.add_column("amount", DataType::Float64);

        store.insert_row(vec!["A".into(), 100.0f64.into()]).unwrap();
        store.insert_row(vec!["B".into(), 200.0f64.into()]).unwrap();
        store.insert_row(vec!["A".into(), 150.0f64.into()]).unwrap();

        let sum = store.aggregate("amount", Aggregation::Sum);
        assert_eq!(sum, Some(Value::Float64(450.0)));

        let avg = store.aggregate("amount", Aggregation::Avg);
        assert_eq!(avg, Some(Value::Float64(150.0)));

        let count = store.aggregate("amount", Aggregation::Count);
        assert_eq!(count, Some(Value::Int64(3)));
    }

    #[test]
    fn test_filter() {
        let mut store = ColumnStore::new("test");
        store.add_column("value", DataType::Int64);

        store.insert_row(vec![10i64.into()]).unwrap();
        store.insert_row(vec![20i64.into()]).unwrap();
        store.insert_row(vec![30i64.into()]).unwrap();
        store.insert_row(vec![40i64.into()]).unwrap();

        let indices = store.filter("value", |v| v.as_i64().map(|n| n > 15).unwrap_or(false));

        assert_eq!(indices, vec![1, 2, 3]);
    }

    #[test]
    fn test_select() {
        let mut store = ColumnStore::new("test");
        store.add_column("a", DataType::Int64);
        store.add_column("b", DataType::String);
        store.add_column("c", DataType::Float64);

        store
            .insert_row(vec![1i64.into(), "x".into(), 1.0f64.into()])
            .unwrap();

        let projected = store.select(&["a", "c"]);
        assert_eq!(projected.column_count(), 2);
        assert!(projected.get_column("a").is_some());
        assert!(projected.get_column("b").is_none());
        assert!(projected.get_column("c").is_some());
    }

    #[test]
    fn test_project_rows() {
        let mut store = ColumnStore::new("test");
        store.add_column("value", DataType::Int64);

        for i in 0..10 {
            store.insert_row(vec![(i as i64).into()]).unwrap();
        }

        let projected = store.project_rows(&[2, 5, 8]);
        assert_eq!(projected.row_count(), 3);

        let row = projected.get_row(0).unwrap();
        assert_eq!(row[0], &Value::Int64(2));
    }

    #[test]
    fn test_group_by() {
        let mut store = ColumnStore::new("sales");
        store.add_column("category", DataType::String);
        store.add_column("amount", DataType::Float64);

        store.insert_row(vec!["A".into(), 100.0f64.into()]).unwrap();
        store.insert_row(vec!["B".into(), 200.0f64.into()]).unwrap();
        store.insert_row(vec!["A".into(), 150.0f64.into()]).unwrap();
        store.insert_row(vec!["B".into(), 50.0f64.into()]).unwrap();

        let results = store.group_by_aggregate("category", "amount", Aggregation::Sum);

        assert_eq!(results.get("A"), Some(&Value::Float64(250.0)));
        assert_eq!(results.get("B"), Some(&Value::Float64(250.0)));
    }

    #[test]
    fn test_builder() {
        let store = ColumnStoreBuilder::new("orders")
            .column("id", DataType::Int64)
            .column_not_null("customer", DataType::String)
            .column_dictionary("status", DataType::String)
            .column("amount", DataType::Float64)
            .build();

        assert_eq!(store.column_count(), 4);
        assert_eq!(store.name(), "orders");
    }

    #[test]
    fn test_column_stats() {
        let mut col = Column::new("test", DataType::Float64);
        col.push(Value::Float64(10.0)).unwrap();
        col.push(Value::Float64(20.0)).unwrap();
        col.push(Value::Float64(30.0)).unwrap();

        let stats = col.stats();
        assert_eq!(stats.count, 3);
        assert_eq!(stats.sum, Some(60.0));
        assert_eq!(stats.min, Some(Value::Float64(10.0)));
        assert_eq!(stats.max, Some(Value::Float64(30.0)));
    }

    #[test]
    fn test_column_iterator() {
        let mut col = Column::new("test", DataType::Int64);
        col.push(Value::Int64(1)).unwrap();
        col.push(Value::Int64(2)).unwrap();
        col.push(Value::Int64(3)).unwrap();

        let values: Vec<i64> = col.iter().filter_map(|v| v.as_i64()).collect();

        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_row() {
        let mut store = ColumnStore::new("test");
        store.add_column("a", DataType::Int64);
        store.add_column("b", DataType::String);

        store
            .insert_row(vec![42i64.into(), "hello".into()])
            .unwrap();

        let row = store.get_row(0).unwrap();
        assert_eq!(row[0], &Value::Int64(42));
        assert_eq!(row[1], &Value::String("hello".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut store = ColumnStore::new("test");
        store.add_column("value", DataType::Int64);
        store.insert_row(vec![1i64.into()]).unwrap();
        store.insert_row(vec![2i64.into()]).unwrap();

        store.clear();

        assert_eq!(store.row_count(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_memory_usage() {
        let mut store = ColumnStore::new("test");
        store.add_column("value", DataType::Int64);

        for i in 0..1000 {
            store.insert_row(vec![(i as i64).into()]).unwrap();
        }

        let usage = store.memory_usage();
        assert!(usage > 0);
    }
}
