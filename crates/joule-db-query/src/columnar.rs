//! Columnar Storage Engine (Prototype)
//!
//! A minimal columnar storage engine optimized for OLAP workloads.
//! Designed to work alongside the row-based storage for analytical queries.
//!
//! ## Features
//! - Column-oriented storage layout
//! - Compression support (RLE, dictionary, delta)
//! - Vectorized predicate pushdown
//! - Zone maps for pruning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ============================================================================
// Column Types and Data
// ============================================================================

/// Columnar data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnType {
    /// 64-bit integer
    Int64,
    /// 64-bit float
    Float64,
    /// Variable-length string
    String,
    /// Boolean
    Boolean,
    /// Binary data
    Bytes,
    /// Balanced ternary trits ({-1, 0, +1} stored as i8)
    Ternary,
}

/// Compressed column data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnData {
    /// Uncompressed i64 values
    Int64(Vec<i64>),
    /// Uncompressed f64 values
    Float64(Vec<f64>),
    /// Uncompressed strings
    String(Vec<String>),
    /// Uncompressed booleans
    Boolean(Vec<bool>),
    /// Uncompressed bytes
    Bytes(Vec<Vec<u8>>),
    /// Run-length encoded i64
    RleInt64 {
        values: Vec<i64>,
        run_lengths: Vec<u32>,
    },
    /// Dictionary encoded strings
    DictString {
        dictionary: Vec<String>,
        indices: Vec<u32>,
    },
    /// Delta encoded i64 (for sorted/sequential data)
    DeltaInt64 { base: i64, deltas: Vec<i32> },
    /// Balanced ternary trits ({-1, 0, +1} as i8)
    Ternary(Vec<i8>),
}

impl ColumnData {
    /// Get the number of values
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Int64(v) => v.len(),
            ColumnData::Float64(v) => v.len(),
            ColumnData::String(v) => v.len(),
            ColumnData::Boolean(v) => v.len(),
            ColumnData::Bytes(v) => v.len(),
            ColumnData::RleInt64 { run_lengths, .. } => {
                run_lengths.iter().map(|r| *r as usize).sum()
            }
            ColumnData::DictString { indices, .. } => indices.len(),
            ColumnData::DeltaInt64 { deltas, .. } => deltas.len() + 1,
            ColumnData::Ternary(v) => v.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get i64 value at index (decoding if needed)
    pub fn get_i64(&self, idx: usize) -> Option<i64> {
        match self {
            ColumnData::Int64(v) => v.get(idx).copied(),
            ColumnData::RleInt64 {
                values,
                run_lengths,
            } => {
                let mut pos = 0;
                for (i, &run_len) in run_lengths.iter().enumerate() {
                    if idx < pos + run_len as usize {
                        return values.get(i).copied();
                    }
                    pos += run_len as usize;
                }
                None
            }
            ColumnData::DeltaInt64 { base, deltas } => {
                if idx == 0 {
                    Some(*base)
                } else if idx <= deltas.len() {
                    let mut val = *base;
                    for delta in deltas.iter().take(idx) {
                        val += *delta as i64;
                    }
                    Some(val)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get string value at index
    pub fn get_string(&self, idx: usize) -> Option<&str> {
        match self {
            ColumnData::String(v) => v.get(idx).map(|s| s.as_str()),
            ColumnData::DictString {
                dictionary,
                indices,
            } => indices
                .get(idx)
                .and_then(|&i| dictionary.get(i as usize).map(|s| s.as_str())),
            _ => None,
        }
    }

    /// Get ternary trit value at index
    pub fn get_ternary(&self, idx: usize) -> Option<i8> {
        match self {
            ColumnData::Ternary(v) => v.get(idx).copied(),
            _ => None,
        }
    }
}

// ============================================================================
// Zone Maps (Min/Max Statistics)
// ============================================================================

/// Zone map for efficient column pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneMap {
    /// Minimum value (as bytes for comparison)
    pub min: Vec<u8>,
    /// Maximum value
    pub max: Vec<u8>,
    /// Number of nulls
    pub null_count: u64,
    /// Total row count
    pub row_count: u64,
}

impl ZoneMap {
    /// Create a zone map for i64 column
    pub fn from_i64(values: &[i64]) -> Self {
        let min = values.iter().copied().min().unwrap_or(0);
        let max = values.iter().copied().max().unwrap_or(0);
        Self {
            min: min.to_le_bytes().to_vec(),
            max: max.to_le_bytes().to_vec(),
            null_count: 0,
            row_count: values.len() as u64,
        }
    }

    /// Check if a value could be in the zone (for predicate pushdown)
    pub fn may_contain_i64(&self, value: i64, op: &str) -> bool {
        if self.min.len() != 8 || self.max.len() != 8 {
            return true; // Assume yes if can't determine
        }
        let min = i64::from_le_bytes(
            self.min[..8]
                .try_into()
                .expect("min slice is exactly 8 bytes"),
        );
        let max = i64::from_le_bytes(
            self.max[..8]
                .try_into()
                .expect("max slice is exactly 8 bytes"),
        );

        match op {
            "=" => value >= min && value <= max,
            "<" => min < value,
            "<=" => min <= value,
            ">" => max > value,
            ">=" => max >= value,
            _ => true,
        }
    }
}

// ============================================================================
// Column Group (Chunk of Columns)
// ============================================================================

/// A group of columns, typically representing a chunk of a table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnGroup {
    /// Number of rows in this group
    pub row_count: usize,
    /// Column data by name
    pub columns: HashMap<String, ColumnData>,
    /// Zone maps for each column
    pub zone_maps: HashMap<String, ZoneMap>,
}

impl ColumnGroup {
    /// Create a new empty column group
    pub fn new() -> Self {
        Self {
            row_count: 0,
            columns: HashMap::new(),
            zone_maps: HashMap::new(),
        }
    }

    /// Add an i64 column
    pub fn add_i64_column(&mut self, name: &str, values: Vec<i64>) {
        let zone_map = ZoneMap::from_i64(&values);
        self.row_count = values.len();
        self.columns
            .insert(name.to_string(), ColumnData::Int64(values));
        self.zone_maps.insert(name.to_string(), zone_map);
    }

    /// Add a string column (with dictionary encoding if beneficial)
    pub fn add_string_column(&mut self, name: &str, values: Vec<String>) {
        let unique_count = values
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();

        // Use dictionary encoding if less than 50% unique values
        let data = if unique_count < values.len() / 2 {
            let mut dictionary = Vec::new();
            let mut dict_map = HashMap::new();
            let mut indices = Vec::with_capacity(values.len());

            for v in &values {
                let idx = *dict_map.entry(v.clone()).or_insert_with(|| {
                    let idx = dictionary.len() as u32;
                    dictionary.push(v.clone());
                    idx
                });
                indices.push(idx);
            }

            ColumnData::DictString {
                dictionary,
                indices,
            }
        } else {
            ColumnData::String(values)
        };

        self.row_count = data.len();
        self.columns.insert(name.to_string(), data);
    }

    /// Add a ternary column (balanced trits: -1, 0, +1)
    pub fn add_ternary_column(&mut self, name: &str, values: Vec<i8>) {
        self.row_count = values.len();
        self.columns
            .insert(name.to_string(), ColumnData::Ternary(values));
    }

    /// Get a column by name
    pub fn get_column(&self, name: &str) -> Option<&ColumnData> {
        self.columns.get(name)
    }

    /// Check if a predicate might have matches (using zone maps)
    pub fn may_match(&self, column: &str, value: i64, op: &str) -> bool {
        if let Some(zone_map) = self.zone_maps.get(column) {
            zone_map.may_contain_i64(value, op)
        } else {
            true // No zone map, assume might match
        }
    }
}

impl Default for ColumnGroup {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Columnar Table
// ============================================================================

/// A columnar table consisting of multiple column groups
pub struct ColumnarTable {
    /// Table name
    name: String,
    /// Column schema
    columns: Vec<(String, ColumnType)>,
    /// Column groups (chunks)
    groups: Vec<ColumnGroup>,
    /// Target rows per group
    group_size: usize,
}

impl ColumnarTable {
    /// Create a new columnar table
    pub fn new(name: &str, columns: Vec<(String, ColumnType)>) -> Self {
        Self {
            name: name.to_string(),
            columns,
            groups: Vec::new(),
            group_size: 10000, // 10k rows per group by default
        }
    }

    /// Get table name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get column schema
    pub fn schema(&self) -> &[(String, ColumnType)] {
        &self.columns
    }

    /// Add a column group
    pub fn add_group(&mut self, group: ColumnGroup) {
        self.groups.push(group);
    }

    /// Get total row count
    pub fn row_count(&self) -> usize {
        self.groups.iter().map(|g| g.row_count).sum()
    }

    /// Get group count
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Scan with optional predicate pushdown
    pub fn scan(&self, columns: &[&str], predicate: Option<(&str, &str, i64)>) -> Vec<ColumnGroup> {
        let mut result = Vec::new();

        for group in &self.groups {
            // Check zone map for early pruning
            if let Some((col, op, val)) = predicate {
                if !group.may_match(col, val, op) {
                    continue; // Skip this group entirely
                }
            }

            // Create result group with only requested columns
            let mut result_group = ColumnGroup::new();
            result_group.row_count = group.row_count;

            for &col_name in columns {
                if let Some(data) = group.columns.get(col_name) {
                    result_group
                        .columns
                        .insert(col_name.to_string(), data.clone());
                }
            }

            result.push(result_group);
        }

        result
    }

    /// Aggregate a column
    pub fn aggregate(&self, column: &str, op: &str) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0u64;
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for group in &self.groups {
            if let Some(data) = group.get_column(column) {
                match data {
                    ColumnData::Int64(values) => {
                        for &v in values {
                            let f = v as f64;
                            sum += f;
                            count += 1;
                            min = min.min(f);
                            max = max.max(f);
                        }
                    }
                    ColumnData::Float64(values) => {
                        for &v in values {
                            sum += v;
                            count += 1;
                            min = min.min(v);
                            max = max.max(v);
                        }
                    }
                    _ => {}
                }
            }
        }

        if count == 0 {
            return None;
        }

        match op {
            "SUM" => Some(sum),
            "AVG" => Some(sum / count as f64),
            "MIN" => Some(min),
            "MAX" => Some(max),
            "COUNT" => Some(count as f64),
            _ => None,
        }
    }
}

// ============================================================================
// Columnar Store (Multi-table)
// ============================================================================

/// A collection of columnar tables
pub struct ColumnarStore {
    /// Tables by name
    tables: RwLock<HashMap<String, Arc<RwLock<ColumnarTable>>>>,
}

impl ColumnarStore {
    /// Create a new columnar store
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new table
    pub fn create_table(&self, name: &str, columns: Vec<(String, ColumnType)>) -> bool {
        let mut tables = self.tables.write().expect("tables lock poisoned");
        if tables.contains_key(name) {
            return false;
        }
        tables.insert(
            name.to_string(),
            Arc::new(RwLock::new(ColumnarTable::new(name, columns))),
        );
        true
    }

    /// Get a table
    pub fn get_table(&self, name: &str) -> Option<Arc<RwLock<ColumnarTable>>> {
        self.tables
            .read()
            .expect("tables lock poisoned")
            .get(name)
            .cloned()
    }

    /// List all tables
    pub fn list_tables(&self) -> Vec<String> {
        self.tables
            .read()
            .expect("tables lock poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// Drop a table
    pub fn drop_table(&self, name: &str) -> bool {
        self.tables
            .write()
            .expect("tables lock poisoned")
            .remove(name)
            .is_some()
    }
}

impl Default for ColumnarStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_data_i64() {
        let data = ColumnData::Int64(vec![1, 2, 3, 4, 5]);
        assert_eq!(data.len(), 5);
        assert_eq!(data.get_i64(2), Some(3));
    }

    #[test]
    fn test_rle_encoding() {
        let data = ColumnData::RleInt64 {
            values: vec![10, 20, 30],
            run_lengths: vec![3, 2, 5],
        };
        assert_eq!(data.len(), 10);
        assert_eq!(data.get_i64(0), Some(10)); // First run
        assert_eq!(data.get_i64(3), Some(20)); // Second run
        assert_eq!(data.get_i64(5), Some(30)); // Third run
    }

    #[test]
    fn test_delta_encoding() {
        let data = ColumnData::DeltaInt64 {
            base: 100,
            deltas: vec![1, 1, 1, 1],
        };
        assert_eq!(data.len(), 5);
        assert_eq!(data.get_i64(0), Some(100));
        assert_eq!(data.get_i64(1), Some(101));
        assert_eq!(data.get_i64(4), Some(104));
    }

    #[test]
    fn test_dict_encoding() {
        let data = ColumnData::DictString {
            dictionary: vec!["apple".to_string(), "banana".to_string()],
            indices: vec![0, 1, 0, 0, 1],
        };
        assert_eq!(data.len(), 5);
        assert_eq!(data.get_string(0), Some("apple"));
        assert_eq!(data.get_string(1), Some("banana"));
    }

    #[test]
    fn test_zone_map() {
        let zone_map = ZoneMap::from_i64(&[10, 50, 30, 20, 40]);
        assert!(zone_map.may_contain_i64(25, "="));
        assert!(!zone_map.may_contain_i64(5, "="));
        assert!(!zone_map.may_contain_i64(60, "="));
        assert!(!zone_map.may_contain_i64(5, "<"));
        assert!(zone_map.may_contain_i64(5, ">"));
    }

    #[test]
    fn test_columnar_table() {
        let mut table = ColumnarTable::new(
            "sales",
            vec![
                ("id".to_string(), ColumnType::Int64),
                ("amount".to_string(), ColumnType::Float64),
            ],
        );

        let mut group = ColumnGroup::new();
        group.add_i64_column("id", vec![1, 2, 3, 4, 5]);
        table.add_group(group);

        assert_eq!(table.row_count(), 5);
        assert_eq!(table.group_count(), 1);
    }

    #[test]
    fn test_columnar_aggregation() {
        let mut table =
            ColumnarTable::new("numbers", vec![("value".to_string(), ColumnType::Int64)]);

        let mut group = ColumnGroup::new();
        group.add_i64_column("value", vec![10, 20, 30, 40, 50]);
        table.add_group(group);

        assert_eq!(table.aggregate("value", "SUM"), Some(150.0));
        assert_eq!(table.aggregate("value", "AVG"), Some(30.0));
        assert_eq!(table.aggregate("value", "MIN"), Some(10.0));
        assert_eq!(table.aggregate("value", "MAX"), Some(50.0));
        assert_eq!(table.aggregate("value", "COUNT"), Some(5.0));
    }

    #[test]
    fn test_columnar_store() {
        let store = ColumnarStore::new();

        store.create_table(
            "users",
            vec![
                ("id".to_string(), ColumnType::Int64),
                ("name".to_string(), ColumnType::String),
            ],
        );

        let table = store.get_table("users");
        assert!(table.is_some());

        assert_eq!(store.list_tables(), vec!["users"]);

        store.drop_table("users");
        assert!(store.get_table("users").is_none());
    }

    #[test]
    fn test_zone_map_pruning() {
        let mut table = ColumnarTable::new("data", vec![("value".to_string(), ColumnType::Int64)]);

        // Group 1: values 1-100
        let mut group1 = ColumnGroup::new();
        group1.add_i64_column("value", (1..=100).collect());
        table.add_group(group1);

        // Group 2: values 101-200
        let mut group2 = ColumnGroup::new();
        group2.add_i64_column("value", (101..=200).collect());
        table.add_group(group2);

        // Scan for value < 50 should only return first group
        let results = table.scan(&["value"], Some(("value", "<", 50)));
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_ternary_column() {
        let trits: Vec<i8> = vec![-1, 0, 1, 1, -1, 0, 1, -1];
        let data = ColumnData::Ternary(trits.clone());
        assert_eq!(data.len(), 8);
        assert_eq!(data.get_ternary(0), Some(-1));
        assert_eq!(data.get_ternary(2), Some(1));
        assert_eq!(data.get_ternary(5), Some(0));
        assert_eq!(data.get_ternary(8), None);
        // i64 accessor returns None for ternary
        assert_eq!(data.get_i64(0), None);
    }

    #[test]
    fn test_ternary_column_group() {
        let mut group = ColumnGroup::new();
        group.add_ternary_column("weights", vec![-1, 0, 1, 1, -1]);
        assert_eq!(group.row_count, 5);
        let col = group.get_column("weights").unwrap();
        assert_eq!(col.get_ternary(3), Some(1));
    }

    #[test]
    fn test_ternary_column_type() {
        let table = ColumnarTable::new(
            "ternary_model",
            vec![("weights".to_string(), ColumnType::Ternary)],
        );
        assert_eq!(table.schema()[0].1, ColumnType::Ternary);
    }
}
