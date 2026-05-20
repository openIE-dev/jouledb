//! Join algorithm implementations — nested loop join, sort-merge join,
//! hash join, index nested loop join. Configurable join conditions,
//! row type (Vec<Value>), join statistics, multi-way join.
//!
//! Replaces hand-rolled join logic with reusable, tested implementations
//! of the four canonical join algorithms.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by join operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinError {
    /// Column index out of bounds.
    ColumnOutOfBounds { index: usize, row_len: usize },
    /// Incompatible row schemas.
    IncompatibleSchemas(String),
    /// Empty input.
    EmptyInput(String),
}

impl fmt::Display for JoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnOutOfBounds { index, row_len } => {
                write!(f, "column index {index} out of bounds (row len {row_len})")
            }
            Self::IncompatibleSchemas(msg) => write!(f, "incompatible schemas: {msg}"),
            Self::EmptyInput(msg) => write!(f, "empty input: {msg}"),
        }
    }
}

impl std::error::Error for JoinError {}

// ── Value type ───────────────────────────────────────────────────

/// A cell value in a row.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Null => {}
            Self::Int(v) => v.hash(state),
            Self::Float(v) => v.to_bits().hash(state),
            Self::Text(v) => v.hash(state),
            Self::Bool(v) => v.hash(state),
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp_value(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp_value(other)
    }
}

impl Value {
    fn cmp_value(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::Null, Self::Null) => std::cmp::Ordering::Equal,
            (Self::Null, _) => std::cmp::Ordering::Less,
            (_, Self::Null) => std::cmp::Ordering::Greater,
            (Self::Int(a), Self::Int(b)) => a.cmp(b),
            (Self::Float(a), Self::Float(b)) => a.total_cmp(b),
            (Self::Int(a), Self::Float(b)) => (*a as f64).total_cmp(b),
            (Self::Float(a), Self::Int(b)) => a.total_cmp(&(*b as f64)),
            (Self::Text(a), Self::Text(b)) => a.cmp(b),
            (Self::Bool(a), Self::Bool(b)) => a.cmp(b),
            (Self::Int(_), _) => std::cmp::Ordering::Less,
            (_, Self::Int(_)) => std::cmp::Ordering::Greater,
            (Self::Float(_), _) => std::cmp::Ordering::Less,
            (_, Self::Float(_)) => std::cmp::Ordering::Greater,
            (Self::Text(_), _) => std::cmp::Ordering::Less,
            (_, Self::Text(_)) => std::cmp::Ordering::Greater,
        }
    }
}

/// A row is a vector of values.
pub type Row = Vec<Value>;

// ── Join condition ───────────────────────────────────────────────

/// A join condition specifying which columns to compare.
#[derive(Debug, Clone)]
pub struct JoinCondition {
    /// Column index in the left relation.
    pub left_col: usize,
    /// Column index in the right relation.
    pub right_col: usize,
}

impl JoinCondition {
    pub fn new(left_col: usize, right_col: usize) -> Self {
        Self { left_col, right_col }
    }

    fn matches(&self, left: &Row, right: &Row) -> bool {
        if self.left_col >= left.len() || self.right_col >= right.len() {
            return false;
        }
        left[self.left_col] == right[self.right_col]
    }
}

// ── Join statistics ──────────────────────────────────────────────

/// Statistics from a join operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinStats {
    pub left_rows: usize,
    pub right_rows: usize,
    pub result_rows: usize,
    pub comparisons: u64,
    pub algorithm: String,
}

// ── Join result ──────────────────────────────────────────────────

/// Result of a join operation: the rows and statistics.
#[derive(Debug, Clone)]
pub struct JoinResult {
    pub rows: Vec<Row>,
    pub stats: JoinStats,
}

// ── Concatenation ────────────────────────────────────────────────

fn concat_rows(left: &Row, right: &Row) -> Row {
    let mut row = left.clone();
    row.extend(right.iter().cloned());
    row
}

// ── Nested Loop Join ─────────────────────────────────────────────

/// Nested loop join: O(n*m) comparison.
pub fn nested_loop_join(
    left: &[Row],
    right: &[Row],
    condition: &JoinCondition,
) -> Result<JoinResult, JoinError> {
    validate_condition(condition, left, right)?;
    let mut results = Vec::new();
    let mut comparisons = 0u64;

    for l_row in left {
        for r_row in right {
            comparisons += 1;
            if condition.matches(l_row, r_row) {
                results.push(concat_rows(l_row, r_row));
            }
        }
    }

    Ok(JoinResult {
        stats: JoinStats {
            left_rows: left.len(),
            right_rows: right.len(),
            result_rows: results.len(),
            comparisons,
            algorithm: "nested_loop".into(),
        },
        rows: results,
    })
}

// ── Sort-Merge Join ──────────────────────────────────────────────

/// Sort-merge join: O(n log n + m log m) sort then O(n+m) merge.
pub fn sort_merge_join(
    left: &[Row],
    right: &[Row],
    condition: &JoinCondition,
) -> Result<JoinResult, JoinError> {
    validate_condition(condition, left, right)?;
    let lc = condition.left_col;
    let rc = condition.right_col;

    let mut left_sorted: Vec<Row> = left.to_vec();
    let mut right_sorted: Vec<Row> = right.to_vec();
    left_sorted.sort_by(|a, b| a[lc].cmp(&b[lc]));
    right_sorted.sort_by(|a, b| a[rc].cmp(&b[rc]));

    let mut results = Vec::new();
    let mut comparisons = 0u64;
    let mut li = 0;
    let mut ri = 0;

    while li < left_sorted.len() && ri < right_sorted.len() {
        comparisons += 1;
        let cmp = left_sorted[li][lc].cmp(&right_sorted[ri][rc]);
        match cmp {
            std::cmp::Ordering::Less => li += 1,
            std::cmp::Ordering::Greater => ri += 1,
            std::cmp::Ordering::Equal => {
                // Collect all matching rows from both sides.
                let key = left_sorted[li][lc].clone();
                let li_start = li;
                while li < left_sorted.len() && left_sorted[li][lc] == key {
                    li += 1;
                }
                let ri_start = ri;
                while ri < right_sorted.len() && right_sorted[ri][rc] == key {
                    ri += 1;
                }
                for l_idx in li_start..li {
                    for r_idx in ri_start..ri {
                        comparisons += 1;
                        results.push(concat_rows(&left_sorted[l_idx], &right_sorted[r_idx]));
                    }
                }
            }
        }
    }

    Ok(JoinResult {
        stats: JoinStats {
            left_rows: left.len(),
            right_rows: right.len(),
            result_rows: results.len(),
            comparisons,
            algorithm: "sort_merge".into(),
        },
        rows: results,
    })
}

// ── Hash Join ────────────────────────────────────────────────────

/// Hash join: O(n+m) with hash table on the smaller relation.
pub fn hash_join(
    left: &[Row],
    right: &[Row],
    condition: &JoinCondition,
) -> Result<JoinResult, JoinError> {
    validate_condition(condition, left, right)?;
    let lc = condition.left_col;
    let rc = condition.right_col;

    // Build phase: hash the left (build) side.
    let mut hash_table: HashMap<Value, Vec<usize>> = HashMap::new();
    for (i, row) in left.iter().enumerate() {
        hash_table
            .entry(row[lc].clone())
            .or_default()
            .push(i);
    }

    // Probe phase: scan right side and probe the hash table.
    let mut results = Vec::new();
    let mut comparisons = 0u64;

    for r_row in right {
        let probe_key = &r_row[rc];
        comparisons += 1;
        if let Some(indices) = hash_table.get(probe_key) {
            for &li in indices {
                results.push(concat_rows(&left[li], r_row));
            }
        }
    }

    Ok(JoinResult {
        stats: JoinStats {
            left_rows: left.len(),
            right_rows: right.len(),
            result_rows: results.len(),
            comparisons,
            algorithm: "hash".into(),
        },
        rows: results,
    })
}

// ── Index Nested Loop Join ───────────────────────────────────────

/// Index for the right relation — a simple hash map on the join column.
pub struct HashLookupIndex {
    index: HashMap<Value, Vec<usize>>,
}

impl HashLookupIndex {
    /// Build an index on the given column of the relation.
    pub fn build(relation: &[Row], col: usize) -> Result<Self, JoinError> {
        if !relation.is_empty() && col >= relation[0].len() {
            return Err(JoinError::ColumnOutOfBounds {
                index: col,
                row_len: relation[0].len(),
            });
        }
        let mut index: HashMap<Value, Vec<usize>> = HashMap::new();
        for (i, row) in relation.iter().enumerate() {
            if col < row.len() {
                index.entry(row[col].clone()).or_default().push(i);
            }
        }
        Ok(Self { index })
    }

    /// Look up row indices matching a key.
    pub fn lookup(&self, key: &Value) -> &[usize] {
        self.index.get(key).map_or(&[], |v| v.as_slice())
    }
}

/// Index nested loop join: uses a pre-built index on the right relation.
pub fn index_nested_loop_join(
    left: &[Row],
    right: &[Row],
    condition: &JoinCondition,
    right_index: &HashLookupIndex,
) -> Result<JoinResult, JoinError> {
    validate_condition(condition, left, right)?;
    let lc = condition.left_col;
    let mut results = Vec::new();
    let mut comparisons = 0u64;

    for l_row in left {
        let key = &l_row[lc];
        let indices = right_index.lookup(key);
        comparisons += 1;
        for &ri in indices {
            results.push(concat_rows(l_row, &right[ri]));
        }
    }

    Ok(JoinResult {
        stats: JoinStats {
            left_rows: left.len(),
            right_rows: right.len(),
            result_rows: results.len(),
            comparisons,
            algorithm: "index_nested_loop".into(),
        },
        rows: results,
    })
}

// ── Multi-way join ───────────────────────────────────────────────

/// A step in a multi-way join pipeline.
#[derive(Debug, Clone)]
pub struct JoinStep {
    /// The right relation to join.
    pub right: Vec<Row>,
    /// Join condition (left_col refers to current accumulated result).
    pub condition: JoinCondition,
    /// Which algorithm to use.
    pub algorithm: JoinAlgorithm,
}

/// Which join algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinAlgorithm {
    NestedLoop,
    SortMerge,
    Hash,
}

/// Perform a multi-way join: starting from `base`, apply each step in order.
pub fn multi_way_join(
    base: &[Row],
    steps: &[JoinStep],
) -> Result<JoinResult, JoinError> {
    let mut current = base.to_vec();
    let mut total_comparisons = 0u64;

    for step in steps {
        let result = match step.algorithm {
            JoinAlgorithm::NestedLoop => {
                nested_loop_join(&current, &step.right, &step.condition)?
            }
            JoinAlgorithm::SortMerge => {
                sort_merge_join(&current, &step.right, &step.condition)?
            }
            JoinAlgorithm::Hash => {
                hash_join(&current, &step.right, &step.condition)?
            }
        };
        total_comparisons += result.stats.comparisons;
        current = result.rows;
    }

    Ok(JoinResult {
        stats: JoinStats {
            left_rows: base.len(),
            right_rows: steps.iter().map(|s| s.right.len()).sum(),
            result_rows: current.len(),
            comparisons: total_comparisons,
            algorithm: "multi_way".into(),
        },
        rows: current,
    })
}

// ── Validation ───────────────────────────────────────────────────

fn validate_condition(
    condition: &JoinCondition,
    left: &[Row],
    right: &[Row],
) -> Result<(), JoinError> {
    if let Some(first) = left.first() {
        if condition.left_col >= first.len() {
            return Err(JoinError::ColumnOutOfBounds {
                index: condition.left_col,
                row_len: first.len(),
            });
        }
    }
    if let Some(first) = right.first() {
        if condition.right_col >= first.len() {
            return Err(JoinError::ColumnOutOfBounds {
                index: condition.right_col,
                row_len: first.len(),
            });
        }
    }
    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn employees() -> Vec<Row> {
        vec![
            vec![Value::Int(1), Value::Text("Alice".into()), Value::Int(10)],
            vec![Value::Int(2), Value::Text("Bob".into()), Value::Int(20)],
            vec![Value::Int(3), Value::Text("Carol".into()), Value::Int(10)],
            vec![Value::Int(4), Value::Text("Dave".into()), Value::Int(30)],
        ]
    }

    fn departments() -> Vec<Row> {
        vec![
            vec![Value::Int(10), Value::Text("Engineering".into())],
            vec![Value::Int(20), Value::Text("Marketing".into())],
            vec![Value::Int(30), Value::Text("Sales".into())],
        ]
    }

    fn cond() -> JoinCondition {
        JoinCondition::new(2, 0) // employees.dept_id = departments.id
    }

    #[test]
    fn nested_loop_basic() {
        let result = nested_loop_join(&employees(), &departments(), &cond()).unwrap();
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.stats.algorithm, "nested_loop");
    }

    #[test]
    fn sort_merge_basic() {
        let result = sort_merge_join(&employees(), &departments(), &cond()).unwrap();
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.stats.algorithm, "sort_merge");
    }

    #[test]
    fn hash_join_basic() {
        let result = hash_join(&employees(), &departments(), &cond()).unwrap();
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.stats.algorithm, "hash");
    }

    #[test]
    fn index_nested_loop_basic() {
        let right = departments();
        let idx = HashLookupIndex::build(&right, 0).unwrap();
        let result =
            index_nested_loop_join(&employees(), &right, &cond(), &idx).unwrap();
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.stats.algorithm, "index_nested_loop");
    }

    #[test]
    fn all_algorithms_same_result() {
        let left = employees();
        let right = departments();
        let c = cond();
        let nl = nested_loop_join(&left, &right, &c).unwrap();
        let sm = sort_merge_join(&left, &right, &c).unwrap();
        let hj = hash_join(&left, &right, &c).unwrap();
        let idx = HashLookupIndex::build(&right, 0).unwrap();
        let inl = index_nested_loop_join(&left, &right, &c, &idx).unwrap();

        assert_eq!(nl.rows.len(), sm.rows.len());
        assert_eq!(sm.rows.len(), hj.rows.len());
        assert_eq!(hj.rows.len(), inl.rows.len());
    }

    #[test]
    fn empty_left_produces_empty() {
        let result = nested_loop_join(&[], &departments(), &cond()).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn empty_right_produces_empty() {
        let result = hash_join(&employees(), &[], &cond()).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn no_matches_produces_empty() {
        let left = vec![vec![Value::Int(1), Value::Text("X".into()), Value::Int(999)]];
        let result = nested_loop_join(&left, &departments(), &cond()).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn duplicate_keys_produce_cartesian() {
        let left = vec![
            vec![Value::Int(1), Value::Text("A".into()), Value::Int(10)],
            vec![Value::Int(2), Value::Text("B".into()), Value::Int(10)],
        ];
        let right = vec![
            vec![Value::Int(10), Value::Text("Eng1".into())],
            vec![Value::Int(10), Value::Text("Eng2".into())],
        ];
        let result = hash_join(&left, &right, &cond()).unwrap();
        // 2 left * 2 right = 4
        assert_eq!(result.rows.len(), 4);
    }

    #[test]
    fn column_out_of_bounds() {
        let bad_cond = JoinCondition::new(99, 0);
        let err = nested_loop_join(&employees(), &departments(), &bad_cond).unwrap_err();
        assert!(matches!(err, JoinError::ColumnOutOfBounds { .. }));
    }

    #[test]
    fn row_width_is_sum() {
        let result = hash_join(&employees(), &departments(), &cond()).unwrap();
        for row in &result.rows {
            assert_eq!(row.len(), 5); // 3 + 2
        }
    }

    #[test]
    fn stats_track_comparisons() {
        let result = nested_loop_join(&employees(), &departments(), &cond()).unwrap();
        assert_eq!(result.stats.comparisons, 12); // 4 * 3
    }

    #[test]
    fn hash_join_fewer_comparisons_than_nested() {
        let left = employees();
        let right = departments();
        let c = cond();
        let nl = nested_loop_join(&left, &right, &c).unwrap();
        let hj = hash_join(&left, &right, &c).unwrap();
        assert!(hj.stats.comparisons <= nl.stats.comparisons);
    }

    #[test]
    fn self_join() {
        let data = vec![
            vec![Value::Int(1), Value::Int(1)],
            vec![Value::Int(2), Value::Int(2)],
        ];
        let c = JoinCondition::new(0, 1);
        let result = hash_join(&data, &data, &c).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn multi_way_join_two_steps() {
        let base = employees();
        let depts = departments();
        let locations = vec![
            vec![Value::Int(10), Value::Text("NYC".into())],
            vec![Value::Int(20), Value::Text("SF".into())],
            vec![Value::Int(30), Value::Text("LA".into())],
        ];
        let steps = vec![
            JoinStep {
                right: depts,
                condition: JoinCondition::new(2, 0),
                algorithm: JoinAlgorithm::Hash,
            },
            JoinStep {
                right: locations,
                condition: JoinCondition::new(3, 0), // dept id from the joined result
                algorithm: JoinAlgorithm::Hash,
            },
        ];
        let result = multi_way_join(&base, &steps).unwrap();
        assert_eq!(result.rows.len(), 4);
        assert_eq!(result.stats.algorithm, "multi_way");
    }

    #[test]
    fn sort_merge_handles_many_duplicates() {
        let left: Vec<Row> = (0..10)
            .map(|i| vec![Value::Int(i), Value::Int(1)])
            .collect();
        let right: Vec<Row> = (0..10)
            .map(|i| vec![Value::Int(1), Value::Int(i)])
            .collect();
        let c = JoinCondition::new(1, 0);
        let result = sort_merge_join(&left, &right, &c).unwrap();
        assert_eq!(result.rows.len(), 100); // 10 * 10
    }

    #[test]
    fn hash_lookup_index_build_and_query() {
        let data = vec![
            vec![Value::Int(1), Value::Text("A".into())],
            vec![Value::Int(2), Value::Text("B".into())],
            vec![Value::Int(1), Value::Text("C".into())],
        ];
        let idx = HashLookupIndex::build(&data, 0).unwrap();
        let matches = idx.lookup(&Value::Int(1));
        assert_eq!(matches.len(), 2);
        let empty = idx.lookup(&Value::Int(99));
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn multi_way_empty_steps() {
        let base = employees();
        let result = multi_way_join(&base, &[]).unwrap();
        assert_eq!(result.rows.len(), base.len());
    }

    #[test]
    fn float_keys_join() {
        let left = vec![
            vec![Value::Float(1.0), Value::Text("A".into())],
            vec![Value::Float(2.0), Value::Text("B".into())],
        ];
        let right = vec![
            vec![Value::Float(1.0), Value::Text("X".into())],
            vec![Value::Float(3.0), Value::Text("Y".into())],
        ];
        let c = JoinCondition::new(0, 0);
        let result = hash_join(&left, &right, &c).unwrap();
        assert_eq!(result.rows.len(), 1);
    }
}
