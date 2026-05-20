//! Aggregation engine — GROUP BY with hash aggregation, aggregate functions
//! (count/sum/avg/min/max/count_distinct), HAVING filter, multi-column
//! grouping, running aggregates, rollup/cube.
//!
//! Replaces ad-hoc aggregation logic with a composable, tested engine.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by aggregation operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggError {
    /// Column index out of bounds.
    ColumnOutOfBounds { index: usize, row_len: usize },
    /// Invalid aggregate function for the data type.
    TypeMismatch(String),
    /// Empty input when at least one row is required.
    EmptyInput,
}

impl fmt::Display for AggError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnOutOfBounds { index, row_len } => {
                write!(f, "column {index} out of bounds (row len {row_len})")
            }
            Self::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Self::EmptyInput => write!(f, "empty input"),
        }
    }
}

impl std::error::Error for AggError {}

// ── Value type ───────────────────────────────────────────────────

/// A cell value.
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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "NULL"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl Value {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(v) => Some(*v as f64),
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }
}

/// A row is a vector of values.
pub type Row = Vec<Value>;

// ── Aggregate functions ──────────────────────────────────────────

/// An aggregate function to apply to a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    CountDistinct,
}

impl fmt::Display for AggFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count => write!(f, "COUNT"),
            Self::Sum => write!(f, "SUM"),
            Self::Avg => write!(f, "AVG"),
            Self::Min => write!(f, "MIN"),
            Self::Max => write!(f, "MAX"),
            Self::CountDistinct => write!(f, "COUNT_DISTINCT"),
        }
    }
}

/// An aggregate specification: function + column index.
#[derive(Debug, Clone)]
pub struct AggSpec {
    pub func: AggFunc,
    pub column: usize,
    /// Optional alias for the result column.
    pub alias: Option<String>,
}

impl AggSpec {
    pub fn new(func: AggFunc, column: usize) -> Self {
        Self {
            func,
            column,
            alias: None,
        }
    }

    pub fn with_alias(mut self, alias: &str) -> Self {
        self.alias = Some(alias.to_string());
        self
    }
}

// ── Accumulator ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Accumulator {
    func: AggFunc,
    count: u64,
    sum: f64,
    min_val: Option<Value>,
    max_val: Option<Value>,
    distinct: HashSet<Value>,
}

impl Accumulator {
    fn new(func: AggFunc) -> Self {
        Self {
            func,
            count: 0,
            sum: 0.0,
            min_val: None,
            max_val: None,
            distinct: HashSet::new(),
        }
    }

    fn accumulate(&mut self, value: &Value) {
        if matches!(value, Value::Null) {
            return;
        }
        self.count += 1;
        if let Some(v) = value.as_f64() {
            self.sum += v;
        }

        // Min
        let should_replace_min = match &self.min_val {
            None => true,
            Some(current) => self.value_less_than(value, current),
        };
        if should_replace_min {
            self.min_val = Some(value.clone());
        }

        // Max
        let should_replace_max = match &self.max_val {
            None => true,
            Some(current) => self.value_less_than(current, value),
        };
        if should_replace_max {
            self.max_val = Some(value.clone());
        }

        // Distinct
        if self.func == AggFunc::CountDistinct {
            self.distinct.insert(value.clone());
        }
    }

    fn value_less_than(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x < y,
            (Value::Float(x), Value::Float(y)) => x < y,
            (Value::Int(x), Value::Float(y)) => (*x as f64) < *y,
            (Value::Float(x), Value::Int(y)) => *x < (*y as f64),
            (Value::Text(x), Value::Text(y)) => x < y,
            _ => false,
        }
    }

    fn result(&self) -> Value {
        match self.func {
            AggFunc::Count => Value::Int(self.count as i64),
            AggFunc::Sum => Value::Float(self.sum),
            AggFunc::Avg => {
                if self.count == 0 {
                    Value::Null
                } else {
                    Value::Float(self.sum / self.count as f64)
                }
            }
            AggFunc::Min => self.min_val.clone().unwrap_or(Value::Null),
            AggFunc::Max => self.max_val.clone().unwrap_or(Value::Null),
            AggFunc::CountDistinct => Value::Int(self.distinct.len() as i64),
        }
    }
}

// ── HAVING filter ────────────────────────────────────────────────

/// A HAVING condition: applied to aggregated results.
#[derive(Debug, Clone)]
pub enum HavingFilter {
    /// Aggregate result at position `agg_index` (in the agg specs) compared to a value.
    Compare {
        agg_index: usize,
        op: CmpOp,
        value: Value,
    },
    And(Box<HavingFilter>, Box<HavingFilter>),
    Or(Box<HavingFilter>, Box<HavingFilter>),
}

/// Comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl HavingFilter {
    fn evaluate(&self, agg_results: &[Value]) -> bool {
        match self {
            Self::Compare { agg_index, op, value } => {
                if *agg_index >= agg_results.len() {
                    return false;
                }
                cmp_values(&agg_results[*agg_index], value, *op)
            }
            Self::And(a, b) => a.evaluate(agg_results) && b.evaluate(agg_results),
            Self::Or(a, b) => a.evaluate(agg_results) || b.evaluate(agg_results),
        }
    }
}

fn cmp_values(left: &Value, right: &Value, op: CmpOp) -> bool {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => cmp_op(*a, *b, op),
        (Value::Float(a), Value::Float(b)) => cmp_op_f64(*a, *b, op),
        (Value::Int(a), Value::Float(b)) => cmp_op_f64(*a as f64, *b, op),
        (Value::Float(a), Value::Int(b)) => cmp_op_f64(*a, *b as f64, op),
        (Value::Text(a), Value::Text(b)) => cmp_op_ord(a, b, op),
        _ => false,
    }
}

fn cmp_op<T: PartialOrd>(a: T, b: T, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn cmp_op_f64(a: f64, b: f64, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => (a - b).abs() < f64::EPSILON,
        CmpOp::Ne => (a - b).abs() >= f64::EPSILON,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn cmp_op_ord<T: Ord>(a: &T, b: &T, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

// ── Group By result ──────────────────────────────────────────────

/// A single group in the aggregation result.
#[derive(Debug, Clone)]
pub struct GroupResult {
    /// The group key values.
    pub key: Vec<Value>,
    /// The aggregate results.
    pub aggregates: Vec<Value>,
}

/// Result of an aggregation.
#[derive(Debug, Clone)]
pub struct AggResult {
    /// All groups.
    pub groups: Vec<GroupResult>,
    /// Column indices used for grouping.
    pub group_columns: Vec<usize>,
    /// Aggregate specifications used.
    pub agg_specs: Vec<AggSpec>,
}

// ── Aggregation engine ───────────────────────────────────────────

/// Perform GROUP BY aggregation.
pub fn group_by(
    rows: &[Row],
    group_columns: &[usize],
    agg_specs: &[AggSpec],
    having: Option<&HavingFilter>,
) -> Result<AggResult, AggError> {
    // Validate columns.
    if let Some(first) = rows.first() {
        for &col in group_columns {
            if col >= first.len() {
                return Err(AggError::ColumnOutOfBounds {
                    index: col,
                    row_len: first.len(),
                });
            }
        }
        for spec in agg_specs {
            if spec.column >= first.len() {
                return Err(AggError::ColumnOutOfBounds {
                    index: spec.column,
                    row_len: first.len(),
                });
            }
        }
    }

    // Build groups.
    let mut groups: HashMap<Vec<Value>, Vec<Accumulator>> = HashMap::new();
    // Track insertion order.
    let mut group_order: Vec<Vec<Value>> = Vec::new();

    for row in rows {
        let key: Vec<Value> = group_columns.iter().map(|c| row[*c].clone()).collect();
        let accumulators = groups.entry(key.clone()).or_insert_with(|| {
            group_order.push(key.clone());
            agg_specs.iter().map(|s| Accumulator::new(s.func)).collect()
        });
        for (i, spec) in agg_specs.iter().enumerate() {
            accumulators[i].accumulate(&row[spec.column]);
        }
    }

    // Collect results.
    let mut result_groups = Vec::new();
    for key in &group_order {
        let accumulators = groups.get(key).unwrap();
        let agg_results: Vec<Value> = accumulators.iter().map(|a| a.result()).collect();

        // Apply HAVING filter.
        if let Some(filter) = having {
            if !filter.evaluate(&agg_results) {
                continue;
            }
        }
        result_groups.push(GroupResult {
            key: key.clone(),
            aggregates: agg_results,
        });
    }

    Ok(AggResult {
        groups: result_groups,
        group_columns: group_columns.to_vec(),
        agg_specs: agg_specs.to_vec(),
    })
}

/// Compute aggregates over the entire input (no GROUP BY).
pub fn aggregate_all(
    rows: &[Row],
    agg_specs: &[AggSpec],
) -> Result<Vec<Value>, AggError> {
    if let Some(first) = rows.first() {
        for spec in agg_specs {
            if spec.column >= first.len() {
                return Err(AggError::ColumnOutOfBounds {
                    index: spec.column,
                    row_len: first.len(),
                });
            }
        }
    }

    let mut accumulators: Vec<Accumulator> =
        agg_specs.iter().map(|s| Accumulator::new(s.func)).collect();

    for row in rows {
        for (i, spec) in agg_specs.iter().enumerate() {
            accumulators[i].accumulate(&row[spec.column]);
        }
    }

    Ok(accumulators.iter().map(|a| a.result()).collect())
}

// ── Running aggregates ───────────────────────────────────────────

/// Compute running (cumulative) aggregates over a column.
/// Returns one aggregate value per row.
pub fn running_aggregate(
    rows: &[Row],
    func: AggFunc,
    column: usize,
) -> Result<Vec<Value>, AggError> {
    if let Some(first) = rows.first() {
        if column >= first.len() {
            return Err(AggError::ColumnOutOfBounds {
                index: column,
                row_len: first.len(),
            });
        }
    }

    let mut acc = Accumulator::new(func);
    let mut results = Vec::with_capacity(rows.len());

    for row in rows {
        acc.accumulate(&row[column]);
        results.push(acc.result());
    }

    Ok(results)
}

// ── Rollup ───────────────────────────────────────────────────────

/// Perform ROLLUP aggregation — produces sub-totals for each prefix of
/// group columns, plus a grand total.
pub fn rollup(
    rows: &[Row],
    group_columns: &[usize],
    agg_specs: &[AggSpec],
) -> Result<Vec<GroupResult>, AggError> {
    let mut all_results = Vec::new();

    // For each prefix length from full down to 0.
    for len in (0..=group_columns.len()).rev() {
        let prefix = &group_columns[..len];
        let result = group_by(rows, prefix, agg_specs, None)?;
        all_results.extend(result.groups);
    }

    Ok(all_results)
}

/// Perform CUBE aggregation — produces sub-totals for every combination
/// of group columns.
pub fn cube(
    rows: &[Row],
    group_columns: &[usize],
    agg_specs: &[AggSpec],
) -> Result<Vec<GroupResult>, AggError> {
    let n = group_columns.len();
    let mut all_results = Vec::new();

    // Iterate all subsets using bitmask.
    for mask in 0..(1u32 << n) {
        let subset: Vec<usize> = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| group_columns[i])
            .collect();
        let result = group_by(rows, &subset, agg_specs, None)?;
        all_results.extend(result.groups);
    }

    Ok(all_results)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> Vec<Row> {
        vec![
            vec![
                Value::Text("Engineering".into()),
                Value::Text("Alice".into()),
                Value::Int(100_000),
            ],
            vec![
                Value::Text("Engineering".into()),
                Value::Text("Bob".into()),
                Value::Int(110_000),
            ],
            vec![
                Value::Text("Marketing".into()),
                Value::Text("Carol".into()),
                Value::Int(90_000),
            ],
            vec![
                Value::Text("Marketing".into()),
                Value::Text("Dave".into()),
                Value::Int(85_000),
            ],
            vec![
                Value::Text("Sales".into()),
                Value::Text("Eve".into()),
                Value::Int(70_000),
            ],
        ]
    }

    #[test]
    fn count_by_department() {
        let result = group_by(
            &sample_data(),
            &[0],
            &[AggSpec::new(AggFunc::Count, 1)],
            None,
        )
        .unwrap();
        assert_eq!(result.groups.len(), 3);
        for g in &result.groups {
            let count = match &g.aggregates[0] {
                Value::Int(v) => *v,
                _ => panic!("expected Int"),
            };
            match g.key[0] {
                Value::Text(ref s) if s == "Engineering" => assert_eq!(count, 2),
                Value::Text(ref s) if s == "Marketing" => assert_eq!(count, 2),
                Value::Text(ref s) if s == "Sales" => assert_eq!(count, 1),
                _ => panic!("unexpected key"),
            }
        }
    }

    #[test]
    fn sum_by_department() {
        let result = group_by(
            &sample_data(),
            &[0],
            &[AggSpec::new(AggFunc::Sum, 2)],
            None,
        )
        .unwrap();
        for g in &result.groups {
            if g.key[0] == Value::Text("Engineering".into()) {
                assert_eq!(g.aggregates[0], Value::Float(210_000.0));
            }
        }
    }

    #[test]
    fn avg_overall() {
        let result = aggregate_all(
            &sample_data(),
            &[AggSpec::new(AggFunc::Avg, 2)],
        )
        .unwrap();
        let avg = match &result[0] {
            Value::Float(v) => *v,
            _ => panic!("expected Float"),
        };
        let expected = (100_000.0 + 110_000.0 + 90_000.0 + 85_000.0 + 70_000.0) / 5.0;
        assert!((avg - expected).abs() < 0.01);
    }

    #[test]
    fn min_max() {
        let result = aggregate_all(
            &sample_data(),
            &[
                AggSpec::new(AggFunc::Min, 2),
                AggSpec::new(AggFunc::Max, 2),
            ],
        )
        .unwrap();
        assert_eq!(result[0], Value::Int(70_000));
        assert_eq!(result[1], Value::Int(110_000));
    }

    #[test]
    fn count_distinct() {
        let data = vec![
            vec![Value::Int(1), Value::Text("A".into())],
            vec![Value::Int(2), Value::Text("A".into())],
            vec![Value::Int(3), Value::Text("B".into())],
        ];
        let result = aggregate_all(&data, &[AggSpec::new(AggFunc::CountDistinct, 1)]).unwrap();
        assert_eq!(result[0], Value::Int(2)); // "A" and "B"
    }

    #[test]
    fn having_filter() {
        let filter = HavingFilter::Compare {
            agg_index: 0,
            op: CmpOp::Ge,
            value: Value::Int(2),
        };
        let result = group_by(
            &sample_data(),
            &[0],
            &[AggSpec::new(AggFunc::Count, 1)],
            Some(&filter),
        )
        .unwrap();
        // Only departments with count >= 2: Engineering and Marketing.
        assert_eq!(result.groups.len(), 2);
    }

    #[test]
    fn having_and() {
        let filter = HavingFilter::And(
            Box::new(HavingFilter::Compare {
                agg_index: 0,
                op: CmpOp::Ge,
                value: Value::Int(2),
            }),
            Box::new(HavingFilter::Compare {
                agg_index: 1,
                op: CmpOp::Gt,
                value: Value::Float(200_000.0),
            }),
        );
        let result = group_by(
            &sample_data(),
            &[0],
            &[
                AggSpec::new(AggFunc::Count, 1),
                AggSpec::new(AggFunc::Sum, 2),
            ],
            Some(&filter),
        )
        .unwrap();
        // Engineering: count=2, sum=210000 > 200000.
        assert_eq!(result.groups.len(), 1);
    }

    #[test]
    fn multi_column_grouping() {
        let data = vec![
            vec![Value::Text("A".into()), Value::Int(1), Value::Int(10)],
            vec![Value::Text("A".into()), Value::Int(1), Value::Int(20)],
            vec![Value::Text("A".into()), Value::Int(2), Value::Int(30)],
            vec![Value::Text("B".into()), Value::Int(1), Value::Int(40)],
        ];
        let result = group_by(
            &data,
            &[0, 1],
            &[AggSpec::new(AggFunc::Sum, 2)],
            None,
        )
        .unwrap();
        assert_eq!(result.groups.len(), 3);
    }

    #[test]
    fn empty_input_produces_no_groups() {
        let result = group_by(
            &[],
            &[0],
            &[AggSpec::new(AggFunc::Count, 0)],
            None,
        )
        .unwrap();
        assert!(result.groups.is_empty());
    }

    #[test]
    fn column_out_of_bounds() {
        let err = group_by(
            &sample_data(),
            &[99],
            &[AggSpec::new(AggFunc::Count, 0)],
            None,
        )
        .unwrap_err();
        assert!(matches!(err, AggError::ColumnOutOfBounds { .. }));
    }

    #[test]
    fn running_sum() {
        let data: Vec<Row> = (1..=5).map(|i| vec![Value::Int(i)]).collect();
        let result = running_aggregate(&data, AggFunc::Sum, 0).unwrap();
        let expected = [1.0, 3.0, 6.0, 10.0, 15.0];
        for (i, val) in result.iter().enumerate() {
            match val {
                Value::Float(v) => assert!((v - expected[i]).abs() < 0.01),
                _ => panic!("expected Float"),
            }
        }
    }

    #[test]
    fn running_count() {
        let data: Vec<Row> = (0..4).map(|_| vec![Value::Int(1)]).collect();
        let result = running_aggregate(&data, AggFunc::Count, 0).unwrap();
        for (i, val) in result.iter().enumerate() {
            assert_eq!(*val, Value::Int((i + 1) as i64));
        }
    }

    #[test]
    fn rollup_produces_subtotals() {
        let data = vec![
            vec![Value::Text("A".into()), Value::Int(1), Value::Int(10)],
            vec![Value::Text("A".into()), Value::Int(2), Value::Int(20)],
            vec![Value::Text("B".into()), Value::Int(1), Value::Int(30)],
        ];
        let result = rollup(&data, &[0, 1], &[AggSpec::new(AggFunc::Sum, 2)]).unwrap();
        // 3 groups at (col0, col1), 2 at (col0), 1 grand total = 6.
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn cube_produces_all_combinations() {
        let data = vec![
            vec![Value::Text("A".into()), Value::Int(1), Value::Int(10)],
            vec![Value::Text("B".into()), Value::Int(2), Value::Int(20)],
        ];
        let result = cube(&data, &[0, 1], &[AggSpec::new(AggFunc::Sum, 2)]).unwrap();
        // 2^2 = 4 subsets: {}, {0}, {1}, {0,1}
        // {}: 1 grand total, {0}: 2, {1}: 2, {0,1}: 2 => 7
        assert!(result.len() >= 4);
    }

    #[test]
    fn null_values_skipped() {
        let data = vec![
            vec![Value::Text("A".into()), Value::Int(10)],
            vec![Value::Text("A".into()), Value::Null],
            vec![Value::Text("A".into()), Value::Int(20)],
        ];
        let result = group_by(
            &data,
            &[0],
            &[AggSpec::new(AggFunc::Count, 1), AggSpec::new(AggFunc::Sum, 1)],
            None,
        )
        .unwrap();
        // Count should skip null: 2, not 3.
        assert_eq!(result.groups[0].aggregates[0], Value::Int(2));
    }

    #[test]
    fn avg_empty_is_null() {
        let data: Vec<Row> = vec![vec![Value::Null]];
        let result = aggregate_all(&data, &[AggSpec::new(AggFunc::Avg, 0)]).unwrap();
        assert_eq!(result[0], Value::Null);
    }

    #[test]
    fn agg_spec_alias() {
        let spec = AggSpec::new(AggFunc::Count, 0).with_alias("total");
        assert_eq!(spec.alias.as_deref(), Some("total"));
    }
}
