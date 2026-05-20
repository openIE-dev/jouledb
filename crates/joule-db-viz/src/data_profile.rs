//! Data profiling: analyze result set shape and column characteristics.

use serde::{Deserialize, Serialize};

use crate::column_classifier::classify_column;
use crate::hint::SemanticType;

/// Profile of query result data used for visualization inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataProfile {
    /// Total number of rows.
    pub row_count: usize,
    /// Number of columns.
    pub col_count: usize,
    /// Per-column profiles.
    pub columns: Vec<ColumnProfile>,
    /// Number of timestamp columns detected.
    pub timestamp_count: usize,
    /// Number of numeric columns detected.
    pub numeric_count: usize,
    /// Number of categorical columns detected.
    pub categorical_count: usize,
    /// Number of geographic columns detected.
    pub geo_count: usize,
    /// Whether the data appears to be a time series.
    pub is_time_series: bool,
    /// Whether results come from a GROUP BY query.
    pub has_group_by: bool,
    /// Whether results use aggregation.
    pub has_aggregates: bool,
    /// Pearson correlation pairs between numeric columns (|r| > 0.3 only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub correlation_pairs: Vec<CorrelationPair>,
}

/// Pearson correlation between two numeric columns.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorrelationPair {
    /// First column index.
    pub col_a: usize,
    /// Second column index.
    pub col_b: usize,
    /// Pearson correlation coefficient (-1.0 to 1.0).
    pub r: f64,
}

/// Per-column statistics and classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnProfile {
    /// Column name.
    pub name: String,
    /// Column index in result set.
    pub index: usize,
    /// Inferred semantic type.
    pub semantic_type: SemanticType,
    /// Number of distinct values (sampled).
    pub distinct_count: usize,
    /// Number of null values.
    pub null_count: usize,
    /// Whether values are monotonically increasing/decreasing.
    pub is_monotonic: bool,
    /// Minimum numeric value (if numeric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_value: Option<f64>,
    /// Maximum numeric value (if numeric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_value: Option<f64>,
    /// Median value (if numeric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median: Option<f64>,
    /// First quartile / 25th percentile (if numeric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q1: Option<f64>,
    /// Third quartile / 75th percentile (if numeric).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q3: Option<f64>,
    /// Skewness — asymmetry of the distribution (if numeric, ≥3 values).
    /// Positive = right-skewed, negative = left-skewed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skewness: Option<f64>,
    /// Uniqueness ratio: distinct_count / row_count (0.0 to 1.0).
    pub uniqueness_ratio: f64,
}

/// Build a DataProfile from column names, sample rows, and query metadata.
pub fn build_profile(
    columns: &[String],
    sample_rows: &[Vec<serde_json::Value>],
    has_group_by: bool,
    has_aggregates: bool,
) -> DataProfile {
    let col_count = columns.len();
    let row_count = sample_rows.len();

    // Collect numeric vectors per column (for correlation computation later).
    let mut numeric_vecs: Vec<Option<Vec<f64>>> = Vec::with_capacity(col_count);
    let mut col_profiles = Vec::with_capacity(col_count);

    for (i, col_name) in columns.iter().enumerate() {
        let values: Vec<&serde_json::Value> =
            sample_rows.iter().filter_map(|row| row.get(i)).collect();

        let semantic_type = classify_column(col_name, &values);
        let distinct_count = count_distinct(&values);
        let null_count = values.iter().filter(|v| v.is_null()).count();
        let is_monotonic = check_monotonic(&values);
        let (min_value, max_value) = numeric_range(&values);

        // Extract sorted numeric values for quartile/skewness computation.
        let mut nums: Vec<f64> = values.iter().filter_map(|v| extract_f64(v)).collect();
        let (median, q1, q3, skewness) = if nums.len() >= 2 {
            nums.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = percentile_sorted(&nums, 0.5);
            let q1_val = percentile_sorted(&nums, 0.25);
            let q3_val = percentile_sorted(&nums, 0.75);
            let skew = if nums.len() >= 3 {
                compute_skewness(&nums)
            } else {
                None
            };
            (Some(med), Some(q1_val), Some(q3_val), skew)
        } else {
            (None, None, None, None)
        };

        let uniqueness_ratio = if row_count > 0 {
            (distinct_count as f64 / row_count as f64).min(1.0)
        } else {
            0.0
        };

        // Store numeric vector for correlation computation.
        if !nums.is_empty() && is_numeric_type(semantic_type) {
            // Re-extract in original order (not sorted) for correlation.
            let ordered: Vec<f64> = values.iter().filter_map(|v| extract_f64(v)).collect();
            numeric_vecs.push(Some(ordered));
        } else {
            numeric_vecs.push(None);
        }

        col_profiles.push(ColumnProfile {
            name: col_name.clone(),
            index: i,
            semantic_type,
            distinct_count,
            null_count,
            is_monotonic,
            min_value,
            max_value,
            median,
            q1,
            q3,
            skewness,
            uniqueness_ratio,
        });
    }

    // Compute Pearson correlations between numeric column pairs.
    let correlation_pairs = compute_correlations(&col_profiles, &numeric_vecs);

    let timestamp_count = col_profiles
        .iter()
        .filter(|c| {
            matches!(
                c.semantic_type,
                SemanticType::Timestamp | SemanticType::Date
            )
        })
        .count();
    let numeric_count = col_profiles
        .iter()
        .filter(|c| is_numeric_type(c.semantic_type))
        .count();
    let categorical_count = col_profiles
        .iter()
        .filter(|c| {
            matches!(
                c.semantic_type,
                SemanticType::Categorical | SemanticType::Boolean
            )
        })
        .count();
    let geo_count = col_profiles
        .iter()
        .filter(|c| {
            matches!(
                c.semantic_type,
                SemanticType::GeoLatitude | SemanticType::GeoLongitude
            )
        })
        .count();

    let is_time_series = timestamp_count >= 1
        && numeric_count >= 1
        && col_profiles.iter().any(|c| {
            matches!(
                c.semantic_type,
                SemanticType::Timestamp | SemanticType::Date
            ) && c.is_monotonic
        });

    DataProfile {
        row_count,
        col_count,
        columns: col_profiles,
        timestamp_count,
        numeric_count,
        categorical_count,
        geo_count,
        is_time_series,
        has_group_by,
        has_aggregates,
        correlation_pairs,
    }
}

/// Check if a semantic type is numeric.
fn is_numeric_type(st: SemanticType) -> bool {
    matches!(
        st,
        SemanticType::NumericContinuous
            | SemanticType::NumericDiscrete
            | SemanticType::Currency
            | SemanticType::Percentage
    )
}

/// Compute percentile from a sorted slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = idx - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Compute skewness (Fisher's definition) from a set of values.
/// Returns None if variance is zero.
fn compute_skewness(values: &[f64]) -> Option<f64> {
    let n = values.len() as f64;
    if n < 3.0 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n;
    let m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if m2 < f64::EPSILON {
        return None; // zero variance
    }
    let m3: f64 = values.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    let std_dev = m2.sqrt();
    Some(m3 / std_dev.powi(3))
}

/// Compute Pearson correlations between all numeric column pairs.
/// Only returns pairs with |r| > 0.3 to keep the output compact.
fn compute_correlations(
    profiles: &[ColumnProfile],
    numeric_vecs: &[Option<Vec<f64>>],
) -> Vec<CorrelationPair> {
    let mut pairs = Vec::new();
    for i in 0..profiles.len() {
        for j in (i + 1)..profiles.len() {
            if let (Some(a), Some(b)) = (&numeric_vecs[i], &numeric_vecs[j]) {
                if let Some(r) = pearson_r(a, b) {
                    if r.abs() > 0.3 {
                        pairs.push(CorrelationPair {
                            col_a: i,
                            col_b: j,
                            r,
                        });
                    }
                }
            }
        }
    }
    pairs
}

/// Compute Pearson correlation coefficient between two equal-length vectors.
fn pearson_r(a: &[f64], b: &[f64]) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 2 {
        return None;
    }
    let mean_a = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b = b[..n].iter().sum::<f64>() / n as f64;

    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        sum_ab += da * db;
        sum_a2 += da * da;
        sum_b2 += db * db;
    }

    let denom = (sum_a2 * sum_b2).sqrt();
    if denom < f64::EPSILON {
        return None;
    }
    Some(sum_ab / denom)
}

/// Count distinct non-null values in a sample.
fn count_distinct(values: &[&serde_json::Value]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for v in values {
        if !v.is_null() {
            seen.insert(v.to_string());
        }
    }
    seen.len()
}

/// Check if values are monotonically ordered (numeric or string/timestamp).
fn check_monotonic(values: &[&serde_json::Value]) -> bool {
    // Try numeric monotonicity first
    let nums: Vec<f64> = values.iter().filter_map(|v| extract_f64(v)).collect();

    if nums.len() >= 2 {
        let increasing = nums.windows(2).all(|w| w[1] >= w[0]);
        let decreasing = nums.windows(2).all(|w| w[1] <= w[0]);
        return increasing || decreasing;
    }

    // Fall back to string comparison (works for ISO 8601 timestamps)
    let strings: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();

    if strings.len() >= 2 {
        let increasing = strings.windows(2).all(|w| w[1] >= w[0]);
        let decreasing = strings.windows(2).all(|w| w[1] <= w[0]);
        return increasing || decreasing;
    }

    false
}

/// Extract min and max numeric values.
fn numeric_range(values: &[&serde_json::Value]) -> (Option<f64>, Option<f64>) {
    let nums: Vec<f64> = values.iter().filter_map(|v| extract_f64(v)).collect();

    if nums.is_empty() {
        return (None, None);
    }

    let min = nums.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (Some(min), Some(max))
}

/// Try to extract an f64 from a JSON value.
fn extract_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}
