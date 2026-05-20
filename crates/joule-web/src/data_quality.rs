//! Data quality metrics.
//!
//! Replaces Great Expectations, Deequ, and similar data quality frameworks with
//! a pure-Rust quality engine. Tracks completeness (non-null %), uniqueness
//! (distinct %), consistency (format match %), freshness (latest record age),
//! accuracy scoring, quality report card, and trend tracking over time.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors from data quality operations.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityError {
    /// No data to evaluate.
    EmptyDataset,
    /// Field not found.
    FieldNotFound(String),
    /// Invalid threshold.
    InvalidThreshold(String),
}

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDataset => write!(f, "cannot evaluate quality on empty dataset"),
            Self::FieldNotFound(field) => write!(f, "field not found: {field}"),
            Self::InvalidThreshold(msg) => write!(f, "invalid threshold: {msg}"),
        }
    }
}

impl std::error::Error for QualityError {}

// ── Row type ─────────────────────────────────────────────────────

/// A data row.
pub type Row = HashMap<String, serde_json::Value>;

// ── Quality dimension ────────────────────────────────────────────

/// A single quality dimension score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionScore {
    /// Dimension name.
    pub name: String,
    /// Score as a fraction [0.0, 1.0].
    pub score: f64,
    /// Number of records evaluated.
    pub records_evaluated: usize,
    /// Number of records that passed.
    pub records_passed: usize,
    /// Number of records that failed.
    pub records_failed: usize,
    /// Threshold for passing.
    pub threshold: f64,
    /// Whether the dimension meets the threshold.
    pub meets_threshold: bool,
}

impl DimensionScore {
    fn new(name: impl Into<String>, passed: usize, total: usize, threshold: f64) -> Self {
        let score = if total == 0 {
            1.0
        } else {
            passed as f64 / total as f64
        };
        Self {
            name: name.into(),
            score,
            records_evaluated: total,
            records_passed: passed,
            records_failed: total.saturating_sub(passed),
            threshold,
            meets_threshold: score >= threshold,
        }
    }
}

// ── Field quality ────────────────────────────────────────────────

/// Quality metrics for a single field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldQuality {
    /// Field name.
    pub field_name: String,
    /// Completeness: fraction of non-null values.
    pub completeness: f64,
    /// Uniqueness: fraction of distinct values.
    pub uniqueness: f64,
    /// Total values.
    pub total_count: usize,
    /// Null count.
    pub null_count: usize,
    /// Distinct value count.
    pub distinct_count: usize,
}

// ── Consistency rule ─────────────────────────────────────────────

/// A consistency rule that checks format compliance.
#[derive(Debug, Clone)]
pub struct ConsistencyRule {
    /// Rule name.
    pub name: String,
    /// Target field.
    pub field: String,
    /// Check function.
    check: ConsistencyCheck,
}

/// Type of consistency check.
#[derive(Debug, Clone)]
enum ConsistencyCheck {
    /// Value must contain the substring.
    Contains(String),
    /// Value must start with the prefix.
    StartsWith(String),
    /// Value must end with the suffix.
    EndsWith(String),
    /// String length must be in [min, max].
    LengthRange(usize, usize),
    /// Custom function.
    Custom(fn(&serde_json::Value) -> bool),
}

impl ConsistencyRule {
    /// Create a "contains" consistency rule.
    pub fn contains(
        name: impl Into<String>,
        field: impl Into<String>,
        substring: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            check: ConsistencyCheck::Contains(substring.into()),
        }
    }

    /// Create a "starts with" consistency rule.
    pub fn starts_with(
        name: impl Into<String>,
        field: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            check: ConsistencyCheck::StartsWith(prefix.into()),
        }
    }

    /// Create an "ends with" consistency rule.
    pub fn ends_with(
        name: impl Into<String>,
        field: impl Into<String>,
        suffix: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            check: ConsistencyCheck::EndsWith(suffix.into()),
        }
    }

    /// Create a string length range rule.
    pub fn length_range(
        name: impl Into<String>,
        field: impl Into<String>,
        min: usize,
        max: usize,
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            check: ConsistencyCheck::LengthRange(min, max),
        }
    }

    /// Create a custom consistency rule.
    pub fn custom(
        name: impl Into<String>,
        field: impl Into<String>,
        checker: fn(&serde_json::Value) -> bool,
    ) -> Self {
        Self {
            name: name.into(),
            field: field.into(),
            check: ConsistencyCheck::Custom(checker),
        }
    }

    /// Evaluate the rule on a value.
    fn evaluate(&self, value: &serde_json::Value) -> bool {
        match &self.check {
            ConsistencyCheck::Contains(sub) => {
                value.as_str().map_or(false, |s| s.contains(sub.as_str()))
            }
            ConsistencyCheck::StartsWith(pre) => {
                value.as_str().map_or(false, |s| s.starts_with(pre.as_str()))
            }
            ConsistencyCheck::EndsWith(suf) => {
                value.as_str().map_or(false, |s| s.ends_with(suf.as_str()))
            }
            ConsistencyCheck::LengthRange(min, max) => value
                .as_str()
                .map_or(false, |s| s.len() >= *min && s.len() <= *max),
            ConsistencyCheck::Custom(f) => f(value),
        }
    }
}

// ── Freshness config ─────────────────────────────────────────────

/// Configuration for freshness measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessConfig {
    /// Field containing the timestamp.
    pub timestamp_field: String,
    /// Maximum age in seconds before data is considered stale.
    pub max_age_seconds: u64,
    /// Current time as Unix timestamp (for deterministic testing).
    pub current_time_epoch: u64,
}

/// Freshness result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreshnessResult {
    /// Whether the data is fresh.
    pub is_fresh: bool,
    /// Age of the newest record in seconds.
    pub newest_age_seconds: u64,
    /// Age of the oldest record in seconds.
    pub oldest_age_seconds: u64,
    /// Number of records with valid timestamps.
    pub valid_timestamps: usize,
    /// Number of records with invalid/missing timestamps.
    pub invalid_timestamps: usize,
}

// ── Quality report card ──────────────────────────────────────────

/// A report card summarizing data quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReportCard {
    /// Overall quality score [0.0, 1.0].
    pub overall_score: f64,
    /// Grade (A/B/C/D/F).
    pub grade: String,
    /// Per-dimension scores.
    pub dimensions: Vec<DimensionScore>,
    /// Per-field quality.
    pub field_quality: Vec<FieldQuality>,
    /// Total records analyzed.
    pub total_records: usize,
    /// Total fields analyzed.
    pub total_fields: usize,
    /// Whether all thresholds are met.
    pub all_thresholds_met: bool,
}

impl QualityReportCard {
    /// Compute a grade from the overall score.
    fn grade_from_score(score: f64) -> String {
        if score >= 0.95 {
            "A".into()
        } else if score >= 0.85 {
            "B".into()
        } else if score >= 0.70 {
            "C".into()
        } else if score >= 0.55 {
            "D".into()
        } else {
            "F".into()
        }
    }
}

// ── Quality trend point ──────────────────────────────────────────

/// A data point in a quality trend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTrendPoint {
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Overall score at this point.
    pub overall_score: f64,
    /// Per-dimension scores.
    pub dimension_scores: HashMap<String, f64>,
    /// Total records.
    pub total_records: usize,
}

// ── Data quality engine ──────────────────────────────────────────

/// The data quality assessment engine.
#[derive(Debug)]
pub struct DataQualityEngine {
    /// Fields to assess (empty = all fields).
    fields: Vec<String>,
    /// Consistency rules.
    consistency_rules: Vec<ConsistencyRule>,
    /// Quality threshold [0.0, 1.0] for each dimension.
    thresholds: HashMap<String, f64>,
    /// Historical trend data.
    trend_history: Vec<QualityTrendPoint>,
}

impl DataQualityEngine {
    /// Create a new engine.
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            consistency_rules: Vec::new(),
            thresholds: HashMap::new(),
            trend_history: Vec::new(),
        }
    }

    /// Specify which fields to assess. If empty, all fields are assessed.
    pub fn set_fields(&mut self, fields: Vec<String>) {
        self.fields = fields;
    }

    /// Add a consistency rule.
    pub fn add_consistency_rule(&mut self, rule: ConsistencyRule) {
        self.consistency_rules.push(rule);
    }

    /// Set a threshold for a dimension.
    pub fn set_threshold(&mut self, dimension: impl Into<String>, threshold: f64) {
        self.thresholds.insert(dimension.into(), threshold);
    }

    /// Get the trend history.
    pub fn trend_history(&self) -> &[QualityTrendPoint] {
        &self.trend_history
    }

    /// Compute completeness for each field.
    pub fn completeness(&self, data: &[Row]) -> Vec<FieldQuality> {
        let field_names = self.resolve_fields(data);
        let total = data.len();

        field_names
            .iter()
            .map(|field| {
                let null_count = data
                    .iter()
                    .filter(|row| {
                        row.get(field.as_str())
                            .map_or(true, |v| v.is_null())
                    })
                    .count();

                let non_null = total.saturating_sub(null_count);
                let completeness = if total == 0 {
                    1.0
                } else {
                    non_null as f64 / total as f64
                };

                // Count distinct non-null values.
                let distinct: HashSet<String> = data
                    .iter()
                    .filter_map(|row| row.get(field.as_str()))
                    .filter(|v| !v.is_null())
                    .map(|v| v.to_string())
                    .collect();

                let uniqueness = if non_null == 0 {
                    1.0
                } else {
                    distinct.len() as f64 / non_null as f64
                };

                FieldQuality {
                    field_name: field.clone(),
                    completeness,
                    uniqueness,
                    total_count: total,
                    null_count,
                    distinct_count: distinct.len(),
                }
            })
            .collect()
    }

    /// Evaluate consistency rules.
    pub fn consistency(&self, data: &[Row]) -> Vec<DimensionScore> {
        self.consistency_rules
            .iter()
            .map(|rule| {
                let threshold = self.thresholds.get(&rule.name).copied().unwrap_or(0.9);
                let mut passed = 0;
                let mut total = 0;

                for row in data {
                    if let Some(value) = row.get(&rule.field) {
                        if value.is_null() {
                            continue;
                        }
                        total += 1;
                        if rule.evaluate(value) {
                            passed += 1;
                        }
                    }
                }

                DimensionScore::new(&rule.name, passed, total, threshold)
            })
            .collect()
    }

    /// Evaluate freshness.
    pub fn freshness(
        &self,
        data: &[Row],
        config: &FreshnessConfig,
    ) -> FreshnessResult {
        let mut valid_timestamps = 0usize;
        let mut invalid_timestamps = 0usize;
        let mut newest_epoch = 0u64;
        let mut oldest_epoch = u64::MAX;

        for row in data {
            if let Some(value) = row.get(&config.timestamp_field) {
                if let Some(ts) = value.as_u64() {
                    valid_timestamps += 1;
                    if ts > newest_epoch {
                        newest_epoch = ts;
                    }
                    if ts < oldest_epoch {
                        oldest_epoch = ts;
                    }
                } else {
                    invalid_timestamps += 1;
                }
            } else {
                invalid_timestamps += 1;
            }
        }

        if valid_timestamps == 0 {
            return FreshnessResult {
                is_fresh: false,
                newest_age_seconds: 0,
                oldest_age_seconds: 0,
                valid_timestamps: 0,
                invalid_timestamps,
            };
        }

        let newest_age = config.current_time_epoch.saturating_sub(newest_epoch);
        let oldest_age = config.current_time_epoch.saturating_sub(oldest_epoch);

        FreshnessResult {
            is_fresh: newest_age <= config.max_age_seconds,
            newest_age_seconds: newest_age,
            oldest_age_seconds: oldest_age,
            valid_timestamps,
            invalid_timestamps,
        }
    }

    /// Compute an accuracy score based on a reference dataset.
    /// Compares fields between data and reference, matching by index.
    pub fn accuracy(
        &self,
        data: &[Row],
        reference: &[Row],
        fields: &[String],
    ) -> DimensionScore {
        let count = data.len().min(reference.len());
        let threshold = self.thresholds.get("accuracy").copied().unwrap_or(0.9);
        let mut passed = 0;

        for i in 0..count {
            let all_match = fields.iter().all(|field| {
                data[i].get(field) == reference[i].get(field)
            });
            if all_match {
                passed += 1;
            }
        }

        DimensionScore::new("accuracy", passed, count, threshold)
    }

    /// Generate a full quality report card.
    pub fn report_card(
        &mut self,
        data: &[Row],
        timestamp: impl Into<String>,
    ) -> Result<QualityReportCard, QualityError> {
        if data.is_empty() {
            return Err(QualityError::EmptyDataset);
        }

        let field_quality = self.completeness(data);
        let consistency_scores = self.consistency(data);

        // Build dimension scores.
        let mut dimensions = Vec::new();

        // Completeness dimension: average across all fields.
        let comp_threshold = self.thresholds.get("completeness").copied().unwrap_or(0.9);
        let avg_completeness = if field_quality.is_empty() {
            1.0
        } else {
            field_quality.iter().map(|f| f.completeness).sum::<f64>()
                / field_quality.len() as f64
        };
        let comp_passed = field_quality
            .iter()
            .filter(|f| f.completeness >= comp_threshold)
            .count();
        dimensions.push(DimensionScore {
            name: "completeness".into(),
            score: avg_completeness,
            records_evaluated: data.len(),
            records_passed: comp_passed,
            records_failed: field_quality.len().saturating_sub(comp_passed),
            threshold: comp_threshold,
            meets_threshold: avg_completeness >= comp_threshold,
        });

        // Uniqueness dimension.
        let uniq_threshold = self.thresholds.get("uniqueness").copied().unwrap_or(0.8);
        let avg_uniqueness = if field_quality.is_empty() {
            1.0
        } else {
            field_quality.iter().map(|f| f.uniqueness).sum::<f64>()
                / field_quality.len() as f64
        };
        dimensions.push(DimensionScore {
            name: "uniqueness".into(),
            score: avg_uniqueness,
            records_evaluated: data.len(),
            records_passed: 0,
            records_failed: 0,
            threshold: uniq_threshold,
            meets_threshold: avg_uniqueness >= uniq_threshold,
        });

        // Consistency dimensions from rules.
        dimensions.extend(consistency_scores);

        // Overall score: average of all dimension scores.
        let overall_score = if dimensions.is_empty() {
            1.0
        } else {
            dimensions.iter().map(|d| d.score).sum::<f64>() / dimensions.len() as f64
        };

        let all_thresholds_met = dimensions.iter().all(|d| d.meets_threshold);
        let grade = QualityReportCard::grade_from_score(overall_score);

        let total_fields = field_quality.len();

        // Record in trend history.
        let ts = timestamp.into();
        let mut dim_scores = HashMap::new();
        for d in &dimensions {
            dim_scores.insert(d.name.clone(), d.score);
        }
        self.trend_history.push(QualityTrendPoint {
            timestamp: ts,
            overall_score,
            dimension_scores: dim_scores,
            total_records: data.len(),
        });

        Ok(QualityReportCard {
            overall_score,
            grade,
            dimensions,
            field_quality,
            total_records: data.len(),
            total_fields,
            all_thresholds_met,
        })
    }

    // ── Private helpers ──

    fn resolve_fields(&self, data: &[Row]) -> Vec<String> {
        if !self.fields.is_empty() {
            return self.fields.clone();
        }
        // Discover all fields from data.
        let mut all_fields = Vec::new();
        let mut seen = HashSet::new();
        for row in data {
            for key in row.keys() {
                if seen.insert(key.clone()) {
                    all_fields.push(key.clone());
                }
            }
        }
        all_fields.sort();
        all_fields
    }
}

impl Default for DataQualityEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    fn sample_data() -> Vec<Row> {
        vec![
            row(&[
                ("name", serde_json::json!("Alice")),
                ("email", serde_json::json!("alice@example.com")),
                ("age", serde_json::json!(30)),
            ]),
            row(&[
                ("name", serde_json::json!("Bob")),
                ("email", serde_json::json!("bob@example.com")),
                ("age", serde_json::json!(25)),
            ]),
            row(&[
                ("name", serde_json::json!("Charlie")),
                ("email", serde_json::Value::Null),
                ("age", serde_json::json!(35)),
            ]),
        ]
    }

    #[test]
    fn completeness_all_present() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("name", serde_json::json!("A"))]),
            row(&[("name", serde_json::json!("B"))]),
        ];
        let quality = engine.completeness(&data);
        let name_q = quality.iter().find(|q| q.field_name == "name").unwrap();
        assert!((name_q.completeness - 1.0).abs() < f64::EPSILON);
        assert_eq!(name_q.null_count, 0);
    }

    #[test]
    fn completeness_with_nulls() {
        let engine = DataQualityEngine::new();
        let data = sample_data();
        let quality = engine.completeness(&data);
        let email_q = quality.iter().find(|q| q.field_name == "email").unwrap();
        assert!(email_q.completeness < 1.0);
        assert_eq!(email_q.null_count, 1);
    }

    #[test]
    fn completeness_missing_fields() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("a", serde_json::json!(1))]),
            row(&[("b", serde_json::json!(2))]),
        ];
        let quality = engine.completeness(&data);
        // "a" is missing in the second row.
        let a_q = quality.iter().find(|q| q.field_name == "a").unwrap();
        assert!(a_q.completeness < 1.0);
    }

    #[test]
    fn uniqueness_all_unique() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("id", serde_json::json!(1))]),
            row(&[("id", serde_json::json!(2))]),
            row(&[("id", serde_json::json!(3))]),
        ];
        let quality = engine.completeness(&data);
        let id_q = quality.iter().find(|q| q.field_name == "id").unwrap();
        assert!((id_q.uniqueness - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn uniqueness_with_duplicates() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("status", serde_json::json!("active"))]),
            row(&[("status", serde_json::json!("active"))]),
            row(&[("status", serde_json::json!("inactive"))]),
        ];
        let quality = engine.completeness(&data);
        let status_q = quality.iter().find(|q| q.field_name == "status").unwrap();
        assert!(status_q.uniqueness < 1.0);
        assert_eq!(status_q.distinct_count, 2);
    }

    #[test]
    fn consistency_contains() {
        let mut engine = DataQualityEngine::new();
        engine.add_consistency_rule(ConsistencyRule::contains(
            "email_format",
            "email",
            "@",
        ));

        let data = sample_data();
        let scores = engine.consistency(&data);
        assert_eq!(scores.len(), 1);
        // 2 out of 2 valid emails (null is skipped)
        assert!((scores[0].score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn consistency_starts_with() {
        let mut engine = DataQualityEngine::new();
        engine.add_consistency_rule(ConsistencyRule::starts_with(
            "prefix_check",
            "code",
            "US-",
        ));

        let data = vec![
            row(&[("code", serde_json::json!("US-001"))]),
            row(&[("code", serde_json::json!("US-002"))]),
            row(&[("code", serde_json::json!("UK-001"))]),
        ];
        let scores = engine.consistency(&data);
        // 2 out of 3 match.
        let expected = 2.0 / 3.0;
        assert!((scores[0].score - expected).abs() < 0.01);
    }

    #[test]
    fn consistency_length_range() {
        let mut engine = DataQualityEngine::new();
        engine.add_consistency_rule(ConsistencyRule::length_range(
            "zip_len",
            "zip",
            5,
            5,
        ));

        let data = vec![
            row(&[("zip", serde_json::json!("10001"))]),
            row(&[("zip", serde_json::json!("123"))]),
        ];
        let scores = engine.consistency(&data);
        assert!((scores[0].score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn freshness_fresh() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("ts", serde_json::json!(1000))]),
            row(&[("ts", serde_json::json!(900))]),
        ];
        let config = FreshnessConfig {
            timestamp_field: "ts".into(),
            max_age_seconds: 200,
            current_time_epoch: 1100,
        };
        let result = engine.freshness(&data, &config);
        assert!(result.is_fresh);
        assert_eq!(result.newest_age_seconds, 100);
        assert_eq!(result.valid_timestamps, 2);
    }

    #[test]
    fn freshness_stale() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("ts", serde_json::json!(100))]),
        ];
        let config = FreshnessConfig {
            timestamp_field: "ts".into(),
            max_age_seconds: 50,
            current_time_epoch: 1000,
        };
        let result = engine.freshness(&data, &config);
        assert!(!result.is_fresh);
        assert_eq!(result.newest_age_seconds, 900);
    }

    #[test]
    fn freshness_no_timestamps() {
        let engine = DataQualityEngine::new();
        let data = vec![row(&[("name", serde_json::json!("Alice"))])];
        let config = FreshnessConfig {
            timestamp_field: "ts".into(),
            max_age_seconds: 100,
            current_time_epoch: 1000,
        };
        let result = engine.freshness(&data, &config);
        assert!(!result.is_fresh);
        assert_eq!(result.valid_timestamps, 0);
    }

    #[test]
    fn accuracy_perfect() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("x", serde_json::json!(1))]),
            row(&[("x", serde_json::json!(2))]),
        ];
        let reference = vec![
            row(&[("x", serde_json::json!(1))]),
            row(&[("x", serde_json::json!(2))]),
        ];
        let score = engine.accuracy(&data, &reference, &["x".into()]);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn accuracy_partial() {
        let engine = DataQualityEngine::new();
        let data = vec![
            row(&[("x", serde_json::json!(1))]),
            row(&[("x", serde_json::json!(99))]),
        ];
        let reference = vec![
            row(&[("x", serde_json::json!(1))]),
            row(&[("x", serde_json::json!(2))]),
        ];
        let score = engine.accuracy(&data, &reference, &["x".into()]);
        assert!((score.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn report_card_basic() {
        let mut engine = DataQualityEngine::new();
        engine.set_threshold("completeness", 0.8);

        let data = sample_data();
        let report = engine.report_card(&data, "2026-03-09T00:00:00Z").unwrap();
        assert!(report.overall_score > 0.0);
        assert!(report.overall_score <= 1.0);
        assert_eq!(report.total_records, 3);
        assert!(!report.grade.is_empty());
    }

    #[test]
    fn report_card_empty_dataset() {
        let mut engine = DataQualityEngine::new();
        let err = engine.report_card(&[], "2026-03-09").unwrap_err();
        assert_eq!(err, QualityError::EmptyDataset);
    }

    #[test]
    fn grade_mapping() {
        assert_eq!(QualityReportCard::grade_from_score(0.96), "A");
        assert_eq!(QualityReportCard::grade_from_score(0.90), "B");
        assert_eq!(QualityReportCard::grade_from_score(0.75), "C");
        assert_eq!(QualityReportCard::grade_from_score(0.60), "D");
        assert_eq!(QualityReportCard::grade_from_score(0.40), "F");
    }

    #[test]
    fn trend_tracking() {
        let mut engine = DataQualityEngine::new();
        let data1 = sample_data();
        let data2 = vec![
            row(&[("name", serde_json::json!("Alice")), ("email", serde_json::json!("a@b.com")), ("age", serde_json::json!(30))]),
        ];

        engine.report_card(&data1, "2026-01-01").unwrap();
        engine.report_card(&data2, "2026-02-01").unwrap();

        let trend = engine.trend_history();
        assert_eq!(trend.len(), 2);
        assert_eq!(trend[0].timestamp, "2026-01-01");
        assert_eq!(trend[1].timestamp, "2026-02-01");
    }

    #[test]
    fn set_specific_fields() {
        let mut engine = DataQualityEngine::new();
        engine.set_fields(vec!["name".into()]);

        let data = sample_data();
        let quality = engine.completeness(&data);
        assert_eq!(quality.len(), 1);
        assert_eq!(quality[0].field_name, "name");
    }

    #[test]
    fn custom_consistency_rule() {
        fn is_positive(v: &serde_json::Value) -> bool {
            v.as_f64().map_or(false, |n| n > 0.0)
        }

        let mut engine = DataQualityEngine::new();
        engine.add_consistency_rule(ConsistencyRule::custom(
            "positive_age",
            "age",
            is_positive,
        ));

        let data = vec![
            row(&[("age", serde_json::json!(25))]),
            row(&[("age", serde_json::json!(-5))]),
        ];
        let scores = engine.consistency(&data);
        assert!((scores[0].score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn dimension_score_threshold() {
        let score = DimensionScore::new("test", 9, 10, 0.9);
        assert!((score.score - 0.9).abs() < f64::EPSILON);
        assert!(score.meets_threshold);
        assert_eq!(score.records_failed, 1);
    }

    #[test]
    fn dimension_score_below_threshold() {
        let score = DimensionScore::new("test", 7, 10, 0.9);
        assert!(!score.meets_threshold);
    }

    #[test]
    fn error_display() {
        let e = QualityError::EmptyDataset;
        assert!(format!("{e}").contains("empty dataset"));
        let e2 = QualityError::FieldNotFound("x".into());
        assert!(format!("{e2}").contains("field not found"));
    }
}
