//! A/B testing framework with deterministic hash-based variant assignment,
//! traffic allocation, mutual exclusion groups, holdout groups, metric
//! tracking, and chi-squared statistical significance.

use std::collections::HashMap;

// ── Hashing ─────────────────────────────────────────────────────

/// Deterministic FNV-1a hash for user → bucket mapping.
fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Map a user ID + experiment key to a bucket in [0, 10000).
fn bucket(user_id: &str, experiment_key: &str) -> u32 {
    let combined = format!("{}:{}", experiment_key, user_id);
    (fnv1a(combined.as_bytes()) % 10000) as u32
}

// ── Variant ─────────────────────────────────────────────────────

/// A variant within an experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    /// Unique key for this variant (e.g. "control", "treatment_a").
    pub key: String,
    /// Percentage of traffic [0.0, 100.0].
    pub weight: f64,
    /// Optional description.
    pub description: String,
}

impl Variant {
    pub fn new(key: impl Into<String>, weight: f64) -> Self {
        Self {
            key: key.into(),
            weight,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

// ── Experiment ──────────────────────────────────────────────────

/// Status of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    Draft,
    Running,
    Paused,
    Completed,
}

/// Definition of an A/B experiment.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub key: String,
    pub description: String,
    pub variants: Vec<Variant>,
    pub status: ExperimentStatus,
    /// What fraction of total traffic enters this experiment [0.0, 100.0].
    pub traffic_percent: f64,
    /// Mutual exclusion group (if any).
    pub exclusion_group: Option<String>,
    /// Is this a holdout group experiment?
    pub holdout: bool,
}

impl Experiment {
    pub fn new(key: impl Into<String>, variants: Vec<Variant>) -> Self {
        Self {
            key: key.into(),
            description: String::new(),
            variants,
            status: ExperimentStatus::Draft,
            traffic_percent: 100.0,
            exclusion_group: None,
            holdout: false,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_status(mut self, status: ExperimentStatus) -> Self {
        self.status = status;
        self
    }

    pub fn with_traffic(mut self, percent: f64) -> Self {
        self.traffic_percent = percent.clamp(0.0, 100.0);
        self
    }

    pub fn with_exclusion_group(mut self, group: impl Into<String>) -> Self {
        self.exclusion_group = Some(group.into());
        self
    }

    pub fn with_holdout(mut self, holdout: bool) -> Self {
        self.holdout = holdout;
        self
    }

    /// Total weight across all variants.
    pub fn total_weight(&self) -> f64 {
        self.variants.iter().map(|v| v.weight).sum()
    }
}

// ── Assignment ──────────────────────────────────────────────────

/// The result of assigning a user to an experiment.
#[derive(Debug, Clone, PartialEq)]
pub enum Assignment {
    /// Assigned to a variant.
    Assigned { experiment: String, variant: String },
    /// Not in the experiment's traffic allocation.
    NotInTraffic,
    /// Excluded due to mutual exclusion.
    Excluded,
    /// Experiment not running.
    NotRunning,
}

// ── Metrics ─────────────────────────────────────────────────────

/// Aggregated metrics for a variant.
#[derive(Debug, Clone, Default)]
pub struct VariantMetrics {
    /// Number of users exposed to this variant.
    pub impressions: u64,
    /// Number of conversions (success events).
    pub conversions: u64,
    /// Sum of numeric metric values.
    pub value_sum: f64,
}

impl VariantMetrics {
    pub fn conversion_rate(&self) -> f64 {
        if self.impressions == 0 {
            0.0
        } else {
            self.conversions as f64 / self.impressions as f64
        }
    }
}

// ── ABTester ────────────────────────────────────────────────────

/// A/B testing engine: manages experiments, assigns users, tracks metrics.
pub struct ABTester {
    experiments: HashMap<String, Experiment>,
    /// Per-experiment, per-variant metrics.
    metrics: HashMap<String, HashMap<String, VariantMetrics>>,
    /// Tracks which exclusion group a user is already in.
    user_exclusions: HashMap<String, String>,
    /// Force overrides: (user_id, experiment_key) → variant_key.
    overrides: HashMap<(String, String), String>,
}

impl ABTester {
    pub fn new() -> Self {
        Self {
            experiments: HashMap::new(),
            metrics: HashMap::new(),
            user_exclusions: HashMap::new(),
            overrides: HashMap::new(),
        }
    }

    /// Register an experiment.
    pub fn register(&mut self, experiment: Experiment) {
        let key = experiment.key.clone();
        // Initialize metrics for each variant.
        let variant_metrics: HashMap<String, VariantMetrics> = experiment
            .variants
            .iter()
            .map(|v| (v.key.clone(), VariantMetrics::default()))
            .collect();
        self.metrics.insert(key.clone(), variant_metrics);
        self.experiments.insert(key, experiment);
    }

    /// Start an experiment.
    pub fn start(&mut self, key: &str) -> bool {
        if let Some(exp) = self.experiments.get_mut(key) {
            exp.status = ExperimentStatus::Running;
            true
        } else {
            false
        }
    }

    /// Pause an experiment.
    pub fn pause(&mut self, key: &str) -> bool {
        if let Some(exp) = self.experiments.get_mut(key) {
            exp.status = ExperimentStatus::Paused;
            true
        } else {
            false
        }
    }

    /// Complete an experiment.
    pub fn complete(&mut self, key: &str) -> bool {
        if let Some(exp) = self.experiments.get_mut(key) {
            exp.status = ExperimentStatus::Completed;
            true
        } else {
            false
        }
    }

    /// Force a specific variant for a user (testing / QA override).
    pub fn set_override(&mut self, user_id: &str, experiment_key: &str, variant_key: &str) {
        self.overrides.insert(
            (user_id.to_string(), experiment_key.to_string()),
            variant_key.to_string(),
        );
    }

    /// Assign a user to a variant.
    pub fn assign(&mut self, user_id: &str, experiment_key: &str) -> Assignment {
        // Check override.
        let override_key = (user_id.to_string(), experiment_key.to_string());
        if let Some(variant_key) = self.overrides.get(&override_key) {
            return Assignment::Assigned {
                experiment: experiment_key.to_string(),
                variant: variant_key.clone(),
            };
        }

        let experiment = match self.experiments.get(experiment_key) {
            Some(e) => e.clone(),
            None => return Assignment::NotRunning,
        };

        if experiment.status != ExperimentStatus::Running {
            return Assignment::NotRunning;
        }

        // Mutual exclusion: if the user is already in another experiment in
        // the same group, exclude them.
        if let Some(group) = &experiment.exclusion_group {
            if let Some(existing_key) = self.user_exclusions.get(user_id) {
                if existing_key != experiment_key {
                    // Check if the existing experiment is in the same group.
                    if let Some(existing_exp) = self.experiments.get(existing_key.as_str()) {
                        if existing_exp.exclusion_group.as_deref() == Some(group.as_str()) {
                            return Assignment::Excluded;
                        }
                    }
                }
            }
        }

        // Traffic allocation check.
        let user_bucket = bucket(user_id, experiment_key);
        let traffic_threshold = (experiment.traffic_percent * 100.0) as u32;
        if user_bucket >= traffic_threshold {
            return Assignment::NotInTraffic;
        }

        // Determine variant by weight-based bucket partitioning.
        let total = experiment.total_weight();
        if total <= 0.0 {
            return Assignment::NotInTraffic;
        }
        let variant_bucket = (user_bucket as f64 / traffic_threshold as f64) * total;
        let mut cumulative = 0.0;
        let mut chosen = &experiment.variants[0];
        for v in &experiment.variants {
            cumulative += v.weight;
            if variant_bucket < cumulative {
                chosen = v;
                break;
            }
        }

        // Record exclusion group membership.
        if experiment.exclusion_group.is_some() {
            self.user_exclusions
                .insert(user_id.to_string(), experiment_key.to_string());
        }

        // Increment impressions.
        if let Some(variant_map) = self.metrics.get_mut(experiment_key) {
            if let Some(m) = variant_map.get_mut(&chosen.key) {
                m.impressions += 1;
            }
        }

        Assignment::Assigned {
            experiment: experiment_key.to_string(),
            variant: chosen.key.clone(),
        }
    }

    /// Record a conversion event for a user in an experiment variant.
    pub fn record_conversion(&mut self, experiment_key: &str, variant_key: &str, value: f64) {
        if let Some(variant_map) = self.metrics.get_mut(experiment_key) {
            if let Some(m) = variant_map.get_mut(variant_key) {
                m.conversions += 1;
                m.value_sum += value;
            }
        }
    }

    /// Get metrics for a variant.
    pub fn get_metrics(&self, experiment_key: &str, variant_key: &str) -> Option<&VariantMetrics> {
        self.metrics
            .get(experiment_key)
            .and_then(|vm| vm.get(variant_key))
    }

    /// Get all variant metrics for an experiment.
    pub fn experiment_metrics(&self, experiment_key: &str) -> Option<&HashMap<String, VariantMetrics>> {
        self.metrics.get(experiment_key)
    }

    /// Compute a chi-squared statistic for an experiment (control vs treatment).
    /// Returns (chi2, degrees_of_freedom).
    pub fn chi_squared(&self, experiment_key: &str) -> Option<(f64, usize)> {
        let variant_map = self.metrics.get(experiment_key)?;
        let variants: Vec<&VariantMetrics> = variant_map.values().collect();
        if variants.len() < 2 {
            return None;
        }

        let total_impressions: u64 = variants.iter().map(|v| v.impressions).sum();
        let total_conversions: u64 = variants.iter().map(|v| v.conversions).sum();

        if total_impressions == 0 {
            return None;
        }

        let overall_rate = total_conversions as f64 / total_impressions as f64;
        let mut chi2 = 0.0;

        for v in &variants {
            if v.impressions == 0 {
                continue;
            }
            let expected_conv = v.impressions as f64 * overall_rate;
            let expected_non = v.impressions as f64 * (1.0 - overall_rate);

            if expected_conv > 0.0 {
                let diff_conv = v.conversions as f64 - expected_conv;
                chi2 += (diff_conv * diff_conv) / expected_conv;
            }
            let actual_non = (v.impressions - v.conversions) as f64;
            if expected_non > 0.0 {
                let diff_non = actual_non - expected_non;
                chi2 += (diff_non * diff_non) / expected_non;
            }
        }

        let df = variants.len() - 1;
        Some((chi2, df))
    }

    /// Simple significance check: chi2 > 3.841 for df=1 is p<0.05.
    pub fn is_significant(&self, experiment_key: &str, alpha_threshold: f64) -> bool {
        if let Some((chi2, df)) = self.chi_squared(experiment_key) {
            // Critical values for common alpha levels at df=1.
            let critical = if df == 1 {
                if alpha_threshold <= 0.001 {
                    10.828
                } else if alpha_threshold <= 0.01 {
                    6.635
                } else if alpha_threshold <= 0.05 {
                    3.841
                } else if alpha_threshold <= 0.10 {
                    2.706
                } else {
                    0.0
                }
            } else {
                // For higher df, use a rough approximation.
                // This is not a full chi-squared table — production code would
                // use a proper inverse CDF.
                df as f64 * 3.841
            };
            chi2 > critical
        } else {
            false
        }
    }

    /// Get experiment by key.
    pub fn get_experiment(&self, key: &str) -> Option<&Experiment> {
        self.experiments.get(key)
    }

    /// List all experiment keys.
    pub fn list_experiments(&self) -> Vec<&str> {
        let mut keys: Vec<_> = self.experiments.keys().map(|k| k.as_str()).collect();
        keys.sort();
        keys
    }
}

impl Default for ABTester {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_experiment() -> Experiment {
        Experiment::new("test_exp", vec![
            Variant::new("control", 50.0),
            Variant::new("treatment", 50.0),
        ])
        .with_status(ExperimentStatus::Running)
    }

    #[test]
    fn register_and_list() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());
        assert_eq!(ab.list_experiments(), vec!["test_exp"]);
    }

    #[test]
    fn assign_returns_variant() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());
        let result = ab.assign("user_1", "test_exp");
        match &result {
            Assignment::Assigned { experiment, variant } => {
                assert_eq!(experiment, "test_exp");
                assert!(variant == "control" || variant == "treatment");
            }
            _ => panic!("expected Assigned, got {:?}", result),
        }
    }

    #[test]
    fn deterministic_assignment() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());
        let r1 = ab.assign("user_42", "test_exp");
        // Re-create the tester and verify same assignment.
        let mut ab2 = ABTester::new();
        ab2.register(simple_experiment());
        let r2 = ab2.assign("user_42", "test_exp");
        assert_eq!(r1, r2);
    }

    #[test]
    fn not_running_experiment() {
        let mut ab = ABTester::new();
        let exp = Experiment::new("draft", vec![Variant::new("a", 100.0)]);
        ab.register(exp);
        assert_eq!(ab.assign("user", "draft"), Assignment::NotRunning);
    }

    #[test]
    fn unknown_experiment() {
        let mut ab = ABTester::new();
        assert_eq!(ab.assign("user", "nope"), Assignment::NotRunning);
    }

    #[test]
    fn traffic_allocation() {
        let mut ab = ABTester::new();
        let exp = simple_experiment().with_traffic(10.0);
        ab.register(exp);
        // With 10% traffic, most users should be NotInTraffic.
        let mut in_traffic = 0;
        for i in 0..100 {
            let user = format!("user_{}", i);
            if matches!(ab.assign(&user, "test_exp"), Assignment::Assigned { .. }) {
                in_traffic += 1;
            }
        }
        // Roughly 10% should be in traffic (allow wide margin for hash distribution).
        assert!(in_traffic < 40, "in_traffic was {}", in_traffic);
    }

    #[test]
    fn override_forces_variant() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());
        ab.set_override("qa_user", "test_exp", "treatment");
        let result = ab.assign("qa_user", "test_exp");
        assert_eq!(result, Assignment::Assigned {
            experiment: "test_exp".into(),
            variant: "treatment".into(),
        });
    }

    #[test]
    fn mutual_exclusion() {
        let mut ab = ABTester::new();
        let exp1 = Experiment::new("exp_a", vec![Variant::new("v1", 100.0)])
            .with_status(ExperimentStatus::Running)
            .with_exclusion_group("checkout");
        let exp2 = Experiment::new("exp_b", vec![Variant::new("v2", 100.0)])
            .with_status(ExperimentStatus::Running)
            .with_exclusion_group("checkout");
        ab.register(exp1);
        ab.register(exp2);

        // User gets into exp_a first.
        let r1 = ab.assign("user_1", "exp_a");
        assert!(matches!(r1, Assignment::Assigned { .. }));

        // Same user excluded from exp_b (same group).
        let r2 = ab.assign("user_1", "exp_b");
        assert_eq!(r2, Assignment::Excluded);
    }

    #[test]
    fn metric_tracking() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());

        // Assign a few users.
        let _ = ab.assign("u1", "test_exp");
        let _ = ab.assign("u2", "test_exp");

        // Record conversions.
        ab.record_conversion("test_exp", "control", 1.0);
        ab.record_conversion("test_exp", "treatment", 1.0);
        ab.record_conversion("test_exp", "treatment", 1.0);

        let control = ab.get_metrics("test_exp", "control").unwrap();
        let treatment = ab.get_metrics("test_exp", "treatment").unwrap();
        assert_eq!(control.conversions, 1);
        assert_eq!(treatment.conversions, 2);
        assert!((treatment.value_sum - 2.0).abs() < 1e-10);
    }

    #[test]
    fn conversion_rate() {
        let m = VariantMetrics {
            impressions: 100,
            conversions: 25,
            value_sum: 25.0,
        };
        assert!((m.conversion_rate() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn conversion_rate_zero_impressions() {
        let m = VariantMetrics::default();
        assert!((m.conversion_rate()).abs() < 1e-10);
    }

    #[test]
    fn chi_squared_returns_value() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());

        // Simulate data.
        if let Some(vm) = ab.metrics.get_mut("test_exp") {
            vm.insert("control".into(), VariantMetrics {
                impressions: 1000,
                conversions: 100,
                value_sum: 100.0,
            });
            vm.insert("treatment".into(), VariantMetrics {
                impressions: 1000,
                conversions: 150,
                value_sum: 150.0,
            });
        }

        let (chi2, df) = ab.chi_squared("test_exp").unwrap();
        assert!(chi2 > 0.0);
        assert_eq!(df, 1);
    }

    #[test]
    fn significance_test() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());

        // Large effect size should be significant.
        if let Some(vm) = ab.metrics.get_mut("test_exp") {
            vm.insert("control".into(), VariantMetrics {
                impressions: 1000,
                conversions: 50,
                value_sum: 50.0,
            });
            vm.insert("treatment".into(), VariantMetrics {
                impressions: 1000,
                conversions: 200,
                value_sum: 200.0,
            });
        }

        assert!(ab.is_significant("test_exp", 0.05));
    }

    #[test]
    fn significance_not_reached() {
        let mut ab = ABTester::new();
        ab.register(simple_experiment());

        // Tiny sample — not significant.
        if let Some(vm) = ab.metrics.get_mut("test_exp") {
            vm.insert("control".into(), VariantMetrics {
                impressions: 10,
                conversions: 5,
                value_sum: 5.0,
            });
            vm.insert("treatment".into(), VariantMetrics {
                impressions: 10,
                conversions: 6,
                value_sum: 6.0,
            });
        }

        assert!(!ab.is_significant("test_exp", 0.05));
    }

    #[test]
    fn start_pause_complete_lifecycle() {
        let mut ab = ABTester::new();
        let exp = Experiment::new("lc", vec![Variant::new("a", 100.0)]);
        ab.register(exp);
        assert_eq!(ab.get_experiment("lc").unwrap().status, ExperimentStatus::Draft);
        ab.start("lc");
        assert_eq!(ab.get_experiment("lc").unwrap().status, ExperimentStatus::Running);
        ab.pause("lc");
        assert_eq!(ab.get_experiment("lc").unwrap().status, ExperimentStatus::Paused);
        ab.complete("lc");
        assert_eq!(ab.get_experiment("lc").unwrap().status, ExperimentStatus::Completed);
    }

    #[test]
    fn holdout_flag() {
        let exp = simple_experiment().with_holdout(true);
        assert!(exp.holdout);
    }

    #[test]
    fn variant_description() {
        let v = Variant::new("test", 50.0).with_description("the test variant");
        assert_eq!(v.description, "the test variant");
    }

    #[test]
    fn bucket_determinism() {
        let b1 = bucket("user1", "exp");
        let b2 = bucket("user1", "exp");
        assert_eq!(b1, b2);
    }

    #[test]
    fn bucket_range() {
        for i in 0..200 {
            let b = bucket(&format!("u{}", i), "exp");
            assert!(b < 10000);
        }
    }
}
