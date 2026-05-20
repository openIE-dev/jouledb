//! A/B Testing — experiment definition, variant assignment via hashing,
//! traffic allocation, feature flags integration, result tracking, and
//! statistical significance.
//!
//! Pure Rust experimentation framework. No browser or network dependencies.

use std::collections::HashMap;
use std::fmt;

// ── Variant ──────────────────────────────────────────────────────

/// A single variant in an experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub weight: f64,
    pub is_control: bool,
    pub config: HashMap<String, String>,
}

impl Variant {
    pub fn new(name: impl Into<String>, weight: f64) -> Self {
        Self {
            name: name.into(),
            weight,
            is_control: false,
            config: HashMap::new(),
        }
    }

    pub fn control(name: impl Into<String>, weight: f64) -> Self {
        Self {
            name: name.into(),
            weight,
            is_control: true,
            config: HashMap::new(),
        }
    }

    pub fn with_config(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.config.insert(key.into(), val.into());
        self
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(w={:.2}{})", self.name, self.weight, if self.is_control { ",ctrl" } else { "" })
    }
}

// ── Experiment Status ────────────────────────────────────────────

/// Status of an experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    Draft,
    Running,
    Paused,
    Completed,
    Archived,
}

impl fmt::Display for ExperimentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Draft => write!(f, "draft"),
            Self::Running => write!(f, "running"),
            Self::Paused => write!(f, "paused"),
            Self::Completed => write!(f, "completed"),
            Self::Archived => write!(f, "archived"),
        }
    }
}

// ── Experiment ───────────────────────────────────────────────────

/// An A/B experiment definition.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub status: ExperimentStatus,
    pub variants: Vec<Variant>,
    pub traffic_fraction: f64,
    pub salt: String,
}

impl Experiment {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let id_str: String = id.into();
        Self {
            salt: id_str.clone(),
            id: id_str,
            name: name.into(),
            description: None,
            status: ExperimentStatus::Draft,
            variants: Vec::new(),
            traffic_fraction: 1.0,
        }
    }

    pub fn description(mut self, d: impl Into<String>) -> Self { self.description = Some(d.into()); self }
    pub fn status(mut self, s: ExperimentStatus) -> Self { self.status = s; self }
    pub fn traffic_fraction(mut self, f: f64) -> Self { self.traffic_fraction = f.clamp(0.0, 1.0); self }
    pub fn salt(mut self, s: impl Into<String>) -> Self { self.salt = s.into(); self }

    pub fn add_variant(mut self, v: Variant) -> Self {
        self.variants.push(v);
        self
    }

    /// Check if the experiment is active and can assign variants.
    pub fn is_active(&self) -> bool {
        self.status == ExperimentStatus::Running && !self.variants.is_empty()
    }

    /// Get the control variant.
    pub fn control(&self) -> Option<&Variant> {
        self.variants.iter().find(|v| v.is_control)
    }

    /// Sum of all variant weights.
    pub fn total_weight(&self) -> f64 {
        self.variants.iter().map(|v| v.weight).sum()
    }

    /// Normalized weights (each variant's fraction of total).
    pub fn normalized_weights(&self) -> Vec<f64> {
        let total = self.total_weight();
        if total <= 0.0 { return vec![]; }
        self.variants.iter().map(|v| v.weight / total).collect()
    }

    /// Assign a variant to a user using deterministic hashing.
    pub fn assign(&self, user_id: &str) -> Option<&Variant> {
        if !self.is_active() { return None; }

        let hash = hash_assignment(&self.salt, user_id);
        let bucket = (hash % 10_000) as f64 / 10_000.0;

        // Check traffic fraction
        if bucket >= self.traffic_fraction {
            return None; // User is not in the experiment
        }

        // Scale bucket into the traffic fraction
        let scaled = bucket / self.traffic_fraction;

        // Weighted variant selection
        let total = self.total_weight();
        if total <= 0.0 { return None; }

        let mut cumulative = 0.0;
        for variant in &self.variants {
            cumulative += variant.weight / total;
            if scaled < cumulative {
                return Some(variant);
            }
        }

        self.variants.last()
    }

    /// Validate the experiment configuration.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.variants.is_empty() {
            issues.push("no variants defined".into());
        } else if self.variants.len() < 2 {
            issues.push("need at least 2 variants for an experiment".into());
        }
        let control_count = self.variants.iter().filter(|v| v.is_control).count();
        if control_count == 0 && !self.variants.is_empty() {
            issues.push("no control variant defined".into());
        }
        if control_count > 1 {
            issues.push("multiple control variants defined".into());
        }
        if self.total_weight() <= 0.0 && !self.variants.is_empty() {
            issues.push("total weight is zero or negative".into());
        }
        if self.traffic_fraction <= 0.0 {
            issues.push("traffic fraction is zero or negative".into());
        }
        issues
    }

    /// Number of variants.
    pub fn variant_count(&self) -> usize { self.variants.len() }
}

// ── Hashing ──────────────────────────────────────────────────────

/// Deterministic hash for variant assignment.
/// Uses a simple FNV-1a-like hash. Same (salt, user_id) always produces the same result.
pub fn hash_assignment(salt: &str, user_id: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in salt.bytes().chain(b":".iter().copied()).chain(user_id.bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Map a hash to a 0..1 range.
pub fn hash_to_fraction(hash: u64) -> f64 {
    (hash % 10_000) as f64 / 10_000.0
}

// ── Feature Flags ────────────────────────────────────────────────

/// Feature flag backed by experiment results.
#[derive(Debug, Clone)]
pub struct FeatureFlag {
    pub name: String,
    pub default_enabled: bool,
    pub experiment_id: Option<String>,
    pub enabled_variant: Option<String>,
    /// Manual overrides: user_id -> enabled.
    pub overrides: HashMap<String, bool>,
}

impl FeatureFlag {
    pub fn new(name: impl Into<String>, default: bool) -> Self {
        Self {
            name: name.into(),
            default_enabled: default,
            experiment_id: None,
            enabled_variant: None,
            overrides: HashMap::new(),
        }
    }

    pub fn with_experiment(mut self, exp_id: impl Into<String>, variant: impl Into<String>) -> Self {
        self.experiment_id = Some(exp_id.into());
        self.enabled_variant = Some(variant.into());
        self
    }

    pub fn add_override(mut self, user_id: impl Into<String>, enabled: bool) -> Self {
        self.overrides.insert(user_id.into(), enabled);
        self
    }

    /// Evaluate whether the flag is enabled for a user.
    /// Priority: overrides > experiment > default.
    pub fn is_enabled(&self, user_id: &str, assigned_variant: Option<&str>) -> bool {
        // 1. Check manual overrides
        if let Some(&enabled) = self.overrides.get(user_id) {
            return enabled;
        }

        // 2. Check experiment assignment
        if let Some(ref enabled_var) = self.enabled_variant {
            if let Some(assigned) = assigned_variant {
                return assigned == enabled_var;
            }
        }

        // 3. Default
        self.default_enabled
    }
}

// ── Result Tracking ──────────────────────────────────────────────

/// Metric type for result tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Binary outcome (e.g. converted or not).
    Conversion,
    /// Continuous value (e.g. revenue, time on page).
    Continuous,
    /// Count (e.g. number of clicks).
    Count,
}

/// Tracked results for a variant.
#[derive(Debug, Clone)]
pub struct VariantResult {
    pub variant_name: String,
    pub sample_size: u64,
    pub conversions: u64,
    pub sum_value: f64,
    pub sum_sq_value: f64,
}

impl VariantResult {
    pub fn new(variant_name: impl Into<String>) -> Self {
        Self {
            variant_name: variant_name.into(),
            sample_size: 0,
            conversions: 0,
            sum_value: 0.0,
            sum_sq_value: 0.0,
        }
    }

    /// Record a conversion event.
    pub fn record_conversion(&mut self, converted: bool) {
        self.sample_size += 1;
        if converted {
            self.conversions += 1;
            self.sum_value += 1.0;
            self.sum_sq_value += 1.0;
        }
    }

    /// Record a continuous value.
    pub fn record_value(&mut self, val: f64) {
        self.sample_size += 1;
        self.sum_value += val;
        self.sum_sq_value += val * val;
    }

    /// Conversion rate (for binary metrics).
    pub fn conversion_rate(&self) -> f64 {
        if self.sample_size == 0 { 0.0 }
        else { self.conversions as f64 / self.sample_size as f64 }
    }

    /// Mean value (for continuous metrics).
    pub fn mean(&self) -> f64 {
        if self.sample_size == 0 { 0.0 }
        else { self.sum_value / self.sample_size as f64 }
    }

    /// Variance (for continuous metrics).
    pub fn variance(&self) -> f64 {
        if self.sample_size < 2 { return 0.0; }
        let n = self.sample_size as f64;
        (self.sum_sq_value / n) - (self.sum_value / n).powi(2)
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Standard error of the mean.
    pub fn std_error(&self) -> f64 {
        if self.sample_size == 0 { return 0.0; }
        self.std_dev() / (self.sample_size as f64).sqrt()
    }
}

// ── Statistical Significance ─────────────────────────────────────

/// Result of a significance test.
#[derive(Debug, Clone)]
pub struct SignificanceResult {
    pub z_score: f64,
    pub p_value_approx: f64,
    pub is_significant: bool,
    pub confidence_level: f64,
    pub lift: f64,
}

impl fmt::Display for SignificanceResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "z={:.3}, p~{:.4}, lift={:.2}%, sig={}",
            self.z_score, self.p_value_approx, self.lift * 100.0,
            if self.is_significant { "yes" } else { "no" }
        )
    }
}

/// Two-proportion z-test for conversion rate experiments.
pub fn significance_test_proportions(
    control: &VariantResult,
    treatment: &VariantResult,
    confidence: f64,
) -> SignificanceResult {
    let p1 = control.conversion_rate();
    let p2 = treatment.conversion_rate();
    let n1 = control.sample_size as f64;
    let n2 = treatment.sample_size as f64;

    let lift = if p1 > 0.0 { (p2 - p1) / p1 } else { 0.0 };

    if n1 < 1.0 || n2 < 1.0 {
        return SignificanceResult {
            z_score: 0.0,
            p_value_approx: 1.0,
            is_significant: false,
            confidence_level: confidence,
            lift,
        };
    }

    // Pooled proportion
    let p_pool = (control.conversions as f64 + treatment.conversions as f64) / (n1 + n2);
    let se = (p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2)).sqrt();

    let z = if se > 0.0 { (p2 - p1) / se } else { 0.0 };

    // Approximate p-value from z-score using rational approximation
    let p_value = two_tail_p_value(z);

    let z_threshold = z_critical(confidence);

    SignificanceResult {
        z_score: z,
        p_value_approx: p_value,
        is_significant: z.abs() > z_threshold,
        confidence_level: confidence,
        lift,
    }
}

/// Two-sample z-test for continuous metrics.
pub fn significance_test_means(
    control: &VariantResult,
    treatment: &VariantResult,
    confidence: f64,
) -> SignificanceResult {
    let m1 = control.mean();
    let m2 = treatment.mean();
    let n1 = control.sample_size as f64;
    let n2 = treatment.sample_size as f64;

    let lift = if m1.abs() > 1e-15 { (m2 - m1) / m1 } else { 0.0 };

    if n1 < 2.0 || n2 < 2.0 {
        return SignificanceResult {
            z_score: 0.0,
            p_value_approx: 1.0,
            is_significant: false,
            confidence_level: confidence,
            lift,
        };
    }

    let se = (control.variance() / n1 + treatment.variance() / n2).sqrt();
    let z = if se > 0.0 { (m2 - m1) / se } else { 0.0 };
    let p_value = two_tail_p_value(z);
    let z_threshold = z_critical(confidence);

    SignificanceResult {
        z_score: z,
        p_value_approx: p_value,
        is_significant: z.abs() > z_threshold,
        confidence_level: confidence,
        lift,
    }
}

/// Required sample size per variant for a given expected effect.
pub fn required_sample_size(
    baseline_rate: f64,
    minimum_detectable_effect: f64,
    confidence: f64,
    power: f64,
) -> u64 {
    let p1 = baseline_rate;
    let p2 = baseline_rate + minimum_detectable_effect;
    let z_alpha = z_critical(confidence);
    let z_beta = z_critical(power);
    let numerator = (z_alpha + z_beta).powi(2) * (p1 * (1.0 - p1) + p2 * (1.0 - p2));
    let denominator = (p2 - p1).powi(2);
    if denominator <= 0.0 { return 0; }
    (numerator / denominator).ceil() as u64
}

// ── Internal helpers ─────────────────────────────────────────────

/// Approximate z critical value for common confidence levels.
fn z_critical(confidence: f64) -> f64 {
    if confidence >= 0.999 { 3.291 }
    else if confidence >= 0.99 { 2.576 }
    else if confidence >= 0.975 { 2.241 }
    else if confidence >= 0.95 { 1.960 }
    else if confidence >= 0.90 { 1.645 }
    else if confidence >= 0.80 { 1.282 }
    else { 1.0 }
}

/// Approximate two-tailed p-value from z-score.
fn two_tail_p_value(z: f64) -> f64 {
    // Abramowitz and Stegun approximation for normal CDF
    let az = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * az);
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p_one_tail = d * (-az * az / 2.0).exp()
        * (t * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937
                    + t * (-1.821255978
                        + t * 1.330274429)))));
    (2.0 * p_one_tail).clamp(0.0, 1.0)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_creation() {
        let v = Variant::new("treatment", 0.5).with_config("color", "blue");
        assert_eq!(v.name, "treatment");
        assert!(!v.is_control);
        assert_eq!(v.config.get("color"), Some(&"blue".to_string()));
    }

    #[test]
    fn test_variant_control() {
        let v = Variant::control("baseline", 0.5);
        assert!(v.is_control);
    }

    #[test]
    fn test_variant_display() {
        let v = Variant::control("control", 0.5);
        let s = v.to_string();
        assert!(s.contains("control"));
        assert!(s.contains("ctrl"));
    }

    #[test]
    fn test_experiment_status() {
        assert_eq!(ExperimentStatus::Running.to_string(), "running");
        assert_eq!(ExperimentStatus::Draft.to_string(), "draft");
        assert_eq!(ExperimentStatus::Paused.to_string(), "paused");
        assert_eq!(ExperimentStatus::Completed.to_string(), "completed");
        assert_eq!(ExperimentStatus::Archived.to_string(), "archived");
    }

    #[test]
    fn test_experiment_basic() {
        let exp = Experiment::new("exp-001", "Button Color Test")
            .description("Testing blue vs green buttons")
            .status(ExperimentStatus::Running)
            .add_variant(Variant::control("green", 0.5))
            .add_variant(Variant::new("blue", 0.5));

        assert!(exp.is_active());
        assert_eq!(exp.variant_count(), 2);
        assert!(exp.control().is_some());
        assert!((exp.total_weight() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_experiment_not_active_draft() {
        let exp = Experiment::new("e", "test");
        assert!(!exp.is_active());
    }

    #[test]
    fn test_experiment_not_active_no_variants() {
        let exp = Experiment::new("e", "test").status(ExperimentStatus::Running);
        assert!(!exp.is_active());
    }

    #[test]
    fn test_deterministic_assignment() {
        let exp = Experiment::new("exp-001", "Test")
            .status(ExperimentStatus::Running)
            .add_variant(Variant::control("A", 0.5))
            .add_variant(Variant::new("B", 0.5));

        let v1 = exp.assign("user-123").map(|v| v.name.clone());
        let v2 = exp.assign("user-123").map(|v| v.name.clone());
        assert_eq!(v1, v2); // Same user always gets same variant
    }

    #[test]
    fn test_assignment_distribution() {
        let exp = Experiment::new("dist-test", "Distribution")
            .status(ExperimentStatus::Running)
            .add_variant(Variant::control("A", 0.5))
            .add_variant(Variant::new("B", 0.5));

        let mut counts = HashMap::new();
        for i in 0..1000 {
            if let Some(v) = exp.assign(&format!("user-{i}")) {
                *counts.entry(v.name.clone()).or_insert(0u32) += 1;
            }
        }

        // With 50/50 split and 1000 users, expect roughly 400-600 each
        let a_count = *counts.get("A").unwrap_or(&0);
        let b_count = *counts.get("B").unwrap_or(&0);
        assert!(a_count > 300, "A count too low: {a_count}");
        assert!(b_count > 300, "B count too low: {b_count}");
    }

    #[test]
    fn test_traffic_fraction() {
        let exp = Experiment::new("traffic-test", "Limited")
            .status(ExperimentStatus::Running)
            .traffic_fraction(0.1) // Only 10% of traffic
            .add_variant(Variant::control("A", 0.5))
            .add_variant(Variant::new("B", 0.5));

        let mut assigned = 0;
        for i in 0..1000 {
            if exp.assign(&format!("u{i}")).is_some() {
                assigned += 1;
            }
        }

        // ~10% should be assigned (allow some variance)
        assert!(assigned > 50, "too few assigned: {assigned}");
        assert!(assigned < 200, "too many assigned: {assigned}");
    }

    #[test]
    fn test_experiment_validate() {
        let exp = Experiment::new("e", "test")
            .add_variant(Variant::new("only-one", 1.0));
        let issues = exp.validate();
        assert!(issues.iter().any(|i| i.contains("at least 2")));
        assert!(issues.iter().any(|i| i.contains("no control")));
    }

    #[test]
    fn test_experiment_validate_good() {
        let exp = Experiment::new("e", "test")
            .add_variant(Variant::control("A", 0.5))
            .add_variant(Variant::new("B", 0.5));
        let issues = exp.validate();
        assert!(issues.is_empty());
    }

    #[test]
    fn test_normalized_weights() {
        let exp = Experiment::new("e", "t")
            .add_variant(Variant::new("A", 1.0))
            .add_variant(Variant::new("B", 3.0));

        let nw = exp.normalized_weights();
        assert!((nw[0] - 0.25).abs() < 0.001);
        assert!((nw[1] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_hash_determinism() {
        let h1 = hash_assignment("salt", "user-1");
        let h2 = hash_assignment("salt", "user-1");
        assert_eq!(h1, h2);

        let h3 = hash_assignment("salt", "user-2");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_different_salts() {
        let h1 = hash_assignment("exp-1", "user-1");
        let h2 = hash_assignment("exp-2", "user-1");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_to_fraction_range() {
        for i in 0..100 {
            let f = hash_to_fraction(hash_assignment("s", &format!("u{i}")));
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn test_feature_flag_default() {
        let flag = FeatureFlag::new("dark-mode", false);
        assert!(!flag.is_enabled("user-1", None));
    }

    #[test]
    fn test_feature_flag_override() {
        let flag = FeatureFlag::new("beta", false)
            .add_override("vip-user", true);

        assert!(flag.is_enabled("vip-user", None));
        assert!(!flag.is_enabled("regular-user", None));
    }

    #[test]
    fn test_feature_flag_experiment() {
        let flag = FeatureFlag::new("new-checkout", false)
            .with_experiment("exp-1", "treatment");

        assert!(flag.is_enabled("user-1", Some("treatment")));
        assert!(!flag.is_enabled("user-2", Some("control")));
        assert!(!flag.is_enabled("user-3", None));
    }

    #[test]
    fn test_feature_flag_override_priority() {
        let flag = FeatureFlag::new("feature", false)
            .with_experiment("exp", "treatment")
            .add_override("forced-off", false);

        // Override wins over experiment
        assert!(!flag.is_enabled("forced-off", Some("treatment")));
    }

    #[test]
    fn test_variant_result_conversion() {
        let mut vr = VariantResult::new("control");
        for i in 0..100 {
            vr.record_conversion(i < 40); // 40% conversion
        }

        assert_eq!(vr.sample_size, 100);
        assert_eq!(vr.conversions, 40);
        assert!((vr.conversion_rate() - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_variant_result_continuous() {
        let mut vr = VariantResult::new("treatment");
        vr.record_value(10.0);
        vr.record_value(20.0);
        vr.record_value(30.0);

        assert_eq!(vr.sample_size, 3);
        assert!((vr.mean() - 20.0).abs() < 0.001);
        assert!(vr.std_dev() > 0.0);
        assert!(vr.std_error() > 0.0);
    }

    #[test]
    fn test_variant_result_empty() {
        let vr = VariantResult::new("v");
        assert_eq!(vr.conversion_rate(), 0.0);
        assert_eq!(vr.mean(), 0.0);
        assert_eq!(vr.variance(), 0.0);
        assert_eq!(vr.std_error(), 0.0);
    }

    #[test]
    fn test_significance_test_clear_winner() {
        let mut control = VariantResult::new("control");
        let mut treatment = VariantResult::new("treatment");

        // Control: 10% conversion, Treatment: 20%
        for i in 0..1000 { control.record_conversion(i < 100); }
        for i in 0..1000 { treatment.record_conversion(i < 200); }

        let result = significance_test_proportions(&control, &treatment, 0.95);
        assert!(result.is_significant);
        assert!(result.lift > 0.0);
    }

    #[test]
    fn test_significance_test_no_difference() {
        let mut control = VariantResult::new("control");
        let mut treatment = VariantResult::new("treatment");

        // Both ~50% conversion (small sample)
        for i in 0..20 { control.record_conversion(i < 10); }
        for i in 0..20 { treatment.record_conversion(i < 10); }

        let result = significance_test_proportions(&control, &treatment, 0.95);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_significance_test_small_sample() {
        let control = VariantResult::new("c");
        let treatment = VariantResult::new("t");
        let result = significance_test_proportions(&control, &treatment, 0.95);
        assert!(!result.is_significant);
        assert!((result.p_value_approx - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_significance_test_means() {
        let mut control = VariantResult::new("control");
        let mut treatment = VariantResult::new("treatment");

        // Use values with variance so the z-test can detect significance.
        // Control mean ~10, treatment mean ~15, both with std dev ~2.
        for i in 0..500 {
            control.record_value(8.0 + (i % 5) as f64);   // values 8..12, mean=10
        }
        for i in 0..500 {
            treatment.record_value(13.0 + (i % 5) as f64); // values 13..17, mean=15
        }

        let result = significance_test_means(&control, &treatment, 0.95);
        assert!(result.is_significant);
        assert!(result.lift > 0.0);
    }

    #[test]
    fn test_required_sample_size() {
        let n = required_sample_size(0.10, 0.02, 0.95, 0.80);
        assert!(n > 0);
        assert!(n > 100); // Need substantial sample for 2pp lift on 10% base
    }

    #[test]
    fn test_required_sample_size_zero_effect() {
        let n = required_sample_size(0.10, 0.0, 0.95, 0.80);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_significance_result_display() {
        let r = SignificanceResult {
            z_score: 2.5,
            p_value_approx: 0.012,
            is_significant: true,
            confidence_level: 0.95,
            lift: 0.25,
        };
        let s = r.to_string();
        assert!(s.contains("z=2.500"));
        assert!(s.contains("sig=yes"));
    }

    #[test]
    fn test_experiment_salt() {
        let exp = Experiment::new("e", "test")
            .salt("custom-salt")
            .status(ExperimentStatus::Running)
            .add_variant(Variant::control("A", 0.5))
            .add_variant(Variant::new("B", 0.5));

        assert_eq!(exp.salt, "custom-salt");
        // Should still assign deterministically
        let v1 = exp.assign("user-x").map(|v| v.name.clone());
        let v2 = exp.assign("user-x").map(|v| v.name.clone());
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_inactive_experiment_returns_none() {
        let exp = Experiment::new("e", "test")
            .status(ExperimentStatus::Paused)
            .add_variant(Variant::control("A", 0.5))
            .add_variant(Variant::new("B", 0.5));

        assert!(exp.assign("user-1").is_none());
    }

    #[test]
    fn test_three_way_split() {
        let exp = Experiment::new("three-way", "Three Way")
            .status(ExperimentStatus::Running)
            .add_variant(Variant::control("A", 1.0))
            .add_variant(Variant::new("B", 1.0))
            .add_variant(Variant::new("C", 1.0));

        let mut counts: HashMap<String, u32> = HashMap::new();
        for i in 0..3000 {
            if let Some(v) = exp.assign(&format!("u{i}")) {
                *counts.entry(v.name.clone()).or_insert(0) += 1;
            }
        }

        // Each should get roughly 1000 +/- 200
        for name in &["A", "B", "C"] {
            let c = *counts.get(*name).unwrap_or(&0);
            assert!(c > 700, "{name} count too low: {c}");
            assert!(c < 1300, "{name} count too high: {c}");
        }
    }
}
