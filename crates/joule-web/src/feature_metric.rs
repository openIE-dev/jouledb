//! Feature usage metrics: feature flag evaluation tracking, A/B test impression
//! counting, conversion tracking, funnel analysis, cohort comparison,
//! statistical significance testing (chi-squared), and feature adoption rate.

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// A feature flag evaluation event.
#[derive(Debug, Clone)]
pub struct FlagEvaluation {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub flag_name: String,
    pub variant: String,
    pub user_id: String,
    pub context: HashMap<String, String>,
}

impl FlagEvaluation {
    pub fn new(flag_name: &str, variant: &str, user_id: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            flag_name: flag_name.to_string(),
            variant: variant.to_string(),
            user_id: user_id.to_string(),
            context: HashMap::new(),
        }
    }

    pub fn with_context(mut self, key: &str, value: &str) -> Self {
        self.context.insert(key.to_string(), value.to_string());
        self
    }
}

/// Impression counter for A/B test variants.
#[derive(Debug, Clone)]
pub struct ImpressionCounter {
    /// experiment -> variant -> count
    counts: HashMap<String, HashMap<String, u64>>,
    /// experiment -> variant -> set of user_ids
    unique_users: HashMap<String, HashMap<String, Vec<String>>>,
}

impl ImpressionCounter {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            unique_users: HashMap::new(),
        }
    }

    pub fn record(&mut self, experiment: &str, variant: &str, user_id: &str) {
        *self
            .counts
            .entry(experiment.to_string())
            .or_default()
            .entry(variant.to_string())
            .or_insert(0) += 1;

        let users = self
            .unique_users
            .entry(experiment.to_string())
            .or_default()
            .entry(variant.to_string())
            .or_default();
        if !users.contains(&user_id.to_string()) {
            users.push(user_id.to_string());
        }
    }

    pub fn impression_count(&self, experiment: &str, variant: &str) -> u64 {
        self.counts
            .get(experiment)
            .and_then(|v| v.get(variant))
            .copied()
            .unwrap_or(0)
    }

    pub fn unique_user_count(&self, experiment: &str, variant: &str) -> usize {
        self.unique_users
            .get(experiment)
            .and_then(|v| v.get(variant))
            .map(|u| u.len())
            .unwrap_or(0)
    }

    pub fn total_impressions(&self, experiment: &str) -> u64 {
        self.counts
            .get(experiment)
            .map(|variants| variants.values().sum())
            .unwrap_or(0)
    }
}

/// Conversion event.
#[derive(Debug, Clone)]
pub struct ConversionEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub experiment: String,
    pub variant: String,
    pub user_id: String,
    pub goal: String,
    pub value: Option<f64>,
}

impl ConversionEvent {
    pub fn new(experiment: &str, variant: &str, user_id: &str, goal: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            experiment: experiment.to_string(),
            variant: variant.to_string(),
            user_id: user_id.to_string(),
            goal: goal.to_string(),
            value: None,
        }
    }

    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }
}

/// Conversion tracker: records conversions and computes rates.
#[derive(Debug)]
pub struct ConversionTracker {
    pub impressions: ImpressionCounter,
    pub conversions: Vec<ConversionEvent>,
}

impl ConversionTracker {
    pub fn new() -> Self {
        Self {
            impressions: ImpressionCounter::new(),
            conversions: Vec::new(),
        }
    }

    pub fn record_impression(&mut self, experiment: &str, variant: &str, user_id: &str) {
        self.impressions.record(experiment, variant, user_id);
    }

    pub fn record_conversion(&mut self, event: ConversionEvent) {
        self.conversions.push(event);
    }

    /// Conversion count for an experiment/variant/goal.
    pub fn conversion_count(&self, experiment: &str, variant: &str, goal: &str) -> usize {
        self.conversions
            .iter()
            .filter(|c| c.experiment == experiment && c.variant == variant && c.goal == goal)
            .count()
    }

    /// Unique converting users for experiment/variant/goal.
    pub fn unique_conversions(&self, experiment: &str, variant: &str, goal: &str) -> usize {
        let mut seen = Vec::new();
        for c in &self.conversions {
            if c.experiment == experiment && c.variant == variant && c.goal == goal {
                if !seen.contains(&c.user_id) {
                    seen.push(c.user_id.clone());
                }
            }
        }
        seen.len()
    }

    /// Conversion rate for a variant.
    pub fn conversion_rate(&self, experiment: &str, variant: &str, goal: &str) -> f64 {
        let impressions = self.impressions.unique_user_count(experiment, variant);
        if impressions == 0 {
            return 0.0;
        }
        let conversions = self.unique_conversions(experiment, variant, goal);
        conversions as f64 / impressions as f64
    }
}

// ── Funnel Analysis ──

/// A funnel step definition.
#[derive(Debug, Clone)]
pub struct FunnelStep {
    pub name: String,
    pub count: u64,
}

/// Funnel analysis: tracks users through a sequence of steps.
#[derive(Debug, Clone)]
pub struct Funnel {
    pub name: String,
    pub steps: Vec<FunnelStep>,
}

impl Funnel {
    pub fn new(name: &str, step_names: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            steps: step_names
                .iter()
                .map(|s| FunnelStep {
                    name: s.to_string(),
                    count: 0,
                })
                .collect(),
        }
    }

    pub fn record_step(&mut self, step_index: usize, count: u64) {
        if step_index < self.steps.len() {
            self.steps[step_index].count += count;
        }
    }

    /// Set absolute counts for each step.
    pub fn set_counts(&mut self, counts: &[u64]) {
        for (i, &c) in counts.iter().enumerate() {
            if i < self.steps.len() {
                self.steps[i].count = c;
            }
        }
    }

    /// Conversion rate between two adjacent steps.
    pub fn step_conversion_rate(&self, from: usize, to: usize) -> f64 {
        if from >= self.steps.len() || to >= self.steps.len() || self.steps[from].count == 0 {
            return 0.0;
        }
        self.steps[to].count as f64 / self.steps[from].count as f64
    }

    /// Overall funnel conversion rate (first step to last).
    pub fn overall_rate(&self) -> f64 {
        if self.steps.is_empty() || self.steps[0].count == 0 {
            return 0.0;
        }
        let last = self.steps.last().unwrap();
        last.count as f64 / self.steps[0].count as f64
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "steps": self.steps.iter().enumerate().map(|(i, s)| {
                let drop_off = if i > 0 && self.steps[i-1].count > 0 {
                    1.0 - (s.count as f64 / self.steps[i-1].count as f64)
                } else {
                    0.0
                };
                json!({
                    "name": s.name,
                    "count": s.count,
                    "drop_off_rate": drop_off,
                })
            }).collect::<Vec<_>>(),
            "overall_rate": self.overall_rate(),
        })
    }
}

// ── Cohort Comparison ──

/// A cohort for comparison.
#[derive(Debug, Clone)]
pub struct Cohort {
    pub name: String,
    pub user_count: u64,
    pub conversion_count: u64,
}

impl Cohort {
    pub fn new(name: &str, users: u64, conversions: u64) -> Self {
        Self {
            name: name.to_string(),
            user_count: users,
            conversion_count: conversions,
        }
    }

    pub fn conversion_rate(&self) -> f64 {
        if self.user_count == 0 {
            return 0.0;
        }
        self.conversion_count as f64 / self.user_count as f64
    }
}

/// Compare two cohorts.
pub fn cohort_lift(control: &Cohort, treatment: &Cohort) -> f64 {
    let cr = control.conversion_rate();
    if cr == 0.0 {
        return 0.0;
    }
    (treatment.conversion_rate() - cr) / cr
}

// ── Chi-Squared Test ──

/// Perform a chi-squared test for two cohorts (2x2 contingency table).
/// Returns (chi2_statistic, p_value_approximate, is_significant_at_005).
pub fn chi_squared_test(control: &Cohort, treatment: &Cohort) -> (f64, f64, bool) {
    let a = control.conversion_count as f64;       // control converted
    let b = (control.user_count - control.conversion_count) as f64; // control not
    let c = treatment.conversion_count as f64;      // treatment converted
    let d = (treatment.user_count - treatment.conversion_count) as f64; // treatment not

    let n = a + b + c + d;
    if n == 0.0 {
        return (0.0, 1.0, false);
    }

    // Yates-corrected chi-squared for 2x2 table:
    // chi2 = n * (|ad - bc| - n/2)^2 / ((a+b)(c+d)(a+c)(b+d))
    let ad_bc = (a * d - b * c).abs();
    let correction = n / 2.0;
    let numerator = if ad_bc > correction {
        n * (ad_bc - correction).powi(2)
    } else {
        0.0
    };
    let denom = (a + b) * (c + d) * (a + c) * (b + d);
    if denom == 0.0 {
        return (0.0, 1.0, false);
    }
    let chi2 = numerator / denom;

    // Approximate p-value for 1 degree of freedom using survival function.
    let p = chi2_survival(chi2);
    (chi2, p, p < 0.05)
}

/// Approximate survival function for chi-squared distribution with 1 df.
/// Uses the Wilson-Hilferty approximation via the normal CDF.
fn chi2_survival(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // For 1 df: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    let z = x.sqrt();
    let phi = normal_cdf(z);
    2.0 * (1.0 - phi)
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp();
    let poly = t * (0.319381530
        + t * (-0.356563782
            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - p * poly
    } else {
        p * poly
    }
}

// ── Feature Adoption Rate ──

/// Track feature adoption over time.
#[derive(Debug)]
pub struct FeatureAdoption {
    pub feature_name: String,
    pub total_users: u64,
    pub adopters: Vec<String>,
    pub daily_counts: Vec<(DateTime<Utc>, u64)>,
}

impl FeatureAdoption {
    pub fn new(feature_name: &str, total_users: u64) -> Self {
        Self {
            feature_name: feature_name.to_string(),
            total_users,
            adopters: Vec::new(),
            daily_counts: Vec::new(),
        }
    }

    pub fn record_adoption(&mut self, user_id: &str) {
        if !self.adopters.contains(&user_id.to_string()) {
            self.adopters.push(user_id.to_string());
        }
    }

    pub fn record_daily_count(&mut self, timestamp: DateTime<Utc>, count: u64) {
        self.daily_counts.push((timestamp, count));
    }

    pub fn adoption_rate(&self) -> f64 {
        if self.total_users == 0 {
            return 0.0;
        }
        self.adopters.len() as f64 / self.total_users as f64
    }

    pub fn adopter_count(&self) -> usize {
        self.adopters.len()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "feature": self.feature_name,
            "total_users": self.total_users,
            "adopters": self.adopters.len(),
            "adoption_rate_pct": self.adoption_rate() * 100.0,
        })
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_evaluation() {
        let eval = FlagEvaluation::new("dark_mode", "enabled", "user-1")
            .with_context("platform", "ios");
        assert_eq!(eval.flag_name, "dark_mode");
        assert_eq!(eval.variant, "enabled");
        assert_eq!(eval.context["platform"], "ios");
    }

    #[test]
    fn test_impression_counter() {
        let mut ic = ImpressionCounter::new();
        ic.record("exp1", "control", "u1");
        ic.record("exp1", "control", "u1"); // duplicate user
        ic.record("exp1", "control", "u2");
        ic.record("exp1", "treatment", "u3");
        assert_eq!(ic.impression_count("exp1", "control"), 3);
        assert_eq!(ic.unique_user_count("exp1", "control"), 2);
        assert_eq!(ic.total_impressions("exp1"), 4);
    }

    #[test]
    fn test_conversion_tracker() {
        let mut tracker = ConversionTracker::new();
        tracker.record_impression("exp1", "control", "u1");
        tracker.record_impression("exp1", "control", "u2");
        tracker.record_impression("exp1", "treatment", "u3");
        tracker.record_impression("exp1", "treatment", "u4");

        tracker.record_conversion(ConversionEvent::new("exp1", "control", "u1", "signup"));
        tracker.record_conversion(ConversionEvent::new("exp1", "treatment", "u3", "signup"));
        tracker.record_conversion(ConversionEvent::new("exp1", "treatment", "u4", "signup"));

        assert_eq!(tracker.conversion_count("exp1", "control", "signup"), 1);
        assert!((tracker.conversion_rate("exp1", "control", "signup") - 0.5).abs() < 1e-10);
        assert!((tracker.conversion_rate("exp1", "treatment", "signup") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_funnel_analysis() {
        let mut funnel = Funnel::new("signup", &["visit", "register", "activate", "subscribe"]);
        funnel.set_counts(&[1000, 500, 200, 50]);
        assert!((funnel.step_conversion_rate(0, 1) - 0.5).abs() < 1e-10);
        assert!((funnel.step_conversion_rate(1, 2) - 0.4).abs() < 1e-10);
        assert!((funnel.overall_rate() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_funnel_json() {
        let mut funnel = Funnel::new("f", &["a", "b"]);
        funnel.set_counts(&[100, 40]);
        let j = funnel.to_json();
        assert_eq!(j["overall_rate"], 0.4);
    }

    #[test]
    fn test_cohort_comparison() {
        let control = Cohort::new("control", 1000, 100);
        let treatment = Cohort::new("treatment", 1000, 150);
        assert!((control.conversion_rate() - 0.1).abs() < 1e-10);
        assert!((treatment.conversion_rate() - 0.15).abs() < 1e-10);
        let lift = cohort_lift(&control, &treatment);
        assert!((lift - 0.5).abs() < 1e-10); // 50% lift
    }

    #[test]
    fn test_chi_squared_significant() {
        // Large sample, big difference => significant.
        let control = Cohort::new("control", 10000, 500);
        let treatment = Cohort::new("treatment", 10000, 700);
        let (chi2, p, sig) = chi_squared_test(&control, &treatment);
        assert!(chi2 > 0.0);
        assert!(p < 0.05);
        assert!(sig);
    }

    #[test]
    fn test_chi_squared_not_significant() {
        // Small sample, tiny difference => not significant.
        let control = Cohort::new("control", 20, 5);
        let treatment = Cohort::new("treatment", 20, 6);
        let (_, _, sig) = chi_squared_test(&control, &treatment);
        assert!(!sig);
    }

    #[test]
    fn test_chi_squared_zero_counts() {
        let control = Cohort::new("control", 0, 0);
        let treatment = Cohort::new("treatment", 0, 0);
        let (chi2, p, sig) = chi_squared_test(&control, &treatment);
        assert_eq!(chi2, 0.0);
        assert_eq!(p, 1.0);
        assert!(!sig);
    }

    #[test]
    fn test_feature_adoption() {
        let mut fa = FeatureAdoption::new("dark_mode", 1000);
        fa.record_adoption("u1");
        fa.record_adoption("u2");
        fa.record_adoption("u1"); // duplicate
        assert_eq!(fa.adopter_count(), 2);
        assert!((fa.adoption_rate() - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_feature_adoption_json() {
        let mut fa = FeatureAdoption::new("feature_x", 100);
        fa.record_adoption("u1");
        let j = fa.to_json();
        assert_eq!(j["feature"], "feature_x");
        assert_eq!(j["adopters"], 1);
        assert_eq!(j["adoption_rate_pct"], 1.0);
    }

    #[test]
    fn test_conversion_event_with_value() {
        let ev = ConversionEvent::new("exp1", "v1", "u1", "purchase").with_value(49.99);
        assert_eq!(ev.value, Some(49.99));
    }

    #[test]
    fn test_unique_conversions() {
        let mut tracker = ConversionTracker::new();
        tracker.record_impression("e", "v", "u1");
        tracker.record_conversion(ConversionEvent::new("e", "v", "u1", "g"));
        tracker.record_conversion(ConversionEvent::new("e", "v", "u1", "g")); // same user
        assert_eq!(tracker.unique_conversions("e", "v", "g"), 1);
        assert_eq!(tracker.conversion_count("e", "v", "g"), 2);
    }
}
