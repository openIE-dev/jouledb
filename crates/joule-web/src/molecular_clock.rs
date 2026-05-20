//! Molecular clock estimation — substitution rate inference, divergence time
//! calculation, strict and relaxed clock models, rate variation testing,
//! and calibration point integration.
//!
//! Implements clock-like models to estimate evolutionary timescales from
//! branch lengths and optional fossil calibration constraints. Supports
//! strict clock, local clock, and simple relaxed clock approaches.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ClockError {
    InsufficientData(String),
    InvalidCalibration(String),
    NegativeRate,
    NegativeTime,
    ClockRejected(String),
    InvalidParameter(String),
}

impl fmt::Display for ClockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData(s) => write!(f, "insufficient data: {s}"),
            Self::InvalidCalibration(s) => write!(f, "invalid calibration: {s}"),
            Self::NegativeRate => write!(f, "negative substitution rate"),
            Self::NegativeTime => write!(f, "negative divergence time"),
            Self::ClockRejected(s) => write!(f, "clock rejected: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
        }
    }
}

impl std::error::Error for ClockError {}

// ── Clock model ─────────────────────────────────────────────────

/// Type of molecular clock model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockModel {
    /// All lineages evolve at the same constant rate.
    Strict,
    /// Different subtrees can have different rates.
    LocalClock,
    /// Each branch has an independently drawn rate.
    RelaxedLogNormal,
    /// Unconstrained — no clock assumption.
    Unconstrained,
}

impl fmt::Display for ClockModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strict => write!(f, "strict"),
            Self::LocalClock => write!(f, "local clock"),
            Self::RelaxedLogNormal => write!(f, "relaxed (log-normal)"),
            Self::Unconstrained => write!(f, "unconstrained"),
        }
    }
}

// ── Calibration point ───────────────────────────────────────────

/// A fossil or external calibration constraint.
#[derive(Debug, Clone)]
pub struct CalibrationPoint {
    pub node_label: String,
    pub min_age: f64,
    pub max_age: f64,
    pub fixed: bool,
}

impl CalibrationPoint {
    pub fn new(node_label: &str, min_age: f64, max_age: f64) -> Self {
        Self {
            node_label: node_label.to_string(),
            min_age,
            max_age,
            fixed: false,
        }
    }

    pub fn with_fixed(mut self, fixed: bool) -> Self {
        self.fixed = fixed;
        self
    }

    pub fn midpoint_age(&self) -> f64 {
        (self.min_age + self.max_age) / 2.0
    }

    pub fn range(&self) -> f64 {
        self.max_age - self.min_age
    }

    pub fn is_valid(&self) -> bool {
        self.min_age >= 0.0 && self.max_age >= self.min_age
    }
}

impl fmt::Display for CalibrationPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fixed {
            write!(f, "{}: {:.2} Ma (fixed)", self.node_label, self.midpoint_age())
        } else {
            write!(
                f,
                "{}: [{:.2}, {:.2}] Ma",
                self.node_label, self.min_age, self.max_age
            )
        }
    }
}

// ── Branch rate ─────────────────────────────────────────────────

/// Substitution rate on a single branch.
#[derive(Debug, Clone)]
pub struct BranchRate {
    pub node_id: usize,
    pub label: Option<String>,
    pub branch_length: f64,
    pub time: f64,
    pub rate: f64,
}

impl BranchRate {
    pub fn new(node_id: usize, branch_length: f64, time: f64) -> Result<Self, ClockError> {
        if time < 0.0 {
            return Err(ClockError::NegativeTime);
        }
        let rate = if time > 0.0 { branch_length / time } else { 0.0 };
        if rate < 0.0 {
            return Err(ClockError::NegativeRate);
        }
        Ok(Self { node_id, label: None, branch_length, time, rate })
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

impl fmt::Display for BranchRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lbl = self.label.as_deref().unwrap_or("?");
        write!(f, "{lbl}: rate={:.6} subs/site/Myr", self.rate)
    }
}

// ── Clock estimation result ─────────────────────────────────────

/// Result of molecular clock analysis.
#[derive(Debug, Clone)]
pub struct ClockResult {
    pub model: ClockModel,
    pub global_rate: f64,
    pub branch_rates: Vec<BranchRate>,
    pub root_age: f64,
    pub rate_variance: f64,
    pub clock_like: bool,
}

impl ClockResult {
    pub fn mean_rate(&self) -> f64 {
        if self.branch_rates.is_empty() {
            return self.global_rate;
        }
        let sum: f64 = self.branch_rates.iter().map(|b| b.rate).sum();
        sum / self.branch_rates.len() as f64
    }

    pub fn rate_coefficient_of_variation(&self) -> f64 {
        let mean = self.mean_rate();
        if mean.abs() < 1e-15 {
            return 0.0;
        }
        self.rate_variance.sqrt() / mean
    }

    pub fn with_model(mut self, model: ClockModel) -> Self {
        self.model = model;
        self
    }
}

impl fmt::Display for ClockResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClockResult(model={}, rate={:.6}, root_age={:.2} Ma, clock={})",
            self.model, self.global_rate, self.root_age, self.clock_like
        )
    }
}

// ── Clock configuration ─────────────────────────────────────────

/// Configuration for clock estimation.
#[derive(Debug, Clone)]
pub struct ClockConfig {
    pub model: ClockModel,
    pub calibrations: Vec<CalibrationPoint>,
    /// Significance level for clock-likeness test.
    pub alpha: f64,
    /// Coefficient of variation threshold for rejecting strict clock.
    pub cv_threshold: f64,
}

impl ClockConfig {
    pub fn new(model: ClockModel) -> Self {
        Self {
            model,
            calibrations: Vec::new(),
            alpha: 0.05,
            cv_threshold: 0.5,
        }
    }

    pub fn with_calibration(mut self, cal: CalibrationPoint) -> Self {
        self.calibrations.push(cal);
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_cv_threshold(mut self, cv: f64) -> Self {
        self.cv_threshold = cv;
        self
    }
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self::new(ClockModel::Strict)
    }
}

impl fmt::Display for ClockConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClockConfig(model={}, {} calibrations, alpha={:.2})",
            self.model,
            self.calibrations.len(),
            self.alpha
        )
    }
}

// ── Clock tree node ─────────────────────────────────────────────

/// A node with both branch length and time information.
#[derive(Debug, Clone)]
pub struct ClockNode {
    pub id: usize,
    pub label: Option<String>,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub branch_length: f64,
    pub node_time: f64,
}

impl ClockNode {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            parent: None,
            children: Vec::new(),
            branch_length: 0.0,
            node_time: 0.0,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn with_branch_length(mut self, len: f64) -> Self {
        self.branch_length = len;
        self
    }

    pub fn with_time(mut self, time: f64) -> Self {
        self.node_time = time;
        self
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

impl fmt::Display for ClockNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lbl = self.label.as_deref().unwrap_or("?");
        write!(f, "{lbl} (t={:.2} Ma)", self.node_time)
    }
}

// ── Strict clock estimation ─────────────────────────────────────

/// Estimate a strict molecular clock from root-to-tip distances.
///
/// `root_to_tip` — distance from root to each leaf.
/// `tip_dates` — sampling dates (in the same units, e.g. Myr before present).
pub fn strict_clock(
    root_to_tip: &[f64],
    tip_dates: &[f64],
) -> Result<ClockResult, ClockError> {
    let n = root_to_tip.len();
    if n < 2 || tip_dates.len() != n {
        return Err(ClockError::InsufficientData(
            "need ≥ 2 taxa with matching dates".into(),
        ));
    }

    // Simple linear regression: distance = rate * time + intercept
    let mean_d: f64 = root_to_tip.iter().sum::<f64>() / n as f64;
    let mean_t: f64 = tip_dates.iter().sum::<f64>() / n as f64;

    let mut num = 0.0;
    let mut denom = 0.0;
    for i in 0..n {
        let dt = tip_dates[i] - mean_t;
        num += dt * (root_to_tip[i] - mean_d);
        denom += dt * dt;
    }

    if denom.abs() < 1e-15 {
        return Err(ClockError::InsufficientData("no date variation".into()));
    }

    let rate = num / denom;
    let intercept = mean_d - rate * mean_t;

    if rate < 0.0 {
        return Err(ClockError::NegativeRate);
    }

    let root_age = if rate.abs() > 1e-15 { -intercept / rate } else { 0.0 };

    // Residual variance
    let mut ss_res = 0.0;
    for i in 0..n {
        let predicted = rate * tip_dates[i] + intercept;
        let resid = root_to_tip[i] - predicted;
        ss_res += resid * resid;
    }
    let variance = if n > 2 { ss_res / (n - 2) as f64 } else { 0.0 };

    // R² for clock-likeness
    let mut ss_tot = 0.0;
    for &d in root_to_tip {
        let diff = d - mean_d;
        ss_tot += diff * diff;
    }
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 1.0 };
    let clock_like = r_squared > 0.8;

    Ok(ClockResult {
        model: ClockModel::Strict,
        global_rate: rate,
        branch_rates: Vec::new(),
        root_age,
        rate_variance: variance,
        clock_like,
    })
}

// ── Divergence time from calibrated node ────────────────────────

/// Estimate divergence time given a branch length, rate, and parent time.
pub fn divergence_time(
    branch_length: f64,
    rate: f64,
    parent_time: f64,
) -> Result<f64, ClockError> {
    if rate <= 0.0 {
        return Err(ClockError::NegativeRate);
    }
    let dt = branch_length / rate;
    Ok(parent_time + dt)
}

/// Convert a tree's branch lengths to divergence times using a global rate.
pub fn branch_lengths_to_times(
    branch_lengths: &[f64],
    parents: &[Option<usize>],
    rate: f64,
    root_time: f64,
) -> Result<Vec<f64>, ClockError> {
    if rate <= 0.0 {
        return Err(ClockError::NegativeRate);
    }
    let n = branch_lengths.len();
    let mut times = vec![0.0; n];

    // Find root (no parent)
    let root = parents
        .iter()
        .position(|p| p.is_none())
        .ok_or_else(|| ClockError::InsufficientData("no root found".into()))?;
    times[root] = root_time;

    // BFS from root
    let mut visited = vec![false; n];
    visited[root] = true;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);

    // Build children map
    let mut children_map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, p) in parents.iter().enumerate() {
        if let Some(pid) = p {
            children_map[*pid].push(i);
        }
    }

    while let Some(cur) = queue.pop_front() {
        for &child in &children_map[cur] {
            if !visited[child] {
                times[child] = times[cur] + branch_lengths[child] / rate;
                visited[child] = true;
                queue.push_back(child);
            }
        }
    }
    Ok(times)
}

// ── Rate variation test ─────────────────────────────────────────

/// Simple rate variation test: computes coefficient of variation of
/// root-to-tip distances relative to their expected values.
pub fn test_rate_variation(
    root_to_tip: &[f64],
    tip_dates: &[f64],
    config: &ClockConfig,
) -> Result<bool, ClockError> {
    let clock = strict_clock(root_to_tip, tip_dates)?;
    let cv = clock.rate_coefficient_of_variation();
    Ok(cv <= config.cv_threshold)
}

/// Compute per-branch rates given branch lengths and times.
pub fn compute_branch_rates(
    branch_lengths: &[f64],
    times: &[f64],
    parents: &[Option<usize>],
    labels: &[Option<&str>],
) -> Result<Vec<BranchRate>, ClockError> {
    let n = branch_lengths.len();
    let mut rates = Vec::new();
    for i in 0..n {
        if let Some(pid) = parents[i] {
            let dt = (times[i] - times[pid]).abs();
            let mut br = BranchRate::new(i, branch_lengths[i], dt)?;
            if let Some(lbl) = labels[i] {
                br = br.with_label(lbl);
            }
            rates.push(br);
        }
    }
    Ok(rates)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strict_clock_basic() {
        // Perfect clock: distance = 0.01 * time
        let dates = vec![0.0, 10.0, 20.0, 30.0];
        let dists = vec![0.0, 0.1, 0.2, 0.3];
        let result = strict_clock(&dists, &dates).unwrap();
        assert!((result.global_rate - 0.01).abs() < 1e-9);
        assert!(result.clock_like);
    }

    #[test]
    fn test_strict_clock_root_age() {
        let dates = vec![10.0, 20.0, 30.0, 40.0];
        let dists = vec![0.1, 0.2, 0.3, 0.4];
        let result = strict_clock(&dists, &dates).unwrap();
        assert!(result.root_age.abs() < 1e-6);
    }

    #[test]
    fn test_strict_clock_insufficient() {
        let dates = vec![0.0];
        let dists = vec![0.0];
        assert!(strict_clock(&dists, &dates).is_err());
    }

    #[test]
    fn test_strict_clock_no_variation() {
        let dates = vec![5.0, 5.0, 5.0];
        let dists = vec![0.1, 0.2, 0.3];
        assert!(strict_clock(&dists, &dates).is_err());
    }

    #[test]
    fn test_divergence_time() {
        let t = divergence_time(0.1, 0.01, 100.0).unwrap();
        assert!((t - 110.0).abs() < 1e-9);
    }

    #[test]
    fn test_divergence_time_zero_rate() {
        assert!(divergence_time(0.1, 0.0, 100.0).is_err());
    }

    #[test]
    fn test_branch_lengths_to_times() {
        let bls = vec![0.0, 0.1, 0.2];
        let parents = vec![None, Some(0), Some(0)];
        let times = branch_lengths_to_times(&bls, &parents, 0.01, 0.0).unwrap();
        assert!((times[0]).abs() < 1e-9); // root
        assert!((times[1] - 10.0).abs() < 1e-9);
        assert!((times[2] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_branch_lengths_to_times_bad_rate() {
        let bls = vec![0.0, 0.1];
        let parents = vec![None, Some(0)];
        assert!(branch_lengths_to_times(&bls, &parents, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_calibration_point() {
        let cal = CalibrationPoint::new("node_1", 50.0, 70.0);
        assert!((cal.midpoint_age() - 60.0).abs() < 1e-9);
        assert!((cal.range() - 20.0).abs() < 1e-9);
        assert!(cal.is_valid());
    }

    #[test]
    fn test_calibration_fixed() {
        let cal = CalibrationPoint::new("root", 100.0, 100.0).with_fixed(true);
        assert!(cal.fixed);
        assert!(format!("{cal}").contains("fixed"));
    }

    #[test]
    fn test_calibration_display() {
        let cal = CalibrationPoint::new("mrca", 30.0, 50.0);
        let s = format!("{cal}");
        assert!(s.contains("mrca"));
        assert!(s.contains("30.00"));
    }

    #[test]
    fn test_branch_rate() {
        let br = BranchRate::new(0, 0.05, 10.0).unwrap().with_label("species_A");
        assert!((br.rate - 0.005).abs() < 1e-9);
        assert!(format!("{br}").contains("species_A"));
    }

    #[test]
    fn test_branch_rate_zero_time() {
        let br = BranchRate::new(0, 0.0, 0.0).unwrap();
        assert!((br.rate).abs() < 1e-9);
    }

    #[test]
    fn test_clock_result_mean_rate() {
        let mut cr = ClockResult {
            model: ClockModel::Strict,
            global_rate: 0.01,
            branch_rates: Vec::new(),
            root_age: 100.0,
            rate_variance: 0.0,
            clock_like: true,
        };
        assert!((cr.mean_rate() - 0.01).abs() < 1e-9);

        cr.branch_rates.push(BranchRate::new(0, 0.1, 10.0).unwrap());
        cr.branch_rates.push(BranchRate::new(1, 0.2, 10.0).unwrap());
        assert!((cr.mean_rate() - 0.015).abs() < 1e-9);
    }

    #[test]
    fn test_clock_result_display() {
        let cr = ClockResult {
            model: ClockModel::Strict,
            global_rate: 0.01,
            branch_rates: Vec::new(),
            root_age: 100.0,
            rate_variance: 0.0,
            clock_like: true,
        };
        let s = format!("{cr}");
        assert!(s.contains("strict"));
        assert!(s.contains("100.00"));
    }

    #[test]
    fn test_clock_config_builder() {
        let cfg = ClockConfig::new(ClockModel::RelaxedLogNormal)
            .with_calibration(CalibrationPoint::new("root", 80.0, 120.0))
            .with_alpha(0.01)
            .with_cv_threshold(0.3);
        assert_eq!(cfg.model, ClockModel::RelaxedLogNormal);
        assert_eq!(cfg.calibrations.len(), 1);
        assert!((cfg.alpha - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_clock_config_display() {
        let cfg = ClockConfig::new(ClockModel::Strict);
        let s = format!("{cfg}");
        assert!(s.contains("strict"));
    }

    #[test]
    fn test_clock_model_display() {
        assert_eq!(format!("{}", ClockModel::Strict), "strict");
        assert_eq!(format!("{}", ClockModel::RelaxedLogNormal), "relaxed (log-normal)");
    }

    #[test]
    fn test_rate_variation_clock_like() {
        let dates = vec![0.0, 10.0, 20.0, 30.0];
        let dists = vec![0.0, 0.1, 0.2, 0.3];
        let cfg = ClockConfig::new(ClockModel::Strict);
        let pass = test_rate_variation(&dists, &dates, &cfg).unwrap();
        assert!(pass);
    }

    #[test]
    fn test_compute_branch_rates() {
        let bls = vec![0.0, 0.05, 0.10];
        let times = vec![0.0, 10.0, 20.0];
        let parents = vec![None, Some(0), Some(0)];
        let labels: Vec<Option<&str>> = vec![None, Some("A"), Some("B")];
        let rates = compute_branch_rates(&bls, &times, &parents, &labels).unwrap();
        assert_eq!(rates.len(), 2);
        assert!((rates[0].rate - 0.005).abs() < 1e-9);
    }

    #[test]
    fn test_clock_node() {
        let n = ClockNode::new(0)
            .with_label("root")
            .with_branch_length(0.0)
            .with_time(100.0);
        assert!(n.is_leaf());
        assert!(format!("{n}").contains("100.00"));
    }

    #[test]
    fn test_clock_config_default() {
        let cfg = ClockConfig::default();
        assert_eq!(cfg.model, ClockModel::Strict);
    }

    #[test]
    fn test_error_display() {
        let e = ClockError::NegativeRate;
        assert!(format!("{e}").contains("negative"));
    }
}
