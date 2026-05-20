//! SLI/SLO monitoring: indicator definitions (availability, latency, throughput,
//! error rate), objective targets, error budget, burn rate alerting, rolling window
//! computation, compliance reporting, and multi-window alerting.

use chrono::{DateTime, Utc};
use serde_json::{json, Value};

// ── Types ──

/// Kind of service level indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliKind {
    Availability,
    Latency,
    Throughput,
    ErrorRate,
}

impl SliKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SliKind::Availability => "availability",
            SliKind::Latency => "latency",
            SliKind::Throughput => "throughput",
            SliKind::ErrorRate => "error_rate",
        }
    }
}

/// A single data point for an SLI measurement.
#[derive(Debug, Clone)]
pub struct SliDataPoint {
    pub timestamp: DateTime<Utc>,
    pub good: u64,
    pub total: u64,
    /// Optional latency value in milliseconds (for latency SLIs).
    pub latency_ms: Option<f64>,
}

impl SliDataPoint {
    pub fn new(good: u64, total: u64) -> Self {
        Self {
            timestamp: Utc::now(),
            good,
            total,
            latency_ms: None,
        }
    }

    pub fn with_latency(mut self, ms: f64) -> Self {
        self.latency_ms = Some(ms);
        self
    }

    pub fn ratio(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.good as f64 / self.total as f64
    }
}

/// Service Level Indicator definition.
#[derive(Debug, Clone)]
pub struct Sli {
    pub name: String,
    pub kind: SliKind,
    pub description: String,
    pub data_points: Vec<SliDataPoint>,
}

impl Sli {
    pub fn new(name: &str, kind: SliKind, description: &str) -> Self {
        Self {
            name: name.to_string(),
            kind,
            description: description.to_string(),
            data_points: Vec::new(),
        }
    }

    pub fn record(&mut self, dp: SliDataPoint) {
        self.data_points.push(dp);
    }

    /// Compute the SLI value across all data points.
    pub fn value(&self) -> f64 {
        let total_good: u64 = self.data_points.iter().map(|d| d.good).sum();
        let total_all: u64 = self.data_points.iter().map(|d| d.total).sum();
        if total_all == 0 {
            return 1.0;
        }
        total_good as f64 / total_all as f64
    }

    /// Compute the SLI value for a rolling window (last N data points).
    pub fn rolling_value(&self, window_size: usize) -> f64 {
        let start = if self.data_points.len() > window_size {
            self.data_points.len() - window_size
        } else {
            0
        };
        let slice = &self.data_points[start..];
        let total_good: u64 = slice.iter().map(|d| d.good).sum();
        let total_all: u64 = slice.iter().map(|d| d.total).sum();
        if total_all == 0 {
            return 1.0;
        }
        total_good as f64 / total_all as f64
    }

    /// Compute SLI for data points within a time range.
    pub fn value_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> f64 {
        let filtered: Vec<&SliDataPoint> = self
            .data_points
            .iter()
            .filter(|d| d.timestamp >= from && d.timestamp <= to)
            .collect();
        let total_good: u64 = filtered.iter().map(|d| d.good).sum();
        let total_all: u64 = filtered.iter().map(|d| d.total).sum();
        if total_all == 0 {
            return 1.0;
        }
        total_good as f64 / total_all as f64
    }
}

/// Service Level Objective: a target for an SLI.
#[derive(Debug, Clone)]
pub struct Slo {
    pub name: String,
    pub sli_name: String,
    /// Target ratio (e.g. 0.999 for 99.9%).
    pub target: f64,
    /// Rolling window size in data points.
    pub window_size: usize,
    pub description: String,
}

impl Slo {
    pub fn new(name: &str, sli_name: &str, target: f64, window_size: usize) -> Self {
        Self {
            name: name.to_string(),
            sli_name: sli_name.to_string(),
            target,
            window_size,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Check compliance: is the SLI meeting the target?
    pub fn is_met(&self, sli: &Sli) -> bool {
        sli.rolling_value(self.window_size) >= self.target
    }

    /// Error budget: fraction of allowed bad events remaining.
    /// Returns a value between 0.0 (exhausted) and 1.0 (fully remaining).
    pub fn error_budget_remaining(&self, sli: &Sli) -> f64 {
        let allowed_error = 1.0 - self.target;
        if allowed_error <= 0.0 {
            return 0.0;
        }
        let actual_error = 1.0 - sli.rolling_value(self.window_size);
        let consumed = actual_error / allowed_error;
        (1.0 - consumed).clamp(0.0, 1.0)
    }

    /// Burn rate: how fast the error budget is being consumed.
    /// A burn rate of 1.0 means budget is consumed exactly at the limit.
    /// > 1.0 means faster than sustainable.
    pub fn burn_rate(&self, sli: &Sli) -> f64 {
        let allowed_error = 1.0 - self.target;
        if allowed_error <= 0.0 {
            return f64::INFINITY;
        }
        let actual_error = 1.0 - sli.rolling_value(self.window_size);
        actual_error / allowed_error
    }
}

/// Multi-window alert: fires when burn rate exceeds threshold in both
/// a long window and a short window.
#[derive(Debug, Clone)]
pub struct MultiWindowAlert {
    pub name: String,
    pub slo_name: String,
    pub long_window: usize,
    pub short_window: usize,
    pub burn_rate_threshold: f64,
}

impl MultiWindowAlert {
    pub fn new(
        name: &str,
        slo_name: &str,
        long_window: usize,
        short_window: usize,
        threshold: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            slo_name: slo_name.to_string(),
            long_window,
            short_window,
            burn_rate_threshold: threshold,
        }
    }

    /// Check if alert should fire given an SLI and its SLO target.
    pub fn should_fire(&self, sli: &Sli, target: f64) -> bool {
        let allowed_error = 1.0 - target;
        if allowed_error <= 0.0 {
            return true;
        }
        let long_error = 1.0 - sli.rolling_value(self.long_window);
        let short_error = 1.0 - sli.rolling_value(self.short_window);
        let long_burn = long_error / allowed_error;
        let short_burn = short_error / allowed_error;
        long_burn > self.burn_rate_threshold && short_burn > self.burn_rate_threshold
    }
}

/// Compliance report for an SLO.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    pub slo_name: String,
    pub target: f64,
    pub actual: f64,
    pub is_met: bool,
    pub error_budget_remaining: f64,
    pub burn_rate: f64,
    pub data_points_count: usize,
}

impl ComplianceReport {
    pub fn generate(slo: &Slo, sli: &Sli) -> Self {
        Self {
            slo_name: slo.name.clone(),
            target: slo.target,
            actual: sli.rolling_value(slo.window_size),
            is_met: slo.is_met(sli),
            error_budget_remaining: slo.error_budget_remaining(sli),
            burn_rate: slo.burn_rate(sli),
            data_points_count: sli.data_points.len(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "slo_name": self.slo_name,
            "target": self.target,
            "actual": self.actual,
            "is_met": self.is_met,
            "error_budget_remaining_pct": (self.error_budget_remaining * 100.0),
            "burn_rate": self.burn_rate,
            "data_points": self.data_points_count,
        })
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sli(good_counts: &[u64], total: u64) -> Sli {
        let mut sli = Sli::new("availability", SliKind::Availability, "API availability");
        for &g in good_counts {
            sli.record(SliDataPoint::new(g, total));
        }
        sli
    }

    #[test]
    fn test_sli_data_point_ratio() {
        let dp = SliDataPoint::new(99, 100);
        assert!((dp.ratio() - 0.99).abs() < 1e-10);
        let empty = SliDataPoint::new(0, 0);
        assert!((empty.ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sli_value() {
        let sli = make_sli(&[95, 98, 100], 100);
        let val = sli.value();
        // (95+98+100) / (100+100+100) = 293/300
        assert!((val - 293.0 / 300.0).abs() < 1e-10);
    }

    #[test]
    fn test_sli_rolling_value() {
        let sli = make_sli(&[50, 90, 100], 100);
        // Last 2 data points: (90+100)/(100+100)
        let val = sli.rolling_value(2);
        assert!((val - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_slo_is_met() {
        let sli = make_sli(&[999, 1000, 998], 1000);
        let slo = Slo::new("api-avail", "availability", 0.999, 3);
        // (999+1000+998)/3000 = 2997/3000 = 0.999
        assert!(slo.is_met(&sli));
    }

    #[test]
    fn test_slo_not_met() {
        let sli = make_sli(&[990, 995, 993], 1000);
        let slo = Slo::new("api-avail", "availability", 0.999, 3);
        assert!(!slo.is_met(&sli));
    }

    #[test]
    fn test_error_budget_remaining() {
        // Target 99.9% => 0.1% error budget. If actual is 99.95%, consumed 50%.
        let sli = make_sli(&[9995], 10000);
        let slo = Slo::new("avail", "availability", 0.999, 1);
        let remaining = slo.error_budget_remaining(&sli);
        assert!((remaining - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_error_budget_exhausted() {
        let sli = make_sli(&[990], 1000);
        let slo = Slo::new("avail", "availability", 0.999, 1);
        let remaining = slo.error_budget_remaining(&sli);
        assert_eq!(remaining, 0.0); // Clamped to 0.
    }

    #[test]
    fn test_burn_rate() {
        // Target 99.9% => 0.1% error budget. Actual error = 0.2% => burn rate 2.0
        let sli = make_sli(&[998], 1000);
        let slo = Slo::new("avail", "availability", 0.999, 1);
        let br = slo.burn_rate(&sli);
        assert!((br - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_window_alert_fires() {
        // Both windows have 2% error rate, target 99.9% => burn rate 20x
        let sli = make_sli(&[980, 980, 980, 980], 1000);
        let alert = MultiWindowAlert::new("fast-burn", "avail", 4, 2, 14.0);
        assert!(alert.should_fire(&sli, 0.999));
    }

    #[test]
    fn test_multi_window_alert_no_fire() {
        // Both windows have 0.05% error rate, target 99.9% => burn rate 0.5x
        let sli = make_sli(&[9995, 9995, 9995, 9995], 10000);
        let alert = MultiWindowAlert::new("fast-burn", "avail", 4, 2, 14.0);
        assert!(!alert.should_fire(&sli, 0.999));
    }

    #[test]
    fn test_compliance_report() {
        let sli = make_sli(&[999, 1000, 998], 1000);
        let slo = Slo::new("api-avail", "availability", 0.999, 3);
        let report = ComplianceReport::generate(&slo, &sli);
        assert_eq!(report.slo_name, "api-avail");
        assert_eq!(report.data_points_count, 3);
        assert!(report.is_met);
    }

    #[test]
    fn test_compliance_report_json() {
        let sli = make_sli(&[999], 1000);
        let slo = Slo::new("avail", "availability", 0.999, 1);
        let report = ComplianceReport::generate(&slo, &sli);
        let j = report.to_json();
        assert_eq!(j["slo_name"], "avail");
        assert!(j["error_budget_remaining_pct"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn test_sli_kind_strings() {
        assert_eq!(SliKind::Availability.as_str(), "availability");
        assert_eq!(SliKind::Latency.as_str(), "latency");
        assert_eq!(SliKind::Throughput.as_str(), "throughput");
        assert_eq!(SliKind::ErrorRate.as_str(), "error_rate");
    }

    #[test]
    fn test_sli_with_latency() {
        let dp = SliDataPoint::new(100, 100).with_latency(45.2);
        assert_eq!(dp.latency_ms, Some(45.2));
    }
}
