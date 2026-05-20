//! SLA/SLO tracking — error budget calculation, burn rate alerts, rolling window
//! compliance, SLI definitions, multi-objective tracking, budget exhaustion prediction.
//!
//! Pure-Rust replacement for Sloth, OpenSLO, and similar SLO frameworks.

use std::collections::HashMap;
use std::fmt;

// ── SLI definitions ──────────────────────────────────────────────

/// A Service Level Indicator — the metric being measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Sli {
    pub name: String,
    pub description: String,
    pub unit: SliUnit,
}

/// Unit of measurement for an SLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliUnit {
    /// Ratio of good events to total events (0.0 to 1.0).
    Ratio,
    /// Latency in milliseconds.
    LatencyMs,
    /// Throughput in requests per second.
    Throughput,
}

impl fmt::Display for SliUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SliUnit::Ratio => write!(f, "ratio"),
            SliUnit::LatencyMs => write!(f, "ms"),
            SliUnit::Throughput => write!(f, "rps"),
        }
    }
}

impl Sli {
    pub fn new(name: impl Into<String>, description: impl Into<String>, unit: SliUnit) -> Self {
        Self { name: name.into(), description: description.into(), unit }
    }
}

// ── SLO definitions ──────────────────────────────────────────────

/// A Service Level Objective — the target for an SLI over a window.
#[derive(Debug, Clone, PartialEq)]
pub struct Slo {
    pub name: String,
    pub sli: Sli,
    pub target: f64,
    pub window: Window,
}

/// Measurement window for an SLO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// Rolling window of N seconds.
    Rolling { seconds: u64 },
    /// Calendar-aligned window.
    Calendar(CalendarPeriod),
}

/// Calendar period alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalendarPeriod {
    Day,
    Week,
    Month,
    Quarter,
}

impl Slo {
    pub fn new(name: impl Into<String>, sli: Sli, target: f64, window: Window) -> Self {
        Self { name: name.into(), sli, target, window }
    }

    /// The error budget as a fraction: 1.0 - target.
    pub fn error_budget_fraction(&self) -> f64 {
        1.0 - self.target
    }
}

// ── Measurement events ───────────────────────────────────────────

/// A single measurement event for an SLI.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Timestamp in seconds since epoch.
    pub timestamp_s: u64,
    /// Total events in this measurement period.
    pub total: u64,
    /// Good events (meeting the SLI threshold).
    pub good: u64,
}

impl Measurement {
    pub fn new(timestamp_s: u64, total: u64, good: u64) -> Self {
        assert!(good <= total, "good events cannot exceed total");
        Self { timestamp_s, total, good }
    }

    pub fn bad(&self) -> u64 {
        self.total - self.good
    }

    pub fn ratio(&self) -> f64 {
        if self.total == 0 { 1.0 } else { self.good as f64 / self.total as f64 }
    }
}

// ── Error budget state ───────────────────────────────────────────

/// Current error budget status.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorBudget {
    /// Total allowed bad events in the window.
    pub total_budget: f64,
    /// Bad events consumed so far.
    pub consumed: f64,
    /// Remaining budget (can be negative if overspent).
    pub remaining: f64,
    /// Fraction of budget consumed (0.0 = untouched, 1.0 = exhausted).
    pub consumption_ratio: f64,
}

impl ErrorBudget {
    pub fn is_exhausted(&self) -> bool {
        self.remaining <= 0.0
    }

    pub fn is_healthy(&self) -> bool {
        self.consumption_ratio < 0.8
    }
}

// ── Burn rate ────────────────────────────────────────────────────

/// Burn rate: how fast the error budget is being consumed.
#[derive(Debug, Clone, PartialEq)]
pub struct BurnRate {
    /// Current burn rate multiplier (1.0 = nominal).
    pub rate: f64,
    /// Window duration in seconds over which burn rate was calculated.
    pub window_seconds: u64,
    /// Alert severity based on burn rate.
    pub severity: BurnRateSeverity,
}

/// Severity levels based on burn rate thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BurnRateSeverity {
    Normal,
    Warning,
    Critical,
    Emergency,
}

impl fmt::Display for BurnRateSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BurnRateSeverity::Normal => write!(f, "normal"),
            BurnRateSeverity::Warning => write!(f, "warning"),
            BurnRateSeverity::Critical => write!(f, "critical"),
            BurnRateSeverity::Emergency => write!(f, "emergency"),
        }
    }
}

// ── Budget exhaustion prediction ─────────────────────────────────

/// Prediction for when the error budget will be exhausted.
#[derive(Debug, Clone, PartialEq)]
pub struct ExhaustionPrediction {
    /// Estimated seconds until budget exhaustion, or None if budget is not being consumed.
    pub seconds_until_exhaustion: Option<u64>,
    /// Current burn rate used for prediction.
    pub burn_rate: f64,
    /// Whether the budget is already exhausted.
    pub already_exhausted: bool,
}

// ── SLO Tracker ──────────────────────────────────────────────────

/// Tracks measurements against an SLO and calculates compliance.
#[derive(Debug, Clone)]
pub struct SloTracker {
    pub slo: Slo,
    measurements: Vec<Measurement>,
    /// Burn rate thresholds: (multiplier, severity).
    burn_rate_thresholds: Vec<(f64, BurnRateSeverity)>,
}

impl SloTracker {
    pub fn new(slo: Slo) -> Self {
        Self {
            slo,
            measurements: Vec::new(),
            burn_rate_thresholds: vec![
                (1.0, BurnRateSeverity::Normal),
                (2.0, BurnRateSeverity::Warning),
                (6.0, BurnRateSeverity::Critical),
                (14.0, BurnRateSeverity::Emergency),
            ],
        }
    }

    /// Set custom burn rate thresholds. Must be sorted ascending by multiplier.
    pub fn set_burn_rate_thresholds(&mut self, thresholds: Vec<(f64, BurnRateSeverity)>) {
        self.burn_rate_thresholds = thresholds;
    }

    /// Record a measurement.
    pub fn record(&mut self, measurement: Measurement) {
        self.measurements.push(measurement);
    }

    /// Get all measurements within the SLO window ending at `now_s`.
    pub fn window_measurements(&self, now_s: u64) -> Vec<&Measurement> {
        let window_start = match self.slo.window {
            Window::Rolling { seconds } => now_s.saturating_sub(seconds),
            Window::Calendar(period) => {
                let period_secs = calendar_period_seconds(period);
                now_s.saturating_sub(period_secs)
            }
        };
        self.measurements.iter()
            .filter(|m| m.timestamp_s >= window_start && m.timestamp_s <= now_s)
            .collect()
    }

    /// Calculate current compliance ratio within the window.
    pub fn compliance(&self, now_s: u64) -> f64 {
        let ms = self.window_measurements(now_s);
        let total: u64 = ms.iter().map(|m| m.total).sum();
        let good: u64 = ms.iter().map(|m| m.good).sum();
        if total == 0 { return 1.0; }
        good as f64 / total as f64
    }

    /// Whether the SLO is currently being met.
    pub fn is_meeting_slo(&self, now_s: u64) -> bool {
        self.compliance(now_s) >= self.slo.target
    }

    /// Calculate the error budget status within the window.
    pub fn error_budget(&self, now_s: u64) -> ErrorBudget {
        let ms = self.window_measurements(now_s);
        let total: u64 = ms.iter().map(|m| m.total).sum();
        let bad: u64 = ms.iter().map(|m| m.bad()).sum();

        let budget_fraction = self.slo.error_budget_fraction();
        let total_budget = total as f64 * budget_fraction;
        let consumed = bad as f64;
        let remaining = total_budget - consumed;
        let consumption_ratio = if total_budget > 0.0 {
            consumed / total_budget
        } else {
            0.0
        };

        ErrorBudget { total_budget, consumed, remaining, consumption_ratio }
    }

    /// Calculate burn rate over a specific lookback period.
    pub fn burn_rate(&self, now_s: u64, lookback_seconds: u64) -> BurnRate {
        let start = now_s.saturating_sub(lookback_seconds);
        let ms: Vec<&Measurement> = self.measurements.iter()
            .filter(|m| m.timestamp_s >= start && m.timestamp_s <= now_s)
            .collect();

        let total: u64 = ms.iter().map(|m| m.total).sum();
        let bad: u64 = ms.iter().map(|m| m.bad()).sum();

        let window_seconds = match self.slo.window {
            Window::Rolling { seconds } => seconds,
            Window::Calendar(period) => calendar_period_seconds(period),
        };

        let error_budget_fraction = self.slo.error_budget_fraction();
        // Burn rate = (observed error rate) / (allowed error rate)
        // normalized to the ratio of lookback to full window.
        let rate = if total == 0 || error_budget_fraction < 1e-15 {
            0.0
        } else {
            let observed_error_rate = bad as f64 / total as f64;
            observed_error_rate / error_budget_fraction
        };

        // Determine severity.
        let mut severity = BurnRateSeverity::Normal;
        for (threshold, sev) in &self.burn_rate_thresholds {
            if rate >= *threshold {
                severity = *sev;
            }
        }

        BurnRate { rate, window_seconds: lookback_seconds, severity }
    }

    /// Predict when the error budget will be exhausted.
    pub fn predict_exhaustion(&self, now_s: u64, lookback_seconds: u64) -> ExhaustionPrediction {
        let budget = self.error_budget(now_s);
        if budget.is_exhausted() {
            return ExhaustionPrediction {
                seconds_until_exhaustion: Some(0),
                burn_rate: self.burn_rate(now_s, lookback_seconds).rate,
                already_exhausted: true,
            };
        }

        let br = self.burn_rate(now_s, lookback_seconds);
        if br.rate < 1e-15 {
            return ExhaustionPrediction {
                seconds_until_exhaustion: None,
                burn_rate: 0.0,
                already_exhausted: false,
            };
        }

        let window_seconds = match self.slo.window {
            Window::Rolling { seconds } => seconds,
            Window::Calendar(period) => calendar_period_seconds(period),
        } as f64;

        // Time to exhaust = (remaining fraction of budget) * window / burn_rate
        let remaining_fraction = 1.0 - budget.consumption_ratio;
        let seconds = (remaining_fraction * window_seconds / br.rate).max(0.0);

        ExhaustionPrediction {
            seconds_until_exhaustion: Some(seconds as u64),
            burn_rate: br.rate,
            already_exhausted: false,
        }
    }

    /// Get total measurement count.
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Prune measurements older than the given timestamp.
    pub fn prune_before(&mut self, timestamp_s: u64) {
        self.measurements.retain(|m| m.timestamp_s >= timestamp_s);
    }
}

// ── Multi-objective tracker ──────────────────────────────────────

/// Tracks multiple SLOs simultaneously and reports aggregate status.
#[derive(Debug, Clone)]
pub struct MultiObjectiveTracker {
    trackers: HashMap<String, SloTracker>,
}

impl MultiObjectiveTracker {
    pub fn new() -> Self {
        Self { trackers: HashMap::new() }
    }

    pub fn add_slo(&mut self, slo: Slo) {
        let name = slo.name.clone();
        self.trackers.insert(name, SloTracker::new(slo));
    }

    pub fn record(&mut self, slo_name: &str, measurement: Measurement) -> bool {
        if let Some(tracker) = self.trackers.get_mut(slo_name) {
            tracker.record(measurement);
            true
        } else {
            false
        }
    }

    pub fn tracker(&self, slo_name: &str) -> Option<&SloTracker> {
        self.trackers.get(slo_name)
    }

    pub fn tracker_mut(&mut self, slo_name: &str) -> Option<&mut SloTracker> {
        self.trackers.get_mut(slo_name)
    }

    /// Returns all SLO names that are currently not meeting their target.
    pub fn violations(&self, now_s: u64) -> Vec<String> {
        let mut result: Vec<String> = self.trackers.iter()
            .filter(|(_, t)| !t.is_meeting_slo(now_s))
            .map(|(name, _)| name.clone())
            .collect();
        result.sort();
        result
    }

    /// Returns the overall health: all SLOs meeting targets.
    pub fn all_healthy(&self, now_s: u64) -> bool {
        self.trackers.values().all(|t| t.is_meeting_slo(now_s))
    }

    /// Summary of all SLOs at a point in time.
    pub fn summary(&self, now_s: u64) -> Vec<SloSummary> {
        let mut results: Vec<SloSummary> = self.trackers.iter()
            .map(|(name, tracker)| {
                SloSummary {
                    name: name.clone(),
                    target: tracker.slo.target,
                    compliance: tracker.compliance(now_s),
                    budget: tracker.error_budget(now_s),
                    meeting_slo: tracker.is_meeting_slo(now_s),
                }
            })
            .collect();
        results.sort_by(|a, b| a.name.cmp(&b.name));
        results
    }

    pub fn slo_count(&self) -> usize {
        self.trackers.len()
    }
}

impl Default for MultiObjectiveTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary entry for a single SLO.
#[derive(Debug, Clone, PartialEq)]
pub struct SloSummary {
    pub name: String,
    pub target: f64,
    pub compliance: f64,
    pub budget: ErrorBudget,
    pub meeting_slo: bool,
}

// ── Helpers ──────────────────────────────────────────────────────

fn calendar_period_seconds(period: CalendarPeriod) -> u64 {
    match period {
        CalendarPeriod::Day => 86_400,
        CalendarPeriod::Week => 604_800,
        CalendarPeriod::Month => 2_592_000, // 30 days
        CalendarPeriod::Quarter => 7_776_000, // 90 days
    }
}

// ── Rolling window compliance helper ─────────────────────────────

/// Calculate compliance over a sliding window, returning compliance at each step.
pub fn rolling_compliance(
    measurements: &[Measurement],
    window_seconds: u64,
) -> Vec<(u64, f64)> {
    let mut result = Vec::new();
    for m in measurements {
        let window_start = m.timestamp_s.saturating_sub(window_seconds);
        let window_ms: Vec<&Measurement> = measurements.iter()
            .filter(|om| om.timestamp_s >= window_start && om.timestamp_s <= m.timestamp_s)
            .collect();
        let total: u64 = window_ms.iter().map(|om| om.total).sum();
        let good: u64 = window_ms.iter().map(|om| om.good).sum();
        let compliance = if total == 0 { 1.0 } else { good as f64 / total as f64 };
        result.push((m.timestamp_s, compliance));
    }
    result
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_availability_sli() -> Sli {
        Sli::new("availability", "Request success rate", SliUnit::Ratio)
    }

    fn make_slo_99() -> Slo {
        Slo::new(
            "api-availability",
            make_availability_sli(),
            0.99,
            Window::Rolling { seconds: 3600 },
        )
    }

    #[test]
    fn test_sli_creation() {
        let sli = make_availability_sli();
        assert_eq!(sli.name, "availability");
        assert_eq!(sli.unit, SliUnit::Ratio);
    }

    #[test]
    fn test_sli_unit_display() {
        assert_eq!(format!("{}", SliUnit::Ratio), "ratio");
        assert_eq!(format!("{}", SliUnit::LatencyMs), "ms");
        assert_eq!(format!("{}", SliUnit::Throughput), "rps");
    }

    #[test]
    fn test_slo_error_budget() {
        let slo = make_slo_99();
        let budget = slo.error_budget_fraction();
        assert!((budget - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_basic() {
        let m = Measurement::new(1000, 100, 98);
        assert_eq!(m.bad(), 2);
        assert!((m.ratio() - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_zero_total() {
        let m = Measurement::new(1000, 0, 0);
        assert!((m.ratio() - 1.0).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "good events cannot exceed total")]
    fn test_measurement_bad_input() {
        Measurement::new(1000, 10, 20);
    }

    #[test]
    fn test_slo_tracker_compliance_perfect() {
        let mut tracker = SloTracker::new(make_slo_99());
        tracker.record(Measurement::new(100, 1000, 1000));
        tracker.record(Measurement::new(200, 1000, 1000));
        assert!((tracker.compliance(300) - 1.0).abs() < 1e-10);
        assert!(tracker.is_meeting_slo(300));
    }

    #[test]
    fn test_slo_tracker_compliance_below_target() {
        let mut tracker = SloTracker::new(make_slo_99());
        tracker.record(Measurement::new(100, 1000, 950));
        let c = tracker.compliance(200);
        assert!((c - 0.95).abs() < 1e-10);
        assert!(!tracker.is_meeting_slo(200));
    }

    #[test]
    fn test_slo_tracker_window_filtering() {
        let slo = Slo::new(
            "test",
            make_availability_sli(),
            0.99,
            Window::Rolling { seconds: 100 },
        );
        let mut tracker = SloTracker::new(slo);
        // This one is outside the window at now=300.
        tracker.record(Measurement::new(100, 1000, 500));
        // This one is inside.
        tracker.record(Measurement::new(250, 1000, 999));
        let c = tracker.compliance(300);
        // Only the second measurement is in window.
        assert!((c - 0.999).abs() < 1e-10);
    }

    #[test]
    fn test_error_budget_calculation() {
        let mut tracker = SloTracker::new(make_slo_99());
        // 10000 requests, 50 bad => 0.5% error rate, budget is 1% = 100 allowed.
        tracker.record(Measurement::new(100, 10000, 9950));
        let budget = tracker.error_budget(200);
        assert!((budget.total_budget - 100.0).abs() < 1e-10);
        assert!((budget.consumed - 50.0).abs() < 1e-10);
        assert!((budget.remaining - 50.0).abs() < 1e-10);
        assert!((budget.consumption_ratio - 0.5).abs() < 1e-10);
        assert!(!budget.is_exhausted());
        assert!(budget.is_healthy());
    }

    #[test]
    fn test_error_budget_exhausted() {
        let mut tracker = SloTracker::new(make_slo_99());
        // 1000 requests, 20 bad => 2% error rate, budget is 1% = 10 allowed.
        tracker.record(Measurement::new(100, 1000, 980));
        let budget = tracker.error_budget(200);
        assert!(budget.is_exhausted());
        assert!(!budget.is_healthy());
    }

    #[test]
    fn test_burn_rate_normal() {
        let mut tracker = SloTracker::new(make_slo_99());
        // Error rate 0.5% with 1% budget => burn rate ~0.5
        tracker.record(Measurement::new(100, 10000, 9950));
        let br = tracker.burn_rate(200, 3600);
        assert!((br.rate - 0.5).abs() < 0.1);
        assert_eq!(br.severity, BurnRateSeverity::Normal);
    }

    #[test]
    fn test_burn_rate_critical() {
        let mut tracker = SloTracker::new(make_slo_99());
        // Error rate 10% with 1% budget => burn rate = 10
        tracker.record(Measurement::new(100, 1000, 900));
        let br = tracker.burn_rate(200, 3600);
        assert!(br.rate > 6.0);
        assert_eq!(br.severity, BurnRateSeverity::Critical);
    }

    #[test]
    fn test_burn_rate_emergency() {
        let mut tracker = SloTracker::new(make_slo_99());
        // Error rate 20% with 1% budget => burn rate = 20
        tracker.record(Measurement::new(100, 1000, 800));
        let br = tracker.burn_rate(200, 3600);
        assert!(br.rate >= 14.0);
        assert_eq!(br.severity, BurnRateSeverity::Emergency);
    }

    #[test]
    fn test_burn_rate_severity_display() {
        assert_eq!(format!("{}", BurnRateSeverity::Normal), "normal");
        assert_eq!(format!("{}", BurnRateSeverity::Emergency), "emergency");
    }

    #[test]
    fn test_predict_exhaustion_no_errors() {
        let mut tracker = SloTracker::new(make_slo_99());
        tracker.record(Measurement::new(100, 1000, 1000));
        let pred = tracker.predict_exhaustion(200, 3600);
        assert!(!pred.already_exhausted);
        assert!(pred.seconds_until_exhaustion.is_none());
    }

    #[test]
    fn test_predict_exhaustion_already_exhausted() {
        let mut tracker = SloTracker::new(make_slo_99());
        tracker.record(Measurement::new(100, 1000, 980));
        let pred = tracker.predict_exhaustion(200, 3600);
        assert!(pred.already_exhausted);
        assert_eq!(pred.seconds_until_exhaustion, Some(0));
    }

    #[test]
    fn test_predict_exhaustion_partial() {
        let mut tracker = SloTracker::new(make_slo_99());
        // 50% of budget consumed, burn rate around 0.5
        tracker.record(Measurement::new(100, 10000, 9950));
        let pred = tracker.predict_exhaustion(200, 3600);
        assert!(!pred.already_exhausted);
        assert!(pred.seconds_until_exhaustion.is_some());
        // Should predict some time remaining
        assert!(pred.seconds_until_exhaustion.unwrap() > 0);
    }

    #[test]
    fn test_prune_before() {
        let mut tracker = SloTracker::new(make_slo_99());
        tracker.record(Measurement::new(100, 100, 99));
        tracker.record(Measurement::new(200, 100, 99));
        tracker.record(Measurement::new(300, 100, 99));
        assert_eq!(tracker.measurement_count(), 3);
        tracker.prune_before(200);
        assert_eq!(tracker.measurement_count(), 2);
    }

    #[test]
    fn test_calendar_period_seconds() {
        assert_eq!(calendar_period_seconds(CalendarPeriod::Day), 86_400);
        assert_eq!(calendar_period_seconds(CalendarPeriod::Week), 604_800);
        assert_eq!(calendar_period_seconds(CalendarPeriod::Month), 2_592_000);
        assert_eq!(calendar_period_seconds(CalendarPeriod::Quarter), 7_776_000);
    }

    #[test]
    fn test_multi_objective_basic() {
        let mut multi = MultiObjectiveTracker::new();
        multi.add_slo(make_slo_99());
        multi.add_slo(Slo::new(
            "latency",
            Sli::new("p99-latency", "99th percentile latency", SliUnit::LatencyMs),
            0.995,
            Window::Rolling { seconds: 3600 },
        ));
        assert_eq!(multi.slo_count(), 2);
    }

    #[test]
    fn test_multi_objective_all_healthy() {
        let mut multi = MultiObjectiveTracker::new();
        multi.add_slo(make_slo_99());
        multi.record("api-availability", Measurement::new(100, 1000, 999));
        assert!(multi.all_healthy(200));
        assert!(multi.violations(200).is_empty());
    }

    #[test]
    fn test_multi_objective_violations() {
        let mut multi = MultiObjectiveTracker::new();
        multi.add_slo(make_slo_99());
        // Bad enough to violate 99% SLO
        multi.record("api-availability", Measurement::new(100, 1000, 950));
        let v = multi.violations(200);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], "api-availability");
        assert!(!multi.all_healthy(200));
    }

    #[test]
    fn test_multi_objective_record_unknown() {
        let mut multi = MultiObjectiveTracker::new();
        assert!(!multi.record("nonexistent", Measurement::new(100, 10, 10)));
    }

    #[test]
    fn test_multi_objective_summary() {
        let mut multi = MultiObjectiveTracker::new();
        multi.add_slo(make_slo_99());
        multi.record("api-availability", Measurement::new(100, 1000, 995));
        let summary = multi.summary(200);
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].name, "api-availability");
        assert!(summary[0].meeting_slo);
        assert!((summary[0].compliance - 0.995).abs() < 1e-10);
    }

    #[test]
    fn test_multi_objective_default() {
        let multi = MultiObjectiveTracker::default();
        assert_eq!(multi.slo_count(), 0);
    }

    #[test]
    fn test_rolling_compliance() {
        let measurements = vec![
            Measurement::new(100, 100, 100),
            Measurement::new(200, 100, 90),
            Measurement::new(300, 100, 100),
        ];
        let rc = rolling_compliance(&measurements, 150);
        assert_eq!(rc.len(), 3);
        // First: only itself, 100/100 = 1.0
        assert!((rc[0].1 - 1.0).abs() < 1e-10);
        // Second: 100+200 in window, (100+90)/(100+100) = 0.95
        assert!((rc[1].1 - 0.95).abs() < 1e-10);
        // Third: 200+300 in window, (90+100)/(100+100) = 0.95
        assert!((rc[2].1 - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_compliance_empty() {
        let rc = rolling_compliance(&[], 100);
        assert!(rc.is_empty());
    }

    #[test]
    fn test_custom_burn_rate_thresholds() {
        let mut tracker = SloTracker::new(make_slo_99());
        tracker.set_burn_rate_thresholds(vec![
            (1.0, BurnRateSeverity::Warning),
            (3.0, BurnRateSeverity::Emergency),
        ]);
        // Error rate 5% with 1% budget => burn rate = 5
        tracker.record(Measurement::new(100, 1000, 950));
        let br = tracker.burn_rate(200, 3600);
        assert_eq!(br.severity, BurnRateSeverity::Emergency);
    }

    #[test]
    fn test_slo_tracker_no_measurements() {
        let tracker = SloTracker::new(make_slo_99());
        assert!((tracker.compliance(100) - 1.0).abs() < 1e-10);
        assert!(tracker.is_meeting_slo(100));
        assert_eq!(tracker.measurement_count(), 0);
    }

    #[test]
    fn test_error_budget_zero_total() {
        let tracker = SloTracker::new(make_slo_99());
        let budget = tracker.error_budget(100);
        assert!((budget.total_budget - 0.0).abs() < 1e-10);
        assert!((budget.consumption_ratio - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_burn_rate_zero_measurements() {
        let tracker = SloTracker::new(make_slo_99());
        let br = tracker.burn_rate(100, 3600);
        assert!((br.rate - 0.0).abs() < 1e-10);
        assert_eq!(br.severity, BurnRateSeverity::Normal);
    }

    #[test]
    fn test_burn_rate_severity_ordering() {
        assert!(BurnRateSeverity::Normal < BurnRateSeverity::Warning);
        assert!(BurnRateSeverity::Warning < BurnRateSeverity::Critical);
        assert!(BurnRateSeverity::Critical < BurnRateSeverity::Emergency);
    }

    #[test]
    fn test_calendar_window_slo() {
        let slo = Slo::new(
            "monthly-avail",
            make_availability_sli(),
            0.999,
            Window::Calendar(CalendarPeriod::Month),
        );
        let mut tracker = SloTracker::new(slo);
        tracker.record(Measurement::new(1_000_000, 100_000, 99_950));
        assert!(tracker.is_meeting_slo(1_100_000));
    }

    #[test]
    fn test_tracker_accessor() {
        let mut multi = MultiObjectiveTracker::new();
        multi.add_slo(make_slo_99());
        assert!(multi.tracker("api-availability").is_some());
        assert!(multi.tracker("nonexistent").is_none());
        assert!(multi.tracker_mut("api-availability").is_some());
    }
}
