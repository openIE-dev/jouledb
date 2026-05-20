//! Capacity planning: resource utilization tracking, trend extrapolation via
//! linear regression, threshold forecasting (when will X% be reached), what-if
//! modeling, scaling recommendations, and headroom calculation.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

// ── Types ──

/// Kind of resource being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    Cpu,
    Memory,
    Disk,
    Network,
    Connections,
    Custom,
}

impl ResourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceKind::Cpu => "cpu",
            ResourceKind::Memory => "memory",
            ResourceKind::Disk => "disk",
            ResourceKind::Network => "network",
            ResourceKind::Connections => "connections",
            ResourceKind::Custom => "custom",
        }
    }
}

/// A single utilization sample.
#[derive(Debug, Clone)]
pub struct UtilizationSample {
    pub timestamp: DateTime<Utc>,
    pub used: f64,
    pub total: f64,
}

impl UtilizationSample {
    pub fn new(timestamp: DateTime<Utc>, used: f64, total: f64) -> Self {
        Self {
            timestamp,
            used,
            total: if total > 0.0 { total } else { 1.0 },
        }
    }

    pub fn utilization_percent(&self) -> f64 {
        (self.used / self.total) * 100.0
    }

    pub fn available(&self) -> f64 {
        (self.total - self.used).max(0.0)
    }
}

/// Linear regression result (y = slope * x + intercept).
#[derive(Debug, Clone, Copy)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

impl LinearRegression {
    /// Fit a linear regression to (x, y) data.
    pub fn fit(xs: &[f64], ys: &[f64]) -> Option<Self> {
        let n = xs.len();
        if n < 2 || n != ys.len() {
            return None;
        }
        let n_f = n as f64;
        let sum_x: f64 = xs.iter().sum();
        let sum_y: f64 = ys.iter().sum();
        let sum_xy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = xs.iter().map(|x| x * x).sum();

        let denom = n_f * sum_x2 - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            return None;
        }

        let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n_f;

        // R-squared
        let mean_y = sum_y / n_f;
        let ss_tot: f64 = ys.iter().map(|y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| {
                let predicted = slope * x + intercept;
                (y - predicted).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() < f64::EPSILON {
            1.0
        } else {
            1.0 - ss_res / ss_tot
        };

        Some(Self {
            slope,
            intercept,
            r_squared,
        })
    }

    /// Predict y for a given x.
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// Solve for x when y equals the given target (x = (y - intercept) / slope).
    pub fn solve_for_x(&self, target_y: f64) -> Option<f64> {
        if self.slope.abs() < f64::EPSILON {
            return None;
        }
        Some((target_y - self.intercept) / self.slope)
    }
}

/// Forecast result: when a threshold will be reached.
#[derive(Debug, Clone)]
pub struct ThresholdForecast {
    pub resource: String,
    pub current_utilization_percent: f64,
    pub target_percent: f64,
    pub estimated_time_to_threshold: Option<Duration>,
    pub estimated_date: Option<DateTime<Utc>>,
    pub confidence: f64,
    pub trend_direction: TrendDirection,
}

/// Direction of the resource usage trend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Rising,
    Falling,
    Stable,
}

impl TrendDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            TrendDirection::Rising => "rising",
            TrendDirection::Falling => "falling",
            TrendDirection::Stable => "stable",
        }
    }

    pub fn from_slope(slope: f64) -> Self {
        if slope > 0.001 {
            TrendDirection::Rising
        } else if slope < -0.001 {
            TrendDirection::Falling
        } else {
            TrendDirection::Stable
        }
    }
}

/// Scaling recommendation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalingAction {
    NoAction,
    ScaleUp { additional_units: u64, reason: String },
    ScaleDown { excess_units: u64, reason: String },
    Alert { message: String },
}

/// Headroom analysis result.
#[derive(Debug, Clone)]
pub struct HeadroomAnalysis {
    pub resource: String,
    pub current_used: f64,
    pub total_capacity: f64,
    pub headroom_absolute: f64,
    pub headroom_percent: f64,
    pub sufficient: bool,
    pub safety_margin_percent: f64,
}

/// What-if scenario result.
#[derive(Debug, Clone)]
pub struct WhatIfResult {
    pub scenario: String,
    pub original_utilization_percent: f64,
    pub projected_utilization_percent: f64,
    pub additional_load: f64,
    pub would_exceed_threshold: bool,
    pub threshold_percent: f64,
}

// ── Resource Tracker ──

/// Tracks utilization samples for a single resource.
#[derive(Debug, Clone)]
pub struct ResourceTracker {
    pub name: String,
    pub kind: ResourceKind,
    pub total_capacity: f64,
    samples: Vec<UtilizationSample>,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
}

impl ResourceTracker {
    pub fn new(name: &str, kind: ResourceKind, total_capacity: f64) -> Self {
        Self {
            name: name.to_string(),
            kind,
            total_capacity: if total_capacity > 0.0 {
                total_capacity
            } else {
                100.0
            },
            samples: Vec::new(),
            warning_threshold: 70.0,
            critical_threshold: 90.0,
        }
    }

    pub fn with_thresholds(mut self, warning: f64, critical: f64) -> Self {
        self.warning_threshold = warning;
        self.critical_threshold = critical;
        self
    }

    pub fn record(&mut self, used: f64) {
        self.samples.push(UtilizationSample::new(
            Utc::now(),
            used,
            self.total_capacity,
        ));
    }

    pub fn record_at(&mut self, timestamp: DateTime<Utc>, used: f64) {
        self.samples
            .push(UtilizationSample::new(timestamp, used, self.total_capacity));
    }

    pub fn current_utilization(&self) -> Option<f64> {
        self.samples.last().map(|s| s.utilization_percent())
    }

    pub fn average_utilization(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|s| s.utilization_percent()).sum();
        sum / self.samples.len() as f64
    }

    pub fn peak_utilization(&self) -> f64 {
        self.samples
            .iter()
            .map(|s| s.utilization_percent())
            .fold(0.0f64, f64::max)
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Fit a linear regression on utilization over time.
    pub fn trend(&self) -> Option<LinearRegression> {
        if self.samples.len() < 2 {
            return None;
        }
        let base = self.samples[0].timestamp;
        let xs: Vec<f64> = self
            .samples
            .iter()
            .map(|s| (s.timestamp - base).num_seconds() as f64)
            .collect();
        let ys: Vec<f64> = self
            .samples
            .iter()
            .map(|s| s.utilization_percent())
            .collect();
        LinearRegression::fit(&xs, &ys)
    }

    /// Forecast when a given utilization threshold (%) will be reached.
    pub fn forecast_threshold(&self, target_percent: f64) -> ThresholdForecast {
        let current = self.current_utilization().unwrap_or(0.0);
        let regression = self.trend();

        match regression {
            Some(reg) if reg.slope.abs() > f64::EPSILON => {
                let direction = TrendDirection::from_slope(reg.slope);
                let base = self.samples[0].timestamp;
                let target_x = reg.solve_for_x(target_percent);

                let (time_to, date) = match target_x {
                    Some(secs) if secs > 0.0 => {
                        let last_x = (self.samples.last().unwrap().timestamp - base)
                            .num_seconds() as f64;
                        let remaining = secs - last_x;
                        if remaining > 0.0 {
                            let dur = Duration::seconds(remaining as i64);
                            let estimated_date = Utc::now() + dur;
                            (Some(dur), Some(estimated_date))
                        } else {
                            (Some(Duration::zero()), Some(Utc::now()))
                        }
                    }
                    _ => (None, None),
                };

                ThresholdForecast {
                    resource: self.name.clone(),
                    current_utilization_percent: current,
                    target_percent,
                    estimated_time_to_threshold: time_to,
                    estimated_date: date,
                    confidence: reg.r_squared,
                    trend_direction: direction,
                }
            }
            _ => ThresholdForecast {
                resource: self.name.clone(),
                current_utilization_percent: current,
                target_percent,
                estimated_time_to_threshold: None,
                estimated_date: None,
                confidence: 0.0,
                trend_direction: TrendDirection::Stable,
            },
        }
    }

    /// Compute headroom analysis.
    pub fn headroom(&self, safety_margin_percent: f64) -> HeadroomAnalysis {
        let current_used = self.samples.last().map(|s| s.used).unwrap_or(0.0);
        let headroom_absolute = (self.total_capacity - current_used).max(0.0);
        let headroom_percent = if self.total_capacity > 0.0 {
            (headroom_absolute / self.total_capacity) * 100.0
        } else {
            0.0
        };

        HeadroomAnalysis {
            resource: self.name.clone(),
            current_used,
            total_capacity: self.total_capacity,
            headroom_absolute,
            headroom_percent,
            sufficient: headroom_percent >= safety_margin_percent,
            safety_margin_percent,
        }
    }

    /// What-if analysis: what happens if we add more load?
    pub fn what_if(&self, additional_load: f64, threshold_percent: f64) -> WhatIfResult {
        let current_used = self.samples.last().map(|s| s.used).unwrap_or(0.0);
        let current_pct = if self.total_capacity > 0.0 {
            (current_used / self.total_capacity) * 100.0
        } else {
            0.0
        };
        let projected_used = current_used + additional_load;
        let projected_pct = if self.total_capacity > 0.0 {
            (projected_used / self.total_capacity) * 100.0
        } else {
            0.0
        };

        WhatIfResult {
            scenario: format!(
                "Add {:.1} units to {}",
                additional_load, self.name
            ),
            original_utilization_percent: current_pct,
            projected_utilization_percent: projected_pct,
            additional_load,
            would_exceed_threshold: projected_pct > threshold_percent,
            threshold_percent,
        }
    }

    /// Get a scaling recommendation based on current and trend data.
    pub fn scaling_recommendation(&self, unit_capacity: f64) -> ScalingAction {
        let current = self.current_utilization().unwrap_or(0.0);

        if current >= self.critical_threshold {
            let needed = current - self.warning_threshold;
            let units = if unit_capacity > 0.0 {
                (needed * self.total_capacity / (100.0 * unit_capacity)).ceil() as u64
            } else {
                1
            };
            return ScalingAction::ScaleUp {
                additional_units: units.max(1),
                reason: format!(
                    "{} at {:.1}%, exceeds critical threshold {:.1}%",
                    self.name, current, self.critical_threshold
                ),
            };
        }

        if current >= self.warning_threshold {
            return ScalingAction::Alert {
                message: format!(
                    "{} at {:.1}%, approaching critical threshold {:.1}%",
                    self.name, current, self.critical_threshold
                ),
            };
        }

        if current < 20.0 && self.samples.len() > 10 {
            let avg = self.average_utilization();
            if avg < 20.0 {
                let excess = ((self.total_capacity * (1.0 - avg / 100.0)) / unit_capacity.max(1.0))
                    .floor() as u64;
                if excess > 0 {
                    return ScalingAction::ScaleDown {
                        excess_units: excess,
                        reason: format!(
                            "{} average utilization {:.1}%, resources underused",
                            self.name, avg
                        ),
                    };
                }
            }
        }

        ScalingAction::NoAction
    }
}

// ── Capacity Planner ──

/// Multi-resource capacity planner.
#[derive(Debug)]
pub struct CapacityPlanner {
    trackers: HashMap<String, ResourceTracker>,
}

impl CapacityPlanner {
    pub fn new() -> Self {
        Self {
            trackers: HashMap::new(),
        }
    }

    pub fn add_resource(&mut self, tracker: ResourceTracker) {
        self.trackers.insert(tracker.name.clone(), tracker);
    }

    pub fn get(&self, name: &str) -> Option<&ResourceTracker> {
        self.trackers.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut ResourceTracker> {
        self.trackers.get_mut(name)
    }

    pub fn resource_count(&self) -> usize {
        self.trackers.len()
    }

    /// Record utilization for a named resource.
    pub fn record(&mut self, resource_name: &str, used: f64) {
        if let Some(tracker) = self.trackers.get_mut(resource_name) {
            tracker.record(used);
        }
    }

    /// Get headroom for all resources.
    pub fn headroom_report(&self, safety_margin: f64) -> Vec<HeadroomAnalysis> {
        let mut results: Vec<HeadroomAnalysis> = self
            .trackers
            .values()
            .map(|t| t.headroom(safety_margin))
            .collect();
        results.sort_by(|a, b| {
            a.headroom_percent
                .partial_cmp(&b.headroom_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    /// Get scaling recommendations for all resources.
    pub fn scaling_recommendations(&self, unit_capacity: f64) -> Vec<(String, ScalingAction)> {
        let mut results: Vec<(String, ScalingAction)> = self
            .trackers
            .values()
            .map(|t| (t.name.clone(), t.scaling_recommendation(unit_capacity)))
            .collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    }
}

impl Default for CapacityPlanner {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timestamps(count: usize, interval_secs: i64) -> Vec<DateTime<Utc>> {
        let base = Utc::now() - Duration::seconds(interval_secs * count as i64);
        (0..count)
            .map(|i| base + Duration::seconds(interval_secs * i as i64))
            .collect()
    }

    #[test]
    fn test_utilization_sample() {
        let s = UtilizationSample::new(Utc::now(), 70.0, 100.0);
        assert!((s.utilization_percent() - 70.0).abs() < f64::EPSILON);
        assert!((s.available() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_regression_fit() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let reg = LinearRegression::fit(&xs, &ys).unwrap();
        assert!((reg.slope - 2.0).abs() < 0.01);
        assert!((reg.intercept - 0.0).abs() < 0.01);
        assert!(reg.r_squared > 0.99);
    }

    #[test]
    fn test_linear_regression_predict() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 3.0, 5.0, 7.0];
        let reg = LinearRegression::fit(&xs, &ys).unwrap();
        assert!((reg.predict(4.0) - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_solve_for_x() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        let reg = LinearRegression::fit(&xs, &ys).unwrap();
        let x = reg.solve_for_x(50.0).unwrap();
        assert!((x - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_regression_too_few_points() {
        assert!(LinearRegression::fit(&[1.0], &[1.0]).is_none());
    }

    #[test]
    fn test_resource_tracker_record() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0);
        t.record(50.0);
        t.record(60.0);
        assert_eq!(t.sample_count(), 2);
        assert!((t.current_utilization().unwrap() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_tracker_average() {
        let mut t = ResourceTracker::new("mem", ResourceKind::Memory, 100.0);
        t.record(40.0);
        t.record(60.0);
        assert!((t.average_utilization() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_tracker_peak() {
        let mut t = ResourceTracker::new("disk", ResourceKind::Disk, 100.0);
        t.record(20.0);
        t.record(80.0);
        t.record(50.0);
        assert!((t.peak_utilization() - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_trend() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0);
        let times = make_timestamps(5, 60);
        for (i, ts) in times.iter().enumerate() {
            t.record_at(*ts, 20.0 + i as f64 * 10.0);
        }
        let trend = t.trend().unwrap();
        assert!(trend.slope > 0.0);
        assert!(trend.r_squared > 0.9);
    }

    #[test]
    fn test_threshold_forecast_rising() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0);
        let times = make_timestamps(10, 3600);
        for (i, ts) in times.iter().enumerate() {
            t.record_at(*ts, 30.0 + i as f64 * 5.0);
        }
        let forecast = t.forecast_threshold(80.0);
        assert_eq!(forecast.trend_direction, TrendDirection::Rising);
        assert!(forecast.estimated_time_to_threshold.is_some());
    }

    #[test]
    fn test_threshold_forecast_stable() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0);
        let times = make_timestamps(10, 3600);
        for ts in &times {
            t.record_at(*ts, 50.0);
        }
        let forecast = t.forecast_threshold(80.0);
        assert_eq!(forecast.trend_direction, TrendDirection::Stable);
    }

    #[test]
    fn test_headroom_analysis() {
        let mut t = ResourceTracker::new("mem", ResourceKind::Memory, 1000.0);
        t.record(600.0);
        let h = t.headroom(20.0);
        assert!((h.headroom_absolute - 400.0).abs() < f64::EPSILON);
        assert!((h.headroom_percent - 40.0).abs() < f64::EPSILON);
        assert!(h.sufficient);
    }

    #[test]
    fn test_headroom_insufficient() {
        let mut t = ResourceTracker::new("disk", ResourceKind::Disk, 100.0);
        t.record(95.0);
        let h = t.headroom(20.0);
        assert!(!h.sufficient);
    }

    #[test]
    fn test_what_if_exceed() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0);
        t.record(70.0);
        let result = t.what_if(25.0, 90.0);
        assert!(result.would_exceed_threshold);
        assert!((result.projected_utilization_percent - 95.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_what_if_safe() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0);
        t.record(50.0);
        let result = t.what_if(10.0, 90.0);
        assert!(!result.would_exceed_threshold);
    }

    #[test]
    fn test_scaling_critical() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0)
            .with_thresholds(70.0, 90.0);
        t.record(95.0);
        match t.scaling_recommendation(10.0) {
            ScalingAction::ScaleUp {
                additional_units,
                reason,
            } => {
                assert!(additional_units >= 1);
                assert!(reason.contains("critical"));
            }
            other => panic!("Expected ScaleUp, got {:?}", other),
        }
    }

    #[test]
    fn test_scaling_warning() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0)
            .with_thresholds(70.0, 90.0);
        t.record(75.0);
        match t.scaling_recommendation(10.0) {
            ScalingAction::Alert { message } => {
                assert!(message.contains("approaching"));
            }
            other => panic!("Expected Alert, got {:?}", other),
        }
    }

    #[test]
    fn test_scaling_no_action() {
        let mut t = ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0)
            .with_thresholds(70.0, 90.0);
        t.record(50.0);
        assert_eq!(t.scaling_recommendation(10.0), ScalingAction::NoAction);
    }

    #[test]
    fn test_capacity_planner_multi_resource() {
        let mut planner = CapacityPlanner::new();
        planner.add_resource(ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0));
        planner.add_resource(ResourceTracker::new("mem", ResourceKind::Memory, 1024.0));
        assert_eq!(planner.resource_count(), 2);

        planner.record("cpu", 60.0);
        planner.record("mem", 800.0);

        let report = planner.headroom_report(20.0);
        assert_eq!(report.len(), 2);
    }

    #[test]
    fn test_trend_direction_from_slope() {
        assert_eq!(TrendDirection::from_slope(0.5), TrendDirection::Rising);
        assert_eq!(TrendDirection::from_slope(-0.5), TrendDirection::Falling);
        assert_eq!(TrendDirection::from_slope(0.0001), TrendDirection::Stable);
    }

    #[test]
    fn test_resource_kind_as_str() {
        assert_eq!(ResourceKind::Cpu.as_str(), "cpu");
        assert_eq!(ResourceKind::Memory.as_str(), "memory");
        assert_eq!(ResourceKind::Disk.as_str(), "disk");
        assert_eq!(ResourceKind::Network.as_str(), "network");
        assert_eq!(ResourceKind::Connections.as_str(), "connections");
        assert_eq!(ResourceKind::Custom.as_str(), "custom");
    }

    #[test]
    fn test_planner_get() {
        let mut planner = CapacityPlanner::new();
        planner.add_resource(ResourceTracker::new("cpu", ResourceKind::Cpu, 100.0));
        assert!(planner.get("cpu").is_some());
        assert!(planner.get("nonexistent").is_none());
    }
}
