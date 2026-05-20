//! Anomaly detection: Z-score based detection, IQR method, EWMA (exponentially
//! weighted moving average), sliding window statistics, threshold alerting,
//! seasonal decomposition, and anomaly scoring.

use chrono::{DateTime, Utc};
use std::collections::VecDeque;

// ── Types ──

/// Kind of anomaly detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMethod {
    ZScore,
    Iqr,
    Ewma,
    SlidingWindow,
    Seasonal,
}

impl DetectionMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            DetectionMethod::ZScore => "z_score",
            DetectionMethod::Iqr => "iqr",
            DetectionMethod::Ewma => "ewma",
            DetectionMethod::SlidingWindow => "sliding_window",
            DetectionMethod::Seasonal => "seasonal",
        }
    }
}

/// Severity of a detected anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AnomalySeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            AnomalySeverity::Low => "low",
            AnomalySeverity::Medium => "medium",
            AnomalySeverity::High => "high",
            AnomalySeverity::Critical => "critical",
        }
    }
}

/// Direction of the anomaly relative to the baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyDirection {
    Above,
    Below,
    Either,
}

/// A single data point in a time series.
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub label: Option<String>,
}

impl DataPoint {
    pub fn new(timestamp: DateTime<Utc>, value: f64) -> Self {
        Self {
            timestamp,
            value,
            label: None,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

/// A detected anomaly.
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub expected: f64,
    pub deviation: f64,
    pub score: f64,
    pub severity: AnomalySeverity,
    pub method: DetectionMethod,
    pub direction: AnomalyDirection,
}

impl Anomaly {
    pub fn deviation_percent(&self) -> f64 {
        if self.expected.abs() < f64::EPSILON {
            return if self.value.abs() < f64::EPSILON {
                0.0
            } else {
                100.0
            };
        }
        ((self.value - self.expected) / self.expected).abs() * 100.0
    }
}

/// Alert threshold configuration.
#[derive(Debug, Clone)]
pub struct AlertThreshold {
    pub name: String,
    pub metric: String,
    pub upper_bound: Option<f64>,
    pub lower_bound: Option<f64>,
    pub consecutive_breaches: usize,
    pub current_breaches: usize,
    pub fired: bool,
}

impl AlertThreshold {
    pub fn new(name: &str, metric: &str) -> Self {
        Self {
            name: name.to_string(),
            metric: metric.to_string(),
            upper_bound: None,
            lower_bound: None,
            consecutive_breaches: 1,
            current_breaches: 0,
            fired: false,
        }
    }

    pub fn with_upper(mut self, bound: f64) -> Self {
        self.upper_bound = Some(bound);
        self
    }

    pub fn with_lower(mut self, bound: f64) -> Self {
        self.lower_bound = Some(bound);
        self
    }

    pub fn with_consecutive(mut self, count: usize) -> Self {
        self.consecutive_breaches = count;
        self
    }

    /// Check a value and return whether the alert just fired.
    pub fn check(&mut self, value: f64) -> bool {
        let breached = match (self.upper_bound, self.lower_bound) {
            (Some(upper), Some(lower)) => value > upper || value < lower,
            (Some(upper), None) => value > upper,
            (None, Some(lower)) => value < lower,
            (None, None) => false,
        };
        if breached {
            self.current_breaches += 1;
        } else {
            self.current_breaches = 0;
            self.fired = false;
        }
        if self.current_breaches >= self.consecutive_breaches && !self.fired {
            self.fired = true;
            return true;
        }
        false
    }

    pub fn reset(&mut self) {
        self.current_breaches = 0;
        self.fired = false;
    }
}

// ── Statistics Helpers ──

/// Compute mean of a slice.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation of a slice (population).
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

/// Compute the median of a slice.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Compute the first quartile (Q1).
fn quartile1(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    median(&sorted[..mid])
}

/// Compute the third quartile (Q3).
fn quartile3(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let start = if sorted.len() % 2 == 0 { mid } else { mid + 1 };
    if start >= sorted.len() {
        return sorted[sorted.len() - 1];
    }
    median(&sorted[start..])
}

// ── Z-Score Detector ──

/// Z-score based anomaly detector.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    pub threshold: f64,
    data: Vec<f64>,
}

impl ZScoreDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: if threshold > 0.0 { threshold } else { 3.0 },
            data: Vec::new(),
        }
    }

    pub fn add(&mut self, value: f64) {
        self.data.push(value);
    }

    pub fn add_all(&mut self, values: &[f64]) {
        self.data.extend_from_slice(values);
    }

    /// Compute z-score for a given value against the stored data.
    pub fn z_score(&self, value: f64) -> f64 {
        let m = mean(&self.data);
        let sd = std_dev(&self.data);
        if sd.abs() < f64::EPSILON {
            return 0.0;
        }
        (value - m) / sd
    }

    /// Check whether a value is anomalous.
    pub fn is_anomaly(&self, value: f64) -> bool {
        self.z_score(value).abs() > self.threshold
    }

    /// Detect anomalies in the stored data and return them.
    pub fn detect(&self) -> Vec<Anomaly> {
        let m = mean(&self.data);
        let sd = std_dev(&self.data);
        if sd.abs() < f64::EPSILON {
            return Vec::new();
        }
        let now = Utc::now();
        self.data
            .iter()
            .filter_map(|v| {
                let z = (v - m) / sd;
                if z.abs() > self.threshold {
                    let score = z.abs() / self.threshold;
                    let severity = severity_from_score(score);
                    let direction = if z > 0.0 {
                        AnomalyDirection::Above
                    } else {
                        AnomalyDirection::Below
                    };
                    Some(Anomaly {
                        timestamp: now,
                        value: *v,
                        expected: m,
                        deviation: z,
                        score,
                        severity,
                        method: DetectionMethod::ZScore,
                        direction,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn data_count(&self) -> usize {
        self.data.len()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

// ── IQR Detector ──

/// IQR (interquartile range) based anomaly detector.
#[derive(Debug, Clone)]
pub struct IqrDetector {
    /// Multiplier for the IQR to set the fence (typically 1.5).
    pub multiplier: f64,
    data: Vec<f64>,
}

impl IqrDetector {
    pub fn new(multiplier: f64) -> Self {
        Self {
            multiplier: if multiplier > 0.0 { multiplier } else { 1.5 },
            data: Vec::new(),
        }
    }

    pub fn add(&mut self, value: f64) {
        self.data.push(value);
    }

    pub fn add_all(&mut self, values: &[f64]) {
        self.data.extend_from_slice(values);
    }

    /// Compute the fences (lower, upper).
    pub fn fences(&self) -> (f64, f64) {
        let q1 = quartile1(&self.data);
        let q3 = quartile3(&self.data);
        let iqr = q3 - q1;
        (q1 - self.multiplier * iqr, q3 + self.multiplier * iqr)
    }

    pub fn is_anomaly(&self, value: f64) -> bool {
        let (lower, upper) = self.fences();
        value < lower || value > upper
    }

    pub fn detect(&self) -> Vec<Anomaly> {
        let (lower, upper) = self.fences();
        let med = median(&self.data);
        let now = Utc::now();
        self.data
            .iter()
            .filter_map(|v| {
                if *v < lower || *v > upper {
                    let deviation = if *v > upper {
                        v - upper
                    } else {
                        lower - v
                    };
                    let iqr = upper - lower;
                    let score = if iqr.abs() < f64::EPSILON {
                        1.0
                    } else {
                        deviation / (iqr / (2.0 * self.multiplier))
                    };
                    let direction = if *v > upper {
                        AnomalyDirection::Above
                    } else {
                        AnomalyDirection::Below
                    };
                    Some(Anomaly {
                        timestamp: now,
                        value: *v,
                        expected: med,
                        deviation,
                        score,
                        severity: severity_from_score(score),
                        method: DetectionMethod::Iqr,
                        direction,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn data_count(&self) -> usize {
        self.data.len()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

// ── EWMA Detector ──

/// Exponentially Weighted Moving Average detector.
#[derive(Debug, Clone)]
pub struct EwmaDetector {
    /// Smoothing factor (0 < alpha <= 1).
    pub alpha: f64,
    /// Number of standard deviations for anomaly threshold.
    pub threshold_sigma: f64,
    ewma: f64,
    ewma_var: f64,
    count: usize,
    anomalies: Vec<Anomaly>,
}

impl EwmaDetector {
    pub fn new(alpha: f64, threshold_sigma: f64) -> Self {
        let alpha = alpha.clamp(0.01, 1.0);
        Self {
            alpha,
            threshold_sigma: if threshold_sigma > 0.0 {
                threshold_sigma
            } else {
                3.0
            },
            ewma: 0.0,
            ewma_var: 0.0,
            count: 0,
            anomalies: Vec::new(),
        }
    }

    /// Feed a new value, returning whether it is anomalous.
    pub fn update(&mut self, value: f64, timestamp: DateTime<Utc>) -> bool {
        if self.count == 0 {
            self.ewma = value;
            self.ewma_var = 0.0;
            self.count = 1;
            return false;
        }
        self.count += 1;

        let diff = value - self.ewma;
        let new_ewma = self.alpha * value + (1.0 - self.alpha) * self.ewma;
        self.ewma_var =
            (1.0 - self.alpha) * (self.ewma_var + self.alpha * diff * diff);

        let sd = self.ewma_var.sqrt();
        let expected = self.ewma;
        self.ewma = new_ewma;

        if sd.abs() < f64::EPSILON {
            return false;
        }

        let z = diff.abs() / sd;
        if z > self.threshold_sigma {
            let score = z / self.threshold_sigma;
            let direction = if diff > 0.0 {
                AnomalyDirection::Above
            } else {
                AnomalyDirection::Below
            };
            self.anomalies.push(Anomaly {
                timestamp,
                value,
                expected,
                deviation: z,
                score,
                severity: severity_from_score(score),
                method: DetectionMethod::Ewma,
                direction,
            });
            return true;
        }
        false
    }

    pub fn current_ewma(&self) -> f64 {
        self.ewma
    }

    pub fn current_std_dev(&self) -> f64 {
        self.ewma_var.sqrt()
    }

    pub fn anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    pub fn sample_count(&self) -> usize {
        self.count
    }

    pub fn reset(&mut self) {
        self.ewma = 0.0;
        self.ewma_var = 0.0;
        self.count = 0;
        self.anomalies.clear();
    }
}

// ── Sliding Window Statistics ──

/// Sliding window statistics tracker.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    pub window_size: usize,
    buffer: VecDeque<f64>,
}

impl SlidingWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: if window_size > 0 { window_size } else { 100 },
            buffer: VecDeque::new(),
        }
    }

    pub fn push(&mut self, value: f64) {
        if self.buffer.len() >= self.window_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(value);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.window_size
    }

    pub fn mean(&self) -> f64 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        let vals: Vec<f64> = self.buffer.iter().copied().collect();
        mean(&vals)
    }

    pub fn std_dev(&self) -> f64 {
        if self.buffer.len() < 2 {
            return 0.0;
        }
        let vals: Vec<f64> = self.buffer.iter().copied().collect();
        std_dev(&vals)
    }

    pub fn min(&self) -> f64 {
        self.buffer
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.buffer
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn median(&self) -> f64 {
        let vals: Vec<f64> = self.buffer.iter().copied().collect();
        median(&vals)
    }

    pub fn values(&self) -> Vec<f64> {
        self.buffer.iter().copied().collect()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

// ── Seasonal Decomposition ──

/// Simple seasonal decomposition (additive model).
/// Decomposes a time series into trend, seasonal, and residual components.
#[derive(Debug, Clone)]
pub struct SeasonalDecomposition {
    pub period: usize,
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

impl SeasonalDecomposition {
    /// Decompose a time series with the given period.
    /// Uses a moving average for trend and averages the detrended values for seasonality.
    pub fn decompose(data: &[f64], period: usize) -> Option<Self> {
        if period == 0 || data.len() < period * 2 {
            return None;
        }

        // Compute trend using centered moving average
        let mut trend = vec![0.0; data.len()];
        let half = period / 2;
        for i in half..data.len().saturating_sub(half) {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(data.len());
            let window = &data[start..end];
            trend[i] = mean(window);
        }
        // Fill edges with nearest computed values
        for i in 0..half {
            trend[i] = trend[half];
        }
        let last_valid = data.len().saturating_sub(half + 1);
        for i in (last_valid + 1)..data.len() {
            trend[i] = trend[last_valid];
        }

        // Compute detrended series
        let detrended: Vec<f64> = data.iter().zip(trend.iter()).map(|(d, t)| d - t).collect();

        // Compute seasonal component by averaging detrended values at each phase
        let mut seasonal_pattern = vec![0.0; period];
        let mut counts = vec![0usize; period];
        for (i, &val) in detrended.iter().enumerate() {
            let phase = i % period;
            seasonal_pattern[phase] += val;
            counts[phase] += 1;
        }
        for i in 0..period {
            if counts[i] > 0 {
                seasonal_pattern[i] /= counts[i] as f64;
            }
        }
        // Center the seasonal component
        let seasonal_mean = mean(&seasonal_pattern);
        for v in &mut seasonal_pattern {
            *v -= seasonal_mean;
        }

        // Extend seasonal to full length
        let seasonal: Vec<f64> = (0..data.len())
            .map(|i| seasonal_pattern[i % period])
            .collect();

        // Residual = data - trend - seasonal
        let residual: Vec<f64> = data
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((d, t), s)| d - t - s)
            .collect();

        Some(Self {
            period,
            trend,
            seasonal,
            residual,
        })
    }

    /// Find anomalies in the residuals using a z-score threshold.
    pub fn find_anomalies(&self, threshold: f64) -> Vec<usize> {
        let m = mean(&self.residual);
        let sd = std_dev(&self.residual);
        if sd.abs() < f64::EPSILON {
            return Vec::new();
        }
        self.residual
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| {
                if ((v - m) / sd).abs() > threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Reconstruct the series from trend + seasonal.
    pub fn reconstructed(&self) -> Vec<f64> {
        self.trend
            .iter()
            .zip(self.seasonal.iter())
            .map(|(t, s)| t + s)
            .collect()
    }
}

// ── Anomaly Scorer ──

/// Composite anomaly scorer that combines multiple detection methods.
#[derive(Debug)]
pub struct AnomalyScorer {
    pub z_weight: f64,
    pub iqr_weight: f64,
    pub ewma_weight: f64,
}

impl AnomalyScorer {
    pub fn new(z_weight: f64, iqr_weight: f64, ewma_weight: f64) -> Self {
        Self {
            z_weight,
            iqr_weight,
            ewma_weight,
        }
    }

    /// Score a value given pre-computed z-score, iqr-distance, and ewma-distance.
    pub fn score(
        &self,
        z_score_abs: f64,
        iqr_distance: f64,
        ewma_distance: f64,
    ) -> f64 {
        let total_weight = self.z_weight + self.iqr_weight + self.ewma_weight;
        if total_weight.abs() < f64::EPSILON {
            return 0.0;
        }
        (self.z_weight * z_score_abs
            + self.iqr_weight * iqr_distance
            + self.ewma_weight * ewma_distance)
            / total_weight
    }

    /// Classify the composite score into a severity.
    pub fn classify(&self, score: f64) -> AnomalySeverity {
        severity_from_score(score)
    }
}

impl Default for AnomalyScorer {
    fn default() -> Self {
        Self {
            z_weight: 1.0,
            iqr_weight: 1.0,
            ewma_weight: 1.0,
        }
    }
}

// ── Helpers ──

fn severity_from_score(score: f64) -> AnomalySeverity {
    if score >= 3.0 {
        AnomalySeverity::Critical
    } else if score >= 2.0 {
        AnomalySeverity::High
    } else if score >= 1.5 {
        AnomalySeverity::Medium
    } else {
        AnomalySeverity::Low
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_zscore_no_anomaly() {
        let mut d = ZScoreDetector::new(3.0);
        d.add_all(&[10.0, 11.0, 10.5, 10.2, 9.8, 10.1, 10.3, 9.9]);
        assert!(!d.is_anomaly(10.0));
    }

    #[test]
    fn test_zscore_detects_outlier() {
        let mut d = ZScoreDetector::new(2.0);
        d.add_all(&[10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 50.0]);
        assert!(d.is_anomaly(50.0));
    }

    #[test]
    fn test_zscore_detect_returns_anomalies() {
        let mut d = ZScoreDetector::new(2.0);
        d.add_all(&[1.0, 1.1, 0.9, 1.0, 1.0, 1.1, 0.9, 100.0]);
        let anomalies = d.detect();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].method, DetectionMethod::ZScore);
    }

    #[test]
    fn test_zscore_clear() {
        let mut d = ZScoreDetector::new(3.0);
        d.add_all(&[1.0, 2.0, 3.0]);
        assert_eq!(d.data_count(), 3);
        d.clear();
        assert_eq!(d.data_count(), 0);
    }

    #[test]
    fn test_iqr_no_anomaly() {
        let mut d = IqrDetector::new(1.5);
        d.add_all(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
        assert!(!d.is_anomaly(13.0));
    }

    #[test]
    fn test_iqr_detects_outlier() {
        let mut d = IqrDetector::new(1.5);
        d.add_all(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(d.is_anomaly(100.0));
    }

    #[test]
    fn test_iqr_fences() {
        let mut d = IqrDetector::new(1.5);
        d.add_all(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let (lower, upper) = d.fences();
        assert!(lower < 1.0);
        assert!(upper > 8.0);
    }

    #[test]
    fn test_iqr_detect() {
        let mut d = IqrDetector::new(1.5);
        d.add_all(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]);
        let anomalies = d.detect();
        assert!(!anomalies.is_empty());
        assert_eq!(anomalies[0].method, DetectionMethod::Iqr);
    }

    #[test]
    fn test_ewma_baseline() {
        let mut d = EwmaDetector::new(0.3, 3.0);
        let now = Utc::now();
        // First value initializes — never anomalous
        assert!(!d.update(10.0, now));
        assert_eq!(d.sample_count(), 1);
    }

    #[test]
    fn test_ewma_detects_spike() {
        let mut d = EwmaDetector::new(0.3, 2.0);
        let now = Utc::now();
        for i in 0..20 {
            d.update(10.0 + (i as f64) * 0.01, now);
        }
        let result = d.update(1000.0, now);
        assert!(result);
        assert!(!d.anomalies().is_empty());
    }

    #[test]
    fn test_ewma_reset() {
        let mut d = EwmaDetector::new(0.3, 3.0);
        let now = Utc::now();
        d.update(10.0, now);
        d.update(20.0, now);
        d.reset();
        assert_eq!(d.sample_count(), 0);
        assert!(d.anomalies().is_empty());
    }

    #[test]
    fn test_sliding_window_stats() {
        let mut w = SlidingWindow::new(5);
        for v in [10.0, 20.0, 30.0, 40.0, 50.0] {
            w.push(v);
        }
        assert_eq!(w.len(), 5);
        assert!(w.is_full());
        assert!((w.mean() - 30.0).abs() < f64::EPSILON);
        assert!((w.min() - 10.0).abs() < f64::EPSILON);
        assert!((w.max() - 50.0).abs() < f64::EPSILON);
        assert!((w.median() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sliding_window_eviction() {
        let mut w = SlidingWindow::new(3);
        w.push(1.0);
        w.push(2.0);
        w.push(3.0);
        w.push(4.0);
        assert_eq!(w.len(), 3);
        let vals = w.values();
        assert_eq!(vals, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sliding_window_empty() {
        let w = SlidingWindow::new(10);
        assert!(w.is_empty());
        assert!(!w.is_full());
        assert!((w.mean() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_seasonal_decomposition() {
        let period = 4;
        let data: Vec<f64> = (0..20)
            .map(|i| {
                let trend = 10.0 + i as f64 * 0.5;
                let seasonal = [2.0, -1.0, 3.0, -4.0][i % period];
                trend + seasonal
            })
            .collect();
        let decomp = SeasonalDecomposition::decompose(&data, period);
        assert!(decomp.is_some());
        let decomp = decomp.unwrap();
        assert_eq!(decomp.trend.len(), 20);
        assert_eq!(decomp.seasonal.len(), 20);
        assert_eq!(decomp.residual.len(), 20);
    }

    #[test]
    fn test_seasonal_too_short() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(SeasonalDecomposition::decompose(&data, 4).is_none());
    }

    #[test]
    fn test_seasonal_find_anomalies() {
        let period = 4;
        let mut data: Vec<f64> = (0..40)
            .map(|i| {
                let trend = 10.0;
                let seasonal = [1.0, -1.0, 1.0, -1.0][i % period];
                trend + seasonal
            })
            .collect();
        // Inject anomaly
        data[20] = 100.0;
        let decomp = SeasonalDecomposition::decompose(&data, period).unwrap();
        let anomaly_indices = decomp.find_anomalies(2.0);
        assert!(anomaly_indices.contains(&20));
    }

    #[test]
    fn test_seasonal_reconstructed() {
        let period = 4;
        let data: Vec<f64> = (0..16)
            .map(|i| 10.0 + [1.0, -1.0, 1.0, -1.0][i % period])
            .collect();
        let decomp = SeasonalDecomposition::decompose(&data, period).unwrap();
        let recon = decomp.reconstructed();
        assert_eq!(recon.len(), 16);
    }

    #[test]
    fn test_alert_threshold_upper() {
        let mut alert = AlertThreshold::new("cpu_high", "cpu_percent")
            .with_upper(90.0);
        assert!(!alert.check(85.0));
        assert!(alert.check(95.0));
    }

    #[test]
    fn test_alert_threshold_lower() {
        let mut alert = AlertThreshold::new("mem_low", "free_memory_mb")
            .with_lower(100.0);
        assert!(!alert.check(200.0));
        assert!(alert.check(50.0));
    }

    #[test]
    fn test_alert_threshold_consecutive() {
        let mut alert = AlertThreshold::new("cpu", "cpu")
            .with_upper(80.0)
            .with_consecutive(3);
        assert!(!alert.check(90.0));
        assert!(!alert.check(91.0));
        assert!(alert.check(92.0)); // 3rd consecutive — fires
        assert!(!alert.check(93.0)); // Already fired
    }

    #[test]
    fn test_alert_threshold_reset_on_normal() {
        let mut alert = AlertThreshold::new("cpu", "cpu")
            .with_upper(80.0)
            .with_consecutive(3);
        alert.check(90.0);
        alert.check(91.0);
        alert.check(50.0); // Normal — resets counter
        assert!(!alert.check(90.0)); // Only 1st consecutive
    }

    #[test]
    fn test_anomaly_deviation_percent() {
        let a = Anomaly {
            timestamp: Utc::now(),
            value: 120.0,
            expected: 100.0,
            deviation: 2.0,
            score: 2.0,
            severity: AnomalySeverity::High,
            method: DetectionMethod::ZScore,
            direction: AnomalyDirection::Above,
        };
        assert!((a.deviation_percent() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_anomaly_scorer_composite() {
        let scorer = AnomalyScorer::new(1.0, 1.0, 1.0);
        let score = scorer.score(3.0, 2.0, 1.0);
        assert!((score - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_anomaly_scorer_classify() {
        let scorer = AnomalyScorer::default();
        assert_eq!(scorer.classify(0.5), AnomalySeverity::Low);
        assert_eq!(scorer.classify(1.5), AnomalySeverity::Medium);
        assert_eq!(scorer.classify(2.5), AnomalySeverity::High);
        assert_eq!(scorer.classify(3.5), AnomalySeverity::Critical);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AnomalySeverity::Low < AnomalySeverity::Medium);
        assert!(AnomalySeverity::Medium < AnomalySeverity::High);
        assert!(AnomalySeverity::High < AnomalySeverity::Critical);
    }

    #[test]
    fn test_detection_method_as_str() {
        assert_eq!(DetectionMethod::ZScore.as_str(), "z_score");
        assert_eq!(DetectionMethod::Iqr.as_str(), "iqr");
        assert_eq!(DetectionMethod::Ewma.as_str(), "ewma");
        assert_eq!(DetectionMethod::SlidingWindow.as_str(), "sliding_window");
        assert_eq!(DetectionMethod::Seasonal.as_str(), "seasonal");
    }

    #[test]
    fn test_datapoint_with_label() {
        let dp = DataPoint::new(Utc::now(), 42.0).with_label("cpu");
        assert_eq!(dp.label.as_deref(), Some("cpu"));
        assert!((dp.value - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ewma_current_values() {
        let mut d = EwmaDetector::new(0.5, 3.0);
        let now = Utc::now();
        d.update(10.0, now);
        d.update(20.0, now);
        // EWMA after second: 0.5*20 + 0.5*10 = 15.0
        assert!((d.current_ewma() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_zscore_identical_values() {
        let mut d = ZScoreDetector::new(3.0);
        d.add_all(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        // All identical — std dev is 0, z_score returns 0
        assert!(!d.is_anomaly(5.0));
        assert!(!d.is_anomaly(100.0)); // sd = 0 → z = 0
    }
}
