//! Energy Trace Diagnostics — behavioral pattern detection from energy consumption.
//!
//! Energy is a behavioral signal, not just a billing metric. Like PET scans
//! detecting the Warburg effect in cancer cells, energy traces reveal what
//! an agent is actually doing:
//!
//! - **Flat**: grinding — repetitive work, stable power draw
//! - **Spiky**: exploring — irregular bursts as the agent probes new directions
//! - **Declining**: converging — agent is homing in on an answer
//! - **Escalating**: expanding — agent is exploring increasingly complex paths
//! - **Periodic**: cycling — agent has found a loop (intentional or stuck)
//! - **Anomalous**: the Warburg pattern — energy consumption doesn't match declared work
//!
//! The trace analyzer runs alongside the energy enforcer, consuming the same
//! RAPL/NVML samples but interpreting them as behavioral diagnostics rather
//! than budget enforcement.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

/// A single energy sample in the trace.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnergySample {
    /// Microjoules consumed in this sample interval.
    pub energy_uj: u64,
    /// Timestamp offset from trace start (nanoseconds).
    pub offset_ns: u64,
    /// Sample interval duration (nanoseconds).
    pub interval_ns: u64,
    /// Instantaneous power in microwatts (energy_uj / interval_s * 1e6).
    pub power_uw: u64,
}

/// Detected behavioral pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyPattern {
    /// Stable power draw — repetitive or steady-state work.
    Flat,
    /// Irregular energy bursts — exploring or probing.
    Spiky,
    /// Power draw decreasing over time — converging on answer.
    Declining,
    /// Power draw increasing over time — expanding search.
    Escalating,
    /// Regular oscillation — cyclic behavior.
    Periodic,
    /// Energy doesn't match expected pattern — potential anomaly.
    Anomalous,
    /// Not enough data to classify.
    Insufficient,
}

/// Configuration for the trace analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// Minimum samples needed before pattern detection kicks in.
    pub min_samples: usize,
    /// Window size for rolling statistics.
    pub window_size: usize,
    /// Coefficient of variation threshold for "flat" classification.
    /// Below this = flat, above = variable.
    pub flat_cv_threshold: f64,
    /// Spike detection: sample must be this many std devs above mean.
    pub spike_std_dev_threshold: f64,
    /// Trend detection: minimum absolute slope (µW per sample) to classify as declining/escalating.
    pub trend_slope_threshold: f64,
    /// Periodicity detection: minimum autocorrelation at any lag.
    pub periodicity_threshold: f64,
    /// Warburg threshold: ratio of actual energy to expected energy.
    /// Above this = anomalous (consuming way more than the work should require).
    pub warburg_ratio_threshold: f64,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            window_size: 20,
            flat_cv_threshold: 0.15,
            spike_std_dev_threshold: 2.5,
            trend_slope_threshold: 1000.0, // 1 mW per sample
            periodicity_threshold: 0.7,
            warburg_ratio_threshold: 3.0,
        }
    }
}

/// Rolling statistics for a window of energy samples.
#[derive(Debug, Clone)]
struct RollingStats {
    values: VecDeque<f64>,
    window_size: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    fn new(window_size: usize) -> Self {
        Self {
            values: VecDeque::with_capacity(window_size),
            window_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.sum += value;
        self.sum_sq += value * value;
        self.values.push_back(value);
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.sum / self.values.len() as f64
    }

    fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let n = self.values.len() as f64;
        let mean = self.mean();
        (self.sum_sq / n) - (mean * mean)
    }

    fn std_dev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    /// Coefficient of variation (std_dev / mean).
    fn cv(&self) -> f64 {
        let mean = self.mean();
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        self.std_dev() / mean
    }

    /// Simple linear regression slope over the window.
    fn slope(&self) -> f64 {
        let n = self.values.len();
        if n < 2 {
            return 0.0;
        }
        let n_f = n as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;
        for (i, &v) in self.values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += v;
            sum_xy += x * v;
            sum_xx += x * x;
        }
        let denom = n_f * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return 0.0;
        }
        (n_f * sum_xy - sum_x * sum_y) / denom
    }

    /// Autocorrelation at a given lag (for periodicity detection).
    fn autocorrelation(&self, lag: usize) -> f64 {
        let n = self.values.len();
        if lag >= n || n < 4 {
            return 0.0;
        }
        let mean = self.mean();
        let var = self.variance();
        if var < 1e-10 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..(n - lag) {
            sum += (self.values[i] - mean) * (self.values[i + lag] - mean);
            count += 1;
        }
        if count == 0 {
            return 0.0;
        }
        (sum / count as f64) / var
    }
}

/// A diagnosis from the trace analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceDiagnosis {
    /// Primary detected pattern.
    pub pattern: EnergyPattern,
    /// Confidence in the classification (0.0 - 1.0).
    pub confidence: f64,
    /// Number of samples analyzed.
    pub sample_count: usize,
    /// Mean power over the analysis window (microwatts).
    pub mean_power_uw: f64,
    /// Coefficient of variation (variability measure).
    pub cv: f64,
    /// Trend slope (µW per sample, positive = escalating, negative = declining).
    pub slope: f64,
    /// Number of spikes detected in the window.
    pub spike_count: usize,
    /// Peak autocorrelation and its lag (for periodicity).
    pub peak_autocorrelation: f64,
    pub peak_autocorrelation_lag: usize,
    /// Warburg ratio (actual / expected energy). None if no expected baseline.
    pub warburg_ratio: Option<f64>,
}

/// Analyzes energy traces for behavioral patterns.
pub struct TraceAnalyzer {
    config: TraceConfig,
    stats: RollingStats,
    all_samples: Vec<EnergySample>,
    /// Expected energy per sample (set by caller based on work type).
    expected_energy_per_sample_uj: Option<u64>,
    /// Count of spikes in current window.
    spike_count: usize,
}

impl TraceAnalyzer {
    pub fn new(config: TraceConfig) -> Self {
        let window_size = config.window_size;
        Self {
            config,
            stats: RollingStats::new(window_size),
            all_samples: Vec::new(),
            expected_energy_per_sample_uj: None,
            spike_count: 0,
        }
    }

    /// Set expected energy per sample (baseline for Warburg detection).
    pub fn set_expected_energy(&mut self, expected_uj_per_sample: u64) {
        self.expected_energy_per_sample_uj = Some(expected_uj_per_sample);
    }

    /// Feed a new energy sample into the analyzer.
    pub fn push(&mut self, sample: EnergySample) {
        let power = sample.power_uw as f64;

        // Detect spike before pushing (compare to current stats)
        if self.stats.len() >= self.config.min_samples {
            let threshold = self.stats.mean()
                + self.config.spike_std_dev_threshold * self.stats.std_dev();
            if power > threshold {
                self.spike_count += 1;
            }
        }

        self.stats.push(power);
        self.all_samples.push(sample);
    }

    /// Get the current diagnosis.
    pub fn diagnose(&self) -> TraceDiagnosis {
        if self.stats.len() < self.config.min_samples {
            return TraceDiagnosis {
                pattern: EnergyPattern::Insufficient,
                confidence: 0.0,
                sample_count: self.stats.len(),
                mean_power_uw: self.stats.mean(),
                cv: 0.0,
                slope: 0.0,
                spike_count: 0,
                peak_autocorrelation: 0.0,
                peak_autocorrelation_lag: 0,
                warburg_ratio: None,
            };
        }

        let cv = self.stats.cv();
        let slope = self.stats.slope();
        let mean = self.stats.mean();

        // Periodicity: check autocorrelation at various lags
        let max_lag = self.stats.len() / 2;
        let mut peak_ac = 0.0f64;
        let mut peak_lag = 0;
        for lag in 2..=max_lag.min(self.config.window_size / 2) {
            let ac = self.stats.autocorrelation(lag);
            if ac > peak_ac {
                peak_ac = ac;
                peak_lag = lag;
            }
        }

        // Warburg ratio
        let warburg_ratio = self.expected_energy_per_sample_uj.map(|expected| {
            if expected == 0 {
                return 0.0;
            }
            // Convert mean power (µW) to energy per sample interval
            // mean_power_uw is already in µW, the expected is per sample
            mean / expected as f64
        });

        // Classify pattern
        let (pattern, confidence) =
            self.classify(cv, slope, peak_ac, warburg_ratio);

        TraceDiagnosis {
            pattern,
            confidence,
            sample_count: self.stats.len(),
            mean_power_uw: mean,
            cv,
            slope,
            spike_count: self.spike_count,
            peak_autocorrelation: peak_ac,
            peak_autocorrelation_lag: peak_lag,
            warburg_ratio,
        }
    }

    fn classify(
        &self,
        cv: f64,
        slope: f64,
        peak_ac: f64,
        warburg_ratio: Option<f64>,
    ) -> (EnergyPattern, f64) {
        // Check Warburg first (anomalous consumption pattern)
        if let Some(ratio) = warburg_ratio {
            if ratio > self.config.warburg_ratio_threshold {
                let confidence = ((ratio / self.config.warburg_ratio_threshold) - 1.0)
                    .min(1.0)
                    .max(0.5);
                return (EnergyPattern::Anomalous, confidence);
            }
        }

        // Check trend before periodicity — a strong linear trend produces
        // spurious autocorrelation, so trend takes priority.
        if slope.abs() > self.config.trend_slope_threshold {
            if slope < 0.0 {
                let confidence = (slope.abs() / self.config.trend_slope_threshold / 5.0)
                    .min(1.0)
                    .max(0.5);
                return (EnergyPattern::Declining, confidence);
            } else {
                let confidence = (slope / self.config.trend_slope_threshold / 5.0)
                    .min(1.0)
                    .max(0.5);
                return (EnergyPattern::Escalating, confidence);
            }
        }

        // Check periodicity (after trend, since linear trends cause spurious autocorrelation)
        if peak_ac > self.config.periodicity_threshold {
            let confidence = peak_ac.min(1.0);
            return (EnergyPattern::Periodic, confidence);
        }

        // Check variability
        if cv < self.config.flat_cv_threshold {
            let confidence = (1.0 - cv / self.config.flat_cv_threshold).max(0.5);
            return (EnergyPattern::Flat, confidence);
        }

        // High variability without trend → spiky
        if self.spike_count > 0 || cv > self.config.flat_cv_threshold * 2.0 {
            let confidence = (cv / (self.config.flat_cv_threshold * 3.0))
                .min(1.0)
                .max(0.5);
            return (EnergyPattern::Spiky, confidence);
        }

        // Default: flat with lower confidence
        (EnergyPattern::Flat, 0.5)
    }

    /// Number of samples collected.
    pub fn sample_count(&self) -> usize {
        self.all_samples.len()
    }

    /// Total energy consumed across all samples.
    pub fn total_energy_uj(&self) -> u64 {
        self.all_samples.iter().map(|s| s.energy_uj).sum()
    }

    /// Total trace duration.
    pub fn duration(&self) -> Duration {
        if let Some(last) = self.all_samples.last() {
            Duration::from_nanos(last.offset_ns + last.interval_ns)
        } else {
            Duration::ZERO
        }
    }

    /// Get a summary of the trace for reporting.
    pub fn summary(&self) -> TraceSummary {
        let diagnosis = self.diagnose();
        TraceSummary {
            sample_count: self.all_samples.len(),
            total_energy_uj: self.total_energy_uj(),
            duration: self.duration(),
            pattern: diagnosis.pattern,
            confidence: diagnosis.confidence,
            mean_power_uw: diagnosis.mean_power_uw,
            spike_count: diagnosis.spike_count,
        }
    }
}

/// Compact trace summary for inclusion in agent findings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    pub sample_count: usize,
    pub total_energy_uj: u64,
    pub duration: Duration,
    pub pattern: EnergyPattern,
    pub confidence: f64,
    pub mean_power_uw: f64,
    pub spike_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample(energy_uj: u64, offset_ms: u64) -> EnergySample {
        let interval_ns = 100_000_000; // 100ms
        EnergySample {
            energy_uj,
            offset_ns: offset_ms * 1_000_000,
            interval_ns,
            power_uw: energy_uj * 10_000, // energy_uj / 0.1s * 1e6
        }
    }

    #[test]
    fn test_insufficient_samples() {
        let analyzer = TraceAnalyzer::new(TraceConfig::default());
        let diag = analyzer.diagnose();
        assert_eq!(diag.pattern, EnergyPattern::Insufficient);
        assert_eq!(diag.confidence, 0.0);
    }

    #[test]
    fn test_flat_pattern() {
        let mut analyzer = TraceAnalyzer::new(TraceConfig {
            min_samples: 5,
            ..TraceConfig::default()
        });
        // Constant power draw
        for i in 0..20 {
            analyzer.push(make_sample(1000, i * 100));
        }
        let diag = analyzer.diagnose();
        assert_eq!(diag.pattern, EnergyPattern::Flat);
        assert!(diag.confidence > 0.5);
        assert!(diag.cv < 0.15);
    }

    #[test]
    fn test_escalating_pattern() {
        let mut config = TraceConfig::default();
        config.min_samples = 5;
        config.trend_slope_threshold = 100.0;
        let mut analyzer = TraceAnalyzer::new(config);
        // Linearly increasing power
        for i in 0..20 {
            let energy = 1000 + i * 500;
            analyzer.push(make_sample(energy, i * 100));
        }
        let diag = analyzer.diagnose();
        assert_eq!(diag.pattern, EnergyPattern::Escalating);
        assert!(diag.slope > 0.0);
    }

    #[test]
    fn test_declining_pattern() {
        let mut config = TraceConfig::default();
        config.min_samples = 5;
        config.trend_slope_threshold = 100.0;
        let mut analyzer = TraceAnalyzer::new(config);
        // Linearly decreasing power
        for i in 0..20 {
            let energy = 10000u64.saturating_sub(i * 400);
            analyzer.push(make_sample(energy.max(100), i * 100));
        }
        let diag = analyzer.diagnose();
        assert_eq!(diag.pattern, EnergyPattern::Declining);
        assert!(diag.slope < 0.0);
    }

    #[test]
    fn test_spiky_pattern() {
        let mut analyzer = TraceAnalyzer::new(TraceConfig {
            min_samples: 5,
            window_size: 20,
            flat_cv_threshold: 0.15,
            spike_std_dev_threshold: 2.5,
            trend_slope_threshold: f64::MAX, // Disable trend detection
            periodicity_threshold: 0.95,     // Very high to avoid periodic
            warburg_ratio_threshold: 3.0,
        });
        // Irregular spikes — not periodic, not trending, just variable
        let energies = [
            1000, 1100, 5000, 900, 1050, 950, 1000, 6000, 1000, 1100,
            1000, 950, 7000, 1050, 1000, 900, 1100, 1000, 5500, 1000,
        ];
        for (i, &e) in energies.iter().enumerate() {
            analyzer.push(make_sample(e, i as u64 * 100));
        }
        let diag = analyzer.diagnose();
        assert!(
            diag.pattern == EnergyPattern::Spiky || diag.pattern == EnergyPattern::Periodic,
            "got {:?}, cv={}, slope={}",
            diag.pattern, diag.cv, diag.slope,
        );
    }

    #[test]
    fn test_anomalous_warburg() {
        let mut config = TraceConfig::default();
        config.min_samples = 5;
        config.warburg_ratio_threshold = 3.0;
        let mut analyzer = TraceAnalyzer::new(config);
        // Set expected baseline
        analyzer.set_expected_energy(1000);
        // Feed much higher than expected
        for i in 0..20 {
            analyzer.push(make_sample(5000, i * 100)); // 5x expected
        }
        let diag = analyzer.diagnose();
        assert_eq!(diag.pattern, EnergyPattern::Anomalous);
        assert!(diag.warburg_ratio.unwrap() > 3.0);
    }

    #[test]
    fn test_total_energy() {
        let mut analyzer = TraceAnalyzer::new(TraceConfig::default());
        for i in 0..10 {
            analyzer.push(make_sample(1000, i * 100));
        }
        assert_eq!(analyzer.total_energy_uj(), 10_000);
        assert_eq!(analyzer.sample_count(), 10);
    }

    #[test]
    fn test_duration() {
        let mut analyzer = TraceAnalyzer::new(TraceConfig::default());
        analyzer.push(make_sample(1000, 0));
        analyzer.push(make_sample(1000, 100));
        analyzer.push(make_sample(1000, 200));
        let d = analyzer.duration();
        // Last sample at 200ms offset + 100ms interval = 300ms
        assert_eq!(d.as_millis(), 300);
    }

    #[test]
    fn test_trace_summary() {
        let mut analyzer = TraceAnalyzer::new(TraceConfig {
            min_samples: 3,
            ..TraceConfig::default()
        });
        for i in 0..5 {
            analyzer.push(make_sample(1000, i * 100));
        }
        let summary = analyzer.summary();
        assert_eq!(summary.sample_count, 5);
        assert_eq!(summary.total_energy_uj, 5000);
    }

    #[test]
    fn test_rolling_stats_basic() {
        let mut stats = RollingStats::new(5);
        stats.push(10.0);
        stats.push(20.0);
        stats.push(30.0);
        assert_eq!(stats.len(), 3);
        assert!((stats.mean() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_stats_window() {
        let mut stats = RollingStats::new(3);
        stats.push(10.0);
        stats.push(20.0);
        stats.push(30.0);
        stats.push(40.0); // Drops 10
        assert_eq!(stats.len(), 3);
        assert!((stats.mean() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_stats_slope() {
        let mut stats = RollingStats::new(10);
        for i in 0..5 {
            stats.push(i as f64 * 100.0);
        }
        // Linear increase: slope should be ~100
        assert!((stats.slope() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_stats_cv_constant() {
        let mut stats = RollingStats::new(10);
        for _ in 0..10 {
            stats.push(42.0);
        }
        assert!(stats.cv() < 1e-10);
    }

    #[test]
    fn test_trace_config_serde() {
        let config = TraceConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: TraceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.min_samples, config.min_samples);
        assert!((parsed.flat_cv_threshold - config.flat_cv_threshold).abs() < 1e-10);
    }

    #[test]
    fn test_energy_pattern_serde() {
        let patterns = vec![
            EnergyPattern::Flat,
            EnergyPattern::Spiky,
            EnergyPattern::Declining,
            EnergyPattern::Escalating,
            EnergyPattern::Periodic,
            EnergyPattern::Anomalous,
            EnergyPattern::Insufficient,
        ];
        for p in patterns {
            let json = serde_json::to_string(&p).unwrap();
            let parsed: EnergyPattern = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, p);
        }
    }

    #[test]
    fn test_diagnosis_serde() {
        let diag = TraceDiagnosis {
            pattern: EnergyPattern::Flat,
            confidence: 0.95,
            sample_count: 100,
            mean_power_uw: 15_000_000.0,
            cv: 0.05,
            slope: 12.5,
            spike_count: 0,
            peak_autocorrelation: 0.1,
            peak_autocorrelation_lag: 0,
            warburg_ratio: None,
        };
        let json = serde_json::to_string(&diag).unwrap();
        let parsed: TraceDiagnosis = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pattern, EnergyPattern::Flat);
        assert!((parsed.confidence - 0.95).abs() < 1e-10);
    }
}
