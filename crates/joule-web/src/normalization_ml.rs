//! Feature Normalization — min-max scaling, z-score standardization,
//! robust scaling, per-channel normalization, and L2 normalization
//! for ML preprocessing pipelines.
//!
//! Pure Rust, std-only. All operations use f64.

use std::fmt;

// ── Normalization Strategy ──────────────────────────────────────

/// Normalization method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMethod {
    MinMax,
    ZScore,
    Robust,
    L2,
    MaxAbs,
}

impl fmt::Display for NormMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinMax => write!(f, "min-max"),
            Self::ZScore => write!(f, "z-score"),
            Self::Robust => write!(f, "robust"),
            Self::L2 => write!(f, "l2"),
            Self::MaxAbs => write!(f, "max-abs"),
        }
    }
}

// ── Feature Stats ───────────────────────────────────────────────

/// Precomputed statistics for a single feature column.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub max_abs: f64,
}

impl FeatureStats {
    /// Compute all stats from a slice of values.
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                min: 0.0, max: 0.0, mean: 0.0, std_dev: 0.0,
                median: 0.0, q1: 0.0, q3: 0.0, max_abs: 0.0,
            };
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let median = percentile_sorted(&sorted, 50.0);
        let q1 = percentile_sorted(&sorted, 25.0);
        let q3 = percentile_sorted(&sorted, 75.0);
        let max_abs = sorted.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        Self { min, max, mean, std_dev, median, q1, q3, max_abs }
    }

    pub fn iqr(&self) -> f64 {
        self.q3 - self.q1
    }

    pub fn range(&self) -> f64 {
        self.max - self.min
    }
}

impl fmt::Display for FeatureStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stats(mean={:.4}, std={:.4}, min={:.4}, max={:.4})",
               self.mean, self.std_dev, self.min, self.max)
    }
}

/// Compute percentile from a pre-sorted slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    if lo == hi || hi >= sorted.len() {
        sorted[lo.min(sorted.len() - 1)]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ── Normalizer ──────────────────────────────────────────────────

/// A fitted normalizer that can transform and inverse-transform data.
#[derive(Debug, Clone)]
pub struct Normalizer {
    method: NormMethod,
    feature_stats: Vec<FeatureStats>,
    feature_range: (f64, f64),
    eps: f64,
}

impl Normalizer {
    pub fn new(method: NormMethod) -> Self {
        Self {
            method,
            feature_stats: Vec::new(),
            feature_range: (0.0, 1.0),
            eps: 1e-12,
        }
    }

    pub fn with_feature_range(mut self, lo: f64, hi: f64) -> Self {
        self.feature_range = (lo, hi);
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Fit the normalizer on column-major data (each inner Vec is one feature column).
    pub fn fit(&mut self, columns: &[Vec<f64>]) {
        self.feature_stats = columns.iter().map(|col| FeatureStats::compute(col)).collect();
    }

    /// Fit on row-major data (each inner Vec is one sample).
    pub fn fit_rows(&mut self, rows: &[Vec<f64>]) {
        if rows.is_empty() {
            return;
        }
        let n_features = rows[0].len();
        let mut columns = vec![Vec::with_capacity(rows.len()); n_features];
        for row in rows {
            for (j, &val) in row.iter().enumerate() {
                if j < n_features {
                    columns[j].push(val);
                }
            }
        }
        self.fit(&columns);
    }

    pub fn is_fitted(&self) -> bool {
        !self.feature_stats.is_empty()
    }

    pub fn num_features(&self) -> usize {
        self.feature_stats.len()
    }

    /// Transform a single row (sample).
    pub fn transform_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, &v)| self.transform_value(j, v))
            .collect()
    }

    /// Inverse-transform a single row.
    pub fn inverse_transform_row(&self, row: &[f64]) -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, &v)| self.inverse_value(j, v))
            .collect()
    }

    /// Transform all rows (row-major).
    pub fn transform_all(&self, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| self.transform_row(r)).collect()
    }

    /// Inverse-transform all rows.
    pub fn inverse_transform_all(&self, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| self.inverse_transform_row(r)).collect()
    }

    fn transform_value(&self, feature_idx: usize, val: f64) -> f64 {
        if feature_idx >= self.feature_stats.len() {
            return val;
        }
        let stats = &self.feature_stats[feature_idx];
        match self.method {
            NormMethod::MinMax => {
                let range = stats.range();
                if range.abs() < self.eps {
                    return 0.0;
                }
                let scaled = (val - stats.min) / range;
                let (lo, hi) = self.feature_range;
                scaled * (hi - lo) + lo
            }
            NormMethod::ZScore => {
                if stats.std_dev.abs() < self.eps {
                    return 0.0;
                }
                (val - stats.mean) / stats.std_dev
            }
            NormMethod::Robust => {
                let iqr = stats.iqr();
                if iqr.abs() < self.eps {
                    return 0.0;
                }
                (val - stats.median) / iqr
            }
            NormMethod::MaxAbs => {
                if stats.max_abs.abs() < self.eps {
                    return 0.0;
                }
                val / stats.max_abs
            }
            NormMethod::L2 => val, // L2 is per-sample, handled separately
        }
    }

    fn inverse_value(&self, feature_idx: usize, val: f64) -> f64 {
        if feature_idx >= self.feature_stats.len() {
            return val;
        }
        let stats = &self.feature_stats[feature_idx];
        match self.method {
            NormMethod::MinMax => {
                let range = stats.range();
                if range.abs() < self.eps {
                    return stats.min;
                }
                let (lo, hi) = self.feature_range;
                let scaled = (val - lo) / (hi - lo);
                scaled * range + stats.min
            }
            NormMethod::ZScore => val * stats.std_dev + stats.mean,
            NormMethod::Robust => val * stats.iqr() + stats.median,
            NormMethod::MaxAbs => val * stats.max_abs,
            NormMethod::L2 => val,
        }
    }
}

impl fmt::Display for Normalizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Normalizer(method={}, features={}, fitted={})",
               self.method, self.feature_stats.len(), self.is_fitted())
    }
}

// ── L2 Normalization (per-sample) ───────────────────────────────

/// Normalize a single row to unit L2 norm.
pub fn l2_normalize(row: &[f64]) -> Vec<f64> {
    let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return row.to_vec();
    }
    row.iter().map(|v| v / norm).collect()
}

/// Normalize each row in a matrix to unit L2 norm.
pub fn l2_normalize_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter().map(|r| l2_normalize(r)).collect()
}

// ── Per-Channel Normalization ───────────────────────────────────

/// Channel-wise normalization parameters (mean and std per channel).
#[derive(Debug, Clone)]
pub struct ChannelNormalizer {
    pub channel_means: Vec<f64>,
    pub channel_stds: Vec<f64>,
}

impl ChannelNormalizer {
    pub fn new(means: Vec<f64>, stds: Vec<f64>) -> Self {
        Self { channel_means: means, channel_stds: stds }
    }

    /// ImageNet-style defaults (3 channels).
    pub fn imagenet() -> Self {
        Self {
            channel_means: vec![0.485, 0.456, 0.406],
            channel_stds: vec![0.229, 0.224, 0.225],
        }
    }

    pub fn num_channels(&self) -> usize {
        self.channel_means.len()
    }

    /// Normalize a flat multi-channel image. Data layout: [C, H*W] flattened.
    pub fn normalize(&self, data: &[f64], pixels_per_channel: usize) -> Vec<f64> {
        let nc = self.num_channels();
        let mut out = data.to_vec();
        for c in 0..nc {
            let offset = c * pixels_per_channel;
            for i in 0..pixels_per_channel {
                if offset + i < out.len() {
                    out[offset + i] = (out[offset + i] - self.channel_means[c])
                        / self.channel_stds[c].max(1e-12);
                }
            }
        }
        out
    }

    /// Denormalize back to original scale.
    pub fn denormalize(&self, data: &[f64], pixels_per_channel: usize) -> Vec<f64> {
        let nc = self.num_channels();
        let mut out = data.to_vec();
        for c in 0..nc {
            let offset = c * pixels_per_channel;
            for i in 0..pixels_per_channel {
                if offset + i < out.len() {
                    out[offset + i] = out[offset + i] * self.channel_stds[c]
                        + self.channel_means[c];
                }
            }
        }
        out
    }
}

impl fmt::Display for ChannelNormalizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ChannelNorm(channels={})", self.num_channels())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_stats_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FeatureStats::compute(&vals);
        assert!((stats.mean - 3.0).abs() < 1e-9);
        assert!((stats.min - 1.0).abs() < 1e-9);
        assert!((stats.max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn feature_stats_median() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = FeatureStats::compute(&vals);
        assert!((stats.median - 3.0).abs() < 1e-9);
    }

    #[test]
    fn feature_stats_empty() {
        let stats = FeatureStats::compute(&[]);
        assert!((stats.mean - 0.0).abs() < 1e-9);
    }

    #[test]
    fn feature_stats_display() {
        let stats = FeatureStats::compute(&[1.0, 2.0, 3.0]);
        let txt = format!("{}", stats);
        assert!(txt.contains("mean="));
    }

    #[test]
    fn minmax_scaling() {
        let mut norm = Normalizer::new(NormMethod::MinMax);
        norm.fit(&[vec![0.0, 5.0, 10.0]]);
        let result = norm.transform_row(&[5.0]);
        assert!((result[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn minmax_inverse() {
        let mut norm = Normalizer::new(NormMethod::MinMax);
        norm.fit(&[vec![0.0, 10.0]]);
        let transformed = norm.transform_row(&[7.0]);
        let back = norm.inverse_transform_row(&transformed);
        assert!((back[0] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn minmax_custom_range() {
        let mut norm = Normalizer::new(NormMethod::MinMax).with_feature_range(-1.0, 1.0);
        norm.fit(&[vec![0.0, 100.0]]);
        let result = norm.transform_row(&[50.0]);
        assert!((result[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn zscore_standardization() {
        let mut norm = Normalizer::new(NormMethod::ZScore);
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        norm.fit(&[data]);
        let result = norm.transform_row(&[norm.feature_stats[0].mean]);
        assert!((result[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn zscore_inverse() {
        let mut norm = Normalizer::new(NormMethod::ZScore);
        norm.fit(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]]);
        let transformed = norm.transform_row(&[3.5]);
        let back = norm.inverse_transform_row(&transformed);
        assert!((back[0] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn robust_scaling() {
        let mut norm = Normalizer::new(NormMethod::Robust);
        norm.fit(&[vec![1.0, 2.0, 3.0, 4.0, 100.0]]);
        // Median should be 3.0, robust scaler uses median and IQR
        let result = norm.transform_row(&[norm.feature_stats[0].median]);
        assert!((result[0] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn maxabs_scaling() {
        let mut norm = Normalizer::new(NormMethod::MaxAbs);
        norm.fit(&[vec![-3.0, 1.0, 5.0]]);
        let result = norm.transform_row(&[5.0]);
        assert!((result[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn l2_normalize_unit() {
        let row = vec![3.0, 4.0];
        let normed = l2_normalize(&row);
        let norm: f64 = normed.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn l2_normalize_zero() {
        let row = vec![0.0, 0.0];
        let normed = l2_normalize(&row);
        assert_eq!(normed, vec![0.0, 0.0]);
    }

    #[test]
    fn normalizer_fit_rows() {
        let rows = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let mut norm = Normalizer::new(NormMethod::MinMax);
        norm.fit_rows(&rows);
        assert_eq!(norm.num_features(), 2);
        assert!(norm.is_fitted());
    }

    #[test]
    fn normalizer_display() {
        let norm = Normalizer::new(NormMethod::ZScore);
        let txt = format!("{}", norm);
        assert!(txt.contains("z-score"));
    }

    #[test]
    fn channel_normalizer_roundtrip() {
        let cn = ChannelNormalizer::imagenet();
        let data = vec![0.5; 12]; // 3 channels, 4 pixels each
        let normed = cn.normalize(&data, 4);
        let back = cn.denormalize(&normed, 4);
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn channel_normalizer_display() {
        let cn = ChannelNormalizer::imagenet();
        assert!(format!("{}", cn).contains("channels=3"));
    }

    #[test]
    fn transform_all_rows() {
        let mut norm = Normalizer::new(NormMethod::MinMax);
        norm.fit(&[vec![0.0, 10.0], vec![0.0, 100.0]]);
        let rows = vec![vec![5.0, 50.0], vec![10.0, 100.0]];
        let result = norm.transform_all(&rows);
        assert!((result[0][0] - 0.5).abs() < 1e-9);
        assert!((result[1][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn constant_feature_handling() {
        let mut norm = Normalizer::new(NormMethod::ZScore);
        norm.fit(&[vec![5.0, 5.0, 5.0]]);
        let result = norm.transform_row(&[5.0]);
        assert!((result[0] - 0.0).abs() < 1e-9); // std=0 => 0
    }
}
