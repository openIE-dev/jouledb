//! Descriptive & Inferential Statistics — pure-Rust replacement for simple-statistics,
//! jStat, and similar JS/TS libraries.

use std::collections::HashMap;
use std::fmt;

// ── Summary ────────────────────────────────────────────────────

/// All descriptive statistics computed in one pass (where possible).
#[derive(Debug, Clone)]
pub struct Summary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Vec<f64>,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
}

impl Summary {
    /// Compute all descriptive statistics for the given data.
    /// Returns `None` if data is empty.
    pub fn compute(data: &[f64]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let mn = mean(data)?;
        let med = median(data)?;
        let md = mode(data);
        let var = variance(data)?;
        let sd = std_dev(data)?;
        let lo = min(data)?;
        let hi = max(data)?;
        let q1_val = percentile(data, 25.0)?;
        let q3_val = percentile(data, 75.0)?;
        Some(Self {
            count: data.len(),
            mean: mn,
            median: med,
            mode: md,
            variance: var,
            std_dev: sd,
            min: lo,
            max: hi,
            range: hi - lo,
            q1: q1_val,
            q3: q3_val,
            iqr: q3_val - q1_val,
        })
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={} mean={:.4} median={:.4} std_dev={:.4} range=[{:.4}, {:.4}]",
            self.count, self.mean, self.median, self.std_dev, self.min, self.max
        )
    }
}

// ── Descriptive ────────────────────────────────────────────────

pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    Some(data.iter().sum::<f64>() / data.len() as f64)
}

pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    } else {
        Some(sorted[n / 2])
    }
}

/// Returns all values with the highest frequency.
pub fn mode(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in data {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let max_count = *counts.values().max().unwrap();
    let mut modes: Vec<f64> = counts
        .iter()
        .filter(|&(_, c)| *c == max_count)
        .map(|(&bits, _)| f64::from_bits(bits))
        .collect();
    modes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    modes
}

/// Population variance.
pub fn variance(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let avg = mean(data)?;
    let sum_sq: f64 = data.iter().map(|x| (x - avg).powi(2)).sum();
    Some(sum_sq / data.len() as f64)
}

/// Sample variance (Bessel's correction).
pub fn sample_variance(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        return None;
    }
    let avg = mean(data)?;
    let sum_sq: f64 = data.iter().map(|x| (x - avg).powi(2)).sum();
    Some(sum_sq / (data.len() - 1) as f64)
}

/// Population standard deviation.
pub fn std_dev(data: &[f64]) -> Option<f64> {
    variance(data).map(f64::sqrt)
}

/// Sample standard deviation.
pub fn sample_std_dev(data: &[f64]) -> Option<f64> {
    sample_variance(data).map(f64::sqrt)
}

pub fn min(data: &[f64]) -> Option<f64> {
    data.iter().copied().reduce(f64::min)
}

pub fn max(data: &[f64]) -> Option<f64> {
    data.iter().copied().reduce(f64::max)
}

pub fn range(data: &[f64]) -> Option<f64> {
    Some(max(data)? - min(data)?)
}

/// Percentile using linear interpolation (p in 0..100).
pub fn percentile(data: &[f64], p: f64) -> Option<f64> {
    if data.is_empty() || p < 0.0 || p > 100.0 {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n == 1 {
        return Some(sorted[0]);
    }
    let rank = (p / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    Some(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
}

/// Interquartile range.
pub fn iqr(data: &[f64]) -> Option<f64> {
    Some(percentile(data, 75.0)? - percentile(data, 25.0)?)
}

// ── Correlation ────────────────────────────────────────────────

/// Pearson correlation coefficient.
pub fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mx = mean(x)?;
    let my = mean(y)?;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    let denom = (sum_x2 * sum_y2).sqrt();
    if denom < 1e-15 {
        return None;
    }
    Some(sum_xy / denom)
}

// ── Linear regression ──────────────────────────────────────────

/// Simple linear regression result.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

impl LinearRegression {
    /// Fit y = slope*x + intercept.
    pub fn fit(x: &[f64], y: &[f64]) -> Option<Self> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }
        let mx = mean(x)?;
        let my = mean(y)?;
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        let mut ss_yy = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mx;
            let dy = y[i] - my;
            ss_xy += dx * dy;
            ss_xx += dx * dx;
            ss_yy += dy * dy;
        }
        if ss_xx < 1e-15 {
            return None;
        }
        let slope = ss_xy / ss_xx;
        let intercept = my - slope * mx;
        let r_squared = if ss_yy < 1e-15 {
            1.0
        } else {
            (ss_xy * ss_xy) / (ss_xx * ss_yy)
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
}

// ── Histogram ──────────────────────────────────────────────────

/// A histogram bin.
#[derive(Debug, Clone)]
pub struct HistogramBin {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
}

/// Bin data into `n_bins` equally-spaced buckets.
pub fn histogram(data: &[f64], n_bins: usize) -> Vec<HistogramBin> {
    if data.is_empty() || n_bins == 0 {
        return vec![];
    }
    let lo = data.iter().copied().reduce(f64::min).unwrap();
    let hi = data.iter().copied().reduce(f64::max).unwrap();
    let width = if (hi - lo).abs() < 1e-15 {
        1.0
    } else {
        (hi - lo) / n_bins as f64
    };
    let mut bins: Vec<HistogramBin> = (0..n_bins)
        .map(|i| HistogramBin {
            lo: lo + i as f64 * width,
            hi: lo + (i + 1) as f64 * width,
            count: 0,
        })
        .collect();
    for &v in data {
        let idx = if (hi - lo).abs() < 1e-15 {
            0
        } else {
            let i = ((v - lo) / width).floor() as usize;
            i.min(n_bins - 1)
        };
        bins[idx].count += 1;
    }
    bins
}

// ── Z-score normalization ──────────────────────────────────────

/// Return z-scores for each value.
pub fn z_scores(data: &[f64]) -> Option<Vec<f64>> {
    let avg = mean(data)?;
    let sd = std_dev(data)?;
    if sd < 1e-15 {
        return Some(vec![0.0; data.len()]);
    }
    Some(data.iter().map(|x| (x - avg) / sd).collect())
}

// ── Welch's t-test ─────────────────────────────────────────────

/// Result of a Welch's t-test (two-sample, unequal variance).
#[derive(Debug, Clone, Copy)]
pub struct WelchTTest {
    pub t_statistic: f64,
    pub degrees_of_freedom: f64,
}

/// Welch's t-test for two independent samples.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> Option<WelchTTest> {
    if a.len() < 2 || b.len() < 2 {
        return None;
    }
    let ma = mean(a)?;
    let mb = mean(b)?;
    let va = sample_variance(a)?;
    let vb = sample_variance(b)?;
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let se = (va / na + vb / nb).sqrt();
    if se < 1e-15 {
        return None;
    }
    let t = (ma - mb) / se;
    let va_n = va / na;
    let vb_n = vb / nb;
    let num = (va_n + vb_n).powi(2);
    let denom = va_n.powi(2) / (na - 1.0) + vb_n.powi(2) / (nb - 1.0);
    let df = num / denom;
    Some(WelchTTest {
        t_statistic: t,
        degrees_of_freedom: df,
    })
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_mean() {
        assert!(approx(mean(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(), 3.0));
        assert!(mean(&[]).is_none());
    }

    #[test]
    fn test_median_odd_even() {
        assert!(approx(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0));
        assert!(approx(median(&[4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5));
    }

    #[test]
    fn test_mode() {
        let m = mode(&[1.0, 2.0, 2.0, 3.0]);
        assert_eq!(m, vec![2.0]);
        let m2 = mode(&[1.0, 1.0, 2.0, 2.0]);
        assert_eq!(m2, vec![1.0, 2.0]);
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = variance(&data).unwrap();
        assert!(approx(var, 4.0));
        assert!(approx(std_dev(&data).unwrap(), 2.0));
    }

    #[test]
    fn test_percentile() {
        let data = [15.0, 20.0, 35.0, 40.0, 50.0];
        assert!(approx(percentile(&data, 0.0).unwrap(), 15.0));
        assert!(approx(percentile(&data, 100.0).unwrap(), 50.0));
        assert!(approx(percentile(&data, 50.0).unwrap(), 35.0));
    }

    #[test]
    fn test_iqr() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let q1 = percentile(&data, 25.0).unwrap();
        let q3 = percentile(&data, 75.0).unwrap();
        let iqr_val = iqr(&data).unwrap();
        assert!(approx(iqr_val, q3 - q1));
    }

    #[test]
    fn test_pearson_perfect() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson(&x, &y).unwrap();
        assert!(approx(r, 1.0));
    }

    #[test]
    fn test_linear_regression() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1
        let reg = LinearRegression::fit(&x, &y).unwrap();
        assert!(approx(reg.slope, 2.0));
        assert!(approx(reg.intercept, 1.0));
        assert!(approx(reg.r_squared, 1.0));
        assert!(approx(reg.predict(6.0), 13.0));
    }

    #[test]
    fn test_histogram() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let bins = histogram(&data, 5);
        assert_eq!(bins.len(), 5);
        let total: usize = bins.iter().map(|b| b.count).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_z_scores() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let zs = z_scores(&data).unwrap();
        assert_eq!(zs.len(), data.len());
        // mean of z-scores should be ~0
        let z_mean: f64 = zs.iter().sum::<f64>() / zs.len() as f64;
        assert!(approx(z_mean, 0.0));
    }

    #[test]
    fn test_welch_t_test() {
        let a = [27.5, 21.0, 19.0, 23.6, 17.0, 17.9, 16.9, 20.1, 21.9, 22.6];
        let b = [27.1, 22.0, 20.8, 23.4, 23.4, 23.5, 25.8, 22.0, 24.8, 20.2];
        let result = welch_t_test(&a, &b).unwrap();
        // t should be negative (a mean < b mean)
        assert!(result.t_statistic < 0.0);
        assert!(result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_summary() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let s = Summary::compute(&data).unwrap();
        assert_eq!(s.count, 5);
        assert!(approx(s.mean, 3.0));
        assert!(approx(s.median, 3.0));
        assert!(approx(s.min, 1.0));
        assert!(approx(s.max, 5.0));
        assert!(approx(s.range, 4.0));
    }

    #[test]
    fn test_sample_variance() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sv = sample_variance(&data).unwrap();
        // sample var = 32/7 ≈ 4.571
        assert!((sv - 32.0 / 7.0).abs() < 1e-10);
    }
}
