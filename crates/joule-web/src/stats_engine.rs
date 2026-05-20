//! Statistical analysis engine — pure-Rust replacement for simple-statistics, jstat.
//!
//! Mean, median, mode, variance, std dev, skewness, kurtosis, percentiles, z-score,
//! t-test, chi-squared test, correlation (Pearson, Spearman), linear regression,
//! confidence intervals.

use std::fmt;

const EPS: f64 = 1e-12;

// ── Descriptive stats ─────────────────────────────────────────

/// Arithmetic mean.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Median (middle value or average of two middle values).
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Mode (most frequent value). Returns the smallest mode if there are ties.
pub fn mode(data: &[f64]) -> Option<f64> {
    if data.is_empty() { return None; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut best_val = sorted[0];
    let mut best_count = 1usize;
    let mut cur_val = sorted[0];
    let mut cur_count = 1usize;

    for &v in &sorted[1..] {
        if (v - cur_val).abs() < EPS {
            cur_count += 1;
        } else {
            if cur_count > best_count {
                best_count = cur_count;
                best_val = cur_val;
            }
            cur_val = v;
            cur_count = 1;
        }
    }
    if cur_count > best_count {
        best_val = cur_val;
    }
    Some(best_val)
}

/// Population variance.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / data.len() as f64
}

/// Sample variance (Bessel's correction).
pub fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (data.len() - 1) as f64
}

/// Population standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Sample standard deviation.
pub fn sample_std_dev(data: &[f64]) -> f64 {
    sample_variance(data).sqrt()
}

/// Skewness (Fisher's definition, sample-based).
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 { return 0.0; }
    let m = mean(data);
    let s = sample_std_dev(data);
    if s < EPS { return 0.0; }
    let sum: f64 = data.iter().map(|x| ((x - m) / s).powi(3)).sum();
    sum * n / ((n - 1.0) * (n - 2.0))
}

/// Excess kurtosis (sample-based).
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 { return 0.0; }
    let m = mean(data);
    let s = sample_std_dev(data);
    if s < EPS { return 0.0; }
    let sum: f64 = data.iter().map(|x| ((x - m) / s).powi(4)).sum();
    let k = (n * (n + 1.0) * sum) / ((n - 1.0) * (n - 2.0) * (n - 3.0))
        - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));
    k
}

/// Percentile (linear interpolation). `p` should be in [0, 100].
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 { return sorted[0]; }

    let rank = (p / 100.0) * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;

    if lower >= n { return sorted[n - 1]; }
    if upper >= n { return sorted[n - 1]; }
    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

/// Z-score (standard score) of a value given population mean and std dev.
pub fn z_score(value: f64, pop_mean: f64, pop_std_dev: f64) -> f64 {
    if pop_std_dev.abs() < EPS { return 0.0; }
    (value - pop_mean) / pop_std_dev
}

// ── Correlation ───────────────────────────────────────────────

/// Pearson correlation coefficient.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Arrays must have equal length");
    let n = x.len();
    if n < 2 { return 0.0; }

    let mx = mean(x);
    let my = mean(y);
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < EPS { return 0.0; }
    num / denom
}

/// Spearman rank correlation coefficient.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Arrays must have equal length");
    let n = x.len();
    if n < 2 { return 0.0; }

    let rank_x = ranks(x);
    let rank_y = ranks(y);
    pearson_correlation(&rank_x, &rank_y)
}

/// Compute ranks for a slice (average ranks for ties).
fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < EPS {
            j += 1;
        }
        // Average rank for ties
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            result[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    result
}

// ── Linear regression ─────────────────────────────────────────

/// Result of simple linear regression y = slope * x + intercept.
#[derive(Debug, Clone, Copy)]
pub struct LinearRegression {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

impl LinearRegression {
    /// Predict y for a given x.
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

impl fmt::Display for LinearRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "y = {:.4}x + {:.4} (R^2 = {:.4})", self.slope, self.intercept, self.r_squared)
    }
}

/// Simple linear regression (ordinary least squares).
pub fn linear_regression(x: &[f64], y: &[f64]) -> LinearRegression {
    assert_eq!(x.len(), y.len(), "Arrays must have equal length");
    let n = x.len() as f64;
    if n < 2.0 {
        return LinearRegression { slope: 0.0, intercept: y.first().copied().unwrap_or(0.0), r_squared: 0.0 };
    }

    let mx = mean(x);
    let my = mean(y);
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

    let slope = if ss_xx.abs() < EPS { 0.0 } else { ss_xy / ss_xx };
    let intercept = my - slope * mx;
    let r_squared = if ss_yy.abs() < EPS { 1.0 } else { (ss_xy * ss_xy) / (ss_xx * ss_yy) };

    LinearRegression { slope, intercept, r_squared }
}

// ── Hypothesis tests ──────────────────────────────────────────

/// Result of a statistical test.
#[derive(Debug, Clone, Copy)]
pub struct TestResult {
    pub statistic: f64,
    pub degrees_of_freedom: f64,
    /// Approximate p-value (using normal approximation for large df).
    pub p_value: f64,
}

/// One-sample t-test: test whether the mean of `data` equals `mu`.
pub fn t_test_one_sample(data: &[f64], mu: f64) -> TestResult {
    let n = data.len() as f64;
    let m = mean(data);
    let s = sample_std_dev(data);
    let t = if s < EPS { 0.0 } else { (m - mu) / (s / n.sqrt()) };
    let df = n - 1.0;
    let p = approximate_t_pvalue(t.abs(), df);
    TestResult { statistic: t, degrees_of_freedom: df, p_value: p }
}

/// Two-sample t-test (Welch's, unequal variances).
pub fn t_test_two_sample(a: &[f64], b: &[f64]) -> TestResult {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let ma = mean(a);
    let mb = mean(b);
    let sa2 = sample_variance(a);
    let sb2 = sample_variance(b);

    let se = (sa2 / na + sb2 / nb).sqrt();
    let t = if se < EPS { 0.0 } else { (ma - mb) / se };

    // Welch-Satterthwaite degrees of freedom
    let num = (sa2 / na + sb2 / nb).powi(2);
    let denom = (sa2 / na).powi(2) / (na - 1.0) + (sb2 / nb).powi(2) / (nb - 1.0);
    let df = if denom < EPS { 1.0 } else { num / denom };

    let p = approximate_t_pvalue(t.abs(), df);
    TestResult { statistic: t, degrees_of_freedom: df, p_value: p }
}

/// Chi-squared goodness-of-fit test.
/// `observed` and `expected` must have the same length.
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> TestResult {
    assert_eq!(observed.len(), expected.len(), "Arrays must have equal length");
    let mut chi2 = 0.0;
    for i in 0..observed.len() {
        if expected[i].abs() < EPS { continue; }
        let diff = observed[i] - expected[i];
        chi2 += (diff * diff) / expected[i];
    }
    let df = (observed.len() as f64 - 1.0).max(1.0);
    let p = approximate_chi2_pvalue(chi2, df);
    TestResult { statistic: chi2, degrees_of_freedom: df, p_value: p }
}

// ── Confidence intervals ──────────────────────────────────────

/// Confidence interval for the mean.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub point_estimate: f64,
    pub confidence_level: f64,
}

/// Compute a confidence interval for the mean using z-approximation.
/// `confidence` is in (0, 1), e.g. 0.95 for a 95% CI.
pub fn confidence_interval_mean(data: &[f64], confidence: f64) -> ConfidenceInterval {
    let n = data.len() as f64;
    let m = mean(data);
    let s = sample_std_dev(data);
    let z = z_critical(confidence);
    let margin = z * s / n.sqrt();
    ConfidenceInterval {
        lower: m - margin,
        upper: m + margin,
        point_estimate: m,
        confidence_level: confidence,
    }
}

// ── Internal approximations ───────────────────────────────────

/// Approximate z-critical value for given confidence level.
/// Uses Abramowitz & Stegun rational approximation for the normal quantile.
fn z_critical(confidence: f64) -> f64 {
    let alpha = 1.0 - confidence;
    let p = 1.0 - alpha / 2.0;
    // Beasley-Springer-Moro approximation
    inv_normal_cdf(p)
}

/// Approximate inverse normal CDF (probit function).
fn inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < EPS { return 0.0; }

    // Rational approximation (Abramowitz & Stegun 26.2.23)
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -val } else { val }
}

/// Approximate two-tailed p-value for t-distribution using normal approximation
/// (good for df > 30, rough for smaller df).
fn approximate_t_pvalue(t: f64, df: f64) -> f64 {
    // For large df, t ~ N(0,1)
    // For smaller df, use a correction
    let x = df / (df + t * t);
    // Regularized incomplete beta function approximation
    // Simple approach: use normal approx with correction
    let z = t * (1.0 - 1.0 / (4.0 * df)).max(0.0);
    2.0 * normal_sf(z.abs())
}

/// Approximate p-value for chi-squared distribution.
fn approximate_chi2_pvalue(chi2: f64, df: f64) -> f64 {
    // Wilson-Hilferty approximation: transform to standard normal
    if df < EPS { return 1.0; }
    let z = ((chi2 / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt();
    normal_sf(z)
}

/// Normal survival function (1 - CDF) approximation.
fn normal_sf(z: f64) -> f64 {
    // Abramowitz & Stegun approximation 7.1.26
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp()
        * (0.319381530 * t
            - 0.356563782 * t * t
            + 1.781477937 * t * t * t
            - 1.821255978 * t * t * t * t
            + 1.330274429 * t * t * t * t * t);
    if z >= 0.0 { p } else { 1.0 - p }
}

// ── Summary statistics ────────────────────────────────────────

/// Comprehensive summary of a dataset.
#[derive(Debug, Clone)]
pub struct Summary {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
}

/// Compute a full summary of a dataset.
pub fn summarize(data: &[f64]) -> Summary {
    let q1 = percentile(data, 25.0);
    let q3 = percentile(data, 75.0);
    Summary {
        count: data.len(),
        mean: mean(data),
        median: median(data),
        mode: mode(data),
        variance: sample_variance(data),
        std_dev: sample_std_dev(data),
        min: data.iter().cloned().fold(f64::INFINITY, f64::min),
        max: data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        skewness: skewness(data),
        kurtosis: kurtosis(data),
        q1,
        q3,
        iqr: q3 - q1,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < EPS);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_median_odd() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_median_even() {
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < EPS);
    }

    #[test]
    fn test_mode() {
        assert!((mode(&[1.0, 2.0, 2.0, 3.0]).unwrap() - 2.0).abs() < EPS);
        assert!((mode(&[5.0]).unwrap() - 5.0).abs() < EPS);
    }

    #[test]
    fn test_variance() {
        // Population variance of [2, 4, 4, 4, 5, 5, 7, 9]
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((variance(&data) - 4.0).abs() < EPS);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&data, 50.0) - 5.5).abs() < EPS);
        assert!((percentile(&data, 0.0) - 1.0).abs() < EPS);
        assert!((percentile(&data, 100.0) - 10.0).abs() < EPS);
    }

    #[test]
    fn test_z_score() {
        assert!((z_score(85.0, 80.0, 10.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // perfectly correlated
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_pearson_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!((pearson_correlation(&x, &y) - (-1.0)).abs() < EPS);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((spearman_correlation(&x, &y) - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let reg = linear_regression(&x, &y);
        assert!((reg.slope - 2.0).abs() < EPS);
        assert!(reg.intercept.abs() < EPS);
        assert!((reg.r_squared - 1.0).abs() < EPS);
    }

    #[test]
    fn test_linear_regression_predict() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0]; // y = 2x + 1
        let reg = linear_regression(&x, &y);
        assert!((reg.predict(6.0) - 13.0).abs() < EPS);
    }

    #[test]
    fn test_t_test_one_sample() {
        // Data clearly different from 0
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = t_test_one_sample(&data, 0.0);
        assert!(result.statistic > 10.0); // should be very significant
        assert!(result.p_value < 0.01);
    }

    #[test]
    fn test_t_test_two_sample() {
        let a = vec![5.0, 5.1, 4.9, 5.0, 5.2];
        let b = vec![10.0, 10.1, 9.9, 10.0, 10.2];
        let result = t_test_two_sample(&a, &b);
        assert!(result.statistic.abs() > 5.0); // clearly different
    }

    #[test]
    fn test_chi_squared() {
        let observed = vec![50.0, 30.0, 20.0];
        let expected = vec![40.0, 35.0, 25.0];
        let result = chi_squared_test(&observed, &expected);
        assert!(result.statistic > 0.0);
        assert_eq!(result.degrees_of_freedom, 2.0);
    }

    #[test]
    fn test_confidence_interval() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let ci = confidence_interval_mean(&data, 0.95);
        assert!(ci.lower < ci.point_estimate);
        assert!(ci.upper > ci.point_estimate);
        assert!((ci.confidence_level - 0.95).abs() < EPS);
    }

    #[test]
    fn test_summary() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = summarize(&data);
        assert_eq!(s.count, 5);
        assert!((s.mean - 3.0).abs() < EPS);
        assert!((s.median - 3.0).abs() < EPS);
        assert!((s.min - 1.0).abs() < EPS);
        assert!((s.max - 5.0).abs() < EPS);
    }

    #[test]
    fn test_skewness_symmetric() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(skewness(&data).abs() < 0.1); // roughly symmetric
    }

    #[test]
    fn test_kurtosis() {
        // Normal-like data should have kurtosis near 0
        let data: Vec<f64> = (0..1000).map(|i| {
            let x = i as f64 / 100.0;
            (-x * x / 2.0).exp()
        }).collect();
        let k = kurtosis(&data);
        // Just verify it computes without panicking and is finite
        assert!(k.is_finite());
    }

    #[test]
    fn test_sample_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sv = sample_variance(&data);
        // Sample variance = n/(n-1) * pop_variance = 8/7 * 4 = 32/7
        assert!((sv - 32.0 / 7.0).abs() < EPS);
    }
}
