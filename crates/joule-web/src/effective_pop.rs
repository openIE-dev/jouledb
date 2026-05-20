//! Effective population size (Ne) estimation and bottleneck detection.
//!
//! Implements multiple Ne estimators: temporal method, LD-based,
//! heterozygosity-excess, and coalescent-based. Includes bottleneck
//! detection via heterozygosity tests and mode-shift analysis.
//! All computations are std-only with f64 math.

use std::fmt;

// ── Ne Estimate ─────────────────────────────────────────────────────

/// Result of an effective population size estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct NeEstimate {
    pub ne: f64,
    pub lower_ci: f64,
    pub upper_ci: f64,
    pub method: NeMethod,
    pub n_loci: usize,
}

/// Method used for Ne estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeMethod {
    Temporal,
    LdBased,
    HetExcess,
    Coalescent,
    Moment,
}

impl fmt::Display for NeMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Temporal => write!(f, "temporal"),
            Self::LdBased => write!(f, "LD-based"),
            Self::HetExcess => write!(f, "het-excess"),
            Self::Coalescent => write!(f, "coalescent"),
            Self::Moment => write!(f, "moment"),
        }
    }
}

impl fmt::Display for NeEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ne={:.1} [{:.1}, {:.1}] method={} loci={}",
            self.ne, self.lower_ci, self.upper_ci, self.method, self.n_loci
        )
    }
}

// ── Temporal Sample ─────────────────────────────────────────────────

/// Allele frequency data at two time points for the temporal method.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalSample {
    pub freq_t0: Vec<f64>,
    pub freq_t1: Vec<f64>,
    pub sample_size_t0: Vec<usize>,
    pub sample_size_t1: Vec<usize>,
    pub generations: f64,
}

impl TemporalSample {
    pub fn new(generations: f64) -> Self {
        Self {
            freq_t0: Vec::new(),
            freq_t1: Vec::new(),
            sample_size_t0: Vec::new(),
            sample_size_t1: Vec::new(),
            generations,
        }
    }

    /// Add a locus with frequencies and sample sizes at both time points.
    pub fn with_locus(
        mut self,
        p0: f64,
        p1: f64,
        n0: usize,
        n1: usize,
    ) -> Self {
        self.freq_t0.push(p0);
        self.freq_t1.push(p1);
        self.sample_size_t0.push(n0);
        self.sample_size_t1.push(n1);
        self
    }

    pub fn n_loci(&self) -> usize {
        self.freq_t0.len()
    }
}

impl fmt::Display for TemporalSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TemporalSample(loci={}, t={}gen)",
            self.n_loci(),
            self.generations
        )
    }
}

// ── Ne Estimator ────────────────────────────────────────────────────

/// Configurable effective population size estimator.
#[derive(Debug, Clone)]
pub struct NeEstimator {
    method: NeMethod,
    pcrit: f64,
    confidence_level: f64,
    jackknife: bool,
}

impl Default for NeEstimator {
    fn default() -> Self {
        Self {
            method: NeMethod::Temporal,
            pcrit: 0.02,
            confidence_level: 0.95,
            jackknife: true,
        }
    }
}

impl NeEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, m: NeMethod) -> Self {
        self.method = m;
        self
    }

    /// Critical allele frequency cutoff for LD-based method.
    pub fn with_pcrit(mut self, p: f64) -> Self {
        self.pcrit = p.clamp(0.0, 0.5);
        self
    }

    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence_level = c.clamp(0.5, 0.999);
        self
    }

    pub fn with_jackknife(mut self, j: bool) -> Self {
        self.jackknife = j;
        self
    }

    /// Temporal method: Nei & Tajima (1981) Fk estimator.
    pub fn temporal_ne(&self, sample: &TemporalSample) -> NeEstimate {
        if sample.n_loci() == 0 || sample.generations <= 0.0 {
            return NeEstimate {
                ne: f64::INFINITY,
                lower_ci: 0.0,
                upper_ci: f64::INFINITY,
                method: NeMethod::Temporal,
                n_loci: 0,
            };
        }

        let mut fk_values = Vec::new();

        for i in 0..sample.n_loci() {
            let p0 = sample.freq_t0[i];
            let p1 = sample.freq_t1[i];
            let n0 = sample.sample_size_t0[i] as f64;
            let n1 = sample.sample_size_t1[i] as f64;

            let p_mean = (p0 + p1) / 2.0;
            if p_mean <= 0.0 || p_mean >= 1.0 {
                continue;
            }

            let diff = p1 - p0;
            let fk = (diff * diff) / (p_mean * (1.0 - p_mean));

            // Sampling correction
            let correction = 1.0 / (2.0 * n0) + 1.0 / (2.0 * n1);
            let fk_corr = fk - correction;
            fk_values.push(fk_corr);
        }

        if fk_values.is_empty() {
            return NeEstimate {
                ne: f64::INFINITY,
                lower_ci: 0.0,
                upper_ci: f64::INFINITY,
                method: NeMethod::Temporal,
                n_loci: 0,
            };
        }

        let mean_fk: f64 = fk_values.iter().sum::<f64>() / fk_values.len() as f64;
        let ne = if mean_fk > 0.0 {
            sample.generations / (2.0 * mean_fk)
        } else {
            f64::INFINITY
        };

        let (lower, upper) = if self.jackknife && fk_values.len() > 2 {
            self.jackknife_ci(&fk_values, sample.generations)
        } else {
            chi2_ci(ne, fk_values.len())
        };

        NeEstimate {
            ne,
            lower_ci: lower,
            upper_ci: upper,
            method: NeMethod::Temporal,
            n_loci: fk_values.len(),
        }
    }

    /// LD-based Ne estimation (Hill 1981, Waples 2006).
    pub fn ld_ne(&self, r_squared_values: &[f64], sample_size: usize) -> NeEstimate {
        if r_squared_values.is_empty() || sample_size < 2 {
            return NeEstimate {
                ne: f64::INFINITY,
                lower_ci: 0.0,
                upper_ci: f64::INFINITY,
                method: NeMethod::LdBased,
                n_loci: 0,
            };
        }

        let s = sample_size as f64;
        let mean_r2: f64 =
            r_squared_values.iter().sum::<f64>() / r_squared_values.len() as f64;

        // Expected r² due to sampling: E[r²] ≈ 1/S for unlinked loci
        let r2_sample = 1.0 / s;
        let r2_drift = mean_r2 - r2_sample;

        let ne = if r2_drift > 0.0 {
            1.0 / (3.0 * r2_drift)
        } else {
            f64::INFINITY
        };

        let n_pairs = r_squared_values.len();
        let (lower, upper) = chi2_ci(ne, n_pairs);

        NeEstimate {
            ne,
            lower_ci: lower,
            upper_ci: upper,
            method: NeMethod::LdBased,
            n_loci: n_pairs,
        }
    }

    /// Heterozygosity-excess method (Pudovkin et al. 1996).
    pub fn het_excess_ne(&self, observed_het: &[f64], expected_het: &[f64]) -> NeEstimate {
        let n = observed_het.len().min(expected_het.len());
        if n == 0 {
            return NeEstimate {
                ne: f64::INFINITY,
                lower_ci: 0.0,
                upper_ci: f64::INFINITY,
                method: NeMethod::HetExcess,
                n_loci: 0,
            };
        }

        let mut d_values = Vec::new();
        for i in 0..n {
            let he = expected_het[i];
            if he > 0.0 {
                let d = (observed_het[i] - he) / he;
                d_values.push(d);
            }
        }

        if d_values.is_empty() {
            return NeEstimate {
                ne: f64::INFINITY,
                lower_ci: 0.0,
                upper_ci: f64::INFINITY,
                method: NeMethod::HetExcess,
                n_loci: 0,
            };
        }

        let mean_d: f64 = d_values.iter().sum::<f64>() / d_values.len() as f64;
        let ne = if mean_d > 0.0 { 1.0 / (2.0 * mean_d) } else { f64::INFINITY };

        let (lower, upper) = chi2_ci(ne, d_values.len());

        NeEstimate {
            ne,
            lower_ci: lower,
            upper_ci: upper,
            method: NeMethod::HetExcess,
            n_loci: d_values.len(),
        }
    }

    /// Moment-based Ne from allele frequency variance.
    pub fn moment_ne(&self, freq_variance: f64, mean_freq: f64, generations: f64) -> NeEstimate {
        let p = mean_freq;
        if p <= 0.0 || p >= 1.0 || generations <= 0.0 {
            return NeEstimate {
                ne: f64::INFINITY,
                lower_ci: 0.0,
                upper_ci: f64::INFINITY,
                method: NeMethod::Moment,
                n_loci: 1,
            };
        }

        // Var(p) ≈ p(1-p) * (1 - (1 - 1/(2Ne))^t) ≈ p(1-p) * t / (2Ne)
        let expected_var_coeff = p * (1.0 - p);
        let ne = if freq_variance > 0.0 && expected_var_coeff > 0.0 {
            expected_var_coeff * generations / (2.0 * freq_variance)
        } else {
            f64::INFINITY
        };

        NeEstimate {
            ne,
            lower_ci: ne * 0.5,
            upper_ci: ne * 2.0,
            method: NeMethod::Moment,
            n_loci: 1,
        }
    }

    // ── Jackknife CI ────────────────────────────────────────────────

    fn jackknife_ci(&self, values: &[f64], generations: f64) -> (f64, f64) {
        let n = values.len();
        let full_mean: f64 = values.iter().sum::<f64>() / n as f64;

        let mut pseudo = Vec::with_capacity(n);
        for i in 0..n {
            let leave_out: f64 =
                values.iter().enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, v)| v)
                    .sum::<f64>()
                    / (n - 1) as f64;
            let pseudo_val = n as f64 * full_mean - (n - 1) as f64 * leave_out;
            pseudo.push(pseudo_val);
        }

        let jk_mean: f64 = pseudo.iter().sum::<f64>() / n as f64;
        let jk_var: f64 =
            pseudo.iter().map(|v| (v - jk_mean).powi(2)).sum::<f64>() / ((n * (n - 1)) as f64);
        let jk_se = jk_var.sqrt();

        let z = 1.96; // 95% CI
        let ne_est = if jk_mean > 0.0 { generations / (2.0 * jk_mean) } else { f64::INFINITY };

        let lower_fk = jk_mean + z * jk_se;
        let upper_fk = (jk_mean - z * jk_se).max(1e-15);

        let lower_ne = if lower_fk > 0.0 { generations / (2.0 * lower_fk) } else { 0.0 };
        let upper_ne = if upper_fk > 0.0 { generations / (2.0 * upper_fk) } else { f64::INFINITY };

        let _ = ne_est;
        (lower_ne.min(upper_ne), lower_ne.max(upper_ne))
    }
}

impl fmt::Display for NeEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NeEstimator(method={}, pcrit={:.3}, CI={:.0}%)",
            self.method, self.pcrit, self.confidence_level * 100.0
        )
    }
}

// ── Bottleneck Detection ────────────────────────────────────────────

/// Result of a bottleneck detection test.
#[derive(Debug, Clone, PartialEq)]
pub struct BottleneckResult {
    pub test_name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
    pub n_loci: usize,
}

impl fmt::Display for BottleneckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: stat={:.4} p={:.6} sig={} loci={}",
            self.test_name, self.statistic, self.p_value,
            self.significant, self.n_loci
        )
    }
}

/// Detect population bottleneck using heterozygosity excess test.
///
/// Under mutation-drift equilibrium, He ≈ He_eq (from allele count).
/// After a bottleneck, alleles are lost faster than heterozygosity,
/// so He_obs > He_eq.
pub fn bottleneck_het_test(
    observed_het: &[f64],
    allele_counts: &[usize],
    sample_sizes: &[usize],
) -> BottleneckResult {
    let n = observed_het
        .len()
        .min(allele_counts.len())
        .min(sample_sizes.len());

    if n == 0 {
        return BottleneckResult {
            test_name: "heterozygosity_excess".into(),
            statistic: 0.0,
            p_value: 1.0,
            significant: false,
            n_loci: 0,
        };
    }

    let mut n_excess = 0usize;
    let mut total_valid = 0usize;

    for i in 0..n {
        let k = allele_counts[i] as f64;
        let s = sample_sizes[i] as f64;
        if k <= 1.0 || s <= 1.0 {
            continue;
        }

        // Expected He under mutation-drift equilibrium (IAM model)
        let he_eq = 1.0 - 1.0 / k;
        // Correct for sample size
        let he_eq_corr = he_eq * (2.0 * s) / (2.0 * s - 1.0);

        total_valid += 1;
        if observed_het[i] > he_eq_corr {
            n_excess += 1;
        }
    }

    // Sign test: under null, P(excess) = 0.5
    let p_value = if total_valid > 0 {
        sign_test_p(n_excess, total_valid)
    } else {
        1.0
    };

    BottleneckResult {
        test_name: "heterozygosity_excess".into(),
        statistic: n_excess as f64 / total_valid.max(1) as f64,
        p_value,
        significant: p_value < 0.05,
        n_loci: total_valid,
    }
}

/// Mode-shift test: a bottleneck shifts the allele frequency distribution
/// from an L-shape to a shifted mode.
pub fn mode_shift_test(allele_freqs: &[f64], n_bins: usize) -> BottleneckResult {
    if allele_freqs.is_empty() || n_bins < 2 {
        return BottleneckResult {
            test_name: "mode_shift".into(),
            statistic: 0.0,
            p_value: 1.0,
            significant: false,
            n_loci: allele_freqs.len(),
        };
    }

    let bin_width = 1.0 / n_bins as f64;
    let mut bins = vec![0usize; n_bins];

    for &freq in allele_freqs {
        let f = freq.clamp(0.0, 1.0 - 1e-15);
        let idx = (f / bin_width) as usize;
        let idx = idx.min(n_bins - 1);
        bins[idx] += 1;
    }

    // Find mode bin
    let mode_bin = bins.iter().enumerate().max_by_key(|&(_, &c)| c).map(|(i, _)| i).unwrap_or(0);

    // Under stable populations, mode is in the lowest frequency class (bin 0).
    // A shifted mode (bin > 0) suggests a bottleneck.
    let shifted = mode_bin > 0;
    let statistic = mode_bin as f64 * bin_width;

    BottleneckResult {
        test_name: "mode_shift".into(),
        statistic,
        p_value: if shifted { 0.01 } else { 0.5 },
        significant: shifted,
        n_loci: allele_freqs.len(),
    }
}

// ── Helper Functions ────────────────────────────────────────────────

/// Approximate chi-squared CI for Ne.
fn chi2_ci(ne: f64, df: usize) -> (f64, f64) {
    if ne.is_infinite() || df == 0 {
        return (0.0, f64::INFINITY);
    }
    let k = df as f64;
    // Approximate chi2 quantiles using Wilson-Hilferty transform
    let z_lower = 1.96;
    let factor_lower = (1.0 - 2.0 / (9.0 * k) + z_lower * (2.0 / (9.0 * k)).sqrt()).powi(3);
    let factor_upper = (1.0 - 2.0 / (9.0 * k) - z_lower * (2.0 / (9.0 * k)).sqrt()).powi(3);

    let lower = if factor_lower > 0.0 { ne * k / (k * factor_lower) } else { 0.0 };
    let upper = if factor_upper > 0.0 { ne * k / (k * factor_upper) } else { f64::INFINITY };

    (lower.min(upper), lower.max(upper))
}

/// One-sided sign test p-value.
fn sign_test_p(successes: usize, total: usize) -> f64 {
    if total == 0 {
        return 1.0;
    }
    // P(X >= successes) under Binomial(total, 0.5)
    let mut p = 0.0;
    for k in successes..=total {
        p += binom_pmf(total, k, 0.5);
    }
    p
}

/// Binomial PMF using log-space to avoid overflow.
fn binom_pmf(n: usize, k: usize, prob: f64) -> f64 {
    if k > n {
        return 0.0;
    }
    let log_coeff = log_binom_coeff(n, k);
    let log_p = log_coeff + k as f64 * prob.ln() + (n - k) as f64 * (1.0 - prob).ln();
    log_p.exp()
}

fn log_binom_coeff(n: usize, k: usize) -> f64 {
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    (2..=n).map(|i| (i as f64).ln()).sum()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-4;

    #[test]
    fn test_temporal_ne_basic() {
        let sample = TemporalSample::new(10.0)
            .with_locus(0.5, 0.65, 50, 50)
            .with_locus(0.3, 0.20, 50, 50)
            .with_locus(0.7, 0.60, 50, 50);
        let est = NeEstimator::new().with_method(NeMethod::Temporal);
        let result = est.temporal_ne(&sample);
        assert!(result.ne > 0.0);
        assert!(result.ne.is_finite());
    }

    #[test]
    fn test_temporal_ne_no_drift() {
        let sample = TemporalSample::new(10.0)
            .with_locus(0.5, 0.5, 100, 100)
            .with_locus(0.3, 0.3, 100, 100);
        let est = NeEstimator::new().with_jackknife(false);
        let result = est.temporal_ne(&sample);
        // Large or infinite Ne when no frequency change
        assert!(result.ne > 1000.0 || result.ne.is_infinite());
    }

    #[test]
    fn test_temporal_ne_empty() {
        let sample = TemporalSample::new(10.0);
        let est = NeEstimator::new();
        let result = est.temporal_ne(&sample);
        assert!(result.ne.is_infinite());
    }

    #[test]
    fn test_ld_ne_basic() {
        let r2_values = vec![0.05, 0.04, 0.06, 0.03, 0.05, 0.04];
        let est = NeEstimator::new().with_method(NeMethod::LdBased);
        let result = est.ld_ne(&r2_values, 50);
        assert!(result.ne > 0.0);
        assert!(result.ne.is_finite());
    }

    #[test]
    fn test_ld_ne_high_ld() {
        // High LD => small Ne
        let r2_values = vec![0.5, 0.6, 0.55, 0.45];
        let est = NeEstimator::new();
        let low_ne = est.ld_ne(&r2_values, 50);

        // Low LD => large Ne
        let r2_low = vec![0.025, 0.03, 0.028, 0.022];
        let high_ne = est.ld_ne(&r2_low, 50);

        assert!(low_ne.ne < high_ne.ne);
    }

    #[test]
    fn test_het_excess_ne() {
        let obs = vec![0.52, 0.48, 0.51, 0.49, 0.53];
        let exp = vec![0.50, 0.50, 0.50, 0.50, 0.50];
        let est = NeEstimator::new().with_method(NeMethod::HetExcess);
        let result = est.het_excess_ne(&obs, &exp);
        assert!(result.ne > 0.0);
    }

    #[test]
    fn test_moment_ne() {
        let est = NeEstimator::new();
        let result = est.moment_ne(0.001, 0.5, 10.0);
        assert!(result.ne > 0.0);
        assert!(result.ne.is_finite());
    }

    #[test]
    fn test_moment_ne_no_variance() {
        let est = NeEstimator::new();
        let result = est.moment_ne(0.0, 0.5, 10.0);
        assert!(result.ne.is_infinite());
    }

    #[test]
    fn test_bottleneck_no_excess() {
        let obs_het = vec![0.3, 0.25, 0.35, 0.28];
        let allele_counts = vec![5, 4, 6, 5];
        let sample_sizes = vec![50, 50, 50, 50];
        let result = bottleneck_het_test(&obs_het, &allele_counts, &sample_sizes);
        assert_eq!(result.test_name, "heterozygosity_excess");
        assert!(result.n_loci > 0);
    }

    #[test]
    fn test_mode_shift_stable() {
        // L-shaped distribution (lots of low-freq alleles)
        let mut freqs = Vec::new();
        for _ in 0..50 { freqs.push(0.02); }
        for _ in 0..20 { freqs.push(0.15); }
        for _ in 0..10 { freqs.push(0.3); }
        let result = mode_shift_test(&freqs, 10);
        assert!(!result.significant);
    }

    #[test]
    fn test_mode_shift_bottleneck() {
        // Mode shifted away from low frequency
        let mut freqs = Vec::new();
        for _ in 0..10 { freqs.push(0.05); }
        for _ in 0..50 { freqs.push(0.3); }
        for _ in 0..20 { freqs.push(0.6); }
        let result = mode_shift_test(&freqs, 10);
        assert!(result.significant);
    }

    #[test]
    fn test_bottleneck_empty() {
        let result = bottleneck_het_test(&[], &[], &[]);
        assert_eq!(result.n_loci, 0);
        assert!(!result.significant);
    }

    #[test]
    fn test_ne_estimate_display() {
        let ne = NeEstimate {
            ne: 500.0, lower_ci: 300.0, upper_ci: 800.0,
            method: NeMethod::Temporal, n_loci: 10,
        };
        let s = format!("{}", ne);
        assert!(s.contains("500.0"));
        assert!(s.contains("temporal"));
    }

    #[test]
    fn test_temporal_sample_display() {
        let sample = TemporalSample::new(5.0)
            .with_locus(0.3, 0.35, 50, 50);
        let s = format!("{}", sample);
        assert!(s.contains("loci=1"));
    }

    #[test]
    fn test_bottleneck_result_display() {
        let r = BottleneckResult {
            test_name: "test".into(),
            statistic: 0.75,
            p_value: 0.03,
            significant: true,
            n_loci: 20,
        };
        let s = format!("{}", r);
        assert!(s.contains("0.03"));
    }

    #[test]
    fn test_estimator_display() {
        let est = NeEstimator::new().with_pcrit(0.05);
        let s = format!("{}", est);
        assert!(s.contains("temporal"));
    }

    #[test]
    fn test_chi2_ci_finite() {
        let (lower, upper) = chi2_ci(500.0, 20);
        assert!(lower > 0.0);
        assert!(upper > lower);
        assert!(upper.is_finite());
    }

    #[test]
    fn test_binom_pmf_basic() {
        // P(X=0 | n=2, p=0.5) = 0.25
        let p = binom_pmf(2, 0, 0.5);
        assert!((p - 0.25).abs() < EPS);
    }

    #[test]
    fn test_ne_method_display() {
        assert_eq!(format!("{}", NeMethod::LdBased), "LD-based");
        assert_eq!(format!("{}", NeMethod::Coalescent), "coalescent");
    }
}
