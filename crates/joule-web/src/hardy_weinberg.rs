//! Hardy-Weinberg equilibrium testing for population genetics.
//!
//! Implements chi-squared goodness-of-fit, exact test (via mid-p),
//! excess heterozygosity detection, and multi-locus HWE screening.
//! All arithmetic is std-only f64.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────

/// Chi-squared critical values for 1 df at common alpha levels.
const CHI2_CRITICAL_005: f64 = 3.841;
const CHI2_CRITICAL_001: f64 = 6.635;
const CHI2_CRITICAL_0001: f64 = 10.828;

// ── Genotype Counts ─────────────────────────────────────────────────

/// Observed genotype counts at a biallelic locus (diploid).
#[derive(Debug, Clone, PartialEq)]
pub struct GenotypeCounts {
    pub n_aa: u64,
    pub n_ab: u64,
    pub n_bb: u64,
}

impl GenotypeCounts {
    /// Create from raw counts.
    pub fn new(n_aa: u64, n_ab: u64, n_bb: u64) -> Self {
        Self { n_aa, n_ab, n_bb }
    }

    /// Total number of individuals.
    pub fn total(&self) -> u64 {
        self.n_aa + self.n_ab + self.n_bb
    }

    /// Frequency of the A allele.
    pub fn freq_a(&self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 { return 0.0; }
        (2.0 * self.n_aa as f64 + self.n_ab as f64) / (2.0 * n)
    }

    /// Frequency of the B allele.
    pub fn freq_b(&self) -> f64 {
        1.0 - self.freq_a()
    }

    /// Observed heterozygosity.
    pub fn observed_het(&self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 { return 0.0; }
        self.n_ab as f64 / n
    }

    /// Expected heterozygosity under HWE: 2pq.
    pub fn expected_het(&self) -> f64 {
        let p = self.freq_a();
        2.0 * p * (1.0 - p)
    }

    /// Expected genotype counts under HWE.
    pub fn expected_counts(&self) -> (f64, f64, f64) {
        let n = self.total() as f64;
        let p = self.freq_a();
        let q = 1.0 - p;
        (n * p * p, n * 2.0 * p * q, n * q * q)
    }
}

impl fmt::Display for GenotypeCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AA={} AB={} BB={} (n={}, p={:.4})",
            self.n_aa,
            self.n_ab,
            self.n_bb,
            self.total(),
            self.freq_a()
        )
    }
}

// ── HWE Test Result ─────────────────────────────────────────────────

/// Outcome of a Hardy-Weinberg equilibrium test.
#[derive(Debug, Clone, PartialEq)]
pub struct HweTestResult {
    pub chi_squared: f64,
    pub p_value: f64,
    pub exact_p: f64,
    pub observed_het: f64,
    pub expected_het: f64,
    pub inbreeding_coeff: f64,
    pub deviation: HweDeviation,
}

/// Direction of deviation from HWE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HweDeviation {
    None,
    ExcessHet,
    DeficitHet,
}

impl fmt::Display for HweDeviation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::ExcessHet => write!(f, "excess_heterozygosity"),
            Self::DeficitHet => write!(f, "deficit_heterozygosity"),
        }
    }
}

impl HweTestResult {
    /// Is the chi-squared significant at the given alpha?
    pub fn is_significant(&self, alpha: f64) -> bool {
        let crit = if alpha <= 0.001 {
            CHI2_CRITICAL_0001
        } else if alpha <= 0.01 {
            CHI2_CRITICAL_001
        } else {
            CHI2_CRITICAL_005
        };
        self.chi_squared > crit
    }
}

impl fmt::Display for HweTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HWE chi2={:.4} p={:.6} exact_p={:.6} F={:.4} deviation={}",
            self.chi_squared, self.p_value, self.exact_p,
            self.inbreeding_coeff, self.deviation
        )
    }
}

// ── HWE Tester ──────────────────────────────────────────────────────

/// Configurable Hardy-Weinberg tester.
#[derive(Debug, Clone)]
pub struct HweTester {
    alpha: f64,
    use_exact: bool,
    min_sample_size: usize,
}

impl Default for HweTester {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            use_exact: true,
            min_sample_size: 10,
        }
    }
}

impl HweTester {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(1e-10, 0.5);
        self
    }

    pub fn with_exact(mut self, use_exact: bool) -> Self {
        self.use_exact = use_exact;
        self
    }

    pub fn with_min_sample_size(mut self, n: usize) -> Self {
        self.min_sample_size = n;
        self
    }

    /// Run HWE test on genotype counts.
    pub fn test(&self, counts: &GenotypeCounts) -> HweTestResult {
        let n = counts.total() as f64;
        let (exp_aa, exp_ab, exp_bb) = counts.expected_counts();
        let obs_het = counts.observed_het();
        let exp_het = counts.expected_het();

        // Chi-squared with 1 df
        let chi2 = if exp_aa > 0.0 && exp_ab > 0.0 && exp_bb > 0.0 {
            let d_aa = counts.n_aa as f64 - exp_aa;
            let d_ab = counts.n_ab as f64 - exp_ab;
            let d_bb = counts.n_bb as f64 - exp_bb;
            d_aa * d_aa / exp_aa + d_ab * d_ab / exp_ab + d_bb * d_bb / exp_bb
        } else {
            0.0
        };

        // Approximate p-value from chi2 with 1 df
        let p_chi2 = chi2_pvalue_1df(chi2);

        // Exact test (mid-p)
        let exact_p = if self.use_exact && n > 0.0 {
            exact_hwe_midp(counts)
        } else {
            p_chi2
        };

        // Inbreeding coefficient F = 1 - Ho/He
        let f_coeff = if exp_het > 0.0 {
            1.0 - obs_het / exp_het
        } else {
            0.0
        };

        let deviation = if chi2 > CHI2_CRITICAL_005 {
            if obs_het > exp_het {
                HweDeviation::ExcessHet
            } else {
                HweDeviation::DeficitHet
            }
        } else {
            HweDeviation::None
        };

        HweTestResult {
            chi_squared: chi2,
            p_value: p_chi2,
            exact_p,
            observed_het: obs_het,
            expected_het: exp_het,
            inbreeding_coeff: f_coeff,
            deviation,
        }
    }

    /// Test multiple loci, returning those that fail HWE at the configured alpha.
    pub fn screen_loci(&self, loci: &[GenotypeCounts]) -> Vec<(usize, HweTestResult)> {
        loci.iter()
            .enumerate()
            .filter_map(|(i, gc)| {
                if (gc.total() as usize) < self.min_sample_size {
                    return None;
                }
                let result = self.test(gc);
                if result.is_significant(self.alpha) {
                    Some((i, result))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Bonferroni-corrected multi-locus HWE test.
    pub fn bonferroni_screen(&self, loci: &[GenotypeCounts]) -> Vec<(usize, HweTestResult)> {
        let n_tests = loci.len() as f64;
        let corrected_alpha = self.alpha / n_tests;
        loci.iter()
            .enumerate()
            .filter_map(|(i, gc)| {
                if (gc.total() as usize) < self.min_sample_size {
                    return None;
                }
                let result = self.test(gc);
                let crit = if corrected_alpha <= 0.001 {
                    CHI2_CRITICAL_0001
                } else if corrected_alpha <= 0.01 {
                    CHI2_CRITICAL_001
                } else {
                    CHI2_CRITICAL_005
                };
                if result.chi_squared > crit {
                    Some((i, result))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl fmt::Display for HweTester {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HweTester(alpha={}, exact={}, min_n={})",
            self.alpha, self.use_exact, self.min_sample_size
        )
    }
}

// ── Inbreeding Coefficient ──────────────────────────────────────────

/// Calculate Wright's inbreeding coefficient F from genotype counts.
pub fn wrights_f(counts: &GenotypeCounts) -> f64 {
    let he = counts.expected_het();
    if he == 0.0 {
        return 0.0;
    }
    1.0 - counts.observed_het() / he
}

/// Estimate Fis (within-population inbreeding) for multiple loci.
pub fn fis_multilocus(loci: &[GenotypeCounts]) -> f64 {
    let mut sum_ho = 0.0;
    let mut sum_he = 0.0;
    for gc in loci {
        sum_ho += gc.observed_het();
        sum_he += gc.expected_het();
    }
    if sum_he == 0.0 { 0.0 } else { 1.0 - sum_ho / sum_he }
}

// ── Helper Functions ────────────────────────────────────────────────

/// Approximate chi-squared p-value for 1 df using rational approximation.
fn chi2_pvalue_1df(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // Upper tail of chi2(1) = 2*(1 - Phi(sqrt(x)))
    let z = x.sqrt();
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
            + t * (1.781_477_937
                + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi_tail = poly * (-0.5 * z * z).exp() * 0.398_942_280_401;
    (2.0 * phi_tail).clamp(0.0, 1.0)
}

/// Exact HWE mid-p test using the enumeration of heterozygote classes.
fn exact_hwe_midp(counts: &GenotypeCounts) -> f64 {
    let n = counts.total() as usize;
    let n_b = (2 * counts.n_bb as usize) + counts.n_ab as usize;
    let obs_het = counts.n_ab as usize;

    // Enumerate all possible het counts with same allele count
    let max_het = n_b.min(2 * n - n_b);
    let max_het = if max_het % 2 != obs_het % 2 {
        if max_het > 0 { max_het - 1 } else { 0 }
    } else {
        max_het
    };

    // Compute log-probabilities using the recursive formula
    let mut log_probs = Vec::new();
    let mut het_values = Vec::new();

    let mut het = obs_het % 2;
    if het == 0 && obs_het % 2 != 0 {
        het = 1;
    }

    while het <= max_het {
        let hom_b = (n_b - het) / 2;
        let hom_a = n - het - hom_b;
        let lp = log_multinomial_coeff(n, hom_a, het, hom_b, n_b);
        log_probs.push(lp);
        het_values.push(het);
        het += 2;
    }

    if log_probs.is_empty() {
        return 1.0;
    }

    // Normalize in log space
    let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_probs.iter().map(|lp| (lp - max_lp).exp()).sum();
    let log_norm = max_lp + sum_exp.ln();

    // Mid-p: P(het < obs) + 0.5 * P(het == obs)
    let mut p = 0.0;
    for (i, &hv) in het_values.iter().enumerate() {
        let prob = (log_probs[i] - log_norm).exp();
        if hv == obs_het {
            p += 0.5 * prob;
        } else if hv < obs_het {
            p += prob;
        }
    }

    // Two-sided: also count the tail above
    let obs_prob = {
        let idx = het_values.iter().position(|v| *v == obs_het);
        idx.map(|i| (log_probs[i] - log_norm).exp()).unwrap_or(0.0)
    };

    let mut upper_tail = 0.0;
    for (i, &_hv) in het_values.iter().enumerate() {
        let prob = (log_probs[i] - log_norm).exp();
        if prob <= obs_prob + 1e-12 {
            upper_tail += prob;
        }
    }

    upper_tail.min(1.0)
}

/// Log of the HWE probability kernel for genotype configuration.
fn log_multinomial_coeff(
    n: usize,
    n_aa: usize,
    n_ab: usize,
    n_bb: usize,
    n_b_alleles: usize,
) -> f64 {
    let _ = n_b_alleles;
    log_factorial(n)
        - log_factorial(n_aa)
        - log_factorial(n_ab)
        - log_factorial(n_bb)
        + n_ab as f64 * 2.0_f64.ln()
}

/// Log factorial using Stirling's approximation for large values.
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n <= 20 {
        let mut result = 0.0;
        for i in 2..=n {
            result += (i as f64).ln();
        }
        return result;
    }
    // Stirling's approximation
    let nf = n as f64;
    0.5 * (2.0 * std::f64::consts::PI * nf).ln() + nf * (nf / std::f64::consts::E).ln()
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_genotype_counts_freq() {
        let gc = GenotypeCounts::new(25, 50, 25);
        assert!((gc.freq_a() - 0.5).abs() < EPS);
        assert!((gc.freq_b() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_expected_het() {
        let gc = GenotypeCounts::new(25, 50, 25);
        assert!((gc.expected_het() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_hwe_in_equilibrium() {
        // Perfect HWE: p=0.5, n=100 -> AA=25, AB=50, BB=25
        let gc = GenotypeCounts::new(25, 50, 25);
        let tester = HweTester::new();
        let result = tester.test(&gc);
        assert!(result.chi_squared < CHI2_CRITICAL_005);
        assert_eq!(result.deviation, HweDeviation::None);
    }

    #[test]
    fn test_hwe_excess_het() {
        // Too many heterozygotes
        let gc = GenotypeCounts::new(5, 90, 5);
        let tester = HweTester::new();
        let result = tester.test(&gc);
        assert!(result.chi_squared > CHI2_CRITICAL_005);
        assert_eq!(result.deviation, HweDeviation::ExcessHet);
    }

    #[test]
    fn test_hwe_deficit_het() {
        // Too few heterozygotes (inbreeding signal)
        let gc = GenotypeCounts::new(45, 10, 45);
        let tester = HweTester::new();
        let result = tester.test(&gc);
        assert!(result.chi_squared > CHI2_CRITICAL_005);
        assert_eq!(result.deviation, HweDeviation::DeficitHet);
    }

    #[test]
    fn test_wrights_f_equilibrium() {
        let gc = GenotypeCounts::new(25, 50, 25);
        let f = wrights_f(&gc);
        assert!(f.abs() < EPS);
    }

    #[test]
    fn test_wrights_f_inbreeding() {
        let gc = GenotypeCounts::new(45, 10, 45);
        let f = wrights_f(&gc);
        assert!(f > 0.0); // positive F = het deficit
    }

    #[test]
    fn test_wrights_f_outbreeding() {
        let gc = GenotypeCounts::new(10, 80, 10);
        let f = wrights_f(&gc);
        assert!(f < 0.0); // negative F = het excess
    }

    #[test]
    fn test_genotype_counts_display() {
        let gc = GenotypeCounts::new(30, 40, 30);
        let s = format!("{}", gc);
        assert!(s.contains("AA=30"));
        assert!(s.contains("n=100"));
    }

    #[test]
    fn test_hwe_result_display() {
        let gc = GenotypeCounts::new(25, 50, 25);
        let result = HweTester::new().test(&gc);
        let s = format!("{}", result);
        assert!(s.contains("HWE"));
        assert!(s.contains("deviation=none"));
    }

    #[test]
    fn test_screen_loci() {
        let loci = vec![
            GenotypeCounts::new(25, 50, 25), // in HWE
            GenotypeCounts::new(45, 10, 45), // out of HWE
            GenotypeCounts::new(24, 52, 24), // near HWE
        ];
        let tester = HweTester::new();
        let failures = tester.screen_loci(&loci);
        assert!(failures.iter().any(|(i, _)| *i == 1));
    }

    #[test]
    fn test_min_sample_filter() {
        let loci = vec![
            GenotypeCounts::new(2, 3, 2), // n=7 < 10
        ];
        let tester = HweTester::new().with_min_sample_size(10);
        let failures = tester.screen_loci(&loci);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_chi2_pvalue_large() {
        let p = chi2_pvalue_1df(20.0);
        assert!(p < 0.001);
    }

    #[test]
    fn test_chi2_pvalue_zero() {
        let p = chi2_pvalue_1df(0.0);
        assert!((p - 1.0).abs() < EPS);
    }

    #[test]
    fn test_fis_multilocus() {
        let loci = vec![
            GenotypeCounts::new(25, 50, 25),
            GenotypeCounts::new(25, 50, 25),
        ];
        let fis = fis_multilocus(&loci);
        assert!(fis.abs() < EPS);
    }

    #[test]
    fn test_monomorphic_locus() {
        let gc = GenotypeCounts::new(100, 0, 0);
        let result = HweTester::new().test(&gc);
        assert!((result.chi_squared - 0.0).abs() < EPS);
        assert!((result.inbreeding_coeff - 0.0).abs() < EPS);
    }

    #[test]
    fn test_expected_counts() {
        let gc = GenotypeCounts::new(25, 50, 25);
        let (ea, eab, eb) = gc.expected_counts();
        assert!((ea - 25.0).abs() < EPS);
        assert!((eab - 50.0).abs() < EPS);
        assert!((eb - 25.0).abs() < EPS);
    }

    #[test]
    fn test_deviation_display() {
        assert_eq!(format!("{}", HweDeviation::ExcessHet), "excess_heterozygosity");
        assert_eq!(format!("{}", HweDeviation::None), "none");
    }

    #[test]
    fn test_log_factorial_small() {
        assert!((log_factorial(0) - 0.0).abs() < EPS);
        assert!((log_factorial(1) - 0.0).abs() < EPS);
        assert!((log_factorial(5) - (120.0_f64).ln()).abs() < EPS);
    }

    #[test]
    fn test_tester_display() {
        let t = HweTester::new().with_alpha(0.01);
        let s = format!("{}", t);
        assert!(s.contains("alpha=0.01"));
    }
}
