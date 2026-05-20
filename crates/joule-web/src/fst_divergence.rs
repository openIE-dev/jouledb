//! Fst fixation index and population differentiation statistics.
//!
//! Implements Wright's Fst, Weir-Cockerham estimator, Nei's Gst,
//! pairwise population Fst matrices, and hierarchical F-statistics.
//! All computations are std-only with f64 math.

use std::fmt;

// ── Population Allele Data ──────────────────────────────────────────

/// Allele frequency data for one population at one locus.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationFreq {
    pub pop_id: usize,
    pub freq_a: f64,
    pub sample_size: usize,
    pub het_obs: f64,
}

impl PopulationFreq {
    pub fn new(pop_id: usize, freq_a: f64, sample_size: usize) -> Self {
        let het_obs = 2.0 * freq_a * (1.0 - freq_a);
        Self { pop_id, freq_a, sample_size, het_obs }
    }

    pub fn with_het_obs(mut self, het_obs: f64) -> Self {
        self.het_obs = het_obs;
        self
    }

    /// Expected heterozygosity under HWE.
    pub fn expected_het(&self) -> f64 {
        2.0 * self.freq_a * (1.0 - self.freq_a)
    }
}

impl fmt::Display for PopulationFreq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pop {} p={:.4} n={} H_obs={:.4}",
            self.pop_id, self.freq_a, self.sample_size, self.het_obs
        )
    }
}

// ── Fst Result ──────────────────────────────────────────────────────

/// Result of an Fst estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct FstResult {
    pub fst: f64,
    pub fis: f64,
    pub fit: f64,
    pub method: FstMethod,
    pub n_pops: usize,
    pub n_loci: usize,
}

/// Method used for Fst estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FstMethod {
    Wright,
    WeirCockerham,
    Nei,
    Hudson,
}

impl fmt::Display for FstMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wright => write!(f, "Wright"),
            Self::WeirCockerham => write!(f, "Weir-Cockerham"),
            Self::Nei => write!(f, "Nei"),
            Self::Hudson => write!(f, "Hudson"),
        }
    }
}

impl fmt::Display for FstResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Fst={:.6} Fis={:.6} Fit={:.6} method={} pops={} loci={}",
            self.fst, self.fis, self.fit, self.method, self.n_pops, self.n_loci
        )
    }
}

// ── Fst Estimator ───────────────────────────────────────────────────

/// Configurable Fst estimator.
#[derive(Debug, Clone)]
pub struct FstEstimator {
    method: FstMethod,
    min_sample_per_pop: usize,
    weighted: bool,
}

impl Default for FstEstimator {
    fn default() -> Self {
        Self {
            method: FstMethod::WeirCockerham,
            min_sample_per_pop: 2,
            weighted: true,
        }
    }
}

impl FstEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_method(mut self, method: FstMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_min_sample(mut self, n: usize) -> Self {
        self.min_sample_per_pop = n;
        self
    }

    pub fn with_weighted(mut self, w: bool) -> Self {
        self.weighted = w;
        self
    }

    /// Estimate Fst from population frequencies at a single locus.
    pub fn estimate_single_locus(&self, pops: &[PopulationFreq]) -> FstResult {
        let valid: Vec<&PopulationFreq> = pops
            .iter()
            .filter(|p| p.sample_size >= self.min_sample_per_pop)
            .collect();

        if valid.len() < 2 {
            return FstResult {
                fst: 0.0, fis: 0.0, fit: 0.0,
                method: self.method, n_pops: valid.len(), n_loci: 1,
            };
        }

        match self.method {
            FstMethod::Wright => self.wright_fst(&valid),
            FstMethod::WeirCockerham => self.weir_cockerham(&valid),
            FstMethod::Nei => self.nei_gst(&valid),
            FstMethod::Hudson => self.hudson_fst(&valid),
        }
    }

    /// Multi-locus Fst (ratio of averages for Weir-Cockerham).
    pub fn estimate_multi_locus(&self, loci: &[Vec<PopulationFreq>]) -> FstResult {
        if loci.is_empty() {
            return FstResult {
                fst: 0.0, fis: 0.0, fit: 0.0,
                method: self.method, n_pops: 0, n_loci: 0,
            };
        }

        let results: Vec<FstResult> = loci
            .iter()
            .map(|pops| self.estimate_single_locus(pops))
            .collect();

        let n = results.len() as f64;
        let avg_fst = results.iter().map(|r| r.fst).sum::<f64>() / n;
        let avg_fis = results.iter().map(|r| r.fis).sum::<f64>() / n;
        let avg_fit = results.iter().map(|r| r.fit).sum::<f64>() / n;
        let max_pops = results.iter().map(|r| r.n_pops).max().unwrap_or(0);

        FstResult {
            fst: avg_fst,
            fis: avg_fis,
            fit: avg_fit,
            method: self.method,
            n_pops: max_pops,
            n_loci: loci.len(),
        }
    }

    /// Pairwise Fst matrix for multiple populations at one locus.
    pub fn pairwise_fst(&self, pops: &[PopulationFreq]) -> Vec<Vec<f64>> {
        let n = pops.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let pair = vec![pops[i].clone(), pops[j].clone()];
                let result = self.estimate_single_locus(&pair);
                matrix[i][j] = result.fst;
                matrix[j][i] = result.fst;
            }
        }
        matrix
    }

    // ── Method Implementations ──────────────────────────────────────

    fn wright_fst(&self, pops: &[&PopulationFreq]) -> FstResult {
        let (p_bar, ht, hs) = self.compute_h_stats(pops);
        let _ = p_bar;
        let fst = if ht > 0.0 { (ht - hs) / ht } else { 0.0 };

        let ho = self.mean_het_obs(pops);
        let fis = if hs > 0.0 { 1.0 - ho / hs } else { 0.0 };
        let fit = if ht > 0.0 { 1.0 - ho / ht } else { 0.0 };

        FstResult {
            fst: fst.clamp(-1.0, 1.0),
            fis,
            fit,
            method: FstMethod::Wright,
            n_pops: pops.len(),
            n_loci: 1,
        }
    }

    fn weir_cockerham(&self, pops: &[&PopulationFreq]) -> FstResult {
        let r = pops.len() as f64;
        let total_n: f64 = pops.iter().map(|p| p.sample_size as f64).sum();
        let n_bar = total_n / r;

        // n_c (sample size correction factor)
        let sum_ni_sq: f64 = pops.iter().map(|p| {
            let ni = p.sample_size as f64;
            ni * ni
        }).sum();
        let n_c = (total_n - sum_ni_sq / total_n) / (r - 1.0);

        // Weighted mean allele frequency
        let p_bar: f64 = pops.iter().map(|p| p.sample_size as f64 * p.freq_a).sum::<f64>() / total_n;

        // S^2 (sample variance)
        let s_sq: f64 = pops.iter().map(|p| {
            let ni = p.sample_size as f64;
            ni * (p.freq_a - p_bar).powi(2)
        }).sum::<f64>() / ((r - 1.0) * n_bar);

        // Mean observed het
        let h_bar: f64 = pops.iter().map(|p| p.het_obs * p.sample_size as f64).sum::<f64>() / total_n;

        // Variance components a, b, c
        let p_bar_q_bar = p_bar * (1.0 - p_bar);

        let a = (n_bar / n_c)
            * (s_sq - (1.0 / (n_bar - 1.0)) * (p_bar_q_bar - (r - 1.0) / r * s_sq - 0.25 * h_bar));

        let b = (n_bar / (n_bar - 1.0))
            * (p_bar_q_bar - (r - 1.0) / r * s_sq - (2.0 * n_bar - 1.0) / (4.0 * n_bar) * h_bar);

        let c = 0.5 * h_bar;

        let total = a + b + c;
        let fst = if total > 0.0 { a / total } else { 0.0 };
        let fis = if (a + b) > 0.0 { 1.0 - c / (a.abs() + b) } else { 0.0 };
        let fit = if total > 0.0 { 1.0 - c / total } else { 0.0 };

        FstResult {
            fst: fst.clamp(-1.0, 1.0),
            fis: fis.clamp(-1.0, 1.0),
            fit: fit.clamp(-1.0, 1.0),
            method: FstMethod::WeirCockerham,
            n_pops: pops.len(),
            n_loci: 1,
        }
    }

    fn nei_gst(&self, pops: &[&PopulationFreq]) -> FstResult {
        let (_, ht, hs) = self.compute_h_stats(pops);
        let n = pops.len() as f64;

        // Nei's Gst with sample size correction
        let ht_corr = ht + hs / (2.0 * n);
        let gst = if ht_corr > 0.0 { (ht_corr - hs) / ht_corr } else { 0.0 };

        let ho = self.mean_het_obs(pops);
        let fis = if hs > 0.0 { 1.0 - ho / hs } else { 0.0 };
        let fit = if ht > 0.0 { 1.0 - ho / ht } else { 0.0 };

        FstResult {
            fst: gst.clamp(-1.0, 1.0),
            fis,
            fit,
            method: FstMethod::Nei,
            n_pops: pops.len(),
            n_loci: 1,
        }
    }

    fn hudson_fst(&self, pops: &[&PopulationFreq]) -> FstResult {
        // Hudson's Fst for two populations
        if pops.len() != 2 {
            return self.wright_fst(pops);
        }
        let p1 = pops[0].freq_a;
        let p2 = pops[1].freq_a;
        let n1 = pops[0].sample_size as f64;
        let n2 = pops[1].sample_size as f64;

        let numerator = (p1 - p2).powi(2) - p1 * (1.0 - p1) / (n1 - 1.0) - p2 * (1.0 - p2) / (n2 - 1.0);
        let denominator = p1 * (1.0 - p2) + p2 * (1.0 - p1);

        let fst = if denominator > 0.0 { numerator / denominator } else { 0.0 };

        FstResult {
            fst: fst.clamp(-1.0, 1.0),
            fis: 0.0,
            fit: 0.0,
            method: FstMethod::Hudson,
            n_pops: 2,
            n_loci: 1,
        }
    }

    // ── Shared helpers ──────────────────────────────────────────────

    fn compute_h_stats(&self, pops: &[&PopulationFreq]) -> (f64, f64, f64) {
        let total_n: f64 = pops.iter().map(|p| p.sample_size as f64).sum();

        let p_bar = if self.weighted {
            pops.iter().map(|p| p.sample_size as f64 * p.freq_a).sum::<f64>() / total_n
        } else {
            pops.iter().map(|p| p.freq_a).sum::<f64>() / pops.len() as f64
        };

        let ht = 2.0 * p_bar * (1.0 - p_bar);

        let hs = if self.weighted {
            pops.iter()
                .map(|p| p.sample_size as f64 * p.expected_het())
                .sum::<f64>()
                / total_n
        } else {
            pops.iter().map(|p| p.expected_het()).sum::<f64>() / pops.len() as f64
        };

        (p_bar, ht, hs)
    }

    fn mean_het_obs(&self, pops: &[&PopulationFreq]) -> f64 {
        if self.weighted {
            let total_n: f64 = pops.iter().map(|p| p.sample_size as f64).sum();
            pops.iter()
                .map(|p| p.sample_size as f64 * p.het_obs)
                .sum::<f64>()
                / total_n
        } else {
            pops.iter().map(|p| p.het_obs).sum::<f64>() / pops.len() as f64
        }
    }
}

impl fmt::Display for FstEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FstEstimator(method={}, min_n={}, weighted={})",
            self.method, self.min_sample_per_pop, self.weighted
        )
    }
}

// ── Convenience Functions ───────────────────────────────────────────

/// Quick Wright's Fst from two allele frequencies.
pub fn fst_two_pops(p1: f64, p2: f64) -> f64 {
    let p_bar = (p1 + p2) / 2.0;
    let ht = 2.0 * p_bar * (1.0 - p_bar);
    let hs = (2.0 * p1 * (1.0 - p1) + 2.0 * p2 * (1.0 - p2)) / 2.0;
    if ht > 0.0 { (ht - hs) / ht } else { 0.0 }
}

/// Nm (number of migrants) from Fst: Nm ≈ (1 - Fst) / (4 * Fst).
pub fn nm_from_fst(fst: f64) -> f64 {
    if fst <= 0.0 || fst >= 1.0 { return 0.0; }
    (1.0 - fst) / (4.0 * fst)
}

/// Jost's D differentiation measure.
pub fn jost_d(ht: f64, hs: f64, n_pops: usize) -> f64 {
    let k = n_pops as f64;
    if hs >= 1.0 { return 0.0; }
    let d = ((ht - hs) / (1.0 - hs)) * (k / (k - 1.0));
    d.clamp(0.0, 1.0)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_fst_identical_pops() {
        let fst = fst_two_pops(0.3, 0.3);
        assert!(fst.abs() < EPS);
    }

    #[test]
    fn test_fst_fixed_diff() {
        let fst = fst_two_pops(1.0, 0.0);
        assert!((fst - 1.0).abs() < EPS);
    }

    #[test]
    fn test_fst_moderate() {
        let fst = fst_two_pops(0.2, 0.8);
        assert!(fst > 0.3);
        assert!(fst < 1.0);
    }

    #[test]
    fn test_nm_from_fst() {
        let nm = nm_from_fst(0.2);
        assert!((nm - 1.0).abs() < EPS);
    }

    #[test]
    fn test_wright_estimator() {
        let pops = vec![
            PopulationFreq::new(0, 0.8, 50),
            PopulationFreq::new(1, 0.2, 50),
        ];
        let est = FstEstimator::new().with_method(FstMethod::Wright);
        let result = est.estimate_single_locus(&pops);
        assert!(result.fst > 0.3);
    }

    #[test]
    fn test_weir_cockerham() {
        let pops = vec![
            PopulationFreq::new(0, 0.8, 50).with_het_obs(0.32),
            PopulationFreq::new(1, 0.2, 50).with_het_obs(0.32),
        ];
        let est = FstEstimator::new().with_method(FstMethod::WeirCockerham);
        let result = est.estimate_single_locus(&pops);
        assert!(result.fst > 0.0);
    }

    #[test]
    fn test_nei_gst() {
        let pops = vec![
            PopulationFreq::new(0, 0.7, 100),
            PopulationFreq::new(1, 0.3, 100),
        ];
        let est = FstEstimator::new().with_method(FstMethod::Nei);
        let result = est.estimate_single_locus(&pops);
        assert!(result.fst > 0.0);
    }

    #[test]
    fn test_hudson_fst() {
        let pops = vec![
            PopulationFreq::new(0, 0.9, 30),
            PopulationFreq::new(1, 0.1, 30),
        ];
        let est = FstEstimator::new().with_method(FstMethod::Hudson);
        let result = est.estimate_single_locus(&pops);
        assert!(result.fst > 0.3);
    }

    #[test]
    fn test_pairwise_fst_matrix() {
        let pops = vec![
            PopulationFreq::new(0, 0.5, 50),
            PopulationFreq::new(1, 0.3, 50),
            PopulationFreq::new(2, 0.7, 50),
        ];
        let est = FstEstimator::new().with_method(FstMethod::Wright);
        let matrix = est.pairwise_fst(&pops);
        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0]).abs() < EPS); // diagonal = 0
        assert!((matrix[0][1] - matrix[1][0]).abs() < EPS); // symmetric
    }

    #[test]
    fn test_multi_locus() {
        let loci = vec![
            vec![PopulationFreq::new(0, 0.8, 50), PopulationFreq::new(1, 0.2, 50)],
            vec![PopulationFreq::new(0, 0.7, 50), PopulationFreq::new(1, 0.3, 50)],
        ];
        let est = FstEstimator::new().with_method(FstMethod::Wright);
        let result = est.estimate_multi_locus(&loci);
        assert_eq!(result.n_loci, 2);
        assert!(result.fst > 0.0);
    }

    #[test]
    fn test_jost_d() {
        let d = jost_d(0.48, 0.32, 2);
        assert!(d > 0.0);
        assert!(d <= 1.0);
    }

    #[test]
    fn test_jost_d_no_diff() {
        let d = jost_d(0.42, 0.42, 2);
        assert!(d.abs() < EPS);
    }

    #[test]
    fn test_pop_freq_display() {
        let pf = PopulationFreq::new(1, 0.35, 100);
        let s = format!("{}", pf);
        assert!(s.contains("Pop 1"));
    }

    #[test]
    fn test_fst_result_display() {
        let r = FstResult {
            fst: 0.15, fis: 0.02, fit: 0.17,
            method: FstMethod::WeirCockerham, n_pops: 3, n_loci: 10,
        };
        let s = format!("{}", r);
        assert!(s.contains("Weir-Cockerham"));
    }

    #[test]
    fn test_estimator_display() {
        let est = FstEstimator::new();
        let s = format!("{}", est);
        assert!(s.contains("Weir-Cockerham"));
    }

    #[test]
    fn test_insufficient_pops() {
        let pops = vec![PopulationFreq::new(0, 0.5, 50)];
        let est = FstEstimator::new();
        let result = est.estimate_single_locus(&pops);
        assert!((result.fst - 0.0).abs() < EPS);
    }

    #[test]
    fn test_nm_edge_cases() {
        assert!((nm_from_fst(0.0) - 0.0).abs() < EPS);
        assert!((nm_from_fst(1.0) - 0.0).abs() < EPS);
    }
}
