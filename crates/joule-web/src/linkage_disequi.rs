//! Linkage disequilibrium (LD) computation and decay analysis.
//!
//! Calculates D, D', r-squared between biallelic loci, LD decay
//! curves over physical distance, and haplotype frequency estimation.
//! All operations are std-only with f64 math.

use std::fmt;

// ── Haplotype Counts ────────────────────────────────────────────────

/// Observed haplotype counts for two biallelic loci.
///
/// Alleles at locus 1: A/a; locus 2: B/b.
/// Haplotypes: AB, Ab, aB, ab.
#[derive(Debug, Clone, PartialEq)]
pub struct HaplotypeCounts {
    pub n_ab_upper: u64,
    pub n_a_lower_b: u64,
    pub n_a_b_lower: u64,
    pub n_lower_ab: u64,
}

impl HaplotypeCounts {
    pub fn new(ab: u64, a_b: u64, a_b2: u64, lower_ab: u64) -> Self {
        Self {
            n_ab_upper: ab,
            n_a_lower_b: a_b,
            n_a_b_lower: a_b2,
            n_lower_ab: lower_ab,
        }
    }

    /// Total haplotype count.
    pub fn total(&self) -> u64 {
        self.n_ab_upper + self.n_a_lower_b + self.n_a_b_lower + self.n_lower_ab
    }

    /// Haplotype frequencies as (f_AB, f_Ab, f_aB, f_ab).
    pub fn frequencies(&self) -> (f64, f64, f64, f64) {
        let n = self.total() as f64;
        if n == 0.0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        (
            self.n_ab_upper as f64 / n,
            self.n_a_lower_b as f64 / n,
            self.n_a_b_lower as f64 / n,
            self.n_lower_ab as f64 / n,
        )
    }

    /// Marginal allele frequencies: (p_A, p_B).
    pub fn marginal_freqs(&self) -> (f64, f64) {
        let (f_ab, f_a_b, f_a_b2, _) = self.frequencies();
        let p_a = f_ab + f_a_b;
        let p_b = f_ab + f_a_b2;
        (p_a, p_b)
    }
}

impl fmt::Display for HaplotypeCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (fa, fb, fc, fd) = self.frequencies();
        write!(
            f,
            "AB={:.4} Ab={:.4} aB={:.4} ab={:.4} (n={})",
            fa, fb, fc, fd, self.total()
        )
    }
}

// ── LD Measure ──────────────────────────────────────────────────────

/// Linkage disequilibrium statistics between two loci.
#[derive(Debug, Clone, PartialEq)]
pub struct LdMeasure {
    pub d: f64,
    pub d_prime: f64,
    pub r_squared: f64,
    pub chi_squared: f64,
    pub n_haplotypes: u64,
}

impl LdMeasure {
    /// Whether LD is significant at given chi-squared threshold (1 df).
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.chi_squared > threshold
    }
}

impl fmt::Display for LdMeasure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "D={:.6} D'={:.4} r²={:.4} chi²={:.2} n={}",
            self.d, self.d_prime, self.r_squared, self.chi_squared, self.n_haplotypes
        )
    }
}

// ── LD Calculator ───────────────────────────────────────────────────

/// Configurable LD calculator.
#[derive(Debug, Clone)]
pub struct LdCalculator {
    min_maf: f64,
    min_sample: usize,
    r_squared_threshold: f64,
}

impl Default for LdCalculator {
    fn default() -> Self {
        Self {
            min_maf: 0.01,
            min_sample: 10,
            r_squared_threshold: 0.0,
        }
    }
}

impl LdCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_min_maf(mut self, maf: f64) -> Self {
        self.min_maf = maf.clamp(0.0, 0.5);
        self
    }

    pub fn with_min_sample(mut self, n: usize) -> Self {
        self.min_sample = n;
        self
    }

    pub fn with_r_squared_threshold(mut self, t: f64) -> Self {
        self.r_squared_threshold = t.clamp(0.0, 1.0);
        self
    }

    /// Calculate LD from haplotype counts.
    pub fn calculate(&self, haps: &HaplotypeCounts) -> LdMeasure {
        let n = haps.total() as f64;
        if n < self.min_sample as f64 {
            return LdMeasure {
                d: 0.0, d_prime: 0.0, r_squared: 0.0, chi_squared: 0.0,
                n_haplotypes: haps.total(),
            };
        }

        let (f_ab, _f_a_b, _f_a_b2, _f_lower_ab) = haps.frequencies();
        let (p_a, p_b) = haps.marginal_freqs();
        let p_a2 = 1.0 - p_a;
        let p_b2 = 1.0 - p_b;

        // D = f(AB) - p(A)*p(B)
        let d = f_ab - p_a * p_b;

        // D' = D / D_max
        let d_max = if d >= 0.0 {
            (p_a * p_b2).min(p_a2 * p_b)
        } else {
            (p_a * p_b).min(p_a2 * p_b2)
        };
        let d_prime = if d_max.abs() > 1e-15 { (d / d_max).clamp(-1.0, 1.0) } else { 0.0 };

        // r² = D² / (p_A * p_a * p_B * p_b)
        let denom = p_a * p_a2 * p_b * p_b2;
        let r_sq = if denom > 1e-15 { (d * d) / denom } else { 0.0 };

        // Chi-squared = n * r²
        let chi2 = n * r_sq;

        LdMeasure {
            d,
            d_prime,
            r_squared: r_sq,
            chi_squared: chi2,
            n_haplotypes: haps.total(),
        }
    }

    /// Calculate LD from genotype data at two loci.
    /// Genotypes encoded as 0, 1, 2 (number of alt alleles), -1 = missing.
    pub fn from_genotypes(&self, geno1: &[i32], geno2: &[i32]) -> LdMeasure {
        let haps = estimate_haplotypes_em(geno1, geno2, 50);
        self.calculate(&haps)
    }

    /// LD matrix for multiple loci (upper triangle + diagonal).
    pub fn ld_matrix(&self, loci_genos: &[Vec<i32>]) -> Vec<Vec<f64>> {
        let n = loci_genos.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let ld = self.from_genotypes(&loci_genos[i], &loci_genos[j]);
                matrix[i][j] = ld.r_squared;
                matrix[j][i] = ld.r_squared;
            }
        }
        matrix
    }

    /// Filter LD pairs above the r-squared threshold.
    pub fn significant_pairs(
        &self,
        loci_genos: &[Vec<i32>],
    ) -> Vec<(usize, usize, LdMeasure)> {
        let n = loci_genos.len();
        let mut result = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let ld = self.from_genotypes(&loci_genos[i], &loci_genos[j]);
                if ld.r_squared >= self.r_squared_threshold {
                    result.push((i, j, ld));
                }
            }
        }
        result
    }
}

impl fmt::Display for LdCalculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LdCalculator(min_maf={:.4}, min_n={}, r²_thresh={:.4})",
            self.min_maf, self.min_sample, self.r_squared_threshold
        )
    }
}

// ── LD Decay ────────────────────────────────────────────────────────

/// A point on the LD decay curve.
#[derive(Debug, Clone, PartialEq)]
pub struct LdDecayPoint {
    pub distance_bp: u64,
    pub mean_r_squared: f64,
    pub n_pairs: usize,
}

impl fmt::Display for LdDecayPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}bp r²={:.4} (n={})", self.distance_bp, self.mean_r_squared, self.n_pairs)
    }
}

/// Compute LD decay curve by binning pairs by distance.
pub fn ld_decay_curve(
    pairs: &[(u64, f64)], // (distance_bp, r_squared)
    bin_size: u64,
    max_distance: u64,
) -> Vec<LdDecayPoint> {
    let n_bins = (max_distance / bin_size) as usize + 1;
    let mut sums = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];

    for &(dist, r2) in pairs {
        if dist > max_distance {
            continue;
        }
        let bin = (dist / bin_size) as usize;
        if bin < n_bins {
            sums[bin] += r2;
            counts[bin] += 1;
        }
    }

    (0..n_bins)
        .filter(|i| counts[*i] > 0)
        .map(|i| LdDecayPoint {
            distance_bp: (i as u64) * bin_size + bin_size / 2,
            mean_r_squared: sums[i] / counts[i] as f64,
            n_pairs: counts[i],
        })
        .collect()
}

/// Fit exponential LD decay: r² = a * exp(-distance / c).
/// Returns (a, c) estimated by log-linear regression.
pub fn fit_ld_decay(points: &[LdDecayPoint]) -> (f64, f64) {
    let filtered: Vec<&LdDecayPoint> = points
        .iter()
        .filter(|p| p.mean_r_squared > 1e-10)
        .collect();

    if filtered.len() < 2 {
        return (0.0, 1.0);
    }

    let n = filtered.len() as f64;
    let sum_x: f64 = filtered.iter().map(|p| p.distance_bp as f64).sum();
    let sum_y: f64 = filtered.iter().map(|p| p.mean_r_squared.ln()).sum();
    let sum_xy: f64 = filtered
        .iter()
        .map(|p| p.distance_bp as f64 * p.mean_r_squared.ln())
        .sum();
    let sum_x2: f64 = filtered
        .iter()
        .map(|p| {
            let x = p.distance_bp as f64;
            x * x
        })
        .sum();

    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, 1.0);
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;

    let a = intercept.exp();
    let c = if slope.abs() > 1e-15 { -1.0 / slope } else { 1.0 };

    (a, c.abs())
}

// ── EM Haplotype Estimation ─────────────────────────────────────────

/// Estimate haplotype frequencies from unphased genotype data using EM.
pub fn estimate_haplotypes_em(
    geno1: &[i32],
    geno2: &[i32],
    max_iter: usize,
) -> HaplotypeCounts {
    let n = geno1.len().min(geno2.len());
    if n == 0 {
        return HaplotypeCounts::new(0, 0, 0, 0);
    }

    // Count unambiguous haplotypes and double-hets
    let mut n_ab = 0.0_f64;
    let mut n_a_b = 0.0_f64;
    let mut n_a_b2 = 0.0_f64;
    let mut n_lower_ab = 0.0_f64;
    let mut n_dhet = 0.0_f64;
    let mut total = 0.0_f64;

    for i in 0..n {
        if geno1[i] < 0 || geno2[i] < 0 {
            continue;
        }
        let (g1, g2) = (geno1[i], geno2[i]);
        total += 2.0;
        match (g1, g2) {
            (0, 0) => n_ab += 2.0,
            (0, 1) => { n_ab += 1.0; n_a_b += 1.0; }
            (0, 2) => n_a_b += 2.0,
            (1, 0) => { n_ab += 1.0; n_a_b2 += 1.0; }
            (1, 1) => n_dhet += 2.0, // ambiguous
            (1, 2) => { n_a_b += 1.0; n_lower_ab += 1.0; }
            (2, 0) => n_a_b2 += 2.0,
            (2, 1) => { n_a_b2 += 1.0; n_lower_ab += 1.0; }
            (2, 2) => n_lower_ab += 2.0,
            _ => {}
        }
    }

    if total == 0.0 {
        return HaplotypeCounts::new(0, 0, 0, 0);
    }

    // EM for double-het phase resolution
    // Initialize with equal split
    let mut p_coupling = 0.5;

    for _ in 0..max_iter {
        let f_ab = (n_ab + p_coupling * n_dhet) / total;
        let f_a_b = (n_a_b + (1.0 - p_coupling) * n_dhet) / total;
        let f_a_b2 = (n_a_b2 + (1.0 - p_coupling) * n_dhet) / total;
        let f_lower_ab = (n_lower_ab + p_coupling * n_dhet) / total;

        let coupling = f_ab * f_lower_ab;
        let repulsion = f_a_b * f_a_b2;
        let denom = coupling + repulsion;

        let new_p = if denom > 1e-15 { coupling / denom } else { 0.5 };
        if (new_p - p_coupling).abs() < 1e-10 {
            break;
        }
        p_coupling = new_p;
    }

    let final_ab = n_ab + p_coupling * n_dhet;
    let final_a_b = n_a_b + (1.0 - p_coupling) * n_dhet;
    let final_a_b2 = n_a_b2 + (1.0 - p_coupling) * n_dhet;
    let final_lower = n_lower_ab + p_coupling * n_dhet;

    HaplotypeCounts::new(
        final_ab.round() as u64,
        final_a_b.round() as u64,
        final_a_b2.round() as u64,
        final_lower.round() as u64,
    )
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_haplotype_frequencies() {
        let hc = HaplotypeCounts::new(30, 20, 20, 30);
        let (a, b, c, d) = hc.frequencies();
        assert!((a - 0.3).abs() < EPS);
        assert!((d - 0.3).abs() < EPS);
        assert!((b + c - 0.4).abs() < EPS);
    }

    #[test]
    fn test_marginal_freqs() {
        let hc = HaplotypeCounts::new(40, 10, 10, 40);
        let (pa, pb) = hc.marginal_freqs();
        assert!((pa - 0.5).abs() < EPS);
        assert!((pb - 0.5).abs() < EPS);
    }

    #[test]
    fn test_ld_no_disequilibrium() {
        // p_A=0.5, p_B=0.5, f(AB)=0.25 -> D=0
        let hc = HaplotypeCounts::new(25, 25, 25, 25);
        let calc = LdCalculator::new();
        let ld = calc.calculate(&hc);
        assert!(ld.d.abs() < EPS);
        assert!(ld.r_squared < EPS);
    }

    #[test]
    fn test_ld_complete() {
        // Complete LD: only AB and ab haplotypes
        let hc = HaplotypeCounts::new(50, 0, 0, 50);
        let calc = LdCalculator::new();
        let ld = calc.calculate(&hc);
        assert!((ld.d_prime - 1.0).abs() < EPS);
        assert!((ld.r_squared - 1.0).abs() < EPS);
    }

    #[test]
    fn test_ld_partial() {
        let hc = HaplotypeCounts::new(35, 15, 15, 35);
        let calc = LdCalculator::new();
        let ld = calc.calculate(&hc);
        assert!(ld.d > 0.0);
        assert!(ld.r_squared > 0.0);
        assert!(ld.r_squared < 1.0);
    }

    #[test]
    fn test_ld_negative() {
        // Repulsion: excess of Ab and aB
        let hc = HaplotypeCounts::new(10, 40, 40, 10);
        let calc = LdCalculator::new();
        let ld = calc.calculate(&hc);
        assert!(ld.d < 0.0);
    }

    #[test]
    fn test_ld_from_genotypes() {
        // All individuals are 0/0 at both loci -> monomorphic, LD=0
        let g1 = vec![0; 20];
        let g2 = vec![0; 20];
        let calc = LdCalculator::new();
        let ld = calc.from_genotypes(&g1, &g2);
        assert!(ld.r_squared.abs() < EPS);
    }

    #[test]
    fn test_ld_matrix_symmetric() {
        let loci = vec![
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
            vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
            vec![2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
        ];
        let calc = LdCalculator::new();
        let mat = calc.ld_matrix(&loci);
        assert!((mat[0][1] - mat[1][0]).abs() < EPS);
        assert!((mat[0][2] - mat[2][0]).abs() < EPS);
        assert!((mat[0][0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_significant_pairs() {
        let loci = vec![
            vec![0, 0, 0, 2, 2, 2, 0, 0, 2, 2],
            vec![0, 0, 0, 2, 2, 2, 0, 0, 2, 2],
        ];
        let calc = LdCalculator::new().with_r_squared_threshold(0.5);
        let pairs = calc.significant_pairs(&loci);
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_ld_decay_binning() {
        let pairs = vec![
            (100, 0.9),
            (200, 0.85),
            (1100, 0.5),
            (1200, 0.45),
            (5100, 0.1),
        ];
        let curve = ld_decay_curve(&pairs, 1000, 10000);
        assert!(curve.len() >= 2);
        // First bin should have higher r² than last
        assert!(curve[0].mean_r_squared > curve.last().unwrap().mean_r_squared);
    }

    #[test]
    fn test_fit_ld_decay() {
        let points = vec![
            LdDecayPoint { distance_bp: 500, mean_r_squared: 0.8, n_pairs: 10 },
            LdDecayPoint { distance_bp: 5000, mean_r_squared: 0.4, n_pairs: 10 },
            LdDecayPoint { distance_bp: 50000, mean_r_squared: 0.05, n_pairs: 10 },
        ];
        let (a, c) = fit_ld_decay(&points);
        assert!(a > 0.0);
        assert!(c > 0.0);
    }

    #[test]
    fn test_em_haplotype_all_homozygous() {
        // All 0/0 at both loci -> all AB haplotypes
        let g1 = vec![0; 10];
        let g2 = vec![0; 10];
        let hc = estimate_haplotypes_em(&g1, &g2, 50);
        assert_eq!(hc.n_ab_upper, 20);
        assert_eq!(hc.n_a_lower_b, 0);
    }

    #[test]
    fn test_em_handles_missing() {
        let g1 = vec![0, 1, -1, 2];
        let g2 = vec![0, 1, 2, -1];
        let hc = estimate_haplotypes_em(&g1, &g2, 50);
        // Only 2 valid individuals
        assert!(hc.total() > 0);
    }

    #[test]
    fn test_haplotype_display() {
        let hc = HaplotypeCounts::new(25, 25, 25, 25);
        let s = format!("{}", hc);
        assert!(s.contains("n=100"));
    }

    #[test]
    fn test_ld_measure_display() {
        let ld = LdMeasure {
            d: 0.05, d_prime: 0.8, r_squared: 0.64,
            chi_squared: 128.0, n_haplotypes: 200,
        };
        let s = format!("{}", ld);
        assert!(s.contains("r²="));
    }

    #[test]
    fn test_calculator_display() {
        let calc = LdCalculator::new().with_min_maf(0.05);
        let s = format!("{}", calc);
        assert!(s.contains("0.0500"));
    }

    #[test]
    fn test_empty_haplotypes() {
        let hc = HaplotypeCounts::new(0, 0, 0, 0);
        let (a, b, c, d) = hc.frequencies();
        assert!((a + b + c + d).abs() < EPS);
    }

    #[test]
    fn test_decay_point_display() {
        let p = LdDecayPoint { distance_bp: 1000, mean_r_squared: 0.5, n_pairs: 42 };
        let s = format!("{}", p);
        assert!(s.contains("1000bp"));
    }
}
