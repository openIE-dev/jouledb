//! Isotope distribution, averagine model, and monoisotopic mass.
//!
//! Computes theoretical isotope envelopes from elemental composition
//! or via the averagine approximation, identifies the monoisotopic
//! peak, and scores observed patterns against theoretical ones.

use std::fmt;

// ── Atomic masses ───────────────────────────────────────────────

/// Average atomic masses of common elements.
pub const MASS_C: f64 = 12.0;
pub const MASS_H: f64 = 1.007_940;
pub const MASS_N: f64 = 14.006_72;
pub const MASS_O: f64 = 15.999_40;
pub const MASS_S: f64 = 32.065_00;

/// Natural abundance of ¹³C (primary contributor to +1 peak).
pub const C13_ABUNDANCE: f64 = 0.010_9;

/// Mass difference between ¹³C and ¹²C.
pub const C13_DELTA: f64 = 1.003_355;

// ── Averagine model ─────────────────────────────────────────────

/// Averagine: average amino acid composition per residue.
///
/// Senko et al. (1995): C₄.₉₃₈₄ H₇.₇₅₈₃ N₁.₃₅₇₇ O₁.₄₇₇₃ S₀.₀₄₁₇
#[derive(Debug, Clone, Copy)]
pub struct Averagine {
    pub c_per_residue: f64,
    pub h_per_residue: f64,
    pub n_per_residue: f64,
    pub o_per_residue: f64,
    pub s_per_residue: f64,
}

impl Averagine {
    /// Standard averagine model.
    pub fn standard() -> Self {
        Self {
            c_per_residue: 4.9384,
            h_per_residue: 7.7583,
            n_per_residue: 1.3577,
            o_per_residue: 1.4773,
            s_per_residue: 0.0417,
        }
    }

    /// Average residue mass from the averagine model.
    pub fn residue_mass(&self) -> f64 {
        self.c_per_residue * MASS_C
            + self.h_per_residue * MASS_H
            + self.n_per_residue * MASS_N
            + self.o_per_residue * MASS_O
            + self.s_per_residue * MASS_S
    }

    /// Estimate the number of residues from a neutral mass.
    pub fn residue_count(&self, neutral_mass: f64) -> f64 {
        neutral_mass / self.residue_mass()
    }

    /// Estimate the number of carbon atoms from a neutral mass.
    pub fn carbon_count(&self, neutral_mass: f64) -> f64 {
        self.residue_count(neutral_mass) * self.c_per_residue
    }
}

impl fmt::Display for Averagine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Averagine(C{:.2}H{:.2}N{:.2}O{:.2}S{:.3})",
            self.c_per_residue,
            self.h_per_residue,
            self.n_per_residue,
            self.o_per_residue,
            self.s_per_residue,
        )
    }
}

// ── ElementalComposition ────────────────────────────────────────

/// Elemental composition of a molecule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementalComposition {
    pub c: u32,
    pub h: u32,
    pub n: u32,
    pub o: u32,
    pub s: u32,
}

impl ElementalComposition {
    pub fn new(c: u32, h: u32, n: u32, o: u32, s: u32) -> Self {
        Self { c, h, n, o, s }
    }

    /// Monoisotopic mass (using ¹²C, ¹H, ¹⁴N, ¹⁶O, ³²S).
    pub fn monoisotopic_mass(&self) -> f64 {
        self.c as f64 * MASS_C
            + self.h as f64 * MASS_H
            + self.n as f64 * MASS_N
            + self.o as f64 * MASS_O
            + self.s as f64 * MASS_S
    }

    /// Estimate from a neutral mass using averagine.
    pub fn from_averagine(neutral_mass: f64) -> Self {
        let avg = Averagine::standard();
        let n_res = avg.residue_count(neutral_mass);
        Self {
            c: (n_res * avg.c_per_residue).round() as u32,
            h: (n_res * avg.h_per_residue).round() as u32,
            n: (n_res * avg.n_per_residue).round() as u32,
            o: (n_res * avg.o_per_residue).round() as u32,
            s: (n_res * avg.s_per_residue).round() as u32,
        }
    }
}

impl fmt::Display for ElementalComposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}H{}N{}O{}S{}", self.c, self.h, self.n, self.o, self.s)
    }
}

// ── IsotopePeak ─────────────────────────────────────────────────

/// A single peak in an isotope distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsotopePeak {
    pub index: usize,
    pub mass: f64,
    pub probability: f64,
}

impl IsotopePeak {
    pub fn new(index: usize, mass: f64, probability: f64) -> Self {
        Self { index, mass, probability }
    }
}

impl fmt::Display for IsotopePeak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M+{} mass={:.4} p={:.4}", self.index, self.mass, self.probability)
    }
}

// ── IsotopeDistribution ─────────────────────────────────────────

/// A computed isotope distribution (envelope).
#[derive(Debug, Clone)]
pub struct IsotopeDistribution {
    pub peaks: Vec<IsotopePeak>,
    pub monoisotopic_mass: f64,
}

impl IsotopeDistribution {
    pub fn new(mono_mass: f64, peaks: Vec<IsotopePeak>) -> Self {
        Self { monoisotopic_mass: mono_mass, peaks }
    }

    /// Index of the most abundant isotope peak.
    pub fn most_abundant_index(&self) -> usize {
        self.peaks
            .iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
            .map(|p| p.index)
            .unwrap_or(0)
    }

    /// The monoisotopic peak probability.
    pub fn monoisotopic_probability(&self) -> f64 {
        self.peaks.first().map(|p| p.probability).unwrap_or(0.0)
    }

    /// Average mass (intensity-weighted).
    pub fn average_mass(&self) -> f64 {
        let total_p: f64 = self.peaks.iter().map(|p| p.probability).sum();
        if total_p <= 0.0 {
            return self.monoisotopic_mass;
        }
        self.peaks.iter().map(|p| p.mass * p.probability).sum::<f64>() / total_p
    }

    /// Number of peaks in the distribution.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Whether the distribution is empty.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }
}

impl fmt::Display for IsotopeDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IsoDist mono={:.4} peaks={} most_abundant=M+{}",
            self.monoisotopic_mass,
            self.peaks.len(),
            self.most_abundant_index(),
        )
    }
}

// ── Poisson approximation ───────────────────────────────────────

/// Compute isotope distribution using the Poisson approximation
/// based on the number of carbon atoms.
///
/// P(k) = (lambda^k * e^(-lambda)) / k!
/// where lambda = n_carbon * C13_ABUNDANCE.
pub fn poisson_isotope_distribution(
    mono_mass: f64,
    n_carbons: f64,
    max_peaks: usize,
) -> IsotopeDistribution {
    let lambda = n_carbons * C13_ABUNDANCE;
    let mut peaks = Vec::with_capacity(max_peaks);
    let mut log_factorial = 0.0_f64;

    for k in 0..max_peaks {
        if k > 0 {
            log_factorial += (k as f64).ln();
        }
        let log_p = k as f64 * lambda.ln() - lambda - log_factorial;
        let prob = log_p.exp();
        let mass = mono_mass + k as f64 * C13_DELTA;
        peaks.push(IsotopePeak::new(k, mass, prob));
    }

    // Normalise.
    let total: f64 = peaks.iter().map(|p| p.probability).sum();
    if total > 0.0 {
        for p in &mut peaks {
            p.probability /= total;
        }
    }

    IsotopeDistribution::new(mono_mass, peaks)
}

/// Convenience: compute isotope distribution from a neutral mass
/// using the averagine model to estimate carbon count.
pub fn averagine_isotope_distribution(neutral_mass: f64, max_peaks: usize) -> IsotopeDistribution {
    let n_c = Averagine::standard().carbon_count(neutral_mass);
    poisson_isotope_distribution(neutral_mass, n_c, max_peaks)
}

// ── Fine-grained binomial model ─────────────────────────────────

/// Binomial model for isotope distribution:
/// P(k) = C(n,k) * p^k * (1-p)^(n-k)
/// where n = number of carbons, p = C13 abundance.
pub fn binomial_isotope_distribution(
    mono_mass: f64,
    n_carbons: u32,
    max_peaks: usize,
) -> IsotopeDistribution {
    let n = n_carbons as f64;
    let p = C13_ABUNDANCE;
    let q = 1.0 - p;
    let mut peaks = Vec::with_capacity(max_peaks);

    for k in 0..max_peaks.min(n_carbons as usize + 1) {
        let kf = k as f64;
        // Use log-space to avoid overflow.
        let log_comb = ln_binomial(n_carbons, k as u32);
        let log_p = log_comb + kf * p.ln() + (n - kf) * q.ln();
        let prob = log_p.exp();
        let mass = mono_mass + k as f64 * C13_DELTA;
        peaks.push(IsotopePeak::new(k, mass, prob));
    }

    // Normalise.
    let total: f64 = peaks.iter().map(|p| p.probability).sum();
    if total > 0.0 {
        for p in &mut peaks {
            p.probability /= total;
        }
    }

    IsotopeDistribution::new(mono_mass, peaks)
}

/// Log of binomial coefficient C(n,k) using Stirling-like sums.
fn ln_binomial(n: u32, k: u32) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

fn ln_factorial(n: u32) -> f64 {
    (2..=n).fold(0.0, |acc, i| acc + (i as f64).ln())
}

// ── Pattern scoring ─────────────────────────────────────────────

/// Dot-product score between an observed and theoretical isotope
/// pattern. Both patterns are treated as intensity vectors.
pub fn isotope_dot_product(observed: &[f64], theoretical: &[f64]) -> f64 {
    let len = observed.len().min(theoretical.len());
    if len == 0 {
        return 0.0;
    }
    let dot: f64 = (0..len).map(|i| observed[i] * theoretical[i]).sum();
    let mag_o: f64 = observed[..len].iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_t: f64 = theoretical[..len].iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_o <= 0.0 || mag_t <= 0.0 {
        return 0.0;
    }
    dot / (mag_o * mag_t)
}

/// Estimate the monoisotopic mass from an observed isotope envelope
/// by subtracting k * C13_DELTA from the most-intense peak.
pub fn estimate_monoisotopic(
    observed_mz: f64,
    charge: u8,
    most_abundant_offset: usize,
) -> f64 {
    observed_mz - most_abundant_offset as f64 * C13_DELTA / charge as f64
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_averagine_residue_mass() {
        let avg = Averagine::standard();
        let m = avg.residue_mass();
        // Should be roughly 111 Da.
        assert!(m > 110.0 && m < 115.0);
    }

    #[test]
    fn test_averagine_carbon_count() {
        let avg = Averagine::standard();
        let nc = avg.carbon_count(10000.0);
        assert!(nc > 400.0 && nc < 500.0);
    }

    #[test]
    fn test_elemental_composition_mass() {
        let comp = ElementalComposition::new(5, 10, 2, 3, 0);
        let m = comp.monoisotopic_mass();
        assert!(m > 0.0);
    }

    #[test]
    fn test_elemental_from_averagine() {
        let comp = ElementalComposition::from_averagine(1000.0);
        assert!(comp.c > 0);
        assert!(comp.h > 0);
    }

    #[test]
    fn test_poisson_distribution_sums_to_one() {
        let dist = poisson_isotope_distribution(1000.0, 45.0, 10);
        let total: f64 = dist.peaks.iter().map(|p| p.probability).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_poisson_monoisotopic_dominant_small() {
        // Small molecule: M+0 should be most abundant.
        let dist = poisson_isotope_distribution(200.0, 9.0, 5);
        assert_eq!(dist.most_abundant_index(), 0);
    }

    #[test]
    fn test_poisson_shift_for_large() {
        // Large protein: most abundant shifts right.
        let dist = poisson_isotope_distribution(20000.0, 900.0, 20);
        assert!(dist.most_abundant_index() > 0);
    }

    #[test]
    fn test_averagine_distribution() {
        let dist = averagine_isotope_distribution(1500.0, 8);
        assert_eq!(dist.len(), 8);
        assert!(dist.monoisotopic_mass > 0.0);
    }

    #[test]
    fn test_binomial_distribution_small() {
        let dist = binomial_isotope_distribution(500.0, 22, 6);
        let total: f64 = dist.peaks.iter().map(|p| p.probability).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_binomial_vs_poisson_agreement() {
        let binom = binomial_isotope_distribution(1000.0, 45, 8);
        let poisson = poisson_isotope_distribution(1000.0, 45.0, 8);
        // For large n, small p, binomial ≈ Poisson.
        for i in 0..8 {
            let diff = (binom.peaks[i].probability - poisson.peaks[i].probability).abs();
            assert!(diff < 0.02, "peak {} diff={}", i, diff);
        }
    }

    #[test]
    fn test_dot_product_identical() {
        let a = vec![1.0, 0.5, 0.1];
        assert!((isotope_dot_product(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(isotope_dot_product(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn test_estimate_monoisotopic() {
        let mono = estimate_monoisotopic(501.5, 1, 1);
        assert!((mono - (501.5 - C13_DELTA)).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_monoisotopic_doubly_charged() {
        let mono = estimate_monoisotopic(251.0, 2, 1);
        let expected = 251.0 - C13_DELTA / 2.0;
        assert!((mono - expected).abs() < 1e-6);
    }

    #[test]
    fn test_isotope_peak_display() {
        let p = IsotopePeak::new(2, 1002.0, 0.35);
        let d = format!("{}", p);
        assert!(d.contains("M+2"));
    }

    #[test]
    fn test_distribution_display() {
        let dist = averagine_isotope_distribution(1000.0, 5);
        let d = format!("{}", dist);
        assert!(d.contains("IsoDist"));
        assert!(d.contains("peaks=5"));
    }

    #[test]
    fn test_average_mass_greater_than_mono() {
        let dist = averagine_isotope_distribution(2000.0, 10);
        assert!(dist.average_mass() >= dist.monoisotopic_mass);
    }

    #[test]
    fn test_composition_display() {
        let c = ElementalComposition::new(10, 20, 3, 5, 1);
        assert_eq!(format!("{}", c), "C10H20N3O5S1");
    }

    #[test]
    fn test_ln_binomial_edge() {
        assert_eq!(ln_binomial(5, 0), 0.0);
        assert!(ln_binomial(10, 11).is_infinite());
    }
}
