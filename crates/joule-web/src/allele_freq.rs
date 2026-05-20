//! Allele frequency estimation for population genetics.
//!
//! Provides minor allele frequency (MAF) calculation, allele counting
//! from genotype data, site frequency spectrum (SFS) construction,
//! and folded/unfolded spectrum transforms. All operations are std-only
//! with f64 arithmetic.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────

const MIN_SAMPLE_SIZE: usize = 2;

// ── Genotype Representation ─────────────────────────────────────────

/// Diploid genotype at a biallelic locus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Genotype {
    /// Homozygous reference (0/0).
    HomRef,
    /// Heterozygous (0/1).
    Het,
    /// Homozygous alternate (1/1).
    HomAlt,
    /// Missing data.
    Missing,
}

impl Genotype {
    /// Number of alternate alleles carried (0, 1, or 2).
    pub fn alt_count(self) -> Option<u32> {
        match self {
            Self::HomRef => Some(0),
            Self::Het => Some(1),
            Self::HomAlt => Some(2),
            Self::Missing => None,
        }
    }

    /// Number of reference alleles carried.
    pub fn ref_count(self) -> Option<u32> {
        self.alt_count().map(|a| 2 - a)
    }
}

impl fmt::Display for Genotype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HomRef => write!(f, "0/0"),
            Self::Het => write!(f, "0/1"),
            Self::HomAlt => write!(f, "1/1"),
            Self::Missing => write!(f, "./."),
        }
    }
}

// ── Allele Counts ───────────────────────────────────────────────────

/// Raw allele counts at a single biallelic locus.
#[derive(Debug, Clone, PartialEq)]
pub struct AlleleCounts {
    pub ref_count: u64,
    pub alt_count: u64,
    pub missing: u64,
}

impl AlleleCounts {
    /// Total observed alleles (excludes missing).
    pub fn total(&self) -> u64 {
        self.ref_count + self.alt_count
    }

    /// Alternate allele frequency (p_alt).
    pub fn alt_frequency(&self) -> f64 {
        let t = self.total();
        if t == 0 { return 0.0; }
        self.alt_count as f64 / t as f64
    }

    /// Reference allele frequency (p_ref = 1 - p_alt).
    pub fn ref_frequency(&self) -> f64 {
        1.0 - self.alt_frequency()
    }

    /// Minor allele frequency: min(p_ref, p_alt).
    pub fn maf(&self) -> f64 {
        let p = self.alt_frequency();
        p.min(1.0 - p)
    }

    /// Which allele is minor? Returns `true` if alt is minor.
    pub fn alt_is_minor(&self) -> bool {
        self.alt_count <= self.ref_count
    }

    /// Number of observed (non-missing) individuals.
    pub fn n_individuals(&self) -> u64 {
        self.total() / 2
    }
}

impl fmt::Display for AlleleCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ref={} alt={} missing={} MAF={:.4}",
            self.ref_count,
            self.alt_count,
            self.missing,
            self.maf()
        )
    }
}

// ── Frequency Estimator ─────────────────────────────────────────────

/// Configuration for allele frequency estimation.
#[derive(Debug, Clone)]
pub struct FrequencyEstimator {
    min_call_rate: f64,
    maf_threshold: f64,
    apply_continuity_correction: bool,
}

impl Default for FrequencyEstimator {
    fn default() -> Self {
        Self {
            min_call_rate: 0.0,
            maf_threshold: 0.0,
            apply_continuity_correction: false,
        }
    }
}

impl FrequencyEstimator {
    /// Create a new estimator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Minimum call rate (fraction of non-missing) to include a site.
    pub fn with_min_call_rate(mut self, rate: f64) -> Self {
        self.min_call_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// MAF filter: sites below this threshold are excluded.
    pub fn with_maf_threshold(mut self, threshold: f64) -> Self {
        self.maf_threshold = threshold.clamp(0.0, 0.5);
        self
    }

    /// Apply Agresti-Coull continuity correction (+2 each allele).
    pub fn with_continuity_correction(mut self, apply: bool) -> Self {
        self.apply_continuity_correction = apply;
        self
    }

    /// Count alleles from a slice of genotypes at one locus.
    pub fn count_alleles(&self, genotypes: &[Genotype]) -> AlleleCounts {
        let mut ref_c = 0u64;
        let mut alt_c = 0u64;
        let mut miss = 0u64;

        for &g in genotypes {
            match g.alt_count() {
                Some(a) => {
                    alt_c += a as u64;
                    ref_c += (2 - a) as u64;
                }
                None => miss += 1,
            }
        }

        if self.apply_continuity_correction {
            ref_c += 2;
            alt_c += 2;
        }

        AlleleCounts {
            ref_count: ref_c,
            alt_count: alt_c,
            missing: miss,
        }
    }

    /// Count alleles across multiple loci.
    pub fn count_multi_locus(&self, loci: &[Vec<Genotype>]) -> Vec<AlleleCounts> {
        loci.iter().map(|g| self.count_alleles(g)).collect()
    }

    /// Check whether a locus passes call-rate and MAF filters.
    pub fn passes_filters(&self, counts: &AlleleCounts, total_individuals: usize) -> bool {
        if total_individuals < MIN_SAMPLE_SIZE {
            return false;
        }
        let call_rate = counts.n_individuals() as f64 / total_individuals as f64;
        if call_rate < self.min_call_rate {
            return false;
        }
        if counts.maf() < self.maf_threshold {
            return false;
        }
        true
    }

    /// Estimate allele frequency with optional Wilson interval (95%).
    pub fn frequency_with_ci(&self, counts: &AlleleCounts) -> FrequencyEstimate {
        let n = counts.total() as f64;
        let p = counts.alt_frequency();

        let (lower, upper) = if n > 0.0 {
            wilson_interval(p, n)
        } else {
            (0.0, 0.0)
        };

        FrequencyEstimate {
            frequency: p,
            lower_95: lower,
            upper_95: upper,
            n_alleles: counts.total(),
        }
    }
}

impl fmt::Display for FrequencyEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FreqEstimator(call_rate>={:.2}, MAF>={:.4}, correction={})",
            self.min_call_rate, self.maf_threshold, self.apply_continuity_correction
        )
    }
}

// ── Frequency Estimate with CI ──────────────────────────────────────

/// Point estimate with 95% Wilson confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyEstimate {
    pub frequency: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub n_alleles: u64,
}

impl fmt::Display for FrequencyEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p={:.4} [{:.4}, {:.4}] n={}",
            self.frequency, self.lower_95, self.upper_95, self.n_alleles
        )
    }
}

// ── Site Frequency Spectrum ─────────────────────────────────────────

/// Site frequency spectrum (SFS) for a sample of `n` diploid individuals.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteFrequencySpectrum {
    /// Counts: entry `i` = number of sites with `i` derived alleles.
    /// Length = 2n + 1 (indices 0..=2n).
    pub counts: Vec<u64>,
    pub sample_size: usize,
    pub folded: bool,
}

impl SiteFrequencySpectrum {
    /// Build an unfolded SFS from a set of allele counts.
    pub fn from_counts(all_counts: &[AlleleCounts], sample_size: usize) -> Self {
        let n2 = 2 * sample_size;
        let mut sfs = vec![0u64; n2 + 1];
        for ac in all_counts {
            let idx = ac.alt_count as usize;
            if idx <= n2 {
                sfs[idx] += 1;
            }
        }
        Self {
            counts: sfs,
            sample_size,
            folded: false,
        }
    }

    /// Fold the SFS (merge frequency `i` with `2n - i`).
    pub fn fold(&self) -> Self {
        let n2 = 2 * self.sample_size;
        let half = n2 / 2;
        let mut folded = vec![0u64; half + 1];
        for (i, &c) in self.counts.iter().enumerate() {
            let j = i.min(n2 - i);
            if j <= half {
                folded[j] += c;
            }
        }
        Self {
            counts: folded,
            sample_size: self.sample_size,
            folded: true,
        }
    }

    /// Total number of segregating sites (exclude monomorphic at 0 and 2n).
    pub fn segregating_sites(&self) -> u64 {
        let n2 = self.counts.len() - 1;
        self.counts[1..n2].iter().sum()
    }

    /// Mean pairwise differences (Tajima's pi) from the SFS.
    pub fn pi(&self) -> f64 {
        let n2 = 2 * self.sample_size;
        let n2f = n2 as f64;
        let mut sum = 0.0;
        for (i, &c) in self.counts.iter().enumerate() {
            if i == 0 || i == n2 {
                continue;
            }
            let freq = i as f64 / n2f;
            sum += c as f64 * 2.0 * freq * (1.0 - freq) * n2f / (n2f - 1.0);
        }
        sum
    }

    /// Watterson's theta from segregating sites.
    pub fn theta_w(&self) -> f64 {
        let s = self.segregating_sites() as f64;
        let n = 2 * self.sample_size;
        let a1 = harmonic(n - 1);
        if a1 == 0.0 { 0.0 } else { s / a1 }
    }

    /// Number of entries in the spectrum.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Whether the spectrum is empty.
    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Normalize to proportions.
    pub fn normalized(&self) -> Vec<f64> {
        let total: u64 = self.counts.iter().sum();
        if total == 0 {
            return vec![0.0; self.counts.len()];
        }
        self.counts.iter().map(|c| *c as f64 / total as f64).collect()
    }
}

impl fmt::Display for SiteFrequencySpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = if self.folded { "folded" } else { "unfolded" };
        write!(
            f,
            "SFS({}, n={}, S={}, pi={:.4}, theta_w={:.4})",
            label,
            self.sample_size,
            self.segregating_sites(),
            self.pi(),
            self.theta_w()
        )
    }
}

// ── Helper Functions ────────────────────────────────────────────────

/// Wilson score interval for proportion `p` with `n` observations (z=1.96).
fn wilson_interval(p: f64, n: f64) -> (f64, f64) {
    let z = 1.96_f64;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denom;
    let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    ((centre - margin).max(0.0), (centre + margin).min(1.0))
}

/// Harmonic number H_n = sum_{i=1}^{n} 1/i.
fn harmonic(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / i as f64).sum()
}

/// Second harmonic number sum_{i=1}^{n} 1/i^2.
pub fn harmonic2(n: usize) -> f64 {
    (1..=n).map(|i| 1.0 / (i as f64 * i as f64)).sum()
}

/// Convert genotype dosage array (0, 1, 2, -1=missing) to `Genotype` vec.
pub fn dosage_to_genotypes(dosages: &[i32]) -> Vec<Genotype> {
    dosages
        .iter()
        .map(|d| match d {
            0 => Genotype::HomRef,
            1 => Genotype::Het,
            2 => Genotype::HomAlt,
            _ => Genotype::Missing,
        })
        .collect()
}

/// Expected heterozygosity from allele frequency: 2p(1-p).
pub fn expected_het(p: f64) -> f64 {
    2.0 * p * (1.0 - p)
}

/// Observed heterozygosity from genotypes.
pub fn observed_het(genotypes: &[Genotype]) -> f64 {
    let mut het = 0u64;
    let mut total = 0u64;
    for &g in genotypes {
        if g == Genotype::Missing {
            continue;
        }
        total += 1;
        if g == Genotype::Het {
            het += 1;
        }
    }
    if total == 0 { 0.0 } else { het as f64 / total as f64 }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_genotype_alt_count() {
        assert_eq!(Genotype::HomRef.alt_count(), Some(0));
        assert_eq!(Genotype::Het.alt_count(), Some(1));
        assert_eq!(Genotype::HomAlt.alt_count(), Some(2));
        assert_eq!(Genotype::Missing.alt_count(), None);
    }

    #[test]
    fn test_genotype_display() {
        assert_eq!(format!("{}", Genotype::Het), "0/1");
        assert_eq!(format!("{}", Genotype::Missing), "./.");
    }

    #[test]
    fn test_allele_counts_basic() {
        let est = FrequencyEstimator::new();
        let genos = vec![Genotype::HomRef, Genotype::Het, Genotype::HomAlt];
        let ac = est.count_alleles(&genos);
        assert_eq!(ac.ref_count, 3);
        assert_eq!(ac.alt_count, 3);
        assert!((ac.maf() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_maf_calculation() {
        let ac = AlleleCounts { ref_count: 18, alt_count: 2, missing: 0 };
        assert!((ac.maf() - 0.1).abs() < EPS);
        assert!(ac.alt_is_minor());
    }

    #[test]
    fn test_all_ref() {
        let est = FrequencyEstimator::new();
        let genos = vec![Genotype::HomRef; 10];
        let ac = est.count_alleles(&genos);
        assert_eq!(ac.alt_count, 0);
        assert!((ac.maf() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_all_missing() {
        let est = FrequencyEstimator::new();
        let genos = vec![Genotype::Missing; 5];
        let ac = est.count_alleles(&genos);
        assert_eq!(ac.total(), 0);
        assert!((ac.alt_frequency() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_continuity_correction() {
        let est = FrequencyEstimator::new().with_continuity_correction(true);
        let genos = vec![Genotype::HomRef; 5];
        let ac = est.count_alleles(&genos);
        // Without correction: ref=10, alt=0. With: ref=12, alt=2.
        assert_eq!(ac.ref_count, 12);
        assert_eq!(ac.alt_count, 2);
    }

    #[test]
    fn test_call_rate_filter() {
        let est = FrequencyEstimator::new().with_min_call_rate(0.8);
        let ac = AlleleCounts { ref_count: 10, alt_count: 2, missing: 4 };
        // 6 individuals out of 10 total = 0.6 < 0.8
        assert!(!est.passes_filters(&ac, 10));
    }

    #[test]
    fn test_maf_filter() {
        let est = FrequencyEstimator::new().with_maf_threshold(0.05);
        let ac = AlleleCounts { ref_count: 99, alt_count: 1, missing: 0 };
        assert!(!est.passes_filters(&ac, 50));
    }

    #[test]
    fn test_wilson_ci() {
        let est = FrequencyEstimator::new();
        let ac = AlleleCounts { ref_count: 80, alt_count: 20, missing: 0 };
        let fe = est.frequency_with_ci(&ac);
        assert!((fe.frequency - 0.2).abs() < EPS);
        assert!(fe.lower_95 < 0.2);
        assert!(fe.upper_95 > 0.2);
        assert!(fe.lower_95 >= 0.0);
        assert!(fe.upper_95 <= 1.0);
    }

    #[test]
    fn test_sfs_construction() {
        // 5 individuals = 10 alleles
        let counts = vec![
            AlleleCounts { ref_count: 8, alt_count: 2, missing: 0 },
            AlleleCounts { ref_count: 6, alt_count: 4, missing: 0 },
            AlleleCounts { ref_count: 10, alt_count: 0, missing: 0 },
        ];
        let sfs = SiteFrequencySpectrum::from_counts(&counts, 5);
        assert_eq!(sfs.counts.len(), 11); // 0..=10
        assert_eq!(sfs.counts[0], 1); // monomorphic ref
        assert_eq!(sfs.counts[2], 1);
        assert_eq!(sfs.counts[4], 1);
        assert_eq!(sfs.segregating_sites(), 2);
    }

    #[test]
    fn test_sfs_fold() {
        let counts = vec![
            AlleleCounts { ref_count: 8, alt_count: 2, missing: 0 },
            AlleleCounts { ref_count: 2, alt_count: 8, missing: 0 },
        ];
        let sfs = SiteFrequencySpectrum::from_counts(&counts, 5);
        let folded = sfs.fold();
        assert!(folded.folded);
        assert_eq!(folded.counts[2], 2); // 2 and 8 both fold to 2
    }

    #[test]
    fn test_pi_monomorphic() {
        let counts = vec![
            AlleleCounts { ref_count: 10, alt_count: 0, missing: 0 },
        ];
        let sfs = SiteFrequencySpectrum::from_counts(&counts, 5);
        assert!((sfs.pi() - 0.0).abs() < EPS);
    }

    #[test]
    fn test_theta_w() {
        let counts = vec![
            AlleleCounts { ref_count: 8, alt_count: 2, missing: 0 },
            AlleleCounts { ref_count: 6, alt_count: 4, missing: 0 },
        ];
        let sfs = SiteFrequencySpectrum::from_counts(&counts, 5);
        let tw = sfs.theta_w();
        assert!(tw > 0.0);
    }

    #[test]
    fn test_expected_het() {
        assert!((expected_het(0.5) - 0.5).abs() < EPS);
        assert!((expected_het(0.0) - 0.0).abs() < EPS);
        assert!((expected_het(1.0) - 0.0).abs() < EPS);
        assert!((expected_het(0.3) - 0.42).abs() < EPS);
    }

    #[test]
    fn test_observed_het() {
        let genos = vec![Genotype::HomRef, Genotype::Het, Genotype::Het, Genotype::HomAlt];
        assert!((observed_het(&genos) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_dosage_conversion() {
        let genos = dosage_to_genotypes(&[0, 1, 2, -1]);
        assert_eq!(genos[0], Genotype::HomRef);
        assert_eq!(genos[1], Genotype::Het);
        assert_eq!(genos[2], Genotype::HomAlt);
        assert_eq!(genos[3], Genotype::Missing);
    }

    #[test]
    fn test_harmonic_numbers() {
        assert!((harmonic(1) - 1.0).abs() < EPS);
        // H_4 = 1 + 1/2 + 1/3 + 1/4 = 2.08333...
        assert!((harmonic(4) - 25.0 / 12.0).abs() < EPS);
    }

    #[test]
    fn test_estimator_display() {
        let est = FrequencyEstimator::new().with_min_call_rate(0.95).with_maf_threshold(0.01);
        let s = format!("{}", est);
        assert!(s.contains("0.95"));
        assert!(s.contains("0.01"));
    }

    #[test]
    fn test_sfs_normalized() {
        let counts = vec![
            AlleleCounts { ref_count: 8, alt_count: 2, missing: 0 },
            AlleleCounts { ref_count: 6, alt_count: 4, missing: 0 },
        ];
        let sfs = SiteFrequencySpectrum::from_counts(&counts, 5);
        let norm = sfs.normalized();
        let total: f64 = norm.iter().sum();
        assert!((total - 1.0).abs() < EPS);
    }
}
