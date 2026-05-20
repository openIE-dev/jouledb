//! Label-free protein quantification, spectral counting, and
//! intensity-based quantification.
//!
//! Provides spectral count-based protein abundance estimation,
//! normalised spectral abundance factor (NSAF), intensity-based
//! absolute quantification (iBAQ), and top-N peptide intensity
//! methods. Includes statistical comparisons between conditions.

use std::fmt;

// ── ProteinEntry ────────────────────────────────────────────────

/// Metadata for a protein used in quantification.
#[derive(Debug, Clone)]
pub struct ProteinEntry {
    pub accession: String,
    pub gene_name: String,
    pub length: usize,
    pub molecular_weight_da: f64,
}

impl ProteinEntry {
    pub fn new(accession: &str, gene_name: &str, length: usize) -> Self {
        // Rough MW estimate: 110 Da per amino acid.
        let mw = length as f64 * 110.0;
        Self {
            accession: accession.to_string(),
            gene_name: gene_name.to_string(),
            length,
            molecular_weight_da: mw,
        }
    }

    pub fn with_molecular_weight(mut self, mw: f64) -> Self {
        self.molecular_weight_da = mw;
        self
    }
}

impl fmt::Display for ProteinEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) len={} MW={:.0}",
            self.accession, self.gene_name, self.length, self.molecular_weight_da
        )
    }
}

// ── PeptideObservation ──────────────────────────────────────────

/// A single peptide observation with measured intensity.
#[derive(Debug, Clone)]
pub struct PeptideObservation {
    pub sequence: String,
    pub protein_accession: String,
    pub charge: u8,
    pub retention_time: f64,
    pub intensity: f64,
    pub score: f64,
}

impl PeptideObservation {
    pub fn new(sequence: &str, protein: &str, intensity: f64) -> Self {
        Self {
            sequence: sequence.to_string(),
            protein_accession: protein.to_string(),
            charge: 2,
            retention_time: 0.0,
            intensity,
            score: 0.0,
        }
    }

    pub fn with_charge(mut self, z: u8) -> Self {
        self.charge = z;
        self
    }

    pub fn with_retention_time(mut self, rt: f64) -> Self {
        self.retention_time = rt;
        self
    }

    pub fn with_score(mut self, s: f64) -> Self {
        self.score = s;
        self
    }
}

impl fmt::Display for PeptideObservation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] z={} I={:.0}",
            self.sequence, self.protein_accession, self.charge, self.intensity
        )
    }
}

// ── QuantMethod ─────────────────────────────────────────────────

/// Quantification strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    SpectralCount,
    Nsaf,
    Ibaq,
    TopN,
    SummedIntensity,
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpectralCount => write!(f, "SpC"),
            Self::Nsaf => write!(f, "NSAF"),
            Self::Ibaq => write!(f, "iBAQ"),
            Self::TopN => write!(f, "Top-N"),
            Self::SummedIntensity => write!(f, "SumInt"),
        }
    }
}

// ── QuantConfig ─────────────────────────────────────────────────

/// Configuration for label-free quantification.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub method: QuantMethod,
    pub top_n: usize,
    pub min_peptides: usize,
    pub unique_only: bool,
    pub log2_transform: bool,
}

impl QuantConfig {
    pub fn new(method: QuantMethod) -> Self {
        Self {
            method,
            top_n: 3,
            min_peptides: 1,
            unique_only: true,
            log2_transform: false,
        }
    }

    pub fn with_top_n(mut self, n: usize) -> Self {
        self.top_n = n.max(1);
        self
    }

    pub fn with_min_peptides(mut self, n: usize) -> Self {
        self.min_peptides = n.max(1);
        self
    }

    pub fn with_unique_only(mut self, yes: bool) -> Self {
        self.unique_only = yes;
        self
    }

    pub fn with_log2_transform(mut self, yes: bool) -> Self {
        self.log2_transform = yes;
        self
    }
}

impl fmt::Display for QuantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quant({}, top_n={}, min_pep={}{})",
            self.method,
            self.top_n,
            self.min_peptides,
            if self.unique_only { ", unique" } else { "" },
        )
    }
}

// ── ProteinQuantResult ──────────────────────────────────────────

/// Quantification result for a single protein.
#[derive(Debug, Clone)]
pub struct ProteinQuantResult {
    pub accession: String,
    pub gene_name: String,
    pub abundance: f64,
    pub peptide_count: usize,
    pub spectral_count: usize,
    pub sequence_coverage: f64,
    pub method: QuantMethod,
}

impl ProteinQuantResult {
    pub fn new(accession: &str, gene_name: &str, abundance: f64, method: QuantMethod) -> Self {
        Self {
            accession: accession.to_string(),
            gene_name: gene_name.to_string(),
            abundance,
            peptide_count: 0,
            spectral_count: 0,
            sequence_coverage: 0.0,
            method,
        }
    }

    pub fn with_peptide_count(mut self, n: usize) -> Self {
        self.peptide_count = n;
        self
    }

    pub fn with_spectral_count(mut self, n: usize) -> Self {
        self.spectral_count = n;
        self
    }

    pub fn with_coverage(mut self, cov: f64) -> Self {
        self.sequence_coverage = cov;
        self
    }
}

impl fmt::Display for ProteinQuantResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) {} abundance={:.4} pep={} SpC={}",
            self.accession,
            self.gene_name,
            self.method,
            self.abundance,
            self.peptide_count,
            self.spectral_count,
        )
    }
}

// ── Spectral counting ───────────────────────────────────────────

/// Count spectra per protein.
pub fn spectral_counts(observations: &[PeptideObservation]) -> Vec<(String, usize)> {
    let mut counts: Vec<(String, usize)> = Vec::new();
    for obs in observations {
        if let Some(entry) = counts.iter_mut().find(|(acc, _)| *acc == obs.protein_accession) {
            entry.1 += 1;
        } else {
            counts.push((obs.protein_accession.clone(), 1));
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1));
    counts
}

/// Normalised Spectral Abundance Factor.
///
/// NSAF_i = (SpC_i / L_i) / Σ(SpC_j / L_j)
pub fn compute_nsaf(
    counts: &[(String, usize)],
    proteins: &[ProteinEntry],
) -> Vec<(String, f64)> {
    let mut saf: Vec<(String, f64)> = counts
        .iter()
        .filter_map(|(acc, spc)| {
            let prot = proteins.iter().find(|p| p.accession == *acc)?;
            if prot.length == 0 {
                return None;
            }
            Some((acc.clone(), *spc as f64 / prot.length as f64))
        })
        .collect();

    let total: f64 = saf.iter().map(|(_, v)| v).sum();
    if total > 0.0 {
        for entry in &mut saf {
            entry.1 /= total;
        }
    }
    saf
}

// ── Intensity-based methods ─────────────────────────────────────

/// Sum intensities per protein.
pub fn summed_intensity(observations: &[PeptideObservation]) -> Vec<(String, f64)> {
    let mut sums: Vec<(String, f64)> = Vec::new();
    for obs in observations {
        if let Some(entry) = sums.iter_mut().find(|(acc, _)| *acc == obs.protein_accession) {
            entry.1 += obs.intensity;
        } else {
            sums.push((obs.protein_accession.clone(), obs.intensity));
        }
    }
    sums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sums
}

/// Top-N peptide intensity: average of the N most intense peptides.
pub fn top_n_intensity(observations: &[PeptideObservation], n: usize) -> Vec<(String, f64)> {
    // Group by protein.
    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for obs in observations {
        if let Some(entry) = groups.iter_mut().find(|(acc, _)| *acc == obs.protein_accession) {
            entry.1.push(obs.intensity);
        } else {
            groups.push((obs.protein_accession.clone(), vec![obs.intensity]));
        }
    }

    let mut results = Vec::new();
    for (acc, mut intensities) in groups {
        intensities.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let take = intensities.len().min(n);
        let avg: f64 = intensities[..take].iter().sum::<f64>() / take as f64;
        results.push((acc, avg));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

/// iBAQ: summed intensity divided by number of theoretical peptides.
///
/// `theoretical_peptides` maps protein accession to the count of
/// tryptic peptides in the theoretical digest.
pub fn compute_ibaq(
    observations: &[PeptideObservation],
    theoretical_peptides: &[(String, usize)],
) -> Vec<(String, f64)> {
    let sums = summed_intensity(observations);
    let mut results = Vec::new();

    for (acc, total_int) in &sums {
        let n_theor = theoretical_peptides
            .iter()
            .find(|(a, _)| a == acc)
            .map(|(_, n)| *n)
            .unwrap_or(1)
            .max(1);
        results.push((acc.clone(), total_int / n_theor as f64));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// ── Normalisation ───────────────────────────────────────────────

/// Median normalisation across samples.
pub fn median_normalise(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    if median > 0.0 {
        for v in values.iter_mut() {
            *v /= median;
        }
    }
}

/// Log2 transform (with small offset to avoid log(0)).
pub fn log2_transform(values: &mut [f64]) {
    for v in values.iter_mut() {
        *v = (*v + 1.0).log2();
    }
}

// ── Fold change ─────────────────────────────────────────────────

/// Compute fold change between two conditions.
pub fn fold_change(condition_a: f64, condition_b: f64) -> f64 {
    if condition_b <= 0.0 {
        return f64::INFINITY;
    }
    condition_a / condition_b
}

/// Log2 fold change.
pub fn log2_fold_change(condition_a: f64, condition_b: f64) -> f64 {
    if condition_a <= 0.0 || condition_b <= 0.0 {
        return 0.0;
    }
    (condition_a / condition_b).log2()
}

/// Coefficient of variation (CV) as a percentage.
pub fn cv_percent(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (var.sqrt() / mean) * 100.0
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_observations() -> Vec<PeptideObservation> {
        vec![
            PeptideObservation::new("PEPTIDE", "P001", 1000.0),
            PeptideObservation::new("ANOTHER", "P001", 2000.0),
            PeptideObservation::new("THIRD", "P001", 500.0),
            PeptideObservation::new("ALPHA", "P002", 3000.0),
            PeptideObservation::new("BETA", "P002", 1500.0),
            PeptideObservation::new("SINGLE", "P003", 800.0),
        ]
    }

    fn sample_proteins() -> Vec<ProteinEntry> {
        vec![
            ProteinEntry::new("P001", "GENE1", 300),
            ProteinEntry::new("P002", "GENE2", 150),
            ProteinEntry::new("P003", "GENE3", 500),
        ]
    }

    #[test]
    fn test_spectral_counts() {
        let obs = sample_observations();
        let counts = spectral_counts(&obs);
        let p001 = counts.iter().find(|(a, _)| a == "P001").unwrap();
        assert_eq!(p001.1, 3);
    }

    #[test]
    fn test_nsaf() {
        let obs = sample_observations();
        let counts = spectral_counts(&obs);
        let prots = sample_proteins();
        let nsaf = compute_nsaf(&counts, &prots);
        let total: f64 = nsaf.iter().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_summed_intensity() {
        let obs = sample_observations();
        let sums = summed_intensity(&obs);
        let p001 = sums.iter().find(|(a, _)| a == "P001").unwrap();
        assert!((p001.1 - 3500.0).abs() < 1e-6);
    }

    #[test]
    fn test_top_n_intensity() {
        let obs = sample_observations();
        let topn = top_n_intensity(&obs, 2);
        let p001 = topn.iter().find(|(a, _)| a == "P001").unwrap();
        // Top 2 of [1000, 2000, 500] → avg(2000, 1000) = 1500.
        assert!((p001.1 - 1500.0).abs() < 1e-6);
    }

    #[test]
    fn test_ibaq() {
        let obs = sample_observations();
        let theor = vec![
            ("P001".to_string(), 10),
            ("P002".to_string(), 5),
            ("P003".to_string(), 20),
        ];
        let ibaq = compute_ibaq(&obs, &theor);
        let p001 = ibaq.iter().find(|(a, _)| a == "P001").unwrap();
        // 3500 / 10 = 350.
        assert!((p001.1 - 350.0).abs() < 1e-6);
    }

    #[test]
    fn test_median_normalise() {
        let mut vals = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        median_normalise(&mut vals);
        // Median is 30 → all divided by 30.
        assert!((vals[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_log2_transform() {
        let mut vals = vec![0.0, 1.0, 3.0, 7.0];
        log2_transform(&mut vals);
        assert!((vals[0] - 0.0).abs() < 1e-9); // log2(1) = 0
        assert!((vals[1] - 1.0).abs() < 1e-9); // log2(2) = 1
    }

    #[test]
    fn test_fold_change() {
        assert!((fold_change(200.0, 100.0) - 2.0).abs() < 1e-9);
        assert_eq!(fold_change(100.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn test_log2_fold_change() {
        let lfc = log2_fold_change(200.0, 100.0);
        assert!((lfc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_log2_fold_change_zero() {
        assert_eq!(log2_fold_change(0.0, 100.0), 0.0);
    }

    #[test]
    fn test_cv_percent() {
        let vals = vec![10.0, 10.0, 10.0];
        assert!(cv_percent(&vals).abs() < 1e-9);
    }

    #[test]
    fn test_cv_nonzero() {
        let vals = vec![10.0, 20.0, 30.0];
        let cv = cv_percent(&vals);
        assert!(cv > 0.0);
    }

    #[test]
    fn test_protein_entry_display() {
        let p = ProteinEntry::new("P12345", "BRCA1", 1863);
        let d = format!("{}", p);
        assert!(d.contains("P12345"));
        assert!(d.contains("BRCA1"));
    }

    #[test]
    fn test_peptide_observation_display() {
        let o = PeptideObservation::new("PEPTIDE", "P001", 1000.0);
        let d = format!("{}", o);
        assert!(d.contains("PEPTIDE"));
        assert!(d.contains("P001"));
    }

    #[test]
    fn test_quant_method_display() {
        assert_eq!(format!("{}", QuantMethod::Nsaf), "NSAF");
        assert_eq!(format!("{}", QuantMethod::Ibaq), "iBAQ");
    }

    #[test]
    fn test_quant_config_display() {
        let cfg = QuantConfig::new(QuantMethod::TopN).with_top_n(3);
        let d = format!("{}", cfg);
        assert!(d.contains("Top-N"));
    }

    #[test]
    fn test_protein_quant_result_display() {
        let r = ProteinQuantResult::new("P001", "GENE1", 0.123, QuantMethod::Nsaf)
            .with_peptide_count(5)
            .with_spectral_count(12);
        let d = format!("{}", r);
        assert!(d.contains("P001"));
        assert!(d.contains("NSAF"));
    }

    #[test]
    fn test_with_molecular_weight() {
        let p = ProteinEntry::new("P001", "G1", 100).with_molecular_weight(11000.0);
        assert!((p.molecular_weight_da - 11000.0).abs() < 1e-6);
    }
}
