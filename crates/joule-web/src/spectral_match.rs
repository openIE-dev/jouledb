//! Spectral library matching, dot product scoring, and FDR estimation.
//!
//! Compares experimental MS/MS spectra against a library of
//! reference spectra using normalised dot-product and cross-
//! correlation scoring. Includes target-decoy FDR estimation
//! and score thresholding for peptide identification.

use std::fmt;

// ── SpectralPeak ────────────────────────────────────────────────

/// A peak in a spectral library or query spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralPeak {
    pub mz: f64,
    pub intensity: f64,
}

impl SpectralPeak {
    pub fn new(mz: f64, intensity: f64) -> Self {
        Self { mz, intensity }
    }
}

impl fmt::Display for SpectralPeak {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.1})", self.mz, self.intensity)
    }
}

// ── LibrarySpectrum ─────────────────────────────────────────────

/// A reference spectrum in the spectral library.
#[derive(Debug, Clone)]
pub struct LibrarySpectrum {
    pub peptide: String,
    pub charge: u8,
    pub precursor_mz: f64,
    pub peaks: Vec<SpectralPeak>,
    pub modifications: Vec<String>,
}

impl LibrarySpectrum {
    pub fn new(peptide: &str, charge: u8, precursor_mz: f64) -> Self {
        Self {
            peptide: peptide.to_string(),
            charge,
            precursor_mz,
            peaks: Vec::new(),
            modifications: Vec::new(),
        }
    }

    pub fn with_peaks(mut self, peaks: Vec<SpectralPeak>) -> Self {
        self.peaks = peaks;
        self
    }

    pub fn with_modifications(mut self, mods: Vec<String>) -> Self {
        self.modifications = mods;
        self
    }

    /// Number of peaks in the library spectrum.
    pub fn len(&self) -> usize {
        self.peaks.len()
    }

    /// Whether the spectrum has no peaks.
    pub fn is_empty(&self) -> bool {
        self.peaks.is_empty()
    }
}

impl fmt::Display for LibrarySpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lib({} z={} mz={:.4} peaks={})",
            self.peptide, self.charge, self.precursor_mz, self.peaks.len()
        )
    }
}

// ── SpectralLibrary ─────────────────────────────────────────────

/// A collection of reference spectra for matching.
#[derive(Debug, Clone)]
pub struct SpectralLibrary {
    pub entries: Vec<LibrarySpectrum>,
}

impl SpectralLibrary {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn with_entries(mut self, entries: Vec<LibrarySpectrum>) -> Self {
        self.entries = entries;
        self
    }

    pub fn add(&mut self, entry: LibrarySpectrum) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search for candidate spectra within precursor m/z tolerance.
    pub fn candidates(&self, precursor_mz: f64, tolerance_da: f64) -> Vec<&LibrarySpectrum> {
        self.entries
            .iter()
            .filter(|e| (e.precursor_mz - precursor_mz).abs() <= tolerance_da)
            .collect()
    }
}

impl Default for SpectralLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SpectralLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpectralLibrary({} entries)", self.entries.len())
    }
}

// ── MatchConfig ─────────────────────────────────────────────────

/// Configuration for spectral matching.
#[derive(Debug, Clone)]
pub struct MatchConfig {
    pub precursor_tolerance_da: f64,
    pub fragment_tolerance_da: f64,
    pub min_matched_peaks: usize,
    pub min_score: f64,
    pub top_n_peaks: usize,
    pub sqrt_transform: bool,
}

impl MatchConfig {
    pub fn new() -> Self {
        Self {
            precursor_tolerance_da: 0.5,
            fragment_tolerance_da: 0.02,
            min_matched_peaks: 3,
            min_score: 0.5,
            top_n_peaks: 50,
            sqrt_transform: true,
        }
    }

    pub fn with_precursor_tolerance(mut self, tol: f64) -> Self {
        self.precursor_tolerance_da = tol;
        self
    }

    pub fn with_fragment_tolerance(mut self, tol: f64) -> Self {
        self.fragment_tolerance_da = tol;
        self
    }

    pub fn with_min_matched_peaks(mut self, n: usize) -> Self {
        self.min_matched_peaks = n;
        self
    }

    pub fn with_min_score(mut self, s: f64) -> Self {
        self.min_score = s;
        self
    }

    pub fn with_top_n_peaks(mut self, n: usize) -> Self {
        self.top_n_peaks = n;
        self
    }

    pub fn with_sqrt_transform(mut self, yes: bool) -> Self {
        self.sqrt_transform = yes;
        self
    }
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── MatchResult ─────────────────────────────────────────────────

/// Result of a spectral library search for a single query.
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub peptide: String,
    pub charge: u8,
    pub dot_product: f64,
    pub matched_peaks: usize,
    pub total_query_peaks: usize,
    pub delta_score: f64,
    pub is_decoy: bool,
}

impl MatchResult {
    pub fn new(
        peptide: &str,
        charge: u8,
        dot_product: f64,
        matched_peaks: usize,
        total_query_peaks: usize,
    ) -> Self {
        Self {
            peptide: peptide.to_string(),
            charge,
            dot_product,
            matched_peaks,
            total_query_peaks,
            delta_score: 0.0,
            is_decoy: false,
        }
    }

    pub fn with_delta_score(mut self, ds: f64) -> Self {
        self.delta_score = ds;
        self
    }

    pub fn with_decoy(mut self, yes: bool) -> Self {
        self.is_decoy = yes;
        self
    }

    /// Fraction of query peaks that were matched.
    pub fn matched_fraction(&self) -> f64 {
        if self.total_query_peaks == 0 {
            return 0.0;
        }
        self.matched_peaks as f64 / self.total_query_peaks as f64
    }
}

impl fmt::Display for MatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} z={} dp={:.3} matched={}/{} delta={:.3}{}",
            self.peptide,
            self.charge,
            self.dot_product,
            self.matched_peaks,
            self.total_query_peaks,
            self.delta_score,
            if self.is_decoy { " [DECOY]" } else { "" },
        )
    }
}

// ── Scoring functions ───────────────────────────────────────────

/// Prepare peak vectors: keep top-N by intensity and optionally
/// apply sqrt transform.
fn prepare_peaks(peaks: &[SpectralPeak], top_n: usize, sqrt_transform: bool) -> Vec<SpectralPeak> {
    let mut sorted = peaks.to_vec();
    sorted.sort_by(|a, b| b.intensity.partial_cmp(&a.intensity).unwrap());
    sorted.truncate(top_n);
    sorted.sort_by(|a, b| a.mz.partial_cmp(&b.mz).unwrap());

    if sqrt_transform {
        sorted
            .iter()
            .map(|p| SpectralPeak::new(p.mz, p.intensity.sqrt()))
            .collect()
    } else {
        sorted
    }
}

/// Normalised dot product between two peak lists.
///
/// Peaks are paired by m/z within `tolerance_da`. Returns cosine
/// similarity and number of matched peaks.
pub fn dot_product_score(
    query: &[SpectralPeak],
    reference: &[SpectralPeak],
    tolerance_da: f64,
    sqrt_transform: bool,
    top_n: usize,
) -> (f64, usize) {
    let q = prepare_peaks(query, top_n, sqrt_transform);
    let r = prepare_peaks(reference, top_n, sqrt_transform);

    if q.is_empty() || r.is_empty() {
        return (0.0, 0);
    }

    let mut dot = 0.0_f64;
    let mut mag_q = 0.0_f64;
    let mut mag_r = 0.0_f64;
    let mut matched = 0usize;
    let mut used_r = vec![false; r.len()];

    for qp in &q {
        mag_q += qp.intensity * qp.intensity;
        let mut best_j = None;
        let mut best_err = f64::INFINITY;
        for (j, rp) in r.iter().enumerate() {
            if used_r[j] {
                continue;
            }
            let err = (qp.mz - rp.mz).abs();
            if err <= tolerance_da && err < best_err {
                best_err = err;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            dot += qp.intensity * r[j].intensity;
            used_r[j] = true;
            matched += 1;
        }
    }

    for rp in &r {
        mag_r += rp.intensity * rp.intensity;
    }

    let denom = mag_q.sqrt() * mag_r.sqrt();
    let score = if denom > 0.0 { dot / denom } else { 0.0 };
    (score, matched)
}

/// Cross-correlation score (shift-based).
///
/// Computes a simplified cross-correlation by binning spectra and
/// computing the zero-lag normalised correlation.
pub fn cross_correlation_score(
    query: &[SpectralPeak],
    reference: &[SpectralPeak],
    bin_width: f64,
    mz_range: (f64, f64),
) -> f64 {
    let n_bins = ((mz_range.1 - mz_range.0) / bin_width).ceil() as usize + 1;
    let mut q_bins = vec![0.0_f64; n_bins];
    let mut r_bins = vec![0.0_f64; n_bins];

    for p in query {
        let idx = ((p.mz - mz_range.0) / bin_width).floor() as usize;
        if idx < n_bins {
            q_bins[idx] += p.intensity;
        }
    }
    for p in reference {
        let idx = ((p.mz - mz_range.0) / bin_width).floor() as usize;
        if idx < n_bins {
            r_bins[idx] += p.intensity;
        }
    }

    let dot: f64 = q_bins.iter().zip(r_bins.iter()).map(|(a, b)| a * b).sum();
    let mag_q: f64 = q_bins.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_r: f64 = r_bins.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_q > 0.0 && mag_r > 0.0 {
        dot / (mag_q * mag_r)
    } else {
        0.0
    }
}

// ── Library search ──────────────────────────────────────────────

/// Search a spectral library for the best match to a query spectrum.
pub fn search_library(
    query_peaks: &[SpectralPeak],
    query_precursor_mz: f64,
    library: &SpectralLibrary,
    config: &MatchConfig,
) -> Vec<MatchResult> {
    if query_peaks.is_empty() {
        return Vec::new();
    }
    let candidates = library.candidates(query_precursor_mz, config.precursor_tolerance_da);
    let mut results: Vec<MatchResult> = candidates
        .iter()
        .filter_map(|lib_spec| {
            let (score, matched) = dot_product_score(
                query_peaks,
                &lib_spec.peaks,
                config.fragment_tolerance_da,
                config.sqrt_transform,
                config.top_n_peaks,
            );
            if matched >= config.min_matched_peaks && score >= config.min_score {
                Some(MatchResult::new(
                    &lib_spec.peptide,
                    lib_spec.charge,
                    score,
                    matched,
                    query_peaks.len(),
                ))
            } else {
                None
            }
        })
        .collect();

    results.sort_by(|a, b| b.dot_product.partial_cmp(&a.dot_product).unwrap());

    // Compute delta scores.
    if results.len() >= 2 {
        let first = results[0].dot_product;
        let second = results[1].dot_product;
        results[0].delta_score = first - second;
    }

    results
}

// ── FDR estimation ──────────────────────────────────────────────

/// Estimate FDR using target-decoy approach.
///
/// Returns (score_threshold, fdr_at_threshold) pairs sorted by
/// descending score threshold.
pub fn estimate_fdr(results: &[MatchResult], fdr_cutoff: f64) -> Vec<(f64, f64)> {
    let mut sorted: Vec<&MatchResult> = results.iter().collect();
    sorted.sort_by(|a, b| b.dot_product.partial_cmp(&a.dot_product).unwrap());

    let mut fdr_table = Vec::new();
    let mut targets = 0usize;
    let mut decoys = 0usize;

    for r in &sorted {
        if r.is_decoy {
            decoys += 1;
        } else {
            targets += 1;
        }
        let fdr = if targets > 0 {
            decoys as f64 / targets as f64
        } else {
            1.0
        };
        fdr_table.push((r.dot_product, fdr));
    }

    // Filter to entries at or below fdr_cutoff.
    fdr_table.retain(|&(_, fdr)| fdr <= fdr_cutoff);
    fdr_table
}

/// Find the score threshold for a given FDR level.
pub fn fdr_threshold(results: &[MatchResult], target_fdr: f64) -> Option<f64> {
    let table = estimate_fdr(results, target_fdr);
    table.last().map(|&(score, _)| score)
}

/// Count identifications at a given FDR level.
pub fn count_ids_at_fdr(results: &[MatchResult], target_fdr: f64) -> usize {
    if let Some(threshold) = fdr_threshold(results, target_fdr) {
        results
            .iter()
            .filter(|r| !r.is_decoy && r.dot_product >= threshold)
            .count()
    } else {
        0
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peaks(pairs: &[(f64, f64)]) -> Vec<SpectralPeak> {
        pairs.iter().map(|&(mz, i)| SpectralPeak::new(mz, i)).collect()
    }

    fn sample_library() -> SpectralLibrary {
        let lib_peaks = make_peaks(&[
            (200.1, 100.0), (300.2, 500.0), (400.3, 800.0),
            (500.4, 300.0), (600.5, 200.0),
        ]);
        let entry = LibrarySpectrum::new("PEPTIDE", 2, 400.22).with_peaks(lib_peaks);
        SpectralLibrary::new().with_entries(vec![entry])
    }

    #[test]
    fn test_dot_product_identical() {
        let peaks = make_peaks(&[(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]);
        let (score, matched) = dot_product_score(&peaks, &peaks, 0.02, false, 50);
        assert!((score - 1.0).abs() < 1e-6);
        assert_eq!(matched, 3);
    }

    #[test]
    fn test_dot_product_no_match() {
        let q = make_peaks(&[(100.0, 1.0)]);
        let r = make_peaks(&[(500.0, 1.0)]);
        let (score, matched) = dot_product_score(&q, &r, 0.02, false, 50);
        assert_eq!(score, 0.0);
        assert_eq!(matched, 0);
    }

    #[test]
    fn test_dot_product_partial_match() {
        let q = make_peaks(&[(100.0, 1.0), (200.0, 2.0), (999.0, 1.0)]);
        let r = make_peaks(&[(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]);
        let (score, matched) = dot_product_score(&q, &r, 0.02, false, 50);
        assert!(score > 0.0 && score < 1.0);
        assert_eq!(matched, 2);
    }

    #[test]
    fn test_dot_product_with_sqrt() {
        let peaks = make_peaks(&[(100.0, 100.0), (200.0, 400.0)]);
        let (score, _) = dot_product_score(&peaks, &peaks, 0.02, true, 50);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_correlation_identical() {
        let peaks = make_peaks(&[(200.0, 1.0), (400.0, 2.0)]);
        let score = cross_correlation_score(&peaks, &peaks, 1.0, (100.0, 600.0));
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_library_candidates() {
        let lib = sample_library();
        let cands = lib.candidates(400.22, 0.5);
        assert_eq!(cands.len(), 1);
        let cands_none = lib.candidates(800.0, 0.5);
        assert!(cands_none.is_empty());
    }

    #[test]
    fn test_search_library() {
        let lib = sample_library();
        let query = make_peaks(&[
            (200.1, 110.0), (300.2, 480.0), (400.3, 790.0),
            (500.4, 310.0), (600.5, 190.0),
        ]);
        let cfg = MatchConfig::new()
            .with_precursor_tolerance(0.5)
            .with_fragment_tolerance(0.05)
            .with_min_matched_peaks(3)
            .with_min_score(0.5);
        let results = search_library(&query, 400.22, &lib, &cfg);
        assert!(!results.is_empty());
        assert!(results[0].dot_product > 0.9);
    }

    #[test]
    fn test_match_result_display() {
        let mr = MatchResult::new("PEPTIDE", 2, 0.95, 5, 8).with_delta_score(0.15);
        let d = format!("{}", mr);
        assert!(d.contains("PEPTIDE"));
        assert!(d.contains("0.95"));
    }

    #[test]
    fn test_decoy_display() {
        let mr = MatchResult::new("EDITPEP", 2, 0.6, 3, 8).with_decoy(true);
        let d = format!("{}", mr);
        assert!(d.contains("[DECOY]"));
    }

    #[test]
    fn test_matched_fraction() {
        let mr = MatchResult::new("PEPTIDE", 2, 0.9, 4, 8);
        assert!((mr.matched_fraction() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_fdr_estimation() {
        let results = vec![
            MatchResult::new("PEP1", 2, 0.95, 5, 8),
            MatchResult::new("PEP2", 2, 0.90, 5, 8),
            MatchResult::new("DEC1", 2, 0.85, 4, 8).with_decoy(true),
            MatchResult::new("PEP3", 2, 0.80, 4, 8),
        ];
        let table = estimate_fdr(&results, 0.5);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_fdr_threshold() {
        let results = vec![
            MatchResult::new("PEP1", 2, 0.95, 5, 8),
            MatchResult::new("PEP2", 2, 0.90, 5, 8),
            MatchResult::new("DEC1", 2, 0.70, 4, 8).with_decoy(true),
        ];
        let thr = fdr_threshold(&results, 1.0);
        assert!(thr.is_some());
    }

    #[test]
    fn test_count_ids() {
        let results = vec![
            MatchResult::new("PEP1", 2, 0.95, 5, 8),
            MatchResult::new("PEP2", 2, 0.90, 5, 8),
            MatchResult::new("DEC1", 2, 0.50, 4, 8).with_decoy(true),
        ];
        let n = count_ids_at_fdr(&results, 1.0);
        assert!(n >= 1);
    }

    #[test]
    fn test_library_display() {
        let lib = sample_library();
        let d = format!("{}", lib);
        assert!(d.contains("1 entries"));
    }

    #[test]
    fn test_spectral_peak_display() {
        let p = SpectralPeak::new(500.123, 999.0);
        let d = format!("{}", p);
        assert!(d.contains("500.123"));
    }

    #[test]
    fn test_library_spectrum_display() {
        let ls = LibrarySpectrum::new("PEPTIDE", 2, 400.0);
        let d = format!("{}", ls);
        assert!(d.contains("PEPTIDE"));
    }

    #[test]
    fn test_empty_query() {
        let lib = sample_library();
        let cfg = MatchConfig::new().with_min_score(0.0).with_min_matched_peaks(0);
        let results = search_library(&[], 400.22, &lib, &cfg);
        assert!(results.is_empty());
    }

    #[test]
    fn test_with_modifications() {
        let ls = LibrarySpectrum::new("PEPTIDE", 2, 400.0)
            .with_modifications(vec!["Oxidation(M)".to_string()]);
        assert_eq!(ls.modifications.len(), 1);
    }
}
