//! K-mer Frequency Counting — k-mer enumeration, canonical k-mers,
//! minimizer selection, frequency spectra, Jaccard similarity.
//!
//! Pure-Rust k-mer analysis engine for genomic sequences with support
//! for canonical (strand-agnostic) k-mers, minimizer sketching, and
//! frequency-based similarity measures.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum KmerError {
    EmptySequence,
    InvalidK(usize),
    InvalidWindowSize(String),
    InvalidBase(u8),
}

impl fmt::Display for KmerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "empty sequence"),
            Self::InvalidK(k) => write!(f, "invalid k: {k}"),
            Self::InvalidWindowSize(s) => write!(f, "invalid window size: {s}"),
            Self::InvalidBase(b) => write!(f, "invalid base: {}", *b as char),
        }
    }
}

impl std::error::Error for KmerError {}

// ── Complement / Canonical ──────────────────────────────────────

/// DNA complement of a single base.
fn complement(base: u8) -> u8 {
    match base.to_ascii_uppercase() {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        other => other,
    }
}

/// Reverse complement of a k-mer.
fn reverse_complement(kmer: &[u8]) -> Vec<u8> {
    kmer.iter().rev().map(|b| complement(*b)).collect()
}

/// Canonical k-mer: lexicographically smaller of forward and reverse complement.
pub fn canonical_kmer(kmer: &[u8]) -> Vec<u8> {
    let upper: Vec<u8> = kmer.iter().map(|b| b.to_ascii_uppercase()).collect();
    let rc = reverse_complement(&upper);
    if upper <= rc { upper } else { rc }
}

// ── K-mer Counter ───────────────────────────────────────────────

/// Counts k-mer occurrences in a sequence.
#[derive(Debug, Clone)]
pub struct KmerCounter {
    k: usize,
    use_canonical: bool,
    min_count: usize,
}

impl KmerCounter {
    pub fn new(k: usize) -> Self {
        Self { k, use_canonical: false, min_count: 1 }
    }

    pub fn with_canonical(mut self, c: bool) -> Self { self.use_canonical = c; self }
    pub fn with_min_count(mut self, m: usize) -> Self { self.min_count = m; self }
    pub fn k(&self) -> usize { self.k }

    /// Count k-mers in the given sequence.
    pub fn count(&self, seq: &[u8]) -> Result<KmerSpectrum, KmerError> {
        if seq.is_empty() {
            return Err(KmerError::EmptySequence);
        }
        if self.k == 0 || self.k > seq.len() {
            return Err(KmerError::InvalidK(self.k));
        }

        let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for i in 0..=seq.len() - self.k {
            let kmer = &seq[i..i + self.k];
            let key = if self.use_canonical {
                canonical_kmer(kmer)
            } else {
                kmer.iter().map(|b| b.to_ascii_uppercase()).collect()
            };
            *counts.entry(key).or_insert(0) += 1;
        }

        if self.min_count > 1 {
            counts.retain(|_, v| *v >= self.min_count);
        }

        let total: usize = counts.values().sum();
        Ok(KmerSpectrum {
            k: self.k,
            counts,
            total_kmers: total,
            canonical: self.use_canonical,
        })
    }

    /// Count k-mers across multiple sequences.
    pub fn count_multi(&self, seqs: &[&[u8]]) -> Result<KmerSpectrum, KmerError> {
        let mut combined: HashMap<Vec<u8>, usize> = HashMap::new();
        for seq in seqs {
            let spectrum = self.count(seq)?;
            for (kmer, cnt) in spectrum.counts {
                *combined.entry(kmer).or_insert(0) += cnt;
            }
        }
        let total: usize = combined.values().sum();
        Ok(KmerSpectrum {
            k: self.k,
            counts: combined,
            total_kmers: total,
            canonical: self.use_canonical,
        })
    }
}

impl fmt::Display for KmerCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KmerCounter(k={}, canonical={}, min_count={})",
            self.k, self.use_canonical, self.min_count
        )
    }
}

// ── K-mer Spectrum ──────────────────────────────────────────────

/// Frequency distribution of k-mers.
#[derive(Debug, Clone)]
pub struct KmerSpectrum {
    pub k: usize,
    pub counts: HashMap<Vec<u8>, usize>,
    pub total_kmers: usize,
    pub canonical: bool,
}

impl KmerSpectrum {
    /// Number of distinct k-mers.
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// Frequency of a specific k-mer.
    pub fn frequency(&self, kmer: &[u8]) -> usize {
        let key: Vec<u8> = kmer.iter().map(|b| b.to_ascii_uppercase()).collect();
        self.counts.get(&key).copied().unwrap_or(0)
    }

    /// Top-n most frequent k-mers.
    pub fn top_n(&self, n: usize) -> Vec<(Vec<u8>, usize)> {
        let mut sorted: Vec<_> = self.counts.iter().map(|(k, &v)| (k.clone(), v)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }

    /// Frequency histogram: count -> number of k-mers with that count.
    pub fn histogram(&self) -> HashMap<usize, usize> {
        let mut hist: HashMap<usize, usize> = HashMap::new();
        for &cnt in self.counts.values() {
            *hist.entry(cnt).or_insert(0) += 1;
        }
        hist
    }

    /// Shannon entropy of the k-mer distribution.
    pub fn entropy(&self) -> f64 {
        if self.total_kmers == 0 {
            return 0.0;
        }
        let n = self.total_kmers as f64;
        self.counts.values().fold(0.0, |acc, &c| {
            if c == 0 { return acc; }
            let p = c as f64 / n;
            acc - p * p.ln()
        })
    }

    /// Jaccard similarity between two spectra (set-based, ignoring counts).
    pub fn jaccard(&self, other: &KmerSpectrum) -> f64 {
        let set_a: std::collections::HashSet<_> = self.counts.keys().collect();
        let set_b: std::collections::HashSet<_> = other.counts.keys().collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
    }

    /// Cosine similarity between two spectra (count-based).
    pub fn cosine_similarity(&self, other: &KmerSpectrum) -> f64 {
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;

        for (kmer, &ca) in &self.counts {
            let cb = other.counts.get(kmer).copied().unwrap_or(0) as f64;
            let ca_f = ca as f64;
            dot += ca_f * cb;
            norm_a += ca_f * ca_f;
        }
        for &cb in other.counts.values() {
            norm_b += (cb as f64) * (cb as f64);
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }
}

impl fmt::Display for KmerSpectrum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KmerSpectrum(k={}, distinct={}, total={}, canonical={})",
            self.k, self.distinct(), self.total_kmers, self.canonical
        )
    }
}

// ── Minimizer Selector ──────────────────────────────────────────

/// Selects minimizers from a sequence for sketching / indexing.
#[derive(Debug, Clone)]
pub struct MinimizerSelector {
    k: usize,
    window: usize,
    use_canonical: bool,
}

impl MinimizerSelector {
    pub fn new(k: usize, window: usize) -> Self {
        Self { k, window, use_canonical: true }
    }

    pub fn with_canonical(mut self, c: bool) -> Self { self.use_canonical = c; self }

    /// Select minimizers: (position, k-mer) for each window.
    pub fn select(&self, seq: &[u8]) -> Result<Vec<(usize, Vec<u8>)>, KmerError> {
        if seq.is_empty() {
            return Err(KmerError::EmptySequence);
        }
        if self.k == 0 || self.k > seq.len() {
            return Err(KmerError::InvalidK(self.k));
        }
        if self.window == 0 || self.window + self.k - 1 > seq.len() {
            return Err(KmerError::InvalidWindowSize(format!(
                "window={}, k={}, seq_len={}",
                self.window, self.k, seq.len()
            )));
        }

        let num_kmers = seq.len() - self.k + 1;
        let kmers: Vec<Vec<u8>> = (0..num_kmers)
            .map(|i| {
                let raw = &seq[i..i + self.k];
                if self.use_canonical {
                    canonical_kmer(raw)
                } else {
                    raw.iter().map(|b| b.to_ascii_uppercase()).collect()
                }
            })
            .collect();

        let mut minimizers = Vec::new();
        let mut prev_min: Option<usize> = None;

        for w_start in 0..=num_kmers.saturating_sub(self.window) {
            let w_end = (w_start + self.window).min(num_kmers);
            let min_pos = (w_start..w_end)
                .min_by(|&a, &b| kmers[a].cmp(&kmers[b]))
                .unwrap();

            if prev_min != Some(min_pos) {
                minimizers.push((min_pos, kmers[min_pos].clone()));
                prev_min = Some(min_pos);
            }
        }

        Ok(minimizers)
    }

    /// Density: minimizers selected / total k-mers.
    pub fn density(&self, seq: &[u8]) -> Result<f64, KmerError> {
        let mins = self.select(seq)?;
        let total_kmers = seq.len() - self.k + 1;
        Ok(mins.len() as f64 / total_kmers as f64)
    }
}

impl fmt::Display for MinimizerSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MinimizerSelector(k={}, w={}, canonical={})", self.k, self.window, self.use_canonical)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_count() {
        let counter = KmerCounter::new(2);
        let sp = counter.count(b"ACGT").unwrap();
        assert_eq!(sp.distinct(), 3); // AC, CG, GT
        assert_eq!(sp.total_kmers, 3);
    }

    #[test]
    fn canonical_kmers() {
        let counter = KmerCounter::new(2).with_canonical(true);
        let sp = counter.count(b"ACGT").unwrap();
        // AC and GT are reverse complements, so canonical reduces count
        assert!(sp.distinct() <= 3);
    }

    #[test]
    fn single_base_kmer() {
        let counter = KmerCounter::new(1);
        let sp = counter.count(b"AAAA").unwrap();
        assert_eq!(sp.distinct(), 1);
        assert_eq!(sp.frequency(b"A"), 4);
    }

    #[test]
    fn empty_seq_err() {
        let counter = KmerCounter::new(3);
        assert!(counter.count(b"").is_err());
    }

    #[test]
    fn k_too_large() {
        let counter = KmerCounter::new(10);
        assert!(counter.count(b"ACG").is_err());
    }

    #[test]
    fn min_count_filter() {
        let counter = KmerCounter::new(2).with_min_count(2);
        let sp = counter.count(b"ACACAC").unwrap();
        for &cnt in sp.counts.values() {
            assert!(cnt >= 2);
        }
    }

    #[test]
    fn top_n_sorted() {
        let counter = KmerCounter::new(1);
        let sp = counter.count(b"AAACGT").unwrap();
        let top = sp.top_n(1);
        assert_eq!(top[0].0, b"A".to_vec());
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn histogram_test() {
        let counter = KmerCounter::new(1);
        let sp = counter.count(b"AACG").unwrap();
        let hist = sp.histogram();
        assert!(hist.contains_key(&2)); // A appears twice
        assert!(hist.contains_key(&1)); // C and G once each
    }

    #[test]
    fn entropy_uniform() {
        let counter = KmerCounter::new(1);
        let sp = counter.count(b"ACGT").unwrap();
        assert!(sp.entropy() > 0.0);
    }

    #[test]
    fn jaccard_identical() {
        let counter = KmerCounter::new(2);
        let sp = counter.count(b"ACGT").unwrap();
        assert!((sp.jaccard(&sp) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn jaccard_disjoint() {
        let counter = KmerCounter::new(2);
        let sp1 = counter.count(b"AAAA").unwrap();
        let sp2 = counter.count(b"CCCC").unwrap();
        assert!((sp1.jaccard(&sp2)).abs() < 1e-9);
    }

    #[test]
    fn cosine_identical() {
        let counter = KmerCounter::new(2);
        let sp = counter.count(b"ACGT").unwrap();
        assert!((sp.cosine_similarity(&sp) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn multi_sequence_count() {
        let counter = KmerCounter::new(2);
        let sp = counter.count_multi(&[b"ACGT", b"ACGT"]).unwrap();
        assert_eq!(sp.frequency(b"AC"), 2);
    }

    #[test]
    fn canonical_kmer_symmetric() {
        assert_eq!(canonical_kmer(b"AC"), canonical_kmer(b"GT"));
    }

    #[test]
    fn reverse_complement_test() {
        assert_eq!(reverse_complement(b"ACGT"), b"ACGT");
        assert_eq!(reverse_complement(b"AAAC"), b"GTTT");
    }

    #[test]
    fn minimizer_basic() {
        let sel = MinimizerSelector::new(3, 3);
        let mins = sel.select(b"ACGTACGTACGT").unwrap();
        assert!(!mins.is_empty());
    }

    #[test]
    fn minimizer_density() {
        let sel = MinimizerSelector::new(3, 3);
        let d = sel.density(b"ACGTACGTACGT").unwrap();
        assert!(d > 0.0 && d <= 1.0);
    }

    #[test]
    fn spectrum_display() {
        let counter = KmerCounter::new(2);
        let sp = counter.count(b"ACGT").unwrap();
        assert!(format!("{sp}").contains("KmerSpectrum"));
    }

    #[test]
    fn counter_display() {
        let counter = KmerCounter::new(5).with_canonical(true);
        assert!(format!("{counter}").contains("k=5"));
    }

    #[test]
    fn minimizer_display() {
        let sel = MinimizerSelector::new(3, 3);
        assert!(format!("{sel}").contains("MinimizerSelector"));
    }
}
