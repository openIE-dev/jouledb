//! Compressed Suffix Array — space-efficient text indexing with sampled
//! suffix array, Psi function, and pattern search.
//!
//! Pure-Rust compressed suffix array (CSA) that stores only every s-th
//! suffix array entry and reconstructs the rest via the Psi function.
//! Supports O(m log n) pattern search, locate queries with tunable
//! space/time trade-off, and inverse suffix array access.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CsaError {
    EmptyText,
    InvalidSampleRate(usize),
    IndexOutOfBounds { index: usize, length: usize },
    PatternNotFound,
}

impl fmt::Display for CsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText => write!(f, "text must not be empty"),
            Self::InvalidSampleRate(r) => {
                write!(f, "sample rate must be >= 1, got {r}")
            }
            Self::IndexOutOfBounds { index, length } => {
                write!(f, "index {index} out of bounds (length {length})")
            }
            Self::PatternNotFound => write!(f, "pattern not found"),
        }
    }
}

impl std::error::Error for CsaError {}

// ── Suffix array construction ───────────────────────────────────

fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    let mut sa: Vec<usize> = (0..n).collect();
    sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
    sa
}

fn build_inverse_sa(sa: &[usize]) -> Vec<usize> {
    let n = sa.len();
    let mut isa = vec![0usize; n];
    for (i, &s) in sa.iter().enumerate() {
        isa[s] = i;
    }
    isa
}

// ── Psi function ────────────────────────────────────────────────

/// Build the Psi function: psi[i] = ISA[SA[i] + 1 mod n].
fn build_psi(sa: &[usize], isa: &[usize]) -> Vec<usize> {
    let n = sa.len();
    let mut psi = vec![0usize; n];
    for i in 0..n {
        let next_pos = (sa[i] + 1) % n;
        psi[i] = isa[next_pos];
    }
    psi
}

// ── Compressed suffix array ─────────────────────────────────────

/// Compressed suffix array with sampled SA and Psi function.
#[derive(Debug, Clone)]
pub struct CompressedSuffixArray {
    /// The original text.
    text: Vec<u8>,
    /// Psi function values.
    psi: Vec<usize>,
    /// Sampled SA entries: sa_sample[i/sample_rate] = sa[i] if i % rate == 0.
    sa_samples: Vec<usize>,
    /// Sample rate.
    sample_rate: usize,
    /// C-table: cumulative character counts.
    c_table: [usize; 257],
    /// Length of text.
    length: usize,
}

impl CompressedSuffixArray {
    /// Build a CSA with default sample rate (4).
    pub fn build(text: &[u8]) -> Result<Self, CsaError> {
        CsaBuilder::new().build(text)
    }

    /// Length of the indexed text.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the CSA is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Sample rate.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Retrieve SA[i] (may require walking Psi).
    pub fn sa(&self, i: usize) -> Result<usize, CsaError> {
        if i >= self.length {
            return Err(CsaError::IndexOutOfBounds {
                index: i,
                length: self.length,
            });
        }
        let mut cur = i;
        let mut steps = 0usize;
        loop {
            if cur % self.sample_rate == 0 {
                let sample_idx = cur / self.sample_rate;
                let val = self.sa_samples[sample_idx];
                return Ok((val + self.length - steps) % self.length);
            }
            cur = self.psi[cur];
            steps += 1;
        }
    }

    /// Psi function value at position i.
    pub fn psi(&self, i: usize) -> Result<usize, CsaError> {
        if i >= self.length {
            return Err(CsaError::IndexOutOfBounds {
                index: i,
                length: self.length,
            });
        }
        Ok(self.psi[i])
    }

    /// Count occurrences of `pattern` in the text.
    pub fn count(&self, pattern: &[u8]) -> usize {
        match self.search_range(pattern) {
            Some((lo, hi)) => hi - lo,
            None => 0,
        }
    }

    /// Check if `pattern` exists in the text.
    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.search_range(pattern).is_some()
    }

    /// Locate all occurrences of `pattern`.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        let (lo, hi) = match self.search_range(pattern) {
            Some(range) => range,
            None => return Vec::new(),
        };
        let mut positions = Vec::with_capacity(hi - lo);
        for i in lo..hi {
            if let Ok(pos) = self.sa(i) {
                positions.push(pos);
            }
        }
        positions.sort();
        positions
    }

    /// Extract a substring from the text.
    pub fn extract(&self, start: usize, end: usize) -> Result<Vec<u8>, CsaError> {
        if end > self.length {
            return Err(CsaError::IndexOutOfBounds {
                index: end,
                length: self.length,
            });
        }
        if start >= end {
            return Ok(Vec::new());
        }
        Ok(self.text[start..end].to_vec())
    }

    /// Approximate memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.text.len()
            + self.psi.len() * 8
            + self.sa_samples.len() * 8
            + 257 * 8
    }

    /// Compression ratio compared to full suffix array.
    pub fn compression_ratio(&self) -> f64 {
        let full_sa_bytes = self.length * 8;
        if full_sa_bytes == 0 {
            return 1.0;
        }
        self.size_bytes() as f64 / full_sa_bytes as f64
    }

    /// Number of sampled SA entries.
    pub fn num_samples(&self) -> usize {
        self.sa_samples.len()
    }

    // ── Internal search ─────────────────────────────────────────

    fn search_range(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() {
            return Some((0, self.length));
        }
        // Binary search for lower bound.
        let lo = self.lower_bound(pattern);
        let hi = self.upper_bound(pattern);
        if lo >= hi {
            return None;
        }
        Some((lo, hi))
    }

    fn lower_bound(&self, pattern: &[u8]) -> usize {
        let mut lo = 0usize;
        let mut hi = self.length;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let sa_mid = self.sa(mid).unwrap_or(0);
            let suffix = &self.text[sa_mid..];
            let cmp_len = pattern.len().min(suffix.len());
            if suffix[..cmp_len] < *pattern {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn upper_bound(&self, pattern: &[u8]) -> usize {
        let mut lo = 0usize;
        let mut hi = self.length;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let sa_mid = self.sa(mid).unwrap_or(0);
            let suffix = &self.text[sa_mid..];
            let cmp_len = pattern.len().min(suffix.len());
            if suffix[..cmp_len] <= *pattern {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

impl fmt::Display for CompressedSuffixArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CSA(len={}, rate={}, samples={}, ratio={:.2})",
            self.length,
            self.sample_rate,
            self.num_samples(),
            self.compression_ratio()
        )
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Builder for compressed suffix arrays.
#[derive(Debug, Clone)]
pub struct CsaBuilder {
    sample_rate: usize,
    sentinel: u8,
}

impl CsaBuilder {
    pub fn new() -> Self {
        Self {
            sample_rate: 4,
            sentinel: 0x00,
        }
    }

    /// Set the suffix array sampling rate.
    pub fn with_sample_rate(mut self, rate: usize) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the sentinel character.
    pub fn with_sentinel(mut self, sentinel: u8) -> Self {
        self.sentinel = sentinel;
        self
    }

    /// Build the CSA from text.
    pub fn build(self, text: &[u8]) -> Result<CompressedSuffixArray, CsaError> {
        if text.is_empty() {
            return Err(CsaError::EmptyText);
        }
        if self.sample_rate < 1 {
            return Err(CsaError::InvalidSampleRate(self.sample_rate));
        }

        // Augment text with sentinel.
        let mut augmented = Vec::with_capacity(text.len() + 1);
        augmented.extend_from_slice(text);
        augmented.push(self.sentinel);

        let n = augmented.len();
        let sa = build_suffix_array(&augmented);
        let isa = build_inverse_sa(&sa);
        let psi = build_psi(&sa, &isa);

        // Build sampled SA.
        let num_samples = (n + self.sample_rate - 1) / self.sample_rate;
        let mut sa_samples = vec![0usize; num_samples];
        for i in 0..n {
            if i % self.sample_rate == 0 {
                sa_samples[i / self.sample_rate] = sa[i];
            }
        }

        // Build C-table.
        let mut counts = [0usize; 257];
        for &b in &augmented {
            counts[b as usize] += 1;
        }
        let mut c_table = [0usize; 257];
        let mut total = 0;
        for i in 0..257 {
            c_table[i] = total;
            total += counts[i];
        }

        Ok(CompressedSuffixArray {
            text: augmented,
            psi,
            sa_samples,
            sample_rate: self.sample_rate,
            c_table,
            length: n,
        })
    }
}

impl Default for CsaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CsaBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CsaBuilder(rate={}, sentinel=0x{:02x})",
            self.sample_rate, self.sentinel
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_basic() {
        let csa = CompressedSuffixArray::build(b"BANANA").unwrap();
        assert_eq!(csa.len(), 7); // text + sentinel
        assert!(!csa.is_empty());
    }

    #[test]
    fn test_empty_error() {
        assert!(CompressedSuffixArray::build(b"").is_err());
    }

    #[test]
    fn test_sa_access() {
        let csa = CompressedSuffixArray::build(b"ABCD").unwrap();
        // Should not panic for valid indices.
        for i in 0..csa.len() {
            assert!(csa.sa(i).is_ok());
        }
    }

    #[test]
    fn test_sa_out_of_bounds() {
        let csa = CompressedSuffixArray::build(b"ABC").unwrap();
        assert!(csa.sa(100).is_err());
    }

    #[test]
    fn test_psi_access() {
        let csa = CompressedSuffixArray::build(b"ABCD").unwrap();
        for i in 0..csa.len() {
            let p = csa.psi(i).unwrap();
            assert!(p < csa.len());
        }
    }

    #[test]
    fn test_count() {
        let csa = CompressedSuffixArray::build(b"ABCABC").unwrap();
        assert_eq!(csa.count(b"ABC"), 2);
        assert_eq!(csa.count(b"XYZ"), 0);
    }

    #[test]
    fn test_contains() {
        let csa = CompressedSuffixArray::build(b"GATTACA").unwrap();
        assert!(csa.contains(b"ATT"));
        assert!(!csa.contains(b"GGG"));
    }

    #[test]
    fn test_locate() {
        let csa = CompressedSuffixArray::build(b"ABCABC").unwrap();
        let positions = csa.locate(b"ABC");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], 0);
        assert_eq!(positions[1], 3);
    }

    #[test]
    fn test_locate_empty() {
        let csa = CompressedSuffixArray::build(b"XYZ").unwrap();
        assert!(csa.locate(b"ABC").is_empty());
    }

    #[test]
    fn test_extract() {
        let csa = CompressedSuffixArray::build(b"HELLO").unwrap();
        let sub = csa.extract(1, 4).unwrap();
        assert_eq!(sub, b"ELL");
    }

    #[test]
    fn test_extract_bounds() {
        let csa = CompressedSuffixArray::build(b"ABC").unwrap();
        assert!(csa.extract(0, 100).is_err());
    }

    #[test]
    fn test_extract_empty_range() {
        let csa = CompressedSuffixArray::build(b"ABC").unwrap();
        let sub = csa.extract(2, 2).unwrap();
        assert!(sub.is_empty());
    }

    #[test]
    fn test_size_bytes() {
        let csa = CompressedSuffixArray::build(b"ACGT").unwrap();
        assert!(csa.size_bytes() > 0);
    }

    #[test]
    fn test_compression_ratio() {
        let csa = CompressedSuffixArray::build(b"ACGTACGT").unwrap();
        let ratio = csa.compression_ratio();
        assert!(ratio > 0.0);
    }

    #[test]
    fn test_num_samples() {
        let csa = CsaBuilder::new()
            .with_sample_rate(2)
            .build(b"ABCDEFGH")
            .unwrap();
        assert!(csa.num_samples() > 0);
        assert!(csa.num_samples() <= csa.len());
    }

    #[test]
    fn test_builder_custom_rate() {
        let csa = CsaBuilder::new()
            .with_sample_rate(8)
            .build(b"ACGTACGTACGT")
            .unwrap();
        assert_eq!(csa.sample_rate(), 8);
    }

    #[test]
    fn test_builder_with_sentinel() {
        let csa = CsaBuilder::new()
            .with_sentinel(b'$')
            .build(b"ABC")
            .unwrap();
        assert!(csa.len() > 0);
    }

    #[test]
    fn test_builder_default() {
        let b = CsaBuilder::default();
        let csa = b.build(b"XYZ").unwrap();
        assert_eq!(csa.sample_rate(), 4);
    }

    #[test]
    fn test_display_csa() {
        let csa = CompressedSuffixArray::build(b"TEST").unwrap();
        let s = format!("{csa}");
        assert!(s.contains("CSA"));
    }

    #[test]
    fn test_display_builder() {
        let b = CsaBuilder::new();
        let s = format!("{b}");
        assert!(s.contains("CsaBuilder"));
    }

    #[test]
    fn test_error_display() {
        let e = CsaError::EmptyText;
        assert_eq!(format!("{e}"), "text must not be empty");
        let e2 = CsaError::InvalidSampleRate(0);
        assert!(format!("{e2}").contains("0"));
    }

    #[test]
    fn test_count_empty_pattern() {
        let csa = CompressedSuffixArray::build(b"ABC").unwrap();
        assert_eq!(csa.count(b""), csa.len());
    }
}
