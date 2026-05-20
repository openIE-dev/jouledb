//! Burrows–Wheeler Transform — BWT construction via suffix array, inverse
//! BWT reconstruction, BWT-based exact pattern search with LF-mapping,
//! and run-length encoding of BWT output.
//!
//! Pure-Rust implementation of the BWT for biological sequence compression
//! and search. Constructs the BWT from a suffix array, supports inverse
//! transform for lossless recovery, and provides backward search for
//! O(m) pattern matching where m is pattern length.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BwtError {
    EmptyInput,
    InvalidSentinel(String),
    ReconstructionFailed(String),
}

impl fmt::Display for BwtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input must not be empty"),
            Self::InvalidSentinel(s) => write!(f, "invalid sentinel: {s}"),
            Self::ReconstructionFailed(s) => {
                write!(f, "reconstruction failed: {s}")
            }
        }
    }
}

impl std::error::Error for BwtError {}

// ── Suffix array construction ───────────────────────────────────

/// Build a suffix array for `text` using the naive O(n log^2 n) method.
fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    let mut sa: Vec<usize> = (0..n).collect();
    sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
    sa
}

// ── BWT result ──────────────────────────────────────────────────

/// The result of a Burrows–Wheeler Transform.
#[derive(Debug, Clone, PartialEq)]
pub struct BwtResult {
    /// The BWT-transformed string.
    pub bwt: Vec<u8>,
    /// Position of the original string's first character in the BWT.
    pub primary_index: usize,
    /// The suffix array used during construction.
    pub suffix_array: Vec<usize>,
}

impl fmt::Display for BwtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BWT(len={}, primary={})",
            self.bwt.len(),
            self.primary_index
        )
    }
}

// ── BWT engine ──────────────────────────────────────────────────

/// Burrows–Wheeler Transform engine with configurable sentinel.
#[derive(Debug, Clone)]
pub struct BurrowsWheeler {
    sentinel: u8,
}

impl BurrowsWheeler {
    /// Create a new BWT engine with default sentinel `$` (0x00).
    pub fn new() -> Self {
        Self { sentinel: 0x00 }
    }

    /// Set the sentinel character (must be lexicographically smallest).
    pub fn with_sentinel(mut self, sentinel: u8) -> Self {
        self.sentinel = sentinel;
        self
    }

    /// Construct the BWT of `text`.
    pub fn transform(&self, text: &[u8]) -> Result<BwtResult, BwtError> {
        if text.is_empty() {
            return Err(BwtError::EmptyInput);
        }
        // Append sentinel.
        let mut augmented = Vec::with_capacity(text.len() + 1);
        augmented.extend_from_slice(text);
        augmented.push(self.sentinel);

        let sa = build_suffix_array(&augmented);
        let n = augmented.len();
        let mut bwt = Vec::with_capacity(n);
        let mut primary_index = 0;

        for (i, &s) in sa.iter().enumerate() {
            if s == 0 {
                bwt.push(augmented[n - 1]);
                primary_index = i;
            } else {
                bwt.push(augmented[s - 1]);
            }
        }

        Ok(BwtResult {
            bwt,
            primary_index,
            suffix_array: sa,
        })
    }

    /// Inverse BWT: reconstruct original text from BWT output.
    pub fn inverse(&self, bwt: &[u8], primary_index: usize) -> Result<Vec<u8>, BwtError> {
        if bwt.is_empty() {
            return Err(BwtError::EmptyInput);
        }
        if primary_index >= bwt.len() {
            return Err(BwtError::ReconstructionFailed(
                "primary index out of bounds".into(),
            ));
        }

        let n = bwt.len();

        // Build LF-mapping via counting sort.
        let mut counts = [0usize; 256];
        for &b in bwt {
            counts[b as usize] += 1;
        }
        let mut cumulative = [0usize; 256];
        let mut total = 0;
        for i in 0..256 {
            cumulative[i] = total;
            total += counts[i];
        }

        let mut lf = vec![0usize; n];
        let mut occ = [0usize; 256];
        for (i, &b) in bwt.iter().enumerate() {
            lf[i] = cumulative[b as usize] + occ[b as usize];
            occ[b as usize] += 1;
        }

        // Walk the LF-mapping from primary_index.
        let mut result = Vec::with_capacity(n - 1);
        let mut idx = primary_index;
        for _ in 0..n - 1 {
            idx = lf[idx];
            result.push(bwt[idx]);
        }

        // The LF-mapping walk produces characters in reverse order.
        result.reverse();

        // Remove sentinel if present at end.
        if let Some(&last) = result.last() {
            if last == self.sentinel {
                result.pop();
            }
        }

        Ok(result)
    }

    /// Backward search: count occurrences of `pattern` in original text.
    pub fn backward_search(
        &self,
        bwt_result: &BwtResult,
        pattern: &[u8],
    ) -> usize {
        if pattern.is_empty() {
            return bwt_result.bwt.len();
        }

        let bwt = &bwt_result.bwt;
        let n = bwt.len();

        // Precompute C table (cumulative counts).
        let mut counts = [0usize; 256];
        for &b in bwt {
            counts[b as usize] += 1;
        }
        let mut c_table = [0usize; 256];
        let mut total = 0;
        for i in 0..256 {
            c_table[i] = total;
            total += counts[i];
        }

        // Precompute occurrence table (Occ).
        // occ[i][c] = number of occurrences of c in bwt[0..i].
        let mut occ = vec![[0usize; 256]; n + 1];
        for i in 0..n {
            occ[i + 1] = occ[i];
            occ[i + 1][bwt[i] as usize] += 1;
        }

        let mut top = 0usize;
        let mut bottom = n;

        for &c in pattern.iter().rev() {
            let ci = c as usize;
            top = c_table[ci] + occ[top][ci];
            bottom = c_table[ci] + occ[bottom][ci];
            if top >= bottom {
                return 0;
            }
        }

        bottom - top
    }

    /// Find all positions where `pattern` occurs.
    pub fn locate_pattern(
        &self,
        bwt_result: &BwtResult,
        pattern: &[u8],
    ) -> Vec<usize> {
        if pattern.is_empty() {
            return Vec::new();
        }

        let bwt = &bwt_result.bwt;
        let n = bwt.len();

        let mut counts = [0usize; 256];
        for &b in bwt {
            counts[b as usize] += 1;
        }
        let mut c_table = [0usize; 256];
        let mut total = 0;
        for i in 0..256 {
            c_table[i] = total;
            total += counts[i];
        }

        let mut occ = vec![[0usize; 256]; n + 1];
        for i in 0..n {
            occ[i + 1] = occ[i];
            occ[i + 1][bwt[i] as usize] += 1;
        }

        let mut top = 0usize;
        let mut bottom = n;

        for &c in pattern.iter().rev() {
            let ci = c as usize;
            top = c_table[ci] + occ[top][ci];
            bottom = c_table[ci] + occ[bottom][ci];
            if top >= bottom {
                return Vec::new();
            }
        }

        let mut positions: Vec<usize> = (top..bottom)
            .map(|i| bwt_result.suffix_array[i])
            .collect();
        positions.sort();
        positions
    }
}

impl Default for BurrowsWheeler {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BurrowsWheeler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BurrowsWheeler(sentinel=0x{:02x})", self.sentinel)
    }
}

// ── Run-length encoding ─────────────────────────────────────────

/// Run-length encode a BWT output.
pub fn rle_encode(data: &[u8]) -> Vec<(u8, u32)> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut current = data[0];
    let mut count = 1u32;
    for &b in &data[1..] {
        if b == current {
            count += 1;
        } else {
            runs.push((current, count));
            current = b;
            count = 1;
        }
    }
    runs.push((current, count));
    runs
}

/// Decode a run-length encoded sequence.
pub fn rle_decode(runs: &[(u8, u32)]) -> Vec<u8> {
    let mut output = Vec::new();
    for &(byte, count) in runs {
        for _ in 0..count {
            output.push(byte);
        }
    }
    output
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_basic() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"BANANA").unwrap();
        assert_eq!(result.bwt.len(), 7); // includes sentinel
    }

    #[test]
    fn test_empty_input_error() {
        let bwt = BurrowsWheeler::new();
        assert!(bwt.transform(b"").is_err());
    }

    #[test]
    fn test_inverse_roundtrip() {
        let bwt = BurrowsWheeler::new();
        let original = b"ABRACADABRA";
        let result = bwt.transform(original).unwrap();
        let recovered = bwt.inverse(&result.bwt, result.primary_index).unwrap();
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_inverse_single_char() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"A").unwrap();
        let recovered = bwt.inverse(&result.bwt, result.primary_index).unwrap();
        assert_eq!(recovered, b"A");
    }

    #[test]
    fn test_backward_search_count() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"ABCABC").unwrap();
        assert_eq!(bwt.backward_search(&result, b"ABC"), 2);
        assert_eq!(bwt.backward_search(&result, b"XYZ"), 0);
    }

    #[test]
    fn test_backward_search_single() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"AABAA").unwrap();
        let count = bwt.backward_search(&result, b"A");
        assert_eq!(count, 4);
    }

    #[test]
    fn test_locate_pattern() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"ABCABC").unwrap();
        let positions = bwt.locate_pattern(&result, b"ABC");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], 0);
        assert_eq!(positions[1], 3);
    }

    #[test]
    fn test_locate_no_match() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"HELLO").unwrap();
        let positions = bwt.locate_pattern(&result, b"XY");
        assert!(positions.is_empty());
    }

    #[test]
    fn test_rle_encode() {
        let encoded = rle_encode(b"AAABBC");
        assert_eq!(encoded, vec![(b'A', 3), (b'B', 2), (b'C', 1)]);
    }

    #[test]
    fn test_rle_decode() {
        let runs = vec![(b'A', 3), (b'B', 2)];
        assert_eq!(rle_decode(&runs), b"AAABB");
    }

    #[test]
    fn test_rle_roundtrip() {
        let data = b"AABBBCCCCDDDDDE";
        assert_eq!(rle_decode(&rle_encode(data)), data);
    }

    #[test]
    fn test_rle_empty() {
        assert!(rle_encode(b"").is_empty());
        assert!(rle_decode(&[]).is_empty());
    }

    #[test]
    fn test_with_sentinel() {
        let bwt = BurrowsWheeler::new().with_sentinel(b'$');
        let result = bwt.transform(b"ABC").unwrap();
        assert!(result.bwt.contains(&b'$'));
    }

    #[test]
    fn test_display_engine() {
        let bwt = BurrowsWheeler::new();
        let s = format!("{bwt}");
        assert!(s.contains("BurrowsWheeler"));
    }

    #[test]
    fn test_display_result() {
        let bwt = BurrowsWheeler::new();
        let result = bwt.transform(b"TEST").unwrap();
        let s = format!("{result}");
        assert!(s.contains("BWT"));
    }

    #[test]
    fn test_default() {
        let bwt = BurrowsWheeler::default();
        let result = bwt.transform(b"XYZ").unwrap();
        assert_eq!(result.bwt.len(), 4);
    }

    #[test]
    fn test_dna_sequence() {
        let bwt = BurrowsWheeler::new();
        let dna = b"GATTACA";
        let result = bwt.transform(dna).unwrap();
        let recovered = bwt.inverse(&result.bwt, result.primary_index).unwrap();
        assert_eq!(recovered, dna);
    }

    #[test]
    fn test_error_display() {
        let e = BwtError::EmptyInput;
        assert_eq!(format!("{e}"), "input must not be empty");
        let e2 = BwtError::InvalidSentinel("bad".into());
        assert!(format!("{e2}").contains("bad"));
    }

    #[test]
    fn test_inverse_empty_error() {
        let bwt = BurrowsWheeler::new();
        assert!(bwt.inverse(b"", 0).is_err());
    }

    #[test]
    fn test_inverse_bad_index() {
        let bwt = BurrowsWheeler::new();
        assert!(bwt.inverse(b"AB", 5).is_err());
    }
}
