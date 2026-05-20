//! FM-Index — Full-text index in Minute space for substring search,
//! occurrence counting, and locate queries over biological sequences.
//!
//! Pure-Rust FM-index built on top of BWT and suffix arrays. Provides
//! O(m) exact pattern counting (m = pattern length), locate queries via
//! sampled suffix array, and efficient rank/select on the BWT column
//! using checkpoint-based occurrence tables.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum FmIndexError {
    EmptyText,
    InvalidCheckpointInterval(usize),
    InvalidSampleRate(usize),
    PatternTooLong { pattern_len: usize, text_len: usize },
}

impl fmt::Display for FmIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyText => write!(f, "text must not be empty"),
            Self::InvalidCheckpointInterval(n) => {
                write!(f, "checkpoint interval must be > 0, got {n}")
            }
            Self::InvalidSampleRate(n) => {
                write!(f, "sample rate must be > 0, got {n}")
            }
            Self::PatternTooLong { pattern_len, text_len } => {
                write!(
                    f,
                    "pattern length {pattern_len} exceeds text length {text_len}"
                )
            }
        }
    }
}

impl std::error::Error for FmIndexError {}

// ── Suffix array builder ────────────────────────────────────────

fn build_suffix_array(text: &[u8]) -> Vec<usize> {
    let n = text.len();
    let mut sa: Vec<usize> = (0..n).collect();
    sa.sort_by(|&a, &b| text[a..].cmp(&text[b..]));
    sa
}

// ── Occurrence table ────────────────────────────────────────────

/// Checkpointed occurrence table for rank queries on the BWT.
#[derive(Debug, Clone)]
struct OccTable {
    /// Checkpoint every `interval` positions.
    interval: usize,
    /// checkpoints[i][c] = count of c in bwt[0..i*interval].
    checkpoints: Vec<[u32; 256]>,
    /// The BWT column itself for scanning between checkpoints.
    bwt: Vec<u8>,
}

impl OccTable {
    fn new(bwt: &[u8], interval: usize) -> Self {
        let n = bwt.len();
        let num_checkpoints = n / interval + 1;
        let mut checkpoints = vec![[0u32; 256]; num_checkpoints + 1];
        let mut counts = [0u32; 256];
        for (i, &b) in bwt.iter().enumerate() {
            if i % interval == 0 {
                checkpoints[i / interval] = counts;
            }
            counts[b as usize] += 1;
        }
        checkpoints[num_checkpoints] = counts;
        Self {
            interval,
            checkpoints,
            bwt: bwt.to_vec(),
        }
    }

    /// Rank query: count of character `c` in bwt[0..pos].
    fn rank(&self, c: u8, pos: usize) -> usize {
        let ci = c as usize;
        let block = pos / self.interval;
        let mut count = self.checkpoints[block][ci] as usize;
        let start = block * self.interval;
        for i in start..pos {
            if self.bwt[i] == c {
                count += 1;
            }
        }
        count
    }
}

// ── C-table ─────────────────────────────────────────────────────

/// C[c] = number of characters in text that are lexicographically smaller
/// than c.
fn build_c_table(bwt: &[u8]) -> [usize; 256] {
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
    c_table
}

// ── FM-Index ────────────────────────────────────────────────────

/// FM-index for efficient substring search over biological text.
#[derive(Debug, Clone)]
pub struct FmIndex {
    bwt: Vec<u8>,
    c_table: [usize; 256],
    occ: OccTable,
    suffix_array: Vec<usize>,
    sa_sample: Vec<Option<usize>>,
    sa_sample_rate: usize,
    text_len: usize,
}

impl FmIndex {
    /// Build an FM-index from raw text using default parameters.
    pub fn build(text: &[u8]) -> Result<Self, FmIndexError> {
        FmIndexBuilder::new().build(text)
    }

    /// Count how many times `pattern` occurs in the indexed text.
    pub fn count(&self, pattern: &[u8]) -> usize {
        if pattern.is_empty() {
            return self.text_len;
        }
        match self.backward_search(pattern) {
            Some((top, bottom)) => bottom - top,
            None => 0,
        }
    }

    /// Locate all positions where `pattern` occurs.
    pub fn locate(&self, pattern: &[u8]) -> Vec<usize> {
        if pattern.is_empty() {
            return Vec::new();
        }
        let (top, bottom) = match self.backward_search(pattern) {
            Some(range) => range,
            None => return Vec::new(),
        };

        let mut positions = Vec::with_capacity(bottom - top);
        for i in top..bottom {
            let pos = self.resolve_sa(i);
            positions.push(pos);
        }
        positions.sort();
        positions
    }

    /// Check if `pattern` exists in the indexed text.
    pub fn contains(&self, pattern: &[u8]) -> bool {
        if pattern.is_empty() {
            return true;
        }
        self.backward_search(pattern).is_some()
    }

    /// Return the length of the original text (without sentinel).
    pub fn text_length(&self) -> usize {
        self.text_len
    }

    /// Return the BWT string.
    pub fn bwt(&self) -> &[u8] {
        &self.bwt
    }

    /// Size of the index in bytes (approximate).
    pub fn size_bytes(&self) -> usize {
        self.bwt.len()
            + self.occ.checkpoints.len() * 256 * 4
            + self.sa_sample.len() * 16
            + 256 * 8
    }

    // ── Internal ────────────────────────────────────────────────

    fn backward_search(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        let n = self.bwt.len();
        let mut top = 0usize;
        let mut bottom = n;

        for &c in pattern.iter().rev() {
            let ci = c as usize;
            top = self.c_table[ci] + self.occ.rank(c, top);
            bottom = self.c_table[ci] + self.occ.rank(c, bottom);
            if top >= bottom {
                return None;
            }
        }
        Some((top, bottom))
    }

    fn resolve_sa(&self, mut idx: usize) -> usize {
        let mut steps = 0usize;
        loop {
            if let Some(pos) = self.sa_sample[idx] {
                return (pos + steps) % self.bwt.len();
            }
            // LF-step.
            let c = self.bwt[idx];
            idx = self.c_table[c as usize] + self.occ.rank(c, idx);
            steps += 1;
        }
    }
}

impl fmt::Display for FmIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FmIndex(text_len={}, bwt_len={}, ~{} bytes)",
            self.text_len,
            self.bwt.len(),
            self.size_bytes()
        )
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Builder for FmIndex with configurable checkpoint interval and SA
/// sample rate.
#[derive(Debug, Clone)]
pub struct FmIndexBuilder {
    checkpoint_interval: usize,
    sa_sample_rate: usize,
    sentinel: u8,
}

impl FmIndexBuilder {
    pub fn new() -> Self {
        Self {
            checkpoint_interval: 32,
            sa_sample_rate: 4,
            sentinel: 0x00,
        }
    }

    /// Set the checkpoint interval for the occurrence table.
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Set the suffix array sampling rate (higher = less memory).
    pub fn with_sa_sample_rate(mut self, rate: usize) -> Self {
        self.sa_sample_rate = rate;
        self
    }

    /// Set the sentinel character.
    pub fn with_sentinel(mut self, sentinel: u8) -> Self {
        self.sentinel = sentinel;
        self
    }

    /// Build the FM-index from the provided text.
    pub fn build(self, text: &[u8]) -> Result<FmIndex, FmIndexError> {
        if text.is_empty() {
            return Err(FmIndexError::EmptyText);
        }
        if self.checkpoint_interval == 0 {
            return Err(FmIndexError::InvalidCheckpointInterval(0));
        }
        if self.sa_sample_rate == 0 {
            return Err(FmIndexError::InvalidSampleRate(0));
        }

        let text_len = text.len();

        // Build augmented text with sentinel.
        let mut augmented = Vec::with_capacity(text_len + 1);
        augmented.extend_from_slice(text);
        augmented.push(self.sentinel);

        let sa = build_suffix_array(&augmented);
        let n = augmented.len();

        // Build BWT from suffix array.
        let mut bwt = Vec::with_capacity(n);
        for &s in &sa {
            if s == 0 {
                bwt.push(augmented[n - 1]);
            } else {
                bwt.push(augmented[s - 1]);
            }
        }

        let c_table = build_c_table(&bwt);
        let occ = OccTable::new(&bwt, self.checkpoint_interval);

        // Build sampled suffix array.
        let mut sa_sample = vec![None; n];
        for (i, &s) in sa.iter().enumerate() {
            if s % self.sa_sample_rate == 0 {
                sa_sample[i] = Some(s);
            }
        }

        Ok(FmIndex {
            bwt,
            c_table,
            occ,
            suffix_array: sa,
            sa_sample,
            sa_sample_rate: self.sa_sample_rate,
            text_len,
        })
    }
}

impl Default for FmIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FmIndexBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FmIndexBuilder(ckpt={}, sa_rate={})",
            self.checkpoint_interval, self.sa_sample_rate
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_basic() {
        let idx = FmIndex::build(b"BANANA").unwrap();
        assert_eq!(idx.text_length(), 6);
    }

    #[test]
    fn test_empty_error() {
        assert!(FmIndex::build(b"").is_err());
    }

    #[test]
    fn test_count_exact() {
        let idx = FmIndex::build(b"ABCABC").unwrap();
        assert_eq!(idx.count(b"ABC"), 2);
        assert_eq!(idx.count(b"XYZ"), 0);
    }

    #[test]
    fn test_count_single_char() {
        let idx = FmIndex::build(b"AABAA").unwrap();
        assert_eq!(idx.count(b"A"), 4);
        assert_eq!(idx.count(b"B"), 1);
    }

    #[test]
    fn test_contains() {
        let idx = FmIndex::build(b"GATTACA").unwrap();
        assert!(idx.contains(b"ATT"));
        assert!(idx.contains(b"ACA"));
        assert!(!idx.contains(b"GGG"));
    }

    #[test]
    fn test_locate_positions() {
        let idx = FmIndex::build(b"ABCABC").unwrap();
        let positions = idx.locate(b"ABC");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0], 0);
        assert_eq!(positions[1], 3);
    }

    #[test]
    fn test_locate_no_match() {
        let idx = FmIndex::build(b"HELLO").unwrap();
        assert!(idx.locate(b"XY").is_empty());
    }

    #[test]
    fn test_locate_empty_pattern() {
        let idx = FmIndex::build(b"TEST").unwrap();
        assert!(idx.locate(b"").is_empty());
    }

    #[test]
    fn test_count_empty_pattern() {
        let idx = FmIndex::build(b"XYZ").unwrap();
        assert_eq!(idx.count(b""), 3); // text length
    }

    #[test]
    fn test_contains_empty_pattern() {
        let idx = FmIndex::build(b"AB").unwrap();
        assert!(idx.contains(b""));
    }

    #[test]
    fn test_builder_custom_params() {
        let idx = FmIndexBuilder::new()
            .with_checkpoint_interval(16)
            .with_sa_sample_rate(2)
            .build(b"ACGTACGT")
            .unwrap();
        assert_eq!(idx.count(b"ACGT"), 2);
    }

    #[test]
    fn test_builder_invalid_interval() {
        let result = FmIndexBuilder::new()
            .with_checkpoint_interval(0)
            .build(b"ABC");
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_sample_rate() {
        let result = FmIndexBuilder::new()
            .with_sa_sample_rate(0)
            .build(b"ABC");
        assert!(result.is_err());
    }

    #[test]
    fn test_display_index() {
        let idx = FmIndex::build(b"HELLO").unwrap();
        let s = format!("{idx}");
        assert!(s.contains("FmIndex"));
        assert!(s.contains("text_len=5"));
    }

    #[test]
    fn test_display_builder() {
        let b = FmIndexBuilder::new();
        let s = format!("{b}");
        assert!(s.contains("FmIndexBuilder"));
    }

    #[test]
    fn test_size_bytes_positive() {
        let idx = FmIndex::build(b"ACGT").unwrap();
        assert!(idx.size_bytes() > 0);
    }

    #[test]
    fn test_bwt_accessor() {
        let idx = FmIndex::build(b"AB").unwrap();
        assert_eq!(idx.bwt().len(), 3); // AB + sentinel
    }

    #[test]
    fn test_dna_long_sequence() {
        let dna: Vec<u8> = (0..500).map(|i| b"ACGT"[i % 4]).collect();
        let idx = FmIndex::build(&dna).unwrap();
        assert!(idx.contains(b"ACGTACGT"));
    }

    #[test]
    fn test_error_display() {
        let e = FmIndexError::EmptyText;
        assert_eq!(format!("{e}"), "text must not be empty");
        let e2 = FmIndexError::InvalidCheckpointInterval(0);
        assert!(format!("{e2}").contains("0"));
    }

    #[test]
    fn test_builder_default() {
        let b = FmIndexBuilder::default();
        let idx = b.build(b"XYZ").unwrap();
        assert_eq!(idx.text_length(), 3);
    }
}
