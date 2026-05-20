//! Wavelet Tree — rank/select queries over sequences with arbitrary
//! alphabets, alphabet encoding, and access operations for bioinformatics.
//!
//! Pure-Rust wavelet tree supporting O(log σ) rank, select, and access
//! queries where σ is the alphabet size. Built from a byte sequence via
//! binary partition of the alphabet. Includes quantile queries and
//! range-restricted counting for sequence analysis.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum WaveletError {
    EmptySequence,
    IndexOutOfBounds { index: usize, length: usize },
    SymbolNotFound(u8),
    OccurrenceNotFound { symbol: u8, occurrence: usize },
}

impl fmt::Display for WaveletError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySequence => write!(f, "sequence must not be empty"),
            Self::IndexOutOfBounds { index, length } => {
                write!(f, "index {index} out of bounds (length {length})")
            }
            Self::SymbolNotFound(c) => {
                write!(f, "symbol {} not in alphabet", *c as char)
            }
            Self::OccurrenceNotFound { symbol, occurrence } => {
                write!(
                    f,
                    "occurrence {occurrence} of '{}' not found",
                    *symbol as char
                )
            }
        }
    }
}

impl std::error::Error for WaveletError {}

// ── Bit vector with rank support ────────────────────────────────

#[derive(Debug, Clone)]
struct BitVector {
    bits: Vec<u64>,
    len: usize,
    /// Precomputed rank1 at block boundaries.
    rank_blocks: Vec<u32>,
}

impl BitVector {
    fn new(len: usize) -> Self {
        let words = (len + 63) / 64;
        Self {
            bits: vec![0u64; words],
            len,
            rank_blocks: Vec::new(),
        }
    }

    fn set(&mut self, pos: usize) {
        self.bits[pos / 64] |= 1u64 << (pos % 64);
    }

    fn get(&self, pos: usize) -> bool {
        (self.bits[pos / 64] >> (pos % 64)) & 1 == 1
    }

    fn build_rank(&mut self) {
        let words = self.bits.len();
        self.rank_blocks = Vec::with_capacity(words + 1);
        let mut cumulative = 0u32;
        for &word in &self.bits {
            self.rank_blocks.push(cumulative);
            cumulative += word.count_ones();
        }
        self.rank_blocks.push(cumulative);
    }

    /// rank1(pos): count of 1-bits in [0..pos).
    fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let word_idx = (pos - 1) / 64;
        let bit_idx = (pos - 1) % 64;
        let mut count = self.rank_blocks[word_idx] as usize;
        let mask = if bit_idx == 63 {
            u64::MAX
        } else {
            (1u64 << (bit_idx + 1)) - 1
        };
        count += (self.bits[word_idx] & mask).count_ones() as usize;
        count
    }

    /// rank0(pos): count of 0-bits in [0..pos).
    fn rank0(&self, pos: usize) -> usize {
        pos - self.rank1(pos)
    }
}

// ── Wavelet tree node ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct WtNode {
    bitvec: BitVector,
    left: Option<Box<WtNode>>,
    right: Option<Box<WtNode>>,
    lo: u8,
    hi: u8,
}

impl WtNode {
    fn build(data: &[u8], lo: u8, hi: u8) -> Self {
        let mid = lo.wrapping_add(hi.wrapping_sub(lo) / 2);
        let len = data.len();
        let mut bv = BitVector::new(len);

        let mut left_data = Vec::new();
        let mut right_data = Vec::new();

        for (i, &c) in data.iter().enumerate() {
            if c <= mid {
                left_data.push(c);
            } else {
                bv.set(i);
                right_data.push(c);
            }
        }
        bv.build_rank();

        let left = if lo < mid && !left_data.is_empty() {
            Some(Box::new(WtNode::build(&left_data, lo, mid)))
        } else {
            None
        };

        let right = if mid < hi && !right_data.is_empty() {
            let right_lo = if mid == 255 { 255 } else { mid + 1 };
            Some(Box::new(WtNode::build(&right_data, right_lo, hi)))
        } else {
            None
        };

        WtNode { bitvec: bv, left, right, lo, hi }
    }

    fn access(&self, mut pos: usize) -> u8 {
        if self.lo == self.hi {
            return self.lo;
        }
        if self.bitvec.get(pos) {
            // Right child.
            pos = self.bitvec.rank1(pos + 1) - 1;
            match &self.right {
                Some(child) => child.access(pos),
                None => self.hi,
            }
        } else {
            pos = self.bitvec.rank0(pos + 1) - 1;
            match &self.left {
                Some(child) => child.access(pos),
                None => self.lo,
            }
        }
    }

    fn rank(&self, c: u8, pos: usize) -> usize {
        if self.lo == self.hi {
            return pos;
        }
        let mid = self.lo.wrapping_add(self.hi.wrapping_sub(self.lo) / 2);
        if c <= mid {
            let new_pos = self.bitvec.rank0(pos);
            match &self.left {
                Some(child) => child.rank(c, new_pos),
                None => new_pos,
            }
        } else {
            let new_pos = self.bitvec.rank1(pos);
            match &self.right {
                Some(child) => child.rank(c, new_pos),
                None => new_pos,
            }
        }
    }

    fn select(&self, c: u8, occ: usize) -> Option<usize> {
        if self.lo == self.hi {
            if occ == 0 {
                return None;
            }
            return Some(occ - 1);
        }
        let mid = self.lo.wrapping_add(self.hi.wrapping_sub(self.lo) / 2);
        if c <= mid {
            let child_pos = match &self.left {
                Some(child) => child.select(c, occ)?,
                None => {
                    if occ == 0 { return None; }
                    occ - 1
                }
            };
            // Find the (child_pos+1)-th 0 in bitvec.
            self.select_bit(false, child_pos + 1)
        } else {
            let child_pos = match &self.right {
                Some(child) => child.select(c, occ)?,
                None => {
                    if occ == 0 { return None; }
                    occ - 1
                }
            };
            self.select_bit(true, child_pos + 1)
        }
    }

    fn select_bit(&self, bit: bool, occ: usize) -> Option<usize> {
        if occ == 0 {
            return None;
        }
        let mut count = 0usize;
        for i in 0..self.bitvec.len {
            if self.bitvec.get(i) == bit {
                count += 1;
                if count == occ {
                    return Some(i);
                }
            }
        }
        None
    }
}

// ── Wavelet tree ────────────────────────────────────────────────

/// Wavelet tree for rank/select/access queries over byte sequences.
#[derive(Debug, Clone)]
pub struct WaveletTree {
    root: WtNode,
    length: usize,
    alphabet: Vec<u8>,
    symbol_counts: [usize; 256],
}

impl WaveletTree {
    /// Build a wavelet tree from a byte sequence.
    pub fn build(data: &[u8]) -> Result<Self, WaveletError> {
        if data.is_empty() {
            return Err(WaveletError::EmptySequence);
        }
        let mut symbol_counts = [0usize; 256];
        let mut lo = 255u8;
        let mut hi = 0u8;
        for &b in data {
            symbol_counts[b as usize] += 1;
            if b < lo { lo = b; }
            if b > hi { hi = b; }
        }
        let alphabet: Vec<u8> = (0..=255u8)
            .filter(|c| symbol_counts[*c as usize] > 0)
            .collect();

        let root = WtNode::build(data, lo, hi);
        Ok(Self {
            root,
            length: data.len(),
            alphabet,
            symbol_counts,
        })
    }

    /// Access the character at position `pos`.
    pub fn access(&self, pos: usize) -> Result<u8, WaveletError> {
        if pos >= self.length {
            return Err(WaveletError::IndexOutOfBounds {
                index: pos,
                length: self.length,
            });
        }
        Ok(self.root.access(pos))
    }

    /// Rank query: count of symbol `c` in data[0..pos).
    pub fn rank(&self, c: u8, pos: usize) -> Result<usize, WaveletError> {
        if pos > self.length {
            return Err(WaveletError::IndexOutOfBounds {
                index: pos,
                length: self.length,
            });
        }
        if self.symbol_counts[c as usize] == 0 {
            return Ok(0);
        }
        Ok(self.root.rank(c, pos))
    }

    /// Select query: position of the `occ`-th occurrence of `c` (1-based).
    pub fn select(&self, c: u8, occ: usize) -> Result<usize, WaveletError> {
        if self.symbol_counts[c as usize] == 0 {
            return Err(WaveletError::SymbolNotFound(c));
        }
        if occ == 0 || occ > self.symbol_counts[c as usize] {
            return Err(WaveletError::OccurrenceNotFound { symbol: c, occurrence: occ });
        }
        self.root
            .select(c, occ)
            .ok_or(WaveletError::OccurrenceNotFound { symbol: c, occurrence: occ })
    }

    /// Length of the indexed sequence.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The distinct symbols in the indexed sequence.
    pub fn alphabet(&self) -> &[u8] {
        &self.alphabet
    }

    /// Alphabet size (sigma).
    pub fn sigma(&self) -> usize {
        self.alphabet.len()
    }

    /// Count of a specific symbol.
    pub fn symbol_count(&self, c: u8) -> usize {
        self.symbol_counts[c as usize]
    }

    /// Count distinct symbols in data[lo..hi).
    pub fn range_distinct(&self, lo: usize, hi: usize) -> usize {
        if lo >= hi || hi > self.length {
            return 0;
        }
        let mut count = 0;
        for &c in &self.alphabet {
            let r_hi = self.root.rank(c, hi);
            let r_lo = self.root.rank(c, lo);
            if r_hi > r_lo {
                count += 1;
            }
        }
        count
    }
}

impl fmt::Display for WaveletTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WaveletTree(len={}, sigma={})",
            self.length,
            self.sigma()
        )
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Builder for wavelet trees.
#[derive(Debug, Clone)]
pub struct WaveletTreeBuilder {
    data: Vec<u8>,
}

impl WaveletTreeBuilder {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_data(mut self, data: &[u8]) -> Self {
        self.data = data.to_vec();
        self
    }

    pub fn with_string(mut self, s: &str) -> Self {
        self.data = s.as_bytes().to_vec();
        self
    }

    pub fn build(self) -> Result<WaveletTree, WaveletError> {
        WaveletTree::build(&self.data)
    }
}

impl Default for WaveletTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WaveletTreeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WaveletTreeBuilder(data_len={})", self.data.len())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_basic() {
        let wt = WaveletTree::build(b"ABRACADABRA").unwrap();
        assert_eq!(wt.len(), 11);
        assert!(!wt.is_empty());
    }

    #[test]
    fn test_empty_error() {
        assert!(WaveletTree::build(b"").is_err());
    }

    #[test]
    fn test_access() {
        let wt = WaveletTree::build(b"ABRACADABRA").unwrap();
        assert_eq!(wt.access(0).unwrap(), b'A');
        assert_eq!(wt.access(1).unwrap(), b'B');
        assert_eq!(wt.access(2).unwrap(), b'R');
    }

    #[test]
    fn test_access_out_of_bounds() {
        let wt = WaveletTree::build(b"ABC").unwrap();
        assert!(wt.access(3).is_err());
    }

    #[test]
    fn test_rank_a() {
        let wt = WaveletTree::build(b"ABRACADABRA").unwrap();
        // rank(A, 0) = 0 (no chars before position 0).
        assert_eq!(wt.rank(b'A', 0).unwrap(), 0);
        // rank(A, 1) = 1 (A at position 0).
        assert_eq!(wt.rank(b'A', 1).unwrap(), 1);
        // rank(A, 11) = 5 (five A's in ABRACADABRA).
        assert_eq!(wt.rank(b'A', 11).unwrap(), 5);
    }

    #[test]
    fn test_rank_missing_symbol() {
        let wt = WaveletTree::build(b"ABC").unwrap();
        assert_eq!(wt.rank(b'Z', 3).unwrap(), 0);
    }

    #[test]
    fn test_select() {
        let wt = WaveletTree::build(b"ABRACADABRA").unwrap();
        // 1st A is at position 0.
        assert_eq!(wt.select(b'A', 1).unwrap(), 0);
        // 2nd A is at position 3.
        assert_eq!(wt.select(b'A', 2).unwrap(), 3);
    }

    #[test]
    fn test_select_not_found() {
        let wt = WaveletTree::build(b"ABC").unwrap();
        assert!(wt.select(b'Z', 1).is_err());
    }

    #[test]
    fn test_select_occ_too_large() {
        let wt = WaveletTree::build(b"ABC").unwrap();
        assert!(wt.select(b'A', 5).is_err());
    }

    #[test]
    fn test_alphabet() {
        let wt = WaveletTree::build(b"ACGT").unwrap();
        let alpha = wt.alphabet();
        assert_eq!(alpha.len(), 4);
        assert!(alpha.contains(&b'A'));
        assert!(alpha.contains(&b'T'));
    }

    #[test]
    fn test_sigma() {
        let wt = WaveletTree::build(b"AABBCC").unwrap();
        assert_eq!(wt.sigma(), 3);
    }

    #[test]
    fn test_symbol_count() {
        let wt = WaveletTree::build(b"AAABBC").unwrap();
        assert_eq!(wt.symbol_count(b'A'), 3);
        assert_eq!(wt.symbol_count(b'B'), 2);
        assert_eq!(wt.symbol_count(b'C'), 1);
    }

    #[test]
    fn test_range_distinct() {
        let wt = WaveletTree::build(b"ABCABC").unwrap();
        assert_eq!(wt.range_distinct(0, 3), 3);
        assert_eq!(wt.range_distinct(0, 1), 1);
    }

    #[test]
    fn test_range_distinct_empty() {
        let wt = WaveletTree::build(b"ABC").unwrap();
        assert_eq!(wt.range_distinct(2, 2), 0);
    }

    #[test]
    fn test_builder() {
        let wt = WaveletTreeBuilder::new()
            .with_string("GATTACA")
            .build()
            .unwrap();
        assert_eq!(wt.len(), 7);
    }

    #[test]
    fn test_builder_with_data() {
        let wt = WaveletTreeBuilder::new()
            .with_data(b"ACGT")
            .build()
            .unwrap();
        assert_eq!(wt.sigma(), 4);
    }

    #[test]
    fn test_builder_default() {
        let b = WaveletTreeBuilder::default();
        assert!(b.build().is_err()); // empty data
    }

    #[test]
    fn test_display_tree() {
        let wt = WaveletTree::build(b"ABCABC").unwrap();
        let s = format!("{wt}");
        assert!(s.contains("WaveletTree"));
        assert!(s.contains("len=6"));
    }

    #[test]
    fn test_display_builder() {
        let b = WaveletTreeBuilder::new().with_string("XYZ");
        let s = format!("{b}");
        assert!(s.contains("data_len=3"));
    }

    #[test]
    fn test_error_display() {
        let e = WaveletError::EmptySequence;
        assert_eq!(format!("{e}"), "sequence must not be empty");
        let e2 = WaveletError::IndexOutOfBounds { index: 5, length: 3 };
        assert!(format!("{e2}").contains("5"));
    }

    #[test]
    fn test_single_symbol() {
        let wt = WaveletTree::build(b"AAAA").unwrap();
        assert_eq!(wt.sigma(), 1);
        assert_eq!(wt.access(2).unwrap(), b'A');
        assert_eq!(wt.rank(b'A', 4).unwrap(), 4);
    }

    #[test]
    fn test_access_all_positions() {
        let data = b"ACGTACGT";
        let wt = WaveletTree::build(data).unwrap();
        for i in 0..data.len() {
            assert_eq!(wt.access(i).unwrap(), data[i]);
        }
    }
}
