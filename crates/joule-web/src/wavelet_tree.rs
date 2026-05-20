//! Wavelet tree — a space-efficient data structure for sequences over an
//! alphabet, supporting rank, select, range frequency, and quantile queries
//! in O(log sigma) time where sigma is the alphabet size.
//!
//! Replaces JS wavelet-tree and rank/select libraries with a pure-Rust
//! implementation using bitvectors at each node.

use std::fmt;

// ── Bitvector ──────────────────────────────────────────────────────────────

/// A simple bitvector with rank/select support.
#[derive(Debug, Clone)]
struct BitVector {
    bits: Vec<u64>,
    len: usize,
    /// Prefix popcount for O(1) rank queries. rank_table[i] = popcount(bits[0..i]).
    rank_table: Vec<u32>,
}

impl BitVector {
    fn new(len: usize) -> Self {
        let words = (len + 63) / 64;
        Self {
            bits: vec![0; words],
            len,
            rank_table: Vec::new(),
        }
    }

    fn set(&mut self, pos: usize) {
        if pos < self.len {
            self.bits[pos / 64] |= 1u64 << (pos % 64);
        }
    }

    fn get(&self, pos: usize) -> bool {
        if pos >= self.len {
            return false;
        }
        (self.bits[pos / 64] >> (pos % 64)) & 1 == 1
    }

    /// Build rank table for O(1) rank queries.
    fn build_rank(&mut self) {
        let words = self.bits.len();
        self.rank_table = Vec::with_capacity(words + 1);
        let mut acc: u32 = 0;
        self.rank_table.push(0);
        for i in 0..words {
            acc += self.bits[i].count_ones();
            self.rank_table.push(acc);
        }
    }

    /// rank1(pos): number of 1-bits in [0, pos) (exclusive upper bound).
    fn rank1(&self, pos: usize) -> usize {
        if pos == 0 {
            return 0;
        }
        let pos = pos.min(self.len);
        let word = pos / 64;
        let bit = pos % 64;
        let mut count = self.rank_table[word] as usize;
        if bit > 0 {
            // Count bits in the partial word
            let mask = (1u64 << bit) - 1;
            count += (self.bits[word] & mask).count_ones() as usize;
        }
        count
    }

    /// rank0(pos): number of 0-bits in [0, pos).
    fn rank0(&self, pos: usize) -> usize {
        let pos = pos.min(self.len);
        pos - self.rank1(pos)
    }

    /// select1(k): position of the (k+1)th 1-bit (0-indexed k).
    /// Returns None if fewer than k+1 ones.
    fn select1(&self, k: usize) -> Option<usize> {
        // Linear scan (sufficient for moderate sizes)
        let mut count = 0;
        for pos in 0..self.len {
            if self.get(pos) {
                if count == k {
                    return Some(pos);
                }
                count += 1;
            }
        }
        None
    }

    /// select0(k): position of the (k+1)th 0-bit (0-indexed k).
    fn select0(&self, k: usize) -> Option<usize> {
        let mut count = 0;
        for pos in 0..self.len {
            if !self.get(pos) {
                if count == k {
                    return Some(pos);
                }
                count += 1;
            }
        }
        None
    }
}

// ── WaveletTree node ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WtNode {
    bv: BitVector,
    left: Option<usize>,
    right: Option<usize>,
    lo: u64, // alphabet range [lo, hi]
    hi: u64,
}

// ── WaveletTree ────────────────────────────────────────────────────────────

/// A wavelet tree over a sequence of unsigned integers, supporting
/// rank, select, range frequency, and quantile queries.
pub struct WaveletTree {
    nodes: Vec<WtNode>,
    root: Option<usize>,
    len: usize,
    alpha_lo: u64,
    alpha_hi: u64,
}

impl fmt::Debug for WaveletTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WaveletTree")
            .field("len", &self.len)
            .field("alpha_lo", &self.alpha_lo)
            .field("alpha_hi", &self.alpha_hi)
            .finish()
    }
}

impl WaveletTree {
    /// Build a wavelet tree from a sequence of unsigned integers.
    pub fn build(data: &[u64]) -> Self {
        if data.is_empty() {
            return Self {
                nodes: Vec::new(),
                root: None,
                len: 0,
                alpha_lo: 0,
                alpha_hi: 0,
            };
        }

        let alpha_lo = *data.iter().min().unwrap();
        let alpha_hi = *data.iter().max().unwrap();

        let mut wt = Self {
            nodes: Vec::new(),
            root: None,
            len: data.len(),
            alpha_lo,
            alpha_hi,
        };

        let root = wt.build_rec(data, alpha_lo, alpha_hi);
        wt.root = Some(root);
        wt
    }

    /// Build from a string (treating each char as its Unicode codepoint).
    pub fn from_str(text: &str) -> Self {
        let data: Vec<u64> = text.chars().map(|c| c as u64).collect();
        Self::build(&data)
    }

    /// Length of the sequence.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Access: return the value at position `pos`.
    pub fn access(&self, pos: usize) -> Option<u64> {
        if pos >= self.len {
            return None;
        }
        let root = self.root?;
        Some(self.access_rec(root, pos))
    }

    /// Rank query: count occurrences of `symbol` in [0, pos) (exclusive).
    pub fn rank(&self, symbol: u64, pos: usize) -> usize {
        if pos == 0 || self.root.is_none() {
            return 0;
        }
        let pos = pos.min(self.len);
        let root = self.root.unwrap();
        self.rank_rec(root, symbol, pos)
    }

    /// Select query: position of the (k+1)th occurrence of `symbol` (0-indexed k).
    /// Returns None if fewer than k+1 occurrences.
    pub fn select(&self, symbol: u64, k: usize) -> Option<usize> {
        let root = self.root?;
        self.select_rec(root, symbol, k)
    }

    /// Range frequency: count occurrences of `symbol` in [l, r) (exclusive upper).
    pub fn range_freq(&self, symbol: u64, l: usize, r: usize) -> usize {
        if l >= r || r > self.len {
            return 0;
        }
        self.rank(symbol, r) - self.rank(symbol, l)
    }

    /// Range frequency for a range of symbols: count values in [sym_lo, sym_hi]
    /// that appear in positions [l, r) (exclusive upper bound on position).
    pub fn range_freq_range(&self, l: usize, r: usize, sym_lo: u64, sym_hi: u64) -> usize {
        if l >= r || r > self.len || sym_lo > sym_hi {
            return 0;
        }
        let root = match self.root {
            Some(r) => r,
            None => return 0,
        };
        self.range_freq_range_rec(root, l, r, sym_lo, sym_hi)
    }

    /// Quantile query: find the kth smallest value in positions [l, r) (0-indexed k).
    pub fn quantile(&self, l: usize, r: usize, k: usize) -> Option<u64> {
        if l >= r || r > self.len || k >= r - l {
            return None;
        }
        let root = self.root?;
        Some(self.quantile_rec(root, l, r, k))
    }

    // ── Internal ──

    fn build_rec(&mut self, data: &[u64], lo: u64, hi: u64) -> usize {
        let idx = self.nodes.len();
        let mut bv = BitVector::new(data.len());

        if lo == hi {
            // Leaf node
            bv.build_rank();
            self.nodes.push(WtNode {
                bv,
                left: None,
                right: None,
                lo,
                hi,
            });
            return idx;
        }

        let mid = lo + (hi - lo) / 2;

        // Mark bits: 0 if value <= mid (goes left), 1 if value > mid (goes right)
        let mut left_data = Vec::new();
        let mut right_data = Vec::new();
        for (i, val) in data.iter().enumerate() {
            if *val > mid {
                bv.set(i);
                right_data.push(*val);
            } else {
                left_data.push(*val);
            }
        }
        bv.build_rank();

        // Placeholder node
        self.nodes.push(WtNode {
            bv,
            left: None,
            right: None,
            lo,
            hi,
        });

        let left = if left_data.is_empty() {
            None
        } else {
            Some(self.build_rec(&left_data, lo, mid))
        };

        let right = if right_data.is_empty() {
            None
        } else {
            Some(self.build_rec(&right_data, mid + 1, hi))
        };

        self.nodes[idx].left = left;
        self.nodes[idx].right = right;

        idx
    }

    fn access_rec(&self, node: usize, pos: usize) -> u64 {
        let n = &self.nodes[node];
        if n.lo == n.hi {
            return n.lo;
        }

        if n.bv.get(pos) {
            // Goes right
            let new_pos = n.bv.rank1(pos);
            match n.right {
                Some(right) => self.access_rec(right, new_pos),
                None => n.hi, // shouldn't happen if tree is well-formed
            }
        } else {
            // Goes left
            let new_pos = n.bv.rank0(pos);
            match n.left {
                Some(left) => self.access_rec(left, new_pos),
                None => n.lo,
            }
        }
    }

    fn rank_rec(&self, node: usize, symbol: u64, pos: usize) -> usize {
        let n = &self.nodes[node];
        if n.lo == n.hi {
            return pos; // all elements in this leaf are `symbol`
        }

        let mid = n.lo + (n.hi - n.lo) / 2;
        if symbol <= mid {
            // Symbol is in left subtree
            let new_pos = n.bv.rank0(pos);
            match n.left {
                Some(left) => self.rank_rec(left, symbol, new_pos),
                None => 0,
            }
        } else {
            // Symbol is in right subtree
            let new_pos = n.bv.rank1(pos);
            match n.right {
                Some(right) => self.rank_rec(right, symbol, new_pos),
                None => 0,
            }
        }
    }

    fn select_rec(&self, node: usize, symbol: u64, k: usize) -> Option<usize> {
        let n = &self.nodes[node];
        if n.lo == n.hi {
            // Leaf: the kth occurrence is at position k in this node's bitvector
            if k >= n.bv.len {
                return None;
            }
            return Some(k);
        }

        let mid = n.lo + (n.hi - n.lo) / 2;
        if symbol <= mid {
            // Recurse left
            let child_pos = match n.left {
                Some(left) => self.select_rec(left, symbol, k)?,
                None => return None,
            };
            // Map back: find the (child_pos+1)th 0-bit
            n.bv.select0(child_pos)
        } else {
            let child_pos = match n.right {
                Some(right) => self.select_rec(right, symbol, k)?,
                None => return None,
            };
            n.bv.select1(child_pos)
        }
    }

    fn range_freq_range_rec(
        &self,
        node: usize,
        l: usize,
        r: usize,
        sym_lo: u64,
        sym_hi: u64,
    ) -> usize {
        let n = &self.nodes[node];

        // If this node's alphabet range is entirely within query range
        if sym_lo <= n.lo && n.hi <= sym_hi {
            return r - l;
        }

        if n.lo == n.hi {
            if n.lo >= sym_lo && n.lo <= sym_hi {
                return r - l;
            }
            return 0;
        }

        let mid = n.lo + (n.hi - n.lo) / 2;
        let mut count = 0;

        if sym_lo <= mid {
            if let Some(left) = n.left {
                let new_l = n.bv.rank0(l);
                let new_r = n.bv.rank0(r);
                if new_l < new_r {
                    count += self.range_freq_range_rec(left, new_l, new_r, sym_lo, sym_hi.min(mid));
                }
            }
        }

        if sym_hi > mid {
            if let Some(right) = n.right {
                let new_l = n.bv.rank1(l);
                let new_r = n.bv.rank1(r);
                if new_l < new_r {
                    count += self.range_freq_range_rec(
                        right,
                        new_l,
                        new_r,
                        sym_lo.max(mid + 1),
                        sym_hi,
                    );
                }
            }
        }

        count
    }

    fn quantile_rec(&self, node: usize, l: usize, r: usize, k: usize) -> u64 {
        let n = &self.nodes[node];
        if n.lo == n.hi {
            return n.lo;
        }

        let zeros_l = n.bv.rank0(l);
        let zeros_r = n.bv.rank0(r);
        let left_count = zeros_r - zeros_l;

        if k < left_count {
            // Answer is in left subtree
            match n.left {
                Some(left) => self.quantile_rec(left, zeros_l, zeros_r, k),
                None => n.lo,
            }
        } else {
            // Answer is in right subtree
            let ones_l = n.bv.rank1(l);
            let ones_r = n.bv.rank1(r);
            match n.right {
                Some(right) => self.quantile_rec(right, ones_l, ones_r, k - left_count),
                None => n.hi,
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BitVector tests ──

    #[test]
    fn test_bitvector_set_get() {
        let mut bv = BitVector::new(100);
        bv.set(0);
        bv.set(5);
        bv.set(63);
        bv.set(64);
        bv.set(99);
        bv.build_rank();

        assert!(bv.get(0));
        assert!(!bv.get(1));
        assert!(bv.get(5));
        assert!(bv.get(63));
        assert!(bv.get(64));
        assert!(bv.get(99));
        assert!(!bv.get(50));
    }

    #[test]
    fn test_bitvector_rank() {
        let mut bv = BitVector::new(10);
        bv.set(1);
        bv.set(3);
        bv.set(5);
        bv.set(7);
        bv.build_rank();

        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank1(2), 1);
        assert_eq!(bv.rank1(4), 2);
        assert_eq!(bv.rank1(8), 4);
        assert_eq!(bv.rank1(10), 4);

        assert_eq!(bv.rank0(2), 1);
        assert_eq!(bv.rank0(4), 2);
    }

    #[test]
    fn test_bitvector_select() {
        let mut bv = BitVector::new(10);
        bv.set(2);
        bv.set(5);
        bv.set(8);
        bv.build_rank();

        assert_eq!(bv.select1(0), Some(2));
        assert_eq!(bv.select1(1), Some(5));
        assert_eq!(bv.select1(2), Some(8));
        assert_eq!(bv.select1(3), None);

        assert_eq!(bv.select0(0), Some(0));
        assert_eq!(bv.select0(1), Some(1));
    }

    // ── WaveletTree tests ──

    #[test]
    fn test_build_empty() {
        let wt = WaveletTree::build(&[]);
        assert!(wt.is_empty());
        assert_eq!(wt.len(), 0);
    }

    #[test]
    fn test_access() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::build(&data);
        for (i, val) in data.iter().enumerate() {
            assert_eq!(wt.access(i), Some(*val), "access({}) failed", i);
        }
        assert_eq!(wt.access(8), None);
    }

    #[test]
    fn test_rank() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::build(&data);
        // rank(1, 4) = count of 1s in [0, 4) = positions 1,3 => 2
        assert_eq!(wt.rank(1, 4), 2);
        // rank(5, 8) = count of 5s in [0, 8) = 1
        assert_eq!(wt.rank(5, 8), 1);
        // rank(1, 0) = 0
        assert_eq!(wt.rank(1, 0), 0);
        // rank(9, 8) = 1
        assert_eq!(wt.rank(9, 8), 1);
    }

    #[test]
    fn test_select() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::build(&data);
        // select(1, 0) = first occurrence of 1 at position 1
        assert_eq!(wt.select(1, 0), Some(1));
        // select(1, 1) = second occurrence of 1 at position 3
        assert_eq!(wt.select(1, 1), Some(3));
        // select(1, 2) = no third occurrence
        assert_eq!(wt.select(1, 2), None);
        // select(9, 0) = first occurrence of 9 at position 5
        assert_eq!(wt.select(9, 0), Some(5));
    }

    #[test]
    fn test_range_freq() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::build(&data);
        // Count of 1 in positions [1, 5)
        assert_eq!(wt.range_freq(1, 1, 5), 2);
        // Count of 4 in positions [2, 3)
        assert_eq!(wt.range_freq(4, 2, 3), 1);
        // Count of 7 in entire sequence
        assert_eq!(wt.range_freq(7, 0, 8), 0);
    }

    #[test]
    fn test_range_freq_range() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::build(&data);
        // Count values in [1, 5] in positions [0, 8)
        assert_eq!(wt.range_freq_range(0, 8, 1, 5), 6); // 3,1,4,1,5,2
        // Count values in [6, 9] in positions [0, 8)
        assert_eq!(wt.range_freq_range(0, 8, 6, 9), 2); // 9,6
    }

    #[test]
    fn test_quantile() {
        let data = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let wt = WaveletTree::build(&data);
        // 0th smallest in [0, 8) = 1
        assert_eq!(wt.quantile(0, 8, 0), Some(1));
        // 1st smallest in [0, 8) = 1
        assert_eq!(wt.quantile(0, 8, 1), Some(1));
        // 7th smallest (largest) in [0, 8) = 9
        assert_eq!(wt.quantile(0, 8, 7), Some(9));
    }

    #[test]
    fn test_quantile_subrange() {
        let data = vec![3, 1, 4, 1, 5];
        let wt = WaveletTree::build(&data);
        // 0th smallest in [0, 3) = {3,1,4} -> 1
        assert_eq!(wt.quantile(0, 3, 0), Some(1));
        // 2nd smallest in [0, 3) = {3,1,4} -> 4
        assert_eq!(wt.quantile(0, 3, 2), Some(4));
    }

    #[test]
    fn test_from_str() {
        let wt = WaveletTree::from_str("abracadabra");
        assert_eq!(wt.len(), 11);
        assert_eq!(wt.access(0), Some('a' as u64));
        assert_eq!(wt.access(1), Some('b' as u64));
        // rank of 'a' up to position 11
        assert_eq!(wt.rank('a' as u64, 11), 5);
    }

    #[test]
    fn test_single_element() {
        let wt = WaveletTree::build(&[42]);
        assert_eq!(wt.len(), 1);
        assert_eq!(wt.access(0), Some(42));
        assert_eq!(wt.rank(42, 1), 1);
        assert_eq!(wt.select(42, 0), Some(0));
    }

    #[test]
    fn test_all_same() {
        let data = vec![5, 5, 5, 5, 5];
        let wt = WaveletTree::build(&data);
        assert_eq!(wt.rank(5, 5), 5);
        assert_eq!(wt.rank(5, 3), 3);
        assert_eq!(wt.select(5, 4), Some(4));
        assert_eq!(wt.quantile(0, 5, 2), Some(5));
    }

    #[test]
    fn test_quantile_out_of_bounds() {
        let data = vec![1, 2, 3];
        let wt = WaveletTree::build(&data);
        assert_eq!(wt.quantile(0, 3, 3), None); // only 3 elements
        assert_eq!(wt.quantile(0, 3, 10), None);
    }

    #[test]
    fn test_range_freq_empty_range() {
        let data = vec![1, 2, 3];
        let wt = WaveletTree::build(&data);
        assert_eq!(wt.range_freq(1, 2, 2), 0); // empty range
        assert_eq!(wt.range_freq(1, 5, 3), 0); // l > r
    }
}
