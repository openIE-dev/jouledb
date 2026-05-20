//! Fenwick tree (Binary Indexed Tree) — efficient prefix sums, point updates,
//! range queries, kth element, range update with range query, and 2D variant.
//!
//! Replaces JS BIT/Fenwick libraries with a pure-Rust implementation
//! supporting O(log n) queries and updates.

use std::fmt;

// ── FenwickTree (1D) ───────────────────────────────────────────────────────

/// A 1D Fenwick tree (Binary Indexed Tree) for prefix sum queries and point updates.
pub struct FenwickTree {
    tree: Vec<i64>,
    n: usize,
}

impl fmt::Debug for FenwickTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FenwickTree")
            .field("n", &self.n)
            .finish()
    }
}

impl FenwickTree {
    /// Create a Fenwick tree of size `n` initialized to all zeros.
    pub fn new(n: usize) -> Self {
        Self {
            tree: vec![0; n + 1], // 1-indexed
            n,
        }
    }

    /// Build a Fenwick tree from an array in O(n).
    pub fn from_array(data: &[i64]) -> Self {
        let n = data.len();
        let mut tree = vec![0i64; n + 1];
        for i in 0..n {
            tree[i + 1] = data[i];
        }
        for i in 1..=n {
            let parent = i + lowbit(i);
            if parent <= n {
                let val = tree[i];
                tree[parent] = tree[parent].saturating_add(val);
            }
        }
        Self { tree, n }
    }

    /// Size of the underlying array.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Point update: add `delta` to position `i` (0-indexed).
    pub fn update(&mut self, i: usize, delta: i64) {
        if i >= self.n {
            return;
        }
        let mut idx = i + 1; // convert to 1-indexed
        while idx <= self.n {
            self.tree[idx] = self.tree[idx].saturating_add(delta);
            idx += lowbit(idx);
        }
    }

    /// Prefix sum query: sum of elements in [0, i] (0-indexed, inclusive).
    pub fn prefix_sum(&self, i: usize) -> i64 {
        if i >= self.n {
            return self.prefix_sum_internal(self.n);
        }
        self.prefix_sum_internal(i + 1)
    }

    fn prefix_sum_internal(&self, mut idx: usize) -> i64 {
        let mut sum: i64 = 0;
        while idx > 0 {
            sum = sum.saturating_add(self.tree[idx]);
            idx -= lowbit(idx);
        }
        sum
    }

    /// Range sum query: sum of elements in [l, r] (0-indexed, inclusive).
    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        if l > r || r >= self.n {
            return 0;
        }
        if l == 0 {
            self.prefix_sum(r)
        } else {
            self.prefix_sum(r) - self.prefix_sum(l - 1)
        }
    }

    /// Find the smallest index `i` such that prefix_sum(i) >= target.
    /// Uses binary lifting in O(log n). Returns None if no such index.
    pub fn find_kth(&self, target: i64) -> Option<usize> {
        if self.n == 0 {
            return None;
        }
        let mut pos: usize = 0;
        let mut remaining = target;
        let mut bit_mask = highest_bit(self.n);

        while bit_mask > 0 {
            let next = pos + bit_mask;
            if next <= self.n && self.tree[next] < remaining {
                remaining -= self.tree[next];
                pos = next;
            }
            bit_mask >>= 1;
        }

        if pos + 1 <= self.n {
            Some(pos) // 0-indexed result
        } else {
            None
        }
    }
}

// ── RangeUpdateFenwick ─────────────────────────────────────────────────────

/// A Fenwick tree supporting both range updates and range queries.
/// Uses the B1/B2 trick: range_add(l, r, v) then query prefix_sum(i).
pub struct RangeUpdateFenwick {
    b1: Vec<i64>,
    b2: Vec<i64>,
    n: usize,
}

impl fmt::Debug for RangeUpdateFenwick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RangeUpdateFenwick")
            .field("n", &self.n)
            .finish()
    }
}

impl RangeUpdateFenwick {
    /// Create a range-update Fenwick tree of size `n`.
    pub fn new(n: usize) -> Self {
        Self {
            b1: vec![0; n + 2],
            b2: vec![0; n + 2],
            n,
        }
    }

    /// Size of the underlying array.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Add `delta` to all elements in [l, r] (0-indexed, inclusive).
    pub fn range_add(&mut self, l: usize, r: usize, delta: i64) {
        if l > r || r >= self.n {
            return;
        }
        let li = l + 1; // 1-indexed
        let ri = r + 2; // exclusive end, 1-indexed
        self.bit_add(&mut true, li, delta);
        self.bit_add(&mut true, ri, -delta);
        let l_val = delta.saturating_mul(li as i64 - 1);
        self.bit_add(&mut false, li, -l_val);
        let r_val = delta.saturating_mul(ri as i64 - 1);
        self.bit_add(&mut false, ri, r_val);
    }

    /// Prefix sum query: sum of elements in [0, i] (0-indexed, inclusive).
    pub fn prefix_sum(&self, i: usize) -> i64 {
        if i >= self.n {
            return self.prefix_sum_internal(self.n);
        }
        self.prefix_sum_internal(i + 1)
    }

    /// Range sum query: sum of elements in [l, r] (0-indexed, inclusive).
    pub fn range_sum(&self, l: usize, r: usize) -> i64 {
        if l > r || r >= self.n {
            return 0;
        }
        if l == 0 {
            self.prefix_sum(r)
        } else {
            self.prefix_sum(r) - self.prefix_sum(l - 1)
        }
    }

    fn prefix_sum_internal(&self, idx: usize) -> i64 {
        let s1 = self.bit_sum(&true, idx);
        let s2 = self.bit_sum(&false, idx);
        s1.saturating_mul(idx as i64).saturating_add(s2)
    }

    fn bit_add(&mut self, is_b1: &mut bool, mut idx: usize, delta: i64) {
        let arr = if *is_b1 { &mut self.b1 } else { &mut self.b2 };
        while idx <= self.n + 1 {
            arr[idx] = arr[idx].saturating_add(delta);
            idx += lowbit(idx);
        }
    }

    fn bit_sum(&self, is_b1: &bool, mut idx: usize) -> i64 {
        let arr = if *is_b1 { &self.b1 } else { &self.b2 };
        let mut sum: i64 = 0;
        while idx > 0 {
            sum = sum.saturating_add(arr[idx]);
            idx -= lowbit(idx);
        }
        sum
    }
}

// ── FenwickTree2D ──────────────────────────────────────────────────────────

/// A 2D Fenwick tree for rectangular prefix sum queries and point updates.
pub struct FenwickTree2D {
    tree: Vec<Vec<i64>>,
    rows: usize,
    cols: usize,
}

impl fmt::Debug for FenwickTree2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FenwickTree2D")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

impl FenwickTree2D {
    /// Create a 2D Fenwick tree of dimensions rows x cols.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            tree: vec![vec![0; cols + 1]; rows + 1],
            rows,
            cols,
        }
    }

    /// Dimensions.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Point update: add `delta` to cell (r, c) (0-indexed).
    pub fn update(&mut self, r: usize, c: usize, delta: i64) {
        if r >= self.rows || c >= self.cols {
            return;
        }
        let mut ri = r + 1;
        while ri <= self.rows {
            let mut ci = c + 1;
            while ci <= self.cols {
                self.tree[ri][ci] = self.tree[ri][ci].saturating_add(delta);
                ci += lowbit(ci);
            }
            ri += lowbit(ri);
        }
    }

    /// Prefix sum: sum of rectangle [(0,0), (r,c)] (0-indexed, inclusive).
    pub fn prefix_sum(&self, r: usize, c: usize) -> i64 {
        if r >= self.rows {
            return self.prefix_sum_internal(self.rows, c.min(self.cols.saturating_sub(1)) + 1);
        }
        self.prefix_sum_internal(r + 1, c.min(self.cols.saturating_sub(1)) + 1)
    }

    fn prefix_sum_internal(&self, mut ri: usize, ci: usize) -> i64 {
        let mut sum: i64 = 0;
        while ri > 0 {
            let mut cj = ci;
            while cj > 0 {
                sum = sum.saturating_add(self.tree[ri][cj]);
                cj -= lowbit(cj);
            }
            ri -= lowbit(ri);
        }
        sum
    }

    /// Range sum over rectangle [(r1,c1), (r2,c2)] (0-indexed, inclusive).
    pub fn range_sum(&self, r1: usize, c1: usize, r2: usize, c2: usize) -> i64 {
        if r1 > r2 || c1 > c2 || r2 >= self.rows || c2 >= self.cols {
            return 0;
        }
        let mut result = self.prefix_sum(r2, c2);
        if r1 > 0 {
            result -= self.prefix_sum(r1 - 1, c2);
        }
        if c1 > 0 {
            result -= self.prefix_sum(r2, c1 - 1);
        }
        if r1 > 0 && c1 > 0 {
            result += self.prefix_sum(r1 - 1, c1 - 1);
        }
        result
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Lowest set bit (isolate least significant 1).
fn lowbit(x: usize) -> usize {
    x & x.wrapping_neg()
}

/// Highest power of 2 <= n.
fn highest_bit(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut bit = 1;
    while bit <= n {
        bit <<= 1;
    }
    bit >> 1
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_fenwick() {
        let ft = FenwickTree::new(10);
        assert_eq!(ft.len(), 10);
        assert!(!ft.is_empty());
        assert_eq!(ft.prefix_sum(9), 0);
    }

    #[test]
    fn test_from_array() {
        let data = vec![1, 2, 3, 4, 5];
        let ft = FenwickTree::from_array(&data);
        assert_eq!(ft.prefix_sum(0), 1);
        assert_eq!(ft.prefix_sum(2), 6);
        assert_eq!(ft.prefix_sum(4), 15);
    }

    #[test]
    fn test_point_update() {
        let mut ft = FenwickTree::new(5);
        ft.update(0, 3);
        ft.update(1, 5);
        ft.update(2, 7);
        assert_eq!(ft.prefix_sum(0), 3);
        assert_eq!(ft.prefix_sum(1), 8);
        assert_eq!(ft.prefix_sum(2), 15);
    }

    #[test]
    fn test_range_sum() {
        let data = vec![1, 3, 5, 7, 9, 11];
        let ft = FenwickTree::from_array(&data);
        assert_eq!(ft.range_sum(1, 3), 15); // 3+5+7
        assert_eq!(ft.range_sum(0, 5), 36);
        assert_eq!(ft.range_sum(2, 2), 5);
    }

    #[test]
    fn test_update_and_query() {
        let data = vec![1, 2, 3, 4, 5];
        let mut ft = FenwickTree::from_array(&data);
        ft.update(2, 10); // index 2 goes from 3 to 13
        assert_eq!(ft.range_sum(0, 4), 25);
        assert_eq!(ft.range_sum(2, 2), 13);
    }

    #[test]
    fn test_find_kth() {
        let data = vec![1, 0, 2, 1, 1, 3, 0, 1];
        let ft = FenwickTree::from_array(&data);
        // Prefix sums: [1, 1, 3, 4, 5, 8, 8, 9]
        // find_kth(1) should be 0 (prefix_sum[0]=1 >= 1)
        assert_eq!(ft.find_kth(1), Some(0));
        // find_kth(3) should be 2 (prefix_sum[2]=3 >= 3)
        assert_eq!(ft.find_kth(3), Some(2));
        // find_kth(5) should be 4
        assert_eq!(ft.find_kth(5), Some(4));
    }

    #[test]
    fn test_find_kth_not_found() {
        let data = vec![1, 1, 1];
        let ft = FenwickTree::from_array(&data);
        // Total sum is 3, looking for 10
        assert_eq!(ft.find_kth(10), None);
    }

    #[test]
    fn test_empty_fenwick() {
        let ft = FenwickTree::new(0);
        assert!(ft.is_empty());
        assert_eq!(ft.prefix_sum(0), 0);
        assert_eq!(ft.range_sum(0, 0), 0);
    }

    #[test]
    fn test_range_update_point_query() {
        let mut ruf = RangeUpdateFenwick::new(5);
        ruf.range_add(1, 3, 10); // [0, 10, 10, 10, 0]
        assert_eq!(ruf.prefix_sum(0), 0);
        assert_eq!(ruf.prefix_sum(1), 10);
        assert_eq!(ruf.prefix_sum(2), 20);
        assert_eq!(ruf.prefix_sum(3), 30);
        assert_eq!(ruf.prefix_sum(4), 30);
    }

    #[test]
    fn test_range_update_range_query() {
        let mut ruf = RangeUpdateFenwick::new(5);
        ruf.range_add(0, 4, 1); // [1, 1, 1, 1, 1]
        ruf.range_add(1, 3, 2); // [1, 3, 3, 3, 1]
        assert_eq!(ruf.range_sum(0, 4), 11);
        assert_eq!(ruf.range_sum(1, 3), 9);
        assert_eq!(ruf.range_sum(0, 0), 1);
    }

    #[test]
    fn test_range_update_len() {
        let ruf = RangeUpdateFenwick::new(10);
        assert_eq!(ruf.len(), 10);
        assert!(!ruf.is_empty());
    }

    #[test]
    fn test_2d_fenwick() {
        let mut ft2d = FenwickTree2D::new(3, 4);
        assert_eq!(ft2d.dimensions(), (3, 4));
        ft2d.update(0, 0, 1);
        ft2d.update(1, 1, 2);
        ft2d.update(2, 3, 3);
        assert_eq!(ft2d.prefix_sum(0, 0), 1);
        assert_eq!(ft2d.prefix_sum(1, 1), 3);
        assert_eq!(ft2d.prefix_sum(2, 3), 6);
    }

    #[test]
    fn test_2d_range_sum() {
        let mut ft2d = FenwickTree2D::new(3, 3);
        // Fill a 3x3 grid:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        let vals = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        for r in 0..3 {
            for c in 0..3 {
                ft2d.update(r, c, vals[r][c]);
            }
        }
        // Total sum
        assert_eq!(ft2d.range_sum(0, 0, 2, 2), 45);
        // Center element
        assert_eq!(ft2d.range_sum(1, 1, 1, 1), 5);
        // Top-right 2x2
        assert_eq!(ft2d.range_sum(0, 1, 1, 2), 16); // 2+3+5+6
    }

    #[test]
    fn test_2d_update_accumulates() {
        let mut ft2d = FenwickTree2D::new(2, 2);
        ft2d.update(0, 0, 5);
        ft2d.update(0, 0, 3);
        assert_eq!(ft2d.prefix_sum(0, 0), 8);
    }

    #[test]
    fn test_lowbit() {
        assert_eq!(lowbit(1), 1);
        assert_eq!(lowbit(6), 2);
        assert_eq!(lowbit(8), 8);
        assert_eq!(lowbit(12), 4);
    }

    #[test]
    fn test_highest_bit() {
        assert_eq!(highest_bit(1), 1);
        assert_eq!(highest_bit(5), 4);
        assert_eq!(highest_bit(8), 8);
        assert_eq!(highest_bit(10), 8);
    }

    #[test]
    fn test_negative_values() {
        let data = vec![-3, 5, -2, 7, -1];
        let ft = FenwickTree::from_array(&data);
        assert_eq!(ft.prefix_sum(4), 6);
        assert_eq!(ft.range_sum(0, 2), 0);
    }

    #[test]
    fn test_large_fenwick() {
        let n = 1000;
        let data: Vec<i64> = (1..=n).collect();
        let ft = FenwickTree::from_array(&data);
        assert_eq!(ft.prefix_sum(n as usize - 1), n * (n + 1) / 2);
    }
}
