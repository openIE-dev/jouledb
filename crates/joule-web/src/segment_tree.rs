//! Segment tree — efficient range queries (sum/min/max) with point updates,
//! lazy propagation for range updates, and a persistent variant.
//!
//! Replaces JS segment-tree and range-query libraries with a pure-Rust
//! implementation supporting O(log n) query and update.

use std::fmt;

// ── Operation ──────────────────────────────────────────────────────────────

/// The aggregation operation for range queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Sum,
    Min,
    Max,
}

// ── SegmentTree ────────────────────────────────────────────────────────────

/// A segment tree supporting range queries and point updates.
pub struct SegmentTree {
    tree: Vec<i64>,
    lazy: Vec<i64>,
    n: usize,
    op: Operation,
}

impl fmt::Debug for SegmentTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentTree")
            .field("n", &self.n)
            .field("op", &self.op)
            .finish()
    }
}

impl SegmentTree {
    /// Build a segment tree from an array with the given aggregation operation.
    pub fn build(data: &[i64], op: Operation) -> Self {
        let n = data.len();
        if n == 0 {
            return Self {
                tree: Vec::new(),
                lazy: Vec::new(),
                n: 0,
                op,
            };
        }
        let size = 4 * n;
        let mut st = Self {
            tree: vec![Self::identity(op); size],
            lazy: vec![0; size],
            n,
            op,
        };
        st.build_rec(data, 1, 0, n - 1);
        st
    }

    /// Query the aggregate over the range [l, r] (inclusive, 0-indexed).
    pub fn query(&mut self, l: usize, r: usize) -> i64 {
        if l > r || r >= self.n {
            return Self::identity(self.op);
        }
        self.query_rec(1, 0, self.n - 1, l, r)
    }

    /// Point update: set data[pos] = value and propagate.
    pub fn point_update(&mut self, pos: usize, value: i64) {
        if pos >= self.n {
            return;
        }
        self.point_update_rec(1, 0, self.n - 1, pos, value);
    }

    /// Range update with lazy propagation: add `delta` to all elements in [l, r].
    pub fn range_update(&mut self, l: usize, r: usize, delta: i64) {
        if l > r || r >= self.n {
            return;
        }
        self.range_update_rec(1, 0, self.n - 1, l, r, delta);
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    // ── Internal ──

    fn identity(op: Operation) -> i64 {
        match op {
            Operation::Sum => 0,
            Operation::Min => i64::MAX,
            Operation::Max => i64::MIN,
        }
    }

    fn combine(&self, a: i64, b: i64) -> i64 {
        match self.op {
            Operation::Sum => a.saturating_add(b),
            Operation::Min => a.min(b),
            Operation::Max => a.max(b),
        }
    }

    fn build_rec(&mut self, data: &[i64], node: usize, start: usize, end: usize) {
        if start == end {
            self.tree[node] = data[start];
            return;
        }
        let mid = start + (end - start) / 2;
        self.build_rec(data, 2 * node, start, mid);
        self.build_rec(data, 2 * node + 1, mid + 1, end);
        let left_val = self.tree[2 * node];
        let right_val = self.tree[2 * node + 1];
        self.tree[node] = self.combine(left_val, right_val);
    }

    fn push_down(&mut self, node: usize, start: usize, end: usize) {
        if self.lazy[node] != 0 {
            let delta = self.lazy[node];
            let mid = start + (end - start) / 2;
            self.apply_lazy(2 * node, start, mid, delta);
            self.apply_lazy(2 * node + 1, mid + 1, end, delta);
            self.lazy[node] = 0;
        }
    }

    fn apply_lazy(&mut self, node: usize, start: usize, end: usize, delta: i64) {
        match self.op {
            Operation::Sum => {
                let count = (end - start + 1) as i64;
                self.tree[node] = self.tree[node].saturating_add(delta.saturating_mul(count));
            }
            Operation::Min | Operation::Max => {
                self.tree[node] = self.tree[node].saturating_add(delta);
            }
        }
        self.lazy[node] = self.lazy[node].saturating_add(delta);
    }

    fn query_rec(&mut self, node: usize, start: usize, end: usize, l: usize, r: usize) -> i64 {
        if r < start || end < l {
            return Self::identity(self.op);
        }
        if l <= start && end <= r {
            return self.tree[node];
        }
        self.push_down(node, start, end);
        let mid = start + (end - start) / 2;
        let left_val = self.query_rec(2 * node, start, mid, l, r);
        let right_val = self.query_rec(2 * node + 1, mid + 1, end, l, r);
        self.combine(left_val, right_val)
    }

    fn point_update_rec(&mut self, node: usize, start: usize, end: usize, pos: usize, value: i64) {
        if start == end {
            self.tree[node] = value;
            self.lazy[node] = 0;
            return;
        }
        self.push_down(node, start, end);
        let mid = start + (end - start) / 2;
        if pos <= mid {
            self.point_update_rec(2 * node, start, mid, pos, value);
        } else {
            self.point_update_rec(2 * node + 1, mid + 1, end, pos, value);
        }
        let left_val = self.tree[2 * node];
        let right_val = self.tree[2 * node + 1];
        self.tree[node] = self.combine(left_val, right_val);
    }

    fn range_update_rec(
        &mut self,
        node: usize,
        start: usize,
        end: usize,
        l: usize,
        r: usize,
        delta: i64,
    ) {
        if r < start || end < l {
            return;
        }
        if l <= start && end <= r {
            self.apply_lazy(node, start, end, delta);
            return;
        }
        self.push_down(node, start, end);
        let mid = start + (end - start) / 2;
        self.range_update_rec(2 * node, start, mid, l, r, delta);
        self.range_update_rec(2 * node + 1, mid + 1, end, l, r, delta);
        let left_val = self.tree[2 * node];
        let right_val = self.tree[2 * node + 1];
        self.tree[node] = self.combine(left_val, right_val);
    }
}

// ── PersistentSegmentTree ──────────────────────────────────────────────────

/// A node in the persistent segment tree (path-copying immutability).
#[derive(Debug, Clone, Copy)]
struct PersNode {
    value: i64,
    left: Option<usize>,
    right: Option<usize>,
}

/// A persistent segment tree supporting point updates with version history.
/// Each update creates a new version without modifying old ones.
pub struct PersistentSegmentTree {
    nodes: Vec<PersNode>,
    roots: Vec<usize>,
    n: usize,
    op: Operation,
}

impl fmt::Debug for PersistentSegmentTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PersistentSegmentTree")
            .field("n", &self.n)
            .field("versions", &self.roots.len())
            .finish()
    }
}

impl PersistentSegmentTree {
    /// Build from an array. Version 0 is the initial state.
    pub fn build(data: &[i64], op: Operation) -> Self {
        let n = data.len();
        let mut pst = Self {
            nodes: Vec::with_capacity(4 * n.max(1)),
            roots: Vec::new(),
            n,
            op,
        };
        if n == 0 {
            let root = pst.new_node(0, None, None);
            pst.roots.push(root);
            return pst;
        }
        let root = pst.build_rec(data, 0, n - 1);
        pst.roots.push(root);
        pst
    }

    /// Number of versions available.
    pub fn versions(&self) -> usize {
        self.roots.len()
    }

    /// Query on a specific version.
    pub fn query(&self, version: usize, l: usize, r: usize) -> i64 {
        if version >= self.roots.len() || l > r || r >= self.n {
            return SegmentTree::identity(self.op);
        }
        self.query_rec(self.roots[version], 0, self.n - 1, l, r)
    }

    /// Point update creating a new version. Returns the new version index.
    pub fn update(&mut self, version: usize, pos: usize, value: i64) -> usize {
        if version >= self.roots.len() || pos >= self.n {
            return version;
        }
        let old_root = self.roots[version];
        let new_root = self.update_rec(old_root, 0, self.n - 1, pos, value);
        let new_version = self.roots.len();
        self.roots.push(new_root);
        new_version
    }

    // ── Internal ──

    fn new_node(&mut self, value: i64, left: Option<usize>, right: Option<usize>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(PersNode { value, left, right });
        idx
    }

    fn combine(&self, a: i64, b: i64) -> i64 {
        match self.op {
            Operation::Sum => a.saturating_add(b),
            Operation::Min => a.min(b),
            Operation::Max => a.max(b),
        }
    }

    fn build_rec(&mut self, data: &[i64], start: usize, end: usize) -> usize {
        if start == end {
            return self.new_node(data[start], None, None);
        }
        let mid = start + (end - start) / 2;
        let left = self.build_rec(data, start, mid);
        let right = self.build_rec(data, mid + 1, end);
        let left_val = self.nodes[left].value;
        let right_val = self.nodes[right].value;
        let val = self.combine(left_val, right_val);
        self.new_node(val, Some(left), Some(right))
    }

    fn query_rec(&self, node: usize, start: usize, end: usize, l: usize, r: usize) -> i64 {
        if r < start || end < l {
            return SegmentTree::identity(self.op);
        }
        if l <= start && end <= r {
            return self.nodes[node].value;
        }
        let mid = start + (end - start) / 2;
        let lv = match self.nodes[node].left {
            Some(ln) => self.query_rec(ln, start, mid, l, r),
            None => SegmentTree::identity(self.op),
        };
        let rv = match self.nodes[node].right {
            Some(rn) => self.query_rec(rn, mid + 1, end, l, r),
            None => SegmentTree::identity(self.op),
        };
        self.combine(lv, rv)
    }

    fn update_rec(&mut self, node: usize, start: usize, end: usize, pos: usize, value: i64) -> usize {
        if start == end {
            return self.new_node(value, None, None);
        }
        let mid = start + (end - start) / 2;
        let old_left = self.nodes[node].left;
        let old_right = self.nodes[node].right;

        let (new_left, new_right) = if pos <= mid {
            let nl = match old_left {
                Some(ln) => self.update_rec(ln, start, mid, pos, value),
                None => self.new_node(value, None, None),
            };
            (Some(nl), old_right)
        } else {
            let nr = match old_right {
                Some(rn) => self.update_rec(rn, mid + 1, end, pos, value),
                None => self.new_node(value, None, None),
            };
            (old_left, Some(nr))
        };

        let lv = new_left
            .map(|i| self.nodes[i].value)
            .unwrap_or_else(|| SegmentTree::identity(self.op));
        let rv = new_right
            .map(|i| self.nodes[i].value)
            .unwrap_or_else(|| SegmentTree::identity(self.op));
        let val = self.combine(lv, rv);
        self.new_node(val, new_left, new_right)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sum() {
        let data = vec![1, 3, 5, 7, 9, 11];
        let mut st = SegmentTree::build(&data, Operation::Sum);
        assert_eq!(st.query(0, 5), 36);
    }

    #[test]
    fn test_build_min() {
        let data = vec![5, 2, 8, 1, 9, 3];
        let mut st = SegmentTree::build(&data, Operation::Min);
        assert_eq!(st.query(0, 5), 1);
        assert_eq!(st.query(0, 2), 2);
        assert_eq!(st.query(3, 5), 1);
    }

    #[test]
    fn test_build_max() {
        let data = vec![5, 2, 8, 1, 9, 3];
        let mut st = SegmentTree::build(&data, Operation::Max);
        assert_eq!(st.query(0, 5), 9);
        assert_eq!(st.query(0, 2), 8);
    }

    #[test]
    fn test_point_update_sum() {
        let data = vec![1, 2, 3, 4, 5];
        let mut st = SegmentTree::build(&data, Operation::Sum);
        assert_eq!(st.query(0, 4), 15);
        st.point_update(2, 10);
        assert_eq!(st.query(0, 4), 22);
        assert_eq!(st.query(2, 2), 10);
    }

    #[test]
    fn test_point_update_min() {
        let data = vec![5, 3, 8, 1, 7];
        let mut st = SegmentTree::build(&data, Operation::Min);
        assert_eq!(st.query(0, 4), 1);
        st.point_update(3, 10);
        assert_eq!(st.query(0, 4), 3);
    }

    #[test]
    fn test_range_update_sum() {
        let data = vec![1, 2, 3, 4, 5];
        let mut st = SegmentTree::build(&data, Operation::Sum);
        st.range_update(1, 3, 10);
        // [1, 12, 13, 14, 5]
        assert_eq!(st.query(0, 4), 45);
        assert_eq!(st.query(1, 3), 39);
    }

    #[test]
    fn test_range_update_min() {
        let data = vec![10, 20, 30, 40, 50];
        let mut st = SegmentTree::build(&data, Operation::Min);
        st.range_update(0, 2, -15);
        // [-5, 5, 15, 40, 50]
        assert_eq!(st.query(0, 4), -5);
        assert_eq!(st.query(2, 4), 15);
    }

    #[test]
    fn test_empty_tree() {
        let mut st = SegmentTree::build(&[], Operation::Sum);
        assert!(st.is_empty());
        assert_eq!(st.len(), 0);
        assert_eq!(st.query(0, 0), 0);
    }

    #[test]
    fn test_single_element() {
        let mut st = SegmentTree::build(&[42], Operation::Sum);
        assert_eq!(st.len(), 1);
        assert_eq!(st.query(0, 0), 42);
        st.point_update(0, 99);
        assert_eq!(st.query(0, 0), 99);
    }

    #[test]
    fn test_subrange_queries() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut st = SegmentTree::build(&data, Operation::Sum);
        assert_eq!(st.query(2, 5), 18); // 3+4+5+6
        assert_eq!(st.query(0, 0), 1);
        assert_eq!(st.query(7, 7), 8);
        assert_eq!(st.query(3, 6), 22); // 4+5+6+7
    }

    #[test]
    fn test_persistent_build() {
        let data = vec![1, 2, 3, 4, 5];
        let pst = PersistentSegmentTree::build(&data, Operation::Sum);
        assert_eq!(pst.versions(), 1);
        assert_eq!(pst.query(0, 0, 4), 15);
    }

    #[test]
    fn test_persistent_update_creates_version() {
        let data = vec![1, 2, 3, 4, 5];
        let mut pst = PersistentSegmentTree::build(&data, Operation::Sum);
        let v1 = pst.update(0, 2, 10);
        assert_eq!(v1, 1);
        assert_eq!(pst.versions(), 2);

        // Old version unchanged
        assert_eq!(pst.query(0, 0, 4), 15);
        // New version updated
        assert_eq!(pst.query(1, 0, 4), 22);
    }

    #[test]
    fn test_persistent_multiple_versions() {
        let data = vec![1, 1, 1, 1];
        let mut pst = PersistentSegmentTree::build(&data, Operation::Sum);
        let v1 = pst.update(0, 0, 10); // version 1: [10,1,1,1]
        let v2 = pst.update(v1, 3, 10); // version 2: [10,1,1,10]
        let v3 = pst.update(0, 1, 5);   // version 3 from v0: [1,5,1,1]

        assert_eq!(pst.query(0, 0, 3), 4);
        assert_eq!(pst.query(v1, 0, 3), 13);
        assert_eq!(pst.query(v2, 0, 3), 22);
        assert_eq!(pst.query(v3, 0, 3), 8);
    }

    #[test]
    fn test_persistent_min() {
        let data = vec![5, 3, 8, 1, 7];
        let mut pst = PersistentSegmentTree::build(&data, Operation::Min);
        assert_eq!(pst.query(0, 0, 4), 1);

        let v1 = pst.update(0, 3, 10); // [5,3,8,10,7]
        assert_eq!(pst.query(v1, 0, 4), 3);
        assert_eq!(pst.query(0, 0, 4), 1); // original unchanged
    }

    #[test]
    fn test_multiple_range_updates() {
        let data = vec![0, 0, 0, 0, 0];
        let mut st = SegmentTree::build(&data, Operation::Sum);
        st.range_update(0, 4, 1);
        st.range_update(1, 3, 2);
        // [1, 3, 3, 3, 1]
        assert_eq!(st.query(0, 4), 11);
        assert_eq!(st.query(1, 1), 3);
    }

    #[test]
    fn test_out_of_bounds() {
        let data = vec![1, 2, 3];
        let mut st = SegmentTree::build(&data, Operation::Sum);
        // Query past end should return identity
        assert_eq!(st.query(0, 10), 0);
        // Point update past end should be a no-op
        st.point_update(10, 99);
        assert_eq!(st.query(0, 2), 6);
    }
}
