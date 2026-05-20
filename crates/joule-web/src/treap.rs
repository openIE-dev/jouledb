//! Treap (tree + heap) — randomized BST maintaining heap property on priorities.
//! Supports insert/delete with rotations, implicit treap for array operations,
//! split/merge, range reverse, kth element, and interval operations.
//!
//! Replaces JS treap/randomized BST libraries with a pure-Rust implementation.

use std::fmt;

// ── Explicit Treap ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TreapNode<K, V> {
    key: K,
    value: V,
    priority: u64,
    left: Option<usize>,
    right: Option<usize>,
    size: usize,
}

/// A treap (tree + heap) — randomized BST with expected O(log n) operations.
pub struct Treap<K: Ord, V> {
    arena: Vec<TreapNode<K, V>>,
    root: Option<usize>,
    len: usize,
    rng: u64,
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for Treap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Treap")
            .field("len", &self.len)
            .finish()
    }
}

impl<K: Ord, V> Default for Treap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> Treap<K, V> {
    /// Create an empty treap.
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            root: None,
            len: 0,
            rng: 0x1234_5678_9ABC_DEF0,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Search for a key.
    pub fn search(&self, key: &K) -> Option<&V> {
        let mut cur = self.root;
        while let Some(idx) = cur {
            let node = &self.arena[idx];
            match key.cmp(&node.key) {
                std::cmp::Ordering::Equal => return Some(&node.value),
                std::cmp::Ordering::Less => cur = node.left,
                std::cmp::Ordering::Greater => cur = node.right,
            }
        }
        None
    }

    /// Insert a key-value pair. Returns old value if key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check for existing key first
        if let Some(idx) = self.find_node(&key) {
            let old = std::mem::replace(&mut self.arena[idx].value, value);
            return Some(old);
        }

        let priority = self.next_random();
        let new_idx = self.arena.len();
        self.arena.push(TreapNode {
            key,
            value,
            priority,
            left: None,
            right: None,
            size: 1,
        });

        self.root = self.insert_rec(self.root, new_idx);
        self.len += 1;
        None
    }

    /// Delete a key. Returns true if found.
    pub fn delete(&mut self, key: &K) -> bool {
        let mut found = false;
        self.root = self.delete_rec(self.root, key, &mut found);
        if found {
            self.len -= 1;
        }
        found
    }

    /// Kth smallest element (0-indexed).
    pub fn kth_smallest(&self, k: usize) -> Option<(&K, &V)> {
        if k >= self.len {
            return None;
        }
        self.kth_rec(self.root, k)
    }

    /// In-order traversal.
    pub fn in_order(&self) -> Vec<(&K, &V)> {
        let mut result = Vec::with_capacity(self.len);
        self.in_order_rec(self.root, &mut result);
        result
    }

    /// Min key-value pair.
    pub fn min(&self) -> Option<(&K, &V)> {
        let idx = self.subtree_min(self.root?);
        Some((&self.arena[idx].key, &self.arena[idx].value))
    }

    /// Max key-value pair.
    pub fn max(&self) -> Option<(&K, &V)> {
        let idx = self.subtree_max(self.root?);
        Some((&self.arena[idx].key, &self.arena[idx].value))
    }

    // ── Internal ──

    fn next_random(&mut self) -> u64 {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn node_size(&self, node: Option<usize>) -> usize {
        node.map_or(0, |i| self.arena[i].size)
    }

    fn update_size(&mut self, idx: usize) {
        let ls = self.node_size(self.arena[idx].left);
        let rs = self.node_size(self.arena[idx].right);
        self.arena[idx].size = 1 + ls + rs;
    }

    fn find_node(&self, key: &K) -> Option<usize> {
        let mut cur = self.root;
        while let Some(idx) = cur {
            match key.cmp(&self.arena[idx].key) {
                std::cmp::Ordering::Equal => return Some(idx),
                std::cmp::Ordering::Less => cur = self.arena[idx].left,
                std::cmp::Ordering::Greater => cur = self.arena[idx].right,
            }
        }
        None
    }

    fn rotate_right(&mut self, y: usize) -> usize {
        let x = self.arena[y].left.unwrap();
        self.arena[y].left = self.arena[x].right;
        self.arena[x].right = Some(y);
        self.update_size(y);
        self.update_size(x);
        x
    }

    fn rotate_left(&mut self, x: usize) -> usize {
        let y = self.arena[x].right.unwrap();
        self.arena[x].right = self.arena[y].left;
        self.arena[y].left = Some(x);
        self.update_size(x);
        self.update_size(y);
        y
    }

    fn insert_rec(&mut self, node: Option<usize>, new_idx: usize) -> Option<usize> {
        let idx = match node {
            None => return Some(new_idx),
            Some(i) => i,
        };

        if self.arena[new_idx].key < self.arena[idx].key {
            self.arena[idx].left = self.insert_rec(self.arena[idx].left, new_idx);
            self.update_size(idx);
            if self.arena[self.arena[idx].left.unwrap()].priority > self.arena[idx].priority {
                return Some(self.rotate_right(idx));
            }
        } else {
            self.arena[idx].right = self.insert_rec(self.arena[idx].right, new_idx);
            self.update_size(idx);
            if self.arena[self.arena[idx].right.unwrap()].priority > self.arena[idx].priority {
                return Some(self.rotate_left(idx));
            }
        }

        Some(idx)
    }

    fn delete_rec(&mut self, node: Option<usize>, key: &K, found: &mut bool) -> Option<usize> {
        let idx = match node {
            None => return None,
            Some(i) => i,
        };

        match key.cmp(&self.arena[idx].key) {
            std::cmp::Ordering::Less => {
                self.arena[idx].left = self.delete_rec(self.arena[idx].left, key, found);
                self.update_size(idx);
                Some(idx)
            }
            std::cmp::Ordering::Greater => {
                self.arena[idx].right = self.delete_rec(self.arena[idx].right, key, found);
                self.update_size(idx);
                Some(idx)
            }
            std::cmp::Ordering::Equal => {
                *found = true;
                let left = self.arena[idx].left;
                let right = self.arena[idx].right;
                match (left, right) {
                    (None, None) => None,
                    (Some(_), None) => left,
                    (None, Some(_)) => right,
                    (Some(l), Some(r)) => {
                        if self.arena[l].priority > self.arena[r].priority {
                            let new_root = self.rotate_right(idx);
                            self.arena[new_root].right =
                                self.delete_rec(self.arena[new_root].right, key, &mut true);
                            self.update_size(new_root);
                            Some(new_root)
                        } else {
                            let new_root = self.rotate_left(idx);
                            self.arena[new_root].left =
                                self.delete_rec(self.arena[new_root].left, key, &mut true);
                            self.update_size(new_root);
                            Some(new_root)
                        }
                    }
                }
            }
        }
    }

    fn kth_rec(&self, node: Option<usize>, k: usize) -> Option<(&K, &V)> {
        let idx = node?;
        let left_size = self.node_size(self.arena[idx].left);
        if k < left_size {
            self.kth_rec(self.arena[idx].left, k)
        } else if k == left_size {
            Some((&self.arena[idx].key, &self.arena[idx].value))
        } else {
            self.kth_rec(self.arena[idx].right, k - left_size - 1)
        }
    }

    fn subtree_min(&self, mut idx: usize) -> usize {
        while let Some(l) = self.arena[idx].left {
            idx = l;
        }
        idx
    }

    fn subtree_max(&self, mut idx: usize) -> usize {
        while let Some(r) = self.arena[idx].right {
            idx = r;
        }
        idx
    }

    fn in_order_rec<'a>(&'a self, node: Option<usize>, out: &mut Vec<(&'a K, &'a V)>) {
        if let Some(idx) = node {
            self.in_order_rec(self.arena[idx].left, out);
            out.push((&self.arena[idx].key, &self.arena[idx].value));
            self.in_order_rec(self.arena[idx].right, out);
        }
    }
}

// ── Implicit Treap ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ImplicitNode {
    value: i64,
    priority: u64,
    left: Option<usize>,
    right: Option<usize>,
    size: usize,
    reversed: bool,
    sum: i64,
}

/// An implicit treap — treap indexed by position rather than key.
/// Supports array operations: insert at position, remove, reverse range,
/// kth element, and range sum.
pub struct ImplicitTreap {
    arena: Vec<ImplicitNode>,
    root: Option<usize>,
    rng: u64,
}

impl fmt::Debug for ImplicitTreap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImplicitTreap")
            .field("len", &self.len())
            .finish()
    }
}

impl Default for ImplicitTreap {
    fn default() -> Self {
        Self::new()
    }
}

impl ImplicitTreap {
    /// Create an empty implicit treap.
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            root: None,
            rng: 0xDEAD_BEEF_CAFE_BABE,
        }
    }

    /// Build from an array.
    pub fn from_array(data: &[i64]) -> Self {
        let mut treap = Self::new();
        for (i, val) in data.iter().enumerate() {
            treap.insert_at(i, *val);
        }
        treap
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.node_size(self.root)
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    /// Insert value at position `pos` (0-indexed).
    pub fn insert_at(&mut self, pos: usize, value: i64) {
        let priority = self.next_random();
        let new_idx = self.arena.len();
        self.arena.push(ImplicitNode {
            value,
            priority,
            left: None,
            right: None,
            size: 1,
            reversed: false,
            sum: value,
        });

        let (left, right) = self.split(self.root, pos);
        let merged_left = self.merge(left, Some(new_idx));
        self.root = self.merge(merged_left, right);
    }

    /// Remove element at position `pos`. Returns the removed value.
    pub fn remove_at(&mut self, pos: usize) -> Option<i64> {
        if pos >= self.len() {
            return None;
        }
        let (left, rest) = self.split(self.root, pos);
        let (mid, right) = self.split(rest, 1);
        self.root = self.merge(left, right);
        mid.map(|idx| self.arena[idx].value)
    }

    /// Get value at position.
    pub fn get(&mut self, pos: usize) -> Option<i64> {
        if pos >= self.len() {
            return None;
        }
        let (left, rest) = self.split(self.root, pos);
        let (mid, right) = self.split(rest, 1);
        let val = mid.map(|idx| self.arena[idx].value);
        let merged_left = self.merge(left, mid);
        self.root = self.merge(merged_left, right);
        val
    }

    /// Reverse the elements in [l, r] (0-indexed, inclusive).
    pub fn reverse_range(&mut self, l: usize, r: usize) {
        if l > r || r >= self.len() {
            return;
        }
        let (left, rest) = self.split(self.root, l);
        let (mid, right) = self.split(rest, r - l + 1);
        if let Some(m) = mid {
            self.arena[m].reversed = !self.arena[m].reversed;
        }
        let merged_left = self.merge(left, mid);
        self.root = self.merge(merged_left, right);
    }

    /// Sum of elements in [l, r] (0-indexed, inclusive).
    pub fn range_sum(&mut self, l: usize, r: usize) -> i64 {
        if l > r || r >= self.len() {
            return 0;
        }
        let (left, rest) = self.split(self.root, l);
        let (mid, right) = self.split(rest, r - l + 1);
        let sum = mid.map_or(0, |m| self.arena[m].sum);
        let merged_left = self.merge(left, mid);
        self.root = self.merge(merged_left, right);
        sum
    }

    /// Collect all elements in order.
    pub fn to_vec(&mut self) -> Vec<i64> {
        let mut result = Vec::with_capacity(self.len());
        self.collect_rec(self.root, &mut result);
        result
    }

    // ── Internal ──

    fn next_random(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn node_size(&self, node: Option<usize>) -> usize {
        node.map_or(0, |i| self.arena[i].size)
    }

    fn update(&mut self, idx: usize) {
        let ls = self.node_size(self.arena[idx].left);
        let rs = self.node_size(self.arena[idx].right);
        self.arena[idx].size = 1 + ls + rs;
        let lsum = self.arena[idx].left.map_or(0, |l| self.arena[l].sum);
        let rsum = self.arena[idx].right.map_or(0, |r| self.arena[r].sum);
        self.arena[idx].sum = self.arena[idx].value + lsum + rsum;
    }

    fn push_down(&mut self, idx: usize) {
        if self.arena[idx].reversed {
            let left = self.arena[idx].left;
            let right = self.arena[idx].right;
            self.arena[idx].left = right;
            self.arena[idx].right = left;
            if let Some(l) = self.arena[idx].left {
                self.arena[l].reversed = !self.arena[l].reversed;
            }
            if let Some(r) = self.arena[idx].right {
                self.arena[r].reversed = !self.arena[r].reversed;
            }
            self.arena[idx].reversed = false;
        }
    }

    fn split(&mut self, node: Option<usize>, count: usize) -> (Option<usize>, Option<usize>) {
        let idx = match node {
            None => return (None, None),
            Some(i) => i,
        };
        self.push_down(idx);
        let left_size = self.node_size(self.arena[idx].left);

        if count <= left_size {
            let (ll, lr) = self.split(self.arena[idx].left, count);
            self.arena[idx].left = lr;
            self.update(idx);
            (ll, Some(idx))
        } else {
            let (rl, rr) = self.split(self.arena[idx].right, count - left_size - 1);
            self.arena[idx].right = rl;
            self.update(idx);
            (Some(idx), rr)
        }
    }

    fn merge(&mut self, left: Option<usize>, right: Option<usize>) -> Option<usize> {
        match (left, right) {
            (None, r) => r,
            (l, None) => l,
            (Some(l), Some(r)) => {
                self.push_down(l);
                self.push_down(r);
                if self.arena[l].priority > self.arena[r].priority {
                    self.arena[l].right = self.merge(self.arena[l].right, Some(r));
                    self.update(l);
                    Some(l)
                } else {
                    self.arena[r].left = self.merge(Some(l), self.arena[r].left);
                    self.update(r);
                    Some(r)
                }
            }
        }
    }

    fn collect_rec(&mut self, node: Option<usize>, out: &mut Vec<i64>) {
        if let Some(idx) = node {
            self.push_down(idx);
            let left = self.arena[idx].left;
            let right = self.arena[idx].right;
            self.collect_rec(left, out);
            out.push(self.arena[idx].value);
            self.collect_rec(right, out);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Explicit Treap tests ──

    #[test]
    fn test_treap_new() {
        let tree: Treap<i32, i32> = Treap::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_treap_insert_search() {
        let mut tree = Treap::new();
        tree.insert(10, 100);
        tree.insert(5, 50);
        tree.insert(15, 150);
        assert_eq!(tree.search(&10), Some(&100));
        assert_eq!(tree.search(&5), Some(&50));
        assert_eq!(tree.search(&99), None);
    }

    #[test]
    fn test_treap_insert_duplicate() {
        let mut tree = Treap::new();
        tree.insert(5, 50);
        let old = tree.insert(5, 99);
        assert_eq!(old, Some(50));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_treap_delete() {
        let mut tree = Treap::new();
        for i in 0..10 {
            tree.insert(i, i);
        }
        assert!(tree.delete(&5));
        assert_eq!(tree.len(), 9);
        assert_eq!(tree.search(&5), None);
    }

    #[test]
    fn test_treap_delete_nonexistent() {
        let mut tree = Treap::new();
        tree.insert(1, 1);
        assert!(!tree.delete(&99));
    }

    #[test]
    fn test_treap_in_order() {
        let mut tree = Treap::new();
        for v in [50, 30, 70, 10, 40, 60, 80] {
            tree.insert(v, v);
        }
        let keys: Vec<i32> = tree.in_order().iter().map(|(k, _)| **k).collect();
        assert_eq!(keys, vec![10, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn test_treap_kth() {
        let mut tree = Treap::new();
        for v in [50, 30, 70, 10, 40] {
            tree.insert(v, v);
        }
        assert_eq!(tree.kth_smallest(0), Some((&10, &10)));
        assert_eq!(tree.kth_smallest(2), Some((&40, &40)));
        assert_eq!(tree.kth_smallest(4), Some((&70, &70)));
        assert_eq!(tree.kth_smallest(5), None);
    }

    #[test]
    fn test_treap_min_max() {
        let mut tree = Treap::new();
        assert!(tree.min().is_none());
        tree.insert(30, 3);
        tree.insert(10, 1);
        tree.insert(50, 5);
        assert_eq!(tree.min(), Some((&10, &1)));
        assert_eq!(tree.max(), Some((&50, &5)));
    }

    #[test]
    fn test_treap_many_inserts() {
        let mut tree = Treap::new();
        for i in 0..200 {
            tree.insert(i, i);
        }
        assert_eq!(tree.len(), 200);
        for i in 0..200 {
            assert_eq!(tree.search(&i), Some(&i));
        }
    }

    #[test]
    fn test_treap_default() {
        let tree: Treap<i32, i32> = Treap::default();
        assert!(tree.is_empty());
    }

    // ── Implicit Treap tests ──

    #[test]
    fn test_implicit_new() {
        let treap = ImplicitTreap::new();
        assert!(treap.is_empty());
        assert_eq!(treap.len(), 0);
    }

    #[test]
    fn test_implicit_from_array() {
        let mut treap = ImplicitTreap::from_array(&[1, 2, 3, 4, 5]);
        assert_eq!(treap.len(), 5);
        assert_eq!(treap.to_vec(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_implicit_insert_at() {
        let mut treap = ImplicitTreap::new();
        treap.insert_at(0, 10);
        treap.insert_at(1, 30);
        treap.insert_at(1, 20); // insert between 10 and 30
        assert_eq!(treap.to_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn test_implicit_remove_at() {
        let mut treap = ImplicitTreap::from_array(&[10, 20, 30, 40, 50]);
        assert_eq!(treap.remove_at(2), Some(30));
        assert_eq!(treap.to_vec(), vec![10, 20, 40, 50]);
    }

    #[test]
    fn test_implicit_get() {
        let mut treap = ImplicitTreap::from_array(&[10, 20, 30]);
        assert_eq!(treap.get(0), Some(10));
        assert_eq!(treap.get(1), Some(20));
        assert_eq!(treap.get(2), Some(30));
        assert_eq!(treap.get(3), None);
    }

    #[test]
    fn test_implicit_reverse_range() {
        let mut treap = ImplicitTreap::from_array(&[1, 2, 3, 4, 5]);
        treap.reverse_range(1, 3); // reverse [2,3,4] -> [4,3,2]
        assert_eq!(treap.to_vec(), vec![1, 4, 3, 2, 5]);
    }

    #[test]
    fn test_implicit_reverse_full() {
        let mut treap = ImplicitTreap::from_array(&[1, 2, 3, 4]);
        treap.reverse_range(0, 3);
        assert_eq!(treap.to_vec(), vec![4, 3, 2, 1]);
    }

    #[test]
    fn test_implicit_range_sum() {
        let mut treap = ImplicitTreap::from_array(&[1, 2, 3, 4, 5]);
        assert_eq!(treap.range_sum(0, 4), 15);
        assert_eq!(treap.range_sum(1, 3), 9);
        assert_eq!(treap.range_sum(2, 2), 3);
    }

    #[test]
    fn test_implicit_default() {
        let treap = ImplicitTreap::default();
        assert!(treap.is_empty());
    }
}
