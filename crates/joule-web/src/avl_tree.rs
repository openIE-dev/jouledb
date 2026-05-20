//! AVL tree — height-balanced binary search tree with O(log n) insert, delete,
//! search, range query, floor/ceiling, and rank operations.
//!
//! Replaces JS balanced-tree libraries with a pure-Rust arena-based
//! implementation that guarantees height balance (|balance factor| <= 1).

use std::fmt;

// ── Node ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
    left: Option<usize>,
    right: Option<usize>,
    height: i32,
    size: usize, // subtree size for rank queries
}

// ── AvlTree ────────────────────────────────────────────────────────────────

/// A height-balanced AVL tree with O(log n) operations and rank support.
pub struct AvlTree<K: Ord, V> {
    arena: Vec<Node<K, V>>,
    root: Option<usize>,
    len: usize,
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for AvlTree<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AvlTree")
            .field("len", &self.len)
            .finish()
    }
}

impl<K: Ord, V> Default for AvlTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> AvlTree<K, V> {
    /// Create an empty AVL tree.
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            root: None,
            len: 0,
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

    /// Check if a key exists.
    pub fn contains(&self, key: &K) -> bool {
        self.search(key).is_some()
    }

    /// Insert a key-value pair. Returns old value if key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let mut old = None;
        self.root = self.insert_rec(self.root, key, value, &mut old);
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    /// Delete a key. Returns true if the key was found and removed.
    pub fn delete(&mut self, key: &K) -> bool {
        let mut found = false;
        self.root = self.delete_rec(self.root, key, &mut found);
        if found {
            self.len -= 1;
        }
        found
    }

    /// In-order traversal returning key-value pairs in sorted order.
    pub fn in_order(&self) -> Vec<(&K, &V)> {
        let mut result = Vec::with_capacity(self.len);
        self.in_order_rec(self.root, &mut result);
        result
    }

    /// Range query: all entries with keys in [lo, hi].
    pub fn range(&self, lo: &K, hi: &K) -> Vec<(&K, &V)> {
        let mut result = Vec::new();
        self.range_rec(self.root, lo, hi, &mut result);
        result
    }

    /// Floor: largest key <= given key.
    pub fn floor(&self, key: &K) -> Option<(&K, &V)> {
        let mut best: Option<usize> = None;
        let mut cur = self.root;
        while let Some(idx) = cur {
            let node = &self.arena[idx];
            match key.cmp(&node.key) {
                std::cmp::Ordering::Equal => return Some((&node.key, &node.value)),
                std::cmp::Ordering::Greater => {
                    best = Some(idx);
                    cur = node.right;
                }
                std::cmp::Ordering::Less => {
                    cur = node.left;
                }
            }
        }
        best.map(|i| (&self.arena[i].key, &self.arena[i].value))
    }

    /// Ceiling: smallest key >= given key.
    pub fn ceiling(&self, key: &K) -> Option<(&K, &V)> {
        let mut best: Option<usize> = None;
        let mut cur = self.root;
        while let Some(idx) = cur {
            let node = &self.arena[idx];
            match key.cmp(&node.key) {
                std::cmp::Ordering::Equal => return Some((&node.key, &node.value)),
                std::cmp::Ordering::Less => {
                    best = Some(idx);
                    cur = node.left;
                }
                std::cmp::Ordering::Greater => {
                    cur = node.right;
                }
            }
        }
        best.map(|i| (&self.arena[i].key, &self.arena[i].value))
    }

    /// Rank: find the kth smallest element (0-indexed).
    pub fn kth_smallest(&self, k: usize) -> Option<(&K, &V)> {
        if k >= self.len {
            return None;
        }
        self.kth_rec(self.root, k)
    }

    /// Rank of a key (number of keys strictly less than `key`).
    pub fn rank(&self, key: &K) -> usize {
        self.rank_rec(self.root, key)
    }

    /// Validate AVL property: |balance factor| <= 1 for all nodes.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_rec(self.root)?;
        Ok(())
    }

    /// Height of the tree.
    pub fn height(&self) -> i32 {
        self.node_height(self.root)
    }

    // ── Internal helpers ──

    fn node_height(&self, node: Option<usize>) -> i32 {
        match node {
            None => 0,
            Some(idx) => self.arena[idx].height,
        }
    }

    fn node_size(&self, node: Option<usize>) -> usize {
        match node {
            None => 0,
            Some(idx) => self.arena[idx].size,
        }
    }

    fn update_height_and_size(&mut self, idx: usize) {
        let lh = self.node_height(self.arena[idx].left);
        let rh = self.node_height(self.arena[idx].right);
        self.arena[idx].height = 1 + lh.max(rh);
        let ls = self.node_size(self.arena[idx].left);
        let rs = self.node_size(self.arena[idx].right);
        self.arena[idx].size = 1 + ls + rs;
    }

    fn balance_factor(&self, idx: usize) -> i32 {
        let lh = self.node_height(self.arena[idx].left);
        let rh = self.node_height(self.arena[idx].right);
        lh - rh
    }

    fn rotate_right(&mut self, y: usize) -> usize {
        let x = self.arena[y].left.unwrap();
        let t2 = self.arena[x].right;
        self.arena[x].right = Some(y);
        self.arena[y].left = t2;
        self.update_height_and_size(y);
        self.update_height_and_size(x);
        x
    }

    fn rotate_left(&mut self, x: usize) -> usize {
        let y = self.arena[x].right.unwrap();
        let t2 = self.arena[y].left;
        self.arena[y].left = Some(x);
        self.arena[x].right = t2;
        self.update_height_and_size(x);
        self.update_height_and_size(y);
        y
    }

    fn rebalance(&mut self, idx: usize) -> usize {
        self.update_height_and_size(idx);
        let bf = self.balance_factor(idx);

        if bf > 1 {
            let left = self.arena[idx].left.unwrap();
            if self.balance_factor(left) < 0 {
                // Left-Right case
                let new_left = self.rotate_left(left);
                self.arena[idx].left = Some(new_left);
            }
            return self.rotate_right(idx);
        }

        if bf < -1 {
            let right = self.arena[idx].right.unwrap();
            if self.balance_factor(right) > 0 {
                // Right-Left case
                let new_right = self.rotate_right(right);
                self.arena[idx].right = Some(new_right);
            }
            return self.rotate_left(idx);
        }

        idx
    }

    fn insert_rec(&mut self, node: Option<usize>, key: K, value: V, old: &mut Option<V>) -> Option<usize> {
        let idx = match node {
            None => {
                let i = self.arena.len();
                self.arena.push(Node {
                    key,
                    value,
                    left: None,
                    right: None,
                    height: 1,
                    size: 1,
                });
                return Some(i);
            }
            Some(i) => i,
        };

        match key.cmp(&self.arena[idx].key) {
            std::cmp::Ordering::Equal => {
                *old = Some(std::mem::replace(&mut self.arena[idx].value, value));
                return Some(idx);
            }
            std::cmp::Ordering::Less => {
                let new_left = self.insert_rec(self.arena[idx].left, key, value, old);
                self.arena[idx].left = new_left;
            }
            std::cmp::Ordering::Greater => {
                let new_right = self.insert_rec(self.arena[idx].right, key, value, old);
                self.arena[idx].right = new_right;
            }
        }

        Some(self.rebalance(idx))
    }

    fn delete_rec(&mut self, node: Option<usize>, key: &K, found: &mut bool) -> Option<usize> {
        let idx = match node {
            None => return None,
            Some(i) => i,
        };

        match key.cmp(&self.arena[idx].key) {
            std::cmp::Ordering::Less => {
                let new_left = self.delete_rec(self.arena[idx].left, key, found);
                self.arena[idx].left = new_left;
            }
            std::cmp::Ordering::Greater => {
                let new_right = self.delete_rec(self.arena[idx].right, key, found);
                self.arena[idx].right = new_right;
            }
            std::cmp::Ordering::Equal => {
                *found = true;
                let left = self.arena[idx].left;
                let right = self.arena[idx].right;

                if left.is_none() {
                    return right;
                }
                if right.is_none() {
                    return left;
                }

                // Find in-order successor (min of right subtree)
                let succ = self.find_min(right.unwrap());
                // Remove successor from right subtree
                let succ_key_matches = |this: &Self, check: &K| -> bool {
                    this.arena[succ].key == *check
                };
                let sk = succ_key_matches(self, &self.arena[succ].key);
                if sk {
                    let mut dummy = false;
                    let succ_key = unsafe {
                        // SAFETY: we're reading the key to pass to delete_rec,
                        // which operates on the right subtree not containing idx
                        &*(&self.arena[succ].key as *const K)
                    };
                    let new_right = self.delete_rec(right, succ_key, &mut dummy);
                    self.arena[idx].right = new_right;
                }

                // Swap data from succ to idx
                // We can't easily swap because succ might already be gone from tree.
                // Instead we copy succ's key/value to idx's slot via swap in the arena.
                // succ's slot is now orphaned.
                self.arena.swap(idx, succ);
                // After swap, 'idx' position has succ's data, succ position has old data.
                // Fix up children: idx should keep the children of the *old* idx (now at succ pos).
                let old_left = self.arena[succ].left;
                let old_right = self.arena[succ].right;
                self.arena[idx].left = old_left;
                self.arena[idx].right = old_right;
            }
        }

        Some(self.rebalance(idx))
    }

    fn find_min(&self, mut idx: usize) -> usize {
        while let Some(l) = self.arena[idx].left {
            idx = l;
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

    fn range_rec<'a>(&'a self, node: Option<usize>, lo: &K, hi: &K, out: &mut Vec<(&'a K, &'a V)>) {
        if let Some(idx) = node {
            let node = &self.arena[idx];
            if node.key >= *lo {
                self.range_rec(node.left, lo, hi, out);
            }
            if node.key >= *lo && node.key <= *hi {
                out.push((&node.key, &node.value));
            }
            if node.key <= *hi {
                self.range_rec(node.right, lo, hi, out);
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

    fn rank_rec(&self, node: Option<usize>, key: &K) -> usize {
        let idx = match node {
            None => return 0,
            Some(i) => i,
        };
        match key.cmp(&self.arena[idx].key) {
            std::cmp::Ordering::Less => self.rank_rec(self.arena[idx].left, key),
            std::cmp::Ordering::Equal => self.node_size(self.arena[idx].left),
            std::cmp::Ordering::Greater => {
                1 + self.node_size(self.arena[idx].left) + self.rank_rec(self.arena[idx].right, key)
            }
        }
    }

    fn validate_rec(&self, node: Option<usize>) -> Result<i32, String> {
        let idx = match node {
            None => return Ok(0),
            Some(i) => i,
        };

        let lh = self.validate_rec(self.arena[idx].left)?;
        let rh = self.validate_rec(self.arena[idx].right)?;
        let bf = lh - rh;

        if bf.abs() > 1 {
            return Err(format!(
                "Balance factor {} at key {:?} violates AVL property",
                bf, idx
            ));
        }

        let expected_h = 1 + lh.max(rh);
        if self.arena[idx].height != expected_h {
            return Err(format!(
                "Height mismatch at {}: stored={}, computed={}",
                idx, self.arena[idx].height, expected_h
            ));
        }

        Ok(expected_h)
    }
}

/// Iterator that yields key-value pairs in sorted order.
pub struct AvlInOrderIter<'a, K: Ord, V> {
    tree: &'a AvlTree<K, V>,
    stack: Vec<usize>,
}

impl<'a, K: Ord, V> AvlInOrderIter<'a, K, V> {
    fn new(tree: &'a AvlTree<K, V>) -> Self {
        let mut iter = Self {
            tree,
            stack: Vec::new(),
        };
        iter.push_left(tree.root);
        iter
    }

    fn push_left(&mut self, mut node: Option<usize>) {
        while let Some(idx) = node {
            self.stack.push(idx);
            node = self.tree.arena[idx].left;
        }
    }
}

impl<'a, K: Ord, V> Iterator for AvlInOrderIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.stack.pop()?;
        self.push_left(self.tree.arena[idx].right);
        Some((&self.tree.arena[idx].key, &self.tree.arena[idx].value))
    }
}

impl<K: Ord, V> AvlTree<K, V> {
    /// Create an in-order iterator.
    pub fn iter(&self) -> AvlInOrderIter<'_, K, V> {
        AvlInOrderIter::new(self)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tree() {
        let tree: AvlTree<i32, i32> = AvlTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn test_insert_single() {
        let mut tree = AvlTree::new();
        assert!(tree.insert(10, 100).is_none());
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.search(&10), Some(&100));
    }

    #[test]
    fn test_insert_duplicate() {
        let mut tree = AvlTree::new();
        tree.insert(5, 50);
        let old = tree.insert(5, 99);
        assert_eq!(old, Some(50));
        assert_eq!(tree.search(&5), Some(&99));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_insert_many() {
        let mut tree = AvlTree::new();
        for i in 0..100 {
            tree.insert(i, i * 2);
        }
        assert_eq!(tree.len(), 100);
        for i in 0..100 {
            assert_eq!(tree.search(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_avl_balance_after_inserts() {
        let mut tree = AvlTree::new();
        // Ascending inserts (worst case for unbalanced BST)
        for i in 0..63 {
            tree.insert(i, i);
            assert!(tree.validate().is_ok(), "Failed at insert {}", i);
        }
        // AVL height <= 1.44 * log2(n+2) - 0.328
        assert!(tree.height() <= 10);
    }

    #[test]
    fn test_delete_leaf() {
        let mut tree = AvlTree::new();
        tree.insert(20, 20);
        tree.insert(10, 10);
        tree.insert(30, 30);
        assert!(tree.delete(&30));
        assert_eq!(tree.len(), 2);
        assert!(!tree.contains(&30));
        assert!(tree.validate().is_ok());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut tree = AvlTree::new();
        tree.insert(10, 10);
        assert!(!tree.delete(&99));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_in_order() {
        let mut tree = AvlTree::new();
        for v in [50, 30, 70, 20, 40, 60, 80] {
            tree.insert(v, v);
        }
        let keys: Vec<i32> = tree.in_order().iter().map(|(k, _)| **k).collect();
        assert_eq!(keys, vec![20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn test_iterator() {
        let mut tree = AvlTree::new();
        for v in [5, 3, 7, 1, 4, 6, 8] {
            tree.insert(v, v * 10);
        }
        let pairs: Vec<_> = tree.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(pairs, vec![(1, 10), (3, 30), (4, 40), (5, 50), (6, 60), (7, 70), (8, 80)]);
    }

    #[test]
    fn test_range_query() {
        let mut tree = AvlTree::new();
        for i in 0..20 {
            tree.insert(i * 5, i);
        }
        let range: Vec<i32> = tree.range(&20, &50).iter().map(|(k, _)| **k).collect();
        assert_eq!(range, vec![20, 25, 30, 35, 40, 45, 50]);
    }

    #[test]
    fn test_floor() {
        let mut tree = AvlTree::new();
        for v in [10, 20, 30, 40, 50] {
            tree.insert(v, v);
        }
        assert_eq!(tree.floor(&25), Some((&20, &20)));
        assert_eq!(tree.floor(&30), Some((&30, &30)));
        assert_eq!(tree.floor(&5), None);
    }

    #[test]
    fn test_ceiling() {
        let mut tree = AvlTree::new();
        for v in [10, 20, 30, 40, 50] {
            tree.insert(v, v);
        }
        assert_eq!(tree.ceiling(&25), Some((&30, &30)));
        assert_eq!(tree.ceiling(&30), Some((&30, &30)));
        assert_eq!(tree.ceiling(&55), None);
    }

    #[test]
    fn test_kth_smallest() {
        let mut tree = AvlTree::new();
        for v in [50, 30, 70, 10, 40, 60, 80] {
            tree.insert(v, v);
        }
        assert_eq!(tree.kth_smallest(0), Some((&10, &10)));
        assert_eq!(tree.kth_smallest(3), Some((&50, &50)));
        assert_eq!(tree.kth_smallest(6), Some((&80, &80)));
        assert_eq!(tree.kth_smallest(7), None);
    }

    #[test]
    fn test_rank() {
        let mut tree = AvlTree::new();
        for v in [10, 20, 30, 40, 50] {
            tree.insert(v, v);
        }
        assert_eq!(tree.rank(&10), 0);
        assert_eq!(tree.rank(&30), 2);
        assert_eq!(tree.rank(&50), 4);
        assert_eq!(tree.rank(&25), 2);
    }

    #[test]
    fn test_contains() {
        let mut tree = AvlTree::new();
        tree.insert(42, "answer");
        assert!(tree.contains(&42));
        assert!(!tree.contains(&43));
    }

    #[test]
    fn test_default() {
        let tree: AvlTree<i32, i32> = AvlTree::default();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_delete_all() {
        let mut tree = AvlTree::new();
        let keys: Vec<i32> = (0..20).collect();
        for k in &keys {
            tree.insert(*k, *k);
        }
        for k in &keys {
            assert!(tree.delete(k));
            assert!(tree.validate().is_ok());
        }
        assert!(tree.is_empty());
    }
}
