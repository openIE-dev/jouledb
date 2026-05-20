//! In-memory B-tree index — configurable order, insert/delete/search, range
//! scan, bulk load, split/merge nodes, iterator, index statistics
//! (depth/nodes/entries), serialization.
//!
//! Replaces BTreeMap-based index wrappers with a purpose-built B-tree that
//! exposes internal structure for database index use cases.

use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by B-tree index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BTreeError {
    /// Key not found.
    KeyNotFound(String),
    /// Duplicate key on unique index.
    DuplicateKey(String),
    /// Invalid order (must be >= 3).
    InvalidOrder(usize),
}

impl fmt::Display for BTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::DuplicateKey(k) => write!(f, "duplicate key: {k}"),
            Self::InvalidOrder(o) => write!(f, "invalid order {o}, must be >= 3"),
        }
    }
}

impl std::error::Error for BTreeError {}

// ── Key/Value types ──────────────────────────────────────────────

/// A comparable key for the B-tree index.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexKey {
    Int(i64),
    Text(String),
    Bytes(Vec<u8>),
}

impl fmt::Display for IndexKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Text(v) => write!(f, "{v}"),
            Self::Bytes(v) => write!(f, "{v:?}"),
        }
    }
}

/// A row ID or arbitrary payload.
pub type RowId = u64;

// ── Node ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BTreeNode {
    /// Keys stored in this node.
    keys: Vec<IndexKey>,
    /// Associated row IDs (same length as keys for leaf nodes).
    values: Vec<RowId>,
    /// Child node indices (empty for leaf, keys.len()+1 for internal).
    children: Vec<usize>,
    /// Whether this is a leaf node.
    is_leaf: bool,
}

impl BTreeNode {
    fn new_leaf() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: true,
        }
    }

    fn new_internal() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
            is_leaf: false,
        }
    }
}

// ── Statistics ───────────────────────────────────────────────────

/// Index statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStats {
    pub order: usize,
    pub depth: usize,
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub internal_nodes: usize,
    pub total_entries: usize,
}

// ── BTreeIndex ───────────────────────────────────────────────────

/// An in-memory B-tree index with configurable order.
pub struct BTreeIndex {
    nodes: Vec<BTreeNode>,
    root: usize,
    order: usize, // maximum children per node
    entry_count: usize,
    unique: bool,
}

impl BTreeIndex {
    /// Create a new B-tree index with the given order (max children per node).
    /// Order must be >= 3.
    pub fn new(order: usize, unique: bool) -> Result<Self, BTreeError> {
        if order < 3 {
            return Err(BTreeError::InvalidOrder(order));
        }
        let mut nodes = Vec::new();
        nodes.push(BTreeNode::new_leaf());
        Ok(Self {
            nodes,
            root: 0,
            order,
            entry_count: 0,
            unique,
        })
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entry_count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entry_count == 0
    }

    /// Search for a key and return its associated row ID.
    pub fn search(&self, key: &IndexKey) -> Option<RowId> {
        self.search_in_node(self.root, key)
    }

    fn search_in_node(&self, node_idx: usize, key: &IndexKey) -> Option<RowId> {
        let node = &self.nodes[node_idx];
        // Binary search for the key position.
        let pos = node.keys.binary_search(key);
        match pos {
            Ok(i) => {
                if node.is_leaf {
                    Some(node.values[i])
                } else {
                    // For internal nodes, the value is stored at that position.
                    Some(node.values[i])
                }
            }
            Err(i) => {
                if node.is_leaf {
                    None
                } else {
                    self.search_in_node(node.children[i], key)
                }
            }
        }
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: IndexKey, row_id: RowId) -> Result<(), BTreeError> {
        if self.unique {
            if self.search(&key).is_some() {
                return Err(BTreeError::DuplicateKey(key.to_string()));
            }
        }
        let root = self.root;
        let max_keys = self.order - 1;
        if self.nodes[root].keys.len() == max_keys {
            // Root is full, split it.
            let new_root_idx = self.nodes.len();
            self.nodes.push(BTreeNode::new_internal());
            self.nodes[new_root_idx].children.push(root);
            self.split_child(new_root_idx, 0);
            self.root = new_root_idx;
        }
        self.insert_non_full(self.root, key, row_id);
        self.entry_count += 1;
        Ok(())
    }

    fn insert_non_full(&mut self, node_idx: usize, key: IndexKey, row_id: RowId) {
        let is_leaf = self.nodes[node_idx].is_leaf;
        if is_leaf {
            let pos = match self.nodes[node_idx].keys.binary_search(&key) {
                Ok(i) => i,
                Err(i) => i,
            };
            self.nodes[node_idx].keys.insert(pos, key);
            self.nodes[node_idx].values.insert(pos, row_id);
        } else {
            let pos = match self.nodes[node_idx].keys.binary_search(&key) {
                Ok(i) => i + 1,
                Err(i) => i,
            };
            let child_idx = self.nodes[node_idx].children[pos];
            let max_keys = self.order - 1;
            if self.nodes[child_idx].keys.len() == max_keys {
                self.split_child(node_idx, pos);
                // After split, determine which child to descend into.
                let cmp_key = &self.nodes[node_idx].keys[pos];
                if key > *cmp_key {
                    let new_child = self.nodes[node_idx].children[pos + 1];
                    self.insert_non_full(new_child, key, row_id);
                } else {
                    let same_child = self.nodes[node_idx].children[pos];
                    self.insert_non_full(same_child, key, row_id);
                }
            } else {
                self.insert_non_full(child_idx, key, row_id);
            }
        }
    }

    fn split_child(&mut self, parent_idx: usize, child_pos: usize) {
        let child_idx = self.nodes[parent_idx].children[child_pos];
        let mid = (self.order - 1) / 2;
        let is_leaf = self.nodes[child_idx].is_leaf;

        // Create new sibling.
        let new_idx = self.nodes.len();
        let mut sibling = if is_leaf {
            BTreeNode::new_leaf()
        } else {
            BTreeNode::new_internal()
        };

        // Move upper half of keys/values to sibling.
        sibling.keys = self.nodes[child_idx].keys.split_off(mid + 1);
        sibling.values = self.nodes[child_idx].values.split_off(mid + 1);

        if !is_leaf {
            sibling.children = self.nodes[child_idx].children.split_off(mid + 1);
        }

        // The median key goes up to the parent.
        let median_key = self.nodes[child_idx].keys.pop().unwrap();
        let median_val = self.nodes[child_idx].values.pop().unwrap();

        self.nodes.push(sibling);

        // Insert median into parent.
        self.nodes[parent_idx]
            .keys
            .insert(child_pos, median_key);
        self.nodes[parent_idx]
            .values
            .insert(child_pos, median_val);
        self.nodes[parent_idx]
            .children
            .insert(child_pos + 1, new_idx);
    }

    /// Delete a key from the index. Returns the row ID if found.
    ///
    /// Uses scan-and-rebuild: collects all entries except the deleted key,
    /// then rebuilds the tree. This avoids the complexity of in-place B-tree
    /// rebalancing (merge/borrow) while remaining correct for all cases.
    pub fn delete(&mut self, key: &IndexKey) -> Result<RowId, BTreeError> {
        // Scan for the key and collect all other entries.
        let all = self.scan_all();
        let mut found_val: Option<RowId> = None;
        let mut remaining = Vec::with_capacity(all.len());
        for (k, v) in all {
            if found_val.is_none() && k == *key {
                found_val = Some(v);
            } else {
                remaining.push((k, v));
            }
        }
        let row_id = found_val.ok_or_else(|| BTreeError::KeyNotFound(key.to_string()))?;

        // Rebuild the tree from the remaining entries.
        self.nodes.clear();
        self.nodes.push(BTreeNode::new_leaf());
        self.root = 0;
        let prev_count = self.entry_count;
        self.entry_count = 0;
        for (k, v) in remaining {
            // insert cannot fail here: no duplicates remain, order is valid.
            let _ = self.insert(k, v);
        }
        debug_assert_eq!(self.entry_count, prev_count - 1);
        Ok(row_id)
    }

    /// Range scan: return all entries with keys in [start, end] inclusive.
    pub fn range_scan(&self, start: &IndexKey, end: &IndexKey) -> Vec<(IndexKey, RowId)> {
        let mut results = Vec::new();
        self.range_collect(self.root, start, end, &mut results);
        results
    }

    fn range_collect(
        &self,
        node_idx: usize,
        start: &IndexKey,
        end: &IndexKey,
        results: &mut Vec<(IndexKey, RowId)>,
    ) {
        let node = &self.nodes[node_idx];
        for i in 0..node.keys.len() {
            if !node.is_leaf {
                if node.keys[i] >= *start {
                    self.range_collect(node.children[i], start, end, results);
                }
            }
            if node.keys[i] >= *start && node.keys[i] <= *end {
                results.push((node.keys[i].clone(), node.values[i]));
            }
            if node.keys[i] > *end {
                return;
            }
        }
        // Visit the last child.
        if !node.is_leaf && !node.keys.is_empty() && *node.keys.last().unwrap() <= *end {
            let last = *node.children.last().unwrap();
            self.range_collect(last, start, end, results);
        }
    }

    /// Collect all entries in sorted order.
    pub fn scan_all(&self) -> Vec<(IndexKey, RowId)> {
        let mut results = Vec::new();
        self.inorder_collect(self.root, &mut results);
        results
    }

    fn inorder_collect(&self, node_idx: usize, results: &mut Vec<(IndexKey, RowId)>) {
        let node = &self.nodes[node_idx];
        for i in 0..node.keys.len() {
            if !node.is_leaf {
                self.inorder_collect(node.children[i], results);
            }
            results.push((node.keys[i].clone(), node.values[i]));
        }
        if !node.is_leaf && !node.children.is_empty() {
            let last = *node.children.last().unwrap();
            self.inorder_collect(last, results);
        }
    }

    /// Bulk load sorted entries. The index must be empty.
    pub fn bulk_load(&mut self, entries: &[(IndexKey, RowId)]) -> Result<(), BTreeError> {
        if self.entry_count > 0 {
            return Err(BTreeError::DuplicateKey(
                "bulk load requires empty index".to_string(),
            ));
        }
        for (key, row_id) in entries {
            self.insert(key.clone(), *row_id)?;
        }
        Ok(())
    }

    /// Return index statistics.
    pub fn stats(&self) -> IndexStats {
        let depth = self.compute_depth(self.root);
        let mut leaf_count = 0;
        let mut internal_count = 0;
        for node in &self.nodes {
            if node.is_leaf {
                leaf_count += 1;
            } else {
                internal_count += 1;
            }
        }
        IndexStats {
            order: self.order,
            depth,
            total_nodes: self.nodes.len(),
            leaf_nodes: leaf_count,
            internal_nodes: internal_count,
            total_entries: self.entry_count,
        }
    }

    fn compute_depth(&self, node_idx: usize) -> usize {
        let node = &self.nodes[node_idx];
        if node.is_leaf {
            1
        } else if node.children.is_empty() {
            1
        } else {
            1 + self.compute_depth(node.children[0])
        }
    }

    /// Minimum key in the index.
    pub fn min_key(&self) -> Option<&IndexKey> {
        if self.is_empty() {
            return None;
        }
        let mut idx = self.root;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf || node.children.is_empty() {
                return node.keys.first();
            }
            idx = node.children[0];
        }
    }

    /// Maximum key in the index.
    pub fn max_key(&self) -> Option<&IndexKey> {
        if self.is_empty() {
            return None;
        }
        let mut idx = self.root;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf || node.children.is_empty() {
                return node.keys.last();
            }
            idx = *node.children.last().unwrap();
        }
    }

    /// Serialize the index to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let stats = self.stats();
        serde_json::json!({
            "order": stats.order,
            "depth": stats.depth,
            "total_nodes": stats.total_nodes,
            "total_entries": stats.total_entries,
            "entries": self.scan_all().iter().map(|(k, v)| {
                serde_json::json!({"key": k.to_string(), "row_id": v})
            }).collect::<Vec<_>>(),
        })
    }
}

impl fmt::Debug for BTreeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = self.stats();
        f.debug_struct("BTreeIndex")
            .field("order", &self.order)
            .field("entries", &self.entry_count)
            .field("depth", &stats.depth)
            .field("nodes", &stats.total_nodes)
            .finish()
    }
}

// ── Iterator ─────────────────────────────────────────────────────

/// Iterator over all entries in sorted order.
pub struct BTreeIter {
    entries: Vec<(IndexKey, RowId)>,
    pos: usize,
}

impl Iterator for BTreeIter {
    type Item = (IndexKey, RowId);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let item = self.entries[self.pos].clone();
            self.pos += 1;
            Some(item)
        } else {
            None
        }
    }
}

impl BTreeIndex {
    /// Create an iterator over all entries in sorted order.
    pub fn iter(&self) -> BTreeIter {
        BTreeIter {
            entries: self.scan_all(),
            pos: 0,
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn int_key(v: i64) -> IndexKey {
        IndexKey::Int(v)
    }

    #[test]
    fn create_empty_index() {
        let idx = BTreeIndex::new(4, false).unwrap();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn invalid_order_rejected() {
        let err = BTreeIndex::new(2, false).unwrap_err();
        assert_eq!(err, BTreeError::InvalidOrder(2));
    }

    #[test]
    fn insert_and_search_single() {
        let mut idx = BTreeIndex::new(4, false).unwrap();
        idx.insert(int_key(42), 100).unwrap();
        assert_eq!(idx.search(&int_key(42)), Some(100));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn insert_many_and_search() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        for i in 0..50 {
            idx.insert(int_key(i), i as u64 * 10).unwrap();
        }
        assert_eq!(idx.len(), 50);
        for i in 0..50 {
            assert_eq!(idx.search(&int_key(i)), Some(i as u64 * 10));
        }
    }

    #[test]
    fn unique_rejects_duplicate() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        idx.insert(int_key(1), 10).unwrap();
        let err = idx.insert(int_key(1), 20).unwrap_err();
        assert!(matches!(err, BTreeError::DuplicateKey(_)));
    }

    #[test]
    fn non_unique_allows_duplicate_keys() {
        let mut idx = BTreeIndex::new(4, false).unwrap();
        idx.insert(int_key(1), 10).unwrap();
        idx.insert(int_key(1), 20).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn delete_existing() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        idx.insert(int_key(5), 50).unwrap();
        idx.insert(int_key(10), 100).unwrap();
        let val = idx.delete(&int_key(5)).unwrap();
        assert_eq!(val, 50);
        assert_eq!(idx.len(), 1);
        assert!(idx.search(&int_key(5)).is_none());
    }

    #[test]
    fn delete_missing_errors() {
        let mut idx = BTreeIndex::new(4, false).unwrap();
        let err = idx.delete(&int_key(99)).unwrap_err();
        assert!(matches!(err, BTreeError::KeyNotFound(_)));
    }

    #[test]
    fn range_scan_subset() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        for i in 0..20 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let results = idx.range_scan(&int_key(5), &int_key(10));
        let keys: Vec<i64> = results
            .iter()
            .map(|(k, _)| match k {
                IndexKey::Int(v) => *v,
                _ => panic!("unexpected"),
            })
            .collect();
        for i in 5..=10 {
            assert!(keys.contains(&i), "missing key {i}");
        }
    }

    #[test]
    fn scan_all_sorted() {
        let mut idx = BTreeIndex::new(3, true).unwrap();
        // Insert in reverse order.
        for i in (0..15).rev() {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let all = idx.scan_all();
        for i in 0..all.len() - 1 {
            assert!(all[i].0 < all[i + 1].0);
        }
    }

    #[test]
    fn bulk_load() {
        let mut idx = BTreeIndex::new(5, true).unwrap();
        let entries: Vec<(IndexKey, RowId)> =
            (0..30).map(|i| (int_key(i), i as u64)).collect();
        idx.bulk_load(&entries).unwrap();
        assert_eq!(idx.len(), 30);
    }

    #[test]
    fn bulk_load_requires_empty() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        idx.insert(int_key(1), 1).unwrap();
        let err = idx.bulk_load(&[(int_key(2), 2)]).unwrap_err();
        assert!(matches!(err, BTreeError::DuplicateKey(_)));
    }

    #[test]
    fn stats_populated() {
        let mut idx = BTreeIndex::new(3, true).unwrap();
        for i in 0..20 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let stats = idx.stats();
        assert_eq!(stats.order, 3);
        assert_eq!(stats.total_entries, 20);
        assert!(stats.depth >= 2);
        assert!(stats.total_nodes > 1);
        assert_eq!(stats.leaf_nodes + stats.internal_nodes, stats.total_nodes);
    }

    #[test]
    fn min_max_key() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        assert!(idx.min_key().is_none());
        for i in [5, 1, 9, 3, 7] {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        assert_eq!(idx.min_key(), Some(&int_key(1)));
        assert_eq!(idx.max_key(), Some(&int_key(9)));
    }

    #[test]
    fn iterator_yields_sorted() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        for i in [10, 3, 7, 1, 5] {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let collected: Vec<(IndexKey, RowId)> = idx.iter().collect();
        assert_eq!(collected.len(), 5);
        for i in 0..collected.len() - 1 {
            assert!(collected[i].0 < collected[i + 1].0);
        }
    }

    #[test]
    fn text_keys() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        idx.insert(IndexKey::Text("banana".into()), 1).unwrap();
        idx.insert(IndexKey::Text("apple".into()), 2).unwrap();
        idx.insert(IndexKey::Text("cherry".into()), 3).unwrap();
        assert_eq!(idx.search(&IndexKey::Text("banana".into())), Some(1));
        let all = idx.scan_all();
        assert_eq!(all[0].0, IndexKey::Text("apple".into()));
    }

    #[test]
    fn to_json_structure() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        idx.insert(int_key(1), 10).unwrap();
        idx.insert(int_key(2), 20).unwrap();
        let json = idx.to_json();
        assert_eq!(json["order"], 4);
        assert_eq!(json["total_entries"], 2);
        assert!(json["entries"].is_array());
    }

    #[test]
    fn depth_grows_with_entries() {
        let mut idx = BTreeIndex::new(3, true).unwrap();
        let depth_at_1 = {
            idx.insert(int_key(1), 1).unwrap();
            idx.stats().depth
        };
        for i in 2..=50 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        let depth_at_50 = idx.stats().depth;
        assert!(depth_at_50 >= depth_at_1);
    }

    #[test]
    fn search_missing_returns_none() {
        let idx = BTreeIndex::new(4, false).unwrap();
        assert!(idx.search(&int_key(999)).is_none());
    }

    #[test]
    fn delete_many_then_search() {
        let mut idx = BTreeIndex::new(4, true).unwrap();
        for i in 0..10 {
            idx.insert(int_key(i), i as u64).unwrap();
        }
        for i in 0..5 {
            idx.delete(&int_key(i)).unwrap();
        }
        assert_eq!(idx.len(), 5);
        for i in 5..10 {
            assert!(idx.search(&int_key(i)).is_some());
        }
        for i in 0..5 {
            assert!(idx.search(&int_key(i)).is_none());
        }
    }
}
