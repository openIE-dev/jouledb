//! Union-Find (disjoint set) — efficient connected component tracking.
//!
//! Supports make_set, find with path compression, union by rank, component count,
//! component size, same_component check, and snapshot/rollback.

// ── DisjointSet ─────────────────────────────────────────────────────────────

/// Union-Find data structure with path compression and union by rank.
#[derive(Debug, Clone)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    component_count: usize,
}

impl DisjointSet {
    /// Create a new empty disjoint set.
    pub fn new() -> Self {
        Self {
            parent: Vec::new(),
            rank: Vec::new(),
            size: Vec::new(),
            component_count: 0,
        }
    }

    /// Create a disjoint set with `n` elements (0..n), each in its own set.
    pub fn with_size(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            component_count: n,
        }
    }

    /// Add a new element and return its id.
    pub fn make_set(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.rank.push(0);
        self.size.push(1);
        self.component_count += 1;
        id
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    /// Number of connected components.
    pub fn component_count(&self) -> usize {
        self.component_count
    }

    /// Find the representative of the set containing `x`, with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        assert!(x < self.parent.len(), "Element out of range");
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing `x` and `y`. Returns false if already in the same set.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }

        // Union by rank
        let (root, child) = if self.rank[rx] < self.rank[ry] {
            (ry, rx)
        } else {
            (rx, ry)
        };

        self.parent[child] = root;
        self.size[root] += self.size[child];
        if self.rank[root] == self.rank[child] {
            self.rank[root] += 1;
        }
        self.component_count -= 1;
        true
    }

    /// Check if `x` and `y` are in the same component.
    pub fn same_component(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    /// Size of the component containing `x`.
    pub fn component_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }

    /// Return all elements in the same component as `x`.
    pub fn component_members(&mut self, x: usize) -> Vec<usize> {
        let root = self.find(x);
        let n = self.parent.len();
        let mut members = Vec::new();
        // We need to find root for each element — but path compression
        // mutates, so we collect by checking parent chain manually.
        for i in 0..n {
            if self.find(i) == root {
                members.push(i);
            }
        }
        members
    }

    /// Return all components as vectors of element ids.
    pub fn all_components(&mut self) -> Vec<Vec<usize>> {
        let n = self.parent.len();
        let mut map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            map.entry(root).or_default().push(i);
        }
        let mut components: Vec<Vec<usize>> = map.into_values().collect();
        components.sort_by_key(|c| c[0]);
        components
    }

    /// Take a snapshot that can be rolled back to.
    pub fn snapshot(&self) -> DisjointSetSnapshot {
        DisjointSetSnapshot {
            parent: self.parent.clone(),
            rank: self.rank.clone(),
            size: self.size.clone(),
            component_count: self.component_count,
        }
    }

    /// Rollback to a previous snapshot.
    pub fn rollback(&mut self, snap: &DisjointSetSnapshot) {
        self.parent = snap.parent.clone();
        self.rank = snap.rank.clone();
        self.size = snap.size.clone();
        self.component_count = snap.component_count;
    }
}

impl Default for DisjointSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot ────────────────────────────────────────────────────────────────

/// Immutable snapshot of a DisjointSet for rollback support.
#[derive(Debug, Clone)]
pub struct DisjointSetSnapshot {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    component_count: usize,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_set() {
        let mut ds = DisjointSet::new();
        let a = ds.make_set();
        let b = ds.make_set();
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.component_count(), 2);
    }

    #[test]
    fn test_with_size() {
        let ds = DisjointSet::with_size(5);
        assert_eq!(ds.len(), 5);
        assert_eq!(ds.component_count(), 5);
    }

    #[test]
    fn test_find_initial() {
        let mut ds = DisjointSet::with_size(3);
        assert_eq!(ds.find(0), 0);
        assert_eq!(ds.find(1), 1);
        assert_eq!(ds.find(2), 2);
    }

    #[test]
    fn test_union() {
        let mut ds = DisjointSet::with_size(5);
        assert!(ds.union(0, 1));
        assert!(ds.same_component(0, 1));
        assert!(!ds.same_component(0, 2));
        assert_eq!(ds.component_count(), 4);
    }

    #[test]
    fn test_union_already_same() {
        let mut ds = DisjointSet::with_size(3);
        ds.union(0, 1);
        assert!(!ds.union(0, 1)); // already same
        assert_eq!(ds.component_count(), 2);
    }

    #[test]
    fn test_component_size() {
        let mut ds = DisjointSet::with_size(5);
        ds.union(0, 1);
        ds.union(1, 2);
        assert_eq!(ds.component_size(0), 3);
        assert_eq!(ds.component_size(3), 1);
    }

    #[test]
    fn test_component_members() {
        let mut ds = DisjointSet::with_size(5);
        ds.union(0, 2);
        ds.union(2, 4);
        let mut members = ds.component_members(0);
        members.sort();
        assert_eq!(members, vec![0, 2, 4]);
    }

    #[test]
    fn test_all_components() {
        let mut ds = DisjointSet::with_size(6);
        ds.union(0, 1);
        ds.union(2, 3);
        ds.union(4, 5);
        let components = ds.all_components();
        assert_eq!(components.len(), 3);
        for c in &components {
            assert_eq!(c.len(), 2);
        }
    }

    #[test]
    fn test_path_compression() {
        let mut ds = DisjointSet::with_size(10);
        // Create a chain: 0-1-2-3-4
        ds.union(0, 1);
        ds.union(1, 2);
        ds.union(2, 3);
        ds.union(3, 4);
        // After find, all should point to root
        let root = ds.find(4);
        assert_eq!(ds.find(0), root);
        assert_eq!(ds.find(1), root);
        assert_eq!(ds.find(2), root);
        assert_eq!(ds.find(3), root);
    }

    #[test]
    fn test_snapshot_and_rollback() {
        let mut ds = DisjointSet::with_size(5);
        ds.union(0, 1);
        let snap = ds.snapshot();

        ds.union(2, 3);
        ds.union(0, 2);
        assert_eq!(ds.component_count(), 2);

        ds.rollback(&snap);
        assert_eq!(ds.component_count(), 4);
        assert!(ds.same_component(0, 1));
        assert!(!ds.same_component(0, 2));
    }

    #[test]
    fn test_empty() {
        let ds = DisjointSet::new();
        assert!(ds.is_empty());
        assert_eq!(ds.component_count(), 0);
    }

    #[test]
    fn test_single_element() {
        let mut ds = DisjointSet::with_size(1);
        assert_eq!(ds.find(0), 0);
        assert_eq!(ds.component_size(0), 1);
        assert_eq!(ds.component_count(), 1);
    }

    #[test]
    fn test_large_union() {
        let mut ds = DisjointSet::with_size(100);
        for i in 0..99 {
            ds.union(i, i + 1);
        }
        assert_eq!(ds.component_count(), 1);
        assert_eq!(ds.component_size(0), 100);
    }
}
