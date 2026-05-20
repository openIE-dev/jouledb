//! Computed/derived values with dependency tracking, lazy evaluation,
//! dirty-flag propagation, topological sorting, memoization, and batch updates.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Types ──

/// Unique ID for a computed value or source.
pub type NodeId = u64;

/// A raw source value in the dependency graph.
#[derive(Debug, Clone)]
struct SourceNode {
    /// Current value (as string for simplicity/type-erasure).
    value: String,
    /// Version counter, bumped on each write.
    version: u64,
}

/// A computed node that derives its value from dependencies.
struct ComputedNode {
    dependencies: Vec<NodeId>,
    compute: Box<dyn Fn(&dyn Fn(NodeId) -> String) -> String>,
    cached_value: Option<String>,
    /// Version of each dependency when cache was last computed.
    dep_versions: HashMap<NodeId, u64>,
    dirty: bool,
}

// ── ComputedGraph ──

/// A dependency graph of source values and computed/derived values with
/// lazy evaluation, memoization, and batch update support.
pub struct ComputedGraph {
    sources: HashMap<NodeId, SourceNode>,
    computed: HashMap<NodeId, ComputedNode>,
    /// Dependents: source_id -> set of computed_ids that depend on it.
    dependents: HashMap<NodeId, HashSet<NodeId>>,
    /// Whether we're in a batch update (defer recomputation).
    batching: bool,
    /// Nodes dirtied during a batch.
    batch_dirty: HashSet<NodeId>,
}

impl Default for ComputedGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputedGraph {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            computed: HashMap::new(),
            dependents: HashMap::new(),
            batching: false,
            batch_dirty: HashSet::new(),
        }
    }

    /// Register a source value.
    pub fn add_source(&mut self, id: NodeId, value: impl Into<String>) {
        self.sources.insert(id, SourceNode {
            value: value.into(),
            version: 1,
        });
    }

    /// Update a source value. Marks dependent computed nodes as dirty.
    pub fn set_source(&mut self, id: NodeId, value: impl Into<String>) {
        if let Some(src) = self.sources.get_mut(&id) {
            src.value = value.into();
            src.version += 1;
            self.mark_dependents_dirty(id);
        }
    }

    /// Read a source value.
    pub fn get_source(&self, id: NodeId) -> Option<&str> {
        self.sources.get(&id).map(|s| s.value.as_str())
    }

    /// Register a computed value with its dependencies and compute function.
    pub fn add_computed(
        &mut self,
        id: NodeId,
        dependencies: Vec<NodeId>,
        compute: impl Fn(&dyn Fn(NodeId) -> String) -> String + 'static,
    ) {
        for &dep in &dependencies {
            self.dependents.entry(dep).or_default().insert(id);
        }
        self.computed.insert(id, ComputedNode {
            dependencies,
            compute: Box::new(compute),
            cached_value: None,
            dep_versions: HashMap::new(),
            dirty: true, // not yet computed
        });
    }

    /// Read a computed value. Lazily evaluates if dirty.
    pub fn get_computed(&mut self, id: NodeId) -> Option<String> {
        // Check if we need to recompute
        let needs_compute = self.computed.get(&id).map_or(false, |c| {
            if c.dirty || c.cached_value.is_none() {
                return true;
            }
            // Check if any dependency version changed
            for (&dep_id, &cached_ver) in &c.dep_versions {
                if let Some(src) = self.sources.get(&dep_id) {
                    if src.version != cached_ver {
                        return true;
                    }
                }
            }
            false
        });

        if needs_compute {
            self.recompute(id);
        }

        self.computed.get(&id).and_then(|c| c.cached_value.clone())
    }

    /// Force recomputation of a specific node.
    fn recompute(&mut self, id: NodeId) {
        // First, recursively recompute any dirty dependencies that are themselves computed.
        let dep_ids: Vec<NodeId> = self
            .computed
            .get(&id)
            .map(|c| c.dependencies.clone())
            .unwrap_or_default();

        for &dep_id in &dep_ids {
            if self.computed.contains_key(&dep_id) {
                let dep_dirty = self.computed.get(&dep_id).map_or(false, |c| c.dirty);
                if dep_dirty {
                    self.recompute(dep_id);
                }
            }
        }

        // Build a snapshot of source values and computed cache values for the reader
        let sources_snap: HashMap<NodeId, String> = self
            .sources
            .iter()
            .map(|(&k, v)| (k, v.value.clone()))
            .collect();
        let computed_snap: HashMap<NodeId, String> = self
            .computed
            .iter()
            .filter_map(|(&k, v)| v.cached_value.as_ref().map(|cv| (k, cv.clone())))
            .collect();

        if let Some(node) = self.computed.get(&id) {
            let reader = |dep_id: NodeId| -> String {
                if let Some(v) = sources_snap.get(&dep_id) {
                    v.clone()
                } else if let Some(v) = computed_snap.get(&dep_id) {
                    v.clone()
                } else {
                    String::new()
                }
            };
            let val = (node.compute)(&reader);

            // Now update the node
            let mut dep_versions = HashMap::new();
            for &dep_id in &dep_ids {
                if let Some(src) = self.sources.get(&dep_id) {
                    dep_versions.insert(dep_id, src.version);
                }
            }

            if let Some(node) = self.computed.get_mut(&id) {
                node.cached_value = Some(val);
                node.dep_versions = dep_versions;
                node.dirty = false;
            }
        }
    }

    /// Mark all dependents of a source as dirty.
    fn mark_dependents_dirty(&mut self, source_id: NodeId) {
        let deps = self
            .dependents
            .get(&source_id)
            .cloned()
            .unwrap_or_default();
        for dep_id in deps {
            if let Some(node) = self.computed.get_mut(&dep_id) {
                node.dirty = true;
            }
            if self.batching {
                self.batch_dirty.insert(dep_id);
            }
            // Propagate to dependents of this computed node
            if self.dependents.contains_key(&dep_id) {
                self.mark_dependents_dirty(dep_id);
            }
        }
    }

    /// Start a batch update. Defers recomputation until `end_batch`.
    pub fn begin_batch(&mut self) {
        self.batching = true;
        self.batch_dirty.clear();
    }

    /// End a batch update and recompute all dirtied nodes in topological order.
    pub fn end_batch(&mut self) {
        self.batching = false;
        let dirty: Vec<NodeId> = self.batch_dirty.drain().collect();
        let sorted = self.topological_sort(&dirty);
        for id in sorted {
            self.recompute(id);
        }
    }

    /// Topological sort of the given node IDs based on their dependency relationships.
    pub fn topological_sort(&self, nodes: &[NodeId]) -> Vec<NodeId> {
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &id in nodes {
            in_degree.entry(id).or_insert(0);
            if let Some(computed) = self.computed.get(&id) {
                for &dep in &computed.dependencies {
                    if node_set.contains(&dep) {
                        adj.entry(dep).or_default().push(id);
                        *in_degree.entry(id).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();
        while let Some(id) = queue.pop_front() {
            result.push(id);
            if let Some(neighbors) = adj.get(&id) {
                for &next in neighbors {
                    if let Some(deg) = in_degree.get_mut(&next) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(next);
                        }
                    }
                }
            }
        }

        result
    }

    /// Check if a computed node is currently dirty.
    pub fn is_dirty(&self, id: NodeId) -> bool {
        self.computed.get(&id).map_or(false, |c| c.dirty)
    }

    /// Invalidate a computed node, forcing recomputation on next read.
    pub fn invalidate(&mut self, id: NodeId) {
        if let Some(node) = self.computed.get_mut(&id) {
            node.dirty = true;
        }
    }

    /// Number of source nodes.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Number of computed nodes.
    pub fn computed_count(&self) -> usize {
        self.computed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_source_and_computed() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "hello");
        g.add_source(2, "world");
        g.add_computed(10, vec![1, 2], |reader| {
            format!("{} {}", reader(1), reader(2))
        });

        assert_eq!(g.get_computed(10), Some("hello world".to_string()));
    }

    #[test]
    fn lazy_evaluation() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "10");
        g.add_computed(10, vec![1], |reader| {
            let v: i32 = reader(1).parse().unwrap_or(0);
            (v * 2).to_string()
        });

        // Not computed yet
        assert!(g.is_dirty(10));
        assert_eq!(g.get_computed(10), Some("20".to_string()));
        assert!(!g.is_dirty(10));
    }

    #[test]
    fn memoization_returns_cached() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "5");
        g.add_computed(10, vec![1], |reader| {
            let v: i32 = reader(1).parse().unwrap_or(0);
            (v + 1).to_string()
        });

        assert_eq!(g.get_computed(10), Some("6".to_string()));
        // Second read without changes still returns cached
        assert_eq!(g.get_computed(10), Some("6".to_string()));
    }

    #[test]
    fn dirty_propagation() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "a");
        g.add_computed(10, vec![1], |reader| {
            reader(1).to_uppercase()
        });

        assert_eq!(g.get_computed(10), Some("A".to_string()));

        g.set_source(1, "b");
        assert!(g.is_dirty(10));
        assert_eq!(g.get_computed(10), Some("B".to_string()));
    }

    #[test]
    fn chained_computed_nodes() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "3");

        // Node 10 = source * 2
        g.add_computed(10, vec![1], |reader| {
            let v: i32 = reader(1).parse().unwrap_or(0);
            (v * 2).to_string()
        });

        // Node 20 depends on computed 10
        // Register 20 as dependent of 10
        g.dependents.entry(10).or_default().insert(20);
        g.computed.insert(20, ComputedNode {
            dependencies: vec![10],
            compute: Box::new(|reader| {
                let v: i32 = reader(10).parse().unwrap_or(0);
                (v + 100).to_string()
            }),
            cached_value: None,
            dep_versions: HashMap::new(),
            dirty: true,
        });

        // First compute 10, then 20 can read from cache
        assert_eq!(g.get_computed(10), Some("6".to_string()));
        assert_eq!(g.get_computed(20), Some("106".to_string()));
    }

    #[test]
    fn batch_updates() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "a");
        g.add_source(2, "b");
        g.add_computed(10, vec![1, 2], |reader| {
            format!("{}+{}", reader(1), reader(2))
        });

        assert_eq!(g.get_computed(10), Some("a+b".to_string()));

        g.begin_batch();
        g.set_source(1, "x");
        g.set_source(2, "y");
        // During batch, node is dirty but we defer recompute
        assert!(g.is_dirty(10));
        g.end_batch();

        // After batch, computed node was recomputed
        assert!(!g.is_dirty(10));
        assert_eq!(g.get_computed(10), Some("x+y".to_string()));
    }

    #[test]
    fn topological_sort_order() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "x");

        // 10 depends on source 1
        g.add_computed(10, vec![1], |reader| reader(1).to_string());
        // 20 depends on computed 10
        g.add_computed(20, vec![10], |reader| reader(10).to_string());
        // 30 depends on computed 20
        g.add_computed(30, vec![20], |reader| reader(20).to_string());

        let sorted = g.topological_sort(&[30, 10, 20]);
        // 10 before 20 before 30
        let pos_10 = sorted.iter().position(|x| *x == 10).unwrap();
        let pos_20 = sorted.iter().position(|x| *x == 20).unwrap();
        let pos_30 = sorted.iter().position(|x| *x == 30).unwrap();
        assert!(pos_10 < pos_20);
        assert!(pos_20 < pos_30);
    }

    #[test]
    fn invalidate_forces_recompute() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "42");
        g.add_computed(10, vec![1], |reader| reader(1).to_string());

        assert_eq!(g.get_computed(10), Some("42".to_string()));
        assert!(!g.is_dirty(10));

        g.invalidate(10);
        assert!(g.is_dirty(10));
        assert_eq!(g.get_computed(10), Some("42".to_string()));
    }

    #[test]
    fn source_and_computed_counts() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "a");
        g.add_source(2, "b");
        g.add_computed(10, vec![1], |reader| reader(1).to_string());
        assert_eq!(g.source_count(), 2);
        assert_eq!(g.computed_count(), 1);
    }

    #[test]
    fn get_source_value() {
        let mut g = ComputedGraph::new();
        g.add_source(1, "hello");
        assert_eq!(g.get_source(1), Some("hello"));
        assert_eq!(g.get_source(999), None);
    }

    #[test]
    fn nonexistent_computed_returns_none() {
        let mut g = ComputedGraph::new();
        assert_eq!(g.get_computed(999), None);
    }
}
