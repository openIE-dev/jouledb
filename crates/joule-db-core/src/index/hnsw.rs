//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! This is the same algorithm used by pgvector, Milvus, Qdrant, and Pinecone.
//! JouleDB's implementation adds energy-aware layer selection.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

/// Distance metric for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
    /// Inner product (negated for min-heap)
    InnerProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

/// Configuration for HNSW index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node per layer (default: 16)
    pub m: usize,
    /// Maximum connections for layer 0 (default: 2 * m)
    pub m0: usize,
    /// Size of dynamic candidate list during construction (default: 200)
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search (default: 50)
    pub ef_search: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Vector dimensions
    pub dimensions: usize,
    /// Normalization factor for level generation (1/ln(M))
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Euclidean,
            dimensions: 0, // must be set before use
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Validate configuration, returning an error message if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.dimensions == 0 {
            return Err("dimensions must be > 0".into());
        }
        if self.m == 0 {
            return Err("m must be > 0".into());
        }
        if self.m0 == 0 {
            return Err("m0 must be > 0".into());
        }
        if self.ef_construction == 0 {
            return Err("ef_construction must be > 0".into());
        }
        if self.ef_search == 0 {
            return Err("ef_search must be > 0".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Min-heap / max-heap wrappers for (distance, id) pairs
// ---------------------------------------------------------------------------

/// A candidate entry: (distance, node_id). Ordered by distance ascending (min-heap).
#[derive(Debug, Clone, Copy)]
struct MinCandidate {
    dist: f32,
    id: u64,
}

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap acts as a min-heap by distance
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then(other.id.cmp(&self.id))
    }
}

/// Max-heap candidate (largest distance at top) for bounding the result set.
#[derive(Debug, Clone, Copy)]
struct MaxCandidate {
    dist: f32,
    id: u64,
}

impl PartialEq for MaxCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for MaxCandidate {}

impl PartialOrd for MaxCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}

// ---------------------------------------------------------------------------
// HNSW Node
// ---------------------------------------------------------------------------

/// A node in the HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HnswNode {
    /// The record ID this vector belongs to
    id: u64,
    /// The vector data
    vector: Vec<f32>,
    /// Connections at each layer: layer index -> vec of neighbor IDs
    connections: Vec<Vec<u64>>,
    /// Maximum layer this node exists on
    max_layer: usize,
}

// ---------------------------------------------------------------------------
// HNSW Index
// ---------------------------------------------------------------------------

/// HNSW Index for approximate nearest neighbor search.
#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    config: HnswConfig,
    /// All nodes indexed by ID
    nodes: HashMap<u64, HnswNode>,
    /// Entry point (highest-layer node)
    entry_point: Option<u64>,
    /// Maximum layer currently in the graph
    max_layer: usize,
    /// Total number of vectors
    count: usize,
}

impl HnswIndex {
    /// Create a new, empty HNSW index with the given configuration.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            entry_point: None,
            max_layer: 0,
            count: 0,
        }
    }

    /// Insert a vector with associated record ID.
    ///
    /// If `id` already exists, its vector and connections are replaced.
    /// Panics if `vector.len() != config.dimensions`.
    pub fn insert(&mut self, id: u64, vector: &[f32]) {
        assert_eq!(
            vector.len(),
            self.config.dimensions,
            "vector length {} != configured dimensions {}",
            vector.len(),
            self.config.dimensions,
        );

        // If already present, remove first so we re-insert cleanly.
        if self.nodes.contains_key(&id) {
            self.remove(id);
        }

        let level = self.random_level();

        // First node: just insert and set as entry point.
        if self.entry_point.is_none() {
            let mut connections = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                connections.push(Vec::new());
            }
            self.nodes.insert(
                id,
                HnswNode {
                    id,
                    vector: vector.to_vec(),
                    connections,
                    max_layer: level,
                },
            );
            self.entry_point = Some(id);
            self.max_layer = level;
            self.count = 1;
            return;
        }

        let mut ep = self.entry_point.unwrap();
        let current_max = self.max_layer;

        // Phase 1: greedy search from top layer down to level+1
        let mut ep_dist = self.distance(vector, &self.nodes[&ep].vector);
        for layer in (level + 1..=current_max).rev() {
            let changed = self.greedy_closest(vector, ep, ep_dist, layer);
            ep = changed.0;
            ep_dist = changed.1;
        }

        // Phase 2: for layers min(level, current_max) down to 0, search and connect
        let top = level.min(current_max);
        for layer in (0..=top).rev() {
            let candidates = self.search_layer(vector, ep, self.config.ef_construction, layer);
            let neighbors = self.select_neighbors(&candidates, self.max_connections(layer));

            // We'll update ep to the closest found so far for the next lower layer.
            if let Some(first) = neighbors.first() {
                if first.1 < ep_dist {
                    ep = first.0;
                    ep_dist = first.1;
                }
            }

            // We need to create / extend the node's connections vector before
            // we start mutating neighbor connections, because we borrow `self.nodes`.
            // Collect neighbor ids first.
            let neighbor_ids: Vec<u64> = neighbors.iter().map(|&(nid, _)| nid).collect();

            // Ensure the new node exists with enough layers allocated.
            let node = self.nodes.entry(id).or_insert_with(|| {
                let mut conns = Vec::with_capacity(level + 1);
                for _ in 0..=level {
                    conns.push(Vec::new());
                }
                HnswNode {
                    id,
                    vector: vector.to_vec(),
                    connections: conns,
                    max_layer: level,
                }
            });
            node.connections[layer] = neighbor_ids.clone();

            // Add reverse connections and prune
            let m_layer = self.max_connections(layer);
            for &nid in &neighbor_ids {
                // Push ourselves into neighbor's connection list
                if let Some(neighbor) = self.nodes.get_mut(&nid) {
                    if layer < neighbor.connections.len() {
                        let conns = &mut neighbor.connections[layer];
                        if !conns.contains(&id) {
                            conns.push(id);
                        }
                        // Prune if over capacity
                        if conns.len() > m_layer {
                            self.prune_connections(nid, layer, m_layer);
                        }
                    }
                }
            }
        }

        // Update entry point if new node has a higher layer
        if level > current_max {
            self.entry_point = Some(id);
            self.max_layer = level;
        }

        self.count = self.nodes.len();
    }

    /// Search for `k` nearest neighbors to `query`.
    ///
    /// Returns `Vec<(id, distance)>` sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() || k == 0 {
            return Vec::new();
        }

        let ep = self.entry_point.unwrap();
        let mut current = ep;
        let mut current_dist = self.distance(query, &self.nodes[&current].vector);

        // Greedy descent from top layer to layer 1
        for layer in (1..=self.max_layer).rev() {
            let (c, d) = self.greedy_closest(query, current, current_dist, layer);
            current = c;
            current_dist = d;
        }

        // Search layer 0 with ef_search
        let ef = self.config.ef_search.max(k);
        let mut candidates = self.search_layer(query, current, ef, 0);

        // Sort by distance and take top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Remove a vector by ID. Returns `true` if found and removed.
    pub fn remove(&mut self, id: u64) -> bool {
        let node = match self.nodes.remove(&id) {
            Some(n) => n,
            None => return false,
        };

        // Remove this node from all neighbors' connection lists
        for (layer, neighbors) in node.connections.iter().enumerate() {
            for &nid in neighbors {
                if let Some(neighbor) = self.nodes.get_mut(&nid) {
                    if layer < neighbor.connections.len() {
                        neighbor.connections[layer].retain(|&x| x != id);
                    }
                }
            }
        }

        self.count = self.nodes.len();

        // If we removed the entry point, pick a new one
        if self.entry_point == Some(id) {
            self.entry_point = None;
            self.max_layer = 0;
            // Find the node with the highest layer as new entry point
            for (&nid, n) in &self.nodes {
                if self.entry_point.is_none() || n.max_layer > self.max_layer {
                    self.entry_point = Some(nid);
                    self.max_layer = n.max_layer;
                }
            }
        }

        true
    }

    /// Number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the index is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the current configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Assign a random layer for a new node using a geometric distribution.
    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let uniform: f64 = rng.random::<f64>();
        // -ln(uniform) * ml, floored
        let level = (-uniform.ln() * self.config.ml).floor() as usize;
        level
    }

    /// Compute the distance between two vectors under the configured metric.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0f32;
                for i in 0..a.len() {
                    let d = a[i] - b[i];
                    sum += d * d;
                }
                sum.sqrt()
            }
            DistanceMetric::Cosine => {
                let mut dot = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    norm_a += a[i] * a[i];
                    norm_b += b[i] * b[i];
                }
                let denom = norm_a.sqrt() * norm_b.sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot / denom
                }
            }
            DistanceMetric::InnerProduct => {
                let mut dot = 0.0f32;
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                }
                -dot // negated: smaller = more similar
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0f32;
                for i in 0..a.len() {
                    sum += (a[i] - b[i]).abs();
                }
                sum
            }
        }
    }

    /// Greedily traverse a single layer to find the closest node to `query`,
    /// starting from `ep` with known distance `ep_dist`.
    /// Returns (closest_id, closest_dist).
    fn greedy_closest(&self, query: &[f32], mut ep: u64, mut ep_dist: f32, layer: usize) -> (u64, f32) {
        loop {
            let mut changed = false;
            let node = &self.nodes[&ep];
            if layer < node.connections.len() {
                for &nid in &node.connections[layer] {
                    if let Some(neighbor) = self.nodes.get(&nid) {
                        let d = self.distance(query, &neighbor.vector);
                        if d < ep_dist {
                            ep = nid;
                            ep_dist = d;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        (ep, ep_dist)
    }

    /// Search a single layer starting from `entry`, returning up to `ef` closest
    /// candidates as `Vec<(id, distance)>`.
    fn search_layer(
        &self,
        query: &[f32],
        entry: u64,
        ef: usize,
        layer: usize,
    ) -> Vec<(u64, f32)> {
        let entry_dist = self.distance(query, &self.nodes[&entry].vector);

        // candidates: min-heap (closest first)
        let mut candidates = BinaryHeap::<MinCandidate>::new();
        // results: max-heap (farthest first) bounded by ef
        let mut results = BinaryHeap::<MaxCandidate>::new();
        let mut visited = HashSet::new();

        candidates.push(MinCandidate {
            dist: entry_dist,
            id: entry,
        });
        results.push(MaxCandidate {
            dist: entry_dist,
            id: entry,
        });
        visited.insert(entry);

        while let Some(MinCandidate { dist: c_dist, id: c_id }) = candidates.pop() {
            // If the closest candidate is farther than the farthest result, stop
            let farthest = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if c_dist > farthest {
                break;
            }

            // Explore neighbors
            if let Some(node) = self.nodes.get(&c_id) {
                if layer < node.connections.len() {
                    for &nid in &node.connections[layer] {
                        if visited.insert(nid) {
                            if let Some(neighbor) = self.nodes.get(&nid) {
                                let d = self.distance(query, &neighbor.vector);
                                let farthest =
                                    results.peek().map(|r| r.dist).unwrap_or(f32::MAX);

                                if d < farthest || results.len() < ef {
                                    candidates.push(MinCandidate { dist: d, id: nid });
                                    results.push(MaxCandidate { dist: d, id: nid });

                                    if results.len() > ef {
                                        results.pop(); // remove farthest
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .map(|mc| (mc.id, mc.dist))
            .collect()
    }

    /// Select the best `m` neighbors from candidates (simple nearest-first).
    fn select_neighbors(&self, candidates: &[(u64, f32)], m: usize) -> Vec<(u64, f32)> {
        let mut sorted: Vec<(u64, f32)> = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted.truncate(m);
        sorted
    }

    /// Maximum connections for a given layer (m0 for layer 0, m otherwise).
    #[inline]
    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m0
        } else {
            self.config.m
        }
    }

    /// Prune a node's connections at a given layer to at most `max_conn`.
    fn prune_connections(&mut self, node_id: u64, layer: usize, max_conn: usize) {
        // We need to compute distances to decide which to keep. Collect
        // (neighbor_id, distance) pairs, sort, and truncate.
        let node_vec = match self.nodes.get(&node_id) {
            Some(n) => n.vector.clone(),
            None => return,
        };
        let conns: Vec<u64> = match self.nodes.get(&node_id) {
            Some(n) if layer < n.connections.len() => n.connections[layer].clone(),
            _ => return,
        };

        let mut scored: Vec<(u64, f32)> = conns
            .iter()
            .filter_map(|&nid| {
                self.nodes
                    .get(&nid)
                    .map(|n| (nid, self.distance(&node_vec, &n.vector)))
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(max_conn);

        let pruned: Vec<u64> = scored.into_iter().map(|(nid, _)| nid).collect();
        if let Some(node) = self.nodes.get_mut(&node_id) {
            if layer < node.connections.len() {
                node.connections[layer] = pruned;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dims: usize) -> HnswConfig {
        HnswConfig {
            dimensions: dims,
            ..HnswConfig::default()
        }
    }

    /// Brute-force kNN for recall comparison.
    fn brute_force_knn(
        vectors: &[(u64, Vec<f32>)],
        query: &[f32],
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<(u64, f32)> {
        let dist = |a: &[f32], b: &[f32]| -> f32 {
            match metric {
                DistanceMetric::Euclidean => {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f32>()
                        .sqrt()
                }
                DistanceMetric::Cosine => {
                    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                    let denom = na * nb;
                    if denom == 0.0 {
                        1.0
                    } else {
                        1.0 - dot / denom
                    }
                }
                DistanceMetric::InnerProduct => {
                    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                    -dot
                }
                DistanceMetric::Manhattan => {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .sum()
                }
            }
        };
        let mut scored: Vec<(u64, f32)> = vectors
            .iter()
            .map(|(id, v)| (*id, dist(v, query)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }

    #[test]
    fn test_empty_index_search() {
        let idx = HnswIndex::new(make_config(4));
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        let results = idx.search(&[1.0, 2.0, 3.0, 4.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_insert_and_search() {
        let mut idx = HnswIndex::new(make_config(3));
        idx.insert(1, &[1.0, 0.0, 0.0]);
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 1e-6); // exact match
    }

    #[test]
    fn test_search_returns_correct_k() {
        let mut idx = HnswIndex::new(make_config(2));
        for i in 0..20 {
            idx.insert(i, &[i as f32, 0.0]);
        }
        assert_eq!(idx.len(), 20);

        let results = idx.search(&[10.0, 0.0], 5);
        assert_eq!(results.len(), 5);

        // Check they are sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let mut idx = HnswIndex::new(make_config(2));
        idx.insert(0, &[0.0, 0.0]);
        idx.insert(1, &[1.0, 1.0]);

        let results = idx.search(&[0.0, 0.0], 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove() {
        let mut idx = HnswIndex::new(make_config(2));
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 1.0]);
        idx.insert(3, &[2.0, 2.0]);
        assert_eq!(idx.len(), 3);

        assert!(idx.remove(2));
        assert_eq!(idx.len(), 2);
        assert!(!idx.remove(2)); // already removed

        let results = idx.search(&[1.0, 1.0], 10);
        assert_eq!(results.len(), 2);
        let ids: HashSet<u64> = results.iter().map(|r| r.0).collect();
        assert!(!ids.contains(&2));
    }

    #[test]
    fn test_remove_entry_point() {
        let mut idx = HnswIndex::new(make_config(2));
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 1.0]);

        // Remove entry point (whichever it is)
        let ep = idx.entry_point.unwrap();
        assert!(idx.remove(ep));
        assert_eq!(idx.len(), 1);

        // Should still be searchable
        let results = idx.search(&[0.5, 0.5], 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_remove_all() {
        let mut idx = HnswIndex::new(make_config(2));
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 1.0]);
        idx.remove(1);
        idx.remove(2);
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn test_euclidean_distance() {
        let idx = HnswIndex::new(HnswConfig {
            dimensions: 3,
            metric: DistanceMetric::Euclidean,
            ..HnswConfig::default()
        });
        let d = idx.distance(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]);
        assert!((d - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_distance() {
        let idx = HnswIndex::new(HnswConfig {
            dimensions: 2,
            metric: DistanceMetric::Cosine,
            ..HnswConfig::default()
        });
        // Same direction => distance ~0
        let d = idx.distance(&[1.0, 0.0], &[2.0, 0.0]);
        assert!(d < 1e-5);
        // Orthogonal => distance ~1
        let d = idx.distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_inner_product_distance() {
        let idx = HnswIndex::new(HnswConfig {
            dimensions: 2,
            metric: DistanceMetric::InnerProduct,
            ..HnswConfig::default()
        });
        // Higher dot product => more negative distance => "closer"
        let d1 = idx.distance(&[1.0, 0.0], &[3.0, 0.0]); // dot=3 => -3
        let d2 = idx.distance(&[1.0, 0.0], &[1.0, 0.0]); // dot=1 => -1
        assert!(d1 < d2);
    }

    #[test]
    fn test_manhattan_distance() {
        let idx = HnswIndex::new(HnswConfig {
            dimensions: 3,
            metric: DistanceMetric::Manhattan,
            ..HnswConfig::default()
        });
        let d = idx.distance(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!((d - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_metric_search() {
        let mut idx = HnswIndex::new(HnswConfig {
            dimensions: 3,
            metric: DistanceMetric::Cosine,
            ..HnswConfig::default()
        });
        // Insert vectors in different directions
        idx.insert(1, &[1.0, 0.0, 0.0]);
        idx.insert(2, &[0.0, 1.0, 0.0]);
        idx.insert(3, &[0.0, 0.0, 1.0]);
        idx.insert(4, &[0.9, 0.1, 0.0]); // close to vector 1

        let results = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        let ids: Vec<u64> = results.iter().map(|r| r.0).collect();
        // ID 1 (exact) and ID 4 (close) should be the top 2
        assert!(ids.contains(&1));
        assert!(ids.contains(&4));
    }

    #[test]
    fn test_manhattan_metric_search() {
        let mut idx = HnswIndex::new(HnswConfig {
            dimensions: 2,
            metric: DistanceMetric::Manhattan,
            ..HnswConfig::default()
        });
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[10.0, 10.0]);
        idx.insert(3, &[1.0, 1.0]);

        let results = idx.search(&[0.0, 0.0], 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_insert_1000_and_recall() {
        let dims = 128;
        let n = 1000;
        let k = 10;
        let num_queries = 20;

        let mut rng = rand::rng();
        let mut idx = HnswIndex::new(HnswConfig {
            dimensions: dims,
            ef_construction: 200,
            ef_search: 100, // boost ef_search for better recall in test
            ..HnswConfig::default()
        });

        // Generate and insert random vectors
        let mut vectors: Vec<(u64, Vec<f32>)> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dims).map(|_| rng.random::<f32>()).collect();
            idx.insert(i as u64, &v);
            vectors.push((i as u64, v));
        }
        assert_eq!(idx.len(), n);

        // Measure recall@k across several queries
        let mut total_recall = 0.0;
        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dims).map(|_| rng.random::<f32>()).collect();

            let hnsw_results = idx.search(&query, k);
            let bf_results = brute_force_knn(&vectors, &query, k, DistanceMetric::Euclidean);

            let hnsw_ids: HashSet<u64> = hnsw_results.iter().map(|r| r.0).collect();
            let bf_ids: HashSet<u64> = bf_results.iter().map(|r| r.0).collect();

            let overlap = hnsw_ids.intersection(&bf_ids).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.8,
            "Recall@{k} = {avg_recall:.3}, expected > 0.8"
        );
    }

    #[test]
    fn test_duplicate_insert_replaces() {
        let mut idx = HnswIndex::new(make_config(2));
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(1, &[10.0, 10.0]); // replace
        assert_eq!(idx.len(), 1);

        let results = idx.search(&[10.0, 10.0], 1);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 1e-5); // should find the updated vector
    }

    #[test]
    fn test_config_validation() {
        let mut cfg = HnswConfig::default();
        assert!(cfg.validate().is_err()); // dimensions == 0

        cfg.dimensions = 128;
        assert!(cfg.validate().is_ok());

        cfg.m = 0;
        assert!(cfg.validate().is_err());

        cfg.m = 16;
        cfg.ef_search = 0;
        assert!(cfg.validate().is_err());

        cfg.ef_search = 50;
        cfg.ef_construction = 0;
        assert!(cfg.validate().is_err());

        cfg.ef_construction = 200;
        cfg.m0 = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_search_zero_k() {
        let mut idx = HnswIndex::new(make_config(2));
        idx.insert(1, &[0.0, 0.0]);
        let results = idx.search(&[0.0, 0.0], 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_results_sorted_by_distance() {
        let mut idx = HnswIndex::new(make_config(2));
        for i in 0..50 {
            idx.insert(i, &[i as f32, 0.0]);
        }
        let results = idx.search(&[25.0, 0.0], 10);
        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1 + 1e-7,
                "Results not sorted: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }
}
