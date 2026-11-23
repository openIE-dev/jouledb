//! Deterministic HNSW (Hierarchical Navigable Small World) approximate
//! nearest-neighbor index — the ANN index behind `CREATE VECTOR INDEX ... USING
//! HNSW`. Replaces the O(n) brute-force scan over the vector distance operators
//! with a graph index that answers top-k in ~O(log n).
//!
//! **Determinism (JouleDB byte-determinism invariant).** HNSW is normally
//! randomized (per-insert level assignment). Here every source of nondeterminism
//! is pinned: the level of a node is derived from a seeded splitmix64 PRNG keyed
//! by the node's stable id, the node store is a `BTreeMap` (never `HashMap`), and
//! all candidate ordering / neighbor selection / tie-breaking is by
//! `(distance, id)`. The same inserts in the same order therefore always build a
//! byte-identical graph, so an index rebuild reproduces exactly.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs" (TPAMI 2018).

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Distance metric — mirrors the SQL vector operators (`<->`, `<=>`, `<#>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Euclidean / L2 (`<->`).
    L2,
    /// Cosine distance `1 - cos` (`<=>`).
    Cosine,
    /// Negative inner product (`<#>`); smaller = more similar.
    Ip,
}

impl Metric {
    /// Distance between two equal-length vectors. Smaller = nearer for all three
    /// (cosine and IP are returned in "distance" orientation so the same
    /// nearest-is-smallest search works for every metric).
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt(),
            Metric::Cosine => {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if na == 0.0 || nb == 0.0 {
                    1.0
                } else {
                    1.0 - dot / (na * nb)
                }
            }
            Metric::Ip => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        }
    }
}

/// Build/search parameters. Defaults follow common HNSW practice (`M=16`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswParams {
    /// Max neighbors per node per layer above 0 (layer 0 uses `2*M`).
    pub m: usize,
    /// Candidate-list width during insertion (higher = better graph, slower build).
    pub ef_construction: usize,
    /// Default candidate-list width during search (higher = better recall).
    pub ef_search: usize,
    /// PRNG seed for level assignment (fixed → deterministic builds).
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x51ED_2701_C0FF_EE00,
        }
    }
}

/// One node: its stable id, its vector, and its neighbor lists per layer
/// (`neighbors[l]` = sorted-unique neighbor ids at layer `l`).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    vector: Vec<f32>,
    neighbors: Vec<Vec<u64>>,
}

/// A deterministic HNSW index over `Vec<f32>` vectors keyed by `u64` ids.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hnsw {
    metric: Metric,
    params: HnswParams,
    dim: usize,
    nodes: BTreeMap<u64, Node>,
    entry: Option<u64>,
    /// Highest layer any node currently occupies.
    top_layer: usize,
    /// Normalization factor `1 / ln(M)` for level assignment.
    ml: f64,
}

/// Ordering key `(distance, id)` giving a *total* order over `(f32, u64)` (no
/// NaN for finite inputs; `total_cmp` is the tie-safe fallback). Used so heaps
/// and neighbor selection are fully deterministic.
#[derive(Debug, Clone, Copy)]
struct Cand {
    dist: f32,
    id: u64,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id && self.dist.to_bits() == o.dist.to_bits()
    }
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// splitmix64 — a tiny, fast, deterministic PRNG for level assignment.
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Hnsw {
    pub fn new(metric: Metric, params: HnswParams, dim: usize) -> Self {
        let ml = if params.m > 1 {
            1.0 / (params.m as f64).ln()
        } else {
            1.0
        };
        Self {
            metric,
            params,
            dim,
            nodes: BTreeMap::new(),
            entry: None,
            top_layer: 0,
            ml,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }

    fn m_max(&self, layer: usize) -> usize {
        if layer == 0 {
            self.params.m * 2
        } else {
            self.params.m
        }
    }

    /// Deterministic level for a node id: `floor(-ln(u) * ml)` with `u` from the
    /// seeded PRNG. Higher ids don't bias levels; the distribution is the standard
    /// exponential decay HNSW relies on.
    fn level_for(&self, id: u64) -> usize {
        let r = splitmix64(self.params.seed ^ id);
        // Map to (0,1]; avoid 0 so ln is finite.
        let u = ((r >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        (-u.ln() * self.ml).floor() as usize
    }

    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        self.metric.distance(a, b)
    }

    /// Insert (or replace) a vector under `id`. Dimension mismatch is ignored
    /// (caller validates); a replace removes the old node first.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if vector.len() != self.dim {
            return;
        }
        if self.nodes.contains_key(&id) {
            self.remove(id);
        }
        let node_level = self.level_for(id);
        let mut neighbors = Vec::with_capacity(node_level + 1);
        neighbors.resize(node_level + 1, Vec::new());
        self.nodes.insert(id, Node { vector, neighbors });

        // First node becomes the entry point.
        let entry = match self.entry {
            None => {
                self.entry = Some(id);
                self.top_layer = node_level;
                return;
            }
            Some(e) => e,
        };

        let vec = self.nodes[&id].vector.clone();
        let mut ep = entry;

        // Descend from the top layer down to node_level+1 with a greedy 1-NN walk.
        // Build-time distance ops aren't metered (only queries are).
        let mut sink = 0u64;
        let mut layer = self.top_layer;
        while layer > node_level {
            ep = self.greedy_nearest(&vec, ep, layer, &mut sink);
            layer -= 1;
        }

        // From node_level down to 0: ef-search, select neighbors, link both ways.
        let mut layer = node_level.min(self.top_layer);
        loop {
            let found =
                self.search_layer(&vec, &[ep], self.params.ef_construction, layer, &mut sink);
            let selected = self.select_neighbors(&vec, &found, self.params.m);
            // Link id -> selected.
            if let Some(n) = self.nodes.get_mut(&id) {
                n.neighbors[layer] = selected.clone();
            }
            // Link selected -> id, pruning each back to m_max(layer).
            for &nb in &selected {
                self.connect(nb, id, layer);
            }
            ep = found.first().map(|c| c.id).unwrap_or(ep);
            if layer == 0 {
                break;
            }
            layer -= 1;
        }

        if node_level > self.top_layer {
            self.top_layer = node_level;
            self.entry = Some(id);
        }
    }

    /// Add `to` as a neighbor of `from` at `layer`, pruning `from`'s neighbor set
    /// back to `m_max(layer)` by the diversity heuristic if it overflows.
    fn connect(&mut self, from: u64, to: u64, layer: usize) {
        let from_vec = self.nodes[&from].vector.clone();
        let list = {
            let Some(n) = self.nodes.get_mut(&from) else {
                return;
            };
            if layer >= n.neighbors.len() {
                return;
            }
            if !n.neighbors[layer].contains(&to) {
                n.neighbors[layer].push(to);
            }
            n.neighbors[layer].clone()
        };
        if list.len() > self.m_max(layer) {
            let cands: Vec<Cand> = list
                .iter()
                .map(|&nb| Cand {
                    dist: self.dist(&from_vec, &self.nodes[&nb].vector),
                    id: nb,
                })
                .collect();
            let pruned = self.select_neighbors(&from_vec, &cands, self.m_max(layer));
            if let Some(n) = self.nodes.get_mut(&from) {
                n.neighbors[layer] = pruned;
            }
        }
    }

    /// Greedy single-nearest walk at one layer (used for the upper-layer descent).
    /// `ops` accumulates the number of distance computations (a deterministic
    /// proxy for the query's compute energy).
    fn greedy_nearest(&self, query: &[f32], entry: u64, layer: usize, ops: &mut u64) -> u64 {
        let mut cur = entry;
        *ops += 1;
        let mut cur_d = self.dist(query, &self.nodes[&cur].vector);
        loop {
            let mut improved = false;
            if let Some(node) = self.nodes.get(&cur) {
                if layer < node.neighbors.len() {
                    for &nb in &node.neighbors[layer] {
                        let nb_vec = match self.nodes.get(&nb) {
                            Some(n) => &n.vector,
                            None => continue,
                        };
                        *ops += 1;
                        let d = self.dist(query, nb_vec);
                        if d < cur_d || (d == cur_d && nb < cur) {
                            cur = nb;
                            cur_d = d;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Layer-local best-first search: return up to `ef` nearest candidates to
    /// `query` reachable from `entry_points`, ascending by `(distance, id)`.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[u64],
        ef: usize,
        layer: usize,
        ops: &mut u64,
    ) -> Vec<Cand> {
        let mut visited: BTreeSet<u64> = BTreeSet::new();
        // Candidates: min-heap by distance (explore nearest first) via Reverse.
        let mut candidates: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new();
        // Results: max-heap by distance so the farthest is at the top for eviction.
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();

        for &ep in entry_points {
            if let Some(node) = self.nodes.get(&ep) {
                *ops += 1;
                let d = self.dist(query, &node.vector);
                let c = Cand { dist: d, id: ep };
                visited.insert(ep);
                candidates.push(std::cmp::Reverse(c));
                results.push(c);
            }
        }

        while let Some(std::cmp::Reverse(cur)) = candidates.pop() {
            // Stop if the nearest unexplored candidate is farther than our current
            // worst result and we already have `ef` results.
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if cur.dist > worst.dist {
                        break;
                    }
                }
            }
            let neighbors = match self.nodes.get(&cur.id) {
                Some(n) if layer < n.neighbors.len() => n.neighbors[layer].clone(),
                _ => Vec::new(),
            };
            for nb in neighbors {
                if !visited.insert(nb) {
                    continue;
                }
                // Defensive: a pruned link can outlive its target after removal.
                let nb_vec = match self.nodes.get(&nb) {
                    Some(n) => &n.vector,
                    None => continue,
                };
                *ops += 1;
                let d = self.dist(query, nb_vec);
                let c = Cand { dist: d, id: nb };
                let worst = results.peek().copied();
                if results.len() < ef || worst.map(|w| c.dist < w.dist).unwrap_or(true) {
                    candidates.push(std::cmp::Reverse(c));
                    results.push(c);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<Cand> = results.into_vec();
        out.sort();
        out
    }

    /// Neighbor-selection heuristic (HNSW Algorithm 4, simplified): from the
    /// candidate pool keep up to `m` that are closer to `base` than to any
    /// already-selected neighbor — favoring a diverse, navigable neighborhood
    /// over the naive m-nearest. Deterministic (candidates pre-sorted).
    fn select_neighbors(&self, base: &[f32], candidates: &[Cand], m: usize) -> Vec<u64> {
        let mut pool: Vec<Cand> = candidates.to_vec();
        pool.sort(); // ascending by (distance, id)
        let mut selected: Vec<u64> = Vec::with_capacity(m);
        for c in &pool {
            if selected.len() >= m {
                break;
            }
            let cand_vec = &self.nodes[&c.id].vector;
            // Keep c only if it is closer to base than to every chosen neighbor.
            let diverse = selected.iter().all(|&s| {
                let d_to_sel = self.dist(cand_vec, &self.nodes[&s].vector);
                c.dist <= d_to_sel
            });
            if diverse {
                selected.push(c.id);
            }
        }
        // If the heuristic was too strict to fill m, top up with the nearest
        // remaining candidates (keeps recall from collapsing on clustered data).
        if selected.len() < m {
            for c in &pool {
                if selected.len() >= m {
                    break;
                }
                if !selected.contains(&c.id) {
                    selected.push(c.id);
                }
            }
        }
        selected
    }

    /// Search: the `k` nearest ids to `query`, ascending by distance. `ef` widens
    /// the candidate list (defaults to `max(k, ef_search)`).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_ef(query, k, self.params.ef_search)
    }

    pub fn search_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        self.search_ef_metered(query, k, ef).0
    }

    /// Like [`search`], but also returns the number of distance computations
    /// performed during the traversal — a deterministic, hardware-independent
    /// proxy for the query's compute energy (the same graph + query always
    /// yields the same count). A brute-force scan would perform [`len`] such
    /// computations; the ratio is the index's energy saving.
    pub fn search_metered(&self, query: &[f32], k: usize) -> (Vec<(u64, f32)>, u64) {
        self.search_ef_metered(query, k, self.params.ef_search)
    }

    fn search_ef_metered(&self, query: &[f32], k: usize, ef: usize) -> (Vec<(u64, f32)>, u64) {
        let Some(mut ep) = self.entry else {
            return (Vec::new(), 0);
        };
        if query.len() != self.dim || k == 0 {
            return (Vec::new(), 0);
        }
        let mut ops = 0u64;
        let mut layer = self.top_layer;
        while layer > 0 {
            ep = self.greedy_nearest(query, ep, layer, &mut ops);
            layer -= 1;
        }
        let ef = ef.max(k);
        let found = self.search_layer(query, &[ep], ef, 0, &mut ops);
        let out = found.into_iter().take(k).map(|c| (c.id, c.dist)).collect();
        (out, ops)
    }

    /// Remove a node and unlink it everywhere. HNSW links can be ASYMMETRIC after
    /// neighbor-list pruning (X may point to id without id pointing back), so we
    /// scan every node's every layer — not just the removed node's own neighbors.
    pub fn remove(&mut self, id: u64) {
        if self.nodes.remove(&id).is_none() {
            return;
        }
        for n in self.nodes.values_mut() {
            for layer in n.neighbors.iter_mut() {
                layer.retain(|&x| x != id);
            }
        }
        // Repair the entry point if we removed it.
        if self.entry == Some(id) {
            self.entry = self.nodes.keys().next().copied();
            self.top_layer = self
                .nodes
                .values()
                .map(|n| n.neighbors.len().saturating_sub(1))
                .max()
                .unwrap_or(0);
        }
    }
}

/// Exact brute-force top-k — the reference oracle for tests and the fallback for
/// tiny tables where building a graph isn't worth it.
pub fn brute_force_topk(
    metric: Metric,
    vectors: &[(u64, Vec<f32>)],
    query: &[f32],
    k: usize,
) -> Vec<(u64, f32)> {
    let mut all: Vec<Cand> = vectors
        .iter()
        .filter(|(_, v)| v.len() == query.len())
        .map(|(id, v)| Cand {
            dist: metric.distance(v, query),
            id: *id,
        })
        .collect();
    all.sort();
    all.into_iter().take(k).map(|c| (c.id, c.dist)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic vector generator (no rand dep): a seeded LCG per component.
    fn gen_vectors(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32
        };
        (0..n as u64)
            .map(|id| (id, (0..dim).map(|_| next() * 2.0 - 1.0).collect()))
            .collect()
    }

    fn recall_at_k(metric: Metric, n: usize, dim: usize, k: usize, queries: usize) -> f64 {
        let data = gen_vectors(n, dim, 0xABCD);
        let mut idx = Hnsw::new(metric, HnswParams::default(), dim);
        for (id, v) in &data {
            idx.insert(*id, v.clone());
        }
        assert_eq!(idx.len(), n);
        let qs = gen_vectors(queries, dim, 0x1234);
        let mut hits = 0usize;
        let mut total = 0usize;
        for (_, q) in &qs {
            let exact: BTreeSet<u64> = brute_force_topk(metric, &data, q, k)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let approx: BTreeSet<u64> = idx
                .search_ef(q, k, 128)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            hits += exact.intersection(&approx).count();
            total += exact.len();
        }
        hits as f64 / total as f64
    }

    #[test]
    fn recall_is_high_l2() {
        // A correct HNSW must recover the exact top-k with high recall.
        let r = recall_at_k(Metric::L2, 1000, 16, 10, 50);
        assert!(r >= 0.90, "L2 recall@10 too low: {r}");
    }

    #[test]
    fn recall_is_high_cosine() {
        let r = recall_at_k(Metric::Cosine, 1000, 16, 10, 50);
        assert!(r >= 0.90, "cosine recall@10 too low: {r}");
    }

    #[test]
    fn recall_is_high_ip() {
        let r = recall_at_k(Metric::Ip, 800, 12, 10, 40);
        assert!(r >= 0.88, "IP recall@10 too low: {r}");
    }

    #[test]
    fn build_is_deterministic() {
        // Byte-determinism invariant: identical inserts → identical graph → identical search.
        let data = gen_vectors(300, 8, 7);
        let build = || {
            let mut idx = Hnsw::new(Metric::L2, HnswParams::default(), 8);
            for (id, v) in &data {
                idx.insert(*id, v.clone());
            }
            idx
        };
        let a = build();
        let b = build();
        let q = vec![0.1f32; 8];
        assert_eq!(a.search_ef(&q, 10, 64), b.search_ef(&q, 10, 64));
        // The whole graph structure matches, not just one query.
        for id in 0..300u64 {
            assert_eq!(
                a.nodes[&id].neighbors, b.nodes[&id].neighbors,
                "node {id} differs"
            );
        }
    }

    #[test]
    fn exact_on_tiny_sets() {
        // With few points, HNSW should return the true nearest (ef covers all).
        let data = gen_vectors(20, 4, 3);
        let mut idx = Hnsw::new(Metric::L2, HnswParams::default(), 4);
        for (id, v) in &data {
            idx.insert(*id, v.clone());
        }
        let q = data[5].1.clone();
        let got = idx.search_ef(&q, 5, 64);
        let exact = brute_force_topk(Metric::L2, &data, &q, 5);
        assert_eq!(got, exact, "tiny-set search must be exact");
        // Its own vector is the nearest (distance 0).
        assert_eq!(got[0].0, 5);
        assert!(got[0].1.abs() < 1e-6);
    }

    #[test]
    fn empty_single_and_edge_cases() {
        let mut idx = Hnsw::new(Metric::L2, HnswParams::default(), 4);
        assert!(
            idx.search(&[0.0; 4], 5).is_empty(),
            "empty index → no results"
        );
        idx.insert(1, vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5), vec![(1, 0.0)]);
        // Dimension mismatch is ignored on insert and empty on search.
        idx.insert(2, vec![1.0, 2.0]);
        assert_eq!(idx.len(), 1);
        assert!(idx.search(&[1.0, 2.0], 3).is_empty());
        // k = 0 → empty.
        assert!(idx.search(&[1.0, 0.0, 0.0, 0.0], 0).is_empty());
    }

    #[test]
    fn remove_unlinks_and_search_still_works() {
        let data = gen_vectors(200, 8, 11);
        let mut idx = Hnsw::new(Metric::L2, HnswParams::default(), 8);
        for (id, v) in &data {
            idx.insert(*id, v.clone());
        }
        // Remove the exact answer for a query and confirm it no longer appears.
        let q = data[42].1.clone();
        idx.remove(42);
        assert_eq!(idx.len(), 199);
        let got = idx.search_ef(&q, 10, 128);
        assert!(
            !got.iter().any(|(id, _)| *id == 42),
            "removed id must not appear"
        );
        // Graph is still navigable: results are sorted and non-empty.
        assert!(!got.is_empty());
        assert!(
            got.windows(2).all(|w| w[0].1 <= w[1].1),
            "results must be ascending by distance"
        );
    }
}
