//! Minimum spanning tree — Kruskal's, Prim's, and Boruvka's algorithms.
//!
//! Operates on weighted undirected graphs. Returns MST edge lists, total weight,
//! and handles disconnected graphs (spanning forest).

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// ── MST Edge ─────────────────────────────────────────────────────────────────

/// An edge in the MST.
#[derive(Debug, Clone, PartialEq)]
pub struct MstEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

// ── MST Result ───────────────────────────────────────────────────────────────

/// Result of an MST computation.
#[derive(Debug, Clone)]
pub struct MstResult {
    /// Edges in the MST.
    pub edges: Vec<MstEdge>,
    /// Total weight of the MST.
    pub total_weight: f64,
    /// Number of connected components (1 for a connected graph).
    pub component_count: usize,
}

impl MstResult {
    /// Whether this is a spanning tree (single component) or forest.
    pub fn is_spanning_tree(&self) -> bool {
        self.component_count == 1
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// ── Undirected Graph ─────────────────────────────────────────────────────────

/// Weighted undirected graph for MST algorithms.
#[derive(Debug, Clone)]
pub struct UndirectedGraph {
    adj: HashMap<usize, Vec<(usize, f64)>>,
    node_count: usize,
}

impl UndirectedGraph {
    /// Create a graph with `n` nodes labeled 0..n.
    pub fn new(n: usize) -> Self {
        let mut adj = HashMap::new();
        for i in 0..n {
            adj.insert(i, Vec::new());
        }
        Self { adj, node_count: n }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Add an undirected edge.
    pub fn add_edge(&mut self, a: usize, b: usize, weight: f64) {
        self.adj.entry(a).or_default().push((b, weight));
        self.adj.entry(b).or_default().push((a, weight));
    }

    /// Get neighbors with weights.
    pub fn neighbors(&self, node: usize) -> &[(usize, f64)] {
        self.adj.get(&node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// All nodes.
    pub fn nodes(&self) -> Vec<usize> {
        (0..self.node_count).collect()
    }

    /// Collect all unique edges (each undirected edge once).
    /// Parallel edges (same endpoints, different weights) are preserved.
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        let mut result = Vec::new();
        for node in 0..self.node_count {
            for &(nb, w) in self.neighbors(node) {
                // Only emit from the lower-numbered endpoint to avoid duplicates
                // from the symmetric adjacency list representation.
                if node <= nb {
                    result.push((node, nb, w));
                }
            }
        }
        result
    }
}

// ── Union-Find (internal) ────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    count: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            count: n,
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            Ordering::Less => self.parent[ra] = rb,
            Ordering::Greater => self.parent[rb] = ra,
            Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        self.count -= 1;
        true
    }

    fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    fn component_count(&self) -> usize {
        self.count
    }
}

// ── Kruskal's Algorithm ──────────────────────────────────────────────────────

/// Kruskal's MST using union-find.
pub fn kruskal(graph: &UndirectedGraph) -> MstResult {
    let mut edges = graph.edges();
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

    let mut uf = UnionFind::new(graph.node_count());
    let mut mst_edges = Vec::new();
    let mut total = 0.0;

    for (a, b, w) in edges {
        if uf.union(a, b) {
            mst_edges.push(MstEdge { from: a, to: b, weight: w });
            total += w;
        }
    }

    MstResult {
        edges: mst_edges,
        total_weight: total,
        component_count: uf.component_count(),
    }
}

// ── Prim's Algorithm ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PrimState {
    weight: f64,
    node: usize,
    from: usize,
}

impl PartialEq for PrimState {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight && self.node == other.node
    }
}

impl Eq for PrimState {}

impl PartialOrd for PrimState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrimState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .weight
            .partial_cmp(&self.weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

/// Prim's MST using a priority queue. Handles disconnected graphs (forest).
pub fn prim(graph: &UndirectedGraph) -> MstResult {
    let n = graph.node_count();
    if n == 0 {
        return MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
            component_count: 0,
        };
    }

    let mut in_mst = vec![false; n];
    let mut mst_edges = Vec::new();
    let mut total = 0.0;
    let mut component_count = 0;

    for start in 0..n {
        if in_mst[start] {
            continue;
        }
        component_count += 1;

        let mut heap = BinaryHeap::new();
        in_mst[start] = true;
        for &(nb, w) in graph.neighbors(start) {
            heap.push(PrimState { weight: w, node: nb, from: start });
        }

        while let Some(PrimState { weight, node, from }) = heap.pop() {
            if in_mst[node] {
                continue;
            }
            in_mst[node] = true;
            mst_edges.push(MstEdge { from, to: node, weight });
            total += weight;

            for &(nb, w) in graph.neighbors(node) {
                if !in_mst[nb] {
                    heap.push(PrimState { weight: w, node: nb, from: node });
                }
            }
        }
    }

    MstResult {
        edges: mst_edges,
        total_weight: total,
        component_count,
    }
}

// ── Boruvka's Algorithm ──────────────────────────────────────────────────────

/// Boruvka's MST algorithm.
pub fn boruvka(graph: &UndirectedGraph) -> MstResult {
    let n = graph.node_count();
    if n == 0 {
        return MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
            component_count: 0,
        };
    }

    let all_edges = graph.edges();
    let mut uf = UnionFind::new(n);
    let mut mst_edges: Vec<MstEdge> = Vec::new();
    let mut total = 0.0;

    loop {
        // For each component, find the cheapest outgoing edge.
        // cheapest[component_root] = (edge_index, weight)
        let mut cheapest: HashMap<usize, (usize, f64)> = HashMap::new();

        for (idx, &(a, b, w)) in all_edges.iter().enumerate() {
            let ca = uf.find(a);
            let cb = uf.find(b);
            if ca == cb {
                continue;
            }
            // Check component ca
            if let Some((_, best_w)) = cheapest.get(&ca) {
                if w < *best_w {
                    cheapest.insert(ca, (idx, w));
                }
            } else {
                cheapest.insert(ca, (idx, w));
            }
            // Check component cb
            if let Some((_, best_w)) = cheapest.get(&cb) {
                if w < *best_w {
                    cheapest.insert(cb, (idx, w));
                }
            } else {
                cheapest.insert(cb, (idx, w));
            }
        }

        if cheapest.is_empty() {
            break;
        }

        // Add cheapest edges (dedup by checking connectivity)
        let mut added = false;
        let mut edge_indices: Vec<usize> = cheapest.values().map(|(idx, _)| *idx).collect();
        edge_indices.sort();
        edge_indices.dedup();
        for idx in edge_indices {
            let (a, b, w) = all_edges[idx];
            if uf.union(a, b) {
                mst_edges.push(MstEdge { from: a, to: b, weight: w });
                total += w;
                added = true;
            }
        }

        if !added {
            break;
        }
    }

    MstResult {
        edges: mst_edges,
        total_weight: total,
        component_count: uf.component_count(),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> UndirectedGraph {
        // Triangle: 0--1 (1), 1--2 (2), 0--2 (3)
        // MST: edges (0,1,1) and (1,2,2), total = 3
        let mut g = UndirectedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 3.0);
        g
    }

    fn larger_graph() -> UndirectedGraph {
        // 0-1: 4, 0-7: 8, 1-2: 8, 1-7: 11, 2-3: 7, 2-5: 4, 2-8: 2
        // 3-4: 9, 3-5: 14, 4-5: 10, 5-6: 2, 6-7: 1, 6-8: 6, 7-8: 7
        let mut g = UndirectedGraph::new(9);
        g.add_edge(0, 1, 4.0);
        g.add_edge(0, 7, 8.0);
        g.add_edge(1, 2, 8.0);
        g.add_edge(1, 7, 11.0);
        g.add_edge(2, 3, 7.0);
        g.add_edge(2, 5, 4.0);
        g.add_edge(2, 8, 2.0);
        g.add_edge(3, 4, 9.0);
        g.add_edge(3, 5, 14.0);
        g.add_edge(4, 5, 10.0);
        g.add_edge(5, 6, 2.0);
        g.add_edge(6, 7, 1.0);
        g.add_edge(6, 8, 6.0);
        g.add_edge(7, 8, 7.0);
        g
    }

    #[test]
    fn test_kruskal_simple() {
        let g = sample_graph();
        let mst = kruskal(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.edge_count(), 2);
        assert!(mst.is_spanning_tree());
    }

    #[test]
    fn test_prim_simple() {
        let g = sample_graph();
        let mst = prim(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.edge_count(), 2);
    }

    #[test]
    fn test_boruvka_simple() {
        let g = sample_graph();
        let mst = boruvka(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.edge_count(), 2);
    }

    #[test]
    fn test_kruskal_larger() {
        let g = larger_graph();
        let mst = kruskal(&g);
        // Known MST weight for this classic example: 37
        assert_eq!(mst.total_weight, 37.0);
        assert_eq!(mst.edge_count(), 8); // n-1 edges
        assert!(mst.is_spanning_tree());
    }

    #[test]
    fn test_prim_larger() {
        let g = larger_graph();
        let mst = prim(&g);
        assert_eq!(mst.total_weight, 37.0);
        assert_eq!(mst.edge_count(), 8);
    }

    #[test]
    fn test_boruvka_larger() {
        let g = larger_graph();
        let mst = boruvka(&g);
        assert_eq!(mst.total_weight, 37.0);
        assert_eq!(mst.edge_count(), 8);
    }

    #[test]
    fn test_disconnected_kruskal() {
        let mut g = UndirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 2.0);
        let mst = kruskal(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.edge_count(), 2);
        assert_eq!(mst.component_count, 2);
        assert!(!mst.is_spanning_tree());
    }

    #[test]
    fn test_disconnected_prim() {
        let mut g = UndirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 2.0);
        let mst = prim(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.component_count, 2);
    }

    #[test]
    fn test_disconnected_boruvka() {
        let mut g = UndirectedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(2, 3, 2.0);
        let mst = boruvka(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.component_count, 2);
    }

    #[test]
    fn test_single_node() {
        let g = UndirectedGraph::new(1);
        let mst = kruskal(&g);
        assert_eq!(mst.total_weight, 0.0);
        assert_eq!(mst.edge_count(), 0);
        assert!(mst.is_spanning_tree());
    }

    #[test]
    fn test_empty_graph() {
        let g = UndirectedGraph::new(0);
        let mst = prim(&g);
        assert_eq!(mst.total_weight, 0.0);
        assert_eq!(mst.component_count, 0);
    }

    #[test]
    fn test_graph_edges_collection() {
        let g = sample_graph();
        let edges = g.edges();
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_all_algorithms_agree() {
        let g = larger_graph();
        let k = kruskal(&g);
        let p = prim(&g);
        let b = boruvka(&g);
        assert_eq!(k.total_weight, p.total_weight);
        assert_eq!(p.total_weight, b.total_weight);
        assert_eq!(k.edge_count(), p.edge_count());
        assert_eq!(p.edge_count(), b.edge_count());
    }

    #[test]
    fn test_parallel_edges() {
        let mut g = UndirectedGraph::new(2);
        g.add_edge(0, 1, 5.0);
        g.add_edge(0, 1, 3.0);
        let mst = kruskal(&g);
        assert_eq!(mst.total_weight, 3.0);
        assert_eq!(mst.edge_count(), 1);
    }

    #[test]
    fn test_mst_result_accessors() {
        let g = sample_graph();
        let mst = kruskal(&g);
        assert!(mst.is_spanning_tree());
        assert_eq!(mst.edge_count(), 2);
        assert!(!mst.edges.is_empty());
    }

    #[test]
    fn test_graph_neighbors() {
        let g = sample_graph();
        assert_eq!(g.neighbors(0).len(), 2);
        assert_eq!(g.node_count(), 3);
    }
}
