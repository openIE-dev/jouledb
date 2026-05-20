//! Shortest path algorithms — Dijkstra, Bellman-Ford, Floyd-Warshall, A*.
//!
//! Operates on weighted directed graphs represented as adjacency lists.
//! Supports path reconstruction, distance matrices, and negative cycle detection.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;

// ── Weighted Graph ───────────────────────────────────────────────────────────

/// A weighted directed graph for shortest path computation.
#[derive(Debug, Clone)]
pub struct WeightedGraph {
    /// Adjacency list: node -> vec of (neighbor, weight).
    adj: HashMap<usize, Vec<(usize, f64)>>,
    node_count: usize,
}

impl WeightedGraph {
    /// Create a new graph with `n` nodes labeled 0..n.
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

    /// Add a directed edge from `from` to `to` with the given weight.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.adj.entry(from).or_default().push((to, weight));
    }

    /// Add an undirected edge (both directions).
    pub fn add_undirected_edge(&mut self, a: usize, b: usize, weight: f64) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node: usize) -> &[(usize, f64)] {
        self.adj.get(&node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// All nodes.
    pub fn nodes(&self) -> Vec<usize> {
        (0..self.node_count).collect()
    }
}

// ── Dijkstra State ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DijkstraState {
    cost: f64,
    node: usize,
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap behavior
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

// ── Shortest Path Result ─────────────────────────────────────────────────────

/// Result of a single-source shortest path computation.
#[derive(Debug, Clone)]
pub struct ShortestPathResult {
    /// Source node.
    pub source: usize,
    /// Distance from source to each node. f64::INFINITY if unreachable.
    pub dist: HashMap<usize, f64>,
    /// Predecessor map for path reconstruction.
    pub prev: HashMap<usize, Option<usize>>,
}

impl ShortestPathResult {
    /// Reconstruct the path from source to the given target.
    pub fn path_to(&self, target: usize) -> Option<Vec<usize>> {
        if !self.dist.contains_key(&target) {
            return None;
        }
        let d = self.dist[&target];
        if d == f64::INFINITY {
            return None;
        }
        let mut path = Vec::new();
        let mut cur = Some(target);
        while let Some(node) = cur {
            path.push(node);
            if node == self.source {
                break;
            }
            cur = self.prev.get(&node).copied().flatten();
            if cur.is_none() && node != self.source {
                return None;
            }
        }
        path.reverse();
        if path.first() == Some(&self.source) {
            Some(path)
        } else {
            None
        }
    }

    /// Distance to target, or None if unreachable.
    pub fn distance_to(&self, target: usize) -> Option<f64> {
        self.dist.get(&target).copied().filter(|d| *d < f64::INFINITY)
    }
}

// ── Dijkstra ─────────────────────────────────────────────────────────────────

/// Dijkstra's algorithm using a binary heap. Requires non-negative edge weights.
pub fn dijkstra(graph: &WeightedGraph, source: usize) -> ShortestPathResult {
    let mut dist: HashMap<usize, f64> = HashMap::new();
    let mut prev: HashMap<usize, Option<usize>> = HashMap::new();

    for node in graph.nodes() {
        dist.insert(node, f64::INFINITY);
        prev.insert(node, None);
    }
    dist.insert(source, 0.0);

    let mut heap = BinaryHeap::new();
    heap.push(DijkstraState { cost: 0.0, node: source });

    while let Some(DijkstraState { cost, node }) = heap.pop() {
        let current_dist = dist[&node];
        if cost > current_dist {
            continue;
        }
        for &(next, weight) in graph.neighbors(node) {
            let new_dist = cost + weight;
            if new_dist < dist.get(&next).copied().unwrap_or(f64::INFINITY) {
                dist.insert(next, new_dist);
                prev.insert(next, Some(node));
                heap.push(DijkstraState { cost: new_dist, node: next });
            }
        }
    }

    ShortestPathResult { source, dist, prev }
}

// ── Bellman-Ford ─────────────────────────────────────────────────────────────

/// Result of Bellman-Ford, which may detect negative cycles.
#[derive(Debug, Clone)]
pub enum BellmanFordResult {
    /// Shortest paths found.
    Ok(ShortestPathResult),
    /// Negative cycle detected. Contains the cycle as a sequence of nodes.
    NegativeCycle(Vec<usize>),
}

/// Bellman-Ford algorithm. Handles negative edge weights.
pub fn bellman_ford(graph: &WeightedGraph, source: usize) -> BellmanFordResult {
    let n = graph.node_count();
    let mut dist: HashMap<usize, f64> = HashMap::new();
    let mut prev: HashMap<usize, Option<usize>> = HashMap::new();

    for node in graph.nodes() {
        dist.insert(node, f64::INFINITY);
        prev.insert(node, None);
    }
    dist.insert(source, 0.0);

    // Collect all edges
    let mut edges = Vec::new();
    for from in graph.nodes() {
        for &(to, w) in graph.neighbors(from) {
            edges.push((from, to, w));
        }
    }

    // Relax edges n-1 times
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for &(from, to, w) in &edges {
            let d_from = dist[&from];
            if d_from < f64::INFINITY {
                let new_dist = d_from + w;
                if new_dist < dist[&to] {
                    dist.insert(to, new_dist);
                    prev.insert(to, Some(from));
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Check for negative cycles
    for &(from, to, w) in &edges {
        let d_from = dist[&from];
        if d_from < f64::INFINITY && d_from + w < dist[&to] {
            // Negative cycle detected — trace it
            let mut cycle_node = to;
            let mut visited = vec![false; n];
            // Walk back n times to ensure we are in the cycle
            for _ in 0..n {
                if let Some(Some(p)) = prev.get(&cycle_node) {
                    cycle_node = *p;
                }
            }
            let start = cycle_node;
            let mut cycle = vec![start];
            if let Some(Some(p)) = prev.get(&cycle_node) {
                cycle_node = *p;
            }
            visited[start] = true;
            while cycle_node != start {
                cycle.push(cycle_node);
                if let Some(Some(p)) = prev.get(&cycle_node) {
                    cycle_node = *p;
                } else {
                    break;
                }
            }
            cycle.push(start);
            cycle.reverse();
            return BellmanFordResult::NegativeCycle(cycle);
        }
    }

    BellmanFordResult::Ok(ShortestPathResult { source, dist, prev })
}

// ── Floyd-Warshall ───────────────────────────────────────────────────────────

/// All-pairs shortest path distance matrix.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// dist[i][j] = shortest distance from i to j. f64::INFINITY if unreachable.
    pub dist: Vec<Vec<f64>>,
    /// next[i][j] = next node on shortest path from i to j.
    pub next: Vec<Vec<Option<usize>>>,
    /// Number of nodes.
    pub n: usize,
}

impl DistanceMatrix {
    /// Distance from i to j.
    pub fn distance(&self, from: usize, to: usize) -> f64 {
        self.dist[from][to]
    }

    /// Reconstruct path from `from` to `to`.
    pub fn path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if self.dist[from][to] == f64::INFINITY {
            return None;
        }
        let mut path = vec![from];
        let mut cur = from;
        while cur != to {
            match self.next[cur][to] {
                Some(n) => {
                    cur = n;
                    path.push(cur);
                }
                None => return None,
            }
        }
        Some(path)
    }

    /// Check if any negative cycle exists.
    pub fn has_negative_cycle(&self) -> bool {
        for i in 0..self.n {
            if self.dist[i][i] < 0.0 {
                return true;
            }
        }
        false
    }
}

/// Floyd-Warshall all-pairs shortest path.
pub fn floyd_warshall(graph: &WeightedGraph) -> DistanceMatrix {
    let n = graph.node_count();
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut next: Vec<Vec<Option<usize>>> = vec![vec![None; n]; n];

    // Initialize
    for i in 0..n {
        dist[i][i] = 0.0;
    }
    for from in graph.nodes() {
        for &(to, w) in graph.neighbors(from) {
            if w < dist[from][to] {
                dist[from][to] = w;
                next[from][to] = Some(to);
            }
        }
    }

    // DP
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_k = dist[i][k] + dist[k][j];
                if through_k < dist[i][j] {
                    dist[i][j] = through_k;
                    next[i][j] = next[i][k];
                }
            }
        }
    }

    DistanceMatrix { dist, next, n }
}

// ── A* ───────────────────────────────────────────────────────────────────────

/// A* search with a heuristic function.
pub fn astar<H>(
    graph: &WeightedGraph,
    source: usize,
    target: usize,
    heuristic: H,
) -> Option<(Vec<usize>, f64)>
where
    H: Fn(usize) -> f64,
{
    let mut g_score: HashMap<usize, f64> = HashMap::new();
    let mut prev: HashMap<usize, usize> = HashMap::new();
    let mut heap = BinaryHeap::new();

    g_score.insert(source, 0.0);
    heap.push(DijkstraState {
        cost: heuristic(source),
        node: source,
    });

    while let Some(DijkstraState { node, .. }) = heap.pop() {
        if node == target {
            // Reconstruct
            let mut path = vec![target];
            let mut cur = target;
            while let Some(&p) = prev.get(&cur) {
                path.push(p);
                cur = p;
            }
            path.reverse();
            let cost = g_score[&target];
            return Some((path, cost));
        }

        let current_g = g_score[&node];

        for &(next, weight) in graph.neighbors(node) {
            let tentative_g = current_g + weight;
            if tentative_g < g_score.get(&next).copied().unwrap_or(f64::INFINITY) {
                g_score.insert(next, tentative_g);
                prev.insert(next, node);
                let f = tentative_g + heuristic(next);
                heap.push(DijkstraState { cost: f, node: next });
            }
        }
    }

    None
}

/// Dijkstra single-pair: returns shortest distance and path from source to target.
pub fn dijkstra_path(
    graph: &WeightedGraph,
    source: usize,
    target: usize,
) -> Option<(Vec<usize>, f64)> {
    let result = dijkstra(graph, source);
    let dist = result.distance_to(target)?;
    let path = result.path_to(target)?;
    Some((path, dist))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> WeightedGraph {
        // 0 --(1)--> 1 --(2)--> 3
        // 0 --(4)--> 2 --(1)--> 3
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 4.0);
        g.add_edge(1, 3, 2.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(1, 2, 1.0);
        g
    }

    #[test]
    fn test_dijkstra_distances() {
        let g = sample_graph();
        let result = dijkstra(&g, 0);
        assert_eq!(result.distance_to(0), Some(0.0));
        assert_eq!(result.distance_to(1), Some(1.0));
        assert_eq!(result.distance_to(2), Some(2.0)); // 0->1->2
        assert_eq!(result.distance_to(3), Some(3.0)); // 0->1->3 or 0->1->2->3
    }

    #[test]
    fn test_dijkstra_path_reconstruction() {
        let g = sample_graph();
        let result = dijkstra(&g, 0);
        let path = result.path_to(3).unwrap();
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_dijkstra_unreachable() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        // 2 is unreachable
        let result = dijkstra(&g, 0);
        assert!(result.distance_to(2).is_none());
        assert!(result.path_to(2).is_none());
    }

    #[test]
    fn test_bellman_ford_positive() {
        let g = sample_graph();
        match bellman_ford(&g, 0) {
            BellmanFordResult::Ok(result) => {
                assert_eq!(result.distance_to(3), Some(3.0));
            }
            BellmanFordResult::NegativeCycle(_) => panic!("unexpected negative cycle"),
        }
    }

    #[test]
    fn test_bellman_ford_negative_edge() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 5.0);
        g.add_edge(1, 2, -3.0);
        g.add_edge(0, 2, 4.0);
        match bellman_ford(&g, 0) {
            BellmanFordResult::Ok(result) => {
                assert_eq!(result.distance_to(2), Some(2.0)); // 0->1->2 = 5-3=2
            }
            BellmanFordResult::NegativeCycle(_) => panic!("no cycle here"),
        }
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, -3.0);
        g.add_edge(2, 0, 1.0);
        match bellman_ford(&g, 0) {
            BellmanFordResult::NegativeCycle(cycle) => {
                assert!(cycle.len() >= 2);
            }
            BellmanFordResult::Ok(_) => panic!("should detect negative cycle"),
        }
    }

    #[test]
    fn test_floyd_warshall_basic() {
        let g = sample_graph();
        let dm = floyd_warshall(&g);
        assert_eq!(dm.distance(0, 3), 3.0);
        assert_eq!(dm.distance(0, 0), 0.0);
    }

    #[test]
    fn test_floyd_warshall_path_reconstruction() {
        let g = sample_graph();
        let dm = floyd_warshall(&g);
        let path = dm.path(0, 3).unwrap();
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_floyd_warshall_unreachable() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        let dm = floyd_warshall(&g);
        assert_eq!(dm.distance(0, 2), f64::INFINITY);
        assert!(dm.path(0, 2).is_none());
    }

    #[test]
    fn test_floyd_warshall_no_negative_cycle() {
        let g = sample_graph();
        let dm = floyd_warshall(&g);
        assert!(!dm.has_negative_cycle());
    }

    #[test]
    fn test_astar_basic() {
        let g = sample_graph();
        // Trivial heuristic (h=0 => behaves like Dijkstra)
        let result = astar(&g, 0, 3, |_| 0.0);
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 3.0);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_astar_with_heuristic() {
        // Linear graph: 0 -> 1 -> 2 -> 3
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        // Heuristic: distance from target
        let result = astar(&g, 0, 3, |node| (3 - node) as f64);
        let (path, cost) = result.unwrap();
        assert_eq!(cost, 3.0);
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_astar_no_path() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        let result = astar(&g, 0, 2, |_| 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_path_fn() {
        let g = sample_graph();
        let (path, dist) = dijkstra_path(&g, 0, 3).unwrap();
        assert_eq!(dist, 3.0);
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_single_node_graph() {
        let g = WeightedGraph::new(1);
        let result = dijkstra(&g, 0);
        assert_eq!(result.distance_to(0), Some(0.0));
    }

    #[test]
    fn test_weighted_graph_construction() {
        let mut g = WeightedGraph::new(3);
        g.add_undirected_edge(0, 1, 5.0);
        assert_eq!(g.neighbors(0).len(), 1);
        assert_eq!(g.neighbors(1).len(), 1);
        assert_eq!(g.node_count(), 3);
    }

    #[test]
    fn test_floyd_warshall_all_pairs() {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 3.0);
        g.add_edge(1, 2, 4.0);
        g.add_edge(0, 2, 8.0);
        let dm = floyd_warshall(&g);
        assert_eq!(dm.distance(0, 2), 7.0); // 0->1->2
    }

    #[test]
    fn test_dijkstra_self_loop() {
        let mut g = WeightedGraph::new(2);
        g.add_edge(0, 0, 5.0);
        g.add_edge(0, 1, 1.0);
        let result = dijkstra(&g, 0);
        assert_eq!(result.distance_to(1), Some(1.0));
    }
}
