//! Shortest Path — Dijkstra, A* (haversine heuristic), bidirectional Dijkstra,
//! contraction hierarchies (simplified), turn restrictions, one-way handling.
//!
//! Pure-Rust shortest-path algorithms for road networks, operating on an
//! adjacency-list representation with multi-weight edges.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PathError {
    NodeNotFound(u64),
    NoPathFound,
    InvalidConfig(String),
    EmptyGraph,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::NoPathFound => write!(f, "no path found"),
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            Self::EmptyGraph => write!(f, "empty graph"),
        }
    }
}

impl std::error::Error for PathError {}

// ── Algorithm selector ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathAlgorithm {
    Dijkstra,
    AStar,
    BidirectionalDijkstra,
    ContractionHierarchy,
}

impl fmt::Display for PathAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Dijkstra => "dijkstra",
            Self::AStar => "a-star",
            Self::BidirectionalDijkstra => "bi-dijkstra",
            Self::ContractionHierarchy => "ch",
        };
        write!(f, "{s}")
    }
}

// ── Geo node ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoNode {
    pub id: u64,
    pub lat: f64,
    pub lon: f64,
}

impl GeoNode {
    pub fn new(id: u64, lat: f64, lon: f64) -> Self { Self { id, lat, lon } }
}

impl fmt::Display for GeoNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GeoN{}({:.5},{:.5})", self.id, self.lat, self.lon)
    }
}

// ── Weighted edge ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WEdge {
    pub to: u64,
    pub weight: f64,
    pub one_way: bool,
}

// ── Turn restriction ────────────────────────────────────────────

/// Forbidden turn: arriving from `via_from` at `via_node`, cannot proceed to `to`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TurnRestriction {
    pub via_from: u64,
    pub via_node: u64,
    pub to: u64,
}

impl TurnRestriction {
    pub fn new(via_from: u64, via_node: u64, to: u64) -> Self {
        Self { via_from, via_node, to }
    }
}

impl fmt::Display for TurnRestriction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NoTurn({}->{}->{})", self.via_from, self.via_node, self.to)
    }
}

// ── Path result ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct PathResult {
    pub nodes: Vec<u64>,
    pub total_cost: f64,
    pub nodes_visited: usize,
    pub algorithm: PathAlgorithm,
}

impl PathResult {
    pub fn hop_count(&self) -> usize {
        if self.nodes.is_empty() { 0 } else { self.nodes.len() - 1 }
    }
}

impl fmt::Display for PathResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Path(hops={} cost={:.2} algo={} visited={})",
            self.hop_count(), self.total_cost, self.algorithm, self.nodes_visited)
    }
}

// ── Priority-queue entry ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct State {
    cost: f64,
    node: u64,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

// ── Road network ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GeoGraph {
    nodes: HashMap<u64, GeoNode>,
    adj: HashMap<u64, Vec<WEdge>>,
    restrictions: HashSet<TurnRestriction>,
    ch_order: Vec<u64>,
}

impl GeoGraph {
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), adj: HashMap::new(), restrictions: HashSet::new(), ch_order: Vec::new() }
    }

    pub fn add_node(&mut self, n: GeoNode) { self.nodes.insert(n.id, n); self.adj.entry(n.id).or_default(); }

    pub fn add_edge(&mut self, from: u64, to: u64, weight: f64, one_way: bool) {
        self.adj.entry(from).or_default().push(WEdge { to, weight, one_way });
        if !one_way {
            self.adj.entry(to).or_default().push(WEdge { to: from, weight, one_way: false });
        }
    }

    pub fn add_restriction(&mut self, r: TurnRestriction) { self.restrictions.insert(r); }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    fn is_restricted(&self, prev: u64, cur: u64, next: u64) -> bool {
        self.restrictions.contains(&TurnRestriction::new(prev, cur, next))
    }

    fn neighbors(&self, id: u64) -> &[WEdge] {
        self.adj.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    fn haversine_h(&self, a: u64, b: u64) -> f64 {
        let na = match self.nodes.get(&a) { Some(n) => n, None => return 0.0 };
        let nb = match self.nodes.get(&b) { Some(n) => n, None => return 0.0 };
        haversine_m(na.lat, na.lon, nb.lat, nb.lon)
    }

    // ── Dijkstra ────────────────────────────────────────────────

    pub fn dijkstra(&self, src: u64, dst: u64) -> Result<PathResult, PathError> {
        if !self.nodes.contains_key(&src) { return Err(PathError::NodeNotFound(src)); }
        if !self.nodes.contains_key(&dst) { return Err(PathError::NodeNotFound(dst)); }
        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut heap = BinaryHeap::new();
        let mut visited = 0usize;
        dist.insert(src, 0.0);
        heap.push(State { cost: 0.0, node: src });
        while let Some(State { cost, node }) = heap.pop() {
            if node == dst {
                return Ok(PathResult {
                    nodes: reconstruct(&prev, src, dst),
                    total_cost: cost,
                    nodes_visited: visited,
                    algorithm: PathAlgorithm::Dijkstra,
                });
            }
            if cost > *dist.get(&node).unwrap_or(&f64::INFINITY) { continue; }
            visited += 1;
            for e in self.neighbors(node) {
                if let Some(&p) = prev.get(&node) {
                    if self.is_restricted(p, node, e.to) { continue; }
                }
                let nc = cost + e.weight;
                if nc < *dist.get(&e.to).unwrap_or(&f64::INFINITY) {
                    dist.insert(e.to, nc);
                    prev.insert(e.to, node);
                    heap.push(State { cost: nc, node: e.to });
                }
            }
        }
        Err(PathError::NoPathFound)
    }

    // ── A* ──────────────────────────────────────────────────────

    pub fn a_star(&self, src: u64, dst: u64) -> Result<PathResult, PathError> {
        if !self.nodes.contains_key(&src) { return Err(PathError::NodeNotFound(src)); }
        if !self.nodes.contains_key(&dst) { return Err(PathError::NodeNotFound(dst)); }
        let mut g_score: HashMap<u64, f64> = HashMap::new();
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut heap = BinaryHeap::new();
        let mut visited = 0usize;
        g_score.insert(src, 0.0);
        heap.push(State { cost: self.haversine_h(src, dst), node: src });
        while let Some(State { node, .. }) = heap.pop() {
            let g = *g_score.get(&node).unwrap_or(&f64::INFINITY);
            if node == dst {
                return Ok(PathResult {
                    nodes: reconstruct(&prev, src, dst),
                    total_cost: g,
                    nodes_visited: visited,
                    algorithm: PathAlgorithm::AStar,
                });
            }
            visited += 1;
            for e in self.neighbors(node) {
                let tentative = g + e.weight;
                if tentative < *g_score.get(&e.to).unwrap_or(&f64::INFINITY) {
                    g_score.insert(e.to, tentative);
                    prev.insert(e.to, node);
                    let f = tentative + self.haversine_h(e.to, dst);
                    heap.push(State { cost: f, node: e.to });
                }
            }
        }
        Err(PathError::NoPathFound)
    }

    // ── Bidirectional Dijkstra ──────────────────────────────────

    pub fn bidirectional_dijkstra(&self, src: u64, dst: u64) -> Result<PathResult, PathError> {
        if !self.nodes.contains_key(&src) { return Err(PathError::NodeNotFound(src)); }
        if !self.nodes.contains_key(&dst) { return Err(PathError::NodeNotFound(dst)); }
        let mut dist_f: HashMap<u64, f64> = HashMap::new();
        let mut dist_b: HashMap<u64, f64> = HashMap::new();
        let mut prev_f: HashMap<u64, u64> = HashMap::new();
        let mut prev_b: HashMap<u64, u64> = HashMap::new();
        let mut heap_f = BinaryHeap::new();
        let mut heap_b = BinaryHeap::new();
        let mut settled_f: HashSet<u64> = HashSet::new();
        let mut settled_b: HashSet<u64> = HashSet::new();
        let mut visited = 0usize;
        let mut mu = f64::INFINITY;
        let mut meeting = src;

        dist_f.insert(src, 0.0);
        dist_b.insert(dst, 0.0);
        heap_f.push(State { cost: 0.0, node: src });
        heap_b.push(State { cost: 0.0, node: dst });

        while !heap_f.is_empty() || !heap_b.is_empty() {
            // forward step
            if let Some(State { cost, node }) = heap_f.pop() {
                if cost <= mu {
                    if cost <= *dist_f.get(&node).unwrap_or(&f64::INFINITY) {
                        settled_f.insert(node);
                        visited += 1;
                        if settled_b.contains(&node) {
                            let total = cost + dist_b.get(&node).unwrap_or(&f64::INFINITY);
                            if total < mu { mu = total; meeting = node; }
                        }
                        for e in self.neighbors(node) {
                            let nc = cost + e.weight;
                            if nc < *dist_f.get(&e.to).unwrap_or(&f64::INFINITY) {
                                dist_f.insert(e.to, nc);
                                prev_f.insert(e.to, node);
                                heap_f.push(State { cost: nc, node: e.to });
                            }
                        }
                    }
                }
            }
            // backward step
            if let Some(State { cost, node }) = heap_b.pop() {
                if cost <= mu {
                    if cost <= *dist_b.get(&node).unwrap_or(&f64::INFINITY) {
                        settled_b.insert(node);
                        visited += 1;
                        if settled_f.contains(&node) {
                            let total = cost + dist_f.get(&node).unwrap_or(&f64::INFINITY);
                            if total < mu { mu = total; meeting = node; }
                        }
                        for e in self.neighbors(node) {
                            let nc = cost + e.weight;
                            if nc < *dist_b.get(&e.to).unwrap_or(&f64::INFINITY) {
                                dist_b.insert(e.to, nc);
                                prev_b.insert(e.to, node);
                                heap_b.push(State { cost: nc, node: e.to });
                            }
                        }
                    }
                }
            }
            let min_f = heap_f.peek().map(|s| s.cost).unwrap_or(f64::INFINITY);
            let min_b = heap_b.peek().map(|s| s.cost).unwrap_or(f64::INFINITY);
            if min_f + min_b >= mu { break; }
        }

        if mu == f64::INFINITY { return Err(PathError::NoPathFound); }
        let mut path_front = reconstruct(&prev_f, src, meeting);
        let mut back = reconstruct_rev(&prev_b, dst, meeting);
        back.reverse();
        if !back.is_empty() { back.remove(0); }
        path_front.extend(back);
        Ok(PathResult {
            nodes: path_front, total_cost: mu,
            nodes_visited: visited, algorithm: PathAlgorithm::BidirectionalDijkstra,
        })
    }

    // ── Contraction hierarchies (simplified) ────────────────────

    pub fn build_contraction_order(&mut self) {
        let mut ids: Vec<u64> = self.nodes.keys().copied().collect();
        ids.sort_by_key(|id| self.adj.get(id).map(|v| v.len()).unwrap_or(0));
        self.ch_order = ids;
    }

    pub fn ch_query(&self, src: u64, dst: u64) -> Result<PathResult, PathError> {
        // Simplified CH: use node order to prioritise search
        if self.ch_order.is_empty() {
            return self.dijkstra(src, dst).map(|mut r| { r.algorithm = PathAlgorithm::ContractionHierarchy; r });
        }
        let rank: HashMap<u64, usize> = self.ch_order.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut heap = BinaryHeap::new();
        let mut visited = 0usize;
        dist.insert(src, 0.0);
        heap.push(State { cost: 0.0, node: src });
        while let Some(State { cost, node }) = heap.pop() {
            if node == dst {
                return Ok(PathResult {
                    nodes: reconstruct(&prev, src, dst),
                    total_cost: cost, nodes_visited: visited,
                    algorithm: PathAlgorithm::ContractionHierarchy,
                });
            }
            if cost > *dist.get(&node).unwrap_or(&f64::INFINITY) { continue; }
            visited += 1;
            let node_rank = rank.get(&node).copied().unwrap_or(0);
            for e in self.neighbors(node) {
                let nb_rank = rank.get(&e.to).copied().unwrap_or(0);
                if nb_rank >= node_rank || e.to == dst {
                    let nc = cost + e.weight;
                    if nc < *dist.get(&e.to).unwrap_or(&f64::INFINITY) {
                        dist.insert(e.to, nc);
                        prev.insert(e.to, node);
                        heap.push(State { cost: nc, node: e.to });
                    }
                }
            }
        }
        // Fallback to Dijkstra if upward-only search finds no path
        self.dijkstra(src, dst).map(|mut r| { r.algorithm = PathAlgorithm::ContractionHierarchy; r })
    }
}

impl fmt::Display for GeoGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GeoGraph(nodes={} restrictions={})", self.node_count(), self.restrictions.len())
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn reconstruct(prev: &HashMap<u64, u64>, src: u64, dst: u64) -> Vec<u64> {
    let mut path = vec![dst];
    let mut cur = dst;
    while cur != src {
        match prev.get(&cur) {
            Some(&p) => { path.push(p); cur = p; }
            None => return path,
        }
    }
    path.reverse();
    path
}

fn reconstruct_rev(prev: &HashMap<u64, u64>, dst: u64, meeting: u64) -> Vec<u64> {
    let mut path = vec![meeting];
    let mut cur = meeting;
    while cur != dst {
        match prev.get(&cur) {
            Some(&p) => { path.push(p); cur = p; }
            None => break,
        }
    }
    path
}

fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0;
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    2.0 * r * a.sqrt().asin()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_graph() -> GeoGraph {
        // 0 -- 1 -- 2
        // |    |    |
        // 3 -- 4 -- 5
        let mut g = GeoGraph::new();
        for i in 0..6u64 {
            g.add_node(GeoNode::new(i, (i / 3) as f64, (i % 3) as f64));
        }
        g.add_edge(0, 1, 1.0, false);
        g.add_edge(1, 2, 1.0, false);
        g.add_edge(3, 4, 1.0, false);
        g.add_edge(4, 5, 1.0, false);
        g.add_edge(0, 3, 1.0, false);
        g.add_edge(1, 4, 1.0, false);
        g.add_edge(2, 5, 1.0, false);
        g
    }

    #[test]
    fn test_dijkstra_simple() {
        let g = grid_graph();
        let r = g.dijkstra(0, 5).unwrap();
        assert!((r.total_cost - 3.0).abs() < 1e-9); // 3 hops on unit-weight grid
    }

    #[test]
    fn test_dijkstra_no_path() {
        let mut g = GeoGraph::new();
        g.add_node(GeoNode::new(0, 0.0, 0.0));
        g.add_node(GeoNode::new(1, 1.0, 1.0));
        assert_eq!(g.dijkstra(0, 1), Err(PathError::NoPathFound));
    }

    #[test]
    fn test_dijkstra_node_not_found() {
        let g = GeoGraph::new();
        assert_eq!(g.dijkstra(0, 1), Err(PathError::NodeNotFound(0)));
    }

    #[test]
    fn test_a_star_simple() {
        let g = grid_graph();
        let r = g.a_star(0, 5).unwrap();
        assert!((r.total_cost - 3.0).abs() < 1e-9);
        assert_eq!(r.algorithm, PathAlgorithm::AStar);
    }

    #[test]
    fn test_a_star_same_as_dijkstra() {
        let g = grid_graph();
        let d = g.dijkstra(0, 5).unwrap();
        let a = g.a_star(0, 5).unwrap();
        assert!((d.total_cost - a.total_cost).abs() < 1e-9);
    }

    #[test]
    fn test_bidirectional() {
        let g = grid_graph();
        let r = g.bidirectional_dijkstra(0, 5).unwrap();
        assert!((r.total_cost - 3.0).abs() < 1e-9);
        assert_eq!(r.algorithm, PathAlgorithm::BidirectionalDijkstra);
    }

    #[test]
    fn test_bidirectional_no_path() {
        let mut g = GeoGraph::new();
        g.add_node(GeoNode::new(0, 0.0, 0.0));
        g.add_node(GeoNode::new(1, 1.0, 1.0));
        assert_eq!(g.bidirectional_dijkstra(0, 1), Err(PathError::NoPathFound));
    }

    #[test]
    fn test_turn_restriction() {
        let mut g = grid_graph();
        g.add_restriction(TurnRestriction::new(0, 1, 2)); // can't go 0->1->2
        let r = g.dijkstra(0, 2).unwrap();
        assert!(r.total_cost > 2.0); // must detour
    }

    #[test]
    fn test_one_way_edge() {
        let mut g = GeoGraph::new();
        g.add_node(GeoNode::new(0, 0.0, 0.0));
        g.add_node(GeoNode::new(1, 0.0, 1.0));
        g.add_edge(0, 1, 5.0, true); // one-way
        assert!(g.dijkstra(0, 1).is_ok());
        assert_eq!(g.dijkstra(1, 0), Err(PathError::NoPathFound));
    }

    #[test]
    fn test_ch_query() {
        let mut g = grid_graph();
        g.build_contraction_order();
        let r = g.ch_query(0, 5).unwrap();
        assert!((r.total_cost - 3.0).abs() < 1e-9);
        assert_eq!(r.algorithm, PathAlgorithm::ContractionHierarchy);
    }

    #[test]
    fn test_ch_without_preprocess() {
        let g = grid_graph();
        let r = g.ch_query(0, 5).unwrap();
        assert!((r.total_cost - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_result_hop_count() {
        let r = PathResult { nodes: vec![0, 1, 2], total_cost: 2.0, nodes_visited: 3, algorithm: PathAlgorithm::Dijkstra };
        assert_eq!(r.hop_count(), 2);
    }

    #[test]
    fn test_path_result_display() {
        let r = PathResult { nodes: vec![0, 1], total_cost: 1.0, nodes_visited: 2, algorithm: PathAlgorithm::AStar };
        let s = format!("{r}");
        assert!(s.contains("a-star"));
    }

    #[test]
    fn test_weighted_edges() {
        let mut g = GeoGraph::new();
        g.add_node(GeoNode::new(0, 0.0, 0.0));
        g.add_node(GeoNode::new(1, 0.0, 1.0));
        g.add_node(GeoNode::new(2, 0.0, 2.0));
        g.add_edge(0, 1, 10.0, false);
        g.add_edge(1, 2, 10.0, false);
        g.add_edge(0, 2, 100.0, false);
        let r = g.dijkstra(0, 2).unwrap();
        assert!((r.total_cost - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_haversine_distance() {
        let d = haversine_m(0.0, 0.0, 0.0, 1.0);
        assert!((d - 111_195.0).abs() < 200.0);
    }

    #[test]
    fn test_self_path() {
        let g = grid_graph();
        let r = g.dijkstra(0, 0).unwrap();
        assert!((r.total_cost - 0.0).abs() < 1e-9);
        assert_eq!(r.nodes, vec![0]);
    }

    #[test]
    fn test_display_impls() {
        let g = grid_graph();
        assert!(format!("{g}").contains("GeoGraph"));
        let tr = TurnRestriction::new(0, 1, 2);
        assert!(format!("{tr}").contains("NoTurn"));
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(format!("{}", PathAlgorithm::Dijkstra), "dijkstra");
        assert_eq!(format!("{}", PathAlgorithm::AStar), "a-star");
    }
}
