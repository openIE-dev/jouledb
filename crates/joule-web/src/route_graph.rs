//! Road Network Graph — directed/undirected edge representation, edge weights
//! (distance/time/cost), node/edge attributes, adjacency list, graph I/O
//! (edge list format), and GraphConfig builder.
//!
//! Pure-Rust road network graph for GIS routing with configurable directionality,
//! multi-weight edges, and compact adjacency-list storage.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    NodeNotFound(u64),
    EdgeNotFound(u64, u64),
    DuplicateNode(u64),
    DuplicateEdge(u64, u64),
    InvalidWeight(String),
    ParseError(String),
    EmptyGraph,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::EdgeNotFound(a, b) => write!(f, "edge {a}->{b} not found"),
            Self::DuplicateNode(id) => write!(f, "duplicate node {id}"),
            Self::DuplicateEdge(a, b) => write!(f, "duplicate edge {a}->{b}"),
            Self::InvalidWeight(s) => write!(f, "invalid weight: {s}"),
            Self::ParseError(s) => write!(f, "parse error: {s}"),
            Self::EmptyGraph => write!(f, "empty graph"),
        }
    }
}

impl std::error::Error for GraphError {}

// ── Edge weight ─────────────────────────────────────────────────

/// Multi-dimensional edge weight for road segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeWeight {
    /// Distance in metres.
    pub distance_m: f64,
    /// Travel time in seconds.
    pub time_s: f64,
    /// Monetary cost (arbitrary unit).
    pub cost: f64,
}

impl EdgeWeight {
    pub fn new(distance_m: f64, time_s: f64, cost: f64) -> Self {
        Self { distance_m, time_s, cost }
    }

    pub fn distance_only(d: f64) -> Self {
        Self { distance_m: d, time_s: 0.0, cost: 0.0 }
    }

    pub fn combined(&self, w_dist: f64, w_time: f64, w_cost: f64) -> f64 {
        self.distance_m * w_dist + self.time_s * w_time + self.cost * w_cost
    }
}

impl Default for EdgeWeight {
    fn default() -> Self { Self { distance_m: 1.0, time_s: 1.0, cost: 0.0 } }
}

impl fmt::Display for EdgeWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d={:.1}m t={:.1}s c={:.2}", self.distance_m, self.time_s, self.cost)
    }
}

// ── Weight mode ─────────────────────────────────────────────────

/// Which weight component to use as the primary metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightMode {
    Distance,
    Time,
    Cost,
    Combined { w_dist: f64, w_time: f64, w_cost: f64 },
}

impl WeightMode {
    pub fn evaluate(&self, w: &EdgeWeight) -> f64 {
        match self {
            Self::Distance => w.distance_m,
            Self::Time => w.time_s,
            Self::Cost => w.cost,
            Self::Combined { w_dist, w_time, w_cost } => w.combined(*w_dist, *w_time, *w_cost),
        }
    }
}

impl fmt::Display for WeightMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Distance => write!(f, "distance"),
            Self::Time => write!(f, "time"),
            Self::Cost => write!(f, "cost"),
            Self::Combined { .. } => write!(f, "combined"),
        }
    }
}

// ── Node ────────────────────────────────────────────────────────

/// A node in the road network.
#[derive(Debug, Clone, PartialEq)]
pub struct RoadNode {
    pub id: u64,
    pub lat: f64,
    pub lon: f64,
    pub attributes: HashMap<String, String>,
}

impl RoadNode {
    pub fn new(id: u64, lat: f64, lon: f64) -> Self {
        Self { id, lat, lon, attributes: HashMap::new() }
    }

    pub fn with_attr(mut self, key: &str, val: &str) -> Self {
        self.attributes.insert(key.to_string(), val.to_string());
        self
    }

    /// Haversine distance in metres to another node.
    pub fn haversine_to(&self, other: &RoadNode) -> f64 {
        haversine_m(self.lat, self.lon, other.lat, other.lon)
    }
}

impl fmt::Display for RoadNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N{}({:.5},{:.5})", self.id, self.lat, self.lon)
    }
}

// ── Edge ────────────────────────────────────────────────────────

/// A directed edge between two road nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct RoadEdge {
    pub from: u64,
    pub to: u64,
    pub weight: EdgeWeight,
    pub one_way: bool,
    pub road_class: RoadClass,
    pub attributes: HashMap<String, String>,
}

impl RoadEdge {
    pub fn new(from: u64, to: u64, weight: EdgeWeight) -> Self {
        Self {
            from, to, weight, one_way: false,
            road_class: RoadClass::Local,
            attributes: HashMap::new(),
        }
    }

    pub fn with_one_way(mut self, v: bool) -> Self { self.one_way = v; self }
    pub fn with_road_class(mut self, c: RoadClass) -> Self { self.road_class = c; self }
    pub fn with_attr(mut self, k: &str, v: &str) -> Self {
        self.attributes.insert(k.to_string(), v.to_string());
        self
    }
}

impl fmt::Display for RoadEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arrow = if self.one_way { "->" } else { "<->" };
        write!(f, "E({}{}{}  {})", self.from, arrow, self.to, self.weight)
    }
}

// ── Road class ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoadClass {
    Motorway,
    Trunk,
    Primary,
    Secondary,
    Tertiary,
    Residential,
    Local,
}

impl RoadClass {
    pub fn default_speed_kmh(&self) -> f64 {
        match self {
            Self::Motorway => 120.0,
            Self::Trunk => 100.0,
            Self::Primary => 80.0,
            Self::Secondary => 60.0,
            Self::Tertiary => 40.0,
            Self::Residential => 30.0,
            Self::Local => 20.0,
        }
    }
}

impl fmt::Display for RoadClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Motorway => "motorway",
            Self::Trunk => "trunk",
            Self::Primary => "primary",
            Self::Secondary => "secondary",
            Self::Tertiary => "tertiary",
            Self::Residential => "residential",
            Self::Local => "local",
        };
        write!(f, "{s}")
    }
}

// ── Graph config ────────────────────────────────────────────────

/// Builder for configuring a `RoadGraph`.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    pub directed: bool,
    pub weight_mode: WeightMode,
    pub allow_u_turns: bool,
    pub default_speed_kmh: f64,
}

impl GraphConfig {
    pub fn new() -> Self {
        Self { directed: true, weight_mode: WeightMode::Distance, allow_u_turns: false, default_speed_kmh: 50.0 }
    }
    pub fn with_directed(mut self, v: bool) -> Self { self.directed = v; self }
    pub fn with_weight_mode(mut self, m: WeightMode) -> Self { self.weight_mode = m; self }
    pub fn with_allow_u_turns(mut self, v: bool) -> Self { self.allow_u_turns = v; self }
    pub fn with_default_speed(mut self, s: f64) -> Self { self.default_speed_kmh = s; self }
}

impl Default for GraphConfig {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for GraphConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GraphConfig(dir={} mode={} uturn={} speed={:.0})",
            self.directed, self.weight_mode, self.allow_u_turns, self.default_speed_kmh)
    }
}

// ── Adjacency entry ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct AdjEntry {
    target: u64,
    weight: EdgeWeight,
    edge_idx: usize,
}

// ── Road graph ──────────────────────────────────────────────────

/// Road network graph backed by an adjacency list.
#[derive(Debug, Clone)]
pub struct RoadGraph {
    pub config: GraphConfig,
    nodes: HashMap<u64, RoadNode>,
    edges: Vec<RoadEdge>,
    adj: HashMap<u64, Vec<AdjEntry>>,
}

impl RoadGraph {
    pub fn new(config: GraphConfig) -> Self {
        Self { config, nodes: HashMap::new(), edges: Vec::new(), adj: HashMap::new() }
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }

    pub fn add_node(&mut self, node: RoadNode) -> Result<(), GraphError> {
        let id = node.id;
        if self.nodes.contains_key(&id) {
            return Err(GraphError::DuplicateNode(id));
        }
        self.nodes.insert(id, node);
        self.adj.entry(id).or_default();
        Ok(())
    }

    pub fn get_node(&self, id: u64) -> Option<&RoadNode> { self.nodes.get(&id) }

    pub fn add_edge(&mut self, edge: RoadEdge) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&edge.from) {
            return Err(GraphError::NodeNotFound(edge.from));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(GraphError::NodeNotFound(edge.to));
        }
        let idx = self.edges.len();
        self.adj.entry(edge.from).or_default().push(AdjEntry {
            target: edge.to, weight: edge.weight, edge_idx: idx,
        });
        if !edge.one_way && !self.config.directed {
            self.adj.entry(edge.to).or_default().push(AdjEntry {
                target: edge.from, weight: edge.weight, edge_idx: idx,
            });
        }
        self.edges.push(edge);
        Ok(())
    }

    pub fn neighbors(&self, node_id: u64) -> Vec<(u64, &EdgeWeight)> {
        self.adj.get(&node_id)
            .map(|entries| entries.iter().map(|e| (e.target, &e.weight)).collect())
            .unwrap_or_default()
    }

    pub fn get_edge(&self, from: u64, to: u64) -> Option<&RoadEdge> {
        self.adj.get(&from)?
            .iter()
            .find(|e| e.target == to)
            .map(|e| &self.edges[e.edge_idx])
    }

    pub fn node_ids(&self) -> Vec<u64> {
        let mut ids: Vec<_> = self.nodes.keys().copied().collect();
        ids.sort();
        ids
    }

    pub fn degree(&self, node_id: u64) -> usize {
        self.adj.get(&node_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Serialize to edge-list text: `from to dist time cost one_way`
    pub fn to_edge_list(&self) -> String {
        let mut out = String::new();
        out.push_str("# from to distance_m time_s cost one_way\n");
        for e in &self.edges {
            out.push_str(&format!(
                "{} {} {:.2} {:.2} {:.2} {}\n",
                e.from, e.to, e.weight.distance_m, e.weight.time_s, e.weight.cost,
                if e.one_way { 1 } else { 0 }
            ));
        }
        out
    }

    /// Parse edge-list text into a graph, auto-creating nodes.
    pub fn from_edge_list(text: &str, config: GraphConfig) -> Result<Self, GraphError> {
        let mut g = Self::new(config);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(GraphError::ParseError(format!("need >=4 fields: {line}")));
            }
            let from: u64 = parts[0].parse().map_err(|_| GraphError::ParseError("bad from".into()))?;
            let to: u64 = parts[1].parse().map_err(|_| GraphError::ParseError("bad to".into()))?;
            let dist: f64 = parts[2].parse().map_err(|_| GraphError::ParseError("bad dist".into()))?;
            let time: f64 = parts[3].parse().map_err(|_| GraphError::ParseError("bad time".into()))?;
            let cost: f64 = if parts.len() > 4 {
                parts[4].parse().map_err(|_| GraphError::ParseError("bad cost".into()))?
            } else { 0.0 };
            let one_way = parts.len() > 5 && parts[5] == "1";

            if !g.nodes.contains_key(&from) {
                g.add_node(RoadNode::new(from, 0.0, 0.0)).ok();
            }
            if !g.nodes.contains_key(&to) {
                g.add_node(RoadNode::new(to, 0.0, 0.0)).ok();
            }
            let edge = RoadEdge::new(from, to, EdgeWeight::new(dist, time, cost))
                .with_one_way(one_way);
            g.add_edge(edge)?;
        }
        Ok(g)
    }

    /// Remove a node and all incident edges.
    pub fn remove_node(&mut self, id: u64) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&id) {
            return Err(GraphError::NodeNotFound(id));
        }
        self.nodes.remove(&id);
        self.adj.remove(&id);
        for entries in self.adj.values_mut() {
            entries.retain(|e| e.target != id);
        }
        self.edges.retain(|e| e.from != id && e.to != id);
        Ok(())
    }

    /// Compute total distance of all edges.
    pub fn total_distance(&self) -> f64 {
        self.edges.iter().map(|e| e.weight.distance_m).sum()
    }
}

impl fmt::Display for RoadGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RoadGraph(nodes={} edges={} {})", self.node_count(), self.edge_count(), self.config)
    }
}

// ── Haversine helper ────────────────────────────────────────────

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
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

    fn simple_graph() -> RoadGraph {
        let cfg = GraphConfig::new().with_directed(false);
        let mut g = RoadGraph::new(cfg);
        for i in 0..4 {
            g.add_node(RoadNode::new(i, 0.0, i as f64)).unwrap();
        }
        g.add_edge(RoadEdge::new(0, 1, EdgeWeight::distance_only(10.0))).unwrap();
        g.add_edge(RoadEdge::new(1, 2, EdgeWeight::distance_only(20.0))).unwrap();
        g.add_edge(RoadEdge::new(2, 3, EdgeWeight::distance_only(30.0))).unwrap();
        g.add_edge(RoadEdge::new(0, 3, EdgeWeight::distance_only(100.0))).unwrap();
        g
    }

    #[test]
    fn test_add_node() {
        let mut g = RoadGraph::new(GraphConfig::new());
        g.add_node(RoadNode::new(1, 40.0, -74.0)).unwrap();
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_duplicate_node() {
        let mut g = RoadGraph::new(GraphConfig::new());
        g.add_node(RoadNode::new(1, 0.0, 0.0)).unwrap();
        assert_eq!(g.add_node(RoadNode::new(1, 0.0, 0.0)), Err(GraphError::DuplicateNode(1)));
    }

    #[test]
    fn test_add_edge() {
        let g = simple_graph();
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_edge_missing_node() {
        let mut g = RoadGraph::new(GraphConfig::new());
        g.add_node(RoadNode::new(0, 0.0, 0.0)).unwrap();
        let r = g.add_edge(RoadEdge::new(0, 99, EdgeWeight::default()));
        assert_eq!(r, Err(GraphError::NodeNotFound(99)));
    }

    #[test]
    fn test_neighbors_undirected() {
        let g = simple_graph();
        let nb = g.neighbors(1);
        assert_eq!(nb.len(), 2); // 0 and 2
    }

    #[test]
    fn test_neighbors_directed() {
        let cfg = GraphConfig::new().with_directed(true);
        let mut g = RoadGraph::new(cfg);
        g.add_node(RoadNode::new(0, 0.0, 0.0)).unwrap();
        g.add_node(RoadNode::new(1, 0.0, 0.0)).unwrap();
        g.add_edge(RoadEdge::new(0, 1, EdgeWeight::default())).unwrap();
        assert_eq!(g.neighbors(0).len(), 1);
        assert_eq!(g.neighbors(1).len(), 0);
    }

    #[test]
    fn test_get_edge() {
        let g = simple_graph();
        let e = g.get_edge(0, 1).unwrap();
        assert!((e.weight.distance_m - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_node_ids_sorted() {
        let g = simple_graph();
        assert_eq!(g.node_ids(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_degree() {
        let g = simple_graph();
        assert_eq!(g.degree(0), 2);
        assert_eq!(g.degree(1), 2);
    }

    #[test]
    fn test_remove_node() {
        let mut g = simple_graph();
        g.remove_node(1).unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.neighbors(0).len(), 1); // only edge to 3
    }

    #[test]
    fn test_remove_node_missing() {
        let mut g = simple_graph();
        assert_eq!(g.remove_node(99), Err(GraphError::NodeNotFound(99)));
    }

    #[test]
    fn test_total_distance() {
        let g = simple_graph();
        assert!((g.total_distance() - 160.0).abs() < 1e-9);
    }

    #[test]
    fn test_edge_list_roundtrip() {
        let g = simple_graph();
        let text = g.to_edge_list();
        let g2 = RoadGraph::from_edge_list(&text, GraphConfig::new()).unwrap();
        assert_eq!(g2.edge_count(), g.edge_count());
    }

    #[test]
    fn test_edge_list_parse_error() {
        let r = RoadGraph::from_edge_list("bad data", GraphConfig::new());
        assert!(r.is_err());
    }

    #[test]
    fn test_haversine() {
        let d = haversine_m(0.0, 0.0, 0.0, 1.0);
        assert!((d - 111_195.0).abs() < 200.0);
    }

    #[test]
    fn test_edge_weight_combined() {
        let w = EdgeWeight::new(100.0, 10.0, 5.0);
        let c = w.combined(1.0, 2.0, 3.0);
        assert!((c - (100.0 + 20.0 + 15.0)).abs() < 1e-9);
    }

    #[test]
    fn test_weight_mode_evaluate() {
        let w = EdgeWeight::new(100.0, 50.0, 25.0);
        assert!((WeightMode::Distance.evaluate(&w) - 100.0).abs() < 1e-9);
        assert!((WeightMode::Time.evaluate(&w) - 50.0).abs() < 1e-9);
        assert!((WeightMode::Cost.evaluate(&w) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_road_class_speed() {
        assert!((RoadClass::Motorway.default_speed_kmh() - 120.0).abs() < 1e-9);
        assert!((RoadClass::Local.default_speed_kmh() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_graph_config_builder() {
        let cfg = GraphConfig::new()
            .with_directed(false)
            .with_weight_mode(WeightMode::Time)
            .with_allow_u_turns(true)
            .with_default_speed(80.0);
        assert!(!cfg.directed);
        assert_eq!(cfg.weight_mode, WeightMode::Time);
        assert!(cfg.allow_u_turns);
        assert!((cfg.default_speed_kmh - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_impls() {
        let n = RoadNode::new(1, 40.12345, -74.56789);
        assert!(format!("{n}").contains("N1"));
        let e = RoadEdge::new(0, 1, EdgeWeight::default());
        assert!(format!("{e}").contains("E("));
        let g = simple_graph();
        assert!(format!("{g}").contains("RoadGraph"));
    }

    #[test]
    fn test_one_way_edge() {
        let cfg = GraphConfig::new().with_directed(false);
        let mut g = RoadGraph::new(cfg);
        g.add_node(RoadNode::new(0, 0.0, 0.0)).unwrap();
        g.add_node(RoadNode::new(1, 0.0, 0.0)).unwrap();
        g.add_edge(RoadEdge::new(0, 1, EdgeWeight::default()).with_one_way(true)).unwrap();
        assert_eq!(g.neighbors(0).len(), 1);
        assert_eq!(g.neighbors(1).len(), 0);
    }
}
