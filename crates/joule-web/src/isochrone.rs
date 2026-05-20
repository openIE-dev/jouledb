//! Isochrone Computation — time-based reachability polygon, distance-based
//! reachability, incremental expansion, concave hull of reachable nodes,
//! multi-modal (walk+drive), and IsochroneConfig builder.
//!
//! Pure-Rust isochrone engine for computing reachability areas from a given
//! origin on a weighted graph, with configurable travel budgets and modes.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum IsochroneError {
    NodeNotFound(u64),
    InvalidBudget(String),
    NoReachableNodes,
    InsufficientNodes(String),
}

impl fmt::Display for IsochroneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::InvalidBudget(s) => write!(f, "invalid budget: {s}"),
            Self::NoReachableNodes => write!(f, "no reachable nodes"),
            Self::InsufficientNodes(s) => write!(f, "insufficient nodes: {s}"),
        }
    }
}

impl std::error::Error for IsochroneError {}

// ── Travel mode ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TravelMode {
    Drive,
    Walk,
    DriveAndWalk { drive_budget_s: f64, walk_budget_s: f64 },
}

impl TravelMode {
    pub fn total_budget_s(&self, cfg: &IsochroneConfig) -> f64 {
        match self {
            Self::Drive | Self::Walk => cfg.budget,
            Self::DriveAndWalk { drive_budget_s, walk_budget_s } => drive_budget_s + walk_budget_s,
        }
    }
}

impl fmt::Display for TravelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Drive => write!(f, "drive"),
            Self::Walk => write!(f, "walk"),
            Self::DriveAndWalk { .. } => write!(f, "drive+walk"),
        }
    }
}

// ── Budget type ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetType {
    Time,
    Distance,
}

impl fmt::Display for BudgetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Time => write!(f, "time"),
            Self::Distance => write!(f, "distance"),
        }
    }
}

// ── Config ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IsochroneConfig {
    pub budget: f64,
    pub budget_type: BudgetType,
    pub mode: TravelMode,
    pub walk_speed_ms: f64,
    pub drive_speed_ms: f64,
    pub incremental_steps: usize,
}

impl IsochroneConfig {
    pub fn new(budget: f64) -> Self {
        Self {
            budget, budget_type: BudgetType::Time, mode: TravelMode::Drive,
            walk_speed_ms: 1.4, drive_speed_ms: 13.9, incremental_steps: 1,
        }
    }
    pub fn with_budget_type(mut self, t: BudgetType) -> Self { self.budget_type = t; self }
    pub fn with_mode(mut self, m: TravelMode) -> Self { self.mode = m; self }
    pub fn with_walk_speed(mut self, s: f64) -> Self { self.walk_speed_ms = s; self }
    pub fn with_drive_speed(mut self, s: f64) -> Self { self.drive_speed_ms = s; self }
    pub fn with_incremental_steps(mut self, n: usize) -> Self { self.incremental_steps = n.max(1); self }
}

impl Default for IsochroneConfig {
    fn default() -> Self { Self::new(600.0) }
}

impl fmt::Display for IsochroneConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsoCfg(budget={:.0} type={} mode={})", self.budget, self.budget_type, self.mode)
    }
}

// ── Isochrone node ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsoNode {
    pub id: u64,
    pub lat: f64,
    pub lon: f64,
}

impl IsoNode {
    pub fn new(id: u64, lat: f64, lon: f64) -> Self { Self { id, lat, lon } }
}

impl fmt::Display for IsoNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IN{}({:.5},{:.5})", self.id, self.lat, self.lon)
    }
}

// ── Edge ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct IsoEdge {
    pub to: u64,
    pub time_s: f64,
    pub distance_m: f64,
    pub walkable: bool,
}

// ── Priority-queue entry ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct QEntry { cost: f64, node: u64 }
impl Eq for QEntry {}
impl Ord for QEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for QEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

// ── Isochrone result ────────────────────────────────────────────

#[derive(Debug, Clone)]
#[derive(PartialEq)]
pub struct IsochroneResult {
    pub reachable: Vec<(u64, f64)>,
    pub hull: Vec<(f64, f64)>,
    pub budget_used: f64,
}

impl IsochroneResult {
    pub fn reachable_count(&self) -> usize { self.reachable.len() }
    pub fn hull_vertices(&self) -> usize { self.hull.len() }
}

impl fmt::Display for IsochroneResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Isochrone(reachable={} hull_pts={})", self.reachable_count(), self.hull_vertices())
    }
}

// ── Isochrone graph ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct IsochroneGraph {
    nodes: HashMap<u64, IsoNode>,
    adj: HashMap<u64, Vec<IsoEdge>>,
}

impl IsochroneGraph {
    pub fn new() -> Self { Self { nodes: HashMap::new(), adj: HashMap::new() } }

    pub fn add_node(&mut self, n: IsoNode) {
        self.nodes.insert(n.id, n);
        self.adj.entry(n.id).or_default();
    }

    pub fn add_edge(&mut self, from: u64, to: u64, time_s: f64, distance_m: f64, walkable: bool) {
        self.adj.entry(from).or_default().push(IsoEdge { to, time_s, distance_m, walkable });
        self.adj.entry(to).or_default().push(IsoEdge { to: from, time_s, distance_m, walkable });
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    fn edge_cost(e: &IsoEdge, budget_type: BudgetType) -> f64 {
        match budget_type {
            BudgetType::Time => e.time_s,
            BudgetType::Distance => e.distance_m,
        }
    }

    /// Compute reachable nodes within budget.
    pub fn compute(&self, origin: u64, config: &IsochroneConfig) -> Result<IsochroneResult, IsochroneError> {
        if !self.nodes.contains_key(&origin) {
            return Err(IsochroneError::NodeNotFound(origin));
        }
        if config.budget <= 0.0 {
            return Err(IsochroneError::InvalidBudget("budget must be positive".into()));
        }

        let reachable = self.expand(origin, config.budget, config.budget_type, &config.mode)?;
        if reachable.is_empty() {
            return Err(IsochroneError::NoReachableNodes);
        }

        let hull = self.concave_hull(&reachable);

        Ok(IsochroneResult { reachable, hull, budget_used: config.budget })
    }

    /// Compute incremental isochrones at equal budget fractions.
    pub fn compute_incremental(&self, origin: u64, config: &IsochroneConfig) -> Result<Vec<IsochroneResult>, IsochroneError> {
        let steps = config.incremental_steps.max(1);
        let step_size = config.budget / steps as f64;
        let mut results = Vec::with_capacity(steps);
        for i in 1..=steps {
            let budget = step_size * i as f64;
            let reachable = self.expand(origin, budget, config.budget_type, &config.mode)?;
            let hull = self.concave_hull(&reachable);
            results.push(IsochroneResult { reachable, hull, budget_used: budget });
        }
        Ok(results)
    }

    fn expand(&self, origin: u64, budget: f64, bt: BudgetType, mode: &TravelMode) -> Result<Vec<(u64, f64)>, IsochroneError> {
        let mut dist: HashMap<u64, f64> = HashMap::new();
        let mut heap = BinaryHeap::new();
        dist.insert(origin, 0.0);
        heap.push(QEntry { cost: 0.0, node: origin });

        let walk_only = matches!(mode, TravelMode::Walk);

        while let Some(QEntry { cost, node }) = heap.pop() {
            if cost > *dist.get(&node).unwrap_or(&f64::INFINITY) { continue; }
            if let Some(edges) = self.adj.get(&node) {
                for e in edges {
                    if walk_only && !e.walkable { continue; }
                    let ec = Self::edge_cost(e, bt);
                    let nc = cost + ec;
                    if nc <= budget && nc < *dist.get(&e.to).unwrap_or(&f64::INFINITY) {
                        dist.insert(e.to, nc);
                        heap.push(QEntry { cost: nc, node: e.to });
                    }
                }
            }
        }

        // Multi-modal: extend walk from each driving-reachable node
        if let TravelMode::DriveAndWalk { drive_budget_s, walk_budget_s } = mode {
            let drive_set: Vec<(u64, f64)> = dist.iter().map(|(&k, &v)| (k, v)).collect();
            for (node, dcost) in &drive_set {
                if *dcost > *drive_budget_s { continue; }
                let mut walk_heap = BinaryHeap::new();
                walk_heap.push(QEntry { cost: 0.0, node: *node });
                let mut wdist: HashMap<u64, f64> = HashMap::new();
                wdist.insert(*node, 0.0);
                while let Some(QEntry { cost, node: wn }) = walk_heap.pop() {
                    if cost > *wdist.get(&wn).unwrap_or(&f64::INFINITY) { continue; }
                    if let Some(edges) = self.adj.get(&wn) {
                        for e in edges {
                            if !e.walkable { continue; }
                            let ec = Self::edge_cost(e, bt);
                            let nc = cost + ec;
                            if nc <= *walk_budget_s && nc < *wdist.get(&e.to).unwrap_or(&f64::INFINITY) {
                                let total = dcost + nc;
                                if total < *dist.get(&e.to).unwrap_or(&f64::INFINITY) {
                                    dist.insert(e.to, total);
                                }
                                wdist.insert(e.to, nc);
                                walk_heap.push(QEntry { cost: nc, node: e.to });
                            }
                        }
                    }
                }
            }
        }

        let mut result: Vec<(u64, f64)> = dist.into_iter().collect();
        result.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(result)
    }

    /// Concave hull approximation: sorted by angle from centroid.
    fn concave_hull(&self, reachable: &[(u64, f64)]) -> Vec<(f64, f64)> {
        let pts: Vec<(f64, f64)> = reachable.iter()
            .filter_map(|(id, _)| self.nodes.get(id).map(|n| (n.lat, n.lon)))
            .collect();
        if pts.len() < 3 { return pts; }
        let cx = pts.iter().map(|p| p.0).sum::<f64>() / pts.len() as f64;
        let cy = pts.iter().map(|p| p.1).sum::<f64>() / pts.len() as f64;
        let mut angles: Vec<(f64, f64, f64)> = pts.iter()
            .map(|&(x, y)| ((y - cy).atan2(x - cx), x, y))
            .collect();
        angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Keep only outermost points per angular bucket
        let n_buckets = (pts.len().min(36)).max(4);
        let bucket_size = std::f64::consts::TAU / n_buckets as f64;
        let mut hull: Vec<(f64, f64)> = Vec::new();
        let mut seen_buckets: HashSet<usize> = HashSet::new();
        for (angle, x, y) in &angles {
            let bucket = ((angle + std::f64::consts::PI) / bucket_size) as usize % n_buckets;
            if !seen_buckets.contains(&bucket) {
                hull.push((*x, *y));
                seen_buckets.insert(bucket);
            } else {
                // Replace if further from centroid
                let dist_new = (x - cx).powi(2) + (y - cy).powi(2);
                if let Some(pos) = hull.iter().position(|&(hx, hy)| {
                    let a2 = ((hy - cy).atan2(hx - cx) + std::f64::consts::PI) / bucket_size;
                    (a2 as usize % n_buckets) == bucket
                }) {
                    let (hx, hy) = hull[pos];
                    let dist_old = (hx - cx).powi(2) + (hy - cy).powi(2);
                    if dist_new > dist_old {
                        hull[pos] = (*x, *y);
                    }
                }
            }
        }
        hull
    }
}

impl fmt::Display for IsochroneGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IsochroneGraph(nodes={})", self.node_count())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> IsochroneGraph {
        let mut g = IsochroneGraph::new();
        for i in 0..9u64 {
            let lat = (i / 3) as f64;
            let lon = (i % 3) as f64;
            g.add_node(IsoNode::new(i, lat, lon));
        }
        // 3x3 grid
        for r in 0..3u64 {
            for c in 0..3u64 {
                let id = r * 3 + c;
                if c < 2 { g.add_edge(id, id + 1, 10.0, 100.0, true); }
                if r < 2 { g.add_edge(id, id + 3, 10.0, 100.0, true); }
            }
        }
        g
    }

    #[test]
    fn test_basic_isochrone() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(25.0);
        let r = g.compute(0, &cfg).unwrap();
        assert!(r.reachable_count() >= 3);
    }

    #[test]
    fn test_full_budget() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(1000.0);
        let r = g.compute(0, &cfg).unwrap();
        assert_eq!(r.reachable_count(), 9);
    }

    #[test]
    fn test_zero_budget() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(-1.0);
        assert!(g.compute(0, &cfg).is_err());
    }

    #[test]
    fn test_node_not_found() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(100.0);
        assert_eq!(g.compute(99, &cfg), Err(IsochroneError::NodeNotFound(99)));
    }

    #[test]
    fn test_distance_budget() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(150.0).with_budget_type(BudgetType::Distance);
        let r = g.compute(4, &cfg).unwrap();
        assert!(r.reachable_count() >= 3);
    }

    #[test]
    fn test_incremental() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(40.0).with_incremental_steps(4);
        let results = g.compute_incremental(0, &cfg).unwrap();
        assert_eq!(results.len(), 4);
        for i in 1..results.len() {
            assert!(results[i].reachable_count() >= results[i - 1].reachable_count());
        }
    }

    #[test]
    fn test_walk_mode() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(100.0).with_mode(TravelMode::Walk);
        let r = g.compute(0, &cfg).unwrap();
        assert!(r.reachable_count() > 0);
    }

    #[test]
    fn test_multimodal() {
        let g = make_graph();
        let mode = TravelMode::DriveAndWalk { drive_budget_s: 20.0, walk_budget_s: 15.0 };
        let cfg = IsochroneConfig::new(35.0).with_mode(mode);
        let r = g.compute(0, &cfg).unwrap();
        assert!(r.reachable_count() >= 3);
    }

    #[test]
    fn test_hull_generation() {
        let g = make_graph();
        let cfg = IsochroneConfig::new(100.0);
        let r = g.compute(0, &cfg).unwrap();
        assert!(r.hull_vertices() > 0);
    }

    #[test]
    fn test_config_builder() {
        let cfg = IsochroneConfig::new(300.0)
            .with_budget_type(BudgetType::Distance)
            .with_mode(TravelMode::Walk)
            .with_walk_speed(1.2)
            .with_drive_speed(15.0)
            .with_incremental_steps(5);
        assert_eq!(cfg.budget_type, BudgetType::Distance);
        assert_eq!(cfg.incremental_steps, 5);
    }

    #[test]
    fn test_config_display() {
        let cfg = IsochroneConfig::new(600.0);
        assert!(format!("{cfg}").contains("IsoCfg"));
    }

    #[test]
    fn test_result_display() {
        let r = IsochroneResult { reachable: vec![(0, 0.0), (1, 10.0)], hull: vec![], budget_used: 100.0 };
        assert!(format!("{r}").contains("Isochrone"));
    }

    #[test]
    fn test_travel_mode_display() {
        assert_eq!(format!("{}", TravelMode::Drive), "drive");
        assert_eq!(format!("{}", TravelMode::Walk), "walk");
    }

    #[test]
    fn test_budget_type_display() {
        assert_eq!(format!("{}", BudgetType::Time), "time");
        assert_eq!(format!("{}", BudgetType::Distance), "distance");
    }

    #[test]
    fn test_single_node_reachable() {
        let mut g = IsochroneGraph::new();
        g.add_node(IsoNode::new(0, 0.0, 0.0));
        let cfg = IsochroneConfig::new(100.0);
        let r = g.compute(0, &cfg).unwrap();
        assert_eq!(r.reachable_count(), 1);
    }

    #[test]
    fn test_non_walkable_edges() {
        let mut g = IsochroneGraph::new();
        g.add_node(IsoNode::new(0, 0.0, 0.0));
        g.add_node(IsoNode::new(1, 1.0, 0.0));
        // edge is not walkable
        g.adj.entry(0).or_default().push(IsoEdge { to: 1, time_s: 5.0, distance_m: 50.0, walkable: false });
        g.adj.entry(1).or_default().push(IsoEdge { to: 0, time_s: 5.0, distance_m: 50.0, walkable: false });
        let cfg = IsochroneConfig::new(100.0).with_mode(TravelMode::Walk);
        let r = g.compute(0, &cfg).unwrap();
        assert_eq!(r.reachable_count(), 1); // only origin
    }

    #[test]
    fn test_total_budget_multimodal() {
        let mode = TravelMode::DriveAndWalk { drive_budget_s: 300.0, walk_budget_s: 100.0 };
        let cfg = IsochroneConfig::new(400.0);
        assert!((mode.total_budget_s(&cfg) - 400.0).abs() < 1e-9);
    }

    #[test]
    fn test_iso_node_display() {
        let n = IsoNode::new(42, 27.3, -82.5);
        assert!(format!("{n}").contains("IN42"));
    }

    #[test]
    fn test_graph_display() {
        let g = make_graph();
        assert!(format!("{g}").contains("IsochroneGraph"));
    }
}
