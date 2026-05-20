//! Probabilistic Roadmap — node sampling, k-nearest connection, graph search,
//! lazy evaluation for sampling-based multi-query motion planning.
//!
//! Pure-Rust PRM planner for 2D/3D configuration spaces with AABB obstacle
//! checking, Dijkstra-based graph search, and lazy collision evaluation.

use std::collections::BinaryHeap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PrmError {
    InvalidBounds(String),
    InvalidParameter(String),
    NoPathFound,
    EmptyRoadmap,
    StartInCollision,
    GoalInCollision,
}

impl fmt::Display for PrmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds(s) => write!(f, "invalid bounds: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoPathFound => write!(f, "no path found in roadmap"),
            Self::EmptyRoadmap => write!(f, "roadmap is empty"),
            Self::StartInCollision => write!(f, "start configuration in collision"),
            Self::GoalInCollision => write!(f, "goal configuration in collision"),
        }
    }
}

impl std::error::Error for PrmError {}

// ── Point ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub coords: Vec<f64>,
}

impl Point {
    pub fn new(coords: Vec<f64>) -> Self { Self { coords } }
    pub fn new2(x: f64, y: f64) -> Self { Self { coords: vec![x, y] } }
    pub fn new3(x: f64, y: f64, z: f64) -> Self { Self { coords: vec![x, y, z] } }

    pub fn distance_to(&self, other: &Point) -> f64 {
        self.coords.iter().zip(other.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.coords.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{c:.3}")?;
        }
        write!(f, ")")
    }
}

// ── AABB Obstacle ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AabbObstacle {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

impl AabbObstacle {
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Self { Self { min, max } }

    pub fn contains(&self, p: &Point) -> bool {
        p.coords.iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .all(|(c, (lo, hi))| *c >= *lo && *c <= *hi)
    }

    pub fn intersects_segment(&self, a: &Point, b: &Point, samples: usize) -> bool {
        for i in 0..=samples {
            let t = i as f64 / samples as f64;
            let coords: Vec<f64> = a.coords.iter().zip(b.coords.iter())
                .map(|(ca, cb)| ca + (cb - ca) * t)
                .collect();
            if self.contains(&Point { coords }) { return true; }
        }
        false
    }
}

// ── Simple LCG RNG ──────────────────────────────────────────────

struct SimpleRng { state: u64 }

impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_range(&mut self, lo: f64, hi: f64) -> f64 { lo + self.next_f64() * (hi - lo) }
}

// ── Roadmap Node & Edge ─────────────────────────────────────────

#[derive(Debug, Clone)]
struct RoadmapNode {
    point: Point,
    edges: Vec<usize>, // indices into edges list
}

#[derive(Debug, Clone)]
struct RoadmapEdge {
    from: usize,
    to: usize,
    cost: f64,
    validated: bool,
    collision_free: bool,
}

// ── PRM Config ──────────────────────────────────────────────────

/// Configuration for the PRM planner.
#[derive(Debug, Clone)]
pub struct PrmConfig {
    pub bounds_min: Vec<f64>,
    pub bounds_max: Vec<f64>,
    pub num_samples: usize,
    pub k_nearest: usize,
    pub max_edge_length: f64,
    pub collision_samples: usize,
    pub seed: u64,
    pub lazy: bool,
}

impl PrmConfig {
    pub fn new(bounds_min: Vec<f64>, bounds_max: Vec<f64>) -> Result<Self, PrmError> {
        if bounds_min.len() != bounds_max.len() || bounds_min.is_empty() {
            return Err(PrmError::InvalidBounds("dimension mismatch or zero".into()));
        }
        for (lo, hi) in bounds_min.iter().zip(bounds_max.iter()) {
            if lo >= hi {
                return Err(PrmError::InvalidBounds(format!("min {lo} >= max {hi}")));
            }
        }
        Ok(Self {
            bounds_min,
            bounds_max,
            num_samples: 500,
            k_nearest: 10,
            max_edge_length: 5.0,
            collision_samples: 10,
            seed: 42,
            lazy: false,
        })
    }

    pub fn with_num_samples(mut self, n: usize) -> Self { self.num_samples = n; self }
    pub fn with_k_nearest(mut self, k: usize) -> Self { self.k_nearest = k; self }
    pub fn with_max_edge_length(mut self, l: f64) -> Self { self.max_edge_length = l; self }
    pub fn with_collision_samples(mut self, n: usize) -> Self { self.collision_samples = n; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
    pub fn with_lazy(mut self, lazy: bool) -> Self { self.lazy = lazy; self }

    fn dim(&self) -> usize { self.bounds_min.len() }
}

impl fmt::Display for PrmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrmConfig(dim={}, samples={}, k={}, lazy={})",
            self.dim(), self.num_samples, self.k_nearest, self.lazy,
        )
    }
}

// ── PRM Result ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PrmResult {
    pub path: Vec<Point>,
    pub cost: f64,
    pub nodes_in_roadmap: usize,
    pub edges_in_roadmap: usize,
    pub edges_validated: usize,
}

impl fmt::Display for PrmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrmResult(waypoints={}, cost={:.3}, nodes={}, edges={}, validated={})",
            self.path.len(), self.cost, self.nodes_in_roadmap,
            self.edges_in_roadmap, self.edges_validated,
        )
    }
}

// ── Dijkstra priority queue entry ───────────────────────────────

#[derive(Debug, Clone)]
struct DijkEntry {
    node: usize,
    cost: f64,
}

impl PartialEq for DijkEntry {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}
impl Eq for DijkEntry {}

impl PartialOrd for DijkEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap
        other.cost.partial_cmp(&self.cost).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── PRM Planner ─────────────────────────────────────────────────

/// Probabilistic Roadmap planner.
pub struct PrmPlanner {
    config: PrmConfig,
    obstacles: Vec<AabbObstacle>,
    nodes: Vec<RoadmapNode>,
    edges: Vec<RoadmapEdge>,
    built: bool,
}

impl PrmPlanner {
    pub fn new(config: PrmConfig) -> Self {
        Self {
            config,
            obstacles: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            built: false,
        }
    }

    pub fn with_obstacles(mut self, obs: Vec<AabbObstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    fn point_free(&self, p: &Point) -> bool {
        !self.obstacles.iter().any(|o| o.contains(p))
    }

    fn edge_free(&self, a: &Point, b: &Point) -> bool {
        !self.obstacles.iter().any(|o| o.intersects_segment(a, b, self.config.collision_samples))
    }

    /// Find k-nearest nodes to a query point.
    fn k_nearest(&self, query: &Point, k: usize) -> Vec<(usize, f64)> {
        let mut dists: Vec<(usize, f64)> = self.nodes.iter()
            .enumerate()
            .map(|(i, n)| (i, n.point.distance_to(query)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    /// Build the roadmap by sampling and connecting nodes.
    pub fn build(&mut self) {
        let mut rng = SimpleRng::new(self.config.seed);
        self.nodes.clear();
        self.edges.clear();

        // Sample nodes
        while self.nodes.len() < self.config.num_samples {
            let coords: Vec<f64> = (0..self.config.dim())
                .map(|d| rng.next_range(self.config.bounds_min[d], self.config.bounds_max[d]))
                .collect();
            let p = Point { coords };
            if self.point_free(&p) {
                self.nodes.push(RoadmapNode { point: p, edges: Vec::new() });
            }
        }

        // Connect k-nearest neighbors
        for i in 0..self.nodes.len() {
            let neighbors = self.k_nearest(&self.nodes[i].point, self.config.k_nearest + 1);
            for (j, dist) in neighbors {
                if j == i || dist > self.config.max_edge_length {
                    continue;
                }
                // Check if edge already exists
                let exists = self.nodes[i].edges.iter().any(|ei| {
                    let e = &self.edges[*ei];
                    (e.from == i && e.to == j) || (e.from == j && e.to == i)
                });
                if exists { continue; }

                let (validated, collision_free) = if self.config.lazy {
                    (false, true) // Assume free, validate later
                } else {
                    let free = self.edge_free(&self.nodes[i].point, &self.nodes[j].point);
                    (true, free)
                };

                if !validated || collision_free {
                    let eidx = self.edges.len();
                    self.edges.push(RoadmapEdge { from: i, to: j, cost: dist, validated, collision_free });
                    self.nodes[i].edges.push(eidx);
                    self.nodes[j].edges.push(eidx);
                }
            }
        }

        self.built = true;
    }

    /// Add a point to the roadmap and connect it.
    fn add_point(&mut self, p: Point) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(RoadmapNode { point: p, edges: Vec::new() });

        let neighbors = self.k_nearest(&self.nodes[idx].point, self.config.k_nearest + 1);
        for (j, dist) in neighbors {
            if j == idx || dist > self.config.max_edge_length { continue; }
            let free = self.edge_free(&self.nodes[idx].point, &self.nodes[j].point);
            if free {
                let eidx = self.edges.len();
                self.edges.push(RoadmapEdge {
                    from: idx, to: j, cost: dist, validated: true, collision_free: true,
                });
                self.nodes[idx].edges.push(eidx);
                self.nodes[j].edges.push(eidx);
            }
        }
        idx
    }

    /// Query the roadmap for a path from start to goal.
    pub fn query(&mut self, start: Point, goal: Point) -> Result<PrmResult, PrmError> {
        if !self.built {
            self.build();
        }
        if !self.point_free(&start) { return Err(PrmError::StartInCollision); }
        if !self.point_free(&goal) { return Err(PrmError::GoalInCollision); }

        let start_idx = self.add_point(start);
        let goal_idx = self.add_point(goal);

        let result = self.dijkstra(start_idx, goal_idx);

        // Clean up added nodes (optional for multi-query, keep for now)
        result
    }

    /// Dijkstra search with lazy collision checking.
    fn dijkstra(&mut self, start: usize, goal: usize) -> Result<PrmResult, PrmError> {
        let n = self.nodes.len();
        let mut dist = vec![f64::MAX; n];
        let mut prev: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut edges_validated = 0usize;

        dist[start] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push(DijkEntry { node: start, cost: 0.0 });

        while let Some(DijkEntry { node, cost }) = heap.pop() {
            if node == goal {
                let path = self.reconstruct_path(&prev, goal);
                let total_edges = self.edges.len();
                return Ok(PrmResult {
                    cost,
                    nodes_in_roadmap: n,
                    edges_in_roadmap: total_edges,
                    edges_validated,
                    path,
                });
            }
            if visited[node] { continue; }
            if cost > dist[node] { continue; }
            visited[node] = true;

            let edge_indices: Vec<usize> = self.nodes[node].edges.clone();
            for eidx in edge_indices {
                let edge = &self.edges[eidx];
                let neighbor = if edge.from == node { edge.to } else { edge.from };
                if visited[neighbor] { continue; }

                // Lazy validation
                if !edge.validated {
                    edges_validated += 1;
                    let free = self.edge_free(
                        &self.nodes[edge.from].point,
                        &self.nodes[edge.to].point,
                    );
                    let e = &mut self.edges[eidx];
                    e.validated = true;
                    e.collision_free = free;
                }

                if !self.edges[eidx].collision_free { continue; }

                let new_dist = dist[node] + self.edges[eidx].cost;
                if new_dist < dist[neighbor] {
                    dist[neighbor] = new_dist;
                    prev[neighbor] = Some(node);
                    heap.push(DijkEntry { node: neighbor, cost: new_dist });
                }
            }
        }

        Err(PrmError::NoPathFound)
    }

    fn reconstruct_path(&self, prev: &[Option<usize>], goal: usize) -> Vec<Point> {
        let mut path = Vec::new();
        let mut current = Some(goal);
        while let Some(idx) = current {
            path.push(self.nodes[idx].point.clone());
            current = prev[idx];
        }
        path.reverse();
        path
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }
    pub fn is_built(&self) -> bool { self.built }
}

impl fmt::Display for PrmPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrmPlanner(nodes={}, edges={}, built={})",
            self.nodes.len(), self.edges.len(), self.built,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg2d() -> PrmConfig {
        PrmConfig::new(vec![0.0, 0.0], vec![10.0, 10.0])
            .unwrap()
            .with_num_samples(200)
            .with_k_nearest(8)
            .with_max_edge_length(3.0)
    }

    #[test]
    fn test_point_distance() {
        let a = Point::new2(0.0, 0.0);
        let b = Point::new2(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_display() {
        let p = Point::new2(1.0, 2.0);
        let s = format!("{p}");
        assert!(s.contains("1.000"));
    }

    #[test]
    fn test_config_invalid_bounds() {
        assert!(PrmConfig::new(vec![5.0], vec![2.0]).is_err());
        assert!(PrmConfig::new(vec![], vec![]).is_err());
    }

    #[test]
    fn test_config_display() {
        let cfg = cfg2d();
        let s = format!("{cfg}");
        assert!(s.contains("dim=2"));
        assert!(s.contains("samples=200"));
    }

    #[test]
    fn test_build_roadmap() {
        let cfg = cfg2d();
        let mut prm = PrmPlanner::new(cfg);
        prm.build();
        assert!(prm.is_built());
        assert_eq!(prm.node_count(), 200);
        assert!(prm.edge_count() > 0);
    }

    #[test]
    fn test_simple_query() {
        let cfg = cfg2d();
        let mut prm = PrmPlanner::new(cfg);
        let result = prm.query(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        assert!(result.path.len() >= 2);
        assert!(result.cost > 0.0);
    }

    #[test]
    fn test_query_with_obstacle() {
        let cfg = cfg2d().with_num_samples(300);
        let obs = vec![AabbObstacle::new(vec![4.0, 0.0], vec![5.0, 8.0])];
        let mut prm = PrmPlanner::new(cfg).with_obstacles(obs);
        let result = prm.query(Point::new2(1.0, 5.0), Point::new2(9.0, 5.0)).unwrap();
        assert!(result.path.len() >= 2);
        assert!(result.cost > 8.0); // must route around wall
    }

    #[test]
    fn test_start_in_obstacle() {
        let cfg = cfg2d();
        let obs = vec![AabbObstacle::new(vec![0.0, 0.0], vec![2.0, 2.0])];
        let mut prm = PrmPlanner::new(cfg).with_obstacles(obs);
        prm.build();
        assert!(prm.query(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).is_err());
    }

    #[test]
    fn test_lazy_prm() {
        let cfg = cfg2d().with_lazy(true).with_num_samples(200);
        let mut prm = PrmPlanner::new(cfg);
        let result = prm.query(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        assert!(result.path.len() >= 2);
        // In lazy mode, not all edges are validated
        assert!(result.edges_validated <= result.edges_in_roadmap);
    }

    #[test]
    fn test_multi_query() {
        let cfg = cfg2d();
        let mut prm = PrmPlanner::new(cfg);
        prm.build();
        let r1 = prm.query(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        let r2 = prm.query(Point::new2(2.0, 2.0), Point::new2(8.0, 8.0)).unwrap();
        assert!(r1.path.len() >= 2);
        assert!(r2.path.len() >= 2);
    }

    #[test]
    fn test_3d_query() {
        let cfg = PrmConfig::new(vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0])
            .unwrap()
            .with_num_samples(300)
            .with_k_nearest(10)
            .with_max_edge_length(5.0);
        let mut prm = PrmPlanner::new(cfg);
        let result = prm.query(
            Point::new3(1.0, 1.0, 1.0),
            Point::new3(9.0, 9.0, 9.0),
        ).unwrap();
        assert!(result.path.len() >= 2);
    }

    #[test]
    fn test_result_display() {
        let r = PrmResult {
            path: vec![Point::new2(0.0, 0.0), Point::new2(1.0, 1.0)],
            cost: 1.414,
            nodes_in_roadmap: 200,
            edges_in_roadmap: 500,
            edges_validated: 30,
        };
        let s = format!("{r}");
        assert!(s.contains("waypoints=2"));
        assert!(s.contains("validated=30"));
    }

    #[test]
    fn test_prm_display() {
        let cfg = cfg2d();
        let prm = PrmPlanner::new(cfg);
        let s = format!("{prm}");
        assert!(s.contains("PrmPlanner"));
    }

    #[test]
    fn test_aabb_contains() {
        let obs = AabbObstacle::new(vec![2.0, 2.0], vec![4.0, 4.0]);
        assert!(obs.contains(&Point::new2(3.0, 3.0)));
        assert!(!obs.contains(&Point::new2(5.0, 5.0)));
    }

    #[test]
    fn test_aabb_segment() {
        let obs = AabbObstacle::new(vec![4.0, 4.0], vec![6.0, 6.0]);
        assert!(obs.intersects_segment(&Point::new2(0.0, 5.0), &Point::new2(10.0, 5.0), 20));
        assert!(!obs.intersects_segment(&Point::new2(0.0, 0.0), &Point::new2(3.0, 3.0), 20));
    }

    #[test]
    fn test_error_display() {
        let e = PrmError::NoPathFound;
        assert_eq!(format!("{e}"), "no path found in roadmap");
    }

    #[test]
    fn test_k_nearest_count() {
        let cfg = cfg2d().with_num_samples(50).with_k_nearest(5);
        let mut prm = PrmPlanner::new(cfg);
        prm.build();
        assert!(prm.edge_count() > 0);
    }
}
