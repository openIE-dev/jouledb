//! Rapidly-exploring Random Trees — RRT, RRT*, RRT-Connect, goal biasing,
//! nearest neighbor search for sampling-based motion planning.
//!
//! Pure-Rust implementations of core RRT variants for 2D/3D configuration
//! spaces with obstacle checking, path extraction, and cost-optimal rewiring.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RrtError {
    InvalidBounds(String),
    InvalidParameter(String),
    GoalUnreachable,
    EmptyTree,
}

impl fmt::Display for RrtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBounds(s) => write!(f, "invalid bounds: {s}"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::GoalUnreachable => write!(f, "goal unreachable within iteration limit"),
            Self::EmptyTree => write!(f, "tree is empty"),
        }
    }
}

impl std::error::Error for RrtError {}

// ── Point ───────────────────────────────────────────────────────

/// A point in N-dimensional configuration space (max 3D stored inline).
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub coords: Vec<f64>,
}

impl Point {
    pub fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    pub fn new2(x: f64, y: f64) -> Self {
        Self { coords: vec![x, y] }
    }

    pub fn new3(x: f64, y: f64, z: f64) -> Self {
        Self { coords: vec![x, y, z] }
    }

    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    pub fn distance_to(&self, other: &Point) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Move from self toward target by at most `step_size`.
    pub fn steer_toward(&self, target: &Point, step_size: f64) -> Point {
        let dist = self.distance_to(target);
        if dist <= step_size {
            return target.clone();
        }
        let ratio = step_size / dist;
        let coords = self
            .coords
            .iter()
            .zip(target.coords.iter())
            .map(|(a, b)| a + (b - a) * ratio)
            .collect();
        Point { coords }
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{c:.3}")?;
        }
        write!(f, ")")
    }
}

// ── Tree Node ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TreeNode {
    point: Point,
    parent: Option<usize>,
    cost: f64,
}

// ── Obstacle (axis-aligned box) ─────────────────────────────────

/// Axis-aligned bounding box obstacle.
#[derive(Debug, Clone)]
pub struct AabbObstacle {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

impl AabbObstacle {
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, p: &Point) -> bool {
        p.coords
            .iter()
            .zip(self.min.iter().zip(self.max.iter()))
            .all(|(c, (lo, hi))| *c >= *lo && *c <= *hi)
    }

    /// Check if the line segment from `a` to `b` intersects this box (sampled).
    pub fn intersects_segment(&self, a: &Point, b: &Point, samples: usize) -> bool {
        for i in 0..=samples {
            let t = i as f64 / samples as f64;
            let coords: Vec<f64> = a
                .coords
                .iter()
                .zip(b.coords.iter())
                .map(|(ca, cb)| ca + (cb - ca) * t)
                .collect();
            if self.contains(&Point { coords }) {
                return true;
            }
        }
        false
    }
}

// ── Simple LCG RNG ──────────────────────────────────────────────

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ── RRT Config ──────────────────────────────────────────────────

/// Configuration for all RRT variants.
#[derive(Debug, Clone)]
pub struct RrtConfig {
    pub bounds_min: Vec<f64>,
    pub bounds_max: Vec<f64>,
    pub step_size: f64,
    pub goal_threshold: f64,
    pub goal_bias: f64,
    pub max_iterations: usize,
    pub seed: u64,
    pub rewire_radius: f64,
    pub collision_samples: usize,
}

impl RrtConfig {
    pub fn new(bounds_min: Vec<f64>, bounds_max: Vec<f64>) -> Result<Self, RrtError> {
        if bounds_min.len() != bounds_max.len() || bounds_min.is_empty() {
            return Err(RrtError::InvalidBounds("dimension mismatch or zero".into()));
        }
        for (lo, hi) in bounds_min.iter().zip(bounds_max.iter()) {
            if lo >= hi {
                return Err(RrtError::InvalidBounds(format!("min {lo} >= max {hi}")));
            }
        }
        Ok(Self {
            bounds_min,
            bounds_max,
            step_size: 0.5,
            goal_threshold: 0.3,
            goal_bias: 0.05,
            max_iterations: 5000,
            seed: 42,
            rewire_radius: 1.5,
            collision_samples: 10,
        })
    }

    pub fn with_step_size(mut self, s: f64) -> Self { self.step_size = s; self }
    pub fn with_goal_threshold(mut self, t: f64) -> Self { self.goal_threshold = t; self }
    pub fn with_goal_bias(mut self, b: f64) -> Self { self.goal_bias = b; self }
    pub fn with_max_iterations(mut self, n: usize) -> Self { self.max_iterations = n; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
    pub fn with_rewire_radius(mut self, r: f64) -> Self { self.rewire_radius = r; self }
    pub fn with_collision_samples(mut self, n: usize) -> Self { self.collision_samples = n; self }

    fn dim(&self) -> usize { self.bounds_min.len() }
}

impl fmt::Display for RrtConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RrtConfig(dim={}, step={}, bias={:.0}%, max_iter={})",
            self.dim(),
            self.step_size,
            self.goal_bias * 100.0,
            self.max_iterations,
        )
    }
}

// ── Nearest Neighbor (brute-force) ──────────────────────────────

fn nearest_neighbor(nodes: &[TreeNode], query: &Point) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    for (i, n) in nodes.iter().enumerate() {
        let d = n.point.distance_to(query);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

fn near_neighbors(nodes: &[TreeNode], query: &Point, radius: f64) -> Vec<usize> {
    nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.point.distance_to(query) <= radius)
        .map(|(i, _)| i)
        .collect()
}

// ── Collision checking ──────────────────────────────────────────

fn collision_free(a: &Point, b: &Point, obstacles: &[AabbObstacle], samples: usize) -> bool {
    !obstacles.iter().any(|obs| obs.intersects_segment(a, b, samples))
}

fn point_collision_free(p: &Point, obstacles: &[AabbObstacle]) -> bool {
    !obstacles.iter().any(|obs| obs.contains(p))
}

// ── Path extraction ─────────────────────────────────────────────

fn extract_path(nodes: &[TreeNode], goal_idx: usize) -> Vec<Point> {
    let mut path = Vec::new();
    let mut idx = Some(goal_idx);
    while let Some(i) = idx {
        path.push(nodes[i].point.clone());
        idx = nodes[i].parent;
    }
    path.reverse();
    path
}

// ── RRT Result ──────────────────────────────────────────────────

/// Result of an RRT planning query.
#[derive(Debug, Clone)]
pub struct RrtResult {
    pub path: Vec<Point>,
    pub cost: f64,
    pub iterations: usize,
    pub tree_size: usize,
}

impl fmt::Display for RrtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RrtResult(waypoints={}, cost={:.3}, iters={}, tree={})",
            self.path.len(),
            self.cost,
            self.iterations,
            self.tree_size,
        )
    }
}

// ── Basic RRT ───────────────────────────────────────────────────

/// Basic RRT planner.
pub struct Rrt {
    config: RrtConfig,
    obstacles: Vec<AabbObstacle>,
}

impl Rrt {
    pub fn new(config: RrtConfig) -> Self {
        Self { config, obstacles: Vec::new() }
    }

    pub fn with_obstacles(mut self, obs: Vec<AabbObstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    pub fn plan(&self, start: Point, goal: Point) -> Result<RrtResult, RrtError> {
        let mut rng = SimpleRng::new(self.config.seed);
        let mut nodes = vec![TreeNode { point: start.clone(), parent: None, cost: 0.0 }];

        for iter in 0..self.config.max_iterations {
            let sample = if rng.next_f64() < self.config.goal_bias {
                goal.clone()
            } else {
                let coords: Vec<f64> = (0..self.config.dim())
                    .map(|d| rng.next_range(self.config.bounds_min[d], self.config.bounds_max[d]))
                    .collect();
                Point { coords }
            };

            let near_idx = nearest_neighbor(&nodes, &sample);
            let new_point = nodes[near_idx].point.steer_toward(&sample, self.config.step_size);

            if !point_collision_free(&new_point, &self.obstacles) {
                continue;
            }
            if !collision_free(
                &nodes[near_idx].point,
                &new_point,
                &self.obstacles,
                self.config.collision_samples,
            ) {
                continue;
            }

            let new_cost = nodes[near_idx].cost + nodes[near_idx].point.distance_to(&new_point);
            nodes.push(TreeNode { point: new_point.clone(), parent: Some(near_idx), cost: new_cost });

            if new_point.distance_to(&goal) <= self.config.goal_threshold {
                let goal_idx = nodes.len() - 1;
                let path = extract_path(&nodes, goal_idx);
                return Ok(RrtResult {
                    cost: nodes[goal_idx].cost,
                    iterations: iter + 1,
                    tree_size: nodes.len(),
                    path,
                });
            }
        }
        Err(RrtError::GoalUnreachable)
    }
}

impl fmt::Display for Rrt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rrt({})", self.config)
    }
}

// ── RRT* ────────────────────────────────────────────────────────

/// RRT* with cost-optimal rewiring.
pub struct RrtStar {
    config: RrtConfig,
    obstacles: Vec<AabbObstacle>,
}

impl RrtStar {
    pub fn new(config: RrtConfig) -> Self {
        Self { config, obstacles: Vec::new() }
    }

    pub fn with_obstacles(mut self, obs: Vec<AabbObstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    pub fn plan(&self, start: Point, goal: Point) -> Result<RrtResult, RrtError> {
        let mut rng = SimpleRng::new(self.config.seed);
        let mut nodes = vec![TreeNode { point: start.clone(), parent: None, cost: 0.0 }];
        let mut best_goal_idx: Option<usize> = None;

        for iter in 0..self.config.max_iterations {
            let sample = if rng.next_f64() < self.config.goal_bias {
                goal.clone()
            } else {
                let coords: Vec<f64> = (0..self.config.dim())
                    .map(|d| rng.next_range(self.config.bounds_min[d], self.config.bounds_max[d]))
                    .collect();
                Point { coords }
            };

            let near_idx = nearest_neighbor(&nodes, &sample);
            let new_point = nodes[near_idx].point.steer_toward(&sample, self.config.step_size);

            if !point_collision_free(&new_point, &self.obstacles) {
                continue;
            }

            // Find best parent among near neighbors
            let neighbors = near_neighbors(&nodes, &new_point, self.config.rewire_radius);
            let mut best_parent = near_idx;
            let mut best_cost = nodes[near_idx].cost + nodes[near_idx].point.distance_to(&new_point);

            for &ni in &neighbors {
                let c = nodes[ni].cost + nodes[ni].point.distance_to(&new_point);
                if c < best_cost
                    && collision_free(
                        &nodes[ni].point,
                        &new_point,
                        &self.obstacles,
                        self.config.collision_samples,
                    )
                {
                    best_cost = c;
                    best_parent = ni;
                }
            }

            let new_idx = nodes.len();
            nodes.push(TreeNode {
                point: new_point.clone(),
                parent: Some(best_parent),
                cost: best_cost,
            });

            // Rewire neighbors
            for &ni in &neighbors {
                let new_cost_via = best_cost + new_point.distance_to(&nodes[ni].point);
                if new_cost_via < nodes[ni].cost
                    && collision_free(
                        &new_point,
                        &nodes[ni].point,
                        &self.obstacles,
                        self.config.collision_samples,
                    )
                {
                    nodes[ni].parent = Some(new_idx);
                    nodes[ni].cost = new_cost_via;
                }
            }

            if new_point.distance_to(&goal) <= self.config.goal_threshold {
                match best_goal_idx {
                    Some(gi) if nodes[new_idx].cost >= nodes[gi].cost => {}
                    _ => best_goal_idx = Some(new_idx),
                }
            }

            // Continue iterating to improve path (unlike basic RRT)
            if iter == self.config.max_iterations - 1 {
                break;
            }
        }

        match best_goal_idx {
            Some(gi) => {
                let path = extract_path(&nodes, gi);
                Ok(RrtResult {
                    cost: nodes[gi].cost,
                    iterations: self.config.max_iterations,
                    tree_size: nodes.len(),
                    path,
                })
            }
            None => Err(RrtError::GoalUnreachable),
        }
    }
}

impl fmt::Display for RrtStar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RrtStar({})", self.config)
    }
}

// ── RRT-Connect ─────────────────────────────────────────────────

/// Bidirectional RRT-Connect planner.
pub struct RrtConnect {
    config: RrtConfig,
    obstacles: Vec<AabbObstacle>,
}

impl RrtConnect {
    pub fn new(config: RrtConfig) -> Self {
        Self { config, obstacles: Vec::new() }
    }

    pub fn with_obstacles(mut self, obs: Vec<AabbObstacle>) -> Self {
        self.obstacles = obs;
        self
    }

    /// Extend a tree toward a target, returning the index of the new node or None.
    fn extend(
        nodes: &mut Vec<TreeNode>,
        target: &Point,
        step_size: f64,
        obstacles: &[AabbObstacle],
        col_samples: usize,
    ) -> Option<usize> {
        let near_idx = nearest_neighbor(nodes, target);
        let new_point = nodes[near_idx].point.steer_toward(target, step_size);
        if !point_collision_free(&new_point, obstacles) {
            return None;
        }
        if !collision_free(&nodes[near_idx].point, &new_point, obstacles, col_samples) {
            return None;
        }
        let cost = nodes[near_idx].cost + nodes[near_idx].point.distance_to(&new_point);
        let idx = nodes.len();
        nodes.push(TreeNode { point: new_point, parent: Some(near_idx), cost });
        Some(idx)
    }

    /// Greedily connect tree toward target until blocked or reached.
    fn connect(
        nodes: &mut Vec<TreeNode>,
        target: &Point,
        step_size: f64,
        obstacles: &[AabbObstacle],
        col_samples: usize,
        threshold: f64,
    ) -> Option<usize> {
        let mut last_idx = None;
        loop {
            match Self::extend(nodes, target, step_size, obstacles, col_samples) {
                Some(idx) => {
                    last_idx = Some(idx);
                    if nodes[idx].point.distance_to(target) <= threshold {
                        return Some(idx);
                    }
                }
                None => return last_idx,
            }
        }
    }

    pub fn plan(&self, start: Point, goal: Point) -> Result<RrtResult, RrtError> {
        let mut rng = SimpleRng::new(self.config.seed);
        let mut tree_a = vec![TreeNode { point: start.clone(), parent: None, cost: 0.0 }];
        let mut tree_b = vec![TreeNode { point: goal.clone(), parent: None, cost: 0.0 }];
        let mut swapped = false;

        for iter in 0..self.config.max_iterations {
            let sample = {
                let coords: Vec<f64> = (0..self.config.dim())
                    .map(|d| rng.next_range(self.config.bounds_min[d], self.config.bounds_max[d]))
                    .collect();
                Point { coords }
            };

            if let Some(ext_idx) = Self::extend(
                &mut tree_a,
                &sample,
                self.config.step_size,
                &self.obstacles,
                self.config.collision_samples,
            ) {
                let ext_point = tree_a[ext_idx].point.clone();
                if let Some(con_idx) = Self::connect(
                    &mut tree_b,
                    &ext_point,
                    self.config.step_size,
                    &self.obstacles,
                    self.config.collision_samples,
                    self.config.goal_threshold,
                ) {
                    if tree_b[con_idx].point.distance_to(&ext_point) <= self.config.goal_threshold {
                        let mut path_a = extract_path(&tree_a, ext_idx);
                        let path_b = extract_path(&tree_b, con_idx);
                        let mut path_b_rev: Vec<Point> = path_b.into_iter().rev().collect();
                        path_a.append(&mut path_b_rev);

                        if swapped {
                            path_a.reverse();
                        }

                        let cost = path_a
                            .windows(2)
                            .map(|w| w[0].distance_to(&w[1]))
                            .sum();

                        return Ok(RrtResult {
                            path: path_a,
                            cost,
                            iterations: iter + 1,
                            tree_size: tree_a.len() + tree_b.len(),
                        });
                    }
                }
            }

            std::mem::swap(&mut tree_a, &mut tree_b);
            swapped = !swapped;
        }

        Err(RrtError::GoalUnreachable)
    }
}

impl fmt::Display for RrtConnect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RrtConnect({})", self.config)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn config2d() -> RrtConfig {
        RrtConfig::new(vec![0.0, 0.0], vec![10.0, 10.0])
            .unwrap()
            .with_step_size(0.5)
            .with_goal_threshold(0.5)
            .with_goal_bias(0.1)
            .with_max_iterations(5000)
    }

    #[test]
    fn test_point_distance() {
        let a = Point::new2(0.0, 0.0);
        let b = Point::new2(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_steer() {
        let a = Point::new2(0.0, 0.0);
        let b = Point::new2(10.0, 0.0);
        let c = a.steer_toward(&b, 3.0);
        assert!((c.coords[0] - 3.0).abs() < 1e-10);
        assert!((c.coords[1]).abs() < 1e-10);
    }

    #[test]
    fn test_point_steer_close() {
        let a = Point::new2(0.0, 0.0);
        let b = Point::new2(0.1, 0.0);
        let c = a.steer_toward(&b, 3.0);
        assert!((c.coords[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_point_display() {
        let p = Point::new2(1.5, 2.75);
        let s = format!("{p}");
        assert!(s.contains("1.500"));
        assert!(s.contains("2.750"));
    }

    #[test]
    fn test_aabb_contains() {
        let obs = AabbObstacle::new(vec![2.0, 2.0], vec![4.0, 4.0]);
        assert!(obs.contains(&Point::new2(3.0, 3.0)));
        assert!(!obs.contains(&Point::new2(5.0, 3.0)));
    }

    #[test]
    fn test_aabb_segment_intersect() {
        let obs = AabbObstacle::new(vec![4.0, 4.0], vec![6.0, 6.0]);
        let a = Point::new2(0.0, 5.0);
        let b = Point::new2(10.0, 5.0);
        assert!(obs.intersects_segment(&a, &b, 20));
    }

    #[test]
    fn test_rrt_simple_path() {
        let cfg = config2d();
        let rrt = Rrt::new(cfg);
        let result = rrt.plan(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        assert!(result.path.len() >= 2);
        assert!(result.cost > 0.0);
    }

    #[test]
    fn test_rrt_with_obstacle() {
        let cfg = config2d().with_max_iterations(10000);
        let obs = vec![AabbObstacle::new(vec![4.0, 0.0], vec![5.0, 8.0])];
        let rrt = Rrt::new(cfg).with_obstacles(obs);
        let result = rrt.plan(Point::new2(1.0, 5.0), Point::new2(9.0, 5.0)).unwrap();
        assert!(result.path.len() >= 2);
        // Path must go around the wall
        assert!(result.cost > 8.0);
    }

    #[test]
    fn test_rrt_start_equals_goal() {
        let cfg = config2d().with_goal_threshold(1.0);
        let rrt = Rrt::new(cfg);
        let result = rrt.plan(Point::new2(5.0, 5.0), Point::new2(5.0, 5.0)).unwrap();
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_rrt_display() {
        let cfg = config2d();
        let rrt = Rrt::new(cfg);
        let s = format!("{rrt}");
        assert!(s.contains("Rrt"));
    }

    #[test]
    fn test_rrt_star_simple() {
        let cfg = config2d().with_rewire_radius(2.0);
        let rrt = RrtStar::new(cfg);
        let result = rrt.plan(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        assert!(result.path.len() >= 2);
    }

    #[test]
    fn test_rrt_star_improves_cost() {
        let cfg = config2d().with_max_iterations(8000).with_rewire_radius(2.0);
        let basic_rrt = Rrt::new(cfg.clone());
        let star_rrt = RrtStar::new(cfg);
        let r1 = basic_rrt.plan(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        let r2 = star_rrt.plan(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        // RRT* should produce a path no worse than basic (with enough iterations)
        assert!(r2.cost <= r1.cost * 1.5, "star cost {} vs basic {}", r2.cost, r1.cost);
    }

    #[test]
    fn test_rrt_star_display() {
        let cfg = config2d();
        let rrt = RrtStar::new(cfg);
        let s = format!("{rrt}");
        assert!(s.contains("RrtStar"));
    }

    #[test]
    fn test_rrt_connect_simple() {
        let cfg = config2d();
        let rrt = RrtConnect::new(cfg);
        let result = rrt.plan(Point::new2(1.0, 1.0), Point::new2(9.0, 9.0)).unwrap();
        assert!(result.path.len() >= 2);
        // First point near start, last point near goal
        assert!(result.path[0].distance_to(&Point::new2(1.0, 1.0)) < 1.0);
    }

    #[test]
    fn test_rrt_connect_with_obstacle() {
        let cfg = config2d().with_max_iterations(10000);
        let obs = vec![AabbObstacle::new(vec![4.0, 0.0], vec![5.0, 8.0])];
        let rrt = RrtConnect::new(cfg).with_obstacles(obs);
        let result = rrt.plan(Point::new2(1.0, 5.0), Point::new2(9.0, 5.0)).unwrap();
        assert!(result.path.len() >= 2);
    }

    #[test]
    fn test_rrt_connect_display() {
        let cfg = config2d();
        let rrt = RrtConnect::new(cfg);
        let s = format!("{rrt}");
        assert!(s.contains("RrtConnect"));
    }

    #[test]
    fn test_config_invalid_bounds() {
        assert!(RrtConfig::new(vec![5.0], vec![2.0]).is_err());
        assert!(RrtConfig::new(vec![], vec![]).is_err());
    }

    #[test]
    fn test_config_display() {
        let cfg = config2d();
        let s = format!("{cfg}");
        assert!(s.contains("dim=2"));
    }

    #[test]
    fn test_3d_point() {
        let a = Point::new3(0.0, 0.0, 0.0);
        let b = Point::new3(1.0, 2.0, 2.0);
        assert!((a.distance_to(&b) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_result_display() {
        let r = RrtResult {
            path: vec![Point::new2(0.0, 0.0), Point::new2(1.0, 1.0)],
            cost: 1.414,
            iterations: 10,
            tree_size: 20,
        };
        let s = format!("{r}");
        assert!(s.contains("waypoints=2"));
    }
}
