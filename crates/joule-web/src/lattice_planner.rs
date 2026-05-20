//! State Lattice Planner — motion primitive library, graph search over lattice,
//! cost functions for structured motion planning in discrete state spaces.
//!
//! Pure-Rust lattice planner with configurable heading discretization, arc-based
//! motion primitives, A*-based graph search, and multi-objective cost evaluation.

use std::collections::BinaryHeap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LatticeError {
    InvalidParameter(String),
    NoPrimitives,
    NoPathFound,
    OutOfBounds,
    StartBlocked,
    GoalBlocked,
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::NoPrimitives => write!(f, "no motion primitives defined"),
            Self::NoPathFound => write!(f, "no path found"),
            Self::OutOfBounds => write!(f, "state out of bounds"),
            Self::StartBlocked => write!(f, "start state is blocked"),
            Self::GoalBlocked => write!(f, "goal state is blocked"),
        }
    }
}

impl std::error::Error for LatticeError {}

// ── Lattice State ───────────────────────────────────────────────

/// Discrete state in the lattice: (grid_x, grid_y, heading_index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LatticeState {
    pub gx: i32,
    pub gy: i32,
    pub theta_idx: usize,
}

impl LatticeState {
    pub fn new(gx: i32, gy: i32, theta_idx: usize) -> Self {
        Self { gx, gy, theta_idx }
    }
}

impl fmt::Display for LatticeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, h={})", self.gx, self.gy, self.theta_idx)
    }
}

// ── Waypoint ────────────────────────────────────────────────────

/// Continuous 2D waypoint along a motion primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Waypoint {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Waypoint {
    pub fn new(x: f64, y: f64, theta: f64) -> Self { Self { x, y, theta } }

    pub fn distance_to(&self, other: &Waypoint) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl fmt::Display for Waypoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.3}, {:.3}, {:.3})", self.x, self.y, self.theta)
    }
}

// ── Motion Primitive (for lattice) ──────────────────────────────

/// A single motion primitive: start heading -> end state offset + path.
#[derive(Debug, Clone)]
pub struct LatticePrimitive {
    pub start_theta_idx: usize,
    pub end_dx: i32,
    pub end_dy: i32,
    pub end_theta_idx: usize,
    pub path: Vec<Waypoint>,
    pub cost: f64,
}

impl LatticePrimitive {
    pub fn new(
        start_theta_idx: usize,
        end_dx: i32,
        end_dy: i32,
        end_theta_idx: usize,
        path: Vec<Waypoint>,
        cost: f64,
    ) -> Self {
        Self { start_theta_idx, end_dx, end_dy, end_theta_idx, path, cost }
    }

    pub fn arc_length(&self) -> f64 {
        if self.path.len() < 2 { return 0.0; }
        self.path.windows(2).map(|w| w[0].distance_to(&w[1])).sum()
    }
}

impl fmt::Display for LatticePrimitive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "Prim(h{}->({},{},h{}), cost={:.3})",
            self.start_theta_idx, self.end_dx, self.end_dy,
            self.end_theta_idx, self.cost,
        )
    }
}

// ── Primitive Library Builder ───────────────────────────────────

/// Generates a standard set of motion primitives for a lattice planner.
pub struct PrimitiveLibrary {
    pub num_headings: usize,
    pub resolution: f64,
    pub primitives: Vec<LatticePrimitive>,
}

impl PrimitiveLibrary {
    /// Generate primitives for the given heading discretization.
    pub fn generate(num_headings: usize, resolution: f64) -> Result<Self, LatticeError> {
        if num_headings == 0 {
            return Err(LatticeError::InvalidParameter("num_headings must be > 0".into()));
        }
        if resolution <= 0.0 {
            return Err(LatticeError::InvalidParameter("resolution must be > 0".into()));
        }

        let mut primitives = Vec::new();
        let d_theta = 2.0 * std::f64::consts::PI / num_headings as f64;

        for hi in 0..num_headings {
            let base_theta = hi as f64 * d_theta;

            // Straight
            let dx = (base_theta.cos() * 1.0).round() as i32;
            let dy = (base_theta.sin() * 1.0).round() as i32;
            let path = Self::straight_path(base_theta, resolution, 1);
            primitives.push(LatticePrimitive::new(hi, dx, dy, hi, path, resolution));

            // Longer straight
            let dx2 = (base_theta.cos() * 2.0).round() as i32;
            let dy2 = (base_theta.sin() * 2.0).round() as i32;
            let path2 = Self::straight_path(base_theta, resolution, 2);
            primitives.push(LatticePrimitive::new(hi, dx2, dy2, hi, path2, resolution * 2.0));

            // Turn left
            let left_hi = (hi + 1) % num_headings;
            let left_theta = left_hi as f64 * d_theta;
            let mid_theta = base_theta + d_theta * 0.5;
            let ldx = (left_theta.cos() * 1.0).round() as i32;
            let ldy = (left_theta.sin() * 1.0).round() as i32;
            let left_path = Self::arc_path(base_theta, left_theta, mid_theta, resolution);
            let left_cost = resolution * 1.2; // slight penalty for turning
            primitives.push(LatticePrimitive::new(hi, ldx, ldy, left_hi, left_path, left_cost));

            // Turn right
            let right_hi = if hi == 0 { num_headings - 1 } else { hi - 1 };
            let right_theta = right_hi as f64 * d_theta;
            let mid_theta_r = base_theta - d_theta * 0.5;
            let rdx = (right_theta.cos() * 1.0).round() as i32;
            let rdy = (right_theta.sin() * 1.0).round() as i32;
            let right_path = Self::arc_path(base_theta, right_theta, mid_theta_r, resolution);
            let right_cost = resolution * 1.2;
            primitives.push(LatticePrimitive::new(hi, rdx, rdy, right_hi, right_path, right_cost));
        }

        Ok(Self { num_headings, resolution, primitives })
    }

    fn straight_path(theta: f64, resolution: f64, steps: usize) -> Vec<Waypoint> {
        let mut path = Vec::new();
        let dx = theta.cos() * resolution;
        let dy = theta.sin() * resolution;
        for i in 0..=steps {
            path.push(Waypoint::new(dx * i as f64, dy * i as f64, theta));
        }
        path
    }

    fn arc_path(start_theta: f64, end_theta: f64, mid_theta: f64, resolution: f64) -> Vec<Waypoint> {
        let n = 5;
        let mut path = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let theta = start_theta + (end_theta - start_theta) * t;
            let x = mid_theta.cos() * resolution * t;
            let y = mid_theta.sin() * resolution * t;
            path.push(Waypoint::new(x, y, theta));
        }
        path
    }

    pub fn primitives_for_heading(&self, theta_idx: usize) -> Vec<&LatticePrimitive> {
        self.primitives.iter()
            .filter(|p| p.start_theta_idx == theta_idx)
            .collect()
    }

    pub fn count(&self) -> usize { self.primitives.len() }
}

impl fmt::Display for PrimitiveLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "PrimitiveLibrary(headings={}, res={:.2}, primitives={})",
            self.num_headings, self.resolution, self.primitives.len(),
        )
    }
}

// ── Cost function ───────────────────────────────────────────────

/// Weights for the multi-objective lattice cost function.
#[derive(Debug, Clone)]
pub struct LatticeCostWeights {
    pub distance: f64,
    pub heading_change: f64,
    pub proximity: f64,
    pub reverse_penalty: f64,
}

impl LatticeCostWeights {
    pub fn new() -> Self {
        Self { distance: 1.0, heading_change: 0.5, proximity: 2.0, reverse_penalty: 5.0 }
    }

    pub fn with_distance(mut self, w: f64) -> Self { self.distance = w; self }
    pub fn with_heading_change(mut self, w: f64) -> Self { self.heading_change = w; self }
    pub fn with_proximity(mut self, w: f64) -> Self { self.proximity = w; self }
    pub fn with_reverse_penalty(mut self, w: f64) -> Self { self.reverse_penalty = w; self }
}

// ── Occupancy grid ──────────────────────────────────────────────

/// 2D occupancy grid for collision checking.
#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    width: usize,
    height: usize,
    offset_x: i32,
    offset_y: i32,
    cells: Vec<bool>,
}

impl OccupancyGrid {
    pub fn new(width: usize, height: usize, offset_x: i32, offset_y: i32) -> Self {
        Self {
            width, height, offset_x, offset_y,
            cells: vec![false; width * height],
        }
    }

    pub fn set_blocked(&mut self, gx: i32, gy: i32, blocked: bool) {
        if let Some(idx) = self.idx(gx, gy) {
            self.cells[idx] = blocked;
        }
    }

    pub fn is_blocked(&self, gx: i32, gy: i32) -> bool {
        match self.idx(gx, gy) {
            Some(idx) => self.cells[idx],
            None => true, // Out of bounds = blocked
        }
    }

    fn idx(&self, gx: i32, gy: i32) -> Option<usize> {
        let lx = gx - self.offset_x;
        let ly = gy - self.offset_y;
        if lx < 0 || ly < 0 || lx >= self.width as i32 || ly >= self.height as i32 {
            None
        } else {
            Some(ly as usize * self.width + lx as usize)
        }
    }

    pub fn block_rect(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                self.set_blocked(x, y, true);
            }
        }
    }
}

// ── Search node ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SearchNode {
    state: LatticeState,
    g_cost: f64,
    f_cost: f64,
}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool { self.f_cost == other.f_cost }
}
impl Eq for SearchNode {}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f_cost.partial_cmp(&self.f_cost).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── Lattice Planner ─────────────────────────────────────────────

/// State lattice planner with A* search.
pub struct LatticePlanner {
    library: PrimitiveLibrary,
    grid: OccupancyGrid,
    cost_weights: LatticeCostWeights,
}

impl LatticePlanner {
    pub fn new(library: PrimitiveLibrary, grid: OccupancyGrid) -> Self {
        Self {
            library,
            grid,
            cost_weights: LatticeCostWeights::new(),
        }
    }

    pub fn with_cost_weights(mut self, w: LatticeCostWeights) -> Self {
        self.cost_weights = w;
        self
    }

    fn heuristic(&self, from: LatticeState, to: LatticeState) -> f64 {
        let dx = (from.gx - to.gx) as f64;
        let dy = (from.gy - to.gy) as f64;
        (dx * dx + dy * dy).sqrt() * self.library.resolution
    }

    fn state_key(&self, s: LatticeState) -> u64 {
        let x = (s.gx as i64 + 100000) as u64;
        let y = (s.gy as i64 + 100000) as u64;
        let h = s.theta_idx as u64;
        x * 200001 * 100 + y * 100 + h
    }

    /// Check if a primitive's path is collision-free.
    fn primitive_collision_free(&self, state: LatticeState, prim: &LatticePrimitive) -> bool {
        // Check endpoint
        let end_gx = state.gx + prim.end_dx;
        let end_gy = state.gy + prim.end_dy;
        if self.grid.is_blocked(end_gx, end_gy) { return false; }

        // Check path waypoints (approximate grid cells)
        for wp in &prim.path {
            let gx = state.gx + (wp.x / self.library.resolution).round() as i32;
            let gy = state.gy + (wp.y / self.library.resolution).round() as i32;
            if self.grid.is_blocked(gx, gy) { return false; }
        }
        true
    }

    /// Plan from start to goal state.
    pub fn plan(
        &self,
        start: LatticeState,
        goal: LatticeState,
        max_expansions: usize,
    ) -> Result<LatticeResult, LatticeError> {
        if self.library.primitives.is_empty() {
            return Err(LatticeError::NoPrimitives);
        }
        if self.grid.is_blocked(start.gx, start.gy) {
            return Err(LatticeError::StartBlocked);
        }
        if self.grid.is_blocked(goal.gx, goal.gy) {
            return Err(LatticeError::GoalBlocked);
        }

        let mut g_costs = std::collections::HashMap::new();
        let mut came_from: std::collections::HashMap<u64, (u64, usize)> = std::collections::HashMap::new();
        let mut open = BinaryHeap::new();
        let mut expansions = 0usize;

        let start_key = self.state_key(start);
        g_costs.insert(start_key, 0.0_f64);
        open.push(SearchNode {
            state: start,
            g_cost: 0.0,
            f_cost: self.heuristic(start, goal),
        });

        // Map keys back to states
        let mut key_to_state = std::collections::HashMap::new();
        key_to_state.insert(start_key, start);

        while let Some(SearchNode { state, g_cost, .. }) = open.pop() {
            let key = self.state_key(state);

            if state.gx == goal.gx && state.gy == goal.gy && state.theta_idx == goal.theta_idx {
                let path = self.reconstruct_path(&came_from, &key_to_state, start_key, key);
                return Ok(LatticeResult {
                    path,
                    cost: g_cost,
                    expansions,
                });
            }

            if let Some(&existing) = g_costs.get(&key) {
                if g_cost > existing { continue; }
            }

            expansions += 1;
            if expansions >= max_expansions { break; }

            let prims = self.library.primitives_for_heading(state.theta_idx);
            for prim in prims {
                if !self.primitive_collision_free(state, prim) { continue; }

                let next = LatticeState::new(
                    state.gx + prim.end_dx,
                    state.gy + prim.end_dy,
                    prim.end_theta_idx,
                );
                let next_key = self.state_key(next);
                let heading_diff = if prim.start_theta_idx != prim.end_theta_idx { 1.0 } else { 0.0 };
                let edge_cost = self.cost_weights.distance * prim.cost
                    + self.cost_weights.heading_change * heading_diff;

                let tentative_g = g_cost + edge_cost;
                let existing_g = g_costs.get(&next_key).copied().unwrap_or(f64::MAX);

                if tentative_g < existing_g {
                    g_costs.insert(next_key, tentative_g);
                    came_from.insert(next_key, (key, 0));
                    key_to_state.insert(next_key, next);
                    open.push(SearchNode {
                        state: next,
                        g_cost: tentative_g,
                        f_cost: tentative_g + self.heuristic(next, goal),
                    });
                }
            }
        }

        Err(LatticeError::NoPathFound)
    }

    fn reconstruct_path(
        &self,
        came_from: &std::collections::HashMap<u64, (u64, usize)>,
        key_to_state: &std::collections::HashMap<u64, LatticeState>,
        start_key: u64,
        goal_key: u64,
    ) -> Vec<LatticeState> {
        let mut path = Vec::new();
        let mut current = goal_key;
        while current != start_key {
            if let Some(&st) = key_to_state.get(&current) {
                path.push(st);
            }
            match came_from.get(&current) {
                Some(&(prev, _)) => current = prev,
                None => break,
            }
        }
        if let Some(&st) = key_to_state.get(&start_key) {
            path.push(st);
        }
        path.reverse();
        path
    }
}

impl fmt::Display for LatticePlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatticePlanner({})", self.library)
    }
}

// ── Lattice Result ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LatticeResult {
    pub path: Vec<LatticeState>,
    pub cost: f64,
    pub expansions: usize,
}

impl fmt::Display for LatticeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "LatticeResult(waypoints={}, cost={:.3}, expansions={})",
            self.path.len(), self.cost, self.expansions,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_state_display() {
        let s = LatticeState::new(3, 4, 2);
        assert_eq!(format!("{s}"), "(3, 4, h=2)");
    }

    #[test]
    fn test_waypoint_distance() {
        let a = Waypoint::new(0.0, 0.0, 0.0);
        let b = Waypoint::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_waypoint_display() {
        let w = Waypoint::new(1.5, 2.5, 0.5);
        let s = format!("{w}");
        assert!(s.contains("1.500"));
    }

    #[test]
    fn test_primitive_arc_length() {
        let path = vec![
            Waypoint::new(0.0, 0.0, 0.0),
            Waypoint::new(1.0, 0.0, 0.0),
            Waypoint::new(2.0, 0.0, 0.0),
        ];
        let prim = LatticePrimitive::new(0, 2, 0, 0, path, 2.0);
        assert!((prim.arc_length() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_primitive_display() {
        let prim = LatticePrimitive::new(0, 1, 0, 0, vec![], 1.0);
        let s = format!("{prim}");
        assert!(s.contains("Prim"));
    }

    #[test]
    fn test_library_generate() {
        let lib = PrimitiveLibrary::generate(8, 1.0).unwrap();
        assert_eq!(lib.num_headings, 8);
        assert!(lib.count() > 0);
        // 4 primitives per heading * 8 headings = 32
        assert_eq!(lib.count(), 32);
    }

    #[test]
    fn test_library_invalid() {
        assert!(PrimitiveLibrary::generate(0, 1.0).is_err());
        assert!(PrimitiveLibrary::generate(4, 0.0).is_err());
    }

    #[test]
    fn test_library_per_heading() {
        let lib = PrimitiveLibrary::generate(4, 1.0).unwrap();
        let prims = lib.primitives_for_heading(0);
        assert_eq!(prims.len(), 4); // straight, long straight, left, right
    }

    #[test]
    fn test_library_display() {
        let lib = PrimitiveLibrary::generate(4, 1.0).unwrap();
        let s = format!("{lib}");
        assert!(s.contains("PrimitiveLibrary"));
    }

    #[test]
    fn test_occupancy_grid() {
        let mut grid = OccupancyGrid::new(20, 20, -10, -10);
        assert!(!grid.is_blocked(0, 0));
        grid.set_blocked(5, 5, true);
        assert!(grid.is_blocked(5, 5));
        assert!(!grid.is_blocked(0, 0));
    }

    #[test]
    fn test_occupancy_out_of_bounds() {
        let grid = OccupancyGrid::new(10, 10, 0, 0);
        assert!(grid.is_blocked(100, 100)); // Out of bounds = blocked
    }

    #[test]
    fn test_occupancy_block_rect() {
        let mut grid = OccupancyGrid::new(20, 20, 0, 0);
        grid.block_rect(2, 2, 5, 5);
        assert!(grid.is_blocked(3, 3));
        assert!(!grid.is_blocked(0, 0));
    }

    #[test]
    fn test_simple_plan() {
        let lib = PrimitiveLibrary::generate(4, 1.0).unwrap();
        let grid = OccupancyGrid::new(40, 40, -20, -20);
        let planner = LatticePlanner::new(lib, grid);
        let start = LatticeState::new(0, 0, 0);
        let goal = LatticeState::new(3, 0, 0);
        let result = planner.plan(start, goal, 10000).unwrap();
        assert!(result.path.len() >= 2);
        assert_eq!(result.path[0], start);
        assert_eq!(*result.path.last().unwrap(), goal);
    }

    #[test]
    fn test_plan_with_obstacle() {
        let lib = PrimitiveLibrary::generate(8, 1.0).unwrap();
        let mut grid = OccupancyGrid::new(40, 40, -20, -20);
        grid.block_rect(2, -1, 2, 1); // Block column at x=2
        let planner = LatticePlanner::new(lib, grid);
        let start = LatticeState::new(0, 0, 0);
        let goal = LatticeState::new(5, 0, 0);
        let result = planner.plan(start, goal, 50000).unwrap();
        assert!(result.path.len() >= 2);
        // Path should not go through blocked cell
        for s in &result.path {
            assert!(s.gx != 2 || s.gy.abs() > 1);
        }
    }

    #[test]
    fn test_plan_start_blocked() {
        let lib = PrimitiveLibrary::generate(4, 1.0).unwrap();
        let mut grid = OccupancyGrid::new(20, 20, -10, -10);
        grid.set_blocked(0, 0, true);
        let planner = LatticePlanner::new(lib, grid);
        assert!(planner.plan(
            LatticeState::new(0, 0, 0),
            LatticeState::new(5, 0, 0),
            1000,
        ).is_err());
    }

    #[test]
    fn test_cost_weights() {
        let w = LatticeCostWeights::new()
            .with_distance(2.0)
            .with_heading_change(1.0);
        assert!((w.distance - 2.0).abs() < 1e-10);
        assert!((w.heading_change - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_result_display() {
        let r = LatticeResult {
            path: vec![LatticeState::new(0, 0, 0)],
            cost: 3.0,
            expansions: 50,
        };
        let s = format!("{r}");
        assert!(s.contains("expansions=50"));
    }

    #[test]
    fn test_planner_display() {
        let lib = PrimitiveLibrary::generate(4, 1.0).unwrap();
        let grid = OccupancyGrid::new(10, 10, 0, 0);
        let planner = LatticePlanner::new(lib, grid);
        let s = format!("{planner}");
        assert!(s.contains("LatticePlanner"));
    }

    #[test]
    fn test_error_display() {
        let e = LatticeError::NoPathFound;
        assert_eq!(format!("{e}"), "no path found");
    }
}
