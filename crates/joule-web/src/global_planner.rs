//! Global Path Planner — A* search on occupancy costmaps, Dijkstra shortest-path,
//! Theta* any-angle planning, cubic-spline path smoothing, and waypoint extraction.
//!
//! All algorithms operate on a 2-D grid costmap where each cell carries a
//! traversal cost in `[0, 255]` (255 = lethal / impassable). The planner
//! returns an ordered sequence of `(x, y)` waypoints that can be fed to a
//! local trajectory planner or waypoint-following controller.

use std::collections::{BinaryHeap, HashMap};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors produced by global planning algorithms.
#[derive(Debug, Clone, PartialEq)]
pub enum PlannerError {
    /// Start or goal lies outside the costmap bounds.
    OutOfBounds(String),
    /// Start or goal sits on a lethal cell.
    LethalCell(String),
    /// No feasible path exists between start and goal.
    NoPath,
    /// Invalid costmap dimensions.
    InvalidCostmap(String),
    /// Smoothing parameter out of range.
    InvalidParameter(String),
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds(m) => write!(f, "out of bounds: {m}"),
            Self::LethalCell(m) => write!(f, "lethal cell: {m}"),
            Self::NoPath => write!(f, "no feasible path found"),
            Self::InvalidCostmap(m) => write!(f, "invalid costmap: {m}"),
            Self::InvalidParameter(m) => write!(f, "invalid parameter: {m}"),
        }
    }
}

impl std::error::Error for PlannerError {}

// ── Grid Position ───────────────────────────────────────────────

/// A 2-D integer position on the costmap grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos {
    pub x: usize,
    pub y: usize,
}

impl GridPos {
    pub fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

impl fmt::Display for GridPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// ── Costmap ─────────────────────────────────────────────────────

/// Lethal cost threshold — cells at or above this value are impassable.
pub const LETHAL_COST: u8 = 255;

/// 2-D occupancy costmap stored in row-major order.
#[derive(Debug, Clone)]
pub struct Costmap {
    pub width: usize,
    pub height: usize,
    cells: Vec<u8>,
}

impl Costmap {
    /// Create a costmap filled with `default_cost`.
    pub fn new(width: usize, height: usize, default_cost: u8) -> Result<Self, PlannerError> {
        if width == 0 || height == 0 {
            return Err(PlannerError::InvalidCostmap("dimensions must be > 0".into()));
        }
        Ok(Self {
            width,
            height,
            cells: vec![default_cost; width * height],
        })
    }

    /// Build from a pre-existing row-major buffer.
    pub fn from_cells(width: usize, height: usize, cells: Vec<u8>) -> Result<Self, PlannerError> {
        if width == 0 || height == 0 {
            return Err(PlannerError::InvalidCostmap("dimensions must be > 0".into()));
        }
        if cells.len() != width * height {
            return Err(PlannerError::InvalidCostmap(format!(
                "expected {} cells, got {}",
                width * height,
                cells.len()
            )));
        }
        Ok(Self { width, height, cells })
    }

    #[inline]
    pub fn in_bounds(&self, p: GridPos) -> bool {
        p.x < self.width && p.y < self.height
    }

    #[inline]
    pub fn cost(&self, p: GridPos) -> u8 {
        self.cells[p.y * self.width + p.x]
    }

    #[inline]
    pub fn set_cost(&mut self, p: GridPos, c: u8) {
        self.cells[p.y * self.width + p.x] = c;
    }

    /// Return 4- or 8-connected neighbours of `p` that are inside bounds.
    fn neighbours(&self, p: GridPos, eight: bool) -> Vec<(GridPos, f64)> {
        let dirs_4: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let dirs_8: [(i32, i32); 8] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ];
        let dirs: &[(i32, i32)] = if eight { &dirs_8 } else { &dirs_4 };
        let mut result = Vec::new();
        for &(dx, dy) in dirs {
            let nx = p.x as i32 + dx;
            let ny = p.y as i32 + dy;
            if nx >= 0 && ny >= 0 {
                let np = GridPos::new(nx as usize, ny as usize);
                if self.in_bounds(np) && self.cost(np) < LETHAL_COST {
                    let dist = if dx.abs() + dy.abs() == 2 {
                        std::f64::consts::SQRT_2
                    } else {
                        1.0
                    };
                    result.push((np, dist));
                }
            }
        }
        result
    }
}

impl fmt::Display for Costmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Costmap({}x{})", self.width, self.height)
    }
}

// ── A* heap node ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct AStarNode {
    pos: GridPos,
    f: f64,
}

impl Eq for AStarNode {}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f.partial_cmp(&self.f).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ── Heuristic helpers ───────────────────────────────────────────

fn octile_heuristic(a: GridPos, b: GridPos) -> f64 {
    let dx = (a.x as f64 - b.x as f64).abs();
    let dy = (a.y as f64 - b.y as f64).abs();
    let mn = dx.min(dy);
    let mx = dx.max(dy);
    mn * std::f64::consts::SQRT_2 + (mx - mn)
}

fn euclidean_dist(a: GridPos, b: GridPos) -> f64 {
    let dx = a.x as f64 - b.x as f64;
    let dy = a.y as f64 - b.y as f64;
    (dx * dx + dy * dy).sqrt()
}

// ── Global Planner ──────────────────────────────────────────────

/// Algorithm selection for the global planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    AStar,
    Dijkstra,
    ThetaStar,
}

impl fmt::Display for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AStar => write!(f, "A*"),
            Self::Dijkstra => write!(f, "Dijkstra"),
            Self::ThetaStar => write!(f, "Theta*"),
        }
    }
}

/// Global path planner with configurable algorithm and post-processing.
#[derive(Debug, Clone)]
pub struct GlobalPlanner {
    algorithm: Algorithm,
    eight_connected: bool,
    cost_weight: f64,
    smooth_iterations: usize,
    smooth_alpha: f64,
    smooth_beta: f64,
}

impl GlobalPlanner {
    pub fn new(algorithm: Algorithm) -> Self {
        Self {
            algorithm,
            eight_connected: true,
            cost_weight: 1.0,
            smooth_iterations: 100,
            smooth_alpha: 0.5,
            smooth_beta: 0.2,
        }
    }

    pub fn with_connectivity(mut self, eight: bool) -> Self {
        self.eight_connected = eight;
        self
    }

    pub fn with_cost_weight(mut self, w: f64) -> Self {
        self.cost_weight = w.max(0.0);
        self
    }

    pub fn with_smooth_params(mut self, iterations: usize, alpha: f64, beta: f64) -> Self {
        self.smooth_iterations = iterations;
        self.smooth_alpha = alpha.clamp(0.0, 1.0);
        self.smooth_beta = beta.clamp(0.0, 1.0);
        self
    }

    /// Plan a path from `start` to `goal` on the given costmap.
    pub fn plan(
        &self,
        costmap: &Costmap,
        start: GridPos,
        goal: GridPos,
    ) -> Result<Vec<GridPos>, PlannerError> {
        self.validate(costmap, start, goal)?;
        if start == goal {
            return Ok(vec![start]);
        }
        let raw = match self.algorithm {
            Algorithm::AStar => self.astar(costmap, start, goal)?,
            Algorithm::Dijkstra => self.dijkstra(costmap, start, goal)?,
            Algorithm::ThetaStar => self.theta_star(costmap, start, goal)?,
        };
        Ok(raw)
    }

    /// Plan and then smooth the result.
    pub fn plan_smooth(
        &self,
        costmap: &Costmap,
        start: GridPos,
        goal: GridPos,
    ) -> Result<Vec<(f64, f64)>, PlannerError> {
        let raw = self.plan(costmap, start, goal)?;
        let float_path: Vec<(f64, f64)> = raw.iter().map(|p| (p.x as f64, p.y as f64)).collect();
        Ok(self.gradient_smooth(&float_path))
    }

    fn validate(&self, cm: &Costmap, s: GridPos, g: GridPos) -> Result<(), PlannerError> {
        if !cm.in_bounds(s) {
            return Err(PlannerError::OutOfBounds(format!("start {s}")));
        }
        if !cm.in_bounds(g) {
            return Err(PlannerError::OutOfBounds(format!("goal {g}")));
        }
        if cm.cost(s) >= LETHAL_COST {
            return Err(PlannerError::LethalCell(format!("start {s}")));
        }
        if cm.cost(g) >= LETHAL_COST {
            return Err(PlannerError::LethalCell(format!("goal {g}")));
        }
        Ok(())
    }

    // ── A* ──

    fn astar(
        &self,
        costmap: &Costmap,
        start: GridPos,
        goal: GridPos,
    ) -> Result<Vec<GridPos>, PlannerError> {
        let mut open = BinaryHeap::new();
        let mut g_score: HashMap<GridPos, f64> = HashMap::new();
        let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();

        g_score.insert(start, 0.0);
        open.push(AStarNode { pos: start, f: octile_heuristic(start, goal) });

        while let Some(current) = open.pop() {
            if current.pos == goal {
                return Ok(Self::reconstruct(&came_from, goal));
            }
            let cur_g = g_score[&current.pos];
            for (nb, step_dist) in costmap.neighbours(current.pos, self.eight_connected) {
                let cell_cost = costmap.cost(nb) as f64 * self.cost_weight;
                let tentative = cur_g + step_dist + cell_cost / 255.0;
                if tentative < *g_score.get(&nb).unwrap_or(&f64::INFINITY) {
                    g_score.insert(nb, tentative);
                    came_from.insert(nb, current.pos);
                    let f = tentative + octile_heuristic(nb, goal);
                    open.push(AStarNode { pos: nb, f });
                }
            }
        }
        Err(PlannerError::NoPath)
    }

    // ── Dijkstra ──

    fn dijkstra(
        &self,
        costmap: &Costmap,
        start: GridPos,
        goal: GridPos,
    ) -> Result<Vec<GridPos>, PlannerError> {
        let mut open = BinaryHeap::new();
        let mut dist: HashMap<GridPos, f64> = HashMap::new();
        let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();

        dist.insert(start, 0.0);
        open.push(AStarNode { pos: start, f: 0.0 });

        while let Some(current) = open.pop() {
            if current.pos == goal {
                return Ok(Self::reconstruct(&came_from, goal));
            }
            let cur_d = dist[&current.pos];
            if (-current.f) > cur_d {
                continue;
            }
            for (nb, step_dist) in costmap.neighbours(current.pos, self.eight_connected) {
                let cell_cost = costmap.cost(nb) as f64 * self.cost_weight;
                let tentative = cur_d + step_dist + cell_cost / 255.0;
                if tentative < *dist.get(&nb).unwrap_or(&f64::INFINITY) {
                    dist.insert(nb, tentative);
                    came_from.insert(nb, current.pos);
                    open.push(AStarNode { pos: nb, f: tentative });
                }
            }
        }
        Err(PlannerError::NoPath)
    }

    // ── Theta* (any-angle) ──

    fn theta_star(
        &self,
        costmap: &Costmap,
        start: GridPos,
        goal: GridPos,
    ) -> Result<Vec<GridPos>, PlannerError> {
        let mut open = BinaryHeap::new();
        let mut g_score: HashMap<GridPos, f64> = HashMap::new();
        let mut came_from: HashMap<GridPos, GridPos> = HashMap::new();

        g_score.insert(start, 0.0);
        came_from.insert(start, start);
        open.push(AStarNode { pos: start, f: euclidean_dist(start, goal) });

        while let Some(current) = open.pop() {
            if current.pos == goal {
                return Ok(Self::reconstruct(&came_from, goal));
            }
            let cur_g = g_score[&current.pos];
            let parent = came_from[&current.pos];

            for (nb, step_dist) in costmap.neighbours(current.pos, self.eight_connected) {
                let parent_g = g_score[&parent];
                let (best_parent, best_g) =
                    if self.line_of_sight(costmap, parent, nb) {
                        let d = euclidean_dist(parent, nb);
                        let cell_cost = costmap.cost(nb) as f64 * self.cost_weight / 255.0;
                        (parent, parent_g + d + cell_cost)
                    } else {
                        let cell_cost = costmap.cost(nb) as f64 * self.cost_weight / 255.0;
                        (current.pos, cur_g + step_dist + cell_cost)
                    };

                if best_g < *g_score.get(&nb).unwrap_or(&f64::INFINITY) {
                    g_score.insert(nb, best_g);
                    came_from.insert(nb, best_parent);
                    open.push(AStarNode {
                        pos: nb,
                        f: best_g + euclidean_dist(nb, goal),
                    });
                }
            }
        }
        Err(PlannerError::NoPath)
    }

    /// Bresenham line-of-sight check.
    fn line_of_sight(&self, costmap: &Costmap, a: GridPos, b: GridPos) -> bool {
        let mut x0 = a.x as i32;
        let mut y0 = a.y as i32;
        let x1 = b.x as i32;
        let y1 = b.y as i32;
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        loop {
            let p = GridPos::new(x0 as usize, y0 as usize);
            if costmap.cost(p) >= LETHAL_COST {
                return false;
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
        true
    }

    fn reconstruct(came_from: &HashMap<GridPos, GridPos>, goal: GridPos) -> Vec<GridPos> {
        let mut path = vec![goal];
        let mut cur = goal;
        while let Some(&prev) = came_from.get(&cur) {
            if prev == cur {
                break;
            }
            path.push(prev);
            cur = prev;
        }
        path.reverse();
        path
    }

    // ── Gradient-descent path smoothing ──

    fn gradient_smooth(&self, path: &[(f64, f64)]) -> Vec<(f64, f64)> {
        if path.len() <= 2 {
            return path.to_vec();
        }
        let mut smoothed: Vec<(f64, f64)> = path.to_vec();
        let alpha = self.smooth_alpha;
        let beta = self.smooth_beta;
        for _ in 0..self.smooth_iterations {
            for i in 1..smoothed.len() - 1 {
                let ox = path[i].0;
                let oy = path[i].1;
                let sx = smoothed[i].0;
                let sy = smoothed[i].1;
                let prev = smoothed[i - 1];
                let next = smoothed[i + 1];
                smoothed[i].0 += alpha * (ox - sx) + beta * (prev.0 + next.0 - 2.0 * sx);
                smoothed[i].1 += alpha * (oy - sy) + beta * (prev.1 + next.1 - 2.0 * sy);
            }
        }
        smoothed
    }
}

impl fmt::Display for GlobalPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GlobalPlanner(algo={}, 8-conn={}, cost_w={:.2})",
            self.algorithm, self.eight_connected, self.cost_weight
        )
    }
}

// ── Waypoint Extraction ─────────────────────────────────────────

/// Extract a reduced set of waypoints from a dense path by keeping only
/// points where the heading changes by more than `angle_threshold` radians.
pub fn extract_waypoints(path: &[(f64, f64)], angle_threshold: f64) -> Vec<(f64, f64)> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut result = vec![path[0]];
    let mut prev_angle = (path[1].1 - path[0].1).atan2(path[1].0 - path[0].0);
    for i in 2..path.len() {
        let angle = (path[i].1 - path[i - 1].1).atan2(path[i].0 - path[i - 1].0);
        let diff = (angle - prev_angle).abs();
        let diff = if diff > std::f64::consts::PI {
            2.0 * std::f64::consts::PI - diff
        } else {
            diff
        };
        if diff > angle_threshold {
            result.push(path[i - 1]);
            prev_angle = angle;
        }
    }
    result.push(*path.last().unwrap());
    result
}

/// Compute the total Euclidean length of a path.
pub fn path_length(path: &[(f64, f64)]) -> f64 {
    path.windows(2)
        .map(|w| {
            let dx = w[1].0 - w[0].0;
            let dy = w[1].1 - w[0].1;
            (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_costmap(w: usize, h: usize) -> Costmap {
        Costmap::new(w, h, 0).unwrap()
    }

    fn costmap_with_wall(w: usize, h: usize) -> Costmap {
        let mut cm = empty_costmap(w, h);
        for y in 0..h - 1 {
            cm.set_cost(GridPos::new(w / 2, y), LETHAL_COST);
        }
        cm
    }

    #[test]
    fn test_costmap_creation() {
        let cm = Costmap::new(10, 10, 0).unwrap();
        assert_eq!(cm.width, 10);
        assert_eq!(cm.cost(GridPos::new(5, 5)), 0);
    }

    #[test]
    fn test_costmap_zero_dims() {
        assert!(Costmap::new(0, 5, 0).is_err());
    }

    #[test]
    fn test_costmap_from_cells() {
        let cells = vec![0u8; 20];
        let cm = Costmap::from_cells(5, 4, cells).unwrap();
        assert_eq!(cm.height, 4);
    }

    #[test]
    fn test_costmap_from_cells_mismatch() {
        let cells = vec![0u8; 10];
        assert!(Costmap::from_cells(5, 4, cells).is_err());
    }

    #[test]
    fn test_astar_trivial() {
        let cm = empty_costmap(5, 5);
        let planner = GlobalPlanner::new(Algorithm::AStar);
        let path = planner.plan(&cm, GridPos::new(0, 0), GridPos::new(4, 4)).unwrap();
        assert!(!path.is_empty());
        assert_eq!(*path.first().unwrap(), GridPos::new(0, 0));
        assert_eq!(*path.last().unwrap(), GridPos::new(4, 4));
    }

    #[test]
    fn test_astar_same_start_goal() {
        let cm = empty_costmap(5, 5);
        let planner = GlobalPlanner::new(Algorithm::AStar);
        let path = planner.plan(&cm, GridPos::new(2, 2), GridPos::new(2, 2)).unwrap();
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_astar_around_wall() {
        let cm = costmap_with_wall(10, 10);
        let planner = GlobalPlanner::new(Algorithm::AStar);
        let path = planner.plan(&cm, GridPos::new(0, 0), GridPos::new(9, 0)).unwrap();
        assert!(path.len() > 2);
    }

    #[test]
    fn test_dijkstra_trivial() {
        let cm = empty_costmap(5, 5);
        let planner = GlobalPlanner::new(Algorithm::Dijkstra);
        let path = planner.plan(&cm, GridPos::new(0, 0), GridPos::new(4, 0)).unwrap();
        assert_eq!(*path.last().unwrap(), GridPos::new(4, 0));
    }

    #[test]
    fn test_theta_star_trivial() {
        let cm = empty_costmap(8, 8);
        let planner = GlobalPlanner::new(Algorithm::ThetaStar);
        let path = planner.plan(&cm, GridPos::new(0, 0), GridPos::new(7, 7)).unwrap();
        assert!(!path.is_empty());
    }

    #[test]
    fn test_no_path() {
        let mut cm = empty_costmap(5, 5);
        for x in 0..5 {
            cm.set_cost(GridPos::new(x, 2), LETHAL_COST);
        }
        let planner = GlobalPlanner::new(Algorithm::AStar);
        assert_eq!(
            planner.plan(&cm, GridPos::new(0, 0), GridPos::new(0, 4)),
            Err(PlannerError::NoPath)
        );
    }

    #[test]
    fn test_start_on_lethal() {
        let mut cm = empty_costmap(5, 5);
        cm.set_cost(GridPos::new(0, 0), LETHAL_COST);
        let planner = GlobalPlanner::new(Algorithm::AStar);
        assert!(planner.plan(&cm, GridPos::new(0, 0), GridPos::new(4, 4)).is_err());
    }

    #[test]
    fn test_out_of_bounds() {
        let cm = empty_costmap(5, 5);
        let planner = GlobalPlanner::new(Algorithm::AStar);
        assert!(planner.plan(&cm, GridPos::new(10, 10), GridPos::new(0, 0)).is_err());
    }

    #[test]
    fn test_smooth_path() {
        let cm = empty_costmap(10, 10);
        let planner = GlobalPlanner::new(Algorithm::AStar)
            .with_smooth_params(50, 0.5, 0.3);
        let smoothed = planner
            .plan_smooth(&cm, GridPos::new(0, 0), GridPos::new(9, 9))
            .unwrap();
        assert!(!smoothed.is_empty());
    }

    #[test]
    fn test_extract_waypoints_straight() {
        let path: Vec<(f64, f64)> = (0..10).map(|i| (i as f64, 0.0)).collect();
        let wps = extract_waypoints(&path, 0.1);
        assert_eq!(wps.len(), 2); // start + end only
    }

    #[test]
    fn test_extract_waypoints_turn() {
        let mut path: Vec<(f64, f64)> = (0..5).map(|i| (i as f64, 0.0)).collect();
        for i in 0..5 {
            path.push((4.0, (i + 1) as f64));
        }
        let wps = extract_waypoints(&path, 0.1);
        assert!(wps.len() >= 3);
    }

    #[test]
    fn test_path_length() {
        let path = vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        let len = path_length(&path);
        assert!((len - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_four_connected() {
        let cm = empty_costmap(5, 5);
        let planner = GlobalPlanner::new(Algorithm::AStar).with_connectivity(false);
        let path = planner.plan(&cm, GridPos::new(0, 0), GridPos::new(4, 4)).unwrap();
        // 4-connected: manhattan-style, path should be longer
        assert!(path.len() >= 9);
    }

    #[test]
    fn test_cost_weight() {
        let mut cm = empty_costmap(5, 1);
        cm.set_cost(GridPos::new(2, 0), 200);
        let p_low = GlobalPlanner::new(Algorithm::AStar).with_cost_weight(0.0);
        let path = p_low.plan(&cm, GridPos::new(0, 0), GridPos::new(4, 0)).unwrap();
        assert!(path.len() >= 2);
    }

    #[test]
    fn test_display_planner() {
        let p = GlobalPlanner::new(Algorithm::AStar);
        let s = format!("{p}");
        assert!(s.contains("A*"));
    }

    #[test]
    fn test_display_gridpos() {
        let gp = GridPos::new(3, 7);
        assert_eq!(format!("{gp}"), "(3, 7)");
    }
}
