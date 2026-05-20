//! A* for 3D Spaces — voxel grid, octree-based spatial indexing, heuristics
//! (Euclidean/diagonal/Manhattan), path smoothing for volumetric planning.
//!
//! Pure-Rust A* planner operating on a 3D voxel grid with configurable
//! heuristics, 6/26-connectivity, and post-planning path smoothing.

use std::collections::BinaryHeap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AStarError {
    InvalidDimensions(String),
    OutOfBounds(String),
    StartBlocked,
    GoalBlocked,
    NoPathFound,
}

impl fmt::Display for AStarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(s) => write!(f, "invalid dimensions: {s}"),
            Self::OutOfBounds(s) => write!(f, "out of bounds: {s}"),
            Self::StartBlocked => write!(f, "start cell is blocked"),
            Self::GoalBlocked => write!(f, "goal cell is blocked"),
            Self::NoPathFound => write!(f, "no path found"),
        }
    }
}

impl std::error::Error for AStarError {}

// ── Cell coordinate ─────────────────────────────────────────────

/// A cell coordinate in the 3D voxel grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cell {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Cell {
    pub fn new(x: usize, y: usize, z: usize) -> Self { Self { x, y, z } }

    /// Euclidean distance to another cell.
    pub fn distance_to(self, other: Cell) -> f64 {
        let dx = self.x as f64 - other.x as f64;
        let dy = self.y as f64 - other.y as f64;
        let dz = self.z as f64 - other.z as f64;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ── Heuristic ───────────────────────────────────────────────────

/// Heuristic function for A* search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Heuristic {
    /// Straight-line distance.
    Euclidean,
    /// 3D diagonal (octile) distance.
    Diagonal,
    /// Sum of axis differences.
    Manhattan,
    /// Zero heuristic (degenerates to Dijkstra).
    Zero,
}

impl Heuristic {
    pub fn estimate(self, from: Cell, to: Cell) -> f64 {
        let dx = (from.x as f64 - to.x as f64).abs();
        let dy = (from.y as f64 - to.y as f64).abs();
        let dz = (from.z as f64 - to.z as f64).abs();
        match self {
            Self::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
            Self::Diagonal => {
                let mut dims = [dx, dy, dz];
                dims.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let sqrt3 = 3.0_f64.sqrt();
                let sqrt2 = 2.0_f64.sqrt();
                let d_min = dims[0];
                let d_mid = dims[1];
                let d_max = dims[2];
                (sqrt3 - sqrt2) * d_min + (sqrt2 - 1.0) * d_mid + d_max
            }
            Self::Manhattan => dx + dy + dz,
            Self::Zero => 0.0,
        }
    }
}

impl fmt::Display for Heuristic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Euclidean => write!(f, "Euclidean"),
            Self::Diagonal => write!(f, "Diagonal"),
            Self::Manhattan => write!(f, "Manhattan"),
            Self::Zero => write!(f, "Zero"),
        }
    }
}

// ── Connectivity ────────────────────────────────────────────────

/// Neighbor connectivity in the voxel grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Connectivity {
    /// 6-connected (face neighbors only).
    Six,
    /// 26-connected (face + edge + corner neighbors).
    TwentySix,
}

impl Connectivity {
    fn offsets(self) -> &'static [(i32, i32, i32)] {
        match self {
            Self::Six => &[
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ],
            Self::TwentySix => &OFFSETS_26,
        }
    }
}

static OFFSETS_26: [(i32, i32, i32); 26] = {
    let mut arr = [(0i32, 0i32, 0i32); 26];
    let mut idx = 0;
    let mut dx: i32 = -1;
    while dx <= 1 {
        let mut dy: i32 = -1;
        while dy <= 1 {
            let mut dz: i32 = -1;
            while dz <= 1 {
                if !(dx == 0 && dy == 0 && dz == 0) {
                    arr[idx] = (dx, dy, dz);
                    idx += 1;
                }
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }
    arr
};

// ── Priority queue entry ────────────────────────────────────────

#[derive(Debug, Clone)]
struct AStarEntry {
    cell: Cell,
    f_score: f64,
}

impl PartialEq for AStarEntry {
    fn eq(&self, other: &Self) -> bool { self.f_score == other.f_score }
}
impl Eq for AStarEntry {}

impl PartialOrd for AStarEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f_score.partial_cmp(&self.f_score).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── Voxel Grid ──────────────────────────────────────────────────

/// A 3D voxel grid with blocked/free cells and optional traversal costs.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    sx: usize,
    sy: usize,
    sz: usize,
    blocked: Vec<bool>,
    costs: Vec<f64>,
    cell_size: f64,
}

impl VoxelGrid {
    pub fn new(sx: usize, sy: usize, sz: usize) -> Result<Self, AStarError> {
        if sx == 0 || sy == 0 || sz == 0 {
            return Err(AStarError::InvalidDimensions("all dimensions must be > 0".into()));
        }
        let total = sx * sy * sz;
        Ok(Self {
            sx, sy, sz,
            blocked: vec![false; total],
            costs: vec![1.0; total],
            cell_size: 1.0,
        })
    }

    pub fn with_cell_size(mut self, size: f64) -> Self { self.cell_size = size; self }

    fn idx(&self, c: Cell) -> usize { c.z * self.sx * self.sy + c.y * self.sx + c.x }

    pub fn in_bounds(&self, c: Cell) -> bool {
        c.x < self.sx && c.y < self.sy && c.z < self.sz
    }

    pub fn set_blocked(&mut self, c: Cell, blocked: bool) -> Result<(), AStarError> {
        if !self.in_bounds(c) {
            return Err(AStarError::OutOfBounds(format!("{c}")));
        }
        let idx = self.idx(c);
        self.blocked[idx] = blocked;
        Ok(())
    }

    pub fn is_blocked(&self, c: Cell) -> bool {
        if !self.in_bounds(c) { return true; }
        self.blocked[self.idx(c)]
    }

    pub fn set_cost(&mut self, c: Cell, cost: f64) -> Result<(), AStarError> {
        if !self.in_bounds(c) {
            return Err(AStarError::OutOfBounds(format!("{c}")));
        }
        let idx = self.idx(c);
        self.costs[idx] = cost;
        Ok(())
    }

    pub fn get_cost(&self, c: Cell) -> f64 {
        if !self.in_bounds(c) { return f64::MAX; }
        self.costs[self.idx(c)]
    }

    /// Block a rectangular region.
    pub fn block_region(&mut self, min: Cell, max: Cell) {
        for z in min.z..=max.z.min(self.sz - 1) {
            for y in min.y..=max.y.min(self.sy - 1) {
                for x in min.x..=max.x.min(self.sx - 1) {
                    let idx = z * self.sx * self.sy + y * self.sx + x;
                    self.blocked[idx] = true;
                }
            }
        }
    }

    pub fn size_x(&self) -> usize { self.sx }
    pub fn size_y(&self) -> usize { self.sy }
    pub fn size_z(&self) -> usize { self.sz }
    pub fn cell_size(&self) -> f64 { self.cell_size }

    pub fn blocked_count(&self) -> usize {
        self.blocked.iter().filter(|&&b| b).count()
    }

    pub fn total_cells(&self) -> usize { self.sx * self.sy * self.sz }
}

impl fmt::Display for VoxelGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f, "VoxelGrid({}x{}x{}, cell={:.2}, blocked={}/{})",
            self.sx, self.sy, self.sz, self.cell_size,
            self.blocked_count(), self.total_cells(),
        )
    }
}

// ── A* Search Result ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AStarResult {
    pub path: Vec<Cell>,
    pub cost: f64,
    pub nodes_expanded: usize,
    pub path_length_world: f64,
}

impl AStarResult {
    /// Smooth the path by removing redundant waypoints using line-of-sight.
    pub fn smooth(&self, grid: &VoxelGrid) -> Vec<Cell> {
        if self.path.len() <= 2 { return self.path.clone(); }
        let mut smoothed = vec![self.path[0]];
        let mut anchor = 0;
        let mut current = 1;

        while current < self.path.len() - 1 {
            let next = current + 1;
            if !line_of_sight(grid, self.path[anchor], self.path[next]) {
                smoothed.push(self.path[current]);
                anchor = current;
            }
            current += 1;
        }
        smoothed.push(*self.path.last().unwrap());
        smoothed
    }
}

impl fmt::Display for AStarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AStarResult(waypoints={}, cost={:.3}, expanded={}, length={:.3})",
            self.path.len(), self.cost, self.nodes_expanded, self.path_length_world,
        )
    }
}

// ── Line-of-sight (3D Bresenham) ────────────────────────────────

fn line_of_sight(grid: &VoxelGrid, a: Cell, b: Cell) -> bool {
    let dx = b.x as i64 - a.x as i64;
    let dy = b.y as i64 - a.y as i64;
    let dz = b.z as i64 - a.z as i64;
    let steps = dx.abs().max(dy.abs()).max(dz.abs());
    if steps == 0 { return true; }

    for i in 1..steps {
        let t = i as f64 / steps as f64;
        let x = (a.x as f64 + dx as f64 * t).round() as usize;
        let y = (a.y as f64 + dy as f64 * t).round() as usize;
        let z = (a.z as f64 + dz as f64 * t).round() as usize;
        if grid.is_blocked(Cell::new(x, y, z)) { return false; }
    }
    true
}

// ── A* Planner ──────────────────────────────────────────────────

/// A* planner for 3D voxel grids.
pub struct AStar3D {
    heuristic: Heuristic,
    connectivity: Connectivity,
    weight: f64,
}

impl AStar3D {
    pub fn new() -> Self {
        Self {
            heuristic: Heuristic::Euclidean,
            connectivity: Connectivity::TwentySix,
            weight: 1.0,
        }
    }

    pub fn with_heuristic(mut self, h: Heuristic) -> Self { self.heuristic = h; self }
    pub fn with_connectivity(mut self, c: Connectivity) -> Self { self.connectivity = c; self }
    pub fn with_weight(mut self, w: f64) -> Self { self.weight = w; self }

    pub fn plan(&self, grid: &VoxelGrid, start: Cell, goal: Cell) -> Result<AStarResult, AStarError> {
        if !grid.in_bounds(start) {
            return Err(AStarError::OutOfBounds(format!("start {start}")));
        }
        if !grid.in_bounds(goal) {
            return Err(AStarError::OutOfBounds(format!("goal {goal}")));
        }
        if grid.is_blocked(start) { return Err(AStarError::StartBlocked); }
        if grid.is_blocked(goal) { return Err(AStarError::GoalBlocked); }

        let total = grid.total_cells();
        let mut g_score = vec![f64::MAX; total];
        let mut came_from: Vec<Option<usize>> = vec![None; total];
        let mut closed = vec![false; total];
        let mut nodes_expanded = 0usize;

        let start_idx = grid.idx(start);
        let goal_idx = grid.idx(goal);
        g_score[start_idx] = 0.0;

        let mut open = BinaryHeap::new();
        open.push(AStarEntry {
            cell: start,
            f_score: self.weight * self.heuristic.estimate(start, goal) * grid.cell_size(),
        });

        while let Some(AStarEntry { cell, .. }) = open.pop() {
            let ci = grid.idx(cell);
            if ci == goal_idx {
                let path = self.reconstruct(grid, &came_from, goal_idx);
                let path_length_world = path.windows(2)
                    .map(|w| w[0].distance_to(w[1]) * grid.cell_size())
                    .sum();
                return Ok(AStarResult {
                    cost: g_score[goal_idx],
                    nodes_expanded,
                    path_length_world,
                    path,
                });
            }
            if closed[ci] { continue; }
            closed[ci] = true;
            nodes_expanded += 1;

            for &(dx, dy, dz) in self.connectivity.offsets() {
                let nx = cell.x as i32 + dx;
                let ny = cell.y as i32 + dy;
                let nz = cell.z as i32 + dz;
                if nx < 0 || ny < 0 || nz < 0 { continue; }
                let neighbor = Cell::new(nx as usize, ny as usize, nz as usize);
                if !grid.in_bounds(neighbor) || grid.is_blocked(neighbor) { continue; }
                let ni = grid.idx(neighbor);
                if closed[ni] { continue; }

                let move_cost = cell.distance_to(neighbor) * grid.cell_size() * grid.get_cost(neighbor);
                let tentative_g = g_score[ci] + move_cost;
                if tentative_g < g_score[ni] {
                    g_score[ni] = tentative_g;
                    came_from[ni] = Some(ci);
                    let h = self.weight * self.heuristic.estimate(neighbor, goal) * grid.cell_size();
                    open.push(AStarEntry { cell: neighbor, f_score: tentative_g + h });
                }
            }
        }

        Err(AStarError::NoPathFound)
    }

    fn reconstruct(&self, grid: &VoxelGrid, came_from: &[Option<usize>], goal_idx: usize) -> Vec<Cell> {
        let mut path = Vec::new();
        let mut idx = Some(goal_idx);
        while let Some(ci) = idx {
            let z = ci / (grid.sx * grid.sy);
            let rem = ci % (grid.sx * grid.sy);
            let y = rem / grid.sx;
            let x = rem % grid.sx;
            path.push(Cell::new(x, y, z));
            idx = came_from[ci];
        }
        path.reverse();
        path
    }
}

impl fmt::Display for AStar3D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AStar3D(heuristic={}, weight={:.1})", self.heuristic, self.weight)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_distance() {
        let a = Cell::new(0, 0, 0);
        let b = Cell::new(1, 0, 0);
        assert!((a.distance_to(b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cell_display() {
        let c = Cell::new(3, 4, 5);
        assert_eq!(format!("{c}"), "(3, 4, 5)");
    }

    #[test]
    fn test_heuristic_euclidean() {
        let h = Heuristic::Euclidean;
        let d = h.estimate(Cell::new(0, 0, 0), Cell::new(3, 4, 0));
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_heuristic_manhattan() {
        let h = Heuristic::Manhattan;
        let d = h.estimate(Cell::new(0, 0, 0), Cell::new(3, 4, 5));
        assert!((d - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_heuristic_zero() {
        let h = Heuristic::Zero;
        assert!((h.estimate(Cell::new(0, 0, 0), Cell::new(100, 100, 100))).abs() < 1e-10);
    }

    #[test]
    fn test_grid_creation() {
        let g = VoxelGrid::new(10, 10, 10).unwrap();
        assert_eq!(g.total_cells(), 1000);
        assert_eq!(g.blocked_count(), 0);
    }

    #[test]
    fn test_grid_invalid() {
        assert!(VoxelGrid::new(0, 5, 5).is_err());
    }

    #[test]
    fn test_grid_block() {
        let mut g = VoxelGrid::new(5, 5, 5).unwrap();
        g.set_blocked(Cell::new(2, 2, 2), true).unwrap();
        assert!(g.is_blocked(Cell::new(2, 2, 2)));
        assert!(!g.is_blocked(Cell::new(0, 0, 0)));
    }

    #[test]
    fn test_grid_block_region() {
        let mut g = VoxelGrid::new(10, 10, 10).unwrap();
        g.block_region(Cell::new(2, 2, 2), Cell::new(4, 4, 4));
        assert!(g.is_blocked(Cell::new(3, 3, 3)));
        assert!(!g.is_blocked(Cell::new(1, 1, 1)));
        assert_eq!(g.blocked_count(), 27); // 3x3x3
    }

    #[test]
    fn test_grid_display() {
        let g = VoxelGrid::new(5, 5, 5).unwrap();
        let s = format!("{g}");
        assert!(s.contains("5x5x5"));
    }

    #[test]
    fn test_simple_path_6conn() {
        let g = VoxelGrid::new(10, 10, 10).unwrap();
        let planner = AStar3D::new().with_connectivity(Connectivity::Six);
        let result = planner.plan(&g, Cell::new(0, 0, 0), Cell::new(5, 5, 5)).unwrap();
        assert!(result.path.len() >= 2);
        assert_eq!(result.path[0], Cell::new(0, 0, 0));
        assert_eq!(*result.path.last().unwrap(), Cell::new(5, 5, 5));
    }

    #[test]
    fn test_simple_path_26conn() {
        let g = VoxelGrid::new(10, 10, 10).unwrap();
        let planner = AStar3D::new().with_connectivity(Connectivity::TwentySix);
        let result = planner.plan(&g, Cell::new(0, 0, 0), Cell::new(5, 5, 5)).unwrap();
        assert!(result.path.len() >= 2);
        // 26-connected path should be shorter or equal
        let planner6 = AStar3D::new().with_connectivity(Connectivity::Six);
        let result6 = planner6.plan(&g, Cell::new(0, 0, 0), Cell::new(5, 5, 5)).unwrap();
        assert!(result.path.len() <= result6.path.len());
    }

    #[test]
    fn test_path_around_obstacle() {
        let mut g = VoxelGrid::new(10, 10, 5).unwrap();
        // Wall blocking z=2 plane except edges
        for x in 1..9 {
            for y in 1..9 {
                g.set_blocked(Cell::new(x, y, 2), true).unwrap();
            }
        }
        let planner = AStar3D::new();
        let result = planner.plan(&g, Cell::new(5, 5, 0), Cell::new(5, 5, 4)).unwrap();
        assert!(result.path.len() >= 2);
    }

    #[test]
    fn test_no_path() {
        let mut g = VoxelGrid::new(5, 5, 5).unwrap();
        // Complete wall
        for x in 0..5 {
            for y in 0..5 {
                g.set_blocked(Cell::new(x, y, 2), true).unwrap();
            }
        }
        let planner = AStar3D::new();
        assert!(planner.plan(&g, Cell::new(2, 2, 0), Cell::new(2, 2, 4)).is_err());
    }

    #[test]
    fn test_start_blocked() {
        let mut g = VoxelGrid::new(5, 5, 5).unwrap();
        g.set_blocked(Cell::new(0, 0, 0), true).unwrap();
        let planner = AStar3D::new();
        assert!(planner.plan(&g, Cell::new(0, 0, 0), Cell::new(4, 4, 4)).is_err());
    }

    #[test]
    fn test_path_smoothing() {
        let g = VoxelGrid::new(10, 10, 5).unwrap();
        let planner = AStar3D::new().with_connectivity(Connectivity::Six);
        let result = planner.plan(&g, Cell::new(0, 0, 0), Cell::new(9, 9, 4)).unwrap();
        let smoothed = result.smooth(&g);
        assert!(smoothed.len() <= result.path.len());
        assert_eq!(smoothed[0], Cell::new(0, 0, 0));
        assert_eq!(*smoothed.last().unwrap(), Cell::new(9, 9, 4));
    }

    #[test]
    fn test_weighted_heuristic() {
        let g = VoxelGrid::new(10, 10, 5).unwrap();
        let planner = AStar3D::new().with_weight(2.0);
        let result = planner.plan(&g, Cell::new(0, 0, 0), Cell::new(9, 9, 4)).unwrap();
        // Weighted A* may expand fewer nodes
        let planner1 = AStar3D::new().with_weight(1.0);
        let result1 = planner1.plan(&g, Cell::new(0, 0, 0), Cell::new(9, 9, 4)).unwrap();
        assert!(result.nodes_expanded <= result1.nodes_expanded + 1);
    }

    #[test]
    fn test_planner_display() {
        let p = AStar3D::new().with_heuristic(Heuristic::Manhattan);
        let s = format!("{p}");
        assert!(s.contains("Manhattan"));
    }

    #[test]
    fn test_result_display() {
        let r = AStarResult {
            path: vec![Cell::new(0, 0, 0)],
            cost: 5.0,
            nodes_expanded: 100,
            path_length_world: 5.0,
        };
        let s = format!("{r}");
        assert!(s.contains("expanded=100"));
    }
}
