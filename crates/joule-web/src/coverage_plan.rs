//! Coverage planning — boustrophedon decomposition, spanning tree coverage,
//! area partitioning, and coverage metrics for multi-robot systems.
//!
//! Pure-Rust coverage path planners for surveying, search-and-rescue,
//! and environmental monitoring. Includes cell decomposition, graph-based
//! coverage, and quantitative coverage analysis.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Coverage planning errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CoverageError {
    /// Invalid grid dimensions.
    InvalidDimensions(String),
    /// Position out of bounds.
    OutOfBounds { x: usize, y: usize },
    /// No valid start position.
    NoStartPosition,
    /// Coverage planning failed.
    PlanFailed(String),
    /// Partition count invalid.
    InvalidPartitionCount(usize),
}

impl fmt::Display for CoverageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(msg) => write!(f, "invalid dimensions: {msg}"),
            Self::OutOfBounds { x, y } => write!(f, "out of bounds: ({x}, {y})"),
            Self::NoStartPosition => write!(f, "no valid start position"),
            Self::PlanFailed(msg) => write!(f, "plan failed: {msg}"),
            Self::InvalidPartitionCount(n) => write!(f, "invalid partition count: {n}"),
        }
    }
}

impl std::error::Error for CoverageError {}

// ── Cell Types ──────────────────────────────────────────────────

/// Status of a grid cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellStatus {
    /// Free space that needs coverage.
    Free,
    /// Obstacle — cannot be traversed.
    Obstacle,
    /// Already covered.
    Covered,
    /// Assigned to a specific robot.
    Assigned(u64),
}

impl fmt::Display for CellStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Free => write!(f, "."),
            Self::Obstacle => write!(f, "#"),
            Self::Covered => write!(f, "*"),
            Self::Assigned(id) => write!(f, "{}", id % 10),
        }
    }
}

// ── Coverage Grid ───────────────────────────────────────────────

/// A 2D grid for coverage planning.
#[derive(Debug, Clone)]
pub struct CoverageGrid {
    pub width: usize,
    pub height: usize,
    pub cell_size: f64,
    cells: Vec<CellStatus>,
    visit_counts: Vec<u32>,
}

impl CoverageGrid {
    pub fn new(width: usize, height: usize, cell_size: f64) -> Result<Self, CoverageError> {
        if width == 0 || height == 0 {
            return Err(CoverageError::InvalidDimensions("dimensions must be > 0".into()));
        }
        Ok(Self {
            width,
            height,
            cell_size,
            cells: vec![CellStatus::Free; width * height],
            visit_counts: vec![0; width * height],
        })
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    pub fn get(&self, x: usize, y: usize) -> Result<CellStatus, CoverageError> {
        if !self.in_bounds(x, y) {
            return Err(CoverageError::OutOfBounds { x, y });
        }
        Ok(self.cells[self.idx(x, y)])
    }

    pub fn set(&mut self, x: usize, y: usize, status: CellStatus) -> Result<(), CoverageError> {
        if !self.in_bounds(x, y) {
            return Err(CoverageError::OutOfBounds { x, y });
        }
        let i = self.idx(x, y);
        self.cells[i] = status;
        Ok(())
    }

    /// Set a rectangular region as obstacle.
    pub fn add_obstacle_rect(
        &mut self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
    ) {
        for dy in 0..h {
            for dx in 0..w {
                let x = x0 + dx;
                let y = y0 + dy;
                if self.in_bounds(x, y) {
                    let i = self.idx(x, y);
                    self.cells[i] = CellStatus::Obstacle;
                }
            }
        }
    }

    /// Mark a cell as covered and increment visit count.
    pub fn cover(&mut self, x: usize, y: usize) -> Result<(), CoverageError> {
        if !self.in_bounds(x, y) {
            return Err(CoverageError::OutOfBounds { x, y });
        }
        let i = self.idx(x, y);
        if self.cells[i] != CellStatus::Obstacle {
            self.cells[i] = CellStatus::Covered;
            self.visit_counts[i] += 1;
        }
        Ok(())
    }

    /// Total number of free (uncovered) cells.
    pub fn free_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == CellStatus::Free).count()
    }

    /// Total number of covered cells.
    pub fn covered_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == CellStatus::Covered).count()
    }

    /// Total number of obstacle cells.
    pub fn obstacle_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == CellStatus::Obstacle).count()
    }

    /// Total coverable cells (free + covered).
    pub fn coverable_count(&self) -> usize {
        self.cells.iter().filter(|c| **c != CellStatus::Obstacle).count()
    }

    /// Coverage ratio [0, 1].
    pub fn coverage_ratio(&self) -> f64 {
        let coverable = self.coverable_count();
        if coverable == 0 {
            return 1.0;
        }
        self.covered_count() as f64 / coverable as f64
    }

    /// Overlap ratio: average extra visits per covered cell.
    pub fn overlap_ratio(&self) -> f64 {
        let covered = self.covered_count();
        if covered == 0 {
            return 0.0;
        }
        let total_visits: u32 = self.visit_counts.iter().sum();
        total_visits as f64 / covered as f64 - 1.0
    }

    /// Four-connected neighbors that are free or covered (traversable).
    fn traversable_neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        let mut nbrs = Vec::new();
        if x > 0 && self.cells[self.idx(x - 1, y)] != CellStatus::Obstacle {
            nbrs.push((x - 1, y));
        }
        if x + 1 < self.width && self.cells[self.idx(x + 1, y)] != CellStatus::Obstacle {
            nbrs.push((x + 1, y));
        }
        if y > 0 && self.cells[self.idx(x, y - 1)] != CellStatus::Obstacle {
            nbrs.push((x, y - 1));
        }
        if y + 1 < self.height && self.cells[self.idx(x, y + 1)] != CellStatus::Obstacle {
            nbrs.push((x, y + 1));
        }
        nbrs
    }

    /// Four-connected neighbors that are Free (uncovered, non-obstacle).
    fn free_neighbors(&self, x: usize, y: usize) -> Vec<(usize, usize)> {
        self.traversable_neighbors(x, y)
            .into_iter()
            .filter(|&(nx, ny)| self.cells[self.idx(nx, ny)] == CellStatus::Free)
            .collect()
    }
}

impl fmt::Display for CoverageGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CoverageGrid({}x{}, coverage={:.1}%, obstacles={})",
            self.width,
            self.height,
            self.coverage_ratio() * 100.0,
            self.obstacle_count(),
        )
    }
}

// ── Boustrophedon Planner ───────────────────────────────────────

/// Boustrophedon (lawn-mower) coverage path planner.
#[derive(Debug, Clone)]
pub struct BoustrophedonPlanner {
    pub sweep_direction: SweepDirection,
}

/// Direction of the boustrophedon sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepDirection {
    Horizontal,
    Vertical,
}

impl fmt::Display for SweepDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Horizontal => write!(f, "Horizontal"),
            Self::Vertical => write!(f, "Vertical"),
        }
    }
}

impl BoustrophedonPlanner {
    pub fn new(direction: SweepDirection) -> Self {
        Self { sweep_direction: direction }
    }

    /// Generate a boustrophedon (zigzag) path over the grid.
    pub fn plan(&self, grid: &CoverageGrid) -> Vec<(usize, usize)> {
        let mut path = Vec::new();
        match self.sweep_direction {
            SweepDirection::Horizontal => {
                for y in 0..grid.height {
                    if y % 2 == 0 {
                        for x in 0..grid.width {
                            if grid.cells[grid.idx(x, y)] != CellStatus::Obstacle {
                                path.push((x, y));
                            }
                        }
                    } else {
                        for x in (0..grid.width).rev() {
                            if grid.cells[grid.idx(x, y)] != CellStatus::Obstacle {
                                path.push((x, y));
                            }
                        }
                    }
                }
            }
            SweepDirection::Vertical => {
                for x in 0..grid.width {
                    if x % 2 == 0 {
                        for y in 0..grid.height {
                            if grid.cells[grid.idx(x, y)] != CellStatus::Obstacle {
                                path.push((x, y));
                            }
                        }
                    } else {
                        for y in (0..grid.height).rev() {
                            if grid.cells[grid.idx(x, y)] != CellStatus::Obstacle {
                                path.push((x, y));
                            }
                        }
                    }
                }
            }
        }
        path
    }

    /// Execute the plan on a mutable grid, marking cells as covered.
    pub fn execute(&self, grid: &mut CoverageGrid) -> Vec<(usize, usize)> {
        let path = self.plan(grid);
        for &(x, y) in &path {
            let _ = grid.cover(x, y);
        }
        path
    }
}

impl fmt::Display for BoustrophedonPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Boustrophedon({})", self.sweep_direction)
    }
}

// ── Spanning Tree Coverage ──────────────────────────────────────

/// Coverage planner based on spanning tree circumnavigation.
/// Builds a spanning tree over free cells and follows its boundary.
#[derive(Debug, Clone)]
pub struct SpanningTreePlanner;

impl SpanningTreePlanner {
    pub fn new() -> Self {
        Self
    }

    /// Build a spanning tree (DFS) over the grid and return coverage path.
    pub fn plan(
        &self,
        grid: &CoverageGrid,
        start_x: usize,
        start_y: usize,
    ) -> Result<Vec<(usize, usize)>, CoverageError> {
        if !grid.in_bounds(start_x, start_y) {
            return Err(CoverageError::OutOfBounds { x: start_x, y: start_y });
        }
        if grid.cells[grid.idx(start_x, start_y)] == CellStatus::Obstacle {
            return Err(CoverageError::NoStartPosition);
        }

        let mut visited = vec![false; grid.width * grid.height];
        let mut path = Vec::new();
        let mut stack = vec![(start_x, start_y)];

        while let Some((x, y)) = stack.pop() {
            let i = grid.idx(x, y);
            if visited[i] {
                continue;
            }
            visited[i] = true;
            path.push((x, y));

            // Push unvisited free neighbors (reverse order for DFS consistency).
            let mut nbrs = grid.traversable_neighbors(x, y);
            nbrs.reverse();
            for (nx, ny) in nbrs {
                if !visited[grid.idx(nx, ny)] {
                    stack.push((nx, ny));
                }
            }
        }
        Ok(path)
    }

    /// Execute the plan on a mutable grid.
    pub fn execute(
        &self,
        grid: &mut CoverageGrid,
        start_x: usize,
        start_y: usize,
    ) -> Result<Vec<(usize, usize)>, CoverageError> {
        let path = self.plan(grid, start_x, start_y)?;
        for &(x, y) in &path {
            let _ = grid.cover(x, y);
        }
        Ok(path)
    }
}

impl fmt::Display for SpanningTreePlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpanningTreePlanner")
    }
}

// ── Area Partitioning ───────────────────────────────────────────

/// Partition a coverage area among multiple robots.
#[derive(Debug, Clone)]
pub struct AreaPartitioner {
    pub num_partitions: usize,
    pub method: PartitionMethod,
}

/// Partitioning method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionMethod {
    /// Divide into horizontal strips.
    HorizontalStrips,
    /// Divide into vertical strips.
    VerticalStrips,
    /// Grid-based partitioning (rectangular blocks).
    GridBlocks,
}

impl fmt::Display for PartitionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HorizontalStrips => write!(f, "HorizontalStrips"),
            Self::VerticalStrips => write!(f, "VerticalStrips"),
            Self::GridBlocks => write!(f, "GridBlocks"),
        }
    }
}

impl AreaPartitioner {
    pub fn new(num_partitions: usize, method: PartitionMethod) -> Result<Self, CoverageError> {
        if num_partitions == 0 {
            return Err(CoverageError::InvalidPartitionCount(0));
        }
        Ok(Self { num_partitions, method })
    }

    /// Partition the grid, returning a map from robot_id to list of cells.
    pub fn partition(
        &self,
        grid: &CoverageGrid,
        robot_ids: &[u64],
    ) -> Result<HashMap<u64, Vec<(usize, usize)>>, CoverageError> {
        if robot_ids.len() != self.num_partitions {
            return Err(CoverageError::InvalidPartitionCount(robot_ids.len()));
        }
        let mut result: HashMap<u64, Vec<(usize, usize)>> = HashMap::new();
        for &id in robot_ids {
            result.insert(id, Vec::new());
        }

        match self.method {
            PartitionMethod::HorizontalStrips => {
                let strip_h = grid.height / self.num_partitions;
                for y in 0..grid.height {
                    let mut part_idx = y / strip_h.max(1);
                    if part_idx >= self.num_partitions {
                        part_idx = self.num_partitions - 1;
                    }
                    let robot_id = robot_ids[part_idx];
                    for x in 0..grid.width {
                        if grid.cells[grid.idx(x, y)] != CellStatus::Obstacle {
                            result.get_mut(&robot_id).unwrap().push((x, y));
                        }
                    }
                }
            }
            PartitionMethod::VerticalStrips => {
                let strip_w = grid.width / self.num_partitions;
                for x in 0..grid.width {
                    let mut part_idx = x / strip_w.max(1);
                    if part_idx >= self.num_partitions {
                        part_idx = self.num_partitions - 1;
                    }
                    let robot_id = robot_ids[part_idx];
                    for y in 0..grid.height {
                        if grid.cells[grid.idx(x, y)] != CellStatus::Obstacle {
                            result.get_mut(&robot_id).unwrap().push((x, y));
                        }
                    }
                }
            }
            PartitionMethod::GridBlocks => {
                let cols = (self.num_partitions as f64).sqrt().ceil() as usize;
                let rows = (self.num_partitions + cols - 1) / cols;
                let block_w = grid.width / cols.max(1);
                let block_h = grid.height / rows.max(1);
                for y in 0..grid.height {
                    for x in 0..grid.width {
                        if grid.cells[grid.idx(x, y)] == CellStatus::Obstacle {
                            continue;
                        }
                        let bx = x / block_w.max(1);
                        let by = y / block_h.max(1);
                        let mut part_idx = by * cols + bx;
                        if part_idx >= self.num_partitions {
                            part_idx = self.num_partitions - 1;
                        }
                        let robot_id = robot_ids[part_idx];
                        result.get_mut(&robot_id).unwrap().push((x, y));
                    }
                }
            }
        }
        Ok(result)
    }

    /// Partition and assign cells in the grid.
    pub fn partition_and_assign(
        &self,
        grid: &mut CoverageGrid,
        robot_ids: &[u64],
    ) -> Result<HashMap<u64, Vec<(usize, usize)>>, CoverageError> {
        let partitions = self.partition(grid, robot_ids)?;
        for (&robot_id, cells) in &partitions {
            for &(x, y) in cells {
                let _ = grid.set(x, y, CellStatus::Assigned(robot_id));
            }
        }
        Ok(partitions)
    }
}

impl fmt::Display for AreaPartitioner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AreaPartitioner({}, n={})", self.method, self.num_partitions)
    }
}

// ── Coverage Metrics ────────────────────────────────────────────

/// Coverage quality metrics for analysis.
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    pub coverage_ratio: f64,
    pub overlap_ratio: f64,
    pub path_length: usize,
    pub efficiency: f64,
    pub coverable_cells: usize,
    pub covered_cells: usize,
}

impl CoverageMetrics {
    /// Compute metrics from a grid and path.
    pub fn compute(grid: &CoverageGrid, path_length: usize) -> Self {
        let coverable = grid.coverable_count();
        let covered = grid.covered_count();
        let ratio = grid.coverage_ratio();
        let overlap = grid.overlap_ratio();
        let efficiency = if path_length == 0 {
            0.0
        } else {
            covered as f64 / path_length as f64
        };
        Self {
            coverage_ratio: ratio,
            overlap_ratio: overlap,
            path_length,
            efficiency,
            coverable_cells: coverable,
            covered_cells: covered,
        }
    }
}

impl fmt::Display for CoverageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Metrics(cov={:.1}%, overlap={:.2}, eff={:.3}, path={})",
            self.coverage_ratio * 100.0,
            self.overlap_ratio,
            self.efficiency,
            self.path_length,
        )
    }
}

// ── Greedy Nearest-Neighbor Planner ─────────────────────────────

/// A simple greedy coverage planner: always move to the nearest uncovered cell.
#[derive(Debug, Clone)]
pub struct GreedyPlanner;

impl GreedyPlanner {
    pub fn new() -> Self {
        Self
    }

    pub fn plan(
        &self,
        grid: &CoverageGrid,
        start_x: usize,
        start_y: usize,
    ) -> Result<Vec<(usize, usize)>, CoverageError> {
        if !grid.in_bounds(start_x, start_y) {
            return Err(CoverageError::OutOfBounds { x: start_x, y: start_y });
        }
        // Work on a local copy of cell statuses.
        let mut local_cells = grid.cells.clone();
        let mut path = Vec::new();
        let mut cx = start_x;
        let mut cy = start_y;

        loop {
            let i = grid.idx(cx, cy);
            if local_cells[i] == CellStatus::Free {
                local_cells[i] = CellStatus::Covered;
                path.push((cx, cy));
            }

            // Find nearest free neighbor via BFS.
            let target = self.bfs_nearest_free(grid, &local_cells, cx, cy);
            match target {
                Some((tx, ty)) => {
                    // Move there (in a straight line on grid, simplified).
                    cx = tx;
                    cy = ty;
                }
                None => break,
            }
        }
        Ok(path)
    }

    fn bfs_nearest_free(
        &self,
        grid: &CoverageGrid,
        cells: &[CellStatus],
        sx: usize,
        sy: usize,
    ) -> Option<(usize, usize)> {
        let mut visited = vec![false; grid.width * grid.height];
        let start_i = grid.idx(sx, sy);
        visited[start_i] = true;
        let mut queue = vec![(sx, sy)];
        let mut qi = 0;
        while qi < queue.len() {
            let (x, y) = queue[qi];
            qi += 1;
            // Check neighbors.
            let nbrs = grid.traversable_neighbors(x, y);
            for (nx, ny) in nbrs {
                let ni = grid.idx(nx, ny);
                if visited[ni] {
                    continue;
                }
                visited[ni] = true;
                if cells[ni] == CellStatus::Free {
                    return Some((nx, ny));
                }
                queue.push((nx, ny));
            }
        }
        None
    }

    /// Execute on a mutable grid.
    pub fn execute(
        &self,
        grid: &mut CoverageGrid,
        start_x: usize,
        start_y: usize,
    ) -> Result<Vec<(usize, usize)>, CoverageError> {
        let path = self.plan(grid, start_x, start_y)?;
        for &(x, y) in &path {
            let _ = grid.cover(x, y);
        }
        Ok(path)
    }
}

impl fmt::Display for GreedyPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GreedyPlanner")
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let g = CoverageGrid::new(10, 10, 1.0).unwrap();
        assert_eq!(g.free_count(), 100);
        assert_eq!(g.covered_count(), 0);
    }

    #[test]
    fn test_grid_zero_dims() {
        assert!(CoverageGrid::new(0, 10, 1.0).is_err());
    }

    #[test]
    fn test_grid_obstacle() {
        let mut g = CoverageGrid::new(10, 10, 1.0).unwrap();
        g.add_obstacle_rect(2, 2, 3, 3);
        assert_eq!(g.obstacle_count(), 9);
        assert_eq!(g.free_count(), 91);
    }

    #[test]
    fn test_grid_cover() {
        let mut g = CoverageGrid::new(5, 5, 1.0).unwrap();
        g.cover(0, 0).unwrap();
        assert_eq!(g.covered_count(), 1);
        assert!((g.coverage_ratio() - 1.0 / 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_grid_overlap() {
        let mut g = CoverageGrid::new(5, 5, 1.0).unwrap();
        g.cover(0, 0).unwrap();
        g.cover(0, 0).unwrap(); // Visit again.
        assert!((g.overlap_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_boustrophedon_horizontal() {
        let g = CoverageGrid::new(5, 3, 1.0).unwrap();
        let planner = BoustrophedonPlanner::new(SweepDirection::Horizontal);
        let path = planner.plan(&g);
        assert_eq!(path.len(), 15); // 5*3 = 15 cells.
        // First row left to right.
        assert_eq!(path[0], (0, 0));
        assert_eq!(path[4], (4, 0));
        // Second row right to left.
        assert_eq!(path[5], (4, 1));
    }

    #[test]
    fn test_boustrophedon_execute() {
        let mut g = CoverageGrid::new(4, 4, 1.0).unwrap();
        let planner = BoustrophedonPlanner::new(SweepDirection::Vertical);
        let path = planner.execute(&mut g);
        assert!((g.coverage_ratio() - 1.0).abs() < 1e-9);
        assert_eq!(path.len(), 16);
    }

    #[test]
    fn test_boustrophedon_with_obstacles() {
        let mut g = CoverageGrid::new(5, 5, 1.0).unwrap();
        g.add_obstacle_rect(2, 2, 1, 1);
        let planner = BoustrophedonPlanner::new(SweepDirection::Horizontal);
        let path = planner.execute(&mut g);
        assert_eq!(path.len(), 24); // 25 - 1 obstacle.
        assert!((g.coverage_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_spanning_tree_coverage() {
        let g = CoverageGrid::new(5, 5, 1.0).unwrap();
        let planner = SpanningTreePlanner::new();
        let path = planner.plan(&g, 0, 0).unwrap();
        assert_eq!(path.len(), 25);
    }

    #[test]
    fn test_spanning_tree_with_obstacles() {
        let mut g = CoverageGrid::new(5, 5, 1.0).unwrap();
        g.add_obstacle_rect(2, 0, 1, 5); // Vertical wall.
        let planner = SpanningTreePlanner::new();
        let path = planner.plan(&g, 0, 0).unwrap();
        // Can only reach left side (10 cells).
        assert_eq!(path.len(), 10);
    }

    #[test]
    fn test_spanning_tree_obstacle_start() {
        let mut g = CoverageGrid::new(5, 5, 1.0).unwrap();
        g.set(0, 0, CellStatus::Obstacle).unwrap();
        let planner = SpanningTreePlanner::new();
        assert!(planner.plan(&g, 0, 0).is_err());
    }

    #[test]
    fn test_area_partition_horizontal() {
        let g = CoverageGrid::new(10, 10, 1.0).unwrap();
        let partitioner = AreaPartitioner::new(2, PartitionMethod::HorizontalStrips).unwrap();
        let parts = partitioner.partition(&g, &[1, 2]).unwrap();
        assert_eq!(parts.len(), 2);
        let total: usize = parts.values().map(|v| v.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_area_partition_vertical() {
        let g = CoverageGrid::new(10, 10, 1.0).unwrap();
        let partitioner = AreaPartitioner::new(2, PartitionMethod::VerticalStrips).unwrap();
        let parts = partitioner.partition(&g, &[1, 2]).unwrap();
        let total: usize = parts.values().map(|v| v.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_area_partition_grid() {
        let g = CoverageGrid::new(10, 10, 1.0).unwrap();
        let partitioner = AreaPartitioner::new(4, PartitionMethod::GridBlocks).unwrap();
        let parts = partitioner.partition(&g, &[1, 2, 3, 4]).unwrap();
        let total: usize = parts.values().map(|v| v.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_partition_count_zero() {
        assert!(AreaPartitioner::new(0, PartitionMethod::HorizontalStrips).is_err());
    }

    #[test]
    fn test_coverage_metrics() {
        let mut g = CoverageGrid::new(10, 10, 1.0).unwrap();
        let planner = BoustrophedonPlanner::new(SweepDirection::Horizontal);
        let path = planner.execute(&mut g);
        let metrics = CoverageMetrics::compute(&g, path.len());
        assert!((metrics.coverage_ratio - 1.0).abs() < 1e-9);
        assert!(metrics.efficiency > 0.5);
    }

    #[test]
    fn test_greedy_planner() {
        let g = CoverageGrid::new(5, 5, 1.0).unwrap();
        let planner = GreedyPlanner::new();
        let path = planner.plan(&g, 0, 0).unwrap();
        assert_eq!(path.len(), 25);
    }

    #[test]
    fn test_greedy_planner_obstacles() {
        let mut g = CoverageGrid::new(5, 5, 1.0).unwrap();
        g.add_obstacle_rect(2, 0, 1, 5);
        let planner = GreedyPlanner::new();
        let path = planner.plan(&g, 0, 0).unwrap();
        assert_eq!(path.len(), 10); // Only left side reachable.
    }

    #[test]
    fn test_display_impls() {
        let g = CoverageGrid::new(5, 5, 1.0).unwrap();
        assert!(format!("{g}").contains("5x5"));
        let b = BoustrophedonPlanner::new(SweepDirection::Horizontal);
        assert!(format!("{b}").contains("Horizontal"));
        let st = SpanningTreePlanner::new();
        assert!(format!("{st}").contains("SpanningTree"));
        let gp = GreedyPlanner::new();
        assert!(format!("{gp}").contains("Greedy"));
        let ap = AreaPartitioner::new(3, PartitionMethod::GridBlocks).unwrap();
        assert!(format!("{ap}").contains("3"));
        let metrics = CoverageMetrics::compute(&g, 10);
        assert!(format!("{metrics}").contains("cov="));
    }
}
