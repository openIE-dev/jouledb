//! Flow field pathfinding for large groups — integration field (cost-to-goal
//! via Dijkstra), flow field (8-direction quantization), multi-goal, LOS
//! optimization, sector-based for large maps, field caching, smooth vector
//! lookup with bilinear interpolation.
//!
//! Replaces JavaScript flow field libraries with a pure-Rust implementation
//! for RTS-style group pathfinding.

use std::collections::VecDeque;

// ── Direction ───────────────────────────────────────────────────

/// Quantized flow direction (8 directions + none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlowDir {
    N,
    NE,
    E,
    SE,
    S,
    SW,
    W,
    NW,
    None,
}

impl FlowDir {
    /// Convert to unit vector (dx, dy).
    pub fn to_vec(self) -> (f64, f64) {
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        match self {
            FlowDir::N  => ( 0.0, -1.0),
            FlowDir::NE => ( s2,  -s2),
            FlowDir::E  => ( 1.0,  0.0),
            FlowDir::SE => ( s2,   s2),
            FlowDir::S  => ( 0.0,  1.0),
            FlowDir::SW => (-s2,   s2),
            FlowDir::W  => (-1.0,  0.0),
            FlowDir::NW => (-s2,  -s2),
            FlowDir::None => (0.0, 0.0),
        }
    }

    /// All 8 compass directions.
    pub fn all() -> [FlowDir; 8] {
        [FlowDir::N, FlowDir::NE, FlowDir::E, FlowDir::SE,
         FlowDir::S, FlowDir::SW, FlowDir::W, FlowDir::NW]
    }
}

/// Direction offsets for neighbor lookup.
const DIR_OFFSETS: [(i32, i32); 8] = [
    ( 0, -1), // N
    ( 1, -1), // NE
    ( 1,  0), // E
    ( 1,  1), // SE
    ( 0,  1), // S
    (-1,  1), // SW
    (-1,  0), // W
    (-1, -1), // NW
];

// ── Cost grid ───────────────────────────────────────────────────

/// The base cost grid representing terrain.
#[derive(Debug, Clone)]
pub struct CostGrid {
    pub width: usize,
    pub height: usize,
    /// Per-cell traversal cost. 0 = impassable (wall). u8::MAX = very expensive.
    costs: Vec<u8>,
}

impl CostGrid {
    /// Create grid with uniform cost.
    pub fn new(width: usize, height: usize, default_cost: u8) -> Self {
        Self {
            width,
            height,
            costs: vec![default_cost; width * height],
        }
    }

    /// Set cost for a cell.
    pub fn set(&mut self, x: usize, y: usize, cost: u8) {
        if x < self.width && y < self.height {
            self.costs[y * self.width + x] = cost;
        }
    }

    /// Get cost for a cell.
    pub fn get(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.costs[y * self.width + x]
        } else {
            0
        }
    }

    /// Check if a cell is passable.
    pub fn is_passable(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 { return false; }
        let ux = x as usize;
        let uy = y as usize;
        ux < self.width && uy < self.height && self.costs[uy * self.width + ux] > 0
    }

    /// Check if coords are in bounds.
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height
    }
}

// ── Integration field ───────────────────────────────────────────

/// Integration field: cost-to-goal for each cell (Dijkstra from goals).
#[derive(Debug, Clone)]
pub struct IntegrationField {
    pub width: usize,
    pub height: usize,
    /// Cost-to-goal for each cell. u32::MAX = unreachable.
    values: Vec<u32>,
}

pub const UNREACHABLE: u32 = u32::MAX;

impl IntegrationField {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            values: vec![UNREACHABLE; width * height],
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn get(&self, x: usize, y: usize) -> u32 {
        if x < self.width && y < self.height {
            self.values[self.idx(x, y)]
        } else {
            UNREACHABLE
        }
    }

    fn set(&mut self, x: usize, y: usize, val: u32) {
        if x < self.width && y < self.height {
            let idx = self.idx(x, y);
            self.values[idx] = val;
        }
    }

    pub fn get_signed(&self, x: i32, y: i32) -> u32 {
        if x < 0 || y < 0 { return UNREACHABLE; }
        self.get(x as usize, y as usize)
    }
}

/// Build an integration field from multiple goals via Dijkstra wavefront.
pub fn build_integration_field(cost_grid: &CostGrid, goals: &[(usize, usize)]) -> IntegrationField {
    let mut field = IntegrationField::new(cost_grid.width, cost_grid.height);
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    for &(gx, gy) in goals {
        if gx < cost_grid.width && gy < cost_grid.height && cost_grid.get(gx, gy) > 0 {
            field.set(gx, gy, 0);
            queue.push_back((gx, gy));
        }
    }

    while let Some((cx, cy)) = queue.pop_front() {
        let current_cost = field.get(cx, cy);

        for &(dx, dy) in &DIR_OFFSETS {
            let nx = cx as i32 + dx;
            let ny = cy as i32 + dy;

            if !cost_grid.in_bounds(nx, ny) {
                continue;
            }

            let nux = nx as usize;
            let nuy = ny as usize;
            let cell_cost = cost_grid.get(nux, nuy);

            if cell_cost == 0 {
                continue; // wall
            }

            // Diagonal moves cost sqrt(2) ~= 14/10, cardinal = 10/10
            let move_cost = if dx != 0 && dy != 0 { 14 } else { 10 };
            let total_cost = current_cost.saturating_add(cell_cost as u32 * move_cost);

            if total_cost < field.get(nux, nuy) {
                field.set(nux, nuy, total_cost);
                queue.push_back((nux, nuy));
            }
        }
    }

    field
}

// ── Flow field ──────────────────────────────────────────────────

/// Flow field: best direction per cell toward the goal.
#[derive(Debug, Clone)]
pub struct FlowField {
    pub width: usize,
    pub height: usize,
    directions: Vec<FlowDir>,
}

impl FlowField {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            directions: vec![FlowDir::None; width * height],
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Get the flow direction at a cell.
    pub fn get(&self, x: usize, y: usize) -> FlowDir {
        if x < self.width && y < self.height {
            self.directions[self.idx(x, y)]
        } else {
            FlowDir::None
        }
    }

    fn set(&mut self, x: usize, y: usize, dir: FlowDir) {
        if x < self.width && y < self.height {
            let idx = self.idx(x, y);
            self.directions[idx] = dir;
        }
    }

    /// Look up a smooth (bilinear interpolated) direction vector at a floating-point position.
    pub fn sample_smooth(&self, fx: f64, fy: f64) -> (f64, f64) {
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;

        let get_vec = |x: i32, y: i32| -> (f64, f64) {
            if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
                (0.0, 0.0)
            } else {
                self.get(x as usize, y as usize).to_vec()
            }
        };

        let v00 = get_vec(x0, y0);
        let v10 = get_vec(x0 + 1, y0);
        let v01 = get_vec(x0, y0 + 1);
        let v11 = get_vec(x0 + 1, y0 + 1);

        let vx = v00.0 * (1.0 - tx) * (1.0 - ty)
               + v10.0 * tx * (1.0 - ty)
               + v01.0 * (1.0 - tx) * ty
               + v11.0 * tx * ty;

        let vy = v00.1 * (1.0 - tx) * (1.0 - ty)
               + v10.1 * tx * (1.0 - ty)
               + v01.1 * (1.0 - tx) * ty
               + v11.1 * tx * ty;

        (vx, vy)
    }
}

/// Build a flow field from an integration field.
pub fn build_flow_field(integration: &IntegrationField, cost_grid: &CostGrid) -> FlowField {
    let mut field = FlowField::new(integration.width, integration.height);

    for y in 0..integration.height {
        for x in 0..integration.width {
            if cost_grid.get(x, y) == 0 {
                continue; // wall
            }
            if integration.get(x, y) == UNREACHABLE {
                continue;
            }

            let mut best_cost = integration.get(x, y);
            let mut best_dir = FlowDir::None;

            for (i, &(dx, dy)) in DIR_OFFSETS.iter().enumerate() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nc = integration.get_signed(nx, ny);
                if nc < best_cost {
                    best_cost = nc;
                    best_dir = FlowDir::all()[i];
                }
            }

            field.set(x, y, best_dir);
        }
    }

    field
}

// ── LOS optimization ────────────────────────────────────────────

/// Check line-of-sight between two cells on the cost grid (Bresenham).
pub fn line_of_sight(cost_grid: &CostGrid, x0: i32, y0: i32, x1: i32, y1: i32) -> bool {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;

    loop {
        if !cost_grid.is_passable(x, y) {
            return false;
        }
        if x == x1 && y == y1 {
            return true;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

// ── Sector-based flow fields ────────────────────────────────────

/// A sector is a rectangular sub-region of the map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SectorId {
    pub sx: usize,
    pub sy: usize,
}

/// Sector-based flow field manager for large maps.
pub struct SectorFlowFieldManager {
    pub sector_size: usize,
    pub sectors_x: usize,
    pub sectors_y: usize,
    /// Cached flow fields per sector per goal-set key.
    cache: std::collections::HashMap<(SectorId, u64), FlowField>,
}

impl SectorFlowFieldManager {
    pub fn new(map_width: usize, map_height: usize, sector_size: usize) -> Self {
        let sectors_x = (map_width + sector_size - 1) / sector_size;
        let sectors_y = (map_height + sector_size - 1) / sector_size;
        Self {
            sector_size,
            sectors_x,
            sectors_y,
            cache: std::collections::HashMap::new(),
        }
    }

    /// Get sector for a world position.
    pub fn sector_for(&self, x: usize, y: usize) -> SectorId {
        SectorId {
            sx: x / self.sector_size,
            sy: y / self.sector_size,
        }
    }

    /// Cache a flow field for a sector+goal combination.
    pub fn cache_field(&mut self, sector: SectorId, goal_key: u64, field: FlowField) {
        self.cache.insert((sector, goal_key), field);
    }

    /// Retrieve a cached flow field.
    pub fn get_cached(&self, sector: SectorId, goal_key: u64) -> Option<&FlowField> {
        self.cache.get(&(sector, goal_key))
    }

    /// Clear cache for a sector (when terrain changes).
    pub fn invalidate_sector(&mut self, sector: SectorId) {
        self.cache.retain(|(s, _), _| *s != sector);
    }

    /// Clear entire cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Number of cached fields.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

// ── Convenience: full pipeline ──────────────────────────────────

/// Build integration + flow fields from cost grid and goals.
pub fn compute_flow_field(cost_grid: &CostGrid, goals: &[(usize, usize)]) -> (IntegrationField, FlowField) {
    let integration = build_integration_field(cost_grid, goals);
    let flow = build_flow_field(&integration, cost_grid);
    (integration, flow)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_open_grid(w: usize, h: usize) -> CostGrid {
        CostGrid::new(w, h, 1)
    }

    #[test]
    fn test_flowdir_to_vec() {
        let (dx, dy) = FlowDir::E.to_vec();
        assert!((dx - 1.0).abs() < 1e-10);
        assert!((dy - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flowdir_none() {
        let (dx, dy) = FlowDir::None.to_vec();
        assert!((dx).abs() < 1e-10);
        assert!((dy).abs() < 1e-10);
    }

    #[test]
    fn test_flowdir_diagonal_unit() {
        let (dx, dy) = FlowDir::NE.to_vec();
        let len = (dx * dx + dy * dy).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cost_grid_passable() {
        let mut grid = make_open_grid(5, 5);
        assert!(grid.is_passable(0, 0));
        grid.set(2, 2, 0);
        assert!(!grid.is_passable(2, 2));
    }

    #[test]
    fn test_cost_grid_out_of_bounds() {
        let grid = make_open_grid(5, 5);
        assert!(!grid.is_passable(-1, 0));
        assert!(!grid.is_passable(5, 0));
    }

    #[test]
    fn test_integration_field_single_goal() {
        let grid = make_open_grid(5, 5);
        let field = build_integration_field(&grid, &[(2, 2)]);
        assert_eq!(field.get(2, 2), 0);
        assert!(field.get(0, 0) > 0);
        assert!(field.get(0, 0) < UNREACHABLE);
    }

    #[test]
    fn test_integration_field_unreachable() {
        let mut grid = make_open_grid(5, 1);
        grid.set(2, 0, 0); // wall
        let field = build_integration_field(&grid, &[(0, 0)]);
        // Cells behind wall are unreachable for 1-row grid
        assert_eq!(field.get(3, 0), UNREACHABLE);
    }

    #[test]
    fn test_integration_field_adjacent_cheaper() {
        let grid = make_open_grid(3, 3);
        let field = build_integration_field(&grid, &[(1, 1)]);
        // Adjacent cells should be cheaper than corner cells
        let adj = field.get(1, 0);  // cardinal
        let corner = field.get(0, 0); // diagonal
        assert!(adj < corner);
    }

    #[test]
    fn test_flow_field_points_toward_goal() {
        let grid = make_open_grid(5, 5);
        let (_, flow) = compute_flow_field(&grid, &[(4, 2)]);
        // Left side should point east
        let dir = flow.get(0, 2);
        assert_eq!(dir, FlowDir::E);
    }

    #[test]
    fn test_flow_field_at_goal_is_none() {
        let grid = make_open_grid(5, 5);
        let (_, flow) = compute_flow_field(&grid, &[(2, 2)]);
        assert_eq!(flow.get(2, 2), FlowDir::None);
    }

    #[test]
    fn test_flow_field_wall_cell() {
        let mut grid = make_open_grid(5, 5);
        grid.set(2, 2, 0);
        let (_, flow) = compute_flow_field(&grid, &[(4, 4)]);
        assert_eq!(flow.get(2, 2), FlowDir::None);
    }

    #[test]
    fn test_multiple_goals() {
        let grid = make_open_grid(10, 1);
        let field = build_integration_field(&grid, &[(0, 0), (9, 0)]);
        // Middle should have lower cost than when only one goal
        let single_field = build_integration_field(&grid, &[(0, 0)]);
        assert!(field.get(5, 0) <= single_field.get(5, 0));
    }

    #[test]
    fn test_bilinear_interpolation() {
        let grid = make_open_grid(5, 5);
        let (_, flow) = compute_flow_field(&grid, &[(4, 2)]);
        let (vx, _vy) = flow.sample_smooth(0.5, 2.0);
        // Should have positive x component (heading east)
        assert!(vx > 0.0);
    }

    #[test]
    fn test_line_of_sight_clear() {
        let grid = make_open_grid(10, 10);
        assert!(line_of_sight(&grid, 0, 0, 9, 9));
    }

    #[test]
    fn test_line_of_sight_blocked() {
        let mut grid = make_open_grid(10, 10);
        grid.set(5, 5, 0);
        assert!(!line_of_sight(&grid, 0, 0, 9, 9));
    }

    #[test]
    fn test_line_of_sight_same_point() {
        let grid = make_open_grid(5, 5);
        assert!(line_of_sight(&grid, 2, 2, 2, 2));
    }

    #[test]
    fn test_sector_manager_basic() {
        let mut mgr = SectorFlowFieldManager::new(100, 100, 16);
        let sector = mgr.sector_for(5, 5);
        assert_eq!(sector, SectorId { sx: 0, sy: 0 });
        let sector2 = mgr.sector_for(20, 5);
        assert_eq!(sector2, SectorId { sx: 1, sy: 0 });

        let field = FlowField::new(16, 16);
        mgr.cache_field(sector, 42, field);
        assert_eq!(mgr.cache_size(), 1);
        assert!(mgr.get_cached(sector, 42).is_some());
        assert!(mgr.get_cached(sector, 99).is_none());
    }

    #[test]
    fn test_sector_invalidate() {
        let mut mgr = SectorFlowFieldManager::new(100, 100, 16);
        let s0 = SectorId { sx: 0, sy: 0 };
        let s1 = SectorId { sx: 1, sy: 0 };
        mgr.cache_field(s0, 1, FlowField::new(16, 16));
        mgr.cache_field(s1, 1, FlowField::new(16, 16));
        assert_eq!(mgr.cache_size(), 2);
        mgr.invalidate_sector(s0);
        assert_eq!(mgr.cache_size(), 1);
        assert!(mgr.get_cached(s1, 1).is_some());
    }

    #[test]
    fn test_sector_clear_cache() {
        let mut mgr = SectorFlowFieldManager::new(100, 100, 16);
        mgr.cache_field(SectorId { sx: 0, sy: 0 }, 1, FlowField::new(16, 16));
        mgr.cache_field(SectorId { sx: 1, sy: 0 }, 1, FlowField::new(16, 16));
        mgr.clear_cache();
        assert_eq!(mgr.cache_size(), 0);
    }

    #[test]
    fn test_high_cost_terrain() {
        let mut grid = make_open_grid(5, 1);
        grid.set(2, 0, 10); // expensive cell
        let field = build_integration_field(&grid, &[(4, 0)]);
        // Cost through expensive cell should be higher
        let normal = field.get(3, 0); // adjacent to goal
        let expensive = field.get(1, 0); // has to cross expensive cell
        assert!(expensive > normal);
    }

    #[test]
    fn test_compute_flow_field_convenience() {
        let grid = make_open_grid(3, 3);
        let (int_field, flow_field) = compute_flow_field(&grid, &[(1, 1)]);
        assert_eq!(int_field.get(1, 1), 0);
        assert_eq!(flow_field.get(1, 1), FlowDir::None);
    }

    #[test]
    fn test_flowdir_all() {
        let dirs = FlowDir::all();
        assert_eq!(dirs.len(), 8);
    }
}
