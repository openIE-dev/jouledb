//! Occupancy grid mapping — log-odds Bayesian update with Bresenham ray casting.
//!
//! Supports multi-resolution grids (quad-tree downsampling), configurable
//! sensor models, and efficient log-odds probability representation.

use std::fmt;

// ── Log-odds helpers ──────────────────────────────────────────────

/// Convert probability [0, 1] to log-odds.
#[inline]
pub fn prob_to_log_odds(p: f64) -> f64 {
    let clamped = p.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

/// Convert log-odds to probability [0, 1].
#[inline]
pub fn log_odds_to_prob(lo: f64) -> f64 {
    1.0 / (1.0 + (-lo).exp())
}

// ── Cell state ────────────────────────────────────────────────────

/// Occupancy state of a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    Free,
    Occupied,
    Unknown,
}

impl fmt::Display for CellState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellState::Free => write!(f, "free"),
            CellState::Occupied => write!(f, "occupied"),
            CellState::Unknown => write!(f, "unknown"),
        }
    }
}

// ── Sensor model ──────────────────────────────────────────────────

/// Inverse sensor model parameters.
#[derive(Debug, Clone, Copy)]
pub struct SensorModel {
    pub log_odds_occ: f64,
    pub log_odds_free: f64,
    pub max_range: f64,
    pub angular_resolution: f64,
}

impl Default for SensorModel {
    fn default() -> Self {
        Self {
            log_odds_occ: 0.85,
            log_odds_free: -0.4,
            max_range: 30.0,
            angular_resolution: 0.01,
        }
    }
}

impl SensorModel {
    pub fn new() -> Self { Self::default() }

    pub fn with_log_odds_occ(mut self, v: f64) -> Self { self.log_odds_occ = v; self }
    pub fn with_log_odds_free(mut self, v: f64) -> Self { self.log_odds_free = v; self }
    pub fn with_max_range(mut self, v: f64) -> Self { self.max_range = v; self }
    pub fn with_angular_resolution(mut self, v: f64) -> Self { self.angular_resolution = v; self }
}

// ── Grid configuration ───────────────────────────────────────────

/// Occupancy grid configuration.
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub log_odds_clamp_min: f64,
    pub log_odds_clamp_max: f64,
    pub sensor: SensorModel,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            width: 200,
            height: 200,
            resolution: 0.1,
            origin_x: -10.0,
            origin_y: -10.0,
            log_odds_clamp_min: -5.0,
            log_odds_clamp_max: 5.0,
            sensor: SensorModel::default(),
        }
    }
}

impl GridConfig {
    pub fn new() -> Self { Self::default() }

    pub fn with_size(mut self, w: usize, h: usize) -> Self {
        self.width = w;
        self.height = h;
        self
    }

    pub fn with_resolution(mut self, r: f64) -> Self {
        self.resolution = r;
        self
    }

    pub fn with_origin(mut self, x: f64, y: f64) -> Self {
        self.origin_x = x;
        self.origin_y = y;
        self
    }

    pub fn with_sensor(mut self, s: SensorModel) -> Self {
        self.sensor = s;
        self
    }

    pub fn with_clamp(mut self, lo: f64, hi: f64) -> Self {
        self.log_odds_clamp_min = lo;
        self.log_odds_clamp_max = hi;
        self
    }
}

// ── Occupancy grid ────────────────────────────────────────────────

/// 2-D occupancy grid backed by a flat log-odds array.
#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    pub config: GridConfig,
    /// Log-odds values, row-major (height × width).
    pub cells: Vec<f64>,
    pub update_count: u64,
}

impl OccupancyGrid {
    pub fn new(config: GridConfig) -> Self {
        let n = config.width * config.height;
        Self { cells: vec![0.0; n], update_count: 0, config }
    }

    /// Cell index from grid coords.
    #[inline]
    pub fn index(&self, col: usize, row: usize) -> usize {
        row * self.config.width + col
    }

    /// World → grid coordinates (returns None if out of bounds).
    pub fn world_to_grid(&self, wx: f64, wy: f64) -> Option<(usize, usize)> {
        let gx = ((wx - self.config.origin_x) / self.config.resolution).floor() as isize;
        let gy = ((wy - self.config.origin_y) / self.config.resolution).floor() as isize;
        if gx < 0 || gy < 0 || gx >= self.config.width as isize || gy >= self.config.height as isize {
            None
        } else {
            Some((gx as usize, gy as usize))
        }
    }

    /// Grid → world coordinates (cell center).
    pub fn grid_to_world(&self, col: usize, row: usize) -> (f64, f64) {
        let wx = self.config.origin_x + (col as f64 + 0.5) * self.config.resolution;
        let wy = self.config.origin_y + (row as f64 + 0.5) * self.config.resolution;
        (wx, wy)
    }

    /// Log-odds at grid coords.
    pub fn log_odds_at(&self, col: usize, row: usize) -> f64 {
        self.cells[self.index(col, row)]
    }

    /// Probability at grid coords.
    pub fn probability_at(&self, col: usize, row: usize) -> f64 {
        log_odds_to_prob(self.log_odds_at(col, row))
    }

    /// Cell state (threshold at 0.65 occupied, 0.35 free).
    pub fn cell_state(&self, col: usize, row: usize) -> CellState {
        let p = self.probability_at(col, row);
        if p > 0.65 {
            CellState::Occupied
        } else if p < 0.35 {
            CellState::Free
        } else {
            CellState::Unknown
        }
    }

    /// Set log-odds directly (clamped).
    pub fn set_log_odds(&mut self, col: usize, row: usize, lo: f64) {
        let idx = self.index(col, row);
        self.cells[idx] = lo.clamp(self.config.log_odds_clamp_min, self.config.log_odds_clamp_max);
    }

    /// Update a single cell with an additive log-odds increment.
    pub fn update_cell(&mut self, col: usize, row: usize, delta_lo: f64) {
        let idx = self.index(col, row);
        let new_lo = (self.cells[idx] + delta_lo)
            .clamp(self.config.log_odds_clamp_min, self.config.log_odds_clamp_max);
        self.cells[idx] = new_lo;
    }

    // ── Bresenham ray casting ─────────────────────────────────────

    /// Bresenham line from `(x0,y0)` to `(x1,y1)`, returning all cells along the ray.
    pub fn bresenham(x0: isize, y0: isize, x1: isize, y1: isize) -> Vec<(isize, isize)> {
        let mut points = Vec::new();
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: isize = if x0 < x1 { 1 } else { -1 };
        let sy: isize = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let mut cx = x0;
        let mut cy = y0;

        loop {
            points.push((cx, cy));
            if cx == x1 && cy == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                cx += sx;
            }
            if e2 <= dx {
                err += dx;
                cy += sy;
            }
        }
        points
    }

    /// Integrate a single laser scan measurement from robot position.
    pub fn integrate_beam(
        &mut self,
        robot_x: f64,
        robot_y: f64,
        robot_theta: f64,
        beam_angle: f64,
        measured_range: f64,
    ) {
        let max_range = self.config.sensor.max_range;
        let hit = measured_range < max_range;
        let effective_range = if hit { measured_range } else { max_range };

        let angle = robot_theta + beam_angle;
        let end_x = robot_x + effective_range * angle.cos();
        let end_y = robot_y + effective_range * angle.sin();

        let start = self.world_to_grid(robot_x, robot_y);
        let end = self.world_to_grid(end_x, end_y);

        if let (Some((sx, sy)), Some((ex, ey))) = (start, end) {
            let ray = Self::bresenham(sx as isize, sy as isize, ex as isize, ey as isize);
            let n = ray.len();
            for (i, &(cx, cy)) in ray.iter().enumerate() {
                if cx < 0 || cy < 0
                    || cx >= self.config.width as isize
                    || cy >= self.config.height as isize
                {
                    continue;
                }
                let col = cx as usize;
                let row = cy as usize;

                if i == n - 1 && hit {
                    self.update_cell(col, row, self.config.sensor.log_odds_occ);
                } else {
                    self.update_cell(col, row, self.config.sensor.log_odds_free);
                }
            }
        }

        self.update_count += 1;
    }

    /// Integrate a full scan (vector of `(angle, range)` pairs).
    pub fn integrate_scan(&mut self, robot_x: f64, robot_y: f64, robot_theta: f64, scan: &[(f64, f64)]) {
        for &(angle, range) in scan {
            self.integrate_beam(robot_x, robot_y, robot_theta, angle, range);
        }
    }

    // ── Multi-resolution downsampling ─────────────────────────────

    /// Create a half-resolution grid by 2×2 averaging.
    pub fn downsample(&self) -> OccupancyGrid {
        let new_w = self.config.width / 2;
        let new_h = self.config.height / 2;
        let new_res = self.config.resolution * 2.0;

        let new_config = GridConfig {
            width: new_w,
            height: new_h,
            resolution: new_res,
            origin_x: self.config.origin_x,
            origin_y: self.config.origin_y,
            ..self.config.clone()
        };

        let mut grid = OccupancyGrid::new(new_config);
        for row in 0..new_h {
            for col in 0..new_w {
                let r = row * 2;
                let c = col * 2;
                let avg = (self.log_odds_at(c, r)
                    + self.log_odds_at(c + 1, r)
                    + self.log_odds_at(c, r + 1)
                    + self.log_odds_at(c + 1, r + 1))
                    / 4.0;
                grid.set_log_odds(col, row, avg);
            }
        }
        grid
    }

    /// Count cells by state.
    pub fn count_states(&self) -> (usize, usize, usize) {
        let mut free = 0;
        let mut occupied = 0;
        let mut unknown = 0;
        for row in 0..self.config.height {
            for col in 0..self.config.width {
                match self.cell_state(col, row) {
                    CellState::Free => free += 1,
                    CellState::Occupied => occupied += 1,
                    CellState::Unknown => unknown += 1,
                }
            }
        }
        (free, occupied, unknown)
    }

    /// Reset all cells to unknown (log-odds = 0).
    pub fn reset(&mut self) {
        for v in &mut self.cells { *v = 0.0; }
        self.update_count = 0;
    }
}

impl fmt::Display for OccupancyGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (free, occ, unk) = self.count_states();
        write!(
            f,
            "OccupancyGrid({}x{}, res={:.2}m, free={}, occ={}, unk={}, updates={})",
            self.config.width, self.config.height, self.config.resolution,
            free, occ, unk, self.update_count
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_grid() -> OccupancyGrid {
        let cfg = GridConfig::new()
            .with_size(100, 100)
            .with_resolution(0.1)
            .with_origin(-5.0, -5.0);
        OccupancyGrid::new(cfg)
    }

    #[test]
    fn test_prob_log_odds_roundtrip() {
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let lo = prob_to_log_odds(p);
            let p2 = log_odds_to_prob(lo);
            assert!((p - p2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_odds_zero_is_fifty_percent() {
        assert!((log_odds_to_prob(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_grid_creation() {
        let g = small_grid();
        assert_eq!(g.cells.len(), 10000);
        assert_eq!(g.update_count, 0);
    }

    #[test]
    fn test_world_to_grid() {
        let g = small_grid();
        assert_eq!(g.world_to_grid(-5.0, -5.0), Some((0, 0)));
        assert_eq!(g.world_to_grid(0.0, 0.0), Some((50, 50)));
        assert_eq!(g.world_to_grid(100.0, 100.0), None);
    }

    #[test]
    fn test_grid_to_world() {
        let g = small_grid();
        let (wx, wy) = g.grid_to_world(50, 50);
        assert!((wx - 0.05).abs() < 1e-10);
        assert!((wy - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_initial_state_unknown() {
        let g = small_grid();
        assert_eq!(g.cell_state(50, 50), CellState::Unknown);
    }

    #[test]
    fn test_update_cell_occupied() {
        let mut g = small_grid();
        for _ in 0..20 {
            g.update_cell(50, 50, 0.85);
        }
        assert_eq!(g.cell_state(50, 50), CellState::Occupied);
    }

    #[test]
    fn test_update_cell_free() {
        let mut g = small_grid();
        for _ in 0..20 {
            g.update_cell(50, 50, -0.4);
        }
        assert_eq!(g.cell_state(50, 50), CellState::Free);
    }

    #[test]
    fn test_log_odds_clamping() {
        let mut g = small_grid();
        g.set_log_odds(10, 10, 999.0);
        assert_eq!(g.log_odds_at(10, 10), g.config.log_odds_clamp_max);
        g.set_log_odds(10, 10, -999.0);
        assert_eq!(g.log_odds_at(10, 10), g.config.log_odds_clamp_min);
    }

    #[test]
    fn test_bresenham_horizontal() {
        let pts = OccupancyGrid::bresenham(0, 0, 5, 0);
        assert_eq!(pts.len(), 6);
        assert_eq!(pts[0], (0, 0));
        assert_eq!(pts[5], (5, 0));
    }

    #[test]
    fn test_bresenham_diagonal() {
        let pts = OccupancyGrid::bresenham(0, 0, 3, 3);
        assert!(pts.len() >= 4);
        assert_eq!(*pts.first().unwrap(), (0, 0));
        assert_eq!(*pts.last().unwrap(), (3, 3));
    }

    #[test]
    fn test_bresenham_vertical() {
        let pts = OccupancyGrid::bresenham(2, 0, 2, 4);
        assert_eq!(pts.len(), 5);
        for &(x, _) in &pts { assert_eq!(x, 2); }
    }

    #[test]
    fn test_integrate_beam() {
        let mut g = small_grid();
        g.integrate_beam(0.0, 0.0, 0.0, 0.0, 2.0);
        assert_eq!(g.update_count, 1);
        // Endpoint should become more occupied
        if let Some((col, row)) = g.world_to_grid(2.0, 0.0) {
            assert!(g.log_odds_at(col, row) > 0.0);
        }
    }

    #[test]
    fn test_integrate_scan() {
        let mut g = small_grid();
        let scan: Vec<(f64, f64)> = (0..10)
            .map(|i| (i as f64 * 0.1 - 0.45, 3.0))
            .collect();
        g.integrate_scan(0.0, 0.0, 0.0, &scan);
        assert_eq!(g.update_count, 10);
    }

    #[test]
    fn test_downsample() {
        let g = small_grid();
        let d = g.downsample();
        assert_eq!(d.config.width, 50);
        assert_eq!(d.config.height, 50);
        assert!((d.config.resolution - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_count_states_initial() {
        let g = small_grid();
        let (free, occ, unk) = g.count_states();
        assert_eq!(free, 0);
        assert_eq!(occ, 0);
        assert_eq!(unk, 10000);
    }

    #[test]
    fn test_reset() {
        let mut g = small_grid();
        g.set_log_odds(10, 10, 3.0);
        g.update_count = 42;
        g.reset();
        assert!((g.log_odds_at(10, 10)).abs() < 1e-10);
        assert_eq!(g.update_count, 0);
    }

    #[test]
    fn test_display() {
        let g = small_grid();
        let s = format!("{}", g);
        assert!(s.contains("OccupancyGrid"));
        assert!(s.contains("100x100"));
    }

    #[test]
    fn test_cell_state_display() {
        assert_eq!(format!("{}", CellState::Free), "free");
        assert_eq!(format!("{}", CellState::Occupied), "occupied");
    }

    #[test]
    fn test_sensor_model_builder() {
        let sm = SensorModel::new().with_max_range(50.0).with_log_odds_occ(1.0);
        assert!((sm.max_range - 50.0).abs() < 1e-10);
        assert!((sm.log_odds_occ - 1.0).abs() < 1e-10);
    }
}
