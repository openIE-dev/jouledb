//! Reaction-diffusion systems — Gray-Scott model, Turing patterns.
//!
//! Replaces WebGL / Processing / p5.js reaction-diffusion shaders. Implements
//! the Gray-Scott model with configurable feed/kill rates, diffusion rates,
//! 5-point Laplacian stencil, forward Euler integration, parameter presets
//! for spots/stripes/waves/mitosis patterns, and f64 grid output.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RdError {
    ZeroDimension,
    InvalidParameter(String),
    SizeMismatch { expected: usize, got: usize },
}

impl fmt::Display for RdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be non-zero"),
            Self::InvalidParameter(s) => write!(f, "invalid parameter: {s}"),
            Self::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for RdError {}

// ── Parameters ─────────────────────────────────────────────────

/// Gray-Scott model parameters.
#[derive(Debug, Clone)]
pub struct GrayScottParams {
    /// Feed rate (replenishment of U).
    pub feed: f64,
    /// Kill rate (removal of V).
    pub kill: f64,
    /// Diffusion rate of U.
    pub du: f64,
    /// Diffusion rate of V.
    pub dv: f64,
    /// Integration timestep.
    pub dt: f64,
}

impl GrayScottParams {
    /// Custom parameters.
    pub fn new(feed: f64, kill: f64, du: f64, dv: f64, dt: f64) -> Result<Self, RdError> {
        if feed < 0.0 || kill < 0.0 || du < 0.0 || dv < 0.0 || dt <= 0.0 {
            return Err(RdError::InvalidParameter("all rates must be non-negative, dt > 0".into()));
        }
        Ok(Self { feed, kill, du, dv, dt })
    }

    /// Spots pattern preset.
    pub fn spots() -> Self {
        Self { feed: 0.035, kill: 0.065, du: 0.16, dv: 0.08, dt: 1.0 }
    }

    /// Stripes pattern preset.
    pub fn stripes() -> Self {
        Self { feed: 0.025, kill: 0.060, du: 0.16, dv: 0.08, dt: 1.0 }
    }

    /// Waves / spirals pattern preset.
    pub fn waves() -> Self {
        Self { feed: 0.014, kill: 0.054, du: 0.16, dv: 0.08, dt: 1.0 }
    }

    /// Mitosis (cell-splitting) pattern preset.
    pub fn mitosis() -> Self {
        Self { feed: 0.028, kill: 0.062, du: 0.16, dv: 0.08, dt: 1.0 }
    }

    /// Coral growth preset.
    pub fn coral() -> Self {
        Self { feed: 0.055, kill: 0.062, du: 0.16, dv: 0.08, dt: 1.0 }
    }

    /// Worms / maze-like pattern.
    pub fn worms() -> Self {
        Self { feed: 0.046, kill: 0.063, du: 0.16, dv: 0.08, dt: 1.0 }
    }
}

impl Default for GrayScottParams {
    fn default() -> Self {
        Self::spots()
    }
}

// ── Grid ───────────────────────────────────────────────────────

/// Gray-Scott reaction-diffusion system on a 2D grid.
#[derive(Debug, Clone)]
pub struct GrayScottGrid {
    width: usize,
    height: usize,
    /// Chemical U concentrations.
    u_grid: Vec<f64>,
    /// Chemical V concentrations.
    v_grid: Vec<f64>,
    /// Buffers for double-buffered update.
    u_buf: Vec<f64>,
    v_buf: Vec<f64>,
    params: GrayScottParams,
    generation: u64,
}

impl GrayScottGrid {
    /// Create a new grid with uniform U=1, V=0.
    pub fn new(width: usize, height: usize, params: GrayScottParams) -> Result<Self, RdError> {
        if width == 0 || height == 0 {
            return Err(RdError::ZeroDimension);
        }
        let size = width * height;
        Ok(Self {
            width,
            height,
            u_grid: vec![1.0; size],
            v_grid: vec![0.0; size],
            u_buf: vec![0.0; size],
            v_buf: vec![0.0; size],
            params,
            generation: 0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn params(&self) -> &GrayScottParams { &self.params }

    /// Set parameters.
    pub fn set_params(&mut self, params: GrayScottParams) {
        self.params = params;
    }

    /// Access U grid.
    pub fn u_grid(&self) -> &[f64] { &self.u_grid }

    /// Access V grid.
    pub fn v_grid(&self) -> &[f64] { &self.v_grid }

    /// Get U value at (x, y).
    pub fn get_u(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.u_grid[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Get V value at (x, y).
    pub fn get_v(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.v_grid[y * self.width + x]
        } else {
            0.0
        }
    }

    /// Set U value at (x, y).
    pub fn set_u(&mut self, x: usize, y: usize, val: f64) {
        if x < self.width && y < self.height {
            self.u_grid[y * self.width + x] = val;
        }
    }

    /// Set V value at (x, y).
    pub fn set_v(&mut self, x: usize, y: usize, val: f64) {
        if x < self.width && y < self.height {
            self.v_grid[y * self.width + x] = val;
        }
    }

    /// Seed a square region of V (and lower U) to start the reaction.
    pub fn seed_square(&mut self, cx: usize, cy: usize, radius: usize) {
        let r = radius as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                    let idx = y as usize * self.width + x as usize;
                    self.u_grid[idx] = 0.5;
                    self.v_grid[idx] = 0.25;
                }
            }
        }
    }

    /// Seed a circular region.
    pub fn seed_circle(&mut self, cx: usize, cy: usize, radius: usize) {
        let r = radius as i32;
        let r2 = (r * r) as f64;
        for dy in -r..=r {
            for dx in -r..=r {
                if (dx * dx + dy * dy) as f64 <= r2 {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
                        let idx = y as usize * self.width + x as usize;
                        self.u_grid[idx] = 0.5;
                        self.v_grid[idx] = 0.25;
                    }
                }
            }
        }
    }

    /// Seed with a deterministic random pattern using LCG.
    pub fn seed_random(&mut self, density: f64, seed: u64) {
        let mut rng = seed;
        for idx in 0..self.u_grid.len() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((rng >> 33) as f64) / (u32::MAX as f64);
            if val < density {
                self.u_grid[idx] = 0.5;
                self.v_grid[idx] = 0.25;
            }
        }
    }

    /// Compute 5-point Laplacian for a grid at (x, y) with wrapping.
    fn laplacian(grid: &[f64], width: usize, height: usize, x: usize, y: usize) -> f64 {
        let w = width;
        let h = height;
        let idx = y * w + x;
        let center = grid[idx];

        let left = grid[y * w + if x == 0 { w - 1 } else { x - 1 }];
        let right = grid[y * w + (x + 1) % w];
        let up = grid[(if y == 0 { h - 1 } else { y - 1 }) * w + x];
        let down = grid[((y + 1) % h) * w + x];

        left + right + up + down - 4.0 * center
    }

    /// Advance by one timestep using forward Euler integration.
    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;
        let f = self.params.feed;
        let k = self.params.kill;
        let du = self.params.du;
        let dv = self.params.dv;
        let dt = self.params.dt;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let u = self.u_grid[idx];
                let v = self.v_grid[idx];
                let uv2 = u * v * v;

                let lap_u = Self::laplacian(&self.u_grid, w, h, x, y);
                let lap_v = Self::laplacian(&self.v_grid, w, h, x, y);

                // Gray-Scott equations:
                // dU/dt = Du * laplacian(U) - UV^2 + f(1-U)
                // dV/dt = Dv * laplacian(V) + UV^2 - (f+k)V
                self.u_buf[idx] = u + dt * (du * lap_u - uv2 + f * (1.0 - u));
                self.v_buf[idx] = v + dt * (dv * lap_v + uv2 - (f + k) * v);

                // Clamp to [0, 1]
                self.u_buf[idx] = self.u_buf[idx].clamp(0.0, 1.0);
                self.v_buf[idx] = self.v_buf[idx].clamp(0.0, 1.0);
            }
        }

        std::mem::swap(&mut self.u_grid, &mut self.u_buf);
        std::mem::swap(&mut self.v_grid, &mut self.v_buf);
        self.generation += 1;
    }

    /// Advance by n timesteps.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Average U concentration.
    pub fn mean_u(&self) -> f64 {
        let sum: f64 = self.u_grid.iter().sum();
        sum / self.u_grid.len() as f64
    }

    /// Average V concentration.
    pub fn mean_v(&self) -> f64 {
        let sum: f64 = self.v_grid.iter().sum();
        sum / self.v_grid.len() as f64
    }

    /// Max V concentration.
    pub fn max_v(&self) -> f64 {
        self.v_grid.iter().cloned().fold(0.0f64, f64::max)
    }

    /// Min U concentration.
    pub fn min_u(&self) -> f64 {
        self.u_grid.iter().cloned().fold(1.0f64, f64::min)
    }

    /// Total energy: sum of V^2 across the grid (activity measure).
    pub fn total_energy(&self) -> f64 {
        self.v_grid.iter().map(|v| v * v).sum()
    }

    /// Render V grid as ASCII: ' ' for low, '.' medium, '#' high.
    pub fn render_v(&self) -> String {
        let mut s = String::with_capacity((self.width + 1) * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let v = self.v_grid[y * self.width + x];
                let ch = if v < 0.05 { ' ' } else if v < 0.15 { '.' } else if v < 0.3 { 'o' } else { '#' };
                s.push(ch);
            }
            if y + 1 < self.height {
                s.push('\n');
            }
        }
        s
    }

    /// Reset to initial state (U=1, V=0).
    pub fn reset(&mut self) {
        self.u_grid.fill(1.0);
        self.v_grid.fill(0.0);
        self.generation = 0;
    }
}

impl fmt::Display for GrayScottGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GrayScott({}x{}, gen={}, f={:.4}, k={:.4})",
            self.width, self.height, self.generation, self.params.feed, self.params.kill)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_new_grid() {
        let g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        assert_eq!(g.width(), 10);
        assert_eq!(g.height(), 10);
        assert_eq!(g.generation(), 0);
    }

    #[test]
    fn test_zero_dimension() {
        assert!(GrayScottGrid::new(0, 10, GrayScottParams::spots()).is_err());
    }

    #[test]
    fn test_initial_state() {
        let g = GrayScottGrid::new(5, 5, GrayScottParams::spots()).unwrap();
        assert!(approx_eq(g.mean_u(), 1.0));
        assert!(approx_eq(g.mean_v(), 0.0));
    }

    #[test]
    fn test_seed_square() {
        let mut g = GrayScottGrid::new(20, 20, GrayScottParams::spots()).unwrap();
        g.seed_square(10, 10, 2);
        // Center should have reduced U and increased V
        assert!(g.get_u(10, 10) < 1.0);
        assert!(g.get_v(10, 10) > 0.0);
    }

    #[test]
    fn test_seed_circle() {
        let mut g = GrayScottGrid::new(20, 20, GrayScottParams::spots()).unwrap();
        g.seed_circle(10, 10, 3);
        assert!(g.get_v(10, 10) > 0.0);
        // Corner of bounding box outside circle should still be 0
        // (3, 3) offset is exactly on boundary at distance sqrt(18) > 3
        assert!(approx_eq(g.get_v(0, 0), 0.0));
    }

    #[test]
    fn test_step_uniform_no_change() {
        // With uniform U=1, V=0: dU/dt = f(1-1) = 0, dV/dt = 0
        let mut g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        g.step();
        assert!(approx_eq(g.mean_u(), 1.0));
        assert!(approx_eq(g.mean_v(), 0.0));
    }

    #[test]
    fn test_step_changes_seeded_grid() {
        let mut g = GrayScottGrid::new(20, 20, GrayScottParams::spots()).unwrap();
        g.seed_square(10, 10, 2);
        let v_before = g.mean_v();
        g.step();
        let v_after = g.mean_v();
        // V should change (reaction + diffusion)
        assert!((v_after - v_before).abs() > 1e-10);
    }

    #[test]
    fn test_step_n() {
        let mut g = GrayScottGrid::new(20, 20, GrayScottParams::spots()).unwrap();
        g.seed_square(10, 10, 2);
        g.step_n(10);
        assert_eq!(g.generation(), 10);
    }

    #[test]
    fn test_clamp_values() {
        let mut g = GrayScottGrid::new(5, 5, GrayScottParams::spots()).unwrap();
        g.seed_square(2, 2, 1);
        g.step_n(50);
        // All values should be in [0, 1]
        for &u in g.u_grid() {
            assert!(u >= 0.0 && u <= 1.0, "U out of range: {u}");
        }
        for &v in g.v_grid() {
            assert!(v >= 0.0 && v <= 1.0, "V out of range: {v}");
        }
    }

    #[test]
    fn test_laplacian_uniform() {
        // Laplacian of a uniform field is zero
        let grid = vec![1.0; 25];
        let lap = GrayScottGrid::laplacian(&grid, 5, 5, 2, 2);
        assert!(approx_eq(lap, 0.0));
    }

    #[test]
    fn test_laplacian_spike() {
        // Single spike at center: laplacian should be negative
        let mut grid = vec![0.0; 25];
        grid[12] = 1.0; // center of 5x5
        let lap = GrayScottGrid::laplacian(&grid, 5, 5, 2, 2);
        assert!(lap < 0.0);
        assert!(approx_eq(lap, -4.0));
    }

    #[test]
    fn test_laplacian_wrapping() {
        // Verify wrapping: corner cell
        let grid = vec![1.0; 9]; // 3x3 all ones
        let lap = GrayScottGrid::laplacian(&grid, 3, 3, 0, 0);
        assert!(approx_eq(lap, 0.0));
    }

    #[test]
    fn test_params_spots() {
        let p = GrayScottParams::spots();
        assert!(approx_eq(p.feed, 0.035));
        assert!(approx_eq(p.kill, 0.065));
    }

    #[test]
    fn test_params_stripes() {
        let p = GrayScottParams::stripes();
        assert!(approx_eq(p.feed, 0.025));
    }

    #[test]
    fn test_params_waves() {
        let p = GrayScottParams::waves();
        assert!(approx_eq(p.feed, 0.014));
    }

    #[test]
    fn test_params_mitosis() {
        let p = GrayScottParams::mitosis();
        assert!(approx_eq(p.feed, 0.028));
    }

    #[test]
    fn test_invalid_params() {
        assert!(GrayScottParams::new(-0.1, 0.065, 0.16, 0.08, 1.0).is_err());
        assert!(GrayScottParams::new(0.035, 0.065, 0.16, 0.08, 0.0).is_err());
    }

    #[test]
    fn test_max_v() {
        let mut g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        g.seed_square(5, 5, 1);
        assert!(g.max_v() > 0.0);
    }

    #[test]
    fn test_min_u() {
        let mut g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        g.seed_square(5, 5, 1);
        assert!(g.min_u() < 1.0);
    }

    #[test]
    fn test_total_energy() {
        let g = GrayScottGrid::new(5, 5, GrayScottParams::spots()).unwrap();
        assert!(approx_eq(g.total_energy(), 0.0));
    }

    #[test]
    fn test_render_v() {
        let mut g = GrayScottGrid::new(5, 5, GrayScottParams::spots()).unwrap();
        g.seed_square(2, 2, 1);
        let s = g.render_v();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        g.seed_square(5, 5, 2);
        g.step_n(5);
        g.reset();
        assert_eq!(g.generation(), 0);
        assert!(approx_eq(g.mean_u(), 1.0));
        assert!(approx_eq(g.mean_v(), 0.0));
    }

    #[test]
    fn test_set_params() {
        let mut g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        g.set_params(GrayScottParams::stripes());
        assert!(approx_eq(g.params().feed, 0.025));
    }

    #[test]
    fn test_seed_random() {
        let mut g = GrayScottGrid::new(20, 20, GrayScottParams::spots()).unwrap();
        g.seed_random(0.1, 42);
        assert!(g.max_v() > 0.0);
    }

    #[test]
    fn test_display() {
        let g = GrayScottGrid::new(10, 10, GrayScottParams::spots()).unwrap();
        let s = format!("{g}");
        assert!(s.contains("GrayScott"));
    }
}
