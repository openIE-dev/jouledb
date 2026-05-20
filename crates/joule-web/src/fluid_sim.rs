//! Simplified fluid simulation — 2D grid-based Navier-Stokes solver.
//!
//! Replaces fluid.js / FluidSim / LiquidFun.js with pure Rust.
//! Supports 2D grid-based density/velocity fields, advection, diffusion,
//! projection (pressure solve via Gauss-Seidel), source injection,
//! boundary conditions, and visualization data output.
//!
//! Based on Jos Stam's "Stable Fluids" (SIGGRAPH 1999).

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for fluid simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FluidError {
    /// Grid dimensions invalid.
    InvalidDimension(String),
    /// Position out of bounds.
    OutOfBounds { x: usize, y: usize, n: usize },
    /// Invalid parameter.
    InvalidParam(String),
}

impl fmt::Display for FluidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(d) => write!(f, "invalid dimension: {d}"),
            Self::OutOfBounds { x, y, n } => write!(f, "({x}, {y}) out of bounds for {n}x{n} grid"),
            Self::InvalidParam(p) => write!(f, "invalid parameter: {p}"),
        }
    }
}

impl std::error::Error for FluidError {}

// ── Index helpers ───────────────────────────────────────────────

/// Convert 2D (i, j) to 1D index in an (N+2)x(N+2) grid.
#[inline]
fn ix(i: usize, j: usize, n: usize) -> usize {
    i + (n + 2) * j
}

// ── Boundary conditions ─────────────────────────────────────────

/// Apply boundary conditions.
/// `b`: 0 for density, 1 for velocity-x, 2 for velocity-y.
fn set_bnd(b: i32, x: &mut [f64], n: usize) {
    for i in 1..=n {
        x[ix(0, i, n)] = if b == 1 { -x[ix(1, i, n)] } else { x[ix(1, i, n)] };
        x[ix(n + 1, i, n)] = if b == 1 { -x[ix(n, i, n)] } else { x[ix(n, i, n)] };
        x[ix(i, 0, n)] = if b == 2 { -x[ix(i, 1, n)] } else { x[ix(i, 1, n)] };
        x[ix(i, n + 1, n)] = if b == 2 { -x[ix(i, n, n)] } else { x[ix(i, n, n)] };
    }
    x[ix(0, 0, n)] = 0.5 * (x[ix(1, 0, n)] + x[ix(0, 1, n)]);
    x[ix(0, n + 1, n)] = 0.5 * (x[ix(1, n + 1, n)] + x[ix(0, n, n)]);
    x[ix(n + 1, 0, n)] = 0.5 * (x[ix(n, 0, n)] + x[ix(n + 1, 1, n)]);
    x[ix(n + 1, n + 1, n)] = 0.5 * (x[ix(n, n + 1, n)] + x[ix(n + 1, n, n)]);
}

/// Diffusion step (implicit Gauss-Seidel).
fn diffuse(b: i32, x: &mut [f64], x0: &[f64], diff: f64, dt: f64, n: usize, iterations: usize) {
    let a = dt * diff * (n as f64) * (n as f64);
    for _ in 0..iterations {
        for j in 1..=n {
            for i in 1..=n {
                x[ix(i, j, n)] = (x0[ix(i, j, n)]
                    + a * (x[ix(i - 1, j, n)] + x[ix(i + 1, j, n)]
                         + x[ix(i, j - 1, n)] + x[ix(i, j + 1, n)]))
                    / (1.0 + 4.0 * a);
            }
        }
        set_bnd(b, x, n);
    }
}

/// Advection step (semi-Lagrangian).
fn advect(b: i32, d: &mut [f64], d0: &[f64], u: &[f64], v: &[f64], dt: f64, n: usize) {
    let dt0 = dt * n as f64;
    let nf = n as f64;
    for j in 1..=n {
        for i in 1..=n {
            let mut x = i as f64 - dt0 * u[ix(i, j, n)];
            let mut y = j as f64 - dt0 * v[ix(i, j, n)];

            x = x.clamp(0.5, nf + 0.5);
            y = y.clamp(0.5, nf + 0.5);

            let i0 = x.floor() as usize;
            let i1 = i0 + 1;
            let j0 = y.floor() as usize;
            let j1 = j0 + 1;

            let s1 = x - i0 as f64;
            let s0 = 1.0 - s1;
            let t1 = y - j0 as f64;
            let t0 = 1.0 - t1;

            d[ix(i, j, n)] = s0 * (t0 * d0[ix(i0, j0, n)] + t1 * d0[ix(i0, j1, n)])
                           + s1 * (t0 * d0[ix(i1, j0, n)] + t1 * d0[ix(i1, j1, n)]);
        }
    }
    set_bnd(b, d, n);
}

/// Projection step (Helmholtz-Hodge decomposition).
fn project(u: &mut [f64], v: &mut [f64], p: &mut [f64], div: &mut [f64], n: usize, iterations: usize) {
    let h = 1.0 / n as f64;
    for j in 1..=n {
        for i in 1..=n {
            div[ix(i, j, n)] = -0.5 * h * (
                u[ix(i + 1, j, n)] - u[ix(i - 1, j, n)]
              + v[ix(i, j + 1, n)] - v[ix(i, j - 1, n)]
            );
            p[ix(i, j, n)] = 0.0;
        }
    }
    set_bnd(0, div, n);
    set_bnd(0, p, n);

    for _ in 0..iterations {
        for j in 1..=n {
            for i in 1..=n {
                p[ix(i, j, n)] = (div[ix(i, j, n)]
                    + p[ix(i - 1, j, n)] + p[ix(i + 1, j, n)]
                    + p[ix(i, j - 1, n)] + p[ix(i, j + 1, n)]) / 4.0;
            }
        }
        set_bnd(0, p, n);
    }

    for j in 1..=n {
        for i in 1..=n {
            u[ix(i, j, n)] -= 0.5 * (p[ix(i + 1, j, n)] - p[ix(i - 1, j, n)]) * n as f64;
            v[ix(i, j, n)] -= 0.5 * (p[ix(i, j + 1, n)] - p[ix(i, j - 1, n)]) * n as f64;
        }
    }
    set_bnd(1, u, n);
    set_bnd(2, v, n);
}

// ── Fluid Simulator ─────────────────────────────────────────────

/// A 2D grid-based fluid simulator.
#[derive(Debug, Clone)]
pub struct FluidSim {
    n: usize,
    dt: f64,
    diffusion: f64,
    viscosity: f64,
    iterations: usize,
    /// Density field.
    density: Vec<f64>,
    density_prev: Vec<f64>,
    /// Velocity X field.
    vx: Vec<f64>,
    vx_prev: Vec<f64>,
    /// Velocity Y field.
    vy: Vec<f64>,
    vy_prev: Vec<f64>,
    /// Temporary buffers for projection.
    p: Vec<f64>,
    div: Vec<f64>,
    step_count: u64,
}

impl FluidSim {
    /// Create a new fluid simulator with an NxN grid.
    pub fn new(n: usize, dt: f64, diffusion: f64, viscosity: f64) -> Result<Self, FluidError> {
        if n == 0 {
            return Err(FluidError::InvalidDimension("N must be >= 1".into()));
        }
        if dt <= 0.0 {
            return Err(FluidError::InvalidParam("dt must be > 0".into()));
        }
        let size = (n + 2) * (n + 2);
        Ok(Self {
            n,
            dt,
            diffusion,
            viscosity,
            iterations: 20,
            density: vec![0.0; size],
            density_prev: vec![0.0; size],
            vx: vec![0.0; size],
            vx_prev: vec![0.0; size],
            vy: vec![0.0; size],
            vy_prev: vec![0.0; size],
            p: vec![0.0; size],
            div: vec![0.0; size],
            step_count: 0,
        })
    }

    /// Set the number of Gauss-Seidel iterations.
    pub fn with_iterations(mut self, iters: usize) -> Self {
        self.iterations = iters;
        self
    }

    /// Grid size N.
    pub fn grid_size(&self) -> usize { self.n }

    /// Step count.
    pub fn step_count(&self) -> u64 { self.step_count }

    /// Add density at position (i, j).
    pub fn add_density(&mut self, i: usize, j: usize, amount: f64) -> Result<(), FluidError> {
        if i == 0 || i > self.n || j == 0 || j > self.n {
            return Err(FluidError::OutOfBounds { x: i, y: j, n: self.n });
        }
        self.density[ix(i, j, self.n)] += amount;
        Ok(())
    }

    /// Add velocity at position (i, j).
    pub fn add_velocity(&mut self, i: usize, j: usize, vx: f64, vy: f64) -> Result<(), FluidError> {
        if i == 0 || i > self.n || j == 0 || j > self.n {
            return Err(FluidError::OutOfBounds { x: i, y: j, n: self.n });
        }
        self.vx[ix(i, j, self.n)] += vx;
        self.vy[ix(i, j, self.n)] += vy;
        Ok(())
    }

    /// Get density at (i, j).
    pub fn get_density(&self, i: usize, j: usize) -> f64 {
        if i <= self.n + 1 && j <= self.n + 1 {
            self.density[ix(i, j, self.n)]
        } else {
            0.0
        }
    }

    /// Get velocity at (i, j).
    pub fn get_velocity(&self, i: usize, j: usize) -> (f64, f64) {
        if i <= self.n + 1 && j <= self.n + 1 {
            (self.vx[ix(i, j, self.n)], self.vy[ix(i, j, self.n)])
        } else {
            (0.0, 0.0)
        }
    }

    /// Total density in the grid.
    pub fn total_density(&self) -> f64 {
        let mut sum = 0.0;
        for j in 1..=self.n {
            for i in 1..=self.n {
                sum += self.density[ix(i, j, self.n)];
            }
        }
        sum
    }

    /// Maximum density value.
    pub fn max_density(&self) -> f64 {
        let mut max = 0.0f64;
        for j in 1..=self.n {
            for i in 1..=self.n {
                let d = self.density[ix(i, j, self.n)];
                if d > max { max = d; }
            }
        }
        max
    }

    /// Maximum velocity magnitude.
    pub fn max_velocity(&self) -> f64 {
        let mut max = 0.0f64;
        for j in 1..=self.n {
            for i in 1..=self.n {
                let u = self.vx[ix(i, j, self.n)];
                let v = self.vy[ix(i, j, self.n)];
                let mag = (u * u + v * v).sqrt();
                if mag > max { max = mag; }
            }
        }
        max
    }

    /// Step the velocity field.
    fn velocity_step(&mut self) {
        let n = self.n;
        let dt = self.dt;
        let visc = self.viscosity;
        let iters = self.iterations;

        // Swap current and previous.
        std::mem::swap(&mut self.vx, &mut self.vx_prev);
        std::mem::swap(&mut self.vy, &mut self.vy_prev);

        diffuse(1, &mut self.vx, &self.vx_prev, visc, dt, n, iters);
        diffuse(2, &mut self.vy, &self.vy_prev, visc, dt, n, iters);

        project(&mut self.vx, &mut self.vy, &mut self.p, &mut self.div, n, iters);

        std::mem::swap(&mut self.vx, &mut self.vx_prev);
        std::mem::swap(&mut self.vy, &mut self.vy_prev);

        advect(1, &mut self.vx, &self.vx_prev, &self.vx_prev, &self.vy_prev, dt, n);
        advect(2, &mut self.vy, &self.vy_prev, &self.vx_prev, &self.vy_prev, dt, n);

        project(&mut self.vx, &mut self.vy, &mut self.p, &mut self.div, n, iters);
    }

    /// Step the density field.
    fn density_step(&mut self) {
        let n = self.n;
        let dt = self.dt;
        let diff = self.diffusion;
        let iters = self.iterations;

        std::mem::swap(&mut self.density, &mut self.density_prev);
        diffuse(0, &mut self.density, &self.density_prev, diff, dt, n, iters);

        std::mem::swap(&mut self.density, &mut self.density_prev);
        advect(0, &mut self.density, &self.density_prev, &self.vx, &self.vy, dt, n);
    }

    /// Advance the simulation by one time step.
    pub fn step(&mut self) {
        self.velocity_step();
        self.density_step();
        self.step_count += 1;
    }

    /// Step multiple times.
    pub fn step_n(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Reset the simulation.
    pub fn reset(&mut self) {
        let size = (self.n + 2) * (self.n + 2);
        self.density = vec![0.0; size];
        self.density_prev = vec![0.0; size];
        self.vx = vec![0.0; size];
        self.vx_prev = vec![0.0; size];
        self.vy = vec![0.0; size];
        self.vy_prev = vec![0.0; size];
        self.p = vec![0.0; size];
        self.div = vec![0.0; size];
        self.step_count = 0;
    }

    /// Export density field as a flat vector (row-major, interior cells only).
    pub fn density_field(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * self.n);
        for j in 1..=self.n {
            for i in 1..=self.n {
                out.push(self.density[ix(i, j, self.n)]);
            }
        }
        out
    }

    /// Export velocity field as (vx, vy) pairs (interior cells only).
    pub fn velocity_field(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.n * self.n);
        for j in 1..=self.n {
            for i in 1..=self.n {
                out.push((self.vx[ix(i, j, self.n)], self.vy[ix(i, j, self.n)]));
            }
        }
        out
    }

    /// Inject a line of density (horizontal at row j from x1 to x2).
    pub fn inject_line(&mut self, j: usize, x1: usize, x2: usize, amount: f64) {
        let start = x1.max(1).min(self.n);
        let end = x2.max(1).min(self.n);
        for i in start..=end {
            self.density[ix(i, j.max(1).min(self.n), self.n)] += amount;
        }
    }

    /// Inject a circular blob of density.
    pub fn inject_blob(&mut self, cx: usize, cy: usize, radius: usize, amount: f64) {
        let r2 = (radius * radius) as f64;
        for j in 1..=self.n {
            for i in 1..=self.n {
                let dx = i as f64 - cx as f64;
                let dy = j as f64 - cy as f64;
                if dx * dx + dy * dy <= r2 {
                    self.density[ix(i, j, self.n)] += amount;
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        assert_eq!(sim.grid_size(), 10);
        assert_eq!(sim.step_count(), 0);
    }

    #[test]
    fn test_zero_dimension() {
        assert!(FluidSim::new(0, 0.1, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_invalid_dt() {
        assert!(FluidSim::new(10, 0.0, 0.0, 0.0).is_err());
        assert!(FluidSim::new(10, -1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_add_density() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_density(5, 5, 100.0).unwrap();
        assert!((sim.get_density(5, 5) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_density_oob() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        assert!(sim.add_density(0, 5, 10.0).is_err());
        assert!(sim.add_density(11, 5, 10.0).is_err());
    }

    #[test]
    fn test_add_velocity() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_velocity(5, 5, 1.0, -1.0).unwrap();
        let (vx, vy) = sim.get_velocity(5, 5);
        assert!((vx - 1.0).abs() < 1e-10);
        assert!((vy - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_total_density() {
        let mut sim = FluidSim::new(5, 0.1, 0.0, 0.0).unwrap();
        sim.add_density(3, 3, 50.0).unwrap();
        assert!((sim.total_density() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_step_advances() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_density(5, 5, 100.0).unwrap();
        sim.step();
        assert_eq!(sim.step_count(), 1);
    }

    #[test]
    fn test_density_spreads_with_diffusion() {
        let mut sim = FluidSim::new(10, 0.1, 1.0, 0.0).unwrap();
        sim.add_density(5, 5, 100.0).unwrap();
        let initial_at_neighbor = sim.get_density(6, 5);
        sim.step_n(5);
        let after = sim.get_density(6, 5);
        assert!(after > initial_at_neighbor, "diffusion should spread density");
    }

    #[test]
    fn test_velocity_moves_density() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_density(5, 5, 100.0).unwrap();
        sim.add_velocity(5, 5, 5.0, 0.0).unwrap();
        sim.step_n(3);
        // Density should have moved rightward.
        let right_density = sim.get_density(7, 5);
        assert!(right_density > 0.0, "density should move with velocity");
    }

    #[test]
    fn test_reset() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_density(5, 5, 100.0).unwrap();
        sim.step();
        sim.reset();
        assert_eq!(sim.step_count(), 0);
        assert!((sim.total_density() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_field_size() {
        let sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        assert_eq!(sim.density_field().len(), 100);
    }

    #[test]
    fn test_velocity_field_size() {
        let sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        assert_eq!(sim.velocity_field().len(), 100);
    }

    #[test]
    fn test_inject_line() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.inject_line(5, 2, 8, 10.0);
        for i in 2..=8 {
            assert!((sim.get_density(i, 5) - 10.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inject_blob() {
        let mut sim = FluidSim::new(20, 0.1, 0.0, 0.0).unwrap();
        sim.inject_blob(10, 10, 3, 5.0);
        assert!(sim.get_density(10, 10) > 0.0);
        assert!(sim.total_density() > 5.0);
    }

    #[test]
    fn test_max_density() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_density(3, 3, 50.0).unwrap();
        sim.add_density(7, 7, 100.0).unwrap();
        assert!((sim.max_density() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_velocity() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        sim.add_velocity(5, 5, 3.0, 4.0).unwrap();
        let max_v = sim.max_velocity();
        assert!((max_v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_sim_no_density() {
        let sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap();
        assert!((sim.total_density() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_with_iterations() {
        let sim = FluidSim::new(10, 0.1, 0.0, 0.0).unwrap().with_iterations(50);
        assert_eq!(sim.iterations, 50);
    }

    #[test]
    fn test_viscosity_damps_velocity() {
        let mut sim = FluidSim::new(10, 0.1, 0.0, 10.0).unwrap();
        sim.add_velocity(5, 5, 10.0, 10.0).unwrap();
        let initial_max = sim.max_velocity();
        sim.step_n(10);
        let final_max = sim.max_velocity();
        assert!(final_max < initial_max, "viscosity should damp velocity: {final_max} vs {initial_max}");
    }

    #[test]
    fn test_step_n() {
        let mut sim = FluidSim::new(5, 0.1, 0.0, 0.0).unwrap();
        sim.step_n(10);
        assert_eq!(sim.step_count(), 10);
    }
}
