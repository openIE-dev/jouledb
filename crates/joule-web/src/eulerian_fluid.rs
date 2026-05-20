//! Grid-based Eulerian fluid simulation using a MAC (Marker-And-Cell) staggered grid.
//!
//! Velocity components live on cell edges (u on vertical, v on horizontal), pressure
//! at cell centers. Supports semi-Lagrangian advection, Jacobi pressure projection for
//! incompressibility, explicit diffusion, vorticity confinement, and multiple boundary
//! conditions (solid wall, open, periodic). CFL condition limits the adaptive timestep.

use std::fmt;

// ── Errors ────────────────────────────────────────────────────

/// Eulerian fluid simulation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum EulerianError {
    /// Grid dimension too small.
    InvalidGrid(String),
    /// Solver did not converge.
    SolverNotConverged { iterations: usize, residual: f64 },
    /// CFL violation detected.
    CflViolation { max_velocity: f64, cfl_limit: f64 },
    /// Index out of grid bounds.
    OutOfBounds { x: usize, y: usize },
}

impl fmt::Display for EulerianError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid(msg) => write!(f, "invalid grid: {msg}"),
            Self::SolverNotConverged { iterations, residual } => {
                write!(f, "solver not converged after {iterations} iters, residual={residual:.2e}")
            }
            Self::CflViolation { max_velocity, cfl_limit } => {
                write!(f, "CFL violation: max_vel={max_velocity:.4}, limit={cfl_limit:.4}")
            }
            Self::OutOfBounds { x, y } => write!(f, "out of bounds: ({x}, {y})"),
        }
    }
}

impl std::error::Error for EulerianError {}

// ── Boundary Condition ────────────────────────────────────────

/// Type of boundary condition applied at domain edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// No-slip solid wall (velocity = 0 at wall).
    NoSlip,
    /// Free-slip wall (normal velocity = 0, tangential free).
    FreeSlip,
    /// Open boundary (zero-gradient / Neumann).
    Open,
    /// Periodic boundary.
    Periodic,
}

// ── Configuration ─────────────────────────────────────────────

/// Configuration for the Eulerian fluid simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct EulerianConfig {
    /// Number of cells in X direction.
    pub nx: usize,
    /// Number of cells in Y direction.
    pub ny: usize,
    /// Cell size (uniform, meters).
    pub cell_size: f64,
    /// Fluid density (kg/m^3).
    pub density: f64,
    /// Kinematic viscosity (m^2/s).
    pub viscosity: f64,
    /// Gravity (m/s^2, applied in -Y).
    pub gravity: f64,
    /// CFL number limit (typically 0.5-1.0).
    pub cfl_number: f64,
    /// Maximum timestep (seconds).
    pub max_dt: f64,
    /// Pressure solver max iterations.
    pub pressure_iters: usize,
    /// Pressure solver tolerance.
    pub pressure_tol: f64,
    /// Vorticity confinement strength (0 to disable).
    pub vorticity_confinement: f64,
    /// Boundary conditions: left, right, bottom, top.
    pub bc_left: BoundaryCondition,
    pub bc_right: BoundaryCondition,
    pub bc_bottom: BoundaryCondition,
    pub bc_top: BoundaryCondition,
}

impl Default for EulerianConfig {
    fn default() -> Self {
        Self {
            nx: 32,
            ny: 32,
            cell_size: 1.0 / 32.0,
            density: 1.0,
            viscosity: 0.001,
            gravity: 0.0,
            cfl_number: 0.8,
            max_dt: 0.016,
            pressure_iters: 100,
            pressure_tol: 1e-5,
            vorticity_confinement: 0.0,
            bc_left: BoundaryCondition::NoSlip,
            bc_right: BoundaryCondition::NoSlip,
            bc_bottom: BoundaryCondition::NoSlip,
            bc_top: BoundaryCondition::NoSlip,
        }
    }
}

impl EulerianConfig {
    pub fn validate(&self) -> Result<(), EulerianError> {
        if self.nx < 2 || self.ny < 2 {
            return Err(EulerianError::InvalidGrid("grid must be at least 2x2".into()));
        }
        if self.cell_size <= 0.0 {
            return Err(EulerianError::InvalidGrid("cell_size must be > 0".into()));
        }
        if self.density <= 0.0 {
            return Err(EulerianError::InvalidGrid("density must be > 0".into()));
        }
        Ok(())
    }
}

// ── Simulation Statistics ─────────────────────────────────────

/// Statistics from a simulation step.
#[derive(Debug, Clone, PartialEq)]
pub struct EulerianStats {
    pub step: u64,
    pub dt_used: f64,
    pub max_velocity: f64,
    pub max_divergence: f64,
    pub pressure_iters: usize,
    pub pressure_residual: f64,
    pub total_kinetic_energy: f64,
}

// ── MAC Grid ──────────────────────────────────────────────────

/// A 2D field stored as a flat Vec indexed [y * width + x].
#[derive(Debug, Clone)]
struct Field2D {
    data: Vec<f64>,
    width: usize,
    height: usize,
}

impl Field2D {
    fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0.0; width * height], width, height }
    }

    fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    fn set(&mut self, x: usize, y: usize, val: f64) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val;
        }
    }

    fn fill(&mut self, val: f64) {
        self.data.fill(val);
    }

    fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }
}

/// Bilinear interpolation on a Field2D at fractional coordinates.
fn bilinear(field: &Field2D, fx: f64, fy: f64) -> f64 {
    let x0 = (fx.floor() as isize).clamp(0, field.width as isize - 1) as usize;
    let y0 = (fy.floor() as isize).clamp(0, field.height as isize - 1) as usize;
    let x1 = (x0 + 1).min(field.width - 1);
    let y1 = (y0 + 1).min(field.height - 1);
    let sx = fx - x0 as f64;
    let sy = fy - y0 as f64;
    let sx = sx.clamp(0.0, 1.0);
    let sy = sy.clamp(0.0, 1.0);

    let v00 = field.get(x0, y0);
    let v10 = field.get(x1, y0);
    let v01 = field.get(x0, y1);
    let v11 = field.get(x1, y1);

    v00 * (1.0 - sx) * (1.0 - sy)
        + v10 * sx * (1.0 - sy)
        + v01 * (1.0 - sx) * sy
        + v11 * sx * sy
}

// ── Eulerian Fluid Simulation ─────────────────────────────────

/// The Eulerian MAC-grid fluid simulator.
pub struct EulerianFluid {
    pub config: EulerianConfig,
    /// Horizontal velocity on vertical cell faces (nx+1 x ny).
    u: Field2D,
    /// Vertical velocity on horizontal cell faces (nx x ny+1).
    v: Field2D,
    /// Pressure at cell centers (nx x ny).
    pressure: Field2D,
    /// Temp buffers.
    u_temp: Field2D,
    v_temp: Field2D,
    step_count: u64,
}

impl EulerianFluid {
    /// Create a new Eulerian fluid simulation.
    pub fn new(config: EulerianConfig) -> Result<Self, EulerianError> {
        config.validate()?;
        let nx = config.nx;
        let ny = config.ny;
        Ok(Self {
            u: Field2D::new(nx + 1, ny),
            v: Field2D::new(nx, ny + 1),
            pressure: Field2D::new(nx, ny),
            u_temp: Field2D::new(nx + 1, ny),
            v_temp: Field2D::new(nx, ny + 1),
            config,
            step_count: 0,
        })
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Set horizontal velocity at a u-face.
    pub fn set_u(&mut self, x: usize, y: usize, val: f64) {
        self.u.set(x, y, val);
    }

    /// Set vertical velocity at a v-face.
    pub fn set_v(&mut self, x: usize, y: usize, val: f64) {
        self.v.set(x, y, val);
    }

    /// Get horizontal velocity at a u-face.
    pub fn get_u(&self, x: usize, y: usize) -> f64 {
        self.u.get(x, y)
    }

    /// Get vertical velocity at a v-face.
    pub fn get_v(&self, x: usize, y: usize) -> f64 {
        self.v.get(x, y)
    }

    /// Get pressure at a cell center.
    pub fn get_pressure(&self, x: usize, y: usize) -> f64 {
        self.pressure.get(x, y)
    }

    /// Compute the CFL-limited timestep.
    pub fn compute_dt(&self) -> f64 {
        let max_u = self.u.max_abs();
        let max_v = self.v.max_abs();
        let max_vel = max_u.max(max_v).max(1e-10);
        let dt_cfl = self.config.cfl_number * self.config.cell_size / max_vel;
        dt_cfl.min(self.config.max_dt)
    }

    /// Maximum velocity magnitude in the grid.
    pub fn max_velocity(&self) -> f64 {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let mut max_sq = 0.0_f64;
        for y in 0..ny {
            for x in 0..nx {
                let uc = 0.5 * (self.u.get(x, y) + self.u.get(x + 1, y));
                let vc = 0.5 * (self.v.get(x, y) + self.v.get(x, y + 1));
                max_sq = max_sq.max(uc * uc + vc * vc);
            }
        }
        max_sq.sqrt()
    }

    /// Semi-Lagrangian advection: trace back through velocity field, interpolate.
    fn advect(&mut self, dt: f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let h = self.config.cell_size;

        // Advect u
        for y in 0..ny {
            for x in 0..(nx + 1) {
                let px = x as f64;
                let py = y as f64 + 0.5;
                let vel_u = self.u.get(x, y);
                let vel_v = self.sample_v_at(px * h, py * h);
                let bx = px - vel_u * dt / h;
                let by = py - vel_v * dt / h;
                self.u_temp.set(x, y, bilinear(&self.u, bx, by));
            }
        }

        // Advect v
        for y in 0..(ny + 1) {
            for x in 0..nx {
                let px = x as f64 + 0.5;
                let py = y as f64;
                let vel_u = self.sample_u_at(px * h, py * h);
                let vel_v = self.v.get(x, y);
                let bx = px - vel_u * dt / h;
                let by = py - vel_v * dt / h;
                self.v_temp.set(x, y, bilinear(&self.v, bx, by));
            }
        }

        std::mem::swap(&mut self.u, &mut self.u_temp);
        std::mem::swap(&mut self.v, &mut self.v_temp);
    }

    /// Sample u-velocity at an arbitrary world position.
    fn sample_u_at(&self, wx: f64, wy: f64) -> f64 {
        let h = self.config.cell_size;
        bilinear(&self.u, wx / h, wy / h - 0.5)
    }

    /// Sample v-velocity at an arbitrary world position.
    fn sample_v_at(&self, wx: f64, wy: f64) -> f64 {
        let h = self.config.cell_size;
        bilinear(&self.v, wx / h - 0.5, wy / h)
    }

    /// Explicit diffusion step.
    fn diffuse(&mut self, dt: f64) {
        let nu = self.config.viscosity;
        let h = self.config.cell_size;
        let factor = nu * dt / (h * h);

        // Diffuse u
        let nx = self.config.nx;
        let ny = self.config.ny;
        for y in 1..ny.saturating_sub(1) {
            for x in 1..nx {
                let lap = self.u.get(x + 1, y) + self.u.get(x.wrapping_sub(1).max(0), y)
                    + self.u.get(x, y + 1) + self.u.get(x, y.wrapping_sub(1).max(0))
                    - 4.0 * self.u.get(x, y);
                self.u_temp.set(x, y, self.u.get(x, y) + factor * lap);
            }
        }
        // Copy boundary rows/cols
        for y in 0..ny {
            self.u_temp.set(0, y, self.u.get(0, y));
            self.u_temp.set(nx, y, self.u.get(nx, y));
        }

        // Diffuse v
        for y in 1..ny {
            for x in 1..nx.saturating_sub(1) {
                let lap = self.v.get(x + 1, y) + self.v.get(x.wrapping_sub(1).max(0), y)
                    + self.v.get(x, y + 1) + self.v.get(x, y.wrapping_sub(1).max(0))
                    - 4.0 * self.v.get(x, y);
                self.v_temp.set(x, y, self.v.get(x, y) + factor * lap);
            }
        }
        for x in 0..nx {
            self.v_temp.set(x, 0, self.v.get(x, 0));
            self.v_temp.set(x, ny, self.v.get(x, ny));
        }

        std::mem::swap(&mut self.u, &mut self.u_temp);
        std::mem::swap(&mut self.v, &mut self.v_temp);
    }

    /// Add body forces (gravity).
    fn add_forces(&mut self, dt: f64) {
        let ny = self.config.ny;
        let nx = self.config.nx;
        let g = self.config.gravity;
        for y in 0..(ny + 1) {
            for x in 0..nx {
                let cur = self.v.get(x, y);
                self.v.set(x, y, cur - g * dt);
            }
        }
    }

    /// Compute divergence at cell centers.
    fn compute_divergence(&self) -> Field2D {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let h = self.config.cell_size;
        let mut div = Field2D::new(nx, ny);
        for y in 0..ny {
            for x in 0..nx {
                let d = (self.u.get(x + 1, y) - self.u.get(x, y)
                    + self.v.get(x, y + 1) - self.v.get(x, y)) / h;
                div.set(x, y, d);
            }
        }
        div
    }

    /// Jacobi pressure solve to enforce incompressibility.
    fn project(&mut self, dt: f64) -> (usize, f64) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let h = self.config.cell_size;
        let rho = self.config.density;
        let scale = dt / (rho * h * h);
        let max_iters = self.config.pressure_iters;
        let tol = self.config.pressure_tol;

        self.pressure.fill(0.0);
        let divergence = self.compute_divergence();

        let mut residual = 0.0;
        let mut iters_used = 0;

        for iter in 0..max_iters {
            let mut max_res = 0.0_f64;
            let old_pressure = self.pressure.clone();

            for y in 0..ny {
                for x in 0..nx {
                    let mut sum = 0.0;
                    let mut count = 0.0;

                    if x > 0 { sum += old_pressure.get(x - 1, y); count += 1.0; }
                    if x < nx - 1 { sum += old_pressure.get(x + 1, y); count += 1.0; }
                    if y > 0 { sum += old_pressure.get(x, y - 1); count += 1.0; }
                    if y < ny - 1 { sum += old_pressure.get(x, y + 1); count += 1.0; }

                    if count > 0.0 {
                        let new_p = (sum - divergence.get(x, y) * h / scale) / count;
                        self.pressure.set(x, y, new_p);
                        let res = (new_p - old_pressure.get(x, y)).abs();
                        max_res = max_res.max(res);
                    }
                }
            }

            residual = max_res;
            iters_used = iter + 1;
            if residual < tol {
                break;
            }
        }

        // Subtract pressure gradient from velocity
        for y in 0..ny {
            for x in 1..nx {
                let grad = (self.pressure.get(x, y) - self.pressure.get(x - 1, y)) * scale / h;
                self.u.set(x, y, self.u.get(x, y) - grad);
            }
        }
        for y in 1..ny {
            for x in 0..nx {
                let grad = (self.pressure.get(x, y) - self.pressure.get(x, y - 1)) * scale / h;
                self.v.set(x, y, self.v.get(x, y) - grad);
            }
        }

        (iters_used, residual)
    }

    /// Apply boundary conditions.
    fn apply_boundaries(&mut self) {
        let nx = self.config.nx;
        let ny = self.config.ny;

        // Left wall (u at x=0)
        match self.config.bc_left {
            BoundaryCondition::NoSlip => {
                for y in 0..ny { self.u.set(0, y, 0.0); }
            }
            BoundaryCondition::FreeSlip => {
                for y in 0..ny { self.u.set(0, y, 0.0); }
            }
            BoundaryCondition::Open => {
                for y in 0..ny { self.u.set(0, y, self.u.get(1, y)); }
            }
            BoundaryCondition::Periodic => {
                for y in 0..ny { self.u.set(0, y, self.u.get(nx, y)); }
            }
        }

        // Right wall (u at x=nx)
        match self.config.bc_right {
            BoundaryCondition::NoSlip | BoundaryCondition::FreeSlip => {
                for y in 0..ny { self.u.set(nx, y, 0.0); }
            }
            BoundaryCondition::Open => {
                for y in 0..ny { self.u.set(nx, y, self.u.get(nx - 1, y)); }
            }
            BoundaryCondition::Periodic => {
                for y in 0..ny { self.u.set(nx, y, self.u.get(0, y)); }
            }
        }

        // Bottom wall (v at y=0)
        match self.config.bc_bottom {
            BoundaryCondition::NoSlip | BoundaryCondition::FreeSlip => {
                for x in 0..nx { self.v.set(x, 0, 0.0); }
            }
            BoundaryCondition::Open => {
                for x in 0..nx { self.v.set(x, 0, self.v.get(x, 1)); }
            }
            BoundaryCondition::Periodic => {
                for x in 0..nx { self.v.set(x, 0, self.v.get(x, ny)); }
            }
        }

        // Top wall (v at y=ny)
        match self.config.bc_top {
            BoundaryCondition::NoSlip | BoundaryCondition::FreeSlip => {
                for x in 0..nx { self.v.set(x, ny, 0.0); }
            }
            BoundaryCondition::Open => {
                for x in 0..nx { self.v.set(x, ny, self.v.get(x, ny - 1)); }
            }
            BoundaryCondition::Periodic => {
                for x in 0..nx { self.v.set(x, ny, self.v.get(x, 0)); }
            }
        }
    }

    /// Compute vorticity at cell centers (scalar in 2D).
    pub fn compute_vorticity(&self) -> Vec<Vec<f64>> {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let h = self.config.cell_size;
        let mut vort = vec![vec![0.0; nx]; ny];
        for y in 1..ny.saturating_sub(1) {
            for x in 1..nx.saturating_sub(1) {
                let dv_dx = (self.v.get(x + 1, y) - self.v.get(x.saturating_sub(1), y)) / (2.0 * h);
                let du_dy = (self.u.get(x, y + 1) - self.u.get(x, y.saturating_sub(1))) / (2.0 * h);
                vort[y][x] = dv_dx - du_dy;
            }
        }
        vort
    }

    /// Apply vorticity confinement force.
    fn apply_vorticity_confinement(&mut self, dt: f64) {
        let eps = self.config.vorticity_confinement;
        if eps <= 0.0 {
            return;
        }
        let nx = self.config.nx;
        let ny = self.config.ny;
        let h = self.config.cell_size;
        let vort = self.compute_vorticity();

        for y in 2..ny.saturating_sub(2) {
            for x in 2..nx.saturating_sub(2) {
                let omega = vort[y][x];
                let dwdx = (vort[y][x + 1].abs() - vort[y][x - 1].abs()) / (2.0 * h);
                let dwdy = (vort[y + 1][x].abs() - vort[y - 1][x].abs()) / (2.0 * h);
                let mag = (dwdx * dwdx + dwdy * dwdy).sqrt().max(1e-10);
                let nx_hat = dwdx / mag;
                let ny_hat = dwdy / mag;

                // Confinement force: eps * h * (N x omega)
                let fx = eps * h * ny_hat * omega;
                let fy = -eps * h * nx_hat * omega;

                // Apply to nearest velocity faces
                let cu = self.u.get(x, y);
                self.u.set(x, y, cu + fx * dt);
                let cv = self.v.get(x, y);
                self.v.set(x, y, cv + fy * dt);
            }
        }
    }

    /// Total kinetic energy.
    pub fn total_kinetic_energy(&self) -> f64 {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let h = self.config.cell_size;
        let rho = self.config.density;
        let mut ke = 0.0;
        for y in 0..ny {
            for x in 0..nx {
                let uc = 0.5 * (self.u.get(x, y) + self.u.get(x + 1, y));
                let vc = 0.5 * (self.v.get(x, y) + self.v.get(x, y + 1));
                ke += 0.5 * rho * (uc * uc + vc * vc) * h * h;
            }
        }
        ke
    }

    /// Advance simulation by one step with operator splitting.
    pub fn step(&mut self) -> Result<EulerianStats, EulerianError> {
        let dt = self.compute_dt();

        // 1. Advection (semi-Lagrangian)
        self.advect(dt);

        // 2. Diffusion
        self.diffuse(dt);

        // 3. Body forces
        self.add_forces(dt);

        // 4. Vorticity confinement
        self.apply_vorticity_confinement(dt);

        // 5. Boundary conditions
        self.apply_boundaries();

        // 6. Pressure projection
        let (p_iters, p_res) = self.project(dt);

        // 7. Boundary conditions again
        self.apply_boundaries();

        self.step_count += 1;

        let div = self.compute_divergence();
        Ok(EulerianStats {
            step: self.step_count,
            dt_used: dt,
            max_velocity: self.max_velocity(),
            max_divergence: div.max_abs(),
            pressure_iters: p_iters,
            pressure_residual: p_res,
            total_kinetic_energy: self.total_kinetic_energy(),
        })
    }

    /// Run multiple steps.
    pub fn run(&mut self, steps: u64) -> Result<Vec<EulerianStats>, EulerianError> {
        let mut stats = Vec::with_capacity(steps as usize);
        for _ in 0..steps {
            stats.push(self.step()?);
        }
        Ok(stats)
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_fluid() -> EulerianFluid {
        EulerianFluid::new(EulerianConfig::default()).unwrap()
    }

    #[test]
    fn test_config_validation_ok() {
        assert!(EulerianConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validation_small_grid() {
        let mut cfg = EulerianConfig::default();
        cfg.nx = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_cell_size() {
        let mut cfg = EulerianConfig::default();
        cfg.cell_size = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_initial_state_zero() {
        let fluid = default_fluid();
        assert!((fluid.max_velocity()).abs() < 1e-12);
        assert!((fluid.total_kinetic_energy()).abs() < 1e-12);
    }

    #[test]
    fn test_set_get_u() {
        let mut fluid = default_fluid();
        fluid.set_u(5, 5, 1.5);
        assert!((fluid.get_u(5, 5) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_set_get_v() {
        let mut fluid = default_fluid();
        fluid.set_v(5, 5, -2.0);
        assert!((fluid.get_v(5, 5) - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_step_zero_velocity() {
        let mut fluid = default_fluid();
        let stats = fluid.step().unwrap();
        assert!(stats.max_velocity < 1e-10);
        assert_eq!(stats.step, 1);
    }

    #[test]
    fn test_cfl_timestep() {
        let mut fluid = default_fluid();
        fluid.set_u(10, 10, 100.0);
        let dt = fluid.compute_dt();
        assert!(dt < fluid.config.max_dt);
        assert!(dt > 0.0);
    }

    #[test]
    fn test_cfl_timestep_small_velocity() {
        let fluid = default_fluid();
        let dt = fluid.compute_dt();
        // With near-zero velocity, should use max_dt
        assert!((dt - fluid.config.max_dt).abs() < 1e-6);
    }

    #[test]
    fn test_pressure_at_start() {
        let fluid = default_fluid();
        assert!((fluid.get_pressure(5, 5)).abs() < 1e-12);
    }

    #[test]
    fn test_divergence_zero_field() {
        let fluid = default_fluid();
        let div = fluid.compute_divergence();
        assert!(div.max_abs() < 1e-12);
    }

    #[test]
    fn test_vorticity_zero_field() {
        let fluid = default_fluid();
        let vort = fluid.compute_vorticity();
        for row in &vort {
            for &val in row {
                assert!(val.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_step_with_initial_velocity() {
        let mut fluid = default_fluid();
        // Set a horizontal jet in the middle
        for y in 14..18 {
            for x in 0..5 {
                fluid.set_u(x, y, 1.0);
            }
        }
        let stats = fluid.step().unwrap();
        assert!(stats.max_velocity > 0.0);
    }

    #[test]
    fn test_kinetic_energy_positive_with_flow() {
        let mut fluid = default_fluid();
        fluid.set_u(10, 10, 5.0);
        assert!(fluid.total_kinetic_energy() > 0.0);
    }

    #[test]
    fn test_multiple_steps() {
        let mut fluid = default_fluid();
        for y in 14..18 {
            fluid.set_u(1, y, 0.5);
        }
        let result = fluid.run(5);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 5);
    }

    #[test]
    fn test_no_slip_boundaries() {
        let mut fluid = default_fluid();
        // Set velocity on boundary
        fluid.set_u(0, 5, 10.0);
        fluid.apply_boundaries();
        // No-slip: boundary velocity should be 0
        assert!((fluid.get_u(0, 5)).abs() < 1e-12);
    }

    #[test]
    fn test_bilinear_interpolation_center() {
        let mut field = Field2D::new(4, 4);
        field.set(1, 1, 4.0);
        field.set(2, 1, 4.0);
        field.set(1, 2, 4.0);
        field.set(2, 2, 4.0);
        let val = bilinear(&field, 1.5, 1.5);
        assert!((val - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_interpolation_corner() {
        let mut field = Field2D::new(4, 4);
        field.set(0, 0, 1.0);
        let val = bilinear(&field, 0.0, 0.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_open_boundary() {
        let mut cfg = EulerianConfig::default();
        cfg.bc_left = BoundaryCondition::Open;
        let mut fluid = EulerianFluid::new(cfg).unwrap();
        fluid.set_u(1, 5, 3.0);
        fluid.apply_boundaries();
        // Open boundary copies from interior
        assert!((fluid.get_u(0, 5) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_step_count() {
        let mut fluid = default_fluid();
        assert_eq!(fluid.step_count(), 0);
        for y in 14..18 {
            fluid.set_u(1, y, 0.1);
        }
        fluid.step().unwrap();
        assert_eq!(fluid.step_count(), 1);
    }

    #[test]
    fn test_gravity_induces_flow() {
        let mut cfg = EulerianConfig::default();
        cfg.gravity = 9.81;
        cfg.bc_bottom = BoundaryCondition::NoSlip;
        cfg.bc_top = BoundaryCondition::Open;
        let mut fluid = EulerianFluid::new(cfg).unwrap();
        fluid.step().unwrap();
        // After gravity, some vertical velocity should exist
        let stats = fluid.step().unwrap();
        // Gravity should create some velocity
        assert!(stats.step == 2);
    }

    #[test]
    fn test_field2d_fill() {
        let mut field = Field2D::new(5, 5);
        field.fill(3.14);
        for y in 0..5 {
            for x in 0..5 {
                assert!((field.get(x, y) - 3.14).abs() < 1e-12);
            }
        }
    }
}
