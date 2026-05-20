//! Navier-Stokes solver components for incompressible fluid dynamics.
//!
//! Provides individual operator-splitting stages: advection (semi-Lagrangian and
//! MacCormack with clamping), Jacobi diffusion, body forces, velocity boundary
//! enforcement, divergence and curl computation, Reynolds number estimation,
//! and energy dissipation tracking. These building blocks compose into a full
//! incompressible Navier-Stokes solver.

use std::fmt;

// ── Errors ────────────────────────────────────────────────────

/// Navier-Stokes solver errors.
#[derive(Debug, Clone, PartialEq)]
pub enum NavierStokesError {
    /// Grid dimension issue.
    InvalidGrid(String),
    /// Solver diverged (NaN or Inf).
    Diverged(String),
    /// Incompatible field dimensions.
    DimensionMismatch { expected: (usize, usize), got: (usize, usize) },
}

impl fmt::Display for NavierStokesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid(msg) => write!(f, "invalid grid: {msg}"),
            Self::Diverged(msg) => write!(f, "solver diverged: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}x{}, got {}x{}", expected.0, expected.1, got.0, got.1)
            }
        }
    }
}

impl std::error::Error for NavierStokesError {}

// ── 2D Scalar / Vector Field ──────────────────────────────────

/// A 2D scalar field on a regular grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    pub data: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
}

impl ScalarField {
    pub fn new(nx: usize, ny: usize) -> Self {
        Self { data: vec![0.0; nx * ny], nx, ny }
    }

    pub fn filled(nx: usize, ny: usize, val: f64) -> Self {
        Self { data: vec![val; nx * ny], nx, ny }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.nx && y < self.ny {
            self.data[y * self.nx + x]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        if x < self.nx && y < self.ny {
            self.data[y * self.nx + x] = val;
        }
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
    }

    pub fn l2_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    /// Bilinear interpolation at fractional grid coordinates.
    pub fn sample(&self, fx: f64, fy: f64) -> f64 {
        let x0 = (fx.floor() as isize).clamp(0, self.nx as isize - 1) as usize;
        let y0 = (fy.floor() as isize).clamp(0, self.ny as isize - 1) as usize;
        let x1 = (x0 + 1).min(self.nx - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let sx = (fx - x0 as f64).clamp(0.0, 1.0);
        let sy = (fy - y0 as f64).clamp(0.0, 1.0);

        self.get(x0, y0) * (1.0 - sx) * (1.0 - sy)
            + self.get(x1, y0) * sx * (1.0 - sy)
            + self.get(x0, y1) * (1.0 - sx) * sy
            + self.get(x1, y1) * sx * sy
    }
}

/// A 2D vector field: two scalar fields (u, v).
#[derive(Debug, Clone, PartialEq)]
pub struct VectorField2D {
    pub u: ScalarField,
    pub v: ScalarField,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
}

impl VectorField2D {
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Self {
            u: ScalarField::new(nx, ny),
            v: ScalarField::new(nx, ny),
            nx,
            ny,
            dx,
        }
    }

    /// Maximum velocity magnitude on the grid.
    pub fn max_speed(&self) -> f64 {
        let mut max_sq = 0.0_f64;
        for i in 0..self.u.data.len() {
            let uu = self.u.data[i];
            let vv = self.v.data[i];
            max_sq = max_sq.max(uu * uu + vv * vv);
        }
        max_sq.sqrt()
    }

    /// Total kinetic energy (per unit density): 0.5 * sum(u^2 + v^2) * dx^2.
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for i in 0..self.u.data.len() {
            ke += self.u.data[i] * self.u.data[i] + self.v.data[i] * self.v.data[i];
        }
        0.5 * ke * self.dx * self.dx
    }
}

// ── Advection ─────────────────────────────────────────────────

/// Semi-Lagrangian advection: trace back through velocity field, interpolate.
pub fn advect_semi_lagrangian(
    field: &ScalarField,
    vel: &VectorField2D,
    dt: f64,
) -> Result<ScalarField, NavierStokesError> {
    if field.nx != vel.nx || field.ny != vel.ny {
        return Err(NavierStokesError::DimensionMismatch {
            expected: (vel.nx, vel.ny),
            got: (field.nx, field.ny),
        });
    }
    let dx = vel.dx;
    let mut result = ScalarField::new(field.nx, field.ny);
    for y in 0..field.ny {
        for x in 0..field.nx {
            let ux = vel.u.get(x, y);
            let vy = vel.v.get(x, y);
            let bx = x as f64 - ux * dt / dx;
            let by = y as f64 - vy * dt / dx;
            result.set(x, y, field.sample(bx, by));
        }
    }
    Ok(result)
}

/// MacCormack advection with clamping for stability.
/// Predictor: forward semi-Lagrangian. Corrector: backward from predicted.
/// Clamp to extrema of local neighborhood for monotonicity.
pub fn advect_maccormack(
    field: &ScalarField,
    vel: &VectorField2D,
    dt: f64,
) -> Result<ScalarField, NavierStokesError> {
    // Predictor: standard semi-Lagrangian (forward)
    let phi_hat = advect_semi_lagrangian(field, vel, dt)?;

    // Corrector: advect backward from predicted
    let phi_back = advect_semi_lagrangian(&phi_hat, vel, -dt)?;

    // MacCormack correction: phi_hat + 0.5 * (phi_n - phi_back)
    let mut result = ScalarField::new(field.nx, field.ny);
    for y in 0..field.ny {
        for x in 0..field.nx {
            let corrected = phi_hat.get(x, y) + 0.5 * (field.get(x, y) - phi_back.get(x, y));

            // Clamp to local extrema for stability
            let dx_inv = vel.dx;
            let bx = x as f64 - vel.u.get(x, y) * dt / dx_inv;
            let by = y as f64 - vel.v.get(x, y) * dt / dx_inv;

            let x0 = (bx.floor() as isize).clamp(0, field.nx as isize - 1) as usize;
            let y0 = (by.floor() as isize).clamp(0, field.ny as isize - 1) as usize;
            let x1 = (x0 + 1).min(field.nx - 1);
            let y1 = (y0 + 1).min(field.ny - 1);

            let local_min = field.get(x0, y0)
                .min(field.get(x1, y0))
                .min(field.get(x0, y1))
                .min(field.get(x1, y1));
            let local_max = field.get(x0, y0)
                .max(field.get(x1, y0))
                .max(field.get(x0, y1))
                .max(field.get(x1, y1));

            result.set(x, y, corrected.clamp(local_min, local_max));
        }
    }
    Ok(result)
}

// ── Diffusion ─────────────────────────────────────────────────

/// Jacobi diffusion iteration: solve (I - nu*dt*Laplacian) * phi_new = phi_old.
/// Returns number of iterations used and final residual.
pub fn diffuse_jacobi(
    field: &mut ScalarField,
    nu: f64,
    dt: f64,
    dx: f64,
    max_iters: usize,
    tolerance: f64,
) -> (usize, f64) {
    let alpha = dx * dx / (nu * dt);
    let beta = 4.0 + alpha;
    let source = field.clone();
    let nx = field.nx;
    let ny = field.ny;
    let mut residual = 0.0;

    for iter in 0..max_iters {
        let old = field.clone();
        let mut max_res = 0.0_f64;
        for y in 1..ny.saturating_sub(1) {
            for x in 1..nx.saturating_sub(1) {
                let neighbors = old.get(x + 1, y) + old.get(x - 1, y)
                    + old.get(x, y + 1) + old.get(x, y - 1);
                let new_val = (neighbors + alpha * source.get(x, y)) / beta;
                field.set(x, y, new_val);
                max_res = max_res.max((new_val - old.get(x, y)).abs());
            }
        }
        residual = max_res;
        if residual < tolerance {
            return (iter + 1, residual);
        }
    }
    (max_iters, residual)
}

// ── Body Forces ───────────────────────────────────────────────

/// Apply gravity to the v-component of velocity.
pub fn apply_gravity(vel: &mut VectorField2D, gravity: f64, dt: f64) {
    for val in &mut vel.v.data {
        *val -= gravity * dt;
    }
}

/// Apply an arbitrary external force field.
pub fn apply_external_force(
    vel: &mut VectorField2D,
    force_u: &ScalarField,
    force_v: &ScalarField,
    dt: f64,
) {
    let n = vel.u.data.len().min(force_u.data.len());
    for i in 0..n {
        vel.u.data[i] += force_u.data[i] * dt;
    }
    let n = vel.v.data.len().min(force_v.data.len());
    for i in 0..n {
        vel.v.data[i] += force_v.data[i] * dt;
    }
}

// ── Divergence and Curl ───────────────────────────────────────

/// Compute divergence of velocity: div(V) = du/dx + dv/dy.
pub fn divergence(vel: &VectorField2D) -> ScalarField {
    let nx = vel.nx;
    let ny = vel.ny;
    let dx = vel.dx;
    let mut div = ScalarField::new(nx, ny);
    for y in 1..ny.saturating_sub(1) {
        for x in 1..nx.saturating_sub(1) {
            let du_dx = (vel.u.get(x + 1, y) - vel.u.get(x - 1, y)) / (2.0 * dx);
            let dv_dy = (vel.v.get(x, y + 1) - vel.v.get(x, y - 1)) / (2.0 * dx);
            div.set(x, y, du_dx + dv_dy);
        }
    }
    div
}

/// Compute curl (vorticity) of 2D velocity: omega = dv/dx - du/dy.
pub fn curl(vel: &VectorField2D) -> ScalarField {
    let nx = vel.nx;
    let ny = vel.ny;
    let dx = vel.dx;
    let mut omega = ScalarField::new(nx, ny);
    for y in 1..ny.saturating_sub(1) {
        for x in 1..nx.saturating_sub(1) {
            let dv_dx = (vel.v.get(x + 1, y) - vel.v.get(x - 1, y)) / (2.0 * dx);
            let du_dy = (vel.u.get(x, y + 1) - vel.u.get(x, y - 1)) / (2.0 * dx);
            omega.set(x, y, dv_dx - du_dy);
        }
    }
    omega
}

// ── Boundary Enforcement ──────────────────────────────────────

/// Enforce no-slip (zero velocity) on all boundary cells.
pub fn enforce_no_slip(vel: &mut VectorField2D) {
    let nx = vel.nx;
    let ny = vel.ny;
    for y in 0..ny {
        vel.u.set(0, y, 0.0);
        vel.u.set(nx - 1, y, 0.0);
        vel.v.set(0, y, 0.0);
        vel.v.set(nx - 1, y, 0.0);
    }
    for x in 0..nx {
        vel.u.set(x, 0, 0.0);
        vel.u.set(x, ny - 1, 0.0);
        vel.v.set(x, 0, 0.0);
        vel.v.set(x, ny - 1, 0.0);
    }
}

/// Enforce free-slip on all boundaries (zero normal, free tangential).
pub fn enforce_free_slip(vel: &mut VectorField2D) {
    let nx = vel.nx;
    let ny = vel.ny;
    // Left/right: zero u (normal), copy v (tangential)
    for y in 0..ny {
        vel.u.set(0, y, 0.0);
        vel.u.set(nx - 1, y, 0.0);
        vel.v.set(0, y, vel.v.get(1, y));
        vel.v.set(nx - 1, y, vel.v.get(nx - 2, y));
    }
    // Top/bottom: zero v (normal), copy u (tangential)
    for x in 0..nx {
        vel.v.set(x, 0, 0.0);
        vel.v.set(x, ny - 1, 0.0);
        vel.u.set(x, 0, vel.u.get(x, 1));
        vel.u.set(x, ny - 1, vel.u.get(x, ny - 2));
    }
}

// ── Reynolds Number ───────────────────────────────────────────

/// Estimate Reynolds number: Re = U * L / nu.
pub fn reynolds_number(
    characteristic_velocity: f64,
    characteristic_length: f64,
    kinematic_viscosity: f64,
) -> f64 {
    if kinematic_viscosity <= 0.0 {
        return f64::INFINITY;
    }
    characteristic_velocity * characteristic_length / kinematic_viscosity
}

// ── Energy Dissipation Tracking ───────────────────────────────

/// Track energy dissipation over time.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyTracker {
    history: Vec<(f64, f64)>, // (time, kinetic_energy)
}

impl EnergyTracker {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    /// Record a kinetic energy sample at the given time.
    pub fn record(&mut self, time: f64, ke: f64) {
        self.history.push((time, ke));
    }

    /// Number of samples recorded.
    pub fn sample_count(&self) -> usize {
        self.history.len()
    }

    /// Instantaneous dissipation rate (backward difference).
    pub fn dissipation_rate(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let (t1, e1) = self.history[self.history.len() - 2];
        let (t2, e2) = self.history[self.history.len() - 1];
        let dt = t2 - t1;
        if dt.abs() < 1e-15 {
            return None;
        }
        Some(-(e2 - e1) / dt)
    }

    /// Total energy lost from first to last sample.
    pub fn total_energy_lost(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let first = self.history[0].1;
        let last = self.history[self.history.len() - 1].1;
        first - last
    }

    /// Average dissipation rate over all samples.
    pub fn average_dissipation_rate(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let total_time = self.history.last().unwrap().0 - self.history[0].0;
        if total_time.abs() < 1e-15 {
            return None;
        }
        Some(self.total_energy_lost() / total_time)
    }

    /// Get the full history.
    pub fn history(&self) -> &[(f64, f64)] {
        &self.history
    }
}

// ── Operator Splitting ────────────────────────────────────────

/// Full Navier-Stokes step config.
#[derive(Debug, Clone, PartialEq)]
pub struct NsStepConfig {
    pub kinematic_viscosity: f64,
    pub gravity: f64,
    pub dt: f64,
    pub diffusion_iters: usize,
    pub diffusion_tol: f64,
    pub use_maccormack: bool,
}

impl Default for NsStepConfig {
    fn default() -> Self {
        Self {
            kinematic_viscosity: 0.001,
            gravity: 0.0,
            dt: 0.01,
            diffusion_iters: 50,
            diffusion_tol: 1e-5,
            use_maccormack: false,
        }
    }
}

/// Perform one operator-splitting step: advect, diffuse, add forces.
/// (Pressure projection is in pressure_solver.rs.)
pub fn operator_split_step(
    vel: &mut VectorField2D,
    cfg: &NsStepConfig,
) -> Result<(), NavierStokesError> {
    let dt = cfg.dt;
    let dx = vel.dx;

    // 1. Advect u and v
    let vel_snapshot = vel.clone();
    if cfg.use_maccormack {
        vel.u = advect_maccormack(&vel.u, &vel_snapshot, dt)?;
        vel.v = advect_maccormack(&vel.v, &vel_snapshot, dt)?;
    } else {
        vel.u = advect_semi_lagrangian(&vel.u, &vel_snapshot, dt)?;
        vel.v = advect_semi_lagrangian(&vel.v, &vel_snapshot, dt)?;
    }

    // 2. Diffuse
    if cfg.kinematic_viscosity > 0.0 {
        diffuse_jacobi(&mut vel.u, cfg.kinematic_viscosity, dt, dx, cfg.diffusion_iters, cfg.diffusion_tol);
        diffuse_jacobi(&mut vel.v, cfg.kinematic_viscosity, dt, dx, cfg.diffusion_iters, cfg.diffusion_tol);
    }

    // 3. Body forces
    if cfg.gravity.abs() > 1e-15 {
        apply_gravity(vel, cfg.gravity, dt);
    }

    // 4. Check for NaN
    for val in &vel.u.data {
        if !val.is_finite() {
            return Err(NavierStokesError::Diverged("NaN in u field".into()));
        }
    }
    for val in &vel.v.data {
        if !val.is_finite() {
            return Err(NavierStokesError::Diverged("NaN in v field".into()));
        }
    }

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vel(nx: usize, ny: usize) -> VectorField2D {
        VectorField2D::new(nx, ny, 1.0 / nx as f64)
    }

    #[test]
    fn test_scalar_field_new() {
        let f = ScalarField::new(10, 10);
        assert_eq!(f.data.len(), 100);
        assert!((f.get(5, 5)).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_field_set_get() {
        let mut f = ScalarField::new(10, 10);
        f.set(3, 4, 7.0);
        assert!((f.get(3, 4) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_scalar_field_sample_exact() {
        let mut f = ScalarField::new(4, 4);
        f.set(2, 2, 5.0);
        assert!((f.sample(2.0, 2.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_field_sample_interpolated() {
        let mut f = ScalarField::new(4, 4);
        f.set(1, 1, 0.0);
        f.set(2, 1, 10.0);
        f.set(1, 2, 0.0);
        f.set(2, 2, 10.0);
        let val = f.sample(1.5, 1.5);
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field_max_speed() {
        let mut vel = make_vel(10, 10);
        vel.u.set(5, 5, 3.0);
        vel.v.set(5, 5, 4.0);
        assert!((vel.max_speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_field_kinetic_energy_zero() {
        let vel = make_vel(10, 10);
        assert!((vel.kinetic_energy()).abs() < 1e-12);
    }

    #[test]
    fn test_divergence_uniform_field() {
        let vel = make_vel(10, 10);
        let div = divergence(&vel);
        assert!(div.max_abs() < 1e-12);
    }

    #[test]
    fn test_curl_uniform_field() {
        let vel = make_vel(10, 10);
        let omega = curl(&vel);
        assert!(omega.max_abs() < 1e-12);
    }

    #[test]
    fn test_advect_semi_lagrangian_zero_vel() {
        let mut f = ScalarField::new(10, 10);
        f.set(5, 5, 1.0);
        let vel = make_vel(10, 10);
        let result = advect_semi_lagrangian(&f, &vel, 0.01).unwrap();
        assert!((result.get(5, 5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_advect_dimension_mismatch() {
        let f = ScalarField::new(10, 10);
        let vel = make_vel(8, 8);
        assert!(advect_semi_lagrangian(&f, &vel, 0.01).is_err());
    }

    #[test]
    fn test_maccormack_zero_vel() {
        let mut f = ScalarField::new(10, 10);
        f.set(5, 5, 2.0);
        let vel = make_vel(10, 10);
        let result = advect_maccormack(&f, &vel, 0.01).unwrap();
        assert!((result.get(5, 5) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_diffuse_jacobi_no_change_uniform() {
        let mut f = ScalarField::filled(10, 10, 5.0);
        let (iters, _) = diffuse_jacobi(&mut f, 0.01, 0.01, 0.1, 50, 1e-6);
        // Uniform field shouldn't change (Laplacian = 0)
        assert!((f.get(5, 5) - 5.0).abs() < 1e-4);
        assert!(iters <= 50);
    }

    #[test]
    fn test_apply_gravity() {
        let mut vel = make_vel(10, 10);
        vel.v.set(5, 5, 0.0);
        apply_gravity(&mut vel, 9.81, 0.1);
        assert!((vel.v.get(5, 5) - (-0.981)).abs() < 1e-10);
    }

    #[test]
    fn test_enforce_no_slip() {
        let mut vel = make_vel(10, 10);
        vel.u.set(0, 5, 5.0);
        vel.u.set(9, 5, 5.0);
        vel.v.set(5, 0, 5.0);
        enforce_no_slip(&mut vel);
        assert!((vel.u.get(0, 5)).abs() < 1e-12);
        assert!((vel.u.get(9, 5)).abs() < 1e-12);
        assert!((vel.v.get(5, 0)).abs() < 1e-12);
    }

    #[test]
    fn test_enforce_free_slip() {
        let mut vel = make_vel(10, 10);
        vel.u.set(0, 5, 5.0);
        vel.v.set(0, 5, 3.0);
        vel.v.set(1, 5, 3.0);
        enforce_free_slip(&mut vel);
        // Normal (u) at boundary = 0
        assert!((vel.u.get(0, 5)).abs() < 1e-12);
        // Tangential (v) at boundary copied from interior
        assert!((vel.v.get(0, 5) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_reynolds_number() {
        let re = reynolds_number(1.0, 1.0, 0.001);
        assert!((re - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_reynolds_number_zero_viscosity() {
        let re = reynolds_number(1.0, 1.0, 0.0);
        assert!(re.is_infinite());
    }

    #[test]
    fn test_energy_tracker_empty() {
        let tracker = EnergyTracker::new();
        assert_eq!(tracker.sample_count(), 0);
        assert!(tracker.dissipation_rate().is_none());
    }

    #[test]
    fn test_energy_tracker_record() {
        let mut tracker = EnergyTracker::new();
        tracker.record(0.0, 100.0);
        tracker.record(0.1, 90.0);
        assert_eq!(tracker.sample_count(), 2);
        let rate = tracker.dissipation_rate().unwrap();
        assert!((rate - 100.0).abs() < 1e-6); // (100 - 90) / 0.1
    }

    #[test]
    fn test_energy_tracker_total_lost() {
        let mut tracker = EnergyTracker::new();
        tracker.record(0.0, 100.0);
        tracker.record(0.5, 80.0);
        tracker.record(1.0, 60.0);
        assert!((tracker.total_energy_lost() - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_energy_tracker_average_rate() {
        let mut tracker = EnergyTracker::new();
        tracker.record(0.0, 100.0);
        tracker.record(1.0, 50.0);
        let avg = tracker.average_dissipation_rate().unwrap();
        assert!((avg - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_operator_split_step_zero_velocity() {
        let mut vel = make_vel(10, 10);
        let cfg = NsStepConfig::default();
        assert!(operator_split_step(&mut vel, &cfg).is_ok());
        assert!(vel.max_speed() < 1e-10);
    }

    #[test]
    fn test_operator_split_step_with_gravity() {
        let mut vel = make_vel(10, 10);
        let cfg = NsStepConfig {
            gravity: 9.81,
            dt: 0.01,
            ..Default::default()
        };
        operator_split_step(&mut vel, &cfg).unwrap();
        // Interior points should have negative v from gravity
        assert!(vel.v.get(5, 5) < 0.0);
    }

    #[test]
    fn test_scalar_field_l2_norm() {
        let mut f = ScalarField::new(2, 2);
        f.set(0, 0, 3.0);
        f.set(1, 0, 4.0);
        let norm = f.l2_norm();
        assert!((norm - 5.0).abs() < 1e-10);
    }
}
