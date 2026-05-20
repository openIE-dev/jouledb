//! Vector field operations for fluid simulation.
//!
//! Provides 2D and 3D velocity fields on regular grids with bilinear/trilinear
//! interpolation, finite-difference divergence/curl/gradient operators, RK4
//! streamline tracing, particle advection, field arithmetic (add, scale, lerp),
//! field energy computation, and arrow-field sampling for visualization.

use std::fmt;

// ── Errors ────────────────────────────────────────────────────

/// Velocity field errors.
#[derive(Debug, Clone, PartialEq)]
pub enum VelocityFieldError {
    /// Grid dimension issue.
    InvalidGrid(String),
    /// Position outside field domain.
    OutOfDomain { position: [f64; 3] },
    /// Dimension mismatch between fields.
    DimensionMismatch(String),
    /// Streamline tracing failure.
    TracingFailed(String),
}

impl fmt::Display for VelocityFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid(msg) => write!(f, "invalid grid: {msg}"),
            Self::OutOfDomain { position } => {
                write!(f, "position out of domain: ({:.4}, {:.4}, {:.4})", position[0], position[1], position[2])
            }
            Self::DimensionMismatch(msg) => write!(f, "dimension mismatch: {msg}"),
            Self::TracingFailed(msg) => write!(f, "tracing failed: {msg}"),
        }
    }
}

impl std::error::Error for VelocityFieldError {}

// ── 2D Velocity Field ─────────────────────────────────────────

/// A 2D velocity field on a regular grid.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocityField2D {
    /// Horizontal velocity component.
    pub u: Vec<f64>,
    /// Vertical velocity component.
    pub v: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    /// Grid spacing.
    pub dx: f64,
    /// Domain origin.
    pub origin_x: f64,
    pub origin_y: f64,
}

impl VelocityField2D {
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        let n = nx * ny;
        Self {
            u: vec![0.0; n],
            v: vec![0.0; n],
            nx, ny, dx,
            origin_x: 0.0,
            origin_y: 0.0,
        }
    }

    pub fn with_origin(mut self, ox: f64, oy: f64) -> Self {
        self.origin_x = ox;
        self.origin_y = oy;
        self
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    pub fn get_u(&self, x: usize, y: usize) -> f64 {
        if x < self.nx && y < self.ny { self.u[self.idx(x, y)] } else { 0.0 }
    }

    pub fn get_v(&self, x: usize, y: usize) -> f64 {
        if x < self.nx && y < self.ny { self.v[self.idx(x, y)] } else { 0.0 }
    }

    pub fn set_u(&mut self, x: usize, y: usize, val: f64) {
        if x < self.nx && y < self.ny { let i = self.idx(x, y); self.u[i] = val; }
    }

    pub fn set_v(&mut self, x: usize, y: usize, val: f64) {
        if x < self.nx && y < self.ny { let i = self.idx(x, y); self.v[i] = val; }
    }

    pub fn set_velocity(&mut self, x: usize, y: usize, ux: f64, vy: f64) {
        self.set_u(x, y, ux);
        self.set_v(x, y, vy);
    }

    /// Speed at a grid point.
    pub fn speed_at(&self, x: usize, y: usize) -> f64 {
        let uu = self.get_u(x, y);
        let vv = self.get_v(x, y);
        (uu * uu + vv * vv).sqrt()
    }

    /// Maximum speed in the field.
    pub fn max_speed(&self) -> f64 {
        let mut max_sq = 0.0_f64;
        for i in 0..self.u.len() {
            max_sq = max_sq.max(self.u[i] * self.u[i] + self.v[i] * self.v[i]);
        }
        max_sq.sqrt()
    }

    /// Convert world position to fractional grid coordinates.
    fn world_to_grid(&self, wx: f64, wy: f64) -> (f64, f64) {
        ((wx - self.origin_x) / self.dx, (wy - self.origin_y) / self.dx)
    }

    /// Bilinear interpolation of u-component at fractional grid coords.
    fn bilinear_u(&self, fx: f64, fy: f64) -> f64 {
        let x0 = (fx.floor() as isize).clamp(0, self.nx as isize - 1) as usize;
        let y0 = (fy.floor() as isize).clamp(0, self.ny as isize - 1) as usize;
        let x1 = (x0 + 1).min(self.nx - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let sx = (fx - x0 as f64).clamp(0.0, 1.0);
        let sy = (fy - y0 as f64).clamp(0.0, 1.0);
        self.get_u(x0, y0) * (1.0 - sx) * (1.0 - sy)
            + self.get_u(x1, y0) * sx * (1.0 - sy)
            + self.get_u(x0, y1) * (1.0 - sx) * sy
            + self.get_u(x1, y1) * sx * sy
    }

    fn bilinear_v(&self, fx: f64, fy: f64) -> f64 {
        let x0 = (fx.floor() as isize).clamp(0, self.nx as isize - 1) as usize;
        let y0 = (fy.floor() as isize).clamp(0, self.ny as isize - 1) as usize;
        let x1 = (x0 + 1).min(self.nx - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let sx = (fx - x0 as f64).clamp(0.0, 1.0);
        let sy = (fy - y0 as f64).clamp(0.0, 1.0);
        self.get_v(x0, y0) * (1.0 - sx) * (1.0 - sy)
            + self.get_v(x1, y0) * sx * (1.0 - sy)
            + self.get_v(x0, y1) * (1.0 - sx) * sy
            + self.get_v(x1, y1) * sx * sy
    }

    /// Sample velocity at an arbitrary world position (bilinear interpolation).
    pub fn sample(&self, wx: f64, wy: f64) -> (f64, f64) {
        let (gx, gy) = self.world_to_grid(wx, wy);
        (self.bilinear_u(gx, gy), self.bilinear_v(gx, gy))
    }

    /// Compute divergence at grid point (central differences).
    pub fn divergence_at(&self, x: usize, y: usize) -> f64 {
        if x == 0 || x >= self.nx - 1 || y == 0 || y >= self.ny - 1 {
            return 0.0;
        }
        let du_dx = (self.get_u(x + 1, y) - self.get_u(x - 1, y)) / (2.0 * self.dx);
        let dv_dy = (self.get_v(x, y + 1) - self.get_v(x, y - 1)) / (2.0 * self.dx);
        du_dx + dv_dy
    }

    /// Compute full divergence field.
    pub fn divergence_field(&self) -> Vec<f64> {
        let mut div = vec![0.0; self.nx * self.ny];
        for y in 1..self.ny.saturating_sub(1) {
            for x in 1..self.nx.saturating_sub(1) {
                div[y * self.nx + x] = self.divergence_at(x, y);
            }
        }
        div
    }

    /// Compute curl (vorticity) at grid point (scalar in 2D).
    pub fn curl_at(&self, x: usize, y: usize) -> f64 {
        if x == 0 || x >= self.nx - 1 || y == 0 || y >= self.ny - 1 {
            return 0.0;
        }
        let dv_dx = (self.get_v(x + 1, y) - self.get_v(x - 1, y)) / (2.0 * self.dx);
        let du_dy = (self.get_u(x, y + 1) - self.get_u(x, y - 1)) / (2.0 * self.dx);
        dv_dx - du_dy
    }

    /// Compute full vorticity field.
    pub fn curl_field(&self) -> Vec<f64> {
        let mut omega = vec![0.0; self.nx * self.ny];
        for y in 1..self.ny.saturating_sub(1) {
            for x in 1..self.nx.saturating_sub(1) {
                omega[y * self.nx + x] = self.curl_at(x, y);
            }
        }
        omega
    }

    /// Gradient of a scalar field at grid point (central differences).
    pub fn gradient_scalar(scalar: &[f64], nx: usize, ny: usize, dx: f64, x: usize, y: usize) -> (f64, f64) {
        if x == 0 || x >= nx - 1 || y == 0 || y >= ny - 1 {
            return (0.0, 0.0);
        }
        let ds_dx = (scalar[y * nx + x + 1] - scalar[y * nx + x - 1]) / (2.0 * dx);
        let ds_dy = (scalar[(y + 1) * nx + x] - scalar[(y - 1) * nx + x]) / (2.0 * dx);
        (ds_dx, ds_dy)
    }

    /// Trace a streamline starting from world position using RK4 integration.
    pub fn trace_streamline(
        &self,
        start_x: f64,
        start_y: f64,
        dt: f64,
        max_steps: usize,
    ) -> Vec<(f64, f64)> {
        let mut path = Vec::with_capacity(max_steps + 1);
        let mut x = start_x;
        let mut y = start_y;
        path.push((x, y));

        let domain_min_x = self.origin_x;
        let domain_min_y = self.origin_y;
        let domain_max_x = self.origin_x + (self.nx - 1) as f64 * self.dx;
        let domain_max_y = self.origin_y + (self.ny - 1) as f64 * self.dx;

        for _ in 0..max_steps {
            // RK4 stages
            let (k1u, k1v) = self.sample(x, y);
            let (k2u, k2v) = self.sample(x + 0.5 * dt * k1u, y + 0.5 * dt * k1v);
            let (k3u, k3v) = self.sample(x + 0.5 * dt * k2u, y + 0.5 * dt * k2v);
            let (k4u, k4v) = self.sample(x + dt * k3u, y + dt * k3v);

            x += dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            y += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);

            // Stop if out of domain
            if x < domain_min_x || x > domain_max_x || y < domain_min_y || y > domain_max_y {
                break;
            }

            // Stop if velocity is negligible
            let (su, sv) = self.sample(x, y);
            if su * su + sv * sv < 1e-20 {
                break;
            }

            path.push((x, y));
        }
        path
    }

    /// Advect a set of particles through the field using RK4.
    pub fn advect_particles(
        &self,
        particles: &mut [(f64, f64)],
        dt: f64,
    ) {
        for p in particles.iter_mut() {
            let (k1u, k1v) = self.sample(p.0, p.1);
            let (k2u, k2v) = self.sample(p.0 + 0.5 * dt * k1u, p.1 + 0.5 * dt * k1v);
            let (k3u, k3v) = self.sample(p.0 + 0.5 * dt * k2u, p.1 + 0.5 * dt * k2v);
            let (k4u, k4v) = self.sample(p.0 + dt * k3u, p.1 + dt * k3v);

            p.0 += dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
            p.1 += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }
    }

    /// Add another field to this field (element-wise).
    pub fn add(&mut self, other: &VelocityField2D) -> Result<(), VelocityFieldError> {
        if self.nx != other.nx || self.ny != other.ny {
            return Err(VelocityFieldError::DimensionMismatch(
                format!("{}x{} vs {}x{}", self.nx, self.ny, other.nx, other.ny),
            ));
        }
        for i in 0..self.u.len() {
            self.u[i] += other.u[i];
            self.v[i] += other.v[i];
        }
        Ok(())
    }

    /// Scale all velocities by a factor.
    pub fn scale(&mut self, factor: f64) {
        for val in &mut self.u { *val *= factor; }
        for val in &mut self.v { *val *= factor; }
    }

    /// Linear interpolation between this field and another.
    pub fn lerp(&self, other: &VelocityField2D, t: f64) -> Result<VelocityField2D, VelocityFieldError> {
        if self.nx != other.nx || self.ny != other.ny {
            return Err(VelocityFieldError::DimensionMismatch(
                format!("{}x{} vs {}x{}", self.nx, self.ny, other.nx, other.ny),
            ));
        }
        let mut result = VelocityField2D::new(self.nx, self.ny, self.dx)
            .with_origin(self.origin_x, self.origin_y);
        for i in 0..self.u.len() {
            result.u[i] = self.u[i] * (1.0 - t) + other.u[i] * t;
            result.v[i] = self.v[i] * (1.0 - t) + other.v[i] * t;
        }
        Ok(result)
    }

    /// Total kinetic energy: 0.5 * sum(u^2 + v^2) * dx^2.
    pub fn energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.u.len() {
            e += self.u[i] * self.u[i] + self.v[i] * self.v[i];
        }
        0.5 * e * self.dx * self.dx
    }

    /// Sample an arrow field for visualization at given spacing.
    pub fn arrow_field(&self, spacing: usize) -> Vec<Arrow2D> {
        let mut arrows = Vec::new();
        for y in (0..self.ny).step_by(spacing.max(1)) {
            for x in (0..self.nx).step_by(spacing.max(1)) {
                let wx = self.origin_x + x as f64 * self.dx;
                let wy = self.origin_y + y as f64 * self.dx;
                let ux = self.get_u(x, y);
                let vy = self.get_v(x, y);
                let mag = (ux * ux + vy * vy).sqrt();
                arrows.push(Arrow2D { x: wx, y: wy, dx: ux, dy: vy, magnitude: mag });
            }
        }
        arrows
    }
}

/// An arrow for visualization.
#[derive(Debug, Clone, PartialEq)]
pub struct Arrow2D {
    pub x: f64,
    pub y: f64,
    pub dx: f64,
    pub dy: f64,
    pub magnitude: f64,
}

// ── 3D Velocity Field ─────────────────────────────────────────

/// A 3D velocity field on a regular grid.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocityField3D {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
}

impl VelocityField3D {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64) -> Self {
        let n = nx * ny * nz;
        Self {
            u: vec![0.0; n], v: vec![0.0; n], w: vec![0.0; n],
            nx, ny, nz, dx,
        }
    }

    fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.ny * self.nx + y * self.nx + x
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> (f64, f64, f64) {
        if x < self.nx && y < self.ny && z < self.nz {
            let i = self.idx(x, y, z);
            (self.u[i], self.v[i], self.w[i])
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, ux: f64, vy: f64, wz: f64) {
        if x < self.nx && y < self.ny && z < self.nz {
            let i = self.idx(x, y, z);
            self.u[i] = ux;
            self.v[i] = vy;
            self.w[i] = wz;
        }
    }

    /// Trilinear interpolation at fractional grid coordinates.
    pub fn sample(&self, fx: f64, fy: f64, fz: f64) -> (f64, f64, f64) {
        let x0 = (fx.floor() as isize).clamp(0, self.nx as isize - 1) as usize;
        let y0 = (fy.floor() as isize).clamp(0, self.ny as isize - 1) as usize;
        let z0 = (fz.floor() as isize).clamp(0, self.nz as isize - 1) as usize;
        let x1 = (x0 + 1).min(self.nx - 1);
        let y1 = (y0 + 1).min(self.ny - 1);
        let z1 = (z0 + 1).min(self.nz - 1);
        let sx = (fx - x0 as f64).clamp(0.0, 1.0);
        let sy = (fy - y0 as f64).clamp(0.0, 1.0);
        let sz = (fz - z0 as f64).clamp(0.0, 1.0);

        let mut result = (0.0, 0.0, 0.0);
        let corners = [
            (x0, y0, z0, (1.0 - sx) * (1.0 - sy) * (1.0 - sz)),
            (x1, y0, z0, sx * (1.0 - sy) * (1.0 - sz)),
            (x0, y1, z0, (1.0 - sx) * sy * (1.0 - sz)),
            (x1, y1, z0, sx * sy * (1.0 - sz)),
            (x0, y0, z1, (1.0 - sx) * (1.0 - sy) * sz),
            (x1, y0, z1, sx * (1.0 - sy) * sz),
            (x0, y1, z1, (1.0 - sx) * sy * sz),
            (x1, y1, z1, sx * sy * sz),
        ];

        for (cx, cy, cz, weight) in corners {
            let (uu, vv, ww) = self.get(cx, cy, cz);
            result.0 += uu * weight;
            result.1 += vv * weight;
            result.2 += ww * weight;
        }
        result
    }

    /// Divergence at a grid point.
    pub fn divergence_at(&self, x: usize, y: usize, z: usize) -> f64 {
        if x == 0 || x >= self.nx - 1 || y == 0 || y >= self.ny - 1 || z == 0 || z >= self.nz - 1 {
            return 0.0;
        }
        let dx2 = 2.0 * self.dx;
        let du_dx = (self.get(x + 1, y, z).0 - self.get(x - 1, y, z).0) / dx2;
        let dv_dy = (self.get(x, y + 1, z).1 - self.get(x, y - 1, z).1) / dx2;
        let dw_dz = (self.get(x, y, z + 1).2 - self.get(x, y, z - 1).2) / dx2;
        du_dx + dv_dy + dw_dz
    }

    /// Curl (vorticity vector) at a grid point.
    pub fn curl_at(&self, x: usize, y: usize, z: usize) -> (f64, f64, f64) {
        if x == 0 || x >= self.nx - 1 || y == 0 || y >= self.ny - 1 || z == 0 || z >= self.nz - 1 {
            return (0.0, 0.0, 0.0);
        }
        let dx2 = 2.0 * self.dx;
        let dw_dy = (self.get(x, y + 1, z).2 - self.get(x, y - 1, z).2) / dx2;
        let dv_dz = (self.get(x, y, z + 1).1 - self.get(x, y, z - 1).1) / dx2;
        let du_dz = (self.get(x, y, z + 1).0 - self.get(x, y, z - 1).0) / dx2;
        let dw_dx = (self.get(x + 1, y, z).2 - self.get(x - 1, y, z).2) / dx2;
        let dv_dx = (self.get(x + 1, y, z).1 - self.get(x - 1, y, z).1) / dx2;
        let du_dy = (self.get(x, y + 1, z).0 - self.get(x, y - 1, z).0) / dx2;

        (dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy)
    }

    /// Total kinetic energy.
    pub fn energy(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.u.len() {
            e += self.u[i] * self.u[i] + self.v[i] * self.v[i] + self.w[i] * self.w[i];
        }
        0.5 * e * self.dx * self.dx * self.dx
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_2d(ux: f64, vy: f64) -> VelocityField2D {
        let mut f = VelocityField2D::new(10, 10, 0.1);
        for y in 0..10 {
            for x in 0..10 {
                f.set_velocity(x, y, ux, vy);
            }
        }
        f
    }

    #[test]
    fn test_field2d_new() {
        let f = VelocityField2D::new(10, 10, 0.1);
        assert_eq!(f.u.len(), 100);
        assert!((f.max_speed()).abs() < 1e-12);
    }

    #[test]
    fn test_field2d_set_get() {
        let mut f = VelocityField2D::new(10, 10, 0.1);
        f.set_velocity(3, 4, 1.0, 2.0);
        assert!((f.get_u(3, 4) - 1.0).abs() < 1e-12);
        assert!((f.get_v(3, 4) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_field2d_speed() {
        let mut f = VelocityField2D::new(10, 10, 0.1);
        f.set_velocity(5, 5, 3.0, 4.0);
        assert!((f.speed_at(5, 5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_max_speed() {
        let mut f = VelocityField2D::new(10, 10, 0.1);
        f.set_velocity(2, 3, 3.0, 4.0);
        assert!((f.max_speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_sample_at_grid_point() {
        let mut f = VelocityField2D::new(10, 10, 0.1);
        f.set_velocity(5, 5, 2.0, 3.0);
        let (su, sv) = f.sample(0.5, 0.5);
        assert!((su - 2.0).abs() < 1e-10);
        assert!((sv - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_sample_interpolated() {
        let f = uniform_2d(1.0, 0.0);
        let (su, sv) = f.sample(0.35, 0.35);
        assert!((su - 1.0).abs() < 1e-10);
        assert!((sv).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_divergence_uniform() {
        let f = uniform_2d(1.0, 0.0);
        // Uniform field has zero divergence
        assert!(f.divergence_at(5, 5).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_curl_uniform() {
        let f = uniform_2d(1.0, 0.0);
        assert!(f.curl_at(5, 5).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_divergence_field() {
        let f = uniform_2d(1.0, 1.0);
        let div = f.divergence_field();
        for y in 1..9 {
            for x in 1..9 {
                assert!(div[y * 10 + x].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_field2d_curl_rotation() {
        // v = (−y, x) is a solid rotation with uniform positive vorticity
        let mut f = VelocityField2D::new(10, 10, 0.1);
        for y in 0..10 {
            for x in 0..10 {
                let wx = x as f64 * 0.1;
                let wy = y as f64 * 0.1;
                f.set_velocity(x, y, -wy, wx);
            }
        }
        let omega = f.curl_at(5, 5);
        // Should be ~2.0 (d(x)/dx + d(-(-y))/dy = 1 + 1 = 2)
        assert!((omega - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_streamline_stationary() {
        let f = VelocityField2D::new(10, 10, 0.1);
        let path = f.trace_streamline(0.5, 0.5, 0.01, 10);
        // Stationary field: streamline stays at start
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_streamline_uniform() {
        let f = uniform_2d(1.0, 0.0);
        let path = f.trace_streamline(0.0, 0.5, 0.01, 50);
        assert!(path.len() > 1);
        // Should move to the right
        let last = path.last().unwrap();
        assert!(last.0 > 0.0);
    }

    #[test]
    fn test_advect_particles() {
        let f = uniform_2d(1.0, 0.0);
        let mut particles = vec![(0.3, 0.3), (0.5, 0.5)];
        f.advect_particles(&mut particles, 0.01);
        assert!(particles[0].0 > 0.3);
        assert!(particles[1].0 > 0.5);
    }

    #[test]
    fn test_field2d_add() {
        let mut a = uniform_2d(1.0, 0.0);
        let b = uniform_2d(0.0, 1.0);
        a.add(&b).unwrap();
        assert!((a.get_u(5, 5) - 1.0).abs() < 1e-12);
        assert!((a.get_v(5, 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_field2d_add_mismatch() {
        let mut a = VelocityField2D::new(10, 10, 0.1);
        let b = VelocityField2D::new(8, 8, 0.1);
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_field2d_scale() {
        let mut f = uniform_2d(2.0, 3.0);
        f.scale(0.5);
        assert!((f.get_u(5, 5) - 1.0).abs() < 1e-12);
        assert!((f.get_v(5, 5) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_field2d_lerp() {
        let a = uniform_2d(0.0, 0.0);
        let b = uniform_2d(10.0, 10.0);
        let c = a.lerp(&b, 0.5).unwrap();
        assert!((c.get_u(5, 5) - 5.0).abs() < 1e-10);
        assert!((c.get_v(5, 5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_field2d_energy_zero() {
        let f = VelocityField2D::new(10, 10, 0.1);
        assert!((f.energy()).abs() < 1e-12);
    }

    #[test]
    fn test_field2d_energy_positive() {
        let f = uniform_2d(1.0, 0.0);
        assert!(f.energy() > 0.0);
    }

    #[test]
    fn test_arrow_field() {
        let f = uniform_2d(1.0, 0.0);
        let arrows = f.arrow_field(2);
        assert!(!arrows.is_empty());
        for a in &arrows {
            assert!((a.magnitude - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_field3d_new() {
        let f = VelocityField3D::new(4, 4, 4, 0.25);
        assert_eq!(f.u.len(), 64);
    }

    #[test]
    fn test_field3d_set_get() {
        let mut f = VelocityField3D::new(4, 4, 4, 0.25);
        f.set(1, 2, 3, 1.0, 2.0, 3.0);
        let (uu, vv, ww) = f.get(1, 2, 3);
        assert!((uu - 1.0).abs() < 1e-12);
        assert!((vv - 2.0).abs() < 1e-12);
        assert!((ww - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_field3d_trilinear() {
        let mut f = VelocityField3D::new(4, 4, 4, 0.25);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    f.set(x, y, z, 1.0, 0.0, 0.0);
                }
            }
        }
        let (uu, vv, ww) = f.sample(1.5, 1.5, 1.5);
        assert!((uu - 1.0).abs() < 1e-10);
        assert!((vv).abs() < 1e-10);
        assert!((ww).abs() < 1e-10);
    }

    #[test]
    fn test_field3d_divergence_uniform() {
        let mut f = VelocityField3D::new(6, 6, 6, 1.0);
        for z in 0..6 {
            for y in 0..6 {
                for x in 0..6 {
                    f.set(x, y, z, 1.0, 0.0, 0.0);
                }
            }
        }
        assert!(f.divergence_at(3, 3, 3).abs() < 1e-10);
    }

    #[test]
    fn test_field3d_curl_zero_uniform() {
        let mut f = VelocityField3D::new(6, 6, 6, 1.0);
        for z in 0..6 {
            for y in 0..6 {
                for x in 0..6 {
                    f.set(x, y, z, 1.0, 2.0, 3.0);
                }
            }
        }
        let (cx, cy, cz) = f.curl_at(3, 3, 3);
        assert!(cx.abs() < 1e-10);
        assert!(cy.abs() < 1e-10);
        assert!(cz.abs() < 1e-10);
    }

    #[test]
    fn test_gradient_scalar() {
        let mut scalar = vec![0.0; 100];
        // Linear ramp in x: scalar = x * 0.1
        for y in 0..10 {
            for x in 0..10 {
                scalar[y * 10 + x] = x as f64 * 0.1;
            }
        }
        let (gx, gy) = VelocityField2D::gradient_scalar(&scalar, 10, 10, 0.1, 5, 5);
        assert!((gx - 1.0).abs() < 1e-10);
        assert!(gy.abs() < 1e-10);
    }
}
