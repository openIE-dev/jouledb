//! Gravitational field computation — field strength, potential, Lagrange points.
//!
//! Replaces SciPy gravity / astropy.coordinates with pure Rust.
//! Point-mass superposition, field sampling on grids, Lagrange point
//! computation, tidal forces, gravitational lensing, escape velocity,
//! Hill sphere, and Roche limit.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for gravity field computation.
#[derive(Debug, Clone, PartialEq)]
pub enum GravityFieldError {
    /// Mass must be positive.
    NonPositiveMass(f64),
    /// Distance must be positive.
    NonPositiveDistance(f64),
    /// Grid resolution must be at least 2.
    InvalidResolution(usize),
    /// Gravitational constant must be positive.
    NonPositiveG(f64),
    /// Lagrange point iteration did not converge.
    LagrangeNoConverge,
    /// No sources in the field.
    NoSources,
}

impl fmt::Display for GravityFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveMass(m) => write!(f, "mass must be positive, got {m}"),
            Self::NonPositiveDistance(d) => write!(f, "distance must be positive, got {d}"),
            Self::InvalidResolution(n) => write!(f, "resolution must be >= 2, got {n}"),
            Self::NonPositiveG(g) => write!(f, "G must be positive, got {g}"),
            Self::LagrangeNoConverge => write!(f, "Lagrange point computation did not converge"),
            Self::NoSources => write!(f, "no mass sources in field"),
        }
    }
}

impl std::error::Error for GravityFieldError {}

// ── Vec2 ────────────────────────────────────────────────────────

/// 2D vector for field sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn magnitude_sq(self) -> f64 {
        self.x * self.x + self.y * self.y
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y } }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y } }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s } }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

// ── Vec3 ────────────────────────────────────────────────────────

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn magnitude(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn magnitude_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// ── Point Mass Source ───────────────────────────────────────────

/// A point mass gravitational source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMass2D {
    pub position: Vec2,
    pub mass: f64,
}

/// A 3D point mass source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMass3D {
    pub position: Vec3,
    pub mass: f64,
}

// ── Gravity Field ───────────────────────────────────────────────

/// Gravitational field from a collection of point masses.
#[derive(Debug, Clone)]
pub struct GravityField2D {
    pub sources: Vec<PointMass2D>,
    pub g_constant: f64,
    pub softening: f64,
}

impl GravityField2D {
    pub fn new(g_constant: f64) -> Result<Self, GravityFieldError> {
        if g_constant <= 0.0 {
            return Err(GravityFieldError::NonPositiveG(g_constant));
        }
        Ok(Self { sources: Vec::new(), g_constant, softening: 0.0 })
    }

    pub fn with_softening(mut self, s: f64) -> Self {
        self.softening = s.max(0.0);
        self
    }

    pub fn add_source(&mut self, source: PointMass2D) -> Result<(), GravityFieldError> {
        if source.mass <= 0.0 {
            return Err(GravityFieldError::NonPositiveMass(source.mass));
        }
        self.sources.push(source);
        Ok(())
    }

    /// Gravitational field (acceleration) at a point due to all sources.
    pub fn field_at(&self, pos: Vec2) -> Vec2 {
        let eps2 = self.softening * self.softening;
        let mut acc = Vec2::ZERO;
        for s in &self.sources {
            let dx = s.position - pos;
            let r2 = dx.magnitude_sq() + eps2;
            let r = r2.sqrt();
            if r < 1e-30 {
                continue;
            }
            let inv_r3 = 1.0 / (r2 * r);
            acc += dx * (self.g_constant * s.mass * inv_r3);
        }
        acc
    }

    /// Gravitational potential at a point.
    pub fn potential_at(&self, pos: Vec2) -> f64 {
        let eps2 = self.softening * self.softening;
        let mut phi = 0.0;
        for s in &self.sources {
            let dx = s.position - pos;
            let r = (dx.magnitude_sq() + eps2).sqrt();
            if r < 1e-30 {
                continue;
            }
            phi -= self.g_constant * s.mass / r;
        }
        phi
    }

    /// Sample the field on a 2D grid. Returns (field_vectors, potential_values).
    pub fn sample_grid(
        &self,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        nx: usize,
        ny: usize,
    ) -> Result<(Vec<Vec<Vec2>>, Vec<Vec<f64>>), GravityFieldError> {
        if nx < 2 || ny < 2 {
            return Err(GravityFieldError::InvalidResolution(nx.min(ny)));
        }
        let dx = (x_max - x_min) / (nx - 1) as f64;
        let dy = (y_max - y_min) / (ny - 1) as f64;
        let mut fields = Vec::with_capacity(ny);
        let mut potentials = Vec::with_capacity(ny);
        for j in 0..ny {
            let mut row_f = Vec::with_capacity(nx);
            let mut row_p = Vec::with_capacity(nx);
            for i in 0..nx {
                let pos = Vec2::new(x_min + i as f64 * dx, y_min + j as f64 * dy);
                row_f.push(self.field_at(pos));
                row_p.push(self.potential_at(pos));
            }
            fields.push(row_f);
            potentials.push(row_p);
        }
        Ok((fields, potentials))
    }

    /// Field magnitude at a point.
    pub fn field_magnitude_at(&self, pos: Vec2) -> f64 {
        self.field_at(pos).magnitude()
    }
}

// ── 3D Field ────────────────────────────────────────────────────

/// 3D gravitational field.
#[derive(Debug, Clone)]
pub struct GravityField3D {
    pub sources: Vec<PointMass3D>,
    pub g_constant: f64,
    pub softening: f64,
}

impl GravityField3D {
    pub fn new(g_constant: f64) -> Result<Self, GravityFieldError> {
        if g_constant <= 0.0 {
            return Err(GravityFieldError::NonPositiveG(g_constant));
        }
        Ok(Self { sources: Vec::new(), g_constant, softening: 0.0 })
    }

    pub fn add_source(&mut self, source: PointMass3D) -> Result<(), GravityFieldError> {
        if source.mass <= 0.0 {
            return Err(GravityFieldError::NonPositiveMass(source.mass));
        }
        self.sources.push(source);
        Ok(())
    }

    /// Gravitational field at a 3D point.
    pub fn field_at(&self, pos: Vec3) -> Vec3 {
        let eps2 = self.softening * self.softening;
        let mut acc = Vec3::ZERO;
        for s in &self.sources {
            let dx = s.position - pos;
            let r2 = dx.magnitude_sq() + eps2;
            let r = r2.sqrt();
            if r < 1e-30 { continue; }
            let inv_r3 = 1.0 / (r2 * r);
            acc += dx * (self.g_constant * s.mass * inv_r3);
        }
        acc
    }

    /// Gravitational potential at a 3D point.
    pub fn potential_at(&self, pos: Vec3) -> f64 {
        let eps2 = self.softening * self.softening;
        let mut phi = 0.0;
        for s in &self.sources {
            let dx = s.position - pos;
            let r = (dx.magnitude_sq() + eps2).sqrt();
            if r < 1e-30 { continue; }
            phi -= self.g_constant * s.mass / r;
        }
        phi
    }
}

// ── Lagrange Points ─────────────────────────────────────────────

/// Lagrange point positions for a two-body system (restricted three-body).
/// Bodies at (-mu2*d, 0) and (mu1*d, 0) where d is separation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagrangePoints {
    pub l1: Vec2,
    pub l2: Vec2,
    pub l3: Vec2,
    pub l4: Vec2,
    pub l5: Vec2,
}

/// Compute Lagrange points for two bodies.
/// m1 is the primary (larger) mass at origin, m2 is secondary at distance d along +x.
pub fn lagrange_points(m1: f64, m2: f64, d: f64) -> Result<LagrangePoints, GravityFieldError> {
    if m1 <= 0.0 {
        return Err(GravityFieldError::NonPositiveMass(m1));
    }
    if m2 <= 0.0 {
        return Err(GravityFieldError::NonPositiveMass(m2));
    }
    if d <= 0.0 {
        return Err(GravityFieldError::NonPositiveDistance(d));
    }
    let mu = m2 / (m1 + m2); // mass ratio

    // L1: between bodies. Solve x - (1-mu)/(x+mu)^2 + mu/(x-1+mu)^2 = 0 (approx).
    // Use Hill approximation: L1 at d * (1 - (mu/3)^(1/3)) from primary.
    let hill = (mu / 3.0).powf(1.0 / 3.0);
    let l1_x = d * (1.0 - hill);

    // L2: beyond secondary. At d * (1 + (mu/3)^(1/3)).
    let l2_x = d * (1.0 + hill);

    // L3: opposite side. Approximate: -d * (1 + 5*mu/12).
    let l3_x = -d * (1.0 + 5.0 * mu / 12.0);

    // L4, L5: equilateral triangle points.
    let l4_x = d * (0.5 - mu);
    let l4_y = d * (3.0_f64.sqrt() / 2.0);
    let l5_y = -l4_y;

    Ok(LagrangePoints {
        l1: Vec2::new(l1_x, 0.0),
        l2: Vec2::new(l2_x, 0.0),
        l3: Vec2::new(l3_x, 0.0),
        l4: Vec2::new(l4_x, l4_y),
        l5: Vec2::new(l4_x, l5_y),
    })
}

// ── Tidal Force ─────────────────────────────────────────────────

/// Tidal acceleration at distance delta from the center of a body
/// at distance R from a source of mass M.
/// Approximate: a_tidal ~ 2*G*M*delta/R^3 (along radial direction).
pub fn tidal_acceleration(g: f64, source_mass: f64, distance: f64, delta: f64) -> f64 {
    2.0 * g * source_mass * delta / distance.powi(3)
}

// ── Gravitational Lensing ───────────────────────────────────────

/// Gravitational lensing deflection angle (weak field) for a point mass.
/// alpha = 4*G*M / (c^2 * b) where b is impact parameter.
/// We use natural units where c=1; user supplies G*M and b.
pub fn lensing_deflection(gm: f64, impact_parameter: f64, c: f64) -> f64 {
    if impact_parameter.abs() < 1e-30 {
        return 0.0;
    }
    4.0 * gm / (c * c * impact_parameter)
}

// ── Escape Velocity ─────────────────────────────────────────────

/// Escape velocity from a combined gravitational field at position `pos`.
pub fn escape_velocity_from_field(field: &GravityField2D, pos: Vec2) -> f64 {
    let phi = field.potential_at(pos);
    // v_esc = sqrt(-2 * phi) since phi < 0
    if phi >= 0.0 {
        return 0.0;
    }
    (-2.0 * phi).sqrt()
}

/// Escape velocity from a point mass.
pub fn escape_velocity(g: f64, mass: f64, distance: f64) -> f64 {
    (2.0 * g * mass / distance).sqrt()
}

// ── Hill Sphere ─────────────────────────────────────────────────

/// Hill sphere radius: r_H = a * (m / (3*M))^(1/3).
/// `a` = orbital distance, `m` = secondary mass, `big_m` = primary mass.
pub fn hill_sphere_radius(a: f64, m: f64, big_m: f64) -> f64 {
    a * (m / (3.0 * big_m)).powf(1.0 / 3.0)
}

// ── Roche Limit ─────────────────────────────────────────────────

/// Roche limit (rigid body): d = R_primary * (2 * rho_primary / rho_secondary)^(1/3).
/// For fluid body: d = 2.44 * R_primary * (rho_primary / rho_secondary)^(1/3).
pub fn roche_limit_rigid(r_primary: f64, rho_primary: f64, rho_secondary: f64) -> f64 {
    r_primary * (2.0 * rho_primary / rho_secondary).powf(1.0 / 3.0)
}

pub fn roche_limit_fluid(r_primary: f64, rho_primary: f64, rho_secondary: f64) -> f64 {
    2.44 * r_primary * (rho_primary / rho_secondary).powf(1.0 / 3.0)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn vec2_approx_eq(a: Vec2, b: Vec2, eps: f64) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps)
    }

    #[test]
    fn single_source_field() {
        let mut field = GravityField2D::new(1.0).unwrap();
        field.add_source(PointMass2D { position: Vec2::ZERO, mass: 1.0 }).unwrap();
        // At (1,0), field should point toward origin: (-G*M/r^2, 0) = (-1, 0)
        let f = field.field_at(Vec2::new(1.0, 0.0));
        assert!(approx_eq(f.x, -1.0, 1e-10));
        assert!(approx_eq(f.y, 0.0, 1e-10));
    }

    #[test]
    fn potential_single_source() {
        let mut field = GravityField2D::new(1.0).unwrap();
        field.add_source(PointMass2D { position: Vec2::ZERO, mass: 1.0 }).unwrap();
        let phi = field.potential_at(Vec2::new(2.0, 0.0));
        // phi = -G*M/r = -1/2 = -0.5
        assert!(approx_eq(phi, -0.5, 1e-10));
    }

    #[test]
    fn superposition_two_sources() {
        let mut field = GravityField2D::new(1.0).unwrap();
        field.add_source(PointMass2D { position: Vec2::new(-1.0, 0.0), mass: 1.0 }).unwrap();
        field.add_source(PointMass2D { position: Vec2::new(1.0, 0.0), mass: 1.0 }).unwrap();
        // At the midpoint (0,0), fields cancel in x.
        let f = field.field_at(Vec2::ZERO);
        assert!(approx_eq(f.x, 0.0, 1e-10));
    }

    #[test]
    fn field_invalid_g() {
        assert!(GravityField2D::new(-1.0).is_err());
        assert!(GravityField2D::new(0.0).is_err());
    }

    #[test]
    fn field_invalid_mass() {
        let mut field = GravityField2D::new(1.0).unwrap();
        assert!(field.add_source(PointMass2D { position: Vec2::ZERO, mass: -1.0 }).is_err());
    }

    #[test]
    fn grid_sampling() {
        let mut field = GravityField2D::new(1.0).unwrap().with_softening(0.1);
        field.add_source(PointMass2D { position: Vec2::ZERO, mass: 1.0 }).unwrap();
        let (fields, pots) = field.sample_grid(-2.0, 2.0, -2.0, 2.0, 5, 5).unwrap();
        assert_eq!(fields.len(), 5);
        assert_eq!(pots[0].len(), 5);
        // Potential should be negative everywhere.
        for row in &pots {
            for &p in row {
                assert!(p < 0.0);
            }
        }
    }

    #[test]
    fn grid_invalid_resolution() {
        let field = GravityField2D::new(1.0).unwrap();
        assert!(field.sample_grid(0.0, 1.0, 0.0, 1.0, 1, 5).is_err());
    }

    #[test]
    fn lagrange_l4_l5_symmetric() {
        let lp = lagrange_points(100.0, 1.0, 10.0).unwrap();
        assert!(approx_eq(lp.l4.y, -lp.l5.y, 1e-10));
        assert!(approx_eq(lp.l4.x, lp.l5.x, 1e-10));
    }

    #[test]
    fn lagrange_l1_between_bodies() {
        let lp = lagrange_points(100.0, 1.0, 10.0).unwrap();
        assert!(lp.l1.x > 0.0 && lp.l1.x < 10.0);
    }

    #[test]
    fn lagrange_l2_beyond_secondary() {
        let lp = lagrange_points(100.0, 1.0, 10.0).unwrap();
        assert!(lp.l2.x > 10.0);
    }

    #[test]
    fn lagrange_l3_opposite() {
        let lp = lagrange_points(100.0, 1.0, 10.0).unwrap();
        assert!(lp.l3.x < 0.0);
    }

    #[test]
    fn lagrange_invalid_mass() {
        assert!(lagrange_points(-1.0, 1.0, 10.0).is_err());
        assert!(lagrange_points(1.0, -1.0, 10.0).is_err());
    }

    #[test]
    fn tidal_acceleration_value() {
        let a = tidal_acceleration(1.0, 10.0, 5.0, 0.1);
        // 2 * 1 * 10 * 0.1 / 125 = 0.016
        assert!(approx_eq(a, 0.016, 1e-10));
    }

    #[test]
    fn lensing_deflection_value() {
        let alpha = lensing_deflection(1.0, 2.0, 1.0);
        // 4 * 1 / (1 * 2) = 2.0
        assert!(approx_eq(alpha, 2.0, 1e-10));
    }

    #[test]
    fn escape_velocity_value() {
        let v = escape_velocity(1.0, 1.0, 1.0);
        assert!(approx_eq(v, 2.0_f64.sqrt(), 1e-10));
    }

    #[test]
    fn escape_velocity_from_field_test() {
        let mut field = GravityField2D::new(1.0).unwrap();
        field.add_source(PointMass2D { position: Vec2::ZERO, mass: 1.0 }).unwrap();
        let v = escape_velocity_from_field(&field, Vec2::new(1.0, 0.0));
        // phi = -1, v_esc = sqrt(2)
        assert!(approx_eq(v, 2.0_f64.sqrt(), 1e-10));
    }

    #[test]
    fn hill_sphere() {
        // Earth-Sun: a~1AU, m_earth/m_sun ~ 3e-6 => r_H ~ 0.01 AU
        let rh = hill_sphere_radius(1.0, 3e-6, 1.0);
        assert!(approx_eq(rh, (1e-6_f64).powf(1.0 / 3.0), 1e-4));
    }

    #[test]
    fn roche_limit_rigid_value() {
        let d = roche_limit_rigid(1.0, 5.0, 2.5);
        // (2 * 5 / 2.5)^(1/3) = 4^(1/3) ≈ 1.587
        assert!(approx_eq(d, 4.0_f64.powf(1.0 / 3.0), 1e-6));
    }

    #[test]
    fn roche_limit_fluid_greater_than_rigid() {
        let rigid = roche_limit_rigid(1.0, 5.0, 3.0);
        let fluid = roche_limit_fluid(1.0, 5.0, 3.0);
        assert!(fluid > rigid);
    }

    #[test]
    fn field_3d_single_source() {
        let mut field = GravityField3D::new(1.0).unwrap();
        field.add_source(PointMass3D { position: Vec3::ZERO, mass: 1.0 }).unwrap();
        let f = field.field_at(Vec3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(f.x, -1.0, 1e-10));
        assert!(approx_eq(f.y, 0.0, 1e-10));
        assert!(approx_eq(f.z, 0.0, 1e-10));
    }

    #[test]
    fn field_3d_potential() {
        let mut field = GravityField3D::new(1.0).unwrap();
        field.add_source(PointMass3D { position: Vec3::ZERO, mass: 4.0 }).unwrap();
        let phi = field.potential_at(Vec3::new(2.0, 0.0, 0.0));
        assert!(approx_eq(phi, -2.0, 1e-10));
    }

    #[test]
    fn softening_smooths_field() {
        let mut field = GravityField2D::new(1.0).unwrap().with_softening(1.0);
        field.add_source(PointMass2D { position: Vec2::ZERO, mass: 1.0 }).unwrap();
        // At origin with softening, field should be zero.
        let f = field.field_at(Vec2::ZERO);
        assert!(vec2_approx_eq(f, Vec2::ZERO, 1e-10));
    }

    #[test]
    fn field_magnitude() {
        let mut field = GravityField2D::new(1.0).unwrap();
        field.add_source(PointMass2D { position: Vec2::ZERO, mass: 1.0 }).unwrap();
        let mag = field.field_magnitude_at(Vec2::new(2.0, 0.0));
        // |g| = G*M/r^2 = 1/4 = 0.25
        assert!(approx_eq(mag, 0.25, 1e-10));
    }
}
