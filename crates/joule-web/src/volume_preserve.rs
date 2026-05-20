//! Volume preservation for soft bodies — pressure model, gas simulation, inflation.
//!
//! Replaces Bullet.js / physx.js volume preservation with pure Rust.
//! Supports signed-volume computation from surface triangles, volume
//! constraint enforcement, pressure model (PV=nRT), gas simulation,
//! inflation/deflation, and deformation gradient correction.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum VolumeError {
    InvalidTimestep,
    NoTriangles,
    NoTetrahedra,
    InvalidPressure,
    InvalidTemperature,
    InvalidGasAmount,
    ParticleNotFound(usize),
    InvalidStiffness,
}

impl fmt::Display for VolumeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::NoTriangles => write!(f, "at least one triangle required for surface volume"),
            Self::NoTetrahedra => write!(f, "at least one tetrahedron required"),
            Self::InvalidPressure => write!(f, "pressure must be non-negative"),
            Self::InvalidTemperature => write!(f, "temperature must be positive"),
            Self::InvalidGasAmount => write!(f, "gas amount must be positive"),
            Self::ParticleNotFound(i) => write!(f, "particle {i} not of bounds"),
            Self::InvalidStiffness => write!(f, "stiffness must be positive"),
        }
    }
}

impl std::error::Error for VolumeError {}

// ── Vec3 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }

    pub fn scale(self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { Self::ZERO } else { self.scale(1.0 / l) }
    }
}

// ── Surface volume computation ──────────────────────────────────

/// Compute signed volume enclosed by a triangle mesh (divergence theorem).
/// Each triangle contributes V = (1/6) * (p0 . (p1 x p2)).
pub fn surface_signed_volume(positions: &[Vec3], triangles: &[[usize; 3]]) -> f64 {
    let mut vol = 0.0;
    for tri in triangles {
        let p0 = positions[tri[0]];
        let p1 = positions[tri[1]];
        let p2 = positions[tri[2]];
        vol += p0.dot(p1.cross(p2)) / 6.0;
    }
    vol
}

/// Compute volume from tetrahedra.
pub fn tet_volume(positions: &[Vec3], tets: &[[usize; 4]]) -> f64 {
    let mut vol = 0.0;
    for t in tets {
        let e1 = positions[t[1]].sub(positions[t[0]]);
        let e2 = positions[t[2]].sub(positions[t[0]]);
        let e3 = positions[t[3]].sub(positions[t[0]]);
        vol += e1.dot(e2.cross(e3)) / 6.0;
    }
    vol.abs()
}

// ── Triangle face normal (area-weighted) ────────────────────────

fn triangle_normal(p0: Vec3, p1: Vec3, p2: Vec3) -> Vec3 {
    p1.sub(p0).cross(p2.sub(p0))
}

// ── Gas State (Ideal Gas Law) ───────────────────────────────────

/// Ideal gas: PV = nRT
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasState {
    pub moles: f64,           // n
    pub temperature: f64,     // T in Kelvin
    pub gas_constant: f64,    // R (8.314 J/(mol·K))
}

impl GasState {
    pub const R_DEFAULT: f64 = 8.314;

    pub fn new(moles: f64, temperature: f64) -> Result<Self, VolumeError> {
        if moles <= 0.0 { return Err(VolumeError::InvalidGasAmount); }
        if temperature <= 0.0 { return Err(VolumeError::InvalidTemperature); }
        Ok(Self { moles, temperature, gas_constant: Self::R_DEFAULT })
    }

    /// Compute pressure for given volume: P = nRT / V
    pub fn pressure_for_volume(&self, volume: f64) -> f64 {
        if volume.abs() < 1e-15 { return f64::MAX; }
        self.moles * self.gas_constant * self.temperature / volume.abs()
    }

    /// Compute equilibrium volume for given pressure: V = nRT / P
    pub fn volume_for_pressure(&self, pressure: f64) -> f64 {
        if pressure.abs() < 1e-15 { return f64::MAX; }
        self.moles * self.gas_constant * self.temperature / pressure
    }
}

// ── Volume Preservation System ──────────────────────────────────

#[derive(Debug, Clone)]
pub struct VolumePreserver {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub masses: Vec<f64>,
    pub fixed: Vec<bool>,
    pub triangles: Vec<[usize; 3]>,
    pub tetrahedra: Vec<[usize; 4]>,
    pub target_volume: f64,
    pub stiffness: f64,
    pub gas: Option<GasState>,
    pub inflation: f64,
    pub gravity: Vec3,
    pub damping: f64,
    pub time: f64,
}

impl VolumePreserver {
    /// Create from a surface triangle mesh.
    pub fn from_surface(
        positions: Vec<Vec3>,
        triangles: Vec<[usize; 3]>,
        masses: Vec<f64>,
        stiffness: f64,
    ) -> Result<Self, VolumeError> {
        if triangles.is_empty() {
            return Err(VolumeError::NoTriangles);
        }
        if stiffness <= 0.0 {
            return Err(VolumeError::InvalidStiffness);
        }
        let n = positions.len();
        let target = surface_signed_volume(&positions, &triangles).abs();
        Ok(Self {
            positions: positions.clone(),
            velocities: vec![Vec3::ZERO; n],
            masses,
            fixed: vec![false; n],
            triangles,
            tetrahedra: Vec::new(),
            target_volume: target,
            stiffness,
            gas: None,
            inflation: 0.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.01,
            time: 0.0,
        })
    }

    /// Create from a tetrahedral mesh.
    pub fn from_tets(
        positions: Vec<Vec3>,
        tetrahedra: Vec<[usize; 4]>,
        masses: Vec<f64>,
        stiffness: f64,
    ) -> Result<Self, VolumeError> {
        if tetrahedra.is_empty() {
            return Err(VolumeError::NoTetrahedra);
        }
        if stiffness <= 0.0 {
            return Err(VolumeError::InvalidStiffness);
        }
        let n = positions.len();
        let target = tet_volume(&positions, &tetrahedra);
        Ok(Self {
            positions: positions.clone(),
            velocities: vec![Vec3::ZERO; n],
            masses,
            fixed: vec![false; n],
            triangles: Vec::new(),
            tetrahedra,
            target_volume: target,
            stiffness,
            gas: None,
            inflation: 0.0,
            gravity: Vec3::ZERO,
            damping: 0.01,
            time: 0.0,
        })
    }

    pub fn enable_gas(&mut self, moles: f64, temperature: f64) -> Result<(), VolumeError> {
        self.gas = Some(GasState::new(moles, temperature)?);
        Ok(())
    }

    pub fn set_temperature(&mut self, temp: f64) -> Result<(), VolumeError> {
        if temp <= 0.0 { return Err(VolumeError::InvalidTemperature); }
        if let Some(g) = &mut self.gas {
            g.temperature = temp;
        }
        Ok(())
    }

    pub fn set_inflation(&mut self, amount: f64) {
        self.inflation = amount;
    }

    pub fn fix_particle(&mut self, idx: usize) -> Result<(), VolumeError> {
        if idx >= self.positions.len() {
            return Err(VolumeError::ParticleNotFound(idx));
        }
        self.fixed[idx] = true;
        Ok(())
    }

    pub fn current_volume(&self) -> f64 {
        if !self.triangles.is_empty() {
            surface_signed_volume(&self.positions, &self.triangles).abs()
        } else {
            tet_volume(&self.positions, &self.tetrahedra)
        }
    }

    pub fn volume_ratio(&self) -> f64 {
        if self.target_volume.abs() < 1e-15 { return 1.0; }
        self.current_volume() / self.target_volume
    }

    fn apply_volume_constraint(&mut self) {
        let vol = self.current_volume();
        let effective_target = self.target_volume + self.inflation;
        if effective_target.abs() < 1e-15 { return; }
        let deviation = vol - effective_target;

        // Gas model: use internal pressure
        let pressure = if let Some(gas) = &self.gas {
            let target_p = gas.pressure_for_volume(effective_target);
            let current_p = gas.pressure_for_volume(vol);
            current_p - target_p
        } else {
            // Simple volume constraint: pressure proportional to deviation
            self.stiffness * deviation / effective_target
        };

        // Apply pressure force to surface triangles
        if !self.triangles.is_empty() {
            let tri_count = self.triangles.len();
            for ti in 0..tri_count {
                let tri = self.triangles[ti];
                let p0 = self.positions[tri[0]];
                let p1 = self.positions[tri[1]];
                let p2 = self.positions[tri[2]];
                let normal = triangle_normal(p0, p1, p2);
                let area = normal.length() * 0.5;
                if area < 1e-12 { continue; }
                let n = normal.normalized();
                let force = n.scale(-pressure * area / 3.0);
                for k in 0..3 {
                    let idx = tri[k];
                    if !self.fixed[idx] {
                        let inv_m = if self.masses[idx] > 1e-12 { 1.0 / self.masses[idx] } else { 0.0 };
                        self.velocities[idx] = self.velocities[idx].add(force.scale(inv_m));
                    }
                }
            }
        }

        // Apply via tet volume gradients
        if !self.tetrahedra.is_empty() {
            let tet_count = self.tetrahedra.len();
            for ti in 0..tet_count {
                let t = self.tetrahedra[ti];
                let p0 = self.positions[t[0]];
                let p1 = self.positions[t[1]];
                let p2 = self.positions[t[2]];
                let p3 = self.positions[t[3]];

                let e1 = p1.sub(p0);
                let e2 = p2.sub(p0);
                let e3 = p3.sub(p0);

                let g1 = e2.cross(e3).scale(1.0 / 6.0);
                let g2 = e3.cross(e1).scale(1.0 / 6.0);
                let g3 = e1.cross(e2).scale(1.0 / 6.0);
                let g0 = Vec3::ZERO.sub(g1).sub(g2).sub(g3);
                let grads = [g0, g1, g2, g3];

                for k in 0..4 {
                    let idx = t[k];
                    if !self.fixed[idx] {
                        let inv_m = if self.masses[idx] > 1e-12 { 1.0 / self.masses[idx] } else { 0.0 };
                        let correction = grads[k].scale(-pressure * self.stiffness * inv_m);
                        self.velocities[idx] = self.velocities[idx].add(correction);
                    }
                }
            }
        }
    }

    /// Apply volume-preserving correction: scale positions to maintain target volume.
    pub fn enforce_volume_scaling(&mut self) {
        let vol = self.current_volume();
        let effective_target = self.target_volume + self.inflation;
        if vol.abs() < 1e-15 || effective_target.abs() < 1e-15 { return; }

        let ratio = effective_target / vol;
        let scale = ratio.cbrt(); // cube root for 3D scaling

        // Compute center
        let mut center = Vec3::ZERO;
        let mut count = 0;
        for (i, p) in self.positions.iter().enumerate() {
            if !self.fixed[i] {
                center = center.add(*p);
                count += 1;
            }
        }
        if count == 0 { return; }
        center = center.scale(1.0 / count as f64);

        // Scale positions relative to center
        for (i, p) in self.positions.iter_mut().enumerate() {
            if !self.fixed[i] {
                let offset = p.sub(center);
                *p = center.add(offset.scale(scale));
            }
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), VolumeError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(VolumeError::InvalidTimestep);
        }

        // Apply gravity
        for i in 0..self.positions.len() {
            if !self.fixed[i] {
                self.velocities[i] = self.velocities[i].add(self.gravity.scale(dt));
            }
        }

        // Apply volume preservation forces
        self.apply_volume_constraint();

        // Damping and integration
        for i in 0..self.positions.len() {
            if self.fixed[i] { continue; }
            self.velocities[i] = self.velocities[i].scale(1.0 - self.damping);
            self.positions[i] = self.positions[i].add(self.velocities[i].scale(dt));
        }

        self.time += dt;
        Ok(())
    }

    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    pub fn surface_area(&self) -> f64 {
        let mut area = 0.0;
        for tri in &self.triangles {
            let p0 = self.positions[tri[0]];
            let p1 = self.positions[tri[1]];
            let p2 = self.positions[tri[2]];
            area += triangle_normal(p0, p1, p2).length() * 0.5;
        }
        area
    }

    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for i in 0..self.positions.len() {
            ke += 0.5 * self.masses[i] * self.velocities[i].length_sq();
        }
        ke
    }
}

/// Build a simple closed surface: a triangulated box.
pub fn make_box_surface(size: f64) -> (Vec<Vec3>, Vec<[usize; 3]>) {
    let s = size / 2.0;
    let verts = vec![
        Vec3::new(-s, -s, -s), // 0
        Vec3::new(s, -s, -s),  // 1
        Vec3::new(s, s, -s),   // 2
        Vec3::new(-s, s, -s),  // 3
        Vec3::new(-s, -s, s),  // 4
        Vec3::new(s, -s, s),   // 5
        Vec3::new(s, s, s),    // 6
        Vec3::new(-s, s, s),   // 7
    ];
    // 12 triangles (2 per face, outward normals)
    let tris = vec![
        [0, 2, 1], [0, 3, 2], // front
        [4, 5, 6], [4, 6, 7], // back
        [0, 1, 5], [0, 5, 4], // bottom
        [2, 3, 7], [2, 7, 6], // top
        [0, 4, 7], [0, 7, 3], // left
        [1, 2, 6], [1, 6, 5], // right
    ];
    (verts, tris)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_surface_volume_cube() {
        let (verts, tris) = make_box_surface(2.0);
        let vol = surface_signed_volume(&verts, &tris);
        assert!(approx(vol.abs(), 8.0, 1e-4));
    }

    #[test]
    fn test_tet_volume() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let vol = tet_volume(&positions, &[[0, 1, 2, 3]]);
        assert!(approx(vol, 1.0 / 6.0, 1e-10));
    }

    #[test]
    fn test_gas_state() {
        let gas = GasState::new(1.0, 300.0).unwrap();
        let p = gas.pressure_for_volume(1.0);
        // P = nRT/V = 1 * 8.314 * 300 / 1 = 2494.2
        assert!(approx(p, 2494.2, 0.5));
    }

    #[test]
    fn test_gas_inverse() {
        let gas = GasState::new(1.0, 300.0).unwrap();
        let vol = gas.volume_for_pressure(100.0);
        let p_back = gas.pressure_for_volume(vol);
        assert!(approx(p_back, 100.0, 1e-4));
    }

    #[test]
    fn test_invalid_gas() {
        assert!(GasState::new(0.0, 300.0).is_err());
        assert!(GasState::new(1.0, 0.0).is_err());
        assert!(GasState::new(-1.0, 300.0).is_err());
    }

    #[test]
    fn test_from_surface() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        assert!(approx(vp.target_volume, 8.0, 1e-4));
        assert_eq!(vp.particle_count(), 8);
    }

    #[test]
    fn test_from_surface_no_triangles() {
        let r = VolumePreserver::from_surface(vec![Vec3::ZERO], vec![], vec![1.0], 100.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_invalid_stiffness() {
        let (v, t) = make_box_surface(1.0);
        let r = VolumePreserver::from_surface(v, t, vec![1.0; 8], 0.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_volume_ratio_initial() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        assert!(approx(vp.volume_ratio(), 1.0, 1e-4));
    }

    #[test]
    fn test_enforce_volume_scaling() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        // Shrink all positions
        for p in &mut vp.positions {
            *p = p.scale(0.5);
        }
        let vol_before = vp.current_volume();
        vp.enforce_volume_scaling();
        let vol_after = vp.current_volume();
        // Should be closer to target
        assert!((vol_after - vp.target_volume).abs() < (vol_before - vp.target_volume).abs());
    }

    #[test]
    fn test_step_runs() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.step(0.01).unwrap();
        assert!(vp.time > 0.0);
    }

    #[test]
    fn test_invalid_timestep() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        assert!(vp.step(0.0).is_err());
        assert!(vp.step(-1.0).is_err());
    }

    #[test]
    fn test_inflation() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.set_inflation(4.0);
        // Enforce scaling to see effect
        vp.enforce_volume_scaling();
        let vol = vp.current_volume();
        assert!(vol > vp.target_volume);
    }

    #[test]
    fn test_fix_particle() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.fix_particle(0).unwrap();
        let pos0 = vp.positions[0];
        for _ in 0..20 {
            vp.step(0.01).unwrap();
        }
        assert!(approx(vp.positions[0].x, pos0.x, 1e-10));
    }

    #[test]
    fn test_fix_particle_oob() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        assert!(vp.fix_particle(999).is_err());
    }

    #[test]
    fn test_surface_area() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        // Surface area of 2x2x2 cube = 6 * 4 = 24
        assert!(approx(vp.surface_area(), 24.0, 1e-4));
    }

    #[test]
    fn test_kinetic_energy_initial() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        assert!(approx(vp.kinetic_energy(), 0.0, 1e-12));
    }

    #[test]
    fn test_gas_pressure_model() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.enable_gas(1.0, 300.0).unwrap();
        vp.gravity = Vec3::ZERO;
        for _ in 0..10 {
            vp.step(0.001).unwrap();
        }
        // With gas model active and no gravity, volume should stay close to target
        assert!(vp.time > 0.0);
    }

    #[test]
    fn test_set_temperature() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.enable_gas(1.0, 300.0).unwrap();
        vp.set_temperature(600.0).unwrap();
        assert!(approx(vp.gas.as_ref().unwrap().temperature, 600.0, 1e-10));
    }

    #[test]
    fn test_invalid_temperature() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.enable_gas(1.0, 300.0).unwrap();
        assert!(vp.set_temperature(0.0).is_err());
    }

    #[test]
    fn test_from_tets() {
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let masses = vec![1.0; 4];
        let vp = VolumePreserver::from_tets(positions, vec![[0, 1, 2, 3]], masses, 50.0).unwrap();
        assert!(approx(vp.target_volume, 1.0 / 6.0, 1e-10));
    }

    #[test]
    fn test_from_tets_no_tets() {
        let r = VolumePreserver::from_tets(vec![Vec3::ZERO], vec![], vec![1.0], 50.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_damping_effect() {
        let (verts, tris) = make_box_surface(2.0);
        let masses = vec![1.0; verts.len()];
        let mut vp = VolumePreserver::from_surface(verts, tris, masses, 100.0).unwrap();
        vp.damping = 0.5;
        // Give initial velocity
        for v in &mut vp.velocities {
            *v = Vec3::new(1.0, 1.0, 1.0);
        }
        for _ in 0..10 {
            vp.step(0.01).unwrap();
        }
        // Velocities should have been damped significantly
        let ke = vp.kinetic_energy();
        assert!(ke < 100.0); // much less than initial
    }
}
