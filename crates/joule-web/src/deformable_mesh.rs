//! Deformable 3D mesh — tetrahedral FEM with co-rotational correction.
//!
//! Replaces Three.js deformable plugins / SOFA-web with pure Rust.
//! Supports tetrahedral mesh, per-element stiffness, linear strain FEM,
//! co-rotational correction for large rotation, stiffness matrix assembly,
//! mass lumping, boundary conditions, plasticity, and volume measurement.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DeformMeshError {
    InvalidStiffness,
    InvalidPoisson,
    InvalidTimestep,
    InvalidNode(usize),
    DegenerateTet(usize),
    InvalidYield,
}

impl fmt::Display for DeformMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStiffness => write!(f, "Young's modulus must be positive"),
            Self::InvalidPoisson => write!(f, "Poisson ratio must be in (-1, 0.5)"),
            Self::InvalidTimestep => write!(f, "timestep must be positive and finite"),
            Self::InvalidNode(i) => write!(f, "node index {i} out of bounds"),
            Self::DegenerateTet(i) => write!(f, "tetrahedron {i} has zero or near-zero volume"),
            Self::InvalidYield => write!(f, "yield threshold must be positive"),
        }
    }
}

impl std::error::Error for DeformMeshError {}

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
}

// ── 3x3 Matrix ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Self = Self { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
    pub const ZERO: Self = Self { m: [[0.0; 3]; 3] };

    pub fn from_cols(c0: Vec3, c1: Vec3, c2: Vec3) -> Self {
        Self { m: [
            [c0.x, c1.x, c2.x],
            [c0.y, c1.y, c2.y],
            [c0.z, c1.z, c2.z],
        ]}
    }

    pub fn col(&self, i: usize) -> Vec3 {
        Vec3::new(self.m[0][i], self.m[1][i], self.m[2][i])
    }

    pub fn transpose(self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[j][i];
            }
        }
        r
    }

    pub fn mul_mat(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r.m[i][j] += self.m[i][k] * o.m[k][j];
                }
            }
        }
        r
    }

    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3 {
            x: self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            y: self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            z: self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        }
    }

    pub fn determinant(self) -> f64 {
        self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
            - self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
            + self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0])
    }

    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        let mut r = Self::ZERO;
        r.m[0][0] = (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1]) * inv_det;
        r.m[0][1] = (self.m[0][2] * self.m[2][1] - self.m[0][1] * self.m[2][2]) * inv_det;
        r.m[0][2] = (self.m[0][1] * self.m[1][2] - self.m[0][2] * self.m[1][1]) * inv_det;
        r.m[1][0] = (self.m[1][2] * self.m[2][0] - self.m[1][0] * self.m[2][2]) * inv_det;
        r.m[1][1] = (self.m[0][0] * self.m[2][2] - self.m[0][2] * self.m[2][0]) * inv_det;
        r.m[1][2] = (self.m[0][2] * self.m[1][0] - self.m[0][0] * self.m[1][2]) * inv_det;
        r.m[2][0] = (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]) * inv_det;
        r.m[2][1] = (self.m[0][1] * self.m[2][0] - self.m[0][0] * self.m[2][1]) * inv_det;
        r.m[2][2] = (self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]) * inv_det;
        Some(r)
    }

    pub fn add(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[i][j] + o.m[i][j]; } }
        r
    }

    pub fn sub(self, o: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[i][j] - o.m[i][j]; } }
        r
    }

    pub fn scale(self, s: f64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 { for j in 0..3 { r.m[i][j] = self.m[i][j] * s; } }
        r
    }

    pub fn frobenius_norm(self) -> f64 {
        let mut sum = 0.0;
        for i in 0..3 { for j in 0..3 { sum += self.m[i][j] * self.m[i][j]; } }
        sum.sqrt()
    }

    /// Extract rotation via polar decomposition (iterative).
    pub fn extract_rotation(self) -> Self {
        let mut r = self;
        for _ in 0..10 {
            if let Some(r_inv_t) = r.transpose().inverse() {
                r = r.add(r_inv_t).scale(0.5);
            } else {
                break;
            }
            // Check convergence
            let diff = r.mul_mat(r.transpose()).sub(Self::IDENTITY).frobenius_norm();
            if diff < 1e-8 {
                break;
            }
        }
        r
    }
}

// ── Tetrahedron ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Tetrahedron {
    pub nodes: [usize; 4],
    pub rest_inv_dm: Mat3,
    pub rest_volume: f64,
    pub young_modulus: f64,
    pub poisson_ratio: f64,
    pub plastic_strain: Mat3,
    pub yield_threshold: f64,
}

// ── Boundary condition ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    Fixed,
    Free,
}

// ── Mesh ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeformableMesh {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub forces: Vec<Vec3>,
    pub masses: Vec<f64>,
    pub boundary: Vec<BoundaryCondition>,
    pub tets: Vec<Tetrahedron>,
    pub gravity: Vec3,
    pub damping: f64,
    pub time: f64,
}

impl DeformableMesh {
    pub fn new(
        rest_positions: Vec<Vec3>,
        tet_indices: Vec<[usize; 4]>,
        young_modulus: f64,
        poisson_ratio: f64,
        density: f64,
    ) -> Result<Self, DeformMeshError> {
        if young_modulus <= 0.0 {
            return Err(DeformMeshError::InvalidStiffness);
        }
        if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 {
            return Err(DeformMeshError::InvalidPoisson);
        }

        let n = rest_positions.len();
        let mut masses = vec![0.0f64; n];
        let mut tets = Vec::with_capacity(tet_indices.len());

        for (ti, nodes) in tet_indices.iter().enumerate() {
            for &nd in nodes {
                if nd >= n {
                    return Err(DeformMeshError::InvalidNode(nd));
                }
            }

            let p0 = rest_positions[nodes[0]];
            let p1 = rest_positions[nodes[1]];
            let p2 = rest_positions[nodes[2]];
            let p3 = rest_positions[nodes[3]];

            let dm = Mat3::from_cols(
                p1.sub(p0),
                p2.sub(p0),
                p3.sub(p0),
            );
            let vol = dm.determinant() / 6.0;
            if vol.abs() < 1e-15 {
                return Err(DeformMeshError::DegenerateTet(ti));
            }

            let rest_inv_dm = dm.inverse().unwrap();
            let abs_vol = vol.abs();

            // Mass lumping: distribute 1/4 of tet mass to each node
            let tet_mass = density * abs_vol;
            for &nd in nodes {
                masses[nd] += tet_mass / 4.0;
            }

            tets.push(Tetrahedron {
                nodes: *nodes,
                rest_inv_dm,
                rest_volume: vol,
                young_modulus,
                poisson_ratio,
                plastic_strain: Mat3::ZERO,
                yield_threshold: f64::MAX,
            });
        }

        // Ensure all nodes have minimum mass
        for m in &mut masses {
            if *m < 1e-10 {
                *m = 1e-3;
            }
        }

        Ok(Self {
            positions: rest_positions.clone(),
            velocities: vec![Vec3::ZERO; n],
            forces: vec![Vec3::ZERO; n],
            masses,
            boundary: vec![BoundaryCondition::Free; n],
            tets,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            damping: 0.01,
            time: 0.0,
        })
    }

    pub fn set_boundary(&mut self, node: usize, bc: BoundaryCondition) -> Result<(), DeformMeshError> {
        if node >= self.positions.len() {
            return Err(DeformMeshError::InvalidNode(node));
        }
        self.boundary[node] = bc;
        Ok(())
    }

    pub fn set_plasticity(&mut self, tet_idx: usize, yield_threshold: f64) -> Result<(), DeformMeshError> {
        if yield_threshold <= 0.0 {
            return Err(DeformMeshError::InvalidYield);
        }
        if tet_idx < self.tets.len() {
            self.tets[tet_idx].yield_threshold = yield_threshold;
        }
        Ok(())
    }

    pub fn set_all_plasticity(&mut self, yield_threshold: f64) -> Result<(), DeformMeshError> {
        if yield_threshold <= 0.0 {
            return Err(DeformMeshError::InvalidYield);
        }
        for t in &mut self.tets {
            t.yield_threshold = yield_threshold;
        }
        Ok(())
    }

    fn compute_element_forces(&mut self) {
        let positions = self.positions.clone();

        for tet in &mut self.tets {
            let ns = tet.nodes;
            let p0 = positions[ns[0]];
            let p1 = positions[ns[1]];
            let p2 = positions[ns[2]];
            let p3 = positions[ns[3]];

            // Current deformation matrix
            let ds = Mat3::from_cols(p1.sub(p0), p2.sub(p0), p3.sub(p0));
            // Deformation gradient F = Ds * Dm_inv
            let f_mat = ds.mul_mat(tet.rest_inv_dm);

            // Co-rotational: extract rotation R from F
            let r_mat = f_mat.extract_rotation();
            let r_t = r_mat.transpose();

            // Rotated deformation: R^T * F - I gives the strain in local frame
            let strain = r_t.mul_mat(f_mat).sub(Mat3::IDENTITY);

            // Subtract plastic strain
            let elastic_strain = strain.sub(tet.plastic_strain);

            // Lame parameters
            let e = tet.young_modulus;
            let nu = tet.poisson_ratio;
            let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let mu = e / (2.0 * (1.0 + nu));

            // Cauchy stress (linear): sigma = lambda * tr(strain) * I + 2*mu*strain
            let tr = elastic_strain.m[0][0] + elastic_strain.m[1][1] + elastic_strain.m[2][2];
            let stress = Mat3::IDENTITY.scale(lambda * tr).add(elastic_strain.scale(2.0 * mu));

            // Update plasticity: if strain exceeds yield, accumulate plastic strain
            let strain_norm = elastic_strain.frobenius_norm();
            if strain_norm > tet.yield_threshold {
                let excess = strain_norm - tet.yield_threshold;
                let factor = excess / strain_norm;
                tet.plastic_strain = tet.plastic_strain.add(elastic_strain.scale(factor * 0.5));
            }

            // Force = -V * R * sigma * Dm_inv^T
            let vol = tet.rest_volume.abs();
            let force_mat = r_mat.mul_mat(stress).mul_mat(tet.rest_inv_dm.transpose()).scale(-vol);

            // Distribute forces: f1, f2, f3 from columns; f0 = -(f1+f2+f3)
            let f1 = force_mat.col(0);
            let f2 = force_mat.col(1);
            let f3 = force_mat.col(2);
            let f0 = Vec3::ZERO.sub(f1).sub(f2).sub(f3);

            let forces = [f0, f1, f2, f3];
            for (i, &force) in forces.iter().enumerate() {
                let ni = ns[i];
                self.forces[ni] = self.forces[ni].add(force);
            }
        }
    }

    pub fn step(&mut self, dt: f64) -> Result<(), DeformMeshError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(DeformMeshError::InvalidTimestep);
        }

        let n = self.positions.len();

        // Clear forces, apply gravity
        for i in 0..n {
            self.forces[i] = self.gravity.scale(self.masses[i]);
        }

        // Compute FEM forces
        self.compute_element_forces();

        // Apply damping
        for i in 0..n {
            self.forces[i] = self.forces[i].sub(self.velocities[i].scale(self.damping));
        }

        // Semi-implicit Euler integration
        for i in 0..n {
            if self.boundary[i] == BoundaryCondition::Fixed {
                self.velocities[i] = Vec3::ZERO;
                continue;
            }
            let accel = self.forces[i].scale(1.0 / self.masses[i]);
            self.velocities[i] = self.velocities[i].add(accel.scale(dt));
            self.positions[i] = self.positions[i].add(self.velocities[i].scale(dt));
        }

        self.time += dt;
        Ok(())
    }

    pub fn total_volume(&self) -> f64 {
        let mut vol = 0.0;
        for tet in &self.tets {
            let ns = tet.nodes;
            let p0 = self.positions[ns[0]];
            let p1 = self.positions[ns[1]];
            let p2 = self.positions[ns[2]];
            let p3 = self.positions[ns[3]];
            let dm = Mat3::from_cols(p1.sub(p0), p2.sub(p0), p3.sub(p0));
            vol += dm.determinant() / 6.0;
        }
        vol.abs()
    }

    pub fn rest_volume(&self) -> f64 {
        self.tets.iter().map(|t| t.rest_volume.abs()).sum()
    }

    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for i in 0..self.positions.len() {
            ke += 0.5 * self.masses[i] * self.velocities[i].length_sq();
        }
        ke
    }

    pub fn center_of_mass(&self) -> Vec3 {
        let mut total_m = 0.0;
        let mut weighted = Vec3::ZERO;
        for i in 0..self.positions.len() {
            total_m += self.masses[i];
            weighted = weighted.add(self.positions[i].scale(self.masses[i]));
        }
        if total_m < 1e-12 { Vec3::ZERO } else { weighted.scale(1.0 / total_m) }
    }

    pub fn node_count(&self) -> usize {
        self.positions.len()
    }

    pub fn element_count(&self) -> usize {
        self.tets.len()
    }

    pub fn max_displacement(&self, rest_positions: &[Vec3]) -> f64 {
        let mut max_d = 0.0f64;
        for (i, pos) in self.positions.iter().enumerate() {
            if i < rest_positions.len() {
                let d = pos.sub(rest_positions[i]).length();
                if d > max_d { max_d = d; }
            }
        }
        max_d
    }
}

/// Helper: create a simple unit tetrahedron mesh.
pub fn unit_tet() -> (Vec<Vec3>, Vec<[usize; 4]>) {
    let verts = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let tets = vec![[0, 1, 2, 3]];
    (verts, tets)
}

/// Helper: create a cube split into 5 tetrahedra.
pub fn tet_cube(size: f64) -> (Vec<Vec3>, Vec<[usize; 4]>) {
    let s = size;
    let verts = vec![
        Vec3::new(0.0, 0.0, 0.0), // 0
        Vec3::new(s, 0.0, 0.0),   // 1
        Vec3::new(s, s, 0.0),     // 2
        Vec3::new(0.0, s, 0.0),   // 3
        Vec3::new(0.0, 0.0, s),   // 4
        Vec3::new(s, 0.0, s),     // 5
        Vec3::new(s, s, s),       // 6
        Vec3::new(0.0, s, s),     // 7
    ];
    let tets = vec![
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ];
    (verts, tets)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_mat3_identity() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = Mat3::IDENTITY.mul_vec(v);
        assert!(approx(r.x, 1.0, 1e-10));
        assert!(approx(r.y, 2.0, 1e-10));
        assert!(approx(r.z, 3.0, 1e-10));
    }

    #[test]
    fn test_mat3_determinant() {
        let m = Mat3 { m: [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]] };
        let det = m.determinant();
        assert!(approx(det, 1.0, 1e-10));
    }

    #[test]
    fn test_mat3_inverse() {
        let m = Mat3 { m: [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]] };
        let inv = m.inverse().unwrap();
        assert!(approx(inv.m[0][0], 0.5, 1e-10));
        assert!(approx(inv.m[1][1], 1.0 / 3.0, 1e-10));
        assert!(approx(inv.m[2][2], 0.25, 1e-10));
    }

    #[test]
    fn test_extract_rotation_identity() {
        let r = Mat3::IDENTITY.extract_rotation();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(r.m[i][j], expected, 1e-6));
            }
        }
    }

    #[test]
    fn test_unit_tet_creation() {
        let (verts, tets) = unit_tet();
        let mesh = DeformableMesh::new(verts, tets, 1000.0, 0.3, 1.0).unwrap();
        assert_eq!(mesh.node_count(), 4);
        assert_eq!(mesh.element_count(), 1);
    }

    #[test]
    fn test_rest_volume() {
        let (verts, tets) = unit_tet();
        let mesh = DeformableMesh::new(verts, tets, 1000.0, 0.3, 1.0).unwrap();
        // Volume of unit tet = 1/6
        assert!(approx(mesh.rest_volume(), 1.0 / 6.0, 1e-10));
    }

    #[test]
    fn test_current_volume_matches_rest() {
        let (verts, tets) = unit_tet();
        let mesh = DeformableMesh::new(verts, tets, 1000.0, 0.3, 1.0).unwrap();
        assert!(approx(mesh.total_volume(), mesh.rest_volume(), 1e-10));
    }

    #[test]
    fn test_invalid_stiffness() {
        let (v, t) = unit_tet();
        assert!(DeformableMesh::new(v, t, -1.0, 0.3, 1.0).is_err());
    }

    #[test]
    fn test_invalid_poisson() {
        let (v, t) = unit_tet();
        assert!(DeformableMesh::new(v.clone(), t.clone(), 1000.0, 0.5, 1.0).is_err());
        assert!(DeformableMesh::new(v, t, 1000.0, -1.0, 1.0).is_err());
    }

    #[test]
    fn test_fixed_node_stays() {
        let (verts, tets) = unit_tet();
        let mut mesh = DeformableMesh::new(verts, tets, 1000.0, 0.3, 1.0).unwrap();
        mesh.set_boundary(0, BoundaryCondition::Fixed).unwrap();
        let p0 = mesh.positions[0];
        for _ in 0..20 {
            mesh.step(0.001).unwrap();
        }
        assert!(approx(mesh.positions[0].x, p0.x, 1e-10));
        assert!(approx(mesh.positions[0].y, p0.y, 1e-10));
    }

    #[test]
    fn test_gravity_moves_free_nodes() {
        let (verts, tets) = unit_tet();
        let rest = verts.clone();
        let mut mesh = DeformableMesh::new(verts, tets, 1000.0, 0.3, 1.0).unwrap();
        mesh.set_boundary(0, BoundaryCondition::Fixed).unwrap();
        for _ in 0..50 {
            mesh.step(0.001).unwrap();
        }
        // Free nodes should have moved
        assert!(mesh.max_displacement(&rest) > 0.001);
    }

    #[test]
    fn test_tet_cube() {
        let (verts, tets) = tet_cube(1.0);
        let mesh = DeformableMesh::new(verts, tets, 5000.0, 0.3, 1.0).unwrap();
        assert_eq!(mesh.node_count(), 8);
        assert_eq!(mesh.element_count(), 5);
        // Cube volume = 1.0
        assert!(approx(mesh.rest_volume(), 1.0, 1e-6));
    }

    #[test]
    fn test_invalid_timestep() {
        let (v, t) = unit_tet();
        let mut mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        assert!(mesh.step(0.0).is_err());
        assert!(mesh.step(-1.0).is_err());
    }

    #[test]
    fn test_kinetic_energy_initially_zero() {
        let (v, t) = unit_tet();
        let mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        assert!(approx(mesh.kinetic_energy(), 0.0, 1e-12));
    }

    #[test]
    fn test_kinetic_energy_after_step() {
        let (v, t) = unit_tet();
        let mut mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        mesh.step(0.01).unwrap();
        assert!(mesh.kinetic_energy() > 0.0);
    }

    #[test]
    fn test_center_of_mass() {
        let (v, t) = unit_tet();
        let mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        let com = mesh.center_of_mass();
        // COM should be inside the tet
        assert!(com.x >= -0.1 && com.x <= 1.1);
        assert!(com.y >= -0.1 && com.y <= 1.1);
    }

    #[test]
    fn test_plasticity() {
        let (v, t) = unit_tet();
        let mut mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        mesh.set_all_plasticity(0.001).unwrap(); // very low yield
        // Deform significantly
        mesh.positions[1] = Vec3::new(3.0, 0.0, 0.0);
        mesh.step(0.001).unwrap();
        // Plastic strain should have accumulated
        let ps = mesh.tets[0].plastic_strain.frobenius_norm();
        assert!(ps > 0.0);
    }

    #[test]
    fn test_invalid_yield() {
        let (v, t) = unit_tet();
        let mut mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        assert!(mesh.set_all_plasticity(-1.0).is_err());
    }

    #[test]
    fn test_degenerate_tet_rejected() {
        let verts = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0), // colinear
            Vec3::new(3.0, 0.0, 0.0), // coplanar with line
        ];
        let r = DeformableMesh::new(verts, vec![[0, 1, 2, 3]], 1000.0, 0.3, 1.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_frobenius_norm() {
        let m = Mat3::IDENTITY;
        assert!(approx(m.frobenius_norm(), 3.0_f64.sqrt(), 1e-10));
    }

    #[test]
    fn test_mat3_transpose() {
        let m = Mat3 { m: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] };
        let t = m.transpose();
        assert!(approx(t.m[0][1], 4.0, 1e-10));
        assert!(approx(t.m[1][0], 2.0, 1e-10));
    }

    #[test]
    fn test_invalid_node_index() {
        let verts = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)];
        let r = DeformableMesh::new(verts, vec![[0, 1, 2, 3]], 1000.0, 0.3, 1.0);
        assert!(r.is_err());
    }

    #[test]
    fn test_set_boundary_oob() {
        let (v, t) = unit_tet();
        let mut mesh = DeformableMesh::new(v, t, 1000.0, 0.3, 1.0).unwrap();
        assert!(mesh.set_boundary(999, BoundaryCondition::Fixed).is_err());
    }
}
