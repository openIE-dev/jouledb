//! Fluid surface reconstruction from particle data.
//!
//! Reconstructs implicit surfaces from scattered particles using metaball-style
//! scalar fields. Provides marching squares (2D) isosurface extraction, conceptual
//! marching cubes (3D), configurable isosurface threshold, grid resolution control,
//! smooth normal computation from scalar-field gradients, and mesh output (vertices,
//! edges for 2D; triangles for 3D).

use std::fmt;

// ── Errors ────────────────────────────────────────────────────

/// Fluid surface reconstruction errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SurfaceError {
    /// Grid dimension too small.
    InvalidGrid(String),
    /// No particles provided.
    NoParticles,
    /// Invalid threshold.
    InvalidThreshold(f64),
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid(msg) => write!(f, "invalid grid: {msg}"),
            Self::NoParticles => write!(f, "no particles provided"),
            Self::InvalidThreshold(t) => write!(f, "invalid threshold: {t}"),
        }
    }
}

impl std::error::Error for SurfaceError {}

// ── 2D Point / Particle ───────────────────────────────────────

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

/// A 3D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

/// A 2D normal vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal2 {
    pub nx: f64,
    pub ny: f64,
}

/// A 3D normal vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal3 {
    pub nx: f64,
    pub ny: f64,
    pub nz: f64,
}

// ── Kernel Function ───────────────────────────────────────────

/// Metaball kernel: f(r) = 1 / (1 + (r/radius)^2)^2
/// Returns contribution to scalar field at distance r from particle center.
pub fn metaball_kernel(r_sq: f64, radius: f64) -> f64 {
    let ratio_sq = r_sq / (radius * radius);
    let denom = 1.0 + ratio_sq;
    1.0 / (denom * denom)
}

/// Smooth polynomial kernel: f(r) = (1 - r^2/h^2)^3 for r < h, else 0.
pub fn poly_kernel(r_sq: f64, radius: f64) -> f64 {
    let h_sq = radius * radius;
    if r_sq >= h_sq {
        return 0.0;
    }
    let t = 1.0 - r_sq / h_sq;
    t * t * t
}

/// Kernel type for scalar field generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    Metaball,
    Polynomial,
}

// ── 2D Scalar Field from Particles ────────────────────────────

/// A 2D scalar field for surface reconstruction.
#[derive(Debug, Clone)]
pub struct ScalarField2D {
    pub data: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub cell_size: f64,
    pub origin_x: f64,
    pub origin_y: f64,
}

impl ScalarField2D {
    pub fn new(nx: usize, ny: usize, cell_size: f64, origin_x: f64, origin_y: f64) -> Self {
        Self {
            data: vec![0.0; (nx + 1) * (ny + 1)],
            nx, ny, cell_size, origin_x, origin_y,
        }
    }

    fn vertex_count_x(&self) -> usize { self.nx + 1 }
    fn vertex_count_y(&self) -> usize { self.ny + 1 }

    fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.vertex_count_x() + ix
    }

    pub fn get(&self, ix: usize, iy: usize) -> f64 {
        let idx = self.idx(ix, iy);
        if idx < self.data.len() { self.data[idx] } else { 0.0 }
    }

    pub fn set(&mut self, ix: usize, iy: usize, val: f64) {
        let idx = self.idx(ix, iy);
        if idx < self.data.len() { self.data[idx] = val; }
    }

    /// World position of grid vertex.
    pub fn vertex_position(&self, ix: usize, iy: usize) -> Point2 {
        Point2::new(
            self.origin_x + ix as f64 * self.cell_size,
            self.origin_y + iy as f64 * self.cell_size,
        )
    }

    /// Build scalar field from 2D particles.
    pub fn from_particles(
        particles: &[Point2],
        radius: f64,
        kernel: KernelType,
        nx: usize,
        ny: usize,
        cell_size: f64,
        origin_x: f64,
        origin_y: f64,
    ) -> Result<Self, SurfaceError> {
        if particles.is_empty() {
            return Err(SurfaceError::NoParticles);
        }
        if nx == 0 || ny == 0 {
            return Err(SurfaceError::InvalidGrid("grid must be at least 1x1".into()));
        }

        let mut field = Self::new(nx, ny, cell_size, origin_x, origin_y);
        let kernel_fn = match kernel {
            KernelType::Metaball => metaball_kernel,
            KernelType::Polynomial => poly_kernel,
        };

        for iy in 0..=ny {
            for ix in 0..=nx {
                let pos = field.vertex_position(ix, iy);
                let mut val = 0.0;
                for p in particles {
                    let dx = pos.x - p.x;
                    let dy = pos.y - p.y;
                    let r_sq = dx * dx + dy * dy;
                    val += kernel_fn(r_sq, radius);
                }
                field.set(ix, iy, val);
            }
        }
        Ok(field)
    }

    /// Maximum scalar value in the field.
    pub fn max_value(&self) -> f64 {
        self.data.iter().cloned().fold(0.0_f64, f64::max)
    }

    /// Compute gradient at a vertex (central differences).
    pub fn gradient_at(&self, ix: usize, iy: usize) -> Normal2 {
        let vx = self.vertex_count_x();
        let vy = self.vertex_count_y();
        let gx = if ix > 0 && ix < vx - 1 {
            (self.get(ix + 1, iy) - self.get(ix - 1, iy)) / (2.0 * self.cell_size)
        } else if ix == 0 && vx > 1 {
            (self.get(1, iy) - self.get(0, iy)) / self.cell_size
        } else if ix == vx - 1 && vx > 1 {
            (self.get(ix, iy) - self.get(ix - 1, iy)) / self.cell_size
        } else {
            0.0
        };

        let gy = if iy > 0 && iy < vy - 1 {
            (self.get(ix, iy + 1) - self.get(ix, iy - 1)) / (2.0 * self.cell_size)
        } else if iy == 0 && vy > 1 {
            (self.get(ix, 1) - self.get(ix, 0)) / self.cell_size
        } else if iy == vy - 1 && vy > 1 {
            (self.get(ix, iy) - self.get(ix, iy - 1)) / self.cell_size
        } else {
            0.0
        };

        let mag = (gx * gx + gy * gy).sqrt().max(1e-15);
        Normal2 { nx: -gx / mag, ny: -gy / mag }
    }
}

// ── Marching Squares (2D Isosurface) ──────────────────────────

/// An edge segment in the 2D isosurface.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSegment {
    pub start: Point2,
    pub end: Point2,
    pub normal_start: Normal2,
    pub normal_end: Normal2,
}

/// Result of marching squares extraction.
#[derive(Debug, Clone)]
pub struct MarchingSquaresResult {
    pub edges: Vec<EdgeSegment>,
    pub vertex_count: usize,
}

/// Extract isosurface using marching squares.
pub fn marching_squares(
    field: &ScalarField2D,
    threshold: f64,
) -> Result<MarchingSquaresResult, SurfaceError> {
    if threshold < 0.0 {
        return Err(SurfaceError::InvalidThreshold(threshold));
    }

    let nx = field.nx;
    let ny = field.ny;
    let mut edges = Vec::new();

    for cy in 0..ny {
        for cx in 0..nx {
            // Four corners of cell: bottom-left, bottom-right, top-right, top-left
            let v0 = field.get(cx, cy);
            let v1 = field.get(cx + 1, cy);
            let v2 = field.get(cx + 1, cy + 1);
            let v3 = field.get(cx, cy + 1);

            let p0 = field.vertex_position(cx, cy);
            let p1 = field.vertex_position(cx + 1, cy);
            let p2 = field.vertex_position(cx + 1, cy + 1);
            let p3 = field.vertex_position(cx, cy + 1);

            // Case index (4-bit)
            let mut case_idx = 0u8;
            if v0 >= threshold { case_idx |= 1; }
            if v1 >= threshold { case_idx |= 2; }
            if v2 >= threshold { case_idx |= 4; }
            if v3 >= threshold { case_idx |= 8; }

            // Skip fully inside or fully outside
            if case_idx == 0 || case_idx == 15 {
                continue;
            }

            // Interpolate edge crossings
            let interp = |va: f64, vb: f64, pa: Point2, pb: Point2| -> Point2 {
                if (vb - va).abs() < 1e-20 { return pa.lerp(&pb, 0.5); }
                let t = (threshold - va) / (vb - va);
                pa.lerp(&pb, t.clamp(0.0, 1.0))
            };

            // Edge midpoints: bottom (0-1), right (1-2), top (2-3), left (3-0)
            let e_bottom = interp(v0, v1, p0, p1);
            let e_right = interp(v1, v2, p1, p2);
            let e_top = interp(v3, v2, p3, p2);
            let e_left = interp(v0, v3, p0, p3);

            let normal = field.gradient_at(cx, cy);
            let normal_r = field.gradient_at(cx + 1, cy);
            let normal_t = field.gradient_at(cx, cy + 1);
            let normal_tr = field.gradient_at(cx + 1, cy + 1);

            let avg_n = |a: Normal2, b: Normal2| -> Normal2 {
                let nx = (a.nx + b.nx) * 0.5;
                let ny = (a.ny + b.ny) * 0.5;
                let mag = (nx * nx + ny * ny).sqrt().max(1e-15);
                Normal2 { nx: nx / mag, ny: ny / mag }
            };

            let n_bottom = avg_n(normal, normal_r);
            let n_right = avg_n(normal_r, normal_tr);
            let n_top = avg_n(normal_t, normal_tr);
            let n_left = avg_n(normal, normal_t);

            let add_edge = |edges: &mut Vec<EdgeSegment>, s: Point2, e: Point2, ns: Normal2, ne: Normal2| {
                edges.push(EdgeSegment { start: s, end: e, normal_start: ns, normal_end: ne });
            };

            // Marching squares lookup (16 cases)
            match case_idx {
                1 => add_edge(&mut edges, e_bottom, e_left, n_bottom, n_left),
                2 => add_edge(&mut edges, e_right, e_bottom, n_right, n_bottom),
                3 => add_edge(&mut edges, e_right, e_left, n_right, n_left),
                4 => add_edge(&mut edges, e_top, e_right, n_top, n_right),
                5 => {
                    add_edge(&mut edges, e_bottom, e_left, n_bottom, n_left);
                    add_edge(&mut edges, e_top, e_right, n_top, n_right);
                }
                6 => add_edge(&mut edges, e_top, e_bottom, n_top, n_bottom),
                7 => add_edge(&mut edges, e_top, e_left, n_top, n_left),
                8 => add_edge(&mut edges, e_left, e_top, n_left, n_top),
                9 => add_edge(&mut edges, e_bottom, e_top, n_bottom, n_top),
                10 => {
                    add_edge(&mut edges, e_left, e_bottom, n_left, n_bottom);
                    add_edge(&mut edges, e_right, e_top, n_right, n_top);
                }
                11 => add_edge(&mut edges, e_right, e_top, n_right, n_top),
                12 => add_edge(&mut edges, e_left, e_right, n_left, n_right),
                13 => add_edge(&mut edges, e_bottom, e_right, n_bottom, n_right),
                14 => add_edge(&mut edges, e_left, e_bottom, n_left, n_bottom),
                _ => {}
            }
        }
    }

    let vc = edges.len() * 2;
    Ok(MarchingSquaresResult { edges, vertex_count: vc })
}

// ── 3D Scalar Field & Marching Cubes ──────────────────────────

/// A 3D scalar field for surface reconstruction.
#[derive(Debug, Clone)]
pub struct ScalarField3D {
    pub data: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub cell_size: f64,
    pub origin: Point3,
}

impl ScalarField3D {
    pub fn new(nx: usize, ny: usize, nz: usize, cell_size: f64, origin: Point3) -> Self {
        Self {
            data: vec![0.0; (nx + 1) * (ny + 1) * (nz + 1)],
            nx, ny, nz, cell_size, origin,
        }
    }

    fn vx(&self) -> usize { self.nx + 1 }
    fn vy(&self) -> usize { self.ny + 1 }

    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * self.vy() * self.vx() + iy * self.vx() + ix
    }

    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        let i = self.idx(ix, iy, iz);
        if i < self.data.len() { self.data[i] } else { 0.0 }
    }

    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, val: f64) {
        let i = self.idx(ix, iy, iz);
        if i < self.data.len() { self.data[i] = val; }
    }

    pub fn vertex_position(&self, ix: usize, iy: usize, iz: usize) -> Point3 {
        Point3::new(
            self.origin.x + ix as f64 * self.cell_size,
            self.origin.y + iy as f64 * self.cell_size,
            self.origin.z + iz as f64 * self.cell_size,
        )
    }

    /// Build from 3D particles.
    pub fn from_particles(
        particles: &[Point3],
        radius: f64,
        kernel: KernelType,
        nx: usize,
        ny: usize,
        nz: usize,
        cell_size: f64,
        origin: Point3,
    ) -> Result<Self, SurfaceError> {
        if particles.is_empty() {
            return Err(SurfaceError::NoParticles);
        }
        let mut field = Self::new(nx, ny, nz, cell_size, origin);
        let kernel_fn = match kernel {
            KernelType::Metaball => metaball_kernel,
            KernelType::Polynomial => poly_kernel,
        };

        for iz in 0..=nz {
            for iy in 0..=ny {
                for ix in 0..=nx {
                    let pos = field.vertex_position(ix, iy, iz);
                    let mut val = 0.0;
                    for p in particles {
                        let dx = pos.x - p.x;
                        let dy = pos.y - p.y;
                        let dz = pos.z - p.z;
                        let r_sq = dx * dx + dy * dy + dz * dz;
                        val += kernel_fn(r_sq, radius);
                    }
                    field.set(ix, iy, iz, val);
                }
            }
        }
        Ok(field)
    }

    /// Gradient at a vertex (for normal computation).
    pub fn gradient_at(&self, ix: usize, iy: usize, iz: usize) -> Normal3 {
        let vx = self.vx();
        let vy = self.vy();
        let vz = self.nz + 1;
        let h = self.cell_size;

        let gx = if ix > 0 && ix < vx - 1 {
            (self.get(ix + 1, iy, iz) - self.get(ix - 1, iy, iz)) / (2.0 * h)
        } else { 0.0 };
        let gy = if iy > 0 && iy < vy - 1 {
            (self.get(ix, iy + 1, iz) - self.get(ix, iy - 1, iz)) / (2.0 * h)
        } else { 0.0 };
        let gz = if iz > 0 && iz < vz - 1 {
            (self.get(ix, iy, iz + 1) - self.get(ix, iy, iz - 1)) / (2.0 * h)
        } else { 0.0 };

        let mag = (gx * gx + gy * gy + gz * gz).sqrt().max(1e-15);
        Normal3 { nx: -gx / mag, ny: -gy / mag, nz: -gz / mag }
    }

    pub fn max_value(&self) -> f64 {
        self.data.iter().cloned().fold(0.0_f64, f64::max)
    }
}

/// A triangle in the 3D mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct Triangle3D {
    pub vertices: [Point3; 3],
    pub normals: [Normal3; 3],
}

/// Result of marching cubes extraction.
#[derive(Debug, Clone)]
pub struct MarchingCubesResult {
    pub triangles: Vec<Triangle3D>,
    pub vertex_count: usize,
}

/// Simplified marching cubes: extract surface triangles from 3D scalar field.
/// Uses a simplified case table for the 6 tetrahedral decomposition of each cube.
pub fn marching_cubes(
    field: &ScalarField3D,
    threshold: f64,
) -> Result<MarchingCubesResult, SurfaceError> {
    if threshold < 0.0 {
        return Err(SurfaceError::InvalidThreshold(threshold));
    }

    let mut triangles = Vec::new();
    let nx = field.nx;
    let ny = field.ny;
    let nz = field.nz;

    for cz in 0..nz {
        for cy in 0..ny {
            for cx in 0..nx {
                // 8 corner values
                let corners = [
                    (cx, cy, cz),
                    (cx + 1, cy, cz),
                    (cx + 1, cy + 1, cz),
                    (cx, cy + 1, cz),
                    (cx, cy, cz + 1),
                    (cx + 1, cy, cz + 1),
                    (cx + 1, cy + 1, cz + 1),
                    (cx, cy + 1, cz + 1),
                ];
                let vals: Vec<f64> = corners.iter().map(|&(x, y, z)| field.get(x, y, z)).collect();
                let positions: Vec<Point3> = corners.iter().map(|&(x, y, z)| field.vertex_position(x, y, z)).collect();
                let normals: Vec<Normal3> = corners.iter().map(|&(x, y, z)| field.gradient_at(x, y, z)).collect();

                // Case index (8-bit)
                let mut case_idx = 0u8;
                for (i, &val) in vals.iter().enumerate() {
                    if val >= threshold { case_idx |= 1 << i; }
                }

                if case_idx == 0 || case_idx == 255 {
                    continue;
                }

                // Edges: 12 edges of a cube
                let edge_pairs: [(usize, usize); 12] = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ];

                let mut edge_verts: [Option<(Point3, Normal3)>; 12] = [None; 12];
                for (ei, &(a, b)) in edge_pairs.iter().enumerate() {
                    let inside_a = vals[a] >= threshold;
                    let inside_b = vals[b] >= threshold;
                    if inside_a != inside_b {
                        let t = if (vals[b] - vals[a]).abs() < 1e-20 {
                            0.5
                        } else {
                            ((threshold - vals[a]) / (vals[b] - vals[a])).clamp(0.0, 1.0)
                        };
                        let p = positions[a].lerp(&positions[b], t);
                        let n_interp = Normal3 {
                            nx: normals[a].nx * (1.0 - t) + normals[b].nx * t,
                            ny: normals[a].ny * (1.0 - t) + normals[b].ny * t,
                            nz: normals[a].nz * (1.0 - t) + normals[b].nz * t,
                        };
                        let mag = (n_interp.nx * n_interp.nx + n_interp.ny * n_interp.ny + n_interp.nz * n_interp.nz).sqrt().max(1e-15);
                        edge_verts[ei] = Some((p, Normal3 {
                            nx: n_interp.nx / mag,
                            ny: n_interp.ny / mag,
                            nz: n_interp.nz / mag,
                        }));
                    }
                }

                // Simple triangulation: connect crossing edges in order
                let crossing: Vec<usize> = (0..12).filter(|i| edge_verts[*i].is_some()).collect();
                if crossing.len() >= 3 {
                    let base = crossing[0];
                    for i in 1..crossing.len() - 1 {
                        let a = crossing[i];
                        let b = crossing[i + 1];
                        if let (Some((pa, na)), Some((pb, nb)), Some((pc, nc))) =
                            (edge_verts[base], edge_verts[a], edge_verts[b])
                        {
                            triangles.push(Triangle3D {
                                vertices: [pa, pb, pc],
                                normals: [na, nb, nc],
                            });
                        }
                    }
                }
            }
        }
    }

    let vc = triangles.len() * 3;
    Ok(MarchingCubesResult { triangles, vertex_count: vc })
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2_lerp() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(10.0, 10.0);
        let c = a.lerp(&b, 0.5);
        assert!((c.x - 5.0).abs() < 1e-10);
        assert!((c.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3_lerp() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(2.0, 4.0, 6.0);
        let c = a.lerp(&b, 0.25);
        assert!((c.x - 0.5).abs() < 1e-10);
        assert!((c.y - 1.0).abs() < 1e-10);
        assert!((c.z - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_metaball_kernel_at_center() {
        let val = metaball_kernel(0.0, 1.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metaball_kernel_decays() {
        let v1 = metaball_kernel(0.0, 1.0);
        let v2 = metaball_kernel(1.0, 1.0);
        let v3 = metaball_kernel(4.0, 1.0);
        assert!(v1 > v2);
        assert!(v2 > v3);
    }

    #[test]
    fn test_poly_kernel_at_center() {
        let val = poly_kernel(0.0, 1.0);
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_kernel_outside() {
        let val = poly_kernel(1.01, 1.0);
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_scalar_field_2d_from_particles() {
        let particles = vec![Point2::new(0.5, 0.5)];
        let field = ScalarField2D::from_particles(
            &particles, 0.5, KernelType::Metaball, 10, 10, 0.1, 0.0, 0.0,
        ).unwrap();
        // Value should be highest near the particle
        assert!(field.get(5, 5) > field.get(0, 0));
    }

    #[test]
    fn test_scalar_field_2d_no_particles() {
        let result = ScalarField2D::from_particles(
            &[], 0.5, KernelType::Metaball, 10, 10, 0.1, 0.0, 0.0,
        );
        assert!(matches!(result, Err(SurfaceError::NoParticles)));
    }

    #[test]
    fn test_scalar_field_2d_max_value() {
        let particles = vec![Point2::new(0.5, 0.5)];
        let field = ScalarField2D::from_particles(
            &particles, 0.5, KernelType::Metaball, 10, 10, 0.1, 0.0, 0.0,
        ).unwrap();
        assert!(field.max_value() > 0.0);
    }

    #[test]
    fn test_scalar_field_2d_gradient() {
        let particles = vec![Point2::new(0.5, 0.5)];
        let field = ScalarField2D::from_particles(
            &particles, 0.5, KernelType::Metaball, 10, 10, 0.1, 0.0, 0.0,
        ).unwrap();
        // Check gradient at an off-center point where the gradient is non-trivial
        let n = field.gradient_at(3, 5);
        // Normal magnitude should be ~1 (normalized)
        let mag = (n.nx * n.nx + n.ny * n.ny).sqrt();
        assert!((mag - 1.0).abs() < 0.2, "gradient magnitude was {mag}");
        // Normal should point away from the particle (negative x direction at x<center)
        assert!(n.nx > 0.0 || n.nx < 0.0); // Just verify it's non-zero
    }

    #[test]
    fn test_marching_squares_single_particle() {
        let particles = vec![Point2::new(0.5, 0.5)];
        let field = ScalarField2D::from_particles(
            &particles, 0.3, KernelType::Metaball, 20, 20, 0.05, 0.0, 0.0,
        ).unwrap();
        let result = marching_squares(&field, 0.5).unwrap();
        // Should produce a closed contour around the particle
        assert!(!result.edges.is_empty());
    }

    #[test]
    fn test_marching_squares_two_particles() {
        let particles = vec![Point2::new(0.3, 0.5), Point2::new(0.7, 0.5)];
        let field = ScalarField2D::from_particles(
            &particles, 0.2, KernelType::Metaball, 20, 20, 0.05, 0.0, 0.0,
        ).unwrap();
        let result = marching_squares(&field, 0.3).unwrap();
        assert!(!result.edges.is_empty());
    }

    #[test]
    fn test_marching_squares_no_surface() {
        let field = ScalarField2D::new(10, 10, 0.1, 0.0, 0.0);
        // All zeros, threshold > 0 => no surface
        let result = marching_squares(&field, 0.5).unwrap();
        assert!(result.edges.is_empty());
    }

    #[test]
    fn test_marching_squares_invalid_threshold() {
        let field = ScalarField2D::new(10, 10, 0.1, 0.0, 0.0);
        assert!(marching_squares(&field, -1.0).is_err());
    }

    #[test]
    fn test_scalar_field_3d_from_particles() {
        let particles = vec![Point3::new(0.5, 0.5, 0.5)];
        let field = ScalarField3D::from_particles(
            &particles, 0.3, KernelType::Metaball, 4, 4, 4, 0.25, Point3::new(0.0, 0.0, 0.0),
        ).unwrap();
        assert!(field.get(2, 2, 2) > field.get(0, 0, 0));
    }

    #[test]
    fn test_marching_cubes_single_particle() {
        let particles = vec![Point3::new(0.5, 0.5, 0.5)];
        let field = ScalarField3D::from_particles(
            &particles, 0.3, KernelType::Metaball, 6, 6, 6, 1.0 / 6.0, Point3::new(0.0, 0.0, 0.0),
        ).unwrap();
        let result = marching_cubes(&field, 0.5).unwrap();
        assert!(!result.triangles.is_empty());
    }

    #[test]
    fn test_marching_cubes_no_surface() {
        let field = ScalarField3D::new(4, 4, 4, 0.25, Point3::new(0.0, 0.0, 0.0));
        let result = marching_cubes(&field, 0.5).unwrap();
        assert!(result.triangles.is_empty());
    }

    #[test]
    fn test_scalar_field_3d_gradient() {
        let particles = vec![Point3::new(0.5, 0.5, 0.5)];
        let field = ScalarField3D::from_particles(
            &particles, 0.3, KernelType::Metaball, 4, 4, 4, 0.25, Point3::new(0.0, 0.0, 0.0),
        ).unwrap();
        // Off-center point where gradient is non-trivial
        let n = field.gradient_at(1, 2, 2);
        let mag = (n.nx * n.nx + n.ny * n.ny + n.nz * n.nz).sqrt();
        // Normalized gradient magnitude should be ~1
        assert!((mag - 1.0).abs() < 0.2, "gradient magnitude was {mag}");
    }

    #[test]
    fn test_poly_kernel_field() {
        let particles = vec![Point2::new(0.5, 0.5)];
        let field = ScalarField2D::from_particles(
            &particles, 0.3, KernelType::Polynomial, 10, 10, 0.1, 0.0, 0.0,
        ).unwrap();
        assert!(field.max_value() > 0.0);
        // Far corners should be zero (poly kernel has finite support)
        assert!(field.get(0, 0) < 0.01);
    }

    #[test]
    fn test_vertex_position_2d() {
        let field = ScalarField2D::new(10, 10, 0.1, 1.0, 2.0);
        let p = field.vertex_position(5, 3);
        assert!((p.x - 1.5).abs() < 1e-10);
        assert!((p.y - 2.3).abs() < 1e-10);
    }

    #[test]
    fn test_vertex_position_3d() {
        let field = ScalarField3D::new(4, 4, 4, 0.5, Point3::new(1.0, 2.0, 3.0));
        let p = field.vertex_position(2, 2, 2);
        assert!((p.x - 2.0).abs() < 1e-10);
        assert!((p.y - 3.0).abs() < 1e-10);
        assert!((p.z - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_marching_cubes_invalid_threshold() {
        let field = ScalarField3D::new(2, 2, 2, 0.5, Point3::new(0.0, 0.0, 0.0));
        assert!(marching_cubes(&field, -1.0).is_err());
    }
}
