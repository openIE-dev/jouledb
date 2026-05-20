//! Parametric Surface — generate triangle meshes from f(u,v) → (x,y,z) over [0,1]².
//!
//! Built-in surfaces: sphere, torus, Klein bottle, Möbius strip, superellipsoid,
//! Boy's surface. Normals via finite-difference cross products. UV coordinates.

use std::f64::consts::PI;

// ── Vector types (inline) ──────────────────────────────────────

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            Self::zero()
        } else {
            Self::new(self.x / len, self.y / len, self.z / len)
        }
    }
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

// ── Mesh output ────────────────────────────────────────────────

/// A generated triangle mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceMesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub indices: Vec<[u32; 3]>,
}

impl SurfaceMesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
}

// ── Parametric surface evaluator ───────────────────────────────

/// A parametric surface defined by a function f(u, v) → Vec3 over domain [0,1]².
pub struct ParametricSurface {
    /// Evaluation function: (u, v) → 3D point.
    eval: Box<dyn Fn(f64, f64) -> Vec3>,
    /// Subdivisions along U.
    pub u_res: usize,
    /// Subdivisions along V.
    pub v_res: usize,
}

impl ParametricSurface {
    /// Create a surface with custom evaluation function.
    pub fn new(eval: impl Fn(f64, f64) -> Vec3 + 'static, u_res: usize, v_res: usize) -> Self {
        Self {
            eval: Box::new(eval),
            u_res: u_res.max(2),
            v_res: v_res.max(2),
        }
    }

    /// Evaluate the surface at (u, v).
    pub fn evaluate(&self, u: f64, v: f64) -> Vec3 {
        (self.eval)(u, v)
    }

    /// Compute normal via finite-difference cross product.
    pub fn normal_at(&self, u: f64, v: f64) -> Vec3 {
        let eps = 1e-5;
        let u0 = (u - eps).max(0.0);
        let u1 = (u + eps).min(1.0);
        let v0 = (v - eps).max(0.0);
        let v1 = (v + eps).min(1.0);

        let du = (self.eval)(u1, v).sub(&(self.eval)(u0, v));
        let dv = (self.eval)(u, v1).sub(&(self.eval)(u, v0));

        du.cross(&dv).normalized()
    }

    /// Tessellate into a triangle mesh.
    pub fn tessellate(&self) -> SurfaceMesh {
        let u_steps = self.u_res;
        let v_steps = self.v_res;
        let vert_count = (u_steps + 1) * (v_steps + 1);
        let mut vertices = Vec::with_capacity(vert_count);
        let mut normals = Vec::with_capacity(vert_count);
        let mut uvs = Vec::with_capacity(vert_count);
        let mut indices = Vec::with_capacity(u_steps * v_steps * 2);

        for vi in 0..=v_steps {
            let v = vi as f64 / v_steps as f64;
            for ui in 0..=u_steps {
                let u = ui as f64 / u_steps as f64;
                vertices.push(self.evaluate(u, v));
                normals.push(self.normal_at(u, v));
                uvs.push(Vec2::new(u, v));
            }
        }

        let w = u_steps + 1;
        for vi in 0..v_steps {
            for ui in 0..u_steps {
                let a = (vi * w + ui) as u32;
                let b = (vi * w + ui + 1) as u32;
                let c = ((vi + 1) * w + ui + 1) as u32;
                let d = ((vi + 1) * w + ui) as u32;
                indices.push([a, b, c]);
                indices.push([a, c, d]);
            }
        }

        SurfaceMesh {
            vertices,
            normals,
            uvs,
            indices,
        }
    }
}

// ── Built-in surface constructors ──────────────────────────────

/// Unit sphere centered at origin.
pub fn sphere(u_res: usize, v_res: usize) -> ParametricSurface {
    ParametricSurface::new(
        move |u, v| {
            let theta = u * 2.0 * PI;
            let phi = v * PI;
            Vec3::new(
                phi.sin() * theta.cos(),
                phi.cos(),
                phi.sin() * theta.sin(),
            )
        },
        u_res,
        v_res,
    )
}

/// Torus with major radius R and minor radius r.
pub fn torus(major_r: f64, minor_r: f64, u_res: usize, v_res: usize) -> ParametricSurface {
    ParametricSurface::new(
        move |u, v| {
            let theta = u * 2.0 * PI;
            let phi = v * 2.0 * PI;
            Vec3::new(
                (major_r + minor_r * phi.cos()) * theta.cos(),
                minor_r * phi.sin(),
                (major_r + minor_r * phi.cos()) * theta.sin(),
            )
        },
        u_res,
        v_res,
    )
}

/// Klein bottle (figure-8 immersion).
pub fn klein_bottle(u_res: usize, v_res: usize) -> ParametricSurface {
    ParametricSurface::new(
        move |u, v| {
            let uu = u * 2.0 * PI;
            let vv = v * 2.0 * PI;
            let r = 4.0 * (1.0 - 0.5 * uu.cos());
            Vec3::new(
                if uu <= PI {
                    6.0 * uu.cos() * (1.0 + uu.sin())
                        + r * (uu.cos() * vv.cos())
                } else {
                    6.0 * uu.cos() * (1.0 + uu.sin())
                        + r * ((PI - uu).cos() * vv.cos())
                },
                if uu <= PI {
                    16.0 * uu.sin()
                } else {
                    16.0 * uu.sin()
                } + r * vv.sin(),
                r * vv.cos() * uu.sin(),
            )
        },
        u_res,
        v_res,
    )
}

/// Möbius strip with given half-width.
pub fn moebius_strip(half_width: f64, u_res: usize, v_res: usize) -> ParametricSurface {
    ParametricSurface::new(
        move |u, v| {
            let theta = u * 2.0 * PI;
            let s = v * 2.0 * half_width - half_width; // [-half_width, half_width]
            let half_theta = theta * 0.5;
            Vec3::new(
                (1.0 + s * half_theta.cos()) * theta.cos(),
                (1.0 + s * half_theta.cos()) * theta.sin(),
                s * half_theta.sin(),
            )
        },
        u_res,
        v_res,
    )
}

/// Superellipsoid with exponents e1 (latitude) and e2 (longitude).
pub fn superellipsoid(e1: f64, e2: f64, u_res: usize, v_res: usize) -> ParametricSurface {
    ParametricSurface::new(
        move |u, v| {
            let eta = v * PI - PI / 2.0; // [-π/2, π/2]
            let omega = u * 2.0 * PI - PI; // [-π, π]
            let sign_cos_eta = eta.cos().signum() * eta.cos().abs().powf(e1);
            let sign_sin_eta = eta.sin().signum() * eta.sin().abs().powf(e1);
            let sign_cos_omega = omega.cos().signum() * omega.cos().abs().powf(e2);
            let sign_sin_omega = omega.sin().signum() * omega.sin().abs().powf(e2);
            Vec3::new(
                sign_cos_eta * sign_cos_omega,
                sign_sin_eta,
                sign_cos_eta * sign_sin_omega,
            )
        },
        u_res,
        v_res,
    )
}

/// Boy's surface (a model of the real projective plane).
pub fn boys_surface(u_res: usize, v_res: usize) -> ParametricSurface {
    ParametricSurface::new(
        move |u, v| {
            let uu = u * PI;     // [0, π]
            let vv = v * 2.0 * PI; // [0, 2π]
            let su = uu.sin();
            let cu = uu.cos();
            let sv = vv.sin();
            let cv = vv.cos();
            let denom = 2.0 - (2.0 * vv).sin().sqrt() * (3.0 * uu).sin();
            let denom = if denom.abs() < 1e-10 { 1e-10 } else { denom };
            let x = (su * (2.0 * vv).cos() * cu * cu
                + (2.0 * vv).sin().sqrt() * cu)
                / denom;
            let y = (su * (2.0 * vv).cos() * su.powi(2) * sv * cv
                + (2.0 * vv).sin().sqrt() * sv)
                / denom;
            let z = 1.5 * cu * cu / denom;
            Vec3::new(x, y, z)
        },
        u_res,
        v_res,
    )
}

// ── Mesh utilities ─────────────────────────────────────────────

/// Recompute normals from triangle geometry (flat → smooth by averaging).
pub fn recompute_smooth_normals(mesh: &mut SurfaceMesh) {
    let n = mesh.vertices.len();
    let mut acc = vec![Vec3::zero(); n];

    for tri in &mesh.indices {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let edge1 = b.sub(&a);
        let edge2 = c.sub(&a);
        let face_normal = edge1.cross(&edge2);
        for &idx in tri {
            let i = idx as usize;
            acc[i] = Vec3::new(
                acc[i].x + face_normal.x,
                acc[i].y + face_normal.y,
                acc[i].z + face_normal.z,
            );
        }
    }

    mesh.normals = acc.into_iter().map(|n| n.normalized()).collect();
}

/// Compute the surface area of a mesh.
pub fn surface_area(mesh: &SurfaceMesh) -> f64 {
    let mut area = 0.0;
    for tri in &mesh.indices {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let edge1 = b.sub(&a);
        let edge2 = c.sub(&a);
        area += edge1.cross(&edge2).length() * 0.5;
    }
    area
}

/// Count unique edges in the mesh.
pub fn edge_count(mesh: &SurfaceMesh) -> usize {
    use std::collections::HashSet;
    let mut edges = HashSet::new();
    for tri in &mesh.indices {
        let mut add = |a: u32, b: u32| {
            let key = if a < b { (a, b) } else { (b, a) };
            edges.insert(key);
        };
        add(tri[0], tri[1]);
        add(tri[1], tri[2]);
        add(tri[2], tri[0]);
    }
    edges.len()
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-4;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_vec3_basic() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(v.length(), 5.0, EPS));
    }

    #[test]
    fn test_vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!(approx(z.x, 0.0, EPS));
        assert!(approx(z.y, 0.0, EPS));
        assert!(approx(z.z, 1.0, EPS));
    }

    #[test]
    fn test_vec3_normalized() {
        let v = Vec3::new(0.0, 0.0, 5.0);
        let n = v.normalized();
        assert!(approx(n.z, 1.0, EPS));
        assert!(approx(n.length(), 1.0, EPS));
    }

    #[test]
    fn test_vec3_zero_normalize() {
        let v = Vec3::zero();
        let n = v.normalized();
        assert!(approx(n.length(), 0.0, EPS));
    }

    #[test]
    fn test_sphere_tessellation_counts() {
        let s = sphere(8, 4);
        let mesh = s.tessellate();
        assert_eq!(mesh.vertex_count(), 9 * 5); // (8+1)*(4+1)
        assert_eq!(mesh.triangle_count(), 8 * 4 * 2);
    }

    #[test]
    fn test_sphere_pole_positions() {
        let s = sphere(16, 8);
        let mesh = s.tessellate();
        // v=0 → north pole (0, 1, 0)
        let north = mesh.vertices[0];
        assert!(approx(north.y, 1.0, EPS));
        // v=1 → south pole (0, -1, 0)
        let south_idx = 8 * (16 + 1);
        let south = mesh.vertices[south_idx];
        assert!(approx(south.y, -1.0, EPS));
    }

    #[test]
    fn test_sphere_radius() {
        let s = sphere(16, 8);
        let mesh = s.tessellate();
        for v in &mesh.vertices {
            assert!(approx(v.length(), 1.0, EPS));
        }
    }

    #[test]
    fn test_sphere_normals_point_outward() {
        let s = sphere(12, 6);
        let mesh = s.tessellate();
        for (v, n) in mesh.vertices.iter().zip(mesh.normals.iter()) {
            // Skip degenerate normals at poles (finite differences collapse)
            if n.length() < 0.5 {
                continue;
            }
            let dot = v.dot(n);
            // Normal should align with position on unit sphere
            assert!(dot > 0.3, "dot={dot}, v={v:?}, n={n:?}");
        }
    }

    #[test]
    fn test_sphere_surface_area() {
        let s = sphere(64, 32);
        let mesh = s.tessellate();
        let area = surface_area(&mesh);
        // Exact surface area of unit sphere = 4π ≈ 12.566
        assert!(approx(area, 4.0 * PI, 0.2));
    }

    #[test]
    fn test_torus_vertex_count() {
        let t = torus(2.0, 0.5, 16, 8);
        let mesh = t.tessellate();
        assert_eq!(mesh.vertex_count(), 17 * 9);
        assert_eq!(mesh.triangle_count(), 16 * 8 * 2);
    }

    #[test]
    fn test_torus_center_hole() {
        let t = torus(3.0, 1.0, 32, 16);
        let mesh = t.tessellate();
        // All vertices should be at distance [2, 4] from Y axis
        for v in &mesh.vertices {
            let dist = (v.x * v.x + v.z * v.z).sqrt();
            assert!(dist > 1.5, "dist too small: {dist}");
            assert!(dist < 4.5, "dist too large: {dist}");
        }
    }

    #[test]
    fn test_moebius_strip_counts() {
        let m = moebius_strip(0.4, 20, 4);
        let mesh = m.tessellate();
        assert_eq!(mesh.vertex_count(), 21 * 5);
        assert_eq!(mesh.triangle_count(), 20 * 4 * 2);
    }

    #[test]
    fn test_klein_bottle_tessellation() {
        let k = klein_bottle(12, 6);
        let mesh = k.tessellate();
        assert_eq!(mesh.vertex_count(), 13 * 7);
        assert_eq!(mesh.triangle_count(), 12 * 6 * 2);
    }

    #[test]
    fn test_superellipsoid_sphere_approx() {
        // e1=1, e2=1 → unit sphere
        let s = superellipsoid(1.0, 1.0, 32, 16);
        let mesh = s.tessellate();
        for v in &mesh.vertices {
            let r = v.length();
            assert!(approx(r, 1.0, 0.15));
        }
    }

    #[test]
    fn test_superellipsoid_cube_approx() {
        // e1→0, e2→0 → cube-like
        let s = superellipsoid(0.1, 0.1, 16, 8);
        let mesh = s.tessellate();
        // All vertices should be within bounding box [-1.1, 1.1]
        for v in &mesh.vertices {
            assert!(v.x.abs() < 1.1);
            assert!(v.y.abs() < 1.1);
            assert!(v.z.abs() < 1.1);
        }
    }

    #[test]
    fn test_boys_surface_tessellation() {
        let b = boys_surface(10, 10);
        let mesh = b.tessellate();
        assert_eq!(mesh.vertex_count(), 11 * 11);
        assert_eq!(mesh.triangle_count(), 10 * 10 * 2);
    }

    #[test]
    fn test_uv_coordinates_range() {
        let s = sphere(8, 4);
        let mesh = s.tessellate();
        for uv in &mesh.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0, "u out of range: {}", uv.x);
            assert!(uv.y >= 0.0 && uv.y <= 1.0, "v out of range: {}", uv.y);
        }
    }

    #[test]
    fn test_recompute_smooth_normals() {
        let s = sphere(8, 4);
        let mut mesh = s.tessellate();
        recompute_smooth_normals(&mut mesh);
        // Most normals should be unit length; degenerate poles may be zero
        let unit_count = mesh.normals.iter().filter(|n| approx(n.length(), 1.0, 0.15)).count();
        assert!(unit_count > mesh.normals.len() / 2);
    }

    #[test]
    fn test_edge_count_sphere() {
        let s = sphere(4, 2);
        let mesh = s.tessellate();
        let edges = edge_count(&mesh);
        let tris = mesh.triangle_count();
        let verts = mesh.vertex_count();
        // Euler: V - E + F = 2 for closed surfaces (approx for grid)
        // For a grid: E = 3*T/2 for manifold (but grid has boundary)
        assert!(edges > 0);
        assert!(edges <= tris * 3);
    }

    #[test]
    fn test_custom_surface() {
        let flat = ParametricSurface::new(
            |u, v| Vec3::new(u, 0.0, v),
            4,
            4,
        );
        let mesh = flat.tessellate();
        for v in &mesh.vertices {
            assert!(approx(v.y, 0.0, EPS));
        }
    }

    #[test]
    fn test_min_resolution_clamped() {
        let s = ParametricSurface::new(|u, v| Vec3::new(u, v, 0.0), 0, 1);
        assert!(s.u_res >= 2);
        assert!(s.v_res >= 2);
    }

    #[test]
    fn test_indices_in_bounds() {
        let s = sphere(10, 5);
        let mesh = s.tessellate();
        let max_idx = mesh.vertex_count() as u32;
        for tri in &mesh.indices {
            assert!(tri[0] < max_idx);
            assert!(tri[1] < max_idx);
            assert!(tri[2] < max_idx);
        }
    }

    #[test]
    fn test_normals_same_count_as_vertices() {
        let s = torus(2.0, 0.5, 12, 6);
        let mesh = s.tessellate();
        assert_eq!(mesh.normals.len(), mesh.vertices.len());
        assert_eq!(mesh.uvs.len(), mesh.vertices.len());
    }

    #[test]
    fn test_empty_mesh_area() {
        let mesh = SurfaceMesh {
            vertices: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
        };
        assert!(approx(surface_area(&mesh), 0.0, EPS));
    }
}
