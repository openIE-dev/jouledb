//! Mesh — vertex/index buffers, geometry generators, stats, and normal recalculation.

use crate::webgl::Vec3;

// ── Vertex ────────────────────────────────────────────────────

/// A vertex with position, normal, UV coordinates, and RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f64; 3],
    pub normal: [f64; 3],
    pub uv: [f64; 2],
    pub color: [f64; 4],
}

impl Vertex {
    pub fn new(position: [f64; 3], normal: [f64; 3], uv: [f64; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// A triangle defined by three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle(pub u32, pub u32, pub u32);

// ── MeshStats ─────────────────────────────────────────────────

/// Computed statistics for a mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounding_box_min: Vec3,
    pub bounding_box_max: Vec3,
    pub bounding_sphere_center: Vec3,
    pub bounding_sphere_radius: f64,
}

// ── Mesh ──────────────────────────────────────────────────────

/// A 3D mesh with vertex and index buffers.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Triangle>,
}

impl Mesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Compute bounding box, bounding sphere, and counts.
    pub fn stats(&self) -> MeshStats {
        if self.vertices.is_empty() {
            return MeshStats {
                vertex_count: 0,
                triangle_count: 0,
                bounding_box_min: Vec3::zero(),
                bounding_box_max: Vec3::zero(),
                bounding_sphere_center: Vec3::zero(),
                bounding_sphere_radius: 0.0,
            };
        }

        let mut min = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max = Vec3::new(f64::MIN, f64::MIN, f64::MIN);
        for v in &self.vertices {
            min.x = min.x.min(v.position[0]);
            min.y = min.y.min(v.position[1]);
            min.z = min.z.min(v.position[2]);
            max.x = max.x.max(v.position[0]);
            max.y = max.y.max(v.position[1]);
            max.z = max.z.max(v.position[2]);
        }

        let center = Vec3::new(
            (min.x + max.x) * 0.5,
            (min.y + max.y) * 0.5,
            (min.z + max.z) * 0.5,
        );

        let mut max_dist_sq = 0.0f64;
        for v in &self.vertices {
            let dx = v.position[0] - center.x;
            let dy = v.position[1] - center.y;
            let dz = v.position[2] - center.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > max_dist_sq {
                max_dist_sq = d2;
            }
        }

        MeshStats {
            vertex_count: self.vertices.len(),
            triangle_count: self.indices.len(),
            bounding_box_min: min,
            bounding_box_max: max,
            bounding_sphere_center: center,
            bounding_sphere_radius: max_dist_sq.sqrt(),
        }
    }

    /// Recalculate normals from triangle faces (area-weighted, averaged per vertex).
    pub fn recalculate_normals(&mut self) {
        // Zero out normals.
        for v in &mut self.vertices {
            v.normal = [0.0, 0.0, 0.0];
        }
        // Accumulate face normals.
        for tri in &self.indices {
            let a = self.vertices[tri.0 as usize].position;
            let b = self.vertices[tri.1 as usize].position;
            let c = self.vertices[tri.2 as usize].position;

            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let nx = ab[1] * ac[2] - ab[2] * ac[1];
            let ny = ab[2] * ac[0] - ab[0] * ac[2];
            let nz = ab[0] * ac[1] - ab[1] * ac[0];

            for &idx in &[tri.0, tri.1, tri.2] {
                let n = &mut self.vertices[idx as usize].normal;
                n[0] += nx;
                n[1] += ny;
                n[2] += nz;
            }
        }
        // Normalize.
        for v in &mut self.vertices {
            let n = &mut v.normal;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-12 {
                n[0] /= len;
                n[1] /= len;
                n[2] /= len;
            }
        }
    }

    // ── Geometry Generators ───────────────────────────────────

    /// Generate a unit cube centered at origin (side length 2).
    pub fn cube() -> Self {
        let positions: [[f64; 3]; 8] = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];

        // 6 faces, 2 triangles each, 4 unique vertices per face (for correct normals).
        let face_indices: [[usize; 4]; 6] = [
            [0, 1, 2, 3], // front (-Z)
            [5, 4, 7, 6], // back (+Z)
            [4, 0, 3, 7], // left (-X)
            [1, 5, 6, 2], // right (+X)
            [3, 2, 6, 7], // top (+Y)
            [4, 5, 1, 0], // bottom (-Y)
        ];

        let normals: [[f64; 3]; 6] = [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];

        let face_uvs: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(12);

        for (fi, face) in face_indices.iter().enumerate() {
            let base = vertices.len() as u32;
            for (vi, &pi) in face.iter().enumerate() {
                vertices.push(Vertex::new(positions[pi], normals[fi], face_uvs[vi]));
            }
            indices.push(Triangle(base, base + 1, base + 2));
            indices.push(Triangle(base, base + 2, base + 3));
        }

        Self { vertices, indices }
    }

    /// Generate an icosphere by subdividing an icosahedron.
    pub fn sphere(subdivisions: u32) -> Self {
        let t = (1.0 + 5.0_f64.sqrt()) / 2.0;

        let mut positions: Vec<[f64; 3]> = vec![
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ];
        // Normalize to unit sphere.
        for p in &mut positions {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            p[0] /= len;
            p[1] /= len;
            p[2] /= len;
        }

        let mut tris: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];

        use std::collections::HashMap;
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();

        for _ in 0..subdivisions {
            let mut new_tris = Vec::with_capacity(tris.len() * 4);
            for tri in &tris {
                let a = tri[0];
                let b = tri[1];
                let c = tri[2];
                let ab = get_midpoint(a, b, &mut positions, &mut midpoint_cache);
                let bc = get_midpoint(b, c, &mut positions, &mut midpoint_cache);
                let ca = get_midpoint(c, a, &mut positions, &mut midpoint_cache);
                new_tris.push([a, ab, ca]);
                new_tris.push([b, bc, ab]);
                new_tris.push([c, ca, bc]);
                new_tris.push([ab, bc, ca]);
            }
            tris = new_tris;
        }

        let vertices: Vec<Vertex> = positions
            .iter()
            .map(|p| {
                let u = 0.5 + p[2].atan2(p[0]) / (2.0 * std::f64::consts::PI);
                let v = 0.5 - p[1].asin() / std::f64::consts::PI;
                Vertex::new(*p, *p, [u, v])
            })
            .collect();

        let indices: Vec<Triangle> = tris.iter().map(|t| Triangle(t[0], t[1], t[2])).collect();

        Self { vertices, indices }
    }

    /// Generate a plane on the XZ plane centered at origin.
    pub fn plane(width: f64, depth: f64, segments_x: u32, segments_z: u32) -> Self {
        let sx = segments_x.max(1);
        let sz = segments_z.max(1);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for iz in 0..=sz {
            for ix in 0..=sx {
                let fx = ix as f64 / sx as f64;
                let fz = iz as f64 / sz as f64;
                let x = (fx - 0.5) * width;
                let z = (fz - 0.5) * depth;
                vertices.push(Vertex::new([x, 0.0, z], [0.0, 1.0, 0.0], [fx, fz]));
            }
        }

        let cols = sx + 1;
        for iz in 0..sz {
            for ix in 0..sx {
                let a = iz * cols + ix;
                let b = a + 1;
                let c = a + cols;
                let d = c + 1;
                indices.push(Triangle(a, c, b));
                indices.push(Triangle(b, c, d));
            }
        }

        Self { vertices, indices }
    }

    /// Generate a cylinder along the Y axis.
    pub fn cylinder(radius: f64, height: f64, segments: u32) -> Self {
        let seg = segments.max(3);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let half_h = height / 2.0;

        // Side vertices: two rings.
        for i in 0..=seg {
            let angle = (i as f64 / seg as f64) * std::f64::consts::TAU;
            let (s, c) = angle.sin_cos();
            let nx = c;
            let nz = s;
            let u = i as f64 / seg as f64;
            vertices.push(Vertex::new(
                [radius * c, -half_h, radius * s],
                [nx, 0.0, nz],
                [u, 0.0],
            ));
            vertices.push(Vertex::new(
                [radius * c, half_h, radius * s],
                [nx, 0.0, nz],
                [u, 1.0],
            ));
        }

        // Side faces.
        for i in 0..seg {
            let base = i * 2;
            indices.push(Triangle(base, base + 2, base + 1));
            indices.push(Triangle(base + 1, base + 2, base + 3));
        }

        // Caps.
        let bottom_center = vertices.len() as u32;
        vertices.push(Vertex::new([0.0, -half_h, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]));
        let top_center = vertices.len() as u32;
        vertices.push(Vertex::new([0.0, half_h, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]));

        for i in 0..seg {
            let angle0 = (i as f64 / seg as f64) * std::f64::consts::TAU;
            let angle1 = ((i + 1) as f64 / seg as f64) * std::f64::consts::TAU;
            let (s0, c0) = angle0.sin_cos();
            let (s1, c1) = angle1.sin_cos();

            // Bottom cap.
            let b0 = vertices.len() as u32;
            vertices.push(Vertex::new(
                [radius * c0, -half_h, radius * s0],
                [0.0, -1.0, 0.0],
                [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5],
            ));
            let b1 = vertices.len() as u32;
            vertices.push(Vertex::new(
                [radius * c1, -half_h, radius * s1],
                [0.0, -1.0, 0.0],
                [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5],
            ));
            indices.push(Triangle(bottom_center, b1, b0));

            // Top cap.
            let t0 = vertices.len() as u32;
            vertices.push(Vertex::new(
                [radius * c0, half_h, radius * s0],
                [0.0, 1.0, 0.0],
                [c0 * 0.5 + 0.5, s0 * 0.5 + 0.5],
            ));
            let t1 = vertices.len() as u32;
            vertices.push(Vertex::new(
                [radius * c1, half_h, radius * s1],
                [0.0, 1.0, 0.0],
                [c1 * 0.5 + 0.5, s1 * 0.5 + 0.5],
            ));
            indices.push(Triangle(top_center, t0, t1));
        }

        Self { vertices, indices }
    }

    /// Generate a torus centered at origin lying on the XZ plane.
    pub fn torus(major_radius: f64, minor_radius: f64, major_segments: u32, minor_segments: u32) -> Self {
        let mseg = major_segments.max(3);
        let nseg = minor_segments.max(3);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for i in 0..=mseg {
            let u = i as f64 / mseg as f64;
            let theta = u * std::f64::consts::TAU;
            let (st, ct) = theta.sin_cos();

            for j in 0..=nseg {
                let v = j as f64 / nseg as f64;
                let phi = v * std::f64::consts::TAU;
                let (sp, cp) = phi.sin_cos();

                let x = (major_radius + minor_radius * cp) * ct;
                let y = minor_radius * sp;
                let z = (major_radius + minor_radius * cp) * st;

                let nx = cp * ct;
                let ny = sp;
                let nz = cp * st;

                vertices.push(Vertex::new([x, y, z], [nx, ny, nz], [u, v]));
            }
        }

        let ring_count = nseg + 1;
        for i in 0..mseg {
            for j in 0..nseg {
                let a = i * ring_count + j;
                let b = a + ring_count;
                let c = a + 1;
                let d = b + 1;
                indices.push(Triangle(a, b, c));
                indices.push(Triangle(c, b, d));
            }
        }

        Self { vertices, indices }
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for icosphere midpoint subdivision.
fn get_midpoint(
    a: u32,
    b: u32,
    positions: &mut Vec<[f64; 3]>,
    cache: &mut std::collections::HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&idx) = cache.get(&key) {
        return idx;
    }
    let pa = positions[a as usize];
    let pb = positions[b as usize];
    let mut mid = [
        (pa[0] + pb[0]) * 0.5,
        (pa[1] + pb[1]) * 0.5,
        (pa[2] + pb[2]) * 0.5,
    ];
    // Project onto unit sphere.
    let len = (mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2]).sqrt();
    mid[0] /= len;
    mid[1] /= len;
    mid[2] /= len;
    let idx = positions.len() as u32;
    positions.push(mid);
    cache.insert(key, idx);
    idx
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn cube_has_correct_topology() {
        let m = Mesh::cube();
        assert_eq!(m.vertex_count(), 24); // 6 faces * 4 verts
        assert_eq!(m.triangle_count(), 12); // 6 faces * 2 tris
    }

    #[test]
    fn sphere_subdivision_increases_detail() {
        let s0 = Mesh::sphere(0);
        let s1 = Mesh::sphere(1);
        assert!(s1.vertex_count() > s0.vertex_count());
        assert!(s1.triangle_count() > s0.triangle_count());
    }

    #[test]
    fn sphere_vertices_on_unit_sphere() {
        let s = Mesh::sphere(2);
        for v in &s.vertices {
            let len = (v.position[0].powi(2) + v.position[1].powi(2) + v.position[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < EPS, "vertex not on unit sphere: len={len}");
        }
    }

    #[test]
    fn plane_topology() {
        let m = Mesh::plane(10.0, 10.0, 4, 4);
        assert_eq!(m.vertex_count(), 5 * 5); // (4+1)*(4+1)
        assert_eq!(m.triangle_count(), 4 * 4 * 2);
    }

    #[test]
    fn cylinder_has_caps() {
        let m = Mesh::cylinder(1.0, 2.0, 8);
        // Should have side + cap vertices and triangles.
        assert!(m.vertex_count() > 16);
        assert!(m.triangle_count() > 16);
    }

    #[test]
    fn torus_topology() {
        let m = Mesh::torus(2.0, 0.5, 8, 6);
        assert_eq!(m.vertex_count(), 9 * 7); // (8+1)*(6+1)
        assert_eq!(m.triangle_count(), 8 * 6 * 2);
    }

    #[test]
    fn bounding_box_of_cube() {
        let m = Mesh::cube();
        let s = m.stats();
        assert!((s.bounding_box_min.x - (-1.0)).abs() < EPS);
        assert!((s.bounding_box_max.x - 1.0).abs() < EPS);
    }

    #[test]
    fn bounding_sphere_of_sphere() {
        let m = Mesh::sphere(2);
        let s = m.stats();
        // Center should be near origin.
        assert!(s.bounding_sphere_center.length_squared() < EPS);
        // Radius should be ~1.0.
        assert!((s.bounding_sphere_radius - 1.0).abs() < 0.01);
    }

    #[test]
    fn recalculate_normals_on_plane() {
        let mut m = Mesh::plane(2.0, 2.0, 1, 1);
        // Zero out normals first.
        for v in &mut m.vertices {
            v.normal = [0.0, 0.0, 0.0];
        }
        m.recalculate_normals();
        for v in &m.vertices {
            // Plane is on XZ, normals should point up.
            assert!((v.normal[1] - 1.0).abs() < EPS, "normal = {:?}", v.normal);
        }
    }

    #[test]
    fn empty_mesh_stats() {
        let m = Mesh::new();
        let s = m.stats();
        assert_eq!(s.vertex_count, 0);
        assert_eq!(s.triangle_count, 0);
        assert_eq!(s.bounding_sphere_radius, 0.0);
    }

    #[test]
    fn cube_normals_are_unit_length() {
        let m = Mesh::cube();
        for v in &m.vertices {
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!((len - 1.0).abs() < EPS, "normal not unit: len={len}");
        }
    }
}
