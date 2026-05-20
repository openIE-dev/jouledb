//! Programmatic mesh construction — build meshes from vertices, normals, UVs,
//! indices. Primitive generators: cube, sphere (UV and ico), cylinder, cone,
//! torus, plane, quad. Vertex merging (weld nearby vertices). Normal
//! recalculation from face normals. Tangent generation for normal mapping.
//! Mesh statistics (vertex count, triangle count, bounding box).

use std::f64::consts::PI;

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0 } }
}

// ── Vec3 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 { Self::zero() } else { self.scale(1.0 / len) }
    }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
    pub fn min_components(self, o: Self) -> Self {
        Self { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) }
    }
    pub fn max_components(self, o: Self) -> Self {
        Self { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) }
    }
}

// ── Vec4 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }
    pub fn zero() -> Self { Self { x: 0.0, y: 0.0, z: 0.0, w: 0.0 } }
}

// ── Vertex ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
    pub tangent: Vec4,
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> Self {
        Self { position, normal, uv, tangent: Vec4::zero() }
    }
}

// ── BoundingBox ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn empty() -> Self {
        Self {
            min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }
    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min_components(p);
        self.max = self.max.max_components(p);
    }
    pub fn center(&self) -> Vec3 { self.min.add(self.max).scale(0.5) }
    pub fn extents(&self) -> Vec3 { self.max.sub(self.min) }
}

// ── MeshStats ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct MeshStats {
    pub vertex_count: usize,
    pub triangle_count: usize,
    pub bounding_box: BoundingBox,
    pub has_normals: bool,
    pub has_uvs: bool,
    pub has_tangents: bool,
}

// ── Mesh ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new() -> Self { Self { vertices: Vec::new(), indices: Vec::new() } }

    pub fn stats(&self) -> MeshStats {
        let mut bb = BoundingBox::empty();
        let mut has_normals = false;
        let mut has_uvs = false;
        let mut has_tangents = false;
        for v in &self.vertices {
            bb.expand_point(v.position);
            if v.normal.length() > 1e-9 { has_normals = true; }
            if v.uv.x.abs() > 1e-12 || v.uv.y.abs() > 1e-12 { has_uvs = true; }
            if v.tangent.x.abs() > 1e-12 || v.tangent.y.abs() > 1e-12
                || v.tangent.z.abs() > 1e-12 { has_tangents = true; }
        }
        MeshStats {
            vertex_count: self.vertices.len(),
            triangle_count: self.indices.len() / 3,
            bounding_box: bb,
            has_normals,
            has_uvs,
            has_tangents,
        }
    }

    /// Recalculate normals from face geometry (flat shading).
    pub fn recalculate_normals(&mut self) {
        for v in self.vertices.iter_mut() {
            v.normal = Vec3::zero();
        }
        let mut i = 0;
        while i + 2 < self.indices.len() {
            let i0 = self.indices[i] as usize;
            let i1 = self.indices[i + 1] as usize;
            let i2 = self.indices[i + 2] as usize;
            let v0 = self.vertices[i0].position;
            let v1 = self.vertices[i1].position;
            let v2 = self.vertices[i2].position;
            let edge1 = v1.sub(v0);
            let edge2 = v2.sub(v0);
            let face_normal = edge1.cross(edge2);
            self.vertices[i0].normal = self.vertices[i0].normal.add(face_normal);
            self.vertices[i1].normal = self.vertices[i1].normal.add(face_normal);
            self.vertices[i2].normal = self.vertices[i2].normal.add(face_normal);
            i += 3;
        }
        for v in self.vertices.iter_mut() {
            v.normal = v.normal.normalized();
        }
    }

    /// Generate tangents for normal mapping (Lengyel's method).
    pub fn generate_tangents(&mut self) {
        let n = self.vertices.len();
        let mut tan1 = vec![Vec3::zero(); n];
        let mut tan2 = vec![Vec3::zero(); n];
        let mut i = 0;
        while i + 2 < self.indices.len() {
            let i0 = self.indices[i] as usize;
            let i1 = self.indices[i + 1] as usize;
            let i2 = self.indices[i + 2] as usize;
            let v0 = &self.vertices[i0];
            let v1 = &self.vertices[i1];
            let v2 = &self.vertices[i2];
            let dp1 = v1.position.sub(v0.position);
            let dp2 = v2.position.sub(v0.position);
            let duv1 = Vec2::new(v1.uv.x - v0.uv.x, v1.uv.y - v0.uv.y);
            let duv2 = Vec2::new(v2.uv.x - v0.uv.x, v2.uv.y - v0.uv.y);
            let r_denom = duv1.x * duv2.y - duv2.x * duv1.y;
            if r_denom.abs() > 1e-12 {
                let r = 1.0 / r_denom;
                let s_dir = Vec3::new(
                    (duv2.y * dp1.x - duv1.y * dp2.x) * r,
                    (duv2.y * dp1.y - duv1.y * dp2.y) * r,
                    (duv2.y * dp1.z - duv1.y * dp2.z) * r,
                );
                let t_dir = Vec3::new(
                    (duv1.x * dp2.x - duv2.x * dp1.x) * r,
                    (duv1.x * dp2.y - duv2.x * dp1.y) * r,
                    (duv1.x * dp2.z - duv2.x * dp1.z) * r,
                );
                tan1[i0] = tan1[i0].add(s_dir);
                tan1[i1] = tan1[i1].add(s_dir);
                tan1[i2] = tan1[i2].add(s_dir);
                tan2[i0] = tan2[i0].add(t_dir);
                tan2[i1] = tan2[i1].add(t_dir);
                tan2[i2] = tan2[i2].add(t_dir);
            }
            i += 3;
        }
        for j in 0..n {
            let norm = self.vertices[j].normal;
            let t = tan1[j];
            // Gram-Schmidt orthogonalize
            let tangent = t.sub(norm.scale(norm.dot(t))).normalized();
            // Handedness
            let w = if norm.cross(t).dot(tan2[j]) < 0.0 { -1.0 } else { 1.0 };
            self.vertices[j].tangent = Vec4::new(tangent.x, tangent.y, tangent.z, w);
        }
    }

    /// Weld vertices closer than `threshold`.
    pub fn weld_vertices(&mut self, threshold: f64) {
        let threshold_sq = threshold * threshold;
        let old_verts = self.vertices.clone();
        let mut remap = vec![0u32; old_verts.len()];
        let mut new_verts: Vec<Vertex> = Vec::new();
        for (i, v) in old_verts.iter().enumerate() {
            let mut found = None;
            for (j, nv) in new_verts.iter().enumerate() {
                if v.position.sub(nv.position).dot(v.position.sub(nv.position)) < threshold_sq {
                    found = Some(j);
                    break;
                }
            }
            match found {
                Some(j) => remap[i] = j as u32,
                None => {
                    remap[i] = new_verts.len() as u32;
                    new_verts.push(*v);
                }
            }
        }
        for idx in self.indices.iter_mut() {
            *idx = remap[*idx as usize];
        }
        self.vertices = new_verts;
    }
}

// ── MeshBuilder ──────────────────────────────────────────────

pub struct MeshBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    pub fn new() -> Self { Self { vertices: Vec::new(), indices: Vec::new() } }

    pub fn add_vertex(&mut self, position: Vec3, normal: Vec3, uv: Vec2) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(Vertex::new(position, normal, uv));
        idx
    }

    pub fn add_triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.push(a);
        self.indices.push(b);
        self.indices.push(c);
    }

    pub fn add_quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.add_triangle(a, b, c);
        self.add_triangle(a, c, d);
    }

    pub fn build(self) -> Mesh {
        Mesh { vertices: self.vertices, indices: self.indices }
    }

    /// Generate a unit quad in XY plane centered at origin.
    pub fn quad() -> Mesh {
        let mut b = Self::new();
        let n = Vec3::new(0.0, 0.0, 1.0);
        let v0 = b.add_vertex(Vec3::new(-0.5, -0.5, 0.0), n, Vec2::new(0.0, 0.0));
        let v1 = b.add_vertex(Vec3::new( 0.5, -0.5, 0.0), n, Vec2::new(1.0, 0.0));
        let v2 = b.add_vertex(Vec3::new( 0.5,  0.5, 0.0), n, Vec2::new(1.0, 1.0));
        let v3 = b.add_vertex(Vec3::new(-0.5,  0.5, 0.0), n, Vec2::new(0.0, 1.0));
        b.add_quad(v0, v1, v2, v3);
        b.build()
    }

    /// Generate a plane in XZ with given subdivisions.
    pub fn plane(subdivisions: u32) -> Mesh {
        let mut b = Self::new();
        let segs = subdivisions.max(1);
        let n = Vec3::new(0.0, 1.0, 0.0);
        for iy in 0..=segs {
            for ix in 0..=segs {
                let u = ix as f64 / segs as f64;
                let v = iy as f64 / segs as f64;
                b.add_vertex(
                    Vec3::new(u - 0.5, 0.0, v - 0.5),
                    n,
                    Vec2::new(u, v),
                );
            }
        }
        let cols = segs + 1;
        for iy in 0..segs {
            for ix in 0..segs {
                let bl = iy * cols + ix;
                let br = bl + 1;
                let tl = bl + cols;
                let tr = tl + 1;
                b.add_quad(bl, br, tr, tl);
            }
        }
        b.build()
    }

    /// Generate a unit cube centered at origin.
    pub fn cube() -> Mesh {
        let mut b = Self::new();
        // Six faces, each with 4 vertices
        let faces: [([f64; 3], [f64; 3]); 6] = [
            ([0.0, 0.0,  1.0], [0.0, 0.0,  1.0]),  // front
            ([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]),  // back
            ([ 1.0, 0.0, 0.0], [ 1.0, 0.0, 0.0]),  // right
            ([-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),  // left
            ([0.0,  1.0, 0.0], [0.0,  1.0, 0.0]),  // top
            ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0]),  // bottom
        ];
        let uvs = [
            Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 1.0), Vec2::new(0.0, 1.0),
        ];
        for (norm_arr, _) in &faces {
            let n = Vec3::new(norm_arr[0], norm_arr[1], norm_arr[2]);
            let (right, up) = if n.y.abs() > 0.9 {
                (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, if n.y > 0.0 { -1.0 } else { 1.0 }))
            } else {
                let right = Vec3::new(0.0, 1.0, 0.0).cross(n).normalized();
                let up = n.cross(right);
                (right, up)
            };
            let center = n.scale(0.5);
            let corners = [
                center.add(right.scale(-0.5)).add(up.scale(-0.5)),
                center.add(right.scale( 0.5)).add(up.scale(-0.5)),
                center.add(right.scale( 0.5)).add(up.scale( 0.5)),
                center.add(right.scale(-0.5)).add(up.scale( 0.5)),
            ];
            let base = b.vertices.len() as u32;
            for (ci, corner) in corners.iter().enumerate() {
                b.add_vertex(*corner, n, uvs[ci]);
            }
            b.add_quad(base, base + 1, base + 2, base + 3);
        }
        b.build()
    }

    /// Generate a UV sphere with given segments and rings.
    pub fn uv_sphere(segments: u32, rings: u32) -> Mesh {
        let mut b = Self::new();
        let segs = segments.max(3);
        let rngs = rings.max(2);
        for iy in 0..=rngs {
            let v = iy as f64 / rngs as f64;
            let phi = v * PI;
            for ix in 0..=segs {
                let u = ix as f64 / segs as f64;
                let theta = u * 2.0 * PI;
                let pos = Vec3::new(
                    phi.sin() * theta.cos(),
                    phi.cos(),
                    phi.sin() * theta.sin(),
                );
                b.add_vertex(pos, pos.normalized(), Vec2::new(u, v));
            }
        }
        let cols = segs + 1;
        for iy in 0..rngs {
            for ix in 0..segs {
                let curr = iy * cols + ix;
                let next = curr + cols;
                b.add_triangle(curr, next, curr + 1);
                b.add_triangle(curr + 1, next, next + 1);
            }
        }
        b.build()
    }

    /// Generate an icosphere by recursive subdivision.
    pub fn ico_sphere(subdivisions: u32) -> Mesh {
        let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let mut positions = vec![
            Vec3::new(-1.0,  t, 0.0).normalized(),
            Vec3::new( 1.0,  t, 0.0).normalized(),
            Vec3::new(-1.0, -t, 0.0).normalized(),
            Vec3::new( 1.0, -t, 0.0).normalized(),
            Vec3::new(0.0, -1.0,  t).normalized(),
            Vec3::new(0.0,  1.0,  t).normalized(),
            Vec3::new(0.0, -1.0, -t).normalized(),
            Vec3::new(0.0,  1.0, -t).normalized(),
            Vec3::new( t, 0.0, -1.0).normalized(),
            Vec3::new( t, 0.0,  1.0).normalized(),
            Vec3::new(-t, 0.0, -1.0).normalized(),
            Vec3::new(-t, 0.0,  1.0).normalized(),
        ];
        let mut tris: Vec<[usize; 3]> = vec![
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
        ];
        for _ in 0..subdivisions {
            let mut new_tris = Vec::new();
            let mut midpoint_cache = std::collections::HashMap::new();
            for tri in &tris {
                let mut mids = [0usize; 3];
                for edge in 0..3 {
                    let (a, b_idx) = (tri[edge], tri[(edge + 1) % 3]);
                    let key = if a < b_idx { (a, b_idx) } else { (b_idx, a) };
                    mids[edge] = *midpoint_cache.entry(key).or_insert_with(|| {
                        let mid = positions[a].add(positions[b_idx]).scale(0.5).normalized();
                        positions.push(mid);
                        positions.len() - 1
                    });
                }
                new_tris.push([tri[0], mids[0], mids[2]]);
                new_tris.push([tri[1], mids[1], mids[0]]);
                new_tris.push([tri[2], mids[2], mids[1]]);
                new_tris.push([mids[0], mids[1], mids[2]]);
            }
            tris = new_tris;
        }
        let mut builder = Self::new();
        for p in &positions {
            let uv = Vec2::new(
                0.5 + p.z.atan2(p.x) / (2.0 * PI),
                0.5 - p.y.asin() / PI,
            );
            builder.add_vertex(*p, p.normalized(), uv);
        }
        for tri in &tris {
            builder.add_triangle(tri[0] as u32, tri[1] as u32, tri[2] as u32);
        }
        builder.build()
    }

    /// Generate a cylinder along Y axis.
    pub fn cylinder(segments: u32, height: f64, radius: f64) -> Mesh {
        let mut b = Self::new();
        let segs = segments.max(3);
        let half_h = height * 0.5;
        // Side vertices
        for iy in 0..=1u32 {
            let y = if iy == 0 { -half_h } else { half_h };
            for ix in 0..=segs {
                let u = ix as f64 / segs as f64;
                let theta = u * 2.0 * PI;
                let nx = theta.cos();
                let nz = theta.sin();
                b.add_vertex(
                    Vec3::new(nx * radius, y, nz * radius),
                    Vec3::new(nx, 0.0, nz),
                    Vec2::new(u, iy as f64),
                );
            }
        }
        let cols = segs + 1;
        for ix in 0..segs {
            let bl = ix;
            let br = ix + 1;
            let tl = ix + cols;
            let tr = ix + 1 + cols;
            b.add_quad(bl, br, tr, tl);
        }
        // Top cap
        let top_center = b.add_vertex(
            Vec3::new(0.0, half_h, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::new(0.5, 0.5),
        );
        let top_base = b.vertices.len() as u32;
        for ix in 0..=segs {
            let u = ix as f64 / segs as f64;
            let theta = u * 2.0 * PI;
            b.add_vertex(
                Vec3::new(theta.cos() * radius, half_h, theta.sin() * radius),
                Vec3::new(0.0, 1.0, 0.0),
                Vec2::new(0.5 + theta.cos() * 0.5, 0.5 + theta.sin() * 0.5),
            );
        }
        for ix in 0..segs {
            b.add_triangle(top_center, top_base + ix, top_base + ix + 1);
        }
        // Bottom cap
        let bot_center = b.add_vertex(
            Vec3::new(0.0, -half_h, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec2::new(0.5, 0.5),
        );
        let bot_base = b.vertices.len() as u32;
        for ix in 0..=segs {
            let u = ix as f64 / segs as f64;
            let theta = u * 2.0 * PI;
            b.add_vertex(
                Vec3::new(theta.cos() * radius, -half_h, theta.sin() * radius),
                Vec3::new(0.0, -1.0, 0.0),
                Vec2::new(0.5 + theta.cos() * 0.5, 0.5 + theta.sin() * 0.5),
            );
        }
        for ix in 0..segs {
            b.add_triangle(bot_center, bot_base + ix + 1, bot_base + ix);
        }
        b.build()
    }

    /// Generate a cone along Y axis.
    pub fn cone(segments: u32, height: f64, radius: f64) -> Mesh {
        let mut b = Self::new();
        let segs = segments.max(3);
        let half_h = height * 0.5;
        let slope_len = (radius * radius + height * height).sqrt();
        let ny = radius / slope_len;
        let nr = height / slope_len;
        // Side
        let apex = b.add_vertex(
            Vec3::new(0.0, half_h, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::new(0.5, 0.0),
        );
        let base_start = b.vertices.len() as u32;
        for ix in 0..=segs {
            let u = ix as f64 / segs as f64;
            let theta = u * 2.0 * PI;
            let nx = theta.cos() * nr;
            let nz = theta.sin() * nr;
            b.add_vertex(
                Vec3::new(theta.cos() * radius, -half_h, theta.sin() * radius),
                Vec3::new(nx, ny, nz).normalized(),
                Vec2::new(u, 1.0),
            );
        }
        for ix in 0..segs {
            b.add_triangle(apex, base_start + ix, base_start + ix + 1);
        }
        // Bottom cap
        let bot_center = b.add_vertex(
            Vec3::new(0.0, -half_h, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec2::new(0.5, 0.5),
        );
        let cap_start = b.vertices.len() as u32;
        for ix in 0..=segs {
            let u = ix as f64 / segs as f64;
            let theta = u * 2.0 * PI;
            b.add_vertex(
                Vec3::new(theta.cos() * radius, -half_h, theta.sin() * radius),
                Vec3::new(0.0, -1.0, 0.0),
                Vec2::new(0.5 + theta.cos() * 0.5, 0.5 + theta.sin() * 0.5),
            );
        }
        for ix in 0..segs {
            b.add_triangle(bot_center, cap_start + ix + 1, cap_start + ix);
        }
        b.build()
    }

    /// Generate a torus centered at origin in XZ plane.
    pub fn torus(major_segments: u32, minor_segments: u32, major_radius: f64, minor_radius: f64) -> Mesh {
        let mut b = Self::new();
        let maj = major_segments.max(3);
        let min = minor_segments.max(3);
        for iy in 0..=maj {
            let u = iy as f64 / maj as f64;
            let theta = u * 2.0 * PI;
            let cx = theta.cos() * major_radius;
            let cz = theta.sin() * major_radius;
            for ix in 0..=min {
                let v = ix as f64 / min as f64;
                let phi = v * 2.0 * PI;
                let px = (major_radius + minor_radius * phi.cos()) * theta.cos();
                let py = minor_radius * phi.sin();
                let pz = (major_radius + minor_radius * phi.cos()) * theta.sin();
                let nx = px - cx;
                let ny = py;
                let nz = pz - cz;
                b.add_vertex(
                    Vec3::new(px, py, pz),
                    Vec3::new(nx, ny, nz).normalized(),
                    Vec2::new(u, v),
                );
            }
        }
        let cols = min + 1;
        for iy in 0..maj {
            for ix in 0..min {
                let curr = iy * cols + ix;
                let next = curr + cols;
                b.add_quad(curr, curr + 1, next + 1, next);
            }
        }
        b.build()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }
    fn v3_approx(a: Vec3, b: Vec3, eps: f64) -> bool {
        approx_eq(a.x, b.x, eps) && approx_eq(a.y, b.y, eps) && approx_eq(a.z, b.z, eps)
    }

    #[test]
    fn test_quad_stats() {
        let m = MeshBuilder::quad();
        let s = m.stats();
        assert_eq!(s.vertex_count, 4);
        assert_eq!(s.triangle_count, 2);
    }

    #[test]
    fn test_quad_bounding_box() {
        let m = MeshBuilder::quad();
        let s = m.stats();
        assert!(approx_eq(s.bounding_box.min.x, -0.5, 1e-9));
        assert!(approx_eq(s.bounding_box.max.x,  0.5, 1e-9));
    }

    #[test]
    fn test_cube_stats() {
        let m = MeshBuilder::cube();
        let s = m.stats();
        assert_eq!(s.vertex_count, 24); // 6 faces * 4 verts
        assert_eq!(s.triangle_count, 12);
    }

    #[test]
    fn test_cube_bounding_box() {
        let m = MeshBuilder::cube();
        let bb = m.stats().bounding_box;
        assert!(approx_eq(bb.min.x, -0.5, 1e-9));
        assert!(approx_eq(bb.max.x,  0.5, 1e-9));
        assert!(approx_eq(bb.min.y, -0.5, 1e-9));
        assert!(approx_eq(bb.max.y,  0.5, 1e-9));
    }

    #[test]
    fn test_uv_sphere_stats() {
        let m = MeshBuilder::uv_sphere(8, 4);
        let s = m.stats();
        assert_eq!(s.vertex_count, 45); // (8+1)*(4+1)
        assert!(s.triangle_count > 0);
    }

    #[test]
    fn test_uv_sphere_radius() {
        let m = MeshBuilder::uv_sphere(16, 8);
        for v in &m.vertices {
            assert!(approx_eq(v.position.length(), 1.0, 1e-9));
        }
    }

    #[test]
    fn test_ico_sphere_radius() {
        let m = MeshBuilder::ico_sphere(2);
        for v in &m.vertices {
            assert!(approx_eq(v.position.length(), 1.0, 1e-9));
        }
    }

    #[test]
    fn test_ico_sphere_subdivisions() {
        let m0 = MeshBuilder::ico_sphere(0);
        let m1 = MeshBuilder::ico_sphere(1);
        assert!(m1.stats().vertex_count > m0.stats().vertex_count);
        assert!(m1.stats().triangle_count > m0.stats().triangle_count);
    }

    #[test]
    fn test_cylinder_stats() {
        let m = MeshBuilder::cylinder(8, 2.0, 1.0);
        assert!(m.stats().vertex_count > 0);
        assert!(m.stats().triangle_count > 0);
    }

    #[test]
    fn test_cylinder_height() {
        let m = MeshBuilder::cylinder(8, 4.0, 1.0);
        let bb = m.stats().bounding_box;
        assert!(approx_eq(bb.min.y, -2.0, 1e-9));
        assert!(approx_eq(bb.max.y,  2.0, 1e-9));
    }

    #[test]
    fn test_cone_stats() {
        let m = MeshBuilder::cone(8, 2.0, 1.0);
        assert!(m.stats().vertex_count > 0);
        assert!(m.stats().triangle_count > 0);
    }

    #[test]
    fn test_torus_stats() {
        let m = MeshBuilder::torus(12, 6, 1.0, 0.3);
        assert!(m.stats().vertex_count > 0);
        assert!(m.stats().triangle_count > 0);
    }

    #[test]
    fn test_plane_subdivisions() {
        let m = MeshBuilder::plane(4);
        let s = m.stats();
        assert_eq!(s.vertex_count, 25); // 5*5
        assert_eq!(s.triangle_count, 32); // 4*4*2
    }

    #[test]
    fn test_recalculate_normals() {
        let mut m = MeshBuilder::quad();
        for v in m.vertices.iter_mut() { v.normal = Vec3::zero(); }
        m.recalculate_normals();
        for v in &m.vertices {
            assert!(v.normal.length() > 0.9);
        }
    }

    #[test]
    fn test_generate_tangents() {
        let mut m = MeshBuilder::quad();
        m.generate_tangents();
        for v in &m.vertices {
            let t_len = Vec3::new(v.tangent.x, v.tangent.y, v.tangent.z).length();
            assert!(t_len > 0.9, "tangent should be unit length, got {}", t_len);
        }
    }

    #[test]
    fn test_tangent_handedness() {
        let mut m = MeshBuilder::quad();
        m.generate_tangents();
        for v in &m.vertices {
            assert!(v.tangent.w == 1.0 || v.tangent.w == -1.0);
        }
    }

    #[test]
    fn test_weld_vertices() {
        let mut b = MeshBuilder::new();
        let v0 = b.add_vertex(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::zero());
        let v1 = b.add_vertex(Vec3::new(0.0001, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::zero());
        let v2 = b.add_vertex(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::zero());
        b.add_triangle(v0, v1, v2);
        let mut mesh = b.build();
        assert_eq!(mesh.vertices.len(), 3);
        mesh.weld_vertices(0.001);
        assert_eq!(mesh.vertices.len(), 2); // v0 and v1 merged
    }

    #[test]
    fn test_weld_preserves_topology() {
        let mut b = MeshBuilder::new();
        let v0 = b.add_vertex(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::zero());
        let v1 = b.add_vertex(Vec3::new(0.0001, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::zero());
        let v2 = b.add_vertex(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec2::zero());
        b.add_triangle(v0, v1, v2);
        let mut mesh = b.build();
        mesh.weld_vertices(0.001);
        assert_eq!(mesh.indices.len(), 3);
        assert_eq!(mesh.indices[0], mesh.indices[1]); // merged
    }

    #[test]
    fn test_builder_add_vertex_returns_index() {
        let mut b = MeshBuilder::new();
        assert_eq!(b.add_vertex(Vec3::zero(), Vec3::zero(), Vec2::zero()), 0);
        assert_eq!(b.add_vertex(Vec3::zero(), Vec3::zero(), Vec2::zero()), 1);
        assert_eq!(b.add_vertex(Vec3::zero(), Vec3::zero(), Vec2::zero()), 2);
    }

    #[test]
    fn test_empty_mesh() {
        let m = Mesh::new();
        let s = m.stats();
        assert_eq!(s.vertex_count, 0);
        assert_eq!(s.triangle_count, 0);
    }

    #[test]
    fn test_bounding_box_center() {
        let m = MeshBuilder::cube();
        let bb = m.stats().bounding_box;
        let c = bb.center();
        assert!(v3_approx(c, Vec3::zero(), 1e-9));
    }

    #[test]
    fn test_sphere_normals_point_outward() {
        let m = MeshBuilder::uv_sphere(8, 4);
        for v in &m.vertices {
            let dot = v.position.normalized().dot(v.normal);
            assert!(dot > 0.99, "normal should align with radial direction, got dot={}", dot);
        }
    }

    #[test]
    fn test_torus_center_hole() {
        let m = MeshBuilder::torus(16, 8, 2.0, 0.5);
        let bb = m.stats().bounding_box;
        assert!(approx_eq(bb.max.x, 2.5, 1e-6));
        assert!(approx_eq(bb.min.x, -2.5, 1e-6));
    }
}
