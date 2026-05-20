//! Lathe Mesh — revolution mesh generator. Rotates a 2D profile (polyline on XY plane)
//! around the Y axis. Configurable radial segments, partial sweep, smooth/flat shading,
//! UV mapping, cap generation, and open/closed profile handling.

use std::f64::consts::PI;

// ── Vector types ───────────────────────────────────────────────

/// 2D point on the XY profile plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
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
        if len < 1e-12 { Self::zero() } else { Self::new(self.x / len, self.y / len, self.z / len) }
    }
    pub fn add(&self, o: &Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(&self, o: &Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn cross(&self, o: &Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn dot(&self, o: &Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
}

// ── Lathe mesh output ──────────────────────────────────────────

/// A generated lathe mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct LatheMesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f64; 2]>,
    pub indices: Vec<[u32; 3]>,
}

impl LatheMesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
}

/// Shading mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadingMode {
    Smooth,
    Flat,
}

/// Configuration for the lathe operation.
#[derive(Debug, Clone)]
pub struct LatheConfig {
    /// 2D profile polyline on the XY plane. X = radial distance from Y axis, Y = height.
    pub profile: Vec<Vec2>,
    /// Number of radial segments.
    pub segments: usize,
    /// Sweep angle in radians (0, 2π]. Use 2π for full revolution.
    pub sweep_angle: f64,
    /// Shading mode.
    pub shading: ShadingMode,
    /// Whether the profile is closed (first point connects to last).
    pub closed_profile: bool,
    /// Whether to generate caps for partial revolution.
    pub generate_caps: bool,
}

impl LatheConfig {
    pub fn new(profile: Vec<Vec2>, segments: usize) -> Self {
        Self {
            profile,
            segments: segments.max(3),
            sweep_angle: 2.0 * PI,
            shading: ShadingMode::Smooth,
            closed_profile: false,
            generate_caps: true,
        }
    }

    pub fn with_sweep(mut self, angle: f64) -> Self {
        self.sweep_angle = angle.clamp(0.001, 2.0 * PI);
        self
    }

    pub fn with_shading(mut self, mode: ShadingMode) -> Self {
        self.shading = mode;
        self
    }

    pub fn with_closed_profile(mut self, closed: bool) -> Self {
        self.closed_profile = closed;
        self
    }

    pub fn with_caps(mut self, caps: bool) -> Self {
        self.generate_caps = caps;
        self
    }
}

/// Compute the 2D profile normal at vertex index i (perpendicular to profile edges).
fn profile_normal_2d(profile: &[Vec2], i: usize, closed: bool) -> Vec2 {
    let n = profile.len();
    if n < 2 {
        return Vec2::new(1.0, 0.0);
    }

    let prev = if i == 0 {
        if closed { n - 1 } else { 0 }
    } else {
        i - 1
    };
    let next = if i == n - 1 {
        if closed { 0 } else { n - 1 }
    } else {
        i + 1
    };

    let edge = profile[next].sub(&profile[prev]);
    // Perpendicular: rotate 90° → (dy, -dx), then normalize
    let nx = edge.y;
    let ny = -edge.x;
    let len = (nx * nx + ny * ny).sqrt();
    if len < 1e-12 {
        Vec2::new(1.0, 0.0)
    } else {
        Vec2::new(nx / len, ny / len)
    }
}

/// Generate a lathe mesh.
pub fn generate_lathe(config: &LatheConfig) -> LatheMesh {
    let profile = &config.profile;
    let segs = config.segments;
    let sweep = config.sweep_angle;
    let is_full = (sweep - 2.0 * PI).abs() < 1e-6;

    if profile.len() < 2 {
        return LatheMesh {
            vertices: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
        };
    }

    let p_len = profile.len();
    let ring_count = if is_full { segs } else { segs + 1 };

    // Compute profile arc lengths for V coordinate
    let mut profile_dists = Vec::with_capacity(p_len);
    profile_dists.push(0.0);
    for i in 1..p_len {
        let d = profile[i].sub(&profile[i - 1]).length();
        profile_dists.push(profile_dists[i - 1] + d);
    }
    let total_profile_len = *profile_dists.last().unwrap_or(&1.0);
    let total_profile_len = if total_profile_len < 1e-12 { 1.0 } else { total_profile_len };

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    match config.shading {
        ShadingMode::Smooth => {
            for ri in 0..ring_count {
                let angle = sweep * ri as f64 / segs as f64;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let u = ri as f64 / segs as f64;

                for pi in 0..p_len {
                    let r = profile[pi].x;
                    let y = profile[pi].y;
                    vertices.push(Vec3::new(r * cos_a, y, r * sin_a));

                    let pn = profile_normal_2d(profile, pi, config.closed_profile);
                    normals.push(Vec3::new(pn.x * cos_a, pn.y, pn.x * sin_a).normalized());

                    uvs.push([u, profile_dists[pi] / total_profile_len]);
                }
            }
        }
        ShadingMode::Flat => {
            // Duplicate vertices per face for flat normals
            for ri in 0..ring_count {
                let angle = sweep * ri as f64 / segs as f64;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let u = ri as f64 / segs as f64;

                for pi in 0..p_len {
                    let r = profile[pi].x;
                    let y = profile[pi].y;
                    vertices.push(Vec3::new(r * cos_a, y, r * sin_a));
                    // Flat normal will be recomputed after indices
                    normals.push(Vec3::zero());
                    uvs.push([u, profile_dists[pi] / total_profile_len]);
                }
            }
        }
    }

    // Build indices for the side surface
    let mut indices = Vec::new();
    let edge_count_profile = if config.closed_profile { p_len } else { p_len - 1 };

    for ri in 0..segs {
        let cur_ring = if is_full { ri % ring_count } else { ri };
        let next_ring = if is_full { (ri + 1) % ring_count } else { ri + 1 };

        for pi in 0..edge_count_profile {
            let next_pi = (pi + 1) % p_len;

            let a = (cur_ring * p_len + pi) as u32;
            let b = (cur_ring * p_len + next_pi) as u32;
            let c = (next_ring * p_len + next_pi) as u32;
            let d = (next_ring * p_len + pi) as u32;

            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }

    // Flat shading: recompute normals from face geometry
    if config.shading == ShadingMode::Flat {
        let mut accum = vec![Vec3::zero(); vertices.len()];
        for tri in &indices {
            let va = vertices[tri[0] as usize];
            let vb = vertices[tri[1] as usize];
            let vc = vertices[tri[2] as usize];
            let face_n = vb.sub(&va).cross(&vc.sub(&va)).normalized();
            for &idx in tri {
                let i = idx as usize;
                accum[i] = accum[i].add(&face_n);
            }
        }
        normals = accum.into_iter().map(|n| n.normalized()).collect();
    }

    // Caps for partial revolution
    if !is_full && config.generate_caps {
        // Start cap (angle = 0)
        let cap_center_start = vertices.len() as u32;
        let centroid_y: f64 = profile.iter().map(|p| p.y).sum::<f64>() / p_len as f64;
        let centroid_x: f64 = profile.iter().map(|p| p.x).sum::<f64>() / p_len as f64;

        vertices.push(Vec3::new(centroid_x, centroid_y, 0.0));
        normals.push(Vec3::new(0.0, 0.0, -1.0));
        uvs.push([0.5, 0.5]);

        for pi in 0..edge_count_profile {
            let next_pi = (pi + 1) % p_len;
            let a = cap_center_start;
            let b = pi as u32; // first ring
            let c = next_pi as u32;
            indices.push([a, c, b]); // reversed winding for inward cap
        }

        // End cap (angle = sweep)
        let end_angle = sweep;
        let cos_e = end_angle.cos();
        let sin_e = end_angle.sin();
        let cap_center_end = vertices.len() as u32;
        vertices.push(Vec3::new(centroid_x * cos_e, centroid_y, centroid_x * sin_e));
        normals.push(Vec3::new(-sin_e, 0.0, cos_e).normalized());
        uvs.push([0.5, 0.5]);

        let last_ring_base = (ring_count - 1) * p_len;
        for pi in 0..edge_count_profile {
            let next_pi = (pi + 1) % p_len;
            let a = cap_center_end;
            let b = (last_ring_base + pi) as u32;
            let c = (last_ring_base + next_pi) as u32;
            indices.push([a, b, c]);
        }
    }

    LatheMesh {
        vertices,
        normals,
        uvs,
        indices,
    }
}

/// Compute the volume of a lathe mesh (assumes watertight mesh, uses divergence theorem).
pub fn lathe_volume(mesh: &LatheMesh) -> f64 {
    let mut vol = 0.0;
    for tri in &mesh.indices {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        // Signed volume of tetrahedron formed by triangle and origin
        vol += a.x * (b.y * c.z - c.y * b.z)
            - b.x * (a.y * c.z - c.y * a.z)
            + c.x * (a.y * b.z - b.y * a.z);
    }
    (vol / 6.0).abs()
}

/// Compute total surface area of the mesh.
pub fn lathe_surface_area(mesh: &LatheMesh) -> f64 {
    let mut area = 0.0;
    for tri in &mesh.indices {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        area += b.sub(&a).cross(&c.sub(&a)).length() * 0.5;
    }
    area
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

    /// Simple rectangle profile: a cylinder outline
    fn cylinder_profile(radius: f64, height: f64) -> Vec<Vec2> {
        vec![
            Vec2::new(radius, 0.0),
            Vec2::new(radius, height),
        ]
    }

    /// Circle profile for a torus-like shape
    fn circle_profile(radius: f64, center_x: f64, n: usize) -> Vec<Vec2> {
        let mut pts = Vec::with_capacity(n);
        for i in 0..n {
            let angle = 2.0 * PI * i as f64 / n as f64;
            pts.push(Vec2::new(center_x + radius * angle.cos(), radius * angle.sin()));
        }
        pts
    }

    #[test]
    fn test_cylinder_vertex_count() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 16);
        let mesh = generate_lathe(&config);
        // Full revolution, smooth: 16 rings × 2 profile points = 32
        assert_eq!(mesh.vertex_count(), 16 * 2);
    }

    #[test]
    fn test_cylinder_triangle_count() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 16);
        let mesh = generate_lathe(&config);
        // 16 segments × 1 profile edge × 2 tris
        assert_eq!(mesh.triangle_count(), 16 * 1 * 2);
    }

    #[test]
    fn test_cylinder_radius() {
        let config = LatheConfig::new(cylinder_profile(2.0, 3.0), 32);
        let mesh = generate_lathe(&config);
        for v in &mesh.vertices {
            let r = (v.x * v.x + v.z * v.z).sqrt();
            assert!(approx(r, 2.0, EPS));
        }
    }

    #[test]
    fn test_cylinder_height_range() {
        let config = LatheConfig::new(cylinder_profile(1.0, 5.0), 16);
        let mesh = generate_lathe(&config);
        for v in &mesh.vertices {
            assert!(v.y >= -EPS && v.y <= 5.0 + EPS);
        }
    }

    #[test]
    fn test_partial_sweep_vertex_count() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 8)
            .with_sweep(PI); // half revolution
        let mesh = generate_lathe(&config);
        // Partial: (8+1) rings × 2 = 18 + 2 cap centers
        assert_eq!(mesh.vertex_count(), 9 * 2 + 2);
    }

    #[test]
    fn test_partial_sweep_caps() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 8)
            .with_sweep(PI)
            .with_caps(true);
        let mesh = generate_lathe(&config);
        // Has cap geometry
        let config_no_caps = LatheConfig::new(cylinder_profile(1.0, 2.0), 8)
            .with_sweep(PI)
            .with_caps(false);
        let mesh_no_caps = generate_lathe(&config_no_caps);
        assert!(mesh.triangle_count() > mesh_no_caps.triangle_count());
    }

    #[test]
    fn test_full_sweep_no_caps() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 16)
            .with_caps(true);
        let mesh = generate_lathe(&config);
        let config2 = LatheConfig::new(cylinder_profile(1.0, 2.0), 16)
            .with_caps(false);
        let mesh2 = generate_lathe(&config2);
        // Full revolution should not add caps regardless of setting
        assert_eq!(mesh.triangle_count(), mesh2.triangle_count());
    }

    #[test]
    fn test_flat_shading() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 8)
            .with_shading(ShadingMode::Flat);
        let mesh = generate_lathe(&config);
        // Normals should be computed (not all zero)
        let has_nonzero = mesh.normals.iter().any(|n| n.length() > 0.5);
        assert!(has_nonzero);
    }

    #[test]
    fn test_smooth_normals_unit() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 16);
        let mesh = generate_lathe(&config);
        for n in &mesh.normals {
            let len = n.length();
            assert!(approx(len, 1.0, 0.1), "normal length: {len}");
        }
    }

    #[test]
    fn test_uv_u_range() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 16);
        let mesh = generate_lathe(&config);
        for uv in &mesh.uvs {
            assert!(uv[0] >= -EPS && uv[0] <= 1.0 + EPS);
            assert!(uv[1] >= -EPS && uv[1] <= 1.0 + EPS);
        }
    }

    #[test]
    fn test_indices_in_bounds() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 12);
        let mesh = generate_lathe(&config);
        let max_idx = mesh.vertex_count() as u32;
        for tri in &mesh.indices {
            assert!(tri[0] < max_idx);
            assert!(tri[1] < max_idx);
            assert!(tri[2] < max_idx);
        }
    }

    #[test]
    fn test_closed_profile() {
        let profile = circle_profile(0.3, 1.5, 8);
        let config = LatheConfig::new(profile, 16)
            .with_closed_profile(true);
        let mesh = generate_lathe(&config);
        // Closed profile: 16 segments × 8 edges × 2 tris = 256
        assert_eq!(mesh.triangle_count(), 16 * 8 * 2);
    }

    #[test]
    fn test_profile_too_short() {
        let config = LatheConfig::new(vec![Vec2::new(1.0, 0.0)], 8);
        let mesh = generate_lathe(&config);
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn test_min_segments_clamped() {
        let config = LatheConfig::new(cylinder_profile(1.0, 1.0), 1);
        assert!(config.segments >= 3);
    }

    #[test]
    fn test_lathe_surface_area_cylinder() {
        // Cylinder: r=1, h=2, lateral area = 2πrh ≈ 12.566
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 64);
        let mesh = generate_lathe(&config);
        let area = lathe_surface_area(&mesh);
        assert!(approx(area, 2.0 * PI * 1.0 * 2.0, 0.3));
    }

    #[test]
    fn test_cone_profile() {
        let profile = vec![
            Vec2::new(0.0, 2.0), // apex
            Vec2::new(1.0, 0.0), // base
        ];
        let config = LatheConfig::new(profile, 32);
        let mesh = generate_lathe(&config);
        assert!(mesh.vertex_count() > 0);
        // Apex vertices should be near Y axis
        for v in &mesh.vertices {
            if approx(v.y, 2.0, 0.01) {
                let r = (v.x * v.x + v.z * v.z).sqrt();
                assert!(approx(r, 0.0, 0.01));
            }
        }
    }

    #[test]
    fn test_lathe_volume_cylinder() {
        // Cylinder r=1, h=2 → volume = πr²h ≈ 6.283
        // Need closed profile (top + bottom edges) for watertight mesh
        let profile = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(1.0, 2.0),
            Vec2::new(0.0, 2.0),
        ];
        let config = LatheConfig::new(profile, 64)
            .with_closed_profile(true);
        let mesh = generate_lathe(&config);
        let vol = lathe_volume(&mesh);
        assert!(approx(vol, PI * 2.0, 0.5));
    }

    #[test]
    fn test_sweep_quarter_turn() {
        let config = LatheConfig::new(cylinder_profile(1.0, 1.0), 8)
            .with_sweep(PI / 2.0);
        let mesh = generate_lathe(&config);
        // All vertices should be in the positive X-Z quadrant (approximately)
        let ring_count = 9; // 8+1 for partial
        for i in 0..ring_count * 2 {
            let v = &mesh.vertices[i];
            assert!(v.x >= -0.01, "x should be non-negative: {}", v.x);
            assert!(v.z >= -0.01, "z should be non-negative: {}", v.z);
        }
    }

    #[test]
    fn test_normals_count_matches_vertices() {
        let config = LatheConfig::new(cylinder_profile(1.0, 2.0), 16);
        let mesh = generate_lathe(&config);
        assert_eq!(mesh.normals.len(), mesh.vertices.len());
        assert_eq!(mesh.uvs.len(), mesh.vertices.len());
    }

    #[test]
    fn test_profile_normal_2d() {
        let profile = vec![Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)];
        let n = profile_normal_2d(&profile, 0, false);
        // Edge goes up (+y), normal should point outward (+x direction)
        assert!(n.x > 0.5);
    }
}
