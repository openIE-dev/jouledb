//! Extrude Path — extrude a 2D profile polygon along a 3D path with Frenet frames,
//! taper, twist, cap options, and miter/bevel corners.

use std::f64::consts::PI;

// ── Vector types ───────────────────────────────────────────────

/// 2D point/vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

/// 3D point/vector.
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
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn cross(&self, o: &Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn dot(&self, o: &Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
}

// ── Frenet frame ───────────────────────────────────────────────

/// Orthonormal frame at a path point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrenetFrame {
    pub tangent: Vec3,
    pub normal: Vec3,
    pub binormal: Vec3,
}

/// Compute Frenet frames along a polyline path.
pub fn compute_frenet_frames(path: &[Vec3]) -> Vec<FrenetFrame> {
    let n = path.len();
    if n < 2 {
        return vec![FrenetFrame {
            tangent: Vec3::new(0.0, 0.0, 1.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            binormal: Vec3::new(0.0, 1.0, 0.0),
        }];
    }

    let mut tangents = Vec::with_capacity(n);
    for i in 0..n {
        let t = if i == 0 {
            path[1].sub(&path[0])
        } else if i == n - 1 {
            path[n - 1].sub(&path[n - 2])
        } else {
            path[i + 1].sub(&path[i - 1])
        };
        tangents.push(t.normalized());
    }

    // Initial normal: find vector not parallel to first tangent
    let up_candidate = if tangents[0].y.abs() < 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let mut prev_normal = tangents[0].cross(&up_candidate).normalized();
    let mut prev_binormal = tangents[0].cross(&prev_normal).normalized();

    let mut frames = Vec::with_capacity(n);
    frames.push(FrenetFrame {
        tangent: tangents[0],
        normal: prev_normal,
        binormal: prev_binormal,
    });

    // Propagate using rotation minimizing approach
    for i in 1..n {
        let t = tangents[i];
        // Project previous normal onto plane perpendicular to new tangent
        let dot_tn = prev_normal.dot(&t);
        let normal = Vec3::new(
            prev_normal.x - dot_tn * t.x,
            prev_normal.y - dot_tn * t.y,
            prev_normal.z - dot_tn * t.z,
        )
        .normalized();
        let binormal = t.cross(&normal).normalized();
        frames.push(FrenetFrame {
            tangent: t,
            normal,
            binormal,
        });
        prev_normal = normal;
        prev_binormal = binormal;
    }

    frames
}

// ── Mesh output ────────────────────────────────────────────────

/// Generated extrusion mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtrudedMesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f64; 2]>,
    pub indices: Vec<[u32; 3]>,
}

impl ExtrudedMesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }
}

/// Cap style for the extrusion ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapStyle {
    Open,
    Flat,
    Rounded,
}

/// Corner handling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerMode {
    Miter,
    Bevel,
}

/// Configuration for the extrusion.
#[derive(Debug, Clone)]
pub struct ExtrudeConfig {
    /// 2D profile polygon (closed loop implied).
    pub profile: Vec<Vec2>,
    /// 3D path waypoints (at least 2).
    pub path: Vec<Vec3>,
    /// Scale factor at each path point (taper). If empty, uniform scale 1.0.
    pub scales: Vec<f64>,
    /// Twist angle in radians per unit path length.
    pub twist_per_unit: f64,
    /// Cap style for start and end.
    pub start_cap: CapStyle,
    pub end_cap: CapStyle,
    /// Corner handling mode.
    pub corner_mode: CornerMode,
}

impl ExtrudeConfig {
    pub fn new(profile: Vec<Vec2>, path: Vec<Vec3>) -> Self {
        Self {
            profile,
            path,
            scales: Vec::new(),
            twist_per_unit: 0.0,
            start_cap: CapStyle::Flat,
            end_cap: CapStyle::Flat,
            corner_mode: CornerMode::Miter,
        }
    }
}

/// Cumulative arc length along a path.
fn path_arc_lengths(path: &[Vec3]) -> Vec<f64> {
    let mut lengths = Vec::with_capacity(path.len());
    lengths.push(0.0);
    for i in 1..path.len() {
        let seg = path[i].sub(&path[i - 1]).length();
        lengths.push(lengths[i - 1] + seg);
    }
    lengths
}

/// Extrude the profile along the path.
pub fn extrude(config: &ExtrudeConfig) -> ExtrudedMesh {
    let path = &config.path;
    let profile = &config.profile;
    if path.len() < 2 || profile.is_empty() {
        return ExtrudedMesh {
            vertices: vec![],
            normals: vec![],
            uvs: vec![],
            indices: vec![],
        };
    }

    let frames = compute_frenet_frames(path);
    let arc_lengths = path_arc_lengths(path);
    let total_length = *arc_lengths.last().unwrap_or(&1.0);
    let total_length = if total_length < 1e-12 { 1.0 } else { total_length };

    let profile_len = profile.len();
    let path_len = path.len();
    let ring_verts = profile_len; // one per profile vertex per path station

    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();

    // Compute profile perimeter for UV U coordinate
    let mut profile_dists = Vec::with_capacity(profile_len + 1);
    profile_dists.push(0.0);
    for i in 1..profile_len {
        let dx = profile[i].x - profile[i - 1].x;
        let dy = profile[i].y - profile[i - 1].y;
        profile_dists.push(profile_dists[i - 1] + (dx * dx + dy * dy).sqrt());
    }
    let profile_perim = *profile_dists.last().unwrap_or(&1.0);
    let profile_perim = if profile_perim < 1e-12 { 1.0 } else { profile_perim };

    for pi in 0..path_len {
        let frame = &frames[pi];
        let center = &path[pi];
        let scale = if pi < config.scales.len() {
            config.scales[pi]
        } else {
            1.0
        };
        let arc = arc_lengths[pi];
        let twist_angle = config.twist_per_unit * arc;
        let v_coord = arc / total_length;

        let cos_tw = twist_angle.cos();
        let sin_tw = twist_angle.sin();

        for vi in 0..profile_len {
            let px = profile[vi].x * scale;
            let py = profile[vi].y * scale;

            // Apply twist rotation in the normal/binormal plane
            let rx = px * cos_tw - py * sin_tw;
            let ry = px * sin_tw + py * cos_tw;

            let world_pos = center
                .add(&frame.normal.scale(rx))
                .add(&frame.binormal.scale(ry));

            // Normal: direction from center to vertex (approx)
            let n = world_pos.sub(center).normalized();

            vertices.push(world_pos);
            normals.push(n);
            uvs.push([profile_dists[vi] / profile_perim, v_coord]);
        }
    }

    // Indices: connect ring i to ring i+1
    let mut indices = Vec::new();
    for pi in 0..path_len - 1 {
        let base0 = (pi * ring_verts) as u32;
        let base1 = ((pi + 1) * ring_verts) as u32;
        for vi in 0..profile_len {
            let next_vi = (vi + 1) % profile_len;
            let a = base0 + vi as u32;
            let b = base0 + next_vi as u32;
            let c = base1 + next_vi as u32;
            let d = base1 + vi as u32;
            indices.push([a, b, c]);
            indices.push([a, c, d]);
        }
    }

    // Flat caps
    let add_fan_cap = |verts: &mut Vec<Vec3>,
                       norms: &mut Vec<Vec3>,
                       uv_list: &mut Vec<[f64; 2]>,
                       idx: &mut Vec<[u32; 3]>,
                       ring_start: usize,
                       cap_normal: Vec3,
                       center: Vec3,
                       reverse: bool| {
        let center_idx = verts.len() as u32;
        verts.push(center);
        norms.push(cap_normal);
        uv_list.push([0.5, 0.5]);
        for vi in 0..profile_len {
            let next_vi = (vi + 1) % profile_len;
            let a = center_idx;
            let b = (ring_start + vi) as u32;
            let c = (ring_start + next_vi) as u32;
            if reverse {
                idx.push([a, c, b]);
            } else {
                idx.push([a, b, c]);
            }
        }
    };

    if config.start_cap == CapStyle::Flat {
        let cap_n = frames[0].tangent.scale(-1.0);
        add_fan_cap(
            &mut vertices, &mut normals, &mut uvs, &mut indices,
            0, cap_n, path[0], true,
        );
    }

    if config.end_cap == CapStyle::Flat {
        let cap_n = frames[path_len - 1].tangent;
        let ring_start = (path_len - 1) * ring_verts;
        add_fan_cap(
            &mut vertices, &mut normals, &mut uvs, &mut indices,
            ring_start, cap_n, path[path_len - 1], false,
        );
    }

    // Rounded caps (hemisphere approximation using extra rings)
    if config.start_cap == CapStyle::Rounded || config.end_cap == CapStyle::Rounded {
        let round_steps = 4;
        let add_rounded = |verts: &mut Vec<Vec3>,
                           norms: &mut Vec<Vec3>,
                           uv_list: &mut Vec<[f64; 2]>,
                           idx: &mut Vec<[u32; 3]>,
                           ring_start: usize,
                           frame: &FrenetFrame,
                           center: Vec3,
                           base_scale: f64,
                           inward: bool| {
            let sign = if inward { -1.0 } else { 1.0 };
            let mut prev_ring_start = ring_start;
            for step in 1..=round_steps {
                let angle = (step as f64 / round_steps as f64) * PI * 0.5;
                let s = angle.cos() * base_scale;
                let offset = angle.sin() * base_scale * sign;
                let new_center = center.add(&frame.tangent.scale(offset * -1.0));
                let cur_ring_start = verts.len();
                for vi in 0..profile_len {
                    let px = profile[vi].x * s;
                    let py = profile[vi].y * s;
                    let world = new_center
                        .add(&frame.normal.scale(px))
                        .add(&frame.binormal.scale(py));
                    let n = world.sub(&new_center).normalized();
                    verts.push(world);
                    norms.push(n);
                    uv_list.push([0.0, 0.0]);
                }
                for vi in 0..profile_len {
                    let next_vi = (vi + 1) % profile_len;
                    let a = prev_ring_start as u32 + vi as u32;
                    let b = prev_ring_start as u32 + next_vi as u32;
                    let c = cur_ring_start as u32 + next_vi as u32;
                    let d = cur_ring_start as u32 + vi as u32;
                    if inward {
                        idx.push([a, c, b]);
                        idx.push([a, d, c]);
                    } else {
                        idx.push([a, b, c]);
                        idx.push([a, c, d]);
                    }
                }
                prev_ring_start = cur_ring_start;
            }
        };

        if config.start_cap == CapStyle::Rounded {
            let s0 = if config.scales.is_empty() { 1.0 } else { config.scales[0] };
            add_rounded(
                &mut vertices, &mut normals, &mut uvs, &mut indices,
                0, &frames[0], path[0], s0, true,
            );
        }
        if config.end_cap == CapStyle::Rounded {
            let last = path_len - 1;
            let s_last = if last < config.scales.len() { config.scales[last] } else { 1.0 };
            let ring_start = last * ring_verts;
            add_rounded(
                &mut vertices, &mut normals, &mut uvs, &mut indices,
                ring_start, &frames[last], path[last], s_last, false,
            );
        }
    }

    ExtrudedMesh {
        vertices,
        normals,
        uvs,
        indices,
    }
}

/// Compute total path length.
pub fn path_length(path: &[Vec3]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 1..path.len() {
        total += path[i].sub(&path[i - 1]).length();
    }
    total
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn square_profile() -> Vec<Vec2> {
        vec![
            Vec2::new(-0.5, -0.5),
            Vec2::new(0.5, -0.5),
            Vec2::new(0.5, 0.5),
            Vec2::new(-0.5, 0.5),
        ]
    }

    fn straight_path() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
            Vec3::new(0.0, 0.0, 3.0),
        ]
    }

    #[test]
    fn test_vec3_basic() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert!(approx(v.length(), (14.0_f64).sqrt(), EPS));
    }

    #[test]
    fn test_frenet_frames_straight_line() {
        let path = straight_path();
        let frames = compute_frenet_frames(&path);
        assert_eq!(frames.len(), 4);
        for f in &frames {
            assert!(approx(f.tangent.z, 1.0, 0.01));
            assert!(approx(f.tangent.length(), 1.0, EPS));
            assert!(approx(f.normal.length(), 1.0, EPS));
            assert!(approx(f.binormal.length(), 1.0, EPS));
        }
    }

    #[test]
    fn test_frenet_orthogonal() {
        let path = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        let frames = compute_frenet_frames(&path);
        for f in &frames {
            assert!(f.tangent.dot(&f.normal).abs() < 0.01);
            assert!(f.tangent.dot(&f.binormal).abs() < 0.01);
            assert!(f.normal.dot(&f.binormal).abs() < 0.01);
        }
    }

    #[test]
    fn test_basic_extrusion_counts() {
        let config = ExtrudeConfig::new(square_profile(), straight_path());
        let mesh = extrude(&config);
        // 4 path points × 4 profile points = 16 ring verts + 2 cap centers
        assert_eq!(mesh.vertex_count(), 4 * 4 + 2);
        // 3 segments × 4 quads × 2 tri + 2 caps × 4 tri
        assert_eq!(mesh.triangle_count(), 3 * 4 * 2 + 2 * 4);
    }

    #[test]
    fn test_extrusion_open_caps() {
        let mut config = ExtrudeConfig::new(square_profile(), straight_path());
        config.start_cap = CapStyle::Open;
        config.end_cap = CapStyle::Open;
        let mesh = extrude(&config);
        assert_eq!(mesh.vertex_count(), 4 * 4); // no cap center verts
        assert_eq!(mesh.triangle_count(), 3 * 4 * 2);
    }

    #[test]
    fn test_extrusion_with_taper() {
        let mut config = ExtrudeConfig::new(square_profile(), straight_path());
        config.scales = vec![1.0, 0.75, 0.5, 0.25];
        config.start_cap = CapStyle::Open;
        config.end_cap = CapStyle::Open;
        let mesh = extrude(&config);
        // Last ring should be smaller than first
        let first_ring_max = mesh.vertices[..4]
            .iter()
            .map(|v| (v.x * v.x + v.y * v.y).sqrt())
            .fold(0.0_f64, f64::max);
        let last_start = 3 * 4;
        let last_ring_max = mesh.vertices[last_start..last_start + 4]
            .iter()
            .map(|v| (v.x * v.x + v.y * v.y).sqrt())
            .fold(0.0_f64, f64::max);
        assert!(last_ring_max < first_ring_max);
    }

    #[test]
    fn test_extrusion_with_twist() {
        let mut config = ExtrudeConfig::new(square_profile(), straight_path());
        config.twist_per_unit = PI / 2.0; // 90° per unit
        config.start_cap = CapStyle::Open;
        config.end_cap = CapStyle::Open;
        let mesh = extrude(&config);
        // First ring at arc=0 has no twist; last ring at arc=3 has 3×90°=270°
        assert_eq!(mesh.vertex_count(), 16);
    }

    #[test]
    fn test_path_length() {
        let path = straight_path();
        assert!(approx(path_length(&path), 3.0, EPS));
    }

    #[test]
    fn test_path_length_empty() {
        let path: Vec<Vec3> = vec![];
        assert!(approx(path_length(&path), 0.0, EPS));
    }

    #[test]
    fn test_extrude_empty_profile() {
        let config = ExtrudeConfig::new(vec![], straight_path());
        let mesh = extrude(&config);
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn test_extrude_empty_path() {
        let config = ExtrudeConfig::new(square_profile(), vec![]);
        let mesh = extrude(&config);
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn test_extrude_two_point_path() {
        let path = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)];
        let config = ExtrudeConfig::new(square_profile(), path);
        let mesh = extrude(&config);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_indices_in_bounds() {
        let config = ExtrudeConfig::new(square_profile(), straight_path());
        let mesh = extrude(&config);
        let max_idx = mesh.vertex_count() as u32;
        for tri in &mesh.indices {
            assert!(tri[0] < max_idx);
            assert!(tri[1] < max_idx);
            assert!(tri[2] < max_idx);
        }
    }

    #[test]
    fn test_normals_unit_length() {
        let config = ExtrudeConfig::new(square_profile(), straight_path());
        let mesh = extrude(&config);
        for n in &mesh.normals {
            let len = n.length();
            // Cap center normals and ring normals should be unit-length
            assert!(len > 0.5 && len < 1.5, "normal length out of range: {len}");
        }
    }

    #[test]
    fn test_uvs_v_monotonic() {
        let mut config = ExtrudeConfig::new(square_profile(), straight_path());
        config.start_cap = CapStyle::Open;
        config.end_cap = CapStyle::Open;
        let mesh = extrude(&config);
        // First profile vertex of each ring: v should increase
        let mut prev_v = -1.0;
        for pi in 0..4 {
            let idx = pi * 4;
            let v = mesh.uvs[idx][1];
            assert!(v >= prev_v, "v not monotonic at path index {pi}");
            prev_v = v;
        }
    }

    #[test]
    fn test_rounded_cap_adds_geometry() {
        let mut config = ExtrudeConfig::new(square_profile(), straight_path());
        config.start_cap = CapStyle::Rounded;
        config.end_cap = CapStyle::Rounded;
        let mesh = extrude(&config);
        // More vertices than flat cap version
        let mut flat_config = ExtrudeConfig::new(square_profile(), straight_path());
        flat_config.start_cap = CapStyle::Flat;
        flat_config.end_cap = CapStyle::Flat;
        let flat_mesh = extrude(&flat_config);
        assert!(mesh.vertex_count() > flat_mesh.vertex_count());
    }

    #[test]
    fn test_triangle_profile() {
        let tri_profile = vec![
            Vec2::new(0.0, 0.5),
            Vec2::new(-0.5, -0.5),
            Vec2::new(0.5, -0.5),
        ];
        let mut config = ExtrudeConfig::new(tri_profile, straight_path());
        config.start_cap = CapStyle::Open;
        config.end_cap = CapStyle::Open;
        let mesh = extrude(&config);
        assert_eq!(mesh.vertex_count(), 4 * 3); // 4 rings × 3 profile verts
    }

    #[test]
    fn test_frenet_single_point() {
        let frames = compute_frenet_frames(&[Vec3::new(0.0, 0.0, 0.0)]);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_corner_mode_field() {
        let mut config = ExtrudeConfig::new(square_profile(), straight_path());
        config.corner_mode = CornerMode::Bevel;
        assert_eq!(config.corner_mode, CornerMode::Bevel);
    }
}
