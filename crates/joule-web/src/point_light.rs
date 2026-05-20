//! Point (omni-directional) light for the lighting engine.
//!
//! Models a light that emits uniformly from a position in all directions.
//! Provides attenuation (inverse-square with configurable falloff),
//! smooth range cutoff (windowing function), cubemap shadow face
//! computation, light volume sphere culling, and clustered-light
//! assignment helpers.

// ── Vector types ───────────────────────────────────────────────

/// 3-component vector (f64).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }

    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }

    pub fn length(self) -> f64 { self.dot(self).sqrt() }

    pub fn length_sq(self) -> f64 { self.dot(self) }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { return Self::ZERO; }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
    }
}

/// Linear RGB color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };

    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }

    pub fn scale(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
}

// ── Attenuation models ─────────────────────────────────────────

/// Attenuation model for the point light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Attenuation {
    /// Physical inverse-square (1 / d^2).
    InverseSquare,
    /// Inverse square with a minimum denominator to avoid singularity.
    InverseSquareClamped { min_dist: f64 },
    /// UE4-style: `1 / (d^2 + 1)` with smooth window.
    Ue4 { falloff_exponent: f64 },
    /// Custom constant/linear/quadratic coefficients.
    Custom { constant: f64, linear: f64, quadratic: f64 },
}

impl Attenuation {
    /// Evaluate raw attenuation at the given distance.
    pub fn evaluate(&self, distance: f64) -> f64 {
        if distance < 0.0 { return 1.0; }
        match self {
            Attenuation::InverseSquare => {
                1.0 / (distance * distance).max(1e-6)
            }
            Attenuation::InverseSquareClamped { min_dist } => {
                let d = distance.max(*min_dist);
                1.0 / (d * d)
            }
            Attenuation::Ue4 { falloff_exponent } => {
                let d2 = distance * distance;
                1.0 / (d2 + 1.0).powf(*falloff_exponent)
            }
            Attenuation::Custom { constant, linear, quadratic } => {
                let denom = constant + linear * distance + quadratic * distance * distance;
                if denom < 1e-12 { return 1.0; }
                1.0 / denom
            }
        }
    }
}

/// Smooth windowing function for range cutoff.
/// Returns a factor in [0, 1] that smoothly goes to 0 at `max_range`.
/// Uses the UE4-style window: `saturate(1 - (d/R)^4)^2`.
fn window_function(distance: f64, max_range: f64) -> f64 {
    if max_range <= 0.0 { return 0.0; }
    let ratio = distance / max_range;
    if ratio >= 1.0 { return 0.0; }
    let r4 = ratio * ratio * ratio * ratio;
    let v = (1.0 - r4).max(0.0);
    v * v
}

// ── Cubemap face ───────────────────────────────────────────────

/// The six faces of a cubemap for point-light shadow maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubemapFace {
    PositiveX,
    NegativeX,
    PositiveY,
    NegativeY,
    PositiveZ,
    NegativeZ,
}

impl CubemapFace {
    pub const ALL: [CubemapFace; 6] = [
        CubemapFace::PositiveX,
        CubemapFace::NegativeX,
        CubemapFace::PositiveY,
        CubemapFace::NegativeY,
        CubemapFace::PositiveZ,
        CubemapFace::NegativeZ,
    ];

    /// Direction vector for this cubemap face.
    pub fn direction(self) -> Vec3 {
        match self {
            Self::PositiveX => Vec3::new(1.0, 0.0, 0.0),
            Self::NegativeX => Vec3::new(-1.0, 0.0, 0.0),
            Self::PositiveY => Vec3::new(0.0, 1.0, 0.0),
            Self::NegativeY => Vec3::new(0.0, -1.0, 0.0),
            Self::PositiveZ => Vec3::new(0.0, 0.0, 1.0),
            Self::NegativeZ => Vec3::new(0.0, 0.0, -1.0),
        }
    }

    /// Up vector for this cubemap face (to build a view matrix).
    pub fn up(self) -> Vec3 {
        match self {
            Self::PositiveY => Vec3::new(0.0, 0.0, -1.0),
            Self::NegativeY => Vec3::new(0.0, 0.0, 1.0),
            _ => Vec3::new(0.0, 1.0, 0.0),
        }
    }
}

// ── Cluster helpers ────────────────────────────────────────────

/// An axis-aligned bounding box used for cluster bins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    /// Test if a sphere (center + radius) intersects this AABB.
    pub fn intersects_sphere(&self, center: Vec3, radius: f64) -> bool {
        let cx = center.x.clamp(self.min.x, self.max.x);
        let cy = center.y.clamp(self.min.y, self.max.y);
        let cz = center.z.clamp(self.min.z, self.max.z);
        let closest = Vec3::new(cx, cy, cz);
        closest.sub(center).length_sq() <= radius * radius
    }
}

/// A 3D cluster grid for tiled/clustered light culling.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterGrid {
    /// Number of bins in each axis.
    pub divisions_x: u32,
    pub divisions_y: u32,
    pub divisions_z: u32,
    /// World-space AABB of the entire grid.
    pub bounds: AABB,
}

impl ClusterGrid {
    pub fn new(divisions_x: u32, divisions_y: u32, divisions_z: u32, bounds: AABB) -> Self {
        Self { divisions_x, divisions_y, divisions_z, bounds }
    }

    /// Total number of clusters.
    pub fn total_clusters(&self) -> u32 {
        self.divisions_x * self.divisions_y * self.divisions_z
    }

    /// Get the AABB for the cluster at index (ix, iy, iz).
    pub fn cluster_bounds(&self, ix: u32, iy: u32, iz: u32) -> AABB {
        let extent = self.bounds.max.sub(self.bounds.min);
        let dx = extent.x / self.divisions_x as f64;
        let dy = extent.y / self.divisions_y as f64;
        let dz = extent.z / self.divisions_z as f64;
        let min = Vec3::new(
            self.bounds.min.x + ix as f64 * dx,
            self.bounds.min.y + iy as f64 * dy,
            self.bounds.min.z + iz as f64 * dz,
        );
        let max = Vec3::new(min.x + dx, min.y + dy, min.z + dz);
        AABB::new(min, max)
    }

    /// Return a list of (ix, iy, iz) cluster indices that a sphere intersects.
    pub fn intersecting_clusters(&self, center: Vec3, radius: f64) -> Vec<(u32, u32, u32)> {
        let mut result = Vec::new();
        for iz in 0..self.divisions_z {
            for iy in 0..self.divisions_y {
                for ix in 0..self.divisions_x {
                    let aabb = self.cluster_bounds(ix, iy, iz);
                    if aabb.intersects_sphere(center, radius) {
                        result.push((ix, iy, iz));
                    }
                }
            }
        }
        result
    }
}

// ── PointLight ─────────────────────────────────────────────────

/// A point (omni-directional) light source.
#[derive(Debug, Clone, PartialEq)]
pub struct PointLight {
    /// World-space position.
    pub position: Vec3,
    /// Light color.
    pub color: Color,
    /// Intensity (luminous power).
    pub intensity: f64,
    /// Maximum range (beyond which contribution is zero).
    pub range: f64,
    /// Attenuation model.
    pub attenuation: Attenuation,
    /// Whether this light casts shadows (via cubemap).
    pub casts_shadows: bool,
    /// Shadow cubemap resolution per face.
    pub shadow_resolution: u32,
}

impl PointLight {
    pub fn new(position: Vec3, color: Color, intensity: f64, range: f64) -> Self {
        Self {
            position,
            color,
            intensity,
            range,
            attenuation: Attenuation::InverseSquare,
            casts_shadows: false,
            shadow_resolution: 512,
        }
    }

    pub fn with_attenuation(mut self, att: Attenuation) -> Self {
        self.attenuation = att;
        self
    }

    pub fn with_shadows(mut self, resolution: u32) -> Self {
        self.casts_shadows = true;
        self.shadow_resolution = resolution;
        self
    }

    /// Compute the attenuated intensity at a given world position.
    /// Includes the smooth windowing function for range cutoff.
    pub fn intensity_at(&self, world_pos: Vec3) -> f64 {
        let dist = world_pos.sub(self.position).length();
        if dist >= self.range { return 0.0; }
        let raw = self.attenuation.evaluate(dist);
        let window = window_function(dist, self.range);
        self.intensity * raw * window
    }

    /// Compute the final color contribution at a world position
    /// (does NOT include surface BRDF — that is the caller's job).
    pub fn color_at(&self, world_pos: Vec3) -> Color {
        let i = self.intensity_at(world_pos);
        self.color.scale(i)
    }

    /// Lambertian diffuse contribution at a surface point with given normal.
    pub fn diffuse_at(&self, surface_pos: Vec3, surface_normal: Vec3) -> Color {
        let to_light = self.position.sub(surface_pos);
        let dist = to_light.length();
        if dist >= self.range || dist < 1e-12 { return Color::BLACK; }
        let dir = to_light.scale(1.0 / dist);
        let n_dot_l = surface_normal.normalized().dot(dir).max(0.0);
        let atten = self.attenuation.evaluate(dist) * window_function(dist, self.range);
        self.color.scale(self.intensity * atten * n_dot_l)
    }

    /// Light volume bounding sphere radius (for culling).
    pub fn bounding_radius(&self) -> f64 { self.range }

    /// Test whether a point is inside the light volume.
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.sub(self.position).length_sq() <= self.range * self.range
    }

    /// Determine which cluster bins this light touches.
    pub fn cluster_assignment(&self, grid: &ClusterGrid) -> Vec<(u32, u32, u32)> {
        grid.intersecting_clusters(self.position, self.range)
    }

    /// Shadow cubemap face directions (returns all 6).
    pub fn shadow_faces(&self) -> [(CubemapFace, Vec3, Vec3); 6] {
        CubemapFace::ALL.map(|f| (f, self.position.add(f.direction()), f.up()))
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn attenuation_inverse_square_at_1() {
        let a = Attenuation::InverseSquare;
        assert!(approx(a.evaluate(1.0), 1.0));
    }

    #[test]
    fn attenuation_inverse_square_at_2() {
        let a = Attenuation::InverseSquare;
        assert!(approx(a.evaluate(2.0), 0.25));
    }

    #[test]
    fn attenuation_clamped_min_dist() {
        let a = Attenuation::InverseSquareClamped { min_dist: 0.5 };
        let at_zero = a.evaluate(0.0);
        let at_half = a.evaluate(0.5);
        assert!(approx(at_zero, at_half));
    }

    #[test]
    fn attenuation_custom() {
        let a = Attenuation::Custom { constant: 1.0, linear: 0.0, quadratic: 1.0 };
        assert!(approx(a.evaluate(1.0), 0.5));
    }

    #[test]
    fn window_at_zero() {
        assert!(approx(window_function(0.0, 10.0), 1.0));
    }

    #[test]
    fn window_at_range() {
        assert!(approx(window_function(10.0, 10.0), 0.0));
    }

    #[test]
    fn window_beyond_range() {
        assert!(approx(window_function(15.0, 10.0), 0.0));
    }

    #[test]
    fn window_smooth_midpoint() {
        let w = window_function(5.0, 10.0);
        assert!(w > 0.0 && w < 1.0);
    }

    #[test]
    fn point_light_intensity_at_origin() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 10.0, 50.0);
        let i = l.intensity_at(Vec3::new(1.0, 0.0, 0.0));
        assert!(i > 0.0);
        assert!(i < 10.0);
    }

    #[test]
    fn point_light_zero_at_range() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 10.0, 5.0);
        let i = l.intensity_at(Vec3::new(5.0, 0.0, 0.0));
        assert!(approx(i, 0.0));
    }

    #[test]
    fn point_light_beyond_range_zero() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 10.0, 5.0);
        let i = l.intensity_at(Vec3::new(10.0, 0.0, 0.0));
        assert!(approx(i, 0.0));
    }

    #[test]
    fn color_at_scales_correctly() {
        let l = PointLight::new(Vec3::ZERO, Color::new(1.0, 0.5, 0.0), 4.0, 100.0);
        let c = l.color_at(Vec3::new(1.0, 0.0, 0.0));
        // Red should be roughly 2× green.
        assert!((c.r - 2.0 * c.g).abs() < 0.01);
    }

    #[test]
    fn diffuse_facing_light() {
        let l = PointLight::new(Vec3::new(0.0, 10.0, 0.0), Color::WHITE, 100.0, 50.0);
        let c = l.diffuse_at(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(c.r > 0.0);
    }

    #[test]
    fn diffuse_facing_away() {
        let l = PointLight::new(Vec3::new(0.0, 10.0, 0.0), Color::WHITE, 100.0, 50.0);
        let c = l.diffuse_at(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0));
        assert!(approx(c.r, 0.0));
    }

    #[test]
    fn bounding_radius() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 1.0, 25.0);
        assert!(approx(l.bounding_radius(), 25.0));
    }

    #[test]
    fn contains_point_inside() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 1.0, 10.0);
        assert!(l.contains_point(Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn contains_point_outside() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 1.0, 10.0);
        assert!(!l.contains_point(Vec3::new(11.0, 0.0, 0.0)));
    }

    #[test]
    fn shadow_faces_count() {
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 1.0, 10.0)
            .with_shadows(1024);
        assert!(l.casts_shadows);
        assert_eq!(l.shadow_faces().len(), 6);
    }

    #[test]
    fn cluster_assignment_basic() {
        let grid = ClusterGrid::new(
            2, 2, 2,
            AABB::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0)),
        );
        let l = PointLight::new(Vec3::ZERO, Color::WHITE, 1.0, 5.0);
        let clusters = l.cluster_assignment(&grid);
        // A light at center with radius 5 in a ±10 grid should touch all 8 clusters.
        assert_eq!(clusters.len(), 8);
    }

    #[test]
    fn cluster_assignment_corner() {
        let grid = ClusterGrid::new(
            4, 4, 4,
            AABB::new(Vec3::ZERO, Vec3::new(40.0, 40.0, 40.0)),
        );
        let l = PointLight::new(Vec3::new(5.0, 5.0, 5.0), Color::WHITE, 1.0, 1.0);
        let clusters = l.cluster_assignment(&grid);
        // Small light in one corner cell.
        assert!(clusters.len() >= 1);
        assert!(clusters.contains(&(0, 0, 0)));
    }

    #[test]
    fn aabb_sphere_intersect_inside() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        assert!(aabb.intersects_sphere(Vec3::new(5.0, 5.0, 5.0), 1.0));
    }

    #[test]
    fn aabb_sphere_no_intersect() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert!(!aabb.intersects_sphere(Vec3::new(10.0, 10.0, 10.0), 1.0));
    }

    #[test]
    fn cubemap_face_directions_orthogonal() {
        for face in CubemapFace::ALL {
            let d = face.direction();
            let u = face.up();
            assert!(d.dot(u).abs() < EPS, "direction and up must be orthogonal");
        }
    }

    #[test]
    fn attenuation_ue4() {
        let a = Attenuation::Ue4 { falloff_exponent: 1.0 };
        let v = a.evaluate(0.0);
        assert!(approx(v, 1.0));
        let v2 = a.evaluate(1.0);
        assert!(approx(v2, 0.5));
    }
}
