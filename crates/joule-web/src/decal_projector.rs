// decal_projector.rs — Decal projection system
// Part of joule-web: Particles & VFX cluster

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-10 { Self::ZERO } else { Self { x: self.x / l, y: self.y / l, z: self.z / l } }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
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
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

/// 2D UV coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self { Self { x, y } }
}

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
}

/// Simple orientation represented as forward+up vectors (orthonormal).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orientation {
    pub forward: Vec3,
    pub up: Vec3,
    pub right: Vec3,
}

impl Orientation {
    /// Build from a forward direction (Z), deriving up and right.
    pub fn from_forward(forward: Vec3) -> Self {
        let f = forward.normalized();
        let world_up = if f.dot(Vec3::new(0.0, 1.0, 0.0)).abs() > 0.99 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let right = f.cross(world_up).normalized();
        let up = right.cross(f).normalized();
        Self { forward: f, up, right }
    }

    pub fn identity() -> Self {
        Self {
            forward: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(1.0, 0.0, 0.0),
        }
    }
}

/// Transform a world-space point into projector local space.
fn world_to_local(point: Vec3, position: Vec3, orientation: &Orientation) -> Vec3 {
    let rel = point - position;
    Vec3::new(
        rel.dot(orientation.right),
        rel.dot(orientation.up),
        rel.dot(orientation.forward),
    )
}

/// Rectangle within a decal atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasRect {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

impl AtlasRect {
    pub fn full() -> Self {
        Self { u_min: 0.0, v_min: 0.0, u_max: 1.0, v_max: 1.0 }
    }

    /// Map normalized [0,1] UV to atlas UV.
    pub fn map_uv(&self, u: f32, v: f32) -> Vec2 {
        Vec2::new(
            self.u_min + u * (self.u_max - self.u_min),
            self.v_min + v * (self.v_max - self.v_min),
        )
    }
}

/// Decal configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct DecalConfig {
    /// Half-extents of the projection box.
    pub half_extents: Vec3,
    /// Angle fade: surfaces beyond this angle from forward are faded.
    pub angle_fade_start: f32,
    /// Angle fade: surfaces beyond this angle get zero alpha.
    pub angle_fade_end: f32,
    /// Distance fade from edges of the projection box.
    pub edge_fade_width: f32,
    /// Atlas rect for this decal's texture.
    pub atlas_rect: AtlasRect,
    /// Render sort order (lower draws first).
    pub sort_order: i32,
    /// Lifetime in seconds (0 = infinite).
    pub lifetime: f32,
    /// Fade-out duration at end of lifetime.
    pub fade_out_duration: f32,
}

impl Default for DecalConfig {
    fn default() -> Self {
        Self {
            half_extents: Vec3::ONE,
            angle_fade_start: 0.8,
            angle_fade_end: 0.2,
            edge_fade_width: 0.1,
            atlas_rect: AtlasRect::full(),
            sort_order: 0,
            lifetime: 0.0,
            fade_out_duration: 1.0,
        }
    }
}

/// Result of projecting a point through the decal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecalProjection {
    pub uv: Vec2,
    pub alpha: f32,
    pub inside: bool,
}

/// A decal instance in the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct Decal {
    pub position: Vec3,
    pub orientation: Orientation,
    pub config: DecalConfig,
    pub age: f32,
    pub active: bool,
    id: u64,
}

impl Decal {
    pub fn new(id: u64, position: Vec3, orientation: Orientation, config: DecalConfig) -> Self {
        Self { position, orientation, config, age: 0.0, active: true, id }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Test if a world-space point is inside the projection box.
    pub fn contains_point(&self, point: Vec3) -> bool {
        let local = world_to_local(point, self.position, &self.orientation);
        local.x.abs() <= self.config.half_extents.x
            && local.y.abs() <= self.config.half_extents.y
            && local.z.abs() <= self.config.half_extents.z
    }

    /// Project a world-space point (with surface normal) through the decal.
    pub fn project(&self, point: Vec3, surface_normal: Vec3) -> DecalProjection {
        let local = world_to_local(point, self.position, &self.orientation);
        let he = self.config.half_extents;

        // Check if inside box
        if local.x.abs() > he.x || local.y.abs() > he.y || local.z.abs() > he.z {
            return DecalProjection {
                uv: Vec2::new(0.0, 0.0),
                alpha: 0.0,
                inside: false,
            };
        }

        // UV from local XY mapped to [0, 1]
        let u_raw = (local.x / he.x + 1.0) * 0.5;
        let v_raw = (local.y / he.y + 1.0) * 0.5;
        let uv = self.config.atlas_rect.map_uv(u_raw, v_raw);

        // Angle-based fade
        let n_dot_f = surface_normal.dot(self.orientation.forward).abs();
        let angle_alpha = if n_dot_f >= self.config.angle_fade_start {
            1.0
        } else if n_dot_f <= self.config.angle_fade_end {
            0.0
        } else {
            (n_dot_f - self.config.angle_fade_end)
                / (self.config.angle_fade_start - self.config.angle_fade_end)
        };

        // Edge fade
        let edge_w = self.config.edge_fade_width.max(1e-6);
        let fade_x = ((he.x - local.x.abs()) / (he.x * edge_w)).clamp(0.0, 1.0);
        let fade_y = ((he.y - local.y.abs()) / (he.y * edge_w)).clamp(0.0, 1.0);
        let fade_z = ((he.z - local.z.abs()) / (he.z * edge_w)).clamp(0.0, 1.0);
        let edge_alpha = fade_x * fade_y * fade_z;

        // Lifetime fade
        let life_alpha = if self.config.lifetime <= 0.0 {
            1.0
        } else if self.age >= self.config.lifetime {
            0.0
        } else {
            let remaining = self.config.lifetime - self.age;
            if remaining < self.config.fade_out_duration && self.config.fade_out_duration > 0.0 {
                remaining / self.config.fade_out_duration
            } else {
                1.0
            }
        };

        DecalProjection {
            uv,
            alpha: angle_alpha * edge_alpha * life_alpha,
            inside: true,
        }
    }

    /// Advance the decal's age.
    pub fn update(&mut self, dt: f32) {
        self.age += dt;
        if self.config.lifetime > 0.0 && self.age >= self.config.lifetime {
            self.active = false;
        }
    }

    pub fn is_expired(&self) -> bool {
        self.config.lifetime > 0.0 && self.age >= self.config.lifetime
    }
}

/// Decal projector system managing multiple decals.
pub struct DecalProjectorSystem {
    decals: Vec<Decal>,
    next_id: u64,
}

impl DecalProjectorSystem {
    pub fn new() -> Self {
        Self { decals: Vec::new(), next_id: 0 }
    }

    /// Add a new decal. Returns its ID.
    pub fn add_decal(
        &mut self,
        position: Vec3,
        orientation: Orientation,
        config: DecalConfig,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.decals.push(Decal::new(id, position, orientation, config));
        id
    }

    /// Remove a decal by ID.
    pub fn remove_decal(&mut self, id: u64) -> bool {
        let before = self.decals.len();
        self.decals.retain(|d| d.id() != id);
        self.decals.len() < before
    }

    /// Update all decals, removing expired ones.
    pub fn update(&mut self, dt: f32) {
        for d in &mut self.decals {
            d.update(dt);
        }
        self.decals.retain(|d| d.active);
    }

    /// Get sorted decals for rendering.
    pub fn sorted_decals(&self) -> Vec<&Decal> {
        let mut sorted: Vec<&Decal> = self.decals.iter().collect();
        sorted.sort_by_key(|d| d.config.sort_order);
        sorted
    }

    pub fn decal_count(&self) -> usize {
        self.decals.len()
    }

    /// Project a point against all decals, returning projections.
    pub fn project_point(&self, point: Vec3, normal: Vec3) -> Vec<(u64, DecalProjection)> {
        let mut results = Vec::new();
        for d in &self.decals {
            let proj = d.project(point, normal);
            if proj.inside && proj.alpha > 1e-6 {
                results.push((d.id(), proj));
            }
        }
        results
    }

    pub fn get_decal(&self, id: u64) -> Option<&Decal> {
        self.decals.iter().find(|d| d.id() == id)
    }

    pub fn clear(&mut self) {
        self.decals.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_decal(pos: Vec3) -> Decal {
        Decal::new(0, pos, Orientation::identity(), DecalConfig::default())
    }

    #[test]
    fn test_orientation_from_forward() {
        let o = Orientation::from_forward(Vec3::new(0.0, 0.0, -1.0));
        assert!((o.forward.z - (-1.0)).abs() < 1e-5);
        assert!(o.right.length() > 0.9);
        assert!(o.up.length() > 0.9);
    }

    #[test]
    fn test_orientation_identity() {
        let o = Orientation::identity();
        assert!((o.forward.z - (-1.0)).abs() < 1e-5);
        assert!((o.up.y - 1.0).abs() < 1e-5);
        assert!((o.right.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_contains_point_inside() {
        let d = default_decal(Vec3::ZERO);
        assert!(d.contains_point(Vec3::new(0.5, 0.5, -0.5)));
    }

    #[test]
    fn test_contains_point_outside() {
        let d = default_decal(Vec3::ZERO);
        assert!(!d.contains_point(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_project_inside() {
        let d = default_decal(Vec3::ZERO);
        let proj = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        assert!(proj.inside);
        assert!(proj.alpha > 0.0);
        // Center point should map near UV (0.5, 0.5)
        assert!((proj.uv.x - 0.5).abs() < 0.1);
        assert!((proj.uv.y - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_project_outside() {
        let d = default_decal(Vec3::ZERO);
        let proj = d.project(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(!proj.inside);
        assert!((proj.alpha).abs() < 1e-6);
    }

    #[test]
    fn test_angle_fade() {
        let cfg = DecalConfig {
            angle_fade_start: 0.8,
            angle_fade_end: 0.2,
            ..Default::default()
        };
        let d = Decal::new(0, Vec3::ZERO, Orientation::identity(), cfg);
        // Surface facing directly toward projector: normal = (0,0,-1)
        let proj_head_on = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        assert!(proj_head_on.alpha > 0.5);
        // Surface almost perpendicular: normal ~ (1,0,0)
        let proj_glancing = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(1.0, 0.0, 0.0));
        // dot with forward (0,0,-1) is ~0 which is < angle_fade_end
        assert!(proj_glancing.alpha < 0.01);
    }

    #[test]
    fn test_edge_fade() {
        let cfg = DecalConfig {
            edge_fade_width: 0.5,
            half_extents: Vec3::ONE,
            ..Default::default()
        };
        let d = Decal::new(0, Vec3::ZERO, Orientation::identity(), cfg);
        // Point at the center
        let center = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        // Point near the edge
        let edge = d.project(Vec3::new(0.95, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        // Edge should have lower alpha due to edge fade
        assert!(edge.alpha < center.alpha || (edge.alpha - center.alpha).abs() < 0.2);
    }

    #[test]
    fn test_lifetime_fade() {
        let cfg = DecalConfig {
            lifetime: 5.0,
            fade_out_duration: 2.0,
            ..Default::default()
        };
        let mut d = Decal::new(0, Vec3::ZERO, Orientation::identity(), cfg);
        // At age 0, fully visible
        let p1 = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        assert!(p1.alpha > 0.5);
        // At age 4, fading (1s remaining with 2s fade = 0.5)
        d.age = 4.0;
        let p2 = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        assert!(p2.alpha < p1.alpha);
        // At age 5, fully faded
        d.age = 5.0;
        let p3 = d.project(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        assert!((p3.alpha).abs() < 1e-5);
    }

    #[test]
    fn test_decal_expires() {
        let cfg = DecalConfig { lifetime: 1.0, ..Default::default() };
        let mut d = Decal::new(0, Vec3::ZERO, Orientation::identity(), cfg);
        assert!(!d.is_expired());
        d.update(1.5);
        assert!(d.is_expired());
        assert!(!d.active);
    }

    #[test]
    fn test_infinite_lifetime() {
        let cfg = DecalConfig { lifetime: 0.0, ..Default::default() };
        let mut d = Decal::new(0, Vec3::ZERO, Orientation::identity(), cfg);
        d.update(1000.0);
        assert!(!d.is_expired());
        assert!(d.active);
    }

    #[test]
    fn test_atlas_rect_full() {
        let ar = AtlasRect::full();
        let uv = ar.map_uv(0.5, 0.5);
        assert!((uv.x - 0.5).abs() < 1e-6);
        assert!((uv.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_atlas_rect_sub_region() {
        let ar = AtlasRect {
            u_min: 0.25,
            v_min: 0.0,
            u_max: 0.5,
            v_max: 0.5,
        };
        let uv = ar.map_uv(0.0, 0.0);
        assert!((uv.x - 0.25).abs() < 1e-6);
        assert!((uv.y - 0.0).abs() < 1e-6);
        let uv2 = ar.map_uv(1.0, 1.0);
        assert!((uv2.x - 0.5).abs() < 1e-6);
        assert!((uv2.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_system_add_and_count() {
        let mut sys = DecalProjectorSystem::new();
        let id = sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig::default());
        assert_eq!(sys.decal_count(), 1);
        assert!(sys.get_decal(id).is_some());
    }

    #[test]
    fn test_system_remove() {
        let mut sys = DecalProjectorSystem::new();
        let id = sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig::default());
        assert!(sys.remove_decal(id));
        assert_eq!(sys.decal_count(), 0);
    }

    #[test]
    fn test_system_remove_nonexistent() {
        let mut sys = DecalProjectorSystem::new();
        assert!(!sys.remove_decal(999));
    }

    #[test]
    fn test_system_update_removes_expired() {
        let mut sys = DecalProjectorSystem::new();
        let cfg = DecalConfig { lifetime: 1.0, ..Default::default() };
        sys.add_decal(Vec3::ZERO, Orientation::identity(), cfg);
        sys.update(2.0);
        assert_eq!(sys.decal_count(), 0);
    }

    #[test]
    fn test_system_sorted_decals() {
        let mut sys = DecalProjectorSystem::new();
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig { sort_order: 5, ..Default::default() });
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig { sort_order: 1, ..Default::default() });
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig { sort_order: 3, ..Default::default() });
        let sorted = sys.sorted_decals();
        assert_eq!(sorted[0].config.sort_order, 1);
        assert_eq!(sorted[1].config.sort_order, 3);
        assert_eq!(sorted[2].config.sort_order, 5);
    }

    #[test]
    fn test_system_project_point() {
        let mut sys = DecalProjectorSystem::new();
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig::default());
        let results = sys.project_point(Vec3::new(0.0, 0.0, -0.5), Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(results.len(), 1);
        assert!(results[0].1.inside);
    }

    #[test]
    fn test_system_project_point_miss() {
        let mut sys = DecalProjectorSystem::new();
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig::default());
        let results = sys.project_point(Vec3::new(100.0, 100.0, 100.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(results.is_empty());
    }

    #[test]
    fn test_system_clear() {
        let mut sys = DecalProjectorSystem::new();
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig::default());
        sys.add_decal(Vec3::ZERO, Orientation::identity(), DecalConfig::default());
        sys.clear();
        assert_eq!(sys.decal_count(), 0);
    }

    #[test]
    fn test_world_to_local() {
        let o = Orientation::identity();
        let local = world_to_local(Vec3::new(1.0, 2.0, -3.0), Vec3::ZERO, &o);
        assert!((local.x - 1.0).abs() < 1e-5);
        assert!((local.y - 2.0).abs() < 1e-5);
        // forward is (0,0,-1), so dot with (0,0,-3) = 3
        assert!((local.z - 3.0).abs() < 1e-5);
    }
}
