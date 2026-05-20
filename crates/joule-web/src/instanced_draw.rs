//! GPU-style instanced drawing — instance buffer with per-instance transform
//! (Mat4), color tint, custom data. Collect instances for same mesh+material.
//! Sort by distance for transparent instances. Instance buffer growth/compaction.
//! Dynamic vs static instance sets. Frustum cull per instance. Statistics
//! (instance count, culled count, draw calls saved).

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
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
    pub fn min_comp(self, o: Self) -> Self { Self { x: self.x.min(o.x), y: self.y.min(o.y), z: self.z.min(o.z) } }
    pub fn max_comp(self, o: Self) -> Self { Self { x: self.x.max(o.x), y: self.y.max(o.y), z: self.z.max(o.z) } }
}

// ── Mat4 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    pub fn identity() -> Self {
        Self { m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]}
    }

    pub fn translation(v: Vec3) -> Self {
        let mut m = Self::identity();
        m.m[0][3] = v.x; m.m[1][3] = v.y; m.m[2][3] = v.z;
        m
    }

    pub fn scaling(v: Vec3) -> Self {
        let mut m = Self::identity();
        m.m[0][0] = v.x; m.m[1][1] = v.y; m.m[2][2] = v.z;
        m
    }

    pub fn get_translation(&self) -> Vec3 {
        Vec3::new(self.m[0][3], self.m[1][3], self.m[2][3])
    }

    pub fn mul(self, o: Self) -> Self {
        let mut r = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    r[i][j] += self.m[i][k] * o.m[k][j];
                }
            }
        }
        Self { m: r }
    }

    pub fn transform_point(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0]*v.x + self.m[0][1]*v.y + self.m[0][2]*v.z + self.m[0][3],
            self.m[1][0]*v.x + self.m[1][1]*v.y + self.m[1][2]*v.z + self.m[1][3],
            self.m[2][0]*v.x + self.m[2][1]*v.y + self.m[2][2]*v.z + self.m[2][3],
        )
    }
}

// ── Frustum (simplified) ─────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub distance: f64,
}

impl Plane {
    pub fn distance_to_point(&self, p: Vec3) -> f64 {
        self.normal.dot(p) + self.distance
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Test if a sphere is outside the frustum.
    pub fn is_sphere_outside(&self, center: Vec3, radius: f64) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(center) < -radius {
                return true;
            }
        }
        false
    }
}

// ── Color ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub fn white() -> Self { Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 } }
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self { Self { r, g, b, a } }
    pub fn is_transparent(&self) -> bool { self.a < 1.0 - 1e-9 }
}

// ── InstanceData ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct InstanceData {
    pub transform: Mat4,
    pub color_tint: Color,
    pub custom: Vec<f64>,
    pub visible: bool,
    pub bounding_radius: f64,
}

impl InstanceData {
    pub fn new(transform: Mat4) -> Self {
        Self {
            transform,
            color_tint: Color::white(),
            custom: Vec::new(),
            visible: true,
            bounding_radius: 1.0,
        }
    }

    pub fn with_color(mut self, color: Color) -> Self { self.color_tint = color; self }
    pub fn with_custom(mut self, data: Vec<f64>) -> Self { self.custom = data; self }
    pub fn with_radius(mut self, r: f64) -> Self { self.bounding_radius = r; self }

    pub fn position(&self) -> Vec3 { self.transform.get_translation() }
    pub fn is_transparent(&self) -> bool { self.color_tint.is_transparent() }
}

// ── MeshMaterialKey ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshMaterialKey {
    pub mesh_id: u64,
    pub material_id: u64,
}

impl MeshMaterialKey {
    pub fn new(mesh_id: u64, material_id: u64) -> Self { Self { mesh_id, material_id } }
}

// ── InstanceSet ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceSetMode {
    Dynamic,
    Static,
}

#[derive(Debug, Clone)]
pub struct InstanceSet {
    pub key: MeshMaterialKey,
    pub instances: Vec<InstanceData>,
    pub mode: InstanceSetMode,
    pub capacity: usize,
    pub dirty: bool,
}

impl InstanceSet {
    pub fn new(key: MeshMaterialKey, mode: InstanceSetMode) -> Self {
        Self { key, instances: Vec::new(), mode, capacity: 0, dirty: true }
    }

    pub fn add(&mut self, instance: InstanceData) -> usize {
        let idx = self.instances.len();
        self.instances.push(instance);
        self.dirty = true;
        idx
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.instances.len() {
            self.instances.swap_remove(index);
            self.dirty = true;
        }
    }

    pub fn len(&self) -> usize { self.instances.len() }
    pub fn is_empty(&self) -> bool { self.instances.is_empty() }

    pub fn visible_count(&self) -> usize {
        self.instances.iter().filter(|i| i.visible).count()
    }

    /// Compact: remove invisible instances (for dynamic sets).
    pub fn compact(&mut self) {
        self.instances.retain(|i| i.visible);
        self.dirty = true;
    }

    /// Ensure capacity grows to at least `new_cap`.
    pub fn ensure_capacity(&mut self, new_cap: usize) {
        if new_cap > self.capacity {
            // Grow by doubling or to new_cap, whichever is larger
            self.capacity = self.capacity.max(16).max(new_cap);
            self.capacity = self.capacity.next_power_of_two();
            self.dirty = true;
        }
    }

    /// Sort transparent instances back-to-front relative to camera.
    pub fn sort_by_distance(&mut self, camera_pos: Vec3) {
        self.instances.sort_by(|a, b| {
            let da = a.position().distance(camera_pos);
            let db = b.position().distance(camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal) // far first
        });
        self.dirty = true;
    }

    /// Frustum-cull all instances, setting visibility flags.
    pub fn frustum_cull(&mut self, frustum: &Frustum) -> CullResult {
        let mut visible = 0usize;
        let mut culled = 0usize;
        for inst in &mut self.instances {
            let pos = inst.position();
            if frustum.is_sphere_outside(pos, inst.bounding_radius) {
                inst.visible = false;
                culled += 1;
            } else {
                inst.visible = true;
                visible += 1;
            }
        }
        CullResult { visible, culled }
    }

    /// Reset all instances to visible.
    pub fn reset_visibility(&mut self) {
        for inst in &mut self.instances { inst.visible = true; }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CullResult {
    pub visible: usize,
    pub culled: usize,
}

// ── DrawCall ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DrawCall {
    pub key: MeshMaterialKey,
    pub instance_count: usize,
    pub is_transparent: bool,
}

// ── InstancedDrawStats ───────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstancedDrawStats {
    pub total_instances: usize,
    pub visible_instances: usize,
    pub culled_instances: usize,
    pub draw_calls: usize,
    pub draw_calls_saved: usize,
    pub opaque_draw_calls: usize,
    pub transparent_draw_calls: usize,
}

// ── InstancedRenderer ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InstancedRenderer {
    sets: Vec<InstanceSet>,
}

impl InstancedRenderer {
    pub fn new() -> Self { Self { sets: Vec::new() } }

    pub fn add_set(&mut self, set: InstanceSet) -> usize {
        let idx = self.sets.len();
        self.sets.push(set);
        idx
    }

    pub fn get_set(&self, index: usize) -> Option<&InstanceSet> { self.sets.get(index) }
    pub fn get_set_mut(&mut self, index: usize) -> Option<&mut InstanceSet> { self.sets.get_mut(index) }

    /// Find or create a set for the given mesh+material.
    pub fn get_or_create_set(&mut self, key: MeshMaterialKey, mode: InstanceSetMode) -> usize {
        for (i, set) in self.sets.iter().enumerate() {
            if set.key == key { return i; }
        }
        self.add_set(InstanceSet::new(key, mode))
    }

    /// Frustum cull all sets.
    pub fn frustum_cull_all(&mut self, frustum: &Frustum) {
        for set in &mut self.sets {
            set.frustum_cull(frustum);
        }
    }

    /// Sort transparent instance sets by distance.
    pub fn sort_transparent(&mut self, camera_pos: Vec3) {
        for set in &mut self.sets {
            let has_transparent = set.instances.iter().any(|i| i.is_transparent());
            if has_transparent {
                set.sort_by_distance(camera_pos);
            }
        }
    }

    /// Generate draw calls from current state.
    pub fn generate_draw_calls(&self) -> Vec<DrawCall> {
        let mut calls = Vec::new();
        for set in &self.sets {
            let visible = set.visible_count();
            if visible == 0 { continue; }
            let is_transparent = set.instances.iter().any(|i| i.visible && i.is_transparent());
            calls.push(DrawCall {
                key: set.key,
                instance_count: visible,
                is_transparent,
            });
        }
        calls
    }

    /// Compute statistics.
    pub fn stats(&self) -> InstancedDrawStats {
        let total: usize = self.sets.iter().map(|s| s.len()).sum();
        let visible: usize = self.sets.iter().map(|s| s.visible_count()).sum();
        let culled = total - visible;
        let draw_calls = self.generate_draw_calls().len();
        let saved = if total > 0 { total.saturating_sub(draw_calls) } else { 0 };
        let opaque = self.generate_draw_calls().iter().filter(|d| !d.is_transparent).count();
        let transparent = draw_calls - opaque;
        InstancedDrawStats {
            total_instances: total,
            visible_instances: visible,
            culled_instances: culled,
            draw_calls,
            draw_calls_saved: saved,
            opaque_draw_calls: opaque,
            transparent_draw_calls: transparent,
        }
    }

    /// Compact all dynamic sets.
    pub fn compact_all(&mut self) {
        for set in &mut self.sets {
            if set.mode == InstanceSetMode::Dynamic {
                set.compact();
            }
        }
    }

    /// Reset visibility on all sets.
    pub fn reset_all_visibility(&mut self) {
        for set in &mut self.sets {
            set.reset_visibility();
        }
    }

    pub fn set_count(&self) -> usize { self.sets.len() }
    pub fn total_instance_count(&self) -> usize {
        self.sets.iter().map(|s| s.len()).sum()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frustum_all_visible() -> Frustum {
        // Frustum that includes everything (planes far away)
        Frustum {
            planes: [
                Plane { normal: Vec3::new( 1.0, 0.0, 0.0), distance: 1000.0 },
                Plane { normal: Vec3::new(-1.0, 0.0, 0.0), distance: 1000.0 },
                Plane { normal: Vec3::new(0.0,  1.0, 0.0), distance: 1000.0 },
                Plane { normal: Vec3::new(0.0, -1.0, 0.0), distance: 1000.0 },
                Plane { normal: Vec3::new(0.0, 0.0,  1.0), distance: 1000.0 },
                Plane { normal: Vec3::new(0.0, 0.0, -1.0), distance: 1000.0 },
            ],
        }
    }

    fn make_tight_frustum() -> Frustum {
        // Tight frustum around origin (roughly -5 to 5 on all axes)
        Frustum {
            planes: [
                Plane { normal: Vec3::new( 1.0, 0.0, 0.0), distance: 5.0 },
                Plane { normal: Vec3::new(-1.0, 0.0, 0.0), distance: 5.0 },
                Plane { normal: Vec3::new(0.0,  1.0, 0.0), distance: 5.0 },
                Plane { normal: Vec3::new(0.0, -1.0, 0.0), distance: 5.0 },
                Plane { normal: Vec3::new(0.0, 0.0,  1.0), distance: 5.0 },
                Plane { normal: Vec3::new(0.0, 0.0, -1.0), distance: 5.0 },
            ],
        }
    }

    fn key(m: u64, mat: u64) -> MeshMaterialKey { MeshMaterialKey::new(m, mat) }

    #[test]
    fn test_instance_data_position() {
        let inst = InstanceData::new(Mat4::translation(Vec3::new(1.0, 2.0, 3.0)));
        let p = inst.position();
        assert!((p.x - 1.0).abs() < 1e-9);
        assert!((p.y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_instance_transparency() {
        let inst = InstanceData::new(Mat4::identity())
            .with_color(Color::new(1.0, 1.0, 1.0, 0.5));
        assert!(inst.is_transparent());
    }

    #[test]
    fn test_instance_opaque() {
        let inst = InstanceData::new(Mat4::identity());
        assert!(!inst.is_transparent());
    }

    #[test]
    fn test_instance_set_add() {
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        let idx = set.add(InstanceData::new(Mat4::identity()));
        assert_eq!(idx, 0);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_instance_set_remove() {
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        set.add(InstanceData::new(Mat4::translation(Vec3::new(1.0, 0.0, 0.0))));
        set.remove(0);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_frustum_cull_all_visible() {
        let frustum = make_frustum_all_visible();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        set.add(InstanceData::new(Mat4::translation(Vec3::new(1.0, 0.0, 0.0))));
        let result = set.frustum_cull(&frustum);
        assert_eq!(result.visible, 2);
        assert_eq!(result.culled, 0);
    }

    #[test]
    fn test_frustum_cull_some_culled() {
        let frustum = make_tight_frustum();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity())); // inside
        set.add(InstanceData::new(Mat4::translation(Vec3::new(100.0, 0.0, 0.0)))); // outside
        let result = set.frustum_cull(&frustum);
        assert_eq!(result.visible, 1);
        assert_eq!(result.culled, 1);
    }

    #[test]
    fn test_sort_by_distance() {
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::translation(Vec3::new(1.0, 0.0, 0.0))));
        set.add(InstanceData::new(Mat4::translation(Vec3::new(10.0, 0.0, 0.0))));
        set.add(InstanceData::new(Mat4::translation(Vec3::new(5.0, 0.0, 0.0))));
        set.sort_by_distance(Vec3::zero());
        // Back-to-front: 10, 5, 1
        assert!((set.instances[0].position().x - 10.0).abs() < 1e-9);
        assert!((set.instances[1].position().x - 5.0).abs() < 1e-9);
        assert!((set.instances[2].position().x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compact() {
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        let mut inst = InstanceData::new(Mat4::translation(Vec3::new(1.0, 0.0, 0.0)));
        inst.visible = false;
        set.add(inst);
        set.compact();
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_ensure_capacity() {
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.ensure_capacity(100);
        assert!(set.capacity >= 100);
        assert!(set.capacity.is_power_of_two());
    }

    #[test]
    fn test_draw_calls_generation() {
        let mut renderer = InstancedRenderer::new();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        set.add(InstanceData::new(Mat4::identity()));
        renderer.add_set(set);
        let calls = renderer.generate_draw_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].instance_count, 2);
    }

    #[test]
    fn test_draw_calls_empty_set() {
        let mut renderer = InstancedRenderer::new();
        renderer.add_set(InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic));
        let calls = renderer.generate_draw_calls();
        assert_eq!(calls.len(), 0);
    }

    #[test]
    fn test_multiple_sets() {
        let mut renderer = InstancedRenderer::new();
        let mut s1 = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        s1.add(InstanceData::new(Mat4::identity()));
        let mut s2 = InstanceSet::new(key(2, 1), InstanceSetMode::Dynamic);
        s2.add(InstanceData::new(Mat4::identity()));
        renderer.add_set(s1);
        renderer.add_set(s2);
        let calls = renderer.generate_draw_calls();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_get_or_create() {
        let mut renderer = InstancedRenderer::new();
        let i1 = renderer.get_or_create_set(key(1, 1), InstanceSetMode::Dynamic);
        let i2 = renderer.get_or_create_set(key(1, 1), InstanceSetMode::Dynamic);
        assert_eq!(i1, i2); // same key returns same set
        let i3 = renderer.get_or_create_set(key(2, 1), InstanceSetMode::Dynamic);
        assert_ne!(i1, i3);
    }

    #[test]
    fn test_stats() {
        let mut renderer = InstancedRenderer::new();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        set.add(InstanceData::new(Mat4::identity()));
        set.add(InstanceData::new(Mat4::identity()));
        renderer.add_set(set);
        let stats = renderer.stats();
        assert_eq!(stats.total_instances, 3);
        assert_eq!(stats.visible_instances, 3);
        assert_eq!(stats.draw_calls, 1);
        assert_eq!(stats.draw_calls_saved, 2);
    }

    #[test]
    fn test_stats_with_culling() {
        let mut renderer = InstancedRenderer::new();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        set.add(InstanceData::new(Mat4::translation(Vec3::new(100.0, 0.0, 0.0))));
        renderer.add_set(set);
        renderer.frustum_cull_all(&make_tight_frustum());
        let stats = renderer.stats();
        assert_eq!(stats.culled_instances, 1);
        assert_eq!(stats.visible_instances, 1);
    }

    #[test]
    fn test_reset_visibility() {
        let mut renderer = InstancedRenderer::new();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::translation(Vec3::new(100.0, 0.0, 0.0))));
        renderer.add_set(set);
        renderer.frustum_cull_all(&make_tight_frustum());
        assert_eq!(renderer.stats().visible_instances, 0);
        renderer.reset_all_visibility();
        assert_eq!(renderer.stats().visible_instances, 1);
    }

    #[test]
    fn test_transparent_draw_calls() {
        let mut renderer = InstancedRenderer::new();
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity())
            .with_color(Color::new(1.0, 0.0, 0.0, 0.5)));
        renderer.add_set(set);
        let stats = renderer.stats();
        assert_eq!(stats.transparent_draw_calls, 1);
        assert_eq!(stats.opaque_draw_calls, 0);
    }

    #[test]
    fn test_custom_data() {
        let inst = InstanceData::new(Mat4::identity())
            .with_custom(vec![1.0, 2.0, 3.0]);
        assert_eq!(inst.custom.len(), 3);
        assert!((inst.custom[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_static_mode() {
        let set = InstanceSet::new(key(1, 1), InstanceSetMode::Static);
        assert_eq!(set.mode, InstanceSetMode::Static);
    }

    #[test]
    fn test_compact_only_dynamic() {
        let mut renderer = InstancedRenderer::new();
        let mut dynamic_set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        let mut inst = InstanceData::new(Mat4::identity());
        inst.visible = false;
        dynamic_set.add(inst);
        renderer.add_set(dynamic_set);

        let mut static_set = InstanceSet::new(key(2, 1), InstanceSetMode::Static);
        let mut inst2 = InstanceData::new(Mat4::identity());
        inst2.visible = false;
        static_set.add(inst2);
        renderer.add_set(static_set);

        renderer.compact_all();
        assert_eq!(renderer.get_set(0).unwrap().len(), 0); // dynamic compacted
        assert_eq!(renderer.get_set(1).unwrap().len(), 1); // static unchanged
    }

    #[test]
    fn test_visible_count() {
        let mut set = InstanceSet::new(key(1, 1), InstanceSetMode::Dynamic);
        set.add(InstanceData::new(Mat4::identity()));
        let mut hidden = InstanceData::new(Mat4::identity());
        hidden.visible = false;
        set.add(hidden);
        assert_eq!(set.visible_count(), 1);
    }

    #[test]
    fn test_bounding_radius() {
        let inst = InstanceData::new(Mat4::identity()).with_radius(5.0);
        assert!((inst.bounding_radius - 5.0).abs() < 1e-9);
    }
}
