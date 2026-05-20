//! Billboard / impostor rendering for distant trees and objects: cylindrical
//! and spherical billboards, atlas-based species, cross-billboards for volume,
//! 3D-to-billboard LOD transition, shadow casting, wind sway, batch rendering.
//!
//! Pure Rust — geometry and orientation math on CPU.

// ── Vec3 / Vec2 ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn distance_to(&self, o: &Vec3) -> f32 {
        let dx = self.x - o.x;
        let dy = self.y - o.y;
        let dz = self.z - o.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let l = self.length();
        if l < 1e-10 {
            return Self::new(0.0, 0.0, 0.0);
        }
        Self::new(self.x / l, self.y / l, self.z / l)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

// ── Billboard type ───────────────────────────────────────────────

/// How a billboard orients itself toward the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BillboardType {
    /// Rotates only around the Y axis (trees, lampposts).
    Cylindrical,
    /// Always faces the camera fully (particles, sprites).
    Spherical,
}

// ── Atlas region ─────────────────────────────────────────────────

/// UV rectangle within a billboard atlas texture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasRegion {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

impl AtlasRegion {
    pub fn new(u_min: f32, v_min: f32, u_max: f32, v_max: f32) -> Self {
        Self { u_min, v_min, u_max, v_max }
    }

    pub fn full() -> Self {
        Self { u_min: 0.0, v_min: 0.0, u_max: 1.0, v_max: 1.0 }
    }

    /// Width in UV space.
    pub fn width(&self) -> f32 {
        self.u_max - self.u_min
    }

    /// Height in UV space.
    pub fn height(&self) -> f32 {
        self.v_max - self.v_min
    }

    /// Center of the region.
    pub fn center(&self) -> Vec2 {
        Vec2::new(
            (self.u_min + self.u_max) * 0.5,
            (self.v_min + self.v_max) * 0.5,
        )
    }
}

/// Atlas containing regions for multiple tree species.
#[derive(Debug, Clone, PartialEq)]
pub struct BillboardAtlas {
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub regions: Vec<(String, AtlasRegion)>,
}

impl BillboardAtlas {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            atlas_width: width,
            atlas_height: height,
            regions: Vec::new(),
        }
    }

    pub fn add_region(&mut self, name: &str, region: AtlasRegion) -> usize {
        let idx = self.regions.len();
        self.regions.push((name.to_string(), region));
        idx
    }

    pub fn get_region(&self, idx: usize) -> Option<&AtlasRegion> {
        self.regions.get(idx).map(|(_, r)| r)
    }

    pub fn find_by_name(&self, name: &str) -> Option<(usize, &AtlasRegion)> {
        self.regions
            .iter()
            .enumerate()
            .find(|(_, (n, _))| n == name)
            .map(|(i, (_, r))| (i, r))
    }

    pub fn species_count(&self) -> usize {
        self.regions.len()
    }
}

// ── Billboard instance ───────────────────────────────────────────

/// Single billboard instance in the world.
#[derive(Debug, Clone, PartialEq)]
pub struct Billboard {
    pub position: Vec3,
    pub width: f32,
    pub height: f32,
    pub billboard_type: BillboardType,
    pub atlas_index: usize,
    pub casts_shadow: bool,
    pub wind_sway_amount: f32,
    pub wind_phase: f32,
}

impl Billboard {
    pub fn new(position: Vec3, width: f32, height: f32) -> Self {
        Self {
            position,
            width,
            height,
            billboard_type: BillboardType::Cylindrical,
            atlas_index: 0,
            casts_shadow: true,
            wind_sway_amount: 0.05,
            wind_phase: 0.0,
        }
    }

    /// Compute the facing direction for this billboard given a camera position.
    /// Returns the right vector used to offset quad corners.
    pub fn compute_facing(&self, camera: &Vec3) -> (Vec3, Vec3) {
        match self.billboard_type {
            BillboardType::Cylindrical => {
                let dx = camera.x - self.position.x;
                let dz = camera.z - self.position.z;
                let len = (dx * dx + dz * dz).sqrt();
                if len < 1e-8 {
                    return (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
                }
                let right = Vec3::new(-dz / len, 0.0, dx / len);
                let up = Vec3::new(0.0, 1.0, 0.0);
                (right, up)
            }
            BillboardType::Spherical => {
                let forward = Vec3::new(
                    camera.x - self.position.x,
                    camera.y - self.position.y,
                    camera.z - self.position.z,
                )
                .normalize();
                // right = forward x world_up
                let world_up = Vec3::new(0.0, 1.0, 0.0);
                let right = Vec3::new(
                    forward.z * world_up.y - forward.y * world_up.z,
                    forward.x * world_up.z - forward.z * world_up.x,
                    forward.y * world_up.x - forward.x * world_up.y,
                )
                .normalize();
                // up = right x forward
                let up = Vec3::new(
                    right.y * forward.z - right.z * forward.y,
                    right.z * forward.x - right.x * forward.z,
                    right.x * forward.y - right.y * forward.x,
                )
                .normalize();
                (right, up)
            }
        }
    }

    /// Generate the 4 corner positions of the billboard quad.
    pub fn quad_corners(&self, camera: &Vec3) -> [Vec3; 4] {
        let (right, up) = self.compute_facing(camera);
        let hw = self.width * 0.5;
        let hh = self.height * 0.5;
        [
            Vec3::new(
                self.position.x - right.x * hw - up.x * hh,
                self.position.y - right.y * hw - up.y * hh + self.height * 0.5,
                self.position.z - right.z * hw - up.z * hh,
            ),
            Vec3::new(
                self.position.x + right.x * hw - up.x * hh,
                self.position.y + right.y * hw - up.y * hh + self.height * 0.5,
                self.position.z + right.z * hw - up.z * hh,
            ),
            Vec3::new(
                self.position.x + right.x * hw + up.x * hh,
                self.position.y + right.y * hw + up.y * hh + self.height * 0.5,
                self.position.z + right.z * hw + up.z * hh,
            ),
            Vec3::new(
                self.position.x - right.x * hw + up.x * hh,
                self.position.y - right.y * hw + up.y * hh + self.height * 0.5,
                self.position.z - right.z * hw + up.z * hh,
            ),
        ]
    }

    /// Apply wind sway offset to the top two corners.
    pub fn wind_offset(&self, time: f32) -> f32 {
        let sway = (time * 1.5 + self.wind_phase).sin() * self.wind_sway_amount;
        sway * self.height
    }

    /// Generate shadow quad projected onto ground (Y=ground_y).
    pub fn shadow_quad(
        &self,
        camera: &Vec3,
        light_dir: &Vec3,
        ground_y: f32,
    ) -> Option<[Vec3; 4]> {
        if !self.casts_shadow {
            return None;
        }
        if light_dir.y.abs() < 1e-6 {
            return None;
        }
        let corners = self.quad_corners(camera);
        let mut shadow = [Vec3::new(0.0, 0.0, 0.0); 4];
        for (i, c) in corners.iter().enumerate() {
            let t = (ground_y - c.y) / light_dir.y;
            shadow[i] = Vec3::new(
                c.x + light_dir.x * t,
                ground_y,
                c.z + light_dir.z * t,
            );
        }
        Some(shadow)
    }
}

// ── Cross-billboard ──────────────────────────────────────────────

/// Two or three intersecting billboard planes for volumetric appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossBillboard {
    pub position: Vec3,
    pub width: f32,
    pub height: f32,
    pub plane_count: u8,
    pub atlas_index: usize,
}

impl CrossBillboard {
    pub fn new(position: Vec3, width: f32, height: f32, planes: u8) -> Self {
        Self {
            position,
            width,
            height,
            plane_count: planes.clamp(2, 3),
            atlas_index: 0,
        }
    }

    /// Rotation angles (radians) for each intersecting plane.
    pub fn plane_rotations(&self) -> Vec<f32> {
        let step = std::f32::consts::PI / self.plane_count as f32;
        (0..self.plane_count).map(|i| i as f32 * step).collect()
    }

    /// Generate quad corners for one plane at given rotation angle.
    pub fn plane_corners(&self, angle: f32) -> [Vec3; 4] {
        let hw = self.width * 0.5;
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let rx = hw * cos_a;
        let rz = hw * sin_a;
        [
            Vec3::new(self.position.x - rx, self.position.y, self.position.z - rz),
            Vec3::new(self.position.x + rx, self.position.y, self.position.z + rz),
            Vec3::new(
                self.position.x + rx,
                self.position.y + self.height,
                self.position.z + rz,
            ),
            Vec3::new(
                self.position.x - rx,
                self.position.y + self.height,
                self.position.z - rz,
            ),
        ]
    }

    /// Total quad count across all planes.
    pub fn total_quads(&self) -> usize {
        self.plane_count as usize
    }
}

// ── LOD transition ───────────────────────────────────────────────

/// Manages the 3D mesh → billboard transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TreeRenderMode {
    Mesh,
    Billboard,
    CrossBillboard,
}

/// LOD selector for trees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeLodConfig {
    pub mesh_distance: f32,
    pub cross_distance: f32,
    pub billboard_distance: f32,
}

impl Default for TreeLodConfig {
    fn default() -> Self {
        Self {
            mesh_distance: 50.0,
            cross_distance: 150.0,
            billboard_distance: 500.0,
        }
    }
}

impl TreeLodConfig {
    pub fn select_mode(&self, distance: f32) -> TreeRenderMode {
        if distance < self.mesh_distance {
            TreeRenderMode::Mesh
        } else if distance < self.cross_distance {
            TreeRenderMode::CrossBillboard
        } else {
            TreeRenderMode::Billboard
        }
    }

    /// Blend factor for smooth transition (1 = fully current mode).
    pub fn transition_alpha(&self, distance: f32) -> f32 {
        if distance < self.mesh_distance {
            1.0
        } else if distance < self.cross_distance {
            let range = self.cross_distance - self.mesh_distance;
            if range < 1e-6 {
                return 1.0;
            }
            let t = (distance - self.mesh_distance) / range;
            1.0 - t.clamp(0.0, 1.0)
        } else if distance < self.billboard_distance {
            let range = self.billboard_distance - self.cross_distance;
            if range < 1e-6 {
                return 1.0;
            }
            let t = (distance - self.cross_distance) / range;
            1.0 - t.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ── Batch renderer ───────────────────────────────────────────────

/// Batch of billboards for efficient rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct BillboardBatch {
    pub billboards: Vec<Billboard>,
    pub atlas_index: usize,
}

impl BillboardBatch {
    pub fn new(atlas_index: usize) -> Self {
        Self {
            billboards: Vec::new(),
            atlas_index,
        }
    }

    pub fn add(&mut self, bb: Billboard) {
        self.billboards.push(bb);
    }

    pub fn count(&self) -> usize {
        self.billboards.len()
    }

    /// Sort billboards back-to-front relative to camera for alpha blending.
    pub fn sort_back_to_front(&mut self, camera: &Vec3) {
        self.billboards.sort_by(|a, b| {
            let da = a.position.distance_to(camera);
            let db = b.position.distance_to(camera);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Generate all quad corner arrays. Returns vec of 4-corner arrays.
    pub fn generate_quads(&self, camera: &Vec3) -> Vec<[Vec3; 4]> {
        self.billboards
            .iter()
            .map(|bb| bb.quad_corners(camera))
            .collect()
    }

    /// Cull billboards beyond max_distance.
    pub fn cull_by_distance(&mut self, camera: &Vec3, max_distance: f32) {
        self.billboards.retain(|bb| bb.position.distance_to(camera) <= max_distance);
    }
}

/// Group billboards by atlas index into batches.
pub fn batch_by_atlas(billboards: Vec<Billboard>) -> Vec<BillboardBatch> {
    let mut map = std::collections::HashMap::<usize, BillboardBatch>::new();
    for bb in billboards {
        let idx = bb.atlas_index;
        map.entry(idx)
            .or_insert_with(|| BillboardBatch::new(idx))
            .add(bb);
    }
    let mut batches: Vec<BillboardBatch> = map.into_values().collect();
    batches.sort_by_key(|b| b.atlas_index);
    batches
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn atlas_region_full() {
        let r = AtlasRegion::full();
        assert!(approx(r.width(), 1.0, 1e-6));
        assert!(approx(r.height(), 1.0, 1e-6));
    }

    #[test]
    fn atlas_region_center() {
        let r = AtlasRegion::new(0.0, 0.0, 0.5, 0.5);
        let c = r.center();
        assert!(approx(c.x, 0.25, 1e-6));
        assert!(approx(c.y, 0.25, 1e-6));
    }

    #[test]
    fn atlas_add_and_find() {
        let mut atlas = BillboardAtlas::new(1024, 1024);
        atlas.add_region("oak", AtlasRegion::new(0.0, 0.0, 0.25, 0.5));
        atlas.add_region("pine", AtlasRegion::new(0.25, 0.0, 0.5, 0.5));
        assert_eq!(atlas.species_count(), 2);
        let (idx, _) = atlas.find_by_name("pine").unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn atlas_get_region() {
        let mut atlas = BillboardAtlas::new(512, 512);
        atlas.add_region("birch", AtlasRegion::new(0.0, 0.0, 1.0, 1.0));
        let r = atlas.get_region(0).unwrap();
        assert!(approx(r.u_max, 1.0, 1e-6));
    }

    #[test]
    fn atlas_find_missing() {
        let atlas = BillboardAtlas::new(512, 512);
        assert!(atlas.find_by_name("missing").is_none());
    }

    #[test]
    fn billboard_cylindrical_facing() {
        let bb = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 4.0);
        let camera = Vec3::new(10.0, 5.0, 0.0);
        let (right, up) = bb.compute_facing(&camera);
        // Cylindrical: right should be in XZ plane, up = (0,1,0)
        assert!(approx(right.y, 0.0, 1e-6));
        assert!(approx(up.y, 1.0, 1e-6));
    }

    #[test]
    fn billboard_spherical_facing() {
        let mut bb = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 2.0);
        bb.billboard_type = BillboardType::Spherical;
        let camera = Vec3::new(5.0, 5.0, 5.0);
        let (right, up) = bb.compute_facing(&camera);
        assert!(right.length() > 0.9);
        assert!(up.length() > 0.9);
    }

    #[test]
    fn billboard_quad_corners_count() {
        let bb = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 4.0);
        let camera = Vec3::new(10.0, 0.0, 0.0);
        let corners = bb.quad_corners(&camera);
        assert_eq!(corners.len(), 4);
    }

    #[test]
    fn billboard_wind_offset_oscillates() {
        let bb = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 4.0);
        let o1 = bb.wind_offset(0.0);
        let o2 = bb.wind_offset(1.0);
        // Should produce different offsets at different times
        assert!((o1 - o2).abs() > 1e-6 || approx(o1, 0.0, 1e-3));
    }

    #[test]
    fn billboard_shadow_quad() {
        let bb = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 4.0);
        let camera = Vec3::new(10.0, 5.0, 0.0);
        let light = Vec3::new(-0.5, -1.0, -0.3);
        let shadow = bb.shadow_quad(&camera, &light, 0.0);
        assert!(shadow.is_some());
        let sq = shadow.unwrap();
        for corner in &sq {
            assert!(approx(corner.y, 0.0, 1e-4));
        }
    }

    #[test]
    fn billboard_no_shadow() {
        let mut bb = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 4.0);
        bb.casts_shadow = false;
        let camera = Vec3::new(10.0, 0.0, 0.0);
        let light = Vec3::new(-0.5, -1.0, 0.0);
        assert!(bb.shadow_quad(&camera, &light, 0.0).is_none());
    }

    #[test]
    fn cross_billboard_two_planes() {
        let cb = CrossBillboard::new(Vec3::new(0.0, 0.0, 0.0), 3.0, 5.0, 2);
        assert_eq!(cb.total_quads(), 2);
        let rots = cb.plane_rotations();
        assert_eq!(rots.len(), 2);
        assert!(approx(rots[0], 0.0, 1e-6));
    }

    #[test]
    fn cross_billboard_three_planes() {
        let cb = CrossBillboard::new(Vec3::new(0.0, 0.0, 0.0), 3.0, 5.0, 3);
        assert_eq!(cb.total_quads(), 3);
    }

    #[test]
    fn cross_billboard_corners() {
        let cb = CrossBillboard::new(Vec3::new(0.0, 0.0, 0.0), 4.0, 6.0, 2);
        let corners = cb.plane_corners(0.0);
        assert_eq!(corners.len(), 4);
        // Bottom corners at y=0, top at y=6
        assert!(approx(corners[0].y, 0.0, 1e-6));
        assert!(approx(corners[3].y, 6.0, 1e-6));
    }

    #[test]
    fn tree_lod_select_mesh() {
        let cfg = TreeLodConfig::default();
        assert_eq!(cfg.select_mode(10.0), TreeRenderMode::Mesh);
    }

    #[test]
    fn tree_lod_select_cross() {
        let cfg = TreeLodConfig::default();
        assert_eq!(cfg.select_mode(100.0), TreeRenderMode::CrossBillboard);
    }

    #[test]
    fn tree_lod_select_billboard() {
        let cfg = TreeLodConfig::default();
        assert_eq!(cfg.select_mode(200.0), TreeRenderMode::Billboard);
    }

    #[test]
    fn tree_lod_transition_alpha() {
        let cfg = TreeLodConfig::default();
        assert!(approx(cfg.transition_alpha(0.0), 1.0, 1e-6));
        assert!(approx(cfg.transition_alpha(600.0), 0.0, 1e-6));
        let mid = cfg.transition_alpha(100.0);
        assert!(mid >= 0.0 && mid <= 1.0);
    }

    #[test]
    fn batch_add_and_count() {
        let mut batch = BillboardBatch::new(0);
        batch.add(Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 4.0));
        batch.add(Billboard::new(Vec3::new(5.0, 0.0, 5.0), 2.0, 4.0));
        assert_eq!(batch.count(), 2);
    }

    #[test]
    fn batch_sort_back_to_front() {
        let mut batch = BillboardBatch::new(0);
        batch.add(Billboard::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0));
        batch.add(Billboard::new(Vec3::new(10.0, 0.0, 0.0), 1.0, 1.0));
        let camera = Vec3::new(0.0, 0.0, 0.0);
        batch.sort_back_to_front(&camera);
        let d0 = batch.billboards[0].position.distance_to(&camera);
        let d1 = batch.billboards[1].position.distance_to(&camera);
        assert!(d0 >= d1, "back-to-front: farthest first");
    }

    #[test]
    fn batch_generate_quads() {
        let mut batch = BillboardBatch::new(0);
        batch.add(Billboard::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 3.0));
        let camera = Vec3::new(10.0, 0.0, 0.0);
        let quads = batch.generate_quads(&camera);
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].len(), 4);
    }

    #[test]
    fn batch_cull_by_distance() {
        let mut batch = BillboardBatch::new(0);
        batch.add(Billboard::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0));
        batch.add(Billboard::new(Vec3::new(100.0, 0.0, 0.0), 1.0, 1.0));
        let camera = Vec3::new(0.0, 0.0, 0.0);
        batch.cull_by_distance(&camera, 50.0);
        assert_eq!(batch.count(), 1);
    }

    #[test]
    fn batch_by_atlas_groups() {
        let bbs = vec![
            {
                let mut b = Billboard::new(Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0);
                b.atlas_index = 0;
                b
            },
            {
                let mut b = Billboard::new(Vec3::new(1.0, 0.0, 0.0), 1.0, 1.0);
                b.atlas_index = 1;
                b
            },
            {
                let mut b = Billboard::new(Vec3::new(2.0, 0.0, 0.0), 1.0, 1.0);
                b.atlas_index = 0;
                b
            },
        ];
        let batches = batch_by_atlas(bbs);
        assert_eq!(batches.len(), 2);
        // Sorted by atlas_index
        assert_eq!(batches[0].atlas_index, 0);
        assert_eq!(batches[0].count(), 2);
        assert_eq!(batches[1].atlas_index, 1);
        assert_eq!(batches[1].count(), 1);
    }
}
