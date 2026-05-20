//! Level-of-detail management — LOD levels with distance thresholds, smooth
//! transitions (cross-fade / dithered), screen-size based LOD (projected pixel
//! size), LOD bias, per-object LOD override, LOD groups, hysteresis (different
//! thresholds for switching up vs down to prevent oscillation), billboard LOD
//! for distant objects.

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
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn length(self) -> f64 { (self.x * self.x + self.y * self.y + self.z * self.z).sqrt() }
    pub fn distance(self, o: Self) -> f64 { self.sub(o).length() }
}

// ── TransitionMode ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionMode {
    /// Snap instantly to the new LOD level.
    Instant,
    /// Cross-fade between LOD levels over a duration.
    CrossFade,
    /// Dithered transition (screen-door transparency).
    Dithered,
}

// ── LodLevel ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct LodLevel {
    /// Maximum distance at which this LOD is used.
    pub max_distance: f64,
    /// Hysteresis: distance threshold when switching *to* a lower-detail LOD.
    /// Must be >= max_distance to prevent oscillation.
    pub switch_out_distance: f64,
    /// Mesh/asset identifier for this LOD.
    pub mesh_id: u64,
    /// Triangle count (for statistics/debugging).
    pub triangle_count: u32,
    /// Screen pixel height below which this LOD activates (0 = use distance).
    pub min_screen_pixels: f64,
}

impl LodLevel {
    pub fn new(max_distance: f64, mesh_id: u64, triangle_count: u32) -> Self {
        Self {
            max_distance,
            switch_out_distance: max_distance * 1.1, // 10% hysteresis by default
            mesh_id,
            triangle_count,
            min_screen_pixels: 0.0,
        }
    }

    pub fn with_hysteresis(mut self, switch_out: f64) -> Self {
        self.switch_out_distance = switch_out;
        self
    }

    pub fn with_screen_pixels(mut self, pixels: f64) -> Self {
        self.min_screen_pixels = pixels;
        self
    }
}

// ── LodConfig ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct LodConfig {
    /// LOD levels sorted by increasing distance.
    pub levels: Vec<LodLevel>,
    /// Global LOD bias: negative = higher quality, positive = lower quality.
    pub bias: f64,
    /// Transition mode between LOD levels.
    pub transition: TransitionMode,
    /// Duration of cross-fade/dither transitions in seconds.
    pub transition_duration: f64,
    /// If true, use screen-size metric instead of distance where available.
    pub use_screen_size: bool,
    /// Billboard LOD: if Some, objects beyond this distance render as billboards.
    pub billboard_distance: Option<f64>,
}

impl LodConfig {
    pub fn new(levels: Vec<LodLevel>) -> Self {
        Self {
            levels,
            bias: 0.0,
            transition: TransitionMode::Instant,
            transition_duration: 0.3,
            use_screen_size: false,
            billboard_distance: None,
        }
    }

    pub fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_transition(mut self, mode: TransitionMode, duration: f64) -> Self {
        self.transition = mode;
        self.transition_duration = duration;
        self
    }

    pub fn with_billboard(mut self, distance: f64) -> Self {
        self.billboard_distance = Some(distance);
        self
    }

    pub fn with_screen_size(mut self, enabled: bool) -> Self {
        self.use_screen_size = enabled;
        self
    }

    /// Select LOD level for a given distance.
    pub fn select_level(&self, distance: f64) -> usize {
        let biased = distance + self.bias;
        for (i, level) in self.levels.iter().enumerate() {
            if biased <= level.max_distance {
                return i;
            }
        }
        self.levels.len().saturating_sub(1)
    }

    /// Select LOD level considering hysteresis (needs current level).
    pub fn select_level_hysteresis(&self, distance: f64, current_level: usize) -> usize {
        let biased = distance + self.bias;
        // Going to lower detail (farther): use switch_out_distance
        if current_level < self.levels.len() {
            let current = &self.levels[current_level];
            if biased > current.switch_out_distance {
                // Check next levels
                for i in (current_level + 1)..self.levels.len() {
                    if biased <= self.levels[i].max_distance {
                        return i;
                    }
                }
                return self.levels.len().saturating_sub(1);
            }
        }
        // Going to higher detail (closer): use max_distance of target
        if current_level > 0 {
            let prev = &self.levels[current_level - 1];
            if biased <= prev.max_distance {
                return current_level - 1;
            }
        }
        current_level
    }

    /// Select LOD level based on projected screen size (pixels).
    pub fn select_level_screen_size(&self, screen_pixels: f64) -> usize {
        // Larger screen size = more detail (lower LOD index).
        // Iterate forward: first level whose min_screen_pixels is satisfied wins.
        for (i, level) in self.levels.iter().enumerate() {
            if level.min_screen_pixels > 0.0 && screen_pixels >= level.min_screen_pixels {
                return i;
            }
        }
        // If screen size very small, use lowest detail
        self.levels.len().saturating_sub(1)
    }

    /// Is the object beyond billboard distance?
    pub fn is_billboard(&self, distance: f64) -> bool {
        self.billboard_distance.map_or(false, |bd| distance > bd)
    }
}

// ── LodTransition ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct LodTransition {
    pub from_level: usize,
    pub to_level: usize,
    pub progress: f64, // 0.0 = fully from, 1.0 = fully to
    pub duration: f64,
    pub elapsed: f64,
    pub mode: TransitionMode,
}

impl LodTransition {
    pub fn new(from: usize, to: usize, mode: TransitionMode, duration: f64) -> Self {
        Self { from_level: from, to_level: to, progress: 0.0, duration, elapsed: 0.0, mode }
    }

    pub fn update(&mut self, dt: f64) {
        self.elapsed += dt;
        if self.duration > 0.0 {
            self.progress = (self.elapsed / self.duration).min(1.0);
        } else {
            self.progress = 1.0;
        }
    }

    pub fn is_complete(&self) -> bool { self.progress >= 1.0 }
    pub fn current_level(&self) -> usize {
        if self.progress >= 0.5 { self.to_level } else { self.from_level }
    }

    /// Cross-fade alpha for the `from` mesh.
    pub fn from_alpha(&self) -> f64 { 1.0 - self.progress }
    /// Cross-fade alpha for the `to` mesh.
    pub fn to_alpha(&self) -> f64 { self.progress }

    /// Dither threshold for current pixel (0..1 range dither pattern).
    pub fn dither_threshold(&self) -> f64 { self.progress }
}

// ── LodObject ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LodObject {
    pub id: u64,
    pub position: Vec3,
    pub bounding_radius: f64,
    pub current_level: usize,
    pub override_level: Option<usize>,
    pub transition: Option<LodTransition>,
    pub config_index: usize, // index into LodSystem.configs
}

impl LodObject {
    pub fn new(id: u64, position: Vec3, bounding_radius: f64, config_index: usize) -> Self {
        Self {
            id,
            position,
            bounding_radius,
            current_level: 0,
            override_level: None,
            transition: None,
            config_index,
        }
    }

    pub fn set_override(&mut self, level: Option<usize>) {
        self.override_level = level;
    }

    pub fn effective_level(&self) -> usize {
        if let Some(ovr) = self.override_level { return ovr; }
        if let Some(tr) = &self.transition {
            if !tr.is_complete() { return tr.current_level(); }
        }
        self.current_level
    }
}

// ── LodGroup ─────────────────────────────────────────────────

/// A set of objects that share LOD state (e.g., a building made of parts).
#[derive(Debug, Clone)]
pub struct LodGroup {
    pub id: u64,
    pub object_ids: Vec<u64>,
    pub center: Vec3,
    pub bounding_radius: f64,
    pub current_level: usize,
}

impl LodGroup {
    pub fn new(id: u64, center: Vec3, bounding_radius: f64) -> Self {
        Self { id, object_ids: Vec::new(), center, bounding_radius, current_level: 0 }
    }

    pub fn add_object(&mut self, object_id: u64) { self.object_ids.push(object_id); }
    pub fn remove_object(&mut self, object_id: u64) { self.object_ids.retain(|o| *o != object_id); }
}

// ── LodStats ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct LodStats {
    pub total_objects: usize,
    pub objects_per_level: Vec<usize>,
    pub billboard_count: usize,
    pub transitioning_count: usize,
    pub total_triangles: u64,
}

// ── Screen-size helper ───────────────────────────────────────

/// Calculate projected screen height in pixels for a sphere.
pub fn projected_screen_height(
    object_radius: f64,
    distance: f64,
    fov_y_rad: f64,
    screen_height_pixels: f64,
) -> f64 {
    if distance < 1e-9 { return screen_height_pixels; }
    let projected_size = object_radius / (distance * (fov_y_rad * 0.5).tan());
    projected_size * screen_height_pixels
}

// ── LodSystem ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LodSystem {
    pub configs: Vec<LodConfig>,
    pub objects: Vec<LodObject>,
    pub groups: Vec<LodGroup>,
}

impl LodSystem {
    pub fn new() -> Self {
        Self { configs: Vec::new(), objects: Vec::new(), groups: Vec::new() }
    }

    pub fn add_config(&mut self, config: LodConfig) -> usize {
        let idx = self.configs.len();
        self.configs.push(config);
        idx
    }

    pub fn add_object(&mut self, obj: LodObject) -> usize {
        let idx = self.objects.len();
        self.objects.push(obj);
        idx
    }

    pub fn add_group(&mut self, group: LodGroup) -> usize {
        let idx = self.groups.len();
        self.groups.push(group);
        idx
    }

    /// Update all objects based on camera position and elapsed time.
    pub fn update(&mut self, camera_pos: Vec3, dt: f64, fov_y_rad: f64, screen_height: f64) {
        for obj in self.objects.iter_mut() {
            if obj.config_index >= self.configs.len() { continue; }
            let config = &self.configs[obj.config_index];

            // Resolve override
            if obj.override_level.is_some() {
                let target = obj.override_level.unwrap();
                if target != obj.current_level {
                    start_transition(obj, target, config);
                }
                update_transition(obj, dt);
                continue;
            }

            let distance = obj.position.distance(camera_pos);

            // Select target level
            let target = if config.use_screen_size {
                let pixels = projected_screen_height(
                    obj.bounding_radius, distance, fov_y_rad, screen_height,
                );
                config.select_level_screen_size(pixels)
            } else {
                config.select_level_hysteresis(distance, obj.current_level)
            };

            if target != obj.current_level && obj.transition.as_ref().map_or(true, |t| t.is_complete()) {
                start_transition(obj, target, config);
            }

            update_transition(obj, dt);
        }
    }

    /// Compute statistics.
    pub fn stats(&self) -> LodStats {
        let mut objects_per_level = Vec::new();
        let mut billboard_count = 0;
        let mut transitioning_count = 0;
        let mut total_triangles = 0u64;

        for obj in &self.objects {
            let level = obj.effective_level();
            while objects_per_level.len() <= level {
                objects_per_level.push(0);
            }
            objects_per_level[level] += 1;

            if let Some(tr) = &obj.transition {
                if !tr.is_complete() { transitioning_count += 1; }
            }

            if obj.config_index < self.configs.len() {
                let config = &self.configs[obj.config_index];
                if level < config.levels.len() {
                    total_triangles += config.levels[level].triangle_count as u64;
                }
            }
        }

        // Count billboards separately
        for obj in &self.objects {
            if obj.config_index < self.configs.len() {
                let config = &self.configs[obj.config_index];
                if config.billboard_distance.is_some() {
                    // Billboard objects use the highest LOD index
                    let eff = obj.effective_level();
                    if eff == config.levels.len().saturating_sub(1) {
                        billboard_count += 1;
                    }
                }
            }
        }

        LodStats {
            total_objects: self.objects.len(),
            objects_per_level,
            billboard_count,
            transitioning_count,
            total_triangles,
        }
    }
}

fn start_transition(obj: &mut LodObject, target: usize, config: &LodConfig) {
    let tr = LodTransition::new(
        obj.current_level,
        target,
        config.transition,
        if config.transition == TransitionMode::Instant { 0.0 } else { config.transition_duration },
    );
    obj.transition = Some(tr);
}

fn update_transition(obj: &mut LodObject, dt: f64) {
    let complete = if let Some(tr) = &mut obj.transition {
        tr.update(dt);
        tr.is_complete()
    } else {
        false
    };
    if complete {
        if let Some(tr) = obj.transition.take() {
            obj.current_level = tr.to_level;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    fn make_config() -> LodConfig {
        LodConfig::new(vec![
            LodLevel::new(10.0, 1, 1000),
            LodLevel::new(30.0, 2, 500),
            LodLevel::new(100.0, 3, 100),
        ])
    }

    #[test]
    fn test_select_level_close() {
        let cfg = make_config();
        assert_eq!(cfg.select_level(5.0), 0);
    }

    #[test]
    fn test_select_level_medium() {
        let cfg = make_config();
        assert_eq!(cfg.select_level(20.0), 1);
    }

    #[test]
    fn test_select_level_far() {
        let cfg = make_config();
        assert_eq!(cfg.select_level(50.0), 2);
    }

    #[test]
    fn test_select_level_beyond() {
        let cfg = make_config();
        assert_eq!(cfg.select_level(500.0), 2);
    }

    #[test]
    fn test_bias_shifts_levels() {
        let cfg = make_config().with_bias(5.0);
        // 5 + 5 bias = 10, still LOD 0
        assert_eq!(cfg.select_level(5.0), 0);
        // 6 + 5 = 11, now LOD 1
        assert_eq!(cfg.select_level(6.0), 1);
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        let cfg = make_config();
        // At distance 10.5: beyond max_distance(10) but within switch_out(11)
        assert_eq!(cfg.select_level_hysteresis(10.5, 0), 0);
        // At distance 12: beyond switch_out(11)
        assert_eq!(cfg.select_level_hysteresis(12.0, 0), 1);
    }

    #[test]
    fn test_hysteresis_switch_back() {
        let cfg = make_config();
        // Currently at level 1, moving closer
        assert_eq!(cfg.select_level_hysteresis(9.0, 1), 0);
    }

    #[test]
    fn test_transition_instant() {
        let mut tr = LodTransition::new(0, 1, TransitionMode::Instant, 0.0);
        tr.update(0.0);
        assert!(tr.is_complete());
    }

    #[test]
    fn test_transition_crossfade() {
        let mut tr = LodTransition::new(0, 1, TransitionMode::CrossFade, 1.0);
        tr.update(0.5);
        assert!(!tr.is_complete());
        assert!((tr.progress - 0.5).abs() < 1e-9);
        assert!((tr.from_alpha() - 0.5).abs() < 1e-9);
        assert!((tr.to_alpha() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_transition_completes() {
        let mut tr = LodTransition::new(0, 1, TransitionMode::CrossFade, 1.0);
        tr.update(1.5);
        assert!(tr.is_complete());
        assert!((tr.progress - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_transition_current_level() {
        let mut tr = LodTransition::new(0, 1, TransitionMode::CrossFade, 1.0);
        tr.update(0.3);
        assert_eq!(tr.current_level(), 0); // < 50%
        tr.update(0.3);
        assert_eq!(tr.current_level(), 1); // >= 50%
    }

    #[test]
    fn test_object_override() {
        let mut obj = LodObject::new(1, Vec3::zero(), 1.0, 0);
        obj.set_override(Some(2));
        assert_eq!(obj.effective_level(), 2);
    }

    #[test]
    fn test_object_no_override() {
        let obj = LodObject::new(1, Vec3::zero(), 1.0, 0);
        assert_eq!(obj.effective_level(), 0);
    }

    #[test]
    fn test_lod_group() {
        let mut group = LodGroup::new(1, Vec3::zero(), 5.0);
        group.add_object(10);
        group.add_object(20);
        assert_eq!(group.object_ids.len(), 2);
        group.remove_object(10);
        assert_eq!(group.object_ids.len(), 1);
    }

    #[test]
    fn test_projected_screen_height() {
        let h = projected_screen_height(1.0, 10.0, FRAC_PI_4 * 2.0, 1080.0);
        assert!(h > 0.0);
        // Closer = larger
        let h_close = projected_screen_height(1.0, 5.0, FRAC_PI_4 * 2.0, 1080.0);
        assert!(h_close > h);
    }

    #[test]
    fn test_projected_screen_height_zero_distance() {
        let h = projected_screen_height(1.0, 0.0, FRAC_PI_4 * 2.0, 1080.0);
        assert!((h - 1080.0).abs() < 1e-9);
    }

    #[test]
    fn test_billboard_detection() {
        let cfg = make_config().with_billboard(50.0);
        assert!(!cfg.is_billboard(30.0));
        assert!(cfg.is_billboard(60.0));
    }

    #[test]
    fn test_system_update() {
        let mut sys = LodSystem::new();
        let cfg_idx = sys.add_config(make_config());
        sys.add_object(LodObject::new(1, Vec3::new(0.0, 0.0, 25.0), 1.0, cfg_idx));
        sys.update(Vec3::zero(), 0.1, FRAC_PI_4 * 2.0, 1080.0);
        assert_eq!(sys.objects[0].current_level, 1);
    }

    #[test]
    fn test_system_stats() {
        let mut sys = LodSystem::new();
        let cfg_idx = sys.add_config(make_config());
        sys.add_object(LodObject::new(1, Vec3::zero(), 1.0, cfg_idx));
        sys.add_object(LodObject::new(2, Vec3::new(0.0, 0.0, 50.0), 1.0, cfg_idx));
        sys.update(Vec3::zero(), 0.1, FRAC_PI_4 * 2.0, 1080.0);
        let stats = sys.stats();
        assert_eq!(stats.total_objects, 2);
    }

    #[test]
    fn test_screen_size_lod_selection() {
        let cfg = LodConfig::new(vec![
            LodLevel::new(10.0, 1, 1000).with_screen_pixels(200.0),
            LodLevel::new(30.0, 2, 500).with_screen_pixels(50.0),
            LodLevel::new(100.0, 3, 100).with_screen_pixels(10.0),
        ]).with_screen_size(true);
        // Large on screen -> highest detail matching
        assert_eq!(cfg.select_level_screen_size(300.0), 0);
        // Medium
        assert_eq!(cfg.select_level_screen_size(60.0), 1);
        // Small
        assert_eq!(cfg.select_level_screen_size(15.0), 2);
    }

    #[test]
    fn test_dithered_transition() {
        let mut tr = LodTransition::new(0, 1, TransitionMode::Dithered, 1.0);
        tr.update(0.5);
        assert!((tr.dither_threshold() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_empty_config_select() {
        let cfg = LodConfig::new(vec![]);
        // Should not panic on empty levels
        assert_eq!(cfg.select_level(10.0), 0);
    }
}
