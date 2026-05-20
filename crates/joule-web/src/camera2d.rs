//! 2D camera — follow, zoom, shake, dead zone, bounds clamping.
//!
//! Pure Rust replacement for Phaser Camera, PixiJS viewport, and similar
//! 2D camera systems. Fully headless with deterministic update.

// ── Vec2 ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn distance_to(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn lerp(self, target: Self, t: f64) -> Self {
        Self {
            x: self.x + (target.x - self.x) * t,
            y: self.y + (target.y - self.y) * t,
        }
    }
}

impl Default for Vec2 {
    fn default() -> Self {
        Self::zero()
    }
}

// ── Camera2D ─────────────────────────────────────────────────

/// 2D camera with position, zoom, rotation, and viewport.
#[derive(Debug, Clone)]
pub struct Camera2D {
    /// Camera center position in world space.
    pub position: Vec2,
    /// Zoom level (1.0 = normal, 2.0 = 2x zoom in).
    pub zoom: f64,
    /// Camera rotation in radians.
    pub rotation: f64,
    /// Viewport size in pixels.
    pub viewport_size: Vec2,

    // ── Follow ──
    follow_target: Option<Vec2>,
    follow_lerp: f64,

    // ── Bounds ──
    bounds: Option<CameraBounds>,

    // ── Shake ──
    shake_intensity: f64,
    shake_duration: f64,
    shake_elapsed: f64,
    shake_offset: Vec2,
    shake_seed: u64,

    // ── Dead zone ──
    dead_zone: Option<DeadZone>,
}

/// Camera world bounds — the camera won't scroll past these edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraBounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

/// Dead zone — camera only follows when target is outside this region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeadZone {
    pub width: f64,
    pub height: f64,
}

impl Camera2D {
    pub fn new(viewport_width: f64, viewport_height: f64) -> Self {
        Self {
            position: Vec2::zero(),
            zoom: 1.0,
            rotation: 0.0,
            viewport_size: Vec2::new(viewport_width, viewport_height),
            follow_target: None,
            follow_lerp: 1.0,
            bounds: None,
            shake_intensity: 0.0,
            shake_duration: 0.0,
            shake_elapsed: 0.0,
            shake_offset: Vec2::zero(),
            shake_seed: 12345,
            dead_zone: None,
        }
    }

    /// Set camera position directly.
    pub fn set_position(&mut self, x: f64, y: f64) {
        self.position = Vec2::new(x, y);
        self.clamp_to_bounds();
    }

    /// Set zoom level.
    pub fn set_zoom(&mut self, zoom: f64) {
        self.zoom = zoom.max(0.01);
    }

    /// Set rotation in radians.
    pub fn set_rotation(&mut self, angle: f64) {
        self.rotation = angle;
    }

    /// Start following a target with smooth interpolation.
    pub fn follow(&mut self, target: Vec2, lerp_factor: f64) {
        self.follow_target = Some(target);
        self.follow_lerp = lerp_factor.clamp(0.0, 1.0);
    }

    /// Stop following.
    pub fn unfollow(&mut self) {
        self.follow_target = None;
    }

    /// Set world bounds.
    pub fn set_bounds(&mut self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) {
        self.bounds = Some(CameraBounds { min_x, min_y, max_x, max_y });
        self.clamp_to_bounds();
    }

    /// Clear world bounds.
    pub fn clear_bounds(&mut self) {
        self.bounds = None;
    }

    /// Set dead zone size (centered on camera).
    pub fn set_dead_zone(&mut self, width: f64, height: f64) {
        self.dead_zone = Some(DeadZone { width, height });
    }

    /// Clear dead zone.
    pub fn clear_dead_zone(&mut self) {
        self.dead_zone = None;
    }

    /// Start a shake effect.
    pub fn shake(&mut self, intensity: f64, duration: f64) {
        self.shake_intensity = intensity;
        self.shake_duration = duration;
        self.shake_elapsed = 0.0;
    }

    /// Zoom to a specific point, keeping that world point under the cursor.
    pub fn zoom_to_point(&mut self, new_zoom: f64, screen_point: Vec2) {
        let world_before = self.screen_to_world(screen_point.x, screen_point.y);
        self.zoom = new_zoom.max(0.01);
        let world_after = self.screen_to_world(screen_point.x, screen_point.y);

        self.position.x += world_before.x - world_after.x;
        self.position.y += world_before.y - world_after.y;
        self.clamp_to_bounds();
    }

    /// Update the camera (call each frame with delta time).
    pub fn update(&mut self, dt: f64) {
        // Follow target
        if let Some(target) = self.follow_target {
            let mut desired = target;

            // Apply dead zone
            if let Some(dz) = self.dead_zone {
                let half_w = dz.width / 2.0;
                let half_h = dz.height / 2.0;
                let dx = target.x - self.position.x;
                let dy = target.y - self.position.y;

                if dx.abs() <= half_w {
                    desired.x = self.position.x;
                } else if dx > 0.0 {
                    desired.x = target.x - half_w;
                } else {
                    desired.x = target.x + half_w;
                }

                if dy.abs() <= half_h {
                    desired.y = self.position.y;
                } else if dy > 0.0 {
                    desired.y = target.y - half_h;
                } else {
                    desired.y = target.y + half_h;
                }
            }

            self.position = self.position.lerp(desired, self.follow_lerp);
        }

        // Update shake
        if self.shake_elapsed < self.shake_duration {
            self.shake_elapsed += dt;
            let progress = (self.shake_elapsed / self.shake_duration).min(1.0);
            let decay = 1.0 - progress;
            let magnitude = self.shake_intensity * decay;

            self.shake_seed = self.shake_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let rx = ((self.shake_seed >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
            self.shake_seed = self.shake_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let ry = ((self.shake_seed >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;

            self.shake_offset = Vec2::new(rx * magnitude, ry * magnitude);
        } else {
            self.shake_offset = Vec2::zero();
        }

        self.clamp_to_bounds();
    }

    /// Transform world coordinates to screen coordinates.
    pub fn world_to_screen(&self, world_x: f64, world_y: f64) -> Vec2 {
        let cx = self.position.x + self.shake_offset.x;
        let cy = self.position.y + self.shake_offset.y;

        let dx = world_x - cx;
        let dy = world_y - cy;

        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        let rx = dx * cos_r - dy * sin_r;
        let ry = dx * sin_r + dy * cos_r;

        Vec2::new(
            rx * self.zoom + self.viewport_size.x / 2.0,
            ry * self.zoom + self.viewport_size.y / 2.0,
        )
    }

    /// Transform screen coordinates to world coordinates.
    pub fn screen_to_world(&self, screen_x: f64, screen_y: f64) -> Vec2 {
        let cx = self.position.x + self.shake_offset.x;
        let cy = self.position.y + self.shake_offset.y;

        let rx = (screen_x - self.viewport_size.x / 2.0) / self.zoom;
        let ry = (screen_y - self.viewport_size.y / 2.0) / self.zoom;

        let cos_r = (-self.rotation).cos();
        let sin_r = (-self.rotation).sin();
        let dx = rx * cos_r - ry * sin_r;
        let dy = rx * sin_r + ry * cos_r;

        Vec2::new(cx + dx, cy + dy)
    }

    /// Get the visible world rectangle (AABB, ignoring rotation).
    pub fn visible_world_rect(&self) -> (f64, f64, f64, f64) {
        let half_w = self.viewport_size.x / (2.0 * self.zoom);
        let half_h = self.viewport_size.y / (2.0 * self.zoom);
        let cx = self.position.x + self.shake_offset.x;
        let cy = self.position.y + self.shake_offset.y;
        (cx - half_w, cy - half_h, half_w * 2.0, half_h * 2.0)
    }

    /// Whether the camera is currently shaking.
    pub fn is_shaking(&self) -> bool {
        self.shake_elapsed < self.shake_duration
    }

    fn clamp_to_bounds(&mut self) {
        if let Some(bounds) = self.bounds {
            let half_w = self.viewport_size.x / (2.0 * self.zoom);
            let half_h = self.viewport_size.y / (2.0 * self.zoom);

            let min_cam_x = bounds.min_x + half_w;
            let max_cam_x = bounds.max_x - half_w;
            let min_cam_y = bounds.min_y + half_h;
            let max_cam_y = bounds.max_y - half_h;

            if min_cam_x <= max_cam_x {
                self.position.x = self.position.x.clamp(min_cam_x, max_cam_x);
            } else {
                self.position.x = (bounds.min_x + bounds.max_x) / 2.0;
            }

            if min_cam_y <= max_cam_y {
                self.position.y = self.position.y.clamp(min_cam_y, max_cam_y);
            } else {
                self.position.y = (bounds.min_y + bounds.max_y) / 2.0;
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.01
    }

    #[test]
    fn default_camera() {
        let cam = Camera2D::new(800.0, 600.0);
        assert_eq!(cam.position, Vec2::zero());
        assert_eq!(cam.zoom, 1.0);
        assert_eq!(cam.rotation, 0.0);
    }

    #[test]
    fn world_to_screen_origin() {
        let cam = Camera2D::new(800.0, 600.0);
        let s = cam.world_to_screen(0.0, 0.0);
        assert!(approx(s.x, 400.0));
        assert!(approx(s.y, 300.0));
    }

    #[test]
    fn screen_to_world_center() {
        let cam = Camera2D::new(800.0, 600.0);
        let w = cam.screen_to_world(400.0, 300.0);
        assert!(approx(w.x, 0.0));
        assert!(approx(w.y, 0.0));
    }

    #[test]
    fn world_screen_roundtrip() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_position(100.0, 200.0);
        cam.set_zoom(1.5);

        let world = Vec2::new(150.0, 250.0);
        let screen = cam.world_to_screen(world.x, world.y);
        let back = cam.screen_to_world(screen.x, screen.y);
        assert!(approx(back.x, world.x));
        assert!(approx(back.y, world.y));
    }

    #[test]
    fn zoom_changes_scale() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_zoom(2.0);
        let s = cam.world_to_screen(100.0, 0.0);
        assert!(approx(s.x, 400.0 + 200.0));
    }

    #[test]
    fn follow_target_instant() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.follow(Vec2::new(100.0, 50.0), 1.0);
        cam.update(0.016);
        assert!(approx(cam.position.x, 100.0));
        assert!(approx(cam.position.y, 50.0));
    }

    #[test]
    fn follow_target_smooth() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.follow(Vec2::new(100.0, 0.0), 0.5);
        cam.update(0.016);
        assert!(approx(cam.position.x, 50.0));
        assert!(approx(cam.position.y, 0.0));
    }

    #[test]
    fn bounds_clamping() {
        let mut cam = Camera2D::new(200.0, 200.0);
        cam.set_bounds(0.0, 0.0, 500.0, 500.0);

        cam.set_position(-100.0, -100.0);
        assert!(approx(cam.position.x, 100.0));
        assert!(approx(cam.position.y, 100.0));

        cam.set_position(600.0, 600.0);
        assert!(approx(cam.position.x, 400.0));
        assert!(approx(cam.position.y, 400.0));
    }

    #[test]
    fn bounds_viewport_larger_than_world() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_bounds(0.0, 0.0, 100.0, 100.0);
        cam.set_position(0.0, 0.0);
        assert!(approx(cam.position.x, 50.0));
        assert!(approx(cam.position.y, 50.0));
    }

    #[test]
    fn shake_decays() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.shake(10.0, 1.0);
        assert!(cam.is_shaking());

        cam.update(0.5);
        cam.update(0.6);
        assert!(!cam.is_shaking());
        assert!(approx(cam.shake_offset.x, 0.0));
        assert!(approx(cam.shake_offset.y, 0.0));
    }

    #[test]
    fn zoom_to_point_preserves_world_point() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_position(100.0, 100.0);

        let screen_point = Vec2::new(600.0, 400.0);
        let world_before = cam.screen_to_world(screen_point.x, screen_point.y);

        cam.zoom_to_point(2.0, screen_point);

        let world_after = cam.screen_to_world(screen_point.x, screen_point.y);
        assert!(approx(world_before.x, world_after.x));
        assert!(approx(world_before.y, world_after.y));
    }

    #[test]
    fn dead_zone_no_movement_inside() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_dead_zone(100.0, 100.0);
        cam.set_position(0.0, 0.0);

        cam.follow(Vec2::new(30.0, 30.0), 1.0);
        cam.update(0.016);
        assert!(approx(cam.position.x, 0.0));
        assert!(approx(cam.position.y, 0.0));
    }

    #[test]
    fn dead_zone_movement_outside() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_dead_zone(100.0, 100.0);
        cam.set_position(0.0, 0.0);

        cam.follow(Vec2::new(200.0, 0.0), 1.0);
        cam.update(0.016);
        assert!(approx(cam.position.x, 150.0));
    }

    #[test]
    fn visible_world_rect() {
        let cam = Camera2D::new(800.0, 600.0);
        let (x, y, w, h) = cam.visible_world_rect();
        assert!(approx(x, -400.0));
        assert!(approx(y, -300.0));
        assert!(approx(w, 800.0));
        assert!(approx(h, 600.0));
    }

    #[test]
    fn visible_world_rect_zoomed() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_zoom(2.0);
        let (_x, _y, w, h) = cam.visible_world_rect();
        assert!(approx(w, 400.0));
        assert!(approx(h, 300.0));
    }

    #[test]
    fn unfollow() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_position(50.0, 50.0);
        cam.follow(Vec2::new(200.0, 200.0), 1.0);
        cam.unfollow();
        cam.update(0.016);
        assert!(approx(cam.position.x, 50.0));
        assert!(approx(cam.position.y, 50.0));
    }

    #[test]
    fn rotation_world_to_screen() {
        let mut cam = Camera2D::new(800.0, 600.0);
        cam.set_rotation(std::f64::consts::FRAC_PI_2);
        let s = cam.world_to_screen(100.0, 0.0);
        assert!(approx(s.x, 400.0));
        assert!(approx(s.y, 400.0));
    }
}
