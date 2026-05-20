//! Parallax scrolling background system: layered backgrounds that move at
//! different speeds relative to the camera, creating a depth illusion.
//!
//! Supports infinite tiling, auto-scroll, and depth-based speed calculation.

// ── Camera ─────────────────────────────────────────────────────

/// Minimal camera state needed for parallax calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    /// Camera center in world coordinates.
    pub x: f64,
    pub y: f64,
    /// Viewport size in pixels.
    pub viewport_w: f64,
    pub viewport_h: f64,
}

impl Camera2D {
    pub fn new(x: f64, y: f64, vw: f64, vh: f64) -> Self {
        Self { x, y, viewport_w: vw, viewport_h: vh }
    }

    /// Top-left corner of what the camera sees.
    pub fn left(&self) -> f64 {
        self.x - self.viewport_w / 2.0
    }

    pub fn top(&self) -> f64 {
        self.y - self.viewport_h / 2.0
    }
}

// ── Layer Configuration ────────────────────────────────────────

/// How a layer tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileMode {
    /// No tiling — image is drawn once.
    None,
    /// Tile along X only.
    TileX,
    /// Tile along Y only.
    TileY,
    /// Tile in both axes.
    TileXY,
}

/// A single parallax background layer.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallaxLayer {
    /// Unique identifier.
    pub id: String,
    /// Image/texture width in pixels (for tiling calculations).
    pub image_width: f64,
    /// Image/texture height in pixels.
    pub image_height: f64,
    /// Scroll speed multiplier along X. 0.0 = static, 1.0 = moves with camera.
    pub speed_x: f64,
    /// Scroll speed multiplier along Y.
    pub speed_y: f64,
    /// Auto-scroll speed in pixels/sec along X.
    pub auto_scroll_x: f64,
    /// Auto-scroll speed in pixels/sec along Y.
    pub auto_scroll_y: f64,
    /// Tiling mode.
    pub tile_mode: TileMode,
    /// Ordering depth (lower = further back, drawn first).
    pub depth: i32,
    /// Opacity (0.0 – 1.0).
    pub opacity: f64,
    /// Whether the layer is visible.
    pub visible: bool,
    /// Optional fixed offset in pixels.
    pub offset_x: f64,
    pub offset_y: f64,
}

impl ParallaxLayer {
    pub fn new(id: &str, img_w: f64, img_h: f64) -> Self {
        Self {
            id: id.to_string(),
            image_width: img_w,
            image_height: img_h,
            speed_x: 1.0,
            speed_y: 1.0,
            auto_scroll_x: 0.0,
            auto_scroll_y: 0.0,
            tile_mode: TileMode::None,
            depth: 0,
            opacity: 1.0,
            visible: true,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }

    /// Builder: set speed multipliers.
    pub fn with_speed(mut self, sx: f64, sy: f64) -> Self {
        self.speed_x = sx;
        self.speed_y = sy;
        self
    }

    pub fn with_auto_scroll(mut self, ax: f64, ay: f64) -> Self {
        self.auto_scroll_x = ax;
        self.auto_scroll_y = ay;
        self
    }

    pub fn with_tile_mode(mut self, mode: TileMode) -> Self {
        self.tile_mode = mode;
        self
    }

    pub fn with_depth(mut self, depth: i32) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_offset(mut self, ox: f64, oy: f64) -> Self {
        self.offset_x = ox;
        self.offset_y = oy;
        self
    }
}

// ── Rendered Layer Output ──────────────────────────────────────

/// Where to draw a layer image instance (there may be multiple if tiling).
#[derive(Debug, Clone, PartialEq)]
pub struct LayerDrawCmd {
    pub layer_id: String,
    pub screen_x: f64,
    pub screen_y: f64,
    pub opacity: f64,
    pub depth: i32,
}

// ── Parallax System ────────────────────────────────────────────

/// The parallax scroll engine.
#[derive(Debug, Clone)]
pub struct ParallaxSystem {
    layers: Vec<ParallaxLayer>,
    /// Accumulated time for auto-scroll.
    time_secs: f64,
}

impl ParallaxSystem {
    pub fn new() -> Self {
        Self { layers: Vec::new(), time_secs: 0.0 }
    }

    pub fn add_layer(&mut self, layer: ParallaxLayer) {
        self.layers.push(layer);
        self.layers.sort_by_key(|l| l.depth);
    }

    pub fn remove_layer(&mut self, id: &str) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.id != id);
        self.layers.len() < before
    }

    pub fn layer(&self, id: &str) -> Option<&ParallaxLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn layer_mut(&mut self, id: &str) -> Option<&mut ParallaxLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Advance time (for auto-scroll).
    pub fn update(&mut self, dt: f64) {
        self.time_secs += dt;
    }

    /// Calculate depth-based speed: layers closer to 0 depth scroll slower.
    /// `near_depth`/`far_depth` define the range; `near_speed`/`far_speed`
    /// define the corresponding scroll multipliers.
    pub fn depth_speed(depth: i32, far_depth: i32, near_depth: i32, far_speed: f64, near_speed: f64) -> f64 {
        if near_depth == far_depth {
            return near_speed;
        }
        let t = (depth - far_depth) as f64 / (near_depth - far_depth) as f64;
        let t = t.clamp(0.0, 1.0);
        far_speed + t * (near_speed - far_speed)
    }

    /// Generate draw commands for all visible layers given the camera state.
    pub fn render(&self, camera: &Camera2D) -> Vec<LayerDrawCmd> {
        let mut cmds = Vec::new();

        for layer in &self.layers {
            if !layer.visible || layer.opacity <= 0.0 {
                continue;
            }
            if layer.image_width <= 0.0 || layer.image_height <= 0.0 {
                continue;
            }

            // Parallax offset = camera position * speed multiplier
            let px = camera.left() * layer.speed_x
                + layer.auto_scroll_x * self.time_secs
                + layer.offset_x;
            let py = camera.top() * layer.speed_y
                + layer.auto_scroll_y * self.time_secs
                + layer.offset_y;

            let tiles_x = self.tiling_needs_x(layer, camera.viewport_w, px);
            let tiles_y = self.tiling_needs_y(layer, camera.viewport_h, py);

            for ty in &tiles_y {
                for tx in &tiles_x {
                    cmds.push(LayerDrawCmd {
                        layer_id: layer.id.clone(),
                        screen_x: *tx,
                        screen_y: *ty,
                        opacity: layer.opacity,
                        depth: layer.depth,
                    });
                }
            }
        }

        cmds
    }

    /// Compute the set of X positions needed to tile a layer across the viewport.
    fn tiling_needs_x(&self, layer: &ParallaxLayer, viewport_w: f64, parallax_x: f64) -> Vec<f64> {
        match layer.tile_mode {
            TileMode::TileX | TileMode::TileXY => {
                Self::tile_positions(parallax_x, layer.image_width, viewport_w)
            }
            _ => vec![-parallax_x],
        }
    }

    fn tiling_needs_y(&self, layer: &ParallaxLayer, viewport_h: f64, parallax_y: f64) -> Vec<f64> {
        match layer.tile_mode {
            TileMode::TileY | TileMode::TileXY => {
                Self::tile_positions(parallax_y, layer.image_height, viewport_h)
            }
            _ => vec![-parallax_y],
        }
    }

    /// Generate tile positions along one axis.
    /// `offset` is the parallax offset, `tile_size` is the image dimension,
    /// `viewport` is the visible size.
    fn tile_positions(offset: f64, tile_size: f64, viewport: f64) -> Vec<f64> {
        if tile_size <= 0.0 {
            return vec![0.0];
        }
        let mut positions = Vec::new();
        // The first tile X in screen space, adjusted for parallax wrapping
        let base = -(offset % tile_size);
        // Ensure base is within (-tile_size, 0]
        let base = if base > 0.0 { base - tile_size } else { base };

        let mut x = base;
        while x < viewport + tile_size {
            positions.push(x);
            x += tile_size;
        }
        positions
    }

    /// Get layers ordered back-to-front (ascending depth).
    pub fn layers_ordered(&self) -> &[ParallaxLayer] {
        &self.layers // already sorted
    }

    /// Set all layer speeds based on their depth using linear interpolation.
    pub fn auto_depth_speeds(&mut self, far_depth: i32, near_depth: i32, far_speed: f64, near_speed: f64) {
        for layer in &mut self.layers {
            let s = Self::depth_speed(layer.depth, far_depth, near_depth, far_speed, near_speed);
            layer.speed_x = s;
            layer.speed_y = s;
        }
    }
}

impl Default for ParallaxSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn cam() -> Camera2D {
        Camera2D::new(400.0, 300.0, 800.0, 600.0)
    }

    fn bg_layer(id: &str, speed: f64, depth: i32) -> ParallaxLayer {
        ParallaxLayer::new(id, 800.0, 600.0)
            .with_speed(speed, speed)
            .with_depth(depth)
    }

    #[test]
    fn camera_left_top() {
        let c = cam();
        assert!((c.left() - 0.0).abs() < EPS);
        assert!((c.top() - 0.0).abs() < EPS);
    }

    #[test]
    fn add_layer_sorts_by_depth() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("mid", 0.5, 5));
        sys.add_layer(bg_layer("far", 0.2, 0));
        sys.add_layer(bg_layer("near", 0.8, 10));
        assert_eq!(sys.layers_ordered()[0].id, "far");
        assert_eq!(sys.layers_ordered()[1].id, "mid");
        assert_eq!(sys.layers_ordered()[2].id, "near");
    }

    #[test]
    fn remove_layer() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("a", 0.5, 0));
        assert!(sys.remove_layer("a"));
        assert_eq!(sys.layer_count(), 0);
    }

    #[test]
    fn remove_nonexistent() {
        let mut sys = ParallaxSystem::new();
        assert!(!sys.remove_layer("ghost"));
    }

    #[test]
    fn static_layer_no_movement() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("bg", 0.0, 0));
        let camera = Camera2D::new(1000.0, 500.0, 800.0, 600.0);
        let cmds = sys.render(&camera);
        assert!(!cmds.is_empty());
        // Static layer: screen_x should be 0 (offset=0, speed=0)
        assert!((cmds[0].screen_x - 0.0).abs() < EPS);
    }

    #[test]
    fn full_speed_layer_moves_with_camera() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("fg", 1.0, 10));
        let camera = Camera2D::new(500.0, 350.0, 800.0, 600.0);
        let cmds = sys.render(&camera);
        assert!(!cmds.is_empty());
        // screen_x = -(camera.left * 1.0) = -(100)
        assert!((cmds[0].screen_x - (-100.0)).abs() < EPS);
    }

    #[test]
    fn tiling_x_produces_multiple_draws() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(
            ParallaxLayer::new("tiled", 256.0, 600.0)
                .with_speed(0.5, 0.0)
                .with_tile_mode(TileMode::TileX)
                .with_depth(0),
        );
        let camera = Camera2D::new(400.0, 300.0, 800.0, 600.0);
        let cmds = sys.render(&camera);
        // 800/256 ≈ 3.1, so need at least 4 tiles
        assert!(cmds.len() >= 4);
    }

    #[test]
    fn tiling_xy_tiles_both_axes() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(
            ParallaxLayer::new("tiled", 200.0, 200.0)
                .with_speed(0.0, 0.0)
                .with_tile_mode(TileMode::TileXY)
                .with_depth(0),
        );
        let camera = Camera2D::new(400.0, 300.0, 800.0, 600.0);
        let cmds = sys.render(&camera);
        // X: ceil(800/200)+1 = 5, Y: ceil(600/200)+1 = 4 → 20 tiles
        assert!(cmds.len() >= 16);
    }

    #[test]
    fn hidden_layer_not_rendered() {
        let mut sys = ParallaxSystem::new();
        let mut layer = bg_layer("hidden", 0.5, 0);
        layer.visible = false;
        sys.add_layer(layer);
        let cmds = sys.render(&cam());
        assert!(cmds.is_empty());
    }

    #[test]
    fn zero_opacity_not_rendered() {
        let mut sys = ParallaxSystem::new();
        let mut layer = bg_layer("transparent", 0.5, 0);
        layer.opacity = 0.0;
        sys.add_layer(layer);
        let cmds = sys.render(&cam());
        assert!(cmds.is_empty());
    }

    #[test]
    fn auto_scroll_advances_with_time() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(
            ParallaxLayer::new("clouds", 800.0, 600.0)
                .with_speed(0.0, 0.0)
                .with_auto_scroll(50.0, 0.0)
                .with_depth(0),
        );
        let camera = Camera2D::new(0.0, 0.0, 800.0, 600.0);
        let cmds1 = sys.render(&camera);
        sys.update(2.0);
        let cmds2 = sys.render(&camera);
        // After 2s at 50px/s, should have moved 100px
        let diff = cmds1[0].screen_x - cmds2[0].screen_x;
        assert!((diff - 100.0).abs() < EPS);
    }

    #[test]
    fn depth_speed_interpolation() {
        let s = ParallaxSystem::depth_speed(5, 0, 10, 0.1, 1.0);
        assert!((s - 0.55).abs() < EPS);
    }

    #[test]
    fn depth_speed_at_extremes() {
        let far = ParallaxSystem::depth_speed(0, 0, 10, 0.1, 1.0);
        assert!((far - 0.1).abs() < EPS);
        let near = ParallaxSystem::depth_speed(10, 0, 10, 0.1, 1.0);
        assert!((near - 1.0).abs() < EPS);
    }

    #[test]
    fn depth_speed_clamped_beyond_range() {
        let beyond = ParallaxSystem::depth_speed(20, 0, 10, 0.1, 1.0);
        assert!((beyond - 1.0).abs() < EPS);
    }

    #[test]
    fn auto_depth_speeds() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("far", 0.0, 0));
        sys.add_layer(bg_layer("near", 0.0, 10));
        sys.auto_depth_speeds(0, 10, 0.1, 1.0);
        assert!((sys.layer("far").unwrap().speed_x - 0.1).abs() < EPS);
        assert!((sys.layer("near").unwrap().speed_x - 1.0).abs() < EPS);
    }

    #[test]
    fn layer_offset() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(
            ParallaxLayer::new("off", 800.0, 600.0)
                .with_speed(0.0, 0.0)
                .with_offset(50.0, 30.0)
                .with_depth(0),
        );
        let camera = Camera2D::new(0.0, 0.0, 800.0, 600.0);
        let cmds = sys.render(&camera);
        // Offset subtracts from screen pos: screen_x = -(offset)
        // Actually: px = cam_left*speed + auto*time + offset = 0 + 0 + 50 = 50
        // screen_x = -px = -50
        assert!((cmds[0].screen_x - (-50.0)).abs() < EPS);
        assert!((cmds[0].screen_y - (-30.0)).abs() < EPS);
    }

    #[test]
    fn layer_mut_access() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("edit", 0.5, 0));
        sys.layer_mut("edit").unwrap().opacity = 0.5;
        assert!((sys.layer("edit").unwrap().opacity - 0.5).abs() < EPS);
    }

    #[test]
    fn render_preserves_depth_order() {
        let mut sys = ParallaxSystem::new();
        sys.add_layer(bg_layer("a", 0.2, 0));
        sys.add_layer(bg_layer("b", 0.5, 5));
        let cmds = sys.render(&cam());
        assert!(cmds.len() >= 2);
        assert!(cmds[0].depth <= cmds[1].depth);
    }

    #[test]
    fn tile_positions_cover_viewport() {
        let positions = ParallaxSystem::tile_positions(0.0, 200.0, 800.0);
        let min = positions.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = positions.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min <= 0.0);
        assert!(max >= 600.0); // last tile starts at 600, covers to 800
    }
}
