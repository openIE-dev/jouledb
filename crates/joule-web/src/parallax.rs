//! Parallax scrolling effects engine.
//!
//! Replaces Rellax.js / Locomotive Scroll parallax features. Supports
//! scroll-driven and mouse-driven parallax with configurable speed
//! factors, depth sorting, and smooth interpolation.

use std::fmt;

// ── Point ──────────────────────────────────────────────────────

/// A 2D offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Offset {
    pub x: f64,
    pub y: f64,
}

impl Offset {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    /// Linear interpolation toward target.
    pub fn lerp(self, target: Offset, t: f64) -> Offset {
        let t = t.clamp(0.0, 1.0);
        Offset {
            x: self.x + (target.x - self.x) * t,
            y: self.y + (target.y - self.y) * t,
        }
    }
}

impl fmt::Display for Offset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2})", self.x, self.y)
    }
}

// ── Parallax Direction ─────────────────────────────────────────

/// Axis along which parallax is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallaxAxis {
    Vertical,
    Horizontal,
    Both,
}

// ── Layer ──────────────────────────────────────────────────────

/// A single parallax layer.
#[derive(Debug, Clone)]
pub struct ParallaxLayer {
    /// Identifier for the element this layer controls.
    pub element_id: String,
    /// Speed factor: 1.0 = normal scroll speed, 0.5 = half, 2.0 = double.
    /// Negative values scroll in the opposite direction.
    pub speed_factor: f64,
    /// Static offset applied before parallax.
    pub base_offset: Offset,
    /// Depth value for sorting (higher = further back / slower).
    pub depth: f64,
    /// Which axis to apply parallax on.
    pub axis: ParallaxAxis,
    /// Current smoothed offset.
    current_offset: Offset,
    /// Raw target offset (before smoothing).
    target_offset: Offset,
}

impl ParallaxLayer {
    pub fn new(element_id: impl Into<String>, speed_factor: f64) -> Self {
        Self {
            element_id: element_id.into(),
            speed_factor,
            base_offset: Offset::zero(),
            depth: speed_factor.abs(),
            axis: ParallaxAxis::Vertical,
            current_offset: Offset::zero(),
            target_offset: Offset::zero(),
        }
    }

    pub fn with_base_offset(mut self, offset: Offset) -> Self {
        self.base_offset = offset;
        self
    }

    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth = depth;
        self
    }

    pub fn with_axis(mut self, axis: ParallaxAxis) -> Self {
        self.axis = axis;
        self
    }

    /// Get the current (smoothed) offset.
    pub fn current_offset(&self) -> Offset {
        Offset {
            x: self.base_offset.x + self.current_offset.x,
            y: self.base_offset.y + self.current_offset.y,
        }
    }

    /// Get the raw target offset.
    pub fn target_offset(&self) -> Offset {
        self.target_offset
    }
}

// ── Scene ──────────────────────────────────────────────────────

/// Origin for mouse parallax calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallaxOrigin {
    pub x: f64,
    pub y: f64,
}

impl ParallaxOrigin {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Center of viewport.
    pub fn center(viewport_width: f64, viewport_height: f64) -> Self {
        Self { x: viewport_width / 2.0, y: viewport_height / 2.0 }
    }
}

/// A parallax scene containing multiple layers.
#[derive(Debug, Clone)]
pub struct ParallaxScene {
    layers: Vec<ParallaxLayer>,
    /// Smoothing factor for lerp (0 = instant, 1 = no movement). Applied per tick.
    smooth_factor: f64,
    /// Origin for mouse-based parallax.
    origin: ParallaxOrigin,
    /// Mouse sensitivity multiplier.
    mouse_sensitivity: f64,
}

impl ParallaxScene {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            smooth_factor: 0.1,
            origin: ParallaxOrigin::new(0.0, 0.0),
            mouse_sensitivity: 1.0,
        }
    }

    pub fn with_smooth_factor(mut self, factor: f64) -> Self {
        self.smooth_factor = factor.clamp(0.0, 1.0);
        self
    }

    pub fn with_origin(mut self, origin: ParallaxOrigin) -> Self {
        self.origin = origin;
        self
    }

    pub fn with_mouse_sensitivity(mut self, sensitivity: f64) -> Self {
        self.mouse_sensitivity = sensitivity;
        self
    }

    /// Add a layer.
    pub fn add_layer(&mut self, layer: ParallaxLayer) {
        self.layers.push(layer);
    }

    /// Get layers.
    pub fn layers(&self) -> &[ParallaxLayer] {
        &self.layers
    }

    /// Get layers sorted by depth (front to back).
    pub fn layers_sorted_by_depth(&self) -> Vec<&ParallaxLayer> {
        let mut sorted: Vec<&ParallaxLayer> = self.layers.iter().collect();
        sorted.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
        sorted
    }

    /// Number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Find a layer by element ID.
    pub fn layer_by_id(&self, id: &str) -> Option<&ParallaxLayer> {
        self.layers.iter().find(|l| l.element_id == id)
    }

    /// Update all layers based on scroll position.
    pub fn update_scroll(&mut self, scroll_x: f64, scroll_y: f64) {
        for layer in &mut self.layers {
            let tx = match layer.axis {
                ParallaxAxis::Horizontal | ParallaxAxis::Both => scroll_x * layer.speed_factor,
                ParallaxAxis::Vertical => 0.0,
            };
            let ty = match layer.axis {
                ParallaxAxis::Vertical | ParallaxAxis::Both => scroll_y * layer.speed_factor,
                ParallaxAxis::Horizontal => 0.0,
            };
            layer.target_offset = Offset::new(tx, ty);
        }
    }

    /// Update all layers based on mouse/cursor position.
    pub fn update_mouse(&mut self, cursor_x: f64, cursor_y: f64) {
        let dx = (cursor_x - self.origin.x) * self.mouse_sensitivity;
        let dy = (cursor_y - self.origin.y) * self.mouse_sensitivity;

        for layer in &mut self.layers {
            let tx = match layer.axis {
                ParallaxAxis::Horizontal | ParallaxAxis::Both => dx * layer.speed_factor,
                ParallaxAxis::Vertical => 0.0,
            };
            let ty = match layer.axis {
                ParallaxAxis::Vertical | ParallaxAxis::Both => dy * layer.speed_factor,
                ParallaxAxis::Horizontal => 0.0,
            };
            layer.target_offset = Offset::new(tx, ty);
        }
    }

    /// Tick: apply smoothing interpolation toward target offsets.
    /// Call this once per frame.
    pub fn tick(&mut self) {
        let t = self.smooth_factor;
        for layer in &mut self.layers {
            layer.current_offset = layer.current_offset.lerp(layer.target_offset, t);
        }
    }

    /// Immediately snap all layers to their target (no smoothing).
    pub fn snap_to_target(&mut self) {
        for layer in &mut self.layers {
            layer.current_offset = layer.target_offset;
        }
    }

    /// Get computed offsets for all layers (element_id, offset).
    pub fn computed_offsets(&self) -> Vec<(&str, Offset)> {
        self.layers.iter()
            .map(|l| (l.element_id.as_str(), l.current_offset()))
            .collect()
    }
}

impl Default for ParallaxScene {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_layer_scene() -> ParallaxScene {
        let mut scene = ParallaxScene::new().with_smooth_factor(1.0); // instant
        scene.add_layer(ParallaxLayer::new("bg", 0.5));
        scene.add_layer(ParallaxLayer::new("fg", 1.5));
        scene
    }

    #[test]
    fn scroll_vertical_offsets() {
        let mut scene = two_layer_scene();
        scene.update_scroll(0.0, 100.0);
        scene.tick();
        let offsets = scene.computed_offsets();
        assert!((offsets[0].1.y - 50.0).abs() < 0.01); // bg at 0.5x
        assert!((offsets[1].1.y - 150.0).abs() < 0.01); // fg at 1.5x
    }

    #[test]
    fn horizontal_parallax() {
        let mut scene = ParallaxScene::new().with_smooth_factor(1.0);
        scene.add_layer(ParallaxLayer::new("h", 0.3).with_axis(ParallaxAxis::Horizontal));
        scene.update_scroll(200.0, 100.0);
        scene.tick();
        let off = scene.layer_by_id("h").unwrap().current_offset();
        assert!((off.x - 60.0).abs() < 0.01); // 200 * 0.3
        assert!((off.y - 0.0).abs() < 0.01); // vertical ignored
    }

    #[test]
    fn both_axes() {
        let mut scene = ParallaxScene::new().with_smooth_factor(1.0);
        scene.add_layer(ParallaxLayer::new("both", 0.5).with_axis(ParallaxAxis::Both));
        scene.update_scroll(100.0, 200.0);
        scene.tick();
        let off = scene.layer_by_id("both").unwrap().current_offset();
        assert!((off.x - 50.0).abs() < 0.01);
        assert!((off.y - 100.0).abs() < 0.01);
    }

    #[test]
    fn depth_sorting() {
        let mut scene = ParallaxScene::new();
        scene.add_layer(ParallaxLayer::new("mid", 1.0).with_depth(5.0));
        scene.add_layer(ParallaxLayer::new("far", 0.2).with_depth(10.0));
        scene.add_layer(ParallaxLayer::new("near", 2.0).with_depth(1.0));

        let sorted = scene.layers_sorted_by_depth();
        assert_eq!(sorted[0].element_id, "near");
        assert_eq!(sorted[1].element_id, "mid");
        assert_eq!(sorted[2].element_id, "far");
    }

    #[test]
    fn mouse_parallax() {
        let mut scene = ParallaxScene::new()
            .with_smooth_factor(1.0)
            .with_origin(ParallaxOrigin::center(800.0, 600.0))
            .with_mouse_sensitivity(0.05);

        scene.add_layer(ParallaxLayer::new("layer", 1.0).with_axis(ParallaxAxis::Both));
        // Cursor at center → no movement.
        scene.update_mouse(400.0, 300.0);
        scene.tick();
        let off = scene.layer_by_id("layer").unwrap().current_offset();
        assert!((off.x).abs() < 0.01);
        assert!((off.y).abs() < 0.01);

        // Cursor offset from center.
        scene.update_mouse(600.0, 400.0);
        scene.tick();
        let off = scene.layer_by_id("layer").unwrap().current_offset();
        // dx = (600-400)*0.05*1.0 = 10, dy = (400-300)*0.05*1.0 = 5
        assert!((off.x - 10.0).abs() < 0.01);
        assert!((off.y - 5.0).abs() < 0.01);
    }

    #[test]
    fn smooth_interpolation() {
        let mut scene = ParallaxScene::new().with_smooth_factor(0.5);
        scene.add_layer(ParallaxLayer::new("smooth", 1.0));
        scene.update_scroll(0.0, 100.0);

        // First tick: lerp from 0 to 100 at t=0.5 → 50.
        scene.tick();
        let off = scene.layer_by_id("smooth").unwrap().current_offset();
        assert!((off.y - 50.0).abs() < 0.01);

        // Second tick: lerp from 50 to 100 at t=0.5 → 75.
        scene.tick();
        let off = scene.layer_by_id("smooth").unwrap().current_offset();
        assert!((off.y - 75.0).abs() < 0.01);
    }

    #[test]
    fn snap_to_target() {
        let mut scene = ParallaxScene::new().with_smooth_factor(0.1);
        scene.add_layer(ParallaxLayer::new("snap", 1.0));
        scene.update_scroll(0.0, 200.0);
        scene.snap_to_target();
        let off = scene.layer_by_id("snap").unwrap().current_offset();
        assert!((off.y - 200.0).abs() < 0.01);
    }

    #[test]
    fn base_offset() {
        let mut scene = ParallaxScene::new().with_smooth_factor(1.0);
        scene.add_layer(
            ParallaxLayer::new("shifted", 1.0)
                .with_base_offset(Offset::new(10.0, 20.0)),
        );
        scene.update_scroll(0.0, 50.0);
        scene.tick();
        let off = scene.layer_by_id("shifted").unwrap().current_offset();
        assert!((off.x - 10.0).abs() < 0.01);
        assert!((off.y - 70.0).abs() < 0.01); // 20 + 50
    }

    #[test]
    fn negative_speed_reverses() {
        let mut scene = ParallaxScene::new().with_smooth_factor(1.0);
        scene.add_layer(ParallaxLayer::new("rev", -0.5));
        scene.update_scroll(0.0, 100.0);
        scene.tick();
        let off = scene.layer_by_id("rev").unwrap().current_offset();
        assert!((off.y - (-50.0)).abs() < 0.01);
    }

    #[test]
    fn layer_count() {
        let scene = two_layer_scene();
        assert_eq!(scene.layer_count(), 2);
    }

    #[test]
    fn offset_display() {
        let o = Offset::new(1.5, -3.25);
        assert_eq!(format!("{o}"), "(1.50, -3.25)");
    }
}
