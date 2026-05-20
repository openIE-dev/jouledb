//! HUD (heads-up display) layout system: anchor-based positioning,
//! responsive scaling, element visibility, animation (pulse, slide),
//! and safe-area insets for notched displays.
//!
//! Pure layout math — outputs screen-space rectangles for the renderer.

use std::collections::HashMap;

// ── Anchor ─────────────────────────────────────────────────────

/// Screen anchor point for positioning HUD elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Anchor {
    /// Returns the (x_factor, y_factor) where 0.0 = left/top, 1.0 = right/bottom.
    fn factors(&self) -> (f64, f64) {
        match self {
            Anchor::TopLeft => (0.0, 0.0),
            Anchor::TopCenter => (0.5, 0.0),
            Anchor::TopRight => (1.0, 0.0),
            Anchor::CenterLeft => (0.0, 0.5),
            Anchor::Center => (0.5, 0.5),
            Anchor::CenterRight => (1.0, 0.5),
            Anchor::BottomLeft => (0.0, 1.0),
            Anchor::BottomCenter => (0.5, 1.0),
            Anchor::BottomRight => (1.0, 1.0),
        }
    }
}

// ── Safe Area Insets ───────────────────────────────────────────

/// Insets for safe area (notch, rounded corners, system bars).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafeAreaInsets {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

impl SafeAreaInsets {
    pub fn zero() -> Self {
        Self { top: 0.0, bottom: 0.0, left: 0.0, right: 0.0 }
    }

    pub fn new(top: f64, bottom: f64, left: f64, right: f64) -> Self {
        Self { top, bottom, left, right }
    }
}

impl Default for SafeAreaInsets {
    fn default() -> Self {
        Self::zero()
    }
}

// ── Screen Rect ────────────────────────────────────────────────

/// A positioned rectangle in screen space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl ScreenRect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, width: w, height: h }
    }

    pub fn center_x(&self) -> f64 {
        self.x + self.width / 2.0
    }

    pub fn center_y(&self) -> f64 {
        self.y + self.height / 2.0
    }

    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px <= self.right() && py >= self.y && py <= self.bottom()
    }
}

// ── HUD Element Kinds ──────────────────────────────────────────

/// Type of HUD element for semantic rendering.
#[derive(Debug, Clone, PartialEq)]
pub enum HudElementKind {
    /// A bar (health, mana, stamina, etc.).
    Bar {
        current: f64,
        max: f64,
        color_r: f64,
        color_g: f64,
        color_b: f64,
    },
    /// Score or numeric counter.
    Counter {
        value: i64,
        label: String,
    },
    /// Ammo display (current / max).
    Ammo {
        current: u32,
        magazine: u32,
        reserve: u32,
    },
    /// Minimap frame (just a placeholder rect — actual minimap rendering
    /// is in the `minimap` module).
    MinimapFrame,
    /// Crosshair / reticle.
    Crosshair {
        size: f64,
        gap: f64,
        thickness: f64,
    },
    /// Generic icon (texture ID, label, etc.).
    Icon {
        icon_id: String,
    },
    /// Text label.
    Label {
        text: String,
        font_size: f64,
    },
}

// ── HUD Animation ──────────────────────────────────────────────

/// Animation state for a HUD element.
#[derive(Debug, Clone, PartialEq)]
pub enum HudAnimation {
    /// No animation.
    None,
    /// Pulsing scale (e.g., on damage).
    Pulse {
        frequency: f64,
        amplitude: f64,
        elapsed: f64,
        duration: f64,
    },
    /// Slide in from an offset.
    SlideIn {
        from_x: f64,
        from_y: f64,
        elapsed: f64,
        duration: f64,
    },
    /// Slide out to an offset.
    SlideOut {
        to_x: f64,
        to_y: f64,
        elapsed: f64,
        duration: f64,
    },
    /// Fade in/out.
    Fade {
        from_alpha: f64,
        to_alpha: f64,
        elapsed: f64,
        duration: f64,
    },
}

impl HudAnimation {
    pub fn is_active(&self) -> bool {
        match self {
            HudAnimation::None => false,
            HudAnimation::Pulse { elapsed, duration, .. } => *elapsed < *duration,
            HudAnimation::SlideIn { elapsed, duration, .. } => *elapsed < *duration,
            HudAnimation::SlideOut { elapsed, duration, .. } => *elapsed < *duration,
            HudAnimation::Fade { elapsed, duration, .. } => *elapsed < *duration,
        }
    }

    pub fn advance(&mut self, dt: f64) {
        match self {
            HudAnimation::Pulse { elapsed, .. } => *elapsed += dt,
            HudAnimation::SlideIn { elapsed, .. } => *elapsed += dt,
            HudAnimation::SlideOut { elapsed, .. } => *elapsed += dt,
            HudAnimation::Fade { elapsed, .. } => *elapsed += dt,
            HudAnimation::None => {}
        }
    }
}

// ── HUD Element ────────────────────────────────────────────────

/// A single HUD element.
#[derive(Debug, Clone, PartialEq)]
pub struct HudElement {
    pub id: String,
    pub kind: HudElementKind,
    pub anchor: Anchor,
    /// Pixel offset from the anchor point.
    pub offset_x: f64,
    pub offset_y: f64,
    /// Size in reference-resolution pixels.
    pub width: f64,
    pub height: f64,
    /// Whether this element is visible.
    pub visible: bool,
    /// Current opacity [0, 1].
    pub opacity: f64,
    /// Current animation.
    pub animation: HudAnimation,
    /// Z-order (higher = on top).
    pub z_order: i32,
}

impl HudElement {
    pub fn new(id: &str, kind: HudElementKind, anchor: Anchor, w: f64, h: f64) -> Self {
        Self {
            id: id.to_string(),
            kind,
            anchor,
            offset_x: 0.0,
            offset_y: 0.0,
            width: w,
            height: h,
            visible: true,
            opacity: 1.0,
            animation: HudAnimation::None,
            z_order: 0,
        }
    }

    pub fn with_offset(mut self, ox: f64, oy: f64) -> Self {
        self.offset_x = ox;
        self.offset_y = oy;
        self
    }

    pub fn with_z_order(mut self, z: i32) -> Self {
        self.z_order = z;
        self
    }

    /// Start a pulse animation.
    pub fn start_pulse(&mut self, frequency: f64, amplitude: f64, duration: f64) {
        self.animation = HudAnimation::Pulse {
            frequency,
            amplitude,
            elapsed: 0.0,
            duration,
        };
    }

    /// Start a slide-in animation.
    pub fn start_slide_in(&mut self, from_x: f64, from_y: f64, duration: f64) {
        self.animation = HudAnimation::SlideIn {
            from_x,
            from_y,
            elapsed: 0.0,
            duration,
        };
    }

    /// Start a slide-out animation.
    pub fn start_slide_out(&mut self, to_x: f64, to_y: f64, duration: f64) {
        self.animation = HudAnimation::SlideOut {
            to_x,
            to_y,
            elapsed: 0.0,
            duration,
        };
    }

    /// Start a fade animation.
    pub fn start_fade(&mut self, from: f64, to: f64, duration: f64) {
        self.animation = HudAnimation::Fade {
            from_alpha: from,
            to_alpha: to,
            elapsed: 0.0,
            duration,
        };
    }
}

// ── Layout Result ──────────────────────────────────────────────

/// The computed layout for one HUD element.
#[derive(Debug, Clone, PartialEq)]
pub struct HudLayoutResult {
    pub element_id: String,
    pub rect: ScreenRect,
    pub opacity: f64,
    pub scale: f64,
    pub z_order: i32,
}

// ── HUD Layout System ─────────────────────────────────────────

/// Top-level HUD layout engine.
#[derive(Debug, Clone)]
pub struct HudLayout {
    elements: HashMap<String, HudElement>,
    /// Reference resolution.
    pub ref_width: f64,
    pub ref_height: f64,
    /// Display resolution.
    pub display_width: f64,
    pub display_height: f64,
    /// Safe area insets.
    pub safe_area: SafeAreaInsets,
}

impl HudLayout {
    pub fn new(ref_w: f64, ref_h: f64) -> Self {
        Self {
            elements: HashMap::new(),
            ref_width: ref_w,
            ref_height: ref_h,
            display_width: ref_w,
            display_height: ref_h,
            safe_area: SafeAreaInsets::zero(),
        }
    }

    pub fn set_display_size(&mut self, w: f64, h: f64) {
        self.display_width = w;
        self.display_height = h;
    }

    pub fn set_safe_area(&mut self, insets: SafeAreaInsets) {
        self.safe_area = insets;
    }

    pub fn add_element(&mut self, elem: HudElement) {
        self.elements.insert(elem.id.clone(), elem);
    }

    pub fn remove_element(&mut self, id: &str) -> bool {
        self.elements.remove(id).is_some()
    }

    pub fn element(&self, id: &str) -> Option<&HudElement> {
        self.elements.get(id)
    }

    pub fn element_mut(&mut self, id: &str) -> Option<&mut HudElement> {
        self.elements.get_mut(id)
    }

    pub fn set_visibility(&mut self, id: &str, visible: bool) {
        if let Some(e) = self.elements.get_mut(id) {
            e.visible = visible;
        }
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Scale factor from reference to display.
    pub fn scale_factor(&self) -> f64 {
        if self.ref_width <= 0.0 || self.ref_height <= 0.0 {
            return 1.0;
        }
        let sx = self.display_width / self.ref_width;
        let sy = self.display_height / self.ref_height;
        sx.min(sy)
    }

    /// The usable area after safe-area insets.
    fn safe_rect(&self) -> ScreenRect {
        let s = self.scale_factor();
        ScreenRect::new(
            self.safe_area.left * s,
            self.safe_area.top * s,
            self.display_width - (self.safe_area.left + self.safe_area.right) * s,
            self.display_height - (self.safe_area.top + self.safe_area.bottom) * s,
        )
    }

    /// Advance animations by `dt`.
    pub fn update(&mut self, dt: f64) {
        let keys: Vec<String> = self.elements.keys().cloned().collect();
        for key in keys {
            if let Some(elem) = self.elements.get_mut(&key) {
                elem.animation.advance(dt);
            }
        }
    }

    /// Compute the layout for all visible elements. Returns results sorted
    /// by z-order (ascending).
    pub fn layout(&self) -> Vec<HudLayoutResult> {
        let safe = self.safe_rect();
        let s = self.scale_factor();

        let mut results: Vec<HudLayoutResult> = self
            .elements
            .values()
            .filter(|e| e.visible && e.opacity > 0.0)
            .map(|elem| {
                let (fx, fy) = elem.anchor.factors();

                // Anchor position within the safe area
                let anchor_x = safe.x + safe.width * fx;
                let anchor_y = safe.y + safe.height * fy;

                let scaled_w = elem.width * s;
                let scaled_h = elem.height * s;

                // Position: anchor point minus the element's own anchor fraction
                let base_x = anchor_x + elem.offset_x * s - scaled_w * fx;
                let base_y = anchor_y + elem.offset_y * s - scaled_h * fy;

                // Apply animation
                let (anim_dx, anim_dy, anim_scale, anim_alpha) =
                    self.compute_animation(elem, s);

                let final_x = base_x + anim_dx;
                let final_y = base_y + anim_dy;
                let final_scale = anim_scale;
                let final_opacity = (elem.opacity * anim_alpha).clamp(0.0, 1.0);

                HudLayoutResult {
                    element_id: elem.id.clone(),
                    rect: ScreenRect::new(final_x, final_y, scaled_w * final_scale, scaled_h * final_scale),
                    opacity: final_opacity,
                    scale: final_scale,
                    z_order: elem.z_order,
                }
            })
            .collect();

        results.sort_by_key(|r| r.z_order);
        results
    }

    fn compute_animation(&self, elem: &HudElement, scale: f64) -> (f64, f64, f64, f64) {
        match &elem.animation {
            HudAnimation::None => (0.0, 0.0, 1.0, 1.0),
            HudAnimation::Pulse { frequency, amplitude, elapsed, duration } => {
                if *elapsed >= *duration {
                    return (0.0, 0.0, 1.0, 1.0);
                }
                let t = elapsed * frequency * std::f64::consts::TAU;
                let pulse = 1.0 + t.sin() * amplitude;
                (0.0, 0.0, pulse, 1.0)
            }
            HudAnimation::SlideIn { from_x, from_y, elapsed, duration } => {
                let t = if *duration > 0.0 { (elapsed / duration).clamp(0.0, 1.0) } else { 1.0 };
                // Ease out
                let ease = 1.0 - (1.0 - t) * (1.0 - t);
                let dx = from_x * scale * (1.0 - ease);
                let dy = from_y * scale * (1.0 - ease);
                (dx, dy, 1.0, t)
            }
            HudAnimation::SlideOut { to_x, to_y, elapsed, duration } => {
                let t = if *duration > 0.0 { (elapsed / duration).clamp(0.0, 1.0) } else { 1.0 };
                let ease = t * t;
                let dx = to_x * scale * ease;
                let dy = to_y * scale * ease;
                (dx, dy, 1.0, 1.0 - t)
            }
            HudAnimation::Fade { from_alpha, to_alpha, elapsed, duration } => {
                let t = if *duration > 0.0 { (elapsed / duration).clamp(0.0, 1.0) } else { 1.0 };
                let alpha = from_alpha + (to_alpha - from_alpha) * t;
                (0.0, 0.0, 1.0, alpha)
            }
        }
    }

    /// Hit test: which element (if any) is at the given screen position?
    /// Returns the topmost (highest z-order) element ID.
    pub fn hit_test(&self, screen_x: f64, screen_y: f64) -> Option<String> {
        let layouts = self.layout();
        // Reverse iterate for highest z-order first
        for result in layouts.iter().rev() {
            if result.rect.contains(screen_x, screen_y) {
                return Some(result.element_id.clone());
            }
        }
        None
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn simple_hud() -> HudLayout {
        let mut hud = HudLayout::new(1920.0, 1080.0);
        hud.set_display_size(1920.0, 1080.0);
        hud
    }

    fn health_bar() -> HudElement {
        HudElement::new(
            "health",
            HudElementKind::Bar {
                current: 80.0,
                max: 100.0,
                color_r: 1.0,
                color_g: 0.2,
                color_b: 0.2,
            },
            Anchor::TopLeft,
            200.0,
            24.0,
        )
        .with_offset(10.0, 10.0)
    }

    fn score_counter() -> HudElement {
        HudElement::new(
            "score",
            HudElementKind::Counter { value: 0, label: "SCORE".to_string() },
            Anchor::TopCenter,
            120.0,
            32.0,
        )
    }

    #[test]
    fn anchor_factors() {
        let (fx, fy) = Anchor::TopLeft.factors();
        assert!((fx).abs() < EPS);
        assert!((fy).abs() < EPS);
        let (fx, fy) = Anchor::BottomRight.factors();
        assert!((fx - 1.0).abs() < EPS);
        assert!((fy - 1.0).abs() < EPS);
        let (fx, fy) = Anchor::Center.factors();
        assert!((fx - 0.5).abs() < EPS);
        assert!((fy - 0.5).abs() < EPS);
    }

    #[test]
    fn screen_rect_contains() {
        let r = ScreenRect::new(10.0, 20.0, 100.0, 50.0);
        assert!(r.contains(50.0, 40.0));
        assert!(!r.contains(5.0, 40.0));
        assert!(!r.contains(50.0, 100.0));
    }

    #[test]
    fn screen_rect_center() {
        let r = ScreenRect::new(0.0, 0.0, 100.0, 60.0);
        assert!((r.center_x() - 50.0).abs() < EPS);
        assert!((r.center_y() - 30.0).abs() < EPS);
    }

    #[test]
    fn add_and_find_element() {
        let mut hud = simple_hud();
        hud.add_element(health_bar());
        assert!(hud.element("health").is_some());
        assert_eq!(hud.element_count(), 1);
    }

    #[test]
    fn remove_element() {
        let mut hud = simple_hud();
        hud.add_element(health_bar());
        assert!(hud.remove_element("health"));
        assert_eq!(hud.element_count(), 0);
    }

    #[test]
    fn visibility_toggle() {
        let mut hud = simple_hud();
        hud.add_element(health_bar());
        hud.set_visibility("health", false);
        let layouts = hud.layout();
        assert!(layouts.is_empty());
    }

    #[test]
    fn layout_top_left_position() {
        let mut hud = simple_hud();
        hud.add_element(health_bar());
        let layouts = hud.layout();
        assert_eq!(layouts.len(), 1);
        let r = &layouts[0].rect;
        // TopLeft with offset (10, 10), scale 1.0
        assert!((r.x - 10.0).abs() < EPS);
        assert!((r.y - 10.0).abs() < EPS);
    }

    #[test]
    fn layout_top_center_position() {
        let mut hud = simple_hud();
        hud.add_element(score_counter());
        let layouts = hud.layout();
        let r = &layouts[0].rect;
        // TopCenter: anchor_x = 960, offset = 0, element center = 960
        assert!((r.center_x() - 960.0).abs() < EPS);
    }

    #[test]
    fn layout_bottom_right() {
        let mut hud = simple_hud();
        hud.add_element(
            HudElement::new(
                "ammo",
                HudElementKind::Ammo { current: 30, magazine: 30, reserve: 120 },
                Anchor::BottomRight,
                100.0,
                40.0,
            )
            .with_offset(-10.0, -10.0),
        );
        let layouts = hud.layout();
        let r = &layouts[0].rect;
        assert!(r.right() < 1920.0 + EPS);
        assert!(r.bottom() < 1080.0 + EPS);
    }

    #[test]
    fn scale_factor_uniform() {
        let mut hud = HudLayout::new(1920.0, 1080.0);
        hud.set_display_size(3840.0, 2160.0);
        assert!((hud.scale_factor() - 2.0).abs() < EPS);
    }

    #[test]
    fn scale_factor_aspect_mismatch() {
        let mut hud = HudLayout::new(1920.0, 1080.0);
        hud.set_display_size(2560.0, 1080.0);
        // min(2560/1920, 1080/1080) = min(1.33, 1.0) = 1.0
        assert!((hud.scale_factor() - 1.0).abs() < EPS);
    }

    #[test]
    fn safe_area_insets_affect_layout() {
        let mut hud = simple_hud();
        hud.set_safe_area(SafeAreaInsets::new(50.0, 50.0, 50.0, 50.0));
        hud.add_element(health_bar());
        let layouts = hud.layout();
        let r = &layouts[0].rect;
        // Should be offset from safe area: 50 + 10 = 60
        assert!((r.x - 60.0).abs() < EPS);
        assert!((r.y - 60.0).abs() < EPS);
    }

    #[test]
    fn z_order_sorting() {
        let mut hud = simple_hud();
        hud.add_element(health_bar().with_z_order(10));
        hud.add_element(score_counter().with_z_order(5));
        let layouts = hud.layout();
        assert_eq!(layouts.len(), 2);
        assert!(layouts[0].z_order <= layouts[1].z_order);
    }

    #[test]
    fn pulse_animation_changes_scale() {
        let mut hud = simple_hud();
        let mut hb = health_bar();
        hb.start_pulse(2.0, 0.2, 1.0);
        hud.add_element(hb);
        hud.update(0.125); // quarter cycle at freq 2 → sin(pi/2)=1 → scale=1.2
        let layouts = hud.layout();
        assert!((layouts[0].scale - 1.2).abs() < 0.01);
    }

    #[test]
    fn slide_in_animation() {
        let mut hud = simple_hud();
        let mut elem = health_bar();
        elem.start_slide_in(-200.0, 0.0, 0.5);
        hud.add_element(elem);
        // At t=0, element should be offset left
        let layouts_start = hud.layout();
        let x_start = layouts_start[0].rect.x;
        hud.update(0.5);
        let layouts_end = hud.layout();
        let x_end = layouts_end[0].rect.x;
        assert!(x_end > x_start);
    }

    #[test]
    fn fade_animation() {
        let mut hud = simple_hud();
        let mut elem = health_bar();
        elem.start_fade(0.0, 1.0, 1.0);
        hud.add_element(elem);
        let layouts = hud.layout();
        // At t=0, alpha ≈ 0
        assert!(layouts[0].opacity < 0.1);
        hud.update(1.0);
        let layouts = hud.layout();
        assert!((layouts[0].opacity - 1.0).abs() < 0.05);
    }

    #[test]
    fn hit_test_finds_element() {
        let mut hud = simple_hud();
        hud.add_element(health_bar());
        let hit = hud.hit_test(50.0, 20.0);
        assert_eq!(hit, Some("health".to_string()));
    }

    #[test]
    fn hit_test_misses() {
        let mut hud = simple_hud();
        hud.add_element(health_bar());
        let hit = hud.hit_test(1500.0, 800.0);
        assert!(hit.is_none());
    }

    #[test]
    fn hit_test_z_order_top_wins() {
        let mut hud = simple_hud();
        hud.add_element(
            HudElement::new("bottom", HudElementKind::MinimapFrame, Anchor::TopLeft, 200.0, 200.0)
                .with_z_order(0),
        );
        hud.add_element(
            HudElement::new("top", HudElementKind::MinimapFrame, Anchor::TopLeft, 200.0, 200.0)
                .with_z_order(10),
        );
        let hit = hud.hit_test(50.0, 50.0);
        assert_eq!(hit, Some("top".to_string()));
    }

    #[test]
    fn element_mut_update_value() {
        let mut hud = simple_hud();
        hud.add_element(score_counter());
        if let Some(elem) = hud.element_mut("score") {
            if let HudElementKind::Counter { value, .. } = &mut elem.kind {
                *value = 999;
            }
        }
        if let HudElementKind::Counter { value, .. } = &hud.element("score").unwrap().kind {
            assert_eq!(*value, 999);
        }
    }

    #[test]
    fn crosshair_center_anchor() {
        let mut hud = simple_hud();
        hud.add_element(HudElement::new(
            "crosshair",
            HudElementKind::Crosshair { size: 20.0, gap: 4.0, thickness: 2.0 },
            Anchor::Center,
            20.0,
            20.0,
        ));
        let layouts = hud.layout();
        let r = &layouts[0].rect;
        assert!((r.center_x() - 960.0).abs() < EPS);
        assert!((r.center_y() - 540.0).abs() < EPS);
    }
}
