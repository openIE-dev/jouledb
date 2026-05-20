//! Cursor/pointer state management for games.
//!
//! Lock modes: free, confined (within bounds), locked (hidden, raw mouse delta).
//! Cursor visibility toggle. Custom cursor shapes (arrow, crosshair, hand, resize,
//! custom). Warp cursor to position. Relative mouse mode for FPS-style look.

// ── Lock Mode ───────────────────────────────────────────────────

/// Cursor lock/confinement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CursorLockMode {
    /// Cursor moves freely, visible.
    Free,
    /// Cursor is confined within a bounding rectangle.
    Confined,
    /// Cursor is locked (hidden) and only raw deltas are reported.
    Locked,
}

// ── Cursor Shape ────────────────────────────────────────────────

/// Standard and custom cursor shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CursorShape {
    Arrow,
    Crosshair,
    Hand,
    Text,
    Wait,
    ResizeNS,
    ResizeEW,
    ResizeNESW,
    ResizeNWSE,
    Move,
    NotAllowed,
    /// Custom cursor identified by a name/key.
    Custom(String),
    /// Hidden cursor (used internally in Locked mode).
    None,
}

impl CursorShape {
    /// CSS cursor value equivalent.
    pub fn to_css(&self) -> &str {
        match self {
            Self::Arrow => "default",
            Self::Crosshair => "crosshair",
            Self::Hand => "pointer",
            Self::Text => "text",
            Self::Wait => "wait",
            Self::ResizeNS => "ns-resize",
            Self::ResizeEW => "ew-resize",
            Self::ResizeNESW => "nesw-resize",
            Self::ResizeNWSE => "nwse-resize",
            Self::Move => "move",
            Self::NotAllowed => "not-allowed",
            Self::Custom(_) => "auto",
            Self::None => "none",
        }
    }
}

// ── Bounds ──────────────────────────────────────────────────────

/// A rectangular bounding area for cursor confinement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CursorBounds {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl CursorBounds {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width: width.max(0.0), height: height.max(0.0) }
    }

    /// Clamp a point to be within these bounds.
    pub fn clamp(&self, px: f64, py: f64) -> (f64, f64) {
        let cx = px.clamp(self.x, self.x + self.width);
        let cy = py.clamp(self.y, self.y + self.height);
        (cx, cy)
    }

    /// Check if a point is inside the bounds.
    pub fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px <= self.x + self.width
            && py >= self.y && py <= self.y + self.height
    }

    /// Center of the bounds.
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

// ── Warp Request ────────────────────────────────────────────────

/// A pending cursor warp request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpRequest {
    pub x: f64,
    pub y: f64,
}

// ── Cursor State ────────────────────────────────────────────────

/// Complete cursor/pointer state for the game.
pub struct CursorState {
    // Position
    pub x: f64,
    pub y: f64,
    // Raw deltas (for Locked mode / FPS look)
    pub delta_x: f64,
    pub delta_y: f64,
    // Accumulated delta since last read
    accum_dx: f64,
    accum_dy: f64,
    // Mode & visibility
    mode: CursorLockMode,
    visible: bool,
    shape: CursorShape,
    // Confinement bounds (used in Confined mode)
    bounds: Option<CursorBounds>,
    // Pending warp
    pending_warp: Option<WarpRequest>,
    // Sensitivity multiplier for relative mode
    sensitivity: f64,
    // Whether relative mode is active (FPS look)
    relative_mode: bool,
}

impl CursorState {
    pub fn new() -> Self {
        Self {
            x: 0.0, y: 0.0,
            delta_x: 0.0, delta_y: 0.0,
            accum_dx: 0.0, accum_dy: 0.0,
            mode: CursorLockMode::Free,
            visible: true,
            shape: CursorShape::Arrow,
            bounds: None,
            pending_warp: None,
            sensitivity: 1.0,
            relative_mode: false,
        }
    }

    // ── Frame management ────────────────────────────────────

    /// Begin a new frame: transfer accumulated deltas and reset.
    pub fn begin_frame(&mut self) {
        self.delta_x = self.accum_dx;
        self.delta_y = self.accum_dy;
        self.accum_dx = 0.0;
        self.accum_dy = 0.0;
    }

    // ── Input handling ──────────────────────────────────────

    /// Feed a mouse move event. In Free/Confined mode, updates absolute position.
    /// In Locked mode, only accumulates deltas.
    pub fn on_mouse_move(&mut self, abs_x: f64, abs_y: f64, raw_dx: f64, raw_dy: f64) {
        let dx = raw_dx * self.sensitivity;
        let dy = raw_dy * self.sensitivity;

        match self.mode {
            CursorLockMode::Free => {
                self.x = abs_x;
                self.y = abs_y;
                self.accum_dx += dx;
                self.accum_dy += dy;
            }
            CursorLockMode::Confined => {
                if let Some(ref bounds) = self.bounds {
                    let (cx, cy) = bounds.clamp(abs_x, abs_y);
                    self.x = cx;
                    self.y = cy;
                } else {
                    self.x = abs_x;
                    self.y = abs_y;
                }
                self.accum_dx += dx;
                self.accum_dy += dy;
            }
            CursorLockMode::Locked => {
                // In locked mode, absolute position doesn't change
                self.accum_dx += dx;
                self.accum_dy += dy;
            }
        }
    }

    /// Feed raw mouse movement (for Locked/FPS mode without absolute coords).
    pub fn on_raw_move(&mut self, dx: f64, dy: f64) {
        let sdx = dx * self.sensitivity;
        let sdy = dy * self.sensitivity;
        self.accum_dx += sdx;
        self.accum_dy += sdy;
    }

    // ── Lock mode ───────────────────────────────────────────

    /// Get current lock mode.
    pub fn lock_mode(&self) -> CursorLockMode { self.mode }

    /// Set cursor to Free mode.
    pub fn set_free(&mut self) {
        self.mode = CursorLockMode::Free;
        if !self.visible {
            self.visible = true;
            self.shape = CursorShape::Arrow;
        }
        self.relative_mode = false;
    }

    /// Set cursor to Confined mode within the given bounds.
    pub fn set_confined(&mut self, bounds: CursorBounds) {
        self.mode = CursorLockMode::Confined;
        self.bounds = Some(bounds);
        // Clamp current position to bounds
        if let Some(ref b) = self.bounds {
            let (cx, cy) = b.clamp(self.x, self.y);
            self.x = cx;
            self.y = cy;
        }
    }

    /// Set cursor to Locked mode (hidden, raw delta only). Used for FPS look.
    pub fn set_locked(&mut self) {
        self.mode = CursorLockMode::Locked;
        self.visible = false;
        self.shape = CursorShape::None;
        self.relative_mode = true;
    }

    // ── Visibility ──────────────────────────────────────────

    /// Is the cursor currently visible?
    pub fn is_visible(&self) -> bool { self.visible }

    /// Show the cursor.
    pub fn show(&mut self) {
        self.visible = true;
        if self.shape == CursorShape::None {
            self.shape = CursorShape::Arrow;
        }
    }

    /// Hide the cursor.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    // ── Shape ───────────────────────────────────────────────

    /// Get current cursor shape.
    pub fn shape(&self) -> &CursorShape { &self.shape }

    /// Set cursor shape.
    pub fn set_shape(&mut self, shape: CursorShape) {
        self.shape = shape;
    }

    // ── Warp ────────────────────────────────────────────────

    /// Request a cursor warp to a specific position.
    pub fn warp_to(&mut self, x: f64, y: f64) {
        self.pending_warp = Some(WarpRequest { x, y });
        self.x = x;
        self.y = y;
        if self.mode == CursorLockMode::Confined {
            if let Some(ref bounds) = self.bounds {
                let (cx, cy) = bounds.clamp(x, y);
                self.x = cx;
                self.y = cy;
            }
        }
    }

    /// Take the pending warp request (platform layer should execute it).
    pub fn take_warp(&mut self) -> Option<WarpRequest> {
        self.pending_warp.take()
    }

    // ── Sensitivity ─────────────────────────────────────────

    /// Get mouse sensitivity multiplier.
    pub fn sensitivity(&self) -> f64 { self.sensitivity }

    /// Set mouse sensitivity multiplier.
    pub fn set_sensitivity(&mut self, s: f64) {
        self.sensitivity = s.max(0.01);
    }

    // ── Relative mode ───────────────────────────────────────

    /// Is relative (FPS-style) mouse mode active?
    pub fn is_relative_mode(&self) -> bool { self.relative_mode }

    /// Enable relative mouse mode (implies locked).
    pub fn enable_relative_mode(&mut self) {
        self.set_locked();
    }

    /// Disable relative mouse mode (returns to free).
    pub fn disable_relative_mode(&mut self) {
        self.set_free();
    }

    // ── Confinement bounds ──────────────────────────────────

    /// Get the confinement bounds (if set).
    pub fn bounds(&self) -> Option<&CursorBounds> { self.bounds.as_ref() }

    /// Clear the confinement bounds.
    pub fn clear_bounds(&mut self) {
        self.bounds = None;
    }

    // ── Position helpers ────────────────────────────────────

    /// Get the cursor position as (x, y).
    pub fn position(&self) -> (f64, f64) { (self.x, self.y) }

    /// Get the frame delta as (dx, dy).
    pub fn delta(&self) -> (f64, f64) { (self.delta_x, self.delta_y) }
}

impl Default for CursorState {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let cs = CursorState::new();
        assert_eq!(cs.lock_mode(), CursorLockMode::Free);
        assert!(cs.is_visible());
        assert_eq!(*cs.shape(), CursorShape::Arrow);
        assert!((cs.x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_free_mode_position() {
        let mut cs = CursorState::new();
        cs.on_mouse_move(100.0, 200.0, 100.0, 200.0);
        assert!((cs.x - 100.0).abs() < 1e-9);
        assert!((cs.y - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_confined_clamp() {
        let mut cs = CursorState::new();
        cs.set_confined(CursorBounds::new(0.0, 0.0, 100.0, 100.0));
        cs.on_mouse_move(150.0, 50.0, 0.0, 0.0);
        assert!((cs.x - 100.0).abs() < 1e-9);
        assert!((cs.y - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_locked_mode_no_position_update() {
        let mut cs = CursorState::new();
        cs.x = 50.0;
        cs.y = 50.0;
        cs.set_locked();
        cs.on_mouse_move(200.0, 300.0, 10.0, 20.0);
        // Position should not change in locked mode
        assert!((cs.x - 50.0).abs() < 1e-9);
        assert!((cs.y - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_locked_mode_accumulates_delta() {
        let mut cs = CursorState::new();
        cs.set_locked();
        cs.on_mouse_move(0.0, 0.0, 5.0, -3.0);
        cs.on_mouse_move(0.0, 0.0, 2.0, 1.0);
        cs.begin_frame();
        assert!((cs.delta_x - 7.0).abs() < 1e-9);
        assert!((cs.delta_y - -2.0).abs() < 1e-9);
    }

    #[test]
    fn test_delta_reset_each_frame() {
        let mut cs = CursorState::new();
        cs.on_mouse_move(10.0, 10.0, 10.0, 10.0);
        cs.begin_frame();
        assert!((cs.delta_x - 10.0).abs() < 1e-9);
        cs.begin_frame();
        assert!((cs.delta_x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_visibility_toggle() {
        let mut cs = CursorState::new();
        assert!(cs.is_visible());
        cs.hide();
        assert!(!cs.is_visible());
        cs.show();
        assert!(cs.is_visible());
    }

    #[test]
    fn test_locked_hides_cursor() {
        let mut cs = CursorState::new();
        cs.set_locked();
        assert!(!cs.is_visible());
        assert_eq!(*cs.shape(), CursorShape::None);
    }

    #[test]
    fn test_free_restores_visibility() {
        let mut cs = CursorState::new();
        cs.set_locked();
        cs.set_free();
        assert!(cs.is_visible());
        assert_eq!(*cs.shape(), CursorShape::Arrow);
    }

    #[test]
    fn test_shape_change() {
        let mut cs = CursorState::new();
        cs.set_shape(CursorShape::Crosshair);
        assert_eq!(*cs.shape(), CursorShape::Crosshair);
        cs.set_shape(CursorShape::Custom("aim".into()));
        assert_eq!(*cs.shape(), CursorShape::Custom("aim".into()));
    }

    #[test]
    fn test_cursor_css_values() {
        assert_eq!(CursorShape::Arrow.to_css(), "default");
        assert_eq!(CursorShape::Crosshair.to_css(), "crosshair");
        assert_eq!(CursorShape::Hand.to_css(), "pointer");
        assert_eq!(CursorShape::None.to_css(), "none");
        assert_eq!(CursorShape::ResizeNS.to_css(), "ns-resize");
    }

    #[test]
    fn test_warp_to() {
        let mut cs = CursorState::new();
        cs.warp_to(500.0, 300.0);
        assert!((cs.x - 500.0).abs() < 1e-9);
        assert!((cs.y - 300.0).abs() < 1e-9);
        let warp = cs.take_warp().unwrap();
        assert!((warp.x - 500.0).abs() < 1e-9);
        assert!(cs.take_warp().is_none());
    }

    #[test]
    fn test_warp_confined_clamps() {
        let mut cs = CursorState::new();
        cs.set_confined(CursorBounds::new(0.0, 0.0, 200.0, 200.0));
        cs.warp_to(300.0, 100.0);
        assert!((cs.x - 200.0).abs() < 1e-9);
        assert!((cs.y - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_sensitivity() {
        let mut cs = CursorState::new();
        cs.set_sensitivity(2.0);
        assert!((cs.sensitivity() - 2.0).abs() < 1e-9);
        cs.set_locked();
        cs.on_mouse_move(0.0, 0.0, 5.0, 5.0);
        cs.begin_frame();
        assert!((cs.delta_x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_sensitivity_min() {
        let mut cs = CursorState::new();
        cs.set_sensitivity(-5.0);
        assert!((cs.sensitivity() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_relative_mode() {
        let mut cs = CursorState::new();
        cs.enable_relative_mode();
        assert!(cs.is_relative_mode());
        assert_eq!(cs.lock_mode(), CursorLockMode::Locked);
        cs.disable_relative_mode();
        assert!(!cs.is_relative_mode());
        assert_eq!(cs.lock_mode(), CursorLockMode::Free);
    }

    #[test]
    fn test_raw_move() {
        let mut cs = CursorState::new();
        cs.set_locked();
        cs.on_raw_move(3.0, -2.0);
        cs.on_raw_move(1.0, 4.0);
        cs.begin_frame();
        assert!((cs.delta_x - 4.0).abs() < 1e-9);
        assert!((cs.delta_y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_bounds_contains() {
        let b = CursorBounds::new(10.0, 10.0, 100.0, 100.0);
        assert!(b.contains(50.0, 50.0));
        assert!(!b.contains(5.0, 50.0));
        assert!(!b.contains(50.0, 120.0));
    }

    #[test]
    fn test_bounds_center() {
        let b = CursorBounds::new(0.0, 0.0, 200.0, 100.0);
        let (cx, cy) = b.center();
        assert!((cx - 100.0).abs() < 1e-9);
        assert!((cy - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_clear_bounds() {
        let mut cs = CursorState::new();
        cs.set_confined(CursorBounds::new(0.0, 0.0, 100.0, 100.0));
        assert!(cs.bounds().is_some());
        cs.clear_bounds();
        assert!(cs.bounds().is_none());
    }

    #[test]
    fn test_position_and_delta_helpers() {
        let mut cs = CursorState::new();
        cs.x = 10.0;
        cs.y = 20.0;
        assert_eq!(cs.position(), (10.0, 20.0));
        cs.delta_x = 3.0;
        cs.delta_y = 4.0;
        assert_eq!(cs.delta(), (3.0, 4.0));
    }

    #[test]
    fn test_default_impl() {
        let cs = CursorState::default();
        assert_eq!(cs.lock_mode(), CursorLockMode::Free);
    }

    #[test]
    fn test_bounds_negative_size_clamped() {
        let b = CursorBounds::new(0.0, 0.0, -10.0, -20.0);
        assert!((b.width - 0.0).abs() < 1e-9);
        assert!((b.height - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_show_after_none_shape() {
        let mut cs = CursorState::new();
        cs.set_shape(CursorShape::None);
        cs.show();
        assert!(cs.is_visible());
        assert_eq!(*cs.shape(), CursorShape::Arrow);
    }
}
