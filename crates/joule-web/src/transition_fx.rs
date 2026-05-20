//! Screen transition effects between scenes/levels: fade, wipe, iris,
//! pixelate dissolve, diamond/cross patterns. State machine drives the
//! lifecycle: Idle → TransitionOut → Hold → TransitionIn → Done.
//!
//! Pure math — outputs a `TransitionFrame` describing what the renderer
//! should draw each tick.

// ── Easing ─────────────────────────────────────────────────────

/// Easing curve for transition timing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionEasing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    /// Quadratic ease in.
    QuadIn,
    /// Quadratic ease out.
    QuadOut,
}

impl TransitionEasing {
    /// Map normalized t ∈ [0,1] through the easing curve.
    pub fn apply(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t * t,
            Self::EaseOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            Self::EaseInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let v = -2.0 * t + 2.0;
                    1.0 - v * v * v / 2.0
                }
            }
            Self::QuadIn => t * t,
            Self::QuadOut => t * (2.0 - t),
        }
    }
}

// ── Transition Kinds ───────────────────────────────────────────

/// The visual style of the transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionKind {
    /// Fade to/from a solid color.
    Fade,
    /// Horizontal wipe (left to right).
    WipeHorizontal,
    /// Vertical wipe (top to bottom).
    WipeVertical,
    /// Diagonal wipe (top-left to bottom-right).
    WipeDiagonal,
    /// Circular iris (closing/opening).
    IrisCircle,
    /// Pixelate dissolve (blocks increase in size).
    Pixelate,
    /// Diamond pattern.
    Diamond,
    /// Cross/plus pattern expanding from center.
    Cross,
}

// ── Color ──────────────────────────────────────────────────────

/// Simple RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitionColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl TransitionColor {
    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
    }

    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }
}

// ── State Machine ──────────────────────────────────────────────

/// Lifecycle state of a screen transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionState {
    /// No transition active.
    Idle,
    /// Outgoing scene is being covered.
    TransitionOut,
    /// Screen is fully covered — hold for scene swap.
    Hold,
    /// New scene is being revealed.
    TransitionIn,
    /// Transition complete.
    Done,
}

// ── Transition Config ──────────────────────────────────────────

/// Configuration for a single transition.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionConfig {
    pub kind: TransitionKind,
    /// Color used for fade/wipe overlays.
    pub color: TransitionColor,
    /// Duration of the out phase in seconds.
    pub out_duration: f64,
    /// Duration of the hold phase in seconds.
    pub hold_duration: f64,
    /// Duration of the in phase in seconds.
    pub in_duration: f64,
    /// Easing for the out phase.
    pub out_easing: TransitionEasing,
    /// Easing for the in phase.
    pub in_easing: TransitionEasing,
    /// Screen width (for spatial transitions).
    pub screen_width: f64,
    /// Screen height.
    pub screen_height: f64,
}

impl TransitionConfig {
    pub fn fade_black(out_dur: f64, hold: f64, in_dur: f64) -> Self {
        Self {
            kind: TransitionKind::Fade,
            color: TransitionColor::black(),
            out_duration: out_dur,
            hold_duration: hold,
            in_duration: in_dur,
            out_easing: TransitionEasing::EaseInOut,
            in_easing: TransitionEasing::EaseInOut,
            screen_width: 1920.0,
            screen_height: 1080.0,
        }
    }

    pub fn with_kind(mut self, kind: TransitionKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_color(mut self, color: TransitionColor) -> Self {
        self.color = color;
        self
    }

    pub fn with_easing(mut self, out: TransitionEasing, in_ease: TransitionEasing) -> Self {
        self.out_easing = out;
        self.in_easing = in_ease;
        self
    }

    pub fn with_screen_size(mut self, w: f64, h: f64) -> Self {
        self.screen_width = w;
        self.screen_height = h;
        self
    }

    pub fn total_duration(&self) -> f64 {
        self.out_duration + self.hold_duration + self.in_duration
    }
}

// ── Transition Frame (render output) ───────────────────────────

/// Per-frame output describing what the renderer should draw.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionFrame {
    pub state: TransitionState,
    /// Overlay color with current alpha.
    pub overlay_color: TransitionColor,
    /// Progress of current phase [0, 1].
    pub phase_progress: f64,
    /// Overall progress [0, 1].
    pub total_progress: f64,
    /// For spatial transitions: coverage amount [0, 1].
    pub coverage: f64,
    /// For pixelate: current block size in pixels.
    pub pixel_block_size: f64,
    /// For iris: current radius.
    pub iris_radius: f64,
}

// ── Transition Controller ──────────────────────────────────────

/// Drives the transition lifecycle.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionController {
    pub config: TransitionConfig,
    pub state: TransitionState,
    elapsed: f64,
}

impl TransitionController {
    pub fn new(config: TransitionConfig) -> Self {
        Self {
            config,
            state: TransitionState::Idle,
            elapsed: 0.0,
        }
    }

    /// Begin the transition.
    pub fn start(&mut self) {
        self.state = TransitionState::TransitionOut;
        self.elapsed = 0.0;
    }

    /// Reset to idle.
    pub fn reset(&mut self) {
        self.state = TransitionState::Idle;
        self.elapsed = 0.0;
    }

    /// Advance by `dt` seconds and return the current frame.
    pub fn update(&mut self, dt: f64) -> TransitionFrame {
        if self.state == TransitionState::Idle || self.state == TransitionState::Done {
            return self.make_frame(0.0, 0.0);
        }

        self.elapsed += dt;

        // Determine current phase and local progress
        let out_end = self.config.out_duration;
        let hold_end = out_end + self.config.hold_duration;
        let in_end = hold_end + self.config.in_duration;

        if self.elapsed < out_end {
            self.state = TransitionState::TransitionOut;
            let raw_t = if out_end > 0.0 { self.elapsed / out_end } else { 1.0 };
            let eased = self.config.out_easing.apply(raw_t);
            self.make_frame(eased, self.elapsed / in_end.max(1e-12))
        } else if self.elapsed < hold_end {
            self.state = TransitionState::Hold;
            self.make_frame(1.0, self.elapsed / in_end.max(1e-12))
        } else if self.elapsed < in_end {
            self.state = TransitionState::TransitionIn;
            let local = self.elapsed - hold_end;
            let raw_t = if self.config.in_duration > 0.0 {
                local / self.config.in_duration
            } else {
                1.0
            };
            let eased = self.config.in_easing.apply(raw_t);
            self.make_frame(1.0 - eased, self.elapsed / in_end.max(1e-12))
        } else {
            self.state = TransitionState::Done;
            self.make_frame(0.0, 1.0)
        }
    }

    fn make_frame(&self, coverage: f64, total_progress: f64) -> TransitionFrame {
        let c = coverage.clamp(0.0, 1.0);
        let phase_progress = match self.state {
            TransitionState::TransitionOut => c,
            TransitionState::Hold => 1.0,
            TransitionState::TransitionIn => 1.0 - c,
            _ => 0.0,
        };

        let overlay_alpha = match self.config.kind {
            TransitionKind::Fade => c,
            TransitionKind::Pixelate => c * 0.5, // partial overlay during pixelate
            _ => c,
        };

        let pixel_block = match self.config.kind {
            TransitionKind::Pixelate => {
                // Block size from 1 (no pixelation) to max (fully pixelated)
                let max_block = (self.config.screen_width.min(self.config.screen_height) / 8.0).max(1.0);
                1.0 + c * (max_block - 1.0)
            }
            _ => 1.0,
        };

        let iris_r = match self.config.kind {
            TransitionKind::IrisCircle => {
                let diag = (self.config.screen_width.powi(2)
                    + self.config.screen_height.powi(2))
                .sqrt()
                    / 2.0;
                diag * (1.0 - c)
            }
            _ => 0.0,
        };

        TransitionFrame {
            state: self.state,
            overlay_color: TransitionColor {
                r: self.config.color.r,
                g: self.config.color.g,
                b: self.config.color.b,
                a: overlay_alpha,
            },
            phase_progress,
            total_progress: total_progress.clamp(0.0, 1.0),
            coverage: c,
            pixel_block_size: pixel_block,
            iris_radius: iris_r,
        }
    }

    pub fn is_active(&self) -> bool {
        self.state != TransitionState::Idle && self.state != TransitionState::Done
    }

    pub fn is_holding(&self) -> bool {
        self.state == TransitionState::Hold
    }

    pub fn elapsed(&self) -> f64 {
        self.elapsed
    }

    /// Compute a spatial coverage test: given a pixel at (px, py), is it
    /// covered by the transition at the given coverage fraction?
    pub fn is_pixel_covered(&self, px: f64, py: f64, coverage: f64) -> bool {
        let c = coverage.clamp(0.0, 1.0);
        let sw = self.config.screen_width;
        let sh = self.config.screen_height;

        match self.config.kind {
            TransitionKind::Fade | TransitionKind::Pixelate => {
                // These are full-screen overlays
                c > 0.0
            }
            TransitionKind::WipeHorizontal => {
                px <= sw * c
            }
            TransitionKind::WipeVertical => {
                py <= sh * c
            }
            TransitionKind::WipeDiagonal => {
                let normalized = (px / sw + py / sh) / 2.0;
                normalized <= c
            }
            TransitionKind::IrisCircle => {
                let cx = sw / 2.0;
                let cy = sh / 2.0;
                let diag = (sw * sw + sh * sh).sqrt() / 2.0;
                let radius = diag * (1.0 - c);
                let dist = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
                dist > radius
            }
            TransitionKind::Diamond => {
                let cx = sw / 2.0;
                let cy = sh / 2.0;
                let max_dist = cx + cy;
                let dist = (px - cx).abs() + (py - cy).abs();
                dist > max_dist * (1.0 - c)
            }
            TransitionKind::Cross => {
                let cx = sw / 2.0;
                let cy = sh / 2.0;
                let hw = sw * c / 2.0;
                let hh = sh * c / 2.0;
                let in_h_bar = py.abs() <= hh + cy && py >= cy - hh;
                let in_v_bar = px.abs() <= hw + cx && px >= cx - hw;
                // Actually: cross = horizontal bar ∪ vertical bar
                let in_h = (py - cy).abs() <= hh;
                let in_v = (px - cx).abs() <= hw;
                in_h || in_v
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn easing_linear() {
        let e = TransitionEasing::Linear;
        assert!((e.apply(0.0)).abs() < EPS);
        assert!((e.apply(0.5) - 0.5).abs() < EPS);
        assert!((e.apply(1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn easing_ease_in_starts_slow() {
        let e = TransitionEasing::EaseIn;
        assert!(e.apply(0.1) < 0.1);
    }

    #[test]
    fn easing_ease_out_starts_fast() {
        let e = TransitionEasing::EaseOut;
        assert!(e.apply(0.1) > 0.1);
    }

    #[test]
    fn easing_quad_in() {
        let e = TransitionEasing::QuadIn;
        assert!((e.apply(0.5) - 0.25).abs() < EPS);
    }

    #[test]
    fn easing_quad_out() {
        let e = TransitionEasing::QuadOut;
        assert!((e.apply(0.5) - 0.75).abs() < EPS);
    }

    #[test]
    fn easing_clamps_input() {
        let e = TransitionEasing::Linear;
        assert!((e.apply(-0.5)).abs() < EPS);
        assert!((e.apply(1.5) - 1.0).abs() < EPS);
    }

    #[test]
    fn config_total_duration() {
        let c = TransitionConfig::fade_black(0.5, 0.2, 0.5);
        assert!((c.total_duration() - 1.2).abs() < EPS);
    }

    #[test]
    fn controller_starts_idle() {
        let ctrl = TransitionController::new(TransitionConfig::fade_black(1.0, 0.0, 1.0));
        assert_eq!(ctrl.state, TransitionState::Idle);
    }

    #[test]
    fn controller_lifecycle_fade() {
        let mut ctrl = TransitionController::new(TransitionConfig::fade_black(1.0, 0.5, 1.0));
        ctrl.start();

        // Transition out
        let f = ctrl.update(0.5);
        assert_eq!(f.state, TransitionState::TransitionOut);
        assert!(f.coverage > 0.0 && f.coverage < 1.0);

        // Complete out phase
        let f = ctrl.update(0.6);
        assert_eq!(f.state, TransitionState::Hold);
        assert!((f.coverage - 1.0).abs() < EPS);

        // Hold
        let f = ctrl.update(0.3);
        assert_eq!(f.state, TransitionState::Hold);

        // Transition in
        let f = ctrl.update(0.3);
        assert_eq!(f.state, TransitionState::TransitionIn);
        assert!(f.coverage < 1.0);

        // Complete
        let f = ctrl.update(2.0);
        assert_eq!(f.state, TransitionState::Done);
        assert!(f.coverage < EPS);
    }

    #[test]
    fn controller_reset() {
        let mut ctrl = TransitionController::new(TransitionConfig::fade_black(1.0, 0.0, 1.0));
        ctrl.start();
        ctrl.update(0.5);
        ctrl.reset();
        assert_eq!(ctrl.state, TransitionState::Idle);
        assert!(ctrl.elapsed() < EPS);
    }

    #[test]
    fn controller_is_active() {
        let mut ctrl = TransitionController::new(TransitionConfig::fade_black(0.5, 0.0, 0.5));
        assert!(!ctrl.is_active());
        ctrl.start();
        assert!(ctrl.is_active());
        ctrl.update(2.0);
        assert!(!ctrl.is_active());
    }

    #[test]
    fn controller_is_holding() {
        let mut ctrl = TransitionController::new(TransitionConfig::fade_black(0.1, 0.5, 0.1));
        ctrl.start();
        ctrl.update(0.15); // Past out phase
        assert!(ctrl.is_holding());
    }

    #[test]
    fn pixelate_block_size_grows() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::Pixelate)
            .with_screen_size(800.0, 600.0);
        let mut ctrl = TransitionController::new(config);
        ctrl.start();
        let f1 = ctrl.update(0.3);
        let f2 = ctrl.update(0.4);
        assert!(f2.pixel_block_size > f1.pixel_block_size);
    }

    #[test]
    fn iris_radius_shrinks() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::IrisCircle)
            .with_screen_size(800.0, 600.0);
        let mut ctrl = TransitionController::new(config);
        ctrl.start();
        let f1 = ctrl.update(0.2);
        let f2 = ctrl.update(0.3);
        assert!(f2.iris_radius < f1.iris_radius);
    }

    #[test]
    fn wipe_horizontal_pixel_coverage() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::WipeHorizontal)
            .with_screen_size(100.0, 100.0);
        let ctrl = TransitionController::new(config);
        // At 50% coverage, pixel at x=25 should be covered
        assert!(ctrl.is_pixel_covered(25.0, 50.0, 0.5));
        // pixel at x=75 should not
        assert!(!ctrl.is_pixel_covered(75.0, 50.0, 0.5));
    }

    #[test]
    fn wipe_vertical_pixel_coverage() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::WipeVertical)
            .with_screen_size(100.0, 100.0);
        let ctrl = TransitionController::new(config);
        assert!(ctrl.is_pixel_covered(50.0, 25.0, 0.5));
        assert!(!ctrl.is_pixel_covered(50.0, 75.0, 0.5));
    }

    #[test]
    fn iris_pixel_coverage_center_last() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::IrisCircle)
            .with_screen_size(100.0, 100.0);
        let ctrl = TransitionController::new(config);
        // At low coverage, center should NOT be covered (iris still open)
        assert!(!ctrl.is_pixel_covered(50.0, 50.0, 0.1));
        // Corners should be covered first
        assert!(ctrl.is_pixel_covered(0.0, 0.0, 0.1));
    }

    #[test]
    fn diamond_pixel_coverage() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::Diamond)
            .with_screen_size(100.0, 100.0);
        let ctrl = TransitionController::new(config);
        // At 50% coverage, far corners should be covered
        assert!(ctrl.is_pixel_covered(0.0, 0.0, 0.5));
    }

    #[test]
    fn fade_overlay_alpha() {
        let mut ctrl = TransitionController::new(TransitionConfig::fade_black(1.0, 0.0, 1.0));
        ctrl.start();
        let f = ctrl.update(0.5);
        // Fade alpha should be between 0 and 1
        assert!(f.overlay_color.a > 0.0 && f.overlay_color.a < 1.0);
    }

    #[test]
    fn white_fade_color() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_color(TransitionColor::white());
        let mut ctrl = TransitionController::new(config);
        ctrl.start();
        let f = ctrl.update(0.5);
        assert!((f.overlay_color.r - 1.0).abs() < EPS);
        assert!((f.overlay_color.g - 1.0).abs() < EPS);
        assert!((f.overlay_color.b - 1.0).abs() < EPS);
    }

    #[test]
    fn cross_pixel_coverage_center() {
        let config = TransitionConfig::fade_black(1.0, 0.0, 1.0)
            .with_kind(TransitionKind::Cross)
            .with_screen_size(100.0, 100.0);
        let ctrl = TransitionController::new(config);
        // Center should be covered even at low coverage
        assert!(ctrl.is_pixel_covered(50.0, 50.0, 0.1));
    }

    #[test]
    fn zero_duration_phases() {
        let config = TransitionConfig::fade_black(0.0, 0.0, 0.0);
        let mut ctrl = TransitionController::new(config);
        ctrl.start();
        let f = ctrl.update(0.001);
        assert_eq!(f.state, TransitionState::Done);
    }
}
