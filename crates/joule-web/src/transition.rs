//! Page/view transition orchestration for SPAs.
//!
//! Replaces the View Transitions API and libraries like Barba.js with a
//! pure-Rust state machine that drives leave/enter animations between routes.

use serde::{Deserialize, Serialize};

// ── Transition Kind ──

/// The visual effect applied during a page transition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionKind {
    Fade,
    SlideLeft,
    SlideRight,
    SlideUp,
    SlideDown,
    Scale,
    CrossFade,
    None_,
    Custom { name: String },
}

// ── Transition Phase ──

/// Current phase of the transition lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionPhase {
    Idle,
    Leaving,
    Entering,
    Complete,
}

// ── Transition Config ──

/// Timing and easing configuration for a transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionConfig {
    pub leave_duration_ms: u64,
    pub enter_duration_ms: u64,
    pub leave_easing: String,
    pub enter_easing: String,
    pub stagger_ms: u64,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self {
            leave_duration_ms: 300,
            enter_duration_ms: 300,
            leave_easing: "ease".into(),
            enter_easing: "ease".into(),
            stagger_ms: 0,
        }
    }
}

// ── Transition State ──

/// Drives the leave → stagger → enter → complete lifecycle.
#[derive(Debug, Clone)]
pub struct TransitionState {
    pub phase: TransitionPhase,
    pub kind: TransitionKind,
    pub config: TransitionConfig,
    pub from_route: Option<String>,
    pub to_route: Option<String>,
    pub elapsed_ms: u64,
    pub progress: f64,
}

impl TransitionState {
    pub fn new() -> Self {
        Self {
            phase: TransitionPhase::Idle,
            kind: TransitionKind::None_,
            config: TransitionConfig::default(),
            from_route: None,
            to_route: None,
            elapsed_ms: 0,
            progress: 0.0,
        }
    }

    /// Begin a transition from one route to another.
    pub fn start(
        &mut self,
        from: &str,
        to: &str,
        kind: TransitionKind,
        config: TransitionConfig,
    ) {
        self.phase = TransitionPhase::Leaving;
        self.kind = kind;
        self.config = config;
        self.from_route = Some(from.to_string());
        self.to_route = Some(to.to_string());
        self.elapsed_ms = 0;
        self.progress = 0.0;
    }

    /// Advance the transition by `dt_ms` milliseconds. Returns the current phase.
    pub fn tick(&mut self, dt_ms: u64) -> TransitionPhase {
        match self.phase {
            TransitionPhase::Idle | TransitionPhase::Complete => {}
            TransitionPhase::Leaving => {
                self.elapsed_ms += dt_ms;
                let dur = self.config.leave_duration_ms.max(1);
                self.progress = (self.elapsed_ms as f64 / dur as f64).min(1.0);
                if self.elapsed_ms >= dur + self.config.stagger_ms {
                    self.phase = TransitionPhase::Entering;
                    self.elapsed_ms = self
                        .elapsed_ms
                        .saturating_sub(dur + self.config.stagger_ms);
                    self.progress = if self.config.enter_duration_ms == 0 {
                        1.0
                    } else {
                        (self.elapsed_ms as f64 / self.config.enter_duration_ms as f64).min(1.0)
                    };
                    // If enter is already done (zero duration), complete
                    if self.elapsed_ms >= self.config.enter_duration_ms {
                        self.phase = TransitionPhase::Complete;
                        self.progress = 1.0;
                    }
                }
            }
            TransitionPhase::Entering => {
                self.elapsed_ms += dt_ms;
                let dur = self.config.enter_duration_ms.max(1);
                self.progress = (self.elapsed_ms as f64 / dur as f64).min(1.0);
                if self.elapsed_ms >= dur {
                    self.phase = TransitionPhase::Complete;
                    self.progress = 1.0;
                }
            }
        }
        self.phase
    }

    pub fn is_animating(&self) -> bool {
        matches!(
            self.phase,
            TransitionPhase::Leaving | TransitionPhase::Entering
        )
    }

    /// Progress of the leaving phase (0..1). Returns 1.0 once leaving is done.
    pub fn leave_progress(&self) -> f64 {
        match self.phase {
            TransitionPhase::Idle => 0.0,
            TransitionPhase::Leaving => {
                let dur = self.config.leave_duration_ms.max(1);
                (self.elapsed_ms as f64 / dur as f64).min(1.0)
            }
            TransitionPhase::Entering | TransitionPhase::Complete => 1.0,
        }
    }

    /// Progress of the entering phase (0..1).
    pub fn enter_progress(&self) -> f64 {
        match self.phase {
            TransitionPhase::Idle | TransitionPhase::Leaving => 0.0,
            TransitionPhase::Entering => self.progress,
            TransitionPhase::Complete => 1.0,
        }
    }

    /// Cancel the transition, jumping straight to Complete.
    pub fn cancel(&mut self) {
        self.phase = TransitionPhase::Complete;
        self.progress = 1.0;
    }

    /// Reset back to Idle.
    pub fn reset(&mut self) {
        self.phase = TransitionPhase::Idle;
        self.kind = TransitionKind::None_;
        self.elapsed_ms = 0;
        self.progress = 0.0;
        self.from_route = None;
        self.to_route = None;
    }
}

impl Default for TransitionState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Transition Style ──

/// Computed visual properties at a given transition progress.
#[derive(Debug, Clone)]
pub struct TransitionStyle {
    pub opacity: f64,
    pub transform_x: f64,
    pub transform_y: f64,
    pub scale: f64,
}

impl Default for TransitionStyle {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            transform_x: 0.0,
            transform_y: 0.0,
            scale: 1.0,
        }
    }
}

/// Compute the leaving element's style at the given progress (0..1).
pub fn compute_leave_style(kind: &TransitionKind, progress: f64) -> TransitionStyle {
    let p = progress.clamp(0.0, 1.0);
    match kind {
        TransitionKind::Fade => TransitionStyle {
            opacity: 1.0 - p,
            ..Default::default()
        },
        TransitionKind::SlideLeft => TransitionStyle {
            transform_x: -100.0 * p,
            ..Default::default()
        },
        TransitionKind::SlideRight => TransitionStyle {
            transform_x: 100.0 * p,
            ..Default::default()
        },
        TransitionKind::SlideUp => TransitionStyle {
            transform_y: -100.0 * p,
            ..Default::default()
        },
        TransitionKind::SlideDown => TransitionStyle {
            transform_y: 100.0 * p,
            ..Default::default()
        },
        TransitionKind::Scale => TransitionStyle {
            scale: 1.0 - p,
            opacity: 1.0 - p,
            ..Default::default()
        },
        TransitionKind::CrossFade => TransitionStyle {
            opacity: 1.0 - p,
            ..Default::default()
        },
        TransitionKind::None_ | TransitionKind::Custom { .. } => TransitionStyle::default(),
    }
}

/// Compute the entering element's style at the given progress (0..1).
pub fn compute_enter_style(kind: &TransitionKind, progress: f64) -> TransitionStyle {
    let p = progress.clamp(0.0, 1.0);
    match kind {
        TransitionKind::Fade => TransitionStyle {
            opacity: p,
            ..Default::default()
        },
        TransitionKind::SlideLeft => TransitionStyle {
            transform_x: 100.0 * (1.0 - p),
            ..Default::default()
        },
        TransitionKind::SlideRight => TransitionStyle {
            transform_x: -100.0 * (1.0 - p),
            ..Default::default()
        },
        TransitionKind::SlideUp => TransitionStyle {
            transform_y: 100.0 * (1.0 - p),
            ..Default::default()
        },
        TransitionKind::SlideDown => TransitionStyle {
            transform_y: -100.0 * (1.0 - p),
            ..Default::default()
        },
        TransitionKind::Scale => TransitionStyle {
            scale: p,
            opacity: p,
            ..Default::default()
        },
        TransitionKind::CrossFade => TransitionStyle {
            opacity: p,
            ..Default::default()
        },
        TransitionKind::None_ | TransitionKind::Custom { .. } => TransitionStyle::default(),
    }
}

// ── Route-Based Transition Router ──

/// A rule mapping route patterns to transition effects.
#[derive(Debug, Clone)]
pub struct TransitionRule {
    pub from_pattern: Option<String>,
    pub to_pattern: Option<String>,
    pub kind: TransitionKind,
    pub config: TransitionConfig,
}

/// Resolves which transition to use based on source/destination routes.
#[derive(Debug, Clone)]
pub struct TransitionRouter {
    pub rules: Vec<TransitionRule>,
    pub default_kind: TransitionKind,
    pub default_config: TransitionConfig,
}

impl TransitionRouter {
    pub fn new(default: TransitionKind) -> Self {
        Self {
            rules: Vec::new(),
            default_kind: default,
            default_config: TransitionConfig::default(),
        }
    }

    pub fn add_rule(&mut self, rule: TransitionRule) {
        self.rules.push(rule);
    }

    /// Find the first matching rule, or return the defaults.
    pub fn resolve(&self, from: &str, to: &str) -> (TransitionKind, TransitionConfig) {
        for rule in &self.rules {
            let from_match = rule
                .from_pattern
                .as_ref()
                .is_none_or(|p| pattern_matches(p, from));
            let to_match = rule
                .to_pattern
                .as_ref()
                .is_none_or(|p| pattern_matches(p, to));
            if from_match && to_match {
                return (rule.kind.clone(), rule.config.clone());
            }
        }
        (self.default_kind.clone(), self.default_config.clone())
    }
}

/// Simple pattern matching: exact match or prefix wildcard (e.g. `/blog/*`).
fn pattern_matches(pattern: &str, route: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        route.starts_with(prefix)
    } else {
        pattern == route
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_sets_leaving_phase() {
        let mut ts = TransitionState::new();
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        assert_eq!(ts.phase, TransitionPhase::Leaving);
        assert_eq!(ts.from_route.as_deref(), Some("/a"));
        assert_eq!(ts.to_route.as_deref(), Some("/b"));
    }

    #[test]
    fn tick_advances_progress() {
        let mut ts = TransitionState::new();
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        ts.tick(150);
        assert_eq!(ts.phase, TransitionPhase::Leaving);
        assert!((ts.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn phase_transitions_leaving_entering_complete() {
        let mut ts = TransitionState::new();
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        // Finish leaving (300ms)
        ts.tick(300);
        assert_eq!(ts.phase, TransitionPhase::Entering);
        // Finish entering (300ms)
        ts.tick(300);
        assert_eq!(ts.phase, TransitionPhase::Complete);
    }

    #[test]
    fn fade_leave_style_opacity_decreases() {
        let style = compute_leave_style(&TransitionKind::Fade, 0.5);
        assert!((style.opacity - 0.5).abs() < 0.01);
        let style_end = compute_leave_style(&TransitionKind::Fade, 1.0);
        assert!(style_end.opacity.abs() < 0.01);
    }

    #[test]
    fn slide_left_moves_x() {
        let style = compute_leave_style(&TransitionKind::SlideLeft, 0.5);
        assert!((style.transform_x - (-50.0)).abs() < 0.01);
        let enter = compute_enter_style(&TransitionKind::SlideLeft, 0.0);
        assert!((enter.transform_x - 100.0).abs() < 0.01);
    }

    #[test]
    fn cancel_jumps_to_complete() {
        let mut ts = TransitionState::new();
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        ts.tick(100);
        ts.cancel();
        assert_eq!(ts.phase, TransitionPhase::Complete);
        assert!(!ts.is_animating());
    }

    #[test]
    fn reset_to_idle() {
        let mut ts = TransitionState::new();
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        ts.tick(600);
        ts.reset();
        assert_eq!(ts.phase, TransitionPhase::Idle);
        assert!(ts.from_route.is_none());
    }

    #[test]
    fn stagger_delays_enter() {
        let mut ts = TransitionState::new();
        let cfg = TransitionConfig {
            stagger_ms: 100,
            ..Default::default()
        };
        ts.start("/a", "/b", TransitionKind::Fade, cfg);
        // 300ms leave done, but stagger not elapsed yet
        ts.tick(300);
        assert_eq!(ts.phase, TransitionPhase::Leaving);
        // +100ms stagger => now enters
        ts.tick(100);
        assert_eq!(ts.phase, TransitionPhase::Entering);
    }

    #[test]
    fn route_rule_matching() {
        let mut router = TransitionRouter::new(TransitionKind::Fade);
        router.add_rule(TransitionRule {
            from_pattern: Some("/blog/*".into()),
            to_pattern: Some("/blog/*".into()),
            kind: TransitionKind::SlideLeft,
            config: TransitionConfig::default(),
        });
        let (kind, _) = router.resolve("/blog/1", "/blog/2");
        assert_eq!(kind, TransitionKind::SlideLeft);
    }

    #[test]
    fn default_fallback() {
        let router = TransitionRouter::new(TransitionKind::Fade);
        let (kind, _) = router.resolve("/a", "/b");
        assert_eq!(kind, TransitionKind::Fade);
    }

    #[test]
    fn crossfade_both_visible_at_midpoint() {
        let leave = compute_leave_style(&TransitionKind::CrossFade, 0.5);
        let enter = compute_enter_style(&TransitionKind::CrossFade, 0.5);
        // Both should have ~0.5 opacity
        assert!((leave.opacity - 0.5).abs() < 0.01);
        assert!((enter.opacity - 0.5).abs() < 0.01);
    }

    #[test]
    fn enter_progress_zero_at_start() {
        let mut ts = TransitionState::new();
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        assert!((ts.enter_progress()).abs() < 0.01);
    }

    #[test]
    fn is_animating_during_transition() {
        let mut ts = TransitionState::new();
        assert!(!ts.is_animating());
        ts.start("/a", "/b", TransitionKind::Fade, TransitionConfig::default());
        assert!(ts.is_animating());
        ts.tick(600);
        assert!(!ts.is_animating());
    }
}
