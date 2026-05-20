//! Tween animation engine with comprehensive easing functions.
//!
//! Replaces GSAP / anime.js tween capabilities. Supports sequencing,
//! parallel playback, repeating, and yoyo (ping-pong) modes.
//! All math is pure Rust — no browser or timer dependency.

use std::f64::consts::PI;

// ── Easing ─────────────────────────────────────────────────────

/// Easing function type for tween interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweenEasing {
    Linear,
    EaseInQuad,
    EaseOutQuad,
    EaseInOutQuad,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    EaseInQuart,
    EaseOutQuart,
    EaseInOutQuart,
    EaseInExpo,
    EaseOutExpo,
    EaseInOutExpo,
    EaseInBack,
    EaseOutBack,
    EaseInOutBack,
    EaseInBounce,
    EaseOutBounce,
    EaseInOutBounce,
    EaseInElastic,
    EaseOutElastic,
    EaseInOutElastic,
}

impl TweenEasing {
    /// Apply the easing function to a normalized time `t` in [0, 1].
    pub fn apply(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseInQuad => t * t,
            Self::EaseOutQuad => t * (2.0 - t),
            Self::EaseInOutQuad => {
                if t < 0.5 { 2.0 * t * t } else { -1.0 + (4.0 - 2.0 * t) * t }
            }
            Self::EaseInCubic => t * t * t,
            Self::EaseOutCubic => { let u = t - 1.0; u * u * u + 1.0 }
            Self::EaseInOutCubic => {
                if t < 0.5 { 4.0 * t * t * t } else { let u = 2.0 * t - 2.0; (u * u * u + 2.0) / 2.0 }
            }
            Self::EaseInQuart => t * t * t * t,
            Self::EaseOutQuart => { let u = t - 1.0; 1.0 - u * u * u * u }
            Self::EaseInOutQuart => {
                if t < 0.5 { 8.0 * t * t * t * t } else { let u = t - 1.0; 1.0 - 8.0 * u * u * u * u }
            }
            Self::EaseInExpo => {
                if t == 0.0 { 0.0 } else { (2.0_f64).powf(10.0 * (t - 1.0)) }
            }
            Self::EaseOutExpo => {
                if t == 1.0 { 1.0 } else { 1.0 - (2.0_f64).powf(-10.0 * t) }
            }
            Self::EaseInOutExpo => {
                if t == 0.0 { return 0.0; }
                if t == 1.0 { return 1.0; }
                if t < 0.5 {
                    (2.0_f64).powf(20.0 * t - 10.0) / 2.0
                } else {
                    (2.0 - (2.0_f64).powf(-20.0 * t + 10.0)) / 2.0
                }
            }
            Self::EaseInBack => {
                let c = 1.70158;
                (c + 1.0) * t * t * t - c * t * t
            }
            Self::EaseOutBack => {
                let c = 1.70158;
                let u = t - 1.0;
                1.0 + (c + 1.0) * u * u * u + c * u * u
            }
            Self::EaseInOutBack => {
                let c = 1.70158 * 1.525;
                if t < 0.5 {
                    let u = 2.0 * t;
                    (u * u * ((c + 1.0) * u - c)) / 2.0
                } else {
                    let u = 2.0 * t - 2.0;
                    (u * u * ((c + 1.0) * u + c) + 2.0) / 2.0
                }
            }
            Self::EaseInBounce => 1.0 - Self::EaseOutBounce.apply(1.0 - t),
            Self::EaseOutBounce => bounce_out(t),
            Self::EaseInOutBounce => {
                if t < 0.5 {
                    (1.0 - bounce_out(1.0 - 2.0 * t)) / 2.0
                } else {
                    (1.0 + bounce_out(2.0 * t - 1.0)) / 2.0
                }
            }
            Self::EaseInElastic => {
                if t == 0.0 { return 0.0; }
                if t == 1.0 { return 1.0; }
                let p = 0.3;
                -(2.0_f64.powf(10.0 * (t - 1.0)) * ((t - 1.0 - p / 4.0) * 2.0 * PI / p).sin())
            }
            Self::EaseOutElastic => {
                if t == 0.0 { return 0.0; }
                if t == 1.0 { return 1.0; }
                let p = 0.3;
                2.0_f64.powf(-10.0 * t) * ((t - p / 4.0) * 2.0 * PI / p).sin() + 1.0
            }
            Self::EaseInOutElastic => {
                if t == 0.0 { return 0.0; }
                if t == 1.0 { return 1.0; }
                let p = 0.45;
                if t < 0.5 {
                    let u = 2.0 * t;
                    -0.5 * (2.0_f64.powf(10.0 * (u - 1.0)) * ((u - 1.0 - p / 4.0) * 2.0 * PI / p).sin())
                } else {
                    let u = 2.0 * t - 1.0;
                    2.0_f64.powf(-10.0 * u) * ((u - p / 4.0) * 2.0 * PI / p).sin() * 0.5 + 1.0
                }
            }
        }
    }
}

fn bounce_out(t: f64) -> f64 {
    let n1 = 7.5625;
    let d1 = 2.75;
    if t < 1.0 / d1 {
        n1 * t * t
    } else if t < 2.0 / d1 {
        let u = t - 1.5 / d1;
        n1 * u * u + 0.75
    } else if t < 2.5 / d1 {
        let u = t - 2.25 / d1;
        n1 * u * u + 0.9375
    } else {
        let u = t - 2.625 / d1;
        n1 * u * u + 0.984375
    }
}

// ── Tween State ────────────────────────────────────────────────

/// Snapshot of a tween at a point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweenState {
    /// Normalized progress in [0, 1].
    pub progress: f64,
    /// Interpolated value.
    pub value: f64,
    /// Whether the tween has finished.
    pub completed: bool,
}

// ── Single Tween ───────────────────────────────────────────────

/// A single value-to-value tween over time.
#[derive(Debug, Clone)]
pub struct Tween {
    pub from: f64,
    pub to: f64,
    pub duration_ms: f64,
    pub easing: TweenEasing,
    pub delay_ms: f64,
    elapsed_ms: f64,
    repeat_count: u32,
    current_repeat: u32,
    yoyo: bool,
    forward: bool,
}

impl Tween {
    pub fn new(from: f64, to: f64, duration_ms: f64) -> Self {
        Self {
            from,
            to,
            duration_ms,
            easing: TweenEasing::Linear,
            delay_ms: 0.0,
            elapsed_ms: 0.0,
            repeat_count: 1,
            current_repeat: 0,
            yoyo: false,
            forward: true,
        }
    }

    pub fn with_easing(mut self, easing: TweenEasing) -> Self {
        self.easing = easing;
        self
    }

    pub fn with_delay(mut self, delay_ms: f64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    pub fn with_repeat(mut self, count: u32) -> Self {
        self.repeat_count = count;
        self
    }

    pub fn with_yoyo(mut self, yoyo: bool) -> Self {
        self.yoyo = yoyo;
        self
    }

    /// Get the current state without advancing.
    pub fn state(&self) -> TweenState {
        let active_elapsed = (self.elapsed_ms - self.delay_ms).max(0.0);
        if active_elapsed <= 0.0 {
            return TweenState { progress: 0.0, value: self.from, completed: false };
        }

        let raw_progress = (active_elapsed / self.duration_ms).min(1.0);
        let progress = if self.forward { raw_progress } else { 1.0 - raw_progress };
        let eased = self.easing.apply(progress);
        let value = self.from + (self.to - self.from) * eased;

        let completed = self.current_repeat >= self.repeat_count
            && active_elapsed >= self.duration_ms;

        TweenState { progress, value, completed }
    }

    /// Advance by `dt_ms` milliseconds.
    pub fn tick(&mut self, dt_ms: f64) -> TweenState {
        self.elapsed_ms += dt_ms;

        let active_elapsed = (self.elapsed_ms - self.delay_ms).max(0.0);

        if active_elapsed >= self.duration_ms && self.current_repeat < self.repeat_count {
            self.current_repeat += 1;
            if self.current_repeat < self.repeat_count {
                self.elapsed_ms = self.delay_ms;
                if self.yoyo {
                    self.forward = !self.forward;
                }
            }
        }

        self.state()
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.elapsed_ms = 0.0;
        self.current_repeat = 0;
        self.forward = true;
    }

    /// Whether the tween is finished.
    pub fn is_complete(&self) -> bool {
        self.state().completed
    }
}

// ── Sequence ───────────────────────────────────────────────────

/// Plays tweens one after another.
#[derive(Debug, Clone)]
pub struct TweenSequence {
    tweens: Vec<Tween>,
    current: usize,
}

impl TweenSequence {
    pub fn new(tweens: Vec<Tween>) -> Self {
        Self { tweens, current: 0 }
    }

    pub fn tick(&mut self, dt_ms: f64) -> Option<TweenState> {
        if self.current >= self.tweens.len() {
            return None;
        }
        let state = self.tweens[self.current].tick(dt_ms);
        if state.completed {
            self.current += 1;
        }
        Some(state)
    }

    pub fn is_complete(&self) -> bool {
        self.current >= self.tweens.len()
    }

    pub fn current_index(&self) -> usize {
        self.current
    }

    pub fn reset(&mut self) {
        self.current = 0;
        for t in &mut self.tweens {
            t.reset();
        }
    }
}

// ── Parallel ───────────────────────────────────────────────────

/// Plays tweens simultaneously.
#[derive(Debug, Clone)]
pub struct TweenParallel {
    tweens: Vec<Tween>,
}

impl TweenParallel {
    pub fn new(tweens: Vec<Tween>) -> Self {
        Self { tweens }
    }

    /// Tick all tweens, return all their states.
    pub fn tick(&mut self, dt_ms: f64) -> Vec<TweenState> {
        self.tweens.iter_mut().map(|t| t.tick(dt_ms)).collect()
    }

    pub fn is_complete(&self) -> bool {
        self.tweens.iter().all(|t| t.is_complete())
    }

    pub fn reset(&mut self) {
        for t in &mut self.tweens {
            t.reset();
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_tween_basic() {
        let mut tw = Tween::new(0.0, 100.0, 1000.0);
        tw.tick(500.0);
        let s = tw.state();
        assert!((s.value - 50.0).abs() < 0.01);
        assert!((s.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn tween_completes() {
        let mut tw = Tween::new(0.0, 100.0, 1000.0);
        tw.tick(1100.0);
        assert!(tw.is_complete());
        assert!((tw.state().value - 100.0).abs() < 0.01);
    }

    #[test]
    fn tween_with_delay() {
        let mut tw = Tween::new(0.0, 100.0, 1000.0).with_delay(500.0);
        tw.tick(250.0);
        assert!((tw.state().value - 0.0).abs() < 0.01, "Still in delay period");
        tw.tick(750.0); // now 500ms of delay + 500ms active
        assert!((tw.state().value - 50.0).abs() < 0.01);
    }

    #[test]
    fn ease_in_quad() {
        let e = TweenEasing::EaseInQuad;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-10);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-10);
        // At t=0.5, quad in = 0.25
        assert!((e.apply(0.5) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn ease_out_bounce_endpoints() {
        let e = TweenEasing::EaseOutBounce;
        assert!((e.apply(0.0) - 0.0).abs() < 1e-10);
        assert!((e.apply(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn elastic_oscillates() {
        let e = TweenEasing::EaseOutElastic;
        // Elastic overshoots past 1.0 at some points.
        let mut found_overshoot = false;
        for i in 1..100 {
            let t = i as f64 / 100.0;
            if e.apply(t) > 1.0 {
                found_overshoot = true;
                break;
            }
        }
        assert!(found_overshoot, "Elastic should overshoot");
    }

    #[test]
    fn back_overshoots() {
        let e = TweenEasing::EaseInBack;
        // EaseInBack goes negative initially.
        assert!(e.apply(0.1) < 0.0);
    }

    #[test]
    fn repeat_works() {
        let mut tw = Tween::new(0.0, 100.0, 100.0).with_repeat(3);
        for _ in 0..3 {
            tw.tick(100.0);
        }
        assert!(tw.is_complete());
    }

    #[test]
    fn yoyo_reverses() {
        let mut tw = Tween::new(0.0, 100.0, 100.0).with_repeat(2).with_yoyo(true);
        // First iteration: forward
        tw.tick(100.0);
        // After completing first iteration and starting second, it should be reversed.
        let s = tw.tick(50.0);
        // In yoyo mode second iteration goes backward, so value should be heading toward 0.
        assert!(s.value < 100.0);
    }

    #[test]
    fn sequence_plays_in_order() {
        let t1 = Tween::new(0.0, 50.0, 100.0);
        let t2 = Tween::new(50.0, 100.0, 100.0);
        let mut seq = TweenSequence::new(vec![t1, t2]);

        // Finish first tween.
        seq.tick(100.0);
        assert_eq!(seq.current_index(), 1);

        // Finish second tween.
        seq.tick(100.0);
        assert!(seq.is_complete());
    }

    #[test]
    fn parallel_plays_simultaneously() {
        let t1 = Tween::new(0.0, 100.0, 100.0);
        let t2 = Tween::new(0.0, 200.0, 100.0);
        let mut par = TweenParallel::new(vec![t1, t2]);

        let states = par.tick(50.0);
        assert_eq!(states.len(), 2);
        assert!((states[0].value - 50.0).abs() < 0.01);
        assert!((states[1].value - 100.0).abs() < 0.01);
    }

    #[test]
    fn all_easings_are_bounded() {
        let easings = [
            TweenEasing::Linear, TweenEasing::EaseInQuad, TweenEasing::EaseOutQuad,
            TweenEasing::EaseInOutQuad, TweenEasing::EaseInCubic, TweenEasing::EaseOutCubic,
            TweenEasing::EaseInOutCubic, TweenEasing::EaseInQuart, TweenEasing::EaseOutQuart,
            TweenEasing::EaseInOutQuart, TweenEasing::EaseInExpo, TweenEasing::EaseOutExpo,
            TweenEasing::EaseInOutExpo, TweenEasing::EaseOutBounce, TweenEasing::EaseInBounce,
            TweenEasing::EaseInOutBounce,
        ];
        for e in &easings {
            assert!((e.apply(0.0) - 0.0).abs() < 1e-9, "{e:?} at t=0");
            assert!((e.apply(1.0) - 1.0).abs() < 1e-9, "{e:?} at t=1");
        }
    }

    #[test]
    fn reset_restarts_tween() {
        let mut tw = Tween::new(0.0, 100.0, 100.0);
        tw.tick(100.0);
        assert!(tw.is_complete());
        tw.reset();
        assert!(!tw.is_complete());
        assert!((tw.state().value - 0.0).abs() < 0.01);
    }
}
