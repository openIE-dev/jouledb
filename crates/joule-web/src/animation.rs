//! Animation engine: easing, keyframes, tracks, timelines, and spring physics.
//!
//! Replaces Framer Motion / GSAP / anime.js. Pure math — no browser
//! dependency. All interpolation happens on `f64` values; mapping to
//! CSS properties is done at the rendering layer.

use std::collections::HashMap;

// ── Easing ──────────────────────────────────────────────────────

/// Easing functions for animation interpolation.
#[derive(Debug, Clone, PartialEq)]
pub enum Easing {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    CubicBezier(f64, f64, f64, f64),
    Spring { stiffness: f64, damping: f64, mass: f64 },
    Steps(u32),
}

impl Easing {
    /// Map a normalized time `t` in [0, 1] to an eased value.
    pub fn interpolate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::EaseIn => t * t * t,
            Easing::EaseOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            Easing::EaseInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let v = -2.0 * t + 2.0;
                    1.0 - v * v * v / 2.0
                }
            }
            Easing::CubicBezier(x1, y1, x2, y2) => {
                cubic_bezier_sample(*x1, *y1, *x2, *y2, t)
            }
            Easing::Spring { stiffness, damping, mass } => {
                let time = t * 4.0;
                1.0 - spring_position(time, *stiffness, *damping, *mass)
            }
            Easing::Steps(n) => {
                if *n == 0 { return t; }
                let nf = *n as f64;
                (t * nf).floor() / nf
            }
        }
    }
}

/// Solve a cubic bezier curve using Newton's method (5 iterations).
fn cubic_bezier_sample(x1: f64, y1: f64, x2: f64, y2: f64, x: f64) -> f64 {
    let mut t = x;
    for _ in 0..5 {
        let bx = bezier_component(x1, x2, t);
        let dx = bezier_derivative(x1, x2, t);
        if dx.abs() < 1e-12 { break; }
        t -= (bx - x) / dx;
        t = t.clamp(0.0, 1.0);
    }
    bezier_component(y1, y2, t)
}

fn bezier_component(p1: f64, p2: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    3.0 * p1 * t * (1.0 - t).powi(2) + 3.0 * p2 * t2 * (1.0 - t) + t3
}

fn bezier_derivative(p1: f64, p2: f64, t: f64) -> f64 {
    let t2 = t * t;
    3.0 * p1 * (1.0 - t).powi(2) + 6.0 * (p2 - p1) * t * (1.0 - t) + 3.0 * (1.0 - p2) * t2
}

/// Damped harmonic oscillator: x'' + (c/m)x' + (k/m)x = 0
/// with x(0) = 1, x'(0) = 0.
pub fn spring_position(t: f64, stiffness: f64, damping: f64, mass: f64) -> f64 {
    let omega0 = (stiffness / mass).sqrt();
    let zeta = damping / (2.0 * (stiffness * mass).sqrt());

    if zeta < 1.0 {
        // Underdamped
        let omega_d = omega0 * (1.0 - zeta * zeta).sqrt();
        let decay = (-zeta * omega0 * t).exp();
        decay * ((zeta * omega0 / omega_d) * (omega_d * t).sin() + (omega_d * t).cos())
    } else if (zeta - 1.0).abs() < 1e-10 {
        // Critically damped
        let decay = (-omega0 * t).exp();
        decay * (1.0 + omega0 * t)
    } else {
        // Overdamped
        let s1 = -omega0 * (zeta + (zeta * zeta - 1.0).sqrt());
        let s2 = -omega0 * (zeta - (zeta * zeta - 1.0).sqrt());
        let a = s1 / (s1 - s2);
        let b = -s2 / (s1 - s2);
        a * (s2 * t).exp() + b * (s1 * t).exp()
    }
}

// ── Keyframe ────────────────────────────────────────────────────

/// A single keyframe: a value at a normalized time with an easing curve.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyframe {
    pub time: f64,
    pub value: f64,
    pub easing: Easing,
}

// ── Track ───────────────────────────────────────────────────────

/// An animation track controlling a single named property via keyframes.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub property_name: String,
    pub keyframes: Vec<Keyframe>,
}

impl Track {
    pub fn new(property: &str) -> Self {
        Self { property_name: property.to_string(), keyframes: Vec::new() }
    }

    pub fn keyframe(mut self, time: f64, value: f64, easing: Easing) -> Self {
        self.keyframes.push(Keyframe { time, value, easing });
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        self
    }

    /// Sample the track at normalized time `t` in [0.0, 1.0].
    pub fn sample(&self, t: f64) -> f64 {
        if self.keyframes.is_empty() { return 0.0; }
        let t = t.clamp(0.0, 1.0);

        if t <= self.keyframes[0].time { return self.keyframes[0].value; }
        let last = self.keyframes.len() - 1;
        if t >= self.keyframes[last].time { return self.keyframes[last].value; }

        for i in 0..last {
            let kf0 = &self.keyframes[i];
            let kf1 = &self.keyframes[i + 1];
            if t >= kf0.time && t <= kf1.time {
                let span = kf1.time - kf0.time;
                if span.abs() < f64::EPSILON { return kf1.value; }
                let local_t = (t - kf0.time) / span;
                let eased = kf1.easing.interpolate(local_t);
                return kf0.value + (kf1.value - kf0.value) * eased;
            }
        }
        self.keyframes[last].value
    }
}

// ── Animation ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RepeatMode { Once, Count(u32), Infinite }

#[derive(Debug, Clone, PartialEq)]
pub enum Direction { Normal, Reverse, Alternate }

#[derive(Debug, Clone, PartialEq)]
pub enum AnimState { Idle, Playing, Paused, Finished }

/// A multi-track animation with timing, repeat, and direction.
#[derive(Debug, Clone)]
pub struct Animation {
    pub tracks: Vec<Track>,
    pub duration_ms: u64,
    pub delay_ms: u64,
    pub repeat: RepeatMode,
    pub direction: Direction,
    pub state: AnimState,
    elapsed_ms: u64,
    iteration: u32,
}

impl Animation {
    pub fn new(duration_ms: u64) -> Self {
        Self {
            tracks: Vec::new(), duration_ms, delay_ms: 0,
            repeat: RepeatMode::Once, direction: Direction::Normal,
            state: AnimState::Idle, elapsed_ms: 0, iteration: 0,
        }
    }

    pub fn track(mut self, track: Track) -> Self { self.tracks.push(track); self }
    pub fn delay(mut self, ms: u64) -> Self { self.delay_ms = ms; self }
    pub fn repeat(mut self, mode: RepeatMode) -> Self { self.repeat = mode; self }
    pub fn direction(mut self, dir: Direction) -> Self { self.direction = dir; self }

    pub fn play(&mut self) { self.state = AnimState::Playing; }
    pub fn pause(&mut self) { if self.state == AnimState::Playing { self.state = AnimState::Paused; } }
    pub fn reset(&mut self) { self.elapsed_ms = 0; self.iteration = 0; self.state = AnimState::Idle; }
    pub fn is_finished(&self) -> bool { self.state == AnimState::Finished }

    pub fn progress(&self) -> f64 {
        if self.duration_ms == 0 { return 1.0; }
        let after_delay = self.elapsed_ms.saturating_sub(self.delay_ms);
        (after_delay as f64 / self.duration_ms as f64).clamp(0.0, 1.0)
    }

    /// Advance the animation and return current track values.
    pub fn tick(&mut self, elapsed_ms: u64) -> HashMap<String, f64> {
        if self.state != AnimState::Playing { return self.sample_at(self.progress()); }

        self.elapsed_ms += elapsed_ms;
        if self.elapsed_ms < self.delay_ms { return self.sample_at(0.0); }

        let after_delay = self.elapsed_ms - self.delay_ms;
        let cycles = if self.duration_ms > 0 { after_delay / self.duration_ms } else { 0 };

        let max_iterations = match self.repeat {
            RepeatMode::Once => 1,
            RepeatMode::Count(n) => n as u64,
            RepeatMode::Infinite => u64::MAX,
        };

        if cycles >= max_iterations {
            self.state = AnimState::Finished;
            self.iteration = max_iterations.min(u32::MAX as u64) as u32;
            return self.sample_at(1.0);
        }

        self.iteration = cycles as u32;
        let within_cycle = if self.duration_ms > 0 {
            (after_delay % self.duration_ms) as f64 / self.duration_ms as f64
        } else { 1.0 };

        let t = match self.direction {
            Direction::Normal => within_cycle,
            Direction::Reverse => 1.0 - within_cycle,
            Direction::Alternate => {
                if cycles % 2 == 0 { within_cycle } else { 1.0 - within_cycle }
            }
        };

        self.sample_at(t)
    }

    fn sample_at(&self, t: f64) -> HashMap<String, f64> {
        self.tracks.iter().map(|track| (track.property_name.clone(), track.sample(t))).collect()
    }
}

// ── Timeline ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TimelineEntry {
    pub animation: Animation,
    pub start_ms: u64,
}

/// A sequencer for staggering multiple animations.
#[derive(Debug, Clone)]
pub struct Timeline {
    pub entries: Vec<TimelineEntry>,
    elapsed_ms: u64,
}

impl Default for Timeline {
    fn default() -> Self { Self::new() }
}

impl Timeline {
    pub fn new() -> Self { Self { entries: Vec::new(), elapsed_ms: 0 } }

    pub fn add(mut self, animation: Animation, start_ms: u64) -> Self {
        self.entries.push(TimelineEntry { animation, start_ms });
        self
    }

    pub fn total_duration_ms(&self) -> u64 {
        self.entries.iter()
            .map(|e| e.start_ms + e.animation.delay_ms + e.animation.duration_ms)
            .max().unwrap_or(0)
    }

    pub fn is_finished(&self) -> bool {
        self.entries.iter().all(|e| e.animation.is_finished())
    }

    pub fn tick(&mut self, elapsed_ms: u64) -> Vec<(usize, HashMap<String, f64>)> {
        self.elapsed_ms += elapsed_ms;
        let mut results = Vec::new();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if self.elapsed_ms < entry.start_ms { continue; }
            if entry.animation.state == AnimState::Idle { entry.animation.play(); }
            let local_elapsed = if self.elapsed_ms >= entry.start_ms + elapsed_ms {
                elapsed_ms
            } else {
                self.elapsed_ms - entry.start_ms
            };
            let values = entry.animation.tick(local_elapsed);
            results.push((i, values));
        }
        results
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_easing_identity() {
        let e = Easing::Linear;
        assert!((e.interpolate(0.0) - 0.0).abs() < 1e-10);
        assert!((e.interpolate(0.5) - 0.5).abs() < 1e-10);
        assert!((e.interpolate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ease_in_starts_slow() {
        let e = Easing::EaseIn;
        let v = e.interpolate(0.25);
        assert!(v < 0.1, "ease-in at 0.25 should be < 0.1, got {v}");
        assert!((e.interpolate(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ease_out_ends_fast() {
        let e = Easing::EaseOut;
        let v = e.interpolate(0.75);
        assert!(v > 0.9, "ease-out at 0.75 should be > 0.9, got {v}");
    }

    #[test]
    fn ease_in_out_symmetric() {
        let e = Easing::EaseInOut;
        let v_mid = e.interpolate(0.5);
        assert!((v_mid - 0.5).abs() < 1e-10, "midpoint should be 0.5, got {v_mid}");
    }

    #[test]
    fn cubic_bezier_matches_known_values() {
        let e = Easing::CubicBezier(0.25, 0.1, 0.25, 1.0);
        let v0 = e.interpolate(0.0);
        let v1 = e.interpolate(1.0);
        assert!((v0 - 0.0).abs() < 0.01);
        assert!((v1 - 1.0).abs() < 0.01);
        let v_mid = e.interpolate(0.5);
        assert!(v_mid > 0.3 && v_mid < 0.9, "mid={v_mid}");
    }

    #[test]
    fn spring_settles_near_zero() {
        let pos = spring_position(10.0, 100.0, 10.0, 1.0);
        assert!(pos.abs() < 0.01, "spring should settle near 0, got {pos}");
    }

    #[test]
    fn steps_easing_quantizes() {
        let e = Easing::Steps(4);
        assert!((e.interpolate(0.0) - 0.0).abs() < 1e-10);
        assert!((e.interpolate(0.24) - 0.0).abs() < 1e-10);
        assert!((e.interpolate(0.26) - 0.25).abs() < 1e-10);
        assert!((e.interpolate(0.5) - 0.5).abs() < 1e-10);
        assert!((e.interpolate(0.99) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn keyframe_sampling_at_boundaries() {
        let track = Track::new("opacity")
            .keyframe(0.0, 0.0, Easing::Linear)
            .keyframe(1.0, 1.0, Easing::Linear);
        assert!((track.sample(0.0) - 0.0).abs() < 1e-10);
        assert!((track.sample(1.0) - 1.0).abs() < 1e-10);
        assert!((track.sample(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn track_with_multiple_keyframes_interpolates() {
        let track = Track::new("x")
            .keyframe(0.0, 0.0, Easing::Linear)
            .keyframe(0.5, 100.0, Easing::Linear)
            .keyframe(1.0, 50.0, Easing::Linear);
        assert!((track.sample(0.25) - 50.0).abs() < 1e-6);
        assert!((track.sample(0.5) - 100.0).abs() < 1e-6);
        assert!((track.sample(0.75) - 75.0).abs() < 1e-6);
    }

    #[test]
    fn animation_tick_returns_values() {
        let track = Track::new("opacity")
            .keyframe(0.0, 0.0, Easing::Linear)
            .keyframe(1.0, 1.0, Easing::Linear);
        let mut anim = Animation::new(1000).track(track);
        anim.play();
        let vals = anim.tick(500);
        let opacity = vals["opacity"];
        assert!((opacity - 0.5).abs() < 0.05, "expected ~0.5, got {opacity}");
    }

    #[test]
    fn animation_repeat_loops() {
        let track = Track::new("v")
            .keyframe(0.0, 0.0, Easing::Linear)
            .keyframe(1.0, 1.0, Easing::Linear);
        let mut anim = Animation::new(100).track(track).repeat(RepeatMode::Count(3));
        anim.play();
        anim.tick(50);
        assert!(!anim.is_finished());
        anim.tick(250);
        assert!(anim.is_finished());
    }

    #[test]
    fn alternate_direction_reverses() {
        let track = Track::new("x")
            .keyframe(0.0, 0.0, Easing::Linear)
            .keyframe(1.0, 100.0, Easing::Linear);
        let mut anim = Animation::new(100)
            .track(track)
            .direction(Direction::Alternate)
            .repeat(RepeatMode::Count(2));
        anim.play();
        let v1 = anim.tick(50);
        let x1 = v1["x"];
        assert!(x1 > 30.0 && x1 < 70.0, "forward mid: {x1}");
        let v2 = anim.tick(100);
        let x2 = v2["x"];
        assert!(x2 > 30.0 && x2 < 70.0, "reverse mid: {x2}");
    }

    #[test]
    fn timeline_staggers_animations() {
        let t1 = Track::new("a").keyframe(0.0, 0.0, Easing::Linear).keyframe(1.0, 1.0, Easing::Linear);
        let t2 = Track::new("b").keyframe(0.0, 0.0, Easing::Linear).keyframe(1.0, 1.0, Easing::Linear);
        let a1 = Animation::new(100).track(t1);
        let a2 = Animation::new(100).track(t2);
        let mut timeline = Timeline::new().add(a1, 0).add(a2, 200);
        let r1 = timeline.tick(50);
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].0, 0);
        let r2 = timeline.tick(200);
        assert!(r2.len() >= 1);
        assert_eq!(timeline.total_duration_ms(), 300);
    }

    #[test]
    fn animation_delay() {
        let track = Track::new("v")
            .keyframe(0.0, 0.0, Easing::Linear)
            .keyframe(1.0, 100.0, Easing::Linear);
        let mut anim = Animation::new(100).track(track).delay(50);
        anim.play();
        let v1 = anim.tick(25);
        assert!((v1["v"] - 0.0).abs() < 1e-6);
        let v2 = anim.tick(50);
        let val = v2["v"];
        assert!(val > 10.0 && val < 40.0, "after delay: {val}");
    }
}
