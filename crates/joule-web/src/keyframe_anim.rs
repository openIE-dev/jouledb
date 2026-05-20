//! CSS `@keyframes`-style animation engine.
//!
//! Replaces CSS animations / Web Animations API for headless rendering.
//! Multi-property keyframes with offset-based interpolation, iteration
//! counts, direction modes, fill modes, and pause/resume support.

use std::collections::HashMap;

// ── Types ──────────────────────────────────────────────────────

/// Direction of playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationDirection {
    Normal,
    Reverse,
    Alternate,
    AlternateReverse,
}

/// Fill mode — what happens before/after the animation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    None,
    Forwards,
    Backwards,
    Both,
}

/// Iteration count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IterationCount {
    Finite(u32),
    Infinite,
}

/// Events emitted during animation playback.
#[derive(Debug, Clone, PartialEq)]
pub enum AnimationEvent {
    Start,
    Iteration(u32),
    End,
}

/// A single keyframe at a given offset [0, 1].
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Offset in [0.0, 1.0] — equivalent to CSS keyframe percentage.
    pub offset: f64,
    /// Property name → value.
    pub properties: HashMap<String, f64>,
}

impl Keyframe {
    pub fn new(offset: f64) -> Self {
        Self {
            offset: offset.clamp(0.0, 1.0),
            properties: HashMap::new(),
        }
    }

    pub fn with_property(mut self, name: impl Into<String>, value: f64) -> Self {
        self.properties.insert(name.into(), value);
        self
    }

    pub fn set(&mut self, name: impl Into<String>, value: f64) {
        self.properties.insert(name.into(), value);
    }
}

// ── Animation ──────────────────────────────────────────────────

/// A keyframe animation definition and playback state.
#[derive(Debug, Clone)]
pub struct KeyframeAnimation {
    keyframes: Vec<Keyframe>,
    duration_ms: f64,
    iteration_count: IterationCount,
    direction: AnimationDirection,
    fill_mode: FillMode,
    elapsed_ms: f64,
    current_iteration: u32,
    paused: bool,
    started: bool,
    finished: bool,
    events: Vec<AnimationEvent>,
}

impl KeyframeAnimation {
    pub fn new(mut keyframes: Vec<Keyframe>, duration_ms: f64) -> Self {
        keyframes.sort_by(|a, b| a.offset.partial_cmp(&b.offset).unwrap());
        Self {
            keyframes,
            duration_ms,
            iteration_count: IterationCount::Finite(1),
            direction: AnimationDirection::Normal,
            fill_mode: FillMode::None,
            elapsed_ms: 0.0,
            current_iteration: 0,
            paused: false,
            started: false,
            finished: false,
            events: Vec::new(),
        }
    }

    pub fn with_iteration_count(mut self, count: IterationCount) -> Self {
        self.iteration_count = count;
        self
    }

    pub fn with_direction(mut self, dir: AnimationDirection) -> Self {
        self.direction = dir;
        self
    }

    pub fn with_fill_mode(mut self, fill: FillMode) -> Self {
        self.fill_mode = fill;
        self
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    pub fn current_iteration(&self) -> u32 {
        self.current_iteration
    }

    /// Drain pending events.
    pub fn take_events(&mut self) -> Vec<AnimationEvent> {
        std::mem::take(&mut self.events)
    }

    /// Reset the animation to the beginning.
    pub fn reset(&mut self) {
        self.elapsed_ms = 0.0;
        self.current_iteration = 0;
        self.paused = false;
        self.started = false;
        self.finished = false;
        self.events.clear();
    }

    /// Advance the animation by `dt_ms` and return interpolated properties.
    pub fn tick(&mut self, dt_ms: f64) -> HashMap<String, f64> {
        if self.finished {
            return self.sample_at_fill();
        }
        if self.paused {
            return self.sample_at_progress(self.current_progress());
        }

        if !self.started {
            self.started = true;
            self.events.push(AnimationEvent::Start);
        }

        self.elapsed_ms += dt_ms;

        // Determine which iteration we're on.
        let iter_progress = self.elapsed_ms / self.duration_ms;
        let max_iters = match self.iteration_count {
            IterationCount::Finite(n) => n,
            IterationCount::Infinite => u32::MAX,
        };

        if iter_progress >= 1.0 && self.current_iteration < max_iters {
            let completed = (iter_progress.floor() as u32).min(max_iters - self.current_iteration);
            let target_iter = self.current_iteration + completed;
            while self.current_iteration < target_iter && self.current_iteration < max_iters {
                self.current_iteration += 1;
                if self.current_iteration < max_iters {
                    self.events.push(AnimationEvent::Iteration(self.current_iteration));
                }
                self.elapsed_ms -= self.duration_ms;
            }
        }

        if self.current_iteration >= max_iters {
            self.finished = true;
            self.events.push(AnimationEvent::End);
            return self.sample_at_fill();
        }

        let progress = self.current_progress();
        self.sample_at_progress(progress)
    }

    fn current_progress(&self) -> f64 {
        let raw = (self.elapsed_ms / self.duration_ms).clamp(0.0, 1.0);

        let is_reverse = match self.direction {
            AnimationDirection::Normal => false,
            AnimationDirection::Reverse => true,
            AnimationDirection::Alternate => self.current_iteration % 2 == 1,
            AnimationDirection::AlternateReverse => self.current_iteration % 2 == 0,
        };

        if is_reverse { 1.0 - raw } else { raw }
    }

    fn sample_at_fill(&self) -> HashMap<String, f64> {
        match self.fill_mode {
            FillMode::None => HashMap::new(),
            FillMode::Forwards | FillMode::Both => {
                let end_progress = match self.direction {
                    AnimationDirection::Normal => 1.0,
                    AnimationDirection::Reverse => 0.0,
                    AnimationDirection::Alternate => {
                        let max = match self.iteration_count {
                            IterationCount::Finite(n) => n,
                            IterationCount::Infinite => 0,
                        };
                        if max % 2 == 0 { 0.0 } else { 1.0 }
                    }
                    AnimationDirection::AlternateReverse => {
                        let max = match self.iteration_count {
                            IterationCount::Finite(n) => n,
                            IterationCount::Infinite => 0,
                        };
                        if max % 2 == 0 { 1.0 } else { 0.0 }
                    }
                };
                self.sample_at_progress(end_progress)
            }
            FillMode::Backwards => self.sample_at_progress(0.0),
        }
    }

    /// Interpolate all properties at a given normalized progress [0, 1].
    fn sample_at_progress(&self, progress: f64) -> HashMap<String, f64> {
        if self.keyframes.is_empty() {
            return HashMap::new();
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].properties.clone();
        }

        // Find the two surrounding keyframes.
        let mut before_idx = 0;
        let mut after_idx = self.keyframes.len() - 1;

        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.offset <= progress {
                before_idx = i;
            }
            if kf.offset >= progress && i < after_idx {
                after_idx = i;
                break;
            }
        }

        let before = &self.keyframes[before_idx];
        let after = &self.keyframes[after_idx];

        if before_idx == after_idx {
            return before.properties.clone();
        }

        let range = after.offset - before.offset;
        let local_t = if range > 0.0 {
            ((progress - before.offset) / range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Collect all property names from both keyframes.
        let mut result = HashMap::new();
        let all_keys: Vec<String> = before.properties.keys()
            .chain(after.properties.keys())
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for key in all_keys {
            let v_before = before.properties.get(&key).copied();
            let v_after = after.properties.get(&key).copied();
            match (v_before, v_after) {
                (Some(a), Some(b)) => { result.insert(key, a + (b - a) * local_t); }
                (Some(a), None) => { result.insert(key, a); }
                (None, Some(b)) => { result.insert(key, b); }
                (None, None) => {}
            }
        }

        result
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_anim() -> KeyframeAnimation {
        let kf0 = Keyframe::new(0.0)
            .with_property("opacity", 0.0)
            .with_property("x", 0.0);
        let kf1 = Keyframe::new(1.0)
            .with_property("opacity", 1.0)
            .with_property("x", 100.0);
        KeyframeAnimation::new(vec![kf0, kf1], 1000.0)
    }

    #[test]
    fn interpolates_at_midpoint() {
        let mut anim = simple_anim();
        let props = anim.tick(500.0);
        assert!((props["opacity"] - 0.5).abs() < 0.01);
        assert!((props["x"] - 50.0).abs() < 0.01);
    }

    #[test]
    fn animation_start_event() {
        let mut anim = simple_anim();
        anim.tick(100.0);
        let events = anim.take_events();
        assert!(events.contains(&AnimationEvent::Start));
    }

    #[test]
    fn animation_end_event() {
        let mut anim = simple_anim();
        anim.tick(1100.0);
        let events = anim.take_events();
        assert!(events.contains(&AnimationEvent::End));
        assert!(anim.is_finished());
    }

    #[test]
    fn multiple_keyframes() {
        let kf0 = Keyframe::new(0.0).with_property("x", 0.0);
        let kf50 = Keyframe::new(0.5).with_property("x", 200.0);
        let kf100 = Keyframe::new(1.0).with_property("x", 100.0);
        let mut anim = KeyframeAnimation::new(vec![kf0, kf50, kf100], 1000.0);

        let props = anim.tick(250.0);
        // 25% progress is between kf0 (0.0) and kf50 (0.5), local_t = 0.5
        assert!((props["x"] - 100.0).abs() < 0.1);
    }

    #[test]
    fn pause_resume() {
        let mut anim = simple_anim();
        anim.tick(300.0);
        anim.pause();
        assert!(anim.is_paused());

        let props1 = anim.tick(500.0); // Should not advance.
        assert!((props1["opacity"] - 0.3).abs() < 0.01);

        anim.resume();
        let props2 = anim.tick(200.0);
        // Now we advance 200ms from 300ms = 500ms total.
        assert!((props2["opacity"] - 0.5).abs() < 0.01);
    }

    #[test]
    fn reverse_direction() {
        let mut anim = simple_anim().with_direction(AnimationDirection::Reverse);
        let props = anim.tick(250.0);
        // Reverse: progress = 1 - 0.25 = 0.75
        assert!((props["opacity"] - 0.75).abs() < 0.01);
    }

    #[test]
    fn finite_iterations() {
        let mut anim = simple_anim()
            .with_iteration_count(IterationCount::Finite(3));
        // Run 3 full iterations.
        for _ in 0..3 {
            anim.tick(1000.0);
        }
        assert!(anim.is_finished());
    }

    #[test]
    fn alternate_direction() {
        let mut anim = simple_anim()
            .with_iteration_count(IterationCount::Finite(3))
            .with_direction(AnimationDirection::Alternate);

        // First iteration: normal.
        let p1 = anim.tick(500.0);
        assert!((p1["opacity"] - 0.5).abs() < 0.01);

        // Complete first iteration.
        anim.tick(500.0);

        // Second iteration: reverse.
        let p2 = anim.tick(500.0);
        assert!((p2["opacity"] - 0.5).abs() < 0.01);
    }

    #[test]
    fn fill_mode_forwards() {
        let mut anim = simple_anim().with_fill_mode(FillMode::Forwards);
        anim.tick(1500.0);
        assert!(anim.is_finished());
        // With forwards fill, the final values should persist.
        let props = anim.tick(0.0);
        assert!((props["opacity"] - 1.0).abs() < 0.01);
    }

    #[test]
    fn fill_mode_none_returns_empty() {
        let mut anim = simple_anim().with_fill_mode(FillMode::None);
        anim.tick(1500.0);
        assert!(anim.is_finished());
        let props = anim.tick(0.0);
        assert!(props.is_empty());
    }

    #[test]
    fn reset_restarts() {
        let mut anim = simple_anim();
        anim.tick(1500.0);
        assert!(anim.is_finished());
        anim.reset();
        assert!(!anim.is_finished());
        assert!(!anim.is_paused());
        let props = anim.tick(0.0);
        assert!((props["opacity"] - 0.0).abs() < 0.01);
    }

    #[test]
    fn single_keyframe() {
        let kf = Keyframe::new(0.5).with_property("scale", 2.0);
        let mut anim = KeyframeAnimation::new(vec![kf], 1000.0);
        let props = anim.tick(500.0);
        assert!((props["scale"] - 2.0).abs() < 0.01);
    }
}
