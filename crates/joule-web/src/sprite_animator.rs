//! Sprite animation controller: clips, playback modes, blending, events,
//! and an animation state machine with configurable transitions.
//!
//! Pure logic — produces frame indices and event signals; the rendering
//! layer maps frame indices to sprite sheet regions.

use std::collections::HashMap;

// ── Playback Modes ─────────────────────────────────────────────

/// How an animation clip plays back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackMode {
    /// Play once and stop on the last frame.
    Once,
    /// Loop from start after reaching end.
    Loop,
    /// Play forward then backward, repeating.
    PingPong,
    /// Play in reverse, once.
    Reverse,
}

// ── Animation Clip ─────────────────────────────────────────────

/// A single frame in an animation clip.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameDef {
    pub frame_index: u32,
    pub duration_secs: f64,
}

/// An animation clip — a named sequence of frames.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimationClip {
    pub name: String,
    pub frames: Vec<FrameDef>,
    pub playback: PlaybackMode,
}

impl AnimationClip {
    pub fn new(name: &str, frames: Vec<FrameDef>, playback: PlaybackMode) -> Self {
        Self { name: name.to_string(), frames, playback }
    }

    /// Total duration of one forward pass.
    pub fn duration(&self) -> f64 {
        self.frames.iter().map(|f| f.duration_secs).sum()
    }
}

// ── Animation Events ───────────────────────────────────────────

/// An event triggered at a specific frame of a clip.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimEvent {
    pub clip_name: String,
    pub frame_position: u32,
    pub event_name: String,
}

// ── Playback State ─────────────────────────────────────────────

/// The internal state of a playing animation.
#[derive(Debug, Clone, PartialEq)]
struct PlaybackState {
    clip_name: String,
    elapsed: f64,
    speed: f64,
    finished: bool,
    /// For ping-pong: true = forward, false = backward
    forward: bool,
}

impl PlaybackState {
    fn new(clip_name: &str, speed: f64) -> Self {
        Self {
            clip_name: clip_name.to_string(),
            elapsed: 0.0,
            speed,
            finished: false,
            forward: true,
        }
    }
}

// ── Crossfade / Blend ──────────────────────────────────────────

/// Active crossfade between two clips.
#[derive(Debug, Clone, PartialEq)]
struct Crossfade {
    from_clip: String,
    from_elapsed: f64,
    blend_duration: f64,
    blend_elapsed: f64,
}

// ── State Machine Transition ───────────────────────────────────

/// A transition rule in the animation state machine.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub from_state: String,
    pub to_state: String,
    pub trigger: String,
    pub blend_duration: f64,
}

// ── Sprite Animator ────────────────────────────────────────────

/// The main sprite animation controller.
#[derive(Debug, Clone)]
pub struct SpriteAnimator {
    clips: HashMap<String, AnimationClip>,
    events: Vec<AnimEvent>,
    transitions: Vec<Transition>,
    current: Option<PlaybackState>,
    crossfade: Option<Crossfade>,
    default_speed: f64,
    /// Accumulated fired events from the last update.
    last_fired: Vec<String>,
}

impl SpriteAnimator {
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
            events: Vec::new(),
            transitions: Vec::new(),
            current: None,
            crossfade: None,
            default_speed: 1.0,
            last_fired: Vec::new(),
        }
    }

    pub fn add_clip(&mut self, clip: AnimationClip) {
        self.clips.insert(clip.name.clone(), clip);
    }

    pub fn add_event(&mut self, event: AnimEvent) {
        self.events.push(event);
    }

    pub fn add_transition(&mut self, t: Transition) {
        self.transitions.push(t);
    }

    pub fn set_default_speed(&mut self, speed: f64) {
        self.default_speed = speed;
    }

    /// Start playing a clip immediately with no blending.
    pub fn play(&mut self, clip_name: &str) {
        if self.clips.contains_key(clip_name) {
            self.current = Some(PlaybackState::new(clip_name, self.default_speed));
            self.crossfade = None;
        }
    }

    /// Play a clip with speed override.
    pub fn play_with_speed(&mut self, clip_name: &str, speed: f64) {
        if self.clips.contains_key(clip_name) {
            self.current = Some(PlaybackState::new(clip_name, speed));
            self.crossfade = None;
        }
    }

    /// Crossfade from the current clip to a new one.
    pub fn crossfade_to(&mut self, clip_name: &str, blend_secs: f64) {
        if !self.clips.contains_key(clip_name) {
            return;
        }
        if let Some(cur) = &self.current {
            let fade = Crossfade {
                from_clip: cur.clip_name.clone(),
                from_elapsed: cur.elapsed,
                blend_duration: blend_secs,
                blend_elapsed: 0.0,
            };
            self.crossfade = Some(fade);
        }
        self.current = Some(PlaybackState::new(clip_name, self.default_speed));
    }

    /// Fire a trigger on the state machine. If a matching transition exists,
    /// crossfade to the target state.
    pub fn trigger(&mut self, trigger_name: &str) {
        let current_name = match &self.current {
            Some(c) => c.clip_name.clone(),
            None => return,
        };
        // Find matching transition
        let transition = self
            .transitions
            .iter()
            .find(|t| t.from_state == current_name && t.trigger == trigger_name)
            .cloned();
        if let Some(t) = transition {
            if t.blend_duration > 0.0 {
                self.crossfade_to(&t.to_state, t.blend_duration);
            } else {
                self.play(&t.to_state);
            }
        }
    }

    /// Advance the animation by `dt` seconds. Returns fired event names.
    pub fn update(&mut self, dt: f64) -> Vec<String> {
        self.last_fired.clear();

        // Advance crossfade
        if let Some(ref mut fade) = self.crossfade {
            fade.blend_elapsed += dt;
            if fade.blend_elapsed >= fade.blend_duration {
                self.crossfade = None;
            }
        }

        let current = match self.current.as_mut() {
            Some(c) => c,
            None => return Vec::new(),
        };
        if current.finished {
            return Vec::new();
        }

        let clip = match self.clips.get(&current.clip_name) {
            Some(c) => c.clone(),
            None => return Vec::new(),
        };
        let total = clip.duration();
        if total <= 0.0 {
            return Vec::new();
        }

        let prev_frame = Self::resolve_frame_index_static(&clip, current.elapsed, current.forward);
        current.elapsed += dt * current.speed;

        match clip.playback {
            PlaybackMode::Once => {
                if current.elapsed >= total {
                    current.elapsed = total - 1e-12;
                    current.finished = true;
                }
            }
            PlaybackMode::Reverse => {
                if current.elapsed >= total {
                    current.elapsed = total - 1e-12;
                    current.finished = true;
                }
            }
            PlaybackMode::Loop => {
                while current.elapsed >= total {
                    current.elapsed -= total;
                }
            }
            PlaybackMode::PingPong => {
                let cycle = total * 2.0;
                while current.elapsed >= cycle {
                    current.elapsed -= cycle;
                }
                current.forward = current.elapsed < total;
            }
        }

        let new_frame = Self::resolve_frame_index_static(&clip, current.elapsed, current.forward);

        // Fire events on frame changes
        if prev_frame != new_frame {
            let events: Vec<String> = self
                .events
                .iter()
                .filter(|e| e.clip_name == clip.name && e.frame_position == new_frame)
                .map(|e| e.event_name.clone())
                .collect();
            self.last_fired = events.clone();
            return events;
        }

        Vec::new()
    }

    /// Resolve the current frame index from elapsed time (instance method wrapper).
    fn resolve_frame_index(&self, clip: &AnimationClip, elapsed: f64, forward: bool) -> u32 {
        Self::resolve_frame_index_static(clip, elapsed, forward)
    }

    /// Resolve the current frame index from elapsed time.
    fn resolve_frame_index_static(clip: &AnimationClip, elapsed: f64, forward: bool) -> u32 {
        if clip.frames.is_empty() {
            return 0;
        }
        let total = clip.duration();
        let t = elapsed.clamp(0.0, total - 1e-12);

        let effective_t = match clip.playback {
            PlaybackMode::Reverse => total - t - 1e-12,
            PlaybackMode::PingPong if !forward => total - (t - total).abs() - 1e-12,
            _ => t,
        };
        let effective_t = effective_t.max(0.0);

        let mut acc = 0.0;
        for frame in &clip.frames {
            acc += frame.duration_secs;
            if effective_t < acc {
                return frame.frame_index;
            }
        }
        clip.frames.last().unwrap().frame_index
    }

    /// Get the current display frame index.
    pub fn current_frame(&self) -> u32 {
        let current = match &self.current {
            Some(c) => c,
            None => return 0,
        };
        let clip = match self.clips.get(&current.clip_name) {
            Some(c) => c,
            None => return 0,
        };
        self.resolve_frame_index(clip, current.elapsed, current.forward)
    }

    /// Get the blend weight of the current clip (1.0 = fully this clip,
    /// < 1.0 during crossfade).
    pub fn blend_weight(&self) -> f64 {
        match &self.crossfade {
            Some(fade) => {
                if fade.blend_duration <= 0.0 {
                    1.0
                } else {
                    (fade.blend_elapsed / fade.blend_duration).clamp(0.0, 1.0)
                }
            }
            None => 1.0,
        }
    }

    /// Get the crossfade source frame (if blending).
    pub fn crossfade_source_frame(&self) -> Option<u32> {
        let fade = self.crossfade.as_ref()?;
        let clip = self.clips.get(&fade.from_clip)?;
        Some(self.resolve_frame_index(clip, fade.from_elapsed, true))
    }

    pub fn current_clip_name(&self) -> Option<&str> {
        self.current.as_ref().map(|c| c.clip_name.as_str())
    }

    pub fn is_finished(&self) -> bool {
        self.current.as_ref().map_or(true, |c| c.finished)
    }

    pub fn is_playing(&self) -> bool {
        self.current.as_ref().map_or(false, |c| !c.finished)
    }

    pub fn elapsed(&self) -> f64 {
        self.current.as_ref().map_or(0.0, |c| c.elapsed)
    }

    pub fn last_fired_events(&self) -> &[String] {
        &self.last_fired
    }
}

impl Default for SpriteAnimator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn walk_clip() -> AnimationClip {
        AnimationClip::new(
            "walk",
            vec![
                FrameDef { frame_index: 0, duration_secs: 0.1 },
                FrameDef { frame_index: 1, duration_secs: 0.1 },
                FrameDef { frame_index: 2, duration_secs: 0.1 },
                FrameDef { frame_index: 3, duration_secs: 0.1 },
            ],
            PlaybackMode::Loop,
        )
    }

    fn idle_clip() -> AnimationClip {
        AnimationClip::new(
            "idle",
            vec![
                FrameDef { frame_index: 10, duration_secs: 0.5 },
                FrameDef { frame_index: 11, duration_secs: 0.5 },
            ],
            PlaybackMode::Loop,
        )
    }

    fn attack_clip() -> AnimationClip {
        AnimationClip::new(
            "attack",
            vec![
                FrameDef { frame_index: 20, duration_secs: 0.1 },
                FrameDef { frame_index: 21, duration_secs: 0.1 },
                FrameDef { frame_index: 22, duration_secs: 0.2 },
            ],
            PlaybackMode::Once,
        )
    }

    fn make_animator() -> SpriteAnimator {
        let mut a = SpriteAnimator::new();
        a.add_clip(walk_clip());
        a.add_clip(idle_clip());
        a.add_clip(attack_clip());
        a
    }

    #[test]
    fn play_sets_current_clip() {
        let mut a = make_animator();
        a.play("walk");
        assert_eq!(a.current_clip_name(), Some("walk"));
    }

    #[test]
    fn initial_frame_is_first() {
        let mut a = make_animator();
        a.play("walk");
        assert_eq!(a.current_frame(), 0);
    }

    #[test]
    fn advance_changes_frame() {
        let mut a = make_animator();
        a.play("walk");
        a.update(0.15);
        assert_eq!(a.current_frame(), 1);
    }

    #[test]
    fn loop_wraps_around() {
        let mut a = make_animator();
        a.play("walk");
        // 0.4s = full cycle, should be back to frame 0
        a.update(0.4);
        assert_eq!(a.current_frame(), 0);
        assert!(a.is_playing());
    }

    #[test]
    fn once_stops_at_end() {
        let mut a = make_animator();
        a.play("attack");
        a.update(1.0); // Way past the end
        assert!(a.is_finished());
        assert_eq!(a.current_frame(), 22);
    }

    #[test]
    fn ping_pong_reverses() {
        let mut a = SpriteAnimator::new();
        a.add_clip(AnimationClip::new(
            "bounce",
            vec![
                FrameDef { frame_index: 0, duration_secs: 0.1 },
                FrameDef { frame_index: 1, duration_secs: 0.1 },
                FrameDef { frame_index: 2, duration_secs: 0.1 },
            ],
            PlaybackMode::PingPong,
        ));
        a.play("bounce");
        // Forward: 0 -> 1 -> 2, Backward: 2 -> 1 -> 0, cycle = 0.6
        a.update(0.05); // frame 0
        assert_eq!(a.current_frame(), 0);
        a.update(0.1); // 0.15 → frame 1
        assert_eq!(a.current_frame(), 1);
    }

    #[test]
    fn reverse_plays_backward() {
        let mut a = SpriteAnimator::new();
        a.add_clip(AnimationClip::new(
            "rev",
            vec![
                FrameDef { frame_index: 0, duration_secs: 0.1 },
                FrameDef { frame_index: 1, duration_secs: 0.1 },
                FrameDef { frame_index: 2, duration_secs: 0.1 },
            ],
            PlaybackMode::Reverse,
        ));
        a.play("rev");
        assert_eq!(a.current_frame(), 2);
        a.update(0.15);
        assert_eq!(a.current_frame(), 1);
    }

    #[test]
    fn speed_multiplier() {
        let mut a = make_animator();
        a.play_with_speed("walk", 2.0);
        // At 2x, 0.05s real = 0.1s anim → crosses into frame 1
        a.update(0.05);
        assert_eq!(a.current_frame(), 1);
    }

    #[test]
    fn events_fire_on_frame_change() {
        let mut a = make_animator();
        a.add_event(AnimEvent {
            clip_name: "walk".to_string(),
            frame_position: 1,
            event_name: "footstep_left".to_string(),
        });
        a.play("walk");
        let events = a.update(0.15); // land on frame 1
        assert!(events.contains(&"footstep_left".to_string()));
    }

    #[test]
    fn events_not_fired_if_same_frame() {
        let mut a = make_animator();
        a.add_event(AnimEvent {
            clip_name: "walk".to_string(),
            frame_position: 0,
            event_name: "step".to_string(),
        });
        a.play("walk");
        let events = a.update(0.01); // Still on frame 0
        assert!(events.is_empty());
    }

    #[test]
    fn crossfade_blend_weight() {
        let mut a = make_animator();
        a.play("idle");
        a.update(0.1);
        a.crossfade_to("walk", 0.5);
        assert!(a.blend_weight() < 1.0 - EPS);
        a.update(0.5); // complete the blend
        assert!((a.blend_weight() - 1.0).abs() < EPS);
    }

    #[test]
    fn crossfade_source_frame() {
        let mut a = make_animator();
        a.play("idle");
        a.update(0.1);
        a.crossfade_to("walk", 1.0);
        assert!(a.crossfade_source_frame().is_some());
    }

    #[test]
    fn trigger_transition() {
        let mut a = make_animator();
        a.add_transition(Transition {
            from_state: "idle".to_string(),
            to_state: "walk".to_string(),
            trigger: "move".to_string(),
            blend_duration: 0.2,
        });
        a.play("idle");
        a.trigger("move");
        assert_eq!(a.current_clip_name(), Some("walk"));
    }

    #[test]
    fn trigger_no_match_stays() {
        let mut a = make_animator();
        a.play("idle");
        a.trigger("nonexistent");
        assert_eq!(a.current_clip_name(), Some("idle"));
    }

    #[test]
    fn clip_duration() {
        let clip = walk_clip();
        assert!((clip.duration() - 0.4).abs() < EPS);
    }

    #[test]
    fn no_clip_returns_zero_frame() {
        let a = SpriteAnimator::new();
        assert_eq!(a.current_frame(), 0);
    }

    #[test]
    fn play_nonexistent_clip_no_panic() {
        let mut a = SpriteAnimator::new();
        a.play("ghost");
        assert!(a.current_clip_name().is_none());
    }

    #[test]
    fn default_speed_applied() {
        let mut a = make_animator();
        a.set_default_speed(0.5);
        a.play("walk");
        // 0.5x speed: 0.2s real = 0.1s anim → just crossed frame 1
        a.update(0.2);
        assert_eq!(a.current_frame(), 1);
    }

    #[test]
    fn last_fired_events_persisted() {
        let mut a = make_animator();
        a.add_event(AnimEvent {
            clip_name: "walk".to_string(),
            frame_position: 1,
            event_name: "step".to_string(),
        });
        a.play("walk");
        a.update(0.15);
        assert_eq!(a.last_fired_events(), &["step".to_string()]);
        a.update(0.01);
        assert!(a.last_fired_events().is_empty());
    }

    #[test]
    fn elapsed_tracks_time() {
        let mut a = make_animator();
        a.play("idle");
        a.update(0.3);
        assert!((a.elapsed() - 0.3).abs() < EPS);
    }

    #[test]
    fn instant_transition_no_blend() {
        let mut a = make_animator();
        a.add_transition(Transition {
            from_state: "idle".to_string(),
            to_state: "attack".to_string(),
            trigger: "attack".to_string(),
            blend_duration: 0.0,
        });
        a.play("idle");
        a.trigger("attack");
        assert_eq!(a.current_clip_name(), Some("attack"));
        assert!((a.blend_weight() - 1.0).abs() < EPS);
    }
}
