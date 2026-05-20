//! Scene lifecycle management.
//!
//! Load / unload scenes, transition between scenes (with optional
//! transition effects like fade), scene stack for overlays (push / pop
//! for pause menus, etc.). Each scene has enter / exit / update callbacks
//! represented as state.

use std::collections::HashMap;

// ── Scene state ─────────────────────────────────────────────────

/// Lifecycle state of a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneState {
    /// Not yet loaded.
    Unloaded,
    /// Loading resources.
    Loading,
    /// Active and receiving updates.
    Active,
    /// Paused (on stack but not the topmost scene).
    Suspended,
    /// Transitioning out.
    Exiting,
}

/// Callback events that a scene owner should handle.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneEvent {
    Enter { scene_id: String },
    Exit { scene_id: String },
    Suspend { scene_id: String },
    Resume { scene_id: String },
    Update { scene_id: String, dt: f64 },
    TransitionProgress { from: String, to: String, progress: f64 },
    TransitionComplete { from: String, to: String },
    Loaded { scene_id: String },
    Unloaded { scene_id: String },
}

// ── Transition ──────────────────────────────────────────────────

/// Visual transition effect between scenes.
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionEffect {
    None,
    Fade { duration_secs: f64 },
    SlideLeft { duration_secs: f64 },
    SlideRight { duration_secs: f64 },
    Custom { name: String, duration_secs: f64 },
}

impl TransitionEffect {
    pub fn duration(&self) -> f64 {
        match self {
            TransitionEffect::None => 0.0,
            TransitionEffect::Fade { duration_secs } => *duration_secs,
            TransitionEffect::SlideLeft { duration_secs } => *duration_secs,
            TransitionEffect::SlideRight { duration_secs } => *duration_secs,
            TransitionEffect::Custom { duration_secs, .. } => *duration_secs,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ActiveTransition {
    from: String,
    to: String,
    effect: TransitionEffect,
    elapsed: f64,
}

impl ActiveTransition {
    fn progress(&self) -> f64 {
        let d = self.effect.duration();
        if d < 1e-12 {
            return 1.0;
        }
        (self.elapsed / d).min(1.0)
    }

    fn is_complete(&self) -> bool {
        self.progress() >= 1.0 - 1e-12
    }
}

// ── Scene descriptor ────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
struct SceneDescriptor {
    id: String,
    state: SceneState,
    /// Arbitrary tags for the scene (e.g., "overlay", "main").
    tags: Vec<String>,
    /// Cumulative time spent active in this scene (seconds).
    active_time: f64,
    update_count: u64,
}

// ── Scene manager ───────────────────────────────────────────────

/// Manages scene lifecycles, transitions, and a scene stack.
#[derive(Debug)]
pub struct SceneManager {
    scenes: HashMap<String, SceneDescriptor>,
    /// Stack of active scene IDs (top = last element = receiving input).
    stack: Vec<String>,
    transition: Option<ActiveTransition>,
    events: Vec<SceneEvent>,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            stack: Vec::new(),
            transition: None,
            events: Vec::new(),
        }
    }

    // ── Registration ────────────────────────────────────────────

    /// Register a new scene (starts in `Unloaded` state).
    pub fn register(&mut self, id: &str, tags: &[&str]) {
        self.scenes.insert(id.to_string(), SceneDescriptor {
            id: id.to_string(),
            state: SceneState::Unloaded,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            active_time: 0.0,
            update_count: 0,
        });
    }

    pub fn is_registered(&self, id: &str) -> bool {
        self.scenes.contains_key(id)
    }

    pub fn scene_state(&self, id: &str) -> Option<SceneState> {
        self.scenes.get(id).map(|s| s.state)
    }

    pub fn scene_tags(&self, id: &str) -> Vec<String> {
        self.scenes.get(id).map(|s| s.tags.clone()).unwrap_or_default()
    }

    // ── Loading ─────────────────────────────────────────────────

    /// Mark a scene as loading (caller drives actual resource loading).
    pub fn begin_load(&mut self, id: &str) -> bool {
        if let Some(scene) = self.scenes.get_mut(id) {
            if scene.state == SceneState::Unloaded {
                scene.state = SceneState::Loading;
                return true;
            }
        }
        false
    }

    /// Mark loading complete. Scene becomes available but not yet active.
    pub fn finish_load(&mut self, id: &str) -> bool {
        if let Some(scene) = self.scenes.get_mut(id) {
            if scene.state == SceneState::Loading {
                scene.state = SceneState::Unloaded; // loaded but not entered
                self.events.push(SceneEvent::Loaded { scene_id: id.to_string() });
                return true;
            }
        }
        false
    }

    /// Unload a scene (must not be on the stack).
    pub fn unload(&mut self, id: &str) -> bool {
        if self.stack.contains(&id.to_string()) {
            return false;
        }
        if let Some(scene) = self.scenes.get_mut(id) {
            scene.state = SceneState::Unloaded;
            scene.active_time = 0.0;
            scene.update_count = 0;
            self.events.push(SceneEvent::Unloaded { scene_id: id.to_string() });
            return true;
        }
        false
    }

    // ── Stack operations ────────────────────────────────────────

    /// Push a scene onto the stack, suspending the current top scene.
    pub fn push(&mut self, id: &str) -> bool {
        if !self.scenes.contains_key(id) {
            return false;
        }
        if self.stack.contains(&id.to_string()) {
            return false;
        }
        // Suspend current top.
        if let Some(top_id) = self.stack.last().cloned() {
            if let Some(top) = self.scenes.get_mut(&top_id) {
                top.state = SceneState::Suspended;
            }
            self.events.push(SceneEvent::Suspend { scene_id: top_id });
        }
        // Activate new scene.
        if let Some(scene) = self.scenes.get_mut(id) {
            scene.state = SceneState::Active;
        }
        self.stack.push(id.to_string());
        self.events.push(SceneEvent::Enter { scene_id: id.to_string() });
        true
    }

    /// Pop the top scene, resuming the one below.
    pub fn pop(&mut self) -> Option<String> {
        let popped = self.stack.pop()?;
        if let Some(scene) = self.scenes.get_mut(&popped) {
            scene.state = SceneState::Unloaded;
        }
        self.events.push(SceneEvent::Exit { scene_id: popped.clone() });
        // Resume new top.
        if let Some(new_top) = self.stack.last().cloned() {
            if let Some(scene) = self.scenes.get_mut(&new_top) {
                scene.state = SceneState::Active;
            }
            self.events.push(SceneEvent::Resume { scene_id: new_top });
        }
        Some(popped)
    }

    /// The currently active (topmost) scene ID.
    pub fn active_scene(&self) -> Option<&str> {
        self.stack.last().map(|s| s.as_str())
    }

    /// Full stack depth.
    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    pub fn stack_contents(&self) -> Vec<String> {
        self.stack.clone()
    }

    // ── Transitions ─────────────────────────────────────────────

    /// Begin a transition from the current active scene to `to_id`.
    pub fn transition_to(&mut self, to_id: &str, effect: TransitionEffect) -> bool {
        if !self.scenes.contains_key(to_id) {
            return false;
        }
        if self.transition.is_some() {
            return false; // already transitioning
        }
        let from_id = self.active_scene().unwrap_or("").to_string();
        if from_id == to_id {
            return false;
        }

        // Mark the source as exiting.
        if let Some(scene) = self.scenes.get_mut(&from_id) {
            scene.state = SceneState::Exiting;
        }

        if effect.duration() < 1e-12 {
            // Instant transition.
            self.complete_transition(&from_id, to_id);
            return true;
        }

        self.transition = Some(ActiveTransition {
            from: from_id,
            to: to_id.to_string(),
            effect,
            elapsed: 0.0,
        });
        true
    }

    /// Is a transition currently in progress?
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_some()
    }

    /// Current transition progress [0, 1] or None.
    pub fn transition_progress(&self) -> Option<f64> {
        self.transition.as_ref().map(|t| t.progress())
    }

    // ── Update ──────────────────────────────────────────────────

    /// Advance the manager by `dt` seconds.
    pub fn update(&mut self, dt: f64) {
        // Advance transition if active.
        let completed = if let Some(trans) = &mut self.transition {
            trans.elapsed += dt;
            let prog = trans.progress();
            self.events.push(SceneEvent::TransitionProgress {
                from: trans.from.clone(),
                to: trans.to.clone(),
                progress: prog,
            });
            if trans.is_complete() {
                Some((trans.from.clone(), trans.to.clone()))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((from, to)) = completed {
            self.transition = None;
            self.complete_transition(&from, &to);
        }

        // Update active scene timing.
        if let Some(active_id) = self.stack.last().cloned() {
            if let Some(scene) = self.scenes.get_mut(&active_id) {
                if scene.state == SceneState::Active {
                    scene.active_time += dt;
                    scene.update_count += 1;
                    self.events.push(SceneEvent::Update {
                        scene_id: active_id,
                        dt,
                    });
                }
            }
        }
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<SceneEvent> {
        std::mem::take(&mut self.events)
    }

    /// Scene active time accumulator.
    pub fn scene_active_time(&self, id: &str) -> f64 {
        self.scenes.get(id).map(|s| s.active_time).unwrap_or(0.0)
    }

    pub fn scene_update_count(&self, id: &str) -> u64 {
        self.scenes.get(id).map(|s| s.update_count).unwrap_or(0)
    }

    // ── Internal ────────────────────────────────────────────────

    fn complete_transition(&mut self, from: &str, to: &str) {
        // Pop old scene from stack if present.
        self.stack.retain(|s| s != from);
        if let Some(scene) = self.scenes.get_mut(from) {
            scene.state = SceneState::Unloaded;
        }
        self.events.push(SceneEvent::Exit { scene_id: from.to_string() });

        // Push new scene.
        if !self.stack.contains(&to.to_string()) {
            self.stack.push(to.to_string());
        }
        if let Some(scene) = self.scenes.get_mut(to) {
            scene.state = SceneState::Active;
        }
        self.events.push(SceneEvent::Enter { scene_id: to.to_string() });
        self.events.push(SceneEvent::TransitionComplete {
            from: from.to_string(),
            to: to.to_string(),
        });
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn test_register_scene() {
        let mut sm = SceneManager::new();
        sm.register("main_menu", &["menu"]);
        assert!(sm.is_registered("main_menu"));
        assert_eq!(sm.scene_state("main_menu"), Some(SceneState::Unloaded));
    }

    #[test]
    fn test_push_scene() {
        let mut sm = SceneManager::new();
        sm.register("game", &[]);
        assert!(sm.push("game"));
        assert_eq!(sm.active_scene(), Some("game"));
        assert_eq!(sm.scene_state("game"), Some(SceneState::Active));
    }

    #[test]
    fn test_push_unregistered_fails() {
        let mut sm = SceneManager::new();
        assert!(!sm.push("unknown"));
    }

    #[test]
    fn test_push_duplicate_fails() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.push("a");
        assert!(!sm.push("a"));
    }

    #[test]
    fn test_pop_scene() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.push("b");
        let popped = sm.pop();
        assert_eq!(popped, Some("b".to_string()));
        assert_eq!(sm.active_scene(), Some("a"));
        assert_eq!(sm.scene_state("a"), Some(SceneState::Active));
    }

    #[test]
    fn test_pop_empty_returns_none() {
        let mut sm = SceneManager::new();
        assert!(sm.pop().is_none());
    }

    #[test]
    fn test_suspend_resume_events() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.drain_events(); // clear
        sm.push("b");
        let events = sm.drain_events();
        assert!(events.iter().any(|e| matches!(e, SceneEvent::Suspend { scene_id } if scene_id == "a")));
        assert!(events.iter().any(|e| matches!(e, SceneEvent::Enter { scene_id } if scene_id == "b")));
    }

    #[test]
    fn test_instant_transition() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.drain_events();
        assert!(sm.transition_to("b", TransitionEffect::None));
        assert!(!sm.is_transitioning());
        assert_eq!(sm.active_scene(), Some("b"));
    }

    #[test]
    fn test_fade_transition() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.drain_events();

        assert!(sm.transition_to("b", TransitionEffect::Fade { duration_secs: 1.0 }));
        assert!(sm.is_transitioning());

        sm.update(0.5);
        let prog = sm.transition_progress().unwrap();
        assert!((prog - 0.5).abs() < 0.01);

        sm.update(0.6);
        assert!(!sm.is_transitioning());
        assert_eq!(sm.active_scene(), Some("b"));
    }

    #[test]
    fn test_transition_to_self_fails() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.push("a");
        assert!(!sm.transition_to("a", TransitionEffect::None));
    }

    #[test]
    fn test_double_transition_fails() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.register("c", &[]);
        sm.push("a");
        sm.transition_to("b", TransitionEffect::Fade { duration_secs: 1.0 });
        assert!(!sm.transition_to("c", TransitionEffect::None));
    }

    #[test]
    fn test_update_accumulates_time() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.push("a");
        sm.update(0.1);
        sm.update(0.2);
        assert!((sm.scene_active_time("a") - 0.3).abs() < EPS);
        assert_eq!(sm.scene_update_count("a"), 2);
    }

    #[test]
    fn test_stack_depth() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.register("c", &[]);
        assert_eq!(sm.stack_depth(), 0);
        sm.push("a");
        sm.push("b");
        sm.push("c");
        assert_eq!(sm.stack_depth(), 3);
    }

    #[test]
    fn test_stack_contents() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.push("b");
        assert_eq!(sm.stack_contents(), vec!["a", "b"]);
    }

    #[test]
    fn test_loading_flow() {
        let mut sm = SceneManager::new();
        sm.register("level1", &[]);
        assert!(sm.begin_load("level1"));
        assert_eq!(sm.scene_state("level1"), Some(SceneState::Loading));
        assert!(sm.finish_load("level1"));
        let events = sm.drain_events();
        assert!(events.iter().any(|e| matches!(e, SceneEvent::Loaded { scene_id } if scene_id == "level1")));
    }

    #[test]
    fn test_unload_while_active_fails() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.push("a");
        assert!(!sm.unload("a"));
    }

    #[test]
    fn test_unload_success() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        assert!(sm.unload("a"));
    }

    #[test]
    fn test_transition_events() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.drain_events();
        sm.transition_to("b", TransitionEffect::Fade { duration_secs: 0.5 });
        sm.update(0.6);
        let events = sm.drain_events();
        assert!(events.iter().any(|e| matches!(e, SceneEvent::TransitionComplete { from, to } if from == "a" && to == "b")));
    }

    #[test]
    fn test_scene_tags() {
        let mut sm = SceneManager::new();
        sm.register("menu", &["ui", "overlay"]);
        let tags = sm.scene_tags("menu");
        assert_eq!(tags, vec!["ui", "overlay"]);
    }

    #[test]
    fn test_transition_effect_duration() {
        assert!(TransitionEffect::None.duration().abs() < EPS);
        assert!((TransitionEffect::Fade { duration_secs: 2.0 }.duration() - 2.0).abs() < EPS);
        assert!((TransitionEffect::SlideLeft { duration_secs: 0.5 }.duration() - 0.5).abs() < EPS);
        assert!((TransitionEffect::SlideRight { duration_secs: 1.5 }.duration() - 1.5).abs() < EPS);
        assert!((TransitionEffect::Custom { name: "wipe".into(), duration_secs: 3.0 }.duration() - 3.0).abs() < EPS);
    }

    #[test]
    fn test_drain_events_clears() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.push("a");
        let first = sm.drain_events();
        assert!(!first.is_empty());
        let second = sm.drain_events();
        assert!(second.is_empty());
    }

    #[test]
    fn test_suspended_scene_no_update() {
        let mut sm = SceneManager::new();
        sm.register("a", &[]);
        sm.register("b", &[]);
        sm.push("a");
        sm.push("b");
        sm.update(0.1);
        // 'a' is suspended, should not accumulate time.
        assert!(sm.scene_active_time("a").abs() < EPS);
        assert!((sm.scene_active_time("b") - 0.1).abs() < EPS);
    }
}
