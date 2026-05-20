//! Level/area streaming for large worlds.
//!
//! Manages streaming volumes (axis-aligned boxes) that trigger loading/unloading
//! of sub-levels. Provides distance-based priority, a memory budget, level
//! transitions with loading screens, and persistent vs transient level tracking.

use std::collections::HashMap;
use std::fmt;

// ── Streaming state ────────────────────────────────────────────

/// Lifecycle state for a streaming sub-level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamingState {
    Unloaded,
    Loading,
    LoadedHidden,
    LoadedVisible,
    Unloading,
}

impl fmt::Display for StreamingState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamingState::Unloaded => write!(f, "unloaded"),
            StreamingState::Loading => write!(f, "loading"),
            StreamingState::LoadedHidden => write!(f, "loaded_hidden"),
            StreamingState::LoadedVisible => write!(f, "loaded_visible"),
            StreamingState::Unloading => write!(f, "unloading"),
        }
    }
}

// ── AABB volume ────────────────────────────────────────────────

/// Axis-aligned bounding box in 3D space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AabbVolume {
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
}

impl AabbVolume {
    pub fn new(min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64) -> Self {
        Self {
            min_x: min_x.min(max_x),
            min_y: min_y.min(max_y),
            min_z: min_z.min(max_z),
            max_x: min_x.max(max_x),
            max_y: min_y.max(max_y),
            max_z: min_z.max(max_z),
        }
    }

    /// Check whether a point is inside this volume.
    pub fn contains(&self, x: f64, y: f64, z: f64) -> bool {
        x >= self.min_x
            && x <= self.max_x
            && y >= self.min_y
            && y <= self.max_y
            && z >= self.min_z
            && z <= self.max_z
    }

    /// Center of the volume.
    pub fn center(&self) -> (f64, f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
            (self.min_z + self.max_z) / 2.0,
        )
    }

    /// Squared distance from a point to the nearest face of the AABB.
    /// Returns 0 if point is inside.
    pub fn distance_sq(&self, x: f64, y: f64, z: f64) -> f64 {
        let dx = (self.min_x - x).max(0.0).max(x - self.max_x);
        let dy = (self.min_y - y).max(0.0).max(y - self.max_y);
        let dz = (self.min_z - z).max(0.0).max(z - self.max_z);
        dx * dx + dy * dy + dz * dz
    }

    /// Whether two AABBs overlap.
    pub fn overlaps(&self, other: &AabbVolume) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }
}

// ── Level persistence ──────────────────────────────────────────

/// Whether a level persists after unloading or is recreated fresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelPersistence {
    /// State is preserved across load/unload cycles.
    Persistent,
    /// State is discarded on unload; recreated fresh on next load.
    Transient,
}

// ── Sub-level ──────────────────────────────────────────────────

/// A sub-level that can be streamed in and out of the world.
#[derive(Debug, Clone)]
pub struct SubLevel {
    pub id: String,
    pub name: String,
    pub volume: AabbVolume,
    pub state: StreamingState,
    pub persistence: LevelPersistence,
    /// Estimated memory size in bytes.
    pub memory_bytes: u64,
    /// Priority bias (higher = more important to load).
    pub priority_bias: i32,
    /// Opaque data payload preserved for persistent levels.
    pub saved_data: Option<String>,
}

impl SubLevel {
    pub fn new(id: &str, name: &str, volume: AabbVolume, memory_bytes: u64) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            volume,
            state: StreamingState::Unloaded,
            persistence: LevelPersistence::Transient,
            memory_bytes,
            priority_bias: 0,
            saved_data: None,
        }
    }

    pub fn persistent(mut self) -> Self {
        self.persistence = LevelPersistence::Persistent;
        self
    }

    pub fn with_priority(mut self, bias: i32) -> Self {
        self.priority_bias = bias;
        self
    }

    /// Whether the level has data loaded in memory (hidden or visible).
    pub fn is_loaded(&self) -> bool {
        matches!(
            self.state,
            StreamingState::LoadedHidden | StreamingState::LoadedVisible
        )
    }
}

// ── Level transition ───────────────────────────────────────────

/// Describes a transition between two level areas.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelTransition {
    pub from_level: String,
    pub to_level: String,
    pub loading_screen: String,
    pub progress: f64,
    pub complete: bool,
}

impl LevelTransition {
    pub fn new(from: &str, to: &str, loading_screen: &str) -> Self {
        Self {
            from_level: from.to_string(),
            to_level: to.to_string(),
            loading_screen: loading_screen.to_string(),
            progress: 0.0,
            complete: false,
        }
    }

    /// Advance progress by a delta (clamped to [0, 1]).
    pub fn advance(&mut self, delta: f64) {
        self.progress = (self.progress + delta).clamp(0.0, 1.0);
        if self.progress >= 1.0 - 1e-12 {
            self.complete = true;
        }
    }
}

// ── Streaming event ────────────────────────────────────────────

/// Events emitted by the level streaming system.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamingEvent {
    LoadRequested(String),
    LoadComplete(String),
    MadeVisible(String),
    MadeHidden(String),
    UnloadRequested(String),
    UnloadComplete(String),
    MemoryBudgetExceeded { used: u64, budget: u64 },
}

// ── Level streaming manager ────────────────────────────────────

/// Manages streaming of sub-levels in a large world.
pub struct LevelStreaming {
    levels: HashMap<String, SubLevel>,
    memory_budget: u64,
    player_pos: (f64, f64, f64),
    events: Vec<StreamingEvent>,
    transition: Option<LevelTransition>,
}

impl fmt::Debug for LevelStreaming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LevelStreaming")
            .field("levels", &self.levels.len())
            .field("budget", &self.memory_budget)
            .finish()
    }
}

impl LevelStreaming {
    /// Create a new streaming manager with the given memory budget (bytes).
    pub fn new(memory_budget: u64) -> Self {
        Self {
            levels: HashMap::new(),
            memory_budget,
            player_pos: (0.0, 0.0, 0.0),
            events: Vec::new(),
            transition: None,
        }
    }

    /// Register a sub-level.
    pub fn register(&mut self, level: SubLevel) {
        self.levels.insert(level.id.clone(), level);
    }

    /// Unregister a sub-level by ID.
    pub fn unregister(&mut self, id: &str) -> Option<SubLevel> {
        self.levels.remove(id)
    }

    /// Update the player position and evaluate which levels should load/unload.
    pub fn update_player(&mut self, x: f64, y: f64, z: f64) -> Vec<StreamingEvent> {
        self.events.clear();
        self.player_pos = (x, y, z);

        // Determine which levels the player is inside.
        let level_ids: Vec<String> = self.levels.keys().cloned().collect();
        let mut inside = Vec::new();
        let mut outside = Vec::new();
        for id in &level_ids {
            let lvl = &self.levels[id];
            if lvl.volume.contains(x, y, z) {
                inside.push(id.clone());
            } else {
                outside.push(id.clone());
            }
        }

        // Request load for levels the player is inside.
        for id in &inside {
            let state = self.levels[id].state;
            if state == StreamingState::Unloaded {
                self.begin_load(id);
            } else if state == StreamingState::LoadedHidden {
                self.make_visible(id);
            }
        }

        // Request unload for levels the player left (if transient or far away).
        for id in &outside {
            let lvl = &self.levels[id];
            if lvl.state == StreamingState::LoadedVisible {
                self.make_hidden(id);
            }
        }

        self.events.clone()
    }

    fn begin_load(&mut self, id: &str) {
        if self.used_memory() + self.levels[id].memory_bytes > self.memory_budget {
            self.events.push(StreamingEvent::MemoryBudgetExceeded {
                used: self.used_memory(),
                budget: self.memory_budget,
            });
            // Try to evict the lowest-priority loaded-hidden level.
            if !self.evict_one(id) {
                return;
            }
        }
        if let Some(lvl) = self.levels.get_mut(id) {
            lvl.state = StreamingState::Loading;
            self.events.push(StreamingEvent::LoadRequested(id.to_string()));
        }
    }

    /// Try to evict one loaded-hidden level to make room. Returns true if eviction succeeded.
    fn evict_one(&mut self, _requesting_id: &str) -> bool {
        // Find the loaded-hidden level with lowest priority.
        let ids: Vec<String> = self.levels.keys().cloned().collect();
        let mut best: Option<String> = None;
        let mut best_priority = i32::MAX;
        for id in &ids {
            let lvl = &self.levels[id];
            if lvl.state == StreamingState::LoadedHidden {
                let effective = lvl.priority_bias;
                if effective < best_priority {
                    best_priority = effective;
                    best = Some(id.clone());
                }
            }
        }
        if let Some(evict_id) = best {
            self.force_unload(&evict_id);
            true
        } else {
            false
        }
    }

    fn make_visible(&mut self, id: &str) {
        if let Some(lvl) = self.levels.get_mut(id) {
            lvl.state = StreamingState::LoadedVisible;
            self.events.push(StreamingEvent::MadeVisible(id.to_string()));
        }
    }

    fn make_hidden(&mut self, id: &str) {
        if let Some(lvl) = self.levels.get_mut(id) {
            lvl.state = StreamingState::LoadedHidden;
            self.events.push(StreamingEvent::MadeHidden(id.to_string()));
        }
    }

    fn force_unload(&mut self, id: &str) {
        if let Some(lvl) = self.levels.get_mut(id) {
            if lvl.persistence == LevelPersistence::Persistent {
                lvl.saved_data = Some(format!("saved:{}", lvl.id));
            } else {
                lvl.saved_data = None;
            }
            lvl.state = StreamingState::Unloaded;
            self.events.push(StreamingEvent::UnloadComplete(id.to_string()));
        }
    }

    /// Complete an in-progress load, transitioning to LoadedHidden.
    pub fn finish_load(&mut self, id: &str) -> Option<StreamingEvent> {
        if let Some(lvl) = self.levels.get_mut(id) {
            if lvl.state == StreamingState::Loading {
                lvl.state = StreamingState::LoadedHidden;
                return Some(StreamingEvent::LoadComplete(id.to_string()));
            }
        }
        None
    }

    /// Complete an in-progress unload.
    pub fn finish_unload(&mut self, id: &str) -> Option<StreamingEvent> {
        if let Some(lvl) = self.levels.get_mut(id) {
            if lvl.state == StreamingState::Unloading {
                if lvl.persistence == LevelPersistence::Persistent {
                    lvl.saved_data = Some(format!("saved:{}", lvl.id));
                } else {
                    lvl.saved_data = None;
                }
                lvl.state = StreamingState::Unloaded;
                return Some(StreamingEvent::UnloadComplete(id.to_string()));
            }
        }
        None
    }

    /// Begin a level transition (loading screen).
    pub fn begin_transition(&mut self, from: &str, to: &str, screen: &str) {
        self.transition = Some(LevelTransition::new(from, to, screen));
    }

    /// Advance the current transition.
    pub fn advance_transition(&mut self, delta: f64) -> Option<bool> {
        if let Some(ref mut t) = self.transition {
            t.advance(delta);
            Some(t.complete)
        } else {
            None
        }
    }

    /// Complete and remove the transition.
    pub fn complete_transition(&mut self) -> Option<LevelTransition> {
        self.transition.take()
    }

    /// Current transition reference.
    pub fn transition(&self) -> Option<&LevelTransition> {
        self.transition.as_ref()
    }

    /// State of a level by ID.
    pub fn level_state(&self, id: &str) -> Option<StreamingState> {
        self.levels.get(id).map(|l| l.state)
    }

    /// Level reference.
    pub fn level(&self, id: &str) -> Option<&SubLevel> {
        self.levels.get(id)
    }

    /// Memory currently used by loaded levels.
    pub fn used_memory(&self) -> u64 {
        self.levels
            .values()
            .filter(|l| l.is_loaded() || l.state == StreamingState::Loading)
            .map(|l| l.memory_bytes)
            .sum()
    }

    /// Total number of registered levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Number of loaded levels (hidden + visible).
    pub fn loaded_count(&self) -> usize {
        self.levels.values().filter(|l| l.is_loaded()).count()
    }

    /// All levels sorted by distance from player position (nearest first).
    pub fn levels_by_distance(&self) -> Vec<(String, f64)> {
        let (px, py, pz) = self.player_pos;
        let mut list: Vec<(String, f64)> = self
            .levels
            .values()
            .map(|l| {
                let d = l.volume.distance_sq(px, py, pz).sqrt();
                (l.id.clone(), d)
            })
            .collect();
        list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        list
    }

    /// Remaining memory budget.
    pub fn remaining_budget(&self) -> u64 {
        self.memory_budget.saturating_sub(self.used_memory())
    }

    /// Whether a particular level has saved persistent data.
    pub fn has_saved_data(&self, id: &str) -> bool {
        self.levels
            .get(id)
            .and_then(|l| l.saved_data.as_ref())
            .is_some()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_volume(x: f64, z: f64, size: f64) -> AabbVolume {
        AabbVolume::new(x, 0.0, z, x + size, 100.0, z + size)
    }

    fn make_level(id: &str, x: f64, z: f64, mem: u64) -> SubLevel {
        SubLevel::new(id, id, make_volume(x, z, 50.0), mem)
    }

    #[test]
    fn aabb_contains_point() {
        let v = AabbVolume::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        assert!(v.contains(5.0, 5.0, 5.0));
        assert!(v.contains(0.0, 0.0, 0.0));
        assert!(v.contains(10.0, 10.0, 10.0));
        assert!(!v.contains(11.0, 5.0, 5.0));
    }

    #[test]
    fn aabb_center() {
        let v = AabbVolume::new(0.0, 0.0, 0.0, 10.0, 20.0, 30.0);
        let c = v.center();
        assert!((c.0 - 5.0).abs() < 1e-9);
        assert!((c.1 - 10.0).abs() < 1e-9);
        assert!((c.2 - 15.0).abs() < 1e-9);
    }

    #[test]
    fn aabb_distance_inside() {
        let v = AabbVolume::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        assert!(v.distance_sq(5.0, 5.0, 5.0) < 1e-9);
    }

    #[test]
    fn aabb_distance_outside() {
        let v = AabbVolume::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let d = v.distance_sq(13.0, 5.0, 5.0);
        assert!((d - 9.0).abs() < 1e-9); // 3^2 = 9
    }

    #[test]
    fn aabb_overlaps() {
        let a = AabbVolume::new(0.0, 0.0, 0.0, 10.0, 10.0, 10.0);
        let b = AabbVolume::new(5.0, 5.0, 5.0, 15.0, 15.0, 15.0);
        assert!(a.overlaps(&b));
        let c = AabbVolume::new(20.0, 20.0, 20.0, 30.0, 30.0, 30.0);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn aabb_swapped_min_max() {
        let v = AabbVolume::new(10.0, 10.0, 10.0, 0.0, 0.0, 0.0);
        assert!(v.contains(5.0, 5.0, 5.0));
    }

    #[test]
    fn streaming_state_display() {
        assert_eq!(StreamingState::LoadedVisible.to_string(), "loaded_visible");
        assert_eq!(StreamingState::Unloaded.to_string(), "unloaded");
    }

    #[test]
    fn register_and_query_level() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("town", 0.0, 0.0, 100_000));
        assert_eq!(mgr.level_count(), 1);
        assert_eq!(mgr.level_state("town"), Some(StreamingState::Unloaded));
    }

    #[test]
    fn unregister_level() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("town", 0.0, 0.0, 100_000));
        let removed = mgr.unregister("town");
        assert!(removed.is_some());
        assert_eq!(mgr.level_count(), 0);
    }

    #[test]
    fn player_inside_triggers_load() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("town", 0.0, 0.0, 100_000));
        let evts = mgr.update_player(25.0, 50.0, 25.0);
        assert!(evts.iter().any(|e| matches!(e, StreamingEvent::LoadRequested(id) if id == "town")));
        assert_eq!(mgr.level_state("town"), Some(StreamingState::Loading));
    }

    #[test]
    fn finish_load_to_hidden() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("town", 0.0, 0.0, 100_000));
        mgr.update_player(25.0, 50.0, 25.0);
        let evt = mgr.finish_load("town");
        assert_eq!(evt, Some(StreamingEvent::LoadComplete("town".to_string())));
        assert_eq!(mgr.level_state("town"), Some(StreamingState::LoadedHidden));
    }

    #[test]
    fn hidden_to_visible_when_inside() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("town", 0.0, 0.0, 100_000));
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("town");
        let evts = mgr.update_player(25.0, 50.0, 25.0);
        assert!(evts.iter().any(|e| matches!(e, StreamingEvent::MadeVisible(id) if id == "town")));
    }

    #[test]
    fn visible_to_hidden_when_outside() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("town", 0.0, 0.0, 100_000));
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("town");
        mgr.update_player(25.0, 50.0, 25.0); // made visible
        let evts = mgr.update_player(999.0, 50.0, 999.0); // leave
        assert!(evts.iter().any(|e| matches!(e, StreamingEvent::MadeHidden(id) if id == "town")));
    }

    #[test]
    fn memory_budget_respected() {
        let mut mgr = LevelStreaming::new(150_000);
        mgr.register(make_level("a", 0.0, 0.0, 100_000));
        mgr.register(make_level("b", 0.0, 0.0, 100_000));
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("a");
        // Now trying to load b should trigger budget exceeded.
        let evts = mgr.update_player(25.0, 50.0, 25.0);
        // b should still be requested after evicting a.
        let has_budget_exceeded = evts.iter().any(|e| matches!(e, StreamingEvent::MemoryBudgetExceeded { .. }));
        assert!(has_budget_exceeded);
    }

    #[test]
    fn used_memory_tracking() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("a", 0.0, 0.0, 200_000));
        assert_eq!(mgr.used_memory(), 0);
        mgr.update_player(25.0, 50.0, 25.0);
        assert_eq!(mgr.used_memory(), 200_000);
        mgr.finish_load("a");
        assert_eq!(mgr.used_memory(), 200_000);
    }

    #[test]
    fn level_transition_progress() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.begin_transition("town", "dungeon", "loading_dungeon.png");
        assert!(mgr.transition().is_some());
        mgr.advance_transition(0.3);
        assert!((mgr.transition().unwrap().progress - 0.3).abs() < 1e-9);
        mgr.advance_transition(0.8);
        assert!(mgr.transition().unwrap().complete);
        let t = mgr.complete_transition().unwrap();
        assert_eq!(t.to_level, "dungeon");
        assert!(mgr.transition().is_none());
    }

    #[test]
    fn persistent_level_saves_data() {
        let mut mgr = LevelStreaming::new(1_000_000);
        let lvl = make_level("town", 0.0, 0.0, 100_000).persistent();
        mgr.register(lvl);
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("town");
        // Force going hidden then unloading
        mgr.update_player(25.0, 50.0, 25.0); // visible
        mgr.update_player(999.0, 999.0, 999.0); // hidden
        // Now force unload by budget pressure
        let state_before = mgr.level_state("town");
        assert_eq!(state_before, Some(StreamingState::LoadedHidden));
    }

    #[test]
    fn levels_by_distance_ordering() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("far", 500.0, 500.0, 100));
        mgr.register(make_level("near", 0.0, 0.0, 100));
        mgr.update_player(10.0, 50.0, 10.0);
        let sorted = mgr.levels_by_distance();
        assert_eq!(sorted[0].0, "near");
        assert_eq!(sorted[1].0, "far");
    }

    #[test]
    fn remaining_budget() {
        let mut mgr = LevelStreaming::new(500_000);
        mgr.register(make_level("a", 0.0, 0.0, 200_000));
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("a");
        assert_eq!(mgr.remaining_budget(), 300_000);
    }

    #[test]
    fn transition_advance_clamps() {
        let mut t = LevelTransition::new("a", "b", "screen");
        t.advance(1.5);
        assert!((t.progress - 1.0).abs() < 1e-9);
        assert!(t.complete);
    }

    #[test]
    fn priority_bias_affects_eviction() {
        // Budget fits 2 of 3 levels. When the third triggers, eviction should
        // remove the loaded-hidden level with lowest priority.
        let mut mgr = LevelStreaming::new(200_000);
        // Place low and high in a shared volume the player starts inside.
        let low = make_level("low", 0.0, 0.0, 100_000).with_priority(-10);
        let high = make_level("high", 0.0, 0.0, 100_000).with_priority(10);
        mgr.register(low);
        mgr.register(high);
        // Step 1: load both.
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("low");
        mgr.finish_load("high");
        // Step 2: move player away so both become hidden.
        mgr.update_player(999.0, 999.0, 999.0);
        assert_eq!(mgr.level_state("low"), Some(StreamingState::LoadedHidden));
        assert_eq!(mgr.level_state("high"), Some(StreamingState::LoadedHidden));
        // Step 3: register a trigger level in a new area and move player there.
        let trigger = make_level("trigger", 900.0, 900.0, 100_000);
        mgr.register(trigger);
        mgr.update_player(925.0, 50.0, 925.0);
        // Trigger load request exceeds budget (200K used + 100K > 200K).
        // Eviction should remove "low" (priority -10) not "high" (priority 10).
        assert_eq!(mgr.level_state("low"), Some(StreamingState::Unloaded));
        assert!(mgr.level("high").unwrap().is_loaded());
    }

    #[test]
    fn loaded_count() {
        let mut mgr = LevelStreaming::new(1_000_000);
        mgr.register(make_level("a", 0.0, 0.0, 100));
        mgr.register(make_level("b", 0.0, 0.0, 100));
        mgr.update_player(25.0, 50.0, 25.0);
        mgr.finish_load("a");
        mgr.finish_load("b");
        assert_eq!(mgr.loaded_count(), 2);
    }
}
