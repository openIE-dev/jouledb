//! Change detection for ECS components.
//!
//! Tracks Added, Changed, and Removed events per component per frame using a
//! tick-based system. Each component instance has an "added tick" and a "last
//! changed tick." Systems record a "last run tick." Querying "what changed
//! since my last run" compares these ticks.

use std::any::TypeId;
use std::collections::{HashMap, HashSet};

// ── ChangeEvent ──

/// Type of change event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeEvent {
    /// Component was newly added to the entity.
    Added,
    /// Component value was modified.
    Changed,
    /// Component was removed from the entity.
    Removed,
}

// ── ComponentTick ──

/// Per-entity, per-component tick information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentTick {
    /// Tick when the component was first added.
    pub added_tick: u64,
    /// Tick when the component was last modified (including initial add).
    pub changed_tick: u64,
}

impl ComponentTick {
    pub fn new(tick: u64) -> Self {
        Self {
            added_tick: tick,
            changed_tick: tick,
        }
    }

    pub fn mark_changed(&mut self, tick: u64) {
        self.changed_tick = tick;
    }

    /// Was this component added after `since_tick`?
    pub fn is_added_since(&self, since_tick: u64) -> bool {
        self.added_tick > since_tick
    }

    /// Was this component changed after `since_tick`?
    pub fn is_changed_since(&self, since_tick: u64) -> bool {
        self.changed_tick > since_tick
    }
}

// ── RemovedRecord ──

/// Record of a component removal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovedRecord {
    pub entity_id: u64,
    pub type_id: TypeId,
    pub removed_tick: u64,
}

// ── SystemTick ──

/// Tracks the last-run tick for a named system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemTick {
    pub name: String,
    pub last_run_tick: u64,
}

// ── ChangeDetector ──

/// Central change detection tracker for the entire ECS.
pub struct ChangeDetector {
    /// Current world tick (incremented each frame).
    current_tick: u64,
    /// (entity_id, TypeId) → ComponentTick.
    component_ticks: HashMap<(u64, TypeId), ComponentTick>,
    /// Removal log — kept for a configurable number of ticks.
    removal_log: Vec<RemovedRecord>,
    /// system_name → last_run_tick.
    system_ticks: HashMap<String, u64>,
    /// How many ticks to retain removal records.
    removal_retention_ticks: u64,
}

impl ChangeDetector {
    /// Create a new change detector starting at tick 0.
    pub fn new() -> Self {
        Self {
            current_tick: 0,
            component_ticks: HashMap::new(),
            removal_log: Vec::new(),
            system_ticks: HashMap::new(),
            removal_retention_ticks: 10,
        }
    }

    /// Create with a custom removal retention window.
    pub fn with_retention(retention_ticks: u64) -> Self {
        Self {
            removal_retention_ticks: retention_ticks,
            ..Self::new()
        }
    }

    /// Current world tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Advance to the next tick (called once per frame).
    pub fn tick(&mut self) {
        self.current_tick += 1;
        self.prune_removals();
    }

    /// Advance by N ticks.
    pub fn advance(&mut self, n: u64) {
        self.current_tick += n;
        self.prune_removals();
    }

    // ── Component lifecycle events ──

    /// Record that a component was added to an entity.
    pub fn record_added(&mut self, entity_id: u64, type_id: TypeId) {
        self.component_ticks.insert(
            (entity_id, type_id),
            ComponentTick::new(self.current_tick),
        );
    }

    /// Record that a component was changed on an entity.
    pub fn record_changed(&mut self, entity_id: u64, type_id: TypeId) {
        if let Some(ct) = self.component_ticks.get_mut(&(entity_id, type_id)) {
            ct.mark_changed(self.current_tick);
        } else {
            // Treat as added if not tracked yet.
            self.record_added(entity_id, type_id);
        }
    }

    /// Record that a component was removed from an entity.
    pub fn record_removed(&mut self, entity_id: u64, type_id: TypeId) {
        self.component_ticks.remove(&(entity_id, type_id));
        self.removal_log.push(RemovedRecord {
            entity_id,
            type_id,
            removed_tick: self.current_tick,
        });
    }

    /// Record that an entity was fully despawned (all components removed).
    pub fn record_entity_despawned(&mut self, entity_id: u64) {
        let keys: Vec<(u64, TypeId)> = self
            .component_ticks
            .keys()
            .filter(|(eid, _)| *eid == entity_id)
            .cloned()
            .collect();
        for (eid, tid) in keys {
            self.record_removed(eid, tid);
        }
    }

    // ── Query methods ──

    /// Get the tick info for a specific entity/component.
    pub fn get_tick(&self, entity_id: u64, type_id: TypeId) -> Option<&ComponentTick> {
        self.component_ticks.get(&(entity_id, type_id))
    }

    /// Was this component added since `since_tick`?
    pub fn is_added_since(
        &self,
        entity_id: u64,
        type_id: TypeId,
        since_tick: u64,
    ) -> bool {
        self.component_ticks
            .get(&(entity_id, type_id))
            .map(|ct| ct.is_added_since(since_tick))
            .unwrap_or(false)
    }

    /// Was this component changed since `since_tick`?
    pub fn is_changed_since(
        &self,
        entity_id: u64,
        type_id: TypeId,
        since_tick: u64,
    ) -> bool {
        self.component_ticks
            .get(&(entity_id, type_id))
            .map(|ct| ct.is_changed_since(since_tick))
            .unwrap_or(false)
    }

    /// Get all entities whose component of `type_id` was added since `since_tick`.
    pub fn added_entities_since(
        &self,
        type_id: TypeId,
        since_tick: u64,
    ) -> Vec<u64> {
        self.component_ticks
            .iter()
            .filter(|((_, tid), ct)| *tid == type_id && ct.is_added_since(since_tick))
            .map(|((eid, _), _)| *eid)
            .collect()
    }

    /// Get all entities whose component of `type_id` was changed since `since_tick`.
    pub fn changed_entities_since(
        &self,
        type_id: TypeId,
        since_tick: u64,
    ) -> Vec<u64> {
        self.component_ticks
            .iter()
            .filter(|((_, tid), ct)| *tid == type_id && ct.is_changed_since(since_tick))
            .map(|((eid, _), _)| *eid)
            .collect()
    }

    /// Get all removal records for a component type since `since_tick`.
    pub fn removed_since(
        &self,
        type_id: TypeId,
        since_tick: u64,
    ) -> Vec<&RemovedRecord> {
        self.removal_log
            .iter()
            .filter(|r| r.type_id == type_id && r.removed_tick > since_tick)
            .collect()
    }

    /// Get all change events for a component type since `since_tick`.
    pub fn events_since(
        &self,
        type_id: TypeId,
        since_tick: u64,
    ) -> Vec<(u64, ChangeEvent)> {
        let mut events = Vec::new();
        for ((eid, tid), ct) in &self.component_ticks {
            if *tid != type_id {
                continue;
            }
            if ct.is_added_since(since_tick) {
                events.push((*eid, ChangeEvent::Added));
            } else if ct.is_changed_since(since_tick) {
                events.push((*eid, ChangeEvent::Changed));
            }
        }
        for record in &self.removal_log {
            if record.type_id == type_id && record.removed_tick > since_tick {
                events.push((record.entity_id, ChangeEvent::Removed));
            }
        }
        events
    }

    // ── System tick management ──

    /// Register a system with the current tick as its last-run tick.
    pub fn register_system(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.system_ticks.entry(name).or_insert(self.current_tick);
    }

    /// Record that a system just ran (update its last-run tick).
    pub fn mark_system_run(&mut self, name: &str) {
        self.system_ticks
            .insert(name.to_string(), self.current_tick);
    }

    /// Get the last-run tick for a system.
    pub fn system_last_run(&self, name: &str) -> Option<u64> {
        self.system_ticks.get(name).copied()
    }

    /// Get all entities with added components since a system's last run.
    pub fn added_since_system_run(
        &self,
        type_id: TypeId,
        system_name: &str,
    ) -> Vec<u64> {
        let since = self.system_ticks.get(system_name).copied().unwrap_or(0);
        self.added_entities_since(type_id, since)
    }

    /// Get all entities with changed components since a system's last run.
    pub fn changed_since_system_run(
        &self,
        type_id: TypeId,
        system_name: &str,
    ) -> Vec<u64> {
        let since = self.system_ticks.get(system_name).copied().unwrap_or(0);
        self.changed_entities_since(type_id, since)
    }

    /// Get removal records since a system's last run.
    pub fn removed_since_system_run(
        &self,
        type_id: TypeId,
        system_name: &str,
    ) -> Vec<&RemovedRecord> {
        let since = self.system_ticks.get(system_name).copied().unwrap_or(0);
        self.removed_since(type_id, since)
    }

    // ── Internal ──

    fn prune_removals(&mut self) {
        if self.current_tick > self.removal_retention_ticks {
            let cutoff = self.current_tick - self.removal_retention_ticks;
            self.removal_log.retain(|r| r.removed_tick >= cutoff);
        }
    }

    /// Total number of tracked component ticks.
    pub fn tracked_count(&self) -> usize {
        self.component_ticks.len()
    }

    /// Number of removal records in the log.
    pub fn removal_log_size(&self) -> usize {
        self.removal_log.len()
    }

    /// Number of registered systems.
    pub fn system_count(&self) -> usize {
        self.system_ticks.len()
    }

    /// Clear all tracking data.
    pub fn clear(&mut self) {
        self.component_ticks.clear();
        self.removal_log.clear();
        self.system_ticks.clear();
        self.current_tick = 0;
    }
}

impl Default for ChangeDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    struct Pos;
    struct Vel;
    struct Hp;

    #[test]
    fn initial_tick() {
        let cd = ChangeDetector::new();
        assert_eq!(cd.current_tick(), 0);
    }

    #[test]
    fn tick_advances() {
        let mut cd = ChangeDetector::new();
        cd.tick();
        assert_eq!(cd.current_tick(), 1);
        cd.tick();
        assert_eq!(cd.current_tick(), 2);
    }

    #[test]
    fn advance_multiple() {
        let mut cd = ChangeDetector::new();
        cd.advance(5);
        assert_eq!(cd.current_tick(), 5);
    }

    #[test]
    fn record_added() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // tick = 1
        cd.record_added(1, TypeId::of::<Pos>());
        let ct = cd.get_tick(1, TypeId::of::<Pos>()).unwrap();
        assert_eq!(ct.added_tick, 1);
        assert_eq!(ct.changed_tick, 1);
    }

    #[test]
    fn record_changed() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // tick = 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.tick(); // tick = 2
        cd.tick(); // tick = 3
        cd.record_changed(1, TypeId::of::<Pos>());
        let ct = cd.get_tick(1, TypeId::of::<Pos>()).unwrap();
        assert_eq!(ct.added_tick, 1);
        assert_eq!(ct.changed_tick, 3);
    }

    #[test]
    fn record_removed() {
        let mut cd = ChangeDetector::new();
        cd.tick();
        cd.record_added(1, TypeId::of::<Pos>());
        cd.tick();
        cd.record_removed(1, TypeId::of::<Pos>());
        assert!(cd.get_tick(1, TypeId::of::<Pos>()).is_none());
        assert_eq!(cd.removal_log_size(), 1);
    }

    #[test]
    fn is_added_since() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // tick = 1
        cd.record_added(1, TypeId::of::<Pos>());
        assert!(cd.is_added_since(1, TypeId::of::<Pos>(), 0));
        assert!(!cd.is_added_since(1, TypeId::of::<Pos>(), 1));
        assert!(!cd.is_added_since(1, TypeId::of::<Pos>(), 2));
    }

    #[test]
    fn is_changed_since() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.tick(); // 2
        cd.tick(); // 3
        cd.record_changed(1, TypeId::of::<Pos>());
        assert!(cd.is_changed_since(1, TypeId::of::<Pos>(), 2));
        assert!(!cd.is_changed_since(1, TypeId::of::<Pos>(), 3));
    }

    #[test]
    fn added_entities_since() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.record_added(2, TypeId::of::<Pos>());
        cd.tick(); // 2
        cd.record_added(3, TypeId::of::<Pos>());
        let mut added = cd.added_entities_since(TypeId::of::<Pos>(), 1);
        added.sort();
        assert_eq!(added, vec![3]);
    }

    #[test]
    fn changed_entities_since() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.record_added(2, TypeId::of::<Pos>());
        cd.tick(); // 2
        cd.record_changed(1, TypeId::of::<Pos>());
        let result = cd.changed_entities_since(TypeId::of::<Pos>(), 1);
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn removed_since() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.tick(); // 2
        cd.record_removed(1, TypeId::of::<Pos>());
        let records = cd.removed_since(TypeId::of::<Pos>(), 1);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].entity_id, 1);
        assert_eq!(records[0].removed_tick, 2);
    }

    #[test]
    fn events_since() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.record_added(2, TypeId::of::<Pos>());
        cd.tick(); // 2
        cd.record_changed(2, TypeId::of::<Pos>());
        cd.record_removed(1, TypeId::of::<Pos>());
        cd.record_added(3, TypeId::of::<Pos>());
        let events = cd.events_since(TypeId::of::<Pos>(), 1);
        let event_types: HashSet<ChangeEvent> =
            events.iter().map(|(_, e)| *e).collect();
        assert!(event_types.contains(&ChangeEvent::Added));
        assert!(event_types.contains(&ChangeEvent::Changed));
        assert!(event_types.contains(&ChangeEvent::Removed));
    }

    #[test]
    fn system_tick_tracking() {
        let mut cd = ChangeDetector::new();
        cd.register_system("physics");
        assert_eq!(cd.system_last_run("physics"), Some(0));
        cd.tick(); // 1
        cd.tick(); // 2
        cd.mark_system_run("physics");
        assert_eq!(cd.system_last_run("physics"), Some(2));
    }

    #[test]
    fn added_since_system_run() {
        let mut cd = ChangeDetector::new();
        cd.register_system("render");
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.mark_system_run("render"); // run at tick 1
        cd.tick(); // 2
        cd.record_added(2, TypeId::of::<Pos>());
        let added = cd.added_since_system_run(TypeId::of::<Pos>(), "render");
        assert_eq!(added, vec![2]);
    }

    #[test]
    fn changed_since_system_run() {
        let mut cd = ChangeDetector::new();
        cd.register_system("ai");
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Vel>());
        cd.mark_system_run("ai"); // run at tick 1
        cd.tick(); // 2
        cd.record_changed(1, TypeId::of::<Vel>());
        let changed = cd.changed_since_system_run(TypeId::of::<Vel>(), "ai");
        assert_eq!(changed, vec![1]);
    }

    #[test]
    fn removed_since_system_run() {
        let mut cd = ChangeDetector::new();
        cd.register_system("cleanup");
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Hp>());
        cd.mark_system_run("cleanup");
        cd.tick(); // 2
        cd.record_removed(1, TypeId::of::<Hp>());
        let removed = cd.removed_since_system_run(TypeId::of::<Hp>(), "cleanup");
        assert_eq!(removed.len(), 1);
    }

    #[test]
    fn removal_log_pruning() {
        let mut cd = ChangeDetector::with_retention(3);
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.record_removed(1, TypeId::of::<Pos>());
        assert_eq!(cd.removal_log_size(), 1);
        // Advance past retention window.
        cd.advance(10); // tick = 11, cutoff = 11-3 = 8
        assert_eq!(cd.removal_log_size(), 0);
    }

    #[test]
    fn entity_despawned() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.record_added(1, TypeId::of::<Vel>());
        cd.record_entity_despawned(1);
        assert!(cd.get_tick(1, TypeId::of::<Pos>()).is_none());
        assert!(cd.get_tick(1, TypeId::of::<Vel>()).is_none());
        assert_eq!(cd.removal_log_size(), 2);
    }

    #[test]
    fn tracked_count() {
        let mut cd = ChangeDetector::new();
        cd.record_added(1, TypeId::of::<Pos>());
        cd.record_added(2, TypeId::of::<Vel>());
        assert_eq!(cd.tracked_count(), 2);
    }

    #[test]
    fn system_count() {
        let mut cd = ChangeDetector::new();
        cd.register_system("a");
        cd.register_system("b");
        assert_eq!(cd.system_count(), 2);
    }

    #[test]
    fn clear_resets_everything() {
        let mut cd = ChangeDetector::new();
        cd.tick();
        cd.record_added(1, TypeId::of::<Pos>());
        cd.register_system("test");
        cd.clear();
        assert_eq!(cd.current_tick(), 0);
        assert_eq!(cd.tracked_count(), 0);
        assert_eq!(cd.system_count(), 0);
        assert_eq!(cd.removal_log_size(), 0);
    }

    #[test]
    fn component_tick_methods() {
        let mut ct = ComponentTick::new(5);
        assert!(ct.is_added_since(4));
        assert!(!ct.is_added_since(5));
        assert!(ct.is_changed_since(4));
        ct.mark_changed(10);
        assert!(ct.is_changed_since(9));
        assert!(!ct.is_changed_since(10));
        // Added tick unchanged.
        assert_eq!(ct.added_tick, 5);
    }

    #[test]
    fn record_changed_auto_adds() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        // Change without prior add should auto-add.
        cd.record_changed(1, TypeId::of::<Pos>());
        let ct = cd.get_tick(1, TypeId::of::<Pos>()).unwrap();
        assert_eq!(ct.added_tick, 1);
    }

    #[test]
    fn unregistered_system_defaults_to_zero() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        // Unregistered system -> since_tick = 0.
        let added = cd.added_since_system_run(TypeId::of::<Pos>(), "never_registered");
        assert_eq!(added, vec![1]);
    }

    #[test]
    fn multiple_types_per_entity() {
        let mut cd = ChangeDetector::new();
        cd.tick(); // 1
        cd.record_added(1, TypeId::of::<Pos>());
        cd.tick(); // 2
        cd.record_added(1, TypeId::of::<Vel>());
        assert!(cd.is_added_since(1, TypeId::of::<Pos>(), 0));
        assert!(cd.is_added_since(1, TypeId::of::<Vel>(), 1));
        assert!(!cd.is_added_since(1, TypeId::of::<Vel>(), 2));
    }
}
