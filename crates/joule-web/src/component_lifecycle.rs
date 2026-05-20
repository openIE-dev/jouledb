//! Component lifecycle management — mount/update/unmount hooks, props comparison,
//! state management, error boundaries, and lifecycle event ordering.
//!
//! Replaces React class component lifecycle / Vue Options API lifecycle with a
//! pure-Rust model. Tracks lifecycle phases, fires hooks in correct order, and
//! supports error boundaries that catch child component failures.

use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────────────

/// Lifecycle domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleError {
    /// Component not found.
    NotFound(u64),
    /// Component already mounted.
    AlreadyMounted(u64),
    /// Component not mounted (cannot update/unmount).
    NotMounted(u64),
    /// Error boundary caught an error.
    BoundaryCaught { boundary_id: u64, error_msg: String },
    /// Hook invocation failed.
    HookFailed { component_id: u64, hook: HookType, reason: String },
    /// Duplicate component ID.
    DuplicateId(u64),
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "component not found: {id}"),
            Self::AlreadyMounted(id) => write!(f, "component {id} already mounted"),
            Self::NotMounted(id) => write!(f, "component {id} not mounted"),
            Self::BoundaryCaught { boundary_id, error_msg } => {
                write!(f, "error boundary {boundary_id}: {error_msg}")
            }
            Self::HookFailed { component_id, hook, reason } => {
                write!(f, "hook {hook:?} failed on {component_id}: {reason}")
            }
            Self::DuplicateId(id) => write!(f, "duplicate component id: {id}"),
        }
    }
}

impl std::error::Error for LifecycleError {}

// ── Types ───────────────────────────────────────────────────────────────

/// Lifecycle phase of a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Created,
    BeforeMount,
    Mounted,
    BeforeUpdate,
    Updated,
    BeforeUnmount,
    Unmounted,
    ErrorCaught,
}

/// Hook types that can fire during lifecycle transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookType {
    OnBeforeMount,
    OnMounted,
    OnBeforeUpdate,
    OnUpdated,
    OnBeforeUnmount,
    OnUnmounted,
    OnErrorCaught,
}

/// A recorded lifecycle event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleEvent {
    pub component_id: u64,
    pub hook: HookType,
    pub sequence: u64,
}

/// Props for a component — simple key/value pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct Props {
    values: HashMap<String, String>,
}

impl Props {
    pub fn new() -> Self {
        Self { values: HashMap::new() }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    pub fn keys(&self) -> Vec<String> {
        let mut ks: Vec<String> = self.values.keys().cloned().collect();
        ks.sort();
        ks
    }

    /// Check if these props differ from another set (should_update).
    pub fn differs_from(&self, other: &Props) -> bool {
        self.values != other.values
    }
}

impl Default for Props {
    fn default() -> Self {
        Self::new()
    }
}

/// Local state for a component.
#[derive(Debug, Clone)]
pub struct ComponentState {
    values: HashMap<String, String>,
    version: u64,
}

impl ComponentState {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            version: 0,
        }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
        self.version += 1;
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Default for ComponentState {
    fn default() -> Self {
        Self::new()
    }
}

/// A component in the tree.
#[derive(Debug, Clone)]
pub struct ComponentEntry {
    pub id: u64,
    pub name: String,
    pub phase: Phase,
    pub props: Props,
    pub state: ComponentState,
    pub parent_id: Option<u64>,
    pub children_ids: Vec<u64>,
    pub is_error_boundary: bool,
    pub error_fallback: Option<String>,
    pub caught_error: Option<String>,
}

// ── ComponentTree ───────────────────────────────────────────────────────

/// Manages a tree of components with lifecycle hooks.
pub struct ComponentTree {
    components: HashMap<u64, ComponentEntry>,
    root_ids: Vec<u64>,
    events: Vec<LifecycleEvent>,
    sequence_counter: u64,
}

impl ComponentTree {
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
            root_ids: Vec::new(),
            events: Vec::new(),
            sequence_counter: 0,
        }
    }

    /// Register a new component.
    pub fn register(
        &mut self,
        id: u64,
        name: &str,
        parent_id: Option<u64>,
    ) -> Result<(), LifecycleError> {
        if self.components.contains_key(&id) {
            return Err(LifecycleError::DuplicateId(id));
        }

        let entry = ComponentEntry {
            id,
            name: name.to_string(),
            phase: Phase::Created,
            props: Props::new(),
            state: ComponentState::new(),
            parent_id,
            children_ids: Vec::new(),
            is_error_boundary: false,
            error_fallback: None,
            caught_error: None,
        };
        self.components.insert(id, entry);

        if let Some(pid) = parent_id {
            if let Some(parent) = self.components.get_mut(&pid) {
                parent.children_ids.push(id);
            }
        } else {
            self.root_ids.push(id);
        }

        Ok(())
    }

    /// Set a component as an error boundary.
    pub fn set_error_boundary(&mut self, id: u64, fallback: &str) -> Result<(), LifecycleError> {
        let comp = self.components.get_mut(&id).ok_or(LifecycleError::NotFound(id))?;
        comp.is_error_boundary = true;
        comp.error_fallback = Some(fallback.to_string());
        Ok(())
    }

    /// Mount a component (Created -> Mounted).
    pub fn mount(&mut self, id: u64) -> Result<(), LifecycleError> {
        let phase = {
            let comp = self.components.get(&id).ok_or(LifecycleError::NotFound(id))?;
            comp.phase
        };
        if phase != Phase::Created {
            return Err(LifecycleError::AlreadyMounted(id));
        }

        self.set_phase(id, Phase::BeforeMount);
        self.record_event(id, HookType::OnBeforeMount);

        self.set_phase(id, Phase::Mounted);
        self.record_event(id, HookType::OnMounted);

        Ok(())
    }

    /// Update a component with new props.
    pub fn update(&mut self, id: u64, new_props: Props) -> Result<bool, LifecycleError> {
        let (phase, should) = {
            let comp = self.components.get(&id).ok_or(LifecycleError::NotFound(id))?;
            (comp.phase, comp.props.differs_from(&new_props))
        };

        if phase != Phase::Mounted && phase != Phase::Updated {
            return Err(LifecycleError::NotMounted(id));
        }

        if !should {
            return Ok(false);
        }

        self.set_phase(id, Phase::BeforeUpdate);
        self.record_event(id, HookType::OnBeforeUpdate);

        // Apply new props
        if let Some(comp) = self.components.get_mut(&id) {
            comp.props = new_props;
        }

        self.set_phase(id, Phase::Updated);
        self.record_event(id, HookType::OnUpdated);

        Ok(true)
    }

    /// Unmount a component (and its subtree).
    pub fn unmount(&mut self, id: u64) -> Result<(), LifecycleError> {
        let (phase, children) = {
            let comp = self.components.get(&id).ok_or(LifecycleError::NotFound(id))?;
            (comp.phase, comp.children_ids.clone())
        };

        if phase == Phase::Unmounted || phase == Phase::Created {
            return Err(LifecycleError::NotMounted(id));
        }

        // Unmount children first (depth-first)
        for child_id in children {
            let child_phase = self
                .components
                .get(&child_id)
                .map(|c| c.phase)
                .unwrap_or(Phase::Unmounted);
            if child_phase != Phase::Unmounted && child_phase != Phase::Created {
                let _ = self.unmount(child_id);
            }
        }

        self.set_phase(id, Phase::BeforeUnmount);
        self.record_event(id, HookType::OnBeforeUnmount);

        self.set_phase(id, Phase::Unmounted);
        self.record_event(id, HookType::OnUnmounted);

        Ok(())
    }

    /// Report an error in a component — propagates to nearest error boundary.
    pub fn report_error(&mut self, id: u64, error_msg: &str) -> Result<u64, LifecycleError> {
        let boundary_id = self.find_error_boundary(id);

        match boundary_id {
            Some(bid) => {
                if let Some(boundary) = self.components.get_mut(&bid) {
                    boundary.phase = Phase::ErrorCaught;
                    boundary.caught_error = Some(error_msg.to_string());
                }
                self.record_event(bid, HookType::OnErrorCaught);
                Ok(bid)
            }
            None => Err(LifecycleError::BoundaryCaught {
                boundary_id: 0,
                error_msg: format!("no error boundary found for component {id}: {error_msg}"),
            }),
        }
    }

    /// Find the nearest error boundary ancestor of a component.
    fn find_error_boundary(&self, id: u64) -> Option<u64> {
        let mut current = self.components.get(&id)?;

        // Check self first
        if current.is_error_boundary {
            return Some(current.id);
        }

        // Walk up the tree
        while let Some(pid) = current.parent_id {
            let parent = self.components.get(&pid)?;
            if parent.is_error_boundary {
                return Some(parent.id);
            }
            current = parent;
        }
        None
    }

    /// Get a component by ID.
    pub fn get(&self, id: u64) -> Option<&ComponentEntry> {
        self.components.get(&id)
    }

    /// Get a mutable reference to a component.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut ComponentEntry> {
        self.components.get_mut(&id)
    }

    /// Set state on a component.
    pub fn set_state(&mut self, id: u64, key: &str, value: &str) -> Result<(), LifecycleError> {
        let comp = self.components.get_mut(&id).ok_or(LifecycleError::NotFound(id))?;
        comp.state.set(key, value);
        Ok(())
    }

    /// Get state from a component.
    pub fn get_state(&self, id: u64, key: &str) -> Option<&str> {
        self.components.get(&id)?.state.get(key)
    }

    /// Get the recorded lifecycle events in order.
    pub fn events(&self) -> &[LifecycleEvent] {
        &self.events
    }

    /// Total component count.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Get root component IDs.
    pub fn root_ids(&self) -> &[u64] {
        &self.root_ids
    }

    /// Get children IDs of a component.
    pub fn children_ids(&self, id: u64) -> Option<&[u64]> {
        self.components.get(&id).map(|c| c.children_ids.as_slice())
    }

    /// Clear all recorded events.
    pub fn clear_events(&mut self) {
        self.events.clear();
    }

    fn set_phase(&mut self, id: u64, phase: Phase) {
        if let Some(comp) = self.components.get_mut(&id) {
            comp.phase = phase;
        }
    }

    fn record_event(&mut self, component_id: u64, hook: HookType) {
        self.sequence_counter += 1;
        self.events.push(LifecycleEvent {
            component_id,
            hook,
            sequence: self.sequence_counter,
        });
    }
}

impl Default for ComponentTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_tree() -> ComponentTree {
        let mut tree = ComponentTree::new();
        tree.register(1, "App", None).unwrap();
        tree.register(2, "Header", Some(1)).unwrap();
        tree.register(3, "Body", Some(1)).unwrap();
        tree
    }

    #[test]
    fn register_and_count() {
        let tree = setup_tree();
        assert_eq!(tree.component_count(), 3);
    }

    #[test]
    fn register_duplicate_errors() {
        let mut tree = setup_tree();
        let err = tree.register(1, "Dup", None).unwrap_err();
        assert!(matches!(err, LifecycleError::DuplicateId(1)));
    }

    #[test]
    fn mount_lifecycle_order() {
        let mut tree = setup_tree();
        tree.mount(1).unwrap();
        let events = tree.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].hook, HookType::OnBeforeMount);
        assert_eq!(events[1].hook, HookType::OnMounted);
        assert!(events[0].sequence < events[1].sequence);
    }

    #[test]
    fn mount_twice_errors() {
        let mut tree = setup_tree();
        tree.mount(1).unwrap();
        let err = tree.mount(1).unwrap_err();
        assert!(matches!(err, LifecycleError::AlreadyMounted(1)));
    }

    #[test]
    fn update_with_changed_props() {
        let mut tree = setup_tree();
        tree.mount(1).unwrap();
        tree.clear_events();

        let mut new_props = Props::new();
        new_props.set("title", "New Title");
        let updated = tree.update(1, new_props).unwrap();
        assert!(updated);

        let events = tree.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].hook, HookType::OnBeforeUpdate);
        assert_eq!(events[1].hook, HookType::OnUpdated);
    }

    #[test]
    fn update_with_same_props_skips() {
        let mut tree = setup_tree();
        tree.mount(1).unwrap();
        let updated = tree.update(1, Props::new()).unwrap();
        assert!(!updated);
    }

    #[test]
    fn update_unmounted_errors() {
        let mut tree = setup_tree();
        let err = tree.update(1, Props::new()).unwrap_err();
        assert!(matches!(err, LifecycleError::NotMounted(1)));
    }

    #[test]
    fn unmount_lifecycle_order() {
        let mut tree = setup_tree();
        tree.mount(1).unwrap();
        tree.mount(2).unwrap();
        tree.clear_events();

        tree.unmount(1).unwrap();
        let events = tree.events();
        // Unmount fires: child2 before_unmount + unmounted, then parent before_unmount + unmounted
        assert_eq!(events.len(), 4);
        // Child unmounted before parent
        assert_eq!(events[0].component_id, 2);
        assert_eq!(events[0].hook, HookType::OnBeforeUnmount);
        assert_eq!(events[1].component_id, 2);
        assert_eq!(events[1].hook, HookType::OnUnmounted);
        assert_eq!(events[2].component_id, 1);
        assert_eq!(events[2].hook, HookType::OnBeforeUnmount);
    }

    #[test]
    fn state_management() {
        let mut tree = setup_tree();
        tree.set_state(1, "count", "42").unwrap();
        assert_eq!(tree.get_state(1, "count"), Some("42"));
        assert_eq!(tree.get_state(1, "missing"), None);
    }

    #[test]
    fn state_version_increments() {
        let mut tree = setup_tree();
        tree.set_state(1, "a", "1").unwrap();
        tree.set_state(1, "b", "2").unwrap();
        let comp = tree.get(1).unwrap();
        assert_eq!(comp.state.version(), 2);
    }

    #[test]
    fn error_boundary_catches() {
        let mut tree = setup_tree();
        tree.set_error_boundary(1, "<ErrorFallback />").unwrap();
        tree.mount(1).unwrap();
        tree.mount(2).unwrap();
        tree.clear_events();

        let boundary = tree.report_error(2, "render failed").unwrap();
        assert_eq!(boundary, 1);

        let comp = tree.get(1).unwrap();
        assert_eq!(comp.phase, Phase::ErrorCaught);
        assert_eq!(comp.caught_error.as_deref(), Some("render failed"));
    }

    #[test]
    fn error_without_boundary() {
        let mut tree = ComponentTree::new();
        tree.register(1, "Orphan", None).unwrap();
        let err = tree.report_error(1, "oops").unwrap_err();
        assert!(matches!(err, LifecycleError::BoundaryCaught { .. }));
    }

    #[test]
    fn parent_child_structure() {
        let tree = setup_tree();
        assert_eq!(tree.root_ids(), &[1]);
        assert_eq!(tree.children_ids(1), Some([2, 3].as_slice()));
        assert_eq!(tree.children_ids(2), Some([].as_slice()));
    }

    #[test]
    fn props_differs_from() {
        let mut a = Props::new();
        a.set("x", "1");
        let mut b = Props::new();
        b.set("x", "2");
        assert!(a.differs_from(&b));
        assert!(!a.differs_from(&a));
    }

    #[test]
    fn component_not_found() {
        let tree = ComponentTree::new();
        assert!(tree.get(999).is_none());
    }

    #[test]
    fn full_lifecycle_sequence() {
        let mut tree = ComponentTree::new();
        tree.register(1, "App", None).unwrap();
        tree.mount(1).unwrap();

        let mut p = Props::new();
        p.set("v", "1");
        tree.update(1, p).unwrap();

        tree.unmount(1).unwrap();

        let events = tree.events();
        let hooks: Vec<HookType> = events.iter().map(|e| e.hook).collect();
        assert_eq!(hooks, vec![
            HookType::OnBeforeMount,
            HookType::OnMounted,
            HookType::OnBeforeUpdate,
            HookType::OnUpdated,
            HookType::OnBeforeUnmount,
            HookType::OnUnmounted,
        ]);
    }

    #[test]
    fn props_keys_sorted() {
        let mut p = Props::new();
        p.set("z", "1");
        p.set("a", "2");
        p.set("m", "3");
        assert_eq!(p.keys(), vec!["a", "m", "z"]);
    }

    #[test]
    fn component_state_empty_check() {
        let s = ComponentState::new();
        assert!(s.is_empty());
        assert_eq!(s.version(), 0);
    }
}
