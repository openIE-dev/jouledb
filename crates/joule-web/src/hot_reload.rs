//! Hot reload concepts — file change detection modeling, module dependency graph,
//! invalidation cascades, HMR-style partial updates, and reload events.
//!
//! Replaces Webpack HMR, Vite HMR, and live-reload with a pure-Rust model of
//! hot module replacement state machines and dependency-aware invalidation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Hot reload errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotReloadError {
    /// Module not found.
    ModuleNotFound(String),
    /// Circular dependency detected.
    CircularDependency(Vec<String>),
    /// Module cannot accept hot update.
    CannotAccept(String),
    /// Reload already in progress.
    ReloadInProgress,
    /// Invalid state transition.
    InvalidState { from: ModuleState, to: ModuleState },
}

impl fmt::Display for HotReloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModuleNotFound(id) => write!(f, "module not found: {id}"),
            Self::CircularDependency(path) => {
                write!(f, "circular dependency: {}", path.join(" -> "))
            }
            Self::CannotAccept(id) => write!(f, "module cannot accept update: {id}"),
            Self::ReloadInProgress => write!(f, "reload already in progress"),
            Self::InvalidState { from, to } => {
                write!(f, "invalid state transition: {from} -> {to}")
            }
        }
    }
}

impl std::error::Error for HotReloadError {}

// ── Module State ───────────────────────────────────────────────

/// State of a module in the HMR lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleState {
    /// Module loaded and active.
    Active,
    /// Module changed, pending update.
    Changed,
    /// Module is being updated.
    Updating,
    /// Module update applied successfully.
    Updated,
    /// Module invalidated (needs reload propagation).
    Invalidated,
    /// Module errored during update.
    Errored,
    /// Module disposed (cleanup complete).
    Disposed,
}

impl fmt::Display for ModuleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Changed => write!(f, "changed"),
            Self::Updating => write!(f, "updating"),
            Self::Updated => write!(f, "updated"),
            Self::Invalidated => write!(f, "invalidated"),
            Self::Errored => write!(f, "errored"),
            Self::Disposed => write!(f, "disposed"),
        }
    }
}

impl ModuleState {
    /// Check if transition from this state to target is valid.
    pub fn can_transition_to(&self, target: ModuleState) -> bool {
        matches!(
            (self, target),
            (Self::Active, ModuleState::Changed)
                | (Self::Active, ModuleState::Invalidated)
                | (Self::Changed, ModuleState::Updating)
                | (Self::Changed, ModuleState::Invalidated)
                | (Self::Updating, ModuleState::Updated)
                | (Self::Updating, ModuleState::Errored)
                | (Self::Updated, ModuleState::Active)
                | (Self::Invalidated, ModuleState::Disposed)
                | (Self::Invalidated, ModuleState::Updating)
                | (Self::Errored, ModuleState::Active)
                | (Self::Errored, ModuleState::Disposed)
                | (Self::Disposed, ModuleState::Active)
        )
    }
}

// ── Change Event ───────────────────────────────────────────────

/// Type of file change detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeKind {
    /// File content modified.
    Modified,
    /// File created.
    Created,
    /// File deleted.
    Deleted,
    /// File renamed.
    Renamed,
}

/// A detected file change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChange {
    /// Path of the changed file.
    pub path: String,
    /// Kind of change.
    pub kind: ChangeKind,
    /// Timestamp (monotonic counter for ordering).
    pub timestamp: u64,
}

// ── Reload Event ───────────────────────────────────────────────

/// An event in the reload pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReloadEvent {
    /// File change detected.
    FileChanged(FileChange),
    /// Module marked as changed.
    ModuleChanged { module_id: String },
    /// Invalidation cascading to dependents.
    InvalidationCascade { root: String, affected: Vec<String> },
    /// Module update applied.
    ModuleUpdated { module_id: String },
    /// Full reload required (HMR boundary not found).
    FullReloadRequired { reason: String },
    /// Update completed.
    UpdateComplete { updated: Vec<String>, duration_us: u64 },
    /// Update failed.
    UpdateFailed { module_id: String, error: String },
}

// ── Module Info ────────────────────────────────────────────────

/// Information about a module in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module identifier (usually file path).
    pub id: String,
    /// Current state.
    pub state: ModuleState,
    /// Whether this module can accept hot updates.
    pub accepts_hot: bool,
    /// Modules this module depends on (imports).
    pub dependencies: HashSet<String>,
    /// Modules that depend on this module.
    pub dependents: HashSet<String>,
    /// Version counter (increments on each update).
    pub version: u64,
}

impl ModuleInfo {
    /// Create a new module.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            state: ModuleState::Active,
            accepts_hot: true,
            dependencies: HashSet::new(),
            dependents: HashSet::new(),
            version: 0,
        }
    }

    /// Set whether module accepts hot updates.
    pub fn with_hot_accept(mut self, accepts: bool) -> Self {
        self.accepts_hot = accepts;
        self
    }
}

// ── Module Graph ───────────────────────────────────────────────

/// Dependency graph for modules, managing HMR state.
#[derive(Debug, Clone)]
pub struct ModuleGraph {
    modules: HashMap<String, ModuleInfo>,
    event_log: Vec<ReloadEvent>,
    reload_in_progress: bool,
}

impl ModuleGraph {
    /// Create a new empty module graph.
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            event_log: Vec::new(),
            reload_in_progress: false,
        }
    }

    /// Add a module to the graph.
    pub fn add_module(&mut self, module: ModuleInfo) {
        self.modules.insert(module.id.clone(), module);
    }

    /// Remove a module from the graph.
    pub fn remove_module(&mut self, id: &str) -> Option<ModuleInfo> {
        if let Some(module) = self.modules.remove(id) {
            // Clean up dependency references
            let deps = module.dependencies.clone();
            for dep_id in &deps {
                if let Some(dep) = self.modules.get_mut(dep_id) {
                    dep.dependents.remove(id);
                }
            }
            let dependents = module.dependents.clone();
            for dep_id in &dependents {
                if let Some(dep) = self.modules.get_mut(dep_id) {
                    dep.dependencies.remove(id);
                }
            }
            Some(module)
        } else {
            None
        }
    }

    /// Add a dependency edge: `from` depends on `to`.
    pub fn add_dependency(&mut self, from: &str, to: &str) -> Result<(), HotReloadError> {
        if !self.modules.contains_key(from) {
            return Err(HotReloadError::ModuleNotFound(from.to_string()));
        }
        if !self.modules.contains_key(to) {
            return Err(HotReloadError::ModuleNotFound(to.to_string()));
        }

        // Check for circular dependency
        if self.has_path(to, from) {
            return Err(HotReloadError::CircularDependency(vec![
                from.to_string(),
                to.to_string(),
                from.to_string(),
            ]));
        }

        self.modules.get_mut(from).unwrap().dependencies.insert(to.to_string());
        self.modules.get_mut(to).unwrap().dependents.insert(from.to_string());
        Ok(())
    }

    /// Check if there is a path from `start` to `end` in the dependency graph.
    pub fn has_path(&self, start: &str, end: &str) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            if current == end {
                return true;
            }
            if !visited.insert(current.clone()) {
                continue;
            }
            if let Some(module) = self.modules.get(&current) {
                for dep in &module.dependencies {
                    if !visited.contains(dep) {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }
        false
    }

    /// Get a module by ID.
    pub fn get_module(&self, id: &str) -> Option<&ModuleInfo> {
        self.modules.get(id)
    }

    /// Get all module IDs (sorted).
    pub fn module_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.modules.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Number of modules.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Transition a module state.
    fn transition(&mut self, id: &str, new_state: ModuleState) -> Result<(), HotReloadError> {
        let module = self.modules.get_mut(id)
            .ok_or_else(|| HotReloadError::ModuleNotFound(id.to_string()))?;
        if !module.state.can_transition_to(new_state) {
            return Err(HotReloadError::InvalidState {
                from: module.state,
                to: new_state,
            });
        }
        module.state = new_state;
        Ok(())
    }

    /// Process a file change, computing the invalidation cascade.
    pub fn process_change(&mut self, change: FileChange) -> Result<Vec<ReloadEvent>, HotReloadError> {
        if self.reload_in_progress {
            return Err(HotReloadError::ReloadInProgress);
        }

        let module_id = change.path.clone();
        if !self.modules.contains_key(&module_id) {
            return Ok(vec![ReloadEvent::FileChanged(change)]);
        }

        self.reload_in_progress = true;
        let mut events = vec![ReloadEvent::FileChanged(change)];

        // Mark the changed module
        self.transition(&module_id, ModuleState::Changed)?;
        events.push(ReloadEvent::ModuleChanged {
            module_id: module_id.clone(),
        });

        // Compute invalidation cascade (BFS up the dependent chain)
        let affected = self.compute_invalidation_cascade(&module_id);
        if !affected.is_empty() {
            events.push(ReloadEvent::InvalidationCascade {
                root: module_id.clone(),
                affected: affected.clone(),
            });
        }

        // Check if we can do HMR or need full reload
        let needs_full = self.needs_full_reload(&module_id, &affected);

        if needs_full {
            events.push(ReloadEvent::FullReloadRequired {
                reason: format!("no HMR boundary found for {module_id}"),
            });
        } else {
            // Simulate update
            self.transition(&module_id, ModuleState::Updating)?;
            self.transition(&module_id, ModuleState::Updated)?;
            if let Some(m) = self.modules.get_mut(&module_id) {
                m.version += 1;
            }
            self.transition(&module_id, ModuleState::Active)?;

            let mut updated = vec![module_id.clone()];
            updated.extend(affected);
            events.push(ReloadEvent::UpdateComplete {
                updated: updated.clone(),
                duration_us: 0,
            });

            for mod_id in &updated {
                events.push(ReloadEvent::ModuleUpdated {
                    module_id: mod_id.clone(),
                });
            }
        }

        self.reload_in_progress = false;
        self.event_log.extend(events.clone());
        Ok(events)
    }

    /// Compute which modules are invalidated by a change to `root`.
    fn compute_invalidation_cascade(&self, root: &str) -> Vec<String> {
        let mut affected = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(module) = self.modules.get(root) {
            for dep in &module.dependents {
                queue.push_back(dep.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }
            affected.push(current.clone());
            if let Some(module) = self.modules.get(&current) {
                if !module.accepts_hot {
                    for dep in &module.dependents {
                        if !visited.contains(dep) {
                            queue.push_back(dep.clone());
                        }
                    }
                }
            }
        }

        affected.sort();
        affected
    }

    /// Check if a full reload is needed (no HMR boundary found).
    fn needs_full_reload(&self, root: &str, affected: &[String]) -> bool {
        if let Some(module) = self.modules.get(root) {
            if !module.accepts_hot {
                // Check if any affected module accepts hot
                return !affected.iter().any(|id| {
                    self.modules.get(id).map_or(false, |m| m.accepts_hot)
                });
            }
        }
        false
    }

    /// Get the event log.
    pub fn event_log(&self) -> &[ReloadEvent] {
        &self.event_log
    }

    /// Clear the event log.
    pub fn clear_log(&mut self) {
        self.event_log.clear();
    }
}

impl Default for ModuleGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> ModuleGraph {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleInfo::new("app.js"));
        g.add_module(ModuleInfo::new("header.js"));
        g.add_module(ModuleInfo::new("footer.js"));
        g.add_module(ModuleInfo::new("utils.js"));
        // app depends on header, footer; header depends on utils
        g.add_dependency("app.js", "header.js").unwrap();
        g.add_dependency("app.js", "footer.js").unwrap();
        g.add_dependency("header.js", "utils.js").unwrap();
        g
    }

    #[test]
    fn test_module_state_transitions() {
        assert!(ModuleState::Active.can_transition_to(ModuleState::Changed));
        assert!(ModuleState::Changed.can_transition_to(ModuleState::Updating));
        assert!(ModuleState::Updating.can_transition_to(ModuleState::Updated));
        assert!(ModuleState::Updated.can_transition_to(ModuleState::Active));
        assert!(!ModuleState::Active.can_transition_to(ModuleState::Updated));
        assert!(!ModuleState::Active.can_transition_to(ModuleState::Updating));
    }

    #[test]
    fn test_module_state_display() {
        assert_eq!(format!("{}", ModuleState::Active), "active");
        assert_eq!(format!("{}", ModuleState::Invalidated), "invalidated");
    }

    #[test]
    fn test_add_module() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleInfo::new("test.js"));
        assert!(g.get_module("test.js").is_some());
        assert_eq!(g.module_count(), 1);
    }

    #[test]
    fn test_remove_module() {
        let mut g = make_graph();
        g.remove_module("footer.js");
        assert!(g.get_module("footer.js").is_none());
        // app.js should no longer depend on footer.js
        let app = g.get_module("app.js").unwrap();
        assert!(!app.dependencies.contains("footer.js"));
    }

    #[test]
    fn test_add_dependency() {
        let g = make_graph();
        let app = g.get_module("app.js").unwrap();
        assert!(app.dependencies.contains("header.js"));
        assert!(app.dependencies.contains("footer.js"));
        let header = g.get_module("header.js").unwrap();
        assert!(header.dependents.contains("app.js"));
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleInfo::new("a.js"));
        g.add_module(ModuleInfo::new("b.js"));
        g.add_dependency("a.js", "b.js").unwrap();
        let result = g.add_dependency("b.js", "a.js");
        assert!(matches!(result, Err(HotReloadError::CircularDependency(_))));
    }

    #[test]
    fn test_has_path() {
        let g = make_graph();
        assert!(g.has_path("app.js", "utils.js"));
        assert!(!g.has_path("utils.js", "app.js"));
        assert!(g.has_path("app.js", "header.js"));
    }

    #[test]
    fn test_module_ids_sorted() {
        let g = make_graph();
        let ids = g.module_ids();
        assert_eq!(ids, vec!["app.js", "footer.js", "header.js", "utils.js"]);
    }

    #[test]
    fn test_process_change_hot_update() {
        let mut g = make_graph();
        let change = FileChange {
            path: "utils.js".to_string(),
            kind: ChangeKind::Modified,
            timestamp: 1,
        };
        let events = g.process_change(change).unwrap();
        // Should have file change, module changed, and update events
        let has_file_change = events.iter().any(|e| matches!(e, ReloadEvent::FileChanged(_)));
        let has_module_changed = events.iter().any(|e| matches!(e, ReloadEvent::ModuleChanged { .. }));
        let has_update_complete = events.iter().any(|e| matches!(e, ReloadEvent::UpdateComplete { .. }));
        assert!(has_file_change);
        assert!(has_module_changed);
        assert!(has_update_complete);
    }

    #[test]
    fn test_process_change_increments_version() {
        let mut g = make_graph();
        let change = FileChange {
            path: "utils.js".to_string(),
            kind: ChangeKind::Modified,
            timestamp: 1,
        };
        g.process_change(change).unwrap();
        assert_eq!(g.get_module("utils.js").unwrap().version, 1);
    }

    #[test]
    fn test_process_change_unknown_module() {
        let mut g = make_graph();
        let change = FileChange {
            path: "unknown.js".to_string(),
            kind: ChangeKind::Created,
            timestamp: 1,
        };
        let events = g.process_change(change).unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], ReloadEvent::FileChanged(_)));
    }

    #[test]
    fn test_process_change_full_reload() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleInfo::new("root.js").with_hot_accept(false));
        let change = FileChange {
            path: "root.js".to_string(),
            kind: ChangeKind::Modified,
            timestamp: 1,
        };
        let events = g.process_change(change).unwrap();
        let needs_full = events.iter().any(|e| matches!(e, ReloadEvent::FullReloadRequired { .. }));
        assert!(needs_full);
    }

    #[test]
    fn test_invalidation_cascade() {
        let g = make_graph();
        let affected = g.compute_invalidation_cascade("utils.js");
        // header.js depends on utils.js, and it accepts hot, so cascade stops there
        assert!(affected.contains(&"header.js".to_string()));
    }

    #[test]
    fn test_event_log() {
        let mut g = make_graph();
        assert!(g.event_log().is_empty());
        let change = FileChange {
            path: "footer.js".to_string(),
            kind: ChangeKind::Modified,
            timestamp: 1,
        };
        g.process_change(change).unwrap();
        assert!(!g.event_log().is_empty());
        g.clear_log();
        assert!(g.event_log().is_empty());
    }

    #[test]
    fn test_dependency_not_found() {
        let mut g = ModuleGraph::new();
        g.add_module(ModuleInfo::new("a.js"));
        let result = g.add_dependency("a.js", "nonexistent.js");
        assert!(matches!(result, Err(HotReloadError::ModuleNotFound(_))));
    }

    #[test]
    fn test_change_kind_variants() {
        let kinds = [ChangeKind::Modified, ChangeKind::Created, ChangeKind::Deleted, ChangeKind::Renamed];
        assert_eq!(kinds.len(), 4);
    }

    #[test]
    fn test_module_with_hot_accept() {
        let m = ModuleInfo::new("test.js").with_hot_accept(false);
        assert!(!m.accepts_hot);
    }

    #[test]
    fn test_error_display() {
        let err = HotReloadError::ModuleNotFound("x.js".to_string());
        assert!(format!("{err}").contains("x.js"));
        let err = HotReloadError::CircularDependency(vec!["a".into(), "b".into()]);
        assert!(format!("{err}").contains("a -> b"));
        let err = HotReloadError::ReloadInProgress;
        assert!(format!("{err}").contains("in progress"));
    }

    #[test]
    fn test_module_state_error_transitions() {
        assert!(ModuleState::Updating.can_transition_to(ModuleState::Errored));
        assert!(ModuleState::Errored.can_transition_to(ModuleState::Active));
        assert!(ModuleState::Errored.can_transition_to(ModuleState::Disposed));
    }

    #[test]
    fn test_disposed_to_active() {
        assert!(ModuleState::Disposed.can_transition_to(ModuleState::Active));
    }

    #[test]
    fn test_graph_default() {
        let g = ModuleGraph::default();
        assert_eq!(g.module_count(), 0);
    }

    #[test]
    fn test_module_version_starts_at_zero() {
        let m = ModuleInfo::new("test.js");
        assert_eq!(m.version, 0);
    }
}
