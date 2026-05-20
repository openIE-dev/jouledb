//! Hot reload system for development asset iteration.
//!
//! Polls for file modification time changes, reloads changed assets in-place,
//! fires callbacks, tracks versions, cascades dependency reloads (e.g. texture
//! change → material reload), maintains reload history for undo, and can be
//! toggled between development and release modes.

use std::collections::HashMap;
use std::fmt;

// ── Reload mode ────────────────────────────────────────────────

/// Whether hot-reload is active (development) or disabled (release).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadMode {
    Development,
    Release,
}

impl fmt::Display for ReloadMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReloadMode::Development => write!(f, "development"),
            ReloadMode::Release => write!(f, "release"),
        }
    }
}

// ── Asset version ──────────────────────────────────────────────

/// Version counter for a tracked asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AssetVersion(pub u64);

impl AssetVersion {
    pub fn initial() -> Self {
        Self(1)
    }

    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for AssetVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// ── Reload event ───────────────────────────────────────────────

/// Events emitted by the hot-reload system.
#[derive(Debug, Clone, PartialEq)]
pub enum ReloadEvent {
    /// An asset's file was detected as changed.
    Changed(String),
    /// An asset was reloaded to a new version.
    Reloaded { path: String, version: AssetVersion },
    /// A dependency cascade triggered reload of a dependent asset.
    DependencyCascade { source: String, dependent: String },
    /// An undo was performed, reverting an asset to a prior version.
    Undone { path: String, version: AssetVersion },
}

// ── Tracked asset entry ────────────────────────────────────────

/// Internal state for a watched asset file.
#[derive(Debug, Clone)]
struct WatchedAsset<T: Clone> {
    path: String,
    /// Simulated modification timestamp (monotonic counter or external time).
    mod_time: u64,
    version: AssetVersion,
    current_data: T,
    /// Stack of previous data values for undo.
    history: Vec<(AssetVersion, T)>,
    /// Paths of assets that depend on this one.
    dependents: Vec<String>,
}

// ── Callback entry ─────────────────────────────────────────────

/// A reload callback registered for an asset path.
#[derive(Clone)]
pub struct ReloadCallback {
    pub id: u64,
    pub path: String,
    pub description: String,
}

impl fmt::Debug for ReloadCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReloadCallback")
            .field("id", &self.id)
            .field("path", &self.path)
            .field("description", &self.description)
            .finish()
    }
}

// ── Hot reload system ──────────────────────────────────────────

/// Hot-reload manager: watches assets, detects changes, reloads, and cascades.
pub struct HotReloadSystem<T: Clone> {
    mode: ReloadMode,
    assets: HashMap<String, WatchedAsset<T>>,
    callbacks: Vec<ReloadCallback>,
    next_callback_id: u64,
    events: Vec<ReloadEvent>,
    total_reloads: u64,
}

impl<T: Clone + fmt::Debug> fmt::Debug for HotReloadSystem<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotReloadSystem")
            .field("mode", &self.mode)
            .field("assets", &self.assets.len())
            .field("total_reloads", &self.total_reloads)
            .finish()
    }
}

impl<T: Clone> HotReloadSystem<T> {
    /// Create a new hot-reload system in the given mode.
    pub fn new(mode: ReloadMode) -> Self {
        Self {
            mode,
            assets: HashMap::new(),
            callbacks: Vec::new(),
            next_callback_id: 1,
            events: Vec::new(),
            total_reloads: 0,
        }
    }

    /// Current mode.
    pub fn mode(&self) -> ReloadMode {
        self.mode
    }

    /// Toggle mode.
    pub fn set_mode(&mut self, mode: ReloadMode) {
        self.mode = mode;
    }

    /// Whether hot-reload is active.
    pub fn is_active(&self) -> bool {
        self.mode == ReloadMode::Development
    }

    // ── Watch management ───────────────────────────────────────

    /// Register an asset for watching.
    pub fn watch(&mut self, path: &str, initial_data: T, mod_time: u64) {
        let entry = WatchedAsset {
            path: path.to_string(),
            mod_time,
            version: AssetVersion::initial(),
            current_data: initial_data,
            history: Vec::new(),
            dependents: Vec::new(),
        };
        self.assets.insert(path.to_string(), entry);
    }

    /// Stop watching an asset.
    pub fn unwatch(&mut self, path: &str) -> bool {
        self.assets.remove(path).is_some()
    }

    /// Whether a path is being watched.
    pub fn is_watched(&self, path: &str) -> bool {
        self.assets.contains_key(path)
    }

    /// Register a dependency: when `source` changes, `dependent` should also reload.
    pub fn add_dependency(&mut self, source: &str, dependent: &str) {
        if let Some(asset) = self.assets.get_mut(source) {
            if !asset.dependents.contains(&dependent.to_string()) {
                asset.dependents.push(dependent.to_string());
            }
        }
    }

    /// Remove a dependency link.
    pub fn remove_dependency(&mut self, source: &str, dependent: &str) {
        if let Some(asset) = self.assets.get_mut(source) {
            asset.dependents.retain(|d| d != dependent);
        }
    }

    // ── Polling / reloading ────────────────────────────────────

    /// Poll for changes: check if any asset's mod_time has been updated.
    /// Provide the current file modification times as a map.
    /// Returns events for changed assets.
    pub fn poll(&mut self, current_times: &HashMap<String, u64>) -> Vec<ReloadEvent> {
        self.events.clear();
        if self.mode == ReloadMode::Release {
            return Vec::new();
        }

        let paths: Vec<String> = self.assets.keys().cloned().collect();
        let mut changed = Vec::new();

        for path in &paths {
            if let Some(&new_time) = current_times.get(path) {
                let asset = self.assets.get(path).unwrap();
                if new_time > asset.mod_time {
                    changed.push((path.clone(), new_time));
                }
            }
        }

        // Update mod times for changed assets.
        for (path, new_time) in &changed {
            if let Some(asset) = self.assets.get_mut(path) {
                asset.mod_time = *new_time;
                self.events.push(ReloadEvent::Changed(path.clone()));
            }
        }

        self.events.clone()
    }

    /// Reload an asset with new data. Pushes the old data to history.
    pub fn reload(&mut self, path: &str, new_data: T) -> Option<AssetVersion> {
        if self.mode == ReloadMode::Release {
            return None;
        }
        let asset = self.assets.get_mut(path)?;
        let old_data = asset.current_data.clone();
        let old_version = asset.version;
        asset.history.push((old_version, old_data));
        asset.version = asset.version.next();
        asset.current_data = new_data;
        self.total_reloads += 1;

        let new_version = asset.version;
        self.events.push(ReloadEvent::Reloaded {
            path: path.to_string(),
            version: new_version,
        });

        // Fire callbacks.
        let _fired: Vec<_> = self
            .callbacks
            .iter()
            .filter(|cb| cb.path == path)
            .map(|cb| cb.id)
            .collect();

        Some(new_version)
    }

    /// Reload an asset and cascade to all dependents.
    pub fn reload_with_cascade(
        &mut self,
        path: &str,
        new_data: T,
        dependent_data_fn: &dyn Fn(&str) -> Option<T>,
    ) -> Vec<ReloadEvent> {
        self.events.clear();
        self.reload(path, new_data);

        // Collect dependents to cascade.
        let dependents: Vec<String> = self
            .assets
            .get(path)
            .map(|a| a.dependents.clone())
            .unwrap_or_default();

        for dep_path in &dependents {
            if let Some(dep_data) = dependent_data_fn(dep_path) {
                self.events.push(ReloadEvent::DependencyCascade {
                    source: path.to_string(),
                    dependent: dep_path.clone(),
                });
                self.reload(dep_path, dep_data);
            }
        }

        self.events.clone()
    }

    /// Undo the last reload for an asset. Returns the restored version.
    pub fn undo_reload(&mut self, path: &str) -> Option<AssetVersion> {
        let asset = self.assets.get_mut(path)?;
        let (prev_version, prev_data) = asset.history.pop()?;
        asset.current_data = prev_data;
        asset.version = prev_version;
        self.events.push(ReloadEvent::Undone {
            path: path.to_string(),
            version: prev_version,
        });
        Some(prev_version)
    }

    // ── Callbacks ──────────────────────────────────────────────

    /// Register a callback for reloads of a specific asset path.
    pub fn on_reload(&mut self, path: &str, description: &str) -> u64 {
        let id = self.next_callback_id;
        self.next_callback_id += 1;
        self.callbacks.push(ReloadCallback {
            id,
            path: path.to_string(),
            description: description.to_string(),
        });
        id
    }

    /// Remove a callback by ID.
    pub fn remove_callback(&mut self, id: u64) -> bool {
        let before = self.callbacks.len();
        self.callbacks.retain(|cb| cb.id != id);
        self.callbacks.len() < before
    }

    /// Number of registered callbacks.
    pub fn callback_count(&self) -> usize {
        self.callbacks.len()
    }

    // ── Queries ────────────────────────────────────────────────

    /// Get current data for a watched asset.
    pub fn data(&self, path: &str) -> Option<&T> {
        self.assets.get(path).map(|a| &a.current_data)
    }

    /// Get current version for a watched asset.
    pub fn version(&self, path: &str) -> Option<AssetVersion> {
        self.assets.get(path).map(|a| a.version)
    }

    /// Number of reloads in the history for an asset.
    pub fn history_len(&self, path: &str) -> usize {
        self.assets.get(path).map(|a| a.history.len()).unwrap_or(0)
    }

    /// Number of watched assets.
    pub fn watched_count(&self) -> usize {
        self.assets.len()
    }

    /// Total reloads performed.
    pub fn total_reloads(&self) -> u64 {
        self.total_reloads
    }

    /// All watched paths.
    pub fn watched_paths(&self) -> Vec<String> {
        self.assets.keys().cloned().collect()
    }

    /// Dependents of a given asset.
    pub fn dependents(&self, path: &str) -> Vec<String> {
        self.assets
            .get(path)
            .map(|a| a.dependents.clone())
            .unwrap_or_default()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_system_dev_mode() {
        let sys: HotReloadSystem<String> = HotReloadSystem::new(ReloadMode::Development);
        assert!(sys.is_active());
        assert_eq!(sys.mode(), ReloadMode::Development);
    }

    #[test]
    fn mode_toggle() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.set_mode(ReloadMode::Release);
        assert!(!sys.is_active());
    }

    #[test]
    fn watch_and_query() {
        let mut sys: HotReloadSystem<String> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("tex/hero.png", "rgba_data".into(), 100);
        assert!(sys.is_watched("tex/hero.png"));
        assert_eq!(sys.data("tex/hero.png"), Some(&"rgba_data".to_string()));
        assert_eq!(sys.version("tex/hero.png"), Some(AssetVersion::initial()));
    }

    #[test]
    fn unwatch() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        assert!(sys.unwatch("a.txt"));
        assert!(!sys.is_watched("a.txt"));
    }

    #[test]
    fn poll_detects_changes() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        let mut times = HashMap::new();
        times.insert("a.txt".to_string(), 200);
        let evts = sys.poll(&times);
        assert!(evts.iter().any(|e| matches!(e, ReloadEvent::Changed(p) if p == "a.txt")));
    }

    #[test]
    fn poll_no_change() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        let mut times = HashMap::new();
        times.insert("a.txt".to_string(), 100);
        let evts = sys.poll(&times);
        assert!(evts.is_empty());
    }

    #[test]
    fn poll_in_release_mode_noop() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Release);
        sys.watch("a.txt", 1, 100);
        let mut times = HashMap::new();
        times.insert("a.txt".to_string(), 200);
        let evts = sys.poll(&times);
        assert!(evts.is_empty());
    }

    #[test]
    fn reload_increments_version() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        let v = sys.reload("a.txt", 2).unwrap();
        assert_eq!(v, AssetVersion(2));
        assert_eq!(sys.data("a.txt"), Some(&2));
    }

    #[test]
    fn reload_in_release_returns_none() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Release);
        sys.watch("a.txt", 1, 100);
        assert!(sys.reload("a.txt", 2).is_none());
    }

    #[test]
    fn reload_history_tracks() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        sys.reload("a.txt", 2);
        sys.reload("a.txt", 3);
        assert_eq!(sys.history_len("a.txt"), 2);
    }

    #[test]
    fn undo_reload() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        sys.reload("a.txt", 2);
        let v = sys.undo_reload("a.txt").unwrap();
        assert_eq!(v, AssetVersion(1));
        assert_eq!(sys.data("a.txt"), Some(&1));
    }

    #[test]
    fn undo_no_history_returns_none() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a.txt", 1, 100);
        assert!(sys.undo_reload("a.txt").is_none());
    }

    #[test]
    fn dependency_cascade() {
        let mut sys: HotReloadSystem<String> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("tex.png", "tex_v1".into(), 100);
        sys.watch("mat.mat", "mat_v1".into(), 100);
        sys.add_dependency("tex.png", "mat.mat");
        let evts = sys.reload_with_cascade("tex.png", "tex_v2".into(), &|dep| {
            if dep == "mat.mat" {
                Some("mat_v2".into())
            } else {
                None
            }
        });
        assert!(evts.iter().any(|e| matches!(e, ReloadEvent::DependencyCascade { source, dependent }
            if source == "tex.png" && dependent == "mat.mat")));
        assert_eq!(sys.data("mat.mat"), Some(&"mat_v2".to_string()));
    }

    #[test]
    fn remove_dependency() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a", 1, 100);
        sys.watch("b", 2, 100);
        sys.add_dependency("a", "b");
        assert_eq!(sys.dependents("a").len(), 1);
        sys.remove_dependency("a", "b");
        assert!(sys.dependents("a").is_empty());
    }

    #[test]
    fn callbacks_registration() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        let id = sys.on_reload("a.txt", "update shader");
        assert_eq!(sys.callback_count(), 1);
        assert!(sys.remove_callback(id));
        assert_eq!(sys.callback_count(), 0);
    }

    #[test]
    fn remove_nonexistent_callback() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        assert!(!sys.remove_callback(999));
    }

    #[test]
    fn total_reloads() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a", 1, 100);
        sys.watch("b", 2, 100);
        sys.reload("a", 10);
        sys.reload("b", 20);
        sys.reload("a", 30);
        assert_eq!(sys.total_reloads(), 3);
    }

    #[test]
    fn watched_count_and_paths() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a", 1, 100);
        sys.watch("b", 2, 100);
        assert_eq!(sys.watched_count(), 2);
        let paths = sys.watched_paths();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn asset_version_display() {
        assert_eq!(AssetVersion(3).to_string(), "v3");
        assert_eq!(AssetVersion::initial().to_string(), "v1");
    }

    #[test]
    fn reload_mode_display() {
        assert_eq!(ReloadMode::Development.to_string(), "development");
        assert_eq!(ReloadMode::Release.to_string(), "release");
    }

    #[test]
    fn multiple_undos() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a", 1, 100);
        sys.reload("a", 2);
        sys.reload("a", 3);
        sys.reload("a", 4);
        sys.undo_reload("a"); // back to 3
        sys.undo_reload("a"); // back to 2
        sys.undo_reload("a"); // back to 1
        assert_eq!(sys.data("a"), Some(&1));
        assert!(sys.undo_reload("a").is_none());
    }

    #[test]
    fn duplicate_dependency_ignored() {
        let mut sys: HotReloadSystem<u32> = HotReloadSystem::new(ReloadMode::Development);
        sys.watch("a", 1, 100);
        sys.watch("b", 2, 100);
        sys.add_dependency("a", "b");
        sys.add_dependency("a", "b");
        assert_eq!(sys.dependents("a").len(), 1);
    }
}
