//! Module system/loader — module resolution (relative/absolute paths),
//! circular dependency detection, module cache, lazy loading, hot module
//! replacement simulation, dependency graph, module metadata.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Module Status ──────────────────────────────────────────────────────────

/// The loading state of a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleStatus {
    /// Not yet loaded.
    Unloaded,
    /// Currently being loaded (used for cycle detection).
    Loading,
    /// Fully loaded and initialized.
    Loaded,
    /// Load failed.
    Failed,
    /// Marked for lazy loading (will load on first access).
    Lazy,
    /// Module was hot-replaced.
    Replaced,
}

impl fmt::Display for ModuleStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Unloaded => "unloaded",
            Self::Loading => "loading",
            Self::Loaded => "loaded",
            Self::Failed => "failed",
            Self::Lazy => "lazy",
            Self::Replaced => "replaced",
        };
        write!(f, "{s}")
    }
}

// ── Module Metadata ────────────────────────────────────────────────────────

/// Metadata about a module.
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    /// Exported symbols.
    pub exports: Vec<String>,
    /// Size in bytes (of the module source / artifact).
    pub size_bytes: u64,
}

impl ModuleMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: String::new(),
            author: String::new(),
            description: String::new(),
            exports: Vec::new(),
            size_bytes: 0,
        }
    }

    pub fn with_version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }

    pub fn with_export(mut self, e: impl Into<String>) -> Self {
        self.exports.push(e.into());
        self
    }
}

// ── Module Entry ───────────────────────────────────────────────────────────

/// A registered module in the loader.
#[derive(Debug, Clone)]
pub struct ModuleEntry {
    pub path: String,
    pub status: ModuleStatus,
    pub metadata: ModuleMetadata,
    /// Paths this module depends on.
    pub dependencies: Vec<String>,
    /// Paths of modules that depend on this module.
    pub dependents: Vec<String>,
    /// Load order (sequence number).
    pub load_order: Option<u32>,
    /// Number of times this module has been loaded / reloaded.
    pub load_count: u32,
    /// Timestamp (simulated) of last load.
    pub last_loaded_ms: u64,
}

// ── Resolution Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    ModuleNotFound(String),
    CircularDependency(Vec<String>),
    LoadFailed { path: String, reason: String },
    AlreadyLoaded(String),
    InvalidPath(String),
    DependencyFailed { path: String, dependency: String },
    VersionConflict { path: String, existing: String, requested: String },
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModuleNotFound(p) => write!(f, "module not found: {p}"),
            Self::CircularDependency(cycle) => {
                write!(f, "circular dependency: {}", cycle.join(" -> "))
            }
            Self::LoadFailed { path, reason } => {
                write!(f, "failed to load {path}: {reason}")
            }
            Self::AlreadyLoaded(p) => write!(f, "already loaded: {p}"),
            Self::InvalidPath(p) => write!(f, "invalid path: {p}"),
            Self::DependencyFailed { path, dependency } => {
                write!(f, "dependency {dependency} failed for {path}")
            }
            Self::VersionConflict {
                path,
                existing,
                requested,
            } => write!(
                f,
                "version conflict for {path}: have {existing}, want {requested}"
            ),
        }
    }
}

// ── Path Resolution ────────────────────────────────────────────────────────

/// Resolve a module specifier relative to a base path.
///
/// - Absolute paths (starting with `/`) are returned as-is.
/// - Relative paths (starting with `./` or `../`) are resolved relative to the base.
/// - Bare specifiers (e.g., `lodash`) are resolved from a virtual node_modules.
pub fn resolve_path(base: &str, specifier: &str) -> Result<String, LoaderError> {
    if specifier.is_empty() {
        return Err(LoaderError::InvalidPath(specifier.to_string()));
    }

    if specifier.starts_with('/') {
        // Absolute.
        return Ok(normalize_path(specifier));
    }

    if specifier.starts_with("./") || specifier.starts_with("../") {
        // Relative to base.
        let base_dir = parent_dir(base);
        let combined = format!("{base_dir}/{specifier}");
        return Ok(normalize_path(&combined));
    }

    // Bare specifier → treat as absolute from root.
    Ok(format!("/node_modules/{specifier}"))
}

/// Get the parent directory of a path.
fn parent_dir(path: &str) -> String {
    if let Some(pos) = path.rfind('/') {
        path[..pos].to_string()
    } else {
        ".".to_string()
    }
}

/// Normalize a path: resolve `.` and `..` segments.
fn normalize_path(path: &str) -> String {
    let mut segments: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    format!("/{}", segments.join("/"))
}

// ── Module Loader ──────────────────────────────────────────────────────────

/// Module loader with caching, dependency tracking, and HMR.
pub struct ModuleLoader {
    modules: HashMap<String, ModuleEntry>,
    load_counter: u32,
    time_counter: u64,
    hmr_version: u64,
}

impl ModuleLoader {
    pub fn new() -> Self {
        Self {
            modules: HashMap::new(),
            load_counter: 0,
            time_counter: 0,
            hmr_version: 0,
        }
    }

    /// Number of registered modules.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Get a module entry.
    pub fn get_module(&self, path: &str) -> Option<&ModuleEntry> {
        self.modules.get(path)
    }

    /// Check if a module is loaded.
    pub fn is_loaded(&self, path: &str) -> bool {
        self.modules
            .get(path)
            .is_some_and(|m| m.status == ModuleStatus::Loaded)
    }

    /// Register a module without loading it.
    pub fn register(
        &mut self,
        path: impl Into<String>,
        metadata: ModuleMetadata,
        dependencies: Vec<String>,
    ) -> Result<(), LoaderError> {
        let path = path.into();
        if path.is_empty() {
            return Err(LoaderError::InvalidPath(path));
        }
        if self.modules.contains_key(&path) {
            return Err(LoaderError::AlreadyLoaded(path));
        }
        self.modules.insert(
            path.clone(),
            ModuleEntry {
                path,
                status: ModuleStatus::Unloaded,
                metadata,
                dependencies,
                dependents: Vec::new(),
                load_order: None,
                load_count: 0,
                last_loaded_ms: 0,
            },
        );
        Ok(())
    }

    /// Register a module for lazy loading.
    pub fn register_lazy(
        &mut self,
        path: impl Into<String>,
        metadata: ModuleMetadata,
        dependencies: Vec<String>,
    ) -> Result<(), LoaderError> {
        let path = path.into();
        if path.is_empty() {
            return Err(LoaderError::InvalidPath(path));
        }
        self.modules.insert(
            path.clone(),
            ModuleEntry {
                path,
                status: ModuleStatus::Lazy,
                metadata,
                dependencies,
                dependents: Vec::new(),
                load_order: None,
                load_count: 0,
                last_loaded_ms: 0,
            },
        );
        Ok(())
    }

    /// Load a module and all its dependencies (depth-first).
    /// Detects circular dependencies.
    pub fn load(&mut self, path: &str) -> Result<(), LoaderError> {
        if self.is_loaded(path) {
            return Ok(());
        }

        // Check for circular deps using DFS.
        let mut visiting = HashSet::new();
        let mut visit_stack = Vec::new();
        self.detect_cycle(path, &mut visiting, &mut visit_stack)?;

        // Topological load order.
        let order = self.topological_order(path)?;

        for p in &order {
            if self.is_loaded(p) {
                continue;
            }
            let entry = self
                .modules
                .get_mut(p)
                .ok_or_else(|| LoaderError::ModuleNotFound(p.clone()))?;
            entry.status = ModuleStatus::Loading;

            // Check that all dependencies are loaded.
            let deps = entry.dependencies.clone();
            for dep in &deps {
                if !self.is_loaded(dep) {
                    // The dep should have been loaded earlier in topo order,
                    // unless it wasn't registered.
                    if !self.modules.contains_key(dep) {
                        let failed_path = p.clone();
                        let entry = self.modules.get_mut(&failed_path).unwrap();
                        entry.status = ModuleStatus::Failed;
                        return Err(LoaderError::DependencyFailed {
                            path: failed_path,
                            dependency: dep.clone(),
                        });
                    }
                }
            }

            self.load_counter += 1;
            self.time_counter += 1;
            let order_num = self.load_counter;
            let time = self.time_counter;
            let entry = self.modules.get_mut(p).unwrap();
            entry.status = ModuleStatus::Loaded;
            entry.load_order = Some(order_num);
            entry.load_count += 1;
            entry.last_loaded_ms = time;
        }

        // Update dependent lists.
        self.rebuild_dependents();

        Ok(())
    }

    /// Detect circular dependencies.
    fn detect_cycle(
        &self,
        path: &str,
        visiting: &mut HashSet<String>,
        stack: &mut Vec<String>,
    ) -> Result<(), LoaderError> {
        if visiting.contains(path) {
            // Build cycle path.
            let start = stack.iter().position(|p| p == path).unwrap_or(0);
            let mut cycle: Vec<String> = stack[start..].to_vec();
            cycle.push(path.to_string());
            return Err(LoaderError::CircularDependency(cycle));
        }

        if self.is_loaded(path) {
            return Ok(());
        }

        visiting.insert(path.to_string());
        stack.push(path.to_string());

        if let Some(entry) = self.modules.get(path) {
            for dep in &entry.dependencies {
                self.detect_cycle(dep, visiting, stack)?;
            }
        }

        stack.pop();
        visiting.remove(path);
        Ok(())
    }

    /// Compute topological load order starting from `path`.
    fn topological_order(&self, path: &str) -> Result<Vec<String>, LoaderError> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        self.topo_visit(path, &mut visited, &mut order)?;
        Ok(order)
    }

    fn topo_visit(
        &self,
        path: &str,
        visited: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) -> Result<(), LoaderError> {
        if visited.contains(path) {
            return Ok(());
        }
        visited.insert(path.to_string());

        if let Some(entry) = self.modules.get(path) {
            for dep in &entry.dependencies {
                self.topo_visit(dep, visited, order)?;
            }
        }
        order.push(path.to_string());
        Ok(())
    }

    /// Rebuild the `dependents` lists from the `dependencies` lists.
    fn rebuild_dependents(&mut self) {
        // Collect dependency info first.
        let dep_pairs: Vec<(String, String)> = self
            .modules
            .values()
            .flat_map(|m| {
                m.dependencies
                    .iter()
                    .map(|d| (d.clone(), m.path.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

        // Clear all dependents.
        for m in self.modules.values_mut() {
            m.dependents.clear();
        }

        // Fill in.
        for (dep_path, dependent_path) in dep_pairs {
            if let Some(entry) = self.modules.get_mut(&dep_path) {
                if !entry.dependents.contains(&dependent_path) {
                    entry.dependents.push(dependent_path);
                }
            }
        }
    }

    // ── Hot Module Replacement ─────────────────────────────────────────

    /// Simulate hot module replacement for a module.
    /// Returns the set of modules that need to be re-evaluated.
    pub fn hot_replace(&mut self, path: &str) -> Result<Vec<String>, LoaderError> {
        if !self.modules.contains_key(path) {
            return Err(LoaderError::ModuleNotFound(path.to_string()));
        }

        self.hmr_version += 1;
        self.time_counter += 1;
        let time = self.time_counter;

        // Mark the module and propagate to dependents.
        let affected = self.affected_modules(path);

        for p in &affected {
            if let Some(entry) = self.modules.get_mut(p) {
                entry.status = ModuleStatus::Replaced;
                entry.load_count += 1;
                entry.last_loaded_ms = time;
            }
        }

        // Reload all affected modules.
        for p in &affected {
            if let Some(entry) = self.modules.get_mut(p) {
                entry.status = ModuleStatus::Loaded;
                self.load_counter += 1;
                entry.load_order = Some(self.load_counter);
            }
        }

        Ok(affected)
    }

    /// Get all modules affected by a change to `path` (BFS through dependents).
    fn affected_modules(&self, path: &str) -> Vec<String> {
        let mut affected = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(path.to_string());
        queue.push_back(path.to_string());

        while let Some(current) = queue.pop_front() {
            affected.push(current.clone());
            if let Some(entry) = self.modules.get(&current) {
                for dep in &entry.dependents {
                    if visited.insert(dep.clone()) {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }

        affected
    }

    // ── Dependency Graph ───────────────────────────────────────────────

    /// Get the full dependency graph as adjacency lists.
    pub fn dependency_graph(&self) -> HashMap<String, Vec<String>> {
        self.modules
            .iter()
            .map(|(path, entry)| (path.clone(), entry.dependencies.clone()))
            .collect()
    }

    /// Get all paths in load order.
    pub fn load_order(&self) -> Vec<String> {
        let mut entries: Vec<(&String, &ModuleEntry)> = self
            .modules
            .iter()
            .filter(|(_, e)| e.load_order.is_some())
            .collect();
        entries.sort_by_key(|(_, e)| e.load_order.unwrap());
        entries.iter().map(|(p, _)| (*p).clone()).collect()
    }

    /// Get modules with a specific status.
    pub fn modules_with_status(&self, status: ModuleStatus) -> Vec<String> {
        let mut paths: Vec<String> = self
            .modules
            .iter()
            .filter(|(_, e)| e.status == status)
            .map(|(p, _)| p.clone())
            .collect();
        paths.sort();
        paths
    }

    /// HMR version counter.
    pub fn hmr_version(&self) -> u64 {
        self.hmr_version
    }
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(name: &str) -> ModuleMetadata {
        ModuleMetadata::new(name)
    }

    #[test]
    fn resolve_absolute_path() {
        let resolved = resolve_path("/src/main.rs", "/lib/util.rs").unwrap();
        assert_eq!(resolved, "/lib/util.rs");
    }

    #[test]
    fn resolve_relative_path() {
        let resolved = resolve_path("/src/main.rs", "./util.rs").unwrap();
        assert_eq!(resolved, "/src/util.rs");
    }

    #[test]
    fn resolve_parent_relative() {
        let resolved = resolve_path("/src/a/b.rs", "../c.rs").unwrap();
        assert_eq!(resolved, "/src/c.rs");
    }

    #[test]
    fn resolve_bare_specifier() {
        let resolved = resolve_path("/src/main.rs", "lodash").unwrap();
        assert_eq!(resolved, "/node_modules/lodash");
    }

    #[test]
    fn resolve_empty_specifier() {
        let result = resolve_path("/src/main.rs", "");
        assert!(result.is_err());
    }

    #[test]
    fn register_and_load() {
        let mut loader = ModuleLoader::new();
        loader.register("/a", meta("a"), vec![]).unwrap();
        loader.load("/a").unwrap();
        assert!(loader.is_loaded("/a"));
    }

    #[test]
    fn load_with_dependencies() {
        let mut loader = ModuleLoader::new();
        loader
            .register("/a", meta("a"), vec!["/b".to_string()])
            .unwrap();
        loader.register("/b", meta("b"), vec![]).unwrap();
        loader.load("/a").unwrap();
        assert!(loader.is_loaded("/a"));
        assert!(loader.is_loaded("/b"));
    }

    #[test]
    fn circular_dependency_detected() {
        let mut loader = ModuleLoader::new();
        loader
            .register("/a", meta("a"), vec!["/b".to_string()])
            .unwrap();
        loader
            .register("/b", meta("b"), vec!["/a".to_string()])
            .unwrap();
        let result = loader.load("/a");
        assert!(matches!(result, Err(LoaderError::CircularDependency(_))));
    }

    #[test]
    fn transitive_circular_dep() {
        let mut loader = ModuleLoader::new();
        loader
            .register("/a", meta("a"), vec!["/b".to_string()])
            .unwrap();
        loader
            .register("/b", meta("b"), vec!["/c".to_string()])
            .unwrap();
        loader
            .register("/c", meta("c"), vec!["/a".to_string()])
            .unwrap();
        let result = loader.load("/a");
        assert!(matches!(result, Err(LoaderError::CircularDependency(_))));
    }

    #[test]
    fn module_not_found() {
        let mut loader = ModuleLoader::new();
        let result = loader.load("/nonexistent");
        assert!(matches!(result, Err(LoaderError::ModuleNotFound(_))));
    }

    #[test]
    fn duplicate_register_rejected() {
        let mut loader = ModuleLoader::new();
        loader.register("/a", meta("a"), vec![]).unwrap();
        let result = loader.register("/a", meta("a"), vec![]);
        assert!(matches!(result, Err(LoaderError::AlreadyLoaded(_))));
    }

    #[test]
    fn lazy_loading() {
        let mut loader = ModuleLoader::new();
        loader.register_lazy("/lazy", meta("lazy"), vec![]).unwrap();
        let entry = loader.get_module("/lazy").unwrap();
        assert_eq!(entry.status, ModuleStatus::Lazy);
        loader.load("/lazy").unwrap();
        assert!(loader.is_loaded("/lazy"));
    }

    #[test]
    fn load_order_tracking() {
        let mut loader = ModuleLoader::new();
        loader
            .register("/a", meta("a"), vec!["/b".to_string()])
            .unwrap();
        loader.register("/b", meta("b"), vec![]).unwrap();
        loader.load("/a").unwrap();
        let order = loader.load_order();
        // b should be loaded before a.
        let b_pos = order.iter().position(|p| p == "/b").unwrap();
        let a_pos = order.iter().position(|p| p == "/a").unwrap();
        assert!(b_pos < a_pos);
    }

    #[test]
    fn hot_module_replacement() {
        let mut loader = ModuleLoader::new();
        loader.register("/a", meta("a"), vec![]).unwrap();
        loader
            .register("/b", meta("b"), vec!["/a".to_string()])
            .unwrap();
        loader.load("/b").unwrap();

        let affected = loader.hot_replace("/a").unwrap();
        assert!(affected.contains(&"/a".to_string()));
        assert!(affected.contains(&"/b".to_string()));
        assert_eq!(loader.hmr_version(), 1);
    }

    #[test]
    fn hmr_on_nonexistent() {
        let mut loader = ModuleLoader::new();
        let result = loader.hot_replace("/nope");
        assert!(matches!(result, Err(LoaderError::ModuleNotFound(_))));
    }

    #[test]
    fn dependency_graph() {
        let mut loader = ModuleLoader::new();
        loader
            .register("/a", meta("a"), vec!["/b".to_string(), "/c".to_string()])
            .unwrap();
        loader.register("/b", meta("b"), vec![]).unwrap();
        loader.register("/c", meta("c"), vec![]).unwrap();
        let graph = loader.dependency_graph();
        assert_eq!(graph["/a"].len(), 2);
        assert!(graph["/b"].is_empty());
    }

    #[test]
    fn modules_with_status() {
        let mut loader = ModuleLoader::new();
        loader.register("/a", meta("a"), vec![]).unwrap();
        loader.register("/b", meta("b"), vec![]).unwrap();
        loader.load("/a").unwrap();
        let loaded = loader.modules_with_status(ModuleStatus::Loaded);
        assert_eq!(loaded, vec!["/a"]);
        let unloaded = loader.modules_with_status(ModuleStatus::Unloaded);
        assert_eq!(unloaded, vec!["/b"]);
    }

    #[test]
    fn metadata_builder() {
        let m = ModuleMetadata::new("test")
            .with_version("1.0.0")
            .with_export("default")
            .with_export("foo");
        assert_eq!(m.version, "1.0.0");
        assert_eq!(m.exports.len(), 2);
    }

    #[test]
    fn normalize_path_dots() {
        assert_eq!(normalize_path("/a/b/../c/./d"), "/a/c/d");
        assert_eq!(normalize_path("/a/b/c/../../d"), "/a/d");
    }

    #[test]
    fn load_count_increments() {
        let mut loader = ModuleLoader::new();
        loader.register("/a", meta("a"), vec![]).unwrap();
        loader.load("/a").unwrap();
        assert_eq!(loader.get_module("/a").unwrap().load_count, 1);
        loader.hot_replace("/a").unwrap();
        assert_eq!(loader.get_module("/a").unwrap().load_count, 2);
    }

    #[test]
    fn loader_error_display() {
        let e = LoaderError::CircularDependency(vec![
            "/a".to_string(),
            "/b".to_string(),
            "/a".to_string(),
        ]);
        assert!(e.to_string().contains("/a -> /b -> /a"));
    }

    #[test]
    fn idempotent_load() {
        let mut loader = ModuleLoader::new();
        loader.register("/a", meta("a"), vec![]).unwrap();
        loader.load("/a").unwrap();
        // Loading again should be a no-op.
        loader.load("/a").unwrap();
        assert_eq!(loader.get_module("/a").unwrap().load_count, 1);
    }
}
