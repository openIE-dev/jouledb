//! Plugin system — plugin trait, registry, lifecycle, hooks, and dependency ordering.
//!
//! Replaces Webpack/Vite plugin systems with a pure-Rust plugin framework.
//! Supports plugin registration, load-order with dependency resolution,
//! lifecycle management (init/start/stop), hook points, plugin configuration,
//! version compatibility checking, and enable/disable.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Errors ──────────────────────────────────────────────────────

/// Plugin system domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginError {
    /// Plugin not found.
    NotFound(String),
    /// Plugin already registered.
    AlreadyRegistered(String),
    /// Dependency not satisfied.
    DependencyNotFound { plugin: String, dependency: String },
    /// Circular dependency.
    CircularDependency { chain: Vec<String> },
    /// Version incompatible.
    VersionIncompatible {
        plugin: String,
        required: String,
        actual: String,
    },
    /// Plugin lifecycle error.
    LifecycleError { plugin: String, reason: String },
    /// Plugin is disabled.
    Disabled(String),
    /// Hook not found.
    HookNotFound(String),
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "plugin not found: {name}"),
            Self::AlreadyRegistered(name) => write!(f, "plugin already registered: {name}"),
            Self::DependencyNotFound { plugin, dependency } => {
                write!(f, "plugin {plugin}: dependency {dependency} not found")
            }
            Self::CircularDependency { chain } => {
                write!(f, "circular dependency: {}", chain.join(" -> "))
            }
            Self::VersionIncompatible {
                plugin,
                required,
                actual,
            } => write!(
                f,
                "plugin {plugin}: requires {required}, got {actual}"
            ),
            Self::LifecycleError { plugin, reason } => {
                write!(f, "plugin {plugin} lifecycle error: {reason}")
            }
            Self::Disabled(name) => write!(f, "plugin {name} is disabled"),
            Self::HookNotFound(name) => write!(f, "hook not found: {name}"),
        }
    }
}

impl std::error::Error for PluginError {}

// ── Plugin State ────────────────────────────────────────────────

/// Lifecycle state of a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    Registered,
    Initialized,
    Started,
    Stopped,
    Failed,
}

// ── Version ─────────────────────────────────────────────────────

/// Semantic version with compatibility checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with a required version.
    /// Compatible if same major and >= minor.
    pub fn is_compatible_with(&self, required: &Version) -> bool {
        if self.major != required.major {
            return false;
        }
        if self.minor < required.minor {
            return false;
        }
        if self.minor == required.minor && self.patch < required.patch {
            return false;
        }
        true
    }

    pub fn to_string_repr(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ── Version Requirement ─────────────────────────────────────────

/// A dependency with an optional version requirement.
#[derive(Debug, Clone)]
pub struct PluginDependency {
    pub name: String,
    pub min_version: Option<Version>,
}

impl PluginDependency {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            min_version: None,
        }
    }

    pub fn with_min_version(mut self, version: Version) -> Self {
        self.min_version = Some(version);
        self
    }
}

// ── Hook ────────────────────────────────────────────────────────

/// A hook point that plugins can subscribe to.
#[derive(Debug, Clone)]
pub struct Hook {
    pub name: String,
    /// Plugin names subscribed to this hook, in order.
    pub subscribers: Vec<String>,
    /// How many times this hook has been called.
    pub call_count: u64,
}

impl Hook {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            subscribers: Vec::new(),
            call_count: 0,
        }
    }
}

// ── Plugin Registration ─────────────────────────────────────────

/// A plugin descriptor.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    pub name: String,
    pub version: Version,
    pub description: String,
    pub dependencies: Vec<PluginDependency>,
    pub state: PluginState,
    pub enabled: bool,
    pub config: HashMap<String, String>,
    /// Hooks this plugin subscribes to.
    pub subscribed_hooks: Vec<String>,
    /// Order in which the plugin was loaded.
    pub load_order: Option<usize>,
}

impl PluginDescriptor {
    pub fn new(
        name: impl Into<String>,
        version: Version,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version,
            description: description.into(),
            dependencies: Vec::new(),
            state: PluginState::Registered,
            enabled: true,
            config: HashMap::new(),
            subscribed_hooks: Vec::new(),
            load_order: None,
        }
    }

    /// Add a dependency.
    pub fn depends_on(mut self, dep: PluginDependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Set a configuration value.
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Subscribe to a hook.
    pub fn subscribe(mut self, hook: impl Into<String>) -> Self {
        self.subscribed_hooks.push(hook.into());
        self
    }
}

// ── Plugin Event ────────────────────────────────────────────────

/// Events emitted by the plugin system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginEvent {
    Registered(String),
    Initialized(String),
    Started(String),
    Stopped(String),
    Enabled(String),
    Disabled(String),
    HookCalled { hook: String, subscribers: Vec<String> },
    Failed { plugin: String, reason: String },
}

// ── Plugin Registry ─────────────────────────────────────────────

/// The plugin registry manages all plugins.
pub struct PluginRegistry {
    plugins: HashMap<String, PluginDescriptor>,
    hooks: HashMap<String, Hook>,
    events: Vec<PluginEvent>,
    load_order: Vec<String>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            hooks: HashMap::new(),
            events: Vec::new(),
            load_order: Vec::new(),
        }
    }

    /// Register a hook point.
    pub fn register_hook(&mut self, name: impl Into<String>) {
        let hook_name = name.into();
        self.hooks
            .entry(hook_name)
            .or_insert_with_key(|k| Hook::new(k.clone()));
    }

    /// Register a plugin.
    pub fn register(&mut self, plugin: PluginDescriptor) -> Result<(), PluginError> {
        if self.plugins.contains_key(&plugin.name) {
            return Err(PluginError::AlreadyRegistered(plugin.name));
        }
        let name = plugin.name.clone();
        // Register subscribed hooks
        let subscribed = plugin.subscribed_hooks.clone();
        self.plugins.insert(name.clone(), plugin);
        for hook_name in &subscribed {
            let hook = self
                .hooks
                .entry(hook_name.clone())
                .or_insert_with_key(|k| Hook::new(k.clone()));
            hook.subscribers.push(name.clone());
        }
        self.events.push(PluginEvent::Registered(name));
        Ok(())
    }

    /// Resolve load order using topological sort of dependencies.
    pub fn resolve_load_order(&mut self) -> Result<Vec<String>, PluginError> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize
        for name in self.plugins.keys() {
            in_degree.entry(name.clone()).or_insert(0);
            adjacency.entry(name.clone()).or_default();
        }

        // Build graph
        for (name, plugin) in &self.plugins {
            if !plugin.enabled {
                continue;
            }
            for dep in &plugin.dependencies {
                if !self.plugins.contains_key(&dep.name) {
                    return Err(PluginError::DependencyNotFound {
                        plugin: name.clone(),
                        dependency: dep.name.clone(),
                    });
                }
                // Check version compatibility
                if let Some(min_ver) = &dep.min_version {
                    let dep_plugin = &self.plugins[&dep.name];
                    if !dep_plugin.version.is_compatible_with(min_ver) {
                        return Err(PluginError::VersionIncompatible {
                            plugin: name.clone(),
                            required: min_ver.to_string_repr(),
                            actual: dep_plugin.version.to_string_repr(),
                        });
                    }
                }
                adjacency.entry(dep.name.clone()).or_default().push(name.clone());
                *in_degree.entry(name.clone()).or_insert(0) += 1;
            }
        }

        // Kahn's algorithm
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(name, deg)| {
                **deg == 0 && self.plugins.get(name.as_str()).map(|p| p.enabled).unwrap_or(false)
            })
            .map(|(name, _)| name.clone())
            .collect();

        // Sort queue for deterministic output
        let mut sorted_queue: Vec<String> = queue.drain(..).collect();
        sorted_queue.sort();
        for item in sorted_queue {
            queue.push_back(item);
        }

        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node.clone());
            if let Some(neighbors) = adjacency.get(&node) {
                let mut next_ready = Vec::new();
                for neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            next_ready.push(neighbor.clone());
                        }
                    }
                }
                // Sort for determinism
                next_ready.sort();
                for n in next_ready {
                    queue.push_back(n);
                }
            }
        }

        let enabled_count = self.plugins.values().filter(|p| p.enabled).count();
        if order.len() != enabled_count {
            // Cycle detected — find cycle
            let in_cycle: Vec<String> = in_degree
                .iter()
                .filter(|(_, deg)| **deg > 0)
                .map(|(name, _)| name.clone())
                .collect();
            return Err(PluginError::CircularDependency { chain: in_cycle });
        }

        // Store load order
        for (idx, name) in order.iter().enumerate() {
            if let Some(plugin) = self.plugins.get_mut(name) {
                plugin.load_order = Some(idx);
            }
        }
        self.load_order = order.clone();
        Ok(order)
    }

    /// Initialize all plugins in load order.
    pub fn init_all(&mut self) -> Result<(), PluginError> {
        if self.load_order.is_empty() {
            self.resolve_load_order()?;
        }
        let order = self.load_order.clone();
        for name in &order {
            if let Some(plugin) = self.plugins.get_mut(name) {
                if plugin.enabled && plugin.state == PluginState::Registered {
                    plugin.state = PluginState::Initialized;
                    self.events.push(PluginEvent::Initialized(name.clone()));
                }
            }
        }
        Ok(())
    }

    /// Start all initialized plugins.
    pub fn start_all(&mut self) -> Result<(), PluginError> {
        let order = self.load_order.clone();
        for name in &order {
            if let Some(plugin) = self.plugins.get_mut(name) {
                if plugin.enabled && plugin.state == PluginState::Initialized {
                    plugin.state = PluginState::Started;
                    self.events.push(PluginEvent::Started(name.clone()));
                }
            }
        }
        Ok(())
    }

    /// Stop all started plugins in reverse order.
    pub fn stop_all(&mut self) -> Result<(), PluginError> {
        let order: Vec<String> = self.load_order.iter().rev().cloned().collect();
        for name in &order {
            if let Some(plugin) = self.plugins.get_mut(name) {
                if plugin.state == PluginState::Started {
                    plugin.state = PluginState::Stopped;
                    self.events.push(PluginEvent::Stopped(name.clone()));
                }
            }
        }
        Ok(())
    }

    /// Initialize, start a single plugin.
    pub fn start_plugin(&mut self, name: &str) -> Result<(), PluginError> {
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        if !plugin.enabled {
            return Err(PluginError::Disabled(name.to_string()));
        }
        plugin.state = PluginState::Started;
        self.events.push(PluginEvent::Started(name.to_string()));
        Ok(())
    }

    /// Stop a single plugin.
    pub fn stop_plugin(&mut self, name: &str) -> Result<(), PluginError> {
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        if plugin.state != PluginState::Started {
            return Err(PluginError::LifecycleError {
                plugin: name.to_string(),
                reason: "not started".to_string(),
            });
        }
        plugin.state = PluginState::Stopped;
        self.events.push(PluginEvent::Stopped(name.to_string()));
        Ok(())
    }

    /// Enable a plugin.
    pub fn enable(&mut self, name: &str) -> Result<(), PluginError> {
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        plugin.enabled = true;
        self.events.push(PluginEvent::Enabled(name.to_string()));
        Ok(())
    }

    /// Disable a plugin.
    pub fn disable(&mut self, name: &str) -> Result<(), PluginError> {
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        plugin.enabled = false;
        // Stop if running
        if plugin.state == PluginState::Started {
            plugin.state = PluginState::Stopped;
            self.events.push(PluginEvent::Stopped(name.to_string()));
        }
        self.events.push(PluginEvent::Disabled(name.to_string()));
        Ok(())
    }

    /// Call a hook — invoke all subscribed plugins.
    pub fn call_hook(&mut self, hook_name: &str) -> Result<Vec<String>, PluginError> {
        let hook = self
            .hooks
            .get_mut(hook_name)
            .ok_or_else(|| PluginError::HookNotFound(hook_name.to_string()))?;
        hook.call_count += 1;
        let subscribers = hook.subscribers.clone();

        // Filter to only enabled, started plugins
        let active_subs: Vec<String> = subscribers
            .iter()
            .filter(|name| {
                self.plugins
                    .get(name.as_str())
                    .map(|p| p.enabled && p.state == PluginState::Started)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        self.events.push(PluginEvent::HookCalled {
            hook: hook_name.to_string(),
            subscribers: active_subs.clone(),
        });
        Ok(active_subs)
    }

    /// Set plugin config value.
    pub fn set_config(
        &mut self,
        plugin_name: &str,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<(), PluginError> {
        let plugin = self
            .plugins
            .get_mut(plugin_name)
            .ok_or_else(|| PluginError::NotFound(plugin_name.to_string()))?;
        plugin.config.insert(key.into(), value.into());
        Ok(())
    }

    /// Get plugin config value.
    pub fn get_config(&self, plugin_name: &str, key: &str) -> Option<String> {
        self.plugins
            .get(plugin_name)
            .and_then(|p| p.config.get(key).cloned())
    }

    /// Get a plugin reference.
    pub fn get_plugin(&self, name: &str) -> Option<&PluginDescriptor> {
        self.plugins.get(name)
    }

    /// Number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// Number of enabled plugins.
    pub fn enabled_count(&self) -> usize {
        self.plugins.values().filter(|p| p.enabled).count()
    }

    /// Number of started plugins.
    pub fn started_count(&self) -> usize {
        self.plugins
            .values()
            .filter(|p| p.state == PluginState::Started)
            .count()
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<PluginEvent> {
        std::mem::take(&mut self.events)
    }

    /// Get load order.
    pub fn load_order(&self) -> &[String] {
        &self.load_order
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> Version {
        Version::new(major, minor, patch)
    }

    #[test]
    fn test_register_plugin() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("auth", v(1, 0, 0), "Auth plugin"))
            .unwrap();
        assert_eq!(reg.plugin_count(), 1);
    }

    #[test]
    fn test_duplicate_registration() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("auth", v(1, 0, 0), "Auth"))
            .unwrap();
        let err = reg
            .register(PluginDescriptor::new("auth", v(2, 0, 0), "Auth v2"))
            .unwrap_err();
        assert_eq!(err, PluginError::AlreadyRegistered("auth".to_string()));
    }

    #[test]
    fn test_load_order_no_deps() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("b", v(1, 0, 0), "B"))
            .unwrap();
        reg.register(PluginDescriptor::new("a", v(1, 0, 0), "A"))
            .unwrap();
        let order = reg.resolve_load_order().unwrap();
        // Alphabetical when no deps
        assert_eq!(order, vec!["a", "b"]);
    }

    #[test]
    fn test_load_order_with_deps() {
        let mut reg = PluginRegistry::new();
        reg.register(
            PluginDescriptor::new("app", v(1, 0, 0), "App")
                .depends_on(PluginDependency::new("db")),
        )
        .unwrap();
        reg.register(PluginDescriptor::new("db", v(1, 0, 0), "Database"))
            .unwrap();
        let order = reg.resolve_load_order().unwrap();
        let db_pos = order.iter().position(|n| n == "db").unwrap();
        let app_pos = order.iter().position(|n| n == "app").unwrap();
        assert!(db_pos < app_pos);
    }

    #[test]
    fn test_circular_dependency() {
        let mut reg = PluginRegistry::new();
        reg.register(
            PluginDescriptor::new("a", v(1, 0, 0), "A")
                .depends_on(PluginDependency::new("b")),
        )
        .unwrap();
        reg.register(
            PluginDescriptor::new("b", v(1, 0, 0), "B")
                .depends_on(PluginDependency::new("a")),
        )
        .unwrap();
        let err = reg.resolve_load_order().unwrap_err();
        assert!(matches!(err, PluginError::CircularDependency { .. }));
    }

    #[test]
    fn test_missing_dependency() {
        let mut reg = PluginRegistry::new();
        reg.register(
            PluginDescriptor::new("app", v(1, 0, 0), "App")
                .depends_on(PluginDependency::new("missing")),
        )
        .unwrap();
        let err = reg.resolve_load_order().unwrap_err();
        assert!(matches!(err, PluginError::DependencyNotFound { .. }));
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = v(2, 3, 0);
        assert!(v1.is_compatible_with(&v(2, 1, 0)));
        assert!(v1.is_compatible_with(&v(2, 3, 0)));
        assert!(!v1.is_compatible_with(&v(2, 4, 0)));
        assert!(!v1.is_compatible_with(&v(3, 0, 0)));
    }

    #[test]
    fn test_version_incompatible_dependency() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("db", v(1, 0, 0), "DB"))
            .unwrap();
        reg.register(
            PluginDescriptor::new("app", v(1, 0, 0), "App")
                .depends_on(PluginDependency::new("db").with_min_version(v(2, 0, 0))),
        )
        .unwrap();
        let err = reg.resolve_load_order().unwrap_err();
        assert!(matches!(err, PluginError::VersionIncompatible { .. }));
    }

    #[test]
    fn test_init_and_start() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("p1", v(1, 0, 0), "P1"))
            .unwrap();
        reg.init_all().unwrap();
        assert_eq!(
            reg.get_plugin("p1").unwrap().state,
            PluginState::Initialized
        );
        reg.start_all().unwrap();
        assert_eq!(reg.get_plugin("p1").unwrap().state, PluginState::Started);
    }

    #[test]
    fn test_stop_all_reverse_order() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("a", v(1, 0, 0), "A"))
            .unwrap();
        reg.register(
            PluginDescriptor::new("b", v(1, 0, 0), "B")
                .depends_on(PluginDependency::new("a")),
        )
        .unwrap();
        reg.init_all().unwrap();
        reg.start_all().unwrap();
        reg.drain_events();
        reg.stop_all().unwrap();
        let events = reg.drain_events();
        // b should stop before a (reverse load order)
        let stop_events: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                PluginEvent::Stopped(name) => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(stop_events, vec!["b", "a"]);
    }

    #[test]
    fn test_enable_disable() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("p1", v(1, 0, 0), "P1"))
            .unwrap();
        reg.disable("p1").unwrap();
        assert!(!reg.get_plugin("p1").unwrap().enabled);
        assert_eq!(reg.enabled_count(), 0);
        reg.enable("p1").unwrap();
        assert!(reg.get_plugin("p1").unwrap().enabled);
    }

    #[test]
    fn test_disable_stops_running_plugin() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("p1", v(1, 0, 0), "P1"))
            .unwrap();
        reg.init_all().unwrap();
        reg.start_all().unwrap();
        reg.disable("p1").unwrap();
        assert_eq!(reg.get_plugin("p1").unwrap().state, PluginState::Stopped);
    }

    #[test]
    fn test_hook_registration_and_call() {
        let mut reg = PluginRegistry::new();
        reg.register_hook("on_request");
        reg.register(
            PluginDescriptor::new("auth", v(1, 0, 0), "Auth").subscribe("on_request"),
        )
        .unwrap();
        reg.init_all().unwrap();
        reg.start_all().unwrap();
        let subs = reg.call_hook("on_request").unwrap();
        assert_eq!(subs, vec!["auth"]);
    }

    #[test]
    fn test_hook_only_started_plugins() {
        let mut reg = PluginRegistry::new();
        reg.register_hook("on_request");
        reg.register(
            PluginDescriptor::new("p1", v(1, 0, 0), "P1").subscribe("on_request"),
        )
        .unwrap();
        // p1 is not started, hook should return empty
        let subs = reg.call_hook("on_request").unwrap();
        assert!(subs.is_empty());
    }

    #[test]
    fn test_hook_not_found() {
        let mut reg = PluginRegistry::new();
        let err = reg.call_hook("nonexistent").unwrap_err();
        assert_eq!(err, PluginError::HookNotFound("nonexistent".to_string()));
    }

    #[test]
    fn test_plugin_config() {
        let mut reg = PluginRegistry::new();
        reg.register(
            PluginDescriptor::new("db", v(1, 0, 0), "DB")
                .with_config("host", "localhost")
                .with_config("port", "5432"),
        )
        .unwrap();
        assert_eq!(reg.get_config("db", "host"), Some("localhost".to_string()));
        assert_eq!(reg.get_config("db", "port"), Some("5432".to_string()));
        reg.set_config("db", "port", "5433").unwrap();
        assert_eq!(reg.get_config("db", "port"), Some("5433".to_string()));
    }

    #[test]
    fn test_started_count() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("a", v(1, 0, 0), "A"))
            .unwrap();
        reg.register(PluginDescriptor::new("b", v(1, 0, 0), "B"))
            .unwrap();
        reg.init_all().unwrap();
        reg.start_all().unwrap();
        assert_eq!(reg.started_count(), 2);
    }

    #[test]
    fn test_disabled_plugin_excluded_from_load_order() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("a", v(1, 0, 0), "A"))
            .unwrap();
        reg.register(PluginDescriptor::new("b", v(1, 0, 0), "B"))
            .unwrap();
        reg.disable("b").unwrap();
        let order = reg.resolve_load_order().unwrap();
        assert_eq!(order, vec!["a"]);
    }

    #[test]
    fn test_start_disabled_plugin() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("p1", v(1, 0, 0), "P1"))
            .unwrap();
        reg.disable("p1").unwrap();
        let err = reg.start_plugin("p1").unwrap_err();
        assert_eq!(err, PluginError::Disabled("p1".to_string()));
    }

    #[test]
    fn test_stop_unstarted_plugin() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("p1", v(1, 0, 0), "P1"))
            .unwrap();
        let err = reg.stop_plugin("p1").unwrap_err();
        assert!(matches!(err, PluginError::LifecycleError { .. }));
    }

    #[test]
    fn test_events_emitted() {
        let mut reg = PluginRegistry::new();
        reg.register(PluginDescriptor::new("p1", v(1, 0, 0), "P1"))
            .unwrap();
        let events = reg.drain_events();
        assert!(events.contains(&PluginEvent::Registered("p1".to_string())));
    }

    #[test]
    fn test_version_to_string() {
        let version = v(1, 2, 3);
        assert_eq!(version.to_string_repr(), "1.2.3");
    }
}
