//! Test fixture management.
//!
//! Replaces `pytest` fixtures, `jest` beforeEach/afterEach, and similar
//! test lifecycle patterns with a pure-Rust fixture system. Supports
//! setup/teardown lifecycle, shared fixtures (singleton), fixture composition,
//! lazy initialization, fixture parameters, cleanup ordering, and fixture factories.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Fixture errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureError {
    /// Fixture not found by name.
    NotFound(String),
    /// Fixture already exists.
    AlreadyExists(String),
    /// Setup failed.
    SetupFailed(String),
    /// Teardown failed.
    TeardownFailed(String),
    /// Dependency cycle detected.
    CyclicDependency(String),
}

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "fixture not found: {name}"),
            Self::AlreadyExists(name) => write!(f, "fixture already exists: {name}"),
            Self::SetupFailed(msg) => write!(f, "setup failed: {msg}"),
            Self::TeardownFailed(msg) => write!(f, "teardown failed: {msg}"),
            Self::CyclicDependency(msg) => write!(f, "cyclic dependency: {msg}"),
        }
    }
}

impl std::error::Error for FixtureError {}

// ── Fixture Scope ────────────────────────────────────────────────

/// Scope controls fixture lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureScope {
    /// Fresh instance per test.
    Test,
    /// Shared across all tests (singleton).
    Session,
    /// Shared within a module/group.
    Module,
}

// ── Fixture State ────────────────────────────────────────────────

/// Current state of a fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureState {
    /// Not yet initialized.
    Pending,
    /// Setup is running.
    Initializing,
    /// Active and usable.
    Active,
    /// Teardown is running.
    TearingDown,
    /// Fully cleaned up.
    Destroyed,
}

// ── Fixture Value ────────────────────────────────────────────────

/// A fixture's stored value and metadata.
#[derive(Debug, Clone)]
pub struct FixtureValue {
    /// The value stored by this fixture.
    pub data: String,
    /// Parameters passed to this fixture at creation.
    pub params: HashMap<String, String>,
    /// Fixture scope.
    pub scope: FixtureScope,
    /// Current state.
    pub state: FixtureState,
    /// Setup counter (how many times setup was called).
    pub setup_count: usize,
    /// Teardown counter.
    pub teardown_count: usize,
}

impl FixtureValue {
    fn new(data: &str, scope: FixtureScope) -> Self {
        Self {
            data: data.to_string(),
            params: HashMap::new(),
            scope,
            state: FixtureState::Active,
            setup_count: 1,
            teardown_count: 0,
        }
    }

    fn with_params(mut self, params: HashMap<String, String>) -> Self {
        self.params = params;
        self
    }
}

// ── Fixture Definition ───────────────────────────────────────────

/// Definition of a fixture: how to set up and tear down.
#[derive(Debug, Clone)]
pub struct FixtureDef {
    /// Fixture name.
    pub name: String,
    /// Scope.
    pub scope: FixtureScope,
    /// Dependencies (names of other fixtures).
    pub dependencies: Vec<String>,
    /// Default parameters.
    pub default_params: HashMap<String, String>,
    /// Setup produces this default value.
    pub default_value: String,
    /// Priority for teardown ordering (lower = torn down later).
    pub teardown_priority: i32,
}

impl FixtureDef {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            scope: FixtureScope::Test,
            dependencies: Vec::new(),
            default_params: HashMap::new(),
            default_value: String::new(),
            teardown_priority: 0,
        }
    }

    pub fn with_scope(mut self, scope: FixtureScope) -> Self {
        self.scope = scope;
        self
    }

    pub fn with_dependency(mut self, dep: &str) -> Self {
        self.dependencies.push(dep.to_string());
        self
    }

    pub fn with_default_param(mut self, key: &str, value: &str) -> Self {
        self.default_params.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_default_value(mut self, value: &str) -> Self {
        self.default_value = value.to_string();
        self
    }

    pub fn with_teardown_priority(mut self, priority: i32) -> Self {
        self.teardown_priority = priority;
        self
    }
}

// ── Fixture Manager ──────────────────────────────────────────────

/// Manages fixture lifecycle: registration, setup, teardown, caching.
#[derive(Debug, Clone)]
pub struct FixtureManager {
    /// Registered definitions.
    definitions: HashMap<String, FixtureDef>,
    /// Active fixture values.
    instances: HashMap<String, FixtureValue>,
    /// Setup log for ordering verification.
    setup_log: Vec<String>,
    /// Teardown log.
    teardown_log: Vec<String>,
}

impl Default for FixtureManager {
    fn default() -> Self {
        Self::new()
    }
}

impl FixtureManager {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            instances: HashMap::new(),
            setup_log: Vec::new(),
            teardown_log: Vec::new(),
        }
    }

    /// Register a fixture definition.
    pub fn register(&mut self, def: FixtureDef) -> Result<(), FixtureError> {
        if self.definitions.contains_key(&def.name) {
            return Err(FixtureError::AlreadyExists(def.name.clone()));
        }
        self.definitions.insert(def.name.clone(), def);
        Ok(())
    }

    /// Check if a fixture is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.definitions.contains_key(name)
    }

    /// Check if a fixture instance is active.
    pub fn is_active(&self, name: &str) -> bool {
        self.instances
            .get(name)
            .is_some_and(|v| v.state == FixtureState::Active)
    }

    /// Set up a fixture, resolving dependencies first.
    pub fn setup(&mut self, name: &str) -> Result<(), FixtureError> {
        self.setup_with_params(name, HashMap::new())
    }

    /// Set up a fixture with custom parameters.
    pub fn setup_with_params(
        &mut self,
        name: &str,
        params: HashMap<String, String>,
    ) -> Result<(), FixtureError> {
        // Check for existing session-scoped instance
        if let Some(existing) = self.instances.get(name) {
            if existing.scope == FixtureScope::Session && existing.state == FixtureState::Active {
                return Ok(()); // Reuse singleton
            }
        }

        let def = self
            .definitions
            .get(name)
            .ok_or_else(|| FixtureError::NotFound(name.to_string()))?
            .clone();

        // Detect cycles with a simple visited set
        self.check_cycles(name, &mut Vec::new())?;

        // Set up dependencies first
        for dep in &def.dependencies {
            if !self.is_active(dep) {
                self.setup(dep)?;
            }
        }

        // Merge default params with provided params
        let mut merged_params = def.default_params.clone();
        for (k, v) in params {
            merged_params.insert(k, v);
        }

        // Create the instance
        let value = FixtureValue::new(&def.default_value, def.scope).with_params(merged_params);
        self.instances.insert(name.to_string(), value);
        self.setup_log.push(name.to_string());

        Ok(())
    }

    /// Check for dependency cycles.
    fn check_cycles(&self, name: &str, visited: &mut Vec<String>) -> Result<(), FixtureError> {
        if visited.contains(&name.to_string()) {
            return Err(FixtureError::CyclicDependency(format!(
                "{} -> {}",
                visited.join(" -> "),
                name
            )));
        }
        visited.push(name.to_string());

        if let Some(def) = self.definitions.get(name) {
            for dep in &def.dependencies {
                self.check_cycles(dep, visited)?;
            }
        }

        visited.pop();
        Ok(())
    }

    /// Get a fixture's current value.
    pub fn get(&self, name: &str) -> Result<&FixtureValue, FixtureError> {
        self.instances
            .get(name)
            .ok_or_else(|| FixtureError::NotFound(name.to_string()))
    }

    /// Get a fixture's data string.
    pub fn get_data(&self, name: &str) -> Result<&str, FixtureError> {
        self.get(name).map(|v| v.data.as_str())
    }

    /// Modify a fixture's data.
    pub fn set_data(&mut self, name: &str, data: &str) -> Result<(), FixtureError> {
        let instance = self
            .instances
            .get_mut(name)
            .ok_or_else(|| FixtureError::NotFound(name.to_string()))?;
        instance.data = data.to_string();
        Ok(())
    }

    /// Tear down a single fixture.
    pub fn teardown(&mut self, name: &str) -> Result<(), FixtureError> {
        let instance = self
            .instances
            .get_mut(name)
            .ok_or_else(|| FixtureError::NotFound(name.to_string()))?;
        instance.state = FixtureState::TearingDown;
        instance.teardown_count += 1;
        instance.state = FixtureState::Destroyed;
        self.teardown_log.push(name.to_string());
        Ok(())
    }

    /// Tear down all active fixtures in priority order (higher priority first).
    pub fn teardown_all(&mut self) -> Result<(), FixtureError> {
        // Collect active fixture names with their priorities
        let mut active: Vec<(String, i32)> = Vec::new();
        for (name, instance) in &self.instances {
            if instance.state == FixtureState::Active {
                let priority = self
                    .definitions
                    .get(name)
                    .map(|d| d.teardown_priority)
                    .unwrap_or(0);
                active.push((name.clone(), priority));
            }
        }

        // Sort by priority descending (higher priority torn down first)
        active.sort_by(|a, b| b.1.cmp(&a.1));

        for (name, _) in active {
            self.teardown(&name)?;
        }
        Ok(())
    }

    /// Remove all instances (without running teardown).
    pub fn clear_instances(&mut self) {
        self.instances.clear();
    }

    /// Remove all definitions and instances.
    pub fn clear(&mut self) {
        self.definitions.clear();
        self.instances.clear();
        self.setup_log.clear();
        self.teardown_log.clear();
    }

    /// Number of registered definitions.
    pub fn num_definitions(&self) -> usize {
        self.definitions.len()
    }

    /// Number of active instances.
    pub fn num_active(&self) -> usize {
        self.instances
            .values()
            .filter(|v| v.state == FixtureState::Active)
            .count()
    }

    /// Setup log (in order).
    pub fn setup_log(&self) -> &[String] {
        &self.setup_log
    }

    /// Teardown log (in order).
    pub fn teardown_log(&self) -> &[String] {
        &self.teardown_log
    }
}

// ── Fixture Factory ──────────────────────────────────────────────

/// Factory for creating fixtures with shared templates.
#[derive(Debug, Clone)]
pub struct FixtureFactory {
    /// Base parameters applied to all fixtures from this factory.
    base_params: HashMap<String, String>,
    /// Counter for unique naming.
    counter: usize,
    /// Scope for all fixtures from this factory.
    scope: FixtureScope,
}

impl Default for FixtureFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl FixtureFactory {
    pub fn new() -> Self {
        Self {
            base_params: HashMap::new(),
            counter: 0,
            scope: FixtureScope::Test,
        }
    }

    pub fn with_scope(mut self, scope: FixtureScope) -> Self {
        self.scope = scope;
        self
    }

    pub fn with_base_param(mut self, key: &str, value: &str) -> Self {
        self.base_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Create a fixture definition from this factory.
    pub fn create(&mut self, name: &str, value: &str) -> FixtureDef {
        self.counter += 1;
        let mut def = FixtureDef::new(name)
            .with_scope(self.scope)
            .with_default_value(value);
        for (k, v) in &self.base_params {
            def = def.with_default_param(k, v);
        }
        def
    }

    /// Create a uniquely-named fixture.
    pub fn create_unique(&mut self, prefix: &str, value: &str) -> FixtureDef {
        let name = format!("{prefix}_{}", self.counter);
        self.create(&name, value)
    }

    /// Number of fixtures created by this factory.
    pub fn created_count(&self) -> usize {
        self.counter
    }
}

// ── Lazy Fixture ─────────────────────────────────────────────────

/// A lazily-initialized fixture that defers setup until first access.
#[derive(Debug, Clone)]
pub struct LazyFixture {
    name: String,
    value: Option<String>,
    default_value: String,
    initialized: bool,
    access_count: usize,
}

impl LazyFixture {
    pub fn new(name: &str, default_value: &str) -> Self {
        Self {
            name: name.to_string(),
            value: None,
            default_value: default_value.to_string(),
            initialized: false,
            access_count: 0,
        }
    }

    /// Access the fixture value, initializing on first access.
    pub fn get(&mut self) -> &str {
        self.access_count += 1;
        if !self.initialized {
            self.value = Some(self.default_value.clone());
            self.initialized = true;
        }
        self.value.as_deref().unwrap_or("")
    }

    /// Check if the fixture has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Number of times the fixture was accessed.
    pub fn access_count(&self) -> usize {
        self.access_count
    }

    /// Name of the fixture.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset to uninitialized state.
    pub fn reset(&mut self) {
        self.value = None;
        self.initialized = false;
        self.access_count = 0;
    }
}

// ── Fixture Composition ──────────────────────────────────────────

/// Compose multiple fixture values into a single structured fixture.
#[derive(Debug, Clone, Default)]
pub struct CompositeFixture {
    parts: HashMap<String, String>,
}

impl CompositeFixture {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named part.
    pub fn add(&mut self, name: &str, value: &str) {
        self.parts.insert(name.to_string(), value.to_string());
    }

    /// Get a part by name.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.parts.get(name).map(|s| s.as_str())
    }

    /// Number of parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Merge another composite into this one.
    pub fn merge(&mut self, other: &CompositeFixture) {
        for (k, v) in &other.parts {
            self.parts.insert(k.clone(), v.clone());
        }
    }

    /// All part names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.parts.keys().cloned().collect();
        names.sort();
        names
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_setup() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("db").with_default_value("test.db"))
            .unwrap();
        mgr.setup("db").unwrap();
        assert!(mgr.is_active("db"));
        assert_eq!(mgr.get_data("db").unwrap(), "test.db");
    }

    #[test]
    fn setup_not_found() {
        let mut mgr = FixtureManager::new();
        let err = mgr.setup("missing").unwrap_err();
        assert!(matches!(err, FixtureError::NotFound(_)));
    }

    #[test]
    fn duplicate_registration_fails() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("x")).unwrap();
        let err = mgr.register(FixtureDef::new("x")).unwrap_err();
        assert!(matches!(err, FixtureError::AlreadyExists(_)));
    }

    #[test]
    fn teardown_changes_state() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("svc").with_default_value("running"))
            .unwrap();
        mgr.setup("svc").unwrap();
        assert!(mgr.is_active("svc"));
        mgr.teardown("svc").unwrap();
        assert!(!mgr.is_active("svc"));
    }

    #[test]
    fn dependency_setup_order() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("db").with_default_value("db"))
            .unwrap();
        mgr.register(
            FixtureDef::new("repo")
                .with_default_value("repo")
                .with_dependency("db"),
        )
        .unwrap();
        mgr.setup("repo").unwrap();

        // Both should be active, and db was set up before repo
        assert!(mgr.is_active("db"));
        assert!(mgr.is_active("repo"));
        let log = mgr.setup_log();
        let db_pos = log.iter().position(|s| s == "db").unwrap();
        let repo_pos = log.iter().position(|s| s == "repo").unwrap();
        assert!(db_pos < repo_pos);
    }

    #[test]
    fn cyclic_dependency_detected() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("a").with_dependency("b"))
            .unwrap();
        mgr.register(FixtureDef::new("b").with_dependency("a"))
            .unwrap();
        let err = mgr.setup("a").unwrap_err();
        assert!(matches!(err, FixtureError::CyclicDependency(_)));
    }

    #[test]
    fn session_scope_reuses_instance() {
        let mut mgr = FixtureManager::new();
        mgr.register(
            FixtureDef::new("config")
                .with_scope(FixtureScope::Session)
                .with_default_value("v1"),
        )
        .unwrap();
        mgr.setup("config").unwrap();
        // Modify the data
        mgr.set_data("config", "modified").unwrap();
        // Second setup should reuse
        mgr.setup("config").unwrap();
        assert_eq!(mgr.get_data("config").unwrap(), "modified");
    }

    #[test]
    fn setup_with_params() {
        let mut mgr = FixtureManager::new();
        mgr.register(
            FixtureDef::new("db")
                .with_default_param("host", "localhost")
                .with_default_value("connected"),
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("host".to_string(), "remote.host".to_string());
        mgr.setup_with_params("db", params).unwrap();

        let fixture = mgr.get("db").unwrap();
        assert_eq!(fixture.params.get("host").unwrap(), "remote.host");
    }

    #[test]
    fn teardown_all_priority_order() {
        let mut mgr = FixtureManager::new();
        mgr.register(
            FixtureDef::new("low")
                .with_default_value("low")
                .with_teardown_priority(1),
        )
        .unwrap();
        mgr.register(
            FixtureDef::new("high")
                .with_default_value("high")
                .with_teardown_priority(10),
        )
        .unwrap();

        mgr.setup("low").unwrap();
        mgr.setup("high").unwrap();
        mgr.teardown_all().unwrap();

        let log = mgr.teardown_log();
        let high_pos = log.iter().position(|s| s == "high").unwrap();
        let low_pos = log.iter().position(|s| s == "low").unwrap();
        assert!(high_pos < low_pos);
    }

    #[test]
    fn fixture_factory_creates_defs() {
        let mut factory = FixtureFactory::new()
            .with_scope(FixtureScope::Test)
            .with_base_param("env", "test");

        let def = factory.create("db", "test.db");
        assert_eq!(def.name, "db");
        assert_eq!(def.scope, FixtureScope::Test);
        assert_eq!(
            def.default_params.get("env").unwrap(),
            "test"
        );
        assert_eq!(factory.created_count(), 1);
    }

    #[test]
    fn fixture_factory_unique_names() {
        let mut factory = FixtureFactory::new();
        let d1 = factory.create_unique("item", "v1");
        let d2 = factory.create_unique("item", "v2");
        assert_ne!(d1.name, d2.name);
    }

    #[test]
    fn lazy_fixture_defers_init() {
        let mut lazy = LazyFixture::new("config", "default_value");
        assert!(!lazy.is_initialized());
        assert_eq!(lazy.access_count(), 0);

        let val = lazy.get();
        assert_eq!(val, "default_value");
        assert!(lazy.is_initialized());
        assert_eq!(lazy.access_count(), 1);
    }

    #[test]
    fn lazy_fixture_reset() {
        let mut lazy = LazyFixture::new("x", "val");
        lazy.get();
        assert!(lazy.is_initialized());
        lazy.reset();
        assert!(!lazy.is_initialized());
        assert_eq!(lazy.access_count(), 0);
    }

    #[test]
    fn composite_fixture_merge() {
        let mut a = CompositeFixture::new();
        a.add("db", "postgres");
        a.add("cache", "redis");

        let mut b = CompositeFixture::new();
        b.add("queue", "rabbitmq");

        a.merge(&b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.get("queue"), Some("rabbitmq"));
    }

    #[test]
    fn composite_fixture_names_sorted() {
        let mut c = CompositeFixture::new();
        c.add("z_service", "v1");
        c.add("a_service", "v2");
        let names = c.names();
        assert_eq!(names[0], "a_service");
        assert_eq!(names[1], "z_service");
    }

    #[test]
    fn manager_clear() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("x").with_default_value("v"))
            .unwrap();
        mgr.setup("x").unwrap();
        mgr.clear();
        assert_eq!(mgr.num_definitions(), 0);
        assert_eq!(mgr.num_active(), 0);
    }

    #[test]
    fn modify_fixture_data() {
        let mut mgr = FixtureManager::new();
        mgr.register(FixtureDef::new("state").with_default_value("initial"))
            .unwrap();
        mgr.setup("state").unwrap();
        mgr.set_data("state", "modified").unwrap();
        assert_eq!(mgr.get_data("state").unwrap(), "modified");
    }

    #[test]
    fn error_display() {
        let err = FixtureError::NotFound("db".to_string());
        assert!(format!("{err}").contains("db"));

        let err = FixtureError::CyclicDependency("a -> b -> a".to_string());
        assert!(format!("{err}").contains("cyclic"));
    }

    #[test]
    fn fixture_scope_values() {
        assert_ne!(FixtureScope::Test, FixtureScope::Session);
        assert_ne!(FixtureScope::Session, FixtureScope::Module);
    }
}
