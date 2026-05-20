//! Dependency injection container — service registration, resolution, and scoping.
//!
//! Replaces InversifyJS / Awilix / tsyringe with a pure-Rust DI container.
//! Supports singleton/transient/scoped lifetimes, constructor injection,
//! interface-based resolution, lifecycle management, circular dependency
//! detection, and child containers (scopes).

use std::collections::{HashMap, HashSet};

// ── Errors ──────────────────────────────────────────────────────

/// DI container domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiError {
    /// Service not registered.
    ServiceNotFound(String),
    /// Circular dependency detected.
    CircularDependency { chain: Vec<String> },
    /// Duplicate registration.
    DuplicateRegistration(String),
    /// Scope not found.
    ScopeNotFound(String),
    /// Invalid lifecycle transition.
    LifecycleError { service: String, reason: String },
    /// Container is frozen (no more registrations).
    ContainerFrozen,
}

impl std::fmt::Display for DiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ServiceNotFound(name) => write!(f, "service not found: {name}"),
            Self::CircularDependency { chain } => {
                write!(f, "circular dependency: {}", chain.join(" -> "))
            }
            Self::DuplicateRegistration(name) => write!(f, "duplicate registration: {name}"),
            Self::ScopeNotFound(name) => write!(f, "scope not found: {name}"),
            Self::LifecycleError { service, reason } => {
                write!(f, "lifecycle error for {service}: {reason}")
            }
            Self::ContainerFrozen => write!(f, "container is frozen"),
        }
    }
}

impl std::error::Error for DiError {}

// ── Service Lifetime ────────────────────────────────────────────

/// Lifetime of a service in the container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifetime {
    /// One instance shared globally.
    Singleton,
    /// New instance on every resolve.
    Transient,
    /// One instance per scope.
    Scoped,
}

// ── Service State ───────────────────────────────────────────────

/// Lifecycle state of a service instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceState {
    Registered,
    Initializing,
    Ready,
    Disposing,
    Disposed,
}

// ── Service Registration ────────────────────────────────────────

/// A registered service descriptor.
#[derive(Debug, Clone)]
pub struct ServiceRegistration {
    pub name: String,
    pub lifetime: Lifetime,
    pub dependencies: Vec<String>,
    pub interface_name: Option<String>,
    pub state: ServiceState,
    /// Factory value: simulated by a string template.
    pub factory_template: String,
    /// Tags for grouping.
    pub tags: Vec<String>,
}

impl ServiceRegistration {
    pub fn new(name: impl Into<String>, lifetime: Lifetime) -> Self {
        let name_str = name.into();
        Self {
            factory_template: name_str.clone(),
            name: name_str,
            lifetime,
            dependencies: Vec::new(),
            interface_name: None,
            state: ServiceState::Registered,
            tags: Vec::new(),
        }
    }

    /// Add a dependency.
    pub fn depends_on(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    /// Register as an implementation of an interface.
    pub fn implements(mut self, iface: impl Into<String>) -> Self {
        self.interface_name = Some(iface.into());
        self
    }

    /// Set the factory template.
    pub fn with_factory(mut self, template: impl Into<String>) -> Self {
        self.factory_template = template.into();
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ── Resolved Instance ───────────────────────────────────────────

/// A resolved service instance.
#[derive(Debug, Clone)]
pub struct ServiceInstance {
    pub name: String,
    pub value: String,
    pub resolve_count: u64,
}

// ── Container ───────────────────────────────────────────────────

/// Dependency injection container.
pub struct Container {
    registrations: HashMap<String, ServiceRegistration>,
    /// Interface -> implementation name mapping.
    interfaces: HashMap<String, String>,
    /// Singleton instances.
    singletons: HashMap<String, ServiceInstance>,
    /// Scoped instances per scope name.
    scoped_instances: HashMap<String, HashMap<String, ServiceInstance>>,
    /// Resolution counter.
    resolve_counter: u64,
    /// Whether the container is frozen.
    frozen: bool,
    /// Active scopes.
    active_scopes: HashSet<String>,
    /// Parent container registrations (for child containers).
    parent_registrations: Option<HashMap<String, ServiceRegistration>>,
    parent_interfaces: Option<HashMap<String, String>>,
    parent_singletons: Option<HashMap<String, ServiceInstance>>,
}

impl Container {
    pub fn new() -> Self {
        Self {
            registrations: HashMap::new(),
            interfaces: HashMap::new(),
            singletons: HashMap::new(),
            scoped_instances: HashMap::new(),
            resolve_counter: 0,
            frozen: false,
            active_scopes: HashSet::new(),
            parent_registrations: None,
            parent_interfaces: None,
            parent_singletons: None,
        }
    }

    /// Register a service.
    pub fn register(
        &mut self,
        registration: ServiceRegistration,
    ) -> Result<(), DiError> {
        if self.frozen {
            return Err(DiError::ContainerFrozen);
        }
        if self.registrations.contains_key(&registration.name) {
            return Err(DiError::DuplicateRegistration(registration.name));
        }
        let name = registration.name.clone();
        if let Some(iface) = &registration.interface_name {
            self.interfaces.insert(iface.clone(), name.clone());
        }
        self.registrations.insert(name, registration);
        Ok(())
    }

    /// Freeze the container — no more registrations allowed.
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Whether the container is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Resolve a service by name.
    pub fn resolve(&mut self, name: &str) -> Result<ServiceInstance, DiError> {
        self.resolve_with_chain(name, &mut Vec::new(), None)
    }

    /// Resolve a service by interface name.
    pub fn resolve_interface(&mut self, interface: &str) -> Result<ServiceInstance, DiError> {
        let impl_name = self
            .interfaces
            .get(interface)
            .or_else(|| {
                self.parent_interfaces
                    .as_ref()
                    .and_then(|pi| pi.get(interface))
            })
            .cloned()
            .ok_or_else(|| DiError::ServiceNotFound(interface.to_string()))?;
        self.resolve(&impl_name)
    }

    /// Resolve a service within a scope.
    pub fn resolve_scoped(
        &mut self,
        name: &str,
        scope: &str,
    ) -> Result<ServiceInstance, DiError> {
        if !self.active_scopes.contains(scope) {
            return Err(DiError::ScopeNotFound(scope.to_string()));
        }
        self.resolve_with_chain(name, &mut Vec::new(), Some(scope))
    }

    fn resolve_with_chain(
        &mut self,
        name: &str,
        chain: &mut Vec<String>,
        scope: Option<&str>,
    ) -> Result<ServiceInstance, DiError> {
        // Check for circular dependencies
        if chain.contains(&name.to_string()) {
            chain.push(name.to_string());
            return Err(DiError::CircularDependency {
                chain: chain.clone(),
            });
        }
        chain.push(name.to_string());

        // Find registration
        let reg = self
            .registrations
            .get(name)
            .or_else(|| {
                self.parent_registrations
                    .as_ref()
                    .and_then(|pr| pr.get(name))
            })
            .cloned()
            .ok_or_else(|| DiError::ServiceNotFound(name.to_string()))?;

        match reg.lifetime {
            Lifetime::Singleton => {
                // Check singleton cache
                if let Some(instance) = self.singletons.get(name) {
                    return Ok(instance.clone());
                }
                if let Some(instance) = self
                    .parent_singletons
                    .as_ref()
                    .and_then(|ps| ps.get(name))
                {
                    return Ok(instance.clone());
                }
                let instance = self.create_instance(&reg, chain, scope)?;
                self.singletons.insert(name.to_string(), instance.clone());
                Ok(instance)
            }
            Lifetime::Transient => self.create_instance(&reg, chain, scope),
            Lifetime::Scoped => {
                let scope_name = scope.unwrap_or("default");
                // Check scope cache
                if let Some(scope_map) = self.scoped_instances.get(scope_name) {
                    if let Some(instance) = scope_map.get(name) {
                        return Ok(instance.clone());
                    }
                }
                let instance = self.create_instance(&reg, chain, scope)?;
                self.scoped_instances
                    .entry(scope_name.to_string())
                    .or_default()
                    .insert(name.to_string(), instance.clone());
                Ok(instance)
            }
        }
    }

    fn create_instance(
        &mut self,
        reg: &ServiceRegistration,
        chain: &mut Vec<String>,
        scope: Option<&str>,
    ) -> Result<ServiceInstance, DiError> {
        // Resolve dependencies first
        let mut dep_values = Vec::new();
        let deps = reg.dependencies.clone();
        for dep in &deps {
            let dep_instance = self.resolve_with_chain(dep, chain, scope)?;
            dep_values.push(dep_instance.value);
        }

        self.resolve_counter += 1;
        let value = if dep_values.is_empty() {
            reg.factory_template.clone()
        } else {
            format!("{}({})", reg.factory_template, dep_values.join(","))
        };

        Ok(ServiceInstance {
            name: reg.name.clone(),
            value,
            resolve_count: self.resolve_counter,
        })
    }

    /// Create a named scope.
    pub fn create_scope(&mut self, name: impl Into<String>) -> String {
        let scope_name = name.into();
        self.active_scopes.insert(scope_name.clone());
        self.scoped_instances
            .entry(scope_name.clone())
            .or_default();
        scope_name
    }

    /// Dispose a scope — remove all scoped instances.
    pub fn dispose_scope(&mut self, name: &str) -> Result<usize, DiError> {
        if !self.active_scopes.remove(name) {
            return Err(DiError::ScopeNotFound(name.to_string()));
        }
        let count = self
            .scoped_instances
            .remove(name)
            .map(|m| m.len())
            .unwrap_or(0);
        Ok(count)
    }

    /// Create a child container that inherits from this one.
    pub fn create_child(&self) -> Container {
        Container {
            registrations: HashMap::new(),
            interfaces: HashMap::new(),
            singletons: HashMap::new(),
            scoped_instances: HashMap::new(),
            resolve_counter: 0,
            frozen: false,
            active_scopes: HashSet::new(),
            parent_registrations: Some(self.registrations.clone()),
            parent_interfaces: Some(self.interfaces.clone()),
            parent_singletons: Some(self.singletons.clone()),
        }
    }

    /// Check if a service is registered.
    pub fn is_registered(&self, name: &str) -> bool {
        self.registrations.contains_key(name)
            || self
                .parent_registrations
                .as_ref()
                .map(|pr| pr.contains_key(name))
                .unwrap_or(false)
    }

    /// Get all service names matching a tag.
    pub fn get_by_tag(&self, tag: &str) -> Vec<String> {
        self.registrations
            .values()
            .filter(|r| r.tags.contains(&tag.to_string()))
            .map(|r| r.name.clone())
            .collect()
    }

    /// Number of registered services.
    pub fn registration_count(&self) -> usize {
        self.registrations.len()
    }

    /// Number of active scopes.
    pub fn scope_count(&self) -> usize {
        self.active_scopes.len()
    }

    /// Number of singleton instances.
    pub fn singleton_count(&self) -> usize {
        self.singletons.len()
    }

    /// Dispose a singleton — remove it from cache.
    pub fn dispose_singleton(&mut self, name: &str) -> Result<(), DiError> {
        if self.singletons.remove(name).is_none() {
            return Err(DiError::ServiceNotFound(name.to_string()));
        }
        Ok(())
    }
}

impl Default for Container {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_resolve() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        let inst = c.resolve("db").unwrap();
        assert_eq!(inst.name, "db");
        assert_eq!(inst.value, "db");
    }

    #[test]
    fn test_resolve_not_found() {
        let mut c = Container::new();
        let err = c.resolve("missing").unwrap_err();
        assert_eq!(err, DiError::ServiceNotFound("missing".to_string()));
    }

    #[test]
    fn test_duplicate_registration() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        let err = c
            .register(ServiceRegistration::new("db", Lifetime::Transient))
            .unwrap_err();
        assert_eq!(err, DiError::DuplicateRegistration("db".to_string()));
    }

    #[test]
    fn test_singleton_returns_same_instance() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        let i1 = c.resolve("db").unwrap();
        let i2 = c.resolve("db").unwrap();
        assert_eq!(i1.resolve_count, i2.resolve_count);
    }

    #[test]
    fn test_transient_creates_new_instance() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("logger", Lifetime::Transient))
            .unwrap();
        let i1 = c.resolve("logger").unwrap();
        let i2 = c.resolve("logger").unwrap();
        assert_ne!(i1.resolve_count, i2.resolve_count);
    }

    #[test]
    fn test_scoped_lifetime() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("ctx", Lifetime::Scoped))
            .unwrap();
        let scope = c.create_scope("request1");
        let i1 = c.resolve_scoped("ctx", &scope).unwrap();
        let i2 = c.resolve_scoped("ctx", &scope).unwrap();
        assert_eq!(i1.resolve_count, i2.resolve_count);
    }

    #[test]
    fn test_different_scopes_different_instances() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("ctx", Lifetime::Scoped))
            .unwrap();
        let s1 = c.create_scope("r1");
        let s2 = c.create_scope("r2");
        let i1 = c.resolve_scoped("ctx", &s1).unwrap();
        let i2 = c.resolve_scoped("ctx", &s2).unwrap();
        assert_ne!(i1.resolve_count, i2.resolve_count);
    }

    #[test]
    fn test_dependency_injection() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        c.register(
            ServiceRegistration::new("repo", Lifetime::Transient)
                .depends_on("db")
                .with_factory("Repository"),
        )
        .unwrap();
        let inst = c.resolve("repo").unwrap();
        assert_eq!(inst.value, "Repository(db)");
    }

    #[test]
    fn test_circular_dependency() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("a", Lifetime::Transient).depends_on("b"))
            .unwrap();
        c.register(ServiceRegistration::new("b", Lifetime::Transient).depends_on("a"))
            .unwrap();
        let err = c.resolve("a").unwrap_err();
        assert!(matches!(err, DiError::CircularDependency { .. }));
    }

    #[test]
    fn test_interface_resolution() {
        let mut c = Container::new();
        c.register(
            ServiceRegistration::new("pg_db", Lifetime::Singleton)
                .implements("Database")
                .with_factory("PostgresDB"),
        )
        .unwrap();
        let inst = c.resolve_interface("Database").unwrap();
        assert_eq!(inst.name, "pg_db");
    }

    #[test]
    fn test_child_container() {
        let mut parent = Container::new();
        parent
            .register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        parent.resolve("db").unwrap(); // Create singleton

        let mut child = parent.create_child();
        // Child can resolve parent registrations
        let inst = child.resolve("db").unwrap();
        assert_eq!(inst.name, "db");
    }

    #[test]
    fn test_child_container_override() {
        let mut parent = Container::new();
        parent
            .register(ServiceRegistration::new("db", Lifetime::Singleton).with_factory("ParentDB"))
            .unwrap();

        let mut child = parent.create_child();
        child
            .register(ServiceRegistration::new("db", Lifetime::Singleton).with_factory("ChildDB"))
            .unwrap();
        let inst = child.resolve("db").unwrap();
        assert_eq!(inst.value, "ChildDB");
    }

    #[test]
    fn test_container_freeze() {
        let mut c = Container::new();
        c.freeze();
        let err = c
            .register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap_err();
        assert_eq!(err, DiError::ContainerFrozen);
    }

    #[test]
    fn test_dispose_scope() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("ctx", Lifetime::Scoped))
            .unwrap();
        let scope = c.create_scope("s1");
        c.resolve_scoped("ctx", &scope).unwrap();
        let count = c.dispose_scope(&scope).unwrap();
        assert_eq!(count, 1);
        assert_eq!(c.scope_count(), 0);
    }

    #[test]
    fn test_dispose_nonexistent_scope() {
        let mut c = Container::new();
        let err = c.dispose_scope("nope").unwrap_err();
        assert_eq!(err, DiError::ScopeNotFound("nope".to_string()));
    }

    #[test]
    fn test_tags() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("s1", Lifetime::Transient).with_tag("http"))
            .unwrap();
        c.register(ServiceRegistration::new("s2", Lifetime::Transient).with_tag("http"))
            .unwrap();
        c.register(ServiceRegistration::new("s3", Lifetime::Transient).with_tag("grpc"))
            .unwrap();
        let http_services = c.get_by_tag("http");
        assert_eq!(http_services.len(), 2);
    }

    #[test]
    fn test_registration_count() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("a", Lifetime::Transient))
            .unwrap();
        c.register(ServiceRegistration::new("b", Lifetime::Singleton))
            .unwrap();
        assert_eq!(c.registration_count(), 2);
    }

    #[test]
    fn test_is_registered() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        assert!(c.is_registered("db"));
        assert!(!c.is_registered("cache"));
    }

    #[test]
    fn test_dispose_singleton() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("db", Lifetime::Singleton))
            .unwrap();
        c.resolve("db").unwrap();
        assert_eq!(c.singleton_count(), 1);
        c.dispose_singleton("db").unwrap();
        assert_eq!(c.singleton_count(), 0);
    }

    #[test]
    fn test_deep_dependency_chain() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("a", Lifetime::Transient).with_factory("A"))
            .unwrap();
        c.register(
            ServiceRegistration::new("b", Lifetime::Transient)
                .depends_on("a")
                .with_factory("B"),
        )
        .unwrap();
        c.register(
            ServiceRegistration::new("c", Lifetime::Transient)
                .depends_on("b")
                .with_factory("C"),
        )
        .unwrap();
        let inst = c.resolve("c").unwrap();
        assert_eq!(inst.value, "C(B(A))");
    }

    #[test]
    fn test_resolve_scoped_without_scope() {
        let mut c = Container::new();
        c.register(ServiceRegistration::new("ctx", Lifetime::Scoped))
            .unwrap();
        let err = c.resolve_scoped("ctx", "nonexistent").unwrap_err();
        assert_eq!(err, DiError::ScopeNotFound("nonexistent".to_string()));
    }
}
