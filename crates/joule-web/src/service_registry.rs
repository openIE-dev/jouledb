//! Service discovery registry — registration, heartbeat, lookup, watch.
//!
//! Pure Rust service registry for service discovery. Supports service
//! registration with name/address/port/metadata, heartbeat-based TTL,
//! service lookup by name, health status tracking, tag-based filtering,
//! change watching, and deregistration.

use std::collections::HashMap;
use std::fmt;

// ── Health ────────────────────────────────────────────────────

/// Service health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceHealth {
    Passing,
    Warning,
    Critical,
    Unknown,
}

impl fmt::Display for ServiceHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Passing => "passing",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

// ── Service Instance ──────────────────────────────────────────

/// A registered service instance.
#[derive(Debug, Clone)]
pub struct ServiceInstance {
    /// Unique instance ID.
    pub id: String,
    /// Service name (multiple instances may share the same name).
    pub name: String,
    /// IP address or hostname.
    pub address: String,
    /// Port number.
    pub port: u16,
    /// Tags for filtering.
    pub tags: Vec<String>,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, String>,
    /// Current health status.
    pub health: ServiceHealth,
    /// Timestamp of last heartbeat (ms since epoch).
    pub last_heartbeat_ms: u64,
    /// TTL for heartbeat in milliseconds. If no heartbeat within TTL, mark critical.
    pub ttl_ms: u64,
    /// Registration timestamp.
    pub registered_at_ms: u64,
    /// Version string.
    pub version: String,
}

impl ServiceInstance {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        address: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            address: address.into(),
            port,
            tags: Vec::new(),
            metadata: HashMap::new(),
            health: ServiceHealth::Unknown,
            last_heartbeat_ms: 0,
            ttl_ms: 30_000,
            registered_at_ms: 0,
            version: String::new(),
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_ttl(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = ttl_ms;
        self
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if TTL has expired at the given timestamp.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        if self.last_heartbeat_ms == 0 {
            return false; // Never heartbeated yet.
        }
        now_ms.saturating_sub(self.last_heartbeat_ms) > self.ttl_ms
    }

    /// Whether this instance has a specific tag.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }

    /// Whether this instance is healthy (passing or warning).
    pub fn is_healthy(&self) -> bool {
        matches!(self.health, ServiceHealth::Passing | ServiceHealth::Warning)
    }

    /// Endpoint string.
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.address, self.port)
    }
}

impl fmt::Display for ServiceInstance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{} @ {}:{}", self.name, self.id, self.address, self.port)
    }
}

// ── Change Events ─────────────────────────────────────────────

/// Events emitted when the registry changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryEvent {
    Registered { service_id: String, service_name: String },
    Deregistered { service_id: String, service_name: String },
    HealthChanged { service_id: String, old: ServiceHealth, new: ServiceHealth },
    HeartbeatReceived { service_id: String },
    Expired { service_id: String, service_name: String },
}

// ── Watcher ───────────────────────────────────────────────────

/// A watcher that receives events for a specific service name.
#[derive(Debug)]
pub struct Watcher {
    pub id: u64,
    /// Service name to watch (empty = watch all).
    pub service_name: String,
    /// Buffered events.
    events: Vec<RegistryEvent>,
}

impl Watcher {
    fn new(id: u64, service_name: impl Into<String>) -> Self {
        Self {
            id,
            service_name: service_name.into(),
            events: Vec::new(),
        }
    }

    /// Drain all buffered events.
    pub fn drain(&mut self) -> Vec<RegistryEvent> {
        std::mem::take(&mut self.events)
    }

    /// Number of pending events.
    pub fn pending(&self) -> usize {
        self.events.len()
    }
}

// ── Service Registry ──────────────────────────────────────────

/// Service discovery registry.
pub struct ServiceRegistry {
    /// All registered instances by instance ID.
    instances: HashMap<String, ServiceInstance>,
    /// Index: service_name -> list of instance IDs.
    by_name: HashMap<String, Vec<String>>,
    /// Active watchers.
    watchers: Vec<Watcher>,
    /// Next watcher ID.
    next_watcher_id: u64,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self {
            instances: HashMap::new(),
            by_name: HashMap::new(),
            watchers: Vec::new(),
            next_watcher_id: 1,
        }
    }

    /// Register a service instance.
    pub fn register(&mut self, mut instance: ServiceInstance, now_ms: u64) -> bool {
        if self.instances.contains_key(&instance.id) {
            return false; // Already registered.
        }

        instance.registered_at_ms = now_ms;
        instance.last_heartbeat_ms = now_ms;
        instance.health = ServiceHealth::Passing;

        let name = instance.name.clone();
        let id = instance.id.clone();

        self.instances.insert(id.clone(), instance);
        self.by_name.entry(name.clone()).or_default().push(id.clone());

        self.emit_event(RegistryEvent::Registered {
            service_id: id,
            service_name: name,
        });

        true
    }

    /// Deregister a service instance.
    pub fn deregister(&mut self, id: &str) -> Option<ServiceInstance> {
        let instance = self.instances.remove(id)?;

        if let Some(ids) = self.by_name.get_mut(&instance.name) {
            ids.retain(|i| i != id);
            if ids.is_empty() {
                self.by_name.remove(&instance.name);
            }
        }

        self.emit_event(RegistryEvent::Deregistered {
            service_id: instance.id.clone(),
            service_name: instance.name.clone(),
        });

        Some(instance)
    }

    /// Record a heartbeat for an instance.
    pub fn heartbeat(&mut self, id: &str, now_ms: u64) -> bool {
        if let Some(instance) = self.instances.get_mut(id) {
            instance.last_heartbeat_ms = now_ms;
            // If it was critical due to TTL expiry, recover.
            if instance.health == ServiceHealth::Critical {
                let old = instance.health;
                instance.health = ServiceHealth::Passing;
                self.emit_event(RegistryEvent::HealthChanged {
                    service_id: id.to_string(),
                    old,
                    new: ServiceHealth::Passing,
                });
            }
            self.emit_event(RegistryEvent::HeartbeatReceived {
                service_id: id.to_string(),
            });
            true
        } else {
            false
        }
    }

    /// Set health status for an instance.
    pub fn set_health(&mut self, id: &str, health: ServiceHealth) -> bool {
        if let Some(instance) = self.instances.get_mut(id) {
            let old = instance.health;
            if old != health {
                instance.health = health;
                self.emit_event(RegistryEvent::HealthChanged {
                    service_id: id.to_string(),
                    old,
                    new: health,
                });
            }
            true
        } else {
            false
        }
    }

    /// Check for expired instances and mark them critical.
    pub fn check_expirations(&mut self, now_ms: u64) -> Vec<String> {
        let mut expired = Vec::new();

        let expired_ids: Vec<(String, String)> = self.instances
            .iter()
            .filter(|(_, inst)| inst.is_expired(now_ms) && inst.health != ServiceHealth::Critical)
            .map(|(id, inst)| (id.clone(), inst.name.clone()))
            .collect();

        for (id, name) in expired_ids {
            if let Some(inst) = self.instances.get_mut(&id) {
                inst.health = ServiceHealth::Critical;
            }
            expired.push(id.clone());
            self.emit_event(RegistryEvent::Expired {
                service_id: id,
                service_name: name,
            });
        }

        expired
    }

    /// Look up all instances of a service by name.
    pub fn lookup(&self, name: &str) -> Vec<&ServiceInstance> {
        match self.by_name.get(name) {
            Some(ids) => ids
                .iter()
                .filter_map(|id| self.instances.get(id))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Look up healthy instances of a service.
    pub fn lookup_healthy(&self, name: &str) -> Vec<&ServiceInstance> {
        self.lookup(name)
            .into_iter()
            .filter(|i| i.is_healthy())
            .collect()
    }

    /// Look up instances by tag.
    pub fn lookup_by_tag(&self, tag: &str) -> Vec<&ServiceInstance> {
        self.instances
            .values()
            .filter(|i| i.has_tag(tag))
            .collect()
    }

    /// Look up instances by name and tag.
    pub fn lookup_by_name_and_tag(&self, name: &str, tag: &str) -> Vec<&ServiceInstance> {
        self.lookup(name)
            .into_iter()
            .filter(|i| i.has_tag(tag))
            .collect()
    }

    /// Get instance by ID.
    pub fn get(&self, id: &str) -> Option<&ServiceInstance> {
        self.instances.get(id)
    }

    /// List all registered service names.
    pub fn service_names(&self) -> Vec<&str> {
        self.by_name.keys().map(|s| s.as_str()).collect()
    }

    /// Total instance count.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    // ── Watchers ──────────────────────────────────────────────

    /// Register a watcher for a service name. Returns watcher ID.
    pub fn watch(&mut self, service_name: impl Into<String>) -> u64 {
        let id = self.next_watcher_id;
        self.next_watcher_id += 1;
        self.watchers.push(Watcher::new(id, service_name));
        id
    }

    /// Remove a watcher.
    pub fn unwatch(&mut self, watcher_id: u64) -> bool {
        let before = self.watchers.len();
        self.watchers.retain(|w| w.id != watcher_id);
        self.watchers.len() < before
    }

    /// Drain events from a watcher.
    pub fn drain_events(&mut self, watcher_id: u64) -> Vec<RegistryEvent> {
        if let Some(w) = self.watchers.iter_mut().find(|w| w.id == watcher_id) {
            w.drain()
        } else {
            Vec::new()
        }
    }

    fn emit_event(&mut self, event: RegistryEvent) {
        let service_name = match &event {
            RegistryEvent::Registered { service_name, .. } => service_name.clone(),
            RegistryEvent::Deregistered { service_name, .. } => service_name.clone(),
            RegistryEvent::HealthChanged { service_id, .. } => {
                self.instances
                    .get(service_id)
                    .map(|i| i.name.clone())
                    .unwrap_or_default()
            }
            RegistryEvent::HeartbeatReceived { service_id } => {
                self.instances
                    .get(service_id)
                    .map(|i| i.name.clone())
                    .unwrap_or_default()
            }
            RegistryEvent::Expired { service_name, .. } => service_name.clone(),
        };

        for watcher in &mut self.watchers {
            if watcher.service_name.is_empty() || watcher.service_name == service_name {
                watcher.events.push(event.clone());
            }
        }
    }

    /// Deregister all expired instances and return their IDs.
    pub fn prune_expired(&mut self, now_ms: u64) -> Vec<String> {
        let expired_ids: Vec<String> = self.instances
            .iter()
            .filter(|(_, inst)| inst.is_expired(now_ms))
            .map(|(id, _)| id.clone())
            .collect();

        let mut pruned = Vec::new();
        for id in expired_ids {
            if self.deregister(&id).is_some() {
                pruned.push(id);
            }
        }
        pruned
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instance(id: &str, name: &str, port: u16) -> ServiceInstance {
        ServiceInstance::new(id, name, "10.0.0.1", port)
    }

    #[test]
    fn test_register_and_lookup() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080), 1000);
        reg.register(make_instance("web-2", "web", 8081), 1000);

        let results = reg.lookup("web");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_register_duplicate_fails() {
        let mut reg = ServiceRegistry::new();
        assert!(reg.register(make_instance("web-1", "web", 8080), 1000));
        assert!(!reg.register(make_instance("web-1", "web", 8080), 2000));
    }

    #[test]
    fn test_deregister() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080), 1000);
        let inst = reg.deregister("web-1").unwrap();
        assert_eq!(inst.name, "web");
        assert!(reg.lookup("web").is_empty());
    }

    #[test]
    fn test_heartbeat() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080), 1000);
        assert!(reg.heartbeat("web-1", 5000));
        assert_eq!(reg.get("web-1").unwrap().last_heartbeat_ms, 5000);
    }

    #[test]
    fn test_heartbeat_nonexistent() {
        let mut reg = ServiceRegistry::new();
        assert!(!reg.heartbeat("nope", 1000));
    }

    #[test]
    fn test_ttl_expiry() {
        let mut reg = ServiceRegistry::new();
        let inst = make_instance("web-1", "web", 8080).with_ttl(5000);
        reg.register(inst, 1000);

        // Not yet expired.
        let expired = reg.check_expirations(4000);
        assert!(expired.is_empty());

        // Expired.
        let expired = reg.check_expirations(10000);
        assert_eq!(expired.len(), 1);
        assert_eq!(reg.get("web-1").unwrap().health, ServiceHealth::Critical);
    }

    #[test]
    fn test_heartbeat_recovers_critical() {
        let mut reg = ServiceRegistry::new();
        let inst = make_instance("web-1", "web", 8080).with_ttl(5000);
        reg.register(inst, 1000);
        reg.check_expirations(10000);
        assert_eq!(reg.get("web-1").unwrap().health, ServiceHealth::Critical);

        reg.heartbeat("web-1", 11000);
        assert_eq!(reg.get("web-1").unwrap().health, ServiceHealth::Passing);
    }

    #[test]
    fn test_set_health() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080), 1000);
        reg.set_health("web-1", ServiceHealth::Warning);
        assert_eq!(reg.get("web-1").unwrap().health, ServiceHealth::Warning);
    }

    #[test]
    fn test_lookup_healthy() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080), 1000);
        reg.register(make_instance("web-2", "web", 8081), 1000);
        reg.set_health("web-2", ServiceHealth::Critical);

        let healthy = reg.lookup_healthy("web");
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].id, "web-1");
    }

    #[test]
    fn test_tag_based_lookup() {
        let mut reg = ServiceRegistry::new();
        let inst = make_instance("web-1", "web", 8080)
            .with_tags(vec!["production".into(), "v2".into()]);
        reg.register(inst, 1000);

        let inst2 = make_instance("web-2", "web", 8081)
            .with_tags(vec!["staging".into()]);
        reg.register(inst2, 1000);

        let prod = reg.lookup_by_tag("production");
        assert_eq!(prod.len(), 1);
        assert_eq!(prod[0].id, "web-1");
    }

    #[test]
    fn test_lookup_by_name_and_tag() {
        let mut reg = ServiceRegistry::new();
        reg.register(
            make_instance("web-1", "web", 8080).with_tags(vec!["v2".into()]),
            1000,
        );
        reg.register(
            make_instance("api-1", "api", 9090).with_tags(vec!["v2".into()]),
            1000,
        );

        let results = reg.lookup_by_name_and_tag("web", "v2");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "web");
    }

    #[test]
    fn test_service_names() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080), 1000);
        reg.register(make_instance("api-1", "api", 9090), 1000);

        let mut names = reg.service_names();
        names.sort();
        assert_eq!(names, vec!["api", "web"]);
    }

    #[test]
    fn test_watcher_events() {
        let mut reg = ServiceRegistry::new();
        let wid = reg.watch("web");

        reg.register(make_instance("web-1", "web", 8080), 1000);
        let events = reg.drain_events(wid);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], RegistryEvent::Registered { .. }));
    }

    #[test]
    fn test_watcher_filtered() {
        let mut reg = ServiceRegistry::new();
        let web_watcher = reg.watch("web");
        let _api_watcher = reg.watch("api");

        reg.register(make_instance("web-1", "web", 8080), 1000);
        assert_eq!(reg.drain_events(web_watcher).len(), 1);
        assert_eq!(reg.drain_events(_api_watcher).len(), 0);
    }

    #[test]
    fn test_watcher_all_events() {
        let mut reg = ServiceRegistry::new();
        let all_watcher = reg.watch("");

        reg.register(make_instance("web-1", "web", 8080), 1000);
        reg.register(make_instance("api-1", "api", 9090), 1000);

        assert_eq!(reg.drain_events(all_watcher).len(), 2);
    }

    #[test]
    fn test_unwatch() {
        let mut reg = ServiceRegistry::new();
        let wid = reg.watch("web");
        assert!(reg.unwatch(wid));
        assert!(!reg.unwatch(wid)); // Already removed.
    }

    #[test]
    fn test_prune_expired() {
        let mut reg = ServiceRegistry::new();
        reg.register(make_instance("web-1", "web", 8080).with_ttl(5000), 1000);
        reg.register(make_instance("web-2", "web", 8081).with_ttl(5000), 1000);

        let pruned = reg.prune_expired(20000);
        assert_eq!(pruned.len(), 2);
        assert_eq!(reg.instance_count(), 0);
    }

    #[test]
    fn test_instance_endpoint() {
        let inst = make_instance("web-1", "web", 8080);
        assert_eq!(inst.endpoint(), "10.0.0.1:8080");
    }

    #[test]
    fn test_instance_display() {
        let inst = make_instance("web-1", "web", 8080);
        assert_eq!(format!("{}", inst), "web/web-1 @ 10.0.0.1:8080");
    }

    #[test]
    fn test_service_with_metadata() {
        let mut reg = ServiceRegistry::new();
        let inst = make_instance("web-1", "web", 8080)
            .with_metadata("region", "us-east")
            .with_version("1.2.3");
        reg.register(inst, 1000);

        let i = reg.get("web-1").unwrap();
        assert_eq!(i.metadata.get("region").unwrap(), "us-east");
        assert_eq!(i.version, "1.2.3");
    }
}
