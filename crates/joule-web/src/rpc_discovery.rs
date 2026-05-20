//! Service discovery for RPC — registration, health, load balancing, routing.
//!
//! Pure-Rust service discovery framework. Supports service registration with
//! health status, load balancing (round-robin, random, weighted), service
//! versioning, service mesh routing, per-service retry policies, circuit
//! breaker integration, and endpoint selection.

use std::collections::HashMap;
use std::fmt;

// ── Health Status ────────────────────────────────────────────

/// Health status of a service endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl HealthStatus {
    /// Whether the endpoint can serve traffic.
    pub fn is_available(self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
            Self::Unknown => "unknown",
        };
        f.write_str(s)
    }
}

// ── Service Version ──────────────────────────────────────────

/// Semantic version for service versioning.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ServiceVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ServiceVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Parse from "major.minor.patch" string.
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;
        Some(Self { major, minor, patch })
    }

    /// Whether this version is compatible with `other` (same major).
    pub fn is_compatible(&self, other: &ServiceVersion) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for ServiceVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for ServiceVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ServiceVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major.cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

// ── Endpoint ─────────────────────────────────────────────────

/// A service endpoint (instance).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    /// Unique endpoint ID.
    pub id: String,
    /// Host address.
    pub host: String,
    /// Port.
    pub port: u16,
    /// Health status.
    pub health: HealthStatus,
    /// Weight for weighted load balancing (1-100).
    pub weight: u32,
    /// Metadata tags.
    pub tags: HashMap<String, String>,
    /// Last heartbeat timestamp (ms since epoch).
    pub last_heartbeat_ms: u64,
    /// Version running on this endpoint.
    pub version: Option<ServiceVersion>,
    /// Current active connections (for load-aware balancing).
    pub active_connections: u32,
}

impl Endpoint {
    pub fn new(id: impl Into<String>, host: impl Into<String>, port: u16) -> Self {
        Self {
            id: id.into(),
            host: host.into(),
            port,
            health: HealthStatus::Unknown,
            weight: 100,
            tags: HashMap::new(),
            last_heartbeat_ms: 0,
            version: None,
            active_connections: 0,
        }
    }

    /// Set health status.
    pub fn with_health(mut self, health: HealthStatus) -> Self {
        self.health = health;
        self
    }

    /// Set weight.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight.clamp(1, 100);
        self
    }

    /// Set version.
    pub fn with_version(mut self, version: ServiceVersion) -> Self {
        self.version = Some(version);
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Address string "host:port".
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Whether this endpoint can serve traffic.
    pub fn is_available(&self) -> bool {
        self.health.is_available()
    }
}

// ── Retry Policy ─────────────────────────────────────────────

/// Per-service retry policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Maximum attempts.
    pub max_attempts: u32,
    /// Initial backoff in ms.
    pub initial_backoff_ms: u64,
    /// Maximum backoff in ms.
    pub max_backoff_ms: u64,
    /// Backoff multiplier (x100 for integer math — 200 = 2.0x).
    pub backoff_multiplier_x100: u32,
    /// Retryable status codes (as u32).
    pub retryable_codes: Vec<u32>,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            initial_backoff_ms: 100,
            max_backoff_ms: 10_000,
            backoff_multiplier_x100: 200,
            retryable_codes: vec![14], // Unavailable
        }
    }

    /// Compute backoff for attempt N (0-indexed).
    pub fn backoff_ms(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            return 0;
        }
        let mut delay = self.initial_backoff_ms;
        for _ in 1..attempt {
            delay = delay * self.backoff_multiplier_x100 as u64 / 100;
            if delay > self.max_backoff_ms {
                delay = self.max_backoff_ms;
                break;
            }
        }
        delay.min(self.max_backoff_ms)
    }

    /// Whether a code is retryable.
    pub fn is_retryable(&self, code: u32) -> bool {
        self.retryable_codes.contains(&code)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

// ── Circuit Breaker State ────────────────────────────────────

/// Circuit breaker state per endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitBreaker {
    /// Current state.
    pub state: CircuitState,
    /// Consecutive failures.
    pub failure_count: u32,
    /// Consecutive successes (in half-open).
    pub success_count: u32,
    /// Failure threshold to open.
    pub failure_threshold: u32,
    /// Success threshold to close from half-open.
    pub success_threshold: u32,
    /// Timestamp when circuit opened (ms).
    pub opened_at_ms: u64,
    /// How long to stay open before half-open (ms).
    pub open_duration_ms: u64,
}

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => f.write_str("closed"),
            Self::Open => f.write_str("open"),
            Self::HalfOpen => f.write_str("half-open"),
        }
    }
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, success_threshold: u32) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            failure_threshold,
            success_threshold,
            opened_at_ms: 0,
            open_duration_ms: 30_000,
        }
    }

    /// Set the open duration.
    pub fn with_open_duration(mut self, ms: u64) -> Self {
        self.open_duration_ms = ms;
        self
    }

    /// Whether the circuit allows requests.
    pub fn allows_request(&self, now_ms: u64) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open.
                now_ms >= self.opened_at_ms + self.open_duration_ms
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Record a success.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failure.
    pub fn record_failure(&mut self, now_ms: u64) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.opened_at_ms = now_ms;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.opened_at_ms = now_ms;
                self.success_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Attempt to transition from Open to HalfOpen.
    pub fn check_transition(&mut self, now_ms: u64) {
        if self.state == CircuitState::Open
            && now_ms >= self.opened_at_ms + self.open_duration_ms
        {
            self.state = CircuitState::HalfOpen;
            self.success_count = 0;
        }
    }

    /// Reset to closed.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.opened_at_ms = 0;
    }
}

// ── Load Balancing ───────────────────────────────────────────

/// Load balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    RoundRobin,
    Random,
    WeightedRoundRobin,
    LeastConnections,
}

/// Load balancer that selects endpoints.
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalanceStrategy,
    /// Current index for round-robin.
    current_index: usize,
    /// Deterministic seed for "random" (for testability).
    random_seed: u64,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalanceStrategy) -> Self {
        Self {
            strategy,
            current_index: 0,
            random_seed: 12345,
        }
    }

    /// Set the random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Select an endpoint from available endpoints. Returns index.
    pub fn select(&mut self, endpoints: &[Endpoint]) -> Option<usize> {
        let available: Vec<usize> = endpoints.iter()
            .enumerate()
            .filter(|(_, e)| e.is_available())
            .map(|(i, _)| i)
            .collect();

        if available.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.current_index % available.len();
                self.current_index = self.current_index.wrapping_add(1);
                Some(available[idx])
            }
            LoadBalanceStrategy::Random => {
                // Simple LCG for deterministic "random".
                self.random_seed = self.random_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (self.random_seed >> 33) as usize % available.len();
                Some(available[idx])
            }
            LoadBalanceStrategy::WeightedRoundRobin => {
                // Select based on weights.
                let total_weight: u32 = available.iter()
                    .map(|i| endpoints[*i].weight)
                    .sum();
                if total_weight == 0 {
                    return Some(available[0]);
                }
                self.random_seed = self.random_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let target = (self.random_seed >> 33) as u32 % total_weight;
                let mut cumulative = 0u32;
                for &i in &available {
                    cumulative += endpoints[i].weight;
                    if target < cumulative {
                        return Some(i);
                    }
                }
                Some(available[available.len() - 1])
            }
            LoadBalanceStrategy::LeastConnections => {
                let mut best = available[0];
                let mut best_conns = endpoints[available[0]].active_connections;
                for &i in &available[1..] {
                    if endpoints[i].active_connections < best_conns {
                        best = i;
                        best_conns = endpoints[i].active_connections;
                    }
                }
                Some(best)
            }
        }
    }

    /// Strategy in use.
    pub fn strategy(&self) -> LoadBalanceStrategy {
        self.strategy
    }

    /// Reset round-robin index.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}

// ── Service Registration ─────────────────────────────────────

/// A registered service with its endpoints.
#[derive(Debug, Clone)]
pub struct ServiceEntry {
    /// Service name.
    pub name: String,
    /// Endpoints.
    pub endpoints: Vec<Endpoint>,
    /// Retry policy.
    pub retry_policy: RetryPolicy,
    /// Per-endpoint circuit breakers.
    pub circuit_breakers: HashMap<String, CircuitBreaker>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// Required version compatibility.
    pub required_version: Option<ServiceVersion>,
}

impl ServiceEntry {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            endpoints: Vec::new(),
            retry_policy: RetryPolicy::default(),
            circuit_breakers: HashMap::new(),
            metadata: HashMap::new(),
            required_version: None,
        }
    }

    /// Add an endpoint.
    pub fn add_endpoint(&mut self, endpoint: Endpoint) {
        let id = endpoint.id.clone();
        self.endpoints.push(endpoint);
        self.circuit_breakers.entry(id).or_insert_with(|| CircuitBreaker::new(5, 2));
    }

    /// Remove an endpoint by ID.
    pub fn remove_endpoint(&mut self, id: &str) -> bool {
        let initial = self.endpoints.len();
        self.endpoints.retain(|e| e.id != id);
        self.circuit_breakers.remove(id);
        self.endpoints.len() < initial
    }

    /// Get endpoint by ID.
    pub fn endpoint(&self, id: &str) -> Option<&Endpoint> {
        self.endpoints.iter().find(|e| e.id == id)
    }

    /// Get mutable endpoint by ID.
    pub fn endpoint_mut(&mut self, id: &str) -> Option<&mut Endpoint> {
        self.endpoints.iter_mut().find(|e| e.id == id)
    }

    /// Available endpoints (healthy/degraded and circuit not open).
    pub fn available_endpoints(&self, now_ms: u64) -> Vec<&Endpoint> {
        self.endpoints.iter()
            .filter(|e| {
                if !e.is_available() {
                    return false;
                }
                if let Some(cb) = self.circuit_breakers.get(&e.id) {
                    cb.allows_request(now_ms)
                } else {
                    true
                }
            })
            .collect()
    }

    /// Set the retry policy.
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Set required version.
    pub fn with_required_version(mut self, version: ServiceVersion) -> Self {
        self.required_version = Some(version);
        self
    }

    /// Endpoints matching a required version.
    pub fn version_compatible_endpoints(&self) -> Vec<&Endpoint> {
        match &self.required_version {
            None => self.endpoints.iter().collect(),
            Some(required) => {
                self.endpoints.iter()
                    .filter(|e| {
                        e.version.as_ref().map(|v| v.is_compatible(required)).unwrap_or(false)
                    })
                    .collect()
            }
        }
    }

    /// Update heartbeat for an endpoint.
    pub fn heartbeat(&mut self, endpoint_id: &str, now_ms: u64) -> bool {
        if let Some(ep) = self.endpoint_mut(endpoint_id) {
            ep.last_heartbeat_ms = now_ms;
            if ep.health == HealthStatus::Unknown {
                ep.health = HealthStatus::Healthy;
            }
            true
        } else {
            false
        }
    }

    /// Mark endpoints as unhealthy if heartbeat is stale.
    pub fn check_stale_heartbeats(&mut self, now_ms: u64, ttl_ms: u64) {
        for ep in &mut self.endpoints {
            if ep.last_heartbeat_ms > 0 && now_ms > ep.last_heartbeat_ms + ttl_ms {
                ep.health = HealthStatus::Unhealthy;
            }
        }
    }

    /// Count of healthy endpoints.
    pub fn healthy_count(&self) -> usize {
        self.endpoints.iter().filter(|e| e.health == HealthStatus::Healthy).count()
    }

    /// Count of all endpoints.
    pub fn endpoint_count(&self) -> usize {
        self.endpoints.len()
    }
}

// ── Service Registry ─────────────────────────────────────────

/// Central service registry for discovery.
#[derive(Debug, Default)]
pub struct ServiceRegistry {
    services: HashMap<String, ServiceEntry>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self { services: HashMap::new() }
    }

    /// Register a service.
    pub fn register(&mut self, entry: ServiceEntry) {
        self.services.insert(entry.name.clone(), entry);
    }

    /// Unregister a service.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.services.remove(name).is_some()
    }

    /// Get a service by name.
    pub fn get(&self, name: &str) -> Option<&ServiceEntry> {
        self.services.get(name)
    }

    /// Get a mutable service.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ServiceEntry> {
        self.services.get_mut(name)
    }

    /// All registered service names (sorted).
    pub fn service_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.services.keys().cloned().collect();
        names.sort();
        names
    }

    /// Total endpoints across all services.
    pub fn total_endpoints(&self) -> usize {
        self.services.values().map(|s| s.endpoint_count()).sum()
    }

    /// Total healthy endpoints across all services.
    pub fn total_healthy(&self) -> usize {
        self.services.values().map(|s| s.healthy_count()).sum()
    }

    /// Number of registered services.
    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    /// Discover: find a service and select an endpoint via load balancer.
    pub fn discover(
        &self,
        name: &str,
        balancer: &mut LoadBalancer,
        now_ms: u64,
    ) -> Option<String> {
        let svc = self.services.get(name)?;
        let available = svc.available_endpoints(now_ms);
        if available.is_empty() {
            return None;
        }
        // Build a temporary vec of available endpoints for the balancer.
        // We need indices into the service's endpoints vec.
        let available_ids: Vec<&str> = available.iter().map(|e| e.id.as_str()).collect();
        let available_eps: Vec<Endpoint> = available.into_iter().cloned().collect();
        let idx = balancer.select(&available_eps)?;
        if idx < available_ids.len() {
            Some(available_eps[idx].address())
        } else {
            None
        }
    }
}

// ── Routing Rule ─────────────────────────────────────────────

/// A routing rule for service mesh routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingRule {
    /// Rule name.
    pub name: String,
    /// Source service (empty = any).
    pub source_service: String,
    /// Destination service.
    pub destination_service: String,
    /// Required tag match on endpoint.
    pub required_tags: HashMap<String, String>,
    /// Required version.
    pub required_version: Option<ServiceVersion>,
    /// Weight (for traffic splitting).
    pub weight: u32,
    /// Priority (lower = higher priority).
    pub priority: u32,
}

impl RoutingRule {
    pub fn new(
        name: impl Into<String>,
        destination: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            source_service: String::new(),
            destination_service: destination.into(),
            required_tags: HashMap::new(),
            required_version: None,
            weight: 100,
            priority: 100,
        }
    }

    /// Set source service filter.
    pub fn from_source(mut self, source: impl Into<String>) -> Self {
        self.source_service = source.into();
        self
    }

    /// Require a tag.
    pub fn require_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.required_tags.insert(key.into(), value.into());
        self
    }

    /// Require a version.
    pub fn require_version(mut self, version: ServiceVersion) -> Self {
        self.required_version = Some(version);
        self
    }

    /// Set weight.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Set priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Check if an endpoint matches this rule.
    pub fn matches_endpoint(&self, endpoint: &Endpoint) -> bool {
        // Check tags.
        for (k, v) in &self.required_tags {
            match endpoint.tags.get(k) {
                Some(ev) if ev == v => {}
                _ => return false,
            }
        }
        // Check version.
        if let Some(req_ver) = &self.required_version {
            match &endpoint.version {
                Some(ep_ver) if ep_ver.is_compatible(req_ver) => {}
                _ => return false,
            }
        }
        true
    }
}

/// Service mesh router with routing rules.
#[derive(Debug, Default)]
pub struct MeshRouter {
    rules: Vec<RoutingRule>,
}

impl MeshRouter {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a routing rule.
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
        // Sort by priority (lower = higher priority).
        self.rules.sort_by_key(|r| r.priority);
    }

    /// Find matching rules for a destination.
    pub fn find_rules(&self, destination: &str, source: &str) -> Vec<&RoutingRule> {
        self.rules.iter()
            .filter(|r| {
                r.destination_service == destination
                    && (r.source_service.is_empty() || r.source_service == source)
            })
            .collect()
    }

    /// Filter endpoints by routing rules.
    pub fn filter_endpoints<'a>(
        &self,
        destination: &str,
        source: &str,
        endpoints: &'a [Endpoint],
    ) -> Vec<&'a Endpoint> {
        let rules = self.find_rules(destination, source);
        if rules.is_empty() {
            // No rules — return all available.
            return endpoints.iter().filter(|e| e.is_available()).collect();
        }
        // Apply the highest-priority rule.
        let rule = rules[0];
        endpoints.iter()
            .filter(|e| e.is_available() && rule.matches_endpoint(e))
            .collect()
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_endpoint(id: &str, port: u16, health: HealthStatus) -> Endpoint {
        Endpoint::new(id, "127.0.0.1", port).with_health(health)
    }

    #[test]
    fn health_status_availability() {
        assert!(HealthStatus::Healthy.is_available());
        assert!(HealthStatus::Degraded.is_available());
        assert!(!HealthStatus::Unhealthy.is_available());
        assert!(!HealthStatus::Unknown.is_available());
    }

    #[test]
    fn health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Unknown.to_string(), "unknown");
    }

    #[test]
    fn service_version_parse() {
        let v = ServiceVersion::parse("1.2.3").unwrap();
        assert_eq!(v, ServiceVersion::new(1, 2, 3));
        assert_eq!(v.to_string(), "1.2.3");
        assert!(ServiceVersion::parse("bad").is_none());
        assert!(ServiceVersion::parse("1.2").is_none());
    }

    #[test]
    fn service_version_compatibility() {
        let v1 = ServiceVersion::new(1, 0, 0);
        let v2 = ServiceVersion::new(1, 5, 3);
        let v3 = ServiceVersion::new(2, 0, 0);
        assert!(v1.is_compatible(&v2));
        assert!(!v1.is_compatible(&v3));
    }

    #[test]
    fn service_version_ordering() {
        let v1 = ServiceVersion::new(1, 0, 0);
        let v2 = ServiceVersion::new(1, 1, 0);
        let v3 = ServiceVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn endpoint_address() {
        let ep = Endpoint::new("ep1", "10.0.0.1", 8080);
        assert_eq!(ep.address(), "10.0.0.1:8080");
    }

    #[test]
    fn endpoint_builder() {
        let ep = Endpoint::new("ep1", "host", 80)
            .with_health(HealthStatus::Healthy)
            .with_weight(50)
            .with_version(ServiceVersion::new(1, 0, 0))
            .with_tag("region", "us-east");
        assert!(ep.is_available());
        assert_eq!(ep.weight, 50);
        assert_eq!(ep.tags.get("region"), Some(&"us-east".to_string()));
    }

    #[test]
    fn endpoint_weight_clamped() {
        let ep = Endpoint::new("ep1", "host", 80).with_weight(0);
        assert_eq!(ep.weight, 1);
        let ep2 = Endpoint::new("ep1", "host", 80).with_weight(200);
        assert_eq!(ep2.weight, 100);
    }

    #[test]
    fn retry_policy_backoff() {
        let policy = RetryPolicy::new(5);
        assert_eq!(policy.backoff_ms(0), 0);
        assert_eq!(policy.backoff_ms(1), 100);
        assert_eq!(policy.backoff_ms(2), 200);
        assert_eq!(policy.backoff_ms(3), 400);
    }

    #[test]
    fn retry_policy_max_backoff() {
        let mut policy = RetryPolicy::new(10);
        policy.max_backoff_ms = 500;
        assert!(policy.backoff_ms(10) <= 500);
    }

    #[test]
    fn retry_policy_retryable() {
        let policy = RetryPolicy::default();
        assert!(policy.is_retryable(14)); // Unavailable
        assert!(!policy.is_retryable(5)); // NotFound
    }

    #[test]
    fn circuit_breaker_closed_to_open() {
        let mut cb = CircuitBreaker::new(3, 2);
        assert_eq!(cb.state, CircuitState::Closed);
        cb.record_failure(1000);
        cb.record_failure(1001);
        assert_eq!(cb.state, CircuitState::Closed);
        cb.record_failure(1002);
        assert_eq!(cb.state, CircuitState::Open);
    }

    #[test]
    fn circuit_breaker_open_to_half_open() {
        let mut cb = CircuitBreaker::new(1, 1).with_open_duration(5000);
        cb.record_failure(1000);
        assert_eq!(cb.state, CircuitState::Open);
        assert!(!cb.allows_request(3000));
        cb.check_transition(6001);
        assert_eq!(cb.state, CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_half_open_to_closed() {
        let mut cb = CircuitBreaker::new(1, 2);
        cb.state = CircuitState::HalfOpen;
        cb.record_success();
        assert_eq!(cb.state, CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state, CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_half_open_failure() {
        let mut cb = CircuitBreaker::new(1, 2);
        cb.state = CircuitState::HalfOpen;
        cb.record_failure(2000);
        assert_eq!(cb.state, CircuitState::Open);
    }

    #[test]
    fn circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(1, 1);
        cb.record_failure(1000);
        assert_eq!(cb.state, CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state, CircuitState::Closed);
        assert_eq!(cb.failure_count, 0);
    }

    #[test]
    fn load_balancer_round_robin() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        let eps = vec![
            make_endpoint("a", 80, HealthStatus::Healthy),
            make_endpoint("b", 81, HealthStatus::Healthy),
            make_endpoint("c", 82, HealthStatus::Healthy),
        ];
        assert_eq!(lb.select(&eps), Some(0));
        assert_eq!(lb.select(&eps), Some(1));
        assert_eq!(lb.select(&eps), Some(2));
        assert_eq!(lb.select(&eps), Some(0)); // wraps
    }

    #[test]
    fn load_balancer_skips_unhealthy() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        let eps = vec![
            make_endpoint("a", 80, HealthStatus::Unhealthy),
            make_endpoint("b", 81, HealthStatus::Healthy),
            make_endpoint("c", 82, HealthStatus::Healthy),
        ];
        // Only b and c are available.
        let selected = lb.select(&eps).unwrap();
        assert!(selected == 1 || selected == 2);
    }

    #[test]
    fn load_balancer_no_available() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        let eps = vec![
            make_endpoint("a", 80, HealthStatus::Unhealthy),
        ];
        assert_eq!(lb.select(&eps), None);
    }

    #[test]
    fn load_balancer_least_connections() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::LeastConnections);
        let mut eps = vec![
            make_endpoint("a", 80, HealthStatus::Healthy),
            make_endpoint("b", 81, HealthStatus::Healthy),
        ];
        eps[0].active_connections = 10;
        eps[1].active_connections = 2;
        assert_eq!(lb.select(&eps), Some(1)); // b has fewer
    }

    #[test]
    fn service_entry_add_remove() {
        let mut svc = ServiceEntry::new("api");
        svc.add_endpoint(make_endpoint("ep1", 80, HealthStatus::Healthy));
        svc.add_endpoint(make_endpoint("ep2", 81, HealthStatus::Healthy));
        assert_eq!(svc.endpoint_count(), 2);

        assert!(svc.remove_endpoint("ep1"));
        assert_eq!(svc.endpoint_count(), 1);
        assert!(!svc.remove_endpoint("missing"));
    }

    #[test]
    fn service_entry_heartbeat() {
        let mut svc = ServiceEntry::new("api");
        svc.add_endpoint(make_endpoint("ep1", 80, HealthStatus::Unknown));
        assert!(svc.heartbeat("ep1", 1000));
        assert_eq!(svc.endpoint("ep1").unwrap().health, HealthStatus::Healthy);
        assert_eq!(svc.endpoint("ep1").unwrap().last_heartbeat_ms, 1000);
    }

    #[test]
    fn service_entry_stale_heartbeat() {
        let mut svc = ServiceEntry::new("api");
        let mut ep = make_endpoint("ep1", 80, HealthStatus::Healthy);
        ep.last_heartbeat_ms = 1000;
        svc.add_endpoint(ep);
        svc.check_stale_heartbeats(6000, 3000);
        assert_eq!(svc.endpoint("ep1").unwrap().health, HealthStatus::Unhealthy);
    }

    #[test]
    fn service_entry_version_compat() {
        let svc = ServiceEntry::new("api")
            .with_required_version(ServiceVersion::new(2, 0, 0));
        let mut entry = svc;
        entry.add_endpoint(
            make_endpoint("ep1", 80, HealthStatus::Healthy)
                .with_version(ServiceVersion::new(2, 1, 0)),
        );
        entry.add_endpoint(
            make_endpoint("ep2", 81, HealthStatus::Healthy)
                .with_version(ServiceVersion::new(1, 9, 0)),
        );
        let compat = entry.version_compatible_endpoints();
        assert_eq!(compat.len(), 1);
        assert_eq!(compat[0].id, "ep1");
    }

    #[test]
    fn service_registry_basic() {
        let mut reg = ServiceRegistry::new();
        let mut svc = ServiceEntry::new("api");
        svc.add_endpoint(make_endpoint("ep1", 80, HealthStatus::Healthy));
        reg.register(svc);

        assert_eq!(reg.service_count(), 1);
        assert_eq!(reg.total_endpoints(), 1);
        assert_eq!(reg.total_healthy(), 1);
        assert!(reg.get("api").is_some());
    }

    #[test]
    fn service_registry_unregister() {
        let mut reg = ServiceRegistry::new();
        reg.register(ServiceEntry::new("api"));
        assert!(reg.unregister("api"));
        assert!(!reg.unregister("api"));
        assert_eq!(reg.service_count(), 0);
    }

    #[test]
    fn service_registry_discover() {
        let mut reg = ServiceRegistry::new();
        let mut svc = ServiceEntry::new("api");
        svc.add_endpoint(make_endpoint("ep1", 8080, HealthStatus::Healthy));
        reg.register(svc);

        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        let addr = reg.discover("api", &mut lb, 0).unwrap();
        assert_eq!(addr, "127.0.0.1:8080");
    }

    #[test]
    fn service_registry_discover_none() {
        let reg = ServiceRegistry::new();
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        assert!(reg.discover("missing", &mut lb, 0).is_none());
    }

    #[test]
    fn routing_rule_matches() {
        let rule = RoutingRule::new("canary", "api")
            .require_tag("env", "canary")
            .require_version(ServiceVersion::new(2, 0, 0));

        let ep1 = Endpoint::new("ep1", "host", 80)
            .with_health(HealthStatus::Healthy)
            .with_tag("env", "canary")
            .with_version(ServiceVersion::new(2, 1, 0));
        assert!(rule.matches_endpoint(&ep1));

        let ep2 = Endpoint::new("ep2", "host", 80)
            .with_health(HealthStatus::Healthy)
            .with_tag("env", "prod")
            .with_version(ServiceVersion::new(2, 0, 0));
        assert!(!rule.matches_endpoint(&ep2));
    }

    #[test]
    fn mesh_router_filter() {
        let mut router = MeshRouter::new();
        router.add_rule(
            RoutingRule::new("canary", "api")
                .require_tag("env", "canary")
                .with_priority(10),
        );

        let eps = vec![
            Endpoint::new("ep1", "host", 80)
                .with_health(HealthStatus::Healthy)
                .with_tag("env", "canary"),
            Endpoint::new("ep2", "host", 81)
                .with_health(HealthStatus::Healthy)
                .with_tag("env", "prod"),
        ];

        let filtered = router.filter_endpoints("api", "", &eps);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, "ep1");
    }

    #[test]
    fn mesh_router_no_rules_returns_all_available() {
        let router = MeshRouter::new();
        let eps = vec![
            make_endpoint("a", 80, HealthStatus::Healthy),
            make_endpoint("b", 81, HealthStatus::Unhealthy),
        ];
        let filtered = router.filter_endpoints("api", "", &eps);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn mesh_router_source_filter() {
        let mut router = MeshRouter::new();
        router.add_rule(
            RoutingRule::new("r1", "api").from_source("frontend"),
        );
        let rules = router.find_rules("api", "frontend");
        assert_eq!(rules.len(), 1);
        let rules = router.find_rules("api", "backend");
        assert_eq!(rules.len(), 0);
    }

    #[test]
    fn circuit_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "closed");
        assert_eq!(CircuitState::Open.to_string(), "open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "half-open");
    }

    #[test]
    fn load_balancer_reset() {
        let mut lb = LoadBalancer::new(LoadBalanceStrategy::RoundRobin);
        let eps = vec![make_endpoint("a", 80, HealthStatus::Healthy)];
        lb.select(&eps);
        lb.select(&eps);
        lb.reset();
        assert_eq!(lb.strategy(), LoadBalanceStrategy::RoundRobin);
    }

    #[test]
    fn service_entry_healthy_count() {
        let mut svc = ServiceEntry::new("api");
        svc.add_endpoint(make_endpoint("a", 80, HealthStatus::Healthy));
        svc.add_endpoint(make_endpoint("b", 81, HealthStatus::Unhealthy));
        svc.add_endpoint(make_endpoint("c", 82, HealthStatus::Healthy));
        assert_eq!(svc.healthy_count(), 2);
    }

    #[test]
    fn registry_service_names_sorted() {
        let mut reg = ServiceRegistry::new();
        reg.register(ServiceEntry::new("zeta"));
        reg.register(ServiceEntry::new("alpha"));
        reg.register(ServiceEntry::new("mid"));
        let names = reg.service_names();
        assert_eq!(names, vec!["alpha", "mid", "zeta"]);
    }
}
