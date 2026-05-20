//! Service mesh concepts — sidecar proxy modeling, mTLS configuration, traffic policies,
//! retry/timeout/circuit-breaker per service, and service identity management.
//!
//! Replaces `istio-client`, `linkerd-config`, and similar JS mesh SDKs with a
//! pure-Rust service mesh policy engine supporting traffic management and resilience.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Service mesh errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// Service not found.
    ServiceNotFound(String),
    /// Duplicate service identity.
    DuplicateService(String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Circuit breaker is open.
    CircuitOpen { service: String, failures: u32 },
    /// Retry budget exhausted.
    RetryExhausted { service: String, attempts: u32 },
    /// mTLS handshake failure.
    MtlsFailure(String),
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServiceNotFound(s) => write!(f, "service not found: {s}"),
            Self::DuplicateService(s) => write!(f, "duplicate service: {s}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::CircuitOpen { service, failures } => {
                write!(f, "circuit open for {service}: {failures} failures")
            }
            Self::RetryExhausted { service, attempts } => {
                write!(f, "retries exhausted for {service}: {attempts} attempts")
            }
            Self::MtlsFailure(msg) => write!(f, "mTLS failure: {msg}"),
        }
    }
}

impl std::error::Error for MeshError {}

// ── Service Identity ───────────────────────────────────────────

/// A service identity in the mesh with SPIFFE-style naming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServiceIdentity {
    /// Namespace (e.g. "production", "staging").
    pub namespace: String,
    /// Service name.
    pub name: String,
    /// Version label.
    pub version: String,
    /// Labels for routing and policy.
    pub labels: HashMap<String, String>,
}

impl std::hash::Hash for ServiceIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.name.hash(state);
        self.version.hash(state);
        // Sort labels for deterministic hashing.
        let mut pairs: Vec<_> = self.labels.iter().collect();
        pairs.sort();
        for (k, v) in pairs {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl ServiceIdentity {
    /// Create a new identity.
    pub fn new(namespace: &str, name: &str, version: &str) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
            version: version.into(),
            labels: HashMap::new(),
        }
    }

    /// Add a label.
    pub fn with_label(mut self, key: &str, value: &str) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// SPIFFE-style URI.
    pub fn spiffe_id(&self) -> String {
        format!("spiffe://mesh/{}/{}/{}", self.namespace, self.name, self.version)
    }
}

impl fmt::Display for ServiceIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.namespace, self.name, self.version)
    }
}

// ── mTLS Configuration ────────────────────────────────────────

/// mTLS mode for service communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MtlsMode {
    /// No TLS (plaintext).
    Disabled,
    /// TLS is optional (accept both).
    Permissive,
    /// TLS is required.
    Strict,
}

/// mTLS configuration for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtlsConfig {
    pub mode: MtlsMode,
    /// Certificate fingerprint (hex string).
    pub cert_fingerprint: Option<String>,
    /// Allowed peer identities (SPIFFE IDs).
    pub allowed_peers: Vec<String>,
    /// Minimum TLS version (e.g. "1.2", "1.3").
    pub min_tls_version: String,
}

impl Default for MtlsConfig {
    fn default() -> Self {
        Self {
            mode: MtlsMode::Permissive,
            cert_fingerprint: None,
            allowed_peers: Vec::new(),
            min_tls_version: "1.2".into(),
        }
    }
}

impl MtlsConfig {
    /// Check if a peer identity is allowed.
    pub fn is_peer_allowed(&self, peer_spiffe: &str) -> bool {
        if self.allowed_peers.is_empty() {
            return true; // No allowlist means all peers allowed
        }
        self.allowed_peers.iter().any(|p| p == peer_spiffe)
    }
}

// ── Retry Policy ───────────────────────────────────────────────

/// Retry policy for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Base delay between retries in milliseconds.
    pub base_delay_ms: u64,
    /// Maximum delay (exponential backoff cap) in milliseconds.
    pub max_delay_ms: u64,
    /// Which status codes trigger a retry.
    pub retryable_status_codes: Vec<u16>,
    /// Whether to retry on connection failure.
    pub retry_on_connect_failure: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 10_000,
            retryable_status_codes: vec![502, 503, 504],
            retry_on_connect_failure: true,
        }
    }
}

impl RetryPolicy {
    /// Calculate delay for a given attempt (exponential backoff).
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            return 0;
        }
        let exp = 1u64.checked_shl(attempt.saturating_sub(1)).unwrap_or(u64::MAX);
        let delay = self.base_delay_ms.saturating_mul(exp);
        delay.min(self.max_delay_ms)
    }

    /// Check if a status code is retryable.
    pub fn is_retryable_status(&self, code: u16) -> bool {
        self.retryable_status_codes.contains(&code)
    }
}

// ── Timeout Policy ─────────────────────────────────────────────

/// Timeout policy for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutPolicy {
    /// Overall request timeout in milliseconds.
    pub request_timeout_ms: u64,
    /// Timeout for establishing a connection in milliseconds.
    pub connect_timeout_ms: u64,
    /// Idle timeout in milliseconds.
    pub idle_timeout_ms: u64,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self {
            request_timeout_ms: 30_000,
            connect_timeout_ms: 5_000,
            idle_timeout_ms: 60_000,
        }
    }
}

// ── Circuit Breaker ────────────────────────────────────────────

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening the circuit.
    pub failure_threshold: u32,
    /// Duration in milliseconds the circuit stays open before going half-open.
    pub open_duration_ms: u64,
    /// Number of successes in half-open state to close the circuit.
    pub half_open_successes: u32,
    /// Maximum concurrent requests (0 = unlimited).
    pub max_concurrent: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            open_duration_ms: 30_000,
            half_open_successes: 3,
            max_concurrent: 0,
        }
    }
}

/// Runtime circuit breaker state for a service.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub config: CircuitBreakerConfig,
    pub state: CircuitState,
    pub consecutive_failures: u32,
    pub consecutive_successes: u32,
    pub opened_at_ms: u64,
    pub total_failures: u64,
    pub total_successes: u64,
}

impl CircuitBreaker {
    /// Create from config.
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: CircuitState::Closed,
            consecutive_failures: 0,
            consecutive_successes: 0,
            opened_at_ms: 0,
            total_failures: 0,
            total_successes: 0,
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        self.total_successes += 1;
        self.consecutive_failures = 0;
        self.consecutive_successes += 1;

        if self.state == CircuitState::HalfOpen
            && self.consecutive_successes >= self.config.half_open_successes
        {
            self.state = CircuitState::Closed;
            self.consecutive_successes = 0;
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self, now_ms: u64) {
        self.total_failures += 1;
        self.consecutive_successes = 0;
        self.consecutive_failures += 1;

        if self.consecutive_failures >= self.config.failure_threshold {
            self.state = CircuitState::Open;
            self.opened_at_ms = now_ms;
        }
    }

    /// Check if a request is allowed.
    pub fn allow_request(&mut self, now_ms: u64) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if now_ms.saturating_sub(self.opened_at_ms) >= self.config.open_duration_ms {
                    self.state = CircuitState::HalfOpen;
                    self.consecutive_successes = 0;
                    self.consecutive_failures = 0;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
}

// ── Traffic Policy ─────────────────────────────────────────────

/// Traffic policy for a service destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficPolicy {
    /// Traffic weight (0-100) for canary/split routing.
    pub weight_percent: u8,
    /// Header match conditions for routing.
    pub header_matches: HashMap<String, String>,
    /// Whether to mirror traffic to this destination.
    pub mirror: bool,
    /// Mirror percentage (0-100).
    pub mirror_percent: u8,
}

impl Default for TrafficPolicy {
    fn default() -> Self {
        Self {
            weight_percent: 100,
            header_matches: HashMap::new(),
            mirror: false,
            mirror_percent: 0,
        }
    }
}

// ── Sidecar Proxy ──────────────────────────────────────────────

/// Sidecar proxy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarConfig {
    /// Port the sidecar listens on for inbound traffic.
    pub inbound_port: u16,
    /// Port the sidecar listens on for outbound traffic.
    pub outbound_port: u16,
    /// Ports to bypass the sidecar (direct passthrough).
    pub bypass_ports: Vec<u16>,
    /// Whether to capture all outbound traffic.
    pub capture_outbound: bool,
    /// Access log format.
    pub access_log_format: String,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            inbound_port: 15006,
            outbound_port: 15001,
            bypass_ports: Vec::new(),
            capture_outbound: true,
            access_log_format: "json".into(),
        }
    }
}

impl SidecarConfig {
    /// Check if a port should bypass the sidecar.
    pub fn should_bypass(&self, port: u16) -> bool {
        self.bypass_ports.contains(&port)
    }
}

// ── Service Entry ──────────────────────────────────────────────

/// Complete service entry in the mesh registry.
#[derive(Debug, Clone)]
pub struct ServiceEntry {
    pub identity: ServiceIdentity,
    pub mtls: MtlsConfig,
    pub retry: RetryPolicy,
    pub timeout: TimeoutPolicy,
    pub circuit_breaker: CircuitBreaker,
    pub traffic_policy: TrafficPolicy,
    pub sidecar: SidecarConfig,
    /// Endpoints (addresses) for this service.
    pub endpoints: Vec<String>,
}

// ── Service Mesh ───────────────────────────────────────────────

/// The service mesh registry and policy engine.
#[derive(Debug)]
pub struct ServiceMesh {
    services: HashMap<String, ServiceEntry>,
}

impl ServiceMesh {
    /// Create a new empty mesh.
    pub fn new() -> Self {
        Self { services: HashMap::new() }
    }

    /// Register a service with default policies.
    pub fn register(
        &mut self,
        identity: ServiceIdentity,
        endpoints: Vec<String>,
    ) -> Result<(), MeshError> {
        let key = identity.to_string();
        if self.services.contains_key(&key) {
            return Err(MeshError::DuplicateService(key));
        }
        self.services.insert(key, ServiceEntry {
            identity,
            mtls: MtlsConfig::default(),
            retry: RetryPolicy::default(),
            timeout: TimeoutPolicy::default(),
            circuit_breaker: CircuitBreaker::new(CircuitBreakerConfig::default()),
            traffic_policy: TrafficPolicy::default(),
            sidecar: SidecarConfig::default(),
            endpoints,
        });
        Ok(())
    }

    /// Get a service by its key (namespace/name/version).
    pub fn get(&self, key: &str) -> Result<&ServiceEntry, MeshError> {
        self.services
            .get(key)
            .ok_or_else(|| MeshError::ServiceNotFound(key.into()))
    }

    /// Get mutable reference to a service.
    pub fn get_mut(&mut self, key: &str) -> Result<&mut ServiceEntry, MeshError> {
        self.services
            .get_mut(key)
            .ok_or_else(|| MeshError::ServiceNotFound(key.into()))
    }

    /// Set the mTLS config for a service.
    pub fn set_mtls(
        &mut self,
        service_key: &str,
        config: MtlsConfig,
    ) -> Result<(), MeshError> {
        self.get_mut(service_key)?.mtls = config;
        Ok(())
    }

    /// Set the retry policy for a service.
    pub fn set_retry(
        &mut self,
        service_key: &str,
        policy: RetryPolicy,
    ) -> Result<(), MeshError> {
        self.get_mut(service_key)?.retry = policy;
        Ok(())
    }

    /// Set the timeout policy for a service.
    pub fn set_timeout(
        &mut self,
        service_key: &str,
        policy: TimeoutPolicy,
    ) -> Result<(), MeshError> {
        self.get_mut(service_key)?.timeout = policy;
        Ok(())
    }

    /// Set the circuit breaker config for a service.
    pub fn set_circuit_breaker(
        &mut self,
        service_key: &str,
        config: CircuitBreakerConfig,
    ) -> Result<(), MeshError> {
        self.get_mut(service_key)?.circuit_breaker = CircuitBreaker::new(config);
        Ok(())
    }

    /// Check if a request to a service is allowed (circuit breaker check).
    pub fn allow_request(
        &mut self,
        service_key: &str,
        now_ms: u64,
    ) -> Result<bool, MeshError> {
        let entry = self.get_mut(service_key)?;
        Ok(entry.circuit_breaker.allow_request(now_ms))
    }

    /// Record a success for a service's circuit breaker.
    pub fn record_success(&mut self, service_key: &str) -> Result<(), MeshError> {
        self.get_mut(service_key)?.circuit_breaker.record_success();
        Ok(())
    }

    /// Record a failure for a service's circuit breaker.
    pub fn record_failure(
        &mut self,
        service_key: &str,
        now_ms: u64,
    ) -> Result<(), MeshError> {
        self.get_mut(service_key)?.circuit_breaker.record_failure(now_ms);
        Ok(())
    }

    /// Validate mTLS peer authorization.
    pub fn authorize_peer(
        &self,
        service_key: &str,
        peer_spiffe: &str,
    ) -> Result<bool, MeshError> {
        let entry = self.get(service_key)?;
        match entry.mtls.mode {
            MtlsMode::Disabled => Ok(true),
            MtlsMode::Permissive | MtlsMode::Strict => {
                Ok(entry.mtls.is_peer_allowed(peer_spiffe))
            }
        }
    }

    /// Remove a service from the mesh.
    pub fn deregister(&mut self, key: &str) -> Result<(), MeshError> {
        self.services
            .remove(key)
            .ok_or_else(|| MeshError::ServiceNotFound(key.into()))?;
        Ok(())
    }

    /// List all service keys.
    pub fn service_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.services.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Get services matching a label selector.
    pub fn select_by_label(&self, key: &str, value: &str) -> Vec<&ServiceEntry> {
        self.services
            .values()
            .filter(|e| e.identity.labels.get(key).map(|v| v == value).unwrap_or(false))
            .collect()
    }

    /// Total number of registered services.
    pub fn service_count(&self) -> usize {
        self.services.len()
    }
}

impl Default for ServiceMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(name: &str) -> ServiceIdentity {
        ServiceIdentity::new("prod", name, "v1")
    }

    fn mesh_with_service(name: &str) -> (ServiceMesh, String) {
        let mut mesh = ServiceMesh::new();
        let id = identity(name);
        let key = id.to_string();
        mesh.register(id, vec!["10.0.0.1:8080".into()]).unwrap();
        (mesh, key)
    }

    // ── Identity tests ──

    #[test]
    fn identity_display() {
        let id = identity("api");
        assert_eq!(id.to_string(), "prod/api/v1");
    }

    #[test]
    fn identity_spiffe_id() {
        let id = identity("api");
        assert_eq!(id.spiffe_id(), "spiffe://mesh/prod/api/v1");
    }

    #[test]
    fn identity_with_labels() {
        let id = identity("api").with_label("env", "prod").with_label("team", "platform");
        assert_eq!(id.labels.get("env").unwrap(), "prod");
        assert_eq!(id.labels.get("team").unwrap(), "platform");
    }

    // ── mTLS tests ──

    #[test]
    fn mtls_default_permissive() {
        let config = MtlsConfig::default();
        assert_eq!(config.mode, MtlsMode::Permissive);
    }

    #[test]
    fn mtls_empty_allowlist_permits_all() {
        let config = MtlsConfig::default();
        assert!(config.is_peer_allowed("spiffe://mesh/prod/any/v1"));
    }

    #[test]
    fn mtls_allowlist_filters() {
        let config = MtlsConfig {
            mode: MtlsMode::Strict,
            cert_fingerprint: None,
            allowed_peers: vec!["spiffe://mesh/prod/api/v1".into()],
            min_tls_version: "1.3".into(),
        };
        assert!(config.is_peer_allowed("spiffe://mesh/prod/api/v1"));
        assert!(!config.is_peer_allowed("spiffe://mesh/prod/evil/v1"));
    }

    // ── Retry policy tests ──

    #[test]
    fn retry_delay_exponential_backoff() {
        let policy = RetryPolicy {
            base_delay_ms: 100,
            max_delay_ms: 5_000,
            ..Default::default()
        };
        assert_eq!(policy.delay_for_attempt(0), 0);
        assert_eq!(policy.delay_for_attempt(1), 100);
        assert_eq!(policy.delay_for_attempt(2), 200);
        assert_eq!(policy.delay_for_attempt(3), 400);
        assert_eq!(policy.delay_for_attempt(10), 5_000); // Capped
    }

    #[test]
    fn retry_is_retryable_status() {
        let policy = RetryPolicy::default();
        assert!(policy.is_retryable_status(502));
        assert!(policy.is_retryable_status(503));
        assert!(!policy.is_retryable_status(200));
        assert!(!policy.is_retryable_status(404));
    }

    // ── Circuit breaker tests ──

    #[test]
    fn circuit_starts_closed() {
        let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
        assert_eq!(cb.state, CircuitState::Closed);
    }

    #[test]
    fn circuit_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        });
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.state, CircuitState::Closed);
        cb.record_failure(300);
        assert_eq!(cb.state, CircuitState::Open);
        assert_eq!(cb.opened_at_ms, 300);
    }

    #[test]
    fn circuit_half_open_after_duration() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration_ms: 1000,
            ..Default::default()
        });
        cb.record_failure(100);
        assert_eq!(cb.state, CircuitState::Open);

        assert!(!cb.allow_request(500)); // Still open
        assert!(cb.allow_request(1200)); // Now half-open
        assert_eq!(cb.state, CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_closes_after_half_open_successes() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 1,
            open_duration_ms: 100,
            half_open_successes: 2,
            ..Default::default()
        });
        cb.record_failure(0);
        cb.allow_request(200); // Transition to half-open

        cb.record_success();
        assert_eq!(cb.state, CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state, CircuitState::Closed);
    }

    #[test]
    fn circuit_success_resets_failures() {
        let mut cb = CircuitBreaker::new(CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        });
        cb.record_failure(100);
        cb.record_failure(200);
        cb.record_success();
        assert_eq!(cb.consecutive_failures, 0);
        // Should still be closed since threshold wasn't reached
        assert_eq!(cb.state, CircuitState::Closed);
    }

    // ── Sidecar tests ──

    #[test]
    fn sidecar_default_ports() {
        let sc = SidecarConfig::default();
        assert_eq!(sc.inbound_port, 15006);
        assert_eq!(sc.outbound_port, 15001);
    }

    #[test]
    fn sidecar_bypass_ports() {
        let sc = SidecarConfig {
            bypass_ports: vec![22, 443],
            ..Default::default()
        };
        assert!(sc.should_bypass(22));
        assert!(sc.should_bypass(443));
        assert!(!sc.should_bypass(8080));
    }

    // ── Mesh registration tests ──

    #[test]
    fn register_and_get_service() {
        let (mesh, key) = mesh_with_service("api");
        let entry = mesh.get(&key).unwrap();
        assert_eq!(entry.identity.name, "api");
        assert_eq!(entry.endpoints.len(), 1);
    }

    #[test]
    fn duplicate_registration_error() {
        let (mut mesh, _) = mesh_with_service("api");
        let err = mesh
            .register(identity("api"), vec!["10.0.0.2:8080".into()])
            .unwrap_err();
        assert!(matches!(err, MeshError::DuplicateService(_)));
    }

    #[test]
    fn service_not_found_error() {
        let mesh = ServiceMesh::new();
        assert!(matches!(mesh.get("nope"), Err(MeshError::ServiceNotFound(_))));
    }

    #[test]
    fn deregister_service() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.deregister(&key).unwrap();
        assert_eq!(mesh.service_count(), 0);
    }

    #[test]
    fn deregister_not_found() {
        let mut mesh = ServiceMesh::new();
        assert!(matches!(mesh.deregister("nope"), Err(MeshError::ServiceNotFound(_))));
    }

    // ── Policy configuration tests ──

    #[test]
    fn set_mtls_config() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_mtls(&key, MtlsConfig {
            mode: MtlsMode::Strict,
            cert_fingerprint: Some("abcdef".into()),
            allowed_peers: vec!["spiffe://mesh/prod/web/v1".into()],
            min_tls_version: "1.3".into(),
        })
        .unwrap();

        let entry = mesh.get(&key).unwrap();
        assert_eq!(entry.mtls.mode, MtlsMode::Strict);
    }

    #[test]
    fn set_retry_policy() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_retry(&key, RetryPolicy {
            max_retries: 5,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(mesh.get(&key).unwrap().retry.max_retries, 5);
    }

    #[test]
    fn set_timeout_policy() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_timeout(&key, TimeoutPolicy {
            request_timeout_ms: 60_000,
            connect_timeout_ms: 10_000,
            idle_timeout_ms: 120_000,
        })
        .unwrap();
        assert_eq!(mesh.get(&key).unwrap().timeout.request_timeout_ms, 60_000);
    }

    #[test]
    fn set_circuit_breaker_config() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_circuit_breaker(&key, CircuitBreakerConfig {
            failure_threshold: 10,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(mesh.get(&key).unwrap().circuit_breaker.config.failure_threshold, 10);
    }

    // ── Circuit breaker via mesh ──

    #[test]
    fn mesh_circuit_breaker_integration() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_circuit_breaker(&key, CircuitBreakerConfig {
            failure_threshold: 2,
            open_duration_ms: 1000,
            ..Default::default()
        })
        .unwrap();

        assert!(mesh.allow_request(&key, 0).unwrap());
        mesh.record_failure(&key, 100).unwrap();
        mesh.record_failure(&key, 200).unwrap();
        assert!(!mesh.allow_request(&key, 300).unwrap());
        assert!(mesh.allow_request(&key, 1300).unwrap()); // Half-open
    }

    #[test]
    fn mesh_record_success() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.record_success(&key).unwrap();
        let entry = mesh.get(&key).unwrap();
        assert_eq!(entry.circuit_breaker.total_successes, 1);
    }

    // ── Peer authorization ──

    #[test]
    fn authorize_peer_disabled_mtls() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_mtls(&key, MtlsConfig {
            mode: MtlsMode::Disabled,
            ..Default::default()
        })
        .unwrap();
        assert!(mesh.authorize_peer(&key, "anyone").unwrap());
    }

    #[test]
    fn authorize_peer_strict_with_allowlist() {
        let (mut mesh, key) = mesh_with_service("api");
        mesh.set_mtls(&key, MtlsConfig {
            mode: MtlsMode::Strict,
            allowed_peers: vec!["spiffe://mesh/prod/web/v1".into()],
            ..Default::default()
        })
        .unwrap();

        assert!(mesh.authorize_peer(&key, "spiffe://mesh/prod/web/v1").unwrap());
        assert!(!mesh.authorize_peer(&key, "spiffe://mesh/prod/evil/v1").unwrap());
    }

    // ── Label selection ──

    #[test]
    fn select_by_label() {
        let mut mesh = ServiceMesh::new();
        mesh.register(
            identity("api").with_label("team", "platform"),
            vec!["10.0.0.1:80".into()],
        )
        .unwrap();
        mesh.register(
            ServiceIdentity::new("prod", "web", "v1").with_label("team", "frontend"),
            vec!["10.0.0.2:80".into()],
        )
        .unwrap();
        mesh.register(
            ServiceIdentity::new("prod", "worker", "v1").with_label("team", "platform"),
            vec!["10.0.0.3:80".into()],
        )
        .unwrap();

        let platform = mesh.select_by_label("team", "platform");
        assert_eq!(platform.len(), 2);
    }

    #[test]
    fn service_keys_sorted() {
        let mut mesh = ServiceMesh::new();
        mesh.register(identity("zeta"), vec![]).unwrap();
        mesh.register(identity("alpha"), vec![]).unwrap();
        let keys = mesh.service_keys();
        assert_eq!(keys[0], "prod/alpha/v1");
        assert_eq!(keys[1], "prod/zeta/v1");
    }

    // ── Error display ──

    #[test]
    fn error_display_coverage() {
        let errs = vec![
            MeshError::ServiceNotFound("svc".into()),
            MeshError::DuplicateService("svc".into()),
            MeshError::InvalidConfig("bad".into()),
            MeshError::CircuitOpen { service: "svc".into(), failures: 5 },
            MeshError::RetryExhausted { service: "svc".into(), attempts: 3 },
            MeshError::MtlsFailure("cert".into()),
        ];
        for e in &errs {
            assert!(!e.to_string().is_empty());
        }
    }

    // ── Traffic policy ──

    #[test]
    fn traffic_policy_default() {
        let tp = TrafficPolicy::default();
        assert_eq!(tp.weight_percent, 100);
        assert!(!tp.mirror);
    }

    // ── Timeout policy ──

    #[test]
    fn timeout_policy_default() {
        let tp = TimeoutPolicy::default();
        assert_eq!(tp.request_timeout_ms, 30_000);
        assert_eq!(tp.connect_timeout_ms, 5_000);
    }
}
