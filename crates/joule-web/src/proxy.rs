//! HTTP proxy / reverse proxy configuration engine.
//!
//! Replaces `http-proxy`, `nginx` config, and `envoy` route matching with pure
//! Rust.  Route matching, path rewriting, header injection, load balancing
//! (round-robin, random, least-connections), health checks, circuit breaker.

use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyError {
    NoBackendAvailable,
    CircuitOpen(String),
    RouteNotFound(String),
    InvalidConfig(String),
}

impl fmt::Display for ProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackendAvailable => write!(f, "no backend available"),
            Self::CircuitOpen(name) => write!(f, "circuit open for {name}"),
            Self::RouteNotFound(path) => write!(f, "no route for {path}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for ProxyError {}

// ── Backend ────────────────────────────────────────────────────

/// Backend health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHealth {
    Healthy,
    Unhealthy,
    Unknown,
}

/// A backend server.
#[derive(Debug, Clone)]
pub struct Backend {
    pub address: String,
    pub weight: u32,
    pub health: BackendHealth,
    pub active_connections: u64,
    pub total_requests: u64,
}

impl Backend {
    pub fn new(address: &str) -> Self {
        Self {
            address: address.to_string(),
            weight: 1,
            health: BackendHealth::Unknown,
            active_connections: 0,
            total_requests: 0,
        }
    }

    pub fn with_weight(mut self, w: u32) -> Self {
        self.weight = w;
        self
    }

    pub fn is_available(&self) -> bool {
        self.health != BackendHealth::Unhealthy
    }
}

// ── Load balancer ──────────────────────────────────────────────

/// Load balancing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbStrategy {
    RoundRobin,
    Random,
    LeastConnections,
    WeightedRoundRobin,
}

/// Load balancer state.
#[derive(Debug)]
pub struct LoadBalancer {
    pub strategy: LbStrategy,
    pub backends: Vec<Backend>,
    round_robin_idx: usize,
    /// Seed for deterministic "random" in no-std context.
    random_seed: u64,
}

impl LoadBalancer {
    pub fn new(strategy: LbStrategy, backends: Vec<Backend>) -> Self {
        Self {
            strategy,
            backends,
            round_robin_idx: 0,
            random_seed: 12345,
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.random_seed = seed;
    }

    fn available_indices(&self) -> Vec<usize> {
        self.backends
            .iter()
            .enumerate()
            .filter(|(_, b)| b.is_available())
            .map(|(i, _)| i)
            .collect()
    }

    /// Select the next backend index.
    pub fn select(&mut self) -> Result<usize, ProxyError> {
        let avail = self.available_indices();
        if avail.is_empty() {
            return Err(ProxyError::NoBackendAvailable);
        }

        match self.strategy {
            LbStrategy::RoundRobin => {
                let idx = self.round_robin_idx % avail.len();
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                Ok(avail[idx])
            }
            LbStrategy::Random => {
                // Simple xorshift.
                self.random_seed ^= self.random_seed << 13;
                self.random_seed ^= self.random_seed >> 7;
                self.random_seed ^= self.random_seed << 17;
                let idx = (self.random_seed as usize) % avail.len();
                Ok(avail[idx])
            }
            LbStrategy::LeastConnections => {
                let best = avail
                    .iter()
                    .copied()
                    .min_by_key(|i| self.backends[*i].active_connections)
                    .unwrap();
                Ok(best)
            }
            LbStrategy::WeightedRoundRobin => {
                // Total weight among available.
                let total_weight: u32 =
                    avail.iter().map(|i| self.backends[*i].weight).sum();
                if total_weight == 0 {
                    return Err(ProxyError::NoBackendAvailable);
                }
                let target = (self.round_robin_idx as u32) % total_weight;
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                let mut acc = 0u32;
                for &i in &avail {
                    acc += self.backends[i].weight;
                    if target < acc {
                        return Ok(i);
                    }
                }
                Ok(*avail.last().unwrap())
            }
        }
    }

    pub fn mark_healthy(&mut self, idx: usize) {
        if idx < self.backends.len() {
            self.backends[idx].health = BackendHealth::Healthy;
        }
    }

    pub fn mark_unhealthy(&mut self, idx: usize) {
        if idx < self.backends.len() {
            self.backends[idx].health = BackendHealth::Unhealthy;
        }
    }
}

// ── Circuit breaker ────────────────────────────────────────────

/// Circuit breaker state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Per-backend circuit breaker.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    pub state: CircuitState,
    pub failure_count: u32,
    pub failure_threshold: u32,
    pub success_count: u32,
    pub success_threshold: u32,
    pub open_since_epoch_ms: u64,
    pub cooldown_ms: u64,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown_ms: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            success_count: 0,
            success_threshold: 2,
            open_since_epoch_ms: 0,
            cooldown_ms,
        }
    }

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

    pub fn record_failure(&mut self, now_epoch_ms: u64) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    self.open_since_epoch_ms = now_epoch_ms;
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
                self.open_since_epoch_ms = now_epoch_ms;
                self.success_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    pub fn can_attempt(&mut self, now_epoch_ms: u64) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                if now_epoch_ms >= self.open_since_epoch_ms + self.cooldown_ms {
                    self.state = CircuitState::HalfOpen;
                    self.success_count = 0;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
}

// ── Route / path rewriting ─────────────────────────────────────

/// Path rewrite rule.
#[derive(Debug, Clone)]
pub struct PathRewrite {
    /// Prefix to strip.
    pub strip_prefix: Option<String>,
    /// Prefix to add.
    pub add_prefix: Option<String>,
}

impl PathRewrite {
    pub fn apply(&self, path: &str) -> String {
        let mut result = path.to_string();
        if let Some(prefix) = &self.strip_prefix {
            if result.starts_with(prefix.as_str()) {
                result = result[prefix.len()..].to_string();
            }
        }
        if let Some(prefix) = &self.add_prefix {
            result = format!("{}{}", prefix, result);
        }
        if !result.starts_with('/') {
            result = format!("/{}", result);
        }
        result
    }
}

/// A proxy route.
#[derive(Debug, Clone)]
pub struct ProxyRoute {
    pub path_prefix: String,
    pub rewrite: Option<PathRewrite>,
    pub inject_headers: HashMap<String, String>,
    pub backend_group: String,
}

impl ProxyRoute {
    pub fn matches(&self, path: &str) -> bool {
        path.starts_with(&self.path_prefix)
    }

    pub fn rewrite_path(&self, path: &str) -> String {
        if let Some(rw) = &self.rewrite {
            rw.apply(path)
        } else {
            path.to_string()
        }
    }
}

// ── Proxy config ───────────────────────────────────────────────

/// Full proxy configuration.
#[derive(Debug)]
pub struct ProxyConfig {
    pub routes: Vec<ProxyRoute>,
    pub backend_groups: HashMap<String, LoadBalancer>,
}

impl ProxyConfig {
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            backend_groups: HashMap::new(),
        }
    }

    pub fn add_route(&mut self, route: ProxyRoute) {
        self.routes.push(route);
    }

    pub fn add_backend_group(&mut self, name: &str, lb: LoadBalancer) {
        self.backend_groups.insert(name.to_string(), lb);
    }

    /// Resolve a request path to a (backend_address, rewritten_path, injected_headers).
    pub fn resolve(
        &mut self,
        path: &str,
    ) -> Result<(String, String, HashMap<String, String>), ProxyError> {
        // Find the first matching route (longest prefix match would be an enhancement).
        let route = self
            .routes
            .iter()
            .find(|r| r.matches(path))
            .ok_or_else(|| ProxyError::RouteNotFound(path.to_string()))?
            .clone();

        let lb = self
            .backend_groups
            .get_mut(&route.backend_group)
            .ok_or_else(|| {
                ProxyError::InvalidConfig(format!(
                    "backend group '{}' not found",
                    route.backend_group
                ))
            })?;

        let idx = lb.select()?;
        let addr = lb.backends[idx].address.clone();
        let rewritten = route.rewrite_path(path);

        Ok((addr, rewritten, route.inject_headers.clone()))
    }
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Health check config ────────────────────────────────────────

/// Health check configuration for a backend group.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub path: String,
    pub interval_ms: u64,
    pub timeout_ms: u64,
    pub healthy_threshold: u32,
    pub unhealthy_threshold: u32,
    pub expected_status: u16,
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self {
            path: "/health".to_string(),
            interval_ms: 10_000,
            timeout_ms: 5_000,
            healthy_threshold: 2,
            unhealthy_threshold: 3,
            expected_status: 200,
        }
    }
}

impl HealthCheck {
    pub fn check_url(&self, backend_addr: &str) -> String {
        format!("{}{}", backend_addr, self.path)
    }

    pub fn is_status_ok(&self, status: u16) -> bool {
        status == self.expected_status
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_selection() {
        let backends = vec![
            Backend::new("http://a:8080"),
            Backend::new("http://b:8080"),
            Backend::new("http://c:8080"),
        ];
        let mut lb = LoadBalancer::new(LbStrategy::RoundRobin, backends);
        // Mark all healthy.
        for i in 0..3 {
            lb.mark_healthy(i);
        }
        assert_eq!(lb.select().unwrap(), 0);
        assert_eq!(lb.select().unwrap(), 1);
        assert_eq!(lb.select().unwrap(), 2);
        assert_eq!(lb.select().unwrap(), 0);
    }

    #[test]
    fn round_robin_skips_unhealthy() {
        let backends = vec![
            Backend::new("http://a:8080"),
            Backend::new("http://b:8080"),
            Backend::new("http://c:8080"),
        ];
        let mut lb = LoadBalancer::new(LbStrategy::RoundRobin, backends);
        lb.mark_healthy(0);
        lb.mark_unhealthy(1);
        lb.mark_healthy(2);
        let first = lb.select().unwrap();
        let second = lb.select().unwrap();
        assert_ne!(first, 1);
        assert_ne!(second, 1);
    }

    #[test]
    fn least_connections() {
        let mut backends = vec![
            Backend::new("http://a:8080"),
            Backend::new("http://b:8080"),
        ];
        backends[0].active_connections = 10;
        backends[0].health = BackendHealth::Healthy;
        backends[1].active_connections = 2;
        backends[1].health = BackendHealth::Healthy;
        let mut lb = LoadBalancer::new(LbStrategy::LeastConnections, backends);
        assert_eq!(lb.select().unwrap(), 1);
    }

    #[test]
    fn weighted_round_robin() {
        let backends = vec![
            Backend::new("http://a:8080").with_weight(3),
            Backend::new("http://b:8080").with_weight(1),
        ];
        let mut lb = LoadBalancer::new(LbStrategy::WeightedRoundRobin, backends);
        lb.mark_healthy(0);
        lb.mark_healthy(1);
        let mut counts = [0u32; 2];
        for _ in 0..4 {
            let idx = lb.select().unwrap();
            counts[idx] += 1;
        }
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 1);
    }

    #[test]
    fn no_backend_available() {
        let backends = vec![Backend::new("http://a:8080")];
        let mut lb = LoadBalancer::new(LbStrategy::RoundRobin, backends);
        lb.mark_unhealthy(0);
        assert_eq!(lb.select(), Err(ProxyError::NoBackendAvailable));
    }

    #[test]
    fn path_rewrite_strip_and_add() {
        let rw = PathRewrite {
            strip_prefix: Some("/api".to_string()),
            add_prefix: Some("/v2".to_string()),
        };
        assert_eq!(rw.apply("/api/users"), "/v2/users");
    }

    #[test]
    fn path_rewrite_strip_only() {
        let rw = PathRewrite {
            strip_prefix: Some("/old".to_string()),
            add_prefix: None,
        };
        assert_eq!(rw.apply("/old/path"), "/path");
    }

    #[test]
    fn proxy_route_matching() {
        let route = ProxyRoute {
            path_prefix: "/api/".to_string(),
            rewrite: None,
            inject_headers: HashMap::new(),
            backend_group: "api".to_string(),
        };
        assert!(route.matches("/api/users"));
        assert!(!route.matches("/web/index"));
    }

    #[test]
    fn proxy_config_resolve() {
        let mut config = ProxyConfig::new();
        config.add_route(ProxyRoute {
            path_prefix: "/api/".to_string(),
            rewrite: Some(PathRewrite {
                strip_prefix: Some("/api".to_string()),
                add_prefix: None,
            }),
            inject_headers: {
                let mut h = HashMap::new();
                h.insert("X-Proxy".to_string(), "true".to_string());
                h
            },
            backend_group: "api".to_string(),
        });
        let backends = vec![Backend::new("http://backend:8080")];
        let mut lb = LoadBalancer::new(LbStrategy::RoundRobin, backends);
        lb.mark_healthy(0);
        config.add_backend_group("api", lb);

        let (addr, path, headers) = config.resolve("/api/users").unwrap();
        assert_eq!(addr, "http://backend:8080");
        assert_eq!(path, "/users");
        assert_eq!(headers.get("X-Proxy").unwrap(), "true");
    }

    #[test]
    fn circuit_breaker_trips() {
        let mut cb = CircuitBreaker::new(3, 5000);
        assert!(cb.can_attempt(0));
        cb.record_failure(100);
        cb.record_failure(200);
        assert_eq!(cb.state, CircuitState::Closed);
        cb.record_failure(300);
        assert_eq!(cb.state, CircuitState::Open);
        assert!(!cb.can_attempt(300));
    }

    #[test]
    fn circuit_breaker_half_open_recovery() {
        let mut cb = CircuitBreaker::new(1, 1000);
        cb.record_failure(0);
        assert_eq!(cb.state, CircuitState::Open);
        // After cooldown, transitions to half-open.
        assert!(cb.can_attempt(1500));
        assert_eq!(cb.state, CircuitState::HalfOpen);
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.state, CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_half_open_failure() {
        let mut cb = CircuitBreaker::new(1, 100);
        cb.record_failure(0);
        assert!(cb.can_attempt(200));
        assert_eq!(cb.state, CircuitState::HalfOpen);
        cb.record_failure(200);
        assert_eq!(cb.state, CircuitState::Open);
    }

    #[test]
    fn health_check_url() {
        let hc = HealthCheck::default();
        assert_eq!(
            hc.check_url("http://backend:8080"),
            "http://backend:8080/health"
        );
        assert!(hc.is_status_ok(200));
        assert!(!hc.is_status_ok(500));
    }
}
