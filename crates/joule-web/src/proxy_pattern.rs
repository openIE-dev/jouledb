//! Proxy pattern — virtual, protection, logging, and caching proxies.
//!
//! Provides a `Service` trait and several proxy wrappers:
//! - `VirtualProxy` — lazy-loads the real service on first use.
//! - `ProtectionProxy` — access-control gate with role/permission checks.
//! - `LoggingProxy` — records all calls with timestamps.
//! - `CachingProxy` — caches results by request key.
//! - `MetricsProxy` — tracks call count and total latency.
//! - `RemoteProxy` — simulates remote service invocation.
//! - `proxy_chain` — composes proxies.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ── Service trait ──────────────────────────────────────────────────

/// A request to a service.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ServiceRequest {
    pub method: String,
    pub key: String,
    pub payload: String,
}

impl ServiceRequest {
    pub fn new(
        method: impl Into<String>,
        key: impl Into<String>,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            method: method.into(),
            key: key.into(),
            payload: payload.into(),
        }
    }

    /// A cache key derived from method + key.
    pub fn cache_key(&self) -> String {
        format!("{}:{}", self.method, self.key)
    }
}

/// A response from a service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceResponse {
    pub status: u16,
    pub body: String,
}

impl ServiceResponse {
    pub fn ok(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            body: body.into(),
        }
    }

    pub fn error(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            body: body.into(),
        }
    }
}

/// The service trait that all proxies and real services implement.
pub trait Service {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse;
    fn name(&self) -> &str;
}

// ── Real service ───────────────────────────────────────────────────

/// A simple echo service for testing.
pub struct EchoService;

impl Service for EchoService {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        ServiceResponse::ok(format!("echo: {} {}", request.method, request.key))
    }

    fn name(&self) -> &str {
        "EchoService"
    }
}

/// A service built from a closure.
pub struct FnService {
    label: String,
    handler: Box<dyn FnMut(&ServiceRequest) -> ServiceResponse>,
}

impl FnService {
    pub fn new(
        label: impl Into<String>,
        handler: impl FnMut(&ServiceRequest) -> ServiceResponse + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            handler: Box::new(handler),
        }
    }
}

impl Service for FnService {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        (self.handler)(request)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

impl fmt::Debug for FnService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnService")
            .field("label", &self.label)
            .finish()
    }
}

// ── Virtual Proxy ──────────────────────────────────────────────────

/// Lazy-loading proxy that only creates the real service on first call.
pub struct VirtualProxy {
    inner: Option<Box<dyn Service>>,
    factory: Option<Box<dyn FnOnce() -> Box<dyn Service>>>,
    initialized: bool,
}

impl VirtualProxy {
    pub fn new(factory: impl FnOnce() -> Box<dyn Service> + 'static) -> Self {
        Self {
            inner: None,
            factory: Some(Box::new(factory)),
            initialized: false,
        }
    }

    /// Whether the real service has been created.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn ensure_init(&mut self) {
        if self.inner.is_none() {
            if let Some(factory) = self.factory.take() {
                self.inner = Some(factory());
                self.initialized = true;
            }
        }
    }
}

impl Service for VirtualProxy {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        self.ensure_init();
        match &mut self.inner {
            Some(svc) => svc.call(request),
            None => ServiceResponse::error(500, "virtual proxy: no service"),
        }
    }

    fn name(&self) -> &str {
        "VirtualProxy"
    }
}

// ── Protection Proxy ───────────────────────────────────────────────

/// Access-control proxy that checks permissions before forwarding.
pub struct ProtectionProxy {
    inner: Box<dyn Service>,
    allowed_roles: Vec<String>,
    current_role: String,
    denied_count: u64,
}

impl ProtectionProxy {
    pub fn new(
        inner: Box<dyn Service>,
        allowed_roles: Vec<String>,
        current_role: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            allowed_roles,
            current_role: current_role.into(),
            denied_count: 0,
        }
    }

    /// Change the current role.
    pub fn set_role(&mut self, role: impl Into<String>) {
        self.current_role = role.into();
    }

    /// Get the current role.
    pub fn current_role(&self) -> &str {
        &self.current_role
    }

    /// How many calls were denied.
    pub fn denied_count(&self) -> u64 {
        self.denied_count
    }

    fn is_allowed(&self) -> bool {
        self.allowed_roles.iter().any(|r| r == &self.current_role)
    }
}

impl Service for ProtectionProxy {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        if self.is_allowed() {
            self.inner.call(request)
        } else {
            self.denied_count += 1;
            ServiceResponse::error(403, format!("access denied for role: {}", self.current_role))
        }
    }

    fn name(&self) -> &str {
        "ProtectionProxy"
    }
}

// ── Logging Proxy ──────────────────────────────────────────────────

/// A log entry recorded by the logging proxy.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub sequence: u64,
    pub method: String,
    pub key: String,
    pub status: u16,
    pub elapsed_us: u64,
}

/// Proxy that records every call.
pub struct LoggingProxy {
    inner: Box<dyn Service>,
    log: Vec<LogEntry>,
    sequence: u64,
}

impl LoggingProxy {
    pub fn new(inner: Box<dyn Service>) -> Self {
        Self {
            inner,
            log: Vec::new(),
            sequence: 0,
        }
    }

    /// All recorded log entries.
    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }

    /// Number of calls logged.
    pub fn call_count(&self) -> u64 {
        self.sequence
    }

    /// Clear the log.
    pub fn clear_log(&mut self) {
        self.log.clear();
        self.sequence = 0;
    }
}

impl Service for LoggingProxy {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        self.sequence += 1;
        let seq = self.sequence;
        let start = Instant::now();
        let method = request.method.clone();
        let key = request.key.clone();
        let response = self.inner.call(request);
        let elapsed = start.elapsed().as_micros() as u64;
        self.log.push(LogEntry {
            sequence: seq,
            method,
            key,
            status: response.status,
            elapsed_us: elapsed,
        });
        response
    }

    fn name(&self) -> &str {
        "LoggingProxy"
    }
}

// ── Caching Proxy ──────────────────────────────────────────────────

/// Proxy that caches responses by request cache key.
pub struct CachingProxy {
    inner: Box<dyn Service>,
    cache: HashMap<String, ServiceResponse>,
    hits: u64,
    misses: u64,
    max_size: usize,
}

impl CachingProxy {
    pub fn new(inner: Box<dyn Service>, max_size: usize) -> Self {
        Self {
            inner,
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            max_size,
        }
    }

    /// Cache hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Cache miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Current cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Hit ratio (0.0 to 1.0).
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Invalidate a specific cache key.
    pub fn invalidate(&mut self, cache_key: &str) -> bool {
        self.cache.remove(cache_key).is_some()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

impl Service for CachingProxy {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        let ck = request.cache_key();
        if let Some(cached) = self.cache.get(&ck) {
            self.hits += 1;
            return cached.clone();
        }

        self.misses += 1;
        let response = self.inner.call(request);

        // Evict the oldest entry if at capacity.
        // Since HashMap doesn't guarantee order, just remove one arbitrary entry.
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&ck) {
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }

        self.cache.insert(ck, response.clone());
        response
    }

    fn name(&self) -> &str {
        "CachingProxy"
    }
}

// ── Metrics Proxy ──────────────────────────────────────────────────

/// Proxy that tracks call count and latency.
pub struct MetricsProxy {
    inner: Box<dyn Service>,
    call_count: u64,
    total_elapsed_us: u64,
    error_count: u64,
}

impl MetricsProxy {
    pub fn new(inner: Box<dyn Service>) -> Self {
        Self {
            inner,
            call_count: 0,
            total_elapsed_us: 0,
            error_count: 0,
        }
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    pub fn error_count(&self) -> u64 {
        self.error_count
    }

    pub fn total_elapsed_us(&self) -> u64 {
        self.total_elapsed_us
    }

    pub fn avg_elapsed_us(&self) -> f64 {
        if self.call_count == 0 {
            return 0.0;
        }
        self.total_elapsed_us as f64 / self.call_count as f64
    }
}

impl Service for MetricsProxy {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        self.call_count += 1;
        let start = Instant::now();
        let response = self.inner.call(request);
        self.total_elapsed_us += start.elapsed().as_micros() as u64;
        if response.status >= 400 {
            self.error_count += 1;
        }
        response
    }

    fn name(&self) -> &str {
        "MetricsProxy"
    }
}

// ── Remote Proxy ───────────────────────────────────────────────────

/// Simulates a remote service call with configurable latency and failure.
pub struct RemoteProxy {
    inner: Box<dyn Service>,
    endpoint: String,
    call_count: u64,
    simulate_failure: bool,
}

impl RemoteProxy {
    pub fn new(inner: Box<dyn Service>, endpoint: impl Into<String>) -> Self {
        Self {
            inner,
            endpoint: endpoint.into(),
            call_count: 0,
            simulate_failure: false,
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    /// Enable or disable failure simulation.
    pub fn set_simulate_failure(&mut self, fail: bool) {
        self.simulate_failure = fail;
    }
}

impl Service for RemoteProxy {
    fn call(&mut self, request: &ServiceRequest) -> ServiceResponse {
        self.call_count += 1;
        if self.simulate_failure {
            return ServiceResponse::error(503, "remote service unavailable");
        }
        self.inner.call(request)
    }

    fn name(&self) -> &str {
        "RemoteProxy"
    }
}

// ── Proxy chaining helper ──────────────────────────────────────────

/// Build a service from an inner service and a chain of proxy wrappers.
///
/// Each wrapper receives a `Box<dyn Service>` and returns a `Box<dyn Service>`.
pub fn proxy_chain(
    inner: Box<dyn Service>,
    wrappers: Vec<Box<dyn FnOnce(Box<dyn Service>) -> Box<dyn Service>>>,
) -> Box<dyn Service> {
    let mut svc = inner;
    for wrap in wrappers {
        svc = wrap(svc);
    }
    svc
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, key: &str) -> ServiceRequest {
        ServiceRequest::new(method, key, "")
    }

    #[test]
    fn echo_service_basic() {
        let mut svc = EchoService;
        let resp = svc.call(&req("GET", "foo"));
        assert_eq!(resp.status, 200);
        assert!(resp.body.contains("echo"));
        assert!(resp.body.contains("foo"));
    }

    #[test]
    fn fn_service() {
        let mut svc = FnService::new("counter", |_| ServiceResponse::ok("counted"));
        let resp = svc.call(&req("GET", "x"));
        assert_eq!(resp.body, "counted");
        assert_eq!(svc.name(), "counter");
    }

    #[test]
    fn virtual_proxy_lazy_init() {
        let mut vp = VirtualProxy::new(|| Box::new(EchoService));
        assert!(!vp.is_initialized());

        let resp = vp.call(&req("GET", "test"));
        assert!(vp.is_initialized());
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn virtual_proxy_subsequent_calls() {
        let mut vp = VirtualProxy::new(|| Box::new(EchoService));
        vp.call(&req("GET", "a"));
        let resp = vp.call(&req("GET", "b"));
        assert!(resp.body.contains("b"));
    }

    #[test]
    fn protection_proxy_allowed() {
        let mut pp = ProtectionProxy::new(
            Box::new(EchoService),
            vec!["admin".to_string(), "user".to_string()],
            "admin",
        );
        let resp = pp.call(&req("GET", "data"));
        assert_eq!(resp.status, 200);
        assert_eq!(pp.denied_count(), 0);
    }

    #[test]
    fn protection_proxy_denied() {
        let mut pp = ProtectionProxy::new(
            Box::new(EchoService),
            vec!["admin".to_string()],
            "guest",
        );
        let resp = pp.call(&req("GET", "secret"));
        assert_eq!(resp.status, 403);
        assert_eq!(pp.denied_count(), 1);
    }

    #[test]
    fn protection_proxy_role_change() {
        let mut pp = ProtectionProxy::new(
            Box::new(EchoService),
            vec!["admin".to_string()],
            "guest",
        );
        assert_eq!(pp.current_role(), "guest");
        pp.set_role("admin");
        assert_eq!(pp.current_role(), "admin");
        let resp = pp.call(&req("GET", "data"));
        assert_eq!(resp.status, 200);
    }

    #[test]
    fn logging_proxy_records() {
        let mut lp = LoggingProxy::new(Box::new(EchoService));
        lp.call(&req("GET", "a"));
        lp.call(&req("POST", "b"));

        assert_eq!(lp.call_count(), 2);
        let log = lp.log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].method, "GET");
        assert_eq!(log[0].key, "a");
        assert_eq!(log[1].method, "POST");
    }

    #[test]
    fn logging_proxy_clear() {
        let mut lp = LoggingProxy::new(Box::new(EchoService));
        lp.call(&req("GET", "x"));
        lp.clear_log();
        assert_eq!(lp.call_count(), 0);
        assert!(lp.log().is_empty());
    }

    #[test]
    fn caching_proxy_hit_miss() {
        let mut cp = CachingProxy::new(Box::new(EchoService), 10);

        // First call — miss.
        let r1 = cp.call(&req("GET", "foo"));
        assert_eq!(cp.misses(), 1);
        assert_eq!(cp.hits(), 0);

        // Second call with same key — hit.
        let r2 = cp.call(&req("GET", "foo"));
        assert_eq!(cp.hits(), 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn caching_proxy_invalidate() {
        let mut cp = CachingProxy::new(Box::new(EchoService), 10);
        cp.call(&req("GET", "key1"));
        assert_eq!(cp.cache_size(), 1);
        assert!(cp.invalidate("GET:key1"));
        assert_eq!(cp.cache_size(), 0);
        assert!(!cp.invalidate("nonexistent"));
    }

    #[test]
    fn caching_proxy_clear() {
        let mut cp = CachingProxy::new(Box::new(EchoService), 10);
        cp.call(&req("GET", "a"));
        cp.call(&req("GET", "b"));
        cp.clear();
        assert_eq!(cp.cache_size(), 0);
    }

    #[test]
    fn caching_proxy_hit_ratio() {
        let mut cp = CachingProxy::new(Box::new(EchoService), 10);
        assert_eq!(cp.hit_ratio(), 0.0);
        cp.call(&req("GET", "x")); // miss
        cp.call(&req("GET", "x")); // hit
        assert!((cp.hit_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn caching_proxy_eviction() {
        let mut cp = CachingProxy::new(Box::new(EchoService), 2);
        cp.call(&req("GET", "a"));
        cp.call(&req("GET", "b"));
        assert_eq!(cp.cache_size(), 2);
        cp.call(&req("GET", "c")); // triggers eviction
        assert!(cp.cache_size() <= 2);
    }

    #[test]
    fn metrics_proxy_tracking() {
        let mut mp = MetricsProxy::new(Box::new(EchoService));
        mp.call(&req("GET", "a"));
        mp.call(&req("GET", "b"));
        assert_eq!(mp.call_count(), 2);
        assert_eq!(mp.error_count(), 0);
    }

    #[test]
    fn metrics_proxy_errors() {
        let mut mp = MetricsProxy::new(Box::new(FnService::new("fail", |_| {
            ServiceResponse::error(500, "boom")
        })));
        mp.call(&req("GET", "x"));
        assert_eq!(mp.error_count(), 1);
    }

    #[test]
    fn metrics_proxy_avg_elapsed() {
        let mp = MetricsProxy::new(Box::new(EchoService));
        assert_eq!(mp.avg_elapsed_us(), 0.0);
    }

    #[test]
    fn remote_proxy_success() {
        let mut rp = RemoteProxy::new(Box::new(EchoService), "https://api.example.com");
        assert_eq!(rp.endpoint(), "https://api.example.com");
        let resp = rp.call(&req("GET", "data"));
        assert_eq!(resp.status, 200);
        assert_eq!(rp.call_count(), 1);
    }

    #[test]
    fn remote_proxy_failure() {
        let mut rp = RemoteProxy::new(Box::new(EchoService), "https://api.example.com");
        rp.set_simulate_failure(true);
        let resp = rp.call(&req("GET", "data"));
        assert_eq!(resp.status, 503);
    }

    #[test]
    fn proxy_chain_composition() {
        let chain = proxy_chain(
            Box::new(EchoService),
            vec![
                Box::new(|svc| Box::new(LoggingProxy::new(svc)) as Box<dyn Service>),
                Box::new(|svc| Box::new(MetricsProxy::new(svc)) as Box<dyn Service>),
            ],
        );
        assert_eq!(chain.name(), "MetricsProxy");
    }

    #[test]
    fn service_request_cache_key() {
        let r = ServiceRequest::new("GET", "users/1", "");
        assert_eq!(r.cache_key(), "GET:users/1");
    }

    #[test]
    fn service_response_constructors() {
        let ok = ServiceResponse::ok("success");
        assert_eq!(ok.status, 200);
        let err = ServiceResponse::error(404, "not found");
        assert_eq!(err.status, 404);
    }
}
