//! Service Worker — cache strategies, precache manifests, route matching,
//! and offline fallback modeling.
//!
//! Models the logical behavior of service workers in pure Rust without
//! web-sys or browser APIs. Useful for generating SW configuration,
//! testing cache strategies, and planning offline-first architectures.

use std::collections::HashMap;
use std::fmt;

// ── Cache Strategy ───────────────────────────────────────────────

/// Caching strategy for a route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheStrategy {
    /// Serve from cache if available, else fetch from network.
    CacheFirst,
    /// Fetch from network, fall back to cache on failure.
    NetworkFirst,
    /// Serve stale from cache immediately, revalidate in background.
    StaleWhileRevalidate,
    /// Always fetch from network, never use cache.
    NetworkOnly,
    /// Always serve from cache, never fetch.
    CacheOnly,
}

impl fmt::Display for CacheStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CacheFirst => write!(f, "CacheFirst"),
            Self::NetworkFirst => write!(f, "NetworkFirst"),
            Self::StaleWhileRevalidate => write!(f, "StaleWhileRevalidate"),
            Self::NetworkOnly => write!(f, "NetworkOnly"),
            Self::CacheOnly => write!(f, "CacheOnly"),
        }
    }
}

// ── Route Pattern ────────────────────────────────────────────────

/// Pattern for matching request URLs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutePattern {
    /// Exact URL match.
    Exact(String),
    /// URL prefix match.
    Prefix(String),
    /// URL suffix/extension match (e.g. ".js", ".css").
    Extension(String),
    /// Match any URL.
    Any,
}

impl RoutePattern {
    /// Check if a URL matches this pattern.
    pub fn matches(&self, url: &str) -> bool {
        match self {
            Self::Exact(s) => url == s,
            Self::Prefix(p) => url.starts_with(p),
            Self::Extension(ext) => {
                // Strip query string before checking extension
                let path = url.split('?').next().unwrap_or(url);
                path.ends_with(ext)
            }
            Self::Any => true,
        }
    }
}

// ── Cache Entry ──────────────────────────────────────────────────

/// A single entry in the precache manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecacheEntry {
    pub url: String,
    /// Revision hash for cache busting.
    pub revision: Option<String>,
    /// Whether the entry is an opaque response (CDN, etc).
    pub integrity: Option<String>,
}

impl PrecacheEntry {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into(), revision: None, integrity: None }
    }

    pub fn with_revision(mut self, rev: impl Into<String>) -> Self {
        self.revision = Some(rev.into());
        self
    }

    pub fn with_integrity(mut self, hash: impl Into<String>) -> Self {
        self.integrity = Some(hash.into());
        self
    }

    /// Cache key: url + revision if present.
    pub fn cache_key(&self) -> String {
        match &self.revision {
            Some(rev) => format!("{}?__rev={}", self.url, rev),
            None => self.url.clone(),
        }
    }
}

// ── Route Rule ───────────────────────────────────────────────────

/// A routing rule binding a pattern to a cache strategy.
#[derive(Debug, Clone)]
pub struct RouteRule {
    pub pattern: RoutePattern,
    pub strategy: CacheStrategy,
    pub cache_name: String,
    /// Max entries in this cache (LRU eviction).
    pub max_entries: Option<usize>,
    /// Max age in seconds before revalidation.
    pub max_age_secs: Option<u64>,
}

impl RouteRule {
    pub fn new(pattern: RoutePattern, strategy: CacheStrategy, cache_name: impl Into<String>) -> Self {
        Self {
            pattern,
            strategy,
            cache_name: cache_name.into(),
            max_entries: None,
            max_age_secs: None,
        }
    }

    pub fn max_entries(mut self, n: usize) -> Self { self.max_entries = Some(n); self }
    pub fn max_age(mut self, secs: u64) -> Self { self.max_age_secs = Some(secs); self }
}

// ── Cache Simulation ─────────────────────────────────────────────

/// Simulated cache state for testing strategies.
#[derive(Debug, Clone)]
pub struct CacheState {
    entries: HashMap<String, CachedResponse>,
    max_entries: Option<usize>,
    /// Insertion-order tracking for LRU.
    access_order: Vec<String>,
}

/// Simulated cached response.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    pub url: String,
    pub status: u16,
    pub body_size: usize,
    pub timestamp: u64,
}

impl CacheState {
    pub fn new(max_entries: Option<usize>) -> Self {
        Self { entries: HashMap::new(), max_entries, access_order: Vec::new() }
    }

    /// Put a response into the cache, evicting LRU if over limit.
    pub fn put(&mut self, url: String, response: CachedResponse) {
        self.access_order.retain(|k| k != &url);
        self.access_order.push(url.clone());
        self.entries.insert(url, response);
        self.evict_if_needed();
    }

    /// Get a cached response (marks as recently used).
    pub fn get(&mut self, url: &str) -> Option<&CachedResponse> {
        if self.entries.contains_key(url) {
            self.access_order.retain(|k| k != url);
            self.access_order.push(url.to_string());
            self.entries.get(url)
        } else {
            None
        }
    }

    /// Check if URL is cached without updating access order.
    pub fn contains(&self, url: &str) -> bool {
        self.entries.contains_key(url)
    }

    /// Remove a cached entry.
    pub fn remove(&mut self, url: &str) -> bool {
        self.access_order.retain(|k| k != url);
        self.entries.remove(url).is_some()
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize { self.entries.len() }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Total body size of all cached entries.
    pub fn total_size(&self) -> usize {
        self.entries.values().map(|r| r.body_size).sum()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    fn evict_if_needed(&mut self) {
        if let Some(max) = self.max_entries {
            while self.entries.len() > max {
                if let Some(oldest) = self.access_order.first().cloned() {
                    self.entries.remove(&oldest);
                    self.access_order.remove(0);
                } else {
                    break;
                }
            }
        }
    }
}

// ── Fetch Decision ───────────────────────────────────────────────

/// Result of strategy evaluation: where to source the response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchDecision {
    /// Serve from cache.
    ServeFromCache,
    /// Fetch from network (with optional cache fallback).
    FetchFromNetwork { cache_fallback: bool },
    /// Serve stale cache, also revalidate in background.
    ServeStaleAndRevalidate,
    /// No response available (offline + no cache).
    Unavailable,
}

/// Evaluate which action a strategy dictates.
pub fn evaluate_strategy(
    strategy: CacheStrategy,
    is_cached: bool,
    is_online: bool,
) -> FetchDecision {
    match strategy {
        CacheStrategy::CacheFirst => {
            if is_cached { FetchDecision::ServeFromCache }
            else if is_online { FetchDecision::FetchFromNetwork { cache_fallback: false } }
            else { FetchDecision::Unavailable }
        }
        CacheStrategy::NetworkFirst => {
            if is_online { FetchDecision::FetchFromNetwork { cache_fallback: true } }
            else if is_cached { FetchDecision::ServeFromCache }
            else { FetchDecision::Unavailable }
        }
        CacheStrategy::StaleWhileRevalidate => {
            if is_cached { FetchDecision::ServeStaleAndRevalidate }
            else if is_online { FetchDecision::FetchFromNetwork { cache_fallback: false } }
            else { FetchDecision::Unavailable }
        }
        CacheStrategy::NetworkOnly => {
            if is_online { FetchDecision::FetchFromNetwork { cache_fallback: false } }
            else { FetchDecision::Unavailable }
        }
        CacheStrategy::CacheOnly => {
            if is_cached { FetchDecision::ServeFromCache }
            else { FetchDecision::Unavailable }
        }
    }
}

// ── Service Worker Config ────────────────────────────────────────

/// Complete service worker configuration.
#[derive(Debug, Clone)]
pub struct ServiceWorkerConfig {
    pub precache: Vec<PrecacheEntry>,
    pub routes: Vec<RouteRule>,
    pub offline_fallback: Option<String>,
    pub navigation_preload: bool,
    pub skip_waiting: bool,
    pub clients_claim: bool,
}

impl ServiceWorkerConfig {
    pub fn new() -> Self {
        Self {
            precache: Vec::new(),
            routes: Vec::new(),
            offline_fallback: None,
            navigation_preload: false,
            skip_waiting: true,
            clients_claim: true,
        }
    }

    pub fn add_precache(mut self, entry: PrecacheEntry) -> Self {
        self.precache.push(entry);
        self
    }

    pub fn add_route(mut self, rule: RouteRule) -> Self {
        self.routes.push(rule);
        self
    }

    pub fn offline_fallback(mut self, url: impl Into<String>) -> Self {
        self.offline_fallback = Some(url.into());
        self
    }

    pub fn navigation_preload(mut self, enabled: bool) -> Self {
        self.navigation_preload = enabled;
        self
    }

    pub fn skip_waiting(mut self, sw: bool) -> Self { self.skip_waiting = sw; self }
    pub fn clients_claim(mut self, cc: bool) -> Self { self.clients_claim = cc; self }

    /// Find the matching route rule for a given URL.
    pub fn match_route(&self, url: &str) -> Option<&RouteRule> {
        self.routes.iter().find(|r| r.pattern.matches(url))
    }

    /// Determine the cache strategy for a URL, falling back to NetworkFirst.
    pub fn strategy_for(&self, url: &str) -> CacheStrategy {
        self.match_route(url).map(|r| r.strategy).unwrap_or(CacheStrategy::NetworkFirst)
    }

    /// Get all precache URLs.
    pub fn precache_urls(&self) -> Vec<&str> {
        self.precache.iter().map(|e| e.url.as_str()).collect()
    }

    /// Total number of precache entries.
    pub fn precache_count(&self) -> usize { self.precache.len() }

    /// Generate a list of all cache names used by routes.
    pub fn cache_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.routes.iter().map(|r| r.cache_name.as_str()).collect();
        names.sort();
        names.dedup();
        names
    }

    /// Validate the configuration for common issues.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.precache.is_empty() {
            warnings.push("no precache entries defined".into());
        }
        if self.routes.is_empty() {
            warnings.push("no route rules defined".into());
        }
        if self.offline_fallback.is_none() {
            warnings.push("no offline fallback page set".into());
        }
        // Check for overlapping exact routes
        let exact_urls: Vec<&str> = self.routes.iter().filter_map(|r| {
            if let RoutePattern::Exact(ref u) = r.pattern { Some(u.as_str()) } else { None }
        }).collect();
        let mut seen = std::collections::HashSet::new();
        for url in &exact_urls {
            if !seen.insert(url) {
                warnings.push(format!("duplicate exact route for: {}", url));
            }
        }
        warnings
    }
}

impl Default for ServiceWorkerConfig {
    fn default() -> Self { Self::new() }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_pattern_exact() {
        let p = RoutePattern::Exact("/index.html".into());
        assert!(p.matches("/index.html"));
        assert!(!p.matches("/index.htm"));
        assert!(!p.matches("/index.html/extra"));
    }

    #[test]
    fn test_route_pattern_prefix() {
        let p = RoutePattern::Prefix("/api/".into());
        assert!(p.matches("/api/users"));
        assert!(p.matches("/api/"));
        assert!(!p.matches("/app/"));
    }

    #[test]
    fn test_route_pattern_extension() {
        let p = RoutePattern::Extension(".js".into());
        assert!(p.matches("/app.js"));
        assert!(p.matches("/static/bundle.js"));
        assert!(!p.matches("/app.css"));
    }

    #[test]
    fn test_route_pattern_extension_with_query() {
        let p = RoutePattern::Extension(".css".into());
        assert!(p.matches("/style.css?v=123"));
    }

    #[test]
    fn test_route_pattern_any() {
        let p = RoutePattern::Any;
        assert!(p.matches("/anything"));
        assert!(p.matches(""));
    }

    #[test]
    fn test_cache_first_online_cached() {
        let d = evaluate_strategy(CacheStrategy::CacheFirst, true, true);
        assert_eq!(d, FetchDecision::ServeFromCache);
    }

    #[test]
    fn test_cache_first_online_not_cached() {
        let d = evaluate_strategy(CacheStrategy::CacheFirst, false, true);
        assert_eq!(d, FetchDecision::FetchFromNetwork { cache_fallback: false });
    }

    #[test]
    fn test_cache_first_offline_not_cached() {
        let d = evaluate_strategy(CacheStrategy::CacheFirst, false, false);
        assert_eq!(d, FetchDecision::Unavailable);
    }

    #[test]
    fn test_network_first_online() {
        let d = evaluate_strategy(CacheStrategy::NetworkFirst, true, true);
        assert_eq!(d, FetchDecision::FetchFromNetwork { cache_fallback: true });
    }

    #[test]
    fn test_network_first_offline_cached() {
        let d = evaluate_strategy(CacheStrategy::NetworkFirst, true, false);
        assert_eq!(d, FetchDecision::ServeFromCache);
    }

    #[test]
    fn test_network_first_offline_not_cached() {
        let d = evaluate_strategy(CacheStrategy::NetworkFirst, false, false);
        assert_eq!(d, FetchDecision::Unavailable);
    }

    #[test]
    fn test_stale_while_revalidate_cached() {
        let d = evaluate_strategy(CacheStrategy::StaleWhileRevalidate, true, true);
        assert_eq!(d, FetchDecision::ServeStaleAndRevalidate);
    }

    #[test]
    fn test_stale_while_revalidate_not_cached_online() {
        let d = evaluate_strategy(CacheStrategy::StaleWhileRevalidate, false, true);
        assert_eq!(d, FetchDecision::FetchFromNetwork { cache_fallback: false });
    }

    #[test]
    fn test_network_only() {
        assert_eq!(
            evaluate_strategy(CacheStrategy::NetworkOnly, true, true),
            FetchDecision::FetchFromNetwork { cache_fallback: false }
        );
        assert_eq!(
            evaluate_strategy(CacheStrategy::NetworkOnly, true, false),
            FetchDecision::Unavailable
        );
    }

    #[test]
    fn test_cache_only() {
        assert_eq!(
            evaluate_strategy(CacheStrategy::CacheOnly, true, false),
            FetchDecision::ServeFromCache
        );
        assert_eq!(
            evaluate_strategy(CacheStrategy::CacheOnly, false, true),
            FetchDecision::Unavailable
        );
    }

    #[test]
    fn test_precache_entry_cache_key() {
        let e1 = PrecacheEntry::new("/app.js");
        assert_eq!(e1.cache_key(), "/app.js");

        let e2 = PrecacheEntry::new("/app.js").with_revision("abc123");
        assert_eq!(e2.cache_key(), "/app.js?__rev=abc123");
    }

    #[test]
    fn test_precache_entry_integrity() {
        let e = PrecacheEntry::new("/lib.js")
            .with_integrity("sha256-abc123");
        assert_eq!(e.integrity.as_deref(), Some("sha256-abc123"));
    }

    #[test]
    fn test_cache_state_put_get() {
        let mut cache = CacheState::new(None);
        cache.put("/a".into(), CachedResponse {
            url: "/a".into(), status: 200, body_size: 100, timestamp: 1000,
        });
        assert!(cache.contains("/a"));
        assert!(!cache.contains("/b"));
        assert_eq!(cache.len(), 1);

        let r = cache.get("/a").unwrap();
        assert_eq!(r.status, 200);
    }

    #[test]
    fn test_cache_state_lru_eviction() {
        let mut cache = CacheState::new(Some(2));
        for (i, url) in ["/a", "/b", "/c"].iter().enumerate() {
            cache.put(url.to_string(), CachedResponse {
                url: url.to_string(), status: 200, body_size: 50, timestamp: i as u64,
            });
        }
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains("/a")); // evicted
        assert!(cache.contains("/b"));
        assert!(cache.contains("/c"));
    }

    #[test]
    fn test_cache_state_lru_access_updates() {
        let mut cache = CacheState::new(Some(2));
        cache.put("/a".into(), CachedResponse { url: "/a".into(), status: 200, body_size: 10, timestamp: 0 });
        cache.put("/b".into(), CachedResponse { url: "/b".into(), status: 200, body_size: 10, timestamp: 1 });

        // Access /a to make it recently used
        let _ = cache.get("/a");

        // Adding /c should evict /b (least recently used)
        cache.put("/c".into(), CachedResponse { url: "/c".into(), status: 200, body_size: 10, timestamp: 2 });
        assert!(cache.contains("/a"));
        assert!(!cache.contains("/b"));
        assert!(cache.contains("/c"));
    }

    #[test]
    fn test_cache_state_remove() {
        let mut cache = CacheState::new(None);
        cache.put("/x".into(), CachedResponse { url: "/x".into(), status: 200, body_size: 10, timestamp: 0 });
        assert!(cache.remove("/x"));
        assert!(!cache.remove("/x"));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_state_total_size() {
        let mut cache = CacheState::new(None);
        cache.put("/a".into(), CachedResponse { url: "/a".into(), status: 200, body_size: 100, timestamp: 0 });
        cache.put("/b".into(), CachedResponse { url: "/b".into(), status: 200, body_size: 200, timestamp: 1 });
        assert_eq!(cache.total_size(), 300);
    }

    #[test]
    fn test_cache_state_clear() {
        let mut cache = CacheState::new(None);
        cache.put("/a".into(), CachedResponse { url: "/a".into(), status: 200, body_size: 10, timestamp: 0 });
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_sw_config_match_route() {
        let config = ServiceWorkerConfig::new()
            .add_route(RouteRule::new(
                RoutePattern::Extension(".js".into()),
                CacheStrategy::CacheFirst,
                "js-cache",
            ))
            .add_route(RouteRule::new(
                RoutePattern::Prefix("/api/".into()),
                CacheStrategy::NetworkFirst,
                "api-cache",
            ));

        assert_eq!(config.strategy_for("/app.js"), CacheStrategy::CacheFirst);
        assert_eq!(config.strategy_for("/api/data"), CacheStrategy::NetworkFirst);
        assert_eq!(config.strategy_for("/page.html"), CacheStrategy::NetworkFirst); // fallback
    }

    #[test]
    fn test_sw_config_precache() {
        let config = ServiceWorkerConfig::new()
            .add_precache(PrecacheEntry::new("/index.html").with_revision("v1"))
            .add_precache(PrecacheEntry::new("/app.js"));

        assert_eq!(config.precache_count(), 2);
        let urls = config.precache_urls();
        assert!(urls.contains(&"/index.html"));
        assert!(urls.contains(&"/app.js"));
    }

    #[test]
    fn test_sw_config_cache_names() {
        let config = ServiceWorkerConfig::new()
            .add_route(RouteRule::new(RoutePattern::Any, CacheStrategy::CacheFirst, "main"))
            .add_route(RouteRule::new(RoutePattern::Extension(".png".into()), CacheStrategy::CacheFirst, "images"))
            .add_route(RouteRule::new(RoutePattern::Extension(".js".into()), CacheStrategy::CacheFirst, "main"));

        let names = config.cache_names();
        assert_eq!(names.len(), 2); // "images" and "main" (deduplicated)
    }

    #[test]
    fn test_sw_config_validate() {
        let config = ServiceWorkerConfig::new();
        let w = config.validate();
        assert!(w.iter().any(|s| s.contains("no precache")));
        assert!(w.iter().any(|s| s.contains("no route")));
        assert!(w.iter().any(|s| s.contains("offline fallback")));
    }

    #[test]
    fn test_sw_config_validate_dup_routes() {
        let config = ServiceWorkerConfig::new()
            .add_precache(PrecacheEntry::new("/a"))
            .add_route(RouteRule::new(RoutePattern::Exact("/x".into()), CacheStrategy::CacheFirst, "c"))
            .add_route(RouteRule::new(RoutePattern::Exact("/x".into()), CacheStrategy::NetworkFirst, "c"))
            .offline_fallback("/offline.html");

        let w = config.validate();
        assert!(w.iter().any(|s| s.contains("duplicate")));
    }

    #[test]
    fn test_sw_config_defaults() {
        let config = ServiceWorkerConfig::new();
        assert!(config.skip_waiting);
        assert!(config.clients_claim);
        assert!(!config.navigation_preload);
    }

    #[test]
    fn test_route_rule_expiration() {
        let rule = RouteRule::new(
            RoutePattern::Extension(".png".into()),
            CacheStrategy::CacheFirst,
            "images",
        ).max_entries(50).max_age(86400);

        assert_eq!(rule.max_entries, Some(50));
        assert_eq!(rule.max_age_secs, Some(86400));
    }

    #[test]
    fn test_cache_strategy_display() {
        assert_eq!(CacheStrategy::CacheFirst.to_string(), "CacheFirst");
        assert_eq!(CacheStrategy::NetworkFirst.to_string(), "NetworkFirst");
        assert_eq!(CacheStrategy::StaleWhileRevalidate.to_string(), "StaleWhileRevalidate");
        assert_eq!(CacheStrategy::NetworkOnly.to_string(), "NetworkOnly");
        assert_eq!(CacheStrategy::CacheOnly.to_string(), "CacheOnly");
    }

    #[test]
    fn test_offline_fallback_config() {
        let config = ServiceWorkerConfig::new()
            .offline_fallback("/offline.html")
            .navigation_preload(true);

        assert_eq!(config.offline_fallback.as_deref(), Some("/offline.html"));
        assert!(config.navigation_preload);
    }

    #[test]
    fn test_sw_config_builder_chaining() {
        let config = ServiceWorkerConfig::new()
            .skip_waiting(false)
            .clients_claim(false);

        assert!(!config.skip_waiting);
        assert!(!config.clients_claim);
    }
}
