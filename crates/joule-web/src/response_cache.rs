//! HTTP response caching.
//!
//! Cache-Control parsing (max-age, s-maxage, no-cache, no-store, must-revalidate,
//! private, public, immutable, stale-while-revalidate, stale-if-error),
//! Vary header handling, cache key generation, freshness calculation,
//! and a simple in-memory cache store. Pure Rust — no HTTP library dependencies.

use std::collections::HashMap;
use std::fmt;

// ── Cache-Control directives ────────────────────────────────────

/// Parsed Cache-Control header directives.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheControl {
    pub max_age: Option<u64>,
    pub s_maxage: Option<u64>,
    pub no_cache: bool,
    pub no_store: bool,
    pub must_revalidate: bool,
    pub proxy_revalidate: bool,
    pub public: bool,
    pub private: bool,
    pub immutable: bool,
    pub no_transform: bool,
    pub stale_while_revalidate: Option<u64>,
    pub stale_if_error: Option<u64>,
}

impl CacheControl {
    /// Parse a Cache-Control header value.
    pub fn parse(header: &str) -> Self {
        let mut cc = Self {
            max_age: None,
            s_maxage: None,
            no_cache: false,
            no_store: false,
            must_revalidate: false,
            proxy_revalidate: false,
            public: false,
            private: false,
            immutable: false,
            no_transform: false,
            stale_while_revalidate: None,
            stale_if_error: None,
        };

        for directive in header.split(',') {
            let trimmed = directive.trim().to_lowercase();
            if let Some(val) = trimmed.strip_prefix("max-age=") {
                cc.max_age = val.trim().parse().ok();
            } else if let Some(val) = trimmed.strip_prefix("s-maxage=") {
                cc.s_maxage = val.trim().parse().ok();
            } else if let Some(val) = trimmed.strip_prefix("stale-while-revalidate=") {
                cc.stale_while_revalidate = val.trim().parse().ok();
            } else if let Some(val) = trimmed.strip_prefix("stale-if-error=") {
                cc.stale_if_error = val.trim().parse().ok();
            } else if trimmed == "no-cache" {
                cc.no_cache = true;
            } else if trimmed == "no-store" {
                cc.no_store = true;
            } else if trimmed == "must-revalidate" {
                cc.must_revalidate = true;
            } else if trimmed == "proxy-revalidate" {
                cc.proxy_revalidate = true;
            } else if trimmed == "public" {
                cc.public = true;
            } else if trimmed == "private" {
                cc.private = true;
            } else if trimmed == "immutable" {
                cc.immutable = true;
            } else if trimmed == "no-transform" {
                cc.no_transform = true;
            }
        }
        cc
    }

    /// Whether this response is cacheable at all.
    pub fn is_cacheable(&self) -> bool {
        !self.no_store
    }

    /// Whether this response requires revalidation before use.
    pub fn requires_revalidation(&self) -> bool {
        self.no_cache || self.must_revalidate
    }

    /// Whether this can be stored in a shared cache.
    pub fn is_shared_cacheable(&self) -> bool {
        self.is_cacheable() && !self.private
    }

    /// Effective max-age for a shared cache (prefers s-maxage).
    pub fn effective_max_age_shared(&self) -> Option<u64> {
        self.s_maxage.or(self.max_age)
    }

    /// Effective max-age for a private cache.
    pub fn effective_max_age_private(&self) -> Option<u64> {
        self.max_age
    }

    /// Serialize back to a Cache-Control header string.
    pub fn to_header(&self) -> String {
        let mut parts = Vec::new();
        if self.public { parts.push("public".to_string()); }
        if self.private { parts.push("private".to_string()); }
        if self.no_cache { parts.push("no-cache".to_string()); }
        if self.no_store { parts.push("no-store".to_string()); }
        if self.must_revalidate { parts.push("must-revalidate".to_string()); }
        if self.proxy_revalidate { parts.push("proxy-revalidate".to_string()); }
        if self.immutable { parts.push("immutable".to_string()); }
        if self.no_transform { parts.push("no-transform".to_string()); }
        if let Some(v) = self.max_age { parts.push(format!("max-age={v}")); }
        if let Some(v) = self.s_maxage { parts.push(format!("s-maxage={v}")); }
        if let Some(v) = self.stale_while_revalidate { parts.push(format!("stale-while-revalidate={v}")); }
        if let Some(v) = self.stale_if_error { parts.push(format!("stale-if-error={v}")); }
        parts.join(", ")
    }
}

impl fmt::Display for CacheControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_header())
    }
}

// ── Vary header ─────────────────────────────────────────────────

/// Parse a Vary header into its field names.
pub fn parse_vary(header: &str) -> Vec<String> {
    header.split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Check if Vary: * is present (uncacheable by downstream).
pub fn vary_is_wildcard(header: &str) -> bool {
    header.split(',').any(|s| s.trim() == "*")
}

// ── Cache key ───────────────────────────────────────────────────

/// Generate a cache key from method, URL, and vary fields.
pub fn cache_key(method: &str, url: &str, vary_fields: &[String], request_headers: &HashMap<String, String>) -> String {
    let mut key = format!("{method}:{url}");
    if !vary_fields.is_empty() {
        key.push_str(":vary(");
        let mut parts = Vec::new();
        for field in vary_fields {
            let lower = field.to_lowercase();
            let val = request_headers.iter()
                .find(|(k, _)| k.to_lowercase() == lower)
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            parts.push(format!("{lower}={val}"));
        }
        parts.sort();
        key.push_str(&parts.join(","));
        key.push(')');
    }
    key
}

// ── Freshness ───────────────────────────────────────────────────

/// Freshness status of a cached response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Response is fresh — can be served as-is.
    Fresh,
    /// Response is stale but within stale-while-revalidate window.
    StaleWhileRevalidate,
    /// Response is stale but within stale-if-error window.
    StaleIfError,
    /// Response is fully stale.
    Stale,
}

/// Calculate freshness of a cached entry.
pub fn check_freshness(
    cc: &CacheControl,
    age_secs: u64,
    is_shared_cache: bool,
) -> Freshness {
    let max_age = if is_shared_cache {
        cc.effective_max_age_shared()
    } else {
        cc.effective_max_age_private()
    };

    if cc.immutable {
        return Freshness::Fresh;
    }

    if cc.requires_revalidation() {
        return Freshness::Stale;
    }

    if let Some(ma) = max_age {
        if age_secs <= ma {
            return Freshness::Fresh;
        }
        let overage = age_secs - ma;
        if let Some(swr) = cc.stale_while_revalidate {
            if overage <= swr {
                return Freshness::StaleWhileRevalidate;
            }
        }
        if let Some(sie) = cc.stale_if_error {
            if overage <= sie {
                return Freshness::StaleIfError;
            }
        }
        return Freshness::Stale;
    }

    Freshness::Stale
}

// ── Cache entry ─────────────────────────────────────────────────

/// A cached response entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub key: String,
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
    pub cache_control: CacheControl,
    pub stored_at_secs: u64,
    pub vary_fields: Vec<String>,
}

impl CacheEntry {
    /// Age of this entry in seconds.
    pub fn age_secs(&self, now_secs: u64) -> u64 {
        now_secs.saturating_sub(self.stored_at_secs)
    }

    /// Check freshness at the given time.
    pub fn freshness(&self, now_secs: u64, is_shared: bool) -> Freshness {
        check_freshness(&self.cache_control, self.age_secs(now_secs), is_shared)
    }
}

// ── In-memory cache store ───────────────────────────────────────

/// Simple in-memory HTTP response cache.
#[derive(Debug)]
pub struct ResponseCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
    is_shared: bool,
    hits: u64,
    misses: u64,
    stale_hits: u64,
}

impl ResponseCache {
    pub fn new(max_entries: usize, is_shared: bool) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            is_shared,
            hits: 0,
            misses: 0,
            stale_hits: 0,
        }
    }

    /// Store a response in the cache.
    pub fn store(&mut self, entry: CacheEntry) -> bool {
        if !entry.cache_control.is_cacheable() {
            return false;
        }
        if self.is_shared && !entry.cache_control.is_shared_cacheable() {
            return false;
        }

        // Evict if at capacity (oldest first by stored_at_secs).
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&entry.key) {
            self.evict_oldest();
        }

        self.entries.insert(entry.key.clone(), entry);
        true
    }

    /// Look up a cached response.
    pub fn get(&mut self, key: &str, now_secs: u64) -> Option<CacheLookup> {
        if let Some(entry) = self.entries.get(key) {
            let freshness = entry.freshness(now_secs, self.is_shared);
            match freshness {
                Freshness::Fresh => {
                    self.hits += 1;
                    Some(CacheLookup { entry: entry.clone(), freshness })
                }
                Freshness::StaleWhileRevalidate | Freshness::StaleIfError => {
                    self.stale_hits += 1;
                    Some(CacheLookup { entry: entry.clone(), freshness })
                }
                Freshness::Stale => {
                    self.misses += 1;
                    None
                }
            }
        } else {
            self.misses += 1;
            None
        }
    }

    /// Invalidate a cache entry.
    pub fn invalidate(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self.entries.iter()
            .min_by_key(|(_, e)| e.stored_at_secs)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest_key);
        }
    }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn hits(&self) -> u64 { self.hits }
    pub fn misses(&self) -> u64 { self.misses }
    pub fn stale_hits(&self) -> u64 { self.stale_hits }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses + self.stale_hits;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// Result of a cache lookup.
#[derive(Debug, Clone)]
pub struct CacheLookup {
    pub entry: CacheEntry,
    pub freshness: Freshness,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_headers() -> HashMap<String, String> { HashMap::new() }

    #[test]
    fn test_parse_max_age() {
        let cc = CacheControl::parse("max-age=3600");
        assert_eq!(cc.max_age, Some(3600));
        assert!(!cc.no_cache);
        assert!(!cc.no_store);
    }

    #[test]
    fn test_parse_no_store() {
        let cc = CacheControl::parse("no-store");
        assert!(cc.no_store);
        assert!(!cc.is_cacheable());
    }

    #[test]
    fn test_parse_no_cache() {
        let cc = CacheControl::parse("no-cache");
        assert!(cc.no_cache);
        assert!(cc.requires_revalidation());
    }

    #[test]
    fn test_parse_must_revalidate() {
        let cc = CacheControl::parse("max-age=0, must-revalidate");
        assert!(cc.must_revalidate);
        assert!(cc.requires_revalidation());
    }

    #[test]
    fn test_parse_combined() {
        let cc = CacheControl::parse("public, max-age=3600, s-maxage=7200, immutable");
        assert!(cc.public);
        assert!(cc.immutable);
        assert_eq!(cc.max_age, Some(3600));
        assert_eq!(cc.s_maxage, Some(7200));
    }

    #[test]
    fn test_parse_stale_while_revalidate() {
        let cc = CacheControl::parse("max-age=300, stale-while-revalidate=60");
        assert_eq!(cc.max_age, Some(300));
        assert_eq!(cc.stale_while_revalidate, Some(60));
    }

    #[test]
    fn test_parse_stale_if_error() {
        let cc = CacheControl::parse("max-age=300, stale-if-error=86400");
        assert_eq!(cc.stale_if_error, Some(86400));
    }

    #[test]
    fn test_parse_private() {
        let cc = CacheControl::parse("private, max-age=600");
        assert!(cc.private);
        assert!(!cc.is_shared_cacheable());
    }

    #[test]
    fn test_parse_proxy_revalidate() {
        let cc = CacheControl::parse("proxy-revalidate");
        assert!(cc.proxy_revalidate);
    }

    #[test]
    fn test_parse_no_transform() {
        let cc = CacheControl::parse("no-transform");
        assert!(cc.no_transform);
    }

    #[test]
    fn test_effective_max_age_shared() {
        let cc = CacheControl::parse("max-age=300, s-maxage=600");
        assert_eq!(cc.effective_max_age_shared(), Some(600));
    }

    #[test]
    fn test_effective_max_age_private() {
        let cc = CacheControl::parse("max-age=300, s-maxage=600");
        assert_eq!(cc.effective_max_age_private(), Some(300));
    }

    #[test]
    fn test_to_header_roundtrip() {
        let cc = CacheControl::parse("public, max-age=3600");
        let header = cc.to_header();
        assert!(header.contains("public"));
        assert!(header.contains("max-age=3600"));
    }

    #[test]
    fn test_display() {
        let cc = CacheControl::parse("no-store");
        assert_eq!(cc.to_string(), "no-store");
    }

    // ── Vary ────────────────────────────────────────────────

    #[test]
    fn test_parse_vary() {
        let fields = parse_vary("Accept-Encoding, Accept-Language");
        assert_eq!(fields, vec!["accept-encoding", "accept-language"]);
    }

    #[test]
    fn test_parse_vary_empty() {
        let fields = parse_vary("");
        assert!(fields.is_empty());
    }

    #[test]
    fn test_vary_wildcard() {
        assert!(vary_is_wildcard("*"));
        assert!(vary_is_wildcard("Accept, *"));
        assert!(!vary_is_wildcard("Accept-Encoding"));
    }

    // ── Cache key ───────────────────────────────────────────

    #[test]
    fn test_cache_key_simple() {
        let key = cache_key("GET", "/api/data", &[], &make_headers());
        assert_eq!(key, "GET:/api/data");
    }

    #[test]
    fn test_cache_key_with_vary() {
        let mut headers = HashMap::new();
        headers.insert("Accept-Encoding".to_string(), "gzip".to_string());
        let vary = vec!["accept-encoding".to_string()];
        let key = cache_key("GET", "/api", &vary, &headers);
        assert!(key.contains("vary(accept-encoding=gzip)"));
    }

    #[test]
    fn test_cache_key_vary_missing_header() {
        let headers = HashMap::new();
        let vary = vec!["accept-encoding".to_string()];
        let key = cache_key("GET", "/api", &vary, &headers);
        assert!(key.contains("accept-encoding="));
    }

    // ── Freshness ───────────────────────────────────────────

    #[test]
    fn test_freshness_fresh() {
        let cc = CacheControl::parse("max-age=3600");
        assert_eq!(check_freshness(&cc, 100, false), Freshness::Fresh);
    }

    #[test]
    fn test_freshness_stale() {
        let cc = CacheControl::parse("max-age=300");
        assert_eq!(check_freshness(&cc, 600, false), Freshness::Stale);
    }

    #[test]
    fn test_freshness_stale_while_revalidate() {
        let cc = CacheControl::parse("max-age=300, stale-while-revalidate=60");
        assert_eq!(check_freshness(&cc, 350, false), Freshness::StaleWhileRevalidate);
        assert_eq!(check_freshness(&cc, 400, false), Freshness::Stale);
    }

    #[test]
    fn test_freshness_stale_if_error() {
        let cc = CacheControl::parse("max-age=300, stale-if-error=600");
        assert_eq!(check_freshness(&cc, 800, false), Freshness::StaleIfError);
        assert_eq!(check_freshness(&cc, 1000, false), Freshness::Stale);
    }

    #[test]
    fn test_freshness_immutable() {
        let cc = CacheControl::parse("max-age=31536000, immutable");
        assert_eq!(check_freshness(&cc, 99999999, false), Freshness::Fresh);
    }

    #[test]
    fn test_freshness_no_cache() {
        let cc = CacheControl::parse("no-cache");
        assert_eq!(check_freshness(&cc, 0, false), Freshness::Stale);
    }

    #[test]
    fn test_freshness_no_max_age() {
        let cc = CacheControl::parse("public");
        assert_eq!(check_freshness(&cc, 0, false), Freshness::Stale);
    }

    #[test]
    fn test_freshness_shared_cache_s_maxage() {
        let cc = CacheControl::parse("max-age=100, s-maxage=500");
        // Shared cache uses s-maxage
        assert_eq!(check_freshness(&cc, 200, true), Freshness::Fresh);
        // Private cache uses max-age
        assert_eq!(check_freshness(&cc, 200, false), Freshness::Stale);
    }

    // ── Cache entry ─────────────────────────────────────────

    #[test]
    fn test_cache_entry_age() {
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("max-age=3600"),
            stored_at_secs: 1000,
            vary_fields: vec![],
        };
        assert_eq!(entry.age_secs(1500), 500);
    }

    #[test]
    fn test_cache_entry_freshness() {
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![1, 2, 3],
            cache_control: CacheControl::parse("max-age=300"),
            stored_at_secs: 1000,
            vary_fields: vec![],
        };
        assert_eq!(entry.freshness(1100, false), Freshness::Fresh);
        assert_eq!(entry.freshness(1400, false), Freshness::Stale);
    }

    // ── Response cache ──────────────────────────────────────

    #[test]
    fn test_cache_store_and_get() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "GET:/api".into(),
            status: 200,
            headers: HashMap::new(),
            body: b"hello".to_vec(),
            cache_control: CacheControl::parse("max-age=3600"),
            stored_at_secs: 1000,
            vary_fields: vec![],
        };
        assert!(cache.store(entry));
        assert_eq!(cache.len(), 1);

        let lookup = cache.get("GET:/api", 1500).unwrap();
        assert_eq!(lookup.freshness, Freshness::Fresh);
        assert_eq!(lookup.entry.body, b"hello");
    }

    #[test]
    fn test_cache_no_store_rejected() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("no-store"),
            stored_at_secs: 0,
            vary_fields: vec![],
        };
        assert!(!cache.store(entry));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_private_rejected_in_shared() {
        let mut cache = ResponseCache::new(10, true);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("private, max-age=300"),
            stored_at_secs: 0,
            vary_fields: vec![],
        };
        assert!(!cache.store(entry));
    }

    #[test]
    fn test_cache_stale_miss() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("max-age=100"),
            stored_at_secs: 1000,
            vary_fields: vec![],
        };
        cache.store(entry);
        assert!(cache.get("k", 1200).is_none()); // stale
        assert_eq!(cache.misses(), 1);
    }

    #[test]
    fn test_cache_stale_while_revalidate_hit() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("max-age=100, stale-while-revalidate=50"),
            stored_at_secs: 1000,
            vary_fields: vec![],
        };
        cache.store(entry);
        let lookup = cache.get("k", 1130).unwrap();
        assert_eq!(lookup.freshness, Freshness::StaleWhileRevalidate);
        assert_eq!(cache.stale_hits(), 1);
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("max-age=3600"),
            stored_at_secs: 0,
            vary_fields: vec![],
        };
        cache.store(entry);
        assert!(cache.invalidate("k"));
        assert!(!cache.invalidate("k"));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = ResponseCache::new(2, false);
        for i in 0..3 {
            let entry = CacheEntry {
                key: format!("k{i}"),
                status: 200,
                headers: HashMap::new(),
                body: vec![],
                cache_control: CacheControl::parse("max-age=3600"),
                stored_at_secs: i as u64 * 100,
                vary_fields: vec![],
            };
            cache.store(entry);
        }
        // Oldest (k0) should have been evicted
        assert_eq!(cache.len(), 2);
        assert!(cache.get("k0", 0).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("max-age=3600"),
            stored_at_secs: 0,
            vary_fields: vec![],
        };
        cache.store(entry);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = ResponseCache::new(10, false);
        let entry = CacheEntry {
            key: "k".into(),
            status: 200,
            headers: HashMap::new(),
            body: vec![],
            cache_control: CacheControl::parse("max-age=3600"),
            stored_at_secs: 0,
            vary_fields: vec![],
        };
        cache.store(entry);
        cache.get("k", 100);   // hit
        cache.get("k", 100);   // hit
        cache.get("miss", 100); // miss

        let rate = cache.hit_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_cache_hit_rate_empty() {
        let cache = ResponseCache::new(10, false);
        assert!((cache.hit_rate() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cache_miss_nonexistent() {
        let mut cache = ResponseCache::new(10, false);
        assert!(cache.get("nope", 0).is_none());
        assert_eq!(cache.misses(), 1);
    }
}
