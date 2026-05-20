//! HTTP caching — Cache-Control header parsing/generation, ETag matching
//! (strong/weak), If-None-Match/If-Modified-Since, max-age/s-maxage,
//! private/public/no-cache/no-store, Vary header, and cache key generation.
//!
//! Replaces `http-cache-semantics`, `apicache`, and similar JS caching
//! libraries with a pure-Rust HTTP cache engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// HTTP cache error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Invalid Cache-Control directive.
    InvalidDirective(String),
    /// Invalid ETag format.
    InvalidETag(String),
    /// Cache miss.
    CacheMiss(String),
    /// Stale entry.
    StaleEntry { key: String, age_secs: u64, max_age: u64 },
    /// Parse error.
    ParseError(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDirective(d) => write!(f, "invalid Cache-Control directive: {d}"),
            Self::InvalidETag(e) => write!(f, "invalid ETag: {e}"),
            Self::CacheMiss(k) => write!(f, "cache miss: {k}"),
            Self::StaleEntry { key, age_secs, max_age } => {
                write!(f, "stale: {key} age={age_secs}s max-age={max_age}s")
            }
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for CacheError {}

// ── Cache-Control ────────────────────────────────────────────────

/// Parsed Cache-Control header.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    /// max-age in seconds.
    pub max_age: Option<u64>,
    /// s-maxage in seconds (shared/proxy caches).
    pub s_maxage: Option<u64>,
    /// no-cache: must revalidate before using cached copy.
    pub no_cache: bool,
    /// no-store: must not store any part of the response.
    pub no_store: bool,
    /// public: any cache may store.
    pub public: bool,
    /// private: only browser cache may store.
    pub private: bool,
    /// must-revalidate.
    pub must_revalidate: bool,
    /// proxy-revalidate.
    pub proxy_revalidate: bool,
    /// no-transform.
    pub no_transform: bool,
    /// immutable.
    pub immutable: bool,
    /// stale-while-revalidate in seconds.
    pub stale_while_revalidate: Option<u64>,
    /// stale-if-error in seconds.
    pub stale_if_error: Option<u64>,
}

impl CacheControl {
    /// Parse a Cache-Control header value.
    pub fn parse(header: &str) -> Result<Self, CacheError> {
        let mut cc = CacheControl::default();

        for directive in header.split(',') {
            let trimmed = directive.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Split on '=' for directives with values.
            if let Some((key, value)) = trimmed.split_once('=') {
                let key = key.trim().to_lowercase();
                let value = value.trim().trim_matches('"');
                match key.as_str() {
                    "max-age" => {
                        cc.max_age = Some(
                            value.parse::<u64>().map_err(|_| {
                                CacheError::InvalidDirective(trimmed.to_string())
                            })?,
                        );
                    }
                    "s-maxage" => {
                        cc.s_maxage = Some(
                            value.parse::<u64>().map_err(|_| {
                                CacheError::InvalidDirective(trimmed.to_string())
                            })?,
                        );
                    }
                    "stale-while-revalidate" => {
                        cc.stale_while_revalidate = Some(
                            value.parse::<u64>().map_err(|_| {
                                CacheError::InvalidDirective(trimmed.to_string())
                            })?,
                        );
                    }
                    "stale-if-error" => {
                        cc.stale_if_error = Some(
                            value.parse::<u64>().map_err(|_| {
                                CacheError::InvalidDirective(trimmed.to_string())
                            })?,
                        );
                    }
                    _ => {
                        // Unknown directives with values are ignored per spec.
                    }
                }
            } else {
                match trimmed.to_lowercase().as_str() {
                    "no-cache" => cc.no_cache = true,
                    "no-store" => cc.no_store = true,
                    "public" => cc.public = true,
                    "private" => cc.private = true,
                    "must-revalidate" => cc.must_revalidate = true,
                    "proxy-revalidate" => cc.proxy_revalidate = true,
                    "no-transform" => cc.no_transform = true,
                    "immutable" => cc.immutable = true,
                    _ => {
                        // Unknown directives are ignored per spec.
                    }
                }
            }
        }

        Ok(cc)
    }

    /// Generate a Cache-Control header string.
    pub fn to_header(&self) -> String {
        let mut parts = Vec::new();

        if self.public {
            parts.push("public".to_string());
        }
        if self.private {
            parts.push("private".to_string());
        }
        if self.no_cache {
            parts.push("no-cache".to_string());
        }
        if self.no_store {
            parts.push("no-store".to_string());
        }
        if let Some(max_age) = self.max_age {
            parts.push(format!("max-age={max_age}"));
        }
        if let Some(s_maxage) = self.s_maxage {
            parts.push(format!("s-maxage={s_maxage}"));
        }
        if self.must_revalidate {
            parts.push("must-revalidate".to_string());
        }
        if self.proxy_revalidate {
            parts.push("proxy-revalidate".to_string());
        }
        if self.no_transform {
            parts.push("no-transform".to_string());
        }
        if self.immutable {
            parts.push("immutable".to_string());
        }
        if let Some(swr) = self.stale_while_revalidate {
            parts.push(format!("stale-while-revalidate={swr}"));
        }
        if let Some(sie) = self.stale_if_error {
            parts.push(format!("stale-if-error={sie}"));
        }

        parts.join(", ")
    }

    /// Whether this response can be cached at all.
    pub fn is_cacheable(&self) -> bool {
        !self.no_store
    }

    /// Whether this response requires revalidation before use.
    pub fn requires_revalidation(&self) -> bool {
        self.no_cache || self.must_revalidate
    }

    /// Effective max-age for the given cache type.
    pub fn effective_max_age(&self, is_shared: bool) -> Option<u64> {
        if is_shared {
            self.s_maxage.or(self.max_age)
        } else {
            self.max_age
        }
    }
}

impl fmt::Display for CacheControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_header())
    }
}

// ── ETag ─────────────────────────────────────────────────────────

/// An HTTP ETag (entity tag).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ETag {
    /// The tag value (without quotes or W/ prefix).
    pub value: String,
    /// Whether this is a weak validator.
    pub weak: bool,
}

impl ETag {
    /// Create a strong ETag.
    pub fn strong(value: &str) -> Self {
        Self { value: value.to_string(), weak: false }
    }

    /// Create a weak ETag.
    pub fn weak(value: &str) -> Self {
        Self { value: value.to_string(), weak: true }
    }

    /// Parse an ETag from a header value.
    pub fn parse(header: &str) -> Result<Self, CacheError> {
        let trimmed = header.trim();
        if let Some(rest) = trimmed.strip_prefix("W/") {
            let val = rest.trim_matches('"');
            if val.is_empty() {
                return Err(CacheError::InvalidETag(header.to_string()));
            }
            Ok(Self { value: val.to_string(), weak: true })
        } else {
            let val = trimmed.trim_matches('"');
            if val.is_empty() {
                return Err(CacheError::InvalidETag(header.to_string()));
            }
            Ok(Self { value: val.to_string(), weak: false })
        }
    }

    /// Format as a header value.
    pub fn to_header(&self) -> String {
        if self.weak {
            format!("W/\"{}\"", self.value)
        } else {
            format!("\"{}\"", self.value)
        }
    }

    /// Strong comparison (RFC 7232 section 2.3.2).
    pub fn strong_match(&self, other: &ETag) -> bool {
        !self.weak && !other.weak && self.value == other.value
    }

    /// Weak comparison (RFC 7232 section 2.3.2).
    pub fn weak_match(&self, other: &ETag) -> bool {
        self.value == other.value
    }

    /// Generate an ETag from content bytes (simple hash).
    pub fn from_content(content: &[u8]) -> Self {
        let hash = content_hash(content);
        Self::strong(&hash)
    }
}

impl fmt::Display for ETag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_header())
    }
}

/// Parse an If-None-Match header (comma-separated list of ETags, or "*").
pub fn parse_if_none_match(header: &str) -> Result<Vec<ETag>, CacheError> {
    let trimmed = header.trim();
    if trimmed == "*" {
        return Ok(Vec::new()); // Wildcard = match any
    }

    let mut tags = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        tags.push(ETag::parse(part)?);
    }
    Ok(tags)
}

/// Check if an If-None-Match header matches a given ETag (returns true if
/// the response would be 304 Not Modified).
pub fn etag_matches(if_none_match: &str, etag: &ETag) -> Result<bool, CacheError> {
    let trimmed = if_none_match.trim();
    if trimmed == "*" {
        return Ok(true);
    }

    let tags = parse_if_none_match(if_none_match)?;
    for tag in &tags {
        if tag.weak_match(etag) {
            return Ok(true);
        }
    }
    Ok(false)
}

// ── Vary Header ──────────────────────────────────────────────────

/// Parsed Vary header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VaryHeader {
    /// Header names that affect caching. Empty + wildcard=true means Vary: *.
    pub headers: Vec<String>,
    /// Whether this is a wildcard Vary (effectively uncacheable).
    pub wildcard: bool,
}

impl VaryHeader {
    /// Parse a Vary header value.
    pub fn parse(header: &str) -> Self {
        let trimmed = header.trim();
        if trimmed == "*" {
            return Self { headers: Vec::new(), wildcard: true };
        }
        let headers: Vec<String> = trimmed
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect();
        Self { headers, wildcard: false }
    }

    /// Format as a Vary header value.
    pub fn to_header(&self) -> String {
        if self.wildcard {
            "*".to_string()
        } else {
            self.headers.join(", ")
        }
    }

    /// Check if this Vary makes the response effectively uncacheable.
    pub fn is_uncacheable(&self) -> bool {
        self.wildcard
    }
}

// ── Cache Key ────────────────────────────────────────────────────

/// Generate a cache key from request properties.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Request method.
    pub method: String,
    /// Request URL/path.
    pub url: String,
    /// Vary header values that differentiate this cache entry.
    pub vary_values: Vec<(String, String)>,
}

impl CacheKey {
    /// Create a simple cache key from method and URL.
    pub fn new(method: &str, url: &str) -> Self {
        Self {
            method: method.to_uppercase(),
            url: url.to_string(),
            vary_values: Vec::new(),
        }
    }

    /// Create a cache key with Vary header values.
    pub fn with_vary(
        method: &str,
        url: &str,
        vary: &VaryHeader,
        request_headers: &HashMap<String, String>,
    ) -> Self {
        let mut vary_values: Vec<(String, String)> = vary
            .headers
            .iter()
            .map(|h| {
                let val = request_headers
                    .get(&h.to_lowercase())
                    .cloned()
                    .unwrap_or_default();
                (h.clone(), val)
            })
            .collect();
        vary_values.sort_by(|a, b| a.0.cmp(&b.0));

        Self {
            method: method.to_uppercase(),
            url: url.to_string(),
            vary_values,
        }
    }

    /// Compute a string key for use in a hash map.
    pub fn to_key_string(&self) -> String {
        let mut key = format!("{}:{}", self.method, self.url);
        for (h, v) in &self.vary_values {
            key.push_str(&format!("|{h}={v}"));
        }
        key
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_key_string())
    }
}

// ── Cache Entry ──────────────────────────────────────────────────

/// A cached response entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Cache key.
    pub key: CacheKey,
    /// Response status code.
    pub status: u16,
    /// Response headers.
    pub headers: HashMap<String, String>,
    /// Response body.
    pub body: String,
    /// ETag.
    pub etag: Option<ETag>,
    /// Cache-Control from the response.
    pub cache_control: CacheControl,
    /// Vary header from the response.
    pub vary: Option<VaryHeader>,
    /// Timestamp when this entry was stored (epoch seconds).
    pub stored_at: u64,
    /// Last-Modified value (epoch seconds).
    pub last_modified: Option<u64>,
}

impl CacheEntry {
    /// Check if this entry is fresh given the current time.
    pub fn is_fresh(&self, now_epoch_secs: u64, is_shared: bool) -> bool {
        if self.cache_control.no_store {
            return false;
        }
        if let Some(max_age) = self.cache_control.effective_max_age(is_shared) {
            let age = now_epoch_secs.saturating_sub(self.stored_at);
            return age <= max_age;
        }
        false
    }

    /// Age of this entry in seconds.
    pub fn age(&self, now_epoch_secs: u64) -> u64 {
        now_epoch_secs.saturating_sub(self.stored_at)
    }

    /// Whether this entry can be served stale while revalidating.
    pub fn can_serve_stale(&self, now_epoch_secs: u64, is_shared: bool) -> bool {
        if let Some(max_age) = self.cache_control.effective_max_age(is_shared) {
            if let Some(swr) = self.cache_control.stale_while_revalidate {
                let age = self.age(now_epoch_secs);
                return age <= max_age + swr;
            }
        }
        false
    }
}

// ── Cache Store ──────────────────────────────────────────────────

/// In-memory HTTP cache store.
#[derive(Debug, Clone)]
pub struct CacheStore {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
}

impl CacheStore {
    /// Create a new cache store with a max entry count.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
        }
    }

    /// Store a cache entry.
    pub fn put(&mut self, entry: CacheEntry) {
        let key = entry.key.to_key_string();
        self.entries.insert(key, entry);

        // Evict oldest if over capacity.
        while self.entries.len() > self.max_entries {
            let oldest_key = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.stored_at)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                self.entries.remove(&k);
            } else {
                break;
            }
        }
    }

    /// Get a cache entry.
    pub fn get(&self, key: &CacheKey) -> Option<&CacheEntry> {
        self.entries.get(&key.to_key_string())
    }

    /// Remove a cache entry.
    pub fn remove(&mut self, key: &CacheKey) -> bool {
        self.entries.remove(&key.to_key_string()).is_some()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Purge stale entries.
    pub fn purge_stale(&mut self, now_epoch_secs: u64, is_shared: bool) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|_, e| e.is_fresh(now_epoch_secs, is_shared));
        before - self.entries.len()
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Simple content hash for ETag generation (FNV-1a).
fn content_hash(data: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in data {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cache_control_simple() {
        let cc = CacheControl::parse("public, max-age=3600").unwrap();
        assert!(cc.public);
        assert_eq!(cc.max_age, Some(3600));
    }

    #[test]
    fn test_parse_cache_control_no_store() {
        let cc = CacheControl::parse("no-store, no-cache").unwrap();
        assert!(cc.no_store);
        assert!(cc.no_cache);
        assert!(!cc.is_cacheable());
    }

    #[test]
    fn test_parse_cache_control_s_maxage() {
        let cc = CacheControl::parse("public, s-maxage=600, max-age=3600").unwrap();
        assert_eq!(cc.s_maxage, Some(600));
        assert_eq!(cc.effective_max_age(true), Some(600));
        assert_eq!(cc.effective_max_age(false), Some(3600));
    }

    #[test]
    fn test_parse_cache_control_all_directives() {
        let cc = CacheControl::parse(
            "private, must-revalidate, no-transform, immutable, stale-while-revalidate=60"
        ).unwrap();
        assert!(cc.private);
        assert!(cc.must_revalidate);
        assert!(cc.no_transform);
        assert!(cc.immutable);
        assert_eq!(cc.stale_while_revalidate, Some(60));
    }

    #[test]
    fn test_cache_control_to_header() {
        let cc = CacheControl {
            max_age: Some(3600),
            public: true,
            ..Default::default()
        };
        let header = cc.to_header();
        assert!(header.contains("public"));
        assert!(header.contains("max-age=3600"));
    }

    #[test]
    fn test_cache_control_roundtrip() {
        let original = "public, max-age=3600, must-revalidate";
        let cc = CacheControl::parse(original).unwrap();
        let regenerated = cc.to_header();
        let reparsed = CacheControl::parse(&regenerated).unwrap();
        assert_eq!(cc, reparsed);
    }

    #[test]
    fn test_etag_strong() {
        let etag = ETag::strong("abc123");
        assert!(!etag.weak);
        assert_eq!(etag.to_header(), "\"abc123\"");
    }

    #[test]
    fn test_etag_weak() {
        let etag = ETag::weak("abc123");
        assert!(etag.weak);
        assert_eq!(etag.to_header(), "W/\"abc123\"");
    }

    #[test]
    fn test_etag_parse_strong() {
        let etag = ETag::parse("\"abc123\"").unwrap();
        assert!(!etag.weak);
        assert_eq!(etag.value, "abc123");
    }

    #[test]
    fn test_etag_parse_weak() {
        let etag = ETag::parse("W/\"abc123\"").unwrap();
        assert!(etag.weak);
        assert_eq!(etag.value, "abc123");
    }

    #[test]
    fn test_etag_strong_match() {
        let a = ETag::strong("x");
        let b = ETag::strong("x");
        let c = ETag::weak("x");
        assert!(a.strong_match(&b));
        assert!(!a.strong_match(&c));
    }

    #[test]
    fn test_etag_weak_match() {
        let a = ETag::strong("x");
        let b = ETag::weak("x");
        assert!(a.weak_match(&b));
    }

    #[test]
    fn test_etag_from_content() {
        let etag1 = ETag::from_content(b"hello");
        let etag2 = ETag::from_content(b"hello");
        let etag3 = ETag::from_content(b"world");
        assert_eq!(etag1.value, etag2.value);
        assert_ne!(etag1.value, etag3.value);
    }

    #[test]
    fn test_if_none_match() {
        let etag = ETag::strong("abc");
        assert!(etag_matches("\"abc\"", &etag).unwrap());
        assert!(etag_matches("\"xyz\", \"abc\"", &etag).unwrap());
        assert!(!etag_matches("\"xyz\"", &etag).unwrap());
        assert!(etag_matches("*", &etag).unwrap());
    }

    #[test]
    fn test_vary_header() {
        let vary = VaryHeader::parse("Accept-Encoding, Accept-Language");
        assert_eq!(vary.headers, vec!["accept-encoding", "accept-language"]);
        assert!(!vary.is_uncacheable());
    }

    #[test]
    fn test_vary_wildcard() {
        let vary = VaryHeader::parse("*");
        assert!(vary.wildcard);
        assert!(vary.is_uncacheable());
    }

    #[test]
    fn test_cache_key_simple() {
        let key = CacheKey::new("GET", "/api/users");
        assert_eq!(key.to_key_string(), "GET:/api/users");
    }

    #[test]
    fn test_cache_key_with_vary() {
        let vary = VaryHeader::parse("Accept-Encoding");
        let mut headers = HashMap::new();
        headers.insert("accept-encoding".to_string(), "gzip".to_string());
        let key = CacheKey::with_vary("GET", "/api", &vary, &headers);
        let key_str = key.to_key_string();
        assert!(key_str.contains("accept-encoding=gzip"));
    }

    #[test]
    fn test_cache_entry_freshness() {
        let entry = CacheEntry {
            key: CacheKey::new("GET", "/test"),
            status: 200,
            headers: HashMap::new(),
            body: "data".to_string(),
            etag: None,
            cache_control: CacheControl {
                max_age: Some(3600),
                ..Default::default()
            },
            vary: None,
            stored_at: 1000,
            last_modified: None,
        };
        assert!(entry.is_fresh(2000, false)); // age=1000s < 3600s
        assert!(!entry.is_fresh(5000, false)); // age=4000s > 3600s
    }

    #[test]
    fn test_cache_store_basic() {
        let mut store = CacheStore::new(10);
        assert!(store.is_empty());

        let entry = CacheEntry {
            key: CacheKey::new("GET", "/test"),
            status: 200,
            headers: HashMap::new(),
            body: "data".to_string(),
            etag: None,
            cache_control: CacheControl::default(),
            vary: None,
            stored_at: 1000,
            last_modified: None,
        };
        store.put(entry);
        assert_eq!(store.len(), 1);

        let key = CacheKey::new("GET", "/test");
        assert!(store.get(&key).is_some());
    }

    #[test]
    fn test_cache_store_eviction() {
        let mut store = CacheStore::new(2);
        for i in 0..3 {
            let entry = CacheEntry {
                key: CacheKey::new("GET", &format!("/test/{i}")),
                status: 200,
                headers: HashMap::new(),
                body: format!("data{i}"),
                etag: None,
                cache_control: CacheControl::default(),
                vary: None,
                stored_at: i as u64,
                last_modified: None,
            };
            store.put(entry);
        }
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_cache_store_remove() {
        let mut store = CacheStore::new(10);
        let entry = CacheEntry {
            key: CacheKey::new("GET", "/test"),
            status: 200,
            headers: HashMap::new(),
            body: "data".to_string(),
            etag: None,
            cache_control: CacheControl::default(),
            vary: None,
            stored_at: 1000,
            last_modified: None,
        };
        store.put(entry);
        let key = CacheKey::new("GET", "/test");
        assert!(store.remove(&key));
        assert!(store.is_empty());
    }

    #[test]
    fn test_stale_while_revalidate() {
        let entry = CacheEntry {
            key: CacheKey::new("GET", "/test"),
            status: 200,
            headers: HashMap::new(),
            body: "data".to_string(),
            etag: None,
            cache_control: CacheControl {
                max_age: Some(100),
                stale_while_revalidate: Some(50),
                ..Default::default()
            },
            vary: None,
            stored_at: 0,
            last_modified: None,
        };
        assert!(entry.is_fresh(50, false));
        assert!(!entry.is_fresh(120, false));
        assert!(entry.can_serve_stale(120, false));
        assert!(!entry.can_serve_stale(200, false));
    }

    #[test]
    fn test_purge_stale() {
        let mut store = CacheStore::new(10);
        let fresh = CacheEntry {
            key: CacheKey::new("GET", "/fresh"),
            status: 200,
            headers: HashMap::new(),
            body: "data".to_string(),
            etag: None,
            cache_control: CacheControl {
                max_age: Some(3600),
                ..Default::default()
            },
            vary: None,
            stored_at: 1000,
            last_modified: None,
        };
        let stale = CacheEntry {
            key: CacheKey::new("GET", "/stale"),
            status: 200,
            headers: HashMap::new(),
            body: "old".to_string(),
            etag: None,
            cache_control: CacheControl {
                max_age: Some(100),
                ..Default::default()
            },
            vary: None,
            stored_at: 0,
            last_modified: None,
        };
        store.put(fresh);
        store.put(stale);
        let purged = store.purge_stale(1500, false);
        assert_eq!(purged, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_requires_revalidation() {
        let cc = CacheControl {
            no_cache: true,
            ..Default::default()
        };
        assert!(cc.requires_revalidation());
    }

    #[test]
    fn test_error_display() {
        let err = CacheError::CacheMiss("key1".to_string());
        assert!(err.to_string().contains("key1"));
    }
}
