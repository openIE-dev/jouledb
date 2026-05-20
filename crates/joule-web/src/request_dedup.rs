//! Request deduplication — idempotency key tracking, response caching for
//! duplicate requests, TTL-based cleanup, concurrent request coalescing,
//! dedup statistics, and configurable key extraction.
//!
//! Replaces `express-idempotency`, `idempotent-request`, and similar JS
//! deduplication libraries with a pure-Rust dedup engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Deduplication error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DedupError {
    /// Missing idempotency key.
    MissingKey,
    /// Invalid idempotency key format.
    InvalidKey(String),
    /// Request is already in-flight (coalescing).
    InFlight(String),
    /// Cache full.
    CacheFull { capacity: usize },
    /// Key expired.
    KeyExpired(String),
    /// Conflict: same key used with different request parameters.
    RequestConflict {
        key: String,
        reason: String,
    },
}

impl fmt::Display for DedupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingKey => write!(f, "missing idempotency key"),
            Self::InvalidKey(k) => write!(f, "invalid idempotency key: {k}"),
            Self::InFlight(k) => write!(f, "request in-flight: {k}"),
            Self::CacheFull { capacity } => {
                write!(f, "dedup cache full (capacity: {capacity})")
            }
            Self::KeyExpired(k) => write!(f, "idempotency key expired: {k}"),
            Self::RequestConflict { key, reason } => {
                write!(f, "request conflict for key {key}: {reason}")
            }
        }
    }
}

impl std::error::Error for DedupError {}

// ── Types ────────────────────────────────────────────────────────

/// Status of a dedup entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryStatus {
    /// Request is currently being processed.
    InFlight,
    /// Response has been cached.
    Complete,
    /// Entry has expired.
    Expired,
}

/// A cached response for a completed idempotent request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body.
    pub body: String,
    /// Response headers.
    pub headers: HashMap<String, String>,
}

impl CachedResponse {
    /// Create a simple cached response.
    pub fn new(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_string(),
            headers: HashMap::new(),
        }
    }

    /// Add a header.
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_lowercase(), value.to_string());
        self
    }
}

/// A request fingerprint for conflict detection.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestFingerprint {
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Content hash (hash of the body).
    pub content_hash: u64,
}

impl RequestFingerprint {
    /// Create a fingerprint.
    pub fn new(method: &str, path: &str, body: &[u8]) -> Self {
        Self {
            method: method.to_uppercase(),
            path: path.to_string(),
            content_hash: fnv_hash(body),
        }
    }
}

/// A dedup cache entry.
#[derive(Debug, Clone)]
struct DedupEntry {
    /// Idempotency key.
    key: String,
    /// Current status.
    status: EntryStatus,
    /// Cached response (populated when complete).
    response: Option<CachedResponse>,
    /// Request fingerprint (for conflict detection).
    fingerprint: RequestFingerprint,
    /// When this entry was created (epoch millis).
    created_at: u64,
    /// When this entry expires (epoch millis).
    expires_at: u64,
    /// Number of times a duplicate was detected.
    hit_count: u64,
}

/// Configuration for the dedup store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// TTL for completed entries in milliseconds.
    pub ttl_ms: u64,
    /// TTL for in-flight entries in milliseconds (timeout).
    pub in_flight_ttl_ms: u64,
    /// Whether to detect request conflicts (same key, different body).
    pub detect_conflicts: bool,
    /// Minimum key length.
    pub min_key_length: usize,
    /// Maximum key length.
    pub max_key_length: usize,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl_ms: 86_400_000,  // 24 hours
            in_flight_ttl_ms: 60_000, // 1 minute
            detect_conflicts: true,
            min_key_length: 1,
            max_key_length: 256,
        }
    }
}

/// Deduplication statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupStats {
    /// Total requests processed.
    pub total_requests: u64,
    /// Number of duplicate requests detected.
    pub duplicates_detected: u64,
    /// Number of in-flight coalesced requests.
    pub coalesced: u64,
    /// Number of entries evicted by TTL.
    pub evictions: u64,
    /// Number of conflict errors.
    pub conflicts: u64,
    /// Current cache size.
    pub cache_size: usize,
}

impl DedupStats {
    /// Dedup hit rate as a percentage.
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.duplicates_detected as f64 / self.total_requests as f64) * 100.0
    }
}

// ── Key Extractor ────────────────────────────────────────────────

/// Strategy for extracting the idempotency key from a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeySource {
    /// From a specific header (e.g. "Idempotency-Key").
    Header,
    /// From a query parameter.
    QueryParam,
    /// Auto-generated from request fingerprint.
    Auto,
}

/// Extracts an idempotency key from request data.
#[derive(Debug, Clone)]
pub struct KeyExtractor {
    /// Primary source for the key.
    pub source: KeySource,
    /// Header name to look for (when source is Header).
    pub header_name: String,
    /// Query parameter name (when source is QueryParam).
    pub param_name: String,
}

impl KeyExtractor {
    /// Create a header-based extractor.
    pub fn from_header(header_name: &str) -> Self {
        Self {
            source: KeySource::Header,
            header_name: header_name.to_lowercase(),
            param_name: String::new(),
        }
    }

    /// Create a query-param-based extractor.
    pub fn from_query_param(param_name: &str) -> Self {
        Self {
            source: KeySource::QueryParam,
            header_name: String::new(),
            param_name: param_name.to_string(),
        }
    }

    /// Create an auto extractor (fingerprint-based).
    pub fn auto() -> Self {
        Self {
            source: KeySource::Auto,
            header_name: String::new(),
            param_name: String::new(),
        }
    }

    /// Extract the key from request metadata.
    pub fn extract(
        &self,
        headers: &HashMap<String, String>,
        query: &str,
        fingerprint: &RequestFingerprint,
    ) -> Option<String> {
        match self.source {
            KeySource::Header => {
                headers.get(&self.header_name).cloned()
            }
            KeySource::QueryParam => {
                for pair in query.split('&') {
                    if let Some((key, value)) = pair.split_once('=') {
                        if key == self.param_name {
                            return Some(value.to_string());
                        }
                    }
                }
                None
            }
            KeySource::Auto => {
                Some(format!(
                    "auto-{}-{}-{:016x}",
                    fingerprint.method, fingerprint.path, fingerprint.content_hash
                ))
            }
        }
    }
}

impl Default for KeyExtractor {
    fn default() -> Self {
        Self::from_header("idempotency-key")
    }
}

// ── Dedup Store ──────────────────────────────────────────────────

/// The result of checking a request against the dedup store.
#[derive(Debug, Clone)]
pub enum DedupCheck {
    /// New request — proceed with processing.
    New,
    /// Duplicate — return the cached response.
    Duplicate(CachedResponse),
    /// In-flight — another request with this key is being processed.
    InFlight,
}

/// In-memory deduplication store.
#[derive(Debug, Clone)]
pub struct DedupStore {
    entries: HashMap<String, DedupEntry>,
    config: DedupConfig,
    stats: DedupStats,
}

impl DedupStore {
    /// Create a new store with default configuration.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            config: DedupConfig::default(),
            stats: DedupStats::default(),
        }
    }

    /// Create a store with custom configuration.
    pub fn with_config(config: DedupConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: DedupStats::default(),
        }
    }

    /// Check if a request is a duplicate and register it if not.
    pub fn check(
        &mut self,
        key: &str,
        fingerprint: RequestFingerprint,
        now_ms: u64,
    ) -> Result<DedupCheck, DedupError> {
        self.stats.total_requests += 1;

        // Validate key.
        self.validate_key(key)?;

        // Check for existing entry.
        if let Some(entry) = self.entries.get_mut(key) {
            // Check if expired.
            if now_ms > entry.expires_at {
                // Remove expired entry and treat as new.
                self.entries.remove(key);
                self.stats.evictions += 1;
            } else {
                // Check for conflict.
                if self.config.detect_conflicts && entry.fingerprint != fingerprint {
                    self.stats.conflicts += 1;
                    return Err(DedupError::RequestConflict {
                        key: key.to_string(),
                        reason: "request parameters differ".to_string(),
                    });
                }

                entry.hit_count += 1;

                match entry.status {
                    EntryStatus::InFlight => {
                        self.stats.coalesced += 1;
                        return Ok(DedupCheck::InFlight);
                    }
                    EntryStatus::Complete => {
                        self.stats.duplicates_detected += 1;
                        if let Some(resp) = &entry.response {
                            return Ok(DedupCheck::Duplicate(resp.clone()));
                        }
                    }
                    EntryStatus::Expired => {
                        // Shouldn't get here, but handle gracefully.
                        self.entries.remove(key);
                    }
                }
            }
        }

        // Evict if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.evict_expired(now_ms);
            if self.entries.len() >= self.config.max_entries {
                self.evict_oldest();
            }
        }

        // Register new entry.
        let entry = DedupEntry {
            key: key.to_string(),
            status: EntryStatus::InFlight,
            response: None,
            fingerprint,
            created_at: now_ms,
            expires_at: now_ms + self.config.in_flight_ttl_ms,
            hit_count: 0,
        };
        self.entries.insert(key.to_string(), entry);
        self.stats.cache_size = self.entries.len();

        Ok(DedupCheck::New)
    }

    /// Record the completion of a request, caching the response.
    pub fn complete(
        &mut self,
        key: &str,
        response: CachedResponse,
        now_ms: u64,
    ) -> Result<(), DedupError> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.status = EntryStatus::Complete;
            entry.response = Some(response);
            entry.expires_at = now_ms + self.config.ttl_ms;
            Ok(())
        } else {
            Err(DedupError::KeyExpired(key.to_string()))
        }
    }

    /// Mark a request as failed (remove from in-flight).
    pub fn fail(&mut self, key: &str) {
        self.entries.remove(key);
        self.stats.cache_size = self.entries.len();
    }

    /// Remove an entry.
    pub fn remove(&mut self, key: &str) -> bool {
        let removed = self.entries.remove(key).is_some();
        self.stats.cache_size = self.entries.len();
        removed
    }

    /// Get the current status of a key.
    pub fn status(&self, key: &str, now_ms: u64) -> Option<EntryStatus> {
        self.entries.get(key).map(|e| {
            if now_ms > e.expires_at {
                EntryStatus::Expired
            } else {
                e.status
            }
        })
    }

    /// Get dedup statistics.
    pub fn stats(&self) -> &DedupStats {
        &self.stats
    }

    /// Current cache size.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clean up expired entries.
    pub fn cleanup(&mut self, now_ms: u64) -> usize {
        let before = self.entries.len();
        self.evict_expired(now_ms);
        let removed = before - self.entries.len();
        self.stats.evictions += removed as u64;
        self.stats.cache_size = self.entries.len();
        removed
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats.cache_size = 0;
    }

    /// Get hit count for a key.
    pub fn hit_count(&self, key: &str) -> u64 {
        self.entries.get(key).map(|e| e.hit_count).unwrap_or(0)
    }

    // ── Internal ─────────────────────────────────────────────────

    fn validate_key(&self, key: &str) -> Result<(), DedupError> {
        if key.is_empty() {
            return Err(DedupError::MissingKey);
        }
        if key.len() < self.config.min_key_length {
            return Err(DedupError::InvalidKey(format!(
                "key too short: {} < {}",
                key.len(),
                self.config.min_key_length
            )));
        }
        if key.len() > self.config.max_key_length {
            return Err(DedupError::InvalidKey(format!(
                "key too long: {} > {}",
                key.len(),
                self.config.max_key_length
            )));
        }
        Ok(())
    }

    fn evict_expired(&mut self, now_ms: u64) {
        self.entries.retain(|_, e| now_ms <= e.expires_at);
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest_key);
            self.stats.evictions += 1;
        }
    }
}

impl Default for DedupStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// FNV-1a hash for fingerprinting.
fn fnv_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in data {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fp(method: &str, path: &str, body: &str) -> RequestFingerprint {
        RequestFingerprint::new(method, path, body.as_bytes())
    }

    #[test]
    fn test_new_request() {
        let mut store = DedupStore::new();
        let result = store.check("key-1", fp("POST", "/api/pay", "amount=100"), 1000).unwrap();
        assert!(matches!(result, DedupCheck::New));
    }

    #[test]
    fn test_duplicate_in_flight() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        let result = store.check("key-1", fp("POST", "/pay", "100"), 1001).unwrap();
        assert!(matches!(result, DedupCheck::InFlight));
    }

    #[test]
    fn test_duplicate_complete() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        store.complete("key-1", CachedResponse::new(200, "ok"), 1001).unwrap();

        let result = store.check("key-1", fp("POST", "/pay", "100"), 1002).unwrap();
        match result {
            DedupCheck::Duplicate(resp) => {
                assert_eq!(resp.status, 200);
                assert_eq!(resp.body, "ok");
            }
            _ => panic!("expected Duplicate"),
        }
    }

    #[test]
    fn test_conflict_detection() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "amount=100"), 1000).unwrap();
        store.complete("key-1", CachedResponse::new(200, "ok"), 1001).unwrap();

        let err = store
            .check("key-1", fp("POST", "/pay", "amount=200"), 1002)
            .unwrap_err();
        assert!(matches!(err, DedupError::RequestConflict { .. }));
    }

    #[test]
    fn test_expired_entry_treated_as_new() {
        let mut store = DedupStore::with_config(DedupConfig {
            ttl_ms: 1000,
            ..Default::default()
        });
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        store.complete("key-1", CachedResponse::new(200, "ok"), 1000).unwrap();

        // After TTL expires.
        let result = store.check("key-1", fp("POST", "/pay", "100"), 3000).unwrap();
        assert!(matches!(result, DedupCheck::New));
    }

    #[test]
    fn test_fail_removes_entry() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        store.fail("key-1");
        assert!(store.is_empty());
    }

    #[test]
    fn test_missing_key() {
        let mut store = DedupStore::new();
        let err = store.check("", fp("POST", "/pay", "100"), 1000).unwrap_err();
        assert_eq!(err, DedupError::MissingKey);
    }

    #[test]
    fn test_key_too_long() {
        let mut store = DedupStore::with_config(DedupConfig {
            max_key_length: 10,
            ..Default::default()
        });
        let long_key = "a".repeat(20);
        let err = store.check(&long_key, fp("POST", "/pay", "100"), 1000).unwrap_err();
        assert!(matches!(err, DedupError::InvalidKey(_)));
    }

    #[test]
    fn test_cleanup() {
        let mut store = DedupStore::with_config(DedupConfig {
            ttl_ms: 1000,
            in_flight_ttl_ms: 500,
            ..Default::default()
        });
        store.check("key-1", fp("POST", "/a", "1"), 1000).unwrap();
        store.complete("key-1", CachedResponse::new(200, "ok"), 1000).unwrap();
        store.check("key-2", fp("POST", "/b", "2"), 1000).unwrap();
        store.complete("key-2", CachedResponse::new(200, "ok"), 1000).unwrap();

        assert_eq!(store.len(), 2);
        let removed = store.cleanup(3000);
        assert_eq!(removed, 2);
        assert!(store.is_empty());
    }

    #[test]
    fn test_stats() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        store.complete("key-1", CachedResponse::new(200, "ok"), 1001).unwrap();
        store.check("key-1", fp("POST", "/pay", "100"), 1002).unwrap();

        let stats = store.stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.duplicates_detected, 1);
        assert!(stats.hit_rate() > 0.0);
    }

    #[test]
    fn test_stats_coalesced() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        store.check("key-1", fp("POST", "/pay", "100"), 1001).unwrap();

        assert_eq!(store.stats().coalesced, 1);
    }

    #[test]
    fn test_key_extractor_header() {
        let extractor = KeyExtractor::from_header("Idempotency-Key");
        let mut headers = HashMap::new();
        headers.insert("idempotency-key".to_string(), "my-key-123".to_string());
        let key = extractor.extract(&headers, "", &fp("POST", "/", "")).unwrap();
        assert_eq!(key, "my-key-123");
    }

    #[test]
    fn test_key_extractor_query() {
        let extractor = KeyExtractor::from_query_param("idem_key");
        let headers = HashMap::new();
        let key = extractor.extract(&headers, "idem_key=abc&other=val", &fp("POST", "/", "")).unwrap();
        assert_eq!(key, "abc");
    }

    #[test]
    fn test_key_extractor_auto() {
        let extractor = KeyExtractor::auto();
        let headers = HashMap::new();
        let key = extractor.extract(&headers, "", &fp("POST", "/api/pay", "amount=100")).unwrap();
        assert!(key.starts_with("auto-POST-/api/pay-"));
    }

    #[test]
    fn test_key_extractor_missing() {
        let extractor = KeyExtractor::from_header("Idempotency-Key");
        let headers = HashMap::new();
        assert!(extractor.extract(&headers, "", &fp("GET", "/", "")).is_none());
    }

    #[test]
    fn test_cached_response_with_header() {
        let resp = CachedResponse::new(201, "created")
            .with_header("Content-Type", "application/json");
        assert_eq!(resp.headers.get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn test_remove_entry() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/a", "1"), 1000).unwrap();
        assert!(store.remove("key-1"));
        assert!(!store.remove("key-1"));
    }

    #[test]
    fn test_status() {
        let mut store = DedupStore::with_config(DedupConfig {
            ttl_ms: 1000,
            ..Default::default()
        });
        store.check("key-1", fp("POST", "/a", "1"), 1000).unwrap();
        assert_eq!(store.status("key-1", 1000), Some(EntryStatus::InFlight));

        store.complete("key-1", CachedResponse::new(200, "ok"), 1001).unwrap();
        assert_eq!(store.status("key-1", 1001), Some(EntryStatus::Complete));

        assert_eq!(store.status("key-1", 5000), Some(EntryStatus::Expired));
        assert_eq!(store.status("nonexistent", 1000), None);
    }

    #[test]
    fn test_hit_count() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/pay", "100"), 1000).unwrap();
        store.complete("key-1", CachedResponse::new(200, "ok"), 1001).unwrap();

        assert_eq!(store.hit_count("key-1"), 0);
        store.check("key-1", fp("POST", "/pay", "100"), 1002).unwrap();
        assert_eq!(store.hit_count("key-1"), 1);
        store.check("key-1", fp("POST", "/pay", "100"), 1003).unwrap();
        assert_eq!(store.hit_count("key-1"), 2);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut store = DedupStore::with_config(DedupConfig {
            max_entries: 2,
            ttl_ms: 100_000,
            in_flight_ttl_ms: 100_000,
            ..Default::default()
        });
        store.check("key-1", fp("POST", "/a", "1"), 1000).unwrap();
        store.check("key-2", fp("POST", "/b", "2"), 2000).unwrap();
        // This should evict key-1 (oldest).
        store.check("key-3", fp("POST", "/c", "3"), 3000).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.status("key-1", 3000).is_none());
    }

    #[test]
    fn test_hit_rate_zero() {
        let stats = DedupStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_error_display() {
        let err = DedupError::CacheFull { capacity: 1000 };
        assert!(err.to_string().contains("1000"));
    }

    #[test]
    fn test_clear() {
        let mut store = DedupStore::new();
        store.check("key-1", fp("POST", "/a", "1"), 1000).unwrap();
        store.check("key-2", fp("POST", "/b", "2"), 1000).unwrap();
        store.clear();
        assert!(store.is_empty());
    }
}
