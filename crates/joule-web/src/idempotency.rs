//! Idempotency — idempotency key management, response caching, duplicate
//! detection, TTL-based cleanup, and key fingerprinting.
//!
//! Replaces JS idempotency libraries (Stripe idempotency, express-idempotency)
//! with a pure-Rust idempotency layer that tracks energy per operation.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Idempotency errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdempotencyError {
    /// Key not found.
    KeyNotFound(String),
    /// Request is already in flight for this key.
    InFlight(String),
    /// Key has expired.
    KeyExpired(String),
    /// Fingerprint mismatch (same key, different request body).
    FingerprintMismatch { key: String, expected: String, got: String },
    /// Key already completed — use cached response.
    AlreadyCompleted(String),
    /// Invalid TTL.
    InvalidTtl(String),
}

impl std::fmt::Display for IdempotencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotFound(k) => write!(f, "idempotency key not found: {k}"),
            Self::InFlight(k) => write!(f, "request in flight for key: {k}"),
            Self::KeyExpired(k) => write!(f, "idempotency key expired: {k}"),
            Self::FingerprintMismatch { key, expected, got } => {
                write!(
                    f,
                    "fingerprint mismatch for key {key}: expected {expected}, got {got}"
                )
            }
            Self::AlreadyCompleted(k) => write!(f, "key already completed: {k}"),
            Self::InvalidTtl(msg) => write!(f, "invalid TTL: {msg}"),
        }
    }
}

impl std::error::Error for IdempotencyError {}

// ── Key State ───────────────────────────────────────────────────

/// State of an idempotency key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyState {
    /// Request is currently being processed.
    InFlight,
    /// Request completed successfully.
    Completed,
    /// Request failed.
    Failed,
    /// Key expired (available for reuse).
    Expired,
}

// ── Idempotency Record ─────────────────────────────────────────

/// A stored idempotency record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    pub key: String,
    /// Fingerprint of the request body.
    pub fingerprint: String,
    pub state: KeyState,
    /// Cached response status code (if completed).
    pub response_status: Option<u16>,
    /// Cached response body (if completed).
    pub response_body: Option<serde_json::Value>,
    /// Cached response headers (if completed).
    pub response_headers: Option<HashMap<String, String>>,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub expires_at: DateTime<Utc>,
    /// Number of duplicate requests detected.
    pub duplicate_count: u64,
}

/// Configuration for the idempotency store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdempotencyConfig {
    /// Default TTL in seconds.
    pub default_ttl_seconds: u64,
    /// Whether to enforce fingerprint matching.
    pub enforce_fingerprint: bool,
    /// Maximum number of keys to store.
    pub max_keys: usize,
}

impl Default for IdempotencyConfig {
    fn default() -> Self {
        Self {
            default_ttl_seconds: 86400, // 24 hours.
            enforce_fingerprint: true,
            max_keys: 100_000,
        }
    }
}

// ── Fingerprinting ──────────────────────────────────────────────

/// Compute a simple fingerprint from request body bytes.
/// Uses a FNV-1a inspired hash for speed.
pub fn compute_fingerprint(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Compute a fingerprint from a JSON value.
pub fn fingerprint_json(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    compute_fingerprint(&bytes)
}

// ── Idempotency Store ───────────────────────────────────────────

/// The idempotency store managing keys and cached responses.
#[derive(Debug, Clone)]
pub struct IdempotencyStore {
    config: IdempotencyConfig,
    records: HashMap<String, IdempotencyRecord>,
    total_energy_uj: u64,
}

impl IdempotencyStore {
    pub fn new(config: IdempotencyConfig) -> Self {
        Self {
            config,
            records: HashMap::new(),
            total_energy_uj: 0,
        }
    }

    /// Begin processing a request with the given idempotency key.
    ///
    /// Returns `Ok(None)` if this is a new request.
    /// Returns `Ok(Some(record))` if there's a cached response.
    /// Returns `Err` on fingerprint mismatch or in-flight conflict.
    pub fn begin(
        &mut self,
        key: &str,
        fingerprint: &str,
        ttl_seconds: Option<u64>,
    ) -> Result<Option<IdempotencyRecord>, IdempotencyError> {
        self.total_energy_uj += 5;

        let ttl = ttl_seconds.unwrap_or(self.config.default_ttl_seconds);
        if ttl == 0 {
            return Err(IdempotencyError::InvalidTtl("TTL must be > 0".into()));
        }

        // Check for existing record.
        if let Some(record) = self.records.get_mut(key) {
            // Check expiration.
            if Utc::now() >= record.expires_at {
                record.state = KeyState::Expired;
                // Allow reuse of expired key.
                self.records.remove(key);
                // Fall through to create new record.
            } else {
                // Fingerprint check.
                if self.config.enforce_fingerprint && record.fingerprint != fingerprint {
                    return Err(IdempotencyError::FingerprintMismatch {
                        key: key.to_string(),
                        expected: record.fingerprint.clone(),
                        got: fingerprint.to_string(),
                    });
                }

                match &record.state {
                    KeyState::InFlight => {
                        return Err(IdempotencyError::InFlight(key.to_string()));
                    }
                    KeyState::Completed => {
                        record.duplicate_count += 1;
                        return Ok(Some(record.clone()));
                    }
                    KeyState::Failed => {
                        // Allow retry of failed requests.
                        record.state = KeyState::InFlight;
                        return Ok(None);
                    }
                    KeyState::Expired => {
                        // Should not reach here due to check above.
                        self.records.remove(key);
                    }
                }
            }
        }

        // Evict if at capacity.
        if self.records.len() >= self.config.max_keys {
            self.evict_expired();
            // If still full, evict oldest.
            if self.records.len() >= self.config.max_keys {
                self.evict_oldest();
            }
        }

        // Create new record.
        let now = Utc::now();
        let record = IdempotencyRecord {
            key: key.to_string(),
            fingerprint: fingerprint.to_string(),
            state: KeyState::InFlight,
            response_status: None,
            response_body: None,
            response_headers: None,
            created_at: now,
            completed_at: None,
            expires_at: now + Duration::seconds(ttl as i64),
            duplicate_count: 0,
        };
        self.records.insert(key.to_string(), record);
        Ok(None)
    }

    /// Complete a request — cache the response.
    pub fn complete(
        &mut self,
        key: &str,
        status: u16,
        body: serde_json::Value,
        headers: Option<HashMap<String, String>>,
    ) -> Result<(), IdempotencyError> {
        let record = self
            .records
            .get_mut(key)
            .ok_or_else(|| IdempotencyError::KeyNotFound(key.to_string()))?;

        if record.state == KeyState::Completed {
            return Err(IdempotencyError::AlreadyCompleted(key.to_string()));
        }

        record.state = KeyState::Completed;
        record.response_status = Some(status);
        record.response_body = Some(body);
        record.response_headers = headers;
        record.completed_at = Some(Utc::now());
        self.total_energy_uj += 8;
        Ok(())
    }

    /// Mark a request as failed.
    pub fn fail(&mut self, key: &str) -> Result<(), IdempotencyError> {
        let record = self
            .records
            .get_mut(key)
            .ok_or_else(|| IdempotencyError::KeyNotFound(key.to_string()))?;

        record.state = KeyState::Failed;
        self.total_energy_uj += 3;
        Ok(())
    }

    /// Check if a key exists and is still valid.
    pub fn exists(&self, key: &str) -> bool {
        self.records
            .get(key)
            .map(|r| Utc::now() < r.expires_at)
            .unwrap_or(false)
    }

    /// Get a record by key.
    pub fn get(&self, key: &str) -> Option<&IdempotencyRecord> {
        self.records.get(key).filter(|r| Utc::now() < r.expires_at)
    }

    /// Remove expired records.
    pub fn cleanup(&mut self) -> usize {
        let now = Utc::now();
        let before = self.records.len();
        self.records.retain(|_, r| now < r.expires_at);
        let removed = before - self.records.len();
        self.total_energy_uj += removed as u64 * 2;
        removed
    }

    /// Total active keys.
    pub fn key_count(&self) -> usize {
        self.records.len()
    }

    /// Total energy consumed.
    pub fn total_energy_uj(&self) -> u64 {
        self.total_energy_uj
    }

    /// Total duplicate requests detected.
    pub fn total_duplicates(&self) -> u64 {
        self.records.values().map(|r| r.duplicate_count).sum()
    }

    // ── Internal ────────────────────────────────────────────────

    fn evict_expired(&mut self) {
        let now = Utc::now();
        self.records.retain(|_, r| now < r.expires_at);
    }

    fn evict_oldest(&mut self) {
        if let Some(oldest_key) = self
            .records
            .iter()
            .min_by_key(|(_, r)| r.created_at)
            .map(|(k, _)| k.clone())
        {
            self.records.remove(&oldest_key);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> IdempotencyStore {
        IdempotencyStore::new(IdempotencyConfig::default())
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let fp1 = compute_fingerprint(b"hello world");
        let fp2 = compute_fingerprint(b"hello world");
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.len(), 16);
    }

    #[test]
    fn test_fingerprint_different_input() {
        let fp1 = compute_fingerprint(b"hello");
        let fp2 = compute_fingerprint(b"world");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_json() {
        let v1 = serde_json::json!({"a": 1, "b": 2});
        let fp = fingerprint_json(&v1);
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_begin_new_request() {
        let mut s = store();
        let result = s.begin("key1", "fp1", None).unwrap();
        assert!(result.is_none());
        assert_eq!(s.key_count(), 1);

        let rec = s.get("key1").unwrap();
        assert_eq!(rec.state, KeyState::InFlight);
    }

    #[test]
    fn test_complete_and_cached_response() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();
        s.complete("key1", 200, serde_json::json!({"ok": true}), None)
            .unwrap();

        let rec = s.get("key1").unwrap();
        assert_eq!(rec.state, KeyState::Completed);
        assert_eq!(rec.response_status, Some(200));

        // Second request with same key returns cached response.
        let result = s.begin("key1", "fp1", None).unwrap();
        assert!(result.is_some());
        let cached = result.unwrap();
        assert_eq!(cached.response_status, Some(200));
        assert_eq!(cached.duplicate_count, 1);
    }

    #[test]
    fn test_in_flight_conflict() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();

        let result = s.begin("key1", "fp1", None);
        assert_eq!(result, Err(IdempotencyError::InFlight("key1".into())));
    }

    #[test]
    fn test_fingerprint_mismatch() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();
        s.complete("key1", 200, serde_json::json!(null), None).unwrap();

        let result = s.begin("key1", "different_fp", None);
        assert!(matches!(
            result,
            Err(IdempotencyError::FingerprintMismatch { .. })
        ));
    }

    #[test]
    fn test_fingerprint_not_enforced() {
        let mut s = IdempotencyStore::new(IdempotencyConfig {
            enforce_fingerprint: false,
            ..Default::default()
        });
        s.begin("key1", "fp1", None).unwrap();
        s.complete("key1", 200, serde_json::json!(null), None).unwrap();

        // Different fingerprint succeeds when not enforced.
        let result = s.begin("key1", "different", None).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_fail_and_retry() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();
        s.fail("key1").unwrap();

        let rec = s.get("key1").unwrap();
        assert_eq!(rec.state, KeyState::Failed);

        // Re-begin should succeed (retry of failed).
        let result = s.begin("key1", "fp1", None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_complete_nonexistent() {
        let mut s = store();
        let result = s.complete("missing", 200, serde_json::json!(null), None);
        assert_eq!(result, Err(IdempotencyError::KeyNotFound("missing".into())));
    }

    #[test]
    fn test_double_complete() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();
        s.complete("key1", 200, serde_json::json!(null), None).unwrap();
        let result = s.complete("key1", 200, serde_json::json!(null), None);
        assert_eq!(result, Err(IdempotencyError::AlreadyCompleted("key1".into())));
    }

    #[test]
    fn test_ttl_expiration() {
        let mut s = store();
        s.begin("key1", "fp1", Some(1)).unwrap();
        s.complete("key1", 200, serde_json::json!(null), None).unwrap();

        // Manually expire.
        s.records.get_mut("key1").unwrap().expires_at =
            Utc::now() - Duration::seconds(10);

        assert!(!s.exists("key1"));
        assert!(s.get("key1").is_none());
    }

    #[test]
    fn test_cleanup() {
        let mut s = store();
        s.begin("key1", "fp1", Some(1)).unwrap();
        s.begin("key2", "fp2", Some(1)).unwrap();

        // Expire key1.
        s.records.get_mut("key1").unwrap().expires_at =
            Utc::now() - Duration::seconds(10);

        let removed = s.cleanup();
        assert_eq!(removed, 1);
        assert_eq!(s.key_count(), 1);
    }

    #[test]
    fn test_invalid_ttl() {
        let mut s = store();
        let result = s.begin("key1", "fp1", Some(0));
        assert_eq!(
            result,
            Err(IdempotencyError::InvalidTtl("TTL must be > 0".into()))
        );
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut s = IdempotencyStore::new(IdempotencyConfig {
            max_keys: 3,
            ..Default::default()
        });

        s.begin("k1", "fp1", None).unwrap();
        s.begin("k2", "fp2", None).unwrap();
        s.begin("k3", "fp3", None).unwrap();

        // This should evict the oldest.
        s.begin("k4", "fp4", None).unwrap();
        assert_eq!(s.key_count(), 3);
    }

    #[test]
    fn test_exists() {
        let mut s = store();
        assert!(!s.exists("key1"));
        s.begin("key1", "fp1", None).unwrap();
        assert!(s.exists("key1"));
    }

    #[test]
    fn test_total_duplicates() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();
        s.complete("key1", 200, serde_json::json!(null), None).unwrap();

        s.begin("key1", "fp1", None).unwrap(); // dup 1
        s.begin("key1", "fp1", None).unwrap(); // dup 2

        assert_eq!(s.total_duplicates(), 2);
    }

    #[test]
    fn test_response_headers_cached() {
        let mut s = store();
        s.begin("key1", "fp1", None).unwrap();

        let mut headers = HashMap::new();
        headers.insert("x-request-id".into(), "abc".into());
        s.complete("key1", 201, serde_json::json!({"id": 42}), Some(headers))
            .unwrap();

        let rec = s.get("key1").unwrap();
        assert_eq!(rec.response_status, Some(201));
        let h = rec.response_headers.as_ref().unwrap();
        assert_eq!(h.get("x-request-id"), Some(&"abc".to_string()));
    }

    #[test]
    fn test_energy_tracking() {
        let mut s = store();
        assert_eq!(s.total_energy_uj(), 0);
        s.begin("key1", "fp1", None).unwrap();
        assert!(s.total_energy_uj() > 0);
    }

    #[test]
    fn test_key_state_serde() {
        let state = KeyState::Completed;
        let json = serde_json::to_string(&state).unwrap();
        let parsed: KeyState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, KeyState::Completed);
    }

    #[test]
    fn test_error_display() {
        let e = IdempotencyError::FingerprintMismatch {
            key: "k".into(),
            expected: "a".into(),
            got: "b".into(),
        };
        let s = e.to_string();
        assert!(s.contains("fingerprint mismatch"));
    }

    #[test]
    fn test_expired_key_reusable() {
        let mut s = store();
        s.begin("key1", "fp1", Some(1)).unwrap();
        s.complete("key1", 200, serde_json::json!(null), None).unwrap();

        // Expire it.
        s.records.get_mut("key1").unwrap().expires_at =
            Utc::now() - Duration::seconds(10);

        // Should be able to reuse the key.
        let result = s.begin("key1", "fp_new", None).unwrap();
        assert!(result.is_none()); // New request, not cached.
    }

    #[test]
    fn test_fail_nonexistent() {
        let mut s = store();
        assert_eq!(
            s.fail("missing"),
            Err(IdempotencyError::KeyNotFound("missing".into()))
        );
    }

    #[test]
    fn test_config_default() {
        let cfg = IdempotencyConfig::default();
        assert_eq!(cfg.default_ttl_seconds, 86400);
        assert!(cfg.enforce_fingerprint);
        assert_eq!(cfg.max_keys, 100_000);
    }
}
