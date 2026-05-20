//! Session management — session creation with token, session data store, TTL with
//! renewal, concurrent session limits, session invalidation (single/all), session
//! fingerprint (IP/UA), and session activity tracking.
//!
//! Replaces `express-session`, `cookie-session`, and `iron-session` with a pure-Rust
//! in-memory session manager supporting TTL, fingerprinting, and concurrency limits.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Session management errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Session not found.
    NotFound(String),
    /// Session has expired.
    Expired { session_id: String, expired_at_ms: u64 },
    /// Session has been invalidated.
    Invalidated(String),
    /// Concurrent session limit exceeded.
    ConcurrentLimitExceeded { user_id: String, limit: usize, current: usize },
    /// Session fingerprint mismatch.
    FingerprintMismatch { session_id: String, expected_ip: String, actual_ip: String },
    /// Invalid session data.
    InvalidData(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "session not found: {id}"),
            Self::Expired { session_id, expired_at_ms } => {
                write!(f, "session {session_id} expired at {expired_at_ms}")
            }
            Self::Invalidated(id) => write!(f, "session invalidated: {id}"),
            Self::ConcurrentLimitExceeded { user_id, limit, current } => {
                write!(f, "concurrent session limit for user {user_id}: {current}/{limit}")
            }
            Self::FingerprintMismatch { session_id, expected_ip, actual_ip } => {
                write!(
                    f,
                    "fingerprint mismatch for session {session_id}: expected IP {expected_ip}, got {actual_ip}"
                )
            }
            Self::InvalidData(msg) => write!(f, "invalid session data: {msg}"),
        }
    }
}

impl std::error::Error for SessionError {}

// ── Types ──────────────────────────────────────────────────────

/// Session status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    Active,
    Expired,
    Invalidated,
}

impl SessionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Expired => "expired",
            Self::Invalidated => "invalidated",
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }
}

impl fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A session fingerprint for binding sessions to a client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionFingerprint {
    /// Client IP address.
    pub ip: String,
    /// User agent string.
    pub user_agent: String,
}

impl SessionFingerprint {
    pub fn new(ip: &str, user_agent: &str) -> Self {
        Self {
            ip: ip.to_string(),
            user_agent: user_agent.to_string(),
        }
    }

    /// Check if another fingerprint matches (IP must match; UA is advisory).
    pub fn matches_ip(&self, other: &SessionFingerprint) -> bool {
        self.ip == other.ip
    }

    /// Full match including user agent.
    pub fn full_match(&self, other: &SessionFingerprint) -> bool {
        self.ip == other.ip && self.user_agent == other.user_agent
    }
}

/// Activity record for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityRecord {
    /// What the user did.
    pub action: String,
    /// Timestamp (epoch ms).
    pub timestamp_ms: u64,
    /// Additional details.
    pub details: Option<String>,
}

/// A session record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session ID (the token).
    pub id: String,
    /// User ID this session belongs to.
    pub user_id: String,
    /// Session status.
    pub status: SessionStatus,
    /// Session data (arbitrary key-value).
    pub data: HashMap<String, Value>,
    /// Creation timestamp (epoch ms).
    pub created_at_ms: u64,
    /// Last accessed timestamp (epoch ms).
    pub last_accessed_ms: u64,
    /// Expiry timestamp (epoch ms).
    pub expires_at_ms: u64,
    /// TTL in milliseconds.
    pub ttl_ms: u64,
    /// Client fingerprint.
    pub fingerprint: Option<SessionFingerprint>,
    /// Activity history.
    pub activity: Vec<ActivityRecord>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// Number of renewals.
    pub renewal_count: u32,
}

impl Session {
    /// Check if the session is expired at the given time.
    pub fn is_expired_at(&self, now_ms: u64) -> bool {
        self.expires_at_ms > 0 && now_ms > self.expires_at_ms
    }

    /// Time remaining in milliseconds (0 if expired).
    pub fn time_remaining(&self, now_ms: u64) -> u64 {
        if self.expires_at_ms == 0 {
            return u64::MAX; // no expiry
        }
        self.expires_at_ms.saturating_sub(now_ms)
    }
}

/// Configuration for the session manager.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Default TTL in milliseconds.
    pub default_ttl_ms: u64,
    /// Maximum concurrent sessions per user (0 = unlimited).
    pub max_concurrent_per_user: usize,
    /// Whether to renew TTL on access.
    pub renew_on_access: bool,
    /// Whether to enforce IP fingerprint matching.
    pub enforce_fingerprint: bool,
    /// Maximum activity records per session (for memory bounds).
    pub max_activity_records: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            default_ttl_ms: 30 * 60 * 1000, // 30 minutes
            max_concurrent_per_user: 0,
            renew_on_access: true,
            enforce_fingerprint: false,
            max_activity_records: 100,
        }
    }
}

/// The session manager.
pub struct SessionManager {
    /// Sessions keyed by session ID.
    sessions: HashMap<String, Session>,
    /// User -> session IDs index.
    user_sessions: HashMap<String, Vec<String>>,
    /// Configuration.
    pub config: SessionConfig,
    /// Counter for generating session IDs.
    next_counter: u64,
}

impl SessionManager {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            sessions: HashMap::new(),
            user_sessions: HashMap::new(),
            config,
            next_counter: 1,
        }
    }

    /// Create a new session for a user.
    pub fn create_session(
        &mut self,
        user_id: &str,
        fingerprint: Option<SessionFingerprint>,
        now_ms: u64,
        seed: u64,
    ) -> Result<&Session, SessionError> {
        // Check concurrent limit.
        if self.config.max_concurrent_per_user > 0 {
            let active_count = self.active_session_count(user_id, now_ms);
            if active_count >= self.config.max_concurrent_per_user {
                return Err(SessionError::ConcurrentLimitExceeded {
                    user_id: user_id.to_string(),
                    limit: self.config.max_concurrent_per_user,
                    current: active_count,
                });
            }
        }

        let session_id = generate_session_token(seed, self.next_counter);
        self.next_counter += 1;
        let ttl = self.config.default_ttl_ms;

        let session = Session {
            id: session_id.clone(),
            user_id: user_id.to_string(),
            status: SessionStatus::Active,
            data: HashMap::new(),
            created_at_ms: now_ms,
            last_accessed_ms: now_ms,
            expires_at_ms: now_ms + ttl,
            ttl_ms: ttl,
            fingerprint,
            activity: Vec::new(),
            metadata: HashMap::new(),
            renewal_count: 0,
        };

        self.sessions.insert(session_id.clone(), session);
        self.user_sessions
            .entry(user_id.to_string())
            .or_default()
            .push(session_id.clone());

        Ok(self.sessions.get(&session_id).unwrap())
    }

    /// Get a session, optionally validating fingerprint and renewing TTL.
    pub fn get_session(
        &mut self,
        session_id: &str,
        fingerprint: Option<&SessionFingerprint>,
        now_ms: u64,
    ) -> Result<&Session, SessionError> {
        // Check existence.
        if !self.sessions.contains_key(session_id) {
            return Err(SessionError::NotFound(session_id.to_string()));
        }

        // Check status and expiry.
        {
            let session = self.sessions.get(session_id).unwrap();
            if session.status == SessionStatus::Invalidated {
                return Err(SessionError::Invalidated(session_id.to_string()));
            }
            if session.is_expired_at(now_ms) {
                let expired_at = session.expires_at_ms;
                let session_mut = self.sessions.get_mut(session_id).unwrap();
                session_mut.status = SessionStatus::Expired;
                return Err(SessionError::Expired {
                    session_id: session_id.to_string(),
                    expired_at_ms: expired_at,
                });
            }
        }

        // Check fingerprint.
        if self.config.enforce_fingerprint {
            if let (Some(stored_fp), Some(client_fp)) = (
                self.sessions.get(session_id).unwrap().fingerprint.as_ref(),
                fingerprint,
            ) {
                if !stored_fp.matches_ip(client_fp) {
                    let expected = stored_fp.ip.clone();
                    let actual = client_fp.ip.clone();
                    return Err(SessionError::FingerprintMismatch {
                        session_id: session_id.to_string(),
                        expected_ip: expected,
                        actual_ip: actual,
                    });
                }
            }
        }

        // Renew TTL if configured.
        let session_mut = self.sessions.get_mut(session_id).unwrap();
        session_mut.last_accessed_ms = now_ms;
        if self.config.renew_on_access {
            session_mut.expires_at_ms = now_ms + session_mut.ttl_ms;
            session_mut.renewal_count += 1;
        }

        Ok(self.sessions.get(session_id).unwrap())
    }

    /// Set a data value in a session.
    pub fn set_data(
        &mut self,
        session_id: &str,
        key: &str,
        value: Value,
    ) -> Result<(), SessionError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;
        if !session.status.is_active() {
            return Err(SessionError::Invalidated(session_id.to_string()));
        }
        session.data.insert(key.to_string(), value);
        Ok(())
    }

    /// Get a data value from a session.
    pub fn get_data(&self, session_id: &str, key: &str) -> Result<Option<&Value>, SessionError> {
        let session = self
            .sessions
            .get(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;
        Ok(session.data.get(key))
    }

    /// Remove a data value.
    pub fn remove_data(&mut self, session_id: &str, key: &str) -> Result<Option<Value>, SessionError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;
        Ok(session.data.remove(key))
    }

    /// Record an activity in a session.
    pub fn record_activity(
        &mut self,
        session_id: &str,
        action: &str,
        now_ms: u64,
        details: Option<&str>,
    ) -> Result<(), SessionError> {
        let max_records = self.config.max_activity_records;
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;

        session.activity.push(ActivityRecord {
            action: action.to_string(),
            timestamp_ms: now_ms,
            details: details.map(|d| d.to_string()),
        });

        // Trim to max records.
        if session.activity.len() > max_records {
            let excess = session.activity.len() - max_records;
            session.activity.drain(0..excess);
        }

        Ok(())
    }

    /// Invalidate a single session.
    pub fn invalidate(&mut self, session_id: &str) -> Result<(), SessionError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;
        session.status = SessionStatus::Invalidated;
        Ok(())
    }

    /// Invalidate all sessions for a user.
    pub fn invalidate_all(&mut self, user_id: &str) -> usize {
        let session_ids: Vec<String> = self
            .user_sessions
            .get(user_id)
            .cloned()
            .unwrap_or_default();

        let mut count = 0;
        for sid in &session_ids {
            if let Some(session) = self.sessions.get_mut(sid) {
                if session.status == SessionStatus::Active {
                    session.status = SessionStatus::Invalidated;
                    count += 1;
                }
            }
        }
        count
    }

    /// Get all active sessions for a user.
    pub fn user_sessions(&self, user_id: &str, now_ms: u64) -> Vec<&Session> {
        self.user_sessions
            .get(user_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.sessions.get(id))
                    .filter(|s| s.status == SessionStatus::Active && !s.is_expired_at(now_ms))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Count active sessions for a user.
    fn active_session_count(&self, user_id: &str, now_ms: u64) -> usize {
        self.user_sessions(user_id, now_ms).len()
    }

    /// Total session count (all states).
    pub fn total_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Clean up expired/invalidated sessions older than `older_than_ms`.
    pub fn cleanup(&mut self, now_ms: u64) -> usize {
        let expired_ids: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.is_expired_at(now_ms) || s.status == SessionStatus::Invalidated)
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired_ids.len();
        for id in &expired_ids {
            if let Some(session) = self.sessions.remove(id) {
                if let Some(user_ids) = self.user_sessions.get_mut(&session.user_id) {
                    user_ids.retain(|sid| sid != id);
                }
            }
        }
        count
    }

    /// Set metadata on a session.
    pub fn set_metadata(
        &mut self,
        session_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), SessionError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;
        session.metadata.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Get a session by ID without validation (for inspection).
    pub fn inspect_session(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Set a custom TTL for a specific session.
    pub fn set_session_ttl(
        &mut self,
        session_id: &str,
        ttl_ms: u64,
        now_ms: u64,
    ) -> Result<(), SessionError> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| SessionError::NotFound(session_id.to_string()))?;
        session.ttl_ms = ttl_ms;
        session.expires_at_ms = now_ms + ttl_ms;
        Ok(())
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new(SessionConfig::default())
    }
}

/// Generate a pseudo-random session token.
fn generate_session_token(seed: u64, counter: u64) -> String {
    let mut h = seed ^ 0x517cc1b727220a95;
    h = h.wrapping_mul(0x100000001b3);
    h ^= counter;
    h = h.wrapping_mul(0x100000001b3);
    let h2 = h.wrapping_mul(0xcbf29ce484222325).wrapping_add(0x6c62272e07bb0142);
    format!("sess_{h:016x}{h2:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn default_mgr() -> SessionManager {
        SessionManager::new(SessionConfig {
            default_ttl_ms: 30_000, // 30 seconds for tests
            max_concurrent_per_user: 0,
            renew_on_access: true,
            enforce_fingerprint: false,
            max_activity_records: 50,
        })
    }

    #[test]
    fn test_create_session() {
        let mut mgr = default_mgr();
        let session = mgr.create_session("alice", None, 1000, 42).unwrap();
        assert_eq!(session.user_id, "alice");
        assert_eq!(session.status, SessionStatus::Active);
        assert!(session.id.starts_with("sess_"));
    }

    #[test]
    fn test_get_session() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        let session = mgr.get_session(&sid, None, 2000).unwrap();
        assert_eq!(session.last_accessed_ms, 2000);
    }

    #[test]
    fn test_session_not_found() {
        let mut mgr = default_mgr();
        let err = mgr.get_session("nonexistent", None, 1000).unwrap_err();
        assert_eq!(err, SessionError::NotFound("nonexistent".into()));
    }

    #[test]
    fn test_session_expiry() {
        let mut mgr = default_mgr();
        mgr.config.renew_on_access = false;
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        // Session expires at 1000 + 30000 = 31000
        let err = mgr.get_session(&sid, None, 50_000).unwrap_err();
        match err {
            SessionError::Expired { .. } => {}
            other => panic!("expected Expired, got: {other}"),
        }
    }

    #[test]
    fn test_session_ttl_renewal() {
        let mut mgr = default_mgr();
        mgr.config.renew_on_access = true;
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        // Access at 20000 — should renew expiry to 20000 + 30000 = 50000
        mgr.get_session(&sid, None, 20_000).unwrap();
        // Access at 45000 — should still be valid
        assert!(mgr.get_session(&sid, None, 45_000).is_ok());
    }

    #[test]
    fn test_session_data() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.set_data(&sid, "cart", json!(["item1", "item2"])).unwrap();
        let val = mgr.get_data(&sid, "cart").unwrap();
        assert_eq!(val, Some(&json!(["item1", "item2"])));
    }

    #[test]
    fn test_remove_session_data() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.set_data(&sid, "key", json!("value")).unwrap();
        let removed = mgr.remove_data(&sid, "key").unwrap();
        assert_eq!(removed, Some(json!("value")));
        assert_eq!(mgr.get_data(&sid, "key").unwrap(), None);
    }

    #[test]
    fn test_invalidate_session() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.invalidate(&sid).unwrap();
        let err = mgr.get_session(&sid, None, 2000).unwrap_err();
        assert_eq!(err, SessionError::Invalidated(sid));
    }

    #[test]
    fn test_invalidate_all_sessions() {
        let mut mgr = default_mgr();
        mgr.create_session("alice", None, 1000, 1).unwrap();
        mgr.create_session("alice", None, 2000, 2).unwrap();
        mgr.create_session("bob", None, 3000, 3).unwrap();

        let count = mgr.invalidate_all("alice");
        assert_eq!(count, 2);
        // Bob's session should still be active
        let bob_sessions = mgr.user_sessions("bob", 4000);
        assert_eq!(bob_sessions.len(), 1);
    }

    #[test]
    fn test_concurrent_session_limit() {
        let mut mgr = SessionManager::new(SessionConfig {
            default_ttl_ms: 30_000,
            max_concurrent_per_user: 2,
            renew_on_access: false,
            enforce_fingerprint: false,
            max_activity_records: 50,
        });
        mgr.create_session("alice", None, 1000, 1).unwrap();
        mgr.create_session("alice", None, 2000, 2).unwrap();
        let err = mgr.create_session("alice", None, 3000, 3).unwrap_err();
        match err {
            SessionError::ConcurrentLimitExceeded { limit: 2, current: 2, .. } => {}
            other => panic!("expected ConcurrentLimitExceeded, got: {other}"),
        }
    }

    #[test]
    fn test_fingerprint_enforcement() {
        let mut mgr = SessionManager::new(SessionConfig {
            default_ttl_ms: 30_000,
            max_concurrent_per_user: 0,
            renew_on_access: false,
            enforce_fingerprint: true,
            max_activity_records: 50,
        });
        let fp = SessionFingerprint::new("10.0.0.1", "Mozilla/5.0");
        let sid = mgr.create_session("alice", Some(fp), 1000, 42).unwrap().id.clone();

        // Same IP — OK
        let same_fp = SessionFingerprint::new("10.0.0.1", "Chrome");
        assert!(mgr.get_session(&sid, Some(&same_fp), 2000).is_ok());

        // Different IP — rejected
        let diff_fp = SessionFingerprint::new("10.0.0.2", "Mozilla/5.0");
        let err = mgr.get_session(&sid, Some(&diff_fp), 3000).unwrap_err();
        match err {
            SessionError::FingerprintMismatch { .. } => {}
            other => panic!("expected FingerprintMismatch, got: {other}"),
        }
    }

    #[test]
    fn test_activity_recording() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.record_activity(&sid, "login", 1000, None).unwrap();
        mgr.record_activity(&sid, "view_page", 2000, Some("/dashboard")).unwrap();

        let session = mgr.inspect_session(&sid).unwrap();
        assert_eq!(session.activity.len(), 2);
        assert_eq!(session.activity[1].action, "view_page");
        assert_eq!(session.activity[1].details, Some("/dashboard".to_string()));
    }

    #[test]
    fn test_activity_trimming() {
        let mut mgr = SessionManager::new(SessionConfig {
            default_ttl_ms: 30_000,
            max_concurrent_per_user: 0,
            renew_on_access: false,
            enforce_fingerprint: false,
            max_activity_records: 3,
        });
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        for i in 0..5 {
            mgr.record_activity(&sid, &format!("action-{i}"), 1000 + i * 100, None)
                .unwrap();
        }
        let session = mgr.inspect_session(&sid).unwrap();
        assert_eq!(session.activity.len(), 3);
        // Should have the last 3
        assert_eq!(session.activity[0].action, "action-2");
    }

    #[test]
    fn test_cleanup_expired() {
        let mut mgr = default_mgr();
        mgr.config.renew_on_access = false;
        mgr.create_session("alice", None, 1000, 1).unwrap();
        mgr.create_session("bob", None, 1000, 2).unwrap();

        // Both expire at 31000
        let cleaned = mgr.cleanup(50_000);
        assert_eq!(cleaned, 2);
        assert_eq!(mgr.total_sessions(), 0);
    }

    #[test]
    fn test_cleanup_invalidated() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.invalidate(&sid).unwrap();
        let cleaned = mgr.cleanup(2000);
        assert_eq!(cleaned, 1);
    }

    #[test]
    fn test_user_sessions_listing() {
        let mut mgr = default_mgr();
        mgr.create_session("alice", None, 1000, 1).unwrap();
        mgr.create_session("alice", None, 2000, 2).unwrap();
        mgr.create_session("bob", None, 3000, 3).unwrap();

        let alice_sessions = mgr.user_sessions("alice", 5000);
        assert_eq!(alice_sessions.len(), 2);
    }

    #[test]
    fn test_session_metadata() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.set_metadata(&sid, "device", "mobile").unwrap();
        let session = mgr.inspect_session(&sid).unwrap();
        assert_eq!(session.metadata.get("device"), Some(&"mobile".to_string()));
    }

    #[test]
    fn test_set_custom_ttl() {
        let mut mgr = default_mgr();
        mgr.config.renew_on_access = false;
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.set_session_ttl(&sid, 5000, 2000).unwrap();
        // Should expire at 2000 + 5000 = 7000
        assert!(mgr.get_session(&sid, None, 6000).is_ok());
        assert!(mgr.get_session(&sid, None, 8000).is_err());
    }

    #[test]
    fn test_session_time_remaining() {
        let mut mgr = default_mgr();
        mgr.config.renew_on_access = false;
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        let session = mgr.inspect_session(&sid).unwrap();
        // expires at 31000, at time 10000 -> 21000 remaining
        assert_eq!(session.time_remaining(10_000), 21_000);
    }

    #[test]
    fn test_token_generation_deterministic() {
        let t1 = generate_session_token(42, 1);
        let t2 = generate_session_token(42, 1);
        assert_eq!(t1, t2);
        let t3 = generate_session_token(42, 2);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_fingerprint_matching() {
        let fp1 = SessionFingerprint::new("10.0.0.1", "Mozilla");
        let fp2 = SessionFingerprint::new("10.0.0.1", "Chrome");
        let fp3 = SessionFingerprint::new("10.0.0.2", "Mozilla");

        assert!(fp1.matches_ip(&fp2));
        assert!(!fp1.matches_ip(&fp3));
        assert!(!fp1.full_match(&fp2)); // different UA
        assert!(fp1.full_match(&SessionFingerprint::new("10.0.0.1", "Mozilla")));
    }

    #[test]
    fn test_set_data_on_invalidated_session() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.invalidate(&sid).unwrap();
        let err = mgr.set_data(&sid, "key", json!("value")).unwrap_err();
        assert_eq!(err, SessionError::Invalidated(sid));
    }

    #[test]
    fn test_renewal_count() {
        let mut mgr = default_mgr();
        let sid = mgr.create_session("alice", None, 1000, 42).unwrap().id.clone();
        mgr.get_session(&sid, None, 2000).unwrap();
        mgr.get_session(&sid, None, 3000).unwrap();
        let session = mgr.inspect_session(&sid).unwrap();
        assert_eq!(session.renewal_count, 2);
    }
}
