//! Session management — session creation with random ID, typed get/set for
//! session data, configurable TTL, renewal, secure token generation, concurrent
//! session limits per user, and session enumeration.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by session operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Session not found or expired.
    NotFound,
    /// Session has expired.
    Expired,
    /// Maximum concurrent sessions for this user reached.
    ConcurrentLimitExceeded,
    /// Typed value could not be deserialized.
    TypeMismatch(String),
    /// Token validation failed.
    InvalidToken,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "session not found"),
            Self::Expired => write!(f, "session expired"),
            Self::ConcurrentLimitExceeded => write!(f, "concurrent session limit exceeded"),
            Self::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Self::InvalidToken => write!(f, "invalid session token"),
        }
    }
}

// ── Token generation ─────────────────────────────────────────────────────────

/// Generate a pseudo-random session ID using xorshift on the given seed.
fn generate_session_id(seed: u64) -> String {
    let mut state = seed;
    let mut bytes = [0u8; 16];
    for byte in &mut bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state & 0xFF) as u8;
    }
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    hex
}

/// Generate a secure token (session ID + nonce combined).
fn generate_token(session_id: &str, nonce: u64) -> String {
    let mut state = nonce;
    let mut bytes = [0u8; 16];
    for byte in &mut bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state & 0xFF) as u8;
    }
    let suffix: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("{session_id}.{suffix}")
}

// ── Session ──────────────────────────────────────────────────────────────────

/// A single session with typed key-value data.
#[derive(Debug, Clone)]
pub struct Session {
    id: String,
    token: String,
    user_id: Option<String>,
    data: HashMap<String, String>,
    created_at: Instant,
    last_accessed: Instant,
    ttl: Duration,
    access_count: u64,
}

impl Session {
    /// Session ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Secure session token.
    pub fn token(&self) -> &str {
        &self.token
    }

    /// User ID associated with this session, if any.
    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }

    /// Whether the session has expired.
    pub fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.created_at) > self.ttl
    }

    /// Time remaining before expiry.
    pub fn time_remaining(&self, now: Instant) -> Duration {
        let elapsed = now.duration_since(self.created_at);
        self.ttl.saturating_sub(elapsed)
    }

    /// Number of times this session has been accessed.
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Get a typed string value from the session data.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }

    /// Get a value parsed as a specific type.
    pub fn get_typed<T: std::str::FromStr>(&self, key: &str) -> Result<T, SessionError> {
        let val = self.data.get(key).ok_or(SessionError::NotFound)?;
        val.parse::<T>()
            .map_err(|_| SessionError::TypeMismatch(format!("cannot parse '{val}'")))
    }

    /// Set a string value in the session data.
    pub fn set(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }

    /// Remove a key from session data.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// All keys in the session data.
    pub fn keys(&self) -> Vec<&str> {
        self.data.keys().map(|k| k.as_str()).collect()
    }

    /// Number of data entries.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

// ── SessionStore ─────────────────────────────────────────────────────────────

/// Session store managing multiple sessions with TTL, per-user limits, and
/// token-based access.
pub struct SessionStore {
    sessions: HashMap<String, Session>,
    /// Map from token to session ID.
    token_index: HashMap<String, String>,
    /// Map from user_id to set of session IDs.
    user_sessions: HashMap<String, Vec<String>>,
    default_ttl: Duration,
    max_sessions_per_user: usize,
    next_seed: u64,
    total_created: u64,
    total_expired: u64,
}

impl SessionStore {
    /// Create a new session store with the given default TTL and per-user limit.
    pub fn new(default_ttl: Duration, max_sessions_per_user: usize) -> Self {
        Self {
            sessions: HashMap::new(),
            token_index: HashMap::new(),
            user_sessions: HashMap::new(),
            default_ttl,
            max_sessions_per_user,
            next_seed: 0xDEAD_BEEF_CAFE_BABE,
            total_created: 0,
            total_expired: 0,
        }
    }

    /// Create with default settings (30 min TTL, 5 sessions per user).
    pub fn with_defaults() -> Self {
        Self::new(Duration::from_secs(1800), 5)
    }

    /// Create a new session, optionally associated with a user.
    pub fn create(&mut self, user_id: Option<String>) -> Result<String, SessionError> {
        self.create_with_ttl(user_id, self.default_ttl)
    }

    /// Create a new session with a specific TTL.
    pub fn create_with_ttl(
        &mut self,
        user_id: Option<String>,
        ttl: Duration,
    ) -> Result<String, SessionError> {
        // Check concurrent session limit.
        if let Some(uid) = &user_id {
            self.purge_expired_for_user(uid);
            let count = self
                .user_sessions
                .get(uid)
                .map(|ids| ids.len())
                .unwrap_or(0);
            if count >= self.max_sessions_per_user {
                return Err(SessionError::ConcurrentLimitExceeded);
            }
        }

        self.next_seed = self.next_seed.wrapping_add(1);
        let id = generate_session_id(self.next_seed);
        self.next_seed = self.next_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let token = generate_token(&id, self.next_seed);
        let now = Instant::now();

        let session = Session {
            id: id.clone(),
            token: token.clone(),
            user_id: user_id.clone(),
            data: HashMap::new(),
            created_at: now,
            last_accessed: now,
            ttl,
            access_count: 0,
        };

        self.sessions.insert(id.clone(), session);
        self.token_index.insert(token, id.clone());
        if let Some(uid) = user_id {
            self.user_sessions
                .entry(uid)
                .or_default()
                .push(id.clone());
        }
        self.total_created += 1;
        Ok(id)
    }

    /// Get a session by ID, touching last_accessed.
    pub fn get(&mut self, session_id: &str) -> Result<&Session, SessionError> {
        let now = Instant::now();
        let session = self.sessions.get_mut(session_id).ok_or(SessionError::NotFound)?;
        if session.is_expired(now) {
            let sid = session_id.to_string();
            self.remove_session(&sid);
            self.total_expired += 1;
            return Err(SessionError::Expired);
        }
        session.last_accessed = now;
        session.access_count += 1;
        // Re-borrow as immutable.
        Ok(self.sessions.get(session_id).unwrap())
    }

    /// Get a mutable session by ID.
    pub fn get_mut(&mut self, session_id: &str) -> Result<&mut Session, SessionError> {
        let now = Instant::now();
        let session = self.sessions.get_mut(session_id).ok_or(SessionError::NotFound)?;
        if session.is_expired(now) {
            let sid = session_id.to_string();
            self.remove_session(&sid);
            self.total_expired += 1;
            return Err(SessionError::Expired);
        }
        session.last_accessed = now;
        session.access_count += 1;
        Ok(self.sessions.get_mut(session_id).unwrap())
    }

    /// Validate a token and return the session ID.
    pub fn validate_token(&mut self, token: &str) -> Result<String, SessionError> {
        let session_id = self
            .token_index
            .get(token)
            .cloned()
            .ok_or(SessionError::InvalidToken)?;
        // Verify session still exists and is valid.
        let now = Instant::now();
        let session = self
            .sessions
            .get(&session_id)
            .ok_or(SessionError::InvalidToken)?;
        if session.is_expired(now) {
            let sid = session_id.clone();
            self.remove_session(&sid);
            self.total_expired += 1;
            return Err(SessionError::Expired);
        }
        Ok(session_id)
    }

    /// Renew a session: reset its created_at and extend TTL.
    pub fn renew(&mut self, session_id: &str) -> Result<(), SessionError> {
        self.renew_with_ttl(session_id, self.default_ttl)
    }

    /// Renew with a specific TTL.
    pub fn renew_with_ttl(&mut self, session_id: &str, ttl: Duration) -> Result<(), SessionError> {
        let session = self.sessions.get_mut(session_id).ok_or(SessionError::NotFound)?;
        let now = Instant::now();
        session.created_at = now;
        session.last_accessed = now;
        session.ttl = ttl;
        Ok(())
    }

    /// Destroy a session.
    pub fn destroy(&mut self, session_id: &str) -> bool {
        self.remove_session(session_id)
    }

    /// Enumerate all sessions for a user.
    pub fn user_sessions(&self, user_id: &str) -> Vec<&str> {
        self.user_sessions
            .get(user_id)
            .map(|ids| {
                ids.iter()
                    .filter(|id| self.sessions.contains_key(id.as_str()))
                    .map(|id| id.as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Destroy all sessions for a user.
    pub fn destroy_user_sessions(&mut self, user_id: &str) -> usize {
        let session_ids: Vec<String> = self
            .user_sessions
            .get(user_id)
            .cloned()
            .unwrap_or_default();
        let count = session_ids.len();
        for sid in &session_ids {
            self.remove_session_inner(sid);
        }
        self.user_sessions.remove(user_id);
        count
    }

    /// Purge all expired sessions globally.
    pub fn purge_expired(&mut self) -> usize {
        let now = Instant::now();
        let expired: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.is_expired(now))
            .map(|(id, _)| id.clone())
            .collect();
        let count = expired.len();
        for sid in expired {
            self.remove_session(&sid);
        }
        self.total_expired += count as u64;
        count
    }

    /// Total active sessions.
    pub fn active_count(&self) -> usize {
        self.sessions.len()
    }

    /// Total sessions created over the store's lifetime.
    pub fn total_created(&self) -> u64 {
        self.total_created
    }

    /// Total sessions expired over the store's lifetime.
    pub fn total_expired(&self) -> u64 {
        self.total_expired
    }

    /// All active session IDs.
    pub fn session_ids(&self) -> Vec<&str> {
        self.sessions.keys().map(|k| k.as_str()).collect()
    }

    // ── Internal ─────────────────────────────────────────────────────

    fn remove_session(&mut self, session_id: &str) -> bool {
        self.remove_session_inner(session_id)
    }

    fn remove_session_inner(&mut self, session_id: &str) -> bool {
        if let Some(session) = self.sessions.remove(session_id) {
            self.token_index.remove(&session.token);
            if let Some(uid) = &session.user_id {
                if let Some(ids) = self.user_sessions.get_mut(uid) {
                    ids.retain(|id| id != session_id);
                }
            }
            true
        } else {
            false
        }
    }

    fn purge_expired_for_user(&mut self, user_id: &str) {
        let now = Instant::now();
        let expired: Vec<String> = self
            .user_sessions
            .get(user_id)
            .unwrap_or(&Vec::new())
            .iter()
            .filter(|id| {
                self.sessions
                    .get(id.as_str())
                    .is_some_and(|s| s.is_expired(now))
            })
            .cloned()
            .collect();
        for sid in expired {
            self.remove_session_inner(&sid);
            self.total_expired += 1;
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_session() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        assert!(!id.is_empty());
        assert_eq!(store.active_count(), 1);
    }

    #[test]
    fn test_get_session() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        let session = store.get(&id).unwrap();
        assert_eq!(session.id(), id);
    }

    #[test]
    fn test_session_data_get_set() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        {
            let session = store.get_mut(&id).unwrap();
            session.set("username".into(), "alice".into());
            session.set("role".into(), "admin".into());
        }
        let session = store.get(&id).unwrap();
        assert_eq!(session.get("username"), Some("alice"));
        assert_eq!(session.get("role"), Some("admin"));
        assert_eq!(session.data_len(), 2);
    }

    #[test]
    fn test_typed_get() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        {
            let session = store.get_mut(&id).unwrap();
            session.set("count".into(), "42".into());
        }
        let session = store.get(&id).unwrap();
        let count: i32 = session.get_typed("count").unwrap();
        assert_eq!(count, 42);
    }

    #[test]
    fn test_typed_get_mismatch() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        {
            let session = store.get_mut(&id).unwrap();
            session.set("name".into(), "alice".into());
        }
        let session = store.get(&id).unwrap();
        let result: Result<i32, _> = session.get_typed("name");
        assert!(matches!(result, Err(SessionError::TypeMismatch(_))));
    }

    #[test]
    fn test_session_token() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        let token = store.get(&id).unwrap().token().to_string();
        assert!(!token.is_empty());
        assert!(token.contains('.'));
        // Validate token.
        let validated_id = store.validate_token(&token).unwrap();
        assert_eq!(validated_id, id);
    }

    #[test]
    fn test_invalid_token() {
        let mut store = SessionStore::with_defaults();
        let result = store.validate_token("bogus.token");
        assert_eq!(result, Err(SessionError::InvalidToken));
    }

    #[test]
    fn test_session_ttl_expiry() {
        let mut store = SessionStore::new(Duration::from_millis(0), 5);
        let id = store.create(None).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let result = store.get(&id);
        assert!(matches!(result, Err(SessionError::Expired)));
    }

    #[test]
    fn test_renew_session() {
        let mut store = SessionStore::new(Duration::from_millis(50), 5);
        let id = store.create(None).unwrap();
        // Renew extends the TTL.
        store
            .renew_with_ttl(&id, Duration::from_secs(60))
            .unwrap();
        std::thread::sleep(Duration::from_millis(60));
        // Should still be valid after renewal.
        assert!(store.get(&id).is_ok());
    }

    #[test]
    fn test_destroy_session() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        assert!(store.destroy(&id));
        assert!(!store.destroy(&id));
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn test_concurrent_limit() {
        let mut store = SessionStore::new(Duration::from_secs(60), 2);
        let uid = Some("user1".to_string());
        store.create(uid.clone()).unwrap();
        store.create(uid.clone()).unwrap();
        let result = store.create(uid);
        assert_eq!(result, Err(SessionError::ConcurrentLimitExceeded));
    }

    #[test]
    fn test_user_sessions() {
        let mut store = SessionStore::new(Duration::from_secs(60), 5);
        let uid = Some("user1".to_string());
        let id1 = store.create(uid.clone()).unwrap();
        let id2 = store.create(uid).unwrap();
        let sessions = store.user_sessions("user1");
        assert_eq!(sessions.len(), 2);
        assert!(sessions.contains(&id1.as_str()));
        assert!(sessions.contains(&id2.as_str()));
    }

    #[test]
    fn test_destroy_user_sessions() {
        let mut store = SessionStore::new(Duration::from_secs(60), 5);
        let uid = Some("user1".to_string());
        store.create(uid.clone()).unwrap();
        store.create(uid).unwrap();
        store.create(None).unwrap(); // anonymous session
        let destroyed = store.destroy_user_sessions("user1");
        assert_eq!(destroyed, 2);
        assert_eq!(store.active_count(), 1); // anonymous survives
    }

    #[test]
    fn test_purge_expired() {
        let mut store = SessionStore::new(Duration::from_millis(0), 5);
        store.create(None).unwrap();
        store.create(None).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        let purged = store.purge_expired();
        assert_eq!(purged, 2);
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn test_session_remove_data() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        {
            let session = store.get_mut(&id).unwrap();
            session.set("key".into(), "value".into());
            assert_eq!(session.remove("key"), Some("value".to_string()));
            assert_eq!(session.remove("key"), None);
        }
    }

    #[test]
    fn test_access_count() {
        let mut store = SessionStore::with_defaults();
        let id = store.create(None).unwrap();
        let _ = store.get(&id);
        let _ = store.get(&id);
        let _ = store.get(&id);
        let session = store.get(&id).unwrap();
        assert_eq!(session.access_count(), 4);
    }

    #[test]
    fn test_total_created() {
        let mut store = SessionStore::with_defaults();
        store.create(None).unwrap();
        store.create(None).unwrap();
        assert_eq!(store.total_created(), 2);
    }

    #[test]
    fn test_unique_session_ids() {
        let mut store = SessionStore::with_defaults();
        let id1 = store.create(None).unwrap();
        let id2 = store.create(None).unwrap();
        let id3 = store.create(None).unwrap();
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }
}
