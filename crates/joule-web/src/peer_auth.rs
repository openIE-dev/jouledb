//! Peer authentication and key exchange — challenge-response protocol, peer
//! identity via public key hash, simplified Diffie-Hellman key derivation,
//! session key generation, authentication state machine, mutual auth, timeouts.

use std::collections::HashMap;
use std::fmt;

// ── AuthState ───────────────────────────────────────────────────────────────

/// State in the authentication state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthState {
    /// Initial state, not yet started.
    Init,
    /// Challenge sent, awaiting response.
    Challenged,
    /// Successfully authenticated.
    Authenticated,
    /// Session has expired.
    Expired,
    /// Authentication failed.
    Failed,
}

impl fmt::Display for AuthState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthState::Init => write!(f, "Init"),
            AuthState::Challenged => write!(f, "Challenged"),
            AuthState::Authenticated => write!(f, "Authenticated"),
            AuthState::Expired => write!(f, "Expired"),
            AuthState::Failed => write!(f, "Failed"),
        }
    }
}

// ── PeerIdentity ────────────────────────────────────────────────────────────

/// Identity of a peer derived from a public key hash.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PeerIdentity {
    /// Hex-encoded hash of the peer's public key.
    pub key_hash: String,
    /// Human-readable label.
    pub label: String,
}

impl PeerIdentity {
    pub fn new(key_hash: impl Into<String>, label: impl Into<String>) -> Self {
        Self { key_hash: key_hash.into(), label: label.into() }
    }

    /// Simplified hash from raw bytes — uses a basic FNV-1a hash for simulation.
    pub fn from_public_key(key: &[u8], label: impl Into<String>) -> Self {
        let hash = fnv1a_hash(key);
        Self {
            key_hash: format!("{:016x}", hash),
            label: label.into(),
        }
    }
}

impl fmt::Display for PeerIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PeerIdentity({}:{})", self.label, &self.key_hash[..8.min(self.key_hash.len())])
    }
}

// ── Inline FNV-1a ───────────────────────────────────────────────────────────

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x00000100000001B3);
    }
    hash
}

// ── AuthChallenge / AuthResponse ────────────────────────────────────────────

/// A challenge sent to a peer.
#[derive(Debug, Clone)]
pub struct AuthChallenge {
    pub nonce: u64,
    pub challenger_id: String,
    pub created_at: u64,
    pub expires_at: u64,
}

impl AuthChallenge {
    pub fn new(nonce: u64, challenger_id: impl Into<String>, now: u64, timeout: u64) -> Self {
        Self {
            nonce,
            challenger_id: challenger_id.into(),
            created_at: now,
            expires_at: now + timeout,
        }
    }

    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

/// A response to an authentication challenge.
#[derive(Debug, Clone)]
pub struct AuthResponse {
    pub nonce: u64,
    pub responder_id: String,
    pub proof: u64,
}

impl AuthResponse {
    pub fn new(nonce: u64, responder_id: impl Into<String>, secret: &[u8]) -> Self {
        // proof = hash(nonce || secret)
        let mut data = nonce.to_le_bytes().to_vec();
        data.extend_from_slice(secret);
        let proof = fnv1a_hash(&data);
        Self {
            nonce,
            responder_id: responder_id.into(),
            proof,
        }
    }
}

// ── DhKeyPair ───────────────────────────────────────────────────────────────

/// Simplified Diffie-Hellman key pair (small prime for testing, NOT secure).
#[derive(Debug, Clone)]
pub struct DhKeyPair {
    pub private_key: u64,
    pub public_key: u64,
    prime: u64,
    generator: u64,
}

impl DhKeyPair {
    /// Create with a given private key. Uses a fixed small prime for simulation.
    pub fn new(private_key: u64) -> Self {
        let prime: u64 = 104729; // small prime for demo
        let generator: u64 = 2;
        let public_key = mod_exp(generator, private_key, prime);
        Self { private_key, public_key, prime, generator }
    }

    /// Derive shared secret from the other party's public key.
    pub fn derive_shared_secret(&self, other_public: u64) -> u64 {
        mod_exp(other_public, self.private_key, self.prime)
    }
}

/// Modular exponentiation: base^exp mod modulus.
fn mod_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 { return 0; }
    let mut result: u128 = 1;
    let mut b = (base as u128) % (modulus as u128);
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * b) % (modulus as u128);
        }
        exp >>= 1;
        b = (b * b) % (modulus as u128);
    }
    result as u64
}

// ── AuthSession ─────────────────────────────────────────────────────────────

/// An authentication session between two peers.
#[derive(Debug, Clone)]
pub struct AuthSession {
    pub peer_id: String,
    pub state: AuthState,
    pub session_key: Option<u64>,
    pub challenge: Option<AuthChallenge>,
    pub started_at: u64,
    pub authenticated_at: Option<u64>,
    pub expires_at: Option<u64>,
}

impl AuthSession {
    pub fn new(peer_id: impl Into<String>, now: u64) -> Self {
        Self {
            peer_id: peer_id.into(),
            state: AuthState::Init,
            session_key: None,
            challenge: None,
            started_at: now,
            authenticated_at: None,
            expires_at: None,
        }
    }

    /// Check if the session is expired.
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at.map(|e| now >= e).unwrap_or(false)
    }
}

// ── AuthManager ─────────────────────────────────────────────────────────────

/// Manages authentication sessions with multiple peers.
pub struct AuthManager {
    self_id: String,
    self_secret: Vec<u8>,
    sessions: HashMap<String, AuthSession>,
    challenge_timeout: u64,
    session_lifetime: u64,
    next_nonce: u64,
    current_tick: u64,
    total_auth_success: u64,
    total_auth_failure: u64,
}

impl AuthManager {
    pub fn new(self_id: impl Into<String>, self_secret: Vec<u8>) -> Self {
        Self {
            self_id: self_id.into(),
            self_secret,
            sessions: HashMap::new(),
            challenge_timeout: 30,
            session_lifetime: 3600,
            next_nonce: 1,
            current_tick: 0,
            total_auth_success: 0,
            total_auth_failure: 0,
        }
    }

    pub fn with_challenge_timeout(mut self, timeout: u64) -> Self {
        self.challenge_timeout = timeout;
        self
    }

    pub fn with_session_lifetime(mut self, lifetime: u64) -> Self {
        self.session_lifetime = lifetime;
        self
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Start authentication with a peer — create and return a challenge.
    pub fn create_challenge(&mut self, peer_id: impl Into<String>) -> AuthChallenge {
        let pid = peer_id.into();
        let nonce = self.next_nonce;
        self.next_nonce += 1;
        let challenge = AuthChallenge::new(nonce, &self.self_id, self.current_tick, self.challenge_timeout);
        let mut session = AuthSession::new(&pid, self.current_tick);
        session.state = AuthState::Challenged;
        session.challenge = Some(challenge.clone());
        self.sessions.insert(pid, session);
        challenge
    }

    /// Verify a response to a challenge.
    pub fn verify_response(&mut self, response: &AuthResponse, peer_secret: &[u8]) -> bool {
        let session = match self.sessions.get_mut(&response.responder_id) {
            Some(s) if s.state == AuthState::Challenged => s,
            _ => return false,
        };

        let challenge = match &session.challenge {
            Some(c) => c,
            None => return false,
        };

        if challenge.is_expired(self.current_tick) {
            session.state = AuthState::Failed;
            self.total_auth_failure += 1;
            return false;
        }

        if response.nonce != challenge.nonce {
            session.state = AuthState::Failed;
            self.total_auth_failure += 1;
            return false;
        }

        // Recompute expected proof
        let expected = AuthResponse::new(response.nonce, &response.responder_id, peer_secret);
        if response.proof != expected.proof {
            session.state = AuthState::Failed;
            self.total_auth_failure += 1;
            return false;
        }

        // Generate session key from shared data
        let mut key_material = response.nonce.to_le_bytes().to_vec();
        key_material.extend_from_slice(peer_secret);
        key_material.extend_from_slice(&self.self_secret);
        session.session_key = Some(fnv1a_hash(&key_material));
        session.state = AuthState::Authenticated;
        session.authenticated_at = Some(self.current_tick);
        session.expires_at = Some(self.current_tick + self.session_lifetime);
        self.total_auth_success += 1;
        true
    }

    /// Get the current auth state for a peer.
    pub fn auth_state(&self, peer_id: &str) -> Option<AuthState> {
        self.sessions.get(peer_id).map(|s| s.state)
    }

    /// Get the session key for an authenticated peer.
    pub fn session_key(&self, peer_id: &str) -> Option<u64> {
        self.sessions.get(peer_id).and_then(|s| {
            if s.state == AuthState::Authenticated { s.session_key } else { None }
        })
    }

    /// Expire timed-out sessions.
    pub fn expire_sessions(&mut self) -> usize {
        let mut expired = 0;
        for session in self.sessions.values_mut() {
            if session.is_expired(self.current_tick) && session.state == AuthState::Authenticated {
                session.state = AuthState::Expired;
                expired += 1;
            }
        }
        expired
    }

    /// Number of active authenticated sessions.
    pub fn authenticated_count(&self) -> usize {
        self.sessions.values().filter(|s| s.state == AuthState::Authenticated).count()
    }

    /// Total success count.
    pub fn success_count(&self) -> u64 {
        self.total_auth_success
    }

    /// Total failure count.
    pub fn failure_count(&self) -> u64 {
        self.total_auth_failure
    }

    /// Self id.
    pub fn self_id(&self) -> &str {
        &self.self_id
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_identity_from_key() {
        let id = PeerIdentity::from_public_key(b"my-public-key", "alice");
        assert_eq!(id.label, "alice");
        assert_eq!(id.key_hash.len(), 16);
    }

    #[test]
    fn test_peer_identity_display() {
        let id = PeerIdentity::new("abcdef0123456789", "bob");
        let s = format!("{}", id);
        assert!(s.contains("bob"));
        assert!(s.contains("abcdef01"));
    }

    #[test]
    fn test_auth_state_display() {
        assert_eq!(format!("{}", AuthState::Init), "Init");
        assert_eq!(format!("{}", AuthState::Authenticated), "Authenticated");
    }

    #[test]
    fn test_challenge_expiry() {
        let c = AuthChallenge::new(1, "alice", 10, 20);
        assert!(!c.is_expired(20));
        assert!(c.is_expired(30));
    }

    #[test]
    fn test_auth_response_proof_deterministic() {
        let r1 = AuthResponse::new(42, "bob", b"secret");
        let r2 = AuthResponse::new(42, "bob", b"secret");
        assert_eq!(r1.proof, r2.proof);
    }

    #[test]
    fn test_auth_response_proof_varies_with_nonce() {
        let r1 = AuthResponse::new(1, "bob", b"secret");
        let r2 = AuthResponse::new(2, "bob", b"secret");
        assert_ne!(r1.proof, r2.proof);
    }

    #[test]
    fn test_dh_shared_secret() {
        let alice = DhKeyPair::new(7);
        let bob = DhKeyPair::new(13);
        let secret_a = alice.derive_shared_secret(bob.public_key);
        let secret_b = bob.derive_shared_secret(alice.public_key);
        assert_eq!(secret_a, secret_b);
    }

    #[test]
    fn test_dh_different_keys() {
        let a = DhKeyPair::new(3);
        let b = DhKeyPair::new(5);
        assert_ne!(a.public_key, b.public_key);
    }

    #[test]
    fn test_full_auth_flow() {
        let peer_secret = b"peer-secret".to_vec();
        let mut mgr = AuthManager::new("server", b"server-secret".to_vec());

        let challenge = mgr.create_challenge("client");
        assert_eq!(mgr.auth_state("client"), Some(AuthState::Challenged));

        let response = AuthResponse::new(challenge.nonce, "client", &peer_secret);
        assert!(mgr.verify_response(&response, &peer_secret));
        assert_eq!(mgr.auth_state("client"), Some(AuthState::Authenticated));
        assert!(mgr.session_key("client").is_some());
    }

    #[test]
    fn test_auth_wrong_secret() {
        let mut mgr = AuthManager::new("server", b"ss".to_vec());
        let challenge = mgr.create_challenge("bad");
        let response = AuthResponse::new(challenge.nonce, "bad", b"wrong");
        assert!(!mgr.verify_response(&response, b"correct"));
        assert_eq!(mgr.auth_state("bad"), Some(AuthState::Failed));
    }

    #[test]
    fn test_auth_expired_challenge() {
        let mut mgr = AuthManager::new("s", b"s".to_vec()).with_challenge_timeout(5);
        mgr.tick(0);
        let challenge = mgr.create_challenge("c");
        mgr.tick(10);
        let response = AuthResponse::new(challenge.nonce, "c", b"sec");
        assert!(!mgr.verify_response(&response, b"sec"));
        assert_eq!(mgr.auth_state("c"), Some(AuthState::Failed));
    }

    #[test]
    fn test_session_expiry() {
        let mut mgr = AuthManager::new("s", b"s".to_vec())
            .with_session_lifetime(10);
        mgr.tick(0);
        let ch = mgr.create_challenge("c");
        let resp = AuthResponse::new(ch.nonce, "c", b"k");
        mgr.verify_response(&resp, b"k");
        mgr.tick(20);
        let expired = mgr.expire_sessions();
        assert_eq!(expired, 1);
        assert_eq!(mgr.auth_state("c"), Some(AuthState::Expired));
    }

    #[test]
    fn test_authenticated_count() {
        let mut mgr = AuthManager::new("s", b"s".to_vec());
        let ch1 = mgr.create_challenge("a");
        let ch2 = mgr.create_challenge("b");
        mgr.verify_response(&AuthResponse::new(ch1.nonce, "a", b"k"), b"k");
        mgr.verify_response(&AuthResponse::new(ch2.nonce, "b", b"k"), b"k");
        assert_eq!(mgr.authenticated_count(), 2);
    }

    #[test]
    fn test_success_failure_counts() {
        let mut mgr = AuthManager::new("s", b"s".to_vec());
        let ch = mgr.create_challenge("ok");
        mgr.verify_response(&AuthResponse::new(ch.nonce, "ok", b"k"), b"k");
        let ch2 = mgr.create_challenge("bad");
        mgr.verify_response(&AuthResponse::new(ch2.nonce, "bad", b"wrong"), b"right");
        assert_eq!(mgr.success_count(), 1);
        assert_eq!(mgr.failure_count(), 1);
    }

    #[test]
    fn test_session_key_none_before_auth() {
        let mut mgr = AuthManager::new("s", b"s".to_vec());
        mgr.create_challenge("c");
        assert!(mgr.session_key("c").is_none());
    }

    #[test]
    fn test_wrong_nonce_response() {
        let mut mgr = AuthManager::new("s", b"s".to_vec());
        mgr.create_challenge("c");
        let response = AuthResponse::new(999, "c", b"k");
        assert!(!mgr.verify_response(&response, b"k"));
    }

    #[test]
    fn test_mod_exp() {
        assert_eq!(mod_exp(2, 10, 1000), 24);
        assert_eq!(mod_exp(2, 0, 1000), 1);
        assert_eq!(mod_exp(5, 1, 100), 5);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        assert_eq!(fnv1a_hash(b"hello"), fnv1a_hash(b"hello"));
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }
}
