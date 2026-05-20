//! HRP Phase 3: Cryptographic integrity for replicated messages.
//!
//! Provides epoch-rotating HMAC-SHA256 write tokens that bind
//! (epoch, sequence, term, payload_hash) together. This ensures:
//!
//! - **Integrity**: Tampered payloads are detected (HMAC mismatch)
//! - **Replay protection**: Monotonic sequence numbers per leader
//! - **Forward secrecy**: Epoch keys derived independently via HKDF
//!
//! The master secret is shared across all Raft nodes (from config).
//! Each epoch key is derived as `HKDF-SHA256(master_secret, epoch)`.

use std::sync::atomic::{AtomicU64, Ordering};

use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

// ============================================================================
// Write Token
// ============================================================================

/// An HMAC-signed token attached to each replicated message.
///
/// Followers verify the token to ensure the message was produced by
/// a legitimate leader holding the current epoch key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteToken {
    /// Key epoch — rotates on term changes
    pub epoch: u64,
    /// Monotonic sequence number (per-leader)
    pub sequence: u64,
    /// Raft term that produced this token
    pub term: u64,
    /// HMAC-SHA256 over (epoch || sequence || term || SHA256(payload))
    pub hmac: [u8; 32],
}

impl WriteToken {
    /// Byte size when serialized inline in the wire format.
    /// 8 (epoch) + 8 (sequence) + 8 (term) + 32 (hmac) = 56 bytes.
    pub const WIRE_SIZE: usize = 56;

    /// Serialize to fixed-size bytes for wire format embedding.
    pub fn to_bytes(&self) -> [u8; Self::WIRE_SIZE] {
        let mut buf = [0u8; Self::WIRE_SIZE];
        buf[0..8].copy_from_slice(&self.epoch.to_be_bytes());
        buf[8..16].copy_from_slice(&self.sequence.to_be_bytes());
        buf[16..24].copy_from_slice(&self.term.to_be_bytes());
        buf[24..56].copy_from_slice(&self.hmac);
        buf
    }

    /// Deserialize from fixed-size bytes.
    pub fn from_bytes(buf: &[u8; Self::WIRE_SIZE]) -> Self {
        let epoch = u64::from_be_bytes(buf[0..8].try_into().unwrap());
        let sequence = u64::from_be_bytes(buf[8..16].try_into().unwrap());
        let term = u64::from_be_bytes(buf[16..24].try_into().unwrap());
        let mut hmac = [0u8; 32];
        hmac.copy_from_slice(&buf[24..56]);
        Self {
            epoch,
            sequence,
            term,
            hmac,
        }
    }
}

// ============================================================================
// Epoch Key Manager
// ============================================================================

/// Manages rotating HMAC keys for write token generation/verification.
///
/// Each epoch key is derived independently from the master secret via HKDF,
/// providing forward secrecy: compromising the current key doesn't reveal
/// past keys.
pub struct EpochKeyManager {
    /// Shared secret across all Raft nodes (from config)
    master_secret: [u8; 32],
    /// Current epoch number (advances on term changes)
    current_epoch: AtomicU64,
    /// Monotonic sequence counter (per-node)
    sequence: AtomicU64,
}

impl EpochKeyManager {
    /// Create a new epoch key manager with a shared master secret.
    pub fn new(master_secret: [u8; 32]) -> Self {
        Self {
            master_secret,
            current_epoch: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
        }
    }

    /// Create from a hex-encoded master secret string.
    pub fn from_hex(hex_secret: &str) -> Result<Self, String> {
        let bytes =
            hex::decode(hex_secret).map_err(|e| format!("invalid hex master secret: {}", e))?;
        if bytes.len() != 32 {
            return Err(format!(
                "master secret must be 32 bytes, got {}",
                bytes.len()
            ));
        }
        let mut secret = [0u8; 32];
        secret.copy_from_slice(&bytes);
        Ok(Self::new(secret))
    }

    /// Derive the epoch key for a given epoch number.
    ///
    /// Uses HKDF-SHA256: `key = HKDF-Expand(HKDF-Extract(epoch_bytes, master_secret), "hrp-epoch-key", 32)`
    pub fn derive_epoch_key(&self, epoch: u64) -> [u8; 32] {
        let hk = Hkdf::<Sha256>::new(Some(&epoch.to_be_bytes()), &self.master_secret);
        let mut key = [0u8; 32];
        hk.expand(b"hrp-epoch-key", &mut key)
            .expect("32 bytes is a valid HKDF-SHA256 output length");
        key
    }

    /// Get the current epoch's key.
    pub fn current_key(&self) -> [u8; 32] {
        self.derive_epoch_key(self.current_epoch.load(Ordering::Acquire))
    }

    /// Get the current epoch number.
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::Acquire)
    }

    /// Advance to a new epoch (typically on Raft term change).
    /// Wraps around on u64::MAX to avoid overflow panic.
    pub fn advance_epoch(&self) -> u64 {
        let prev = self.current_epoch.fetch_add(1, Ordering::AcqRel);
        prev.wrapping_add(1)
    }

    /// Set the epoch directly (for synchronization with leader).
    pub fn set_epoch(&self, epoch: u64) {
        self.current_epoch.store(epoch, Ordering::Release);
    }

    /// Get the next monotonic sequence number.
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::AcqRel)
    }

    /// Generate a write token for a payload.
    ///
    /// The token binds (epoch, sequence, term, SHA256(payload)) with HMAC-SHA256.
    pub fn generate_token(&self, term: u64, payload: &[u8]) -> WriteToken {
        let epoch = self.current_epoch();
        let sequence = self.next_sequence();
        let epoch_key = self.derive_epoch_key(epoch);

        let hmac = compute_token_hmac(&epoch_key, epoch, sequence, term, payload);

        WriteToken {
            epoch,
            sequence,
            term,
            hmac,
        }
    }

    /// Verify a write token against the expected payload.
    ///
    /// Accepts tokens signed with the current epoch key OR the previous
    /// epoch key (grace window for rotation).
    pub fn verify_token(
        &self,
        token: &WriteToken,
        expected_term: u64,
        payload: &[u8],
    ) -> Result<(), SecurityError> {
        // Check term matches
        if token.term != expected_term {
            return Err(SecurityError::TermMismatch {
                token_term: token.term,
                expected: expected_term,
            });
        }

        let current_epoch = self.current_epoch();

        // Accept current epoch or previous epoch (grace window)
        if token.epoch != current_epoch && token.epoch + 1 != current_epoch {
            return Err(SecurityError::EpochTooOld {
                token_epoch: token.epoch,
                current: current_epoch,
            });
        }

        // Recompute HMAC with the token's epoch key
        let epoch_key = self.derive_epoch_key(token.epoch);
        let expected_hmac =
            compute_token_hmac(&epoch_key, token.epoch, token.sequence, token.term, payload);

        if token.hmac != expected_hmac {
            return Err(SecurityError::HmacMismatch);
        }

        Ok(())
    }
}

/// Compute the HMAC-SHA256 for a write token.
fn compute_token_hmac(
    epoch_key: &[u8; 32],
    epoch: u64,
    sequence: u64,
    term: u64,
    payload: &[u8],
) -> [u8; 32] {
    // Hash the payload first (so HMAC input is fixed-size prefix + hash)
    let payload_hash = <Sha256 as sha2::Digest>::digest(payload);

    let mut mac = HmacSha256::new_from_slice(epoch_key).expect("HMAC can take key of any size");
    mac.update(&epoch.to_be_bytes());
    mac.update(&sequence.to_be_bytes());
    mac.update(&term.to_be_bytes());
    mac.update(&payload_hash);

    let result = mac.finalize().into_bytes();
    let mut hmac = [0u8; 32];
    hmac.copy_from_slice(&result);
    hmac
}

// ============================================================================
// Errors
// ============================================================================

/// Security verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityError {
    /// HMAC doesn't match — payload was tampered with
    HmacMismatch,
    /// Token term doesn't match the expected Raft term
    TermMismatch { token_term: u64, expected: u64 },
    /// Token epoch is too old (not current or previous)
    EpochTooOld { token_epoch: u64, current: u64 },
    /// Sequence number is not monotonically increasing (replay)
    SequenceReplay { received: u64, last_seen: u64 },
}

impl std::fmt::Display for SecurityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityError::HmacMismatch => write!(f, "HMAC verification failed — message tampered"),
            SecurityError::TermMismatch {
                token_term,
                expected,
            } => {
                write!(
                    f,
                    "term mismatch: token has {}, expected {}",
                    token_term, expected
                )
            }
            SecurityError::EpochTooOld {
                token_epoch,
                current,
            } => {
                write!(
                    f,
                    "epoch too old: token has {}, current is {}",
                    token_epoch, current
                )
            }
            SecurityError::SequenceReplay {
                received,
                last_seen,
            } => {
                write!(
                    f,
                    "sequence replay: received {}, last seen {}",
                    received, last_seen
                )
            }
        }
    }
}

impl std::error::Error for SecurityError {}

/// Tracks per-peer sequence numbers for replay detection.
pub struct SequenceTracker {
    /// Last seen sequence per peer node
    peers: std::sync::RwLock<std::collections::HashMap<String, u64>>,
}

impl SequenceTracker {
    pub fn new() -> Self {
        Self {
            peers: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Check and update the sequence number for a peer.
    /// First message from a new peer is always accepted.
    /// Subsequent messages must have strictly increasing sequence numbers.
    pub fn check_and_update(&self, peer_id: &str, sequence: u64) -> Result<(), SecurityError> {
        let mut peers = self.peers.write().unwrap_or_else(|e| e.into_inner());

        // First message from this peer: accept any sequence
        if let Some(&last_seen) = peers.get(peer_id) {
            // Subsequent messages: require strictly increasing (rejects replay)
            if sequence <= last_seen {
                return Err(SecurityError::SequenceReplay {
                    received: sequence,
                    last_seen,
                });
            }
        }

        peers.insert(peer_id.to_string(), sequence);
        Ok(())
    }
}

impl Default for SequenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_secret() -> [u8; 32] {
        let mut s = [0u8; 32];
        s[0] = 0x42;
        s[31] = 0xFF;
        s
    }

    #[test]
    fn test_write_token_roundtrip() {
        let mgr = EpochKeyManager::new(test_secret());
        let payload = b"INSERT INTO users VALUES (1, 'Alice')";
        let token = mgr.generate_token(5, payload);

        assert_eq!(token.epoch, 0);
        assert_eq!(token.sequence, 0);
        assert_eq!(token.term, 5);

        // Verify succeeds
        assert!(mgr.verify_token(&token, 5, payload).is_ok());
    }

    #[test]
    fn test_write_token_tamper_rejected() {
        let mgr = EpochKeyManager::new(test_secret());
        let payload = b"INSERT INTO users VALUES (1, 'Alice')";
        let token = mgr.generate_token(5, payload);

        // Tamper with payload
        let tampered = b"INSERT INTO users VALUES (1, 'Eve')";
        let result = mgr.verify_token(&token, 5, tampered);
        assert_eq!(result, Err(SecurityError::HmacMismatch));
    }

    #[test]
    fn test_write_token_replay_rejected() {
        let tracker = SequenceTracker::new();

        // First message at sequence 5
        assert!(tracker.check_and_update("node1", 5).is_ok());
        // Sequence 10 is fine
        assert!(tracker.check_and_update("node1", 10).is_ok());
        // Sequence 3 is a replay (< 10)
        let result = tracker.check_and_update("node1", 3);
        assert!(matches!(result, Err(SecurityError::SequenceReplay { .. })));
        // Sequence 10 again is also a replay (== last_seen, not strictly >)
        let result = tracker.check_and_update("node1", 10);
        assert!(matches!(result, Err(SecurityError::SequenceReplay { .. })));
        // Sequence 11 is fine (strictly > 10)
        assert!(tracker.check_and_update("node1", 11).is_ok());
    }

    #[test]
    fn test_write_token_wrong_term_rejected() {
        let mgr = EpochKeyManager::new(test_secret());
        let payload = b"some data";
        let token = mgr.generate_token(5, payload);

        let result = mgr.verify_token(&token, 6, payload);
        assert_eq!(
            result,
            Err(SecurityError::TermMismatch {
                token_term: 5,
                expected: 6,
            })
        );
    }

    #[test]
    fn test_epoch_key_rotation() {
        let mgr = EpochKeyManager::new(test_secret());
        let payload = b"data";

        // Generate token at epoch 0
        let token_e0 = mgr.generate_token(1, payload);
        assert_eq!(token_e0.epoch, 0);

        // Advance to epoch 1
        mgr.advance_epoch();
        assert_eq!(mgr.current_epoch(), 1);

        // Token from epoch 0 is still accepted (grace window: current - 1)
        assert!(mgr.verify_token(&token_e0, 1, payload).is_ok());

        // Generate new token at epoch 1
        let token_e1 = mgr.generate_token(1, payload);
        assert_eq!(token_e1.epoch, 1);
        assert!(mgr.verify_token(&token_e1, 1, payload).is_ok());

        // Advance to epoch 2 — epoch 0 tokens are now too old
        mgr.advance_epoch();
        assert_eq!(mgr.current_epoch(), 2);
        let result = mgr.verify_token(&token_e0, 1, payload);
        assert!(matches!(result, Err(SecurityError::EpochTooOld { .. })));

        // Epoch 1 token still accepted (grace window)
        assert!(mgr.verify_token(&token_e1, 1, payload).is_ok());
    }

    #[test]
    fn test_epoch_key_derivation_deterministic() {
        let mgr1 = EpochKeyManager::new(test_secret());
        let mgr2 = EpochKeyManager::new(test_secret());

        // Same master secret + epoch → same key
        assert_eq!(mgr1.derive_epoch_key(0), mgr2.derive_epoch_key(0));
        assert_eq!(mgr1.derive_epoch_key(42), mgr2.derive_epoch_key(42));

        // Different epochs → different keys
        assert_ne!(mgr1.derive_epoch_key(0), mgr1.derive_epoch_key(1));
    }

    #[test]
    fn test_from_hex() {
        let hex = "0042000000000000000000000000000000000000000000000000000000000000";
        let mgr = EpochKeyManager::from_hex(hex).unwrap();
        assert_eq!(mgr.master_secret[0], 0x00);
        assert_eq!(mgr.master_secret[1], 0x42);

        // Wrong length
        assert!(EpochKeyManager::from_hex("abcd").is_err());
        // Invalid hex
        assert!(EpochKeyManager::from_hex("zzzz").is_err());
    }

    #[test]
    fn test_write_token_wire_bytes_roundtrip() {
        let mgr = EpochKeyManager::new(test_secret());
        let token = mgr.generate_token(7, b"payload");

        let bytes = token.to_bytes();
        assert_eq!(bytes.len(), WriteToken::WIRE_SIZE);

        let recovered = WriteToken::from_bytes(&bytes);
        assert_eq!(token, recovered);
    }
}
