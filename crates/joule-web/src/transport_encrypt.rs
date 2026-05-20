//! Transport encryption layer.
//!
//! Wraps a plaintext transport with XOR-based cipher simulation
//! (not real cryptography), key exchange, nonce/counter management,
//! AEAD-like authenticated encryption (encrypt + MAC), key rotation,
//! encrypted frame format (nonce + ciphertext + tag), and decryption
//! with authentication verification.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Transport encryption domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptError {
    /// Key not established.
    NoKey,
    /// Authentication tag mismatch.
    AuthenticationFailed,
    /// Frame too short to contain header.
    FrameTooShort { size: usize, min: usize },
    /// Nonce reuse detected.
    NonceReuse(u64),
    /// Key rotation failed.
    KeyRotationFailed(String),
    /// Invalid frame format.
    InvalidFrame(String),
}

impl fmt::Display for EncryptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoKey => write!(f, "encryption key not established"),
            Self::AuthenticationFailed => write!(f, "authentication tag mismatch"),
            Self::FrameTooShort { size, min } => {
                write!(f, "frame too short: {size} bytes (min {min})")
            }
            Self::NonceReuse(n) => write!(f, "nonce reuse detected: {n}"),
            Self::KeyRotationFailed(reason) => write!(f, "key rotation failed: {reason}"),
            Self::InvalidFrame(msg) => write!(f, "invalid frame: {msg}"),
        }
    }
}

impl std::error::Error for EncryptError {}

// ── Key Material ────────────────────────────────────────────────

/// Symmetric key material (XOR-based simulation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyMaterial {
    bytes: Vec<u8>,
    generation: u64,
}

impl KeyMaterial {
    /// Create key material from raw bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes, generation: 0 }
    }

    /// Create a key with a specific generation number.
    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = generation;
        self
    }

    /// Key length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the key is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Key generation (increments on rotation).
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl fmt::Display for KeyMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Key(len={}, gen={})", self.bytes.len(), self.generation)
    }
}

// ── Encrypted Frame ─────────────────────────────────────────────

/// Wire format: 8-byte nonce + ciphertext + 8-byte tag.
pub const NONCE_SIZE: usize = 8;
pub const TAG_SIZE: usize = 8;
pub const FRAME_OVERHEAD: usize = NONCE_SIZE + TAG_SIZE;

/// An encrypted frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedFrame {
    pub nonce: u64,
    pub ciphertext: Vec<u8>,
    pub tag: Vec<u8>,
}

impl EncryptedFrame {
    /// Serialize to wire bytes: nonce (8) + ciphertext + tag (8).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(FRAME_OVERHEAD + self.ciphertext.len());
        out.extend_from_slice(&self.nonce.to_be_bytes());
        out.extend_from_slice(&self.ciphertext);
        out.extend_from_slice(&self.tag);
        out
    }

    /// Parse from wire bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, EncryptError> {
        if data.len() < FRAME_OVERHEAD {
            return Err(EncryptError::FrameTooShort {
                size: data.len(),
                min: FRAME_OVERHEAD,
            });
        }
        let nonce = u64::from_be_bytes(data[..8].try_into().unwrap());
        let ciphertext = data[8..data.len() - TAG_SIZE].to_vec();
        let tag = data[data.len() - TAG_SIZE..].to_vec();
        Ok(Self { nonce, ciphertext, tag })
    }

    /// Wire size.
    pub fn wire_size(&self) -> usize {
        FRAME_OVERHEAD + self.ciphertext.len()
    }
}

impl fmt::Display for EncryptedFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncryptedFrame(nonce={}, ct={}B, tag={}B)",
            self.nonce,
            self.ciphertext.len(),
            self.tag.len(),
        )
    }
}

// ── Encryption Statistics ───────────────────────────────────────

/// Encryption statistics.
#[derive(Debug, Clone, Default)]
pub struct EncryptionStats {
    pub frames_encrypted: u64,
    pub frames_decrypted: u64,
    pub auth_failures: u64,
    pub key_rotations: u64,
    pub bytes_encrypted: u64,
    pub bytes_decrypted: u64,
}

impl fmt::Display for EncryptionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "encrypted={} decrypted={} auth_fail={} rotations={}",
            self.frames_encrypted,
            self.frames_decrypted,
            self.auth_failures,
            self.key_rotations,
        )
    }
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the encrypted transport.
#[derive(Debug, Clone)]
pub struct EncryptConfig {
    pub auto_rotate_after_frames: u64,
    pub max_key_reuses: u64,
    pub allow_nonce_reuse_check: bool,
}

impl Default for EncryptConfig {
    fn default() -> Self {
        Self {
            auto_rotate_after_frames: 0,
            max_key_reuses: u64::MAX,
            allow_nonce_reuse_check: true,
        }
    }
}

impl EncryptConfig {
    pub fn with_auto_rotate(mut self, frames: u64) -> Self {
        self.auto_rotate_after_frames = frames;
        self
    }

    pub fn with_nonce_reuse_check(mut self, enabled: bool) -> Self {
        self.allow_nonce_reuse_check = enabled;
        self
    }
}

// ── Encrypted Transport ─────────────────────────────────────────

/// Transport layer that encrypts and authenticates all frames.
pub struct EncryptedTransport {
    config: EncryptConfig,
    key: Option<KeyMaterial>,
    nonce_counter: u64,
    seen_nonces: VecDeque<u64>,
    max_seen_nonces: usize,
    frames_since_rotation: u64,
    stats: EncryptionStats,
}

impl EncryptedTransport {
    pub fn new(config: EncryptConfig) -> Self {
        Self {
            config,
            key: None,
            nonce_counter: 0,
            seen_nonces: VecDeque::new(),
            max_seen_nonces: 4096,
            frames_since_rotation: 0,
            stats: EncryptionStats::default(),
        }
    }

    /// Establish key material (simulated key exchange).
    pub fn establish_key(&mut self, key: KeyMaterial) {
        self.key = Some(key);
        self.nonce_counter = 0;
        self.frames_since_rotation = 0;
    }

    /// Whether a key has been established.
    pub fn has_key(&self) -> bool {
        self.key.is_some()
    }

    /// Current key generation.
    pub fn key_generation(&self) -> Option<u64> {
        self.key.as_ref().map(|k| k.generation())
    }

    /// Encrypt plaintext into an encrypted frame.
    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<EncryptedFrame, EncryptError> {
        let key = self.key.as_ref().ok_or(EncryptError::NoKey)?;

        let nonce = self.nonce_counter;
        self.nonce_counter += 1;

        // XOR cipher with key bytes (cycled).
        let ciphertext = xor_cipher(plaintext, &key.bytes, nonce);

        // Compute MAC: simple XOR-based hash (simulation only).
        let tag = compute_tag(&ciphertext, &key.bytes, nonce);

        self.stats.frames_encrypted += 1;
        self.stats.bytes_encrypted += plaintext.len() as u64;
        self.frames_since_rotation += 1;

        // Check auto-rotation.
        if self.config.auto_rotate_after_frames > 0
            && self.frames_since_rotation >= self.config.auto_rotate_after_frames
        {
            self.rotate_key()?;
        }

        Ok(EncryptedFrame { nonce, ciphertext, tag })
    }

    /// Decrypt and authenticate an encrypted frame.
    pub fn decrypt(&mut self, frame: &EncryptedFrame) -> Result<Vec<u8>, EncryptError> {
        let key = self.key.as_ref().ok_or(EncryptError::NoKey)?;

        // Nonce reuse check.
        if self.config.allow_nonce_reuse_check && self.seen_nonces.contains(&frame.nonce) {
            return Err(EncryptError::NonceReuse(frame.nonce));
        }

        // Verify tag.
        let expected_tag = compute_tag(&frame.ciphertext, &key.bytes, frame.nonce);
        if frame.tag != expected_tag {
            self.stats.auth_failures += 1;
            return Err(EncryptError::AuthenticationFailed);
        }

        // Decrypt.
        let plaintext = xor_cipher(&frame.ciphertext, &key.bytes, frame.nonce);

        // Record nonce.
        self.seen_nonces.push_back(frame.nonce);
        if self.seen_nonces.len() > self.max_seen_nonces {
            self.seen_nonces.pop_front();
        }

        self.stats.frames_decrypted += 1;
        self.stats.bytes_decrypted += plaintext.len() as u64;

        Ok(plaintext)
    }

    /// Rotate the key (derive new key from current).
    pub fn rotate_key(&mut self) -> Result<(), EncryptError> {
        let old_key = self.key.as_ref().ok_or(EncryptError::NoKey)?;
        let new_gen = old_key.generation() + 1;
        // Simple key derivation: rotate bytes and XOR with generation.
        let mut new_bytes = old_key.bytes.clone();
        for (i, byte) in new_bytes.iter_mut().enumerate() {
            *byte ^= (new_gen as u8).wrapping_add(i as u8);
        }
        self.key = Some(KeyMaterial::new(new_bytes).with_generation(new_gen));
        self.nonce_counter = 0;
        self.frames_since_rotation = 0;
        self.stats.key_rotations += 1;
        Ok(())
    }

    /// Current nonce counter.
    pub fn nonce_counter(&self) -> u64 {
        self.nonce_counter
    }

    /// Get statistics.
    pub fn stats(&self) -> &EncryptionStats {
        &self.stats
    }
}

impl fmt::Display for EncryptedTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncryptedTransport(keyed={}, nonce={}, {})",
            self.has_key(),
            self.nonce_counter,
            self.stats,
        )
    }
}

// ── Cipher Helpers ──────────────────────────────────────────────

/// XOR cipher with nonce-derived key stream (simulation only).
fn xor_cipher(data: &[u8], key: &[u8], nonce: u64) -> Vec<u8> {
    if key.is_empty() {
        return data.to_vec();
    }
    let nonce_bytes = nonce.to_be_bytes();
    data.iter()
        .enumerate()
        .map(|(i, &b)| {
            b ^ key[i % key.len()] ^ nonce_bytes[i % 8]
        })
        .collect()
}

/// Compute a simple XOR-based authentication tag (simulation only).
fn compute_tag(data: &[u8], key: &[u8], nonce: u64) -> Vec<u8> {
    let mut tag = [0u8; TAG_SIZE];
    let nonce_bytes = nonce.to_be_bytes();

    for (i, &b) in data.iter().enumerate() {
        tag[i % TAG_SIZE] ^= b;
    }
    for (i, &b) in key.iter().enumerate() {
        tag[i % TAG_SIZE] ^= b;
    }
    for (i, &b) in nonce_bytes.iter().enumerate() {
        tag[i % TAG_SIZE] ^= b;
    }

    tag.to_vec()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn keyed_transport() -> EncryptedTransport {
        let mut t = EncryptedTransport::new(EncryptConfig::default());
        t.establish_key(KeyMaterial::new(vec![0xAB, 0xCD, 0xEF, 0x01]));
        t
    }

    #[test]
    fn encrypt_without_key_fails() {
        let mut t = EncryptedTransport::new(EncryptConfig::default());
        let err = t.encrypt(b"hello").unwrap_err();
        assert!(matches!(err, EncryptError::NoKey));
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let mut t = keyed_transport();
        let plaintext = b"hello, transport encryption!";
        let frame = t.encrypt(plaintext).unwrap();
        let decrypted = t.decrypt(&frame).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn ciphertext_differs_from_plaintext() {
        let mut t = keyed_transport();
        let plaintext = b"secret data here";
        let frame = t.encrypt(plaintext).unwrap();
        assert_ne!(frame.ciphertext, plaintext.to_vec());
    }

    #[test]
    fn tampered_ciphertext_fails_auth() {
        let mut t = keyed_transport();
        let frame = t.encrypt(b"data").unwrap();
        let mut tampered = frame.clone();
        tampered.ciphertext[0] ^= 0xFF;
        let err = t.decrypt(&tampered).unwrap_err();
        assert!(matches!(err, EncryptError::AuthenticationFailed));
    }

    #[test]
    fn tampered_tag_fails_auth() {
        let mut t = keyed_transport();
        let frame = t.encrypt(b"data").unwrap();
        let mut tampered = frame.clone();
        tampered.tag[0] ^= 0xFF;
        let err = t.decrypt(&tampered).unwrap_err();
        assert!(matches!(err, EncryptError::AuthenticationFailed));
    }

    #[test]
    fn nonce_increments() {
        let mut t = keyed_transport();
        let f1 = t.encrypt(b"a").unwrap();
        let f2 = t.encrypt(b"b").unwrap();
        assert_eq!(f1.nonce, 0);
        assert_eq!(f2.nonce, 1);
    }

    #[test]
    fn nonce_reuse_detected() {
        let mut t = keyed_transport();
        let frame = t.encrypt(b"data").unwrap();
        t.decrypt(&frame).unwrap();
        let err = t.decrypt(&frame).unwrap_err();
        assert!(matches!(err, EncryptError::NonceReuse(_)));
    }

    #[test]
    fn frame_serialization_roundtrip() {
        let frame = EncryptedFrame {
            nonce: 42,
            ciphertext: vec![1, 2, 3, 4, 5],
            tag: vec![10, 20, 30, 40, 50, 60, 70, 80],
        };
        let bytes = frame.to_bytes();
        let parsed = EncryptedFrame::from_bytes(&bytes).unwrap();
        assert_eq!(parsed, frame);
    }

    #[test]
    fn frame_too_short() {
        let err = EncryptedFrame::from_bytes(&[0, 1, 2]).unwrap_err();
        assert!(matches!(err, EncryptError::FrameTooShort { .. }));
    }

    #[test]
    fn key_rotation() {
        let mut t = keyed_transport();
        assert_eq!(t.key_generation(), Some(0));
        t.rotate_key().unwrap();
        assert_eq!(t.key_generation(), Some(1));
        assert_eq!(t.nonce_counter(), 0);
    }

    #[test]
    fn encrypt_decrypt_after_rotation() {
        let mut t = keyed_transport();
        t.rotate_key().unwrap();
        let frame = t.encrypt(b"post-rotation data").unwrap();
        let decrypted = t.decrypt(&frame).unwrap();
        assert_eq!(decrypted, b"post-rotation data");
    }

    #[test]
    fn auto_rotation() {
        let config = EncryptConfig::default().with_auto_rotate(3);
        let mut t = EncryptedTransport::new(config);
        t.establish_key(KeyMaterial::new(vec![1, 2, 3, 4]));
        t.encrypt(b"a").unwrap();
        t.encrypt(b"b").unwrap();
        assert_eq!(t.key_generation(), Some(0));
        t.encrypt(b"c").unwrap(); // Triggers rotation.
        assert_eq!(t.key_generation(), Some(1));
    }

    #[test]
    fn stats_tracking() {
        let mut t = keyed_transport();
        let frame = t.encrypt(b"hello").unwrap();
        t.decrypt(&frame).unwrap();
        assert_eq!(t.stats().frames_encrypted, 1);
        assert_eq!(t.stats().frames_decrypted, 1);
        assert_eq!(t.stats().bytes_encrypted, 5);
        assert_eq!(t.stats().bytes_decrypted, 5);
    }

    #[test]
    fn auth_failure_counted() {
        let mut t = keyed_transport();
        let frame = t.encrypt(b"data").unwrap();
        let mut bad = frame;
        bad.tag[0] ^= 0xFF;
        let _ = t.decrypt(&bad);
        assert_eq!(t.stats().auth_failures, 1);
    }

    #[test]
    fn key_material_display() {
        let k = KeyMaterial::new(vec![1, 2, 3]).with_generation(5);
        let s = format!("{k}");
        assert!(s.contains("len=3"));
        assert!(s.contains("gen=5"));
    }

    #[test]
    fn frame_display() {
        let frame = EncryptedFrame {
            nonce: 7,
            ciphertext: vec![0; 10],
            tag: vec![0; 8],
        };
        let s = format!("{frame}");
        assert!(s.contains("nonce=7"));
        assert!(s.contains("10B"));
    }

    #[test]
    fn transport_display() {
        let t = keyed_transport();
        let s = format!("{t}");
        assert!(s.contains("EncryptedTransport"));
        assert!(s.contains("keyed=true"));
    }

    #[test]
    fn frame_wire_size() {
        let frame = EncryptedFrame {
            nonce: 0,
            ciphertext: vec![0; 100],
            tag: vec![0; 8],
        };
        assert_eq!(frame.wire_size(), 116); // 8 + 100 + 8
    }

    #[test]
    fn different_nonces_produce_different_ciphertext() {
        let mut t = keyed_transport();
        let f1 = t.encrypt(b"same data").unwrap();
        let f2 = t.encrypt(b"same data").unwrap();
        assert_ne!(f1.ciphertext, f2.ciphertext);
    }
}
