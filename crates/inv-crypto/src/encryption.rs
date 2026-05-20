use aws_lc_rs::aead::{AES_256_GCM, Aad, LessSafeKey, Nonce, UnboundKey};
use aws_lc_rs::hmac;
use aws_lc_rs::rand as aws_rand;
use zeroize::{Zeroize, ZeroizeOnDrop};

const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
const TAG_LEN: usize = 16;

/// A 256-bit symmetric key for AES-256-GCM encryption.
/// Automatically zeroized on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SymmetricKey {
    bytes: [u8; KEY_LEN],
}

impl SymmetricKey {
    /// Generate a new key from the OS CSPRNG.
    pub fn generate() -> Self {
        let mut bytes = [0u8; KEY_LEN];
        aws_rand::fill(&mut bytes).expect("OS RNG available");
        Self { bytes }
    }

    /// Construct from an existing 32-byte slice.
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self { bytes }
    }

    /// Raw key bytes. Handle with care.
    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.bytes
    }

    fn to_less_safe_key(&self) -> LessSafeKey {
        let unbound = UnboundKey::new(&AES_256_GCM, &self.bytes).expect("valid AES-256-GCM key");
        LessSafeKey::new(unbound)
    }
}

impl std::fmt::Debug for SymmetricKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymmetricKey")
            .field("bytes", &"[REDACTED]")
            .finish()
    }
}

/// AES-256-GCM authenticated encryption.
///
/// Encrypted output format: `[nonce: 12 bytes] || [ciphertext + tag]`
///
/// Uses aws-lc-rs (FIPS 140-3 validated) as the crypto backend.
pub struct Aes256Gcm {
    key: LessSafeKey,
}

impl Aes256Gcm {
    /// Create from a SymmetricKey.
    pub fn new(key: &SymmetricKey) -> Self {
        Self {
            key: key.to_less_safe_key(),
        }
    }

    /// Encrypt plaintext with optional additional authenticated data (AAD).
    /// Returns `nonce || ciphertext || tag`.
    pub fn encrypt(&self, plaintext: &[u8], aad: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        aws_rand::fill(&mut nonce_bytes).map_err(|_| EncryptionError::RngFailure)?;
        let nonce = Nonce::assume_unique_for_key(nonce_bytes);

        let mut in_out = Vec::with_capacity(NONCE_LEN + plaintext.len() + TAG_LEN);
        in_out.extend_from_slice(&nonce_bytes);
        in_out.extend_from_slice(plaintext);

        let tag = self
            .key
            .seal_in_place_separate_tag(
                nonce,
                Aad::from(aad),
                &mut in_out[NONCE_LEN..NONCE_LEN + plaintext.len()],
            )
            .map_err(|_| EncryptionError::EncryptFailed)?;

        in_out.extend_from_slice(tag.as_ref());
        Ok(in_out)
    }

    /// Decrypt ciphertext produced by [`encrypt`].
    /// Expects `[nonce: 12] || [ciphertext + tag]`.
    pub fn decrypt(&self, ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>, EncryptionError> {
        if ciphertext.len() < NONCE_LEN + TAG_LEN {
            return Err(EncryptionError::CiphertextTooShort);
        }

        let (nonce_bytes, ct_and_tag) = ciphertext.split_at(NONCE_LEN);
        let nonce = Nonce::assume_unique_for_key(nonce_bytes.try_into().unwrap());

        let mut buf = ct_and_tag.to_vec();
        let plaintext = self
            .key
            .open_in_place(nonce, Aad::from(aad), &mut buf)
            .map_err(|_| EncryptionError::DecryptFailed)?;

        Ok(plaintext.to_vec())
    }
}

/// HMAC-SHA256 for message authentication (e.g., gossip messages, wire frames).
pub struct HmacSha256;

impl HmacSha256 {
    /// Compute HMAC-SHA256 over data with the given key.
    pub fn sign(key: &[u8], data: &[u8]) -> [u8; 32] {
        let signing_key = hmac::Key::new(hmac::HMAC_SHA256, key);
        let tag = hmac::sign(&signing_key, data);
        let mut out = [0u8; 32];
        out.copy_from_slice(tag.as_ref());
        out
    }

    /// Verify an HMAC-SHA256 tag in constant time.
    pub fn verify(key: &[u8], data: &[u8], expected: &[u8]) -> bool {
        let signing_key = hmac::Key::new(hmac::HMAC_SHA256, key);
        hmac::verify(&signing_key, data, expected).is_ok()
    }

    /// Compute and return the first 8 bytes (truncated HMAC for wire frames).
    pub fn sign_truncated(key: &[u8], data: &[u8]) -> [u8; 8] {
        let full = Self::sign(key, data);
        let mut out = [0u8; 8];
        out.copy_from_slice(&full[..8]);
        out
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EncryptionError {
    #[error("encryption failed")]
    EncryptFailed,
    #[error("decryption failed (corrupted or tampered ciphertext)")]
    DecryptFailed,
    #[error("ciphertext too short")]
    CiphertextTooShort,
    #[error("random number generator failure")]
    RngFailure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_key_generate_and_debug() {
        let key = SymmetricKey::generate();
        let debug = format!("{key:?}");
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn aes_gcm_roundtrip() {
        let key = SymmetricKey::generate();
        let cipher = Aes256Gcm::new(&key);

        let plaintext = b"sensitive data for invisible infrastructure";
        let aad = b"workload-id:abc123";

        let encrypted = cipher.encrypt(plaintext, aad).unwrap();
        assert_ne!(&encrypted[NONCE_LEN..], plaintext);

        let decrypted = cipher.decrypt(&encrypted, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn aes_gcm_wrong_aad_fails() {
        let key = SymmetricKey::generate();
        let cipher = Aes256Gcm::new(&key);

        let encrypted = cipher.encrypt(b"hello", b"correct-aad").unwrap();
        let result = cipher.decrypt(&encrypted, b"wrong-aad");
        assert!(result.is_err());
    }

    #[test]
    fn aes_gcm_tampered_ciphertext_fails() {
        let key = SymmetricKey::generate();
        let cipher = Aes256Gcm::new(&key);

        let mut encrypted = cipher.encrypt(b"hello", b"").unwrap();
        // Flip a byte in the ciphertext
        encrypted[NONCE_LEN + 2] ^= 0xFF;

        let result = cipher.decrypt(&encrypted, b"");
        assert!(result.is_err());
    }

    #[test]
    fn aes_gcm_wrong_key_fails() {
        let key1 = SymmetricKey::generate();
        let key2 = SymmetricKey::generate();
        let cipher1 = Aes256Gcm::new(&key1);
        let cipher2 = Aes256Gcm::new(&key2);

        let encrypted = cipher1.encrypt(b"hello", b"").unwrap();
        let result = cipher2.decrypt(&encrypted, b"");
        assert!(result.is_err());
    }

    #[test]
    fn aes_gcm_short_ciphertext_fails() {
        let key = SymmetricKey::generate();
        let cipher = Aes256Gcm::new(&key);
        assert!(cipher.decrypt(&[0u8; 10], b"").is_err());
    }

    #[test]
    fn aes_gcm_empty_plaintext() {
        let key = SymmetricKey::generate();
        let cipher = Aes256Gcm::new(&key);

        let encrypted = cipher.encrypt(b"", b"aad").unwrap();
        let decrypted = cipher.decrypt(&encrypted, b"aad").unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn hmac_sign_and_verify() {
        let key = b"org-shared-secret-key";
        let data = b"gossip message payload";

        let tag = HmacSha256::sign(key, data);
        assert!(HmacSha256::verify(key, data, &tag));
        assert!(!HmacSha256::verify(key, b"wrong data", &tag));
        assert!(!HmacSha256::verify(b"wrong-key", data, &tag));
    }

    #[test]
    fn hmac_truncated() {
        let key = b"org-shared-secret-key";
        let data = b"wire frame header";

        let truncated = HmacSha256::sign_truncated(key, data);
        assert_eq!(truncated.len(), 8);

        let full = HmacSha256::sign(key, data);
        assert_eq!(&full[..8], &truncated);
    }
}
