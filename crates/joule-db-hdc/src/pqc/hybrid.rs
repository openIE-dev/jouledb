//! Hybrid Encryption combining classical and post-quantum KEMs
//!
//! Provides defense-in-depth by combining:
//! - Classical key exchange (simulated X25519/ECDH)
//! - Post-quantum ML-KEM
//!
//! The shared secret is derived from both to ensure security even if
//! one primitive is broken.

use super::common::{SecureZeroingVec, Sha3_256, Shake256};
use super::ml_kem::{MlKem768, MlKemCiphertext, MlKemDecapsulationKey, MlKemEncapsulationKey};
use super::{PqcError, PqcResult};
use rand::Rng;

// ============================================================================
// Simulated Classical KEM (X25519-like)
// ============================================================================

/// Simulated X25519 public key (32 bytes)
#[derive(Clone)]
pub struct ClassicalPublicKey {
    data: [u8; 32],
}

/// Simulated X25519 secret key (32 bytes)
#[derive(Clone)]
pub struct ClassicalSecretKey {
    data: SecureZeroingVec,
}

/// Simulated X25519 ciphertext (ephemeral public key, 32 bytes)
#[derive(Clone)]
pub struct ClassicalCiphertext {
    data: [u8; 32],
}

/// Simple classical KEM based on DH
pub struct ClassicalKem;

impl ClassicalKem {
    /// Generate key pair
    pub fn keygen() -> (ClassicalPublicKey, ClassicalSecretKey) {
        let mut rng = rand::rng();
        let mut sk_bytes = vec![0u8; 32];
        rng.fill(&mut sk_bytes[..]);

        // Simulate scalar multiplication for public key
        // In reality, this would be X25519 base point multiplication
        let pk_bytes = Sha3_256::hash(&sk_bytes);

        (
            ClassicalPublicKey { data: pk_bytes },
            ClassicalSecretKey {
                data: SecureZeroingVec::from_vec(sk_bytes),
            },
        )
    }

    /// Encapsulate: generate ephemeral keypair, compute shared secret
    pub fn encapsulate(pk: &ClassicalPublicKey) -> (ClassicalCiphertext, [u8; 32]) {
        let mut rng = rand::rng();
        let mut eph_sk = [0u8; 32];
        rng.fill(&mut eph_sk);

        // Ephemeral public key (ciphertext)
        let eph_pk = Sha3_256::hash(&eph_sk);

        // Shared secret = H(pk || eph_pk)
        // Simulates ECDH: both sides know pk and eph_pk (= ciphertext)
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(&pk.data);
        input.extend_from_slice(&eph_pk);
        let ss = Sha3_256::hash(&input);

        (ClassicalCiphertext { data: eph_pk }, ss)
    }

    /// Decapsulate: compute shared secret from ciphertext
    pub fn decapsulate(sk: &ClassicalSecretKey, ct: &ClassicalCiphertext) -> [u8; 32] {
        // Recompute pk = H(sk), then shared secret = H(pk || eph_pk)
        // Simulates ECDH: both sides know pk and eph_pk (= ct)
        let pk = Sha3_256::hash(sk.data.as_slice());
        let mut input = Vec::with_capacity(64);
        input.extend_from_slice(&pk);
        input.extend_from_slice(&ct.data);
        Sha3_256::hash(&input)
    }
}

// ============================================================================
// Hybrid KEM
// ============================================================================

/// Hybrid public key (classical + post-quantum)
#[derive(Clone)]
pub struct HybridPublicKey {
    classical: ClassicalPublicKey,
    pq: MlKemEncapsulationKey,
}

impl HybridPublicKey {
    /// Create from components
    pub fn new(classical: ClassicalPublicKey, pq: MlKemEncapsulationKey) -> Self {
        Self { classical, pq }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.classical.data);
        bytes.extend_from_slice(self.pq.as_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.len() < 32 {
            return Err(PqcError::InvalidKey);
        }

        let classical = ClassicalPublicKey {
            data: bytes[..32].try_into().unwrap(),
        };

        let pq = MlKemEncapsulationKey::from_bytes(&bytes[32..], MlKem768::PARAMS)?;

        Ok(Self { classical, pq })
    }

    /// Get size
    pub fn size() -> usize {
        32 + MlKem768::PARAMS.encapsulation_key_size()
    }
}

/// Hybrid secret key (classical + post-quantum)
#[derive(Clone)]
pub struct HybridSecretKey {
    classical: ClassicalSecretKey,
    pq: MlKemDecapsulationKey,
}

impl HybridSecretKey {
    /// Create from components
    pub fn new(classical: ClassicalSecretKey, pq: MlKemDecapsulationKey) -> Self {
        Self { classical, pq }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.classical.data.as_slice());
        bytes.extend_from_slice(self.pq.as_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.len() < 32 {
            return Err(PqcError::InvalidKey);
        }

        let classical = ClassicalSecretKey {
            data: SecureZeroingVec::from_vec(bytes[..32].to_vec()),
        };

        let pq = MlKemDecapsulationKey::from_bytes(&bytes[32..], MlKem768::PARAMS)?;

        Ok(Self { classical, pq })
    }

    /// Get size
    pub fn size() -> usize {
        32 + MlKem768::PARAMS.decapsulation_key_size()
    }
}

/// Hybrid ciphertext
#[derive(Clone)]
pub struct HybridCiphertext {
    classical: ClassicalCiphertext,
    pq: MlKemCiphertext,
}

impl HybridCiphertext {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.classical.data);
        bytes.extend_from_slice(self.pq.as_bytes());
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> PqcResult<Self> {
        if bytes.len() < 32 {
            return Err(PqcError::InvalidCiphertext);
        }

        let classical = ClassicalCiphertext {
            data: bytes[..32].try_into().unwrap(),
        };

        let pq = MlKemCiphertext::from_bytes(&bytes[32..], MlKem768::PARAMS)?;

        Ok(Self { classical, pq })
    }

    /// Get size
    pub fn size() -> usize {
        32 + MlKem768::PARAMS.ciphertext_size()
    }
}

/// Hybrid shared secret
#[derive(Clone, Debug)]
pub struct HybridSharedSecret {
    data: SecureZeroingVec,
}

impl HybridSharedSecret {
    /// Get bytes
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    /// Size (always 32)
    pub const fn size() -> usize {
        32
    }
}

impl PartialEq for HybridSharedSecret {
    fn eq(&self, other: &Self) -> bool {
        use super::common::ConstantTime;
        ConstantTime::ct_eq(self.data.as_slice(), other.data.as_slice())
    }
}

/// Hybrid KEM combining classical and post-quantum
pub struct HybridKem;

impl HybridKem {
    /// Generate hybrid key pair
    pub fn keygen() -> PqcResult<(HybridPublicKey, HybridSecretKey)> {
        let (classical_pk, classical_sk) = ClassicalKem::keygen();
        let (pq_pk, pq_sk) = MlKem768::keygen()?;

        Ok((
            HybridPublicKey::new(classical_pk, pq_pk),
            HybridSecretKey::new(classical_sk, pq_sk),
        ))
    }

    /// Encapsulate: create ciphertext and shared secret
    pub fn encapsulate(pk: &HybridPublicKey) -> PqcResult<(HybridCiphertext, HybridSharedSecret)> {
        // Classical encapsulation
        let (classical_ct, classical_ss) = ClassicalKem::encapsulate(&pk.classical);

        // Post-quantum encapsulation
        let (pq_ct, pq_ss) = MlKem768::encapsulate(&pk.pq)?;

        // Combine shared secrets: H(classical_ss || pq_ss)
        let combined = Self::combine_secrets(&classical_ss, pq_ss.as_bytes());

        Ok((
            HybridCiphertext {
                classical: classical_ct,
                pq: pq_ct,
            },
            HybridSharedSecret {
                data: SecureZeroingVec::from_vec(combined.to_vec()),
            },
        ))
    }

    /// Decapsulate: recover shared secret from ciphertext
    pub fn decapsulate(
        sk: &HybridSecretKey,
        ct: &HybridCiphertext,
    ) -> PqcResult<HybridSharedSecret> {
        // Classical decapsulation
        let classical_ss = ClassicalKem::decapsulate(&sk.classical, &ct.classical);

        // Post-quantum decapsulation
        let pq_ss = MlKem768::decapsulate(&sk.pq, &ct.pq)?;

        // Combine shared secrets
        let combined = Self::combine_secrets(&classical_ss, pq_ss.as_bytes());

        Ok(HybridSharedSecret {
            data: SecureZeroingVec::from_vec(combined.to_vec()),
        })
    }

    /// Combine two shared secrets using KDF
    fn combine_secrets(ss1: &[u8], ss2: &[u8]) -> [u8; 32] {
        let mut input = Vec::with_capacity(ss1.len() + ss2.len() + 16);
        // Domain separator
        input.extend_from_slice(b"HybridKEM-Combine");
        input.extend_from_slice(ss1);
        input.extend_from_slice(ss2);

        Sha3_256::hash(&input)
    }
}

// ============================================================================
// Hybrid Encryption (KEM + DEM)
// ============================================================================

/// Authenticated encryption using derived key
pub struct HybridEncryption;

impl HybridEncryption {
    /// Encrypt message using hybrid KEM
    pub fn encrypt(pk: &HybridPublicKey, plaintext: &[u8]) -> PqcResult<Vec<u8>> {
        // Encapsulate to get shared secret
        let (ct, ss) = HybridKem::encapsulate(pk)?;

        // Derive encryption key and nonce
        let (enc_key, nonce) = Self::derive_key_nonce(ss.as_bytes());

        // Encrypt with ChaCha20-like stream cipher (simplified)
        let ciphertext = Self::stream_cipher_encrypt(&enc_key, &nonce, plaintext);

        // Compute authentication tag
        let tag = Self::compute_tag(&enc_key, &ct.to_bytes(), &ciphertext);

        // Output: KEM ciphertext || encrypted data || tag
        let mut output = ct.to_bytes();
        output.extend(ciphertext);
        output.extend_from_slice(&tag);

        Ok(output)
    }

    /// Decrypt message using hybrid KEM
    pub fn decrypt(sk: &HybridSecretKey, encrypted: &[u8]) -> PqcResult<Vec<u8>> {
        let ct_size = HybridCiphertext::size();
        let tag_size = 32;

        if encrypted.len() < ct_size + tag_size {
            return Err(PqcError::InvalidCiphertext);
        }

        // Parse components
        let ct = HybridCiphertext::from_bytes(&encrypted[..ct_size])?;
        let ciphertext = &encrypted[ct_size..encrypted.len() - tag_size];
        let tag = &encrypted[encrypted.len() - tag_size..];

        // Decapsulate to get shared secret
        let ss = HybridKem::decapsulate(sk, &ct)?;

        // Derive keys
        let (enc_key, nonce) = Self::derive_key_nonce(ss.as_bytes());

        // Verify tag
        let expected_tag = Self::compute_tag(&enc_key, &ct.to_bytes(), ciphertext);
        if !super::common::ConstantTime::ct_eq(tag, &expected_tag) {
            return Err(PqcError::DecapsulationFailed);
        }

        // Decrypt
        let plaintext = Self::stream_cipher_decrypt(&enc_key, &nonce, ciphertext);

        Ok(plaintext)
    }

    /// Derive encryption key and nonce from shared secret
    fn derive_key_nonce(ss: &[u8]) -> ([u8; 32], [u8; 12]) {
        let expanded = Shake256::xof(&[b"HybridEnc-KeyNonce".as_slice(), ss].concat(), 44);

        let mut key = [0u8; 32];
        let mut nonce = [0u8; 12];
        key.copy_from_slice(&expanded[..32]);
        nonce.copy_from_slice(&expanded[32..44]);

        (key, nonce)
    }

    /// Simplified stream cipher (XOR with keystream)
    fn stream_cipher_encrypt(key: &[u8], nonce: &[u8], plaintext: &[u8]) -> Vec<u8> {
        let keystream = Self::generate_keystream(key, nonce, plaintext.len());
        plaintext
            .iter()
            .zip(keystream.iter())
            .map(|(p, k)| p ^ k)
            .collect()
    }

    /// Simplified stream cipher decrypt (same as encrypt for XOR)
    fn stream_cipher_decrypt(key: &[u8], nonce: &[u8], ciphertext: &[u8]) -> Vec<u8> {
        Self::stream_cipher_encrypt(key, nonce, ciphertext)
    }

    /// Generate keystream using SHAKE
    fn generate_keystream(key: &[u8], nonce: &[u8], len: usize) -> Vec<u8> {
        Shake256::xof(&[key, nonce].concat(), len)
    }

    /// Compute authentication tag
    fn compute_tag(key: &[u8], ct: &[u8], ciphertext: &[u8]) -> [u8; 32] {
        let mut input = Vec::new();
        input.extend_from_slice(b"HybridEnc-Tag");
        input.extend_from_slice(key);
        input.extend_from_slice(ct);
        input.extend_from_slice(ciphertext);
        Sha3_256::hash(&input)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classical_kem_roundtrip() {
        let (pk, sk) = ClassicalKem::keygen();
        let (ct, ss_enc) = ClassicalKem::encapsulate(&pk);
        let ss_dec = ClassicalKem::decapsulate(&sk, &ct);

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_hybrid_kem_roundtrip() {
        let (pk, sk) = HybridKem::keygen().expect("keygen failed");

        let (ct, ss_enc) = HybridKem::encapsulate(&pk).expect("encapsulate failed");
        let ss_dec = HybridKem::decapsulate(&sk, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_hybrid_encryption_roundtrip() {
        let (pk, sk) = HybridKem::keygen().expect("keygen failed");

        let plaintext = b"Hello, hybrid post-quantum encryption!";
        let encrypted = HybridEncryption::encrypt(&pk, plaintext).expect("encrypt failed");
        let decrypted = HybridEncryption::decrypt(&sk, &encrypted).expect("decrypt failed");

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_hybrid_encryption_tamper_detection() {
        let (pk, sk) = HybridKem::keygen().expect("keygen failed");

        let plaintext = b"Secret message";
        let mut encrypted = HybridEncryption::encrypt(&pk, plaintext).expect("encrypt failed");

        // Tamper with ciphertext
        let tamper_idx = encrypted.len() / 2;
        if let Some(byte) = encrypted.get_mut(tamper_idx) {
            *byte ^= 0xFF;
        }

        let result = HybridEncryption::decrypt(&sk, &encrypted);
        assert!(result.is_err(), "Should detect tampering");
    }

    #[test]
    fn test_key_serialization() {
        let (pk, sk) = HybridKem::keygen().expect("keygen failed");

        // Serialize and deserialize public key
        let pk_bytes = pk.to_bytes();
        let pk_restored = HybridPublicKey::from_bytes(&pk_bytes).expect("deserialize pk failed");

        // Serialize and deserialize secret key
        let sk_bytes = sk.to_bytes();
        let sk_restored = HybridSecretKey::from_bytes(&sk_bytes).expect("deserialize sk failed");

        // Verify roundtrip works
        let (ct, ss_enc) = HybridKem::encapsulate(&pk_restored).expect("encapsulate failed");
        let ss_dec = HybridKem::decapsulate(&sk_restored, &ct).expect("decapsulate failed");

        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_empty_message() {
        let (pk, sk) = HybridKem::keygen().expect("keygen failed");

        let plaintext = b"";
        let encrypted = HybridEncryption::encrypt(&pk, plaintext).expect("encrypt failed");
        let decrypted = HybridEncryption::decrypt(&sk, &encrypted).expect("decrypt failed");

        assert_eq!(plaintext.as_slice(), decrypted.as_slice());
    }

    #[test]
    fn test_large_message() {
        let (pk, sk) = HybridKem::keygen().expect("keygen failed");

        let plaintext = vec![0xAB; 10000];
        let encrypted = HybridEncryption::encrypt(&pk, &plaintext).expect("encrypt failed");
        let decrypted = HybridEncryption::decrypt(&sk, &encrypted).expect("decrypt failed");

        assert_eq!(plaintext, decrypted);
    }
}
