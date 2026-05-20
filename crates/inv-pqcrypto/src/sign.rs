//! ML-DSA-65 digital signatures (NIST FIPS 204) — real implementation.
//!
//! Uses the `fips204` crate (IntegrityChain) for genuine ML-DSA-65 lattice-based
//! signatures, and `ed25519-dalek` for classical Ed25519 in hybrid mode.

use ed25519_dalek::{Signer as _, Verifier as _};
use fips204::ml_dsa_65;
use fips204::traits::{SerDes, Signer, Verifier};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::PqCryptoError;

/// ML-DSA-65 public key size in bytes.
pub const MLDSA65_PK_SIZE: usize = ml_dsa_65::PK_LEN;
/// ML-DSA-65 secret key size in bytes.
pub const MLDSA65_SK_SIZE: usize = ml_dsa_65::SK_LEN;
/// ML-DSA-65 signature size in bytes.
pub const MLDSA65_SIG_SIZE: usize = ml_dsa_65::SIG_LEN;
/// Classical Ed25519 public key size.
pub const ED25519_PK_SIZE: usize = 32;
/// Classical Ed25519 secret key size (seed form).
pub const ED25519_SK_SIZE: usize = 32;
/// Classical Ed25519 signature size.
pub const ED25519_SIG_SIZE: usize = 64;

/// A zeroizable wrapper around signing secret key bytes.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SigningSecretKey(Vec<u8>);

impl SigningSecretKey {
    /// Create from raw bytes.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// Length of the secret key in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the secret key is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for SigningSecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SigningSecretKey")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// An ML-DSA-65 public verification key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PqPublicKey(pub Vec<u8>);

impl PqPublicKey {
    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// An ML-DSA-65 digital signature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PqSignature(pub Vec<u8>);

impl PqSignature {
    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// An ML-DSA-65 signing key pair (public key + secret key).
#[derive(Debug, Clone)]
pub struct SigningKeyPair {
    /// The public verification key (1952 bytes for ML-DSA-65).
    pub public_key: PqPublicKey,
    /// The secret signing key (4032 bytes for ML-DSA-65).
    pub secret_key: SigningSecretKey,
}

/// Generate an ML-DSA-65 signing key pair using real lattice-based cryptography.
///
/// # Panics
///
/// Panics if the underlying CSPRNG fails (should never happen with a properly
/// seeded OS RNG).
pub fn sign_keygen() -> SigningKeyPair {
    let (pk, sk) = ml_dsa_65::try_keygen().expect("ML-DSA-65 keygen should not fail with OS RNG");

    SigningKeyPair {
        public_key: PqPublicKey(pk.into_bytes().to_vec()),
        secret_key: SigningSecretKey::new(sk.into_bytes().to_vec()),
    }
}

/// Sign a message with an ML-DSA-65 secret key.
///
/// # Errors
///
/// Returns [`PqCryptoError::InvalidKeySize`] if the secret key length is wrong.
/// Returns [`PqCryptoError::Signing`] if deserialization or signing fails.
pub fn sign_message(
    secret_key: &SigningSecretKey,
    message: &[u8],
) -> Result<PqSignature, PqCryptoError> {
    if secret_key.len() != MLDSA65_SK_SIZE {
        return Err(PqCryptoError::InvalidKeySize {
            expected: MLDSA65_SK_SIZE,
            actual: secret_key.len(),
        });
    }

    // Deserialize the private key
    let sk_bytes: [u8; ml_dsa_65::SK_LEN] = secret_key
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Signing("invalid ML-DSA-65 secret key encoding".into()))?;
    let sk = ml_dsa_65::PrivateKey::try_from_bytes(sk_bytes)
        .map_err(|e| PqCryptoError::Signing(format!("ML-DSA-65 key deserialization: {e}")))?;

    // Real ML-DSA-65 signing (empty context string)
    let sig = sk
        .try_sign(message, &[])
        .map_err(|e| PqCryptoError::Signing(format!("ML-DSA-65 signing failed: {e}")))?;

    Ok(PqSignature(sig.to_vec()))
}

/// Verify an ML-DSA-65 signature against a public key and message.
///
/// # Errors
///
/// Returns [`PqCryptoError::InvalidKeySize`] if the public key length is wrong.
/// Returns [`PqCryptoError::Verification`] if the signature or key is malformed.
pub fn verify_signature(
    public_key: &PqPublicKey,
    message: &[u8],
    signature: &PqSignature,
) -> Result<bool, PqCryptoError> {
    if public_key.as_bytes().len() != MLDSA65_PK_SIZE {
        return Err(PqCryptoError::InvalidKeySize {
            expected: MLDSA65_PK_SIZE,
            actual: public_key.as_bytes().len(),
        });
    }

    if signature.as_bytes().len() != MLDSA65_SIG_SIZE {
        return Err(PqCryptoError::Verification(
            "signature length mismatch".into(),
        ));
    }

    // Deserialize the public key
    let pk_bytes: [u8; ml_dsa_65::PK_LEN] = public_key
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Verification("invalid ML-DSA-65 public key encoding".into()))?;
    let pk = ml_dsa_65::PublicKey::try_from_bytes(pk_bytes)
        .map_err(|e| PqCryptoError::Verification(format!("ML-DSA-65 key deserialization: {e}")))?;

    // Deserialize the signature
    let sig_bytes: [u8; ml_dsa_65::SIG_LEN] = signature
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Verification("invalid ML-DSA-65 signature encoding".into()))?;

    // Real ML-DSA-65 verification (empty context string)
    Ok(pk.verify(message, &sig_bytes, &[]))
}

// ---------------------------------------------------------------------------
// Hybrid Signing: Ed25519 + ML-DSA-65
// ---------------------------------------------------------------------------

/// A hybrid signing key pair combining classical Ed25519 keys with ML-DSA-65.
#[derive(Debug, Clone)]
pub struct HybridSigningKeyPair {
    /// Classical Ed25519 public key (32 bytes).
    pub classical_pk: Vec<u8>,
    /// Classical Ed25519 secret key (32-byte seed, zeroizable).
    pub classical_sk: SigningSecretKey,
    /// Post-quantum ML-DSA-65 key pair.
    pub pq_keypair: SigningKeyPair,
}

/// A hybrid signature combining classical and post-quantum signatures.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HybridSignature {
    /// Classical Ed25519 signature (64 bytes).
    pub classical_sig: Vec<u8>,
    /// Post-quantum ML-DSA-65 signature.
    pub pq_sig: PqSignature,
}

/// Generate a hybrid Ed25519 + ML-DSA-65 signing key pair using real cryptography.
pub fn hybrid_sign_keygen() -> HybridSigningKeyPair {
    // Real Ed25519 key generation
    let mut rng = rand_core_06::OsRng;
    let ed_signing_key = ed25519_dalek::SigningKey::generate(&mut rng);
    let ed_verifying_key = ed_signing_key.verifying_key();

    // Real ML-DSA-65 key generation
    let pq_keypair = sign_keygen();

    HybridSigningKeyPair {
        classical_pk: ed_verifying_key.as_bytes().to_vec(),
        classical_sk: SigningSecretKey::new(ed_signing_key.to_bytes().to_vec()),
        pq_keypair,
    }
}

/// Produce a hybrid signature (Ed25519 + ML-DSA-65) over a message.
///
/// Both the classical and post-quantum signatures must be present for the
/// hybrid signature to be considered valid.
///
/// # Errors
///
/// Returns [`PqCryptoError::Signing`] if the classical secret key is the
/// wrong size, or delegates to [`sign_message`] for PQ errors.
pub fn hybrid_sign(
    keypair: &HybridSigningKeyPair,
    message: &[u8],
) -> Result<HybridSignature, PqCryptoError> {
    if keypair.classical_sk.len() != ED25519_SK_SIZE {
        return Err(PqCryptoError::Signing(
            "classical secret key must be 32 bytes".into(),
        ));
    }

    // Real Ed25519 signing
    let sk_bytes: [u8; 32] = keypair
        .classical_sk
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Signing("classical secret key must be 32 bytes".into()))?;
    let ed_signing_key = ed25519_dalek::SigningKey::from_bytes(&sk_bytes);
    let ed_sig = ed_signing_key.sign(message);

    // Real PQ signature
    let pq_sig = sign_message(&keypair.pq_keypair.secret_key, message)?;

    Ok(HybridSignature {
        classical_sig: ed_sig.to_bytes().to_vec(),
        pq_sig,
    })
}

/// Verify a hybrid signature. Both the classical and PQ components must pass.
///
/// # Errors
///
/// Returns errors from the underlying verification steps.
pub fn hybrid_verify(
    keypair: &HybridSigningKeyPair,
    message: &[u8],
    signature: &HybridSignature,
) -> Result<bool, PqCryptoError> {
    // Verify classical Ed25519 component
    if signature.classical_sig.len() != ED25519_SIG_SIZE {
        return Ok(false);
    }

    let pk_bytes: [u8; 32] =
        keypair.classical_pk.as_slice().try_into().map_err(|_| {
            PqCryptoError::Verification("classical public key must be 32 bytes".into())
        })?;
    let ed_verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes)
        .map_err(|e| PqCryptoError::Verification(format!("invalid Ed25519 public key: {e}")))?;

    let sig_bytes: [u8; 64] =
        signature.classical_sig.as_slice().try_into().map_err(|_| {
            PqCryptoError::Verification("classical signature must be 64 bytes".into())
        })?;
    let ed_sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);

    let classical_valid = ed_verifying_key.verify(message, &ed_sig).is_ok();

    // Verify PQ component
    let pq_valid = verify_signature(&keypair.pq_keypair.public_key, message, &signature.pq_sig)?;

    // Both must pass
    Ok(classical_valid && pq_valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_keygen_produces_correct_sizes() {
        let kp = sign_keygen();
        assert_eq!(kp.public_key.as_bytes().len(), MLDSA65_PK_SIZE);
        assert_eq!(kp.secret_key.len(), MLDSA65_SK_SIZE);
    }

    #[test]
    fn sign_and_verify_roundtrip() {
        let kp = sign_keygen();
        let message = b"invisible infrastructure control plane";
        let sig = sign_message(&kp.secret_key, message).unwrap();
        assert_eq!(sig.as_bytes().len(), MLDSA65_SIG_SIZE);

        let valid = verify_signature(&kp.public_key, message, &sig).unwrap();
        assert!(valid, "signature must verify against the correct message");
    }

    #[test]
    fn verify_rejects_wrong_message() {
        let kp = sign_keygen();
        let sig = sign_message(&kp.secret_key, b"original message").unwrap();
        let valid = verify_signature(&kp.public_key, b"tampered message", &sig).unwrap();
        assert!(
            !valid,
            "signature must NOT verify against a different message"
        );
    }

    #[test]
    fn verify_rejects_wrong_key() {
        let kp1 = sign_keygen();
        let kp2 = sign_keygen();
        let message = b"test message";
        let sig = sign_message(&kp1.secret_key, message).unwrap();
        let valid = verify_signature(&kp2.public_key, message, &sig).unwrap();
        assert!(
            !valid,
            "signature must NOT verify against a different public key"
        );
    }

    #[test]
    fn sign_rejects_wrong_sk_size() {
        let bad_sk = SigningSecretKey::new(vec![0u8; 100]);
        let result = sign_message(&bad_sk, b"hello");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PqCryptoError::InvalidKeySize { .. }
        ));
    }

    #[test]
    fn verify_rejects_wrong_pk_size() {
        let bad_pk = PqPublicKey(vec![0u8; 100]);
        let sig = PqSignature(vec![0u8; MLDSA65_SIG_SIZE]);
        let result = verify_signature(&bad_pk, b"hello", &sig);
        assert!(result.is_err());
    }

    #[test]
    fn verify_rejects_wrong_sig_size() {
        let kp = sign_keygen();
        let bad_sig = PqSignature(vec![0u8; 100]);
        let result = verify_signature(&kp.public_key, b"hello", &bad_sig);
        assert!(result.is_err());
    }

    #[test]
    fn signing_secret_key_debug_is_redacted() {
        let kp = sign_keygen();
        let debug = format!("{:?}", kp.secret_key);
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn signature_serialization_roundtrip() {
        let kp = sign_keygen();
        let sig = sign_message(&kp.secret_key, b"test data").unwrap();
        let json = serde_json::to_string(&sig).unwrap();
        let deserialized: PqSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig, deserialized);
    }

    #[test]
    fn hybrid_sign_keygen_produces_correct_sizes() {
        let hkp = hybrid_sign_keygen();
        assert_eq!(hkp.classical_pk.len(), ED25519_PK_SIZE);
        assert_eq!(hkp.classical_sk.len(), ED25519_SK_SIZE);
        assert_eq!(hkp.pq_keypair.public_key.as_bytes().len(), MLDSA65_PK_SIZE);
        assert_eq!(hkp.pq_keypair.secret_key.len(), MLDSA65_SK_SIZE);
    }

    #[test]
    fn hybrid_sign_and_verify_roundtrip() {
        let hkp = hybrid_sign_keygen();
        let message = b"hybrid post-quantum signature test";
        let sig = hybrid_sign(&hkp, message).unwrap();

        assert_eq!(sig.classical_sig.len(), ED25519_SIG_SIZE);
        assert_eq!(sig.pq_sig.as_bytes().len(), MLDSA65_SIG_SIZE);

        let valid = hybrid_verify(&hkp, message, &sig).unwrap();
        assert!(valid, "hybrid signature must verify");
    }

    #[test]
    fn hybrid_verify_rejects_tampered_classical_sig() {
        let hkp = hybrid_sign_keygen();
        let message = b"test";
        let mut sig = hybrid_sign(&hkp, message).unwrap();
        sig.classical_sig = vec![0u8; ED25519_SIG_SIZE]; // invalid signature bytes
        let valid = hybrid_verify(&hkp, message, &sig).unwrap();
        assert!(!valid, "tampered classical signature must not verify");
    }

    #[test]
    fn hybrid_verify_rejects_bad_classical_sig_size() {
        let hkp = hybrid_sign_keygen();
        let message = b"test";
        let mut sig = hybrid_sign(&hkp, message).unwrap();
        sig.classical_sig = vec![0u8; 10]; // wrong size
        let valid = hybrid_verify(&hkp, message, &sig).unwrap();
        assert!(!valid);
    }
}
