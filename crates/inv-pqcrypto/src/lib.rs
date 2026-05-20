//! Post-quantum cryptography for Invisible Infrastructure.
//!
//! This crate provides real implementations of:
//!
//! - **ML-KEM-768** (NIST FIPS 203) — key encapsulation via the `ml-kem` crate (RustCrypto)
//! - **ML-DSA-65** (NIST FIPS 204) — digital signatures via the `fips204` crate (IntegrityChain)
//! - **Hybrid modes** — combining classical (`x25519-dalek`/`ed25519-dalek`) with PQ algorithms
//! - **PQ-hybrid TLS** — configuration and handshake negotiation with real key exchange

pub mod kem;
pub mod sign;
pub mod tls;

// Re-export key types from kem module.
pub use kem::{
    HybridKemKeyPair, KemCiphertext, KemKeyPair, KemPublicKey, KemSharedSecret, SecretKeyBytes,
    hybrid_decapsulate, hybrid_encapsulate, hybrid_kem_keygen, kem_decapsulate, kem_encapsulate,
    kem_keygen,
};

// Re-export key types from sign module.
pub use sign::{
    HybridSignature, HybridSigningKeyPair, PqPublicKey, PqSignature, SigningKeyPair,
    SigningSecretKey, hybrid_sign, hybrid_sign_keygen, hybrid_verify, sign_keygen, sign_message,
    verify_signature,
};

// Re-export key types from tls module.
pub use tls::{
    HybridHandshakeResult, KemAlgorithm, PqCipherSuite, PqTlsConfig, SigAlgorithm,
    negotiate_pq_handshake,
};

/// Errors returned by post-quantum cryptographic operations.
#[derive(Debug, thiserror::Error)]
pub enum PqCryptoError {
    /// Key generation failed.
    #[error("key generation failed: {0}")]
    KeyGeneration(String),

    /// Encapsulation failed.
    #[error("encapsulation failed: {0}")]
    Encapsulation(String),

    /// Decapsulation failed.
    #[error("decapsulation failed: {0}")]
    Decapsulation(String),

    /// Signing failed.
    #[error("signing failed: {0}")]
    Signing(String),

    /// Signature verification failed.
    #[error("verification failed: {0}")]
    Verification(String),

    /// TLS handshake negotiation failed.
    #[error("negotiation failed: {0}")]
    Negotiation(String),

    /// A key or parameter had an invalid size.
    #[error("invalid key size: expected {expected}, got {actual}")]
    InvalidKeySize {
        /// The expected size in bytes.
        expected: usize,
        /// The actual size in bytes.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let e = PqCryptoError::KeyGeneration("rng failure".into());
        assert_eq!(e.to_string(), "key generation failed: rng failure");

        let e = PqCryptoError::InvalidKeySize {
            expected: 1184,
            actual: 100,
        };
        assert_eq!(e.to_string(), "invalid key size: expected 1184, got 100");

        let e = PqCryptoError::Negotiation("no common suite".into());
        assert_eq!(e.to_string(), "negotiation failed: no common suite");
    }

    #[test]
    fn full_kem_workflow() {
        let kp = kem_keygen();
        let (ct, _ss) = kem_encapsulate(&kp.public_key).unwrap();
        let _decap_ss = kem_decapsulate(&kp.secret_key, &ct).unwrap();
    }

    #[test]
    fn full_sign_workflow() {
        let kp = sign_keygen();
        let msg = b"invisible infrastructure";
        let sig = sign_message(&kp.secret_key, msg).unwrap();
        let valid = verify_signature(&kp.public_key, msg, &sig).unwrap();
        assert!(valid);
    }

    #[test]
    fn full_hybrid_workflow() {
        // Hybrid KEM
        let hkem = hybrid_kem_keygen();
        let (ct, _ss) =
            hybrid_encapsulate(&hkem.classical_pk, &hkem.pq_keypair.public_key).unwrap();
        let _decap =
            hybrid_decapsulate(&hkem.classical_sk, &hkem.pq_keypair.secret_key, &ct).unwrap();

        // Hybrid signing
        let hsign = hybrid_sign_keygen();
        let msg = b"cross-algorithm verification";
        let sig = hybrid_sign(&hsign, msg).unwrap();
        let valid = hybrid_verify(&hsign, msg, &sig).unwrap();
        assert!(valid);
    }
}
