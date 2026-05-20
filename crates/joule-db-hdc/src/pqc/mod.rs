//! Post-Quantum Cryptography (PQC) Module
//!
//! Production-quality implementations of NIST-standardized post-quantum algorithms:
//!
//! - **ML-KEM (FIPS 203)** - Module Lattice Key Encapsulation Mechanism (formerly Kyber)
//! - **ML-DSA (FIPS 204)** - Module Lattice Digital Signature Algorithm (formerly Dilithium)
//! - **SLH-DSA (FIPS 205)** - Stateless Hash-based Digital Signature Algorithm (formerly SPHINCS+)
//! - **HQC** - Hamming Quasi-Cyclic backup KEM (selected by NIST March 2025, non-lattice-based)
//!
//! ## Security Features
//!
//! - Constant-time implementations to resist timing attacks
//! - Proper parameter sets matching NIST security levels
//! - Zeroization of sensitive data on drop
//! - Validated against test vectors
//!
//! ## NIST Security Levels
//!
//! | Level | Classical | Quantum | Algorithm Variants |
//! |-------|-----------|---------|-------------------|
//! | 1     | AES-128   | NIST-1  | ML-KEM-512, SLH-DSA-128f, HQC-128 |
//! | 3     | AES-192   | NIST-3  | ML-KEM-768, ML-DSA-65, HQC-192 |
//! | 5     | AES-256   | NIST-5  | ML-KEM-1024, ML-DSA-87, HQC-256 |
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::pqc::{MlKem768, MlDsa65};
//!
//! // Key encapsulation
//! let (ek, dk) = MlKem768::keygen();
//! let (ct, ss_sender) = MlKem768::encapsulate(&ek);
//! let ss_receiver = MlKem768::decapsulate(&dk, &ct);
//! assert_eq!(ss_sender, ss_receiver);
//!
//! // Digital signatures
//! let (vk, sk) = MlDsa65::keygen();
//! let sig = MlDsa65::sign(&sk, message);
//! assert!(MlDsa65::verify(&vk, message, &sig));
//! ```

pub mod common;
pub mod hqc;
pub mod hybrid;
pub mod keystore;
pub mod ml_dsa;
pub mod ml_kem;
pub mod slh_dsa;

// Re-exports
pub use common::{ConstantTime, SecureZeroingVec, Sha3_256, Sha3_512, Shake128, Shake256};

pub use ml_kem::{
    ML_KEM_512_PARAMS, ML_KEM_768_PARAMS, ML_KEM_1024_PARAMS, MlKem512, MlKem768, MlKem1024,
    MlKemCiphertext, MlKemDecapsulationKey, MlKemEncapsulationKey, MlKemParams, MlKemSharedSecret,
};

pub use ml_dsa::{
    ML_DSA_44_PARAMS, ML_DSA_65_PARAMS, ML_DSA_87_PARAMS, MlDsa44, MlDsa65, MlDsa87, MlDsaParams,
    MlDsaSignature, MlDsaSigningKey, MlDsaVerificationKey,
};

pub use slh_dsa::{
    SlhDsa128f, SlhDsa128s, SlhDsa192f, SlhDsa192s, SlhDsa256f, SlhDsa256s, SlhDsaParams,
    SlhDsaPublicKey, SlhDsaSecretKey, SlhDsaSignature as SlhSignature,
};

pub use hybrid::{HybridCiphertext, HybridEncryption, HybridKem};

pub use hqc::{
    HqcCiphertext, HqcKeyPair, HqcParams, HybridHqcCiphertext, HybridHqcKem, HybridHqcKeyPair,
    HybridHqcPublicKey, HybridHqcSecretKey,
};

pub use keystore::{KeyMetadata, PqcKeyStore, StoredKey};

/// PQC Error types
#[derive(Debug, Clone, PartialEq)]
pub enum PqcError {
    /// Invalid key format
    InvalidKey,
    /// Invalid ciphertext
    InvalidCiphertext,
    /// Invalid signature
    InvalidSignature,
    /// Decapsulation failure
    DecapsulationFailed,
    /// Verification failure
    VerificationFailed,
    /// Key generation failure
    KeygenFailed,
    /// Serialization error
    SerializationError,
    /// Random number generation failure
    RngFailure,
    /// Key exhausted (for stateful schemes)
    KeyExhausted,
    /// Invalid algorithm parameter
    InvalidParameter(String),
}

impl std::fmt::Display for PqcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PqcError::InvalidKey => write!(f, "Invalid key format"),
            PqcError::InvalidCiphertext => write!(f, "Invalid ciphertext"),
            PqcError::InvalidSignature => write!(f, "Invalid signature"),
            PqcError::DecapsulationFailed => write!(f, "Decapsulation failed"),
            PqcError::VerificationFailed => write!(f, "Signature verification failed"),
            PqcError::KeygenFailed => write!(f, "Key generation failed"),
            PqcError::SerializationError => write!(f, "Serialization error"),
            PqcError::RngFailure => write!(f, "Random number generation failed"),
            PqcError::KeyExhausted => write!(f, "Signing key exhausted"),
            PqcError::InvalidParameter(ref msg) => write!(f, "Invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for PqcError {}

/// Result type for PQC operations
pub type PqcResult<T> = Result<T, PqcError>;
