//! ML-KEM-768 key encapsulation (NIST FIPS 203) — real implementation.
//!
//! Uses the `ml-kem` crate (RustCrypto) for genuine ML-KEM-768 key
//! encapsulation, and `x25519-dalek` for classical X25519 Diffie-Hellman
//! in hybrid mode.

use ml_kem::kem::{Decapsulate, Encapsulate};
use ml_kem::{Encoded, EncodedSizeUser, KemCore, MlKem768, MlKem768Params};
use serde::{Deserialize, Serialize};
use x25519_dalek::{EphemeralSecret, PublicKey as X25519Public, StaticSecret};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::PqCryptoError;

type Ek = ml_kem::kem::EncapsulationKey<MlKem768Params>;
type Dk = ml_kem::kem::DecapsulationKey<MlKem768Params>;

/// ML-KEM-768 public key size in bytes.
pub const MLKEM768_PK_SIZE: usize = 1184;
/// ML-KEM-768 secret key size in bytes.
pub const MLKEM768_SK_SIZE: usize = 2400;
/// ML-KEM-768 ciphertext size in bytes.
pub const MLKEM768_CT_SIZE: usize = 1088;
/// Shared secret size in bytes (256-bit).
pub const SHARED_SECRET_SIZE: usize = 32;
/// Classical X25519 key size in bytes.
pub const X25519_KEY_SIZE: usize = 32;

/// A zeroizable wrapper around secret key bytes.
///
/// Automatically erases the secret key material from memory on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SecretKeyBytes(Vec<u8>);

impl SecretKeyBytes {
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

impl std::fmt::Debug for SecretKeyBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SecretKeyBytes")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// An ML-KEM-768 public key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KemPublicKey(pub Vec<u8>);

impl KemPublicKey {
    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// An ML-KEM-768 ciphertext.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KemCiphertext(pub Vec<u8>);

impl KemCiphertext {
    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// A shared secret derived from key encapsulation.
///
/// Automatically zeroized on drop.
#[derive(Clone, Zeroize, ZeroizeOnDrop, PartialEq, Eq)]
pub struct KemSharedSecret(Vec<u8>);

impl KemSharedSecret {
    /// Access the raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for KemSharedSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("KemSharedSecret")
            .field(&"[REDACTED]")
            .finish()
    }
}

/// An ML-KEM-768 key pair (public key + secret key).
#[derive(Debug, Clone)]
pub struct KemKeyPair {
    /// The public encapsulation key (1184 bytes for ML-KEM-768).
    pub public_key: KemPublicKey,
    /// The secret decapsulation key (2400 bytes for ML-KEM-768).
    pub secret_key: SecretKeyBytes,
}

/// Generate an ML-KEM-768 key pair using real lattice-based cryptography.
pub fn kem_keygen() -> KemKeyPair {
    let mut rng = rand_core_06::OsRng;
    let (dk, ek) = MlKem768::generate(&mut rng);

    let ek_bytes = ek.as_bytes();
    let dk_bytes = dk.as_bytes();

    KemKeyPair {
        public_key: KemPublicKey(ek_bytes.to_vec()),
        secret_key: SecretKeyBytes::new(dk_bytes.to_vec()),
    }
}

/// Encapsulate a shared secret against an ML-KEM-768 public key.
///
/// Returns the ciphertext and the shared secret. The ciphertext must be sent
/// to the holder of the corresponding secret key for decapsulation.
///
/// # Errors
///
/// Returns [`PqCryptoError::InvalidKeySize`] if the public key length is wrong.
/// Returns [`PqCryptoError::Encapsulation`] if the key bytes are invalid.
pub fn kem_encapsulate(
    public_key: &KemPublicKey,
) -> Result<(KemCiphertext, KemSharedSecret), PqCryptoError> {
    if public_key.as_bytes().len() != MLKEM768_PK_SIZE {
        return Err(PqCryptoError::InvalidKeySize {
            expected: MLKEM768_PK_SIZE,
            actual: public_key.as_bytes().len(),
        });
    }

    let mut rng = rand_core_06::OsRng;

    // Deserialize the encapsulation key from bytes
    let ek_encoded: &Encoded<Ek> = public_key
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Encapsulation("invalid ML-KEM-768 public key".into()))?;
    let ek = Ek::from_bytes(ek_encoded);

    // Real ML-KEM-768 encapsulation
    let (ct, ss) = ek.encapsulate(&mut rng).map_err(|e| {
        PqCryptoError::Encapsulation(format!("ML-KEM-768 encapsulation failed: {e:?}"))
    })?;

    Ok((KemCiphertext(ct.to_vec()), KemSharedSecret(ss.to_vec())))
}

/// Decapsulate a shared secret from an ML-KEM-768 ciphertext.
///
/// # Errors
///
/// Returns [`PqCryptoError::InvalidKeySize`] if the secret key length is wrong.
/// Returns [`PqCryptoError::Decapsulation`] if the ciphertext is malformed.
pub fn kem_decapsulate(
    secret_key: &SecretKeyBytes,
    ciphertext: &KemCiphertext,
) -> Result<KemSharedSecret, PqCryptoError> {
    if secret_key.len() != MLKEM768_SK_SIZE {
        return Err(PqCryptoError::InvalidKeySize {
            expected: MLKEM768_SK_SIZE,
            actual: secret_key.len(),
        });
    }

    if ciphertext.as_bytes().len() != MLKEM768_CT_SIZE {
        return Err(PqCryptoError::Decapsulation(
            "ciphertext length mismatch".into(),
        ));
    }

    // Deserialize the decapsulation key from bytes
    let dk_encoded: &Encoded<Dk> = secret_key
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Decapsulation("invalid ML-KEM-768 secret key".into()))?;
    let dk = Dk::from_bytes(dk_encoded);

    // Deserialize ciphertext
    let ct_arr: &ml_kem::Ciphertext<MlKem768> = ciphertext
        .as_bytes()
        .try_into()
        .map_err(|_| PqCryptoError::Decapsulation("invalid ML-KEM-768 ciphertext".into()))?;

    // Real ML-KEM-768 decapsulation
    let ss = dk.decapsulate(ct_arr).map_err(|e| {
        PqCryptoError::Decapsulation(format!("ML-KEM-768 decapsulation failed: {e:?}"))
    })?;

    Ok(KemSharedSecret(ss.to_vec()))
}

// ---------------------------------------------------------------------------
// Hybrid KEM: X25519 + ML-KEM-768
// ---------------------------------------------------------------------------

/// A hybrid key pair combining classical X25519 keys with ML-KEM-768.
#[derive(Debug, Clone)]
pub struct HybridKemKeyPair {
    /// Classical X25519 public key (32 bytes).
    pub classical_pk: Vec<u8>,
    /// Classical X25519 secret key (32 bytes, zeroizable).
    pub classical_sk: SecretKeyBytes,
    /// Post-quantum ML-KEM-768 key pair.
    pub pq_keypair: KemKeyPair,
}

/// Generate a hybrid X25519 + ML-KEM-768 key pair using real cryptography.
pub fn hybrid_kem_keygen() -> HybridKemKeyPair {
    // Real X25519 key generation
    let classical_secret = StaticSecret::random_from_rng(rand_core_06::OsRng);
    let classical_public = X25519Public::from(&classical_secret);

    // Real ML-KEM-768 key generation
    let pq_keypair = kem_keygen();

    HybridKemKeyPair {
        classical_pk: classical_public.as_bytes().to_vec(),
        classical_sk: SecretKeyBytes::new(classical_secret.to_bytes().to_vec()),
        pq_keypair,
    }
}

/// Encapsulate a hybrid shared secret (X25519 + ML-KEM-768).
///
/// The resulting shared secret is the concatenation of the classical and
/// post-quantum shared secrets, providing security as long as **either**
/// algorithm remains unbroken.
///
/// # Errors
///
/// Returns [`PqCryptoError::Encapsulation`] if the classical public key
/// length is wrong, or delegates to [`kem_encapsulate`] for PQ errors.
pub fn hybrid_encapsulate(
    classical_pk: &[u8],
    pq_pk: &KemPublicKey,
) -> Result<(Vec<u8>, KemSharedSecret), PqCryptoError> {
    if classical_pk.len() != X25519_KEY_SIZE {
        return Err(PqCryptoError::Encapsulation(
            "classical public key must be 32 bytes".into(),
        ));
    }

    // Real X25519 Diffie-Hellman
    let peer_pk_bytes: [u8; 32] = classical_pk.try_into().map_err(|_| {
        PqCryptoError::Encapsulation("classical public key must be 32 bytes".into())
    })?;
    let peer_public = X25519Public::from(peer_pk_bytes);
    let ephemeral_secret = EphemeralSecret::random_from_rng(rand_core_06::OsRng);
    let ephemeral_public = X25519Public::from(&ephemeral_secret);
    let classical_ss = ephemeral_secret.diffie_hellman(&peer_public);

    // Real PQ encapsulation
    let (pq_ct, pq_ss) = kem_encapsulate(pq_pk)?;

    // Combined ciphertext: X25519 ephemeral public (32 bytes) || ML-KEM ciphertext
    let mut combined_ct = ephemeral_public.as_bytes().to_vec();
    combined_ct.extend_from_slice(pq_ct.as_bytes());

    // Combined shared secret: classical_ss || pq_ss
    let mut combined_ss = classical_ss.as_bytes().to_vec();
    combined_ss.extend_from_slice(pq_ss.as_bytes());

    Ok((combined_ct, KemSharedSecret(combined_ss)))
}

/// Decapsulate a hybrid shared secret (X25519 + ML-KEM-768).
///
/// # Errors
///
/// Returns [`PqCryptoError::Decapsulation`] if the ciphertext is too short
/// or the keys are invalid.
pub fn hybrid_decapsulate(
    classical_sk: &SecretKeyBytes,
    pq_sk: &SecretKeyBytes,
    combined_ct: &[u8],
) -> Result<KemSharedSecret, PqCryptoError> {
    if classical_sk.len() != X25519_KEY_SIZE {
        return Err(PqCryptoError::Decapsulation(
            "classical secret key must be 32 bytes".into(),
        ));
    }

    let expected_ct_len = X25519_KEY_SIZE + MLKEM768_CT_SIZE;
    if combined_ct.len() != expected_ct_len {
        return Err(PqCryptoError::Decapsulation(format!(
            "hybrid ciphertext must be {expected_ct_len} bytes, got {}",
            combined_ct.len()
        )));
    }

    let (ephemeral_pk_bytes, pq_ct_bytes) = combined_ct.split_at(X25519_KEY_SIZE);

    // Real X25519 Diffie-Hellman decapsulation
    let sk_bytes: [u8; 32] = classical_sk.as_bytes().try_into().map_err(|_| {
        PqCryptoError::Decapsulation("classical secret key must be 32 bytes".into())
    })?;
    let static_secret = StaticSecret::from(sk_bytes);
    let eph_pk_bytes: [u8; 32] = ephemeral_pk_bytes.try_into().map_err(|_| {
        PqCryptoError::Decapsulation("ephemeral public key must be 32 bytes".into())
    })?;
    let ephemeral_public = X25519Public::from(eph_pk_bytes);
    let classical_ss = static_secret.diffie_hellman(&ephemeral_public);

    // Real PQ decapsulation
    let pq_ct = KemCiphertext(pq_ct_bytes.to_vec());
    let pq_ss = kem_decapsulate(pq_sk, &pq_ct)?;

    // Combined shared secret: classical_ss || pq_ss
    let mut combined_ss = classical_ss.as_bytes().to_vec();
    combined_ss.extend_from_slice(pq_ss.as_bytes());

    Ok(KemSharedSecret(combined_ss))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keygen_produces_correct_sizes() {
        let kp = kem_keygen();
        assert_eq!(kp.public_key.as_bytes().len(), MLKEM768_PK_SIZE);
        assert_eq!(kp.secret_key.len(), MLKEM768_SK_SIZE);
    }

    #[test]
    fn encapsulate_produces_correct_sizes() {
        let kp = kem_keygen();
        let (ct, ss) = kem_encapsulate(&kp.public_key).unwrap();
        assert_eq!(ct.as_bytes().len(), MLKEM768_CT_SIZE);
        assert_eq!(ss.as_bytes().len(), SHARED_SECRET_SIZE);
    }

    #[test]
    fn encapsulate_and_decapsulate_produce_same_shared_secret() {
        let kp = kem_keygen();
        let (ct, encap_ss) = kem_encapsulate(&kp.public_key).unwrap();
        let decap_ss = kem_decapsulate(&kp.secret_key, &ct).unwrap();
        // THE fundamental KEM correctness property:
        assert_eq!(
            encap_ss, decap_ss,
            "encap and decap must produce the same shared secret"
        );
    }

    #[test]
    fn different_keypairs_produce_different_shared_secrets() {
        let kp1 = kem_keygen();
        let kp2 = kem_keygen();
        let (_, ss1) = kem_encapsulate(&kp1.public_key).unwrap();
        let (_, ss2) = kem_encapsulate(&kp2.public_key).unwrap();
        assert_ne!(ss1, ss2);
    }

    #[test]
    fn encapsulate_rejects_wrong_pk_size() {
        let bad_pk = KemPublicKey(vec![0u8; 100]);
        let result = kem_encapsulate(&bad_pk);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PqCryptoError::InvalidKeySize { .. }
        ));
    }

    #[test]
    fn decapsulate_rejects_wrong_sk_size() {
        let bad_sk = SecretKeyBytes::new(vec![0u8; 100]);
        let ct = KemCiphertext(vec![0u8; MLKEM768_CT_SIZE]);
        let result = kem_decapsulate(&bad_sk, &ct);
        assert!(result.is_err());
    }

    #[test]
    fn decapsulate_rejects_wrong_ct_size() {
        let kp = kem_keygen();
        let bad_ct = KemCiphertext(vec![0u8; 42]);
        let result = kem_decapsulate(&kp.secret_key, &bad_ct);
        assert!(result.is_err());
    }

    #[test]
    fn shared_secret_debug_is_redacted() {
        let kp = kem_keygen();
        let (_ct, ss) = kem_encapsulate(&kp.public_key).unwrap();
        let debug = format!("{ss:?}");
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn secret_key_debug_is_redacted() {
        let kp = kem_keygen();
        let debug = format!("{:?}", kp.secret_key);
        assert!(debug.contains("[REDACTED]"));
    }

    #[test]
    fn hybrid_keygen_produces_correct_sizes() {
        let hkp = hybrid_kem_keygen();
        assert_eq!(hkp.classical_pk.len(), X25519_KEY_SIZE);
        assert_eq!(hkp.classical_sk.len(), X25519_KEY_SIZE);
        assert_eq!(hkp.pq_keypair.public_key.as_bytes().len(), MLKEM768_PK_SIZE);
        assert_eq!(hkp.pq_keypair.secret_key.len(), MLKEM768_SK_SIZE);
    }

    #[test]
    fn hybrid_encapsulate_and_decapsulate_produce_same_shared_secret() {
        let hkp = hybrid_kem_keygen();
        let (combined_ct, encap_ss) =
            hybrid_encapsulate(&hkp.classical_pk, &hkp.pq_keypair.public_key).unwrap();

        let expected_ct_len = X25519_KEY_SIZE + MLKEM768_CT_SIZE;
        assert_eq!(combined_ct.len(), expected_ct_len);

        let decap_ss =
            hybrid_decapsulate(&hkp.classical_sk, &hkp.pq_keypair.secret_key, &combined_ct)
                .unwrap();

        // Combined shared secret should be 64 bytes (32 classical + 32 PQ).
        assert_eq!(encap_ss.as_bytes().len(), 64);
        assert_eq!(decap_ss.as_bytes().len(), 64);
        // THE fundamental hybrid KEM correctness property:
        assert_eq!(
            encap_ss, decap_ss,
            "hybrid encap and decap must produce the same shared secret"
        );
    }

    #[test]
    fn hybrid_encapsulate_rejects_bad_classical_pk() {
        let kp = kem_keygen();
        let result = hybrid_encapsulate(&[0u8; 16], &kp.public_key);
        assert!(result.is_err());
    }
}
