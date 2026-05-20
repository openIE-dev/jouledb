//! WebAuthn assertion verification.
//!
//! Verifies authenticator signatures using stored COSE public keys.
//! Supports:
//! - EdDSA (Ed25519) via ed25519-dalek
//! - ES256 (ECDSA P-256) via p256
//! - RS256 (RSA PKCS#1v1.5 SHA-256) via rsa
//!
//! COSE key types: https://www.iana.org/assignments/cose/cose.xhtml
//! - kty=1 (OKP) + crv=6 (Ed25519) -> EdDSA
//! - kty=2 (EC2) + crv=1 (P-256) -> ES256
//! - kty=3 (RSA) -> RS256

use sha2::{Digest, Sha256};
use thiserror::Error;

/// COSE algorithm identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoseAlgorithm {
    /// EdDSA (Ed25519) -- COSE alg = -8
    EdDsa,
    /// ES256 (ECDSA P-256) -- COSE alg = -7
    Es256,
    /// RS256 (RSA SHA-256) -- COSE alg = -257
    Rs256,
}

impl CoseAlgorithm {
    /// Parse from COSE algorithm number.
    pub fn from_cose_alg(alg: i64) -> Option<Self> {
        match alg {
            -8 => Some(Self::EdDsa),
            -7 => Some(Self::Es256),
            -257 => Some(Self::Rs256),
            _ => None,
        }
    }

    /// Convert to COSE algorithm number.
    pub fn to_cose_alg(self) -> i64 {
        match self {
            Self::EdDsa => -8,
            Self::Es256 => -7,
            Self::Rs256 => -257,
        }
    }
}

/// WebAuthn verification errors.
#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("unsupported COSE algorithm: {0}")]
    UnsupportedAlgorithm(i64),
    #[error("invalid public key: {0}")]
    InvalidPublicKey(String),
    #[error("invalid signature: {0}")]
    InvalidSignature(String),
    #[error("signature verification failed")]
    SignatureMismatch,
    #[error("invalid authenticator data")]
    InvalidAuthenticatorData,
    #[error("RP ID hash mismatch")]
    RpIdMismatch,
    #[error("user presence flag not set")]
    UserPresenceNotSet,
}

/// Authenticator data parsed from a WebAuthn assertion.
#[derive(Debug, Clone)]
pub struct AuthenticatorData {
    /// SHA-256 hash of the RP ID.
    pub rp_id_hash: [u8; 32],
    /// Flags byte.
    pub flags: u8,
    /// Sign counter (4 bytes, big-endian).
    pub sign_count: u32,
    /// Raw authenticator data bytes (for signature verification).
    pub raw: Vec<u8>,
}

impl AuthenticatorData {
    /// Parse authenticator data from raw bytes.
    /// Format: rpIdHash (32) || flags (1) || signCount (4) || [extensions...]
    pub fn parse(data: &[u8]) -> Result<Self, VerifyError> {
        if data.len() < 37 {
            return Err(VerifyError::InvalidAuthenticatorData);
        }

        let mut rp_id_hash = [0u8; 32];
        rp_id_hash.copy_from_slice(&data[..32]);

        let flags = data[32];
        let sign_count = u32::from_be_bytes([data[33], data[34], data[35], data[36]]);

        Ok(Self {
            rp_id_hash,
            flags,
            sign_count,
            raw: data.to_vec(),
        })
    }

    /// Check the User Present (UP) flag (bit 0).
    pub fn user_present(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// Check the User Verified (UV) flag (bit 2).
    pub fn user_verified(&self) -> bool {
        self.flags & 0x04 != 0
    }

    /// Verify RP ID matches by comparing SHA-256 hashes.
    pub fn verify_rp_id(&self, rp_id: &str) -> Result<(), VerifyError> {
        let expected = Sha256::digest(rp_id.as_bytes());
        if self.rp_id_hash != expected[..] {
            return Err(VerifyError::RpIdMismatch);
        }
        Ok(())
    }
}

/// Verify an EdDSA (Ed25519) assertion signature.
///
/// The signed data is: authenticator_data || SHA-256(client_data_json)
pub fn verify_eddsa(
    public_key_bytes: &[u8],
    authenticator_data: &[u8],
    client_data_hash: &[u8; 32],
    signature: &[u8],
) -> Result<(), VerifyError> {
    use ed25519_dalek::{Signature, VerifyingKey};

    let verifying_key = VerifyingKey::try_from(public_key_bytes)
        .map_err(|e| VerifyError::InvalidPublicKey(e.to_string()))?;

    let sig =
        Signature::try_from(signature).map_err(|e| VerifyError::InvalidSignature(e.to_string()))?;

    // Construct signed data: authenticator_data || client_data_hash
    let mut signed_data = Vec::with_capacity(authenticator_data.len() + 32);
    signed_data.extend_from_slice(authenticator_data);
    signed_data.extend_from_slice(client_data_hash);

    use ed25519_dalek::Verifier;
    verifying_key
        .verify(&signed_data, &sig)
        .map_err(|_| VerifyError::SignatureMismatch)
}

/// Verify an ES256 (ECDSA P-256) assertion signature.
///
/// The signed data is: authenticator_data || SHA-256(client_data_json)
///
/// The public key should be a 65-byte uncompressed P-256 point (0x04 || x || y)
/// or a 33-byte compressed point. The signature is DER-encoded.
pub fn verify_es256(
    public_key_bytes: &[u8],
    authenticator_data: &[u8],
    client_data_hash: &[u8; 32],
    signature: &[u8],
) -> Result<(), VerifyError> {
    use p256::EncodedPoint;
    use p256::ecdsa::{DerSignature, VerifyingKey, signature::Verifier};

    let point = EncodedPoint::from_bytes(public_key_bytes)
        .map_err(|e| VerifyError::InvalidPublicKey(format!("P-256 point: {e}")))?;

    let verifying_key = VerifyingKey::from_encoded_point(&point)
        .map_err(|e| VerifyError::InvalidPublicKey(format!("P-256 key: {e}")))?;

    let sig = DerSignature::try_from(signature)
        .map_err(|e| VerifyError::InvalidSignature(format!("DER: {e}")))?;

    // Construct signed data: authenticator_data || client_data_hash
    let mut signed_data = Vec::with_capacity(authenticator_data.len() + 32);
    signed_data.extend_from_slice(authenticator_data);
    signed_data.extend_from_slice(client_data_hash);

    verifying_key
        .verify(&signed_data, &sig)
        .map_err(|_| VerifyError::SignatureMismatch)
}

/// Verify an RS256 (RSASSA-PKCS1-v1_5 SHA-256) assertion signature.
///
/// The signed data is: authenticator_data || SHA-256(client_data_json)
///
/// The public key should be DER-encoded (PKCS#1 or SubjectPublicKeyInfo).
pub fn verify_rs256(
    public_key_der: &[u8],
    authenticator_data: &[u8],
    client_data_hash: &[u8; 32],
    signature: &[u8],
) -> Result<(), VerifyError> {
    use rsa::RsaPublicKey;
    use rsa::pkcs1::DecodeRsaPublicKey;
    use rsa::pkcs1v15::Pkcs1v15Sign;

    // Try PKCS#1 DER first, then SPKI DER
    let rsa_key = RsaPublicKey::from_pkcs1_der(public_key_der)
        .or_else(|_| {
            use rsa::pkcs8::DecodePublicKey;
            RsaPublicKey::from_public_key_der(public_key_der)
        })
        .map_err(|e| VerifyError::InvalidPublicKey(format!("RSA key: {e}")))?;

    // Construct signed data: authenticator_data || client_data_hash
    let mut signed_data = Vec::with_capacity(authenticator_data.len() + 32);
    signed_data.extend_from_slice(authenticator_data);
    signed_data.extend_from_slice(client_data_hash);

    // Hash the signed data with SHA-256 and verify PKCS#1v1.5
    let digest = Sha256::digest(&signed_data);
    let scheme = Pkcs1v15Sign::new::<Sha256>();

    rsa_key
        .verify(scheme, &digest, signature)
        .map_err(|_| VerifyError::SignatureMismatch)
}

/// High-level assertion verification.
///
/// 1. Parse authenticator data
/// 2. Verify RP ID hash
/// 3. Check user presence flag
/// 4. Verify signature over (auth_data || hash(client_data_json))
/// 5. Return the sign counter from authenticator data
pub fn verify_assertion(
    cose_alg: i64,
    public_key_bytes: &[u8],
    authenticator_data_bytes: &[u8],
    client_data_json: &[u8],
    signature: &[u8],
    expected_rp_id: &str,
) -> Result<u32, VerifyError> {
    let algorithm = CoseAlgorithm::from_cose_alg(cose_alg)
        .ok_or(VerifyError::UnsupportedAlgorithm(cose_alg))?;

    // Parse authenticator data
    let auth_data = AuthenticatorData::parse(authenticator_data_bytes)?;

    // Verify RP ID
    auth_data.verify_rp_id(expected_rp_id)?;

    // Check user presence
    if !auth_data.user_present() {
        return Err(VerifyError::UserPresenceNotSet);
    }

    // Hash client_data_json
    let client_data_hash: [u8; 32] = Sha256::digest(client_data_json).into();

    // Verify signature based on algorithm
    match algorithm {
        CoseAlgorithm::EdDsa => {
            verify_eddsa(
                public_key_bytes,
                authenticator_data_bytes,
                &client_data_hash,
                signature,
            )?;
        }
        CoseAlgorithm::Es256 => {
            verify_es256(
                public_key_bytes,
                authenticator_data_bytes,
                &client_data_hash,
                signature,
            )?;
        }
        CoseAlgorithm::Rs256 => {
            verify_rs256(
                public_key_bytes,
                authenticator_data_bytes,
                &client_data_hash,
                signature,
            )?;
        }
    }

    Ok(auth_data.sign_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cose_algorithm_roundtrip() {
        assert_eq!(CoseAlgorithm::from_cose_alg(-8), Some(CoseAlgorithm::EdDsa));
        assert_eq!(CoseAlgorithm::from_cose_alg(-7), Some(CoseAlgorithm::Es256));
        assert_eq!(
            CoseAlgorithm::from_cose_alg(-257),
            Some(CoseAlgorithm::Rs256)
        );
        assert_eq!(CoseAlgorithm::from_cose_alg(0), None);
        assert_eq!(CoseAlgorithm::EdDsa.to_cose_alg(), -8);
    }

    #[test]
    fn parse_authenticator_data_valid() {
        // 32 bytes RP ID hash + 1 byte flags (UP=1) + 4 bytes counter
        let mut data = vec![0u8; 37];
        // Set flags: UP bit set
        data[32] = 0x01;
        // Set counter to 42
        data[33..37].copy_from_slice(&42u32.to_be_bytes());

        let auth_data = AuthenticatorData::parse(&data).unwrap();
        assert!(auth_data.user_present());
        assert!(!auth_data.user_verified());
        assert_eq!(auth_data.sign_count, 42);
    }

    #[test]
    fn parse_authenticator_data_too_short() {
        let data = vec![0u8; 36]; // Too short
        assert!(AuthenticatorData::parse(&data).is_err());
    }

    #[test]
    fn parse_authenticator_data_uv_flag() {
        let mut data = vec![0u8; 37];
        data[32] = 0x05; // UP + UV
        let auth_data = AuthenticatorData::parse(&data).unwrap();
        assert!(auth_data.user_present());
        assert!(auth_data.user_verified());
    }

    #[test]
    fn verify_rp_id_correct() {
        let rp_id = "invisible.dev";
        let hash = Sha256::digest(rp_id.as_bytes());
        let mut data = vec![0u8; 37];
        data[..32].copy_from_slice(&hash);
        data[32] = 0x01;

        let auth_data = AuthenticatorData::parse(&data).unwrap();
        assert!(auth_data.verify_rp_id("invisible.dev").is_ok());
    }

    #[test]
    fn verify_rp_id_mismatch() {
        let hash = Sha256::digest(b"invisible.dev");
        let mut data = vec![0u8; 37];
        data[..32].copy_from_slice(&hash);
        data[32] = 0x01;

        let auth_data = AuthenticatorData::parse(&data).unwrap();
        assert!(auth_data.verify_rp_id("evil.com").is_err());
    }

    #[test]
    fn verify_eddsa_valid_signature() {
        use ed25519_dalek::{Signer, SigningKey};

        // Generate a keypair
        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let verifying_key = signing_key.verifying_key();

        // Construct authenticator data (37 bytes: rp_id_hash + flags + counter)
        let mut auth_data = vec![0u8; 37];
        auth_data[32] = 0x01; // UP flag

        // Client data hash
        let client_data_hash: [u8; 32] = Sha256::digest(b"test client data").into();

        // Construct signed data: auth_data || client_data_hash
        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);

        // Sign
        let signature = signing_key.sign(&signed_data);

        // Verify
        let result = verify_eddsa(
            verifying_key.as_bytes(),
            &auth_data,
            &client_data_hash,
            &signature.to_bytes(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn verify_eddsa_wrong_key() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let wrong_key = SigningKey::from_bytes(&[99u8; 32]);

        let auth_data = vec![0u8; 37];
        let client_data_hash: [u8; 32] = Sha256::digest(b"test").into();

        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let signature = signing_key.sign(&signed_data);

        let result = verify_eddsa(
            wrong_key.verifying_key().as_bytes(),
            &auth_data,
            &client_data_hash,
            &signature.to_bytes(),
        );
        assert!(matches!(result, Err(VerifyError::SignatureMismatch)));
    }

    #[test]
    fn verify_assertion_full_flow() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let rp_id = "invisible.dev";
        let rp_id_hash = Sha256::digest(rp_id.as_bytes());

        // Build authenticator data
        let mut auth_data = vec![0u8; 37];
        auth_data[..32].copy_from_slice(&rp_id_hash);
        auth_data[32] = 0x05; // UP + UV
        auth_data[33..37].copy_from_slice(&7u32.to_be_bytes()); // counter = 7

        let client_data_json = b"{\"type\":\"webauthn.get\",\"challenge\":\"abc\"}";
        let client_data_hash: [u8; 32] = Sha256::digest(client_data_json).into();

        // Sign: auth_data || client_data_hash
        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let signature = signing_key.sign(&signed_data);

        let sign_count = verify_assertion(
            -8, // EdDSA
            verifying_key.as_bytes(),
            &auth_data,
            client_data_json,
            &signature.to_bytes(),
            rp_id,
        )
        .unwrap();

        assert_eq!(sign_count, 7);
    }

    #[test]
    fn verify_assertion_user_presence_not_set() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let rp_id = "invisible.dev";
        let rp_id_hash = Sha256::digest(rp_id.as_bytes());

        let mut auth_data = vec![0u8; 37];
        auth_data[..32].copy_from_slice(&rp_id_hash);
        auth_data[32] = 0x00; // NO flags set

        let client_data_json = b"{}";
        let client_data_hash: [u8; 32] = Sha256::digest(client_data_json).into();

        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let signature = signing_key.sign(&signed_data);

        let result = verify_assertion(
            -8,
            verifying_key.as_bytes(),
            &auth_data,
            client_data_json,
            &signature.to_bytes(),
            rp_id,
        );
        assert!(matches!(result, Err(VerifyError::UserPresenceNotSet)));
    }

    #[test]
    fn verify_assertion_unknown_algorithm() {
        // Build valid authenticator data so we reach the algorithm dispatch
        let rp_id = "example.com";
        let rp_id_hash = Sha256::digest(rp_id.as_bytes());
        let mut auth_data = vec![0u8; 37];
        auth_data[..32].copy_from_slice(&rp_id_hash);
        auth_data[32] = 0x01; // UP flag set

        let result = verify_assertion(
            -99, // Unknown algorithm
            &[],
            &auth_data,
            b"{}",
            &[],
            rp_id,
        );
        assert!(matches!(
            result,
            Err(VerifyError::UnsupportedAlgorithm(-99))
        ));
    }

    #[test]
    fn verify_es256_valid_signature() {
        use p256::ecdsa::{SigningKey, signature::Signer};

        // Generate a P-256 keypair
        let signing_key = SigningKey::random(&mut rand::rng());
        let verifying_key = signing_key.verifying_key();

        // Uncompressed P-256 public key (65 bytes: 0x04 || x || y)
        let public_key_bytes = verifying_key.to_encoded_point(false);

        // Construct authenticator data (37 bytes)
        let mut auth_data = vec![0u8; 37];
        auth_data[32] = 0x01; // UP flag

        // Client data hash
        let client_data_hash: [u8; 32] = Sha256::digest(b"es256 test data").into();

        // Construct signed data and sign
        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let signature: p256::ecdsa::DerSignature = signing_key.sign(&signed_data);

        let result = verify_es256(
            public_key_bytes.as_bytes(),
            &auth_data,
            &client_data_hash,
            signature.as_bytes(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn verify_es256_wrong_key() {
        use p256::ecdsa::{SigningKey, signature::Signer};

        let signing_key = SigningKey::random(&mut rand::rng());
        let wrong_key = SigningKey::random(&mut rand::rng());

        let auth_data = vec![0u8; 37];
        let client_data_hash: [u8; 32] = Sha256::digest(b"test").into();

        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let signature: p256::ecdsa::DerSignature = signing_key.sign(&signed_data);

        let result = verify_es256(
            wrong_key.verifying_key().to_encoded_point(false).as_bytes(),
            &auth_data,
            &client_data_hash,
            signature.as_bytes(),
        );
        assert!(matches!(result, Err(VerifyError::SignatureMismatch)));
    }

    #[test]
    fn verify_rs256_valid_signature() {
        use rsa::RsaPrivateKey;
        use rsa::pkcs1::EncodeRsaPublicKey;
        use rsa::pkcs1v15::Pkcs1v15Sign;

        // Generate a 2048-bit RSA keypair
        let mut rng = rand::rng();
        let private_key = RsaPrivateKey::new(&mut rng, 2048).unwrap();
        let public_key = private_key.to_public_key();
        let public_key_der = public_key.to_pkcs1_der().unwrap();

        // Construct authenticator data (37 bytes)
        let mut auth_data = vec![0u8; 37];
        auth_data[32] = 0x01; // UP flag

        let client_data_hash: [u8; 32] = Sha256::digest(b"rs256 test data").into();

        // Construct signed data and sign using low-level API
        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let digest = Sha256::digest(&signed_data);
        let scheme = Pkcs1v15Sign::new::<Sha256>();
        let signature = private_key.sign(scheme, &digest).unwrap();

        let result = verify_rs256(
            public_key_der.as_bytes(),
            &auth_data,
            &client_data_hash,
            &signature,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn verify_assertion_es256_full_flow() {
        use p256::ecdsa::{SigningKey, signature::Signer};

        let signing_key = SigningKey::random(&mut rand::rng());
        let verifying_key = signing_key.verifying_key();

        let rp_id = "invisible.dev";
        let rp_id_hash = Sha256::digest(rp_id.as_bytes());

        let mut auth_data = vec![0u8; 37];
        auth_data[..32].copy_from_slice(&rp_id_hash);
        auth_data[32] = 0x05; // UP + UV
        auth_data[33..37].copy_from_slice(&12u32.to_be_bytes());

        let client_data_json = b"{\"type\":\"webauthn.get\",\"challenge\":\"xyz\"}";
        let client_data_hash: [u8; 32] = Sha256::digest(client_data_json).into();

        let mut signed_data = auth_data.clone();
        signed_data.extend_from_slice(&client_data_hash);
        let signature: p256::ecdsa::DerSignature = signing_key.sign(&signed_data);

        let sign_count = verify_assertion(
            -7, // ES256
            verifying_key.to_encoded_point(false).as_bytes(),
            &auth_data,
            client_data_json,
            signature.as_bytes(),
            rp_id,
        )
        .unwrap();

        assert_eq!(sign_count, 12);
    }
}
