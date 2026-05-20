//! W3C Verifiable Credentials 2.0.
//!
//! Provides credential issuance and verification for the Invisible
//! Infrastructure mesh, including node capability, energy compliance,
//! and mesh membership credentials.
//!
//! Proofs use real Ed25519 signatures via `ed25519-dalek`. The issuer
//! signs the canonical JSON of the unsigned credential, and the verifier
//! checks the signature against the issuer's public key.

use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{Signer as _, Verifier as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::IdentityError;

/// A W3C Verifiable Credential 2.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiableCredential {
    /// JSON-LD context URIs.
    pub context: Vec<String>,
    /// Unique identifier for this credential.
    pub id: String,
    /// Credential types (e.g., `["VerifiableCredential", "NodeCapability"]`).
    pub credential_type: Vec<String>,
    /// DID of the issuer.
    pub issuer: String,
    /// When the credential was issued.
    pub issuance_date: DateTime<Utc>,
    /// When the credential expires (if any).
    pub expiration_date: Option<DateTime<Utc>>,
    /// The credential subject (JSON value).
    pub credential_subject: serde_json::Value,
    /// Cryptographic proof binding the credential to the issuer.
    pub proof: CredentialProof,
}

/// Cryptographic proof within a verifiable credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialProof {
    /// Proof type (e.g., "Ed25519Signature2020").
    pub proof_type: String,
    /// When the proof was created.
    pub created: DateTime<Utc>,
    /// The verification method used to create the proof.
    pub verification_method: String,
    /// The proof value (Ed25519 signature bytes — 64 bytes).
    pub proof_value: Vec<u8>,
}

/// Types of credentials issued in the mesh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialType {
    /// A node's advertised capabilities.
    NodeCapability,
    /// Energy compliance certification.
    EnergyCompliance,
    /// Mesh membership credential.
    MeshMembership,
    /// Custom credential type.
    Custom(String),
}

impl CredentialType {
    /// Return the credential type string for use in VC `type` arrays.
    pub fn as_type_str(&self) -> &str {
        match self {
            Self::NodeCapability => "NodeCapability",
            Self::EnergyCompliance => "EnergyCompliance",
            Self::MeshMembership => "MeshMembership",
            Self::Custom(s) => s,
        }
    }
}

/// Issues verifiable credentials with real Ed25519 signatures.
#[derive(Debug, Clone)]
pub struct VcIssuer {
    /// The DID of the issuer.
    pub issuer_did: String,
    /// The Ed25519 signing key (32-byte seed).
    signing_key: ed25519_dalek::SigningKey,
}

impl VcIssuer {
    /// Create a new VC issuer from a DID and a 32-byte Ed25519 signing key seed.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError::CredentialIssuance`] if `signing_key_bytes` is
    /// not exactly 32 bytes.
    pub fn new(issuer_did: String, signing_key_bytes: Vec<u8>) -> Result<Self, IdentityError> {
        let key_bytes: [u8; 32] = signing_key_bytes.try_into().map_err(|v: Vec<u8>| {
            IdentityError::CredentialIssuance(format!(
                "signing key must be 32 bytes, got {}",
                v.len()
            ))
        })?;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&key_bytes);
        Ok(Self {
            issuer_did,
            signing_key,
        })
    }

    /// Create a new VC issuer from an existing `ed25519_dalek::SigningKey`.
    pub fn from_signing_key(issuer_did: String, signing_key: ed25519_dalek::SigningKey) -> Self {
        Self {
            issuer_did,
            signing_key,
        }
    }

    /// Return the verifying (public) key for this issuer.
    pub fn verifying_key(&self) -> ed25519_dalek::VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Issue a verifiable credential.
    ///
    /// The proof is a real Ed25519 signature over the canonical JSON of
    /// the unsigned credential.
    pub fn issue(
        &self,
        credential_type: CredentialType,
        subject: serde_json::Value,
        subject_did: &str,
        expiry_duration: Option<Duration>,
    ) -> Result<VerifiableCredential, IdentityError> {
        let now = Utc::now();
        let expiration_date = expiry_duration.map(|d| now + d);

        let vc_id = format!("urn:uuid:{}", uuid_from_hash(subject_did, now));

        let credential_subject = serde_json::json!({
            "id": subject_did,
            "data": subject,
        });

        // Build the unsigned credential as canonical JSON for signing.
        let unsigned = serde_json::json!({
            "context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiableCredential", credential_type.as_type_str()],
            "issuer": self.issuer_did,
            "issuanceDate": now.to_rfc3339(),
            "credentialSubject": credential_subject,
        });

        let unsigned_bytes = serde_json::to_vec(&unsigned)
            .map_err(|e| IdentityError::CredentialIssuance(e.to_string()))?;

        // Real Ed25519 signature.
        let signature = self.signing_key.sign(&unsigned_bytes);

        let proof = CredentialProof {
            proof_type: "Ed25519Signature2020".into(),
            created: now,
            verification_method: format!("{}#key-1", self.issuer_did),
            proof_value: signature.to_bytes().to_vec(),
        };

        Ok(VerifiableCredential {
            context: vec!["https://www.w3.org/2018/credentials/v1".into()],
            id: vc_id,
            credential_type: vec![
                "VerifiableCredential".into(),
                credential_type.as_type_str().into(),
            ],
            issuer: self.issuer_did.clone(),
            issuance_date: now,
            expiration_date,
            credential_subject,
            proof,
        })
    }
}

/// Verifies verifiable credentials using real Ed25519 signature verification.
#[derive(Debug, Clone)]
pub struct VcVerifier;

impl VcVerifier {
    /// Verify a verifiable credential against the issuer's Ed25519 public key.
    ///
    /// Checks:
    /// 1. The Ed25519 signature is valid for the reconstructed unsigned credential.
    /// 2. The credential is not expired.
    /// 3. The issuer DID matches the proof's verification method.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError::CredentialVerification`] if the proof is
    /// structurally invalid (wrong signature size, invalid public key encoding).
    pub fn verify(
        vc: &VerifiableCredential,
        issuer_public_key: &[u8],
    ) -> Result<bool, IdentityError> {
        // Signature must be exactly 64 bytes (Ed25519).
        if vc.proof.proof_value.len() != 64 {
            return Err(IdentityError::CredentialVerification(format!(
                "proof value must be 64 bytes (Ed25519 signature), got {}",
                vc.proof.proof_value.len()
            )));
        }

        // Public key must be exactly 32 bytes.
        if issuer_public_key.len() != 32 {
            return Err(IdentityError::CredentialVerification(format!(
                "issuer public key must be 32 bytes, got {}",
                issuer_public_key.len()
            )));
        }

        // Check expiration before expensive crypto.
        if let Some(expiry) = vc.expiration_date
            && Utc::now() > expiry
        {
            return Ok(false);
        }

        // Verify issuer matches proof verification method.
        if !vc.proof.verification_method.starts_with(&vc.issuer) {
            return Ok(false);
        }

        // Reconstruct the unsigned credential exactly as issued.
        let unsigned = serde_json::json!({
            "context": vc.context,
            "type": vc.credential_type,
            "issuer": vc.issuer,
            "issuanceDate": vc.issuance_date.to_rfc3339(),
            "credentialSubject": vc.credential_subject,
        });

        let unsigned_bytes = serde_json::to_vec(&unsigned)
            .map_err(|e| IdentityError::CredentialVerification(e.to_string()))?;

        // Deserialize the Ed25519 verifying key.
        let pk_bytes: [u8; 32] = issuer_public_key.try_into().map_err(|_| {
            IdentityError::CredentialVerification("invalid public key encoding".into())
        })?;
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&pk_bytes).map_err(|e| {
            IdentityError::CredentialVerification(format!("invalid Ed25519 public key: {e}"))
        })?;

        // Deserialize the signature.
        let sig_bytes: [u8; 64] = vc.proof.proof_value.as_slice().try_into().map_err(|_| {
            IdentityError::CredentialVerification("invalid signature encoding".into())
        })?;
        let signature = ed25519_dalek::Signature::from_bytes(&sig_bytes);

        // Real Ed25519 verification.
        Ok(verifying_key.verify(&unsigned_bytes, &signature).is_ok())
    }
}

/// Generate a deterministic UUID-like string from subject + timestamp.
fn uuid_from_hash(subject: &str, time: DateTime<Utc>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(subject.as_bytes());
    hasher.update(time.to_rfc3339().as_bytes());
    let hash = hasher.finalize();
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes(hash[0..4].try_into().unwrap()),
        u16::from_be_bytes(hash[4..6].try_into().unwrap()),
        u16::from_be_bytes(hash[6..8].try_into().unwrap()),
        u16::from_be_bytes(hash[8..10].try_into().unwrap()),
        // Use 6 bytes for the last segment.
        u64::from_be_bytes({
            let mut buf = [0u8; 8];
            buf[2..8].copy_from_slice(&hash[10..16]);
            buf
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_keypair() -> (VcIssuer, ed25519_dalek::VerifyingKey) {
        let mut rng = rand::rng();
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        let issuer =
            VcIssuer::from_signing_key("did:webvh:example.com:issuer1".into(), signing_key);
        (issuer, verifying_key)
    }

    #[test]
    fn issue_credential() {
        let (issuer, _vk) = test_keypair();
        let vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({"cpu_cores": 8}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        assert!(vc.id.starts_with("urn:uuid:"));
        assert_eq!(vc.issuer, "did:webvh:example.com:issuer1");
        assert!(vc.credential_type.contains(&"NodeCapability".to_string()));
        assert!(vc.expiration_date.is_none());
        // Ed25519 signature is exactly 64 bytes.
        assert_eq!(vc.proof.proof_value.len(), 64);
    }

    #[test]
    fn issue_credential_with_expiry() {
        let (issuer, _vk) = test_keypair();
        let vc = issuer
            .issue(
                CredentialType::EnergyCompliance,
                serde_json::json!({"renewable_pct": 90}),
                "did:webvh:example.com:node1",
                Some(Duration::hours(24)),
            )
            .unwrap();
        assert!(vc.expiration_date.is_some());
    }

    #[test]
    fn issue_fails_with_wrong_key_size() {
        let result = VcIssuer::new("did:webvh:example.com:issuer1".into(), vec![0xAA; 16]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("32 bytes"));
    }

    #[test]
    fn verify_valid_credential() {
        let (issuer, vk) = test_keypair();
        let vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({"cores": 4}),
                "did:webvh:example.com:node1",
                Some(Duration::hours(1)),
            )
            .unwrap();
        let result = VcVerifier::verify(&vc, vk.as_bytes()).unwrap();
        assert!(result, "credential must verify with correct public key");
    }

    #[test]
    fn verify_rejects_wrong_public_key() {
        let (issuer, _vk) = test_keypair();
        let vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({"cores": 4}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        // Use a different key pair's public key.
        let (_other_issuer, other_vk) = test_keypair();
        let result = VcVerifier::verify(&vc, other_vk.as_bytes()).unwrap();
        assert!(
            !result,
            "credential must NOT verify with a different public key"
        );
    }

    #[test]
    fn verify_rejects_tampered_credential() {
        let (issuer, vk) = test_keypair();
        let mut vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({"cores": 4}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        // Tamper with the credential subject.
        vc.credential_subject =
            serde_json::json!({"id": "did:webvh:example.com:node1", "data": {"cores": 999}});
        let result = VcVerifier::verify(&vc, vk.as_bytes()).unwrap();
        assert!(!result, "tampered credential must NOT verify");
    }

    #[test]
    fn verify_empty_proof_fails() {
        let (issuer, vk) = test_keypair();
        let mut vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        vc.proof.proof_value = vec![];
        let err = VcVerifier::verify(&vc, vk.as_bytes()).unwrap_err();
        assert!(err.to_string().contains("64 bytes"));
    }

    #[test]
    fn verify_issuer_mismatch() {
        let (issuer, vk) = test_keypair();
        let mut vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        vc.proof.verification_method = "did:webvh:other.com:attacker#key-1".into();
        let result = VcVerifier::verify(&vc, vk.as_bytes()).unwrap();
        assert!(!result);
    }

    #[test]
    fn verify_rejects_wrong_pk_size() {
        let (issuer, _vk) = test_keypair();
        let vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        let err = VcVerifier::verify(&vc, &[0xBB; 16]).unwrap_err();
        assert!(err.to_string().contains("32 bytes"));
    }

    #[test]
    fn credential_type_as_str() {
        assert_eq!(
            CredentialType::NodeCapability.as_type_str(),
            "NodeCapability"
        );
        assert_eq!(
            CredentialType::EnergyCompliance.as_type_str(),
            "EnergyCompliance"
        );
        assert_eq!(
            CredentialType::MeshMembership.as_type_str(),
            "MeshMembership"
        );
        assert_eq!(
            CredentialType::Custom("FooBar".into()).as_type_str(),
            "FooBar"
        );
    }

    #[test]
    fn credential_serialization_roundtrip() {
        let (issuer, vk) = test_keypair();
        let vc = issuer
            .issue(
                CredentialType::NodeCapability,
                serde_json::json!({"test": true}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();
        let json = serde_json::to_string(&vc).unwrap();
        let parsed: VerifiableCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.issuer, vc.issuer);
        assert_eq!(parsed.proof.proof_value, vc.proof.proof_value);
        // Deserialized credential still verifies.
        assert!(VcVerifier::verify(&parsed, vk.as_bytes()).unwrap());
    }

    #[test]
    fn credential_type_equality() {
        assert_eq!(
            CredentialType::NodeCapability,
            CredentialType::NodeCapability
        );
        assert_ne!(
            CredentialType::NodeCapability,
            CredentialType::MeshMembership
        );
        assert_eq!(
            CredentialType::Custom("A".into()),
            CredentialType::Custom("A".into())
        );
    }

    #[test]
    fn uuid_deterministic() {
        let time = Utc::now();
        let a = uuid_from_hash("test", time);
        let b = uuid_from_hash("test", time);
        assert_eq!(a, b);
    }
}
