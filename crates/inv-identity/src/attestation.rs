//! Bridge between IETF RATS attestation (from `inv-core`) and DID/VC.
//!
//! Converts hardware attestation evidence into verifiable credentials,
//! enabling the mesh to treat attestation results as portable, signed claims.

use serde::{Deserialize, Serialize};

use crate::IdentityError;
use crate::vc::{CredentialType, VcIssuer, VcVerifier, VerifiableCredential};

/// A verifiable credential wrapping attestation evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationCredential {
    /// The underlying verifiable credential.
    pub credential: VerifiableCredential,
}

impl AttestationCredential {
    /// Access the inner verifiable credential.
    pub fn as_vc(&self) -> &VerifiableCredential {
        &self.credential
    }

    /// Check whether this credential contains attestation data.
    pub fn is_attestation(&self) -> bool {
        self.credential
            .credential_type
            .iter()
            .any(|t| t == "RemoteAttestation")
    }
}

/// Issue a verifiable credential from attestation evidence.
///
/// Serializes the [`inv_core::AttestationEvidence`] as the credential subject
/// and creates a `RemoteAttestation` typed VC.
pub fn issue_attestation_vc(
    issuer: &VcIssuer,
    evidence: &inv_core::AttestationEvidence,
    subject_did: &str,
) -> Result<VerifiableCredential, IdentityError> {
    let subject = serde_json::to_value(evidence)
        .map_err(|e| IdentityError::AttestationBridge(e.to_string()))?;

    issuer.issue(
        CredentialType::Custom("RemoteAttestation".into()),
        subject,
        subject_did,
        None,
    )
}

/// Verify an attestation verifiable credential.
///
/// Checks both the VC proof (real Ed25519 signature verification) and
/// that the credential type includes `RemoteAttestation`.
pub fn verify_attestation_vc(
    vc: &VerifiableCredential,
    issuer_public_key: &[u8],
) -> Result<bool, IdentityError> {
    // Verify the underlying VC proof.
    let proof_valid = VcVerifier::verify(vc, issuer_public_key)?;
    if !proof_valid {
        return Ok(false);
    }

    // Check that the credential is an attestation type.
    let is_attestation = vc.credential_type.iter().any(|t| t == "RemoteAttestation");

    Ok(is_attestation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_evidence() -> inv_core::AttestationEvidence {
        inv_core::AttestationEvidence {
            platform: "Intel-SGX".into(),
            measurement: [0xAB; 48],
            report_data: vec![0x01, 0x02],
            signature: vec![0xFF; 64],
            timestamp_ms: 1_700_000_000_000,
        }
    }

    fn test_issuer() -> (VcIssuer, ed25519_dalek::VerifyingKey) {
        let mut rng = rand::rng();
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rng);
        let verifying_key = signing_key.verifying_key();
        let issuer =
            VcIssuer::from_signing_key("did:webvh:example.com:verifier".into(), signing_key);
        (issuer, verifying_key)
    }

    #[test]
    fn issue_and_verify_attestation_vc() {
        let (issuer, vk) = test_issuer();
        let evidence = test_evidence();
        let vc = issue_attestation_vc(&issuer, &evidence, "did:webvh:example.com:node1").unwrap();

        assert!(
            vc.credential_type
                .contains(&"RemoteAttestation".to_string())
        );

        let valid = verify_attestation_vc(&vc, vk.as_bytes()).unwrap();
        assert!(valid);
    }

    #[test]
    fn verify_non_attestation_vc_fails() {
        let (issuer, vk) = test_issuer();
        let vc = issuer
            .issue(
                crate::vc::CredentialType::NodeCapability,
                serde_json::json!({}),
                "did:webvh:example.com:node1",
                None,
            )
            .unwrap();

        // This is a NodeCapability VC, not RemoteAttestation.
        let valid = verify_attestation_vc(&vc, vk.as_bytes()).unwrap();
        assert!(!valid);
    }

    #[test]
    fn verify_attestation_vc_rejects_wrong_key() {
        let (issuer, _vk) = test_issuer();
        let evidence = test_evidence();
        let vc = issue_attestation_vc(&issuer, &evidence, "did:webvh:example.com:node1").unwrap();

        // Use a different key pair — signature must not verify.
        let (_other_issuer, other_vk) = test_issuer();
        let valid = verify_attestation_vc(&vc, other_vk.as_bytes()).unwrap();
        assert!(!valid, "attestation VC must not verify with wrong key");
    }

    #[test]
    fn attestation_credential_wrapper() {
        let (issuer, _vk) = test_issuer();
        let evidence = test_evidence();
        let vc = issue_attestation_vc(&issuer, &evidence, "did:webvh:example.com:node1").unwrap();

        let att_cred = AttestationCredential {
            credential: vc.clone(),
        };
        assert!(att_cred.is_attestation());
        assert_eq!(att_cred.as_vc().issuer, vc.issuer);
    }

    #[test]
    fn attestation_credential_serialization() {
        let (issuer, _vk) = test_issuer();
        let evidence = test_evidence();
        let vc = issue_attestation_vc(&issuer, &evidence, "did:webvh:example.com:node1").unwrap();

        let att = AttestationCredential { credential: vc };
        let json = serde_json::to_string(&att).unwrap();
        let parsed: AttestationCredential = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_attestation());
    }
}
