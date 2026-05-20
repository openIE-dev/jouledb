//! Decentralized identity for Invisible Infrastructure.
//!
//! This crate provides:
//!
//! - **DID Documents** (`did:webvh` method) — decentralized identifiers for mesh nodes
//! - **W3C Verifiable Credentials 2.0** — cryptographically signed claims
//! - **IETF RATS attestation bridge** — linking hardware attestation to verifiable credentials

pub mod attestation;
pub mod did;
pub mod vc;

// Re-export key types from the did module.
pub use did::{
    DidDocument, DidResolver, DidService, LocalDidResolver, VerificationMethod,
    VerificationMethodType, create_node_did,
};

// Re-export key types from the vc module.
pub use vc::{CredentialProof, CredentialType, VcIssuer, VcVerifier, VerifiableCredential};

// Re-export key types from the attestation module.
pub use attestation::{AttestationCredential, issue_attestation_vc, verify_attestation_vc};

/// Errors returned by identity operations.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    /// DID resolution failed.
    #[error("DID resolution failed: {0}")]
    DidResolution(String),

    /// Credential issuance failed.
    #[error("credential issuance failed: {0}")]
    CredentialIssuance(String),

    /// Credential verification failed.
    #[error("credential verification failed: {0}")]
    CredentialVerification(String),

    /// Attestation bridge error.
    #[error("attestation bridge error: {0}")]
    AttestationBridge(String),

    /// The document is structurally invalid.
    #[error("invalid document: {0}")]
    InvalidDocument(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let e = IdentityError::DidResolution("not found".into());
        assert_eq!(e.to_string(), "DID resolution failed: not found");

        let e = IdentityError::CredentialIssuance("expired key".into());
        assert_eq!(e.to_string(), "credential issuance failed: expired key");

        let e = IdentityError::CredentialVerification("bad proof".into());
        assert_eq!(e.to_string(), "credential verification failed: bad proof");

        let e = IdentityError::AttestationBridge("missing evidence".into());
        assert_eq!(e.to_string(), "attestation bridge error: missing evidence");

        let e = IdentityError::InvalidDocument("no controller".into());
        assert_eq!(e.to_string(), "invalid document: no controller");
    }
}
