//! Supply chain verification.
//!
//! SBOM (CycloneDX/SPDX) management, Sigstore verification data validation,
//! and binary provenance checks. Integrates with `inv-core::SupplyChainAttestation`.
//!
//! **Note:** Full Sigstore cryptographic verification (certificate chain,
//! signature, Rekor transparency log) requires the `sigstore-rs` crate.
//! This module validates structural completeness of verification data and
//! performs real SHA-256 hash-based provenance checks.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// SBOM format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SbomFormat {
    /// CycloneDX JSON/XML format.
    CycloneDx,
    /// SPDX format.
    Spdx,
}

/// A software bill of materials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sbom {
    /// The SBOM format.
    pub format: SbomFormat,
    /// Components listed in the SBOM.
    pub components: Vec<SbomComponent>,
    /// When the SBOM was generated.
    pub generated_at: DateTime<Utc>,
    /// Version of the tool that generated the SBOM.
    pub tool_version: String,
}

impl Sbom {
    /// Compute the SHA-256 hash of the SBOM content.
    pub fn content_hash(&self) -> [u8; 32] {
        let bytes = serde_json::to_vec(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }

    /// Total number of components.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }
}

/// A component in an SBOM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbomComponent {
    /// Component name (e.g., "serde").
    pub name: String,
    /// Component version (e.g., "1.0.228").
    pub version: String,
    /// Package URL (purl), if available.
    pub purl: Option<String>,
    /// Hashes of the component artifact.
    pub hashes: Vec<ComponentHash>,
    /// License identifier, if known.
    pub license: Option<String>,
}

/// A hash of a component artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHash {
    /// Hash algorithm (e.g., "SHA-256", "SHA-512").
    pub algorithm: String,
    /// Hex-encoded hash value.
    pub value: String,
}

/// Sigstore verification data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigstoreVerification {
    /// The Sigstore signature bytes.
    pub signature: Vec<u8>,
    /// The signing certificate bytes.
    pub certificate: Vec<u8>,
    /// Rekor transparency log entry.
    pub log_entry: String,
}

/// Result of a `cargo audit` scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoAuditResult {
    /// Number of known vulnerabilities found.
    pub vulnerabilities: u32,
    /// Number of warnings.
    pub warnings: u32,
    /// Advisory IDs (e.g., "RUSTSEC-2024-0001").
    pub advisories: Vec<String>,
}

impl CargoAuditResult {
    /// Whether the audit passed with no vulnerabilities.
    pub fn is_clean(&self) -> bool {
        self.vulnerabilities == 0
    }
}

/// Result of a `cargo vet` audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoVetResult {
    /// Number of fully audited crates.
    pub audited: u32,
    /// Number of unaudited crates.
    pub unaudited: u32,
    /// Number of exempted crates.
    pub exempted: u32,
}

impl CargoVetResult {
    /// Whether all crates are either audited or exempted.
    pub fn is_complete(&self) -> bool {
        self.unaudited == 0
    }
}

/// Errors from supply chain verification.
#[derive(Debug, thiserror::Error)]
pub enum SupplyChainError {
    /// SBOM-related error.
    #[error("SBOM error: {0}")]
    SbomError(String),
    /// Signature verification error.
    #[error("signature error: {0}")]
    SignatureError(String),
    /// Binary provenance error.
    #[error("provenance error: {0}")]
    ProvenanceError(String),
    /// Audit error.
    #[error("audit error: {0}")]
    AuditError(String),
}

/// Validate the structural completeness of Sigstore verification data for an SBOM.
///
/// This function checks that all required fields are present and the SBOM
/// content hashes to a non-trivial value. It does **not** perform full
/// Sigstore cryptographic verification (certificate chain validation,
/// signature verification, or Rekor transparency log inclusion proof).
///
/// Full cryptographic verification requires a Sigstore verifier backend
/// (e.g., `sigstore-rs`) and is out of scope for this abstraction layer.
///
/// # Checks performed
///
/// 1. Signature bytes are non-empty.
/// 2. Certificate bytes are non-empty.
/// 3. Rekor log entry identifier is non-empty.
/// 4. SBOM content hash is non-trivial (not all zeros).
pub fn verify_sbom_signature(
    sbom: &Sbom,
    verification: &SigstoreVerification,
) -> Result<bool, SupplyChainError> {
    if verification.signature.is_empty() {
        return Err(SupplyChainError::SignatureError(
            "signature is empty".into(),
        ));
    }
    if verification.certificate.is_empty() {
        return Err(SupplyChainError::SignatureError(
            "certificate is empty".into(),
        ));
    }
    if verification.log_entry.is_empty() {
        return Err(SupplyChainError::SignatureError(
            "Rekor log entry is empty".into(),
        ));
    }

    // Structural check: SBOM must hash to a non-trivial value.
    let sbom_hash = sbom.content_hash();
    Ok(!sbom_hash.iter().all(|&b| b == 0))
}

/// Verify binary provenance against an SBOM and supply chain attestation.
///
/// Checks that the binary hash matches the attestation and the SBOM
/// content hash matches the attestation's SBOM hash.
pub fn verify_binary_provenance(
    binary_hash: &[u8; 32],
    sbom: &Sbom,
    attestation: &inv_core::SupplyChainAttestation,
) -> Result<bool, SupplyChainError> {
    // Check binary hash matches attestation.
    if *binary_hash != attestation.binary_hash {
        return Ok(false);
    }

    // Check SBOM hash matches attestation.
    let computed_sbom_hash = sbom.content_hash();
    if computed_sbom_hash != attestation.sbom_hash {
        return Ok(false);
    }

    // Verify the attestation's composite hash.
    let composite = attestation.composite_hash();
    Ok(attestation.verify_supply_chain(&composite))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sbom() -> Sbom {
        Sbom {
            format: SbomFormat::CycloneDx,
            components: vec![
                SbomComponent {
                    name: "serde".into(),
                    version: "1.0.228".into(),
                    purl: Some("pkg:cargo/serde@1.0.228".into()),
                    hashes: vec![ComponentHash {
                        algorithm: "SHA-256".into(),
                        value: "abc123".into(),
                    }],
                    license: Some("MIT OR Apache-2.0".into()),
                },
                SbomComponent {
                    name: "tokio".into(),
                    version: "1.43.0".into(),
                    purl: Some("pkg:cargo/tokio@1.43.0".into()),
                    hashes: vec![],
                    license: Some("MIT".into()),
                },
            ],
            generated_at: Utc::now(),
            tool_version: "1.0.0".into(),
        }
    }

    fn test_verification() -> SigstoreVerification {
        SigstoreVerification {
            signature: vec![0xFF; 64],
            certificate: vec![0xAA; 256],
            log_entry: "rekor-entry-12345".into(),
        }
    }

    #[test]
    fn verify_sbom_signature_valid() {
        let sbom = test_sbom();
        let verification = test_verification();
        let result = verify_sbom_signature(&sbom, &verification).unwrap();
        assert!(result);
    }

    #[test]
    fn verify_sbom_signature_empty_sig() {
        let sbom = test_sbom();
        let mut verification = test_verification();
        verification.signature = vec![];
        let err = verify_sbom_signature(&sbom, &verification).unwrap_err();
        assert!(matches!(err, SupplyChainError::SignatureError(_)));
    }

    #[test]
    fn verify_sbom_signature_empty_cert() {
        let sbom = test_sbom();
        let mut verification = test_verification();
        verification.certificate = vec![];
        let err = verify_sbom_signature(&sbom, &verification).unwrap_err();
        assert!(matches!(err, SupplyChainError::SignatureError(_)));
    }

    #[test]
    fn verify_binary_provenance_valid() {
        let sbom = test_sbom();
        let sbom_hash = sbom.content_hash();
        let binary_hash = [0x11; 32];
        let attestation = inv_core::SupplyChainAttestation {
            binary_hash,
            sbom_hash,
            signer: "ci@example.com".into(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        let result = verify_binary_provenance(&binary_hash, &sbom, &attestation).unwrap();
        assert!(result);
    }

    #[test]
    fn verify_binary_provenance_hash_mismatch() {
        let sbom = test_sbom();
        let sbom_hash = sbom.content_hash();
        let attestation = inv_core::SupplyChainAttestation {
            binary_hash: [0x11; 32],
            sbom_hash,
            signer: "ci@example.com".into(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        let wrong_binary = [0x22; 32];
        let result = verify_binary_provenance(&wrong_binary, &sbom, &attestation).unwrap();
        assert!(!result);
    }

    #[test]
    fn sbom_component_count() {
        let sbom = test_sbom();
        assert_eq!(sbom.component_count(), 2);
    }

    #[test]
    fn sbom_content_hash_deterministic() {
        let sbom = test_sbom();
        let h1 = sbom.content_hash();
        let h2 = sbom.content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn cargo_audit_clean() {
        let result = CargoAuditResult {
            vulnerabilities: 0,
            warnings: 2,
            advisories: vec![],
        };
        assert!(result.is_clean());
    }

    #[test]
    fn cargo_audit_dirty() {
        let result = CargoAuditResult {
            vulnerabilities: 1,
            warnings: 0,
            advisories: vec!["RUSTSEC-2024-0001".into()],
        };
        assert!(!result.is_clean());
    }

    #[test]
    fn cargo_vet_complete() {
        let result = CargoVetResult {
            audited: 150,
            unaudited: 0,
            exempted: 5,
        };
        assert!(result.is_complete());
    }

    #[test]
    fn sbom_format_serialization() {
        let json = serde_json::to_string(&SbomFormat::CycloneDx).unwrap();
        let parsed: SbomFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, SbomFormat::CycloneDx);
    }
}
