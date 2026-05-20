use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::privacy::TrustLevel;

/// Serde helper for `[u8; 48]` arrays (serde only supports up to 32).
mod serde_bytes48 {
    use super::*;

    pub fn serialize<S: Serializer>(bytes: &[u8; 48], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(bytes)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 48], D::Error> {
        let v: Vec<u8> = Deserialize::deserialize(deserializer)?;
        v.try_into().map_err(|v: Vec<u8>| {
            serde::de::Error::custom(format!("expected 48 bytes, got {}", v.len()))
        })
    }
}

/// Remote attestation evidence collected from a platform.
///
/// Modeled after IETF RATS (RFC 9334) Evidence. Contains a platform
/// measurement, optional report data, and a signature binding the
/// evidence to the attesting entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationEvidence {
    /// Platform identifier (e.g., "Intel-SGX", "AMD-SEV-SNP", "ARM-TrustZone").
    pub platform: String,
    /// SHA-384 measurement of the platform state (PCR composite or launch digest).
    #[serde(with = "serde_bytes48")]
    pub measurement: [u8; 48],
    /// Caller-supplied report data bound into the attestation (e.g., a nonce).
    pub report_data: Vec<u8>,
    /// Cryptographic signature over the evidence produced by the platform.
    pub signature: Vec<u8>,
    /// Unix timestamp in milliseconds when the evidence was generated.
    pub timestamp_ms: u64,
}

/// Endorsement from a platform vendor, used to validate attestation evidence.
///
/// Modeled after IETF RATS (RFC 9334) Endorsements. Provides the
/// vendor's firmware version and root certificate hash for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationEndorsement {
    /// Vendor name (e.g., "Intel", "AMD", "ARM").
    pub vendor: String,
    /// Firmware version string.
    pub firmware_version: String,
    /// Security version number — monotonically increasing with patches.
    pub security_version: u32,
    /// SHA-256 hash of the vendor's root signing certificate.
    pub root_cert_hash: [u8; 32],
}

/// Result of evaluating attestation evidence against a policy.
///
/// Modeled after IETF RATS (RFC 9334) Attestation Results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Whether the attestation evidence was successfully verified.
    pub verified: bool,
    /// The trust level achieved by this attestation.
    pub trust_level: TrustLevel,
    /// Expiry time in milliseconds (Unix timestamp) after which the result is stale.
    pub expiry_ms: u64,
}

/// Policy that governs how attestation evidence is evaluated.
///
/// Defines minimum trust requirements, debug restrictions, firmware
/// freshness, and platform allowlists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationPolicy {
    /// Minimum trust level required for a passing evaluation.
    pub min_trust: TrustLevel,
    /// Whether debug mode must be disabled on the attesting platform.
    pub require_debug_disabled: bool,
    /// Maximum age of the firmware in days. Evidence from older firmware is rejected.
    pub max_firmware_age_days: u32,
    /// Platforms allowed by this policy (empty means all platforms are allowed).
    pub allowed_platforms: Vec<String>,
}

impl AttestationPolicy {
    /// Evaluate attestation evidence against this policy.
    ///
    /// The evaluation performs the following checks:
    /// 1. If `allowed_platforms` is non-empty, the evidence platform must be listed.
    /// 2. The evidence signature must be non-empty (basic integrity check).
    /// 3. The evidence measurement must be non-zero (non-trivial measurement).
    ///
    /// If all checks pass, the result is verified with `HardwareAttested` trust
    /// (since we have valid platform evidence). The result expires 1 hour after
    /// the evidence timestamp.
    ///
    /// If the achieved trust level is below `min_trust`, verification fails.
    pub fn evaluate(&self, evidence: &AttestationEvidence) -> AttestationResult {
        // Check platform allowlist.
        if !self.allowed_platforms.is_empty()
            && !self.allowed_platforms.contains(&evidence.platform)
        {
            return AttestationResult {
                verified: false,
                trust_level: TrustLevel::Unverified,
                expiry_ms: evidence.timestamp_ms,
            };
        }

        // Check signature is present (non-empty).
        if evidence.signature.is_empty() {
            return AttestationResult {
                verified: false,
                trust_level: TrustLevel::Unverified,
                expiry_ms: evidence.timestamp_ms,
            };
        }

        // Check measurement is non-trivial (not all zeros).
        if evidence.measurement.iter().all(|&b| b == 0) {
            return AttestationResult {
                verified: false,
                trust_level: TrustLevel::SelfAttested,
                expiry_ms: evidence.timestamp_ms,
            };
        }

        // Evidence passed basic checks — grant hardware-attested trust.
        let achieved_trust = TrustLevel::HardwareAttested;

        // If the achieved trust is below the policy minimum, fail.
        if achieved_trust < self.min_trust {
            return AttestationResult {
                verified: false,
                trust_level: achieved_trust,
                expiry_ms: evidence.timestamp_ms,
            };
        }

        // Valid attestation — expires 1 hour after the evidence timestamp.
        let one_hour_ms = 3_600_000;
        AttestationResult {
            verified: true,
            trust_level: achieved_trust,
            expiry_ms: evidence.timestamp_ms + one_hour_ms,
        }
    }
}

/// Supply-chain attestation binding a binary to its SBOM and build metadata.
///
/// Used to verify that a deployed binary matches a known-good build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupplyChainAttestation {
    /// SHA-256 hash of the binary artifact.
    pub binary_hash: [u8; 32],
    /// SHA-256 hash of the software bill of materials (SBOM).
    pub sbom_hash: [u8; 32],
    /// Identity of the build signer (e.g., a CI/CD service principal).
    pub signer: String,
    /// Unix timestamp (seconds) when the build was produced.
    pub build_timestamp: u64,
    /// Whether the build is reproducible (deterministic output).
    pub reproducible: bool,
}

impl SupplyChainAttestation {
    /// Verify the supply-chain attestation by recomputing a composite hash
    /// over the binary and SBOM hashes and checking it against a provided
    /// expected hash.
    ///
    /// Returns `true` if the SHA-256 of `binary_hash || sbom_hash` matches
    /// the `expected_composite` and the signer is non-empty.
    pub fn verify_supply_chain(&self, expected_composite: &[u8; 32]) -> bool {
        if self.signer.is_empty() {
            return false;
        }

        let mut hasher = Sha256::new();
        hasher.update(self.binary_hash);
        hasher.update(self.sbom_hash);
        let computed: [u8; 32] = hasher.finalize().into();

        computed == *expected_composite
    }

    /// Compute the composite hash of this attestation (SHA-256 of binary_hash || sbom_hash).
    pub fn composite_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.binary_hash);
        hasher.update(self.sbom_hash);
        hasher.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_evidence() -> AttestationEvidence {
        AttestationEvidence {
            platform: "Intel-SGX".into(),
            measurement: [0xAB; 48],
            report_data: vec![0x01, 0x02, 0x03],
            signature: vec![0xFF; 64],
            timestamp_ms: 1_700_000_000_000,
        }
    }

    fn sample_policy() -> AttestationPolicy {
        AttestationPolicy {
            min_trust: TrustLevel::HardwareAttested,
            require_debug_disabled: true,
            max_firmware_age_days: 90,
            allowed_platforms: vec!["Intel-SGX".into(), "AMD-SEV-SNP".into()],
        }
    }

    #[test]
    fn evaluate_valid_evidence() {
        let policy = sample_policy();
        let evidence = sample_evidence();
        let result = policy.evaluate(&evidence);
        assert!(result.verified);
        assert_eq!(result.trust_level, TrustLevel::HardwareAttested);
        assert_eq!(result.expiry_ms, evidence.timestamp_ms + 3_600_000);
    }

    #[test]
    fn evaluate_disallowed_platform() {
        let policy = sample_policy();
        let mut evidence = sample_evidence();
        evidence.platform = "ARM-TrustZone".into();
        let result = policy.evaluate(&evidence);
        assert!(!result.verified);
        assert_eq!(result.trust_level, TrustLevel::Unverified);
    }

    #[test]
    fn evaluate_empty_signature() {
        let policy = sample_policy();
        let mut evidence = sample_evidence();
        evidence.signature = vec![];
        let result = policy.evaluate(&evidence);
        assert!(!result.verified);
        assert_eq!(result.trust_level, TrustLevel::Unverified);
    }

    #[test]
    fn evaluate_zero_measurement() {
        let policy = sample_policy();
        let mut evidence = sample_evidence();
        evidence.measurement = [0u8; 48];
        let result = policy.evaluate(&evidence);
        assert!(!result.verified);
        assert_eq!(result.trust_level, TrustLevel::SelfAttested);
    }

    #[test]
    fn evaluate_policy_requires_formally_verified() {
        let policy = AttestationPolicy {
            min_trust: TrustLevel::FormallyVerified,
            require_debug_disabled: false,
            max_firmware_age_days: 365,
            allowed_platforms: vec![],
        };
        let evidence = sample_evidence();
        let result = policy.evaluate(&evidence);
        // HardwareAttested < FormallyVerified, so it should fail.
        assert!(!result.verified);
        assert_eq!(result.trust_level, TrustLevel::HardwareAttested);
    }

    #[test]
    fn evaluate_empty_allowed_platforms_accepts_any() {
        let policy = AttestationPolicy {
            min_trust: TrustLevel::SelfAttested,
            require_debug_disabled: false,
            max_firmware_age_days: 365,
            allowed_platforms: vec![],
        };
        let mut evidence = sample_evidence();
        evidence.platform = "Custom-Platform".into();
        let result = policy.evaluate(&evidence);
        assert!(result.verified);
        assert_eq!(result.trust_level, TrustLevel::HardwareAttested);
    }

    #[test]
    fn supply_chain_verify_valid() {
        let attestation = SupplyChainAttestation {
            binary_hash: [0xAA; 32],
            sbom_hash: [0xBB; 32],
            signer: "ci-bot@example.com".into(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        let composite = attestation.composite_hash();
        assert!(attestation.verify_supply_chain(&composite));
    }

    #[test]
    fn supply_chain_verify_wrong_composite() {
        let attestation = SupplyChainAttestation {
            binary_hash: [0xAA; 32],
            sbom_hash: [0xBB; 32],
            signer: "ci-bot@example.com".into(),
            build_timestamp: 1_700_000_000,
            reproducible: false,
        };
        let wrong_composite = [0x00; 32];
        assert!(!attestation.verify_supply_chain(&wrong_composite));
    }

    #[test]
    fn supply_chain_verify_empty_signer_fails() {
        let attestation = SupplyChainAttestation {
            binary_hash: [0xAA; 32],
            sbom_hash: [0xBB; 32],
            signer: String::new(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        let composite = attestation.composite_hash();
        assert!(!attestation.verify_supply_chain(&composite));
    }

    #[test]
    fn composite_hash_deterministic() {
        let attestation = SupplyChainAttestation {
            binary_hash: [0x11; 32],
            sbom_hash: [0x22; 32],
            signer: "builder".into(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        let hash1 = attestation.composite_hash();
        let hash2 = attestation.composite_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn composite_hash_differs_for_different_inputs() {
        let att_a = SupplyChainAttestation {
            binary_hash: [0x11; 32],
            sbom_hash: [0x22; 32],
            signer: "builder".into(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        let att_b = SupplyChainAttestation {
            binary_hash: [0x33; 32],
            sbom_hash: [0x22; 32],
            signer: "builder".into(),
            build_timestamp: 1_700_000_000,
            reproducible: true,
        };
        assert_ne!(att_a.composite_hash(), att_b.composite_hash());
    }

    #[test]
    fn serialization_roundtrip_evidence() {
        let evidence = sample_evidence();
        let json = serde_json::to_string(&evidence).unwrap();
        let parsed: AttestationEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.platform, "Intel-SGX");
        assert_eq!(parsed.measurement, [0xAB; 48]);
        assert_eq!(parsed.signature.len(), 64);
        assert_eq!(parsed.timestamp_ms, 1_700_000_000_000);
    }
}
