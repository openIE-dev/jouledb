use serde::{Deserialize, Serialize};
use std::fmt;

/// Progressive privacy tier from the architecture stack.
///
/// Each tier provides increasing isolation guarantees, from lightweight
/// WASM sandboxes up to confidential GPU computing with hardware-backed
/// memory encryption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PrivacyTier {
    /// WebAssembly sandbox — lightweight isolation via linear memory.
    Tier0WasmSandbox,
    /// Unikernel — single-address-space OS with minimal attack surface.
    Tier1Unikernel,
    /// Micro-VM — lightweight virtual machine (e.g., Firecracker).
    Tier2MicroVm,
    /// Hardware TEE — trusted execution environment (SGX, TrustZone, SEV).
    Tier3HardwareTee,
    /// Confidential GPU — hardware-encrypted GPU memory (NVIDIA CC, AMD SEV-SNP GPU).
    Tier4ConfidentialGpu,
}

impl fmt::Display for PrivacyTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tier0WasmSandbox => write!(f, "Tier0-WasmSandbox"),
            Self::Tier1Unikernel => write!(f, "Tier1-Unikernel"),
            Self::Tier2MicroVm => write!(f, "Tier2-MicroVM"),
            Self::Tier3HardwareTee => write!(f, "Tier3-HardwareTEE"),
            Self::Tier4ConfidentialGpu => write!(f, "Tier4-ConfidentialGPU"),
        }
    }
}

/// Trust level assigned to a node or attestation result.
///
/// Ordered from least to most trustworthy. Comparisons use derived
/// `Ord` so `HardwareAttested > SelfAttested` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustLevel {
    /// No verification has been performed.
    Unverified,
    /// The node attested its own state (software-only).
    SelfAttested,
    /// A third-party service has verified the attestation.
    ThirdPartyAttested,
    /// Hardware-rooted attestation (TPM, SEV, SGX).
    HardwareAttested,
    /// Formally verified software and hardware stack.
    FormallyVerified,
}

impl fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unverified => write!(f, "Unverified"),
            Self::SelfAttested => write!(f, "SelfAttested"),
            Self::ThirdPartyAttested => write!(f, "ThirdPartyAttested"),
            Self::HardwareAttested => write!(f, "HardwareAttested"),
            Self::FormallyVerified => write!(f, "FormallyVerified"),
        }
    }
}

/// A workload's privacy requirements — the minimum isolation and trust
/// guarantees it demands from a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyRequirement {
    /// Minimum privacy tier the node must support.
    pub min_tier: PrivacyTier,
    /// Minimum trust level the node must have achieved.
    pub required_trust_level: TrustLevel,
    /// Whether data must be encrypted at rest.
    pub encrypt_at_rest: bool,
    /// Whether data must be encrypted in transit.
    pub encrypt_in_transit: bool,
    /// Whether zero-knowledge verification is required for the workload.
    pub zk_verification: bool,
}

impl PrivacyRequirement {
    /// Create a new privacy requirement with the given tier and trust level.
    /// Encryption flags default to `true` and ZK verification defaults to `false`.
    pub fn new(min_tier: PrivacyTier, required_trust_level: TrustLevel) -> Self {
        Self {
            min_tier,
            required_trust_level,
            encrypt_at_rest: true,
            encrypt_in_transit: true,
            zk_verification: false,
        }
    }
}

/// A node's privacy capability — what isolation and attestation it provides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyCapability {
    /// The highest privacy tier this node supports.
    pub supported_tier: PrivacyTier,
    /// The trust level this node has achieved.
    pub trust_level: TrustLevel,
    /// The type of TEE available, if any (e.g., "SGX", "SEV-SNP", "TrustZone").
    pub tee_type: Option<String>,
    /// Whether remote attestation is available on this node.
    pub attestation_available: bool,
}

impl PrivacyCapability {
    /// Check whether this capability satisfies the given privacy requirement.
    ///
    /// A capability satisfies a requirement when:
    /// - `supported_tier >= min_tier`
    /// - `trust_level >= required_trust_level`
    /// - If `zk_verification` is required, `attestation_available` must be `true`
    pub fn satisfies(&self, requirement: &PrivacyRequirement) -> bool {
        if self.supported_tier < requirement.min_tier {
            return false;
        }
        if self.trust_level < requirement.required_trust_level {
            return false;
        }
        if requirement.zk_verification && !self.attestation_available {
            return false;
        }
        true
    }
}

impl fmt::Display for PrivacyCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrivacyCapability(tier={}, trust={}, tee={}, attestation={})",
            self.supported_tier,
            self.trust_level,
            self.tee_type.as_deref().unwrap_or("none"),
            self.attestation_available,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn privacy_tier_ordering() {
        assert!(PrivacyTier::Tier0WasmSandbox < PrivacyTier::Tier1Unikernel);
        assert!(PrivacyTier::Tier1Unikernel < PrivacyTier::Tier2MicroVm);
        assert!(PrivacyTier::Tier2MicroVm < PrivacyTier::Tier3HardwareTee);
        assert!(PrivacyTier::Tier3HardwareTee < PrivacyTier::Tier4ConfidentialGpu);
    }

    #[test]
    fn trust_level_ordering() {
        assert!(TrustLevel::Unverified < TrustLevel::SelfAttested);
        assert!(TrustLevel::SelfAttested < TrustLevel::ThirdPartyAttested);
        assert!(TrustLevel::ThirdPartyAttested < TrustLevel::HardwareAttested);
        assert!(TrustLevel::HardwareAttested < TrustLevel::FormallyVerified);
    }

    #[test]
    fn satisfies_exact_match() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier2MicroVm,
            trust_level: TrustLevel::ThirdPartyAttested,
            tee_type: None,
            attestation_available: false,
        };
        let req =
            PrivacyRequirement::new(PrivacyTier::Tier2MicroVm, TrustLevel::ThirdPartyAttested);
        assert!(cap.satisfies(&req));
    }

    #[test]
    fn satisfies_higher_tier_than_required() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier4ConfidentialGpu,
            trust_level: TrustLevel::HardwareAttested,
            tee_type: Some("SEV-SNP".into()),
            attestation_available: true,
        };
        let req = PrivacyRequirement::new(PrivacyTier::Tier1Unikernel, TrustLevel::SelfAttested);
        assert!(cap.satisfies(&req));
    }

    #[test]
    fn does_not_satisfy_lower_tier() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier0WasmSandbox,
            trust_level: TrustLevel::FormallyVerified,
            tee_type: None,
            attestation_available: true,
        };
        let req = PrivacyRequirement::new(PrivacyTier::Tier3HardwareTee, TrustLevel::Unverified);
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn does_not_satisfy_lower_trust() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier3HardwareTee,
            trust_level: TrustLevel::SelfAttested,
            tee_type: Some("SGX".into()),
            attestation_available: true,
        };
        let req =
            PrivacyRequirement::new(PrivacyTier::Tier0WasmSandbox, TrustLevel::HardwareAttested);
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn zk_verification_requires_attestation() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier4ConfidentialGpu,
            trust_level: TrustLevel::FormallyVerified,
            tee_type: Some("SEV-SNP".into()),
            attestation_available: false,
        };
        let mut req =
            PrivacyRequirement::new(PrivacyTier::Tier0WasmSandbox, TrustLevel::Unverified);
        req.zk_verification = true;
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn zk_verification_satisfied_with_attestation() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier3HardwareTee,
            trust_level: TrustLevel::HardwareAttested,
            tee_type: Some("SGX".into()),
            attestation_available: true,
        };
        let mut req =
            PrivacyRequirement::new(PrivacyTier::Tier2MicroVm, TrustLevel::ThirdPartyAttested);
        req.zk_verification = true;
        assert!(cap.satisfies(&req));
    }

    #[test]
    fn privacy_tier_display() {
        assert_eq!(
            format!("{}", PrivacyTier::Tier0WasmSandbox),
            "Tier0-WasmSandbox"
        );
        assert_eq!(
            format!("{}", PrivacyTier::Tier3HardwareTee),
            "Tier3-HardwareTEE"
        );
        assert_eq!(
            format!("{}", PrivacyTier::Tier4ConfidentialGpu),
            "Tier4-ConfidentialGPU"
        );
    }

    #[test]
    fn trust_level_display() {
        assert_eq!(format!("{}", TrustLevel::Unverified), "Unverified");
        assert_eq!(
            format!("{}", TrustLevel::FormallyVerified),
            "FormallyVerified"
        );
    }

    #[test]
    fn privacy_capability_display() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier3HardwareTee,
            trust_level: TrustLevel::HardwareAttested,
            tee_type: Some("SGX".into()),
            attestation_available: true,
        };
        let display = format!("{cap}");
        assert!(display.contains("Tier3-HardwareTEE"));
        assert!(display.contains("HardwareAttested"));
        assert!(display.contains("SGX"));
        assert!(display.contains("attestation=true"));
    }

    #[test]
    fn privacy_requirement_default_encryption() {
        let req = PrivacyRequirement::new(PrivacyTier::Tier0WasmSandbox, TrustLevel::Unverified);
        assert!(req.encrypt_at_rest);
        assert!(req.encrypt_in_transit);
        assert!(!req.zk_verification);
    }

    #[test]
    fn serialization_roundtrip_requirement() {
        let req = PrivacyRequirement {
            min_tier: PrivacyTier::Tier3HardwareTee,
            required_trust_level: TrustLevel::HardwareAttested,
            encrypt_at_rest: true,
            encrypt_in_transit: true,
            zk_verification: true,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: PrivacyRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.min_tier, PrivacyTier::Tier3HardwareTee);
        assert_eq!(parsed.required_trust_level, TrustLevel::HardwareAttested);
        assert!(parsed.zk_verification);
    }

    #[test]
    fn serialization_roundtrip_capability() {
        let cap = PrivacyCapability {
            supported_tier: PrivacyTier::Tier4ConfidentialGpu,
            trust_level: TrustLevel::FormallyVerified,
            tee_type: Some("SEV-SNP".into()),
            attestation_available: true,
        };
        let json = serde_json::to_string(&cap).unwrap();
        let parsed: PrivacyCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.supported_tier, PrivacyTier::Tier4ConfidentialGpu);
        assert_eq!(parsed.trust_level, TrustLevel::FormallyVerified);
        assert_eq!(parsed.tee_type.as_deref(), Some("SEV-SNP"));
        assert!(parsed.attestation_available);
    }
}
