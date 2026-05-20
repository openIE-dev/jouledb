//! Attestation-based authentication for TEE-backed mesh nodes.
//!
//! Provides an alternative to JWT-based auth: nodes presenting valid
//! IETF RATS attestation evidence receive short-lived tokens at a
//! trust level derived from the hardware attestation result.

use inv_core::{AttestationEvidence, AttestationResult, TrustLevel};
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from attestation-based authentication.
#[derive(Debug, Error)]
pub enum AttestationAuthError {
    #[error("attestation verification failed: {0}")]
    VerificationFailed(String),

    #[error("attestation evidence expired (age {age_secs}s, max {max_secs}s)")]
    EvidenceExpired { age_secs: u64, max_secs: u64 },

    #[error("platform not allowed: {0}")]
    PlatformNotAllowed(String),

    #[error("insufficient trust level: required {required:?}, got {actual:?}")]
    InsufficientTrust {
        required: TrustLevel,
        actual: TrustLevel,
    },

    #[error("token expired")]
    TokenExpired,

    #[error("invalid token: {0}")]
    InvalidToken(String),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for attestation-based authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationAuthConfig {
    /// Minimum trust level required for authentication.
    pub required_trust_level: TrustLevel,
    /// Maximum age of attestation evidence in seconds.
    pub max_evidence_age_secs: u64,
    /// Platforms allowed to authenticate (empty = all allowed).
    pub allowed_platforms: Vec<String>,
    /// Lifetime of issued attestation tokens in seconds.
    pub token_ttl_secs: u64,
}

impl Default for AttestationAuthConfig {
    fn default() -> Self {
        Self {
            required_trust_level: TrustLevel::HardwareAttested,
            max_evidence_age_secs: 300,
            allowed_platforms: Vec::new(),
            token_ttl_secs: 3600,
        }
    }
}

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// Short-lived token issued after successful attestation verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationToken {
    /// Unique token identifier.
    pub token_id: String,
    /// The platform that was attested.
    pub platform: String,
    /// Trust level established by the attestation.
    pub trust_level: TrustLevel,
    /// Token issuance timestamp (epoch millis).
    pub issued_at_ms: u64,
    /// Token expiry timestamp (epoch millis).
    pub expires_at_ms: u64,
    /// Hash of the original attestation evidence measurement.
    pub evidence_hash: String,
}

impl AttestationToken {
    /// Returns true if this token has expired relative to `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_at_ms
    }

    /// Returns the remaining lifetime in seconds, or 0 if expired.
    pub fn remaining_secs(&self, now_ms: u64) -> u64 {
        if now_ms >= self.expires_at_ms {
            0
        } else {
            (self.expires_at_ms - now_ms) / 1000
        }
    }
}

// ---------------------------------------------------------------------------
// Authenticator
// ---------------------------------------------------------------------------

/// Validates attestation evidence and issues short-lived tokens.
pub struct AttestationAuthenticator {
    config: AttestationAuthConfig,
    /// Counter for generating unique token IDs.
    next_token_id: std::sync::atomic::AtomicU64,
}

impl AttestationAuthenticator {
    /// Creates a new authenticator with the given configuration.
    pub fn new(config: AttestationAuthConfig) -> Self {
        Self {
            config,
            next_token_id: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// Authenticates a node by verifying its attestation evidence.
    ///
    /// Steps:
    /// 1. Check evidence freshness.
    /// 2. Check platform allowlist.
    /// 3. Evaluate evidence against attestation policy.
    /// 4. Check trust level meets minimum.
    /// 5. Issue attestation token.
    pub fn authenticate(
        &self,
        evidence: &AttestationEvidence,
        now_ms: u64,
    ) -> Result<AttestationToken, AttestationAuthError> {
        // 1. Check evidence freshness.
        let age_secs = now_ms.saturating_sub(evidence.timestamp_ms) / 1000;
        if age_secs > self.config.max_evidence_age_secs {
            return Err(AttestationAuthError::EvidenceExpired {
                age_secs,
                max_secs: self.config.max_evidence_age_secs,
            });
        }

        // 2. Check platform allowlist.
        if !self.config.allowed_platforms.is_empty()
            && !self.config.allowed_platforms.contains(&evidence.platform)
        {
            return Err(AttestationAuthError::PlatformNotAllowed(
                evidence.platform.clone(),
            ));
        }

        // 3. Evaluate evidence against policy.
        let result = evaluate_evidence(evidence);

        // 4. Check trust level.
        if result.trust_level < self.config.required_trust_level {
            return Err(AttestationAuthError::InsufficientTrust {
                required: self.config.required_trust_level,
                actual: result.trust_level,
            });
        }

        // 5. Issue token.
        let token_seq = self
            .next_token_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let evidence_hash = hex::encode(&evidence.measurement[..16]);

        Ok(AttestationToken {
            token_id: format!("att-{token_seq}"),
            platform: evidence.platform.clone(),
            trust_level: result.trust_level,
            issued_at_ms: now_ms,
            expires_at_ms: now_ms + self.config.token_ttl_secs * 1000,
            evidence_hash,
        })
    }

    /// Validates an existing attestation token.
    pub fn validate_token(
        &self,
        token: &AttestationToken,
        now_ms: u64,
    ) -> Result<(), AttestationAuthError> {
        if token.is_expired(now_ms) {
            return Err(AttestationAuthError::TokenExpired);
        }

        if token.trust_level < self.config.required_trust_level {
            return Err(AttestationAuthError::InsufficientTrust {
                required: self.config.required_trust_level,
                actual: token.trust_level,
            });
        }

        Ok(())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &AttestationAuthConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Trust upgrade
// ---------------------------------------------------------------------------

/// Upgrades an existing JWT token's trust level based on attestation.
///
/// Returns the effective trust level: the higher of the existing token
/// trust and the attestation-derived trust.
pub fn upgrade_trust_from_attestation(
    current_trust: TrustLevel,
    attestation_token: &AttestationToken,
) -> TrustLevel {
    if attestation_token.trust_level > current_trust {
        attestation_token.trust_level
    } else {
        current_trust
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Evaluates attestation evidence and produces an attestation result.
///
/// In production this would call into a hardware-specific verifier;
/// here we derive trust level from the platform string.
fn evaluate_evidence(evidence: &AttestationEvidence) -> AttestationResult {
    let trust_level = match evidence.platform.as_str() {
        "sgx" | "sev-snp" | "tdx" | "arm-cca" => TrustLevel::HardwareAttested,
        "tpm2" => TrustLevel::ThirdPartyAttested,
        "sw-tee" | "simulated" => TrustLevel::SelfAttested,
        _ => TrustLevel::Unverified,
    };

    let expiry_ms = evidence.timestamp_ms + 3600 * 1000;

    AttestationResult {
        verified: trust_level >= TrustLevel::SelfAttested,
        trust_level,
        expiry_ms,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now_ms() -> u64 {
        1_700_000_000_000
    }

    fn sample_evidence(platform: &str, timestamp_ms: u64) -> AttestationEvidence {
        AttestationEvidence {
            platform: platform.to_string(),
            measurement: [0xABu8; 48],
            report_data: vec![1, 2, 3],
            signature: vec![4, 5, 6],
            timestamp_ms,
        }
    }

    #[test]
    fn default_config() {
        let c = AttestationAuthConfig::default();
        assert_eq!(c.required_trust_level, TrustLevel::HardwareAttested);
        assert_eq!(c.max_evidence_age_secs, 300);
        assert!(c.allowed_platforms.is_empty());
        assert_eq!(c.token_ttl_secs, 3600);
    }

    #[test]
    fn authenticate_sgx_evidence() {
        let auth = AttestationAuthenticator::new(AttestationAuthConfig::default());
        let evidence = sample_evidence("sgx", now_ms() - 10_000);

        let token = auth.authenticate(&evidence, now_ms()).unwrap();
        assert_eq!(token.trust_level, TrustLevel::HardwareAttested);
        assert_eq!(token.platform, "sgx");
        assert!(!token.is_expired(now_ms()));
    }

    #[test]
    fn authenticate_sev_snp() {
        let auth = AttestationAuthenticator::new(AttestationAuthConfig::default());
        let evidence = sample_evidence("sev-snp", now_ms() - 5_000);

        let token = auth.authenticate(&evidence, now_ms()).unwrap();
        assert_eq!(token.trust_level, TrustLevel::HardwareAttested);
    }

    #[test]
    fn expired_evidence_rejected() {
        let auth = AttestationAuthenticator::new(AttestationAuthConfig::default());
        // Evidence is 600s old, max is 300s.
        let evidence = sample_evidence("sgx", now_ms() - 600_000);

        let result = auth.authenticate(&evidence, now_ms());
        assert!(matches!(
            result,
            Err(AttestationAuthError::EvidenceExpired { .. })
        ));
    }

    #[test]
    fn platform_not_allowed() {
        let config = AttestationAuthConfig {
            allowed_platforms: vec!["sgx".into(), "tdx".into()],
            ..Default::default()
        };
        let auth = AttestationAuthenticator::new(config);
        let evidence = sample_evidence("sev-snp", now_ms() - 10_000);

        let result = auth.authenticate(&evidence, now_ms());
        assert!(matches!(
            result,
            Err(AttestationAuthError::PlatformNotAllowed(_))
        ));
    }

    #[test]
    fn insufficient_trust_level() {
        let auth = AttestationAuthenticator::new(AttestationAuthConfig::default());
        // "simulated" platform → SelfAttested, but we require HardwareAttested.
        let evidence = sample_evidence("simulated", now_ms() - 10_000);

        let result = auth.authenticate(&evidence, now_ms());
        assert!(matches!(
            result,
            Err(AttestationAuthError::InsufficientTrust { .. })
        ));
    }

    #[test]
    fn token_expiry() {
        let token = AttestationToken {
            token_id: "att-1".into(),
            platform: "sgx".into(),
            trust_level: TrustLevel::HardwareAttested,
            issued_at_ms: now_ms(),
            expires_at_ms: now_ms() + 3_600_000,
            evidence_hash: "abcd".into(),
        };

        assert!(!token.is_expired(now_ms()));
        assert!(token.is_expired(now_ms() + 4_000_000));
        assert_eq!(token.remaining_secs(now_ms()), 3600);
        assert_eq!(token.remaining_secs(now_ms() + 4_000_000), 0);
    }

    #[test]
    fn validate_token_ok() {
        let auth = AttestationAuthenticator::new(AttestationAuthConfig::default());
        let evidence = sample_evidence("sgx", now_ms() - 10_000);
        let token = auth.authenticate(&evidence, now_ms()).unwrap();

        assert!(auth.validate_token(&token, now_ms()).is_ok());
    }

    #[test]
    fn validate_token_expired() {
        let auth = AttestationAuthenticator::new(AttestationAuthConfig::default());
        let evidence = sample_evidence("sgx", now_ms() - 10_000);
        let token = auth.authenticate(&evidence, now_ms()).unwrap();

        let result = auth.validate_token(&token, now_ms() + 4_000_000);
        assert!(matches!(result, Err(AttestationAuthError::TokenExpired)));
    }

    #[test]
    fn upgrade_trust() {
        let token = AttestationToken {
            token_id: "att-1".into(),
            platform: "sgx".into(),
            trust_level: TrustLevel::HardwareAttested,
            issued_at_ms: now_ms(),
            expires_at_ms: now_ms() + 3_600_000,
            evidence_hash: "abcd".into(),
        };

        let upgraded = upgrade_trust_from_attestation(TrustLevel::SelfAttested, &token);
        assert_eq!(upgraded, TrustLevel::HardwareAttested);

        // Already at a higher level → no change.
        let same = upgrade_trust_from_attestation(TrustLevel::FormallyVerified, &token);
        assert_eq!(same, TrustLevel::FormallyVerified);
    }

    #[test]
    fn error_display() {
        let e = AttestationAuthError::VerificationFailed("bad sig".into());
        assert_eq!(e.to_string(), "attestation verification failed: bad sig");

        let e = AttestationAuthError::EvidenceExpired {
            age_secs: 600,
            max_secs: 300,
        };
        assert_eq!(
            e.to_string(),
            "attestation evidence expired (age 600s, max 300s)"
        );

        let e = AttestationAuthError::PlatformNotAllowed("arm-v8".into());
        assert_eq!(e.to_string(), "platform not allowed: arm-v8");

        let e = AttestationAuthError::TokenExpired;
        assert_eq!(e.to_string(), "token expired");
    }

    #[test]
    fn evaluate_evidence_platforms() {
        let sgx = evaluate_evidence(&sample_evidence("sgx", now_ms()));
        assert_eq!(sgx.trust_level, TrustLevel::HardwareAttested);
        assert!(sgx.verified);

        let tpm = evaluate_evidence(&sample_evidence("tpm2", now_ms()));
        assert_eq!(tpm.trust_level, TrustLevel::ThirdPartyAttested);

        let unknown = evaluate_evidence(&sample_evidence("unknown-hw", now_ms()));
        assert_eq!(unknown.trust_level, TrustLevel::Unverified);
        assert!(!unknown.verified);
    }
}
