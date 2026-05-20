//! PQ-hybrid TLS bridge.
//!
//! Integrates post-quantum key encapsulation and signature algorithms
//! into the existing TLS stack, providing server/client configuration
//! and negotiation results.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// PQ-hybrid TLS server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqTlsServerConfig {
    /// PEM-encoded classical certificate.
    pub classical_cert_pem: String,
    /// PEM-encoded classical private key.
    pub classical_key_pem: String,
    /// Post-quantum TLS configuration.
    pub pq_config: inv_pqcrypto::PqTlsConfig,
    /// Whether to require client certificates.
    pub require_client_auth: bool,
}

/// PQ-hybrid TLS client configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqTlsClientConfig {
    /// Post-quantum TLS configuration.
    pub pq_config: inv_pqcrypto::PqTlsConfig,
    /// Expected server name for SNI.
    pub server_name: String,
    /// Optional PEM-encoded client certificate.
    pub client_cert_pem: Option<String>,
    /// Optional PEM-encoded client private key.
    pub client_key_pem: Option<String>,
}

/// A PQ-hybrid certificate combining classical and post-quantum signatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCertificate {
    /// Certificate subject (e.g., node DID or hostname).
    pub subject: String,
    /// Certificate issuer.
    pub issuer: String,
    /// The PQ signature algorithm name.
    pub pq_signature_algorithm: String,
    /// Classical signature bytes.
    pub classical_signature: Vec<u8>,
    /// Post-quantum signature bytes.
    pub pq_signature: Vec<u8>,
    /// Validity start time.
    pub valid_from: DateTime<Utc>,
    /// Validity end time.
    pub valid_until: DateTime<Utc>,
}

impl PqCertificate {
    /// Check whether the certificate is currently valid.
    pub fn is_valid(&self, now: DateTime<Utc>) -> bool {
        now >= self.valid_from && now <= self.valid_until
    }

    /// Check whether both signatures are present.
    pub fn has_dual_signatures(&self) -> bool {
        !self.classical_signature.is_empty() && !self.pq_signature.is_empty()
    }
}

/// Result of PQ-hybrid TLS negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqTlsNegotiationResult {
    /// The selected cipher suite.
    pub selected_suite: inv_pqcrypto::PqCipherSuite,
    /// Whether PQ algorithms are enabled.
    pub pq_enabled: bool,
    /// Whether hybrid mode is active (classical + PQ).
    pub hybrid_mode: bool,
}

/// Errors from PQ TLS operations.
#[derive(Debug, thiserror::Error)]
pub enum PqTlsError {
    /// Certificate-related error.
    #[error("PQ TLS certificate error: {0}")]
    CertificateError(String),
    /// Key-related error.
    #[error("PQ TLS key error: {0}")]
    KeyError(String),
    /// Handshake negotiation failed.
    #[error("PQ TLS negotiation failed: {0}")]
    NegotiationFailed(String),
    /// Configuration error.
    #[error("PQ TLS config error: {0}")]
    ConfigError(String),
}

/// Create a PQ TLS server negotiation result from server config.
pub fn create_pq_server_config(
    config: &PqTlsServerConfig,
) -> Result<PqTlsNegotiationResult, PqTlsError> {
    if config.classical_cert_pem.is_empty() {
        return Err(PqTlsError::CertificateError(
            "classical cert PEM is empty".into(),
        ));
    }
    if config.classical_key_pem.is_empty() {
        return Err(PqTlsError::KeyError("classical key PEM is empty".into()));
    }

    let pq_enabled = config.pq_config.supports_pq_kem();
    let hybrid_mode = matches!(
        config.pq_config.kem_algorithm,
        inv_pqcrypto::KemAlgorithm::X25519MlKem768Hybrid
    );

    let selected_suite = if pq_enabled {
        inv_pqcrypto::PqCipherSuite::X25519MlKem768
    } else {
        inv_pqcrypto::PqCipherSuite::ClassicalOnly
    };

    Ok(PqTlsNegotiationResult {
        selected_suite,
        pq_enabled,
        hybrid_mode,
    })
}

/// Create a PQ TLS client negotiation result from client config.
pub fn create_pq_client_config(
    config: &PqTlsClientConfig,
) -> Result<PqTlsNegotiationResult, PqTlsError> {
    if config.server_name.is_empty() {
        return Err(PqTlsError::ConfigError("server name is empty".into()));
    }

    let pq_enabled = config.pq_config.supports_pq_kem();
    let hybrid_mode = matches!(
        config.pq_config.kem_algorithm,
        inv_pqcrypto::KemAlgorithm::X25519MlKem768Hybrid
    );

    let selected_suite = if pq_enabled {
        inv_pqcrypto::PqCipherSuite::X25519MlKem768
    } else {
        inv_pqcrypto::PqCipherSuite::ClassicalOnly
    };

    Ok(PqTlsNegotiationResult {
        selected_suite,
        pq_enabled,
        hybrid_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hybrid_server_config() -> PqTlsServerConfig {
        PqTlsServerConfig {
            classical_cert_pem: "-----BEGIN CERTIFICATE-----\nMIIB...\n-----END CERTIFICATE-----"
                .into(),
            classical_key_pem: "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----"
                .into(),
            pq_config: inv_pqcrypto::PqTlsConfig::hybrid_default(),
            require_client_auth: false,
        }
    }

    fn hybrid_client_config() -> PqTlsClientConfig {
        PqTlsClientConfig {
            pq_config: inv_pqcrypto::PqTlsConfig::hybrid_default(),
            server_name: "node.invisible.dev".into(),
            client_cert_pem: None,
            client_key_pem: None,
        }
    }

    #[test]
    fn server_config_hybrid() {
        let result = create_pq_server_config(&hybrid_server_config()).unwrap();
        assert!(result.pq_enabled);
        assert!(result.hybrid_mode);
        assert_eq!(
            result.selected_suite,
            inv_pqcrypto::PqCipherSuite::X25519MlKem768
        );
    }

    #[test]
    fn server_config_classical_only() {
        let mut config = hybrid_server_config();
        config.pq_config = inv_pqcrypto::PqTlsConfig::new(
            inv_pqcrypto::KemAlgorithm::ClassicalX25519,
            inv_pqcrypto::SigAlgorithm::ClassicalEd25519,
            true,
        );
        let result = create_pq_server_config(&config).unwrap();
        assert!(!result.pq_enabled);
        assert!(!result.hybrid_mode);
        assert_eq!(
            result.selected_suite,
            inv_pqcrypto::PqCipherSuite::ClassicalOnly
        );
    }

    #[test]
    fn server_config_empty_cert_fails() {
        let mut config = hybrid_server_config();
        config.classical_cert_pem = String::new();
        let err = create_pq_server_config(&config).unwrap_err();
        assert!(matches!(err, PqTlsError::CertificateError(_)));
    }

    #[test]
    fn server_config_empty_key_fails() {
        let mut config = hybrid_server_config();
        config.classical_key_pem = String::new();
        let err = create_pq_server_config(&config).unwrap_err();
        assert!(matches!(err, PqTlsError::KeyError(_)));
    }

    #[test]
    fn client_config_hybrid() {
        let result = create_pq_client_config(&hybrid_client_config()).unwrap();
        assert!(result.pq_enabled);
        assert!(result.hybrid_mode);
    }

    #[test]
    fn client_config_empty_server_name_fails() {
        let mut config = hybrid_client_config();
        config.server_name = String::new();
        let err = create_pq_client_config(&config).unwrap_err();
        assert!(matches!(err, PqTlsError::ConfigError(_)));
    }

    #[test]
    fn pq_certificate_validity() {
        let now = Utc::now();
        let cert = PqCertificate {
            subject: "node1.invisible.dev".into(),
            issuer: "ca.invisible.dev".into(),
            pq_signature_algorithm: "ML-DSA-65".into(),
            classical_signature: vec![0xAA; 64],
            pq_signature: vec![0xBB; 3309],
            valid_from: now - chrono::Duration::hours(1),
            valid_until: now + chrono::Duration::hours(23),
        };
        assert!(cert.is_valid(now));
        assert!(cert.has_dual_signatures());
    }

    #[test]
    fn pq_certificate_expired() {
        let now = Utc::now();
        let cert = PqCertificate {
            subject: "node1".into(),
            issuer: "ca".into(),
            pq_signature_algorithm: "ML-DSA-65".into(),
            classical_signature: vec![0xAA; 64],
            pq_signature: vec![0xBB; 3309],
            valid_from: now - chrono::Duration::hours(48),
            valid_until: now - chrono::Duration::hours(24),
        };
        assert!(!cert.is_valid(now));
    }

    #[test]
    fn pq_certificate_missing_pq_sig() {
        let now = Utc::now();
        let cert = PqCertificate {
            subject: "node1".into(),
            issuer: "ca".into(),
            pq_signature_algorithm: "ML-DSA-65".into(),
            classical_signature: vec![0xAA; 64],
            pq_signature: vec![],
            valid_from: now - chrono::Duration::hours(1),
            valid_until: now + chrono::Duration::hours(23),
        };
        assert!(!cert.has_dual_signatures());
    }

    #[test]
    fn negotiation_result_serialization() {
        let result = PqTlsNegotiationResult {
            selected_suite: inv_pqcrypto::PqCipherSuite::X25519MlKem768,
            pq_enabled: true,
            hybrid_mode: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: PqTlsNegotiationResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.pq_enabled);
        assert!(parsed.hybrid_mode);
    }
}
