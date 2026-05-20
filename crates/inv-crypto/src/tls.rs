use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use std::sync::Arc;

use crate::certificates::{CaCertificate, CertError, NodeCertificate};

/// Install the aws-lc-rs crypto provider for rustls (FIPS 140-3).
/// Safe to call multiple times — only the first call takes effect.
pub fn install_crypto_provider() {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

/// Build a rustls ServerConfig for mTLS.
///
/// The server presents its node certificate and requires clients to present
/// a certificate signed by the organization CA.
pub fn server_config(
    node_cert: &NodeCertificate,
    ca_cert: &CaCertificate,
) -> Result<ServerConfig, CertError> {
    install_crypto_provider();
    let mut root_store = RootCertStore::empty();
    root_store
        .add(CertificateDer::from(ca_cert.cert_der.clone()))
        .map_err(|e| CertError::Verification(format!("failed to add CA to root store: {e}")))?;

    let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store))
        .build()
        .map_err(|e| CertError::Verification(format!("failed to build client verifier: {e}")))?;

    let cert_chain = vec![CertificateDer::from(node_cert.cert_der.clone())];
    let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(node_cert.key_der.clone()));

    let config = ServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
        .with_single_cert(cert_chain, private_key)
        .map_err(|e| CertError::Verification(format!("failed to build server config: {e}")))?;

    Ok(config)
}

/// Build a rustls ClientConfig for mTLS.
///
/// The client presents its node certificate and verifies the server's
/// certificate against the organization CA.
pub fn client_config(
    node_cert: &NodeCertificate,
    ca_cert: &CaCertificate,
) -> Result<ClientConfig, CertError> {
    install_crypto_provider();
    let mut root_store = RootCertStore::empty();
    root_store
        .add(CertificateDer::from(ca_cert.cert_der.clone()))
        .map_err(|e| CertError::Verification(format!("failed to add CA to root store: {e}")))?;

    let cert_chain = vec![CertificateDer::from(node_cert.cert_der.clone())];
    let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(node_cert.key_der.clone()));

    let config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_client_auth_cert(cert_chain, private_key)
        .map_err(|e| CertError::Verification(format!("failed to build client config: {e}")))?;

    Ok(config)
}

/// Build a rustls ServerConfig for server-only TLS (no client auth).
/// Used for public-facing endpoints like the API gateway.
pub fn server_config_no_client_auth(
    node_cert: &NodeCertificate,
) -> Result<ServerConfig, CertError> {
    install_crypto_provider();
    let cert_chain = vec![CertificateDer::from(node_cert.cert_der.clone())];
    let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(node_cert.key_der.clone()));

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, private_key)
        .map_err(|e| CertError::Verification(format!("failed to build server config: {e}")))?;

    Ok(config)
}

/// Compute the SHA-256 fingerprint of a CA certificate for pinning verification.
///
/// This fingerprint can be logged at startup and compared during inter-node
/// TLS handshakes to verify that the expected CA is in use. If a CA compromise
/// is detected (fingerprint mismatch), the node should refuse connections.
///
/// Regulatory basis: NIST SC-17 (PKI Certificates), ISO 27001 A.8.24 (Cryptography).
pub fn ca_fingerprint(ca_cert: &CaCertificate) -> String {
    use sha2::Digest;
    let hash = sha2::Sha256::digest(&ca_cert.cert_der);
    hex::encode(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificates::CertificateAuthority;

    #[test]
    fn build_mtls_configs() {
        let (ca, ca_cert) = CertificateAuthority::new_root("acme-corp").unwrap();

        let server_node = ca
            .sign_node_cert("inv_server_node", "acme-corp", "cloud")
            .unwrap();
        let client_node = ca
            .sign_node_cert("inv_client_node", "acme-corp", "workstation")
            .unwrap();

        let _server_cfg = server_config(&server_node, &ca_cert).unwrap();
        let _client_cfg = client_config(&client_node, &ca_cert).unwrap();
    }

    #[test]
    fn build_server_only_tls() {
        let (ca, _ca_cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let node = ca
            .sign_node_cert("inv_api_node", "acme-corp", "cloud")
            .unwrap();

        let _cfg = server_config_no_client_auth(&node).unwrap();
    }

    #[test]
    fn ca_fingerprint_is_deterministic() {
        let (_ca, ca_cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let fp1 = ca_fingerprint(&ca_cert);
        let fp2 = ca_fingerprint(&ca_cert);
        assert_eq!(fp1, fp2);
        // SHA-256 hex = 64 characters
        assert_eq!(fp1.len(), 64);
    }

    #[test]
    fn different_cas_have_different_fingerprints() {
        let (_, ca1) = CertificateAuthority::new_root("org-alpha").unwrap();
        let (_, ca2) = CertificateAuthority::new_root("org-beta").unwrap();
        assert_ne!(ca_fingerprint(&ca1), ca_fingerprint(&ca2));
    }
}
