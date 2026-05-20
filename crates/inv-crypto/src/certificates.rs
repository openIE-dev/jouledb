use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};

use x509_parser::prelude::*;

/// A self-signed root CA certificate for an organization.
#[derive(Debug, Clone)]
pub struct CaCertificate {
    /// DER-encoded certificate.
    pub cert_der: Vec<u8>,
    /// PEM-encoded certificate (for file storage).
    pub cert_pem: String,
}

/// A signed node certificate, issued by a CA.
#[derive(Debug, Clone)]
pub struct NodeCertificate {
    /// DER-encoded certificate.
    pub cert_der: Vec<u8>,
    /// PEM-encoded certificate.
    pub cert_pem: String,
    /// DER-encoded private key.
    pub key_der: Vec<u8>,
    /// PEM-encoded private key.
    pub key_pem: String,
}

/// Certificate authority for issuing and signing certificates.
///
/// Hierarchy: Root CA -> (optional Intermediate CA) -> Node Certs -> Workload Certs
pub struct CertificateAuthority {
    key_pair: KeyPair,
    ca_cert_der: Vec<u8>,
    /// Stored params so we can reconstruct an Issuer without re-parsing DER.
    ca_params: CertificateParams,
}

impl CertificateAuthority {
    /// Create a new root CA for an organization.
    pub fn new_root(org_name: &str) -> Result<(Self, CaCertificate), CertError> {
        let key_pair = KeyPair::generate_for(&rcgen::PKCS_ED25519)
            .map_err(|e| CertError::Generation(format!("failed to generate CA keypair: {e}")))?;

        let mut params = CertificateParams::default();
        params
            .distinguished_name
            .push(DnType::OrganizationName, org_name);
        params
            .distinguished_name
            .push(DnType::CommonName, format!("{org_name} Root CA"));
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];

        let cert = params
            .self_signed(&key_pair)
            .map_err(|e| CertError::Generation(format!("failed to self-sign root CA: {e}")))?;

        let cert_der = cert.der().to_vec();
        let cert_pem = cert.pem();

        Ok((
            Self {
                key_pair,
                ca_cert_der: cert_der.clone(),
                ca_params: params,
            },
            CaCertificate { cert_der, cert_pem },
        ))
    }

    /// Build an Issuer from the stored CA params and key pair.
    fn issuer(&self) -> Issuer<'_, &KeyPair> {
        Issuer::new(self.ca_params.clone(), &self.key_pair)
    }

    /// Create an intermediate CA signed by this CA.
    pub fn new_intermediate(
        &self,
        region: &str,
        org_name: &str,
    ) -> Result<(Self, CaCertificate), CertError> {
        let inter_key_pair = KeyPair::generate_for(&rcgen::PKCS_ED25519).map_err(|e| {
            CertError::Generation(format!("failed to generate intermediate keypair: {e}"))
        })?;

        let mut params = CertificateParams::default();
        params
            .distinguished_name
            .push(DnType::OrganizationName, org_name);
        params.distinguished_name.push(
            DnType::CommonName,
            format!("{org_name} Intermediate CA ({region})"),
        );
        params.is_ca = IsCa::Ca(BasicConstraints::Constrained(0));
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];

        let issuer = self.issuer();

        let cert = params
            .signed_by(&inter_key_pair, &issuer)
            .map_err(|e| CertError::Generation(format!("failed to sign intermediate CA: {e}")))?;

        let cert_der = cert.der().to_vec();
        let cert_pem = cert.pem();

        Ok((
            Self {
                key_pair: inter_key_pair,
                ca_cert_der: cert_der.clone(),
                ca_params: params,
            },
            CaCertificate { cert_der, cert_pem },
        ))
    }

    /// Sign a node certificate with this CA.
    /// Generates a fresh Ed25519 keypair for the TLS certificate.
    pub fn sign_node_cert(
        &self,
        node_id: &str,
        org_name: &str,
        _node_class: &str,
    ) -> Result<NodeCertificate, CertError> {
        let node_key_pair = KeyPair::generate_for(&rcgen::PKCS_ED25519)
            .map_err(|e| CertError::Generation(format!("failed to generate node keypair: {e}")))?;

        let mut params = CertificateParams::default();
        params
            .distinguished_name
            .push(DnType::OrganizationName, org_name);
        params.distinguished_name.push(DnType::CommonName, node_id);

        // Encode node_id as a DNS SAN for TLS validation
        params.subject_alt_names.push(rcgen::SanType::DnsName(
            node_id
                .to_string()
                .try_into()
                .map_err(|e| CertError::Generation(format!("invalid node_id for SAN: {e}")))?,
        ));

        params.is_ca = IsCa::NoCa;
        params.key_usages = vec![
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::KeyEncipherment,
        ];
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];

        let issuer = self.issuer();

        let cert = params
            .signed_by(&node_key_pair, &issuer)
            .map_err(|e| CertError::Generation(format!("failed to sign node cert: {e}")))?;

        let cert_der = cert.der().to_vec();
        let cert_pem = cert.pem();
        let key_der = node_key_pair.serialize_der();
        let key_pem = node_key_pair.serialize_pem();

        Ok(NodeCertificate {
            cert_der,
            cert_pem,
            key_der,
            key_pem,
        })
    }

    /// The CA's public key in PEM.
    pub fn public_key_pem(&self) -> String {
        self.key_pair.public_key_pem()
    }

    /// The CA cert in DER form.
    pub fn ca_cert_der(&self) -> &[u8] {
        &self.ca_cert_der
    }

    /// Serialize the CA's private key to PKCS8 PEM format.
    pub fn key_pem(&self) -> String {
        self.key_pair.serialize_pem()
    }

    /// Reconstruct a CA from PEM-encoded certificate and private key.
    pub fn from_pem(ca_cert_pem: &str, key_pem: &str) -> Result<Self, CertError> {
        let ca_cert_der = cert_pem_to_der(ca_cert_pem)?;
        let key_pair = KeyPair::from_pem(key_pem)
            .map_err(|e| CertError::Parse(format!("failed to parse CA key PEM: {e}")))?;
        // Reconstruct minimal CA params for Issuer creation
        let ca_params = CertificateParams::default();
        Ok(Self {
            key_pair,
            ca_cert_der,
            ca_params,
        })
    }
}

impl CaCertificate {
    /// Reconstruct from a PEM-encoded certificate string.
    pub fn from_pem(cert_pem: &str) -> Result<Self, CertError> {
        let cert_der = cert_pem_to_der(cert_pem)?;
        Ok(Self {
            cert_der,
            cert_pem: cert_pem.to_string(),
        })
    }
}

impl NodeCertificate {
    /// Reconstruct from PEM-encoded certificate and private key strings.
    pub fn from_pem(cert_pem: &str, key_pem: &str) -> Result<Self, CertError> {
        let cert_der = cert_pem_to_der(cert_pem)?;
        let key_pair = KeyPair::from_pem(key_pem)
            .map_err(|e| CertError::Parse(format!("failed to parse node key PEM: {e}")))?;
        let key_der = key_pair.serialize_der();
        Ok(Self {
            cert_der,
            cert_pem: cert_pem.to_string(),
            key_der,
            key_pem: key_pem.to_string(),
        })
    }
}

/// Parse a PEM-encoded certificate and return the DER bytes.
fn cert_pem_to_der(pem: &str) -> Result<Vec<u8>, CertError> {
    let mut reader = std::io::BufReader::new(pem.as_bytes());
    let certs: Result<Vec<_>, _> = rustls_pemfile::certs(&mut reader).collect();
    let certs =
        certs.map_err(|e| CertError::Parse(format!("failed to parse certificate PEM: {e}")))?;
    let cert = certs
        .into_iter()
        .next()
        .ok_or_else(|| CertError::Parse("no certificate found in PEM".into()))?;
    Ok(cert.to_vec())
}

impl std::fmt::Debug for CertificateAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CertificateAuthority")
            .field("key", &"[REDACTED]")
            .finish()
    }
}

/// Parse and inspect an X.509 certificate from DER bytes.
pub fn parse_certificate(der: &[u8]) -> Result<CertInfo, CertError> {
    let (_, cert) = X509Certificate::from_der(der)
        .map_err(|e| CertError::Parse(format!("invalid X.509 DER: {e}")))?;

    let subject = cert.subject().to_string();
    let issuer = cert.issuer().to_string();
    let is_ca = cert.is_ca();
    let serial = hex::encode(cert.serial.to_bytes_be());

    Ok(CertInfo {
        subject,
        issuer,
        serial,
        is_ca,
    })
}

/// Parsed certificate metadata.
#[derive(Debug, Clone)]
pub struct CertInfo {
    pub subject: String,
    pub issuer: String,
    pub serial: String,
    pub is_ca: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum CertError {
    #[error("certificate generation error: {0}")]
    Generation(String),
    #[error("certificate parse error: {0}")]
    Parse(String),
    #[error("certificate verification error: {0}")]
    Verification(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_root_ca() {
        let (_ca, cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        assert!(!cert.cert_pem.is_empty());
        assert!(!cert.cert_der.is_empty());

        let info = parse_certificate(&cert.cert_der).unwrap();
        assert!(info.subject.contains("acme-corp"));
        assert!(info.is_ca);
    }

    #[test]
    fn sign_node_certificate() {
        let (ca, _ca_cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let node_kp = crate::identity::NodeKeypair::generate();
        let node_id = node_kp.node_id().to_string_repr();

        let node_cert = ca
            .sign_node_cert(&node_id, "acme-corp", "workstation")
            .unwrap();

        assert!(!node_cert.cert_pem.is_empty());
        assert!(!node_cert.key_pem.is_empty());

        let info = parse_certificate(&node_cert.cert_der).unwrap();
        assert!(!info.is_ca);
    }

    #[test]
    fn intermediate_ca() {
        let (root_ca, _root_cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let (_inter_ca, inter_cert) = root_ca.new_intermediate("us-east", "acme-corp").unwrap();

        let info = parse_certificate(&inter_cert.cert_der).unwrap();
        assert!(info.is_ca);
        assert!(info.subject.contains("Intermediate"));
    }

    #[test]
    fn full_hierarchy() {
        let (root_ca, _root_cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let (inter_ca, _inter_cert) = root_ca.new_intermediate("us-east", "acme-corp").unwrap();
        let node_cert = inter_ca
            .sign_node_cert("inv_test_node_123", "acme-corp", "workstation")
            .unwrap();

        let info = parse_certificate(&node_cert.cert_der).unwrap();
        assert!(!info.is_ca);
    }

    #[test]
    fn ca_roundtrip_through_pem() {
        // Create a root CA
        let (root_ca, root_cert) = CertificateAuthority::new_root("acme-corp").unwrap();

        // Serialize to PEM
        let key_pem = root_ca.key_pem();
        assert!(key_pem.contains("BEGIN PRIVATE KEY"));

        // Reconstruct from PEM
        let restored_ca = CertificateAuthority::from_pem(&root_cert.cert_pem, &key_pem).unwrap();

        // Verify the restored CA can sign node certs
        let node_cert = restored_ca
            .sign_node_cert("inv_test_node_456", "acme-corp", "cloud")
            .unwrap();
        let info = parse_certificate(&node_cert.cert_der).unwrap();
        assert!(!info.is_ca);
        assert!(info.subject.contains("inv_test_node_456"));
    }

    #[test]
    fn intermediate_ca_roundtrip_through_pem() {
        let (root_ca, _root_cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let (inter_ca, inter_cert) = root_ca.new_intermediate("us-east", "acme-corp").unwrap();

        // Serialize and reconstruct the intermediate CA
        let key_pem = inter_ca.key_pem();
        let restored_ca = CertificateAuthority::from_pem(&inter_cert.cert_pem, &key_pem).unwrap();

        // Verify it can still sign node certs
        let node_cert = restored_ca
            .sign_node_cert("inv_restored_node", "acme-corp", "edge")
            .unwrap();
        assert!(!node_cert.cert_pem.is_empty());
        assert!(!node_cert.key_pem.is_empty());
    }

    #[test]
    fn ca_cert_roundtrip_through_pem() {
        let (_ca, cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let restored = CaCertificate::from_pem(&cert.cert_pem).unwrap();
        assert_eq!(restored.cert_der, cert.cert_der);
        assert_eq!(restored.cert_pem, cert.cert_pem);
    }

    #[test]
    fn node_cert_roundtrip_through_pem() {
        let (ca, _cert) = CertificateAuthority::new_root("acme-corp").unwrap();
        let node_cert = ca
            .sign_node_cert("inv_test_node_789", "acme-corp", "workstation")
            .unwrap();

        let restored = NodeCertificate::from_pem(&node_cert.cert_pem, &node_cert.key_pem).unwrap();
        assert_eq!(restored.cert_der, node_cert.cert_der);
        assert_eq!(restored.key_der, node_cert.key_der);
    }
}
