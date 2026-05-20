//! PQ Certificates — X.509-like certificate structure with post-quantum
//! public key embedding, signature verification, validity periods,
//! subject/issuer fields, and certificate chain construction.
//!
//! Pure-Rust certificate primitives for post-quantum PKI. Certificates
//! carry a PQ public key, a signature from the issuer, and metadata
//! (validity window, subject, issuer). Chain validation walks from leaf
//! to a trusted root, verifying each link.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PqCertError {
    Expired(String),
    NotYetValid(String),
    SignatureInvalid(String),
    ChainBroken(String),
    MissingField(String),
    KeyMismatch(String),
    SerializationFailed(String),
}

impl fmt::Display for PqCertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expired(s) => write!(f, "certificate expired: {s}"),
            Self::NotYetValid(s) => write!(f, "certificate not yet valid: {s}"),
            Self::SignatureInvalid(s) => write!(f, "signature invalid: {s}"),
            Self::ChainBroken(s) => write!(f, "chain broken: {s}"),
            Self::MissingField(s) => write!(f, "missing field: {s}"),
            Self::KeyMismatch(s) => write!(f, "key mismatch: {s}"),
            Self::SerializationFailed(s) => write!(f, "serialization failed: {s}"),
        }
    }
}

impl std::error::Error for PqCertError {}

// ── PQ Algorithm Identifier ─────────────────────────────────────

/// Post-quantum algorithm identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PqAlgorithm {
    /// ML-KEM-768 (lattice-based KEM).
    MlKem768,
    /// ML-KEM-1024 (lattice-based KEM, higher security).
    MlKem1024,
    /// ML-DSA-65 (lattice-based signature).
    MlDsa65,
    /// ML-DSA-87 (lattice-based signature, higher security).
    MlDsa87,
    /// SLH-DSA-128s (hash-based signature, small).
    SlhDsa128s,
    /// SLH-DSA-256f (hash-based signature, fast).
    SlhDsa256f,
}

impl PqAlgorithm {
    /// Nominal public key size in bytes.
    pub fn public_key_size(&self) -> usize {
        match self {
            Self::MlKem768 => 1184,
            Self::MlKem1024 => 1568,
            Self::MlDsa65 => 1952,
            Self::MlDsa87 => 2592,
            Self::SlhDsa128s => 32,
            Self::SlhDsa256f => 64,
        }
    }

    /// Whether this algorithm is a signature scheme.
    pub fn is_signature(&self) -> bool {
        matches!(self, Self::MlDsa65 | Self::MlDsa87 | Self::SlhDsa128s | Self::SlhDsa256f)
    }

    /// Whether this algorithm is a KEM.
    pub fn is_kem(&self) -> bool {
        matches!(self, Self::MlKem768 | Self::MlKem1024)
    }
}

impl fmt::Display for PqAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MlKem768 => write!(f, "ML-KEM-768"),
            Self::MlKem1024 => write!(f, "ML-KEM-1024"),
            Self::MlDsa65 => write!(f, "ML-DSA-65"),
            Self::MlDsa87 => write!(f, "ML-DSA-87"),
            Self::SlhDsa128s => write!(f, "SLH-DSA-128s"),
            Self::SlhDsa256f => write!(f, "SLH-DSA-256f"),
        }
    }
}

// ── Validity Period ─────────────────────────────────────────────

/// Certificate validity window (epoch seconds).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Validity {
    pub not_before: u64,
    pub not_after: u64,
}

impl Validity {
    pub fn new(not_before: u64, not_after: u64) -> Self {
        Self { not_before, not_after }
    }

    /// Check whether the given timestamp falls within the validity window.
    pub fn is_valid_at(&self, timestamp: u64) -> bool {
        timestamp >= self.not_before && timestamp <= self.not_after
    }

    /// Duration of the validity window in seconds.
    pub fn duration_secs(&self) -> u64 {
        self.not_after.saturating_sub(self.not_before)
    }
}

impl fmt::Display for Validity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{}]", self.not_before, self.not_after)
    }
}

// ── Distinguished Name ──────────────────────────────────────────

/// Simplified distinguished name with common fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistinguishedName {
    pub common_name: String,
    pub organization: Option<String>,
    pub country: Option<String>,
    pub fields: HashMap<String, String>,
}

impl DistinguishedName {
    pub fn new(common_name: &str) -> Self {
        Self {
            common_name: common_name.to_string(),
            organization: None,
            country: None,
            fields: HashMap::new(),
        }
    }

    pub fn with_organization(mut self, org: &str) -> Self {
        self.organization = Some(org.to_string());
        self
    }

    pub fn with_country(mut self, c: &str) -> Self {
        self.country = Some(c.to_string());
        self
    }

    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    /// Produce a canonical byte representation for hashing.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(self.common_name.as_bytes());
        if let Some(ref o) = self.organization {
            out.push(b'|');
            out.extend_from_slice(o.as_bytes());
        }
        if let Some(ref c) = self.country {
            out.push(b'|');
            out.extend_from_slice(c.as_bytes());
        }
        out
    }
}

impl fmt::Display for DistinguishedName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CN={}", self.common_name)?;
        if let Some(ref o) = self.organization {
            write!(f, ", O={o}")?;
        }
        if let Some(ref c) = self.country {
            write!(f, ", C={c}")?;
        }
        Ok(())
    }
}

// ── PQ Public Key ───────────────────────────────────────────────

/// A post-quantum public key (opaque byte blob plus algorithm tag).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqPublicKey {
    pub algorithm: PqAlgorithm,
    pub key_bytes: Vec<u8>,
}

impl PqPublicKey {
    pub fn new(algorithm: PqAlgorithm, key_bytes: Vec<u8>) -> Self {
        Self { algorithm, key_bytes }
    }

    /// Fingerprint: simple 8-byte hash of the key material.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in &self.key_bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Validate that the key length matches the algorithm's expected size.
    pub fn validate_length(&self) -> Result<(), PqCertError> {
        let expected = self.algorithm.public_key_size();
        if self.key_bytes.len() != expected {
            return Err(PqCertError::KeyMismatch(format!(
                "expected {} bytes for {}, got {}",
                expected,
                self.algorithm,
                self.key_bytes.len()
            )));
        }
        Ok(())
    }
}

impl fmt::Display for PqPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqPubKey({}, {} bytes, fp={:016x})", self.algorithm, self.key_bytes.len(), self.fingerprint())
    }
}

// ── Signature ───────────────────────────────────────────────────

/// A PQ digital signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqSignature {
    pub algorithm: PqAlgorithm,
    pub sig_bytes: Vec<u8>,
}

impl PqSignature {
    pub fn new(algorithm: PqAlgorithm, sig_bytes: Vec<u8>) -> Self {
        Self { algorithm, sig_bytes }
    }
}

impl fmt::Display for PqSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PqSig({}, {} bytes)", self.algorithm, self.sig_bytes.len())
    }
}

// ── Certificate ─────────────────────────────────────────────────

/// X.509-like certificate carrying a PQ public key.
#[derive(Debug, Clone)]
pub struct PqCertificate {
    pub serial: u64,
    pub subject: DistinguishedName,
    pub issuer: DistinguishedName,
    pub validity: Validity,
    pub public_key: PqPublicKey,
    pub signature: PqSignature,
    pub is_ca: bool,
    pub max_path_length: Option<u32>,
    pub extensions: HashMap<String, Vec<u8>>,
}

impl PqCertificate {
    /// Compute the TBS (to-be-signed) digest for this certificate.
    pub fn tbs_digest(&self) -> u64 {
        let mut h: u64 = 0x6c62272e07bb0142;
        h = h.wrapping_mul(31).wrapping_add(self.serial);
        for &b in &self.subject.canonical_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &b in &self.issuer.canonical_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h = h.wrapping_mul(31).wrapping_add(self.validity.not_before);
        h = h.wrapping_mul(31).wrapping_add(self.validity.not_after);
        for &b in &self.public_key.key_bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Verify the certificate's signature against the issuer's public key.
    /// Uses a simplified verification: recomputes TBS digest and checks that
    /// the signature bytes encode the same digest XORed with the issuer key hash.
    pub fn verify_signature(&self, issuer_key: &PqPublicKey) -> Result<(), PqCertError> {
        if !self.signature.algorithm.is_signature() {
            return Err(PqCertError::SignatureInvalid(
                "algorithm is not a signature scheme".into(),
            ));
        }
        let tbs = self.tbs_digest();
        let key_hash = issuer_key.fingerprint();
        let expected = tbs ^ key_hash;
        let sig_val = bytes_to_u64(&self.signature.sig_bytes);
        if sig_val != expected {
            return Err(PqCertError::SignatureInvalid(format!(
                "expected {expected:016x}, got {sig_val:016x}"
            )));
        }
        Ok(())
    }

    /// Check validity at a given timestamp.
    pub fn check_validity(&self, now: u64) -> Result<(), PqCertError> {
        if now < self.validity.not_before {
            return Err(PqCertError::NotYetValid(format!(
                "valid from {}, now {now}",
                self.validity.not_before
            )));
        }
        if now > self.validity.not_after {
            return Err(PqCertError::Expired(format!(
                "expired at {}, now {now}",
                self.validity.not_after
            )));
        }
        Ok(())
    }

    /// Sign a child certificate's TBS with this certificate's (simulated) private key.
    pub fn sign_tbs(&self, tbs_digest: u64) -> PqSignature {
        let key_hash = self.public_key.fingerprint();
        let sig_val = tbs_digest ^ key_hash;
        PqSignature {
            algorithm: self.signature.algorithm,
            sig_bytes: u64_to_bytes(sig_val),
        }
    }

    /// Whether this certificate is self-signed (subject == issuer).
    pub fn is_self_signed(&self) -> bool {
        self.subject == self.issuer
    }
}

impl fmt::Display for PqCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PqCert(serial={}, subject={}, issuer={}, ca={})",
            self.serial, self.subject, self.issuer, self.is_ca
        )
    }
}

// ── Certificate Chain ───────────────────────────────────────────

/// An ordered chain of certificates from leaf to root.
#[derive(Debug, Clone)]
pub struct CertChain {
    pub certs: Vec<PqCertificate>,
}

impl CertChain {
    pub fn new(certs: Vec<PqCertificate>) -> Self {
        Self { certs }
    }

    /// Number of certificates in the chain.
    pub fn len(&self) -> usize {
        self.certs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.certs.is_empty()
    }

    /// Leaf certificate (first in chain).
    pub fn leaf(&self) -> Option<&PqCertificate> {
        self.certs.first()
    }

    /// Root certificate (last in chain).
    pub fn root(&self) -> Option<&PqCertificate> {
        self.certs.last()
    }

    /// Validate the chain: each cert is signed by the next, and the root is self-signed.
    pub fn validate(&self, now: u64) -> Result<(), PqCertError> {
        if self.certs.is_empty() {
            return Err(PqCertError::ChainBroken("empty chain".into()));
        }
        for cert in &self.certs {
            cert.check_validity(now)?;
        }
        for i in 0..self.certs.len() - 1 {
            let child = &self.certs[i];
            let parent = &self.certs[i + 1];
            if child.issuer != parent.subject {
                return Err(PqCertError::ChainBroken(format!(
                    "cert {} issuer != cert {} subject",
                    i,
                    i + 1
                )));
            }
            if !parent.is_ca {
                return Err(PqCertError::ChainBroken(format!(
                    "cert {} is not a CA",
                    i + 1
                )));
            }
            child.verify_signature(&parent.public_key)?;
        }
        let root = self.certs.last().unwrap();
        if !root.is_self_signed() {
            return Err(PqCertError::ChainBroken("root is not self-signed".into()));
        }
        Ok(())
    }
}

impl fmt::Display for CertChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CertChain({} certs)", self.certs.len())
    }
}

// ── Config Builder ──────────────────────────────────────────────

/// Configuration for PQ certificate generation.
#[derive(Debug, Clone)]
pub struct PqCertConfig {
    pub algorithm: PqAlgorithm,
    pub validity_secs: u64,
    pub is_ca: bool,
    pub max_path_length: Option<u32>,
    pub key_size_override: Option<usize>,
}

impl PqCertConfig {
    pub fn new() -> Self {
        Self {
            algorithm: PqAlgorithm::MlDsa65,
            validity_secs: 365 * 24 * 3600,
            is_ca: false,
            max_path_length: None,
            key_size_override: None,
        }
    }

    pub fn with_algorithm(mut self, alg: PqAlgorithm) -> Self {
        self.algorithm = alg;
        self
    }

    pub fn with_validity_secs(mut self, secs: u64) -> Self {
        self.validity_secs = secs;
        self
    }

    pub fn with_ca(mut self, is_ca: bool) -> Self {
        self.is_ca = is_ca;
        self
    }

    pub fn with_max_path_length(mut self, len: u32) -> Self {
        self.max_path_length = Some(len);
        self
    }

    pub fn with_key_size_override(mut self, size: usize) -> Self {
        self.key_size_override = Some(size);
        self
    }

    /// Generate a certificate from this config.
    pub fn build_certificate(
        &self,
        serial: u64,
        subject: DistinguishedName,
        issuer: DistinguishedName,
        not_before: u64,
    ) -> PqCertificate {
        let key_size = self.key_size_override.unwrap_or(self.algorithm.public_key_size());
        let key_bytes: Vec<u8> = (0..key_size).map(|i| ((serial.wrapping_mul(31).wrapping_add(i as u64)) & 0xff) as u8).collect();
        let public_key = PqPublicKey::new(self.algorithm, key_bytes);

        let validity = Validity::new(not_before, not_before + self.validity_secs);

        let mut cert = PqCertificate {
            serial,
            subject,
            issuer,
            validity,
            public_key,
            signature: PqSignature::new(self.algorithm, vec![0u8; 8]),
            is_ca: self.is_ca,
            max_path_length: self.max_path_length,
            extensions: HashMap::new(),
        };

        // Self-sign: signature = tbs ^ own key fingerprint
        let tbs = cert.tbs_digest();
        let fp = cert.public_key.fingerprint();
        cert.signature = PqSignature::new(self.algorithm, u64_to_bytes(tbs ^ fp));
        cert
    }
}

impl Default for PqCertConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PqCertConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PqCertConfig(alg={}, validity={}s, ca={})",
            self.algorithm, self.validity_secs, self.is_ca
        )
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn u64_to_bytes(val: u64) -> Vec<u8> {
    val.to_le_bytes().to_vec()
}

fn bytes_to_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(buf)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ca() -> PqCertificate {
        PqCertConfig::new()
            .with_algorithm(PqAlgorithm::MlDsa65)
            .with_ca(true)
            .with_validity_secs(3600 * 24 * 365)
            .build_certificate(1, DistinguishedName::new("RootCA"), DistinguishedName::new("RootCA"), 1000)
    }

    #[test]
    fn test_algorithm_properties() {
        assert!(PqAlgorithm::MlDsa65.is_signature());
        assert!(!PqAlgorithm::MlDsa65.is_kem());
        assert!(PqAlgorithm::MlKem768.is_kem());
        assert!(!PqAlgorithm::MlKem768.is_signature());
    }

    #[test]
    fn test_algorithm_key_sizes() {
        assert_eq!(PqAlgorithm::MlKem768.public_key_size(), 1184);
        assert_eq!(PqAlgorithm::MlDsa87.public_key_size(), 2592);
        assert_eq!(PqAlgorithm::SlhDsa128s.public_key_size(), 32);
    }

    #[test]
    fn test_validity_window() {
        let v = Validity::new(100, 200);
        assert!(v.is_valid_at(100));
        assert!(v.is_valid_at(150));
        assert!(v.is_valid_at(200));
        assert!(!v.is_valid_at(99));
        assert!(!v.is_valid_at(201));
        assert_eq!(v.duration_secs(), 100);
    }

    #[test]
    fn test_distinguished_name_builder() {
        let dn = DistinguishedName::new("Test")
            .with_organization("Org")
            .with_country("US");
        assert_eq!(dn.common_name, "Test");
        assert_eq!(dn.organization.as_deref(), Some("Org"));
        let s = format!("{dn}");
        assert!(s.contains("CN=Test"));
        assert!(s.contains("O=Org"));
        assert!(s.contains("C=US"));
    }

    #[test]
    fn test_public_key_fingerprint_deterministic() {
        let k = PqPublicKey::new(PqAlgorithm::MlDsa65, vec![1, 2, 3, 4]);
        let fp1 = k.fingerprint();
        let fp2 = k.fingerprint();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_public_key_fingerprint_differs() {
        let k1 = PqPublicKey::new(PqAlgorithm::MlDsa65, vec![1, 2, 3]);
        let k2 = PqPublicKey::new(PqAlgorithm::MlDsa65, vec![4, 5, 6]);
        assert_ne!(k1.fingerprint(), k2.fingerprint());
    }

    #[test]
    fn test_public_key_validate_length() {
        let k = PqPublicKey::new(PqAlgorithm::SlhDsa128s, vec![0u8; 32]);
        assert!(k.validate_length().is_ok());
        let k2 = PqPublicKey::new(PqAlgorithm::SlhDsa128s, vec![0u8; 10]);
        assert!(k2.validate_length().is_err());
    }

    #[test]
    fn test_self_signed_cert() {
        let cert = make_ca();
        assert!(cert.is_self_signed());
        assert!(cert.is_ca);
        assert!(cert.check_validity(2000).is_ok());
    }

    #[test]
    fn test_cert_expired() {
        let cert = make_ca();
        let far_future = cert.validity.not_after + 1;
        assert!(cert.check_validity(far_future).is_err());
    }

    #[test]
    fn test_cert_not_yet_valid() {
        let cert = make_ca();
        assert!(cert.check_validity(999).is_err());
    }

    #[test]
    fn test_self_signed_verify() {
        let cert = make_ca();
        assert!(cert.verify_signature(&cert.public_key).is_ok());
    }

    #[test]
    fn test_chain_single_root() {
        let root = make_ca();
        let chain = CertChain::new(vec![root]);
        assert_eq!(chain.len(), 1);
        assert!(chain.validate(2000).is_ok());
    }

    #[test]
    fn test_chain_two_certs() {
        let root = make_ca();
        let leaf_config = PqCertConfig::new().with_algorithm(PqAlgorithm::MlDsa65);
        let mut leaf = leaf_config.build_certificate(
            2,
            DistinguishedName::new("Leaf"),
            DistinguishedName::new("RootCA"),
            1000,
        );
        let tbs = leaf.tbs_digest();
        leaf.signature = root.sign_tbs(tbs);
        let chain = CertChain::new(vec![leaf, root]);
        assert_eq!(chain.len(), 2);
        assert!(chain.validate(2000).is_ok());
    }

    #[test]
    fn test_chain_empty() {
        let chain = CertChain::new(vec![]);
        assert!(chain.is_empty());
        assert!(chain.validate(0).is_err());
    }

    #[test]
    fn test_chain_broken_issuer() {
        let root = make_ca();
        let leaf_config = PqCertConfig::new().with_algorithm(PqAlgorithm::MlDsa65);
        let leaf = leaf_config.build_certificate(
            3,
            DistinguishedName::new("Leaf"),
            DistinguishedName::new("WrongIssuer"),
            1000,
        );
        let chain = CertChain::new(vec![leaf, root]);
        assert!(chain.validate(2000).is_err());
    }

    #[test]
    fn test_config_builder() {
        let cfg = PqCertConfig::new()
            .with_algorithm(PqAlgorithm::MlDsa87)
            .with_validity_secs(7200)
            .with_ca(true)
            .with_max_path_length(2);
        assert_eq!(cfg.algorithm, PqAlgorithm::MlDsa87);
        assert_eq!(cfg.validity_secs, 7200);
        assert!(cfg.is_ca);
        assert_eq!(cfg.max_path_length, Some(2));
    }

    #[test]
    fn test_config_default() {
        let cfg = PqCertConfig::default();
        assert_eq!(cfg.algorithm, PqAlgorithm::MlDsa65);
        assert!(!cfg.is_ca);
    }

    #[test]
    fn test_display_impls() {
        let cert = make_ca();
        let s = format!("{cert}");
        assert!(s.contains("PqCert"));
        assert!(s.contains("RootCA"));

        let cfg = PqCertConfig::new();
        let s2 = format!("{cfg}");
        assert!(s2.contains("PqCertConfig"));
    }

    #[test]
    fn test_u64_roundtrip() {
        let v: u64 = 0xdeadbeefcafe1234;
        let bytes = u64_to_bytes(v);
        let back = bytes_to_u64(&bytes);
        assert_eq!(v, back);
    }

    #[test]
    fn test_signature_display() {
        let sig = PqSignature::new(PqAlgorithm::MlDsa65, vec![0u8; 16]);
        let s = format!("{sig}");
        assert!(s.contains("PqSig"));
        assert!(s.contains("16"));
    }
}
