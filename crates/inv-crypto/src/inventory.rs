//! Cryptographic inventory and quantum risk assessment.
//!
//! Automated discovery, cataloging, and quantum-vulnerability scoring of all
//! cryptographic assets across the mesh. Each node scans its own crypto
//! footprint (certificates, TLS configs, key algorithms, HMAC tokens, KEM/DSA
//! settings) and reports a [`NodeCryptoInventory`] via gossip.
//!
//! Inspired by NIST SP 1800-38 (Migration to Post-Quantum Cryptography) and
//! the Quantum Computing Cybersecurity Preparedness Act (H.R. 7535).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Crypto asset types
// ---------------------------------------------------------------------------

/// A single discovered cryptographic asset on a node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CryptoAsset {
    /// Unique identifier for this asset (e.g., "tls-server-cert", "mesh-hmac", "pq-kem-config").
    pub asset_id: String,
    /// Human-readable name.
    pub name: String,
    /// Where this asset was found.
    pub location: AssetLocation,
    /// The cryptographic algorithm in use.
    pub algorithm: CryptoAlgorithm,
    /// Key size in bits (e.g., 256 for AES-256, 2048 for RSA-2048, 768 for ML-KEM-768).
    pub key_bits: u32,
    /// What this asset is used for.
    pub usage: AssetUsage,
    /// Quantum risk assessment.
    pub quantum_risk: QuantumRisk,
    /// Expiration time, if applicable (certificates).
    pub expires_at: Option<DateTime<Utc>>,
    /// Additional metadata (issuer, subject, etc.).
    pub metadata: HashMap<String, String>,
}

/// Where a cryptographic asset was discovered.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AssetLocation {
    /// Node's TLS server configuration.
    TlsServer,
    /// Node's TLS client configuration (mTLS).
    TlsClient,
    /// Inter-node mesh authentication (HMAC tokens).
    MeshAuth,
    /// Certificate authority (root or intermediate).
    CertificateAuthority,
    /// Node identity keypair.
    NodeIdentity,
    /// PQ-hybrid TLS configuration.
    PqTls,
    /// Encryption at rest (secrets store, DB encryption).
    EncryptionAtRest,
    /// Application-level signing (JWTs, attestations).
    ApplicationSigning,
    /// QUIC transport layer.
    QuicTransport,
    /// Custom location.
    Custom(String),
}

/// Recognized cryptographic algorithms with quantum vulnerability data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CryptoAlgorithm {
    // -- Asymmetric (classical, quantum-vulnerable) --
    Rsa2048,
    Rsa4096,
    EcdsaP256,
    EcdsaP384,
    Ed25519,
    X25519,

    // -- Symmetric (quantum-resistant with sufficient key size) --
    Aes128Gcm,
    Aes256Gcm,
    ChaCha20Poly1305,

    // -- Hash / MAC (quantum implications via Grover's) --
    Sha256,
    Sha384,
    Sha512,
    HmacSha256,
    HmacSha512,

    // -- Post-quantum (NIST standards) --
    MlKem512,
    MlKem768,
    MlKem1024,
    MlDsa44,
    MlDsa65,
    MlDsa87,

    // -- Hybrid modes --
    X25519MlKem768,
    Ed25519MlDsa65,

    // -- Unknown / custom --
    Other(String),
}

impl CryptoAlgorithm {
    /// Classify the quantum risk level for this algorithm.
    pub fn quantum_risk(&self) -> QuantumRisk {
        match self {
            // Asymmetric: broken by Shor's algorithm
            Self::Rsa2048 => QuantumRisk::Critical,
            Self::Rsa4096 => QuantumRisk::Critical,
            Self::EcdsaP256 => QuantumRisk::Critical,
            Self::EcdsaP384 => QuantumRisk::Critical,
            Self::Ed25519 => QuantumRisk::Critical,
            Self::X25519 => QuantumRisk::Critical,

            // Symmetric: Grover's halves effective key size
            // AES-128 → 64-bit security (insufficient)
            Self::Aes128Gcm => QuantumRisk::Medium,
            // AES-256 → 128-bit security (sufficient)
            Self::Aes256Gcm => QuantumRisk::Safe,
            Self::ChaCha20Poly1305 => QuantumRisk::Safe,

            // Hashes: Grover's halves preimage resistance
            // SHA-256 → 128-bit (sufficient for most uses)
            Self::Sha256 | Self::HmacSha256 => QuantumRisk::Low,
            Self::Sha384 | Self::Sha512 | Self::HmacSha512 => QuantumRisk::Safe,

            // Post-quantum: designed to resist quantum attacks
            Self::MlKem512 => QuantumRisk::Safe,
            Self::MlKem768 => QuantumRisk::Safe,
            Self::MlKem1024 => QuantumRisk::Safe,
            Self::MlDsa44 => QuantumRisk::Safe,
            Self::MlDsa65 => QuantumRisk::Safe,
            Self::MlDsa87 => QuantumRisk::Safe,

            // Hybrid: safe (PQ component provides quantum resistance)
            Self::X25519MlKem768 => QuantumRisk::Safe,
            Self::Ed25519MlDsa65 => QuantumRisk::Safe,

            Self::Other(_) => QuantumRisk::Unknown,
        }
    }

    /// NIST category for this algorithm.
    pub fn nist_category(&self) -> &'static str {
        match self {
            Self::Rsa2048 | Self::Rsa4096 => "Asymmetric Encryption / Signature",
            Self::EcdsaP256 | Self::EcdsaP384 | Self::Ed25519 => "Digital Signature",
            Self::X25519 => "Key Agreement",
            Self::Aes128Gcm | Self::Aes256Gcm | Self::ChaCha20Poly1305 => "Symmetric Encryption",
            Self::Sha256 | Self::Sha384 | Self::Sha512 => "Hash Function",
            Self::HmacSha256 | Self::HmacSha512 => "Message Authentication",
            Self::MlKem512 | Self::MlKem768 | Self::MlKem1024 => "PQ Key Encapsulation (FIPS 203)",
            Self::MlDsa44 | Self::MlDsa65 | Self::MlDsa87 => "PQ Digital Signature (FIPS 204)",
            Self::X25519MlKem768 => "Hybrid Key Agreement",
            Self::Ed25519MlDsa65 => "Hybrid Digital Signature",
            Self::Other(_) => "Unknown",
        }
    }
}

impl std::fmt::Display for CryptoAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rsa2048 => write!(f, "RSA-2048"),
            Self::Rsa4096 => write!(f, "RSA-4096"),
            Self::EcdsaP256 => write!(f, "ECDSA P-256"),
            Self::EcdsaP384 => write!(f, "ECDSA P-384"),
            Self::Ed25519 => write!(f, "Ed25519"),
            Self::X25519 => write!(f, "X25519"),
            Self::Aes128Gcm => write!(f, "AES-128-GCM"),
            Self::Aes256Gcm => write!(f, "AES-256-GCM"),
            Self::ChaCha20Poly1305 => write!(f, "ChaCha20-Poly1305"),
            Self::Sha256 => write!(f, "SHA-256"),
            Self::Sha384 => write!(f, "SHA-384"),
            Self::Sha512 => write!(f, "SHA-512"),
            Self::HmacSha256 => write!(f, "HMAC-SHA-256"),
            Self::HmacSha512 => write!(f, "HMAC-SHA-512"),
            Self::MlKem512 => write!(f, "ML-KEM-512"),
            Self::MlKem768 => write!(f, "ML-KEM-768"),
            Self::MlKem1024 => write!(f, "ML-KEM-1024"),
            Self::MlDsa44 => write!(f, "ML-DSA-44"),
            Self::MlDsa65 => write!(f, "ML-DSA-65"),
            Self::MlDsa87 => write!(f, "ML-DSA-87"),
            Self::X25519MlKem768 => write!(f, "X25519 + ML-KEM-768 (hybrid)"),
            Self::Ed25519MlDsa65 => write!(f, "Ed25519 + ML-DSA-65 (hybrid)"),
            Self::Other(name) => write!(f, "{name}"),
        }
    }
}

/// What a cryptographic asset is used for.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AssetUsage {
    KeyExchange,
    DigitalSignature,
    Encryption,
    Authentication,
    Hashing,
    CertificateSigning,
}

// ---------------------------------------------------------------------------
// Quantum risk classification
// ---------------------------------------------------------------------------

/// Quantum vulnerability classification for a cryptographic asset.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QuantumRisk {
    /// Quantum-safe — no action needed.
    Safe,
    /// Minor theoretical risk (e.g., Grover's halving on 256-bit hashes).
    Low,
    /// Needs attention (e.g., AES-128 reduced to 64-bit security).
    Medium,
    /// Vulnerable to known quantum algorithms (Shor's). Migrate ASAP.
    Critical,
    /// Risk unknown — algorithm not recognized.
    Unknown,
}

impl QuantumRisk {
    /// Numeric score for sorting/aggregation (higher = worse).
    pub fn score(&self) -> u8 {
        match self {
            Self::Safe => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::Critical => 3,
            Self::Unknown => 4,
        }
    }

    /// Migration urgency label.
    pub fn urgency(&self) -> &'static str {
        match self {
            Self::Safe => "none",
            Self::Low => "monitor",
            Self::Medium => "plan",
            Self::Critical => "migrate",
            Self::Unknown => "investigate",
        }
    }

    /// CNSA 2.0 / NIST recommendation.
    pub fn recommendation(&self) -> &'static str {
        match self {
            Self::Safe => "Compliant with CNSA 2.0 timeline",
            Self::Low => "Acceptable through 2030; consider upgrade path",
            Self::Medium => "Upgrade to 256-bit equivalent before 2028",
            Self::Critical => "Replace with PQ/hybrid algorithm per NIST SP 800-208",
            Self::Unknown => "Audit algorithm and classify before next review cycle",
        }
    }
}

impl std::fmt::Display for QuantumRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe => write!(f, "SAFE"),
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::Critical => write!(f, "CRITICAL"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// Node crypto inventory
// ---------------------------------------------------------------------------

/// Complete cryptographic inventory for a single mesh node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCryptoInventory {
    /// Node identifier.
    pub node_id: String,
    /// When this inventory was last scanned.
    pub scanned_at: DateTime<Utc>,
    /// All discovered crypto assets.
    pub assets: Vec<CryptoAsset>,
    /// Summary statistics.
    pub summary: InventorySummary,
}

/// Aggregated statistics from a node's crypto inventory.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InventorySummary {
    /// Total number of discovered crypto assets.
    pub total_assets: usize,
    /// Count by quantum risk level.
    pub risk_counts: RiskCounts,
    /// Overall PQ readiness as a percentage (0.0 = fully vulnerable, 1.0 = fully safe).
    pub pq_readiness: f64,
    /// Whether all critical assets have PQ migration paths.
    pub migration_coverage: bool,
}

/// Counts of assets at each risk level.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiskCounts {
    pub safe: usize,
    pub low: usize,
    pub medium: usize,
    pub critical: usize,
    pub unknown: usize,
}

impl NodeCryptoInventory {
    /// Build inventory from a list of discovered assets.
    pub fn from_assets(node_id: String, assets: Vec<CryptoAsset>) -> Self {
        let summary = InventorySummary::compute(&assets);
        Self {
            node_id,
            scanned_at: Utc::now(),
            assets,
            summary,
        }
    }
}

impl InventorySummary {
    /// Compute summary statistics from a list of crypto assets.
    pub fn compute(assets: &[CryptoAsset]) -> Self {
        let mut counts = RiskCounts::default();
        for asset in assets {
            match asset.quantum_risk {
                QuantumRisk::Safe => counts.safe += 1,
                QuantumRisk::Low => counts.low += 1,
                QuantumRisk::Medium => counts.medium += 1,
                QuantumRisk::Critical => counts.critical += 1,
                QuantumRisk::Unknown => counts.unknown += 1,
            }
        }

        let total = assets.len();
        let pq_readiness = if total == 0 {
            1.0
        } else {
            (counts.safe + counts.low) as f64 / total as f64
        };

        // Migration coverage: true if no critical assets remain
        let migration_coverage = counts.critical == 0;

        Self {
            total_assets: total,
            risk_counts: counts,
            pq_readiness,
            migration_coverage,
        }
    }
}

// ---------------------------------------------------------------------------
// Org-wide crypto inventory (aggregated across mesh)
// ---------------------------------------------------------------------------

/// Organization-wide cryptographic inventory aggregated from all mesh nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrgCryptoInventory {
    /// Organization identifier.
    pub org_id: String,
    /// Per-node inventories.
    pub nodes: Vec<NodeCryptoInventory>,
    /// Aggregate summary across all nodes.
    pub summary: InventorySummary,
    /// Migration roadmap — ordered list of assets to upgrade.
    pub migration_queue: Vec<MigrationItem>,
    /// When this aggregate was last computed.
    pub aggregated_at: DateTime<Utc>,
}

/// A single item in the PQ migration roadmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationItem {
    /// Node where this asset lives.
    pub node_id: String,
    /// The asset to migrate.
    pub asset_id: String,
    /// Current algorithm.
    pub current_algorithm: CryptoAlgorithm,
    /// Recommended replacement algorithm.
    pub target_algorithm: CryptoAlgorithm,
    /// Priority score (lower = more urgent).
    pub priority: u32,
    /// Current risk level.
    pub risk: QuantumRisk,
    /// Whether this migration has been completed.
    pub migrated: bool,
}

impl OrgCryptoInventory {
    /// Build org inventory from per-node inventories.
    pub fn aggregate(org_id: String, nodes: Vec<NodeCryptoInventory>) -> Self {
        // Collect all assets for aggregate summary
        let all_assets: Vec<&CryptoAsset> = nodes.iter().flat_map(|n| &n.assets).collect();
        let summary =
            InventorySummary::compute(&all_assets.iter().map(|a| (*a).clone()).collect::<Vec<_>>());

        // Build migration queue from critical and medium-risk assets
        let mut migration_queue: Vec<MigrationItem> = Vec::new();
        for node in &nodes {
            for asset in &node.assets {
                if matches!(
                    asset.quantum_risk,
                    QuantumRisk::Critical | QuantumRisk::Medium
                ) {
                    let target = recommend_replacement(&asset.algorithm);
                    migration_queue.push(MigrationItem {
                        node_id: node.node_id.clone(),
                        asset_id: asset.asset_id.clone(),
                        current_algorithm: asset.algorithm.clone(),
                        target_algorithm: target,
                        priority: migration_priority(asset),
                        risk: asset.quantum_risk,
                        migrated: false,
                    });
                }
            }
        }
        migration_queue.sort_by_key(|m| m.priority);

        Self {
            org_id,
            nodes,
            summary,
            migration_queue,
            aggregated_at: Utc::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-node scanner
// ---------------------------------------------------------------------------

/// Scan a node's cryptographic configuration and produce an inventory.
///
/// This examines:
/// - TLS server/client certificate algorithms and key sizes
/// - Mesh HMAC authentication tokens
/// - PQ-hybrid TLS configuration
/// - Encryption-at-rest settings
/// - Node identity keypair type
///
/// In production, this runs on each node agent and reports via gossip.
/// The scanner takes no external dependencies — it examines the node's
/// own configuration state.
pub fn scan_node_crypto(node_id: &str, config: &NodeCryptoConfig) -> NodeCryptoInventory {
    let mut assets = Vec::new();

    // 1. TLS server certificate
    if let Some(ref tls) = config.tls_cert_algorithm {
        let algo = parse_tls_algorithm(tls);
        assets.push(CryptoAsset {
            asset_id: "tls-server-cert".into(),
            name: "TLS Server Certificate".into(),
            location: AssetLocation::TlsServer,
            algorithm: algo.clone(),
            key_bits: config.tls_key_bits.unwrap_or(256),
            usage: AssetUsage::KeyExchange,
            quantum_risk: algo.quantum_risk(),
            expires_at: config.tls_cert_expires,
            metadata: config.tls_cert_metadata.clone().unwrap_or_default(),
        });
    }

    // 2. TLS cipher suite for bulk encryption
    if let Some(ref cipher) = config.tls_cipher_suite {
        let algo = parse_cipher_suite(cipher);
        assets.push(CryptoAsset {
            asset_id: "tls-bulk-cipher".into(),
            name: "TLS Bulk Encryption".into(),
            location: AssetLocation::TlsServer,
            algorithm: algo.clone(),
            key_bits: cipher_key_bits(&algo),
            usage: AssetUsage::Encryption,
            quantum_risk: algo.quantum_risk(),
            expires_at: None,
            metadata: HashMap::new(),
        });
    }

    // 3. Mesh HMAC authentication
    if config.mesh_hmac_enabled {
        let algo = CryptoAlgorithm::HmacSha256;
        assets.push(CryptoAsset {
            asset_id: "mesh-hmac-auth".into(),
            name: "Mesh Inter-Node HMAC".into(),
            location: AssetLocation::MeshAuth,
            algorithm: algo.clone(),
            key_bits: 256,
            usage: AssetUsage::Authentication,
            quantum_risk: algo.quantum_risk(),
            expires_at: None,
            metadata: HashMap::new(),
        });
    }

    // 4. Node identity keypair
    if let Some(ref identity_algo) = config.node_identity_algorithm {
        let algo = parse_identity_algorithm(identity_algo);
        assets.push(CryptoAsset {
            asset_id: "node-identity".into(),
            name: "Node Identity Keypair".into(),
            location: AssetLocation::NodeIdentity,
            algorithm: algo.clone(),
            key_bits: identity_key_bits(&algo),
            usage: AssetUsage::DigitalSignature,
            quantum_risk: algo.quantum_risk(),
            expires_at: None,
            metadata: HashMap::new(),
        });
    }

    // 5. PQ-hybrid TLS (if configured)
    if let Some(ref pq_config) = config.pq_tls_config {
        let kem_algo = match pq_config.kem_algorithm {
            inv_pqcrypto::KemAlgorithm::MlKem768 => CryptoAlgorithm::MlKem768,
            inv_pqcrypto::KemAlgorithm::X25519MlKem768Hybrid => CryptoAlgorithm::X25519MlKem768,
            inv_pqcrypto::KemAlgorithm::ClassicalX25519 => CryptoAlgorithm::X25519,
        };
        assets.push(CryptoAsset {
            asset_id: "pq-tls-kem".into(),
            name: "PQ-TLS Key Encapsulation".into(),
            location: AssetLocation::PqTls,
            algorithm: kem_algo.clone(),
            key_bits: kem_key_bits(&kem_algo),
            usage: AssetUsage::KeyExchange,
            quantum_risk: kem_algo.quantum_risk(),
            expires_at: None,
            metadata: HashMap::new(),
        });

        let sig_algo = match pq_config.sig_algorithm {
            inv_pqcrypto::SigAlgorithm::MlDsa65 => CryptoAlgorithm::MlDsa65,
            inv_pqcrypto::SigAlgorithm::Ed25519MlDsa65Hybrid => CryptoAlgorithm::Ed25519MlDsa65,
            inv_pqcrypto::SigAlgorithm::ClassicalEd25519 => CryptoAlgorithm::Ed25519,
        };
        assets.push(CryptoAsset {
            asset_id: "pq-tls-sig".into(),
            name: "PQ-TLS Digital Signature".into(),
            location: AssetLocation::PqTls,
            algorithm: sig_algo.clone(),
            key_bits: sig_key_bits(&sig_algo),
            usage: AssetUsage::DigitalSignature,
            quantum_risk: sig_algo.quantum_risk(),
            expires_at: None,
            metadata: HashMap::new(),
        });
    }

    // 6. Encryption at rest
    if let Some(ref enc_algo) = config.encryption_at_rest {
        let algo = parse_encryption_algorithm(enc_algo);
        assets.push(CryptoAsset {
            asset_id: "encryption-at-rest".into(),
            name: "Encryption at Rest".into(),
            location: AssetLocation::EncryptionAtRest,
            algorithm: algo.clone(),
            key_bits: cipher_key_bits(&algo),
            usage: AssetUsage::Encryption,
            quantum_risk: algo.quantum_risk(),
            expires_at: None,
            metadata: HashMap::new(),
        });
    }

    // 7. CA certificate (if this node is a CA or has a CA cert)
    if let Some(ref ca_algo) = config.ca_algorithm {
        let algo = parse_identity_algorithm(ca_algo);
        assets.push(CryptoAsset {
            asset_id: "ca-certificate".into(),
            name: "Certificate Authority".into(),
            location: AssetLocation::CertificateAuthority,
            algorithm: algo.clone(),
            key_bits: identity_key_bits(&algo),
            usage: AssetUsage::CertificateSigning,
            quantum_risk: algo.quantum_risk(),
            expires_at: config.ca_cert_expires,
            metadata: HashMap::new(),
        });
    }

    NodeCryptoInventory::from_assets(node_id.to_string(), assets)
}

/// Configuration inputs for the per-node crypto scanner.
/// Populated from the node agent's runtime state.
#[derive(Debug, Clone, Default)]
pub struct NodeCryptoConfig {
    /// TLS certificate algorithm name (e.g., "Ed25519", "ECDSA-P256").
    pub tls_cert_algorithm: Option<String>,
    /// TLS certificate key size in bits.
    pub tls_key_bits: Option<u32>,
    /// TLS certificate expiration.
    pub tls_cert_expires: Option<DateTime<Utc>>,
    /// Additional TLS certificate metadata (subject, issuer, etc.).
    pub tls_cert_metadata: Option<HashMap<String, String>>,
    /// TLS cipher suite name (e.g., "TLS_AES_256_GCM_SHA384").
    pub tls_cipher_suite: Option<String>,
    /// Whether mesh HMAC authentication is active.
    pub mesh_hmac_enabled: bool,
    /// Node identity keypair algorithm (e.g., "Ed25519").
    pub node_identity_algorithm: Option<String>,
    /// PQ-hybrid TLS configuration, if enabled.
    pub pq_tls_config: Option<inv_pqcrypto::PqTlsConfig>,
    /// Encryption-at-rest algorithm name (e.g., "AES-256-GCM").
    pub encryption_at_rest: Option<String>,
    /// CA certificate algorithm, if this node has CA duties.
    pub ca_algorithm: Option<String>,
    /// CA certificate expiration.
    pub ca_cert_expires: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Algorithm parsers and helpers
// ---------------------------------------------------------------------------

fn parse_tls_algorithm(name: &str) -> CryptoAlgorithm {
    let lower = name.to_lowercase();
    if lower.contains("ed25519") {
        CryptoAlgorithm::Ed25519
    } else if lower.contains("ecdsa") && lower.contains("256") {
        CryptoAlgorithm::EcdsaP256
    } else if lower.contains("ecdsa") && lower.contains("384") {
        CryptoAlgorithm::EcdsaP384
    } else if lower.contains("rsa") && lower.contains("4096") {
        CryptoAlgorithm::Rsa4096
    } else if lower.contains("rsa") {
        CryptoAlgorithm::Rsa2048
    } else {
        CryptoAlgorithm::Other(name.to_string())
    }
}

fn parse_cipher_suite(name: &str) -> CryptoAlgorithm {
    let lower = name.to_lowercase();
    if lower.contains("aes_256") || lower.contains("aes-256") {
        CryptoAlgorithm::Aes256Gcm
    } else if lower.contains("aes_128") || lower.contains("aes-128") {
        CryptoAlgorithm::Aes128Gcm
    } else if lower.contains("chacha20") {
        CryptoAlgorithm::ChaCha20Poly1305
    } else {
        CryptoAlgorithm::Other(name.to_string())
    }
}

fn parse_identity_algorithm(name: &str) -> CryptoAlgorithm {
    let lower = name.to_lowercase();
    if lower.contains("ed25519") {
        CryptoAlgorithm::Ed25519
    } else if lower.contains("ecdsa") || lower.contains("p-256") || lower.contains("p256") {
        CryptoAlgorithm::EcdsaP256
    } else if lower.contains("rsa") {
        CryptoAlgorithm::Rsa2048
    } else if lower.contains("ml-dsa") || lower.contains("dilithium") {
        CryptoAlgorithm::MlDsa65
    } else {
        CryptoAlgorithm::Other(name.to_string())
    }
}

fn parse_encryption_algorithm(name: &str) -> CryptoAlgorithm {
    let lower = name.to_lowercase();
    if lower.contains("aes-256") || lower.contains("aes_256") {
        CryptoAlgorithm::Aes256Gcm
    } else if lower.contains("aes-128") || lower.contains("aes_128") {
        CryptoAlgorithm::Aes128Gcm
    } else if lower.contains("chacha20") {
        CryptoAlgorithm::ChaCha20Poly1305
    } else {
        CryptoAlgorithm::Other(name.to_string())
    }
}

fn cipher_key_bits(algo: &CryptoAlgorithm) -> u32 {
    match algo {
        CryptoAlgorithm::Aes128Gcm => 128,
        CryptoAlgorithm::Aes256Gcm => 256,
        CryptoAlgorithm::ChaCha20Poly1305 => 256,
        _ => 0,
    }
}

fn identity_key_bits(algo: &CryptoAlgorithm) -> u32 {
    match algo {
        CryptoAlgorithm::Ed25519 => 256,
        CryptoAlgorithm::EcdsaP256 => 256,
        CryptoAlgorithm::EcdsaP384 => 384,
        CryptoAlgorithm::Rsa2048 => 2048,
        CryptoAlgorithm::Rsa4096 => 4096,
        CryptoAlgorithm::MlDsa65 => 4032,
        _ => 0,
    }
}

fn kem_key_bits(algo: &CryptoAlgorithm) -> u32 {
    match algo {
        CryptoAlgorithm::X25519 => 256,
        CryptoAlgorithm::MlKem768 => 2400,
        CryptoAlgorithm::X25519MlKem768 => 2656,
        _ => 0,
    }
}

fn sig_key_bits(algo: &CryptoAlgorithm) -> u32 {
    match algo {
        CryptoAlgorithm::Ed25519 => 256,
        CryptoAlgorithm::MlDsa65 => 4032,
        CryptoAlgorithm::Ed25519MlDsa65 => 4288,
        _ => 0,
    }
}

/// Recommend the PQ replacement algorithm for a vulnerable algorithm.
pub fn recommend_replacement(algo: &CryptoAlgorithm) -> CryptoAlgorithm {
    match algo {
        // Key exchange → hybrid KEM
        CryptoAlgorithm::X25519 | CryptoAlgorithm::EcdsaP256 | CryptoAlgorithm::EcdsaP384 => {
            CryptoAlgorithm::X25519MlKem768
        }
        // Signatures → hybrid DSA
        CryptoAlgorithm::Ed25519 => CryptoAlgorithm::Ed25519MlDsa65,
        // RSA → hybrid KEM + DSA depending on usage (default to KEM)
        CryptoAlgorithm::Rsa2048 | CryptoAlgorithm::Rsa4096 => CryptoAlgorithm::X25519MlKem768,
        // AES-128 → AES-256
        CryptoAlgorithm::Aes128Gcm => CryptoAlgorithm::Aes256Gcm,
        // Already safe or unknown — recommend self
        other => other.clone(),
    }
}

/// Compute migration priority for an asset (lower = more urgent).
/// Factors: risk severity, asset usage criticality, expiration proximity.
fn migration_priority(asset: &CryptoAsset) -> u32 {
    let risk_weight = match asset.quantum_risk {
        QuantumRisk::Critical => 0,
        QuantumRisk::Medium => 100,
        QuantumRisk::Low => 200,
        QuantumRisk::Unknown => 150,
        QuantumRisk::Safe => 300,
    };

    let usage_weight = match asset.usage {
        AssetUsage::KeyExchange => 0,
        AssetUsage::CertificateSigning => 10,
        AssetUsage::DigitalSignature => 20,
        AssetUsage::Authentication => 30,
        AssetUsage::Encryption => 40,
        AssetUsage::Hashing => 50,
    };

    // Expiring certs get higher priority
    let expiry_weight = match asset.expires_at {
        Some(exp) if exp < Utc::now() + chrono::Duration::days(90) => 0,
        Some(exp) if exp < Utc::now() + chrono::Duration::days(365) => 10,
        _ => 20,
    };

    risk_weight + usage_weight + expiry_weight
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantum_risk_classification() {
        assert_eq!(
            CryptoAlgorithm::Rsa2048.quantum_risk(),
            QuantumRisk::Critical
        );
        assert_eq!(
            CryptoAlgorithm::Ed25519.quantum_risk(),
            QuantumRisk::Critical
        );
        assert_eq!(
            CryptoAlgorithm::X25519.quantum_risk(),
            QuantumRisk::Critical
        );
        assert_eq!(
            CryptoAlgorithm::EcdsaP256.quantum_risk(),
            QuantumRisk::Critical
        );
        assert_eq!(
            CryptoAlgorithm::Aes128Gcm.quantum_risk(),
            QuantumRisk::Medium
        );
        assert_eq!(CryptoAlgorithm::Aes256Gcm.quantum_risk(), QuantumRisk::Safe);
        assert_eq!(CryptoAlgorithm::MlKem768.quantum_risk(), QuantumRisk::Safe);
        assert_eq!(CryptoAlgorithm::MlDsa65.quantum_risk(), QuantumRisk::Safe);
        assert_eq!(
            CryptoAlgorithm::X25519MlKem768.quantum_risk(),
            QuantumRisk::Safe
        );
        assert_eq!(
            CryptoAlgorithm::Ed25519MlDsa65.quantum_risk(),
            QuantumRisk::Safe
        );
        assert_eq!(CryptoAlgorithm::HmacSha256.quantum_risk(), QuantumRisk::Low);
        assert_eq!(CryptoAlgorithm::Sha512.quantum_risk(), QuantumRisk::Safe);
    }

    #[test]
    fn risk_ordering() {
        assert!(QuantumRisk::Safe < QuantumRisk::Low);
        assert!(QuantumRisk::Low < QuantumRisk::Medium);
        assert!(QuantumRisk::Medium < QuantumRisk::Critical);
        assert!(QuantumRisk::Critical < QuantumRisk::Unknown);
    }

    #[test]
    fn risk_score_and_urgency() {
        assert_eq!(QuantumRisk::Safe.score(), 0);
        assert_eq!(QuantumRisk::Critical.score(), 3);
        assert_eq!(QuantumRisk::Critical.urgency(), "migrate");
        assert_eq!(QuantumRisk::Safe.urgency(), "none");
        assert_eq!(QuantumRisk::Medium.urgency(), "plan");
    }

    #[test]
    fn algorithm_display() {
        assert_eq!(CryptoAlgorithm::Rsa2048.to_string(), "RSA-2048");
        assert_eq!(CryptoAlgorithm::MlKem768.to_string(), "ML-KEM-768");
        assert_eq!(
            CryptoAlgorithm::X25519MlKem768.to_string(),
            "X25519 + ML-KEM-768 (hybrid)"
        );
    }

    #[test]
    fn algorithm_nist_category() {
        assert_eq!(
            CryptoAlgorithm::Rsa2048.nist_category(),
            "Asymmetric Encryption / Signature"
        );
        assert_eq!(
            CryptoAlgorithm::MlKem768.nist_category(),
            "PQ Key Encapsulation (FIPS 203)"
        );
        assert_eq!(
            CryptoAlgorithm::MlDsa65.nist_category(),
            "PQ Digital Signature (FIPS 204)"
        );
        assert_eq!(
            CryptoAlgorithm::Aes256Gcm.nist_category(),
            "Symmetric Encryption"
        );
    }

    #[test]
    fn scan_typical_node() {
        let config = NodeCryptoConfig {
            tls_cert_algorithm: Some("ECDSA-P256".into()),
            tls_key_bits: Some(256),
            tls_cert_expires: Some(Utc::now() + chrono::Duration::days(365)),
            tls_cert_metadata: None,
            tls_cipher_suite: Some("TLS_AES_256_GCM_SHA384".into()),
            mesh_hmac_enabled: true,
            node_identity_algorithm: Some("Ed25519".into()),
            pq_tls_config: Some(inv_pqcrypto::PqTlsConfig::hybrid_default()),
            encryption_at_rest: Some("AES-256-GCM".into()),
            ca_algorithm: Some("Ed25519".into()),
            ca_cert_expires: None,
        };

        let inventory = scan_node_crypto("inv-seed-ash", &config);
        assert_eq!(inventory.node_id, "inv-seed-ash");

        // Should find: tls cert, tls cipher, mesh hmac, node identity,
        // pq-kem, pq-sig, encryption-at-rest, ca-cert = 8 assets
        assert_eq!(inventory.assets.len(), 8);

        // Classical assets: tls cert (ECDSA-P256), node identity (Ed25519), CA (Ed25519) = 3 critical
        assert_eq!(inventory.summary.risk_counts.critical, 3);

        // PQ assets: pq-kem (hybrid), pq-sig (hybrid) = 2 safe
        // Also safe: tls cipher (AES-256), encryption-at-rest (AES-256) = 2 more safe
        assert_eq!(inventory.summary.risk_counts.safe, 4);

        // HMAC-SHA-256 = low
        assert_eq!(inventory.summary.risk_counts.low, 1);

        // PQ readiness: (4 safe + 1 low) / 8 total = 0.625
        assert!((inventory.summary.pq_readiness - 0.625).abs() < 0.001);

        // Not fully migrated (3 critical remain)
        assert!(!inventory.summary.migration_coverage);
    }

    #[test]
    fn scan_fully_pq_node() {
        let config = NodeCryptoConfig {
            tls_cert_algorithm: None,
            tls_key_bits: None,
            tls_cert_expires: None,
            tls_cert_metadata: None,
            tls_cipher_suite: Some("TLS_AES_256_GCM_SHA384".into()),
            mesh_hmac_enabled: true,
            node_identity_algorithm: None,
            pq_tls_config: Some(inv_pqcrypto::PqTlsConfig::pq_only()),
            encryption_at_rest: Some("AES-256-GCM".into()),
            ca_algorithm: None,
            ca_cert_expires: None,
        };

        let inventory = scan_node_crypto("pq-node", &config);
        // Assets: tls cipher, mesh hmac, pq-kem (MlKem768), pq-sig (MlDsa65), enc-at-rest = 5
        assert_eq!(inventory.assets.len(), 5);
        assert_eq!(inventory.summary.risk_counts.critical, 0);
        assert!(inventory.summary.migration_coverage);
    }

    #[test]
    fn org_inventory_aggregation() {
        let node1 = NodeCryptoInventory::from_assets(
            "node-1".into(),
            vec![CryptoAsset {
                asset_id: "tls-cert".into(),
                name: "TLS Cert".into(),
                location: AssetLocation::TlsServer,
                algorithm: CryptoAlgorithm::Ed25519,
                key_bits: 256,
                usage: AssetUsage::KeyExchange,
                quantum_risk: QuantumRisk::Critical,
                expires_at: None,
                metadata: HashMap::new(),
            }],
        );
        let node2 = NodeCryptoInventory::from_assets(
            "node-2".into(),
            vec![CryptoAsset {
                asset_id: "pq-kem".into(),
                name: "PQ KEM".into(),
                location: AssetLocation::PqTls,
                algorithm: CryptoAlgorithm::MlKem768,
                key_bits: 2400,
                usage: AssetUsage::KeyExchange,
                quantum_risk: QuantumRisk::Safe,
                expires_at: None,
                metadata: HashMap::new(),
            }],
        );

        let org = OrgCryptoInventory::aggregate("openie".into(), vec![node1, node2]);
        assert_eq!(org.summary.total_assets, 2);
        assert_eq!(org.summary.risk_counts.critical, 1);
        assert_eq!(org.summary.risk_counts.safe, 1);
        assert_eq!(org.summary.pq_readiness, 0.5);
        assert!(!org.summary.migration_coverage);

        // Migration queue should have 1 entry (Ed25519 → hybrid)
        assert_eq!(org.migration_queue.len(), 1);
        assert_eq!(org.migration_queue[0].node_id, "node-1");
        assert_eq!(
            org.migration_queue[0].target_algorithm,
            CryptoAlgorithm::Ed25519MlDsa65
        );
    }

    #[test]
    fn replacement_recommendations() {
        assert_eq!(
            recommend_replacement(&CryptoAlgorithm::Rsa2048),
            CryptoAlgorithm::X25519MlKem768
        );
        assert_eq!(
            recommend_replacement(&CryptoAlgorithm::Ed25519),
            CryptoAlgorithm::Ed25519MlDsa65
        );
        assert_eq!(
            recommend_replacement(&CryptoAlgorithm::X25519),
            CryptoAlgorithm::X25519MlKem768
        );
        assert_eq!(
            recommend_replacement(&CryptoAlgorithm::Aes128Gcm),
            CryptoAlgorithm::Aes256Gcm
        );
        assert_eq!(
            recommend_replacement(&CryptoAlgorithm::Aes256Gcm),
            CryptoAlgorithm::Aes256Gcm
        );
    }

    #[test]
    fn parse_algorithms() {
        assert_eq!(parse_tls_algorithm("Ed25519"), CryptoAlgorithm::Ed25519);
        assert_eq!(
            parse_tls_algorithm("ECDSA-P256"),
            CryptoAlgorithm::EcdsaP256
        );
        assert_eq!(parse_tls_algorithm("RSA-4096"), CryptoAlgorithm::Rsa4096);
        assert_eq!(
            parse_cipher_suite("TLS_AES_256_GCM_SHA384"),
            CryptoAlgorithm::Aes256Gcm
        );
        assert_eq!(
            parse_cipher_suite("TLS_CHACHA20_POLY1305_SHA256"),
            CryptoAlgorithm::ChaCha20Poly1305
        );
    }

    #[test]
    fn inventory_summary_empty() {
        let summary = InventorySummary::compute(&[]);
        assert_eq!(summary.total_assets, 0);
        assert_eq!(summary.pq_readiness, 1.0);
        assert!(summary.migration_coverage);
    }

    #[test]
    fn inventory_serialization_roundtrip() {
        let config = NodeCryptoConfig {
            tls_cert_algorithm: Some("Ed25519".into()),
            tls_key_bits: Some(256),
            mesh_hmac_enabled: true,
            ..Default::default()
        };
        let inventory = scan_node_crypto("test-node", &config);
        let json = serde_json::to_string(&inventory).unwrap();
        let parsed: NodeCryptoInventory = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_id, "test-node");
        assert_eq!(parsed.assets.len(), inventory.assets.len());
    }

    #[test]
    fn quantum_risk_display() {
        assert_eq!(QuantumRisk::Critical.to_string(), "CRITICAL");
        assert_eq!(QuantumRisk::Safe.to_string(), "SAFE");
    }

    #[test]
    fn migration_priority_ordering() {
        let critical_ke = CryptoAsset {
            asset_id: "test".into(),
            name: "test".into(),
            location: AssetLocation::TlsServer,
            algorithm: CryptoAlgorithm::X25519,
            key_bits: 256,
            usage: AssetUsage::KeyExchange,
            quantum_risk: QuantumRisk::Critical,
            expires_at: None,
            metadata: HashMap::new(),
        };
        let medium_enc = CryptoAsset {
            asset_id: "test2".into(),
            name: "test2".into(),
            location: AssetLocation::EncryptionAtRest,
            algorithm: CryptoAlgorithm::Aes128Gcm,
            key_bits: 128,
            usage: AssetUsage::Encryption,
            quantum_risk: QuantumRisk::Medium,
            expires_at: None,
            metadata: HashMap::new(),
        };
        // Critical key exchange should be higher priority (lower number) than medium encryption
        assert!(migration_priority(&critical_ke) < migration_priority(&medium_enc));
    }
}
