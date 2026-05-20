//! JouleDB Cyber Link
//!
//! HDC-powered Cybersecurity and Threat Intelligence module.
//! Provides O(1) similarity matching for IOCs, malware variants, and network anomalies.

pub use joule_db_hdc::{BinaryHV, BundleAccumulator};
use std::collections::HashMap;
use std::net::IpAddr;

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

/// Indicator of Compromise (IOC) types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum IocType {
    IpAddress,
    Domain,
    Url,
    FileHash,
    Email,
    Registry,
    Mutex,
    Certificate,
}

/// Threat severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ThreatSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

impl ThreatSeverity {
    pub fn score(&self) -> u8 {
        match self {
            ThreatSeverity::Critical => 10,
            ThreatSeverity::High => 8,
            ThreatSeverity::Medium => 5,
            ThreatSeverity::Low => 3,
            ThreatSeverity::Info => 1,
        }
    }
}

/// MITRE ATT&CK Tactic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MitreTactic {
    InitialAccess,
    Execution,
    Persistence,
    PrivilegeEscalation,
    DefenseEvasion,
    CredentialAccess,
    Discovery,
    LateralMovement,
    Collection,
    CommandAndControl,
    Exfiltration,
    Impact,
}

/// An Indicator of Compromise
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Ioc {
    pub id: String,
    pub ioc_type: IocType,
    pub value: String,
    pub severity: ThreatSeverity,
    pub tags: Vec<String>,
    pub first_seen: u64,
    pub last_seen: u64,
    pub confidence: f64,
}

/// A malware sample
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MalwareSample {
    pub sha256: String,
    pub sha1: String,
    pub md5: String,
    pub family: String,
    pub variant: Option<String>,
    pub file_type: String,
    pub file_size: u64,
    pub tactics: Vec<MitreTactic>,
    pub techniques: Vec<String>,
    pub iocs: Vec<String>,
    pub first_seen: u64,
}

/// A network flow event
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NetworkFlow {
    pub src_ip: String,
    pub dst_ip: String,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: Protocol,
    pub bytes_sent: u64,
    pub bytes_recv: u64,
    pub packets: u32,
    pub duration_ms: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Protocol {
    Tcp,
    Udp,
    Icmp,
    Other,
}

/// A security event/alert
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecurityEvent {
    pub id: String,
    pub event_type: String,
    pub source: String,
    pub severity: ThreatSeverity,
    pub description: String,
    pub raw_log: String,
    pub matched_iocs: Vec<String>,
    pub tactics: Vec<MitreTactic>,
    pub timestamp: u64,
}

// ============================================================================
// Cyber Link Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// VSA Encoder for cybersecurity data
    pub struct CyberLink {
        seed: 0xC0BE_5EC0,
        dimension: 10000,
        fields: ["ioc_type", "value", "severity", "tags", "confidence",
                 "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
                 "bytes", "duration", "family", "techniques", "hash"],
        scalars: ["port", "bytes", "duration", "confidence", "severity_score"],
        enums: {
            ioc_type_vectors: IocType => [IocType::IpAddress, IocType::Domain, IocType::Url, IocType::FileHash, IocType::Email, IocType::Registry, IocType::Mutex, IocType::Certificate],
            severity_vectors: ThreatSeverity => [ThreatSeverity::Critical, ThreatSeverity::High, ThreatSeverity::Medium, ThreatSeverity::Low, ThreatSeverity::Info],
            protocol_vectors: Protocol => [Protocol::Tcp, Protocol::Udp, Protocol::Icmp, Protocol::Other],
            tactic_vectors: MitreTactic => [MitreTactic::InitialAccess, MitreTactic::Execution, MitreTactic::Persistence, MitreTactic::PrivilegeEscalation, MitreTactic::DefenseEvasion, MitreTactic::CredentialAccess, MitreTactic::Discovery, MitreTactic::LateralMovement, MitreTactic::Collection, MitreTactic::CommandAndControl, MitreTactic::Exfiltration, MitreTactic::Impact]
        },
    }
}

impl CyberLink {
    /// Encode an IOC into a hypervector
    pub fn encode_ioc(&self, ioc: &Ioc) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // IOC Type
        let type_vec = &self.ioc_type_vectors[&ioc.ioc_type];
        acc.add(&self.field_vectors["ioc_type"].bind(type_vec));

        // Value (hash-based for exact matching)
        let value_hv = BinaryHV::from_hash(ioc.value.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["value"].bind(&value_hv));

        // Severity
        let sev_vec = &self.severity_vectors[&ioc.severity];
        acc.add(&self.field_vectors["severity"].bind(sev_vec));

        // Confidence (scalar)
        let conf_shift = (ioc.confidence * 100.0) as usize;
        let conf_vec = self.scalar_bases["confidence"].permute_words(conf_shift % 157);
        acc.add(&self.field_vectors["confidence"].bind(&conf_vec));

        // Tags (superposition of all tag hashes)
        if !ioc.tags.is_empty() {
            let mut tag_acc = BundleAccumulator::new(DIMENSION);
            for tag in &ioc.tags {
                tag_acc.add(&BinaryHV::from_hash(tag.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["tags"].bind(&tag_acc.threshold()));
        }

        acc.threshold()
    }

    /// Encode a malware sample
    pub fn encode_malware(&self, sample: &MalwareSample) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // SHA256 hash
        let hash_hv = BinaryHV::from_hash(sample.sha256.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["hash"].bind(&hash_hv));

        // Family
        let family_hv = BinaryHV::from_hash(sample.family.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["family"].bind(&family_hv));

        // Tactics (superposition)
        if !sample.tactics.is_empty() {
            let mut tactic_acc = BundleAccumulator::new(DIMENSION);
            for tactic in &sample.tactics {
                tactic_acc.add(&self.tactic_vectors[tactic]);
            }
            acc.add(&tactic_acc.threshold());
        }

        // Techniques (superposition of hashes)
        if !sample.techniques.is_empty() {
            let mut tech_acc = BundleAccumulator::new(DIMENSION);
            for tech in &sample.techniques {
                tech_acc.add(&BinaryHV::from_hash(tech.as_bytes(), DIMENSION));
            }
            acc.add(&self.field_vectors["techniques"].bind(&tech_acc.threshold()));
        }

        acc.threshold()
    }

    /// Encode a network flow
    pub fn encode_flow(&self, flow: &NetworkFlow) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Source IP (hash)
        let src_hv = BinaryHV::from_hash(flow.src_ip.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["src_ip"].bind(&src_hv));

        // Destination IP (hash)
        let dst_hv = BinaryHV::from_hash(flow.dst_ip.as_bytes(), DIMENSION);
        acc.add(&self.field_vectors["dst_ip"].bind(&dst_hv));

        // Source port (scalar)
        let src_port_vec = self.scalar_bases["port"].permute_words(flow.src_port as usize % 157);
        acc.add(&self.field_vectors["src_port"].bind(&src_port_vec));

        // Destination port (scalar)
        let dst_port_vec = self.scalar_bases["port"].permute_words(flow.dst_port as usize % 157);
        acc.add(&self.field_vectors["dst_port"].bind(&dst_port_vec));

        // Protocol
        let proto_vec = &self.protocol_vectors[&flow.protocol];
        acc.add(&self.field_vectors["protocol"].bind(proto_vec));

        // Bytes (log-scale)
        let bytes_total = flow.bytes_sent + flow.bytes_recv;
        let bytes_shift = if bytes_total > 0 {
            ((bytes_total as f64).log10() * 20.0) as usize % 157
        } else {
            0
        };
        let bytes_vec = self.scalar_bases["bytes"].permute_words(bytes_shift);
        acc.add(&self.field_vectors["bytes"].bind(&bytes_vec));

        // Duration (log-scale)
        let dur_shift = if flow.duration_ms > 0 {
            ((flow.duration_ms as f64).log10() * 30.0) as usize % 157
        } else {
            0
        };
        let dur_vec = self.scalar_bases["duration"].permute_words(dur_shift);
        acc.add(&self.field_vectors["duration"].bind(&dur_vec));

        acc.threshold()
    }

    /// Encode a security event
    pub fn encode_event(&self, event: &SecurityEvent) -> BinaryHV {
        let mut acc = BundleAccumulator::new(DIMENSION);

        // Event type
        let type_hv = BinaryHV::from_hash(event.event_type.as_bytes(), DIMENSION);
        acc.add(&type_hv);

        // Severity
        let sev_vec = &self.severity_vectors[&event.severity];
        acc.add(&self.field_vectors["severity"].bind(sev_vec));

        // Tactics
        for tactic in &event.tactics {
            acc.add(&self.tactic_vectors[tactic]);
        }

        // Matched IOCs
        for ioc_id in &event.matched_iocs {
            acc.add(&BinaryHV::from_hash(ioc_id.as_bytes(), DIMENSION));
        }

        acc.threshold()
    }
}

// ============================================================================
// Threat Intelligence Database
// ============================================================================

/// Holographic Threat Intelligence Database
pub struct ThreatIntelDb {
    /// All IOCs bundled by type
    ioc_bundles: HashMap<IocType, BundleAccumulator>,
    /// All IOC vectors for similarity search
    ioc_vectors: HashMap<String, BinaryHV>,
    /// Malware family bundles
    malware_bundles: HashMap<String, BundleAccumulator>,
    /// Individual malware vectors
    malware_vectors: HashMap<String, BinaryHV>,
    /// Known-bad flow patterns
    bad_flow_bundle: BundleAccumulator,
    /// Encoder
    encoder: CyberLink,
}

impl ThreatIntelDb {
    pub fn new() -> Self {
        Self {
            ioc_bundles: HashMap::new(),
            ioc_vectors: HashMap::new(),
            malware_bundles: HashMap::new(),
            malware_vectors: HashMap::new(),
            bad_flow_bundle: BundleAccumulator::new(DIMENSION),
            encoder: CyberLink::new(),
        }
    }

    /// Add an IOC to the database
    pub fn add_ioc(&mut self, ioc: &Ioc) {
        let hv = self.encoder.encode_ioc(ioc);

        // Add to type bundle
        let bundle = self
            .ioc_bundles
            .entry(ioc.ioc_type)
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        // Store individual vector
        self.ioc_vectors.insert(ioc.id.clone(), hv);
    }

    /// Add a malware sample
    pub fn add_malware(&mut self, sample: &MalwareSample) {
        let hv = self.encoder.encode_malware(sample);

        // Add to family bundle
        let bundle = self
            .malware_bundles
            .entry(sample.family.clone())
            .or_insert_with(|| BundleAccumulator::new(DIMENSION));
        bundle.add(&hv);

        // Store individual vector
        self.malware_vectors.insert(sample.sha256.clone(), hv);
    }

    /// Add a known-bad flow pattern
    pub fn add_bad_flow(&mut self, flow: &NetworkFlow) {
        let hv = self.encoder.encode_flow(flow);
        self.bad_flow_bundle.add(&hv);
    }

    /// Check if an IOC matches known threats (O(1) similarity)
    pub fn check_ioc(&self, ioc: &Ioc) -> Option<ThreatMatch> {
        let query_hv = self.encoder.encode_ioc(ioc);

        // Check against bundle for this IOC type
        if let Some(bundle) = self.ioc_bundles.get(&ioc.ioc_type) {
            let bundle_hv = bundle.threshold();
            let similarity = query_hv.similarity(&bundle_hv);

            if similarity > 0.6 {
                return Some(ThreatMatch {
                    match_type: MatchType::IocMatch,
                    similarity,
                    matched_ids: self.find_similar_iocs(&query_hv, 0.7),
                });
            }
        }

        None
    }

    /// Check if a malware sample is similar to known families
    pub fn check_malware(&self, sample: &MalwareSample) -> Option<ThreatMatch> {
        let query_hv = self.encoder.encode_malware(sample);

        // Check each family
        let mut best_match: Option<(String, f32)> = None;

        for (family, bundle) in &self.malware_bundles {
            let bundle_hv = bundle.threshold();
            let similarity = query_hv.similarity(&bundle_hv);

            if similarity > 0.65 {
                if best_match.is_none() || similarity > best_match.as_ref().unwrap().1 {
                    best_match = Some((family.clone(), similarity));
                }
            }
        }

        best_match.map(|(family, similarity)| ThreatMatch {
            match_type: MatchType::MalwareFamily(family),
            similarity,
            matched_ids: vec![],
        })
    }

    /// Check if a flow matches known-bad patterns
    pub fn check_flow(&self, flow: &NetworkFlow) -> f32 {
        let query_hv = self.encoder.encode_flow(flow);
        let bundle_hv = self.bad_flow_bundle.threshold();
        query_hv.similarity(&bundle_hv)
    }

    /// Find similar IOCs
    fn find_similar_iocs(&self, query: &BinaryHV, threshold: f32) -> Vec<String> {
        self.ioc_vectors
            .iter()
            .filter(|(_, hv)| query.similarity(hv) > threshold)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get IOC count
    pub fn ioc_count(&self) -> usize {
        self.ioc_vectors.len()
    }

    /// Get malware sample count
    pub fn malware_count(&self) -> usize {
        self.malware_vectors.len()
    }
}

impl Default for ThreatIntelDb {
    fn default() -> Self {
        Self::new()
    }
}

/// A threat match result
#[derive(Debug, Clone)]
pub struct ThreatMatch {
    pub match_type: MatchType,
    pub similarity: f32,
    pub matched_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum MatchType {
    IocMatch,
    MalwareFamily(String),
    FlowAnomaly,
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Network anomaly detector using holographic baseline
pub struct AnomalyDetector {
    /// Normal traffic baseline (bundled flows)
    baseline: BundleAccumulator,
    /// Observation count
    observation_count: usize,
    /// Anomaly threshold (lower = more sensitive)
    threshold: f32,
    /// Encoder
    encoder: CyberLink,
}

impl AnomalyDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            baseline: BundleAccumulator::new(DIMENSION),
            observation_count: 0,
            threshold,
            encoder: CyberLink::new(),
        }
    }

    /// Learn from normal traffic
    pub fn learn(&mut self, flow: &NetworkFlow) {
        let hv = self.encoder.encode_flow(flow);
        self.baseline.add(&hv);
        self.observation_count += 1;
    }

    /// Check if a flow is anomalous
    pub fn is_anomalous(&self, flow: &NetworkFlow) -> (bool, f32) {
        if self.observation_count < 100 {
            return (false, 1.0); // Not enough baseline data
        }

        let query_hv = self.encoder.encode_flow(flow);
        let baseline_hv = self.baseline.threshold();
        let similarity = query_hv.similarity(&baseline_hv);

        (similarity < self.threshold, similarity)
    }

    /// Get observation count
    pub fn observations(&self) -> usize {
        self.observation_count
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ioc_encoding() {
        let link = CyberLink::new();

        let ioc = Ioc {
            id: "ioc_001".to_string(),
            ioc_type: IocType::IpAddress,
            value: "192.168.1.100".to_string(),
            severity: ThreatSeverity::High,
            tags: vec!["c2".to_string(), "emotet".to_string()],
            first_seen: 1000,
            last_seen: 2000,
            confidence: 0.95,
        };

        let hv = link.encode_ioc(&ioc);

        // Should produce non-trivial vector
        let density = hv.similarity(&BinaryHV::from_hash(b"random", DIMENSION));
        assert!(density > 0.3 && density < 0.7);
    }

    #[test]
    fn test_similar_iocs_high_similarity() {
        let link = CyberLink::new();

        let ioc1 = Ioc {
            id: "ioc_001".to_string(),
            ioc_type: IocType::Domain,
            value: "malware.evil.com".to_string(),
            severity: ThreatSeverity::Critical,
            tags: vec!["c2".to_string()],
            first_seen: 1000,
            last_seen: 2000,
            confidence: 0.9,
        };

        let ioc2 = Ioc {
            id: "ioc_002".to_string(),
            ioc_type: IocType::Domain,
            value: "malware.evil.com".to_string(), // Same value
            severity: ThreatSeverity::Critical,
            tags: vec!["c2".to_string()],
            first_seen: 1100,
            last_seen: 2100,
            confidence: 0.85,
        };

        let hv1 = link.encode_ioc(&ioc1);
        let hv2 = link.encode_ioc(&ioc2);

        let similarity = hv1.similarity(&hv2);
        println!("Same IOC similarity: {}", similarity);
        assert!(similarity > 0.8, "Same IOC should have high similarity");
    }

    #[test]
    fn test_threat_intel_db() {
        let mut db = ThreatIntelDb::new();

        // Add known malicious IP
        db.add_ioc(&Ioc {
            id: "bad_ip_1".to_string(),
            ioc_type: IocType::IpAddress,
            value: "10.0.0.1".to_string(),
            severity: ThreatSeverity::Critical,
            tags: vec!["c2".to_string()],
            first_seen: 1000,
            last_seen: 2000,
            confidence: 0.99,
        });

        assert_eq!(db.ioc_count(), 1);

        // Check same IP
        let result = db.check_ioc(&Ioc {
            id: "query".to_string(),
            ioc_type: IocType::IpAddress,
            value: "10.0.0.1".to_string(),
            severity: ThreatSeverity::High,
            tags: vec![],
            first_seen: 3000,
            last_seen: 3000,
            confidence: 0.8,
        });

        assert!(result.is_some());
    }

    #[test]
    fn test_malware_similarity() {
        let mut db = ThreatIntelDb::new();

        // Add known emotet sample
        db.add_malware(&MalwareSample {
            sha256: "abc123".to_string(),
            sha1: "def".to_string(),
            md5: "ghi".to_string(),
            family: "emotet".to_string(),
            variant: Some("v4".to_string()),
            file_type: "PE32".to_string(),
            file_size: 50000,
            tactics: vec![MitreTactic::InitialAccess, MitreTactic::Execution],
            techniques: vec!["T1566".to_string()],
            iocs: vec![],
            first_seen: 1000,
        });

        // Check similar sample (same family, different hash)
        let result = db.check_malware(&MalwareSample {
            sha256: "xyz789".to_string(),
            sha1: "aaa".to_string(),
            md5: "bbb".to_string(),
            family: "emotet".to_string(),
            variant: Some("v5".to_string()),
            file_type: "PE32".to_string(),
            file_size: 51000,
            tactics: vec![MitreTactic::InitialAccess, MitreTactic::Execution],
            techniques: vec!["T1566".to_string()],
            iocs: vec![],
            first_seen: 2000,
        });

        assert!(result.is_some());
        if let Some(m) = result {
            println!("Emotet match similarity: {}", m.similarity);
        }
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = AnomalyDetector::new(0.4);

        // Learn normal HTTP traffic
        for i in 0..150 {
            detector.learn(&NetworkFlow {
                src_ip: format!("192.168.1.{}", i % 50),
                dst_ip: "8.8.8.8".to_string(),
                src_port: 50000 + (i as u16),
                dst_port: 443,
                protocol: Protocol::Tcp,
                bytes_sent: 1000,
                bytes_recv: 5000,
                packets: 10,
                duration_ms: 100,
                timestamp: i as u64,
            });
        }

        // Check anomalous flow (unusual port, unusual protocol)
        let (is_anomaly, sim) = detector.is_anomalous(&NetworkFlow {
            src_ip: "192.168.1.1".to_string(),
            dst_ip: "1.2.3.4".to_string(),
            src_port: 12345,
            dst_port: 4444, // Unusual port
            protocol: Protocol::Tcp,
            bytes_sent: 1000000, // Large transfer
            bytes_recv: 100,
            packets: 1000,
            duration_ms: 1,
            timestamp: 1000,
        });

        println!(
            "Anomaly check: is_anomaly={}, similarity={}",
            is_anomaly, sim
        );
    }
}
