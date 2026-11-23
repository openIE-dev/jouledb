use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

/// Controlled vocabulary for query execution stages.
///
/// Each stage represents a distinct phase where energy is consumed.
/// Using an enum ensures receipts are comparable across deployments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStage {
    /// SQL parsing and AST construction.
    Parse,
    /// Query planning and optimization.
    Plan,
    /// Query execution (table scan, index lookup, join, etc.).
    Execute,
    /// Disk/network I/O for data retrieval.
    Io,
    /// Result serialization (JSON, binary protocol, etc.).
    Serialize,
    /// HDC encoding/binding operations.
    HdcEncode,
    /// GPU dispatch and computation.
    GpuCompute,
    /// Replication overhead (WAL, Raft).
    Replicate,
}

impl ExecutionStage {
    /// All known stages, for iteration.
    pub const ALL: &'static [ExecutionStage] = &[
        Self::Parse,
        Self::Plan,
        Self::Execute,
        Self::Io,
        Self::Serialize,
        Self::HdcEncode,
        Self::GpuCompute,
        Self::Replicate,
    ];
}

impl fmt::Display for ExecutionStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse => write!(f, "parse"),
            Self::Plan => write!(f, "plan"),
            Self::Execute => write!(f, "execute"),
            Self::Io => write!(f, "io"),
            Self::Serialize => write!(f, "serialize"),
            Self::HdcEncode => write!(f, "hdc_encode"),
            Self::GpuCompute => write!(f, "gpu_compute"),
            Self::Replicate => write!(f, "replicate"),
        }
    }
}

/// Provenance tier for how an energy figure was obtained.
///
/// Matches the `Acquisition` contract in `_SOTA-CHANGELOG.md`: a receipt must
/// never present a modeled or estimated value as hardware-measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionTier {
    /// Sampled directly from hardware counters (RAPL / Apple IOReport / NVML / INA226).
    Measured,
    /// Derived from runtime telemetry proxy (e.g., measured wall/CPU time x calibrated TDP).
    Estimated,
    /// Analytical / LUT op-count model (default when hardware telemetry is absent).
    #[default]
    Modeled,
}

/// Deterministic cascade tier that satisfied the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CascadeTierTag {
    Lookup,
    Formula,
    Extract,
    Aggregate,
    Reason,
}

/// Per-channel silicon energy telemetry breakdown (in joules) + Landauer lower bound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct HardwareChannelJoules {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub npu_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dram_joules: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bits_erased: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub landauer_floor_joules: Option<f64>,
}

/// Strictly segregated external cloud/frontier API usage (Tier 4 escape hatch).
///
/// Closes Whitepaper v0.2 §6 "The Ugly" #3: cloud API dollars and modeled remote
/// datacenter energy are tracked explicitly here and NEVER silently added into
/// local `Measured` silicon joules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternalApiSpend {
    pub provider: String,
    pub model: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modeled_remote_joules: Option<f64>,
}

/// Optional cryptographic signature (e.g. NIST FIPS 204 ML-DSA PQC or Ed25519)
/// over the canonical `content_hash()` of this receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestation {
    /// Signature algorithm identifier (e.g. `"ml-dsa-65"`, `"ed25519"`, `"hmac-sha256"`).
    pub scheme: String,
    /// Hex-encoded signer public key or key ID.
    pub signer_id: String,
    /// Hex-encoded signature over the 32-byte canonical `content_hash` (excluding `attestation`).
    pub signature_hex: String,
}

/// Rich SOTA energy provenance envelope attached to a [`LedgerEnergyReceipt`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct EnergyProvenance {
    /// How `energy_joules_total` (local silicon energy) was acquired.
    pub acquisition: AcquisitionTier,
    /// Which deterministic cascade tier produced the result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cascade_tier: Option<CascadeTierTag>,
    /// Optional per-rail silicon energy split (CPU / GPU / NPU / DRAM / Landauer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<HardwareChannelJoules>,
    /// Segregated Tier-4 external cloud API spend (never conflated with silicon joules).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_api: Option<ExternalApiSpend>,
    /// Optional cryptographic signature attesting receipt authenticity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation: Option<ReceiptAttestation>,
}

/// Provenance validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    NegativeSiliconEnergy,
    NegativeExternalCost,
    MeasuredSiliconZeroWithPositiveStageSum,
    ChannelSumExceedsTotalSiliconEnergy,
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeSiliconEnergy => write!(f, "local silicon energy cannot be negative"),
            Self::NegativeExternalCost => write!(f, "external API USD cost cannot be negative"),
            Self::MeasuredSiliconZeroWithPositiveStageSum => {
                write!(f, "total silicon energy is 0 while stage breakdown is positive")
            }
            Self::ChannelSumExceedsTotalSiliconEnergy => {
                write!(f, "hardware channel sum exceeds reported total silicon energy")
            }
        }
    }
}

impl std::error::Error for ProvenanceError {}

/// Frozen per-query energy receipt — the canonical attestation unit.
///
/// Each query executed through the energy-aware executor produces one receipt.
/// The `receipt_id` is deterministic (SHA-256 of qid + tenant + timestamp),
/// ensuring the same execution always yields the same ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEnergyReceipt {
    /// Deterministic receipt ID: SHA-256(qid || tenant_id || timestamp_start_ns).
    pub receipt_id: String,
    /// Query identifier (session ID or server-assigned UUID).
    pub qid: String,
    /// Tenant / user identifier.
    pub tenant_id: String,
    /// Optional workload tag (e.g., "analytics", "oltp", "vector-search").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workload_tag: Option<String>,
    /// Branch the query executed on (None = "main").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_id: Option<String>,

    // --- Energy fields (local silicon / host execution ONLY) ---
    /// Total local silicon energy consumed in joules.
    pub energy_joules_total: f64,
    /// Energy breakdown by execution stage (parse, plan, execute, serialize).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub energy_joules_by_stage: HashMap<String, f64>,
    /// Energy in kilowatt-hours (derived from joules).
    pub kwh: f64,
    /// Carbon emissions estimate in kg CO2e.
    pub kg_co2e: f64,

    // --- Grid context ---
    /// ISO region / balancing authority (e.g., "US-CAL-CISO").
    pub grid_region: String,
    /// Source of the grid carbon factor (e.g., "electricity-maps-2025").
    pub grid_factor_source: String,

    // --- Timing ---
    /// Query start timestamp (UTC).
    pub timestamp_start: DateTime<Utc>,
    /// Query end timestamp (UTC).
    pub timestamp_end: DateTime<Utc>,

    // --- Device context ---
    /// Device that executed the query (cpu, gpu, npu, tpu).
    pub device_target: String,
    /// Algorithm classification (btree, scan, hdc, columnar, holographic).
    pub algorithm_type: String,

    // --- SOTA Provenance & External Spend Split ---
    /// Optional provenance envelope separating `Measured|Estimated|Modeled` silicon
    /// energy from external cloud API spend and carrying PQC/Ed25519 attestations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<EnergyProvenance>,
}

impl LedgerEnergyReceipt {
    /// Compute the deterministic receipt ID from query identifiers.
    pub fn compute_id(qid: &str, tenant_id: &str, timestamp_start: &DateTime<Utc>) -> String {
        let mut hasher = Sha256::new();
        hasher.update(qid.as_bytes());
        hasher.update(tenant_id.as_bytes());
        hasher.update(
            timestamp_start
                .timestamp_nanos_opt()
                .unwrap_or(0)
                .to_be_bytes(),
        );
        hex::encode(hasher.finalize())
    }

    /// Set the per-stage energy breakdown using typed stages.
    pub fn set_stage_energy(&mut self, stages: HashMap<ExecutionStage, f64>) {
        self.energy_joules_by_stage = stages
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    }

    /// Get the per-stage energy breakdown with typed stages.
    /// Entries with unrecognized keys are silently skipped.
    pub fn typed_stage_energy(&self) -> HashMap<ExecutionStage, f64> {
        self.energy_joules_by_stage
            .iter()
            .filter_map(|(k, v)| {
                serde_json::from_value::<ExecutionStage>(serde_json::Value::String(k.clone()))
                    .ok()
                    .map(|stage| (stage, *v))
            })
            .collect()
    }

    /// Canonical SHA-256 hash of the receipt content (used as Merkle leaf).
    ///
    /// Recursively sorts all JSON object keys (`energy_joules_by_stage`, nested maps)
    /// so `HashMap` iteration order never causes cross-process hash divergence.
    /// Excludes any outer `attestation` signature so signatures can be verified
    /// directly against `signable_hash()`.
    pub fn content_hash(&self) -> [u8; 32] {
        let val = serde_json::to_value(self).expect("receipt value serialization");
        let canonical = canonicalize_json_value(val);
        let json = serde_json::to_string(&canonical).expect("canonical receipt serialization");
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        hasher.finalize().into()
    }

    /// Canonical SHA-256 hash excluding `provenance.attestation`, suitable as the
    /// message digest signed by PQC (`ML-DSA`) or Ed25519 signers.
    pub fn signable_hash(&self) -> [u8; 32] {
        let mut clone = self.clone();
        if let Some(ref mut prov) = clone.provenance {
            prov.attestation = None;
        }
        clone.content_hash()
    }

    /// Validate receipt provenance invariants (non-negative silicon/USD figures and
    /// consistent rail-to-total accounting).
    pub fn verify_provenance(&self) -> Result<(), ProvenanceError> {
        if self.energy_joules_total < 0.0 || self.energy_joules_total.is_nan() {
            return Err(ProvenanceError::NegativeSiliconEnergy);
        }
        let stage_sum: f64 = self.energy_joules_by_stage.values().copied().sum();
        if self.energy_joules_total == 0.0 && stage_sum > 1e-12 {
            return Err(ProvenanceError::MeasuredSiliconZeroWithPositiveStageSum);
        }
        if let Some(ref prov) = self.provenance {
            if let Some(ref ext) = prov.external_api {
                if ext.cost_usd < 0.0 || ext.cost_usd.is_nan() {
                    return Err(ProvenanceError::NegativeExternalCost);
                }
            }
            if let Some(ref ch) = prov.channels {
                let rail_sum = ch.cpu_joules.unwrap_or(0.0)
                    + ch.gpu_joules.unwrap_or(0.0)
                    + ch.npu_joules.unwrap_or(0.0)
                    + ch.dram_joules.unwrap_or(0.0);
                if rail_sum > self.energy_joules_total + 1e-9 {
                    return Err(ProvenanceError::ChannelSumExceedsTotalSiliconEnergy);
                }
            }
        }
        Ok(())
    }
}

fn canonicalize_json_value(val: serde_json::Value) -> serde_json::Value {
    match val {
        serde_json::Value::Object(map) => {
            let mut sorted: Vec<(String, serde_json::Value)> = map
                .into_iter()
                .map(|(k, v)| (k, canonicalize_json_value(v)))
                .collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = serde_json::Map::with_capacity(sorted.len());
            for (k, v) in sorted {
                out.insert(k, v);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(canonicalize_json_value).collect())
        }
        primitive => primitive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sample_receipt() -> LedgerEnergyReceipt {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        LedgerEnergyReceipt {
            receipt_id: LedgerEnergyReceipt::compute_id("q1", "tenant1", &ts),
            qid: "q1".to_string(),
            tenant_id: "tenant1".to_string(),
            workload_tag: Some("oltp".to_string()),
            branch_id: None,
            energy_joules_total: 0.005,
            energy_joules_by_stage: HashMap::new(),
            kwh: 0.005 / 3_600_000.0,
            kg_co2e: (0.005 / 3_600_000.0) * 0.4,
            grid_region: "US-CAL-CISO".to_string(),
            grid_factor_source: "default-world-average-2025".to_string(),
            timestamp_start: ts,
            timestamp_end: ts + chrono::Duration::milliseconds(50),
            device_target: "cpu".to_string(),
            algorithm_type: "btree".to_string(),
            provenance: None,
        }
    }

    #[test]
    fn deterministic_id() {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let id1 = LedgerEnergyReceipt::compute_id("q1", "tenant1", &ts);
        let id2 = LedgerEnergyReceipt::compute_id("q1", "tenant1", &ts);
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn different_inputs_different_ids() {
        let ts = Utc.with_ymd_and_hms(2026, 2, 24, 12, 0, 0).unwrap();
        let id1 = LedgerEnergyReceipt::compute_id("q1", "tenant1", &ts);
        let id2 = LedgerEnergyReceipt::compute_id("q2", "tenant1", &ts);
        let id3 = LedgerEnergyReceipt::compute_id("q1", "tenant2", &ts);
        assert_ne!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn content_hash_deterministic() {
        let r = sample_receipt();
        let h1 = r.content_hash();
        let h2 = r.content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_changes_with_data() {
        let r1 = sample_receipt();
        let mut r2 = sample_receipt();
        r2.energy_joules_total = 0.010;
        assert_ne!(r1.content_hash(), r2.content_hash());
    }

    #[test]
    fn stage_enum_serde_roundtrip() {
        for stage in ExecutionStage::ALL {
            let json = serde_json::to_string(stage).unwrap();
            let deserialized: ExecutionStage = serde_json::from_str(&json).unwrap();
            assert_eq!(*stage, deserialized);
        }
    }

    #[test]
    fn set_stage_energy_populates_hashmap() {
        let mut r = sample_receipt();
        let mut stages = HashMap::new();
        stages.insert(ExecutionStage::Parse, 0.001);
        stages.insert(ExecutionStage::Execute, 0.004);
        r.set_stage_energy(stages);
        assert_eq!(r.energy_joules_by_stage.len(), 2);
        assert!(r.energy_joules_by_stage.contains_key("parse"));
        assert!(r.energy_joules_by_stage.contains_key("execute"));
    }

    #[test]
    fn typed_stage_energy_roundtrip() {
        let mut r = sample_receipt();
        let mut stages = HashMap::new();
        stages.insert(ExecutionStage::Io, 0.002);
        stages.insert(ExecutionStage::Replicate, 0.001);
        r.set_stage_energy(stages.clone());
        let typed = r.typed_stage_energy();
        assert_eq!(typed.len(), 2);
        assert!((typed[&ExecutionStage::Io] - 0.002).abs() < 1e-15);
        assert!((typed[&ExecutionStage::Replicate] - 0.001).abs() < 1e-15);
    }

    #[test]
    fn typed_stage_energy_ignores_unknown() {
        let mut r = sample_receipt();
        r.energy_joules_by_stage.insert("parse".to_string(), 0.001);
        r.energy_joules_by_stage
            .insert("custom_unknown_stage".to_string(), 0.009);
        let typed = r.typed_stage_energy();
        assert_eq!(typed.len(), 1);
        assert!(typed.contains_key(&ExecutionStage::Parse));
    }

    #[test]
    fn all_stages_constant() {
        assert_eq!(ExecutionStage::ALL.len(), 8);
    }

    #[test]
    fn stage_display_format() {
        assert_eq!(ExecutionStage::Parse.to_string(), "parse");
        assert_eq!(ExecutionStage::HdcEncode.to_string(), "hdc_encode");
        assert_eq!(ExecutionStage::GpuCompute.to_string(), "gpu_compute");
    }

    #[test]
    fn serde_roundtrip() {
        let r = sample_receipt();
        let json = serde_json::to_string(&r).unwrap();
        let r2: LedgerEnergyReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(r.receipt_id, r2.receipt_id);
        assert_eq!(r.qid, r2.qid);
        assert!((r.energy_joules_total - r2.energy_joules_total).abs() < 1e-15);
    }

    #[test]
    fn provenance_segregates_silicon_and_external_api_and_hashes_canonically() {
        let mut r1 = sample_receipt();
        r1.energy_joules_by_stage.insert("parse".into(), 0.001);
        r1.energy_joules_by_stage.insert("execute".into(), 0.004);
        r1.provenance = Some(EnergyProvenance {
            acquisition: AcquisitionTier::Measured,
            cascade_tier: Some(CascadeTierTag::Reason),
            channels: Some(HardwareChannelJoules {
                cpu_joules: Some(0.002),
                gpu_joules: Some(0.002),
                npu_joules: None,
                dram_joules: Some(0.001),
                bits_erased: Some(4096),
                landauer_floor_joules: Some(1.17e-17),
            }),
            external_api: Some(ExternalApiSpend {
                provider: "frontier-cloud".into(),
                model: "tier4-escape".into(),
                tokens_in: 512,
                tokens_out: 128,
                cost_usd: 0.0042,
                modeled_remote_joules: Some(18.5),
            }),
            attestation: None,
        });
        assert!(r1.verify_provenance().is_ok());
        // Local silicon total remains 0.005 J, never polluted by 18.5 J remote estimate
        assert!((r1.energy_joules_total - 0.005).abs() < 1e-15);

        // Reverse insertion order into HashMap must yield bit-identical Merkle leaf hash
        let mut r2 = r1.clone();
        r2.energy_joules_by_stage.clear();
        r2.energy_joules_by_stage.insert("execute".into(), 0.004);
        r2.energy_joules_by_stage.insert("parse".into(), 0.001);
        assert_eq!(r1.content_hash(), r2.content_hash());

        // Signable hash ignores outer signature
        let pre_sig_hash = r1.signable_hash();
        r1.provenance.as_mut().unwrap().attestation = Some(ReceiptAttestation {
            scheme: "ml-dsa-65".into(),
            signer_id: "node-01".into(),
            signature_hex: "deadbeef".into(),
        });
        assert_eq!(r1.signable_hash(), pre_sig_hash);
    }
}
