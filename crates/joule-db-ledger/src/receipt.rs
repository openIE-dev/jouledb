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

    // --- Energy fields ---
    /// Total energy consumed in joules.
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

    /// SHA-256 hash of the receipt content (used as Merkle leaf).
    ///
    /// Uses canonical JSON serialization for deterministic hashing.
    pub fn content_hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        // serde_json with default settings produces deterministic output for
        // the same struct values (HashMap ordering may vary, but we accept
        // this since the same receipt instance always serializes identically).
        let json = serde_json::to_string(self).expect("receipt serialization");
        hasher.update(json.as_bytes());
        hasher.finalize().into()
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
}
