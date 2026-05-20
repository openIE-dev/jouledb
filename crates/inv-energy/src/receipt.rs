//! Unified energy receipt returned with every API response.
//!
//! The `EnergyReceipt` is the atomic unit of energy transparency in Invisible
//! Infrastructure. Every API call, database query, function invocation, and
//! managed-service operation returns one, making energy consumption visible,
//! auditable, and verifiable.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Silicon type that executed the operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SiliconType {
    #[default]
    Cpu,
    Gpu,
    Npu,
    Tpu,
    Fpga,
    /// Apple Neural Engine / Apple Silicon unified memory.
    AppleSilicon,
    /// ARM-based edge device.
    Arm,
    /// WebAssembly (silicon-agnostic execution).
    Wasm,
}

/// Memory tier that served the operation's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    #[default]
    Dram,
    /// CXL-attached near memory (~200ns latency).
    CxlNear,
    /// CXL-attached far memory (~1μs latency).
    CxlFar,
    /// NVMe / persistent storage.
    Nvme,
    /// In-cache (L1/L2/L3).
    Cache,
}

/// Measurement source for the energy value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementSource {
    /// Measured via eBPF PMU / RAPL hardware counters.
    Kernel,
    /// Estimated from WASM fuel consumption.
    #[default]
    FuelEstimate,
    /// Estimated from hardware TDP model.
    TdpModel,
    /// Measured via NVML (NVIDIA GPU).
    Nvml,
    /// Measured via Apple powermetrics.
    ApplePowermetrics,
}

/// Unified energy receipt attached to every API response.
///
/// This is the design guide's core primitive: every operation returns
/// `{joules, carbon_gco2eq, silicon, memory_tier, measurement_source}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyReceipt {
    /// Energy consumed by this operation (joules).
    pub energy_joules: f64,
    /// Carbon emissions for this operation (grams CO2 equivalent).
    pub carbon_gco2eq: f64,
    /// Silicon type that executed the operation.
    pub silicon: SiliconType,
    /// Memory tier that served the data.
    pub memory_tier: MemoryTier,
    /// How the energy value was measured.
    pub measurement_source: MeasurementSource,
    /// Node that executed the operation.
    pub node_id: String,
    /// Region of the executing node.
    pub region: String,
    /// Timestamp (nanoseconds since epoch).
    pub timestamp_ns: u64,
    /// SHA-256 hash of the receipt fields for third-party attestation.
    pub audit_hash: String,
}

impl EnergyReceipt {
    /// Create a new energy receipt with computed audit hash.
    pub fn new(
        energy_joules: f64,
        carbon_gco2eq: f64,
        silicon: SiliconType,
        memory_tier: MemoryTier,
        measurement_source: MeasurementSource,
        node_id: String,
        region: String,
    ) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut receipt = Self {
            energy_joules,
            carbon_gco2eq,
            silicon,
            memory_tier,
            measurement_source,
            node_id,
            region,
            timestamp_ns,
            audit_hash: String::new(),
        };
        receipt.audit_hash = receipt.compute_audit_hash();
        receipt
    }

    /// Create a minimal receipt for lightweight operations.
    pub fn estimate(energy_joules: f64, node_id: &str, region: &str) -> Self {
        Self::new(
            energy_joules,
            energy_joules * 0.000_233, // global average: 233 gCO2/kWh → per-joule
            SiliconType::Cpu,
            MemoryTier::Dram,
            MeasurementSource::FuelEstimate,
            node_id.to_string(),
            region.to_string(),
        )
    }

    /// Create a zero-cost receipt (e.g. cache hits, no-ops).
    pub fn zero(node_id: &str, region: &str) -> Self {
        Self::new(
            0.0,
            0.0,
            SiliconType::Cpu,
            MemoryTier::Cache,
            MeasurementSource::FuelEstimate,
            node_id.to_string(),
            region.to_string(),
        )
    }

    /// Compute SHA-256 audit hash over all receipt fields.
    pub fn compute_audit_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.energy_joules.to_le_bytes());
        hasher.update(self.carbon_gco2eq.to_le_bytes());
        hasher.update(format!("{:?}", self.silicon).as_bytes());
        hasher.update(format!("{:?}", self.memory_tier).as_bytes());
        hasher.update(format!("{:?}", self.measurement_source).as_bytes());
        hasher.update(self.node_id.as_bytes());
        hasher.update(self.region.as_bytes());
        hasher.update(self.timestamp_ns.to_le_bytes());
        format!("sha256:{}", hex::encode(hasher.finalize()))
    }

    /// Verify the audit hash matches the receipt data.
    pub fn verify(&self) -> bool {
        self.audit_hash == self.compute_audit_hash()
    }

    /// Merge two receipts (e.g. for multi-step operations).
    pub fn merge(&self, other: &EnergyReceipt) -> Self {
        Self::new(
            self.energy_joules + other.energy_joules,
            self.carbon_gco2eq + other.carbon_gco2eq,
            self.silicon,
            self.memory_tier,
            self.measurement_source,
            self.node_id.clone(),
            self.region.clone(),
        )
    }
}

impl Default for EnergyReceipt {
    fn default() -> Self {
        Self::new(
            0.0,
            0.0,
            SiliconType::default(),
            MemoryTier::default(),
            MeasurementSource::default(),
            "local".to_string(),
            "unknown".to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_creation() {
        let receipt = EnergyReceipt::new(
            0.05,
            0.012,
            SiliconType::Gpu,
            MemoryTier::Dram,
            MeasurementSource::Nvml,
            "node-1".to_string(),
            "us-east".to_string(),
        );
        assert!((receipt.energy_joules - 0.05).abs() < 1e-10);
        assert!(!receipt.audit_hash.is_empty());
        assert!(receipt.audit_hash.starts_with("sha256:"));
    }

    #[test]
    fn receipt_verification() {
        let receipt = EnergyReceipt::estimate(0.1, "node-1", "eu-west");
        assert!(receipt.verify());
    }

    #[test]
    fn tampered_receipt_fails_verification() {
        let mut receipt = EnergyReceipt::estimate(0.1, "node-1", "eu-west");
        receipt.energy_joules = 0.001; // tamper
        assert!(!receipt.verify());
    }

    #[test]
    fn receipt_merge() {
        let a = EnergyReceipt::estimate(0.1, "node-1", "us-east");
        let b = EnergyReceipt::estimate(0.2, "node-1", "us-east");
        let merged = a.merge(&b);
        assert!((merged.energy_joules - 0.3).abs() < 1e-10);
        assert!(merged.verify());
    }

    #[test]
    fn zero_receipt() {
        let r = EnergyReceipt::zero("node-1", "us-east");
        assert!((r.energy_joules).abs() < 1e-10);
        assert!(r.verify());
    }

    #[test]
    fn receipt_serialization() {
        let receipt = EnergyReceipt::estimate(0.05, "node-1", "eu-west");
        let json = serde_json::to_string(&receipt).unwrap();
        let deserialized: EnergyReceipt = serde_json::from_str(&json).unwrap();
        assert!((deserialized.energy_joules - 0.05).abs() < 1e-10);
        assert_eq!(deserialized.audit_hash, receipt.audit_hash);
    }

    #[test]
    fn silicon_type_serde() {
        let json = serde_json::to_string(&SiliconType::Npu).unwrap();
        assert_eq!(json, "\"npu\"");
    }

    #[test]
    fn memory_tier_serde() {
        let json = serde_json::to_string(&MemoryTier::CxlNear).unwrap();
        assert_eq!(json, "\"cxl_near\"");
    }
}
