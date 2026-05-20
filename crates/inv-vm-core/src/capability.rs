use serde::{Deserialize, Serialize};

use crate::energy::{EnergySource, ThermalState, Watts};

/// A node's advertised capabilities — what it can offer to the mesh.
/// Gossiped via SWIM protocol and updated as conditions change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    /// Number of available CPU cores.
    pub cpu_cores: u32,
    /// CPU architecture.
    pub cpu_arch: CpuArch,
    /// Available memory in megabytes.
    pub memory_mb: u64,
    /// GPU info, if present.
    pub gpu: Option<GpuInfo>,
    /// Available storage in gigabytes.
    pub storage_gb: u64,
    /// Network bandwidth in megabits per second.
    pub bandwidth_mbps: u32,
    /// Power source for this node.
    pub energy_source: EnergySource,
    /// Battery percentage (0-100), if on battery.
    pub battery_pct: Option<u8>,
    /// Whether this node can run WASM workloads.
    pub wasm_support: bool,
    /// Whether this node can run OCI containers.
    pub container_support: bool,
    /// Current CPU load (0.0 = idle, 1.0 = fully loaded).
    pub current_load: f32,
    /// Current power draw.
    pub energy_rate: Watts,
    /// Current thermal state.
    pub thermal_state: ThermalState,
    /// Locality zone for data-aware scheduling.
    pub locality_zone: String,
    /// Compliance tags this node satisfies (e.g., "hipaa", "gdpr").
    pub compliance_tags: Vec<String>,
    /// Effective resources after contribution filtering.
    #[serde(default)]
    pub effective_resources: Option<EffectiveResources>,
    /// All detected accelerators (GPU, NPU, TPU, FPGA).
    #[serde(default)]
    pub accelerators: Vec<AcceleratorInfo>,
    /// Energy zone classification — drives time-of-day pricing curve.
    /// Populated by the energy scanner or mining substrate scanner.
    #[serde(default)]
    pub energy_zone: Option<String>,
    /// Live electricity cost in $/kWh (effective rate after time-of-day adjustment).
    /// When set, the scheduler uses this instead of hard-coded class defaults.
    #[serde(default)]
    pub energy_cost_usd_kwh: Option<f64>,
    /// Post-quantum readiness score (0.0 = fully vulnerable, 1.0 = fully safe).
    /// Populated by the crypto inventory scanner. Gossiped for mesh-wide visibility.
    #[serde(default)]
    pub pq_readiness: Option<f64>,
    /// Number of critical quantum-vulnerable crypto assets on this node.
    /// Zero means fully migrated to PQ-safe algorithms.
    #[serde(default)]
    pub crypto_risk_critical: Option<u32>,
}

impl Default for Capability {
    fn default() -> Self {
        Self {
            cpu_cores: 0,
            cpu_arch: CpuArch::X86_64,
            memory_mb: 0,
            gpu: None,
            storage_gb: 0,
            bandwidth_mbps: 0,
            energy_source: EnergySource::default(),
            battery_pct: None,
            wasm_support: false,
            container_support: false,
            current_load: 0.0,
            energy_rate: Watts::ZERO,
            thermal_state: ThermalState::Normal,
            locality_zone: String::new(),
            compliance_tags: vec![],
            effective_resources: None,
            accelerators: vec![],
            energy_zone: None,
            energy_cost_usd_kwh: None,
            pq_readiness: None,
            crypto_risk_critical: None,
        }
    }
}

impl Capability {
    /// Whether this node is available for new workloads.
    pub fn is_available(&self) -> bool {
        self.current_load < 0.95
            && self.thermal_state != ThermalState::Critical
            && self.battery_pct.is_none_or(|pct| pct > 5)
    }

    /// Estimated available compute capacity (0.0 to 1.0).
    pub fn available_capacity(&self) -> f32 {
        let base = 1.0 - self.current_load;
        let thermal_factor = match self.thermal_state {
            ThermalState::Normal => 1.0,
            ThermalState::OrbitalVacuum => 0.9, // radiative-only cooling, slight derating
            ThermalState::Warm => 0.8,
            ThermalState::Throttled => 0.5,
            ThermalState::Critical => 0.0,
        };
        base * thermal_factor
    }

    /// Whether this node satisfies all required compliance tags.
    pub fn satisfies_compliance(&self, required: &[String]) -> bool {
        required
            .iter()
            .all(|tag| self.compliance_tags.contains(tag))
    }
}

/// CPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CpuArch {
    X86_64,
    Aarch64,
    RiscV64,
    Wasm32,
}

impl std::fmt::Display for CpuArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64"),
            Self::Aarch64 => write!(f, "aarch64"),
            Self::RiscV64 => write!(f, "riscv64"),
            Self::Wasm32 => write!(f, "wasm32"),
        }
    }
}

/// GPU information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU model name.
    pub name: String,
    /// GPU memory in megabytes.
    pub memory_mb: u64,
    /// GPU vendor.
    pub vendor: GpuVendor,
}

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    /// Virtual silicon — emulated on commodity hardware fleet.
    Virtual,
    Other,
}

/// The class of a node in the mesh hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeClass {
    /// Cloud VM or bare metal server.
    Cloud,
    /// Edge compute node (Cloudflare Worker, etc.).
    Edge,
    /// Developer workstation (laptop, desktop).
    Workstation,
    /// Mobile device (phone, tablet).
    Mobile,
    /// IoT or embedded device.
    IoT,
    /// GPU compute node.
    Gpu,
    /// Orbital compute node (satellite with on-board processing).
    Orbital,
}

impl std::fmt::Display for NodeClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cloud => write!(f, "cloud"),
            Self::Edge => write!(f, "edge"),
            Self::Workstation => write!(f, "workstation"),
            Self::Mobile => write!(f, "mobile"),
            Self::IoT => write!(f, "iot"),
            Self::Gpu => write!(f, "gpu"),
            Self::Orbital => write!(f, "orbital"),
        }
    }
}

/// The role a node plays in the mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRole {
    /// The first node in a new mesh — generates the org CA.
    Seed,
    /// A cloud backbone node participating in Raft consensus.
    Backbone,
    /// A leaf node contributing resources but not in consensus.
    Leaf,
}

/// A gossip-propagated state for a node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeState {
    /// Node is alive and healthy.
    #[default]
    Alive,
    /// Node missed heartbeats — suspected of failure.
    Suspect,
    /// Node confirmed failed — workloads rescheduled.
    Failed,
}

/// A gossip message carrying node state and capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipMessage {
    /// The node this message is about.
    pub node_id: crate::identity::NodeId,
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Node liveness state.
    pub state: NodeState,
    /// Node capabilities (updated periodically).
    pub capabilities: Option<Capability>,
    /// Node class.
    pub node_class: NodeClass,
    /// Cumulative energy consumed by this node in µWh (from JWP frames / energy meter).
    /// Defaults to 0 for backward compatibility with older peers.
    #[serde(default)]
    pub cumulative_energy_uwh: u64,
}

// ── Accelerator types ──────────────────────────────────────────────

/// Accelerator type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorType {
    Gpu,
    Npu,
    Tpu,
    Fpga,
    /// Virtual TPU — systolic array GEMM emulated on commodity CPU fleet.
    VirtualTpu,
    /// Virtual LPU — deterministic stream processing emulated on pinned-memory workers.
    VirtualLpu,
    /// Virtual WSE — spatial dataflow pipeline emulated across mesh workers.
    VirtualWse,
    /// Virtual RDU — fused dataflow with IR chain optimization on distributed CPUs.
    VirtualRdu,
    /// Virtual IPU — BSP compute/exchange rounds on mesh workers.
    VirtualIpu,
    /// Virtual Tensix — mesh NOC with reader/compute/writer DMA model.
    VirtualTensix,
    /// Virtual FPGA — reconfigurable logic fabric emulated via JIT bitstream compilation.
    VirtualFpga,
    /// Virtual DPU — network/storage offload (encrypt, compress, route) on dedicated workers.
    VirtualDpu,
    /// Virtual PIM — processing-in-memory, compute co-located with data to eliminate memory wall.
    VirtualPim,
    /// Virtual Photonic — butterfly-structured O(N log N) matmul with unitary constraints on INT8 tensor cores.
    VirtualPhotonic,
}

impl std::fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Npu => write!(f, "NPU"),
            Self::Tpu => write!(f, "TPU"),
            Self::Fpga => write!(f, "FPGA"),
            Self::VirtualTpu => write!(f, "vTPU"),
            Self::VirtualLpu => write!(f, "vLPU"),
            Self::VirtualWse => write!(f, "vWSE"),
            Self::VirtualRdu => write!(f, "vRDU"),
            Self::VirtualIpu => write!(f, "vIPU"),
            Self::VirtualTensix => write!(f, "vTensix"),
            Self::VirtualFpga => write!(f, "vFPGA"),
            Self::VirtualDpu => write!(f, "vDPU"),
            Self::VirtualPim => write!(f, "vPIM"),
            Self::VirtualPhotonic => write!(f, "vPhotonic"),
        }
    }
}

impl AcceleratorType {
    /// Whether this is a virtual silicon type (emulated on commodity hardware).
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            Self::VirtualTpu
                | Self::VirtualLpu
                | Self::VirtualWse
                | Self::VirtualRdu
                | Self::VirtualIpu
                | Self::VirtualTensix
                | Self::VirtualFpga
                | Self::VirtualDpu
                | Self::VirtualPim
                | Self::VirtualPhotonic
        )
    }
}

/// Extended accelerator info supporting GPU, NPU, TPU, and FPGA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorInfo {
    /// Device index on the host (e.g., GPU 0, NPU 1).
    pub device_index: usize,
    /// Accelerator type.
    pub accel_type: AcceleratorType,
    /// Model name (e.g., "NVIDIA H100", "Apple Neural Engine").
    pub name: String,
    /// Memory in megabytes (0 for accelerators without dedicated memory).
    pub memory_mb: u64,
    /// Vendor.
    pub vendor: GpuVendor,
}

// ── Hardware inventory ─────────────────────────────────────────────

/// Detected hardware inventory — the ground truth of what the machine has.
/// Immutable after detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInventory {
    /// CPU model name (e.g., "Apple M3 Max", "AMD EPYC 9654").
    pub cpu_model: String,
    /// Total physical CPU cores.
    pub cpu_cores: u32,
    /// CPU architecture.
    pub cpu_arch: CpuArch,
    /// Total physical RAM in megabytes.
    pub total_memory_mb: u64,
    /// Total available storage in gigabytes.
    pub total_storage_gb: u64,
    /// All detected accelerators.
    pub accelerators: Vec<AcceleratorInfo>,
    /// Estimated network bandwidth in Mbps.
    pub bandwidth_mbps: u32,
    /// Power source at detection time.
    pub energy_source: EnergySource,
    /// Battery percentage, if on battery.
    pub battery_pct: Option<u8>,
}

// ── Resource contribution ──────────────────────────────────────────

/// Owner-defined contribution limits — what they are willing to share.
/// The platform uses min(contributed, detected) for each resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContribution {
    /// CPU cores to share (e.g., 8 of 16).
    pub cpu_cores: u32,
    /// Memory to share in megabytes.
    pub memory_mb: u64,
    /// Storage to share in gigabytes.
    pub storage_gb: u64,
    /// Accelerator device indices to share (index into HardwareInventory::accelerators).
    pub shared_accelerators: Vec<usize>,
    /// Bandwidth limit in Mbps (None = no limit beyond hardware).
    pub bandwidth_limit_mbps: Option<u32>,
    /// Optional time-based sharing schedule.
    pub schedule: Option<TimeSchedule>,
}

impl ResourceContribution {
    /// Share everything the hardware has.
    pub fn all(inventory: &HardwareInventory) -> Self {
        let all_indices: Vec<usize> = (0..inventory.accelerators.len()).collect();
        Self {
            cpu_cores: inventory.cpu_cores,
            memory_mb: inventory.total_memory_mb,
            storage_gb: inventory.total_storage_gb,
            shared_accelerators: all_indices,
            bandwidth_limit_mbps: None,
            schedule: None,
        }
    }
}

/// Time-based sharing schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSchedule {
    /// Time windows when sharing is active.
    pub windows: Vec<TimeWindow>,
}

/// A single time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start hour in UTC (0-23).
    pub start_hour: u8,
    /// End hour in UTC (0-23). If less than start, wraps past midnight.
    pub end_hour: u8,
    /// Days of week (0=Sunday, 6=Saturday). Empty = every day.
    #[serde(default)]
    pub days: Vec<u8>,
}

impl TimeSchedule {
    /// Check if sharing is active at the given UTC timestamp.
    pub fn is_active(&self, timestamp_secs: u64) -> bool {
        if self.windows.is_empty() {
            return true;
        }
        let secs_since_midnight = timestamp_secs % 86400;
        let hour = (secs_since_midnight / 3600) as u8;
        // Unix epoch (1970-01-01) was a Thursday (day 4).
        let day_of_week = ((timestamp_secs / 86400 + 4) % 7) as u8;

        self.windows.iter().any(|w| {
            let day_ok = w.days.is_empty() || w.days.contains(&day_of_week);
            let hour_ok = if w.start_hour <= w.end_hour {
                hour >= w.start_hour && hour < w.end_hour
            } else {
                // Wraps past midnight (e.g., 22:00 → 08:00).
                hour >= w.start_hour || hour < w.end_hour
            };
            day_ok && hour_ok
        })
    }
}

// ── Effective resources ────────────────────────────────────────────

/// The effective resources available to the mesh after applying contribution limits.
/// This is what gets gossiped and used for scheduling: min(contributed, detected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveResources {
    /// Available CPU cores.
    pub cpu_cores: u32,
    /// Available memory in MB.
    pub memory_mb: u64,
    /// Available storage in GB.
    pub storage_gb: u64,
    /// Shared accelerators.
    pub accelerators: Vec<AcceleratorInfo>,
    /// Available bandwidth in Mbps.
    pub bandwidth_mbps: u32,
}

impl EffectiveResources {
    /// Compute effective resources from hardware inventory and contribution config.
    pub fn from_inventory_and_contribution(
        inventory: &HardwareInventory,
        contribution: &ResourceContribution,
    ) -> Self {
        let cpu_cores = contribution.cpu_cores.min(inventory.cpu_cores);
        let memory_mb = contribution.memory_mb.min(inventory.total_memory_mb);
        let storage_gb = contribution.storage_gb.min(inventory.total_storage_gb);
        let bandwidth_mbps = contribution
            .bandwidth_limit_mbps
            .map(|limit| limit.min(inventory.bandwidth_mbps))
            .unwrap_or(inventory.bandwidth_mbps);

        let accelerators = contribution
            .shared_accelerators
            .iter()
            .filter_map(|&idx| inventory.accelerators.get(idx).cloned())
            .collect();

        Self {
            cpu_cores,
            memory_mb,
            storage_gb,
            accelerators,
            bandwidth_mbps,
        }
    }
}

// ── Minimum requirements ───────────────────────────────────────────

/// Platform-defined minimum requirements to join the mesh.
#[derive(Debug, Clone)]
pub struct MinimumRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub storage_gb: u64,
    pub bandwidth_mbps: u32,
}

impl Default for MinimumRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_mb: 512,
            storage_gb: 5,
            bandwidth_mbps: 50,
        }
    }
}

impl MinimumRequirements {
    /// Validate that a hardware inventory meets the minimum requirements.
    pub fn validate(&self, inventory: &HardwareInventory) -> Result<(), String> {
        if inventory.cpu_cores < self.cpu_cores {
            return Err(format!(
                "CPU cores {} < minimum {}",
                inventory.cpu_cores, self.cpu_cores
            ));
        }
        if inventory.total_memory_mb < self.memory_mb {
            return Err(format!(
                "Memory {}MB < minimum {}MB",
                inventory.total_memory_mb, self.memory_mb
            ));
        }
        if inventory.total_storage_gb < self.storage_gb {
            return Err(format!(
                "Storage {}GB < minimum {}GB",
                inventory.total_storage_gb, self.storage_gb
            ));
        }
        if inventory.bandwidth_mbps < self.bandwidth_mbps {
            return Err(format!(
                "Bandwidth {}Mbps < minimum {}Mbps",
                inventory.bandwidth_mbps, self.bandwidth_mbps
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::Watts;

    fn test_capability() -> Capability {
        Capability {
            cpu_cores: 10,
            cpu_arch: CpuArch::Aarch64,
            memory_mb: 32768,
            gpu: Some(GpuInfo {
                name: "Apple M3 Max 40-core".into(),
                memory_mb: 40960,
                vendor: GpuVendor::Apple,
            }),
            storage_gb: 500,
            bandwidth_mbps: 1000,
            energy_source: EnergySource::WallPower,
            battery_pct: None,
            wasm_support: true,
            container_support: true,
            current_load: 0.15,
            energy_rate: Watts::new(12.4),
            thermal_state: ThermalState::Normal,
            locality_zone: "us-east-office-3f".into(),
            compliance_tags: vec!["hipaa".into(), "gdpr".into()],
            effective_resources: None,
            accelerators: vec![],
            energy_zone: None,
            energy_cost_usd_kwh: None,
            pq_readiness: None,
            crypto_risk_critical: None,
        }
    }

    fn test_inventory() -> HardwareInventory {
        HardwareInventory {
            cpu_model: "AMD EPYC 9654".into(),
            cpu_cores: 16,
            cpu_arch: CpuArch::X86_64,
            total_memory_mb: 65536,
            total_storage_gb: 1000,
            accelerators: vec![
                AcceleratorInfo {
                    device_index: 0,
                    accel_type: AcceleratorType::Gpu,
                    name: "NVIDIA H100".into(),
                    memory_mb: 81920,
                    vendor: GpuVendor::Nvidia,
                },
                AcceleratorInfo {
                    device_index: 1,
                    accel_type: AcceleratorType::Gpu,
                    name: "NVIDIA H100".into(),
                    memory_mb: 81920,
                    vendor: GpuVendor::Nvidia,
                },
                AcceleratorInfo {
                    device_index: 2,
                    accel_type: AcceleratorType::Npu,
                    name: "Intel Gaudi 3".into(),
                    memory_mb: 0,
                    vendor: GpuVendor::Intel,
                },
            ],
            bandwidth_mbps: 1000,
            energy_source: EnergySource::WallPower,
            battery_pct: None,
        }
    }

    #[test]
    fn capability_available() {
        let cap = test_capability();
        assert!(cap.is_available());
    }

    #[test]
    fn capability_critical_thermal_unavailable() {
        let mut cap = test_capability();
        cap.thermal_state = ThermalState::Critical;
        assert!(!cap.is_available());
    }

    #[test]
    fn capability_low_battery_unavailable() {
        let mut cap = test_capability();
        cap.battery_pct = Some(3);
        assert!(!cap.is_available());
    }

    #[test]
    fn capability_available_capacity() {
        let cap = test_capability();
        assert!((cap.available_capacity() - 0.85).abs() < 0.01);
    }

    #[test]
    fn capability_compliance_check() {
        let cap = test_capability();
        assert!(cap.satisfies_compliance(&["hipaa".into()]));
        assert!(cap.satisfies_compliance(&["hipaa".into(), "gdpr".into()]));
        assert!(!cap.satisfies_compliance(&["fedramp".into()]));
    }

    #[test]
    fn node_class_display() {
        assert_eq!(format!("{}", NodeClass::Cloud), "cloud");
        assert_eq!(format!("{}", NodeClass::Workstation), "workstation");
    }

    #[test]
    fn gossip_serialization_roundtrip() {
        use crate::identity::NodeId;

        let msg = GossipMessage {
            node_id: NodeId::from_bytes([0xAB; 32]),
            sequence: 42,
            timestamp: 1739640000,
            state: NodeState::Alive,
            capabilities: Some(test_capability()),
            node_class: NodeClass::Workstation,
            cumulative_energy_uwh: 0,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: GossipMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.sequence, 42);
        assert_eq!(parsed.state, NodeState::Alive);
    }

    // ── Accelerator tests ──────────────────────────────────────────

    #[test]
    fn accelerator_type_display() {
        assert_eq!(format!("{}", AcceleratorType::Gpu), "GPU");
        assert_eq!(format!("{}", AcceleratorType::Npu), "NPU");
        assert_eq!(format!("{}", AcceleratorType::Tpu), "TPU");
        assert_eq!(format!("{}", AcceleratorType::Fpga), "FPGA");
    }

    #[test]
    fn accelerator_info_serde_roundtrip() {
        let info = AcceleratorInfo {
            device_index: 0,
            accel_type: AcceleratorType::Gpu,
            name: "NVIDIA H100".into(),
            memory_mb: 81920,
            vendor: GpuVendor::Nvidia,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: AcceleratorInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.accel_type, AcceleratorType::Gpu);
        assert_eq!(parsed.memory_mb, 81920);
    }

    // ── Hardware inventory tests ────────────────────────────────────

    #[test]
    fn hardware_inventory_serde_roundtrip() {
        let inv = test_inventory();
        let json = serde_json::to_string(&inv).unwrap();
        let parsed: HardwareInventory = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cpu_cores, 16);
        assert_eq!(parsed.total_memory_mb, 65536);
        assert_eq!(parsed.accelerators.len(), 3);
    }

    // ── Resource contribution tests ────────────────────────────────

    #[test]
    fn contribution_all_shares_everything() {
        let inv = test_inventory();
        let contrib = ResourceContribution::all(&inv);
        assert_eq!(contrib.cpu_cores, 16);
        assert_eq!(contrib.memory_mb, 65536);
        assert_eq!(contrib.storage_gb, 1000);
        assert_eq!(contrib.shared_accelerators.len(), 3);
        assert!(contrib.bandwidth_limit_mbps.is_none());
        assert!(contrib.schedule.is_none());
    }

    #[test]
    fn contribution_serde_roundtrip() {
        let contrib = ResourceContribution {
            cpu_cores: 8,
            memory_mb: 32768,
            storage_gb: 500,
            shared_accelerators: vec![0],
            bandwidth_limit_mbps: Some(500),
            schedule: None,
        };
        let json = serde_json::to_string(&contrib).unwrap();
        let parsed: ResourceContribution = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cpu_cores, 8);
        assert_eq!(parsed.shared_accelerators, vec![0]);
    }

    // ── Effective resources tests ──────────────────────────────────

    #[test]
    fn effective_resources_caps_to_contribution() {
        let inv = test_inventory();
        let contrib = ResourceContribution {
            cpu_cores: 8,
            memory_mb: 32768,
            storage_gb: 500,
            shared_accelerators: vec![0],
            bandwidth_limit_mbps: Some(500),
            schedule: None,
        };
        let eff = EffectiveResources::from_inventory_and_contribution(&inv, &contrib);
        assert_eq!(eff.cpu_cores, 8);
        assert_eq!(eff.memory_mb, 32768);
        assert_eq!(eff.storage_gb, 500);
        assert_eq!(eff.accelerators.len(), 1);
        assert_eq!(eff.bandwidth_mbps, 500);
    }

    #[test]
    fn effective_resources_caps_to_hardware() {
        let inv = test_inventory();
        // Ask for more than hardware has.
        let contrib = ResourceContribution {
            cpu_cores: 64,
            memory_mb: 999999,
            storage_gb: 5000,
            shared_accelerators: vec![0, 1, 2],
            bandwidth_limit_mbps: Some(10000),
            schedule: None,
        };
        let eff = EffectiveResources::from_inventory_and_contribution(&inv, &contrib);
        assert_eq!(eff.cpu_cores, 16); // capped to hardware
        assert_eq!(eff.memory_mb, 65536);
        assert_eq!(eff.storage_gb, 1000);
        assert_eq!(eff.bandwidth_mbps, 1000); // capped to hardware
    }

    #[test]
    fn effective_resources_filters_accelerators_by_index() {
        let inv = test_inventory();
        let contrib = ResourceContribution {
            cpu_cores: 16,
            memory_mb: 65536,
            storage_gb: 1000,
            shared_accelerators: vec![0, 2], // GPU 0 + NPU 2, skip GPU 1
            bandwidth_limit_mbps: None,
            schedule: None,
        };
        let eff = EffectiveResources::from_inventory_and_contribution(&inv, &contrib);
        assert_eq!(eff.accelerators.len(), 2);
        assert_eq!(eff.accelerators[0].accel_type, AcceleratorType::Gpu);
        assert_eq!(eff.accelerators[1].accel_type, AcceleratorType::Npu);
    }

    #[test]
    fn effective_resources_invalid_index_skipped() {
        let inv = test_inventory();
        let contrib = ResourceContribution {
            cpu_cores: 16,
            memory_mb: 65536,
            storage_gb: 1000,
            shared_accelerators: vec![0, 99], // 99 doesn't exist
            bandwidth_limit_mbps: None,
            schedule: None,
        };
        let eff = EffectiveResources::from_inventory_and_contribution(&inv, &contrib);
        assert_eq!(eff.accelerators.len(), 1);
    }

    // ── Time schedule tests ────────────────────────────────────────

    #[test]
    fn schedule_empty_windows_always_active() {
        let sched = TimeSchedule { windows: vec![] };
        assert!(sched.is_active(0));
        assert!(sched.is_active(999999999));
    }

    #[test]
    fn schedule_daytime_window() {
        let sched = TimeSchedule {
            windows: vec![TimeWindow {
                start_hour: 9,
                end_hour: 17,
                days: vec![],
            }],
        };
        // 2024-01-15 12:00 UTC (Monday)
        let noon = 1705320000;
        assert!(sched.is_active(noon));
        // 2024-01-15 20:00 UTC
        let evening = 1705348800;
        assert!(!sched.is_active(evening));
    }

    #[test]
    fn schedule_overnight_window() {
        let sched = TimeSchedule {
            windows: vec![TimeWindow {
                start_hour: 22,
                end_hour: 8,
                days: vec![],
            }],
        };
        // 23:00 UTC on any day — should be active.
        let late_night = 86400 + 23 * 3600; // day 1, 23:00
        assert!(sched.is_active(late_night));
        // 03:00 UTC — should be active (past midnight).
        let early_morning = 86400 + 3 * 3600;
        assert!(sched.is_active(early_morning));
        // 12:00 UTC — should NOT be active.
        let midday = 86400 + 12 * 3600;
        assert!(!sched.is_active(midday));
    }

    #[test]
    fn schedule_weekday_filter() {
        let sched = TimeSchedule {
            windows: vec![TimeWindow {
                start_hour: 0,
                end_hour: 24,
                days: vec![1, 2, 3, 4, 5], // Mon-Fri
            }],
        };
        // 2024-01-15 is a Monday (day 1).
        let monday = 1705320000;
        assert!(sched.is_active(monday));
        // 2024-01-14 is a Sunday (day 0).
        let sunday = 1705233600;
        assert!(!sched.is_active(sunday));
    }

    // ── Minimum requirements tests ─────────────────────────────────

    #[test]
    fn minimum_requirements_pass() {
        let inv = test_inventory();
        let req = MinimumRequirements::default();
        assert!(req.validate(&inv).is_ok());
    }

    #[test]
    fn minimum_requirements_fail_cpu() {
        let mut inv = test_inventory();
        inv.cpu_cores = 0;
        let req = MinimumRequirements::default();
        let err = req.validate(&inv).unwrap_err();
        assert!(err.contains("CPU cores"));
    }

    #[test]
    fn minimum_requirements_fail_memory() {
        let mut inv = test_inventory();
        inv.total_memory_mb = 256;
        let req = MinimumRequirements::default();
        let err = req.validate(&inv).unwrap_err();
        assert!(err.contains("Memory"));
    }

    #[test]
    fn minimum_requirements_fail_storage() {
        let mut inv = test_inventory();
        inv.total_storage_gb = 4;
        let req = MinimumRequirements::default();
        let err = req.validate(&inv).unwrap_err();
        assert!(err.contains("Storage"));
    }

    #[test]
    fn minimum_requirements_fail_bandwidth() {
        let mut inv = test_inventory();
        inv.bandwidth_mbps = 10;
        let req = MinimumRequirements::default();
        let err = req.validate(&inv).unwrap_err();
        assert!(err.contains("Bandwidth"));
    }

    // ── Backward compatibility ─────────────────────────────────────

    #[test]
    fn capability_without_new_fields_deserializes() {
        // Simulate an old-format capability JSON (no effective_resources or accelerators).
        let json = r#"{
            "cpu_cores": 4, "cpu_arch": "X86_64", "memory_mb": 8192,
            "gpu": null, "storage_gb": 100, "bandwidth_mbps": 1000,
            "energy_source": "WallPower", "battery_pct": null,
            "wasm_support": true, "container_support": true,
            "current_load": 0.5, "energy_rate": 10.0,
            "thermal_state": "Normal", "locality_zone": "us-east",
            "compliance_tags": []
        }"#;
        let cap: Capability = serde_json::from_str(json).unwrap();
        assert_eq!(cap.cpu_cores, 4);
        assert!(cap.effective_resources.is_none());
        assert!(cap.accelerators.is_empty());
    }
}
