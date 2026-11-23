//! Energy-First Heterogeneous Compute Fabric (`joule-db-energy::fabric`)
//!
//! Unifies every compute substrate where JouleDB deploys—from 50 mW Cortex-M/RISC-V
//! C-FFI sensors (`McuSensor`), `wasm32` browser OPFS/WebGPU tabs (`WasmBrowser`),
//! battery-powered Jetson/Hailo/Coral/RK3588 robots (`EdgeRobotic`), zero-copy
//! Apple Silicon Unified Memory workstations (`AppleUma`), to Linux `io_uring` +
//! CXL 3.1 + GPU/TPU/LPU datacenters (`CloudCxl`)—under one **thermodynamic,
//! data-gravity, battery-survival, and grid-carbon scheduler**.
//!
//! ## First-Principles Thermodynamic Objective
//!
//! For any workload $W$ evaluated on candidate `(Node, SiliconLane)`:
//! $$J_{\text{total}} = J_{\text{compute}}(\text{lane}, \text{ops}) + J_{\text{memory}}(\text{UMA vs CXL vs PCIe}) + J_{\text{network}}(\text{Data Gravity})$$
//! $$\Phi = J_{\text{total}} \cdot \left(1 + \alpha_{\text{CO}_2} \frac{I_{\text{grid}}}{400}\right) \cdot \Psi_{\text{battery}}(\text{SoC}) \cdot \Psi_{\text{thermal}}(T)$$

use crate::monitor::ThermalState;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::RwLock;

/// Boltzmann Landauer limit at $T = 300\text{ K}$: $k_B T \ln 2 \approx 2.87 \times 10^{-21}\text{ J/bit}$.
pub const LANDAUER_JOULES_PER_BIT_300K: f64 = 2.87e-21;

/// Hardware deployment class in the universal JouleDB fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabricNodeClass {
    /// Bare-metal / RTOS / C99 FFI sensor or actuator node (~50 mW – 1 W).
    McuSensor,
    /// Browser OPFS/IndexedDB + WebGPU or WASI 0.2 worker (~0.5 – 3 W).
    WasmBrowser,
    /// Embodied robot, drone, or gateway with edge NPU/GPU (Jetson, Hailo, Coral, Pi 5; ~2 – 15 W).
    EdgeRobotic,
    /// Unified Memory Architecture workstation/mini-cluster (Apple Silicon CPU+Metal+ANE; ~15 – 60 W).
    AppleUma,
    /// Disaggregated datacenter node (`io_uring` + CXL 3.1 + discrete GPU/TPU/LPU; ~150 – 1,000 W).
    CloudCxl,
}

/// Silicon execution lane inside a fabric node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiliconLane {
    /// General-purpose CPU scalar + NEON/AVX-512 SIMD (B-Tree, SSI MVCC, WCOJ).
    CpuScalarSimd,
    /// Bitwise `POPCNT` + XOR hardware path (`RaBitQ` 1-bit ANN & 10,000-bit `BinaryHV` HDC).
    PopcntBitwise,
    /// Multiplier-free 1.58-bit `{-1, 0, +1}` ternary matvec lane.
    Ternary158Bit,
    /// Parallel compute shader lane (`wgpu` Metal / Vulkan / DX12 / WebGPU / CUDA).
    GpuShader,
    /// Systolic neural engine lane (Apple ANE, Hailo-8, EdgeTPU, Hexagon).
    NpuSystolic,
    /// Deterministic SRAM token/graph streaming lane (LPU / Wafer-Scale).
    LpuDeterministicSram,
    /// Fuel-metered deterministic WebAssembly reducer lane.
    WasmSandbox,
}

/// Physical memory-bus topology between host storage pages and accelerator lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBusKind {
    /// Unified Memory Architecture (Apple Silicon / Grace Hopper): zero PCIe copy (`~4 pJ/B` fabric read).
    UnifiedZeroCopy,
    /// Standard local LPDDR5/DDR5 DRAM (`~12 pJ/B`).
    LocalDram,
    /// Pooled CXL 3.1 memory (`~38 pJ/B`).
    CxlPooled,
    /// Discrete accelerator over PCIe Gen4/Gen5 copy (`~85 pJ/B` host-to-device + device-to-host).
    DiscretePcie,
}

impl MemoryBusKind {
    /// Energy cost in picojoules per byte transferred between DB page cache and silicon lane.
    pub fn transfer_pj_per_byte(self, lane: SiliconLane) -> f64 {
        match (self, lane) {
            (Self::UnifiedZeroCopy, _) => 4.0,
            (_, SiliconLane::CpuScalarSimd | SiliconLane::PopcntBitwise | SiliconLane::Ternary158Bit | SiliconLane::WasmSandbox) => 12.0,
            (Self::LocalDram, _) => 28.0,
            (Self::CxlPooled, _) => 38.0,
            (Self::DiscretePcie, _) => 85.0,
        }
    }
}

/// Network interconnect class from the submitting origin node to a target fabric node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterconnectKind {
    /// Same process / local node (`0 pJ/B` wire cost, `0 µs` RTT).
    LocalInProcess,
    /// Shared-host loopback / Unix domain socket (`5 pJ/B`, `40 µs` RTT).
    HostLoopback,
    /// Datacenter RDMA / 100GbE LAN (`120 pJ/B`, `250 µs` RTT).
    DatacenterLan,
    /// Edge Wi-Fi 6E / 5G无线 (`1,800 pJ/B`, `6,000 µs` RTT).
    WirelessEdge,
    /// Wide-area internet / LEO satcom (`6,500 pJ/B`, `35,000 µs` RTT).
    WideAreaWan,
}

impl InterconnectKind {
    /// Wire transport energy in picojoules per byte shipped across the link.
    pub fn wire_pj_per_byte(self) -> f64 {
        match self {
            Self::LocalInProcess => 0.0,
            Self::HostLoopback => 5.0,
            Self::DatacenterLan => 120.0,
            Self::WirelessEdge => 1_800.0,
            Self::WideAreaWan => 6_500.0,
        }
    }

    /// Baseline round-trip latency in microseconds.
    pub fn base_rtt_us(self) -> u64 {
        match self {
            Self::LocalInProcess => 0,
            Self::HostLoopback => 40,
            Self::DatacenterLan => 250,
            Self::WirelessEdge => 6_000,
            Self::WideAreaWan => 35_000,
        }
    }
}

/// Descriptor for a single live node in the heterogeneous JouleDB fabric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricNodeDescriptor {
    /// Stable node identifier (e.g. `"robot-arm-01"`, `"mac-studio-uma"`, `"hydro-nordic-01"`).
    pub node_id: String,
    /// Hardware class of this node.
    pub node_class: FabricNodeClass,
    /// Optional HTTP/JWP endpoint URL for live cross-node execution (`None` = in-process only).
    pub endpoint_url: Option<String>,
    /// Available silicon execution lanes on this node.
    pub lanes: Vec<SiliconLane>,
    /// Memory-bus architecture on this node.
    pub memory_bus: MemoryBusKind,
    /// Interconnect from the local coordinator to this node.
    pub interconnect: InterconnectKind,
    /// Nominal TDP in watts.
    pub tdp_watts: f64,
    /// Instantaneous active power draw in watts.
    pub current_power_watts: f64,
    /// Grid carbon intensity in $\text{gCO}_2\text{eq}/\text{kWh}$ (`0.0` = off-grid solar).
    pub grid_carbon_g_per_kwh: f64,
    /// Optional battery state-of-charge percentage (`0.0..=100.0`; `None` = mains-powered).
    pub battery_soc_pct: Option<f64>,
    /// Whether the battery is currently charging.
    pub battery_charging: bool,
    /// Current thermal state.
    pub thermal_state: ThermalState,
    /// Current load factor (`0.0..=1.0`).
    pub utilization: f64,
}

/// Workload archetype submitted to the heterogeneous fabric planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FabricWorkloadKind {
    /// Transactional B-tree / SSI MVCC / point or small range OLTP.
    OltpPointTransaction,
    /// Causal sensorimotor `ASOF` join, `SE(3)` kinematic lookup, or `SurpriseGate` reflex.
    EmbodiedCausalReflex,
    /// `RaBitQ` 1-bit ANN or 10,000-bit `BinaryHV` HDC `Bind`/`Bundle`/`PathStore` reasoning.
    BitwiseHdcVector,
    /// 1.58-bit ternary classifier or linear projection (`{-1, 0, +1}`).
    TernaryInference,
    /// Parallel columnar scan, aggregation, `SigQL` FFT, or dense cosine batch.
    ParallelColumnarOrDsp,
    /// Multi-hop `WCOJ` graph pattern or `DBSP` incremental join batch.
    GraphOrDbspJoin,
    /// Heavy sparse-MoE weight gather or multi-million-op synthesis.
    HeavyModelSynthesis,
    /// Deterministic fuel-metered reducer execution.
    DeterministicReducer,
}

impl FabricWorkloadKind {
    /// Compatible silicon lanes ordered from highest intrinsic thermodynamic efficiency to fallback.
    pub fn compatible_lanes(self) -> &'static [SiliconLane] {
        match self {
            Self::OltpPointTransaction => &[SiliconLane::CpuScalarSimd, SiliconLane::WasmSandbox],
            Self::EmbodiedCausalReflex => &[
                SiliconLane::NpuSystolic,
                SiliconLane::PopcntBitwise,
                SiliconLane::CpuScalarSimd,
            ],
            Self::BitwiseHdcVector => &[
                SiliconLane::PopcntBitwise,
                SiliconLane::LpuDeterministicSram,
                SiliconLane::NpuSystolic,
                SiliconLane::GpuShader,
                SiliconLane::CpuScalarSimd,
            ],
            Self::TernaryInference => &[
                SiliconLane::Ternary158Bit,
                SiliconLane::NpuSystolic,
                SiliconLane::LpuDeterministicSram,
                SiliconLane::GpuShader,
                SiliconLane::CpuScalarSimd,
            ],
            Self::ParallelColumnarOrDsp => &[
                SiliconLane::GpuShader,
                SiliconLane::NpuSystolic,
                SiliconLane::CpuScalarSimd,
            ],
            Self::GraphOrDbspJoin => &[
                SiliconLane::LpuDeterministicSram,
                SiliconLane::CpuScalarSimd,
                SiliconLane::GpuShader,
            ],
            Self::HeavyModelSynthesis => &[
                SiliconLane::NpuSystolic,
                SiliconLane::LpuDeterministicSram,
                SiliconLane::GpuShader,
                SiliconLane::Ternary158Bit,
                SiliconLane::CpuScalarSimd,
            ],
            Self::DeterministicReducer => &[SiliconLane::WasmSandbox, SiliconLane::CpuScalarSimd],
        }
    }

    /// Intrinsic compute energy in picojoules per logical operation on a given silicon lane.
    pub fn base_compute_pj_per_op(self, lane: SiliconLane) -> f64 {
        match (self, lane) {
            (Self::BitwiseHdcVector, SiliconLane::PopcntBitwise) => 1.8,
            (Self::BitwiseHdcVector, SiliconLane::NpuSystolic | SiliconLane::LpuDeterministicSram) => 2.5,
            (Self::BitwiseHdcVector, SiliconLane::GpuShader) => 4.2,
            (Self::BitwiseHdcVector, _) => 18.0,

            (Self::TernaryInference, SiliconLane::Ternary158Bit) => 3.2,
            (Self::TernaryInference, SiliconLane::NpuSystolic | SiliconLane::LpuDeterministicSram) => 4.5,
            (Self::TernaryInference, SiliconLane::GpuShader) => 9.0,
            (Self::TernaryInference, _) => 28.0,

            (Self::ParallelColumnarOrDsp, SiliconLane::GpuShader) => 5.5,
            (Self::ParallelColumnarOrDsp, SiliconLane::NpuSystolic) => 6.5,
            (Self::ParallelColumnarOrDsp, _) => 32.0,

            (Self::HeavyModelSynthesis, SiliconLane::NpuSystolic) => 14.0,
            (Self::HeavyModelSynthesis, SiliconLane::LpuDeterministicSram) => 12.0,
            (Self::HeavyModelSynthesis, SiliconLane::GpuShader) => 22.0,
            (Self::HeavyModelSynthesis, SiliconLane::Ternary158Bit) => 16.0,
            (Self::HeavyModelSynthesis, _) => 110.0,

            (Self::EmbodiedCausalReflex, SiliconLane::NpuSystolic) => 3.5,
            (Self::EmbodiedCausalReflex, SiliconLane::PopcntBitwise) => 2.2,
            (Self::EmbodiedCausalReflex, _) => 12.0,

            (_, SiliconLane::WasmSandbox) => 16.0,
            _ => 14.0,
        }
    }

    /// Throughput in millions of logical ops per second on `(node_class, lane)` for SLA estimation.
    pub fn throughput_mops_per_sec(self, class: FabricNodeClass, lane: SiliconLane) -> f64 {
        let class_scale: f64 = match class {
            FabricNodeClass::McuSensor => 8.0,
            FabricNodeClass::WasmBrowser => 45.0,
            FabricNodeClass::EdgeRobotic => 180.0,
            FabricNodeClass::AppleUma => 850.0,
            FabricNodeClass::CloudCxl => 1_800.0,
        };
        let lane_mult = match lane {
            SiliconLane::GpuShader => 12.0,
            SiliconLane::NpuSystolic => 15.0,
            SiliconLane::LpuDeterministicSram => 22.0,
            SiliconLane::PopcntBitwise => 10.0,
            SiliconLane::Ternary158Bit => 8.0,
            SiliconLane::CpuScalarSimd => 2.0,
            SiliconLane::WasmSandbox => 1.4,
        };
        (class_scale * lane_mult).max(0.05)
    }
}

/// Workload placement request submitted to [`HeterogeneousFabric`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricWorkload {
    /// Workload archetype.
    pub kind: FabricWorkloadKind,
    /// Number of logical operations (e.g. rows, vectors, FFT samples, or matvec MACs).
    pub ops_count: u64,
    /// Working-set bytes read from storage/page cache on the executing node.
    pub working_set_bytes: u64,
    /// Payload bytes that must traverse the network if shipped to a non-local node.
    pub payload_bytes_if_remote: u64,
    /// Optional strict latency ceiling in microseconds.
    pub latency_sla_us: Option<u64>,
    /// Optional strict physical energy ceiling in microjoules.
    pub max_energy_budget_uj: Option<u64>,
    /// If `true`, hard-pins execution to local/loopback nodes (no wireless/WAN control-loop jitter).
    pub causal_reflex: bool,
    /// Carbon-weighting strength (`0.0` = pure joules, `1.0` = balanced joules + $\text{CO}_2$, `3.0` = carbon-strict).
    pub carbon_weight: f64,
}

/// Scored candidate `(node, silicon_lane)` in a placement plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricCandidateEvaluation {
    /// Candidate node ID.
    pub node_id: String,
    /// Candidate hardware class.
    pub node_class: FabricNodeClass,
    /// Candidate silicon lane.
    pub lane: SiliconLane,
    /// Optional peer URL if remote.
    pub endpoint_url: Option<String>,
    /// Silicon compute energy in microjoules.
    pub compute_uj: f64,
    /// Memory-bus data movement energy in microjoules (`UMA` vs `CXL` vs `PCIe`).
    pub memory_bus_uj: f64,
    /// Network wire transport energy in microjoules (`Data Gravity` term).
    pub network_wire_uj: f64,
    /// Total physical energy `compute_uj + memory_bus_uj + network_wire_uj` in microjoules.
    pub total_energy_uj: f64,
    /// Estimated carbon footprint in micrograms $\text{CO}_2\text{eq}$.
    pub carbon_ug_co2: f64,
    /// Ratio vs the theoretical 300 K Landauer thermodynamic bit-erasure floor.
    pub multiple_of_landauer_floor: f64,
    /// Estimated end-to-end latency in microseconds (compute + wire RTT).
    pub estimated_latency_us: u64,
    /// Composite thermodynamic-carbon-survival objective score $\Phi$ (lower is better).
    pub objective_score: f64,
    /// Human-readable reason if disqualified (`None` = eligible).
    pub rejected_reason: Option<String>,
}

/// Final thermodynamic placement decision + full counterfactual comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FabricPlacementPlan {
    /// Winning candidate.
    pub selected: FabricCandidateEvaluation,
    /// Microjoules saved compared to the naive CPU/remote baseline candidate.
    pub energy_saved_vs_worst_uj: f64,
    /// Micrograms $\text{CO}_2$ saved compared to the highest-carbon eligible candidate.
    pub carbon_saved_vs_worst_ug: f64,
    /// All evaluated `(node, lane)` candidates ranked by `objective_score`.
    pub candidates: Vec<FabricCandidateEvaluation>,
}

/// Errors returned when no node in the fabric can satisfy hard physical/SLA constraints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FabricError {
    /// No nodes have been registered in the fabric.
    EmptyFabric,
    /// Every candidate violated a hard constraint (thermal, causal reflex, latency SLA, or µJ budget).
    NoFeasibleCandidate {
        /// Diagnostic summary of rejected candidates.
        summary: String,
    },
}

impl std::fmt::Display for FabricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFabric => write!(f, "heterogeneous fabric has no registered nodes"),
            Self::NoFeasibleCandidate { summary } => {
                write!(f, "no feasible fabric placement: {summary}")
            }
        }
    }
}

impl std::error::Error for FabricError {}

/// Thread-safe Energy-First Heterogeneous Compute Fabric registry and planner.
pub struct HeterogeneousFabric {
    local_node_id: String,
    nodes: RwLock<BTreeMap<String, FabricNodeDescriptor>>,
}

impl HeterogeneousFabric {
    /// Create a new fabric anchored on `local_node`.
    pub fn new(local_node: FabricNodeDescriptor) -> Self {
        let local_id = local_node.node_id.clone();
        let mut map = BTreeMap::new();
        map.insert(local_id.clone(), local_node);
        Self {
            local_node_id: local_id,
            nodes: RwLock::new(map),
        }
    }

    /// Return the local node ID.
    pub fn local_node_id(&self) -> &str {
        &self.local_node_id
    }

    /// Register or update a peer/local node's live telemetry and silicon capabilities.
    pub fn upsert_node(&self, node: FabricNodeDescriptor) {
        if let Ok(mut guard) = self.nodes.write() {
            guard.insert(node.node_id.clone(), node);
        }
    }

    /// Remove a node from the fabric (cannot remove the local anchor node).
    pub fn remove_node(&self, node_id: &str) -> bool {
        if node_id == self.local_node_id {
            return false;
        }
        if let Ok(mut guard) = self.nodes.write() {
            guard.remove(node_id).is_some()
        } else {
            false
        }
    }

    /// Snapshot all registered fabric nodes.
    pub fn topology(&self) -> Vec<FabricNodeDescriptor> {
        self.nodes
            .read()
            .map(|g| g.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Compute the optimal Energy-First placement across all nodes and silicon lanes.
    pub fn plan(&self, workload: &FabricWorkload) -> Result<FabricPlacementPlan, FabricError> {
        let nodes = self.topology();
        if nodes.is_empty() {
            return Err(FabricError::EmptyFabric);
        }

        let compatible = workload.kind.compatible_lanes();
        let mut evaluations = Vec::new();

        for node in &nodes {
            for &lane in &node.lanes {
                if !compatible.contains(&lane) {
                    continue;
                }
                evaluations.push(Self::evaluate_candidate(node, lane, workload));
            }
        }

        evaluations.sort_by(|a, b| {
            a.rejected_reason
                .is_some()
                .cmp(&b.rejected_reason.is_some())
                .then_with(|| a.objective_score.total_cmp(&b.objective_score))
        });

        let eligible: Vec<&FabricCandidateEvaluation> = evaluations
            .iter()
            .filter(|c| c.rejected_reason.is_none())
            .collect();

        let Some(&winner) = eligible.first() else {
            let reasons: Vec<String> = evaluations
                .iter()
                .take(4)
                .map(|c| {
                    format!(
                        "{}:{:?}={}",
                        c.node_id,
                        c.lane,
                        c.rejected_reason.as_deref().unwrap_or("ineligible")
                    )
                })
                .collect();
            return Err(FabricError::NoFeasibleCandidate {
                summary: reasons.join("; "),
            });
        };

        let worst_energy = eligible
            .iter()
            .map(|c| c.total_energy_uj)
            .fold(winner.total_energy_uj, f64::max);
        let worst_carbon = eligible
            .iter()
            .map(|c| c.carbon_ug_co2)
            .fold(winner.carbon_ug_co2, f64::max);

        Ok(FabricPlacementPlan {
            selected: winner.clone(),
            energy_saved_vs_worst_uj: (worst_energy - winner.total_energy_uj).max(0.0),
            carbon_saved_vs_worst_ug: (worst_carbon - winner.carbon_ug_co2).max(0.0),
            candidates: evaluations,
        })
    }

    fn evaluate_candidate(
        node: &FabricNodeDescriptor,
        lane: SiliconLane,
        w: &FabricWorkload,
    ) -> FabricCandidateEvaluation {
        let mut rejected_reason: Option<String> = None;

        // 1. Hard causal-reflex pin: embodied control loops must not cross wireless/WAN links
        if w.causal_reflex
            && !matches!(
                node.interconnect,
                InterconnectKind::LocalInProcess | InterconnectKind::HostLoopback
            )
        {
            rejected_reason = Some("causal_reflex requires local/loopback execution".into());
        }

        // 2. Hard thermal protection: never schedule GPU/CPU heavy bursts onto Critical silicon
        if node.thermal_state == ThermalState::Critical
            && matches!(lane, SiliconLane::GpuShader | SiliconLane::CpuScalarSimd)
            && w.ops_count > 10_000
        {
            rejected_reason = Some("node in Critical thermal state".into());
        }

        // 3. GPU dispatch amortization: tiny micro-batches (< 512 ops) pay ~8 µJ kernel launch overhead
        let gpu_launch_overhead_uj = if matches!(lane, SiliconLane::GpuShader) {
            8.0
        } else {
            0.0
        };

        // Compute energy (µJ): pJ -> µJ is * 1e-6
        let compute_pj = (w.ops_count as f64) * w.kind.base_compute_pj_per_op(lane);
        let compute_uj = compute_pj * 1e-6 + gpu_launch_overhead_uj;

        // Memory-bus movement (µJ)
        let memory_bus_uj =
            (w.working_set_bytes as f64) * node.memory_bus.transfer_pj_per_byte(lane) * 1e-6;

        // Network wire Data-Gravity energy (µJ)
        let network_wire_uj = if node.interconnect == InterconnectKind::LocalInProcess {
            0.0
        } else {
            (w.payload_bytes_if_remote as f64) * node.interconnect.wire_pj_per_byte() * 1e-6
        };

        let total_energy_uj = compute_uj + memory_bus_uj + network_wire_uj;

        // Carbon footprint (µg CO2eq):
        // 1 kWh = 3.6e9 mJ = 3.6e12 µJ; gCO2 -> µgCO2 is * 1e6 => factor = 1e6 / 3.6e12 = 1 / 3.6e6
        let carbon_ug_co2 = total_energy_uj * node.grid_carbon_g_per_kwh / 3_600_000.0;

        // Theoretical Landauer floor for `ops_count * 64` irreversible bit transitions
        let bits_touched = (w.ops_count.max(1) as f64) * 64.0;
        let landauer_joules = bits_touched * LANDAUER_JOULES_PER_BIT_300K;
        let multiple_of_landauer_floor = (total_energy_uj * 1e-6) / landauer_joules.max(1e-30);

        // End-to-end latency (µs)
        let mops_per_sec = w.kind.throughput_mops_per_sec(node.node_class, lane);
        let contention = 1.0 + node.utilization.clamp(0.0, 1.0);
        let compute_us = (((w.ops_count as f64) / mops_per_sec) * contention).ceil() as u64;
        let wire_serialize_us = if node.interconnect == InterconnectKind::LocalInProcess {
            0
        } else {
            node.interconnect.base_rtt_us() + (w.payload_bytes_if_remote / 1_024)
        };
        let estimated_latency_us = compute_us + wire_serialize_us;

        if rejected_reason.is_none()
            && let Some(sla) = w.latency_sla_us
            && estimated_latency_us > sla
        {
            rejected_reason = Some(format!(
                "estimated latency {estimated_latency_us}us exceeds SLA {sla}us"
            ));
        }
        if rejected_reason.is_none()
            && let Some(cap_uj) = w.max_energy_budget_uj
            && total_energy_uj > cap_uj as f64
        {
            rejected_reason = Some(format!(
                "total energy {total_energy_uj:.2}uJ exceeds budget {cap_uj}uJ"
            ));
        }

        // Survival multipliers: battery SoC preservation & thermal throttling
        let battery_penalty = match (node.battery_soc_pct, node.battery_charging) {
            (Some(soc), false) if soc < 15.0 => 12.0,
            (Some(soc), false) if soc < 30.0 => 3.5,
            (Some(soc), false) if soc < 50.0 => 1.5,
            _ => 1.0,
        };
        let thermal_penalty = match node.thermal_state {
            ThermalState::Nominal => 1.0,
            ThermalState::Fair => 1.25,
            ThermalState::Serious => 2.5,
            ThermalState::Critical => 8.0,
        };
        let carbon_factor =
            1.0 + w.carbon_weight.max(0.0) * (node.grid_carbon_g_per_kwh.max(0.0) / 400.0);

        let objective_score = total_energy_uj * carbon_factor * battery_penalty * thermal_penalty;

        FabricCandidateEvaluation {
            node_id: node.node_id.clone(),
            node_class: node.node_class,
            lane,
            endpoint_url: node.endpoint_url.clone(),
            compute_uj,
            memory_bus_uj,
            network_wire_uj,
            total_energy_uj,
            carbon_ug_co2,
            multiple_of_landauer_floor,
            estimated_latency_us,
            objective_score,
            rejected_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fabric() -> HeterogeneousFabric {
        let edge_robot = FabricNodeDescriptor {
            node_id: "edge-jetson-robot".into(),
            node_class: FabricNodeClass::EdgeRobotic,
            endpoint_url: None,
            lanes: vec![
                SiliconLane::CpuScalarSimd,
                SiliconLane::PopcntBitwise,
                SiliconLane::NpuSystolic,
            ],
            memory_bus: MemoryBusKind::LocalDram,
            interconnect: InterconnectKind::LocalInProcess,
            tdp_watts: 10.0,
            current_power_watts: 4.2,
            grid_carbon_g_per_kwh: 120.0,
            battery_soc_pct: Some(11.0), // Low battery!
            battery_charging: false,
            thermal_state: ThermalState::Nominal,
            utilization: 0.2,
        };
        let fabric = HeterogeneousFabric::new(edge_robot);

        fabric.upsert_node(FabricNodeDescriptor {
            node_id: "mac-studio-uma".into(),
            node_class: FabricNodeClass::AppleUma,
            endpoint_url: Some("http://127.0.0.1:18080".into()),
            lanes: vec![
                SiliconLane::CpuScalarSimd,
                SiliconLane::PopcntBitwise,
                SiliconLane::Ternary158Bit,
                SiliconLane::GpuShader,
                SiliconLane::NpuSystolic,
            ],
            memory_bus: MemoryBusKind::UnifiedZeroCopy,
            interconnect: InterconnectKind::DatacenterLan,
            tdp_watts: 45.0,
            current_power_watts: 14.0,
            grid_carbon_g_per_kwh: 85.0,
            battery_soc_pct: None,
            battery_charging: false,
            thermal_state: ThermalState::Nominal,
            utilization: 0.15,
        });

        fabric.upsert_node(FabricNodeDescriptor {
            node_id: "coal-cloud-east".into(),
            node_class: FabricNodeClass::CloudCxl,
            endpoint_url: Some("http://10.0.0.50:8080".into()),
            lanes: vec![
                SiliconLane::CpuScalarSimd,
                SiliconLane::GpuShader,
                SiliconLane::LpuDeterministicSram,
            ],
            memory_bus: MemoryBusKind::DiscretePcie,
            interconnect: InterconnectKind::WideAreaWan,
            tdp_watts: 700.0,
            current_power_watts: 420.0,
            grid_carbon_g_per_kwh: 620.0,
            battery_soc_pct: None,
            battery_charging: false,
            thermal_state: ThermalState::Nominal,
            utilization: 0.4,
        });

        fabric.upsert_node(FabricNodeDescriptor {
            node_id: "hydro-cloud-nordic".into(),
            node_class: FabricNodeClass::CloudCxl,
            endpoint_url: Some("http://10.0.1.50:8080".into()),
            lanes: vec![
                SiliconLane::CpuScalarSimd,
                SiliconLane::GpuShader,
                SiliconLane::NpuSystolic,
                SiliconLane::LpuDeterministicSram,
            ],
            memory_bus: MemoryBusKind::CxlPooled,
            interconnect: InterconnectKind::DatacenterLan,
            tdp_watts: 500.0,
            current_power_watts: 280.0,
            grid_carbon_g_per_kwh: 15.0, // Hydroelectric!
            battery_soc_pct: None,
            battery_charging: false,
            thermal_state: ThermalState::Nominal,
            utilization: 0.25,
        });

        fabric
    }

    #[test]
    fn test_causal_reflex_pins_locally_even_on_low_battery_robot() {
        let fabric = sample_fabric();
        let plan = fabric
            .plan(&FabricWorkload {
                kind: FabricWorkloadKind::EmbodiedCausalReflex,
                ops_count: 4_096,
                working_set_bytes: 16_384,
                payload_bytes_if_remote: 16_384,
                latency_sla_us: Some(500),
                max_energy_budget_uj: Some(500),
                causal_reflex: true,
                carbon_weight: 1.0,
            })
            .unwrap();

        assert_eq!(plan.selected.node_id, "edge-jetson-robot");
        assert_eq!(plan.selected.lane, SiliconLane::PopcntBitwise);
        assert_eq!(plan.selected.network_wire_uj, 0.0);
    }

    #[test]
    fn test_low_battery_robot_offloads_heavy_synthesis_to_uma_or_hydro_node() {
        let fabric = sample_fabric();
        let plan = fabric
            .plan(&FabricWorkload {
                kind: FabricWorkloadKind::HeavyModelSynthesis,
                ops_count: 50_000_000,
                working_set_bytes: 32 * 1024 * 1024,
                payload_bytes_if_remote: 8_192, // Tiny prompt/delta shipped!
                latency_sla_us: Some(50_000),
                max_energy_budget_uj: None,
                causal_reflex: false,
                carbon_weight: 2.0,
            })
            .unwrap();

        // Must offload away from 11%-SoC robot and avoid 620 gCO2/kWh WAN coal cloud!
        assert!(
            plan.selected.node_id == "mac-studio-uma"
                || plan.selected.node_id == "hydro-cloud-nordic",
            "unexpected winner: {:?}",
            plan.selected
        );
        assert!(plan.energy_saved_vs_worst_uj > 100.0);
        assert!(plan.carbon_saved_vs_worst_ug > 0.0);
    }

    #[test]
    fn test_data_gravity_prevents_shipping_large_payload_over_expensive_wire() {
        let local_uma = FabricNodeDescriptor {
            node_id: "browser-opfs-tab".into(),
            node_class: FabricNodeClass::WasmBrowser,
            endpoint_url: None,
            lanes: vec![SiliconLane::PopcntBitwise, SiliconLane::WasmSandbox],
            memory_bus: MemoryBusKind::LocalDram,
            interconnect: InterconnectKind::LocalInProcess,
            tdp_watts: 3.0,
            current_power_watts: 1.1,
            grid_carbon_g_per_kwh: 100.0,
            battery_soc_pct: Some(85.0),
            battery_charging: true,
            thermal_state: ThermalState::Nominal,
            utilization: 0.1,
        };
        let fabric = HeterogeneousFabric::new(local_uma);
        fabric.upsert_node(FabricNodeDescriptor {
            node_id: "remote-5g-cloud".into(),
            node_class: FabricNodeClass::CloudCxl,
            endpoint_url: Some("https://cloud.example.com".into()),
            lanes: vec![SiliconLane::LpuDeterministicSram],
            memory_bus: MemoryBusKind::CxlPooled,
            interconnect: InterconnectKind::WirelessEdge,
            tdp_watts: 400.0,
            current_power_watts: 200.0,
            grid_carbon_g_per_kwh: 50.0,
            battery_soc_pct: None,
            battery_charging: false,
            thermal_state: ThermalState::Nominal,
            utilization: 0.1,
        });

        // 10,000 RaBitQ POPCNT comparisons over 4 MB local OPFS index:
        // Local compute ~ 0.018 µJ + 48 µJ DRAM = ~48 µJ; shipping 4 MB over 5G costs ~7,500 µJ!
        let plan = fabric
            .plan(&FabricWorkload {
                kind: FabricWorkloadKind::BitwiseHdcVector,
                ops_count: 10_000,
                working_set_bytes: 4 * 1024 * 1024,
                payload_bytes_if_remote: 4 * 1024 * 1024,
                latency_sla_us: None,
                max_energy_budget_uj: None,
                causal_reflex: false,
                carbon_weight: 1.0,
            })
            .unwrap();

        assert_eq!(plan.selected.node_id, "browser-opfs-tab");
        assert_eq!(plan.selected.lane, SiliconLane::PopcntBitwise);
        assert!(plan.energy_saved_vs_worst_uj > 5_000.0);
    }
}
