//! Compute Substrate — heterogeneous hardware as a first-class dispatch input.
//!
//! The honest system gives the agent the best substrate for the work and
//! measures what it actually costs at the physical layer. Constraining an
//! agent to inferior hardware doesn't make it efficient — it inflates the
//! cost and time, creating a self-fulfilling prophecy.
//!
//! **No substrate is the center of the story.** Every task class has an
//! optimal substrate. The CPU is not the baseline — it is one substrate
//! among many, optimal only for unpredictable scalar control flow. Efficiency
//! is measured relative to the *best known substrate for each task class*,
//! not relative to a CPU doing the work badly.
//!
//! The substrate model bridges three systems:
//! - **Dispatch**: nodes advertise available substrates; bids include substrate
//!   selection with energy estimates per hardware class
//! - **Promotion**: deterministic paths have substrate-dependent costs
//!   (the same hash lookup is ~1 µJ on CPU, irrelevant to route through GPU)
//! - **Contracts**: energy budgets are expressed in work units — abstract
//!   measures of computation denominated against the optimal substrate,
//!   so they're comparable across hardware configurations
//!
//! Key principle: the dispatch function is a joint optimization over
//! (domain competence × compute substrate × energy cost). A node with GPU
//! access bidding on a matrix multiply should outbid a CPU-only node even if
//! the CPU node has higher domain competence, because the energy differential
//! dominates.

use crate::AcceleratorKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A compute substrate — the hardware class that will execute the work.
///
/// This is NOT the same as AcceleratorKind (which is a hardware type for
/// allocation). ComputeSubstrate represents the *execution target* for a
/// specific task, including CPU which is not an "accelerator" but is always
/// a substrate option.
///
/// Covers all silicon from MCU to orbital datacenter:
/// - MCU-class: MCU, DSP
/// - Edge: NPU, VPU, FPGA
/// - Workstation/Cloud: CPU, GPU, TPU, LPU, RDU, WSE, DPU
/// - Emerging: Neuromorphic, Photonic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeSubstrate {
    /// General-purpose CPU — x86 (AVX-512, AMX), Arm (NEON, SVE), RISC-V.
    /// From Cortex-A on phones to Xeon/EPYC in datacenters.
    CPU,
    /// Bare-metal microcontroller — Cortex-M, RISC-V RV32, ESP32.
    /// No OS, no MMU, sub-milliwatt to milliwatt power. Runs WASM or native.
    MCU,
    /// GPU — CUDA (NVIDIA Blackwell/Rubin), ROCm (AMD MI355X), Metal (Apple M5),
    /// Vulkan (cross-vendor). Parallel compute, matrix ops, inference.
    GPU,
    /// TPU — Google Tensor Processing Unit (Ironwood v7, Trillium v6, Coral edge).
    /// Matrix multiply / systolic array architecture.
    TPU,
    /// NPU — Neural Processing Unit. Apple Neural Engine (ANE), Intel NPU,
    /// Qualcomm Hexagon, Arm Ethos-U85, Samsung Exynos NPU.
    /// Purpose-built for inference at milliwatt-to-watt power.
    NPU,
    /// LPU — Language Processing Unit (Groq). SRAM-only, deterministic latency,
    /// 1-3 J/token. Optimized for sequential token generation.
    LPU,
    /// FPGA — Field-Programmable Gate Array. Xilinx/Versal (AMD), Intel Agilex,
    /// Lattice. Reconfigurable compute, dominant in defense/aerospace/telecom.
    /// Low latency, custom data paths, bitstream-programmable.
    FPGA,
    /// DPU/IPU — Data Processing Unit / Infrastructure Processing Unit.
    /// NVIDIA BlueField, Intel IPU, AMD Pensando. Network packet processing,
    /// storage offload, encryption offload. Frees CPU from data-plane work.
    DPU,
    /// VPU — Vision Processing Unit. Intel Movidius, Arm Ethos-U (vision mode),
    /// Ambarella CV-series. Optimized for image/video pipelines at edge power.
    VPU,
    /// DSP — Digital Signal Processor. Qualcomm Hexagon DSP, TI C66x,
    /// Cadence Tensilica. Signal processing, audio, radar, sensor fusion.
    /// MCU-adjacent power budgets, fixed-point arithmetic.
    DSP,
    /// RDU — Reconfigurable Dataflow Unit (SambaNova SN50). Dataflow architecture
    /// with tiered memory (SRAM + HBM + large-capacity). Purpose-built for
    /// agentic inference. 3.2 PFLOPS FP8.
    RDU,
    /// WSE — Wafer-Scale Engine (Cerebras). 900K cores, 4T transistors on a single
    /// wafer. 22x better energy efficiency than A100 for matrix ops.
    /// Datacenter-only, liquid-cooled.
    WSE,
    /// Neuromorphic — Intel Loihi 2, BrainChip Akida, SpiNNaker2.
    /// Event-driven spiking neural networks. 100x less energy than GPU for
    /// applicable workloads. Non-von Neumann architecture.
    Neuromorphic,
    /// Photonic — Lightmatter, Luminous Computing. Optical interconnects and
    /// optical matrix multiply. Emerging, not at production scale.
    /// Theoretical: speed-of-light latency, near-zero energy per MAC.
    Photonic,
}

impl ComputeSubstrate {
    /// Map from AcceleratorKind to ComputeSubstrate.
    pub fn from_accelerator(kind: &AcceleratorKind) -> Option<Self> {
        match kind {
            AcceleratorKind::GPU => Some(Self::GPU),
            AcceleratorKind::TPU => Some(Self::TPU),
            AcceleratorKind::NPU => Some(Self::NPU),
            AcceleratorKind::LPU => Some(Self::LPU),
            AcceleratorKind::FPGA => Some(Self::FPGA),
            AcceleratorKind::DPU => Some(Self::DPU),
            AcceleratorKind::VPU => Some(Self::VPU),
            AcceleratorKind::DSP => Some(Self::DSP),
            AcceleratorKind::RDU => Some(Self::RDU),
            AcceleratorKind::WSE => Some(Self::WSE),
            AcceleratorKind::Neuromorphic => Some(Self::Neuromorphic),
            AcceleratorKind::Photonic => Some(Self::Photonic),
            AcceleratorKind::Custom(_) => None, // can't profile unknown hardware
        }
    }

    /// All standard substrates (for enumeration).
    pub fn all() -> &'static [ComputeSubstrate] {
        &[
            Self::CPU,
            Self::MCU,
            Self::GPU,
            Self::TPU,
            Self::NPU,
            Self::LPU,
            Self::FPGA,
            Self::DPU,
            Self::VPU,
            Self::DSP,
            Self::RDU,
            Self::WSE,
            Self::Neuromorphic,
            Self::Photonic,
        ]
    }

    /// Whether this substrate is a datacenter-class accelerator.
    pub fn is_datacenter(&self) -> bool {
        matches!(self, Self::GPU | Self::TPU | Self::LPU | Self::RDU | Self::WSE)
    }

    /// Whether this substrate is edge-class (milliwatt to single-digit watt).
    pub fn is_edge(&self) -> bool {
        matches!(self, Self::MCU | Self::NPU | Self::VPU | Self::DSP | Self::FPGA)
    }

    /// Whether this substrate is emerging (not yet at production scale).
    pub fn is_emerging(&self) -> bool {
        matches!(self, Self::Neuromorphic | Self::Photonic)
    }
}

impl std::fmt::Display for ComputeSubstrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CPU => write!(f, "cpu"),
            Self::MCU => write!(f, "mcu"),
            Self::GPU => write!(f, "gpu"),
            Self::TPU => write!(f, "tpu"),
            Self::NPU => write!(f, "npu"),
            Self::LPU => write!(f, "lpu"),
            Self::FPGA => write!(f, "fpga"),
            Self::DPU => write!(f, "dpu"),
            Self::VPU => write!(f, "vpu"),
            Self::DSP => write!(f, "dsp"),
            Self::RDU => write!(f, "rdu"),
            Self::WSE => write!(f, "wse"),
            Self::Neuromorphic => write!(f, "neuromorphic"),
            Self::Photonic => write!(f, "photonic"),
        }
    }
}

/// Task class — the nature of the computation, which determines which
/// substrate is optimal.
///
/// The same "domain" (e.g., "search") may involve different task classes
/// at different stages (hash lookup vs embedding comparison vs reranking).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    /// Scalar computation: hash lookups, string ops, control flow.
    /// Best on: CPU, MCU.
    Scalar,
    /// Batch data transformation: map, filter, sort, aggregate.
    /// Best on: CPU (small), GPU (large batches).
    BatchTransform,
    /// Matrix/tensor operations: embeddings, similarity, linear algebra.
    /// Best on: GPU, TPU, WSE.
    TensorOp,
    /// Neural network inference: LLM, vision, classification.
    /// Best on: LPU, NPU, RDU, GPU (energy efficiency order varies by model size).
    Inference,
    /// Graph traversal: BFS, DFS, shortest path, PageRank.
    /// Best on: CPU (sparse), GPU (dense/large).
    GraphTraversal,
    /// IO-bound: network calls, disk reads, database queries.
    /// Substrate-neutral (bottleneck is not compute).
    IOBound,
    /// Signal processing: FFT, FIR/IIR filter, convolution, spectral analysis.
    /// Best on: DSP, FPGA. Radar, audio, sensor fusion workloads.
    SignalProcessing,
    /// Spiking neural network: event-driven computation, temporal coding,
    /// spike-timing dependent plasticity. Best on: Neuromorphic (100x less
    /// energy than GPU for applicable workloads). Not tensor-based.
    SpikingNetwork,
    /// Bitwise/crypto: hash computation, encryption, bit manipulation,
    /// Reed-Solomon coding. Best on: FPGA (custom data paths), DPU (inline).
    Bitwise,
    /// Data movement: packet processing, storage offload, DMA transfers,
    /// network function virtualization. Best on: DPU/IPU. Frees CPU from
    /// data-plane work.
    DataMovement,
    /// Image/video pipeline: encode, decode, resize, color conversion,
    /// object detection at edge. Best on: VPU, NPU, GPU.
    ImagePipeline,
    /// Photonic matrix multiply: optical interference-based linear algebra.
    /// Best on: Photonic processors. Emerging, near-zero energy per MAC.
    PhotonicCompute,
}

/// Energy cost profile for a (task class, substrate) pair.
///
/// These are order-of-magnitude estimates based on published measurements
/// (Jetson Orin, Apple M-series, NVIDIA A100, Google Coral, Groq LPU).
/// The exact values matter less than the ratios — the dispatch function
/// uses relative cost, not absolute.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EnergyCostEstimate {
    /// Energy cost in microjoules per unit of work.
    pub energy_per_unit_uj: u64,
    /// Latency in microseconds per unit of work.
    pub latency_per_unit_us: u64,
    /// Closeness to optimal: ratio of best-known substrate energy to this
    /// substrate's energy for the same task class. Range (0.0, 1.0].
    /// 1.0 = this IS the optimal substrate. 0.01 = 100× worse than optimal.
    /// Not CPU-referenced — each task class has its own optimal.
    pub proximity_to_optimal: f64,
}

/// Energy profile table: maps (TaskClass, ComputeSubstrate) → EnergyCostEstimate.
///
/// This is the core data structure that makes substrate-aware bidding possible.
/// A node with GPU access can compute "my estimated energy for this tensor op
/// is 100 µJ" while a CPU-only node estimates "my cost is 100,000 µJ" — and
/// the dispatch mesh picks the honest winner.
#[derive(Debug, Clone)]
pub struct SubstrateEnergyProfile {
    /// Cost estimates keyed by (task class, substrate).
    costs: HashMap<(TaskClass, ComputeSubstrate), EnergyCostEstimate>,
}

/// Serializable entry for the energy profile.
#[derive(Serialize, Deserialize)]
struct ProfileEntry {
    task_class: TaskClass,
    substrate: ComputeSubstrate,
    estimate: EnergyCostEstimate,
}

impl Serialize for SubstrateEnergyProfile {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let entries: Vec<ProfileEntry> = self
            .costs
            .iter()
            .map(|((tc, s), e)| ProfileEntry {
                task_class: *tc,
                substrate: *s,
                estimate: *e,
            })
            .collect();
        entries.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SubstrateEnergyProfile {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries: Vec<ProfileEntry> = Vec::deserialize(deserializer)?;
        let mut costs = HashMap::new();
        for e in entries {
            costs.insert((e.task_class, e.substrate), e.estimate);
        }
        Ok(Self { costs })
    }
}

impl SubstrateEnergyProfile {
    /// Create an empty profile.
    pub fn new() -> Self {
        Self {
            costs: HashMap::new(),
        }
    }

    /// Create the default profile based on published hardware measurements.
    ///
    /// Sources: NVIDIA GTC 2026, Google Ironwood/Trillium specs, Apple M5/ANE
    /// measurements, Intel Loihi 2 @ Sandia, Cerebras WSE-3, SambaNova SN50,
    /// Groq LPU published J/token, Arm Ethos-U85 TOPS/W, Qualcomm X2 NPU,
    /// LEAF benchmark (J/token on edge), TokenPowerBench, Stanford Intelligence
    /// per Joule. Values are order-of-magnitude estimates — dispatch uses
    /// relative cost, not absolute.
    ///
    /// Substrate key (14 types):
    ///   CPU, MCU, GPU, TPU, NPU, LPU, FPGA, DPU, VPU, DSP, RDU, WSE,
    ///   Neuromorphic, Photonic
    ///
    /// Task class key (12 types):
    ///   Scalar, BatchTransform, TensorOp, Inference, GraphTraversal, IOBound,
    ///   SignalProcessing, SpikingNetwork, Bitwise, DataMovement, ImagePipeline,
    ///   PhotonicCompute
    pub fn default_profile() -> Self {
        let mut p = Self::new();

        // ── Scalar: hash lookups, string ops, control flow ──
        // CPU/MCU are king. Accelerators pay dispatch overhead.
        p.set(TaskClass::Scalar, ComputeSubstrate::CPU,  1, 1);
        p.set(TaskClass::Scalar, ComputeSubstrate::MCU,  2, 3);    // slower clock, but runs
        p.set(TaskClass::Scalar, ComputeSubstrate::GPU,  10, 50);
        p.set(TaskClass::Scalar, ComputeSubstrate::TPU,  20, 100);
        p.set(TaskClass::Scalar, ComputeSubstrate::NPU,  15, 80);
        p.set(TaskClass::Scalar, ComputeSubstrate::LPU,  20, 100);
        p.set(TaskClass::Scalar, ComputeSubstrate::FPGA, 5, 10);   // custom pipeline possible
        p.set(TaskClass::Scalar, ComputeSubstrate::DPU,  8, 20);  // Arm cores, lower clock
        p.set(TaskClass::Scalar, ComputeSubstrate::VPU,  15, 80);
        p.set(TaskClass::Scalar, ComputeSubstrate::DSP,  3, 5);   // fixed-point fast
        p.set(TaskClass::Scalar, ComputeSubstrate::RDU,  20, 100);
        p.set(TaskClass::Scalar, ComputeSubstrate::WSE,  5, 5);    // massive cores, but overkill
        p.set(TaskClass::Scalar, ComputeSubstrate::Neuromorphic, 50, 200);
        p.set(TaskClass::Scalar, ComputeSubstrate::Photonic, 100, 500);

        // ── BatchTransform: map, filter, sort, aggregate ──
        // GPU wins at scale. FPGA good for fixed pipelines.
        p.set(TaskClass::BatchTransform, ComputeSubstrate::CPU,  100, 100);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::MCU,  500, 1000);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::GPU,  10, 20);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::TPU,  15, 25);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::NPU,  30, 40);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::LPU,  50, 60);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::FPGA, 20, 15);   // streaming pipeline
        p.set(TaskClass::BatchTransform, ComputeSubstrate::DPU,  60, 80);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::VPU,  80, 100);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::DSP,  70, 80);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::RDU,  12, 18);   // dataflow architecture
        p.set(TaskClass::BatchTransform, ComputeSubstrate::WSE,  8, 10);   // massive parallelism
        p.set(TaskClass::BatchTransform, ComputeSubstrate::Neuromorphic, 200, 300);
        p.set(TaskClass::BatchTransform, ComputeSubstrate::Photonic, 150, 200);

        // ── TensorOp: embeddings, similarity, linear algebra ──
        // GPU/TPU/WSE dominate. NPU efficient at edge scale.
        p.set(TaskClass::TensorOp, ComputeSubstrate::CPU,  100_000, 10_000);
        p.set(TaskClass::TensorOp, ComputeSubstrate::MCU,  500_000, 100_000);
        p.set(TaskClass::TensorOp, ComputeSubstrate::GPU,  100, 10);
        p.set(TaskClass::TensorOp, ComputeSubstrate::TPU,  50, 8);
        p.set(TaskClass::TensorOp, ComputeSubstrate::NPU,  30, 5);
        p.set(TaskClass::TensorOp, ComputeSubstrate::LPU,  80, 12);
        p.set(TaskClass::TensorOp, ComputeSubstrate::FPGA, 200, 20);     // good but not best
        p.set(TaskClass::TensorOp, ComputeSubstrate::DPU,  50_000, 5_000);
        p.set(TaskClass::TensorOp, ComputeSubstrate::VPU,  500, 50);     // vision-optimized matmul
        p.set(TaskClass::TensorOp, ComputeSubstrate::DSP,  10_000, 2_000);
        p.set(TaskClass::TensorOp, ComputeSubstrate::RDU,  40, 6);      // dataflow = native tensor
        p.set(TaskClass::TensorOp, ComputeSubstrate::WSE,  20, 3);      // 22x over A100 (Cerebras)
        p.set(TaskClass::TensorOp, ComputeSubstrate::Neuromorphic, 80_000, 8_000);
        p.set(TaskClass::TensorOp, ComputeSubstrate::Photonic, 10, 1); // theoretical: near-zero per MAC

        // ── Inference: LLM, vision, classification ──
        // LPU/RDU purpose-built. NPU efficient at edge. WSE fastest.
        p.set(TaskClass::Inference, ComputeSubstrate::CPU,  1_000_000, 500_000);
        p.set(TaskClass::Inference, ComputeSubstrate::MCU,  5_000_000, 5_000_000); // tiny models only
        p.set(TaskClass::Inference, ComputeSubstrate::GPU,  10_000, 5_000);
        p.set(TaskClass::Inference, ComputeSubstrate::TPU,  5_000, 3_000);
        p.set(TaskClass::Inference, ComputeSubstrate::NPU,  1_000, 500);   // ANE: 80x over A100 per op
        p.set(TaskClass::Inference, ComputeSubstrate::LPU,  500, 200);     // Groq: 1-3 J/token
        p.set(TaskClass::Inference, ComputeSubstrate::FPGA, 5_000, 2_000);
        p.set(TaskClass::Inference, ComputeSubstrate::DPU,  500_000, 200_000);
        p.set(TaskClass::Inference, ComputeSubstrate::VPU,  2_000, 1_000);  // vision models native
        p.set(TaskClass::Inference, ComputeSubstrate::DSP,  100_000, 50_000);
        p.set(TaskClass::Inference, ComputeSubstrate::RDU,  400, 150);     // SambaNova: "best tokens/watt"
        p.set(TaskClass::Inference, ComputeSubstrate::WSE,  300, 50);      // Cerebras: 2000+ tok/s on 480B
        p.set(TaskClass::Inference, ComputeSubstrate::Neuromorphic, 10_000, 5_000); // only for SNN models
        p.set(TaskClass::Inference, ComputeSubstrate::Photonic, 200, 30);  // emerging

        // ── GraphTraversal: BFS, DFS, PageRank ──
        // CPU for sparse (cache-friendly). GPU for dense/large. WSE for massive.
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::CPU,  10, 5);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::MCU,  50, 50);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::GPU,  100, 20);    // sparse = bad for GPU
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::TPU,  200, 50);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::NPU,  150, 40);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::LPU,  200, 50);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::FPGA, 15, 8);    // custom BFS pipeline
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::DPU,  30, 15);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::VPU,  150, 40);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::DSP,  50, 30);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::RDU,  20, 10);
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::WSE,  5, 2);      // massive on-chip SRAM
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::Neuromorphic, 3, 2); // native graph structure
        p.set(TaskClass::GraphTraversal, ComputeSubstrate::Photonic, 200, 100);

        // ── IOBound: network, disk, database ──
        // Substrate-neutral except DPU which offloads network/storage.
        for substrate in ComputeSubstrate::all() {
            p.set(TaskClass::IOBound, *substrate, 1, 1000);
        }
        // DPU is purpose-built for data-plane IO
        p.set(TaskClass::IOBound, ComputeSubstrate::DPU, 1, 200);

        // ── SignalProcessing: FFT, FIR/IIR, convolution, spectral ──
        // DSP is king. FPGA close second. GPU good at scale.
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::CPU,  100, 50);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::MCU,  200, 200);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::GPU,  20, 10);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::TPU,  50, 30);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::NPU,  40, 25);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::LPU,  80, 40);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::FPGA, 5, 2);   // custom pipeline, lowest latency
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::DPU,  60, 30);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::VPU,  50, 25);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::DSP,  3, 1);   // native instruction set
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::RDU,  30, 15);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::WSE,  15, 5);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::Neuromorphic, 200, 100);
        p.set(TaskClass::SignalProcessing, ComputeSubstrate::Photonic, 80, 40);

        // ── SpikingNetwork: event-driven, temporal coding, STDP ──
        // Neuromorphic is 100x better. Everything else is simulation.
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::CPU,  100_000, 50_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::MCU,  500_000, 500_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::GPU,  10_000, 5_000);    // GPU can simulate
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::TPU,  20_000, 10_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::NPU,  15_000, 8_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::LPU,  20_000, 10_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::FPGA, 5_000, 2_000);     // reconfigurable spike routing
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::DPU,  80_000, 40_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::VPU,  20_000, 10_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::DSP,  30_000, 15_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::RDU,  8_000, 4_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::WSE,  3_000, 1_000);
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::Neuromorphic, 1_000, 500); // native hardware
        p.set(TaskClass::SpikingNetwork, ComputeSubstrate::Photonic, 50_000, 20_000);

        // ── Bitwise: hash, encrypt, bit manipulation, error coding ──
        // FPGA is king (custom data paths). DPU inline crypto.
        p.set(TaskClass::Bitwise, ComputeSubstrate::CPU,  10, 5);     // AES-NI, SHA extensions
        p.set(TaskClass::Bitwise, ComputeSubstrate::MCU,  30, 30);
        p.set(TaskClass::Bitwise, ComputeSubstrate::GPU,  20, 10);
        p.set(TaskClass::Bitwise, ComputeSubstrate::TPU,  50, 30);
        p.set(TaskClass::Bitwise, ComputeSubstrate::NPU,  40, 20);
        p.set(TaskClass::Bitwise, ComputeSubstrate::LPU,  50, 30);
        p.set(TaskClass::Bitwise, ComputeSubstrate::FPGA, 2, 1);     // custom crypto pipeline
        p.set(TaskClass::Bitwise, ComputeSubstrate::DPU,  3, 2);     // inline encryption engine
        p.set(TaskClass::Bitwise, ComputeSubstrate::VPU,  40, 20);
        p.set(TaskClass::Bitwise, ComputeSubstrate::DSP,  15, 10);
        p.set(TaskClass::Bitwise, ComputeSubstrate::RDU,  30, 20);
        p.set(TaskClass::Bitwise, ComputeSubstrate::WSE,  10, 5);
        p.set(TaskClass::Bitwise, ComputeSubstrate::Neuromorphic, 100, 50);
        p.set(TaskClass::Bitwise, ComputeSubstrate::Photonic, 100, 50);

        // ── DataMovement: packet processing, storage offload, NVF ──
        // DPU is purpose-built. FPGA for custom protocols.
        p.set(TaskClass::DataMovement, ComputeSubstrate::CPU,  50, 20);
        p.set(TaskClass::DataMovement, ComputeSubstrate::MCU,  200, 200);
        p.set(TaskClass::DataMovement, ComputeSubstrate::GPU,  100, 50);   // PCIe overhead
        p.set(TaskClass::DataMovement, ComputeSubstrate::TPU,  200, 100);
        p.set(TaskClass::DataMovement, ComputeSubstrate::NPU,  150, 80);
        p.set(TaskClass::DataMovement, ComputeSubstrate::LPU,  200, 100);
        p.set(TaskClass::DataMovement, ComputeSubstrate::FPGA, 10, 3);    // custom NIC pipeline
        p.set(TaskClass::DataMovement, ComputeSubstrate::DPU,  5, 2);    // native: SmartNIC + storage offload
        p.set(TaskClass::DataMovement, ComputeSubstrate::VPU,  150, 80);
        p.set(TaskClass::DataMovement, ComputeSubstrate::DSP,  80, 40);
        p.set(TaskClass::DataMovement, ComputeSubstrate::RDU,  100, 50);
        p.set(TaskClass::DataMovement, ComputeSubstrate::WSE,  50, 20);
        p.set(TaskClass::DataMovement, ComputeSubstrate::Neuromorphic, 200, 100);
        p.set(TaskClass::DataMovement, ComputeSubstrate::Photonic, 3, 1); // optical interconnect

        // ── ImagePipeline: encode, decode, resize, detection ──
        // VPU is purpose-built. NPU/GPU strong. MCU can run tiny models.
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::CPU,  1_000, 500);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::MCU,  10_000, 10_000);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::GPU,  50, 20);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::TPU,  100, 40);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::NPU,  30, 15);    // vision is core NPU workload
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::LPU,  200, 100);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::FPGA, 40, 10);    // ISP pipeline
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::DPU,  500, 200);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::VPU,  20, 8);     // native: encode/decode/detect
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::DSP,  200, 100);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::RDU,  80, 30);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::WSE,  60, 20);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::Neuromorphic, 500, 200);
        p.set(TaskClass::ImagePipeline, ComputeSubstrate::Photonic, 100, 40);

        // ── PhotonicCompute: optical matrix multiply ──
        // Photonic is native. Everything else simulates.
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::CPU,  100_000, 10_000);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::MCU,  500_000, 100_000);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::GPU,  1_000, 100);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::TPU,  500, 50);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::NPU,  300, 30);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::LPU,  800, 80);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::FPGA, 2_000, 200);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::DPU,  50_000, 5_000);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::VPU,  10_000, 1_000);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::DSP,  20_000, 2_000);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::RDU,  400, 40);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::WSE,  200, 20);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::Neuromorphic, 50_000, 5_000);
        p.set(TaskClass::PhotonicCompute, ComputeSubstrate::Photonic, 1, 1); // native: speed of light

        // Derive proximity_to_optimal from the energy data itself.
        // No hand-coded efficiency ratios. The data speaks for itself.
        p.compute_proximities();

        p
    }

    /// Set a raw cost estimate for a (task class, substrate) pair.
    ///
    /// `proximity_to_optimal` is set to 0.0 as a placeholder. Call
    /// `compute_proximities()` after all entries are populated to derive
    /// the actual values from the energy data.
    fn set_raw(
        &mut self,
        task_class: TaskClass,
        substrate: ComputeSubstrate,
        energy_per_unit_uj: u64,
        latency_per_unit_us: u64,
    ) {
        self.costs.insert(
            (task_class, substrate),
            EnergyCostEstimate {
                energy_per_unit_uj,
                latency_per_unit_us,
                proximity_to_optimal: 0.0, // computed in finalize step
            },
        );
    }

    /// Compute `proximity_to_optimal` for every entry from the energy data.
    ///
    /// For each task class, finds the minimum energy across all substrates
    /// and sets `proximity_to_optimal = min_energy / this_energy`.
    /// The optimal substrate gets 1.0. Everything else gets the fraction.
    fn compute_proximities(&mut self) {
        // Collect all task classes present
        let task_classes: Vec<TaskClass> = self.costs.keys().map(|(tc, _)| *tc).collect::<std::collections::HashSet<_>>().into_iter().collect();

        for tc in task_classes {
            // Find minimum energy for this task class — this is the optimal
            let min_energy = self.costs.iter()
                .filter(|((t, _), _)| *t == tc)
                .map(|(_, e)| e.energy_per_unit_uj)
                .filter(|e| *e > 0)
                .min()
                .unwrap_or(1);

            // Set proximity for each substrate: optimal_energy / this_energy
            let keys: Vec<(TaskClass, ComputeSubstrate)> = self.costs.keys()
                .filter(|(t, _)| *t == tc)
                .copied()
                .collect();

            for key in keys {
                if let Some(estimate) = self.costs.get_mut(&key) {
                    estimate.proximity_to_optimal = if estimate.energy_per_unit_uj > 0 {
                        min_energy as f64 / estimate.energy_per_unit_uj as f64
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    /// Set a cost estimate and return &mut self for chaining.
    pub fn set(
        &mut self,
        task_class: TaskClass,
        substrate: ComputeSubstrate,
        energy_per_unit_uj: u64,
        latency_per_unit_us: u64,
    ) {
        self.set_raw(task_class, substrate, energy_per_unit_uj, latency_per_unit_us);
    }

    /// Get the cost estimate for a (task class, substrate) pair.
    pub fn get(
        &self,
        task_class: TaskClass,
        substrate: ComputeSubstrate,
    ) -> Option<&EnergyCostEstimate> {
        self.costs.get(&(task_class, substrate))
    }

    /// Find the optimal substrate for a task class from available substrates.
    ///
    /// Returns (substrate, estimate) with the lowest energy cost.
    /// This is the honest answer: give the agent the best place to work.
    pub fn optimal_substrate(
        &self,
        task_class: TaskClass,
        available: &[ComputeSubstrate],
    ) -> Option<(ComputeSubstrate, EnergyCostEstimate)> {
        available
            .iter()
            .filter_map(|s| self.get(task_class, *s).map(|e| (*s, *e)))
            .min_by_key(|(_, e)| e.energy_per_unit_uj)
    }

    /// Compute the energy ratio between two substrates for a task class.
    ///
    /// Returns Some(ratio) where ratio = cost_a / cost_b.
    /// A ratio > 1 means substrate A is more expensive.
    pub fn energy_ratio(
        &self,
        task_class: TaskClass,
        substrate_a: ComputeSubstrate,
        substrate_b: ComputeSubstrate,
    ) -> Option<f64> {
        let cost_a = self.get(task_class, substrate_a)?;
        let cost_b = self.get(task_class, substrate_b)?;
        if cost_b.energy_per_unit_uj == 0 {
            return None;
        }
        Some(cost_a.energy_per_unit_uj as f64 / cost_b.energy_per_unit_uj as f64)
    }

    /// Find the globally optimal substrate for a task class across ALL known substrates.
    ///
    /// This is what proximity_to_optimal is measured against.
    pub fn optimal_for(&self, task_class: TaskClass) -> Option<(ComputeSubstrate, EnergyCostEstimate)> {
        self.costs
            .iter()
            .filter(|((tc, _), _)| *tc == task_class)
            .min_by_key(|(_, e)| e.energy_per_unit_uj)
            .map(|((_, s), e)| (*s, *e))
    }
}

impl Default for SubstrateEnergyProfile {
    fn default() -> Self {
        Self::default_profile()
    }
}

/// A node's available compute substrates (what hardware it can use).
///
/// Every dispatch node carries this. CPU is always available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateCapability {
    /// Available substrates on this node.
    substrates: Vec<ComputeSubstrate>,
    /// Per-substrate device IDs (for allocation tracking).
    device_ids: HashMap<ComputeSubstrate, Vec<String>>,
}

impl SubstrateCapability {
    /// CPU-only node.
    pub fn cpu_only() -> Self {
        Self {
            substrates: vec![ComputeSubstrate::CPU],
            device_ids: HashMap::new(),
        }
    }

    /// Build from detected accelerators (CPU always included).
    pub fn from_accelerators(accelerators: &[AcceleratorKind]) -> Self {
        let mut substrates = vec![ComputeSubstrate::CPU];
        let mut device_ids = HashMap::new();

        for (i, kind) in accelerators.iter().enumerate() {
            if let Some(substrate) = ComputeSubstrate::from_accelerator(kind) {
                if !substrates.contains(&substrate) {
                    substrates.push(substrate);
                }
                device_ids
                    .entry(substrate)
                    .or_insert_with(Vec::new)
                    .push(format!("{}-{}", substrate, i));
            }
        }

        Self {
            substrates,
            device_ids,
        }
    }

    /// Add a substrate with device ID.
    pub fn add(&mut self, substrate: ComputeSubstrate, device_id: String) {
        if !self.substrates.contains(&substrate) {
            self.substrates.push(substrate);
        }
        self.device_ids
            .entry(substrate)
            .or_insert_with(Vec::new)
            .push(device_id);
    }

    /// Available substrates.
    pub fn available(&self) -> &[ComputeSubstrate] {
        &self.substrates
    }

    /// Whether a specific substrate is available.
    pub fn has(&self, substrate: ComputeSubstrate) -> bool {
        self.substrates.contains(&substrate)
    }

    /// Device IDs for a substrate.
    pub fn devices(&self, substrate: ComputeSubstrate) -> &[String] {
        self.device_ids
            .get(&substrate)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of devices for a substrate.
    pub fn device_count(&self, substrate: ComputeSubstrate) -> usize {
        self.devices(substrate).len()
    }

    /// Whether this node has any accelerator (non-CPU substrate).
    pub fn has_accelerator(&self) -> bool {
        self.substrates.iter().any(|s| *s != ComputeSubstrate::CPU)
    }
}

impl Default for SubstrateCapability {
    fn default() -> Self {
        Self::cpu_only()
    }
}

/// A substrate-aware bid: extends TaskBid with hardware selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateBid {
    /// Node submitting this bid.
    pub node_id: String,
    /// Task being bid on.
    pub task_id: String,
    /// Domain competence score.
    pub competence_score: f64,
    /// Selected substrate (the best available for this task class).
    pub substrate: ComputeSubstrate,
    /// Task class (determines optimal substrate).
    pub task_class: TaskClass,
    /// Estimated energy cost on the selected substrate (µJ).
    pub estimated_energy_uj: u64,
    /// Estimated latency on the selected substrate (µs).
    pub estimated_latency_us: u64,
    /// How close this substrate is to the optimal for this task class.
    /// 1.0 = this IS the optimal. 0.01 = 100× worse than optimal.
    pub proximity_to_optimal: f64,
    /// Composite score: competence + substrate fitness.
    /// A node on the optimal substrate gets the full competence bonus.
    /// A node far from optimal gets penalized proportionally.
    pub composite_score: f64,
}

impl SubstrateBid {
    /// Create a substrate-aware bid.
    ///
    /// The composite score jointly optimizes competence and hardware fit.
    /// `proximity_to_optimal` is in (0, 1] where 1.0 = this is the best
    /// known substrate for this task class. The composite score is:
    ///
    ///   competence - log10(1 / proximity)
    ///
    /// Which means: on the optimal substrate (proximity=1.0), the score
    /// equals the raw competence. On a substrate that's 1000× worse
    /// (proximity=0.001), the score drops by 3 points. No substrate is
    /// privileged — the penalty is measured from the task's own optimal.
    pub fn new(
        node_id: String,
        task_id: String,
        competence_score: f64,
        substrate: ComputeSubstrate,
        task_class: TaskClass,
        estimate: &EnergyCostEstimate,
    ) -> Self {
        // proximity_to_optimal is in (0, 1].
        // log10(proximity) is in (-inf, 0].
        // Optimal substrate: log10(1.0) = 0, no penalty.
        // 10× worse: log10(0.1) = -1, penalty of 1 point.
        // 1000× worse: log10(0.001) = -3, penalty of 3 points.
        let proximity = estimate.proximity_to_optimal.clamp(1e-10, 1.0);
        let substrate_penalty = proximity.log10(); // ≤ 0
        let composite = competence_score.max(0.0) + substrate_penalty;

        Self {
            node_id,
            task_id,
            competence_score,
            substrate,
            task_class,
            estimated_energy_uj: estimate.energy_per_unit_uj,
            estimated_latency_us: estimate.latency_per_unit_us,
            proximity_to_optimal: estimate.proximity_to_optimal,
            composite_score: composite,
        }
    }
}

/// Energy normalization: express budgets in work units denominated against
/// the optimal substrate.
///
/// A "work unit" is 1 µJ of computation as if performed on the optimal
/// substrate for the task class. This makes contracts comparable across
/// hardware configurations without privileging any particular substrate:
///
/// - On the optimal substrate: work_uj == actual_uj (no overhead)
/// - On a 100× worse substrate: work_uj = actual_uj × proximity (tiny fraction)
///
/// The budget expresses *work to be done*, not the cost of doing it badly.
/// The energy overhead on non-optimal substrates is visible and measurable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NormalizedEnergy {
    /// Energy in work units — denominated against the optimal substrate.
    /// On the optimal substrate, work_uj == actual_uj.
    pub work_uj: u64,
    /// Actual energy on the execution substrate (the physical cost).
    pub actual_uj: u64,
    /// Substrate the work ran on.
    pub substrate: ComputeSubstrate,
    /// Proximity to optimal (0, 1]. 1.0 = this IS optimal for this task class.
    pub proximity: f64,
}

impl NormalizedEnergy {
    /// Convert an actual energy measurement to work units.
    ///
    /// work_uj = actual_uj × proximity_to_optimal
    ///
    /// On the optimal substrate (proximity=1.0): work = actual.
    /// On a 100× worse substrate (proximity=0.01): work = actual / 100.
    /// Both express the same amount of computation in comparable units.
    pub fn from_actual(actual_uj: u64, substrate: ComputeSubstrate, profile: &SubstrateEnergyProfile, task_class: TaskClass) -> Self {
        let proximity = profile
            .get(task_class, substrate)
            .map(|e| e.proximity_to_optimal)
            .unwrap_or(1.0)
            .clamp(1e-10, 1.0);

        Self {
            work_uj: (actual_uj as f64 * proximity) as u64,
            actual_uj,
            substrate,
            proximity,
        }
    }

    /// Convert a work-unit budget to actual cost on a substrate.
    ///
    /// actual_uj = work_uj / proximity_to_optimal
    ///
    /// On the optimal substrate: actual = budget (no overhead).
    /// On a 100× worse substrate: actual = budget × 100.
    pub fn to_actual(work_uj: u64, substrate: ComputeSubstrate, profile: &SubstrateEnergyProfile, task_class: TaskClass) -> Self {
        let proximity = profile
            .get(task_class, substrate)
            .map(|e| e.proximity_to_optimal)
            .unwrap_or(1.0)
            .clamp(1e-10, 1.0);

        let actual = (work_uj as f64 / proximity) as u64;

        Self {
            work_uj,
            actual_uj: actual,
            substrate,
            proximity,
        }
    }

    /// Energy overhead: how much more this cost than optimal would have.
    ///
    /// actual_uj - work_uj. On optimal substrate: 0. On worse substrates:
    /// the energy wasted by not being on optimal hardware.
    pub fn energy_overhead_uj(&self) -> u64 {
        self.actual_uj.saturating_sub(self.work_uj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Substrate enumeration + mapping ──

    #[test]
    fn test_all_substrates_count() {
        assert_eq!(ComputeSubstrate::all().len(), 14);
    }

    #[test]
    fn test_substrate_from_accelerator() {
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::GPU), Some(ComputeSubstrate::GPU));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::NPU), Some(ComputeSubstrate::NPU));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::FPGA), Some(ComputeSubstrate::FPGA));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::DPU), Some(ComputeSubstrate::DPU));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::VPU), Some(ComputeSubstrate::VPU));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::DSP), Some(ComputeSubstrate::DSP));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::RDU), Some(ComputeSubstrate::RDU));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::WSE), Some(ComputeSubstrate::WSE));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::Neuromorphic), Some(ComputeSubstrate::Neuromorphic));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::Photonic), Some(ComputeSubstrate::Photonic));
        assert_eq!(ComputeSubstrate::from_accelerator(&AcceleratorKind::Custom("quantum".into())), None);
    }

    #[test]
    fn test_substrate_display() {
        assert_eq!(ComputeSubstrate::CPU.to_string(), "cpu");
        assert_eq!(ComputeSubstrate::MCU.to_string(), "mcu");
        assert_eq!(ComputeSubstrate::GPU.to_string(), "gpu");
        assert_eq!(ComputeSubstrate::LPU.to_string(), "lpu");
        assert_eq!(ComputeSubstrate::FPGA.to_string(), "fpga");
        assert_eq!(ComputeSubstrate::DPU.to_string(), "dpu");
        assert_eq!(ComputeSubstrate::VPU.to_string(), "vpu");
        assert_eq!(ComputeSubstrate::DSP.to_string(), "dsp");
        assert_eq!(ComputeSubstrate::RDU.to_string(), "rdu");
        assert_eq!(ComputeSubstrate::WSE.to_string(), "wse");
        assert_eq!(ComputeSubstrate::Neuromorphic.to_string(), "neuromorphic");
        assert_eq!(ComputeSubstrate::Photonic.to_string(), "photonic");
    }

    #[test]
    fn test_substrate_classification() {
        assert!(ComputeSubstrate::GPU.is_datacenter());
        assert!(ComputeSubstrate::TPU.is_datacenter());
        assert!(ComputeSubstrate::WSE.is_datacenter());
        assert!(ComputeSubstrate::RDU.is_datacenter());
        assert!(!ComputeSubstrate::CPU.is_datacenter());

        assert!(ComputeSubstrate::MCU.is_edge());
        assert!(ComputeSubstrate::NPU.is_edge());
        assert!(ComputeSubstrate::VPU.is_edge());
        assert!(ComputeSubstrate::DSP.is_edge());
        assert!(ComputeSubstrate::FPGA.is_edge());
        assert!(!ComputeSubstrate::GPU.is_edge());

        assert!(ComputeSubstrate::Neuromorphic.is_emerging());
        assert!(ComputeSubstrate::Photonic.is_emerging());
        assert!(!ComputeSubstrate::GPU.is_emerging());
    }

    // ── Energy profile completeness ──

    #[test]
    fn test_default_profile_has_all_168_entries() {
        let profile = SubstrateEnergyProfile::default_profile();
        let task_classes = [
            TaskClass::Scalar, TaskClass::BatchTransform, TaskClass::TensorOp,
            TaskClass::Inference, TaskClass::GraphTraversal, TaskClass::IOBound,
            TaskClass::SignalProcessing, TaskClass::SpikingNetwork, TaskClass::Bitwise,
            TaskClass::DataMovement, TaskClass::ImagePipeline, TaskClass::PhotonicCompute,
        ];
        let mut count = 0;
        for tc in task_classes {
            for s in ComputeSubstrate::all() {
                assert!(
                    profile.get(tc, *s).is_some(),
                    "missing entry for ({:?}, {:?})", tc, s
                );
                count += 1;
            }
        }
        assert_eq!(count, 168, "expected 12 task classes × 14 substrates = 168 entries");
    }

    #[test]
    fn test_all_proximities_valid() {
        let profile = SubstrateEnergyProfile::default_profile();
        for ((tc, s), e) in &profile.costs {
            assert!(
                e.proximity_to_optimal > 0.0 && e.proximity_to_optimal <= 1.0,
                "({:?}, {:?}) has invalid proximity: {}", tc, s, e.proximity_to_optimal
            );
            assert!(
                e.energy_per_unit_uj > 0,
                "({:?}, {:?}) has zero energy cost", tc, s
            );
            assert!(
                e.latency_per_unit_us > 0,
                "({:?}, {:?}) has zero latency", tc, s
            );
        }
    }

    #[test]
    fn test_optimal_substrate_has_proximity_one() {
        let profile = SubstrateEnergyProfile::default_profile();
        let task_classes = [
            TaskClass::Scalar, TaskClass::BatchTransform, TaskClass::TensorOp,
            TaskClass::Inference, TaskClass::GraphTraversal, TaskClass::IOBound,
            TaskClass::SignalProcessing, TaskClass::SpikingNetwork, TaskClass::Bitwise,
            TaskClass::DataMovement, TaskClass::ImagePipeline, TaskClass::PhotonicCompute,
        ];
        for tc in task_classes {
            let (optimal_sub, optimal_est) = profile.optimal_for(tc).unwrap();
            assert!(
                (optimal_est.proximity_to_optimal - 1.0).abs() < 1e-10,
                "{:?}: optimal substrate {:?} has proximity {} (expected 1.0)",
                tc, optimal_sub, optimal_est.proximity_to_optimal
            );
        }
    }

    // ── Optimal substrate per task class (the honest answer) ──

    #[test]
    fn test_optimal_substrate_scalar_prefers_cpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::Scalar, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::CPU);
    }

    #[test]
    fn test_optimal_substrate_inference_with_all_silicon() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::Inference, ComputeSubstrate::all())
            .unwrap();
        // Photonic is theoretical best (200 µJ), but Cerebras WSE (300 µJ) and
        // RDU (400 µJ) and LPU (500 µJ) are all viable. Profile says Photonic wins.
        assert_eq!(substrate, ComputeSubstrate::Photonic);
    }

    #[test]
    fn test_optimal_substrate_inference_production_only() {
        let profile = SubstrateEnergyProfile::default_profile();
        // Exclude emerging substrates — what wins among production silicon?
        let production: Vec<ComputeSubstrate> = ComputeSubstrate::all()
            .iter()
            .copied()
            .filter(|s| !s.is_emerging())
            .collect();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::Inference, &production)
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::WSE); // Cerebras: 300 µJ
    }

    #[test]
    fn test_optimal_substrate_tensor_prefers_photonic() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::TensorOp, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::Photonic); // 10 µJ, 10,000× efficiency
    }

    #[test]
    fn test_optimal_substrate_tensor_production_prefers_wse() {
        let profile = SubstrateEnergyProfile::default_profile();
        let production: Vec<ComputeSubstrate> = ComputeSubstrate::all()
            .iter()
            .copied()
            .filter(|s| !s.is_emerging())
            .collect();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::TensorOp, &production)
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::WSE); // 20 µJ, 5000×
    }

    #[test]
    fn test_optimal_substrate_signal_processing_prefers_dsp() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::SignalProcessing, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::DSP); // 3 µJ, 33×
    }

    #[test]
    fn test_optimal_substrate_spiking_prefers_neuromorphic() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::SpikingNetwork, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::Neuromorphic); // 1000 µJ, 100×
    }

    #[test]
    fn test_optimal_substrate_bitwise_prefers_fpga() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::Bitwise, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::FPGA); // 2 µJ, 5×
    }

    #[test]
    fn test_optimal_substrate_data_movement_prefers_photonic() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::DataMovement, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::Photonic); // 3 µJ, optical interconnect
    }

    #[test]
    fn test_optimal_substrate_data_movement_production_prefers_dpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let production: Vec<ComputeSubstrate> = ComputeSubstrate::all()
            .iter()
            .copied()
            .filter(|s| !s.is_emerging())
            .collect();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::DataMovement, &production)
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::DPU); // 5 µJ, SmartNIC native
    }

    #[test]
    fn test_optimal_substrate_image_pipeline_prefers_vpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::ImagePipeline, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::VPU); // 20 µJ, 50×
    }

    #[test]
    fn test_optimal_substrate_photonic_compute_prefers_photonic() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::PhotonicCompute, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::Photonic); // 1 µJ, 100,000×
    }

    #[test]
    fn test_optimal_substrate_graph_traversal() {
        let profile = SubstrateEnergyProfile::default_profile();
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::GraphTraversal, ComputeSubstrate::all())
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::Neuromorphic); // 3 µJ, native graph structure
    }

    #[test]
    fn test_optimal_substrate_limited_hardware() {
        let profile = SubstrateEnergyProfile::default_profile();
        // CPU-only node: inference still picks CPU (only option)
        let (substrate, estimate) = profile
            .optimal_substrate(TaskClass::Inference, &[ComputeSubstrate::CPU])
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::CPU);
        assert_eq!(estimate.energy_per_unit_uj, 1_000_000);
    }

    #[test]
    fn test_optimal_substrate_mcu_edge_node() {
        // A real MCU node: MCU + DSP (e.g., STM32N6 with DSP extensions)
        let profile = SubstrateEnergyProfile::default_profile();
        let mcu_node = [ComputeSubstrate::MCU, ComputeSubstrate::DSP];

        // Signal processing: DSP wins
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::SignalProcessing, &mcu_node)
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::DSP);

        // Scalar: MCU wins (lower energy than DSP for pure scalar)
        let (substrate, _) = profile
            .optimal_substrate(TaskClass::Scalar, &mcu_node)
            .unwrap();
        assert_eq!(substrate, ComputeSubstrate::MCU);
    }

    // ── Energy ratios ──

    #[test]
    fn test_energy_ratio_inference_gpu_vs_cpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let ratio = profile
            .energy_ratio(TaskClass::Inference, ComputeSubstrate::CPU, ComputeSubstrate::GPU)
            .unwrap();
        assert_eq!(ratio, 100.0);
    }

    #[test]
    fn test_energy_ratio_scalar_gpu_vs_cpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let ratio = profile
            .energy_ratio(TaskClass::Scalar, ComputeSubstrate::GPU, ComputeSubstrate::CPU)
            .unwrap();
        assert_eq!(ratio, 10.0);
    }

    #[test]
    fn test_energy_ratio_spiking_neuromorphic_vs_gpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let ratio = profile
            .energy_ratio(TaskClass::SpikingNetwork, ComputeSubstrate::GPU, ComputeSubstrate::Neuromorphic)
            .unwrap();
        // GPU = 10,000 µJ, Neuromorphic = 1,000 µJ → 10×
        assert_eq!(ratio, 10.0);
    }

    #[test]
    fn test_energy_ratio_tensor_wse_vs_gpu() {
        let profile = SubstrateEnergyProfile::default_profile();
        let ratio = profile
            .energy_ratio(TaskClass::TensorOp, ComputeSubstrate::GPU, ComputeSubstrate::WSE)
            .unwrap();
        // GPU = 100 µJ, WSE = 20 µJ → 5×
        assert_eq!(ratio, 5.0);
    }

    // ── Capability ──

    #[test]
    fn test_capability_cpu_only() {
        let cap = SubstrateCapability::cpu_only();
        assert!(cap.has(ComputeSubstrate::CPU));
        assert!(!cap.has(ComputeSubstrate::GPU));
        assert!(!cap.has_accelerator());
    }

    #[test]
    fn test_capability_from_all_accelerators() {
        let accel = vec![
            AcceleratorKind::GPU, AcceleratorKind::NPU, AcceleratorKind::FPGA,
            AcceleratorKind::DPU, AcceleratorKind::VPU, AcceleratorKind::DSP,
            AcceleratorKind::RDU, AcceleratorKind::WSE, AcceleratorKind::Neuromorphic,
            AcceleratorKind::Photonic,
        ];
        let cap = SubstrateCapability::from_accelerators(&accel);
        assert!(cap.has(ComputeSubstrate::CPU)); // always
        assert!(cap.has(ComputeSubstrate::GPU));
        assert!(cap.has(ComputeSubstrate::FPGA));
        assert!(cap.has(ComputeSubstrate::DPU));
        assert!(cap.has(ComputeSubstrate::VPU));
        assert!(cap.has(ComputeSubstrate::DSP));
        assert!(cap.has(ComputeSubstrate::RDU));
        assert!(cap.has(ComputeSubstrate::WSE));
        assert!(cap.has(ComputeSubstrate::Neuromorphic));
        assert!(cap.has(ComputeSubstrate::Photonic));
        assert!(cap.has_accelerator());
    }

    #[test]
    fn test_capability_add_device() {
        let mut cap = SubstrateCapability::cpu_only();
        cap.add(ComputeSubstrate::GPU, "gpu-0".into());
        cap.add(ComputeSubstrate::GPU, "gpu-1".into());
        cap.add(ComputeSubstrate::FPGA, "fpga-0".into());
        assert!(cap.has(ComputeSubstrate::GPU));
        assert!(cap.has(ComputeSubstrate::FPGA));
        assert_eq!(cap.device_count(ComputeSubstrate::GPU), 2);
        assert_eq!(cap.device_count(ComputeSubstrate::FPGA), 1);
    }

    // ── Substrate-aware bidding ──

    #[test]
    fn test_substrate_bid_composite_score_gpu_beats_cpu_for_inference() {
        let profile = SubstrateEnergyProfile::default_profile();

        let cpu_estimate = profile.get(TaskClass::Inference, ComputeSubstrate::CPU).unwrap();
        let cpu_bid = SubstrateBid::new(
            "cpu-node".into(), "t1".into(), 0.9,
            ComputeSubstrate::CPU, TaskClass::Inference, cpu_estimate,
        );

        let gpu_estimate = profile.get(TaskClass::Inference, ComputeSubstrate::GPU).unwrap();
        let gpu_bid = SubstrateBid::new(
            "gpu-node".into(), "t1".into(), 0.6,
            ComputeSubstrate::GPU, TaskClass::Inference, gpu_estimate,
        );

        // CPU proximity for inference: 200/1M = 0.0002 → penalty = log10(0.0002) = -3.7
        // CPU composite: 0.9 + (-3.7) = -2.8
        // GPU proximity for inference: 200/10K = 0.02 → penalty = log10(0.02) = -1.7
        // GPU composite: 0.6 + (-1.7) = -1.1
        // GPU wins: less penalty from being closer to optimal
        assert!(gpu_bid.composite_score > cpu_bid.composite_score);
    }

    #[test]
    fn test_substrate_bid_wse_dominates_for_tensor() {
        let profile = SubstrateEnergyProfile::default_profile();

        let gpu_estimate = profile.get(TaskClass::TensorOp, ComputeSubstrate::GPU).unwrap();
        let gpu_bid = SubstrateBid::new(
            "gpu-node".into(), "t1".into(), 0.9,
            ComputeSubstrate::GPU, TaskClass::TensorOp, gpu_estimate,
        );

        let wse_estimate = profile.get(TaskClass::TensorOp, ComputeSubstrate::WSE).unwrap();
        let wse_bid = SubstrateBid::new(
            "wse-node".into(), "t1".into(), 0.5,
            ComputeSubstrate::WSE, TaskClass::TensorOp, wse_estimate,
        );

        // GPU proximity: 10/100 = 0.1 → penalty = -1.0, composite = 0.9 - 1.0 = -0.1
        // WSE proximity: 10/20 = 0.5 → penalty = -0.3, composite = 0.5 - 0.3 = 0.2
        assert!(
            wse_bid.composite_score > gpu_bid.composite_score,
            "WSE ({:.2}) should beat GPU ({:.2}) for tensor ops",
            wse_bid.composite_score, gpu_bid.composite_score,
        );
    }

    #[test]
    fn test_substrate_bid_neuromorphic_dominates_for_spiking() {
        let profile = SubstrateEnergyProfile::default_profile();

        let gpu_estimate = profile.get(TaskClass::SpikingNetwork, ComputeSubstrate::GPU).unwrap();
        let gpu_bid = SubstrateBid::new(
            "gpu-node".into(), "t1".into(), 0.9,
            ComputeSubstrate::GPU, TaskClass::SpikingNetwork, gpu_estimate,
        );

        let neuro_estimate = profile.get(TaskClass::SpikingNetwork, ComputeSubstrate::Neuromorphic).unwrap();
        let neuro_bid = SubstrateBid::new(
            "neuro-node".into(), "t1".into(), 0.5,
            ComputeSubstrate::Neuromorphic, TaskClass::SpikingNetwork, neuro_estimate,
        );

        // GPU proximity: 1000/10000 = 0.1 → penalty = -1.0, composite = 0.9 - 1.0 = -0.1
        // Neuro proximity: 1000/1000 = 1.0 → penalty = 0, composite = 0.5
        assert!(neuro_bid.composite_score > gpu_bid.composite_score);
    }

    #[test]
    fn test_substrate_bid_dsp_dominates_for_signal() {
        let profile = SubstrateEnergyProfile::default_profile();

        let cpu_estimate = profile.get(TaskClass::SignalProcessing, ComputeSubstrate::CPU).unwrap();
        let cpu_bid = SubstrateBid::new(
            "cpu-node".into(), "t1".into(), 0.9,
            ComputeSubstrate::CPU, TaskClass::SignalProcessing, cpu_estimate,
        );

        let dsp_estimate = profile.get(TaskClass::SignalProcessing, ComputeSubstrate::DSP).unwrap();
        let dsp_bid = SubstrateBid::new(
            "dsp-node".into(), "t1".into(), 0.5,
            ComputeSubstrate::DSP, TaskClass::SignalProcessing, dsp_estimate,
        );

        // CPU proximity: 3/100 = 0.03 → penalty = -1.52, composite = 0.9 - 1.52 = -0.62
        // DSP proximity: 3/3 = 1.0 → penalty = 0, composite = 0.5
        assert!(dsp_bid.composite_score > cpu_bid.composite_score);
    }

    #[test]
    fn test_substrate_bid_scalar_cpu_wins() {
        let profile = SubstrateEnergyProfile::default_profile();

        let cpu_estimate = profile.get(TaskClass::Scalar, ComputeSubstrate::CPU).unwrap();
        let cpu_bid = SubstrateBid::new(
            "cpu-node".into(), "t1".into(), 0.5,
            ComputeSubstrate::CPU, TaskClass::Scalar, cpu_estimate,
        );

        let gpu_estimate = profile.get(TaskClass::Scalar, ComputeSubstrate::GPU).unwrap();
        let gpu_bid = SubstrateBid::new(
            "gpu-node".into(), "t1".into(), 0.5,
            ComputeSubstrate::GPU, TaskClass::Scalar, gpu_estimate,
        );

        // CPU proximity for scalar: 1.0 → no penalty, composite = 0.5
        // GPU proximity for scalar: 1/10 = 0.1 → penalty = -1.0, composite = -0.5
        // CPU wins on composite score, AND on energy tiebreak
        assert!(cpu_bid.composite_score > gpu_bid.composite_score);
        assert!(cpu_bid.estimated_energy_uj < gpu_bid.estimated_energy_uj);
    }

    // ── Energy normalization (work units) ──

    #[test]
    fn test_normalized_energy_optimal_substrate_identity() {
        let profile = SubstrateEnergyProfile::default_profile();
        // Neuromorphic is optimal for spiking (proximity = 1.0)
        let norm = NormalizedEnergy::from_actual(
            1_000, ComputeSubstrate::Neuromorphic, &profile, TaskClass::SpikingNetwork,
        );
        assert_eq!(norm.actual_uj, 1_000);
        assert_eq!(norm.work_uj, 1_000); // on optimal: work = actual
        assert!((norm.proximity - 1.0).abs() < 1e-10);
        assert_eq!(norm.energy_overhead_uj(), 0); // no overhead on optimal
    }

    #[test]
    fn test_normalized_energy_non_optimal_overhead() {
        let profile = SubstrateEnergyProfile::default_profile();
        // GPU for spiking: proximity = 1000/10000 = 0.1
        let norm = NormalizedEnergy::from_actual(
            10_000, ComputeSubstrate::GPU, &profile, TaskClass::SpikingNetwork,
        );
        assert_eq!(norm.actual_uj, 10_000);
        assert_eq!(norm.work_uj, 1_000); // 10000 × 0.1 = 1000
        assert_eq!(norm.energy_overhead_uj(), 9_000); // 9000 µJ wasted
    }

    #[test]
    fn test_normalized_energy_to_actual_on_optimal() {
        let profile = SubstrateEnergyProfile::default_profile();
        // DSP is optimal for signal processing (proximity = 1.0)
        let norm = NormalizedEnergy::to_actual(
            5_000, ComputeSubstrate::DSP, &profile, TaskClass::SignalProcessing,
        );
        assert_eq!(norm.work_uj, 5_000);
        assert_eq!(norm.actual_uj, 5_000); // on optimal: actual = budget
    }

    #[test]
    fn test_normalized_energy_to_actual_on_worse_substrate() {
        let profile = SubstrateEnergyProfile::default_profile();
        // CPU for signal processing: proximity = 3/100 = 0.03
        let norm = NormalizedEnergy::to_actual(
            5_000, ComputeSubstrate::CPU, &profile, TaskClass::SignalProcessing,
        );
        assert_eq!(norm.work_uj, 5_000);
        // actual = 5000 / 0.03 = ~166,666
        assert!(norm.actual_uj > 100_000);
        assert!(norm.energy_overhead_uj() > 100_000);
    }

    #[test]
    fn test_normalized_energy_photonic_tensor() {
        let profile = SubstrateEnergyProfile::default_profile();
        // Photonic is optimal for TensorOp (proximity = 1.0)
        let norm = NormalizedEnergy::from_actual(
            10, ComputeSubstrate::Photonic, &profile, TaskClass::TensorOp,
        );
        assert_eq!(norm.work_uj, 10); // on optimal: work = actual
        assert_eq!(norm.energy_overhead_uj(), 0);
    }

    #[test]
    fn test_normalized_energy_cpu_tensor_huge_overhead() {
        let profile = SubstrateEnergyProfile::default_profile();
        // CPU for TensorOp: proximity = 10/100000 = 0.0001
        let norm = NormalizedEnergy::from_actual(
            100_000, ComputeSubstrate::CPU, &profile, TaskClass::TensorOp,
        );
        assert_eq!(norm.actual_uj, 100_000);
        assert_eq!(norm.work_uj, 10); // 100000 × 0.0001 = 10 work units
        assert_eq!(norm.energy_overhead_uj(), 99_990);
    }

    // ── Serde ──

    #[test]
    fn test_substrate_bid_serde_roundtrip() {
        let bid = SubstrateBid {
            node_id: "n1".into(), task_id: "t1".into(),
            competence_score: 0.8, substrate: ComputeSubstrate::FPGA,
            task_class: TaskClass::Bitwise, estimated_energy_uj: 2,
            estimated_latency_us: 1, proximity_to_optimal: 1.0, composite_score: 0.8,
        };
        let json = serde_json::to_string(&bid).unwrap();
        let parsed: SubstrateBid = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.substrate, ComputeSubstrate::FPGA);
        assert_eq!(parsed.task_class, TaskClass::Bitwise);
        assert!((parsed.proximity_to_optimal - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalized_energy_serde_roundtrip() {
        let norm = NormalizedEnergy {
            work_uj: 1_000, actual_uj: 1_000,
            substrate: ComputeSubstrate::Neuromorphic, proximity: 1.0,
        };
        let json = serde_json::to_string(&norm).unwrap();
        let parsed: NormalizedEnergy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.substrate, ComputeSubstrate::Neuromorphic);
        assert_eq!(parsed.work_uj, 1_000);
        assert!((parsed.proximity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_profile_serde_roundtrip() {
        let profile = SubstrateEnergyProfile::default_profile();
        let json = serde_json::to_string(&profile).unwrap();
        let parsed: SubstrateEnergyProfile = serde_json::from_str(&json).unwrap();

        // Verify a few entries survive roundtrip
        for (tc, s) in [
            (TaskClass::Inference, ComputeSubstrate::LPU),
            (TaskClass::SpikingNetwork, ComputeSubstrate::Neuromorphic),
            (TaskClass::SignalProcessing, ComputeSubstrate::DSP),
            (TaskClass::PhotonicCompute, ComputeSubstrate::Photonic),
        ] {
            let original = profile.get(tc, s).unwrap();
            let restored = parsed.get(tc, s).unwrap();
            assert_eq!(original.energy_per_unit_uj, restored.energy_per_unit_uj);
        }
    }

    // ── The core thesis: honest substrate selection ──

    #[test]
    fn test_self_fulfilling_prophecy_avoided() {
        // Same work budget, different substrates → vastly different actual costs.
        // The budget expresses work to be done, not the cost of doing it badly.
        let profile = SubstrateEnergyProfile::default_profile();
        let work_budget = 50_000_u64; // 50,000 work units

        let on_lpu = NormalizedEnergy::to_actual(
            work_budget, ComputeSubstrate::LPU, &profile, TaskClass::Inference,
        );
        let on_cpu = NormalizedEnergy::to_actual(
            work_budget, ComputeSubstrate::CPU, &profile, TaskClass::Inference,
        );

        // Both represent the same amount of work
        assert_eq!(on_lpu.work_uj, on_cpu.work_uj);
        // LPU is far closer to optimal → actual cost is far less
        assert!(on_lpu.actual_uj < on_cpu.actual_uj);
        let ratio = on_cpu.actual_uj as f64 / on_lpu.actual_uj as f64;
        assert!(ratio > 100.0, "LPU should be >100× more efficient than CPU, got {ratio:.0}×");
        // CPU has massive overhead; LPU has moderate overhead
        assert!(on_cpu.energy_overhead_uj() > on_lpu.energy_overhead_uj());
    }

    #[test]
    fn test_self_fulfilling_prophecy_neuromorphic() {
        // Same thesis for neuromorphic: don't constrain spiking workloads to GPU
        let profile = SubstrateEnergyProfile::default_profile();
        let work_budget = 10_000_u64;

        let on_neuro = NormalizedEnergy::to_actual(
            work_budget, ComputeSubstrate::Neuromorphic, &profile, TaskClass::SpikingNetwork,
        );
        let on_gpu = NormalizedEnergy::to_actual(
            work_budget, ComputeSubstrate::GPU, &profile, TaskClass::SpikingNetwork,
        );

        // Neuromorphic is optimal (proximity=1.0): actual = budget
        assert_eq!(on_neuro.actual_uj, work_budget);
        assert_eq!(on_neuro.energy_overhead_uj(), 0);
        // GPU is 10× worse: actual = budget / 0.1 = 10× more
        assert!(on_gpu.actual_uj > on_neuro.actual_uj);
        let ratio = on_gpu.actual_uj as f64 / on_neuro.actual_uj as f64;
        assert!(ratio > 5.0, "Neuromorphic should be >5× more efficient than GPU for spiking, got {ratio:.0}×");
    }

    #[test]
    fn test_mcu_to_datacenter_full_span() {
        // The foundational test: we can express work on any substrate from MCU to WSE.
        // Work units are comparable across the full hardware span.
        let profile = SubstrateEnergyProfile::default_profile();

        // Inference on MCU (tiny model, huge cost)
        let mcu = NormalizedEnergy::from_actual(
            5_000_000, ComputeSubstrate::MCU, &profile, TaskClass::Inference,
        );
        // Same work on WSE (massive hardware, tiny cost)
        let wse = NormalizedEnergy::from_actual(
            300, ComputeSubstrate::WSE, &profile, TaskClass::Inference,
        );

        // Both express work in the same units (optimal-denominated)
        assert!(mcu.work_uj > 0);
        assert!(wse.work_uj > 0);

        // MCU: 5M actual × (200/5M) proximity = 200 work units
        // WSE: 300 actual × (200/300) proximity = 200 work units
        // Both did roughly the same amount of work
        let mcu_work = mcu.work_uj as f64;
        let wse_work = wse.work_uj as f64;
        assert!(
            (mcu_work / wse_work) > 0.5 && (mcu_work / wse_work) < 2.0,
            "MCU ({}) and WSE ({}) should represent similar amounts of work",
            mcu.work_uj, wse.work_uj,
        );

        // But MCU wastes far more energy to do the same work
        assert!(mcu.energy_overhead_uj() > wse.energy_overhead_uj());
    }
}
