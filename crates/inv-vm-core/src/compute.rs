//! Compute paradigm types for the Invisible Infrastructure platform.
//!
//! Defines execution paradigms, TEE capabilities, memory tiers, and execution
//! modes used across the platform for workload placement and scheduling.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Compute execution paradigm.
///
/// Each variant represents a distinct isolation / execution model, from
/// lightweight WASM sandboxes through neuromorphic processors and
/// confidential computing enclaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeParadigm {
    /// WebAssembly sandbox.
    Wasm,
    /// OCI container.
    Container,
    /// Lightweight VM (Firecracker, Cloud Hypervisor).
    MicroVm,
    /// Single-address-space OS.
    Unikernel,
    /// Spiking neural network processor.
    Neuromorphic,
    /// TEE-backed confidential compute.
    Confidential,
}

impl fmt::Display for ComputeParadigm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wasm => write!(f, "Wasm"),
            Self::Container => write!(f, "Container"),
            Self::MicroVm => write!(f, "MicroVM"),
            Self::Unikernel => write!(f, "Unikernel"),
            Self::Neuromorphic => write!(f, "Neuromorphic"),
            Self::Confidential => write!(f, "Confidential"),
        }
    }
}

/// Trusted Execution Environment capability.
///
/// Ordered from least to most capable so that derived `Ord`
/// comparisons work correctly: `None < SevSnp < Tdx < ArmCca`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TeeCapability {
    /// No TEE.
    None,
    /// AMD SEV-SNP.
    SevSnp,
    /// Intel TDX.
    Tdx,
    /// ARM Confidential Compute Architecture.
    ArmCca,
}

impl fmt::Display for TeeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::SevSnp => write!(f, "SEV-SNP"),
            Self::Tdx => write!(f, "TDX"),
            Self::ArmCca => write!(f, "ARM-CCA"),
        }
    }
}

/// Memory tier describing the class of memory available.
///
/// Ordered from fastest to slowest so that derived `Ord`
/// comparisons express capability: `DramOnly < CxlNear < CxlFar`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MemoryTier {
    /// Standard DRAM.
    DramOnly,
    /// CXL-attached, low latency (~200 ns).
    CxlNear,
    /// CXL-attached, higher latency (~1 μs).
    CxlFar,
}

impl MemoryTier {
    /// Approximate access latency in nanoseconds for this memory tier.
    pub fn latency_ns(&self) -> u64 {
        match self {
            Self::DramOnly => 80,
            Self::CxlNear => 200,
            Self::CxlFar => 1000,
        }
    }
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DramOnly => write!(f, "DRAM-Only"),
            Self::CxlNear => write!(f, "CXL-Near"),
            Self::CxlFar => write!(f, "CXL-Far"),
        }
    }
}

/// Execution mode for a workload.
///
/// Controls how the runtime handles state, determinism, and resilience.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Standard execution.
    Normal,
    /// Replay-safe deterministic execution.
    Deterministic,
    /// Journal-backed durable execution.
    Durable,
    /// For energy-harvesting devices.
    Intermittent,
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Normal => write!(f, "Normal"),
            Self::Deterministic => write!(f, "Deterministic"),
            Self::Durable => write!(f, "Durable"),
            Self::Intermittent => write!(f, "Intermittent"),
        }
    }
}

/// A workload's compute requirements — the minimum resources and capabilities
/// it demands from a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequirements {
    /// Required execution paradigm.
    pub paradigm: ComputeParadigm,
    /// Minimum TEE capability the node must provide.
    pub tee: TeeCapability,
    /// Minimum memory tier the node must support.
    pub memory_tier: MemoryTier,
    /// Required execution mode.
    pub mode: ExecutionMode,
    /// Minimum number of virtual CPUs.
    pub min_vcpus: u32,
    /// Minimum memory in megabytes.
    pub min_memory_mb: u64,
}

/// A node's compute capabilities — what execution paradigms, TEE levels,
/// memory tiers, and resources it can offer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    /// Execution paradigms this node supports.
    pub supported_paradigms: Vec<ComputeParadigm>,
    /// TEE capability available on this node.
    pub tee: TeeCapability,
    /// Memory tier available on this node.
    pub memory_tier: MemoryTier,
    /// Execution modes this node supports.
    pub supported_modes: Vec<ExecutionMode>,
    /// Number of virtual CPUs available.
    pub available_vcpus: u32,
    /// Available memory in megabytes.
    pub available_memory_mb: u64,
}

impl ComputeCapabilities {
    /// Check whether this capability satisfies the given compute requirements.
    ///
    /// A capability satisfies a requirement when:
    /// - `paradigm` is in `supported_paradigms`
    /// - `tee >= requirements.tee`
    /// - `memory_tier >= requirements.memory_tier`
    /// - `mode` is in `supported_modes`
    /// - `available_vcpus >= min_vcpus`
    /// - `available_memory_mb >= min_memory_mb`
    pub fn satisfies(&self, requirements: &ComputeRequirements) -> bool {
        if !self.supported_paradigms.contains(&requirements.paradigm) {
            return false;
        }
        if self.tee < requirements.tee {
            return false;
        }
        if self.memory_tier < requirements.memory_tier {
            return false;
        }
        if !self.supported_modes.contains(&requirements.mode) {
            return false;
        }
        if self.available_vcpus < requirements.min_vcpus {
            return false;
        }
        if self.available_memory_mb < requirements.min_memory_mb {
            return false;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_paradigm_display() {
        assert_eq!(format!("{}", ComputeParadigm::Wasm), "Wasm");
        assert_eq!(format!("{}", ComputeParadigm::Container), "Container");
        assert_eq!(format!("{}", ComputeParadigm::MicroVm), "MicroVM");
        assert_eq!(format!("{}", ComputeParadigm::Unikernel), "Unikernel");
        assert_eq!(format!("{}", ComputeParadigm::Neuromorphic), "Neuromorphic");
        assert_eq!(format!("{}", ComputeParadigm::Confidential), "Confidential");
    }

    #[test]
    fn tee_capability_ordering() {
        assert!(TeeCapability::None < TeeCapability::SevSnp);
        assert!(TeeCapability::SevSnp < TeeCapability::Tdx);
        assert!(TeeCapability::Tdx < TeeCapability::ArmCca);
    }

    #[test]
    fn memory_tier_ordering() {
        assert!(MemoryTier::DramOnly < MemoryTier::CxlNear);
        assert!(MemoryTier::CxlNear < MemoryTier::CxlFar);
    }

    #[test]
    fn memory_tier_latency() {
        assert_eq!(MemoryTier::DramOnly.latency_ns(), 80);
        assert_eq!(MemoryTier::CxlNear.latency_ns(), 200);
        assert_eq!(MemoryTier::CxlFar.latency_ns(), 1000);
    }

    #[test]
    fn execution_mode_display() {
        assert_eq!(format!("{}", ExecutionMode::Normal), "Normal");
        assert_eq!(format!("{}", ExecutionMode::Deterministic), "Deterministic");
        assert_eq!(format!("{}", ExecutionMode::Durable), "Durable");
        assert_eq!(format!("{}", ExecutionMode::Intermittent), "Intermittent");
    }

    // Helper to build a standard set of capabilities for tests.
    fn base_capabilities() -> ComputeCapabilities {
        ComputeCapabilities {
            supported_paradigms: vec![ComputeParadigm::Wasm, ComputeParadigm::Container],
            tee: TeeCapability::SevSnp,
            memory_tier: MemoryTier::CxlNear,
            supported_modes: vec![ExecutionMode::Normal, ExecutionMode::Deterministic],
            available_vcpus: 8,
            available_memory_mb: 16384,
        }
    }

    // Helper to build a standard set of requirements for tests.
    fn base_requirements() -> ComputeRequirements {
        ComputeRequirements {
            paradigm: ComputeParadigm::Wasm,
            tee: TeeCapability::SevSnp,
            memory_tier: MemoryTier::CxlNear,
            mode: ExecutionMode::Normal,
            min_vcpus: 4,
            min_memory_mb: 8192,
        }
    }

    #[test]
    fn satisfies_exact_match() {
        let cap = base_capabilities();
        let req = base_requirements();
        assert!(cap.satisfies(&req));
    }

    #[test]
    fn satisfies_higher_tee() {
        let mut cap = base_capabilities();
        cap.tee = TeeCapability::ArmCca;
        let req = base_requirements();
        assert!(cap.satisfies(&req));
    }

    #[test]
    fn does_not_satisfy_paradigm_mismatch() {
        let cap = base_capabilities();
        let mut req = base_requirements();
        req.paradigm = ComputeParadigm::Neuromorphic;
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn does_not_satisfy_insufficient_tee() {
        let mut cap = base_capabilities();
        cap.tee = TeeCapability::None;
        let req = base_requirements();
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn does_not_satisfy_insufficient_memory() {
        let mut cap = base_capabilities();
        cap.available_memory_mb = 4096;
        let req = base_requirements();
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn does_not_satisfy_insufficient_vcpus() {
        let mut cap = base_capabilities();
        cap.available_vcpus = 2;
        let req = base_requirements();
        assert!(!cap.satisfies(&req));
    }

    #[test]
    fn serialization_roundtrip() {
        let req = ComputeRequirements {
            paradigm: ComputeParadigm::Confidential,
            tee: TeeCapability::Tdx,
            memory_tier: MemoryTier::CxlFar,
            mode: ExecutionMode::Durable,
            min_vcpus: 16,
            min_memory_mb: 65536,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ComputeRequirements = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.paradigm, ComputeParadigm::Confidential);
        assert_eq!(parsed.tee, TeeCapability::Tdx);
        assert_eq!(parsed.memory_tier, MemoryTier::CxlFar);
        assert_eq!(parsed.mode, ExecutionMode::Durable);
        assert_eq!(parsed.min_vcpus, 16);
        assert_eq!(parsed.min_memory_mb, 65536);

        let cap = base_capabilities();
        let json = serde_json::to_string(&cap).unwrap();
        let parsed: ComputeCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.supported_paradigms.len(), 2);
        assert_eq!(parsed.tee, TeeCapability::SevSnp);
        assert_eq!(parsed.memory_tier, MemoryTier::CxlNear);
        assert_eq!(parsed.available_vcpus, 8);
        assert_eq!(parsed.available_memory_mb, 16384);
    }
}
