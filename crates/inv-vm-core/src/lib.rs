pub mod attestation;
pub mod capability;
pub mod compliance;
pub mod compute;
pub mod disarm;
pub mod energy;
pub mod error;
pub mod identity;
pub mod payment;
pub mod policy;
pub mod privacy;

// Re-export commonly used types at the crate root.
pub use attestation::{
    AttestationEndorsement, AttestationEvidence, AttestationPolicy, AttestationResult,
    SupplyChainAttestation,
};
pub use capability::{
    AcceleratorInfo, AcceleratorType, Capability, CpuArch, EffectiveResources, GpuInfo, GpuVendor,
    HardwareInventory, MinimumRequirements, NodeClass, NodeRole, NodeState, ResourceContribution,
    TimeSchedule, TimeWindow,
};
pub use compliance::{
    ComplianceCheckResult, ComplianceChecker, ComplianceControl, ComplianceFramework,
    ComplianceReport, ComplianceStatus, ControlSeverity,
};
pub use compute::{
    ComputeCapabilities, ComputeParadigm, ComputeRequirements, ExecutionMode, MemoryTier,
    TeeCapability,
};
pub use disarm::{DisarmLevel, MeshTierHint, classify_query};
pub use energy::{EnergyBudget, EnergyReading, EnergySource, Joules, ThermalState, Watts};
pub use error::{InvError, InvResult};
pub use identity::{NodeId, OrgId, RegionId, WorkloadId};
pub use payment::{PaymentMethod, SettlementNetwork, WalletAddress, WalletAddressError};
pub use policy::{
    DataClassification, DataPolicy, EnergyPolicy, PlacementPolicy, PolicyEvaluator,
    PolicyViolation, SchedulingMode, ViolationSeverity, ViolationType,
};
pub use privacy::{PrivacyCapability, PrivacyRequirement, PrivacyTier, TrustLevel};
