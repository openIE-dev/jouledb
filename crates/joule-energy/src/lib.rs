//! # Joule Energy
//!
//! Unified energy tracking, policy, and heterogeneous compute scheduling.
//! Builds on `joule-db-energy` (hardware monitoring, compute routing, budgets)
//! and adds application-level energy management for productivity tools.
//!
//! ## Architecture
//!
//! ```text
//! joule-db-energy (hardware layer)
//!   ├── EnergyMonitor      — real-time power/thermal/battery sampling
//!   ├── ComputeRouter      — algorithm → device routing
//!   ├── AdaptiveController  — thermal/power-aware parameter adaptation
//!   ├── EnergyBudget       — per-query budget enforcement
//!   └── OperationEnergyTracker — RAII per-operation measurement
//!
//! joule-energy (application layer — this crate)
//!   ├── EnergyLedger       — cumulative mJ tracking per compute unit (CPU/GPU/NPU)
//!   ├── EnergyPolicy       — quality scaling based on thermal/battery/power state
//!   ├── ComputeScheduler   — task queue with priority, deadline, energy budget
//!   ├── OperationTag       — per-operation energy attribution for productivity tools
//!   └── re-exports joule-db-energy publicly
//! ```

pub mod context;
pub mod ledger;
pub mod policy;
pub mod scheduler;
pub mod tag;

// Re-export the hardware layer so consumers only need one dependency.
pub use joule_db_energy as hw;
pub use joule_db_energy::{
    ComputeRouter, EnergyBudget, EnergySnapshot, ExecutionHint,
    HardwareAdvisor, OperationEnergyTracker, RoutingDecision, ThermalState,
    detect_platform, PlatformInfo,
};
#[cfg(not(target_arch = "wasm32"))]
pub use joule_db_energy::EnergyMonitor;
pub use joule_db_energy::tracker::{AlgorithmType, DeviceTarget, OperationType};
