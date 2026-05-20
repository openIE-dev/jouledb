//! Shared energy accounting traits for the system-wide energy chain.
//!
//! **kernel measures → mesh aggregates → database indexes → applications display**

use chrono::{DateTime, Utc};
use std::time::Duration;

/// Unique identifier for an in-flight energy-tracked operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub u64);

/// Energy source classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergySource {
    Grid,
    Battery,
    Solar,
    Unknown,
}

/// System thermal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// A receipt issued when an energy-tracked operation completes.
#[derive(Debug, Clone)]
pub struct EnergyReceipt {
    pub operation_id: OperationId,
    pub label: String,
    pub joules: f64,
    pub duration: Duration,
    pub avg_watts: f64,
    pub thermal_state: ThermalState,
    pub source: EnergySource,
    pub completed_at: DateTime<Utc>,
}

/// A point-in-time energy snapshot.
#[derive(Debug, Clone)]
pub struct EnergySnapshot {
    pub current_watts: f64,
    pub thermal_state: ThermalState,
    pub source: EnergySource,
    pub cumulative_joules: f64,
    pub budget_remaining_joules: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

/// Errors from energy accounting operations.
#[derive(Debug, thiserror::Error)]
pub enum EnergyError {
    #[error("unknown operation: {0:?}")]
    UnknownOperation(OperationId),
    #[error("budget exceeded: {0}")]
    BudgetExceeded(String),
}

/// Trait for components that track energy consumption of operations.
pub trait EnergyAccountant: Send + Sync {
    /// Begin tracking an operation. Returns an ID to pass to `end_operation`.
    fn begin_operation(&self, label: &str) -> OperationId;
    /// End tracking and receive an energy receipt.
    fn end_operation(&self, id: OperationId) -> Result<EnergyReceipt, EnergyError>;
    /// Get a point-in-time energy snapshot.
    fn snapshot(&self) -> EnergySnapshot;
}
