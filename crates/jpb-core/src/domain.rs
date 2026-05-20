//! Shared domain traits for energy-metered stores.
//!
//! Every productivity domain crate (joule-crm, joule-commerce, joule-finance,
//! joule-hr, etc.) implements `DomainStore` so they all expose a consistent
//! energy tracking interface. This eliminates the 52-crate duplication of
//! independent `*Receipt` types and `track()` methods.

use crate::constants;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ── Energy receipt ───────────────────────────────────────────────────────────

/// Standard energy receipt returned from metered domain operations.
///
/// Replaces per-crate `CrmReceipt`, `EnergyReceipt`, etc. with one canonical
/// type. Callers can aggregate these into the energy ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyReceipt {
    /// Human-readable operation label (e.g., "create_contact", "checkout").
    pub operation: String,
    /// Energy consumed by this operation in microjoules.
    pub energy_uj: u64,
}

impl EnergyReceipt {
    /// Create a receipt with a fixed energy cost.
    pub fn fixed(operation: impl Into<String>, energy_uj: u64) -> Self {
        Self {
            operation: operation.into(),
            energy_uj: energy_uj.max(constants::MIN_OP_ENERGY_UJ),
        }
    }

    /// Create a receipt estimated from elapsed wall-clock time.
    ///
    /// Uses `IDLE_CPU_POWER_MW` (2 mW) * elapsed microseconds.
    pub fn timed(operation: impl Into<String>, start: Instant) -> Self {
        let elapsed_us = start.elapsed().as_micros() as f64;
        let energy_uj = (constants::IDLE_CPU_POWER_MW * elapsed_us) as u64;
        Self {
            operation: operation.into(),
            energy_uj: energy_uj.max(constants::MIN_OP_ENERGY_UJ),
        }
    }
}

// ── Domain store trait ───────────────────────────────────────────────────────

/// Trait for energy-metered domain stores.
///
/// Implementors track cumulative energy and can be queried for their total.
/// This provides a uniform interface across all 52+ productivity crates.
pub trait DomainStore {
    /// Returns the cumulative energy consumed by all operations (microjoules).
    fn total_energy_uj(&self) -> u64;

    /// Records energy from a receipt and returns it.
    fn record(&mut self, receipt: EnergyReceipt) -> EnergyReceipt;
}

// ── Domain error trait ───────────────────────────────────────────────────────

/// Marker trait for domain-specific error types.
///
/// All domain errors should implement `std::error::Error` + `DomainError`.
/// This allows generic error handling across domains without erasing types.
pub trait DomainError: std::error::Error + Send + Sync + 'static {}

// ── Default store mixin ──────────────────────────────────────────────────────

/// Helper struct that domain stores can embed to get standard energy tracking.
///
/// # Usage
/// ```ignore
/// pub struct CrmStore {
///     energy: EnergyAccumulator,
///     contacts: HashMap<Uuid, Contact>,
///     // ...
/// }
///
/// impl DomainStore for CrmStore {
///     fn total_energy_uj(&self) -> u64 { self.energy.total_uj() }
///     fn record(&mut self, receipt: EnergyReceipt) -> EnergyReceipt {
///         self.energy.record(receipt)
///     }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct EnergyAccumulator {
    total_uj: u64,
}

impl EnergyAccumulator {
    pub fn new() -> Self {
        Self { total_uj: 0 }
    }

    pub fn total_uj(&self) -> u64 {
        self.total_uj
    }

    pub fn record(&mut self, receipt: EnergyReceipt) -> EnergyReceipt {
        self.total_uj = self.total_uj.saturating_add(receipt.energy_uj);
        receipt
    }

    /// Convenience: create a fixed-cost receipt and record it in one call.
    pub fn track_fixed(&mut self, operation: impl Into<String>, energy_uj: u64) -> EnergyReceipt {
        let receipt = EnergyReceipt::fixed(operation, energy_uj);
        self.record(receipt)
    }

    /// Convenience: create a time-based receipt and record it in one call.
    pub fn track_timed(&mut self, operation: impl Into<String>, start: Instant) -> EnergyReceipt {
        let receipt = EnergyReceipt::timed(operation, start);
        self.record(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receipt_fixed_minimum() {
        let r = EnergyReceipt::fixed("test", 0);
        assert_eq!(r.energy_uj, constants::MIN_OP_ENERGY_UJ);
    }

    #[test]
    fn receipt_timed() {
        let start = Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let r = EnergyReceipt::timed("test", start);
        assert!(r.energy_uj >= 1);
    }

    #[test]
    fn accumulator_tracks() {
        let mut acc = EnergyAccumulator::new();
        assert_eq!(acc.total_uj(), 0);
        acc.track_fixed("op1", 100);
        assert_eq!(acc.total_uj(), 100);
        acc.track_fixed("op2", 200);
        assert_eq!(acc.total_uj(), 300);
    }

    #[test]
    fn accumulator_saturates() {
        let mut acc = EnergyAccumulator::new();
        acc.total_uj = u64::MAX - 10;
        acc.track_fixed("op", 100);
        assert_eq!(acc.total_uj(), u64::MAX);
    }

    #[test]
    fn receipt_serde_roundtrip() {
        let r = EnergyReceipt::fixed("checkout", 95);
        let json = serde_json::to_string(&r).unwrap();
        let parsed: EnergyReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.energy_uj, 95);
        assert_eq!(parsed.operation, "checkout");
    }

    #[test]
    fn track_timed_convenience() {
        let mut acc = EnergyAccumulator::new();
        let start = Instant::now();
        let r = acc.track_timed("slow_op", start);
        assert!(r.energy_uj >= 1);
        assert_eq!(acc.total_uj(), r.energy_uj);
    }
}
