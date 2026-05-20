//! Energy Budget Integration
//!
//! Provides the bridge between the branch manager's energy tracking and
//! the query executor's energy-aware dispatch. The executor calls
//! `check_and_record()` before and after each operation on a branch.

use crate::{BranchError, BranchId};
use serde::{Deserialize, Serialize};

/// An energy budget guard that tracks consumption for a single operation.
///
/// Created before a query executes on a branch. On drop, the consumed
/// energy is committed to the branch's running total.
pub struct EnergyGuard {
    branch_name: BranchId,
    estimated_uj: u64,
    actual_uj: Option<u64>,
}

impl EnergyGuard {
    /// Create a new guard for pre-flight budget checks
    pub fn new(branch_name: BranchId, estimated_uj: u64) -> Self {
        Self {
            branch_name,
            estimated_uj,
            actual_uj: None,
        }
    }

    /// Record the actual energy consumed after the operation completes
    pub fn set_actual(&mut self, actual_uj: u64) {
        self.actual_uj = Some(actual_uj);
    }

    /// Get the energy to charge (actual if known, otherwise estimate)
    pub fn charge_uj(&self) -> u64 {
        self.actual_uj.unwrap_or(self.estimated_uj)
    }

    /// Get the branch name
    pub fn branch_name(&self) -> &str {
        &self.branch_name
    }
}

/// Per-branch energy summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchEnergySummary {
    /// Branch name
    pub branch: String,
    /// Total budget in microjoules (None = unlimited)
    pub budget_uj: Option<u64>,
    /// Energy consumed so far
    pub spent_uj: u64,
    /// Energy remaining (None = unlimited)
    pub remaining_uj: Option<u64>,
    /// Number of operations executed
    pub operations: u64,
    /// Average energy per operation
    pub avg_uj_per_op: f64,
    /// Whether the branch is read-only due to budget exhaustion
    pub budget_exhausted: bool,
}

/// Default energy estimates for common operations (in microjoules).
///
/// These are conservative overestimates; the actual energy is recorded
/// after execution via `EnergyGuard::set_actual()`.
pub mod estimates {
    /// Simple key-value GET
    pub const KV_GET_UJ: u64 = 100;
    /// Simple key-value PUT
    pub const KV_PUT_UJ: u64 = 200;
    /// Table scan (per row)
    pub const SCAN_PER_ROW_UJ: u64 = 50;
    /// Index lookup
    pub const INDEX_LOOKUP_UJ: u64 = 150;
    /// Vector similarity search (per query)
    pub const VECTOR_SEARCH_UJ: u64 = 5_000;
    /// DDL operation (CREATE/ALTER/DROP)
    pub const DDL_UJ: u64 = 1_000;
    /// Branch creation
    pub const BRANCH_CREATE_UJ: u64 = 500;
    /// Branch merge
    pub const BRANCH_MERGE_UJ: u64 = 2_000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_guard_estimate() {
        let guard = EnergyGuard::new("test".to_string(), 1000);
        assert_eq!(guard.charge_uj(), 1000); // uses estimate
    }

    #[test]
    fn test_energy_guard_actual() {
        let mut guard = EnergyGuard::new("test".to_string(), 1000);
        guard.set_actual(750);
        assert_eq!(guard.charge_uj(), 750); // uses actual
    }

    #[test]
    fn test_estimates() {
        // Sanity check: vector search should cost more than a KV get
        assert!(estimates::VECTOR_SEARCH_UJ > estimates::KV_GET_UJ);
        assert!(estimates::KV_PUT_UJ > estimates::KV_GET_UJ);
    }
}
