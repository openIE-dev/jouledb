//! Copy-on-Write Database Branching with Energy Budgets
//!
//! Provides Neon-style instant database branching for JouleDB, with a unique
//! energy-budget mechanism that caps how many joules an agent or workflow can
//! spend on a branch before auto-rollback.
//!
//! ## Architecture
//!
//! ```text
//! main ──────────────────────────────────────────────►
//!            │ (branch at LSN 42)
//!            ├── feature-x ──── (CoW pages) ────────►
//!            │       │ (branch at LSN 58)
//!            │       └── experiment-1 ──── (budget: 5000 µJ) ──►
//!            │
//!            └── staging ────────────────────────────►
//! ```
//!
//! Branches share parent storage via copy-on-write: only modified pages are
//! duplicated. Each branch can optionally carry an energy budget that the
//! query executor enforces before dispatching.

pub mod energy;
pub mod manager;
pub mod storage;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

use joule_db_core::persistence::traits::LSN;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum BranchError {
    #[error("branch not found: {0}")]
    NotFound(String),

    #[error("branch already exists: {0}")]
    AlreadyExists(String),

    #[error("cannot delete the main branch")]
    CannotDeleteMain,

    #[error("energy budget exhausted: spent {spent_uj} µJ of {budget_uj} µJ budget")]
    EnergyBudgetExhausted { spent_uj: u64, budget_uj: u64 },

    #[error("merge conflict on {0} keys")]
    MergeConflict(usize),

    #[error("branch has uncommitted transactions")]
    UncommittedTransactions,

    #[error("storage error: {0}")]
    Storage(String),

    #[error("parent branch not found: {0}")]
    ParentNotFound(String),

    #[error("invalid branch name: {0}")]
    InvalidName(String),
}

// ============================================================================
// Branch metadata
// ============================================================================

/// Unique identifier for a branch
pub type BranchId = String;

/// Metadata for a single database branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Human-readable name (e.g., "feature-x", "staging")
    pub name: String,

    /// LSN at branch creation (the fork point)
    pub root_lsn: LSN,

    /// Current HEAD LSN on this branch
    pub head_lsn: LSN,

    /// Parent branch (None for "main")
    pub parent_id: Option<BranchId>,

    /// Creation timestamp (Unix millis)
    pub created_at: u64,

    /// Optional description
    pub description: Option<String>,

    /// Energy budget in microjoules (None = unlimited)
    pub energy_budget_uj: Option<u64>,

    /// Energy consumed so far in microjoules
    #[serde(
        serialize_with = "serialize_atomic_u64",
        deserialize_with = "deserialize_atomic_u64"
    )]
    pub energy_spent_uj: Arc<AtomicU64>,

    /// Whether this branch is read-only (e.g., after budget exhaustion)
    pub read_only: bool,

    /// Tags for grouping/filtering
    pub tags: Vec<String>,
}

fn serialize_atomic_u64<S: serde::Serializer>(
    val: &Arc<AtomicU64>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_u64(val.load(Ordering::Relaxed))
}

fn deserialize_atomic_u64<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Arc<AtomicU64>, D::Error> {
    let val = u64::deserialize(deserializer)?;
    Ok(Arc::new(AtomicU64::new(val)))
}

impl Branch {
    /// Remaining energy budget in microjoules (None = unlimited)
    pub fn energy_remaining_uj(&self) -> Option<u64> {
        self.energy_budget_uj.map(|budget| {
            let spent = self.energy_spent_uj.load(Ordering::Relaxed);
            budget.saturating_sub(spent)
        })
    }

    /// Check whether the branch can afford an operation with estimated cost
    pub fn can_afford(&self, estimated_uj: u64) -> bool {
        match self.energy_remaining_uj() {
            Some(remaining) => remaining >= estimated_uj,
            None => true,
        }
    }

    /// Record energy consumption. Returns Err if budget exhausted.
    pub fn record_energy(&self, consumed_uj: u64) -> Result<u64, BranchError> {
        let prev = self
            .energy_spent_uj
            .fetch_add(consumed_uj, Ordering::Relaxed);
        let total = prev + consumed_uj;

        if let Some(budget) = self.energy_budget_uj {
            if total > budget {
                return Err(BranchError::EnergyBudgetExhausted {
                    spent_uj: total,
                    budget_uj: budget,
                });
            }
        }
        Ok(total)
    }

    /// Check if this is the main/default branch
    pub fn is_main(&self) -> bool {
        self.parent_id.is_none() && self.name == "main"
    }
}

// ============================================================================
// Branch creation request
// ============================================================================

/// Request to create a new branch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateBranchRequest {
    /// Name for the new branch
    pub name: String,

    /// Parent branch to fork from (defaults to "main")
    pub parent: Option<BranchId>,

    /// Optional: fork at a specific LSN (defaults to parent HEAD)
    pub at_lsn: Option<LSN>,

    /// Optional energy budget in microjoules
    pub energy_budget_uj: Option<u64>,

    /// Optional description
    pub description: Option<String>,

    /// Optional tags
    pub tags: Vec<String>,
}

// ============================================================================
// Merge result
// ============================================================================

/// Result of merging a branch back into its parent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// Branch that was merged
    pub source_branch: String,

    /// Target branch merged into
    pub target_branch: String,

    /// Number of pages that were modified
    pub pages_merged: u64,

    /// Total energy consumed on the source branch
    pub energy_consumed_uj: u64,

    /// New HEAD LSN on the target after merge
    pub new_head_lsn: LSN,

    /// Whether the source branch was deleted after merge
    pub source_deleted: bool,
}

// ============================================================================
// Branch diff
// ============================================================================

/// A diff between a branch and its parent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchDiff {
    /// Pages that were modified on this branch
    pub modified_pages: Vec<u64>,

    /// Pages that were created on this branch
    pub new_pages: Vec<u64>,

    /// Pages that were deleted on this branch
    pub deleted_pages: Vec<u64>,

    /// WAL entries on this branch since fork
    pub wal_entries: u64,

    /// Total energy consumed
    pub energy_consumed_uj: u64,
}

// ============================================================================
// Branch listing / info
// ============================================================================

/// Summary info for listing branches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    pub name: String,
    pub parent: Option<String>,
    pub root_lsn: LSN,
    pub head_lsn: LSN,
    pub created_at: u64,
    pub energy_budget_uj: Option<u64>,
    pub energy_spent_uj: u64,
    pub read_only: bool,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

impl From<&Branch> for BranchInfo {
    fn from(b: &Branch) -> Self {
        Self {
            name: b.name.clone(),
            parent: b.parent_id.clone(),
            root_lsn: b.root_lsn,
            head_lsn: b.head_lsn,
            created_at: b.created_at,
            energy_budget_uj: b.energy_budget_uj,
            energy_spent_uj: b.energy_spent_uj.load(Ordering::Relaxed),
            read_only: b.read_only,
            tags: b.tags.clone(),
            description: b.description.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_branch(name: &str, budget: Option<u64>) -> Branch {
        Branch {
            name: name.to_string(),
            root_lsn: 0,
            head_lsn: 0,
            parent_id: Some("main".to_string()),
            created_at: 1709654400000,
            description: None,
            energy_budget_uj: budget,
            energy_spent_uj: Arc::new(AtomicU64::new(0)),
            read_only: false,
            tags: vec![],
        }
    }

    #[test]
    fn test_unlimited_budget() {
        let branch = make_branch("test", None);
        assert!(branch.can_afford(u64::MAX));
        assert!(branch.record_energy(1_000_000).is_ok());
    }

    #[test]
    fn test_energy_budget_enforcement() {
        let branch = make_branch("test", Some(1000));

        assert!(branch.can_afford(500));
        assert!(branch.record_energy(500).is_ok());

        assert!(branch.can_afford(500));
        assert!(branch.record_energy(500).is_ok());

        // Budget exhausted
        assert!(!branch.can_afford(1));
        assert!(branch.record_energy(1).is_err());
    }

    #[test]
    fn test_energy_remaining() {
        let branch = make_branch("test", Some(5000));
        assert_eq!(branch.energy_remaining_uj(), Some(5000));

        branch.record_energy(2000).unwrap();
        assert_eq!(branch.energy_remaining_uj(), Some(3000));
    }

    #[test]
    fn test_main_branch() {
        let main = Branch {
            name: "main".to_string(),
            root_lsn: 0,
            head_lsn: 100,
            parent_id: None,
            created_at: 0,
            description: None,
            energy_budget_uj: None,
            energy_spent_uj: Arc::new(AtomicU64::new(0)),
            read_only: false,
            tags: vec![],
        };
        assert!(main.is_main());

        let feature = make_branch("feature", None);
        assert!(!feature.is_main());
    }
}
