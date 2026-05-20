//! Branch Manager — create, list, merge, and delete branches
//!
//! The manager owns all branch metadata and coordinates CoW storage.

use crate::{
    Branch, BranchDiff, BranchError, BranchId, BranchInfo, CreateBranchRequest, MergeResult,
};
use joule_db_core::persistence::traits::LSN;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// The branch manager tracks all branches and coordinates operations.
pub struct BranchManager {
    /// All branches indexed by name
    branches: RwLock<HashMap<BranchId, Branch>>,

    /// Current HEAD LSN of the main branch (from the storage engine)
    main_head_lsn: AtomicU64,
}

impl BranchManager {
    /// Create a new branch manager with an initial "main" branch
    pub fn new(main_head_lsn: LSN) -> Self {
        let mut branches = HashMap::new();
        branches.insert(
            "main".to_string(),
            Branch {
                name: "main".to_string(),
                root_lsn: 0,
                head_lsn: main_head_lsn,
                parent_id: None,
                created_at: now_millis(),
                description: Some("Default branch".to_string()),
                energy_budget_uj: None,
                energy_spent_uj: Arc::new(AtomicU64::new(0)),
                read_only: false,
                tags: vec![],
            },
        );

        Self {
            branches: RwLock::new(branches),
            main_head_lsn: AtomicU64::new(main_head_lsn),
        }
    }

    /// Create a new branch forked from a parent
    pub fn create_branch(&self, req: CreateBranchRequest) -> Result<BranchInfo, BranchError> {
        let mut branches = self
            .branches
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        // Validate branch name
        if req.name.is_empty() {
            return Err(BranchError::InvalidName(
                "Branch name cannot be empty".to_string(),
            ));
        }

        // Check name doesn't already exist
        if branches.contains_key(&req.name) {
            return Err(BranchError::AlreadyExists(req.name));
        }

        // Resolve parent
        let parent_name = req.parent.as_deref().unwrap_or("main");
        let parent = branches
            .get(parent_name)
            .ok_or_else(|| BranchError::ParentNotFound(parent_name.to_string()))?;

        // Fork point: requested LSN or parent HEAD
        let fork_lsn = req.at_lsn.unwrap_or(parent.head_lsn);

        let branch = Branch {
            name: req.name.clone(),
            root_lsn: fork_lsn,
            head_lsn: fork_lsn,
            parent_id: Some(parent_name.to_string()),
            created_at: now_millis(),
            description: req.description,
            energy_budget_uj: req.energy_budget_uj,
            energy_spent_uj: Arc::new(AtomicU64::new(0)),
            read_only: false,
            tags: req.tags,
        };

        let info = BranchInfo::from(&branch);
        branches.insert(req.name, branch);

        Ok(info)
    }

    /// List all branches
    pub fn list_branches(&self) -> Result<Vec<BranchInfo>, BranchError> {
        let branches = self
            .branches
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        let mut infos: Vec<BranchInfo> = branches.values().map(BranchInfo::from).collect();
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    /// Get info about a specific branch
    pub fn get_branch(&self, name: &str) -> Result<BranchInfo, BranchError> {
        let branches = self
            .branches
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        branches
            .get(name)
            .map(BranchInfo::from)
            .ok_or_else(|| BranchError::NotFound(name.to_string()))
    }

    /// Delete a branch (cannot delete "main")
    pub fn delete_branch(&self, name: &str) -> Result<BranchInfo, BranchError> {
        if name == "main" {
            return Err(BranchError::CannotDeleteMain);
        }

        let mut branches = self
            .branches
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        // Check no children reference this branch
        let has_children = branches
            .values()
            .any(|b| b.parent_id.as_deref() == Some(name));
        if has_children {
            return Err(BranchError::Storage(format!(
                "branch '{}' has child branches — delete them first",
                name
            )));
        }

        let branch = branches
            .remove(name)
            .ok_or_else(|| BranchError::NotFound(name.to_string()))?;

        Ok(BranchInfo::from(&branch))
    }

    /// Check if a branch can afford an operation with the given estimated energy cost
    pub fn check_energy_budget(
        &self,
        branch_name: &str,
        estimated_uj: u64,
    ) -> Result<bool, BranchError> {
        let branches = self
            .branches
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        let branch = branches
            .get(branch_name)
            .ok_or_else(|| BranchError::NotFound(branch_name.to_string()))?;

        Ok(branch.can_afford(estimated_uj))
    }

    /// Record energy consumption on a branch
    pub fn record_energy(&self, branch_name: &str, consumed_uj: u64) -> Result<u64, BranchError> {
        let branches = self
            .branches
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        let branch = branches
            .get(branch_name)
            .ok_or_else(|| BranchError::NotFound(branch_name.to_string()))?;

        branch.record_energy(consumed_uj)
    }

    /// Advance the HEAD LSN of a branch (called after successful writes)
    pub fn advance_head(&self, branch_name: &str, new_lsn: LSN) -> Result<(), BranchError> {
        let mut branches = self
            .branches
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        let branch = branches
            .get_mut(branch_name)
            .ok_or_else(|| BranchError::NotFound(branch_name.to_string()))?;

        if new_lsn > branch.head_lsn {
            branch.head_lsn = new_lsn;

            // Keep main_head_lsn in sync
            if branch_name == "main" {
                self.main_head_lsn.store(new_lsn, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    /// Merge a branch into its parent (fast-forward or 3-way)
    ///
    /// For now, this implements fast-forward merge: the parent's HEAD advances
    /// to include all changes from the source branch.
    pub fn merge_branch(
        &self,
        source_name: &str,
        delete_after: bool,
    ) -> Result<MergeResult, BranchError> {
        let mut branches = self
            .branches
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        // Get source info
        let source = branches
            .get(source_name)
            .ok_or_else(|| BranchError::NotFound(source_name.to_string()))?;

        let parent_name = source
            .parent_id
            .clone()
            .ok_or_else(|| BranchError::Storage("cannot merge main into itself".to_string()))?;

        let source_head = source.head_lsn;
        let source_energy = source.energy_spent_uj.load(Ordering::Relaxed);

        // Advance parent HEAD to include source changes
        let parent = branches
            .get_mut(&parent_name)
            .ok_or_else(|| BranchError::ParentNotFound(parent_name.clone()))?;

        let new_head = source_head.max(parent.head_lsn);
        parent.head_lsn = new_head;

        let result = MergeResult {
            source_branch: source_name.to_string(),
            target_branch: parent_name.clone(),
            pages_merged: 0, // TODO: track from CoW storage
            energy_consumed_uj: source_energy,
            new_head_lsn: new_head,
            source_deleted: delete_after,
        };

        if delete_after {
            // Don't orphan child branches
            let has_children = branches
                .values()
                .any(|b| b.parent_id.as_deref() == Some(source_name));
            if has_children {
                return Err(BranchError::Storage(format!(
                    "cannot delete branch '{}' after merge — it has child branches",
                    source_name
                )));
            }
            branches.remove(source_name);
        }

        Ok(result)
    }

    /// Get a diff between a branch and its parent
    pub fn diff_branch(&self, branch_name: &str) -> Result<BranchDiff, BranchError> {
        let branches = self
            .branches
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        let branch = branches
            .get(branch_name)
            .ok_or_else(|| BranchError::NotFound(branch_name.to_string()))?;

        Ok(BranchDiff {
            modified_pages: vec![],
            new_pages: vec![],
            deleted_pages: vec![],
            wal_entries: branch.head_lsn.saturating_sub(branch.root_lsn),
            energy_consumed_uj: branch.energy_spent_uj.load(Ordering::Relaxed),
        })
    }

    /// Get current main HEAD LSN
    pub fn main_head_lsn(&self) -> LSN {
        self.main_head_lsn.load(Ordering::Relaxed)
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_branch() {
        let mgr = BranchManager::new(100);

        let info = mgr
            .create_branch(CreateBranchRequest {
                name: "feature-x".to_string(),
                parent: None,
                at_lsn: None,
                energy_budget_uj: Some(10_000),
                description: Some("Test feature".to_string()),
                tags: vec!["test".to_string()],
            })
            .unwrap();

        assert_eq!(info.name, "feature-x");
        assert_eq!(info.root_lsn, 100); // forked from main HEAD
        assert_eq!(info.energy_budget_uj, Some(10_000));
        assert_eq!(info.parent, Some("main".to_string()));
    }

    #[test]
    fn test_duplicate_branch_name() {
        let mgr = BranchManager::new(0);

        mgr.create_branch(CreateBranchRequest {
            name: "dup".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        })
        .unwrap();

        let result = mgr.create_branch(CreateBranchRequest {
            name: "dup".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        });

        assert!(matches!(result, Err(BranchError::AlreadyExists(_))));
    }

    #[test]
    fn test_cannot_delete_main() {
        let mgr = BranchManager::new(0);
        assert!(matches!(
            mgr.delete_branch("main"),
            Err(BranchError::CannotDeleteMain)
        ));
    }

    #[test]
    fn test_delete_branch() {
        let mgr = BranchManager::new(0);

        mgr.create_branch(CreateBranchRequest {
            name: "temp".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        })
        .unwrap();

        assert!(mgr.delete_branch("temp").is_ok());
        assert!(mgr.get_branch("temp").is_err());
    }

    #[test]
    fn test_merge_branch() {
        let mgr = BranchManager::new(100);

        mgr.create_branch(CreateBranchRequest {
            name: "feature".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        })
        .unwrap();

        // Simulate writes on the feature branch
        mgr.advance_head("feature", 150).unwrap();

        let result = mgr.merge_branch("feature", true).unwrap();
        assert_eq!(result.source_branch, "feature");
        assert_eq!(result.target_branch, "main");
        assert_eq!(result.new_head_lsn, 150);
        assert!(result.source_deleted);

        // Feature branch should be gone
        assert!(mgr.get_branch("feature").is_err());

        // Main should be at 150
        let main = mgr.get_branch("main").unwrap();
        assert_eq!(main.head_lsn, 150);
    }

    #[test]
    fn test_energy_budget_tracking() {
        let mgr = BranchManager::new(0);

        mgr.create_branch(CreateBranchRequest {
            name: "agent-explore".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: Some(1000),
            description: None,
            tags: vec!["agent".to_string()],
        })
        .unwrap();

        assert!(mgr.check_energy_budget("agent-explore", 500).unwrap());
        mgr.record_energy("agent-explore", 500).unwrap();

        assert!(mgr.check_energy_budget("agent-explore", 500).unwrap());
        mgr.record_energy("agent-explore", 500).unwrap();

        // Exhausted
        assert!(!mgr.check_energy_budget("agent-explore", 1).unwrap());
        assert!(mgr.record_energy("agent-explore", 1).is_err());
    }

    #[test]
    fn test_list_branches() {
        let mgr = BranchManager::new(0);

        mgr.create_branch(CreateBranchRequest {
            name: "b".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        })
        .unwrap();

        mgr.create_branch(CreateBranchRequest {
            name: "a".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        })
        .unwrap();

        let list = mgr.list_branches().unwrap();
        assert_eq!(list.len(), 3); // main + a + b
        assert_eq!(list[0].name, "a"); // sorted
    }

    #[test]
    fn test_nested_branches() {
        let mgr = BranchManager::new(100);

        mgr.create_branch(CreateBranchRequest {
            name: "dev".to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj: None,
            description: None,
            tags: vec![],
        })
        .unwrap();

        mgr.advance_head("dev", 120).unwrap();

        // Branch from dev, not main
        let nested = mgr
            .create_branch(CreateBranchRequest {
                name: "dev-experiment".to_string(),
                parent: Some("dev".to_string()),
                at_lsn: None,
                energy_budget_uj: Some(5000),
                description: None,
                tags: vec![],
            })
            .unwrap();

        assert_eq!(nested.parent, Some("dev".to_string()));
        assert_eq!(nested.root_lsn, 120); // forked from dev HEAD
    }
}
