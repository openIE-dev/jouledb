//! JouleAgentFS: Copy-on-Write Agent Sandbox, Virtual Filesystem, Scratchpad KV,
//! and Hash-Chained Tool Audit Trail with Microjoule Budgets.
//!
//! Inspired by the 2025-2026 convergence of Turso AgentFS (embedded POSIX file tree +
//! KV working memory + append-only tool audit log) and Databricks Lakebase / Neon
//! (instant CoW branching), with JouleDB's unique addition: **every speculative agent
//! workspace branch is metered and hard-capped in microjoules (`energy_budget_uj`)**.

use crate::manager::BranchManager;
use crate::{BranchError, BranchInfo, CreateBranchRequest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Default modeled energy cost per file/KV metadata operation (1 µJ) + 1 µJ per 4 KiB payload.
const BASE_OP_ENERGY_UJ: u64 = 1;

fn payload_energy_uj(bytes_len: usize) -> u64 {
    BASE_OP_ENERGY_UJ + (bytes_len as u64).div_ceil(4096)
}

/// Normalize a virtual POSIX path (e.g. `"workspace//src/main.rs"` -> `"/workspace/src/main.rs"`).
pub fn normalize_vfs_path(path: &str) -> Result<String, BranchError> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(BranchError::InvalidName("empty AgentFS path".to_string()));
    }
    let mut parts: Vec<&str> = Vec::new();
    for seg in trimmed.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            valid => parts.push(valid),
        }
    }
    Ok(format!("/{}", parts.join("/")))
}

/// Deterministic 128-bit FNV-1a hex digest (zero external crypto dep required).
fn digest_hex_128(chunks: &[&[u8]]) -> String {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0x8422_2325_cbf2_9ce4;
    for chunk in chunks {
        for &b in *chunk {
            h1 ^= b as u64;
            h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
            h2 = h2.wrapping_mul(0x0000_0100_0000_01b3) ^ (b as u64);
        }
    }
    format!("{h1:016x}{h2:016x}")
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Metadata for a virtual file in [`JouleAgentFs`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentFileStat {
    pub path: String,
    pub size_bytes: usize,
    pub content_digest: String,
    pub modified_at_ms: u64,
    pub version: u64,
    pub origin_branch: String,
}

#[derive(Debug, Clone)]
struct FileEntry {
    data: Vec<u8>,
    stat: AgentFileStat,
}

#[derive(Debug, Clone)]
enum OverlaySlot<T> {
    Present(T),
    Deleted { version: u64 },
}

/// Tamper-evident hash-chained tool invocation record on an agent branch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolAuditRecord {
    pub seq: u64,
    pub branch: String,
    pub timestamp_ms: u64,
    pub tool_name: String,
    pub args_summary: String,
    pub outcome: String,
    pub energy_uj: u64,
    pub prev_hash: String,
    pub entry_hash: String,
}

/// Structural diff between a child workspace branch and its parent at merge/inspection time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceDiff {
    pub branch: String,
    pub parent: String,
    pub added_files: Vec<String>,
    pub modified_files: Vec<String>,
    pub deleted_files: Vec<String>,
    pub mutated_kv_keys: Vec<String>,
    pub conflicting_paths_or_keys: Vec<String>,
    pub tool_calls_recorded: usize,
    pub energy_spent_uj: u64,
    pub energy_remaining_uj: Option<u64>,
}

#[derive(Debug, Default)]
struct BranchWorkspaceState {
    /// Snapshot of parent file versions at fork time (for 3-way conflict detection on merge).
    base_file_versions: HashMap<String, u64>,
    /// Snapshot of parent KV versions at fork time.
    base_kv_versions: HashMap<String, u64>,
    /// Branch-local Copy-on-Write file overlay.
    files: BTreeMap<String, OverlaySlot<FileEntry>>,
    /// Branch-local Copy-on-Write scratchpad KV overlay `(value, version)`.
    kv: BTreeMap<String, OverlaySlot<(serde_json::Value, u64)>>,
    /// Branch-local append-only hash-chained tool audit log.
    audit_log: Vec<ToolAuditRecord>,
}

/// Copy-on-Write multi-agent workspace engine layered on [`BranchManager`].
pub struct JouleAgentFs {
    branches: Arc<BranchManager>,
    workspaces: RwLock<HashMap<String, BranchWorkspaceState>>,
}

impl JouleAgentFs {
    /// Create a new `JouleAgentFs` attached to an existing `BranchManager`.
    pub fn new(branches: Arc<BranchManager>) -> Self {
        let mut map = HashMap::new();
        map.insert("main".to_string(), BranchWorkspaceState::default());
        Self {
            branches,
            workspaces: RwLock::new(map),
        }
    }

    /// Create a standalone `JouleAgentFs` with a fresh `BranchManager` at LSN 0.
    pub fn standalone() -> Self {
        Self::new(Arc::new(BranchManager::new(0)))
    }

    /// Access the underlying `BranchManager`.
    pub fn branch_manager(&self) -> &Arc<BranchManager> {
        &self.branches
    }

    /// O(1) Copy-on-Write fork of an agent workspace from `parent` (default `"main"`),
    /// optionally bounded by a strict `energy_budget_uj` microjoule ceiling.
    pub fn fork_workspace(
        &self,
        name: &str,
        parent: Option<&str>,
        energy_budget_uj: Option<u64>,
        description: Option<String>,
    ) -> Result<BranchInfo, BranchError> {
        let parent_name = parent.unwrap_or("main");
        let info = self.branches.create_branch(CreateBranchRequest {
            name: name.to_string(),
            parent: Some(parent_name.to_string()),
            at_lsn: None,
            energy_budget_uj,
            description,
            tags: vec!["agentfs".to_string()],
        })?;

        // Capture lightweight version watermarks for 3-way conflict detection (no payload copy!)
        let base_files = self.effective_file_versions(parent_name)?;
        let base_kv = self.effective_kv_versions(parent_name)?;

        let mut ws = self
            .workspaces
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        ws.insert(
            name.to_string(),
            BranchWorkspaceState {
                base_file_versions: base_files,
                base_kv_versions: base_kv,
                files: BTreeMap::new(),
                kv: BTreeMap::new(),
                audit_log: Vec::new(),
            },
        );
        Ok(info)
    }

    /// Write a virtual file into `branch`, enforcing the branch's microjoule ceiling.
    pub fn write_file(
        &self,
        branch: &str,
        path: &str,
        content: &[u8],
    ) -> Result<AgentFileStat, BranchError> {
        let norm = normalize_vfs_path(path)?;
        let cost_uj = payload_energy_uj(content.len());
        self.charge_branch_or_fail(branch, cost_uj)?;

        let prev_ver = self
            .resolve_file_entry(branch, &norm)?
            .map(|e| e.stat.version)
            .unwrap_or(0);
        let stat = AgentFileStat {
            path: norm.clone(),
            size_bytes: content.len(),
            content_digest: digest_hex_128(&[content]),
            modified_at_ms: now_millis(),
            version: prev_ver + 1,
            origin_branch: branch.to_string(),
        };

        let mut ws = self
            .workspaces
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let state = ws.entry(branch.to_string()).or_default();
        state.files.insert(
            norm,
            OverlaySlot::Present(FileEntry {
                data: content.to_vec(),
                stat: stat.clone(),
            }),
        );
        Ok(stat)
    }

    /// Read a virtual file from `branch` (resolving CoW overlays up the ancestor chain).
    pub fn read_file(&self, branch: &str, path: &str) -> Result<Option<Vec<u8>>, BranchError> {
        let norm = normalize_vfs_path(path)?;
        self.charge_branch_or_fail(branch, BASE_OP_ENERGY_UJ)?;
        Ok(self.resolve_file_entry(branch, &norm)?.map(|e| e.data))
    }

    /// Stat a virtual file without copying its payload.
    pub fn stat_file(
        &self,
        branch: &str,
        path: &str,
    ) -> Result<Option<AgentFileStat>, BranchError> {
        let norm = normalize_vfs_path(path)?;
        Ok(self.resolve_file_entry(branch, &norm)?.map(|e| e.stat))
    }

    /// Delete a virtual file on `branch` (records a CoW tombstone over parent).
    pub fn delete_file(&self, branch: &str, path: &str) -> Result<bool, BranchError> {
        let norm = normalize_vfs_path(path)?;
        self.charge_branch_or_fail(branch, BASE_OP_ENERGY_UJ)?;
        let prev = self.resolve_file_entry(branch, &norm)?;
        if prev.is_none() {
            return Ok(false);
        }
        let next_ver = prev.map(|e| e.stat.version + 1).unwrap_or(1);
        let mut ws = self
            .workspaces
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        ws.entry(branch.to_string())
            .or_default()
            .files
            .insert(norm, OverlaySlot::Deleted { version: next_ver });
        Ok(true)
    }

    /// List all visible files under `dir_prefix` on `branch` in lexicographic order.
    pub fn list_files(
        &self,
        branch: &str,
        dir_prefix: &str,
    ) -> Result<Vec<AgentFileStat>, BranchError> {
        let prefix = if dir_prefix.trim().is_empty() || dir_prefix.trim() == "/" {
            "/".to_string()
        } else {
            normalize_vfs_path(dir_prefix)?
        };
        let chain = self.ancestor_chain(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;

        // Replay from root ("main") down to `branch`
        let mut merged: BTreeMap<String, AgentFileStat> = BTreeMap::new();
        for b in chain.iter().rev() {
            if let Some(state) = ws.get(b) {
                for (path, slot) in &state.files {
                    match slot {
                        OverlaySlot::Present(entry) => {
                            merged.insert(path.clone(), entry.stat.clone());
                        }
                        OverlaySlot::Deleted { .. } => {
                            merged.remove(path);
                        }
                    }
                }
            }
        }

        Ok(merged
            .into_iter()
            .filter(|(p, _)| {
                prefix == "/"
                    || *p == prefix
                    || p.strip_prefix(&prefix)
                        .is_some_and(|rest| rest.starts_with('/'))
            })
            .map(|(_, stat)| stat)
            .collect())
    }

    /// Store a JSON scratchpad value in the branch's CoW KV working memory.
    pub fn kv_put(
        &self,
        branch: &str,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), BranchError> {
        let est_bytes = value.to_string().len() + key.len();
        self.charge_branch_or_fail(branch, payload_energy_uj(est_bytes))?;
        let prev_ver = self
            .resolve_kv_entry(branch, key)?
            .map(|(_, v)| v)
            .unwrap_or(0);
        let mut ws = self
            .workspaces
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        ws.entry(branch.to_string())
            .or_default()
            .kv
            .insert(key.to_string(), OverlaySlot::Present((value, prev_ver + 1)));
        Ok(())
    }

    /// Retrieve a JSON scratchpad value from `branch` (with CoW inheritance).
    pub fn kv_get(
        &self,
        branch: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>, BranchError> {
        self.charge_branch_or_fail(branch, BASE_OP_ENERGY_UJ)?;
        Ok(self.resolve_kv_entry(branch, key)?.map(|(v, _)| v))
    }

    /// Append a hash-chained, tamper-evident tool-call record on `branch` and meter its energy.
    pub fn record_tool_call(
        &self,
        branch: &str,
        tool_name: &str,
        args_summary: &str,
        outcome: &str,
        energy_uj: u64,
    ) -> Result<ToolAuditRecord, BranchError> {
        self.charge_branch_or_fail(branch, energy_uj.max(1))?;

        let mut ws = self
            .workspaces
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let state = ws.entry(branch.to_string()).or_default();
        let seq = state.audit_log.len() as u64 + 1;
        let prev_hash = state
            .audit_log
            .last()
            .map(|r| r.entry_hash.clone())
            .unwrap_or_else(|| "0".repeat(32));
        let ts = now_millis();
        let seq_bytes = seq.to_le_bytes();
        let ts_bytes = ts.to_le_bytes();
        let e_bytes = energy_uj.to_le_bytes();
        let entry_hash = digest_hex_128(&[
            prev_hash.as_bytes(),
            branch.as_bytes(),
            &seq_bytes,
            &ts_bytes,
            tool_name.as_bytes(),
            args_summary.as_bytes(),
            outcome.as_bytes(),
            &e_bytes,
        ]);

        let rec = ToolAuditRecord {
            seq,
            branch: branch.to_string(),
            timestamp_ms: ts,
            tool_name: tool_name.to_string(),
            args_summary: args_summary.to_string(),
            outcome: outcome.to_string(),
            energy_uj,
            prev_hash,
            entry_hash,
        };
        state.audit_log.push(rec.clone());
        Ok(rec)
    }

    /// Verify the cryptographic/FNV hash chain integrity of all tool audit records on `branch`.
    pub fn verify_audit_chain(&self, branch: &str) -> Result<bool, BranchError> {
        let _ = self.branches.get_branch(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let Some(state) = ws.get(branch) else {
            return Ok(true);
        };
        let mut expected_prev = "0".repeat(32);
        for rec in &state.audit_log {
            if rec.prev_hash != expected_prev {
                return Ok(false);
            }
            let seq_bytes = rec.seq.to_le_bytes();
            let ts_bytes = rec.timestamp_ms.to_le_bytes();
            let e_bytes = rec.energy_uj.to_le_bytes();
            let recomputed = digest_hex_128(&[
                rec.prev_hash.as_bytes(),
                rec.branch.as_bytes(),
                &seq_bytes,
                &ts_bytes,
                rec.tool_name.as_bytes(),
                rec.args_summary.as_bytes(),
                rec.outcome.as_bytes(),
                &e_bytes,
            ]);
            if recomputed != rec.entry_hash {
                return Ok(false);
            }
            expected_prev = rec.entry_hash.clone();
        }
        Ok(true)
    }

    /// Retrieve the branch-local tool audit trail.
    pub fn audit_log(&self, branch: &str) -> Result<Vec<ToolAuditRecord>, BranchError> {
        let _ = self.branches.get_branch(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        Ok(ws
            .get(branch)
            .map(|s| s.audit_log.clone())
            .unwrap_or_default())
    }

    /// Compute a 3-way structural diff (files, KV keys, conflicts, energy) between `branch` and its parent.
    pub fn diff_workspace(&self, branch: &str) -> Result<WorkspaceDiff, BranchError> {
        let info = self.branches.get_branch(branch)?;
        let parent = info
            .parent
            .clone()
            .ok_or_else(|| BranchError::Storage("main branch has no parent to diff".to_string()))?;

        let parent_file_vers = self.effective_file_versions(&parent)?;
        let parent_kv_vers = self.effective_kv_versions(&parent)?;

        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let state = ws
            .get(branch)
            .ok_or_else(|| BranchError::NotFound(branch.to_string()))?;

        let mut added_files = Vec::new();
        let mut modified_files = Vec::new();
        let mut deleted_files = Vec::new();
        let mut conflicts = BTreeSet::new();

        for (path, slot) in &state.files {
            let base_v = state.base_file_versions.get(path).copied().unwrap_or(0);
            let cur_parent_v = parent_file_vers.get(path).copied().unwrap_or(0);
            if cur_parent_v != base_v {
                conflicts.insert(path.clone());
            }
            match slot {
                OverlaySlot::Present(_) => {
                    if base_v == 0 {
                        added_files.push(path.clone());
                    } else {
                        modified_files.push(path.clone());
                    }
                }
                OverlaySlot::Deleted { .. } => {
                    deleted_files.push(path.clone());
                }
            }
        }

        let mut mutated_kv_keys = Vec::new();
        for key in state.kv.keys() {
            let base_v = state.base_kv_versions.get(key).copied().unwrap_or(0);
            let cur_parent_v = parent_kv_vers.get(key).copied().unwrap_or(0);
            if cur_parent_v != base_v {
                conflicts.insert(format!("kv:{key}"));
            }
            mutated_kv_keys.push(key.clone());
        }

        let remaining = info
            .energy_budget_uj
            .map(|b| b.saturating_sub(info.energy_spent_uj));

        Ok(WorkspaceDiff {
            branch: branch.to_string(),
            parent,
            added_files,
            modified_files,
            deleted_files,
            mutated_kv_keys,
            conflicting_paths_or_keys: conflicts.into_iter().collect(),
            tool_calls_recorded: state.audit_log.len(),
            energy_spent_uj: info.energy_spent_uj,
            energy_remaining_uj: remaining,
        })
    }

    /// Merge `branch`'s CoW files and KV mutations atomically into its parent branch,
    /// failing with [`BranchError::MergeConflict`] if concurrent parent writes collided.
    pub fn merge_workspace(
        &self,
        branch: &str,
        delete_after: bool,
    ) -> Result<WorkspaceDiff, BranchError> {
        let diff = self.diff_workspace(branch)?;
        if !diff.conflicting_paths_or_keys.is_empty() {
            return Err(BranchError::MergeConflict(
                diff.conflicting_paths_or_keys.len(),
            ));
        }

        {
            let mut ws = self
                .workspaces
                .write()
                .map_err(|e| BranchError::Storage(e.to_string()))?;
            let child_files = ws.get(branch).map(|s| s.files.clone()).unwrap_or_default();
            let child_kv = ws.get(branch).map(|s| s.kv.clone()).unwrap_or_default();

            let parent_state = ws.entry(diff.parent.clone()).or_default();
            for (path, slot) in child_files {
                parent_state.files.insert(path, slot);
            }
            for (key, slot) in child_kv {
                parent_state.kv.insert(key, slot);
            }
            if delete_after {
                ws.remove(branch);
            }
        }

        let _ = self.branches.merge_branch(branch, delete_after)?;
        Ok(diff)
    }

    // --- Internal resolution helpers ---

    fn charge_branch_or_fail(&self, branch: &str, cost_uj: u64) -> Result<(), BranchError> {
        if !self.branches.check_energy_budget(branch, cost_uj)? {
            let info = self.branches.get_branch(branch)?;
            return Err(BranchError::EnergyBudgetExhausted {
                spent_uj: info.energy_spent_uj + cost_uj,
                budget_uj: info.energy_budget_uj.unwrap_or(0),
            });
        }
        self.branches.record_energy(branch, cost_uj)?;
        Ok(())
    }

    /// Returns `[branch, parent, grandparent, ..., "main"]`.
    fn ancestor_chain(&self, branch: &str) -> Result<Vec<String>, BranchError> {
        let mut chain = Vec::new();
        let mut cursor = Some(branch.to_string());
        let mut guard = 0usize;
        while let Some(curr) = cursor {
            guard += 1;
            if guard > 256 {
                return Err(BranchError::Storage(
                    "branch ancestry depth exceeded 256".to_string(),
                ));
            }
            let info = self.branches.get_branch(&curr)?;
            cursor = info.parent;
            chain.push(curr);
        }
        Ok(chain)
    }

    fn resolve_file_entry(
        &self,
        branch: &str,
        norm_path: &str,
    ) -> Result<Option<FileEntry>, BranchError> {
        let chain = self.ancestor_chain(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        for b in &chain {
            if let Some(state) = ws.get(b) {
                if let Some(slot) = state.files.get(norm_path) {
                    return Ok(match slot {
                        OverlaySlot::Present(entry) => Some(entry.clone()),
                        OverlaySlot::Deleted { .. } => None,
                    });
                }
            }
        }
        Ok(None)
    }

    fn resolve_kv_entry(
        &self,
        branch: &str,
        key: &str,
    ) -> Result<Option<(serde_json::Value, u64)>, BranchError> {
        let chain = self.ancestor_chain(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        for b in &chain {
            if let Some(state) = ws.get(b) {
                if let Some(slot) = state.kv.get(key) {
                    return Ok(match slot {
                        OverlaySlot::Present(pair) => Some(pair.clone()),
                        OverlaySlot::Deleted { .. } => None,
                    });
                }
            }
        }
        Ok(None)
    }

    fn effective_file_versions(&self, branch: &str) -> Result<HashMap<String, u64>, BranchError> {
        let chain = self.ancestor_chain(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let mut map = HashMap::new();
        for b in chain.iter().rev() {
            if let Some(state) = ws.get(b) {
                for (path, slot) in &state.files {
                    let v = match slot {
                        OverlaySlot::Present(e) => e.stat.version,
                        OverlaySlot::Deleted { version } => *version,
                    };
                    map.insert(path.clone(), v);
                }
            }
        }
        Ok(map)
    }

    fn effective_kv_versions(&self, branch: &str) -> Result<HashMap<String, u64>, BranchError> {
        let chain = self.ancestor_chain(branch)?;
        let ws = self
            .workspaces
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let mut map = HashMap::new();
        for b in chain.iter().rev() {
            if let Some(state) = ws.get(b) {
                for (key, slot) in &state.kv {
                    let v = match slot {
                        OverlaySlot::Present((_, ver)) => *ver,
                        OverlaySlot::Deleted { version } => *version,
                    };
                    map.insert(key.clone(), v);
                }
            }
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cow_fork_isolation_and_merge() {
        let fs = JouleAgentFs::standalone();
        fs.write_file("main", "/src/lib.rs", b"pub fn base() -> i32 { 1 }")
            .unwrap();
        fs.kv_put("main", "goal", serde_json::json!("ship sota"))
            .unwrap();

        // Spawn two competing subagents with microjoule ceilings
        fs.fork_workspace("agent-a", Some("main"), Some(500), None)
            .unwrap();
        fs.fork_workspace("agent-b", Some("main"), Some(500), None)
            .unwrap();

        // Both read inherited file without copying upfront
        assert_eq!(
            fs.read_file("agent-a", "/src/lib.rs").unwrap().as_deref(),
            Some(&b"pub fn base() -> i32 { 1 }"[..])
        );

        // Agent A edits lib.rs and adds a test file + tool audit trace
        fs.write_file("agent-a", "/src/lib.rs", b"pub fn base() -> i32 { 42 }")
            .unwrap();
        fs.write_file("agent-a", "/tests/a.rs", b"#[test] fn ok() {}")
            .unwrap();
        fs.record_tool_call("agent-a", "cargo_test", "{}", "pass", 25)
            .unwrap();
        assert!(fs.verify_audit_chain("agent-a").unwrap());

        // Main and Agent B remain completely untouched!
        assert_eq!(
            fs.read_file("main", "/src/lib.rs").unwrap().as_deref(),
            Some(&b"pub fn base() -> i32 { 1 }"[..])
        );
        assert!(fs.read_file("agent-b", "/tests/a.rs").unwrap().is_none());

        // Merge Agent A into main
        let diff = fs.merge_workspace("agent-a", true).unwrap();
        assert_eq!(diff.added_files, vec!["/tests/a.rs".to_string()]);
        assert_eq!(diff.modified_files, vec!["/src/lib.rs".to_string()]);
        assert_eq!(
            fs.read_file("main", "/src/lib.rs").unwrap().as_deref(),
            Some(&b"pub fn base() -> i32 { 42 }"[..])
        );

        // Agent B now conflicts if it also modifies /src/lib.rs based on stale fork!
        fs.write_file("agent-b", "/src/lib.rs", b"pub fn base() -> i32 { 99 }")
            .unwrap();
        let err = fs.merge_workspace("agent-b", false).unwrap_err();
        assert!(matches!(err, BranchError::MergeConflict(1)));
    }

    #[test]
    fn test_agent_workspace_hard_microjoule_cap() {
        let fs = JouleAgentFs::standalone();
        fs.fork_workspace("rogue-agent", Some("main"), Some(10), None)
            .unwrap();

        // Spend 8 uJ on a tool call -> succeeds
        fs.record_tool_call("rogue-agent", "db.query", "SELECT 1", "ok", 8)
            .unwrap();

        // Attempting a large 32 KiB write (~9 uJ) exceeds 10 uJ cap -> rejected before mutating!
        let big = vec![b'x'; 32 * 1024];
        let err = fs
            .write_file("rogue-agent", "/escape/dump.bin", &big)
            .unwrap_err();
        assert!(matches!(err, BranchError::EnergyBudgetExhausted { .. }));
        assert!(
            fs.stat_file("rogue-agent", "/escape/dump.bin")
                .unwrap()
                .is_none()
        );
    }
}
