//! Session-Isolated Copy-on-Write SQL Branching, 3-Way Conflict-Checked `MERGE`,
//! Agent Blast-Radius Guardrails, Row-Level `SELECT ... FOR UPDATE` Locking, and
//! DBOS-Style Durable Agentic Workflow Checkpoints.
//!
//! Bridges the "Agent Database vs Human Database" divide inside a single JouleDB engine:
//! 1. **Real Isolated CoW SQL Branches (`CREATE BRANCH`, `USE BRANCH`, `DIFF BRANCH`, `MERGE BRANCH`):**
//!    Agents fork an O(1) logical snapshot of parent SQL tables; writes mutate an isolated branch
//!    executor (`main` stays 100% untouched) with optional microjoule (`µJ`) caps and 3-way
//!    write-write conflict detection on `MERGE BRANCH`.
//! 2. **Agent Blast-Radius Guardrails:** Blocks destructive DDL (`DROP TABLE`, `TRUNCATE`) and
//!    unbounded `DELETE` / `UPDATE` (missing `WHERE`) when strict agent guardrails are active.
//! 3. **Durable Agentic Workflow Checkpointing (`WORKFLOW CHECKPOINT`, `WORKFLOW RESUME`):**
//!    Persists step outputs to `workflow_steps.jwal` so crashed multi-step agents resume
//!    deterministically without re-executing committed side effects.
//! 4. **Pessimistic Row Lock Manager (`SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`):**
//!    Coordinates row-level exclusive/shared locks across concurrent `pgwire`/HTTP transactions.

use crate::amorphic_adapter::{AmorphicTableStorage, ColumnDefInfo};
use crate::query::{
    QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse, SimpleQueryExecutor,
};
use joule_db_query::{RowData, TableStorage};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const WORKFLOW_WAL_FILENAME: &str = "workflow_steps.jwal";

#[derive(Debug, Clone)]
pub struct TableBaseSnapshot {
    pub columns: Vec<String>,
    pub col_defs: Vec<ColumnDefInfo>,
    pub primary_key_idx: Option<usize>,
    /// Raw fork-point rows captured without JSON serialization or SQL re-parsing (`O(1)` Arc share across child branches).
    pub raw_rows: Arc<Vec<RowData>>,
    /// Lazily computed `key -> canonical JSON row` map (populated only if `DIFF BRANCH` or `MERGE BRANCH` inspects a mutated table).
    pub rows_by_key: Arc<OnceLock<BTreeMap<String, Vec<serde_json::Value>>>>,
}

impl TableBaseSnapshot {
    pub fn keyed_rows(&self) -> &BTreeMap<String, Vec<serde_json::Value>> {
        self.rows_by_key.get_or_init(|| {
            let json_rows = raw_rows_to_json_rows(&self.raw_rows, &self.columns);
            index_rows_by_key(&json_rows, self.primary_key_idx)
        })
    }
}

pub struct SqlBranchEntry {
    pub name: String,
    pub parent: String,
    pub created_at_ms: u64,
    pub energy_budget_uj: Option<u64>,
    pub energy_spent_uj: Arc<AtomicU64>,
    pub guardrails_strict: bool,
    pub base_tables: HashMap<String, TableBaseSnapshot>,
    /// Tables that have been lazily materialized into `executor` on first SQL reference.
    pub materialized_tables: Mutex<HashSet<String>>,
    /// Lazily constructed isolated executor (`CREATE BRANCH` avoids WGPU/HDC init until first query).
    pub executor: OnceLock<Arc<SimpleQueryExecutor>>,
}

impl SqlBranchEntry {
    #[inline]
    pub fn executor(&self) -> &Arc<SimpleQueryExecutor> {
        self.executor
            .get_or_init(|| Arc::new(SimpleQueryExecutor::new()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableBranchDiff {
    pub table: String,
    pub is_new_table: bool,
    pub inserted: usize,
    pub updated: usize,
    pub deleted: usize,
    pub conflicts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlBranchDiffReport {
    pub branch: String,
    pub parent: String,
    pub energy_spent_uj: u64,
    pub energy_budget_uj: Option<u64>,
    pub tables: Vec<TableBranchDiff>,
    pub total_inserted: usize,
    pub total_updated: usize,
    pub total_deleted: usize,
    pub total_conflicts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlBranchMergeReport {
    pub branch: String,
    pub target: String,
    pub tables_merged: usize,
    pub rows_inserted: usize,
    pub rows_updated: usize,
    pub rows_deleted: usize,
    pub branch_deleted: bool,
}

/// Manages isolated Copy-on-Write SQL branches + per-session active branch & guardrail settings.
pub struct SqlBranchSandboxManager {
    branches: RwLock<HashMap<String, Arc<SqlBranchEntry>>>,
    session_branches: RwLock<HashMap<String, String>>,
    session_guardrails: RwLock<HashMap<String, SessionGuardrailPolicy>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionGuardrailPolicy {
    pub strict: bool,
    pub allow_destructive_ddl: bool,
    pub require_where_on_write: bool,
    pub max_joules_uj: Option<u64>,
    pub spent_joules_uj: u64,
}

impl Default for SqlBranchSandboxManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SqlBranchSandboxManager {
    pub fn new() -> Self {
        Self {
            branches: RwLock::new(HashMap::new()),
            session_branches: RwLock::new(HashMap::new()),
            session_guardrails: RwLock::new(HashMap::new()),
        }
    }

    pub fn set_session_branch(&self, session_key: &str, branch: Option<String>) {
        if let Ok(mut map) = self.session_branches.write() {
            match branch {
                Some(b) if !b.eq_ignore_ascii_case("main") && !b.is_empty() => {
                    map.insert(session_key.to_string(), b);
                }
                _ => {
                    map.remove(session_key);
                }
            }
        }
    }

    pub fn get_session_branch(&self, session_key: &str) -> Option<String> {
        self.session_branches
            .read()
            .ok()
            .and_then(|m| m.get(session_key).cloned())
    }

    pub fn update_session_guardrail<F: FnOnce(&mut SessionGuardrailPolicy)>(
        &self,
        session_key: &str,
        f: F,
    ) {
        if let Ok(mut map) = self.session_guardrails.write() {
            let entry =
                map.entry(session_key.to_string())
                    .or_insert_with(|| SessionGuardrailPolicy {
                        strict: false,
                        allow_destructive_ddl: true,
                        require_where_on_write: false,
                        max_joules_uj: None,
                        spent_joules_uj: 0,
                    });
            f(entry);
        }
    }

    pub fn check_session_guardrails(
        &self,
        session_key: Option<&str>,
        branch_name: Option<&str>,
        sql: &str,
    ) -> Result<(), QueryErrorResponse> {
        let mut block_ddl = false;
        let mut require_where = false;

        if let Some(sk) = session_key
            && let Ok(map) = self.session_guardrails.read()
            && let Some(pol) = map.get(sk)
        {
            if pol.strict || !pol.allow_destructive_ddl {
                block_ddl = true;
            }
            if pol.strict || pol.require_where_on_write {
                require_where = true;
            }
            if let Some(max_uj) = pol.max_joules_uj
                && pol.spent_joules_uj >= max_uj
            {
                return Err(sqlstate_err(
                    "53200",
                    format!(
                        "agent session energy budget exhausted ({} >= {} µJ)",
                        pol.spent_joules_uj, max_uj
                    ),
                ));
            }
        }

        if let Some(bn) = branch_name
            && !bn.eq_ignore_ascii_case("main")
            && let Ok(bmap) = self.branches.read()
            && let Some(b) = bmap.get(bn)
        {
            if b.guardrails_strict {
                block_ddl = true;
                require_where = true;
            }
            let spent = b.energy_spent_uj.load(Ordering::Relaxed);
            if let Some(budget) = b.energy_budget_uj
                && spent >= budget
            {
                return Err(sqlstate_err(
                    "53200",
                    format!(
                        "branch '{}' energy budget exhausted ({} >= {} µJ)",
                        b.name, spent, budget
                    ),
                ));
            }
        }

        if block_ddl || require_where {
            validate_sql_against_guardrails(sql, block_ddl, require_where)?;
        }
        Ok(())
    }

    pub fn record_energy_uj(
        &self,
        session_key: Option<&str>,
        branch_name: Option<&str>,
        joules: Option<f64>,
    ) {
        let uj = joules
            .map(|j| (j.max(0.0) * 1_000_000.0).round() as u64)
            .unwrap_or(10)
            .max(1);
        if let Some(sk) = session_key
            && let Ok(mut map) = self.session_guardrails.write()
            && let Some(pol) = map.get_mut(sk)
        {
            pol.spent_joules_uj = pol.spent_joules_uj.saturating_add(uj);
        }
        if let Some(bn) = branch_name
            && !bn.eq_ignore_ascii_case("main")
            && let Ok(bmap) = self.branches.read()
            && let Some(b) = bmap.get(bn)
        {
            b.energy_spent_uj.fetch_add(uj, Ordering::Relaxed);
        }
    }

    pub fn get_branch_executor(&self, name: &str) -> Option<Arc<SimpleQueryExecutor>> {
        let entry = self
            .branches
            .read()
            .ok()
            .and_then(|m| m.get(name).cloned())?;
        let _ = materialize_all_branch_tables(&entry);
        Some(entry.executor().clone())
    }

    /// Return the isolated branch executor after lazily materializing ONLY the parent tables
    /// referenced by `sql` (keep `CREATE BRANCH` `< 1 ms` and skip untouched tables on `DIFF`/`MERGE`).
    pub fn get_branch_executor_for_sql(
        &self,
        name: &str,
        sql: &str,
    ) -> Result<Arc<SimpleQueryExecutor>, QueryErrorResponse> {
        let entry = self
            .branches
            .read()
            .ok()
            .and_then(|m| m.get(name).cloned())
            .ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!("branch '{name}' not found"))
            })?;
        materialize_referenced_branch_tables(&entry, sql)?;
        Ok(entry.executor().clone())
    }

    pub fn has_branch(&self, name: &str) -> bool {
        if name.eq_ignore_ascii_case("main") {
            return true;
        }
        self.branches
            .read()
            .map(|m| m.contains_key(name))
            .unwrap_or(false)
    }

    /// Fork an isolated SQL branch from `main` (or an existing parent branch).
    pub fn create_branch(
        &self,
        main_amorphic: &AmorphicTableStorage,
        name: &str,
        parent: &str,
        energy_budget_uj: Option<u64>,
        guardrails_strict: bool,
    ) -> Result<Arc<SqlBranchEntry>, QueryErrorResponse> {
        let clean = name.trim();
        if clean.is_empty() || clean.eq_ignore_ascii_case("main") {
            return Err(QueryErrorResponse::syntax_error(
                "invalid branch name (cannot be empty or 'main')",
                1,
                1,
            ));
        }

        let source_amorphic: &AmorphicTableStorage = if parent.eq_ignore_ascii_case("main") {
            main_amorphic
        } else {
            let bmap = self
                .branches
                .read()
                .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
            let p = bmap.get(parent).ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!("parent branch '{parent}' not found"))
            })?;
            let entry = build_branch_entry_from_parent_branch(
                p,
                clean,
                parent,
                energy_budget_uj,
                guardrails_strict,
            )?;
            drop(bmap);
            let mut wmap = self
                .branches
                .write()
                .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
            if wmap.contains_key(clean) {
                return Err(sqlstate_err(
                    "42710",
                    format!("branch '{clean}' already exists"),
                ));
            }
            let arc = Arc::new(entry);
            wmap.insert(clean.to_string(), arc.clone());
            return Ok(arc);
        };

        let entry = build_branch_entry_from_storage(
            source_amorphic,
            clean,
            parent,
            energy_budget_uj,
            guardrails_strict,
        )?;
        let mut wmap = self
            .branches
            .write()
            .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
        if wmap.contains_key(clean) {
            return Err(sqlstate_err(
                "42710",
                format!("branch '{clean}' already exists"),
            ));
        }
        let arc = Arc::new(entry);
        wmap.insert(clean.to_string(), arc.clone());
        Ok(arc)
    }

    pub fn drop_branch(&self, name: &str, if_exists: bool) -> Result<bool, QueryErrorResponse> {
        if name.eq_ignore_ascii_case("main") {
            return Err(sqlstate_err(
                "42501",
                "cannot drop the 'main' branch".into(),
            ));
        }
        let mut wmap = self
            .branches
            .write()
            .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
        if wmap.remove(name).is_some() {
            Ok(true)
        } else if if_exists {
            Ok(false)
        } else {
            Err(QueryErrorResponse::execution_error(&format!(
                "branch '{name}' not found"
            )))
        }
    }

    pub fn diff_branch(
        &self,
        main_amorphic: &AmorphicTableStorage,
        name: &str,
    ) -> Result<SqlBranchDiffReport, QueryErrorResponse> {
        let entry = {
            let bmap = self
                .branches
                .read()
                .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
            bmap.get(name).cloned().ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!("branch '{name}' not found"))
            })?
        };

        let parent_entry = if entry.parent.eq_ignore_ascii_case("main") {
            None
        } else {
            let bmap = self
                .branches
                .read()
                .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
            Some(bmap.get(&entry.parent).cloned().ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!(
                    "parent branch '{}' not found",
                    entry.parent
                ))
            })?)
        };
        let parent_storage: &AmorphicTableStorage = match &parent_entry {
            Some(pe) => {
                materialize_all_branch_tables(pe)?;
                pe.executor().amorphic()
            }
            None => main_amorphic,
        };

        compute_three_way_diff(&entry, parent_storage)
    }

    /// Perform 3-way conflict-checked merge of `name` into its parent (`main` or parent branch).
    pub fn merge_branch(
        &self,
        main_executor: &SimpleQueryExecutor,
        name: &str,
        delete_after: bool,
    ) -> Result<SqlBranchMergeReport, QueryErrorResponse> {
        let entry = {
            let bmap = self
                .branches
                .read()
                .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
            bmap.get(name).cloned().ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!("branch '{name}' not found"))
            })?
        };

        let target_exec: &SimpleQueryExecutor = if entry.parent.eq_ignore_ascii_case("main") {
            main_executor
        } else {
            let bmap = self
                .branches
                .read()
                .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
            let pe = bmap.get(&entry.parent).cloned().ok_or_else(|| {
                QueryErrorResponse::execution_error(&format!(
                    "parent branch '{}' not found",
                    entry.parent
                ))
            })?;
            drop(bmap);
            materialize_all_branch_tables(&pe)?;
            let report = apply_three_way_merge(&entry, pe.executor(), delete_after)?;
            if delete_after {
                let _ = self.drop_branch(name, true);
            }
            return Ok(report);
        };

        let report = apply_three_way_merge(&entry, target_exec, delete_after)?;
        if delete_after {
            let _ = self.drop_branch(name, true);
        }
        Ok(report)
    }

    pub fn list_branches_rows(&self) -> (Vec<String>, Vec<Vec<serde_json::Value>>) {
        let cols = vec![
            "branch".into(),
            "parent".into(),
            "energy_spent_uj".into(),
            "energy_budget_uj".into(),
            "guardrails_strict".into(),
            "created_at_ms".into(),
        ];
        let mut rows = vec![vec![
            serde_json::json!("main"),
            serde_json::Value::Null,
            serde_json::json!(0i64),
            serde_json::Value::Null,
            serde_json::json!(false),
            serde_json::json!(0i64),
        ]];
        if let Ok(bmap) = self.branches.read() {
            let mut list: Vec<_> = bmap.values().cloned().collect();
            list.sort_by(|a, b| a.name.cmp(&b.name));
            for b in list {
                rows.push(vec![
                    serde_json::json!(b.name),
                    serde_json::json!(b.parent),
                    serde_json::json!(b.energy_spent_uj.load(Ordering::Relaxed) as i64),
                    b.energy_budget_uj
                        .map(|v| serde_json::json!(v as i64))
                        .unwrap_or(serde_json::Value::Null),
                    serde_json::json!(b.guardrails_strict),
                    serde_json::json!(b.created_at_ms as i64),
                ]);
            }
        }
        (cols, rows)
    }
}

// ============================================================================
// DBOS-Style Durable Agentic Workflow Checkpoint Engine
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStepRecord {
    pub workflow_id: String,
    pub step_seq: u64,
    pub step_id: String,
    pub status: String,
    pub input_hash: Option<String>,
    pub output_json: serde_json::Value,
    pub energy_uj: u64,
    pub attempts: u32,
    pub updated_at_ms: u64,
}

struct WorkflowStoreInner {
    by_workflow: BTreeMap<String, Vec<WorkflowStepRecord>>,
    wal_file: Option<File>,
}

pub struct DurableWorkflowCheckpointStore {
    inner: Mutex<WorkflowStoreInner>,
}

impl Default for DurableWorkflowCheckpointStore {
    fn default() -> Self {
        Self::new_in_memory()
    }
}

impl DurableWorkflowCheckpointStore {
    pub fn new_in_memory() -> Self {
        Self {
            inner: Mutex::new(WorkflowStoreInner {
                by_workflow: BTreeMap::new(),
                wal_file: None,
            }),
        }
    }

    pub fn open(dir: impl AsRef<Path>) -> Result<Self, String> {
        let p = dir.as_ref();
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
        let wal_path: PathBuf = p.join(WORKFLOW_WAL_FILENAME);
        let mut by_workflow: BTreeMap<String, Vec<WorkflowStepRecord>> = BTreeMap::new();
        if wal_path.exists() {
            let f = File::open(&wal_path).map_err(|e| e.to_string())?;
            for line_res in BufReader::new(f).lines() {
                let line = line_res.map_err(|e| e.to_string())?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(rec) = serde_json::from_str::<WorkflowStepRecord>(line.trim()) {
                    upsert_step_in_map(&mut by_workflow, rec);
                }
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            inner: Mutex::new(WorkflowStoreInner {
                by_workflow,
                wal_file: Some(file),
            }),
        })
    }

    /// Record or idempotently replay a workflow step checkpoint.
    /// If `(workflow_id, step_id)` is already `"completed"`, returns `(existing, true)` (`memoized = true`)!
    pub fn checkpoint_step(
        &self,
        workflow_id: &str,
        step_id: &str,
        status: &str,
        input_hash: Option<String>,
        output_json: serde_json::Value,
        energy_uj: u64,
    ) -> Result<(WorkflowStepRecord, bool), String> {
        let mut guard = self.inner.lock().map_err(|e| e.to_string())?;
        let steps = guard
            .by_workflow
            .entry(workflow_id.to_string())
            .or_default();

        if let Some(existing) = steps.iter().find(|s| s.step_id == step_id)
            && existing.status.eq_ignore_ascii_case("completed")
        {
            if let (Some(eh), Some(nh)) = (&existing.input_hash, &input_hash)
                && eh != nh
            {
                return Err(format!(
                    "non-deterministic workflow replay detected on workflow '{workflow_id}' step '{step_id}': input_hash mismatch"
                ));
            }
            return Ok((existing.clone(), true));
        }

        let next_seq = steps.len() as u64 + 1;
        let prev_attempts = steps
            .iter()
            .find(|s| s.step_id == step_id)
            .map(|s| s.attempts)
            .unwrap_or(0);
        let rec = WorkflowStepRecord {
            workflow_id: workflow_id.to_string(),
            step_seq: steps
                .iter()
                .find(|s| s.step_id == step_id)
                .map(|s| s.step_seq)
                .unwrap_or(next_seq),
            step_id: step_id.to_string(),
            status: status.to_ascii_lowercase(),
            input_hash,
            output_json,
            energy_uj,
            attempts: prev_attempts + 1,
            updated_at_ms: now_ms(),
        };

        if let Some(ref mut f) = guard.wal_file {
            let mut bytes = serde_json::to_vec(&rec).map_err(|e| e.to_string())?;
            bytes.push(b'\n');
            f.write_all(&bytes).map_err(|e| e.to_string())?;
            f.sync_data().map_err(|e| e.to_string())?;
        }
        upsert_step_in_map(&mut guard.by_workflow, rec.clone());
        Ok((rec, false))
    }

    pub fn resume_workflow(&self, workflow_id: &str) -> Vec<WorkflowStepRecord> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.by_workflow.get(workflow_id).cloned())
            .unwrap_or_default()
    }

    pub fn all_steps_virtual_table(&self) -> (Vec<String>, Vec<Vec<serde_json::Value>>) {
        let cols = vec![
            "workflow_id".into(),
            "step_seq".into(),
            "step_id".into(),
            "status".into(),
            "input_hash".into(),
            "output_json".into(),
            "energy_uj".into(),
            "attempts".into(),
            "updated_at_ms".into(),
        ];
        let mut rows = Vec::new();
        if let Ok(guard) = self.inner.lock() {
            for steps in guard.by_workflow.values() {
                for s in steps {
                    rows.push(vec![
                        serde_json::json!(s.workflow_id),
                        serde_json::json!(s.step_seq as i64),
                        serde_json::json!(s.step_id),
                        serde_json::json!(s.status),
                        serde_json::json!(s.input_hash),
                        serde_json::json!(s.output_json.to_string()),
                        serde_json::json!(s.energy_uj as i64),
                        serde_json::json!(s.attempts as i64),
                        serde_json::json!(s.updated_at_ms as i64),
                    ]);
                }
            }
        }
        (cols, rows)
    }
}

// ============================================================================
// Pessimistic Row Lock Manager (`SELECT ... FOR UPDATE [NOWAIT | SKIP LOCKED]`)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLockWaitPolicy {
    Wait,
    NoWait,
    SkipLocked,
}

#[derive(Debug, Clone)]
struct HeldRowLock {
    owner_session: String,
    exclusive: bool,
}

#[derive(Default)]
pub struct RowLockManager {
    /// `(table_lower, row_key) -> Vec<HeldRowLock>`
    locks: Mutex<HashMap<(String, String), Vec<HeldRowLock>>>,
}

impl RowLockManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to acquire a row lock for `session_id`.
    /// Returns:
    /// - `Ok(true)` if lock acquired (or no active transaction session).
    /// - `Ok(false)` if locked by another session and `policy == SkipLocked`.
    /// - `Err(QueryErrorResponse)` (`SQLSTATE 55P03` `lock_not_available`) if blocked!
    pub fn acquire_row(
        &self,
        session_id: Option<&str>,
        table: &str,
        row_key: &str,
        exclusive: bool,
        policy: RowLockWaitPolicy,
    ) -> Result<bool, QueryErrorResponse> {
        let Some(sid) = session_id else {
            // Outside an explicit BEGIN block, statement-level snapshot completes immediately.
            return Ok(true);
        };
        let key = (table.to_ascii_lowercase(), row_key.to_string());
        let deadline = Instant::now() + std::time::Duration::from_millis(250);

        loop {
            if let Ok(mut map) = self.locks.lock() {
                let holders = map.entry(key.clone()).or_default();
                let conflict = holders
                    .iter()
                    .any(|h| h.owner_session != sid && (exclusive || h.exclusive));
                if !conflict {
                    if let Some(mine) = holders.iter_mut().find(|h| h.owner_session == sid) {
                        mine.exclusive |= exclusive;
                    } else {
                        holders.push(HeldRowLock {
                            owner_session: sid.to_string(),
                            exclusive,
                        });
                    }
                    return Ok(true);
                }
            }

            match policy {
                RowLockWaitPolicy::SkipLocked => return Ok(false),
                RowLockWaitPolicy::NoWait => {
                    return Err(sqlstate_err(
                        "55P03",
                        format!(
                            "could not obtain lock on row '{}' in relation \"{}\"",
                            row_key, table
                        ),
                    ));
                }
                RowLockWaitPolicy::Wait => {
                    if Instant::now() >= deadline {
                        return Err(sqlstate_err(
                            "55P03",
                            format!(
                                "lock timeout waiting for row '{}' in relation \"{}\"",
                                row_key, table
                            ),
                        ));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        }
    }

    /// Release all row locks owned by `session_id` (called on `COMMIT` / `ROLLBACK`).
    pub fn release_session(&self, session_id: &str) {
        if let Ok(mut map) = self.locks.lock() {
            map.retain(|_, holders| {
                holders.retain(|h| h.owner_session != session_id);
                !holders.is_empty()
            });
        }
    }
}

// ============================================================================
// SQL Command Dispatchers for Branching & Durable Workflows
// ============================================================================

pub fn is_branch_or_workflow_sql(sql: &str) -> bool {
    let t = sql.trim_start();
    let up = t.get(..32.min(t.len())).unwrap_or(t).to_ascii_uppercase();
    up.starts_with("CREATE BRANCH ")
        || up.starts_with("USE BRANCH ")
        || up == "SHOW BRANCHES"
        || up.starts_with("SHOW BRANCHES;")
        || up.starts_with("DIFF BRANCH ")
        || up.starts_with("MERGE BRANCH ")
        || up.starts_with("DROP BRANCH ")
        || up.starts_with("WORKFLOW CHECKPOINT ")
        || up.starts_with("WORKFLOW RESUME ")
}

pub fn execute_branch_or_workflow_sql(
    main_executor: &SimpleQueryExecutor,
    branch_mgr: &SqlBranchSandboxManager,
    wf_store: &DurableWorkflowCheckpointStore,
    sql: &str,
    session_key: Option<&str>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let trimmed = sql.trim().trim_end_matches(';').trim();
    let toks = tokenize_simple(trimmed);
    if toks.is_empty() {
        return Err(QueryErrorResponse::syntax_error("empty command", 1, 1));
    }

    let w0 = toks[0].to_ascii_uppercase();
    let w1 = toks
        .get(1)
        .map(|s| s.to_ascii_uppercase())
        .unwrap_or_default();

    match (w0.as_str(), w1.as_str()) {
        ("SHOW", "BRANCHES") => {
            let (columns, rows) = branch_mgr.list_branches_rows();
            Ok(simple_rows_response(
                columns,
                rows,
                None,
                start,
                "cow-branch-list",
            ))
        }
        ("CREATE", "BRANCH") => {
            // CREATE BRANCH <name> [FROM <parent>] [ENERGY BUDGET <u64>] [GUARDRAILS STRICT|OFF]
            let name = toks.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("expected branch name after CREATE BRANCH", 1, 1)
            })?;
            let mut parent = "main".to_string();
            let mut budget_uj = None;
            let mut strict = false;
            let mut i = 3usize;
            while i < toks.len() {
                match toks[i].to_ascii_uppercase().as_str() {
                    "FROM" => {
                        i += 1;
                        parent = toks.get(i).cloned().unwrap_or_else(|| "main".into());
                    }
                    "ENERGY" => {
                        if toks
                            .get(i + 1)
                            .is_some_and(|t| t.eq_ignore_ascii_case("BUDGET"))
                        {
                            i += 2;
                        } else {
                            i += 1;
                        }
                        budget_uj = toks.get(i).and_then(|s| s.replace('_', "").parse().ok());
                    }
                    "BUDGET" => {
                        i += 1;
                        budget_uj = toks.get(i).and_then(|s| s.replace('_', "").parse().ok());
                    }
                    "GUARDRAILS" => {
                        i += 1;
                        strict = toks.get(i).is_some_and(|v| {
                            v.eq_ignore_ascii_case("STRICT") || v.eq_ignore_ascii_case("ON")
                        });
                    }
                    _ => {}
                }
                i += 1;
            }
            let entry = branch_mgr.create_branch(
                main_executor.amorphic(),
                &name,
                &parent,
                budget_uj,
                strict,
            )?;
            Ok(simple_rows_response(
                vec![
                    "branch".into(),
                    "parent".into(),
                    "energy_budget_uj".into(),
                    "guardrails_strict".into(),
                ],
                vec![vec![
                    serde_json::json!(entry.name),
                    serde_json::json!(entry.parent),
                    entry
                        .energy_budget_uj
                        .map(|u| serde_json::json!(u as i64))
                        .unwrap_or(serde_json::Value::Null),
                    serde_json::json!(entry.guardrails_strict),
                ]],
                Some(1),
                start,
                "cow-branch-fork",
            ))
        }
        ("USE", "BRANCH") => {
            let name = toks.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("expected branch name after USE BRANCH", 1, 1)
            })?;
            if !branch_mgr.has_branch(&name) {
                return Err(QueryErrorResponse::execution_error(&format!(
                    "branch '{name}' does not exist"
                )));
            }
            if let Some(sk) = session_key {
                branch_mgr.set_session_branch(sk, Some(name.clone()));
            }
            Ok(simple_rows_response(
                vec!["active_branch".into()],
                vec![vec![serde_json::json!(name)]],
                None,
                start,
                "cow-branch-switch",
            ))
        }
        ("DIFF", "BRANCH") => {
            let name = toks.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("expected branch name after DIFF BRANCH", 1, 1)
            })?;
            let rep = branch_mgr.diff_branch(main_executor.amorphic(), &name)?;
            let rows = rep
                .tables
                .into_iter()
                .map(|t| {
                    vec![
                        serde_json::json!(t.table),
                        serde_json::json!(t.is_new_table),
                        serde_json::json!(t.inserted as i64),
                        serde_json::json!(t.updated as i64),
                        serde_json::json!(t.deleted as i64),
                        serde_json::json!(t.conflicts.len() as i64),
                        serde_json::json!(t.conflicts.join(",")),
                    ]
                })
                .collect();
            Ok(simple_rows_response(
                vec![
                    "table_name".into(),
                    "is_new_table".into(),
                    "inserted".into(),
                    "updated".into(),
                    "deleted".into(),
                    "conflicts".into(),
                    "conflict_keys".into(),
                ],
                rows,
                None,
                start,
                "cow-branch-3way-diff",
            ))
        }
        ("MERGE", "BRANCH") => {
            let name = toks.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("expected branch name after MERGE BRANCH", 1, 1)
            })?;
            let delete_after = toks
                .iter()
                .skip(3)
                .any(|t| t.eq_ignore_ascii_case("DELETE"));
            let rep = branch_mgr.merge_branch(main_executor, &name, delete_after)?;
            Ok(simple_rows_response(
                vec![
                    "branch".into(),
                    "target".into(),
                    "tables_merged".into(),
                    "rows_inserted".into(),
                    "rows_updated".into(),
                    "rows_deleted".into(),
                    "branch_deleted".into(),
                ],
                vec![vec![
                    serde_json::json!(rep.branch),
                    serde_json::json!(rep.target),
                    serde_json::json!(rep.tables_merged as i64),
                    serde_json::json!(rep.rows_inserted as i64),
                    serde_json::json!(rep.rows_updated as i64),
                    serde_json::json!(rep.rows_deleted as i64),
                    serde_json::json!(rep.branch_deleted),
                ]],
                Some(rep.rows_inserted + rep.rows_updated + rep.rows_deleted),
                start,
                "cow-branch-3way-merge",
            ))
        }
        ("DROP", "BRANCH") => {
            let (if_exists, name_idx) = if toks.get(2).is_some_and(|t| t.eq_ignore_ascii_case("IF"))
                && toks
                    .get(3)
                    .is_some_and(|t| t.eq_ignore_ascii_case("EXISTS"))
            {
                (true, 4)
            } else {
                (false, 2)
            };
            let name = toks.get(name_idx).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("expected branch name in DROP BRANCH", 1, 1)
            })?;
            let dropped = branch_mgr.drop_branch(&name, if_exists)?;
            Ok(simple_rows_response(
                vec!["dropped".into(), "branch".into()],
                vec![vec![serde_json::json!(dropped), serde_json::json!(name)]],
                Some(usize::from(dropped)),
                start,
                "cow-branch-drop",
            ))
        }
        ("WORKFLOW", "CHECKPOINT") => {
            // WORKFLOW CHECKPOINT '<wf_id>' STEP '<step_id>' [STATUS '<st>'] [OUTPUT '<json>'] [INPUT_HASH '<h>'] [ENERGY_UJ <u64>]
            let wf_id = toks
                .get(2)
                .cloned()
                .ok_or_else(|| QueryErrorResponse::syntax_error("missing workflow_id", 1, 1))?;
            let mut step_id = String::new();
            let mut status = "completed".to_string();
            let mut output_json = serde_json::json!({});
            let mut input_hash = None;
            let mut energy_uj = 0u64;
            let mut i = 3usize;
            while i < toks.len() {
                match toks[i].to_ascii_uppercase().as_str() {
                    "STEP" => {
                        i += 1;
                        step_id = toks.get(i).cloned().unwrap_or_default();
                    }
                    "STATUS" => {
                        i += 1;
                        status = toks.get(i).cloned().unwrap_or_else(|| "completed".into());
                    }
                    "OUTPUT" => {
                        i += 1;
                        if let Some(raw) = toks.get(i) {
                            output_json = serde_json::from_str(raw)
                                .unwrap_or_else(|_| serde_json::Value::String(raw.clone()));
                        }
                    }
                    "INPUT_HASH" => {
                        i += 1;
                        input_hash = toks.get(i).cloned();
                    }
                    "ENERGY_UJ" => {
                        i += 1;
                        energy_uj = toks.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
                    }
                    _ => {}
                }
                i += 1;
            }
            if step_id.is_empty() {
                return Err(QueryErrorResponse::syntax_error(
                    "missing STEP '<step_id>' in WORKFLOW CHECKPOINT",
                    1,
                    1,
                ));
            }
            let (rec, memoized) = wf_store
                .checkpoint_step(
                    &wf_id,
                    &step_id,
                    &status,
                    input_hash,
                    output_json,
                    energy_uj,
                )
                .map_err(|e| sqlstate_err("23505", e))?;
            Ok(simple_rows_response(
                vec![
                    "workflow_id".into(),
                    "step_seq".into(),
                    "step_id".into(),
                    "status".into(),
                    "memoized_replay".into(),
                    "output_json".into(),
                    "attempts".into(),
                ],
                vec![vec![
                    serde_json::json!(rec.workflow_id),
                    serde_json::json!(rec.step_seq as i64),
                    serde_json::json!(rec.step_id),
                    serde_json::json!(rec.status),
                    serde_json::json!(memoized),
                    serde_json::json!(rec.output_json.to_string()),
                    serde_json::json!(rec.attempts as i64),
                ]],
                Some(usize::from(!memoized)),
                start,
                "durable-workflow-checkpoint",
            ))
        }
        ("WORKFLOW", "RESUME") => {
            let wf_id = toks.get(2).cloned().ok_or_else(|| {
                QueryErrorResponse::syntax_error("missing workflow_id after WORKFLOW RESUME", 1, 1)
            })?;
            let steps = wf_store.resume_workflow(&wf_id);
            let rows = steps
                .into_iter()
                .map(|s| {
                    vec![
                        serde_json::json!(s.step_seq as i64),
                        serde_json::json!(s.step_id),
                        serde_json::json!(s.status),
                        serde_json::json!(s.output_json.to_string()),
                        serde_json::json!(s.attempts as i64),
                        serde_json::json!(s.energy_uj as i64),
                    ]
                })
                .collect();
            Ok(simple_rows_response(
                vec![
                    "step_seq".into(),
                    "step_id".into(),
                    "status".into(),
                    "output_json".into(),
                    "attempts".into(),
                    "energy_uj".into(),
                ],
                rows,
                None,
                start,
                "durable-workflow-resume",
            ))
        }
        _ => Err(QueryErrorResponse::syntax_error(
            "unrecognized branch/workflow statement",
            1,
            1,
        )),
    }
}

// ============================================================================
// Internal Helpers (Lazy Copy-on-Write Table Materialization)
// ============================================================================

fn build_branch_entry_from_storage(
    source_amorphic: &AmorphicTableStorage,
    name: &str,
    parent: &str,
    energy_budget_uj: Option<u64>,
    guardrails_strict: bool,
) -> Result<SqlBranchEntry, QueryErrorResponse> {
    let mut base_tables = HashMap::new();

    for table in source_amorphic.list_tables() {
        let col_defs = source_amorphic.get_column_defs(&table).unwrap_or_default();
        let columns = source_amorphic.columns(&table).unwrap_or_default();
        let pk_idx = detect_primary_key_idx(&columns, &col_defs);
        let raw_rows: Vec<RowData> = source_amorphic
            .scan_with_record_ids(&table)
            .unwrap_or_default()
            .into_iter()
            .map(|(_, row)| row)
            .collect();

        base_tables.insert(
            table,
            TableBaseSnapshot {
                columns,
                col_defs,
                primary_key_idx: pk_idx,
                raw_rows: Arc::new(raw_rows),
                rows_by_key: Arc::new(OnceLock::new()),
            },
        );
    }

    Ok(SqlBranchEntry {
        name: name.to_string(),
        parent: parent.to_string(),
        created_at_ms: now_ms(),
        energy_budget_uj,
        energy_spent_uj: Arc::new(AtomicU64::new(0)),
        guardrails_strict,
        base_tables,
        materialized_tables: Mutex::new(HashSet::new()),
        executor: OnceLock::new(),
    })
}

fn build_branch_entry_from_parent_branch(
    parent_entry: &SqlBranchEntry,
    name: &str,
    parent: &str,
    energy_budget_uj: Option<u64>,
    guardrails_strict: bool,
) -> Result<SqlBranchEntry, QueryErrorResponse> {
    let mut base_tables = HashMap::new();
    let parent_mat: HashSet<String> = parent_entry
        .materialized_tables
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();

    for (table, pbase) in &parent_entry.base_tables {
        if !parent_mat.contains(table) {
            base_tables.insert(table.clone(), pbase.clone());
        }
    }

    if let Some(parent_exec) = parent_entry.executor.get() {
        let parent_storage = parent_exec.amorphic();
        for table in parent_storage.list_tables() {
            let col_defs = parent_storage.get_column_defs(&table).unwrap_or_default();
            let columns = parent_storage.columns(&table).unwrap_or_default();
            let pk_idx = detect_primary_key_idx(&columns, &col_defs);
            let raw_rows: Vec<RowData> = parent_storage
                .scan_with_record_ids(&table)
                .unwrap_or_default()
                .into_iter()
                .map(|(_, row)| row)
                .collect();
            base_tables.insert(
                table,
                TableBaseSnapshot {
                    columns,
                    col_defs,
                    primary_key_idx: pk_idx,
                    raw_rows: Arc::new(raw_rows),
                    rows_by_key: Arc::new(OnceLock::new()),
                },
            );
        }
    }

    Ok(SqlBranchEntry {
        name: name.to_string(),
        parent: parent.to_string(),
        created_at_ms: now_ms(),
        energy_budget_uj,
        energy_spent_uj: Arc::new(AtomicU64::new(0)),
        guardrails_strict,
        base_tables,
        materialized_tables: Mutex::new(HashSet::new()),
        executor: OnceLock::new(),
    })
}

fn materialize_one_branch_table(
    entry: &SqlBranchEntry,
    table: &str,
    base: &TableBaseSnapshot,
) -> Result<(), QueryErrorResponse> {
    let exec = entry.executor();
    if !base.col_defs.is_empty() {
        let ddl = render_create_table_sql(table, &base.col_defs);
        let _ = exec.execute(&QueryRequest {
            sql: ddl,
            ..Default::default()
        })?;
    } else if !base.columns.is_empty() {
        let cols_ddl: Vec<String> = base
            .columns
            .iter()
            .map(|c| format!("\"{c}\" TEXT"))
            .collect();
        let ddl = format!("CREATE TABLE \"{table}\" ({})", cols_ddl.join(", "));
        let _ = exec.execute(&QueryRequest {
            sql: ddl,
            ..Default::default()
        })?;
    }

    if !base.raw_rows.is_empty() && !base.columns.is_empty() {
        exec.amorphic()
            .insert_rows(table, &base.raw_rows)
            .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
    }
    Ok(())
}

fn materialize_all_branch_tables(entry: &SqlBranchEntry) -> Result<(), QueryErrorResponse> {
    let mut mat = entry
        .materialized_tables
        .lock()
        .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
    for (table, base) in &entry.base_tables {
        if !mat.contains(table) {
            materialize_one_branch_table(entry, table, base)?;
            mat.insert(table.clone());
        }
    }
    Ok(())
}

fn sql_mentions_identifier_ci(sql_lower: &str, ident_lower: &str) -> bool {
    let bytes = sql_lower.as_bytes();
    let needle = ident_lower.as_bytes();
    if needle.is_empty() || needle.len() > bytes.len() {
        return false;
    }
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut pos = 0usize;
    while let Some(rel) = sql_lower[pos..].find(ident_lower) {
        let idx = pos + rel;
        let end = idx + needle.len();
        let left_ok = idx == 0 || !is_word(bytes[idx - 1]);
        let right_ok = end == bytes.len() || !is_word(bytes[end]);
        if left_ok && right_ok {
            return true;
        }
        pos = idx + 1;
    }
    false
}

fn materialize_referenced_branch_tables(
    entry: &SqlBranchEntry,
    sql: &str,
) -> Result<(), QueryErrorResponse> {
    if entry.base_tables.is_empty() {
        return Ok(());
    }
    let sql_lower = sql.to_ascii_lowercase();
    let needs_catalog_all = sql_lower.contains("show tables")
        || sql_lower.contains("information_schema")
        || sql_lower.contains("pg_class")
        || sql_lower.contains("pg_tables");
    if needs_catalog_all {
        return materialize_all_branch_tables(entry);
    }
    let mut mat = entry
        .materialized_tables
        .lock()
        .map_err(|e| QueryErrorResponse::execution_error(&e.to_string()))?;
    for (table, base) in &entry.base_tables {
        if mat.contains(table) {
            continue;
        }
        let tbl_lower = table.to_ascii_lowercase();
        if sql_mentions_identifier_ci(&sql_lower, &tbl_lower) {
            materialize_one_branch_table(entry, table, base)?;
            mat.insert(table.clone());
        }
    }
    Ok(())
}

fn raw_rows_to_json_rows(raw_rows: &[RowData], columns: &[String]) -> Vec<Vec<serde_json::Value>> {
    raw_rows
        .iter()
        .map(|row_data| {
            columns
                .iter()
                .map(|col_name| {
                    row_data
                        .columns
                        .iter()
                        .position(|c| c == col_name)
                        .and_then(|idx| row_data.values.get(idx))
                        .map(crate::json_ops::ast_value_to_json)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect()
        })
        .collect()
}

fn compute_three_way_diff(
    entry: &SqlBranchEntry,
    parent_storage: &AmorphicTableStorage,
) -> Result<SqlBranchDiffReport, QueryErrorResponse> {
    let Some(branch_exec) = entry.executor.get() else {
        return Ok(SqlBranchDiffReport {
            branch: entry.name.clone(),
            parent: entry.parent.clone(),
            energy_spent_uj: entry.energy_spent_uj.load(Ordering::Relaxed),
            energy_budget_uj: entry.energy_budget_uj,
            tables: Vec::new(),
            total_inserted: 0,
            total_updated: 0,
            total_deleted: 0,
            total_conflicts: 0,
        });
    };
    let branch_storage = branch_exec.amorphic();
    let materialized: HashSet<String> = entry
        .materialized_tables
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    let mut all_tables: HashSet<String> = entry.base_tables.keys().cloned().collect();
    for t in branch_storage.list_tables() {
        all_tables.insert(t);
    }
    let mut sorted_tables: Vec<String> = all_tables.into_iter().collect();
    sorted_tables.sort();

    let mut table_diffs = Vec::new();
    let mut total_inserted = 0usize;
    let mut total_updated = 0usize;
    let mut total_deleted = 0usize;
    let mut total_conflicts = 0usize;

    for table in sorted_tables {
        if let Some(base) = entry.base_tables.get(&table) {
            if !materialized.contains(&table) {
                continue;
            }
            let base_keyed = base.keyed_rows();
            let b_rows = snapshot_table_rows(branch_storage, &table, &base.columns);
            let p_rows = snapshot_table_rows(parent_storage, &table, &base.columns);
            let b_map = index_rows_by_key(&b_rows, base.primary_key_idx);
            let p_map = index_rows_by_key(&p_rows, base.primary_key_idx);

            let mut all_keys: HashSet<&String> = base_keyed.keys().collect();
            all_keys.extend(b_map.keys());

            let mut ins = 0usize;
            let mut upd = 0usize;
            let mut del = 0usize;
            let mut conflicts = Vec::new();

            for k in all_keys {
                let base_v = base_keyed.get(k);
                let branch_v = b_map.get(k);
                let parent_v = p_map.get(k);

                if base_v == branch_v {
                    continue;
                }

                if parent_v != base_v && parent_v != branch_v {
                    conflicts.push(k.clone());
                }

                match (base_v, branch_v) {
                    (None, Some(_)) => ins += 1,
                    (Some(_), Some(_)) => upd += 1,
                    (Some(_), None) => del += 1,
                    (None, None) => {}
                }
            }
            conflicts.sort();
            if ins > 0 || upd > 0 || del > 0 || !conflicts.is_empty() {
                total_inserted += ins;
                total_updated += upd;
                total_deleted += del;
                total_conflicts += conflicts.len();
                table_diffs.push(TableBranchDiff {
                    table,
                    is_new_table: false,
                    inserted: ins,
                    updated: upd,
                    deleted: del,
                    conflicts,
                });
            }
        } else {
            let cols = branch_storage.columns(&table).unwrap_or_default();
            let b_rows = snapshot_table_rows(branch_storage, &table, &cols);
            let parent_also_created = parent_storage.list_tables().iter().any(|t| t == &table);
            let conflicts = if parent_also_created {
                vec!["__table_ddl__".to_string()]
            } else {
                Vec::new()
            };
            total_inserted += b_rows.len();
            total_conflicts += conflicts.len();
            table_diffs.push(TableBranchDiff {
                table,
                is_new_table: true,
                inserted: b_rows.len(),
                updated: 0,
                deleted: 0,
                conflicts,
            });
        }
    }

    Ok(SqlBranchDiffReport {
        branch: entry.name.clone(),
        parent: entry.parent.clone(),
        energy_spent_uj: entry.energy_spent_uj.load(Ordering::Relaxed),
        energy_budget_uj: entry.energy_budget_uj,
        tables: table_diffs,
        total_inserted,
        total_updated,
        total_deleted,
        total_conflicts,
    })
}

fn apply_three_way_merge(
    entry: &SqlBranchEntry,
    target_exec: &SimpleQueryExecutor,
    delete_after: bool,
) -> Result<SqlBranchMergeReport, QueryErrorResponse> {
    let diff = compute_three_way_diff(entry, target_exec.amorphic())?;
    if diff.total_conflicts > 0 {
        let details: Vec<String> = diff
            .tables
            .iter()
            .filter(|t| !t.conflicts.is_empty())
            .map(|t| format!("{}: [{}]", t.table, t.conflicts.join(", ")))
            .collect();
        return Err(sqlstate_err(
            "40001",
            format!(
                "3-way merge conflict between branch '{}' and '{}' on {} key(s): {}",
                entry.name,
                entry.parent,
                diff.total_conflicts,
                details.join("; ")
            ),
        ));
    }

    if diff.tables.is_empty() {
        return Ok(SqlBranchMergeReport {
            branch: entry.name.clone(),
            target: entry.parent.clone(),
            tables_merged: 0,
            rows_inserted: 0,
            rows_updated: 0,
            rows_deleted: 0,
            branch_deleted: delete_after,
        });
    }
    let branch_storage = entry.executor().amorphic();
    let target_storage = target_exec.amorphic();
    let mut rows_inserted = 0usize;
    let mut rows_updated = 0usize;
    let mut rows_deleted = 0usize;

    for td in &diff.tables {
        if td.is_new_table {
            let col_defs = branch_storage
                .get_column_defs(&td.table)
                .unwrap_or_default();
            let cols = branch_storage.columns(&td.table).unwrap_or_default();
            let ddl = if !col_defs.is_empty() {
                render_create_table_sql(&td.table, &col_defs)
            } else {
                let c: Vec<String> = cols.iter().map(|n| format!("\"{n}\" TEXT")).collect();
                format!("CREATE TABLE \"{}\" ({})", td.table, c.join(", "))
            };
            let _ = target_exec.execute(&QueryRequest {
                sql: ddl,
                ..Default::default()
            })?;
            let rows = snapshot_table_rows(branch_storage, &td.table, &cols);
            seed_rows_into_executor(target_exec, &td.table, &cols, &rows)?;
            rows_inserted += rows.len();
            continue;
        }

        let Some(base) = entry.base_tables.get(&td.table) else {
            continue;
        };
        let base_keyed = base.keyed_rows();
        let b_rows = snapshot_table_rows(branch_storage, &td.table, &base.columns);
        let b_map = index_rows_by_key(&b_rows, base.primary_key_idx);
        let p_rows = snapshot_table_rows(target_storage, &td.table, &base.columns);
        let p_map = index_rows_by_key(&p_rows, base.primary_key_idx);

        let pk_col = base
            .primary_key_idx
            .and_then(|idx| base.columns.get(idx))
            .cloned();

        for (k, base_row) in base_keyed.iter() {
            match b_map.get(k) {
                None => {
                    if p_map.contains_key(k) {
                        delete_row_in_executor(
                            target_exec,
                            &td.table,
                            &base.columns,
                            pk_col.as_deref(),
                            base_row,
                        )?;
                        rows_deleted += 1;
                    }
                }
                Some(new_row) if new_row != base_row => {
                    delete_row_in_executor(
                        target_exec,
                        &td.table,
                        &base.columns,
                        pk_col.as_deref(),
                        base_row,
                    )?;
                    seed_rows_into_executor(
                        target_exec,
                        &td.table,
                        &base.columns,
                        std::slice::from_ref(new_row),
                    )?;
                    rows_updated += 1;
                }
                _ => {}
            }
        }

        let mut to_insert = Vec::new();
        for (k, new_row) in &b_map {
            if !base_keyed.contains_key(k) && !p_map.contains_key(k) {
                to_insert.push(new_row.clone());
            }
        }
        if !to_insert.is_empty() {
            seed_rows_into_executor(target_exec, &td.table, &base.columns, &to_insert)?;
            rows_inserted += to_insert.len();
        }
    }

    Ok(SqlBranchMergeReport {
        branch: entry.name.clone(),
        target: entry.parent.clone(),
        tables_merged: diff.tables.len(),
        rows_inserted,
        rows_updated,
        rows_deleted,
        branch_deleted: delete_after,
    })
}

fn sqlstate_err(code: &str, message: String) -> QueryErrorResponse {
    QueryErrorResponse {
        code: code.to_string(),
        message,
        line: None,
        column: None,
    }
}

fn validate_sql_against_guardrails(
    sql: &str,
    block_ddl: bool,
    require_where: bool,
) -> Result<(), QueryErrorResponse> {
    let trimmed = sql.trim();
    let upper = trimmed.to_ascii_uppercase();
    if block_ddl && (upper.starts_with("DROP TABLE") || upper.starts_with("TRUNCATE")) {
        return Err(sqlstate_err(
            "42501",
            "agent guardrail blocked destructive DDL (DROP TABLE / TRUNCATE)".into(),
        ));
    }
    if require_where {
        if upper.starts_with("DELETE FROM ") && !upper.contains(" WHERE ") {
            return Err(sqlstate_err(
                "42501",
                "agent guardrail blocked unbounded DELETE without WHERE clause".into(),
            ));
        }
        if upper.starts_with("UPDATE ") && upper.contains(" SET ") && !upper.contains(" WHERE ") {
            return Err(sqlstate_err(
                "42501",
                "agent guardrail blocked unbounded UPDATE without WHERE clause".into(),
            ));
        }
    }
    Ok(())
}

fn detect_primary_key_idx(columns: &[String], col_defs: &[ColumnDefInfo]) -> Option<usize> {
    for def in col_defs {
        if def.primary_key
            && let Some(pos) = columns.iter().position(|c| c == &def.name)
        {
            return Some(pos);
        }
    }
    columns.iter().position(|c| c.eq_ignore_ascii_case("id"))
}

fn snapshot_table_rows(
    storage: &AmorphicTableStorage,
    table: &str,
    columns: &[String],
) -> Vec<Vec<serde_json::Value>> {
    let Ok(rows_with_ids) = storage.scan_with_record_ids(table) else {
        return Vec::new();
    };
    rows_with_ids
        .into_iter()
        .map(|(_, row_data)| {
            columns
                .iter()
                .map(|col_name| {
                    row_data
                        .columns
                        .iter()
                        .position(|c| c == col_name)
                        .and_then(|idx| row_data.values.get(idx))
                        .map(crate::json_ops::ast_value_to_json)
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect()
        })
        .collect()
}

fn index_rows_by_key(
    rows: &[Vec<serde_json::Value>],
    pk_idx: Option<usize>,
) -> BTreeMap<String, Vec<serde_json::Value>> {
    let mut map = BTreeMap::new();
    for (idx, r) in rows.iter().enumerate() {
        let key = if let Some(pi) = pk_idx
            && let Some(val) = r.get(pi)
            && !val.is_null()
        {
            format!("pk:{val}")
        } else {
            format!(
                "row:{}:{}",
                serde_json::to_string(r).unwrap_or_default(),
                idx
            )
        };
        map.insert(key, r.clone());
    }
    map
}

fn render_create_table_sql(table: &str, defs: &[ColumnDefInfo]) -> String {
    let parts: Vec<String> = defs
        .iter()
        .map(|d| {
            let mut s = format!("\"{}\" {}", d.name, d.data_type);
            if d.primary_key {
                s.push_str(" PRIMARY KEY");
            } else if !d.nullable {
                s.push_str(" NOT NULL");
            }
            s
        })
        .collect();
    format!("CREATE TABLE \"{table}\" ({})", parts.join(", "))
}

fn seed_rows_into_executor(
    exec: &SimpleQueryExecutor,
    table: &str,
    columns: &[String],
    rows: &[Vec<serde_json::Value>],
) -> Result<(), QueryErrorResponse> {
    if rows.is_empty() || columns.is_empty() {
        return Ok(());
    }
    let quoted_cols: Vec<String> = columns.iter().map(|c| format!("\"{c}\"")).collect();
    for chunk in rows.chunks(200) {
        let tuples: Vec<String> = chunk
            .iter()
            .map(|r| {
                let lits: Vec<String> = r.iter().map(json_to_sql_literal).collect();
                format!("({})", lits.join(", "))
            })
            .collect();
        let sql = format!(
            "INSERT INTO \"{table}\" ({}) VALUES {}",
            quoted_cols.join(", "),
            tuples.join(", ")
        );
        let _ = exec.execute(&QueryRequest {
            sql,
            ..Default::default()
        })?;
    }
    Ok(())
}

fn sql_ident(name: &str) -> String {
    let mut chars = name.chars();
    let ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if ok {
        name.to_string()
    } else {
        format!("\"{}\"", name.replace('"', "\"\""))
    }
}

fn delete_row_in_executor(
    exec: &SimpleQueryExecutor,
    table: &str,
    columns: &[String],
    pk_col: Option<&str>,
    row: &[serde_json::Value],
) -> Result<(), QueryErrorResponse> {
    let where_sql = if let Some(pk) = pk_col
        && let Some(pos) = columns.iter().position(|c| c == pk)
        && let Some(val) = row.get(pos)
    {
        format!("{} = {}", sql_ident(pk), json_to_sql_literal(val))
    } else {
        let preds: Vec<String> = columns
            .iter()
            .zip(row.iter())
            .map(|(c, v)| {
                if v.is_null() {
                    format!("{} IS NULL", sql_ident(c))
                } else {
                    format!("{} = {}", sql_ident(c), json_to_sql_literal(v))
                }
            })
            .collect();
        preds.join(" AND ")
    };
    let sql = format!("DELETE FROM {} WHERE {where_sql}", sql_ident(table));
    let _ = exec.execute(&QueryRequest {
        sql,
        ..Default::default()
    })?;
    Ok(())
}

fn json_to_sql_literal(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "NULL".into(),
        serde_json::Value::Bool(b) => {
            if *b {
                "TRUE".into()
            } else {
                "FALSE".into()
            }
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => format!("'{}'", s.replace('\'', "''")),
        other => format!("'{}'", other.to_string().replace('\'', "''")),
    }
}

fn upsert_step_in_map(
    map: &mut BTreeMap<String, Vec<WorkflowStepRecord>>,
    rec: WorkflowStepRecord,
) {
    let list = map.entry(rec.workflow_id.clone()).or_default();
    if let Some(existing) = list.iter_mut().find(|s| s.step_id == rec.step_id) {
        *existing = rec;
    } else {
        list.push(rec);
    }
}

fn simple_rows_response(
    columns: Vec<String>,
    rows: Vec<Vec<serde_json::Value>>,
    affected_rows: Option<usize>,
    start: Instant,
    algo: &str,
) -> QueryResponse {
    QueryResponse {
        columns,
        rows,
        affected_rows,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some(algo.to_string()),
        session_id: None,
        viz_hint: None,
        energy_receipt: None,
    }
}

fn tokenize_simple(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c == '\'' || c == '"' {
            let q = c;
            chars.next();
            let mut s = String::new();
            while let Some(ch) = chars.next() {
                if ch == q {
                    break;
                }
                s.push(ch);
            }
            out.push(s);
        } else {
            let mut s = String::new();
            while let Some(&ch) = chars.peek() {
                if ch.is_whitespace() || ch == ';' {
                    break;
                }
                s.push(ch);
                chars.next();
            }
            out.push(s);
        }
    }
    out
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cow_sql_branch_isolation_diff_3way_conflict_and_merge() {
        let exec = SimpleQueryExecutor::new();
        exec.execute(&QueryRequest {
            sql: "CREATE TABLE accounts (id INT PRIMARY KEY, owner TEXT, balance NUMERIC(18, 2))"
                .into(),
            ..Default::default()
        })
        .unwrap();
        exec.execute(&QueryRequest {
            sql: "INSERT INTO accounts VALUES (1, 'Alice', 100.25), (2, 'Bob', 50.75)".into(),
            ..Default::default()
        })
        .unwrap();

        // Fork isolated agent branch with strict guardrails
        exec.execute(&QueryRequest {
            sql: "CREATE BRANCH agent_sandbox ENERGY BUDGET 5000000 GUARDRAILS STRICT".into(),
            ..Default::default()
        })
        .unwrap();

        // Destructive DDL & unbounded DELETE must be blocked on strict branch
        let drop_err = exec.execute(&QueryRequest {
            sql: "DROP TABLE accounts".into(),
            branch_id: Some("agent_sandbox".into()),
            ..Default::default()
        });
        assert!(drop_err.is_err());
        assert_eq!(drop_err.unwrap_err().code, "42501");

        let unb_del = exec.execute(&QueryRequest {
            sql: "DELETE FROM accounts".into(),
            branch_id: Some("agent_sandbox".into()),
            ..Default::default()
        });
        assert!(unb_del.is_err());

        // Scoped mutations on branch stay invisible on main
        exec.execute(&QueryRequest {
            sql: "INSERT INTO accounts VALUES (3, 'Charlie', 25.00)".into(),
            branch_id: Some("agent_sandbox".into()),
            ..Default::default()
        })
        .unwrap();
        exec.execute(&QueryRequest {
            sql: "UPDATE accounts SET balance = 125.25 WHERE id = 1".into(),
            branch_id: Some("agent_sandbox".into()),
            ..Default::default()
        })
        .unwrap();

        let main_rows = exec
            .execute(&QueryRequest {
                sql: "SELECT COUNT(*), SUM(balance) FROM accounts".into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(main_rows.rows[0][0], serde_json::json!(2));
        assert!(crate::json_ops::json_equals(
            &main_rows.rows[0][1],
            &serde_json::json!(151)
        ));

        // DIFF shows 1 insert + 1 update + 0 conflicts
        let diff = exec
            .execute(&QueryRequest {
                sql: "DIFF BRANCH agent_sandbox".into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(diff.rows[0][2], serde_json::json!(1)); // inserted
        assert_eq!(diff.rows[0][3], serde_json::json!(1)); // updated
        assert_eq!(diff.rows[0][5], serde_json::json!(0)); // conflicts

        // MERGE applies branch changes cleanly to main
        exec.execute(&QueryRequest {
            sql: "MERGE BRANCH agent_sandbox DELETE AFTER".into(),
            ..Default::default()
        })
        .unwrap();
        let merged = exec
            .execute(&QueryRequest {
                sql: "SELECT COUNT(*), SUM(balance) FROM accounts".into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(merged.rows[0][0], serde_json::json!(3));
        assert!(crate::json_ops::json_equals(
            &merged.rows[0][1],
            &serde_json::json!(201)
        ));

        // Now verify true 3-way write-write conflict detection (`SQLSTATE 40001`)
        exec.execute(&QueryRequest {
            sql: "CREATE BRANCH conflicting_agent".into(),
            ..Default::default()
        })
        .unwrap();
        exec.execute(&QueryRequest {
            sql: "UPDATE accounts SET balance = 999.00 WHERE id = 2".into(),
            branch_id: Some("conflicting_agent".into()),
            ..Default::default()
        })
        .unwrap();
        exec.execute(&QueryRequest {
            sql: "UPDATE accounts SET balance = 777.00 WHERE id = 2".into(),
            ..Default::default()
        })
        .unwrap();
        let merge_conflict = exec.execute(&QueryRequest {
            sql: "MERGE BRANCH conflicting_agent".into(),
            ..Default::default()
        });
        assert!(merge_conflict.is_err());
        assert_eq!(merge_conflict.unwrap_err().code, "40001");
    }

    #[test]
    fn durable_workflow_checkpoint_and_for_update_skip_locked() {
        let dir = tempfile::tempdir().unwrap();
        {
            let exec = SimpleQueryExecutor::open(dir.path());
            let r1 = exec
                .execute(&QueryRequest {
                    sql: "WORKFLOW CHECKPOINT 'wf_pay_1' STEP 'kyc_check' STATUS 'completed' OUTPUT '{\"passed\":true}' INPUT_HASH 'abc123' ENERGY_UJ 350".into(),
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(r1.rows[0][4], serde_json::json!(false)); // first run
        }
        // Reopen database after simulated crash -> memoized step replays without re-execution!
        let exec = SimpleQueryExecutor::open(dir.path());
        let r2 = exec
            .execute(&QueryRequest {
                sql: "WORKFLOW CHECKPOINT 'wf_pay_1' STEP 'kyc_check' STATUS 'completed' OUTPUT '{\"passed\":true}' INPUT_HASH 'abc123' ENERGY_UJ 350".into(),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(r2.rows[0][4], serde_json::json!(true)); // memoized_replay = true

        // Test SELECT ... FOR UPDATE NOWAIT and SKIP LOCKED
        exec.execute(&QueryRequest {
            sql: "CREATE TABLE jobs (id INT PRIMARY KEY, payload TEXT)".into(),
            ..Default::default()
        })
        .unwrap();
        exec.execute(&QueryRequest {
            sql: "INSERT INTO jobs VALUES (10, 'wire_a'), (20, 'wire_b')".into(),
            ..Default::default()
        })
        .unwrap();
        let s1 = exec
            .execute(&QueryRequest {
                sql: "BEGIN".into(),
                ..Default::default()
            })
            .unwrap()
            .session_id
            .unwrap();
        let s2 = exec
            .execute(&QueryRequest {
                sql: "BEGIN".into(),
                ..Default::default()
            })
            .unwrap()
            .session_id
            .unwrap();

        let locked_10 = exec
            .execute(&QueryRequest {
                sql: "SELECT id FROM jobs WHERE id = 10 FOR UPDATE".into(),
                session_id: Some(s1.clone()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(locked_10.rows.len(), 1);

        let nowait_err = exec.execute(&QueryRequest {
            sql: "SELECT id FROM jobs WHERE id = 10 FOR UPDATE NOWAIT".into(),
            session_id: Some(s2.clone()),
            ..Default::default()
        });
        assert_eq!(nowait_err.unwrap_err().code, "55P03");

        let skipped = exec
            .execute(&QueryRequest {
                sql: "SELECT id FROM jobs ORDER BY id FOR UPDATE SKIP LOCKED".into(),
                session_id: Some(s2.clone()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(skipped.rows, vec![vec![serde_json::json!(20)]]);

        exec.execute(&QueryRequest {
            sql: "COMMIT".into(),
            session_id: Some(s1),
            ..Default::default()
        })
        .unwrap();
        exec.execute(&QueryRequest {
            sql: "COMMIT".into(),
            session_id: Some(s2),
            ..Default::default()
        })
        .unwrap();
    }
}
