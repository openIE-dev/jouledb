//! SpacetimeDB-Class In-Engine Deterministic Fuel-Metered Reducers
//!
//! Co-locates deterministic application state reducers directly with JouleDB's transactional
//! storage and Copy-on-Write branches (`joule_db_branch::BranchManager`):
//! - **Instruction Fuel <-> Microjoule Isomorphism**: Every reducer step and byte of KV I/O
//!   consumes calibrated instruction fuel mapped directly into microjoules (`fuel_per_uj`),
//!   enforcing hard per-invocation AND per-branch [`joule_db_branch::Branch`] energy budgets.
//! - **Atomic Rollback on Fuel Exhaustion**: Reducers stage writes in a transactional write-set;
//!   if an infinite loop or runaway agent reducer exhausts its fuel or branch µJ cap, the
//!   entire write-set is discarded with zero partial state corruption!

use joule_db_branch::{BranchError, manager::BranchManager};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Errors returned when registering or invoking a deterministic reducer.
#[derive(Debug, Error)]
pub enum ReducerError {
    #[error("reducer not found: {0}")]
    NotFound(String),

    #[error("fuel exhausted during reducer '{reducer}': limit {fuel_limit} fuel ({budget_uj} µJ)")]
    FuelExhausted {
        reducer: String,
        fuel_limit: u64,
        budget_uj: u64,
    },

    #[error("branch energy error: {0}")]
    BranchEnergy(#[from] BranchError),

    #[error("reducer assertion failed: {0}")]
    AssertionFailed(String),

    #[error("internal reducer lock error: {0}")]
    Internal(String),
}

/// Calibrated conversion between deterministic VM/Wasm fuel and physical microjoules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReducerEnergyPolicy {
    /// VM/Wasm instructions per 1 microjoule (default: `1_000` instructions ≈ `1 µJ` @ ~1 nJ/op).
    pub fuel_per_uj: u64,
    /// Fuel cost per KV read byte (`1` fuel ≈ `1 nJ`).
    pub fuel_per_read_byte: u64,
    /// Fuel cost per KV write byte (`4` fuel ≈ `4 nJ`).
    pub fuel_per_write_byte: u64,
    /// Maximum microjoules allowed for a single reducer call (`max_call_budget_uj`).
    pub max_call_budget_uj: u64,
}

impl Default for ReducerEnergyPolicy {
    fn default() -> Self {
        Self {
            fuel_per_uj: 1_000,
            fuel_per_read_byte: 1,
            fuel_per_write_byte: 4,
            max_call_budget_uj: 500, // 500 µJ = 500,000 fuel instructions default per call
        }
    }
}

impl ReducerEnergyPolicy {
    /// Convert consumed fuel units into microjoules (ceiling-rounded, minimum 1 µJ).
    pub fn fuel_to_uj(&self, fuel: u64) -> u64 {
        let denom = self.fuel_per_uj.max(1);
        fuel.div_ceil(denom).max(1)
    }

    /// Convert a microjoule ceiling into deterministic instruction fuel.
    pub fn uj_to_fuel(&self, uj: u64) -> u64 {
        uj.saturating_mul(self.fuel_per_uj.max(1))
    }
}

/// Deterministic reducer bytecode operation (executes identically on every platform without
/// requiring JIT compilation, and mirrors Wasmtime host imports `db_get`/`db_put`/`db_delete`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ReducerIrOp {
    /// Put literal or argument-interpolated UTF-8 value into KV key (`$arg` replaced with call arg).
    KvPut { key: String, value_template: String },
    /// Delete a KV key.
    KvDelete { key: String },
    /// Atomically increment an `i64` decimal counter stored at `key` by `delta`.
    AddCounter { key: String, delta: i64 },
    /// Assert that `key` currently equals `expected` (aborts & rolls back if violated).
    AssertEq { key: String, expected: String },
    /// Emit a structured subscription event to downstream reactive clients.
    EmitEvent {
        topic: String,
        payload_template: String,
    },
    /// Burn `iterations` loop steps (used for compute kernels & deterministic fuel-cap tests).
    ComputeLoop { iterations: u64 },
}

/// Registered deterministic reducer definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReducerModule {
    pub name: String,
    pub ops: Vec<ReducerIrOp>,
}

/// Auditable receipt emitted for every committed reducer invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReducerExecutionReceipt {
    pub reducer_name: String,
    pub branch_name: String,
    pub fuel_consumed: u64,
    pub energy_uj: u64,
    pub keys_modified: usize,
    pub emitted_events: Vec<(String, String)>,
}

/// Transactional in-engine reducer runtime with branch-level microjoule enforcement.
pub struct ReducerRuntime {
    policy: ReducerEnergyPolicy,
    branch_manager: Arc<BranchManager>,
    modules: RwLock<HashMap<String, ReducerModule>>,
    /// Per-branch KV state (`branch_name -> key -> value`).
    branch_kv: RwLock<HashMap<String, BTreeMap<String, Vec<u8>>>>,
}

impl ReducerRuntime {
    pub fn new(branch_manager: Arc<BranchManager>, policy: ReducerEnergyPolicy) -> Self {
        Self {
            policy,
            branch_manager,
            modules: RwLock::new(HashMap::new()),
            branch_kv: RwLock::new(HashMap::new()),
        }
    }

    /// Register a deterministic reducer module.
    pub fn register_reducer(&self, module: ReducerModule) -> Result<(), ReducerError> {
        self.modules
            .write()
            .map_err(|e| ReducerError::Internal(e.to_string()))?
            .insert(module.name.clone(), module);
        Ok(())
    }

    /// Read committed KV value on `branch_name`.
    pub fn get_committed(
        &self,
        branch_name: &str,
        key: &str,
    ) -> Result<Option<Vec<u8>>, ReducerError> {
        let store = self
            .branch_kv
            .read()
            .map_err(|e| ReducerError::Internal(e.to_string()))?;
        Ok(store.get(branch_name).and_then(|m| m.get(key).cloned()))
    }

    /// Execute `reducer_name` atomically on `branch_name` with input string `arg`.
    ///
    /// Stages all KV mutations in an isolated write-set; if fuel or the branch's microjoule
    /// budget is exceeded at any instruction, the write-set is dropped untouched!
    pub fn invoke(
        &self,
        branch_name: &str,
        reducer_name: &str,
        arg: &str,
    ) -> Result<ReducerExecutionReceipt, ReducerError> {
        let module = self
            .modules
            .read()
            .map_err(|e| ReducerError::Internal(e.to_string()))?
            .get(reducer_name)
            .cloned()
            .ok_or_else(|| ReducerError::NotFound(reducer_name.to_string()))?;

        // Effective budget = min(policy.max_call_budget_uj, branch.energy_remaining_uj)
        let branch_info = self.branch_manager.get_branch(branch_name)?;
        let remaining_opt = branch_info
            .energy_budget_uj
            .map(|b| b.saturating_sub(branch_info.energy_spent_uj));
        let effective_budget_uj = match remaining_opt {
            Some(0) => {
                return Err(ReducerError::BranchEnergy(
                    BranchError::EnergyBudgetExhausted {
                        spent_uj: branch_info.energy_spent_uj,
                        budget_uj: branch_info.energy_budget_uj.unwrap_or(0),
                    },
                ));
            }
            Some(rem) => rem.min(self.policy.max_call_budget_uj),
            None => self.policy.max_call_budget_uj,
        };
        let fuel_limit = self.policy.uj_to_fuel(effective_budget_uj);
        let mut fuel_used: u64 = 10; // Base dispatch cost

        let base_snapshot: BTreeMap<String, Vec<u8>> = self
            .branch_kv
            .read()
            .map_err(|e| ReducerError::Internal(e.to_string()))?
            .get(branch_name)
            .cloned()
            .unwrap_or_default();

        // Staged transactional overlay (`Some(bytes)` = upsert, `None` = tombstone)
        let mut staged: BTreeMap<String, Option<Vec<u8>>> = BTreeMap::new();
        let mut events: Vec<(String, String)> = Vec::new();

        let charge_fuel = |used: &mut u64, delta: u64| -> Result<(), ReducerError> {
            *used = used.saturating_add(delta);
            if *used > fuel_limit {
                return Err(ReducerError::FuelExhausted {
                    reducer: reducer_name.to_string(),
                    fuel_limit,
                    budget_uj: effective_budget_uj,
                });
            }
            Ok(())
        };

        let read_staged =
            |key: &str, staged_map: &BTreeMap<String, Option<Vec<u8>>>| -> Option<Vec<u8>> {
                match staged_map.get(key) {
                    Some(opt) => opt.clone(),
                    None => base_snapshot.get(key).cloned(),
                }
            };

        for op in &module.ops {
            charge_fuel(&mut fuel_used, 5)?;
            match op {
                ReducerIrOp::KvPut {
                    key,
                    value_template,
                } => {
                    let val = value_template.replace("$arg", arg).into_bytes();
                    let bytes_cost = ((key.len() + val.len()) as u64)
                        .saturating_mul(self.policy.fuel_per_write_byte);
                    charge_fuel(&mut fuel_used, bytes_cost)?;
                    staged.insert(key.clone(), Some(val));
                }
                ReducerIrOp::KvDelete { key } => {
                    charge_fuel(&mut fuel_used, (key.len() as u64) * 2)?;
                    staged.insert(key.clone(), None);
                }
                ReducerIrOp::AddCounter { key, delta } => {
                    let cur_bytes = read_staged(key, &staged).unwrap_or_else(|| b"0".to_vec());
                    charge_fuel(
                        &mut fuel_used,
                        (cur_bytes.len() as u64) * self.policy.fuel_per_read_byte,
                    )?;
                    let cur_val = std::str::from_utf8(&cur_bytes)
                        .ok()
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or(0);
                    let next_bytes = (cur_val + delta).to_string().into_bytes();
                    charge_fuel(
                        &mut fuel_used,
                        (next_bytes.len() as u64) * self.policy.fuel_per_write_byte,
                    )?;
                    staged.insert(key.clone(), Some(next_bytes));
                }
                ReducerIrOp::AssertEq { key, expected } => {
                    let actual = read_staged(key, &staged)
                        .and_then(|b| String::from_utf8(b).ok())
                        .unwrap_or_default();
                    charge_fuel(&mut fuel_used, (actual.len() as u64) + 5)?;
                    if actual != *expected {
                        return Err(ReducerError::AssertionFailed(format!(
                            "key '{key}' = '{actual}', expected '{expected}'"
                        )));
                    }
                }
                ReducerIrOp::EmitEvent {
                    topic,
                    payload_template,
                } => {
                    let payload = payload_template.replace("$arg", arg);
                    charge_fuel(&mut fuel_used, (topic.len() + payload.len()) as u64)?;
                    events.push((topic.clone(), payload));
                }
                ReducerIrOp::ComputeLoop { iterations } => {
                    // Charge in bounded chunks so runaway loops fail fast with exact fuel cap!
                    charge_fuel(&mut fuel_used, *iterations)?;
                }
            }
        }

        let energy_uj = self.policy.fuel_to_uj(fuel_used);
        // Atomically charge the CoW branch energy budget before publishing write-set!
        self.branch_manager.record_energy(branch_name, energy_uj)?;

        let keys_modified = staged.len();
        {
            let mut store = self
                .branch_kv
                .write()
                .map_err(|e| ReducerError::Internal(e.to_string()))?;
            let branch_map = store.entry(branch_name.to_string()).or_default();
            for (k, v_opt) in staged {
                match v_opt {
                    Some(v) => {
                        branch_map.insert(k, v);
                    }
                    None => {
                        branch_map.remove(&k);
                    }
                }
            }
        }

        Ok(ReducerExecutionReceipt {
            reducer_name: reducer_name.to_string(),
            branch_name: branch_name.to_string(),
            fuel_consumed: fuel_used,
            energy_uj,
            keys_modified,
            emitted_events: events,
        })
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct InvokeReducerHttpReq {
    #[serde(default)]
    pub arg: String,
}

/// Build the Axum sub-router exposing SpacetimeDB-class fuel-metered reducer endpoints.
pub fn reducer_routes(runtime: Arc<ReducerRuntime>) -> axum::Router {
    use axum::{
        Json,
        extract::{Path, State},
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
    };

    axum::Router::new()
        .route(
            "/api/v1/reducers/register",
            post(
                |State(rt): State<Arc<ReducerRuntime>>, Json(module): Json<ReducerModule>| async move {
                    match rt.register_reducer(module) {
                        Ok(()) => (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/reducers/{branch}/{name}/invoke",
            post(
                |State(rt): State<Arc<ReducerRuntime>>,
                 Path((branch, name)): Path<(String, String)>,
                 Json(req): Json<InvokeReducerHttpReq>| async move {
                    match rt.invoke(&branch, &name, &req.arg) {
                        Ok(receipt) => (StatusCode::OK, Json(serde_json::json!(receipt))).into_response(),
                        Err(e) => (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            Json(serde_json::json!({ "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/reducers/{branch}/kv/{key}",
            get(
                |State(rt): State<Arc<ReducerRuntime>>,
                 Path((branch, key)): Path<(String, String)>| async move {
                    match rt.get_committed(&branch, &key) {
                        Ok(Some(bytes)) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "found": true,
                                "value": String::from_utf8_lossy(&bytes),
                            })),
                        )
                            .into_response(),
                        Ok(None) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "found": false, "value": null })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({ "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .with_state(runtime)
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_db_branch::CreateBranchRequest;

    #[test]
    fn test_fuel_metered_reducer_commit_and_atomic_infinite_loop_rollback() {
        let bm = Arc::new(BranchManager::new(0));
        // Create an agent branch capped at 50 µJ (= 50,000 instruction fuel)
        bm.create_branch(CreateBranchRequest {
            name: "game-world-1".to_string(),
            parent: Some("main".to_string()),
            at_lsn: Some(0),
            energy_budget_uj: Some(50),
            description: Some("SpacetimeDB-style reducer branch".to_string()),
            tags: vec![],
        })
        .unwrap();

        let rt = ReducerRuntime::new(bm.clone(), ReducerEnergyPolicy::default());
        rt.register_reducer(ReducerModule {
            name: "move_player".to_string(),
            ops: vec![
                ReducerIrOp::KvPut {
                    key: "player:1:pos".to_string(),
                    value_template: "$arg".to_string(),
                },
                ReducerIrOp::AddCounter {
                    key: "world:ticks".to_string(),
                    delta: 1,
                },
                ReducerIrOp::EmitEvent {
                    topic: "player.moved".to_string(),
                    payload_template: "pos=$arg".to_string(),
                },
            ],
        })
        .unwrap();

        let r1 = rt.invoke("game-world-1", "move_player", "10,24").unwrap();
        assert_eq!(r1.keys_modified, 2);
        assert_eq!(r1.emitted_events.len(), 1);
        assert!(r1.energy_uj >= 1 && r1.energy_uj <= 5);
        assert_eq!(
            rt.get_committed("game-world-1", "player:1:pos")
                .unwrap()
                .unwrap(),
            b"10,24"
        );

        // Register a runaway reducer that stages a mutation and then enters a 10,000,000-step loop!
        rt.register_reducer(ReducerModule {
            name: "runaway_shader".to_string(),
            ops: vec![
                ReducerIrOp::KvPut {
                    key: "player:1:pos".to_string(),
                    value_template: "CORRUPTED".to_string(),
                },
                ReducerIrOp::ComputeLoop {
                    iterations: 10_000_000,
                },
            ],
        })
        .unwrap();

        let err = rt
            .invoke("game-world-1", "runaway_shader", "")
            .expect_err("runaway loop must hit deterministic fuel cap");
        assert!(matches!(err, ReducerError::FuelExhausted { .. }));

        // Verify atomic rollback: `player:1:pos` is STILL `"10,24"`, NOT `"CORRUPTED"`!
        assert_eq!(
            rt.get_committed("game-world-1", "player:1:pos")
                .unwrap()
                .unwrap(),
            b"10,24"
        );
    }
}
