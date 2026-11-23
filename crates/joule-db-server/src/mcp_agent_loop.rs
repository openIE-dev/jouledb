//! Wave 17 Frontier 3: Database-Native `MCP` Host & Autonomous Multi-Turn SQL / `JouleWard` /
//! `Titans` Agent Loop (`POST /api/v1/mcp/host/servers`, `GET /api/v1/mcp/host/tools`,
//! `POST /api/v1/mcp/host/call`, `POST /api/v1/agent/run`).
//!
//! Disintermediates standalone client-side agent frameworks (`LangGraph`, `CrewAI`, `PydanticAI`,
//! `Claude Desktop` MCP host) by executing multi-turn grammar-constrained tool selection, live
//! SQL / Streamable-HTTP MCP `JSON-RPC 2.0` dispatch, `JouleWard` `ThermoCap` capability
//! enforcement, content-addressed `JouleHarness` observation distillation, and `Titans` + `Engram`
//! MVCC memory commits inside a single server-side transaction.

use crate::{
    pulse_mesh_routes::PulseMeshRuntime,
    ward_openai_routes::{WardOpenAiRuntime, energy_headers, run_thermodynamic_cascade},
};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
};
use joule_db_branch::{
    pulse_flow::inspect_guardrail,
    ward::{
        FabricHopEnergyReceipt, FabricTraceReceipt, WardActionContext, WardCapabilityToken,
        WardDecision, authorize_ward_action,
    },
};
use joule_db_weights::grammar_constrain::ToolSpec;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const MAX_REMOTE_MCP_SERVERS: usize = 64;
const MAX_AUTONOMOUS_TURNS: usize = 8;

/// Registered external Streamable-HTTP MCP server (`JSON-RPC 2.0` `2025-06-18` / `2024-11-05`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredMcpServer {
    pub server_id: String,
    pub endpoint_url: String,
    #[serde(skip_serializing)]
    pub bearer_token: Option<String>,
    pub tools: Vec<McpHostedToolEntry>,
}

/// Unified tool descriptor exposed by JouleDB's Database-Native `MCP` Host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpHostedToolEntry {
    pub qualified_name: String,
    pub server_id: String,
    pub raw_name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Thread-safe registry of built-in native JouleDB tools + mounted remote Streamable-HTTP MCP servers.
#[derive(Debug, Default)]
pub struct McpHostRegistry {
    remote_servers: Mutex<BTreeMap<String, RegisteredMcpServer>>,
}

impl McpHostRegistry {
    pub fn new() -> Self {
        Self {
            remote_servers: Mutex::new(BTreeMap::new()),
        }
    }

    /// Built-in zero-copy database/synapse tools always mounted on `server_id = "native"`.
    pub fn builtin_tools() -> Vec<McpHostedToolEntry> {
        vec![
            McpHostedToolEntry {
                qualified_name: "jouledb.sql_query".to_string(),
                server_id: "native".to_string(),
                raw_name: "sql_query".to_string(),
                description: "Execute read or write SQL queries against JouleDB tables and return structured JSON rows".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "sql": { "type": "string" },
                        "limit": { "type": "integer" }
                    },
                    "required": ["sql"]
                }),
            },
            McpHostedToolEntry {
                qualified_name: "jouledb.engram_lookup".to_string(),
                server_id: "native".to_string(),
                raw_name: "engram_lookup".to_string(),
                description: "Lookup O(1) conditional factual memory slot in JouleSynapse Engram table".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "phrase": { "type": "string" }
                    },
                    "required": ["phrase"]
                }),
            },
            McpHostedToolEntry {
                qualified_name: "jouledb.engram_upsert".to_string(),
                server_id: "native".to_string(),
                raw_name: "engram_upsert".to_string(),
                description: "Store or update a grounded factual key-value slot in JouleSynapse Engram + Titans memory".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "phrase": { "type": "string" },
                        "fact_payload": { "type": "string" }
                    },
                    "required": ["phrase", "fact_payload"]
                }),
            },
            McpHostedToolEntry {
                qualified_name: "jouledb.synapse_status".to_string(),
                server_id: "native".to_string(),
                raw_name: "synapse_status".to_string(),
                description: "Inspect active JouleSynapse sub-1-bit models, Titans LSN, and Mooncake KV cache metrics".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "include_models": { "type": "boolean" }
                    },
                    "required": ["include_models"]
                }),
            },
        ]
    }

    pub fn upsert_remote_server(&self, server: RegisteredMcpServer) {
        let mut guard = match self.remote_servers.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if !guard.contains_key(&server.server_id)
            && guard.len() >= MAX_REMOTE_MCP_SERVERS
            && let Some(oldest) = guard.keys().next().cloned()
        {
            guard.remove(&oldest);
        }
        guard.insert(server.server_id.clone(), server);
    }

    pub fn list_all_tools(&self) -> Vec<McpHostedToolEntry> {
        let mut all = Self::builtin_tools();
        let guard = match self.remote_servers.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        for srv in guard.values() {
            all.extend(srv.tools.iter().cloned());
        }
        all
    }

    pub fn list_remote_servers(&self) -> Vec<RegisteredMcpServer> {
        let guard = match self.remote_servers.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        guard.values().cloned().collect()
    }

    pub fn resolve_tool(&self, requested_name: &str) -> Option<(McpHostedToolEntry, Option<RegisteredMcpServer>)> {
        for bt in Self::builtin_tools() {
            if bt.qualified_name == requested_name || bt.raw_name == requested_name {
                return Some((bt, None));
            }
        }
        let guard = match self.remote_servers.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        for srv in guard.values() {
            for t in &srv.tools {
                if t.qualified_name == requested_name || t.raw_name == requested_name {
                    return Some((t.clone(), Some(srv.clone())));
                }
            }
        }
        None
    }
}

// ============================================================================
// HTTP DTOs
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct RegisterMcpServerReq {
    pub server_id: String,
    pub endpoint_url: String,
    #[serde(default)]
    pub bearer_token: Option<String>,
    /// Optional static tool schemas when registering an offline/air-gapped server without live probe.
    #[serde(default)]
    pub static_tools: Vec<StaticMcpToolDef>,
    #[serde(default = "default_discover_live")]
    pub discover_live: bool,
}

fn default_discover_live() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct StaticMcpToolDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct CallMcpHostToolReq {
    pub tool_name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
    #[serde(default)]
    pub x_ward_token: Option<WardCapabilityToken>,
    #[serde(default)]
    pub x_hitl_approved: bool,
}

#[derive(Debug, Deserialize)]
pub struct AutonomousAgentRunReq {
    pub goal: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    #[serde(default = "default_commit_titans")]
    pub commit_to_titans: bool,
    #[serde(default)]
    pub x_ward_token: Option<WardCapabilityToken>,
    #[serde(default)]
    pub x_hitl_approved: bool,
}

fn default_max_turns() -> usize {
    2
}

fn default_commit_titans() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentTurnStepReceipt {
    pub turn_index: usize,
    pub selected_tool: String,
    pub server_id: String,
    pub arguments: serde_json::Value,
    pub output: serde_json::Value,
    pub observation_cas_sha256: String,
    pub step_energy_nj: u64,
    pub elapsed_ns: u64,
}

// ============================================================================
// Helpers: SQL Table Extraction, Guardrail + ThermoCap Enforcement, Tool Execution
// ============================================================================

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn extract_egress_host(url: &str) -> Option<String> {
    let after_scheme = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let host_port = after_scheme.split('/').next()?;
    let host = host_port.split(':').next()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

fn extract_sql_tables(sql: &str) -> Vec<String> {
    let mut tables = Vec::new();
    let tokens: Vec<&str> = sql.split_whitespace().collect();
    for i in 0..tokens.len() {
        let kw = tokens[i].to_ascii_uppercase();
        if (kw == "FROM" || kw == "INTO" || kw == "UPDATE" || kw == "JOIN" || kw == "TABLE")
            && let Some(&next_tok) = tokens.get(i + 1)
        {
            let upper_next = next_tok.to_ascii_uppercase();
            if upper_next == "IF" || upper_next == "EXISTS" || upper_next == "NOT" {
                if let Some(&after_not) = tokens.get(i + 3) {
                    let clean = after_not
                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                        .to_string();
                    if !clean.is_empty() && !tables.contains(&clean) {
                        tables.push(clean);
                    }
                }
                continue;
            }
            let clean = next_tok
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                .to_string();
            if !clean.is_empty() && !tables.contains(&clean) {
                tables.push(clean);
            }
        }
    }
    tables
}

fn extract_inline_sql_from_goal(goal: &str) -> Option<String> {
    let upper = goal.to_ascii_uppercase();
    for kw in ["SELECT ", "INSERT INTO ", "CREATE TABLE ", "UPDATE ", "DELETE FROM "] {
        if let Some(start_idx) = upper.find(kw) {
            let tail = &goal[start_idx..];
            let stmt = match tail.find(';') {
                Some(semi) => tail[..semi].trim(),
                None => tail.trim(),
            };
            if !stmt.is_empty() {
                return Some(stmt.to_string());
            }
        }
    }
    None
}

async fn execute_hosted_tool_checked(
    ward_rt: &Arc<WardOpenAiRuntime>,
    entry: &McpHostedToolEntry,
    remote_srv: Option<&RegisteredMcpServer>,
    arguments: &serde_json::Value,
    cap_token: Option<&WardCapabilityToken>,
    hitl_approved: bool,
) -> Result<(serde_json::Value, FabricTraceReceipt), (StatusCode, String)> {
    let t0 = Instant::now();
    let args_str = serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string());

    // 1. Inline 45 nJ JoulePulse Guardrail Check
    let g_res = inspect_guardrail(&args_str);
    if !g_res.allowed {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("JoulePulse guardrail blocked MCP tool '{}': {:?}", entry.qualified_name, g_res.triggered_rules),
        ));
    }

    // 2. JouleWard ThermoCap Capability Check (SQL tables, weight mutation, remote egress, cumulative nJ!)
    let requested_sql_tables = if entry.qualified_name == "jouledb.sql_query" {
        arguments
            .get("sql")
            .and_then(|v| v.as_str())
            .map(extract_sql_tables)
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mutates_weights = entry.qualified_name == "jouledb.engram_upsert";
    let egress_host = remote_srv.and_then(|s| extract_egress_host(&s.endpoint_url));

    let (est_rails, _est_src, _hw_fused) = ward_rt.estimate_rails_fused(
        if remote_srv.is_some() {
            "mcp_streamable_http_rpc"
        } else {
            "mcp_native_in_process"
        },
        350_000,
        32 * 1024,
        0,
        if remote_srv.is_some() { 1024 } else { 0 },
    );

    if let Some(cap) = cap_token {
        let ctx = WardActionContext {
            now_epoch_ms: now_ms(),
            target_fabric_class: ward_rt.ward.fabric_class(),
            silicon_lane: "mcp_tool_dispatch".into(),
            requested_paths: Vec::new(),
            requires_fs_write: false,
            requested_egress_host: egress_host,
            requested_sql_tables,
            mutates_synapse_weights: mutates_weights,
            estimated_rails: est_rails,
            hitl_approved,
        };
        let spent = ward_rt.ward.ledger.token_spent_nj(&cap.root_token_id);
        match authorize_ward_action(ward_rt.ward.root_secret(), cap, &ctx, spent) {
            WardDecision::Permit { .. } => {}
            WardDecision::RequiresHumanApproval { reason, .. } => {
                return Err((StatusCode::PRECONDITION_REQUIRED, reason));
            }
            WardDecision::Deny { reason } => {
                return Err((StatusCode::FORBIDDEN, reason));
            }
        }
    }

    // 3. Execute Native Zero-Copy Tool OR Remote Streamable-HTTP MCP `tools/call`
    let mut wire_bytes = 0u64;
    let output_val = if let Some(srv) = remote_srv {
        let rpc_req = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": entry.raw_name,
                "arguments": arguments,
            }
        });
        let req_bytes = serde_json::to_vec(&rpc_req).unwrap_or_default();
        wire_bytes += req_bytes.len() as u64;
        let mut builder = ward_rt
            .fabric_rt
            .http_client
            .post(&srv.endpoint_url)
            .header("content-type", "application/json")
            .body(req_bytes);
        if let Some(ref tok) = srv.bearer_token {
            builder = builder.bearer_auth(tok);
        }
        let resp = builder
            .send()
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("remote MCP call failed: {e}")))?;
        let status = resp.status();
        let raw = resp
            .bytes()
            .await
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("remote MCP read failed: {e}")))?;
        wire_bytes += raw.len() as u64;
        if !status.is_success() {
            return Err((
                StatusCode::BAD_GATEWAY,
                format!("remote MCP HTTP {status}: {}", String::from_utf8_lossy(&raw)),
            ));
        }
        let parsed: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| (StatusCode::BAD_GATEWAY, format!("invalid remote MCP JSON-RPC: {e}")))?;
        if let Some(err_obj) = parsed.get("error") {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("remote MCP JSON-RPC error: {err_obj}"),
            ));
        }
        parsed
            .get("result")
            .cloned()
            .unwrap_or(parsed)
    } else {
        match entry.qualified_name.as_str() {
            "jouledb.sql_query" => {
                let sql = arguments
                    .get("sql")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| (StatusCode::BAD_REQUEST, "missing 'sql' string argument".to_string()))?;
                let limit = arguments
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .map(|n| (n as usize).clamp(1, 10_000));
                ward_rt
                    .synapse_rt
                    .execute_sql_json(sql, limit)
                    .map_err(|e| (StatusCode::BAD_REQUEST, e))?
            }
            "jouledb.engram_lookup" => {
                let phrase = arguments
                    .get("phrase")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let hit = ward_rt.synapse_rt.lookup_engram_fact(phrase);
                serde_json::json!({
                    "phrase": phrase,
                    "hit": hit.is_some(),
                    "fact_payload": hit,
                })
            }
            "jouledb.engram_upsert" => {
                let phrase = arguments
                    .get("phrase")
                    .and_then(|v| v.as_str())
                    .unwrap_or("fact");
                let payload = arguments
                    .get("fact_payload")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                ward_rt.synapse_rt.commit_agent_observation_and_titans(
                    "mcp:engram_upsert",
                    payload,
                    phrase,
                    payload,
                    true,
                )
            }
            "jouledb.synapse_status" => {
                let models = ward_rt.synapse_rt.list_living_models();
                serde_json::json!({
                    "living_models": models,
                })
            }
            other => {
                return Err((
                    StatusCode::NOT_FOUND,
                    format!("unknown native MCP tool '{other}'"),
                ));
            }
        }
    };

    let elapsed_ns = (t0.elapsed().as_nanos() as u64).max(1);
    let out_str = serde_json::to_string(&output_val).unwrap_or_default();
    let (rails, cal_src, _hw_fused) = ward_rt.estimate_rails_fused(
        if remote_srv.is_some() {
            "mcp_streamable_http_rpc"
        } else {
            "mcp_native_in_process"
        },
        elapsed_ns,
        (args_str.len() + out_str.len()).max(4096) as u64,
        0,
        wire_bytes,
    );

    let mut trace = FabricTraceReceipt::default();
    trace.push_hop(FabricHopEnergyReceipt {
        node_id: ward_rt.ward.node_id().to_string(),
        fabric_class: ward_rt.ward.fabric_class(),
        silicon_lane: entry.qualified_name.clone(),
        calibration_source: cal_src,
        grid_carbon_g_per_kwh: ward_rt.ward.grid_carbon_g_per_kwh(),
        rails,
    });
    let root_tid = cap_token.map(|t| t.root_token_id.as_str());
    ward_rt.ward.ledger.record_trace(root_tid, &trace);

    Ok((output_val, trace))
}

// ============================================================================
// Route Handlers
// ============================================================================

pub async fn handle_mcp_register_server(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<RegisterMcpServerReq>,
) -> impl IntoResponse {
    if req.server_id.trim().is_empty() || req.endpoint_url.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "server_id and endpoint_url are required" })),
        );
    }

    let mut discovered_tools: Vec<McpHostedToolEntry> = req
        .static_tools
        .iter()
        .map(|st| McpHostedToolEntry {
            qualified_name: format!("{}.{}", req.server_id, st.name),
            server_id: req.server_id.clone(),
            raw_name: st.name.clone(),
            description: st.description.clone(),
            input_schema: if st.input_schema.is_null() {
                serde_json::json!({"type": "object", "properties": {}})
            } else {
                st.input_schema.clone()
            },
        })
        .collect();

    if req.discover_live {
        let list_rpc = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        });
        let mut builder = rt
            .ward_rt
            .fabric_rt
            .http_client
            .post(&req.endpoint_url)
            .header("content-type", "application/json")
            .json(&list_rpc);
        if let Some(ref tok) = req.bearer_token {
            builder = builder.bearer_auth(tok);
        }
        match builder.send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(val) = resp.json::<serde_json::Value>().await
                    && let Some(arr) = val
                        .get("result")
                        .and_then(|r| r.get("tools"))
                        .and_then(|t| t.as_array())
                {
                    discovered_tools.clear();
                    for item in arr {
                        if let Some(name) = item.get("name").and_then(|v| v.as_str()) {
                            let desc = item
                                .get("description")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            let schema = item
                                .get("inputSchema")
                                .or_else(|| item.get("input_schema"))
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
                            discovered_tools.push(McpHostedToolEntry {
                                qualified_name: format!("{}.{}", req.server_id, name),
                                server_id: req.server_id.clone(),
                                raw_name: name.to_string(),
                                description: desc,
                                input_schema: schema,
                            });
                        }
                    }
                }
            }
            Ok(resp) if discovered_tools.is_empty() => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "ok": false,
                        "error": format!("live MCP tools/list returned HTTP {}", resp.status())
                    })),
                );
            }
            Err(e) if discovered_tools.is_empty() => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "ok": false,
                        "error": format!("failed to reach MCP server '{}': {e}", req.endpoint_url)
                    })),
                );
            }
            _ => {}
        }
    }

    let tool_count = discovered_tools.len();
    let srv = RegisteredMcpServer {
        server_id: req.server_id.clone(),
        endpoint_url: req.endpoint_url.clone(),
        bearer_token: req.bearer_token,
        tools: discovered_tools.clone(),
    };
    rt.mcp_host.upsert_remote_server(srv);

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "server_id": req.server_id,
            "endpoint_url": req.endpoint_url,
            "discovered_tools_count": tool_count,
            "tools": discovered_tools,
        })),
    )
}

pub async fn handle_mcp_list_tools(
    State(rt): State<Arc<PulseMeshRuntime>>,
) -> impl IntoResponse {
    let tools = rt.mcp_host.list_all_tools();
    let servers = rt.mcp_host.list_remote_servers();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "total_tools": tools.len(),
            "remote_servers_count": servers.len(),
            "tools": tools,
        })),
    )
}

pub async fn handle_mcp_call_tool(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<CallMcpHostToolReq>,
) -> impl IntoResponse {
    let Some((entry, remote_opt)) = rt.mcp_host.resolve_tool(&req.tool_name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "ok": false,
                "error": format!("MCP tool '{}' is not registered", req.tool_name)
            })),
        )
            .into_response();
    };

    match execute_hosted_tool_checked(
        &rt.ward_rt,
        &entry,
        remote_opt.as_ref(),
        &req.arguments,
        req.x_ward_token.as_ref(),
        req.x_hitl_approved,
    )
    .await
    {
        Ok((result, trace)) => (
            StatusCode::OK,
            energy_headers(&trace, "mcp_host_dispatch"),
            Json(serde_json::json!({
                "ok": true,
                "tool": entry.qualified_name,
                "server_id": entry.server_id,
                "result": result,
                "x_jouledb_energy": {
                    "aggregate_rails": trace.aggregate_rails,
                    "trace": trace,
                }
            })),
        )
            .into_response(),
        Err((status, msg)) => (
            status,
            Json(serde_json::json!({ "ok": false, "error": msg })),
        )
            .into_response(),
    }
}

/// Execute an autonomous multi-turn tool/SQL/`JouleWard`/`Titans` agent loop inside a single request!
pub async fn handle_autonomous_agent_run(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<AutonomousAgentRunReq>,
) -> impl IntoResponse {
    let g_goal = inspect_guardrail(&req.goal);
    if !g_goal.allowed {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "ok": false,
                "error": "JoulePulse guardrail blocked unsafe autonomous goal",
                "guardrail": g_goal,
            })),
        )
            .into_response();
    }

    let max_turns = req.max_turns.clamp(1, MAX_AUTONOMOUS_TURNS);
    let all_tools = rt.mcp_host.list_all_tools();
    let candidate_tools: Vec<McpHostedToolEntry> = if req.allowed_tools.is_empty() {
        all_tools
    } else {
        all_tools
            .into_iter()
            .filter(|t| {
                req.allowed_tools
                    .iter()
                    .any(|a| a == &t.qualified_name || a == &t.raw_name)
            })
            .collect()
    };

    if candidate_tools.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": "no matching MCP tools available for agent run" })),
        )
            .into_response();
    }

    let tool_specs: Vec<ToolSpec> = candidate_tools
        .iter()
        .map(|t| ToolSpec {
            name: t.qualified_name.clone(),
            description: t.description.clone(),
            parameters_schema: t.input_schema.clone(),
        })
        .collect();

    let run_id = format!("agent-run-{:x}", now_ms());
    let mut combined_trace = FabricTraceReceipt::default();
    let mut steps: Vec<AgentTurnStepReceipt> = Vec::new();
    let mut halted_reason: Option<String> = None;
    let mut working_context = req.goal.clone();
    let inline_sql = extract_inline_sql_from_goal(&req.goal);

    for turn_idx in 0..max_turns {
        // On turn 0, if the goal explicitly includes a SQL statement and `jouledb.sql_query` is allowed,
        // execute that exact SQL statement first; otherwise use the zero-hallucination grammar selector!
        let (chosen_name, mut chosen_args) = if turn_idx == 0
            && let Some(ref sql_stmt) = inline_sql
            && candidate_tools.iter().any(|t| t.qualified_name == "jouledb.sql_query")
        {
            (
                "jouledb.sql_query".to_string(),
                serde_json::json!({ "sql": sql_stmt, "limit": 100 }),
            )
        } else if turn_idx > 0
            && req.commit_to_titans
            && candidate_tools.iter().any(|t| t.qualified_name == "jouledb.engram_upsert")
            && !steps.iter().any(|s| s.selected_tool == "jouledb.engram_upsert")
        {
            let prev_summary = steps
                .last()
                .map(|s| serde_json::to_string(&s.output).unwrap_or_default())
                .unwrap_or_default();
            (
                "jouledb.engram_upsert".to_string(),
                serde_json::json!({
                    "phrase": req.goal,
                    "fact_payload": prev_summary,
                }),
            )
        } else {
            let forced = if candidate_tools.len() == 1 {
                Some(candidate_tools[0].qualified_name.as_str())
            } else {
                None
            };
            match rt.ward_rt.synapse_rt.constrain_tool_call(
                req.model.as_deref(),
                &working_context,
                &tool_specs,
                forced,
            ) {
                Ok(tc) => (tc.name, tc.arguments_parsed),
                Err(e) => {
                    halted_reason = Some(format!("grammar tool planning error: {e}"));
                    break;
                }
            }
        };

        // If `jouledb.sql_query` was grammar-selected with a generic string, upgrade to inline SQL when present
        if chosen_name == "jouledb.sql_query"
            && let Some(ref sql_stmt) = inline_sql
            && let Some(obj) = chosen_args.as_object_mut()
        {
            obj.insert("sql".to_string(), serde_json::Value::String(sql_stmt.clone()));
        }

        let Some((entry, remote_opt)) = rt.mcp_host.resolve_tool(&chosen_name) else {
            halted_reason = Some(format!("selected tool '{chosen_name}' not found"));
            break;
        };

        match execute_hosted_tool_checked(
            &rt.ward_rt,
            &entry,
            remote_opt.as_ref(),
            &chosen_args,
            req.x_ward_token.as_ref(),
            req.x_hitl_approved,
        )
        .await
        {
            Ok((tool_out, step_trace)) => {
                let raw_out = serde_json::to_string(&tool_out).unwrap_or_default();
                let mem_receipt = rt.ward_rt.synapse_rt.commit_agent_observation_and_titans(
                    &format!("agent_turn_{turn_idx}:{}", entry.qualified_name),
                    &raw_out,
                    if turn_idx + 1 == max_turns { &req.goal } else { "" },
                    &raw_out,
                    req.commit_to_titans && (turn_idx + 1 == max_turns),
                );
                let cas_hash = mem_receipt
                    .get("observation")
                    .and_then(|o| o.get("cas_sha256"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let step_nj = step_trace.aggregate_rails.total_nj();
                let step_ns = step_trace.aggregate_rails.wall_duration_ns;
                let syn_lsn_after = mem_receipt
                    .get("synapse_lsn_after")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                rt.pulse.record_from_trace(
                    &run_id,
                    &format!("agent.mcp.{}", entry.qualified_name),
                    "main",
                    turn_idx as u64 + 1,
                    syn_lsn_after,
                    &entry.server_id,
                    &step_trace,
                );
                for hop in step_trace.hops {
                    combined_trace.push_hop(hop);
                }

                steps.push(AgentTurnStepReceipt {
                    turn_index: turn_idx,
                    selected_tool: entry.qualified_name.clone(),
                    server_id: entry.server_id.clone(),
                    arguments: chosen_args,
                    output: tool_out,
                    observation_cas_sha256: cas_hash,
                    step_energy_nj: step_nj,
                    elapsed_ns: step_ns,
                });

                working_context = format!("{working_context}\nObservation[{turn_idx}]: {raw_out}");

                // If single-step query with no further tools needed and commit_to_titans already satisfied, stop cleanly
                if max_turns == 1 || entry.qualified_name == "jouledb.engram_upsert" {
                    break;
                }
            }
            Err((_status, deny_msg)) => {
                halted_reason = Some(deny_msg);
                break;
            }
        }
    }

    // Final grounded synthesis (uses living cascade if within capability budget)
    let model_name = req.model.as_deref().unwrap_or("jouledb-cascade-v1");
    let final_answer = match run_thermodynamic_cascade(
        &rt.ward_rt,
        model_name,
        &working_context,
        req.x_ward_token.as_ref(),
        req.x_hitl_approved,
    ) {
        Ok(outcome) => {
            for hop in outcome.trace.hops {
                combined_trace.push_hop(hop);
            }
            outcome.reply_text
        }
        Err((_st, _kind, msg)) => {
            if halted_reason.is_none() {
                halted_reason = Some(msg);
            }
            steps
                .last()
                .map(|s| serde_json::to_string(&s.output).unwrap_or_default())
                .unwrap_or_else(|| "Agent halted by capability ceiling.".to_string())
        }
    };

    let stop_reason = if halted_reason.is_some() {
        "halted_by_ward_or_guardrail"
    } else {
        "completed"
    };

    (
        StatusCode::OK,
        energy_headers(&combined_trace, "autonomous_mcp_agent_loop"),
        Json(serde_json::json!({
            "ok": halted_reason.is_none() || !steps.is_empty(),
            "run_id": run_id,
            "stop_reason": stop_reason,
            "halted_reason": halted_reason,
            "turns_executed": steps.len(),
            "steps": steps,
            "final_answer": final_answer,
            "x_jouledb_energy": {
                "aggregate_rails": combined_trace.aggregate_rails,
                "total_nj": combined_trace.aggregate_rails.total_nj(),
                "ward_ledger_events": rt.ward_rt.ward.ledger.snapshot().total_events,
                "trace": combined_trace,
            }
        })),
    )
        .into_response()
}
