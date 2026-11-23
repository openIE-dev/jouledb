//! Wave 13 HTTP Surface:
//! 1. **Drop-In Anthropic Messages API (`POST /v1/messages`)**: Completes the OpenAI + Anthropic +
//!    Ollama disintermediation triad with inline `JoulePulse` `45 nJ` guardrail protection and
//!    high-resolution 6-rail nanojoule receipts (`x_jouledb_energy`).
//! 2. **`JoulePulse` Observability, Guardrails & Pareto Eval Gate** (`POST /v1/traces` OTLP,
//!    `GET /api/v1/pulse/spans`, `POST /api/v1/pulse/guardrail`, `POST /api/v1/pulse/eval`):
//!    Disintermediates `LangSmith`, `Arize Phoenix`, `Braintrust`, `W&B Weave`, & `Guardrails AI`.
//! 3. **`JouleFlow` Durable Thermodynamic DAGs** (`POST /api/v1/flow/execute`,
//!    `GET /api/v1/flow/runs/{run_id}`): Disintermediates `Temporal`, `Inngest`, `Trigger.dev`,
//!    and `Airflow` with crash-idempotent step journals, carbon-window deferral, & Saga rollbacks.
//! 4. **`JouleSynapseMesh` Federated Living-Weight Sync** (`POST /api/v1/synapse/mesh/export-delta`,
//!    `POST /api/v1/synapse/mesh/merge`): Disintermediates `Flower`, `Ray` parameter servers, and
//!    multi-GB LoRA sync via Multi-Krum Byzantine-resilient `< 4 KB` `SVF` + `Engram` deltas.

use crate::ward_openai_routes::{WardOpenAiRuntime, energy_headers, run_thermodynamic_cascade};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::{HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use joule_db_branch::{
    pulse_flow::{
        CausalPulseSpan, DurableFlowEngine, DurableFlowSpec, EvalSuiteSummary, PulseEngine,
        evaluate_pareto_gate, inspect_guardrail,
    },
    ward::{FabricHopEnergyReceipt, FabricTraceReceipt, HeterogeneousRailSample, WardCapabilityToken},
};
use joule_db_weights::{federated_mesh::SynapseSkillDelta, grammar_constrain::ToolSpec};
use serde::Deserialize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared server state for `JoulePulse`, `JouleFlow`, `JouleSynapseMesh`, `/v1/messages`, and `MCP` Host.
pub struct PulseMeshRuntime {
    pub pulse: PulseEngine,
    pub flow: DurableFlowEngine,
    pub mcp_host: crate::mcp_agent_loop::McpHostRegistry,
    pub ward_rt: Arc<WardOpenAiRuntime>,
}

impl PulseMeshRuntime {
    pub fn new(ward_rt: Arc<WardOpenAiRuntime>) -> Arc<Self> {
        Arc::new(Self {
            pulse: PulseEngine::new(),
            flow: DurableFlowEngine::new(),
            mcp_host: crate::mcp_agent_loop::McpHostRegistry::new(),
            ward_rt,
        })
    }
}

// ============================================================================
// DTOs — Anthropic `/v1/messages`, `JoulePulse`, `JouleFlow`, `JouleSynapseMesh`
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Deserialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesReq {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub x_ward_token: Option<WardCapabilityToken>,
    #[serde(default)]
    pub x_hitl_approved: bool,
}

fn default_model() -> String {
    "jouledb-cascade-v1".to_string()
}

fn default_max_tokens() -> usize {
    512
}

#[derive(Debug, Deserialize)]
pub struct GuardrailCheckReq {
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct ParetoEvalReq {
    pub baseline: EvalSuiteSummary,
    pub candidate: EvalSuiteSummary,
    #[serde(default = "default_max_acc_drop")]
    pub max_accuracy_drop: f64,
    #[serde(default = "default_max_energy_ratio")]
    pub max_energy_regression_ratio: f64,
    #[serde(default)]
    pub power_cap_mw: Option<u64>,
}

fn default_max_acc_drop() -> f64 {
    0.005
}

fn default_max_energy_ratio() -> f64 {
    1.10
}

#[derive(Debug, Deserialize)]
pub struct FlowExecHttpReq {
    #[serde(flatten)]
    pub spec: DurableFlowSpec,
    /// Optional override for current grid carbon intensity (`gCO2/kWh`) in scheduling simulations/tests.
    #[serde(default)]
    pub grid_carbon_override_g_kwh: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ExportMeshDeltaReq {
    pub delta_id: String,
    pub expert_name: String,
    #[serde(default = "default_clip_l2")]
    pub clip_l2_norm: f32,
    #[serde(default)]
    pub dp_noise_sigma: f32,
    #[serde(default)]
    pub engram_facts: Vec<(String, String)>,
    #[serde(default = "default_train_nj")]
    pub training_energy_nj: u64,
    #[serde(default = "default_peak_mw")]
    pub peak_power_mw: u64,
}

fn default_clip_l2() -> f32 {
    4.0
}

fn default_train_nj() -> u64 {
    250_000
}

fn default_peak_mw() -> u64 {
    4_800
}

#[derive(Debug, Deserialize)]
pub struct MergeMeshDeltasReq {
    pub expert_name: String,
    pub deltas: Vec<SynapseSkillDelta>,
    #[serde(default = "default_max_byz")]
    pub max_byzantine_peers: usize,
    #[serde(default = "default_clip_l2")]
    pub max_allowed_l2: f32,
    /// Optional peer URLs to pull/broadcast across the compute fabric.
    #[serde(default)]
    pub broadcast_to_fabric_peers: bool,
}

fn default_max_byz() -> usize {
    1
}

/// Build Axum router for Wave 13 (`/v1/messages`, `/v1/traces`, `/api/v1/pulse/*`, `/api/v1/flow/*`, `/api/v1/synapse/mesh/*`).
pub fn pulse_mesh_router(rt: Arc<PulseMeshRuntime>) -> Router {
    Router::new()
        .route("/v1/messages", post(handle_anthropic_messages))
        .route("/v1/traces", post(handle_otlp_traces))
        .route("/api/v1/pulse/spans", get(handle_pulse_spans))
        .route("/api/v1/pulse/guardrail", post(handle_pulse_guardrail))
        .route("/api/v1/pulse/eval", post(handle_pulse_eval))
        .route("/api/v1/flow/execute", post(handle_flow_execute))
        .route("/api/v1/flow/runs/{run_id}", get(handle_flow_get_run))
        .route(
            "/api/v1/synapse/mesh/export-delta",
            post(handle_mesh_export_delta),
        )
        .route("/api/v1/synapse/mesh/merge", post(handle_mesh_merge))
        .route(
            "/api/v1/mcp/host/servers",
            post(crate::mcp_agent_loop::handle_mcp_register_server),
        )
        .route(
            "/api/v1/mcp/host/tools",
            get(crate::mcp_agent_loop::handle_mcp_list_tools),
        )
        .route(
            "/api/v1/mcp/host/call",
            post(crate::mcp_agent_loop::handle_mcp_call_tool),
        )
        .route(
            "/api/v1/agent/run",
            post(crate::mcp_agent_loop::handle_autonomous_agent_run),
        )
        .with_state(rt)
}

// ============================================================================
// Handlers
// ============================================================================

async fn handle_anthropic_messages(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<AnthropicMessagesReq>,
) -> impl IntoResponse {
    let raw_prompt = extract_anthropic_prompt(&req);
    let guardrail = inspect_guardrail(&raw_prompt);
    if !guardrail.allowed {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": "guardrail_violation",
                    "message": "Blocked by JoulePulse inline guardrail firewall",
                    "triggered_rules": guardrail.triggered_rules,
                    "scan_cost_nj": guardrail.scan_cost_nj,
                }
            })),
        )
            .into_response();
    }

    let _max_tok = req.max_tokens;
    let in_tokens = guardrail.sanitized_text.split_whitespace().count().max(1);
    let hint = if req.model == "jouledb-cascade-v1" {
        None
    } else {
        Some(req.model.as_str())
    };

    // Anthropic Native Tool-Use Grammar Constraining (`tools` provided)
    if !req.tools.is_empty() {
        let specs: Vec<ToolSpec> = req
            .tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name").and_then(|v| v.as_str())?;
                let desc = t
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let schema = t
                    .get("input_schema")
                    .or_else(|| t.get("parameters"))
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                Some(ToolSpec {
                    name: name.to_string(),
                    description: desc,
                    parameters_schema: schema,
                })
            })
            .collect();
        let forced = req
            .tool_choice
            .as_ref()
            .and_then(|tc| tc.get("name"))
            .and_then(|n| n.as_str());

        if !specs.is_empty() {
            match rt
                .ward_rt
                .synapse_rt
                .constrain_tool_call(hint, &guardrail.sanitized_text, &specs, forced)
            {
                Ok(tc) => {
                    let (mut rails, cal_src, hw_fused) = rt.ward_rt.estimate_rails_fused(
                        "xgrammar_jump_forward_dfa",
                        tc.receipt.elapsed_ns,
                        16 * 1024,
                        0,
                        0,
                    );
                    rails.cpu_nj = rails
                        .cpu_nj
                        .saturating_add(guardrail.scan_cost_nj)
                        .saturating_add(tc.receipt.mask_eval_cost_nj);
                    let mut trace = FabricTraceReceipt::default();
                    trace.push_hop(FabricHopEnergyReceipt {
                        node_id: rt.ward_rt.ward.node_id().to_string(),
                        fabric_class: rt.ward_rt.ward.fabric_class(),
                        silicon_lane: "xgrammar_jump_forward_dfa".to_string(),
                        calibration_source: cal_src,
                        grid_carbon_g_per_kwh: rt.ward_rt.ward.grid_carbon_g_per_kwh(),
                        rails,
                    });
                    let ts = now_ms();
                    let msg_id = format!("msg_jdb_{ts}");
                    let trace_id = format!("tr-msg-{ts}");
                    let syn_lsn = rt.ward_rt.synapse_rt.current_synapse_lsn();
                    rt.pulse.record_from_trace(
                        &trace_id,
                        "anthropic.messages.tool_use",
                        "main",
                        0,
                        syn_lsn,
                        "tier4_xgrammar_constrained_tool_call",
                        &trace,
                    );
                    let energy_meta = serde_json::json!({
                        "trace_id": trace_id,
                        "guardrail_rules_fired": guardrail.triggered_rules,
                        "guardrail_scan_nj": guardrail.scan_cost_nj,
                        "cascade_tier": "tier4_xgrammar_constrained_tool_call",
                        "kernel": "xgrammar_jump_forward_dfa",
                        "hardware_sensor_fused": hw_fused,
                        "aggregate_rails": trace.aggregate_rails,
                        "trace": trace,
                    });

                    if req.stream {
                        let mut hdrs = energy_headers(&trace, "tier4_xgrammar_constrained_tool_call");
                        hdrs.insert("content-type", HeaderValue::from_static("text/event-stream"));
                        hdrs.insert("cache-control", HeaderValue::from_static("no-cache"));
                        let sse = build_anthropic_tool_sse(
                            &msg_id,
                            &req.model,
                            &tc.call_id,
                            &tc.name,
                            &tc.arguments_json,
                            in_tokens,
                            trace.aggregate_rails.total_nj(),
                            &energy_meta,
                        );
                        return (StatusCode::OK, hdrs, sse).into_response();
                    }

                    return (
                        StatusCode::OK,
                        energy_headers(&trace, "tier4_xgrammar_constrained_tool_call"),
                        Json(serde_json::json!({
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": req.model,
                            "content": [{
                                "type": "tool_use",
                                "id": tc.call_id,
                                "name": tc.name,
                                "input": tc.arguments_parsed,
                            }],
                            "stop_reason": "tool_use",
                            "usage": {
                                "input_tokens": in_tokens,
                                "output_tokens": 8,
                            },
                            "x_jouledb_energy": energy_meta,
                        })),
                    )
                        .into_response();
                }
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({
                            "type": "error",
                            "error": { "type": "invalid_request_error", "message": e }
                        })),
                    )
                        .into_response();
                }
            }
        }
    }

    match run_thermodynamic_cascade(
        &rt.ward_rt,
        &req.model,
        &guardrail.sanitized_text,
        req.x_ward_token.as_ref(),
        req.x_hitl_approved,
    ) {
        Ok(mut outcome) => {
            // Charge the 45 nJ inline guardrail scan onto cpu_nj
            outcome.trace.aggregate_rails.cpu_nj = outcome
                .trace
                .aggregate_rails
                .cpu_nj
                .saturating_add(guardrail.scan_cost_nj);

            let ts = now_ms();
            let msg_id = format!("msg_jdb_{ts}");
            let trace_id = format!("tr-msg-{ts}");
            let syn_lsn = rt.ward_rt.synapse_rt.current_synapse_lsn();
            rt.pulse.record_from_trace(
                &trace_id,
                "anthropic.messages",
                "main",
                0,
                syn_lsn,
                &outcome.cascade_tier,
                &outcome.trace,
            );

            let out_tokens = outcome.reply_text.split_whitespace().count().max(1);
            let energy_meta = serde_json::json!({
                "trace_id": trace_id,
                "guardrail_rules_fired": guardrail.triggered_rules,
                "guardrail_scan_nj": guardrail.scan_cost_nj,
                "cascade_tier": outcome.cascade_tier,
                "kernel": outcome.kernel,
                "device": outcome.device,
                "bits_per_weight": outcome.bits_per_weight,
                "aggregate_rails": outcome.trace.aggregate_rails,
                "saved_by_memoization_nj": outcome.trace.saved_by_memoization_nj,
                "trace": outcome.trace,
            });

            if req.stream {
                let mut hdrs = energy_headers(&outcome.trace, &outcome.cascade_tier);
                hdrs.insert("content-type", HeaderValue::from_static("text/event-stream"));
                hdrs.insert("cache-control", HeaderValue::from_static("no-cache"));
                let sse = build_anthropic_text_sse(
                    &msg_id,
                    &outcome.resolved_model,
                    &outcome.reply_text,
                    in_tokens,
                    out_tokens,
                    outcome.trace.aggregate_rails.total_nj(),
                    &energy_meta,
                );
                return (StatusCode::OK, hdrs, sse).into_response();
            }

            (
                StatusCode::OK,
                energy_headers(&outcome.trace, &outcome.cascade_tier),
                Json(serde_json::json!({
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": outcome.resolved_model,
                    "content": [{
                        "type": "text",
                        "text": outcome.reply_text,
                    }],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": in_tokens,
                        "output_tokens": out_tokens,
                    },
                    "x_jouledb_energy": energy_meta,
                })),
            )
                .into_response()
        }
        Err((status, err_type, msg)) => (
            status,
            Json(serde_json::json!({
                "type": "error",
                "error": {
                    "type": err_type,
                    "message": msg,
                }
            })),
        )
            .into_response(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_anthropic_text_sse(
    msg_id: &str,
    model: &str,
    full_text: &str,
    in_tokens: usize,
    out_tokens: usize,
    total_nj: u64,
    energy_meta: &serde_json::Value,
) -> String {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for ch in full_text.chars() {
        cur.push(ch);
        if ch.is_whitespace() && cur.len() >= 4 {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() || chunks.is_empty() {
        chunks.push(cur);
    }

    let n = chunks.len().max(1) as u64;
    let per_tok_nj = (total_nj / n).max(1);
    let mut buf = String::with_capacity(full_text.len() + 640);

    let m_start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": serde_json::Value::Null,
            "usage": { "input_tokens": in_tokens, "output_tokens": 0 }
        }
    });
    buf.push_str(&format!("event: message_start\ndata: {m_start}\n\n"));

    let cb_start = serde_json::json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "text", "text": "" }
    });
    buf.push_str(&format!("event: content_block_start\ndata: {cb_start}\n\n"));

    let mut cum_nj = 0u64;
    for (idx, piece) in chunks.iter().enumerate() {
        let tok_nj = if idx + 1 == chunks.len() {
            total_nj.saturating_sub(cum_nj).max(1)
        } else {
            per_tok_nj
        };
        cum_nj = cum_nj.saturating_add(tok_nj);
        let delta = serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": { "type": "text_delta", "text": piece },
            "x_jouledb_token_nj": tok_nj,
            "x_jouledb_cumulative_nj": cum_nj,
        });
        buf.push_str(&format!("event: content_block_delta\ndata: {delta}\n\n"));
    }

    buf.push_str("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");
    let m_delta = serde_json::json!({
        "type": "message_delta",
        "delta": { "stop_reason": "end_turn" },
        "usage": { "output_tokens": out_tokens },
        "x_jouledb_energy": energy_meta,
    });
    buf.push_str(&format!("event: message_delta\ndata: {m_delta}\n\n"));
    buf.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    buf
}

#[allow(clippy::too_many_arguments)]
fn build_anthropic_tool_sse(
    msg_id: &str,
    model: &str,
    tool_id: &str,
    tool_name: &str,
    args_json: &str,
    in_tokens: usize,
    total_nj: u64,
    energy_meta: &serde_json::Value,
) -> String {
    let mut buf = String::with_capacity(args_json.len() + 640);
    let m_start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": serde_json::Value::Null,
            "usage": { "input_tokens": in_tokens, "output_tokens": 0 }
        }
    });
    buf.push_str(&format!("event: message_start\ndata: {m_start}\n\n"));

    let cb_start = serde_json::json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": { "type": "tool_use", "id": tool_id, "name": tool_name, "input": {} }
    });
    buf.push_str(&format!("event: content_block_start\ndata: {cb_start}\n\n"));

    let cb_delta = serde_json::json!({
        "type": "content_block_delta",
        "index": 0,
        "delta": { "type": "input_json_delta", "partial_json": args_json },
        "x_jouledb_token_nj": total_nj.max(1),
    });
    buf.push_str(&format!("event: content_block_delta\ndata: {cb_delta}\n\n"));
    buf.push_str("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n");

    let m_delta = serde_json::json!({
        "type": "message_delta",
        "delta": { "stop_reason": "tool_use" },
        "usage": { "output_tokens": 8 },
        "x_jouledb_energy": energy_meta,
    });
    buf.push_str(&format!("event: message_delta\ndata: {m_delta}\n\n"));
    buf.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");
    buf
}

async fn handle_otlp_traces(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    // Supports standard OpenTelemetry JSON (`resourceSpans -> scopeSpans -> spans`) AND direct span arrays
    let mut ingested = 0usize;
    let syn_lsn = rt.ward_rt.synapse_rt.current_synapse_lsn();

    if let Some(res_spans) = payload.get("resourceSpans").and_then(|v| v.as_array()) {
        for rs in res_spans {
            if let Some(scope_spans) = rs.get("scopeSpans").and_then(|v| v.as_array()) {
                for ss in scope_spans {
                    if let Some(spans) = ss.get("spans").and_then(|v| v.as_array()) {
                        for sp in spans {
                            let tid = sp
                                .get("traceId")
                                .and_then(|v| v.as_str())
                                .unwrap_or("otlp-trace")
                                .to_string();
                            let sid = sp
                                .get("spanId")
                                .and_then(|v| v.as_str())
                                .unwrap_or("otlp-span")
                                .to_string();
                            let name = sp
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("otlp.operation")
                                .to_string();
                            let start_ns = sp
                                .get("startTimeUnixNano")
                                .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
                                .unwrap_or(0);
                            let end_ns = sp
                                .get("endTimeUnixNano")
                                .and_then(|v| v.as_u64().or_else(|| v.as_str()?.parse().ok()))
                                .unwrap_or(start_ns + 50_000);
                            let dur_ns = end_ns.saturating_sub(start_ns).max(1_000);
                            let rails = HeterogeneousRailSample::estimate_calibrated(
                                rt.ward_rt.ward.fabric_class(),
                                "cpu_simd",
                                dur_ns,
                                4096,
                                0,
                                0,
                                rt.ward_rt.ward.grid_carbon_g_per_kwh(),
                            );
                            rt.pulse.record_span(CausalPulseSpan {
                                trace_id: tid,
                                span_id: sid,
                                operation_name: name,
                                branch_id: "main".into(),
                                db_lsn: 0,
                                synapse_lsn: syn_lsn,
                                cascade_tier: "otlp_ingested".into(),
                                status_ok: true,
                                rails,
                            });
                            ingested += 1;
                        }
                    }
                }
            }
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "partialSuccess": {},
            "x_jouledb_ingested_spans": ingested,
            "retained_spans": rt.pulse.span_count(),
        })),
    )
}

async fn handle_pulse_spans(State(rt): State<Arc<PulseMeshRuntime>>) -> impl IntoResponse {
    let spans = rt.pulse.recent_spans(100);
    let tiers = rt.pulse.observed_tiers();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "retained_span_count": rt.pulse.span_count(),
            "observed_cascade_tiers": tiers,
            "spans": spans,
        })),
    )
}

async fn handle_pulse_guardrail(Json(req): Json<GuardrailCheckReq>) -> impl IntoResponse {
    let verdict = inspect_guardrail(&req.text);
    let code = if verdict.allowed {
        StatusCode::OK
    } else {
        StatusCode::FORBIDDEN
    };
    (code, Json(verdict))
}

async fn handle_pulse_eval(Json(req): Json<ParetoEvalReq>) -> impl IntoResponse {
    let verdict = evaluate_pareto_gate(
        req.baseline,
        req.candidate,
        req.max_accuracy_drop,
        req.max_energy_regression_ratio,
        req.power_cap_mw,
    );
    (StatusCode::OK, Json(verdict))
}

async fn handle_flow_execute(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<FlowExecHttpReq>,
) -> impl IntoResponse {
    let carbon = req
        .grid_carbon_override_g_kwh
        .unwrap_or_else(|| rt.ward_rt.ward.grid_carbon_g_per_kwh());
    let fclass = rt.ward_rt.ward.fabric_class();

    match rt.flow.execute_or_resume(&req.spec, fclass, carbon) {
        Ok(receipt) => {
            let mut trace = FabricTraceReceipt {
                hops: Vec::new(),
                aggregate_rails: HeterogeneousRailSample::default(),
                per_class_totals_nj: Default::default(),
                saved_by_memoization_nj: receipt.saved_by_journal_replay_nj,
            };
            trace.push_hop(FabricHopEnergyReceipt {
                node_id: rt.ward_rt.ward.node_id().to_string(),
                fabric_class: fclass,
                silicon_lane: format!("jouleflow:{}", req.spec.workflow_name),
                calibration_source: rt.ward_rt.ward.default_calibration_source(),
                grid_carbon_g_per_kwh: carbon,
                rails: receipt.aggregate_rails,
            });
            rt.ward_rt.ward.ledger.record_trace(None, &trace);
            rt.pulse.record_from_trace(
                &receipt.run_id,
                format!("flow.{}", receipt.workflow_name),
                "main",
                0,
                rt.ward_rt.synapse_rt.current_synapse_lsn(),
                "jouleflow_durable_dag",
                &trace,
            );
            (StatusCode::OK, Json(receipt)).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn handle_flow_get_run(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Path(run_id): Path<String>,
) -> impl IntoResponse {
    match rt.flow.get_run(&run_id) {
        Some(run) => (StatusCode::OK, Json(serde_json::json!(run))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": format!("flow run '{run_id}' not found") })),
        )
            .into_response(),
    }
}

async fn handle_mesh_export_delta(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<ExportMeshDeltaReq>,
) -> impl IntoResponse {
    let delta = rt.ward_rt.synapse_rt.export_federated_delta(
        rt.ward_rt.ward.root_secret(),
        &req.delta_id,
        rt.ward_rt.ward.node_id(),
        rt.ward_rt.ward.fabric_class().as_str(),
        &req.expert_name,
        req.clip_l2_norm,
        req.dp_noise_sigma,
        req.engram_facts,
        req.training_energy_nj,
        req.peak_power_mw,
    );
    let wire_bytes = serde_json::to_vec(&delta).map(|v| v.len()).unwrap_or(0);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "wire_byte_size": wire_bytes,
            "delta": delta,
        })),
    )
}

async fn handle_mesh_merge(
    State(rt): State<Arc<PulseMeshRuntime>>,
    Json(req): Json<MergeMeshDeltasReq>,
) -> impl IntoResponse {
    let receipt = rt.ward_rt.synapse_rt.merge_federated_deltas(
        rt.ward_rt.ward.root_secret(),
        &req.expert_name,
        &req.deltas,
        req.max_byzantine_peers,
        req.max_allowed_l2,
    );

    let mut peers_notified = 0usize;
    if req.broadcast_to_fabric_peers {
        let local_id = rt.ward_rt.fabric_rt.fabric.local_node_id().to_string();
        for peer in rt.ward_rt.fabric_rt.fabric.topology() {
            if peer.node_id != local_id
                && let Some(url) = peer.endpoint_url
            {
                let endpoint = format!("{}/api/v1/synapse/mesh/merge", url.trim_end_matches('/'));
                let fwd_body = serde_json::json!({
                    "expert_name": req.expert_name,
                    "deltas": req.deltas,
                    "max_byzantine_peers": req.max_byzantine_peers,
                    "max_allowed_l2": req.max_allowed_l2,
                    "broadcast_to_fabric_peers": false,
                });
                if let Ok(r) = rt
                    .ward_rt
                    .fabric_rt
                    .http_client
                    .post(&endpoint)
                    .json(&fwd_body)
                    .send()
                    .await
                    && r.status().is_success()
                {
                    peers_notified += 1;
                }
            }
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "peers_notified": peers_notified,
            "receipt": receipt,
        })),
    )
}

fn extract_anthropic_prompt(req: &AnthropicMessagesReq) -> String {
    let mut parts = Vec::new();
    if let Some(ref sys) = req.system
        && !sys.trim().is_empty()
    {
        parts.push(sys.trim().to_string());
    }
    if let Some(last_msg) = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .or_else(|| req.messages.last())
    {
        match &last_msg.content {
            AnthropicContent::Text(s) => parts.push(s.clone()),
            AnthropicContent::Blocks(blocks) => {
                for b in blocks {
                    if b.kind == "text"
                        && let Some(ref t) = b.text
                    {
                        parts.push(t.clone());
                    }
                }
            }
        }
    }
    parts.join("\n")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
