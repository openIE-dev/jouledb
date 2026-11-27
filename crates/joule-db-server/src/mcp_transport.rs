//! MCP Transport Layer for JouleDB
//!
//! Provides stdio and SSE transports for the Model Context Protocol,
//! enabling AI agents (Claude, GPT, etc.) to interact with JouleDB
//! via standard MCP JSON-RPC messages.

use crate::mcp_bridge::DatabaseToolHandler;
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use futures::stream::Stream;
use inv_mcp_core::ToolCallRequest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::broadcast;

// ============================================================================
// MCP JSON-RPC types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpErrorBody>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpErrorBody {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl McpResponse {
    pub fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn error(id: Option<serde_json::Value>, code: i64, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(McpErrorBody {
                code,
                message,
                data: None,
            }),
        }
    }
}

// ============================================================================
// MCP Tool and Resource definitions
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct McpToolDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpResourceDef {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

// ============================================================================
// SSE transport state
// ============================================================================

#[derive(Clone)]
pub struct McpSseState {
    pub sender: broadcast::Sender<String>,
    /// Optional bridge to DatabaseToolHandler for tools/call dispatch.
    pub tool_handler: Option<Arc<DatabaseToolHandler>>,
}

impl McpSseState {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(128);
        Self {
            sender,
            tool_handler: None,
        }
    }

    pub fn with_handler(handler: Arc<DatabaseToolHandler>) -> Self {
        let (sender, _) = broadcast::channel(128);
        Self {
            sender,
            tool_handler: Some(handler),
        }
    }
}

/// Build the MCP HTTP/SSE route group (mounted at /mcp/*).
pub fn mcp_routes(state: McpSseState) -> Router {
    Router::new()
        .route("/mcp/sse", get(mcp_sse_handler))
        .route("/mcp/stream", get(mcp_sse_handler))
        .route("/mcp/messages", post(mcp_message_handler))
        .route("/mcp/request", post(mcp_message_handler))
        .route("/mcp", post(mcp_message_handler))
        .with_state(state)
}

// ============================================================================
// SSE endpoint: GET /mcp/sse
// ============================================================================

/// SSE stream wrapping a broadcast receiver
struct McpSseStream {
    rx: broadcast::Receiver<String>,
    sent_init: bool,
}

impl Stream for McpSseStream {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if !self.sent_init {
            self.sent_init = true;
            return Poll::Ready(Some(Ok(Event::default().data("connected"))));
        }

        match self.rx.try_recv() {
            Ok(msg) => Poll::Ready(Some(Ok(Event::default().data(msg)))),
            Err(broadcast::error::TryRecvError::Empty) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(broadcast::error::TryRecvError::Closed) => Poll::Ready(None),
        }
    }
}

pub async fn mcp_sse_handler(
    State(state): State<McpSseState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.sender.subscribe();
    Sse::new(McpSseStream {
        rx,
        sent_init: false,
    })
}

// ============================================================================
// Message endpoint: POST /mcp/messages
// ============================================================================

pub async fn mcp_message_handler(
    State(state): State<McpSseState>,
    axum::extract::Query(query_params): axum::extract::Query<HashMap<String, String>>,
    Json(mut request): Json<McpRequest>,
) -> Json<McpResponse> {
    if let Some(power) = query_params.get("power").filter(|p| !p.trim().is_empty())
        && request.method == "tools/list"
    {
        if !request.params.is_object() {
            request.params = serde_json::json!({});
        }
        if let Some(obj) = request.params.as_object_mut() {
            obj.entry("power".to_string())
                .or_insert_with(|| serde_json::Value::String(power.clone()));
        }
    }
    let response = handle_mcp_request(&request, state.tool_handler.as_deref());

    // Also broadcast the response over SSE
    if let Ok(json) = serde_json::to_string(&response) {
        let _ = state.sender.send(json);
    }

    Json(response)
}

// ============================================================================
// MCP request dispatcher
// ============================================================================

fn handle_mcp_request(
    request: &McpRequest,
    tool_handler: Option<&DatabaseToolHandler>,
) -> McpResponse {
    if request.method.starts_with("notifications/") {
        return McpResponse::success(request.id.clone(), serde_json::json!({}));
    }
    match request.method.as_str() {
        "initialize" => handle_initialize(request),
        "tools/list" => handle_tools_list(request, tool_handler),
        "resources/list" => handle_resources_list(request),
        "resources/read" => handle_resources_read(request, tool_handler),
        "tools/call" => handle_tools_call(request, tool_handler),
        "ping" => McpResponse::success(request.id.clone(), serde_json::json!({})),
        _ => McpResponse::error(
            request.id.clone(),
            -32601,
            format!("Method not found: {}", request.method),
        ),
    }
}

fn handle_initialize(request: &McpRequest) -> McpResponse {
    let client_proto = request
        .params
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("2024-11-05");
    let negotiated = match client_proto {
        "2025-03-26" | "2025-06-18" | "2025-11-25" => client_proto,
        _ => "2024-11-05",
    };
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({
            "protocolVersion": negotiated,
            "capabilities": {
                "tools": { "listChanged": false },
                "resources": { "listChanged": false }
            },
            "serverInfo": {
                "name": "jouledb",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn handle_tools_list(
    request: &McpRequest,
    tool_handler: Option<&DatabaseToolHandler>,
) -> McpResponse {
    let requested_power = request
        .params
        .get("power")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| tool_handler.and_then(|h| h.default_power()));

    if let (Some(domain), Some(handler)) = (requested_power, tool_handler) {
        let bundle = handler.export_domain_power(domain);
        let mut scoped_tools: Vec<serde_json::Value> = bundle.mcp_tools.clone();
        // Include 4 ultra-compact read/diagnostic/rerank tools alongside the governed domain actions
        for core_def in [
            plain_tool(
                "db.query",
                "Execute governed read SQL query against JouleDB",
                serde_json::json!({
                    "type": "object",
                    "properties": { "sql": { "type": "string" } },
                    "required": ["sql"]
                }),
            ),
            plain_tool(
                "sql.diagnose",
                "Sub-millisecond OpenCode/LSP SQL & Ontology governance diagnostics",
                serde_json::json!({
                    "type": "object",
                    "properties": { "sql": { "type": "string" } },
                    "required": ["sql"]
                }),
            ),
            plain_tool(
                "router.rerank",
                "Voyage AI / Cohere v2 compatible neural+lexical document reranker",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" },
                        "documents": { "type": "array", "items": { "type": "string" } },
                        "top_n": { "type": "integer" }
                    },
                    "required": ["query", "documents"]
                }),
            ),
            plain_tool(
                "router.rtk_compress",
                "Deterministic RTK noisy terminal/log token compressor",
                serde_json::json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
            ),
        ] {
            if let Ok(v) = serde_json::to_value(core_def) {
                scoped_tools.push(v);
            }
        }
        let scoped_json_len = serde_json::to_string(&scoped_tools)
            .map(|s| s.len())
            .unwrap_or(0);
        let est_tokens = (scoped_json_len + bundle.steering_markdown.len()).div_ceil(4);
        let full_baseline_tokens = 14_800usize;
        let saved = full_baseline_tokens.saturating_sub(est_tokens);
        return McpResponse::success(
            request.id.clone(),
            serde_json::json!({
                "tools": scoped_tools,
                "_meta": {
                    "jouledb_power": {
                        "domain_power": bundle.domain_power,
                        "steering_markdown": bundle.steering_markdown,
                        "estimated_prompt_tokens": est_tokens,
                        "tokens_saved_vs_full_mcp": saved,
                        "savings_pct": ((saved as f64 / full_baseline_tokens as f64) * 100.0),
                    }
                }
            }),
        );
    }

    let include_sota = request
        .params
        .get("includeAgenticExtensions")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let tools = if include_sota {
        all_sota_tool_definitions()
    } else {
        all_tool_definitions()
    };
    McpResponse::success(request.id.clone(), serde_json::json!({ "tools": tools }))
}

fn handle_resources_list(request: &McpRequest) -> McpResponse {
    let resources = vec![
        McpResourceDef {
            uri: crate::mcp_app_ui::STUDIO_APP_URI.to_string(),
            name: "JouleDB Studio".to_string(),
            description: "Interactive MCP App UI (SEP-1865) for SQL queries, visual charts, CoW database branches, and joule telemetry".to_string(),
            mime_type: crate::mcp_app_ui::MCP_APP_MIME.to_string(),
            meta: Some(serde_json::json!({
                "ui": { "prefersBorder": false }
            })),
        },
        McpResourceDef {
            uri: "jouledb://tables".to_string(),
            name: "Table Catalog".to_string(),
            description: "List of all tables in the database".to_string(),
            mime_type: "application/json".to_string(),
            meta: None,
        },
        McpResourceDef {
            uri: "jouledb://energy".to_string(),
            name: "Energy State".to_string(),
            description: "Current energy consumption and budget state".to_string(),
            mime_type: "application/json".to_string(),
            meta: None,
        },
        McpResourceDef {
            uri: "jouledb://timeseries".to_string(),
            name: "Time Series Metrics".to_string(),
            description: "Metrics stored in the joule-db-features TimeSeriesStore (TSLIST)"
                .to_string(),
            mime_type: "application/json".to_string(),
            meta: None,
        },
    ];
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({ "resources": resources }),
    )
}

fn handle_resources_read(
    request: &McpRequest,
    tool_handler: Option<&DatabaseToolHandler>,
) -> McpResponse {
    let uri = request
        .params
        .get("uri")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if uri.is_empty() {
        return McpResponse::error(
            request.id.clone(),
            -32602,
            "resources/read requires params.uri".into(),
        );
    }

    if uri == crate::mcp_app_ui::STUDIO_APP_URI {
        return McpResponse::success(
            request.id.clone(),
            serde_json::json!({
                "contents": [{
                    "uri": crate::mcp_app_ui::STUDIO_APP_URI,
                    "mimeType": crate::mcp_app_ui::MCP_APP_MIME,
                    "text": crate::mcp_app_ui::studio_app_html(),
                    "_meta": {
                        "ui": { "prefersBorder": false }
                    }
                }]
            }),
        );
    }

    let Some(handler) = tool_handler else {
        return McpResponse::error(
            request.id.clone(),
            -32603,
            "MCP tool handler not configured (DatabaseToolHandler missing)".into(),
        );
    };
    let tool_resp = handler.read_resource(uri);
    if let Some(err) = tool_resp.error {
        return McpResponse::error(request.id.clone(), err.code as i64, err.message);
    }
    let body = tool_resp.result.unwrap_or(serde_json::Value::Null);
    let text = serde_json::to_string(&body).unwrap_or_else(|_| body.to_string());
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({
            "contents": [{
                "uri": uri,
                "mimeType": "application/json",
                "text": text
            }]
        }),
    )
}

fn handle_tools_call(
    request: &McpRequest,
    tool_handler: Option<&DatabaseToolHandler>,
) -> McpResponse {
    let tool_name = request
        .params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let arguments = request
        .params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));

    if tool_name == "db.studio" {
        let Some(handler) = tool_handler else {
            return McpResponse::error(
                request.id.clone(),
                -32603,
                "MCP tool handler not configured (DatabaseToolHandler missing)".to_string(),
            );
        };
        let default_sql = "SELECT table_name FROM information_schema.tables";
        let sql = arguments
            .get("sql")
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or(default_sql);
        let tab = arguments
            .get("tab")
            .and_then(|v| v.as_str())
            .unwrap_or("sql");

        let mut q_args = HashMap::new();
        q_args.insert("sql".to_string(), serde_json::json!(sql));
        let q_resp = handler.call_tool(ToolCallRequest {
            id: "studio-q".to_string(),
            tool: "db.query".to_string(),
            arguments: q_args,
        });
        let t_resp = handler.call_tool(ToolCallRequest {
            id: "studio-t".to_string(),
            tool: "db.list_tables".to_string(),
            arguments: HashMap::new(),
        });
        let b_resp = handler.call_tool(ToolCallRequest {
            id: "studio-b".to_string(),
            tool: "db.branch_list".to_string(),
            arguments: HashMap::new(),
        });

        let tables_val = t_resp
            .result
            .as_ref()
            .and_then(|v| v.get("tables").cloned())
            .unwrap_or_else(|| serde_json::json!([]));
        let branches_val = b_resp
            .result
            .as_ref()
            .and_then(|v| v.get("branches").cloned())
            .unwrap_or_else(|| serde_json::json!([]));
        let query_json = q_resp.result.unwrap_or_else(|| serde_json::json!({}));
        let joules = q_resp.energy_joules.unwrap_or(0.000002);
        let row_cnt = query_json
            .get("rows")
            .and_then(|r| r.as_array())
            .map_or(0, Vec::len);
        let summary = format!(
            "Opened JouleDB Studio (`{}` tab): executed `{}` ({} rows, {:.6} J est_j).",
            tab, sql, row_cnt, joules
        );
        let structured = serde_json::json!({
            "app": "jouledb-studio",
            "version": env!("CARGO_PKG_VERSION"),
            "active_tab": tab,
            "sql": sql,
            "query_result": query_json,
            "tables": tables_val,
            "branches": branches_val,
            "energy_joules": joules,
            "energy_evidence": "est_j"
        });
        return McpResponse::success(
            request.id.clone(),
            serde_json::json!({
                "content": [{ "type": "text", "text": summary }],
                "structuredContent": structured,
                "energy_joules": joules
            }),
        );
    }

    if let Some(frontier_json) = dispatch_frontier_or_analytic_mcp_tool(tool_name, &arguments) {
        let text = frontier_json.to_string();
        return McpResponse::success(
            request.id.clone(),
            serde_json::json!({
                "content": [{ "type": "text", "text": text }],
                "structuredContent": frontier_json,
                "energy_joules": 0.000002
            }),
        );
    }

    let args_map: HashMap<String, serde_json::Value> = match arguments {
        serde_json::Value::Object(map) => map.into_iter().collect(),
        _ => HashMap::new(),
    };

    let call_id = request
        .id
        .clone()
        .map(|v| match v {
            serde_json::Value::String(s) => s,
            other => other.to_string(),
        })
        .unwrap_or_else(|| "0".to_string());

    let Some(handler) = tool_handler else {
        return McpResponse::error(
            request.id.clone(),
            -32603,
            "MCP tool handler not configured (DatabaseToolHandler missing)".to_string(),
        );
    };

    let tool_req = ToolCallRequest {
        id: call_id,
        tool: tool_name.to_string(),
        arguments: args_map,
    };

    let tool_resp = handler.call_tool(tool_req);

    if let Some(err) = tool_resp.error {
        return McpResponse::error(request.id.clone(), err.code as i64, err.message);
    }

    let result_json = tool_resp.result.unwrap_or(serde_json::Value::Null);
    let text = match &result_json {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    McpResponse::success(
        request.id.clone(),
        serde_json::json!({
            "content": [{
                "type": "text",
                "text": text
            }],
            "structuredContent": result_json,
            "energy_joules": tool_resp.energy_joules.unwrap_or(0.0)
        }),
    )
}

// ============================================================================
// Tool definitions (all 24 tools)
// ============================================================================

fn plain_tool(name: &str, description: &str, input_schema: serde_json::Value) -> McpToolDef {
    McpToolDef {
        name: name.to_string(),
        title: None,
        description: description.to_string(),
        input_schema,
        meta: None,
    }
}

pub fn all_tool_definitions() -> Vec<McpToolDef> {
    vec![
        // Interactive MCP App Studio entrypoint (`SEP-1865` + Antigravity side-pane/global)
        McpToolDef {
            name: "db.studio".to_string(),
            title: Some("JouleDB Studio".to_string()),
            description: "Open the interactive JouleDB Studio MCP App (SQL workbench, visual charts, schema explorer, O(1) CoW branches, and joule telemetry)".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "Optional initial SQL query to execute in Studio" },
                    "tab": { "type": "string", "enum": ["sql", "schema", "branches", "energy"], "description": "Optional initial tab" }
                }
            }),
            meta: Some(crate::mcp_app_ui::studio_entrypoint_meta()),
        },
        // Existing DB tools
        McpToolDef {
            name: "db.query".to_string(),
            title: Some("Run SQL Query".to_string()),
            description: "Execute a SQL query against JouleDB and render interactive results".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "SQL query to execute" },
                    "params": { "type": "object", "description": "Named parameters" }
                },
                "required": ["sql"]
            }),
            meta: Some(crate::mcp_app_ui::inline_ui_meta()),
        },
        plain_tool(
            "db.insert",
            "Insert a row into a table",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "table": { "type": "string" },
                    "row": { "type": "object" }
                },
                "required": ["table", "row"]
            }),
        ),
        plain_tool(
            "db.list_tables",
            "List all tables in the database",
            serde_json::json!({ "type": "object", "properties": {} }),
        ),
        plain_tool(
            "db.describe_table",
            "Get column definitions for a table",
            serde_json::json!({
                "type": "object",
                "properties": { "table": { "type": "string" } },
                "required": ["table"]
            }),
        ),
        plain_tool(
            "db.create_table",
            "Create a new table with DDL",
            serde_json::json!({
                "type": "object",
                "properties": { "sql": { "type": "string" } },
                "required": ["sql"]
            }),
        ),
        plain_tool(
            "db.drop_table",
            "Drop a table",
            serde_json::json!({
                "type": "object",
                "properties": { "table": { "type": "string" } },
                "required": ["table"]
            }),
        ),
        // Branch tools
        plain_tool(
            "db.branch_create",
            "Create a CoW database branch with optional energy budget",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Branch name" },
                    "energy_budget_uj": { "type": "integer", "description": "Energy budget in microjoules" }
                },
                "required": ["name"]
            }),
        ),
        McpToolDef {
            name: "db.branch_list".to_string(),
            title: Some("CoW Database Branches".to_string()),
            description: "List all database branches".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            meta: Some(crate::mcp_app_ui::inline_ui_meta()),
        },
        plain_tool(
            "db.branch_merge",
            "Merge a branch back to its parent",
            serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
        ),
        plain_tool(
            "db.branch_delete",
            "Delete a database branch",
            serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
        ),
        // Vector tools
        plain_tool(
            "db.vector_search",
            "Similarity search on vector columns. Returns results + energy cost.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "table": { "type": "string" },
                    "column": { "type": "string" },
                    "query": { "type": "array", "items": { "type": "number" } },
                    "k": { "type": "integer", "default": 10 },
                    "metric": { "type": "string", "enum": ["l2", "cosine", "ip"] }
                },
                "required": ["table", "column", "query"]
            }),
        ),
        plain_tool(
            "db.vector_upsert",
            "Insert or update vectors in a table",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "table": { "type": "string" },
                    "column": { "type": "string" },
                    "vectors": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["table", "column", "vectors"]
            }),
        ),
        // Tenant tools
        plain_tool(
            "db.tenant_create",
            "Create a new tenant with resource quotas",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "energy_budget_uj": { "type": "integer" }
                },
                "required": ["name"]
            }),
        ),
        plain_tool(
            "db.tenant_list",
            "List all tenants and their resource usage",
            serde_json::json!({ "type": "object", "properties": {} }),
        ),
        // Memory tools
        plain_tool(
            "db.memory_store",
            "Store a memory (episodic, semantic, or working) with temporal metadata",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Memory content" },
                    "memory_type": { "type": "string", "enum": ["episodic", "semantic", "working"] },
                    "metadata": { "type": "object" },
                    "agent_id": { "type": "string" }
                },
                "required": ["content"]
            }),
        ),
        plain_tool(
            "db.memory_recall",
            "Recall memories by similarity with temporal decay scoring",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Query to match against" },
                    "k": { "type": "integer", "default": 5 },
                    "half_life_hours": { "type": "number", "default": 168.0 },
                    "memory_type": { "type": "string" },
                    "agent_id": { "type": "string" }
                },
                "required": ["query"]
            }),
        ),
        // Schema/Status tools
        McpToolDef {
            name: "db.schema_inspect".to_string(),
            title: Some("Inspect Schema".to_string()),
            description: "Full schema dump — all tables, columns, and types".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
            meta: Some(crate::mcp_app_ui::inline_ui_meta()),
        },
        plain_tool(
            "db.energy_budget",
            "Check or set the energy budget for the current session",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "budget_uj": { "type": "integer", "description": "Set energy budget (microjoules)" }
                }
            }),
        ),
        plain_tool(
            "db.status",
            "Server lifecycle state, uptime, energy stats",
            serde_json::json!({ "type": "object", "properties": {} }),
        ),
        // Workflow tools
        plain_tool(
            "db.workflow_create",
            "Create a durable workflow definition with energy-metered steps",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Workflow name" },
                    "steps": { "type": "array", "items": { "type": "object" } },
                    "energy_budget_uj": { "type": "integer" }
                },
                "required": ["name", "steps"]
            }),
        ),
        plain_tool(
            "db.workflow_run",
            "Execute a workflow definition by ID. Returns step results + energy cost.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "workflow_id": { "type": "string", "description": "Workflow definition ID" }
                },
                "required": ["workflow_id"]
            }),
        ),
        plain_tool(
            "db.workflow_status",
            "Get the status of a workflow instance",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "instance_id": { "type": "string", "description": "Workflow instance ID" }
                },
                "required": ["instance_id"]
            }),
        ),
        plain_tool(
            "db.queue_publish",
            "Publish a message to a durable queue topic",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": { "type": "string", "description": "Queue topic name" },
                    "payload": { "type": "string", "description": "Message payload" }
                },
                "required": ["topic", "payload"]
            }),
        ),
        plain_tool(
            "db.edge_pop_list",
            "List all Edge Points of Presence and their sync status",
            serde_json::json!({ "type": "object", "properties": {} }),
        ),
    ]
}

fn dispatch_frontier_or_analytic_mcp_tool(
    tool_name: &str,
    args: &serde_json::Value,
) -> Option<serde_json::Value> {
    match tool_name {
        "sql.dialect_repair" => {
            let faulty_sql = args
                .get("sql")
                .or_else(|| args.get("faulty_sql"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let known_cols: Vec<String> = args
                .get("known_columns")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            Some(serde_json::json!(
                joule_db_query::sql_critic::diagnose_and_repair_sql(faulty_sql, &known_cols)
            ))
        }
        "robotics.stl_trajectory_clamp" => {
            let start_xyz: [f64; 3] = args
                .get("start_xyz")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or([0.0, 0.0, 0.0]);
            let envelope: joule_db_core::frontier_eval::SafeVlaEnvelope = args
                .get("envelope")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or(joule_db_core::frontier_eval::SafeVlaEnvelope {
                    max_force_newtons: 15.0,
                    max_step_delta_m: 0.25,
                    keep_out_spheres: Vec::new(),
                });
            let trajectory: Vec<joule_db_core::frontier_eval::VlaStepSample> = args
                .get("trajectory")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let (shielded, raw_audit, safe_audit) =
                joule_db_core::frontier_eval::shield_and_audit_vla_trajectory(
                    start_xyz,
                    &trajectory,
                    &envelope,
                );
            Some(serde_json::json!({
                "shielded_trajectory": shielded,
                "raw_audit": raw_audit,
                "shielded_audit": safe_audit,
            }))
        }
        "db.synthetic_columnar_smoke" => {
            let rows = args
                .get("rows")
                .or_else(|| args.get("clickbench_rows"))
                .and_then(|v| v.as_u64())
                .unwrap_or(4_096) as usize;
            let cb = joule_db_query::analytic_benchmarks::run_clickbench_columnar_suite(rows);
            let job = joule_db_query::analytic_benchmarks::run_job_skewed_wcoj_suite(100, 10);
            let stac = joule_db_query::analytic_benchmarks::run_tsbs_stac_m3_suite(500, 100);
            let beir = joule_db_query::analytic_benchmarks::run_synthetic_beir_suite(64);
            Some(serde_json::json!({
                "workload_kind": "synthetic_in_memory_smoke_test",
                "energy_evidence": "est_j",
                "synthetic_columnar_codec": cb,
                "synthetic_skewed_wcoj": job,
                "synthetic_asof_ticks": stac,
                "synthetic_hybrid_rrf": beir,
            }))
        }
        "amorphic.unsilo_morph" => {
            let frags: Vec<joule_db_amorphic::unsilo::RawSiloFragment> = args
                .get("fragments")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let intent: joule_db_viz::amorphic_display::ObserverIntent = args
                .get("intent")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            if frags.is_empty() {
                let bench = joule_db_amorphic::unsilo::run_unsilo_benchmark_suite();
                Some(serde_json::json!({ "benchmark_receipt": bench }))
            } else {
                let mut cont = joule_db_amorphic::unsilo::AmorphicSemanticContinuum::new();
                for f in &frags {
                    cont.ingest_fragment(f);
                }
                let disp: Vec<_> = cont
                    .atoms
                    .iter()
                    .map(crate::pq_verifiable::bridge_atom_to_display)
                    .collect();
                let canvas = joule_db_viz::amorphic_display::AmorphicDisplaySynthesizer::synthesize(
                    &disp, intent,
                );
                Some(serde_json::json!({
                    "unified_entities": cont.materialize_unified_entities(),
                    "canvas_spec": canvas,
                }))
            }
        }
        "sovereign.continual_plasticity" => {
            let tasks = args.get("num_tasks").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
            joule_db_weights::continual_plasticity::run_continual_plasticity_suite(
                32, 8, tasks, 250, 12,
            )
            .ok()
            .map(|r| serde_json::json!(r))
        }
        "sovereign.temporal_graph_smoke" => joule_db_query::rel_rfm::run_relbench_suite()
            .ok()
            .map(|(churn, ltv)| serde_json::json!({ "synthetic_only": true, "user_churn": churn, "user_ltv": ltv })),
        "sovereign.pq_verifiable_attest" => {
            crate::pq_verifiable::execute_wave28_global_sota_suite()
                .ok()
                .map(|v| v["b25_post_quantum_verifiable_sovereign"].clone())
        }
        "sovereign.multilingual_span_smoke" => Some(serde_json::json!({
            "synthetic_only": true,
            "energy_evidence": "est_j",
            "report": joule_db_core::multilingual_global::run_global_mmlu_12script_suite()
        })),
        "amorphic.morph_rules_and_lens" => {
            let frags: Vec<joule_db_amorphic::unsilo::RawSiloFragment> = args
                .get("fragments")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let morph_dsl = args.get("morph_dsl").and_then(|v| v.as_str());
            let lens_dsl = args.get("custom_lens_dsl").and_then(|v| v.as_str());
            if frags.is_empty() && morph_dsl.is_none() {
                Some(serde_json::json!({
                    "morph_benchmark_receipt": joule_db_amorphic::morph_rules::run_programmable_morph_suite().ok()
                }))
            } else {
                let mut cont = joule_db_amorphic::unsilo::AmorphicSemanticContinuum::new();
                for f in &frags {
                    cont.ingest_fragment(f);
                }
                let rules = morph_dsl
                    .and_then(|d| {
                        joule_db_amorphic::morph_rules::MorphRuleEngine::parse_dsl(d).ok()
                    })
                    .unwrap_or_default();
                let (morphed, rep) = joule_db_amorphic::morph_rules::MorphRuleEngine::apply_rules(
                    &cont.atoms,
                    &rules,
                );
                let disp: Vec<_> = morphed
                    .iter()
                    .map(crate::pq_verifiable::bridge_atom_to_display)
                    .collect();
                let compiled_lens = lens_dsl
                    .and_then(|ld| {
                        joule_db_viz::custom_lens::CustomLensCompiler::parse_dsl(ld).ok()
                    })
                    .map(|spec| {
                        joule_db_viz::custom_lens::CustomLensCompiler::compile(&disp, &spec)
                    });
                Some(serde_json::json!({
                    "morph_report": rep,
                    "custom_lens": compiled_lens,
                }))
            }
        }
        "amorphic.stream_corpus_scale" => {
            let target_bytes = args
                .get("target_bytes")
                .and_then(|v| v.as_u64())
                .unwrap_or(2_500_000);
            let n_ent = args
                .get("num_unique_entities")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize;
            let scratch =
                std::env::temp_dir().join(format!("jouledb_mcp_stream_{}", std::process::id()));
            let rec =
                joule_db_amorphic::streaming_continuum::generate_and_stream_heterogeneous_corpus(
                    &scratch,
                    target_bytes,
                    n_ent,
                    &[],
                )
                .ok();
            let _ = std::fs::remove_dir_all(&scratch);
            Some(serde_json::json!({ "streaming_receipt": rec }))
        }
        "bridge.entity_merge_smoke" => {
            Some(crate::universal_bridge::run_universal_bridge_certification())
        }
        "trust.security_controls_self_test" => {
            crate::trust_compliance::execute_full_trust_and_compliance_suite()
                .ok()
                .map(|v| serde_json::json!(v))
        }
        "trust.sbom_cyclonedx" => {
            Some(crate::trust_compliance::generate_cyclonedx_sbom_and_aibom())
        }
        "migrate.shadow_cutover" => {
            Some(crate::live_migration::run_all_ecosystems_migration_certification())
        }
        "core.tlv_and_pitr_smoke" => {
            joule_db_core::v1_stability_and_pitr::certify_v1_abi_stability_and_pitr()
                .ok()
                .map(|r| serde_json::json!(r))
        }
        "query.window_cube_aqe_smoke" => Some(serde_json::json!(
            joule_db_query::sql2023_powerhouse::run_sql2023_powerhouse_certification()
        )),
        "cluster.in_memory_partition_sim" => Some(serde_json::json!(
            crate::distributed_jepsen_consensus::run_geo_raft_jepsen_certification()
        )),
        "deploy.iac_blueprint" => {
            let tier = args.get("tier").and_then(|v| v.as_u64()).unwrap_or(5) as u8;
            Some(crate::distributed_jepsen_consensus::generate_iac_compliance_blueprint(tier))
        }
        "bench.subsystem_smoke" => {
            Some(crate::distributed_jepsen_consensus::run_v1_global_powerhouse_suite())
        }
        _ => None,
    }
}

pub fn all_sota_tool_definitions() -> Vec<McpToolDef> {
    let mut tools = all_tool_definitions();
    for (name, desc) in [
        (
            "db.branch_diff",
            "Compute 3-way structural file/KV/SQL diff vs parent",
        ),
        (
            "db.workflow_checkpoint",
            "Persist or idempotently replay a durable agentic workflow step checkpoint",
        ),
        (
            "db.workflow_resume",
            "Inspect completed durable workflow step checkpoints for crash-proof resume",
        ),
        (
            "ledger.create_account",
            "Create an exact 128-bit double-entry financial ledger account with optional overdraft floor",
        ),
        (
            "ledger.transfer",
            "Execute an atomic double-entry transfer with mandatory idempotency_key",
        ),
        (
            "ledger.transfer_batch",
            "Execute a high-throughput single-fdatasync batch of double-entry transfers (atomic or per-item clearing)",
        ),
        (
            "ledger.hold",
            "Place a two-phase pending authorization hold reserving funds atomically",
        ),
        (
            "ledger.post_hold",
            "Capture (full or partial) a pending financial hold with idempotency_key",
        ),
        (
            "ledger.void_hold",
            "Void a pending financial hold and release reserved balance",
        ),
        (
            "ledger.verify",
            "Verify the SHA-256 hash chain and global double-entry conservation across the ledger",
        ),
        (
            "db.agentfs_write",
            "Write a virtual file into a branch-isolated JouleAgentFS workspace",
        ),
        (
            "db.agentfs_read",
            "Read a virtual file from a branch-isolated JouleAgentFS workspace",
        ),
        (
            "db.agentfs_list",
            "List virtual files under a directory prefix on a JouleAgentFS branch",
        ),
        (
            "db.memory_inhibit",
            "Record explicit negative knowledge / inhibit superseded facts",
        ),
        (
            "db.explain_energy",
            "Return pre-execution CascadeTier and analytical energy estimate (est_j)",
        ),
        (
            "sql.dialect_repair",
            "Rewrite SQL dialect helpers (SAFE_DIVIDE, IFNULL), NOT IN (NULL) guards, and column typos",
        ),
        (
            "robotics.stl_trajectory_clamp",
            "Clamp 3D trajectory points against force, step-delta, and keep-out sphere constraints",
        ),
        (
            "db.synthetic_columnar_smoke",
            "Run synthetic in-memory ALP/FSST codec, skewed WCOJ, ASOF, and hybrid RRF smoke checks",
        ),
        (
            "amorphic.unsilo_morph",
            "Extract semantic atoms from JSON/CSV/Prose/Triples into 10,240-bit HDC vectors & visual lenses",
        ),
        (
            "sovereign.continual_plasticity",
            "Run synthetic small-matrix orthogonal continual update & CoW rollback smoke check",
        ),
        (
            "sovereign.temporal_graph_smoke",
            "Run synthetic temporal relational graph chronological-filter smoke check",
        ),
        (
            "sovereign.pq_verifiable_attest",
            "Issue & verify hybrid Ed25519 + NIST FIPS 204 ML-DSA-65 Merkle-rooted execution envelopes",
        ),
        (
            "sovereign.multilingual_span_smoke",
            "Run 12-script UTF-8 span tokenization & analytical (est_j) token-fertility smoke check",
        ),
        (
            "amorphic.morph_rules_and_lens",
            "Execute declarative MorphRule DSL unit harmonization, VSA role unbinding, and custom chart specs",
        ),
        (
            "amorphic.stream_corpus_scale",
            "Stream multi-format heterogeneous records (JSONL/CSV/XML/k=v/Prose/Triples) in bounded memory",
        ),
        (
            "bridge.entity_merge_smoke",
            "Run in-process 5-fragment entity merge and 4D vector lookup smoke check",
        ),
        (
            "trust.security_controls_self_test",
            "Run self-tests of implemented technical security controls (hash chain, WORM flag, DEK zeroization, Luhn redaction, MAC lattice; unaudited)",
        ),
        (
            "trust.sbom_cyclonedx",
            "Generate CycloneDX v1.6 JSON SBOM and loaded model manifest",
        ),
        (
            "migrate.shadow_cutover",
            "Run multi-format record import and multiset BLAKE3 reconciliation smoke check",
        ),
        (
            "core.tlv_and_pitr_smoke",
            "Run binary TLV/CRC-32 framing and WAL LSN/timestamp point-in-time recovery smoke check",
        ),
        (
            "query.window_cube_aqe_smoke",
            "Run SQL window frames, 3D CUBE, recursive CTE, and runtime AQE join-switch unit check",
        ),
        (
            "cluster.in_memory_partition_sim",
            "Run scripted 5-struct in-memory partition and HLC monotonicity unit simulation (not TCP Raft or Jepsen)",
        ),
        (
            "deploy.iac_blueprint",
            "Return starter Helm & Terraform configuration flags for control profiles 0..5 (unaudited)",
        ),
        (
            "bench.subsystem_smoke",
            "Run the 6 fast in-process deterministic subsystem smoke checks (v0.6.0)",
        ),
        (
            "ontology.register_object",
            "Register a governed OntologyObject mapped to a JouleDB backing SQL table",
        ),
        (
            "ontology.register_action",
            "Register a governed KineticAction with EARS preconditions & optional FinancialLedger hook",
        ),
        (
            "ontology.execute_action",
            "Execute a KineticAction on main or inside a pure-RAM CoW branch dry-run",
        ),
        (
            "ontology.domain_power",
            "Export dynamically scoped Kiro Power MCP toolset + steering bundle for a domain",
        ),
        (
            "sql.diagnose",
            "Run sub-millisecond OpenCode/LSP SQL syntax, catalog, blast-radius, and ontology diagnostics",
        ),
        (
            "router.rtk_compress",
            "Run deterministic RTK context token compression on noisy CLI/agent logs",
        ),
        (
            "router.rerank",
            "Rank candidate documents via Voyage AI / Cohere v2 compatible neural+lexical reranker",
        ),
        (
            "synapse.compose_super_adapter",
            "Compose orthogonal specialist Harness LoRA adapters via rank concatenation",
        ),
    ] {
        tools.push(plain_tool(
            name,
            desc,
            serde_json::json!({ "type": "object", "properties": {} }),
        ));
    }
    tools
}

// ============================================================================
// Stdio transport (for Claude Desktop / Claude Code / Antigravity / local agents)
// ============================================================================

/// Run the MCP stdio transport — reads JSON-RPC from stdin, writes to stdout.
/// Per JSON-RPC 2.0 & MCP spec, client notifications (`id` absent, e.g.
/// `notifications/initialized`) MUST NOT emit a response frame on stdout.
pub async fn run_stdio_transport(tool_handler: Option<Arc<DatabaseToolHandler>>) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let response = match serde_json::from_str::<McpRequest>(trimmed) {
                    Ok(request) => {
                        if request.id.is_none() {
                            // Notification (e.g. `notifications/initialized`) — no reply frame
                            continue;
                        }
                        handle_mcp_request(&request, tool_handler.as_deref())
                    }
                    Err(e) => McpResponse::error(None, -32700, format!("Parse error: {}", e)),
                };

                if let Ok(json) = serde_json::to_string(&response) {
                    let _ = stdout.write_all(json.as_bytes()).await;
                    let _ = stdout.write_all(b"\n").await;
                    let _ = stdout.flush().await;
                }
            }
            Err(e) => {
                tracing::error!("MCP stdio read error: {}", e);
                break;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(1)),
            method: "initialize".to_string(),
            params: serde_json::json!({}),
        };
        let resp = handle_mcp_request(&req, None);
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["protocolVersion"], "2024-11-05");
        assert_eq!(result["serverInfo"]["name"], "jouledb");
    }

    #[test]
    fn test_tools_list() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(2)),
            method: "tools/list".to_string(),
            params: serde_json::json!({}),
        };
        let resp = handle_mcp_request(&req, None);
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 25);

        // Verify all tools have required fields and db.studio exposes SEP-1865 UI metadata
        for tool in tools {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert!(tool["inputSchema"].is_object());
        }
        let studio = tools
            .iter()
            .find(|t| t["name"] == "db.studio")
            .expect("db.studio tool");
        assert_eq!(
            studio["_meta"]["ui"]["resourceUri"],
            crate::mcp_app_ui::STUDIO_APP_URI
        );
    }

    #[test]
    fn test_resources_list() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(3)),
            method: "resources/list".to_string(),
            params: serde_json::json!({}),
        };
        let resp = handle_mcp_request(&req, None);
        let result = resp.result.unwrap();
        let resources = result["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 4);
        let uris: Vec<&str> = resources.iter().filter_map(|r| r["uri"].as_str()).collect();
        assert!(uris.contains(&crate::mcp_app_ui::STUDIO_APP_URI));
        assert!(uris.contains(&"jouledb://tables"));
        assert!(uris.contains(&"jouledb://energy"));
        assert!(uris.contains(&"jouledb://timeseries"));

        // Verify SEP-1865 UI HTML resource reads cleanly with text/html;profile=mcp-app
        let read_ui = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(31)),
            method: "resources/read".to_string(),
            params: serde_json::json!({ "uri": crate::mcp_app_ui::STUDIO_APP_URI }),
        };
        let ui_resp = handle_mcp_request(&read_ui, None);
        let ui_res = ui_resp.result.expect("ui result");
        assert_eq!(
            ui_res["contents"][0]["mimeType"],
            crate::mcp_app_ui::MCP_APP_MIME
        );
        let html = ui_res["contents"][0]["text"].as_str().expect("html");
        assert!(html.contains("ui/initialize"));
        assert!(html.contains("ui/notifications/initialized"));
    }

    #[test]
    fn test_ping() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(4)),
            method: "ping".to_string(),
            params: serde_json::json!({}),
        };
        let resp = handle_mcp_request(&req, None);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_unknown_method() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(5)),
            method: "nonexistent/method".to_string(),
            params: serde_json::json!({}),
        };
        let resp = handle_mcp_request(&req, None);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_tools_call_without_handler_errors() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(6)),
            method: "tools/call".to_string(),
            params: serde_json::json!({ "name": "db.energy" }),
        };
        let resp = handle_mcp_request(&req, None);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32603);
    }

    #[test]
    fn test_tools_call_dispatches_to_bridge() {
        use crate::mcp_bridge::DatabaseToolHandler;
        use crate::query::{QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse};
        use std::collections::HashMap as StdHashMap;

        struct MockExec;
        impl QueryExecutor for MockExec {
            fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
                Ok(QueryResponse {
                    columns: vec!["result".into()],
                    rows: vec![vec![serde_json::json!(format!(
                        "executed: {}",
                        request.sql
                    ))]],
                    affected_rows: Some(1),
                    execution_time_ms: 1,
                    truncated: false,
                    warnings: vec![],
                    energy_joules: Some(0.001),
                    power_watts: Some(5.0),
                    device_target: Some("cpu".into()),
                    algorithm_type: Some("btree".into()),
                    session_id: None,
                    viz_hint: None,
                    energy_receipt: None,
                })
            }
        }

        let handler = DatabaseToolHandler::new(Arc::new(MockExec));
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(7)),
            method: "tools/call".to_string(),
            params: serde_json::json!({
                "name": "db.query",
                "arguments": { "sql": "SELECT 1" }
            }),
        };
        let resp = handle_mcp_request(&req, Some(&handler));
        assert!(resp.error.is_none(), "unexpected error: {:?}", resp.error);
        let result = resp.result.expect("result");
        assert!(result["content"].is_array());
        assert!(result["structuredContent"].is_object() || result["structuredContent"].is_array());
        assert!(result["energy_joules"].as_f64().unwrap_or(0.0) > 0.0);

        // Unknown tool must not pretend success (db.status is a real tool now).
        let bad = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(8)),
            method: "tools/call".to_string(),
            params: serde_json::json!({ "name": "db.no_such_tool" }),
        };
        let bad_resp = handle_mcp_request(&bad, Some(&handler));
        assert!(bad_resp.error.is_some());
        let _ = StdHashMap::<String, serde_json::Value>::new();
    }

    #[test]
    fn test_all_tool_definitions_complete() {
        let tools = all_tool_definitions();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // Verify key tools exist
        assert!(names.contains(&"db.query"));
        assert!(names.contains(&"db.branch_create"));
        assert!(names.contains(&"db.vector_search"));
        assert!(names.contains(&"db.tenant_create"));
        assert!(names.contains(&"db.memory_store"));
        assert!(names.contains(&"db.memory_recall"));
        assert!(names.contains(&"db.schema_inspect"));
        assert!(names.contains(&"db.status"));
    }

    #[test]
    fn test_resources_read_uses_catalog_sql() {
        use crate::mcp_bridge::DatabaseToolHandler;
        use crate::query::{QueryErrorResponse, QueryExecutor, QueryRequest, QueryResponse};

        struct MockExec;
        impl QueryExecutor for MockExec {
            fn execute(&self, request: &QueryRequest) -> Result<QueryResponse, QueryErrorResponse> {
                Ok(QueryResponse {
                    columns: vec!["sql".into()],
                    rows: vec![vec![serde_json::json!(request.sql.clone())]],
                    affected_rows: None,
                    execution_time_ms: 1,
                    truncated: false,
                    warnings: vec![],
                    energy_joules: Some(0.0004),
                    power_watts: Some(1.0),
                    device_target: Some("cpu".into()),
                    algorithm_type: Some("catalog".into()),
                    session_id: None,
                    viz_hint: None,
                    energy_receipt: None,
                })
            }
        }

        let missing = McpRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(9)),
            method: "resources/read".into(),
            params: serde_json::json!({ "uri": "jouledb://tables" }),
        };
        let missing_resp = handle_mcp_request(&missing, None);
        assert_eq!(missing_resp.error.unwrap().code, -32603);

        let handler = DatabaseToolHandler::new(Arc::new(MockExec));
        let req = McpRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(10)),
            method: "resources/read".into(),
            params: serde_json::json!({ "uri": "jouledb://tables" }),
        };
        let resp = handle_mcp_request(&req, Some(&handler));
        assert!(resp.error.is_none(), "{:?}", resp.error);
        let text = resp.result.unwrap()["contents"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("information_schema.tables"), "{text}");

        let series = McpRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(11)),
            method: "resources/read".into(),
            params: serde_json::json!({ "uri": "jouledb://timeseries/cpu?start=0&end=10" }),
        };
        let series_resp = handle_mcp_request(&series, Some(&handler));
        let series_text = series_resp.result.unwrap()["contents"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(series_text.contains("TSQUERY cpu 0 10"), "{series_text}");

        let unknown = McpRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(12)),
            method: "resources/read".into(),
            params: serde_json::json!({ "uri": "jouledb://nope" }),
        };
        let unknown_resp = handle_mcp_request(&unknown, Some(&handler));
        assert_eq!(unknown_resp.error.unwrap().code, -32002);
    }
}
