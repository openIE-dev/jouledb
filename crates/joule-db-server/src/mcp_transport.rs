//! MCP Transport Layer for JouleDB
//!
//! Provides stdio and SSE transports for the Model Context Protocol,
//! enabling AI agents (Claude, GPT, etc.) to interact with JouleDB
//! via standard MCP JSON-RPC messages.

use crate::mcp_bridge::DatabaseToolHandler;
use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::Router;
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
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct McpResourceDef {
    pub uri: String,
    pub name: String,
    pub description: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
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
    Json(request): Json<McpRequest>,
) -> Json<McpResponse> {
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
    match request.method.as_str() {
        "initialize" => handle_initialize(request),
        "tools/list" => handle_tools_list(request),
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
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "serverInfo": {
                "name": "jouledb",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn handle_tools_list(request: &McpRequest) -> McpResponse {
    let tools = all_tool_definitions();
    McpResponse::success(request.id.clone(), serde_json::json!({ "tools": tools }))
}

fn handle_resources_list(request: &McpRequest) -> McpResponse {
    let resources = vec![
        McpResourceDef {
            uri: "jouledb://tables".to_string(),
            name: "Table Catalog".to_string(),
            description: "List of all tables in the database".to_string(),
            mime_type: "application/json".to_string(),
        },
        McpResourceDef {
            uri: "jouledb://energy".to_string(),
            name: "Energy State".to_string(),
            description: "Current energy consumption and budget state".to_string(),
            mime_type: "application/json".to_string(),
        },
        McpResourceDef {
            uri: "jouledb://timeseries".to_string(),
            name: "Time Series Metrics".to_string(),
            description: "Metrics stored in the joule-db-features TimeSeriesStore (TSLIST)".to_string(),
            mime_type: "application/json".to_string(),
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

pub fn all_tool_definitions() -> Vec<McpToolDef> {
    vec![
        // Existing DB tools
        McpToolDef {
            name: "db.query".to_string(),
            description: "Execute a SQL query against JouleDB".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "SQL query to execute" },
                    "params": { "type": "object", "description": "Named parameters" }
                },
                "required": ["sql"]
            }),
        },
        McpToolDef {
            name: "db.insert".to_string(),
            description: "Insert a row into a table".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": { "type": "string" },
                    "row": { "type": "object" }
                },
                "required": ["table", "row"]
            }),
        },
        McpToolDef {
            name: "db.list_tables".to_string(),
            description: "List all tables in the database".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "db.describe_table".to_string(),
            description: "Get column definitions for a table".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "table": { "type": "string" } },
                "required": ["table"]
            }),
        },
        McpToolDef {
            name: "db.create_table".to_string(),
            description: "Create a new table with DDL".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "sql": { "type": "string" } },
                "required": ["sql"]
            }),
        },
        McpToolDef {
            name: "db.drop_table".to_string(),
            description: "Drop a table".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "table": { "type": "string" } },
                "required": ["table"]
            }),
        },
        // Branch tools
        McpToolDef {
            name: "db.branch_create".to_string(),
            description: "Create a CoW database branch with optional energy budget".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Branch name" },
                    "energy_budget_uj": { "type": "integer", "description": "Energy budget in microjoules" }
                },
                "required": ["name"]
            }),
        },
        McpToolDef {
            name: "db.branch_list".to_string(),
            description: "List all database branches".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "db.branch_merge".to_string(),
            description: "Merge a branch back to its parent".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
        },
        McpToolDef {
            name: "db.branch_delete".to_string(),
            description: "Delete a database branch".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "name": { "type": "string" } },
                "required": ["name"]
            }),
        },
        // Vector tools
        McpToolDef {
            name: "db.vector_search".to_string(),
            description: "Similarity search on vector columns. Returns results + energy cost."
                .to_string(),
            input_schema: serde_json::json!({
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
        },
        McpToolDef {
            name: "db.vector_upsert".to_string(),
            description: "Insert or update vectors in a table".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "table": { "type": "string" },
                    "column": { "type": "string" },
                    "vectors": { "type": "array", "items": { "type": "object" } }
                },
                "required": ["table", "column", "vectors"]
            }),
        },
        // Tenant tools
        McpToolDef {
            name: "db.tenant_create".to_string(),
            description: "Create a new tenant with resource quotas".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "energy_budget_uj": { "type": "integer" }
                },
                "required": ["name"]
            }),
        },
        McpToolDef {
            name: "db.tenant_list".to_string(),
            description: "List all tenants and their resource usage".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        },
        // Memory tools
        McpToolDef {
            name: "db.memory_store".to_string(),
            description: "Store a memory (episodic, semantic, or working) with temporal metadata"
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Memory content" },
                    "memory_type": { "type": "string", "enum": ["episodic", "semantic", "working"] },
                    "metadata": { "type": "object" },
                    "agent_id": { "type": "string" }
                },
                "required": ["content"]
            }),
        },
        McpToolDef {
            name: "db.memory_recall".to_string(),
            description: "Recall memories by similarity with temporal decay scoring".to_string(),
            input_schema: serde_json::json!({
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
        },
        // Schema/Status tools
        McpToolDef {
            name: "db.schema_inspect".to_string(),
            description: "Full schema dump — all tables, columns, and types".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        },
        McpToolDef {
            name: "db.energy_budget".to_string(),
            description: "Check or set the energy budget for the current session".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "budget_uj": { "type": "integer", "description": "Set energy budget (microjoules)" }
                }
            }),
        },
        McpToolDef {
            name: "db.status".to_string(),
            description: "Server lifecycle state, uptime, energy stats".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        },
        // Workflow tools
        McpToolDef {
            name: "db.workflow_create".to_string(),
            description: "Create a durable workflow definition with energy-metered steps"
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Workflow name" },
                    "steps": { "type": "array", "items": { "type": "object" } },
                    "energy_budget_uj": { "type": "integer" }
                },
                "required": ["name", "steps"]
            }),
        },
        McpToolDef {
            name: "db.workflow_run".to_string(),
            description: "Execute a workflow definition by ID. Returns step results + energy cost."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "workflow_id": { "type": "string", "description": "Workflow definition ID" }
                },
                "required": ["workflow_id"]
            }),
        },
        McpToolDef {
            name: "db.workflow_status".to_string(),
            description: "Get the status of a workflow instance".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "instance_id": { "type": "string", "description": "Workflow instance ID" }
                },
                "required": ["instance_id"]
            }),
        },
        McpToolDef {
            name: "db.queue_publish".to_string(),
            description: "Publish a message to a durable queue topic".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "topic": { "type": "string", "description": "Queue topic name" },
                    "payload": { "type": "string", "description": "Message payload" }
                },
                "required": ["topic", "payload"]
            }),
        },
        McpToolDef {
            name: "db.edge_pop_list".to_string(),
            description: "List all Edge Points of Presence and their sync status".to_string(),
            input_schema: serde_json::json!({ "type": "object", "properties": {} }),
        },
    ]
}

// ============================================================================
// Stdio transport (for Claude Code / local agents)
// ============================================================================

/// Run the MCP stdio transport — reads JSON-RPC from stdin, writes to stdout.
/// This blocks the calling task until stdin is closed.
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
                    Ok(request) => handle_mcp_request(&request, tool_handler.as_deref()),
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
        assert_eq!(tools.len(), 24);

        // Verify all tools have required fields
        for tool in tools {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert!(tool["inputSchema"].is_object());
        }
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
        assert_eq!(resources.len(), 3);
        let uris: Vec<&str> = resources.iter().filter_map(|r| r["uri"].as_str()).collect();
        assert!(uris.contains(&"jouledb://tables"));
        assert!(uris.contains(&"jouledb://energy"));
        assert!(uris.contains(&"jouledb://timeseries"));
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
            fn execute(
                &self,
                request: &QueryRequest,
            ) -> Result<QueryResponse, QueryErrorResponse> {
                Ok(QueryResponse {
                    columns: vec!["result".into()],
                    rows: vec![vec![serde_json::json!(format!("executed: {}", request.sql))]],
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
            fn execute(
                &self,
                request: &QueryRequest,
            ) -> Result<QueryResponse, QueryErrorResponse> {
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
        let text = resp.result.unwrap()["contents"][0]["text"].as_str().unwrap().to_string();
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
