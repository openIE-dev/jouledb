//! MCP Transport Layer for JouleDB
//!
//! Provides stdio and SSE transports for the Model Context Protocol,
//! enabling AI agents (Claude, GPT, etc.) to interact with JouleDB
//! via standard MCP JSON-RPC messages.

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::pin::Pin;
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
}

impl McpSseState {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(128);
        Self { sender }
    }
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
    let response = handle_mcp_request(&request);

    // Also broadcast the response over SSE
    if let Ok(json) = serde_json::to_string(&response) {
        let _ = state.sender.send(json);
    }

    Json(response)
}

// ============================================================================
// MCP request dispatcher
// ============================================================================

fn handle_mcp_request(request: &McpRequest) -> McpResponse {
    match request.method.as_str() {
        "initialize" => handle_initialize(request),
        "tools/list" => handle_tools_list(request),
        "resources/list" => handle_resources_list(request),
        "tools/call" => handle_tools_call(request),
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
    ];
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({ "resources": resources }),
    )
}

fn handle_tools_call(request: &McpRequest) -> McpResponse {
    let tool_name = request
        .params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // For now, return a placeholder — actual tool execution is wired through mcp_bridge.rs
    McpResponse::success(
        request.id.clone(),
        serde_json::json!({
            "content": [{
                "type": "text",
                "text": format!("Tool '{}' called. Wire through mcp_bridge for full execution.", tool_name)
            }],
            "energy_joules": 0.0
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
pub async fn run_stdio_transport() {
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
                    Ok(request) => handle_mcp_request(&request),
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
        let resp = handle_mcp_request(&req);
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
        let resp = handle_mcp_request(&req);
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
        let resp = handle_mcp_request(&req);
        let result = resp.result.unwrap();
        let resources = result["resources"].as_array().unwrap();
        assert_eq!(resources.len(), 2);
    }

    #[test]
    fn test_ping() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(4)),
            method: "ping".to_string(),
            params: serde_json::json!({}),
        };
        let resp = handle_mcp_request(&req);
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
        let resp = handle_mcp_request(&req);
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_tools_call() {
        let req = McpRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(serde_json::json!(6)),
            method: "tools/call".to_string(),
            params: serde_json::json!({ "name": "db.status" }),
        };
        let resp = handle_mcp_request(&req);
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert!(result["content"].is_array());
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
}
