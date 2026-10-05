//! Bridge to invisible-os `McpToolHandler` trait.
//!
//! Adapts joule-db-server's query and storage surface to the shared
//! `inv-mcp-core` protocol, enabling any layer to call database tools
//! via the `db.*` namespace:
//!
//! - `db.query` — execute SQL (SELECT, INSERT, UPDATE, DELETE, DDL)
//! - `db.get` — KV retrieve by key
//! - `db.put` — KV store with optional TTL
//! - `db.delete` — KV delete by key
//! - `db.semantic_search` — vector similarity search
//! - `db.energy` — energy metrics snapshot

use crate::agent_memory::{MemoryManager, RecallMemoryRequest, StoreMemoryRequest};
use crate::edge_pop::EdgePopManager;
use crate::query::{QueryExecutor, QueryRequest, QueryResponse};
use crate::scale_to_zero::ActivityTracker;
use crate::tenant::{CreateTenantRequest, TenantManager, TenantQuotas};
use crate::workflow::{WorkflowManager, WorkflowStep};
use inv_mcp_core::{
    Layer, McpError, McpToolHandler, ParameterSchema, ToolCallRequest, ToolCallResponse,
    ToolDefinition,
};
use joule_db_branch::manager::BranchManager;
use joule_db_branch::CreateBranchRequest;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// MCP tool handler for the database layer.
///
/// Wraps a `QueryExecutor` for SQL operations and provides KV access
/// through SQL commands (GET/SET are mapped to SELECT/INSERT).
pub struct DatabaseToolHandler {
    query_executor: Arc<dyn QueryExecutor>,
    branches: Arc<BranchManager>,
    tenants: Arc<TenantManager>,
    memory: Arc<MemoryManager>,
    workflows: Arc<WorkflowManager>,
    edge: Arc<EdgePopManager>,
    activity: Arc<ActivityTracker>,
    /// Session energy budget in microjoules. Stored and reported; never a gate.
    reported_budget_uj: AtomicU64,
}

impl DatabaseToolHandler {
    pub fn new(query_executor: Arc<dyn QueryExecutor>) -> Self {
        Self {
            query_executor,
            branches: Arc::new(BranchManager::new(0)),
            tenants: Arc::new(TenantManager::new()),
            memory: Arc::new(MemoryManager::new()),
            workflows: Arc::new(WorkflowManager::new()),
            edge: Arc::new(EdgePopManager::new()),
            activity: Arc::new(ActivityTracker::new()),
            reported_budget_uj: AtomicU64::new(0),
        }
    }

    fn handle_query(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let sql = match request.arguments.get("sql").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "missing required parameter: sql".into(),
                        data: None,
                    },
                );
            }
        };

        let params = request
            .arguments
            .get("params")
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let args = request
            .arguments
            .get("args")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let query_req = QueryRequest {
            sql: sql.to_string(),
            params,
            args,
            explain: request
                .arguments
                .get("explain")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            limit: request
                .arguments
                .get("limit")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&query_req) {
            Ok(resp) => {
                let result = query_response_to_json(&resp);
                let mut response = ToolCallResponse::success(request.id.clone(), result);
                if let Some(joules) = resp.energy_joules {
                    response = response.with_energy(joules);
                }
                response
            }
            Err(err) => ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INTERNAL_ERROR,
                    message: err.message,
                    data: None,
                },
            ),
        }
    }

    fn handle_get(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let key = match request.arguments.get("key").and_then(|v| v.as_str()) {
            Some(k) => k,
            None => {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "missing required parameter: key".into(),
                        data: None,
                    },
                );
            }
        };

        let table = request
            .arguments
            .get("table")
            .and_then(|v| v.as_str())
            .unwrap_or("kv");

        // Execute as SQL: SELECT value FROM {table} WHERE key = '{key}'
        let sql = format!(
            "SELECT value FROM {} WHERE key = '{}'",
            sanitize_identifier(table),
            key.replace('\'', "''")
        );

        let query_req = QueryRequest {
            sql,
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: Some(1),
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&query_req) {
            Ok(resp) => {
                let value = resp.rows.first().and_then(|row| row.first()).cloned();

                let result = serde_json::json!({
                    "found": value.is_some(),
                    "value": value,
                });
                ToolCallResponse::success(request.id.clone(), result)
            }
            Err(err) => ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INTERNAL_ERROR,
                    message: err.message,
                    data: None,
                },
            ),
        }
    }

    fn handle_put(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let key = match request.arguments.get("key").and_then(|v| v.as_str()) {
            Some(k) => k,
            None => {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "missing required parameter: key".into(),
                        data: None,
                    },
                );
            }
        };

        let value = match request.arguments.get("value") {
            Some(v) => v,
            None => {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "missing required parameter: value".into(),
                        data: None,
                    },
                );
            }
        };

        let table = request
            .arguments
            .get("table")
            .and_then(|v| v.as_str())
            .unwrap_or("kv");

        let value_str = match value.as_str() {
            Some(s) => s.replace('\'', "''"),
            None => value.to_string().replace('\'', "''"),
        };

        // INSERT OR REPLACE
        let sql = format!(
            "INSERT INTO {} (key, value) VALUES ('{}', '{}') ON CONFLICT (key) DO UPDATE SET value = '{}'",
            sanitize_identifier(table),
            key.replace('\'', "''"),
            value_str,
            value_str,
        );

        let query_req = QueryRequest {
            sql,
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&query_req) {
            Ok(resp) => {
                let mut response = ToolCallResponse::success(
                    request.id.clone(),
                    serde_json::json!({ "stored": true }),
                );
                if let Some(joules) = resp.energy_joules {
                    response = response.with_energy(joules);
                }
                response
            }
            Err(err) => ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INTERNAL_ERROR,
                    message: err.message,
                    data: None,
                },
            ),
        }
    }

    fn handle_delete(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let key = match request.arguments.get("key").and_then(|v| v.as_str()) {
            Some(k) => k,
            None => {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "missing required parameter: key".into(),
                        data: None,
                    },
                );
            }
        };

        let table = request
            .arguments
            .get("table")
            .and_then(|v| v.as_str())
            .unwrap_or("kv");

        let sql = format!(
            "DELETE FROM {} WHERE key = '{}'",
            sanitize_identifier(table),
            key.replace('\'', "''")
        );

        let query_req = QueryRequest {
            sql,
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&query_req) {
            Ok(resp) => {
                let deleted = resp.affected_rows.unwrap_or(0) > 0;
                ToolCallResponse::success(
                    request.id.clone(),
                    serde_json::json!({ "deleted": deleted }),
                )
            }
            Err(err) => ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INTERNAL_ERROR,
                    message: err.message,
                    data: None,
                },
            ),
        }
    }

    fn handle_semantic_search(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let query = match request.arguments.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "missing required parameter: query".into(),
                        data: None,
                    },
                );
            }
        };

        let k = request
            .arguments
            .get("k")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let index = request
            .arguments
            .get("index")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        // Use the EMBED SIMILAR command from features_bridge
        let sql = format!(
            "EMBED SIMILAR '{}' {} {}",
            query.replace('\'', "''"),
            k,
            index
        );

        let query_req = QueryRequest {
            sql,
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: None,
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&query_req) {
            Ok(resp) => {
                let result = query_response_to_json(&resp);
                let mut response = ToolCallResponse::success(request.id.clone(), result);
                if let Some(joules) = resp.energy_joules {
                    response = response.with_energy(joules);
                }
                response
            }
            Err(err) => ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INTERNAL_ERROR,
                    message: err.message,
                    data: None,
                },
            ),
        }
    }

    fn handle_energy(&self, request: &ToolCallRequest) -> ToolCallResponse {
        // Energy status is best-effort — read from executor's last snapshot
        let sql = "SELECT 1".to_string();
        let query_req = QueryRequest {
            sql,
            params: Default::default(),
            args: vec![],
            explain: false,
            limit: Some(1),
            session_id: None,
            query_timeout_ms: None,
            branch_id: None,
            tenant_id: None,
        };

        match self.query_executor.execute(&query_req) {
            Ok(resp) => {
                let result = serde_json::json!({
                    "energy_joules": resp.energy_joules,
                    "power_watts": resp.power_watts,
                    "device_target": resp.device_target,
                    "algorithm_type": resp.algorithm_type,
                });
                ToolCallResponse::success(request.id.clone(), result)
            }
            Err(_) => ToolCallResponse::success(
                request.id.clone(),
                serde_json::json!({ "status": "unavailable" }),
            ),
        }
    }
}

/// Convert a QueryResponse to a JSON value suitable for MCP.
fn query_response_to_json(resp: &QueryResponse) -> serde_json::Value {
    serde_json::json!({
        "columns": resp.columns,
        "rows": resp.rows,
        "affected_rows": resp.affected_rows,
        "execution_time_ms": resp.execution_time_ms,
        "truncated": resp.truncated,
        "warnings": resp.warnings,
        "energy_joules": resp.energy_joules,
        "device_target": resp.device_target,
    })
}

/// Sanitize a SQL identifier (table name) — only allow alphanumeric and underscore.
fn sanitize_identifier(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Tool definitions for the database layer.
fn db_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "db.query".into(),
            description: "Execute a SQL query against JouleDB. Supports SELECT, INSERT, UPDATE, \
                DELETE, CREATE TABLE, and JouleDB extensions (VECTOR, EMBED, TSQUERY, etc.). \
                Returns columns, rows, energy cost, and device target."
                .into(),
            parameters: vec![
                ParameterSchema {
                    name: "sql".into(),
                    schema_type: "string".into(),
                    description: "SQL query to execute".into(),
                    required: true,
                    default: None,
                },
                ParameterSchema {
                    name: "params".into(),
                    schema_type: "object".into(),
                    description: "Named parameters for the query".into(),
                    required: false,
                    default: None,
                },
                ParameterSchema {
                    name: "args".into(),
                    schema_type: "array".into(),
                    description: "Positional parameters ($1, $2, ...)".into(),
                    required: false,
                    default: None,
                },
                ParameterSchema {
                    name: "explain".into(),
                    schema_type: "boolean".into(),
                    description: "Return execution plan instead of results".into(),
                    required: false,
                    default: Some(serde_json::json!(false)),
                },
                ParameterSchema {
                    name: "limit".into(),
                    schema_type: "number".into(),
                    description: "Maximum rows to return".into(),
                    required: false,
                    default: None,
                },
            ],
            layer: Layer::Database,
            energy_estimate_joules: Some(0.001),
        },
        ToolDefinition {
            name: "db.get".into(),
            description: "Retrieve a value by key from a KV table.".into(),
            parameters: vec![
                ParameterSchema {
                    name: "key".into(),
                    schema_type: "string".into(),
                    description: "The key to look up".into(),
                    required: true,
                    default: None,
                },
                ParameterSchema {
                    name: "table".into(),
                    schema_type: "string".into(),
                    description: "KV table name (default: 'kv')".into(),
                    required: false,
                    default: Some(serde_json::json!("kv")),
                },
            ],
            layer: Layer::Database,
            energy_estimate_joules: Some(0.0001),
        },
        ToolDefinition {
            name: "db.put".into(),
            description: "Store a key-value pair. Creates or updates the entry.".into(),
            parameters: vec![
                ParameterSchema {
                    name: "key".into(),
                    schema_type: "string".into(),
                    description: "The key to store".into(),
                    required: true,
                    default: None,
                },
                ParameterSchema {
                    name: "value".into(),
                    schema_type: "string".into(),
                    description: "The value to store".into(),
                    required: true,
                    default: None,
                },
                ParameterSchema {
                    name: "table".into(),
                    schema_type: "string".into(),
                    description: "KV table name (default: 'kv')".into(),
                    required: false,
                    default: Some(serde_json::json!("kv")),
                },
            ],
            layer: Layer::Database,
            energy_estimate_joules: Some(0.0002),
        },
        ToolDefinition {
            name: "db.delete".into(),
            description: "Delete a key-value pair by key.".into(),
            parameters: vec![
                ParameterSchema {
                    name: "key".into(),
                    schema_type: "string".into(),
                    description: "The key to delete".into(),
                    required: true,
                    default: None,
                },
                ParameterSchema {
                    name: "table".into(),
                    schema_type: "string".into(),
                    description: "KV table name (default: 'kv')".into(),
                    required: false,
                    default: Some(serde_json::json!("kv")),
                },
            ],
            layer: Layer::Database,
            energy_estimate_joules: Some(0.0001),
        },
        ToolDefinition {
            name: "db.semantic_search".into(),
            description: "Search by meaning using JouleDB's embedding similarity engine. \
                Finds the top-k most similar entries to a natural language query."
                .into(),
            parameters: vec![
                ParameterSchema {
                    name: "query".into(),
                    schema_type: "string".into(),
                    description: "Natural language search query".into(),
                    required: true,
                    default: None,
                },
                ParameterSchema {
                    name: "k".into(),
                    schema_type: "number".into(),
                    description: "Number of results to return (default: 5)".into(),
                    required: false,
                    default: Some(serde_json::json!(5)),
                },
                ParameterSchema {
                    name: "index".into(),
                    schema_type: "string".into(),
                    description: "Embedding index name (default: 'default')".into(),
                    required: false,
                    default: Some(serde_json::json!("default")),
                },
            ],
            layer: Layer::Database,
            energy_estimate_joules: Some(0.005),
        },
        ToolDefinition {
            name: "db.energy".into(),
            description: "Get current energy metrics: power draw, thermal state, device \
                utilization. Reports the energy cost of the most recent operation."
                .into(),
            parameters: vec![],
            layer: Layer::Database,
            energy_estimate_joules: Some(0.00001),
        },
    ]
}

impl DatabaseToolHandler {
    /// Synchronously dispatch a tool call for the MCP JSON-RPC transport.
    ///
    /// Known `db.*` tools are executed via the query executor; unknown tools
    /// return a tool-not-found error (no placeholder success).
    pub fn call_tool(&self, request: ToolCallRequest) -> ToolCallResponse {
        match request.tool.as_str() {
            "db.query" => self.handle_query(&request),
            "db.get" => self.handle_get(&request),
            "db.put" => self.handle_put(&request),
            "db.delete" => self.handle_delete(&request),
            "db.semantic_search" => self.handle_semantic_search(&request),
            "db.energy" => self.handle_energy(&request),
            "db.list_tables" => self.query_sql(&request.id, "SELECT table_name FROM information_schema.tables".into()),
            "db.describe_table" => self.handle_describe_table(&request),
            "db.create_table" => self.handle_query(&request),
            "db.drop_table" => self.handle_drop_table(&request),
            "db.insert" => self.handle_insert(&request),
            "db.schema_inspect" => self.query_sql(
                &request.id,
                "SELECT table_name, column_name, data_type, ordinal_position FROM information_schema.columns".into(),
            ),
            "db.vector_search" => self.handle_vector_search(&request),
            "db.vector_upsert" => self.handle_vector_upsert(&request),
            "db.branch_create" => self.handle_branch_create(&request),
            "db.branch_list" => self.handle_branch_list(&request),
            "db.branch_merge" => self.handle_branch_merge(&request),
            "db.branch_delete" => self.handle_branch_delete(&request),
            "db.tenant_create" => self.handle_tenant_create(&request),
            "db.tenant_list" => self.handle_tenant_list(&request),
            "db.memory_store" => self.handle_memory_store(&request),
            "db.memory_recall" => self.handle_memory_recall(&request),
            "db.workflow_create" => self.handle_workflow_create(&request),
            "db.workflow_run" => self.handle_workflow_run(&request),
            "db.workflow_status" => self.handle_workflow_status(&request),
            "db.queue_publish" => self.handle_queue_publish(&request),
            "db.edge_pop_list" => self.handle_edge_pop_list(&request),
            "db.energy_budget" => self.handle_energy_budget(&request),
            "db.status" => self.handle_status(&request),
            _ => ToolCallResponse::error(request.id, McpError::tool_not_found(&request.tool)),
        }
    }
}


impl DatabaseToolHandler {
    fn missing(id: &str, message: &str) -> ToolCallResponse {
        ToolCallResponse::error(
            id.to_string(),
            McpError {
                code: McpError::INVALID_PARAMS,
                message: message.to_string(),
                data: None,
            },
        )
    }

    fn type_error(id: &str, err: impl std::fmt::Display) -> ToolCallResponse {
        ToolCallResponse::error(
            id.to_string(),
            McpError {
                code: McpError::INTERNAL_ERROR,
                message: err.to_string(),
                data: None,
            },
        )
    }

    fn json_ok(id: &str, value: impl serde::Serialize) -> ToolCallResponse {
        match serde_json::to_value(value) {
            Ok(v) => ToolCallResponse::success(id.to_string(), v),
            Err(err) => Self::type_error(id, err),
        }
    }

    fn handle_branch_create(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(name) = request.arguments.get("name").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: name");
        };
        let energy_budget_uj = request
            .arguments
            .get("energy_budget_uj")
            .and_then(|v| v.as_u64());
        match self.branches.create_branch(CreateBranchRequest {
            name: name.to_string(),
            parent: None,
            at_lsn: None,
            energy_budget_uj,
            description: None,
            tags: Vec::new(),
        }) {
            Ok(info) => Self::json_ok(&request.id, info),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_branch_list(&self, request: &ToolCallRequest) -> ToolCallResponse {
        match self.branches.list_branches() {
            Ok(list) => Self::json_ok(&request.id, list),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_branch_merge(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(name) = request.arguments.get("name").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: name");
        };
        // delete_after stays false so the branch remains queryable after the merge.
        match self.branches.merge_branch(name, false) {
            Ok(result) => Self::json_ok(&request.id, result),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_branch_delete(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(name) = request.arguments.get("name").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: name");
        };
        match self.branches.delete_branch(name) {
            Ok(info) => Self::json_ok(&request.id, info),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_tenant_create(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(name) = request.arguments.get("name").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: name");
        };
        let mut quotas = TenantQuotas::default();
        if let Some(uj) = request
            .arguments
            .get("energy_budget_uj")
            .and_then(|v| v.as_u64())
        {
            // Recorded on the tenant for reporting. Creation itself always proceeds.
            quotas.energy_budget_uj = Some(uj);
        }
        match self.tenants.create_tenant(CreateTenantRequest {
            name: name.to_string(),
            quotas: Some(quotas),
        }) {
            Ok(info) => Self::json_ok(&request.id, info),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_tenant_list(&self, request: &ToolCallRequest) -> ToolCallResponse {
        match self.tenants.list_tenants() {
            Ok(list) => Self::json_ok(&request.id, list),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_memory_store(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(content) = request.arguments.get("content").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: content");
        };
        let mut metadata = std::collections::HashMap::new();
        if let Some(obj) = request.arguments.get("metadata").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                let text = v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string());
                metadata.insert(k.clone(), text);
            }
        }
        let req = StoreMemoryRequest {
            content: content.to_string(),
            memory_type: request
                .arguments
                .get("memory_type")
                .and_then(|v| v.as_str())
                .unwrap_or("episodic")
                .to_string(),
            metadata,
            agent_id: request
                .arguments
                .get("agent_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            decay_rate: None,
        };
        match self.memory.store(&req, "default") {
            Ok(id) => ToolCallResponse::success(
                request.id.clone(),
                serde_json::json!({ "id": id }),
            ),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_memory_recall(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(query) = request.arguments.get("query").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: query");
        };
        let req = RecallMemoryRequest {
            query: query.to_string(),
            k: request
                .arguments
                .get("k")
                .and_then(|v| v.as_u64())
                .unwrap_or(5) as usize,
            time_range_hours: None,
            half_life_hours: request
                .arguments
                .get("half_life_hours")
                .and_then(|v| v.as_f64()),
            memory_type: request
                .arguments
                .get("memory_type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            agent_id: request
                .arguments
                .get("agent_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        };
        match self.memory.recall(&req, "default") {
            Ok(hits) => Self::json_ok(&request.id, hits),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_workflow_create(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(name) = request.arguments.get("name").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: name");
        };
        let Some(steps_val) = request.arguments.get("steps") else {
            return Self::missing(&request.id, "missing required parameter: steps");
        };
        let steps: Vec<WorkflowStep> = match serde_json::from_value(steps_val.clone()) {
            Ok(steps) => steps,
            Err(err) => {
                return Self::missing(
                    &request.id,
                    &format!("steps could not be decoded as workflow steps: {err}"),
                );
            }
        };
        // energy_budget_uj is reported back but not installed as a run() abort.
        let reported = request
            .arguments
            .get("energy_budget_uj")
            .and_then(|v| v.as_u64());
        match self
            .workflows
            .create_definition(name.to_string(), steps, None, None)
        {
            Ok(def) => {
                let mut value = serde_json::to_value(&def).unwrap_or(serde_json::json!({}));
                if let Some(obj) = value.as_object_mut() {
                    obj.insert(
                        "reported_energy_budget_uj".into(),
                        serde_json::json!(reported),
                    );
                    obj.insert(
                        "energy_enforced".into(),
                        serde_json::Value::Bool(false),
                    );
                }
                ToolCallResponse::success(request.id.clone(), value)
            }
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_workflow_run(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(id) = request
            .arguments
            .get("workflow_id")
            .and_then(|v| v.as_str())
        else {
            return Self::missing(&request.id, "missing required parameter: workflow_id");
        };
        match self.workflows.run(id) {
            Ok(instance) => Self::json_ok(&request.id, instance),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_workflow_status(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(id) = request
            .arguments
            .get("instance_id")
            .and_then(|v| v.as_str())
        else {
            return Self::missing(&request.id, "missing required parameter: instance_id");
        };
        match self.workflows.get_instance(id) {
            Ok(instance) => Self::json_ok(&request.id, instance),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_queue_publish(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(topic) = request.arguments.get("topic").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: topic");
        };
        let Some(payload) = request.arguments.get("payload").and_then(|v| v.as_str()) else {
            return Self::missing(&request.id, "missing required parameter: payload");
        };
        match self.workflows.publish(topic, payload.to_string()) {
            Ok(message) => Self::json_ok(&request.id, message),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_edge_pop_list(&self, request: &ToolCallRequest) -> ToolCallResponse {
        match self.edge.list() {
            Ok(pops) => Self::json_ok(&request.id, pops),
            Err(err) => Self::type_error(&request.id, err),
        }
    }

    fn handle_energy_budget(&self, request: &ToolCallRequest) -> ToolCallResponse {
        if let Some(uj) = request
            .arguments
            .get("budget_uj")
            .and_then(|v| v.as_u64())
        {
            self.reported_budget_uj.store(uj, Ordering::Relaxed);
        }
        let budget_uj = self.reported_budget_uj.load(Ordering::Relaxed);
        // One real CPU reduction so the receipt has measured joules. Not a gate.
        let receipt = crate::fabric::dispatch_sum(
            &[1.0],
            joule_db_energy::AlgorithmType::BTree,
            true,
        );
        ToolCallResponse::success(
            request.id.clone(),
            serde_json::json!({
                "budget_uj": budget_uj,
                "enforced": false,
                "device": receipt.device,
                "joules": receipt.joules,
                "note": "energy budget is reported, not enforced",
            }),
        )
        .with_energy(receipt.joules)
    }

    fn handle_status(&self, request: &ToolCallRequest) -> ToolCallResponse {
        self.activity.touch();
        Self::json_ok(&request.id, self.activity.status())
    }

    /// Read an MCP resource URI against the live catalog and feature stores.
    ///
    /// Supported URIs:
    /// - `jouledb://tables`
    /// - `jouledb://tables/{table}`
    /// - `jouledb://energy`
    /// - `jouledb://timeseries`
    /// - `jouledb://timeseries/{metric}?start={ns}&end={ns}`
    pub fn read_resource(&self, uri: &str) -> ToolCallResponse {
        let (path, query) = match uri.split_once('?') {
            Some((path, query)) => (path, Some(query)),
            None => (uri, None),
        };
        if path == "jouledb://tables" {
            return self.query_sql(
                "resource",
                "SELECT table_name FROM information_schema.tables".into(),
            );
        }
        if let Some(table) = path.strip_prefix("jouledb://tables/") {
            let table = sanitize_identifier(table);
            if table.is_empty() {
                return ToolCallResponse::error(
                    "resource".into(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "table name is empty after sanitization".into(),
                        data: None,
                    },
                );
            }
            return self.query_sql(
                "resource",
                format!(
                    "SELECT column_name, data_type, ordinal_position FROM information_schema.columns WHERE table_name = '{table}'"
                ),
            );
        }
        if path == "jouledb://energy" {
            let req = ToolCallRequest {
                id: "resource".into(),
                tool: "db.energy".into(),
                arguments: Default::default(),
            };
            return self.handle_energy(&req);
        }
        if path == "jouledb://timeseries" {
            return self.query_sql("resource", "TSLIST".into());
        }
        if let Some(metric) = path.strip_prefix("jouledb://timeseries/") {
            let metric = sanitize_metric(metric);
            if metric.is_empty() {
                return ToolCallResponse::error(
                    "resource".into(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: "metric name is empty".into(),
                        data: None,
                    },
                );
            }
            let query = query.unwrap_or("");
            let start = query_param(query, "start");
            let end = query_param(query, "end");
            match (start, end) {
                (Some(start), Some(end))
                    if start.parse::<i64>().is_ok() && end.parse::<i64>().is_ok() =>
                {
                    return self.query_sql(
                        "resource",
                        format!("TSQUERY {metric} {start} {end}"),
                    );
                }
                _ => {
                    return ToolCallResponse::error(
                        "resource".into(),
                        McpError {
                            code: McpError::INVALID_PARAMS,
                            message: "jouledb://timeseries/{metric} requires integer start and end query parameters (nanoseconds)".into(),
                            data: None,
                        },
                    );
                }
            }
        }
        ToolCallResponse::error(
            "resource".into(),
            McpError {
                code: -32002,
                message: format!("resource not found: {uri}"),
                data: None,
            },
        )
    }

    fn query_sql(&self, id: &str, sql: String) -> ToolCallResponse {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("sql".into(), serde_json::Value::String(sql));
        let request = ToolCallRequest {
            id: id.to_string(),
            tool: "db.query".into(),
            arguments,
        };
        self.handle_query(&request)
    }

    fn handle_describe_table(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(table) = request.arguments.get("table").and_then(|v| v.as_str()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: table".into(),
                    data: None,
                },
            );
        };
        let table = sanitize_identifier(table);
        self.query_sql(
            &request.id,
            format!(
                "SELECT column_name, data_type, ordinal_position FROM information_schema.columns WHERE table_name = '{table}'"
            ),
        )
    }

    fn handle_drop_table(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(table) = request.arguments.get("table").and_then(|v| v.as_str()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: table".into(),
                    data: None,
                },
            );
        };
        let table = sanitize_identifier(table);
        if table.is_empty() {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "table name is empty after sanitization".into(),
                    data: None,
                },
            );
        }
        self.query_sql(&request.id, format!("DROP TABLE {table}"))
    }

    fn handle_insert(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(table) = request.arguments.get("table").and_then(|v| v.as_str()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: table".into(),
                    data: None,
                },
            );
        };
        let Some(row) = request.arguments.get("row").and_then(|v| v.as_object()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: row".into(),
                    data: None,
                },
            );
        };
        let table = sanitize_identifier(table);
        if table.is_empty() || row.is_empty() {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "insert requires a table name and at least one column".into(),
                    data: None,
                },
            );
        }
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (key, value) in row {
            let col = sanitize_identifier(key);
            if col.is_empty() {
                continue;
            }
            cols.push(col);
            vals.push(sql_literal(value));
        }
        if cols.is_empty() {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "row had no usable column names".into(),
                    data: None,
                },
            );
        }
        self.query_sql(
            &request.id,
            format!(
                "INSERT INTO {table} ({}) VALUES ({})",
                cols.join(", "),
                vals.join(", ")
            ),
        )
    }

    fn handle_vector_search(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(index) = request
            .arguments
            .get("table")
            .and_then(|v| v.as_str())
            .or_else(|| request.arguments.get("index").and_then(|v| v.as_str()))
        else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: table".into(),
                    data: None,
                },
            );
        };
        let Some(query) = request.arguments.get("query").and_then(|v| v.as_array()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: query".into(),
                    data: None,
                },
            );
        };
        let k = request
            .arguments
            .get("k")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);
        let index = sanitize_identifier(index);
        let literal = query
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "0".into())
            })
            .collect::<Vec<_>>()
            .join(",");
        self.query_sql(
            &request.id,
            format!("VECTOR SEARCH {index} [{literal}] {k}"),
        )
    }

    fn handle_vector_upsert(&self, request: &ToolCallRequest) -> ToolCallResponse {
        let Some(index) = request.arguments.get("table").and_then(|v| v.as_str()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: table".into(),
                    data: None,
                },
            );
        };
        let Some(vectors) = request.arguments.get("vectors").and_then(|v| v.as_array()) else {
            return ToolCallResponse::error(
                request.id.clone(),
                McpError {
                    code: McpError::INVALID_PARAMS,
                    message: "missing required parameter: vectors".into(),
                    data: None,
                },
            );
        };
        let index = sanitize_identifier(index);
        let mut stored = 0u64;
        let mut last_energy = None;
        for (i, item) in vectors.iter().enumerate() {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| sanitize_identifier(s))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("v{i}"));
            let values = item
                .get("values")
                .or_else(|| item.get("vector"))
                .and_then(|v| v.as_array());
            let Some(values) = values else {
                return ToolCallResponse::error(
                    request.id.clone(),
                    McpError {
                        code: McpError::INVALID_PARAMS,
                        message: format!("vector {i} is missing values"),
                        data: None,
                    },
                );
            };
            let literal = values
                .iter()
                .map(|v| v.as_f64().map(|n| n.to_string()).unwrap_or_else(|| "0".into()))
                .collect::<Vec<_>>()
                .join(",");
            let resp = self.query_sql(
                &request.id,
                format!("VECTOR INSERT {index} {id} [{literal}]"),
            );
            if let Some(err) = resp.error {
                return ToolCallResponse::error(request.id.clone(), err);
            }
            last_energy = resp.energy_joules.or(last_energy);
            stored += 1;
        }
        let mut response = ToolCallResponse::success(
            request.id.clone(),
            serde_json::json!({ "stored": stored }),
        );
        if let Some(joules) = last_energy {
            response = response.with_energy(joules);
        }
        response
    }
}

fn sql_literal(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "NULL".into(),
        serde_json::Value::Bool(b) => if *b { "TRUE".into() } else { "FALSE".into() },
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => format!("'{}'", s.replace('\'', "''")),
        other => format!("'{}'", other.to_string().replace('\'', "''")),
    }
}

fn sanitize_metric(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(*c, '_' | '.' | '-'))
        .collect()
}

fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        if k == key { Some(v) } else { None }
    })
}

#[async_trait::async_trait]
impl McpToolHandler for DatabaseToolHandler {
    async fn handle(&self, request: ToolCallRequest) -> ToolCallResponse {
        self.call_tool(request)
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        db_tool_definitions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Mock query executor for testing the MCP bridge.
    struct MockQueryExecutor;

    impl QueryExecutor for MockQueryExecutor {
        fn execute(
            &self,
            request: &QueryRequest,
        ) -> Result<QueryResponse, crate::query::QueryErrorResponse> {
            // Return a simple response for any query
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
            })
        }
    }

    fn make_handler() -> DatabaseToolHandler {
        DatabaseToolHandler::new(Arc::new(MockQueryExecutor))
    }

    #[test]
    fn tool_definitions_have_db_namespace() {
        let handler = make_handler();
        let tools = handler.tools();

        assert_eq!(tools.len(), 6);

        for tool in &tools {
            assert!(
                tool.name.starts_with("db."),
                "tool {} should have db.* namespace",
                tool.name
            );
            assert_eq!(tool.layer, Layer::Database);
        }
    }

    #[test]
    fn tool_names_match_expected() {
        let handler = make_handler();
        let tools = handler.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        assert!(names.contains(&"db.query"));
        assert!(names.contains(&"db.get"));
        assert!(names.contains(&"db.put"));
        assert!(names.contains(&"db.delete"));
        assert!(names.contains(&"db.semantic_search"));
        assert!(names.contains(&"db.energy"));
    }

    #[test]
    fn branch_tenant_memory_workflow_queue_edge_status_call_real_types() {
        let handler = make_handler();
        let mut args = std::collections::HashMap::new();
        args.insert("name".into(), serde_json::json!("dev"));
        let created = handler.call_tool(ToolCallRequest {
            id: "1".into(),
            tool: "db.branch_create".into(),
            arguments: args,
        });
        assert!(created.error.is_none(), "{:?}", created.error);
        assert_eq!(
            created.result.as_ref().and_then(|v| v.get("name")).and_then(|v| v.as_str()),
            Some("dev")
        );

        let listed = handler.call_tool(ToolCallRequest {
            id: "2".into(),
            tool: "db.branch_list".into(),
            arguments: std::collections::HashMap::new(),
        });
        assert!(listed.error.is_none(), "{:?}", listed.error);
        let names = listed.result.unwrap();
        assert!(names.as_array().unwrap().iter().any(|b| b.get("name").and_then(|n| n.as_str()) == Some("dev")));

        let mut targs = std::collections::HashMap::new();
        targs.insert("name".into(), serde_json::json!("acme"));
        let tenant = handler.call_tool(ToolCallRequest {
            id: "3".into(),
            tool: "db.tenant_create".into(),
            arguments: targs,
        });
        assert!(tenant.error.is_none(), "{:?}", tenant.error);

        let mut margs = std::collections::HashMap::new();
        margs.insert("content".into(), serde_json::json!("remember the fallback"));
        let stored = handler.call_tool(ToolCallRequest {
            id: "4".into(),
            tool: "db.memory_store".into(),
            arguments: margs,
        });
        assert!(stored.error.is_none(), "{:?}", stored.error);

        let mut rargs = std::collections::HashMap::new();
        rargs.insert("query".into(), serde_json::json!("fallback"));
        let recalled = handler.call_tool(ToolCallRequest {
            id: "5".into(),
            tool: "db.memory_recall".into(),
            arguments: rargs,
        });
        assert!(recalled.error.is_none(), "{:?}", recalled.error);

        let mut wargs = std::collections::HashMap::new();
        wargs.insert("name".into(), serde_json::json!("ping"));
        wargs.insert(
            "steps".into(),
            serde_json::json!([{
                "label": "one",
                "operation": { "sql": "SELECT 1" },
                "depends_on": [],
                "timeout_ms": null,
                "retry": null
            }]),
        );
        wargs.insert("energy_budget_uj".into(), serde_json::json!(1));
        let wf = handler.call_tool(ToolCallRequest {
            id: "6".into(),
            tool: "db.workflow_create".into(),
            arguments: wargs,
        });
        assert!(wf.error.is_none(), "{:?}", wf.error);
        let wf_id = wf.result.as_ref().unwrap().get("id").unwrap().as_str().unwrap().to_string();
        let mut run_args = std::collections::HashMap::new();
        run_args.insert("workflow_id".into(), serde_json::json!(wf_id));
        let ran = handler.call_tool(ToolCallRequest {
            id: "7".into(),
            tool: "db.workflow_run".into(),
            arguments: run_args,
        });
        assert!(ran.error.is_none(), "{:?}", ran.error);

        let mut qargs = std::collections::HashMap::new();
        qargs.insert("topic".into(), serde_json::json!("jobs"));
        qargs.insert("payload".into(), serde_json::json!("hello"));
        let queued = handler.call_tool(ToolCallRequest {
            id: "8".into(),
            tool: "db.queue_publish".into(),
            arguments: qargs,
        });
        assert!(queued.error.is_none(), "{:?}", queued.error);
        assert_eq!(queued.result.as_ref().unwrap().get("topic").and_then(|v| v.as_str()), Some("jobs"));

        let pops = handler.call_tool(ToolCallRequest {
            id: "9".into(),
            tool: "db.edge_pop_list".into(),
            arguments: std::collections::HashMap::new(),
        });
        assert!(pops.error.is_none(), "{:?}", pops.error);
        assert!(pops.result.unwrap().is_array());

        let status = handler.call_tool(ToolCallRequest {
            id: "10".into(),
            tool: "db.status".into(),
            arguments: std::collections::HashMap::new(),
        });
        assert!(status.error.is_none(), "{:?}", status.error);
        assert!(status.result.unwrap().get("state").is_some());

        let mut bargs = std::collections::HashMap::new();
        bargs.insert("budget_uj".into(), serde_json::json!(999));
        let budget = handler.call_tool(ToolCallRequest {
            id: "11".into(),
            tool: "db.energy_budget".into(),
            arguments: bargs,
        });
        assert!(budget.error.is_none(), "{:?}", budget.error);
        let body = budget.result.unwrap();
        assert_eq!(body.get("enforced").and_then(|v| v.as_bool()), Some(false));
        assert_eq!(body.get("budget_uj").and_then(|v| v.as_u64()), Some(999));
        assert!(body.get("joules").and_then(|v| v.as_f64()).unwrap() >= 0.0);
    }

    #[test]
    fn namespace_extraction() {
        let handler = make_handler();
        for tool in handler.tools() {
            assert_eq!(tool.namespace(), "db");
        }
    }

    #[test]
    fn energy_estimates_present() {
        let handler = make_handler();
        for tool in handler.tools() {
            assert!(
                tool.energy_estimate_joules.is_some(),
                "tool {} should have energy estimate",
                tool.name
            );
        }
    }

    #[tokio::test]
    async fn handle_query_success() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "q-1".into(),
            tool: "db.query".into(),
            arguments: HashMap::from([("sql".into(), serde_json::json!("SELECT * FROM users"))]),
        };

        let response = handler.handle(request).await;
        assert!(response.is_ok());
        assert_eq!(response.id, "q-1");
        assert!(response.energy_joules.is_some());

        let result = response.result.unwrap();
        assert!(result["columns"].is_array());
        assert!(result["rows"].is_array());
    }

    #[tokio::test]
    async fn handle_query_missing_sql() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "q-2".into(),
            tool: "db.query".into(),
            arguments: HashMap::new(),
        };

        let response = handler.handle(request).await;
        assert!(!response.is_ok());
        assert_eq!(response.error.unwrap().code, McpError::INVALID_PARAMS);
    }

    #[tokio::test]
    async fn handle_get() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "g-1".into(),
            tool: "db.get".into(),
            arguments: HashMap::from([("key".into(), serde_json::json!("test_key"))]),
        };

        let response = handler.handle(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn handle_put() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "p-1".into(),
            tool: "db.put".into(),
            arguments: HashMap::from([
                ("key".into(), serde_json::json!("test_key")),
                ("value".into(), serde_json::json!("test_value")),
            ]),
        };

        let response = handler.handle(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn handle_delete() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "d-1".into(),
            tool: "db.delete".into(),
            arguments: HashMap::from([("key".into(), serde_json::json!("test_key"))]),
        };

        let response = handler.handle(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn handle_semantic_search() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "s-1".into(),
            tool: "db.semantic_search".into(),
            arguments: HashMap::from([("query".into(), serde_json::json!("find similar items"))]),
        };

        let response = handler.handle(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn handle_energy() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "e-1".into(),
            tool: "db.energy".into(),
            arguments: HashMap::new(),
        };

        let response = handler.handle(request).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn handle_unknown_tool() {
        let handler = make_handler();
        let request = ToolCallRequest {
            id: "u-1".into(),
            tool: "db.nonexistent".into(),
            arguments: HashMap::new(),
        };

        let response = handler.handle(request).await;
        assert!(!response.is_ok());
        assert_eq!(response.error.unwrap().code, McpError::TOOL_NOT_FOUND);
    }

    #[test]
    fn sanitize_identifier_strips_bad_chars() {
        assert_eq!(sanitize_identifier("my_table"), "my_table");
        assert_eq!(sanitize_identifier("my;table"), "mytable");
        assert_eq!(sanitize_identifier("DROP TABLE--"), "DROPTABLE");
    }
}
