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

use crate::query::{QueryExecutor, QueryRequest, QueryResponse};
use inv_mcp_core::{
    Layer, McpError, McpToolHandler, ParameterSchema, ToolCallRequest, ToolCallResponse,
    ToolDefinition,
};
use std::sync::Arc;

/// MCP tool handler for the database layer.
///
/// Wraps a `QueryExecutor` for SQL operations and provides KV access
/// through SQL commands (GET/SET are mapped to SELECT/INSERT).
pub struct DatabaseToolHandler {
    query_executor: Arc<dyn QueryExecutor>,
}

impl DatabaseToolHandler {
    pub fn new(query_executor: Arc<dyn QueryExecutor>) -> Self {
        Self { query_executor }
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

#[async_trait::async_trait]
impl McpToolHandler for DatabaseToolHandler {
    async fn handle(&self, request: ToolCallRequest) -> ToolCallResponse {
        match request.tool.as_str() {
            "db.query" => self.handle_query(&request),
            "db.get" => self.handle_get(&request),
            "db.put" => self.handle_put(&request),
            "db.delete" => self.handle_delete(&request),
            "db.semantic_search" => self.handle_semantic_search(&request),
            "db.energy" => self.handle_energy(&request),
            _ => ToolCallResponse::error(request.id, McpError::tool_not_found(&request.tool)),
        }
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
