//! Shared MCP (Model Context Protocol) tool handler traits.
//!
//! Enables any layer to expose tools via a uniform `db.*`, `search.*`, etc.
//! namespace that MCP clients can discover and call.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which layer of the stack a tool belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Layer {
    Database,
    Search,
    Agent,
    Application,
    System,
    Mesh,
    Experience,
}

/// Schema for a single tool parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub name: String,
    pub schema_type: String,
    pub description: String,
    pub required: bool,
    pub default: Option<serde_json::Value>,
}

/// Definition of a tool that can be discovered by MCP clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ParameterSchema>,
    pub layer: Layer,
    pub energy_estimate_joules: Option<f64>,
}

impl ToolDefinition {
    /// Extract the namespace prefix (e.g. "db" from "db.query").
    pub fn namespace(&self) -> &str {
        self.name.split('.').next().unwrap_or(&self.name)
    }
}

/// A request to call a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub id: String,
    pub tool: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// A response from a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub id: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<McpError>,
    pub energy_joules: Option<f64>,
}

impl ToolCallResponse {
    pub fn success(id: String, result: serde_json::Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
            energy_joules: None,
        }
    }

    pub fn error(id: String, error: McpError) -> Self {
        Self {
            id,
            result: None,
            error: Some(error),
            energy_joules: None,
        }
    }

    pub fn with_energy(mut self, joules: f64) -> Self {
        self.energy_joules = Some(joules);
        self
    }

    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }
}

/// MCP-compatible error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl McpError {
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    pub const TOOL_NOT_FOUND: i32 = -32601;

    pub fn tool_not_found(name: &str) -> Self {
        Self {
            code: Self::TOOL_NOT_FOUND,
            message: format!("tool not found: {name}"),
            data: None,
        }
    }
}

/// Trait for components that handle MCP tool calls.
#[async_trait::async_trait]
pub trait McpToolHandler: Send + Sync {
    /// Handle a tool call request and return a response.
    async fn handle(&self, request: ToolCallRequest) -> ToolCallResponse;
    /// List all tools this handler provides.
    fn tools(&self) -> Vec<ToolDefinition>;
}
