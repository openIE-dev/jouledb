//! Gremlin Execution Engine
//!
//! Executes parsed Gremlin traversals against JouleDB's graph storage.
//! Maps Gremlin steps to operations on _graph_nodes and _graph_edges tables.

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::Value as AstValue;
use joule_db_query::executor::TableStorage;
use joule_db_query::gremlin::*;
use std::sync::Arc;
use std::time::Instant;

const NODES_TABLE: &str = "_graph_nodes";
const EDGES_TABLE: &str = "_graph_edges";

/// Execute a parsed Gremlin query against amorphic storage.
pub fn execute_gremlin(
    query: &GremlinQuery,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    ensure_graph_tables(amorphic);

    // Load graph data.
    let node_rows = amorphic.scan(NODES_TABLE).unwrap_or_default();
    let edge_rows = amorphic.scan(EDGES_TABLE).unwrap_or_default();

    // Convert RowData to serde_json maps for easier manipulation.
    let nodes: Vec<serde_json::Map<String, serde_json::Value>> = node_rows
        .iter()
        .map(|r| row_to_json_map(r))
        .collect();
    let edges: Vec<serde_json::Map<String, serde_json::Value>> = edge_rows
        .iter()
        .map(|r| row_to_json_map(r))
        .collect();

    // Start with the source traversers.
    let mut traversers: Vec<serde_json::Value> = match &query.source {
        GremlinSource::V(None) => {
            nodes.iter().map(|n| serde_json::Value::Object(n.clone())).collect()
        }
        GremlinSource::V(Some(id)) => {
            let id_str = match id {
                GremlinValue::Integer(n) => n.to_string(),
                GremlinValue::String(s) => s.clone(),
                _ => String::new(),
            };
            nodes
                .iter()
                .filter(|n| n.get("id").and_then(|v| v.as_str()) == Some(&id_str))
                .map(|n| serde_json::Value::Object(n.clone()))
                .collect()
        }
        GremlinSource::E(None) => {
            edges.iter().map(|e| serde_json::Value::Object(e.clone())).collect()
        }
        _ => Vec::new(),
    };

    // Apply each step.
    for step in &query.steps {
        traversers = apply_step(step, traversers, &nodes, &edges);
    }

    // Convert traversers to response.
    let columns = if traversers.is_empty() {
        Vec::new()
    } else {
        match &traversers[0] {
            serde_json::Value::Object(map) => map.keys().cloned().collect(),
            _ => vec!["value".to_string()],
        }
    };

    let rows: Vec<Vec<serde_json::Value>> = traversers
        .into_iter()
        .map(|t| match t {
            serde_json::Value::Object(map) => {
                columns.iter().map(|c| map.get(c).cloned().unwrap_or(serde_json::Value::Null)).collect()
            }
            other => vec![other],
        })
        .collect();

    Ok(QueryResponse {
        columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("gremlin_traversal".to_string()),
        session_id: None,
        viz_hint: None,
    })
}

fn row_to_json_map(row: &joule_db_query::executor::RowData) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    for (i, col) in row.columns.iter().enumerate() {
        let val = match row.values.get(i) {
            Some(AstValue::String(s)) => serde_json::Value::String(s.clone()),
            Some(AstValue::Int(n)) => serde_json::json!(n),
            Some(AstValue::Float(f)) => serde_json::json!(f),
            Some(AstValue::Bool(b)) => serde_json::json!(b),
            Some(AstValue::Null) | None => serde_json::Value::Null,
            Some(other) => serde_json::Value::String(format!("{:?}", other)),
        };
        map.insert(col.clone(), val);
    }
    map
}

fn apply_step(
    step: &GremlinStep,
    traversers: Vec<serde_json::Value>,
    nodes: &[serde_json::Map<String, serde_json::Value>],
    edges: &[serde_json::Map<String, serde_json::Value>],
) -> Vec<serde_json::Value> {
    match step {
        GremlinStep::HasLabel(label) => {
            traversers
                .into_iter()
                .filter(|t| {
                    t.get("labels")
                        .and_then(|v| v.as_str())
                        .map(|s| s.contains(label.as_str()))
                        .unwrap_or(false)
                })
                .collect()
        }
        GremlinStep::Has(key, value) => {
            traversers
                .into_iter()
                .filter(|t| match value {
                    Some(GremlinValue::String(s)) => {
                        t.get(key).and_then(|v| v.as_str()) == Some(s.as_str())
                    }
                    None => t.get(key).is_some(),
                    _ => true,
                })
                .collect()
        }
        GremlinStep::Out(edge_type) => {
            let mut result = Vec::new();
            for t in &traversers {
                let id = t.get("id").and_then(|v| v.as_str()).unwrap_or("");
                for edge in edges {
                    let from = edge.get("start_node").and_then(|v| v.as_str()).unwrap_or("");
                    if from != id { continue; }
                    if let Some(et) = edge_type {
                        let etype = edge.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if etype != et { continue; }
                    }
                    let to = edge.get("end_node").and_then(|v| v.as_str()).unwrap_or("");
                    if let Some(node) = nodes.iter().find(|n| n.get("id").and_then(|v| v.as_str()) == Some(to)) {
                        result.push(serde_json::Value::Object(node.clone()));
                    }
                }
            }
            result
        }
        GremlinStep::In(edge_type) => {
            let mut result = Vec::new();
            for t in &traversers {
                let id = t.get("id").and_then(|v| v.as_str()).unwrap_or("");
                for edge in edges {
                    let to = edge.get("end_node").and_then(|v| v.as_str()).unwrap_or("");
                    if to != id { continue; }
                    if let Some(et) = edge_type {
                        let etype = edge.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if etype != et { continue; }
                    }
                    let from = edge.get("start_node").and_then(|v| v.as_str()).unwrap_or("");
                    if let Some(node) = nodes.iter().find(|n| n.get("id").and_then(|v| v.as_str()) == Some(from)) {
                        result.push(serde_json::Value::Object(node.clone()));
                    }
                }
            }
            result
        }
        GremlinStep::Values(keys) => {
            traversers
                .into_iter()
                .map(|t| {
                    if keys.len() == 1 {
                        t.get(&keys[0]).cloned().unwrap_or(serde_json::Value::Null)
                    } else {
                        let mut map = serde_json::Map::new();
                        for k in keys {
                            if let Some(v) = t.get(k) {
                                map.insert(k.clone(), v.clone());
                            }
                        }
                        serde_json::Value::Object(map)
                    }
                })
                .collect()
        }
        GremlinStep::Count => {
            vec![serde_json::json!(traversers.len())]
        }
        GremlinStep::Limit(n) => {
            traversers.into_iter().take(*n).collect()
        }
        GremlinStep::Dedup => {
            let mut seen = std::collections::HashSet::new();
            traversers
                .into_iter()
                .filter(|t| seen.insert(t.to_string()))
                .collect()
        }
        GremlinStep::Drop => Vec::new(),
        GremlinStep::Property(key, value) => {
            traversers
                .into_iter()
                .map(|mut t| {
                    if let serde_json::Value::Object(ref mut map) = t {
                        let val = match value {
                            GremlinValue::String(s) => serde_json::json!(s),
                            GremlinValue::Integer(n) => serde_json::json!(n),
                            GremlinValue::Float(f) => serde_json::json!(f),
                            GremlinValue::Boolean(b) => serde_json::json!(b),
                        };
                        map.insert(key.clone(), val);
                    }
                    t
                })
                .collect()
        }
        _ => traversers, // Unhandled steps pass through.
    }
}

fn ensure_graph_tables(amorphic: &AmorphicTableStorage) {
    let tables = amorphic.list_tables();
    if !tables.contains(&NODES_TABLE.to_string()) {
        let cols = vec!["id".to_string(), "labels".to_string(), "properties".to_string()];
        let _ = amorphic.create_table(NODES_TABLE, &cols);
    }
    if !tables.contains(&EDGES_TABLE.to_string()) {
        let cols = vec!["id".to_string(), "type".to_string(), "start_node".to_string(), "end_node".to_string(), "properties".to_string()];
        let _ = amorphic.create_table(EDGES_TABLE, &cols);
    }
}

/// Detect whether a query string looks like Gremlin.
pub fn is_gremlin_query(sql: &str) -> bool {
    let t = sql.trim();
    t.starts_with("g.V(")
        || t.starts_with("g.E(")
        || t.starts_with("g.addV(")
        || t.starts_with("g.addE(")
}
