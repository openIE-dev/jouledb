//! SPARQL Execution Engine
//!
//! Executes parsed SPARQL queries against JouleDB's amorphic storage.
//! Triple patterns map to the `_rdf_triples` table or HDC knowledge core.

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::Value as AstValue;
use joule_db_query::executor::TableStorage;
use joule_db_query::sparql::*;
use std::sync::Arc;
use std::time::Instant;

const TRIPLES_TABLE: &str = "_rdf_triples";

/// Execute a parsed SPARQL query against amorphic storage.
pub fn execute_sparql(
    query: &SparqlQuery,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let triples = load_triples(amorphic);

    let result = evaluate_sparql(query, &triples).map_err(|e| {
        QueryErrorResponse::syntax_error(&e.to_string(), 1, 1)
    })?;

    if let Some(ask) = result.ask_result {
        return Ok(QueryResponse {
            columns: vec!["result".to_string()],
            rows: vec![vec![serde_json::json!(ask)]],
            affected_rows: None,
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated: false,
            warnings: Vec::new(),
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: Some("sparql_bgp".to_string()),
            session_id: None,
            viz_hint: None,
        });
    }

    let rows: Vec<Vec<serde_json::Value>> = result
        .rows
        .iter()
        .map(|row| row.iter().map(|s| serde_json::Value::String(s.clone())).collect())
        .collect();

    Ok(QueryResponse {
        columns: result.columns,
        rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: Some("sparql_bgp".to_string()),
        session_id: None,
        viz_hint: None,
    })
}

fn load_triples(amorphic: &AmorphicTableStorage) -> Vec<Triple> {
    let rows = match amorphic.scan(TRIPLES_TABLE) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    rows.iter()
        .filter_map(|row| {
            let subject = match row.get("subject") {
                Some(AstValue::String(s)) => s.clone(),
                _ => return None,
            };
            let predicate = match row.get("predicate") {
                Some(AstValue::String(s)) => s.clone(),
                _ => return None,
            };
            let object = match row.get("object") {
                Some(AstValue::String(s)) => s.clone(),
                _ => return None,
            };
            Some(Triple { subject, predicate, object })
        })
        .collect()
}

/// Detect whether a query string looks like SPARQL.
pub fn is_sparql_query(sql: &str) -> bool {
    let t = sql.trim();
    let upper = t.to_uppercase();
    t.starts_with("PREFIX ") || t.starts_with("prefix ")
        || upper.starts_with("ASK ")
        || upper.starts_with("CONSTRUCT ")
        || upper.starts_with("DESCRIBE ")
        || (upper.starts_with("SELECT ") && t.contains('?') && upper.contains("WHERE {"))
}
