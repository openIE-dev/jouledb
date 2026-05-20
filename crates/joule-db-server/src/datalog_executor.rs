//! Datalog Execution Engine
//!
//! Executes parsed Datalog programs against JouleDB's amorphic storage.
//! Rules derive new facts via semi-naive evaluation with stratified negation.
//! Base predicates map to existing tables in the amorphic store.

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::Value as AstValue;
use joule_db_query::datalog::*;
use joule_db_query::executor::TableStorage;
use std::sync::Arc;
use std::time::Instant;

/// Execute a parsed Datalog program against amorphic storage.
pub fn execute_datalog(
    program: &DatalogProgram,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    // Load base predicates from amorphic tables.
    let mut enriched = program.clone();

    let table_names: Vec<String> = amorphic.list_tables();
    for rule in &program.rules {
        for lit in &rule.body {
            if let DatalogLiteral::Positive(atom) = lit {
                if table_names.contains(&atom.predicate) && !has_facts_for(&enriched, &atom.predicate) {
                    if let Ok(rows) = amorphic.scan(&atom.predicate) {
                        for row in &rows {
                            let terms: Vec<DatalogTerm> = row
                                .columns
                                .iter()
                                .enumerate()
                                .map(|(i, _col)| {
                                    match row.values.get(i) {
                                        Some(AstValue::String(s)) => DatalogTerm::String(s.clone()),
                                        Some(AstValue::Int(n)) => DatalogTerm::Integer(*n),
                                        Some(AstValue::Float(f)) => DatalogTerm::Float(*f),
                                        _ => DatalogTerm::String(
                                            row.values.get(i).map(|v| format!("{:?}", v)).unwrap_or_default(),
                                        ),
                                    }
                                })
                                .collect();
                            enriched.facts.push(DatalogAtom {
                                predicate: atom.predicate.clone(),
                                terms,
                            });
                        }
                    }
                }
            }
        }
    }

    let result = evaluate(&enriched).map_err(|e| {
        QueryErrorResponse::syntax_error(&e.to_string(), 1, 1)
    })?;

    let rows: Vec<Vec<serde_json::Value>> = result
        .rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| match v {
                    AstValue::String(s) => serde_json::Value::String(s.clone()),
                    AstValue::Int(n) => serde_json::json!(n),
                    AstValue::Float(f) => serde_json::json!(f),
                    AstValue::Bool(b) => serde_json::json!(b),
                    AstValue::Null => serde_json::Value::Null,
                    _ => serde_json::Value::String(format!("{:?}", v)),
                })
                .collect()
        })
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
        algorithm_type: Some("datalog_semi_naive".to_string()),
        session_id: None,
        viz_hint: None,
    })
}

fn has_facts_for(program: &DatalogProgram, predicate: &str) -> bool {
    program.facts.iter().any(|f| f.predicate == predicate)
}

/// Detect whether a query string looks like Datalog.
pub fn is_datalog_query(sql: &str) -> bool {
    let t = sql.trim();
    t.starts_with("?-")
        || (t.contains(":-") && !t.to_uppercase().contains("SELECT"))
}
