//! CQL (Cassandra Query Language) Execution Engine
//!
//! Executes parsed CQL queries against JouleDB's amorphic storage.
//! Maps Cassandra concepts to JouleDB:
//! - Keyspaces → table name prefix (`keyspace.table` → `keyspace__table`)
//! - Tables → AmorphicTableStorage tables
//! - Partition/clustering keys → stored as regular columns

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::{Expression, Value as AstValue};
use joule_db_query::cql::*;
use joule_db_query::executor::{RowData, TableStorage};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Metadata table for keyspace definitions.
const KEYSPACES_TABLE: &str = "_cql_keyspaces";
/// Metadata table for CQL table schemas.
const SCHEMAS_TABLE: &str = "_cql_schemas";

/// Execute a parsed CQL query against amorphic storage.
pub fn execute_cql(
    query: &CqlQuery,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    ensure_meta_tables(amorphic);

    match &query.statement {
        CqlStatement::Select(sel) => execute_select(sel, amorphic, start),
        CqlStatement::Insert(ins) => execute_insert(ins, amorphic, start),
        CqlStatement::Update(upd) => execute_update(upd, amorphic, start),
        CqlStatement::Delete(del) => execute_delete(del, amorphic, start),
        CqlStatement::CreateKeyspace(ck) => execute_create_keyspace(ck, amorphic, start),
        CqlStatement::CreateTable(ct) => execute_create_table(ct, amorphic, start),
        CqlStatement::CreateIndex(_ci) => Ok(ok_response(start)),
        CqlStatement::DropKeyspace(name) => execute_drop_keyspace(name, amorphic, start),
        CqlStatement::DropTable(dt) => execute_drop_table(dt, amorphic, start),
        CqlStatement::DropIndex(_) => Ok(ok_response(start)),
        CqlStatement::Truncate(dt) => execute_truncate(dt, amorphic, start),
        CqlStatement::Use(keyspace) => execute_use(keyspace, start),
        CqlStatement::Batch(stmts) => execute_batch(stmts, amorphic, start),
    }
}

/// Ensure CQL metadata tables exist.
fn ensure_meta_tables(amorphic: &AmorphicTableStorage) {
    let tables = amorphic.list_tables();
    if !tables.contains(&KEYSPACES_TABLE.to_string()) {
        let _ = amorphic.create_table(
            KEYSPACES_TABLE,
            &[
                "name".to_string(),
                "replication".to_string(),
                "durable_writes".to_string(),
            ],
        );
    }
    if !tables.contains(&SCHEMAS_TABLE.to_string()) {
        let _ = amorphic.create_table(
            SCHEMAS_TABLE,
            &[
                "table_name".to_string(),
                "columns".to_string(),
                "primary_key".to_string(),
            ],
        );
    }
}

/// Resolve a CQL table name (keyspace.table → keyspace__table).
fn resolve_table_name(keyspace: &Option<String>, table: &str) -> String {
    match keyspace {
        Some(ks) => format!("{}__{}", ks, table),
        None => table.to_string(),
    }
}

fn execute_select(
    sel: &CqlSelect,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&sel.keyspace, &sel.table);
    let rows = amorphic.scan(&table_name).unwrap_or_default();

    if rows.is_empty() && !amorphic.list_tables().contains(&table_name) {
        return Err(QueryErrorResponse {
            code: "TABLE_NOT_FOUND".to_string(),
            message: format!("Table '{}' does not exist", sel.table),
            line: None,
            column: None,
        });
    }

    // Filter
    let filtered: Vec<RowData> = if let Some(w) = &sel.where_clause {
        rows.into_iter()
            .filter(|r| evaluate_cql_predicate(w, r))
            .collect()
    } else {
        rows
    };

    // Check for aggregate functions (COUNT, SUM, AVG, MIN, MAX)
    let has_agg = sel.columns.iter().any(|c| {
        let u = c.to_uppercase();
        u.starts_with("COUNT(") || u.starts_with("SUM(") || u.starts_with("AVG(")
            || u.starts_with("MIN(") || u.starts_with("MAX(")
    });

    if has_agg {
        let mut agg_cols = Vec::new();
        let mut agg_vals = Vec::new();
        for col in &sel.columns {
            let u = col.to_uppercase();
            if u.starts_with("COUNT(") {
                agg_cols.push(col.clone());
                agg_vals.push(serde_json::json!(filtered.len() as i64));
            } else if u.starts_with("SUM(") || u.starts_with("AVG(")
                || u.starts_with("MIN(") || u.starts_with("MAX(")
            {
                let inner = &col[4..col.len()-1].trim().to_string();
                let vals: Vec<f64> = filtered.iter().filter_map(|r| {
                    r.get(inner).and_then(|v| {
                        let s = value_to_string_cql(Some(v));
                        s.parse::<f64>().ok()
                    })
                }).collect();
                agg_cols.push(col.clone());
                if u.starts_with("SUM(") {
                    agg_vals.push(serde_json::json!(vals.iter().sum::<f64>()));
                } else if u.starts_with("AVG(") {
                    let avg = if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 };
                    agg_vals.push(serde_json::json!(avg));
                } else if u.starts_with("MIN(") {
                    agg_vals.push(vals.iter().copied().reduce(f64::min).map_or(serde_json::Value::Null, |v| serde_json::json!(v)));
                } else {
                    agg_vals.push(vals.iter().copied().reduce(f64::max).map_or(serde_json::Value::Null, |v| serde_json::json!(v)));
                }
            } else {
                agg_cols.push(col.clone());
                agg_vals.push(serde_json::Value::Null);
            }
        }
        return Ok(QueryResponse {
            columns: agg_cols,
            rows: vec![agg_vals],
            affected_rows: None,
            execution_time_ms: start.elapsed().as_millis() as u64,
            truncated: false,
            warnings: Vec::new(),
            energy_joules: None,
            power_watts: None,
            device_target: None,
            algorithm_type: None,
            session_id: None,
            viz_hint: None,
        });
    }

    // Determine columns
    let all_columns: Vec<String> = if sel.columns.contains(&"*".to_string()) {
        if let Some(first) = filtered.first() {
            first.columns.clone()
        } else {
            Vec::new()
        }
    } else {
        sel.columns.clone()
    };

    // Project
    let mut result_rows: Vec<Vec<serde_json::Value>> = filtered
        .iter()
        .map(|row| {
            all_columns
                .iter()
                .map(|col| {
                    row.get(col)
                        .map(|v| value_to_json(v))
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect()
        })
        .collect();

    // DISTINCT
    if sel.distinct {
        let mut seen = std::collections::HashSet::new();
        result_rows.retain(|row| seen.insert(serde_json::to_string(row).unwrap_or_default()));
    }

    // ORDER BY
    if !sel.order_by.is_empty() {
        let col_idx: Vec<(usize, bool)> = sel
            .order_by
            .iter()
            .filter_map(|(col, desc)| {
                all_columns
                    .iter()
                    .position(|c| c == col)
                    .map(|i| (i, *desc))
            })
            .collect();
        result_rows.sort_by(|a, b| {
            for &(idx, desc) in &col_idx {
                if idx >= a.len() || idx >= b.len() {
                    continue;
                }
                let cmp = compare_json(&a[idx], &b[idx]);
                let cmp = if desc { cmp.reverse() } else { cmp };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    // LIMIT
    if let Some(limit) = sel.limit {
        result_rows.truncate(limit);
    }

    Ok(QueryResponse {
        columns: all_columns,
        rows: result_rows,
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    })
}

fn execute_insert(
    ins: &CqlInsert,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&ins.keyspace, &ins.table);

    let columns: Vec<String> = ins.columns.clone();
    let values: Vec<AstValue> = ins
        .columns
        .iter()
        .enumerate()
        .map(|(i, _)| {
            ins.values
                .get(i)
                .map(expr_to_ast_value)
                .unwrap_or(AstValue::Null)
        })
        .collect();
    let row = RowData::new(columns, values);

    // IF NOT EXISTS
    if ins.if_not_exists {
        let existing = amorphic.scan(&table_name).unwrap_or_default();
        let pk_col = ins.columns.first().cloned().unwrap_or_default();
        let pk_val = row.get(&pk_col).cloned().unwrap_or(AstValue::Null);
        if existing.iter().any(|r| r.get(&pk_col) == Some(&pk_val)) {
            return Ok(QueryResponse {
                columns: vec!["[applied]".to_string()],
                rows: vec![vec![serde_json::Value::Bool(false)]],
                affected_rows: Some(0),
                execution_time_ms: start.elapsed().as_millis() as u64,
                truncated: false,
                warnings: Vec::new(),
                energy_joules: None,
                power_watts: None,
                device_target: None,
                algorithm_type: None,
                session_id: None,
                viz_hint: None,
            });
        }
    }

    let _ = amorphic.insert(&table_name, &row);

    Ok(QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        affected_rows: Some(1),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    })
}

fn execute_update(
    upd: &CqlUpdate,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&upd.keyspace, &upd.table);

    let mut assignments = HashMap::new();
    for a in &upd.assignments {
        match a {
            CqlAssignment::Set(col, expr) => {
                assignments.insert(col.clone(), expr_to_ast_value(expr));
            }
            CqlAssignment::Increment(col, expr) => {
                // counter = counter + delta: read current, add, write back
                let delta = match expr_to_ast_value(expr) {
                    AstValue::Int(n) => n,
                    AstValue::Float(f) => f as i64,
                    _ => 1,
                };
                let rows = amorphic.scan(&table_name).unwrap_or_default();
                let current = rows
                    .iter()
                    .find(|r| evaluate_cql_predicate(&upd.where_clause, r))
                    .and_then(|r| r.get(col))
                    .map(|v| match v {
                        AstValue::Int(n) => *n,
                        AstValue::Float(f) => *f as i64,
                        _ => 0,
                    })
                    .unwrap_or(0);
                assignments.insert(col.clone(), AstValue::Int(current + delta));
            }
            CqlAssignment::Decrement(col, expr) => {
                // counter = counter - delta: read current, subtract, write back
                let delta = match expr_to_ast_value(expr) {
                    AstValue::Int(n) => n,
                    AstValue::Float(f) => f as i64,
                    _ => 1,
                };
                let rows = amorphic.scan(&table_name).unwrap_or_default();
                let current = rows
                    .iter()
                    .find(|r| evaluate_cql_predicate(&upd.where_clause, r))
                    .and_then(|r| r.get(col))
                    .map(|v| match v {
                        AstValue::Int(n) => *n,
                        AstValue::Float(f) => *f as i64,
                        _ => 0,
                    })
                    .unwrap_or(0);
                assignments.insert(col.clone(), AstValue::Int(current - delta));
            }
            _ => {
                // Collection ops (Append, Prepend, AddToSet, etc.) — treat as set
                if let CqlAssignment::Append(col, expr)
                | CqlAssignment::Prepend(col, expr)
                | CqlAssignment::RemoveFromList(col, expr)
                | CqlAssignment::AddToSet(col, expr)
                | CqlAssignment::RemoveFromSet(col, expr)
                | CqlAssignment::RemoveFromMap(col, expr) = a
                {
                    assignments.insert(col.clone(), expr_to_ast_value(expr));
                }
                if let CqlAssignment::PutInMap(col, _key, expr) = a {
                    assignments.insert(col.clone(), expr_to_ast_value(expr));
                }
            }
        }
    }

    let affected = amorphic
        .update(&table_name, &assignments, Some(&upd.where_clause))
        .unwrap_or(0);

    Ok(QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        affected_rows: Some(affected),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    })
}

fn execute_delete(
    del: &CqlDelete,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&del.keyspace, &del.table);
    let affected = amorphic
        .delete(&table_name, Some(&del.where_clause))
        .unwrap_or(0);

    Ok(QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        affected_rows: Some(affected),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    })
}

fn execute_create_keyspace(
    ck: &CqlCreateKeyspace,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let row = RowData::new(
        vec!["name".into(), "replication".into(), "durable_writes".into()],
        vec![
            AstValue::String(ck.name.clone()),
            AstValue::String(serde_json::to_string(&ck.replication).unwrap_or_default()),
            AstValue::String(ck.durable_writes.to_string()),
        ],
    );
    let _ = amorphic.insert(KEYSPACES_TABLE, &row);
    Ok(ok_response(start))
}

fn execute_create_table(
    ct: &CqlCreateTable,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&ct.keyspace, &ct.name);

    if ct.if_not_exists && amorphic.list_tables().contains(&table_name) {
        return Ok(ok_response(start));
    }

    let columns: Vec<String> = ct.columns.iter().map(|c| c.name.clone()).collect();
    let _ = amorphic.create_table(&table_name, &columns);

    // Store schema metadata
    let meta = RowData::new(
        vec!["table_name".into(), "columns".into(), "primary_key".into()],
        vec![
            AstValue::String(table_name),
            AstValue::String(serde_json::to_string(&columns).unwrap_or_default()),
            AstValue::String(ct.primary_key.partition_key.join(",")),
        ],
    );
    let _ = amorphic.insert(SCHEMAS_TABLE, &meta);

    Ok(ok_response(start))
}

fn execute_drop_keyspace(
    name: &str,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let filter = Expression::eq(
        Expression::Column("name".into()),
        Expression::Literal(AstValue::String(name.to_string())),
    );
    let _ = amorphic.delete(KEYSPACES_TABLE, Some(&filter));

    // Drop all tables with this keyspace prefix
    let prefix = format!("{}__", name);
    for table in amorphic.list_tables() {
        if table.starts_with(&prefix) {
            let _ = amorphic.drop_table(&table);
        }
    }

    Ok(ok_response(start))
}

fn execute_drop_table(
    dt: &CqlDropTable,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&dt.keyspace, &dt.table);

    if dt.if_exists && !amorphic.list_tables().contains(&table_name) {
        return Ok(ok_response(start));
    }

    let _ = amorphic.drop_table(&table_name);

    let filter = Expression::eq(
        Expression::Column("table_name".into()),
        Expression::Literal(AstValue::String(table_name)),
    );
    let _ = amorphic.delete(SCHEMAS_TABLE, Some(&filter));

    Ok(ok_response(start))
}

fn execute_truncate(
    dt: &CqlDropTable,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let table_name = resolve_table_name(&dt.keyspace, &dt.table);
    // Delete all rows
    let _ = amorphic.delete(&table_name, None);
    Ok(ok_response(start))
}

fn execute_use(_keyspace: &str, start: Instant) -> Result<QueryResponse, QueryErrorResponse> {
    Ok(QueryResponse {
        columns: vec!["status".to_string()],
        rows: vec![vec![serde_json::Value::String("OK".into())]],
        affected_rows: None,
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    })
}

fn execute_batch(
    stmts: &[CqlStatement],
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let mut total = 0usize;
    for stmt in stmts {
        let q = CqlQuery {
            statement: stmt.clone(),
        };
        let r = execute_cql(&q, amorphic, start)?;
        total += r.affected_rows.unwrap_or(0);
    }
    Ok(QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        affected_rows: Some(total),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    })
}

fn evaluate_cql_predicate(expr: &Expression, row: &RowData) -> bool {
    match expr {
        Expression::Binary { left, op, right } => match op {
            joule_db_query::ast::Operator::And => {
                evaluate_cql_predicate(left, row) && evaluate_cql_predicate(right, row)
            }
            joule_db_query::ast::Operator::Or => {
                evaluate_cql_predicate(left, row) || evaluate_cql_predicate(right, row)
            }
            joule_db_query::ast::Operator::Eq => {
                eval_cql_expr(left, row) == eval_cql_expr(right, row)
            }
            joule_db_query::ast::Operator::Ne => {
                eval_cql_expr(left, row) != eval_cql_expr(right, row)
            }
            joule_db_query::ast::Operator::Lt => {
                compare_strs(&eval_cql_expr(left, row), &eval_cql_expr(right, row))
                    == std::cmp::Ordering::Less
            }
            joule_db_query::ast::Operator::Le => {
                compare_strs(&eval_cql_expr(left, row), &eval_cql_expr(right, row))
                    != std::cmp::Ordering::Greater
            }
            joule_db_query::ast::Operator::Gt => {
                compare_strs(&eval_cql_expr(left, row), &eval_cql_expr(right, row))
                    == std::cmp::Ordering::Greater
            }
            joule_db_query::ast::Operator::Ge => {
                compare_strs(&eval_cql_expr(left, row), &eval_cql_expr(right, row))
                    != std::cmp::Ordering::Less
            }
            _ => true,
        },
        Expression::In {
            expr,
            list,
            negated,
        } => {
            let val = eval_cql_expr(expr, row);
            let matches = list.iter().any(|item| eval_cql_expr(item, row) == val);
            if *negated { !matches } else { matches }
        }
        Expression::Literal(AstValue::Bool(b)) => *b,
        _ => true,
    }
}

fn eval_cql_expr(expr: &Expression, row: &RowData) -> String {
    match expr {
        Expression::Column(name) => value_to_string_cql(row.get(name)),
        Expression::Literal(v) => match v {
            AstValue::String(s) => s.clone(),
            AstValue::Int(n) => n.to_string(),
            AstValue::Float(f) => f.to_string(),
            AstValue::Bool(b) => b.to_string(),
            AstValue::Null => String::new(),
            _ => format!("{:?}", v),
        },
        _ => String::new(),
    }
}

fn compare_strs(a: &str, b: &str) -> std::cmp::Ordering {
    if let (Ok(an), Ok(bn)) = (a.parse::<f64>(), b.parse::<f64>()) {
        an.partial_cmp(&bn).unwrap_or(std::cmp::Ordering::Equal)
    } else {
        a.cmp(b)
    }
}

fn expression_to_string(expr: &Expression) -> String {
    match expr {
        Expression::Literal(v) => match v {
            AstValue::String(s) => s.clone(),
            AstValue::Int(n) => n.to_string(),
            AstValue::Float(f) => f.to_string(),
            AstValue::Bool(b) => b.to_string(),
            AstValue::Null => String::new(),
            _ => format!("{:?}", v),
        },
        Expression::Column(name) => name.clone(),
        _ => String::new(),
    }
}

fn expr_to_ast_value(expr: &Expression) -> AstValue {
    match expr {
        Expression::Literal(v) => v.clone(),
        _ => AstValue::String(expression_to_string(expr)),
    }
}

/// Convert a Value to serde_json::Value.
fn value_to_json(v: &AstValue) -> serde_json::Value {
    match v {
        AstValue::Null => serde_json::Value::Null,
        AstValue::Bool(b) => serde_json::Value::Bool(*b),
        AstValue::Int(n) => serde_json::json!(*n),
        AstValue::Float(f) => serde_json::json!(*f),
        AstValue::String(s) => serde_json::Value::String(s.clone()),
        AstValue::Array(arr) => serde_json::Value::Array(arr.iter().map(value_to_json).collect()),
        AstValue::Bytes(b) => serde_json::Value::String(format!("0x{}", hex::encode(b))),
        AstValue::Object(map) => {
            let m: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(m)
        }
        AstValue::Timestamp(ts) => serde_json::json!(*ts),
        AstValue::Uuid(u) => serde_json::Value::String(u.clone()),
        AstValue::Vector(v) => {
            serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
        }
    }
}

/// Convert an `Option<&Value>` (from RowData::get) to a String for predicate evaluation.
fn value_to_string_cql(v: Option<&AstValue>) -> String {
    match v {
        None => String::new(),
        Some(AstValue::String(s)) => s.clone(),
        Some(AstValue::Int(n)) => n.to_string(),
        Some(AstValue::Float(f)) => f.to_string(),
        Some(AstValue::Bool(b)) => b.to_string(),
        Some(AstValue::Null) => String::new(),
        Some(AstValue::Bytes(b)) => format!("0x{}", hex::encode(b)),
        Some(AstValue::Array(_)) => format!("{:?}", v),
        Some(AstValue::Object(_)) => format!("{:?}", v),
        Some(AstValue::Timestamp(ts)) => ts.to_string(),
        Some(AstValue::Uuid(u)) => u.clone(),
        Some(AstValue::Vector(v)) => format!("{:?}", v),
    }
}

fn compare_json(a: &serde_json::Value, b: &serde_json::Value) -> std::cmp::Ordering {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => a
            .as_f64()
            .unwrap_or(0.0)
            .partial_cmp(&b.as_f64().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal),
        (serde_json::Value::String(a), serde_json::Value::String(b)) => a.cmp(b),
        _ => std::cmp::Ordering::Equal,
    }
}

fn ok_response(start: Instant) -> QueryResponse {
    QueryResponse {
        columns: Vec::new(),
        rows: Vec::new(),
        affected_rows: Some(0),
        execution_time_ms: start.elapsed().as_millis() as u64,
        truncated: false,
        warnings: Vec::new(),
        energy_joules: None,
        power_watts: None,
        device_target: None,
        algorithm_type: None,
        session_id: None,
        viz_hint: None,
    }
}

/// Detect whether a query string looks like CQL.
///
/// Note: TRUNCATE is intentionally NOT included here because it is also valid SQL.
/// CQL queries using TRUNCATE will be handled by the SQL executor (which supports TRUNCATE).
pub fn is_cql_query(sql: &str) -> bool {
    let t = sql.trim().to_uppercase();
    t.starts_with("CREATE KEYSPACE")
        || t.starts_with("DROP KEYSPACE")
        || t.starts_with("USE ")
        || t.contains("ALLOW FILTERING")
        || t.starts_with("BEGIN BATCH")
        || t.starts_with("BEGIN UNLOGGED")
        || t.starts_with("BEGIN COUNTER")
        || t.contains("USING TTL")
        || has_keyspace_qualified_table(&t)
}

/// Detect CQL statements that reference keyspace-qualified table names (e.g. ks.table).
/// Checks for `word.word` patterns after keywords like INTO, FROM, TABLE, UPDATE, TRUNCATE.
/// Excludes known SQL schema prefixes (information_schema, pg_catalog).
fn has_keyspace_qualified_table(upper: &str) -> bool {
    const SQL_SCHEMAS: &[&str] = &["INFORMATION_SCHEMA", "PG_CATALOG"];

    for prefix in &["INTO ", "FROM ", "TABLE ", "UPDATE ", "TRUNCATE "] {
        if let Some(pos) = upper.find(prefix) {
            let rest = &upper[pos + prefix.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.')
                .collect();
            if let Some(dot) = name.find('.') {
                if dot > 0 && dot + 1 < name.len() && name[dot + 1..].find('.').is_none() {
                    let schema = &name[..dot];
                    if !SQL_SCHEMAS.iter().any(|s| *s == schema) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_amorphic() -> Arc<AmorphicTableStorage> {
        let dir = std::env::temp_dir().join(format!("jouledb-cql-test-{}", uuid::Uuid::new_v4()));
        let store = joule_db_amorphic::DurableAmorphicStore::open(&dir).expect("temp store");
        Arc::new(AmorphicTableStorage::new(store))
    }

    #[test]
    fn test_create_keyspace() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();
        let q = parser
            .parse("CREATE KEYSPACE myks WITH REPLICATION = {'class': 'SimpleStrategy'}")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(0));
    }

    #[test]
    fn test_create_table_and_insert() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));
    }

    #[test]
    fn test_select() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();
        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("SELECT * FROM users").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_select_with_where() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();
        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_delete() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("DELETE FROM users WHERE id = 1").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert!(result.affected_rows.unwrap_or(0) > 0);
    }

    #[test]
    fn test_use_keyspace() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();
        let q = parser.parse("USE myks").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.columns, vec!["status"]);
    }

    #[test]
    fn test_is_cql() {
        assert!(is_cql_query("CREATE KEYSPACE myks WITH REPLICATION = {}"));
        assert!(is_cql_query("USE myks"));
        assert!(is_cql_query(
            "BEGIN BATCH INSERT INTO users (id) VALUES (1) APPLY BATCH"
        ));
        // TRUNCATE is handled by SQL executor (shared syntax)
        assert!(!is_cql_query("TRUNCATE users"));
        assert!(!is_cql_query("SELECT * FROM users"));
        assert!(!is_cql_query("INSERT INTO users (id) VALUES (1)"));
    }

    #[test]
    fn test_drop_keyspace() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE KEYSPACE myks WITH REPLICATION = {'class': 'SimpleStrategy'}")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("CREATE TABLE myks.users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("DROP KEYSPACE myks").unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        assert!(!amorphic.list_tables().contains(&"myks__users".to_string()));
    }

    #[test]
    fn test_update_set() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));

        let q = parser
            .parse("UPDATE users SET name = 'Bob' WHERE id = 1")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));

        let q = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        // Find the name column index and verify the updated value
        let name_idx = result
            .columns
            .iter()
            .position(|c| c == "name")
            .expect("name column");
        assert_eq!(
            result.rows[0][name_idx],
            serde_json::Value::String("Bob".to_string())
        );
    }

    #[test]
    fn test_insert_if_not_exists_duplicate() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));

        // Second insert with IF NOT EXISTS for same primary key should not insert
        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Bob') IF NOT EXISTS")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.columns, vec!["[applied]"]);
        assert_eq!(result.rows[0][0], serde_json::Value::Bool(false));
        assert_eq!(result.affected_rows, Some(0));

        // Verify original row is still Alice
        let q = parser.parse("SELECT * FROM users WHERE id = 1").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        let name_idx = result
            .columns
            .iter()
            .position(|c| c == "name")
            .expect("name column");
        assert_eq!(
            result.rows[0][name_idx],
            serde_json::Value::String("Alice".to_string())
        );
    }

    #[test]
    fn test_batch_operations() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse(
                "BEGIN BATCH \
             INSERT INTO users (id, name) VALUES (1, 'Alice'); \
             INSERT INTO users (id, name) VALUES (2, 'Bob'); \
             APPLY BATCH",
            )
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(2));

        let q = parser.parse("SELECT * FROM users").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_truncate_table() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();
        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (2, 'Bob')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("TRUNCATE users").unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        // After TRUNCATE, the table still exists but has no rows.
        // SELECT * returns empty columns/rows since column names come from first row.
        let q = parser.parse("SELECT * FROM users").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(0));

        // Insert a row so we can verify the table survives the second CREATE
        let q = parser
            .parse("INSERT INTO users (id, name) VALUES (1, 'Alice')")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        // Second CREATE TABLE IF NOT EXISTS should succeed without error
        let q = parser
            .parse("CREATE TABLE IF NOT EXISTS users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(0));

        // Verify data is still intact
        let q = parser.parse("SELECT * FROM users").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_drop_table_if_exists() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        // DROP TABLE IF EXISTS on a nonexistent table should return ok_response with no error
        let q = parser.parse("DROP TABLE IF EXISTS nonexistent").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(0));
    }

    #[test]
    fn test_select_with_limit() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        for i in 1..=5 {
            let q = parser
                .parse(&format!(
                    "INSERT INTO users (id, name) VALUES ({}, 'User{}')",
                    i, i
                ))
                .unwrap();
            execute_cql(&q, &amorphic, Instant::now()).unwrap();
        }

        let q = parser.parse("SELECT * FROM users LIMIT 2").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_select_specific_columns() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("SELECT id, name FROM users").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.columns, vec!["id", "name"]);
        assert_eq!(result.rows.len(), 1);
        // Should only have 2 values per row (id and name)
        assert_eq!(result.rows[0].len(), 2);
    }

    #[test]
    fn test_keyspaced_table_operations() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE KEYSPACE testks WITH REPLICATION = {'class': 'SimpleStrategy'}")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("CREATE TABLE testks.items (id INT PRIMARY KEY, val TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO testks.items (id, val) VALUES (1, 'hello')")
            .unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));

        let q = parser.parse("SELECT * FROM testks.items").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);

        // Verify the underlying table name uses keyspace__table convention
        assert!(
            amorphic
                .list_tables()
                .contains(&"testks__items".to_string())
        );
    }

    #[test]
    fn test_empty_table_select() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        // SELECT from an existing but empty table should return 0 rows (not an error)
        // The table exists in list_tables, so the TABLE_NOT_FOUND check is skipped.
        let q = parser.parse("SELECT * FROM users").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 0);
        assert_eq!(result.columns.len(), 0); // columns come from first row, which is empty
    }

    #[test]
    fn test_increment_counter() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE counters (id INT PRIMARY KEY, count INT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO counters (id, count) VALUES (1, 10)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("UPDATE counters SET count = count + 5 WHERE id = 1")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("SELECT * FROM counters WHERE id = 1").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        // count should be 15 (10 + 5)
        let count_idx = result.columns.iter().position(|c| c == "count").unwrap();
        assert_eq!(result.rows[0][count_idx], serde_json::json!(15));
    }

    #[test]
    fn test_decrement_counter() {
        let amorphic = test_amorphic();
        let mut parser = CqlParser::new();

        let q = parser
            .parse("CREATE TABLE counters (id INT PRIMARY KEY, count INT)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("INSERT INTO counters (id, count) VALUES (1, 10)")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser
            .parse("UPDATE counters SET count = count - 3 WHERE id = 1")
            .unwrap();
        execute_cql(&q, &amorphic, Instant::now()).unwrap();

        let q = parser.parse("SELECT * FROM counters WHERE id = 1").unwrap();
        let result = execute_cql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        // count should be 7 (10 - 3)
        let count_idx = result.columns.iter().position(|c| c == "count").unwrap();
        assert_eq!(result.rows[0][count_idx], serde_json::json!(7));
    }
}
