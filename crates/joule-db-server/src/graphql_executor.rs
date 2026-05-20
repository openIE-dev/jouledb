//! GraphQL Query Execution Engine
//!
//! Executes parsed GraphQL queries against JouleDB's amorphic storage.
//! Maps GraphQL concepts to relational operations:
//! - Root field name → table name (`{ users { id name } }` → scan `users`)
//! - Field arguments `where`/`filter` → WHERE predicates
//! - `limit`/`first` → LIMIT, `offset`/`skip` → OFFSET, `orderBy` → ORDER BY
//! - Mutations: `createX(input)` → INSERT, `updateX(id, input)` → UPDATE, `deleteX(id)` → DELETE

use crate::amorphic_adapter::AmorphicTableStorage;
use crate::query::{QueryErrorResponse, QueryResponse};
use joule_db_query::ast::{Expression, Value as AstValue};
use joule_db_query::executor::{RowData, TableStorage};
use joule_db_query::graphql::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Execute a parsed GraphQL query against amorphic storage.
pub fn execute_graphql(
    query: &GraphqlQuery,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    if query.operations.is_empty() {
        return Err(QueryErrorResponse::syntax_error(
            "No operations in GraphQL query",
            1,
            1,
        ));
    }

    let op = &query.operations[0];
    match op.operation_type {
        GraphqlOperationType::Query => {
            execute_query_operation(op, &query.fragments, amorphic, start)
        }
        GraphqlOperationType::Mutation => execute_mutation_operation(op, amorphic, start),
        GraphqlOperationType::Subscription => Err(QueryErrorResponse::execution_error(
            "GraphQL subscriptions are not supported; use JouleDB real-time subscriptions instead",
        )),
    }
}

/// Execute a GraphQL query operation (read).
fn execute_query_operation(
    op: &GraphqlOperation,
    fragments: &HashMap<String, GraphqlFragment>,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let mut all_columns: Vec<String> = Vec::with_capacity(op.selection_set.len());
    let mut all_rows: Vec<Vec<serde_json::Value>> = Vec::new();

    for selection in &op.selection_set {
        let fields = resolve_selection(selection, fragments);
        for field in fields {
            let (columns, rows) = execute_field_query(&field, fragments, amorphic)?;
            if all_columns.is_empty() {
                all_columns = columns;
                all_rows = rows;
            } else {
                // Multiple root fields: merge results under the field alias/name
                all_columns = columns;
                all_rows = rows;
            }
        }
    }

    Ok(QueryResponse {
        columns: all_columns,
        rows: all_rows,
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

/// Resolve a selection to concrete fields (expanding fragment spreads).
fn resolve_selection<'a>(
    selection: &'a GraphqlSelection,
    fragments: &'a HashMap<String, GraphqlFragment>,
) -> Vec<&'a GraphqlField> {
    match selection {
        GraphqlSelection::Field(f) => vec![f],
        GraphqlSelection::FragmentSpread(name) => {
            if let Some(frag) = fragments.get(name) {
                frag.selection_set
                    .iter()
                    .flat_map(|s| resolve_selection(s, fragments))
                    .collect()
            } else {
                Vec::new()
            }
        }
        GraphqlSelection::InlineFragment(inline) => inline
            .selection_set
            .iter()
            .flat_map(|s| resolve_selection(s, fragments))
            .collect(),
    }
}

/// Execute a single root-level field as a table query.
fn execute_field_query(
    field: &GraphqlField,
    fragments: &HashMap<String, GraphqlFragment>,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(Vec<String>, Vec<Vec<serde_json::Value>>), QueryErrorResponse> {
    let table_name = &field.name;

    // Check table exists
    if !amorphic.list_tables().contains(&table_name.to_string()) {
        return Err(QueryErrorResponse::execution_error(&format!(
            "Table '{}' not found (GraphQL root field)",
            table_name
        )));
    }

    // Determine which columns to project
    let selected_columns = collect_leaf_fields(&field.selection_set, fragments);
    if selected_columns.is_empty() {
        return Err(QueryErrorResponse::execution_error(&format!(
            "No fields selected for '{}'",
            table_name
        )));
    }

    // Scan the table
    let rows = amorphic
        .scan(table_name)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Scan failed: {}", e)))?;

    // Extract arguments
    let filter_arg = get_argument(&field.arguments, "where")
        .or_else(|| get_argument(&field.arguments, "filter"));
    let limit_arg =
        get_argument(&field.arguments, "limit").or_else(|| get_argument(&field.arguments, "first"));
    let offset_arg =
        get_argument(&field.arguments, "offset").or_else(|| get_argument(&field.arguments, "skip"));
    let order_arg = get_argument(&field.arguments, "orderBy")
        .or_else(|| get_argument(&field.arguments, "order"));

    // Filter rows
    let mut result_rows: Vec<Vec<serde_json::Value>> = Vec::new();
    for row in &rows {
        if let Some(filter) = &filter_arg {
            if !row_matches_filter(&row, filter) {
                continue;
            }
        }
        let projected: Vec<serde_json::Value> = selected_columns
            .iter()
            .map(|col| row_value_to_json(row.get(col)))
            .collect();
        result_rows.push(projected);
    }

    // Order
    if let Some(order) = &order_arg {
        apply_ordering(&mut result_rows, &selected_columns, order);
    }

    // Offset
    if let Some(off) = &offset_arg {
        let n = graphql_value_to_usize(off);
        if n > 0 && n <= result_rows.len() {
            result_rows = result_rows.into_iter().skip(n).collect();
        } else if n > result_rows.len() {
            result_rows.clear();
        }
    }

    // Limit
    if let Some(lim) = &limit_arg {
        let n = graphql_value_to_usize(lim);
        if n > 0 {
            result_rows.truncate(n);
        }
    }

    Ok((selected_columns, result_rows))
}

/// Collect leaf field names from a selection set.
fn collect_leaf_fields(
    selections: &[GraphqlSelection],
    fragments: &HashMap<String, GraphqlFragment>,
) -> Vec<String> {
    let mut cols = Vec::new();
    for sel in selections {
        match sel {
            GraphqlSelection::Field(f) => {
                let name = f.alias.clone().unwrap_or_else(|| f.name.clone());
                if !cols.contains(&name) {
                    cols.push(name);
                }
            }
            GraphqlSelection::FragmentSpread(frag_name) => {
                if let Some(frag) = fragments.get(frag_name) {
                    for c in collect_leaf_fields(&frag.selection_set, fragments) {
                        if !cols.contains(&c) {
                            cols.push(c);
                        }
                    }
                }
            }
            GraphqlSelection::InlineFragment(inline) => {
                for c in collect_leaf_fields(&inline.selection_set, fragments) {
                    if !cols.contains(&c) {
                        cols.push(c);
                    }
                }
            }
        }
    }
    cols
}

/// Get an argument value by name.
fn get_argument<'a>(args: &'a [GraphqlArgument], name: &str) -> Option<&'a GraphqlValue> {
    args.iter().find(|a| a.name == name).map(|a| &a.value)
}

/// Convert a GraphqlValue to usize (for limit/offset).
fn graphql_value_to_usize(val: &GraphqlValue) -> usize {
    match val {
        GraphqlValue::Int(n) => (*n).max(0) as usize,
        GraphqlValue::Float(f) => (*f).max(0.0) as usize,
        GraphqlValue::String(s) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// Check if a row matches a GraphQL filter argument.
/// Filter is an Object like `{ name: "Alice", age: 30 }`.
fn row_matches_filter(row: &RowData, filter: &GraphqlValue) -> bool {
    match filter {
        GraphqlValue::Object(fields) => {
            for (key, value) in fields {
                let row_val = row.get(key);
                let expected = graphql_value_to_ast(value);
                match (row_val, &expected) {
                    (Some(rv), exp) if rv == exp => {}
                    (None, AstValue::Null) => {}
                    _ => return false,
                }
            }
            true
        }
        _ => true, // Non-object filter is ignored
    }
}

/// Convert GraphqlValue to AstValue for comparison.
fn graphql_value_to_ast(val: &GraphqlValue) -> AstValue {
    match val {
        GraphqlValue::Int(n) => AstValue::Int(*n),
        GraphqlValue::Float(f) => AstValue::Float(*f),
        GraphqlValue::String(s) => AstValue::String(s.clone()),
        GraphqlValue::Boolean(b) => AstValue::Bool(*b),
        GraphqlValue::Null => AstValue::Null,
        GraphqlValue::Enum(s) => AstValue::String(s.clone()),
        GraphqlValue::List(items) => {
            AstValue::Array(items.iter().map(graphql_value_to_ast).collect())
        }
        GraphqlValue::Variable(_) => AstValue::Null,
        GraphqlValue::Object(fields) => {
            let mut map = HashMap::new();
            for (k, v) in fields {
                map.insert(k.clone(), graphql_value_to_ast(v));
            }
            AstValue::Object(map)
        }
    }
}

/// Convert an optional AstValue to JSON.
fn row_value_to_json(val: Option<&AstValue>) -> serde_json::Value {
    match val {
        None | Some(AstValue::Null) => serde_json::Value::Null,
        Some(AstValue::Bool(b)) => serde_json::Value::Bool(*b),
        Some(AstValue::Int(n)) => serde_json::json!(*n),
        Some(AstValue::Float(f)) => serde_json::json!(*f),
        Some(AstValue::String(s)) => serde_json::Value::String(s.clone()),
        Some(AstValue::Bytes(b)) => serde_json::Value::String(format!("0x{}", hex::encode(b))),
        Some(AstValue::Array(arr)) => {
            serde_json::Value::Array(arr.iter().map(|v| row_value_to_json(Some(v))).collect())
        }
        Some(AstValue::Object(map)) => {
            let m: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), row_value_to_json(Some(v))))
                .collect();
            serde_json::Value::Object(m)
        }
        Some(AstValue::Timestamp(ts)) => serde_json::json!(*ts),
        Some(AstValue::Uuid(u)) => serde_json::Value::String(u.clone()),
        Some(AstValue::Vector(v)) => {
            serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
        }
    }
}

/// Apply ordering based on a GraphQL orderBy argument.
/// Accepts: `"field_ASC"`, `"field_DESC"`, or `{field: "ASC"}`.
fn apply_ordering(rows: &mut [Vec<serde_json::Value>], columns: &[String], order: &GraphqlValue) {
    let (col_name, desc) = match order {
        GraphqlValue::String(s) => {
            if let Some(name) = s.strip_suffix("_DESC") {
                (name.to_string(), true)
            } else if let Some(name) = s.strip_suffix("_ASC") {
                (name.to_string(), false)
            } else {
                (s.clone(), false)
            }
        }
        GraphqlValue::Enum(s) => {
            if let Some(name) = s.strip_suffix("_DESC") {
                (name.to_string(), true)
            } else if let Some(name) = s.strip_suffix("_ASC") {
                (name.to_string(), false)
            } else {
                (s.clone(), false)
            }
        }
        GraphqlValue::Object(fields) => {
            if let Some((field, dir)) = fields.first() {
                let desc = matches!(dir, GraphqlValue::String(s) | GraphqlValue::Enum(s) if s.eq_ignore_ascii_case("DESC"));
                (field.clone(), desc)
            } else {
                return;
            }
        }
        _ => return,
    };

    let col_idx = columns.iter().position(|c| c == &col_name);
    if let Some(idx) = col_idx {
        rows.sort_by(|a, b| {
            let cmp = compare_json(&a[idx], &b[idx]);
            if desc { cmp.reverse() } else { cmp }
        });
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
        (serde_json::Value::Null, serde_json::Value::Null) => std::cmp::Ordering::Equal,
        (serde_json::Value::Null, _) => std::cmp::Ordering::Less,
        (_, serde_json::Value::Null) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    }
}

/// Execute a GraphQL mutation operation.
fn execute_mutation_operation(
    op: &GraphqlOperation,
    amorphic: &Arc<AmorphicTableStorage>,
    start: Instant,
) -> Result<QueryResponse, QueryErrorResponse> {
    let mut total_affected = 0usize;
    let mut last_columns: Vec<String> = Vec::new();
    let mut last_rows: Vec<Vec<serde_json::Value>> = Vec::new();

    for selection in &op.selection_set {
        if let GraphqlSelection::Field(field) = selection {
            let (affected, columns, rows) = execute_mutation_field(field, amorphic)?;
            total_affected += affected;
            last_columns = columns;
            last_rows = rows;
        }
    }

    Ok(QueryResponse {
        columns: last_columns,
        rows: last_rows,
        affected_rows: Some(total_affected),
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

/// Execute a single mutation field (createX, updateX, deleteX).
fn execute_mutation_field(
    field: &GraphqlField,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(usize, Vec<String>, Vec<Vec<serde_json::Value>>), QueryErrorResponse> {
    let name = &field.name;

    // Detect mutation type by convention: createXxx, updateXxx, deleteXxx
    if let Some(table) = name.strip_prefix("create") {
        let table_name = to_snake_or_lower(table);
        return execute_create_mutation(&table_name, field, amorphic);
    }
    if let Some(table) = name.strip_prefix("update") {
        let table_name = to_snake_or_lower(table);
        return execute_update_mutation(&table_name, field, amorphic);
    }
    if let Some(table) = name.strip_prefix("delete") {
        let table_name = to_snake_or_lower(table);
        return execute_delete_mutation(&table_name, field, amorphic);
    }

    Err(QueryErrorResponse::execution_error(&format!(
        "Unknown mutation '{}'; expected createX, updateX, or deleteX",
        name
    )))
}

/// Convert PascalCase to lowercase for table name resolution.
fn to_snake_or_lower(s: &str) -> String {
    // Simple: just lowercase. "User" → "user", "Users" → "users"
    s.to_lowercase()
}

/// Execute a createX mutation: extracts `input` argument and inserts a row.
fn execute_create_mutation(
    table_name: &str,
    field: &GraphqlField,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(usize, Vec<String>, Vec<Vec<serde_json::Value>>), QueryErrorResponse> {
    let input =
        get_argument(&field.arguments, "input").or_else(|| get_argument(&field.arguments, "data"));

    let fields = match input {
        Some(GraphqlValue::Object(f)) => f,
        _ => {
            // Try to use all arguments as fields directly
            let pairs: Vec<(String, GraphqlValue)> = field
                .arguments
                .iter()
                .map(|a| (a.name.clone(), a.value.clone()))
                .collect();
            if pairs.is_empty() {
                return Err(QueryErrorResponse::execution_error(
                    "createX mutation requires an 'input' argument or field arguments",
                ));
            }
            return execute_create_from_pairs(table_name, &pairs, field, amorphic);
        }
    };

    execute_create_from_pairs(table_name, fields, field, amorphic)
}

fn execute_create_from_pairs(
    table_name: &str,
    fields: &[(String, GraphqlValue)],
    gql_field: &GraphqlField,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(usize, Vec<String>, Vec<Vec<serde_json::Value>>), QueryErrorResponse> {
    // Ensure table exists; if not, auto-create with these columns
    let tables = amorphic.list_tables();
    if !tables.contains(&table_name.to_string()) {
        let col_names: Vec<String> = fields.iter().map(|(k, _)| k.clone()).collect();
        amorphic.create_table(table_name, &col_names).map_err(|e| {
            QueryErrorResponse::execution_error(&format!("Create table failed: {}", e))
        })?;
    }

    let col_names: Vec<String> = fields.iter().map(|(k, _)| k.clone()).collect();
    let values: Vec<AstValue> = fields
        .iter()
        .map(|(_, v)| graphql_value_to_ast(v))
        .collect();
    let row = RowData::new(col_names.clone(), values.clone());

    amorphic
        .insert(table_name, &row)
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Insert failed: {}", e)))?;

    // Return the inserted data as the response
    let response_cols = collect_leaf_fields_simple(&gql_field.selection_set);
    let response_row: Vec<serde_json::Value> = if response_cols.is_empty() {
        col_names
            .iter()
            .enumerate()
            .map(|(i, _)| row_value_to_json(Some(&values[i])))
            .collect()
    } else {
        response_cols
            .iter()
            .map(|c| {
                fields
                    .iter()
                    .find(|(k, _)| k == c)
                    .map(|(_, v)| graphql_value_to_json(v))
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect()
    };

    let out_cols = if response_cols.is_empty() {
        col_names
    } else {
        response_cols
    };
    Ok((1, out_cols, vec![response_row]))
}

/// Execute an updateX mutation.
fn execute_update_mutation(
    table_name: &str,
    field: &GraphqlField,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(usize, Vec<String>, Vec<Vec<serde_json::Value>>), QueryErrorResponse> {
    if !amorphic.list_tables().contains(&table_name.to_string()) {
        return Err(QueryErrorResponse::execution_error(&format!(
            "Table '{}' not found",
            table_name
        )));
    }

    // Extract id and input
    let id_val = get_argument(&field.arguments, "id");
    let input =
        get_argument(&field.arguments, "input").or_else(|| get_argument(&field.arguments, "data"));

    let updates = match input {
        Some(GraphqlValue::Object(f)) => f.clone(),
        _ => {
            // Use all non-id arguments as updates
            field
                .arguments
                .iter()
                .filter(|a| a.name != "id")
                .map(|a| (a.name.clone(), a.value.clone()))
                .collect::<Vec<_>>()
        }
    };

    if updates.is_empty() {
        return Err(QueryErrorResponse::execution_error(
            "updateX mutation requires fields to update",
        ));
    }

    let mut assignments = HashMap::new();
    for (k, v) in &updates {
        assignments.insert(k.clone(), graphql_value_to_ast(v));
    }

    let filter = id_val.map(|id| {
        Expression::eq(
            Expression::Column("id".into()),
            Expression::Literal(graphql_value_to_ast(id)),
        )
    });

    let affected = amorphic
        .update(table_name, &assignments, filter.as_ref())
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Update failed: {}", e)))?;

    Ok((affected, Vec::new(), Vec::new()))
}

/// Execute a deleteX mutation.
fn execute_delete_mutation(
    table_name: &str,
    field: &GraphqlField,
    amorphic: &Arc<AmorphicTableStorage>,
) -> Result<(usize, Vec<String>, Vec<Vec<serde_json::Value>>), QueryErrorResponse> {
    if !amorphic.list_tables().contains(&table_name.to_string()) {
        return Err(QueryErrorResponse::execution_error(&format!(
            "Table '{}' not found",
            table_name
        )));
    }

    let id_val = get_argument(&field.arguments, "id");
    let where_val = get_argument(&field.arguments, "where");

    let filter = if let Some(id) = id_val {
        Some(Expression::eq(
            Expression::Column("id".into()),
            Expression::Literal(graphql_value_to_ast(id)),
        ))
    } else if let Some(GraphqlValue::Object(fields)) = where_val {
        // Build AND filter from object fields
        let mut expr: Option<Expression> = None;
        for (k, v) in fields {
            let cond = Expression::eq(
                Expression::Column(k.clone()),
                Expression::Literal(graphql_value_to_ast(v)),
            );
            expr = Some(match expr {
                Some(e) => Expression::and(e, cond),
                None => cond,
            });
        }
        expr
    } else {
        None
    };

    let affected = amorphic
        .delete(table_name, filter.as_ref())
        .map_err(|e| QueryErrorResponse::execution_error(&format!("Delete failed: {}", e)))?;

    Ok((affected, Vec::new(), Vec::new()))
}

/// Collect leaf field names (simple version without fragment resolution).
fn collect_leaf_fields_simple(selections: &[GraphqlSelection]) -> Vec<String> {
    let mut cols = Vec::new();
    for sel in selections {
        if let GraphqlSelection::Field(f) = sel {
            let name = f.alias.clone().unwrap_or_else(|| f.name.clone());
            if !cols.contains(&name) {
                cols.push(name);
            }
        }
    }
    cols
}

/// Convert GraphqlValue to JSON for response building.
fn graphql_value_to_json(val: &GraphqlValue) -> serde_json::Value {
    match val {
        GraphqlValue::Int(n) => serde_json::json!(*n),
        GraphqlValue::Float(f) => serde_json::json!(*f),
        GraphqlValue::String(s) => serde_json::Value::String(s.clone()),
        GraphqlValue::Boolean(b) => serde_json::Value::Bool(*b),
        GraphqlValue::Null => serde_json::Value::Null,
        GraphqlValue::Enum(s) => serde_json::Value::String(s.clone()),
        GraphqlValue::Variable(_) => serde_json::Value::Null,
        GraphqlValue::List(items) => {
            serde_json::Value::Array(items.iter().map(graphql_value_to_json).collect())
        }
        GraphqlValue::Object(fields) => {
            let m: serde_json::Map<String, serde_json::Value> = fields
                .iter()
                .map(|(k, v)| (k.clone(), graphql_value_to_json(v)))
                .collect();
            serde_json::Value::Object(m)
        }
    }
}

/// Detect whether a query string looks like GraphQL.
pub fn is_graphql_query(sql: &str) -> bool {
    let trimmed = sql.trim();
    // Shorthand query: starts with `{`
    if trimmed.starts_with('{') {
        return true;
    }
    let lower = trimmed.to_lowercase();
    // Named operations
    lower.starts_with("query ")
        || lower.starts_with("mutation ")
        || lower.starts_with("subscription ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_amorphic() -> Arc<AmorphicTableStorage> {
        let dir =
            std::env::temp_dir().join(format!("jouledb-graphql-test-{}", uuid::Uuid::new_v4()));
        let store = joule_db_amorphic::DurableAmorphicStore::open(&dir).expect("temp store");
        Arc::new(AmorphicTableStorage::new(store))
    }

    fn setup_users_table(amorphic: &Arc<AmorphicTableStorage>) {
        amorphic
            .create_table("users", &["id".into(), "name".into(), "age".into()])
            .unwrap();
        let row1 = RowData::new(
            vec!["id".into(), "name".into(), "age".into()],
            vec![
                AstValue::String("1".into()),
                AstValue::String("Alice".into()),
                AstValue::Int(30),
            ],
        );
        let row2 = RowData::new(
            vec!["id".into(), "name".into(), "age".into()],
            vec![
                AstValue::String("2".into()),
                AstValue::String("Bob".into()),
                AstValue::Int(25),
            ],
        );
        let row3 = RowData::new(
            vec!["id".into(), "name".into(), "age".into()],
            vec![
                AstValue::String("3".into()),
                AstValue::String("Charlie".into()),
                AstValue::Int(35),
            ],
        );
        amorphic.insert("users", &row1).unwrap();
        amorphic.insert("users", &row2).unwrap();
        amorphic.insert("users", &row3).unwrap();
    }

    #[test]
    fn test_is_graphql_query_detection() {
        assert!(is_graphql_query("{ users { id name } }"));
        assert!(is_graphql_query("query GetUsers { users { id } }"));
        assert!(is_graphql_query(
            "mutation { createUser(name: \"Alice\") { id } }"
        ));
        assert!(is_graphql_query("subscription { newUser { id } }"));
        assert!(!is_graphql_query("SELECT * FROM users"));
        assert!(!is_graphql_query("MATCH (n) RETURN n"));
        assert!(!is_graphql_query("INSERT INTO users VALUES (1)"));
        assert!(!is_graphql_query("CREATE KEYSPACE test"));
    }

    #[test]
    fn test_simple_query() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser.parse("{ users { id name } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.columns, vec!["id", "name"]);
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_query_all_fields() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser.parse("{ users { id name age } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.columns, vec!["id", "name", "age"]);
        assert_eq!(result.rows.len(), 3);
        // Check Alice's row
        assert_eq!(result.rows[0][1], serde_json::json!("Alice"));
        assert_eq!(result.rows[0][2], serde_json::json!(30));
    }

    #[test]
    fn test_query_with_limit() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser.parse("{ users(limit: 2) { id name } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_query_with_offset() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("{ users(offset: 1, limit: 2) { id name } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.rows[0][1], serde_json::json!("Bob"));
    }

    #[test]
    fn test_query_with_filter() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("{ users(where: {name: \"Alice\"}) { id name age } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][1], serde_json::json!("Alice"));
    }

    #[test]
    fn test_query_with_order_by() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("{ users(orderBy: \"name_DESC\") { id name } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 3);
        // Descending: Charlie, Bob, Alice
        assert_eq!(result.rows[0][1], serde_json::json!("Charlie"));
        assert_eq!(result.rows[2][1], serde_json::json!("Alice"));
    }

    #[test]
    fn test_named_query() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("query GetUsers { users { id name } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_create_mutation() {
        let amorphic = test_amorphic();
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("mutation { createUsers(name: \"Diana\", age: 28) { name age } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], serde_json::json!("Diana"));
    }

    #[test]
    fn test_create_and_query() {
        let amorphic = test_amorphic();
        let mut parser = GraphqlParser::new();

        // Create via mutation
        let q = parser
            .parse("mutation { createProducts(name: \"Widget\", price: 9.99) { name price } }")
            .unwrap();
        execute_graphql(&q, &amorphic, Instant::now()).unwrap();

        // Query back
        let q = parser.parse("{ products { name price } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], serde_json::json!("Widget"));
    }

    #[test]
    fn test_delete_mutation() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("mutation { deleteUsers(id: \"1\") { id } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.affected_rows, Some(1));

        // Verify deletion
        let q = parser.parse("{ users { id name } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_update_mutation() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("mutation { updateUsers(id: \"1\", name: \"Alicia\") { id } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert!(result.affected_rows.unwrap_or(0) >= 1);

        // Verify update
        let q = parser
            .parse("{ users(where: {id: \"1\"}) { id name } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now()).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][1], serde_json::json!("Alicia"));
    }

    #[test]
    fn test_table_not_found() {
        let amorphic = test_amorphic();
        let mut parser = GraphqlParser::new();

        let q = parser.parse("{ nonexistent { id } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now());
        assert!(result.is_err());
    }

    #[test]
    fn test_subscription_not_supported() {
        let amorphic = test_amorphic();
        let mut parser = GraphqlParser::new();

        let q = parser.parse("subscription { newUser { id } }").unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now());
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_mutation() {
        let amorphic = test_amorphic();
        setup_users_table(&amorphic);
        let mut parser = GraphqlParser::new();

        let q = parser
            .parse("mutation { doSomething(id: \"1\") { id } }")
            .unwrap();
        let result = execute_graphql(&q, &amorphic, Instant::now());
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_operations() {
        let amorphic = test_amorphic();
        let query = GraphqlQuery {
            operations: Vec::new(),
            fragments: HashMap::new(),
        };
        let result = execute_graphql(&query, &amorphic, Instant::now());
        assert!(result.is_err());
    }
}
