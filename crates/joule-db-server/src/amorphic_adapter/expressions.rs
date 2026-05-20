//! Expression evaluation: predicate evaluation, arithmetic, functions, UPDATE/DELETE with expressions.

use super::*;
use super::conversions::*;

/// Namespace for expression evaluation methods that are logically associated with AmorphicTableStorage
/// but don't take `&self` — they operate on RowData directly.
pub(super) struct AmorphicExprEval;

impl AmorphicExprEval {
    /// Evaluate a simple WHERE predicate against row data.
    pub(super) fn evaluate_predicate(row: &RowData, predicate: &Expression) -> bool {
        match predicate {
            Expression::Binary { left, op, right } => match op {
                Operator::And => {
                    Self::evaluate_predicate(row, left) && Self::evaluate_predicate(row, right)
                }
                Operator::Or => {
                    Self::evaluate_predicate(row, left) || Self::evaluate_predicate(row, right)
                }
                _ => {
                    let left_val = Self::eval_expr(row, left);
                    let right_val = Self::eval_expr(row, right);
                    compare_values(&left_val, &right_val, op)
                }
            },
            Expression::Unary {
                op: UnaryOperator::Not,
                expr,
            } => !Self::evaluate_predicate(row, expr),
            Expression::IsNull { expr, negated } => {
                let is_null = matches!(Self::eval_expr(row, expr), AstValue::Null);
                if *negated { !is_null } else { is_null }
            }
            Expression::In { expr, list, negated } => {
                let val = Self::eval_expr(row, expr);
                let found = list.iter().any(|item| {
                    let item_val = Self::eval_expr(row, item);
                    compare_values(&val, &item_val, &Operator::Eq)
                });
                if *negated { !found } else { found }
            }
            Expression::Between { expr, low, high, negated } => {
                let val = Self::eval_expr(row, expr);
                let low_val = Self::eval_expr(row, low);
                let high_val = Self::eval_expr(row, high);
                let in_range = compare_values(&val, &low_val, &Operator::Ge)
                    && compare_values(&val, &high_val, &Operator::Le);
                if *negated { !in_range } else { in_range }
            }
            Expression::Like { expr, pattern, negated, case_insensitive } => {
                let val = Self::eval_expr(row, expr);
                let val_str = match &val {
                    AstValue::String(s) => s.clone(),
                    AstValue::Null => return if *negated { true } else { false },
                    other => format!("{:?}", other),
                };
                let text = if *case_insensitive { val_str.to_lowercase() } else { val_str };
                let pat = if *case_insensitive { pattern.to_lowercase() } else { pattern.clone() };
                let matched = like_match(&text, &pat);
                if *negated { !matched } else { matched }
            }
            _ => {
                // Try evaluating the expression -- if it returns a Bool, use it
                match Self::eval_expr(row, predicate) {
                    AstValue::Bool(b) => b,
                    AstValue::Null => false,
                    _ => true, // Non-boolean non-null values pass through
                }
            }
        }
    }

    /// Evaluate an expression to a value, given a row context
    pub(super) fn eval_expr(row: &RowData, expr: &Expression) -> AstValue {
        match expr {
            Expression::Column(name) => row.get(name).cloned().unwrap_or(AstValue::Null),
            Expression::QualifiedColumn { column, .. } => {
                row.get(column).cloned().unwrap_or(AstValue::Null)
            }
            Expression::Literal(v) => v.clone(),
            Expression::Binary { left, op, right } => {
                let l = Self::eval_expr(row, left);
                let r = Self::eval_expr(row, right);
                Self::eval_arithmetic(&l, &r, op)
            }
            Expression::Unary { op, expr } => {
                let val = Self::eval_expr(row, expr);
                match op {
                    UnaryOperator::Neg => match val {
                        AstValue::Int(n) => AstValue::Int(-n),
                        AstValue::Float(n) => AstValue::Float(-n),
                        _ => AstValue::Null,
                    },
                    UnaryOperator::Not => match val {
                        AstValue::Bool(b) => AstValue::Bool(!b),
                        _ => AstValue::Null,
                    },
                    _ => AstValue::Null,
                }
            }
            Expression::Function { name, args } => {
                Self::eval_function(row, name, args)
            }
            Expression::Case { operand, when_clauses, else_clause } => {
                if let Some(op_expr) = operand {
                    // Simple CASE: CASE expr WHEN val THEN result ...
                    let op_val = Self::eval_expr(row, op_expr);
                    for (when_expr, then_expr) in when_clauses {
                        let when_val = Self::eval_expr(row, when_expr);
                        if compare_values(&op_val, &when_val, &Operator::Eq) {
                            return Self::eval_expr(row, then_expr);
                        }
                    }
                } else {
                    // Searched CASE: CASE WHEN cond THEN result ...
                    for (when_expr, then_expr) in when_clauses {
                        if Self::evaluate_predicate(row, when_expr) {
                            return Self::eval_expr(row, then_expr);
                        }
                    }
                }
                if let Some(else_expr) = else_clause {
                    Self::eval_expr(row, else_expr)
                } else {
                    AstValue::Null
                }
            }
            Expression::IsNull { expr, negated } => {
                let is_null = matches!(Self::eval_expr(row, expr), AstValue::Null);
                AstValue::Bool(if *negated { !is_null } else { is_null })
            }
            Expression::Like { expr, pattern, negated, case_insensitive } => {
                let val = Self::eval_expr(row, expr);
                let val_str = match &val {
                    AstValue::String(s) => s.clone(),
                    AstValue::Null => return AstValue::Bool(*negated),
                    other => format!("{:?}", other),
                };
                let text = if *case_insensitive { val_str.to_lowercase() } else { val_str };
                let pat = if *case_insensitive { pattern.to_lowercase() } else { pattern.clone() };
                let matched = like_match(&text, &pat);
                AstValue::Bool(if *negated { !matched } else { matched })
            }
            Expression::Between { expr, low, high, negated } => {
                let val = Self::eval_expr(row, expr);
                let low_val = Self::eval_expr(row, low);
                let high_val = Self::eval_expr(row, high);
                let in_range = compare_values(&val, &low_val, &Operator::Ge)
                    && compare_values(&val, &high_val, &Operator::Le);
                AstValue::Bool(if *negated { !in_range } else { in_range })
            }
            Expression::In { expr, list, negated } => {
                let val = Self::eval_expr(row, expr);
                let found = list.iter().any(|item| {
                    let item_val = Self::eval_expr(row, item);
                    compare_values(&val, &item_val, &Operator::Eq)
                });
                AstValue::Bool(if *negated { !found } else { found })
            }
            _ => AstValue::Null,
        }
    }

    /// Evaluate arithmetic operations between two values
    fn eval_arithmetic(left: &AstValue, right: &AstValue, op: &Operator) -> AstValue {
        match op {
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div | Operator::Mod => {
                // Try integer arithmetic first
                if let (Some(l), Some(r)) = (Self::as_i64(left), Self::as_i64(right)) {
                    match op {
                        Operator::Add => AstValue::Int(l.wrapping_add(r)),
                        Operator::Sub => AstValue::Int(l.wrapping_sub(r)),
                        Operator::Mul => AstValue::Int(l.wrapping_mul(r)),
                        Operator::Div => if r != 0 { AstValue::Int(l / r) } else { AstValue::Null },
                        Operator::Mod => if r != 0 { AstValue::Int(l % r) } else { AstValue::Null },
                        _ => AstValue::Null,
                    }
                } else if let (Some(l), Some(r)) = (Self::as_f64(left), Self::as_f64(right)) {
                    match op {
                        Operator::Add => AstValue::Float(l + r),
                        Operator::Sub => AstValue::Float(l - r),
                        Operator::Mul => AstValue::Float(l * r),
                        Operator::Div => if r != 0.0 { AstValue::Float(l / r) } else { AstValue::Null },
                        Operator::Mod => if r != 0.0 { AstValue::Float(l % r) } else { AstValue::Null },
                        _ => AstValue::Null,
                    }
                } else {
                    // String concatenation for ||
                    AstValue::Null
                }
            }
            // Comparison operators return Bool
            Operator::Eq | Operator::Ne | Operator::Lt | Operator::Le | Operator::Gt | Operator::Ge => {
                AstValue::Bool(compare_values(left, right, op))
            }
            // String concat
            Operator::Concat => {
                let l_str = Self::val_to_string(left);
                let r_str = Self::val_to_string(right);
                AstValue::String(format!("{}{}", l_str, r_str))
            }
            _ => AstValue::Null,
        }
    }

    fn as_i64(val: &AstValue) -> Option<i64> {
        match val {
            AstValue::Int(n) => Some(*n),
            _ => None,
        }
    }

    fn as_f64(val: &AstValue) -> Option<f64> {
        match val {
            AstValue::Float(n) => Some(*n),
            AstValue::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    fn val_to_string(val: &AstValue) -> String {
        match val {
            AstValue::String(s) => s.clone(),
            AstValue::Int(n) => n.to_string(),
            AstValue::Float(n) => n.to_string(),
            AstValue::Bool(b) => b.to_string(),
            AstValue::Null => "NULL".to_string(),
            _ => format!("{:?}", val),
        }
    }

    /// Evaluate built-in SQL functions
    fn eval_function(row: &RowData, name: &str, args: &[Expression]) -> AstValue {
        match name.to_uppercase().as_str() {
            "COALESCE" => {
                for arg in args {
                    let val = Self::eval_expr(row, arg);
                    if !val.is_null() {
                        return val;
                    }
                }
                AstValue::Null
            }
            "UPPER" => {
                if let Some(arg) = args.first() {
                    match Self::eval_expr(row, arg) {
                        AstValue::String(s) => AstValue::String(s.to_uppercase()),
                        other => other,
                    }
                } else { AstValue::Null }
            }
            "LOWER" => {
                if let Some(arg) = args.first() {
                    match Self::eval_expr(row, arg) {
                        AstValue::String(s) => AstValue::String(s.to_lowercase()),
                        other => other,
                    }
                } else { AstValue::Null }
            }
            "LENGTH" | "LEN" | "CHAR_LENGTH" => {
                if let Some(arg) = args.first() {
                    match Self::eval_expr(row, arg) {
                        AstValue::String(s) => AstValue::Int(s.len() as i64),
                        AstValue::Null => AstValue::Null,
                        _ => AstValue::Null,
                    }
                } else { AstValue::Null }
            }
            "TRIM" => {
                if let Some(arg) = args.first() {
                    match Self::eval_expr(row, arg) {
                        AstValue::String(s) => AstValue::String(s.trim().to_string()),
                        other => other,
                    }
                } else { AstValue::Null }
            }
            "ABS" => {
                if let Some(arg) = args.first() {
                    match Self::eval_expr(row, arg) {
                        AstValue::Int(n) => AstValue::Int(n.abs()),
                        AstValue::Float(n) => AstValue::Float(n.abs()),
                        other => other,
                    }
                } else { AstValue::Null }
            }
            "ROUND" => {
                if let Some(arg) = args.first() {
                    match Self::eval_expr(row, arg) {
                        AstValue::Float(n) => {
                            let places = args.get(1)
                                .and_then(|a| if let AstValue::Int(p) = Self::eval_expr(row, a) { Some(p) } else { None })
                                .unwrap_or(0);
                            let factor = 10f64.powi(places as i32);
                            AstValue::Float((n * factor).round() / factor)
                        }
                        AstValue::Int(n) => AstValue::Int(n),
                        other => other,
                    }
                } else { AstValue::Null }
            }
            "SUBSTR" | "SUBSTRING" => {
                if args.len() >= 2 {
                    let s = match Self::eval_expr(row, &args[0]) {
                        AstValue::String(s) => s,
                        _ => return AstValue::Null,
                    };
                    let start = match Self::eval_expr(row, &args[1]) {
                        AstValue::Int(n) => (n - 1).max(0) as usize,
                        _ => return AstValue::Null,
                    };
                    let chars: Vec<char> = s.chars().collect();
                    if let Some(len_arg) = args.get(2) {
                        let len = match Self::eval_expr(row, len_arg) {
                            AstValue::Int(n) => n.max(0) as usize,
                            _ => return AstValue::Null,
                        };
                        AstValue::String(chars.iter().skip(start).take(len).collect())
                    } else {
                        AstValue::String(chars.iter().skip(start).collect())
                    }
                } else { AstValue::Null }
            }
            _ => AstValue::Null, // Unknown function
        }
    }
}

// ==================== Expression-based UPDATE/DELETE on AmorphicTableStorage ====================

impl AmorphicTableStorage {
    /// Columnar aggregate short-circuit.
    pub fn columnar_aggregate(
        &self,
        table: &str,
        agg_func: &str,
        column: &str,
    ) -> QueryResult<Option<f64>> {
        let store = self.store.read().map_err(lock_error)?;

        // Get table record IDs
        let table_result =
            store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let table_ids: std::collections::HashSet<u64> =
            table_result.records().iter().map(|r| r.id).collect();

        if table_ids.is_empty() {
            return Ok(None);
        }

        // Access columnar data for the target column
        let columnar = store.columnar();
        let col = match columnar.get_column(column) {
            Some(c) => c,
            None => return Ok(None),
        };

        // Filter columnar values to only those belonging to this table
        let mut sum = 0.0f64;
        let mut count = 0usize;
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;

        for (i, &record_id) in col.record_ids.iter().enumerate() {
            if table_ids.contains(&record_id) {
                let v = col.values[i];
                sum += v;
                count += 1;
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
        }

        if count == 0 {
            return Ok(None);
        }

        let result = match agg_func.to_uppercase().as_str() {
            "SUM" => sum,
            "AVG" => sum / count as f64,
            "COUNT" => count as f64,
            "MIN" => min_val,
            "MAX" => max_val,
            _ => return Ok(None),
        };

        Ok(Some(result))
    }

    /// Update matching rows by evaluating SET expressions per-row.
    pub fn update_with_expressions(
        &self,
        table: &str,
        assignments: &[(String, Expression)],
        predicate: Option<&Expression>,
        params: &[serde_json::Value],
    ) -> QueryResult<usize> {
        let mut store = self.store.write().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;
        let column_defs = self
            .get_column_defs_from_store(&store, table)
            .unwrap_or_default();

        // Find matching records
        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut updated = 0;

        // Collect matching record ids and their updates first to avoid borrow issues
        let mut pending_updates: Vec<(RecordId, HashMap<String, AmorphicValue>)> = Vec::new();

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                // Convert row to json for expression evaluation
                let json_row: Vec<serde_json::Value> = row
                    .values
                    .iter()
                    .map(|v| crate::json_ops::ast_value_to_json(v))
                    .collect();

                // Evaluate each SET expression with row context
                let updates: HashMap<String, AmorphicValue> = assignments
                    .iter()
                    .map(|(col, expr)| {
                        let json_val =
                            crate::query::evaluate_expression(expr, &columns, &json_row, params);
                        let ast_val = crate::json_ops::json_to_ast_value(&json_val);
                        (col.clone(), ast_to_amorphic(&ast_val))
                    })
                    .collect();

                // Build post-update row for constraint validation
                if !column_defs.is_empty() {
                    let mut updated_values = row.values.clone();
                    for (col, val) in &updates {
                        if let Some(idx) = columns.iter().position(|c| c == col) {
                            updated_values[idx] = amorphic_to_ast(val);
                        }
                    }
                    // Apply type coercion to updated values
                    coerce_row_types(&columns, &mut updated_values, &column_defs);
                    let updated_row = RowData::new(columns.clone(), updated_values);

                    // Validate CHECK constraints
                    crate::query::validate_check_constraints(&column_defs, &updated_row)
                        .map_err(|e| QueryError::ExecutionError(e))?;

                    // Validate UNIQUE constraints (excluding the row being updated)
                    Self::check_unique_in_store(
                        &store,
                        table,
                        &columns,
                        &column_defs,
                        &updated_row,
                        Some(record.id),
                    )?;

                    // Validate NOT NULL
                    for (col, val) in updated_row.columns.iter().zip(updated_row.values.iter()) {
                        if matches!(val, AstValue::Null) {
                            if let Some(def) = column_defs.iter().find(|d| d.name == *col) {
                                if !def.nullable {
                                    return Err(QueryError::ExecutionError(format!(
                                        "NOT NULL constraint failed: column '{}' cannot be NULL",
                                        col
                                    )));
                                }
                            }
                        }
                    }
                }

                pending_updates.push((record.id, updates));
            }
        }

        for (rid, updates) in pending_updates {
            store.update_fields(rid, updates).map_err(amorphic_error)?;
            updated += 1;
        }

        Ok(updated)
    }

    /// Like update_with_expressions, but also returns the updated rows as JSON.
    pub fn update_with_expressions_returning(
        &self,
        table: &str,
        assignments: &[(String, Expression)],
        predicate: Option<&Expression>,
        params: &[serde_json::Value],
    ) -> QueryResult<(usize, Vec<Vec<serde_json::Value>>)> {
        let mut store = self.store.write().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;
        let column_defs = self
            .get_column_defs_from_store(&store, table)
            .unwrap_or_default();

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut updated = 0;
        let mut returned_rows = Vec::new();

        // Collect pending updates for constraint validation before writing
        let mut pending: Vec<(
            RecordId,
            HashMap<String, AmorphicValue>,
            Vec<serde_json::Value>,
        )> = Vec::new();

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                let json_row: Vec<serde_json::Value> = row
                    .values
                    .iter()
                    .map(|v| crate::json_ops::ast_value_to_json(v))
                    .collect();

                let updates: HashMap<String, AmorphicValue> = assignments
                    .iter()
                    .map(|(col, expr)| {
                        let json_val =
                            crate::query::evaluate_expression(expr, &columns, &json_row, params);
                        let ast_val = crate::json_ops::json_to_ast_value(&json_val);
                        (col.clone(), ast_to_amorphic(&ast_val))
                    })
                    .collect();

                // Build post-update row for constraint validation
                if !column_defs.is_empty() {
                    let mut updated_values = row.values.clone();
                    for (col, val) in &updates {
                        if let Some(idx) = columns.iter().position(|c| c == col) {
                            updated_values[idx] = amorphic_to_ast(val);
                        }
                    }
                    coerce_row_types(&columns, &mut updated_values, &column_defs);
                    let updated_row_data = RowData::new(columns.clone(), updated_values);

                    crate::query::validate_check_constraints(&column_defs, &updated_row_data)
                        .map_err(|e| QueryError::ExecutionError(e))?;

                    Self::check_unique_in_store(
                        &store,
                        table,
                        &columns,
                        &column_defs,
                        &updated_row_data,
                        Some(record.id),
                    )?;

                    for (col, val) in updated_row_data
                        .columns
                        .iter()
                        .zip(updated_row_data.values.iter())
                    {
                        if matches!(val, AstValue::Null) {
                            if let Some(def) = column_defs.iter().find(|d| d.name == *col) {
                                if !def.nullable {
                                    return Err(QueryError::ExecutionError(format!(
                                        "NOT NULL constraint failed: column '{}' cannot be NULL",
                                        col
                                    )));
                                }
                            }
                        }
                    }
                }

                // Build the updated row for returning: start from original, apply updates
                let mut updated_json = json_row;
                for (col, val) in &updates {
                    if let Some(idx) = columns.iter().position(|c| c == col) {
                        updated_json[idx] = crate::json_ops::ast_value_to_json(&amorphic_to_ast(val));
                    }
                }

                pending.push((record.id, updates, updated_json));
            }
        }

        for (rid, updates, updated_json) in pending {
            store.update_fields(rid, updates).map_err(amorphic_error)?;
            returned_rows.push(updated_json);
            updated += 1;
        }

        Ok((updated, returned_rows))
    }

    /// Delete matching rows and return the deleted rows as JSON.
    pub fn delete_returning(
        &self,
        table: &str,
        predicate: Option<&Expression>,
    ) -> QueryResult<(usize, Vec<Vec<serde_json::Value>>)> {
        let mut store = self.store.write().map_err(lock_error)?;
        let columns = self.get_schema_columns(&store, table)?;

        let result = store.query_equals(TABLE_FIELD, &AmorphicValue::String(table.to_string()));
        let mut to_delete = Vec::new();

        for record in result.records() {
            let row = Self::record_to_row(record, &columns);
            let matches = predicate
                .map(|p| AmorphicExprEval::evaluate_predicate(&row, p))
                .unwrap_or(true);

            if matches {
                let json_row: Vec<serde_json::Value> = row
                    .values
                    .iter()
                    .map(|v| crate::json_ops::ast_value_to_json(v))
                    .collect();
                to_delete.push((record.id, json_row));
            }
        }

        let count = to_delete.len();
        let mut returned_rows = Vec::with_capacity(count);
        let deleted_ids: Vec<u64> = to_delete.iter().map(|(id, _)| *id).collect();
        for (id, json_row) in to_delete {
            store.delete(id).map_err(amorphic_error)?;
            returned_rows.push(json_row);
        }
        drop(store);

        // Clean up fulltext index entries for deleted records
        for id in &deleted_ids {
            let _ = self.update_fulltext_indexes_on_delete(table, *id);
        }

        Ok((count, returned_rows))
    }
}
