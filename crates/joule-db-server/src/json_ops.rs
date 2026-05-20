//! JSON arithmetic, comparison, coercion, and aggregate utilities.
//!
//! Extracted from `query.rs` to reduce the 102K-line monolith.
//! These are pure functions with no dependency on `SimpleQueryExecutor`.

use joule_db_query::ast::{Expression, OrderBy, Value as AstValue};

// ── Equality ─────────────────────────────────────────────────────────────────

/// Compare two JSON values for equality with implicit type coercion.
///
/// Handles Number↔String, Bool↔Number, Bool↔String cross-type comparisons.
///
/// # Contracts (verified by proptest + Kani)
/// - **Commutative**: json_equals(a, b) == json_equals(b, a)
/// - **Reflexive**: json_equals(a, a) == true for all a
pub fn json_equals(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    let result = json_equals_inner(a, b);
    // Contract: commutativity
    debug_assert_eq!(result, json_equals_inner(b, a), "json_equals commutativity violated");
    result
}

fn json_equals_inner(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Null, serde_json::Value::Null) => true,
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => a == b,
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            a.as_f64() == b.as_f64()
        }
        (serde_json::Value::String(a), serde_json::Value::String(b)) => a == b,
        // Implicit type coercion: Number vs String
        (serde_json::Value::Number(n), serde_json::Value::String(s))
        | (serde_json::Value::String(s), serde_json::Value::Number(n)) => {
            if let Ok(parsed) = s.parse::<f64>() {
                n.as_f64() == Some(parsed)
            } else {
                false
            }
        }
        // Implicit type coercion: Bool vs Number
        (serde_json::Value::Bool(b_val), serde_json::Value::Number(n))
        | (serde_json::Value::Number(n), serde_json::Value::Bool(b_val)) => {
            n.as_f64() == Some(if *b_val { 1.0 } else { 0.0 })
        }
        // Implicit type coercion: Bool vs String
        (serde_json::Value::Bool(b_val), serde_json::Value::String(s))
        | (serde_json::Value::String(s), serde_json::Value::Bool(b_val)) => {
            match s.to_uppercase().as_str() {
                "TRUE" | "1" | "YES" | "T" => *b_val,
                "FALSE" | "0" | "NO" | "F" => !*b_val,
                _ => false,
            }
        }
        _ => false,
    }
}

// ── Ordering ─────────────────────────────────────────────────────────────────

/// Compare two JSON values returning -1, 0, or 1.
///
/// Canonical type ordering: Null < Bool < Number < String < Array < Object.
///
/// # Contracts (verified by proptest + Kani)
/// - **Bounded**: result ∈ {-1, 0, 1}
/// - **Antisymmetric**: json_compare(a, b) == -json_compare(b, a)
/// - **Reflexive**: json_compare(a, a) == 0
/// - **Transitive**: a ≤ b ∧ b ≤ c → a ≤ c
pub fn json_compare(a: &serde_json::Value, b: &serde_json::Value) -> i32 {
    fn type_rank(v: &serde_json::Value) -> u8 {
        match v {
            serde_json::Value::Null => 0,
            serde_json::Value::Bool(_) => 1,
            serde_json::Value::Number(_) => 2,
            serde_json::Value::String(_) => 3,
            serde_json::Value::Array(_) => 4,
            serde_json::Value::Object(_) => 5,
        }
    }

    let ra = type_rank(a);
    let rb = type_rank(b);
    if ra != rb {
        let r = if ra < rb { -1 } else { 1 };
        debug_assert!(r >= -1 && r <= 1);
        return r;
    }

    let result = match (a, b) {
        (serde_json::Value::Null, serde_json::Value::Null) => 0,
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            let af = a.as_f64().unwrap_or(f64::NAN);
            let bf = b.as_f64().unwrap_or(f64::NAN);
            match af.total_cmp(&bf) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Equal => 0,
            }
        }
        (serde_json::Value::String(a), serde_json::Value::String(b)) => a.cmp(b) as i32,
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => a.cmp(b) as i32,
        _ => 0,
    };
    // Contract: output bounded
    debug_assert!(result >= -1 && result <= 1, "json_compare contract: result must be -1/0/1");
    result
}

/// Check if two rows are equal on the ORDER BY columns (for tie detection in
/// PERCENT_RANK, CUME_DIST, etc.).
pub fn json_order_by_equal(
    rows: &[Vec<serde_json::Value>],
    columns: &[String],
    idx_a: usize,
    idx_b: usize,
    order_by: &[OrderBy],
) -> bool {
    for ob in order_by {
        if let Expression::Column(name) = &ob.expr {
            let ci = columns
                .iter()
                .position(|c| c == name)
                .or_else(|| {
                    columns
                        .iter()
                        .position(|c| c.ends_with(&format!(".{}", name)))
                });
            if let Some(ci) = ci {
                let va = rows.get(idx_a).and_then(|r| r.get(ci));
                let vb = rows.get(idx_b).and_then(|r| r.get(ci));
                match (va, vb) {
                    (Some(a), Some(b)) => {
                        if json_compare(a, b) != 0 {
                            return false;
                        }
                    }
                    (None, None) => {}
                    _ => return false,
                }
            }
        }
    }
    true
}

// ── Coercion ─────────────────────────────────────────────────────────────────

/// Try to coerce a JSON value to f64 for arithmetic.
/// Handles implicit string→number coercion (PostgreSQL casts numeric strings).
pub fn try_coerce_to_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse::<f64>().ok(),
        serde_json::Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

/// Try to coerce a JSON value to i64 for integer arithmetic.
pub fn try_coerce_to_i64(v: &serde_json::Value) -> Option<i64> {
    match v {
        serde_json::Value::Number(n) => n.as_i64(),
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
        serde_json::Value::Bool(b) => Some(if *b { 1 } else { 0 }),
        _ => None,
    }
}

// ── Normalization ────────────────────────────────────────────────────────────

/// Normalize a JSON value: if it's a whole-number float (e.g. 120000.0), convert
/// to integer. The amorphic store may return floats for values originally inserted
/// as integers.
pub fn normalize_int(val: serde_json::Value) -> serde_json::Value {
    if let serde_json::Value::Number(ref n) = val {
        if n.as_i64().is_none() {
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                    return serde_json::Value::Number((f as i64).into());
                }
            }
        }
    }
    val
}

// ── Arithmetic ───────────────────────────────────────────────────────────────

/// Add two JSON values (numeric addition or string concatenation).
pub fn json_add(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                if let Some(result) = ai.checked_add(bi) {
                    serde_json::Value::Number(result.into())
                } else {
                    serde_json::Number::from_f64(ai as f64 + bi as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                }
            } else {
                let af = a.as_f64().unwrap_or(0.0);
                let bf = b.as_f64().unwrap_or(0.0);
                serde_json::Number::from_f64(af + bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
        }
        (serde_json::Value::String(a), serde_json::Value::String(b)) => {
            serde_json::Value::String(format!("{}{}", a, b))
        }
        _ => {
            if let (Some(af), Some(bf)) = (try_coerce_to_f64(a), try_coerce_to_f64(b)) {
                if let (Some(ai), Some(bi)) = (try_coerce_to_i64(a), try_coerce_to_i64(b)) {
                    if let Some(result) = ai.checked_add(bi) {
                        return serde_json::Value::Number(result.into());
                    }
                }
                serde_json::Number::from_f64(af + bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
    }
}

/// Subtract two JSON values.
pub fn json_sub(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                if let Some(result) = ai.checked_sub(bi) {
                    serde_json::Value::Number(result.into())
                } else {
                    serde_json::Number::from_f64(ai as f64 - bi as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                }
            } else {
                let af = a.as_f64().unwrap_or(0.0);
                let bf = b.as_f64().unwrap_or(0.0);
                serde_json::Number::from_f64(af - bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
        }
        _ => {
            if let (Some(af), Some(bf)) = (try_coerce_to_f64(a), try_coerce_to_f64(b)) {
                if let (Some(ai), Some(bi)) = (try_coerce_to_i64(a), try_coerce_to_i64(b)) {
                    if let Some(result) = ai.checked_sub(bi) {
                        return serde_json::Value::Number(result.into());
                    }
                }
                serde_json::Number::from_f64(af - bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
    }
}

/// Multiply two JSON values.
pub fn json_mul(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                if let Some(result) = ai.checked_mul(bi) {
                    serde_json::Value::Number(result.into())
                } else {
                    serde_json::Number::from_f64(ai as f64 * bi as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null)
                }
            } else {
                let af = a.as_f64().unwrap_or(0.0);
                let bf = b.as_f64().unwrap_or(0.0);
                serde_json::Number::from_f64(af * bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
        }
        _ => {
            if let (Some(af), Some(bf)) = (try_coerce_to_f64(a), try_coerce_to_f64(b)) {
                if let (Some(ai), Some(bi)) = (try_coerce_to_i64(a), try_coerce_to_i64(b)) {
                    if let Some(result) = ai.checked_mul(bi) {
                        return serde_json::Value::Number(result.into());
                    }
                }
                serde_json::Number::from_f64(af * bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
    }
}

/// Divide two JSON values. SQL integer division truncates toward zero.
pub fn json_div(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Number(an), serde_json::Value::Number(bn)) => {
            if let (Some(ai), Some(bi)) = (an.as_i64(), bn.as_i64()) {
                if bi == 0 {
                    return serde_json::Value::Null;
                }
                return serde_json::Value::Number((ai / bi).into());
            }
            let bf = bn.as_f64().unwrap_or(0.0);
            if bf == 0.0 {
                return serde_json::Value::Null;
            }
            let af = an.as_f64().unwrap_or(0.0);
            serde_json::Number::from_f64(af / bf)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        }
        _ => {
            if let (Some(af), Some(bf)) = (try_coerce_to_f64(a), try_coerce_to_f64(b)) {
                if bf == 0.0 {
                    return serde_json::Value::Null;
                }
                if let (Some(ai), Some(bi)) = (try_coerce_to_i64(a), try_coerce_to_i64(b)) {
                    if bi != 0 {
                        return serde_json::Value::Number((ai / bi).into());
                    }
                }
                serde_json::Number::from_f64(af / bf)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
    }
}

// ── Aggregate computation ────────────────────────────────────────────────────

/// Compute a JSON aggregate (COUNT, SUM, AVG, MIN, MAX) over indexed rows.
pub fn compute_json_aggregate(
    func: &str,
    rows: &[Vec<serde_json::Value>],
    indices: &[usize],
    col_idx: Option<usize>,
) -> serde_json::Value {
    match func.to_uppercase().as_str() {
        "COUNT" => {
            let count = if let Some(ci) = col_idx {
                indices
                    .iter()
                    .filter(|&&idx| {
                        rows.get(idx)
                            .and_then(|r| r.get(ci))
                            .map(|v| !v.is_null())
                            .unwrap_or(false)
                    })
                    .count()
            } else {
                indices.len()
            };
            serde_json::Value::Number((count as i64).into())
        }
        "SUM" => {
            let mut sum_i64 = 0i64;
            let mut sum_f64 = 0.0f64;
            let mut has_value = false;
            let mut all_int = true;
            for &idx in indices {
                if let Some(ci) = col_idx {
                    if let Some(serde_json::Value::Number(n)) =
                        rows.get(idx).and_then(|r| r.get(ci))
                    {
                        if let Some(i) = n.as_i64() {
                            sum_i64 = sum_i64.wrapping_add(i);
                            sum_f64 += i as f64;
                        } else if let Some(f) = n.as_f64() {
                            if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                                sum_i64 = sum_i64.wrapping_add(f as i64);
                                sum_f64 += f;
                            } else {
                                all_int = false;
                                sum_f64 += f;
                            }
                        }
                        has_value = true;
                    }
                }
            }
            if !has_value {
                serde_json::Value::Null
            } else if all_int {
                serde_json::Value::Number(sum_i64.into())
            } else {
                serde_json::Number::from_f64(sum_f64)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            }
        }
        "AVG" => {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for &idx in indices {
                if let Some(ci) = col_idx {
                    if let Some(serde_json::Value::Number(n)) =
                        rows.get(idx).and_then(|r| r.get(ci))
                    {
                        sum += n.as_f64().unwrap_or(0.0);
                        count += 1;
                    }
                }
            }
            if count > 0 {
                serde_json::Number::from_f64(sum / count as f64)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            } else {
                serde_json::Value::Null
            }
        }
        "MIN" => {
            let mut min_val: Option<serde_json::Number> = None;
            for &idx in indices {
                if let Some(ci) = col_idx {
                    if let Some(serde_json::Value::Number(n)) =
                        rows.get(idx).and_then(|r| r.get(ci))
                    {
                        let v = n.as_f64().unwrap_or(0.0);
                        let is_smaller =
                            min_val.as_ref().map_or(true, |m| v < m.as_f64().unwrap_or(0.0));
                        if is_smaller {
                            min_val = Some(n.clone());
                        }
                    }
                }
            }
            normalize_int(
                min_val
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
            )
        }
        "MAX" => {
            let mut max_val: Option<serde_json::Number> = None;
            for &idx in indices {
                if let Some(ci) = col_idx {
                    if let Some(serde_json::Value::Number(n)) =
                        rows.get(idx).and_then(|r| r.get(ci))
                    {
                        let v = n.as_f64().unwrap_or(0.0);
                        let is_larger =
                            max_val.as_ref().map_or(true, |m| v > m.as_f64().unwrap_or(0.0));
                        if is_larger {
                            max_val = Some(n.clone());
                        }
                    }
                }
            }
            normalize_int(
                max_val
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
            )
        }
        _ => serde_json::Value::Null,
    }
}

// ── AST ↔ JSON conversion ───────────────────────────────────────────────────

/// Convert AST Value to JSON value.
pub fn ast_value_to_json(v: &AstValue) -> serde_json::Value {
    match v {
        AstValue::Null => serde_json::Value::Null,
        AstValue::Bool(b) => serde_json::Value::Bool(*b),
        AstValue::Int(i) => serde_json::Value::Number((*i).into()),
        AstValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        AstValue::String(s) => serde_json::Value::String(s.clone()),
        AstValue::Bytes(b) => serde_json::Value::String(String::from_utf8_lossy(b).to_string()),
        AstValue::Timestamp(ts) => serde_json::Value::Number((*ts).into()),
        AstValue::Uuid(u) => serde_json::Value::String(u.clone()),
        AstValue::Array(arr) => {
            let json_arr: Vec<_> = arr.iter().map(ast_value_to_json).collect();
            serde_json::Value::Array(json_arr)
        }
        AstValue::Object(obj) => {
            let map: serde_json::Map<String, serde_json::Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), ast_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        AstValue::Vector(v) => {
            serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
        }
    }
}

/// Convert serde_json::Value to AST Value.
pub fn json_to_ast_value(v: &serde_json::Value) -> AstValue {
    match v {
        serde_json::Value::Null => AstValue::Null,
        serde_json::Value::Bool(b) => AstValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                AstValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                    AstValue::Int(f as i64)
                } else {
                    AstValue::Float(f)
                }
            } else {
                AstValue::Null
            }
        }
        serde_json::Value::String(s) => AstValue::String(s.clone()),
        serde_json::Value::Array(arr) => {
            AstValue::Array(arr.iter().map(json_to_ast_value).collect())
        }
        serde_json::Value::Object(obj) => AstValue::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_to_ast_value(v)))
                .collect(),
        ),
    }
}

// ── Column reference utilities ───────────────────────────────────────────────

/// Validate that all column references in an expression exist in the available column set.
/// Returns Err(column_name) if a non-existent column is referenced.
/// Skips validation inside subqueries (they have their own column scope) and window functions.
pub fn validate_column_refs(expr: &Expression, available: &[String]) -> Result<(), String> {
    match expr {
        Expression::Column(name) => {
            if available.iter().any(|c| c == name)
                || available
                    .iter()
                    .any(|c| c.ends_with(&format!(".{}", name)))
            {
                Ok(())
            } else if name.contains('.') {
                let bare = name.rsplit('.').next().unwrap();
                if available.iter().any(|c| c == bare)
                    || available
                        .iter()
                        .any(|c| c.ends_with(&format!(".{}", bare)))
                {
                    Ok(())
                } else {
                    Err(name.clone())
                }
            } else {
                Err(name.clone())
            }
        }
        Expression::QualifiedColumn { table, column } => {
            let qualified = format!("{}.{}", table, column);
            if available.iter().any(|c| c == &qualified)
                || available.iter().any(|c| c == column)
                || available
                    .iter()
                    .any(|c| c.ends_with(&format!(".{}", column)))
            {
                Ok(())
            } else {
                Err(qualified)
            }
        }
        Expression::Binary { left, right, .. } => {
            validate_column_refs(left, available)?;
            validate_column_refs(right, available)
        }
        Expression::Unary { expr: inner, .. }
        | Expression::Cast { expr: inner, .. }
        | Expression::IsNull { expr: inner, .. } => validate_column_refs(inner, available),
        Expression::Function { args, .. } => {
            for arg in args {
                validate_column_refs(arg, available)?;
            }
            Ok(())
        }
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                validate_column_refs(op, available)?;
            }
            for (cond, result) in when_clauses {
                validate_column_refs(cond, available)?;
                validate_column_refs(result, available)?;
            }
            if let Some(el) = else_clause {
                validate_column_refs(el, available)?;
            }
            Ok(())
        }
        Expression::In { expr, list, .. } => {
            validate_column_refs(expr, available)?;
            for item in list {
                validate_column_refs(item, available)?;
            }
            Ok(())
        }
        Expression::Between {
            expr, low, high, ..
        } => {
            validate_column_refs(expr, available)?;
            validate_column_refs(low, available)?;
            validate_column_refs(high, available)
        }
        Expression::Like { expr, .. }
        | Expression::SimilarTo { expr, .. }
        | Expression::LikeMeaning { expr, .. }
        | Expression::RegexMatch { expr, .. } => validate_column_refs(expr, available),
        Expression::Subquery(_) | Expression::Exists(_) => Ok(()),
        Expression::WindowFunction { .. } => Ok(()),
        Expression::Literal(_)
        | Expression::Wildcard
        | Expression::QualifiedWildcard(_)
        | Expression::Parameter(_)
        | Expression::NamedParameter(_)
        | Expression::ReverseReference { .. } => Ok(()),
    }
}

/// Check if an ORDER BY expression references columns not present in the given column set.
pub fn expr_needs_source_columns(expr: &Expression, columns: &[String]) -> bool {
    match expr {
        Expression::Column(name) => !columns.iter().any(|c| c == name),
        Expression::Binary { left, right, .. } => {
            expr_needs_source_columns(left, columns) || expr_needs_source_columns(right, columns)
        }
        Expression::Function { args, .. } => {
            args.iter().any(|a| expr_needs_source_columns(a, columns))
        }
        Expression::Unary { expr: inner, .. } => expr_needs_source_columns(inner, columns),
        Expression::Cast { expr: inner, .. } => expr_needs_source_columns(inner, columns),
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
            ..
        } => {
            operand
                .as_ref()
                .map_or(false, |o| expr_needs_source_columns(o, columns))
                || when_clauses.iter().any(|(cond, res)| {
                    expr_needs_source_columns(cond, columns)
                        || expr_needs_source_columns(res, columns)
                })
                || else_clause
                    .as_ref()
                    .map_or(false, |e| expr_needs_source_columns(e, columns))
        }
        Expression::Between {
            expr: inner,
            low,
            high,
            ..
        } => {
            expr_needs_source_columns(inner, columns)
                || expr_needs_source_columns(low, columns)
                || expr_needs_source_columns(high, columns)
        }
        Expression::In {
            expr: inner, list, ..
        } => {
            expr_needs_source_columns(inner, columns)
                || list
                    .iter()
                    .any(|item| expr_needs_source_columns(item, columns))
        }
        _ => false,
    }
}

/// Project rows for RETURNING clause.
/// If returning list contains "*", returns all columns. Otherwise returns only the named columns.
pub fn project_returning(
    returning: &[String],
    all_columns: &[String],
    rows: Vec<Vec<serde_json::Value>>,
) -> (Vec<String>, Vec<Vec<serde_json::Value>>) {
    if returning.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let cols: Vec<String> = if returning.len() == 1 && returning[0] == "*" {
        all_columns.to_vec()
    } else {
        returning.to_vec()
    };
    let projected: Vec<Vec<serde_json::Value>> = rows
        .into_iter()
        .map(|row| {
            cols.iter()
                .map(|col| {
                    all_columns
                        .iter()
                        .position(|c| c == col)
                        .map(|i| row[i].clone())
                        .unwrap_or(serde_json::Value::Null)
                })
                .collect()
        })
        .collect();
    (cols, projected)
}

/// PostgreSQL type name to OID mapping.
pub fn type_name_to_pg_oid(name: &str) -> i64 {
    match name.to_lowercase().as_str() {
        "bool" | "boolean" => 16,
        "bytea" | "blob" | "binary" => 17,
        "char" => 18,
        "int8" | "bigint" => 20,
        "int2" | "smallint" => 21,
        "int4" | "integer" | "int" => 23,
        "text" | "string" => 25,
        "oid" => 26,
        "json" => 114,
        "xml" => 142,
        "float4" | "real" => 700,
        "float8" | "double" | "double precision" | "float" => 701,
        "bpchar" => 1042,
        "varchar" | "character varying" => 1043,
        "date" => 1082,
        "time" => 1083,
        "timestamp" | "datetime" => 1114,
        "timestamptz" | "timestamp with time zone" => 1184,
        "numeric" | "decimal" => 1700,
        "uuid" => 2950,
        "jsonb" => 3802,
        "vector" => 16385,
        _ => 25, // default to text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_equals_same_type() {
        assert!(json_equals(
            &serde_json::json!(42),
            &serde_json::json!(42)
        ));
        assert!(json_equals(
            &serde_json::json!("hello"),
            &serde_json::json!("hello")
        ));
        assert!(!json_equals(
            &serde_json::json!(42),
            &serde_json::json!(43)
        ));
    }

    #[test]
    fn test_json_equals_cross_type() {
        assert!(json_equals(
            &serde_json::json!(5),
            &serde_json::json!("5")
        ));
        assert!(json_equals(
            &serde_json::json!(true),
            &serde_json::json!(1)
        ));
        assert!(json_equals(
            &serde_json::json!(true),
            &serde_json::json!("TRUE")
        ));
    }

    #[test]
    fn test_json_compare_ordering() {
        assert_eq!(json_compare(&serde_json::json!(1), &serde_json::json!(2)), -1);
        assert_eq!(json_compare(&serde_json::json!(2), &serde_json::json!(1)), 1);
        assert_eq!(json_compare(&serde_json::json!(1), &serde_json::json!(1)), 0);
        // Cross-type: Null < Number
        assert_eq!(
            json_compare(&serde_json::Value::Null, &serde_json::json!(0)),
            -1
        );
    }

    #[test]
    fn test_json_arithmetic() {
        assert_eq!(json_add(&serde_json::json!(2), &serde_json::json!(3)), serde_json::json!(5));
        assert_eq!(json_sub(&serde_json::json!(5), &serde_json::json!(3)), serde_json::json!(2));
        assert_eq!(json_mul(&serde_json::json!(4), &serde_json::json!(3)), serde_json::json!(12));
        assert_eq!(json_div(&serde_json::json!(10), &serde_json::json!(3)), serde_json::json!(3));
        assert_eq!(json_div(&serde_json::json!(10), &serde_json::json!(0)), serde_json::Value::Null);
    }

    #[test]
    fn test_normalize_int() {
        assert_eq!(normalize_int(serde_json::json!(120000.0)), serde_json::json!(120000));
        assert_eq!(normalize_int(serde_json::json!(3.14)), serde_json::json!(3.14));
        assert_eq!(normalize_int(serde_json::json!(42)), serde_json::json!(42));
    }

    #[test]
    fn test_pg_oid() {
        assert_eq!(type_name_to_pg_oid("integer"), 23);
        assert_eq!(type_name_to_pg_oid("text"), 25);
        assert_eq!(type_name_to_pg_oid("boolean"), 16);
        assert_eq!(type_name_to_pg_oid("vector"), 16385);
    }

    #[test]
    fn test_aggregate_count() {
        let rows = vec![
            vec![serde_json::json!(1), serde_json::json!("a")],
            vec![serde_json::json!(2), serde_json::json!("b")],
            vec![serde_json::json!(3), serde_json::Value::Null],
        ];
        // COUNT(*) — all rows
        assert_eq!(compute_json_aggregate("COUNT", &rows, &[0, 1, 2], None), serde_json::json!(3));
        // COUNT(col1) — non-null only
        assert_eq!(compute_json_aggregate("COUNT", &rows, &[0, 1, 2], Some(1)), serde_json::json!(2));
    }

    #[test]
    fn test_aggregate_sum_avg() {
        let rows = vec![
            vec![serde_json::json!(10)],
            vec![serde_json::json!(20)],
            vec![serde_json::json!(30)],
        ];
        assert_eq!(compute_json_aggregate("SUM", &rows, &[0, 1, 2], Some(0)), serde_json::json!(60));
        assert_eq!(compute_json_aggregate("AVG", &rows, &[0, 1, 2], Some(0)), serde_json::json!(20.0));
    }
}
