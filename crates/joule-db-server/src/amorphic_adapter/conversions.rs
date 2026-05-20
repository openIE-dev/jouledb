//! Value conversion helpers, error helpers, and free functions used across the module.

use super::*;

/// SQL LIKE pattern matching: % = any sequence, _ = any single char
pub(super) fn like_match(text: &str, pattern: &str) -> bool {
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    let (tl, pl) = (t.len(), p.len());
    // dp[i][j] = does t[..i] match p[..j]
    let mut dp = vec![vec![false; pl + 1]; tl + 1];
    dp[0][0] = true;
    for j in 1..=pl {
        if p[j - 1] == '%' { dp[0][j] = dp[0][j - 1]; }
    }
    for i in 1..=tl {
        for j in 1..=pl {
            if p[j - 1] == '%' {
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if p[j - 1] == '_' || p[j - 1] == t[i - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    dp[tl][pl]
}

/// Convert amorphic Value to AST Value
pub(super) fn amorphic_to_ast(v: &AmorphicValue) -> AstValue {
    match v {
        AmorphicValue::Null => AstValue::Null,
        AmorphicValue::Bool(b) => AstValue::Bool(*b),
        AmorphicValue::Int(i) => AstValue::Int(*i),
        AmorphicValue::Float(f) => AstValue::Float(*f),
        AmorphicValue::String(s) => AstValue::String(s.clone()),
        AmorphicValue::Array(arr) => AstValue::Array(arr.iter().map(amorphic_to_ast).collect()),
        AmorphicValue::Object(obj) => AstValue::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), amorphic_to_ast(v)))
                .collect(),
        ),
    }
}

/// Convert AST Value to AmorphicValue (public for use by mvcc_adapter)
pub fn ast_to_amorphic_value(v: &AstValue) -> AmorphicValue {
    ast_to_amorphic(v)
}

pub(super) fn ast_to_amorphic(v: &AstValue) -> AmorphicValue {
    match v {
        AstValue::Null => AmorphicValue::Null,
        AstValue::Bool(b) => AmorphicValue::Bool(*b),
        AstValue::Int(i) => AmorphicValue::Int(*i),
        AstValue::Float(f) => AmorphicValue::Float(*f),
        AstValue::String(s) => AmorphicValue::String(s.clone()),
        AstValue::Bytes(b) => AmorphicValue::String(hex::encode(b)),
        AstValue::Array(arr) => AmorphicValue::Array(arr.iter().map(ast_to_amorphic).collect()),
        AstValue::Object(obj) => AmorphicValue::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), ast_to_amorphic(v)))
                .collect(),
        ),
        AstValue::Timestamp(t) => AmorphicValue::Int(*t),
        AstValue::Uuid(u) => AmorphicValue::String(u.clone()),
        AstValue::Vector(v) => {
            AmorphicValue::Array(v.iter().map(|f| AmorphicValue::Float(*f as f64)).collect())
        }
    }
}

/// Convert AST Value to serde_json::Value
pub fn ast_to_json(v: &AstValue) -> serde_json::Value {
    match v {
        AstValue::Null => serde_json::Value::Null,
        AstValue::Bool(b) => serde_json::Value::Bool(*b),
        AstValue::Int(i) => serde_json::json!(i),
        AstValue::Float(f) => serde_json::json!(f),
        AstValue::String(s) => serde_json::Value::String(s.clone()),
        AstValue::Bytes(b) => serde_json::Value::String(hex::encode(b)),
        AstValue::Array(arr) => serde_json::Value::Array(arr.iter().map(ast_to_json).collect()),
        AstValue::Object(obj) => {
            let map: serde_json::Map<String, serde_json::Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), ast_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
        AstValue::Timestamp(t) => serde_json::json!(t),
        AstValue::Uuid(u) => serde_json::Value::String(u.clone()),
        AstValue::Vector(v) => {
            serde_json::Value::Array(v.iter().map(|f| serde_json::json!(*f)).collect())
        }
    }
}

/// Convert a default Expression to a JSON-serializable value
pub(super) fn expr_to_default_value(expr: &Expression) -> serde_json::Value {
    match expr {
        Expression::Literal(val) => ast_to_json(val),
        Expression::Unary {
            op: UnaryOperator::Neg,
            expr,
        } => {
            if let Expression::Literal(AstValue::Int(i)) = expr.as_ref() {
                serde_json::json!(-i)
            } else if let Expression::Literal(AstValue::Float(f)) = expr.as_ref() {
                serde_json::json!(-f)
            } else {
                serde_json::Value::Null
            }
        }
        _ => serde_json::Value::Null,
    }
}

/// Convert an Expression to a SQL string representation.
/// Used for serializing CHECK constraint expressions so they can be re-parsed later.
pub fn expression_to_sql(expr: &Expression) -> String {
    match expr {
        Expression::Literal(val) => match val {
            AstValue::Null => "NULL".to_string(),
            AstValue::Bool(b) => {
                if *b {
                    "TRUE".to_string()
                } else {
                    "FALSE".to_string()
                }
            }
            AstValue::Int(i) => i.to_string(),
            AstValue::Float(f) => format!("{}", f),
            AstValue::String(s) => format!("'{}'", s.replace('\'', "''")),
            _ => format!("{:?}", val),
        },
        Expression::Column(name) => {
            // Quote column names that might be keywords
            name.clone()
        }
        Expression::QualifiedColumn { table, column } => {
            format!("{}.{}", table, column)
        }
        Expression::Binary { left, op, right } => {
            let op_str = match op {
                Operator::Eq => "=",
                Operator::Ne => "!=",
                Operator::Lt => "<",
                Operator::Le => "<=",
                Operator::Gt => ">",
                Operator::Ge => ">=",
                Operator::And => "AND",
                Operator::Or => "OR",
                Operator::Add => "+",
                Operator::Sub => "-",
                Operator::Mul => "*",
                Operator::Div => "/",
                Operator::Mod => "%",
                Operator::Concat => "||",
                Operator::BitAnd => "&",
                Operator::BitOr => "|",
                Operator::BitXor => "^",
                Operator::JsonArrow => "->",
                Operator::JsonDoubleArrow => "->>",
                Operator::JsonHashArrow => "#>",
                Operator::JsonHashDoubleArrow => "#>>",
                Operator::JsonContains => "@>",
                Operator::JsonContainedBy => "<@",
                Operator::JsonExists => "?",
                Operator::VectorL2Distance => "<->",
                Operator::VectorIPDistance => "<#>",
                Operator::VectorCosineDistance => "<=>",
            };
            format!(
                "({} {} {})",
                expression_to_sql(left),
                op_str,
                expression_to_sql(right)
            )
        }
        Expression::Unary { op, expr } => {
            let op_str = match op {
                UnaryOperator::Not => "NOT ",
                UnaryOperator::Neg => "-",
                UnaryOperator::BitNot => "~",
            };
            format!("({}{})", op_str, expression_to_sql(expr))
        }
        Expression::Function { name, args } => {
            let arg_strs: Vec<String> = args.iter().map(expression_to_sql).collect();
            format!("{}({})", name, arg_strs.join(", "))
        }
        Expression::In {
            expr,
            list,
            negated,
        } => {
            let list_strs: Vec<String> = list.iter().map(expression_to_sql).collect();
            if *negated {
                format!(
                    "({} NOT IN ({}))",
                    expression_to_sql(expr),
                    list_strs.join(", ")
                )
            } else {
                format!(
                    "({} IN ({}))",
                    expression_to_sql(expr),
                    list_strs.join(", ")
                )
            }
        }
        Expression::Between {
            expr,
            low,
            high,
            negated,
        } => {
            if *negated {
                format!(
                    "({} NOT BETWEEN {} AND {})",
                    expression_to_sql(expr),
                    expression_to_sql(low),
                    expression_to_sql(high)
                )
            } else {
                format!(
                    "({} BETWEEN {} AND {})",
                    expression_to_sql(expr),
                    expression_to_sql(low),
                    expression_to_sql(high)
                )
            }
        }
        Expression::IsNull { expr, negated } => {
            if *negated {
                format!("({} IS NOT NULL)", expression_to_sql(expr))
            } else {
                format!("({} IS NULL)", expression_to_sql(expr))
            }
        }
        Expression::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
        } => {
            let kw = if *case_insensitive { "ILIKE" } else { "LIKE" };
            if *negated {
                format!(
                    "({} NOT {} '{}')",
                    expression_to_sql(expr),
                    kw,
                    pattern.replace('\'', "''")
                )
            } else {
                format!(
                    "({} {} '{}')",
                    expression_to_sql(expr),
                    kw,
                    pattern.replace('\'', "''")
                )
            }
        }
        Expression::Cast { expr, target_type } => {
            format!("CAST({} AS {})", expression_to_sql(expr), target_type)
        }
        Expression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            let mut s = String::from("CASE");
            if let Some(op) = operand {
                s.push_str(&format!(" {}", expression_to_sql(op)));
            }
            for (when_expr, then_expr) in when_clauses {
                s.push_str(&format!(
                    " WHEN {} THEN {}",
                    expression_to_sql(when_expr),
                    expression_to_sql(then_expr)
                ));
            }
            if let Some(el) = else_clause {
                s.push_str(&format!(" ELSE {}", expression_to_sql(el)));
            }
            s.push_str(" END");
            s
        }
        Expression::Parameter(idx) => format!("${}", idx),
        Expression::NamedParameter(name) => format!(":{}", name),
        Expression::Wildcard => "*".to_string(),
        Expression::ReverseReference { reference_name } => {
            format!("~>{}", reference_name)
        }
        _ => format!("{:?}", expr), // Fallback for complex expressions
    }
}

/// Convert AmorphicValue to serde_json::Value (for re-ingestion)
pub(super) fn amorphic_to_json(v: &AmorphicValue) -> serde_json::Value {
    match v {
        AmorphicValue::Null => serde_json::Value::Null,
        AmorphicValue::Bool(b) => serde_json::Value::Bool(*b),
        AmorphicValue::Int(i) => serde_json::json!(i),
        AmorphicValue::Float(f) => serde_json::json!(f),
        AmorphicValue::String(s) => serde_json::Value::String(s.clone()),
        AmorphicValue::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(amorphic_to_json).collect())
        }
        AmorphicValue::Object(obj) => {
            let map: serde_json::Map<String, serde_json::Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), amorphic_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

/// Compare two AST values with an operator
pub(super) fn compare_values(left: &AstValue, right: &AstValue, op: &Operator) -> bool {
    // SQL three-valued logic: any comparison involving NULL returns UNKNOWN (treated as FALSE)
    if matches!(left, AstValue::Null) || matches!(right, AstValue::Null) {
        return false;
    }
    match op {
        // Use cmp_ordering for Eq/Ne to handle cross-type comparisons (Float vs Int)
        Operator::Eq => left == right || cmp_ordering(left, right).map_or(false, |o| o.is_eq()),
        Operator::Ne => left != right && cmp_ordering(left, right).map_or(true, |o| !o.is_eq()),
        Operator::Lt => cmp_ordering(left, right).map_or(false, |o| o.is_lt()),
        Operator::Le => cmp_ordering(left, right).map_or(false, |o| !o.is_gt()),
        Operator::Gt => cmp_ordering(left, right).map_or(false, |o| o.is_gt()),
        Operator::Ge => cmp_ordering(left, right).map_or(false, |o| !o.is_lt()),
        _ => false,
    }
}

/// Attempt to compare two AST values as ordered
fn cmp_ordering(left: &AstValue, right: &AstValue) -> Option<std::cmp::Ordering> {
    match (left, right) {
        (AstValue::Int(a), AstValue::Int(b)) => Some(a.cmp(b)),
        (AstValue::Float(a), AstValue::Float(b)) => a.partial_cmp(b),
        (AstValue::Int(a), AstValue::Float(b)) => (*a as f64).partial_cmp(b),
        (AstValue::Float(a), AstValue::Int(b)) => a.partial_cmp(&(*b as f64)),
        (AstValue::String(a), AstValue::String(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

/// Extract a simple `field = value` from a predicate expression
pub(super) fn extract_eq_predicate(expr: &Expression) -> Option<(String, AstValue)> {
    match expr {
        Expression::Binary {
            left,
            op: Operator::Eq,
            right,
        } => {
            if let Expression::Column(col) = left.as_ref() {
                if let Expression::Literal(val) = right.as_ref() {
                    return Some((col.clone(), val.clone()));
                }
            }
            if let Expression::Column(col) = right.as_ref() {
                if let Expression::Literal(val) = left.as_ref() {
                    return Some((col.clone(), val.clone()));
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract a range predicate `field >= min AND field < max` (or >, <=, etc.)
/// Returns (field_name, min_f64, max_f64) if both bounds reference the same column.
pub(super) fn extract_range_predicate(expr: &Expression) -> Option<(String, f64, f64)> {
    if let Expression::Binary {
        left,
        op: Operator::And,
        right,
    } = expr
    {
        let (col1, bound1, is_lower1) = extract_bound(left)?;
        let (col2, bound2, is_lower2) = extract_bound(right)?;

        if col1 == col2 && is_lower1 != is_lower2 {
            let (min, max) = if is_lower1 {
                (bound1, bound2)
            } else {
                (bound2, bound1)
            };
            return Some((col1, min, max));
        }
    }

    // Single comparison: field >= val -> range [val, MAX] or field < val -> range [MIN, val]
    if let Some((col, bound, is_lower)) = extract_bound(expr) {
        if is_lower {
            return Some((col, bound, f64::MAX));
        } else {
            return Some((col, f64::MIN, bound));
        }
    }

    None
}

/// Extract a single bound from a comparison expression.
/// Returns (column_name, bound_value, is_lower_bound).
fn extract_bound(expr: &Expression) -> Option<(String, f64, bool)> {
    match expr {
        Expression::Binary { left, op, right } => {
            // column >= val or column > val
            if let Expression::Column(col) = left.as_ref() {
                let val = literal_to_f64(right)?;
                match op {
                    Operator::Ge | Operator::Gt => return Some((col.clone(), val, true)),
                    Operator::Le | Operator::Lt => return Some((col.clone(), val, false)),
                    _ => {}
                }
            }
            // val <= column or val < column
            if let Expression::Column(col) = right.as_ref() {
                let val = literal_to_f64(left)?;
                match op {
                    Operator::Le | Operator::Lt => return Some((col.clone(), val, true)),
                    Operator::Ge | Operator::Gt => return Some((col.clone(), val, false)),
                    _ => {}
                }
            }
            None
        }
        _ => None,
    }
}

/// Convert an Expression::Literal to f64 if numeric
fn literal_to_f64(expr: &Expression) -> Option<f64> {
    match expr {
        Expression::Literal(AstValue::Int(i)) => Some(*i as f64),
        Expression::Literal(AstValue::Float(f)) => Some(*f),
        _ => None,
    }
}

/// Convert amorphic Value to serde_json::Value (for REST API responses)
pub(super) fn amorphic_value_to_json(v: &AmorphicValue) -> serde_json::Value {
    match v {
        AmorphicValue::Null => serde_json::Value::Null,
        AmorphicValue::Bool(b) => serde_json::Value::Bool(*b),
        AmorphicValue::Int(i) => serde_json::json!(i),
        AmorphicValue::Float(f) => serde_json::json!(f),
        AmorphicValue::String(s) => serde_json::Value::String(s.clone()),
        AmorphicValue::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(amorphic_value_to_json).collect())
        }
        AmorphicValue::Object(obj) => {
            let map: serde_json::Map<String, serde_json::Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), amorphic_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

pub(super) fn lock_error<T>(_: T) -> QueryError {
    QueryError::ExecutionError("Lock poisoned".to_string())
}

pub(super) fn amorphic_error(e: joule_db_amorphic::AmorphicError) -> QueryError {
    QueryError::ExecutionError(e.to_string())
}

/// Infer SQL column type from a JSON value.
pub(super) fn infer_column_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::String(_) => "TEXT",
        serde_json::Value::Number(n) => {
            if n.is_i64() {
                "INT"
            } else {
                "FLOAT"
            }
        }
        serde_json::Value::Bool(_) => "BOOLEAN",
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => "TEXT",
        serde_json::Value::Null => "TEXT",
    }
}

/// Apply implicit type coercion to row values based on column definitions.
pub(super) fn coerce_row_types(columns: &[String], values: &mut Vec<AstValue>, column_defs: &[ColumnDefInfo]) {
    for i in 0..columns.len() {
        if let Some(def) = column_defs.iter().find(|d| d.name == columns[i]) {
            let target = def.data_type.to_uppercase();
            match (&target[..], &values[i]) {
                ("INT" | "INTEGER" | "BIGINT" | "SMALLINT", AstValue::String(s)) => {
                    if let Ok(n) = s.parse::<i64>() {
                        values[i] = AstValue::Int(n);
                    }
                }
                ("INT" | "INTEGER" | "BIGINT" | "SMALLINT", AstValue::Float(f)) => {
                    values[i] = AstValue::Int(*f as i64);
                }
                ("FLOAT" | "DOUBLE" | "REAL" | "NUMERIC" | "DECIMAL", AstValue::String(s)) => {
                    if let Ok(f) = s.parse::<f64>() {
                        values[i] = AstValue::Float(f);
                    }
                }
                ("FLOAT" | "DOUBLE" | "REAL" | "NUMERIC" | "DECIMAL", AstValue::Int(n)) => {
                    values[i] = AstValue::Float(*n as f64);
                }
                ("BOOLEAN" | "BOOL", AstValue::String(s)) => match s.to_lowercase().as_str() {
                    "true" | "1" | "yes" | "t" => values[i] = AstValue::Bool(true),
                    "false" | "0" | "no" | "f" => values[i] = AstValue::Bool(false),
                    _ => {}
                },
                ("BOOLEAN" | "BOOL", AstValue::Int(n)) => {
                    values[i] = AstValue::Bool(*n != 0);
                }
                ("TEXT" | "VARCHAR" | "CHAR" | "STRING", AstValue::Int(n)) => {
                    values[i] = AstValue::String(n.to_string());
                }
                ("TEXT" | "VARCHAR" | "CHAR" | "STRING", AstValue::Float(f)) => {
                    values[i] = AstValue::String(f.to_string());
                }
                _ => {}
            }
        }
    }
}
