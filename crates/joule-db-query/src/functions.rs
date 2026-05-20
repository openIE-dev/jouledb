//! Unified scalar function registry for JouleDB SQL.
//!
//! All scalar SQL functions are dispatched from `eval_scalar_function()`.
//! The 4 execution paths (executor, execution, storage_executor, server/query)
//! all delegate here instead of maintaining independent copies.

use crate::QueryError;
use crate::QueryResult;
use crate::ast::Value;
use std::collections::HashMap;

/// Callback for functions that need to scan tables (graph functions).
/// Returns (column_names, rows_as_json) for the given table name.
pub type TableScanFn<'a> =
    &'a dyn Fn(&str) -> QueryResult<(Vec<String>, Vec<Vec<serde_json::Value>>)>;

/// Compare two Values for equality.
pub fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Int(a), Value::Int(b)) => a == b,
        (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
        (Value::Int(a), Value::Float(b)) => (*a as f64 - b).abs() < f64::EPSILON,
        (Value::Float(a), Value::Int(b)) => (a - *b as f64).abs() < f64::EPSILON,
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Bytes(a), Value::Bytes(b)) => a == b,
        _ => false,
    }
}

/// Convert serde_json::Value to ast::Value.
pub fn serde_json_to_value(v: serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s),
        serde_json::Value::Array(arr) => {
            Value::Array(arr.into_iter().map(serde_json_to_value).collect())
        }
        serde_json::Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (k, serde_json_to_value(v)))
                .collect(),
        ),
    }
}

/// Convert ast::Value to serde_json::Value (for graph/JSON function interop).
pub fn ast_value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::json!(b),
        Value::Int(i) => serde_json::json!(i),
        Value::Float(f) => serde_json::json!(f),
        Value::String(s) => serde_json::json!(s),
        Value::Timestamp(t) => serde_json::json!(t),
        Value::Uuid(u) => serde_json::json!(u),
        Value::Bytes(b) => serde_json::json!(format!("{:?}", b)),
        Value::Array(arr) => serde_json::Value::Array(arr.iter().map(ast_value_to_json).collect()),
        Value::Vector(v) => serde_json::json!(v),
        Value::Object(obj) => {
            let map: serde_json::Map<String, serde_json::Value> = obj
                .iter()
                .map(|(k, v)| (k.clone(), ast_value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

/// Helper: extract graph edges from table via scanner.
fn load_graph_edges(
    table: &str,
    src_col: &str,
    dst_col: &str,
    scanner: Option<TableScanFn>,
) -> QueryResult<Vec<(String, String)>> {
    let scanner = scanner.ok_or_else(|| {
        QueryError::ExecutionError("Graph functions require table access".to_string())
    })?;
    let (columns, json_rows) = scanner(table)?;
    Ok(crate::graph::extract_edges(
        &columns, &json_rows, src_col, dst_col,
    ))
}

/// Helper: extract a numeric value as f64 from a Value.
fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        _ => None,
    }
}

// ==================== FTS Tokenization Helpers ====================
// Standard text analysis: unicode word split, lowercase, stopword removal, basic stemming.
// These match the standard analyzer behavior in fts_analyzer.rs for consistent
// query-side tokenization when no index is available.

/// English stopwords (most common, matching fts_analyzer::StandardAnalyzer).
const FTS_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Basic Porter-style stemmer: handles the most impactful English suffixes.
/// Covers ~90% of stemming benefit (plurals, -ing, -ed, -ly, -tion).
fn basic_stem(word: &str) -> String {
    if word.len() <= 3 {
        return word.to_string();
    }
    let mut s = word.to_string();
    // Step 1a: plurals
    if s.ends_with("sses") {
        s.truncate(s.len() - 2);
        return s;
    }
    if s.ends_with("ies") && s.len() > 4 {
        s.truncate(s.len() - 2);
        return s;
    }
    if s.ends_with("ss") { /* keep */
    } else if s.ends_with('s') && s.len() > 4 {
        s.truncate(s.len() - 1);
    }
    // Step 1b: -ed, -ing
    let has_vowel = |stem: &str| {
        stem.chars()
            .any(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u'))
    };
    if s.ends_with("eed") && s.len() > 5 {
        s.truncate(s.len() - 1); // -eed -> -ee
        return s;
    }
    if s.ends_with("ed") && s.len() > 5 && has_vowel(&s[..s.len() - 2]) {
        s.truncate(s.len() - 2);
        if s.ends_with("at") || s.ends_with("bl") || s.ends_with("iz") {
            s.push('e');
        }
    } else if s.ends_with("ing") && s.len() > 6 && has_vowel(&s[..s.len() - 3]) {
        s.truncate(s.len() - 3);
        if s.ends_with("at") || s.ends_with("bl") || s.ends_with("iz") {
            s.push('e');
        }
    }
    // Step 1c: trailing y -> i
    if s.ends_with('y') && s.len() > 3 && has_vowel(&s[..s.len() - 1]) {
        s.pop();
        s.push('i');
    }
    // Step 2/3: common suffix normalization
    let suffix_map: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("ization", "ize"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
    ];
    for &(suffix, replacement) in suffix_map {
        if s.ends_with(suffix) && s.len() > suffix.len() + 2 {
            s.truncate(s.len() - suffix.len());
            s.push_str(replacement);
            break;
        }
    }
    s
}

/// Standard FTS tokenization: split on non-alphanumeric, lowercase, remove stopwords, stem.
fn fts_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch.is_alphanumeric() || ch == '_' {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            let word = &text[s..i];
            let lower = word.to_lowercase();
            if lower.len() >= 2 && !FTS_STOPWORDS.contains(&lower.as_str()) {
                tokens.push(basic_stem(&lower));
            }
            start = None;
        }
    }
    if let Some(s) = start {
        let word = &text[s..];
        let lower = word.to_lowercase();
        if lower.len() >= 2 && !FTS_STOPWORDS.contains(&lower.as_str()) {
            tokens.push(basic_stem(&lower));
        }
    }
    tokens
}

/// Levenshtein edit distance for inline fuzzy matching.
fn levenshtein_inline(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Helper: extract a node ID string from a Value.
fn node_id(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Int(i) => Some(i.to_string()),
        _ => None,
    }
}

// ============================================================================
// Canonical operator evaluation
// ============================================================================

/// Canonical evaluation of binary operators on `ast::Value`.
///
/// All execution paths (executor, execution, amorphic_executor, server/query)
/// should delegate here instead of maintaining independent copies.
pub fn eval_binary_op(
    left: &Value,
    op: &crate::ast::Operator,
    right: &Value,
) -> QueryResult<Value> {
    use crate::ast::Operator;

    // SQL NULL semantics: ordered comparisons with NULL yield NULL
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        match op {
            Operator::Lt | Operator::Le | Operator::Gt | Operator::Ge => {
                return Ok(Value::Null);
            }
            _ => {}
        }
    }

    match op {
        // Comparison
        Operator::Eq => Ok(Value::Bool(values_equal(left, right))),
        Operator::Ne => Ok(Value::Bool(!values_equal(left, right))),
        Operator::Lt => Ok(Value::Bool(
            compare_values(left, right) == std::cmp::Ordering::Less,
        )),
        Operator::Le => Ok(Value::Bool(
            compare_values(left, right) != std::cmp::Ordering::Greater,
        )),
        Operator::Gt => Ok(Value::Bool(
            compare_values(left, right) == std::cmp::Ordering::Greater,
        )),
        Operator::Ge => Ok(Value::Bool(
            compare_values(left, right) != std::cmp::Ordering::Less,
        )),

        // Logical
        Operator::And => {
            let a = left.as_bool().unwrap_or(false);
            let b = right.as_bool().unwrap_or(false);
            Ok(Value::Bool(a && b))
        }
        Operator::Or => {
            let a = left.as_bool().unwrap_or(false);
            let b = right.as_bool().unwrap_or(false);
            Ok(Value::Bool(a || b))
        }

        // Arithmetic (with Timestamp support)
        Operator::Add => {
            match (left, right) {
                (Value::Timestamp(ts), Value::Int(n)) | (Value::Int(n), Value::Timestamp(ts)) => {
                    return ts.checked_add(*n).map(Value::Timestamp).ok_or_else(|| {
                        QueryError::ExecutionError("Timestamp overflow".to_string())
                    });
                }
                _ => {}
            }
            numeric_binary_op(left, right, |a, b| a.wrapping_add(b), |a, b| a + b)
        }
        Operator::Sub => {
            match (left, right) {
                (Value::Timestamp(a), Value::Timestamp(b)) => {
                    return a.checked_sub(*b).map(Value::Int).ok_or_else(|| {
                        QueryError::ExecutionError("Timestamp subtraction overflow".to_string())
                    });
                }
                (Value::Timestamp(ts), Value::Int(n)) => {
                    return ts.checked_sub(*n).map(Value::Timestamp).ok_or_else(|| {
                        QueryError::ExecutionError("Timestamp overflow".to_string())
                    });
                }
                _ => {}
            }
            numeric_binary_op(left, right, |a, b| a.wrapping_sub(b), |a, b| a - b)
        }
        Operator::Mul => numeric_binary_op(left, right, |a, b| a.wrapping_mul(b), |a, b| a * b),
        Operator::Div => {
            if let Some(r) = right.as_float() {
                if r == 0.0 {
                    return Err(QueryError::ExecutionError("Division by zero".to_string()));
                }
            }
            numeric_binary_op(
                left,
                right,
                |a, b| if b != 0 { a.wrapping_div(b) } else { 0 },
                |a, b| a / b,
            )
        }
        Operator::Mod => {
            if let Some(r) = right.as_float() {
                if r == 0.0 {
                    return Err(QueryError::ExecutionError("Division by zero".to_string()));
                }
            }
            numeric_binary_op(
                left,
                right,
                |a, b| if b != 0 { a.wrapping_rem(b) } else { 0 },
                |a, b| if b != 0.0 { a % b } else { f64::NAN },
            )
        }

        // String
        Operator::Concat => {
            let ls = match left {
                Value::String(s) => s.clone(),
                v => format!("{:?}", v),
            };
            let rs = match right {
                Value::String(s) => s.clone(),
                v => format!("{:?}", v),
            };
            Ok(Value::String(format!("{}{}", ls, rs)))
        }

        // Bitwise
        Operator::BitAnd | Operator::BitOr | Operator::BitXor => {
            match (left.as_int(), right.as_int()) {
                (Some(a), Some(b)) => {
                    let result = match op {
                        Operator::BitAnd => a & b,
                        Operator::BitOr => a | b,
                        Operator::BitXor => a ^ b,
                        _ => unreachable!(),
                    };
                    Ok(Value::Int(result))
                }
                _ => Ok(Value::Null),
            }
        }

        // JSON
        Operator::JsonArrow
        | Operator::JsonDoubleArrow
        | Operator::JsonHashArrow
        | Operator::JsonHashDoubleArrow
        | Operator::JsonContains
        | Operator::JsonContainedBy
        | Operator::JsonExists => Ok(crate::ast::eval_json_operator(left, op, right)),

        // Vector distance operators
        Operator::VectorL2Distance
        | Operator::VectorIPDistance
        | Operator::VectorCosineDistance => {
            let lv = extract_f32_vec(left);
            let rv = extract_f32_vec(right);
            match (lv, rv) {
                (Some(a), Some(b)) if a.len() == b.len() => {
                    let dist = match op {
                        Operator::VectorL2Distance => a
                            .iter()
                            .zip(b.iter())
                            .map(|(x, y)| (x - y).powi(2))
                            .sum::<f32>()
                            .sqrt(),
                        Operator::VectorCosineDistance => {
                            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if na == 0.0 || nb == 0.0 {
                                1.0
                            } else {
                                1.0 - dot / (na * nb)
                            }
                        }
                        Operator::VectorIPDistance => {
                            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                            -dot
                        }
                        _ => unreachable!(),
                    };
                    Ok(Value::Float(dist as f64))
                }
                _ => Ok(Value::Null),
            }
        }
    }
}

/// Extract a Vec<f32> from a Value (Vector or Array of floats).
fn extract_f32_vec(v: &Value) -> Option<Vec<f32>> {
    match v {
        Value::Vector(vec) => Some(vec.clone()),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Float(f) => out.push(*f as f32),
                    Value::Int(i) => out.push(*i as f32),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

/// Canonical evaluation of unary operators on `ast::Value`.
pub fn eval_unary_op(op: &crate::ast::UnaryOperator, val: &Value) -> QueryResult<Value> {
    use crate::ast::UnaryOperator;
    match op {
        UnaryOperator::Not => {
            let b = val.as_bool().unwrap_or(false);
            Ok(Value::Bool(!b))
        }
        UnaryOperator::Neg => {
            if let Some(i) = val.as_int() {
                Ok(Value::Int(-i))
            } else if let Some(f) = val.as_float() {
                Ok(Value::Float(-f))
            } else {
                Ok(Value::Null)
            }
        }
        UnaryOperator::BitNot => {
            if let Some(i) = val.as_int() {
                Ok(Value::Int(!i))
            } else {
                Ok(Value::Null)
            }
        }
    }
}

/// Compare two Values, returning an `Ordering`.
fn compare_values(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => a.cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
        (Value::Timestamp(a), Value::Timestamp(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Helper for binary numeric operations on `Value`.
fn numeric_binary_op<F, G>(
    left: &Value,
    right: &Value,
    int_op: F,
    float_op: G,
) -> QueryResult<Value>
where
    F: Fn(i64, i64) -> i64,
    G: Fn(f64, f64) -> f64,
{
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(int_op(*a, *b))),
        (Value::Float(a), Value::Float(b)) => Ok(Value::Float(float_op(*a, *b))),
        (Value::Int(a), Value::Float(b)) => Ok(Value::Float(float_op(*a as f64, *b))),
        (Value::Float(a), Value::Int(b)) => Ok(Value::Float(float_op(*a, *b as f64))),
        _ => Ok(Value::Null),
    }
}

/// Evaluate a scalar SQL function by name.
///
/// All ~100+ scalar functions are dispatched here. Aggregate functions
/// (COUNT, SUM, AVG, etc.) and window functions are NOT handled here —
/// those stay in each execution path's aggregate/window logic.
pub fn eval_scalar_function(
    name: &str,
    args: &[Value],
    table_scanner: Option<TableScanFn>,
) -> QueryResult<Value> {
    let name_upper = name.to_uppercase();
    match name_upper.as_str() {
        // ==================== STRING FUNCTIONS ====================
        "UPPER" => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.to_uppercase()))
            } else {
                Ok(Value::Null)
            }
        }
        "LOWER" => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.to_lowercase()))
            } else {
                Ok(Value::Null)
            }
        }
        "LENGTH" | "LEN" => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::Int(s.chars().count() as i64))
            } else {
                Ok(Value::Null)
            }
        }
        "SUBSTR" | "SUBSTRING" => {
            if args.len() >= 2 {
                if let Value::String(s) = &args[0] {
                    let start_raw = match &args[1] {
                        Value::Int(i) => *i,
                        _ => return Ok(Value::Null),
                    };
                    let len = if args.len() >= 3 {
                        match &args[2] {
                            Value::Int(i) => *i,
                            _ => s.chars().count() as i64,
                        }
                    } else {
                        s.chars().count() as i64
                    };
                    // SQL standard: effective_len = len - (1 - start) when start < 1
                    let (start, effective_len) = if start_raw < 1 {
                        let adjusted_len = len + start_raw - 1; // len - (1 - start_raw)
                        if adjusted_len <= 0 {
                            return Ok(Value::String(String::new()));
                        }
                        (0usize, adjusted_len as usize)
                    } else {
                        if len < 0 {
                            return Ok(Value::String(String::new()));
                        }
                        ((start_raw as usize).saturating_sub(1), len as usize)
                    };
                    let result: String = s.chars().skip(start).take(effective_len).collect();
                    return Ok(Value::String(result));
                }
            }
            Ok(Value::Null)
        }
        "TRIM" => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        "LTRIM" => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim_start().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        "RTRIM" => {
            if let Some(Value::String(s)) = args.first() {
                Ok(Value::String(s.trim_end().to_string()))
            } else {
                Ok(Value::Null)
            }
        }
        "REPLACE" => {
            if args.len() >= 3 {
                if let (Value::String(s), Value::String(from), Value::String(to)) =
                    (&args[0], &args[1], &args[2])
                {
                    // PostgreSQL: empty 'from' string returns original unchanged
                    if from.is_empty() {
                        return Ok(Value::String(s.clone()));
                    }
                    return Ok(Value::String(s.replace(from.as_str(), to.as_str())));
                }
            }
            Ok(Value::Null)
        }
        "SPLIT_PART" => {
            if args.len() >= 3 {
                if let (Value::String(s), Value::String(delim), Value::Int(n)) =
                    (&args[0], &args[1], &args[2])
                {
                    if *n == 0 {
                        return Err(QueryError::ExecutionError(
                            "field position must not be zero".to_string(),
                        ));
                    }
                    let parts: Vec<&str> = s.split(delim.as_str()).collect();
                    let idx = if *n < 0 {
                        // Negative: count from end (PostgreSQL 14+)
                        let from_end = (-*n) as usize;
                        if from_end > parts.len() {
                            return Ok(Value::String(String::new()));
                        }
                        parts.len() - from_end
                    } else {
                        (*n as usize).saturating_sub(1)
                    };
                    return Ok(Value::String(parts.get(idx).unwrap_or(&"").to_string()));
                }
            }
            Ok(Value::Null)
        }
        "TRANSLATE" => {
            if args.len() >= 3 {
                if let (Value::String(s), Value::String(from), Value::String(to)) =
                    (&args[0], &args[1], &args[2])
                {
                    let to_chars: Vec<char> = to.chars().collect();
                    let result: String = s
                        .chars()
                        .map(|c| {
                            if let Some(pos) = from.chars().position(|fc| fc == c) {
                                to_chars.get(pos).copied().unwrap_or('\0')
                            } else {
                                c
                            }
                        })
                        .filter(|&c| c != '\0')
                        .collect();
                    return Ok(Value::String(result));
                }
            }
            Ok(Value::Null)
        }
        "CONCAT" => {
            let result: String = args
                .iter()
                .map(|a| match a {
                    Value::String(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => String::new(),
                    _ => format!("{:?}", a),
                })
                .collect();
            Ok(Value::String(result))
        }
        "CONCAT_WS" => {
            if args.is_empty() {
                return Ok(Value::Null);
            }
            let separator = match &args[0] {
                Value::String(s) => s.clone(),
                Value::Null => return Ok(Value::Null),
                v => format!("{:?}", v),
            };
            let parts: Vec<String> = args[1..]
                .iter()
                .filter(|v| !matches!(v, Value::Null))
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    _ => format!("{:?}", v),
                })
                .collect();
            Ok(Value::String(parts.join(&separator)))
        }
        "INITCAP" => match args.first() {
            Some(Value::String(s)) => {
                let result: String = s
                    .chars()
                    .scan(true, |cap, c| {
                        let out = if *cap {
                            c.to_uppercase().next().unwrap_or(c)
                        } else {
                            c.to_lowercase().next().unwrap_or(c)
                        };
                        *cap = !c.is_alphanumeric();
                        Some(out)
                    })
                    .collect();
                Ok(Value::String(result))
            }
            _ => Ok(Value::Null),
        },
        "ASCII" => match args.first() {
            Some(Value::String(s)) => {
                Ok(Value::Int(s.chars().next().map(|c| c as i64).unwrap_or(0)))
            }
            _ => Ok(Value::Null),
        },
        "CHR" => match args.first() {
            Some(Value::Int(code)) => {
                if *code < 0 || *code > u32::MAX as i64 {
                    return Ok(Value::Null);
                }
                match char::from_u32(*code as u32) {
                    Some(c) => Ok(Value::String(c.to_string())),
                    None => Ok(Value::Null),
                }
            }
            _ => Ok(Value::Null),
        },
        "CHAR_LENGTH" | "CHARACTER_LENGTH" => match args.first() {
            Some(Value::String(s)) => Ok(Value::Int(s.chars().count() as i64)),
            _ => Ok(Value::Null),
        },
        "POSITION" => match (args.get(0), args.get(1)) {
            (Some(Value::String(sub)), Some(Value::String(s))) => {
                // Find character position (not byte offset) for correct multibyte handling
                if sub.is_empty() {
                    return Ok(Value::Int(1));
                }
                let pos = s
                    .char_indices()
                    .enumerate()
                    .find(|(_, (byte_idx, _))| s[*byte_idx..].starts_with(sub.as_str()))
                    .map(|(char_idx, _)| char_idx as i64 + 1)
                    .unwrap_or(0);
                Ok(Value::Int(pos))
            }
            _ => Ok(Value::Null),
        },
        "LPAD" => match (args.get(0), args.get(1)) {
            (Some(Value::String(s)), Some(Value::Int(len))) if *len >= 0 => {
                let len = (*len as usize).min(1_000_000);
                let pad = args
                    .get(2)
                    .and_then(|v| {
                        if let Value::String(p) = v {
                            Some(p.as_str())
                        } else {
                            None
                        }
                    })
                    .unwrap_or(" ");
                let char_count = s.chars().count();
                if char_count >= len {
                    Ok(Value::String(s.chars().take(len).collect()))
                } else {
                    let needed = len - char_count;
                    let padding: String = pad.chars().cycle().take(needed).collect();
                    Ok(Value::String(format!("{}{}", padding, s)))
                }
            }
            _ => Ok(Value::Null),
        },
        "RPAD" => match (args.get(0), args.get(1)) {
            (Some(Value::String(s)), Some(Value::Int(len))) if *len >= 0 => {
                let len = (*len as usize).min(1_000_000);
                let pad = args
                    .get(2)
                    .and_then(|v| {
                        if let Value::String(p) = v {
                            Some(p.as_str())
                        } else {
                            None
                        }
                    })
                    .unwrap_or(" ");
                let char_count = s.chars().count();
                if char_count >= len {
                    Ok(Value::String(s.chars().take(len).collect()))
                } else {
                    let needed = len - char_count;
                    let padding: String = pad.chars().cycle().take(needed).collect();
                    Ok(Value::String(format!("{}{}", s, padding)))
                }
            }
            _ => Ok(Value::Null),
        },
        "REVERSE" => match args.first() {
            Some(Value::String(s)) => Ok(Value::String(s.chars().rev().collect())),
            _ => Ok(Value::Null),
        },
        "REPEAT" => match (args.get(0), args.get(1)) {
            (Some(Value::String(s)), Some(Value::Int(n))) if *n >= 0 => {
                let count = (*n as usize).min(1_000_000);
                Ok(Value::String(s.repeat(count)))
            }
            _ => Ok(Value::Null),
        },
        "LEFT" => match (args.get(0), args.get(1)) {
            (Some(Value::String(s)), Some(Value::Int(n))) if *n >= 0 => {
                Ok(Value::String(s.chars().take(*n as usize).collect()))
            }
            _ => Ok(Value::Null),
        },
        "RIGHT" => match (args.get(0), args.get(1)) {
            (Some(Value::String(s)), Some(Value::Int(n))) if *n >= 0 => {
                let n = *n as usize;
                let chars: Vec<char> = s.chars().collect();
                Ok(Value::String(if n >= chars.len() {
                    s.clone()
                } else {
                    chars[chars.len() - n..].iter().collect()
                }))
            }
            _ => Ok(Value::Null),
        },

        // ==================== MATH FUNCTIONS ====================
        "ABS" => {
            if let Some(Value::Int(i)) = args.first() {
                // i64::MIN has no positive i64 representation — promote to f64
                Ok(match i.checked_abs() {
                    Some(v) => Value::Int(v),
                    None => Value::Float((*i as f64).abs()),
                })
            } else if let Some(Value::Float(f)) = args.first() {
                Ok(Value::Float(f.abs()))
            } else {
                Ok(Value::Null)
            }
        }
        "ROUND" => {
            let n = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => *i as f64,
                _ => return Ok(Value::Null),
            };
            let decimals = match args.get(1) {
                Some(Value::Int(d)) => (*d).max(-308).min(308) as i32,
                _ => 0,
            };
            let factor = 10f64.powi(decimals);
            Ok(Value::Float((n * factor).round() / factor))
        }
        "CEIL" | "CEILING" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.ceil())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        "FLOOR" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.floor())),
            Some(Value::Int(i)) => Ok(Value::Int(*i)),
            _ => Ok(Value::Null),
        },
        "SQRT" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).sqrt())),
            _ => Ok(Value::Null),
        },
        "POWER" | "POW" => {
            if args.len() >= 2 {
                let base = match &args[0] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                let exp = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                Ok(Value::Float(base.powf(exp)))
            } else {
                Ok(Value::Null)
            }
        }
        "LOG" | "LN" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.ln())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).ln())),
            _ => Ok(Value::Null),
        },
        "LOG10" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.log10())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).log10())),
            _ => Ok(Value::Null),
        },
        "LOG2" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.log2())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).log2())),
            _ => Ok(Value::Null),
        },
        "SIGN" => match args.first() {
            Some(Value::Int(i)) => Ok(Value::Int(if *i > 0 {
                1
            } else if *i < 0 {
                -1
            } else {
                0
            })),
            Some(Value::Float(f)) => Ok(Value::Int(if *f > 0.0 {
                1
            } else if *f < 0.0 {
                -1
            } else {
                0
            })),
            _ => Ok(Value::Null),
        },
        "MOD" => match (args.get(0), args.get(1)) {
            (Some(Value::Int(a)), Some(Value::Int(b))) if *b != 0 => {
                Ok(Value::Int(a.wrapping_rem(*b)))
            }
            (Some(Value::Float(a)), Some(Value::Float(b))) if *b != 0.0 => Ok(Value::Float(a % b)),
            (Some(Value::Int(a)), Some(Value::Float(b))) if *b != 0.0 => {
                Ok(Value::Float(*a as f64 % b))
            }
            (Some(Value::Float(a)), Some(Value::Int(b))) if *b != 0 => {
                Ok(Value::Float(a % *b as f64))
            }
            _ => Ok(Value::Null),
        },
        "EXP" => match args.first() {
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).exp())),
            Some(Value::Float(f)) => Ok(Value::Float(f.exp())),
            _ => Ok(Value::Null),
        },
        "TRUNC" => {
            let n = match args.first() {
                Some(Value::Float(f)) => *f,
                Some(Value::Int(i)) => *i as f64,
                _ => return Ok(Value::Null),
            };
            let decimals = match args.get(1) {
                Some(Value::Int(d)) => *d as i32,
                _ => 0,
            };
            let factor = 10f64.powi(decimals);
            Ok(Value::Float((n * factor).trunc() / factor))
        }
        "CBRT" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.cbrt())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).cbrt())),
            _ => Ok(Value::Null),
        },
        "PI" => Ok(Value::Float(std::f64::consts::PI)),
        // Trigonometric
        "SIN" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sin())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).sin())),
            _ => Ok(Value::Null),
        },
        "COS" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.cos())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).cos())),
            _ => Ok(Value::Null),
        },
        "TAN" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.tan())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).tan())),
            _ => Ok(Value::Null),
        },
        "ASIN" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.asin())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).asin())),
            _ => Ok(Value::Null),
        },
        "ACOS" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.acos())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).acos())),
            _ => Ok(Value::Null),
        },
        "ATAN" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.atan())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).atan())),
            _ => Ok(Value::Null),
        },
        "ATAN2" => {
            if args.len() >= 2 {
                let y = match &args[0] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                let x = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                Ok(Value::Float(y.atan2(x)))
            } else {
                Ok(Value::Null)
            }
        }
        "DEGREES" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.to_degrees())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).to_degrees())),
            _ => Ok(Value::Null),
        },
        "RADIANS" => match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.to_radians())),
            Some(Value::Int(i)) => Ok(Value::Float((*i as f64).to_radians())),
            _ => Ok(Value::Null),
        },

        // ==================== COMPARISON / UTILITY ====================
        "COALESCE" => {
            for arg in args {
                if !matches!(arg, Value::Null) {
                    return Ok(arg.clone());
                }
            }
            Ok(Value::Null)
        }
        "NULLIF" => {
            if args.len() >= 2 && values_equal(&args[0], &args[1]) {
                Ok(Value::Null)
            } else {
                Ok(args.first().cloned().unwrap_or(Value::Null))
            }
        }
        "IFNULL" | "NVL" => {
            if args.len() >= 2 {
                if matches!(&args[0], Value::Null) {
                    Ok(args[1].clone())
                } else {
                    Ok(args[0].clone())
                }
            } else {
                Ok(Value::Null)
            }
        }
        "GREATEST" => {
            let non_null: Vec<Value> = args
                .iter()
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
                .collect();
            if non_null.is_empty() {
                return Ok(Value::Null);
            }
            let mut best = non_null[0].clone();
            for v in &non_null[1..] {
                match (&best, v) {
                    (Value::Int(a), Value::Int(b)) => {
                        if b > a {
                            best = Value::Int(*b);
                        }
                    }
                    (Value::Float(a), Value::Float(b)) => {
                        if b > a {
                            best = Value::Float(*b);
                        }
                    }
                    (Value::String(a), Value::String(b)) => {
                        if b > a {
                            best = Value::String(b.clone());
                        }
                    }
                    (Value::Int(a), Value::Float(b)) => {
                        if *b > *a as f64 {
                            best = Value::Float(*b);
                        } else {
                            best = Value::Float(*a as f64);
                        }
                    }
                    (Value::Float(a), Value::Int(b)) => {
                        if (*b as f64) > *a {
                            best = Value::Int(*b);
                        }
                    }
                    _ => {}
                }
            }
            Ok(best)
        }
        "LEAST" => {
            let non_null: Vec<Value> = args
                .iter()
                .filter(|v| !matches!(v, Value::Null))
                .cloned()
                .collect();
            if non_null.is_empty() {
                return Ok(Value::Null);
            }
            let mut best = non_null[0].clone();
            for v in &non_null[1..] {
                match (&best, v) {
                    (Value::Int(a), Value::Int(b)) => {
                        if b < a {
                            best = Value::Int(*b);
                        }
                    }
                    (Value::Float(a), Value::Float(b)) => {
                        if b < a {
                            best = Value::Float(*b);
                        }
                    }
                    (Value::String(a), Value::String(b)) => {
                        if b < a {
                            best = Value::String(b.clone());
                        }
                    }
                    (Value::Int(a), Value::Float(b)) => {
                        if *b < *a as f64 {
                            best = Value::Float(*b);
                        } else {
                            best = Value::Float(*a as f64);
                        }
                    }
                    (Value::Float(a), Value::Int(b)) => {
                        if (*b as f64) < *a {
                            best = Value::Int(*b);
                        }
                    }
                    _ => {}
                }
            }
            Ok(best)
        }
        "TYPEOF" => {
            let type_name = match args.first() {
                Some(Value::Null) => "NULL",
                Some(Value::Bool(_)) => "BOOLEAN",
                Some(Value::Int(_)) => "INTEGER",
                Some(Value::Float(_)) => "REAL",
                Some(Value::String(_)) => "TEXT",
                Some(Value::Bytes(_)) => "BLOB",
                Some(Value::Timestamp(_)) => "TIMESTAMP",
                _ => "UNKNOWN",
            };
            Ok(Value::String(type_name.to_string()))
        }
        "RANDOM" | "RAND" => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            Ok(Value::Float((nanos as f64) / 1_000_000_000.0))
        }

        // ==================== DATE/TIME FUNCTIONS ====================
        "NOW" | "CURRENT_TIMESTAMP" => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            Ok(Value::Timestamp(now))
        }
        "EXTRACT" | "DATE_PART" => {
            if args.len() >= 2 {
                let ts_opt = match &args[1] {
                    Value::Timestamp(ts) => Some(*ts),
                    Value::Int(ts) => Some(*ts),
                    _ => None,
                };
                if let (Value::String(part), Some(ts)) = (&args[0], ts_opt) {
                    use chrono::{Datelike, Timelike};
                    let dt = chrono::DateTime::from_timestamp(ts, 0)
                        .unwrap_or(chrono::DateTime::UNIX_EPOCH);
                    let result = match part.to_uppercase().as_str() {
                        "YEAR" => dt.year() as i64,
                        "MONTH" => dt.month() as i64,
                        "DAY" => dt.day() as i64,
                        "HOUR" => dt.hour() as i64,
                        "MINUTE" => dt.minute() as i64,
                        "SECOND" => dt.second() as i64,
                        "DOW" | "DAYOFWEEK" => dt.weekday().num_days_from_sunday() as i64,
                        "DOY" | "DAYOFYEAR" => dt.ordinal() as i64,
                        "EPOCH" => ts,
                        _ => return Ok(Value::Null),
                    };
                    return Ok(Value::Int(result));
                }
            }
            Ok(Value::Null)
        }
        "DATE_TRUNC" => {
            if args.len() >= 2 {
                let ts_opt = match &args[1] {
                    Value::Timestamp(ts) => Some(*ts),
                    Value::Int(ts) => Some(*ts),
                    _ => None,
                };
                if let (Value::String(field), Some(ts)) = (&args[0], ts_opt) {
                    use chrono::Datelike;
                    use chrono::Timelike;
                    let dt = chrono::DateTime::from_timestamp(ts, 0)
                        .unwrap_or(chrono::DateTime::UNIX_EPOCH);
                    let truncated = match field.to_uppercase().as_str() {
                        "YEAR" => chrono::NaiveDate::from_ymd_opt(dt.year(), 1, 1)
                            .and_then(|d| d.and_hms_opt(0, 0, 0))
                            .map(|ndt| ndt.and_utc().timestamp()),
                        "MONTH" => chrono::NaiveDate::from_ymd_opt(dt.year(), dt.month(), 1)
                            .and_then(|d| d.and_hms_opt(0, 0, 0))
                            .map(|ndt| ndt.and_utc().timestamp()),
                        "DAY" => chrono::NaiveDate::from_ymd_opt(dt.year(), dt.month(), dt.day())
                            .and_then(|d| d.and_hms_opt(0, 0, 0))
                            .map(|ndt| ndt.and_utc().timestamp()),
                        "HOUR" => chrono::NaiveDate::from_ymd_opt(dt.year(), dt.month(), dt.day())
                            .and_then(|d| d.and_hms_opt(dt.hour(), 0, 0))
                            .map(|ndt| ndt.and_utc().timestamp()),
                        "MINUTE" => {
                            chrono::NaiveDate::from_ymd_opt(dt.year(), dt.month(), dt.day())
                                .and_then(|d| d.and_hms_opt(dt.hour(), dt.minute(), 0))
                                .map(|ndt| ndt.and_utc().timestamp())
                        }
                        _ => Some(ts),
                    };
                    return Ok(Value::Timestamp(truncated.unwrap_or(ts)));
                }
            }
            Ok(Value::Null)
        }
        "TO_CHAR" => {
            if args.len() >= 2 {
                let ts_opt = match &args[0] {
                    Value::Timestamp(ts) => Some(*ts),
                    Value::Int(ts) => Some(*ts),
                    _ => None,
                };
                if let (Some(ts), Value::String(fmt)) = (ts_opt, &args[1]) {
                    let dt = chrono::DateTime::from_timestamp(ts, 0)
                        .unwrap_or(chrono::DateTime::UNIX_EPOCH);
                    let chrono_fmt = fmt
                        .replace("YYYY", "%Y")
                        .replace("YY", "%y")
                        .replace("MM", "%m")
                        .replace("DD", "%d")
                        .replace("HH24", "%H")
                        .replace("HH12", "%I")
                        .replace("HH", "%H")
                        .replace("MI", "%M")
                        .replace("SS", "%S")
                        .replace("AM", "%p")
                        .replace("PM", "%p");
                    return Ok(Value::String(dt.format(&chrono_fmt).to_string()));
                }
            }
            Ok(Value::Null)
        }
        "AGE" => {
            let a = match args.first() {
                Some(Value::Timestamp(t)) | Some(Value::Int(t)) => Some(*t),
                _ => None,
            };
            let b = match args.get(1) {
                Some(Value::Timestamp(t)) | Some(Value::Int(t)) => Some(*t),
                _ => None,
            };
            match (a, b) {
                (Some(a), Some(b)) => Ok(Value::Int(a - b)),
                _ => Ok(Value::Null),
            }
        }

        // ==================== FULL-TEXT SEARCH ====================
        "MATCH_AGAINST" | "TS_RANK" => {
            // BM25-based relevance scoring: MATCH_AGAINST(query, col1, col2, ...)
            // Uses standard analyzer tokenization (stopwords + stemming) for consistency
            // with FTS index behavior.
            if let Some(Value::String(query)) = args.first() {
                let query_terms = fts_tokenize(query);
                if query_terms.is_empty() {
                    return Ok(Value::Float(0.0));
                }
                let mut total_score = 0.0_f64;
                for arg in &args[1..] {
                    if let Value::String(text) = arg {
                        let doc_tokens = fts_tokenize(text);
                        let doc_len = doc_tokens.len() as f64;
                        let avg_dl = 100.0_f64; // approximation for inline scoring
                        for term in &query_terms {
                            let tf = doc_tokens.iter().filter(|t| t == &term).count() as f64;
                            if tf > 0.0 {
                                let idf = 1.5_f64;
                                let k1 = 1.2_f64;
                                let b = 0.75_f64;
                                let numerator = tf * (k1 + 1.0);
                                let denominator =
                                    tf + k1 * (1.0 - b + b * doc_len / avg_dl.max(1.0));
                                total_score += idf * numerator / denominator;
                            }
                        }
                    }
                }
                Ok(Value::Float(total_score))
            } else {
                Ok(Value::Float(0.0))
            }
        }
        // FTS_FUZZY_MATCH(query, text, max_distance) — Levenshtein fuzzy matching
        "FTS_FUZZY_MATCH" => {
            if args.len() >= 2 {
                let query = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Bool(false)),
                };
                let text = match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Bool(false)),
                };
                let max_dist = match args.get(2) {
                    Some(Value::Int(n)) => *n as usize,
                    Some(Value::Float(f)) => *f as usize,
                    _ => 2,
                };
                let query_tokens: Vec<String> = query
                    .split_whitespace()
                    .map(|t| t.to_lowercase())
                    .filter(|t| !t.is_empty())
                    .collect();
                let text_tokens: Vec<String> = text
                    .split_whitespace()
                    .map(|t| {
                        t.chars()
                            .filter(|c| c.is_alphanumeric())
                            .collect::<String>()
                            .to_lowercase()
                    })
                    .filter(|t| !t.is_empty())
                    .collect();
                let mut all_match = true;
                for qt in &query_tokens {
                    let found = text_tokens
                        .iter()
                        .any(|tt| levenshtein_inline(qt, tt) <= max_dist);
                    if !found {
                        all_match = false;
                        break;
                    }
                }
                Ok(Value::Bool(all_match && !query_tokens.is_empty()))
            } else {
                Ok(Value::Bool(false))
            }
        }
        // FTS_PHRASE_MATCH(phrase, text) — consecutive term matching
        "FTS_PHRASE_MATCH" => {
            if args.len() >= 2 {
                let phrase = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Bool(false)),
                };
                let text = match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Bool(false)),
                };
                let phrase_tokens: Vec<String> = phrase
                    .split_whitespace()
                    .map(|t| t.to_lowercase())
                    .filter(|t| !t.is_empty())
                    .collect();
                if phrase_tokens.is_empty() {
                    return Ok(Value::Bool(false));
                }
                let text_tokens: Vec<String> = text
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .map(|w| w.to_lowercase())
                    .filter(|w| !w.is_empty())
                    .collect();
                if text_tokens.len() < phrase_tokens.len() {
                    return Ok(Value::Bool(false));
                }
                let found = (0..=(text_tokens.len() - phrase_tokens.len()))
                    .any(|i| text_tokens[i..i + phrase_tokens.len()] == phrase_tokens[..]);
                Ok(Value::Bool(found))
            } else {
                Ok(Value::Bool(false))
            }
        }
        "TS_HEADLINE" => {
            if args.len() >= 2 {
                if let (Some(Value::String(text)), Some(Value::String(query))) =
                    (args.first(), args.get(1))
                {
                    let terms = fts_tokenize(query);
                    let mut result = text.clone();
                    for term in &terms {
                        let mut new_result = String::new();
                        let lower = result.to_lowercase();
                        let mut last = 0;
                        // Match stemmed term against original text words for highlighting
                        for (idx, _) in lower.match_indices(term.as_str()) {
                            new_result.push_str(&result[last..idx]);
                            new_result.push_str("<b>");
                            new_result.push_str(&result[idx..idx + term.len()]);
                            new_result.push_str("</b>");
                            last = idx + term.len();
                        }
                        new_result.push_str(&result[last..]);
                        result = new_result;
                    }
                    return Ok(Value::String(result));
                }
            }
            Ok(Value::Null)
        }
        "TO_TSVECTOR" => {
            if let Some(Value::String(text)) = args.first() {
                let mut tokens = fts_tokenize(text);
                tokens.sort();
                tokens.dedup();
                Ok(Value::String(tokens.join(" ")))
            } else {
                Ok(Value::Null)
            }
        }
        "TO_TSQUERY" => {
            if let Some(Value::String(query)) = args.first() {
                let tokens = fts_tokenize(query);
                Ok(Value::String(tokens.join(" & ")))
            } else {
                Ok(Value::Null)
            }
        }
        // FTS_BOOLEAN_MATCH(query, text) — boolean query matching (+required -excluded "phrase")
        "FTS_BOOLEAN_MATCH" => {
            if args.len() >= 2 {
                let query = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Bool(false)),
                };
                let text = match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => return Ok(Value::Bool(false)),
                };
                // Simple inline boolean matching
                let mut required = Vec::new();
                let mut excluded = Vec::new();
                let mut phrases = Vec::new();
                let chars: Vec<char> = query.chars().collect();
                let mut i = 0;
                while i < chars.len() {
                    if chars[i].is_whitespace() {
                        i += 1;
                        continue;
                    }
                    if chars[i] == '"' {
                        i += 1;
                        let start = i;
                        while i < chars.len() && chars[i] != '"' {
                            i += 1;
                        }
                        let phrase: String = chars[start..i].iter().collect();
                        if !phrase.trim().is_empty() {
                            phrases.push(phrase.trim().to_lowercase());
                        }
                        if i < chars.len() {
                            i += 1;
                        }
                        continue;
                    }
                    let prefix = if chars[i] == '+' {
                        i += 1;
                        Some('+')
                    } else if chars[i] == '-' {
                        i += 1;
                        Some('-')
                    } else {
                        None
                    };
                    let start = i;
                    while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '"' {
                        i += 1;
                    }
                    let term: String = chars[start..i].iter().collect();
                    if term.is_empty() {
                        continue;
                    }
                    match prefix {
                        Some('+') => required.push(term.to_lowercase()),
                        Some('-') => excluded.push(term.to_lowercase()),
                        _ => {} // optional terms don't affect boolean match
                    }
                }
                let text_lower = text.to_lowercase();
                let text_words: Vec<String> = text_lower
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .map(|w| w.to_string())
                    .filter(|w| !w.is_empty())
                    .collect();
                // Check required
                for req in &required {
                    if !text_words.iter().any(|w| w == req) {
                        return Ok(Value::Bool(false));
                    }
                }
                // Check excluded
                for exc in &excluded {
                    if text_words.iter().any(|w| w == exc) {
                        return Ok(Value::Bool(false));
                    }
                }
                // Check phrases
                for phrase in &phrases {
                    let phrase_words: Vec<&str> = phrase.split_whitespace().collect();
                    if phrase_words.is_empty() {
                        continue;
                    }
                    let found = if text_words.len() >= phrase_words.len() {
                        (0..=(text_words.len() - phrase_words.len())).any(|j| {
                            text_words[j..j + phrase_words.len()]
                                .iter()
                                .zip(phrase_words.iter())
                                .all(|(a, b)| a == b)
                        })
                    } else {
                        false
                    };
                    if !found {
                        return Ok(Value::Bool(false));
                    }
                }
                Ok(Value::Bool(true))
            } else {
                Ok(Value::Bool(false))
            }
        }
        // FTS_TERM_FREQ(term, text) — count occurrences of a term in text
        "FTS_TERM_FREQ" => {
            if args.len() >= 2 {
                let term = match &args[0] {
                    Value::String(s) => s.to_lowercase(),
                    _ => return Ok(Value::Int(0)),
                };
                let text = match &args[1] {
                    Value::String(s) => s.to_lowercase(),
                    _ => return Ok(Value::Int(0)),
                };
                let count = text
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .filter(|w| w.to_lowercase() == term)
                    .count();
                Ok(Value::Int(count as i64))
            } else {
                Ok(Value::Int(0))
            }
        }
        // FTS_DOC_COUNT(col_values...) — count non-null, non-empty text arguments
        "FTS_DOC_COUNT" => {
            let count = args
                .iter()
                .filter(|a| matches!(a, Value::String(s) if !s.is_empty()))
                .count();
            Ok(Value::Int(count as i64))
        }
        // FTS_HIGHLIGHT(text, query, open_tag, close_tag) — highlight matches with configurable tags
        "FTS_HIGHLIGHT" => {
            if args.len() >= 2 {
                if let (Some(Value::String(text)), Some(Value::String(query))) =
                    (args.first(), args.get(1))
                {
                    let open_tag = match args.get(2) {
                        Some(Value::String(s)) => s.clone(),
                        _ => "<b>".to_string(),
                    };
                    let close_tag = match args.get(3) {
                        Some(Value::String(s)) => s.clone(),
                        _ => "</b>".to_string(),
                    };
                    let terms: Vec<String> = query
                        .split_whitespace()
                        .map(|t| {
                            t.trim_matches(|c: char| !c.is_alphanumeric())
                                .to_lowercase()
                        })
                        .filter(|t| !t.is_empty())
                        .collect();
                    let mut result = text.clone();
                    for term in &terms {
                        let mut new_result = String::new();
                        let lower = result.to_lowercase();
                        let mut last = 0;
                        for (idx, _) in lower.match_indices(term.as_str()) {
                            new_result.push_str(&result[last..idx]);
                            new_result.push_str(&open_tag);
                            new_result.push_str(&result[idx..idx + term.len()]);
                            new_result.push_str(&close_tag);
                            last = idx + term.len();
                        }
                        new_result.push_str(&result[last..]);
                        result = new_result;
                    }
                    return Ok(Value::String(result));
                }
            }
            Ok(Value::Null)
        }

        // ==================== JSON FUNCTIONS ====================
        "JSON_EXTRACT" | "JSON_EXTRACT_PATH" => {
            if args.is_empty() {
                return Ok(Value::Null);
            }
            let mut current = match &args[0] {
                Value::String(s) => match serde_json::from_str::<serde_json::Value>(s) {
                    Ok(v) => serde_json_to_value(v),
                    Err(_) => return Ok(Value::Null),
                },
                other => other.clone(),
            };
            for key in &args[1..] {
                // Support JSONPath-style keys: "$.foo.bar" → ["foo", "bar"]
                if let Value::String(path) = key {
                    if path.starts_with("$.") || path.starts_with("$[") {
                        let path_str = &path[1..]; // strip leading $
                        for segment in path_str.split('.').filter(|s| !s.is_empty()) {
                            current = match &current {
                                Value::Object(map) => {
                                    map.get(segment).cloned().unwrap_or(Value::Null)
                                }
                                Value::Array(arr) => {
                                    if let Ok(i) = segment.parse::<usize>() {
                                        arr.get(i).cloned().unwrap_or(Value::Null)
                                    } else {
                                        Value::Null
                                    }
                                }
                                _ => Value::Null,
                            };
                        }
                        continue;
                    }
                }
                current = match (&current, key) {
                    (Value::Object(map), Value::String(k)) => {
                        map.get(k).cloned().unwrap_or(Value::Null)
                    }
                    (Value::Array(arr), Value::Int(idx)) => {
                        let i = *idx as usize;
                        arr.get(i).cloned().unwrap_or(Value::Null)
                    }
                    (Value::Array(arr), Value::String(k)) => {
                        if let Ok(i) = k.parse::<usize>() {
                            arr.get(i).cloned().unwrap_or(Value::Null)
                        } else {
                            Value::Null
                        }
                    }
                    _ => Value::Null,
                };
            }
            Ok(current)
        }
        "JSON_OBJECT" => {
            let mut map = HashMap::new();
            let mut i = 0;
            while i + 1 < args.len() {
                let key = match &args[i] {
                    Value::String(s) => s.clone(),
                    Value::Int(n) => n.to_string(),
                    Value::Float(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => "null".to_string(),
                    other => format!("{:?}", other),
                };
                map.insert(key, args[i + 1].clone());
                i += 2;
            }
            Ok(Value::Object(map))
        }
        "JSON_ARRAY" => Ok(Value::Array(args.to_vec())),
        "JSON_TYPEOF" => {
            let val = match args.first() {
                Some(Value::String(s)) => match serde_json::from_str::<serde_json::Value>(s) {
                    Ok(v) => serde_json_to_value(v),
                    Err(_) => Value::String(s.clone()),
                },
                Some(other) => other.clone(),
                None => return Ok(Value::Null),
            };
            let type_name = match &val {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Int(_) => "number",
                Value::Float(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
                _ => "unknown",
            };
            Ok(Value::String(type_name.to_string()))
        }
        "JSON_ARRAY_LENGTH" => {
            let val = match args.first() {
                Some(Value::String(s)) => match serde_json::from_str::<serde_json::Value>(s) {
                    Ok(v) => serde_json_to_value(v),
                    Err(_) => return Ok(Value::Null),
                },
                Some(other) => other.clone(),
                None => return Ok(Value::Null),
            };
            match &val {
                Value::Array(arr) => Ok(Value::Int(arr.len() as i64)),
                _ => Ok(Value::Null),
            }
        }
        "JSON_VALID" => match args.first() {
            Some(Value::String(s)) => {
                if serde_json::from_str::<serde_json::Value>(s).is_ok() {
                    Ok(Value::Int(1))
                } else {
                    Ok(Value::Int(0))
                }
            }
            Some(Value::Object(_)) | Some(Value::Array(_)) => Ok(Value::Int(1)),
            _ => Ok(Value::Int(0)),
        },
        "JSON_KEYS" | "JSON_OBJECT_KEYS" => {
            let val = match args.first() {
                Some(Value::String(s)) => match serde_json::from_str::<serde_json::Value>(s) {
                    Ok(v) => serde_json_to_value(v),
                    Err(_) => return Ok(Value::Null),
                },
                Some(other) => other.clone(),
                None => return Ok(Value::Null),
            };
            match &val {
                Value::Object(map) => {
                    let mut keys: Vec<String> = map.keys().cloned().collect();
                    keys.sort();
                    Ok(Value::Array(keys.into_iter().map(Value::String).collect()))
                }
                _ => Ok(Value::Null),
            }
        }
        "JSON_EXTRACT_PATH_TEXT" => {
            if args.is_empty() {
                return Ok(Value::Null);
            }
            let json = crate::ast::value_to_serde_json(&args[0]);
            let mut current = json;
            for key in &args[1..] {
                current = match key {
                    Value::String(k) => current.get(k).cloned().unwrap_or(serde_json::Value::Null),
                    Value::Int(i) => current
                        .get(*i as usize)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                    _ => serde_json::Value::Null,
                };
            }
            match current {
                serde_json::Value::Null => Ok(Value::Null),
                serde_json::Value::String(s) => Ok(Value::String(s)),
                other => Ok(Value::String(other.to_string())),
            }
        }
        "JSON_BUILD_OBJECT" => {
            let mut map = HashMap::new();
            let mut i = 0;
            while i + 1 < args.len() {
                let key = match &args[i] {
                    Value::String(s) => s.clone(),
                    other => format!("{:?}", other),
                };
                map.insert(key, args[i + 1].clone());
                i += 2;
            }
            let json_map: serde_json::Map<String, serde_json::Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), crate::ast::value_to_serde_json(v)))
                .collect();
            Ok(Value::String(
                serde_json::Value::Object(json_map).to_string(),
            ))
        }
        "JSON_BUILD_ARRAY" => {
            let arr: Vec<serde_json::Value> =
                args.iter().map(crate::ast::value_to_serde_json).collect();
            Ok(Value::String(serde_json::Value::Array(arr).to_string()))
        }
        "JSONB_SET" | "JSONB_INSERT" => {
            if args.len() < 3 {
                return Ok(Value::Null);
            }
            let mut json = crate::ast::value_to_serde_json(&args[0]);
            let path = match &args[1] {
                Value::String(s) => {
                    let s = s.trim();
                    if s.starts_with('{') && s.ends_with('}') {
                        s[1..s.len() - 1]
                            .split(',')
                            .map(|p| p.trim().to_string())
                            .collect::<Vec<_>>()
                    } else {
                        vec![s.to_string()]
                    }
                }
                _ => return Ok(Value::Null),
            };
            // Third arg is a JSON value — try parsing string as JSON first
            let new_val = match &args[2] {
                Value::String(s) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.clone()))
                }
                other => crate::ast::value_to_serde_json(other),
            };
            if path.is_empty() {
                return Ok(Value::Null);
            }
            fn set_at_path(json: &mut serde_json::Value, path: &[String], val: serde_json::Value) {
                if path.len() == 1 {
                    match json {
                        serde_json::Value::Object(map) => {
                            map.insert(path[0].clone(), val);
                        }
                        serde_json::Value::Array(arr) => {
                            if let Ok(idx) = path[0].parse::<usize>() {
                                if idx < arr.len() {
                                    arr[idx] = val;
                                } else {
                                    arr.push(val);
                                }
                            }
                        }
                        _ => {}
                    }
                } else if !path.is_empty() {
                    let next = match json {
                        serde_json::Value::Object(map) => map
                            .entry(path[0].clone())
                            .or_insert(serde_json::Value::Object(serde_json::Map::new())),
                        serde_json::Value::Array(arr) => {
                            if let Ok(idx) = path[0].parse::<usize>() {
                                if idx < arr.len() {
                                    &mut arr[idx]
                                } else {
                                    return;
                                }
                            } else {
                                return;
                            }
                        }
                        _ => return,
                    };
                    set_at_path(next, &path[1..], val);
                }
            }
            set_at_path(&mut json, &path, new_val);
            Ok(Value::String(json.to_string()))
        }
        "JSON_STRIP_NULLS" => {
            if args.is_empty() {
                return Ok(Value::Null);
            }
            let json = crate::ast::value_to_serde_json(&args[0]);
            fn strip_nulls(v: serde_json::Value) -> serde_json::Value {
                match v {
                    serde_json::Value::Object(map) => {
                        let filtered: serde_json::Map<String, serde_json::Value> = map
                            .into_iter()
                            .filter(|(_, v)| !v.is_null())
                            .map(|(k, v)| (k, strip_nulls(v)))
                            .collect();
                        serde_json::Value::Object(filtered)
                    }
                    serde_json::Value::Array(arr) => {
                        serde_json::Value::Array(arr.into_iter().map(strip_nulls).collect())
                    }
                    other => other,
                }
            }
            Ok(Value::String(strip_nulls(json).to_string()))
        }
        "JSON_MERGE_PATCH" => {
            if args.len() < 2 {
                return Ok(args.first().cloned().unwrap_or(Value::Null));
            }
            let mut target = crate::ast::value_to_serde_json(&args[0]);
            let patch = crate::ast::value_to_serde_json(&args[1]);
            fn merge_patch(target: &mut serde_json::Value, patch: &serde_json::Value) {
                if let serde_json::Value::Object(p) = patch {
                    if !target.is_object() {
                        *target = serde_json::Value::Object(serde_json::Map::new());
                    }
                    if let serde_json::Value::Object(t) = target {
                        for (k, v) in p {
                            if v.is_null() {
                                t.remove(k);
                            } else {
                                let entry = t.entry(k.clone()).or_insert(serde_json::Value::Null);
                                merge_patch(entry, v);
                            }
                        }
                    }
                } else {
                    *target = patch.clone();
                }
            }
            merge_patch(&mut target, &patch);
            Ok(Value::String(target.to_string()))
        }
        "JSON_ARRAY_ELEMENTS" => {
            let json = match args.first() {
                Some(v) => crate::ast::value_to_serde_json(v),
                None => return Ok(Value::Null),
            };
            match json {
                serde_json::Value::Array(arr) => {
                    let elements: Vec<Value> = arr
                        .into_iter()
                        .map(|v| match v {
                            serde_json::Value::String(s) => Value::String(s),
                            serde_json::Value::Null => Value::Null,
                            serde_json::Value::Bool(b) => Value::Bool(b),
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    Value::Int(i)
                                } else {
                                    Value::Float(n.as_f64().unwrap_or(0.0))
                                }
                            }
                            other => Value::String(other.to_string()),
                        })
                        .collect();
                    Ok(Value::Array(elements))
                }
                _ => Ok(Value::Null),
            }
        }

        // ==================== REGEX FUNCTIONS ====================
        "REGEXP_REPLACE" => {
            if args.len() >= 3 {
                if let (Value::String(s), Value::String(pattern), Value::String(replacement)) =
                    (&args[0], &args[1], &args[2])
                {
                    // Check for optional flags argument (4th arg)
                    let global = if let Some(Value::String(flags)) = args.get(3) {
                        flags.contains('g')
                    } else {
                        false
                    };
                    if let Ok(re) = regex::Regex::new(pattern) {
                        let result = if global {
                            re.replace_all(s, replacement.as_str()).to_string()
                        } else {
                            re.replace(s, replacement.as_str()).to_string()
                        };
                        return Ok(Value::String(result));
                    }
                }
            }
            Ok(Value::Null)
        }
        "REGEXP_MATCH" => {
            if args.len() >= 2 {
                if let (Value::String(s), Value::String(pattern)) = (&args[0], &args[1]) {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if let Some(m) = re.find(s) {
                            return Ok(Value::String(m.as_str().to_string()));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }
        "REGEXP_MATCHES" => {
            if args.len() >= 2 {
                if let (Value::String(s), Value::String(pattern)) = (&args[0], &args[1]) {
                    if let Ok(re) = regex::Regex::new(pattern) {
                        if let Some(caps) = re.captures(s) {
                            // Return array of capture groups (group 0 = full match if no groups)
                            let matches: Vec<Value> = if caps.len() > 1 {
                                (1..caps.len())
                                    .map(|i| {
                                        caps.get(i)
                                            .map(|m| Value::String(m.as_str().to_string()))
                                            .unwrap_or(Value::Null)
                                    })
                                    .collect()
                            } else {
                                vec![Value::String(
                                    caps.get(0).map(|m| m.as_str().to_string()).unwrap_or_default(),
                                )]
                            };
                            return Ok(Value::Array(matches));
                        }
                        return Ok(Value::Null);
                    }
                }
            }
            Ok(Value::Null)
        }

        // ==================== HASH FUNCTIONS ====================
        "MD5" => {
            if let Some(Value::String(s)) = args.first() {
                use md5::Digest;
                let mut hasher = md5::Md5::new();
                hasher.update(s.as_bytes());
                let result = hasher.finalize();
                return Ok(Value::String(format!("{:x}", result)));
            }
            Ok(Value::Null)
        }
        "SHA256" | "SHA2" => {
            if let Some(Value::String(s)) = args.first() {
                use sha2::Digest;
                let mut hasher = sha2::Sha256::new();
                hasher.update(s.as_bytes());
                let result = hasher.finalize();
                return Ok(Value::String(format!("{:x}", result)));
            }
            Ok(Value::Null)
        }

        // ==================== ARRAY FUNCTIONS ====================
        "GENERATE_SERIES" => {
            if args.len() >= 2 {
                if let (Some(start), Some(stop)) = (args[0].as_int(), args[1].as_int()) {
                    let step = args
                        .get(2)
                        .and_then(|a| a.as_int())
                        .unwrap_or(if stop >= start { 1 } else { -1 });
                    if step == 0 {
                        return Ok(Value::Null);
                    }
                    let mut result = Vec::new();
                    let mut i = start;
                    while (step > 0 && i <= stop) || (step < 0 && i >= stop) {
                        result.push(Value::Int(i));
                        i += step;
                    }
                    return Ok(Value::Array(result));
                }
            }
            Ok(Value::Null)
        }
        "ARRAY_LENGTH" => {
            if let Some(Value::Array(arr)) = args.first() {
                return Ok(Value::Int(arr.len() as i64));
            }
            Ok(Value::Null)
        }
        "ARRAY_APPEND" => {
            if args.len() >= 2 {
                if let Value::Array(arr) = &args[0] {
                    let mut new_arr = arr.clone();
                    new_arr.push(args[1].clone());
                    return Ok(Value::Array(new_arr));
                }
            }
            Ok(Value::Null)
        }
        "ARRAY_CAT" => {
            if args.len() >= 2 {
                if let (Value::Array(a), Value::Array(b)) = (&args[0], &args[1]) {
                    let mut result = a.clone();
                    result.extend(b.iter().cloned());
                    return Ok(Value::Array(result));
                }
            }
            Ok(Value::Null)
        }
        "ARRAY_CONTAINS" => {
            if args.len() >= 2 {
                if let Value::Array(arr) = &args[0] {
                    return Ok(Value::Bool(arr.contains(&args[1])));
                }
            }
            Ok(Value::Bool(false))
        }
        "ARRAY_PREPEND" => {
            if args.len() >= 2 {
                if let Value::Array(arr) = &args[1] {
                    let mut new_arr = vec![args[0].clone()];
                    new_arr.extend(arr.iter().cloned());
                    return Ok(Value::Array(new_arr));
                }
            }
            Ok(Value::Null)
        }
        "ARRAY_REMOVE" => {
            if args.len() >= 2 {
                if let Value::Array(arr) = &args[0] {
                    let filtered: Vec<Value> = arr
                        .iter()
                        .filter(|v| !values_equal(v, &args[1]))
                        .cloned()
                        .collect();
                    return Ok(Value::Array(filtered));
                }
            }
            Ok(Value::Null)
        }
        "ARRAY_POSITION" => {
            if args.len() >= 2 {
                if let Value::Array(arr) = &args[0] {
                    for (i, v) in arr.iter().enumerate() {
                        if values_equal(v, &args[1]) {
                            return Ok(Value::Int((i + 1) as i64)); // 1-based
                        }
                    }
                    return Ok(Value::Null);
                }
            }
            Ok(Value::Null)
        }
        "ARRAY_DISTINCT" => {
            if let Some(Value::Array(arr)) = args.first() {
                let mut seen = Vec::new();
                for v in arr {
                    if !seen.iter().any(|s| values_equal(s, v)) {
                        seen.push(v.clone());
                    }
                }
                return Ok(Value::Array(seen));
            }
            Ok(Value::Null)
        }
        "ARRAY_SORT" => {
            if let Some(Value::Array(arr)) = args.first() {
                let mut sorted = arr.clone();
                sorted.sort_by(|a, b| {
                    let fa = as_f64(a);
                    let fb = as_f64(b);
                    match (fa, fb) {
                        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
                        _ => {
                            let sa = match a {
                                Value::String(s) => s.clone(),
                                other => format!("{:?}", other),
                            };
                            let sb = match b {
                                Value::String(s) => s.clone(),
                                other => format!("{:?}", other),
                            };
                            sa.cmp(&sb)
                        }
                    }
                });
                return Ok(Value::Array(sorted));
            }
            Ok(Value::Null)
        }
        "ARRAY_REVERSE" => {
            if let Some(Value::Array(arr)) = args.first() {
                let mut reversed = arr.clone();
                reversed.reverse();
                return Ok(Value::Array(reversed));
            }
            Ok(Value::Null)
        }
        "ARRAY_SLICE" => {
            // array_slice(array, start, end) — 1-based, inclusive
            if args.len() >= 3 {
                if let Value::Array(arr) = &args[0] {
                    let start = args[1].as_int().unwrap_or(1).max(1) as usize;
                    let end = args[2].as_int().unwrap_or(arr.len() as i64) as usize;
                    // Convert 1-based inclusive to 0-based
                    let start_idx = start.saturating_sub(1);
                    let end_idx = end.min(arr.len());
                    if start_idx < end_idx {
                        return Ok(Value::Array(arr[start_idx..end_idx].to_vec()));
                    }
                    return Ok(Value::Array(Vec::new()));
                }
            }
            Ok(Value::Null)
        }
        "UNNEST" => {
            // For scalar context, return the array as-is (true SRF requires row expansion)
            if let Some(Value::Array(arr)) = args.first() {
                return Ok(Value::Array(arr.clone()));
            }
            Ok(Value::Null)
        }
        "JSON_SET" => {
            // Alias for JSONB_SET: json_set(json, path_array, new_value)
            if args.len() < 3 {
                return Ok(Value::Null);
            }
            let mut json = crate::ast::value_to_serde_json(&args[0]);
            let path = match &args[1] {
                Value::String(s) => {
                    let s = s.trim();
                    if s.starts_with('{') && s.ends_with('}') {
                        s[1..s.len() - 1]
                            .split(',')
                            .map(|p| p.trim().to_string())
                            .collect::<Vec<_>>()
                    } else {
                        vec![s.to_string()]
                    }
                }
                Value::Array(arr) => arr
                    .iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        Value::Int(i) => Some(i.to_string()),
                        _ => None,
                    })
                    .collect(),
                _ => return Ok(Value::Null),
            };
            let new_val = match &args[2] {
                Value::String(s) => {
                    serde_json::from_str(s).unwrap_or(serde_json::Value::String(s.clone()))
                }
                other => crate::ast::value_to_serde_json(other),
            };
            if path.is_empty() {
                return Ok(Value::Null);
            }
            fn set_at_path_json(json: &mut serde_json::Value, path: &[String], val: serde_json::Value) {
                if path.len() == 1 {
                    match json {
                        serde_json::Value::Object(map) => {
                            map.insert(path[0].clone(), val);
                        }
                        serde_json::Value::Array(arr) => {
                            if let Ok(idx) = path[0].parse::<usize>() {
                                if idx < arr.len() {
                                    arr[idx] = val;
                                } else {
                                    arr.push(val);
                                }
                            }
                        }
                        _ => {}
                    }
                } else if !path.is_empty() {
                    let next = match json {
                        serde_json::Value::Object(map) => map
                            .entry(path[0].clone())
                            .or_insert(serde_json::Value::Object(serde_json::Map::new())),
                        serde_json::Value::Array(arr) => {
                            if let Ok(idx) = path[0].parse::<usize>() {
                                if idx < arr.len() {
                                    &mut arr[idx]
                                } else {
                                    return;
                                }
                            } else {
                                return;
                            }
                        }
                        _ => return,
                    };
                    set_at_path_json(next, &path[1..], val);
                }
            }
            set_at_path_json(&mut json, &path, new_val);
            Ok(Value::String(json.to_string()))
        }

        // ==================== TIME-SERIES FUNCTIONS ====================
        "TIME_BUCKET" => {
            if args.len() >= 2 {
                let interval_secs = match &args[0] {
                    Value::String(s) => crate::geo::parse_interval_to_seconds(s),
                    Value::Int(n) => Some(*n),
                    _ => None,
                };
                let ts = match &args[1] {
                    Value::Timestamp(t) => Some(*t),
                    Value::Int(t) => Some(*t),
                    _ => None,
                };
                if let (Some(interval), Some(timestamp)) = (interval_secs, ts) {
                    if interval > 0 {
                        let bucketed = (timestamp / interval) * interval;
                        return Ok(Value::Timestamp(bucketed));
                    }
                }
            }
            Ok(Value::Null)
        }
        "HISTOGRAM" => {
            if args.len() >= 4 {
                let val = as_f64(&args[0]);
                let min_v = as_f64(&args[1]);
                let max_v = as_f64(&args[2]);
                let buckets = match &args[3] {
                    Value::Int(i) => Some(*i),
                    _ => None,
                };
                if let (Some(v), Some(mn), Some(mx), Some(n)) = (val, min_v, max_v, buckets) {
                    if n > 0 && mx > mn {
                        let bucket_width = (mx - mn) / n as f64;
                        let idx = ((v - mn) / bucket_width).floor() as i64;
                        let idx = idx.max(0).min(n - 1);
                        return Ok(Value::Int(idx));
                    }
                }
            }
            Ok(Value::Null)
        }

        // ==================== SPATIAL FUNCTIONS ====================
        "ST_POINT" => {
            if args.len() >= 2 {
                let x = match &args[0] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                let y = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                return Ok(Value::String(format!("POINT({} {})", x, y)));
            }
            Ok(Value::Null)
        }
        "ST_GEOMFROMTEXT" => {
            if let Some(Value::String(s)) = args.first() {
                if crate::geo::parse_wkt(s).is_some() {
                    return Ok(Value::String(s.clone()));
                }
            }
            Ok(Value::Null)
        }
        "ST_GEOMFROMGEOJSON" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::parse_geojson(s) {
                    return Ok(Value::String(crate::geo::to_wkt(&g)));
                }
            }
            Ok(Value::Null)
        }
        "ST_DISTANCE" => {
            if args.len() >= 2 {
                let s1 = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let s2 = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                if let (Some(g1), Some(g2)) = (
                    crate::geo::value_to_geometry(s1),
                    crate::geo::value_to_geometry(s2),
                ) {
                    if let Some(d) = crate::geo::geometry_distance(&g1, &g2) {
                        return Ok(Value::Float(d));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_AREA" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some(a) = crate::geo::polygon_area(&g) {
                        return Ok(Value::Float(a));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_LENGTH" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some(l) = crate::geo::linestring_length(&g) {
                        return Ok(Value::Float(l));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_CONTAINS" => {
            if args.len() >= 2 {
                let s1 = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let s2 = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                if let (Some(g1), Some(g2)) = (
                    crate::geo::value_to_geometry(s1),
                    crate::geo::value_to_geometry(s2),
                ) {
                    return Ok(Value::Bool(crate::geo::contains(&g1, &g2)));
                }
            }
            Ok(Value::Null)
        }
        "ST_WITHIN" => {
            if args.len() >= 2 {
                let s1 = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let s2 = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                if let (Some(g1), Some(g2)) = (
                    crate::geo::value_to_geometry(s1),
                    crate::geo::value_to_geometry(s2),
                ) {
                    return Ok(Value::Bool(crate::geo::contains(&g2, &g1)));
                }
            }
            Ok(Value::Null)
        }
        "ST_INTERSECTS" => {
            if args.len() >= 2 {
                let s1 = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let s2 = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                if let (Some(g1), Some(g2)) = (
                    crate::geo::value_to_geometry(s1),
                    crate::geo::value_to_geometry(s2),
                ) {
                    return Ok(Value::Bool(crate::geo::intersects(&g1, &g2)));
                }
            }
            Ok(Value::Null)
        }
        "ST_DWITHIN" => {
            if args.len() >= 3 {
                let s1 = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let s2 = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dist = match &args[2] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                if let (Some(g1), Some(g2)) = (
                    crate::geo::value_to_geometry(s1),
                    crate::geo::value_to_geometry(s2),
                ) {
                    return Ok(Value::Bool(crate::geo::dwithin(&g1, &g2, dist)));
                }
            }
            Ok(Value::Null)
        }
        "ST_X" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some((x, _)) = crate::geo::point_coords(&g) {
                        return Ok(Value::Float(x));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_Y" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some((_, y)) = crate::geo::point_coords(&g) {
                        return Ok(Value::Float(y));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_ASTEXT" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    return Ok(Value::String(crate::geo::to_wkt(&g)));
                }
            }
            Ok(Value::Null)
        }
        "ST_ASGEOJSON" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    return Ok(Value::String(crate::geo::to_geojson(&g)));
                }
            }
            Ok(Value::Null)
        }
        "ST_BUFFER" => {
            if args.len() >= 2 {
                let s = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dist = match &args[1] {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Ok(Value::Null),
                };
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some(buffered) = crate::geo::buffer(&g, dist) {
                        return Ok(Value::String(crate::geo::to_wkt(&buffered)));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_CENTROID" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some(c) = crate::geo::centroid(&g) {
                        return Ok(Value::String(crate::geo::to_wkt(&c)));
                    }
                }
            }
            Ok(Value::Null)
        }
        "ST_ENVELOPE" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(g) = crate::geo::value_to_geometry(s) {
                    if let Some(e) = crate::geo::envelope(&g) {
                        return Ok(Value::String(crate::geo::to_wkt(&e)));
                    }
                }
            }
            Ok(Value::Null)
        }

        // ==================== SPATIAL 3D FUNCTIONS ====================
        // 3D analogues of the 2D ST_* GIS functions. Text encoding:
        //   POINT3(x y z)             — a 3D point
        //   BBOX3(x1 y1 z1, x2 y2 z2) — an axis-aligned 3D bbox
        // These are cheap function calls (picojoules) — the cascade
        // picks them before LLM. Backed by the Point3/Bbox3 types in
        // joule_db_core::types::spatial.

        "ST_DISTANCE3D" => {
            // Euclidean distance between two POINT3 values.
            if args.len() >= 2 {
                if let (Some(a), Some(b)) = (parse_point3_arg(&args[0]), parse_point3_arg(&args[1])) {
                    let dx = a.x - b.x;
                    let dy = a.y - b.y;
                    let dz = a.z - b.z;
                    return Ok(Value::Float((dx * dx + dy * dy + dz * dz).sqrt()));
                }
            }
            Ok(Value::Null)
        }
        "ST_DWITHIN3D" => {
            // True if two POINT3 values are within `dist` of each other.
            if args.len() >= 3 {
                if let (Some(a), Some(b)) = (parse_point3_arg(&args[0]), parse_point3_arg(&args[1])) {
                    let dist = match &args[2] {
                        Value::Float(f) => *f,
                        Value::Int(i) => *i as f64,
                        _ => return Ok(Value::Null),
                    };
                    let dx = a.x - b.x;
                    let dy = a.y - b.y;
                    let dz = a.z - b.z;
                    return Ok(Value::Bool((dx * dx + dy * dy + dz * dz).sqrt() <= dist));
                }
            }
            Ok(Value::Null)
        }
        "ST_CONTAINS3D" => {
            // True if BBOX3 `a` fully contains POINT3 `b`.
            if args.len() >= 2 {
                if let (Some(bbox), Some(pt)) = (parse_bbox3_arg(&args[0]), parse_point3_arg(&args[1])) {
                    return Ok(Value::Bool(bbox.contains(pt)));
                }
            }
            Ok(Value::Null)
        }
        "ST_WITHIN3D" => {
            // True if POINT3 `a` is inside BBOX3 `b`.
            if args.len() >= 2 {
                if let (Some(pt), Some(bbox)) = (parse_point3_arg(&args[0]), parse_bbox3_arg(&args[1])) {
                    return Ok(Value::Bool(bbox.contains(pt)));
                }
            }
            Ok(Value::Null)
        }
        "ST_INTERSECTS3D" => {
            // True if two BBOX3 values overlap.
            if args.len() >= 2 {
                if let (Some(a), Some(b)) = (parse_bbox3_arg(&args[0]), parse_bbox3_arg(&args[1])) {
                    return Ok(Value::Bool(a.intersects(&b)));
                }
            }
            Ok(Value::Null)
        }
        "ST_X3D" => {
            if let Some(p) = args.first().and_then(parse_point3_arg) {
                return Ok(Value::Float(p.x));
            }
            Ok(Value::Null)
        }
        "ST_Y3D" => {
            if let Some(p) = args.first().and_then(parse_point3_arg) {
                return Ok(Value::Float(p.y));
            }
            Ok(Value::Null)
        }
        "ST_Z3D" => {
            if let Some(p) = args.first().and_then(parse_point3_arg) {
                return Ok(Value::Float(p.z));
            }
            Ok(Value::Null)
        }
        "ST_MAKEPOINT3D" => {
            // ST_MAKEPOINT3D(x, y, z) → 'POINT3(x y z)'
            if args.len() >= 3 {
                let x = match &args[0] { Value::Float(f) => Some(*f), Value::Int(i) => Some(*i as f64), _ => None };
                let y = match &args[1] { Value::Float(f) => Some(*f), Value::Int(i) => Some(*i as f64), _ => None };
                let z = match &args[2] { Value::Float(f) => Some(*f), Value::Int(i) => Some(*i as f64), _ => None };
                if let (Some(x), Some(y), Some(z)) = (x, y, z) {
                    return Ok(Value::String(format!("POINT3({} {} {})", x, y, z)));
                }
            }
            Ok(Value::Null)
        }
        "ST_MAKEBBOX3D" => {
            // ST_MAKEBBOX3D(x1, y1, z1, x2, y2, z2) → 'BBOX3(x1 y1 z1, x2 y2 z2)'
            if args.len() >= 6 {
                let coords: Vec<Option<f64>> = args[..6].iter().map(|a| match a {
                    Value::Float(f) => Some(*f), Value::Int(i) => Some(*i as f64), _ => None
                }).collect();
                if coords.iter().all(|c| c.is_some()) {
                    let c: Vec<f64> = coords.into_iter().map(|c| c.unwrap()).collect();
                    return Ok(Value::String(format!(
                        "BBOX3({} {} {}, {} {} {})", c[0], c[1], c[2], c[3], c[4], c[5]
                    )));
                }
            }
            Ok(Value::Null)
        }

        // ==================== VECTOR FUNCTIONS ====================
        "L2_DISTANCE" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(va), Some(vb)) = (
                        crate::vector::parse_vector(a),
                        crate::vector::parse_vector(b),
                    ) {
                        if let Some(d) = crate::vector::l2_distance(&va, &vb) {
                            return Ok(Value::Float(d));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }
        "COSINE_SIMILARITY" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(va), Some(vb)) = (
                        crate::vector::parse_vector(a),
                        crate::vector::parse_vector(b),
                    ) {
                        if let Some(d) = crate::vector::cosine_similarity(&va, &vb) {
                            return Ok(Value::Float(d));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }
        "COSINE_DISTANCE" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(va), Some(vb)) = (
                        crate::vector::parse_vector(a),
                        crate::vector::parse_vector(b),
                    ) {
                        if let Some(d) = crate::vector::cosine_distance(&va, &vb) {
                            return Ok(Value::Float(d));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }
        "INNER_PRODUCT" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(va), Some(vb)) = (
                        crate::vector::parse_vector(a),
                        crate::vector::parse_vector(b),
                    ) {
                        if let Some(d) = crate::vector::inner_product(&va, &vb) {
                            return Ok(Value::Float(d));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }
        "VECTOR_DISTANCE" => {
            if args.len() >= 3 {
                if let (Value::String(a), Value::String(b), Value::String(metric)) =
                    (&args[0], &args[1], &args[2])
                {
                    if let (Some(va), Some(vb)) = (
                        crate::vector::parse_vector(a),
                        crate::vector::parse_vector(b),
                    ) {
                        if let Some(d) = crate::vector::vector_distance(&va, &vb, metric) {
                            return Ok(Value::Float(d));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }
        "VECTOR_DIMS" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(v) = crate::vector::parse_vector(s) {
                    return Ok(Value::Int(crate::vector::vector_dims(&v) as i64));
                }
            }
            Ok(Value::Null)
        }
        "VECTOR_NORM" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(v) = crate::vector::parse_vector(s) {
                    return Ok(Value::Float(crate::vector::vector_norm(&v)));
                }
            }
            Ok(Value::Null)
        }
        "VECTOR_NORMALIZE" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(v) = crate::vector::parse_vector(s) {
                    return Ok(Value::String(crate::vector::vector_to_string(
                        &crate::vector::vector_normalize(&v),
                    )));
                }
            }
            Ok(Value::Null)
        }

        // ==================== GRAPH FUNCTIONS ====================
        "SHORTEST_PATH" => {
            if args.len() >= 5 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let start = node_id(&args[3])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid start node".to_string()))?;
                let end = node_id(&args[4])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid end node".to_string()))?;
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                match crate::graph::shortest_path(&edges, &start, &end) {
                    Some(path) => Ok(Value::Array(path.into_iter().map(Value::String).collect())),
                    None => Ok(Value::Null),
                }
            } else {
                Ok(Value::Null)
            }
        }
        "SHORTEST_PATH_LENGTH" => {
            if args.len() >= 5 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let start = node_id(&args[3])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid start node".to_string()))?;
                let end = node_id(&args[4])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid end node".to_string()))?;
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                match crate::graph::shortest_path_length(&edges, &start, &end) {
                    Some(len) => Ok(Value::Int(len)),
                    None => Ok(Value::Null),
                }
            } else {
                Ok(Value::Null)
            }
        }
        "NEIGHBORS" => {
            if args.len() >= 4 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let start = node_id(&args[3])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid start node".to_string()))?;
                let max_depth = args.get(4).and_then(|v| v.as_int()).unwrap_or(1);
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                let nbrs = crate::graph::neighbors(&edges, &start, max_depth);
                Ok(Value::Array(nbrs.into_iter().map(Value::String).collect()))
            } else {
                Ok(Value::Null)
            }
        }
        "CONNECTED_COMPONENTS" => {
            if args.len() >= 3 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                let comps = crate::graph::connected_components(&edges);
                if let Some(node_arg) = args.get(3) {
                    let node = node_id(node_arg)
                        .ok_or_else(|| QueryError::ExecutionError("Invalid node".to_string()))?;
                    match comps.get(&node) {
                        Some(id) => Ok(Value::Int(*id)),
                        None => Ok(Value::Null),
                    }
                } else {
                    let obj: HashMap<String, Value> =
                        comps.into_iter().map(|(k, v)| (k, Value::Int(v))).collect();
                    Ok(Value::Object(obj))
                }
            } else {
                Ok(Value::Null)
            }
        }
        "PAGERANK" => {
            // PAGERANK(table, src_col, dst_col [, node_id [, iterations [, damping]]])
            // OR: PAGERANK(table, src_col, dst_col [, iterations [, damping]])
            // Disambiguation: if args[3] is a String, it's a node ID; if numeric, it's iterations
            if args.len() >= 3 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                // Determine if args[3] is a node ID (String) or iterations (numeric)
                let node_arg = args.get(3).and_then(|v| {
                    if let Value::String(_) = v {
                        Some(v)
                    } else {
                        None
                    }
                });
                let param_start = if node_arg.is_some() { 4 } else { 3 };
                let iterations =
                    args.get(param_start).and_then(|v| v.as_int()).unwrap_or(20) as usize;
                let damping = args
                    .get(param_start + 1)
                    .and_then(|v| v.as_float())
                    .unwrap_or(0.85);
                let scores = crate::graph::pagerank(&edges, iterations, damping);
                if let Some(node_val) = node_arg {
                    let node = node_id(node_val)
                        .ok_or_else(|| QueryError::ExecutionError("Invalid node".to_string()))?;
                    match scores.get(&node) {
                        Some(score) => Ok(Value::Float(*score)),
                        None => Ok(Value::Null),
                    }
                } else {
                    let obj: HashMap<String, Value> = scores
                        .into_iter()
                        .map(|(k, v)| (k, Value::Float(v)))
                        .collect();
                    Ok(Value::Object(obj))
                }
            } else {
                Ok(Value::Null)
            }
        }
        "GRAPH_REACH" => {
            if args.len() >= 5 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let start = node_id(&args[3])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid start node".to_string()))?;
                let end = node_id(&args[4])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid end node".to_string()))?;
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                Ok(Value::Bool(crate::graph::graph_reach(&edges, &start, &end)))
            } else {
                Ok(Value::Null)
            }
        }
        "BFS_TRAVERSE" => {
            if args.len() >= 4 {
                let table = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let src_col = match &args[1] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let dst_col = match &args[2] {
                    Value::String(s) => s.as_str(),
                    _ => return Ok(Value::Null),
                };
                let start = node_id(&args[3])
                    .ok_or_else(|| QueryError::ExecutionError("Invalid start node".to_string()))?;
                let max_depth = args.get(4).map(|v| v.as_int().unwrap_or(10));
                let edges = load_graph_edges(table, src_col, dst_col, table_scanner)?;
                let traversal = crate::graph::bfs_traverse(&edges, &start, max_depth);
                Ok(Value::Array(
                    traversal.into_iter().map(Value::String).collect(),
                ))
            } else {
                Ok(Value::Null)
            }
        }

        // ==================== HDC FUNCTIONS ====================
        #[cfg(feature = "hdc")]
        "HDC_ENCODE" => {
            if args.len() >= 3 {
                if let (Value::String(domain), Value::String(method), Value::String(json_str)) =
                    (&args[0], &args[1], &args[2])
                {
                    match crate::hdc::hdc_encode(domain, method, json_str) {
                        Ok(hv_text) => return Ok(Value::String(hv_text)),
                        Err(e) => {
                            return Err(QueryError::ExecutionError(format!(
                                "HDC_ENCODE error: {}",
                                e
                            )));
                        }
                    }
                }
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_ENCODE_TEXT" => {
            if let Some(Value::String(text)) = args.first() {
                let dim = args
                    .get(1)
                    .and_then(|v| match v {
                        Value::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(crate::hdc::DEFAULT_DIMENSION);
                let hv = joule_db_hdc::BinaryHV::from_bytes(text.as_bytes(), dim);
                return Ok(Value::String(crate::hdc::serialize_hv(&hv)));
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_ENCODE_HASH" => {
            if let Some(Value::String(data)) = args.first() {
                let dim = args
                    .get(1)
                    .and_then(|v| match v {
                        Value::Int(i) => Some(*i as usize),
                        _ => None,
                    })
                    .unwrap_or(crate::hdc::DEFAULT_DIMENSION);
                let hv = joule_db_hdc::BinaryHV::from_data(data.as_bytes(), dim);
                return Ok(Value::String(crate::hdc::serialize_hv(&hv)));
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_SIMILARITY" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(hv_a), Some(hv_b)) =
                        (crate::hdc::deserialize_hv(a), crate::hdc::deserialize_hv(b))
                    {
                        return Ok(Value::Float(hv_a.similarity(&hv_b) as f64));
                    }
                }
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_BIPOLAR_SIMILARITY" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(hv_a), Some(hv_b)) =
                        (crate::hdc::deserialize_hv(a), crate::hdc::deserialize_hv(b))
                    {
                        return Ok(Value::Float(hv_a.bipolar_similarity(&hv_b) as f64));
                    }
                }
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_DISTANCE" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(hv_a), Some(hv_b)) =
                        (crate::hdc::deserialize_hv(a), crate::hdc::deserialize_hv(b))
                    {
                        return Ok(Value::Int(hv_a.hamming_distance(&hv_b) as i64));
                    }
                }
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_BIND" => {
            if args.len() >= 2 {
                if let (Value::String(a), Value::String(b)) = (&args[0], &args[1]) {
                    if let (Some(hv_a), Some(hv_b)) =
                        (crate::hdc::deserialize_hv(a), crate::hdc::deserialize_hv(b))
                    {
                        let bound = hv_a.bind(&hv_b);
                        return Ok(Value::String(crate::hdc::serialize_hv(&bound)));
                    }
                }
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_BUNDLE" => {
            if args.len() >= 2 {
                let hvs: Vec<_> = args
                    .iter()
                    .filter_map(|arg| {
                        if let Value::String(s) = arg {
                            crate::hdc::deserialize_hv(s)
                        } else {
                            None
                        }
                    })
                    .collect();
                if hvs.len() >= 2 {
                    let dim = hvs[0].dimension();
                    let mut acc = joule_db_hdc::BundleAccumulator::new(dim);
                    for hv in &hvs {
                        acc.add(hv);
                    }
                    return Ok(Value::String(crate::hdc::serialize_hv(&acc.threshold())));
                }
            }
            Ok(Value::Null)
        }

        #[cfg(feature = "hdc")]
        "HDC_DIMS" => {
            if let Some(Value::String(s)) = args.first() {
                if let Some(hv) = crate::hdc::deserialize_hv(s) {
                    return Ok(Value::Int(hv.dimension() as i64));
                }
            }
            Ok(Value::Null)
        }

        // ARRAY[expr, ...] constructor — builds a Value::Array from arguments
        "ARRAY" => {
            let elements: Vec<Value> = args.to_vec();
            Ok(Value::Array(elements))
        }

        _ => Err(QueryError::UnknownFunction(name.to_string())),
    }
}

// ============================================================================
// Spatial 3D text format parsers
// ============================================================================
//
// Text encoding:
//   POINT3(1.0 2.0 3.0)
//   BBOX3(0 0 0, 10 10 10)
//
// These are the 3D analogues of WKT, parsed from `Value::String`.

use joule_db_core::types::spatial::{Bbox3, Point3};

/// Try to extract a `Point3` from a `Value`.
fn parse_point3_arg(v: &Value) -> Option<Point3> {
    match v {
        Value::String(s) => parse_point3_text(s),
        _ => None,
    }
}

/// Try to extract a `Bbox3` from a `Value`.
fn parse_bbox3_arg(v: &Value) -> Option<Bbox3> {
    match v {
        Value::String(s) => parse_bbox3_text(s),
        _ => None,
    }
}

/// Parse `"POINT3(x y z)"` → `Point3`.
fn parse_point3_text(s: &str) -> Option<Point3> {
    let s = s.trim();
    let inner = s.strip_prefix("POINT3(")?.strip_suffix(')')?;
    let parts: Vec<f64> = inner
        .split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect();
    if parts.len() == 3 {
        Some(Point3::new(parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

/// Parse `"BBOX3(x1 y1 z1, x2 y2 z2)"` → `Bbox3`.
fn parse_bbox3_text(s: &str) -> Option<Bbox3> {
    let s = s.trim();
    let inner = s.strip_prefix("BBOX3(")?.strip_suffix(')')?;
    let halves: Vec<&str> = inner.splitn(2, ',').collect();
    if halves.len() != 2 {
        return None;
    }
    let min_parts: Vec<f64> = halves[0]
        .split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect();
    let max_parts: Vec<f64> = halves[1]
        .split_whitespace()
        .filter_map(|p| p.parse().ok())
        .collect();
    if min_parts.len() == 3 && max_parts.len() == 3 {
        Some(Bbox3::new(
            Point3::new(min_parts[0], min_parts[1], min_parts[2]),
            Point3::new(max_parts[0], max_parts[1], max_parts[2]),
        ))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Spatial 3D function tests ──────────────────────────────────

    #[test]
    fn test_st_distance3d() {
        let a = Value::String("POINT3(0 0 0)".into());
        let b = Value::String("POINT3(3 4 0)".into());
        let r = eval_scalar_function("ST_DISTANCE3D", &[a, b], None).unwrap();
        match r {
            Value::Float(d) => assert!((d - 5.0).abs() < 1e-9),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_st_dwithin3d() {
        let a = Value::String("POINT3(0 0 0)".into());
        let b = Value::String("POINT3(1 1 1)".into());
        // sqrt(3) ≈ 1.732 — within 2.0 but not 1.0.
        let r1 = eval_scalar_function("ST_DWITHIN3D", &[a.clone(), b.clone(), Value::Float(2.0)], None).unwrap();
        assert_eq!(r1, Value::Bool(true));
        let r2 = eval_scalar_function("ST_DWITHIN3D", &[a, b, Value::Float(1.0)], None).unwrap();
        assert_eq!(r2, Value::Bool(false));
    }

    #[test]
    fn test_st_contains3d() {
        let bbox = Value::String("BBOX3(0 0 0, 10 10 10)".into());
        let inside = Value::String("POINT3(5 5 5)".into());
        let outside = Value::String("POINT3(15 5 5)".into());
        let r1 = eval_scalar_function("ST_CONTAINS3D", &[bbox.clone(), inside], None).unwrap();
        assert_eq!(r1, Value::Bool(true));
        let r2 = eval_scalar_function("ST_CONTAINS3D", &[bbox, outside], None).unwrap();
        assert_eq!(r2, Value::Bool(false));
    }

    #[test]
    fn test_st_within3d() {
        let pt = Value::String("POINT3(5 5 5)".into());
        let bbox = Value::String("BBOX3(0 0 0, 10 10 10)".into());
        let r = eval_scalar_function("ST_WITHIN3D", &[pt, bbox], None).unwrap();
        assert_eq!(r, Value::Bool(true));
    }

    #[test]
    fn test_st_intersects3d() {
        let a = Value::String("BBOX3(0 0 0, 5 5 5)".into());
        let b = Value::String("BBOX3(3 3 3, 10 10 10)".into());
        let c = Value::String("BBOX3(6 6 6, 10 10 10)".into());
        let r1 = eval_scalar_function("ST_INTERSECTS3D", &[a.clone(), b], None).unwrap();
        assert_eq!(r1, Value::Bool(true));
        let r2 = eval_scalar_function("ST_INTERSECTS3D", &[a, c], None).unwrap();
        assert_eq!(r2, Value::Bool(false));
    }

    #[test]
    fn test_st_xyz3d() {
        let pt = Value::String("POINT3(1.5 2.5 3.5)".into());
        assert_eq!(eval_scalar_function("ST_X3D", &[pt.clone()], None).unwrap(), Value::Float(1.5));
        assert_eq!(eval_scalar_function("ST_Y3D", &[pt.clone()], None).unwrap(), Value::Float(2.5));
        assert_eq!(eval_scalar_function("ST_Z3D", &[pt], None).unwrap(), Value::Float(3.5));
    }

    #[test]
    fn test_st_makepoint3d() {
        let r = eval_scalar_function("ST_MAKEPOINT3D", &[Value::Float(1.0), Value::Float(2.0), Value::Float(3.0)], None).unwrap();
        assert_eq!(r, Value::String("POINT3(1 2 3)".into()));
    }

    #[test]
    fn test_st_makebbox3d() {
        let r = eval_scalar_function("ST_MAKEBBOX3D", &[
            Value::Float(0.0), Value::Float(0.0), Value::Float(0.0),
            Value::Float(10.0), Value::Float(10.0), Value::Float(10.0),
        ], None).unwrap();
        assert_eq!(r, Value::String("BBOX3(0 0 0, 10 10 10)".into()));
    }

    #[test]
    fn test_parse_point3_text() {
        let p = parse_point3_text("POINT3(1.5 -2.5 3.25)").unwrap();
        assert_eq!(p.x, 1.5);
        assert_eq!(p.y, -2.5);
        assert_eq!(p.z, 3.25);
        assert!(parse_point3_text("garbage").is_none());
        assert!(parse_point3_text("POINT3(1 2)").is_none()); // only 2 coords
    }

    #[test]
    fn test_parse_bbox3_text() {
        let b = parse_bbox3_text("BBOX3(0 0 0, 10 20 30)").unwrap();
        assert_eq!(b.min.x, 0.0);
        assert_eq!(b.max.z, 30.0);
        assert!(parse_bbox3_text("BBOX3(0 0 0)").is_none()); // missing max
    }

    // ── Original tests (kept intact) ──────────────────────────────

    #[test]
    fn test_upper() {
        let result = eval_scalar_function("UPPER", &[Value::String("hello".into())], None).unwrap();
        assert_eq!(result, Value::String("HELLO".into()));
    }

    #[test]
    fn test_lower() {
        let result = eval_scalar_function("lower", &[Value::String("HELLO".into())], None).unwrap();
        assert_eq!(result, Value::String("hello".into()));
    }

    #[test]
    fn test_coalesce() {
        let result =
            eval_scalar_function("COALESCE", &[Value::Null, Value::Int(42)], None).unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_nullif_equal() {
        let result = eval_scalar_function("NULLIF", &[Value::Int(1), Value::Int(1)], None).unwrap();
        assert_eq!(result, Value::Null);
    }

    #[test]
    fn test_abs() {
        let result = eval_scalar_function("ABS", &[Value::Int(-5)], None).unwrap();
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_length() {
        let result =
            eval_scalar_function("LENGTH", &[Value::String("hello".into())], None).unwrap();
        assert_eq!(result, Value::Int(5));
    }

    #[test]
    fn test_unknown_function() {
        let result = eval_scalar_function("NONEXISTENT", &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_values_equal() {
        assert!(values_equal(&Value::Int(1), &Value::Int(1)));
        assert!(values_equal(&Value::Int(1), &Value::Float(1.0)));
        assert!(!values_equal(&Value::Int(1), &Value::Int(2)));
    }

    #[test]
    fn test_json_to_value_roundtrip() {
        let original = Value::Object(HashMap::from([("key".to_string(), Value::Int(42))]));
        let json = ast_value_to_json(&original);
        let back = serde_json_to_value(json);
        assert_eq!(original, back);
    }
}
