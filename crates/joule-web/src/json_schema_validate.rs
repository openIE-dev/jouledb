// SPDX-License-Identifier: MIT
//! JSON Schema validation (draft 2020-12 subset).
//!
//! Supports: type, properties, required, enum, const, additionalProperties,
//! minimum/maximum, minLength/maxLength, pattern, minItems/maxItems,
//! allOf/anyOf/oneOf/not, $ref resolution, format hints, and full error paths.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

/// A single validation error with its JSON-Pointer path inside the instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

/// Result of validating one instance against one schema.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

// ── Schema store (for $ref) ─────────────────────────────────────────────────

/// Registry that holds named schemas so `$ref` can resolve them.
#[derive(Debug, Default)]
pub struct SchemaStore {
    schemas: HashMap<String, Value>,
}

impl SchemaStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, uri: &str, schema: Value) {
        self.schemas.insert(uri.to_string(), schema);
    }

    pub fn get(&self, uri: &str) -> Option<&Value> {
        self.schemas.get(uri)
    }
}

// ── Validator ───────────────────────────────────────────────────────────────

pub fn validate(schema: &Value, instance: &Value) -> ValidationResult {
    validate_with_store(schema, instance, &SchemaStore::new())
}

pub fn validate_with_store(
    schema: &Value,
    instance: &Value,
    store: &SchemaStore,
) -> ValidationResult {
    let mut errors = Vec::new();
    validate_inner(schema, instance, store, String::new(), &mut errors);
    ValidationResult { errors }
}

fn push_err(errors: &mut Vec<ValidationError>, path: &str, msg: impl Into<String>) {
    errors.push(ValidationError {
        path: path.to_string(),
        message: msg.into(),
    });
}

fn validate_inner(
    schema: &Value,
    instance: &Value,
    store: &SchemaStore,
    path: String,
    errors: &mut Vec<ValidationError>,
) {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => {
            // `true` accepts everything, `false` rejects everything.
            if schema.as_bool() == Some(false) {
                push_err(errors, &path, "schema is false — nothing validates");
            }
            return;
        }
    };

    // ── $ref ────────────────────────────────────────────────────────────
    if let Some(Value::String(uri)) = obj.get("$ref") {
        if let Some(target) = store.get(uri) {
            let target = target.clone();
            validate_inner(&target, instance, store, path.clone(), errors);
        } else {
            push_err(errors, &path, format!("unresolved $ref: {uri}"));
        }
        return; // $ref replaces the schema
    }

    // ── type ────────────────────────────────────────────────────────────
    if let Some(ty) = obj.get("type") {
        let types: Vec<&str> = if let Some(s) = ty.as_str() {
            vec![s]
        } else if let Some(arr) = ty.as_array() {
            arr.iter().filter_map(|v| v.as_str()).collect()
        } else {
            vec![]
        };
        if !types.is_empty() && !types.iter().any(|t| type_matches(t, instance)) {
            push_err(
                errors,
                &path,
                format!("expected type {:?}, got {}", types, json_type_name(instance)),
            );
        }
    }

    // ── enum ────────────────────────────────────────────────────────────
    if let Some(Value::Array(variants)) = obj.get("enum") {
        if !variants.iter().any(|v| v == instance) {
            push_err(errors, &path, format!("value not in enum: {instance}"));
        }
    }

    // ── const ───────────────────────────────────────────────────────────
    if let Some(c) = obj.get("const") {
        if c != instance {
            push_err(errors, &path, format!("expected const {c}, got {instance}"));
        }
    }

    // ── numeric ─────────────────────────────────────────────────────────
    if let Some(n) = instance.as_f64() {
        if let Some(min) = obj.get("minimum").and_then(|v| v.as_f64()) {
            if n < min {
                push_err(errors, &path, format!("value {n} < minimum {min}"));
            }
        }
        if let Some(max) = obj.get("maximum").and_then(|v| v.as_f64()) {
            if n > max {
                push_err(errors, &path, format!("value {n} > maximum {max}"));
            }
        }
        if let Some(ex_min) = obj.get("exclusiveMinimum").and_then(|v| v.as_f64()) {
            if n <= ex_min {
                push_err(errors, &path, format!("value {n} <= exclusiveMinimum {ex_min}"));
            }
        }
        if let Some(ex_max) = obj.get("exclusiveMaximum").and_then(|v| v.as_f64()) {
            if n >= ex_max {
                push_err(errors, &path, format!("value {n} >= exclusiveMaximum {ex_max}"));
            }
        }
        if let Some(mult) = obj.get("multipleOf").and_then(|v| v.as_f64()) {
            if mult != 0.0 && (n / mult).fract().abs() > 1e-9 {
                push_err(errors, &path, format!("value {n} not multipleOf {mult}"));
            }
        }
    }

    // ── string ──────────────────────────────────────────────────────────
    if let Some(s) = instance.as_str() {
        let char_len = s.chars().count();
        if let Some(ml) = obj.get("minLength").and_then(|v| v.as_u64()) {
            if (char_len as u64) < ml {
                push_err(errors, &path, format!("string length {char_len} < minLength {ml}"));
            }
        }
        if let Some(ml) = obj.get("maxLength").and_then(|v| v.as_u64()) {
            if (char_len as u64) > ml {
                push_err(errors, &path, format!("string length {char_len} > maxLength {ml}"));
            }
        }
        if let Some(Value::String(pat)) = obj.get("pattern") {
            if !simple_regex_match(pat, s) {
                push_err(errors, &path, format!("string does not match pattern \"{pat}\""));
            }
        }
        if let Some(Value::String(fmt)) = obj.get("format") {
            if !check_format(fmt, s) {
                push_err(errors, &path, format!("string does not match format \"{fmt}\""));
            }
        }
    }

    // ── array ───────────────────────────────────────────────────────────
    if let Some(arr) = instance.as_array() {
        if let Some(mi) = obj.get("minItems").and_then(|v| v.as_u64()) {
            if (arr.len() as u64) < mi {
                push_err(errors, &path, format!("array length {} < minItems {mi}", arr.len()));
            }
        }
        if let Some(mi) = obj.get("maxItems").and_then(|v| v.as_u64()) {
            if (arr.len() as u64) > mi {
                push_err(errors, &path, format!("array length {} > maxItems {mi}", arr.len()));
            }
        }
        if let Some(items_schema) = obj.get("items") {
            for (i, item) in arr.iter().enumerate() {
                let p = format!("{path}/{i}");
                validate_inner(items_schema, item, store, p, errors);
            }
        }
        if obj.get("uniqueItems") == Some(&Value::Bool(true)) {
            for i in 0..arr.len() {
                for j in (i + 1)..arr.len() {
                    if arr[i] == arr[j] {
                        push_err(errors, &path, format!("duplicate items at [{i}] and [{j}]"));
                    }
                }
            }
        }
    }

    // ── object ──────────────────────────────────────────────────────────
    if let Some(map) = instance.as_object() {
        // properties
        if let Some(Value::Object(props)) = obj.get("properties") {
            for (key, sub_schema) in props {
                if let Some(val) = map.get(key) {
                    let p = format!("{path}/{}", escape_pointer(key));
                    validate_inner(sub_schema, val, store, p, errors);
                }
            }
        }
        // required
        if let Some(Value::Array(req)) = obj.get("required") {
            for r in req {
                if let Some(name) = r.as_str() {
                    if !map.contains_key(name) {
                        push_err(errors, &path, format!("missing required property \"{name}\""));
                    }
                }
            }
        }
        // additionalProperties
        if let Some(ap) = obj.get("additionalProperties") {
            let allowed_keys: Vec<&String> = obj
                .get("properties")
                .and_then(|v| v.as_object())
                .map(|p| p.keys().collect())
                .unwrap_or_default();
            for key in map.keys() {
                if !allowed_keys.contains(&key) {
                    let p = format!("{path}/{}", escape_pointer(key));
                    if ap.as_bool() == Some(false) {
                        push_err(errors, &p, "additional property not allowed".to_string());
                    } else if ap.is_object() {
                        validate_inner(ap, &map[key], store, p, errors);
                    }
                }
            }
        }
        // minProperties / maxProperties
        if let Some(mp) = obj.get("minProperties").and_then(|v| v.as_u64()) {
            if (map.len() as u64) < mp {
                push_err(errors, &path, format!("object has {} properties < minProperties {mp}", map.len()));
            }
        }
        if let Some(mp) = obj.get("maxProperties").and_then(|v| v.as_u64()) {
            if (map.len() as u64) > mp {
                push_err(errors, &path, format!("object has {} properties > maxProperties {mp}", map.len()));
            }
        }
    }

    // ── allOf ───────────────────────────────────────────────────────────
    if let Some(Value::Array(schemas)) = obj.get("allOf") {
        for s in schemas {
            validate_inner(s, instance, store, path.clone(), errors);
        }
    }

    // ── anyOf ───────────────────────────────────────────────────────────
    if let Some(Value::Array(schemas)) = obj.get("anyOf") {
        let any_ok = schemas.iter().any(|s| {
            let mut tmp = Vec::new();
            validate_inner(s, instance, store, path.clone(), &mut tmp);
            tmp.is_empty()
        });
        if !any_ok {
            push_err(errors, &path, "value does not match any of anyOf schemas");
        }
    }

    // ── oneOf ───────────────────────────────────────────────────────────
    if let Some(Value::Array(schemas)) = obj.get("oneOf") {
        let count = schemas
            .iter()
            .filter(|s| {
                let mut tmp = Vec::new();
                validate_inner(s, instance, store, path.clone(), &mut tmp);
                tmp.is_empty()
            })
            .count();
        if count != 1 {
            push_err(errors, &path, format!("expected exactly 1 oneOf match, got {count}"));
        }
    }

    // ── not ─────────────────────────────────────────────────────────────
    if let Some(not_schema) = obj.get("not") {
        let mut tmp = Vec::new();
        validate_inner(not_schema, instance, store, path.clone(), &mut tmp);
        if tmp.is_empty() {
            push_err(errors, &path, "value should NOT validate against 'not' schema");
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn type_matches(ty: &str, val: &Value) -> bool {
    match ty {
        "null" => val.is_null(),
        "boolean" => val.is_boolean(),
        "integer" => val.as_f64().map(|n| n.fract() == 0.0).unwrap_or(false) && val.is_number(),
        "number" => val.is_number(),
        "string" => val.is_string(),
        "array" => val.is_array(),
        "object" => val.is_object(),
        _ => false,
    }
}

fn json_type_name(val: &Value) -> &'static str {
    match val {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn escape_pointer(seg: &str) -> String {
    seg.replace('~', "~0").replace('/', "~1")
}

/// Minimal regex-like matching supporting `^`, `$`, `.`, `*`, `+`, `?`, `\d`, `\w`.
fn simple_regex_match(pattern: &str, text: &str) -> bool {
    let anchored_start = pattern.starts_with('^');
    let anchored_end = pattern.ends_with('$') && !pattern.ends_with("\\$");
    let pat = if anchored_start { &pattern[1..] } else { pattern };
    let pat = if anchored_end { &pat[..pat.len() - 1] } else { pat };

    if pat.is_empty() {
        return if anchored_start && anchored_end {
            text.is_empty()
        } else {
            true
        };
    }

    // Simple substring / prefix / suffix / exact for literal patterns
    let is_literal = !pat.contains(|c: char| matches!(c, '.' | '*' | '+' | '?' | '[' | '\\'));
    if is_literal {
        return match (anchored_start, anchored_end) {
            (true, true) => text == pat,
            (true, false) => text.starts_with(pat),
            (false, true) => text.ends_with(pat),
            (false, false) => text.contains(pat),
        };
    }
    // Fallback: character-class-aware prefix match (good enough for schema patterns)
    try_match_at(pat, text, anchored_start, anchored_end)
}

fn try_match_at(pat: &str, text: &str, anchored_start: bool, anchored_end: bool) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let start_positions = if anchored_start { 0..1 } else { 0..chars.len().max(1) };
    for start in start_positions {
        if let Some(end) = match_from(pat, &chars, start) {
            if !anchored_end || end == chars.len() {
                return true;
            }
        }
    }
    false
}

fn match_from(pat: &str, chars: &[char], mut pos: usize) -> Option<usize> {
    let pbytes = pat.as_bytes();
    let mut pi = 0;
    while pi < pbytes.len() {
        // Peek at quantifier
        let (class_end, quantifier) = {
            let ce = class_len(pbytes, pi);
            if pi + ce < pbytes.len() {
                match pbytes[pi + ce] {
                    b'*' => (ce, b'*'),
                    b'+' => (ce, b'+'),
                    b'?' => (ce, b'?'),
                    _ => (ce, 0u8),
                }
            } else {
                (ce, 0u8)
            }
        };
        let class_pat = &pat[pi..pi + class_end];
        match quantifier {
            b'*' => {
                pi += class_end + 1;
                // greedy
                let saved = pos;
                while pos < chars.len() && char_class_match(class_pat, chars[pos]) {
                    pos += 1;
                }
                while pos >= saved {
                    if let Some(r) = match_from(&pat[pi..], chars, pos) {
                        return Some(r);
                    }
                    if pos == saved { break; }
                    pos -= 1;
                }
                return None;
            }
            b'+' => {
                pi += class_end + 1;
                if pos >= chars.len() || !char_class_match(class_pat, chars[pos]) {
                    return None;
                }
                pos += 1;
                let saved = pos;
                while pos < chars.len() && char_class_match(class_pat, chars[pos]) {
                    pos += 1;
                }
                while pos >= saved {
                    if let Some(r) = match_from(&pat[pi..], chars, pos) {
                        return Some(r);
                    }
                    if pos == saved { break; }
                    pos -= 1;
                }
                return None;
            }
            b'?' => {
                pi += class_end + 1;
                // try with
                if pos < chars.len() && char_class_match(class_pat, chars[pos]) {
                    if let Some(r) = match_from(&pat[pi..], chars, pos + 1) {
                        return Some(r);
                    }
                }
                // try without
                return match_from(&pat[pi..], chars, pos);
            }
            _ => {
                if pos >= chars.len() || !char_class_match(class_pat, chars[pos]) {
                    return None;
                }
                pos += 1;
                pi += class_end;
            }
        }
    }
    Some(pos)
}

fn class_len(pat: &[u8], i: usize) -> usize {
    if i < pat.len() && pat[i] == b'\\' { 2 } else { 1 }
}

fn char_class_match(class: &str, ch: char) -> bool {
    let b = class.as_bytes();
    if b.len() == 2 && b[0] == b'\\' {
        match b[1] {
            b'd' => ch.is_ascii_digit(),
            b'w' => ch.is_ascii_alphanumeric() || ch == '_',
            b's' => ch.is_ascii_whitespace(),
            _ => ch == b[1] as char,
        }
    } else if b.len() == 1 {
        if b[0] == b'.' { true } else { ch == b[0] as char }
    } else {
        false
    }
}

fn check_format(fmt: &str, s: &str) -> bool {
    match fmt {
        "email" => {
            let parts: Vec<&str> = s.splitn(2, '@').collect();
            parts.len() == 2 && !parts[0].is_empty() && parts[1].contains('.')
        }
        "date" => {
            // YYYY-MM-DD
            let p: Vec<&str> = s.split('-').collect();
            p.len() == 3
                && p[0].len() == 4
                && p[1].len() == 2
                && p[2].len() == 2
                && p.iter().all(|x| x.chars().all(|c| c.is_ascii_digit()))
        }
        "uri" | "uri-reference" => s.contains(':') || s.starts_with('/'),
        "ipv4" => {
            let parts: Vec<&str> = s.split('.').collect();
            parts.len() == 4 && parts.iter().all(|p| p.parse::<u8>().is_ok())
        }
        "ipv6" => s.contains(':') && s.len() >= 2,
        "uuid" => {
            let stripped = s.replace('-', "");
            stripped.len() == 32 && stripped.chars().all(|c| c.is_ascii_hexdigit())
        }
        _ => true, // unknown formats pass
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ok(schema: Value, instance: Value) {
        let r = validate(&schema, &instance);
        assert!(r.is_valid(), "expected valid, got errors: {:?}", r.errors);
    }

    fn fail(schema: Value, instance: Value) -> Vec<ValidationError> {
        let r = validate(&schema, &instance);
        assert!(!r.is_valid(), "expected invalid");
        r.errors
    }

    #[test]
    fn type_string() {
        ok(json!({"type": "string"}), json!("hello"));
        fail(json!({"type": "string"}), json!(42));
    }

    #[test]
    fn type_integer() {
        ok(json!({"type": "integer"}), json!(5));
        fail(json!({"type": "integer"}), json!(5.5));
    }

    #[test]
    fn type_number() {
        ok(json!({"type": "number"}), json!(3.14));
        fail(json!({"type": "number"}), json!("nope"));
    }

    #[test]
    fn type_boolean() {
        ok(json!({"type": "boolean"}), json!(true));
        fail(json!({"type": "boolean"}), json!(1));
    }

    #[test]
    fn type_null() {
        ok(json!({"type": "null"}), json!(null));
        fail(json!({"type": "null"}), json!(0));
    }

    #[test]
    fn type_array() {
        ok(json!({"type": "array"}), json!([1,2]));
        fail(json!({"type": "array"}), json!({}));
    }

    #[test]
    fn type_object() {
        ok(json!({"type": "object"}), json!({}));
        fail(json!({"type": "object"}), json!([]));
    }

    #[test]
    fn type_union() {
        ok(json!({"type": ["string", "null"]}), json!(null));
        ok(json!({"type": ["string", "null"]}), json!("hi"));
        fail(json!({"type": ["string", "null"]}), json!(42));
    }

    #[test]
    fn enum_keyword() {
        ok(json!({"enum": [1, "two", null]}), json!("two"));
        fail(json!({"enum": [1, "two", null]}), json!(3));
    }

    #[test]
    fn const_keyword() {
        ok(json!({"const": 42}), json!(42));
        fail(json!({"const": 42}), json!(43));
    }

    #[test]
    fn minimum_maximum() {
        ok(json!({"type": "number", "minimum": 0, "maximum": 100}), json!(50));
        fail(json!({"minimum": 10}), json!(5));
        fail(json!({"maximum": 10}), json!(15));
    }

    #[test]
    fn exclusive_min_max() {
        fail(json!({"exclusiveMinimum": 5}), json!(5));
        ok(json!({"exclusiveMinimum": 5}), json!(6));
        fail(json!({"exclusiveMaximum": 10}), json!(10));
        ok(json!({"exclusiveMaximum": 10}), json!(9));
    }

    #[test]
    fn multiple_of() {
        ok(json!({"multipleOf": 3}), json!(9));
        fail(json!({"multipleOf": 3}), json!(10));
    }

    #[test]
    fn string_length() {
        ok(json!({"minLength": 2, "maxLength": 5}), json!("abc"));
        fail(json!({"minLength": 2}), json!("a"));
        fail(json!({"maxLength": 3}), json!("abcd"));
    }

    #[test]
    fn string_pattern() {
        ok(json!({"pattern": "^\\d+$"}), json!("1234"));
        fail(json!({"pattern": "^\\d+$"}), json!("12ab"));
    }

    #[test]
    fn format_email() {
        ok(json!({"format": "email"}), json!("a@b.com"));
        fail(json!({"format": "email"}), json!("nope"));
    }

    #[test]
    fn format_date() {
        ok(json!({"format": "date"}), json!("2025-01-15"));
        fail(json!({"format": "date"}), json!("25-1-5"));
    }

    #[test]
    fn format_ipv4() {
        ok(json!({"format": "ipv4"}), json!("192.168.1.1"));
        fail(json!({"format": "ipv4"}), json!("999.0.0.0"));
    }

    #[test]
    fn format_uuid() {
        ok(json!({"format": "uuid"}), json!("550e8400-e29b-41d4-a716-446655440000"));
        fail(json!({"format": "uuid"}), json!("not-a-uuid"));
    }

    #[test]
    fn array_items() {
        ok(
            json!({"type": "array", "items": {"type": "number"}}),
            json!([1, 2, 3]),
        );
        let errs = fail(
            json!({"type": "array", "items": {"type": "number"}}),
            json!([1, "oops", 3]),
        );
        assert!(errs[0].path.contains("/1"));
    }

    #[test]
    fn array_min_max_items() {
        fail(json!({"minItems": 2}), json!([1]));
        fail(json!({"maxItems": 1}), json!([1, 2]));
    }

    #[test]
    fn unique_items() {
        ok(json!({"uniqueItems": true}), json!([1, 2, 3]));
        fail(json!({"uniqueItems": true}), json!([1, 2, 1]));
    }

    #[test]
    fn properties_and_required() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        });
        ok(schema.clone(), json!({"name": "Alice", "age": 30}));
        fail(schema.clone(), json!({"age": 30}));
        let errs = fail(schema, json!({"name": 42}));
        assert!(errs[0].path.contains("name"));
    }

    #[test]
    fn additional_properties_false() {
        let schema = json!({
            "properties": {"a": {"type": "number"}},
            "additionalProperties": false
        });
        ok(schema.clone(), json!({"a": 1}));
        fail(schema, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn additional_properties_schema() {
        let schema = json!({
            "properties": {"a": {"type": "number"}},
            "additionalProperties": {"type": "string"}
        });
        ok(schema.clone(), json!({"a": 1, "b": "ok"}));
        fail(schema, json!({"a": 1, "b": 99}));
    }

    #[test]
    fn min_max_properties() {
        fail(json!({"minProperties": 2}), json!({"a": 1}));
        fail(json!({"maxProperties": 1}), json!({"a": 1, "b": 2}));
    }

    #[test]
    fn all_of() {
        let schema = json!({
            "allOf": [
                {"type": "number"},
                {"minimum": 5}
            ]
        });
        ok(schema.clone(), json!(10));
        fail(schema, json!(3));
    }

    #[test]
    fn any_of() {
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"type": "number"}
            ]
        });
        ok(schema.clone(), json!("hi"));
        ok(schema.clone(), json!(42));
        fail(schema, json!(true));
    }

    #[test]
    fn one_of() {
        let schema = json!({
            "oneOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        });
        // "hello" matches type:string only -> 1 match -> ok
        ok(schema.clone(), json!("hello"));
        // 42 matches type:integer only -> 1 match -> ok
        ok(schema.clone(), json!(42));
        // true matches neither -> 0 matches -> fail
        fail(schema.clone(), json!(true));
        // Test overlapping schemas where both match
        let overlapping = json!({
            "oneOf": [
                {"type": "number"},
                {"type": "integer"}
            ]
        });
        // 5 is both a number and an integer -> 2 matches -> fail
        fail(overlapping, json!(5));
    }

    #[test]
    fn not_keyword() {
        ok(json!({"not": {"type": "string"}}), json!(42));
        fail(json!({"not": {"type": "string"}}), json!("hi"));
    }

    #[test]
    fn ref_resolution() {
        let mut store = SchemaStore::new();
        store.insert("defs/positive", json!({"type": "integer", "minimum": 1}));
        let schema = json!({"$ref": "defs/positive"});
        let r = validate_with_store(&schema, &json!(5), &store);
        assert!(r.is_valid());
        let r = validate_with_store(&schema, &json!(-1), &store);
        assert!(!r.is_valid());
    }

    #[test]
    fn ref_unresolved() {
        let r = validate(&json!({"$ref": "nope"}), &json!(1));
        assert!(!r.is_valid());
        assert!(r.errors[0].message.contains("unresolved"));
    }

    #[test]
    fn nested_path_tracking() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        });
        let errs = fail(schema, json!({"items": ["ok", 42, "ok"]}));
        assert_eq!(errs[0].path, "/items/1");
    }

    #[test]
    fn boolean_schema() {
        ok(json!(true), json!("anything"));
        fail(json!(false), json!("anything"));
    }

    #[test]
    fn complex_nested_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "email": {"type": "string", "format": "email"}
                        },
                        "required": ["name", "email"]
                    }
                }
            },
            "required": ["users"]
        });
        ok(schema.clone(), json!({"users": [{"name": "A", "email": "a@b.com"}]}));
        let errs = fail(schema, json!({"users": [{"name": ""}]}));
        assert!(errs.len() >= 2); // missing email + minLength
    }

    #[test]
    fn error_display() {
        let e = ValidationError { path: "/a/b".into(), message: "bad".into() };
        assert_eq!(format!("{e}"), "/a/b: bad");
        let e = ValidationError { path: String::new(), message: "root".into() };
        assert_eq!(format!("{e}"), "root");
    }
}
