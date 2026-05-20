//! Column classification via name patterns and value heuristics.
//!
//! Determines the [`SemanticType`] of a column by examining its name and
//! a sample of its values.

use crate::hint::SemanticType;

/// Classify a column by name pattern matching and value inspection.
pub fn classify_column(name: &str, values: &[&serde_json::Value]) -> SemanticType {
    let lower = name.to_lowercase();

    // 1. Name-based heuristics (highest priority).
    if let Some(st) = classify_by_name(&lower) {
        return st;
    }

    // 2. Value-based heuristics.
    classify_by_values(values)
}

/// Name pattern matching.
fn classify_by_name(lower: &str) -> Option<SemanticType> {
    // Timestamps / dates
    if matches!(
        lower.as_ref(),
        "timestamp"
            | "created_at"
            | "updated_at"
            | "deleted_at"
            | "datetime"
            | "ts"
            | "event_time"
            | "event_timestamp"
            | "log_time"
            | "recorded_at"
            | "occurred_at"
            | "start_time"
            | "end_time"
            | "modified_at"
    ) {
        return Some(SemanticType::Timestamp);
    }
    if lower.ends_with("_at") || lower.ends_with("_timestamp") || lower.ends_with("_time") {
        return Some(SemanticType::Timestamp);
    }
    if matches!(lower.as_ref(), "date" | "day" | "month" | "year") || lower.ends_with("_date") {
        return Some(SemanticType::Date);
    }

    // Identifiers
    if matches!(lower.as_ref(), "id" | "uuid" | "guid" | "pk" | "key")
        || lower.ends_with("_id")
        || lower.ends_with("_uuid")
        || lower.ends_with("_key")
    {
        return Some(SemanticType::Identifier);
    }

    // Geographic
    if matches!(
        lower.as_ref(),
        "lat" | "latitude" | "geo_lat" | "start_lat" | "end_lat"
    ) || lower.ends_with("_lat")
        || lower.ends_with("_latitude")
    {
        return Some(SemanticType::GeoLatitude);
    }
    if matches!(
        lower.as_ref(),
        "lon"
            | "lng"
            | "longitude"
            | "geo_lon"
            | "geo_lng"
            | "start_lon"
            | "end_lon"
            | "start_lng"
            | "end_lng"
    ) || lower.ends_with("_lon")
        || lower.ends_with("_lng")
        || lower.ends_with("_longitude")
    {
        return Some(SemanticType::GeoLongitude);
    }

    // Currency
    if matches!(
        lower.as_ref(),
        "price"
            | "cost"
            | "revenue"
            | "amount"
            | "total"
            | "subtotal"
            | "tax"
            | "discount"
            | "salary"
            | "budget"
            | "balance"
            | "fee"
            | "payment"
    ) || lower.ends_with("_price")
        || lower.ends_with("_cost")
        || lower.ends_with("_amount")
        || lower.ends_with("_revenue")
    {
        return Some(SemanticType::Currency);
    }

    // Percentage
    if matches!(
        lower.as_ref(),
        "percent" | "percentage" | "pct" | "ratio" | "rate"
    ) || lower.ends_with("_pct")
        || lower.ends_with("_percent")
        || lower.ends_with("_rate")
        || lower.ends_with("_ratio")
    {
        return Some(SemanticType::Percentage);
    }

    // Boolean
    if matches!(
        lower.as_ref(),
        "active"
            | "enabled"
            | "visible"
            | "deleted"
            | "archived"
            | "published"
            | "verified"
            | "approved"
    ) || lower.starts_with("is_")
        || lower.starts_with("has_")
        || lower.starts_with("can_")
    {
        return Some(SemanticType::Boolean);
    }

    // Aggregation results → numeric continuous
    if matches!(
        lower.as_ref(),
        "count" | "sum" | "avg" | "average" | "min" | "max" | "mean" | "median" | "stddev"
    ) || lower.starts_with("count_")
        || lower.starts_with("sum_")
        || lower.starts_with("avg_")
        || lower.starts_with("total_")
    {
        return Some(SemanticType::NumericContinuous);
    }

    // Common categorical
    if matches!(
        lower.as_ref(),
        "status"
            | "state"
            | "type"
            | "category"
            | "kind"
            | "group"
            | "class"
            | "tier"
            | "level"
            | "region"
            | "country"
            | "city"
            | "department"
            | "role"
            | "gender"
            | "color"
            | "colour"
            | "brand"
            | "vendor"
            | "channel"
            | "source"
            | "platform"
    ) {
        return Some(SemanticType::Categorical);
    }

    // Text-heavy
    if matches!(
        lower.as_ref(),
        "name"
            | "title"
            | "label"
            | "description"
            | "comment"
            | "note"
            | "notes"
            | "message"
            | "body"
            | "content"
            | "text"
            | "bio"
            | "summary"
            | "email"
            | "url"
            | "address"
            | "path"
    ) {
        return Some(SemanticType::Text);
    }

    None
}

/// Value-based classification when name heuristics fail.
fn classify_by_values(values: &[&serde_json::Value]) -> SemanticType {
    if values.is_empty() {
        return SemanticType::Unknown;
    }

    let non_null: Vec<&&serde_json::Value> = values.iter().filter(|v| !v.is_null()).collect();
    if non_null.is_empty() {
        return SemanticType::Unknown;
    }

    // All booleans
    if non_null.iter().all(|v| v.is_boolean()) {
        return SemanticType::Boolean;
    }

    // All numbers
    if non_null.iter().all(|v| v.is_number()) {
        let all_integer = non_null.iter().all(|v| {
            v.as_f64()
                .map(|f| f.fract() == 0.0 && f.abs() < i64::MAX as f64)
                .unwrap_or(false)
        });

        if all_integer {
            // Fewer than ~20 distinct values in sample → discrete
            let mut distinct = std::collections::HashSet::new();
            for v in &non_null {
                if let Some(n) = v.as_i64() {
                    distinct.insert(n);
                }
            }
            if distinct.len() <= 20 && non_null.len() >= 5 {
                return SemanticType::NumericDiscrete;
            }
            return SemanticType::NumericContinuous;
        }
        return SemanticType::NumericContinuous;
    }

    // All strings — try to detect patterns
    if non_null.iter().all(|v| v.is_string()) {
        let strings: Vec<&str> = non_null.iter().filter_map(|v| v.as_str()).collect();

        // Timestamp patterns
        if strings.iter().all(|s| looks_like_timestamp(s)) {
            return SemanticType::Timestamp;
        }

        // Date patterns
        if strings.iter().all(|s| looks_like_date(s)) {
            return SemanticType::Date;
        }

        // UUID/identifier patterns
        if strings.iter().all(|s| looks_like_uuid(s)) {
            return SemanticType::Identifier;
        }

        // Low cardinality strings → categorical
        let mut distinct = std::collections::HashSet::new();
        for s in &strings {
            distinct.insert(*s);
        }
        if distinct.len() <= 20 && strings.len() >= 5 {
            return SemanticType::Categorical;
        }

        // Long strings → text
        let avg_len: f64 =
            strings.iter().map(|s| s.len() as f64).sum::<f64>() / strings.len() as f64;
        if avg_len > 50.0 {
            return SemanticType::Text;
        }

        return SemanticType::Categorical;
    }

    SemanticType::Unknown
}

/// Check if a string looks like a timestamp (ISO 8601 with time component).
fn looks_like_timestamp(s: &str) -> bool {
    // e.g. "2024-01-15T10:30:00Z" or "2024-01-15 10:30:00"
    let len = s.len();
    if len < 19 {
        return false;
    }
    let bytes = s.as_bytes();
    // YYYY-MM-DD and contains T or space then digits
    bytes.len() >= 19
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && (bytes[10] == b'T' || bytes[10] == b' ')
        && bytes[13] == b':'
}

/// Check if a string looks like a date (YYYY-MM-DD).
fn looks_like_date(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() >= 10
        && bytes.len() <= 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes[0..4].iter().all(|b| b.is_ascii_digit())
        && bytes[5..7].iter().all(|b| b.is_ascii_digit())
        && bytes[8..10].iter().all(|b| b.is_ascii_digit())
}

/// Check if a string looks like a UUID.
fn looks_like_uuid(s: &str) -> bool {
    // 8-4-4-4-12 hex pattern
    let bytes = s.as_bytes();
    bytes.len() == 36
        && bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| matches!(i, 8 | 13 | 18 | 23) || b.is_ascii_hexdigit())
}
