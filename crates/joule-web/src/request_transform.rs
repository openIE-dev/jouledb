//! Request transformation — header add/remove/rewrite, body transformation, query
//! parameter manipulation, path rewrite rules, and request enrichment.
//!
//! Replaces `express-transform`, `http-proxy-middleware`, and similar JS middleware
//! with a pure-Rust request transformation pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Request transformation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformError {
    /// Invalid transformation rule.
    InvalidRule(String),
    /// Body transformation failed.
    BodyTransformFailed(String),
    /// Header name is invalid.
    InvalidHeader(String),
    /// Path rewrite pattern failed to match.
    PathRewriteFailed { pattern: String, path: String },
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRule(msg) => write!(f, "invalid rule: {msg}"),
            Self::BodyTransformFailed(msg) => write!(f, "body transform failed: {msg}"),
            Self::InvalidHeader(name) => write!(f, "invalid header: {name}"),
            Self::PathRewriteFailed { pattern, path } => {
                write!(f, "path rewrite failed: pattern={pattern} path={path}")
            }
        }
    }
}

impl std::error::Error for TransformError {}

// ── Request ────────────────────────────────────────────────────

/// A mutable HTTP request representation for transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Query parameters.
    pub query: HashMap<String, String>,
    /// Headers (name -> values).
    pub headers: HashMap<String, Vec<String>>,
    /// Body bytes (optional).
    #[serde(skip)]
    pub body: Option<Vec<u8>>,
    /// Metadata for enrichment (key-value pairs added by transforms).
    pub metadata: HashMap<String, String>,
}

impl Request {
    /// Create a minimal request.
    pub fn new(method: &str, path: &str) -> Self {
        Self {
            method: method.to_string(),
            path: path.to_string(),
            query: HashMap::new(),
            headers: HashMap::new(),
            body: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the first value of a header.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.first().map(|s| s.as_str()))
    }

    /// Get all values of a header.
    pub fn header_all(&self, name: &str) -> &[String] {
        self.headers.get(name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get a query parameter.
    pub fn query_param(&self, name: &str) -> Option<&str> {
        self.query.get(name).map(|s| s.as_str())
    }

    /// Full request URI (path + query string).
    pub fn uri(&self) -> String {
        if self.query.is_empty() {
            self.path.clone()
        } else {
            let mut pairs: Vec<String> = self
                .query
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            pairs.sort(); // Deterministic output
            format!("{}?{}", self.path, pairs.join("&"))
        }
    }

    /// Body as UTF-8 string.
    pub fn body_str(&self) -> Option<&str> {
        self.body.as_ref().and_then(|b| std::str::from_utf8(b).ok())
    }
}

// ── Header Transform ───────────────────────────────────────────

/// Header transformation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeaderTransform {
    /// Add a header value (appends to existing).
    Add { name: String, value: String },
    /// Set a header (replaces all values).
    Set { name: String, value: String },
    /// Remove a header entirely.
    Remove { name: String },
    /// Rename a header.
    Rename { from: String, to: String },
    /// Copy a header to another name.
    Copy { from: String, to: String },
    /// Set a header only if it does not already exist.
    SetIfAbsent { name: String, value: String },
    /// Prefix all values of a header.
    PrefixValues { name: String, prefix: String },
}

/// Apply a single header transform.
pub fn apply_header_transform(
    headers: &mut HashMap<String, Vec<String>>,
    transform: &HeaderTransform,
) {
    match transform {
        HeaderTransform::Add { name, value } => {
            headers.entry(name.clone()).or_default().push(value.clone());
        }
        HeaderTransform::Set { name, value } => {
            headers.insert(name.clone(), vec![value.clone()]);
        }
        HeaderTransform::Remove { name } => {
            headers.remove(name);
        }
        HeaderTransform::Rename { from, to } => {
            if let Some(vals) = headers.remove(from) {
                headers.insert(to.clone(), vals);
            }
        }
        HeaderTransform::Copy { from, to } => {
            if let Some(vals) = headers.get(from).cloned() {
                headers.insert(to.clone(), vals);
            }
        }
        HeaderTransform::SetIfAbsent { name, value } => {
            headers
                .entry(name.clone())
                .or_insert_with(|| vec![value.clone()]);
        }
        HeaderTransform::PrefixValues { name, prefix } => {
            if let Some(vals) = headers.get_mut(name) {
                for v in vals.iter_mut() {
                    *v = format!("{prefix}{v}");
                }
            }
        }
    }
}

// ── Query Transform ────────────────────────────────────────────

/// Query parameter transformation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryTransform {
    /// Add or set a query parameter.
    Set { name: String, value: String },
    /// Remove a query parameter.
    Remove { name: String },
    /// Rename a query parameter.
    Rename { from: String, to: String },
    /// Set only if not already present.
    SetIfAbsent { name: String, value: String },
}

/// Apply a single query transform.
pub fn apply_query_transform(
    query: &mut HashMap<String, String>,
    transform: &QueryTransform,
) {
    match transform {
        QueryTransform::Set { name, value } => {
            query.insert(name.clone(), value.clone());
        }
        QueryTransform::Remove { name } => {
            query.remove(name);
        }
        QueryTransform::Rename { from, to } => {
            if let Some(val) = query.remove(from) {
                query.insert(to.clone(), val);
            }
        }
        QueryTransform::SetIfAbsent { name, value } => {
            query.entry(name.clone()).or_insert_with(|| value.clone());
        }
    }
}

// ── Path Rewrite ───────────────────────────────────────────────

/// Path rewrite rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathRewriteRule {
    /// Prefix to strip.
    pub strip_prefix: Option<String>,
    /// Prefix to add.
    pub add_prefix: Option<String>,
    /// If set, replace the entire path.
    pub replace_path: Option<String>,
    /// Replace a substring in the path.
    pub replace_segment: Option<(String, String)>,
}

impl PathRewriteRule {
    /// Apply this rule to a path, returning the new path.
    pub fn apply(&self, path: &str) -> String {
        if let Some(replacement) = &self.replace_path {
            return replacement.clone();
        }

        let mut result = path.to_string();

        if let Some(prefix) = &self.strip_prefix {
            if let Some(stripped) = result.strip_prefix(prefix.as_str()) {
                result = stripped.to_string();
                if !result.starts_with('/') {
                    result = format!("/{result}");
                }
            }
        }

        if let Some((from, to)) = &self.replace_segment {
            result = result.replace(from.as_str(), to);
        }

        if let Some(prefix) = &self.add_prefix {
            let clean = prefix.trim_end_matches('/');
            if result.starts_with('/') {
                result = format!("{clean}{result}");
            } else {
                result = format!("{clean}/{result}");
            }
        }

        result
    }
}

// ── Body Transform ─────────────────────────────────────────────

/// Body transformation action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BodyTransform {
    /// Replace the entire body.
    Replace(Vec<u8>),
    /// Remove the body.
    Clear,
    /// If body is JSON, set a field at top level.
    JsonSetField { field: String, value: serde_json::Value },
    /// If body is JSON, remove a top-level field.
    JsonRemoveField { field: String },
    /// Wrap the body in a JSON envelope: `{ "wrapper_key": <original_body> }`.
    JsonWrap { wrapper_key: String },
}

/// Apply a body transform to a request.
pub fn apply_body_transform(
    body: &mut Option<Vec<u8>>,
    transform: &BodyTransform,
) -> Result<(), TransformError> {
    match transform {
        BodyTransform::Replace(new_body) => {
            *body = Some(new_body.clone());
        }
        BodyTransform::Clear => {
            *body = None;
        }
        BodyTransform::JsonSetField { field, value } => {
            let current = body.as_deref().unwrap_or(b"{}");
            let mut obj: serde_json::Value =
                serde_json::from_slice(current).map_err(|e| {
                    TransformError::BodyTransformFailed(format!("invalid JSON: {e}"))
                })?;
            if let Some(map) = obj.as_object_mut() {
                map.insert(field.clone(), value.clone());
            } else {
                return Err(TransformError::BodyTransformFailed(
                    "body is not a JSON object".into(),
                ));
            }
            *body = Some(
                serde_json::to_vec(&obj)
                    .map_err(|e| TransformError::BodyTransformFailed(e.to_string()))?,
            );
        }
        BodyTransform::JsonRemoveField { field } => {
            if let Some(data) = body.as_deref() {
                let mut obj: serde_json::Value =
                    serde_json::from_slice(data).map_err(|e| {
                        TransformError::BodyTransformFailed(format!("invalid JSON: {e}"))
                    })?;
                if let Some(map) = obj.as_object_mut() {
                    map.remove(field);
                }
                *body = Some(
                    serde_json::to_vec(&obj)
                        .map_err(|e| TransformError::BodyTransformFailed(e.to_string()))?,
                );
            }
        }
        BodyTransform::JsonWrap { wrapper_key } => {
            if let Some(data) = body.as_deref() {
                let inner: serde_json::Value =
                    serde_json::from_slice(data).map_err(|e| {
                        TransformError::BodyTransformFailed(format!("invalid JSON: {e}"))
                    })?;
                let mut wrapper = serde_json::Map::new();
                wrapper.insert(wrapper_key.clone(), inner);
                *body = Some(
                    serde_json::to_vec(&serde_json::Value::Object(wrapper))
                        .map_err(|e| TransformError::BodyTransformFailed(e.to_string()))?,
                );
            }
        }
    }
    Ok(())
}

// ── Enrichment ─────────────────────────────────────────────────

/// Request enrichment action (adds metadata or derived headers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Enrichment {
    /// Add a metadata key-value pair.
    AddMetadata { key: String, value: String },
    /// Copy a header value into metadata.
    HeaderToMetadata { header: String, metadata_key: String },
    /// Copy a query param into metadata.
    QueryToMetadata { param: String, metadata_key: String },
    /// Set a header from metadata.
    MetadataToHeader { metadata_key: String, header: String },
}

/// Apply an enrichment to a request.
pub fn apply_enrichment(request: &mut Request, enrichment: &Enrichment) {
    match enrichment {
        Enrichment::AddMetadata { key, value } => {
            request.metadata.insert(key.clone(), value.clone());
        }
        Enrichment::HeaderToMetadata { header, metadata_key } => {
            if let Some(val) = request.header(header) {
                request.metadata.insert(metadata_key.clone(), val.to_string());
            }
        }
        Enrichment::QueryToMetadata { param, metadata_key } => {
            if let Some(val) = request.query_param(param) {
                request.metadata.insert(metadata_key.clone(), val.to_string());
            }
        }
        Enrichment::MetadataToHeader { metadata_key, header } => {
            if let Some(val) = request.metadata.get(metadata_key) {
                request.headers.insert(header.clone(), vec![val.clone()]);
            }
        }
    }
}

// ── Transform Pipeline ─────────────────────────────────────────

/// A step in the transformation pipeline.
#[derive(Debug, Clone)]
pub enum TransformStep {
    Header(HeaderTransform),
    Query(QueryTransform),
    PathRewrite(PathRewriteRule),
    Body(BodyTransform),
    Enrich(Enrichment),
    /// Set the HTTP method.
    SetMethod(String),
}

/// A request transformation pipeline.
#[derive(Debug, Clone)]
pub struct TransformPipeline {
    steps: Vec<TransformStep>,
}

impl TransformPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the pipeline.
    pub fn add(&mut self, step: TransformStep) {
        self.steps.push(step);
    }

    /// Number of steps in the pipeline.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Apply all steps to a request.
    pub fn apply(&self, request: &mut Request) -> Result<(), TransformError> {
        for step in &self.steps {
            match step {
                TransformStep::Header(t) => {
                    apply_header_transform(&mut request.headers, t);
                }
                TransformStep::Query(t) => {
                    apply_query_transform(&mut request.query, t);
                }
                TransformStep::PathRewrite(rule) => {
                    request.path = rule.apply(&request.path);
                }
                TransformStep::Body(t) => {
                    apply_body_transform(&mut request.body, t)?;
                }
                TransformStep::Enrich(e) => {
                    apply_enrichment(request, e);
                }
                TransformStep::SetMethod(m) => {
                    request.method = m.clone();
                }
            }
        }
        Ok(())
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req(method: &str, path: &str) -> Request {
        Request::new(method, path)
    }

    // ── Header transforms ──

    #[test]
    fn header_add() {
        let mut headers = HashMap::new();
        apply_header_transform(
            &mut headers,
            &HeaderTransform::Add { name: "X-Req".into(), value: "a".into() },
        );
        assert_eq!(headers.get("X-Req").unwrap(), &vec!["a".to_string()]);
    }

    #[test]
    fn header_add_appends() {
        let mut headers = HashMap::new();
        headers.insert("X-Tag".into(), vec!["a".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::Add { name: "X-Tag".into(), value: "b".into() },
        );
        assert_eq!(headers.get("X-Tag").unwrap().len(), 2);
    }

    #[test]
    fn header_set_replaces() {
        let mut headers = HashMap::new();
        headers.insert("X-Tag".into(), vec!["a".into(), "b".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::Set { name: "X-Tag".into(), value: "c".into() },
        );
        assert_eq!(headers.get("X-Tag").unwrap(), &vec!["c".to_string()]);
    }

    #[test]
    fn header_remove() {
        let mut headers = HashMap::new();
        headers.insert("X-Secret".into(), vec!["hide".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::Remove { name: "X-Secret".into() },
        );
        assert!(!headers.contains_key("X-Secret"));
    }

    #[test]
    fn header_rename() {
        let mut headers = HashMap::new();
        headers.insert("Old".into(), vec!["val".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::Rename { from: "Old".into(), to: "New".into() },
        );
        assert!(!headers.contains_key("Old"));
        assert_eq!(headers.get("New").unwrap(), &vec!["val".to_string()]);
    }

    #[test]
    fn header_copy() {
        let mut headers = HashMap::new();
        headers.insert("Source".into(), vec!["val".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::Copy { from: "Source".into(), to: "Dest".into() },
        );
        assert!(headers.contains_key("Source"));
        assert_eq!(headers.get("Dest").unwrap(), &vec!["val".to_string()]);
    }

    #[test]
    fn header_set_if_absent_new() {
        let mut headers = HashMap::new();
        apply_header_transform(
            &mut headers,
            &HeaderTransform::SetIfAbsent { name: "X-New".into(), value: "default".into() },
        );
        assert_eq!(headers.get("X-New").unwrap(), &vec!["default".to_string()]);
    }

    #[test]
    fn header_set_if_absent_existing() {
        let mut headers = HashMap::new();
        headers.insert("X-Existing".into(), vec!["original".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::SetIfAbsent {
                name: "X-Existing".into(),
                value: "ignored".into(),
            },
        );
        assert_eq!(
            headers.get("X-Existing").unwrap(),
            &vec!["original".to_string()]
        );
    }

    #[test]
    fn header_prefix_values() {
        let mut headers = HashMap::new();
        headers.insert("Auth".into(), vec!["token123".into()]);
        apply_header_transform(
            &mut headers,
            &HeaderTransform::PrefixValues { name: "Auth".into(), prefix: "Bearer ".into() },
        );
        assert_eq!(
            headers.get("Auth").unwrap(),
            &vec!["Bearer token123".to_string()]
        );
    }

    // ── Query transforms ──

    #[test]
    fn query_set() {
        let mut query = HashMap::new();
        apply_query_transform(
            &mut query,
            &QueryTransform::Set { name: "page".into(), value: "1".into() },
        );
        assert_eq!(query.get("page").unwrap(), "1");
    }

    #[test]
    fn query_remove() {
        let mut query = HashMap::new();
        query.insert("secret".into(), "hidden".into());
        apply_query_transform(
            &mut query,
            &QueryTransform::Remove { name: "secret".into() },
        );
        assert!(!query.contains_key("secret"));
    }

    #[test]
    fn query_rename() {
        let mut query = HashMap::new();
        query.insert("old_name".into(), "val".into());
        apply_query_transform(
            &mut query,
            &QueryTransform::Rename { from: "old_name".into(), to: "new_name".into() },
        );
        assert!(!query.contains_key("old_name"));
        assert_eq!(query.get("new_name").unwrap(), "val");
    }

    #[test]
    fn query_set_if_absent() {
        let mut query = HashMap::new();
        query.insert("existing".into(), "val".into());
        apply_query_transform(
            &mut query,
            &QueryTransform::SetIfAbsent { name: "existing".into(), value: "ignored".into() },
        );
        apply_query_transform(
            &mut query,
            &QueryTransform::SetIfAbsent { name: "new_key".into(), value: "default".into() },
        );
        assert_eq!(query.get("existing").unwrap(), "val");
        assert_eq!(query.get("new_key").unwrap(), "default");
    }

    // ── Path rewrite ──

    #[test]
    fn path_strip_prefix() {
        let rule = PathRewriteRule {
            strip_prefix: Some("/api/v1".into()),
            add_prefix: None,
            replace_path: None,
            replace_segment: None,
        };
        assert_eq!(rule.apply("/api/v1/users"), "/users");
    }

    #[test]
    fn path_add_prefix() {
        let rule = PathRewriteRule {
            strip_prefix: None,
            add_prefix: Some("/backend".into()),
            replace_path: None,
            replace_segment: None,
        };
        assert_eq!(rule.apply("/users"), "/backend/users");
    }

    #[test]
    fn path_replace() {
        let rule = PathRewriteRule {
            strip_prefix: None,
            add_prefix: None,
            replace_path: Some("/fixed".into()),
            replace_segment: None,
        };
        assert_eq!(rule.apply("/anything"), "/fixed");
    }

    #[test]
    fn path_replace_segment() {
        let rule = PathRewriteRule {
            strip_prefix: None,
            add_prefix: None,
            replace_path: None,
            replace_segment: Some(("v1".into(), "v2".into())),
        };
        assert_eq!(rule.apply("/api/v1/users"), "/api/v2/users");
    }

    // ── Body transforms ──

    #[test]
    fn body_replace() {
        let mut body: Option<Vec<u8>> = Some(b"old".to_vec());
        apply_body_transform(&mut body, &BodyTransform::Replace(b"new".to_vec())).unwrap();
        assert_eq!(body.unwrap(), b"new");
    }

    #[test]
    fn body_clear() {
        let mut body: Option<Vec<u8>> = Some(b"data".to_vec());
        apply_body_transform(&mut body, &BodyTransform::Clear).unwrap();
        assert!(body.is_none());
    }

    #[test]
    fn body_json_set_field() {
        let mut body: Option<Vec<u8>> = Some(br#"{"name":"alice"}"#.to_vec());
        apply_body_transform(
            &mut body,
            &BodyTransform::JsonSetField {
                field: "age".into(),
                value: serde_json::json!(30),
            },
        )
        .unwrap();

        let obj: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(obj["age"], 30);
        assert_eq!(obj["name"], "alice");
    }

    #[test]
    fn body_json_remove_field() {
        let mut body: Option<Vec<u8>> = Some(br#"{"name":"alice","secret":"hide"}"#.to_vec());
        apply_body_transform(
            &mut body,
            &BodyTransform::JsonRemoveField { field: "secret".into() },
        )
        .unwrap();

        let obj: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert!(obj.get("secret").is_none());
        assert_eq!(obj["name"], "alice");
    }

    #[test]
    fn body_json_wrap() {
        let mut body: Option<Vec<u8>> = Some(br#"{"x":1}"#.to_vec());
        apply_body_transform(
            &mut body,
            &BodyTransform::JsonWrap { wrapper_key: "data".into() },
        )
        .unwrap();

        let obj: serde_json::Value = serde_json::from_slice(body.as_ref().unwrap()).unwrap();
        assert_eq!(obj["data"]["x"], 1);
    }

    #[test]
    fn body_json_set_on_non_object_fails() {
        let mut body: Option<Vec<u8>> = Some(br#"[1,2,3]"#.to_vec());
        let err = apply_body_transform(
            &mut body,
            &BodyTransform::JsonSetField {
                field: "x".into(),
                value: serde_json::json!(1),
            },
        )
        .unwrap_err();
        assert!(matches!(err, TransformError::BodyTransformFailed(_)));
    }

    // ── Enrichment ──

    #[test]
    fn enrich_add_metadata() {
        let mut request = req("GET", "/test");
        apply_enrichment(
            &mut request,
            &Enrichment::AddMetadata { key: "env".into(), value: "prod".into() },
        );
        assert_eq!(request.metadata.get("env").unwrap(), "prod");
    }

    #[test]
    fn enrich_header_to_metadata() {
        let mut request = req("GET", "/test");
        request.headers.insert("X-User-Id".into(), vec!["42".into()]);
        apply_enrichment(
            &mut request,
            &Enrichment::HeaderToMetadata {
                header: "X-User-Id".into(),
                metadata_key: "user_id".into(),
            },
        );
        assert_eq!(request.metadata.get("user_id").unwrap(), "42");
    }

    #[test]
    fn enrich_query_to_metadata() {
        let mut request = req("GET", "/test");
        request.query.insert("token".into(), "abc".into());
        apply_enrichment(
            &mut request,
            &Enrichment::QueryToMetadata {
                param: "token".into(),
                metadata_key: "auth_token".into(),
            },
        );
        assert_eq!(request.metadata.get("auth_token").unwrap(), "abc");
    }

    #[test]
    fn enrich_metadata_to_header() {
        let mut request = req("GET", "/test");
        request.metadata.insert("trace_id".into(), "xyz".into());
        apply_enrichment(
            &mut request,
            &Enrichment::MetadataToHeader {
                metadata_key: "trace_id".into(),
                header: "X-Trace-Id".into(),
            },
        );
        assert_eq!(request.header("X-Trace-Id").unwrap(), "xyz");
    }

    // ── Pipeline ──

    #[test]
    fn pipeline_multiple_steps() {
        let mut pipeline = TransformPipeline::new();
        pipeline.add(TransformStep::Header(HeaderTransform::Set {
            name: "X-Gateway".into(),
            value: "joule".into(),
        }));
        pipeline.add(TransformStep::Query(QueryTransform::Set {
            name: "version".into(),
            value: "2".into(),
        }));
        pipeline.add(TransformStep::PathRewrite(PathRewriteRule {
            strip_prefix: Some("/api".into()),
            add_prefix: Some("/internal".into()),
            replace_path: None,
            replace_segment: None,
        }));

        let mut request = req("GET", "/api/users");
        pipeline.apply(&mut request).unwrap();

        assert_eq!(request.header("X-Gateway").unwrap(), "joule");
        assert_eq!(request.query.get("version").unwrap(), "2");
        assert_eq!(request.path, "/internal/users");
    }

    #[test]
    fn pipeline_set_method() {
        let mut pipeline = TransformPipeline::new();
        pipeline.add(TransformStep::SetMethod("POST".into()));

        let mut request = req("GET", "/submit");
        pipeline.apply(&mut request).unwrap();
        assert_eq!(request.method, "POST");
    }

    #[test]
    fn pipeline_empty() {
        let pipeline = TransformPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);

        let mut request = req("GET", "/test");
        pipeline.apply(&mut request).unwrap();
        assert_eq!(request.path, "/test");
    }

    // ── Request helpers ──

    #[test]
    fn request_uri_no_query() {
        let request = req("GET", "/path");
        assert_eq!(request.uri(), "/path");
    }

    #[test]
    fn request_uri_with_query() {
        let mut request = req("GET", "/path");
        request.query.insert("b".into(), "2".into());
        request.query.insert("a".into(), "1".into());
        // Sorted deterministically
        assert_eq!(request.uri(), "/path?a=1&b=2");
    }

    #[test]
    fn request_body_str() {
        let mut request = req("POST", "/data");
        request.body = Some(b"hello world".to_vec());
        assert_eq!(request.body_str().unwrap(), "hello world");
    }

    #[test]
    fn request_body_str_none() {
        let request = req("GET", "/test");
        assert!(request.body_str().is_none());
    }

    // ── Error display ──

    #[test]
    fn error_display_coverage() {
        let errs = vec![
            TransformError::InvalidRule("bad".into()),
            TransformError::BodyTransformFailed("fail".into()),
            TransformError::InvalidHeader("bad header".into()),
            TransformError::PathRewriteFailed {
                pattern: "p".into(),
                path: "/x".into(),
            },
        ];
        for e in &errs {
            assert!(!e.to_string().is_empty());
        }
    }
}
