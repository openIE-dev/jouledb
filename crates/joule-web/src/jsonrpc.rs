//! JSON-RPC 2.0 — request/response/notification/batch, error codes,
//! parameter types, method dispatch, ID generation.
//!
//! Pure-Rust replacement for jayson, jsonrpc-core, etc.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

// ── ID ────────────────────────────────────────────────────────────

/// A JSON-RPC request ID (number or string per spec).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RpcId {
    Number(i64),
    Str(String),
}

impl RpcId {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Number(n) => serde_json::Value::Number((*n).into()),
            Self::Str(s) => serde_json::Value::String(s.clone()),
        }
    }

    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        match v {
            serde_json::Value::Number(n) => n.as_i64().map(Self::Number),
            serde_json::Value::String(s) => Some(Self::Str(s.clone())),
            _ => None,
        }
    }
}

impl fmt::Display for RpcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Str(s) => write!(f, "{s}"),
        }
    }
}

/// Thread-safe auto-incrementing ID generator.
pub struct IdGenerator {
    next: AtomicU64,
}

impl IdGenerator {
    pub fn new() -> Self { Self { next: AtomicU64::new(1) } }

    pub fn next_id(&self) -> RpcId {
        RpcId::Number(self.next.fetch_add(1, Ordering::Relaxed) as i64)
    }
}

impl Default for IdGenerator {
    fn default() -> Self { Self::new() }
}

// ── Standard error codes ──────────────────────────────────────────

/// Standard JSON-RPC 2.0 error codes.
pub mod error_code {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
    /// Start of server-defined error range.
    pub const SERVER_ERROR_START: i64 = -32099;
    /// End of server-defined error range.
    pub const SERVER_ERROR_END: i64 = -32000;

    pub fn message(code: i64) -> &'static str {
        match code {
            PARSE_ERROR => "Parse error",
            INVALID_REQUEST => "Invalid Request",
            METHOD_NOT_FOUND => "Method not found",
            INVALID_PARAMS => "Invalid params",
            INTERNAL_ERROR => "Internal error",
            c if (SERVER_ERROR_START..=SERVER_ERROR_END).contains(&c) => "Server error",
            _ => "Unknown error",
        }
    }
}

// ── Error ─────────────────────────────────────────────────────────

/// A JSON-RPC error object.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

impl RpcError {
    pub fn new(code: i64, message: &str) -> Self {
        Self { code, message: message.into(), data: None }
    }

    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data); self
    }

    pub fn parse_error() -> Self {
        Self::new(error_code::PARSE_ERROR, error_code::message(error_code::PARSE_ERROR))
    }

    pub fn invalid_request() -> Self {
        Self::new(error_code::INVALID_REQUEST, error_code::message(error_code::INVALID_REQUEST))
    }

    pub fn method_not_found() -> Self {
        Self::new(error_code::METHOD_NOT_FOUND, error_code::message(error_code::METHOD_NOT_FOUND))
    }

    pub fn invalid_params(detail: &str) -> Self {
        Self::new(error_code::INVALID_PARAMS, detail)
    }

    pub fn internal_error() -> Self {
        Self::new(error_code::INTERNAL_ERROR, error_code::message(error_code::INTERNAL_ERROR))
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("code".into(), serde_json::Value::Number(self.code.into()));
        m.insert("message".into(), serde_json::Value::String(self.message.clone()));
        if let Some(ref d) = self.data {
            m.insert("data".into(), d.clone());
        }
        serde_json::Value::Object(m)
    }

    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        let code = v.get("code")?.as_i64()?;
        let message = v.get("message")?.as_str()?.to_string();
        let data = v.get("data").cloned();
        Some(Self { code, message, data })
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RPC error {}: {}", self.code, self.message)
    }
}

// ── Request ───────────────────────────────────────────────────────

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcRequest {
    pub id: RpcId,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

impl RpcRequest {
    pub fn new(id: RpcId, method: &str) -> Self {
        Self { id, method: method.into(), params: None }
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params); self
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("jsonrpc".into(), serde_json::Value::String("2.0".into()));
        m.insert("id".into(), self.id.to_json());
        m.insert("method".into(), serde_json::Value::String(self.method.clone()));
        if let Some(ref p) = self.params {
            m.insert("params".into(), p.clone());
        }
        serde_json::Value::Object(m)
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self, RpcError> {
        let jsonrpc = v.get("jsonrpc").and_then(|v| v.as_str());
        if jsonrpc != Some("2.0") {
            return Err(RpcError::invalid_request());
        }
        let id = v.get("id")
            .and_then(RpcId::from_json)
            .ok_or_else(RpcError::invalid_request)?;
        let method = v.get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(RpcError::invalid_request)?
            .to_string();
        let params = v.get("params").cloned();
        Ok(Self { id, method, params })
    }
}

// ── Notification ──────────────────────────────────────────────────

/// A JSON-RPC 2.0 notification (request with no id).
#[derive(Debug, Clone, PartialEq)]
pub struct RpcNotification {
    pub method: String,
    pub params: Option<serde_json::Value>,
}

impl RpcNotification {
    pub fn new(method: &str) -> Self {
        Self { method: method.into(), params: None }
    }

    pub fn with_params(mut self, params: serde_json::Value) -> Self {
        self.params = Some(params); self
    }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("jsonrpc".into(), serde_json::Value::String("2.0".into()));
        m.insert("method".into(), serde_json::Value::String(self.method.clone()));
        if let Some(ref p) = self.params {
            m.insert("params".into(), p.clone());
        }
        serde_json::Value::Object(m)
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self, RpcError> {
        let jsonrpc = v.get("jsonrpc").and_then(|v| v.as_str());
        if jsonrpc != Some("2.0") {
            return Err(RpcError::invalid_request());
        }
        if v.get("id").is_some() {
            return Err(RpcError::invalid_request());
        }
        let method = v.get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(RpcError::invalid_request)?
            .to_string();
        let params = v.get("params").cloned();
        Ok(Self { method, params })
    }
}

// ── Response ──────────────────────────────────────────────────────

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcResponse {
    pub id: RpcId,
    pub result: Result<serde_json::Value, RpcError>,
}

impl RpcResponse {
    pub fn success(id: RpcId, result: serde_json::Value) -> Self {
        Self { id, result: Ok(result) }
    }

    pub fn error(id: RpcId, error: RpcError) -> Self {
        Self { id, result: Err(error) }
    }

    pub fn is_success(&self) -> bool { self.result.is_ok() }
    pub fn is_error(&self) -> bool { self.result.is_err() }

    pub fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("jsonrpc".into(), serde_json::Value::String("2.0".into()));
        m.insert("id".into(), self.id.to_json());
        match &self.result {
            Ok(val) => { m.insert("result".into(), val.clone()); }
            Err(err) => { m.insert("error".into(), err.to_json()); }
        }
        serde_json::Value::Object(m)
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self, RpcError> {
        let id = v.get("id")
            .and_then(RpcId::from_json)
            .ok_or_else(RpcError::invalid_request)?;
        if let Some(result) = v.get("result") {
            return Ok(Self::success(id, result.clone()));
        }
        if let Some(error) = v.get("error") {
            let rpc_err = RpcError::from_json(error)
                .ok_or_else(RpcError::invalid_request)?;
            return Ok(Self::error(id, rpc_err));
        }
        Err(RpcError::invalid_request())
    }
}

// ── Message (unified) ─────────────────────────────────────────────

/// A parsed incoming JSON-RPC message.
#[derive(Debug, Clone)]
pub enum RpcMessage {
    Request(RpcRequest),
    Notification(RpcNotification),
    Response(RpcResponse),
}

/// Parse a single JSON value as a JSON-RPC message.
pub fn parse_message(v: &serde_json::Value) -> Result<RpcMessage, RpcError> {
    if !v.is_object() {
        return Err(RpcError::invalid_request());
    }
    // If it has "method", it's either a request or notification
    if v.get("method").is_some() {
        if v.get("id").is_some() {
            return Ok(RpcMessage::Request(RpcRequest::from_json(v)?));
        }
        return Ok(RpcMessage::Notification(RpcNotification::from_json(v)?));
    }
    // If it has "result" or "error", it's a response
    if v.get("result").is_some() || v.get("error").is_some() {
        return Ok(RpcMessage::Response(RpcResponse::from_json(v)?));
    }
    Err(RpcError::invalid_request())
}

// ── Batch ─────────────────────────────────────────────────────────

/// Parse a batch JSON-RPC message.
pub fn parse_batch(v: &serde_json::Value) -> Result<Vec<Result<RpcMessage, RpcError>>, RpcError> {
    let arr = v.as_array().ok_or_else(RpcError::invalid_request)?;
    if arr.is_empty() {
        return Err(RpcError::invalid_request());
    }
    Ok(arr.iter().map(parse_message).collect())
}

/// Serialize a batch of responses to a JSON array.
pub fn batch_response(responses: &[RpcResponse]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = responses.iter().map(|r| r.to_json()).collect();
    serde_json::Value::Array(arr)
}

// ── Parse from string ─────────────────────────────────────────────

/// Parse a raw JSON string into messages. Handles single or batch.
pub fn parse_raw(input: &str) -> Result<Vec<Result<RpcMessage, RpcError>>, RpcError> {
    let v: serde_json::Value = serde_json::from_str(input)
        .map_err(|_| RpcError::parse_error())?;
    if v.is_array() {
        parse_batch(&v)
    } else {
        let msg = parse_message(&v)?;
        Ok(vec![Ok(msg)])
    }
}

// ── Dispatcher ────────────────────────────────────────────────────

/// A simple method dispatcher.
pub struct Dispatcher {
    handlers: HashMap<String, Box<dyn Fn(Option<serde_json::Value>) -> Result<serde_json::Value, RpcError> + Send + Sync>>,
}

impl Dispatcher {
    pub fn new() -> Self {
        Self { handlers: HashMap::new() }
    }

    /// Register a handler for a method.
    pub fn register<F>(&mut self, method: &str, handler: F)
    where
        F: Fn(Option<serde_json::Value>) -> Result<serde_json::Value, RpcError> + Send + Sync + 'static,
    {
        self.handlers.insert(method.into(), Box::new(handler));
    }

    /// Check if a method is registered.
    pub fn has_method(&self, method: &str) -> bool {
        self.handlers.contains_key(method)
    }

    /// List all registered methods.
    pub fn methods(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }

    /// Handle a single request. Returns a response.
    pub fn handle_request(&self, req: &RpcRequest) -> RpcResponse {
        match self.handlers.get(&req.method) {
            Some(handler) => {
                match handler(req.params.clone()) {
                    Ok(result) => RpcResponse::success(req.id.clone(), result),
                    Err(err) => RpcResponse::error(req.id.clone(), err),
                }
            }
            None => RpcResponse::error(req.id.clone(), RpcError::method_not_found()),
        }
    }

    /// Handle a batch of messages. Returns responses (notifications produce none).
    pub fn handle_batch(&self, messages: &[Result<RpcMessage, RpcError>]) -> Vec<RpcResponse> {
        let mut responses = Vec::new();
        for msg in messages {
            match msg {
                Ok(RpcMessage::Request(req)) => {
                    responses.push(self.handle_request(req));
                }
                Ok(RpcMessage::Notification(_)) => {
                    // Notifications don't get responses
                }
                Ok(RpcMessage::Response(_)) => {
                    // Responses from client — ignore in server context
                }
                Err(err) => {
                    responses.push(RpcResponse::error(
                        RpcId::Number(0),
                        err.clone(),
                    ));
                }
            }
        }
        responses
    }
}

impl Default for Dispatcher {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rpc_id_number() {
        let id = RpcId::Number(42);
        assert_eq!(id.to_json(), serde_json::json!(42));
        assert_eq!(id.to_string(), "42");
    }

    #[test]
    fn rpc_id_string() {
        let id = RpcId::Str("abc-123".into());
        assert_eq!(id.to_json(), serde_json::json!("abc-123"));
        assert_eq!(id.to_string(), "abc-123");
    }

    #[test]
    fn rpc_id_from_json() {
        assert_eq!(RpcId::from_json(&serde_json::json!(5)), Some(RpcId::Number(5)));
        assert_eq!(RpcId::from_json(&serde_json::json!("x")), Some(RpcId::Str("x".into())));
        assert_eq!(RpcId::from_json(&serde_json::json!(true)), None);
    }

    #[test]
    fn id_generator() {
        let generator = IdGenerator::new();
        assert_eq!(generator.next_id(), RpcId::Number(1));
        assert_eq!(generator.next_id(), RpcId::Number(2));
        assert_eq!(generator.next_id(), RpcId::Number(3));
    }

    #[test]
    fn error_standard_codes() {
        let e = RpcError::parse_error();
        assert_eq!(e.code, -32700);
        assert_eq!(e.message, "Parse error");

        let e = RpcError::method_not_found();
        assert_eq!(e.code, -32601);

        let e = RpcError::invalid_params("bad type");
        assert_eq!(e.code, -32602);
        assert_eq!(e.message, "bad type");
    }

    #[test]
    fn error_with_data() {
        let e = RpcError::internal_error()
            .with_data(serde_json::json!({"detail": "stack overflow"}));
        let j = e.to_json();
        assert_eq!(j["code"], -32603);
        assert_eq!(j["data"]["detail"], "stack overflow");
    }

    #[test]
    fn error_roundtrip() {
        let original = RpcError::new(-32000, "Custom error")
            .with_data(serde_json::json!("extra"));
        let j = original.to_json();
        let parsed = RpcError::from_json(&j).unwrap();
        assert_eq!(parsed.code, -32000);
        assert_eq!(parsed.message, "Custom error");
        assert_eq!(parsed.data, Some(serde_json::json!("extra")));
    }

    #[test]
    fn error_display() {
        let e = RpcError::parse_error();
        assert_eq!(e.to_string(), "RPC error -32700: Parse error");
    }

    #[test]
    fn request_basic() {
        let req = RpcRequest::new(RpcId::Number(1), "subtract");
        let j = req.to_json();
        assert_eq!(j["jsonrpc"], "2.0");
        assert_eq!(j["id"], 1);
        assert_eq!(j["method"], "subtract");
        assert!(j.get("params").is_none());
    }

    #[test]
    fn request_with_positional_params() {
        let req = RpcRequest::new(RpcId::Number(1), "subtract")
            .with_params(serde_json::json!([42, 23]));
        let j = req.to_json();
        assert_eq!(j["params"][0], 42);
        assert_eq!(j["params"][1], 23);
    }

    #[test]
    fn request_with_named_params() {
        let req = RpcRequest::new(RpcId::Number(3), "subtract")
            .with_params(serde_json::json!({"minuend": 42, "subtrahend": 23}));
        let j = req.to_json();
        assert_eq!(j["params"]["minuend"], 42);
    }

    #[test]
    fn request_roundtrip() {
        let original = RpcRequest::new(RpcId::Number(7), "getUser")
            .with_params(serde_json::json!({"id": 42}));
        let j = original.to_json();
        let parsed = RpcRequest::from_json(&j).unwrap();
        assert_eq!(parsed.id, RpcId::Number(7));
        assert_eq!(parsed.method, "getUser");
        assert_eq!(parsed.params, Some(serde_json::json!({"id": 42})));
    }

    #[test]
    fn request_invalid_version() {
        let j = serde_json::json!({"jsonrpc": "1.0", "id": 1, "method": "test"});
        assert!(RpcRequest::from_json(&j).is_err());
    }

    #[test]
    fn notification_basic() {
        let notif = RpcNotification::new("update");
        let j = notif.to_json();
        assert_eq!(j["jsonrpc"], "2.0");
        assert_eq!(j["method"], "update");
        assert!(j.get("id").is_none());
    }

    #[test]
    fn notification_with_params() {
        let notif = RpcNotification::new("update")
            .with_params(serde_json::json!([1, 2, 3]));
        let j = notif.to_json();
        assert_eq!(j["params"][0], 1);
    }

    #[test]
    fn notification_roundtrip() {
        let original = RpcNotification::new("notify")
            .with_params(serde_json::json!({"event": "ready"}));
        let j = original.to_json();
        let parsed = RpcNotification::from_json(&j).unwrap();
        assert_eq!(parsed.method, "notify");
    }

    #[test]
    fn notification_rejects_id() {
        let j = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "test"});
        assert!(RpcNotification::from_json(&j).is_err());
    }

    #[test]
    fn response_success() {
        let resp = RpcResponse::success(RpcId::Number(1), serde_json::json!(19));
        assert!(resp.is_success());
        let j = resp.to_json();
        assert_eq!(j["result"], 19);
        assert!(j.get("error").is_none());
    }

    #[test]
    fn response_error() {
        let resp = RpcResponse::error(
            RpcId::Number(1),
            RpcError::method_not_found(),
        );
        assert!(resp.is_error());
        let j = resp.to_json();
        assert_eq!(j["error"]["code"], -32601);
        assert!(j.get("result").is_none());
    }

    #[test]
    fn response_roundtrip_success() {
        let original = RpcResponse::success(RpcId::Str("abc".into()), serde_json::json!({"ok": true}));
        let j = original.to_json();
        let parsed = RpcResponse::from_json(&j).unwrap();
        assert_eq!(parsed.id, RpcId::Str("abc".into()));
        assert!(parsed.is_success());
    }

    #[test]
    fn response_roundtrip_error() {
        let original = RpcResponse::error(
            RpcId::Number(5),
            RpcError::invalid_params("missing field"),
        );
        let j = original.to_json();
        let parsed = RpcResponse::from_json(&j).unwrap();
        assert!(parsed.is_error());
        if let Err(ref e) = parsed.result {
            assert_eq!(e.code, -32602);
        }
    }

    #[test]
    fn parse_message_request() {
        let j = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "test"});
        let msg = parse_message(&j).unwrap();
        assert!(matches!(msg, RpcMessage::Request(_)));
    }

    #[test]
    fn parse_message_notification() {
        let j = serde_json::json!({"jsonrpc": "2.0", "method": "update"});
        let msg = parse_message(&j).unwrap();
        assert!(matches!(msg, RpcMessage::Notification(_)));
    }

    #[test]
    fn parse_message_response() {
        let j = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": 42});
        let msg = parse_message(&j).unwrap();
        assert!(matches!(msg, RpcMessage::Response(_)));
    }

    #[test]
    fn parse_message_invalid() {
        let j = serde_json::json!(42);
        assert!(parse_message(&j).is_err());
    }

    #[test]
    fn parse_batch_basic() {
        let j = serde_json::json!([
            {"jsonrpc": "2.0", "id": 1, "method": "add", "params": [1, 2]},
            {"jsonrpc": "2.0", "method": "notify"},
            {"jsonrpc": "2.0", "id": 2, "method": "subtract", "params": [5, 3]},
        ]);
        let messages = parse_batch(&j).unwrap();
        assert_eq!(messages.len(), 3);
        assert!(messages[0].is_ok());
        assert!(messages[1].is_ok());
    }

    #[test]
    fn parse_batch_empty_rejected() {
        let j = serde_json::json!([]);
        assert!(parse_batch(&j).is_err());
    }

    #[test]
    fn batch_response_serialization() {
        let responses = vec![
            RpcResponse::success(RpcId::Number(1), serde_json::json!(3)),
            RpcResponse::success(RpcId::Number(2), serde_json::json!(2)),
        ];
        let j = batch_response(&responses);
        let arr = j.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["result"], 3);
    }

    #[test]
    fn parse_raw_single() {
        let input = r#"{"jsonrpc": "2.0", "id": 1, "method": "test"}"#;
        let messages = parse_raw(input).unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn parse_raw_batch() {
        let input = r#"[{"jsonrpc": "2.0", "id": 1, "method": "a"},{"jsonrpc": "2.0", "id": 2, "method": "b"}]"#;
        let messages = parse_raw(input).unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn parse_raw_invalid_json() {
        assert!(parse_raw("not json").is_err());
    }

    #[test]
    fn dispatcher_basic() {
        let mut dispatch = Dispatcher::new();
        dispatch.register("add", |params| {
            let arr = params.unwrap();
            let a = arr[0].as_i64().unwrap();
            let b = arr[1].as_i64().unwrap();
            Ok(serde_json::json!(a + b))
        });
        assert!(dispatch.has_method("add"));
        assert!(!dispatch.has_method("sub"));

        let req = RpcRequest::new(RpcId::Number(1), "add")
            .with_params(serde_json::json!([3, 4]));
        let resp = dispatch.handle_request(&req);
        assert!(resp.is_success());
        assert_eq!(resp.to_json()["result"], 7);
    }

    #[test]
    fn dispatcher_method_not_found() {
        let dispatch = Dispatcher::new();
        let req = RpcRequest::new(RpcId::Number(1), "missing");
        let resp = dispatch.handle_request(&req);
        assert!(resp.is_error());
        assert_eq!(resp.to_json()["error"]["code"], -32601);
    }

    #[test]
    fn dispatcher_handler_error() {
        let mut dispatch = Dispatcher::new();
        dispatch.register("fail", |_| {
            Err(RpcError::invalid_params("bad input"))
        });
        let req = RpcRequest::new(RpcId::Number(1), "fail");
        let resp = dispatch.handle_request(&req);
        assert!(resp.is_error());
        assert_eq!(resp.to_json()["error"]["code"], -32602);
    }

    #[test]
    fn dispatcher_batch() {
        let mut dispatch = Dispatcher::new();
        dispatch.register("echo", |params| {
            Ok(params.unwrap_or(serde_json::Value::Null))
        });

        let messages = vec![
            Ok(RpcMessage::Request(RpcRequest::new(RpcId::Number(1), "echo")
                .with_params(serde_json::json!("hello")))),
            Ok(RpcMessage::Notification(RpcNotification::new("ping"))),
            Ok(RpcMessage::Request(RpcRequest::new(RpcId::Number(2), "missing"))),
        ];

        let responses = dispatch.handle_batch(&messages);
        // Notification produces no response
        assert_eq!(responses.len(), 2);
        assert!(responses[0].is_success());
        assert!(responses[1].is_error());
    }

    #[test]
    fn dispatcher_methods_list() {
        let mut dispatch = Dispatcher::new();
        dispatch.register("alpha", |_| Ok(serde_json::Value::Null));
        dispatch.register("beta", |_| Ok(serde_json::Value::Null));
        let mut methods = dispatch.methods();
        methods.sort();
        assert_eq!(methods, vec!["alpha", "beta"]);
    }

    #[test]
    fn error_code_message() {
        assert_eq!(error_code::message(-32700), "Parse error");
        assert_eq!(error_code::message(-32601), "Method not found");
        assert_eq!(error_code::message(-32050), "Server error");
        assert_eq!(error_code::message(999), "Unknown error");
    }

    #[test]
    fn spec_example_subtract() {
        // From JSON-RPC 2.0 spec
        let req_json = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "subtract",
            "params": [42, 23],
            "id": 1
        });
        let req = RpcRequest::from_json(&req_json).unwrap();
        assert_eq!(req.method, "subtract");

        let resp = RpcResponse::success(req.id.clone(), serde_json::json!(19));
        let j = resp.to_json();
        assert_eq!(j["jsonrpc"], "2.0");
        assert_eq!(j["result"], 19);
        assert_eq!(j["id"], 1);
    }

    #[test]
    fn spec_example_notification() {
        let notif_json = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "update",
            "params": [1, 2, 3, 4, 5]
        });
        let notif = RpcNotification::from_json(&notif_json).unwrap();
        assert_eq!(notif.method, "update");
    }
}
