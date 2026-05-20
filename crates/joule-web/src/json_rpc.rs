//! JSON-RPC 2.0 protocol.
//!
//! Replaces `jsonrpc-core` with a pure-Rust JSON-RPC 2.0 implementation.
//! Supports request/response/notification/error objects, batch requests,
//! standard error codes (-32700 parse error, -32600 invalid request,
//! -32601 method not found, -32602 invalid params, -32603 internal error),
//! and ID matching.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// JSON-RPC error codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Invalid JSON was received.
    ParseError,
    /// The JSON sent is not a valid Request object.
    InvalidRequest,
    /// The method does not exist / is not available.
    MethodNotFound,
    /// Invalid method parameter(s).
    InvalidParams,
    /// Internal JSON-RPC error.
    InternalError,
    /// Server-defined error.
    ServerError(i64),
}

impl ErrorCode {
    pub fn code(self) -> i64 {
        match self {
            Self::ParseError => -32700,
            Self::InvalidRequest => -32600,
            Self::MethodNotFound => -32601,
            Self::InvalidParams => -32602,
            Self::InternalError => -32603,
            Self::ServerError(c) => c,
        }
    }

    pub fn from_code(code: i64) -> Self {
        match code {
            -32700 => Self::ParseError,
            -32600 => Self::InvalidRequest,
            -32601 => Self::MethodNotFound,
            -32602 => Self::InvalidParams,
            -32603 => Self::InternalError,
            c => Self::ServerError(c),
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::ParseError => "Parse error",
            Self::InvalidRequest => "Invalid Request",
            Self::MethodNotFound => "Method not found",
            Self::InvalidParams => "Invalid params",
            Self::InternalError => "Internal error",
            Self::ServerError(_) => "Server error",
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.message(), self.code())
    }
}

// ── ID ──────────────────────────────────────────────────────

/// JSON-RPC request/response ID.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Id {
    Number(i64),
    String(String),
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::String(s) => write!(f, "\"{s}\""),
        }
    }
}

// ── RPC Error ───────────────────────────────────────────────

/// A JSON-RPC error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn new(error_code: ErrorCode) -> Self {
        Self {
            code: error_code.code(),
            message: error_code.message().to_string(),
            data: None,
        }
    }

    pub fn with_message(error_code: ErrorCode, message: &str) -> Self {
        Self {
            code: error_code.code(),
            message: message.to_string(),
            data: None,
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Get the error code enum.
    pub fn error_code(&self) -> ErrorCode {
        ErrorCode::from_code(self.code)
    }
}

impl fmt::Display for RpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for RpcError {}

// ── Request ─────────────────────────────────────────────────

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Id>,
}

impl Request {
    /// Create a request with a numeric ID.
    pub fn new(method: &str, params: Option<Value>, id: i64) -> Self {
        Self {
            method: method.to_string(),
            params,
            id: Some(Id::Number(id)),
        }
    }

    /// Create a notification (no ID).
    pub fn notification(method: &str, params: Option<Value>) -> Self {
        Self {
            method: method.to_string(),
            params,
            id: None,
        }
    }

    /// Whether this is a notification (no ID).
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }

    /// Serialize to JSON value.
    pub fn to_json(&self) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("jsonrpc".into(), Value::String("2.0".into()));
        obj.insert("method".into(), Value::String(self.method.clone()));
        if let Some(params) = &self.params {
            obj.insert("params".into(), params.clone());
        }
        if let Some(id) = &self.id {
            obj.insert("id".into(), match id {
                Id::Number(n) => Value::Number((*n).into()),
                Id::String(s) => Value::String(s.clone()),
            });
        }
        Value::Object(obj)
    }

    /// Parse from JSON value.
    pub fn from_json(val: &Value) -> Result<Self, RpcError> {
        let obj = val.as_object().ok_or_else(|| RpcError::new(ErrorCode::InvalidRequest))?;

        // Check jsonrpc version
        match obj.get("jsonrpc").and_then(|v| v.as_str()) {
            Some("2.0") => {}
            _ => return Err(RpcError::with_message(ErrorCode::InvalidRequest, "jsonrpc must be \"2.0\"")),
        }

        let method = obj
            .get("method")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RpcError::with_message(ErrorCode::InvalidRequest, "missing method"))?
            .to_string();

        let params = obj.get("params").cloned();
        let id = obj.get("id").map(|v| {
            if let Some(n) = v.as_i64() {
                Id::Number(n)
            } else if let Some(s) = v.as_str() {
                Id::String(s.to_string())
            } else {
                Id::Number(0) // fallback
            }
        });

        Ok(Self { method, params, id })
    }
}

// ── Response ────────────────────────────────────────────────

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub id: Option<Id>,
    pub result: Result<Value, RpcError>,
}

impl Response {
    /// Create a success response.
    pub fn success(id: Id, result: Value) -> Self {
        Self {
            id: Some(id),
            result: Ok(result),
        }
    }

    /// Create an error response.
    pub fn error(id: Option<Id>, error: RpcError) -> Self {
        Self {
            id,
            result: Err(error),
        }
    }

    /// Serialize to JSON value.
    pub fn to_json(&self) -> Value {
        let mut obj = serde_json::Map::new();
        obj.insert("jsonrpc".into(), Value::String("2.0".into()));

        match &self.result {
            Ok(val) => {
                obj.insert("result".into(), val.clone());
            }
            Err(err) => {
                let error_val = serde_json::to_value(err).unwrap_or(Value::Null);
                obj.insert("error".into(), error_val);
            }
        }

        if let Some(id) = &self.id {
            obj.insert("id".into(), match id {
                Id::Number(n) => Value::Number((*n).into()),
                Id::String(s) => Value::String(s.clone()),
            });
        } else {
            obj.insert("id".into(), Value::Null);
        }

        Value::Object(obj)
    }

    /// Parse from JSON value.
    pub fn from_json(val: &Value) -> Result<Self, RpcError> {
        let obj = val.as_object().ok_or_else(|| RpcError::new(ErrorCode::InvalidRequest))?;

        let id = obj.get("id").and_then(|v| {
            if v.is_null() {
                None
            } else if let Some(n) = v.as_i64() {
                Some(Id::Number(n))
            } else {
                v.as_str().map(|s| Id::String(s.to_string()))
            }
        });

        if let Some(error_val) = obj.get("error") {
            let rpc_error: RpcError = serde_json::from_value(error_val.clone())
                .map_err(|_| RpcError::new(ErrorCode::InternalError))?;
            return Ok(Self { id, result: Err(rpc_error) });
        }

        let result = obj
            .get("result")
            .cloned()
            .unwrap_or(Value::Null);

        Ok(Self { id, result: Ok(result) })
    }
}

// ── Batch ───────────────────────────────────────────────────

/// A batch of JSON-RPC requests.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub requests: Vec<Request>,
}

impl BatchRequest {
    pub fn new() -> Self {
        Self { requests: Vec::new() }
    }

    pub fn add(mut self, req: Request) -> Self {
        self.requests.push(req);
        self
    }

    pub fn to_json(&self) -> Value {
        Value::Array(self.requests.iter().map(|r| r.to_json()).collect())
    }

    pub fn from_json(val: &Value) -> Result<Self, RpcError> {
        let arr = val.as_array().ok_or_else(|| RpcError::new(ErrorCode::InvalidRequest))?;
        if arr.is_empty() {
            return Err(RpcError::with_message(ErrorCode::InvalidRequest, "empty batch"));
        }
        let mut requests = Vec::new();
        for item in arr {
            requests.push(Request::from_json(item)?);
        }
        Ok(Self { requests })
    }
}

/// A batch of JSON-RPC responses.
#[derive(Debug, Clone)]
pub struct BatchResponse {
    pub responses: Vec<Response>,
}

impl BatchResponse {
    pub fn new() -> Self {
        Self { responses: Vec::new() }
    }

    pub fn add(mut self, resp: Response) -> Self {
        self.responses.push(resp);
        self
    }

    pub fn to_json(&self) -> Value {
        Value::Array(self.responses.iter().map(|r| r.to_json()).collect())
    }

    /// Find response by ID.
    pub fn find_by_id(&self, id: &Id) -> Option<&Response> {
        self.responses.iter().find(|r| r.id.as_ref() == Some(id))
    }
}

// ── Convenience Helpers ─────────────────────────────────────

/// Parse a JSON-RPC message (request, batch, or response).
pub fn parse_message(json: &str) -> Result<MessageKind, RpcError> {
    let val: Value = serde_json::from_str(json)
        .map_err(|_| RpcError::new(ErrorCode::ParseError))?;

    if val.is_array() {
        let batch = BatchRequest::from_json(&val)?;
        return Ok(MessageKind::BatchRequest(batch));
    }

    let obj = val.as_object().ok_or_else(|| RpcError::new(ErrorCode::InvalidRequest))?;

    // If it has "method", it's a request/notification
    if obj.contains_key("method") {
        let req = Request::from_json(&val)?;
        return Ok(MessageKind::Request(req));
    }

    // Otherwise it's a response
    let resp = Response::from_json(&val)?;
    Ok(MessageKind::Response(resp))
}

/// Classified JSON-RPC message.
#[derive(Debug, Clone)]
pub enum MessageKind {
    Request(Request),
    BatchRequest(BatchRequest),
    Response(Response),
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_roundtrip() {
        let req = Request::new("subtract", Some(json!([42, 23])), 1);
        let j = req.to_json();
        let parsed = Request::from_json(&j).unwrap();
        assert_eq!(parsed.method, "subtract");
        assert_eq!(parsed.id, Some(Id::Number(1)));
        assert_eq!(parsed.params, Some(json!([42, 23])));
    }

    #[test]
    fn notification() {
        let req = Request::notification("update", Some(json!([1, 2, 3])));
        assert!(req.is_notification());
        let j = req.to_json();
        let parsed = Request::from_json(&j).unwrap();
        assert!(parsed.is_notification());
    }

    #[test]
    fn success_response() {
        let resp = Response::success(Id::Number(1), json!(19));
        let j = resp.to_json();
        let parsed = Response::from_json(&j).unwrap();
        assert_eq!(parsed.id, Some(Id::Number(1)));
        assert_eq!(parsed.result.unwrap(), json!(19));
    }

    #[test]
    fn error_response() {
        let err = RpcError::new(ErrorCode::MethodNotFound);
        let resp = Response::error(Some(Id::Number(1)), err);
        let j = resp.to_json();
        let parsed = Response::from_json(&j).unwrap();
        assert!(parsed.result.is_err());
        let rpc_err = parsed.result.unwrap_err();
        assert_eq!(rpc_err.code, -32601);
    }

    #[test]
    fn error_codes() {
        assert_eq!(ErrorCode::ParseError.code(), -32700);
        assert_eq!(ErrorCode::InvalidRequest.code(), -32600);
        assert_eq!(ErrorCode::MethodNotFound.code(), -32601);
        assert_eq!(ErrorCode::InvalidParams.code(), -32602);
        assert_eq!(ErrorCode::InternalError.code(), -32603);
        assert_eq!(ErrorCode::from_code(-32700), ErrorCode::ParseError);
        assert_eq!(ErrorCode::from_code(-999), ErrorCode::ServerError(-999));
    }

    #[test]
    fn error_with_data() {
        let err = RpcError::new(ErrorCode::InternalError)
            .with_data(json!({"details": "stack overflow"}));
        assert!(err.data.is_some());
    }

    #[test]
    fn batch_request() {
        let batch = BatchRequest::new()
            .add(Request::new("add", Some(json!([1, 2])), 1))
            .add(Request::new("sub", Some(json!([5, 3])), 2))
            .add(Request::notification("log", None));
        let j = batch.to_json();
        let parsed = BatchRequest::from_json(&j).unwrap();
        assert_eq!(parsed.requests.len(), 3);
        assert!(parsed.requests[2].is_notification());
    }

    #[test]
    fn batch_response_find_by_id() {
        let batch = BatchResponse::new()
            .add(Response::success(Id::Number(1), json!(3)))
            .add(Response::success(Id::Number(2), json!(2)));
        let found = batch.find_by_id(&Id::Number(2)).unwrap();
        assert_eq!(found.result.as_ref().unwrap(), &json!(2));
        assert!(batch.find_by_id(&Id::Number(99)).is_none());
    }

    #[test]
    fn parse_message_request() {
        let json_str = r#"{"jsonrpc":"2.0","method":"test","id":1}"#;
        match parse_message(json_str).unwrap() {
            MessageKind::Request(req) => assert_eq!(req.method, "test"),
            _ => panic!("expected request"),
        }
    }

    #[test]
    fn parse_message_batch() {
        let json_str = r#"[{"jsonrpc":"2.0","method":"a","id":1},{"jsonrpc":"2.0","method":"b","id":2}]"#;
        match parse_message(json_str).unwrap() {
            MessageKind::BatchRequest(b) => assert_eq!(b.requests.len(), 2),
            _ => panic!("expected batch"),
        }
    }

    #[test]
    fn parse_invalid_json() {
        let result = parse_message("{invalid");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, -32700);
    }

    #[test]
    fn string_id() {
        let req = Request {
            method: "test".into(),
            params: None,
            id: Some(Id::String("abc-123".into())),
        };
        let j = req.to_json();
        let parsed = Request::from_json(&j).unwrap();
        assert_eq!(parsed.id, Some(Id::String("abc-123".into())));
    }

    #[test]
    fn missing_version_error() {
        let val = json!({"method": "test", "id": 1});
        let err = Request::from_json(&val).unwrap_err();
        assert_eq!(err.code, -32600);
    }
}
