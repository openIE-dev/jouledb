//! Network Protocol for JouleDB Server
//!
//! Defines message types for HTTP and WebSocket communication.

use serde::{Deserialize, Serialize};

// ============================================================================
// Protocol Messages
// ============================================================================

/// Protocol message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProtocolMessage {
    /// Query message (SQL or other query language)
    Query {
        query: String,
        params: Option<Vec<serde_json::Value>>,
    },
    /// Set key-value
    Set {
        key: String,
        value: Vec<u8>,
        ttl: Option<u64>,
    },
    /// Get key-value
    Get { key: String },
    /// Delete key-value
    Delete { key: String },
    /// Subscribe to changes
    Subscribe { pattern: String },
    /// Unsubscribe from changes
    Unsubscribe { pattern: String },
    /// Sync request
    Sync { last_sync: Option<u64> },
    /// Batch operations
    Batch { operations: Vec<BatchOperation> },
    /// Ping (health check)
    Ping,
    /// Pong (response to ping)
    Pong,
    /// Authentication request
    Auth { token: String },
}

/// Batch operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum BatchOperation {
    Set {
        key: String,
        value: Vec<u8>,
        ttl: Option<u64>,
    },
    Delete {
        key: String,
    },
}

// ============================================================================
// Protocol Responses
// ============================================================================

/// Protocol response
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ProtocolResponse {
    /// Query result
    QueryResult {
        rows: Vec<serde_json::Value>,
        columns: Vec<String>,
        affected_rows: Option<usize>,
    },
    /// Key-value result
    Value { value: Option<Vec<u8>> },
    /// Success response
    Success { message: String },
    /// Error response
    Error { code: String, message: String },
    /// Subscription notification
    Notification {
        pattern: String,
        key: String,
        value: Option<Vec<u8>>,
        operation: String, // "set", "delete"
    },
    /// Sync result
    SyncResult {
        changes: Vec<SyncChange>,
        last_sync: u64,
    },
    /// Batch result
    BatchResult { results: Vec<BatchItemResult> },
    /// Pong response
    Pong,
    /// Authentication result
    AuthResult {
        success: bool,
        session_id: Option<String>,
        expires_at: Option<u64>,
    },
}

/// Sync change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncChange {
    pub key: String,
    pub value: Option<Vec<u8>>,
    pub version: u64,
    pub timestamp: u64,
    pub operation: String, // "set", "delete"
}

/// Batch item result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItemResult {
    pub success: bool,
    pub error: Option<String>,
}

// ============================================================================
// Protocol Handler
// ============================================================================

/// Protocol error
#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolError {
    ParseError(String),
    SerializeError(String),
    InvalidMessage(String),
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::SerializeError(msg) => write!(f, "Serialize error: {}", msg),
            Self::InvalidMessage(msg) => write!(f, "Invalid message: {}", msg),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Protocol handler for message parsing and serialization
pub struct ProtocolHandler;

impl ProtocolHandler {
    /// Create new protocol handler
    pub fn new() -> Self {
        Self
    }

    /// Parse message from JSON
    pub fn parse_message(&self, json: &str) -> Result<ProtocolMessage, ProtocolError> {
        serde_json::from_str(json).map_err(|e| ProtocolError::ParseError(e.to_string()))
    }

    /// Parse message from bytes
    pub fn parse_bytes(&self, data: &[u8]) -> Result<ProtocolMessage, ProtocolError> {
        serde_json::from_slice(data).map_err(|e| ProtocolError::ParseError(e.to_string()))
    }

    /// Serialize message to JSON
    pub fn serialize_message(&self, message: &ProtocolMessage) -> Result<String, ProtocolError> {
        serde_json::to_string(message).map_err(|e| ProtocolError::SerializeError(e.to_string()))
    }

    /// Serialize response to JSON
    pub fn serialize_response(&self, response: &ProtocolResponse) -> Result<String, ProtocolError> {
        serde_json::to_string(response).map_err(|e| ProtocolError::SerializeError(e.to_string()))
    }

    /// Serialize response to bytes
    pub fn serialize_response_bytes(
        &self,
        response: &ProtocolResponse,
    ) -> Result<Vec<u8>, ProtocolError> {
        serde_json::to_vec(response).map_err(|e| ProtocolError::SerializeError(e.to_string()))
    }

    /// Create success response
    pub fn success(message: impl Into<String>) -> ProtocolResponse {
        ProtocolResponse::Success {
            message: message.into(),
        }
    }

    /// Create error response
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> ProtocolResponse {
        ProtocolResponse::Error {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl Default for ProtocolHandler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WebSocket Frame Types
// ============================================================================

/// WebSocket frame type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Text,
    Binary,
    Ping,
    Pong,
    Close,
}

/// WebSocket frame
#[derive(Debug, Clone)]
pub struct Frame {
    pub frame_type: FrameType,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Create text frame
    pub fn text(data: impl Into<String>) -> Self {
        Self {
            frame_type: FrameType::Text,
            payload: data.into().into_bytes(),
        }
    }

    /// Create binary frame
    pub fn binary(data: impl Into<Vec<u8>>) -> Self {
        Self {
            frame_type: FrameType::Binary,
            payload: data.into(),
        }
    }

    /// Create ping frame
    pub fn ping() -> Self {
        Self {
            frame_type: FrameType::Ping,
            payload: Vec::new(),
        }
    }

    /// Create pong frame
    pub fn pong() -> Self {
        Self {
            frame_type: FrameType::Pong,
            payload: Vec::new(),
        }
    }

    /// Create close frame
    pub fn close() -> Self {
        Self {
            frame_type: FrameType::Close,
            payload: Vec::new(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_get_message() {
        let handler = ProtocolHandler::new();
        let json = r#"{"type":"Get","key":"mykey"}"#;
        let message = handler.parse_message(json).unwrap();

        match message {
            ProtocolMessage::Get { key } => assert_eq!(key, "mykey"),
            _ => panic!("Expected Get message"),
        }
    }

    #[test]
    fn test_parse_set_message() {
        let handler = ProtocolHandler::new();
        let json = r#"{"type":"Set","key":"mykey","value":[1,2,3],"ttl":null}"#;
        let message = handler.parse_message(json).unwrap();

        match message {
            ProtocolMessage::Set { key, value, ttl } => {
                assert_eq!(key, "mykey");
                assert_eq!(value, vec![1, 2, 3]);
                assert!(ttl.is_none());
            }
            _ => panic!("Expected Set message"),
        }
    }

    #[test]
    fn test_parse_query_message() {
        let handler = ProtocolHandler::new();
        let json = r#"{"type":"Query","query":"SELECT * FROM users","params":null}"#;
        let message = handler.parse_message(json).unwrap();

        match message {
            ProtocolMessage::Query { query, params } => {
                assert_eq!(query, "SELECT * FROM users");
                assert!(params.is_none());
            }
            _ => panic!("Expected Query message"),
        }
    }

    #[test]
    fn test_serialize_response() {
        let handler = ProtocolHandler::new();
        let response = ProtocolResponse::Success {
            message: "OK".to_string(),
        };
        let json = handler.serialize_response(&response).unwrap();
        assert!(json.contains("Success"));
        assert!(json.contains("OK"));
    }

    #[test]
    fn test_error_response() {
        let response = ProtocolHandler::error("NOT_FOUND", "Key not found");
        match response {
            ProtocolResponse::Error { code, message } => {
                assert_eq!(code, "NOT_FOUND");
                assert_eq!(message, "Key not found");
            }
            _ => panic!("Expected Error response"),
        }
    }

    #[test]
    fn test_sync_change() {
        let change = SyncChange {
            key: "key1".to_string(),
            value: Some(vec![1, 2, 3]),
            version: 1,
            timestamp: 12345678,
            operation: "set".to_string(),
        };

        let json = serde_json::to_string(&change).unwrap();
        let parsed: SyncChange = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.key, "key1");
        assert_eq!(parsed.version, 1);
    }

    #[test]
    fn test_batch_operation() {
        let handler = ProtocolHandler::new();
        let json = r#"{"type":"Batch","operations":[{"op":"Set","key":"k1","value":[1],"ttl":null},{"op":"Delete","key":"k2"}]}"#;
        let message = handler.parse_message(json).unwrap();

        match message {
            ProtocolMessage::Batch { operations } => {
                assert_eq!(operations.len(), 2);
            }
            _ => panic!("Expected Batch message"),
        }
    }

    #[test]
    fn test_frame_types() {
        let text_frame = Frame::text("hello");
        assert_eq!(text_frame.frame_type, FrameType::Text);
        assert_eq!(text_frame.payload, b"hello");

        let binary_frame = Frame::binary(vec![1, 2, 3]);
        assert_eq!(binary_frame.frame_type, FrameType::Binary);

        let ping_frame = Frame::ping();
        assert_eq!(ping_frame.frame_type, FrameType::Ping);
    }
}
