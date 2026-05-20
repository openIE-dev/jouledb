//! Error pages — status code mapping, HTML error templates, JSON error responses,
//! error detail levels by environment, stack trace formatting, and error ID tracking.
//!
//! Replaces Express error middleware, Next.js error pages, and custom error handlers
//! with a pure-Rust error page engine for both HTML and JSON responses.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

// ── Errors ─────────────────────────────────────────────────────

/// Error page errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorPageError {
    /// Template not found for status code.
    TemplateNotFound(u16),
    /// Invalid status code.
    InvalidStatus(u16),
    /// Missing required field in template.
    MissingField(String),
}

impl fmt::Display for ErrorPageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TemplateNotFound(code) => write!(f, "template not found for status {code}"),
            Self::InvalidStatus(code) => write!(f, "invalid status code: {code}"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
        }
    }
}

impl std::error::Error for ErrorPageError {}

// ── Environment ────────────────────────────────────────────────

/// Application environment controlling error detail level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Environment {
    /// Development — show full details including stack traces.
    Development,
    /// Staging — show error messages but not stack traces.
    Staging,
    /// Production — show only generic messages.
    Production,
}

impl Default for Environment {
    fn default() -> Self {
        Self::Production
    }
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Development => write!(f, "development"),
            Self::Staging => write!(f, "staging"),
            Self::Production => write!(f, "production"),
        }
    }
}

// ── Error Detail Level ─────────────────────────────────────────

/// How much detail to include in error responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DetailLevel {
    /// Only generic error message ("Internal Server Error").
    Minimal,
    /// Include error message and error ID.
    Standard,
    /// Include error message, error ID, and context.
    Detailed,
    /// Include everything including stack traces (dev only).
    Full,
}

impl Environment {
    /// Get the appropriate detail level for this environment.
    pub fn detail_level(&self) -> DetailLevel {
        match self {
            Self::Development => DetailLevel::Full,
            Self::Staging => DetailLevel::Standard,
            Self::Production => DetailLevel::Minimal,
        }
    }
}

// ── Stack Frame ────────────────────────────────────────────────

/// A single frame in a stack trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function or method name.
    pub function: String,
    /// Source file path.
    pub file: Option<String>,
    /// Line number.
    pub line: Option<u32>,
    /// Column number.
    pub column: Option<u32>,
}

impl fmt::Display for StackFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "  at {}", self.function)?;
        if let Some(ref file) = self.file {
            write!(f, " ({file}")?;
            if let Some(line) = self.line {
                write!(f, ":{line}")?;
                if let Some(col) = self.column {
                    write!(f, ":{col}")?;
                }
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Format a list of stack frames.
pub fn format_stack_trace(frames: &[StackFrame]) -> String {
    frames
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

// ── Error Info ─────────────────────────────────────────────────

/// Structured error information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Unique error ID for tracking.
    pub error_id: String,
    /// HTTP status code.
    pub status: u16,
    /// Human-readable status text.
    pub status_text: String,
    /// Error message.
    pub message: String,
    /// Detailed description (shown based on detail level).
    pub detail: Option<String>,
    /// Error code for programmatic handling.
    pub error_code: Option<String>,
    /// Stack trace frames.
    pub stack_trace: Vec<StackFrame>,
    /// Additional context key-value pairs.
    pub context: HashMap<String, String>,
    /// Timestamp as ISO 8601 string.
    pub timestamp: String,
    /// Request path that caused the error.
    pub request_path: Option<String>,
}

impl ErrorInfo {
    /// Create a new error info with generated ID and timestamp.
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            error_id: Uuid::new_v4().to_string(),
            status,
            status_text: status_text(status).to_string(),
            message: message.into(),
            detail: None,
            error_code: None,
            stack_trace: Vec::new(),
            context: HashMap::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            request_path: None,
        }
    }

    /// Set detail.
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Set error code.
    pub fn with_error_code(mut self, code: impl Into<String>) -> Self {
        self.error_code = Some(code.into());
        self
    }

    /// Add a stack trace frame.
    pub fn with_frame(mut self, frame: StackFrame) -> Self {
        self.stack_trace.push(frame);
        self
    }

    /// Add context.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Set request path.
    pub fn with_request_path(mut self, path: impl Into<String>) -> Self {
        self.request_path = Some(path.into());
        self
    }

    /// Set a specific error ID (useful for deterministic testing).
    pub fn with_error_id(mut self, id: impl Into<String>) -> Self {
        self.error_id = id.into();
        self
    }
}

// ── Status Code Helpers ────────────────────────────────────────

/// Get standard status text for a code.
pub fn status_text(code: u16) -> &'static str {
    match code {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown Error",
    }
}

/// Check if a status code is a client error (4xx).
pub fn is_client_error(code: u16) -> bool {
    (400..500).contains(&code)
}

/// Check if a status code is a server error (5xx).
pub fn is_server_error(code: u16) -> bool {
    (500..600).contains(&code)
}

// ── JSON Response ──────────────────────────────────────────────

/// Generate a JSON error response filtered by detail level.
pub fn json_error_response(info: &ErrorInfo, level: DetailLevel) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("error".into(), serde_json::Value::String(info.status_text.clone()));
    obj.insert("status".into(), serde_json::Value::Number(info.status.into()));

    match level {
        DetailLevel::Minimal => {
            // Only generic error and status
        }
        DetailLevel::Standard => {
            obj.insert("message".into(), serde_json::Value::String(info.message.clone()));
            obj.insert("error_id".into(), serde_json::Value::String(info.error_id.clone()));
        }
        DetailLevel::Detailed => {
            obj.insert("message".into(), serde_json::Value::String(info.message.clone()));
            obj.insert("error_id".into(), serde_json::Value::String(info.error_id.clone()));
            if let Some(ref detail) = info.detail {
                obj.insert("detail".into(), serde_json::Value::String(detail.clone()));
            }
            if let Some(ref code) = info.error_code {
                obj.insert("error_code".into(), serde_json::Value::String(code.clone()));
            }
            if !info.context.is_empty() {
                let ctx: serde_json::Map<String, serde_json::Value> = info.context.iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect();
                obj.insert("context".into(), serde_json::Value::Object(ctx));
            }
        }
        DetailLevel::Full => {
            obj.insert("message".into(), serde_json::Value::String(info.message.clone()));
            obj.insert("error_id".into(), serde_json::Value::String(info.error_id.clone()));
            if let Some(ref detail) = info.detail {
                obj.insert("detail".into(), serde_json::Value::String(detail.clone()));
            }
            if let Some(ref code) = info.error_code {
                obj.insert("error_code".into(), serde_json::Value::String(code.clone()));
            }
            if !info.context.is_empty() {
                let ctx: serde_json::Map<String, serde_json::Value> = info.context.iter()
                    .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
                    .collect();
                obj.insert("context".into(), serde_json::Value::Object(ctx));
            }
            if !info.stack_trace.is_empty() {
                let trace = format_stack_trace(&info.stack_trace);
                obj.insert("stack_trace".into(), serde_json::Value::String(trace));
            }
            obj.insert("timestamp".into(), serde_json::Value::String(info.timestamp.clone()));
            if let Some(ref path) = info.request_path {
                obj.insert("request_path".into(), serde_json::Value::String(path.clone()));
            }
        }
    }

    serde_json::Value::Object(obj)
}

// ── HTML Response ──────────────────────────────────────────────

/// Generate an HTML error page filtered by detail level.
pub fn html_error_page(info: &ErrorInfo, level: DetailLevel) -> String {
    let mut body = String::new();
    body.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    body.push_str(&format!("  <title>{} {}</title>\n", info.status, info.status_text));
    body.push_str("  <meta charset=\"utf-8\">\n");
    body.push_str("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    body.push_str("  <style>\n");
    body.push_str("    body { font-family: system-ui, sans-serif; margin: 2rem; color: #333; }\n");
    body.push_str("    h1 { font-size: 2rem; }\n");
    body.push_str("    .error-id { color: #999; font-size: 0.875rem; }\n");
    body.push_str("    .detail { margin-top: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 4px; }\n");
    body.push_str("    pre { background: #1a1a2e; color: #eee; padding: 1rem; overflow-x: auto; border-radius: 4px; }\n");
    body.push_str("  </style>\n");
    body.push_str("</head>\n<body>\n");
    body.push_str(&format!("  <h1>{} {}</h1>\n", info.status, info.status_text));

    match level {
        DetailLevel::Minimal => {
            body.push_str("  <p>An error occurred while processing your request.</p>\n");
        }
        DetailLevel::Standard => {
            body.push_str(&format!("  <p>{}</p>\n", html_escape(&info.message)));
            body.push_str(&format!(
                "  <p class=\"error-id\">Error ID: {}</p>\n",
                html_escape(&info.error_id)
            ));
        }
        DetailLevel::Detailed => {
            body.push_str(&format!("  <p>{}</p>\n", html_escape(&info.message)));
            body.push_str(&format!(
                "  <p class=\"error-id\">Error ID: {}</p>\n",
                html_escape(&info.error_id)
            ));
            if let Some(ref detail) = info.detail {
                body.push_str(&format!(
                    "  <div class=\"detail\">{}</div>\n",
                    html_escape(detail)
                ));
            }
        }
        DetailLevel::Full => {
            body.push_str(&format!("  <p>{}</p>\n", html_escape(&info.message)));
            body.push_str(&format!(
                "  <p class=\"error-id\">Error ID: {}</p>\n",
                html_escape(&info.error_id)
            ));
            if let Some(ref detail) = info.detail {
                body.push_str(&format!(
                    "  <div class=\"detail\">{}</div>\n",
                    html_escape(detail)
                ));
            }
            if !info.stack_trace.is_empty() {
                let trace = format_stack_trace(&info.stack_trace);
                body.push_str(&format!("  <pre>{}</pre>\n", html_escape(&trace)));
            }
            body.push_str(&format!("  <p><small>Timestamp: {}</small></p>\n", html_escape(&info.timestamp)));
            if let Some(ref path) = info.request_path {
                body.push_str(&format!("  <p><small>Path: {}</small></p>\n", html_escape(path)));
            }
        }
    }

    body.push_str("</body>\n</html>");
    body
}

/// Simple HTML escaping.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

// ── Error Page Registry ────────────────────────────────────────

/// Registry managing custom error templates and configuration.
#[derive(Debug, Clone)]
pub struct ErrorPageRegistry {
    /// Environment setting.
    pub environment: Environment,
    /// Custom message overrides per status code.
    custom_messages: HashMap<u16, String>,
    /// Whether to include error IDs in responses.
    pub include_error_id: bool,
}

impl ErrorPageRegistry {
    /// Create a new registry.
    pub fn new(environment: Environment) -> Self {
        Self {
            environment,
            custom_messages: HashMap::new(),
            include_error_id: true,
        }
    }

    /// Register a custom message for a status code.
    pub fn set_custom_message(&mut self, status: u16, message: impl Into<String>) {
        self.custom_messages.insert(status, message.into());
    }

    /// Get the message for a status code (custom or default).
    pub fn message_for(&self, status: u16) -> String {
        self.custom_messages
            .get(&status)
            .cloned()
            .unwrap_or_else(|| status_text(status).to_string())
    }

    /// Build an ErrorInfo from a status code and message.
    pub fn build_error(&self, status: u16, message: impl Into<String>) -> ErrorInfo {
        ErrorInfo::new(status, message)
    }

    /// Render a JSON error response.
    pub fn render_json(&self, info: &ErrorInfo) -> serde_json::Value {
        json_error_response(info, self.environment.detail_level())
    }

    /// Render an HTML error page.
    pub fn render_html(&self, info: &ErrorInfo) -> String {
        html_error_page(info, self.environment.detail_level())
    }
}

impl Default for ErrorPageRegistry {
    fn default() -> Self {
        Self::new(Environment::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_error() -> ErrorInfo {
        ErrorInfo::new(500, "Something went wrong")
            .with_error_id("err-123")
            .with_detail("Database connection failed")
            .with_error_code("DB_CONN_FAIL")
            .with_request_path("/api/users")
            .with_context("db_host", "localhost")
            .with_frame(StackFrame {
                function: "db::connect".to_string(),
                file: Some("src/db.rs".to_string()),
                line: Some(42),
                column: Some(5),
            })
            .with_frame(StackFrame {
                function: "main".to_string(),
                file: Some("src/main.rs".to_string()),
                line: Some(10),
                column: None,
            })
    }

    #[test]
    fn test_status_text() {
        assert_eq!(status_text(404), "Not Found");
        assert_eq!(status_text(500), "Internal Server Error");
        assert_eq!(status_text(200), "Unknown Error");
    }

    #[test]
    fn test_is_client_error() {
        assert!(is_client_error(400));
        assert!(is_client_error(404));
        assert!(is_client_error(429));
        assert!(!is_client_error(500));
        assert!(!is_client_error(200));
    }

    #[test]
    fn test_is_server_error() {
        assert!(is_server_error(500));
        assert!(is_server_error(503));
        assert!(!is_server_error(404));
        assert!(!is_server_error(200));
    }

    #[test]
    fn test_error_info_creation() {
        let info = ErrorInfo::new(404, "Page not found");
        assert_eq!(info.status, 404);
        assert_eq!(info.status_text, "Not Found");
        assert_eq!(info.message, "Page not found");
        assert!(!info.error_id.is_empty());
        assert!(!info.timestamp.is_empty());
    }

    #[test]
    fn test_error_info_builder() {
        let info = sample_error();
        assert_eq!(info.error_id, "err-123");
        assert_eq!(info.detail.as_deref(), Some("Database connection failed"));
        assert_eq!(info.error_code.as_deref(), Some("DB_CONN_FAIL"));
        assert_eq!(info.request_path.as_deref(), Some("/api/users"));
        assert_eq!(info.stack_trace.len(), 2);
        assert_eq!(info.context.get("db_host").unwrap(), "localhost");
    }

    #[test]
    fn test_stack_frame_display() {
        let frame = StackFrame {
            function: "my_func".to_string(),
            file: Some("src/lib.rs".to_string()),
            line: Some(10),
            column: Some(3),
        };
        let s = format!("{frame}");
        assert!(s.contains("my_func"));
        assert!(s.contains("src/lib.rs:10:3"));
    }

    #[test]
    fn test_stack_frame_no_file() {
        let frame = StackFrame {
            function: "anonymous".to_string(),
            file: None,
            line: None,
            column: None,
        };
        let s = format!("{frame}");
        assert!(s.contains("anonymous"));
        assert!(!s.contains("("));
    }

    #[test]
    fn test_format_stack_trace() {
        let frames = vec![
            StackFrame { function: "a".to_string(), file: None, line: None, column: None },
            StackFrame { function: "b".to_string(), file: Some("x.rs".to_string()), line: Some(1), column: None },
        ];
        let trace = format_stack_trace(&frames);
        assert!(trace.contains("a"));
        assert!(trace.contains("b"));
        assert!(trace.contains("x.rs:1"));
    }

    #[test]
    fn test_json_minimal() {
        let info = sample_error();
        let json = json_error_response(&info, DetailLevel::Minimal);
        assert_eq!(json.get("status").unwrap().as_u64().unwrap(), 500);
        assert!(json.get("error").is_some());
        assert!(json.get("message").is_none());
        assert!(json.get("error_id").is_none());
        assert!(json.get("stack_trace").is_none());
    }

    #[test]
    fn test_json_standard() {
        let info = sample_error();
        let json = json_error_response(&info, DetailLevel::Standard);
        assert_eq!(json.get("message").unwrap().as_str().unwrap(), "Something went wrong");
        assert_eq!(json.get("error_id").unwrap().as_str().unwrap(), "err-123");
        assert!(json.get("stack_trace").is_none());
        assert!(json.get("detail").is_none());
    }

    #[test]
    fn test_json_detailed() {
        let info = sample_error();
        let json = json_error_response(&info, DetailLevel::Detailed);
        assert!(json.get("message").is_some());
        assert!(json.get("error_id").is_some());
        assert!(json.get("detail").is_some());
        assert!(json.get("error_code").is_some());
        assert!(json.get("context").is_some());
        assert!(json.get("stack_trace").is_none());
    }

    #[test]
    fn test_json_full() {
        let info = sample_error();
        let json = json_error_response(&info, DetailLevel::Full);
        assert!(json.get("message").is_some());
        assert!(json.get("error_id").is_some());
        assert!(json.get("detail").is_some());
        assert!(json.get("error_code").is_some());
        assert!(json.get("stack_trace").is_some());
        assert!(json.get("timestamp").is_some());
        assert!(json.get("request_path").is_some());
    }

    #[test]
    fn test_html_minimal() {
        let info = sample_error();
        let html = html_error_page(&info, DetailLevel::Minimal);
        assert!(html.contains("500 Internal Server Error"));
        assert!(html.contains("An error occurred"));
        assert!(!html.contains("err-123"));
        assert!(!html.contains("db::connect"));
    }

    #[test]
    fn test_html_standard() {
        let info = sample_error();
        let html = html_error_page(&info, DetailLevel::Standard);
        assert!(html.contains("Something went wrong"));
        assert!(html.contains("err-123"));
        assert!(!html.contains("db::connect"));
    }

    #[test]
    fn test_html_detailed() {
        let info = sample_error();
        let html = html_error_page(&info, DetailLevel::Detailed);
        assert!(html.contains("Database connection failed"));
        assert!(!html.contains("db::connect"));
    }

    #[test]
    fn test_html_full() {
        let info = sample_error();
        let html = html_error_page(&info, DetailLevel::Full);
        assert!(html.contains("db::connect"));
        assert!(html.contains("src/db.rs:42:5"));
        assert!(html.contains("/api/users"));
    }

    #[test]
    fn test_html_escape() {
        let info = ErrorInfo::new(400, "invalid <script>alert(1)</script>")
            .with_error_id("e1");
        let html = html_error_page(&info, DetailLevel::Standard);
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_environment_detail_level() {
        assert_eq!(Environment::Development.detail_level(), DetailLevel::Full);
        assert_eq!(Environment::Staging.detail_level(), DetailLevel::Standard);
        assert_eq!(Environment::Production.detail_level(), DetailLevel::Minimal);
    }

    #[test]
    fn test_environment_default() {
        assert_eq!(Environment::default(), Environment::Production);
    }

    #[test]
    fn test_environment_display() {
        assert_eq!(format!("{}", Environment::Development), "development");
        assert_eq!(format!("{}", Environment::Production), "production");
    }

    #[test]
    fn test_registry_custom_message() {
        let mut reg = ErrorPageRegistry::new(Environment::Production);
        reg.set_custom_message(404, "We could not find that page.");
        assert_eq!(reg.message_for(404), "We could not find that page.");
        assert_eq!(reg.message_for(500), "Internal Server Error");
    }

    #[test]
    fn test_registry_render_json() {
        let reg = ErrorPageRegistry::new(Environment::Production);
        let info = reg.build_error(404, "Not found");
        let json = reg.render_json(&info);
        assert_eq!(json.get("status").unwrap().as_u64().unwrap(), 404);
        // Production = minimal, so no message
        assert!(json.get("message").is_none());
    }

    #[test]
    fn test_registry_render_json_dev() {
        let reg = ErrorPageRegistry::new(Environment::Development);
        let info = ErrorInfo::new(500, "oops")
            .with_error_id("e1")
            .with_frame(StackFrame {
                function: "handler".to_string(),
                file: None,
                line: None,
                column: None,
            });
        let json = reg.render_json(&info);
        assert!(json.get("stack_trace").is_some());
    }

    #[test]
    fn test_registry_render_html() {
        let reg = ErrorPageRegistry::new(Environment::Staging);
        let info = ErrorInfo::new(503, "Service unavailable").with_error_id("e1");
        let html = reg.render_html(&info);
        assert!(html.contains("503"));
        assert!(html.contains("Service unavailable"));
    }

    #[test]
    fn test_registry_default() {
        let reg = ErrorPageRegistry::default();
        assert_eq!(reg.environment, Environment::Production);
    }

    #[test]
    fn test_error_page_error_display() {
        assert!(format!("{}", ErrorPageError::TemplateNotFound(404)).contains("404"));
        assert!(format!("{}", ErrorPageError::InvalidStatus(999)).contains("999"));
        assert!(format!("{}", ErrorPageError::MissingField("x".into())).contains("x"));
    }

    #[test]
    fn test_detail_level_ordering() {
        assert!(DetailLevel::Minimal < DetailLevel::Standard);
        assert!(DetailLevel::Standard < DetailLevel::Detailed);
        assert!(DetailLevel::Detailed < DetailLevel::Full);
    }

    #[test]
    fn test_json_no_optional_fields() {
        let info = ErrorInfo::new(400, "bad request").with_error_id("e1");
        let json = json_error_response(&info, DetailLevel::Detailed);
        assert!(json.get("detail").is_none());
        assert!(json.get("error_code").is_none());
        assert!(json.get("context").is_none());
    }

    #[test]
    fn test_all_common_status_codes() {
        let codes = [400, 401, 403, 404, 405, 408, 409, 410, 413, 415, 422, 429, 500, 501, 502, 503, 504];
        for code in codes {
            let text = status_text(code);
            assert_ne!(text, "Unknown Error", "missing text for {code}");
        }
    }
}
