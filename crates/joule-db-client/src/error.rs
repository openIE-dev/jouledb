//! Error types for the JouleDB client SDK.

use std::fmt;

/// All errors that the JouleDB client can produce.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// Failed to establish a TCP connection to the server.
    #[error("connection failed: {reason}")]
    ConnectionFailed {
        /// Human-readable reason for the failure.
        reason: String,
    },

    /// The connection was closed unexpectedly.
    #[error("connection closed")]
    ConnectionClosed,

    /// An operation timed out.
    #[error("operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// The server sent data that does not conform to the binary wire protocol.
    #[error("protocol error: {0}")]
    ProtocolError(String),

    /// The server returned an explicit error response.
    #[error("server error [{code}]: {message}")]
    ServerError {
        /// Error code from the server (e.g. "NOT_FOUND").
        code: String,
        /// Descriptive error message from the server.
        message: String,
    },

    /// The server response could not be parsed or was for a different request.
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// An underlying I/O error.
    #[error("io error: {0}")]
    IoError(#[from] std::io::Error),

    /// The connection pool has no available connections and the timeout expired.
    #[error("connection pool exhausted")]
    PoolExhausted,
}

impl ClientError {
    /// Create a `ConnectionFailed` variant from any displayable reason.
    pub fn connection_failed(reason: impl fmt::Display) -> Self {
        Self::ConnectionFailed {
            reason: reason.to_string(),
        }
    }

    /// Create a `ProtocolError` variant.
    pub fn protocol(msg: impl fmt::Display) -> Self {
        Self::ProtocolError(msg.to_string())
    }

    /// Create a `ServerError` variant.
    pub fn server(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ServerError {
            code: code.into(),
            message: message.into(),
        }
    }

    /// Create an `InvalidResponse` variant.
    pub fn invalid_response(msg: impl fmt::Display) -> Self {
        Self::InvalidResponse(msg.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ClientError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ClientError::connection_failed("refused");
        assert_eq!(err.to_string(), "connection failed: refused");

        let err = ClientError::server("NOT_FOUND", "Key does not exist");
        assert_eq!(
            err.to_string(),
            "server error [NOT_FOUND]: Key does not exist"
        );

        let err = ClientError::ConnectionClosed;
        assert_eq!(err.to_string(), "connection closed");

        let err = ClientError::PoolExhausted;
        assert_eq!(err.to_string(), "connection pool exhausted");

        let err = ClientError::Timeout(std::time::Duration::from_secs(5));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken");
        let client_err: ClientError = io_err.into();
        assert!(matches!(client_err, ClientError::IoError(_)));
        assert!(client_err.to_string().contains("broken"));
    }

    #[test]
    fn test_convenience_constructors() {
        let err = ClientError::protocol("bad magic");
        assert!(matches!(err, ClientError::ProtocolError(_)));

        let err = ClientError::invalid_response("wrong type");
        assert!(matches!(err, ClientError::InvalidResponse(_)));
    }
}
