//! Query Error Types

use std::fmt;

/// Query error type
#[derive(Debug, Clone)]
pub enum QueryError {
    /// Parse error
    ParseError(String),
    /// Syntax error with position
    SyntaxError {
        message: String,
        line: usize,
        column: usize,
    },
    /// Unknown function
    UnknownFunction(String),
    /// Unknown column/field
    UnknownColumn(String),
    /// Unknown table/collection
    UnknownTable(String),
    /// Type mismatch
    TypeMismatch { expected: String, found: String },
    /// Invalid argument count
    InvalidArgumentCount {
        function: String,
        expected: usize,
        found: usize,
    },
    /// Type error (invalid cast, etc.)
    TypeError(String),
    /// Unsupported operation
    Unsupported(String),
    /// Execution error
    ExecutionError(String),
    /// Timeout error
    Timeout,
    /// Resource limit exceeded
    ResourceLimit(String),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::SyntaxError {
                message,
                line,
                column,
            } => {
                write!(f, "Syntax error at {}:{}: {}", line, column, message)
            }
            Self::UnknownFunction(name) => write!(f, "Unknown function: {}", name),
            Self::UnknownColumn(name) => write!(f, "Unknown column: {}", name),
            Self::UnknownTable(name) => write!(f, "Unknown table: {}", name),
            Self::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            Self::InvalidArgumentCount {
                function,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Invalid argument count for {}: expected {}, found {}",
                    function, expected, found
                )
            }
            Self::TypeError(msg) => write!(f, "Type error: {}", msg),
            Self::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
            Self::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            Self::Timeout => write!(f, "Query timeout"),
            Self::ResourceLimit(msg) => write!(f, "Resource limit exceeded: {}", msg),
        }
    }
}

impl std::error::Error for QueryError {}

/// Query result type
pub type QueryResult<T> = Result<T, QueryError>;

// Conversion from Arrow errors (when arrow-execution feature is enabled)
#[cfg(feature = "arrow-execution")]
impl From<arrow::error::ArrowError> for QueryError {
    fn from(err: arrow::error::ArrowError) -> Self {
        QueryError::ExecutionError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = QueryError::SyntaxError {
            message: "unexpected token".to_string(),
            line: 1,
            column: 10,
        };
        assert!(err.to_string().contains("1:10"));
    }

    #[test]
    fn test_parse_error() {
        let err = QueryError::ParseError("invalid query".to_string());
        assert!(err.to_string().contains("invalid query"));
    }
}
