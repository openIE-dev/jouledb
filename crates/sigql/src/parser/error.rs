//! Parser error types

use thiserror::Error;

/// Errors that can occur during parsing
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Parse error: {0}")]
    NomError(String),

    #[error("Incomplete input, remaining: {0}")]
    IncompleteInput(String),

    #[error("Unknown transform: {0}")]
    UnknownTransform(String),

    #[error("Unknown aggregate: {0}")]
    UnknownAggregate(String),

    #[error("Invalid frequency: {0}")]
    InvalidFrequency(String),

    #[error("Invalid duration: {0}")]
    InvalidDuration(String),

    #[error("Missing required clause: {0}")]
    MissingClause(String),

    #[error("Type error: {0}")]
    TypeError(String),
}
