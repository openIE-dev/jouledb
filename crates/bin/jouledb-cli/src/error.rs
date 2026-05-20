//! CLI Error Types

use thiserror::Error;

/// CLI Result type
pub type Result<T> = std::result::Result<T, CliError>;

/// CLI Error
#[derive(Error, Debug)]
pub enum CliError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Operation cancelled")]
    Cancelled,

    #[error("Cloud API error: {status} - {message}")]
    CloudApi { status: u16, message: String },

    #[error("Not authenticated. Run 'jouledb cloud login' first.")]
    NotAuthenticated,

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("{0}")]
    Other(String),
}

impl From<String> for CliError {
    fn from(s: String) -> Self {
        CliError::Other(s)
    }
}

impl From<&str> for CliError {
    fn from(s: &str) -> Self {
        CliError::Other(s.to_string())
    }
}
