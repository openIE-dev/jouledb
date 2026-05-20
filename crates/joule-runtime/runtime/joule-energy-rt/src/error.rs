//! Error types for energy monitoring

use std::io;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Hardware not supported: {0}")]
    Unsupported(String),

    #[error("Permission denied: {0}")]
    Permission(String),

    #[error("Energy counter overflow")]
    Overflow,

    #[error("Monitor already running")]
    AlreadyRunning,

    #[error("Monitor not started")]
    NotStarted,
}
