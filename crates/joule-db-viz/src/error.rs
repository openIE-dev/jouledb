//! Error types for joule-db-viz

use thiserror::Error;

/// Result type alias for visualization operations.
pub type VizResult<T> = Result<T, VizError>;

/// Errors that can occur during visualization inference or rendering.
#[derive(Debug, Error)]
pub enum VizError {
    /// Data is incompatible with the requested chart type
    #[error("incompatible data for {chart_type}: {reason}")]
    IncompatibleData { chart_type: String, reason: String },

    /// Rendering failed
    #[error("render error: {0}")]
    RenderError(String),

    /// GPU initialization or execution error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Serialization error
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Invalid configuration
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Data error (query execution, transformation)
    #[error("data error: {0}")]
    DataError(String),
}
