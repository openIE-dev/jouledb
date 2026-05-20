use thiserror::Error;

#[derive(Debug, Error)]
pub enum LedgerError {
    #[error("Receipt not found: {0}")]
    ReceiptNotFound(String),

    #[error("Batch not found: {0}")]
    BatchNotFound(String),

    #[error("Merkle proof generation failed: {0}")]
    ProofError(String),

    #[error("Backend commit failed: {0}")]
    CommitError(String),

    #[error("Persistence error: {0}")]
    PersistError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Collector channel closed")]
    ChannelClosed,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Backend error: {0}")]
    Backend(String),
}
