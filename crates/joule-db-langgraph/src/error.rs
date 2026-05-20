//! Error types for LangGraph integration

use thiserror::Error;

/// Result type for LangGraph operations
pub type LangGraphResult<T> = Result<T, LangGraphError>;

/// Errors that can occur in LangGraph operations
#[derive(Debug, Error)]
pub enum LangGraphError {
    /// Checkpoint not found
    #[error("Checkpoint not found: thread={thread_id}, checkpoint={checkpoint_id}")]
    CheckpointNotFound {
        /// Thread ID
        thread_id: String,
        /// Checkpoint ID
        checkpoint_id: String,
    },

    /// Thread not found
    #[error("Thread not found: {thread_id}")]
    ThreadNotFound {
        /// Thread ID
        thread_id: String,
    },

    /// Message not found
    #[error("Message not found: {message_id}")]
    MessageNotFound {
        /// Message ID
        message_id: String,
    },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Storage error
    #[error("Storage error: {0}")]
    Storage(String),

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

impl From<serde_json::Error> for LangGraphError {
    fn from(e: serde_json::Error) -> Self {
        LangGraphError::Serialization(e.to_string())
    }
}

impl From<joule_db_amorphic::AmorphicError> for LangGraphError {
    fn from(e: joule_db_amorphic::AmorphicError) -> Self {
        LangGraphError::Storage(e.to_string())
    }
}
