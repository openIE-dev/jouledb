//! Error types for the Metal runtime
//!
//! This module defines the error types used throughout the Metal runtime.
//! All errors are designed to provide clear, actionable information about
//! what went wrong.

use thiserror::Error;

/// Result type alias for Metal operations
pub type MetalResult<T> = Result<T, MetalError>;

/// Errors that can occur during Metal operations
#[derive(Debug, Error)]
pub enum MetalError {
    /// Metal is not available on this platform
    #[error("Metal is not available: {reason}")]
    NotAvailable {
        /// Reason Metal is not available
        reason: String,
    },

    /// No Metal-capable device was found
    #[error("no Metal device found")]
    NoDevice,

    /// Failed to create a Metal device
    #[error("failed to create Metal device: {message}")]
    DeviceCreation {
        /// Error message
        message: String,
    },

    /// Buffer allocation failed
    #[error("buffer allocation failed: {message}")]
    BufferAllocation {
        /// Error message
        message: String,
        /// Requested size in bytes
        size: usize,
    },

    /// Buffer is too large for the device
    #[error("buffer size {size} exceeds device maximum {max_size}")]
    BufferTooLarge {
        /// Requested size
        size: usize,
        /// Maximum allowed size
        max_size: usize,
    },

    /// Invalid buffer access
    #[error("invalid buffer access: {message}")]
    InvalidBufferAccess {
        /// Error message
        message: String,
    },

    /// Command queue creation failed
    #[error("failed to create command queue: {message}")]
    CommandQueueCreation {
        /// Error message
        message: String,
    },

    /// Command buffer creation failed
    #[error("failed to create command buffer: {message}")]
    CommandBufferCreation {
        /// Error message
        message: String,
    },

    /// Command encoding failed
    #[error("command encoding failed: {message}")]
    CommandEncoding {
        /// Error message
        message: String,
    },

    /// Command execution failed
    #[error("command execution failed: {message}")]
    CommandExecution {
        /// Error message
        message: String,
    },

    /// Library compilation failed
    #[error("library compilation failed: {message}")]
    LibraryCompilation {
        /// Error message
        message: String,
    },

    /// Function not found in library
    #[error("function '{name}' not found in library")]
    FunctionNotFound {
        /// Function name
        name: String,
    },

    /// Pipeline creation failed
    #[error("pipeline creation failed: {message}")]
    PipelineCreation {
        /// Error message
        message: String,
    },

    /// Invalid threadgroup size
    #[error("invalid threadgroup size: {width}x{height}x{depth} exceeds maximum {max_total}")]
    InvalidThreadgroupSize {
        /// Requested width
        width: u32,
        /// Requested height
        height: u32,
        /// Requested depth
        depth: u32,
        /// Maximum total threads
        max_total: u32,
    },

    /// Synchronization error
    #[error("synchronization error: {message}")]
    Synchronization {
        /// Error message
        message: String,
    },

    /// Timeout waiting for GPU operation
    #[error("timeout waiting for GPU operation after {timeout_ms}ms")]
    Timeout {
        /// Timeout duration in milliseconds
        timeout_ms: u64,
    },

    /// GPU execution error (returned in command buffer status)
    #[error("GPU execution error: {message}")]
    GpuExecution {
        /// Error message
        message: String,
    },

    /// Resource not available
    #[error("resource not available: {resource}")]
    ResourceNotAvailable {
        /// Resource name
        resource: String,
    },

    /// Internal error (should not happen in normal operation)
    #[error("internal error: {message}")]
    Internal {
        /// Error message
        message: String,
    },
}

impl MetalError {
    /// Create a NotAvailable error for non-macOS platforms
    #[must_use]
    pub fn not_macos() -> Self {
        Self::NotAvailable {
            reason: "Metal is only available on macOS and iOS".to_string(),
        }
    }

    /// Create a NotAvailable error for missing Metal SDK
    #[must_use]
    pub fn no_metal_sdk() -> Self {
        Self::NotAvailable {
            reason: "Metal SDK bindings not enabled (enable 'metal-sdk' feature)".to_string(),
        }
    }

    /// Check if this error is recoverable
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. }
                | Self::ResourceNotAvailable { .. }
                | Self::BufferAllocation { .. }
        )
    }

    /// Check if this error indicates a programming mistake
    #[must_use]
    pub fn is_programming_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidBufferAccess { .. }
                | Self::InvalidThreadgroupSize { .. }
                | Self::FunctionNotFound { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MetalError::NoDevice;
        assert_eq!(err.to_string(), "no Metal device found");
    }

    #[test]
    fn test_buffer_allocation_error() {
        let err = MetalError::BufferAllocation {
            message: "out of memory".to_string(),
            size: 1024,
        };
        assert!(err.to_string().contains("out of memory"));
    }

    #[test]
    fn test_not_macos() {
        let err = MetalError::not_macos();
        assert!(matches!(err, MetalError::NotAvailable { .. }));
    }

    #[test]
    fn test_is_recoverable() {
        assert!(MetalError::Timeout { timeout_ms: 1000 }.is_recoverable());
        assert!(!MetalError::NoDevice.is_recoverable());
    }

    #[test]
    fn test_is_programming_error() {
        let err = MetalError::InvalidThreadgroupSize {
            width: 1024,
            height: 1024,
            depth: 1,
            max_total: 1024,
        };
        assert!(err.is_programming_error());
        assert!(!MetalError::Timeout { timeout_ms: 1000 }.is_programming_error());
    }
}
