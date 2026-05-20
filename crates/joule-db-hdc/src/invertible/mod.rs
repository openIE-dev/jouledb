//! Invertible Visualization
//!
//! Embed and extract data from visualization images using steganography.
//!
//! ## Key Concepts
//!
//! - **LSB Encoding**: Hide data in least significant bits of image pixels
//! - **Capacity**: Amount of data that can be embedded
//! - **Error Detection**: CRC checksums for data integrity
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_hdc::invertible::{InvVis, VisCode};
//!
//! // Embed data into image
//! let encoder = VisCode::new();
//! let image_data = vec![255u8; 1000]; // Dummy image
//! let embedded = encoder.encode(b"secret data", &image_data)?;
//!
//! // Extract data
//! let extracted = encoder.decode(&embedded)?;
//! assert_eq!(extracted, b"secret data");
//! ```

mod codec;
mod steganography;

pub use codec::{EncodingMode, InvVis, VisCode};
pub use steganography::{LSBDecoder, LSBEncoder, StegoImage};

use thiserror::Error;

/// Errors for invertible operations
#[derive(Error, Debug, Clone)]
pub enum InvertibleError {
    /// Data too large for image capacity
    #[error("data too large: {data_size} bytes exceeds capacity of {capacity} bytes")]
    CapacityExceeded {
        /// Size of data
        data_size: usize,
        /// Available capacity
        capacity: usize,
    },

    /// Image too small
    #[error("image too small: need at least {needed} bytes, got {actual}")]
    ImageTooSmall {
        /// Needed size
        needed: usize,
        /// Actual size
        actual: usize,
    },

    /// Checksum mismatch
    #[error("checksum mismatch: data may be corrupted")]
    ChecksumMismatch,

    /// No data found
    #[error("no embedded data found")]
    NoDataFound,

    /// Invalid header
    #[error("invalid header: {0}")]
    InvalidHeader(String),

    /// Encoding error
    #[error("encoding error: {0}")]
    EncodingError(String),

    /// Decoding error
    #[error("decoding error: {0}")]
    DecodingError(String),
}

/// Result type for invertible operations
pub type InvertibleResult<T> = Result<T, InvertibleError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = InvertibleError::CapacityExceeded {
            data_size: 100,
            capacity: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }
}
