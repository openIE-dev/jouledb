use thiserror::Error;

/// Errors that can occur during JWP frame encoding/decoding.
#[derive(Debug, Error)]
pub enum JwpError {
    #[error("unknown frame type: 0x{0:02x}")]
    UnknownFrameType(u8),

    #[error("payload too large: {0} bytes (max {MAX_PAYLOAD_LEN})")]
    PayloadTooLarge(usize),

    #[error("incomplete frame: need {needed} bytes, have {available}")]
    IncompleteFrame { needed: usize, available: usize },

    #[error("invalid protocol version: {0}")]
    InvalidVersion(u8),

    #[error("CBOR decode error: {0}")]
    CborDecode(String),

    #[error("CBOR encode error: {0}")]
    CborEncode(String),

    #[error("invalid state transition: {from} -> {event}")]
    InvalidTransition { from: String, event: String },

    #[error("compression error: {0}")]
    CompressionError(String),

    #[error("negotiation failed: {0}")]
    NegotiationFailed(String),

    #[error("unsupported encoding: 0x{0:02x}")]
    UnsupportedEncoding(u8),

    #[error("invalid header format: 0x{0:02x}")]
    InvalidHeaderFormat(u8),

    #[error("batch decode error: {0}")]
    BatchDecodeError(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("authentication error: {0}")]
    AuthError(String),
}

/// Maximum payload size: 16 MiB.
pub const MAX_PAYLOAD_LEN: usize = 16 * 1024 * 1024;
