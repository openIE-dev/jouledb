//! Persistence error types

use joule_db_core::storage::PageId;

/// Persistence error types
#[derive(Debug, Clone)]
pub enum PersistenceError {
    /// Storage not available
    NotAvailable(String),
    /// Storage quota exceeded
    QuotaExceeded,
    /// Page not found
    PageNotFound(PageId),
    /// Checksum mismatch (data corruption)
    ChecksumMismatch {
        page_id: PageId,
        expected: u32,
        actual: u32,
    },
    /// Hash mismatch (SHA-256 integrity failure)
    HashMismatch {
        page_id: PageId,
        expected: Vec<u8>,
        actual: Vec<u8>,
    },
    /// Transaction error
    TransactionError(String),
    /// I/O error
    IoError(String),
    /// Initialization error
    InitError(String),
    /// WAL recovery error
    WalRecoveryError(String),
    /// Encryption error
    EncryptionError(String),
    /// Compression error
    CompressionError(String),
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable(msg) => write!(f, "Storage not available: {}", msg),
            Self::QuotaExceeded => write!(f, "Storage quota exceeded"),
            Self::PageNotFound(id) => write!(f, "Page {} not found", id),
            Self::ChecksumMismatch {
                page_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Checksum mismatch for page {}: expected {:08x}, got {:08x}",
                    page_id, expected, actual
                )
            }
            Self::HashMismatch {
                page_id,
                expected,
                actual,
            } => {
                let expected_hex: String = expected.iter().map(|b| format!("{:02x}", b)).collect();
                let actual_hex: String = actual.iter().map(|b| format!("{:02x}", b)).collect();
                write!(
                    f,
                    "Hash mismatch for page {}: expected {}, got {}",
                    page_id, expected_hex, actual_hex
                )
            }
            Self::TransactionError(msg) => write!(f, "Transaction error: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::InitError(msg) => write!(f, "Initialization error: {}", msg),
            Self::WalRecoveryError(msg) => write!(f, "WAL recovery error: {}", msg),
            Self::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            Self::CompressionError(msg) => write!(f, "Compression error: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {}

impl From<PersistenceError> for String {
    fn from(e: PersistenceError) -> String {
        e.to_string()
    }
}

#[cfg(target_arch = "wasm32")]
impl From<PersistenceError> for wasm_bindgen::JsValue {
    fn from(e: PersistenceError) -> wasm_bindgen::JsValue {
        wasm_bindgen::JsValue::from_str(&e.to_string())
    }
}
