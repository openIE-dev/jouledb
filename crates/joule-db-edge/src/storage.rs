//! Edge-optimized storage backend
//!
//! Storage implementation designed for edge devices with:
//! - Flash-friendly write patterns
//! - Power-loss safe operations
//! - Minimal memory footprint

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

use crate::config::EdgeConfig;
use crate::memory::AlignedBuffer;

/// Storage statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes read
    pub bytes_read: u64,
    /// Number of write operations
    pub write_ops: u64,
    /// Number of read operations
    pub read_ops: u64,
    /// Number of sync operations
    pub sync_ops: u64,
    /// Write amplification factor
    pub write_amplification: f32,
}

/// Storage error
#[derive(Debug, Clone)]
pub enum StorageError {
    /// I/O error
    Io(String),
    /// I/O error (legacy alias)
    IoError(String),
    /// Corruption detected
    Corruption(String),
    /// Corrupted data
    Corrupted(String),
    /// Out of space
    OutOfSpace,
    /// No space available
    NoSpace(String),
    /// Invalid operation
    InvalidOperation(String),
    /// Page/Key not found
    NotFound(String),
    /// Data exceeds maximum size
    DataTooLarge,
    /// Bad block (flash)
    BadBlock,
    /// Out of bounds access
    OutOfBounds,
}

impl core::fmt::Display for StorageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(s) | Self::IoError(s) => write!(f, "I/O error: {}", s),
            Self::Corruption(s) | Self::Corrupted(s) => write!(f, "Corruption: {}", s),
            Self::OutOfSpace => write!(f, "Out of storage space"),
            Self::NoSpace(s) => write!(f, "No space: {}", s),
            Self::InvalidOperation(s) => write!(f, "Invalid operation: {}", s),
            Self::NotFound(s) => write!(f, "Not found: {}", s),
            Self::DataTooLarge => write!(f, "Data exceeds maximum size"),
            Self::BadBlock => write!(f, "Bad block"),
            Self::OutOfBounds => write!(f, "Out of bounds"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StorageError {}

/// Write-ahead log entry
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Log sequence number
    pub lsn: u64,
    /// Entry type
    pub entry_type: WalEntryType,
    /// Key
    pub key: Vec<u8>,
    /// Value (for Put operations)
    pub value: Option<Vec<u8>>,
    /// CRC32 checksum
    pub checksum: u32,
}

/// WAL entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    /// Begin transaction
    Begin = 1,
    /// Put operation
    Put = 2,
    /// Delete operation
    Delete = 3,
    /// Commit transaction
    Commit = 4,
    /// Rollback transaction
    Rollback = 5,
    /// Checkpoint marker
    Checkpoint = 6,
}

impl WalEntry {
    /// Encode entry to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // LSN (8 bytes)
        buf.extend_from_slice(&self.lsn.to_le_bytes());

        // Entry type (1 byte)
        buf.push(self.entry_type as u8);

        // Key length + key
        buf.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.key);

        // Value length + value (if present)
        if let Some(ref value) = self.value {
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            buf.extend_from_slice(value);
        } else {
            buf.extend_from_slice(&0u32.to_le_bytes());
        }

        // CRC32 (4 bytes) - computed over all previous bytes
        let crc = crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Decode entry from bytes
    pub fn decode(data: &[u8]) -> Result<Self, StorageError> {
        if data.len() < 17 {
            return Err(StorageError::Corruption("Entry too short".into()));
        }

        // Verify CRC first
        let stored_crc = u32::from_le_bytes([
            data[data.len() - 4],
            data[data.len() - 3],
            data[data.len() - 2],
            data[data.len() - 1],
        ]);
        let computed_crc = crc32(&data[..data.len() - 4]);
        if stored_crc != computed_crc {
            return Err(StorageError::Corruption("CRC mismatch".into()));
        }

        let lsn = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);

        let entry_type = match data[8] {
            1 => WalEntryType::Begin,
            2 => WalEntryType::Put,
            3 => WalEntryType::Delete,
            4 => WalEntryType::Commit,
            5 => WalEntryType::Rollback,
            6 => WalEntryType::Checkpoint,
            _ => return Err(StorageError::Corruption("Invalid entry type".into())),
        };

        let key_len = u32::from_le_bytes([data[9], data[10], data[11], data[12]]) as usize;
        let key = data[13..13 + key_len].to_vec();

        let value_offset = 13 + key_len;
        let value_len = u32::from_le_bytes([
            data[value_offset],
            data[value_offset + 1],
            data[value_offset + 2],
            data[value_offset + 3],
        ]) as usize;

        let value = if value_len > 0 {
            Some(data[value_offset + 4..value_offset + 4 + value_len].to_vec())
        } else {
            None
        };

        Ok(Self {
            lsn,
            entry_type,
            key,
            value,
            checksum: stored_crc,
        })
    }
}

/// Flash-aware page layout
///
/// Designed to minimize write amplification on flash storage:
/// - Aligned to erase block boundaries
/// - Append-only within page
/// - Version tracking for wear leveling
#[derive(Debug)]
pub struct FlashPage {
    /// Page ID
    pub id: u64,
    /// Page version (for wear leveling)
    pub version: u32,
    /// Data
    pub data: AlignedBuffer,
    /// Number of bytes used
    pub used: usize,
    /// Dirty flag
    pub dirty: bool,
}

impl FlashPage {
    /// Create a new flash page
    pub fn new(id: u64, size: usize) -> Self {
        Self {
            id,
            version: 0,
            data: AlignedBuffer::new_page_aligned(size),
            used: 0,
            dirty: false,
        }
    }

    /// Remaining space in page
    pub fn remaining(&self) -> usize {
        self.data.len() - self.used
    }

    /// Append data to page
    pub fn append(&mut self, data: &[u8]) -> Result<(), StorageError> {
        if data.len() > self.remaining() {
            return Err(StorageError::OutOfSpace);
        }
        self.data[self.used..self.used + data.len()].copy_from_slice(data);
        self.used += data.len();
        self.dirty = true;
        Ok(())
    }

    /// Clear page for reuse
    pub fn clear(&mut self) {
        self.version += 1;
        self.used = 0;
        self.dirty = true;
        // Zero out data for security
        for byte in self.data.as_mut_slice() {
            *byte = 0;
        }
    }
}

/// Edge storage configuration
#[derive(Debug, Clone)]
pub struct EdgeStorageConfig {
    /// Page size (should match flash erase block)
    pub page_size: usize,
    /// Number of pages to cache
    pub cache_pages: usize,
    /// Enable WAL
    pub enable_wal: bool,
    /// WAL buffer size
    pub wal_buffer_size: usize,
    /// Path to WAL file
    #[cfg(feature = "std")]
    pub wal_path: String,
    /// Path to data file
    #[cfg(feature = "std")]
    pub data_path: String,
}

impl Default for EdgeStorageConfig {
    fn default() -> Self {
        Self {
            page_size: 4096,
            cache_pages: 256,
            enable_wal: true,
            wal_buffer_size: 64 * 1024, // 64KB
            #[cfg(feature = "std")]
            wal_path: "edge_db.wal".to_string(),
            #[cfg(feature = "std")]
            data_path: "edge_db.data".to_string(),
        }
    }
}

impl From<&EdgeConfig> for EdgeStorageConfig {
    fn from(config: &EdgeConfig) -> Self {
        Self {
            page_size: config.page_size,
            cache_pages: config.cache_pages,
            enable_wal: config.enable_wal,
            wal_buffer_size: 64 * 1024,
            #[cfg(feature = "std")]
            wal_path: "edge_db.wal".to_string(),
            #[cfg(feature = "std")]
            data_path: "edge_db.data".to_string(),
        }
    }
}

/// CRC32 implementation (for checksums)
fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = crc32_table();
    let mut crc = 0xFFFFFFFF;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                0xEDB88320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_encode_decode() {
        let entry = WalEntry {
            lsn: 42,
            entry_type: WalEntryType::Put,
            key: b"test_key".to_vec(),
            value: Some(b"test_value".to_vec()),
            checksum: 0, // Will be computed
        };

        let encoded = entry.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.lsn, 42);
        assert_eq!(decoded.entry_type, WalEntryType::Put);
        assert_eq!(decoded.key, b"test_key");
        assert_eq!(decoded.value, Some(b"test_value".to_vec()));
    }

    #[test]
    fn test_wal_entry_crc_validation() {
        let entry = WalEntry {
            lsn: 1,
            entry_type: WalEntryType::Delete,
            key: b"key".to_vec(),
            value: None,
            checksum: 0,
        };

        let mut encoded = entry.encode();
        // Corrupt a byte
        encoded[5] ^= 0xFF;

        assert!(WalEntry::decode(&encoded).is_err());
    }

    #[test]
    fn test_flash_page() {
        let mut page = FlashPage::new(1, 4096);
        assert_eq!(page.remaining(), 4096);

        page.append(b"hello").unwrap();
        assert_eq!(page.used, 5);
        assert_eq!(page.remaining(), 4091);
        assert!(page.dirty);

        page.clear();
        assert_eq!(page.used, 0);
        assert_eq!(page.version, 1);
    }

    #[test]
    fn test_flash_page_overflow() {
        let mut page = FlashPage::new(1, 16);
        assert!(page.append(b"short").is_ok());
        assert!(page.append(b"this is too long").is_err());
    }

    #[test]
    fn test_crc32() {
        let crc1 = crc32(b"hello");
        let crc2 = crc32(b"world");
        let crc3 = crc32(b"hello");

        assert_ne!(crc1, crc2);
        assert_eq!(crc1, crc3);
    }
}
