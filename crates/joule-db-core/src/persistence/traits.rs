//! Platform-agnostic persistence traits
//!
//! These traits define the interface for persistence across all platforms:
//! - Native (filesystem)
//! - Browser (IndexedDB/OPFS)
//! - Embedded (flash/EEPROM)
//!
//! ## Architecture
//!
//! ```text
//! +------------------+     +------------------+     +------------------+
//! |   Application    |     |   Application    |     |   Application    |
//! +--------+---------+     +--------+---------+     +--------+---------+
//!          |                        |                        |
//! +--------v---------+     +--------v---------+     +--------v---------+
//! | PersistenceLayer |     | PersistenceLayer |     | PersistenceLayer |
//! | (joule-db-core) |     | (joule-db-core) |     | (joule-db-core) |
//! +--------+---------+     +--------+---------+     +--------+---------+
//!          |                        |                        |
//! +--------v---------+     +--------v---------+     +--------v---------+
//! |   FileStorage    |     | IndexedDBStorage |     |   FlashStorage   |
//! |(joule-db-local) |     |(joule-db-browser)|     |(joule-db-embedded)|
//! +------------------+     +------------------+     +------------------+
//! ```

use crate::error::StorageError;
use crate::storage::PageId;

/// Log Sequence Number - monotonically increasing identifier for WAL entries
pub type LSN = u64;

/// Transaction identifier
pub type TxId = u64;

/// Durability policy for writes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DurabilityPolicy {
    /// Sync to durable storage on every commit (safest, slowest)
    #[default]
    SyncOnCommit,
    /// Sync periodically (balance of safety and performance)
    Periodic {
        /// Maximum time between syncs in milliseconds
        interval_ms: u32,
        /// Maximum number of commits before sync
        max_commits: u32,
    },
    /// Never explicitly sync - rely on OS/platform (fastest, least safe)
    NoSync,
    /// Group commits together (good for high throughput)
    GroupCommit {
        /// Maximum wait time in milliseconds
        max_wait_ms: u32,
        /// Maximum transactions to group
        max_group_size: u32,
    },
}

/// WAL entry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalEntryType {
    /// Page write operation
    PageWrite = 0x01,
    /// Transaction begin
    Begin = 0x02,
    /// Transaction commit
    Commit = 0x03,
    /// Transaction rollback/abort
    Rollback = 0x04,
    /// Checkpoint marker
    Checkpoint = 0x05,
    /// Savepoint
    Savepoint = 0x06,
    /// Savepoint release
    SavepointRelease = 0x07,
    /// Savepoint rollback
    SavepointRollback = 0x08,
}

impl TryFrom<u8> for WalEntryType {
    type Error = StorageError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(WalEntryType::PageWrite),
            0x02 => Ok(WalEntryType::Begin),
            0x03 => Ok(WalEntryType::Commit),
            0x04 => Ok(WalEntryType::Rollback),
            0x05 => Ok(WalEntryType::Checkpoint),
            0x06 => Ok(WalEntryType::Savepoint),
            0x07 => Ok(WalEntryType::SavepointRelease),
            0x08 => Ok(WalEntryType::SavepointRollback),
            _ => Err(StorageError::Backend(format!(
                "Invalid WAL entry type: {}",
                value
            ))),
        }
    }
}

/// WAL entry - platform-agnostic representation
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Log sequence number
    pub lsn: LSN,
    /// Entry type
    pub entry_type: WalEntryType,
    /// Transaction ID
    pub tx_id: TxId,
    /// Page ID (for PageWrite)
    pub page_id: Option<PageId>,
    /// Savepoint ID (for Savepoint operations)
    pub savepoint_id: Option<u32>,
    /// Data payload
    pub data: Vec<u8>,
    /// CRC32 checksum
    pub checksum: u32,
}

impl WalEntry {
    /// Create a page write entry
    pub fn page_write(lsn: LSN, tx_id: TxId, page_id: PageId, data: Vec<u8>) -> Self {
        let mut entry = Self {
            lsn,
            entry_type: WalEntryType::PageWrite,
            tx_id,
            page_id: Some(page_id),
            savepoint_id: None,
            data,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a begin entry
    pub fn begin(lsn: LSN, tx_id: TxId) -> Self {
        let mut entry = Self {
            lsn,
            entry_type: WalEntryType::Begin,
            tx_id,
            page_id: None,
            savepoint_id: None,
            data: Vec::new(),
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a commit entry
    pub fn commit(lsn: LSN, tx_id: TxId) -> Self {
        let mut entry = Self {
            lsn,
            entry_type: WalEntryType::Commit,
            tx_id,
            page_id: None,
            savepoint_id: None,
            data: Vec::new(),
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a rollback entry
    pub fn rollback(lsn: LSN, tx_id: TxId) -> Self {
        let mut entry = Self {
            lsn,
            entry_type: WalEntryType::Rollback,
            tx_id,
            page_id: None,
            savepoint_id: None,
            data: Vec::new(),
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a checkpoint entry
    pub fn checkpoint(lsn: LSN, active_tx_ids: Vec<TxId>) -> Self {
        // Encode active transaction IDs
        let mut data = Vec::with_capacity(active_tx_ids.len() * 8);
        for tx_id in active_tx_ids {
            data.extend_from_slice(&tx_id.to_le_bytes());
        }

        let mut entry = Self {
            lsn,
            entry_type: WalEntryType::Checkpoint,
            tx_id: 0,
            page_id: None,
            savepoint_id: None,
            data,
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Create a savepoint entry
    pub fn savepoint(lsn: LSN, tx_id: TxId, savepoint_id: u32) -> Self {
        let mut entry = Self {
            lsn,
            entry_type: WalEntryType::Savepoint,
            tx_id,
            page_id: None,
            savepoint_id: Some(savepoint_id),
            data: Vec::new(),
            checksum: 0,
        };
        entry.checksum = entry.compute_checksum();
        entry
    }

    /// Compute CRC32 checksum of entry (excluding checksum field)
    pub fn compute_checksum(&self) -> u32 {
        let mut data = Vec::new();
        data.extend_from_slice(&self.lsn.to_le_bytes());
        data.push(self.entry_type as u8);
        data.extend_from_slice(&self.tx_id.to_le_bytes());
        if let Some(page_id) = self.page_id {
            data.extend_from_slice(&page_id.to_le_bytes());
        }
        if let Some(savepoint_id) = self.savepoint_id {
            data.extend_from_slice(&savepoint_id.to_le_bytes());
        }
        data.extend_from_slice(&self.data);
        crc32(&data)
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.compute_checksum()
    }

    /// Encode entry to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic number
        buf.extend_from_slice(&WAL_MAGIC);

        // Checksum (4 bytes)
        buf.extend_from_slice(&self.checksum.to_le_bytes());

        // LSN (8 bytes)
        buf.extend_from_slice(&self.lsn.to_le_bytes());

        // Entry type (1 byte)
        buf.push(self.entry_type as u8);

        // Transaction ID (8 bytes)
        buf.extend_from_slice(&self.tx_id.to_le_bytes());

        // Flags (1 byte): has_page_id | has_savepoint_id
        let flags = (self.page_id.is_some() as u8) | ((self.savepoint_id.is_some() as u8) << 1);
        buf.push(flags);

        // Optional page_id (8 bytes)
        if let Some(page_id) = self.page_id {
            buf.extend_from_slice(&page_id.to_le_bytes());
        }

        // Optional savepoint_id (4 bytes)
        if let Some(savepoint_id) = self.savepoint_id {
            buf.extend_from_slice(&savepoint_id.to_le_bytes());
        }

        // Data length (4 bytes)
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());

        // Data
        buf.extend_from_slice(&self.data);

        buf
    }

    /// Decode entry from bytes
    pub fn decode(buf: &[u8]) -> Result<Self, StorageError> {
        if buf.len() < WAL_HEADER_MIN_SIZE {
            return Err(StorageError::Backend("WAL entry too short".to_string()));
        }

        // Verify magic
        if &buf[0..4] != &WAL_MAGIC {
            return Err(StorageError::Backend("Invalid WAL magic".to_string()));
        }

        let mut offset = 4;

        // Checksum
        let checksum = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]);
        offset += 4;

        // LSN
        let lsn = u64::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
            buf[offset + 4],
            buf[offset + 5],
            buf[offset + 6],
            buf[offset + 7],
        ]);
        offset += 8;

        // Entry type
        let entry_type = WalEntryType::try_from(buf[offset])?;
        offset += 1;

        // Transaction ID
        let tx_id = u64::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
            buf[offset + 4],
            buf[offset + 5],
            buf[offset + 6],
            buf[offset + 7],
        ]);
        offset += 8;

        // Flags
        let flags = buf[offset];
        offset += 1;
        let has_page_id = (flags & 0x01) != 0;
        let has_savepoint_id = (flags & 0x02) != 0;

        // Optional page_id
        let page_id = if has_page_id {
            let id = u64::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
                buf[offset + 4],
                buf[offset + 5],
                buf[offset + 6],
                buf[offset + 7],
            ]);
            offset += 8;
            Some(id)
        } else {
            None
        };

        // Optional savepoint_id
        let savepoint_id = if has_savepoint_id {
            let id = u32::from_le_bytes([
                buf[offset],
                buf[offset + 1],
                buf[offset + 2],
                buf[offset + 3],
            ]);
            offset += 4;
            Some(id)
        } else {
            None
        };

        // Data length
        let data_len = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]) as usize;
        offset += 4;

        // Data
        if buf.len() < offset + data_len {
            return Err(StorageError::Backend("WAL entry truncated".to_string()));
        }
        let data = buf[offset..offset + data_len].to_vec();

        let entry = Self {
            lsn,
            entry_type,
            tx_id,
            page_id,
            savepoint_id,
            data,
            checksum,
        };

        // Verify checksum
        if !entry.verify_checksum() {
            return Err(StorageError::Backend("WAL checksum mismatch".to_string()));
        }

        Ok(entry)
    }

    /// Get the encoded size of this entry
    pub fn encoded_size(&self) -> usize {
        WAL_HEADER_MIN_SIZE
            + if self.page_id.is_some() { 8 } else { 0 }
            + if self.savepoint_id.is_some() { 4 } else { 0 }
            + self.data.len()
    }
}

/// WAL magic number: "WDB1"
const WAL_MAGIC: [u8; 4] = [0x57, 0x44, 0x42, 0x31];

/// Minimum WAL header size: magic(4) + checksum(4) + lsn(8) + type(1) + tx_id(8) + flags(1) + data_len(4)
const WAL_HEADER_MIN_SIZE: usize = 30;

/// Write-ahead log backend trait
///
/// Platform-specific implementations provide the actual storage.
pub trait WalBackend: Send + Sync {
    /// Append an entry to the WAL
    fn append(&mut self, entry: &WalEntry) -> Result<LSN, StorageError>;

    /// Sync WAL to durable storage
    fn sync(&mut self) -> Result<(), StorageError>;

    /// Read all entries from the WAL
    fn read_all(&self) -> Result<Vec<WalEntry>, StorageError>;

    /// Read entries after a specific LSN
    fn read_after(&self, lsn: LSN) -> Result<Vec<WalEntry>, StorageError>;

    /// Truncate WAL up to (and including) a specific LSN
    fn truncate_before(&mut self, lsn: LSN) -> Result<(), StorageError>;

    /// Get the current (next) LSN
    fn current_lsn(&self) -> LSN;

    /// Get the last checkpoint LSN
    fn last_checkpoint_lsn(&self) -> LSN;

    /// Get WAL size in bytes
    fn size_bytes(&self) -> u64;
}

/// Snapshot metadata
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    /// Snapshot ID
    pub id: u64,
    /// Creation timestamp (Unix millis)
    pub created_at: u64,
    /// LSN at snapshot time
    pub lsn: LSN,
    /// Number of pages in snapshot
    pub page_count: u64,
    /// Snapshot size in bytes
    pub size_bytes: u64,
    /// Optional description
    pub description: Option<String>,
}

/// Snapshot backend trait
///
/// Provides point-in-time snapshots for backup and recovery.
pub trait SnapshotBackend: Send + Sync {
    /// Create a new snapshot
    fn create_snapshot(
        &mut self,
        description: Option<String>,
    ) -> Result<SnapshotMetadata, StorageError>;

    /// List all snapshots
    fn list_snapshots(&self) -> Result<Vec<SnapshotMetadata>, StorageError>;

    /// Delete a snapshot
    fn delete_snapshot(&mut self, id: u64) -> Result<(), StorageError>;

    /// Restore from a snapshot
    fn restore_snapshot(&mut self, id: u64) -> Result<(), StorageError>;

    /// Export snapshot to bytes
    fn export_snapshot(&self, id: u64) -> Result<Vec<u8>, StorageError>;

    /// Import snapshot from bytes
    fn import_snapshot(&mut self, data: &[u8]) -> Result<SnapshotMetadata, StorageError>;
}

/// Recovery result
#[derive(Debug, Default)]
pub struct RecoveryResult {
    /// Pages that need to be applied
    pub pages_to_apply: Vec<(PageId, Vec<u8>)>,
    /// Committed transaction IDs
    pub committed_transactions: Vec<TxId>,
    /// Rolled back transaction IDs
    pub rolled_back_transactions: Vec<TxId>,
    /// Uncommitted transactions (need rollback)
    pub uncommitted_transactions: Vec<TxId>,
    /// Last LSN processed
    pub last_lsn: LSN,
    /// Recovery took this many milliseconds
    pub duration_ms: u64,
}

/// Recovery strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecoveryStrategy {
    /// Apply all committed transactions (standard ARIES-style)
    #[default]
    Standard,
    /// Also include uncommitted transactions (for debugging)
    IncludeUncommitted,
    /// Recover to a specific point in time
    PointInTime {
        /// Target LSN
        target_lsn: LSN,
    },
}

/// Persistence configuration
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Durability policy
    pub durability: DurabilityPolicy,
    /// Maximum WAL size before automatic checkpoint (bytes)
    pub max_wal_size: u64,
    /// Checkpoint interval in milliseconds (0 = disabled)
    pub checkpoint_interval_ms: u64,
    /// Enable snapshots
    pub snapshots_enabled: bool,
    /// Maximum number of snapshots to retain
    pub max_snapshots: u32,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            durability: DurabilityPolicy::SyncOnCommit,
            max_wal_size: 64 * 1024 * 1024, // 64MB
            checkpoint_interval_ms: 60_000, // 1 minute
            snapshots_enabled: true,
            max_snapshots: 10,
            recovery_strategy: RecoveryStrategy::Standard,
        }
    }
}

/// Unified persistence manager trait
///
/// Combines WAL, snapshots, and recovery into a single interface.
pub trait PersistenceManager: Send + Sync {
    /// Get the WAL backend
    fn wal(&self) -> &dyn WalBackend;

    /// Get the WAL backend mutably
    fn wal_mut(&mut self) -> &mut dyn WalBackend;

    /// Get the snapshot backend (if available)
    fn snapshots(&self) -> Option<&dyn SnapshotBackend>;

    /// Get the snapshot backend mutably (if available)
    fn snapshots_mut(&mut self) -> Option<&mut dyn SnapshotBackend>;

    /// Run crash recovery
    fn recover(&mut self) -> Result<RecoveryResult, StorageError>;

    /// Create a checkpoint
    fn checkpoint(&mut self) -> Result<LSN, StorageError>;

    /// Get current configuration
    fn config(&self) -> &PersistenceConfig;

    /// Update configuration
    fn set_config(&mut self, config: PersistenceConfig);
}

/// CRC32 calculation (IEEE polynomial)
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_entry_encode_decode() {
        let entry = WalEntry::page_write(1, 100, 42, b"test data".to_vec());
        let encoded = entry.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.lsn, 1);
        assert_eq!(decoded.tx_id, 100);
        assert_eq!(decoded.page_id, Some(42));
        assert_eq!(decoded.data, b"test data");
        assert!(decoded.verify_checksum());
    }

    #[test]
    fn test_wal_entry_types() {
        let begin = WalEntry::begin(1, 100);
        let encoded = begin.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.entry_type, WalEntryType::Begin);

        let commit = WalEntry::commit(2, 100);
        let encoded = commit.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.entry_type, WalEntryType::Commit);

        let rollback = WalEntry::rollback(3, 100);
        let encoded = rollback.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.entry_type, WalEntryType::Rollback);

        let checkpoint = WalEntry::checkpoint(4, vec![100, 101, 102]);
        let encoded = checkpoint.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();
        assert_eq!(decoded.entry_type, WalEntryType::Checkpoint);
        assert_eq!(decoded.data.len(), 24); // 3 * 8 bytes
    }

    #[test]
    fn test_wal_entry_savepoint() {
        let savepoint = WalEntry::savepoint(5, 100, 1);
        let encoded = savepoint.encode();
        let decoded = WalEntry::decode(&encoded).unwrap();

        assert_eq!(decoded.entry_type, WalEntryType::Savepoint);
        assert_eq!(decoded.savepoint_id, Some(1));
    }

    #[test]
    fn test_crc32() {
        // Known test vector
        let data = b"123456789";
        let crc = crc32(data);
        assert_eq!(crc, 0xCBF43926);
    }

    #[test]
    fn test_durability_policy_default() {
        let policy = DurabilityPolicy::default();
        assert_eq!(policy, DurabilityPolicy::SyncOnCommit);
    }

    #[test]
    fn test_persistence_config_default() {
        let config = PersistenceConfig::default();
        assert_eq!(config.durability, DurabilityPolicy::SyncOnCommit);
        assert_eq!(config.max_wal_size, 64 * 1024 * 1024);
        assert!(config.snapshots_enabled);
    }
}
