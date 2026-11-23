//! Edge database implementation
//!
//! A key-value database optimized for edge devices with hardware acceleration.

#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use crate::accelerator::{Accelerator, CpuAccelerator, HashAlgorithm};
use crate::config::EdgeConfig;
use crate::memory::BufferPool;
use crate::storage::{EdgeStorageConfig, StorageError, WalEntry, WalEntryType};

/// Edge database
///
/// A lightweight key-value store designed for edge devices.
/// Supports hardware-accelerated operations when available.
pub struct EdgeDatabase {
    /// Configuration
    config: EdgeConfig,
    /// In-memory data (B-tree for ordered access)
    data: BTreeMap<Vec<u8>, Vec<u8>>,
    /// Hardware accelerator
    accelerator: Box<dyn Accelerator>,
    /// Buffer pool for I/O
    buffer_pool: BufferPool,
    /// Current LSN
    current_lsn: u64,
    /// Pending WAL entries
    wal_buffer: Vec<WalEntry>,
    /// Statistics
    stats: DatabaseStats,
    /// Storage configuration
    _storage_config: EdgeStorageConfig,
}

/// Database statistics
#[derive(Debug, Clone, Default)]
pub struct DatabaseStats {
    /// Number of keys
    pub key_count: u64,
    /// Total key bytes
    pub key_bytes: u64,
    /// Total value bytes
    pub value_bytes: u64,
    /// Get operations
    pub get_ops: u64,
    /// Put operations
    pub put_ops: u64,
    /// Delete operations
    pub delete_ops: u64,
    /// Accelerated operations
    pub accelerated_ops: u64,
}

/// Database error
#[derive(Debug, Clone)]
pub enum DatabaseError {
    /// Storage error
    Storage(StorageError),
    /// Key not found
    NotFound,
    /// Invalid key
    InvalidKey(String),
    /// Configuration error
    Config(String),
}

impl core::fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Storage(e) => write!(f, "Storage error: {}", e),
            Self::NotFound => write!(f, "Key not found"),
            Self::InvalidKey(s) => write!(f, "Invalid key: {}", s),
            Self::Config(s) => write!(f, "Configuration error: {}", s),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DatabaseError {}

impl From<StorageError> for DatabaseError {
    fn from(e: StorageError) -> Self {
        Self::Storage(e)
    }
}

impl EdgeDatabase {
    /// Open or create a database with default configuration
    pub fn open(_path: &str) -> Result<Self, DatabaseError> {
        Self::with_config(_path, EdgeConfig::auto_detect())
    }

    /// Open or create a database with custom configuration
    pub fn with_config(path: &str, config: EdgeConfig) -> Result<Self, DatabaseError> {
        config
            .validate()
            .map_err(|e| DatabaseError::Config(e.to_string()))?;

        let mut storage_config = EdgeStorageConfig::from(&config);

        // Set paths based on provided path
        #[cfg(feature = "std")]
        {
            if !path.is_empty() {
                storage_config.data_path = format!("{}.data", path);
                storage_config.wal_path = format!("{}.wal", path);
            }
        }

        let buffer_pool = BufferPool::new(
            config.page_size,
            64, // Cache line alignment
            config.cache_pages / 4,
            config.cache_pages,
        );

        let mut db = Self {
            config,
            data: BTreeMap::new(),
            accelerator: Box::new(CpuAccelerator::new()),
            buffer_pool,
            current_lsn: 0,
            wal_buffer: Vec::new(),
            stats: DatabaseStats::default(),
            _storage_config: storage_config,
        };

        // Load from storage if path is provided
        if !path.is_empty() {
            db.load_from_storage()?;
        }

        Ok(db)
    }

    /// Load database from storage (data file + WAL replay)
    fn load_from_storage(&mut self) -> Result<(), DatabaseError> {
        #[cfg(feature = "std")]
        {
            use std::path::PathBuf;

            // Clone paths to avoid borrow issues
            let data_path = PathBuf::from(&self._storage_config.data_path);
            let wal_path = PathBuf::from(&self._storage_config.wal_path);

            // Load data file if it exists
            if data_path.exists() {
                self.load_data_file(&data_path)?;
            }

            // Replay WAL if it exists
            if wal_path.exists() {
                self.replay_wal(&wal_path)?;
            }
        }

        Ok(())
    }

    /// Load data from the main data file
    #[cfg(feature = "std")]
    fn load_data_file(&mut self, path: &std::path::Path) -> Result<(), DatabaseError> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let file = File::open(path)
            .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;
        let mut reader = BufReader::new(file);

        // Read header: key count
        let mut count_buf = [0u8; 8];
        if reader.read_exact(&mut count_buf).is_err() {
            // Empty file, nothing to load
            return Ok(());
        }
        let key_count = u64::from_le_bytes(count_buf);

        // Read each key-value pair
        for _ in 0..key_count {
            // Read key length
            let mut len_buf = [0u8; 4];
            reader
                .read_exact(&mut len_buf)
                .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;
            let key_len = u32::from_le_bytes(len_buf) as usize;

            // Read key
            let mut key = vec![0u8; key_len];
            reader
                .read_exact(&mut key)
                .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

            // Read value length
            reader
                .read_exact(&mut len_buf)
                .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;
            let value_len = u32::from_le_bytes(len_buf) as usize;

            // Read value
            let mut value = vec![0u8; value_len];
            reader
                .read_exact(&mut value)
                .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

            // Insert into data map
            self.stats.key_count += 1;
            self.stats.key_bytes += key.len() as u64;
            self.stats.value_bytes += value.len() as u64;
            self.data.insert(key, value);
        }

        Ok(())
    }

    /// Replay write-ahead log to recover uncommitted transactions
    #[cfg(feature = "std")]
    fn replay_wal(&mut self, path: &std::path::Path) -> Result<(), DatabaseError> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let file = File::open(path)
            .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;
        let mut reader = BufReader::new(file);

        // Read WAL entries until EOF
        let mut buffer = Vec::new();
        reader
            .read_to_end(&mut buffer)
            .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

        let mut offset = 0;
        let mut max_lsn = self.current_lsn;

        while offset < buffer.len() {
            // Try to decode entry
            // Minimum entry size: 8 (lsn) + 1 (type) + 4 (key len) + 4 (value len) + 4 (crc) = 21
            if buffer.len() - offset < 21 {
                break;
            }

            // Find entry length by reading key and value lengths
            let key_len = u32::from_le_bytes([
                buffer[offset + 9],
                buffer[offset + 10],
                buffer[offset + 11],
                buffer[offset + 12],
            ]) as usize;

            let value_len_offset = offset + 13 + key_len;
            if value_len_offset + 4 > buffer.len() {
                break;
            }

            let value_len = u32::from_le_bytes([
                buffer[value_len_offset],
                buffer[value_len_offset + 1],
                buffer[value_len_offset + 2],
                buffer[value_len_offset + 3],
            ]) as usize;

            let entry_len = 13 + key_len + 4 + value_len + 4; // header + key + value_len + value + crc

            if offset + entry_len > buffer.len() {
                break;
            }

            // Decode entry
            match WalEntry::decode(&buffer[offset..offset + entry_len]) {
                Ok(entry) => {
                    max_lsn = max_lsn.max(entry.lsn);

                    // Apply entry to database
                    match entry.entry_type {
                        WalEntryType::Put => {
                            if let Some(value) = entry.value {
                                if !self.data.contains_key(&entry.key) {
                                    self.stats.key_count += 1;
                                    self.stats.key_bytes += entry.key.len() as u64;
                                }
                                self.stats.value_bytes += value.len() as u64;
                                self.data.insert(entry.key, value);
                            }
                        }
                        WalEntryType::Delete => {
                            if let Some(old) = self.data.remove(&entry.key) {
                                self.stats.key_count -= 1;
                                self.stats.key_bytes -= entry.key.len() as u64;
                                self.stats.value_bytes -= old.len() as u64;
                            }
                        }
                        WalEntryType::Checkpoint => {
                            // Checkpoint marker - no action needed during replay
                        }
                        _ => {
                            // Begin, Commit, Rollback - handled by transaction layer
                        }
                    }

                    offset += entry_len;
                }
                Err(_) => {
                    // Corrupted entry, stop replay
                    break;
                }
            }
        }

        // Update LSN to continue from where we left off
        self.current_lsn = max_lsn;

        Ok(())
    }

    /// Get a value by key
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, DatabaseError> {
        if key.is_empty() {
            return Err(DatabaseError::InvalidKey("Empty key".into()));
        }

        // Update stats (would need interior mutability in real impl)
        // self.stats.get_ops += 1;

        Ok(self.data.get(key).cloned())
    }

    /// Put a key-value pair
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<(), DatabaseError> {
        if key.is_empty() {
            return Err(DatabaseError::InvalidKey("Empty key".into()));
        }

        // Create WAL entry
        if self.config.enable_wal {
            self.current_lsn += 1;
            let entry = WalEntry {
                lsn: self.current_lsn,
                entry_type: WalEntryType::Put,
                key: key.to_vec(),
                value: Some(value.to_vec()),
                checksum: 0,
            };
            self.wal_buffer.push(entry);
        }

        // Update in-memory data
        let key_vec = key.to_vec();
        let value_vec = value.to_vec();

        if !self.data.contains_key(&key_vec) {
            self.stats.key_count += 1;
            self.stats.key_bytes += key.len() as u64;
        } else {
            // Subtract old value size
            if let Some(old) = self.data.get(&key_vec) {
                self.stats.value_bytes -= old.len() as u64;
            }
        }

        self.stats.value_bytes += value.len() as u64;
        self.stats.put_ops += 1;

        self.data.insert(key_vec, value_vec);

        Ok(())
    }

    /// Delete a key
    pub fn delete(&mut self, key: &[u8]) -> Result<bool, DatabaseError> {
        if key.is_empty() {
            return Err(DatabaseError::InvalidKey("Empty key".into()));
        }

        // Create WAL entry
        if self.config.enable_wal {
            self.current_lsn += 1;
            let entry = WalEntry {
                lsn: self.current_lsn,
                entry_type: WalEntryType::Delete,
                key: key.to_vec(),
                value: None,
                checksum: 0,
            };
            self.wal_buffer.push(entry);
        }

        self.stats.delete_ops += 1;

        if let Some(old) = self.data.remove(key) {
            self.stats.key_count -= 1;
            self.stats.key_bytes -= key.len() as u64;
            self.stats.value_bytes -= old.len() as u64;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if a key exists
    pub fn contains(&self, key: &[u8]) -> bool {
        self.data.contains_key(key)
    }

    /// Get the number of keys
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over keys in range
    pub fn range(&self, start: &[u8], end: &[u8]) -> impl Iterator<Item = (&Vec<u8>, &Vec<u8>)> {
        self.data.range(start.to_vec()..end.to_vec())
    }

    /// Get all keys with a prefix
    pub fn prefix_scan(&self, prefix: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
        self.data
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Batch put multiple key-value pairs
    pub fn put_batch(&mut self, pairs: &[(&[u8], &[u8])]) -> Result<(), DatabaseError> {
        for (key, value) in pairs {
            self.put(key, value)?;
        }
        Ok(())
    }

    /// Batch get multiple keys
    pub fn get_batch(&self, keys: &[&[u8]]) -> Vec<Option<Vec<u8>>> {
        keys.iter().map(|k| self.data.get(*k).cloned()).collect()
    }

    /// Compute hash of a value using hardware acceleration
    pub fn hash(&self, data: &[u8]) -> u64 {
        let hashes = self
            .accelerator
            .hash_batch(HashAlgorithm::Xxhash64, &[data]);
        hashes[0]
    }

    /// Compute hashes of multiple values using hardware acceleration
    pub fn hash_batch(&self, data: &[&[u8]]) -> Vec<u64> {
        self.accelerator.hash_batch(HashAlgorithm::Xxhash64, data)
    }

    /// Sync WAL to storage
    pub fn sync(&mut self) -> Result<(), DatabaseError> {
        #[cfg(feature = "std")]
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            use std::path::Path;

            // Write WAL entries to file
            if !self.wal_buffer.is_empty() {
                let wal_path = Path::new(&self._storage_config.wal_path);

                // Open WAL file in append mode, create if doesn't exist
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(wal_path)
                    .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

                // Write each WAL entry
                for entry in &self.wal_buffer {
                    let encoded = entry.encode();
                    file.write_all(&encoded).map_err(|e| {
                        DatabaseError::Storage(StorageError::IoError(e.to_string()))
                    })?;
                }

                // Sync to disk
                file.sync_all()
                    .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

                // Clear the buffer after successful write
                self.wal_buffer.clear();
            }
        }

        #[cfg(not(feature = "std"))]
        {
            // In no_std mode, just clear the buffer
            // Actual persistence would be handled by platform-specific implementation
            self.wal_buffer.clear();
        }

        Ok(())
    }

    /// Create a checkpoint
    pub fn checkpoint(&mut self) -> Result<(), DatabaseError> {
        if self.config.enable_wal {
            self.current_lsn += 1;
            let entry = WalEntry {
                lsn: self.current_lsn,
                entry_type: WalEntryType::Checkpoint,
                key: Vec::new(),
                value: None,
                checksum: 0,
            };
            self.wal_buffer.push(entry);
        }
        self.sync()
    }

    /// Get database statistics
    pub fn stats(&self) -> &DatabaseStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &EdgeConfig {
        &self.config
    }

    /// Compact the database
    pub fn compact(&mut self) -> Result<(), DatabaseError> {
        #[cfg(feature = "std")]
        {
            use std::fs::{OpenOptions, rename};
            use std::io::{BufWriter, Write};
            use std::path::Path;

            // Compaction: rewrite data file with only current values
            let data_path = Path::new(&self._storage_config.data_path);
            let tmp_path = data_path.with_extension("tmp");

            // Write current state to temp file
            {
                let file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp_path)
                    .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

                let mut writer = BufWriter::new(file);

                // Write header: key count
                let key_count = self.data.len() as u64;
                writer
                    .write_all(&key_count.to_le_bytes())
                    .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

                // Write each key-value pair
                for (key, value) in &self.data {
                    // Key length and data
                    let key_len = key.len() as u32;
                    writer.write_all(&key_len.to_le_bytes()).map_err(|e| {
                        DatabaseError::Storage(StorageError::IoError(e.to_string()))
                    })?;
                    writer.write_all(key).map_err(|e| {
                        DatabaseError::Storage(StorageError::IoError(e.to_string()))
                    })?;

                    // Value length and data
                    let value_len = value.len() as u32;
                    writer.write_all(&value_len.to_le_bytes()).map_err(|e| {
                        DatabaseError::Storage(StorageError::IoError(e.to_string()))
                    })?;
                    writer.write_all(value).map_err(|e| {
                        DatabaseError::Storage(StorageError::IoError(e.to_string()))
                    })?;
                }

                writer
                    .flush()
                    .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;
            }

            // Atomic rename temp to data file
            rename(&tmp_path, data_path)
                .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;

            // Truncate WAL since we've persisted all data
            let wal_path = Path::new(&self._storage_config.wal_path);
            if wal_path.exists() {
                OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(wal_path)
                    .map_err(|e| DatabaseError::Storage(StorageError::IoError(e.to_string())))?;
            }

            // Reset LSN counter after compaction
            self.current_lsn = 0;
        }

        Ok(())
    }

    /// Close the database
    pub fn close(mut self) -> Result<(), DatabaseError> {
        self.sync()?;
        self.buffer_pool.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db() -> EdgeDatabase {
        EdgeDatabase::with_config("", EdgeConfig::minimal()).unwrap()
    }

    #[test]
    fn test_basic_operations() {
        let mut db = create_test_db();

        // Put
        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();

        // Get
        assert_eq!(db.get(b"key1").unwrap(), Some(b"value1".to_vec()));
        assert_eq!(db.get(b"key2").unwrap(), Some(b"value2".to_vec()));
        assert_eq!(db.get(b"key3").unwrap(), None);

        // Update
        db.put(b"key1", b"updated").unwrap();
        assert_eq!(db.get(b"key1").unwrap(), Some(b"updated".to_vec()));

        // Delete
        assert!(db.delete(b"key1").unwrap());
        assert!(!db.delete(b"key1").unwrap()); // Already deleted
        assert_eq!(db.get(b"key1").unwrap(), None);
    }

    #[test]
    fn test_contains() {
        let mut db = create_test_db();
        db.put(b"exists", b"yes").unwrap();

        assert!(db.contains(b"exists"));
        assert!(!db.contains(b"not_exists"));
    }

    #[test]
    fn test_len() {
        let mut db = create_test_db();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);

        db.put(b"key1", b"value1").unwrap();
        db.put(b"key2", b"value2").unwrap();
        assert_eq!(db.len(), 2);

        db.delete(b"key1").unwrap();
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_prefix_scan() {
        let mut db = create_test_db();
        db.put(b"user:1", b"alice").unwrap();
        db.put(b"user:2", b"bob").unwrap();
        db.put(b"user:3", b"charlie").unwrap();
        db.put(b"order:1", b"order1").unwrap();

        let users = db.prefix_scan(b"user:");
        assert_eq!(users.len(), 3);

        let orders = db.prefix_scan(b"order:");
        assert_eq!(orders.len(), 1);
    }

    #[test]
    fn test_batch_operations() {
        let mut db = create_test_db();

        let pairs: Vec<(&[u8], &[u8])> = vec![(b"k1", b"v1"), (b"k2", b"v2"), (b"k3", b"v3")];
        db.put_batch(&pairs).unwrap();

        let keys: Vec<&[u8]> = vec![b"k1", b"k2", b"k3", b"k4"];
        let values = db.get_batch(&keys);

        assert_eq!(values.len(), 4);
        assert_eq!(values[0], Some(b"v1".to_vec()));
        assert_eq!(values[1], Some(b"v2".to_vec()));
        assert_eq!(values[2], Some(b"v3".to_vec()));
        assert_eq!(values[3], None);
    }

    #[test]
    fn test_hash() {
        let db = create_test_db();

        let hash1 = db.hash(b"hello");
        let hash2 = db.hash(b"world");
        let hash3 = db.hash(b"hello");

        assert_ne!(hash1, hash2);
        assert_eq!(hash1, hash3); // Deterministic
    }

    #[test]
    fn test_hash_batch() {
        let db = create_test_db();

        let data: Vec<&[u8]> = vec![b"hello", b"world", b"test"];
        let hashes = db.hash_batch(&data);

        assert_eq!(hashes.len(), 3);
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn test_empty_key_error() {
        let mut db = create_test_db();

        assert!(db.put(b"", b"value").is_err());
        assert!(db.get(b"").is_err());
        assert!(db.delete(b"").is_err());
    }

    #[test]
    fn test_stats() {
        let mut db = create_test_db();

        db.put(b"key", b"value").unwrap();
        let stats = db.stats();

        assert_eq!(stats.key_count, 1);
        assert_eq!(stats.put_ops, 1);
    }

    #[test]
    fn test_sync_and_checkpoint() {
        let mut db = create_test_db();

        db.put(b"key", b"value").unwrap();
        db.sync().unwrap();
        db.checkpoint().unwrap();
        // Should not panic
    }
}
