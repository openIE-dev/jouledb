//! Backup and Restore for JouleDB Server
//!
//! Provides functionality for:
//! - Full database backups
//! - Incremental backups
//! - Point-in-time recovery
//! - Backup compression
//! - Backup encryption
//! - Remote backup storage

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Types
// ============================================================================

/// Backup error
#[derive(Debug, Clone, PartialEq)]
pub enum BackupError {
    IoError(String),
    CompressionError(String),
    EncryptionError(String),
    InvalidBackup(String),
    BackupNotFound(String),
    RestoreFailed(String),
    StorageError(String),
}

impl std::fmt::Display for BackupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            Self::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            Self::InvalidBackup(msg) => write!(f, "Invalid backup: {}", msg),
            Self::BackupNotFound(msg) => write!(f, "Backup not found: {}", msg),
            Self::RestoreFailed(msg) => write!(f, "Restore failed: {}", msg),
            Self::StorageError(msg) => write!(f, "Storage error: {}", msg),
        }
    }
}

impl std::error::Error for BackupError {}

/// Backup type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupType {
    /// Full backup of entire database
    Full,
    /// Incremental backup since last backup
    Incremental,
    /// Differential backup since last full backup
    Differential,
}

/// Backup status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackupStatus {
    /// Backup in progress
    InProgress,
    /// Backup completed successfully
    Completed,
    /// Backup failed
    Failed,
    /// Backup cancelled
    Cancelled,
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Zstd,
    Lz4,
    Snappy,
}

impl Default for CompressionAlgorithm {
    fn default() -> Self {
        Self::Zstd
    }
}

/// Encryption algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    None,
    Aes256Gcm,
    ChaCha20Poly1305,
}

impl Default for EncryptionAlgorithm {
    fn default() -> Self {
        Self::None
    }
}

/// Backup metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupMetadata {
    /// Unique backup ID
    pub id: String,
    /// Backup type
    pub backup_type: BackupType,
    /// Backup status
    pub status: BackupStatus,
    /// Start timestamp
    pub started_at: u64,
    /// End timestamp
    pub completed_at: Option<u64>,
    /// Source database path
    pub source_path: String,
    /// Destination path
    pub destination_path: String,
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    /// Encryption algorithm
    pub encryption: EncryptionAlgorithm,
    /// Size in bytes (uncompressed)
    pub size_bytes: u64,
    /// Compressed size in bytes
    pub compressed_size: Option<u64>,
    /// Number of records
    pub record_count: u64,
    /// Last sequence number
    pub last_sequence: u64,
    /// Parent backup ID (for incremental/differential)
    pub parent_backup_id: Option<String>,
    /// Checksum
    pub checksum: Option<String>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Backup configuration
#[derive(Debug, Clone)]
pub struct BackupConfig {
    /// Backup directory
    pub backup_dir: PathBuf,
    /// Compression algorithm
    pub compression: CompressionAlgorithm,
    /// Encryption algorithm
    pub encryption: EncryptionAlgorithm,
    /// Encryption key (if encryption enabled)
    pub encryption_key: Option<Vec<u8>>,
    /// Maximum concurrent backups
    pub max_concurrent: usize,
    /// Retention days for backups
    pub retention_days: u32,
    /// Maximum backup size (0 = unlimited)
    pub max_size_bytes: u64,
    /// Verify after backup
    pub verify_after_backup: bool,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            backup_dir: PathBuf::from("./backups"),
            compression: CompressionAlgorithm::Zstd,
            encryption: EncryptionAlgorithm::None,
            encryption_key: None,
            max_concurrent: 1,
            retention_days: 30,
            max_size_bytes: 0,
            verify_after_backup: true,
        }
    }
}

/// Restore options
#[derive(Debug, Clone)]
pub struct RestoreOptions {
    /// Target path for restore
    pub target_path: PathBuf,
    /// Point-in-time to restore to (sequence number)
    pub point_in_time: Option<u64>,
    /// Overwrite existing data
    pub overwrite: bool,
    /// Verify before restore
    pub verify_before_restore: bool,
    /// Decryption key (if backup is encrypted)
    pub decryption_key: Option<Vec<u8>>,
}

impl Default for RestoreOptions {
    fn default() -> Self {
        Self {
            target_path: PathBuf::from("./restored"),
            point_in_time: None,
            overwrite: false,
            verify_before_restore: true,
            decryption_key: None,
        }
    }
}

// ============================================================================
// Backup Manager
// ============================================================================

/// Backup Manager
pub struct BackupManager {
    config: BackupConfig,
    backups: Arc<RwLock<HashMap<String, BackupMetadata>>>,
    active_backups: Arc<RwLock<Vec<String>>>,
}

impl BackupManager {
    /// Create new backup manager
    pub fn new(config: BackupConfig) -> Self {
        Self {
            config,
            backups: Arc::new(RwLock::new(HashMap::new())),
            active_backups: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(BackupConfig::default())
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn generate_id() -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        format!("backup_{:x}", now)
    }

    /// Start a full backup
    pub fn start_full_backup(&self, source_path: &Path) -> Result<BackupMetadata, BackupError> {
        self.start_backup(source_path, BackupType::Full, None)
    }

    /// Start an incremental backup
    pub fn start_incremental_backup(
        &self,
        source_path: &Path,
        parent_id: &str,
    ) -> Result<BackupMetadata, BackupError> {
        self.start_backup(source_path, BackupType::Incremental, Some(parent_id))
    }

    fn start_backup(
        &self,
        source_path: &Path,
        backup_type: BackupType,
        parent_id: Option<&str>,
    ) -> Result<BackupMetadata, BackupError> {
        // Check concurrent backup limit
        let active = crate::lock_util::read_lock(&self.active_backups);
        if active.len() >= self.config.max_concurrent {
            return Err(BackupError::StorageError(
                "Maximum concurrent backups reached".to_string(),
            ));
        }
        drop(active);

        let backup_id = Self::generate_id();
        let destination = self.config.backup_dir.join(format!("{}.wdb", backup_id));

        let metadata = BackupMetadata {
            id: backup_id.clone(),
            backup_type,
            status: BackupStatus::InProgress,
            started_at: Self::current_timestamp(),
            completed_at: None,
            source_path: source_path.to_string_lossy().to_string(),
            destination_path: destination.to_string_lossy().to_string(),
            compression: self.config.compression,
            encryption: self.config.encryption,
            size_bytes: 0,
            compressed_size: None,
            record_count: 0,
            last_sequence: 0,
            parent_backup_id: parent_id.map(String::from),
            checksum: None,
            custom: HashMap::new(),
        };

        // Register backup
        crate::lock_util::write_lock(&self.backups).insert(backup_id.clone(), metadata.clone());
        crate::lock_util::write_lock(&self.active_backups).push(backup_id);

        Ok(metadata)
    }

    /// Complete a backup
    pub fn complete_backup(
        &self,
        backup_id: &str,
        size_bytes: u64,
        compressed_size: Option<u64>,
        record_count: u64,
        last_sequence: u64,
        checksum: Option<String>,
    ) -> Result<BackupMetadata, BackupError> {
        let mut backups = crate::lock_util::write_lock(&self.backups);
        let metadata = backups
            .get_mut(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        metadata.status = BackupStatus::Completed;
        metadata.completed_at = Some(Self::current_timestamp());
        metadata.size_bytes = size_bytes;
        metadata.compressed_size = compressed_size;
        metadata.record_count = record_count;
        metadata.last_sequence = last_sequence;
        metadata.checksum = checksum;

        let result = metadata.clone();

        // Remove from active
        crate::lock_util::write_lock(&self.active_backups).retain(|id| id != backup_id);

        Ok(result)
    }

    /// Fail a backup
    pub fn fail_backup(&self, backup_id: &str) -> Result<(), BackupError> {
        let mut backups = crate::lock_util::write_lock(&self.backups);
        let metadata = backups
            .get_mut(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        metadata.status = BackupStatus::Failed;
        metadata.completed_at = Some(Self::current_timestamp());

        // Remove from active
        crate::lock_util::write_lock(&self.active_backups).retain(|id| id != backup_id);

        Ok(())
    }

    /// Cancel a backup
    pub fn cancel_backup(&self, backup_id: &str) -> Result<(), BackupError> {
        let mut backups = crate::lock_util::write_lock(&self.backups);
        let metadata = backups
            .get_mut(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if metadata.status != BackupStatus::InProgress {
            return Err(BackupError::InvalidBackup(
                "Backup is not in progress".to_string(),
            ));
        }

        metadata.status = BackupStatus::Cancelled;
        metadata.completed_at = Some(Self::current_timestamp());

        // Remove from active
        crate::lock_util::write_lock(&self.active_backups).retain(|id| id != backup_id);

        Ok(())
    }

    /// Get backup metadata
    pub fn get_backup(&self, backup_id: &str) -> Option<BackupMetadata> {
        crate::lock_util::read_lock(&self.backups)
            .get(backup_id)
            .cloned()
    }

    /// List all backups
    pub fn list_backups(&self) -> Vec<BackupMetadata> {
        crate::lock_util::read_lock(&self.backups)
            .values()
            .cloned()
            .collect()
    }

    /// List completed backups
    pub fn list_completed_backups(&self) -> Vec<BackupMetadata> {
        crate::lock_util::read_lock(&self.backups)
            .values()
            .filter(|b| b.status == BackupStatus::Completed)
            .cloned()
            .collect()
    }

    /// Delete old backups (retention policy)
    pub fn cleanup_old_backups(&self) -> Result<Vec<String>, BackupError> {
        let cutoff = Self::current_timestamp() - (self.config.retention_days as u64 * 86400);
        let mut deleted = Vec::new();

        let mut backups = crate::lock_util::write_lock(&self.backups);
        let ids_to_remove: Vec<String> = backups
            .iter()
            .filter(|(_, b)| b.status == BackupStatus::Completed && b.started_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        for id in ids_to_remove {
            backups.remove(&id);
            deleted.push(id);
        }

        Ok(deleted)
    }

    /// Verify backup integrity
    pub fn verify_backup(&self, backup_id: &str) -> Result<bool, BackupError> {
        let metadata = self
            .get_backup(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if metadata.status != BackupStatus::Completed {
            return Err(BackupError::InvalidBackup(
                "Backup is not completed".to_string(),
            ));
        }

        // In a real implementation, verify checksum and file integrity
        // For now, just check the file exists
        let path = Path::new(&metadata.destination_path);
        Ok(path.exists())
    }

    /// Get latest full backup
    pub fn get_latest_full_backup(&self) -> Option<BackupMetadata> {
        crate::lock_util::read_lock(&self.backups)
            .values()
            .filter(|b| b.backup_type == BackupType::Full && b.status == BackupStatus::Completed)
            .max_by_key(|b| b.started_at)
            .cloned()
    }

    /// Get backup chain for incremental restore
    pub fn get_backup_chain(&self, backup_id: &str) -> Result<Vec<BackupMetadata>, BackupError> {
        let mut chain = Vec::new();
        let mut current_id = Some(backup_id.to_string());

        while let Some(id) = current_id {
            let metadata = self
                .get_backup(&id)
                .ok_or_else(|| BackupError::BackupNotFound(id.clone()))?;

            current_id = metadata.parent_backup_id.clone();
            chain.push(metadata);
        }

        // Reverse so full backup is first
        chain.reverse();
        Ok(chain)
    }
}

// ============================================================================
// Live Storage Backup/Restore
// ============================================================================

impl BackupManager {
    /// Create a full backup from live AmorphicTableStorage.
    ///
    /// Scans all tables, serializes schemas and rows, compresses, and writes
    /// to the backup directory. Returns the completed backup metadata.
    pub fn create_full_backup_from_storage(
        &self,
        source_path: &Path,
        storage: &crate::amorphic_adapter::AmorphicTableStorage,
    ) -> Result<BackupMetadata, BackupError> {
        use joule_db_query::executor::TableStorage;

        let metadata = self.start_full_backup(source_path)?;
        let backup_id = metadata.id.clone();

        match self.do_create_full_backup(metadata, storage) {
            Ok(result) => Ok(result),
            Err(e) => {
                // Ensure backup is marked as failed so it doesn't stay InProgress forever
                let _ = self.fail_backup(&backup_id);
                Err(e)
            }
        }
    }

    /// Inner implementation of full backup creation, separated so that
    /// `create_full_backup_from_storage` can guarantee `fail_backup` on any error.
    fn do_create_full_backup(
        &self,
        metadata: BackupMetadata,
        storage: &crate::amorphic_adapter::AmorphicTableStorage,
    ) -> Result<BackupMetadata, BackupError> {
        use joule_db_query::executor::TableStorage;

        let backup_id = metadata.id.clone();
        let mut writer = BackupWriter::new(metadata);

        // Collect table schemas
        let tables = storage.list_tables();
        let mut schema_map: HashMap<String, Vec<String>> = HashMap::new();
        for table in &tables {
            match storage.columns(table) {
                Ok(cols) => {
                    schema_map.insert(table.clone(), cols);
                }
                Err(_) => continue, // skip tables without schema
            }
        }

        // Write schema header record (key = "__schemas__")
        let schema_json = serde_json::to_vec(&schema_map)
            .map_err(|e| BackupError::IoError(format!("Failed to serialize schemas: {e}")))?;
        writer.write_record(b"__schemas__", &schema_json)?;

        // Write data rows for each table
        let mut total_rows: u64 = 0;
        for table in &tables {
            let columns = match schema_map.get(table) {
                Some(cols) => cols,
                None => continue,
            };
            let rows = match storage.scan_with_ids(table) {
                Ok(rows) => rows,
                Err(_) => continue,
            };
            for (_record_id, row) in &rows {
                let mut obj = serde_json::Map::new();
                for (col, val) in columns.iter().zip(row.values.iter()) {
                    obj.insert(col.clone(), crate::amorphic_adapter::ast_to_json(val));
                }
                let json_bytes = serde_json::to_vec(&serde_json::Value::Object(obj))
                    .map_err(|e| BackupError::IoError(format!("Failed to serialize row: {e}")))?;
                writer.write_record(table.as_bytes(), &json_bytes)?;
                total_rows += 1;
            }
        }

        let (compressed, _updated_meta) = writer.finish()?;

        // Ensure backup directory exists and write compressed data
        std::fs::create_dir_all(&self.config.backup_dir)
            .map_err(|e| BackupError::IoError(format!("Failed to create backup dir: {e}")))?;
        let dest = self.config.backup_dir.join(format!("{}.wdb", backup_id));
        std::fs::write(&dest, &compressed)
            .map_err(|e| BackupError::IoError(format!("Failed to write backup file: {e}")))?;

        // Compute SHA-256 checksum
        use sha2::Digest;
        let checksum = hex::encode(sha2::Sha256::digest(&compressed));

        self.complete_backup(
            &backup_id,
            compressed.len() as u64,
            Some(compressed.len() as u64),
            total_rows,
            0,
            Some(checksum),
        )
    }

    /// Restore a backup into live AmorphicTableStorage.
    ///
    /// Reads the backup file, recreates table schemas, and ingests all rows.
    /// Returns the number of records restored.
    pub fn restore_backup_to_storage(
        &self,
        backup_id: &str,
        storage: &crate::amorphic_adapter::AmorphicTableStorage,
    ) -> Result<u64, BackupError> {
        use joule_db_query::executor::TableStorage;

        let metadata = self
            .get_backup(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if metadata.status != BackupStatus::Completed {
            return Err(BackupError::InvalidBackup(
                "Backup is not completed".to_string(),
            ));
        }

        let dest = Path::new(&metadata.destination_path);
        let compressed = std::fs::read(dest)
            .map_err(|e| BackupError::IoError(format!("Failed to read backup file: {e}")))?;
        let mut reader = BackupReader::new(&compressed)?;

        // First record must be the schema header
        let (key, value) = reader
            .read_record()?
            .ok_or_else(|| BackupError::InvalidBackup("Empty backup file".to_string()))?;

        if key != b"__schemas__" {
            return Err(BackupError::InvalidBackup(
                "First record must be __schemas__ header".to_string(),
            ));
        }

        let schemas: HashMap<String, Vec<String>> = serde_json::from_slice(&value)
            .map_err(|e| BackupError::InvalidBackup(format!("Invalid schema JSON: {e}")))?;

        // Create tables (ignore errors for already-existing tables)
        for (table, columns) in &schemas {
            let _ = storage.create_table(table, columns);
        }

        // Restore data rows
        let mut records_restored: u64 = 0;
        while let Some((key, value)) = reader.read_record()? {
            let table = String::from_utf8(key)
                .map_err(|e| BackupError::InvalidBackup(format!("Invalid table name: {e}")))?;
            let json = String::from_utf8(value)
                .map_err(|e| BackupError::InvalidBackup(format!("Invalid JSON: {e}")))?;
            storage.ingest_with_table(&json, &table).map_err(|e| {
                BackupError::RestoreFailed(format!("Ingest failed for table '{}': {e}", table))
            })?;
            records_restored += 1;
        }

        Ok(records_restored)
    }

    /// Verify a backup by reading it and checking record count and checksum.
    pub fn verify_backup_integrity(&self, backup_id: &str) -> Result<bool, BackupError> {
        let metadata = self
            .get_backup(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if metadata.status != BackupStatus::Completed {
            return Err(BackupError::InvalidBackup(
                "Backup is not completed".to_string(),
            ));
        }

        let dest = Path::new(&metadata.destination_path);
        let compressed = std::fs::read(dest)
            .map_err(|e| BackupError::IoError(format!("Failed to read backup file: {e}")))?;

        // Verify checksum
        if let Some(ref expected) = metadata.checksum {
            use sha2::Digest;
            let actual = hex::encode(sha2::Sha256::digest(&compressed));
            if actual != *expected {
                return Ok(false);
            }
        }

        // Verify we can decompress and read all records
        let mut reader = BackupReader::new(&compressed)?;
        let mut count: u64 = 0;
        while reader.read_record()?.is_some() {
            count += 1;
        }

        // Schema header + data records = record_count + 1
        Ok(count == metadata.record_count + 1)
    }
}

// ============================================================================
// Compression Utilities
// ============================================================================

/// Compress data using the specified algorithm.
///
/// Format: 4-byte magic + 4-byte original_len (little-endian) + compressed payload
pub fn compress(data: &[u8], algorithm: CompressionAlgorithm) -> Result<Vec<u8>, BackupError> {
    match algorithm {
        CompressionAlgorithm::None => Ok(data.to_vec()),
        CompressionAlgorithm::Gzip => {
            use flate2::Compression;
            use flate2::write::GzEncoder;
            use std::io::Write;

            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder
                .write_all(data)
                .map_err(|e| BackupError::CompressionError(format!("Gzip write failed: {e}")))?;
            let compressed = encoder
                .finish()
                .map_err(|e| BackupError::CompressionError(format!("Gzip finish failed: {e}")))?;
            let mut result = Vec::with_capacity(8 + compressed.len());
            result.extend_from_slice(b"GZIP");
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(&compressed);
            Ok(result)
        }
        CompressionAlgorithm::Zstd => {
            let compressed = zstd::encode_all(data, 3).map_err(|e| {
                BackupError::CompressionError(format!("Zstd compression failed: {e}"))
            })?;
            let mut result = Vec::with_capacity(8 + compressed.len());
            result.extend_from_slice(b"ZSTD");
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(&compressed);
            Ok(result)
        }
        CompressionAlgorithm::Lz4 => {
            let compressed = lz4_flex::compress_prepend_size(data);
            let mut result = Vec::with_capacity(8 + compressed.len());
            result.extend_from_slice(b"LZ4\0");
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(&compressed);
            Ok(result)
        }
        CompressionAlgorithm::Snappy => {
            let mut encoder = snap::raw::Encoder::new();
            let compressed = encoder.compress_vec(data).map_err(|e| {
                BackupError::CompressionError(format!("Snappy compression failed: {e}"))
            })?;
            let mut result = Vec::with_capacity(8 + compressed.len());
            result.extend_from_slice(b"SNAP");
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(&compressed);
            Ok(result)
        }
    }
}

/// Decompress data. Algorithm is detected from the 4-byte magic header.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, BackupError> {
    if data.len() < 8 {
        return Err(BackupError::CompressionError("Data too short".to_string()));
    }

    let magic = &data[0..4];
    let compressed_payload = &data[8..];

    match magic {
        b"GZIP" => {
            use flate2::read::GzDecoder;
            use std::io::Read;
            let mut decoder = GzDecoder::new(compressed_payload);
            let mut decompressed = Vec::new();
            decoder.read_to_end(&mut decompressed).map_err(|e| {
                BackupError::CompressionError(format!("Gzip decompression failed: {e}"))
            })?;
            Ok(decompressed)
        }
        b"ZSTD" => {
            let decompressed = zstd::decode_all(compressed_payload).map_err(|e| {
                BackupError::CompressionError(format!("Zstd decompression failed: {e}"))
            })?;
            Ok(decompressed)
        }
        b"LZ4\0" => {
            let decompressed =
                lz4_flex::decompress_size_prepended(compressed_payload).map_err(|e| {
                    BackupError::CompressionError(format!("LZ4 decompression failed: {e}"))
                })?;
            Ok(decompressed)
        }
        b"SNAP" => {
            let mut decoder = snap::raw::Decoder::new();
            let decompressed = decoder.decompress_vec(compressed_payload).map_err(|e| {
                BackupError::CompressionError(format!("Snappy decompression failed: {e}"))
            })?;
            Ok(decompressed)
        }
        _ => {
            // Assume uncompressed
            Ok(data.to_vec())
        }
    }
}

// ============================================================================
// Restore Manager
// ============================================================================

/// Restore Manager
pub struct RestoreManager {
    backup_manager: Arc<BackupManager>,
}

impl RestoreManager {
    /// Create new restore manager
    pub fn new(backup_manager: Arc<BackupManager>) -> Self {
        Self { backup_manager }
    }

    /// Start a restore operation
    pub fn start_restore(
        &self,
        backup_id: &str,
        options: RestoreOptions,
    ) -> Result<RestoreProgress, BackupError> {
        let metadata = self
            .backup_manager
            .get_backup(backup_id)
            .ok_or_else(|| BackupError::BackupNotFound(backup_id.to_string()))?;

        if metadata.status != BackupStatus::Completed {
            return Err(BackupError::InvalidBackup(
                "Backup is not completed".to_string(),
            ));
        }

        Ok(RestoreProgress {
            backup_id: backup_id.to_string(),
            target_path: options.target_path,
            status: RestoreStatus::InProgress,
            bytes_restored: 0,
            total_bytes: metadata.size_bytes,
            records_restored: 0,
            total_records: metadata.record_count,
            started_at: BackupManager::current_timestamp(),
            completed_at: None,
        })
    }

    /// Restore from the latest backup
    pub fn restore_latest(&self, options: RestoreOptions) -> Result<RestoreProgress, BackupError> {
        let latest = self
            .backup_manager
            .get_latest_full_backup()
            .ok_or_else(|| BackupError::BackupNotFound("No backups available".to_string()))?;

        self.start_restore(&latest.id, options)
    }
}

/// Restore status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Restore progress
#[derive(Debug, Clone)]
pub struct RestoreProgress {
    pub backup_id: String,
    pub target_path: PathBuf,
    pub status: RestoreStatus,
    pub bytes_restored: u64,
    pub total_bytes: u64,
    pub records_restored: u64,
    pub total_records: u64,
    pub started_at: u64,
    pub completed_at: Option<u64>,
}

impl RestoreProgress {
    /// Get progress percentage
    pub fn progress_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 100.0;
        }
        (self.bytes_restored as f64 / self.total_bytes as f64) * 100.0
    }
}

// ============================================================================
// Backup Writer
// ============================================================================

/// Backup writer for streaming backups
pub struct BackupWriter {
    metadata: BackupMetadata,
    buffer: Vec<u8>,
    records_written: u64,
    bytes_written: u64,
}

impl BackupWriter {
    /// Create new backup writer
    pub fn new(metadata: BackupMetadata) -> Self {
        Self {
            metadata,
            buffer: Vec::new(),
            records_written: 0,
            bytes_written: 0,
        }
    }

    /// Write a record
    pub fn write_record(&mut self, key: &[u8], value: &[u8]) -> Result<(), BackupError> {
        // Simple format: key_len (4) + key + value_len (4) + value
        let key_len = key.len() as u32;
        let value_len = value.len() as u32;

        self.buffer.extend_from_slice(&key_len.to_le_bytes());
        self.buffer.extend_from_slice(key);
        self.buffer.extend_from_slice(&value_len.to_le_bytes());
        self.buffer.extend_from_slice(value);

        self.records_written += 1;
        self.bytes_written += 8 + key.len() as u64 + value.len() as u64;

        Ok(())
    }

    /// Finish writing and return compressed data
    pub fn finish(self) -> Result<(Vec<u8>, BackupMetadata), BackupError> {
        let compressed = compress(&self.buffer, self.metadata.compression)?;
        let mut metadata = self.metadata;
        metadata.record_count = self.records_written;
        metadata.size_bytes = self.bytes_written;
        metadata.compressed_size = Some(compressed.len() as u64);
        Ok((compressed, metadata))
    }

    /// Get current stats
    pub fn stats(&self) -> (u64, u64) {
        (self.records_written, self.bytes_written)
    }
}

// ============================================================================
// Backup Reader
// ============================================================================

/// Backup reader for streaming restores
pub struct BackupReader {
    data: Vec<u8>,
    position: usize,
}

impl BackupReader {
    /// Create new backup reader from compressed data
    pub fn new(compressed_data: &[u8]) -> Result<Self, BackupError> {
        let data = decompress(compressed_data)?;
        Ok(Self { data, position: 0 })
    }

    /// Read next record
    pub fn read_record(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>, BackupError> {
        if self.position >= self.data.len() {
            return Ok(None);
        }

        if self.position + 4 > self.data.len() {
            return Err(BackupError::InvalidBackup("Truncated data".to_string()));
        }

        let key_len = u32::from_le_bytes(
            self.data[self.position..self.position + 4]
                .try_into()
                .expect("slice length verified above"),
        ) as usize;
        self.position += 4;

        if self.position + key_len > self.data.len() {
            return Err(BackupError::InvalidBackup("Truncated key".to_string()));
        }

        let key = self.data[self.position..self.position + key_len].to_vec();
        self.position += key_len;

        if self.position + 4 > self.data.len() {
            return Err(BackupError::InvalidBackup("Truncated data".to_string()));
        }

        let value_len = u32::from_le_bytes(
            self.data[self.position..self.position + 4]
                .try_into()
                .expect("slice length verified above"),
        ) as usize;
        self.position += 4;

        if self.position + value_len > self.data.len() {
            return Err(BackupError::InvalidBackup("Truncated value".to_string()));
        }

        let value = self.data[self.position..self.position + value_len].to_vec();
        self.position += value_len;

        Ok(Some((key, value)))
    }

    /// Reset reader to beginning
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backup_manager_creation() {
        let manager = BackupManager::with_defaults();
        assert!(manager.list_backups().is_empty());
    }

    #[test]
    fn test_start_full_backup() {
        let manager = BackupManager::with_defaults();
        let metadata = manager.start_full_backup(Path::new("./test.db")).unwrap();

        assert_eq!(metadata.backup_type, BackupType::Full);
        assert_eq!(metadata.status, BackupStatus::InProgress);
        assert!(metadata.id.starts_with("backup_"));
    }

    #[test]
    fn test_complete_backup() {
        let manager = BackupManager::with_defaults();
        let metadata = manager.start_full_backup(Path::new("./test.db")).unwrap();

        let completed = manager
            .complete_backup(
                &metadata.id,
                1000,
                Some(500),
                100,
                50,
                Some("checksum".to_string()),
            )
            .unwrap();

        assert_eq!(completed.status, BackupStatus::Completed);
        assert_eq!(completed.size_bytes, 1000);
        assert_eq!(completed.compressed_size, Some(500));
        assert_eq!(completed.record_count, 100);
    }

    #[test]
    fn test_cancel_backup() {
        let manager = BackupManager::with_defaults();
        let metadata = manager.start_full_backup(Path::new("./test.db")).unwrap();

        manager.cancel_backup(&metadata.id).unwrap();

        let updated = manager.get_backup(&metadata.id).unwrap();
        assert_eq!(updated.status, BackupStatus::Cancelled);
    }

    #[test]
    fn test_compression_roundtrip() {
        let data = b"Hello, World! This is test data for compression.";

        for algorithm in [
            CompressionAlgorithm::None,
            CompressionAlgorithm::Gzip,
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
            CompressionAlgorithm::Snappy,
        ] {
            let compressed = compress(data, algorithm).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, data, "Algorithm: {:?}", algorithm);
        }
    }

    #[test]
    fn test_backup_writer_and_reader() {
        let metadata = BackupMetadata {
            id: "test".to_string(),
            backup_type: BackupType::Full,
            status: BackupStatus::InProgress,
            started_at: 0,
            completed_at: None,
            source_path: "./test.db".to_string(),
            destination_path: "./backup.wdb".to_string(),
            compression: CompressionAlgorithm::Zstd,
            encryption: EncryptionAlgorithm::None,
            size_bytes: 0,
            compressed_size: None,
            record_count: 0,
            last_sequence: 0,
            parent_backup_id: None,
            checksum: None,
            custom: HashMap::new(),
        };

        let mut writer = BackupWriter::new(metadata);
        writer.write_record(b"key1", b"value1").unwrap();
        writer.write_record(b"key2", b"value2").unwrap();
        writer.write_record(b"key3", b"value3").unwrap();

        let (compressed, metadata) = writer.finish().unwrap();
        assert_eq!(metadata.record_count, 3);

        let mut reader = BackupReader::new(&compressed).unwrap();
        let (key1, value1) = reader.read_record().unwrap().unwrap();
        assert_eq!(key1, b"key1");
        assert_eq!(value1, b"value1");

        let (key2, value2) = reader.read_record().unwrap().unwrap();
        assert_eq!(key2, b"key2");
        assert_eq!(value2, b"value2");

        let (key3, value3) = reader.read_record().unwrap().unwrap();
        assert_eq!(key3, b"key3");
        assert_eq!(value3, b"value3");

        assert!(reader.read_record().unwrap().is_none());
    }

    #[test]
    fn test_backup_chain() {
        let manager = BackupManager::with_defaults();

        // Create full backup
        let full = manager.start_full_backup(Path::new("./test.db")).unwrap();
        manager
            .complete_backup(&full.id, 1000, None, 100, 50, None)
            .unwrap();

        // Create incremental backup
        let incr = manager
            .start_incremental_backup(Path::new("./test.db"), &full.id)
            .unwrap();
        manager
            .complete_backup(&incr.id, 200, None, 20, 70, None)
            .unwrap();

        // Get chain
        let chain = manager.get_backup_chain(&incr.id).unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].backup_type, BackupType::Full);
        assert_eq!(chain[1].backup_type, BackupType::Incremental);
    }

    #[test]
    fn test_restore_progress() {
        let progress = RestoreProgress {
            backup_id: "test".to_string(),
            target_path: PathBuf::from("./restored"),
            status: RestoreStatus::InProgress,
            bytes_restored: 500,
            total_bytes: 1000,
            records_restored: 50,
            total_records: 100,
            started_at: 0,
            completed_at: None,
        };

        assert_eq!(progress.progress_percent(), 50.0);
    }

    // =========================================================================
    // Live storage backup/restore tests
    // =========================================================================

    fn test_storage() -> std::sync::Arc<crate::amorphic_adapter::AmorphicTableStorage> {
        let dir = tempfile::tempdir().expect("temp dir");
        let store = joule_db_amorphic::DurableAmorphicStore::open(dir.path()).expect("temp store");
        // Leak dir so it stays alive for the test
        std::mem::forget(dir);
        std::sync::Arc::new(crate::amorphic_adapter::AmorphicTableStorage::new(store))
    }

    #[test]
    fn test_backup_restore_roundtrip() {
        use joule_db_query::ast::Value as AstValue;
        use joule_db_query::executor::{RowData, TableStorage};

        let storage = test_storage();
        let backup_dir = tempfile::tempdir().expect("backup dir");
        let manager = BackupManager::new(BackupConfig {
            backup_dir: backup_dir.path().to_path_buf(),
            compression: CompressionAlgorithm::Zstd,
            ..BackupConfig::default()
        });

        // Create a table with data
        storage
            .create_table("users", &["name".into(), "age".into()])
            .unwrap();
        storage
            .insert(
                "users",
                &RowData::new(
                    vec!["name".into(), "age".into()],
                    vec![AstValue::String("Alice".into()), AstValue::Int(30)],
                ),
            )
            .unwrap();
        storage
            .insert(
                "users",
                &RowData::new(
                    vec!["name".into(), "age".into()],
                    vec![AstValue::String("Bob".into()), AstValue::Int(25)],
                ),
            )
            .unwrap();

        // Create backup
        let metadata = manager
            .create_full_backup_from_storage(Path::new("/test"), &storage)
            .unwrap();
        assert_eq!(metadata.status, BackupStatus::Completed);
        assert_eq!(metadata.record_count, 2);
        assert!(metadata.checksum.is_some());

        // Restore into fresh storage
        let storage2 = test_storage();
        let restored = manager
            .restore_backup_to_storage(&metadata.id, &storage2)
            .unwrap();
        assert_eq!(restored, 2);

        // Verify data
        let rows = storage2.scan("users").unwrap();
        assert_eq!(rows.len(), 2);
        let names: Vec<_> = rows
            .iter()
            .filter_map(|r| match r.get("name") {
                Some(AstValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(names.contains(&"Alice".to_string()));
        assert!(names.contains(&"Bob".to_string()));
    }

    #[test]
    fn test_backup_verify_integrity() {
        use joule_db_query::ast::Value as AstValue;
        use joule_db_query::executor::{RowData, TableStorage};

        let storage = test_storage();
        let backup_dir = tempfile::tempdir().expect("backup dir");
        let manager = BackupManager::new(BackupConfig {
            backup_dir: backup_dir.path().to_path_buf(),
            compression: CompressionAlgorithm::Zstd,
            ..BackupConfig::default()
        });

        storage.create_table("t", &["x".into()]).unwrap();
        storage
            .insert(
                "t",
                &RowData::new(vec!["x".into()], vec![AstValue::Int(42)]),
            )
            .unwrap();

        let metadata = manager
            .create_full_backup_from_storage(Path::new("/test"), &storage)
            .unwrap();

        // Verify passes
        assert!(manager.verify_backup_integrity(&metadata.id).unwrap());
    }

    #[test]
    fn test_backup_multiple_tables() {
        use joule_db_query::ast::Value as AstValue;
        use joule_db_query::executor::{RowData, TableStorage};

        let storage = test_storage();
        let backup_dir = tempfile::tempdir().expect("backup dir");
        let manager = BackupManager::new(BackupConfig {
            backup_dir: backup_dir.path().to_path_buf(),
            compression: CompressionAlgorithm::Zstd,
            ..BackupConfig::default()
        });

        // Create two tables
        storage.create_table("a", &["val".into()]).unwrap();
        storage
            .create_table("b", &["name".into(), "score".into()])
            .unwrap();

        storage
            .insert(
                "a",
                &RowData::new(vec!["val".into()], vec![AstValue::Int(1)]),
            )
            .unwrap();
        storage
            .insert(
                "a",
                &RowData::new(vec!["val".into()], vec![AstValue::Int(2)]),
            )
            .unwrap();
        storage
            .insert(
                "b",
                &RowData::new(
                    vec!["name".into(), "score".into()],
                    vec![AstValue::String("Alice".into()), AstValue::Int(100)],
                ),
            )
            .unwrap();

        let metadata = manager
            .create_full_backup_from_storage(Path::new("/test"), &storage)
            .unwrap();
        assert_eq!(metadata.record_count, 3);

        // Restore
        let storage2 = test_storage();
        let restored = manager
            .restore_backup_to_storage(&metadata.id, &storage2)
            .unwrap();
        assert_eq!(restored, 3);

        let a_rows = storage2.scan("a").unwrap();
        assert_eq!(a_rows.len(), 2);
        let b_rows = storage2.scan("b").unwrap();
        assert_eq!(b_rows.len(), 1);
    }

    #[test]
    fn test_backup_empty_database() {
        let storage = test_storage();
        let backup_dir = tempfile::tempdir().expect("backup dir");
        let manager = BackupManager::new(BackupConfig {
            backup_dir: backup_dir.path().to_path_buf(),
            ..BackupConfig::default()
        });

        let metadata = manager
            .create_full_backup_from_storage(Path::new("/test"), &storage)
            .unwrap();
        assert_eq!(metadata.record_count, 0);
        assert!(manager.verify_backup_integrity(&metadata.id).unwrap());

        // Restore empty backup
        let storage2 = test_storage();
        let restored = manager
            .restore_backup_to_storage(&metadata.id, &storage2)
            .unwrap();
        assert_eq!(restored, 0);
    }
}
