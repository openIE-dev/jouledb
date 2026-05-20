//! Durable AmorphicStore with Write-Ahead Logging
//!
//! This module wraps AmorphicStore with WAL-based durability, ensuring crash recovery.
//!
//! ## Recovery Semantics
//!
//! On startup, the DurableAmorphicStore:
//! 1. Opens the WAL file
//! 2. Replays committed transactions
//! 3. Discards uncommitted changes
//!
//! ## Transaction Model
//!
//! Each `ingest_*` operation is its own transaction:
//! 1. Log the record data to WAL
//! 2. Apply to in-memory store
//! 3. Commit (sync to disk)
//!
//! For batch operations, use `begin_batch()` and `commit_batch()`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use joule_db_local::storage::wal::{RecoveryManager, RecoveryResult, WalManager};

use joule_db_hdc::BinaryHV;

use crate::{AmorphicError, AmorphicRecord, AmorphicResult, AmorphicStore, RecordId, Value};

/// Durable wrapper for AmorphicStore with WAL-based crash recovery
///
/// All write operations are logged to the WAL before being applied,
/// ensuring data survives crashes.
pub struct DurableAmorphicStore {
    /// The underlying in-memory store
    store: AmorphicStore,

    /// Write-ahead log manager
    wal: WalManager,

    /// Transaction counter
    tx_counter: AtomicU64,

    /// Database directory path
    path: PathBuf,

    /// Active batch transaction (if any)
    active_batch: Option<BatchTransaction>,
}

/// A batch transaction for grouping multiple operations
struct BatchTransaction {
    tx_id: u64,
    records: Vec<(RecordId, Vec<u8>)>,
}

/// Serializable record representation for WAL (legacy format, kept for backward compat)
#[derive(serde::Serialize, serde::Deserialize)]
struct WalRecordData {
    /// Record ID assigned
    id: RecordId,
    /// Field data (JSON serialized)
    fields: HashMap<String, Value>,
    /// Edges (if any)
    edges: Vec<(String, RecordId)>,
    /// Timestamp (if any)
    timestamp: Option<u64>,
}

/// Tagged WAL operation for new operation types (delete, update, edge)
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "op")]
enum WalOperation {
    /// Ingest a record (same data as WalRecordData)
    #[serde(rename = "ingest")]
    Ingest {
        id: RecordId,
        fields: HashMap<String, Value>,
        edges: Vec<(String, RecordId)>,
        timestamp: Option<u64>,
    },
    /// Ingest an edge between two named entities
    #[serde(rename = "ingest_edge")]
    IngestEdge {
        source: String,
        relation: String,
        target: String,
    },
    /// Delete a record by ID
    #[serde(rename = "delete")]
    Delete { id: RecordId },
    /// Update specific fields of a record
    #[serde(rename = "update_fields")]
    UpdateFields {
        id: RecordId,
        updates: HashMap<String, Value>,
    },
}

impl DurableAmorphicStore {
    /// Open or create a durable AmorphicStore at the given directory
    ///
    /// This will:
    /// 1. Open/create the WAL file
    /// 2. Recover any committed but not applied transactions
    /// 3. Initialize the in-memory store
    pub fn open<P: AsRef<Path>>(dir: P) -> AmorphicResult<Self> {
        let path = dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&path).map_err(|e| {
            AmorphicError::IngestionError(format!("Failed to create directory: {}", e))
        })?;

        // Open WAL manager
        let wal = WalManager::open(&path)
            .map_err(|e| AmorphicError::IngestionError(format!("Failed to open WAL: {}", e)))?;

        // Create empty store
        let mut store = AmorphicStore::new();

        // Recover from WAL
        let recovery_result = RecoveryManager::recover(&wal)
            .map_err(|e| AmorphicError::IngestionError(format!("Recovery failed: {}", e)))?;

        // Apply recovered records
        Self::apply_recovery(&mut store, &recovery_result)?;

        // Determine next transaction ID
        let max_tx_id = recovery_result
            .committed_transactions
            .iter()
            .max()
            .copied()
            .unwrap_or(0);

        Ok(Self {
            store,
            wal,
            tx_counter: AtomicU64::new(max_tx_id + 1),
            path,
            active_batch: None,
        })
    }

    /// Apply recovery result to the store
    ///
    /// Handles both legacy WalRecordData format and new WalOperation format.
    /// Operations are sorted by page_id to ensure correct replay order:
    /// ingest operations use small record IDs as page_ids, while delete/update
    /// operations use virtual page_ids near u64::MAX, so ascending sort
    /// guarantees ingests are replayed before mutations.
    fn apply_recovery(store: &mut AmorphicStore, result: &RecoveryResult) -> AmorphicResult<()> {
        let mut applied = 0usize;

        // Sort by page_id ascending: record ingests (small IDs) before
        // deletes/updates (virtual page_ids near u64::MAX)
        let mut sorted_pages: Vec<_> = result.pages_to_apply.iter().collect();
        sorted_pages.sort_by_key(|(page_id, _)| *page_id);

        for (_page_id, data) in sorted_pages {
            // Try new tagged WalOperation format first, fall back to legacy WalRecordData
            if let Ok(op) = serde_json::from_slice::<WalOperation>(data) {
                match op {
                    WalOperation::Ingest {
                        id,
                        fields,
                        edges,
                        timestamp,
                    } => {
                        store.apply_recovered_record(id, fields, edges, timestamp)?;
                    }
                    WalOperation::IngestEdge {
                        source,
                        relation,
                        target,
                    } => {
                        let _ = store.ingest_edge(&source, &relation, &target);
                    }
                    WalOperation::Delete { id } => {
                        let _ = store.delete(id);
                    }
                    WalOperation::UpdateFields { id, updates } => {
                        let _ = store.update_fields(id, updates);
                    }
                }
            } else if let Ok(record_data) = serde_json::from_slice::<WalRecordData>(data) {
                // Legacy format: plain WalRecordData without "op" tag
                store.apply_recovered_record(
                    record_data.id,
                    record_data.fields,
                    record_data.edges,
                    record_data.timestamp,
                )?;
            } else {
                eprintln!("⚠ Skipping unrecognized WAL record during recovery");
                continue;
            }

            applied += 1;
        }

        eprintln!(
            "✓ Recovered {} committed transactions, {} operations applied",
            result.committed_transactions.len(),
            applied
        );

        if !result.uncommitted_transactions.is_empty() {
            eprintln!(
                "⚠ Discarded {} uncommitted transactions",
                result.uncommitted_transactions.len()
            );
        }

        Ok(())
    }

    /// Get next transaction ID
    fn next_tx_id(&self) -> u64 {
        self.tx_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Ingest a JSON document with durability
    ///
    /// This logs to WAL, applies to store, and commits atomically.
    pub fn ingest_json(&mut self, json: &str) -> AmorphicResult<RecordId> {
        let tx_id = self.next_tx_id();

        // Parse and prepare the record
        let (id, fields) = self.store.prepare_json_ingest(json)?;

        // Create WAL record data
        let record_data = WalRecordData {
            id,
            fields: fields.clone(),
            edges: vec![],
            timestamp: None,
        };
        let serialized = serde_json::to_vec(&record_data)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        // Log to WAL BEFORE applying
        self.wal
            .log_page_write(tx_id, id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to in-memory store
        self.store.apply_prepared_ingest(id, fields)?;

        // Commit transaction (syncs to disk)
        self.wal
            .log_commit(tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL commit failed: {}", e)))?;

        Ok(id)
    }

    /// Ingest a row (columns + values) with durability
    pub fn ingest_row(&mut self, columns: &[&str], values: &[&str]) -> AmorphicResult<RecordId> {
        let tx_id = self.next_tx_id();

        // Parse and prepare the record
        let (id, fields) = self.store.prepare_row_ingest(columns, values)?;

        // Create WAL record data
        let record_data = WalRecordData {
            id,
            fields: fields.clone(),
            edges: vec![],
            timestamp: None,
        };
        let serialized = serde_json::to_vec(&record_data)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        // Log to WAL
        self.wal
            .log_page_write(tx_id, id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        self.store.apply_prepared_ingest(id, fields)?;

        // Commit
        self.wal
            .log_commit(tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL commit failed: {}", e)))?;

        Ok(id)
    }

    /// Begin a batch transaction
    ///
    /// Multiple operations can be grouped and committed together.
    pub fn begin_batch(&mut self) -> AmorphicResult<()> {
        if self.active_batch.is_some() {
            return Err(AmorphicError::IngestionError("Batch already active".into()));
        }

        let tx_id = self.next_tx_id();
        self.active_batch = Some(BatchTransaction {
            tx_id,
            records: Vec::new(),
        });

        Ok(())
    }

    /// Add a JSON document to the current batch
    pub fn batch_ingest_json(&mut self, json: &str) -> AmorphicResult<RecordId> {
        let batch = self
            .active_batch
            .as_mut()
            .ok_or_else(|| AmorphicError::IngestionError("No active batch".into()))?;

        let tx_id = batch.tx_id;

        // Parse and prepare
        let (id, fields) = self.store.prepare_json_ingest(json)?;

        // Create WAL record
        let record_data = WalRecordData {
            id,
            fields: fields.clone(),
            edges: vec![],
            timestamp: None,
        };
        let serialized = serde_json::to_vec(&record_data)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        // Log to WAL (no commit yet)
        self.wal
            .log_page_write(tx_id, id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        self.store.apply_prepared_ingest(id, fields)?;

        // Track in batch
        batch.records.push((id, serialized));

        Ok(id)
    }

    /// Delete a record within the current batch transaction
    pub fn batch_delete(&mut self, id: RecordId) -> AmorphicResult<()> {
        let batch = self
            .active_batch
            .as_mut()
            .ok_or_else(|| AmorphicError::IngestionError("No active batch".into()))?;

        let tx_id = batch.tx_id;

        // WAL-log the delete operation
        let op = WalOperation::Delete { id };
        let serialized = serde_json::to_vec(&op)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        let virtual_page_id = u64::MAX - tx_id - batch.records.len() as u64 - 1;
        self.wal
            .log_page_write(tx_id, virtual_page_id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        self.store.delete(id)?;

        // Track in batch (for rollback — re-ingest is not straightforward, but we track it)
        batch.records.push((virtual_page_id, serialized));

        Ok(())
    }

    /// Update specific fields of a record within the current batch transaction
    pub fn batch_update_fields(
        &mut self,
        id: RecordId,
        updates: HashMap<String, Value>,
    ) -> AmorphicResult<()> {
        let batch = self
            .active_batch
            .as_mut()
            .ok_or_else(|| AmorphicError::IngestionError("No active batch".into()))?;

        let tx_id = batch.tx_id;

        // WAL-log the update operation
        let op = WalOperation::UpdateFields {
            id,
            updates: updates.clone(),
        };
        let serialized = serde_json::to_vec(&op)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        let virtual_page_id = u64::MAX - tx_id - batch.records.len() as u64 - 1;
        self.wal
            .log_page_write(tx_id, virtual_page_id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        self.store.update_fields(id, updates)?;

        // Track in batch
        batch.records.push((virtual_page_id, serialized));

        Ok(())
    }

    /// Commit the current batch transaction
    pub fn commit_batch(&mut self) -> AmorphicResult<usize> {
        let batch = self
            .active_batch
            .take()
            .ok_or_else(|| AmorphicError::IngestionError("No active batch".into()))?;

        let count = batch.records.len();

        // Commit transaction (syncs to disk)
        self.wal
            .log_commit(batch.tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL commit failed: {}", e)))?;

        Ok(count)
    }

    /// Rollback the current batch transaction
    pub fn rollback_batch(&mut self) -> AmorphicResult<()> {
        let batch = self
            .active_batch
            .take()
            .ok_or_else(|| AmorphicError::IngestionError("No active batch".into()))?;

        // Log rollback
        self.wal
            .log_rollback(batch.tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL rollback failed: {}", e)))?;

        // Undo changes in store (remove records added in this batch)
        for (id, _) in batch.records {
            // Best effort removal - may fail if record wasn't fully applied
            let _ = self.store.delete(id);
        }

        Ok(())
    }

    /// Create a checkpoint
    ///
    /// This marks a point in the WAL where all prior transactions are
    /// guaranteed to be committed. Used to truncate old WAL entries.
    pub fn checkpoint(&mut self) -> AmorphicResult<u64> {
        let lsn = self
            .wal
            .log_checkpoint()
            .map_err(|e| AmorphicError::IngestionError(format!("Checkpoint failed: {}", e)))?;

        Ok(lsn)
    }

    /// Sync WAL to disk without creating checkpoint
    pub fn sync(&mut self) -> AmorphicResult<()> {
        self.wal
            .sync()
            .map_err(|e| AmorphicError::IngestionError(format!("Sync failed: {}", e)))?;

        Ok(())
    }

    // ==================== Write Operations (WAL-logged) ====================

    /// Ingest a graph edge with durability
    ///
    /// Creates an edge between two named entities. If the entities don't exist,
    /// they are auto-created. The edge operation is logged to WAL for crash recovery.
    pub fn ingest_edge(
        &mut self,
        source: &str,
        relation: &str,
        target: &str,
    ) -> AmorphicResult<RecordId> {
        let tx_id = self.next_tx_id();

        // WAL-log the edge operation BEFORE applying
        let op = WalOperation::IngestEdge {
            source: source.to_string(),
            relation: relation.to_string(),
            target: target.to_string(),
        };
        let serialized = serde_json::to_vec(&op)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        // Use virtual page_id to avoid colliding with record-based page_ids
        let virtual_page_id = u64::MAX - tx_id;
        self.wal
            .log_page_write(tx_id, virtual_page_id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        let id = self.store.ingest_edge(source, relation, target)?;

        // Commit
        self.wal
            .log_commit(tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL commit failed: {}", e)))?;

        Ok(id)
    }

    /// Delete a record by ID with durability
    ///
    /// Removes the record from all indices. The deletion is logged to WAL.
    /// Note: The global hologram retains traces of deleted records until rebuild.
    pub fn delete(&mut self, id: RecordId) -> AmorphicResult<()> {
        let tx_id = self.next_tx_id();

        // WAL-log the delete operation BEFORE applying.
        // Use a virtual page_id (u64::MAX - tx_id) to avoid colliding with
        // record-based page_ids used by ingest operations. The WAL recovery
        // deduplicates by page_id, so each operation needs a unique one.
        let op = WalOperation::Delete { id };
        let serialized = serde_json::to_vec(&op)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        let virtual_page_id = u64::MAX - tx_id;
        self.wal
            .log_page_write(tx_id, virtual_page_id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        self.store.delete(id)?;

        // Commit
        self.wal
            .log_commit(tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL commit failed: {}", e)))?;

        Ok(())
    }

    /// Update specific fields of a record with durability
    ///
    /// Updates the specified fields and re-encodes the HDC hologram.
    /// The update is logged to WAL for crash recovery.
    pub fn update_fields(
        &mut self,
        id: RecordId,
        updates: HashMap<String, Value>,
    ) -> AmorphicResult<()> {
        let tx_id = self.next_tx_id();

        // WAL-log the update operation BEFORE applying.
        // Use virtual page_id to avoid colliding with the record's ingest WAL entry.
        let op = WalOperation::UpdateFields {
            id,
            updates: updates.clone(),
        };
        let serialized = serde_json::to_vec(&op)
            .map_err(|e| AmorphicError::IngestionError(format!("Serialization failed: {}", e)))?;

        let virtual_page_id = u64::MAX - tx_id;
        self.wal
            .log_page_write(tx_id, virtual_page_id, &serialized)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL write failed: {}", e)))?;

        // Apply to store
        self.store.update_fields(id, updates)?;

        // Commit
        self.wal
            .log_commit(tx_id)
            .map_err(|e| AmorphicError::IngestionError(format!("WAL commit failed: {}", e)))?;

        Ok(())
    }

    // ==================== Read Operations (delegate to inner store) ====================

    /// Query by exact field-value match
    pub fn query_equals(&self, field: &str, value: &Value) -> crate::QueryResult {
        self.store.query_equals(field, value)
    }

    /// Query by range
    pub fn query_range(&self, field: &str, min: f64, max: f64) -> crate::QueryResult {
        self.store.query_range(field, min, max)
    }

    /// Query by field existence
    pub fn query_has_field(&self, field: &str) -> crate::QueryResult {
        self.store.query_has_field(field)
    }

    /// Query similar records by name
    pub fn query_similar_to(&self, name: &str, k: usize) -> crate::QueryResult {
        self.store.query_similar_to(name, k)
    }

    /// Query similar records by probe vector
    pub fn query_similar(&self, probe: &BinaryHV, threshold: f32) -> crate::QueryResult {
        self.store.query_similar(probe, threshold)
    }

    /// Graph traversal query
    pub fn query_graph(&self, start: &str, relation: &str, depth: usize) -> crate::QueryResult {
        self.store.query_graph(start, relation, depth)
    }

    /// Time range query
    pub fn query_time_range(&self, start: u64, end: u64) -> crate::QueryResult {
        self.store.query_time_range(start, end)
    }

    /// SQL-like query
    pub fn query_sql(&self, query: &str) -> AmorphicResult<crate::QueryResult> {
        self.store.query_sql(query)
    }

    /// Get record by ID
    pub fn get(&self, id: RecordId) -> Option<&AmorphicRecord> {
        self.store.get(id)
    }

    /// Get by name
    pub fn get_by_name(&self, name: &str) -> Option<&AmorphicRecord> {
        self.store.get_by_name(name)
    }

    /// Get record count
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Health check
    pub fn health_check(&self) -> crate::HealthStatus {
        self.store.health_check()
    }

    /// Get statistics
    pub fn stats(&self) -> crate::StoreStats {
        self.store.stats()
    }

    /// Access the columnar store for aggregate operations
    pub fn columnar(&self) -> &crate::ColumnarStore {
        self.store.columnar()
    }

    /// Get database directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get current WAL LSN
    pub fn current_lsn(&self) -> u64 {
        self.wal.current_lsn()
    }

    /// Check if the WAL needs a checkpoint (exceeds size threshold)
    pub fn needs_checkpoint(&self) -> bool {
        self.wal.needs_checkpoint()
    }

    /// Find amorphic RecordId for a SQL table row matching given column/value pairs.
    /// Returns the first matching record's ID, or None.
    pub fn find_record_id(
        &self,
        table: &str,
        columns: &[String],
        values: &[Value],
    ) -> Option<RecordId> {
        let table_val = Value::String(table.to_string());
        let result = self.store.query_equals("__table__", &table_val);
        for record in result.records() {
            let mut matches = true;
            for (col, val) in columns.iter().zip(values.iter()) {
                if record.get(col) != Some(val) {
                    matches = false;
                    break;
                }
            }
            if matches {
                return Some(record.id);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_durable_basic_ingest() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        assert_eq!(store.len(), 1);

        let record = store.get(id).unwrap();
        assert_eq!(
            record.get("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_durable_crash_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Write some data
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            store
                .ingest_json(r#"{"name": "Bob", "value": 42}"#)
                .unwrap();
            store
                .ingest_json(r#"{"name": "Carol", "value": 100}"#)
                .unwrap();
            // Drop store (simulates crash - data is in WAL but store is gone)
        }

        // Recover
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            assert_eq!(store.len(), 2);

            assert!(store.get_by_name("Bob").is_some());
            assert!(store.get_by_name("Carol").is_some());
        }
    }

    #[test]
    fn test_durable_batch_commit() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        store.begin_batch().unwrap();
        store.batch_ingest_json(r#"{"name": "Item1"}"#).unwrap();
        store.batch_ingest_json(r#"{"name": "Item2"}"#).unwrap();
        store.batch_ingest_json(r#"{"name": "Item3"}"#).unwrap();
        let count = store.commit_batch().unwrap();

        assert_eq!(count, 3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_durable_batch_rollback() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        // First, commit some data
        store.ingest_json(r#"{"name": "Committed"}"#).unwrap();

        // Start a batch that we'll rollback
        store.begin_batch().unwrap();
        store.batch_ingest_json(r#"{"name": "Rollback1"}"#).unwrap();
        store.batch_ingest_json(r#"{"name": "Rollback2"}"#).unwrap();
        store.rollback_batch().unwrap();

        // Only the first record should remain
        assert_eq!(store.len(), 1);
        assert!(store.get_by_name("Committed").is_some());
        assert!(store.get_by_name("Rollback1").is_none());
    }

    #[test]
    fn test_durable_uncommitted_discarded() {
        let temp_dir = TempDir::new().unwrap();

        // Write committed data
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            store.ingest_json(r#"{"name": "Committed"}"#).unwrap();
        }

        // Simulate a crash during batch (no commit)
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            store.begin_batch().unwrap();
            store
                .batch_ingest_json(r#"{"name": "Uncommitted"}"#)
                .unwrap();
            // Drop without commit (simulates crash)
        }

        // Recover - uncommitted should be discarded
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            assert_eq!(store.len(), 1);
            assert!(store.get_by_name("Committed").is_some());
            // Uncommitted is gone
            assert!(store.get_by_name("Uncommitted").is_none());
        }
    }

    #[test]
    fn test_durable_checkpoint() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        store.ingest_json(r#"{"name": "Before"}"#).unwrap();
        let checkpoint_lsn = store.checkpoint().unwrap();
        store.ingest_json(r#"{"name": "After"}"#).unwrap();

        assert!(checkpoint_lsn > 0);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_durable_ingest_edge() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        // Create entities first
        store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        store.ingest_json(r#"{"name": "Bob", "age": 25}"#).unwrap();

        // Create edge
        let edge_id = store.ingest_edge("Alice", "KNOWS", "Bob").unwrap();
        assert!(edge_id > 0);

        // Verify graph traversal works
        let result = store.query_graph("Alice", "KNOWS", 1);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_durable_ingest_edge_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Create entities and edge, then "crash"
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
            store.ingest_json(r#"{"name": "Bob"}"#).unwrap();
            store.ingest_edge("Alice", "KNOWS", "Bob").unwrap();
        }

        // Recover and verify
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            assert!(store.get_by_name("Alice").is_some());
            assert!(store.get_by_name("Bob").is_some());
            // Edge entities are recovered; graph traversal should work
            let result = store.query_graph("Alice", "KNOWS", 1);
            assert!(!result.is_empty());
        }
    }

    #[test]
    fn test_durable_delete() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        let id1 = store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
        let _id2 = store.ingest_json(r#"{"name": "Bob"}"#).unwrap();
        assert_eq!(store.len(), 2);

        // Delete Alice
        store.delete(id1).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.get(id1).is_none());
        assert!(store.get_by_name("Alice").is_none());
    }

    #[test]
    fn test_durable_delete_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Ingest then delete, then "crash"
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            let id = store.ingest_json(r#"{"name": "Alice"}"#).unwrap();
            store.ingest_json(r#"{"name": "Bob"}"#).unwrap();
            store.delete(id).unwrap();
        }

        // Recover — Alice should still be deleted
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            assert_eq!(store.len(), 1);
            assert!(store.get_by_name("Alice").is_none());
            assert!(store.get_by_name("Bob").is_some());
        }
    }

    #[test]
    fn test_durable_update_fields() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

        let id = store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();

        // Update age
        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(31));
        store.update_fields(id, updates).unwrap();

        let record = store.get(id).unwrap();
        assert_eq!(record.get("age"), Some(&Value::Int(31)));
        assert_eq!(
            record.get("name"),
            Some(&Value::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_durable_update_fields_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Ingest then update, then "crash"
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            let id = store
                .ingest_json(r#"{"name": "Alice", "age": 30}"#)
                .unwrap();

            let mut updates = HashMap::new();
            updates.insert("age".to_string(), Value::Int(31));
            updates.insert("city".to_string(), Value::String("NYC".to_string()));
            store.update_fields(id, updates).unwrap();
        }

        // Recover — update should be applied
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            let record = store.get_by_name("Alice").unwrap();
            assert_eq!(record.get("age"), Some(&Value::Int(31)));
            assert_eq!(record.get("city"), Some(&Value::String("NYC".to_string())));
        }
    }

    #[test]
    fn test_durable_delete_nonexistent() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
        store.ingest_json(r#"{"name": "Alice"}"#).unwrap();

        // Deleting a non-existent record should fail
        let result = store.delete(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_delete_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Ingest two records in a batch, then delete one within a second batch, then crash
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

            // First batch: ingest two records
            store.begin_batch().unwrap();
            store.batch_ingest_json(r#"{"name": "Alice"}"#).unwrap();
            store.batch_ingest_json(r#"{"name": "Bob"}"#).unwrap();
            store.commit_batch().unwrap();
            assert_eq!(store.len(), 2);

            // Find Alice's ID
            let alice_id = store.get_by_name("Alice").unwrap().id;

            // Second batch: delete Alice
            store.begin_batch().unwrap();
            store.batch_delete(alice_id).unwrap();
            store.commit_batch().unwrap();
            assert_eq!(store.len(), 1);
        }

        // Recover — Alice should still be deleted, Bob should survive
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            assert_eq!(store.len(), 1);
            assert!(store.get_by_name("Alice").is_none());
            assert!(store.get_by_name("Bob").is_some());
        }
    }

    #[test]
    fn test_batch_update_recovery() {
        let temp_dir = TempDir::new().unwrap();

        // Ingest a record, then update it in a batch, then crash
        {
            let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();

            let id = store
                .ingest_json(r#"{"name": "Alice", "age": 30}"#)
                .unwrap();

            // Update via batch
            store.begin_batch().unwrap();
            let mut updates = HashMap::new();
            updates.insert("age".to_string(), Value::Int(31));
            updates.insert("city".to_string(), Value::String("NYC".to_string()));
            store.batch_update_fields(id, updates).unwrap();
            store.commit_batch().unwrap();

            // Verify update applied in-memory
            let record = store.get(id).unwrap();
            assert_eq!(record.get("age"), Some(&Value::Int(31)));
        }

        // Recover — update should be applied
        {
            let store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
            let record = store.get_by_name("Alice").unwrap();
            assert_eq!(record.get("age"), Some(&Value::Int(31)));
            assert_eq!(record.get("city"), Some(&Value::String("NYC".to_string())));
        }
    }

    #[test]
    fn test_durable_query_has_field() {
        let temp_dir = TempDir::new().unwrap();

        let mut store = DurableAmorphicStore::open(temp_dir.path()).unwrap();
        store
            .ingest_json(r#"{"name": "Alice", "age": 30}"#)
            .unwrap();
        store.ingest_json(r#"{"name": "Bob"}"#).unwrap();

        let with_age = store.query_has_field("age");
        assert_eq!(with_age.len(), 1);

        let with_name = store.query_has_field("name");
        assert_eq!(with_name.len(), 2);
    }
}
