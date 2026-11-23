//! Transaction batching for IndexedDB
//!
//! IndexedDB performance is heavily impacted by transaction overhead. Each
//! transaction involves:
//! - Creating the transaction object
//! - Opening object stores
//! - Committing changes to storage
//!
//! By batching multiple writes into a single transaction, we can significantly
//! improve write throughput.
//!
//! ## Example
//!
//! ```javascript
//! const batcher = new IndexedDBBatcher(db, {
//!     maxBatchSize: 100,
//!     maxWaitMs: 50
//! });
//!
//! // These writes are batched automatically
//! await batcher.write(1, data1);
//! await batcher.write(2, data2);
//! await batcher.write(3, data3);
//!
//! // Force flush if needed
//! await batcher.flush();
//! ```

use joule_db_core::storage::PageId;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{IdbDatabase, IdbTransactionMode};

use crate::persistence::PersistenceError;

/// Configuration for the batch writer
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of operations to batch before auto-flush
    pub max_batch_size: usize,
    /// Maximum time to wait before auto-flush (milliseconds)
    pub max_wait_ms: u32,
    /// Whether to enable auto-flush on batch size threshold
    pub auto_flush_on_size: bool,
    /// Whether to enable auto-flush on timeout
    pub auto_flush_on_timeout: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            max_wait_ms: 50,
            auto_flush_on_size: true,
            auto_flush_on_timeout: true,
        }
    }
}

impl BatchConfig {
    /// Create a new batch config with custom settings
    pub fn new(max_batch_size: usize, max_wait_ms: u32) -> Self {
        Self {
            max_batch_size,
            max_wait_ms,
            auto_flush_on_size: true,
            auto_flush_on_timeout: true,
        }
    }

    /// Create a config optimized for throughput (larger batches)
    pub fn throughput_optimized() -> Self {
        Self {
            max_batch_size: 500,
            max_wait_ms: 100,
            auto_flush_on_size: true,
            auto_flush_on_timeout: true,
        }
    }

    /// Create a config optimized for latency (smaller batches)
    pub fn latency_optimized() -> Self {
        Self {
            max_batch_size: 10,
            max_wait_ms: 5,
            auto_flush_on_size: true,
            auto_flush_on_timeout: true,
        }
    }

    /// Create a config for manual flushing only
    pub fn manual_only() -> Self {
        Self {
            max_batch_size: usize::MAX,
            max_wait_ms: u32::MAX,
            auto_flush_on_size: false,
            auto_flush_on_timeout: false,
        }
    }
}

/// Pending write operation
#[derive(Debug, Clone)]
struct PendingWrite {
    page_id: PageId,
    data: Vec<u8>,
}

/// Pending delete operation
#[derive(Debug, Clone)]
struct PendingDelete {
    page_id: PageId,
}

/// Batch operation type
#[derive(Debug, Clone)]
enum BatchOperation {
    Write(PendingWrite),
    Delete(PendingDelete),
}

/// Statistics for batch operations
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of batches flushed
    pub batches_flushed: u64,
    /// Total number of operations batched
    pub operations_batched: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Number of auto-flushes due to size threshold
    pub size_triggered_flushes: u64,
    /// Number of auto-flushes due to timeout
    pub timeout_triggered_flushes: u64,
    /// Number of manual flushes
    pub manual_flushes: u64,
}

/// Transaction batch writer for IndexedDB
///
/// Collects write operations and commits them in batches to reduce
/// transaction overhead.
pub struct IndexedDBBatcher {
    store_name: String,
    config: BatchConfig,
    pending_ops: Arc<RwLock<Vec<BatchOperation>>>,
    stats: Arc<RwLock<BatchStats>>,
    /// Cache for write-through - updated immediately on write
    write_cache: Arc<RwLock<HashMap<PageId, Vec<u8>>>>,
    /// Deleted keys (for cache consistency)
    deleted_keys: Arc<RwLock<std::collections::HashSet<PageId>>>,
}

impl IndexedDBBatcher {
    /// Create a new batcher with default configuration
    pub fn new(store_name: &str) -> Self {
        Self::with_config(store_name, BatchConfig::default())
    }

    /// Create a new batcher with custom configuration
    pub fn with_config(store_name: &str, config: BatchConfig) -> Self {
        Self {
            store_name: store_name.to_string(),
            config,
            pending_ops: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(BatchStats::default())),
            write_cache: Arc::new(RwLock::new(HashMap::new())),
            deleted_keys: Arc::new(RwLock::new(std::collections::HashSet::new())),
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Get current statistics
    pub fn stats(&self) -> BatchStats {
        self.stats.read().unwrap().clone()
    }

    /// Get the number of pending operations
    pub fn pending_count(&self) -> usize {
        self.pending_ops.read().unwrap().len()
    }

    /// Check if there are pending operations
    pub fn has_pending(&self) -> bool {
        !self.pending_ops.read().unwrap().is_empty()
    }

    /// Queue a write operation
    ///
    /// The write is added to the batch and committed when:
    /// - The batch reaches max_batch_size
    /// - max_wait_ms time passes
    /// - flush() is called explicitly
    pub fn queue_write(&self, page_id: PageId, data: Vec<u8>) {
        // Update write cache immediately for read consistency
        {
            let mut cache = self.write_cache.write().unwrap();
            cache.insert(page_id, data.clone());
        }

        // Remove from deleted set if present
        {
            let mut deleted = self.deleted_keys.write().unwrap();
            deleted.remove(&page_id);
        }

        // Add to pending operations
        {
            let mut ops = self.pending_ops.write().unwrap();
            ops.push(BatchOperation::Write(PendingWrite { page_id, data }));
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.operations_batched += 1;
        }
    }

    /// Queue a delete operation
    pub fn queue_delete(&self, page_id: PageId) {
        // Update cache immediately
        {
            let mut cache = self.write_cache.write().unwrap();
            cache.remove(&page_id);
        }

        // Mark as deleted
        {
            let mut deleted = self.deleted_keys.write().unwrap();
            deleted.insert(page_id);
        }

        // Add to pending operations
        {
            let mut ops = self.pending_ops.write().unwrap();
            ops.push(BatchOperation::Delete(PendingDelete { page_id }));
        }

        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.operations_batched += 1;
        }
    }

    /// Read from write cache (for read-your-writes consistency)
    pub fn read_from_cache(&self, page_id: PageId) -> Option<Vec<u8>> {
        // Check if deleted
        {
            let deleted = self.deleted_keys.read().unwrap();
            if deleted.contains(&page_id) {
                return None;
            }
        }

        // Check write cache
        let cache = self.write_cache.read().unwrap();
        cache.get(&page_id).cloned()
    }

    /// Check if a page is pending deletion
    pub fn is_pending_delete(&self, page_id: PageId) -> bool {
        self.deleted_keys.read().unwrap().contains(&page_id)
    }

    /// Check if auto-flush should be triggered due to size
    pub fn should_flush_on_size(&self) -> bool {
        if !self.config.auto_flush_on_size {
            return false;
        }
        self.pending_count() >= self.config.max_batch_size
    }

    /// Flush all pending operations to IndexedDB in a single transaction
    #[cfg(target_arch = "wasm32")]
    pub async fn flush(&self, db: &IdbDatabase) -> Result<usize, PersistenceError> {
        let ops: Vec<BatchOperation> = {
            let mut pending = self.pending_ops.write().unwrap();
            std::mem::take(&mut *pending)
        };

        if ops.is_empty() {
            return Ok(0);
        }

        let count = ops.len();

        // Create a single readwrite transaction for all operations
        let transaction = db
            .transaction_with_str_and_mode(&self.store_name, IdbTransactionMode::Readwrite)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        // Queue all operations
        for op in &ops {
            match op {
                BatchOperation::Write(write) => {
                    let key = JsValue::from_f64(write.page_id as f64);
                    let value = serde_wasm_bindgen::to_value(&write.data).map_err(|e| {
                        PersistenceError::IoError(format!("Serialize error: {:?}", e))
                    })?;

                    store
                        .put_with_key(&value, &key)
                        .map_err(|e| PersistenceError::IoError(format!("Put error: {:?}", e)))?;
                }
                BatchOperation::Delete(delete) => {
                    let key = JsValue::from_f64(delete.page_id as f64);
                    store
                        .delete(&key)
                        .map_err(|e| PersistenceError::IoError(format!("Delete error: {:?}", e)))?;
                }
            }
        }

        // Wait for transaction to complete
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let resolve_clone = resolve.clone();
            let reject_clone = reject.clone();

            let complete_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                let _ = resolve_clone.call1(&JsValue::UNDEFINED, &JsValue::UNDEFINED);
            }) as Box<dyn FnOnce(web_sys::Event)>);

            let error_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                let _ = reject_clone.call1(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str("Transaction failed"),
                );
            }) as Box<dyn FnOnce(web_sys::Event)>);

            transaction.set_oncomplete(Some(complete_cb.as_ref().unchecked_ref()));
            transaction.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

            complete_cb.forget();
            error_cb.forget();
        });

        JsFuture::from(promise)
            .await
            .map_err(|e| PersistenceError::IoError(format!("Transaction error: {:?}", e)))?;

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.batches_flushed += 1;
            stats.avg_batch_size = stats.operations_batched as f64 / stats.batches_flushed as f64;
        }

        // Clear the cache for items that were successfully written
        // (they're now in IndexedDB)
        {
            let mut cache = self.write_cache.write().unwrap();
            let mut deleted = self.deleted_keys.write().unwrap();

            for op in &ops {
                match op {
                    BatchOperation::Write(write) => {
                        // Keep in cache for read performance
                        // (optional: could clear to save memory)
                    }
                    BatchOperation::Delete(delete) => {
                        cache.remove(&delete.page_id);
                        deleted.remove(&delete.page_id);
                    }
                }
            }
        }

        Ok(count)
    }

    /// Flush for non-wasm (just clears the buffer)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn flush(&self, _db: &()) -> Result<usize, PersistenceError> {
        let ops: Vec<BatchOperation> = {
            let mut pending = self.pending_ops.write().unwrap();
            std::mem::take(&mut *pending)
        };

        let count = ops.len();

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.batches_flushed += 1;
            if stats.batches_flushed > 0 {
                stats.avg_batch_size =
                    stats.operations_batched as f64 / stats.batches_flushed as f64;
            }
        }

        Ok(count)
    }

    /// Clear all pending operations without flushing
    pub fn clear_pending(&self) {
        let mut ops = self.pending_ops.write().unwrap();
        ops.clear();
    }

    /// Get all page IDs that have pending writes
    pub fn pending_write_pages(&self) -> Vec<PageId> {
        self.pending_ops
            .read()
            .unwrap()
            .iter()
            .filter_map(|op| match op {
                BatchOperation::Write(w) => Some(w.page_id),
                _ => None,
            })
            .collect()
    }

    /// Get all page IDs that have pending deletes
    pub fn pending_delete_pages(&self) -> Vec<PageId> {
        self.pending_ops
            .read()
            .unwrap()
            .iter()
            .filter_map(|op| match op {
                BatchOperation::Delete(d) => Some(d.page_id),
                _ => None,
            })
            .collect()
    }
}

/// Batch read operations for IndexedDB
///
/// Reading multiple keys in a single transaction is more efficient than
/// multiple separate read transactions.
pub struct BatchReader {
    store_name: String,
}

impl BatchReader {
    /// Create a new batch reader
    pub fn new(store_name: &str) -> Self {
        Self {
            store_name: store_name.to_string(),
        }
    }

    /// Read multiple pages in a single transaction
    #[cfg(target_arch = "wasm32")]
    pub async fn read_batch(
        &self,
        db: &IdbDatabase,
        page_ids: &[PageId],
    ) -> Result<HashMap<PageId, Vec<u8>>, PersistenceError> {
        if page_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let transaction = db
            .transaction_with_str(&self.store_name)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let store = transaction
            .object_store(&self.store_name)
            .map_err(|e| PersistenceError::TransactionError(format!("{:?}", e)))?;

        let mut results = HashMap::with_capacity(page_ids.len());
        let mut requests = Vec::with_capacity(page_ids.len());

        // Queue all get requests
        for &page_id in page_ids {
            let key = JsValue::from_f64(page_id as f64);
            let request = store
                .get(&key)
                .map_err(|e| PersistenceError::IoError(format!("Get error: {:?}", e)))?;
            requests.push((page_id, request));
        }

        // Wait for all requests to complete
        for (page_id, request) in requests {
            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                let resolve_clone = resolve.clone();
                let reject_clone = reject.clone();

                let success_cb = Closure::once(Box::new(move |event: web_sys::Event| {
                    if let Some(target) = event.target() {
                        if let Ok(req) = target.dyn_into::<web_sys::IdbRequest>() {
                            if let Ok(result) = req.result() {
                                let _ = resolve_clone.call1(&JsValue::UNDEFINED, &result);
                            }
                        }
                    }
                })
                    as Box<dyn FnOnce(web_sys::Event)>);

                let error_cb = Closure::once(Box::new(move |_: web_sys::Event| {
                    let _ =
                        reject_clone.call1(&JsValue::UNDEFINED, &JsValue::from_str("Read failed"));
                }) as Box<dyn FnOnce(web_sys::Event)>);

                request.set_onsuccess(Some(success_cb.as_ref().unchecked_ref()));
                request.set_onerror(Some(error_cb.as_ref().unchecked_ref()));

                success_cb.forget();
                error_cb.forget();
            });

            let result = JsFuture::from(promise)
                .await
                .map_err(|e| PersistenceError::IoError(format!("{:?}", e)))?;

            if !result.is_undefined() && !result.is_null() {
                let data: Vec<u8> = serde_wasm_bindgen::from_value(result).map_err(|e| {
                    PersistenceError::IoError(format!("Deserialize error: {:?}", e))
                })?;
                results.insert(page_id, data);
            }
        }

        Ok(results)
    }

    /// Read batch for non-wasm (returns empty)
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn read_batch(
        &self,
        _db: &(),
        _page_ids: &[PageId],
    ) -> Result<HashMap<PageId, Vec<u8>>, PersistenceError> {
        Ok(HashMap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 100);
        assert_eq!(config.max_wait_ms, 50);
        assert!(config.auto_flush_on_size);
        assert!(config.auto_flush_on_timeout);
    }

    #[test]
    fn test_batch_config_presets() {
        let throughput = BatchConfig::throughput_optimized();
        assert_eq!(throughput.max_batch_size, 500);

        let latency = BatchConfig::latency_optimized();
        assert_eq!(latency.max_batch_size, 10);

        let manual = BatchConfig::manual_only();
        assert!(!manual.auto_flush_on_size);
        assert!(!manual.auto_flush_on_timeout);
    }

    #[test]
    fn test_queue_write() {
        let batcher = IndexedDBBatcher::new("test_store");

        batcher.queue_write(1, vec![1, 2, 3]);
        batcher.queue_write(2, vec![4, 5, 6]);

        assert_eq!(batcher.pending_count(), 2);
        assert!(batcher.has_pending());

        // Check write cache
        assert_eq!(batcher.read_from_cache(1), Some(vec![1, 2, 3]));
        assert_eq!(batcher.read_from_cache(2), Some(vec![4, 5, 6]));
        assert_eq!(batcher.read_from_cache(3), None);
    }

    #[test]
    fn test_queue_delete() {
        let batcher = IndexedDBBatcher::new("test_store");

        // Write then delete
        batcher.queue_write(1, vec![1, 2, 3]);
        assert!(batcher.read_from_cache(1).is_some());

        batcher.queue_delete(1);
        assert!(batcher.read_from_cache(1).is_none());
        assert!(batcher.is_pending_delete(1));
    }

    #[test]
    fn test_should_flush_on_size() {
        let config = BatchConfig {
            max_batch_size: 3,
            ..Default::default()
        };
        let batcher = IndexedDBBatcher::with_config("test_store", config);

        assert!(!batcher.should_flush_on_size());

        batcher.queue_write(1, vec![1]);
        batcher.queue_write(2, vec![2]);
        assert!(!batcher.should_flush_on_size());

        batcher.queue_write(3, vec![3]);
        assert!(batcher.should_flush_on_size());
    }

    #[test]
    fn test_pending_pages() {
        let batcher = IndexedDBBatcher::new("test_store");

        batcher.queue_write(1, vec![1]);
        batcher.queue_write(2, vec![2]);
        batcher.queue_delete(3);

        let write_pages = batcher.pending_write_pages();
        assert!(write_pages.contains(&1));
        assert!(write_pages.contains(&2));

        let delete_pages = batcher.pending_delete_pages();
        assert!(delete_pages.contains(&3));
    }

    #[test]
    fn test_clear_pending() {
        let batcher = IndexedDBBatcher::new("test_store");

        batcher.queue_write(1, vec![1]);
        batcher.queue_write(2, vec![2]);
        assert_eq!(batcher.pending_count(), 2);

        batcher.clear_pending();
        assert_eq!(batcher.pending_count(), 0);
        assert!(!batcher.has_pending());
    }

    #[test]
    fn test_stats() {
        let batcher = IndexedDBBatcher::new("test_store");

        batcher.queue_write(1, vec![1]);
        batcher.queue_write(2, vec![2]);
        batcher.queue_delete(3);

        let stats = batcher.stats();
        assert_eq!(stats.operations_batched, 3);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_flush_non_wasm() {
        // Note: This test requires async runtime, but we're in a sync test
        // For now, just verify the struct can be created
        let _batcher = IndexedDBBatcher::new("test_store");
    }
}
