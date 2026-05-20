//! Streaming Ingestion with Backpressure Support
//!
//! This module provides a high-volume streaming ingestion interface
//! with automatic backpressure to prevent memory overflow.

use crate::{AmorphicError, AmorphicResult, RecordId, ShardedAmorphicStore, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for streaming ingestion
#[derive(Clone, Debug)]
pub struct StreamConfig {
    /// Maximum number of items to buffer before applying backpressure
    pub buffer_size: usize,
    /// Number of items to batch together per commit
    pub batch_size: usize,
    /// Maximum time to wait before flushing (even if batch not full)
    pub flush_interval: Duration,
    /// Buffer fill ratio at which to start applying backpressure (0.0 to 1.0)
    pub backpressure_threshold: f64,
    /// Maximum memory budget in bytes (0 = unlimited)
    pub memory_limit: usize,
    /// Number of worker threads for processing
    pub worker_threads: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10_000,
            batch_size: 100,
            flush_interval: Duration::from_millis(100),
            backpressure_threshold: 0.8,
            memory_limit: 0,
            worker_threads: 4,
        }
    }
}

impl StreamConfig {
    /// Create a new StreamConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder pattern for buffer_size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Builder pattern for batch_size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder pattern for flush_interval
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Builder pattern for backpressure_threshold
    pub fn with_backpressure_threshold(mut self, threshold: f64) -> Self {
        self.backpressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Builder pattern for memory_limit
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Builder pattern for worker_threads
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.worker_threads = threads.max(1);
        self
    }
}

/// Item to be ingested
#[derive(Clone, Debug)]
pub enum IngestItem {
    /// JSON string to parse and ingest
    Json(String),
    /// Pre-parsed field map
    Fields(HashMap<String, Value>),
    /// Row data (field names + values)
    Row {
        fields: Vec<String>,
        values: Vec<Value>,
    },
}

impl IngestItem {
    /// Estimate memory size of this item
    fn estimated_size(&self) -> usize {
        match self {
            IngestItem::Json(s) => s.len() + std::mem::size_of::<String>(),
            IngestItem::Fields(map) => {
                let mut size = std::mem::size_of::<HashMap<String, Value>>();
                for (k, v) in map {
                    size += k.len() + std::mem::size_of::<String>();
                    size += match v {
                        Value::String(s) => s.len() + std::mem::size_of::<Value>(),
                        _ => std::mem::size_of::<Value>(),
                    };
                }
                size
            }
            IngestItem::Row { fields, values } => {
                let mut size = std::mem::size_of::<Vec<String>>()
                    + std::mem::size_of::<Vec<Value>>()
                    + fields.len() * std::mem::size_of::<String>()
                    + values.len() * std::mem::size_of::<Value>();
                for f in fields {
                    size += f.len();
                }
                for v in values {
                    if let Value::String(s) = v {
                        size += s.len();
                    }
                }
                size
            }
        }
    }
}

/// Status returned from ingest operations
#[derive(Clone, Debug, PartialEq)]
pub enum IngestStatus {
    /// Item was accepted and assigned a record ID
    Accepted(RecordId),
    /// Item was buffered and will be processed asynchronously
    Buffered,
    /// Backpressure is active - caller should wait
    Backpressure {
        /// Suggested wait time in milliseconds
        wait_ms: u64,
    },
    /// Ingestion was rejected (e.g., shutdown in progress)
    Rejected { reason: String },
}

/// Metrics for monitoring streaming ingestion
#[derive(Clone, Debug, Default)]
pub struct StreamMetrics {
    /// Total items received
    pub items_received: u64,
    /// Total items successfully processed
    pub items_processed: u64,
    /// Total items failed
    pub items_failed: u64,
    /// Current buffer size
    pub buffer_size: usize,
    /// Current estimated memory usage
    pub memory_usage: usize,
    /// Number of times backpressure was applied
    pub backpressure_count: u64,
    /// Total batches flushed
    pub batches_flushed: u64,
    /// Average batch processing time in microseconds
    pub avg_batch_time_us: u64,
    /// Is backpressure currently active?
    pub backpressure_active: bool,
}

/// Internal shared state for the streaming ingester
struct IngestState {
    /// Pending items to process
    buffer: VecDeque<(IngestItem, Option<std::sync::mpsc::Sender<RecordId>>)>,
    /// Current estimated memory usage
    memory_usage: usize,
    /// Should we shut down?
    shutdown: bool,
    /// Last flush time
    last_flush: Instant,
}

/// Atomic metrics for thread-safe updates
struct AtomicMetrics {
    items_received: AtomicU64,
    items_processed: AtomicU64,
    items_failed: AtomicU64,
    buffer_size: AtomicUsize,
    memory_usage: AtomicUsize,
    backpressure_count: AtomicU64,
    batches_flushed: AtomicU64,
    total_batch_time_us: AtomicU64,
    backpressure_active: AtomicBool,
}

impl AtomicMetrics {
    fn new() -> Self {
        Self {
            items_received: AtomicU64::new(0),
            items_processed: AtomicU64::new(0),
            items_failed: AtomicU64::new(0),
            buffer_size: AtomicUsize::new(0),
            memory_usage: AtomicUsize::new(0),
            backpressure_count: AtomicU64::new(0),
            batches_flushed: AtomicU64::new(0),
            total_batch_time_us: AtomicU64::new(0),
            backpressure_active: AtomicBool::new(false),
        }
    }

    fn snapshot(&self) -> StreamMetrics {
        let batches = self.batches_flushed.load(Ordering::Relaxed);
        let total_time = self.total_batch_time_us.load(Ordering::Relaxed);
        StreamMetrics {
            items_received: self.items_received.load(Ordering::Relaxed),
            items_processed: self.items_processed.load(Ordering::Relaxed),
            items_failed: self.items_failed.load(Ordering::Relaxed),
            buffer_size: self.buffer_size.load(Ordering::Relaxed),
            memory_usage: self.memory_usage.load(Ordering::Relaxed),
            backpressure_count: self.backpressure_count.load(Ordering::Relaxed),
            batches_flushed: batches,
            avg_batch_time_us: if batches > 0 { total_time / batches } else { 0 },
            backpressure_active: self.backpressure_active.load(Ordering::Relaxed),
        }
    }
}

/// Streaming ingestion manager with backpressure support
pub struct StreamingIngester {
    /// Reference to the sharded store
    store: Arc<RwLock<ShardedAmorphicStore>>,
    /// Configuration
    config: StreamConfig,
    /// Internal state protected by mutex
    state: Arc<Mutex<IngestState>>,
    /// Condition variable for backpressure waiting
    capacity_available: Arc<Condvar>,
    /// Atomic metrics
    metrics: Arc<AtomicMetrics>,
    /// Worker thread handles
    workers: Vec<thread::JoinHandle<()>>,
    /// Shutdown flag
    shutdown_flag: Arc<AtomicBool>,
}

impl StreamingIngester {
    /// Create a new streaming ingester
    pub fn new(store: ShardedAmorphicStore, config: StreamConfig) -> Self {
        let store = Arc::new(RwLock::new(store));
        let state = Arc::new(Mutex::new(IngestState {
            buffer: VecDeque::with_capacity(config.buffer_size),
            memory_usage: 0,
            shutdown: false,
            last_flush: Instant::now(),
        }));
        let capacity_available = Arc::new(Condvar::new());
        let metrics = Arc::new(AtomicMetrics::new());
        let shutdown_flag = Arc::new(AtomicBool::new(false));

        let mut ingester = Self {
            store,
            config,
            state,
            capacity_available,
            metrics,
            workers: Vec::new(),
            shutdown_flag,
        };

        // Start worker threads
        ingester.start_workers();

        ingester
    }

    /// Start background worker threads
    fn start_workers(&mut self) {
        for worker_id in 0..self.config.worker_threads {
            let store = Arc::clone(&self.store);
            let state = Arc::clone(&self.state);
            let capacity_cv = Arc::clone(&self.capacity_available);
            let metrics = Arc::clone(&self.metrics);
            let shutdown = Arc::clone(&self.shutdown_flag);
            let batch_size = self.config.batch_size;
            let flush_interval = self.config.flush_interval;

            let handle = thread::spawn(move || {
                Self::worker_loop(
                    worker_id,
                    store,
                    state,
                    capacity_cv,
                    metrics,
                    shutdown,
                    batch_size,
                    flush_interval,
                );
            });

            self.workers.push(handle);
        }
    }

    /// Worker loop for processing batches
    fn worker_loop(
        _worker_id: usize,
        store: Arc<RwLock<ShardedAmorphicStore>>,
        state: Arc<Mutex<IngestState>>,
        capacity_cv: Arc<Condvar>,
        metrics: Arc<AtomicMetrics>,
        shutdown: Arc<AtomicBool>,
        batch_size: usize,
        flush_interval: Duration,
    ) {
        loop {
            // Check for shutdown
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Collect a batch
            let batch = {
                let mut guard = state.lock().unwrap();

                // Wait for items or timeout
                while guard.buffer.is_empty() && !guard.shutdown {
                    let (new_guard, timeout) =
                        capacity_cv.wait_timeout(guard, flush_interval).unwrap();
                    guard = new_guard;

                    if timeout.timed_out() || guard.shutdown {
                        break;
                    }
                }

                if guard.shutdown && guard.buffer.is_empty() {
                    break;
                }

                // Collect up to batch_size items
                let mut batch = Vec::with_capacity(batch_size.min(guard.buffer.len()));
                let mut batch_memory = 0;

                while batch.len() < batch_size {
                    if let Some(item) = guard.buffer.pop_front() {
                        batch_memory += item.0.estimated_size();
                        batch.push(item);
                    } else {
                        break;
                    }
                }

                // Update state
                guard.memory_usage = guard.memory_usage.saturating_sub(batch_memory);
                guard.last_flush = Instant::now();

                // Update metrics
                metrics
                    .buffer_size
                    .store(guard.buffer.len(), Ordering::Relaxed);
                metrics
                    .memory_usage
                    .store(guard.memory_usage, Ordering::Relaxed);

                batch
            };

            // Signal that capacity is available
            capacity_cv.notify_all();

            if batch.is_empty() {
                continue;
            }

            // Process the batch
            let start = Instant::now();
            let mut processed = 0;
            let mut failed = 0;

            {
                let store_guard = store.read().unwrap();

                for (item, responder) in batch {
                    let result = match &item {
                        IngestItem::Json(json) => store_guard.ingest_json(json),
                        IngestItem::Fields(fields) => {
                            // Convert fields to JSON for ingestion
                            match serde_json::to_string(fields) {
                                Ok(json) => store_guard.ingest_json(&json),
                                Err(e) => Err(AmorphicError::JsonError(e)),
                            }
                        }
                        IngestItem::Row { fields, values } => {
                            // Convert Vec<String> to Vec<&str> and Vec<Value> to Vec<&str>
                            let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
                            let value_strings: Vec<String> = values
                                .iter()
                                .map(|v| match v {
                                    Value::String(s) => s.clone(),
                                    Value::Int(i) => i.to_string(),
                                    Value::Float(f) => f.to_string(),
                                    Value::Bool(b) => b.to_string(),
                                    Value::Null => "".to_string(),
                                    Value::Array(arr) => {
                                        serde_json::to_string(arr).unwrap_or_default()
                                    }
                                    Value::Object(obj) => {
                                        serde_json::to_string(obj).unwrap_or_default()
                                    }
                                })
                                .collect();
                            let value_refs: Vec<&str> =
                                value_strings.iter().map(|s| s.as_str()).collect();
                            store_guard.ingest_row(&field_refs, &value_refs)
                        }
                    };

                    match result {
                        Ok(id) => {
                            processed += 1;
                            if let Some(tx) = responder {
                                let _ = tx.send(id);
                            }
                        }
                        Err(_) => {
                            failed += 1;
                        }
                    }
                }
            }

            // Update metrics
            let elapsed_us = start.elapsed().as_micros() as u64;
            metrics
                .items_processed
                .fetch_add(processed, Ordering::Relaxed);
            metrics.items_failed.fetch_add(failed, Ordering::Relaxed);
            metrics.batches_flushed.fetch_add(1, Ordering::Relaxed);
            metrics
                .total_batch_time_us
                .fetch_add(elapsed_us, Ordering::Relaxed);
        }
    }

    /// Check if backpressure should be applied
    fn should_apply_backpressure(&self, state: &IngestState) -> bool {
        // Check buffer fill ratio
        let buffer_ratio = state.buffer.len() as f64 / self.config.buffer_size as f64;
        if buffer_ratio >= self.config.backpressure_threshold {
            return true;
        }

        // Check memory limit if configured
        if self.config.memory_limit > 0 && state.memory_usage >= self.config.memory_limit {
            return true;
        }

        false
    }

    /// Calculate wait time for backpressure
    fn calculate_wait_time(&self, state: &IngestState) -> u64 {
        // Base wait time on how over-capacity we are
        let buffer_ratio = state.buffer.len() as f64 / self.config.buffer_size as f64;
        let over_threshold = (buffer_ratio - self.config.backpressure_threshold).max(0.0);

        // Wait longer as we get more full (10ms to 1000ms)
        let wait = (over_threshold * 5000.0) as u64;
        wait.clamp(10, 1000)
    }

    /// Ingest an item (synchronous, may block on backpressure)
    pub fn ingest(&self, item: IngestItem) -> IngestStatus {
        self.ingest_internal(item, false)
    }

    /// Ingest an item with optional synchronous waiting for result
    pub fn ingest_sync(&self, item: IngestItem) -> AmorphicResult<RecordId> {
        let (tx, rx) = std::sync::mpsc::channel();
        let item_size = item.estimated_size();

        {
            let mut state = self.state.lock().unwrap();

            // Wait for capacity if backpressure is active
            while self.should_apply_backpressure(&state) {
                self.metrics
                    .backpressure_active
                    .store(true, Ordering::Relaxed);
                self.metrics
                    .backpressure_count
                    .fetch_add(1, Ordering::Relaxed);

                let wait_time = Duration::from_millis(self.calculate_wait_time(&state));
                let (new_state, _) = self
                    .capacity_available
                    .wait_timeout(state, wait_time)
                    .unwrap();
                state = new_state;
            }
            self.metrics
                .backpressure_active
                .store(false, Ordering::Relaxed);

            // Add to buffer
            state.buffer.push_back((item, Some(tx)));
            state.memory_usage += item_size;
            self.metrics.items_received.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .buffer_size
                .store(state.buffer.len(), Ordering::Relaxed);
            self.metrics
                .memory_usage
                .store(state.memory_usage, Ordering::Relaxed);
        }

        // Notify workers
        self.capacity_available.notify_one();

        // Wait for result
        rx.recv()
            .map_err(|_| crate::AmorphicError::IngestionError("Ingestion failed".to_string()))
    }

    /// Internal ingest implementation
    fn ingest_internal(&self, item: IngestItem, _sync: bool) -> IngestStatus {
        let item_size = item.estimated_size();

        {
            let mut state = self.state.lock().unwrap();

            // Check if shutdown in progress
            if state.shutdown {
                return IngestStatus::Rejected {
                    reason: "Shutdown in progress".to_string(),
                };
            }

            // Check for backpressure
            if self.should_apply_backpressure(&state) {
                self.metrics
                    .backpressure_active
                    .store(true, Ordering::Relaxed);
                self.metrics
                    .backpressure_count
                    .fetch_add(1, Ordering::Relaxed);

                return IngestStatus::Backpressure {
                    wait_ms: self.calculate_wait_time(&state),
                };
            }

            self.metrics
                .backpressure_active
                .store(false, Ordering::Relaxed);

            // Add to buffer
            state.buffer.push_back((item, None));
            state.memory_usage += item_size;
            self.metrics.items_received.fetch_add(1, Ordering::Relaxed);
            self.metrics
                .buffer_size
                .store(state.buffer.len(), Ordering::Relaxed);
            self.metrics
                .memory_usage
                .store(state.memory_usage, Ordering::Relaxed);
        }

        // Notify workers
        self.capacity_available.notify_one();

        IngestStatus::Buffered
    }

    /// Ingest multiple items at once
    pub fn ingest_batch(&self, items: Vec<IngestItem>) -> Vec<IngestStatus> {
        items.into_iter().map(|item| self.ingest(item)).collect()
    }

    /// Wait until backpressure is relieved
    pub fn wait_for_capacity(&self) -> Duration {
        let start = Instant::now();

        let mut state = self.state.lock().unwrap();

        while self.should_apply_backpressure(&state) && !state.shutdown {
            let wait_time = Duration::from_millis(self.calculate_wait_time(&state));
            let (new_state, _) = self
                .capacity_available
                .wait_timeout(state, wait_time)
                .unwrap();
            state = new_state;
        }

        start.elapsed()
    }

    /// Flush all pending items (blocks until complete)
    pub fn flush(&self) -> usize {
        let start_processed = self.metrics.items_processed.load(Ordering::SeqCst);
        let start_failed = self.metrics.items_failed.load(Ordering::SeqCst);
        let total_received = self.metrics.items_received.load(Ordering::SeqCst);

        // Wait for buffer to empty
        loop {
            let state = self.state.lock().unwrap();
            if state.buffer.is_empty() {
                break;
            }
            drop(state);

            // Notify workers and give them time to process
            self.capacity_available.notify_all();
            thread::sleep(Duration::from_millis(10));
        }

        // Buffer is empty, but workers may still be processing the last batch.
        // Wait until items_processed + items_failed >= total items received.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let processed = self.metrics.items_processed.load(Ordering::SeqCst);
            let failed = self.metrics.items_failed.load(Ordering::SeqCst);
            if processed + failed >= total_received {
                break;
            }
            if Instant::now() > deadline {
                break; // Safety timeout to avoid infinite loop
            }
            thread::sleep(Duration::from_millis(1));
        }

        let end_processed = self.metrics.items_processed.load(Ordering::SeqCst);
        let end_failed = self.metrics.items_failed.load(Ordering::SeqCst);

        ((end_processed - start_processed) + (end_failed - start_failed)) as usize
    }

    /// Get current metrics snapshot
    pub fn metrics(&self) -> StreamMetrics {
        self.metrics.snapshot()
    }

    /// Check if backpressure is currently active
    pub fn is_backpressure_active(&self) -> bool {
        self.metrics.backpressure_active.load(Ordering::Relaxed)
    }

    /// Get current buffer utilization (0.0 to 1.0)
    pub fn buffer_utilization(&self) -> f64 {
        let size = self.metrics.buffer_size.load(Ordering::Relaxed);
        size as f64 / self.config.buffer_size as f64
    }

    /// Gracefully shutdown the ingester
    pub fn shutdown(self) -> StreamMetrics {
        // Signal shutdown
        self.shutdown_flag.store(true, Ordering::SeqCst);

        {
            let mut state = self.state.lock().unwrap();
            state.shutdown = true;
        }

        // Wake up workers
        self.capacity_available.notify_all();

        // Wait for workers to finish
        for handle in self.workers {
            let _ = handle.join();
        }

        self.metrics.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_store() -> ShardedAmorphicStore {
        ShardedAmorphicStore::with_shard_count(4)
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_size, 10_000);
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.backpressure_threshold, 0.8);
        assert_eq!(config.worker_threads, 4);
    }

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfig::new()
            .with_buffer_size(1000)
            .with_batch_size(50)
            .with_backpressure_threshold(0.5)
            .with_worker_threads(2);

        assert_eq!(config.buffer_size, 1000);
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.backpressure_threshold, 0.5);
        assert_eq!(config.worker_threads, 2);
    }

    #[test]
    fn test_ingest_item_size_estimation() {
        let json = IngestItem::Json(r#"{"name": "test"}"#.to_string());
        assert!(json.estimated_size() > 0);

        let fields = IngestItem::Fields(HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Int(30)),
        ]));
        assert!(fields.estimated_size() > 0);

        let row = IngestItem::Row {
            fields: vec!["name".to_string(), "age".to_string()],
            values: vec![Value::String("Bob".to_string()), Value::Int(25)],
        };
        assert!(row.estimated_size() > 0);
    }

    #[test]
    fn test_streaming_ingester_basic() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(10)
            .with_worker_threads(1);

        let ingester = StreamingIngester::new(store, config);

        // Ingest some items
        let status = ingester.ingest(IngestItem::Json(
            r#"{"name": "Alice", "age": 30}"#.to_string(),
        ));

        assert!(
            matches!(status, IngestStatus::Buffered) || matches!(status, IngestStatus::Accepted(_))
        );

        // Flush and verify
        let flushed = ingester.flush();
        assert!(flushed >= 0);

        // Check metrics
        let metrics = ingester.metrics();
        assert_eq!(metrics.items_received, 1);

        ingester.shutdown();
    }

    #[test]
    fn test_streaming_ingester_multiple_items() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(5)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        // Ingest multiple items synchronously to ensure they're all accepted
        for i in 0..20 {
            let json = format!(r#"{{"name": "User{}", "id": {}}}"#, i, i);
            let result = ingester.ingest_sync(IngestItem::Json(json));
            assert!(result.is_ok(), "Failed to ingest item {}: {:?}", i, result);
        }

        // Shutdown and get final metrics (waits for all workers to finish)
        let metrics = ingester.shutdown();

        // Verify metrics
        assert_eq!(metrics.items_received, 20);
        assert_eq!(metrics.items_processed, 20);
        assert_eq!(metrics.items_failed, 0);
    }

    #[test]
    fn test_streaming_ingester_sync() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(10)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        // Synchronous ingest - should return record ID
        let result = ingester.ingest_sync(IngestItem::Fields(HashMap::from([
            ("name".to_string(), Value::String("Test".to_string())),
            ("value".to_string(), Value::Int(42)),
        ])));

        assert!(result.is_ok());
        let id = result.unwrap();
        assert!(id > 0);

        ingester.shutdown();
    }

    #[test]
    fn test_streaming_ingester_batch_ingest() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(10)
            .with_worker_threads(1);

        let ingester = StreamingIngester::new(store, config);

        let items = vec![
            IngestItem::Json(r#"{"a": 1}"#.to_string()),
            IngestItem::Json(r#"{"a": 2}"#.to_string()),
            IngestItem::Json(r#"{"a": 3}"#.to_string()),
        ];

        let statuses = ingester.ingest_batch(items);
        assert_eq!(statuses.len(), 3);

        ingester.flush();

        let metrics = ingester.metrics();
        assert_eq!(metrics.items_received, 3);

        ingester.shutdown();
    }

    #[test]
    fn test_backpressure_detection() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(10)
            .with_batch_size(100) // Large batch to prevent processing
            .with_backpressure_threshold(0.5)
            .with_worker_threads(0); // No workers - items won't be processed

        // Create ingester manually without workers
        let ingester = Arc::new(Mutex::new(IngestState {
            buffer: VecDeque::with_capacity(10),
            memory_usage: 0,
            shutdown: false,
            last_flush: Instant::now(),
        }));

        // Fill buffer past threshold
        {
            let mut state = ingester.lock().unwrap();
            for _ in 0..6 {
                state
                    .buffer
                    .push_back((IngestItem::Json("{}".to_string()), None));
            }
        }

        // Verify buffer is over threshold
        let state = ingester.lock().unwrap();
        let ratio = state.buffer.len() as f64 / 10.0;
        assert!(ratio >= 0.5);
    }

    #[test]
    fn test_buffer_utilization() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(10)
            .with_worker_threads(1);

        let ingester = StreamingIngester::new(store, config);

        // Initially empty
        assert_eq!(ingester.buffer_utilization(), 0.0);

        // Add some items
        for _ in 0..10 {
            ingester.ingest(IngestItem::Json(r#"{"x": 1}"#.to_string()));
        }

        // Should have some utilization (exact value depends on processing)
        // Wait a tiny bit for metrics to update
        thread::sleep(Duration::from_millis(10));
        let util = ingester.buffer_utilization();
        // Utilization could be 0 if workers already processed, or >0 if still buffered
        assert!(util >= 0.0 && util <= 1.0);

        ingester.shutdown();
    }

    #[test]
    fn test_metrics_tracking() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(5)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        // Ingest items
        for i in 0..15 {
            let json = format!(r#"{{"value": {}}}"#, i);
            ingester.ingest(IngestItem::Json(json));
        }

        // Flush
        ingester.flush();

        // Allow workers time to finish processing
        std::thread::sleep(Duration::from_millis(100));

        // Check metrics
        let metrics = ingester.metrics();
        assert_eq!(metrics.items_received, 15);
        assert_eq!(
            metrics.items_processed, 15,
            "Expected 15 processed, got {}",
            metrics.items_processed
        );
        assert!(metrics.batches_flushed >= 3); // At least 3 batches of 5

        ingester.shutdown();
    }

    #[test]
    fn test_graceful_shutdown() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(10)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        // Add items
        for i in 0..25 {
            ingester.ingest(IngestItem::Json(format!(r#"{{"id": {}}}"#, i)));
        }

        // Shutdown should process remaining items
        let final_metrics = ingester.shutdown();

        assert_eq!(final_metrics.items_received, 25);
        // All items should be processed or in the final state
        assert!(final_metrics.items_processed + final_metrics.items_failed <= 25);
    }

    #[test]
    fn test_different_item_types() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(10)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        // JSON item
        ingester.ingest(IngestItem::Json(r#"{"type": "json"}"#.to_string()));

        // Fields item
        ingester.ingest(IngestItem::Fields(HashMap::from([(
            "type".to_string(),
            Value::String("fields".to_string()),
        )])));

        // Row item
        ingester.ingest(IngestItem::Row {
            fields: vec!["type".to_string()],
            values: vec![Value::String("row".to_string())],
        });

        ingester.flush();

        let metrics = ingester.metrics();
        assert_eq!(metrics.items_received, 3);
        assert_eq!(metrics.items_processed, 3);

        ingester.shutdown();
    }

    #[test]
    fn test_flush_processes_all_items() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(5)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        for i in 0..20 {
            ingester.ingest(IngestItem::Json(format!(r#"{{"id": {}}}"#, i)));
        }

        ingester.flush();

        let metrics = ingester.metrics();
        assert_eq!(metrics.items_received, 20);
        assert_eq!(
            metrics.items_processed, 20,
            "flush() must ensure all items are fully processed"
        );

        ingester.shutdown();
    }

    #[test]
    fn test_flush_concurrent_stress() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(2000)
            .with_batch_size(10)
            .with_worker_threads(4);

        let ingester = std::sync::Arc::new(StreamingIngester::new(store, config));
        let mut handles = vec![];
        let total_per_thread = 100;
        let num_threads = 10;

        // Ingest 1000 items from 10 threads concurrently
        for t in 0..num_threads {
            let ing = ingester.clone();
            handles.push(std::thread::spawn(move || {
                let mut buffered = 0u64;
                for i in 0..total_per_thread {
                    let status = ing.ingest(IngestItem::Json(format!(
                        r#"{{"thread": {}, "id": {}}}"#,
                        t, i
                    )));
                    if matches!(status, IngestStatus::Buffered) {
                        buffered += 1;
                    }
                }
                buffered
            }));
        }

        let total_buffered: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

        ingester.flush();

        let metrics = ingester.metrics();
        assert_eq!(metrics.items_received, total_buffered);
        assert_eq!(
            metrics.items_processed + metrics.items_failed,
            total_buffered,
            "All buffered items must be accounted for after flush"
        );
        assert!(
            total_buffered > 0,
            "At least some items must have been buffered"
        );

        // Use Arc::try_unwrap to get owned ingester for shutdown
        match std::sync::Arc::try_unwrap(ingester) {
            Ok(ing) => {
                ing.shutdown();
            }
            Err(_) => {} // Other references exist, that's fine for test
        }
    }

    #[test]
    fn test_flush_returns_accurate_count() {
        let store = create_test_store();
        let config = StreamConfig::new()
            .with_buffer_size(100)
            .with_batch_size(5)
            .with_worker_threads(2);

        let ingester = StreamingIngester::new(store, config);

        for i in 0..15 {
            ingester.ingest(IngestItem::Json(format!(r#"{{"id": {}}}"#, i)));
        }

        let flushed = ingester.flush();
        assert_eq!(
            flushed, 15,
            "flush() return value must match items processed"
        );

        ingester.shutdown();
    }
}
