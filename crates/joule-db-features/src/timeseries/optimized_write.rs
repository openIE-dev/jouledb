//! Optimized Time-Series Write Path
//!
//! High-performance batch writes for time-series data targeting 1M+ writes/second.
//! Implements columnar batching, compression, and efficient memory layout.

use crate::timeseries::{DataPoint, TimeSeriesStore};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Write buffer for batching time-series writes
pub struct WriteBuffer {
    /// Metric name
    metric: String,
    /// Buffered data points
    points: Vec<DataPoint>,
    /// Buffer size threshold (flush when reached)
    threshold: usize,
    /// Store reference
    store: Arc<TimeSeriesStore>,
}

impl WriteBuffer {
    /// Create new write buffer
    pub fn new(metric: String, store: Arc<TimeSeriesStore>, threshold: usize) -> Self {
        Self {
            metric,
            points: Vec::with_capacity(threshold),
            threshold,
            store,
        }
    }

    /// Add data point to buffer
    pub fn add(&mut self, point: DataPoint) {
        self.points.push(point);

        // Flush if threshold reached
        if self.points.len() >= self.threshold {
            self.flush();
        }
    }

    /// Add multiple data points
    pub fn add_batch(&mut self, mut points: Vec<DataPoint>) {
        self.points.append(&mut points);

        // Flush if threshold reached
        if self.points.len() >= self.threshold {
            self.flush();
        }
    }

    /// Flush buffer to store
    pub fn flush(&mut self) {
        if self.points.is_empty() {
            return;
        }

        // Sort by timestamp for better compression
        self.points.sort_by_key(|p| p.timestamp);

        // Group by partition for efficient writes
        let mut partition_map: HashMap<i64, Vec<DataPoint>> = HashMap::new();

        for point in self.points.drain(..) {
            // Calculate partition
            let partition_duration = 3600_000_000_000i64; // 1 hour in nanoseconds
            let partition_start = (point.timestamp / partition_duration) * partition_duration;

            partition_map
                .entry(partition_start)
                .or_insert_with(Vec::new)
                .push(point);
        }

        // Write each partition batch
        for (_, points) in partition_map {
            // Use optimized batch write
            self.store.write_batch_optimized(&self.metric, points);
        }
    }

    /// Force flush remaining points
    pub fn force_flush(&mut self) {
        self.flush();
    }
}

/// High-performance time-series writer
pub struct OptimizedTimeSeriesWriter {
    /// Write buffers per metric
    buffers: Arc<RwLock<HashMap<String, WriteBuffer>>>,
    /// Default buffer threshold
    default_threshold: usize,
    /// Store reference
    store: Arc<TimeSeriesStore>,
}

impl OptimizedTimeSeriesWriter {
    /// Create new optimized writer
    pub fn new(store: Arc<TimeSeriesStore>, default_threshold: usize) -> Self {
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
            default_threshold,
            store,
        }
    }

    /// Write data point (buffered)
    pub fn write(&self, metric: &str, point: DataPoint) {
        let mut buffers = self.buffers.write().unwrap();
        let buffer = buffers.entry(metric.to_string()).or_insert_with(|| {
            WriteBuffer::new(
                metric.to_string(),
                self.store.clone(),
                self.default_threshold,
            )
        });

        buffer.add(point);
    }

    /// Write batch of data points (optimized)
    pub fn write_batch(&self, metric: &str, points: Vec<DataPoint>) {
        let mut buffers = self.buffers.write().unwrap();
        let buffer = buffers.entry(metric.to_string()).or_insert_with(|| {
            WriteBuffer::new(
                metric.to_string(),
                self.store.clone(),
                self.default_threshold,
            )
        });

        buffer.add_batch(points);
    }

    /// Flush all buffers
    pub fn flush_all(&self) {
        let mut buffers = self.buffers.write().unwrap();
        for buffer in buffers.values_mut() {
            buffer.force_flush();
        }
    }

    /// Flush specific metric buffer
    pub fn flush_metric(&self, metric: &str) {
        let mut buffers = self.buffers.write().unwrap();
        if let Some(buffer) = buffers.get_mut(metric) {
            buffer.force_flush();
        }
    }
}
