//! Dynamic batching for inference: request coalescing, padding, timeout-based
//! dispatch, and throughput optimisation.
//!
//! Collects individual inference requests into batches to maximise hardware
//! utilisation. Supports configurable max batch size, timeout-based
//! flushing, input padding/truncation, priority queues, and throughput
//! tracking.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

// ── Request ────────────────────────────────────────────────────

/// Unique request identifier.
pub type RequestId = u64;

/// A single inference request with its input data.
#[derive(Debug, Clone)]
pub struct InferRequest {
    pub id: RequestId,
    pub input: Vec<f64>,
    pub input_len: usize,
    pub priority: Priority,
    pub enqueued_at: Instant,
    pub metadata: HashMap<String, String>,
}

impl InferRequest {
    pub fn new(id: RequestId, input: Vec<f64>) -> Self {
        let input_len = input.len();
        Self {
            id,
            input,
            input_len,
            priority: Priority::Normal,
            enqueued_at: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_priority(mut self, p: Priority) -> Self {
        self.priority = p;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }

    /// Time spent waiting in the queue.
    pub fn wait_time(&self) -> Duration {
        self.enqueued_at.elapsed()
    }
}

impl fmt::Display for InferRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InferRequest(id={}, len={}, priority={})",
            self.id, self.input_len, self.priority
        )
    }
}

// ── Priority ───────────────────────────────────────────────────

/// Request priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Priority::Low => write!(f, "low"),
            Priority::Normal => write!(f, "normal"),
            Priority::High => write!(f, "high"),
            Priority::Critical => write!(f, "critical"),
        }
    }
}

// ── Padding Strategy ───────────────────────────────────────────

/// Strategy for handling variable-length inputs in a batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingStrategy {
    /// Pad all inputs to the max length in the batch.
    PadToMax,
    /// Pad to a fixed length.
    PadToFixed(usize),
    /// Truncate inputs longer than the given length.
    Truncate(usize),
    /// No padding (all inputs must be the same length).
    None,
}

impl PaddingStrategy {
    /// Apply padding to a single input.
    pub fn apply(&self, input: &[f64], pad_value: f64, max_in_batch: usize) -> Vec<f64> {
        match self {
            PaddingStrategy::PadToMax => {
                let mut padded = input.to_vec();
                padded.resize(max_in_batch, pad_value);
                padded
            }
            PaddingStrategy::PadToFixed(len) => {
                let mut padded = input.to_vec();
                padded.resize(*len, pad_value);
                padded
            }
            PaddingStrategy::Truncate(len) => {
                if input.len() > *len {
                    input[..*len].to_vec()
                } else {
                    input.to_vec()
                }
            }
            PaddingStrategy::None => input.to_vec(),
        }
    }
}

impl fmt::Display for PaddingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PaddingStrategy::PadToMax => write!(f, "pad-to-max"),
            PaddingStrategy::PadToFixed(n) => write!(f, "pad-to-{n}"),
            PaddingStrategy::Truncate(n) => write!(f, "truncate-{n}"),
            PaddingStrategy::None => write!(f, "none"),
        }
    }
}

// ── Batch ──────────────────────────────────────────────────────

/// A coalesced batch of inference requests.
#[derive(Debug)]
pub struct InferBatch {
    pub batch_id: u64,
    pub requests: Vec<InferRequest>,
    /// Padded input matrix: [batch_size, seq_len].
    pub padded_inputs: Vec<Vec<f64>>,
    /// Original lengths before padding.
    pub original_lengths: Vec<usize>,
    pub created_at: Instant,
}

impl InferBatch {
    /// Batch size.
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Sequence length after padding.
    pub fn seq_len(&self) -> usize {
        self.padded_inputs.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Maximum wait time among all requests in this batch.
    pub fn max_wait(&self) -> Duration {
        self.requests
            .iter()
            .map(|r| r.wait_time())
            .max()
            .unwrap_or(Duration::ZERO)
    }

    /// Average input length (before padding).
    pub fn avg_original_len(&self) -> f64 {
        if self.original_lengths.is_empty() {
            return 0.0;
        }
        self.original_lengths.iter().sum::<usize>() as f64 / self.original_lengths.len() as f64
    }

    /// Padding efficiency: ratio of real tokens to total padded tokens.
    pub fn padding_efficiency(&self) -> f64 {
        let total_padded: usize = self.padded_inputs.iter().map(|v| v.len()).sum();
        let total_real: usize = self.original_lengths.iter().sum();
        if total_padded == 0 {
            return 1.0;
        }
        total_real as f64 / total_padded as f64
    }
}

impl fmt::Display for InferBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InferBatch(id={}, size={}, seq_len={}, efficiency={:.2}%)",
            self.batch_id,
            self.size(),
            self.seq_len(),
            self.padding_efficiency() * 100.0
        )
    }
}

// ── Batch Result ───────────────────────────────────────────────

/// Result of a batched inference.
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch_id: u64,
    pub outputs: Vec<InferOutput>,
    pub latency: Duration,
}

/// Single request output.
#[derive(Debug, Clone)]
pub struct InferOutput {
    pub request_id: RequestId,
    pub output: Vec<f64>,
}

impl fmt::Display for BatchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchResult(batch={}, outputs={}, latency={:?})",
            self.batch_id,
            self.outputs.len(),
            self.latency
        )
    }
}

// ── Throughput Tracker ─────────────────────────────────────────

/// Tracks inference throughput over a sliding window.
#[derive(Debug)]
pub struct ThroughputTracker {
    window: Duration,
    samples: VecDeque<(Instant, usize)>,
}

impl ThroughputTracker {
    pub fn new(window: Duration) -> Self {
        Self { window, samples: VecDeque::new() }
    }

    /// Record that `count` requests were processed at this instant.
    pub fn record(&mut self, count: usize) {
        let now = Instant::now();
        self.samples.push_back((now, count));
        self.prune(now);
    }

    /// Current throughput in requests per second.
    pub fn requests_per_second(&self) -> f64 {
        let now = Instant::now();
        let cutoff = now - self.window;
        let total: usize = self.samples
            .iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, c)| c)
            .sum();
        let elapsed = self.window.as_secs_f64();
        if elapsed == 0.0 {
            return 0.0;
        }
        total as f64 / elapsed
    }

    fn prune(&mut self, now: Instant) {
        let cutoff = now - self.window;
        while let Some(&(t, _)) = self.samples.front() {
            if t < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

impl fmt::Display for ThroughputTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThroughputTracker({:.1} req/s)", self.requests_per_second())
    }
}

// ── Batcher Config ─────────────────────────────────────────────

/// Configuration for the dynamic batcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    pub max_batch_size: usize,
    pub max_wait: Duration,
    pub padding: PaddingStrategy,
    pub pad_value: f64,
    pub priority_boost_ms: u64,
}

impl BatcherConfig {
    pub fn new() -> Self {
        Self {
            max_batch_size: 32,
            max_wait: Duration::from_millis(50),
            padding: PaddingStrategy::PadToMax,
            pad_value: 0.0,
            priority_boost_ms: 10,
        }
    }

    pub fn with_max_batch_size(mut self, n: usize) -> Self {
        self.max_batch_size = n;
        self
    }

    pub fn with_max_wait(mut self, d: Duration) -> Self {
        self.max_wait = d;
        self
    }

    pub fn with_padding(mut self, p: PaddingStrategy) -> Self {
        self.padding = p;
        self
    }

    pub fn with_pad_value(mut self, v: f64) -> Self {
        self.pad_value = v;
        self
    }

    pub fn with_priority_boost_ms(mut self, ms: u64) -> Self {
        self.priority_boost_ms = ms;
        self
    }
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BatcherConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatcherConfig(max_batch={}, max_wait={:?}, padding={})",
            self.max_batch_size, self.max_wait, self.padding
        )
    }
}

// ── Dynamic Batcher ────────────────────────────────────────────

/// Collects requests and dispatches them as batches.
#[derive(Debug)]
pub struct DynamicBatcher {
    config: BatcherConfig,
    queue: VecDeque<InferRequest>,
    next_batch_id: u64,
    batches_dispatched: u64,
    total_requests: u64,
}

impl DynamicBatcher {
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            queue: VecDeque::new(),
            next_batch_id: 0,
            batches_dispatched: 0,
            total_requests: 0,
        }
    }

    /// Enqueue a request.
    pub fn enqueue(&mut self, req: InferRequest) {
        // Insert based on priority (higher priority toward front).
        let pos = self.queue.iter().position(|r| r.priority < req.priority);
        match pos {
            Some(idx) => self.queue.insert(idx, req),
            None => self.queue.push_back(req),
        }
        self.total_requests += 1;
    }

    /// Number of pending requests.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Check if a batch should be dispatched.
    pub fn should_dispatch(&self) -> bool {
        if self.queue.is_empty() {
            return false;
        }
        // Dispatch if batch is full.
        if self.queue.len() >= self.config.max_batch_size {
            return true;
        }
        // Dispatch if oldest request has waited too long.
        if let Some(oldest) = self.queue.front() {
            if oldest.wait_time() >= self.config.max_wait {
                return true;
            }
        }
        false
    }

    /// Force-dispatch whatever is in the queue (up to max_batch_size).
    pub fn dispatch(&mut self) -> Option<InferBatch> {
        if self.queue.is_empty() {
            return None;
        }

        let take = self.queue.len().min(self.config.max_batch_size);
        let requests: Vec<InferRequest> = self.queue.drain(..take).collect();

        let max_len = requests.iter().map(|r| r.input_len).max().unwrap_or(0);
        let original_lengths: Vec<usize> = requests.iter().map(|r| r.input_len).collect();

        let padded_inputs: Vec<Vec<f64>> = requests
            .iter()
            .map(|r| self.config.padding.apply(&r.input, self.config.pad_value, max_len))
            .collect();

        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;
        self.batches_dispatched += 1;

        Some(InferBatch {
            batch_id,
            requests,
            padded_inputs,
            original_lengths,
            created_at: Instant::now(),
        })
    }

    /// Try to dispatch if conditions are met.
    pub fn try_dispatch(&mut self) -> Option<InferBatch> {
        if self.should_dispatch() {
            self.dispatch()
        } else {
            None
        }
    }

    /// Total batches dispatched so far.
    pub fn batches_dispatched(&self) -> u64 {
        self.batches_dispatched
    }

    /// Total requests ever enqueued.
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Average batch size.
    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_dispatched == 0 {
            return 0.0;
        }
        self.total_requests as f64 / self.batches_dispatched as f64
    }
}

impl fmt::Display for DynamicBatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DynamicBatcher(pending={}, dispatched={}, avg_batch={:.1})",
            self.pending(),
            self.batches_dispatched,
            self.avg_batch_size()
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_request_basic() {
        let req = InferRequest::new(1, vec![1.0, 2.0, 3.0]);
        assert_eq!(req.id, 1);
        assert_eq!(req.input_len, 3);
        assert_eq!(req.priority, Priority::Normal);
    }

    #[test]
    fn test_infer_request_priority() {
        let req = InferRequest::new(1, vec![]).with_priority(Priority::High);
        assert_eq!(req.priority, Priority::High);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn test_padding_pad_to_max() {
        let input = vec![1.0, 2.0];
        let padded = PaddingStrategy::PadToMax.apply(&input, 0.0, 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[2], 0.0);
    }

    #[test]
    fn test_padding_pad_to_fixed() {
        let input = vec![1.0, 2.0];
        let padded = PaddingStrategy::PadToFixed(4).apply(&input, -1.0, 10);
        assert_eq!(padded.len(), 4);
        assert_eq!(padded[3], -1.0);
    }

    #[test]
    fn test_padding_truncate() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let truncated = PaddingStrategy::Truncate(3).apply(&input, 0.0, 5);
        assert_eq!(truncated.len(), 3);
    }

    #[test]
    fn test_padding_none() {
        let input = vec![1.0, 2.0];
        let out = PaddingStrategy::None.apply(&input, 0.0, 5);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_batcher_enqueue_dispatch() {
        let cfg = BatcherConfig::new().with_max_batch_size(4);
        let mut batcher = DynamicBatcher::new(cfg);

        for i in 0..4 {
            batcher.enqueue(InferRequest::new(i, vec![i as f64; 3]));
        }
        assert_eq!(batcher.pending(), 4);

        let batch = batcher.dispatch().unwrap();
        assert_eq!(batch.size(), 4);
        assert_eq!(batcher.pending(), 0);
    }

    #[test]
    fn test_batcher_partial_dispatch() {
        let cfg = BatcherConfig::new().with_max_batch_size(10);
        let mut batcher = DynamicBatcher::new(cfg);

        batcher.enqueue(InferRequest::new(1, vec![1.0]));
        batcher.enqueue(InferRequest::new(2, vec![2.0]));

        let batch = batcher.dispatch().unwrap();
        assert_eq!(batch.size(), 2);
    }

    #[test]
    fn test_batcher_empty_dispatch() {
        let cfg = BatcherConfig::new();
        let mut batcher = DynamicBatcher::new(cfg);
        assert!(batcher.dispatch().is_none());
    }

    #[test]
    fn test_batcher_priority_ordering() {
        let cfg = BatcherConfig::new().with_max_batch_size(3);
        let mut batcher = DynamicBatcher::new(cfg);

        batcher.enqueue(InferRequest::new(1, vec![1.0]).with_priority(Priority::Low));
        batcher.enqueue(InferRequest::new(2, vec![2.0]).with_priority(Priority::Critical));
        batcher.enqueue(InferRequest::new(3, vec![3.0]).with_priority(Priority::Normal));

        let batch = batcher.dispatch().unwrap();
        assert_eq!(batch.requests[0].id, 2); // Critical first
    }

    #[test]
    fn test_batch_padding_efficiency() {
        let batch = InferBatch {
            batch_id: 0,
            requests: Vec::new(),
            padded_inputs: vec![vec![0.0; 10], vec![0.0; 10]],
            original_lengths: vec![5, 10],
            created_at: Instant::now(),
        };
        // 15 real / 20 padded = 0.75
        assert!((batch.padding_efficiency() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_batch_avg_original_len() {
        let batch = InferBatch {
            batch_id: 0,
            requests: Vec::new(),
            padded_inputs: Vec::new(),
            original_lengths: vec![4, 6, 8],
            created_at: Instant::now(),
        };
        assert!((batch.avg_original_len() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_throughput_tracker() {
        let mut tracker = ThroughputTracker::new(Duration::from_secs(60));
        tracker.record(10);
        tracker.record(20);
        // Should be > 0 since within window.
        assert!(tracker.requests_per_second() > 0.0);
    }

    #[test]
    fn test_batcher_config_builder() {
        let cfg = BatcherConfig::new()
            .with_max_batch_size(64)
            .with_max_wait(Duration::from_millis(100))
            .with_padding(PaddingStrategy::PadToFixed(128))
            .with_pad_value(-1.0)
            .with_priority_boost_ms(20);
        assert_eq!(cfg.max_batch_size, 64);
        assert_eq!(cfg.pad_value, -1.0);
    }

    #[test]
    fn test_batcher_config_default() {
        let cfg = BatcherConfig::default();
        assert_eq!(cfg.max_batch_size, 32);
    }

    #[test]
    fn test_batcher_stats() {
        let cfg = BatcherConfig::new().with_max_batch_size(2);
        let mut batcher = DynamicBatcher::new(cfg);
        batcher.enqueue(InferRequest::new(1, vec![1.0]));
        batcher.enqueue(InferRequest::new(2, vec![2.0]));
        batcher.dispatch();
        batcher.enqueue(InferRequest::new(3, vec![3.0]));
        batcher.dispatch();
        assert_eq!(batcher.batches_dispatched(), 2);
        assert_eq!(batcher.total_requests(), 3);
        assert!((batcher.avg_batch_size() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_display_impls() {
        let req = InferRequest::new(1, vec![]);
        assert!(format!("{req}").contains("id=1"));

        assert!(format!("{}", Priority::High).contains("high"));
        assert!(format!("{}", PaddingStrategy::PadToMax).contains("pad-to-max"));

        let cfg = BatcherConfig::new();
        assert!(format!("{cfg}").contains("BatcherConfig"));

        let batcher = DynamicBatcher::new(BatcherConfig::new());
        assert!(format!("{batcher}").contains("DynamicBatcher"));
    }

    #[test]
    fn test_batch_result_display() {
        let br = BatchResult {
            batch_id: 5,
            outputs: vec![InferOutput { request_id: 1, output: vec![0.5] }],
            latency: Duration::from_micros(500),
        };
        assert!(format!("{br}").contains("batch=5"));
    }

    #[test]
    fn test_throughput_tracker_display() {
        let tracker = ThroughputTracker::new(Duration::from_secs(10));
        assert!(format!("{tracker}").contains("req/s"));
    }
}
