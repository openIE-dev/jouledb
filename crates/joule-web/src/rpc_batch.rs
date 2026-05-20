//! Batched RPC calls — grouping, deduplication, partial-failure handling.
//!
//! Provides [`BatchRequest`] for collecting multiple RPC calls into a single batch,
//! [`BatchResponse`] with individual results, automatic deduplication of identical
//! calls, configurable batch size limits, priority ordering, and execution
//! statistics. Supports both atomic (all-or-nothing) and independent (partial
//! failure) execution modes.

use std::collections::HashMap;
use std::fmt;

// ── Call Status ────────────────────────────────────────────────

/// Status of an individual call within a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallStatus {
    Success,
    Error(String),
    Skipped,
    Deduplicated(usize),
}

impl fmt::Display for CallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "OK"),
            Self::Error(msg) => write!(f, "ERR: {msg}"),
            Self::Skipped => write!(f, "SKIPPED"),
            Self::Deduplicated(idx) => write!(f, "DEDUP({idx})"),
        }
    }
}

// ── Execution Mode ─────────────────────────────────────────────

/// How to execute the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// All calls succeed or the entire batch fails.
    Atomic,
    /// Each call is independent; partial failure is allowed.
    Independent,
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Atomic => write!(f, "Atomic"),
            Self::Independent => write!(f, "Independent"),
        }
    }
}

// ── Priority ───────────────────────────────────────────────────

/// Priority level for ordering calls within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

// ── Batch Call ─────────────────────────────────────────────────

/// A single RPC call within a batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchCall {
    pub method: String,
    pub params: Vec<u8>,
    pub priority: Priority,
    pub idempotent: bool,
}

impl BatchCall {
    pub fn new(method: impl Into<String>, params: Vec<u8>) -> Self {
        Self {
            method: method.into(),
            params,
            priority: Priority::Normal,
            idempotent: false,
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority; self
    }

    pub fn with_idempotent(mut self, idempotent: bool) -> Self {
        self.idempotent = idempotent; self
    }

    /// Compute a deduplication key from method + params.
    fn dedup_key(&self) -> (String, Vec<u8>) {
        (self.method.clone(), self.params.clone())
    }
}

impl fmt::Display for BatchCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}B, {})", self.method, self.params.len(), self.priority)
    }
}

// ── Batch Call Result ──────────────────────────────────────────

/// Result of a single call within the batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchCallResult {
    pub index: usize,
    pub method: String,
    pub status: CallStatus,
    pub response: Vec<u8>,
    pub duration_us: u64,
}

impl fmt::Display for BatchCallResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} -> {} ({}us)", self.index, self.method, self.status, self.duration_us)
    }
}

// ── Batch Error ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchError {
    Empty,
    TooLarge { size: usize, limit: usize },
    AtomicFailure { failed_index: usize, error: String },
    AllFailed,
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "batch is empty"),
            Self::TooLarge { size, limit } =>
                write!(f, "batch too large: {size} > {limit}"),
            Self::AtomicFailure { failed_index, error } =>
                write!(f, "atomic batch failed at index {failed_index}: {error}"),
            Self::AllFailed => write!(f, "all calls in batch failed"),
        }
    }
}

// ── Batch Request ──────────────────────────────────────────────

/// A collection of RPC calls to be executed as a batch.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    calls: Vec<BatchCall>,
    mode: ExecutionMode,
    max_size: usize,
    dedup: bool,
}

impl BatchRequest {
    pub fn new() -> Self {
        Self {
            calls: Vec::new(),
            mode: ExecutionMode::Independent,
            max_size: 1000,
            dedup: false,
        }
    }

    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode; self
    }

    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size; self
    }

    pub fn with_dedup(mut self, enable: bool) -> Self {
        self.dedup = enable; self
    }

    /// Add a call to the batch.
    pub fn add(&mut self, call: BatchCall) -> Result<usize, BatchError> {
        if self.calls.len() >= self.max_size {
            return Err(BatchError::TooLarge { size: self.calls.len() + 1, limit: self.max_size });
        }
        let idx = self.calls.len();
        self.calls.push(call);
        Ok(idx)
    }

    /// Number of calls in the batch.
    pub fn len(&self) -> usize { self.calls.len() }
    pub fn is_empty(&self) -> bool { self.calls.is_empty() }
    pub fn mode(&self) -> ExecutionMode { self.mode }
    pub fn calls(&self) -> &[BatchCall] { &self.calls }

    /// Sort calls by priority (highest first).
    pub fn sort_by_priority(&mut self) {
        self.calls.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Deduplicate identical calls. Returns the mapping from original to canonical index.
    pub fn deduplicate(&mut self) -> Vec<Option<usize>> {
        if !self.dedup {
            return (0..self.calls.len()).map(|_| None).collect();
        }
        let mut seen: HashMap<(String, Vec<u8>), usize> = HashMap::new();
        let mut dedup_map = Vec::new();
        for (i, call) in self.calls.iter().enumerate() {
            let key = call.dedup_key();
            if let Some(&canonical) = seen.get(&key) {
                dedup_map.push(Some(canonical));
            } else {
                seen.insert(key, i);
                dedup_map.push(None);
            }
        }
        dedup_map
    }
}

impl Default for BatchRequest {
    fn default() -> Self { Self::new() }
}

// ── Batch Response ─────────────────────────────────────────────

/// Aggregated results from executing a batch.
#[derive(Debug, Clone)]
pub struct BatchResponse {
    results: Vec<BatchCallResult>,
    mode: ExecutionMode,
    total_duration_us: u64,
}

impl BatchResponse {
    pub fn new(mode: ExecutionMode) -> Self {
        Self { results: Vec::new(), mode, total_duration_us: 0 }
    }

    pub fn add_result(&mut self, result: BatchCallResult) {
        self.total_duration_us += result.duration_us;
        self.results.push(result);
    }

    pub fn results(&self) -> &[BatchCallResult] { &self.results }
    pub fn mode(&self) -> ExecutionMode { self.mode }
    pub fn total_duration_us(&self) -> u64 { self.total_duration_us }

    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.status == CallStatus::Success).count()
    }

    pub fn error_count(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.status, CallStatus::Error(_))).count()
    }

    pub fn dedup_count(&self) -> usize {
        self.results.iter().filter(|r| matches!(r.status, CallStatus::Deduplicated(_))).count()
    }

    /// Whether all calls succeeded.
    pub fn all_succeeded(&self) -> bool { self.error_count() == 0 }

    /// Get a specific result by index.
    pub fn get(&self, idx: usize) -> Option<&BatchCallResult> { self.results.get(idx) }
}

// ── Batch Executor ─────────────────────────────────────────────

/// Executes a batch using a provided handler function.
pub type BatchHandler = fn(&str, &[u8]) -> Result<Vec<u8>, String>;

/// Execute a batch request with the given handler.
pub fn execute_batch(
    request: &mut BatchRequest,
    handler: BatchHandler,
) -> Result<BatchResponse, BatchError> {
    if request.is_empty() {
        return Err(BatchError::Empty);
    }

    request.sort_by_priority();
    let dedup_map = request.deduplicate();
    let mode = request.mode();
    let mut response = BatchResponse::new(mode);

    // First pass: execute non-deduplicated calls
    let mut results_by_index: Vec<Option<BatchCallResult>> = vec![None; request.len()];

    for (i, call) in request.calls().iter().enumerate() {
        if let Some(canonical) = dedup_map[i] {
            // Will be filled in second pass
            results_by_index[i] = Some(BatchCallResult {
                index: i,
                method: call.method.clone(),
                status: CallStatus::Deduplicated(canonical),
                response: Vec::new(),
                duration_us: 0,
            });
            continue;
        }

        let start = i as u64 * 10; // simulated timing
        match handler(&call.method, &call.params) {
            Ok(resp) => {
                results_by_index[i] = Some(BatchCallResult {
                    index: i,
                    method: call.method.clone(),
                    status: CallStatus::Success,
                    response: resp,
                    duration_us: start + 10,
                });
            }
            Err(err) => {
                if mode == ExecutionMode::Atomic {
                    return Err(BatchError::AtomicFailure {
                        failed_index: i,
                        error: err,
                    });
                }
                results_by_index[i] = Some(BatchCallResult {
                    index: i,
                    method: call.method.clone(),
                    status: CallStatus::Error(err),
                    response: Vec::new(),
                    duration_us: start + 10,
                });
            }
        }
    }

    // Copy dedup results from canonical results
    for i in 0..results_by_index.len() {
        if let Some(canonical) = dedup_map[i] {
            if let Some(canonical_result) = &results_by_index[canonical] {
                results_by_index[i] = Some(BatchCallResult {
                    index: i,
                    method: results_by_index[i].as_ref().unwrap().method.clone(),
                    status: CallStatus::Deduplicated(canonical),
                    response: canonical_result.response.clone(),
                    duration_us: 0,
                });
            }
        }
    }

    for result in results_by_index.into_iter().flatten() {
        response.add_result(result);
    }

    Ok(response)
}

// ── Batch Statistics ───────────────────────────────────────────

/// Aggregated statistics across multiple batch executions.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub batches_executed: u64,
    pub total_calls: u64,
    pub total_successes: u64,
    pub total_errors: u64,
    pub total_deduped: u64,
    pub total_duration_us: u64,
    pub largest_batch: usize,
}

impl BatchStats {
    pub fn new() -> Self { Self::default() }

    pub fn record(&mut self, response: &BatchResponse) {
        self.batches_executed += 1;
        self.total_calls += response.results().len() as u64;
        self.total_successes += response.success_count() as u64;
        self.total_errors += response.error_count() as u64;
        self.total_deduped += response.dedup_count() as u64;
        self.total_duration_us += response.total_duration_us();
        if response.results().len() > self.largest_batch {
            self.largest_batch = response.results().len();
        }
    }

    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_executed == 0 { return 0.0; }
        self.total_calls as f64 / self.batches_executed as f64
    }

    pub fn avg_duration_us(&self) -> f64 {
        if self.batches_executed == 0 { return 0.0; }
        self.total_duration_us as f64 / self.batches_executed as f64
    }
}

// ── Batch Collector ────────────────────────────────────────────

/// Auto-batching collector that groups calls within a time window.
#[derive(Debug)]
pub struct BatchCollector {
    pending: Vec<BatchCall>,
    window_ms: u64,
    max_size: usize,
    window_start_ms: Option<u64>,
}

impl BatchCollector {
    pub fn new(window_ms: u64, max_size: usize) -> Self {
        Self { pending: Vec::new(), window_ms, max_size, window_start_ms: None }
    }

    /// Submit a call. Returns a ready batch if the window has elapsed or size limit is reached.
    pub fn submit(&mut self, call: BatchCall, now_ms: u64) -> Option<Vec<BatchCall>> {
        if self.window_start_ms.is_none() {
            self.window_start_ms = Some(now_ms);
        }
        self.pending.push(call);
        if self.pending.len() >= self.max_size {
            return Some(self.flush());
        }
        if let Some(start) = self.window_start_ms {
            if now_ms.saturating_sub(start) >= self.window_ms {
                return Some(self.flush());
            }
        }
        None
    }

    /// Flush all pending calls regardless of window.
    pub fn flush(&mut self) -> Vec<BatchCall> {
        self.window_start_ms = None;
        std::mem::take(&mut self.pending)
    }

    pub fn pending_count(&self) -> usize { self.pending.len() }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_handler(_method: &str, params: &[u8]) -> Result<Vec<u8>, String> {
        Ok(params.to_vec())
    }

    fn fail_handler(_method: &str, _params: &[u8]) -> Result<Vec<u8>, String> {
        Err("handler error".into())
    }

    fn selective_handler(method: &str, params: &[u8]) -> Result<Vec<u8>, String> {
        if method == "fail" { Err("fail".into()) } else { Ok(params.to_vec()) }
    }

    #[test]
    fn batch_add_and_len() {
        let mut batch = BatchRequest::new();
        batch.add(BatchCall::new("foo", vec![1])).unwrap();
        batch.add(BatchCall::new("bar", vec![2])).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn batch_size_limit() {
        let mut batch = BatchRequest::new().with_max_size(1);
        batch.add(BatchCall::new("a", vec![])).unwrap();
        assert!(matches!(
            batch.add(BatchCall::new("b", vec![])),
            Err(BatchError::TooLarge { .. })
        ));
    }

    #[test]
    fn batch_empty_execution_fails() {
        let mut batch = BatchRequest::new();
        assert!(matches!(execute_batch(&mut batch, ok_handler), Err(BatchError::Empty)));
    }

    #[test]
    fn batch_all_succeed() {
        let mut batch = BatchRequest::new();
        batch.add(BatchCall::new("a", vec![1])).unwrap();
        batch.add(BatchCall::new("b", vec![2])).unwrap();
        let response = execute_batch(&mut batch, ok_handler).unwrap();
        assert_eq!(response.success_count(), 2);
        assert_eq!(response.error_count(), 0);
        assert!(response.all_succeeded());
    }

    #[test]
    fn batch_independent_partial_failure() {
        let mut batch = BatchRequest::new().with_mode(ExecutionMode::Independent);
        batch.add(BatchCall::new("ok", vec![])).unwrap();
        batch.add(BatchCall::new("fail", vec![])).unwrap();
        let response = execute_batch(&mut batch, selective_handler).unwrap();
        assert_eq!(response.success_count(), 1);
        assert_eq!(response.error_count(), 1);
    }

    #[test]
    fn batch_atomic_rolls_back_on_failure() {
        let mut batch = BatchRequest::new().with_mode(ExecutionMode::Atomic);
        batch.add(BatchCall::new("fail", vec![])).unwrap();
        assert!(matches!(
            execute_batch(&mut batch, fail_handler),
            Err(BatchError::AtomicFailure { .. })
        ));
    }

    #[test]
    fn priority_ordering() {
        let mut batch = BatchRequest::new();
        batch.add(BatchCall::new("low", vec![]).with_priority(Priority::Low)).unwrap();
        batch.add(BatchCall::new("high", vec![]).with_priority(Priority::High)).unwrap();
        batch.add(BatchCall::new("normal", vec![]).with_priority(Priority::Normal)).unwrap();
        batch.sort_by_priority();
        assert_eq!(batch.calls()[0].method, "high");
        assert_eq!(batch.calls()[1].method, "normal");
        assert_eq!(batch.calls()[2].method, "low");
    }

    #[test]
    fn deduplication_identifies_dupes() {
        let mut batch = BatchRequest::new().with_dedup(true);
        batch.add(BatchCall::new("foo", vec![1, 2])).unwrap();
        batch.add(BatchCall::new("bar", vec![3])).unwrap();
        batch.add(BatchCall::new("foo", vec![1, 2])).unwrap(); // duplicate
        let map = batch.deduplicate();
        assert_eq!(map[0], None);       // canonical
        assert_eq!(map[1], None);       // unique
        assert_eq!(map[2], Some(0));    // duplicate of 0
    }

    #[test]
    fn dedup_disabled_no_dedup() {
        let mut batch = BatchRequest::new().with_dedup(false);
        batch.add(BatchCall::new("foo", vec![1])).unwrap();
        batch.add(BatchCall::new("foo", vec![1])).unwrap();
        let map = batch.deduplicate();
        assert!(map.iter().all(|m| m.is_none()));
    }

    #[test]
    fn batch_response_access() {
        let mut resp = BatchResponse::new(ExecutionMode::Independent);
        resp.add_result(BatchCallResult {
            index: 0, method: "a".into(), status: CallStatus::Success,
            response: vec![1], duration_us: 100,
        });
        assert_eq!(resp.get(0).unwrap().method, "a");
        assert_eq!(resp.total_duration_us(), 100);
    }

    #[test]
    fn batch_stats_record() {
        let mut stats = BatchStats::new();
        let mut resp = BatchResponse::new(ExecutionMode::Independent);
        resp.add_result(BatchCallResult {
            index: 0, method: "a".into(), status: CallStatus::Success,
            response: vec![], duration_us: 50,
        });
        resp.add_result(BatchCallResult {
            index: 1, method: "b".into(), status: CallStatus::Error("x".into()),
            response: vec![], duration_us: 30,
        });
        stats.record(&resp);
        assert_eq!(stats.batches_executed, 1);
        assert_eq!(stats.total_calls, 2);
        assert_eq!(stats.total_successes, 1);
        assert_eq!(stats.total_errors, 1);
        assert_eq!(stats.largest_batch, 2);
        assert!((stats.avg_batch_size() - 2.0).abs() < 0.01);
    }

    #[test]
    fn batch_collector_size_flush() {
        let mut collector = BatchCollector::new(1000, 2);
        let r1 = collector.submit(BatchCall::new("a", vec![]), 0);
        assert!(r1.is_none());
        let r2 = collector.submit(BatchCall::new("b", vec![]), 5);
        assert!(r2.is_some());
        assert_eq!(r2.unwrap().len(), 2);
    }

    #[test]
    fn batch_collector_time_flush() {
        let mut collector = BatchCollector::new(100, 1000);
        collector.submit(BatchCall::new("a", vec![]), 0);
        let r = collector.submit(BatchCall::new("b", vec![]), 200);
        assert!(r.is_some());
    }

    #[test]
    fn batch_collector_manual_flush() {
        let mut collector = BatchCollector::new(1000, 1000);
        collector.submit(BatchCall::new("a", vec![]), 0);
        assert_eq!(collector.pending_count(), 1);
        let flushed = collector.flush();
        assert_eq!(flushed.len(), 1);
        assert_eq!(collector.pending_count(), 0);
    }

    #[test]
    fn call_display() {
        let call = BatchCall::new("test", vec![0; 10]).with_priority(Priority::High);
        let s = format!("{call}");
        assert!(s.contains("test"));
        assert!(s.contains("10B"));
        assert!(s.contains("High"));
    }

    #[test]
    fn status_display() {
        assert_eq!(format!("{}", CallStatus::Success), "OK");
        assert!(format!("{}", CallStatus::Error("bad".into())).contains("bad"));
        assert!(format!("{}", CallStatus::Deduplicated(2)).contains("2"));
    }

    #[test]
    fn batch_call_builder() {
        let call = BatchCall::new("m", vec![])
            .with_priority(Priority::Critical)
            .with_idempotent(true);
        assert_eq!(call.priority, Priority::Critical);
        assert!(call.idempotent);
    }
}
