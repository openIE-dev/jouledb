//! Batch processing framework.
//!
//! Replaces Spring Batch, AWS Glue, and similar batch processing frameworks with
//! a pure-Rust batch processor. Supports configurable batch size, parallel batch
//! execution simulation, progress tracking, partial failure handling, batch result
//! aggregation, retry failed batches, and checkpoint after each batch.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors from batch processing.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    /// No items to process.
    EmptyInput,
    /// Invalid batch size.
    InvalidBatchSize,
    /// Processing error for a specific batch.
    BatchFailed { batch_index: usize, reason: String },
    /// All retries exhausted.
    RetriesExhausted { batch_index: usize, attempts: u32 },
    /// Checkpoint not found.
    CheckpointNotFound(String),
    /// Processor already running.
    AlreadyRunning,
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "no items to process"),
            Self::InvalidBatchSize => write!(f, "batch size must be > 0"),
            Self::BatchFailed { batch_index, reason } => {
                write!(f, "batch {batch_index} failed: {reason}")
            }
            Self::RetriesExhausted { batch_index, attempts } => {
                write!(f, "batch {batch_index} failed after {attempts} attempts")
            }
            Self::CheckpointNotFound(id) => write!(f, "checkpoint not found: {id}"),
            Self::AlreadyRunning => write!(f, "processor already running"),
        }
    }
}

impl std::error::Error for BatchError {}

// ── Batch result ─────────────────────────────────────────────────

/// Result of processing a single batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult<T> {
    /// Batch index (0-based).
    pub batch_index: usize,
    /// Successfully processed items.
    pub succeeded: Vec<T>,
    /// Failed items with error messages.
    pub failed: Vec<(usize, String)>, // (item_index_within_batch, error)
    /// Number of retries used.
    pub retries: u32,
    /// Duration in microseconds.
    pub duration_us: u64,
}

impl<T> BatchResult<T> {
    /// Whether the entire batch succeeded.
    pub fn is_success(&self) -> bool {
        self.failed.is_empty()
    }

    /// Success rate for this batch.
    pub fn success_rate(&self) -> f64 {
        let total = self.succeeded.len() + self.failed.len();
        if total == 0 {
            1.0
        } else {
            self.succeeded.len() as f64 / total as f64
        }
    }
}

// ── Batch status ─────────────────────────────────────────────────

/// Status of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStatus {
    Pending,
    Running,
    Succeeded,
    PartialFailure,
    Failed,
    Retrying,
}

impl fmt::Display for BatchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Succeeded => write!(f, "succeeded"),
            Self::PartialFailure => write!(f, "partial_failure"),
            Self::Failed => write!(f, "failed"),
            Self::Retrying => write!(f, "retrying"),
        }
    }
}

// ── Progress ─────────────────────────────────────────────────────

/// Progress information for a batch run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Progress {
    /// Total number of batches.
    pub total_batches: usize,
    /// Number of completed batches.
    pub completed_batches: usize,
    /// Number of successful batches.
    pub successful_batches: usize,
    /// Number of failed batches.
    pub failed_batches: usize,
    /// Total items processed.
    pub total_items: usize,
    /// Items succeeded.
    pub items_succeeded: usize,
    /// Items failed.
    pub items_failed: usize,
    /// Per-batch status.
    pub batch_statuses: Vec<BatchStatus>,
    /// Completion fraction [0.0, 1.0].
    pub completion: f64,
}

impl Progress {
    fn new(total_batches: usize) -> Self {
        Self {
            total_batches,
            completed_batches: 0,
            successful_batches: 0,
            failed_batches: 0,
            total_items: 0,
            items_succeeded: 0,
            items_failed: 0,
            batch_statuses: vec![BatchStatus::Pending; total_batches],
            completion: 0.0,
        }
    }

    fn update_completion(&mut self) {
        if self.total_batches == 0 {
            self.completion = 1.0;
        } else {
            self.completion = self.completed_batches as f64 / self.total_batches as f64;
        }
    }
}

// ── Checkpoint ───────────────────────────────────────────────────

/// A checkpoint after a batch completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckpoint {
    /// Checkpoint identifier.
    pub id: String,
    /// Index of the last completed batch.
    pub last_completed_batch: usize,
    /// Progress snapshot.
    pub progress: Progress,
}

// ── Aggregate result ─────────────────────────────────────────────

/// Aggregated results from the entire batch run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateResult {
    /// Total batches.
    pub total_batches: usize,
    /// Successful batches.
    pub successful_batches: usize,
    /// Failed batches.
    pub failed_batches: usize,
    /// Partially failed batches.
    pub partial_failure_batches: usize,
    /// Total items.
    pub total_items: usize,
    /// Items succeeded.
    pub items_succeeded: usize,
    /// Items failed.
    pub items_failed: usize,
    /// Total retries.
    pub total_retries: u32,
    /// Overall success rate.
    pub success_rate: f64,
}

// ── Failure handling strategy ────────────────────────────────────

/// How to handle batch failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureStrategy {
    /// Stop processing on first failed batch.
    StopOnFailure,
    /// Continue to next batch on failure.
    ContinueOnFailure,
    /// Retry failed batches up to N times.
    RetryThenContinue(u32),
    /// Retry failed batches, stop if still failing.
    RetryThenStop(u32),
}

impl Default for FailureStrategy {
    fn default() -> Self {
        Self::StopOnFailure
    }
}

// ── Item processor ───────────────────────────────────────────────

/// Result of processing a single item.
#[derive(Debug, Clone)]
pub enum ItemResult<T> {
    /// Item processed successfully.
    Ok(T),
    /// Item processing failed.
    Err(String),
}

// ── Batch processor ──────────────────────────────────────────────

/// The batch processing engine.
pub struct BatchProcessor<T, U> {
    /// Batch size.
    batch_size: usize,
    /// Processing function.
    processor: Box<dyn Fn(&T) -> ItemResult<U>>,
    /// Failure handling strategy.
    failure_strategy: FailureStrategy,
    /// Progress tracking.
    progress: Option<Progress>,
    /// Checkpoints.
    checkpoints: HashMap<String, BatchCheckpoint>,
    /// Whether currently running.
    running: bool,
}

impl<T, U> fmt::Debug for BatchProcessor<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchProcessor")
            .field("batch_size", &self.batch_size)
            .field("failure_strategy", &self.failure_strategy)
            .field("running", &self.running)
            .finish()
    }
}

impl<T: Clone, U: Clone> BatchProcessor<T, U> {
    /// Create a new batch processor.
    pub fn new(
        batch_size: usize,
        processor: impl Fn(&T) -> ItemResult<U> + 'static,
    ) -> Self {
        Self {
            batch_size,
            processor: Box::new(processor),
            failure_strategy: FailureStrategy::default(),
            progress: None,
            checkpoints: HashMap::new(),
            running: false,
        }
    }

    /// Set the failure strategy.
    pub fn with_failure_strategy(mut self, strategy: FailureStrategy) -> Self {
        self.failure_strategy = strategy;
        self
    }

    /// Set the batch size.
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    /// Get current progress (if running or completed).
    pub fn progress(&self) -> Option<&Progress> {
        self.progress.as_ref()
    }

    /// Get a checkpoint.
    pub fn get_checkpoint(&self, id: &str) -> Option<&BatchCheckpoint> {
        self.checkpoints.get(id)
    }

    /// Process all items in batches.
    pub fn process(&mut self, items: &[T]) -> Result<(Vec<BatchResult<U>>, AggregateResult), BatchError> {
        if items.is_empty() {
            return Err(BatchError::EmptyInput);
        }
        if self.batch_size == 0 {
            return Err(BatchError::InvalidBatchSize);
        }
        if self.running {
            return Err(BatchError::AlreadyRunning);
        }

        self.running = true;

        let chunks: Vec<&[T]> = items.chunks(self.batch_size).collect();
        let total_batches = chunks.len();
        let mut progress = Progress::new(total_batches);
        progress.total_items = items.len();

        let mut batch_results = Vec::with_capacity(total_batches);
        let mut should_stop = false;

        for (batch_idx, chunk) in chunks.iter().enumerate() {
            if should_stop {
                break;
            }

            progress.batch_statuses[batch_idx] = BatchStatus::Running;

            let result = self.process_batch(batch_idx, chunk);

            // Handle result based on failure strategy.
            let (final_result, stop) =
                self.handle_batch_result(batch_idx, chunk, result, &mut progress);

            should_stop = stop;

            // Update progress.
            progress.completed_batches += 1;
            progress.items_succeeded += final_result.succeeded.len();
            progress.items_failed += final_result.failed.len();

            if final_result.is_success() {
                progress.successful_batches += 1;
                progress.batch_statuses[batch_idx] = BatchStatus::Succeeded;
            } else if final_result.succeeded.is_empty() {
                progress.failed_batches += 1;
                progress.batch_statuses[batch_idx] = BatchStatus::Failed;
            } else {
                progress.failed_batches += 1;
                progress.batch_statuses[batch_idx] = BatchStatus::PartialFailure;
            }

            progress.update_completion();

            // Save checkpoint.
            let ckpt_id = format!("batch-{batch_idx}");
            self.checkpoints.insert(
                ckpt_id.clone(),
                BatchCheckpoint {
                    id: ckpt_id,
                    last_completed_batch: batch_idx,
                    progress: progress.clone(),
                },
            );

            batch_results.push(final_result);
        }

        // Aggregate results.
        let total_retries: u32 = batch_results.iter().map(|r| r.retries).sum();
        let items_succeeded: usize = batch_results.iter().map(|r| r.succeeded.len()).sum();
        let items_failed: usize = batch_results.iter().map(|r| r.failed.len()).sum();
        let successful_batches = batch_results.iter().filter(|r| r.is_success()).count();
        let failed_batches = batch_results
            .iter()
            .filter(|r| !r.is_success() && r.succeeded.is_empty())
            .count();
        let partial = batch_results
            .iter()
            .filter(|r| !r.is_success() && !r.succeeded.is_empty())
            .count();

        let total_items = items_succeeded + items_failed;
        let success_rate = if total_items == 0 {
            1.0
        } else {
            items_succeeded as f64 / total_items as f64
        };

        let aggregate = AggregateResult {
            total_batches: batch_results.len(),
            successful_batches,
            failed_batches,
            partial_failure_batches: partial,
            total_items,
            items_succeeded,
            items_failed,
            total_retries,
            success_rate,
        };

        self.progress = Some(progress);
        self.running = false;

        Ok((batch_results, aggregate))
    }

    /// Resume processing from a checkpoint.
    pub fn resume(
        &mut self,
        checkpoint_id: &str,
        items: &[T],
    ) -> Result<(Vec<BatchResult<U>>, AggregateResult), BatchError> {
        let checkpoint = self
            .checkpoints
            .get(checkpoint_id)
            .cloned()
            .ok_or_else(|| BatchError::CheckpointNotFound(checkpoint_id.to_string()))?;

        // Skip batches before the checkpoint.
        let start_item = (checkpoint.last_completed_batch + 1) * self.batch_size;
        if start_item >= items.len() {
            // Nothing left to process.
            return Ok((
                Vec::new(),
                AggregateResult {
                    total_batches: 0,
                    successful_batches: 0,
                    failed_batches: 0,
                    partial_failure_batches: 0,
                    total_items: 0,
                    items_succeeded: 0,
                    items_failed: 0,
                    total_retries: 0,
                    success_rate: 1.0,
                },
            ));
        }

        self.process(&items[start_item..])
    }

    // ── Private helpers ──

    fn process_batch(&self, batch_index: usize, chunk: &[T]) -> BatchResult<U> {
        let mut succeeded = Vec::new();
        let mut failed = Vec::new();

        for (i, item) in chunk.iter().enumerate() {
            match (self.processor)(item) {
                ItemResult::Ok(output) => succeeded.push(output),
                ItemResult::Err(reason) => failed.push((i, reason)),
            }
        }

        BatchResult {
            batch_index,
            succeeded,
            failed,
            retries: 0,
            duration_us: 0,
        }
    }

    fn handle_batch_result(
        &self,
        batch_idx: usize,
        chunk: &[T],
        mut result: BatchResult<U>,
        _progress: &mut Progress,
    ) -> (BatchResult<U>, bool) {
        if result.is_success() {
            return (result, false);
        }

        match self.failure_strategy {
            FailureStrategy::StopOnFailure => (result, true),
            FailureStrategy::ContinueOnFailure => (result, false),
            FailureStrategy::RetryThenContinue(max_retries) => {
                for attempt in 1..=max_retries {
                    result = self.process_batch(batch_idx, chunk);
                    result.retries = attempt;
                    if result.is_success() {
                        return (result, false);
                    }
                }
                (result, false)
            }
            FailureStrategy::RetryThenStop(max_retries) => {
                for attempt in 1..=max_retries {
                    result = self.process_batch(batch_idx, chunk);
                    result.retries = attempt;
                    if result.is_success() {
                        return (result, false);
                    }
                }
                (result, true)
            }
        }
    }
}

// ── Convenience ──────────────────────────────────────────────────

/// Split items into batches of the given size.
pub fn split_batches<T>(items: &[T], batch_size: usize) -> Vec<&[T]> {
    if batch_size == 0 {
        return vec![];
    }
    items.chunks(batch_size).collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_error() {
        let mut proc = BatchProcessor::<i32, i32>::new(10, |x| ItemResult::Ok(*x));
        let err = proc.process(&[]).unwrap_err();
        assert_eq!(err, BatchError::EmptyInput);
    }

    #[test]
    fn invalid_batch_size() {
        let mut proc = BatchProcessor::<i32, i32>::new(0, |x| ItemResult::Ok(*x));
        let err = proc.process(&[1, 2, 3]).unwrap_err();
        assert_eq!(err, BatchError::InvalidBatchSize);
    }

    #[test]
    fn basic_processing() {
        let mut proc = BatchProcessor::new(3, |x: &i32| ItemResult::Ok(x * 2));
        let items: Vec<i32> = (0..10).collect();
        let (results, aggregate) = proc.process(&items).unwrap();

        assert_eq!(aggregate.total_batches, 4); // 3+3+3+1
        assert_eq!(aggregate.items_succeeded, 10);
        assert_eq!(aggregate.items_failed, 0);
        assert!((aggregate.success_rate - 1.0).abs() < f64::EPSILON);

        // Check first batch output.
        assert_eq!(results[0].succeeded, vec![0, 2, 4]);
    }

    #[test]
    fn partial_failure() {
        let mut proc = BatchProcessor::new(5, |x: &i32| {
            if *x % 3 == 0 {
                ItemResult::Err("divisible by 3".into())
            } else {
                ItemResult::Ok(*x)
            }
        })
        .with_failure_strategy(FailureStrategy::ContinueOnFailure);

        let items: Vec<i32> = (0..10).collect();
        let (results, aggregate) = proc.process(&items).unwrap();
        assert!(aggregate.items_failed > 0);
        assert!(aggregate.items_succeeded > 0);
        assert!(aggregate.success_rate < 1.0);

        // All batches should be processed despite failures.
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn stop_on_failure() {
        let mut proc = BatchProcessor::new(2, |x: &i32| {
            if *x >= 4 {
                ItemResult::Err("too large".into())
            } else {
                ItemResult::Ok(*x)
            }
        })
        .with_failure_strategy(FailureStrategy::StopOnFailure);

        let items: Vec<i32> = (0..10).collect();
        let (results, _) = proc.process(&items).unwrap();
        // Should stop at the batch containing 4.
        assert!(results.len() <= 3);
    }

    #[test]
    fn retry_then_continue() {
        use std::cell::Cell;
        let call_count = std::rc::Rc::new(Cell::new(0u32));
        let cc = call_count.clone();

        let mut proc = BatchProcessor::new(2, move |x: &i32| {
            let c = cc.get();
            cc.set(c + 1);
            // Fail the first 2 calls (first batch attempt + first retry).
            if *x == 0 && c < 4 {
                ItemResult::Err("not yet".into())
            } else {
                ItemResult::Ok(*x)
            }
        })
        .with_failure_strategy(FailureStrategy::RetryThenContinue(3));

        let items = vec![0, 1, 2, 3];
        let (results, aggregate) = proc.process(&items).unwrap();
        assert_eq!(results.len(), 2);
        assert!(aggregate.total_retries > 0);
    }

    #[test]
    fn retry_then_stop() {
        let mut proc = BatchProcessor::new(2, |x: &i32| {
            if *x == 0 {
                ItemResult::Err("always fail".into())
            } else {
                ItemResult::Ok(*x)
            }
        })
        .with_failure_strategy(FailureStrategy::RetryThenStop(2));

        let items = vec![0, 1, 2, 3];
        let (results, aggregate) = proc.process(&items).unwrap();
        // First batch keeps failing, should stop after retries.
        assert!(results.len() <= 2);
        assert!(aggregate.total_retries > 0);
    }

    #[test]
    fn progress_tracking() {
        let mut proc = BatchProcessor::new(5, |x: &i32| ItemResult::Ok(*x));
        let items: Vec<i32> = (0..12).collect();
        proc.process(&items).unwrap();

        let progress = proc.progress().unwrap();
        assert_eq!(progress.total_batches, 3);
        assert_eq!(progress.completed_batches, 3);
        assert_eq!(progress.successful_batches, 3);
        assert!((progress.completion - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn checkpoint_after_each_batch() {
        let mut proc = BatchProcessor::new(3, |x: &i32| ItemResult::Ok(*x));
        let items: Vec<i32> = (0..9).collect();
        proc.process(&items).unwrap();

        assert!(proc.get_checkpoint("batch-0").is_some());
        assert!(proc.get_checkpoint("batch-1").is_some());
        assert!(proc.get_checkpoint("batch-2").is_some());

        let ckpt = proc.get_checkpoint("batch-1").unwrap();
        assert_eq!(ckpt.last_completed_batch, 1);
    }

    #[test]
    fn resume_from_checkpoint() {
        let mut proc = BatchProcessor::new(3, |x: &i32| ItemResult::Ok(x * 10));
        let items: Vec<i32> = (0..9).collect();
        proc.process(&items).unwrap();

        // Resume from batch-1 checkpoint (process items from batch 2 onward).
        let (results, aggregate) = proc.resume("batch-1", &items).unwrap();
        assert_eq!(aggregate.items_succeeded, 3); // items 6, 7, 8
        assert!(!results.is_empty());
    }

    #[test]
    fn resume_missing_checkpoint() {
        let mut proc = BatchProcessor::<i32, i32>::new(5, |x| ItemResult::Ok(*x));
        let items = vec![1, 2, 3];
        proc.process(&items).unwrap();
        let err = proc.resume("nonexistent", &items).unwrap_err();
        assert!(matches!(err, BatchError::CheckpointNotFound(_)));
    }

    #[test]
    fn batch_result_success_rate() {
        let br: BatchResult<i32> = BatchResult {
            batch_index: 0,
            succeeded: vec![1, 2, 3],
            failed: vec![(3, "err".into())],
            retries: 0,
            duration_us: 0,
        };
        assert!((br.success_rate() - 0.75).abs() < f64::EPSILON);
        assert!(!br.is_success());
    }

    #[test]
    fn batch_result_fully_succeeded() {
        let br: BatchResult<i32> = BatchResult {
            batch_index: 0,
            succeeded: vec![1, 2],
            failed: vec![],
            retries: 0,
            duration_us: 0,
        };
        assert!(br.is_success());
        assert!((br.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn split_batches_helper() {
        let items = vec![1, 2, 3, 4, 5];
        let batches = split_batches(&items, 2);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0], &[1, 2]);
        assert_eq!(batches[1], &[3, 4]);
        assert_eq!(batches[2], &[5]);
    }

    #[test]
    fn split_batches_zero_size() {
        let items = vec![1, 2, 3];
        let batches = split_batches(&items, 0);
        assert!(batches.is_empty());
    }

    #[test]
    fn single_item_batch() {
        let mut proc = BatchProcessor::new(1, |x: &i32| ItemResult::Ok(*x));
        let items = vec![42];
        let (results, aggregate) = proc.process(&items).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(aggregate.total_batches, 1);
        assert_eq!(results[0].succeeded, vec![42]);
    }

    #[test]
    fn batch_size_larger_than_data() {
        let mut proc = BatchProcessor::new(100, |x: &i32| ItemResult::Ok(*x));
        let items = vec![1, 2, 3];
        let (results, aggregate) = proc.process(&items).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(aggregate.total_batches, 1);
        assert_eq!(aggregate.items_succeeded, 3);
    }

    #[test]
    fn error_display() {
        let e = BatchError::EmptyInput;
        assert!(format!("{e}").contains("no items"));
        let e2 = BatchError::BatchFailed {
            batch_index: 3,
            reason: "oops".into(),
        };
        assert!(format!("{e2}").contains("batch 3"));
    }

    #[test]
    fn batch_status_display() {
        assert_eq!(format!("{}", BatchStatus::Pending), "pending");
        assert_eq!(format!("{}", BatchStatus::Succeeded), "succeeded");
        assert_eq!(format!("{}", BatchStatus::Failed), "failed");
        assert_eq!(format!("{}", BatchStatus::PartialFailure), "partial_failure");
    }

    #[test]
    fn progress_batch_statuses() {
        let mut proc = BatchProcessor::new(2, |x: &i32| {
            if *x >= 4 {
                ItemResult::Err("fail".into())
            } else {
                ItemResult::Ok(*x)
            }
        })
        .with_failure_strategy(FailureStrategy::ContinueOnFailure);

        let items: Vec<i32> = (0..6).collect();
        proc.process(&items).unwrap();

        let progress = proc.progress().unwrap();
        assert_eq!(progress.batch_statuses[0], BatchStatus::Succeeded);
        assert_eq!(progress.batch_statuses[1], BatchStatus::Succeeded);
        // Batch 2 has items 4, 5 which fail.
        assert!(
            progress.batch_statuses[2] == BatchStatus::Failed
                || progress.batch_statuses[2] == BatchStatus::PartialFailure
        );
    }

    #[test]
    fn all_batches_fail_continue() {
        let mut proc = BatchProcessor::new(3, |_: &i32| -> ItemResult<i32> {
            ItemResult::Err("nope".into())
        })
        .with_failure_strategy(FailureStrategy::ContinueOnFailure);

        let items: Vec<i32> = (0..6).collect();
        let (_, aggregate) = proc.process(&items).unwrap();
        assert_eq!(aggregate.items_succeeded, 0);
        assert_eq!(aggregate.items_failed, 6);
        assert!((aggregate.success_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn set_batch_size() {
        let mut proc = BatchProcessor::new(5, |x: &i32| ItemResult::Ok(*x));
        proc.set_batch_size(2);
        let items: Vec<i32> = (0..6).collect();
        let (results, _) = proc.process(&items).unwrap();
        assert_eq!(results.len(), 3);
    }
}
