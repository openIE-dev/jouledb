//! ETL pipeline framework.
//!
//! Replaces `Apache NiFi`, `Luigi`, `Bonobo`, and similar ETL frameworks with a
//! pure-Rust pipeline that models Extract/Transform/Load stages, stage composition,
//! error handling (skip/retry/dead-letter), pipeline metrics, checkpoint/resume,
//! and dry-run mode.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors returned by ETL pipeline operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EtlError {
    /// Extraction failed for a record.
    ExtractFailed { record_index: usize, reason: String },
    /// Transformation failed for a record.
    TransformFailed { record_index: usize, reason: String },
    /// Loading failed for a record.
    LoadFailed { record_index: usize, reason: String },
    /// Pipeline configuration error.
    ConfigError(String),
    /// Pipeline already running.
    AlreadyRunning,
    /// No stages configured.
    NoStages,
    /// Checkpoint not found.
    CheckpointNotFound(String),
}

impl fmt::Display for EtlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExtractFailed { record_index, reason } => {
                write!(f, "extract failed at record {record_index}: {reason}")
            }
            Self::TransformFailed { record_index, reason } => {
                write!(f, "transform failed at record {record_index}: {reason}")
            }
            Self::LoadFailed { record_index, reason } => {
                write!(f, "load failed at record {record_index}: {reason}")
            }
            Self::ConfigError(msg) => write!(f, "config error: {msg}"),
            Self::AlreadyRunning => write!(f, "pipeline already running"),
            Self::NoStages => write!(f, "no stages configured"),
            Self::CheckpointNotFound(id) => write!(f, "checkpoint not found: {id}"),
        }
    }
}

impl std::error::Error for EtlError {}

// ── Error handling strategies ────────────────────────────────────

/// How to handle errors during processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorStrategy {
    /// Stop on first error.
    Fail,
    /// Skip the failed record and continue.
    Skip,
    /// Retry the failed record up to N times.
    Retry(u32),
    /// Send to dead-letter queue and continue.
    DeadLetter,
}

impl Default for ErrorStrategy {
    fn default() -> Self {
        Self::Fail
    }
}

// ── Record ───────────────────────────────────────────────────────

/// A single data record flowing through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Unique identifier for this record.
    pub id: String,
    /// Key-value fields.
    pub fields: HashMap<String, serde_json::Value>,
    /// Metadata (source, timestamps, etc.).
    pub metadata: HashMap<String, String>,
}

impl Record {
    /// Create a new record with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set a field value.
    pub fn set_field(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.fields.insert(key.into(), value);
    }

    /// Get a field value.
    pub fn get_field(&self, key: &str) -> Option<&serde_json::Value> {
        self.fields.get(key)
    }

    /// Set metadata.
    pub fn set_meta(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

// ── Dead letter entry ────────────────────────────────────────────

/// A record that failed processing, stored in the dead-letter queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    /// The original record.
    pub record: Record,
    /// Which stage failed.
    pub stage_name: String,
    /// Error description.
    pub error: String,
    /// Number of retry attempts before dead-lettering.
    pub attempts: u32,
}

// ── Stage result ─────────────────────────────────────────────────

/// Result of processing a single record through a stage.
#[derive(Debug, Clone)]
pub enum StageResult {
    /// Record processed successfully (possibly transformed).
    Success(Record),
    /// Record should be filtered out (not passed downstream).
    Filtered,
    /// Record split into multiple output records.
    Split(Vec<Record>),
    /// Processing failed with an error.
    Failed(String),
}

// ── Stage definition ─────────────────────────────────────────────

/// The kind of processing a stage performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageKind {
    Extract,
    Transform,
    Load,
    Filter,
    Validate,
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Extract => write!(f, "extract"),
            Self::Transform => write!(f, "transform"),
            Self::Load => write!(f, "load"),
            Self::Filter => write!(f, "filter"),
            Self::Validate => write!(f, "validate"),
        }
    }
}

/// A processing stage in the pipeline. Uses a function pointer for the
/// processing logic.
pub struct Stage {
    /// Stage name.
    pub name: String,
    /// Kind of stage.
    pub kind: StageKind,
    /// Processing function. Takes a record, returns a stage result.
    pub processor: Box<dyn Fn(&Record) -> StageResult>,
    /// Error handling strategy for this stage.
    pub error_strategy: ErrorStrategy,
}

impl fmt::Debug for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stage")
            .field("name", &self.name)
            .field("kind", &self.kind)
            .field("error_strategy", &self.error_strategy)
            .finish()
    }
}

impl Stage {
    /// Create a new stage.
    pub fn new(
        name: impl Into<String>,
        kind: StageKind,
        processor: impl Fn(&Record) -> StageResult + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            kind,
            processor: Box::new(processor),
            error_strategy: ErrorStrategy::default(),
        }
    }

    /// Set error strategy.
    pub fn with_error_strategy(mut self, strategy: ErrorStrategy) -> Self {
        self.error_strategy = strategy;
        self
    }

    /// Process a single record.
    pub fn process(&self, record: &Record) -> StageResult {
        (self.processor)(record)
    }
}

// ── Stage metrics ────────────────────────────────────────────────

/// Metrics collected for a single stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageMetrics {
    /// Stage name.
    pub name: String,
    /// Number of records input.
    pub records_in: u64,
    /// Number of records output.
    pub records_out: u64,
    /// Number of records filtered out.
    pub records_filtered: u64,
    /// Number of errors.
    pub errors: u64,
    /// Number of retries.
    pub retries: u64,
    /// Number of dead-lettered records.
    pub dead_lettered: u64,
    /// Number of records skipped.
    pub skipped: u64,
    /// Duration in microseconds.
    pub duration_us: u64,
}

// ── Pipeline metrics ─────────────────────────────────────────────

/// Overall pipeline metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Total records processed.
    pub total_records: u64,
    /// Total records output at the end of the pipeline.
    pub total_output: u64,
    /// Total errors across all stages.
    pub total_errors: u64,
    /// Total duration in microseconds.
    pub total_duration_us: u64,
    /// Per-stage metrics, keyed by stage name.
    pub stage_metrics: Vec<StageMetrics>,
    /// Whether this was a dry run.
    pub dry_run: bool,
}

impl PipelineMetrics {
    /// Error rate as a fraction [0.0, 1.0].
    pub fn error_rate(&self) -> f64 {
        if self.total_records == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_records as f64
        }
    }

    /// Success rate as a fraction [0.0, 1.0].
    pub fn success_rate(&self) -> f64 {
        1.0 - self.error_rate()
    }
}

// ── Checkpoint ───────────────────────────────────────────────────

/// A checkpoint for resume capability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint identifier.
    pub id: String,
    /// Stage index where processing stopped.
    pub stage_index: usize,
    /// Record index within that stage's input.
    pub record_index: usize,
    /// Number of records successfully processed so far.
    pub records_processed: u64,
    /// Snapshot of metrics at checkpoint time.
    pub metrics_snapshot: PipelineMetrics,
}

// ── Pipeline state ───────────────────────────────────────────────

/// Pipeline execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    Idle,
    Running,
    Paused,
    Completed,
    Failed,
}

// ── Pipeline ─────────────────────────────────────────────────────

/// The ETL pipeline that orchestrates stages.
pub struct Pipeline {
    /// Pipeline name.
    name: String,
    /// Ordered list of stages.
    stages: Vec<Stage>,
    /// Current state.
    state: PipelineState,
    /// Dead-letter queue.
    dead_letter_queue: Vec<DeadLetterEntry>,
    /// Checkpoints.
    checkpoints: HashMap<String, Checkpoint>,
    /// Whether to run in dry-run mode (no side effects in load stages).
    dry_run: bool,
    /// Global error strategy (used when stage doesn't specify one).
    default_error_strategy: ErrorStrategy,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("stages", &self.stages.len())
            .field("state", &self.state)
            .field("dry_run", &self.dry_run)
            .finish()
    }
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            state: PipelineState::Idle,
            dead_letter_queue: Vec::new(),
            checkpoints: HashMap::new(),
            dry_run: false,
            default_error_strategy: ErrorStrategy::Fail,
        }
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(&mut self, stage: Stage) {
        self.stages.push(stage);
    }

    /// Set dry-run mode.
    pub fn set_dry_run(&mut self, dry_run: bool) {
        self.dry_run = dry_run;
    }

    /// Set the default error strategy.
    pub fn set_default_error_strategy(&mut self, strategy: ErrorStrategy) {
        self.default_error_strategy = strategy;
    }

    /// Get the current state.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Get the dead-letter queue.
    pub fn dead_letter_queue(&self) -> &[DeadLetterEntry] {
        &self.dead_letter_queue
    }

    /// Get a checkpoint by id.
    pub fn get_checkpoint(&self, id: &str) -> Option<&Checkpoint> {
        self.checkpoints.get(id)
    }

    /// Run the pipeline on the given input records.
    pub fn run(&mut self, input: Vec<Record>) -> Result<(Vec<Record>, PipelineMetrics), EtlError> {
        if self.stages.is_empty() {
            return Err(EtlError::NoStages);
        }
        if self.state == PipelineState::Running {
            return Err(EtlError::AlreadyRunning);
        }

        self.state = PipelineState::Running;
        self.dead_letter_queue.clear();

        let total_input = input.len() as u64;
        let mut current_records = input;
        let mut all_stage_metrics = Vec::new();
        let mut total_errors = 0u64;
        let num_stages = self.stages.len();

        for stage_idx in 0..num_stages {
            // Extract values from the stage before any mutable borrow.
            let stage_name = self.stages[stage_idx].name.clone();
            let stage_kind = self.stages[stage_idx].kind;
            let stage_error_strategy = self.stages[stage_idx].error_strategy;

            let mut stage_metrics = StageMetrics {
                name: stage_name,
                records_in: current_records.len() as u64,
                ..Default::default()
            };

            let mut output_records = Vec::new();
            let error_strategy = if stage_error_strategy != ErrorStrategy::default() {
                stage_error_strategy
            } else {
                self.default_error_strategy
            };

            for (rec_idx, record) in current_records.iter().enumerate() {
                // In dry-run mode, Load stages pass through without processing.
                if self.dry_run && stage_kind == StageKind::Load {
                    output_records.push(record.clone());
                    stage_metrics.records_out += 1;
                    continue;
                }

                let result = process_with_strategy(
                    &self.stages[stage_idx],
                    record,
                    error_strategy,
                    rec_idx,
                    &mut stage_metrics,
                    &mut self.dead_letter_queue,
                );

                match result {
                    ProcessResult::Output(recs) => {
                        stage_metrics.records_out += recs.len() as u64;
                        output_records.extend(recs);
                    }
                    ProcessResult::Filtered => {
                        stage_metrics.records_filtered += 1;
                    }
                    ProcessResult::Skipped => {
                        stage_metrics.skipped += 1;
                    }
                    ProcessResult::DeadLettered => {
                        stage_metrics.dead_lettered += 1;
                        stage_metrics.errors += 1;
                    }
                    ProcessResult::Error(e) => {
                        stage_metrics.errors += 1;
                        total_errors += 1;
                        self.state = PipelineState::Failed;
                        // Save checkpoint before failing.
                        let ckpt_id = format!("fail-{stage_idx}-{rec_idx}");
                        self.checkpoints.insert(
                            ckpt_id.clone(),
                            Checkpoint {
                                id: ckpt_id,
                                stage_index: stage_idx,
                                record_index: rec_idx,
                                records_processed: stage_metrics.records_out,
                                metrics_snapshot: PipelineMetrics {
                                    total_records: total_input,
                                    total_output: 0,
                                    total_errors,
                                    total_duration_us: 0,
                                    stage_metrics: all_stage_metrics.clone(),
                                    dry_run: self.dry_run,
                                },
                            },
                        );
                        return Err(e);
                    }
                }
            }

            total_errors += stage_metrics.errors;
            all_stage_metrics.push(stage_metrics);
            current_records = output_records;
        }

        // Save completion checkpoint.
        let ckpt_id = "completed".to_string();
        let metrics = PipelineMetrics {
            total_records: total_input,
            total_output: current_records.len() as u64,
            total_errors,
            total_duration_us: 0,
            stage_metrics: all_stage_metrics,
            dry_run: self.dry_run,
        };
        self.checkpoints.insert(
            ckpt_id.clone(),
            Checkpoint {
                id: ckpt_id,
                stage_index: self.stages.len(),
                record_index: 0,
                records_processed: metrics.total_output,
                metrics_snapshot: metrics.clone(),
            },
        );

        self.state = PipelineState::Completed;
        Ok((current_records, metrics))
    }

    /// Resume from a checkpoint.
    pub fn resume(
        &mut self,
        checkpoint_id: &str,
        remaining_input: Vec<Record>,
    ) -> Result<(Vec<Record>, PipelineMetrics), EtlError> {
        let checkpoint = self
            .checkpoints
            .get(checkpoint_id)
            .cloned()
            .ok_or_else(|| EtlError::CheckpointNotFound(checkpoint_id.to_string()))?;

        // Resume from the checkpoint's stage and record index.
        // For simplicity, we re-run from the checkpoint stage with the remaining input.
        if checkpoint.stage_index >= self.stages.len() {
            // Already completed.
            return Ok((remaining_input, checkpoint.metrics_snapshot));
        }

        self.state = PipelineState::Running;

        let total_input = remaining_input.len() as u64;
        let mut current_records = remaining_input;
        let mut all_stage_metrics = checkpoint.metrics_snapshot.stage_metrics.clone();
        let mut total_errors = checkpoint.metrics_snapshot.total_errors;
        let num_stages = self.stages.len();

        for stage_idx in checkpoint.stage_index..num_stages {
            let stage_name = self.stages[stage_idx].name.clone();
            let stage_kind = self.stages[stage_idx].kind;
            let stage_error_strategy = self.stages[stage_idx].error_strategy;

            let mut stage_metrics = StageMetrics {
                name: stage_name,
                records_in: current_records.len() as u64,
                ..Default::default()
            };

            let mut output_records = Vec::new();
            let error_strategy = if stage_error_strategy != ErrorStrategy::default() {
                stage_error_strategy
            } else {
                self.default_error_strategy
            };

            for (rec_idx, record) in current_records.iter().enumerate() {
                if self.dry_run && stage_kind == StageKind::Load {
                    output_records.push(record.clone());
                    stage_metrics.records_out += 1;
                    continue;
                }

                let result = process_with_strategy(
                    &self.stages[stage_idx],
                    record,
                    error_strategy,
                    rec_idx,
                    &mut stage_metrics,
                    &mut self.dead_letter_queue,
                );

                match result {
                    ProcessResult::Output(recs) => {
                        stage_metrics.records_out += recs.len() as u64;
                        output_records.extend(recs);
                    }
                    ProcessResult::Filtered => {
                        stage_metrics.records_filtered += 1;
                    }
                    ProcessResult::Skipped => {
                        stage_metrics.skipped += 1;
                    }
                    ProcessResult::DeadLettered => {
                        stage_metrics.dead_lettered += 1;
                        stage_metrics.errors += 1;
                    }
                    ProcessResult::Error(e) => {
                        self.state = PipelineState::Failed;
                        return Err(e);
                    }
                }
            }

            total_errors += stage_metrics.errors;
            all_stage_metrics.push(stage_metrics);
            current_records = output_records;
        }

        self.state = PipelineState::Completed;

        let metrics = PipelineMetrics {
            total_records: total_input,
            total_output: current_records.len() as u64,
            total_errors,
            total_duration_us: 0,
            stage_metrics: all_stage_metrics,
            dry_run: self.dry_run,
        };

        Ok((current_records, metrics))
    }
}

/// Process a record through a stage with the given error strategy.
/// Free function to avoid borrow conflicts on Pipeline.
fn process_with_strategy(
    stage: &Stage,
    record: &Record,
    strategy: ErrorStrategy,
    rec_idx: usize,
    metrics: &mut StageMetrics,
    dead_letter_queue: &mut Vec<DeadLetterEntry>,
) -> ProcessResult {
    match strategy {
        ErrorStrategy::Fail => match stage.process(record) {
            StageResult::Success(r) => ProcessResult::Output(vec![r]),
            StageResult::Filtered => ProcessResult::Filtered,
            StageResult::Split(recs) => ProcessResult::Output(recs),
            StageResult::Failed(reason) => {
                let err = match stage.kind {
                    StageKind::Extract => EtlError::ExtractFailed {
                        record_index: rec_idx,
                        reason,
                    },
                    StageKind::Load => EtlError::LoadFailed {
                        record_index: rec_idx,
                        reason,
                    },
                    _ => EtlError::TransformFailed {
                        record_index: rec_idx,
                        reason,
                    },
                };
                ProcessResult::Error(err)
            }
        },
        ErrorStrategy::Skip => match stage.process(record) {
            StageResult::Success(r) => ProcessResult::Output(vec![r]),
            StageResult::Filtered => ProcessResult::Filtered,
            StageResult::Split(recs) => ProcessResult::Output(recs),
            StageResult::Failed(_) => ProcessResult::Skipped,
        },
        ErrorStrategy::Retry(max_retries) => {
            let mut attempts = 0;
            loop {
                match stage.process(record) {
                    StageResult::Success(r) => return ProcessResult::Output(vec![r]),
                    StageResult::Filtered => return ProcessResult::Filtered,
                    StageResult::Split(recs) => return ProcessResult::Output(recs),
                    StageResult::Failed(reason) => {
                        attempts += 1;
                        metrics.retries += 1;
                        if attempts >= max_retries {
                            let err = EtlError::TransformFailed {
                                record_index: rec_idx,
                                reason,
                            };
                            return ProcessResult::Error(err);
                        }
                    }
                }
            }
        }
        ErrorStrategy::DeadLetter => match stage.process(record) {
            StageResult::Success(r) => ProcessResult::Output(vec![r]),
            StageResult::Filtered => ProcessResult::Filtered,
            StageResult::Split(recs) => ProcessResult::Output(recs),
            StageResult::Failed(error) => {
                dead_letter_queue.push(DeadLetterEntry {
                    record: record.clone(),
                    stage_name: stage.name.clone(),
                    error,
                    attempts: 1,
                });
                ProcessResult::DeadLettered
            }
        },
    }
}

/// Internal result of processing with error strategy applied.
enum ProcessResult {
    Output(Vec<Record>),
    Filtered,
    Skipped,
    DeadLettered,
    Error(EtlError),
}

// ── Builder helpers ──────────────────────────────────────────────

/// Create a passthrough stage that forwards records unchanged.
pub fn passthrough_stage(name: impl Into<String>, kind: StageKind) -> Stage {
    Stage::new(name, kind, |record| StageResult::Success(record.clone()))
}

/// Create a filter stage that keeps records matching the predicate.
pub fn filter_stage(
    name: impl Into<String>,
    predicate: impl Fn(&Record) -> bool + 'static,
) -> Stage {
    Stage::new(name, StageKind::Filter, move |record| {
        if predicate(record) {
            StageResult::Success(record.clone())
        } else {
            StageResult::Filtered
        }
    })
}

/// Create a transform stage that maps each record.
pub fn map_stage(
    name: impl Into<String>,
    mapper: impl Fn(&Record) -> Record + 'static,
) -> Stage {
    Stage::new(name, StageKind::Transform, move |record| {
        StageResult::Success(mapper(record))
    })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_records(count: usize) -> Vec<Record> {
        (0..count)
            .map(|i| {
                let mut r = Record::new(format!("rec-{i}"));
                r.set_field("value", serde_json::json!(i));
                r.set_meta("source", "test");
                r
            })
            .collect()
    }

    #[test]
    fn empty_pipeline_fails() {
        let mut pipeline = Pipeline::new("empty");
        let err = pipeline.run(vec![]).unwrap_err();
        assert_eq!(err, EtlError::NoStages);
    }

    #[test]
    fn passthrough_pipeline() {
        let mut pipeline = Pipeline::new("passthrough");
        pipeline.add_stage(passthrough_stage("extract", StageKind::Extract));
        pipeline.add_stage(passthrough_stage("load", StageKind::Load));

        let input = make_records(5);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 5);
        assert_eq!(metrics.total_records, 5);
        assert_eq!(metrics.total_output, 5);
        assert_eq!(metrics.total_errors, 0);
    }

    #[test]
    fn filter_stage_removes_records() {
        let mut pipeline = Pipeline::new("filter");
        pipeline.add_stage(passthrough_stage("extract", StageKind::Extract));
        pipeline.add_stage(filter_stage("only-even", |r| {
            r.get_field("value")
                .and_then(|v| v.as_u64())
                .map_or(false, |v| v % 2 == 0)
        }));

        let input = make_records(10);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 5); // 0, 2, 4, 6, 8
        assert_eq!(metrics.stage_metrics[1].records_filtered, 5);
    }

    #[test]
    fn transform_stage_modifies_records() {
        let mut pipeline = Pipeline::new("transform");
        pipeline.add_stage(map_stage("double", |r| {
            let mut out = r.clone();
            if let Some(v) = r.get_field("value").and_then(|v| v.as_u64()) {
                out.set_field("value", serde_json::json!(v * 2));
            }
            out
        }));

        let input = make_records(3);
        let (output, _) = pipeline.run(input).unwrap();
        assert_eq!(
            output[1].get_field("value").unwrap().as_u64().unwrap(),
            2
        );
    }

    #[test]
    fn fail_strategy_stops_pipeline() {
        let mut pipeline = Pipeline::new("fail-test");
        pipeline.add_stage(Stage::new("boom", StageKind::Transform, |_| {
            StageResult::Failed("kaboom".into())
        }));

        let input = make_records(3);
        let err = pipeline.run(input).unwrap_err();
        match err {
            EtlError::TransformFailed { reason, .. } => assert_eq!(reason, "kaboom"),
            other => panic!("unexpected error: {other:?}"),
        }
        assert_eq!(pipeline.state(), PipelineState::Failed);
    }

    #[test]
    fn skip_strategy_continues() {
        let mut pipeline = Pipeline::new("skip-test");
        let stage = Stage::new("maybe-fail", StageKind::Transform, |r| {
            let val = r.get_field("value").and_then(|v| v.as_u64()).unwrap_or(0);
            if val == 1 {
                StageResult::Failed("bad".into())
            } else {
                StageResult::Success(r.clone())
            }
        })
        .with_error_strategy(ErrorStrategy::Skip);
        pipeline.add_stage(stage);

        let input = make_records(3);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 2); // records 0 and 2
        assert_eq!(metrics.stage_metrics[0].skipped, 1);
    }

    #[test]
    fn dead_letter_strategy() {
        let mut pipeline = Pipeline::new("dead-letter-test");
        let stage = Stage::new("fail-some", StageKind::Transform, |r| {
            let val = r.get_field("value").and_then(|v| v.as_u64()).unwrap_or(0);
            if val % 3 == 0 {
                StageResult::Failed("divisible by 3".into())
            } else {
                StageResult::Success(r.clone())
            }
        })
        .with_error_strategy(ErrorStrategy::DeadLetter);
        pipeline.add_stage(stage);

        let input = make_records(9);
        let (output, _) = pipeline.run(input).unwrap();
        // Records 0, 3, 6 fail -> dead lettered
        assert_eq!(output.len(), 6);
        assert_eq!(pipeline.dead_letter_queue().len(), 3);
        assert_eq!(pipeline.dead_letter_queue()[0].stage_name, "fail-some");
    }

    #[test]
    fn retry_strategy_retries() {
        use std::cell::Cell;
        let counter = std::rc::Rc::new(Cell::new(0u32));
        let c = counter.clone();
        let stage = Stage::new("retry-stage", StageKind::Transform, move |r| {
            let n = c.get();
            c.set(n + 1);
            if n < 2 {
                StageResult::Failed("not yet".into())
            } else {
                StageResult::Success(r.clone())
            }
        })
        .with_error_strategy(ErrorStrategy::Retry(5));

        let mut pipeline = Pipeline::new("retry-test");
        pipeline.add_stage(stage);

        let input = make_records(1);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 1);
        assert!(metrics.stage_metrics[0].retries >= 2);
    }

    #[test]
    fn retry_exhaustion_fails() {
        let stage = Stage::new("always-fail", StageKind::Transform, |_| {
            StageResult::Failed("nope".into())
        })
        .with_error_strategy(ErrorStrategy::Retry(3));

        let mut pipeline = Pipeline::new("retry-exhaust");
        pipeline.add_stage(stage);

        let input = make_records(1);
        let err = pipeline.run(input).unwrap_err();
        assert!(matches!(err, EtlError::TransformFailed { .. }));
    }

    #[test]
    fn dry_run_skips_load() {
        let load_called = std::rc::Rc::new(Cell::new(false));
        let lc = load_called.clone();

        let mut pipeline = Pipeline::new("dry-run");
        pipeline.set_dry_run(true);
        pipeline.add_stage(passthrough_stage("extract", StageKind::Extract));
        pipeline.add_stage(Stage::new("load", StageKind::Load, move |r| {
            lc.set(true);
            StageResult::Success(r.clone())
        }));

        let input = make_records(3);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 3);
        assert!(!load_called.get());
        assert!(metrics.dry_run);
    }

    use std::cell::Cell;

    #[test]
    fn multi_stage_pipeline() {
        let mut pipeline = Pipeline::new("multi");
        pipeline.add_stage(passthrough_stage("extract", StageKind::Extract));
        pipeline.add_stage(filter_stage("filter", |r| {
            r.get_field("value")
                .and_then(|v| v.as_u64())
                .map_or(false, |v| v > 2)
        }));
        pipeline.add_stage(map_stage("transform", |r| {
            let mut out = r.clone();
            out.set_field("transformed", serde_json::json!(true));
            out
        }));
        pipeline.add_stage(passthrough_stage("load", StageKind::Load));

        let input = make_records(5);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 2); // values 3, 4
        assert_eq!(metrics.stage_metrics.len(), 4);
        for rec in &output {
            assert_eq!(rec.get_field("transformed"), Some(&serde_json::json!(true)));
        }
    }

    #[test]
    fn split_stage_expands_records() {
        let mut pipeline = Pipeline::new("split");
        pipeline.add_stage(Stage::new("splitter", StageKind::Transform, |r| {
            let mut r1 = r.clone();
            let mut r2 = r.clone();
            r1.id = format!("{}-a", r.id);
            r2.id = format!("{}-b", r.id);
            StageResult::Split(vec![r1, r2])
        }));

        let input = make_records(2);
        let (output, metrics) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 4);
        assert_eq!(metrics.stage_metrics[0].records_out, 4);
    }

    #[test]
    fn checkpoint_saved_on_completion() {
        let mut pipeline = Pipeline::new("ckpt");
        pipeline.add_stage(passthrough_stage("s1", StageKind::Extract));

        let input = make_records(3);
        pipeline.run(input).unwrap();

        let ckpt = pipeline.get_checkpoint("completed").unwrap();
        assert_eq!(ckpt.records_processed, 3);
    }

    #[test]
    fn checkpoint_saved_on_failure() {
        let mut pipeline = Pipeline::new("fail-ckpt");
        pipeline.add_stage(Stage::new("fail", StageKind::Extract, |_| {
            StageResult::Failed("oops".into())
        }));

        let input = make_records(1);
        let _ = pipeline.run(input);

        // Should have a failure checkpoint.
        let ckpt = pipeline.get_checkpoint("fail-0-0");
        assert!(ckpt.is_some());
    }

    #[test]
    fn pipeline_state_transitions() {
        let mut pipeline = Pipeline::new("states");
        pipeline.add_stage(passthrough_stage("s1", StageKind::Extract));

        assert_eq!(pipeline.state(), PipelineState::Idle);
        pipeline.run(make_records(1)).unwrap();
        assert_eq!(pipeline.state(), PipelineState::Completed);
    }

    #[test]
    fn record_metadata() {
        let mut r = Record::new("test");
        r.set_meta("source", "csv");
        r.set_meta("line", "42");
        assert_eq!(r.metadata.get("source"), Some(&"csv".to_string()));
        assert_eq!(r.metadata.get("line"), Some(&"42".to_string()));
    }

    #[test]
    fn metrics_error_rate() {
        let m = PipelineMetrics {
            total_records: 100,
            total_errors: 10,
            ..Default::default()
        };
        assert!((m.error_rate() - 0.1).abs() < f64::EPSILON);
        assert!((m.success_rate() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_zero_records() {
        let m = PipelineMetrics::default();
        assert!((m.error_rate()).abs() < f64::EPSILON);
    }

    #[test]
    fn default_error_strategy_propagates() {
        let mut pipeline = Pipeline::new("default-strategy");
        pipeline.set_default_error_strategy(ErrorStrategy::Skip);
        // Stage uses default (Fail) which will be overridden by pipeline default.
        pipeline.add_stage(Stage::new("maybe", StageKind::Transform, |r| {
            let val = r.get_field("value").and_then(|v| v.as_u64()).unwrap_or(0);
            if val == 0 {
                StageResult::Failed("zero".into())
            } else {
                StageResult::Success(r.clone())
            }
        }));

        let input = make_records(3);
        let (output, _) = pipeline.run(input).unwrap();
        assert_eq!(output.len(), 2); // record 0 skipped
    }

    #[test]
    fn resume_from_checkpoint() {
        let mut pipeline = Pipeline::new("resume");
        pipeline.add_stage(passthrough_stage("s1", StageKind::Extract));
        pipeline.add_stage(passthrough_stage("s2", StageKind::Load));

        // First run.
        let input = make_records(5);
        pipeline.run(input).unwrap();

        // Resume with new data from the completed checkpoint.
        let more_input = make_records(3);
        let (output, _metrics) = pipeline.resume("completed", more_input).unwrap();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn resume_missing_checkpoint() {
        let mut pipeline = Pipeline::new("no-ckpt");
        pipeline.add_stage(passthrough_stage("s1", StageKind::Extract));

        let err = pipeline.resume("nonexistent", vec![]).unwrap_err();
        assert!(matches!(err, EtlError::CheckpointNotFound(_)));
    }

    #[test]
    fn error_display() {
        let e = EtlError::ExtractFailed {
            record_index: 5,
            reason: "bad data".into(),
        };
        let s = format!("{e}");
        assert!(s.contains("extract failed"));
        assert!(s.contains("5"));
    }

    #[test]
    fn stage_kind_display() {
        assert_eq!(format!("{}", StageKind::Extract), "extract");
        assert_eq!(format!("{}", StageKind::Load), "load");
        assert_eq!(format!("{}", StageKind::Transform), "transform");
    }
}
