//! Data pipeline — stage definitions, typed inputs/outputs, fan-out/fan-in,
//! error handling, metrics, backpressure, composition, and dry-run mode.
//!
//! Replaces Node.js stream/pipeline libraries (Highland.js, RxJS pipelines)
//! with a pure-Rust data pipeline engine that tracks every stage.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Pipeline domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// Stage not found.
    StageNotFound(String),
    /// Pipeline not found.
    PipelineNotFound(String),
    /// Duplicate stage ID.
    DuplicateStage(String),
    /// Stage failed.
    StageFailed { stage: String, reason: String },
    /// Type mismatch between stages.
    TypeMismatch { from_stage: String, to_stage: String, expected: String, got: String },
    /// Pipeline already running.
    AlreadyRunning(String),
    /// Backpressure limit reached.
    BackpressureLimit { stage: String, limit: usize },
    /// Invalid pipeline configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StageNotFound(id) => write!(f, "stage not found: {id}"),
            Self::PipelineNotFound(id) => write!(f, "pipeline not found: {id}"),
            Self::DuplicateStage(id) => write!(f, "duplicate stage: {id}"),
            Self::StageFailed { stage, reason } => write!(f, "stage {stage} failed: {reason}"),
            Self::TypeMismatch { from_stage, to_stage, expected, got } => {
                write!(f, "type mismatch {from_stage}->{to_stage}: expected {expected}, got {got}")
            }
            Self::AlreadyRunning(id) => write!(f, "pipeline {id} already running"),
            Self::BackpressureLimit { stage, limit } => {
                write!(f, "backpressure limit {limit} reached at stage {stage}")
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for PipelineError {}

// ── Data Item ───────────────────────────────────────────────────

/// A typed data item flowing through the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataItem {
    pub id: String,
    pub data_type: String,
    pub payload: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

impl DataItem {
    pub fn new(id: impl Into<String>, data_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            data_type: data_type.into(),
            payload: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_field(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.payload.insert(key.into(), val.into());
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

// ── Enums ───────────────────────────────────────────────────────

/// Pipeline execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStatus {
    Idle,
    Running,
    Completed,
    Failed,
    Paused,
}

/// Stage execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Error handling strategy for a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorStrategy {
    /// Stop the pipeline on error.
    StopPipeline,
    /// Skip the failed item and continue.
    SkipItem,
    /// Send failed items to a dead letter queue.
    DeadLetterQueue,
    /// Retry the item.
    Retry { max_attempts: u32 },
}

/// Stage topology type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageTopology {
    /// Linear: single input, single output.
    Linear,
    /// Fan-out: one input, multiple outputs.
    FanOut { output_stages: Vec<String> },
    /// Fan-in: multiple inputs, single output.
    FanIn { input_stages: Vec<String> },
}

// ── Stage Definition ────────────────────────────────────────────

/// A pipeline stage definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDefinition {
    pub id: String,
    pub name: String,
    pub input_type: String,
    pub output_type: String,
    pub topology: StageTopology,
    pub error_strategy: ErrorStrategy,
    pub backpressure_limit: Option<usize>,
    pub description: String,
}

impl StageDefinition {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input_type: input_type.into(),
            output_type: output_type.into(),
            topology: StageTopology::Linear,
            error_strategy: ErrorStrategy::StopPipeline,
            backpressure_limit: None,
            description: String::new(),
        }
    }

    pub fn with_topology(mut self, t: StageTopology) -> Self {
        self.topology = t;
        self
    }

    pub fn with_error_strategy(mut self, s: ErrorStrategy) -> Self {
        self.error_strategy = s;
        self
    }

    pub fn with_backpressure(mut self, limit: usize) -> Self {
        self.backpressure_limit = Some(limit);
        self
    }
}

// ── Stage Metrics ───────────────────────────────────────────────

/// Metrics for a single stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageMetrics {
    pub items_in: u64,
    pub items_out: u64,
    pub items_failed: u64,
    pub items_skipped: u64,
    pub total_processing_ms: u64,
    pub min_processing_ms: Option<u64>,
    pub max_processing_ms: Option<u64>,
}

impl StageMetrics {
    pub fn record_success(&mut self, processing_ms: u64) {
        self.items_in += 1;
        self.items_out += 1;
        self.total_processing_ms += processing_ms;
        self.min_processing_ms = Some(
            self.min_processing_ms.map_or(processing_ms, |m| m.min(processing_ms)),
        );
        self.max_processing_ms = Some(
            self.max_processing_ms.map_or(processing_ms, |m| m.max(processing_ms)),
        );
    }

    pub fn record_failure(&mut self) {
        self.items_in += 1;
        self.items_failed += 1;
    }

    pub fn record_skip(&mut self) {
        self.items_in += 1;
        self.items_skipped += 1;
    }

    pub fn avg_processing_ms(&self) -> Option<f64> {
        if self.items_out > 0 {
            Some(self.total_processing_ms as f64 / self.items_out as f64)
        } else {
            None
        }
    }
}

// ── Stage Execution ─────────────────────────────────────────────

/// Runtime state of a stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageExecution {
    pub stage_id: String,
    pub status: StageStatus,
    pub metrics: StageMetrics,
    pub buffer: Vec<DataItem>,
    pub dead_letter: Vec<DataItem>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

impl StageExecution {
    fn new(stage_id: &str) -> Self {
        Self {
            stage_id: stage_id.to_string(),
            status: StageStatus::Pending,
            metrics: StageMetrics::default(),
            buffer: Vec::new(),
            dead_letter: Vec::new(),
            started_at: None,
            completed_at: None,
            error: None,
        }
    }
}

// ── Pipeline Definition ─────────────────────────────────────────

/// A data pipeline definition with ordered stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDefinition {
    pub id: String,
    pub name: String,
    pub stages: Vec<StageDefinition>,
    pub created_at: DateTime<Utc>,
}

impl PipelineDefinition {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            stages: Vec::new(),
            created_at: Utc::now(),
        }
    }

    pub fn add_stage(&mut self, stage: StageDefinition) -> Result<(), PipelineError> {
        if self.stages.iter().any(|s| s.id == stage.id) {
            return Err(PipelineError::DuplicateStage(stage.id));
        }
        self.stages.push(stage);
        Ok(())
    }

    /// Validate type compatibility between adjacent stages.
    pub fn validate_types(&self) -> Result<(), PipelineError> {
        for window in self.stages.windows(2) {
            if window[0].output_type != window[1].input_type {
                return Err(PipelineError::TypeMismatch {
                    from_stage: window[0].id.clone(),
                    to_stage: window[1].id.clone(),
                    expected: window[1].input_type.clone(),
                    got: window[0].output_type.clone(),
                });
            }
        }
        Ok(())
    }
}

// ── Pipeline Instance ───────────────────────────────────────────

/// A running pipeline instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInstance {
    pub instance_id: String,
    pub definition_id: String,
    pub status: PipelineStatus,
    pub stage_executions: Vec<StageExecution>,
    pub dry_run: bool,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl PipelineInstance {
    pub fn new(instance_id: impl Into<String>, definition: &PipelineDefinition) -> Self {
        let execs = definition.stages.iter()
            .map(|s| StageExecution::new(&s.id))
            .collect();
        Self {
            instance_id: instance_id.into(),
            definition_id: definition.id.clone(),
            status: PipelineStatus::Idle,
            stage_executions: execs,
            dry_run: false,
            started_at: None,
            completed_at: None,
        }
    }

    pub fn with_dry_run(mut self, dry: bool) -> Self {
        self.dry_run = dry;
        self
    }

    /// Start the pipeline.
    pub fn start(&mut self) -> Result<(), PipelineError> {
        if self.status == PipelineStatus::Running {
            return Err(PipelineError::AlreadyRunning(self.instance_id.clone()));
        }
        self.status = PipelineStatus::Running;
        self.started_at = Some(Utc::now());
        Ok(())
    }

    /// Feed items into the first stage.
    pub fn feed(&mut self, items: Vec<DataItem>) -> Result<(), PipelineError> {
        if self.stage_executions.is_empty() {
            return Err(PipelineError::InvalidConfig("no stages".into()));
        }
        self.stage_executions[0].buffer.extend(items);
        Ok(())
    }

    /// Process a stage: take items from buffer, transform, push to next.
    /// The `transform` closure applies the stage logic.
    pub fn process_stage<F>(
        &mut self,
        stage_index: usize,
        definition: &PipelineDefinition,
        transform: F,
    ) -> Result<Vec<DataItem>, PipelineError>
    where
        F: Fn(&DataItem) -> Result<DataItem, String>,
    {
        if stage_index >= self.stage_executions.len() {
            return Err(PipelineError::StageNotFound(format!("index {stage_index}")));
        }

        let stage_def = &definition.stages[stage_index];
        let exec = &mut self.stage_executions[stage_index];
        exec.status = StageStatus::Running;
        exec.started_at = Some(Utc::now());

        let items: Vec<DataItem> = std::mem::take(&mut exec.buffer);
        let mut outputs = Vec::new();

        // Check backpressure.
        if let Some(limit) = stage_def.backpressure_limit {
            if items.len() > limit {
                return Err(PipelineError::BackpressureLimit {
                    stage: stage_def.id.clone(),
                    limit,
                });
            }
        }

        for item in &items {
            if self.dry_run {
                exec.metrics.record_success(0);
                outputs.push(item.clone());
                continue;
            }

            match transform(item) {
                Ok(out) => {
                    exec.metrics.record_success(1); // simplified timing
                    outputs.push(out);
                }
                Err(reason) => match stage_def.error_strategy {
                    ErrorStrategy::StopPipeline => {
                        exec.status = StageStatus::Failed;
                        exec.error = Some(reason.clone());
                        exec.metrics.record_failure();
                        return Err(PipelineError::StageFailed {
                            stage: stage_def.id.clone(),
                            reason,
                        });
                    }
                    ErrorStrategy::SkipItem => {
                        exec.metrics.record_skip();
                    }
                    ErrorStrategy::DeadLetterQueue => {
                        exec.dead_letter.push(item.clone());
                        exec.metrics.record_failure();
                    }
                    ErrorStrategy::Retry { max_attempts } => {
                        let mut succeeded = false;
                        for _ in 1..max_attempts {
                            if let Ok(out) = transform(item) {
                                outputs.push(out);
                                exec.metrics.record_success(1);
                                succeeded = true;
                                break;
                            }
                        }
                        if !succeeded {
                            exec.metrics.record_failure();
                        }
                    }
                },
            }
        }

        exec.status = StageStatus::Completed;
        exec.completed_at = Some(Utc::now());

        // Push to next stage buffer.
        if stage_index + 1 < self.stage_executions.len() {
            self.stage_executions[stage_index + 1].buffer.extend(outputs.clone());
        }

        Ok(outputs)
    }

    /// Mark the pipeline as completed.
    pub fn complete(&mut self) {
        self.status = PipelineStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Mark the pipeline as failed.
    pub fn fail(&mut self, _reason: &str) {
        self.status = PipelineStatus::Failed;
        self.completed_at = Some(Utc::now());
    }

    /// Get aggregate metrics.
    pub fn aggregate_metrics(&self) -> HashMap<String, StageMetrics> {
        self.stage_executions.iter()
            .map(|e| (e.stage_id.clone(), e.metrics.clone()))
            .collect()
    }

    /// Get all dead-letter items across stages.
    pub fn dead_letter_items(&self) -> Vec<&DataItem> {
        self.stage_executions.iter()
            .flat_map(|e| e.dead_letter.iter())
            .collect()
    }
}

// ── Pipeline Composition ────────────────────────────────────────

/// Compose two pipeline definitions sequentially.
pub fn compose(
    a: &PipelineDefinition,
    b: &PipelineDefinition,
    combined_id: &str,
    combined_name: &str,
) -> Result<PipelineDefinition, PipelineError> {
    // Validate that a's last output type matches b's first input type.
    if let (Some(last_a), Some(first_b)) = (a.stages.last(), b.stages.first()) {
        if last_a.output_type != first_b.input_type {
            return Err(PipelineError::TypeMismatch {
                from_stage: last_a.id.clone(),
                to_stage: first_b.id.clone(),
                expected: first_b.input_type.clone(),
                got: last_a.output_type.clone(),
            });
        }
    }

    let mut combined = PipelineDefinition::new(combined_id, combined_name);
    for stage in &a.stages {
        combined.add_stage(stage.clone())?;
    }
    for stage in &b.stages {
        combined.add_stage(stage.clone())?;
    }
    Ok(combined)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn etl_pipeline() -> PipelineDefinition {
        let mut def = PipelineDefinition::new("etl", "ETL Pipeline");
        def.add_stage(StageDefinition::new("extract", "Extract", "raw", "raw")).unwrap();
        def.add_stage(StageDefinition::new("transform", "Transform", "raw", "clean")).unwrap();
        def.add_stage(StageDefinition::new("load", "Load", "clean", "clean")).unwrap();
        def
    }

    #[test]
    fn test_pipeline_definition() {
        let def = etl_pipeline();
        assert_eq!(def.stages.len(), 3);
    }

    #[test]
    fn test_duplicate_stage() {
        let mut def = PipelineDefinition::new("p", "P");
        def.add_stage(StageDefinition::new("a", "A", "x", "y")).unwrap();
        assert!(matches!(
            def.add_stage(StageDefinition::new("a", "A2", "x", "y")),
            Err(PipelineError::DuplicateStage(_))
        ));
    }

    #[test]
    fn test_type_validation() {
        let mut def = PipelineDefinition::new("p", "P");
        def.add_stage(StageDefinition::new("s1", "S1", "a", "b")).unwrap();
        def.add_stage(StageDefinition::new("s2", "S2", "c", "d")).unwrap();
        assert!(matches!(def.validate_types(), Err(PipelineError::TypeMismatch { .. })));

        let def2 = etl_pipeline();
        // raw->raw OK, raw->clean OK
        // Note: the third stage input is "clean" and second output is "clean" — OK.
        assert!(def2.validate_types().is_ok());
    }

    #[test]
    fn test_pipeline_execution() {
        let def = etl_pipeline();
        let mut inst = PipelineInstance::new("inst-1", &def);
        inst.start().unwrap();

        let items = vec![
            DataItem::new("1", "raw").with_field("val", "hello"),
            DataItem::new("2", "raw").with_field("val", "world"),
        ];
        inst.feed(items).unwrap();

        let out = inst.process_stage(0, &def, |item| Ok(item.clone())).unwrap();
        assert_eq!(out.len(), 2);

        let out = inst.process_stage(1, &def, |item| {
            let mut o = item.clone();
            o.data_type = "clean".into();
            if let Some(v) = o.payload.get("val").cloned() {
                o.payload.insert("val".to_string(), v.to_uppercase());
            }
            Ok(o)
        }).unwrap();
        assert_eq!(out[0].payload.get("val"), Some(&"HELLO".to_string()));

        inst.complete();
        assert_eq!(inst.status, PipelineStatus::Completed);
    }

    #[test]
    fn test_stage_failure_stop() {
        let def = etl_pipeline();
        let mut inst = PipelineInstance::new("inst-1", &def);
        inst.start().unwrap();
        inst.feed(vec![DataItem::new("1", "raw")]).unwrap();

        let err = inst.process_stage(0, &def, |_| Err("boom".into())).unwrap_err();
        assert!(matches!(err, PipelineError::StageFailed { .. }));
    }

    #[test]
    fn test_skip_item_strategy() {
        let mut def = PipelineDefinition::new("p", "P");
        def.add_stage(
            StageDefinition::new("s1", "S1", "a", "a")
                .with_error_strategy(ErrorStrategy::SkipItem),
        ).unwrap();
        let mut inst = PipelineInstance::new("i", &def);
        inst.start().unwrap();
        inst.feed(vec![DataItem::new("1", "a"), DataItem::new("2", "a")]).unwrap();

        let out = inst.process_stage(0, &def, |item| {
            if item.id == "1" { Err("skip".into()) } else { Ok(item.clone()) }
        }).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(inst.stage_executions[0].metrics.items_skipped, 1);
    }

    #[test]
    fn test_dead_letter_queue() {
        let mut def = PipelineDefinition::new("p", "P");
        def.add_stage(
            StageDefinition::new("s1", "S1", "a", "a")
                .with_error_strategy(ErrorStrategy::DeadLetterQueue),
        ).unwrap();
        let mut inst = PipelineInstance::new("i", &def);
        inst.start().unwrap();
        inst.feed(vec![DataItem::new("1", "a")]).unwrap();

        let _out = inst.process_stage(0, &def, |_| Err("fail".into())).unwrap();
        assert_eq!(inst.dead_letter_items().len(), 1);
    }

    #[test]
    fn test_backpressure() {
        let mut def = PipelineDefinition::new("p", "P");
        def.add_stage(
            StageDefinition::new("s1", "S1", "a", "a").with_backpressure(1),
        ).unwrap();
        let mut inst = PipelineInstance::new("i", &def);
        inst.start().unwrap();
        inst.feed(vec![DataItem::new("1", "a"), DataItem::new("2", "a")]).unwrap();

        let err = inst.process_stage(0, &def, |i| Ok(i.clone())).unwrap_err();
        assert!(matches!(err, PipelineError::BackpressureLimit { .. }));
    }

    #[test]
    fn test_dry_run() {
        let def = etl_pipeline();
        let mut inst = PipelineInstance::new("i", &def).with_dry_run(true);
        inst.start().unwrap();
        inst.feed(vec![DataItem::new("1", "raw")]).unwrap();

        let out = inst.process_stage(0, &def, |_| Err("would fail".into())).unwrap();
        // Dry run passes items through without calling transform.
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_metrics() {
        let def = etl_pipeline();
        let mut inst = PipelineInstance::new("i", &def);
        inst.start().unwrap();
        inst.feed(vec![DataItem::new("1", "raw"), DataItem::new("2", "raw")]).unwrap();

        inst.process_stage(0, &def, |i| Ok(i.clone())).unwrap();
        let metrics = inst.aggregate_metrics();
        assert_eq!(metrics["extract"].items_out, 2);
    }

    #[test]
    fn test_compose_pipelines() {
        let mut a = PipelineDefinition::new("a", "A");
        a.add_stage(StageDefinition::new("s1", "S1", "raw", "mid")).unwrap();
        let mut b = PipelineDefinition::new("b", "B");
        b.add_stage(StageDefinition::new("s2", "S2", "mid", "final")).unwrap();

        let combined = compose(&a, &b, "ab", "AB").unwrap();
        assert_eq!(combined.stages.len(), 2);
    }

    #[test]
    fn test_compose_type_mismatch() {
        let mut a = PipelineDefinition::new("a", "A");
        a.add_stage(StageDefinition::new("s1", "S1", "raw", "mid")).unwrap();
        let mut b = PipelineDefinition::new("b", "B");
        b.add_stage(StageDefinition::new("s2", "S2", "other", "final")).unwrap();

        assert!(matches!(compose(&a, &b, "ab", "AB"), Err(PipelineError::TypeMismatch { .. })));
    }

    #[test]
    fn test_fan_out_topology() {
        let stage = StageDefinition::new("s", "S", "a", "b")
            .with_topology(StageTopology::FanOut { output_stages: vec!["s2".into(), "s3".into()] });
        assert!(matches!(stage.topology, StageTopology::FanOut { .. }));
    }
}
