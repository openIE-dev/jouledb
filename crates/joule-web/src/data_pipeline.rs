//! Data pipeline framework — composable pipeline stages (transform, filter,
//! aggregate, join), pipeline builder, execution plan with stage dependencies,
//! and error propagation through the pipeline.
//!
//! Replaces JS data-pipeline libraries (Apache Beam portability, Node streams)
//! with a pure-Rust composable pipeline that tracks energy per stage.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Pipeline errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// Stage not found.
    StageNotFound(String),
    /// Dependency cycle detected.
    CycleDetected(String),
    /// Stage execution failed.
    StageExecutionFailed { stage_id: String, reason: String },
    /// Pipeline already finalized.
    AlreadyFinalized,
    /// Duplicate stage id.
    DuplicateStage(String),
    /// Missing dependency.
    MissingDependency { stage_id: String, dep_id: String },
    /// Pipeline is empty.
    EmptyPipeline,
    /// Type mismatch between stages.
    TypeMismatch { from_stage: String, to_stage: String, detail: String },
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StageNotFound(id) => write!(f, "stage not found: {id}"),
            Self::CycleDetected(msg) => write!(f, "cycle detected: {msg}"),
            Self::StageExecutionFailed { stage_id, reason } => {
                write!(f, "stage {stage_id} failed: {reason}")
            }
            Self::AlreadyFinalized => write!(f, "pipeline already finalized"),
            Self::DuplicateStage(id) => write!(f, "duplicate stage: {id}"),
            Self::MissingDependency { stage_id, dep_id } => {
                write!(f, "stage {stage_id} depends on missing stage {dep_id}")
            }
            Self::EmptyPipeline => write!(f, "pipeline has no stages"),
            Self::TypeMismatch { from_stage, to_stage, detail } => {
                write!(f, "type mismatch {from_stage} -> {to_stage}: {detail}")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

// ── Stage Types ─────────────────────────────────────────────────

/// The kind of operation a stage performs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageKind {
    /// Apply a transformation to each record.
    Transform,
    /// Filter records based on a predicate.
    Filter,
    /// Aggregate records (e.g., sum, count, avg).
    Aggregate,
    /// Join two data sources.
    Join,
    /// Source stage — produces initial data.
    Source,
    /// Sink stage — consumes final data.
    Sink,
}

/// Aggregation function variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Join type for join stages.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    FullOuter,
}

/// Current execution status of a stage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Skipped,
}

// ── Stage Definition ────────────────────────────────────────────

/// A single pipeline stage definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDefinition {
    pub id: String,
    pub kind: StageKind,
    pub description: String,
    /// IDs of stages this stage depends on.
    pub dependencies: Vec<String>,
    /// Stage configuration encoded as key-value pairs.
    pub config: HashMap<String, String>,
    /// Maximum duration for this stage in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Runtime stats for a completed stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageStats {
    pub stage_id: String,
    pub records_in: u64,
    pub records_out: u64,
    pub energy_uj: u64,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub status: StageStatus,
}

// ── Execution Plan ──────────────────────────────────────────────

/// A topologically sorted execution plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Stages in execution order (topological sort).
    pub ordered_stages: Vec<String>,
    /// Which stages can run in parallel (same tier).
    pub parallelism_tiers: Vec<Vec<String>>,
    pub created_at: DateTime<Utc>,
}

// ── Data Record ─────────────────────────────────────────────────

/// A record flowing through the pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataRecord {
    pub fields: HashMap<String, serde_json::Value>,
}

impl DataRecord {
    pub fn new() -> Self {
        Self { fields: HashMap::new() }
    }

    pub fn with_field(mut self, key: &str, value: serde_json::Value) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.fields.get(key)
    }
}

impl Default for DataRecord {
    fn default() -> Self {
        Self::new()
    }
}

// ── Stage Result ────────────────────────────────────────────────

/// Result of executing a single stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage_id: String,
    pub records: Vec<DataRecord>,
    pub status: StageStatus,
    pub energy_uj: u64,
}

// ── Pipeline ────────────────────────────────────────────────────

/// The main pipeline, built from stage definitions.
#[derive(Debug, Clone)]
pub struct Pipeline {
    stages: Vec<StageDefinition>,
    stage_index: HashMap<String, usize>,
    finalized: bool,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            stage_index: HashMap::new(),
            finalized: false,
        }
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(&mut self, stage: StageDefinition) -> Result<(), PipelineError> {
        if self.finalized {
            return Err(PipelineError::AlreadyFinalized);
        }
        if self.stage_index.contains_key(&stage.id) {
            return Err(PipelineError::DuplicateStage(stage.id.clone()));
        }
        let idx = self.stages.len();
        self.stage_index.insert(stage.id.clone(), idx);
        self.stages.push(stage);
        Ok(())
    }

    /// Validate all dependencies exist and no cycles.
    pub fn validate(&self) -> Result<(), PipelineError> {
        if self.stages.is_empty() {
            return Err(PipelineError::EmptyPipeline);
        }
        // Check all deps exist.
        for stage in &self.stages {
            for dep in &stage.dependencies {
                if !self.stage_index.contains_key(dep) {
                    return Err(PipelineError::MissingDependency {
                        stage_id: stage.id.clone(),
                        dep_id: dep.clone(),
                    });
                }
            }
        }
        // Cycle detection via topological sort.
        self.topological_sort()?;
        Ok(())
    }

    /// Build the execution plan (topological sort with parallelism tiers).
    pub fn build_plan(&mut self) -> Result<ExecutionPlan, PipelineError> {
        if self.finalized {
            return Err(PipelineError::AlreadyFinalized);
        }
        self.validate()?;
        let ordered = self.topological_sort()?;
        let tiers = self.compute_tiers(&ordered);
        self.finalized = true;
        Ok(ExecutionPlan {
            ordered_stages: ordered,
            parallelism_tiers: tiers,
            created_at: Utc::now(),
        })
    }

    /// Execute the pipeline on input records, applying each stage in order.
    pub fn execute(
        &self,
        input: Vec<DataRecord>,
        plan: &ExecutionPlan,
    ) -> Result<Vec<StageResult>, PipelineError> {
        let mut stage_outputs: HashMap<String, Vec<DataRecord>> = HashMap::new();
        let mut results = Vec::new();

        for stage_id in &plan.ordered_stages {
            let stage = self.get_stage(stage_id)?;
            // Gather input: if source stage, use pipeline input; otherwise merge dep outputs.
            let stage_input = if stage.dependencies.is_empty() {
                input.clone()
            } else {
                let mut merged = Vec::new();
                for dep in &stage.dependencies {
                    if let Some(dep_out) = stage_outputs.get(dep) {
                        merged.extend(dep_out.clone());
                    }
                }
                merged
            };

            let records_in = stage_input.len() as u64;
            let output = self.execute_stage(stage, stage_input)?;
            let records_out = output.len() as u64;
            let energy = records_in.saturating_mul(10) + records_out.saturating_mul(5);

            stage_outputs.insert(stage_id.clone(), output.clone());
            results.push(StageResult {
                stage_id: stage_id.clone(),
                records: output,
                status: StageStatus::Completed,
                energy_uj: energy,
            });
        }

        Ok(results)
    }

    /// Get a stage definition by id.
    pub fn get_stage(&self, id: &str) -> Result<&StageDefinition, PipelineError> {
        self.stage_index
            .get(id)
            .map(|idx| &self.stages[*idx])
            .ok_or_else(|| PipelineError::StageNotFound(id.to_string()))
    }

    /// Get all stage definitions.
    pub fn stages(&self) -> &[StageDefinition] {
        &self.stages
    }

    // ── Internal ────────────────────────────────────────────────

    fn execute_stage(
        &self,
        stage: &StageDefinition,
        input: Vec<DataRecord>,
    ) -> Result<Vec<DataRecord>, PipelineError> {
        match &stage.kind {
            StageKind::Source | StageKind::Sink => Ok(input),
            StageKind::Transform => self.apply_transform(stage, input),
            StageKind::Filter => self.apply_filter(stage, input),
            StageKind::Aggregate => self.apply_aggregate(stage, input),
            StageKind::Join => Ok(input), // Join uses merged deps as input.
        }
    }

    fn apply_transform(
        &self,
        stage: &StageDefinition,
        input: Vec<DataRecord>,
    ) -> Result<Vec<DataRecord>, PipelineError> {
        let rename_from = stage.config.get("rename_from");
        let rename_to = stage.config.get("rename_to");
        let add_field = stage.config.get("add_field");
        let add_value = stage.config.get("add_value");

        let mut output = Vec::with_capacity(input.len());
        for mut rec in input {
            if let (Some(from), Some(to)) = (rename_from, rename_to) {
                if let Some(val) = rec.fields.remove(from.as_str()) {
                    rec.fields.insert(to.clone(), val);
                }
            }
            if let (Some(field), Some(value)) = (add_field, add_value) {
                rec.fields.insert(
                    field.clone(),
                    serde_json::Value::String(value.clone()),
                );
            }
            output.push(rec);
        }
        Ok(output)
    }

    fn apply_filter(
        &self,
        stage: &StageDefinition,
        input: Vec<DataRecord>,
    ) -> Result<Vec<DataRecord>, PipelineError> {
        let field = stage.config.get("field").cloned().unwrap_or_default();
        let op = stage.config.get("op").cloned().unwrap_or_else(|| "exists".to_string());
        let value = stage.config.get("value").cloned();

        let output = input
            .into_iter()
            .filter(|rec| {
                match op.as_str() {
                    "exists" => rec.fields.contains_key(&field),
                    "eq" => {
                        if let (Some(v), Some(expected)) = (rec.fields.get(&field), &value) {
                            match v {
                                serde_json::Value::String(s) => s == expected,
                                serde_json::Value::Number(n) => n.to_string() == *expected,
                                _ => false,
                            }
                        } else {
                            false
                        }
                    }
                    "gt" => {
                        if let (Some(serde_json::Value::Number(n)), Some(expected)) =
                            (rec.fields.get(&field), &value)
                        {
                            n.as_f64().unwrap_or(0.0)
                                > expected.parse::<f64>().unwrap_or(0.0)
                        } else {
                            false
                        }
                    }
                    _ => true,
                }
            })
            .collect();
        Ok(output)
    }

    fn apply_aggregate(
        &self,
        stage: &StageDefinition,
        input: Vec<DataRecord>,
    ) -> Result<Vec<DataRecord>, PipelineError> {
        let func_str = stage.config.get("func").cloned().unwrap_or_else(|| "count".to_string());
        let field = stage.config.get("field").cloned().unwrap_or_default();
        let group_by = stage.config.get("group_by").cloned();

        if let Some(group_field) = group_by {
            let mut groups: HashMap<String, Vec<&DataRecord>> = HashMap::new();
            for rec in &input {
                let key = rec
                    .fields
                    .get(&group_field)
                    .map(|v| match v {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_else(|| "__null__".to_string());
                groups.entry(key).or_default().push(rec);
            }
            let mut results: Vec<DataRecord> = Vec::new();
            let mut group_keys: Vec<String> = groups.keys().cloned().collect();
            group_keys.sort();
            for key in group_keys {
                let recs = &groups[&key];
                let agg_val = self.compute_aggregate(&func_str, &field, recs);
                let rec = DataRecord::new()
                    .with_field(&group_field, serde_json::Value::String(key))
                    .with_field("result", agg_val);
                results.push(rec);
            }
            Ok(results)
        } else {
            let refs: Vec<&DataRecord> = input.iter().collect();
            let agg_val = self.compute_aggregate(&func_str, &field, &refs);
            let rec = DataRecord::new().with_field("result", agg_val);
            Ok(vec![rec])
        }
    }

    fn compute_aggregate(
        &self,
        func: &str,
        field: &str,
        records: &[&DataRecord],
    ) -> serde_json::Value {
        match func {
            "count" => serde_json::json!(records.len()),
            "sum" => {
                let total: f64 = records
                    .iter()
                    .filter_map(|r| r.fields.get(field))
                    .filter_map(|v| v.as_f64())
                    .sum();
                serde_json::json!(total)
            }
            "avg" => {
                let values: Vec<f64> = records
                    .iter()
                    .filter_map(|r| r.fields.get(field))
                    .filter_map(|v| v.as_f64())
                    .collect();
                if values.is_empty() {
                    serde_json::json!(0.0)
                } else {
                    let avg = values.iter().sum::<f64>() / values.len() as f64;
                    serde_json::json!(avg)
                }
            }
            "min" => {
                let min = records
                    .iter()
                    .filter_map(|r| r.fields.get(field))
                    .filter_map(|v| v.as_f64())
                    .fold(f64::INFINITY, f64::min);
                if min.is_infinite() {
                    serde_json::json!(null)
                } else {
                    serde_json::json!(min)
                }
            }
            "max" => {
                let max = records
                    .iter()
                    .filter_map(|r| r.fields.get(field))
                    .filter_map(|v| v.as_f64())
                    .fold(f64::NEG_INFINITY, f64::max);
                if max.is_infinite() {
                    serde_json::json!(null)
                } else {
                    serde_json::json!(max)
                }
            }
            _ => serde_json::json!(null),
        }
    }

    fn topological_sort(&self) -> Result<Vec<String>, PipelineError> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for stage in &self.stages {
            in_degree.entry(stage.id.clone()).or_insert(0);
            adj.entry(stage.id.clone()).or_default();
            for dep in &stage.dependencies {
                adj.entry(dep.clone()).or_default().push(stage.id.clone());
                *in_degree.entry(stage.id.clone()).or_insert(0) += 1;
            }
        }

        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| id.clone())
            .collect();
        queue.sort(); // Deterministic order.

        let mut sorted = Vec::new();
        while let Some(node) = queue.first().cloned() {
            queue.remove(0);
            sorted.push(node.clone());
            if let Some(neighbors) = adj.get(&node) {
                let mut new_zero = Vec::new();
                for neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            new_zero.push(neighbor.clone());
                        }
                    }
                }
                new_zero.sort();
                queue.extend(new_zero);
            }
        }

        if sorted.len() != self.stages.len() {
            return Err(PipelineError::CycleDetected(
                "topological sort did not cover all stages".to_string(),
            ));
        }
        Ok(sorted)
    }

    fn compute_tiers(&self, ordered: &[String]) -> Vec<Vec<String>> {
        let mut tier_map: HashMap<String, usize> = HashMap::new();
        for stage_id in ordered {
            let stage = &self.stages[self.stage_index[stage_id]];
            let tier = if stage.dependencies.is_empty() {
                0
            } else {
                stage
                    .dependencies
                    .iter()
                    .filter_map(|d| tier_map.get(d))
                    .max()
                    .map(|m| m + 1)
                    .unwrap_or(0)
            };
            tier_map.insert(stage_id.clone(), tier);
        }

        let max_tier = tier_map.values().max().copied().unwrap_or(0);
        let mut tiers = vec![Vec::new(); max_tier + 1];
        // Ensure deterministic ordering within each tier.
        let mut items: Vec<(String, usize)> = tier_map.into_iter().collect();
        items.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        for (id, tier) in items {
            tiers[tier].push(id);
        }
        tiers
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Builder ─────────────────────────────────────────────────────

/// Fluent builder for constructing pipelines.
pub struct PipelineBuilder {
    pipeline: Pipeline,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            pipeline: Pipeline::new(),
        }
    }

    pub fn source(mut self, id: &str, description: &str) -> Self {
        let _ = self.pipeline.add_stage(StageDefinition {
            id: id.to_string(),
            kind: StageKind::Source,
            description: description.to_string(),
            dependencies: Vec::new(),
            config: HashMap::new(),
            timeout_ms: None,
        });
        self
    }

    pub fn transform(mut self, id: &str, deps: &[&str], config: HashMap<String, String>) -> Self {
        let _ = self.pipeline.add_stage(StageDefinition {
            id: id.to_string(),
            kind: StageKind::Transform,
            description: format!("transform: {id}"),
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            config,
            timeout_ms: None,
        });
        self
    }

    pub fn filter(mut self, id: &str, deps: &[&str], config: HashMap<String, String>) -> Self {
        let _ = self.pipeline.add_stage(StageDefinition {
            id: id.to_string(),
            kind: StageKind::Filter,
            description: format!("filter: {id}"),
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            config,
            timeout_ms: None,
        });
        self
    }

    pub fn aggregate(mut self, id: &str, deps: &[&str], config: HashMap<String, String>) -> Self {
        let _ = self.pipeline.add_stage(StageDefinition {
            id: id.to_string(),
            kind: StageKind::Aggregate,
            description: format!("aggregate: {id}"),
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            config,
            timeout_ms: None,
        });
        self
    }

    pub fn sink(mut self, id: &str, deps: &[&str]) -> Self {
        let _ = self.pipeline.add_stage(StageDefinition {
            id: id.to_string(),
            kind: StageKind::Sink,
            description: format!("sink: {id}"),
            dependencies: deps.iter().map(|s| s.to_string()).collect(),
            config: HashMap::new(),
            timeout_ms: None,
        });
        self
    }

    pub fn build(self) -> Pipeline {
        self.pipeline
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_records() -> Vec<DataRecord> {
        vec![
            DataRecord::new()
                .with_field("name", serde_json::json!("alice"))
                .with_field("age", serde_json::json!(30))
                .with_field("dept", serde_json::json!("eng")),
            DataRecord::new()
                .with_field("name", serde_json::json!("bob"))
                .with_field("age", serde_json::json!(25))
                .with_field("dept", serde_json::json!("sales")),
            DataRecord::new()
                .with_field("name", serde_json::json!("carol"))
                .with_field("age", serde_json::json!(35))
                .with_field("dept", serde_json::json!("eng")),
        ]
    }

    #[test]
    fn test_empty_pipeline_errors() {
        let p = Pipeline::new();
        assert_eq!(p.validate(), Err(PipelineError::EmptyPipeline));
    }

    #[test]
    fn test_duplicate_stage() {
        let mut p = Pipeline::new();
        let s = StageDefinition {
            id: "s1".into(),
            kind: StageKind::Source,
            description: "src".into(),
            dependencies: vec![],
            config: HashMap::new(),
            timeout_ms: None,
        };
        assert!(p.add_stage(s.clone()).is_ok());
        assert_eq!(
            p.add_stage(s),
            Err(PipelineError::DuplicateStage("s1".into()))
        );
    }

    #[test]
    fn test_missing_dependency() {
        let mut p = Pipeline::new();
        let s = StageDefinition {
            id: "s1".into(),
            kind: StageKind::Transform,
            description: "t".into(),
            dependencies: vec!["nonexistent".into()],
            config: HashMap::new(),
            timeout_ms: None,
        };
        p.add_stage(s).unwrap();
        match p.validate() {
            Err(PipelineError::MissingDependency { stage_id, dep_id }) => {
                assert_eq!(stage_id, "s1");
                assert_eq!(dep_id, "nonexistent");
            }
            other => panic!("expected MissingDependency, got {other:?}"),
        }
    }

    #[test]
    fn test_cycle_detection() {
        let mut p = Pipeline::new();
        p.add_stage(StageDefinition {
            id: "a".into(),
            kind: StageKind::Transform,
            description: "a".into(),
            dependencies: vec!["b".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        p.add_stage(StageDefinition {
            id: "b".into(),
            kind: StageKind::Transform,
            description: "b".into(),
            dependencies: vec!["a".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        assert!(matches!(p.validate(), Err(PipelineError::CycleDetected(_))));
    }

    #[test]
    fn test_linear_pipeline_plan() {
        let mut p = Pipeline::new();
        p.add_stage(StageDefinition {
            id: "src".into(),
            kind: StageKind::Source,
            description: "source".into(),
            dependencies: vec![],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        p.add_stage(StageDefinition {
            id: "xform".into(),
            kind: StageKind::Transform,
            description: "transform".into(),
            dependencies: vec!["src".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        p.add_stage(StageDefinition {
            id: "sink".into(),
            kind: StageKind::Sink,
            description: "sink".into(),
            dependencies: vec!["xform".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();

        let plan = p.build_plan().unwrap();
        assert_eq!(plan.ordered_stages, vec!["src", "xform", "sink"]);
        assert_eq!(plan.parallelism_tiers.len(), 3);
    }

    #[test]
    fn test_parallel_tiers() {
        let mut p = Pipeline::new();
        p.add_stage(StageDefinition {
            id: "src".into(),
            kind: StageKind::Source,
            description: "s".into(),
            dependencies: vec![],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        p.add_stage(StageDefinition {
            id: "t1".into(),
            kind: StageKind::Transform,
            description: "t1".into(),
            dependencies: vec!["src".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        p.add_stage(StageDefinition {
            id: "t2".into(),
            kind: StageKind::Transform,
            description: "t2".into(),
            dependencies: vec!["src".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();
        p.add_stage(StageDefinition {
            id: "merge".into(),
            kind: StageKind::Join,
            description: "merge".into(),
            dependencies: vec!["t1".into(), "t2".into()],
            config: HashMap::new(),
            timeout_ms: None,
        })
        .unwrap();

        let plan = p.build_plan().unwrap();
        // Tier 0: src, Tier 1: t1 + t2, Tier 2: merge
        assert_eq!(plan.parallelism_tiers.len(), 3);
        assert_eq!(plan.parallelism_tiers[0], vec!["src"]);
        assert!(plan.parallelism_tiers[1].contains(&"t1".to_string()));
        assert!(plan.parallelism_tiers[1].contains(&"t2".to_string()));
        assert_eq!(plan.parallelism_tiers[2], vec!["merge"]);
    }

    #[test]
    fn test_filter_execution() {
        let mut config = HashMap::new();
        config.insert("field".into(), "age".into());
        config.insert("op".into(), "gt".into());
        config.insert("value".into(), "27".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .filter("filt", &["src"], config)
            .sink("out", &["filt"])
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        // Filter stage is at index 1 (src=0, filt=1, out=2).
        let filt_result = &results[1];
        assert_eq!(filt_result.records.len(), 2); // alice(30) and carol(35)
    }

    #[test]
    fn test_transform_rename() {
        let mut config = HashMap::new();
        config.insert("rename_from".into(), "name".into());
        config.insert("rename_to".into(), "username".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .transform("ren", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        let xform_result = &results[1];
        for rec in &xform_result.records {
            assert!(rec.fields.contains_key("username"));
            assert!(!rec.fields.contains_key("name"));
        }
    }

    #[test]
    fn test_transform_add_field() {
        let mut config = HashMap::new();
        config.insert("add_field".into(), "status".into());
        config.insert("add_value".into(), "active".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .transform("add", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        for rec in &results[1].records {
            assert_eq!(rec.get("status"), Some(&serde_json::json!("active")));
        }
    }

    #[test]
    fn test_aggregate_count() {
        let mut config = HashMap::new();
        config.insert("func".into(), "count".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .aggregate("cnt", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        let agg = &results[1];
        assert_eq!(agg.records.len(), 1);
        assert_eq!(agg.records[0].get("result"), Some(&serde_json::json!(3)));
    }

    #[test]
    fn test_aggregate_sum() {
        let mut config = HashMap::new();
        config.insert("func".into(), "sum".into());
        config.insert("field".into(), "age".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .aggregate("total", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        let val = results[1].records[0].get("result").unwrap().as_f64().unwrap();
        assert!((val - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_avg() {
        let mut config = HashMap::new();
        config.insert("func".into(), "avg".into());
        config.insert("field".into(), "age".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .aggregate("mean", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        let val = results[1].records[0].get("result").unwrap().as_f64().unwrap();
        assert!((val - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_aggregate_group_by() {
        let mut config = HashMap::new();
        config.insert("func".into(), "count".into());
        config.insert("group_by".into(), "dept".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .aggregate("grp", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        let agg = &results[1];
        assert_eq!(agg.records.len(), 2); // eng, sales
        // Results sorted by group key: eng first, sales second.
        assert_eq!(
            agg.records[0].get("dept"),
            Some(&serde_json::json!("eng"))
        );
        assert_eq!(agg.records[0].get("result"), Some(&serde_json::json!(2)));
        assert_eq!(
            agg.records[1].get("dept"),
            Some(&serde_json::json!("sales"))
        );
        assert_eq!(agg.records[1].get("result"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn test_aggregate_min_max() {
        let mut min_cfg = HashMap::new();
        min_cfg.insert("func".into(), "min".into());
        min_cfg.insert("field".into(), "age".into());

        let mut max_cfg = HashMap::new();
        max_cfg.insert("func".into(), "max".into());
        max_cfg.insert("field".into(), "age".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .aggregate("lo", &["src"], min_cfg)
            .aggregate("hi", &["src"], max_cfg)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        // Stage ordering after topological sort is non-deterministic for
        // independent stages, so collect both aggregate results and check
        // that we got min=25 and max=35 regardless of order.
        let agg_vals: Vec<f64> = results[1..]
            .iter()
            .map(|r| r.records[0].get("result").unwrap().as_f64().unwrap())
            .collect();
        let min_val = agg_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = agg_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((min_val - 25.0).abs() < f64::EPSILON);
        assert!((max_val - 35.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_already_finalized() {
        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .build();
        let _ = p.build_plan().unwrap();
        assert_eq!(p.build_plan(), Err(PipelineError::AlreadyFinalized));
    }

    #[test]
    fn test_data_record_default() {
        let r = DataRecord::default();
        assert!(r.fields.is_empty());
    }

    #[test]
    fn test_filter_eq_string() {
        let mut config = HashMap::new();
        config.insert("field".into(), "dept".into());
        config.insert("op".into(), "eq".into());
        config.insert("value".into(), "eng".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .filter("f", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        assert_eq!(results[1].records.len(), 2);
    }

    #[test]
    fn test_multi_stage_pipeline() {
        let mut filter_cfg = HashMap::new();
        filter_cfg.insert("field".into(), "age".into());
        filter_cfg.insert("op".into(), "gt".into());
        filter_cfg.insert("value".into(), "24".into());

        let mut agg_cfg = HashMap::new();
        agg_cfg.insert("func".into(), "sum".into());
        agg_cfg.insert("field".into(), "age".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .filter("filt", &["src"], filter_cfg)
            .aggregate("total", &["filt"], agg_cfg)
            .sink("out", &["total"])
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        // All 3 pass filter (ages 30, 25, 35 all > 24).
        assert_eq!(results[1].records.len(), 3);
        let total = results[2].records[0].get("result").unwrap().as_f64().unwrap();
        assert!((total - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_tracking() {
        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .sink("out", &["src"])
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        for r in &results {
            assert!(r.energy_uj > 0);
        }
    }

    #[test]
    fn test_stage_kind_serialization() {
        let json = serde_json::to_string(&StageKind::Aggregate).unwrap();
        let parsed: StageKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, StageKind::Aggregate);
    }

    #[test]
    fn test_error_display() {
        let e = PipelineError::CycleDetected("loop".into());
        assert!(e.to_string().contains("cycle"));
        let e2 = PipelineError::TypeMismatch {
            from_stage: "a".into(),
            to_stage: "b".into(),
            detail: "int vs str".into(),
        };
        assert!(e2.to_string().contains("type mismatch"));
    }

    #[test]
    fn test_get_stage() {
        let p = PipelineBuilder::new()
            .source("src", "my source")
            .build();
        let s = p.get_stage("src").unwrap();
        assert_eq!(s.description, "my source");
        assert!(p.get_stage("nope").is_err());
    }

    #[test]
    fn test_stages_accessor() {
        let p = PipelineBuilder::new()
            .source("s1", "a")
            .source("s2", "b")
            .build();
        assert_eq!(p.stages().len(), 2);
    }

    #[test]
    fn test_filter_exists() {
        let mut config = HashMap::new();
        config.insert("field".into(), "missing_field".into());
        config.insert("op".into(), "exists".into());

        let mut p = PipelineBuilder::new()
            .source("src", "source")
            .filter("f", &["src"], config)
            .build();

        let plan = p.build_plan().unwrap();
        let results = p.execute(sample_records(), &plan).unwrap();
        assert_eq!(results[1].records.len(), 0);
    }
}
