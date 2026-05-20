//! Workflow engine — step definitions, sequential/parallel execution, branching,
//! retry policies, timeouts, state persistence, dependency graphs, and history.
//!
//! Replaces Node.js workflow libraries (Temporal SDK, Bull queues) with a
//! pure-Rust workflow engine that tracks every step from definition to completion.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Workflow domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowError {
    /// Step not found.
    StepNotFound(String),
    /// Workflow not found.
    WorkflowNotFound(String),
    /// Circular dependency detected.
    CircularDependency { step: String, cycle: Vec<String> },
    /// Step already exists.
    DuplicateStep(String),
    /// Invalid state transition.
    InvalidTransition { from: WorkflowStatus, to: WorkflowStatus },
    /// Dependency not met.
    DependencyNotMet { step: String, depends_on: String },
    /// Step timed out.
    StepTimeout { step: String, timeout_ms: u64 },
    /// Max retries exceeded.
    MaxRetriesExceeded { step: String, attempts: u32 },
}

impl std::fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepNotFound(id) => write!(f, "step not found: {id}"),
            Self::WorkflowNotFound(id) => write!(f, "workflow not found: {id}"),
            Self::CircularDependency { step, cycle } => {
                write!(f, "circular dependency at {step}: {cycle:?}")
            }
            Self::DuplicateStep(id) => write!(f, "duplicate step: {id}"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition from {from:?} to {to:?}")
            }
            Self::DependencyNotMet { step, depends_on } => {
                write!(f, "step {step} depends on {depends_on} which is not completed")
            }
            Self::StepTimeout { step, timeout_ms } => {
                write!(f, "step {step} timed out after {timeout_ms}ms")
            }
            Self::MaxRetriesExceeded { step, attempts } => {
                write!(f, "step {step} exceeded max retries ({attempts} attempts)")
            }
        }
    }
}

impl std::error::Error for WorkflowError {}

// ── Enums ───────────────────────────────────────────────────────

/// Workflow execution status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Individual step status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
    TimedOut,
}

/// Execution mode for a group of steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

// ── Retry Policy ────────────────────────────────────────────────

/// Retry policy for a step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub backoff_multiplier: u32,
    pub max_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            backoff_multiplier: 2,
            max_delay_ms: 30_000,
        }
    }
}

impl RetryPolicy {
    /// Compute delay for the given attempt (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let mut delay = self.initial_delay_ms;
        for _ in 0..attempt {
            delay = delay.saturating_mul(self.backoff_multiplier as u64);
        }
        delay.min(self.max_delay_ms)
    }
}

// ── Condition ───────────────────────────────────────────────────

/// A branching condition for conditional steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Condition {
    /// Always true.
    Always,
    /// True when a specific step completed successfully.
    StepCompleted(String),
    /// True when a specific step failed.
    StepFailed(String),
    /// True when an output key equals a value.
    OutputEquals { step_id: String, key: String, value: String },
    /// All conditions must be true.
    All(Vec<Condition>),
    /// Any condition must be true.
    Any(Vec<Condition>),
}

// ── Step Definition ─────────────────────────────────────────────

/// A workflow step definition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub depends_on: Vec<String>,
    pub input_keys: Vec<String>,
    pub output_keys: Vec<String>,
    pub timeout_ms: Option<u64>,
    pub retry_policy: Option<RetryPolicy>,
    pub condition: Option<Condition>,
    pub execution_mode: ExecutionMode,
}

impl StepDefinition {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            depends_on: Vec::new(),
            input_keys: Vec::new(),
            output_keys: Vec::new(),
            timeout_ms: None,
            retry_policy: None,
            condition: None,
            execution_mode: ExecutionMode::Sequential,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.depends_on.push(dep.into());
        self
    }

    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    pub fn with_condition(mut self, cond: Condition) -> Self {
        self.condition = Some(cond);
        self
    }

    pub fn with_input(mut self, key: impl Into<String>) -> Self {
        self.input_keys.push(key.into());
        self
    }

    pub fn with_output(mut self, key: impl Into<String>) -> Self {
        self.output_keys.push(key.into());
        self
    }
}

// ── Step Execution Record ───────────────────────────────────────

/// Record of a single step execution attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepExecution {
    pub step_id: String,
    pub attempt: u32,
    pub status: StepStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub inputs: HashMap<String, String>,
    pub outputs: HashMap<String, String>,
    pub error: Option<String>,
}

// ── Execution History Entry ─────────────────────────────────────

/// An entry in the workflow execution history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub event: HistoryEvent,
}

/// Events recorded in execution history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryEvent {
    WorkflowStarted,
    WorkflowCompleted,
    WorkflowFailed { error: String },
    WorkflowCancelled,
    WorkflowPaused,
    WorkflowResumed,
    StepStarted { step_id: String, attempt: u32 },
    StepCompleted { step_id: String, attempt: u32 },
    StepFailed { step_id: String, attempt: u32, error: String },
    StepSkipped { step_id: String, reason: String },
    StepTimedOut { step_id: String, attempt: u32 },
}

// ── Workflow Definition ─────────────────────────────────────────

/// A workflow definition with steps and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub steps: Vec<StepDefinition>,
    pub created_at: DateTime<Utc>,
    pub version: u32,
}

impl WorkflowDefinition {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            steps: Vec::new(),
            created_at: Utc::now(),
            version: 1,
        }
    }

    /// Add a step, checking for duplicates.
    pub fn add_step(&mut self, step: StepDefinition) -> Result<(), WorkflowError> {
        if self.steps.iter().any(|s| s.id == step.id) {
            return Err(WorkflowError::DuplicateStep(step.id));
        }
        self.steps.push(step);
        Ok(())
    }

    /// Get the topological order of steps respecting dependencies.
    pub fn topological_order(&self) -> Result<Vec<String>, WorkflowError> {
        let mut visited: HashMap<String, u8> = HashMap::new(); // 0=unvisited, 1=in-progress, 2=done
        let mut order = Vec::new();

        for step in &self.steps {
            visited.insert(step.id.clone(), 0);
        }

        for step in &self.steps {
            if visited[&step.id] == 0 {
                self.topo_visit(&step.id, &mut visited, &mut order, &mut Vec::new())?;
            }
        }

        Ok(order)
    }

    fn topo_visit(
        &self,
        node: &str,
        visited: &mut HashMap<String, u8>,
        order: &mut Vec<String>,
        path: &mut Vec<String>,
    ) -> Result<(), WorkflowError> {
        if let Some(&state) = visited.get(node) {
            if state == 1 {
                return Err(WorkflowError::CircularDependency {
                    step: node.to_string(),
                    cycle: path.clone(),
                });
            }
            if state == 2 {
                return Ok(());
            }
        }

        visited.insert(node.to_string(), 1);
        path.push(node.to_string());

        if let Some(step) = self.steps.iter().find(|s| s.id == node) {
            for dep in &step.depends_on {
                self.topo_visit(dep, visited, order, path)?;
            }
        }

        path.pop();
        visited.insert(node.to_string(), 2);
        order.push(node.to_string());
        Ok(())
    }

    /// Validate the workflow definition.
    pub fn validate(&self) -> Result<(), WorkflowError> {
        // Check for duplicate IDs.
        let mut seen = HashMap::new();
        for step in &self.steps {
            if seen.contains_key(&step.id) {
                return Err(WorkflowError::DuplicateStep(step.id.clone()));
            }
            seen.insert(step.id.clone(), true);
        }

        // Check dependencies exist.
        let ids: Vec<&str> = self.steps.iter().map(|s| s.id.as_str()).collect();
        for step in &self.steps {
            for dep in &step.depends_on {
                if !ids.contains(&dep.as_str()) {
                    return Err(WorkflowError::StepNotFound(dep.clone()));
                }
            }
        }

        // Check for cycles.
        let _ = self.topological_order()?;

        Ok(())
    }
}

// ── Workflow Instance ───────────────────────────────────────────

/// A running instance of a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowInstance {
    pub instance_id: String,
    pub definition_id: String,
    pub status: WorkflowStatus,
    pub step_statuses: HashMap<String, StepStatus>,
    pub step_executions: Vec<StepExecution>,
    pub history: Vec<HistoryEntry>,
    pub context: HashMap<String, String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl WorkflowInstance {
    pub fn new(instance_id: impl Into<String>, definition: &WorkflowDefinition) -> Self {
        let mut step_statuses = HashMap::new();
        for step in &definition.steps {
            step_statuses.insert(step.id.clone(), StepStatus::Pending);
        }
        Self {
            instance_id: instance_id.into(),
            definition_id: definition.id.clone(),
            status: WorkflowStatus::Pending,
            step_statuses,
            step_executions: Vec::new(),
            history: Vec::new(),
            context: HashMap::new(),
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    /// Record a history event.
    pub fn record(&mut self, event: HistoryEvent) {
        self.history.push(HistoryEntry {
            timestamp: Utc::now(),
            event,
        });
    }

    /// Start the workflow.
    pub fn start(&mut self) -> Result<(), WorkflowError> {
        if self.status != WorkflowStatus::Pending {
            return Err(WorkflowError::InvalidTransition {
                from: self.status,
                to: WorkflowStatus::Running,
            });
        }
        self.status = WorkflowStatus::Running;
        self.record(HistoryEvent::WorkflowStarted);
        Ok(())
    }

    /// Pause the workflow.
    pub fn pause(&mut self) -> Result<(), WorkflowError> {
        if self.status != WorkflowStatus::Running {
            return Err(WorkflowError::InvalidTransition {
                from: self.status,
                to: WorkflowStatus::Paused,
            });
        }
        self.status = WorkflowStatus::Paused;
        self.record(HistoryEvent::WorkflowPaused);
        Ok(())
    }

    /// Resume the workflow.
    pub fn resume(&mut self) -> Result<(), WorkflowError> {
        if self.status != WorkflowStatus::Paused {
            return Err(WorkflowError::InvalidTransition {
                from: self.status,
                to: WorkflowStatus::Running,
            });
        }
        self.status = WorkflowStatus::Running;
        self.record(HistoryEvent::WorkflowResumed);
        Ok(())
    }

    /// Cancel the workflow.
    pub fn cancel(&mut self) -> Result<(), WorkflowError> {
        if self.status == WorkflowStatus::Completed || self.status == WorkflowStatus::Cancelled {
            return Err(WorkflowError::InvalidTransition {
                from: self.status,
                to: WorkflowStatus::Cancelled,
            });
        }
        self.status = WorkflowStatus::Cancelled;
        self.record(HistoryEvent::WorkflowCancelled);
        Ok(())
    }

    /// Start a step.
    pub fn start_step(&mut self, step_id: &str, attempt: u32) -> Result<(), WorkflowError> {
        let status = self.step_statuses.get(step_id)
            .ok_or_else(|| WorkflowError::StepNotFound(step_id.to_string()))?;
        if *status != StepStatus::Pending && *status != StepStatus::Failed {
            return Ok(());
        }
        self.step_statuses.insert(step_id.to_string(), StepStatus::Running);
        let exec = StepExecution {
            step_id: step_id.to_string(),
            attempt,
            status: StepStatus::Running,
            started_at: Utc::now(),
            completed_at: None,
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            error: None,
        };
        self.step_executions.push(exec);
        self.record(HistoryEvent::StepStarted {
            step_id: step_id.to_string(),
            attempt,
        });
        Ok(())
    }

    /// Complete a step with outputs.
    pub fn complete_step(
        &mut self,
        step_id: &str,
        attempt: u32,
        outputs: HashMap<String, String>,
    ) -> Result<(), WorkflowError> {
        self.step_statuses
            .get(step_id)
            .ok_or_else(|| WorkflowError::StepNotFound(step_id.to_string()))?;
        self.step_statuses.insert(step_id.to_string(), StepStatus::Completed);
        // Update execution record.
        for exec in self.step_executions.iter_mut().rev() {
            if exec.step_id == step_id && exec.attempt == attempt {
                exec.status = StepStatus::Completed;
                exec.completed_at = Some(Utc::now());
                exec.outputs = outputs.clone();
                break;
            }
        }
        // Merge outputs into context.
        for (k, v) in &outputs {
            self.context.insert(k.clone(), v.clone());
        }
        self.record(HistoryEvent::StepCompleted {
            step_id: step_id.to_string(),
            attempt,
        });
        Ok(())
    }

    /// Fail a step.
    pub fn fail_step(
        &mut self,
        step_id: &str,
        attempt: u32,
        error: &str,
    ) -> Result<(), WorkflowError> {
        self.step_statuses
            .get(step_id)
            .ok_or_else(|| WorkflowError::StepNotFound(step_id.to_string()))?;
        self.step_statuses.insert(step_id.to_string(), StepStatus::Failed);
        for exec in self.step_executions.iter_mut().rev() {
            if exec.step_id == step_id && exec.attempt == attempt {
                exec.status = StepStatus::Failed;
                exec.completed_at = Some(Utc::now());
                exec.error = Some(error.to_string());
                break;
            }
        }
        self.record(HistoryEvent::StepFailed {
            step_id: step_id.to_string(),
            attempt,
            error: error.to_string(),
        });
        Ok(())
    }

    /// Skip a step.
    pub fn skip_step(&mut self, step_id: &str, reason: &str) -> Result<(), WorkflowError> {
        self.step_statuses
            .get(step_id)
            .ok_or_else(|| WorkflowError::StepNotFound(step_id.to_string()))?;
        self.step_statuses.insert(step_id.to_string(), StepStatus::Skipped);
        self.record(HistoryEvent::StepSkipped {
            step_id: step_id.to_string(),
            reason: reason.to_string(),
        });
        Ok(())
    }

    /// Check whether all steps are terminal (completed/skipped/failed).
    pub fn is_terminal(&self) -> bool {
        self.step_statuses.values().all(|s| {
            matches!(s, StepStatus::Completed | StepStatus::Skipped | StepStatus::Failed)
        })
    }

    /// Mark workflow as completed if all steps are terminal.
    pub fn try_complete(&mut self) -> bool {
        if self.is_terminal() {
            let all_ok = self.step_statuses.values().all(|s| {
                matches!(s, StepStatus::Completed | StepStatus::Skipped)
            });
            if all_ok {
                self.status = WorkflowStatus::Completed;
                self.completed_at = Some(Utc::now());
                self.record(HistoryEvent::WorkflowCompleted);
            } else {
                self.status = WorkflowStatus::Failed;
                self.completed_at = Some(Utc::now());
                self.record(HistoryEvent::WorkflowFailed {
                    error: "one or more steps failed".to_string(),
                });
            }
            true
        } else {
            false
        }
    }

    /// Get steps that are ready to run (all dependencies completed).
    pub fn ready_steps(&self, definition: &WorkflowDefinition) -> Vec<String> {
        let mut ready = Vec::new();
        for step in &definition.steps {
            if self.step_statuses.get(&step.id) != Some(&StepStatus::Pending) {
                continue;
            }
            let deps_met = step.depends_on.iter().all(|dep| {
                matches!(
                    self.step_statuses.get(dep),
                    Some(StepStatus::Completed) | Some(StepStatus::Skipped)
                )
            });
            if deps_met {
                ready.push(step.id.clone());
            }
        }
        ready
    }
}

// ── Condition Evaluator ─────────────────────────────────────────

/// Evaluate a condition against a workflow instance.
pub fn evaluate_condition(cond: &Condition, instance: &WorkflowInstance) -> bool {
    match cond {
        Condition::Always => true,
        Condition::StepCompleted(id) => {
            instance.step_statuses.get(id) == Some(&StepStatus::Completed)
        }
        Condition::StepFailed(id) => {
            instance.step_statuses.get(id) == Some(&StepStatus::Failed)
        }
        Condition::OutputEquals { step_id: _, key, value } => {
            instance.context.get(key).map_or(false, |v| v == value)
        }
        Condition::All(conds) => conds.iter().all(|c| evaluate_condition(c, instance)),
        Condition::Any(conds) => conds.iter().any(|c| evaluate_condition(c, instance)),
    }
}

// ── Workflow Registry ───────────────────────────────────────────

/// Registry of workflow definitions and instances.
#[derive(Debug, Default)]
pub struct WorkflowRegistry {
    pub definitions: HashMap<String, WorkflowDefinition>,
    pub instances: HashMap<String, WorkflowInstance>,
}

impl WorkflowRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a workflow definition.
    pub fn register(&mut self, def: WorkflowDefinition) -> Result<(), WorkflowError> {
        def.validate()?;
        self.definitions.insert(def.id.clone(), def);
        Ok(())
    }

    /// Create a new instance of a workflow definition.
    pub fn create_instance(
        &mut self,
        definition_id: &str,
        instance_id: impl Into<String>,
    ) -> Result<&WorkflowInstance, WorkflowError> {
        let def = self.definitions.get(definition_id)
            .ok_or_else(|| WorkflowError::WorkflowNotFound(definition_id.to_string()))?;
        let inst = WorkflowInstance::new(instance_id, def);
        let iid = inst.instance_id.clone();
        self.instances.insert(iid.clone(), inst);
        Ok(self.instances.get(&iid).unwrap())
    }

    /// Get a mutable reference to an instance.
    pub fn instance_mut(&mut self, id: &str) -> Result<&mut WorkflowInstance, WorkflowError> {
        self.instances
            .get_mut(id)
            .ok_or_else(|| WorkflowError::WorkflowNotFound(id.to_string()))
    }

    /// Serialize workflow state to JSON.
    pub fn persist_instance(&self, id: &str) -> Result<String, WorkflowError> {
        let inst = self.instances.get(id)
            .ok_or_else(|| WorkflowError::WorkflowNotFound(id.to_string()))?;
        Ok(serde_json::to_string_pretty(inst).unwrap_or_default())
    }

    /// Restore workflow state from JSON.
    pub fn restore_instance(&mut self, json: &str) -> Result<String, WorkflowError> {
        let inst: WorkflowInstance =
            serde_json::from_str(json).map_err(|_| WorkflowError::WorkflowNotFound("parse error".to_string()))?;
        let id = inst.instance_id.clone();
        self.instances.insert(id.clone(), inst);
        Ok(id)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_def() -> WorkflowDefinition {
        let mut def = WorkflowDefinition::new("wf-1", "Order Processing");
        def.add_step(StepDefinition::new("validate", "Validate Order")).unwrap();
        def.add_step(
            StepDefinition::new("charge", "Charge Payment")
                .with_dependency("validate")
                .with_timeout_ms(5000),
        )
        .unwrap();
        def.add_step(
            StepDefinition::new("ship", "Ship Items")
                .with_dependency("charge"),
        )
        .unwrap();
        def
    }

    #[test]
    fn test_workflow_definition_creation() {
        let def = sample_def();
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.name, "Order Processing");
    }

    #[test]
    fn test_duplicate_step_rejected() {
        let mut def = WorkflowDefinition::new("w", "W");
        def.add_step(StepDefinition::new("a", "A")).unwrap();
        let err = def.add_step(StepDefinition::new("a", "A2")).unwrap_err();
        assert!(matches!(err, WorkflowError::DuplicateStep(_)));
    }

    #[test]
    fn test_topological_order() {
        let def = sample_def();
        let order = def.topological_order().unwrap();
        let vi = order.iter().position(|s| s == "validate").unwrap();
        let ci = order.iter().position(|s| s == "charge").unwrap();
        let si = order.iter().position(|s| s == "ship").unwrap();
        assert!(vi < ci);
        assert!(ci < si);
    }

    #[test]
    fn test_circular_dependency_detected() {
        let mut def = WorkflowDefinition::new("w", "W");
        def.add_step(StepDefinition::new("a", "A").with_dependency("b")).unwrap();
        def.add_step(StepDefinition::new("b", "B").with_dependency("a")).unwrap();
        assert!(matches!(def.validate(), Err(WorkflowError::CircularDependency { .. })));
    }

    #[test]
    fn test_instance_lifecycle() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();
        assert_eq!(inst.status, WorkflowStatus::Running);

        inst.pause().unwrap();
        assert_eq!(inst.status, WorkflowStatus::Paused);

        inst.resume().unwrap();
        assert_eq!(inst.status, WorkflowStatus::Running);
    }

    #[test]
    fn test_step_execution() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();

        inst.start_step("validate", 1).unwrap();
        let mut out = HashMap::new();
        out.insert("order_valid".to_string(), "true".to_string());
        inst.complete_step("validate", 1, out).unwrap();

        assert_eq!(inst.step_statuses["validate"], StepStatus::Completed);
        assert_eq!(inst.context["order_valid"], "true");
    }

    #[test]
    fn test_ready_steps() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();

        let ready = inst.ready_steps(&def);
        assert_eq!(ready, vec!["validate"]);

        inst.start_step("validate", 1).unwrap();
        inst.complete_step("validate", 1, HashMap::new()).unwrap();

        let ready = inst.ready_steps(&def);
        assert_eq!(ready, vec!["charge"]);
    }

    #[test]
    fn test_workflow_completion() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();

        for step_id in ["validate", "charge", "ship"] {
            inst.start_step(step_id, 1).unwrap();
            inst.complete_step(step_id, 1, HashMap::new()).unwrap();
        }

        assert!(inst.try_complete());
        assert_eq!(inst.status, WorkflowStatus::Completed);
    }

    #[test]
    fn test_workflow_failure_on_step_failure() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();

        inst.start_step("validate", 1).unwrap();
        inst.fail_step("validate", 1, "bad data").unwrap();
        inst.skip_step("charge", "dependency failed").unwrap();
        inst.skip_step("ship", "dependency failed").unwrap();

        assert!(inst.try_complete());
        assert_eq!(inst.status, WorkflowStatus::Failed);
    }

    #[test]
    fn test_condition_evaluation() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();

        assert!(!evaluate_condition(&Condition::StepCompleted("validate".into()), &inst));
        inst.start_step("validate", 1).unwrap();
        inst.complete_step("validate", 1, HashMap::new()).unwrap();
        assert!(evaluate_condition(&Condition::StepCompleted("validate".into()), &inst));
        assert!(evaluate_condition(&Condition::Always, &inst));
    }

    #[test]
    fn test_retry_policy_backoff() {
        let policy = RetryPolicy {
            max_attempts: 5,
            initial_delay_ms: 100,
            backoff_multiplier: 2,
            max_delay_ms: 1000,
        };
        assert_eq!(policy.delay_for_attempt(0), 100);
        assert_eq!(policy.delay_for_attempt(1), 200);
        assert_eq!(policy.delay_for_attempt(2), 400);
        assert_eq!(policy.delay_for_attempt(3), 800);
        assert_eq!(policy.delay_for_attempt(4), 1000); // capped
    }

    #[test]
    fn test_registry_persist_restore() {
        let def = sample_def();
        let mut registry = WorkflowRegistry::new();
        registry.register(def).unwrap();
        registry.create_instance("wf-1", "inst-1").unwrap();

        let json = registry.persist_instance("inst-1").unwrap();
        assert!(!json.is_empty());

        let mut registry2 = WorkflowRegistry::new();
        let id = registry2.restore_instance(&json).unwrap();
        assert_eq!(id, "inst-1");
        assert!(registry2.instances.contains_key("inst-1"));
    }

    #[test]
    fn test_cancel_workflow() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();
        inst.cancel().unwrap();
        assert_eq!(inst.status, WorkflowStatus::Cancelled);
        assert!(inst.cancel().is_err());
    }

    #[test]
    fn test_history_recording() {
        let def = sample_def();
        let mut inst = WorkflowInstance::new("inst-1", &def);
        inst.start().unwrap();
        inst.start_step("validate", 1).unwrap();
        inst.complete_step("validate", 1, HashMap::new()).unwrap();
        assert!(inst.history.len() >= 3);
        assert!(matches!(inst.history[0].event, HistoryEvent::WorkflowStarted));
    }
}
