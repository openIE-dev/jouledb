//! Saga pattern — compensating transactions, forward/backward recovery,
//! saga coordinator, step execute/compensate, saga log, partial failure,
//! and per-step timeouts.
//!
//! Replaces JS saga libraries (redux-saga, NestJS sagas) with a pure-Rust
//! saga orchestrator for distributed transaction coordination.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Saga domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SagaError {
    /// Step not found.
    StepNotFound(String),
    /// Saga not found.
    SagaNotFound(String),
    /// Step execution failed.
    StepFailed { step: String, reason: String },
    /// Compensation failed.
    CompensationFailed { step: String, reason: String },
    /// Step timed out.
    StepTimeout { step: String, timeout_ms: u64 },
    /// Saga already completed.
    AlreadyCompleted(String),
    /// Duplicate step ID.
    DuplicateStep(String),
    /// Invalid saga state.
    InvalidState(String),
}

impl std::fmt::Display for SagaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepNotFound(id) => write!(f, "step not found: {id}"),
            Self::SagaNotFound(id) => write!(f, "saga not found: {id}"),
            Self::StepFailed { step, reason } => write!(f, "step {step} failed: {reason}"),
            Self::CompensationFailed { step, reason } => {
                write!(f, "compensation for {step} failed: {reason}")
            }
            Self::StepTimeout { step, timeout_ms } => {
                write!(f, "step {step} timed out after {timeout_ms}ms")
            }
            Self::AlreadyCompleted(id) => write!(f, "saga {id} already completed"),
            Self::DuplicateStep(id) => write!(f, "duplicate step: {id}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
        }
    }
}

impl std::error::Error for SagaError {}

// ── Enums ───────────────────────────────────────────────────────

/// Overall saga status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SagaStatus {
    Pending,
    Running,
    Completed,
    Compensating,
    CompensationCompleted,
    CompensationFailed,
    Failed,
}

/// Individual step status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Executing,
    Executed,
    Failed,
    Compensating,
    Compensated,
    CompensationFailed,
    Skipped,
}

/// Recovery strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Compensate all completed steps in reverse (backward recovery).
    Backward,
    /// Retry the failed step and continue (forward recovery).
    Forward,
}

// ── Step Definition ─────────────────────────────────────────────

/// A saga step with execute and compensate descriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SagaStep {
    pub id: String,
    pub name: String,
    pub description: String,
    pub timeout_ms: Option<u64>,
    pub max_retries: u32,
    pub has_compensation: bool,
}

impl SagaStep {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            timeout_ms: None,
            max_retries: 0,
            has_compensation: true,
        }
    }

    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    pub fn with_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn without_compensation(mut self) -> Self {
        self.has_compensation = false;
        self
    }
}

// ── Saga Log Entry ──────────────────────────────────────────────

/// A log entry in the saga execution log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SagaLogEntry {
    pub timestamp: DateTime<Utc>,
    pub step_id: String,
    pub event: SagaLogEvent,
    pub data: HashMap<String, String>,
}

/// Events logged during saga execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SagaLogEvent {
    SagaStarted,
    StepStarted,
    StepCompleted,
    StepFailed { reason: String },
    CompensationStarted,
    CompensationCompleted,
    CompensationFailed { reason: String },
    SagaCompleted,
    SagaCompensationCompleted,
    SagaFailed { reason: String },
}

// ── Step Execution ──────────────────────────────────────────────

/// Runtime state of a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecution {
    pub step_id: String,
    pub status: StepStatus,
    pub attempt: u32,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub result_data: HashMap<String, String>,
    pub error: Option<String>,
}

impl StepExecution {
    fn new(step_id: &str) -> Self {
        Self {
            step_id: step_id.to_string(),
            status: StepStatus::Pending,
            attempt: 0,
            started_at: None,
            completed_at: None,
            result_data: HashMap::new(),
            error: None,
        }
    }
}

// ── Saga Definition ─────────────────────────────────────────────

/// A saga definition with ordered steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaDefinition {
    pub id: String,
    pub name: String,
    pub steps: Vec<SagaStep>,
    pub recovery_strategy: RecoveryStrategy,
}

impl SagaDefinition {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            steps: Vec::new(),
            recovery_strategy: RecoveryStrategy::Backward,
        }
    }

    pub fn add_step(&mut self, step: SagaStep) -> Result<(), SagaError> {
        if self.steps.iter().any(|s| s.id == step.id) {
            return Err(SagaError::DuplicateStep(step.id));
        }
        self.steps.push(step);
        Ok(())
    }

    pub fn with_recovery(mut self, strategy: RecoveryStrategy) -> Self {
        self.recovery_strategy = strategy;
        self
    }
}

// ── Saga Instance ───────────────────────────────────────────────

/// A running saga instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SagaInstance {
    pub instance_id: String,
    pub definition_id: String,
    pub status: SagaStatus,
    pub current_step_index: usize,
    pub step_executions: Vec<StepExecution>,
    pub log: Vec<SagaLogEntry>,
    pub context: HashMap<String, String>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl SagaInstance {
    pub fn new(instance_id: impl Into<String>, definition: &SagaDefinition) -> Self {
        let step_execs = definition.steps.iter()
            .map(|s| StepExecution::new(&s.id))
            .collect();
        Self {
            instance_id: instance_id.into(),
            definition_id: definition.id.clone(),
            status: SagaStatus::Pending,
            current_step_index: 0,
            step_executions: step_execs,
            log: Vec::new(),
            context: HashMap::new(),
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    fn record(&mut self, step_id: &str, event: SagaLogEvent) {
        self.log.push(SagaLogEntry {
            timestamp: Utc::now(),
            step_id: step_id.to_string(),
            event,
            data: self.context.clone(),
        });
    }

    /// Start the saga.
    pub fn start(&mut self) -> Result<(), SagaError> {
        if self.status != SagaStatus::Pending {
            return Err(SagaError::InvalidState("saga not pending".into()));
        }
        self.status = SagaStatus::Running;
        self.record("", SagaLogEvent::SagaStarted);
        Ok(())
    }

    /// Begin executing the current step.
    pub fn begin_step(&mut self) -> Result<&str, SagaError> {
        if self.status != SagaStatus::Running {
            return Err(SagaError::InvalidState("saga not running".into()));
        }
        if self.current_step_index >= self.step_executions.len() {
            return Err(SagaError::InvalidState("no more steps".into()));
        }
        let exec = &mut self.step_executions[self.current_step_index];
        exec.status = StepStatus::Executing;
        exec.attempt += 1;
        exec.started_at = Some(Utc::now());
        let id = exec.step_id.clone();
        self.record(&id, SagaLogEvent::StepStarted);
        Ok(&self.step_executions[self.current_step_index].step_id)
    }

    /// Complete the current step with result data.
    pub fn complete_step(&mut self, data: HashMap<String, String>) -> Result<(), SagaError> {
        if self.current_step_index >= self.step_executions.len() {
            return Err(SagaError::InvalidState("no step in progress".into()));
        }
        let exec = &mut self.step_executions[self.current_step_index];
        exec.status = StepStatus::Executed;
        exec.completed_at = Some(Utc::now());
        exec.result_data = data.clone();
        let id = exec.step_id.clone();
        for (k, v) in data {
            self.context.insert(k, v);
        }
        self.record(&id, SagaLogEvent::StepCompleted);
        self.current_step_index += 1;

        // If all steps done, mark completed.
        if self.current_step_index >= self.step_executions.len() {
            self.status = SagaStatus::Completed;
            self.completed_at = Some(Utc::now());
            self.record("", SagaLogEvent::SagaCompleted);
        }
        Ok(())
    }

    /// Fail the current step.
    pub fn fail_step(&mut self, reason: &str) -> Result<(), SagaError> {
        if self.current_step_index >= self.step_executions.len() {
            return Err(SagaError::InvalidState("no step in progress".into()));
        }
        let exec = &mut self.step_executions[self.current_step_index];
        exec.status = StepStatus::Failed;
        exec.completed_at = Some(Utc::now());
        exec.error = Some(reason.to_string());
        let id = exec.step_id.clone();
        self.record(&id, SagaLogEvent::StepFailed { reason: reason.to_string() });
        self.status = SagaStatus::Failed;
        Ok(())
    }

    /// Begin backward compensation from the last completed step.
    pub fn begin_compensation(&mut self) -> Result<Vec<String>, SagaError> {
        self.status = SagaStatus::Compensating;
        let mut to_compensate = Vec::new();
        // Walk backward through executed steps.
        for i in (0..self.step_executions.len()).rev() {
            if self.step_executions[i].status == StepStatus::Executed {
                to_compensate.push(self.step_executions[i].step_id.clone());
                self.step_executions[i].status = StepStatus::Compensating;
                let id = self.step_executions[i].step_id.clone();
                self.record(&id, SagaLogEvent::CompensationStarted);
            }
        }
        Ok(to_compensate)
    }

    /// Mark a step's compensation as completed.
    pub fn complete_compensation(&mut self, step_id: &str) -> Result<(), SagaError> {
        let exec = self.step_executions.iter_mut()
            .find(|e| e.step_id == step_id)
            .ok_or_else(|| SagaError::StepNotFound(step_id.to_string()))?;
        exec.status = StepStatus::Compensated;
        self.record(step_id, SagaLogEvent::CompensationCompleted);

        // Check if all compensations done.
        let all_compensated = self.step_executions.iter().all(|e| {
            !matches!(e.status, StepStatus::Compensating)
        });
        if all_compensated {
            self.status = SagaStatus::CompensationCompleted;
            self.completed_at = Some(Utc::now());
            self.record("", SagaLogEvent::SagaCompensationCompleted);
        }
        Ok(())
    }

    /// Mark a step's compensation as failed.
    pub fn fail_compensation(&mut self, step_id: &str, reason: &str) -> Result<(), SagaError> {
        let exec = self.step_executions.iter_mut()
            .find(|e| e.step_id == step_id)
            .ok_or_else(|| SagaError::StepNotFound(step_id.to_string()))?;
        exec.status = StepStatus::CompensationFailed;
        exec.error = Some(reason.to_string());
        self.record(step_id, SagaLogEvent::CompensationFailed { reason: reason.to_string() });
        self.status = SagaStatus::CompensationFailed;
        Ok(())
    }

    /// Get all completed step IDs.
    pub fn completed_steps(&self) -> Vec<&str> {
        self.step_executions.iter()
            .filter(|e| e.status == StepStatus::Executed || e.status == StepStatus::Compensated)
            .map(|e| e.step_id.as_str())
            .collect()
    }

    /// Serialize the saga state.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Restore from JSON.
    pub fn from_json(json: &str) -> Result<Self, SagaError> {
        serde_json::from_str(json).map_err(|e| SagaError::InvalidState(e.to_string()))
    }
}

// ── Saga Coordinator ────────────────────────────────────────────

/// Coordinates saga definitions and instances.
#[derive(Debug, Default)]
pub struct SagaCoordinator {
    pub definitions: HashMap<String, SagaDefinition>,
    pub instances: HashMap<String, SagaInstance>,
}

impl SagaCoordinator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a saga definition.
    pub fn register(&mut self, def: SagaDefinition) {
        self.definitions.insert(def.id.clone(), def);
    }

    /// Create a new saga instance.
    pub fn create_instance(
        &mut self,
        definition_id: &str,
        instance_id: impl Into<String>,
    ) -> Result<&SagaInstance, SagaError> {
        let def = self.definitions.get(definition_id)
            .ok_or_else(|| SagaError::SagaNotFound(definition_id.to_string()))?;
        let inst = SagaInstance::new(instance_id, def);
        let iid = inst.instance_id.clone();
        self.instances.insert(iid.clone(), inst);
        Ok(self.instances.get(&iid).unwrap())
    }

    /// Get a mutable instance.
    pub fn instance_mut(&mut self, id: &str) -> Result<&mut SagaInstance, SagaError> {
        self.instances.get_mut(id)
            .ok_or_else(|| SagaError::SagaNotFound(id.to_string()))
    }

    /// List instances by status.
    pub fn instances_by_status(&self, status: SagaStatus) -> Vec<&SagaInstance> {
        self.instances.values().filter(|i| i.status == status).collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn order_saga() -> SagaDefinition {
        let mut def = SagaDefinition::new("order-saga", "Order Saga");
        def.add_step(SagaStep::new("reserve-inventory", "Reserve Inventory")).unwrap();
        def.add_step(SagaStep::new("charge-payment", "Charge Payment").with_timeout_ms(5000)).unwrap();
        def.add_step(SagaStep::new("ship-order", "Ship Order")).unwrap();
        def
    }

    #[test]
    fn test_saga_definition() {
        let def = order_saga();
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.recovery_strategy, RecoveryStrategy::Backward);
    }

    #[test]
    fn test_duplicate_step() {
        let mut def = SagaDefinition::new("s", "S");
        def.add_step(SagaStep::new("a", "A")).unwrap();
        assert!(matches!(def.add_step(SagaStep::new("a", "A2")), Err(SagaError::DuplicateStep(_))));
    }

    #[test]
    fn test_happy_path() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();
        assert_eq!(inst.status, SagaStatus::Running);

        for _ in 0..3 {
            inst.begin_step().unwrap();
            inst.complete_step(HashMap::new()).unwrap();
        }
        assert_eq!(inst.status, SagaStatus::Completed);
    }

    #[test]
    fn test_step_failure_and_compensation() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();

        // Step 1 succeeds.
        inst.begin_step().unwrap();
        inst.complete_step(HashMap::new()).unwrap();

        // Step 2 succeeds.
        inst.begin_step().unwrap();
        inst.complete_step(HashMap::new()).unwrap();

        // Step 3 fails.
        inst.begin_step().unwrap();
        inst.fail_step("shipping unavailable").unwrap();
        assert_eq!(inst.status, SagaStatus::Failed);

        // Compensate.
        let to_comp = inst.begin_compensation().unwrap();
        assert_eq!(to_comp.len(), 2); // steps 1 and 2
        assert_eq!(inst.status, SagaStatus::Compensating);

        for step_id in &to_comp {
            inst.complete_compensation(step_id).unwrap();
        }
        assert_eq!(inst.status, SagaStatus::CompensationCompleted);
    }

    #[test]
    fn test_compensation_failure() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();

        inst.begin_step().unwrap();
        inst.complete_step(HashMap::new()).unwrap();
        inst.begin_step().unwrap();
        inst.fail_step("error").unwrap();

        let to_comp = inst.begin_compensation().unwrap();
        inst.fail_compensation(&to_comp[0], "cannot undo").unwrap();
        assert_eq!(inst.status, SagaStatus::CompensationFailed);
    }

    #[test]
    fn test_saga_log() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();
        inst.begin_step().unwrap();
        inst.complete_step(HashMap::new()).unwrap();
        // SagaStarted, StepStarted, StepCompleted
        assert!(inst.log.len() >= 3);
    }

    #[test]
    fn test_context_propagation() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();

        inst.begin_step().unwrap();
        let mut data = HashMap::new();
        data.insert("reservation_id".to_string(), "R-123".to_string());
        inst.complete_step(data).unwrap();

        assert_eq!(inst.context.get("reservation_id"), Some(&"R-123".to_string()));
    }

    #[test]
    fn test_serialize_restore() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();
        inst.begin_step().unwrap();
        inst.complete_step(HashMap::new()).unwrap();

        let json = inst.to_json();
        let restored = SagaInstance::from_json(&json).unwrap();
        assert_eq!(restored.instance_id, "inst-1");
        assert_eq!(restored.status, SagaStatus::Running);
    }

    #[test]
    fn test_coordinator() {
        let mut coord = SagaCoordinator::new();
        coord.register(order_saga());
        coord.create_instance("order-saga", "inst-1").unwrap();

        let inst = coord.instance_mut("inst-1").unwrap();
        inst.start().unwrap();
        assert_eq!(inst.status, SagaStatus::Running);
    }

    #[test]
    fn test_coordinator_missing_definition() {
        let mut coord = SagaCoordinator::new();
        assert!(matches!(
            coord.create_instance("missing", "i"),
            Err(SagaError::SagaNotFound(_))
        ));
    }

    #[test]
    fn test_step_with_retries() {
        let step = SagaStep::new("s", "S").with_retries(3).with_timeout_ms(1000);
        assert_eq!(step.max_retries, 3);
        assert_eq!(step.timeout_ms, Some(1000));
    }

    #[test]
    fn test_instances_by_status() {
        let mut coord = SagaCoordinator::new();
        coord.register(order_saga());
        coord.create_instance("order-saga", "i1").unwrap();
        coord.create_instance("order-saga", "i2").unwrap();
        coord.instance_mut("i1").unwrap().start().unwrap();

        assert_eq!(coord.instances_by_status(SagaStatus::Running).len(), 1);
        assert_eq!(coord.instances_by_status(SagaStatus::Pending).len(), 1);
    }

    #[test]
    fn test_completed_steps() {
        let def = order_saga();
        let mut inst = SagaInstance::new("inst-1", &def);
        inst.start().unwrap();
        inst.begin_step().unwrap();
        inst.complete_step(HashMap::new()).unwrap();
        assert_eq!(inst.completed_steps(), vec!["reserve-inventory"]);
    }
}
