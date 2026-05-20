//! Process manager / saga coordinator — long-running process definition,
//! event-driven step progression, timeout handling, compensation on failure,
//! process state persistence, and process restart.
//!
//! Replaces JS saga/process manager libraries (NestJS sagas, MassTransit) with
//! a pure-Rust process manager for orchestrating multi-step business processes.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Process manager errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessManagerError {
    /// Process not found.
    ProcessNotFound(String),
    /// Process already exists.
    ProcessAlreadyExists(String),
    /// Step not found.
    StepNotFound { process_id: String, step_id: String },
    /// Step execution failed.
    StepFailed { process_id: String, step_id: String, reason: String },
    /// Compensation failed.
    CompensationFailed { process_id: String, step_id: String, reason: String },
    /// Process already completed.
    AlreadyCompleted(String),
    /// Process already failed.
    AlreadyFailed(String),
    /// Step timed out.
    StepTimeout { process_id: String, step_id: String, timeout_ms: u64 },
    /// Invalid state transition.
    InvalidTransition { process_id: String, from: String, to: String },
    /// Process definition not found.
    DefinitionNotFound(String),
}

impl std::fmt::Display for ProcessManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProcessNotFound(id) => write!(f, "process not found: {id}"),
            Self::ProcessAlreadyExists(id) => write!(f, "process already exists: {id}"),
            Self::StepNotFound { process_id, step_id } => {
                write!(f, "step {step_id} not found in process {process_id}")
            }
            Self::StepFailed { process_id, step_id, reason } => {
                write!(f, "step {step_id} failed in {process_id}: {reason}")
            }
            Self::CompensationFailed { process_id, step_id, reason } => {
                write!(f, "compensation for {step_id} failed in {process_id}: {reason}")
            }
            Self::AlreadyCompleted(id) => write!(f, "process already completed: {id}"),
            Self::AlreadyFailed(id) => write!(f, "process already failed: {id}"),
            Self::StepTimeout { process_id, step_id, timeout_ms } => {
                write!(f, "step {step_id} in {process_id} timed out after {timeout_ms}ms")
            }
            Self::InvalidTransition { process_id, from, to } => {
                write!(f, "invalid transition in {process_id}: {from} -> {to}")
            }
            Self::DefinitionNotFound(id) => write!(f, "process definition not found: {id}"),
        }
    }
}

impl std::error::Error for ProcessManagerError {}

// ── Enums ───────────────────────────────────────────────────────

/// Overall process status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProcessStatus {
    Created,
    Running,
    Completed,
    Failed,
    Compensating,
    CompensationCompleted,
    CompensationFailed,
    TimedOut,
}

impl std::fmt::Display for ProcessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Created => "Created",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Compensating => "Compensating",
            Self::CompensationCompleted => "CompensationCompleted",
            Self::CompensationFailed => "CompensationFailed",
            Self::TimedOut => "TimedOut",
        };
        write!(f, "{label}")
    }
}

/// Individual step status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Executing,
    Completed,
    Failed,
    Compensating,
    Compensated,
    CompensationFailed,
    Skipped,
    TimedOut,
}

// ── Triggering Event ────────────────────────────────────────────

/// An event that can trigger process step progression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TriggerEvent {
    pub event_type: String,
    pub data: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

impl TriggerEvent {
    pub fn new(event_type: impl Into<String>, data: HashMap<String, String>) -> Self {
        Self {
            event_type: event_type.into(),
            data,
            timestamp: Utc::now(),
        }
    }
}

// ── Step Definition ─────────────────────────────────────────────

/// Definition of a process step.
#[derive(Clone)]
pub struct StepDefinition {
    pub step_id: String,
    /// The event type that triggers this step.
    pub trigger_event: String,
    /// Execute function: takes process state + event data, returns updated state or error.
    execute_fn: fn(&HashMap<String, String>, &HashMap<String, String>) -> Result<HashMap<String, String>, String>,
    /// Compensation function: takes process state, returns compensated state or error.
    compensate_fn: Option<fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>>,
    /// Timeout in milliseconds (0 = no timeout).
    pub timeout_ms: u64,
}

impl StepDefinition {
    pub fn new(
        step_id: impl Into<String>,
        trigger_event: impl Into<String>,
        execute_fn: fn(&HashMap<String, String>, &HashMap<String, String>) -> Result<HashMap<String, String>, String>,
    ) -> Self {
        Self {
            step_id: step_id.into(),
            trigger_event: trigger_event.into(),
            execute_fn,
            compensate_fn: None,
            timeout_ms: 0,
        }
    }

    pub fn with_compensation(
        mut self,
        compensate_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
    ) -> Self {
        self.compensate_fn = Some(compensate_fn);
        self
    }

    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }
}

impl std::fmt::Debug for StepDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StepDefinition")
            .field("step_id", &self.step_id)
            .field("trigger_event", &self.trigger_event)
            .field("timeout_ms", &self.timeout_ms)
            .field("has_compensation", &self.compensate_fn.is_some())
            .finish()
    }
}

// ── Step State ──────────────────────────────────────────────────

/// Runtime state of a step instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepState {
    pub step_id: String,
    pub status: StepStatus,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

// ── Process Definition ──────────────────────────────────────────

/// Defines a process (sequence of steps).
#[derive(Debug, Clone)]
pub struct ProcessDefinition {
    pub definition_id: String,
    pub steps: Vec<StepDefinition>,
}

impl ProcessDefinition {
    pub fn new(definition_id: impl Into<String>) -> Self {
        Self {
            definition_id: definition_id.into(),
            steps: Vec::new(),
        }
    }

    pub fn add_step(&mut self, step: StepDefinition) {
        self.steps.push(step);
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

// ── Process Instance ────────────────────────────────────────────

/// A running instance of a process.
#[derive(Debug, Clone)]
pub struct ProcessInstance {
    pub process_id: String,
    pub definition_id: String,
    pub status: ProcessStatus,
    /// Current step index in the definition.
    pub current_step: usize,
    /// Process-wide state (accumulated through steps).
    pub state: HashMap<String, String>,
    /// Per-step runtime state.
    pub step_states: Vec<StepState>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    /// Number of restart attempts.
    pub restart_count: u32,
}

impl ProcessInstance {
    pub fn new(
        process_id: impl Into<String>,
        definition_id: impl Into<String>,
        step_count: usize,
        initial_state: HashMap<String, String>,
    ) -> Self {
        let now = Utc::now();
        let def_id = definition_id.into();
        let step_states = (0..step_count)
            .map(|i| StepState {
                step_id: format!("step-{i}"),
                status: StepStatus::Pending,
                started_at: None,
                completed_at: None,
                error_message: None,
            })
            .collect();

        Self {
            process_id: process_id.into(),
            definition_id: def_id,
            status: ProcessStatus::Created,
            current_step: 0,
            state: initial_state,
            step_states,
            created_at: now,
            updated_at: now,
            completed_at: None,
            error_message: None,
            restart_count: 0,
        }
    }

    /// Check if the process is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            ProcessStatus::Completed
                | ProcessStatus::CompensationCompleted
                | ProcessStatus::CompensationFailed
        )
    }

    /// Get how many steps have completed.
    pub fn completed_steps(&self) -> usize {
        self.step_states
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count()
    }

    /// Get how many steps are pending.
    pub fn pending_steps(&self) -> usize {
        self.step_states
            .iter()
            .filter(|s| s.status == StepStatus::Pending)
            .count()
    }
}

// ── Process Manager ─────────────────────────────────────────────

/// Manages process definitions and instances.
#[derive(Debug)]
pub struct ProcessManager {
    definitions: HashMap<String, ProcessDefinition>,
    processes: HashMap<String, ProcessInstance>,
}

impl ProcessManager {
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            processes: HashMap::new(),
        }
    }

    /// Register a process definition.
    pub fn register_definition(&mut self, definition: ProcessDefinition) {
        self.definitions
            .insert(definition.definition_id.clone(), definition);
    }

    /// Start a new process instance.
    pub fn start_process(
        &mut self,
        process_id: impl Into<String>,
        definition_id: &str,
        initial_state: HashMap<String, String>,
    ) -> Result<String, ProcessManagerError> {
        let pid = process_id.into();

        if self.processes.contains_key(&pid) {
            return Err(ProcessManagerError::ProcessAlreadyExists(pid));
        }

        let def = self
            .definitions
            .get(definition_id)
            .ok_or_else(|| ProcessManagerError::DefinitionNotFound(definition_id.to_string()))?;

        let step_count = def.steps.len();
        let mut instance = ProcessInstance::new(&pid, definition_id, step_count, initial_state);

        // Copy step IDs from definition.
        for (i, step_def) in def.steps.iter().enumerate() {
            if i < instance.step_states.len() {
                instance.step_states[i].step_id = step_def.step_id.clone();
            }
        }

        instance.status = ProcessStatus::Running;
        instance.updated_at = Utc::now();

        self.processes.insert(pid.clone(), instance);
        Ok(pid)
    }

    /// Handle a trigger event, advancing the appropriate process.
    pub fn handle_event(
        &mut self,
        process_id: &str,
        event: &TriggerEvent,
    ) -> Result<ProcessStatus, ProcessManagerError> {
        // Look up definition for this process.
        let def_id = {
            let proc = self
                .processes
                .get(process_id)
                .ok_or_else(|| ProcessManagerError::ProcessNotFound(process_id.to_string()))?;

            if proc.is_terminal() {
                return Err(ProcessManagerError::AlreadyCompleted(process_id.to_string()));
            }
            if proc.status == ProcessStatus::Failed {
                return Err(ProcessManagerError::AlreadyFailed(process_id.to_string()));
            }
            proc.definition_id.clone()
        };

        let def = self
            .definitions
            .get(&def_id)
            .ok_or_else(|| ProcessManagerError::DefinitionNotFound(def_id.clone()))?
            .clone();

        let proc = self.processes.get_mut(process_id).unwrap();
        let step_idx = proc.current_step;

        if step_idx >= def.steps.len() {
            // All steps done.
            proc.status = ProcessStatus::Completed;
            proc.completed_at = Some(Utc::now());
            proc.updated_at = Utc::now();
            return Ok(ProcessStatus::Completed);
        }

        let step_def = &def.steps[step_idx];

        // Check if this event triggers the current step.
        if event.event_type != step_def.trigger_event {
            return Ok(proc.status);
        }

        // Mark step as executing.
        proc.step_states[step_idx].status = StepStatus::Executing;
        proc.step_states[step_idx].started_at = Some(Utc::now());

        // Execute step.
        let state_clone = proc.state.clone();
        let result = (step_def.execute_fn)(&state_clone, &event.data);

        match result {
            Ok(new_state) => {
                proc.state = new_state;
                proc.step_states[step_idx].status = StepStatus::Completed;
                proc.step_states[step_idx].completed_at = Some(Utc::now());
                proc.current_step += 1;
                proc.updated_at = Utc::now();

                // Check if all steps done.
                if proc.current_step >= def.steps.len() {
                    proc.status = ProcessStatus::Completed;
                    proc.completed_at = Some(Utc::now());
                }

                Ok(proc.status)
            }
            Err(reason) => {
                proc.step_states[step_idx].status = StepStatus::Failed;
                proc.step_states[step_idx].error_message = Some(reason.clone());
                proc.step_states[step_idx].completed_at = Some(Utc::now());
                proc.status = ProcessStatus::Failed;
                proc.error_message = Some(reason);
                proc.updated_at = Utc::now();

                Ok(ProcessStatus::Failed)
            }
        }
    }

    /// Run compensation for a failed process (backward from the failed step).
    pub fn compensate(
        &mut self,
        process_id: &str,
    ) -> Result<ProcessStatus, ProcessManagerError> {
        let def_id = {
            let proc = self
                .processes
                .get(process_id)
                .ok_or_else(|| ProcessManagerError::ProcessNotFound(process_id.to_string()))?;

            if proc.status != ProcessStatus::Failed && proc.status != ProcessStatus::TimedOut {
                return Err(ProcessManagerError::InvalidTransition {
                    process_id: process_id.to_string(),
                    from: proc.status.to_string(),
                    to: "Compensating".to_string(),
                });
            }
            proc.definition_id.clone()
        };

        let def = self
            .definitions
            .get(&def_id)
            .ok_or_else(|| ProcessManagerError::DefinitionNotFound(def_id.clone()))?
            .clone();

        let proc = self.processes.get_mut(process_id).unwrap();
        proc.status = ProcessStatus::Compensating;
        proc.updated_at = Utc::now();

        // Compensate completed steps in reverse order.
        let mut all_ok = true;
        for i in (0..proc.step_states.len()).rev() {
            if proc.step_states[i].status != StepStatus::Completed {
                continue;
            }

            if i < def.steps.len() {
                if let Some(comp_fn) = def.steps[i].compensate_fn {
                    proc.step_states[i].status = StepStatus::Compensating;
                    let state_clone = proc.state.clone();
                    match comp_fn(&state_clone) {
                        Ok(new_state) => {
                            proc.state = new_state;
                            proc.step_states[i].status = StepStatus::Compensated;
                            proc.step_states[i].completed_at = Some(Utc::now());
                        }
                        Err(reason) => {
                            proc.step_states[i].status = StepStatus::CompensationFailed;
                            proc.step_states[i].error_message = Some(reason);
                            all_ok = false;
                            break;
                        }
                    }
                } else {
                    // No compensation defined — mark as skipped.
                    proc.step_states[i].status = StepStatus::Skipped;
                }
            }
        }

        if all_ok {
            proc.status = ProcessStatus::CompensationCompleted;
        } else {
            proc.status = ProcessStatus::CompensationFailed;
        }
        proc.updated_at = Utc::now();

        Ok(proc.status)
    }

    /// Mark a step as timed out.
    pub fn timeout_step(
        &mut self,
        process_id: &str,
    ) -> Result<(), ProcessManagerError> {
        let proc = self
            .processes
            .get_mut(process_id)
            .ok_or_else(|| ProcessManagerError::ProcessNotFound(process_id.to_string()))?;

        if proc.is_terminal() {
            return Err(ProcessManagerError::AlreadyCompleted(process_id.to_string()));
        }

        let idx = proc.current_step;
        if idx < proc.step_states.len() {
            proc.step_states[idx].status = StepStatus::TimedOut;
            proc.step_states[idx].completed_at = Some(Utc::now());
        }
        proc.status = ProcessStatus::TimedOut;
        proc.updated_at = Utc::now();

        Ok(())
    }

    /// Restart a failed or timed-out process from its current step.
    pub fn restart(
        &mut self,
        process_id: &str,
    ) -> Result<(), ProcessManagerError> {
        let proc = self
            .processes
            .get_mut(process_id)
            .ok_or_else(|| ProcessManagerError::ProcessNotFound(process_id.to_string()))?;

        if proc.is_terminal() {
            return Err(ProcessManagerError::AlreadyCompleted(process_id.to_string()));
        }

        match proc.status {
            ProcessStatus::Failed | ProcessStatus::TimedOut => {
                // Reset the current failed step to pending.
                let idx = proc.current_step;
                if idx < proc.step_states.len() {
                    proc.step_states[idx].status = StepStatus::Pending;
                    proc.step_states[idx].error_message = None;
                    proc.step_states[idx].started_at = None;
                    proc.step_states[idx].completed_at = None;
                }
                proc.status = ProcessStatus::Running;
                proc.error_message = None;
                proc.restart_count += 1;
                proc.updated_at = Utc::now();
                Ok(())
            }
            _ => Err(ProcessManagerError::InvalidTransition {
                process_id: process_id.to_string(),
                from: proc.status.to_string(),
                to: "Running".to_string(),
            }),
        }
    }

    /// Get a process instance.
    pub fn get_process(&self, process_id: &str) -> Option<&ProcessInstance> {
        self.processes.get(process_id)
    }

    /// List all process IDs.
    pub fn process_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.processes.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Count processes by status.
    pub fn count_by_status(&self, status: ProcessStatus) -> usize {
        self.processes.values().filter(|p| p.status == status).count()
    }

    /// Total process count.
    pub fn process_count(&self) -> usize {
        self.processes.len()
    }

    /// Get a definition.
    pub fn get_definition(&self, def_id: &str) -> Option<&ProcessDefinition> {
        self.definitions.get(def_id)
    }
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn success_step(
        state: &HashMap<String, String>,
        event_data: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        let mut new_state = state.clone();
        for (k, v) in event_data {
            new_state.insert(k.clone(), v.clone());
        }
        Ok(new_state)
    }

    fn fail_step(
        _state: &HashMap<String, String>,
        _event_data: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        Err("step failed deliberately".to_string())
    }

    fn compensate_step(
        state: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        let mut new_state = state.clone();
        new_state.insert("compensated".to_string(), "true".to_string());
        Ok(new_state)
    }

    fn fail_compensate(
        _state: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        Err("compensation failed".to_string())
    }

    fn make_def() -> ProcessDefinition {
        let mut def = ProcessDefinition::new("order-process");
        def.add_step(
            StepDefinition::new("reserve-stock", "OrderPlaced", success_step)
                .with_compensation(compensate_step),
        );
        def.add_step(
            StepDefinition::new("charge-payment", "StockReserved", success_step)
                .with_compensation(compensate_step),
        );
        def.add_step(StepDefinition::new("ship-order", "PaymentCharged", success_step));
        def
    }

    fn make_trigger(event_type: &str) -> TriggerEvent {
        TriggerEvent::new(event_type, HashMap::new())
    }

    fn make_trigger_with_data(event_type: &str, key: &str, val: &str) -> TriggerEvent {
        let mut data = HashMap::new();
        data.insert(key.to_string(), val.to_string());
        TriggerEvent::new(event_type, data)
    }

    #[test]
    fn test_start_process() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        let pid = mgr.start_process("p1", "order-process", HashMap::new()).unwrap();
        assert_eq!(pid, "p1");

        let proc = mgr.get_process("p1").unwrap();
        assert_eq!(proc.status, ProcessStatus::Running);
        assert_eq!(proc.step_states.len(), 3);
    }

    #[test]
    fn test_duplicate_process() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();
        let err = mgr.start_process("p1", "order-process", HashMap::new()).unwrap_err();
        assert!(matches!(err, ProcessManagerError::ProcessAlreadyExists(_)));
    }

    #[test]
    fn test_missing_definition() {
        let mut mgr = ProcessManager::new();
        let err = mgr.start_process("p1", "missing", HashMap::new()).unwrap_err();
        assert!(matches!(err, ProcessManagerError::DefinitionNotFound(_)));
    }

    #[test]
    fn test_full_process_completion() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();

        let status = mgr.handle_event("p1", &make_trigger("OrderPlaced")).unwrap();
        assert_eq!(status, ProcessStatus::Running);

        let status = mgr.handle_event("p1", &make_trigger("StockReserved")).unwrap();
        assert_eq!(status, ProcessStatus::Running);

        let status = mgr.handle_event("p1", &make_trigger("PaymentCharged")).unwrap();
        assert_eq!(status, ProcessStatus::Completed);

        let proc = mgr.get_process("p1").unwrap();
        assert!(proc.completed_at.is_some());
        assert_eq!(proc.completed_steps(), 3);
    }

    #[test]
    fn test_wrong_event_ignored() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();

        let status = mgr.handle_event("p1", &make_trigger("WrongEvent")).unwrap();
        assert_eq!(status, ProcessStatus::Running);
        assert_eq!(mgr.get_process("p1").unwrap().current_step, 0);
    }

    #[test]
    fn test_step_failure() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("fail-process");
        def.add_step(StepDefinition::new("s1", "E1", fail_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "fail-process", HashMap::new()).unwrap();
        let status = mgr.handle_event("p1", &make_trigger("E1")).unwrap();
        assert_eq!(status, ProcessStatus::Failed);

        let proc = mgr.get_process("p1").unwrap();
        assert!(proc.error_message.is_some());
        assert_eq!(proc.step_states[0].status, StepStatus::Failed);
    }

    #[test]
    fn test_compensation_success() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("comp-process");
        def.add_step(
            StepDefinition::new("s1", "E1", success_step)
                .with_compensation(compensate_step),
        );
        def.add_step(StepDefinition::new("s2", "E2", fail_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "comp-process", HashMap::new()).unwrap();
        mgr.handle_event("p1", &make_trigger("E1")).unwrap();
        mgr.handle_event("p1", &make_trigger("E2")).unwrap(); // Fails.

        let status = mgr.compensate("p1").unwrap();
        assert_eq!(status, ProcessStatus::CompensationCompleted);

        let proc = mgr.get_process("p1").unwrap();
        assert_eq!(proc.state.get("compensated").map(|s| s.as_str()), Some("true"));
    }

    #[test]
    fn test_compensation_failure() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("fail-comp");
        def.add_step(
            StepDefinition::new("s1", "E1", success_step)
                .with_compensation(fail_compensate),
        );
        def.add_step(StepDefinition::new("s2", "E2", fail_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "fail-comp", HashMap::new()).unwrap();
        mgr.handle_event("p1", &make_trigger("E1")).unwrap();
        mgr.handle_event("p1", &make_trigger("E2")).unwrap();

        let status = mgr.compensate("p1").unwrap();
        assert_eq!(status, ProcessStatus::CompensationFailed);
    }

    #[test]
    fn test_compensate_requires_failed_state() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();

        let err = mgr.compensate("p1").unwrap_err();
        assert!(matches!(err, ProcessManagerError::InvalidTransition { .. }));
    }

    #[test]
    fn test_timeout_step() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();

        mgr.timeout_step("p1").unwrap();
        let proc = mgr.get_process("p1").unwrap();
        assert_eq!(proc.status, ProcessStatus::TimedOut);
        assert_eq!(proc.step_states[0].status, StepStatus::TimedOut);
    }

    #[test]
    fn test_restart_after_failure() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("restart-test");
        def.add_step(StepDefinition::new("s1", "E1", fail_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "restart-test", HashMap::new()).unwrap();
        mgr.handle_event("p1", &make_trigger("E1")).unwrap();
        assert_eq!(mgr.get_process("p1").unwrap().status, ProcessStatus::Failed);

        mgr.restart("p1").unwrap();
        let proc = mgr.get_process("p1").unwrap();
        assert_eq!(proc.status, ProcessStatus::Running);
        assert_eq!(proc.restart_count, 1);
        assert_eq!(proc.step_states[0].status, StepStatus::Pending);
    }

    #[test]
    fn test_restart_after_timeout() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();
        mgr.timeout_step("p1").unwrap();

        mgr.restart("p1").unwrap();
        assert_eq!(mgr.get_process("p1").unwrap().status, ProcessStatus::Running);
    }

    #[test]
    fn test_restart_running_fails() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();

        let err = mgr.restart("p1").unwrap_err();
        assert!(matches!(err, ProcessManagerError::InvalidTransition { .. }));
    }

    #[test]
    fn test_event_data_passed_to_step() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("data-test");
        def.add_step(StepDefinition::new("s1", "E1", success_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "data-test", HashMap::new()).unwrap();
        mgr.handle_event("p1", &make_trigger_with_data("E1", "order_id", "42")).unwrap();

        let proc = mgr.get_process("p1").unwrap();
        assert_eq!(proc.state.get("order_id").map(|s| s.as_str()), Some("42"));
    }

    #[test]
    fn test_process_is_terminal() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("t");
        def.add_step(StepDefinition::new("s1", "E1", success_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "t", HashMap::new()).unwrap();
        assert!(!mgr.get_process("p1").unwrap().is_terminal());

        mgr.handle_event("p1", &make_trigger("E1")).unwrap();
        assert!(mgr.get_process("p1").unwrap().is_terminal());
    }

    #[test]
    fn test_completed_process_rejects_events() {
        let mut mgr = ProcessManager::new();
        let mut def = ProcessDefinition::new("t");
        def.add_step(StepDefinition::new("s1", "E1", success_step));
        mgr.register_definition(def);

        mgr.start_process("p1", "t", HashMap::new()).unwrap();
        mgr.handle_event("p1", &make_trigger("E1")).unwrap();

        let err = mgr.handle_event("p1", &make_trigger("E1")).unwrap_err();
        assert!(matches!(err, ProcessManagerError::AlreadyCompleted(_)));
    }

    #[test]
    fn test_process_ids_sorted() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("zulu", "order-process", HashMap::new()).unwrap();
        mgr.start_process("alpha", "order-process", HashMap::new()).unwrap();
        assert_eq!(mgr.process_ids(), vec!["alpha", "zulu"]);
    }

    #[test]
    fn test_count_by_status() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();
        mgr.start_process("p2", "order-process", HashMap::new()).unwrap();
        assert_eq!(mgr.count_by_status(ProcessStatus::Running), 2);
    }

    #[test]
    fn test_step_definition_timeout() {
        let step = StepDefinition::new("s1", "E1", success_step).with_timeout(5000);
        assert_eq!(step.timeout_ms, 5000);
    }

    #[test]
    fn test_pending_steps_count() {
        let mut mgr = ProcessManager::new();
        mgr.register_definition(make_def());
        mgr.start_process("p1", "order-process", HashMap::new()).unwrap();

        assert_eq!(mgr.get_process("p1").unwrap().pending_steps(), 3);
        mgr.handle_event("p1", &make_trigger("OrderPlaced")).unwrap();
        assert_eq!(mgr.get_process("p1").unwrap().pending_steps(), 2);
    }
}
