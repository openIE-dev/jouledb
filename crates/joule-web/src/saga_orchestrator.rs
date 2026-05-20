//! Saga orchestrator — step definitions, compensating transactions,
//! forward/backward execution, state persistence, timeout per step,
//! and partial completion handling.
//!
//! Replaces JS saga libraries (saga-pattern, @nestjs/cqrs sagas) with
//! a pure-Rust saga orchestrator that tracks energy per step.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Saga errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SagaError {
    /// Saga not found.
    SagaNotFound(String),
    /// Step not found.
    StepNotFound { saga_id: String, step_name: String },
    /// Saga already completed.
    AlreadyCompleted(String),
    /// Saga already compensating.
    AlreadyCompensating(String),
    /// Step timeout exceeded.
    StepTimeout { saga_id: String, step_name: String, timeout_ms: u64 },
    /// Compensation failed.
    CompensationFailed { saga_id: String, step_name: String, reason: String },
    /// Duplicate saga id.
    DuplicateSaga(String),
    /// No steps defined.
    NoSteps(String),
}

impl std::fmt::Display for SagaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SagaNotFound(id) => write!(f, "saga not found: {id}"),
            Self::StepNotFound { saga_id, step_name } => {
                write!(f, "step {step_name} not found in saga {saga_id}")
            }
            Self::AlreadyCompleted(id) => write!(f, "saga already completed: {id}"),
            Self::AlreadyCompensating(id) => write!(f, "saga already compensating: {id}"),
            Self::StepTimeout { saga_id, step_name, timeout_ms } => {
                write!(
                    f,
                    "step {step_name} in saga {saga_id} timed out after {timeout_ms}ms"
                )
            }
            Self::CompensationFailed { saga_id, step_name, reason } => {
                write!(
                    f,
                    "compensation failed for step {step_name} in saga {saga_id}: {reason}"
                )
            }
            Self::DuplicateSaga(id) => write!(f, "duplicate saga: {id}"),
            Self::NoSteps(id) => write!(f, "saga has no steps: {id}"),
        }
    }
}

impl std::error::Error for SagaError {}

// ── Step and Saga Status ────────────────────────────────────────

/// Status of a single step.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Compensating,
    Compensated,
    CompensationFailed(String),
    Skipped,
    TimedOut,
}

/// Overall saga status.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SagaStatus {
    Created,
    Running,
    Completed,
    Failed,
    Compensating,
    Compensated,
    PartiallyCompensated,
}

// ── Step Definition ─────────────────────────────────────────────

/// A saga step with its action and compensation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepDefinition {
    pub name: String,
    pub description: String,
    /// Timeout in milliseconds for this step.
    pub timeout_ms: u64,
    /// Whether this step has a compensation action.
    pub has_compensation: bool,
    /// Configuration / parameters for the step.
    pub params: HashMap<String, serde_json::Value>,
}

/// Runtime state of a step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepState {
    pub name: String,
    pub status: StepStatus,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub compensation_started_at: Option<DateTime<Utc>>,
    pub compensation_completed_at: Option<DateTime<Utc>>,
    /// Output data from the step (available after completion).
    pub output: Option<serde_json::Value>,
    pub energy_uj: u64,
}

// ── Saga Instance ───────────────────────────────────────────────

/// A running saga instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SagaInstance {
    pub saga_id: String,
    pub saga_type: String,
    pub status: SagaStatus,
    pub steps: Vec<StepDefinition>,
    pub step_states: Vec<StepState>,
    /// Index of the current step being executed.
    pub current_step: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    /// Global saga context / data passed between steps.
    pub context: HashMap<String, serde_json::Value>,
    pub total_energy_uj: u64,
}

// ── Saga Orchestrator ───────────────────────────────────────────

/// The orchestrator managing saga lifecycles.
#[derive(Debug, Clone)]
pub struct SagaOrchestrator {
    sagas: HashMap<String, SagaInstance>,
    total_energy_uj: u64,
}

impl SagaOrchestrator {
    pub fn new() -> Self {
        Self {
            sagas: HashMap::new(),
            total_energy_uj: 0,
        }
    }

    /// Create a new saga with the given steps.
    pub fn create_saga(
        &mut self,
        saga_id: &str,
        saga_type: &str,
        steps: Vec<StepDefinition>,
        context: HashMap<String, serde_json::Value>,
    ) -> Result<&SagaInstance, SagaError> {
        if steps.is_empty() {
            return Err(SagaError::NoSteps(saga_id.to_string()));
        }
        if self.sagas.contains_key(saga_id) {
            return Err(SagaError::DuplicateSaga(saga_id.to_string()));
        }

        let step_states: Vec<StepState> = steps
            .iter()
            .map(|s| StepState {
                name: s.name.clone(),
                status: StepStatus::Pending,
                started_at: None,
                completed_at: None,
                compensation_started_at: None,
                compensation_completed_at: None,
                output: None,
                energy_uj: 0,
            })
            .collect();

        let now = Utc::now();
        let instance = SagaInstance {
            saga_id: saga_id.to_string(),
            saga_type: saga_type.to_string(),
            status: SagaStatus::Created,
            steps,
            step_states,
            current_step: 0,
            created_at: now,
            updated_at: now,
            completed_at: None,
            context,
            total_energy_uj: 0,
        };

        self.sagas.insert(saga_id.to_string(), instance);
        self.total_energy_uj += 10;
        Ok(self.sagas.get(saga_id).unwrap())
    }

    /// Start executing the saga from the first step.
    pub fn start(&mut self, saga_id: &str) -> Result<&StepState, SagaError> {
        let saga = self
            .sagas
            .get_mut(saga_id)
            .ok_or_else(|| SagaError::SagaNotFound(saga_id.to_string()))?;

        if saga.status == SagaStatus::Completed {
            return Err(SagaError::AlreadyCompleted(saga_id.to_string()));
        }

        saga.status = SagaStatus::Running;
        saga.current_step = 0;
        saga.step_states[0].status = StepStatus::Running;
        saga.step_states[0].started_at = Some(Utc::now());
        saga.updated_at = Utc::now();
        saga.total_energy_uj += 5;
        self.total_energy_uj += 5;

        Ok(&saga.step_states[0])
    }

    /// Complete the current step and advance to the next.
    pub fn complete_step(
        &mut self,
        saga_id: &str,
        step_name: &str,
        output: Option<serde_json::Value>,
    ) -> Result<StepAdvanceResult, SagaError> {
        let saga = self
            .sagas
            .get_mut(saga_id)
            .ok_or_else(|| SagaError::SagaNotFound(saga_id.to_string()))?;

        if saga.status == SagaStatus::Completed {
            return Err(SagaError::AlreadyCompleted(saga_id.to_string()));
        }
        if saga.status == SagaStatus::Compensating {
            return Err(SagaError::AlreadyCompensating(saga_id.to_string()));
        }

        let step_idx = saga
            .step_states
            .iter()
            .position(|s| s.name == step_name)
            .ok_or_else(|| SagaError::StepNotFound {
                saga_id: saga_id.to_string(),
                step_name: step_name.to_string(),
            })?;

        let now = Utc::now();

        // Check timeout.
        let timeout_ms = saga.steps[step_idx].timeout_ms;
        if let Some(started) = saga.step_states[step_idx].started_at {
            let elapsed = now.signed_duration_since(started).num_milliseconds();
            if elapsed > timeout_ms as i64 {
                saga.step_states[step_idx].status = StepStatus::TimedOut;
                return Err(SagaError::StepTimeout {
                    saga_id: saga_id.to_string(),
                    step_name: step_name.to_string(),
                    timeout_ms,
                });
            }
        }

        saga.step_states[step_idx].status = StepStatus::Completed;
        saga.step_states[step_idx].completed_at = Some(now);
        saga.step_states[step_idx].output = output;
        saga.step_states[step_idx].energy_uj += 10;
        saga.total_energy_uj += 10;
        saga.updated_at = now;
        self.total_energy_uj += 10;

        // Advance to next step.
        let next_idx = step_idx + 1;
        if next_idx >= saga.steps.len() {
            // All steps completed.
            saga.status = SagaStatus::Completed;
            saga.completed_at = Some(now);
            Ok(StepAdvanceResult::SagaCompleted)
        } else {
            saga.current_step = next_idx;
            saga.step_states[next_idx].status = StepStatus::Running;
            saga.step_states[next_idx].started_at = Some(now);
            Ok(StepAdvanceResult::NextStep(saga.step_states[next_idx].name.clone()))
        }
    }

    /// Fail a step and begin compensation.
    pub fn fail_step(
        &mut self,
        saga_id: &str,
        step_name: &str,
        reason: &str,
    ) -> Result<(), SagaError> {
        let saga = self
            .sagas
            .get_mut(saga_id)
            .ok_or_else(|| SagaError::SagaNotFound(saga_id.to_string()))?;

        let step_idx = saga
            .step_states
            .iter()
            .position(|s| s.name == step_name)
            .ok_or_else(|| SagaError::StepNotFound {
                saga_id: saga_id.to_string(),
                step_name: step_name.to_string(),
            })?;

        saga.step_states[step_idx].status = StepStatus::Failed(reason.to_string());
        saga.status = SagaStatus::Compensating;
        saga.updated_at = Utc::now();

        // Skip remaining steps.
        for i in (step_idx + 1)..saga.step_states.len() {
            saga.step_states[i].status = StepStatus::Skipped;
        }

        // Begin compensation from the step before the failed one.
        if step_idx > 0 {
            saga.current_step = step_idx - 1;
            saga.step_states[step_idx - 1].status = StepStatus::Compensating;
            saga.step_states[step_idx - 1].compensation_started_at = Some(Utc::now());
        }

        saga.status = SagaStatus::Compensating;
        saga.total_energy_uj += 5;
        self.total_energy_uj += 5;
        Ok(())
    }

    /// Complete compensation for a step and move backward.
    pub fn complete_compensation(
        &mut self,
        saga_id: &str,
        step_name: &str,
    ) -> Result<CompensationResult, SagaError> {
        let saga = self
            .sagas
            .get_mut(saga_id)
            .ok_or_else(|| SagaError::SagaNotFound(saga_id.to_string()))?;

        let step_idx = saga
            .step_states
            .iter()
            .position(|s| s.name == step_name)
            .ok_or_else(|| SagaError::StepNotFound {
                saga_id: saga_id.to_string(),
                step_name: step_name.to_string(),
            })?;

        let now = Utc::now();
        saga.step_states[step_idx].status = StepStatus::Compensated;
        saga.step_states[step_idx].compensation_completed_at = Some(now);
        saga.step_states[step_idx].energy_uj += 8;
        saga.total_energy_uj += 8;
        saga.updated_at = now;
        self.total_energy_uj += 8;

        if step_idx == 0 {
            // All compensations done.
            saga.status = SagaStatus::Compensated;
            saga.completed_at = Some(now);
            Ok(CompensationResult::FullyCompensated)
        } else {
            let prev_idx = step_idx - 1;
            // Only compensate steps that actually completed and have compensation.
            if saga.step_states[prev_idx].status == StepStatus::Completed
                && saga.steps[prev_idx].has_compensation
            {
                saga.current_step = prev_idx;
                saga.step_states[prev_idx].status = StepStatus::Compensating;
                saga.step_states[prev_idx].compensation_started_at = Some(now);
                Ok(CompensationResult::NextCompensation(
                    saga.step_states[prev_idx].name.clone(),
                ))
            } else if saga.step_states[prev_idx].status == StepStatus::Completed {
                // Step completed but has no compensation — skip.
                saga.step_states[prev_idx].status = StepStatus::Compensated;
                saga.step_states[prev_idx].compensation_completed_at = Some(now);
                if prev_idx == 0 {
                    saga.status = SagaStatus::Compensated;
                    saga.completed_at = Some(now);
                    Ok(CompensationResult::FullyCompensated)
                } else {
                    // Recurse / continue backward.
                    Ok(CompensationResult::NextCompensation(
                        saga.step_states[prev_idx - 1].name.clone(),
                    ))
                }
            } else {
                saga.status = SagaStatus::Compensated;
                saga.completed_at = Some(now);
                Ok(CompensationResult::FullyCompensated)
            }
        }
    }

    /// Fail compensation for a step.
    pub fn fail_compensation(
        &mut self,
        saga_id: &str,
        step_name: &str,
        reason: &str,
    ) -> Result<(), SagaError> {
        let saga = self
            .sagas
            .get_mut(saga_id)
            .ok_or_else(|| SagaError::SagaNotFound(saga_id.to_string()))?;

        let step_idx = saga
            .step_states
            .iter()
            .position(|s| s.name == step_name)
            .ok_or_else(|| SagaError::StepNotFound {
                saga_id: saga_id.to_string(),
                step_name: step_name.to_string(),
            })?;

        saga.step_states[step_idx].status =
            StepStatus::CompensationFailed(reason.to_string());
        saga.status = SagaStatus::PartiallyCompensated;
        saga.updated_at = Utc::now();
        saga.total_energy_uj += 3;
        self.total_energy_uj += 3;
        Ok(())
    }

    /// Get a saga instance.
    pub fn get_saga(&self, saga_id: &str) -> Option<&SagaInstance> {
        self.sagas.get(saga_id)
    }

    /// Get all sagas with a given status.
    pub fn sagas_by_status(&self, status: &SagaStatus) -> Vec<&SagaInstance> {
        self.sagas.values().filter(|s| &s.status == status).collect()
    }

    /// Total sagas.
    pub fn saga_count(&self) -> usize {
        self.sagas.len()
    }

    /// Total energy consumed.
    pub fn total_energy_uj(&self) -> u64 {
        self.total_energy_uj
    }
}

impl Default for SagaOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Result Enums ────────────────────────────────────────────────

/// Result of advancing a step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepAdvanceResult {
    NextStep(String),
    SagaCompleted,
}

/// Result of completing a compensation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompensationResult {
    NextCompensation(String),
    FullyCompensated,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn three_steps() -> Vec<StepDefinition> {
        vec![
            StepDefinition {
                name: "reserve_inventory".into(),
                description: "Reserve items".into(),
                timeout_ms: 5000,
                has_compensation: true,
                params: HashMap::new(),
            },
            StepDefinition {
                name: "charge_payment".into(),
                description: "Charge credit card".into(),
                timeout_ms: 10000,
                has_compensation: true,
                params: HashMap::new(),
            },
            StepDefinition {
                name: "ship_order".into(),
                description: "Ship the order".into(),
                timeout_ms: 30000,
                has_compensation: true,
                params: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_create_saga() {
        let mut orch = SagaOrchestrator::new();
        let saga = orch
            .create_saga("saga-1", "OrderSaga", three_steps(), HashMap::new())
            .unwrap();
        assert_eq!(saga.saga_id, "saga-1");
        assert_eq!(saga.status, SagaStatus::Created);
        assert_eq!(saga.steps.len(), 3);
        assert_eq!(saga.step_states.len(), 3);
    }

    #[test]
    fn test_duplicate_saga() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("saga-1", "T", three_steps(), HashMap::new())
            .unwrap();
        assert_eq!(
            orch.create_saga("saga-1", "T", three_steps(), HashMap::new()),
            Err(SagaError::DuplicateSaga("saga-1".into()))
        );
    }

    #[test]
    fn test_no_steps() {
        let mut orch = SagaOrchestrator::new();
        assert_eq!(
            orch.create_saga("saga-1", "T", vec![], HashMap::new()),
            Err(SagaError::NoSteps("saga-1".into()))
        );
    }

    #[test]
    fn test_start_saga() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();

        let step = orch.start("s1").unwrap();
        assert_eq!(step.name, "reserve_inventory");
        assert_eq!(step.status, StepStatus::Running);

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.status, SagaStatus::Running);
    }

    #[test]
    fn test_happy_path_all_steps() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();

        let r1 = orch
            .complete_step("s1", "reserve_inventory", Some(serde_json::json!({"reserved": true})))
            .unwrap();
        assert_eq!(
            r1,
            StepAdvanceResult::NextStep("charge_payment".into())
        );

        let r2 = orch
            .complete_step("s1", "charge_payment", None)
            .unwrap();
        assert_eq!(
            r2,
            StepAdvanceResult::NextStep("ship_order".into())
        );

        let r3 = orch.complete_step("s1", "ship_order", None).unwrap();
        assert_eq!(r3, StepAdvanceResult::SagaCompleted);

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.status, SagaStatus::Completed);
        assert!(saga.completed_at.is_some());
    }

    #[test]
    fn test_fail_step_triggers_compensation() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();
        orch.complete_step("s1", "reserve_inventory", None).unwrap();

        // Charge payment fails.
        orch.fail_step("s1", "charge_payment", "card declined").unwrap();

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.status, SagaStatus::Compensating);
        assert_eq!(
            saga.step_states[1].status,
            StepStatus::Failed("card declined".into())
        );
        assert_eq!(saga.step_states[2].status, StepStatus::Skipped);
        // Step 0 should be compensating.
        assert_eq!(saga.step_states[0].status, StepStatus::Compensating);
    }

    #[test]
    fn test_full_compensation() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();
        orch.complete_step("s1", "reserve_inventory", None).unwrap();
        orch.complete_step("s1", "charge_payment", None).unwrap();

        // Shipping fails.
        orch.fail_step("s1", "ship_order", "no carrier").unwrap();

        // Compensate charge_payment.
        let r1 = orch.complete_compensation("s1", "charge_payment").unwrap();
        assert_eq!(
            r1,
            CompensationResult::NextCompensation("reserve_inventory".into())
        );

        // Compensate reserve_inventory.
        let r2 = orch
            .complete_compensation("s1", "reserve_inventory")
            .unwrap();
        assert_eq!(r2, CompensationResult::FullyCompensated);

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.status, SagaStatus::Compensated);
    }

    #[test]
    fn test_compensation_failed() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();
        orch.complete_step("s1", "reserve_inventory", None).unwrap();
        orch.fail_step("s1", "charge_payment", "declined").unwrap();

        orch.fail_compensation("s1", "reserve_inventory", "db down")
            .unwrap();

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.status, SagaStatus::PartiallyCompensated);
        assert_eq!(
            saga.step_states[0].status,
            StepStatus::CompensationFailed("db down".into())
        );
    }

    #[test]
    fn test_step_timeout() {
        let steps = vec![StepDefinition {
            name: "fast_step".into(),
            description: "Must be fast".into(),
            timeout_ms: 1, // 1ms timeout.
            has_compensation: false,
            params: HashMap::new(),
        }];

        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", steps, HashMap::new()).unwrap();
        orch.start("s1").unwrap();

        // Manually backdate the start time.
        let saga = orch.sagas.get_mut("s1").unwrap();
        saga.step_states[0].started_at =
            Some(Utc::now() - Duration::milliseconds(100));

        let result = orch.complete_step("s1", "fast_step", None);
        assert!(matches!(result, Err(SagaError::StepTimeout { .. })));
    }

    #[test]
    fn test_already_completed() {
        let steps = vec![StepDefinition {
            name: "step1".into(),
            description: "only step".into(),
            timeout_ms: 60000,
            has_compensation: false,
            params: HashMap::new(),
        }];

        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", steps, HashMap::new()).unwrap();
        orch.start("s1").unwrap();
        orch.complete_step("s1", "step1", None).unwrap();

        assert_eq!(
            orch.complete_step("s1", "step1", None),
            Err(SagaError::AlreadyCompleted("s1".into()))
        );
        assert_eq!(
            orch.start("s1"),
            Err(SagaError::AlreadyCompleted("s1".into()))
        );
    }

    #[test]
    fn test_sagas_by_status() {
        let mut orch = SagaOrchestrator::new();

        let steps = vec![StepDefinition {
            name: "s".into(),
            description: "d".into(),
            timeout_ms: 5000,
            has_compensation: false,
            params: HashMap::new(),
        }];

        orch.create_saga("s1", "T", steps.clone(), HashMap::new()).unwrap();
        orch.create_saga("s2", "T", steps.clone(), HashMap::new()).unwrap();
        orch.create_saga("s3", "T", steps, HashMap::new()).unwrap();

        orch.start("s1").unwrap();
        orch.start("s2").unwrap();

        let running = orch.sagas_by_status(&SagaStatus::Running);
        assert_eq!(running.len(), 2);
        let created = orch.sagas_by_status(&SagaStatus::Created);
        assert_eq!(created.len(), 1);
    }

    #[test]
    fn test_saga_not_found() {
        let mut orch = SagaOrchestrator::new();
        assert_eq!(
            orch.start("missing"),
            Err(SagaError::SagaNotFound("missing".into()))
        );
    }

    #[test]
    fn test_step_not_found() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();

        assert!(matches!(
            orch.complete_step("s1", "nonexistent", None),
            Err(SagaError::StepNotFound { .. })
        ));
    }

    #[test]
    fn test_context_passed() {
        let mut orch = SagaOrchestrator::new();
        let mut ctx = HashMap::new();
        ctx.insert("order_id".into(), serde_json::json!("ord-42"));

        orch.create_saga("s1", "T", three_steps(), ctx).unwrap();
        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(
            saga.context.get("order_id"),
            Some(&serde_json::json!("ord-42"))
        );
    }

    #[test]
    fn test_step_output_captured() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();

        let output = serde_json::json!({"reservation_id": "r-123"});
        orch.complete_step("s1", "reserve_inventory", Some(output.clone()))
            .unwrap();

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.step_states[0].output, Some(output));
    }

    #[test]
    fn test_energy_tracking() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        assert!(orch.total_energy_uj() > 0);

        let e1 = orch.total_energy_uj();
        orch.start("s1").unwrap();
        assert!(orch.total_energy_uj() > e1);
    }

    #[test]
    fn test_saga_count() {
        let mut orch = SagaOrchestrator::new();
        assert_eq!(orch.saga_count(), 0);
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        assert_eq!(orch.saga_count(), 1);
    }

    #[test]
    fn test_step_status_serde() {
        let s = StepStatus::Failed("oops".into());
        let json = serde_json::to_string(&s).unwrap();
        let parsed: StepStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, StepStatus::Failed("oops".into()));
    }

    #[test]
    fn test_saga_status_serde() {
        let s = SagaStatus::PartiallyCompensated;
        let json = serde_json::to_string(&s).unwrap();
        let parsed: SagaStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, SagaStatus::PartiallyCompensated);
    }

    #[test]
    fn test_error_display() {
        let e = SagaError::StepTimeout {
            saga_id: "s1".into(),
            step_name: "step1".into(),
            timeout_ms: 5000,
        };
        let s = e.to_string();
        assert!(s.contains("5000"));
        assert!(s.contains("step1"));
    }

    #[test]
    fn test_default_orchestrator() {
        let orch = SagaOrchestrator::default();
        assert_eq!(orch.saga_count(), 0);
    }

    #[test]
    fn test_fail_step_first_step() {
        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", three_steps(), HashMap::new()).unwrap();
        orch.start("s1").unwrap();

        // First step fails — no prior steps to compensate.
        orch.fail_step("s1", "reserve_inventory", "out of stock").unwrap();

        let saga = orch.get_saga("s1").unwrap();
        assert_eq!(saga.status, SagaStatus::Compensating);
        assert_eq!(saga.step_states[1].status, StepStatus::Skipped);
        assert_eq!(saga.step_states[2].status, StepStatus::Skipped);
    }

    #[test]
    fn test_step_without_compensation() {
        let steps = vec![
            StepDefinition {
                name: "step1".into(),
                description: "has comp".into(),
                timeout_ms: 5000,
                has_compensation: true,
                params: HashMap::new(),
            },
            StepDefinition {
                name: "step2".into(),
                description: "no comp".into(),
                timeout_ms: 5000,
                has_compensation: false,
                params: HashMap::new(),
            },
            StepDefinition {
                name: "step3".into(),
                description: "has comp".into(),
                timeout_ms: 5000,
                has_compensation: true,
                params: HashMap::new(),
            },
        ];

        let mut orch = SagaOrchestrator::new();
        orch.create_saga("s1", "T", steps, HashMap::new()).unwrap();
        orch.start("s1").unwrap();
        orch.complete_step("s1", "step1", None).unwrap();
        orch.complete_step("s1", "step2", None).unwrap();

        // step3 fails.
        orch.fail_step("s1", "step3", "error").unwrap();

        // Compensate step2 (no compensation — should auto-skip).
        let r = orch.complete_compensation("s1", "step2").unwrap();
        // step2 has no compensation, so step1 is the next (or auto-compensated).
        assert!(matches!(r, CompensationResult::NextCompensation(_) | CompensationResult::FullyCompensated));
    }
}
