//! Compensating transactions — action registration, forward/compensation pairs,
//! compensation execution on failure, partial compensation, compensation log,
//! idempotent compensation, and compensation ordering.
//!
//! Pure Rust implementation of the compensating transaction pattern for
//! maintaining consistency when distributed operations fail mid-way.

use uuid::Uuid;

// ── Compensation Error ──────────────────────────────────────────

/// Errors from the compensation system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompensationError {
    /// The forward action failed.
    ForwardFailed { step: String, reason: String },
    /// A compensation action failed.
    CompensationFailed { step: String, reason: String },
    /// Step not found.
    StepNotFound(String),
    /// Transaction already completed.
    AlreadyCompleted(String),
    /// Transaction already compensated.
    AlreadyCompensated(String),
    /// Duplicate step name.
    DuplicateStep(String),
    /// Idempotency key collision.
    IdempotencyCollision { key: String },
}

impl std::fmt::Display for CompensationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardFailed { step, reason } => {
                write!(f, "forward action '{step}' failed: {reason}")
            }
            Self::CompensationFailed { step, reason } => {
                write!(f, "compensation for '{step}' failed: {reason}")
            }
            Self::StepNotFound(s) => write!(f, "step not found: {s}"),
            Self::AlreadyCompleted(id) => write!(f, "transaction {id} already completed"),
            Self::AlreadyCompensated(id) => write!(f, "transaction {id} already compensated"),
            Self::DuplicateStep(s) => write!(f, "duplicate step: {s}"),
            Self::IdempotencyCollision { key } => {
                write!(f, "idempotency key collision: {key}")
            }
        }
    }
}

impl std::error::Error for CompensationError {}

// ── Step Status ─────────────────────────────────────────────────

/// Status of a compensation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    ForwardExecuted,
    ForwardFailed,
    Compensated,
    CompensationFailed,
    Skipped,
}

impl StepStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::ForwardExecuted => "forward_executed",
            Self::ForwardFailed => "forward_failed",
            Self::Compensated => "compensated",
            Self::CompensationFailed => "compensation_failed",
            Self::Skipped => "skipped",
        }
    }

    pub fn needs_compensation(&self) -> bool {
        matches!(self, Self::ForwardExecuted)
    }
}

// ── Transaction Status ──────────────────────────────────────────

/// Status of the overall compensating transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionStatus {
    Building,
    InProgress,
    Completed,
    Compensating,
    Compensated,
    PartiallyCompensated,
    Failed,
}

impl TransactionStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Building => "building",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Compensating => "compensating",
            Self::Compensated => "compensated",
            Self::PartiallyCompensated => "partially_compensated",
            Self::Failed => "failed",
        }
    }
}

// ── Log Entry ───────────────────────────────────────────────────

/// An entry in the compensation log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub step_name: String,
    pub action: LogAction,
    pub success: bool,
    pub detail: String,
    pub sequence: u64,
}

/// What action was logged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogAction {
    ForwardExecuted,
    ForwardFailed,
    CompensationExecuted,
    CompensationFailed,
    CompensationSkipped,
}

impl LogAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ForwardExecuted => "forward_executed",
            Self::ForwardFailed => "forward_failed",
            Self::CompensationExecuted => "compensation_executed",
            Self::CompensationFailed => "compensation_failed",
            Self::CompensationSkipped => "compensation_skipped",
        }
    }
}

// ── Compensation Step ───────────────────────────────────────────

/// A single forward/compensation pair.
pub struct CompensationStep {
    name: String,
    status: StepStatus,
    forward: Box<dyn Fn() -> Result<String, String>>,
    compensate: Box<dyn Fn() -> Result<String, String>>,
    idempotency_key: Option<String>,
    forward_result: Option<String>,
}

impl std::fmt::Debug for CompensationStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompensationStep")
            .field("name", &self.name)
            .field("status", &self.status)
            .field("idempotency_key", &self.idempotency_key)
            .field("forward_result", &self.forward_result)
            .finish()
    }
}

impl CompensationStep {
    pub fn new(
        name: impl Into<String>,
        forward: impl Fn() -> Result<String, String> + 'static,
        compensate: impl Fn() -> Result<String, String> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            status: StepStatus::Pending,
            forward: Box::new(forward),
            compensate: Box::new(compensate),
            idempotency_key: None,
            forward_result: None,
        }
    }

    pub fn with_idempotency_key(mut self, key: impl Into<String>) -> Self {
        self.idempotency_key = Some(key.into());
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn status(&self) -> StepStatus {
        self.status
    }

    pub fn idempotency_key(&self) -> Option<&str> {
        self.idempotency_key.as_deref()
    }

    pub fn forward_result(&self) -> Option<&str> {
        self.forward_result.as_deref()
    }
}

// ── Compensating Transaction ────────────────────────────────────

/// A compensating transaction: a sequence of forward/compensation pairs
/// that are automatically compensated in reverse order on failure.
pub struct CompensatingTransaction {
    id: String,
    steps: Vec<CompensationStep>,
    status: TransactionStatus,
    log: Vec<LogEntry>,
    log_sequence: u64,
    idempotency_keys: Vec<String>,
}

impl std::fmt::Debug for CompensatingTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompensatingTransaction")
            .field("id", &self.id)
            .field("status", &self.status)
            .field("step_count", &self.steps.len())
            .field("log_entries", &self.log.len())
            .finish()
    }
}

impl CompensatingTransaction {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            steps: Vec::new(),
            status: TransactionStatus::Building,
            log: Vec::new(),
            log_sequence: 0,
            idempotency_keys: Vec::new(),
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Add a step to the transaction.
    pub fn add_step(&mut self, step: CompensationStep) -> Result<(), CompensationError> {
        // Check for duplicate names.
        let name = step.name.clone();
        if self.steps.iter().any(|s| s.name == name) {
            return Err(CompensationError::DuplicateStep(name));
        }

        // Check idempotency key uniqueness.
        if let Some(key) = &step.idempotency_key {
            if self.idempotency_keys.contains(key) {
                return Err(CompensationError::IdempotencyCollision {
                    key: key.clone(),
                });
            }
            self.idempotency_keys.push(key.clone());
        }

        self.steps.push(step);
        Ok(())
    }

    /// Execute all forward actions in order. On first failure,
    /// compensate all previously executed steps in reverse.
    pub fn execute(&mut self) -> Result<Vec<String>, CompensationError> {
        if self.status == TransactionStatus::Completed {
            return Err(CompensationError::AlreadyCompleted(self.id.clone()));
        }
        if self.status == TransactionStatus::Compensated {
            return Err(CompensationError::AlreadyCompensated(self.id.clone()));
        }

        self.status = TransactionStatus::InProgress;
        let mut results = Vec::new();
        let mut failed_at: Option<(usize, String, String)> = None;

        for i in 0..self.steps.len() {
            // Call the forward closure while borrowing only self.steps[i],
            // then release that borrow before mutating.
            let outcome = (self.steps[i].forward)();
            match outcome {
                Ok(result) => {
                    let step_name = self.steps[i].name.clone();
                    self.steps[i].status = StepStatus::ForwardExecuted;
                    self.steps[i].forward_result = Some(result.clone());
                    self.append_log(&step_name, LogAction::ForwardExecuted, true, &result);
                    results.push(result);
                }
                Err(reason) => {
                    let step_name = self.steps[i].name.clone();
                    self.steps[i].status = StepStatus::ForwardFailed;
                    self.append_log(&step_name, LogAction::ForwardFailed, false, &reason);
                    failed_at = Some((i, step_name, reason));
                    break;
                }
            }
        }

        if let Some((failed_idx, step_name, reason)) = failed_at {
            // Mark remaining steps as skipped.
            for j in (failed_idx + 1)..self.steps.len() {
                self.steps[j].status = StepStatus::Skipped;
            }

            // Compensate in reverse order.
            self.compensate_from(failed_idx);

            return Err(CompensationError::ForwardFailed {
                step: step_name,
                reason,
            });
        }

        self.status = TransactionStatus::Completed;
        Ok(results)
    }

    /// Compensate all executed steps from index `from` (exclusive) down to 0.
    fn compensate_from(&mut self, from: usize) {
        self.status = TransactionStatus::Compensating;
        let mut all_compensated = true;

        // Compensate in reverse order, only steps that were executed.
        let indices: Vec<usize> = (0..from).rev().collect();
        for i in indices {
            if !self.steps[i].status.needs_compensation() {
                continue;
            }
            let outcome = (self.steps[i].compensate)();
            match outcome {
                Ok(detail) => {
                    let step_name = self.steps[i].name.clone();
                    self.steps[i].status = StepStatus::Compensated;
                    self.append_log(&step_name, LogAction::CompensationExecuted, true, &detail);
                }
                Err(reason) => {
                    let step_name = self.steps[i].name.clone();
                    self.steps[i].status = StepStatus::CompensationFailed;
                    self.append_log(&step_name, LogAction::CompensationFailed, false, &reason);
                    all_compensated = false;
                }
            }
        }

        self.status = if all_compensated {
            TransactionStatus::Compensated
        } else {
            TransactionStatus::PartiallyCompensated
        };
    }

    /// Manually trigger compensation for all executed steps.
    pub fn compensate_all(&mut self) -> Result<(), CompensationError> {
        if self.status == TransactionStatus::Compensated {
            return Err(CompensationError::AlreadyCompensated(self.id.clone()));
        }

        self.status = TransactionStatus::Compensating;
        let mut all_compensated = true;

        let indices: Vec<usize> = (0..self.steps.len()).rev().collect();
        for i in indices {
            if !self.steps[i].status.needs_compensation() {
                let step_name = self.steps[i].name.clone();
                self.append_log(
                    &step_name,
                    LogAction::CompensationSkipped,
                    true,
                    "not executed",
                );
                continue;
            }
            let outcome = (self.steps[i].compensate)();
            match outcome {
                Ok(detail) => {
                    let step_name = self.steps[i].name.clone();
                    self.steps[i].status = StepStatus::Compensated;
                    self.append_log(&step_name, LogAction::CompensationExecuted, true, &detail);
                }
                Err(reason) => {
                    let step_name = self.steps[i].name.clone();
                    self.steps[i].status = StepStatus::CompensationFailed;
                    self.append_log(&step_name, LogAction::CompensationFailed, false, &reason);
                    all_compensated = false;
                }
            }
        }

        self.status = if all_compensated {
            TransactionStatus::Compensated
        } else {
            TransactionStatus::PartiallyCompensated
        };

        Ok(())
    }

    fn append_log(&mut self, step_name: &str, action: LogAction, success: bool, detail: &str) {
        self.log_sequence += 1;
        self.log.push(LogEntry {
            step_name: step_name.to_string(),
            action,
            success,
            detail: detail.to_string(),
            sequence: self.log_sequence,
        });
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn status(&self) -> TransactionStatus {
        self.status
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }

    pub fn steps(&self) -> impl Iterator<Item = (&str, StepStatus)> {
        self.steps.iter().map(|s| (s.name.as_str(), s.status))
    }

    /// Count of steps with a given status.
    pub fn count_with_status(&self, status: StepStatus) -> usize {
        self.steps.iter().filter(|s| s.status == status).count()
    }

    /// Get a step's status by name.
    pub fn step_status(&self, name: &str) -> Option<StepStatus> {
        self.steps.iter().find(|s| s.name == name).map(|s| s.status)
    }

    /// Get a step's forward result by name.
    pub fn step_result(&self, name: &str) -> Option<&str> {
        self.steps
            .iter()
            .find(|s| s.name == name)
            .and_then(|s| s.forward_result.as_deref())
    }
}

impl Default for CompensatingTransaction {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_all_succeed() {
        let mut tx = CompensatingTransaction::new().with_id("tx1");
        tx.add_step(CompensationStep::new(
            "step1",
            || Ok("created user".to_string()),
            || Ok("deleted user".to_string()),
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "step2",
            || Ok("created order".to_string()),
            || Ok("cancelled order".to_string()),
        ))
        .unwrap();
        let results = tx.execute().unwrap();
        assert_eq!(results, vec!["created user", "created order"]);
        assert_eq!(tx.status(), TransactionStatus::Completed);
    }

    #[test]
    fn test_second_step_fails_compensates_first() {
        let compensated = Rc::new(RefCell::new(Vec::new()));
        let comp_clone = compensated.clone();

        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "step1",
            || Ok("done".to_string()),
            move || {
                comp_clone.borrow_mut().push("step1");
                Ok("compensated".to_string())
            },
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "step2",
            || Err("boom".to_string()),
            || Ok("compensated".to_string()),
        ))
        .unwrap();

        let result = tx.execute();
        assert!(result.is_err());
        assert_eq!(compensated.borrow().as_slice(), &["step1"]);
        assert_eq!(tx.step_status("step1"), Some(StepStatus::Compensated));
        assert_eq!(tx.step_status("step2"), Some(StepStatus::ForwardFailed));
    }

    #[test]
    fn test_first_step_fails_no_compensation() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "step1",
            || Err("fail".to_string()),
            || Ok("compensated".to_string()),
        ))
        .unwrap();
        let result = tx.execute();
        assert!(result.is_err());
        assert_eq!(tx.step_status("step1"), Some(StepStatus::ForwardFailed));
    }

    #[test]
    fn test_compensation_failure_partial() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "step1",
            || Ok("ok".to_string()),
            || Err("comp fail".to_string()), // Compensation fails.
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "step2",
            || Err("forward fail".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();

        let result = tx.execute();
        assert!(result.is_err());
        assert_eq!(tx.status(), TransactionStatus::PartiallyCompensated);
        assert_eq!(
            tx.step_status("step1"),
            Some(StepStatus::CompensationFailed)
        );
    }

    #[test]
    fn test_duplicate_step() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        let result = tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ));
        assert!(matches!(result, Err(CompensationError::DuplicateStep(_))));
    }

    #[test]
    fn test_idempotency_key() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(
            CompensationStep::new(
                "s1",
                || Ok("ok".to_string()),
                || Ok("ok".to_string()),
            )
            .with_idempotency_key("key1"),
        )
        .unwrap();
        let result = tx.add_step(
            CompensationStep::new(
                "s2",
                || Ok("ok".to_string()),
                || Ok("ok".to_string()),
            )
            .with_idempotency_key("key1"),
        );
        assert!(matches!(
            result,
            Err(CompensationError::IdempotencyCollision { .. })
        ));
    }

    #[test]
    fn test_compensate_all_after_complete() {
        let compensated = Rc::new(RefCell::new(Vec::new()));
        let c1 = compensated.clone();
        let c2 = compensated.clone();

        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            move || {
                c1.borrow_mut().push("s1");
                Ok("compensated".to_string())
            },
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "s2",
            || Ok("ok".to_string()),
            move || {
                c2.borrow_mut().push("s2");
                Ok("compensated".to_string())
            },
        ))
        .unwrap();
        tx.execute().unwrap();

        tx.compensate_all().unwrap();
        let comp = compensated.borrow();
        // Reverse order: s2 first, then s1.
        assert_eq!(comp.as_slice(), &["s2", "s1"]);
        assert_eq!(tx.status(), TransactionStatus::Compensated);
    }

    #[test]
    fn test_cannot_execute_twice() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        tx.execute().unwrap();
        let result = tx.execute();
        assert!(matches!(
            result,
            Err(CompensationError::AlreadyCompleted(_))
        ));
    }

    #[test]
    fn test_cannot_compensate_twice() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        tx.execute().unwrap();
        tx.compensate_all().unwrap();
        let result = tx.compensate_all();
        assert!(matches!(
            result,
            Err(CompensationError::AlreadyCompensated(_))
        ));
    }

    #[test]
    fn test_log_entries() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("result1".to_string()),
            || Ok("comp1".to_string()),
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "s2",
            || Err("boom".to_string()),
            || Ok("comp2".to_string()),
        ))
        .unwrap();
        let _ = tx.execute();
        let log = tx.log();
        // s1 forward ok, s2 forward fail, s1 compensation ok.
        assert_eq!(log.len(), 3);
        assert_eq!(log[0].action, LogAction::ForwardExecuted);
        assert_eq!(log[1].action, LogAction::ForwardFailed);
        assert_eq!(log[2].action, LogAction::CompensationExecuted);
        // Sequence numbers are monotonically increasing.
        assert!(log[0].sequence < log[1].sequence);
        assert!(log[1].sequence < log[2].sequence);
    }

    #[test]
    fn test_step_result() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("result_value".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        tx.execute().unwrap();
        assert_eq!(tx.step_result("s1"), Some("result_value"));
        assert!(tx.step_result("nonexistent").is_none());
    }

    #[test]
    fn test_count_with_status() {
        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "s2",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        tx.execute().unwrap();
        assert_eq!(tx.count_with_status(StepStatus::ForwardExecuted), 2);
        assert_eq!(tx.count_with_status(StepStatus::Pending), 0);
    }

    #[test]
    fn test_step_status_as_str() {
        assert_eq!(StepStatus::Pending.as_str(), "pending");
        assert_eq!(StepStatus::ForwardExecuted.as_str(), "forward_executed");
        assert_eq!(StepStatus::Compensated.as_str(), "compensated");
    }

    #[test]
    fn test_transaction_status_as_str() {
        assert_eq!(TransactionStatus::Building.as_str(), "building");
        assert_eq!(TransactionStatus::Completed.as_str(), "completed");
        assert_eq!(
            TransactionStatus::PartiallyCompensated.as_str(),
            "partially_compensated"
        );
    }

    #[test]
    fn test_three_step_middle_fails() {
        let compensated = Rc::new(RefCell::new(Vec::new()));
        let c1 = compensated.clone();

        let mut tx = CompensatingTransaction::new();
        tx.add_step(CompensationStep::new(
            "s1",
            || Ok("ok".to_string()),
            move || {
                c1.borrow_mut().push("s1");
                Ok("ok".to_string())
            },
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "s2",
            || Err("fail".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();
        tx.add_step(CompensationStep::new(
            "s3",
            || Ok("ok".to_string()),
            || Ok("ok".to_string()),
        ))
        .unwrap();

        let _ = tx.execute();
        // s1 should be compensated, s3 should be skipped.
        assert_eq!(compensated.borrow().as_slice(), &["s1"]);
        assert_eq!(tx.step_status("s3"), Some(StepStatus::Skipped));
    }

    #[test]
    fn test_needs_compensation() {
        assert!(StepStatus::ForwardExecuted.needs_compensation());
        assert!(!StepStatus::Pending.needs_compensation());
        assert!(!StepStatus::ForwardFailed.needs_compensation());
        assert!(!StepStatus::Compensated.needs_compensation());
        assert!(!StepStatus::Skipped.needs_compensation());
    }

    #[test]
    fn test_error_display() {
        let e = CompensationError::ForwardFailed {
            step: "s1".to_string(),
            reason: "boom".to_string(),
        };
        let s = format!("{e}");
        assert!(s.contains("s1"));
        assert!(s.contains("boom"));
    }
}
