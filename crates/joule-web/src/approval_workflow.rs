//! Approval workflow engine — multi-step approval chains, approval/rejection,
//! escalation on timeout, quorum voting (majority/unanimous), delegation,
//! approval history, and conditional routing based on attributes.
//!
//! Replaces Node.js approval workflow libraries with a pure-Rust engine
//! that tracks every approval decision from request to resolution.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Approval workflow domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalError {
    /// Workflow not found.
    WorkflowNotFound(String),
    /// Step not found.
    StepNotFound(String),
    /// Approver not found.
    ApproverNotFound(String),
    /// Duplicate workflow ID.
    DuplicateWorkflow(String),
    /// Duplicate step ID.
    DuplicateStep(String),
    /// Not authorized to approve.
    NotAuthorized { approver: String, step: String },
    /// Step already resolved (approved/rejected).
    AlreadyResolved(String),
    /// Invalid delegation (cannot delegate to self).
    InvalidDelegation { from: String, to: String },
    /// Workflow already completed.
    WorkflowCompleted(String),
    /// No steps defined.
    NoSteps,
}

impl std::fmt::Display for ApprovalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkflowNotFound(id) => write!(f, "workflow not found: {id}"),
            Self::StepNotFound(id) => write!(f, "step not found: {id}"),
            Self::ApproverNotFound(id) => write!(f, "approver not found: {id}"),
            Self::DuplicateWorkflow(id) => write!(f, "duplicate workflow: {id}"),
            Self::DuplicateStep(id) => write!(f, "duplicate step: {id}"),
            Self::NotAuthorized { approver, step } => {
                write!(f, "approver {approver} not authorized for step {step}")
            }
            Self::AlreadyResolved(id) => write!(f, "step already resolved: {id}"),
            Self::InvalidDelegation { from, to } => {
                write!(f, "invalid delegation from {from} to {to}")
            }
            Self::WorkflowCompleted(id) => write!(f, "workflow already completed: {id}"),
            Self::NoSteps => write!(f, "no steps defined in workflow"),
        }
    }
}

impl std::error::Error for ApprovalError {}

// ── Enums ───────────────────────────────────────────────────────

/// Overall workflow status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Draft,
    Pending,
    InProgress,
    Approved,
    Rejected,
    Cancelled,
    Escalated,
}

/// Status of a single approval step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    AwaitingVotes,
    Approved,
    Rejected,
    Escalated,
    Skipped,
}

/// Quorum type for multi-approver steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuorumType {
    /// Any single approver suffices.
    Any,
    /// Simple majority (> 50%).
    Majority,
    /// All approvers must approve.
    Unanimous,
    /// At least N approvers must approve.
    AtLeast(u32),
}

/// Vote decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoteDecision {
    Approve,
    Reject,
    Abstain,
}

/// Escalation strategy when a step times out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationStrategy {
    /// Auto-approve on timeout.
    AutoApprove,
    /// Auto-reject on timeout.
    AutoReject,
    /// Escalate to another approver.
    EscalateTo(String),
    /// Skip this step and move on.
    Skip,
}

/// Routing condition for conditional routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Always route to this step.
    Always,
    /// Route only if attribute equals value.
    AttributeEquals { key: String, value: String },
    /// Route if attribute is greater than threshold (numeric comparison).
    AttributeGreaterThan { key: String, threshold: i64 },
    /// Route if attribute is less than threshold.
    AttributeLessThan { key: String, threshold: i64 },
    /// Route if attribute is present.
    AttributePresent(String),
}

// ── Data Structures ─────────────────────────────────────────────

/// A vote cast by an approver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vote {
    pub approver_id: String,
    pub decision: VoteDecision,
    pub comment: Option<String>,
    pub voted_at: DateTime<Utc>,
}

/// Delegation record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Delegation {
    pub from: String,
    pub to: String,
    pub step_id: String,
    pub delegated_at: DateTime<Utc>,
    pub reason: Option<String>,
}

/// History entry for audit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalHistoryEntry {
    pub step_id: String,
    pub action: String,
    pub actor: String,
    pub timestamp: DateTime<Utc>,
    pub details: Option<String>,
}

/// Definition of an approval step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalStepDef {
    pub id: String,
    pub name: String,
    pub approvers: Vec<String>,
    pub quorum: QuorumType,
    pub timeout_seconds: Option<u64>,
    pub escalation: Option<EscalationStrategy>,
    pub routing_condition: RoutingCondition,
    /// Order in the chain (lower = earlier).
    pub order: u32,
}

/// Runtime state of an approval step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalStep {
    pub def: ApprovalStepDef,
    pub status: StepStatus,
    pub votes: Vec<Vote>,
    pub delegations: Vec<Delegation>,
    pub started_at: Option<DateTime<Utc>>,
    pub resolved_at: Option<DateTime<Utc>>,
}

impl ApprovalStep {
    fn new(def: ApprovalStepDef) -> Self {
        Self {
            def,
            status: StepStatus::Pending,
            votes: Vec::new(),
            delegations: Vec::new(),
            started_at: None,
            resolved_at: None,
        }
    }

    /// Get the effective approver (checking delegations).
    fn effective_approver(&self, approver_id: &str) -> Option<String> {
        // Check if anyone delegated to this approver_id for this step.
        for d in &self.delegations {
            if d.to == approver_id {
                return Some(approver_id.to_string());
            }
        }
        // Check if approver is in the original list.
        if self.def.approvers.iter().any(|a| a == approver_id) {
            // But check if they delegated away.
            let delegated_away = self.delegations.iter().any(|d| d.from == approver_id);
            if delegated_away {
                return None;
            }
            return Some(approver_id.to_string());
        }
        None
    }

    /// Check if a vote was already cast by this approver.
    fn has_voted(&self, approver_id: &str) -> bool {
        self.votes.iter().any(|v| v.approver_id == approver_id)
    }

    /// Count of approval votes.
    fn approval_count(&self) -> u32 {
        self.votes
            .iter()
            .filter(|v| v.decision == VoteDecision::Approve)
            .count() as u32
    }

    /// Count of rejection votes.
    fn rejection_count(&self) -> u32 {
        self.votes
            .iter()
            .filter(|v| v.decision == VoteDecision::Reject)
            .count() as u32
    }

    /// Total approvers who can vote (original - delegated_away + delegated_to).
    fn total_voters(&self) -> u32 {
        let mut voters: Vec<String> = Vec::new();
        for a in &self.def.approvers {
            let delegated_away = self.delegations.iter().any(|d| d.from == *a);
            if !delegated_away {
                voters.push(a.clone());
            }
        }
        for d in &self.delegations {
            if !voters.contains(&d.to) {
                voters.push(d.to.clone());
            }
        }
        voters.len() as u32
    }

    /// Check if quorum is met for approval.
    fn is_quorum_met(&self) -> Option<bool> {
        let total = self.total_voters();
        let approvals = self.approval_count();
        let rejections = self.rejection_count();
        let votes_cast = self.votes.len() as u32;

        match self.def.quorum {
            QuorumType::Any => {
                if approvals >= 1 {
                    Some(true)
                } else if rejections >= 1 {
                    Some(false)
                } else {
                    None
                }
            }
            QuorumType::Majority => {
                let needed = total / 2 + 1;
                if approvals >= needed {
                    Some(true)
                } else if rejections >= needed {
                    Some(false)
                } else if votes_cast >= total {
                    // All voted, check majority
                    Some(approvals > rejections)
                } else {
                    None
                }
            }
            QuorumType::Unanimous => {
                if rejections >= 1 {
                    Some(false)
                } else if approvals >= total {
                    Some(true)
                } else {
                    None
                }
            }
            QuorumType::AtLeast(n) => {
                if approvals >= n {
                    Some(true)
                } else if votes_cast >= total && approvals < n {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }

    /// Check if step has timed out given the current time.
    fn is_timed_out(&self, now: DateTime<Utc>) -> bool {
        if let (Some(started), Some(timeout_s)) = (self.started_at, self.def.timeout_seconds) {
            let deadline = started + Duration::seconds(timeout_s as i64);
            now >= deadline
        } else {
            false
        }
    }
}

/// The approval workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalWorkflow {
    pub id: String,
    pub name: String,
    pub status: WorkflowStatus,
    pub attributes: HashMap<String, String>,
    pub steps: Vec<ApprovalStep>,
    pub history: Vec<ApprovalHistoryEntry>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub requester: String,
}

// ── Engine ──────────────────────────────────────────────────────

/// Manages approval workflows.
pub struct ApprovalEngine {
    workflows: HashMap<String, ApprovalWorkflow>,
}

impl ApprovalEngine {
    pub fn new() -> Self {
        Self {
            workflows: HashMap::new(),
        }
    }

    /// Create a new workflow.
    pub fn create_workflow(
        &mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        requester: impl Into<String>,
        attributes: HashMap<String, String>,
        step_defs: Vec<ApprovalStepDef>,
    ) -> Result<&ApprovalWorkflow, ApprovalError> {
        let id = id.into();
        if self.workflows.contains_key(&id) {
            return Err(ApprovalError::DuplicateWorkflow(id));
        }
        if step_defs.is_empty() {
            return Err(ApprovalError::NoSteps);
        }

        // Check for duplicate step IDs.
        let mut seen_step_ids = std::collections::HashSet::new();
        for sd in &step_defs {
            if !seen_step_ids.insert(&sd.id) {
                return Err(ApprovalError::DuplicateStep(sd.id.clone()));
            }
        }

        let now = Utc::now();
        let mut sorted_defs = step_defs;
        sorted_defs.sort_by_key(|s| s.order);

        let steps: Vec<ApprovalStep> = sorted_defs
            .into_iter()
            .map(ApprovalStep::new)
            .collect();

        let wf = ApprovalWorkflow {
            id: id.clone(),
            name: name.into(),
            status: WorkflowStatus::Draft,
            attributes,
            steps,
            history: Vec::new(),
            created_at: now,
            updated_at: now,
            requester: requester.into(),
        };
        self.workflows.insert(id.clone(), wf);
        Ok(self.workflows.get(&id).unwrap())
    }

    /// Start the workflow — activates the first applicable step.
    pub fn start_workflow(&mut self, workflow_id: &str) -> Result<(), ApprovalError> {
        let wf = self
            .workflows
            .get_mut(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;

        if wf.status == WorkflowStatus::Approved
            || wf.status == WorkflowStatus::Rejected
            || wf.status == WorkflowStatus::Cancelled
        {
            return Err(ApprovalError::WorkflowCompleted(workflow_id.to_string()));
        }

        let now = Utc::now();
        wf.status = WorkflowStatus::InProgress;
        wf.updated_at = now;

        // Find first step that passes routing condition.
        let attrs = wf.attributes.clone();
        for step in &mut wf.steps {
            if step.status == StepStatus::Pending && evaluate_condition(&step.def.routing_condition, &attrs) {
                step.status = StepStatus::AwaitingVotes;
                step.started_at = Some(now);
                let step_id = step.def.id.clone();
                wf.history.push(ApprovalHistoryEntry {
                    step_id,
                    action: "started".to_string(),
                    actor: "system".to_string(),
                    timestamp: now,
                    details: None,
                });
                break;
            } else if step.status == StepStatus::Pending {
                step.status = StepStatus::Skipped;
            }
        }

        Ok(())
    }

    /// Cast a vote on the currently active step.
    pub fn cast_vote(
        &mut self,
        workflow_id: &str,
        step_id: &str,
        approver_id: &str,
        decision: VoteDecision,
        comment: Option<String>,
    ) -> Result<StepStatus, ApprovalError> {
        let wf = self
            .workflows
            .get_mut(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;

        if wf.status == WorkflowStatus::Approved
            || wf.status == WorkflowStatus::Rejected
            || wf.status == WorkflowStatus::Cancelled
        {
            return Err(ApprovalError::WorkflowCompleted(workflow_id.to_string()));
        }

        let step = wf
            .steps
            .iter_mut()
            .find(|s| s.def.id == step_id)
            .ok_or_else(|| ApprovalError::StepNotFound(step_id.to_string()))?;

        if step.status != StepStatus::AwaitingVotes {
            return Err(ApprovalError::AlreadyResolved(step_id.to_string()));
        }

        if step.effective_approver(approver_id).is_none() {
            return Err(ApprovalError::NotAuthorized {
                approver: approver_id.to_string(),
                step: step_id.to_string(),
            });
        }

        if step.has_voted(approver_id) {
            return Err(ApprovalError::AlreadyResolved(
                format!("{approver_id} already voted on {step_id}"),
            ));
        }

        let now = Utc::now();
        step.votes.push(Vote {
            approver_id: approver_id.to_string(),
            decision,
            comment: comment.clone(),
            voted_at: now,
        });

        wf.history.push(ApprovalHistoryEntry {
            step_id: step_id.to_string(),
            action: format!("vote:{decision:?}"),
            actor: approver_id.to_string(),
            timestamp: now,
            details: comment,
        });

        // Check quorum.
        let quorum_result = step.is_quorum_met();
        let step_status = match quorum_result {
            Some(true) => {
                step.status = StepStatus::Approved;
                step.resolved_at = Some(now);
                StepStatus::Approved
            }
            Some(false) => {
                step.status = StepStatus::Rejected;
                step.resolved_at = Some(now);
                StepStatus::Rejected
            }
            None => StepStatus::AwaitingVotes,
        };

        wf.updated_at = now;

        // If step resolved, advance workflow.
        if step_status == StepStatus::Approved {
            self.advance_workflow(workflow_id);
        } else if step_status == StepStatus::Rejected {
            let wf2 = self.workflows.get_mut(workflow_id).unwrap();
            wf2.status = WorkflowStatus::Rejected;
            wf2.updated_at = Utc::now();
        }

        Ok(step_status)
    }

    /// Delegate approval authority.
    pub fn delegate(
        &mut self,
        workflow_id: &str,
        step_id: &str,
        from: &str,
        to: &str,
        reason: Option<String>,
    ) -> Result<(), ApprovalError> {
        if from == to {
            return Err(ApprovalError::InvalidDelegation {
                from: from.to_string(),
                to: to.to_string(),
            });
        }

        let wf = self
            .workflows
            .get_mut(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;

        let step = wf
            .steps
            .iter_mut()
            .find(|s| s.def.id == step_id)
            .ok_or_else(|| ApprovalError::StepNotFound(step_id.to_string()))?;

        if !step.def.approvers.contains(&from.to_string()) {
            return Err(ApprovalError::NotAuthorized {
                approver: from.to_string(),
                step: step_id.to_string(),
            });
        }

        let now = Utc::now();
        step.delegations.push(Delegation {
            from: from.to_string(),
            to: to.to_string(),
            step_id: step_id.to_string(),
            delegated_at: now,
            reason: reason.clone(),
        });

        wf.history.push(ApprovalHistoryEntry {
            step_id: step_id.to_string(),
            action: "delegated".to_string(),
            actor: from.to_string(),
            timestamp: now,
            details: reason,
        });
        wf.updated_at = now;

        Ok(())
    }

    /// Check timeouts and apply escalation strategies.
    pub fn check_timeouts(
        &mut self,
        workflow_id: &str,
        now: DateTime<Utc>,
    ) -> Result<Vec<String>, ApprovalError> {
        let wf = self
            .workflows
            .get_mut(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;

        let mut escalated_steps = Vec::new();

        for step in &mut wf.steps {
            if step.status != StepStatus::AwaitingVotes {
                continue;
            }
            if !step.is_timed_out(now) {
                continue;
            }

            let step_id = step.def.id.clone();
            let escalation = step.def.escalation.clone();

            match escalation {
                Some(EscalationStrategy::AutoApprove) => {
                    step.status = StepStatus::Approved;
                    step.resolved_at = Some(now);
                    escalated_steps.push(step_id.clone());
                }
                Some(EscalationStrategy::AutoReject) => {
                    step.status = StepStatus::Rejected;
                    step.resolved_at = Some(now);
                    escalated_steps.push(step_id.clone());
                }
                Some(EscalationStrategy::EscalateTo(new_approver)) => {
                    step.def.approvers.push(new_approver);
                    step.status = StepStatus::Escalated;
                    escalated_steps.push(step_id.clone());
                }
                Some(EscalationStrategy::Skip) => {
                    step.status = StepStatus::Skipped;
                    step.resolved_at = Some(now);
                    escalated_steps.push(step_id.clone());
                }
                None => {
                    step.status = StepStatus::Escalated;
                    escalated_steps.push(step_id.clone());
                }
            }

            wf.history.push(ApprovalHistoryEntry {
                step_id: step_id,
                action: "escalated".to_string(),
                actor: "system".to_string(),
                timestamp: now,
                details: Some("timeout".to_string()),
            });
        }

        wf.updated_at = now;
        Ok(escalated_steps)
    }

    /// Cancel a workflow.
    pub fn cancel_workflow(
        &mut self,
        workflow_id: &str,
        actor: &str,
    ) -> Result<(), ApprovalError> {
        let wf = self
            .workflows
            .get_mut(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;

        if wf.status == WorkflowStatus::Approved || wf.status == WorkflowStatus::Rejected {
            return Err(ApprovalError::WorkflowCompleted(workflow_id.to_string()));
        }

        let now = Utc::now();
        wf.status = WorkflowStatus::Cancelled;
        wf.updated_at = now;
        wf.history.push(ApprovalHistoryEntry {
            step_id: String::new(),
            action: "cancelled".to_string(),
            actor: actor.to_string(),
            timestamp: now,
            details: None,
        });

        Ok(())
    }

    /// Get workflow by ID.
    pub fn get_workflow(&self, id: &str) -> Option<&ApprovalWorkflow> {
        self.workflows.get(id)
    }

    /// Get history for a workflow.
    pub fn get_history(&self, workflow_id: &str) -> Result<&[ApprovalHistoryEntry], ApprovalError> {
        let wf = self
            .workflows
            .get(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;
        Ok(&wf.history)
    }

    /// Get current active step.
    pub fn active_step(&self, workflow_id: &str) -> Result<Option<&ApprovalStep>, ApprovalError> {
        let wf = self
            .workflows
            .get(workflow_id)
            .ok_or_else(|| ApprovalError::WorkflowNotFound(workflow_id.to_string()))?;
        Ok(wf
            .steps
            .iter()
            .find(|s| s.status == StepStatus::AwaitingVotes || s.status == StepStatus::Escalated))
    }

    /// List all workflow IDs.
    pub fn list_workflows(&self) -> Vec<&str> {
        self.workflows.keys().map(|k| k.as_str()).collect()
    }

    /// Advance to the next step after approval.
    fn advance_workflow(&mut self, workflow_id: &str) {
        let wf = match self.workflows.get_mut(workflow_id) {
            Some(w) => w,
            None => return,
        };

        let now = Utc::now();
        let attrs = wf.attributes.clone();

        // Find next pending step.
        let mut found_next = false;
        for step in &mut wf.steps {
            if step.status == StepStatus::Pending {
                if evaluate_condition(&step.def.routing_condition, &attrs) {
                    step.status = StepStatus::AwaitingVotes;
                    step.started_at = Some(now);
                    found_next = true;
                    let step_id = step.def.id.clone();
                    wf.history.push(ApprovalHistoryEntry {
                        step_id,
                        action: "started".to_string(),
                        actor: "system".to_string(),
                        timestamp: now,
                        details: None,
                    });
                    break;
                } else {
                    step.status = StepStatus::Skipped;
                }
            }
        }

        if !found_next {
            // All steps done — workflow approved.
            wf.status = WorkflowStatus::Approved;
        }
        wf.updated_at = now;
    }
}

impl Default for ApprovalEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Routing Condition Evaluation ────────────────────────────────

fn evaluate_condition(condition: &RoutingCondition, attrs: &HashMap<String, String>) -> bool {
    match condition {
        RoutingCondition::Always => true,
        RoutingCondition::AttributeEquals { key, value } => {
            attrs.get(key).map_or(false, |v| v == value)
        }
        RoutingCondition::AttributeGreaterThan { key, threshold } => {
            attrs
                .get(key)
                .and_then(|v| v.parse::<i64>().ok())
                .map_or(false, |v| v > *threshold)
        }
        RoutingCondition::AttributeLessThan { key, threshold } => {
            attrs
                .get(key)
                .and_then(|v| v.parse::<i64>().ok())
                .map_or(false, |v| v < *threshold)
        }
        RoutingCondition::AttributePresent(key) => attrs.contains_key(key),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn step_def(id: &str, approvers: Vec<&str>, order: u32) -> ApprovalStepDef {
        ApprovalStepDef {
            id: id.to_string(),
            name: format!("Step {id}"),
            approvers: approvers.into_iter().map(|s| s.to_string()).collect(),
            quorum: QuorumType::Any,
            timeout_seconds: None,
            escalation: None,
            routing_condition: RoutingCondition::Always,
            order,
        }
    }

    fn setup_engine() -> ApprovalEngine {
        let mut engine = ApprovalEngine::new();
        engine
            .create_workflow(
                "wf1",
                "Test Workflow",
                "requester1",
                HashMap::new(),
                vec![
                    step_def("s1", vec!["alice"], 1),
                    step_def("s2", vec!["bob"], 2),
                ],
            )
            .unwrap();
        engine.start_workflow("wf1").unwrap();
        engine
    }

    #[test]
    fn test_create_workflow() {
        let mut engine = ApprovalEngine::new();
        let wf = engine
            .create_workflow(
                "wf1",
                "Test",
                "req",
                HashMap::new(),
                vec![step_def("s1", vec!["alice"], 1)],
            )
            .unwrap();
        assert_eq!(wf.status, WorkflowStatus::Draft);
        assert_eq!(wf.steps.len(), 1);
    }

    #[test]
    fn test_duplicate_workflow() {
        let mut engine = ApprovalEngine::new();
        engine
            .create_workflow("wf1", "A", "r", HashMap::new(), vec![step_def("s1", vec!["a"], 1)])
            .unwrap();
        let err = engine
            .create_workflow("wf1", "B", "r", HashMap::new(), vec![step_def("s2", vec!["b"], 1)])
            .unwrap_err();
        assert_eq!(err, ApprovalError::DuplicateWorkflow("wf1".into()));
    }

    #[test]
    fn test_no_steps_error() {
        let mut engine = ApprovalEngine::new();
        let err = engine
            .create_workflow("wf1", "A", "r", HashMap::new(), vec![])
            .unwrap_err();
        assert_eq!(err, ApprovalError::NoSteps);
    }

    #[test]
    fn test_start_workflow() {
        let engine = setup_engine();
        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.status, WorkflowStatus::InProgress);
        assert_eq!(wf.steps[0].status, StepStatus::AwaitingVotes);
    }

    #[test]
    fn test_simple_approval_flow() {
        let mut engine = setup_engine();
        let result = engine
            .cast_vote("wf1", "s1", "alice", VoteDecision::Approve, None)
            .unwrap();
        assert_eq!(result, StepStatus::Approved);

        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.steps[1].status, StepStatus::AwaitingVotes);
    }

    #[test]
    fn test_rejection_ends_workflow() {
        let mut engine = setup_engine();
        let result = engine
            .cast_vote("wf1", "s1", "alice", VoteDecision::Reject, Some("no".into()))
            .unwrap();
        assert_eq!(result, StepStatus::Rejected);

        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.status, WorkflowStatus::Rejected);
    }

    #[test]
    fn test_full_approval_completes_workflow() {
        let mut engine = setup_engine();
        engine
            .cast_vote("wf1", "s1", "alice", VoteDecision::Approve, None)
            .unwrap();
        engine
            .cast_vote("wf1", "s2", "bob", VoteDecision::Approve, None)
            .unwrap();
        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.status, WorkflowStatus::Approved);
    }

    #[test]
    fn test_majority_quorum() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["a", "b", "c"], 1);
        sd.quorum = QuorumType::Majority;
        engine
            .create_workflow("wf1", "Test", "r", HashMap::new(), vec![sd])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        // First approval — not enough.
        let r = engine
            .cast_vote("wf1", "s1", "a", VoteDecision::Approve, None)
            .unwrap();
        assert_eq!(r, StepStatus::AwaitingVotes);

        // Second approval — majority met.
        let r = engine
            .cast_vote("wf1", "s1", "b", VoteDecision::Approve, None)
            .unwrap();
        assert_eq!(r, StepStatus::Approved);
    }

    #[test]
    fn test_unanimous_quorum() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["a", "b"], 1);
        sd.quorum = QuorumType::Unanimous;
        engine
            .create_workflow("wf1", "Test", "r", HashMap::new(), vec![sd])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        engine
            .cast_vote("wf1", "s1", "a", VoteDecision::Approve, None)
            .unwrap();
        // One rejection kills unanimous.
        let r = engine
            .cast_vote("wf1", "s1", "b", VoteDecision::Reject, None)
            .unwrap();
        assert_eq!(r, StepStatus::Rejected);
    }

    #[test]
    fn test_at_least_quorum() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["a", "b", "c"], 1);
        sd.quorum = QuorumType::AtLeast(2);
        engine
            .create_workflow("wf1", "Test", "r", HashMap::new(), vec![sd])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        engine
            .cast_vote("wf1", "s1", "a", VoteDecision::Approve, None)
            .unwrap();
        let r = engine
            .cast_vote("wf1", "s1", "b", VoteDecision::Approve, None)
            .unwrap();
        assert_eq!(r, StepStatus::Approved);
    }

    #[test]
    fn test_delegation() {
        let mut engine = setup_engine();
        engine
            .delegate("wf1", "s1", "alice", "charlie", Some("OOO".into()))
            .unwrap();

        // Alice can no longer vote.
        let err = engine
            .cast_vote("wf1", "s1", "alice", VoteDecision::Approve, None)
            .unwrap_err();
        assert!(matches!(err, ApprovalError::NotAuthorized { .. }));

        // Charlie can vote.
        let r = engine
            .cast_vote("wf1", "s1", "charlie", VoteDecision::Approve, None)
            .unwrap();
        assert_eq!(r, StepStatus::Approved);
    }

    #[test]
    fn test_self_delegation_error() {
        let mut engine = setup_engine();
        let err = engine
            .delegate("wf1", "s1", "alice", "alice", None)
            .unwrap_err();
        assert!(matches!(err, ApprovalError::InvalidDelegation { .. }));
    }

    #[test]
    fn test_escalation_auto_approve() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["alice"], 1);
        sd.timeout_seconds = Some(60);
        sd.escalation = Some(EscalationStrategy::AutoApprove);
        engine
            .create_workflow("wf1", "Test", "r", HashMap::new(), vec![sd])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        let future = Utc::now() + Duration::seconds(120);
        let escalated = engine.check_timeouts("wf1", future).unwrap();
        assert_eq!(escalated, vec!["s1"]);

        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.steps[0].status, StepStatus::Approved);
    }

    #[test]
    fn test_escalation_auto_reject() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["alice"], 1);
        sd.timeout_seconds = Some(60);
        sd.escalation = Some(EscalationStrategy::AutoReject);
        engine
            .create_workflow("wf1", "Test", "r", HashMap::new(), vec![sd])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        let future = Utc::now() + Duration::seconds(120);
        engine.check_timeouts("wf1", future).unwrap();

        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.steps[0].status, StepStatus::Rejected);
    }

    #[test]
    fn test_escalation_skip() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["alice"], 1);
        sd.timeout_seconds = Some(60);
        sd.escalation = Some(EscalationStrategy::Skip);
        engine
            .create_workflow(
                "wf1",
                "Test",
                "r",
                HashMap::new(),
                vec![sd, step_def("s2", vec!["bob"], 2)],
            )
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        let future = Utc::now() + Duration::seconds(120);
        engine.check_timeouts("wf1", future).unwrap();

        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.steps[0].status, StepStatus::Skipped);
    }

    #[test]
    fn test_conditional_routing_attribute_equals() {
        let mut engine = ApprovalEngine::new();
        let mut sd1 = step_def("s1", vec!["alice"], 1);
        sd1.routing_condition = RoutingCondition::AttributeEquals {
            key: "dept".into(),
            value: "finance".into(),
        };
        let sd2 = step_def("s2", vec!["bob"], 2);

        let mut attrs = HashMap::new();
        attrs.insert("dept".into(), "engineering".into());

        engine
            .create_workflow("wf1", "Test", "r", attrs, vec![sd1, sd2])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        let wf = engine.get_workflow("wf1").unwrap();
        // s1 should be skipped because dept != finance.
        assert_eq!(wf.steps[0].status, StepStatus::Skipped);
        assert_eq!(wf.steps[1].status, StepStatus::AwaitingVotes);
    }

    #[test]
    fn test_conditional_routing_greater_than() {
        let mut engine = ApprovalEngine::new();
        let mut sd = step_def("s1", vec!["alice"], 1);
        sd.routing_condition = RoutingCondition::AttributeGreaterThan {
            key: "amount".into(),
            threshold: 1000,
        };

        let mut attrs = HashMap::new();
        attrs.insert("amount".into(), "5000".into());

        engine
            .create_workflow("wf1", "Test", "r", attrs, vec![sd])
            .unwrap();
        engine.start_workflow("wf1").unwrap();

        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.steps[0].status, StepStatus::AwaitingVotes);
    }

    #[test]
    fn test_cancel_workflow() {
        let mut engine = setup_engine();
        engine.cancel_workflow("wf1", "admin").unwrap();
        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.status, WorkflowStatus::Cancelled);
    }

    #[test]
    fn test_cannot_vote_on_completed_workflow() {
        let mut engine = setup_engine();
        engine.cancel_workflow("wf1", "admin").unwrap();
        let err = engine
            .cast_vote("wf1", "s1", "alice", VoteDecision::Approve, None)
            .unwrap_err();
        assert!(matches!(err, ApprovalError::WorkflowCompleted(_)));
    }

    #[test]
    fn test_history_tracking() {
        let mut engine = setup_engine();
        engine
            .cast_vote("wf1", "s1", "alice", VoteDecision::Approve, None)
            .unwrap();
        let history = engine.get_history("wf1").unwrap();
        assert!(history.len() >= 2); // started + vote
        assert!(history.iter().any(|h| h.action.starts_with("vote:")));
    }

    #[test]
    fn test_active_step() {
        let engine = setup_engine();
        let active = engine.active_step("wf1").unwrap().unwrap();
        assert_eq!(active.def.id, "s1");
    }

    #[test]
    fn test_unauthorized_approver() {
        let mut engine = setup_engine();
        let err = engine
            .cast_vote("wf1", "s1", "bob", VoteDecision::Approve, None)
            .unwrap_err();
        assert!(matches!(err, ApprovalError::NotAuthorized { .. }));
    }

    #[test]
    fn test_duplicate_step_ids() {
        let mut engine = ApprovalEngine::new();
        let err = engine
            .create_workflow(
                "wf1",
                "Test",
                "r",
                HashMap::new(),
                vec![step_def("s1", vec!["a"], 1), step_def("s1", vec!["b"], 2)],
            )
            .unwrap_err();
        assert!(matches!(err, ApprovalError::DuplicateStep(_)));
    }

    #[test]
    fn test_attribute_present_routing() {
        let cond = RoutingCondition::AttributePresent("flag".into());
        let mut attrs = HashMap::new();
        assert!(!evaluate_condition(&cond, &attrs));
        attrs.insert("flag".into(), "yes".into());
        assert!(evaluate_condition(&cond, &attrs));
    }

    #[test]
    fn test_attribute_less_than_routing() {
        let cond = RoutingCondition::AttributeLessThan {
            key: "priority".into(),
            threshold: 5,
        };
        let mut attrs = HashMap::new();
        attrs.insert("priority".into(), "3".into());
        assert!(evaluate_condition(&cond, &attrs));
        attrs.insert("priority".into(), "7".into());
        assert!(!evaluate_condition(&cond, &attrs));
    }

    #[test]
    fn test_list_workflows() {
        let engine = setup_engine();
        let ids = engine.list_workflows();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&"wf1"));
    }

    #[test]
    fn test_step_ordering() {
        let mut engine = ApprovalEngine::new();
        // Insert steps out of order — should be sorted by `order`.
        engine
            .create_workflow(
                "wf1",
                "Test",
                "r",
                HashMap::new(),
                vec![
                    step_def("s3", vec!["charlie"], 3),
                    step_def("s1", vec!["alice"], 1),
                    step_def("s2", vec!["bob"], 2),
                ],
            )
            .unwrap();
        let wf = engine.get_workflow("wf1").unwrap();
        assert_eq!(wf.steps[0].def.id, "s1");
        assert_eq!(wf.steps[1].def.id, "s2");
        assert_eq!(wf.steps[2].def.id, "s3");
    }
}
