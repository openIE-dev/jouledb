//! Federated Dispatch — work flows to competence, not from authority.
//!
//! In centralized architectures, the orchestrator assigns work. In federated
//! dispatch, nodes acquire work through self-selection based on competence
//! match, delegation level, and resource availability.
//!
//! The observer consumes status — it does NOT produce decisions.
//!
//! From the axioms:
//! - A1 (SUBSIDIARITY): decisions at the lowest competent authority
//! - A2 (INTENT): higher levels provide (purpose, constraints), not instructions
//! - A3 (OBLIGATION): halt obligation checked at every transition
//!
//! Architecture:
//! ```text
//!  Intent Layer (observer)
//!       │ intent (purpose + constraints)
//!       │ status (results + metrics)
//!       ▼
//!  Dispatch Mesh
//!    Node A ◄─► Node B ◄─► Node C
//!    L7:git    L5:fs     L3:web
//!       │
//!  Execution Layer (sandbox, native, wasm, remote)
//! ```

use crate::competence::{CompetenceLedger, DelegationLevel, DomainId};
use crate::substrate::{
    ComputeSubstrate, SubstrateBid, SubstrateCapability, SubstrateEnergyProfile, TaskClass,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// A task entering the dispatch mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskIntent {
    /// Unique task identifier.
    pub task_id: String,
    /// The domain this task belongs to.
    pub domain: DomainId,
    /// Purpose: what needs to happen (not how).
    pub purpose: String,
    /// Constraints: boundaries the agent must respect.
    pub constraints: Vec<String>,
    /// Energy budget in microjoules.
    pub energy_budget_uj: u64,
    /// Deadline (wall-clock duration from now).
    pub deadline: Option<Duration>,
    /// Minimum delegation level required to handle this task.
    pub required_level: DelegationLevel,
    /// Task class: the nature of the computation (determines optimal substrate).
    pub task_class: Option<TaskClass>,
    /// Structural similarity vector (optional, for fine-grained matching).
    /// In practice this comes from the dgk-formula embedded knowledge base.
    pub similarity_hint: Option<Vec<f32>>,
}

/// A node's bid to acquire a task (self-selection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBid {
    /// The node submitting this bid.
    pub node_id: String,
    /// The task being bid on.
    pub task_id: String,
    /// Competence score in the task domain.
    pub competence_score: f64,
    /// Delegation level in the task domain.
    pub delegation_level: DelegationLevel,
    /// Estimated energy cost in microjoules.
    pub estimated_energy_uj: u64,
    /// Estimated duration.
    pub estimated_duration: Option<Duration>,
    /// Confidence in completing the task (0.0-1.0).
    pub confidence: f64,
}

impl TaskBid {
    /// The composite selection score: competence × confidence.
    /// Ties broken by lowest energy cost (handled by the mesh).
    pub fn selection_score(&self) -> f64 {
        self.competence_score * self.confidence
    }
}

/// Outcome of an agent's work on a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskAction {
    /// Node handled the task autonomously.
    Acted,
    /// Node escalated (insufficient competence or confidence).
    Escalated,
    /// Node halted on defect detection (A3 obligation).
    Halted,
}

/// Outcome status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskOutcome {
    Success,
    Failure,
    Pending,
}

/// Status report from a node to the observer (§4.3).
///
/// The observer uses these to update competence views and detect patterns.
/// The observer does NOT use these to approve, reject, or reassign work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReport {
    /// Node that produced this report.
    pub node_id: String,
    /// Task that was worked on.
    pub task_id: String,
    /// Domain of the task.
    pub domain: DomainId,
    /// Delegation level at which the task was handled.
    pub delegation: DelegationLevel,
    /// What the node did.
    pub action: TaskAction,
    /// Outcome of the work.
    pub outcome: TaskOutcome,
    /// Energy consumed in microjoules.
    pub energy_uj: u64,
    /// Wall-clock duration.
    pub duration: Duration,
    /// Artifacts produced (serialized references).
    pub artifacts: Vec<String>,
    /// Node's confidence in the result (0.0-1.0).
    pub confidence: f64,
    /// Timestamp (nanos since epoch).
    pub timestamp_ns: u64,
}

/// A node in the dispatch mesh.
#[derive(Debug)]
pub struct DispatchNode {
    /// Unique node identifier.
    pub node_id: String,
    /// The node's competence ledger.
    pub ledger: CompetenceLedger,
    /// Available compute substrates (CPU always present, plus detected hardware).
    pub substrates: SubstrateCapability,
    /// Currently assigned task (if any).
    pub current_task: Option<String>,
    /// Energy remaining in the node's budget.
    pub energy_remaining_uj: u64,
    /// Whether the node is available for work.
    pub available: bool,
}

impl DispatchNode {
    pub fn new(node_id: String, energy_budget_uj: u64) -> Self {
        Self {
            node_id,
            ledger: CompetenceLedger::new(),
            substrates: SubstrateCapability::cpu_only(),
            current_task: None,
            energy_remaining_uj: energy_budget_uj,
            available: true,
        }
    }

    /// Evaluate whether this node should bid on a task.
    ///
    /// Returns Some(bid) if the node is competent and available.
    /// Returns None if the node cannot handle this task.
    pub fn evaluate_task(&self, intent: &TaskIntent) -> Option<TaskBid> {
        // Can't bid if busy or unavailable
        if !self.available || self.current_task.is_some() {
            return None;
        }

        // Can't bid if insufficient energy
        if self.energy_remaining_uj < intent.energy_budget_uj {
            return None;
        }

        // Check competence in the task domain
        let (score, level) = self.ledger.bid(&intent.domain)?;

        // Can't bid if below required delegation level
        if level < intent.required_level {
            return None;
        }

        // Compute confidence: higher score + higher level = more confident
        let confidence = (score.max(0.0) + level.as_u8() as f64 / 7.0) / 2.0;

        Some(TaskBid {
            node_id: self.node_id.clone(),
            task_id: intent.task_id.clone(),
            competence_score: score,
            delegation_level: level,
            estimated_energy_uj: intent.energy_budget_uj, // conservative estimate
            estimated_duration: intent.deadline,
            confidence: confidence.clamp(0.0, 1.0),
        })
    }

    /// Assign a task to this node (after winning the bid).
    pub fn assign_task(&mut self, task_id: &str) {
        self.current_task = Some(task_id.to_string());
        self.available = false;
    }

    /// Create a node with specific substrate capabilities.
    pub fn with_substrates(
        node_id: String,
        energy_budget_uj: u64,
        substrates: SubstrateCapability,
    ) -> Self {
        Self {
            node_id,
            ledger: CompetenceLedger::new(),
            substrates,
            current_task: None,
            energy_remaining_uj: energy_budget_uj,
            available: true,
        }
    }

    /// Substrate-aware task evaluation.
    ///
    /// Jointly optimizes (competence × hardware fit). A GPU node with moderate
    /// domain competence outbids a CPU-only node with high competence for
    /// tensor/inference tasks, because the energy differential dominates.
    ///
    /// This is the honest answer: give the agent the best place to work.
    pub fn evaluate_task_with_substrate(
        &self,
        intent: &TaskIntent,
        profile: &SubstrateEnergyProfile,
    ) -> Option<SubstrateBid> {
        if !self.available || self.current_task.is_some() {
            return None;
        }

        // Check competence in the task domain
        let (score, level) = self.ledger.bid(&intent.domain)?;
        if level < intent.required_level {
            return None;
        }

        // Determine task class (default to Scalar if not specified)
        let task_class = intent.task_class.unwrap_or(TaskClass::Scalar);

        // Find optimal substrate from what this node has
        let (substrate, estimate) =
            profile.optimal_substrate(task_class, self.substrates.available())?;

        // Check energy budget (use substrate-aware estimate)
        if self.energy_remaining_uj < estimate.energy_per_unit_uj {
            return None;
        }

        Some(SubstrateBid::new(
            self.node_id.clone(),
            intent.task_id.clone(),
            score,
            substrate,
            task_class,
            &estimate,
        ))
    }

    /// Complete the current task.
    pub fn complete_task(&mut self, energy_consumed_uj: u64) {
        self.current_task = None;
        self.energy_remaining_uj = self.energy_remaining_uj.saturating_sub(energy_consumed_uj);
        self.available = true;
    }
}

/// Halt condition — checked before every state transition (§4.4).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HaltReason {
    /// Energy budget exhausted.
    Budget,
    /// Sandbox violation detected.
    Safety,
    /// Integrity chain invalid.
    Integrity,
    /// User stop signal received.
    UserSignal,
}

/// Result of a halt check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionCheck {
    /// Proceed with the transition.
    Proceed,
    /// Escalate to higher authority (confidence below threshold).
    Escalate,
    /// Halt immediately (defect condition detected).
    Halt(HaltReason),
}

/// Check halt conditions before every transition (§4.4).
///
/// This is not optional. It is the first evaluation in every state handler.
/// A node that halts is never penalized. A node that fails to halt when
/// conditions are met suffers maximum penalty (reset to Level 1 in all domains).
pub fn check_halt_conditions(
    energy_remaining_uj: u64,
    min_energy_uj: u64,
    confidence: f64,
    confidence_threshold: f64,
    sandbox_violation: bool,
    integrity_valid: bool,
    user_stop: bool,
) -> TransitionCheck {
    if energy_remaining_uj < min_energy_uj {
        return TransitionCheck::Halt(HaltReason::Budget);
    }
    if sandbox_violation {
        return TransitionCheck::Halt(HaltReason::Safety);
    }
    if !integrity_valid {
        return TransitionCheck::Halt(HaltReason::Integrity);
    }
    if user_stop {
        return TransitionCheck::Halt(HaltReason::UserSignal);
    }
    if confidence < confidence_threshold {
        return TransitionCheck::Escalate;
    }
    TransitionCheck::Proceed
}

/// The dispatch mesh: coordinates self-selection across nodes.
///
/// The mesh does NOT assign work. It facilitates the self-selection process:
/// 1. Task enters the mesh
/// 2. All nodes evaluate their competence bid
/// 3. Highest (competence × confidence) wins
/// 4. Ties broken by lowest energy cost, then lowest latency
/// 5. If no node exceeds confidence threshold → escalate to LLM fallback
pub struct DispatchMesh {
    /// Registered nodes.
    nodes: HashMap<String, DispatchNode>,
    /// Status reports (append-only log for the observer).
    reports: Vec<StatusReport>,
    /// Confidence threshold: bids below this → escalate to LLM.
    confidence_threshold: f64,
    /// Minimum energy to stay operational.
    min_energy_uj: u64,
    /// Substrate energy profile (for substrate-aware dispatch).
    substrate_profile: SubstrateEnergyProfile,
}

impl DispatchMesh {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            reports: Vec::new(),
            confidence_threshold: 0.3,
            min_energy_uj: 1000, // 1 mJ minimum
            substrate_profile: SubstrateEnergyProfile::default_profile(),
        }
    }

    pub fn with_thresholds(confidence_threshold: f64, min_energy_uj: u64) -> Self {
        Self {
            nodes: HashMap::new(),
            reports: Vec::new(),
            confidence_threshold,
            min_energy_uj,
            substrate_profile: SubstrateEnergyProfile::default_profile(),
        }
    }

    /// Get a reference to the substrate energy profile.
    pub fn substrate_profile(&self) -> &SubstrateEnergyProfile {
        &self.substrate_profile
    }

    /// Register a node in the mesh.
    pub fn register_node(&mut self, node: DispatchNode) {
        self.nodes.insert(node.node_id.clone(), node);
    }

    /// Remove a node from the mesh.
    pub fn remove_node(&mut self, node_id: &str) -> Option<DispatchNode> {
        self.nodes.remove(node_id)
    }

    /// Get a reference to a node.
    pub fn node(&self, node_id: &str) -> Option<&DispatchNode> {
        self.nodes.get(node_id)
    }

    /// Get a mutable reference to a node.
    pub fn node_mut(&mut self, node_id: &str) -> Option<&mut DispatchNode> {
        self.nodes.get_mut(node_id)
    }

    /// Self-selection: find the best node for a task.
    ///
    /// Returns the winning bid, or None if no node qualifies
    /// (in which case the task should escalate to LLM fallback).
    pub fn select_for_task(&self, intent: &TaskIntent) -> Option<TaskBid> {
        let mut best: Option<TaskBid> = None;

        for node in self.nodes.values() {
            if let Some(bid) = node.evaluate_task(intent) {
                if bid.confidence < self.confidence_threshold {
                    continue;
                }
                match &best {
                    None => best = Some(bid),
                    Some(current_best) => {
                        // Higher selection score wins
                        if bid.selection_score() > current_best.selection_score() {
                            best = Some(bid);
                        } else if (bid.selection_score() - current_best.selection_score()).abs()
                            < 1e-10
                        {
                            // Tie-break: lowest energy cost
                            if bid.estimated_energy_uj < current_best.estimated_energy_uj {
                                best = Some(bid);
                            }
                        }
                    }
                }
            }
        }

        best
    }

    /// Dispatch a task: self-select and assign to the winning node.
    ///
    /// Returns the winning node ID, or None if escalation to LLM is needed.
    pub fn dispatch(&mut self, intent: &TaskIntent) -> Option<String> {
        let bid = self.select_for_task(intent)?;
        let node_id = bid.node_id.clone();

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.assign_task(&intent.task_id);
        }

        Some(node_id)
    }

    /// Substrate-aware dispatch: jointly optimize competence and hardware fit.
    ///
    /// Returns (node_id, substrate_bid) — the winning node and the substrate
    /// it selected for the task. Returns None if no node qualifies (→ LLM fallback).
    ///
    /// A GPU node with moderate competence outbids a CPU-only node with high
    /// competence for inference tasks. This is the honest answer.
    pub fn substrate_select(&self, intent: &TaskIntent) -> Option<SubstrateBid> {
        let mut best: Option<SubstrateBid> = None;

        for node in self.nodes.values() {
            if let Some(bid) = node.evaluate_task_with_substrate(intent, &self.substrate_profile) {
                match &best {
                    None => best = Some(bid),
                    Some(current) => {
                        if bid.composite_score > current.composite_score {
                            best = Some(bid);
                        } else if (bid.composite_score - current.composite_score).abs() < 1e-10 {
                            // Tie: lowest energy wins
                            if bid.estimated_energy_uj < current.estimated_energy_uj {
                                best = Some(bid);
                            }
                        }
                    }
                }
            }
        }

        best
    }

    /// Substrate-aware dispatch: select and assign.
    ///
    /// Returns the SubstrateBid (which includes node_id and selected substrate),
    /// or None if escalation to LLM is needed.
    pub fn substrate_dispatch(&mut self, intent: &TaskIntent) -> Option<SubstrateBid> {
        let bid = self.substrate_select(intent)?;
        let node_id = bid.node_id.clone();

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.assign_task(&intent.task_id);
        }

        Some(bid)
    }

    /// Record a status report from a node (observer pattern).
    ///
    /// This also updates the node's competence ledger based on outcome.
    pub fn report(&mut self, report: StatusReport) {
        // Update the node's competence ledger based on outcome
        if let Some(node) = self.nodes.get_mut(&report.node_id) {
            match (&report.action, &report.outcome) {
                (TaskAction::Acted, TaskOutcome::Success) => {
                    node.ledger.record_success(&report.domain, report.energy_uj);
                    node.complete_task(report.energy_uj);
                }
                (TaskAction::Acted, TaskOutcome::Failure) => {
                    node.ledger.record_failure(&report.domain, report.energy_uj);
                    node.complete_task(report.energy_uj);
                }
                (TaskAction::Halted, _) => {
                    node.ledger.record_halt(&report.domain);
                    node.complete_task(report.energy_uj);
                }
                (TaskAction::Escalated, _) => {
                    node.ledger.record_escalation(&report.domain);
                    node.complete_task(report.energy_uj);
                }
                (_, TaskOutcome::Pending) => {
                    // Task still in progress — no competence update yet
                }
            }
        }

        self.reports.push(report);
    }

    /// Penalize a node that failed to halt on a detected defect.
    ///
    /// This is the maximum penalty: reset to Level 1 in ALL domains.
    /// The system must make it safer to stop than to continue.
    pub fn penalize_missed_halt(&mut self, node_id: &str) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.ledger.reset_all();
        }
    }

    /// Get all status reports (for the observer).
    pub fn reports(&self) -> &[StatusReport] {
        &self.reports
    }

    /// Count of reports by domain.
    pub fn reports_by_domain(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for r in &self.reports {
            *counts.entry(r.domain.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Identify domains with insufficient coverage (no competent node exists).
    pub fn uncovered_domains(&self, min_level: DelegationLevel) -> Vec<DomainId> {
        // Collect all domains from all reports
        let mut all_domains: Vec<DomainId> = self
            .reports
            .iter()
            .map(|r| r.domain.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        all_domains.retain(|domain| {
            // Check if any node has the required level in this domain
            !self.nodes.values().any(|node| {
                node.ledger.delegation_level(domain) >= min_level
            })
        });

        all_domains
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of available (non-busy) nodes.
    pub fn available_count(&self) -> usize {
        self.nodes.values().filter(|n| n.available).count()
    }

    /// The promotion rate P(t) = deterministic / total.
    ///
    /// Measures what fraction of tasks were handled without LLM escalation.
    pub fn promotion_rate(&self) -> f64 {
        if self.reports.is_empty() {
            return 0.0;
        }
        let total = self.reports.len() as f64;
        let deterministic = self
            .reports
            .iter()
            .filter(|r| r.action == TaskAction::Acted)
            .count() as f64;
        deterministic / total
    }
}

impl Default for DispatchMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_intent(task_id: &str, domain: &str, level: DelegationLevel) -> TaskIntent {
        TaskIntent {
            task_id: task_id.into(),
            domain: domain.into(),
            purpose: format!("do {domain} work"),
            constraints: vec![],
            energy_budget_uj: 10_000,
            deadline: Some(Duration::from_secs(30)),
            required_level: level,
            task_class: None,
            similarity_hint: None,
        }
    }

    fn make_intent_with_class(
        task_id: &str,
        domain: &str,
        level: DelegationLevel,
        task_class: TaskClass,
    ) -> TaskIntent {
        TaskIntent {
            task_id: task_id.into(),
            domain: domain.into(),
            purpose: format!("do {domain} work"),
            constraints: vec![],
            energy_budget_uj: 10_000,
            deadline: Some(Duration::from_secs(30)),
            required_level: level,
            task_class: Some(task_class),
            similarity_hint: None,
        }
    }

    fn make_report(
        node_id: &str,
        task_id: &str,
        domain: &str,
        action: TaskAction,
        outcome: TaskOutcome,
    ) -> StatusReport {
        StatusReport {
            node_id: node_id.into(),
            task_id: task_id.into(),
            domain: domain.into(),
            delegation: DelegationLevel::Advise,
            action,
            outcome,
            energy_uj: 5000,
            duration: Duration::from_millis(100),
            artifacts: vec![],
            confidence: 0.9,
            timestamp_ns: 0,
        }
    }

    fn build_competent_node(node_id: &str, domain: &str, successes: u32) -> DispatchNode {
        let mut node = DispatchNode::new(node_id.into(), 1_000_000);
        for _ in 0..successes {
            node.ledger.record_success(domain, 1000);
        }
        node
    }

    #[test]
    fn test_halt_check_proceed() {
        let result = check_halt_conditions(100_000, 1000, 0.8, 0.3, false, true, false);
        assert_eq!(result, TransitionCheck::Proceed);
    }

    #[test]
    fn test_halt_check_budget() {
        let result = check_halt_conditions(500, 1000, 0.8, 0.3, false, true, false);
        assert_eq!(result, TransitionCheck::Halt(HaltReason::Budget));
    }

    #[test]
    fn test_halt_check_safety() {
        let result = check_halt_conditions(100_000, 1000, 0.8, 0.3, true, true, false);
        assert_eq!(result, TransitionCheck::Halt(HaltReason::Safety));
    }

    #[test]
    fn test_halt_check_integrity() {
        let result = check_halt_conditions(100_000, 1000, 0.8, 0.3, false, false, false);
        assert_eq!(result, TransitionCheck::Halt(HaltReason::Integrity));
    }

    #[test]
    fn test_halt_check_user_stop() {
        let result = check_halt_conditions(100_000, 1000, 0.8, 0.3, false, true, true);
        assert_eq!(result, TransitionCheck::Halt(HaltReason::UserSignal));
    }

    #[test]
    fn test_halt_check_escalate() {
        let result = check_halt_conditions(100_000, 1000, 0.2, 0.3, false, true, false);
        assert_eq!(result, TransitionCheck::Escalate);
    }

    #[test]
    fn test_halt_priority_budget_before_safety() {
        // Budget checked first
        let result = check_halt_conditions(500, 1000, 0.8, 0.3, true, false, true);
        assert_eq!(result, TransitionCheck::Halt(HaltReason::Budget));
    }

    #[test]
    fn test_empty_mesh_no_selection() {
        let mesh = DispatchMesh::new();
        let intent = make_intent("t1", "git", DelegationLevel::Tell);
        assert!(mesh.select_for_task(&intent).is_none());
    }

    #[test]
    fn test_self_selection_best_node_wins() {
        let mut mesh = DispatchMesh::new();

        // Node A: expert in git (50 successes)
        mesh.register_node(build_competent_node("node-a", "git", 50));
        // Node B: novice in git (5 successes)
        mesh.register_node(build_competent_node("node-b", "git", 5));

        let intent = make_intent("t1", "git", DelegationLevel::Tell);
        let bid = mesh.select_for_task(&intent).unwrap();

        assert_eq!(bid.node_id, "node-a");
    }

    #[test]
    fn test_self_selection_unknown_domain_no_bid() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_competent_node("node-a", "git", 50));

        // Task in unknown domain: no node has competence
        let intent = make_intent("t1", "quantum", DelegationLevel::Tell);
        assert!(mesh.select_for_task(&intent).is_none());
    }

    #[test]
    fn test_self_selection_below_required_level() {
        let mut mesh = DispatchMesh::new();
        // Node has only 3 successes → low delegation level
        mesh.register_node(build_competent_node("node-a", "deploy", 3));

        // Task requires Delegate level
        let intent = make_intent("t1", "deploy", DelegationLevel::Delegate);
        assert!(mesh.select_for_task(&intent).is_none());
    }

    #[test]
    fn test_dispatch_assigns_to_winner() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_competent_node("node-a", "search", 30));

        let intent = make_intent("t1", "search", DelegationLevel::Tell);
        let winner = mesh.dispatch(&intent).unwrap();

        assert_eq!(winner, "node-a");
        assert!(!mesh.node("node-a").unwrap().available);
        assert_eq!(
            mesh.node("node-a").unwrap().current_task.as_deref(),
            Some("t1")
        );
    }

    #[test]
    fn test_busy_node_cannot_bid() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_competent_node("node-a", "search", 30));

        // First dispatch succeeds
        let intent1 = make_intent("t1", "search", DelegationLevel::Tell);
        mesh.dispatch(&intent1).unwrap();

        // Second dispatch fails (node busy)
        let intent2 = make_intent("t2", "search", DelegationLevel::Tell);
        assert!(mesh.dispatch(&intent2).is_none());
    }

    #[test]
    fn test_report_updates_competence() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(DispatchNode::new("node-a".into(), 1_000_000));

        // Record success → should update node's ledger
        let report = make_report("node-a", "t1", "git", TaskAction::Acted, TaskOutcome::Success);
        mesh.report(report);

        let node = mesh.node("node-a").unwrap();
        assert_eq!(node.ledger.domain("git").unwrap().successes, 1);
        assert!(node.available); // completed task, available again
    }

    #[test]
    fn test_report_failure_updates_competence() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(DispatchNode::new("node-a".into(), 1_000_000));

        let report = make_report("node-a", "t1", "fs", TaskAction::Acted, TaskOutcome::Failure);
        mesh.report(report);

        let node = mesh.node("node-a").unwrap();
        assert_eq!(node.ledger.domain("fs").unwrap().failures, 1);
    }

    #[test]
    fn test_report_halt_no_penalty() {
        let mut mesh = DispatchMesh::new();
        let mut node = build_competent_node("node-a", "deploy", 20);
        let score_before = node.ledger.score("deploy");
        mesh.register_node(node);

        let report = make_report("node-a", "t1", "deploy", TaskAction::Halted, TaskOutcome::Failure);
        mesh.report(report);

        // Score should be unchanged (halt is neutral)
        let score_after = mesh.node("node-a").unwrap().ledger.score("deploy");
        assert!((score_before - score_after).abs() < 1e-10);
    }

    #[test]
    fn test_penalize_missed_halt() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_competent_node("node-a", "git", 50));
        mesh.register_node(build_competent_node("node-a-git2", "fs", 50));

        // Before penalty: high competence
        assert_eq!(
            mesh.node("node-a").unwrap().ledger.delegation_level("git"),
            DelegationLevel::Delegate
        );

        // Nuclear penalty
        mesh.penalize_missed_halt("node-a");

        // After: reset to Tell in all domains
        assert_eq!(
            mesh.node("node-a").unwrap().ledger.delegation_level("git"),
            DelegationLevel::Tell
        );
    }

    #[test]
    fn test_promotion_rate() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(DispatchNode::new("node-a".into(), 1_000_000));

        // 3 acted, 1 escalated → P = 0.75
        for i in 0..3 {
            mesh.report(make_report(
                "node-a",
                &format!("t{i}"),
                "git",
                TaskAction::Acted,
                TaskOutcome::Success,
            ));
        }
        mesh.report(make_report(
            "node-a",
            "t3",
            "git",
            TaskAction::Escalated,
            TaskOutcome::Pending,
        ));

        assert!((mesh.promotion_rate() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_promotion_rate_empty() {
        let mesh = DispatchMesh::new();
        assert!((mesh.promotion_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_uncovered_domains() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_competent_node("node-a", "git", 50));

        // Report tasks in git and quantum domains
        mesh.report(make_report("node-a", "t1", "git", TaskAction::Acted, TaskOutcome::Success));
        mesh.report(make_report("node-a", "t2", "quantum", TaskAction::Escalated, TaskOutcome::Pending));

        // Quantum is uncovered (no node at Advise level)
        let uncovered = mesh.uncovered_domains(DelegationLevel::Advise);
        assert!(uncovered.contains(&"quantum".to_string()));
        // Git is covered (node-a is at Delegate)
        assert!(!uncovered.contains(&"git".to_string()));
    }

    #[test]
    fn test_node_energy_depletion() {
        let mut mesh = DispatchMesh::new();
        let mut node = build_competent_node("node-a", "search", 30);
        node.energy_remaining_uj = 5000; // only 5000 µJ left
        mesh.register_node(node);

        // Task needs 10000 µJ: node can't afford it
        let intent = make_intent("t1", "search", DelegationLevel::Tell);
        assert!(mesh.select_for_task(&intent).is_none());
    }

    #[test]
    fn test_mesh_counts() {
        let mut mesh = DispatchMesh::new();
        assert_eq!(mesh.node_count(), 0);
        assert_eq!(mesh.available_count(), 0);

        mesh.register_node(build_competent_node("node-a", "git", 10));
        mesh.register_node(build_competent_node("node-b", "git", 10));
        assert_eq!(mesh.node_count(), 2);
        assert_eq!(mesh.available_count(), 2);

        // Assign one node
        let intent = make_intent("t1", "git", DelegationLevel::Tell);
        mesh.dispatch(&intent);
        assert_eq!(mesh.available_count(), 1);
    }

    #[test]
    fn test_task_bid_selection_score() {
        let bid = TaskBid {
            node_id: "n1".into(),
            task_id: "t1".into(),
            competence_score: 0.8,
            delegation_level: DelegationLevel::Advise,
            estimated_energy_uj: 5000,
            estimated_duration: None,
            confidence: 0.9,
        };
        assert!((bid.selection_score() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_reports_by_domain() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(DispatchNode::new("node-a".into(), 1_000_000));

        mesh.report(make_report("node-a", "t1", "git", TaskAction::Acted, TaskOutcome::Success));
        mesh.report(make_report("node-a", "t2", "git", TaskAction::Acted, TaskOutcome::Success));
        mesh.report(make_report("node-a", "t3", "fs", TaskAction::Acted, TaskOutcome::Success));

        let by_domain = mesh.reports_by_domain();
        assert_eq!(*by_domain.get("git").unwrap(), 2);
        assert_eq!(*by_domain.get("fs").unwrap(), 1);
    }

    #[test]
    fn test_intent_serde_roundtrip() {
        let intent = make_intent("t1", "search", DelegationLevel::Consult);
        let json = serde_json::to_string(&intent).unwrap();
        let parsed: TaskIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.task_id, "t1");
        assert_eq!(parsed.domain, "search");
        assert_eq!(parsed.required_level, DelegationLevel::Consult);
    }

    #[test]
    fn test_status_report_serde_roundtrip() {
        let report = make_report("node-a", "t1", "git", TaskAction::Acted, TaskOutcome::Success);
        let json = serde_json::to_string(&report).unwrap();
        let parsed: StatusReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.node_id, "node-a");
        assert_eq!(parsed.action, TaskAction::Acted);
        assert_eq!(parsed.outcome, TaskOutcome::Success);
    }

    // --- Substrate-aware dispatch tests ---

    fn build_gpu_node(node_id: &str, domain: &str, successes: u32) -> DispatchNode {
        let mut cap = SubstrateCapability::cpu_only();
        cap.add(ComputeSubstrate::GPU, "gpu-0".into());
        let mut node = DispatchNode::with_substrates(node_id.into(), 1_000_000, cap);
        for _ in 0..successes {
            node.ledger.record_success(domain, 1000);
        }
        node
    }

    #[test]
    fn test_substrate_gpu_beats_cpu_for_inference() {
        let mut mesh = DispatchMesh::new();

        // CPU-only node with HIGH domain competence (50 successes)
        mesh.register_node(build_competent_node("cpu-expert", "ml", 50));
        // GPU node with MODERATE domain competence (20 successes)
        mesh.register_node(build_gpu_node("gpu-moderate", "ml", 20));

        let intent = make_intent_with_class("t1", "ml", DelegationLevel::Tell, TaskClass::Inference);
        let bid = mesh.substrate_select(&intent).unwrap();

        // GPU should win despite lower competence — the honest answer
        assert_eq!(bid.node_id, "gpu-moderate");
        assert_eq!(bid.substrate, ComputeSubstrate::GPU);
    }

    #[test]
    fn test_substrate_cpu_wins_for_scalar() {
        let mut mesh = DispatchMesh::new();

        mesh.register_node(build_competent_node("cpu-node", "hash", 30));
        mesh.register_node(build_gpu_node("gpu-node", "hash", 30));

        let intent = make_intent_with_class("t1", "hash", DelegationLevel::Tell, TaskClass::Scalar);
        let bid = mesh.substrate_select(&intent).unwrap();

        // For scalar work, CPU is optimal (GPU adds dispatch overhead)
        assert_eq!(bid.substrate, ComputeSubstrate::CPU);
    }

    #[test]
    fn test_substrate_dispatch_assigns_winner() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_gpu_node("gpu-node", "vision", 30));

        let intent =
            make_intent_with_class("t1", "vision", DelegationLevel::Tell, TaskClass::TensorOp);
        let bid = mesh.substrate_dispatch(&intent).unwrap();

        assert_eq!(bid.node_id, "gpu-node");
        assert!(!mesh.node("gpu-node").unwrap().available);
    }

    #[test]
    fn test_substrate_no_competence_no_bid() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_gpu_node("gpu-node", "vision", 30));

        // Different domain — no competence
        let intent =
            make_intent_with_class("t1", "audio", DelegationLevel::Tell, TaskClass::Inference);
        assert!(mesh.substrate_select(&intent).is_none());
    }

    #[test]
    fn test_substrate_bid_includes_energy_estimate() {
        let mut mesh = DispatchMesh::new();
        mesh.register_node(build_gpu_node("gpu-node", "ml", 30));

        let intent =
            make_intent_with_class("t1", "ml", DelegationLevel::Tell, TaskClass::Inference);
        let bid = mesh.substrate_select(&intent).unwrap();

        // GPU inference estimate from default profile: 10,000 µJ
        assert_eq!(bid.estimated_energy_uj, 10_000);
        // GPU proximity to optimal for inference: optimal (Photonic=200) / GPU (10000) = 0.02
        assert!(bid.proximity_to_optimal > 0.0 && bid.proximity_to_optimal <= 1.0);
    }

    #[test]
    fn test_substrate_composite_score_dominates() {
        let mut mesh = DispatchMesh::new();

        // CPU node: very high competence
        mesh.register_node(build_competent_node("cpu-star", "embedding", 100));
        // GPU node: low competence but GPU for tensor ops
        mesh.register_node(build_gpu_node("gpu-rookie", "embedding", 5));

        let intent = make_intent_with_class(
            "t1",
            "embedding",
            DelegationLevel::Tell,
            TaskClass::TensorOp,
        );
        let bid = mesh.substrate_select(&intent).unwrap();

        // GPU wins: tensor ops on GPU are 1000× more efficient
        // GPU composite: low_score + log10(1000) = low + 3.0
        // CPU composite: high_score + log10(1.0) = high + 0.0
        assert_eq!(bid.substrate, ComputeSubstrate::GPU);
    }
}
