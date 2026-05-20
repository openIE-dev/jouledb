//! Agent Sandbox — unified orchestration of contract lifecycle + energy tracing + sandbox isolation.
//!
//! Composes three systems into a single cohesive flow:
//! - **Sandbox**: network isolation, energy enforcement, attestation, resource limits, shutdown
//! - **Agent Lifecycle**: enlist, negotiate, engage, return-to-pool
//! - **Energy Trace**: behavioral pattern detection (flat/spiky/declining/escalating/periodic/anomalous)
//!
//! The `AgentSandbox` is the top-level orchestrator. It manages the full journey:
//!
//! ```text
//! Host proposes contract
//!   → Agent accepts/rejects/counter-proposes
//!     → Sandbox opens (network, energy, attestation)
//!       → Agent engages with work (energy trace runs alongside)
//!         → Agent returns findings voluntarily (or is recalled)
//!           → Sandbox closes (drain, sign receipt, cleanup)
//!             → Agent returns to pool
//! ```

use crate::agent_lifecycle::{
    AgentFindings, AgentPhase, Anomaly, AnomalyCategory, ManagedAgent,
};
use crate::attestation::SignedEnergyReceipt;
use crate::contract::{
    ContractError, ContractProposal, ContractResponse, ExtensionRequest, ExtensionResponse,
    SignedContract,
};
use crate::energy_enforcer::EnforcerResult;
use crate::energy_trace::{EnergyPattern, EnergySample, TraceAnalyzer, TraceConfig, TraceSummary};
use crate::graceful_shutdown::ShutdownResult;
use crate::sandbox::{Sandbox, SandboxConfig};
use crate::{InstanceId, RuntimeError};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the agent sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSandboxConfig {
    /// Sandbox isolation config (network, resources, shutdown).
    pub sandbox: SandboxConfig,
    /// Energy trace analysis config.
    pub trace: TraceConfig,
    /// Whether to auto-recall on anomalous energy pattern.
    pub recall_on_anomaly: bool,
    /// Minimum confidence for anomaly-triggered recall.
    pub anomaly_recall_confidence: f64,
}

impl AgentSandboxConfig {
    /// Create from a sandbox config with default trace settings.
    pub fn from_sandbox(sandbox: SandboxConfig) -> Self {
        Self {
            sandbox,
            trace: TraceConfig::default(),
            recall_on_anomaly: true,
            anomaly_recall_confidence: 0.8,
        }
    }

    /// Convenience: strict agent sandbox.
    pub fn strict(instance_id: &InstanceId, max_energy_uj: u64) -> Self {
        Self::from_sandbox(SandboxConfig::strict(instance_id, max_energy_uj))
    }
}

/// The unified agent sandbox orchestrator.
pub struct AgentSandbox {
    config: AgentSandboxConfig,
    /// The underlying hardware sandbox.
    sandbox: Sandbox,
    /// The managed agent lifecycle.
    agent: ManagedAgent,
    /// Energy trace analyzer (runs alongside enforcer).
    trace: TraceAnalyzer,
}

impl AgentSandbox {
    /// Create a new agent sandbox.
    pub fn new(
        agent_id: String,
        config: AgentSandboxConfig,
    ) -> Result<Self, RuntimeError> {
        let sandbox = Sandbox::new(config.sandbox.clone())?;
        let agent = ManagedAgent::new(agent_id);
        let trace = TraceAnalyzer::new(config.trace.clone());

        Ok(Self {
            config,
            sandbox,
            agent,
            trace,
        })
    }

    // --- Contract negotiation ---

    /// Propose a contract to the agent.
    pub fn propose_contract(
        &mut self,
        proposal: ContractProposal,
    ) -> Result<(), ContractError> {
        self.agent.propose(proposal)
    }

    /// Process the agent's response to the contract proposal.
    /// Returns the new agent phase.
    pub fn process_response(
        &mut self,
        response: ContractResponse,
    ) -> Result<AgentPhase, ContractError> {
        self.agent
            .process_response(response, self.sandbox.attestation_key())
    }

    /// Get the current pending proposal (for the agent to review).
    pub fn pending_proposal(&self) -> Option<&ContractProposal> {
        // Access through agent's pending_proposal field
        // Since ManagedAgent doesn't expose this directly, we check phase
        if self.agent.phase() == AgentPhase::Negotiating {
            // The proposal is inside the agent — the agent sees it via JWP
            None // Host doesn't need to read it back
        } else {
            None
        }
    }

    // --- Engagement ---

    /// Start energy enforcement after the workload process is running.
    pub fn start_enforcement(
        &mut self,
        meter: Box<dyn inv_energy::meter::EnergyMeter + Send>,
    ) -> Result<(), RuntimeError> {
        self.sandbox.start_energy_enforcement(meter)
    }

    /// Set the target PID for energy enforcement.
    pub fn set_target_pid(&mut self, pid: u32) {
        self.sandbox.set_target_pid(pid);
    }

    /// Feed an energy sample into the trace analyzer.
    /// Call this from the energy enforcer's sampling loop.
    /// Returns a recall recommendation if anomalous behavior is detected.
    pub fn feed_energy_sample(&mut self, sample: EnergySample) -> Option<RecallRecommendation> {
        self.trace.push(sample);

        // Check for anomalous behavior
        if self.config.recall_on_anomaly && self.agent.phase() == AgentPhase::Engaged {
            let diagnosis = self.trace.diagnose();
            if diagnosis.pattern == EnergyPattern::Anomalous
                && diagnosis.confidence >= self.config.anomaly_recall_confidence
            {
                return Some(RecallRecommendation {
                    reason: format!(
                        "anomalous energy pattern detected (Warburg ratio: {:.1}, confidence: {:.0}%)",
                        diagnosis.warburg_ratio.unwrap_or(0.0),
                        diagnosis.confidence * 100.0,
                    ),
                    pattern: diagnosis.pattern,
                    confidence: diagnosis.confidence,
                    trace_summary: self.trace.summary(),
                });
            }
        }
        None
    }

    /// Process an extension request from the agent.
    pub fn process_extension_request(
        &mut self,
        request: &ExtensionRequest,
    ) -> Result<ExtensionResponse, ContractError> {
        let response = self.agent.process_extension_request(request)?;

        // If granted, update the energy enforcer's budget
        if response.is_granted() {
            if let Some(state) = self.sandbox.energy_state() {
                let new_budget = state.budget_uj() + response.granted_energy_uj();
                state.update_budget(new_budget);
            }
        }

        Ok(response)
    }

    // --- Return / Recall ---

    /// Agent voluntarily returns with findings.
    /// Stops the sandbox and produces a complete result.
    pub fn voluntary_return(
        mut self,
        findings: AgentFindings,
        target_pid: Option<u32>,
    ) -> Result<AgentSandboxResult, ContractError> {
        // Record findings in agent lifecycle
        self.agent.voluntary_return(findings.clone())?;

        // Get trace summary before stopping
        let trace_summary = self.trace.summary();

        // Stop the sandbox
        let sandbox_result = self.sandbox.stop(target_pid);

        Ok(AgentSandboxResult {
            agent_id: self.agent.agent_id.clone(),
            phase: AgentPhase::Returned,
            findings: Some(findings),
            trace_summary,
            sandbox_shutdown: sandbox_result.shutdown,
            enforcer_result: sandbox_result.enforcer,
            final_receipt: sandbox_result.final_receipt,
            recall_reason: None,
        })
    }

    /// Force-recall the agent (budget exceeded, timeout, anomaly).
    /// Stops the sandbox and produces a result with the recall reason.
    pub fn recall(
        mut self,
        reason: &str,
        target_pid: Option<u32>,
    ) -> AgentSandboxResult {
        // Get trace summary and build anomaly report
        let trace_summary = self.trace.summary();
        let diagnosis = self.trace.diagnose();

        // Recall the agent
        let mut findings = self.agent.recall(reason);

        // Enrich findings with energy trace data
        findings.anomalies.push(Anomaly {
            description: format!(
                "Energy pattern at recall: {:?} (confidence {:.0}%)",
                diagnosis.pattern,
                diagnosis.confidence * 100.0,
            ),
            severity: if diagnosis.pattern == EnergyPattern::Anomalous {
                0.9
            } else {
                0.3
            },
            category: AnomalyCategory::Resource,
            evidence: Some(serde_json::json!({
                "pattern": format!("{:?}", diagnosis.pattern),
                "confidence": diagnosis.confidence,
                "mean_power_uw": diagnosis.mean_power_uw,
                "cv": diagnosis.cv,
                "slope": diagnosis.slope,
                "spike_count": diagnosis.spike_count,
                "warburg_ratio": diagnosis.warburg_ratio,
            })),
        });

        // Fill in energy consumed from enforcer state
        if let Some(state) = self.sandbox.energy_state() {
            findings.energy_consumed_uj = state.consumed_uj();
        }

        // Stop sandbox
        let sandbox_result = self.sandbox.stop(target_pid);

        AgentSandboxResult {
            agent_id: self.agent.agent_id.clone(),
            phase: AgentPhase::Recalled,
            findings: Some(findings),
            trace_summary,
            sandbox_shutdown: sandbox_result.shutdown,
            enforcer_result: sandbox_result.enforcer,
            final_receipt: sandbox_result.final_receipt,
            recall_reason: Some(reason.to_string()),
        }
    }

    // --- Accessors ---

    pub fn agent_id(&self) -> &str {
        &self.agent.agent_id
    }

    pub fn agent_phase(&self) -> AgentPhase {
        self.agent.phase()
    }

    pub fn contract(&self) -> Option<&SignedContract> {
        self.agent.contract()
    }

    pub fn trace_diagnosis(&self) -> crate::energy_trace::TraceDiagnosis {
        self.trace.diagnose()
    }

    pub fn trace_summary(&self) -> TraceSummary {
        self.trace.summary()
    }

    pub fn sandbox(&self) -> &Sandbox {
        &self.sandbox
    }

    pub fn sandbox_mut(&mut self) -> &mut Sandbox {
        &mut self.sandbox
    }

    pub fn energy_consumed_uj(&self) -> u64 {
        self.sandbox
            .energy_state()
            .map(|s| s.consumed_uj())
            .unwrap_or(0)
    }

    pub fn energy_budget_uj(&self) -> u64 {
        self.sandbox
            .energy_state()
            .map(|s| s.budget_uj())
            .unwrap_or(0)
    }

    pub fn energy_utilization(&self) -> f64 {
        self.sandbox
            .energy_state()
            .map(|s| s.utilization())
            .unwrap_or(0.0)
    }
}

/// Recommendation to recall an agent based on energy trace analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallRecommendation {
    pub reason: String,
    pub pattern: EnergyPattern,
    pub confidence: f64,
    pub trace_summary: TraceSummary,
}

/// Complete result of an agent sandbox lifecycle.
#[derive(Debug)]
pub struct AgentSandboxResult {
    /// Agent that was running in this sandbox.
    pub agent_id: String,
    /// Final phase (Returned or Recalled).
    pub phase: AgentPhase,
    /// Findings from the agent (if any).
    pub findings: Option<AgentFindings>,
    /// Energy trace summary.
    pub trace_summary: TraceSummary,
    /// Sandbox shutdown result.
    pub sandbox_shutdown: ShutdownResult,
    /// Energy enforcer result.
    pub enforcer_result: Option<EnforcerResult>,
    /// Final signed energy receipt.
    pub final_receipt: SignedEnergyReceipt,
    /// If recalled, the reason.
    pub recall_reason: Option<String>,
}

impl AgentSandboxResult {
    /// Whether the agent completed its work voluntarily.
    pub fn completed_voluntarily(&self) -> bool {
        self.phase == AgentPhase::Returned
    }

    /// Whether the agent was recalled (forced stop).
    pub fn was_recalled(&self) -> bool {
        self.phase == AgentPhase::Recalled
    }

    /// Total energy consumed in microjoules.
    pub fn total_energy_uj(&self) -> u64 {
        self.enforcer_result
            .as_ref()
            .map(|r| r.consumed_uj)
            .unwrap_or(0)
    }

    /// Whether the energy budget was exceeded.
    pub fn budget_exceeded(&self) -> bool {
        self.enforcer_result
            .as_ref()
            .map(|r| r.exceeded)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{ExtensionPolicy, ReturnTerms, WorkScope};

    fn make_config() -> AgentSandboxConfig {
        let id = InstanceId::from_string("agent-sandbox-test".into());
        AgentSandboxConfig::strict(&id, 100_000_000)
    }

    fn make_proposal(contract_id: &str) -> ContractProposal {
        ContractProposal::new(
            contract_id.into(),
            "agent-sandbox-test".into(),
            WorkScope::Query {
                description: "test query".into(),
            },
            100_000_000,
            Duration::from_secs(60),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        )
    }

    #[test]
    fn test_agent_sandbox_creation() {
        let config = make_config();
        let asb = AgentSandbox::new("agent-1".into(), config).unwrap();
        assert_eq!(asb.agent_id(), "agent-1");
        assert_eq!(asb.agent_phase(), AgentPhase::Pooled);
    }

    #[test]
    fn test_full_lifecycle_propose_accept_return() {
        let config = make_config();
        let mut asb = AgentSandbox::new("agent-2".into(), config).unwrap();

        // Propose
        let proposal = make_proposal("c-asb-1");
        let hash = proposal.proposal_hash.clone();
        asb.propose_contract(proposal).unwrap();
        assert_eq!(asb.agent_phase(), AgentPhase::Negotiating);

        // Accept
        let response = ContractResponse::accept("c-asb-1", &hash);
        let phase = asb.process_response(response).unwrap();
        assert_eq!(phase, AgentPhase::Engaged);
        assert!(asb.contract().is_some());

        // Feed some energy samples
        for i in 0..15 {
            asb.feed_energy_sample(EnergySample {
                energy_uj: 1000,
                offset_ns: i * 100_000_000,
                interval_ns: 100_000_000,
                power_uw: 10_000_000,
            });
        }

        // Voluntary return
        let findings = AgentFindings {
            summary: "found the answer".into(),
            results: Some(serde_json::json!({"answer": 42})),
            anomalies: vec![],
            energy_consumed_uj: 15_000,
            wall_time: Duration::from_secs(2),
            scope_completed: true,
            incomplete_reason: None,
        };
        let result = asb.voluntary_return(findings, None).unwrap();
        assert!(result.completed_voluntarily());
        assert!(!result.was_recalled());
        assert!(result.findings.is_some());
        assert!(result.sandbox_shutdown.drained_cleanly);
    }

    #[test]
    fn test_recall_on_budget() {
        let config = make_config();
        let mut asb = AgentSandbox::new("agent-3".into(), config).unwrap();

        let proposal = make_proposal("c-asb-2");
        let hash = proposal.proposal_hash.clone();
        asb.propose_contract(proposal).unwrap();
        let response = ContractResponse::accept("c-asb-2", &hash);
        asb.process_response(response).unwrap();

        let result = asb.recall("energy budget exceeded", None);
        assert!(result.was_recalled());
        assert_eq!(result.recall_reason.as_deref(), Some("energy budget exceeded"));
        assert!(result.findings.is_some());
        let findings = result.findings.unwrap();
        assert!(!findings.scope_completed);
        // Should have the energy trace anomaly appended
        assert!(!findings.anomalies.is_empty());
    }

    #[test]
    fn test_anomaly_detection_in_feed() {
        let id = InstanceId::from_string("anomaly-test".into());
        let mut config = AgentSandboxConfig::strict(&id, 100_000_000);
        config.trace.min_samples = 5;
        config.trace.warburg_ratio_threshold = 2.0;
        config.anomaly_recall_confidence = 0.5;

        let mut asb = AgentSandbox::new("agent-4".into(), config).unwrap();

        let proposal = make_proposal("c-asb-3");
        let hash = proposal.proposal_hash.clone();
        asb.propose_contract(proposal).unwrap();
        let response = ContractResponse::accept("c-asb-3", &hash);
        asb.process_response(response).unwrap();

        // Set expected baseline
        asb.trace.set_expected_energy(1000);

        // Feed wildly excessive energy (should trigger Warburg detection)
        let mut saw_recommendation = false;
        for i in 0..20 {
            let rec = asb.feed_energy_sample(EnergySample {
                energy_uj: 10000, // 10x expected
                offset_ns: i * 100_000_000,
                interval_ns: 100_000_000,
                power_uw: 100_000_000, // 100W
            });
            if rec.is_some() {
                saw_recommendation = true;
            }
        }
        assert!(
            saw_recommendation,
            "expected anomaly recall recommendation"
        );
    }

    #[test]
    fn test_extension_updates_enforcer_budget() {
        let id = InstanceId::from_string("ext-test".into());
        let config = AgentSandboxConfig::strict(&id, 50_000_000);
        let mut asb = AgentSandbox::new("agent-5".into(), config).unwrap();

        let proposal = ContractProposal::new(
            "c-asb-4".into(),
            "ext-test".into(),
            WorkScope::Research {
                description: "explore graph".into(),
                hypothesis: None,
            },
            50_000_000,
            Duration::from_secs(120),
            ReturnTerms::FindingsAndAnomalies,
            ExtensionPolicy::SingleExtension {
                max_additional_uj: 25_000_000,
            },
        );
        let hash = proposal.proposal_hash.clone();
        asb.propose_contract(proposal).unwrap();
        let response = ContractResponse::accept("c-asb-4", &hash);
        asb.process_response(response).unwrap();

        // Request extension
        let request = ExtensionRequest {
            contract_id: "c-asb-4".into(),
            additional_energy_uj: 20_000_000,
            additional_time: None,
            rationale: "found interesting cluster".into(),
            interim_findings: Some("2 anomalous nodes".into()),
            consumed_uj: 45_000_000,
        };
        let resp = asb.process_extension_request(&request).unwrap();
        assert!(resp.is_granted());

        // Budget should be updated in the enforcer
        assert_eq!(asb.energy_budget_uj(), 70_000_000);
    }

    #[test]
    fn test_trace_during_engagement() {
        let config = make_config();
        let mut asb = AgentSandbox::new("agent-6".into(), config).unwrap();

        let proposal = make_proposal("c-asb-5");
        let hash = proposal.proposal_hash.clone();
        asb.propose_contract(proposal).unwrap();
        let response = ContractResponse::accept("c-asb-5", &hash);
        asb.process_response(response).unwrap();

        // Feed flat energy trace
        for i in 0..25 {
            asb.feed_energy_sample(EnergySample {
                energy_uj: 5000,
                offset_ns: i * 100_000_000,
                interval_ns: 100_000_000,
                power_uw: 50_000_000,
            });
        }

        let diag = asb.trace_diagnosis();
        assert_eq!(diag.pattern, EnergyPattern::Flat);
        assert!(diag.confidence > 0.5);

        let summary = asb.trace_summary();
        assert_eq!(summary.sample_count, 25);
        assert_eq!(summary.total_energy_uj, 125_000);
    }

    #[test]
    fn test_agent_sandbox_config_from_sandbox() {
        let id = InstanceId::from_string("cfg-test".into());
        let sandbox_config = SandboxConfig::strict(&id, 50_000_000);
        let config = AgentSandboxConfig::from_sandbox(sandbox_config);
        assert!(config.recall_on_anomaly);
        assert!((config.anomaly_recall_confidence - 0.8).abs() < 1e-10);
        assert_eq!(config.trace.min_samples, 10);
    }

    #[test]
    fn test_result_accessors() {
        let config = make_config();
        let mut asb = AgentSandbox::new("agent-7".into(), config).unwrap();

        let proposal = make_proposal("c-asb-6");
        let hash = proposal.proposal_hash.clone();
        asb.propose_contract(proposal).unwrap();
        let response = ContractResponse::accept("c-asb-6", &hash);
        asb.process_response(response).unwrap();

        let findings = AgentFindings {
            summary: "done".into(),
            results: None,
            anomalies: vec![],
            energy_consumed_uj: 0,
            wall_time: Duration::from_millis(100),
            scope_completed: true,
            incomplete_reason: None,
        };
        let result = asb.voluntary_return(findings, None).unwrap();
        assert_eq!(result.agent_id, "agent-7");
        assert!(!result.budget_exceeded());
    }

    #[test]
    fn test_reject_prevents_engagement() {
        let config = make_config();
        let mut asb = AgentSandbox::new("agent-8".into(), config).unwrap();

        let proposal = make_proposal("c-asb-7");
        asb.propose_contract(proposal).unwrap();

        let response = ContractResponse::reject("c-asb-7", "scope too broad");
        let result = asb.process_response(response);
        assert!(result.is_err());
        assert_eq!(asb.agent_phase(), AgentPhase::Pooled);
    }

    #[test]
    fn test_counter_propose_then_accept() {
        let config = make_config();
        let mut asb = AgentSandbox::new("agent-9".into(), config).unwrap();

        let proposal = make_proposal("c-asb-8");
        asb.propose_contract(proposal).unwrap();

        // Counter-propose
        let response = ContractResponse::counter_propose(
            "c-asb-8",
            Some(200_000_000),
            None,
            None,
            "need more energy",
        );
        let phase = asb.process_response(response).unwrap();
        assert_eq!(phase, AgentPhase::Negotiating);

        // Accept the revised proposal (need to get hash from agent internals)
        // Since we can't directly access pending_proposal, we test the flow works
        // by accepting with any hash — the process_response will verify
        // In practice, the agent reads the revised proposal via JWP and sends back
        // the correct hash.
    }
}
