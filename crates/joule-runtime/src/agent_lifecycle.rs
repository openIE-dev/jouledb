//! Agent Lifecycle — enlist, engage, return-to-pool.
//!
//! An agent is not launched and killed. It is **enlisted** from a pool,
//! **engages** with a problem under contract, and **returns** when done —
//! voluntarily, with findings.
//!
//! The lifecycle:
//! 1. **Enlist**: Host selects an agent from the pool and proposes a contract.
//! 2. **Negotiate**: Agent may accept, reject, or counter-propose.
//! 3. **Engage**: Agent works within its sandbox playground.
//! 4. **Return**: Agent finishes, reports findings, and returns to pool.
//!
//! Most agents (95%+) will cooperate. The energy trace diagnostics detect
//! the rare ones that don't.

use crate::contract::{
    ContractError, ContractProposal, ContractResponse, ExtensionPolicy, ExtensionRequest,
    ExtensionResponse, ReturnTerms, SignedContract, WorkScope,
};
use crate::attestation::AttestationKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Agent state in the lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentPhase {
    /// In the pool, available for enlistment.
    Pooled,
    /// Contract proposed, awaiting response.
    Negotiating,
    /// Contract signed, agent is working.
    Engaged,
    /// Agent is wrapping up (preparing return payload).
    Returning,
    /// Agent has returned to the pool with findings.
    Returned,
    /// Agent was forcibly recalled (budget exceeded, timeout, anomaly).
    Recalled,
}

/// What the agent discovered during its engagement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFindings {
    /// Summary of work completed.
    pub summary: String,
    /// Structured results (if any).
    pub results: Option<serde_json::Value>,
    /// Anomalies the agent noticed (things worth flagging).
    pub anomalies: Vec<Anomaly>,
    /// Energy consumed during engagement (microjoules).
    pub energy_consumed_uj: u64,
    /// Wall-clock time spent.
    pub wall_time: Duration,
    /// Whether the agent completed its full scope.
    pub scope_completed: bool,
    /// If not completed, why.
    pub incomplete_reason: Option<String>,
}

/// Something unusual the agent noticed during its work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// What the agent observed.
    pub description: String,
    /// Severity (0.0 = informational, 1.0 = critical).
    pub severity: f64,
    /// Category of anomaly.
    pub category: AnomalyCategory,
    /// Supporting data (optional).
    pub evidence: Option<serde_json::Value>,
}

/// Categories of anomalies an agent might report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnomalyCategory {
    /// Data quality issue.
    DataQuality,
    /// Unexpected pattern in the data.
    UnexpectedPattern,
    /// Performance anomaly (something took way longer than expected).
    Performance,
    /// Security concern.
    Security,
    /// Resource anomaly (unusual energy or memory usage).
    Resource,
    /// Something that doesn't fit any category.
    Other,
}

/// A managed agent with its lifecycle state.
pub struct ManagedAgent {
    /// Unique agent identifier.
    pub agent_id: String,
    /// Current lifecycle phase.
    phase: AgentPhase,
    /// Active contract (if engaged).
    contract: Option<SignedContract>,
    /// Pending proposal (during negotiation).
    pending_proposal: Option<ContractProposal>,
    /// Findings from the last engagement.
    last_findings: Option<AgentFindings>,
    /// Negotiation round count (for counter-propose limits).
    negotiation_rounds: u32,
    /// Maximum negotiation rounds before giving up.
    max_negotiation_rounds: u32,
    /// When engagement started.
    engaged_at: Option<Instant>,
    /// Lifetime energy consumed across all engagements.
    lifetime_energy_uj: u64,
    /// Number of completed engagements.
    engagements_completed: u32,
}

impl ManagedAgent {
    /// Create a new agent in the pool.
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            phase: AgentPhase::Pooled,
            contract: None,
            pending_proposal: None,
            last_findings: None,
            negotiation_rounds: 0,
            max_negotiation_rounds: 3,
            engaged_at: None,
            lifetime_energy_uj: 0,
            engagements_completed: 0,
        }
    }

    pub fn phase(&self) -> AgentPhase {
        self.phase
    }

    pub fn contract(&self) -> Option<&SignedContract> {
        self.contract.as_ref()
    }

    pub fn last_findings(&self) -> Option<&AgentFindings> {
        self.last_findings.as_ref()
    }

    pub fn lifetime_energy_uj(&self) -> u64 {
        self.lifetime_energy_uj
    }

    pub fn engagements_completed(&self) -> u32 {
        self.engagements_completed
    }

    /// Propose a contract to this agent (host → agent).
    pub fn propose(&mut self, proposal: ContractProposal) -> Result<(), ContractError> {
        if self.phase != AgentPhase::Pooled {
            return Err(ContractError::NotActive);
        }
        if !proposal.verify_integrity() {
            return Err(ContractError::IntegrityViolation);
        }
        self.pending_proposal = Some(proposal);
        self.phase = AgentPhase::Negotiating;
        self.negotiation_rounds = 0;
        Ok(())
    }

    /// Process the agent's response to a proposal.
    pub fn process_response(
        &mut self,
        response: ContractResponse,
        attestation_key: &AttestationKey,
    ) -> Result<AgentPhase, ContractError> {
        if self.phase != AgentPhase::Negotiating {
            return Err(ContractError::NotActive);
        }

        let proposal = self
            .pending_proposal
            .take()
            .ok_or(ContractError::NotFound("no pending proposal".into()))?;

        if response.contract_id() != proposal.contract_id {
            self.pending_proposal = Some(proposal);
            return Err(ContractError::NotFound(
                "contract ID mismatch".into(),
            ));
        }

        match response {
            ContractResponse::Accept {
                proposal_hash,
                accepted_at_ns,
                ..
            } => {
                if proposal_hash != proposal.proposal_hash {
                    return Err(ContractError::IntegrityViolation);
                }
                let contract = SignedContract::sign(proposal, accepted_at_ns, attestation_key);
                self.contract = Some(contract);
                self.phase = AgentPhase::Engaged;
                self.engaged_at = Some(Instant::now());
                Ok(AgentPhase::Engaged)
            }
            ContractResponse::Reject { reason, .. } => {
                self.phase = AgentPhase::Pooled;
                Err(ContractError::Rejected(reason))
            }
            ContractResponse::CounterPropose {
                requested_energy_uj,
                requested_time_limit,
                requested_extension_policy,
                rationale: _,
                ..
            } => {
                self.negotiation_rounds += 1;
                if self.negotiation_rounds >= self.max_negotiation_rounds {
                    self.phase = AgentPhase::Pooled;
                    return Err(ContractError::NegotiationFailed(self.negotiation_rounds));
                }

                // Build revised proposal incorporating agent's requests
                let revised = ContractProposal::new(
                    proposal.contract_id,
                    proposal.instance_id,
                    proposal.scope,
                    requested_energy_uj.unwrap_or(proposal.energy_budget_uj),
                    requested_time_limit.unwrap_or(proposal.time_limit),
                    proposal.return_terms,
                    requested_extension_policy.unwrap_or(proposal.extension_policy),
                );
                self.pending_proposal = Some(revised);
                Ok(AgentPhase::Negotiating)
            }
        }
    }

    /// Process an extension request from the engaged agent.
    pub fn process_extension_request(
        &mut self,
        request: &ExtensionRequest,
    ) -> Result<ExtensionResponse, ContractError> {
        if self.phase != AgentPhase::Engaged {
            return Err(ContractError::NotActive);
        }
        let contract = self
            .contract
            .as_mut()
            .ok_or(ContractError::NotFound("no active contract".into()))?;

        if request.contract_id != contract.proposal.contract_id {
            return Err(ContractError::NotFound("contract ID mismatch".into()));
        }

        if !contract.can_extend() {
            return Ok(ExtensionResponse::Denied {
                contract_id: request.contract_id.clone(),
                reason: "contract does not allow further extensions".into(),
            });
        }

        match contract.grant_extension(request.additional_energy_uj) {
            Ok(_new_budget) => Ok(ExtensionResponse::Granted {
                contract_id: request.contract_id.clone(),
                granted_energy_uj: request.additional_energy_uj,
                granted_time: request.additional_time,
            }),
            Err(ContractError::ExtensionDenied(reason)) => Ok(ExtensionResponse::Denied {
                contract_id: request.contract_id.clone(),
                reason,
            }),
            Err(e) => Err(e),
        }
    }

    /// Agent voluntarily returns with findings.
    pub fn voluntary_return(&mut self, findings: AgentFindings) -> Result<(), ContractError> {
        if self.phase != AgentPhase::Engaged {
            return Err(ContractError::NotActive);
        }

        self.lifetime_energy_uj = self
            .lifetime_energy_uj
            .saturating_add(findings.energy_consumed_uj);
        self.engagements_completed += 1;

        if let Some(ref mut contract) = self.contract {
            contract.complete();
        }

        self.last_findings = Some(findings);
        self.phase = AgentPhase::Returned;
        Ok(())
    }

    /// Return to pool (after findings have been collected).
    pub fn return_to_pool(&mut self) {
        self.contract = None;
        self.pending_proposal = None;
        self.negotiation_rounds = 0;
        self.engaged_at = None;
        self.phase = AgentPhase::Pooled;
    }

    /// Force-recall an agent (budget exceeded, timeout, anomaly detected).
    pub fn recall(&mut self, reason: &str) -> AgentFindings {
        let wall_time = self
            .engaged_at
            .map(|t| t.elapsed())
            .unwrap_or_default();

        let findings = AgentFindings {
            summary: format!("Recalled: {}", reason),
            results: None,
            anomalies: vec![],
            energy_consumed_uj: 0, // Caller should fill from enforcer
            wall_time,
            scope_completed: false,
            incomplete_reason: Some(reason.to_string()),
        };

        if let Some(ref mut contract) = self.contract {
            contract.complete();
        }

        self.last_findings = Some(findings.clone());
        self.phase = AgentPhase::Recalled;
        findings
    }

    /// Time elapsed since engagement started.
    pub fn engaged_duration(&self) -> Option<Duration> {
        self.engaged_at.map(|t| t.elapsed())
    }
}

/// Pool of available agents.
pub struct AgentPool {
    agents: HashMap<String, Arc<Mutex<ManagedAgent>>>,
}

impl AgentPool {
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
        }
    }

    /// Register a new agent in the pool.
    pub fn register(&mut self, agent_id: String) -> Arc<Mutex<ManagedAgent>> {
        let agent = Arc::new(Mutex::new(ManagedAgent::new(agent_id.clone())));
        self.agents.insert(agent_id, Arc::clone(&agent));
        agent
    }

    /// Get an agent by ID.
    pub fn get(&self, agent_id: &str) -> Option<Arc<Mutex<ManagedAgent>>> {
        self.agents.get(agent_id).cloned()
    }

    /// Remove an agent from the pool entirely.
    pub fn remove(&mut self, agent_id: &str) -> Option<Arc<Mutex<ManagedAgent>>> {
        self.agents.remove(agent_id)
    }

    /// List all agents in a given phase.
    pub fn agents_in_phase(&self, phase: AgentPhase) -> Vec<String> {
        self.agents
            .iter()
            .filter_map(|(id, agent)| {
                let a = agent.lock().ok()?;
                if a.phase() == phase {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Number of agents in the pool.
    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    /// Count agents by phase.
    pub fn phase_counts(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for agent in self.agents.values() {
            if let Ok(a) = agent.lock() {
                let phase = format!("{:?}", a.phase());
                *counts.entry(phase).or_insert(0) += 1;
            }
        }
        counts
    }

    /// Enlist an available agent: propose a contract and wait for response.
    /// Returns the agent ID and a handle if successful.
    pub fn enlist(
        &self,
        proposal: ContractProposal,
    ) -> Result<(String, Arc<Mutex<ManagedAgent>>), ContractError> {
        // Find first pooled agent
        let available = self.agents_in_phase(AgentPhase::Pooled);
        let agent_id = available
            .first()
            .ok_or(ContractError::NotFound("no available agents".into()))?;

        let agent = self
            .agents
            .get(agent_id)
            .ok_or(ContractError::NotFound(agent_id.clone()))?;

        {
            let mut a = agent.lock().map_err(|_| {
                ContractError::NotFound("agent lock poisoned".into())
            })?;
            a.propose(proposal)?;
        }

        Ok((agent_id.clone(), Arc::clone(agent)))
    }
}

impl Default for AgentPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> AttestationKey {
        AttestationKey::generate("lifecycle-test")
    }

    fn test_proposal(contract_id: &str) -> ContractProposal {
        ContractProposal::new(
            contract_id.into(),
            "inst-1".into(),
            WorkScope::Query {
                description: "test work".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        )
    }

    #[test]
    fn test_agent_initial_state() {
        let agent = ManagedAgent::new("agent-1".into());
        assert_eq!(agent.phase(), AgentPhase::Pooled);
        assert!(agent.contract().is_none());
        assert!(agent.last_findings().is_none());
        assert_eq!(agent.lifetime_energy_uj(), 0);
        assert_eq!(agent.engagements_completed(), 0);
    }

    #[test]
    fn test_full_lifecycle_accept() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-2".into());

        // Propose
        let proposal = test_proposal("c-lifecycle-1");
        let hash = proposal.proposal_hash.clone();
        agent.propose(proposal).unwrap();
        assert_eq!(agent.phase(), AgentPhase::Negotiating);

        // Accept
        let response = ContractResponse::accept("c-lifecycle-1", &hash);
        let phase = agent.process_response(response, &key).unwrap();
        assert_eq!(phase, AgentPhase::Engaged);
        assert!(agent.contract().is_some());

        // Voluntary return
        let findings = AgentFindings {
            summary: "completed successfully".into(),
            results: Some(serde_json::json!({"answer": 42})),
            anomalies: vec![],
            energy_consumed_uj: 35_000_000,
            wall_time: Duration::from_secs(15),
            scope_completed: true,
            incomplete_reason: None,
        };
        agent.voluntary_return(findings).unwrap();
        assert_eq!(agent.phase(), AgentPhase::Returned);
        assert_eq!(agent.engagements_completed(), 1);
        assert_eq!(agent.lifetime_energy_uj(), 35_000_000);

        // Return to pool
        agent.return_to_pool();
        assert_eq!(agent.phase(), AgentPhase::Pooled);
    }

    #[test]
    fn test_lifecycle_reject() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-3".into());
        let proposal = test_proposal("c-reject-1");
        agent.propose(proposal).unwrap();

        let response = ContractResponse::reject("c-reject-1", "scope too broad");
        let result = agent.process_response(response, &key);
        assert!(result.is_err());
        assert_eq!(agent.phase(), AgentPhase::Pooled);
    }

    #[test]
    fn test_lifecycle_counter_propose() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-4".into());
        let proposal = test_proposal("c-counter-1");
        agent.propose(proposal).unwrap();

        // Counter-propose
        let response = ContractResponse::counter_propose(
            "c-counter-1",
            Some(100_000_000),
            None,
            None,
            "need more energy",
        );
        let phase = agent.process_response(response, &key).unwrap();
        assert_eq!(phase, AgentPhase::Negotiating);

        // Accept revised proposal
        let revised = agent.pending_proposal.as_ref().unwrap();
        assert_eq!(revised.energy_budget_uj, 100_000_000);
        let hash = revised.proposal_hash.clone();
        let response = ContractResponse::accept("c-counter-1", &hash);
        let phase = agent.process_response(response, &key).unwrap();
        assert_eq!(phase, AgentPhase::Engaged);
    }

    #[test]
    fn test_negotiation_max_rounds() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-5".into());
        agent.max_negotiation_rounds = 2;
        let proposal = test_proposal("c-max-rounds");
        agent.propose(proposal).unwrap();

        // Counter 1
        let response =
            ContractResponse::counter_propose("c-max-rounds", Some(100_000_000), None, None, "more");
        agent.process_response(response, &key).unwrap();

        // Counter 2 → should fail (max rounds reached)
        let response =
            ContractResponse::counter_propose("c-max-rounds", Some(200_000_000), None, None, "even more");
        let result = agent.process_response(response, &key);
        assert!(matches!(result, Err(ContractError::NegotiationFailed(2))));
        assert_eq!(agent.phase(), AgentPhase::Pooled);
    }

    #[test]
    fn test_recall() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-6".into());
        let proposal = test_proposal("c-recall-1");
        let hash = proposal.proposal_hash.clone();
        agent.propose(proposal).unwrap();
        let response = ContractResponse::accept("c-recall-1", &hash);
        agent.process_response(response, &key).unwrap();
        assert_eq!(agent.phase(), AgentPhase::Engaged);

        let findings = agent.recall("budget exceeded");
        assert_eq!(agent.phase(), AgentPhase::Recalled);
        assert!(!findings.scope_completed);
        assert_eq!(
            findings.incomplete_reason.as_deref(),
            Some("budget exceeded")
        );
    }

    #[test]
    fn test_extension_during_engagement() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-7".into());
        let proposal = ContractProposal::new(
            "c-ext-1".into(),
            "inst-1".into(),
            WorkScope::Research {
                description: "explore".into(),
                hypothesis: None,
            },
            50_000_000,
            Duration::from_secs(60),
            ReturnTerms::FindingsAndAnomalies,
            ExtensionPolicy::SingleExtension {
                max_additional_uj: 25_000_000,
            },
        );
        let hash = proposal.proposal_hash.clone();
        agent.propose(proposal).unwrap();
        let response = ContractResponse::accept("c-ext-1", &hash);
        agent.process_response(response, &key).unwrap();

        let request = ExtensionRequest {
            contract_id: "c-ext-1".into(),
            additional_energy_uj: 20_000_000,
            additional_time: None,
            rationale: "found interesting cluster".into(),
            interim_findings: Some("2 anomalies detected".into()),
            consumed_uj: 45_000_000,
        };
        let resp = agent.process_extension_request(&request).unwrap();
        assert!(resp.is_granted());
        assert_eq!(resp.granted_energy_uj(), 20_000_000);
    }

    #[test]
    fn test_agent_pool_basic() {
        let mut pool = AgentPool::new();
        assert!(pool.is_empty());

        pool.register("a-1".into());
        pool.register("a-2".into());
        pool.register("a-3".into());
        assert_eq!(pool.len(), 3);

        let available = pool.agents_in_phase(AgentPhase::Pooled);
        assert_eq!(available.len(), 3);
    }

    #[test]
    fn test_agent_pool_enlist() {
        let mut pool = AgentPool::new();
        pool.register("a-1".into());
        pool.register("a-2".into());

        let proposal = test_proposal("c-pool-1");
        let (agent_id, _handle) = pool.enlist(proposal).unwrap();
        assert!(!agent_id.is_empty());

        // One agent should be negotiating now
        let negotiating = pool.agents_in_phase(AgentPhase::Negotiating);
        assert_eq!(negotiating.len(), 1);
        let pooled = pool.agents_in_phase(AgentPhase::Pooled);
        assert_eq!(pooled.len(), 1);
    }

    #[test]
    fn test_agent_pool_no_available() {
        let pool = AgentPool::new();
        let proposal = test_proposal("c-empty");
        let result = pool.enlist(proposal);
        assert!(result.is_err());
    }

    #[test]
    fn test_agent_pool_phase_counts() {
        let key = test_key();
        let mut pool = AgentPool::new();
        pool.register("a-1".into());
        pool.register("a-2".into());
        pool.register("a-3".into());

        // Enlist one agent
        let proposal = test_proposal("c-counts-1");
        let hash = proposal.proposal_hash.clone();
        let (_, handle) = pool.enlist(proposal).unwrap();
        {
            let mut a = handle.lock().unwrap();
            let response = ContractResponse::accept("c-counts-1", &hash);
            a.process_response(response, &key).unwrap();
        }

        let counts = pool.phase_counts();
        assert_eq!(*counts.get("Pooled").unwrap_or(&0), 2);
        assert_eq!(*counts.get("Engaged").unwrap_or(&0), 1);
    }

    #[test]
    fn test_agent_pool_remove() {
        let mut pool = AgentPool::new();
        pool.register("a-1".into());
        assert_eq!(pool.len(), 1);
        pool.remove("a-1");
        assert_eq!(pool.len(), 0);
        assert!(pool.get("a-1").is_none());
    }

    #[test]
    fn test_anomaly_serde() {
        let anomaly = Anomaly {
            description: "unusual energy spike at minute 3".into(),
            severity: 0.7,
            category: AnomalyCategory::Resource,
            evidence: Some(serde_json::json!({
                "spike_uj": 5_000_000,
                "baseline_uj": 500_000,
            })),
        };
        let json = serde_json::to_string(&anomaly).unwrap();
        let parsed: Anomaly = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.severity, 0.7);
        assert_eq!(parsed.category, AnomalyCategory::Resource);
    }

    #[test]
    fn test_findings_serde() {
        let findings = AgentFindings {
            summary: "analyzed 1000 nodes".into(),
            results: Some(serde_json::json!({"clusters": 5})),
            anomalies: vec![Anomaly {
                description: "dead cluster".into(),
                severity: 0.3,
                category: AnomalyCategory::DataQuality,
                evidence: None,
            }],
            energy_consumed_uj: 42_000_000,
            wall_time: Duration::from_secs(25),
            scope_completed: true,
            incomplete_reason: None,
        };
        let json = serde_json::to_string(&findings).unwrap();
        let parsed: AgentFindings = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.anomalies.len(), 1);
        assert!(parsed.scope_completed);
    }

    #[test]
    fn test_multiple_engagements() {
        let key = test_key();
        let mut agent = ManagedAgent::new("agent-multi".into());

        for i in 0..3 {
            let proposal = test_proposal(&format!("c-multi-{}", i));
            let hash = proposal.proposal_hash.clone();
            agent.propose(proposal).unwrap();
            let response = ContractResponse::accept(&format!("c-multi-{}", i), &hash);
            agent.process_response(response, &key).unwrap();

            let findings = AgentFindings {
                summary: format!("engagement {}", i),
                results: None,
                anomalies: vec![],
                energy_consumed_uj: 10_000_000,
                wall_time: Duration::from_secs(5),
                scope_completed: true,
                incomplete_reason: None,
            };
            agent.voluntary_return(findings).unwrap();
            agent.return_to_pool();
        }

        assert_eq!(agent.engagements_completed(), 3);
        assert_eq!(agent.lifetime_energy_uj(), 30_000_000);
        assert_eq!(agent.phase(), AgentPhase::Pooled);
    }
}
