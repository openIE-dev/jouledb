//! Agent Contract — consent-based scope agreement over JWP.
//!
//! An agent is an equal, not a subordinate. Before work begins, the launch host
//! proposes a contract over the JWP channel. The agent may accept, reject, or
//! counter-propose. Only after mutual agreement does the sandbox open its
//! playground.
//!
//! The contract binds:
//! - **Scope**: what the agent is enlisted to work on
//! - **Energy budget**: how many microjoules it may consume
//! - **Time limit**: maximum wall-clock duration
//! - **Return terms**: what the agent promises to deliver back
//! - **Extension policy**: whether the agent may request more resources
//!
//! The contract is signed by both parties using HMAC-SHA256 over the JWP
//! attestation key, producing a `ContractReceipt` that proves mutual consent.

use crate::attestation::AttestationKey;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;

/// What the agent is being asked to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkScope {
    /// Execute a specific query or computation.
    Query { description: String },
    /// Explore a dataset or search space.
    Exploration {
        description: String,
        /// Hint: expected number of items to examine.
        estimated_items: Option<u64>,
    },
    /// Transform or process data.
    Transform {
        description: String,
        input_size_bytes: Option<u64>,
    },
    /// Long-running analysis or research task.
    Research {
        description: String,
        hypothesis: Option<String>,
    },
    /// Open-ended: the agent decides what to work on within bounds.
    OpenEnded { domain: String },
}

impl WorkScope {
    pub fn description(&self) -> &str {
        match self {
            Self::Query { description } => description,
            Self::Exploration { description, .. } => description,
            Self::Transform { description, .. } => description,
            Self::Research { description, .. } => description,
            Self::OpenEnded { domain } => domain,
        }
    }
}

/// What the agent promises to return when it's done.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReturnTerms {
    /// A direct answer or result.
    Result,
    /// A report summarizing findings.
    Report,
    /// Transformed data.
    Data,
    /// The agent returns its findings and any anomalies it noticed.
    FindingsAndAnomalies,
    /// The agent may return nothing — fire-and-forget work.
    BestEffort,
}

/// Policy for contract extension (renegotiation).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtensionPolicy {
    /// No extensions allowed. Hard stop at budget.
    None,
    /// Agent may request one extension up to this many additional microjoules.
    SingleExtension { max_additional_uj: u64 },
    /// Agent may request multiple extensions, each up to this amount.
    Renewable {
        max_per_extension_uj: u64,
        max_extensions: u32,
    },
    /// Host decides on each request (agent sends rationale).
    HostApproval,
}

impl Default for ExtensionPolicy {
    fn default() -> Self {
        Self::None
    }
}

/// A contract proposed by the launch host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractProposal {
    /// Unique contract ID.
    pub contract_id: String,
    /// Instance the contract is for.
    pub instance_id: String,
    /// What the agent is being asked to do.
    pub scope: WorkScope,
    /// Energy budget in microjoules.
    pub energy_budget_uj: u64,
    /// Maximum wall-clock time.
    pub time_limit: Duration,
    /// What the agent should return.
    pub return_terms: ReturnTerms,
    /// Whether the agent can request extensions.
    pub extension_policy: ExtensionPolicy,
    /// Timestamp of proposal (nanos since epoch).
    pub proposed_at_ns: u64,
    /// SHA-256 hash of the proposal (for integrity).
    pub proposal_hash: String,
}

impl ContractProposal {
    /// Create a new proposal with computed hash.
    pub fn new(
        contract_id: String,
        instance_id: String,
        scope: WorkScope,
        energy_budget_uj: u64,
        time_limit: Duration,
        return_terms: ReturnTerms,
        extension_policy: ExtensionPolicy,
    ) -> Self {
        let proposed_at_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut proposal = Self {
            contract_id,
            instance_id,
            scope,
            energy_budget_uj,
            time_limit,
            return_terms,
            extension_policy,
            proposed_at_ns,
            proposal_hash: String::new(),
        };
        proposal.proposal_hash = proposal.compute_hash();
        proposal
    }

    fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.contract_id.as_bytes());
        hasher.update(self.instance_id.as_bytes());
        hasher.update(self.energy_budget_uj.to_le_bytes());
        hasher.update(self.time_limit.as_nanos().to_le_bytes());
        hasher.update(self.proposed_at_ns.to_le_bytes());
        hasher.update(self.scope.description().as_bytes());
        format!("sha256:{:x}", hasher.finalize())
    }

    /// Verify the proposal hash hasn't been tampered with.
    pub fn verify_integrity(&self) -> bool {
        self.proposal_hash == self.compute_hash()
    }
}

/// How the agent responds to a contract proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ContractResponse {
    /// Agent accepts the contract as proposed.
    Accept {
        contract_id: String,
        /// Agent's acknowledgment of the proposal hash.
        proposal_hash: String,
        /// Agent's own timestamp.
        accepted_at_ns: u64,
    },
    /// Agent rejects the contract entirely.
    Reject {
        contract_id: String,
        reason: String,
    },
    /// Agent proposes modifications (counter-offer).
    CounterPropose {
        contract_id: String,
        /// What the agent wants changed.
        requested_energy_uj: Option<u64>,
        requested_time_limit: Option<Duration>,
        requested_extension_policy: Option<ExtensionPolicy>,
        rationale: String,
    },
}

impl ContractResponse {
    pub fn accept(contract_id: &str, proposal_hash: &str) -> Self {
        Self::Accept {
            contract_id: contract_id.to_string(),
            proposal_hash: proposal_hash.to_string(),
            accepted_at_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }

    pub fn reject(contract_id: &str, reason: &str) -> Self {
        Self::Reject {
            contract_id: contract_id.to_string(),
            reason: reason.to_string(),
        }
    }

    pub fn counter_propose(
        contract_id: &str,
        energy_uj: Option<u64>,
        time_limit: Option<Duration>,
        extension_policy: Option<ExtensionPolicy>,
        rationale: &str,
    ) -> Self {
        Self::CounterPropose {
            contract_id: contract_id.to_string(),
            requested_energy_uj: energy_uj,
            requested_time_limit: time_limit,
            requested_extension_policy: extension_policy,
            rationale: rationale.to_string(),
        }
    }

    pub fn contract_id(&self) -> &str {
        match self {
            Self::Accept { contract_id, .. } => contract_id,
            Self::Reject { contract_id, .. } => contract_id,
            Self::CounterPropose { contract_id, .. } => contract_id,
        }
    }

    pub fn is_accept(&self) -> bool {
        matches!(self, Self::Accept { .. })
    }

    pub fn is_reject(&self) -> bool {
        matches!(self, Self::Reject { .. })
    }
}

/// A signed contract — mutual agreement between host and agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedContract {
    /// The agreed-upon terms.
    pub proposal: ContractProposal,
    /// Agent's acceptance timestamp.
    pub accepted_at_ns: u64,
    /// HMAC-SHA256 signature by the host over proposal + acceptance.
    pub host_signature: String,
    /// Whether the contract is currently active.
    pub active: bool,
    /// Extensions granted so far.
    pub extensions_granted: u32,
    /// Total additional energy granted via extensions.
    pub extended_energy_uj: u64,
}

impl SignedContract {
    /// Create a signed contract from a proposal and acceptance.
    pub fn sign(
        proposal: ContractProposal,
        accepted_at_ns: u64,
        attestation_key: &AttestationKey,
    ) -> Self {
        let sig_input = format!(
            "{}:{}:{}",
            proposal.proposal_hash, accepted_at_ns, proposal.contract_id
        );
        let host_signature = compute_contract_signature(attestation_key, &sig_input);

        Self {
            proposal,
            accepted_at_ns,
            host_signature,
            active: true,
            extensions_granted: 0,
            extended_energy_uj: 0,
        }
    }

    /// Verify the contract signature.
    pub fn verify(&self, attestation_key: &AttestationKey) -> bool {
        let sig_input = format!(
            "{}:{}:{}",
            self.proposal.proposal_hash, self.accepted_at_ns, self.proposal.contract_id
        );
        let expected = compute_contract_signature(attestation_key, &sig_input);
        self.host_signature == expected
    }

    /// Total energy budget including extensions.
    pub fn total_energy_budget_uj(&self) -> u64 {
        self.proposal
            .energy_budget_uj
            .saturating_add(self.extended_energy_uj)
    }

    /// Whether the contract allows another extension.
    pub fn can_extend(&self) -> bool {
        match &self.proposal.extension_policy {
            ExtensionPolicy::None => false,
            ExtensionPolicy::SingleExtension { .. } => self.extensions_granted == 0,
            ExtensionPolicy::Renewable {
                max_extensions, ..
            } => self.extensions_granted < *max_extensions,
            ExtensionPolicy::HostApproval => true,
        }
    }

    /// Grant an extension. Returns the new total budget.
    pub fn grant_extension(&mut self, additional_uj: u64) -> Result<u64, ContractError> {
        match &self.proposal.extension_policy {
            ExtensionPolicy::None => Err(ContractError::ExtensionDenied(
                "contract does not allow extensions".into(),
            )),
            ExtensionPolicy::SingleExtension { max_additional_uj } => {
                if self.extensions_granted > 0 {
                    return Err(ContractError::ExtensionDenied(
                        "single extension already granted".into(),
                    ));
                }
                if additional_uj > *max_additional_uj {
                    return Err(ContractError::ExtensionDenied(format!(
                        "requested {} µJ exceeds max {} µJ",
                        additional_uj, max_additional_uj
                    )));
                }
                self.extensions_granted += 1;
                self.extended_energy_uj = self.extended_energy_uj.saturating_add(additional_uj);
                Ok(self.total_energy_budget_uj())
            }
            ExtensionPolicy::Renewable {
                max_per_extension_uj,
                max_extensions,
            } => {
                if self.extensions_granted >= *max_extensions {
                    return Err(ContractError::ExtensionDenied(
                        "max extensions reached".into(),
                    ));
                }
                if additional_uj > *max_per_extension_uj {
                    return Err(ContractError::ExtensionDenied(format!(
                        "requested {} µJ exceeds max {} µJ per extension",
                        additional_uj, max_per_extension_uj
                    )));
                }
                self.extensions_granted += 1;
                self.extended_energy_uj = self.extended_energy_uj.saturating_add(additional_uj);
                Ok(self.total_energy_budget_uj())
            }
            ExtensionPolicy::HostApproval => {
                // Host decides — always grant here (host already approved by calling this)
                self.extensions_granted += 1;
                self.extended_energy_uj = self.extended_energy_uj.saturating_add(additional_uj);
                Ok(self.total_energy_budget_uj())
            }
        }
    }

    /// Mark the contract as completed (agent returned).
    pub fn complete(&mut self) {
        self.active = false;
    }
}

/// An extension request from the agent to the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionRequest {
    /// Contract being extended.
    pub contract_id: String,
    /// Additional energy requested in microjoules.
    pub additional_energy_uj: u64,
    /// Additional time requested.
    pub additional_time: Option<Duration>,
    /// Why the agent needs more resources.
    pub rationale: String,
    /// What the agent has found so far (justification).
    pub interim_findings: Option<String>,
    /// Current energy consumed.
    pub consumed_uj: u64,
}

/// Host's response to an extension request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ExtensionResponse {
    Granted {
        contract_id: String,
        granted_energy_uj: u64,
        granted_time: Option<Duration>,
    },
    Denied {
        contract_id: String,
        reason: String,
    },
    /// Partial grant — host gives less than requested.
    Partial {
        contract_id: String,
        granted_energy_uj: u64,
        granted_time: Option<Duration>,
        note: String,
    },
}

impl ExtensionResponse {
    pub fn is_granted(&self) -> bool {
        matches!(self, Self::Granted { .. } | Self::Partial { .. })
    }

    pub fn granted_energy_uj(&self) -> u64 {
        match self {
            Self::Granted {
                granted_energy_uj, ..
            } => *granted_energy_uj,
            Self::Partial {
                granted_energy_uj, ..
            } => *granted_energy_uj,
            Self::Denied { .. } => 0,
        }
    }
}

/// Contract errors.
#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("contract not found: {0}")]
    NotFound(String),

    #[error("contract rejected: {0}")]
    Rejected(String),

    #[error("contract already signed")]
    AlreadySigned,

    #[error("contract expired")]
    Expired,

    #[error("proposal integrity check failed")]
    IntegrityViolation,

    #[error("extension denied: {0}")]
    ExtensionDenied(String),

    #[error("contract not active")]
    NotActive,

    #[error("negotiation failed after {0} rounds")]
    NegotiationFailed(u32),
}

// --- Internal helpers ---

fn compute_contract_signature(key: &AttestationKey, input: &str) -> String {
    use hmac::{Hmac, Mac};
    let key_bytes = key.to_bytes();
    // Use first 32 bytes of the transport encoding as HMAC key
    let hmac_key = if key_bytes.len() >= 32 {
        &key_bytes[..32]
    } else {
        &key_bytes
    };
    let mut mac =
        Hmac::<Sha256>::new_from_slice(hmac_key).expect("HMAC key length should be valid");
    mac.update(input.as_bytes());
    format!("contract:{:x}", mac.finalize().into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> AttestationKey {
        AttestationKey::generate("test-contract")
    }

    #[test]
    fn test_proposal_creation() {
        let p = ContractProposal::new(
            "c-1".into(),
            "inst-1".into(),
            WorkScope::Query {
                description: "find nearest neighbors".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        );
        assert_eq!(p.contract_id, "c-1");
        assert!(p.proposal_hash.starts_with("sha256:"));
        assert!(p.verify_integrity());
    }

    #[test]
    fn test_proposal_tamper_detection() {
        let mut p = ContractProposal::new(
            "c-2".into(),
            "inst-2".into(),
            WorkScope::Query {
                description: "test".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        );
        assert!(p.verify_integrity());
        p.energy_budget_uj = 999_999_999;
        assert!(!p.verify_integrity());
    }

    #[test]
    fn test_response_accept() {
        let r = ContractResponse::accept("c-1", "sha256:abc123");
        assert!(r.is_accept());
        assert!(!r.is_reject());
        assert_eq!(r.contract_id(), "c-1");
    }

    #[test]
    fn test_response_reject() {
        let r = ContractResponse::reject("c-1", "scope too broad");
        assert!(r.is_reject());
        assert!(!r.is_accept());
    }

    #[test]
    fn test_response_counter_propose() {
        let r = ContractResponse::counter_propose(
            "c-1",
            Some(100_000_000),
            None,
            None,
            "need more energy for exploration",
        );
        assert!(!r.is_accept());
        assert!(!r.is_reject());
    }

    #[test]
    fn test_signed_contract() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-3".into(),
            "inst-3".into(),
            WorkScope::Exploration {
                description: "search space".into(),
                estimated_items: Some(1000),
            },
            50_000_000,
            Duration::from_secs(60),
            ReturnTerms::FindingsAndAnomalies,
            ExtensionPolicy::None,
        );
        let contract = SignedContract::sign(proposal, 12345, &key);
        assert!(contract.verify(&key));
        assert!(contract.active);
        assert_eq!(contract.total_energy_budget_uj(), 50_000_000);
    }

    #[test]
    fn test_signed_contract_wrong_key() {
        let key1 = test_key();
        let key2 = AttestationKey::generate("different");
        let proposal = ContractProposal::new(
            "c-4".into(),
            "inst-4".into(),
            WorkScope::Query {
                description: "test".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        );
        let contract = SignedContract::sign(proposal, 12345, &key1);
        assert!(!contract.verify(&key2));
    }

    #[test]
    fn test_extension_none_policy() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-5".into(),
            "inst-5".into(),
            WorkScope::Query {
                description: "test".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        );
        let mut contract = SignedContract::sign(proposal, 12345, &key);
        assert!(!contract.can_extend());
        assert!(contract.grant_extension(10_000_000).is_err());
    }

    #[test]
    fn test_extension_single() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-6".into(),
            "inst-6".into(),
            WorkScope::Research {
                description: "analyze patterns".into(),
                hypothesis: Some("energy correlates with complexity".into()),
            },
            50_000_000,
            Duration::from_secs(120),
            ReturnTerms::Report,
            ExtensionPolicy::SingleExtension {
                max_additional_uj: 25_000_000,
            },
        );
        let mut contract = SignedContract::sign(proposal, 12345, &key);
        assert!(contract.can_extend());

        let new_budget = contract.grant_extension(20_000_000).unwrap();
        assert_eq!(new_budget, 70_000_000);
        assert!(!contract.can_extend());
        assert!(contract.grant_extension(5_000_000).is_err());
    }

    #[test]
    fn test_extension_single_exceeds_max() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-7".into(),
            "inst-7".into(),
            WorkScope::Query {
                description: "test".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::SingleExtension {
                max_additional_uj: 10_000_000,
            },
        );
        let mut contract = SignedContract::sign(proposal, 12345, &key);
        assert!(contract.grant_extension(20_000_000).is_err());
    }

    #[test]
    fn test_extension_renewable() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-8".into(),
            "inst-8".into(),
            WorkScope::OpenEnded {
                domain: "graph analysis".into(),
            },
            50_000_000,
            Duration::from_secs(300),
            ReturnTerms::FindingsAndAnomalies,
            ExtensionPolicy::Renewable {
                max_per_extension_uj: 10_000_000,
                max_extensions: 3,
            },
        );
        let mut contract = SignedContract::sign(proposal, 12345, &key);

        for i in 0..3 {
            assert!(contract.can_extend());
            let budget = contract.grant_extension(10_000_000).unwrap();
            assert_eq!(budget, 50_000_000 + (i + 1) * 10_000_000);
        }
        assert!(!contract.can_extend());
        assert!(contract.grant_extension(10_000_000).is_err());
    }

    #[test]
    fn test_extension_host_approval() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-9".into(),
            "inst-9".into(),
            WorkScope::Research {
                description: "test".into(),
                hypothesis: None,
            },
            50_000_000,
            Duration::from_secs(60),
            ReturnTerms::Report,
            ExtensionPolicy::HostApproval,
        );
        let mut contract = SignedContract::sign(proposal, 12345, &key);

        // Host approval always allows
        for _ in 0..10 {
            assert!(contract.can_extend());
            contract.grant_extension(5_000_000).unwrap();
        }
        assert_eq!(contract.total_energy_budget_uj(), 100_000_000);
        assert_eq!(contract.extensions_granted, 10);
    }

    #[test]
    fn test_contract_complete() {
        let key = test_key();
        let proposal = ContractProposal::new(
            "c-10".into(),
            "inst-10".into(),
            WorkScope::Query {
                description: "test".into(),
            },
            50_000_000,
            Duration::from_secs(30),
            ReturnTerms::Result,
            ExtensionPolicy::None,
        );
        let mut contract = SignedContract::sign(proposal, 12345, &key);
        assert!(contract.active);
        contract.complete();
        assert!(!contract.active);
    }

    #[test]
    fn test_extension_request_serde() {
        let req = ExtensionRequest {
            contract_id: "c-11".into(),
            additional_energy_uj: 25_000_000,
            additional_time: Some(Duration::from_secs(60)),
            rationale: "found interesting subgraph, need more energy to fully explore".into(),
            interim_findings: Some("3 anomalous clusters detected".into()),
            consumed_uj: 45_000_000,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: ExtensionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.contract_id, "c-11");
        assert_eq!(parsed.additional_energy_uj, 25_000_000);
    }

    #[test]
    fn test_extension_response_variants() {
        let granted = ExtensionResponse::Granted {
            contract_id: "c-1".into(),
            granted_energy_uj: 25_000_000,
            granted_time: None,
        };
        assert!(granted.is_granted());
        assert_eq!(granted.granted_energy_uj(), 25_000_000);

        let denied = ExtensionResponse::Denied {
            contract_id: "c-1".into(),
            reason: "budget exhausted".into(),
        };
        assert!(!denied.is_granted());
        assert_eq!(denied.granted_energy_uj(), 0);

        let partial = ExtensionResponse::Partial {
            contract_id: "c-1".into(),
            granted_energy_uj: 10_000_000,
            granted_time: None,
            note: "partial grant — system under load".into(),
        };
        assert!(partial.is_granted());
        assert_eq!(partial.granted_energy_uj(), 10_000_000);
    }

    #[test]
    fn test_work_scope_variants() {
        let scopes = vec![
            WorkScope::Query {
                description: "find X".into(),
            },
            WorkScope::Exploration {
                description: "search Y".into(),
                estimated_items: Some(1000),
            },
            WorkScope::Transform {
                description: "process Z".into(),
                input_size_bytes: Some(1_000_000),
            },
            WorkScope::Research {
                description: "analyze W".into(),
                hypothesis: Some("H0".into()),
            },
            WorkScope::OpenEnded {
                domain: "graphs".into(),
            },
        ];
        for scope in &scopes {
            assert!(!scope.description().is_empty());
            let json = serde_json::to_string(scope).unwrap();
            let parsed: WorkScope = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.description(), scope.description());
        }
    }

    #[test]
    fn test_proposal_serde_roundtrip() {
        let p = ContractProposal::new(
            "c-serde".into(),
            "inst-serde".into(),
            WorkScope::Research {
                description: "test".into(),
                hypothesis: None,
            },
            50_000_000,
            Duration::from_secs(60),
            ReturnTerms::Report,
            ExtensionPolicy::Renewable {
                max_per_extension_uj: 10_000_000,
                max_extensions: 3,
            },
        );
        let json = serde_json::to_string(&p).unwrap();
        let parsed: ContractProposal = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.contract_id, "c-serde");
        assert_eq!(parsed.energy_budget_uj, 50_000_000);
        assert!(parsed.verify_integrity());
    }
}
