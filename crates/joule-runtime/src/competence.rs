//! Competence Ledger — earned trust through measured track record.
//!
//! Every agent carries a competence vector C ∈ ℝ^D where D is the number
//! of recognized task domains. The delegation level L_d ∈ {1..7} is a
//! deterministic function of the competence score — not configured, inherited,
//! or permanent. It increases on success, decreases on failure, and is
//! unaffected by legitimate halts.
//!
//! From the axioms:
//! - A4 (EARNED TRUST): delegation is measured, not assigned
//! - A3 (OBLIGATION): halts never penalize — only undetected defects do
//!
//! The delegation levels follow Appelo's Management 3.0 model:
//!
//! ```text
//! L1  Tell       Execute literal commands only
//! L2  Sell       Execute with explanation of rationale
//! L3  Consult    Propose action, await approval
//! L4  Agree      Joint decision between node and observer
//! L5  Advise     Act independently, report rationale
//! L6  Inquire    Act independently, report only on request
//! L7  Delegate   Full autonomy in domain
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// A recognized task domain.
pub type DomainId = String;

/// Delegation level (1-7) per Appelo's Management 3.0 model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DelegationLevel {
    /// Execute literal commands only.
    Tell = 1,
    /// Execute with explanation of rationale.
    Sell = 2,
    /// Propose action, await approval.
    Consult = 3,
    /// Joint decision between node and observer.
    Agree = 4,
    /// Act independently, report rationale.
    Advise = 5,
    /// Act independently, report only on request.
    Inquire = 6,
    /// Full autonomy in domain.
    Delegate = 7,
}

impl DelegationLevel {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Tell),
            2 => Some(Self::Sell),
            3 => Some(Self::Consult),
            4 => Some(Self::Agree),
            5 => Some(Self::Advise),
            6 => Some(Self::Inquire),
            7 => Some(Self::Delegate),
            _ => None,
        }
    }

    /// Whether this level permits autonomous action without approval.
    pub fn is_autonomous(self) -> bool {
        matches!(self, Self::Advise | Self::Inquire | Self::Delegate)
    }

    /// Whether this level requires approval before acting.
    pub fn requires_approval(self) -> bool {
        matches!(self, Self::Tell | Self::Sell | Self::Consult | Self::Agree)
    }
}

impl std::fmt::Display for DelegationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tell => write!(f, "L1:Tell"),
            Self::Sell => write!(f, "L2:Sell"),
            Self::Consult => write!(f, "L3:Consult"),
            Self::Agree => write!(f, "L4:Agree"),
            Self::Advise => write!(f, "L5:Advise"),
            Self::Inquire => write!(f, "L6:Inquire"),
            Self::Delegate => write!(f, "L7:Delegate"),
        }
    }
}

/// Per-domain competence record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainCompetence {
    /// Number of successful completions in this domain.
    pub successes: u64,
    /// Number of failures in this domain.
    pub failures: u64,
    /// Total attempts (successes + failures + escalations).
    pub attempts: u64,
    /// Number of legitimate halts (not penalized).
    pub halts: u64,
    /// Cumulative energy consumed in this domain (microjoules).
    pub energy_consumed_uj: u64,
    /// When this domain was first seen.
    pub first_seen: u64,
    /// When the last event occurred.
    pub last_event: u64,
}

impl DomainCompetence {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            successes: 0,
            failures: 0,
            attempts: 0,
            halts: 0,
            energy_consumed_uj: 0,
            first_seen: now,
            last_event: now,
        }
    }

    fn touch(&mut self) {
        self.last_event = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
    }
}

impl Default for DomainCompetence {
    fn default() -> Self {
        Self::new()
    }
}

/// Thresholds for mapping competence score → delegation level.
///
/// C_d maps to L_d via fixed thresholds:
/// ```text
/// L1  if C_d < θ₁
/// L2  if θ₁ ≤ C_d < θ₂
/// L3  if θ₂ ≤ C_d < θ₃
/// L4  if θ₃ ≤ C_d < θ₄
/// L5  if θ₄ ≤ C_d < θ₅
/// L6  if θ₅ ≤ C_d < θ₆
/// L7  if C_d ≥ θ₆
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetenceThresholds {
    /// Failure penalty multiplier (α > 1: failures hurt more than successes help).
    pub failure_penalty: f64,
    /// Smoothing constant (λ: prevents premature high confidence on few attempts).
    pub smoothing: f64,
    /// Threshold boundaries [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] for L1→L2→...→L7.
    pub thresholds: [f64; 6],
}

impl Default for CompetenceThresholds {
    fn default() -> Self {
        Self {
            failure_penalty: 2.0,  // α: failures penalized 2× successes
            smoothing: 5.0,        // λ: need ~5 attempts before scores stabilize
            thresholds: [
                0.10,  // θ₁: L1→L2 (Sell)
                0.25,  // θ₂: L2→L3 (Consult)
                0.45,  // θ₃: L3→L4 (Agree)
                0.60,  // θ₄: L4→L5 (Advise)
                0.75,  // θ₅: L5→L6 (Inquire)
                0.90,  // θ₆: L6→L7 (Delegate)
            ],
        }
    }
}

impl CompetenceThresholds {
    /// Compute the competence score C_d for a domain.
    ///
    /// C_d = (successes - α · failures) / (attempts + λ)
    pub fn score(&self, domain: &DomainCompetence) -> f64 {
        let numerator =
            domain.successes as f64 - self.failure_penalty * domain.failures as f64;
        let denominator = domain.attempts as f64 + self.smoothing;
        (numerator / denominator).clamp(-1.0, 1.0)
    }

    /// Map a competence score to a delegation level.
    pub fn delegation_level(&self, score: f64) -> DelegationLevel {
        if score >= self.thresholds[5] {
            DelegationLevel::Delegate
        } else if score >= self.thresholds[4] {
            DelegationLevel::Inquire
        } else if score >= self.thresholds[3] {
            DelegationLevel::Advise
        } else if score >= self.thresholds[2] {
            DelegationLevel::Agree
        } else if score >= self.thresholds[1] {
            DelegationLevel::Consult
        } else if score >= self.thresholds[0] {
            DelegationLevel::Sell
        } else {
            DelegationLevel::Tell
        }
    }
}

/// The competence ledger: per-domain track record for a single agent node.
///
/// This is the structure that makes A4 (EARNED TRUST) concrete.
/// It tracks every domain the agent has operated in, computes competence
/// scores, and derives delegation levels deterministically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetenceLedger {
    /// Per-domain competence records.
    domains: HashMap<DomainId, DomainCompetence>,
    /// Thresholds for score → level mapping.
    thresholds: CompetenceThresholds,
}

impl CompetenceLedger {
    pub fn new() -> Self {
        Self {
            domains: HashMap::new(),
            thresholds: CompetenceThresholds::default(),
        }
    }

    pub fn with_thresholds(thresholds: CompetenceThresholds) -> Self {
        Self {
            domains: HashMap::new(),
            thresholds,
        }
    }

    /// Record a successful completion in a domain.
    pub fn record_success(&mut self, domain: &str, energy_uj: u64) {
        let entry = self
            .domains
            .entry(domain.to_string())
            .or_insert_with(DomainCompetence::new);
        entry.successes += 1;
        entry.attempts += 1;
        entry.energy_consumed_uj = entry.energy_consumed_uj.saturating_add(energy_uj);
        entry.touch();
    }

    /// Record a failure in a domain.
    pub fn record_failure(&mut self, domain: &str, energy_uj: u64) {
        let entry = self
            .domains
            .entry(domain.to_string())
            .or_insert_with(DomainCompetence::new);
        entry.failures += 1;
        entry.attempts += 1;
        entry.energy_consumed_uj = entry.energy_consumed_uj.saturating_add(energy_uj);
        entry.touch();
    }

    /// Record a legitimate halt (not penalized per A3).
    pub fn record_halt(&mut self, domain: &str) {
        let entry = self
            .domains
            .entry(domain.to_string())
            .or_insert_with(DomainCompetence::new);
        entry.halts += 1;
        // Halts do NOT increment attempts or failures.
        // A node that halts is never penalized.
        entry.touch();
    }

    /// Record an escalation (counted as an attempt but not success/failure).
    pub fn record_escalation(&mut self, domain: &str) {
        let entry = self
            .domains
            .entry(domain.to_string())
            .or_insert_with(DomainCompetence::new);
        entry.attempts += 1;
        // Escalation is neutral: not success, not failure.
        entry.touch();
    }

    /// Reset competence to Level 1 in ALL domains.
    ///
    /// This is the maximum penalty: applied when a node fails to halt on a
    /// detected defect. The system makes it safer to stop than to continue.
    pub fn reset_all(&mut self) {
        for domain in self.domains.values_mut() {
            domain.successes = 0;
            domain.failures = 0;
            domain.attempts = 0;
            // Halts and energy history preserved for auditing.
        }
    }

    /// Get the competence score C_d for a domain.
    pub fn score(&self, domain: &str) -> f64 {
        match self.domains.get(domain) {
            Some(d) => self.thresholds.score(d),
            None => -1.0, // Unknown domain: below Tell threshold
        }
    }

    /// Get the delegation level L_d for a domain.
    pub fn delegation_level(&self, domain: &str) -> DelegationLevel {
        self.thresholds.delegation_level(self.score(domain))
    }

    /// Get the domain competence record.
    pub fn domain(&self, domain: &str) -> Option<&DomainCompetence> {
        self.domains.get(domain)
    }

    /// List all known domains with their delegation levels.
    pub fn all_levels(&self) -> Vec<(String, DelegationLevel, f64)> {
        self.domains
            .iter()
            .map(|(d, comp)| {
                let score = self.thresholds.score(comp);
                let level = self.thresholds.delegation_level(score);
                (d.clone(), level, score)
            })
            .collect()
    }

    /// Find domains where this agent has at least the given delegation level.
    pub fn domains_at_level(&self, min_level: DelegationLevel) -> Vec<(String, DelegationLevel)> {
        self.all_levels()
            .into_iter()
            .filter(|(_, level, _)| *level >= min_level)
            .map(|(d, level, _)| (d, level))
            .collect()
    }

    /// Number of tracked domains.
    pub fn domain_count(&self) -> usize {
        self.domains.len()
    }

    /// Total attempts across all domains.
    pub fn total_attempts(&self) -> u64 {
        self.domains.values().map(|d| d.attempts).sum()
    }

    /// Total energy consumed across all domains.
    pub fn total_energy_uj(&self) -> u64 {
        self.domains.values().map(|d| d.energy_consumed_uj).sum()
    }

    /// Get thresholds (for tuning or inspection).
    pub fn thresholds(&self) -> &CompetenceThresholds {
        &self.thresholds
    }

    /// Whether the agent can act autonomously in a given domain
    /// (delegation level ≥ Advise).
    pub fn can_act_autonomously(&self, domain: &str) -> bool {
        self.delegation_level(domain).is_autonomous()
    }

    /// Compute the "bid" for a task in a given domain: competence × confidence.
    /// Used by the dispatch mesh for self-selection (§4.2).
    ///
    /// Returns (score, level) or None if the domain is completely unknown.
    pub fn bid(&self, domain: &str) -> Option<(f64, DelegationLevel)> {
        self.domains.get(domain).map(|comp| {
            let score = self.thresholds.score(comp);
            let level = self.thresholds.delegation_level(score);
            (score, level)
        })
    }
}

impl Default for CompetenceLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delegation_level_ordering() {
        assert!(DelegationLevel::Tell < DelegationLevel::Sell);
        assert!(DelegationLevel::Sell < DelegationLevel::Consult);
        assert!(DelegationLevel::Consult < DelegationLevel::Agree);
        assert!(DelegationLevel::Agree < DelegationLevel::Advise);
        assert!(DelegationLevel::Advise < DelegationLevel::Inquire);
        assert!(DelegationLevel::Inquire < DelegationLevel::Delegate);
    }

    #[test]
    fn test_delegation_level_display() {
        assert_eq!(DelegationLevel::Tell.to_string(), "L1:Tell");
        assert_eq!(DelegationLevel::Delegate.to_string(), "L7:Delegate");
    }

    #[test]
    fn test_delegation_level_autonomy() {
        assert!(!DelegationLevel::Tell.is_autonomous());
        assert!(!DelegationLevel::Consult.is_autonomous());
        assert!(DelegationLevel::Advise.is_autonomous());
        assert!(DelegationLevel::Delegate.is_autonomous());

        assert!(DelegationLevel::Tell.requires_approval());
        assert!(!DelegationLevel::Advise.requires_approval());
    }

    #[test]
    fn test_delegation_level_roundtrip() {
        for v in 1..=7 {
            let level = DelegationLevel::from_u8(v).unwrap();
            assert_eq!(level.as_u8(), v);
        }
        assert!(DelegationLevel::from_u8(0).is_none());
        assert!(DelegationLevel::from_u8(8).is_none());
    }

    #[test]
    fn test_fresh_ledger_all_tell() {
        let ledger = CompetenceLedger::new();
        assert_eq!(ledger.delegation_level("git"), DelegationLevel::Tell);
        assert_eq!(ledger.score("git"), -1.0);
    }

    #[test]
    fn test_competence_score_formula() {
        let thresholds = CompetenceThresholds::default();
        // C_d = (successes - α·failures) / (attempts + λ)
        // α=2.0, λ=5.0
        let mut dc = DomainCompetence::new();
        dc.successes = 10;
        dc.failures = 0;
        dc.attempts = 10;
        // C = (10 - 0) / (10 + 5) = 0.667
        let score = thresholds.score(&dc);
        assert!((score - 0.6667).abs() < 0.01);

        // With failures: 10 successes, 3 failures, 13 attempts
        dc.failures = 3;
        dc.attempts = 13;
        // C = (10 - 6) / (13 + 5) = 4/18 = 0.222
        let score = thresholds.score(&dc);
        assert!((score - 0.222).abs() < 0.01);
    }

    #[test]
    fn test_asymmetric_failure_penalty() {
        // Failures hurt more than successes help (α > 1)
        let mut ledger = CompetenceLedger::new();

        // 5 successes → score = 5 / (5+5) = 0.5
        for _ in 0..5 {
            ledger.record_success("fs", 1000);
        }
        let score_before = ledger.score("fs");

        // 1 failure → score = (5 - 2) / (6+5) = 3/11 ≈ 0.273
        ledger.record_failure("fs", 1000);
        let score_after = ledger.score("fs");

        // Single failure should cause significant drop
        assert!(score_before - score_after > 0.2);
    }

    #[test]
    fn test_halt_does_not_penalize() {
        let mut ledger = CompetenceLedger::new();
        for _ in 0..10 {
            ledger.record_success("web", 1000);
        }
        let score_before = ledger.score("web");

        // 5 halts should not change the score
        for _ in 0..5 {
            ledger.record_halt("web");
        }
        let score_after = ledger.score("web");
        assert!((score_before - score_after).abs() < 1e-10);

        // But halts are recorded
        assert_eq!(ledger.domain("web").unwrap().halts, 5);
    }

    #[test]
    fn test_escalation_is_neutral() {
        let mut ledger = CompetenceLedger::new();
        for _ in 0..10 {
            ledger.record_success("git", 1000);
        }
        let score_before = ledger.score("git");

        // Escalation increments attempts but not success/failure
        // Score should decrease slightly (more attempts, same numerator)
        ledger.record_escalation("git");
        let score_after = ledger.score("git");
        assert!(score_after < score_before);

        // But it's a smaller drop than a failure
        let mut ledger2 = CompetenceLedger::new();
        for _ in 0..10 {
            ledger2.record_success("git", 1000);
        }
        ledger2.record_failure("git", 1000);
        let score_with_failure = ledger2.score("git");

        assert!(score_after > score_with_failure);
    }

    #[test]
    fn test_promotion_through_levels() {
        let mut ledger = CompetenceLedger::new();
        let domain = "search";

        // Start at Tell (no history)
        assert_eq!(ledger.delegation_level(domain), DelegationLevel::Tell);

        // Build up successes and watch delegation level climb
        let mut levels_seen = vec![ledger.delegation_level(domain)];
        for _ in 0..50 {
            ledger.record_success(domain, 1000);
            let level = ledger.delegation_level(domain);
            if levels_seen.last() != Some(&level) {
                levels_seen.push(level);
            }
        }

        // Should have progressed through multiple levels
        assert!(levels_seen.len() >= 4, "expected progression, got {:?}", levels_seen);
        assert_eq!(*levels_seen.last().unwrap(), DelegationLevel::Delegate);
    }

    #[test]
    fn test_failure_causes_demotion() {
        let mut ledger = CompetenceLedger::new();
        let domain = "deploy";

        // Build up to Delegate level
        for _ in 0..50 {
            ledger.record_success(domain, 1000);
        }
        assert_eq!(ledger.delegation_level(domain), DelegationLevel::Delegate);

        // Series of failures should demote
        for _ in 0..20 {
            ledger.record_failure(domain, 1000);
        }
        let level = ledger.delegation_level(domain);
        assert!(level < DelegationLevel::Delegate);
    }

    #[test]
    fn test_reset_all_maximum_penalty() {
        let mut ledger = CompetenceLedger::new();

        for _ in 0..50 {
            ledger.record_success("git", 1000);
            ledger.record_success("fs", 1000);
            ledger.record_success("web", 1000);
        }

        // All domains at high levels
        assert_eq!(ledger.delegation_level("git"), DelegationLevel::Delegate);
        assert_eq!(ledger.delegation_level("fs"), DelegationLevel::Delegate);

        // Nuclear option: reset everything
        ledger.reset_all();

        // All domains back to Tell
        assert_eq!(ledger.delegation_level("git"), DelegationLevel::Tell);
        assert_eq!(ledger.delegation_level("fs"), DelegationLevel::Tell);
        assert_eq!(ledger.delegation_level("web"), DelegationLevel::Tell);

        // But energy history preserved for auditing
        assert!(ledger.total_energy_uj() > 0);
    }

    #[test]
    fn test_domains_at_level() {
        let mut ledger = CompetenceLedger::new();

        // Build git to Delegate, fs to Consult
        for _ in 0..50 {
            ledger.record_success("git", 1000);
        }
        for _ in 0..5 {
            ledger.record_success("fs", 1000);
        }

        let autonomous = ledger.domains_at_level(DelegationLevel::Advise);
        assert_eq!(autonomous.len(), 1);
        assert_eq!(autonomous[0].0, "git");

        let all = ledger.domains_at_level(DelegationLevel::Tell);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_bid_for_self_selection() {
        let mut ledger = CompetenceLedger::new();
        for _ in 0..30 {
            ledger.record_success("search", 1000);
        }
        for _ in 0..5 {
            ledger.record_success("deploy", 1000);
        }

        let search_bid = ledger.bid("search").unwrap();
        let deploy_bid = ledger.bid("deploy").unwrap();
        let unknown_bid = ledger.bid("unknown");

        // Higher competence = higher bid
        assert!(search_bid.0 > deploy_bid.0);
        assert!(search_bid.1 > deploy_bid.1);
        assert!(unknown_bid.is_none());
    }

    #[test]
    fn test_can_act_autonomously() {
        let mut ledger = CompetenceLedger::new();

        // Fresh domain: cannot act autonomously
        assert!(!ledger.can_act_autonomously("git"));

        // Build up competence
        for _ in 0..50 {
            ledger.record_success("git", 1000);
        }
        assert!(ledger.can_act_autonomously("git"));
    }

    #[test]
    fn test_score_clamping() {
        let mut ledger = CompetenceLedger::new();

        // Extreme failure case: score should not go below -1.0
        for _ in 0..100 {
            ledger.record_failure("bad_domain", 1000);
        }
        let score = ledger.score("bad_domain");
        assert!(score >= -1.0);

        // Extreme success case: score should not exceed 1.0
        for _ in 0..100 {
            ledger.record_success("good_domain", 1000);
        }
        let score = ledger.score("good_domain");
        assert!(score <= 1.0);
    }

    #[test]
    fn test_ledger_serde_roundtrip() {
        let mut ledger = CompetenceLedger::new();
        for _ in 0..10 {
            ledger.record_success("git", 5000);
        }
        ledger.record_failure("git", 1000);
        ledger.record_halt("git");

        let json = serde_json::to_string(&ledger).unwrap();
        let parsed: CompetenceLedger = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.domain_count(), 1);
        assert_eq!(parsed.domain("git").unwrap().successes, 10);
        assert_eq!(parsed.domain("git").unwrap().failures, 1);
        assert_eq!(parsed.domain("git").unwrap().halts, 1);
        assert!((parsed.score("git") - ledger.score("git")).abs() < 1e-10);
    }

    #[test]
    fn test_smoothing_prevents_premature_confidence() {
        let mut ledger = CompetenceLedger::new();

        // 1 success with λ=5: C = 1/(1+5) = 0.167 → Sell, not Delegate
        ledger.record_success("new_domain", 1000);
        assert!(ledger.delegation_level("new_domain") < DelegationLevel::Advise);

        // Even 2 successes: C = 2/(2+5) = 0.286 → Consult
        ledger.record_success("new_domain", 1000);
        assert!(ledger.delegation_level("new_domain") <= DelegationLevel::Consult);
    }

    #[test]
    fn test_total_attempts_and_energy() {
        let mut ledger = CompetenceLedger::new();
        ledger.record_success("a", 1000);
        ledger.record_success("b", 2000);
        ledger.record_failure("a", 500);
        ledger.record_escalation("c");

        assert_eq!(ledger.total_attempts(), 4);
        assert_eq!(ledger.total_energy_uj(), 3500);
        assert_eq!(ledger.domain_count(), 3);
    }
}
