//! Promotion Pipeline — convergent replacement of probabilistic inference
//! with deterministic execution.
//!
//! From Axiom A5 (CONVERGENCE): every LLM invocation that produces a validated
//! result MUST generate a candidate for promotion to deterministic execution.
//! The system's reliance on probabilistic inference monotonically decreases
//! over time.
//!
//! The pipeline:
//! ```text
//!  LLM invocation (kJ)
//!    → validated result
//!      → PromotionCandidate
//!        → compiled to deterministic path (µJ)
//!          → future identical requests skip LLM entirely
//! ```
//!
//! The promotion curve P(t) = 1 − (N_llm(t) / N_total(t)) measures what
//! fraction of requests are served deterministically. For finite task domains,
//! P(t) → 1 − ε as t → ∞.

use crate::competence::DomainId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// What the LLM produced that can be compiled into a deterministic path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionTarget {
    /// A tool invocation pattern (e.g., "run git status" for "what's changed?").
    /// Post-promotion cost: L0 (µJ) — direct parse grammar rule.
    ToolPattern,
    /// A plan template (e.g., "test → commit → push" for "ship this").
    /// Post-promotion cost: L0 (µJ) — plan library entry.
    PlanTemplate,
    /// A concept mapping (e.g., "deployment" → domain:ops).
    /// Post-promotion cost: L1 (µJ) — formula score update.
    ConceptMapping,
    /// An intent resolution (e.g., "clean up" → "remove dead code").
    /// Post-promotion cost: L0 (µJ) — resolve lookup table.
    IntentResolution,
    /// A novel reasoning chain that cannot be promoted (remains L3-L4, kJ).
    NovelReasoning,
}

impl PromotionTarget {
    /// Whether this target can be promoted to deterministic execution.
    pub fn is_promotable(&self) -> bool {
        !matches!(self, Self::NovelReasoning)
    }

    /// Estimated post-promotion energy cost in microjoules.
    pub fn post_promotion_cost_uj(&self) -> u64 {
        match self {
            Self::ToolPattern => 1,         // ~1 µJ: hash lookup
            Self::PlanTemplate => 1,        // ~1 µJ: template expansion
            Self::ConceptMapping => 10,     // ~10 µJ: similarity computation
            Self::IntentResolution => 1,    // ~1 µJ: table lookup
            Self::NovelReasoning => 1_000_000, // ~1 kJ: still needs LLM
        }
    }
}

/// A candidate for promotion from probabilistic to deterministic execution.
///
/// Generated every time an LLM invocation produces a validated result (A5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionCandidate {
    /// Unique candidate identifier.
    pub candidate_id: String,
    /// The input pattern that triggered the LLM invocation.
    pub trigger_pattern: String,
    /// What the LLM produced.
    pub resolution: String,
    /// Which competence domain this belongs to.
    pub domain: DomainId,
    /// What type of deterministic path this can be compiled to.
    pub target: PromotionTarget,
    /// Whether the result was validated (EVALUATE confirmed success).
    pub validated: bool,
    /// Energy cost of the original LLM invocation (microjoules).
    pub llm_energy_uj: u64,
    /// How many times this pattern has been seen.
    pub occurrences: u32,
    /// Timestamp when first seen (nanos since epoch).
    pub first_seen_ns: u64,
    /// Timestamp when last seen.
    pub last_seen_ns: u64,
}

/// A compiled deterministic execution path.
///
/// Once promoted, this replaces the LLM invocation for matching inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterministicPath {
    /// The trigger pattern (normalized).
    pub trigger: String,
    /// The compiled resolution.
    pub resolution: String,
    /// Domain this path serves.
    pub domain: DomainId,
    /// What type of path this is.
    pub target: PromotionTarget,
    /// Estimated energy cost per invocation (µJ).
    pub energy_uj: u64,
    /// Number of times this path has been used since promotion.
    pub invocations: u64,
    /// Total energy saved vs. LLM invocation (µJ).
    pub energy_saved_uj: u64,
    /// Energy cost of the original LLM invocation that produced this.
    pub original_llm_energy_uj: u64,
    /// When this path was promoted.
    pub promoted_at_ns: u64,
}

impl DeterministicPath {
    /// Record a successful invocation of this path.
    pub fn record_invocation(&mut self) {
        self.invocations += 1;
        // Energy saved = what the LLM would have cost minus what we actually cost
        self.energy_saved_uj = self
            .energy_saved_uj
            .saturating_add(self.original_llm_energy_uj.saturating_sub(self.energy_uj));
    }
}

/// Configuration for the promotion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionConfig {
    /// Minimum occurrences before a candidate is eligible for promotion.
    pub min_occurrences: u32,
    /// Minimum validation rate (validated / occurrences) for promotion.
    pub min_validation_rate: f64,
    /// Whether to auto-promote candidates that meet thresholds.
    pub auto_promote: bool,
}

impl Default for PromotionConfig {
    fn default() -> Self {
        Self {
            min_occurrences: 3,
            min_validation_rate: 1.0, // All occurrences must validate
            auto_promote: true,
        }
    }
}

/// The promotion pipeline: tracks candidates and compiles deterministic paths.
pub struct PromotionPipeline {
    /// Configuration.
    config: PromotionConfig,
    /// Pending candidates (keyed by normalized trigger pattern).
    candidates: HashMap<String, PromotionCandidate>,
    /// Promoted deterministic paths (keyed by normalized trigger pattern).
    promoted: HashMap<String, DeterministicPath>,
    /// Counter for total LLM invocations.
    total_llm_invocations: u64,
    /// Counter for total requests (LLM + deterministic).
    total_requests: u64,
}

impl PromotionPipeline {
    pub fn new() -> Self {
        Self {
            config: PromotionConfig::default(),
            candidates: HashMap::new(),
            promoted: HashMap::new(),
            total_llm_invocations: 0,
            total_requests: 0,
        }
    }

    pub fn with_config(config: PromotionConfig) -> Self {
        Self {
            config,
            candidates: HashMap::new(),
            promoted: HashMap::new(),
            total_llm_invocations: 0,
            total_requests: 0,
        }
    }

    /// Try to resolve a request deterministically (skip LLM).
    ///
    /// Returns the deterministic resolution if a promoted path exists.
    /// Returns None if the request must go to the LLM.
    pub fn try_resolve(&mut self, trigger: &str) -> Option<&str> {
        self.total_requests += 1;
        let key = normalize_trigger(trigger);

        if let Some(path) = self.promoted.get_mut(&key) {
            path.record_invocation();
            return Some(&path.resolution);
        }

        // No deterministic path — this will need LLM
        self.total_llm_invocations += 1;
        None
    }

    /// Record an LLM invocation result as a promotion candidate.
    ///
    /// This is the A5 obligation: every validated LLM result generates a candidate.
    /// Each call counts as both a request and an LLM invocation (it required inference).
    pub fn record_llm_result(
        &mut self,
        trigger: &str,
        resolution: &str,
        domain: &str,
        target: PromotionTarget,
        validated: bool,
        llm_energy_uj: u64,
    ) -> Option<String> {
        // This was an LLM invocation, so count it
        self.total_requests += 1;
        self.total_llm_invocations += 1;

        let key = normalize_trigger(trigger);
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Update existing candidate or create new one
        let candidate = self
            .candidates
            .entry(key.clone())
            .or_insert_with(|| PromotionCandidate {
                candidate_id: format!("pc-{}", now),
                trigger_pattern: key.clone(),
                resolution: resolution.to_string(),
                domain: domain.to_string(),
                target: target.clone(),
                validated: false,
                llm_energy_uj,
                occurrences: 0,
                first_seen_ns: now,
                last_seen_ns: now,
            });

        candidate.occurrences += 1;
        candidate.last_seen_ns = now;
        if validated {
            candidate.validated = true;
        }
        // Update resolution to latest (in case it improved)
        candidate.resolution = resolution.to_string();

        // Check if candidate is ready for promotion
        if self.config.auto_promote && self.is_promotable(&key) {
            return self.promote(&key);
        }

        None
    }

    /// Check if a candidate is ready for promotion.
    fn is_promotable(&self, key: &str) -> bool {
        if let Some(candidate) = self.candidates.get(key) {
            candidate.target.is_promotable()
                && candidate.validated
                && candidate.occurrences >= self.config.min_occurrences
        } else {
            false
        }
    }

    /// Promote a candidate to a deterministic path.
    ///
    /// Returns the path key on success.
    fn promote(&mut self, key: &str) -> Option<String> {
        let candidate = self.candidates.remove(key)?;

        if !candidate.target.is_promotable() {
            // Put it back — novel reasoning can't be promoted
            self.candidates.insert(key.to_string(), candidate);
            return None;
        }

        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let path = DeterministicPath {
            trigger: key.to_string(),
            resolution: candidate.resolution,
            domain: candidate.domain,
            energy_uj: candidate.target.post_promotion_cost_uj(),
            target: candidate.target,
            invocations: 0,
            energy_saved_uj: 0,
            original_llm_energy_uj: candidate.llm_energy_uj,
            promoted_at_ns: now,
        };

        self.promoted.insert(key.to_string(), path);
        Some(key.to_string())
    }

    /// Manually promote a candidate (for HostApproval workflows).
    pub fn force_promote(&mut self, key: &str) -> Option<String> {
        self.promote(key)
    }

    /// Invalidate a promoted path (if the deterministic resolution is found wrong).
    pub fn invalidate(&mut self, key: &str) -> bool {
        self.promoted.remove(key).is_some()
    }

    // --- Metrics (§5.2) ---

    /// The promotion curve P(t) = 1 − (N_llm / N_total).
    ///
    /// Measures the fraction of requests served deterministically.
    /// For finite task domain diversity, P(t) → 1 − ε as t → ∞.
    pub fn promotion_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        1.0 - (self.total_llm_invocations as f64 / self.total_requests as f64)
    }

    /// Energy ratio: E_subsidiary / E_centralized.
    ///
    /// At P=0: ratio=1.0 (equivalent to pure LLM).
    /// At P=0.99: ratio≈0.01 (100× reduction).
    /// At P→1: ratio→C_det/C_llm ≈ 10⁻⁹ (theoretical bound).
    pub fn energy_ratio(&self, c_llm_uj: u64, c_det_uj: u64) -> f64 {
        let p = self.promotion_rate();
        (1.0 - p) + p * (c_det_uj as f64 / c_llm_uj as f64)
    }

    /// Total energy saved by all promoted paths.
    pub fn total_energy_saved_uj(&self) -> u64 {
        self.promoted.values().map(|p| p.energy_saved_uj).sum()
    }

    /// Number of pending candidates.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// Number of promoted deterministic paths.
    pub fn promoted_count(&self) -> usize {
        self.promoted.len()
    }

    /// Get a promoted path.
    pub fn get_promoted(&self, key: &str) -> Option<&DeterministicPath> {
        self.promoted.get(key)
    }

    /// Get a candidate.
    pub fn get_candidate(&self, key: &str) -> Option<&PromotionCandidate> {
        self.candidates.get(key)
    }

    /// List all promoted paths.
    pub fn all_promoted(&self) -> Vec<&DeterministicPath> {
        self.promoted.values().collect()
    }

    /// Total requests processed.
    pub fn total_requests(&self) -> u64 {
        self.total_requests
    }

    /// Total LLM invocations.
    pub fn total_llm_invocations(&self) -> u64 {
        self.total_llm_invocations
    }

    /// Summary of promotion pipeline health.
    pub fn summary(&self) -> PromotionSummary {
        PromotionSummary {
            total_requests: self.total_requests,
            llm_invocations: self.total_llm_invocations,
            deterministic_invocations: self.total_requests - self.total_llm_invocations,
            promotion_rate: self.promotion_rate(),
            candidates_pending: self.candidates.len(),
            paths_promoted: self.promoted.len(),
            total_energy_saved_uj: self.total_energy_saved_uj(),
        }
    }
}

impl Default for PromotionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of the promotion pipeline state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionSummary {
    pub total_requests: u64,
    pub llm_invocations: u64,
    pub deterministic_invocations: u64,
    pub promotion_rate: f64,
    pub candidates_pending: usize,
    pub paths_promoted: usize,
    pub total_energy_saved_uj: u64,
}

/// Normalize a trigger pattern for consistent matching.
fn normalize_trigger(trigger: &str) -> String {
    trigger.trim().to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promotion_target_promotable() {
        assert!(PromotionTarget::ToolPattern.is_promotable());
        assert!(PromotionTarget::PlanTemplate.is_promotable());
        assert!(PromotionTarget::ConceptMapping.is_promotable());
        assert!(PromotionTarget::IntentResolution.is_promotable());
        assert!(!PromotionTarget::NovelReasoning.is_promotable());
    }

    #[test]
    fn test_promotion_target_cost() {
        assert_eq!(PromotionTarget::ToolPattern.post_promotion_cost_uj(), 1);
        assert_eq!(PromotionTarget::NovelReasoning.post_promotion_cost_uj(), 1_000_000);
    }

    #[test]
    fn test_empty_pipeline_no_resolution() {
        let mut pipeline = PromotionPipeline::new();
        assert!(pipeline.try_resolve("what's changed?").is_none());
        assert_eq!(pipeline.total_requests(), 1);
        assert_eq!(pipeline.total_llm_invocations(), 1);
    }

    #[test]
    fn test_record_and_auto_promote() {
        let mut pipeline = PromotionPipeline::new();

        // Record the same pattern 3 times (min_occurrences=3)
        for _ in 0..2 {
            pipeline.record_llm_result(
                "what's changed?",
                "git status",
                "git",
                PromotionTarget::ToolPattern,
                true,
                1_000_000, // 1 kJ
            );
        }
        assert_eq!(pipeline.candidate_count(), 1);
        assert_eq!(pipeline.promoted_count(), 0);

        // Third occurrence triggers auto-promotion
        let promoted = pipeline.record_llm_result(
            "what's changed?",
            "git status",
            "git",
            PromotionTarget::ToolPattern,
            true,
            1_000_000,
        );
        assert!(promoted.is_some());
        assert_eq!(pipeline.promoted_count(), 1);
        assert_eq!(pipeline.candidate_count(), 0);
    }

    #[test]
    fn test_deterministic_resolution_after_promotion() {
        let mut pipeline = PromotionPipeline::new();

        // Promote a pattern
        for _ in 0..3 {
            pipeline.record_llm_result(
                "what's changed?",
                "git status",
                "git",
                PromotionTarget::ToolPattern,
                true,
                1_000_000,
            );
        }

        // Now it should resolve deterministically
        let result = pipeline.try_resolve("what's changed?");
        assert_eq!(result, Some("git status"));

        // And track energy savings
        let path = pipeline.get_promoted("what's changed?").unwrap();
        assert_eq!(path.invocations, 1);
        assert!(path.energy_saved_uj > 0);
    }

    #[test]
    fn test_promotion_curve() {
        let mut pipeline = PromotionPipeline::new();

        // Promote a pattern
        for _ in 0..3 {
            pipeline.record_llm_result(
                "status",
                "git status",
                "git",
                PromotionTarget::ToolPattern,
                true,
                1_000_000,
            );
        }

        // 10 total requests, 7 deterministic (after promotion)
        for _ in 0..7 {
            pipeline.try_resolve("status");
        }
        // 3 LLM calls during promotion + 0 after = 3 LLM out of 10 total
        // P = 1 - 3/10 = 0.7
        assert!((pipeline.promotion_rate() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_novel_reasoning_not_promoted() {
        let mut pipeline = PromotionPipeline::new();

        for _ in 0..10 {
            pipeline.record_llm_result(
                "explain quantum entanglement",
                "long reasoning chain...",
                "physics",
                PromotionTarget::NovelReasoning,
                true,
                5_000_000,
            );
        }

        // Should remain as candidate, never promoted
        assert_eq!(pipeline.promoted_count(), 0);
        assert_eq!(pipeline.candidate_count(), 1);
    }

    #[test]
    fn test_unvalidated_not_promoted() {
        let mut pipeline = PromotionPipeline::new();

        // Record without validation
        for _ in 0..5 {
            pipeline.record_llm_result(
                "deploy to prod",
                "kubectl apply",
                "ops",
                PromotionTarget::ToolPattern,
                false, // not validated
                1_000_000,
            );
        }

        // Not promoted because never validated
        assert_eq!(pipeline.promoted_count(), 0);
    }

    #[test]
    fn test_invalidate_promoted_path() {
        let mut pipeline = PromotionPipeline::new();

        for _ in 0..3 {
            pipeline.record_llm_result(
                "check health",
                "curl /health",
                "ops",
                PromotionTarget::ToolPattern,
                true,
                1_000_000,
            );
        }
        assert_eq!(pipeline.promoted_count(), 1);

        // Invalidate because the resolution was wrong
        assert!(pipeline.invalidate("check health"));
        assert_eq!(pipeline.promoted_count(), 0);

        // Should no longer resolve deterministically
        assert!(pipeline.try_resolve("check health").is_none());
    }

    #[test]
    fn test_energy_ratio() {
        let mut pipeline = PromotionPipeline::new();

        // No promotion yet → ratio = 1.0
        let ratio = pipeline.energy_ratio(1_000_000, 1);
        assert!((ratio - 1.0).abs() < 1e-10);

        // Promote and use
        for _ in 0..3 {
            pipeline.record_llm_result(
                "status",
                "git status",
                "git",
                PromotionTarget::ToolPattern,
                true,
                1_000_000,
            );
        }
        for _ in 0..97 {
            pipeline.try_resolve("status");
        }
        // 3 LLM out of 100 total → P = 0.97
        // E_ratio = (1-0.97) + 0.97*(1/1000000) ≈ 0.03
        let ratio = pipeline.energy_ratio(1_000_000, 1);
        assert!(ratio < 0.04);
        assert!(ratio > 0.02);
    }

    #[test]
    fn test_normalize_trigger() {
        assert_eq!(normalize_trigger("  What's Changed?  "), "what's changed?");
        assert_eq!(normalize_trigger("GIT STATUS"), "git status");
    }

    #[test]
    fn test_summary() {
        let mut pipeline = PromotionPipeline::new();

        for _ in 0..3 {
            pipeline.record_llm_result(
                "deploy",
                "kubectl apply",
                "ops",
                PromotionTarget::ToolPattern,
                true,
                1_000_000,
            );
        }
        for _ in 0..7 {
            pipeline.try_resolve("deploy");
        }

        let summary = pipeline.summary();
        assert_eq!(summary.total_requests, 10);
        assert_eq!(summary.llm_invocations, 3);
        assert_eq!(summary.deterministic_invocations, 7);
        assert_eq!(summary.paths_promoted, 1);
        assert!(summary.total_energy_saved_uj > 0);
    }

    #[test]
    fn test_force_promote() {
        let config = PromotionConfig {
            min_occurrences: 10, // high threshold
            auto_promote: false,
            ..Default::default()
        };
        let mut pipeline = PromotionPipeline::with_config(config);

        // Record once — not enough for auto-promote
        pipeline.record_llm_result(
            "build",
            "cargo build",
            "rust",
            PromotionTarget::ToolPattern,
            true,
            1_000_000,
        );
        assert_eq!(pipeline.promoted_count(), 0);

        // Force promote
        pipeline.force_promote("build");
        assert_eq!(pipeline.promoted_count(), 1);
    }

    #[test]
    fn test_deterministic_path_energy_tracking() {
        let mut path = DeterministicPath {
            trigger: "test".into(),
            resolution: "cargo test".into(),
            domain: "rust".into(),
            target: PromotionTarget::ToolPattern,
            energy_uj: 1,
            invocations: 0,
            energy_saved_uj: 0,
            original_llm_energy_uj: 1_000_000,
            promoted_at_ns: 0,
        };

        path.record_invocation();
        assert_eq!(path.invocations, 1);
        assert_eq!(path.energy_saved_uj, 999_999); // 1M - 1

        path.record_invocation();
        assert_eq!(path.invocations, 2);
        assert_eq!(path.energy_saved_uj, 1_999_998);
    }

    #[test]
    fn test_multiple_domains_promoted() {
        let mut pipeline = PromotionPipeline::new();

        // Promote in two different domains
        for _ in 0..3 {
            pipeline.record_llm_result("status", "git status", "git", PromotionTarget::ToolPattern, true, 1_000_000);
            pipeline.record_llm_result("build", "cargo build", "rust", PromotionTarget::ToolPattern, true, 1_000_000);
        }

        assert_eq!(pipeline.promoted_count(), 2);
        assert_eq!(pipeline.try_resolve("status"), Some("git status"));
        assert_eq!(pipeline.try_resolve("build"), Some("cargo build"));
    }

    #[test]
    fn test_candidate_serde_roundtrip() {
        let candidate = PromotionCandidate {
            candidate_id: "pc-1".into(),
            trigger_pattern: "deploy".into(),
            resolution: "kubectl apply".into(),
            domain: "ops".into(),
            target: PromotionTarget::ToolPattern,
            validated: true,
            llm_energy_uj: 1_000_000,
            occurrences: 5,
            first_seen_ns: 100,
            last_seen_ns: 500,
        };
        let json = serde_json::to_string(&candidate).unwrap();
        let parsed: PromotionCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.trigger_pattern, "deploy");
        assert_eq!(parsed.occurrences, 5);
        assert!(parsed.validated);
    }

    #[test]
    fn test_summary_serde() {
        let summary = PromotionSummary {
            total_requests: 1000,
            llm_invocations: 50,
            deterministic_invocations: 950,
            promotion_rate: 0.95,
            candidates_pending: 3,
            paths_promoted: 47,
            total_energy_saved_uj: 950_000_000,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: PromotionSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_requests, 1000);
        assert!((parsed.promotion_rate - 0.95).abs() < 1e-10);
    }
}
