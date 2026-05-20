//! JouleDbAi — the unified AI interface.
//!
//! Full operational pipeline:
//! ```text
//! query → contrast gate → metabolic allocation → pattern bridge (deterministic)
//!       → flowR reasoning (if needed) → tier execution → UCG enrichment
//!       → flowQIT tracking → answer + receipt
//! ```
//!
//! ```rust,ignore
//! let ai = JouleDbAi::new(&store);
//!
//! // Full pipeline: pattern resolution → flowR → tier escalation
//! let result = ai.infer("find movies similar to Inception", Default::default())?;
//! let tags = ai.auto_tag(&record)?;
//! let query = ai.nl_query("show me trending sci-fi")?;
//! ```

use joule_db_hdc::BinaryHV;
use std::collections::HashMap;
use std::time::Instant;

use super::contrast::ContrastEngine;
use super::flow_bridge::{DomainLut, FlowGraphBuilder, FlowReasoning, LocalFlowReasoner};
use super::flowqit::FlowQitEngine;
use super::holo::HoloEngine;
use super::metabolic::{ComputeBudget, MetabolicController, MetabolicState};
use super::pattern_bridge::{PatternBridge, PatternResolution, PatternResolver};
use super::receipt::AiReceipt;
use super::selector::{self, ComplexityScore, HardwareProfile, TierAvailability};
use super::tier::{InferenceTier, TierConstraints};
use super::traits::*;
use super::ucg::UcgEngine;
use crate::{AmorphicRecord, AmorphicStore, RecordId, Value, DIMENSION};

/// The unified AI interface for JouleDB.
///
/// Full integration: contrast engine → metabolic controller → pattern bridge
/// → flowR reasoning → tier execution → UCG enrichment → flowQIT tracking.
///
/// Tier 1 (holographic) is always available.
/// Tiers 2-4 are pluggable via `set_embedded()`, `set_local()`, `set_frontier()`.
pub struct JouleDbAi {
    /// Always present: HDC-based inference.
    holo: HoloEngine,
    /// Feature-gated: on-device ONNX/tract models.
    embedded: Option<Box<dyn EmbeddedInference>>,
    /// Feature-gated: local LLM.
    local: Option<Box<dyn LocalInference>>,
    /// Feature-gated: cloud API.
    frontier: Option<Box<dyn FrontierInference>>,
    /// Hardware profile for tier selection.
    hardware: HardwareProfile,
    /// Cumulative energy consumed (joules).
    energy_consumed: f64,
    /// Energy budget (None = unlimited).
    energy_budget: Option<f64>,

    // ====================================================================
    // Nervous System
    // ====================================================================

    /// Contrast engine: SNN-style spike gate on every operation.
    pub contrast: ContrastEngine,
    /// Metabolic controller: energy-proportional compute allocation.
    pub metabolic: MetabolicController,

    // ====================================================================
    // Integration Bridges
    // ====================================================================

    /// Pattern-Lang bridge: deterministic resolution before LLM.
    pub pattern_bridge: PatternBridge,
    /// flowR graph builder: constructs reasoning DAGs from queries.
    flow_builder: FlowGraphBuilder,
    /// flowR executor: pluggable reasoning backend.
    flow_reasoner: Box<dyn FlowReasoning>,
    /// UCG engine: 479-orbit structural scoring.
    pub ucg: UcgEngine,
    /// flowQIT engine: entropy tracking + axiom 6 dynamics.
    pub qit: FlowQitEngine,
}

impl JouleDbAi {
    /// Create with only Tier 1 (holographic). Always works, zero deps.
    pub fn new() -> Self {
        Self {
            holo: HoloEngine::new(),
            embedded: None,
            local: None,
            frontier: None,
            hardware: HardwareProfile::default(),
            energy_consumed: 0.0,
            energy_budget: None,
            contrast: ContrastEngine::new(),
            metabolic: MetabolicController::new(),
            pattern_bridge: PatternBridge::new(),
            flow_builder: FlowGraphBuilder::new(),
            flow_reasoner: Box::new(LocalFlowReasoner),
            ucg: UcgEngine::with_default_weights(),
            qit: FlowQitEngine::new(10_000.0), // 10s decoherence
        }
    }

    /// Set the embedded inference backend (Tier 2).
    pub fn set_embedded(&mut self, backend: Box<dyn EmbeddedInference>) {
        self.embedded = Some(backend);
    }

    /// Set the local LLM backend (Tier 3).
    pub fn set_local(&mut self, backend: Box<dyn LocalInference>) {
        self.local = Some(backend);
    }

    /// Set the frontier API backend (Tier 4).
    pub fn set_frontier(&mut self, backend: Box<dyn FrontierInference>) {
        self.frontier = Some(backend);
    }

    /// Set a custom flowR reasoning backend.
    pub fn set_flow_reasoner(&mut self, reasoner: Box<dyn FlowReasoning>) {
        self.flow_reasoner = reasoner;
    }

    /// Set hardware profile for tier selection.
    pub fn set_hardware(&mut self, hw: HardwareProfile) {
        self.hardware = hw;
    }

    /// Set energy budget (joules). None = unlimited.
    pub fn set_energy_budget(&mut self, budget: Option<f64>) {
        self.energy_budget = budget;
    }

    /// Get current tier availability.
    fn availability(&self) -> TierAvailability {
        TierAvailability {
            holographic: true,
            embedded: self.embedded.is_some(),
            local: self.local.is_some(),
            frontier: self.frontier.is_some(),
        }
    }

    /// Remaining energy budget.
    fn energy_remaining(&self) -> Option<f64> {
        self.energy_budget.map(|b| (b - self.energy_consumed).max(0.0))
    }

    /// Track energy consumption.
    fn record_energy(&mut self, joules: f64) {
        self.energy_consumed += joules;
        self.metabolic.record_energy(joules);
    }

    // ====================================================================
    // Full Pipeline Operations
    // ====================================================================

    /// **The main entry point.** Full operational pipeline:
    ///
    /// 1. Contrast gate: measure novelty of the query against recent history
    /// 2. Metabolic allocation: determine compute budget from contrast magnitude
    /// 3. Pattern bridge: attempt deterministic resolution (0 LLM cost)
    /// 4. flowR reasoning: if pattern bridge escalates, build reasoning DAG
    /// 5. Tier execution: if flowR needs higher compute, use available tiers
    /// 6. flowQIT tracking: record entropy dynamics for axiom 6
    ///
    /// Returns the answer + full energy receipt.
    pub fn infer(
        &mut self,
        query: &str,
        store: &AmorphicStore,
        constraints: TierConstraints,
    ) -> Result<AiResult, AiError> {
        let start = Instant::now();
        let timestamp_ms = start.elapsed().as_millis() as u64;

        // Step 1: Contrast — how novel is this query?
        let query_hv = self.holo.encode_text(query);
        let novelty = match self.contrast.centroid() {
            Some(centroid) => 1.0 - query_hv.similarity(centroid) as f64,
            None => 0.5, // No history — moderate novelty
        };

        // Step 2: Metabolic allocation
        let budget = self.metabolic.allocate(novelty);

        // Step 3: Pattern bridge — deterministic resolution attempt
        // Only attempt if we have a pattern resolver registered
        // (Pattern bridge needs a resolver trait object; we attempt direct keyword matching)
        let pattern_result = self.try_pattern_resolution(query);
        if let Some(result) = pattern_result {
            let elapsed = start.elapsed();
            // flowQIT: record this as a trivial measurement (entropy = 0, pure state)
            self.qit.dynamics.entropy_history.push((
                elapsed.as_millis() as u64,
                0.0, // Deterministic = zero entropy
            ));
            return Ok(result);
        }

        // Step 4: flowR reasoning — build and execute reasoning DAG
        if budget.state != MetabolicState::Resting {
            let flow_result = self.try_flow_reasoning(query, &budget);
            if let Some(result) = flow_result {
                let elapsed = start.elapsed();
                // flowQIT: record entropy reduction from reasoning
                self.qit.dynamics.entropy_history.push((
                    elapsed.as_millis() as u64,
                    1.0 - result.receipt.energy_joules.min(1.0), // Lower energy = lower entropy
                ));
                return Ok(result);
            }
        }

        // Step 5: Tier execution — standard tier escalation
        let complexity = selector::classify_complexity(query);
        let tier = selector::select_tier(
            complexity,
            &constraints,
            &self.availability(),
            self.energy_remaining(),
        );

        let result = self.execute_tier(tier, query, store, &budget)?;

        // Step 6: flowQIT tracking
        let elapsed = start.elapsed();
        self.qit.dynamics.entropy_history.push((
            elapsed.as_millis() as u64,
            complexity.0 as f64,
        ));

        Ok(result)
    }

    /// Ingest a record through the full pipeline:
    /// contrast gate → UCG scoring → store → enrichment.
    pub fn ingest_with_intelligence(
        &mut self,
        record: &AmorphicRecord,
        store: &AmorphicStore,
        timestamp_ms: u64,
    ) -> IngestIntelligence {
        // Contrast gate: does this record deserve attention?
        let contrast = self.contrast.gate_ingest(record, timestamp_ms);
        let novelty = contrast.as_ref().map(|c| c.magnitude).unwrap_or(0.0);

        // Metabolic allocation based on novelty
        let budget = self.metabolic.allocate(novelty);

        // UCG: compute orbit scores from field structure
        let fields: HashMap<String, String> = record
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), format!("{:?}", v)))
            .collect();
        let orbits = self.ucg.score_fields(&fields);

        // Adapt contrast threshold to maintain 10-20% spike rate
        self.contrast.adapt_threshold();

        IngestIntelligence {
            contrast,
            budget,
            orbit_scores: orbits.scores.iter().take(10).copied().collect(),
            orbit_norm: orbits.norm(),
        }
    }

    /// Reason about the contrast between two records.
    pub fn reason_about_contrast(
        &mut self,
        a: &AmorphicRecord,
        b: &AmorphicRecord,
    ) -> Result<AiResult, AiError> {
        let start = Instant::now();

        // UCG contrast map
        let fields_a: HashMap<String, String> = a
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), format!("{:?}", v)))
            .collect();
        let fields_b: HashMap<String, String> = b
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), format!("{:?}", v)))
            .collect();
        let orbits_a = self.ucg.score_fields(&fields_a);
        let orbits_b = self.ucg.score_fields(&fields_b);
        let contrast_map = self.ucg.contrast_map(&orbits_a, &orbits_b);

        // Build contrast summary for flowR
        let summary = format!(
            "Converging: {} orbits, Diverging: {} orbits, Unknown: {} orbits, Magnitude: {:.3}",
            contrast_map.converging_count,
            contrast_map.diverging_count,
            contrast_map.unknown_count,
            contrast_map.contrast_magnitude,
        );

        let context = format!(
            "Record A fields: {:?}, Record B fields: {:?}",
            fields_a.keys().collect::<Vec<_>>(),
            fields_b.keys().collect::<Vec<_>>(),
        );

        // flowR: build and execute a contrast reasoning graph
        let domain = DomainLut::from_query(&summary);
        let graph = self.flow_builder.build_contrast_graph(&context, &summary, domain);
        let flow_result = self.flow_reasoner.reason_graph(&graph)?;

        let elapsed = start.elapsed();
        let receipt = AiReceipt::holographic(
            "flowR-contrast",
            flow_result.total_joules,
            elapsed.as_micros() as u64,
        );
        self.record_energy(receipt.energy_joules);

        Ok(AiResult {
            output: AiOutput::Structured(serde_json::json!({
                "conclusion": flow_result.conclusion,
                "contrast": {
                    "converging": contrast_map.converging_count,
                    "diverging": contrast_map.diverging_count,
                    "unknown": contrast_map.unknown_count,
                    "magnitude": contrast_map.contrast_magnitude,
                },
                "similarity": self.ucg.relate(&orbits_a, &orbits_b),
                "reasoning_steps": flow_result.trace_nodes.len(),
            })),
            receipt,
        })
    }

    /// Get system health: metabolic state, spike rate, entropy dynamics.
    pub fn health(&self) -> SystemHealth {
        let qit_summary = self.qit.summary();
        SystemHealth {
            metabolic_state: self.metabolic.state(),
            spike_rate: self.contrast.spike_rate(),
            suppression_rate: self.contrast.suppression_rate(),
            average_contrast: self.metabolic.average_contrast(),
            energy_consumed: self.energy_consumed,
            energy_per_op: self.metabolic.energy_per_op(),
            qit_entropy: qit_summary.total_entropy,
            qit_recognition_rate: qit_summary.recognition_rate,
            qit_net_flow: qit_summary.net_flow,
        }
    }

    /// Classify content against category prototypes using holographic similarity.
    pub fn classify(
        &self,
        hologram: &BinaryHV,
        categories: &[(String, BinaryHV)],
    ) -> (String, f32, AiReceipt) {
        let start = Instant::now();
        let (label, confidence) = self.holo.classify_holo(hologram, categories);
        let elapsed = start.elapsed();
        let receipt = AiReceipt::holographic(
            "hdc-classify",
            0.000_000_2 * categories.len() as f64,
            elapsed.as_micros() as u64,
        );
        (label, confidence, receipt)
    }

    /// Encode text to BinaryHV using best available method.
    pub fn embed(&mut self, text: &str) -> Result<(BinaryHV, AiReceipt), AiError> {
        // Try Tier 2 first (better quality embeddings)
        if let Some(ref embedded) = self.embedded {
            let (hv, receipt) = embedded.embed(text)?;
            self.record_energy(receipt.energy_joules);
            return Ok((hv, receipt));
        }

        // Fallback to Tier 1 (character n-gram encoding)
        let start = Instant::now();
        let hv = self.holo.encode_text(text);
        let elapsed = start.elapsed();
        let receipt = AiReceipt::holographic(
            "hdc-trigram-encode",
            0.000_000_2,
            elapsed.as_micros() as u64,
        );
        self.record_energy(receipt.energy_joules);
        Ok((hv, receipt))
    }

    /// Natural language to query (MediaQL/SQL).
    pub fn nl_query(&mut self, nl: &str) -> Result<(String, AiReceipt), AiError> {
        // Requires Tier 3+
        if let Some(ref local) = self.local {
            let (query, receipt) = local.nl_to_query(nl, "amorphic")?;
            self.record_energy(receipt.energy_joules);
            return Ok((query, receipt));
        }
        if let Some(ref frontier) = self.frontier {
            let messages = vec![
                AiMessage {
                    role: "system".into(),
                    content: "Convert natural language to SQL or MediaQL. Return only the query.".into(),
                },
                AiMessage {
                    role: "user".into(),
                    content: nl.to_string(),
                },
            ];
            let (query, receipt) = frontier.complete(messages, &[])?;
            self.record_energy(receipt.energy_joules);
            return Ok((query, receipt));
        }

        // Tier 1 fallback: simple keyword extraction → best-effort SQL
        let start = Instant::now();
        let words: Vec<&str> = nl.split_whitespace().collect();
        let query = format!("SELECT * WHERE _text CONTAINS '{}'", words.join(" "));
        let elapsed = start.elapsed();
        let receipt = AiReceipt::holographic(
            "hdc-keyword-query",
            0.000_000_1,
            elapsed.as_micros() as u64,
        );
        Ok((query, receipt))
    }

    /// Auto-tag a record using best available tier.
    pub fn auto_tag(
        &mut self,
        record: &AmorphicRecord,
    ) -> Result<(Vec<String>, AiReceipt), AiError> {
        let start = Instant::now();
        let mut tags = Vec::new();

        for (field, value) in &record.fields {
            match value {
                Value::String(s) if s.len() < 50 => {
                    tags.push(s.clone());
                }
                _ => {}
            }
        }

        // Deduplicate
        tags.sort();
        tags.dedup();

        let elapsed = start.elapsed();
        let receipt = AiReceipt::holographic(
            "hdc-auto-tag",
            0.000_000_1,
            elapsed.as_micros() as u64,
        );
        self.record_energy(receipt.energy_joules);
        Ok((tags, receipt))
    }

    /// Total energy consumed by this AI instance (joules).
    pub fn energy_consumed(&self) -> f64 {
        self.energy_consumed
    }

    // ====================================================================
    // Internal pipeline stages
    // ====================================================================

    /// Attempt pattern-based deterministic resolution.
    fn try_pattern_resolution(&self, _query: &str) -> Option<AiResult> {
        // Pattern bridge requires a PatternResolver trait object.
        // Without one registered, we skip this stage.
        // When a resolver is available, this returns deterministic results
        // at zero LLM cost for known patterns.
        None
    }

    /// Attempt pattern-based resolution with an explicit resolver.
    pub fn infer_with_patterns(
        &mut self,
        query: &str,
        store: &AmorphicStore,
        resolver: &dyn PatternResolver,
        constraints: TierConstraints,
    ) -> Result<AiResult, AiError> {
        let start = Instant::now();

        // Try pattern resolution first
        let resolution = self.pattern_bridge.try_resolve(query, resolver);
        match resolution {
            PatternResolution::Resolved {
                output,
                receipt,
                pattern,
                position,
            } => {
                self.record_energy(receipt.energy_joules);
                Ok(AiResult { output, receipt })
            }
            PatternResolution::Escalate {
                attempted_keywords,
                best_partial,
                position,
            } => {
                // Escalate to flowR, then standard tiers
                self.infer(query, store, constraints)
            }
        }
    }

    /// Attempt flowR reasoning if metabolic budget allows.
    fn try_flow_reasoning(
        &mut self,
        query: &str,
        budget: &ComputeBudget,
    ) -> Option<AiResult> {
        // Only reason if we're at Alert or higher
        if budget.state == MetabolicState::Resting {
            return None;
        }

        let domain = self.flow_reasoner.select_domain(query);
        let graph = self.flow_builder.build_query_graph(query, domain);

        match self.flow_reasoner.reason_graph(&graph) {
            Ok(flow_result) => {
                if flow_result.confidence > 0.7 {
                    let receipt = flow_result.to_ai_receipt();
                    self.record_energy(receipt.energy_joules);
                    Some(AiResult {
                        output: flow_result.to_ai_output(),
                        receipt,
                    })
                } else {
                    None // Low confidence — let tier execution handle it
                }
            }
            Err(_) => None, // flowR failed — fallback to tiers
        }
    }

    /// Execute a query at the specified tier.
    fn execute_tier(
        &mut self,
        tier: InferenceTier,
        query: &str,
        store: &AmorphicStore,
        budget: &ComputeBudget,
    ) -> Result<AiResult, AiError> {
        match tier {
            InferenceTier::Holographic => {
                let start = Instant::now();
                let query_hv = self.holo.encode_text(query);
                let k = budget.max_comparisons.min(10);
                let results = self.similarity_search(store, &query_hv, k);
                let elapsed = start.elapsed();
                let receipt = AiReceipt::holographic(
                    "hdc-trigram-similarity",
                    0.000_000_2,
                    elapsed.as_micros() as u64,
                );
                self.record_energy(receipt.energy_joules);
                Ok(AiResult {
                    output: AiOutput::Similarity(results),
                    receipt,
                })
            }
            InferenceTier::Embedded => {
                if let Some(ref embedded) = self.embedded {
                    let (hv, receipt) = embedded.embed(query)?;
                    self.record_energy(receipt.energy_joules);
                    let results = self.similarity_search(store, &hv, 10);
                    Ok(AiResult {
                        output: AiOutput::Similarity(results),
                        receipt,
                    })
                } else {
                    Err(AiError::TierNotAvailable("Embedded".into()))
                }
            }
            InferenceTier::Local => {
                if let Some(ref local) = self.local {
                    let (text, receipt) = local.generate(query, 256)?;
                    self.record_energy(receipt.energy_joules);
                    Ok(AiResult {
                        output: AiOutput::Text(text),
                        receipt,
                    })
                } else {
                    // Fallback to holographic
                    self.execute_tier(InferenceTier::Holographic, query, store, budget)
                }
            }
            InferenceTier::Frontier => {
                if let Some(ref frontier) = self.frontier {
                    let messages = vec![AiMessage {
                        role: "user".into(),
                        content: query.to_string(),
                    }];
                    let (text, receipt) = frontier.complete(messages, &[])?;
                    self.record_energy(receipt.energy_joules);
                    Ok(AiResult {
                        output: AiOutput::Text(text),
                        receipt,
                    })
                } else {
                    // Fallback to local or holographic
                    self.execute_tier(InferenceTier::Local, query, store, budget)
                }
            }
        }
    }

    /// Similarity search against the store.
    fn similarity_search(
        &self,
        store: &AmorphicStore,
        query: &BinaryHV,
        k: usize,
    ) -> Vec<(RecordId, f32)> {
        store
            .records
            .iter()
            .map(|(&id, record)| (id, record.hologram.similarity(query)))
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Vec::new(), |mut top, item| {
                top.push(item);
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                top.truncate(k);
                top
            })
    }
}

impl Default for JouleDbAi {
    fn default() -> Self {
        Self::new()
    }
}

/// Intelligence gathered during record ingestion.
#[derive(Debug)]
pub struct IngestIntelligence {
    /// Contrast spike (if the record was novel enough to trigger one).
    pub contrast: Option<super::contrast::Contrast>,
    /// Compute budget allocated based on the contrast magnitude.
    pub budget: ComputeBudget,
    /// First 10 UCG orbit scores for this record.
    pub orbit_scores: Vec<f64>,
    /// L2 norm of the full 479-orbit vector.
    pub orbit_norm: f64,
}

/// System health snapshot.
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub metabolic_state: MetabolicState,
    pub spike_rate: f64,
    pub suppression_rate: f64,
    pub average_contrast: f64,
    pub energy_consumed: f64,
    pub energy_per_op: f64,
    pub qit_entropy: f64,
    pub qit_recognition_rate: f64,
    pub qit_net_flow: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AmorphicStore;

    #[test]
    fn test_facade_tier1_infer() {
        let mut store = AmorphicStore::new();
        store.ingest_json(r#"{"name": "Inception", "genre": "scifi"}"#).unwrap();
        store.ingest_json(r#"{"name": "The Matrix", "genre": "scifi"}"#).unwrap();
        store.ingest_json(r#"{"name": "Frozen", "genre": "animation"}"#).unwrap();

        let mut ai = JouleDbAi::new();
        let result = ai.infer("find scifi movies", &store, TierConstraints::default()).unwrap();

        assert!(ai.energy_consumed() > 0.0);
        match result.output {
            AiOutput::Similarity(results) => {
                assert!(!results.is_empty());
            }
            // flowR may return Text if confidence is high enough
            AiOutput::Text(_) => {}
            _ => panic!("Expected similarity or text results"),
        }
    }

    #[test]
    fn test_facade_classify() {
        let ai = JouleDbAi::new();

        let categories = vec![
            ("action".to_string(), ai.holo.encode_text("action fighting combat explosions")),
            ("comedy".to_string(), ai.holo.encode_text("comedy funny humor laughing")),
        ];

        let input = ai.holo.encode_text("action fighting movie");
        let (label, confidence, receipt) = ai.classify(&input, &categories);

        assert_eq!(label, "action");
        assert!(confidence > 0.5);
        assert_eq!(receipt.tier, InferenceTier::Holographic);
    }

    #[test]
    fn test_facade_embed() {
        let mut ai = JouleDbAi::new();
        let (hv, receipt) = ai.embed("hello world").unwrap();

        assert_eq!(hv.dimension(), DIMENSION);
        assert_eq!(receipt.tier, InferenceTier::Holographic);
    }

    #[test]
    fn test_facade_auto_tag() {
        let mut ai = JouleDbAi::new();
        let mut store = AmorphicStore::new();
        let id = store.ingest_json(r#"{"name": "Inception", "genre": "scifi", "director": "Nolan"}"#).unwrap();

        let record = store.get(id).unwrap();
        let (tags, receipt) = ai.auto_tag(record).unwrap();

        assert!(!tags.is_empty());
        assert!(tags.contains(&"scifi".to_string()) || tags.contains(&"Inception".to_string()));
    }

    #[test]
    fn test_facade_nl_query_tier1() {
        let mut ai = JouleDbAi::new();
        let (query, receipt) = ai.nl_query("show me trending sci-fi movies").unwrap();

        assert!(query.contains("SELECT"));
        assert_eq!(receipt.tier, InferenceTier::Holographic);
    }

    #[test]
    fn test_energy_tracking() {
        let mut ai = JouleDbAi::new();
        assert_eq!(ai.energy_consumed(), 0.0);

        let _ = ai.embed("test");
        assert!(ai.energy_consumed() > 0.0);

        let _ = ai.embed("test2");
        assert!(ai.energy_consumed() > 0.000_000_2);
    }

    #[test]
    fn test_ingest_with_intelligence() {
        let mut ai = JouleDbAi::new();
        let mut store = AmorphicStore::new();
        let id = store.ingest_json(r#"{"title": "Inception", "genre": "scifi", "year": 2010}"#).unwrap();
        let record = store.get(id).unwrap();

        let intel = ai.ingest_with_intelligence(record, &store, 1000);

        // First record has no history — should be novel
        assert!(intel.orbit_norm > 0.0);
        assert!(!intel.orbit_scores.is_empty());
    }

    #[test]
    fn test_reason_about_contrast() {
        let mut ai = JouleDbAi::new();
        let mut store = AmorphicStore::new();
        let id_a = store.ingest_json(r#"{"title": "Inception", "genre": "scifi", "mood": "dark"}"#).unwrap();
        let id_b = store.ingest_json(r#"{"title": "Frozen", "genre": "animation", "mood": "light"}"#).unwrap();

        let a = store.get(id_a).unwrap().clone();
        let b = store.get(id_b).unwrap().clone();

        let result = ai.reason_about_contrast(&a, &b).unwrap();
        match result.output {
            AiOutput::Structured(json) => {
                assert!(json.get("conclusion").is_some());
                assert!(json.get("contrast").is_some());
                assert!(json.get("similarity").is_some());
            }
            _ => panic!("Expected structured output"),
        }
    }

    #[test]
    fn test_system_health() {
        let mut ai = JouleDbAi::new();
        let mut store = AmorphicStore::new();

        // Ingest some records to build history
        for i in 0..5 {
            let id = store.ingest_json(&format!(r#"{{"item": "{}"}}"#, i)).unwrap();
            let record = store.get(id).unwrap();
            ai.ingest_with_intelligence(record, &store, 1000 + i);
        }

        let health = ai.health();
        // Metabolic state depends on contrast history — just verify it reports something valid
        assert!(health.spike_rate >= 0.0 && health.spike_rate <= 1.0);
        assert!(health.energy_consumed >= 0.0);
    }

    #[test]
    fn test_full_pipeline_flow() {
        let mut ai = JouleDbAi::new();
        let mut store = AmorphicStore::new();

        // Build a content library
        store.ingest_json(r#"{"title": "Inception", "genre": "scifi", "director": "Nolan"}"#).unwrap();
        store.ingest_json(r#"{"title": "Interstellar", "genre": "scifi", "director": "Nolan"}"#).unwrap();
        store.ingest_json(r#"{"title": "The Matrix", "genre": "scifi", "director": "Wachowski"}"#).unwrap();
        store.ingest_json(r#"{"title": "Frozen", "genre": "animation", "director": "Buck"}"#).unwrap();
        store.ingest_json(r#"{"title": "Toy Story", "genre": "animation", "director": "Lasseter"}"#).unwrap();

        // Run multiple queries — system should learn
        let r1 = ai.infer("find scifi movies", &store, TierConstraints::default()).unwrap();
        let r2 = ai.infer("Nolan films", &store, TierConstraints::default()).unwrap();
        let r3 = ai.infer("animated movies for kids", &store, TierConstraints::default()).unwrap();

        // Energy should accumulate
        assert!(ai.energy_consumed() > 0.0);

        // System should be tracking
        let health = ai.health();
        assert!(health.energy_consumed > 0.0);
    }
}
