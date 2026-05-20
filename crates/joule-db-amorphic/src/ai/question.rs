//! The Question Layer — a database that knows what it doesn't know.
//!
//! Intelligence is not the library, not the search, not the answer.
//! Intelligence is the ability to ask a question you don't know the answer to.
//!
//! This module implements the question-generating cycle:
//!
//! ```text
//! 1. DETECT    — What don't I know? (frontiers, gaps, prediction errors)
//! 2. HYPOTHESIZE — What could fill this gap? (structural decomposition, causal inference)
//! 3. QUESTION  — What specific query would test the hypothesis?
//! 4. PREDICT   — How much would the answer reduce uncertainty? (information gain)
//! 5. ROUTE     — What's the cheapest way to get the answer? (tier selection)
//! 6. OBSERVE   — Get the answer, update the model
//! 7. LEARN     — Did the frontier shrink? What new frontiers emerged?
//! ```

use joule_db_hdc::{BinaryHV, BinaryCodebook, BinaryResonator};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::facade::JouleDbAi;
use super::receipt::AiReceipt;
use super::tier::InferenceTier;
use super::traits::{AiError, AiOutput};
use crate::{AmorphicStore, RecordId, Value, DIMENSION};

// ============================================================================
// Frontier: A detected knowledge gap
// ============================================================================

/// A detected gap in the database's knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frontier {
    /// What we were examining when we found the gap
    pub context: String,
    /// Which aspects are unknown
    pub unknown_dimensions: Vec<String>,
    /// How confident are we this is a real gap (not noise)? 0.0-1.0
    pub confidence: f32,
    /// How much would resolving this change our model?
    pub estimated_impact: f64,
    /// Where was this gap detected?
    pub source: FrontierSource,
}

/// Where a knowledge gap was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierSource {
    /// Hologram entropy rising — store is degrading toward noise
    HologramDegradation,
    /// Resonator couldn't converge — compound can't be decomposed
    ResonatorFailure,
    /// User query returned empty or low-confidence results
    QueryFailure,
    /// Temporal fields have no valid entry for queried time+territory
    TemporalGap,
    /// Two similar records differ in unexpected ways
    ContrastAnomaly,
    /// A field exists in some records but not others of the same type
    SchemaSparse,
}

// ============================================================================
// Question: A formulated query targeting a frontier
// ============================================================================

/// A generated question with predicted information gain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// Natural language question
    pub text: String,
    /// Structured query (SQL/MediaQL) if applicable
    pub query: Option<String>,
    /// Hypothesis: "If we learn X, then Y would change by Z"
    pub hypothesis: String,
    /// Which gap this question targets
    pub frontier: Frontier,
    /// Expected bits of information gained
    pub predicted_info_gain: f64,
    /// Cheapest tier that could answer this
    pub recommended_tier: InferenceTier,
    /// Max energy worth spending on this answer (joules)
    pub energy_budget: f64,
}

/// The result of asking a question and observing the answer.
#[derive(Debug, Clone)]
pub struct QuestionOutcome {
    /// The question that was asked
    pub question: Question,
    /// The answer received
    pub answer: String,
    /// How much the frontier actually moved (bits)
    pub actual_info_gain: f64,
    /// Did this fully resolve the gap?
    pub frontier_resolved: bool,
    /// New gaps discovered by the answer
    pub new_frontiers: Vec<Frontier>,
    /// Energy consumed
    pub receipt: AiReceipt,
}

// ============================================================================
// QuestionEngine: The curiosity loop
// ============================================================================

/// The question-generating engine.
///
/// Scans the store for gaps, generates questions, asks them,
/// and learns from the answers.
pub struct QuestionEngine {
    /// Active frontiers, prioritized by estimated impact
    frontiers: Vec<Frontier>,
    /// History of questions and outcomes
    history: Vec<QuestionOutcome>,
    /// Energy budget for autonomous questioning (joules)
    pub question_budget: f64,
    /// Energy consumed so far
    energy_consumed: f64,
    /// Minimum confidence to pursue a frontier
    pub min_frontier_confidence: f32,
}

impl QuestionEngine {
    pub fn new(question_budget: f64) -> Self {
        Self {
            frontiers: Vec::new(),
            history: Vec::new(),
            question_budget,
            energy_consumed: 0.0,
            min_frontier_confidence: 0.3,
        }
    }

    /// Scan the store for knowledge gaps.
    pub fn detect_frontiers(&mut self, store: &AmorphicStore) -> &[Frontier] {
        self.frontiers.clear();

        // 1. Schema sparsity: fields that exist in some records but not others
        self.detect_schema_gaps(store);

        // 2. Hologram health: entropy/SNR degradation
        self.detect_hologram_degradation(store);

        // 3. Query coverage: records with very few fields
        self.detect_thin_records(store);

        // 4. Resonator failures: compounds that can't be decomposed
        self.detect_decomposition_gaps(store);

        // Sort by impact (highest first)
        self.frontiers.sort_by(|a, b| {
            b.estimated_impact
                .partial_cmp(&a.estimated_impact)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        &self.frontiers
    }

    /// Generate the best question for the highest-impact frontier.
    pub fn generate_question(&self) -> Option<Question> {
        let frontier = self.frontiers.first()?;
        Some(self.question_for_frontier(frontier))
    }

    /// Generate questions for the top-k frontiers.
    pub fn generate_questions(&self, k: usize) -> Vec<Question> {
        self.frontiers
            .iter()
            .take(k)
            .map(|f| self.question_for_frontier(f))
            .collect()
    }

    /// Run the full curiosity cycle: detect → question → ask → learn.
    pub fn curiosity_cycle(
        &mut self,
        store: &AmorphicStore,
        ai: &mut JouleDbAi,
        max_questions: usize,
    ) -> Vec<QuestionOutcome> {
        let mut outcomes = Vec::new();

        self.detect_frontiers(store);

        for _ in 0..max_questions {
            // Check energy budget
            if self.energy_consumed >= self.question_budget {
                break;
            }

            // Get the highest-impact frontier
            let question = match self.generate_question() {
                Some(q) => q,
                None => break, // No more frontiers
            };

            // Ask the question via the AI tier system
            let outcome = self.ask_and_observe(question, ai, store);

            // Learn from the outcome
            if outcome.frontier_resolved {
                // Remove the resolved frontier
                self.frontiers.retain(|f| {
                    f.context != outcome.question.frontier.context
                });
            }

            // Add any new frontiers discovered
            for new_frontier in &outcome.new_frontiers {
                self.frontiers.push(new_frontier.clone());
            }

            // Re-sort frontiers
            self.frontiers.sort_by(|a, b| {
                b.estimated_impact
                    .partial_cmp(&a.estimated_impact)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            self.energy_consumed += outcome.receipt.energy_joules;
            outcomes.push(outcome);
        }

        outcomes
    }

    /// Number of active frontiers.
    pub fn frontier_count(&self) -> usize {
        self.frontiers.len()
    }

    /// Total questions asked.
    pub fn questions_asked(&self) -> usize {
        self.history.len()
    }

    /// Energy consumed by the question engine.
    pub fn energy_consumed(&self) -> f64 {
        self.energy_consumed
    }

    // ========================================================================
    // Frontier Detection Strategies
    // ========================================================================

    /// Detect fields that appear in some records but not others.
    fn detect_schema_gaps(&mut self, store: &AmorphicStore) {
        // Build field frequency map
        let mut field_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let total_records = store.record_count();

        for record in store.records.values() {
            for field in record.fields.keys() {
                *field_counts.entry(field.clone()).or_default() += 1;
            }
        }

        // Fields present in >10% but <90% of records = schema gap
        for (field, count) in &field_counts {
            let coverage = *count as f64 / total_records.max(1) as f64;
            if coverage > 0.1 && coverage < 0.9 {
                let missing = total_records - count;
                self.frontiers.push(Frontier {
                    context: format!(
                        "Field '{}' exists in {}/{} records ({:.0}% coverage)",
                        field, count, total_records, coverage * 100.0
                    ),
                    unknown_dimensions: vec![field.clone()],
                    confidence: (1.0 - coverage) as f32,
                    estimated_impact: missing as f64 / total_records as f64,
                    source: FrontierSource::SchemaSparse,
                });
            }
        }
    }

    /// Detect hologram health degradation.
    fn detect_hologram_degradation(&mut self, store: &AmorphicStore) {
        let stats = store.stats();

        // High record count relative to dimension = potential degradation
        let capacity_ratio = stats.record_count as f64 / DIMENSION as f64;
        if capacity_ratio > 0.03 {
            // Approaching √D capacity limit
            self.frontiers.push(Frontier {
                context: format!(
                    "Store has {} records vs dimension {} (capacity ratio {:.2})",
                    stats.record_count, DIMENSION, capacity_ratio
                ),
                unknown_dimensions: vec!["partition_strategy".to_string()],
                confidence: (capacity_ratio * 10.0).min(1.0) as f32,
                estimated_impact: capacity_ratio,
                source: FrontierSource::HologramDegradation,
            });
        }
    }

    /// Detect records with very few fields (thin records).
    fn detect_thin_records(&mut self, store: &AmorphicStore) {
        let stats = store.stats();
        let avg_fields = stats.avg_fields_per_record;

        if avg_fields > 2.0 {
            // Only if there's a meaningful average
            for record in store.records.values() {
                if (record.fields.len() as f64) < avg_fields * 0.3 {
                    // Record has <30% of average fields
                    let name = record
                        .fields
                        .get("name")
                        .or_else(|| record.fields.get("_name"))
                        .map(|v| format!("{:?}", v))
                        .unwrap_or_else(|| format!("record_{}", record.id));

                    self.frontiers.push(Frontier {
                        context: format!(
                            "{} has {} fields (avg: {:.1})",
                            name,
                            record.fields.len(),
                            avg_fields
                        ),
                        unknown_dimensions: vec!["missing_fields".to_string()],
                        confidence: 0.5,
                        estimated_impact: (avg_fields - record.fields.len() as f64) / avg_fields,
                        source: FrontierSource::SchemaSparse,
                    });
                }
            }
        }
    }

    /// Detect compounds that resist factorization.
    fn detect_decomposition_gaps(&mut self, store: &AmorphicStore) {
        // Sample a few records and try to factorize them
        // against a codebook of common field values.
        // If factorization doesn't converge, it's a structural gap.

        // This is computationally expensive so we only sample
        let sample_size = store.record_count().min(10);
        let resonator = BinaryResonator::new();

        // Build a simple codebook from field names
        let field_names: Vec<&str> = store
            .records
            .values()
            .flat_map(|r| r.fields.keys().map(|k| k.as_str()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(20)
            .collect();

        if field_names.len() < 2 {
            return;
        }

        let codebook = BinaryCodebook::from_labels("fields", &field_names, DIMENSION);

        for record in store.records.values().take(sample_size) {
            let result = resonator.factorize(&record.hologram, &codebook, &codebook);

            if !result.converged && result.similarity < 0.5 {
                let name = record
                    .fields
                    .get("name")
                    .map(|v| format!("{:?}", v))
                    .unwrap_or_else(|| format!("record_{}", record.id));

                self.frontiers.push(Frontier {
                    context: format!(
                        "{} has undecomposable hologram (similarity: {:.2})",
                        name, result.similarity
                    ),
                    unknown_dimensions: vec!["structure".to_string()],
                    confidence: (1.0 - result.similarity) as f32,
                    estimated_impact: 0.3,
                    source: FrontierSource::ResonatorFailure,
                });
            }
        }
    }

    // ========================================================================
    // Question Generation
    // ========================================================================

    /// Generate a question for a specific frontier.
    fn question_for_frontier(&self, frontier: &Frontier) -> Question {
        match frontier.source {
            FrontierSource::SchemaSparse => {
                let dim = frontier
                    .unknown_dimensions
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");
                Question {
                    text: format!(
                        "What is the '{}' for records that are missing it?",
                        dim
                    ),
                    query: Some(format!(
                        "SELECT name FROM records WHERE {} IS NULL",
                        dim
                    )),
                    hypothesis: format!(
                        "If we fill in '{}' for all records, schema coverage increases and similarity search improves",
                        dim
                    ),
                    frontier: frontier.clone(),
                    predicted_info_gain: self.estimate_info_gain(frontier),
                    recommended_tier: InferenceTier::Holographic,
                    energy_budget: 0.001,
                }
            }
            FrontierSource::HologramDegradation => Question {
                text: "Should the store be partitioned to reduce hologram noise?".to_string(),
                query: None,
                hypothesis: "Partitioning would restore retrieval probability above 90%".to_string(),
                frontier: frontier.clone(),
                predicted_info_gain: self.estimate_info_gain(frontier),
                recommended_tier: InferenceTier::Holographic,
                energy_budget: 0.0001,
            },
            FrontierSource::ResonatorFailure => Question {
                text: format!(
                    "What hidden structure exists in this record? Context: {}",
                    frontier.context
                ),
                query: None,
                hypothesis: "There may be a missing codebook entry or a novel combination of known factors".to_string(),
                frontier: frontier.clone(),
                predicted_info_gain: self.estimate_info_gain(frontier),
                recommended_tier: InferenceTier::Embedded,
                energy_budget: 0.01,
            },
            FrontierSource::QueryFailure => Question {
                text: format!("Why did the query fail? Context: {}", frontier.context),
                query: None,
                hypothesis: "The query may use terms not represented in the store's vocabulary".to_string(),
                frontier: frontier.clone(),
                predicted_info_gain: self.estimate_info_gain(frontier),
                recommended_tier: InferenceTier::Holographic,
                energy_budget: 0.001,
            },
            FrontierSource::TemporalGap => {
                let dim = frontier
                    .unknown_dimensions
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("field");
                Question {
                    text: format!(
                        "What is the value of '{}' for the missing time period?",
                        dim
                    ),
                    query: Some(format!(
                        "SELECT {} FROM temporal_fields WHERE valid_at(now)",
                        dim
                    )),
                    hypothesis: format!("Filling the temporal gap in '{}' would enable rights queries for that period", dim),
                    frontier: frontier.clone(),
                    predicted_info_gain: self.estimate_info_gain(frontier),
                    recommended_tier: InferenceTier::Holographic,
                    energy_budget: 0.001,
                }
            }
            FrontierSource::ContrastAnomaly => Question {
                text: format!(
                    "What explains the unexpected difference? Context: {}",
                    frontier.context
                ),
                query: None,
                hypothesis: "The contrast may reveal a missing category or mislabeled record".to_string(),
                frontier: frontier.clone(),
                predicted_info_gain: self.estimate_info_gain(frontier),
                recommended_tier: InferenceTier::Embedded,
                energy_budget: 0.01,
            },
        }
    }

    /// Estimate information gain from resolving a frontier.
    ///
    /// info_gain ≈ log2(unknown_dims) × impact
    fn estimate_info_gain(&self, frontier: &Frontier) -> f64 {
        let dims = frontier.unknown_dimensions.len().max(1) as f64;
        dims.log2().max(0.1) * frontier.estimated_impact
    }

    // ========================================================================
    // Ask and Observe
    // ========================================================================

    /// Ask a question and observe the result.
    fn ask_and_observe(
        &mut self,
        question: Question,
        ai: &mut JouleDbAi,
        store: &AmorphicStore,
    ) -> QuestionOutcome {
        let query_text = question
            .query
            .as_deref()
            .unwrap_or(&question.text);

        // Use the AI tier system to get an answer
        let result = ai.infer(
            query_text,
            store,
            super::tier::TierConstraints::default(),
        );

        let (answer_text, receipt) = match result {
            Ok(result) => {
                let text = match &result.output {
                    AiOutput::Text(t) => t.clone(),
                    AiOutput::Similarity(sims) => {
                        format!("Found {} similar records", sims.len())
                    }
                    AiOutput::Tags(tags) => tags.join(", "),
                    AiOutput::Classification { label, confidence } => {
                        format!("{} ({:.0}%)", label, confidence * 100.0)
                    }
                    other => format!("{:?}", other),
                };
                (text, result.receipt)
            }
            Err(_) => (
                "Question could not be answered".to_string(),
                AiReceipt::holographic("error", 0.0, 0),
            ),
        };

        // Estimate actual information gain
        // (In a full system, this would re-measure hologram health)
        let actual_gain = if answer_text.contains("not") || answer_text.contains("error") {
            0.0
        } else {
            question.predicted_info_gain * 0.5 // Conservative estimate
        };

        let frontier_resolved = actual_gain > question.predicted_info_gain * 0.3;

        let outcome = QuestionOutcome {
            question,
            answer: answer_text,
            actual_info_gain: actual_gain,
            frontier_resolved,
            new_frontiers: vec![], // Would be detected on next cycle
            receipt,
        };

        self.history.push(outcome.clone());
        outcome
    }
}

impl Default for QuestionEngine {
    fn default() -> Self {
        Self::new(1.0) // 1 joule default budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_schema_gaps() {
        let mut store = AmorphicStore::new();

        // Insert records with inconsistent schemas
        store.ingest_json(r#"{"name": "A", "genre": "action", "rating": 8}"#).unwrap();
        store.ingest_json(r#"{"name": "B", "genre": "comedy", "rating": 7}"#).unwrap();
        store.ingest_json(r#"{"name": "C", "genre": "drama"}"#).unwrap(); // Missing rating
        store.ingest_json(r#"{"name": "D"}"#).unwrap(); // Missing genre AND rating

        let mut engine = QuestionEngine::new(1.0);
        let frontiers = engine.detect_frontiers(&store);

        // Should detect gaps in "genre" and "rating"
        assert!(!frontiers.is_empty(), "Should detect schema gaps");

        // At least one frontier should be about missing fields
        assert!(frontiers.iter().any(|f| f.source == FrontierSource::SchemaSparse));
    }

    #[test]
    fn test_generate_question_from_frontier() {
        let engine = QuestionEngine::new(1.0);

        let frontier = Frontier {
            context: "Field 'rating' exists in 2/4 records".into(),
            unknown_dimensions: vec!["rating".into()],
            confidence: 0.5,
            estimated_impact: 0.5,
            source: FrontierSource::SchemaSparse,
        };

        let question = engine.question_for_frontier(&frontier);

        assert!(question.text.contains("rating"));
        assert!(question.hypothesis.contains("rating"));
        assert!(question.predicted_info_gain > 0.0);
    }

    #[test]
    fn test_curiosity_cycle() {
        let mut store = AmorphicStore::new();

        // Create a store with known gaps
        store.ingest_json(r#"{"name": "Complete", "genre": "action", "year": 2020, "rating": 9}"#).unwrap();
        store.ingest_json(r#"{"name": "Partial", "genre": "comedy"}"#).unwrap();
        store.ingest_json(r#"{"name": "Minimal"}"#).unwrap();

        let mut ai = JouleDbAi::new();
        let mut engine = QuestionEngine::new(0.01); // Small budget

        let outcomes = engine.curiosity_cycle(&store, &mut ai, 5);

        // Should have asked at least one question
        assert!(!outcomes.is_empty(), "Should ask at least one question");

        // Questions should target the gaps
        for outcome in &outcomes {
            assert!(!outcome.question.text.is_empty());
            assert!(outcome.receipt.energy_joules >= 0.0);
        }

        // Energy should be tracked
        assert!(engine.energy_consumed() > 0.0);
    }

    #[test]
    fn test_info_gain_estimation() {
        let engine = QuestionEngine::new(1.0);

        let high_impact = Frontier {
            context: "test".into(),
            unknown_dimensions: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            confidence: 0.9,
            estimated_impact: 0.8,
            source: FrontierSource::SchemaSparse,
        };

        let low_impact = Frontier {
            context: "test".into(),
            unknown_dimensions: vec!["x".into()],
            confidence: 0.3,
            estimated_impact: 0.1,
            source: FrontierSource::SchemaSparse,
        };

        let high_gain = engine.estimate_info_gain(&high_impact);
        let low_gain = engine.estimate_info_gain(&low_impact);

        assert!(
            high_gain > low_gain,
            "More unknown dims + higher impact = more info gain: {} vs {}",
            high_gain,
            low_gain
        );
    }

    #[test]
    fn test_energy_budget_respected() {
        let mut store = AmorphicStore::new();
        for i in 0..20 {
            store.ingest_json(&format!(r#"{{"name": "item_{}", "score": {}}}"#, i, i)).unwrap();
        }

        let mut ai = JouleDbAi::new();
        let mut engine = QuestionEngine::new(0.000_001); // Tiny budget

        let outcomes = engine.curiosity_cycle(&store, &mut ai, 100);

        // Should stop early due to budget
        assert!(outcomes.len() < 100);
    }

    #[test]
    fn test_frontier_priority() {
        let mut store = AmorphicStore::new();

        // High-coverage field (present in 3/4)
        store.ingest_json(r#"{"name": "A", "genre": "x", "rare": "y"}"#).unwrap();
        store.ingest_json(r#"{"name": "B", "genre": "x"}"#).unwrap();
        store.ingest_json(r#"{"name": "C", "genre": "x"}"#).unwrap();
        store.ingest_json(r#"{"name": "D"}"#).unwrap();

        let mut engine = QuestionEngine::new(1.0);
        engine.detect_frontiers(&store);

        // Frontiers should be sorted by impact (highest first)
        if engine.frontiers.len() >= 2 {
            assert!(engine.frontiers[0].estimated_impact >= engine.frontiers[1].estimated_impact);
        }
    }
}
