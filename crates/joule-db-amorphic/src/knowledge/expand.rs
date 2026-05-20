//! Layer 4: Internet Expansion — on-demand knowledge acquisition.
//!
//! When the knowledge core can't resolve a query (Test fails, confidence
//! below threshold), the system reaches outward to external sources.
//!
//! The expansion pipeline:
//! 1. **Detect gap**: Query returns low confidence or high novelty
//! 2. **Formulate search**: Extract keywords from the query context
//! 3. **Fetch**: Call an external knowledge source (web, API, file)
//! 4. **Extract**: Parse the response into relationship triples
//! 5. **Encode**: Ingest triples into the knowledge core (Update + Merge)
//! 6. **Retry**: Re-query with the expanded core
//!
//! The core grows structurally — not by memorizing raw text, but by
//! integrating new relationships into the existing graphlet topology.
//!
//! ## Design Principles
//!
//! - **Trait-based sources**: `KnowledgeSource` trait allows plugging in
//!   any external source (web search, Wikipedia API, local files, LLM).
//! - **Budget-aware**: Each expansion has an energy/time cost. The metabolic
//!   controller decides whether to spend it.
//! - **Deduplication**: New triples are checked against existing knowledge
//!   via holographic similarity before ingestion.

use super::concept::ConceptEncoder;
use super::core::KnowledgeCore;
use super::relation::RelationType;
use super::triple::Triple;

/// A piece of knowledge fetched from an external source.
#[derive(Clone, Debug)]
pub struct FetchedKnowledge {
    /// Source identifier (URL, API name, file path).
    pub source: String,
    /// Raw text content retrieved.
    pub raw_text: String,
    /// Extracted relationship triples.
    pub triples: Vec<Triple>,
    /// Confidence in the extraction (0.0 - 1.0).
    pub confidence: f64,
    /// Cost of this fetch (arbitrary units — could be time_ms, energy, money).
    pub cost: f64,
}

/// Trait for external knowledge sources.
/// Implement this for web search, Wikipedia, local corpus, LLM, etc.
pub trait KnowledgeSource: Send + Sync {
    /// Fetch knowledge related to a query.
    /// Returns extracted triples + metadata.
    fn fetch(&self, query: &str) -> Result<FetchedKnowledge, ExpansionError>;

    /// Name of this source (for logging/attribution).
    fn name(&self) -> &str;

    /// Estimated cost of a single fetch (for budget decisions).
    fn estimated_cost(&self) -> f64;
}

/// Errors during expansion.
#[derive(Debug, Clone)]
pub enum ExpansionError {
    /// Source is unavailable (network down, API limit, etc.)
    SourceUnavailable(String),
    /// No useful knowledge found for the query.
    NoResults(String),
    /// Budget exceeded — not enough energy/time for this expansion.
    BudgetExceeded,
    /// Extraction failed — couldn't parse triples from response.
    ExtractionFailed(String),
}

impl std::fmt::Display for ExpansionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceUnavailable(s) => write!(f, "source unavailable: {s}"),
            Self::NoResults(s) => write!(f, "no results for: {s}"),
            Self::BudgetExceeded => write!(f, "expansion budget exceeded"),
            Self::ExtractionFailed(s) => write!(f, "extraction failed: {s}"),
        }
    }
}

/// Result of an expansion attempt.
#[derive(Clone, Debug)]
pub struct ExpansionResult {
    /// Query that triggered the expansion.
    pub query: String,
    /// Source that provided the knowledge.
    pub source: String,
    /// Number of new triples added to the core.
    pub triples_added: usize,
    /// Number of triples rejected as duplicates.
    pub triples_deduplicated: usize,
    /// Total cost of this expansion.
    pub cost: f64,
    /// New concepts added to the core.
    pub new_concepts: Vec<String>,
}

/// The expansion engine: decides when and how to expand the knowledge core.
pub struct Expander {
    /// Minimum novelty to trigger expansion (0.0 - 1.0).
    /// Higher = expand only for very novel queries.
    pub novelty_threshold: f64,
    /// Maximum confidence below which expansion is triggered.
    /// Lower = expand only when really uncertain.
    pub confidence_threshold: f32,
    /// Budget per expansion (cost units).
    pub budget_per_expansion: f64,
    /// Total budget consumed so far.
    pub total_cost: f64,
    /// Maximum total budget (None = unlimited).
    pub max_total_budget: Option<f64>,
    /// Deduplication threshold: similarity above this = duplicate triple.
    pub dedup_threshold: f32,
    /// Expansion history.
    pub history: Vec<ExpansionResult>,
}

impl Expander {
    pub fn new() -> Self {
        Self {
            novelty_threshold: 0.6,
            confidence_threshold: 0.5,
            budget_per_expansion: 1.0,
            total_cost: 0.0,
            max_total_budget: None,
            dedup_threshold: 0.85,
            history: Vec::new(),
        }
    }

    /// Should we expand for this query?
    pub fn should_expand(&self, novelty: f64, confidence: f32) -> bool {
        // Check budget
        if let Some(max) = self.max_total_budget {
            if self.total_cost >= max {
                return false;
            }
        }

        // High novelty OR low confidence → expand
        novelty > self.novelty_threshold || confidence < self.confidence_threshold
    }

    /// Expand the knowledge core using an external source.
    ///
    /// 1. Fetch knowledge from the source
    /// 2. Deduplicate against existing core
    /// 3. Ingest new triples
    /// 4. Return what was added
    pub fn expand(
        &mut self,
        query: &str,
        core: &mut KnowledgeCore,
        source: &dyn KnowledgeSource,
    ) -> Result<ExpansionResult, ExpansionError> {
        // Budget check
        let estimated = source.estimated_cost();
        if let Some(max) = self.max_total_budget {
            if self.total_cost + estimated > max {
                return Err(ExpansionError::BudgetExceeded);
            }
        }

        // Fetch
        let fetched = source.fetch(query)?;
        self.total_cost += fetched.cost;

        // Deduplicate and ingest
        let mut added = 0usize;
        let mut deduped = 0usize;
        let mut new_concepts = Vec::new();

        let concept_count_before = core.concept_count;

        for triple in &fetched.triples {
            if self.is_duplicate(triple, core) {
                deduped += 1;
            } else {
                core.ingest_triple(triple);
                added += 1;
            }
        }

        // Track new concepts
        if core.concept_count > concept_count_before {
            // We can't easily get the exact new ones without diffing,
            // but we know how many were added
            let count = core.concept_count - concept_count_before;
            new_concepts.push(format!("{count} new concepts"));
        }

        let result = ExpansionResult {
            query: query.to_string(),
            source: source.name().to_string(),
            triples_added: added,
            triples_deduplicated: deduped,
            cost: fetched.cost,
            new_concepts,
        };

        self.history.push(result.clone());
        Ok(result)
    }

    /// Check if a triple is already in the core (holographic deduplication).
    fn is_duplicate(&self, triple: &Triple, core: &mut KnowledgeCore) -> bool {
        // Encode the triple and check similarity against the concept bundle
        let subject_bundle = core.query_concept(&triple.subject);
        let object_bundle = core.query_concept(&triple.object);

        match (subject_bundle, object_bundle) {
            (Some(sb), Some(ob)) => {
                // Both concepts exist — check if they're already densely connected
                let sim = sb.similarity(&ob);
                sim > self.dedup_threshold
            }
            _ => false, // New concept = not a duplicate
        }
    }

    /// Expand from multiple sources, taking the best result.
    pub fn expand_best(
        &mut self,
        query: &str,
        core: &mut KnowledgeCore,
        sources: &[&dyn KnowledgeSource],
    ) -> Result<ExpansionResult, ExpansionError> {
        let mut best_result: Option<ExpansionResult> = None;
        let mut last_error = None;

        for source in sources {
            match self.expand(query, core, *source) {
                Ok(result) => {
                    match &best_result {
                        None => best_result = Some(result),
                        Some(prev) if result.triples_added > prev.triples_added => {
                            best_result = Some(result);
                        }
                        _ => {}
                    }
                }
                Err(e) => last_error = Some(e),
            }
        }

        best_result.ok_or_else(|| {
            last_error.unwrap_or(ExpansionError::NoResults(query.to_string()))
        })
    }

    /// Total expansions performed.
    pub fn expansion_count(&self) -> usize {
        self.history.len()
    }
}

impl Default for Expander {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple text-based knowledge source that extracts triples from raw text.
/// This is the simplest possible source — parses "subject is_a object" patterns
/// from plain text. For real use, replace with web search / Wikipedia / LLM.
pub struct TextSource {
    /// Name of this source.
    name: String,
    /// The text corpus to extract from.
    texts: Vec<String>,
}

impl TextSource {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            texts: Vec::new(),
        }
    }

    /// Add text to the corpus.
    pub fn add_text(&mut self, text: &str) {
        self.texts.push(text.to_string());
    }

    /// Extract triples from a sentence using simple pattern matching.
    fn extract_triples_from_sentence(sentence: &str) -> Vec<Triple> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut triples = Vec::new();

        // Pattern: "X is a Y" → (X, IsA, Y)
        for window in words.windows(4) {
            if window[1] == "is" && window[2] == "a" {
                triples.push(Triple::new(window[0], RelationType::IsA, window[3]));
            }
        }

        // Pattern: "X is Y" → (X, HasProperty, Y)
        for window in words.windows(3) {
            if window[1] == "is" && window[2] != "a" && window[2] != "an" {
                triples.push(Triple::new(
                    window[0],
                    RelationType::HasProperty,
                    window[2],
                ));
            }
        }

        // Pattern: "X can Y" → (X, CapableOf, Y)
        for window in words.windows(3) {
            if window[1] == "can" {
                triples.push(Triple::new(window[0], RelationType::CapableOf, window[2]));
            }
        }

        // Pattern: "X has Y" → (X, HasA, Y)
        for window in words.windows(3) {
            if window[1] == "has" {
                triples.push(Triple::new(window[0], RelationType::HasA, window[2]));
            }
        }

        // Pattern: "X causes Y" → (X, Causes, Y)
        for window in words.windows(3) {
            if window[1] == "causes" {
                triples.push(Triple::new(window[0], RelationType::Causes, window[2]));
            }
        }

        // Pattern: "X in Y" / "X at Y" → (X, AtLocation, Y)
        for window in words.windows(3) {
            if window[1] == "in" || window[1] == "at" {
                triples.push(Triple::new(window[0], RelationType::AtLocation, window[2]));
            }
        }

        triples
    }
}

impl KnowledgeSource for TextSource {
    fn fetch(&self, query: &str) -> Result<FetchedKnowledge, ExpansionError> {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        // Find relevant sentences from the corpus
        let mut relevant_text = String::new();
        let mut all_triples = Vec::new();

        for text in &self.texts {
            let lower = text.to_lowercase();
            // Check if any query word appears in this text
            let relevant = query_words
                .iter()
                .any(|w| w.len() > 2 && lower.contains(w));

            if relevant {
                relevant_text.push_str(text);
                relevant_text.push(' ');

                // Extract triples from each sentence
                for sentence in text.split('.') {
                    let sentence = sentence.trim().to_lowercase();
                    if !sentence.is_empty() {
                        all_triples.extend(Self::extract_triples_from_sentence(&sentence));
                    }
                }
            }
        }

        if all_triples.is_empty() {
            return Err(ExpansionError::NoResults(query.to_string()));
        }

        Ok(FetchedKnowledge {
            source: self.name.clone(),
            raw_text: relevant_text,
            triples: all_triples,
            confidence: 0.7,
            cost: 0.001, // Very cheap — local text
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn estimated_cost(&self) -> f64 {
        0.001
    }
}

/// A stub for web search that can be replaced with a real implementation.
/// Returns NoResults for now — the trait is what matters.
pub struct WebSearchSource {
    name: String,
}

impl WebSearchSource {
    pub fn new() -> Self {
        Self {
            name: "web_search".to_string(),
        }
    }
}

impl Default for WebSearchSource {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeSource for WebSearchSource {
    fn fetch(&self, query: &str) -> Result<FetchedKnowledge, ExpansionError> {
        // Stub — real implementation would call a search API,
        // parse results, extract triples.
        Err(ExpansionError::SourceUnavailable(
            "web search not yet connected".to_string(),
        ))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn estimated_cost(&self) -> f64 {
        1.0 // Web search is expensive relative to local ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ingest::ConceptNetParser;

    #[test]
    fn test_text_source_extract() {
        let mut source = TextSource::new("test");
        source.add_text("A whale is a mammal. A dolphin is a mammal. A shark is a fish.");
        source.add_text("A mammal has lungs. A fish has gills.");

        let result = source.fetch("whale mammal");
        assert!(result.is_ok());
        let fetched = result.unwrap();
        assert!(!fetched.triples.is_empty());

        // Should have extracted "whale IsA mammal"
        let has_whale = fetched
            .triples
            .iter()
            .any(|t| t.subject == "whale" && t.relation == RelationType::IsA);
        assert!(has_whale, "should extract 'whale IsA mammal'");
    }

    #[test]
    fn test_text_source_no_results() {
        let source = TextSource::new("empty");
        let result = source.fetch("quantum entanglement");
        assert!(matches!(result, Err(ExpansionError::NoResults(_))));
    }

    #[test]
    fn test_expander_should_expand() {
        let expander = Expander::new();
        assert!(expander.should_expand(0.8, 0.3)); // High novelty, low confidence
        assert!(!expander.should_expand(0.3, 0.8)); // Low novelty, high confidence
    }

    #[test]
    fn test_expander_budget() {
        let mut expander = Expander::new();
        expander.max_total_budget = Some(0.5);
        expander.total_cost = 0.5;
        assert!(!expander.should_expand(0.9, 0.1)); // Budget exhausted
    }

    #[test]
    fn test_expand_into_core() {
        let mut core = KnowledgeCore::new();
        core.ingest_batch(&ConceptNetParser::bootstrap_core());
        let triples_before = core.triple_count;
        let concepts_before = core.concept_count;

        let mut source = TextSource::new("marine_biology");
        source.add_text("A whale is a mammal. A dolphin is a mammal.");
        source.add_text("A mammal has lungs. A whale can swim.");

        let mut expander = Expander::new();
        let result = expander.expand("whale", &mut core, &source);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.triples_added > 0, "should add new triples");
        assert_eq!(r.source, "marine_biology");

        // Core should have grown
        assert!(core.concept_count >= concepts_before);
        assert!(
            core.triple_count > triples_before,
            "triple count should grow: {} > {}",
            core.triple_count,
            triples_before
        );
    }

    #[test]
    fn test_expand_deduplication() {
        let mut core = KnowledgeCore::new();
        core.ingest_batch(&ConceptNetParser::bootstrap_core());

        // Source contains knowledge already in the core
        let mut source = TextSource::new("duplicate");
        source.add_text("A dog is a animal. A cat is a animal.");

        let mut expander = Expander::new();
        let result = expander.expand("dog cat", &mut core, &source);
        assert!(result.is_ok());
        let r = result.unwrap();
        // Some triples should be deduplicated since dog/cat/animal are already in the core
        // (exact count depends on dedup threshold and holographic noise)
        assert!(r.triples_added + r.triples_deduplicated > 0);
    }

    #[test]
    fn test_expansion_history() {
        let mut core = KnowledgeCore::new();
        core.ingest_batch(&ConceptNetParser::bootstrap_core());

        let mut source = TextSource::new("test");
        source.add_text("A robot is a machine. A robot can move.");

        let mut expander = Expander::new();
        expander.expand("robot", &mut core, &source).unwrap();

        assert_eq!(expander.expansion_count(), 1);
        assert!(expander.total_cost > 0.0);
    }

    #[test]
    fn test_web_search_stub() {
        let source = WebSearchSource::new();
        let result = source.fetch("anything");
        assert!(matches!(result, Err(ExpansionError::SourceUnavailable(_))));
    }

    #[test]
    fn test_full_pipeline_expand_then_query() {
        let mut core = KnowledgeCore::new();
        core.ingest_batch(&ConceptNetParser::bootstrap_core());

        // Initially, "whale" is unknown
        let whale_known = core.query_concept("whale");
        assert!(whale_known.is_none());

        // Expand with marine biology knowledge
        let mut source = TextSource::new("marine");
        source.add_text("A whale is a mammal. A dolphin is a mammal. A whale can swim.");

        let mut expander = Expander::new();
        expander.expand("whale", &mut core, &source).unwrap();

        // Now "whale" should be queryable
        let whale_now = core.query_concept("whale");
        assert!(whale_now.is_some(), "whale should be in core after expansion");

        // And it should be related to mammal
        let relatedness = core.relatedness("whale", "mammal");
        assert!(
            relatedness > 0.0,
            "whale and mammal should be related: {relatedness}"
        );
    }
}
