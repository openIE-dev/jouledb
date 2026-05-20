//! Tier 0: Database exact retrieval + Eigenbasis structural lookup.
//!
//! Before any HDC operation, before any holographic unbinding, before any
//! neural inference — check if the answer is just a database lookup.
//!
//! "What is the capital of France?" → SELECT capital FROM countries WHERE name = 'France'
//! Cost: 0.1µJ. Latency: <1ms. Accuracy: 100%.
//!
//! If not an exact match, use the eigenbasis to find the structural neighborhood:
//! "How is cancer related to war?" → project both onto 34 patterns → cosine similarity
//! Cost: 1µJ. Latency: <1ms. Accuracy: determined by pattern scores.
//!
//! Only escalate to Tier 1 (holographic) if Tier 0 can't answer.
//!
//! ```text
//! Query → Tier 0a: exact DB lookup → found? → answer (0.1µJ)
//!       → Tier 0b: eigenbasis structural → confident? → answer (1µJ)
//!       → Tier 1+: holographic/embedded/local/frontier
//! ```

use std::collections::HashMap;

use super::eigenbasis::{Eigenbasis, PatternScores, NUM_PATTERNS};

/// Query classification: what kind of question is this?
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryType {
    /// Exact factual lookup: "What is X?" where X maps to a known key.
    ExactLookup(String),
    /// Structural comparison: "How are X and Y related?"
    StructuralCompare(String, String),
    /// Structural neighborhood: "What is similar to X?"
    StructuralNeighborhood(String),
    /// Pattern query: "What things exhibit Transformation?"
    PatternQuery(String),
    /// Can't classify — escalate to Tier 1+
    Unclassified(String),
}

/// Result from Tier 0.
#[derive(Clone, Debug)]
pub struct Tier0Result {
    /// The answer (if found).
    pub answer: Option<String>,
    /// Confidence (1.0 for exact lookup, 0.0-1.0 for structural).
    pub confidence: f64,
    /// Which sub-tier answered.
    pub source: Tier0Source,
    /// Energy consumed (joules).
    pub energy: f64,
    /// Whether to escalate to higher tiers.
    pub escalate: bool,
}

/// Which part of Tier 0 produced the answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tier0Source {
    /// Exact database lookup.
    DatabaseExact,
    /// Eigenbasis structural similarity.
    EigenbasisStructural,
    /// No answer — escalate.
    None,
}

/// The Tier 0 engine.
pub struct Tier0 {
    /// Exact fact store: key → value. The simplest possible database.
    facts: HashMap<String, String>,
    /// Concept pattern scores: concept → PatternScores.
    concept_scores: HashMap<String, PatternScores>,
    /// The eigenbasis (1KB brain).
    pub eigenbasis: Option<Eigenbasis>,
    /// Pre-computed positions for scored concepts.
    positions: HashMap<String, Vec<f64>>,
    /// Minimum confidence to answer from Tier 0 (below → escalate).
    pub min_confidence: f64,
}

impl Tier0 {
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
            concept_scores: HashMap::new(),
            eigenbasis: None,
            positions: HashMap::new(),
            min_confidence: 0.6,
        }
    }

    /// Load the eigenbasis from pre-computed data.
    pub fn set_eigenbasis(&mut self, basis: Eigenbasis) {
        self.eigenbasis = Some(basis);
        // Reproject all scored concepts
        self.reproject_all();
    }

    /// Register an exact fact: key → value.
    pub fn register_fact(&mut self, key: &str, value: &str) {
        self.facts.insert(key.to_lowercase(), value.to_string());
    }

    /// Register many facts.
    pub fn register_facts(&mut self, facts: &[(&str, &str)]) {
        for (k, v) in facts {
            self.register_fact(k, v);
        }
    }

    /// Register a concept's pattern scores.
    pub fn register_concept(&mut self, name: &str, scores: PatternScores) {
        let key = name.to_lowercase();
        if let Some(ref basis) = self.eigenbasis {
            let pos = basis.project(&scores);
            self.positions.insert(key.clone(), pos);
        }
        self.concept_scores.insert(key, scores);
    }

    /// Query Tier 0. Returns result + whether to escalate.
    pub fn query(&self, question: &str) -> Tier0Result {
        let qtype = self.classify_query(question);

        match qtype {
            QueryType::ExactLookup(key) => self.exact_lookup(&key),
            QueryType::StructuralCompare(a, b) => self.structural_compare(&a, &b),
            QueryType::StructuralNeighborhood(concept) => self.structural_neighborhood(&concept),
            QueryType::PatternQuery(pattern) => self.pattern_query(&pattern),
            QueryType::Unclassified(_) => Tier0Result {
                answer: None,
                confidence: 0.0,
                source: Tier0Source::None,
                energy: 0.000_000_1, // 0.1µJ for classification attempt
                escalate: true,
            },
        }
    }

    /// Classify a query into QueryType.
    fn classify_query(&self, question: &str) -> QueryType {
        let lower = question.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Pattern: "what is X" / "what's X" / "define X"
        if lower.starts_with("what is ") || lower.starts_with("what's ") || lower.starts_with("define ")
        {
            let subject = words[2..].join(" ").trim_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string();
            // Check if it's in the fact store
            if self.facts.contains_key(&subject) {
                return QueryType::ExactLookup(subject);
            }
            // Check if we have pattern scores for it
            if self.concept_scores.contains_key(&subject) {
                return QueryType::StructuralNeighborhood(subject);
            }
            return QueryType::ExactLookup(subject);
        }

        // Pattern: "how are X and Y related" / "compare X and Y" / "X vs Y"
        if lower.contains(" and ") && (lower.contains("related") || lower.contains("compare") || lower.contains("similar")) {
            let parts: Vec<&str> = lower.split(" and ").collect();
            if parts.len() >= 2 {
                let a = extract_concept(parts[0]);
                let b = extract_concept(parts[1]);
                return QueryType::StructuralCompare(a, b);
            }
        }
        if lower.contains(" vs ") {
            let parts: Vec<&str> = lower.split(" vs ").collect();
            if parts.len() >= 2 {
                let a = extract_concept(parts[0]);
                let b = extract_concept(parts[1]);
                return QueryType::StructuralCompare(a, b);
            }
        }

        // Pattern: "what exhibits X" / "things with X pattern"
        for pattern in super::eigenbasis::PATTERN_NAMES.iter() {
            if lower.contains(pattern) {
                return QueryType::PatternQuery(pattern.to_string());
            }
        }

        // Can't classify
        QueryType::Unclassified(question.to_string())
    }

    fn exact_lookup(&self, key: &str) -> Tier0Result {
        if let Some(value) = self.facts.get(key) {
            Tier0Result {
                answer: Some(value.clone()),
                confidence: 1.0,
                source: Tier0Source::DatabaseExact,
                energy: 0.000_000_1, // 0.1µJ
                escalate: false,
            }
        } else {
            Tier0Result {
                answer: None,
                confidence: 0.0,
                source: Tier0Source::None,
                energy: 0.000_000_1,
                escalate: true,
            }
        }
    }

    fn structural_compare(&self, a: &str, b: &str) -> Tier0Result {
        let basis = match &self.eigenbasis {
            Some(b) => b,
            None => {
                return Tier0Result {
                    answer: None,
                    confidence: 0.0,
                    source: Tier0Source::None,
                    energy: 0.000_000_1,
                    escalate: true,
                }
            }
        };

        let scores_a = self.concept_scores.get(a);
        let scores_b = self.concept_scores.get(b);

        match (scores_a, scores_b) {
            (Some(sa), Some(sb)) => {
                let similarity = basis.relate(sa, sb);
                let confidence = similarity.abs(); // High similarity = high confidence

                // Find shared patterns (what makes them similar)
                let shared: Vec<String> = (0..NUM_PATTERNS)
                    .filter(|&i| sa.scores[i] > 0.3 && sb.scores[i] > 0.3)
                    .map(|i| super::eigenbasis::PATTERN_NAMES[i].to_string())
                    .collect();

                let answer = format!(
                    "{} and {} are {:.0}% structurally similar{}",
                    a,
                    b,
                    similarity * 100.0,
                    if shared.is_empty() {
                        String::new()
                    } else {
                        format!(". Shared patterns: {}", shared.join(", "))
                    }
                );

                Tier0Result {
                    answer: Some(answer),
                    confidence,
                    source: Tier0Source::EigenbasisStructural,
                    energy: 0.000_001, // 1µJ
                    escalate: confidence < self.min_confidence,
                }
            }
            _ => Tier0Result {
                answer: None,
                confidence: 0.0,
                source: Tier0Source::None,
                energy: 0.000_000_5,
                escalate: true,
            },
        }
    }

    fn structural_neighborhood(&self, concept: &str) -> Tier0Result {
        let basis = match &self.eigenbasis {
            Some(b) => b,
            None => {
                return Tier0Result {
                    answer: None,
                    confidence: 0.0,
                    source: Tier0Source::None,
                    energy: 0.000_000_1,
                    escalate: true,
                }
            }
        };

        let scores = match self.concept_scores.get(concept) {
            Some(s) => s,
            None => {
                return Tier0Result {
                    answer: None,
                    confidence: 0.0,
                    source: Tier0Source::None,
                    energy: 0.000_000_1,
                    escalate: true,
                }
            }
        };

        let pos = basis.project(scores);

        // Find nearest scored concepts
        let mut neighbors: Vec<(String, f64)> = self
            .positions
            .iter()
            .filter(|(name, _)| *name != concept)
            .map(|(name, other_pos)| {
                let sim = cosine_sim(&pos, other_pos);
                (name.clone(), sim)
            })
            .collect();

        neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(5);

        // Describe using top pattern scores
        let top_patterns: Vec<String> = (0..NUM_PATTERNS)
            .filter(|&i| scores.scores[i] > 0.4)
            .map(|i| {
                format!(
                    "{} ({:.0}%)",
                    super::eigenbasis::PATTERN_NAMES[i],
                    scores.scores[i] * 100.0
                )
            })
            .collect();

        let neighbor_str: String = neighbors
            .iter()
            .take(3)
            .map(|(n, s)| format!("{} ({:.0}%)", n, s * 100.0))
            .collect::<Vec<_>>()
            .join(", ");

        let answer = format!(
            "{} exhibits: {}. Structurally similar to: {}",
            concept,
            if top_patterns.is_empty() {
                "no strong patterns measured".to_string()
            } else {
                top_patterns.join(", ")
            },
            if neighbor_str.is_empty() {
                "no scored neighbors".to_string()
            } else {
                neighbor_str
            }
        );

        let confidence = if top_patterns.is_empty() { 0.3 } else { 0.8 };

        Tier0Result {
            answer: Some(answer),
            confidence,
            source: Tier0Source::EigenbasisStructural,
            energy: 0.000_001,
            escalate: confidence < self.min_confidence,
        }
    }

    fn pattern_query(&self, pattern: &str) -> Tier0Result {
        let idx = match super::eigenbasis::pattern_index(pattern) {
            Some(i) => i,
            None => {
                return Tier0Result {
                    answer: None,
                    confidence: 0.0,
                    source: Tier0Source::None,
                    energy: 0.000_000_1,
                    escalate: true,
                }
            }
        };

        // Find concepts that score high on this pattern
        let mut exemplars: Vec<(String, f64)> = self
            .concept_scores
            .iter()
            .filter(|(_, scores)| scores.scores[idx] > 0.3)
            .map(|(name, scores)| (name.clone(), scores.scores[idx]))
            .collect();

        exemplars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        exemplars.truncate(5);

        if exemplars.is_empty() {
            return Tier0Result {
                answer: None,
                confidence: 0.0,
                source: Tier0Source::None,
                energy: 0.000_000_5,
                escalate: true,
            };
        }

        let answer = format!(
            "Concepts exhibiting {}: {}",
            pattern,
            exemplars
                .iter()
                .map(|(n, s)| format!("{} ({:.0}%)", n, s * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        );

        Tier0Result {
            answer: Some(answer),
            confidence: 0.85,
            source: Tier0Source::EigenbasisStructural,
            energy: 0.000_001,
            escalate: false,
        }
    }

    /// Reproject all scored concepts through the eigenbasis.
    fn reproject_all(&mut self) {
        if let Some(ref basis) = self.eigenbasis {
            self.positions.clear();
            for (name, scores) in &self.concept_scores {
                let pos = basis.project(scores);
                self.positions.insert(name.clone(), pos);
            }
        }
    }

    /// Number of registered facts.
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Number of scored concepts.
    pub fn concept_count(&self) -> usize {
        self.concept_scores.len()
    }
}

impl Default for Tier0 {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract a concept name from a phrase (strip stop words, punctuation).
fn extract_concept(phrase: &str) -> String {
    let stop = [
        "how", "are", "is", "the", "a", "an", "what", "compare", "related",
        "similar", "between", "to", "and", "or",
    ];
    phrase
        .split_whitespace()
        .filter(|w| !stop.contains(w))
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'))
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na < 1e-15 || nb < 1e-15 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::eigenbasis::{Eigenbasis, PatternScores};

    fn build_tier0() -> Tier0 {
        let mut t0 = Tier0::new();

        // Register some facts
        t0.register_facts(&[
            ("capital of france", "Paris"),
            ("capital of japan", "Tokyo"),
            ("speed of light", "299,792,458 m/s"),
            ("boiling point of water", "100°C at 1 atm"),
        ]);

        // Register concepts with pattern scores
        let cancer = PatternScores::from_sparse(&[
            ("replication", 0.9),
            ("feedback", 0.8),
            ("emergence", 0.7),
            ("selection", 0.6),
            ("transformation", 0.5),
        ]);
        let war = PatternScores::from_sparse(&[
            ("replication", 0.7),
            ("feedback", 0.7),
            ("emergence", 0.8),
            ("selection", 0.5),
            ("hierarchy", 0.6),
        ]);
        let brain = PatternScores::from_sparse(&[
            ("network", 0.9),
            ("hierarchy", 0.7),
            ("signal", 0.8),
            ("feedback", 0.7),
            ("emergence", 0.8),
        ]);
        let internet = PatternScores::from_sparse(&[
            ("network", 0.9),
            ("hierarchy", 0.6),
            ("signal", 0.7),
            ("flow", 0.8),
            ("emergence", 0.6),
        ]);
        let lens = PatternScores::from_sparse(&[
            ("transformation", 0.9),
            ("interface", 0.8),
            ("signal", 0.6),
        ]);

        // Build eigenbasis from these concepts
        let all_scores = vec![
            cancer.clone(), war.clone(), brain.clone(), internet.clone(), lens.clone(),
        ];
        let basis = Eigenbasis::from_scores(&all_scores, 0.80);
        t0.set_eigenbasis(basis);

        t0.register_concept("cancer", cancer);
        t0.register_concept("war", war);
        t0.register_concept("brain", brain);
        t0.register_concept("internet", internet);
        t0.register_concept("lens", lens);

        t0
    }

    #[test]
    fn test_exact_lookup() {
        let t0 = build_tier0();
        let result = t0.query("What is capital of france?");
        assert_eq!(result.answer.as_deref(), Some("Paris"));
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.source, Tier0Source::DatabaseExact);
        assert!(!result.escalate);
        assert!(result.energy < 0.000_001); // Sub-µJ
    }

    #[test]
    fn test_exact_lookup_miss_escalates() {
        let t0 = build_tier0();
        let result = t0.query("What is capital of narnia?");
        assert!(result.answer.is_none());
        assert!(result.escalate);
    }

    #[test]
    fn test_structural_compare() {
        let t0 = build_tier0();
        let result = t0.query("compare cancer and war");
        assert!(
            result.answer.is_some(),
            "structural compare should produce answer, got: {:?}",
            result
        );
        let answer = result.answer.unwrap();
        assert!(answer.contains("structurally similar"));
        assert_eq!(result.source, Tier0Source::EigenbasisStructural);
    }

    #[test]
    fn test_structural_neighborhood() {
        let t0 = build_tier0();
        let result = t0.query("What is cancer?");
        // cancer is in concept_scores, so should get structural neighborhood
        assert!(result.answer.is_some());
        let answer = result.answer.unwrap();
        assert!(answer.contains("cancer"));
        assert!(answer.contains("exhibits"));
    }

    #[test]
    fn test_pattern_query() {
        let t0 = build_tier0();
        let result = t0.query("What things exhibit emergence?");
        assert!(result.answer.is_some());
        let answer = result.answer.unwrap();
        assert!(answer.contains("emergence"));
        // Cancer, war, brain all score on emergence
    }

    #[test]
    fn test_cancer_war_more_similar_than_cancer_lens() {
        let t0 = build_tier0();
        let r1 = t0.query("cancer vs war");
        let r2 = t0.query("cancer vs lens");
        // Both should have answers
        assert!(r1.answer.is_some());
        assert!(r2.answer.is_some());
        // Cancer-war should have higher confidence than cancer-lens
        assert!(
            r1.confidence > r2.confidence,
            "cancer~war ({}) should be more confident than cancer~lens ({})",
            r1.confidence, r2.confidence
        );
    }

    #[test]
    fn test_brain_internet_similarity() {
        let t0 = build_tier0();
        let result = t0.query("brain vs internet");
        assert!(result.answer.is_some(), "brain vs internet should work");
        let answer = result.answer.unwrap();
        assert!(
            answer.contains("network") || answer.contains("similar") || answer.contains("structurally"),
            "brain~internet should mention structure: {}",
            answer
        );
    }

    #[test]
    fn test_energy_cost() {
        let t0 = build_tier0();

        let exact = t0.query("What is speed of light?");
        let structural = t0.query("cancer vs war");

        // Exact should be cheaper than structural
        assert!(exact.energy <= structural.energy);
        // Both should be sub-µJ to µJ range
        assert!(exact.energy < 0.000_01);
        assert!(structural.energy < 0.000_01);
    }

    #[test]
    fn test_unclassified_escalates() {
        let t0 = build_tier0();
        let result = t0.query("flurble garbonzo wibble");
        assert!(result.escalate);
        assert_eq!(result.source, Tier0Source::None);
    }

    #[test]
    fn test_fact_and_concept_counts() {
        let t0 = build_tier0();
        assert_eq!(t0.fact_count(), 4);
        assert_eq!(t0.concept_count(), 5);
    }
}
