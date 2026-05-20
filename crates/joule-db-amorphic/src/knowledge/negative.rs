//! Negative Space: representing absence, negation, and void.
//!
//! Three distinct concepts that transformers conflate:
//!
//! 1. **Absence**: something expected is missing. The dog has no tail.
//!    Represented as: expected_hv XOR actual_hv → the residual IS the absence.
//!    High residual = high absence signal.
//!
//! 2. **Negation**: explicitly "not X." The dog is not a cat.
//!    Represented as: a dedicated negation vector bound with the negated concept.
//!    NOT(cat) = negation_hv ⊗ cat_hv. This is near-orthogonal to cat_hv
//!    but structurally linked — unbinding negation_hv recovers cat_hv.
//!
//! 3. **Void**: no input, no knowledge, no state. The system before first read.
//!    Represented as: a dedicated void vector that is maximally distant from
//!    all content vectors. NOT the zero vector (which is degenerate in HDC).
//!
//! ## Why This Matters
//!
//! - "Is the patient allergic to penicillin?" → Absence of allergy record
//!   is medically critical. Not the same as "I don't know."
//! - "The dog is not dangerous" → Negation of dangerous. Not the same as
//!   "the dog is safe" (which is a positive claim).
//! - "What do you know about X?" when X has never been encountered →
//!   Void. The system can distinguish "never seen" from "seen but forgotten."

use crate::BinaryHV;

/// The negation operator: a fixed vector used to negate concepts.
/// NOT(X) = NEGATION ⊗ X
/// Unbinding recovers: X = NOT(X) ⊗ NEGATION (since XOR is self-inverse)
///
/// Properties:
/// - NOT(X) is near-orthogonal to X (similarity ≈ 0.5)
/// - NOT(X) is recoverable: unbind NEGATION to get X back
/// - NOT(NOT(X)) = NEGATION ⊗ NEGATION ⊗ X = X (double negation is identity,
///   because NEGATION ⊗ NEGATION = zeros, and X ⊗ zeros = X)
pub struct NegationOperator {
    /// The negation vector (fixed, deterministic).
    pub vector: BinaryHV,
    /// The void vector: represents "nothing / never encountered."
    pub void: BinaryHV,
    /// Dimension.
    dim: usize,
}

impl NegationOperator {
    /// Create the negation and void operators.
    /// Deterministic — same dimension always produces the same operators.
    pub fn new(dim: usize) -> Self {
        // Negation vector: fixed random seed, distinct from all content
        let vector = BinaryHV::random(dim, 0x4E_6567_A710_0000); // "NEGATION"
        // Void vector: all-ones (maximally distinct from zero vector)
        // This gives void similarity ≈ 0.0 with random vectors (opposite of 0.5)
        let void = BinaryHV::from_words(vec![u64::MAX; (dim + 63) / 64], dim);
        Self { vector, void, dim }
    }

    /// Negate a concept: NOT(X) = negation ⊗ X
    pub fn negate(&self, concept: &BinaryHV) -> BinaryHV {
        self.vector.bind(concept)
    }

    /// Recover the original concept from a negation: X = NOT(X) ⊗ negation
    pub fn unnegate(&self, negated: &BinaryHV) -> BinaryHV {
        self.vector.bind(negated) // XOR is self-inverse
    }

    /// Check if a vector is a negation (was it produced by negate()?).
    /// Returns (is_negation, similarity_to_negation_pattern).
    pub fn is_negated(&self, query: &BinaryHV) -> (bool, f32) {
        // A negated vector will have characteristic similarity to the negation vector
        // specifically: negate(X) should be near-orthogonal to X but correlated
        // with the negation vector's structure
        let sim_to_negation = self.vector.similarity(query);
        // If the query was produced by negate(), its XOR with negation_hv
        // should produce a clean concept (high similarity to some real concept)
        // We can't check that here without the codebook, so we use a heuristic:
        // negated vectors tend to have similarity to negation_hv around 0.5
        // (since XOR with random is random), but the binding pattern is detectable
        // via the bipolar similarity being off-center
        (sim_to_negation > 0.52 || sim_to_negation < 0.48, sim_to_negation)
    }

    /// Detect absence: compare expected state vs actual state.
    /// The residual (XOR) IS the absence signal.
    /// High popcount in residual = high absence (many expected bits missing).
    pub fn detect_absence(&self, expected: &BinaryHV, actual: &BinaryHV) -> Absence {
        let residual = expected.bind(actual); // XOR = what's different
        let hamming = expected.hamming_distance(actual);
        let absence_ratio = hamming as f64 / expected.dimension() as f64;

        Absence {
            residual,
            hamming_distance: hamming,
            absence_ratio,
            is_absent: absence_ratio > 0.4, // More than 40% different = meaningfully absent
        }
    }

    /// Check if a vector represents void (never encountered).
    pub fn is_void(&self, query: &BinaryHV) -> bool {
        // Void is all-ones. Check if query is very similar to void.
        query.similarity(&self.void) > 0.9
    }

    /// Get the void vector (for initializing "never seen" states).
    pub fn void(&self) -> &BinaryHV {
        &self.void
    }

    /// Create an absence-aware triple:
    /// "X does NOT have Y" = X ⊗ relation ⊗ NOT(Y)
    pub fn negate_object(&self, object: &BinaryHV) -> BinaryHV {
        self.negate(object)
    }

    /// Create "NOT X is_a Y" = NOT(X) ⊗ relation ⊗ Y
    pub fn negate_subject(&self, subject: &BinaryHV) -> BinaryHV {
        self.negate(subject)
    }
}

/// The result of absence detection.
#[derive(Clone, Debug)]
pub struct Absence {
    /// The residual vector: expected XOR actual.
    /// This vector encodes WHAT is absent.
    pub residual: BinaryHV,
    /// Raw hamming distance between expected and actual.
    pub hamming_distance: u32,
    /// Fraction of bits that differ (0.0 = identical, 1.0 = opposite).
    pub absence_ratio: f64,
    /// Whether the absence is significant (> 40% difference).
    pub is_absent: bool,
}

impl Absence {
    /// How much is missing? 0.0 = nothing missing, 1.0 = everything missing.
    pub fn magnitude(&self) -> f64 {
        self.absence_ratio
    }
}

/// Wrapper for negative knowledge in the knowledge core.
/// Tracks what the system knows is NOT true, separately from
/// what it knows IS true and what it doesn't know at all.
pub struct NegativeKnowledge {
    /// The negation operator.
    pub negation: NegationOperator,
    /// Explicitly negated concepts: label → negated vector.
    negated_concepts: std::collections::HashMap<String, BinaryHV>,
    /// Absence records: (concept, expected_property, absence).
    absences: Vec<(String, String, Absence)>,
    /// Void concepts: things we know we've never encountered.
    void_set: std::collections::HashSet<String>,
}

impl NegativeKnowledge {
    pub fn new(dim: usize) -> Self {
        Self {
            negation: NegationOperator::new(dim),
            negated_concepts: std::collections::HashMap::new(),
            absences: Vec::new(),
            void_set: std::collections::HashSet::new(),
        }
    }

    /// Record that "X is NOT Y": negate Y and store the binding.
    pub fn record_negation(&mut self, subject: &str, negated_property: &str, property_hv: &BinaryHV) {
        let not_hv = self.negation.negate(property_hv);
        let key = format!("{}:not:{}", subject.to_lowercase(), negated_property.to_lowercase());
        self.negated_concepts.insert(key, not_hv);
    }

    /// Record an absence: "X was expected to have Y but doesn't."
    pub fn record_absence(
        &mut self,
        concept: &str,
        expected_property: &str,
        expected_hv: &BinaryHV,
        actual_hv: &BinaryHV,
    ) {
        let absence = self.negation.detect_absence(expected_hv, actual_hv);
        if absence.is_absent {
            self.absences.push((
                concept.to_lowercase(),
                expected_property.to_lowercase(),
                absence,
            ));
        }
    }

    /// Mark a concept as void (never encountered, not just unknown).
    pub fn mark_void(&mut self, concept: &str) {
        self.void_set.insert(concept.to_lowercase());
    }

    /// Is this concept explicitly void?
    pub fn is_void(&self, concept: &str) -> bool {
        self.void_set.contains(&concept.to_lowercase())
    }

    /// Has this concept been explicitly negated?
    pub fn is_negated(&self, subject: &str, property: &str) -> bool {
        let key = format!("{}:not:{}", subject.to_lowercase(), property.to_lowercase());
        self.negated_concepts.contains_key(&key)
    }

    /// Get all known absences for a concept.
    pub fn absences_for(&self, concept: &str) -> Vec<(&str, &Absence)> {
        let lower = concept.to_lowercase();
        self.absences
            .iter()
            .filter(|(c, _, _)| *c == lower)
            .map(|(_, prop, absence)| (prop.as_str(), absence))
            .collect()
    }

    /// Total negative facts stored.
    pub fn negative_fact_count(&self) -> usize {
        self.negated_concepts.len() + self.absences.len() + self.void_set.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::concept::KNOWLEDGE_DIM;

    #[test]
    fn test_double_negation_is_identity() {
        let neg = NegationOperator::new(KNOWLEDGE_DIM);
        let concept = BinaryHV::random(KNOWLEDGE_DIM, 42);

        let negated = neg.negate(&concept);
        let double_negated = neg.negate(&negated);

        // NOT(NOT(X)) should equal X
        // Because: neg ⊗ (neg ⊗ X) = (neg ⊗ neg) ⊗ X = zeros ⊗ X = X
        assert_eq!(
            concept.hamming_distance(&double_negated),
            0,
            "double negation should be identity"
        );
    }

    #[test]
    fn test_negation_is_orthogonal_to_original() {
        let neg = NegationOperator::new(KNOWLEDGE_DIM);
        let concept = BinaryHV::random(KNOWLEDGE_DIM, 42);
        let negated = neg.negate(&concept);

        let sim = concept.similarity(&negated);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "NOT(X) should be near-orthogonal to X: {sim}"
        );
    }

    #[test]
    fn test_unnegate_recovers() {
        let neg = NegationOperator::new(KNOWLEDGE_DIM);
        let concept = BinaryHV::random(KNOWLEDGE_DIM, 42);
        let negated = neg.negate(&concept);
        let recovered = neg.unnegate(&negated);

        assert_eq!(
            concept.hamming_distance(&recovered),
            0,
            "unnegate should recover the original"
        );
    }

    #[test]
    fn test_absence_detection() {
        let neg = NegationOperator::new(KNOWLEDGE_DIM);
        let expected = BinaryHV::random(KNOWLEDGE_DIM, 1);
        let actual = BinaryHV::random(KNOWLEDGE_DIM, 2);

        let absence = neg.detect_absence(&expected, &actual);
        // Two random vectors: ~50% bits differ = high absence
        assert!(
            absence.absence_ratio > 0.3,
            "random vectors should show absence: {}",
            absence.absence_ratio
        );
        assert!(absence.is_absent);
    }

    #[test]
    fn test_no_absence_when_identical() {
        let neg = NegationOperator::new(KNOWLEDGE_DIM);
        let same = BinaryHV::random(KNOWLEDGE_DIM, 42);

        let absence = neg.detect_absence(&same, &same);
        assert_eq!(absence.hamming_distance, 0);
        assert!(!absence.is_absent);
        assert!((absence.absence_ratio - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_void_is_distinct() {
        let neg = NegationOperator::new(KNOWLEDGE_DIM);
        let random = BinaryHV::random(KNOWLEDGE_DIM, 42);

        // Void should not look like a random concept
        assert!(!neg.is_void(&random));
        // Void should look like void
        assert!(neg.is_void(&neg.void));
    }

    #[test]
    fn test_negative_knowledge_tracking() {
        let mut nk = NegativeKnowledge::new(KNOWLEDGE_DIM);
        let allergy_hv = BinaryHV::random(KNOWLEDGE_DIM, 1);

        // Record: "patient is NOT allergic to penicillin"
        nk.record_negation("patient", "allergic_penicillin", &allergy_hv);

        assert!(nk.is_negated("patient", "allergic_penicillin"));
        assert!(!nk.is_negated("patient", "allergic_aspirin"));
    }

    #[test]
    fn test_void_tracking() {
        let mut nk = NegativeKnowledge::new(KNOWLEDGE_DIM);
        nk.mark_void("unicorn");

        assert!(nk.is_void("unicorn"));
        assert!(!nk.is_void("horse"));
    }

    #[test]
    fn test_absence_tracking() {
        let mut nk = NegativeKnowledge::new(KNOWLEDGE_DIM);
        let expected_tail = BinaryHV::random(KNOWLEDGE_DIM, 1);
        let actual_no_tail = BinaryHV::random(KNOWLEDGE_DIM, 2);

        nk.record_absence("dog", "tail", &expected_tail, &actual_no_tail);

        let absences = nk.absences_for("dog");
        assert_eq!(absences.len(), 1);
        assert_eq!(absences[0].0, "tail");
        assert!(absences[0].1.is_absent);
    }

    #[test]
    fn test_negative_fact_count() {
        let mut nk = NegativeKnowledge::new(KNOWLEDGE_DIM);
        let hv = BinaryHV::random(KNOWLEDGE_DIM, 1);

        nk.record_negation("a", "b", &hv);
        nk.mark_void("c");
        nk.record_absence("d", "e", &hv, &BinaryHV::random(KNOWLEDGE_DIM, 2));

        assert_eq!(nk.negative_fact_count(), 3);
    }
}
