//! Structural Encoder: encode concepts by their graph position, not their name.
//!
//! Trigram encoding: "metharbital" ≈ "midazolam" (share characters) — WRONG.
//! Structural encoding: metharbital is defined by its relationships:
//!   trade_name → metharbital, metharbital → barbiturate, metharbital → sedative
//!
//! Each concept starts with a random seed vector (maximally orthogonal).
//! After all triples are ingested, concepts are RE-encoded by bundling
//! their relationship vectors. The structural encoding captures WHERE
//! a concept sits in the graph, not what its name looks like.
//!
//! This is the UCG formula applied to BinaryHV:
//!   R(A, B) = similarity(structural_encoding(A), structural_encoding(B))
//!
//! Concepts that share many relationships (both hypernyms of the same parent,
//! both used for the same purpose) are structurally similar even if their
//! names are completely different.

use crate::BinaryHV;
use joule_db_hdc::BundleAccumulator;
use std::collections::HashMap;

use super::concept::KNOWLEDGE_DIM;
use super::relation::{RelationCodebook, RelationType};

/// A structural encoder: concepts encoded by graph position.
pub struct StructuralEncoder {
    /// Seed vectors: concept → random orthogonal vector (identity).
    seeds: HashMap<String, BinaryHV>,
    /// Structural vectors: concept → bundled relationship encoding.
    /// Built after ingestion by bundling all relationship vectors.
    structural: HashMap<String, BinaryHV>,
    /// Relation codebook.
    codebook: RelationCodebook,
    /// Relationship log: (subject, relation, object) for re-encoding.
    relationships: Vec<(String, RelationType, String)>,
    /// Dimension.
    dim: usize,
    /// Base seed.
    seed: u64,
    /// Whether structural vectors have been computed.
    compiled: bool,
}

impl StructuralEncoder {
    pub fn new(dim: usize) -> Self {
        Self {
            seeds: HashMap::new(),
            structural: HashMap::new(),
            codebook: RelationCodebook::new(dim),
            relationships: Vec::new(),
            dim,
            seed: 0x57_80C7_08A1_E4C0, // "STRUCTURAL"
            compiled: false,
        }
    }

    /// Get or create a seed vector for a concept.
    /// Seed vectors are random and maximally orthogonal.
    pub fn get_seed(&mut self, concept: &str) -> BinaryHV {
        let key = concept.to_lowercase();
        if let Some(hv) = self.seeds.get(&key) {
            return hv.clone();
        }

        // Deterministic seed from concept name
        let concept_seed = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            key.hash(&mut hasher);
            self.seed.wrapping_add(hasher.finish())
        };

        let hv = BinaryHV::random(self.dim, concept_seed);
        self.seeds.insert(key, hv.clone());
        hv
    }

    /// Record a relationship for structural encoding.
    pub fn record(&mut self, subject: &str, relation: RelationType, object: &str) {
        self.relationships.push((
            subject.to_lowercase(),
            relation,
            object.to_lowercase(),
        ));
        self.compiled = false;

        // Ensure seeds exist
        self.get_seed(subject);
        self.get_seed(object);
    }

    /// Record a batch of relationships.
    pub fn record_batch(&mut self, triples: &[(String, RelationType, String)]) {
        for (s, r, o) in triples {
            self.record(s, *r, o);
        }
    }

    /// Compile structural vectors from recorded relationships.
    /// For each concept, bundle all relationship vectors:
    ///   structural(A) = Σ (relation_hv ⊗ neighbor_seed_hv)
    ///
    /// This means: a concept is the sum of its relationships.
    /// Two concepts with identical relationships get identical structural vectors.
    pub fn compile(&mut self) {
        let mut accumulators: HashMap<String, BundleAccumulator> = HashMap::new();

        for (subject, relation, object) in &self.relationships {
            let relation_hv = self.codebook.get(relation).clone();
            let object_seed = self.seeds.get(object).cloned()
                .unwrap_or_else(|| BinaryHV::random(self.dim, 0));
            let subject_seed = self.seeds.get(subject).cloned()
                .unwrap_or_else(|| BinaryHV::random(self.dim, 0));

            // Subject's structural vector gets: relation ⊗ object_seed
            let subject_component = relation_hv.bind(&object_seed);
            accumulators
                .entry(subject.clone())
                .or_insert_with(|| BundleAccumulator::new(self.dim))
                .add(&subject_component);

            // Object's structural vector gets: relation ⊗ subject_seed
            // (bidirectional — the object knows who points to it)
            let object_component = relation_hv.bind(&subject_seed);
            accumulators
                .entry(object.clone())
                .or_insert_with(|| BundleAccumulator::new(self.dim))
                .add(&object_component);
        }

        // Threshold all accumulators to get structural vectors
        self.structural.clear();
        for (concept, acc) in accumulators {
            self.structural.insert(concept, acc.threshold());
        }

        self.compiled = true;
    }

    /// Get the structural encoding for a concept.
    /// Returns the structural vector if compiled, otherwise the seed.
    pub fn encode(&self, concept: &str) -> BinaryHV {
        let key = concept.to_lowercase();
        if self.compiled {
            if let Some(sv) = self.structural.get(&key) {
                return sv.clone();
            }
        }
        self.seeds.get(&key).cloned()
            .unwrap_or_else(|| BinaryHV::zeros(self.dim))
    }

    /// Structural similarity between two concepts.
    pub fn similarity(&self, a: &str, b: &str) -> f32 {
        let va = self.encode(a);
        let vb = self.encode(b);
        va.similarity(&vb)
    }

    /// Find the K concepts most structurally similar to a query vector.
    pub fn nearest(&self, query: &BinaryHV, k: usize) -> Vec<(String, f32)> {
        let source = if self.compiled { &self.structural } else { &self.seeds };
        let mut scored: Vec<(String, f32)> = source
            .iter()
            .map(|(label, hv)| (label.clone(), hv.similarity(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Number of concepts.
    pub fn concept_count(&self) -> usize {
        self.seeds.len()
    }

    /// Number of relationships recorded.
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    /// Is the encoder compiled?
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Get the codebook (for triple encoding).
    pub fn codebook(&self) -> &RelationCodebook {
        &self.codebook
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seeds_are_orthogonal() {
        let mut enc = StructuralEncoder::new(KNOWLEDGE_DIM);
        let a = enc.get_seed("dog");
        let b = enc.get_seed("cat");
        let sim = a.similarity(&b);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "seeds should be near-orthogonal: {sim}"
        );
    }

    #[test]
    fn test_seeds_are_deterministic() {
        let mut enc1 = StructuralEncoder::new(KNOWLEDGE_DIM);
        let mut enc2 = StructuralEncoder::new(KNOWLEDGE_DIM);
        let a = enc1.get_seed("dog");
        let b = enc2.get_seed("dog");
        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn test_structural_similarity_shared_relations() {
        let mut enc = StructuralEncoder::new(KNOWLEDGE_DIM);

        // Dog and cat share many relationships
        enc.record("dog", RelationType::IsA, "animal");
        enc.record("cat", RelationType::IsA, "animal");
        enc.record("dog", RelationType::AtLocation, "house");
        enc.record("cat", RelationType::AtLocation, "house");
        enc.record("dog", RelationType::HasProperty, "furry");
        enc.record("cat", RelationType::HasProperty, "furry");

        // Car has different relationships
        enc.record("car", RelationType::IsA, "vehicle");
        enc.record("car", RelationType::AtLocation, "road");
        enc.record("car", RelationType::HasProperty, "fast");

        enc.compile();

        let dog_cat = enc.similarity("dog", "cat");
        let dog_car = enc.similarity("dog", "car");

        assert!(
            dog_cat > dog_car,
            "dog~cat ({dog_cat}) should be more similar than dog~car ({dog_car}) structurally"
        );
    }

    #[test]
    fn test_compile_updates_vectors() {
        let mut enc = StructuralEncoder::new(KNOWLEDGE_DIM);
        enc.record("a", RelationType::IsA, "b");
        assert!(!enc.is_compiled());

        enc.compile();
        assert!(enc.is_compiled());

        let sv = enc.encode("a");
        assert_eq!(sv.dimension(), KNOWLEDGE_DIM);
    }

    #[test]
    fn test_nearest_structural() {
        let mut enc = StructuralEncoder::new(KNOWLEDGE_DIM);
        enc.record("dog", RelationType::IsA, "animal");
        enc.record("cat", RelationType::IsA, "animal");
        enc.record("bird", RelationType::IsA, "animal");
        enc.record("car", RelationType::IsA, "vehicle");
        enc.compile();

        let dog_sv = enc.encode("dog");
        let nearest = enc.nearest(&dog_sv, 3);

        // Dog, cat, bird all share IsA animal so structural vectors are very similar.
        // The top result should be one of {dog, cat, bird}, not car.
        assert_ne!(nearest[0].0, "car", "car should not be closest to dog");
        let names: Vec<&str> = nearest.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"cat") || names.contains(&"bird"),
            "structural neighbors should share relationships: {:?}",
            names
        );
    }

    #[test]
    fn test_structural_vs_trigram() {
        // Structural encoding: "metharbital" and "midazolam" are NOT similar
        // unless they share graph relationships.
        // Trigram encoding: they ARE similar because they share character patterns.
        let mut enc = StructuralEncoder::new(KNOWLEDGE_DIM);

        // metharbital is a barbiturate
        enc.record("metharbital", RelationType::IsA, "barbiturate");
        enc.record("metharbital", RelationType::HasProperty, "sedative");

        // midazolam is a benzodiazepine (different class)
        enc.record("midazolam", RelationType::IsA, "benzodiazepine");
        enc.record("midazolam", RelationType::HasProperty, "anxiolytic");

        // aspirin is completely different
        enc.record("aspirin", RelationType::IsA, "nsaid");
        enc.record("aspirin", RelationType::HasProperty, "analgesic");

        enc.compile();

        let meth_mida = enc.similarity("metharbital", "midazolam");
        let meth_asp = enc.similarity("metharbital", "aspirin");

        // Structurally, metharbital and midazolam are equally different from each other
        // as metharbital and aspirin — they share NO relationships.
        // (In trigram encoding, meth~mida would be spuriously high.)
        assert!(
            (meth_mida - meth_asp).abs() < 0.15,
            "structural encoding should not give spurious similarity: meth~mida={meth_mida}, meth~asp={meth_asp}"
        );
    }
}
