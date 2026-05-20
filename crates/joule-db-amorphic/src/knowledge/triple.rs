//! Triple encoding: (subject, relation, object) → holographic binding.
//!
//! A knowledge triple like ("dog", IsA, "animal") becomes:
//!   `triple_hv = dog_hv ⊗ IsA_hv ⊗ Permute(animal_hv)`
//!
//! The permutation on the object breaks symmetry: "dog IsA animal" ≠ "animal IsA dog".
//! Unbinding recovers any component given the other two:
//!   `animal_hv ≈ Unpermute(triple_hv ⊗ dog_hv ⊗ IsA_hv)`

use crate::BinaryHV;
use super::concept::{ConceptEncoder, EncodedConcept};
use super::relation::{RelationCodebook, RelationType};

/// A raw knowledge triple before encoding.
#[derive(Clone, Debug)]
pub struct Triple {
    pub subject: String,
    pub relation: RelationType,
    pub object: String,
    /// ConceptNet-style weight (higher = more confident). Default 1.0.
    pub weight: f64,
}

impl Triple {
    pub fn new(subject: &str, relation: RelationType, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            relation,
            object: object.to_string(),
            weight: 1.0,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// A triple that has been encoded into the holographic algebra.
#[derive(Clone, Debug)]
pub struct EncodedTriple {
    /// The holographic encoding: subject ⊗ relation ⊗ Permute(object)
    pub vector: BinaryHV,
    /// Original triple data
    pub subject: String,
    pub relation: RelationType,
    pub object: String,
    pub weight: f64,
}

impl EncodedTriple {
    /// Encode a triple using the concept encoder and relation codebook.
    pub fn encode(
        triple: &Triple,
        encoder: &mut ConceptEncoder,
        codebook: &RelationCodebook,
    ) -> Self {
        let subject_hv = encoder.encode(&triple.subject);
        let relation_hv = codebook.get(&triple.relation);
        let object_hv = encoder.encode(&triple.object);

        // Bind: subject ⊗ relation ⊗ Permute(object)
        // Permute(object) breaks symmetry — order matters
        let vector = subject_hv
            .vector
            .bind(relation_hv)
            .bind(&object_hv.vector.permute(1));

        Self {
            vector,
            subject: triple.subject.clone(),
            relation: triple.relation,
            object: triple.object.clone(),
            weight: triple.weight,
        }
    }

    /// Recover the object given subject + relation.
    /// Unbinds subject and relation from the triple vector, then unpermutes.
    pub fn recover_object(
        triple_hv: &BinaryHV,
        subject_hv: &BinaryHV,
        relation_hv: &BinaryHV,
    ) -> BinaryHV {
        // object_hv ≈ Unpermute(triple ⊗ subject ⊗ relation)
        // Since XOR is self-inverse: unbind = bind
        triple_hv.bind(subject_hv).bind(relation_hv).permute(
            // Unpermute by (dim - 1) positions = permute by (dim - 1)
            // For BinaryHV permute is circular, so unpermute(1) = permute(dim - 1)
            // But since it's bit-level, we need: total_bits - 1
            subject_hv.dimension() - 1,
        )
    }

    /// Recover the subject given relation + object.
    pub fn recover_subject(
        triple_hv: &BinaryHV,
        relation_hv: &BinaryHV,
        object_hv: &BinaryHV,
    ) -> BinaryHV {
        // subject_hv ≈ triple ⊗ relation ⊗ Permute(object)
        triple_hv.bind(relation_hv).bind(&object_hv.permute(1))
    }

    /// Recover the relation given subject + object.
    pub fn recover_relation(
        triple_hv: &BinaryHV,
        subject_hv: &BinaryHV,
        object_hv: &BinaryHV,
    ) -> BinaryHV {
        // relation_hv ≈ triple ⊗ subject ⊗ Permute(object)
        triple_hv.bind(subject_hv).bind(&object_hv.permute(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_triple() {
        let mut enc = ConceptEncoder::with_default_dim();
        let cb = RelationCodebook::new(10_000);
        let triple = Triple::new("dog", RelationType::IsA, "animal");
        let encoded = EncodedTriple::encode(&triple, &mut enc, &cb);
        assert_eq!(encoded.vector.dimension(), 10_000);
    }

    #[test]
    fn test_different_triples_different_vectors() {
        let mut enc = ConceptEncoder::with_default_dim();
        let cb = RelationCodebook::new(10_000);

        let t1 = Triple::new("dog", RelationType::IsA, "animal");
        let t2 = Triple::new("cat", RelationType::IsA, "animal");

        let e1 = EncodedTriple::encode(&t1, &mut enc, &cb);
        let e2 = EncodedTriple::encode(&t2, &mut enc, &cb);

        let sim = e1.vector.similarity(&e2.vector);
        assert!(sim < 0.6, "different triples should produce different vectors: {sim}");
    }

    #[test]
    fn test_asymmetry() {
        let mut enc = ConceptEncoder::with_default_dim();
        let cb = RelationCodebook::new(10_000);

        // "dog IsA animal" should differ from "animal IsA dog"
        let t1 = Triple::new("dog", RelationType::IsA, "animal");
        let t2 = Triple::new("animal", RelationType::IsA, "dog");

        let e1 = EncodedTriple::encode(&t1, &mut enc, &cb);
        let e2 = EncodedTriple::encode(&t2, &mut enc, &cb);

        let sim = e1.vector.similarity(&e2.vector);
        assert!(sim < 0.6, "reversed triples should be different: {sim}");
    }

    #[test]
    fn test_recover_object() {
        let mut enc = ConceptEncoder::with_default_dim();
        let cb = RelationCodebook::new(10_000);

        let triple = Triple::new("dog", RelationType::IsA, "animal");
        let encoded = EncodedTriple::encode(&triple, &mut enc, &cb);

        let subject_hv = &enc.encode("dog").vector;
        let relation_hv = cb.get(&RelationType::IsA);
        let actual_object = &enc.encode("animal").vector;

        let recovered = EncodedTriple::recover_object(&encoded.vector, subject_hv, relation_hv);

        // Recovered should be similar to actual object
        let sim = recovered.similarity(actual_object);
        assert!(
            sim > 0.5,
            "recovered object should resemble actual: {sim}"
        );
    }

    #[test]
    fn test_recover_subject() {
        let mut enc = ConceptEncoder::with_default_dim();
        let cb = RelationCodebook::new(10_000);

        let triple = Triple::new("dog", RelationType::IsA, "animal");
        let encoded = EncodedTriple::encode(&triple, &mut enc, &cb);

        let relation_hv = cb.get(&RelationType::IsA);
        let object_hv = &enc.encode("animal").vector;
        let actual_subject = &enc.encode("dog").vector;

        let recovered = EncodedTriple::recover_subject(&encoded.vector, relation_hv, object_hv);

        let sim = recovered.similarity(actual_subject);
        assert!(
            sim > 0.5,
            "recovered subject should resemble actual: {sim}"
        );
    }
}
