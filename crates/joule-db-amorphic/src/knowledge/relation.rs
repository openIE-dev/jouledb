//! Relation types and their holographic codebook.
//!
//! Each relation type (IsA, HasA, PartOf, Causes, etc.) gets a fixed
//! random BinaryHV. These are the "verbs" of the knowledge algebra.
//! Binding a subject with a relation and an object creates a triple.
//!
//! The codebook covers ConceptNet's relation types plus extensions
//! for temporal, causal, and structural relationships.

use crate::BinaryHV;
use std::collections::HashMap;

/// The known relation types in the knowledge core.
/// Covers ConceptNet's relations plus structural extensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelationType {
    // ConceptNet core relations
    IsA,
    HasA,
    PartOf,
    UsedFor,
    CapableOf,
    AtLocation,
    Causes,
    HasProperty,
    MotivatedByGoal,
    HasSubevent,
    HasFirstSubevent,
    HasLastSubevent,
    HasPrerequisite,
    CreatedBy,
    Desires,
    MadeOf,
    ReceivesAction,
    DefinedAs,
    SymbolOf,
    InstanceOf,
    DerivedFrom,
    RelatedTo,
    SimilarTo,
    Antonym,
    DistinctFrom,
    FormOf,
    EtymologicallyRelatedTo,
    MannerOf,

    // Temporal
    Before,
    After,
    During,
    CausesDesire,

    // Structural extensions
    ContainedIn,
    ConnectedTo,
    OppositeOf,
    Entails,
    ExternalUrl,

    // Catch-all for unknown relations
    Other,
}

impl RelationType {
    /// Parse a ConceptNet relation string.
    pub fn from_conceptnet(rel: &str) -> Self {
        match rel {
            "/r/IsA" | "IsA" | "is_a" => Self::IsA,
            "/r/HasA" | "HasA" | "has_a" => Self::HasA,
            "/r/PartOf" | "PartOf" | "part_of" => Self::PartOf,
            "/r/UsedFor" | "UsedFor" | "used_for" => Self::UsedFor,
            "/r/CapableOf" | "CapableOf" | "capable_of" => Self::CapableOf,
            "/r/AtLocation" | "AtLocation" | "at_location" => Self::AtLocation,
            "/r/Causes" | "Causes" | "causes" => Self::Causes,
            "/r/HasProperty" | "HasProperty" | "has_property" => Self::HasProperty,
            "/r/MotivatedByGoal" | "MotivatedByGoal" => Self::MotivatedByGoal,
            "/r/HasSubevent" | "HasSubevent" => Self::HasSubevent,
            "/r/HasFirstSubevent" | "HasFirstSubevent" => Self::HasFirstSubevent,
            "/r/HasLastSubevent" | "HasLastSubevent" => Self::HasLastSubevent,
            "/r/HasPrerequisite" | "HasPrerequisite" => Self::HasPrerequisite,
            "/r/CreatedBy" | "CreatedBy" | "created_by" => Self::CreatedBy,
            "/r/Desires" | "Desires" | "desires" => Self::Desires,
            "/r/MadeOf" | "MadeOf" | "made_of" => Self::MadeOf,
            "/r/ReceivesAction" | "ReceivesAction" => Self::ReceivesAction,
            "/r/DefinedAs" | "DefinedAs" | "defined_as" => Self::DefinedAs,
            "/r/SymbolOf" | "SymbolOf" | "symbol_of" => Self::SymbolOf,
            "/r/InstanceOf" | "InstanceOf" | "instance_of" => Self::InstanceOf,
            "/r/DerivedFrom" | "DerivedFrom" | "derived_from" => Self::DerivedFrom,
            "/r/RelatedTo" | "RelatedTo" | "related_to" => Self::RelatedTo,
            "/r/SimilarTo" | "SimilarTo" | "similar_to" => Self::SimilarTo,
            "/r/Antonym" | "Antonym" | "antonym" => Self::Antonym,
            "/r/DistinctFrom" | "DistinctFrom" | "distinct_from" => Self::DistinctFrom,
            "/r/FormOf" | "FormOf" | "form_of" => Self::FormOf,
            "/r/EtymologicallyRelatedTo" | "EtymologicallyRelatedTo" => {
                Self::EtymologicallyRelatedTo
            }
            "/r/MannerOf" | "MannerOf" | "manner_of" => Self::MannerOf,
            "/r/CausesDesire" | "CausesDesire" => Self::CausesDesire,
            "/r/Entails" | "Entails" | "entails" => Self::Entails,
            "/r/ExternalURL" | "ExternalURL" => Self::ExternalUrl,
            _ => Self::Other,
        }
    }

    /// All defined relation types.
    pub fn all() -> &'static [RelationType] {
        &[
            Self::IsA,
            Self::HasA,
            Self::PartOf,
            Self::UsedFor,
            Self::CapableOf,
            Self::AtLocation,
            Self::Causes,
            Self::HasProperty,
            Self::MotivatedByGoal,
            Self::HasSubevent,
            Self::HasFirstSubevent,
            Self::HasLastSubevent,
            Self::HasPrerequisite,
            Self::CreatedBy,
            Self::Desires,
            Self::MadeOf,
            Self::ReceivesAction,
            Self::DefinedAs,
            Self::SymbolOf,
            Self::InstanceOf,
            Self::DerivedFrom,
            Self::RelatedTo,
            Self::SimilarTo,
            Self::Antonym,
            Self::DistinctFrom,
            Self::FormOf,
            Self::EtymologicallyRelatedTo,
            Self::MannerOf,
            Self::Before,
            Self::After,
            Self::During,
            Self::CausesDesire,
            Self::ContainedIn,
            Self::ConnectedTo,
            Self::OppositeOf,
            Self::Entails,
            Self::ExternalUrl,
            Self::Other,
        ]
    }

    /// Semantic weight: how much structural information does this relation carry?
    /// Higher = more informative for knowledge structure.
    pub fn weight(&self) -> f64 {
        match self {
            Self::IsA => 1.0,          // Taxonomy: most structural
            Self::PartOf => 0.95,      // Meronymy: strong structure
            Self::HasA => 0.9,         // Possession: strong
            Self::Causes => 0.9,       // Causality: very structural
            Self::HasPrerequisite => 0.85,
            Self::HasSubevent => 0.8,
            Self::CapableOf => 0.8,
            Self::UsedFor => 0.75,
            Self::AtLocation => 0.7,
            Self::HasProperty => 0.7,
            Self::MadeOf => 0.7,
            Self::InstanceOf => 0.7,
            Self::DefinedAs => 0.65,
            Self::Entails => 0.65,
            Self::CreatedBy => 0.6,
            Self::MotivatedByGoal => 0.6,
            Self::Desires => 0.55,
            Self::MannerOf => 0.5,
            Self::SimilarTo => 0.4,
            Self::RelatedTo => 0.3,    // Weak: "related" is vague
            Self::DerivedFrom => 0.3,
            Self::FormOf => 0.25,
            Self::EtymologicallyRelatedTo => 0.2,
            Self::Antonym => 0.4,      // Contrast: structurally informative
            Self::DistinctFrom => 0.35,
            Self::SymbolOf => 0.3,
            Self::ReceivesAction => 0.5,
            Self::Before | Self::After | Self::During => 0.7,
            Self::CausesDesire => 0.6,
            Self::HasFirstSubevent | Self::HasLastSubevent => 0.75,
            Self::ContainedIn | Self::ConnectedTo => 0.6,
            Self::OppositeOf => 0.4,
            Self::ExternalUrl => 0.1,
            Self::Other => 0.2,
        }
    }
}

/// Codebook of relation type vectors. Each relation gets a fixed random BinaryHV.
pub struct RelationCodebook {
    vectors: HashMap<RelationType, BinaryHV>,
    dim: usize,
}

impl RelationCodebook {
    /// Create the codebook. Deterministic — same seed always produces same codebook.
    pub fn new(dim: usize) -> Self {
        let base_seed: u64 = 0xBE_1A71_04C0_DE80; // "RELATION_CODEBOOK"
        let mut vectors = HashMap::new();

        for (i, rel) in RelationType::all().iter().enumerate() {
            let seed = base_seed.wrapping_add(i as u64 * 7919); // Prime spacing
            vectors.insert(*rel, BinaryHV::random(dim, seed));
        }

        Self { vectors, dim }
    }

    /// Get the vector for a relation type.
    pub fn get(&self, rel: &RelationType) -> &BinaryHV {
        self.vectors
            .get(rel)
            .expect("all relation types should be in codebook")
    }

    /// Identify the closest relation type to a given vector.
    pub fn identify(&self, vector: &BinaryHV) -> (RelationType, f32) {
        self.vectors
            .iter()
            .map(|(rel, v)| (*rel, v.similarity(vector)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_orthogonal() {
        let cb = RelationCodebook::new(10000);
        let is_a = cb.get(&RelationType::IsA);
        let has_a = cb.get(&RelationType::HasA);
        let sim = is_a.similarity(has_a);
        assert!(
            sim < 0.55,
            "relation vectors should be near-orthogonal: {sim}"
        );
    }

    #[test]
    fn test_codebook_deterministic() {
        let cb1 = RelationCodebook::new(10000);
        let cb2 = RelationCodebook::new(10000);
        let v1 = cb1.get(&RelationType::Causes);
        let v2 = cb2.get(&RelationType::Causes);
        assert_eq!(v1.hamming_distance(v2), 0);
    }

    #[test]
    fn test_identify_relation() {
        let cb = RelationCodebook::new(10000);
        let is_a_vec = cb.get(&RelationType::IsA).clone();
        let (identified, sim) = cb.identify(&is_a_vec);
        assert_eq!(identified, RelationType::IsA);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_conceptnet_relation() {
        assert_eq!(RelationType::from_conceptnet("/r/IsA"), RelationType::IsA);
        assert_eq!(
            RelationType::from_conceptnet("/r/AtLocation"),
            RelationType::AtLocation
        );
        assert_eq!(
            RelationType::from_conceptnet("unknown_thing"),
            RelationType::Other
        );
    }

    #[test]
    fn test_weights() {
        assert!(RelationType::IsA.weight() > RelationType::RelatedTo.weight());
        assert!(RelationType::Causes.weight() > RelationType::FormOf.weight());
    }
}
