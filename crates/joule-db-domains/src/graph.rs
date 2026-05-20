//! HDC-powered Knowledge Graphs and Entity Relationship module
//!
//! Provides holographic encoding for:
//! - Entity and relationship encoding
//! - Link prediction via similarity
//! - Subgraph pattern matching
//! - Knowledge reasoning

use joule_db_hdc::{BinaryHV, BundleAccumulator};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const DIMENSION: usize = 10000;

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub entity_type: String,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: String,
    pub relation_type: String,
    pub source_id: String,
    pub target_id: String,
    pub properties: HashMap<String, String>,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone)]
pub struct PathQuery {
    pub start_entity: String,
    pub relation_pattern: Vec<String>,
    pub max_hops: usize,
}

// ============================================================================
// Graph Encoder
// ============================================================================

joule_db_hdc::define_domain_module! {
    /// HDC encoder for knowledge graph domain data
    pub struct GraphLink {
        seed: 0x6AA0_0001,
        dimension: 10000,
        fields: ["entity", "type", "relation", "source", "target", "property", "value"],
        scalars: [],
        enums: {
        },
        dynamic: {
            entity_type_vectors: "entity_type",
            relation_type_vectors: "relation_type",
            entity_vectors: "entity"
        },
    }
}

impl GraphLink {
    pub fn encode_entity(&mut self, entity: &Entity) -> BinaryHV {
        let id_hv = BinaryHV::from_hash(entity.id.as_bytes(), DIMENSION);
        let type_vec = self.entity_type_vectors(&entity.entity_type);

        let mut components = vec![
            self.field_vectors["entity"].bind(&id_hv),
            self.field_vectors["type"].bind(&type_vec),
        ];

        for (key, value) in &entity.properties {
            let prop_hv = BinaryHV::from_hash(key.as_bytes(), DIMENSION);
            let val_hv = BinaryHV::from_hash(value.as_bytes(), DIMENSION);
            components.push(self.field_vectors["property"].bind(&prop_hv).bind(&val_hv));
        }

        let entity_hv = self.bundle(&components);
        self.entity_vectors
            .insert(entity.id.clone(), entity_hv.clone());
        entity_hv
    }

    pub fn encode_relationship(&mut self, rel: &Relationship) -> BinaryHV {
        let rel_type_vec = self.relation_type_vectors(&rel.relation_type);
        let source_hv = self
            .entity_vectors
            .get(&rel.source_id)
            .cloned()
            .unwrap_or_else(|| BinaryHV::from_hash(rel.source_id.as_bytes(), DIMENSION));
        let target_hv = self
            .entity_vectors
            .get(&rel.target_id)
            .cloned()
            .unwrap_or_else(|| BinaryHV::from_hash(rel.target_id.as_bytes(), DIMENSION));

        let components = vec![
            self.field_vectors["relation"].bind(&rel_type_vec),
            self.field_vectors["source"].bind(&source_hv),
            self.field_vectors["target"].bind(&target_hv),
        ];

        self.bundle(&components)
    }

    pub fn encode_triple(&mut self, triple: &Triple) -> BinaryHV {
        let subj_hv = BinaryHV::from_hash(triple.subject.as_bytes(), DIMENSION);
        let pred_hv = self.relation_type_vectors(&triple.predicate);
        let obj_hv = BinaryHV::from_hash(triple.object.as_bytes(), DIMENSION);

        // Role-filler binding: subject XOR predicate XOR object with positional encoding
        subj_hv.bind(&pred_hv.permute(1)).bind(&obj_hv.permute(2))
    }
}

// ============================================================================
// Knowledge Graph Database
// ============================================================================

pub struct KnowledgeGraph {
    encoder: GraphLink,
    entity_hologram: BundleAccumulator,
    relationship_hologram: BundleAccumulator,
    triple_hologram: BundleAccumulator,
    entities: HashMap<String, Entity>,
    relationships: HashMap<String, Relationship>,
    adjacency: HashMap<String, HashSet<String>>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            encoder: GraphLink::new(),
            entity_hologram: BundleAccumulator::new(DIMENSION),
            relationship_hologram: BundleAccumulator::new(DIMENSION),
            triple_hologram: BundleAccumulator::new(DIMENSION),
            entities: HashMap::new(),
            relationships: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }

    pub fn add_entity(&mut self, entity: Entity) {
        let hv = self.encoder.encode_entity(&entity);
        self.entity_hologram.add(&hv);
        self.adjacency
            .entry(entity.id.clone())
            .or_insert_with(HashSet::new);
        self.entities.insert(entity.id.clone(), entity);
    }

    pub fn add_relationship(&mut self, rel: Relationship) {
        let hv = self.encoder.encode_relationship(&rel);
        self.relationship_hologram.add(&hv);

        // Update adjacency
        self.adjacency
            .entry(rel.source_id.clone())
            .or_insert_with(HashSet::new)
            .insert(rel.target_id.clone());
        self.adjacency
            .entry(rel.target_id.clone())
            .or_insert_with(HashSet::new)
            .insert(rel.source_id.clone());

        // Add as triple
        let triple = Triple {
            subject: rel.source_id.clone(),
            predicate: rel.relation_type.clone(),
            object: rel.target_id.clone(),
        };
        let triple_hv = self.encoder.encode_triple(&triple);
        self.triple_hologram.add(&triple_hv);

        self.relationships.insert(rel.id.clone(), rel);
    }

    pub fn find_similar_entities(
        &self,
        entity_id: &str,
        min_sim: f32,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let query_hv = match self.encoder.entity_vectors.get(entity_id) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut results: Vec<(String, f32)> = self
            .encoder
            .entity_vectors
            .iter()
            .filter(|(id, _)| *id != entity_id)
            .map(|(id, hv)| (id.clone(), query_hv.similarity(hv)))
            .filter(|(_, sim)| *sim >= min_sim)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    pub fn predict_link(&mut self, source_id: &str, target_id: &str) -> f32 {
        // Check if entities exist
        let source_hv = match self.encoder.entity_vectors.get(source_id) {
            Some(hv) => hv.clone(),
            None => return 0.0,
        };
        let target_hv = match self.encoder.entity_vectors.get(target_id) {
            Some(hv) => hv.clone(),
            None => return 0.0,
        };

        // Create hypothetical relationship vector
        let hypothetical = source_hv.bind(&target_hv);
        let rel_hologram = self.relationship_hologram.threshold();

        hypothetical.similarity(&rel_hologram)
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Link Predictor
// ============================================================================

pub struct LinkPredictor {
    encoder: GraphLink,
    positive_patterns: BundleAccumulator,
    negative_patterns: BundleAccumulator,
    pattern_count: usize,
}

#[derive(Debug, Clone)]
pub struct LinkPrediction {
    pub source_id: String,
    pub target_id: String,
    pub predicted_relation: String,
    pub confidence: f32,
}

impl LinkPredictor {
    pub fn new() -> Self {
        Self {
            encoder: GraphLink::new(),
            positive_patterns: BundleAccumulator::new(DIMENSION),
            negative_patterns: BundleAccumulator::new(DIMENSION),
            pattern_count: 0,
        }
    }

    pub fn train_positive(&mut self, rel: &Relationship) {
        let hv = self.encoder.encode_relationship(rel);
        self.positive_patterns.add(&hv);
        self.pattern_count += 1;
    }

    pub fn train_negative(&mut self, source_id: &str, target_id: &str) {
        let source_hv = BinaryHV::from_hash(source_id.as_bytes(), DIMENSION);
        let target_hv = BinaryHV::from_hash(target_id.as_bytes(), DIMENSION);
        let negative_hv = source_hv.bind(&target_hv);
        self.negative_patterns.add(&negative_hv);
    }

    pub fn predict(&self, source_id: &str, target_id: &str) -> f32 {
        let source_hv = BinaryHV::from_hash(source_id.as_bytes(), DIMENSION);
        let target_hv = BinaryHV::from_hash(target_id.as_bytes(), DIMENSION);
        let query_hv = source_hv.bind(&target_hv);

        let pos_sim = query_hv.similarity(&self.positive_patterns.threshold());
        let neg_sim = query_hv.similarity(&self.negative_patterns.threshold());

        // Return positive - negative similarity
        (pos_sim - neg_sim).max(0.0)
    }

    pub fn pattern_count(&self) -> usize {
        self.pattern_count
    }
}

impl Default for LinkPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_encoding() {
        let mut encoder = GraphLink::new();
        let entity = Entity {
            id: "person_1".to_string(),
            entity_type: "Person".to_string(),
            properties: [("name".to_string(), "Alice".to_string())]
                .into_iter()
                .collect(),
        };
        let hv = encoder.encode_entity(&entity);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_relationship_encoding() {
        let mut encoder = GraphLink::new();
        let rel = Relationship {
            id: "rel_1".to_string(),
            relation_type: "knows".to_string(),
            source_id: "person_1".to_string(),
            target_id: "person_2".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
        };
        let hv = encoder.encode_relationship(&rel);
        assert_eq!(hv.dimension(), DIMENSION);
    }

    #[test]
    fn test_knowledge_graph() {
        let mut kg = KnowledgeGraph::new();

        kg.add_entity(Entity {
            id: "alice".to_string(),
            entity_type: "Person".to_string(),
            properties: HashMap::new(),
        });
        kg.add_entity(Entity {
            id: "bob".to_string(),
            entity_type: "Person".to_string(),
            properties: HashMap::new(),
        });

        kg.add_relationship(Relationship {
            id: "rel_1".to_string(),
            relation_type: "knows".to_string(),
            source_id: "alice".to_string(),
            target_id: "bob".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
        });

        assert_eq!(kg.entity_count(), 2);
        assert_eq!(kg.relationship_count(), 1);
    }

    #[test]
    fn test_link_prediction() {
        let mut predictor = LinkPredictor::new();

        // Train with positive examples
        predictor.train_positive(&Relationship {
            id: "r1".to_string(),
            relation_type: "follows".to_string(),
            source_id: "user_1".to_string(),
            target_id: "user_2".to_string(),
            properties: HashMap::new(),
            weight: 1.0,
        });

        assert_eq!(predictor.pattern_count(), 1);

        // Prediction should return a score
        let score = predictor.predict("user_3", "user_4");
        assert!(score >= 0.0);
    }

    #[test]
    fn test_similar_entities() {
        let mut kg = KnowledgeGraph::new();

        // Add similar entities
        kg.add_entity(Entity {
            id: "company_a".to_string(),
            entity_type: "Company".to_string(),
            properties: [("industry".to_string(), "tech".to_string())]
                .into_iter()
                .collect(),
        });
        kg.add_entity(Entity {
            id: "company_b".to_string(),
            entity_type: "Company".to_string(),
            properties: [("industry".to_string(), "tech".to_string())]
                .into_iter()
                .collect(),
        });

        let similar = kg.find_similar_entities("company_a", 0.3, 10);
        assert!(!similar.is_empty());
    }
}
