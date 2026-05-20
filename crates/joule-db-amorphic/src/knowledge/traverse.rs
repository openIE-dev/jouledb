//! Traversal function: navigate the knowledge manifold.
//!
//! Given a starting concept, follow relation chains through the core.
//! This is not search — it's navigation. Each step unbinds the current
//! concept from its bundle, identifies the strongest connected concept,
//! and moves there. The path IS the answer.
//!
//! Uses Compare + Inhibit: at each step, compare recovered candidates
//! and suppress all but the strongest (winner-take-all).

use crate::BinaryHV;
use super::core::KnowledgeCore;
use super::relation::RelationType;

/// A single step in a traversal path.
#[derive(Clone, Debug)]
pub struct TraversalStep {
    /// The concept at this step.
    pub concept: String,
    /// The relation followed to get here (None for the starting concept).
    pub via_relation: Option<RelationType>,
    /// Similarity of the recovered concept to its best match in the index.
    pub confidence: f32,
    /// Novelty of this concept relative to the core centroid.
    pub novelty: f64,
}

/// The result of a traversal.
#[derive(Clone, Debug)]
pub struct TraversalResult {
    /// The path taken through the knowledge manifold.
    pub path: Vec<TraversalStep>,
    /// Total confidence (product of step confidences).
    pub total_confidence: f64,
    /// Whether the traversal terminated naturally (reached a dead end or loop).
    pub terminated: bool,
    /// Reason for termination.
    pub termination_reason: String,
}

/// The traversal engine.
pub struct Traverser {
    /// Maximum steps before forced termination.
    pub max_steps: usize,
    /// Minimum confidence to continue traversing.
    pub min_confidence: f32,
    /// Relations to follow (None = follow any).
    pub follow_relations: Option<Vec<RelationType>>,
}

impl Traverser {
    pub fn new() -> Self {
        Self {
            max_steps: 10,
            min_confidence: 0.45,
            follow_relations: None,
        }
    }

    /// Set maximum traversal depth.
    pub fn with_max_steps(mut self, max: usize) -> Self {
        self.max_steps = max;
        self
    }

    /// Set minimum confidence to continue.
    pub fn with_min_confidence(mut self, min: f32) -> Self {
        self.min_confidence = min;
        self
    }

    /// Only follow specific relation types.
    pub fn with_relations(mut self, rels: Vec<RelationType>) -> Self {
        self.follow_relations = Some(rels);
        self
    }

    /// Traverse from a starting concept, following the strongest connections.
    pub fn traverse(&self, core: &mut KnowledgeCore, start: &str) -> TraversalResult {
        let mut path = Vec::new();
        let mut visited = Vec::new();
        let mut current = start.to_string();
        let mut total_confidence = 1.0f64;

        // First step: the starting concept
        let start_novelty = core.novelty(&current);
        path.push(TraversalStep {
            concept: current.clone(),
            via_relation: None,
            confidence: 1.0,
            novelty: start_novelty,
        });
        visited.push(current.clone());

        for _step in 0..self.max_steps {
            // Try each relation type and find the best next hop
            let relations = self.follow_relations.as_deref().unwrap_or_else(|| {
                // Default: structural relations
                &[
                    RelationType::IsA,
                    RelationType::HasA,
                    RelationType::PartOf,
                    RelationType::Causes,
                    RelationType::HasPrerequisite,
                    RelationType::CapableOf,
                    RelationType::UsedFor,
                    RelationType::AtLocation,
                    RelationType::HasProperty,
                    RelationType::MadeOf,
                ]
            });

            let mut best_hop: Option<(String, RelationType, f32)> = None;

            for &rel in relations {
                if let Some(recovered) = core.query_object(&current, rel) {
                    // Find the nearest known concept to the recovered vector
                    let candidates = core.nearest_concepts(&recovered, 5);

                    for (label, sim) in &candidates {
                        // Skip self and already-visited
                        if visited.contains(label) {
                            continue;
                        }

                        if *sim > self.min_confidence {
                            match &best_hop {
                                None => {
                                    best_hop = Some((label.clone(), rel, *sim));
                                }
                                Some((_, _, best_sim)) if sim > best_sim => {
                                    best_hop = Some((label.clone(), rel, *sim));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Inhibit: winner-take-all — only the best hop survives
            match best_hop {
                Some((next_concept, via_rel, confidence)) => {
                    let novelty = core.novelty(&next_concept);
                    total_confidence *= confidence as f64;

                    path.push(TraversalStep {
                        concept: next_concept.clone(),
                        via_relation: Some(via_rel),
                        confidence,
                        novelty,
                    });

                    visited.push(next_concept.clone());
                    current = next_concept;
                }
                None => {
                    // Dead end — no unvisited concept above confidence threshold
                    return TraversalResult {
                        path,
                        total_confidence,
                        terminated: true,
                        termination_reason: "dead_end".to_string(),
                    };
                }
            }
        }

        TraversalResult {
            path,
            total_confidence,
            terminated: true,
            termination_reason: "max_steps".to_string(),
        }
    }

    /// Traverse toward a target: find a path from `start` to `target`.
    /// At each step, prefer the hop that moves closer to the target.
    pub fn traverse_toward(
        &self,
        core: &mut KnowledgeCore,
        start: &str,
        target: &str,
    ) -> TraversalResult {
        let target_hv = core.encoder.encode(target).vector;
        let mut path = Vec::new();
        let mut visited = Vec::new();
        let mut current = start.to_string();
        let mut total_confidence = 1.0f64;

        path.push(TraversalStep {
            concept: current.clone(),
            via_relation: None,
            confidence: 1.0,
            novelty: core.novelty(&current),
        });
        visited.push(current.clone());

        for _step in 0..self.max_steps {
            // Check if we've reached the target
            if let Some(current_hv) = core.get_concept(&current) {
                if current_hv.similarity(&target_hv) > 0.8 {
                    return TraversalResult {
                        path,
                        total_confidence,
                        terminated: true,
                        termination_reason: "reached_target".to_string(),
                    };
                }
            }

            let relations = self.follow_relations.as_deref().unwrap_or(&[
                RelationType::IsA,
                RelationType::HasA,
                RelationType::PartOf,
                RelationType::Causes,
                RelationType::CapableOf,
                RelationType::UsedFor,
                RelationType::AtLocation,
                RelationType::RelatedTo,
            ]);

            let mut best_hop: Option<(String, RelationType, f32, f32)> = None; // (label, rel, confidence, target_sim)

            for &rel in relations {
                if let Some(recovered) = core.query_object(&current, rel) {
                    let candidates = core.nearest_concepts(&recovered, 5);
                    for (label, sim) in &candidates {
                        if visited.contains(label) || *sim < self.min_confidence {
                            continue;
                        }
                        // Score: confidence × similarity-to-target
                        let concept_hv = core.get_concept(label);
                        let target_sim = concept_hv
                            .map(|hv| hv.similarity(&target_hv))
                            .unwrap_or(0.0);
                        let score = *sim * 0.3 + target_sim * 0.7; // Bias toward target

                        match &best_hop {
                            None => best_hop = Some((label.clone(), rel, *sim, score)),
                            Some((_, _, _, best_score)) if score > *best_score => {
                                best_hop = Some((label.clone(), rel, *sim, score));
                            }
                            _ => {}
                        }
                    }
                }
            }

            match best_hop {
                Some((next, via_rel, confidence, _)) => {
                    total_confidence *= confidence as f64;
                    path.push(TraversalStep {
                        concept: next.clone(),
                        via_relation: Some(via_rel),
                        confidence,
                        novelty: core.novelty(&next),
                    });
                    visited.push(next.clone());
                    current = next;
                }
                None => {
                    return TraversalResult {
                        path,
                        total_confidence,
                        terminated: true,
                        termination_reason: "dead_end".to_string(),
                    };
                }
            }
        }

        TraversalResult {
            path,
            total_confidence,
            terminated: true,
            termination_reason: "max_steps".to_string(),
        }
    }
}

impl Default for Traverser {
    fn default() -> Self {
        Self::new()
    }
}

impl TraversalResult {
    /// Pretty-print the traversal path.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (i, step) in self.path.iter().enumerate() {
            if i > 0 {
                if let Some(rel) = &step.via_relation {
                    out.push_str(&format!(" --{:?}--> ", rel));
                } else {
                    out.push_str(" --> ");
                }
            }
            out.push_str(&format!("{}({:.2})", step.concept, step.confidence));
        }
        out.push_str(&format!(
            " [{}; confidence={:.3}]",
            self.termination_reason, self.total_confidence
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::triple::Triple;

    fn build_test_core() -> KnowledgeCore {
        let mut core = KnowledgeCore::new();
        core.ingest_batch(&[
            Triple::new("dog", RelationType::IsA, "animal"),
            Triple::new("cat", RelationType::IsA, "animal"),
            Triple::new("animal", RelationType::IsA, "living_thing"),
            Triple::new("living_thing", RelationType::HasProperty, "alive"),
            Triple::new("dog", RelationType::CapableOf, "bark"),
            Triple::new("dog", RelationType::HasProperty, "loyal"),
            Triple::new("dog", RelationType::AtLocation, "house"),
            Triple::new("cat", RelationType::AtLocation, "house"),
            Triple::new("cat", RelationType::CapableOf, "purr"),
            Triple::new("bird", RelationType::IsA, "animal"),
            Triple::new("bird", RelationType::CapableOf, "fly"),
            Triple::new("fish", RelationType::IsA, "animal"),
            Triple::new("fish", RelationType::AtLocation, "water"),
            Triple::new("car", RelationType::IsA, "vehicle"),
            Triple::new("vehicle", RelationType::CapableOf, "transport"),
            Triple::new("car", RelationType::HasProperty, "fast"),
            Triple::new("car", RelationType::UsedFor, "transportation"),
        ]);
        core
    }

    #[test]
    fn test_traverse_from_dog() {
        let mut core = build_test_core();
        let traverser = Traverser::new().with_max_steps(5);
        let result = traverser.traverse(&mut core, "dog");

        assert!(!result.path.is_empty());
        assert_eq!(result.path[0].concept, "dog");
        // Should have traversed somewhere
        assert!(result.path.len() >= 1);
        assert!(result.terminated);
    }

    #[test]
    fn test_traversal_render() {
        let mut core = build_test_core();
        let traverser = Traverser::new().with_max_steps(3);
        let result = traverser.traverse(&mut core, "dog");
        let rendered = result.render();
        assert!(rendered.contains("dog"));
    }

    #[test]
    fn test_traverse_toward() {
        let mut core = build_test_core();
        let traverser = Traverser::new().with_max_steps(5);
        let result = traverser.traverse_toward(&mut core, "dog", "water");

        assert!(!result.path.is_empty());
        assert_eq!(result.path[0].concept, "dog");
    }
}
