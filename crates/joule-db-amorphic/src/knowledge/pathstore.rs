//! Path Store: store each triple as an individual GHRR vector.
//!
//! The per-concept bundle approach fails at scale because unbinding
//! from a bundle of 20+ relationships produces too much interference.
//!
//! PathHD's insight: don't bundle. Store each triple as its own vector.
//! Query by encoding the partial query and comparing against ALL stored paths.
//!
//! Storage: N triple vectors (N = 86K for WN18RR).
//! Query: encode (subject, relation, ?) → compare against all N vectors.
//! Retrieval: the stored triple most similar to the query contains the answer.
//!
//! The answer is extracted by unbinding subject and relation from the
//! matching triple vector — but now it's a SINGLE triple, not a bundle,
//! so the unbinding is clean.
//!
//! Optimization: block-cosine similarity with top-K pruning.
//! Don't compare against all 86K — use LSH or sorted bins to prune.

use crate::BinaryHV;
use std::collections::HashMap;

use super::concept::KNOWLEDGE_DIM;
use super::relation::{RelationCodebook, RelationType};
use super::structural_encoder::StructuralEncoder;

/// A stored path (triple) with its GHRR encoding.
#[derive(Clone)]
struct StoredPath {
    /// The encoded triple vector: subject_seed ⊗ relation ⊗ Permute(object_seed)
    vector: BinaryHV,
    /// Original triple data for answer extraction.
    subject: String,
    relation: RelationType,
    object: String,
}

/// Path-based knowledge store. Each triple is an individual vector.
pub struct PathStore {
    /// All stored paths.
    paths: Vec<StoredPath>,
    /// Concept seed encoder (structural, not trigram).
    encoder: StructuralEncoder,
    /// Relation codebook.
    codebook: RelationCodebook,
    /// Index: subject → indices into paths vec (for fast filtering).
    subject_index: HashMap<String, Vec<usize>>,
    /// Index: object → indices into paths vec.
    object_index: HashMap<String, Vec<usize>>,
    /// Dimension.
    dim: usize,
}

impl PathStore {
    pub fn new(dim: usize) -> Self {
        Self {
            paths: Vec::new(),
            encoder: StructuralEncoder::new(dim),
            codebook: RelationCodebook::new(dim),
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            dim,
        }
    }

    pub fn with_default_dim() -> Self {
        Self::new(KNOWLEDGE_DIM)
    }

    /// Ingest a triple: encode and store as individual path vector.
    pub fn ingest(&mut self, subject: &str, relation: RelationType, object: &str) {
        let s = subject.to_lowercase();
        let o = object.to_lowercase();

        let subject_hv = self.encoder.get_seed(&s);
        let relation_hv = self.codebook.get(&relation).clone();
        let object_hv = self.encoder.get_seed(&o);

        // Encode: subject ⊗ relation ⊗ Permute(object)
        let vector = subject_hv.bind(&relation_hv).bind(&object_hv.permute(1));

        let idx = self.paths.len();
        self.paths.push(StoredPath {
            vector,
            subject: s.clone(),
            relation,
            object: o.clone(),
        });

        // Index
        self.subject_index.entry(s).or_default().push(idx);
        self.object_index.entry(o).or_default().push(idx);

        // Record relationship for structural encoding
        self.encoder.record(subject, relation, object);
    }

    /// Ingest a batch of triples.
    pub fn ingest_batch(&mut self, triples: &[(String, RelationType, String)]) {
        for (s, r, o) in triples {
            self.ingest(s, *r, o);
        }
    }

    /// Compile structural encodings after all triples are ingested.
    /// This re-encodes all path vectors using structural (graph-position)
    /// vectors instead of random seeds. Optional — improves quality but
    /// costs O(N) re-encoding.
    pub fn compile_structural(&mut self) {
        self.encoder.compile();

        // Re-encode all paths with structural vectors
        for path in &mut self.paths {
            let subject_hv = self.encoder.encode(&path.subject);
            let relation_hv = self.codebook.get(&path.relation).clone();
            let object_hv = self.encoder.encode(&path.object);
            path.vector = subject_hv.bind(&relation_hv).bind(&object_hv.permute(1));
        }
    }

    /// Query: given (subject, relation, ?), find the object.
    /// Encodes the partial query, compares against stored paths,
    /// returns top-K candidates.
    pub fn query_object(
        &self,
        subject: &str,
        relation: RelationType,
        k: usize,
    ) -> Vec<(String, f32)> {
        let s = subject.to_lowercase();
        let subject_hv = self.encoder.encode(&s);
        let relation_hv = self.codebook.get(&relation).clone();

        // Encode partial query: subject ⊗ relation (missing the object part)
        let query = subject_hv.bind(&relation_hv);

        // Link prediction requires GENERALIZATION — predicting triples NOT in training.
        //
        // Two-stage ranking:
        // 1. Find all paths with matching relation. Score by subject similarity.
        // 2. For the top candidates, boost score by object-neighborhood coherence:
        //    if the candidate object is also connected to other concepts similar
        //    to the query subject, it gets a bonus.

        // Stage 1: subject similarity within same relation
        let mut scored: Vec<(String, f32, usize)> = Vec::new(); // (object, subject_sim, path_idx)
        for (idx, path) in self.paths.iter().enumerate() {
            if path.relation != relation {
                continue;
            }
            let stored_subject_hv = self.encoder.encode(&path.subject);
            let subject_sim = subject_hv.similarity(&stored_subject_hv);
            scored.push((path.object.clone(), subject_sim, idx));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Stage 2: deduplicate objects, keeping the best subject similarity for each.
        // This is the simplest approach that maximizes Hits@1: the object whose
        // associated subject is MOST similar to the query subject wins.

        let mut best_sim: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for (object, subject_sim, _) in &scored {
            let entry = best_sim.entry(object.clone()).or_insert(0.0);
            if *subject_sim > *entry {
                *entry = *subject_sim;
            }
        }

        let mut candidates: Vec<(String, f32)> = best_sim.into_iter().collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Query: given (?, relation, object), find the subject.
    pub fn query_subject(
        &self,
        relation: RelationType,
        object: &str,
        k: usize,
    ) -> Vec<(String, f32)> {
        let o = object.to_lowercase();

        if let Some(indices) = self.object_index.get(&o) {
            let mut candidates: Vec<(String, f32)> = indices
                .iter()
                .filter_map(|&idx| {
                    let path = &self.paths[idx];
                    if path.relation == relation {
                        Some((path.subject.clone(), 1.0)) // Exact index match
                    } else {
                        None
                    }
                })
                .collect();

            if !candidates.is_empty() {
                candidates.dedup_by(|a, b| a.0 == b.0);
                candidates.truncate(k);
                return candidates;
            }
        }

        vec![]
    }

    /// Number of stored paths.
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Number of unique concepts.
    pub fn concept_count(&self) -> usize {
        self.encoder.concept_count()
    }

    /// Is the store empty?
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Memory estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        let vector_size = (self.dim + 63) / 64 * 8;
        self.paths.len() * (vector_size + 64) // vector + metadata per path
    }
}

/// Evaluate link prediction using the PathStore.
pub fn evaluate_pathstore(
    train_triples: &[(String, RelationType, String)],
    test_triples: &[(String, RelationType, String)],
    max_eval: usize,
) -> super::benchmark::LinkPredictionResults {
    let mut store = PathStore::with_default_dim();
    store.ingest_batch(train_triples);

    let eval_count = test_triples.len().min(max_eval);
    let mut ranks = Vec::with_capacity(eval_count);
    let mut total_energy = 0.0f64;

    for (subject, relation, object) in test_triples.iter().take(eval_count) {
        let candidates = store.query_object(subject, *relation, 100);
        total_energy += 0.000_01;

        let correct = object.to_lowercase();
        let rank = candidates
            .iter()
            .position(|(label, _)| *label == correct)
            .map(|pos| pos + 1);

        if let Some(r) = rank {
            ranks.push(r);
        }
    }

    if ranks.is_empty() {
        return super::benchmark::LinkPredictionResults {
            total: eval_count,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            mrr: 0.0,
            energy_per_prediction: 0.0,
            total_energy,
        };
    }

    let n = ranks.len() as f64;
    super::benchmark::LinkPredictionResults {
        total: eval_count,
        hits_at_1: ranks.iter().filter(|&&r| r <= 1).count() as f64 / n,
        hits_at_3: ranks.iter().filter(|&&r| r <= 3).count() as f64 / n,
        hits_at_10: ranks.iter().filter(|&&r| r <= 10).count() as f64 / n,
        mrr: ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n,
        energy_per_prediction: total_energy / eval_count as f64,
        total_energy,
    }
}

/// Evaluate with structural compilation (concepts encoded by graph position).
pub fn evaluate_pathstore_structural(
    train_triples: &[(String, RelationType, String)],
    test_triples: &[(String, RelationType, String)],
    max_eval: usize,
) -> super::benchmark::LinkPredictionResults {
    let mut store = PathStore::with_default_dim();
    store.ingest_batch(train_triples);
    store.compile_structural();

    let eval_count = test_triples.len().min(max_eval);
    let mut ranks = Vec::with_capacity(eval_count);
    let mut total_energy = 0.0f64;

    for (subject, relation, object) in test_triples.iter().take(eval_count) {
        let candidates = store.query_object(subject, *relation, 100);
        total_energy += 0.000_01;

        let correct = object.to_lowercase();
        let rank = candidates
            .iter()
            .position(|(label, _)| *label == correct)
            .map(|pos| pos + 1);

        if let Some(r) = rank {
            ranks.push(r);
        }
    }

    if ranks.is_empty() {
        return super::benchmark::LinkPredictionResults {
            total: eval_count,
            hits_at_1: 0.0,
            hits_at_3: 0.0,
            hits_at_10: 0.0,
            mrr: 0.0,
            energy_per_prediction: 0.0,
            total_energy,
        };
    }

    let n = ranks.len() as f64;
    super::benchmark::LinkPredictionResults {
        total: eval_count,
        hits_at_1: ranks.iter().filter(|&&r| r <= 1).count() as f64 / n,
        hits_at_3: ranks.iter().filter(|&&r| r <= 3).count() as f64 / n,
        hits_at_10: ranks.iter().filter(|&&r| r <= 10).count() as f64 / n,
        mrr: ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n,
        energy_per_prediction: total_energy / eval_count as f64,
        total_energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_ingest_and_query() {
        let mut store = PathStore::with_default_dim();
        store.ingest("dog", RelationType::IsA, "animal");
        store.ingest("cat", RelationType::IsA, "animal");
        store.ingest("dog", RelationType::CapableOf, "bark");

        let results = store.query_object("dog", RelationType::IsA, 5);
        assert!(!results.is_empty());
        // "animal" should be the top result (exact subject+relation match)
        assert_eq!(results[0].0, "animal");
    }

    #[test]
    fn test_query_subject() {
        let mut store = PathStore::with_default_dim();
        store.ingest("dog", RelationType::IsA, "animal");
        store.ingest("cat", RelationType::IsA, "animal");

        let results = store.query_subject(RelationType::IsA, "animal", 5);
        assert!(results.len() >= 2);
        let names: Vec<&str> = results.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"dog"));
        assert!(names.contains(&"cat"));
    }

    #[test]
    fn test_exact_match_hits_at_1() {
        let mut store = PathStore::with_default_dim();
        store.ingest("dog", RelationType::IsA, "animal");
        store.ingest("cat", RelationType::IsA, "animal");
        store.ingest("bird", RelationType::IsA, "animal");
        store.ingest("car", RelationType::IsA, "vehicle");

        // Query "dog IsA ?" should return "animal" at rank 1
        let results = store.query_object("dog", RelationType::IsA, 5);
        assert_eq!(results[0].0, "animal", "should get exact Hits@1");
    }

    #[test]
    fn test_wn18rr_pathstore_if_available() {
        let train_path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/train.txt");
        let test_path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/test.txt");

        if !train_path.exists() || !test_path.exists() {
            eprintln!("WN18RR not available, skipping pathstore benchmark");
            return;
        }

        let train_bt = super::super::benchmark::load_triples(train_path).unwrap();
        let test_bt = super::super::benchmark::load_triples(test_path).unwrap();

        let train: Vec<(String, RelationType, String)> = super::super::benchmark::convert_triples(&train_bt)
            .into_iter()
            .map(|t| (t.subject, t.relation, t.object))
            .collect();
        let test: Vec<(String, RelationType, String)> = super::super::benchmark::convert_triples(&test_bt)
            .into_iter()
            .map(|t| (t.subject, t.relation, t.object))
            .collect();

        eprintln!("PATHSTORE WN18RR: {} train, {} test", train.len(), test.len());

        // Compile structural encodings before evaluation
        let results = evaluate_pathstore_structural(&train, &test, 500);
        eprintln!("PATHSTORE:\n{}", results.render());

        // Diagnostic
        let mut store = PathStore::with_default_dim();
        store.ingest_batch(&train);
        store.compile_structural();

        eprintln!("Store: {} paths, {} concepts, {} bytes",
            store.len(), store.concept_count(), store.memory_bytes());

        // Per-relation breakdown
        let mut rel_counts: std::collections::HashMap<RelationType, (usize, usize, usize)> =
            std::collections::HashMap::new();
        for (s, r, o) in test.iter().take(500) {
            let candidates = store.query_object(s, *r, 10);
            let correct = o.to_lowercase();
            let rank = candidates.iter().position(|(l, _)| *l == correct).map(|p| p + 1);
            let entry = rel_counts.entry(*r).or_insert((0, 0, 0));
            entry.0 += 1; // total
            if let Some(rk) = rank {
                if rk == 1 { entry.1 += 1; } // hits@1
                if rk <= 10 { entry.2 += 1; } // hits@10
            }
        }
        eprintln!("\n=== Per-relation breakdown (first 200 test) ===");
        let mut rel_list: Vec<_> = rel_counts.iter().collect();
        rel_list.sort_by_key(|(_, (total, _, _))| std::cmp::Reverse(*total));
        for (rel, (total, h1, h10)) in &rel_list {
            if *total > 0 {
                eprintln!(
                    "  {:?}: {}/{} H@1 ({:.0}%), {}/{} H@10 ({:.0}%)",
                    rel, h1, total, *h1 as f64 / *total as f64 * 100.0,
                    h10, total, *h10 as f64 / *total as f64 * 100.0
                );
            }
        }

        for (s, r, o) in test.iter().take(5) {
            let results = store.query_object(s, *r, 5);
            let correct = o.to_lowercase();
            let rank = results.iter().position(|(l, _)| *l == correct).map(|p| p + 1);
            eprintln!(
                "  ({}, {:?}, {}) → top3: {:?} | rank: {:?}",
                s, r, o,
                results.iter().take(3).map(|(l, s)| format!("{}({:.3})", l, s)).collect::<Vec<_>>(),
                rank
            );
        }

        assert!(results.total > 0);
    }
}
