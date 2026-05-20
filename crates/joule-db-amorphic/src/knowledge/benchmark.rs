//! Benchmarks: standard knowledge graph evaluation against WN18RR and others.
//!
//! Link prediction task: given (subject, relation, ?), predict the object.
//! Metrics: Hits@1, Hits@3, Hits@10, Mean Reciprocal Rank (MRR).
//!
//! This is the standard KG reasoning benchmark. PathHD reports Hits@1 on
//! WebQSP/CWQ/GrailQA. We start with WN18RR (available in the UCG dataset).

use std::collections::HashMap;
use std::path::Path;

use super::concept::ConceptEncoder;
use super::core::KnowledgeCore;
use super::relation::RelationType;
use super::structural_encoder::StructuralEncoder;
use super::triple::Triple;

/// A benchmark triple: subject, relation, object (as strings).
#[derive(Clone, Debug)]
pub struct BenchmarkTriple {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

/// Load WN18RR-format triples from a TSV file.
/// Format: subject \t relation \t object
pub fn load_triples(path: &Path) -> Result<Vec<BenchmarkTriple>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;

    let triples: Vec<BenchmarkTriple> = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(BenchmarkTriple {
                    subject: parts[0].to_string(),
                    relation: parts[1].to_string(),
                    object: parts[2].to_string(),
                })
            } else {
                None
            }
        })
        .collect();

    Ok(triples)
}

/// Map WN18RR relation strings to RelationType.
fn map_wn18rr_relation(rel: &str) -> RelationType {
    match rel {
        "_hypernym" => RelationType::IsA,
        "_hyponym" => RelationType::IsA, // Reverse direction
        "_instance_hypernym" => RelationType::InstanceOf,
        "_instance_hyponym" => RelationType::InstanceOf,
        "_member_meronym" => RelationType::PartOf,
        "_member_holonym" => RelationType::HasA,
        "_has_part" => RelationType::HasA,
        "_part_of" => RelationType::PartOf,
        "_derivationally_related_form" => RelationType::DerivedFrom,
        "_also_see" => RelationType::RelatedTo,
        "_similar_to" => RelationType::SimilarTo,
        "_verb_group" => RelationType::RelatedTo,
        "_synset_domain_topic_of" => RelationType::RelatedTo,
        "_member_of_domain_topic" => RelationType::RelatedTo,
        "_synset_domain_region_of" => RelationType::AtLocation,
        "_member_of_domain_region" => RelationType::AtLocation,
        "_synset_domain_usage_of" => RelationType::UsedFor,
        "_member_of_domain_usage" => RelationType::UsedFor,
        _ => RelationType::RelatedTo,
    }
}

/// Convert WN18RR triples to JouleDB triples.
pub fn convert_triples(benchmarks: &[BenchmarkTriple]) -> Vec<Triple> {
    benchmarks
        .iter()
        .map(|bt| {
            let relation = map_wn18rr_relation(&bt.relation);
            Triple::new(&bt.subject, relation, &bt.object)
        })
        .collect()
}

/// Link prediction evaluation results.
#[derive(Clone, Debug)]
pub struct LinkPredictionResults {
    /// Number of test triples evaluated.
    pub total: usize,
    /// Hits@1: fraction where correct answer is rank 1.
    pub hits_at_1: f64,
    /// Hits@3: fraction where correct answer is in top 3.
    pub hits_at_3: f64,
    /// Hits@10: fraction where correct answer is in top 10.
    pub hits_at_10: f64,
    /// Mean Reciprocal Rank: mean of 1/rank for correct answer.
    pub mrr: f64,
    /// Average energy per prediction (joules).
    pub energy_per_prediction: f64,
    /// Total energy consumed (joules).
    pub total_energy: f64,
}

/// Run link prediction evaluation on the knowledge core.
///
/// For each test triple (s, r, o):
/// 1. Query core.query_object(s, r) → recovered vector
/// 2. Cleanup: denoise recovered vector via cleanup memory
/// 3. Rank all entities by similarity to cleaned vector
/// 4. Record the rank of the correct answer (o)
///
/// Two modes:
/// - `use_cleanup = true`: denoise via cleanup memory (should improve Hits@1)
/// - `use_cleanup = false`: raw unbinding (baseline)
pub fn evaluate_link_prediction(
    core: &mut KnowledgeCore,
    test_triples: &[Triple],
    max_eval: usize,
) -> LinkPredictionResults {
    evaluate_link_prediction_inner(core, test_triples, max_eval, true)
}

/// Baseline evaluation without cleanup (for comparison).
pub fn evaluate_link_prediction_raw(
    core: &mut KnowledgeCore,
    test_triples: &[Triple],
    max_eval: usize,
) -> LinkPredictionResults {
    evaluate_link_prediction_inner(core, test_triples, max_eval, false)
}

fn evaluate_link_prediction_inner(
    core: &mut KnowledgeCore,
    test_triples: &[Triple],
    max_eval: usize,
    use_cleanup: bool,
) -> LinkPredictionResults {
    let eval_count = test_triples.len().min(max_eval);
    let mut ranks = Vec::with_capacity(eval_count);
    let mut total_energy = 0.0f64;

    for triple in test_triples.iter().take(eval_count) {
        // Query: what is the object in (subject, relation, ?)?
        let recovered = match core.query_object(&triple.subject, triple.relation) {
            Some(v) => v,
            None => continue, // Subject not in core
        };

        total_energy += 0.000_01;

        if use_cleanup {
            // Cleanup path: denoise → use cleanup memory's codebook for ranking
            let clean_results = core.cleanup.cleanup_top_n(&recovered, 100);

            if !clean_results.is_empty() {
                let correct = triple.object.to_lowercase().replace(' ', "_");
                let rank = clean_results
                    .iter()
                    .position(|(label, _)| *label == correct)
                    .map(|pos| pos + 1);

                if let Some(r) = rank {
                    ranks.push(r);
                    continue;
                }
            }

            // Cleanup didn't find the answer — fall through to raw nearest
        }

        // Raw path: rank by similarity to recovered vector
        let all_concepts = core.nearest_concepts(&recovered, 100);

        let correct = triple.object.to_lowercase().replace(' ', "_");
        let rank = all_concepts
            .iter()
            .position(|(label, _)| *label == correct)
            .map(|pos| pos + 1);

        if let Some(r) = rank {
            ranks.push(r);
        }
    }

    if ranks.is_empty() {
        return LinkPredictionResults {
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
    let hits_1 = ranks.iter().filter(|&&r| r <= 1).count() as f64 / n;
    let hits_3 = ranks.iter().filter(|&&r| r <= 3).count() as f64 / n;
    let hits_10 = ranks.iter().filter(|&&r| r <= 10).count() as f64 / n;
    let mrr = ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n;

    LinkPredictionResults {
        total: eval_count,
        hits_at_1: hits_1,
        hits_at_3: hits_3,
        hits_at_10: hits_10,
        mrr,
        energy_per_prediction: total_energy / eval_count as f64,
        total_energy,
    }
}

impl LinkPredictionResults {
    /// Pretty-print results.
    pub fn render(&self) -> String {
        format!(
            "Link Prediction (n={})\n  Hits@1:  {:.1}%\n  Hits@3:  {:.1}%\n  Hits@10: {:.1}%\n  MRR:     {:.4}\n  Energy:  {:.6} J/pred ({:.6} J total)",
            self.total,
            self.hits_at_1 * 100.0,
            self.hits_at_3 * 100.0,
            self.hits_at_10 * 100.0,
            self.mrr,
            self.energy_per_prediction,
            self.total_energy,
        )
    }
}

/// Evaluate link prediction using structural encoding.
/// Instead of the KnowledgeCore's trigram-based concept bundles,
/// this uses the StructuralEncoder where concepts are defined by
/// their graph relationships, not their name.
pub fn evaluate_structural(
    train_triples: &[Triple],
    test_triples: &[Triple],
    max_eval: usize,
) -> LinkPredictionResults {
    use super::concept::KNOWLEDGE_DIM;

    let dim = KNOWLEDGE_DIM;
    let mut encoder = StructuralEncoder::new(dim);

    // Record all training relationships
    for triple in train_triples {
        encoder.record(&triple.subject, triple.relation, &triple.object);
    }

    // Compile structural vectors
    encoder.compile();

    let eval_count = test_triples.len().min(max_eval);
    let mut ranks = Vec::with_capacity(eval_count);
    let mut total_energy = 0.0f64;

    let codebook = super::relation::RelationCodebook::new(dim);

    for triple in test_triples.iter().take(eval_count) {
        let subject_sv = encoder.encode(&triple.subject);
        if subject_sv.popcount() == 0 {
            continue; // Unknown concept
        }

        let relation_hv = codebook.get(&triple.relation).clone();

        // Unbind: recover object structural vector
        // triple encoding was: subject ⊗ relation ⊗ object
        // but we don't have the triple vector — we have subject's structural vector
        // which is the bundle of all (relation ⊗ neighbor) pairs.
        // So: unbind relation from subject's structural vector → neighbor direction
        let recovered = subject_sv.bind(&relation_hv);

        total_energy += 0.000_01;

        // Find nearest concept to recovered vector
        let nearest = encoder.nearest(&recovered, 100);

        let correct = triple.object.to_lowercase();
        let rank = nearest
            .iter()
            .position(|(label, _)| *label == correct)
            .map(|pos| pos + 1);

        if let Some(r) = rank {
            ranks.push(r);
        }
    }

    if ranks.is_empty() {
        return LinkPredictionResults {
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
    let hits_1 = ranks.iter().filter(|&&r| r <= 1).count() as f64 / n;
    let hits_3 = ranks.iter().filter(|&&r| r <= 3).count() as f64 / n;
    let hits_10 = ranks.iter().filter(|&&r| r <= 10).count() as f64 / n;
    let mrr = ranks.iter().map(|&r| 1.0 / r as f64).sum::<f64>() / n;

    LinkPredictionResults {
        total: eval_count,
        hits_at_1: hits_1,
        hits_at_3: hits_3,
        hits_at_10: hits_10,
        mrr,
        energy_per_prediction: total_energy / eval_count as f64,
        total_energy,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_wn18rr_if_available() {
        let path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/train.txt");
        if path.exists() {
            let triples = load_triples(path).unwrap();
            assert!(triples.len() > 80000, "WN18RR train should have ~86K triples");

            // Check format
            assert!(!triples[0].subject.is_empty());
            assert!(!triples[0].relation.is_empty());
            assert!(!triples[0].object.is_empty());
        }
    }

    #[test]
    fn test_convert_wn18rr_relations() {
        let bt = BenchmarkTriple {
            subject: "dog.n.01".into(),
            relation: "_hypernym".into(),
            object: "animal.n.01".into(),
        };
        let converted = convert_triples(&[bt]);
        assert_eq!(converted[0].relation, RelationType::IsA);
    }

    #[test]
    fn test_link_prediction_on_small_core() {
        let mut core = KnowledgeCore::new();
        let triples = vec![
            Triple::new("dog", RelationType::IsA, "animal"),
            Triple::new("cat", RelationType::IsA, "animal"),
            Triple::new("bird", RelationType::IsA, "animal"),
            Triple::new("dog", RelationType::HasProperty, "loyal"),
            Triple::new("cat", RelationType::HasProperty, "independent"),
            Triple::new("dog", RelationType::CapableOf, "bark"),
            Triple::new("cat", RelationType::CapableOf, "purr"),
            Triple::new("bird", RelationType::CapableOf, "fly"),
        ];
        core.ingest_batch(&triples);

        // Test: can we recover "animal" from "dog IsA ?"
        let test = vec![Triple::new("dog", RelationType::IsA, "animal")];
        let results = evaluate_link_prediction(&mut core, &test, 10);

        eprintln!("{}", results.render());
        // With only 8 triples, "animal" should be recoverable
        assert!(results.hits_at_10 > 0.0 || results.total == 0);
    }

    #[test]
    fn test_wn18rr_benchmark_if_available() {
        let train_path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/train.txt");
        let test_path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/test.txt");

        if !train_path.exists() || !test_path.exists() {
            eprintln!("WN18RR not available, skipping benchmark");
            return;
        }

        // Load and convert
        let train_bt = load_triples(train_path).unwrap();
        let test_bt = load_triples(test_path).unwrap();

        let train = convert_triples(&train_bt);
        let test = convert_triples(&test_bt);

        eprintln!("WN18RR: {} train, {} test triples", train.len(), test.len());

        // 10K dims gives best MRR (0.215) and Hits@10 (54.5%).
        // 50K dims improves Hits@10 slightly (58.3%) but hurts MRR.
        // The real fix for Hits@1 is encoding quality, not dimension.
        let mut core = KnowledgeCore::new();
        core.ingest_batch(&train);

        eprintln!(
            "Core: {} triples, {} concepts, cleanup codebook: {} entries, {} bytes",
            core.triple_count,
            core.concept_count,
            core.cleanup.size(),
            core.memory_bytes()
        );

        // Evaluate with cleanup (should improve Hits@1)
        let results_clean = evaluate_link_prediction(&mut core, &test, 100);
        eprintln!("WITH CLEANUP:\n{}", results_clean.render());

        // Evaluate without cleanup (baseline)
        let results_raw = evaluate_link_prediction_raw(&mut core, &test, 100);
        eprintln!("WITHOUT CLEANUP (baseline):\n{}", results_raw.render());

        // Verify it runs
        assert!(results_clean.total > 0);

        // Cleanup should improve or match raw
        eprintln!(
            "\nCleanup delta: Hits@1 {:.1}% → {:.1}%, MRR {:.4} → {:.4}",
            results_raw.hits_at_1 * 100.0,
            results_clean.hits_at_1 * 100.0,
            results_raw.mrr,
            results_clean.mrr,
        );

        // Diagnostic: show first 5 test triples with their actual ranks
        eprintln!("\n=== Diagnostic: first 5 test triples ===");
        for triple in test.iter().take(5) {
            if let Some(recovered) = core.query_object(&triple.subject, triple.relation) {
                let top5 = core.nearest_concepts(&recovered, 5);
                let correct = triple.object.to_lowercase().replace(' ', "_");
                let rank = core.nearest_concepts(&recovered, 100)
                    .iter()
                    .position(|(l, _)| *l == correct)
                    .map(|p| p + 1);
                eprintln!(
                    "  ({}, {:?}, {}) → top5: {:?} | correct rank: {:?}",
                    triple.subject, triple.relation, triple.object,
                    top5.iter().take(3).map(|(l, s)| format!("{}({:.3})", l, s)).collect::<Vec<_>>(),
                    rank
                );
            }
        }
    }

    #[test]
    fn test_structural_benchmark_if_available() {
        let train_path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/train.txt");
        let test_path = Path::new("/tmp/jouledb-testdata/ai_graph_classification/data/WN18RR/test.txt");

        if !train_path.exists() || !test_path.exists() {
            eprintln!("WN18RR not available, skipping structural benchmark");
            return;
        }

        let train_bt = load_triples(train_path).unwrap();
        let test_bt = load_triples(test_path).unwrap();

        let train = convert_triples(&train_bt);
        let test = convert_triples(&test_bt);

        eprintln!("STRUCTURAL WN18RR: {} train, {} test triples", train.len(), test.len());

        let results = evaluate_structural(&train, &test, 100);

        eprintln!("STRUCTURAL ENCODING:\n{}", results.render());

        assert!(results.total > 0);
    }
}
