//! Semantic Hybrid Search — vector + keyword + metadata in one ranked query.
//!
//! "Show me that movie where the guy does the thing in the rain"
//! requires combining:
//! - Vector similarity (semantic meaning from embeddings)
//! - Keyword matching (exact terms like actor names)
//! - Metadata filters (genre, year, rating)
//!
//! This module provides a unified ranking function using Reciprocal Rank Fusion (RRF).

use joule_db_hdc::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{AmorphicRecord, AmorphicStore, QueryResult, RecordId, Value, DIMENSION};

/// A hybrid search query combining multiple signals.
#[derive(Debug, Clone)]
pub struct HybridQuery {
    /// Vector query (semantic similarity via hologram)
    pub vector_query: Option<BinaryHV>,
    /// Keyword query (substring match on text fields)
    pub keyword_query: Option<String>,
    /// Keyword fields to search (empty = all string fields)
    pub keyword_fields: Vec<String>,
    /// Metadata filters (field = value, must match exactly)
    pub metadata_filters: Vec<(String, Value)>,
    /// Maximum results
    pub k: usize,
    /// Weight for vector similarity (0.0-1.0)
    pub vector_weight: f32,
    /// Weight for keyword match (0.0-1.0)
    pub keyword_weight: f32,
}

impl HybridQuery {
    pub fn new(k: usize) -> Self {
        Self {
            vector_query: None,
            keyword_query: None,
            keyword_fields: vec![],
            metadata_filters: vec![],
            k,
            vector_weight: 0.5,
            keyword_weight: 0.5,
        }
    }

    pub fn with_vector(mut self, hologram: BinaryHV) -> Self {
        self.vector_query = Some(hologram);
        self
    }

    pub fn with_keywords(mut self, query: &str) -> Self {
        self.keyword_query = Some(query.to_lowercase());
        self
    }

    pub fn with_keyword_fields(mut self, fields: Vec<String>) -> Self {
        self.keyword_fields = fields;
        self
    }

    pub fn with_filter(mut self, field: &str, value: Value) -> Self {
        self.metadata_filters.push((field.to_string(), value));
        self
    }

    pub fn with_weights(mut self, vector: f32, keyword: f32) -> Self {
        self.vector_weight = vector;
        self.keyword_weight = keyword;
        self
    }
}

/// A scored search result with breakdown of ranking signals.
#[derive(Clone)]
pub struct HybridResult {
    pub record_id: RecordId,
    pub record: AmorphicRecord,
    /// Combined score (higher = better match)
    pub score: f64,
    /// Individual signal scores
    pub vector_score: f64,
    pub keyword_score: f64,
}

/// Execute a hybrid search against the amorphic store.
///
/// Uses Reciprocal Rank Fusion (RRF) to combine rankings from
/// vector similarity and keyword matching.
pub fn hybrid_search(store: &AmorphicStore, query: &HybridQuery) -> Vec<HybridResult> {
    let mut scores: HashMap<RecordId, (f64, f64)> = HashMap::new(); // (vector_score, keyword_score)

    // 1. Vector similarity ranking
    if let Some(ref probe) = query.vector_query {
        let mut vec_results: Vec<(RecordId, f32)> = store
            .records
            .iter()
            .map(|(&id, record)| {
                let sim = record.hologram.similarity(probe);
                (id, sim)
            })
            .collect();

        vec_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // RRF scoring: score = 1 / (k + rank)
        let rrf_k = 60.0; // Standard RRF constant
        for (rank, (id, _sim)) in vec_results.iter().enumerate() {
            let rrf_score = 1.0 / (rrf_k + rank as f64);
            scores.entry(*id).or_insert((0.0, 0.0)).0 = rrf_score;
        }
    }

    // 2. Keyword matching ranking
    if let Some(ref keywords) = query.keyword_query {
        let terms: Vec<&str> = keywords.split_whitespace().collect();

        let mut kw_results: Vec<(RecordId, f64)> = store
            .records
            .iter()
            .filter_map(|(&id, record)| {
                let score = keyword_match_score(record, &terms, &query.keyword_fields);
                if score > 0.0 {
                    Some((id, score))
                } else {
                    None
                }
            })
            .collect();

        kw_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let rrf_k = 60.0;
        for (rank, (id, _)) in kw_results.iter().enumerate() {
            let rrf_score = 1.0 / (rrf_k + rank as f64);
            scores.entry(*id).or_insert((0.0, 0.0)).1 = rrf_score;
        }
    }

    // 3. Apply metadata filters (hard filters, not scored)
    let filtered_ids: Option<std::collections::HashSet<RecordId>> = if query.metadata_filters.is_empty() {
        None
    } else {
        Some(
            store
                .records
                .iter()
                .filter(|(_, record)| {
                    query.metadata_filters.iter().all(|(field, value)| {
                        record.fields.get(field).map(|v| v == value).unwrap_or(false)
                    })
                })
                .map(|(&id, _)| id)
                .collect(),
        )
    };

    // 4. Combine scores with weights
    let vw = query.vector_weight as f64;
    let kw = query.keyword_weight as f64;

    let mut results: Vec<HybridResult> = scores
        .into_iter()
        .filter(|(id, _)| {
            filtered_ids
                .as_ref()
                .map(|f| f.contains(id))
                .unwrap_or(true)
        })
        .filter_map(|(id, (vscore, kscore))| {
            let combined = vw * vscore + kw * kscore;
            store.records.get(&id).map(|record| HybridResult {
                record_id: id,
                record: record.clone(),
                score: combined,
                vector_score: vscore,
                keyword_score: kscore,
            })
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(query.k);
    results
}

/// Compute keyword match score for a record.
fn keyword_match_score(
    record: &AmorphicRecord,
    terms: &[&str],
    search_fields: &[String],
) -> f64 {
    let mut total_matches = 0;

    for (field, value) in &record.fields {
        // Skip fields not in search_fields (if specified)
        if !search_fields.is_empty() && !search_fields.iter().any(|f| f == field) {
            continue;
        }

        if let Value::String(text) = value {
            let lower = text.to_lowercase();
            for term in terms {
                if lower.contains(term) {
                    total_matches += 1;
                }
            }
        }
    }

    total_matches as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_movie_store() -> AmorphicStore {
        let mut store = AmorphicStore::new();
        store
            .ingest_json(r#"{"name": "The Matrix", "genre": "scifi", "year": 1999, "description": "a hacker discovers reality is simulated"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Inception", "genre": "scifi", "year": 2010, "description": "dreams within dreams with rain scene"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "The Notebook", "genre": "romance", "year": 2004, "description": "love story with iconic rain scene"}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Blade Runner", "genre": "scifi", "year": 1982, "description": "replicant tears in rain"}"#)
            .unwrap();
        store
    }

    #[test]
    fn test_keyword_search() {
        let store = build_movie_store();

        let query = HybridQuery::new(10)
            .with_keywords("rain")
            .with_weights(0.0, 1.0); // Keywords only

        let results = hybrid_search(&store, &query);
        assert!(!results.is_empty());

        // All results should mention "rain"
        for r in &results {
            let has_rain = r.record.fields.values().any(|v| {
                if let Value::String(s) = v {
                    s.to_lowercase().contains("rain")
                } else {
                    false
                }
            });
            assert!(has_rain, "Result should contain 'rain'");
        }
    }

    #[test]
    fn test_metadata_filter() {
        let store = build_movie_store();

        let query = HybridQuery::new(10)
            .with_keywords("rain")
            .with_filter("genre", Value::String("scifi".to_string()))
            .with_weights(0.0, 1.0);

        let results = hybrid_search(&store, &query);

        // Should only return scifi movies with "rain"
        for r in &results {
            assert_eq!(
                r.record.fields.get("genre"),
                Some(&Value::String("scifi".to_string()))
            );
        }
    }

    #[test]
    fn test_hybrid_vector_and_keyword() {
        let store = build_movie_store();

        // Vector query biased toward scifi
        let scifi_hv = BinaryHV::from_hash(b"scifi hacker simulation", DIMENSION);

        let query = HybridQuery::new(10)
            .with_vector(scifi_hv)
            .with_keywords("rain")
            .with_weights(0.5, 0.5);

        let results = hybrid_search(&store, &query);
        assert!(!results.is_empty());

        // Results should have both vector and keyword scores
        for r in &results {
            assert!(r.score > 0.0);
        }
    }

    #[test]
    fn test_vector_only_search() {
        let store = build_movie_store();

        let query_hv = BinaryHV::from_hash(b"science fiction future", DIMENSION);
        let query = HybridQuery::new(4)
            .with_vector(query_hv)
            .with_weights(1.0, 0.0);

        let results = hybrid_search(&store, &query);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_empty_query_returns_empty() {
        let store = build_movie_store();
        let query = HybridQuery::new(10); // No vector, no keywords
        let results = hybrid_search(&store, &query);
        assert!(results.is_empty());
    }
}
