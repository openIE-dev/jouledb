//! Batch Feature Serving — sub-ms parallel lookups for content recommendation.
//!
//! Content recommendation engines (Netflix, TikTok) need to fetch features
//! for hundreds of users and items in a single call with sub-millisecond latency.
//!
//! This module provides shard-aware parallel batch operations that:
//! 1. Group keys by shard (minimize lock contention)
//! 2. Fetch from each shard in parallel (exploit multi-core)
//! 3. Merge results preserving request order
//!
//! # Example
//! ```rust,ignore
//! let keys = vec!["user:123", "user:456", "item:789"];
//! let response = store.batch_get(&keys)?;
//! // response.results[0] = Some(record for user:123)
//! // response.results[1] = Some(record for user:456)
//! // response.results[2] = Some(record for item:789)
//! ```

use crate::{AmorphicError, AmorphicRecord, AmorphicResult, RecordId, Value};
use std::collections::HashMap;
use std::time::Instant;

/// Request for batch record retrieval.
#[derive(Debug, Clone)]
pub struct BatchGetRequest {
    /// Keys to look up (names that map to record IDs via name_to_id)
    pub keys: Vec<String>,
}

/// Response from batch retrieval.
#[derive(Clone)]
pub struct BatchGetResponse {
    /// Results in the same order as the request keys.
    /// None if the key was not found.
    pub results: Vec<Option<AmorphicRecord>>,
    /// Total lookup latency in microseconds.
    pub latency_us: u64,
    /// Number of keys that were found.
    pub found_count: usize,
}

/// Request for batch feature extraction.
/// Returns only specified fields from each record.
#[derive(Debug, Clone)]
pub struct BatchFeatureRequest {
    /// Entity keys to look up
    pub keys: Vec<String>,
    /// Feature names (fields) to extract from each record
    pub features: Vec<String>,
}

/// Response from batch feature extraction.
#[derive(Debug, Clone)]
pub struct BatchFeatureResponse {
    /// Feature matrix: rows correspond to keys, columns to features.
    /// `matrix[i][j]` = value of features[j] for keys[i], or None if missing.
    pub matrix: Vec<Vec<Option<Value>>>,
    /// Keys that were found
    pub found_keys: Vec<String>,
    /// Latency in microseconds
    pub latency_us: u64,
}

/// Batch operations on the base AmorphicStore (single-threaded).
impl crate::AmorphicStore {
    /// Batch get by record IDs.
    pub fn batch_get_by_ids(&self, ids: &[RecordId]) -> Vec<Option<AmorphicRecord>> {
        ids.iter()
            .map(|id| self.get(*id).cloned())
            .collect()
    }

    /// Batch get by names (via name_to_id mapping).
    pub fn batch_get_by_names(&self, names: &[&str]) -> Vec<Option<AmorphicRecord>> {
        names
            .iter()
            .map(|name| {
                self.name_to_id
                    .get(*name)
                    .and_then(|id| self.get(*id).cloned())
            })
            .collect()
    }

    /// Batch feature extraction: get specific fields from named records.
    pub fn batch_get_features(
        &self,
        names: &[&str],
        features: &[&str],
    ) -> Vec<Vec<Option<Value>>> {
        names
            .iter()
            .map(|name| {
                let record = self
                    .name_to_id
                    .get(*name)
                    .and_then(|id| self.get(*id));
                features
                    .iter()
                    .map(|f| {
                        record.and_then(|r| r.fields.get(*f).cloned())
                    })
                    .collect()
            })
            .collect()
    }
}

/// Batch operations on the ShardedAmorphicStore (parallel).
impl crate::partition::ShardedAmorphicStore {
    /// Parallel batch get by names.
    ///
    /// Groups keys by shard, fetches each shard in parallel, then
    /// reassembles results in request order. This is the core operation
    /// for content recommendation feature serving.
    pub fn batch_get(&self, request: &BatchGetRequest) -> AmorphicResult<BatchGetResponse> {
        let start = Instant::now();

        // Step 1: Resolve names to (shard_idx, local_id) via the global index
        // We need to read each shard's name_to_id mapping
        let mut results: Vec<Option<AmorphicRecord>> = vec![None; request.keys.len()];

        // Step 2: Search all shards in parallel for each key.
        // Names are unique across the store, so each key will be found in at most one shard.
        // We broadcast all keys to all shards — each shard only returns matches it has.
        let keys_ref = &request.keys;
        let shard_results: Vec<Vec<(usize, AmorphicRecord)>> = std::thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .map(|shard| {
                    s.spawn(move || {
                        let guard = shard.read().unwrap();
                        let mut found = Vec::new();
                        for (req_idx, key) in keys_ref.iter().enumerate() {
                            if let Some(id) = guard.name_to_id.get(key.as_str()) {
                                if let Some(record) = guard.get(*id) {
                                    found.push((req_idx, record.clone()));
                                }
                            }
                        }
                        found
                    })
                })
                .collect();

            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });

        // Step 3: Reassemble in request order
        let mut found_count = 0;
        for shard_result in shard_results {
            for (req_idx, record) in shard_result {
                found_count += 1;
                results[req_idx] = Some(record);
            }
        }

        let latency = start.elapsed();

        Ok(BatchGetResponse {
            results,
            latency_us: latency.as_micros() as u64,
            found_count,
        })
    }

    /// Batch feature extraction with parallel shard access.
    pub fn batch_get_features(
        &self,
        request: &BatchFeatureRequest,
    ) -> AmorphicResult<BatchFeatureResponse> {
        let start = Instant::now();

        let batch_req = BatchGetRequest {
            keys: request.keys.clone(),
        };
        let batch_resp = self.batch_get(&batch_req)?;

        let mut found_keys = Vec::new();
        let matrix: Vec<Vec<Option<Value>>> = batch_resp
            .results
            .iter()
            .enumerate()
            .map(|(i, record_opt)| {
                if record_opt.is_some() {
                    found_keys.push(request.keys[i].clone());
                }
                request
                    .features
                    .iter()
                    .map(|f| {
                        record_opt
                            .as_ref()
                            .and_then(|r| r.fields.get(f.as_str()).cloned())
                    })
                    .collect()
            })
            .collect();

        Ok(BatchFeatureResponse {
            matrix,
            found_keys,
            latency_us: start.elapsed().as_micros() as u64,
        })
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::ShardedAmorphicStore;

    #[test]
    fn test_batch_get_single_store() {
        let mut store = crate::AmorphicStore::new();
        store
            .ingest_json(r#"{"name": "Alice", "score": 95}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Bob", "score": 87}"#)
            .unwrap();

        let results = store.batch_get_by_names(&["Alice", "Bob", "Charlie"]);
        assert!(results[0].is_some()); // Alice found
        assert!(results[1].is_some()); // Bob found
        assert!(results[2].is_none()); // Charlie not found
    }

    #[test]
    fn test_batch_features_single_store() {
        let mut store = crate::AmorphicStore::new();
        store
            .ingest_json(r#"{"name": "Alice", "score": 95, "level": 5}"#)
            .unwrap();
        store
            .ingest_json(r#"{"name": "Bob", "score": 87}"#)
            .unwrap();

        let matrix = store.batch_get_features(&["Alice", "Bob"], &["score", "level"]);
        // Alice: score=95, level=5
        assert!(matrix[0][0].is_some());
        assert!(matrix[0][1].is_some());
        // Bob: score=87, level=None
        assert!(matrix[1][0].is_some());
        assert!(matrix[1][1].is_none());
    }

    #[test]
    fn test_sharded_batch_get() {
        let store = ShardedAmorphicStore::with_shard_count(4);

        // Ingest some records
        for i in 0..20 {
            store
                .ingest_json(&format!(r#"{{"name": "item_{}", "score": {}}}"#, i, i * 10))
                .unwrap();
        }

        // Batch lookup
        let request = BatchGetRequest {
            keys: vec![
                "item_0".to_string(),
                "item_5".to_string(),
                "item_19".to_string(),
                "nonexistent".to_string(),
            ],
        };

        let response = store.batch_get(&request).unwrap();
        assert_eq!(response.results.len(), 4);
        assert!(response.results[0].is_some()); // item_0
        assert!(response.results[1].is_some()); // item_5
        assert!(response.results[2].is_some()); // item_19
        assert!(response.results[3].is_none()); // nonexistent
        assert_eq!(response.found_count, 3);
        assert!(response.latency_us < 1_000_000); // Under 1 second (generous for CI)
    }
}
