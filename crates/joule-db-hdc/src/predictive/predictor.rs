//! Query prediction using Markov chains and N-grams

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Predictor errors
#[derive(Error, Debug, Clone)]
pub enum PredictorError {
    /// No predictions available
    #[error("No predictions available")]
    NoPredictions,

    /// Lock error
    #[error("Lock poisoned")]
    LockPoisoned,
}

/// A predicted query with confidence
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Hash of the predicted query
    pub hash: u64,
    /// Number of times this transition was observed
    pub count: u32,
    /// Probability of this prediction
    pub probability: f64,
}

/// Statistics about the predictor
#[derive(Debug, Clone)]
pub struct PredictorStats {
    /// Number of unique queries seen
    pub unique_queries: usize,
    /// Total number of transitions learned
    pub total_transitions: usize,
    /// Current cache size
    pub cache_size: usize,
    /// History length
    pub history_length: usize,
}

/// Markov-chain based query predictor
pub struct QueryPredictor {
    transitions: Arc<RwLock<HashMap<u64, HashMap<u64, u32>>>>,
    last_query: Arc<RwLock<Option<u64>>>,
    prediction_cache: Arc<RwLock<HashMap<u64, Vec<u8>>>>,
    query_history: Arc<RwLock<Vec<u64>>>,
    history_size: usize,
    cache_size: usize,
    cache_hits: Arc<RwLock<u64>>,
    cache_misses: Arc<RwLock<u64>>,
}

impl QueryPredictor {
    /// Create new query predictor
    pub fn new(history_size: usize, cache_size: usize) -> Self {
        Self {
            transitions: Arc::new(RwLock::new(HashMap::new())),
            last_query: Arc::new(RwLock::new(None)),
            prediction_cache: Arc::new(RwLock::new(HashMap::new())),
            query_history: Arc::new(RwLock::new(Vec::new())),
            history_size,
            cache_size,
            cache_hits: Arc::new(RwLock::new(0)),
            cache_misses: Arc::new(RwLock::new(0)),
        }
    }

    /// Hash a query string
    pub fn hash_query(query: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }

    /// Observe a query (learn pattern)
    pub fn observe(&self, query: &str) {
        let query_hash = Self::hash_query(query);

        // Update transition matrix
        if let Some(prev) = *self.last_query.read().unwrap() {
            let mut transitions = self.transitions.write().unwrap();
            *transitions
                .entry(prev)
                .or_default()
                .entry(query_hash)
                .or_insert(0) += 1;
        }

        // Update last query
        *self.last_query.write().unwrap() = Some(query_hash);

        // Update history
        let mut history = self.query_history.write().unwrap();
        history.push(query_hash);
        if history.len() > self.history_size {
            history.remove(0);
        }
    }

    /// Predict next likely queries
    pub fn predict_next(&self, top_k: usize) -> Vec<Prediction> {
        if let Some(current) = *self.last_query.read().unwrap() {
            let transitions = self.transitions.read().unwrap();
            if let Some(nexts) = transitions.get(&current) {
                let total: u32 = nexts.values().sum();
                let mut predictions: Vec<_> = nexts
                    .iter()
                    .map(|(&hash, &count)| Prediction {
                        hash,
                        count,
                        probability: count as f64 / total as f64,
                    })
                    .collect();
                predictions.sort_by(|a, b| b.count.cmp(&a.count));
                predictions.truncate(top_k);
                return predictions;
            }
        }
        Vec::new()
    }

    /// Get prediction hashes only
    pub fn predict_next_hashes(&self, top_k: usize) -> Vec<u64> {
        self.predict_next(top_k)
            .into_iter()
            .map(|p| p.hash)
            .collect()
    }

    /// Cache a query result
    pub fn cache_result(&self, query: &str, result: &[u8]) {
        let query_hash = Self::hash_query(query);
        let mut cache = self.prediction_cache.write().unwrap();

        // Evict if cache full (simple FIFO)
        if cache.len() >= self.cache_size {
            if let Some(key) = cache.keys().next().cloned() {
                cache.remove(&key);
            }
        }

        cache.insert(query_hash, result.to_vec());
    }

    /// Get cached result
    pub fn get_cached(&self, query: &str) -> Option<Vec<u8>> {
        let query_hash = Self::hash_query(query);
        let cache = self.prediction_cache.read().unwrap();
        let result = cache.get(&query_hash).cloned();

        // Update hit/miss stats
        if result.is_some() {
            *self.cache_hits.write().unwrap() += 1;
        } else {
            *self.cache_misses.write().unwrap() += 1;
        }

        result
    }

    /// Check if query is in cache
    pub fn is_cached(&self, query: &str) -> bool {
        let query_hash = Self::hash_query(query);
        self.prediction_cache
            .read()
            .unwrap()
            .contains_key(&query_hash)
    }

    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = *self.cache_hits.read().unwrap();
        let misses = *self.cache_misses.read().unwrap();
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Get prefetch candidates (predicted but not cached)
    pub fn get_prefetch_candidates(&self, max_candidates: usize) -> Vec<u64> {
        let predictions = self.predict_next_hashes(max_candidates * 2);
        let cache = self.prediction_cache.read().unwrap();

        predictions
            .into_iter()
            .filter(|hash| !cache.contains_key(hash))
            .take(max_candidates)
            .collect()
    }

    /// Clear prediction cache
    pub fn clear_cache(&self) {
        self.prediction_cache.write().unwrap().clear();
        *self.cache_hits.write().unwrap() = 0;
        *self.cache_misses.write().unwrap() = 0;
    }

    /// Clear learned patterns
    pub fn clear_patterns(&self) {
        self.transitions.write().unwrap().clear();
        *self.last_query.write().unwrap() = None;
        self.query_history.write().unwrap().clear();
    }

    /// Get statistics
    pub fn stats(&self) -> PredictorStats {
        let transitions = self.transitions.read().unwrap();
        let cache = self.prediction_cache.read().unwrap();
        let history = self.query_history.read().unwrap();

        let total_transitions: usize = transitions.values().map(|m| m.len()).sum();

        PredictorStats {
            unique_queries: transitions.len(),
            total_transitions,
            cache_size: cache.len(),
            history_length: history.len(),
        }
    }
}

/// N-gram based query predictor (higher-order context)
pub struct NGramPredictor {
    ngrams: Arc<RwLock<HashMap<Vec<u64>, HashMap<u64, u32>>>>,
    history: Arc<RwLock<Vec<u64>>>,
    n: usize,
}

impl NGramPredictor {
    /// Create new N-gram predictor
    pub fn new(n: usize) -> Self {
        Self {
            ngrams: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            n: n.max(1),
        }
    }

    /// Observe a query
    pub fn observe(&self, query: &str) {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        let query_hash = hasher.finish();

        let mut history = self.history.write().unwrap();

        // Learn from current context
        if history.len() >= self.n - 1 {
            let context: Vec<u64> = history
                .iter()
                .rev()
                .take(self.n - 1)
                .rev()
                .cloned()
                .collect();

            let mut ngrams = self.ngrams.write().unwrap();
            *ngrams
                .entry(context)
                .or_default()
                .entry(query_hash)
                .or_insert(0) += 1;
        }

        // Update history
        history.push(query_hash);
        if history.len() > self.n * 10 {
            history.remove(0);
        }
    }

    /// Predict next queries based on N-gram context
    pub fn predict(&self, top_k: usize) -> Vec<Prediction> {
        let history = self.history.read().unwrap();
        let ngrams = self.ngrams.read().unwrap();

        if history.len() >= self.n - 1 {
            let context: Vec<u64> = history
                .iter()
                .rev()
                .take(self.n - 1)
                .rev()
                .cloned()
                .collect();

            if let Some(nexts) = ngrams.get(&context) {
                let total: u32 = nexts.values().sum();
                let mut predictions: Vec<_> = nexts
                    .iter()
                    .map(|(&hash, &count)| Prediction {
                        hash,
                        count,
                        probability: count as f64 / total as f64,
                    })
                    .collect();
                predictions.sort_by(|a, b| b.count.cmp(&a.count));
                predictions.truncate(top_k);
                return predictions;
            }
        }

        Vec::new()
    }

    /// Get N value
    pub fn n(&self) -> usize {
        self.n
    }

    /// Clear learned patterns
    pub fn clear(&self) {
        self.ngrams.write().unwrap().clear();
        self.history.write().unwrap().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_creation() {
        let predictor = QueryPredictor::new(100, 50);
        let stats = predictor.stats();
        assert_eq!(stats.unique_queries, 0);
        assert_eq!(stats.cache_size, 0);
    }

    #[test]
    fn test_observe_and_predict() {
        let predictor = QueryPredictor::new(100, 50);

        // Train a sequence
        predictor.observe("query_a");
        predictor.observe("query_b");
        predictor.observe("query_a");
        predictor.observe("query_b");
        predictor.observe("query_a");
        predictor.observe("query_b");
        predictor.observe("query_a"); // Set last query to a

        // After observing a->b multiple times, predicting from 'a' should return 'b'
        let predictions = predictor.predict_next(5);
        assert!(!predictions.is_empty());

        // The hash of query_b should be predicted
        let b_hash = QueryPredictor::hash_query("query_b");
        assert!(predictions.iter().any(|p| p.hash == b_hash));
    }

    #[test]
    fn test_cache_operations() {
        let predictor = QueryPredictor::new(100, 10);

        predictor.cache_result("query1", b"result1");
        assert!(predictor.is_cached("query1"));
        assert!(!predictor.is_cached("query2"));

        let cached = predictor.get_cached("query1").unwrap();
        assert_eq!(cached, b"result1");

        assert!(predictor.get_cached("query2").is_none());
    }

    #[test]
    fn test_cache_hit_rate() {
        let predictor = QueryPredictor::new(100, 10);

        predictor.cache_result("query1", b"result1");

        // Hit
        predictor.get_cached("query1");
        // Miss
        predictor.get_cached("query2");
        predictor.get_cached("query3");

        let hit_rate = predictor.cache_hit_rate();
        assert!((hit_rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_cache_eviction() {
        let predictor = QueryPredictor::new(100, 3); // Small cache

        predictor.cache_result("q1", b"r1");
        predictor.cache_result("q2", b"r2");
        predictor.cache_result("q3", b"r3");

        assert_eq!(predictor.stats().cache_size, 3);

        // Adding a 4th should evict one
        predictor.cache_result("q4", b"r4");
        assert_eq!(predictor.stats().cache_size, 3);
    }

    #[test]
    fn test_prefetch_candidates() {
        let predictor = QueryPredictor::new(100, 50);

        // Train sequence
        predictor.observe("a");
        predictor.observe("b");
        predictor.observe("c");
        predictor.observe("a");
        predictor.observe("b");
        predictor.observe("a"); // Last query is 'a'

        // Cache 'b' so it shouldn't be a prefetch candidate
        let b_hash = QueryPredictor::hash_query("b");
        predictor.cache_result("b", b"result_b");

        let candidates = predictor.get_prefetch_candidates(5);
        // 'b' should not be in candidates since it's cached
        assert!(!candidates.contains(&b_hash));
    }

    #[test]
    fn test_ngram_predictor() {
        let predictor = NGramPredictor::new(3);

        // Train trigram pattern: a,b -> c
        predictor.observe("a");
        predictor.observe("b");
        predictor.observe("c");
        predictor.observe("a");
        predictor.observe("b");
        predictor.observe("c");
        predictor.observe("a");
        predictor.observe("b"); // Context is now [a, b]

        let predictions = predictor.predict(3);
        // Should predict 'c' after 'a','b'
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_clear_operations() {
        let predictor = QueryPredictor::new(100, 50);

        predictor.observe("a");
        predictor.observe("b");
        predictor.cache_result("a", b"data");

        predictor.clear_cache();
        assert!(!predictor.is_cached("a"));

        predictor.clear_patterns();
        assert_eq!(predictor.stats().unique_queries, 0);
    }
}
