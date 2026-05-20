//! Neurosymbolic Integration
//!
//! Hybrid system combining neural pattern matching with symbolic reasoning.

use super::neural::{NeuralLayer, PatternMatch};
use super::symbolic::{Binding, Fact, SymbolicReasoner};
use super::{NeurosymbolicError, NeurosymbolicResult};
use std::sync::{Arc, RwLock};

/// Query type for hybrid processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Use only neural pattern matching
    Neural,
    /// Use only symbolic reasoning
    Symbolic,
    /// Combine neural and symbolic results
    Hybrid,
}

/// Result from hybrid query
#[derive(Debug, Clone)]
pub struct HybridResult {
    /// Neural matches (if any)
    pub neural_matches: Vec<PatternMatch>,
    /// Symbolic matches (if any)
    pub symbolic_matches: Vec<(Fact, Binding)>,
    /// Combined score (0.0 - 1.0)
    pub confidence: f32,
    /// Query type used
    pub query_type: QueryType,
}

impl HybridResult {
    /// Check if any results were found
    pub fn has_results(&self) -> bool {
        !self.neural_matches.is_empty() || !self.symbolic_matches.is_empty()
    }

    /// Get total number of results
    pub fn total_count(&self) -> usize {
        self.neural_matches.len() + self.symbolic_matches.len()
    }
}

/// Neurosymbolic Database
///
/// Combines neural pattern matching with symbolic reasoning.
pub struct NeurosymbolicDB {
    /// Neural layer for pattern matching
    neural: Arc<RwLock<NeuralLayer>>,
    /// Symbolic layer for reasoning
    symbolic: Arc<RwLock<SymbolicReasoner>>,
    /// Integration settings
    config: IntegrationConfig,
}

/// Configuration for integration
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Minimum neural similarity threshold
    pub neural_threshold: f32,
    /// Maximum neural results to consider
    pub neural_top_k: usize,
    /// Weight for neural results in hybrid scoring
    pub neural_weight: f32,
    /// Weight for symbolic results in hybrid scoring
    pub symbolic_weight: f32,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            neural_threshold: 0.5,
            neural_top_k: 10,
            neural_weight: 0.5,
            symbolic_weight: 0.5,
        }
    }
}

impl NeurosymbolicDB {
    /// Create new neurosymbolic database
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            neural: Arc::new(RwLock::new(NeuralLayer::new(vector_dimension))),
            symbolic: Arc::new(RwLock::new(SymbolicReasoner::new())),
            config: IntegrationConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(vector_dimension: usize, config: IntegrationConfig) -> Self {
        Self {
            neural: Arc::new(RwLock::new(NeuralLayer::new(vector_dimension))),
            symbolic: Arc::new(RwLock::new(SymbolicReasoner::new())),
            config,
        }
    }

    /// Get vector dimension
    pub fn dimension(&self) -> usize {
        self.neural.read().unwrap().dimension()
    }

    // ========================================================================
    // Neural Operations
    // ========================================================================

    /// Add a neural pattern
    pub fn add_pattern(&self, name: &str, embedding: &[f32]) -> NeurosymbolicResult<usize> {
        self.neural.read().unwrap().add_pattern(name, embedding)
    }

    /// Add pattern from raw data (generates embedding)
    pub fn add_pattern_from_data(&self, name: &str, data: &[u8]) -> NeurosymbolicResult<usize> {
        let neural = self.neural.read().unwrap();
        let embedding = neural.generate_embedding(data);
        neural.add_pattern(name, &embedding)
    }

    /// Match patterns
    pub fn match_patterns(
        &self,
        query: &[f32],
        top_k: usize,
    ) -> NeurosymbolicResult<Vec<PatternMatch>> {
        self.neural.read().unwrap().match_pattern(query, top_k)
    }

    /// Match patterns from raw data
    pub fn match_patterns_from_data(
        &self,
        data: &[u8],
        top_k: usize,
    ) -> NeurosymbolicResult<Vec<PatternMatch>> {
        let neural = self.neural.read().unwrap();
        let query = neural.generate_embedding(data);
        neural.match_pattern(&query, top_k)
    }

    // ========================================================================
    // Symbolic Operations
    // ========================================================================

    /// Add a rule
    pub fn add_rule(&self, name: &str, rule_str: &str) -> NeurosymbolicResult<()> {
        self.symbolic.read().unwrap().add_rule_str(name, rule_str)
    }

    /// Add a fact
    pub fn add_fact(&self, fact_str: &str) -> NeurosymbolicResult<()> {
        self.symbolic.read().unwrap().add_fact_str(fact_str)
    }

    /// Add a simple fact
    pub fn add_simple_fact(&self, predicate: &str, args: Vec<&str>) {
        self.symbolic
            .read()
            .unwrap()
            .add_simple_fact(predicate, args);
    }

    /// Run forward chaining
    pub fn infer(&self) -> usize {
        self.symbolic.read().unwrap().forward_chain()
    }

    /// Query symbolic knowledge
    pub fn query_symbolic(&self, query: &str) -> NeurosymbolicResult<Vec<(Fact, Binding)>> {
        self.symbolic.read().unwrap().query_str(query)
    }

    // ========================================================================
    // Hybrid Operations
    // ========================================================================

    /// Process a hybrid query
    ///
    /// For hybrid queries:
    /// 1. Runs neural pattern matching on the query embedding
    /// 2. Runs symbolic reasoning to derive facts
    /// 3. Combines results based on configuration
    pub fn query(
        &self,
        query_str: &str,
        query_embedding: Option<&[f32]>,
        query_type: QueryType,
    ) -> NeurosymbolicResult<HybridResult> {
        match query_type {
            QueryType::Neural => {
                let embedding = query_embedding.ok_or_else(|| {
                    NeurosymbolicError::QueryError("neural query requires embedding".to_string())
                })?;

                let neural = self.neural.read().unwrap();
                let matches = neural.match_pattern(embedding, self.config.neural_top_k)?;

                let confidence = matches.first().map(|m| m.similarity).unwrap_or(0.0);

                Ok(HybridResult {
                    neural_matches: matches,
                    symbolic_matches: Vec::new(),
                    confidence,
                    query_type,
                })
            }
            QueryType::Symbolic => {
                let symbolic = self.symbolic.read().unwrap();

                // First run forward chaining
                symbolic.forward_chain();

                // Then query
                let matches = symbolic.query_str(query_str)?;

                let confidence = if matches.is_empty() { 0.0 } else { 1.0 };

                Ok(HybridResult {
                    neural_matches: Vec::new(),
                    symbolic_matches: matches,
                    confidence,
                    query_type,
                })
            }
            QueryType::Hybrid => {
                // Run both
                let neural = self.neural.read().unwrap();
                let symbolic = self.symbolic.read().unwrap();

                // Neural matching
                let neural_matches = if let Some(emb) = query_embedding {
                    neural.match_pattern(emb, self.config.neural_top_k)?
                } else {
                    Vec::new()
                };

                // Symbolic reasoning
                symbolic.forward_chain();
                let symbolic_matches = symbolic.query_str(query_str)?;

                // Compute combined confidence
                let neural_conf = neural_matches.first().map(|m| m.similarity).unwrap_or(0.0);
                let symbolic_conf = if symbolic_matches.is_empty() {
                    0.0
                } else {
                    1.0
                };

                let confidence = self.config.neural_weight * neural_conf
                    + self.config.symbolic_weight * symbolic_conf;

                Ok(HybridResult {
                    neural_matches,
                    symbolic_matches,
                    confidence,
                    query_type,
                })
            }
        }
    }

    /// Neural-guided symbolic reasoning
    ///
    /// Uses neural similarity to prioritize symbolic rules.
    pub fn neural_guided_inference(
        &self,
        query_embedding: &[f32],
    ) -> NeurosymbolicResult<Vec<(Fact, f32)>> {
        let neural = self.neural.read().unwrap();
        let symbolic = self.symbolic.read().unwrap();

        // Get neural matches
        let matches = neural.match_pattern(query_embedding, self.config.neural_top_k)?;

        // Run symbolic inference
        symbolic.forward_chain();

        // Score symbolic facts by neural similarity
        let mut scored_facts = Vec::new();

        for pattern_match in &matches {
            // Query for facts related to this pattern
            let related = symbolic.query_str(&format!("{}(?X)", pattern_match.name));
            if let Ok(facts) = related {
                for (fact, _) in facts {
                    scored_facts.push((fact, pattern_match.similarity));
                }
            }
        }

        // Sort by score
        scored_facts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_facts)
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get statistics
    pub fn stats(&self) -> NeurosymbolicStats {
        let neural = self.neural.read().unwrap();
        let symbolic = self.symbolic.read().unwrap();

        NeurosymbolicStats {
            pattern_count: neural.len(),
            vector_dimension: neural.dimension(),
            fact_count: symbolic.fact_count(),
            rule_count: symbolic.rule_count(),
        }
    }
}

/// Statistics about the neurosymbolic database
#[derive(Debug, Clone)]
pub struct NeurosymbolicStats {
    /// Number of neural patterns
    pub pattern_count: usize,
    /// Vector dimension
    pub vector_dimension: usize,
    /// Number of symbolic facts
    pub fact_count: usize,
    /// Number of symbolic rules
    pub rule_count: usize,
}

impl Clone for NeurosymbolicDB {
    fn clone(&self) -> Self {
        Self {
            neural: Arc::new(RwLock::new(self.neural.read().unwrap().clone())),
            symbolic: Arc::new(RwLock::new(self.symbolic.read().unwrap().clone())),
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neurosymbolic_creation() {
        let db = NeurosymbolicDB::new(100);
        assert_eq!(db.dimension(), 100);

        let stats = db.stats();
        assert_eq!(stats.pattern_count, 0);
        assert_eq!(stats.fact_count, 0);
    }

    #[test]
    fn test_neural_query() {
        let db = NeurosymbolicDB::new(4);

        db.add_pattern("cat", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        db.add_pattern("dog", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let query = vec![0.9, 0.1, 0.0, 0.0]; // Similar to cat
        let result = db.query("", Some(&query), QueryType::Neural).unwrap();

        assert!(result.has_results());
        assert_eq!(result.neural_matches[0].name, "cat");
    }

    #[test]
    fn test_symbolic_query() {
        let db = NeurosymbolicDB::new(4);

        db.add_rule("r1", "human(X) => mortal(X)").unwrap();
        db.add_fact("human(socrates)").unwrap();

        let result = db
            .query("mortal(socrates)", None, QueryType::Symbolic)
            .unwrap();

        assert!(result.has_results());
        assert_eq!(result.symbolic_matches.len(), 1);
    }

    #[test]
    fn test_hybrid_query() {
        let db = NeurosymbolicDB::new(4);

        // Add neural patterns
        db.add_pattern("philosopher", &[1.0, 0.0, 0.0, 0.0])
            .unwrap();

        // Add symbolic knowledge
        db.add_rule("r1", "philosopher(X) => thinker(X)").unwrap();
        db.add_fact("philosopher(socrates)").unwrap();

        let query_embedding = vec![0.9, 0.1, 0.0, 0.0];
        let result = db
            .query(
                "thinker(socrates)",
                Some(&query_embedding),
                QueryType::Hybrid,
            )
            .unwrap();

        assert!(result.has_results());
        assert!(!result.neural_matches.is_empty());
        assert!(!result.symbolic_matches.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_add_pattern_from_data() {
        let db = NeurosymbolicDB::new(10);

        db.add_pattern_from_data("hello", b"hello world").unwrap();
        db.add_pattern_from_data("goodbye", b"goodbye world")
            .unwrap();

        let matches = db.match_patterns_from_data(b"hello", 2).unwrap();
        assert_eq!(matches.len(), 2);
        // "hello" should be most similar
        assert_eq!(matches[0].name, "hello");
    }

    #[test]
    fn test_simple_facts() {
        let db = NeurosymbolicDB::new(4);

        db.add_simple_fact("likes", vec!["alice", "bob"]);
        db.add_simple_fact("likes", vec!["bob", "charlie"]);

        let result = db.query_symbolic("likes(alice, bob)").unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_infer() {
        let db = NeurosymbolicDB::new(4);

        db.add_rule("r1", "a(X) => b(X)").unwrap();
        db.add_fact("a(foo)").unwrap();

        let new_facts = db.infer();
        assert_eq!(new_facts, 1);

        let result = db.query_symbolic("b(foo)").unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_hybrid_result() {
        let result = HybridResult {
            neural_matches: vec![PatternMatch {
                name: "test".to_string(),
                similarity: 0.9,
                index: 0,
            }],
            symbolic_matches: Vec::new(),
            confidence: 0.9,
            query_type: QueryType::Neural,
        };

        assert!(result.has_results());
        assert_eq!(result.total_count(), 1);
    }

    #[test]
    fn test_stats() {
        let db = NeurosymbolicDB::new(10);

        db.add_pattern("p1", &vec![0.0; 10]).unwrap();
        db.add_pattern("p2", &vec![0.0; 10]).unwrap();
        db.add_rule("r1", "a(X) => b(X)").unwrap();
        db.add_fact("a(foo)").unwrap();

        let stats = db.stats();
        assert_eq!(stats.pattern_count, 2);
        assert_eq!(stats.vector_dimension, 10);
        assert_eq!(stats.rule_count, 1);
        assert_eq!(stats.fact_count, 1);
    }
}
