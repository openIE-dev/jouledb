//! The Oracle: on-demand structural query interface to external knowledge.
//!
//! The brain is small (kilobytes). The library is large (gigabytes).
//! The Oracle is the librarian — it knows how to find the right answer
//! without carrying the library on its back.
//!
//! Design:
//! - The knowledge core holds the eigenbasis (2KB), relation codebook,
//!   top-weighted orbits, and recently-used concepts (LRU cache).
//! - When the core can't answer a query, the Oracle looks it up from
//!   an external reference (UCG dataset, ConceptNet, internet).
//! - The answer is encoded into the core via Update + Merge.
//! - The core grows only by what it actually uses.
//!
//! This is NOT a bulk loader. This is a query interface.
//!
//! ```text
//! Query → Core (cache hit?) → yes → answer
//!                            → no  → Oracle → Reference → encode → Core → answer
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use crate::BinaryHV;
use super::concept::ConceptEncoder;
use super::core::KnowledgeCore;
use super::relation::RelationType;
use super::triple::Triple;

/// A structural lookup result from the Oracle.
#[derive(Clone, Debug)]
pub struct OracleResult {
    /// The concept queried.
    pub query: String,
    /// Orbit vector (479 dimensions) if found in motif reference.
    pub orbits: Option<Vec<f64>>,
    /// Topology embedding (464 dimensions) if found in topology reference.
    pub topology: Option<Vec<f32>>,
    /// Related concepts discovered during lookup.
    pub related: Vec<(String, RelationType, f64)>, // (concept, relation, weight)
    /// Whether this was a cache hit.
    pub cached: bool,
    /// Source of the result.
    pub source: OracleSource,
}

/// Where the Oracle found its answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OracleSource {
    /// Found in the local LRU cache (zero cost).
    Cache,
    /// Looked up from the motif reference (local file).
    MotifReference,
    /// Looked up from the topology map (local file).
    TopologyReference,
    /// Looked up from a graph file (local).
    GraphReference,
    /// Looked up from the internet (highest cost).
    Internet,
    /// Not found anywhere.
    NotFound,
}

/// Trait for backends that can answer structural queries.
/// Implement this for different reference stores (NPZ files, databases, APIs).
pub trait OracleBackend: Send + Sync {
    /// Look up a concept's orbit vector (479-dim).
    fn lookup_orbits(&self, concept: &str) -> Option<Vec<f64>>;

    /// Look up a concept's topology embedding (464-dim).
    fn lookup_topology(&self, concept: &str) -> Option<Vec<f32>>;

    /// Find concepts related to the query.
    fn lookup_related(&self, concept: &str, max_results: usize)
        -> Vec<(String, RelationType, f64)>;

    /// Name of this backend.
    fn name(&self) -> &str;
}

/// The Oracle: on-demand structural knowledge lookup with LRU caching.
pub struct Oracle {
    /// LRU cache of recent lookups. Key = concept label.
    cache: HashMap<String, OracleResult>,
    /// Maximum cache size.
    cache_capacity: usize,
    /// Cache access order (most recent last) for LRU eviction.
    access_order: Vec<String>,
    /// Registered backends (tried in order).
    backends: Vec<Box<dyn OracleBackend>>,
    /// Statistics.
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub lookups: u64,
}

impl Oracle {
    pub fn new(cache_capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            cache_capacity,
            access_order: Vec::new(),
            backends: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            lookups: 0,
        }
    }

    /// Register a backend. Backends are tried in registration order.
    pub fn register_backend(&mut self, backend: Box<dyn OracleBackend>) {
        self.backends.push(backend);
    }

    /// Query the Oracle for a concept.
    /// Checks cache first, then backends in order.
    pub fn query(&mut self, concept: &str) -> OracleResult {
        self.lookups += 1;
        let normalized = concept.to_lowercase().replace(' ', "_");

        // Cache check
        if self.cache.contains_key(&normalized) {
            self.cache_hits += 1;
            self.touch_cache(&normalized);
            let mut result = self.cache.get(&normalized).unwrap().clone();
            result.cached = true;
            result.source = OracleSource::Cache;
            return result;
        }

        self.cache_misses += 1;

        // Try each backend
        for backend in &self.backends {
            let orbits = backend.lookup_orbits(&normalized);
            let topology = backend.lookup_topology(&normalized);
            let related = backend.lookup_related(&normalized, 10);

            if orbits.is_some() || topology.is_some() || !related.is_empty() {
                let result = OracleResult {
                    query: normalized.clone(),
                    orbits,
                    topology,
                    related,
                    cached: false,
                    source: OracleSource::MotifReference, // Simplified; real impl checks which backend answered
                };

                self.insert_cache(normalized, result.clone());
                return result;
            }
        }

        // Not found
        OracleResult {
            query: normalized,
            orbits: None,
            topology: None,
            related: Vec::new(),
            cached: false,
            source: OracleSource::NotFound,
        }
    }

    /// Query and automatically ingest results into the knowledge core.
    /// Returns the Oracle result and number of triples added.
    pub fn query_and_ingest(
        &mut self,
        concept: &str,
        core: &mut KnowledgeCore,
    ) -> (OracleResult, usize) {
        let result = self.query(concept);
        let mut added = 0;

        // Convert related concepts into triples and ingest
        for (related_concept, relation, weight) in &result.related {
            let triple = Triple::new(concept, *relation, related_concept)
                .with_weight(*weight);
            core.ingest_triple(&triple);
            added += 1;
        }

        // If we have orbit data, encode it as a structural property
        if let Some(ref orbits) = result.orbits {
            let norm: f64 = orbits.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                // Store the structural signature as a HasProperty triple
                let sig = format!("structural_norm_{:.2}", norm);
                let triple = Triple::new(concept, RelationType::HasProperty, &sig);
                core.ingest_triple(&triple);
                added += 1;
            }
        }

        (result, added)
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.lookups as f64
    }

    /// Current cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    // LRU cache internals

    fn touch_cache(&mut self, key: &str) {
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
        }
        self.access_order.push(key.to_string());
    }

    fn insert_cache(&mut self, key: String, value: OracleResult) {
        // Evict LRU if at capacity
        while self.cache.len() >= self.cache_capacity && !self.access_order.is_empty() {
            let evicted = self.access_order.remove(0);
            self.cache.remove(&evicted);
        }

        self.access_order.push(key.clone());
        self.cache.insert(key, value);
    }
}

impl Default for Oracle {
    fn default() -> Self {
        Self::new(10_000) // Default: cache 10K concepts
    }
}

/// In-memory backend for testing: holds concepts as HashMaps.
pub struct InMemoryBackend {
    name: String,
    orbits: HashMap<String, Vec<f64>>,
    topology: HashMap<String, Vec<f32>>,
    relations: HashMap<String, Vec<(String, RelationType, f64)>>,
}

impl InMemoryBackend {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            orbits: HashMap::new(),
            topology: HashMap::new(),
            relations: HashMap::new(),
        }
    }

    pub fn add_concept(
        &mut self,
        label: &str,
        orbits: Option<Vec<f64>>,
        topology: Option<Vec<f32>>,
        related: Vec<(String, RelationType, f64)>,
    ) {
        let key = label.to_lowercase().replace(' ', "_");
        if let Some(o) = orbits {
            self.orbits.insert(key.clone(), o);
        }
        if let Some(t) = topology {
            self.topology.insert(key.clone(), t);
        }
        if !related.is_empty() {
            self.relations.insert(key, related);
        }
    }
}

impl OracleBackend for InMemoryBackend {
    fn lookup_orbits(&self, concept: &str) -> Option<Vec<f64>> {
        self.orbits.get(concept).cloned()
    }

    fn lookup_topology(&self, concept: &str) -> Option<Vec<f32>> {
        self.topology.get(concept).cloned()
    }

    fn lookup_related(
        &self,
        concept: &str,
        max_results: usize,
    ) -> Vec<(String, RelationType, f64)> {
        self.relations
            .get(concept)
            .map(|r| r.iter().take(max_results).cloned().collect())
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ingest::ConceptNetParser;

    fn build_test_backend() -> InMemoryBackend {
        let mut backend = InMemoryBackend::new("test_ucg");

        backend.add_concept(
            "whale",
            Some(vec![0.1; 479]), // Mock orbit vector
            Some(vec![0.2; 464]), // Mock topology embedding
            vec![
                ("mammal".to_string(), RelationType::IsA, 0.95),
                ("ocean".to_string(), RelationType::AtLocation, 0.8),
                ("swim".to_string(), RelationType::CapableOf, 0.9),
            ],
        );

        backend.add_concept(
            "dolphin",
            Some(vec![0.15; 479]),
            Some(vec![0.25; 464]),
            vec![
                ("mammal".to_string(), RelationType::IsA, 0.95),
                ("ocean".to_string(), RelationType::AtLocation, 0.85),
                ("echolocation".to_string(), RelationType::CapableOf, 0.7),
            ],
        );

        backend.add_concept(
            "quantum_computer",
            Some(vec![0.05; 479]),
            None,
            vec![
                ("computer".to_string(), RelationType::IsA, 0.9),
                ("superposition".to_string(), RelationType::UsedFor, 0.8),
            ],
        );

        backend
    }

    #[test]
    fn test_oracle_cache_miss_then_hit() {
        let mut oracle = Oracle::new(100);
        oracle.register_backend(Box::new(build_test_backend()));

        // First query: cache miss
        let r1 = oracle.query("whale");
        assert!(!r1.cached);
        assert!(r1.orbits.is_some());
        assert_eq!(r1.related.len(), 3);
        assert_eq!(oracle.cache_misses, 1);

        // Second query: cache hit
        let r2 = oracle.query("whale");
        assert!(r2.cached);
        assert_eq!(r2.source, OracleSource::Cache);
        assert_eq!(oracle.cache_hits, 1);
    }

    #[test]
    fn test_oracle_not_found() {
        let mut oracle = Oracle::new(100);
        oracle.register_backend(Box::new(build_test_backend()));

        let result = oracle.query("unicorn");
        assert_eq!(result.source, OracleSource::NotFound);
        assert!(result.orbits.is_none());
        assert!(result.related.is_empty());
    }

    #[test]
    fn test_oracle_lru_eviction() {
        let mut oracle = Oracle::new(2); // Tiny cache
        oracle.register_backend(Box::new(build_test_backend()));

        oracle.query("whale");
        oracle.query("dolphin");
        assert_eq!(oracle.cache_size(), 2);

        // This should evict "whale" (LRU)
        oracle.query("quantum_computer");
        assert_eq!(oracle.cache_size(), 2);

        // "whale" should be a miss now (evicted)
        let whale = oracle.query("whale");
        assert!(!whale.cached);
    }

    #[test]
    fn test_query_and_ingest() {
        let mut oracle = Oracle::new(100);
        oracle.register_backend(Box::new(build_test_backend()));

        let mut core = KnowledgeCore::new();
        core.ingest_batch(&ConceptNetParser::bootstrap_core());
        let triples_before = core.triple_count;

        let (result, added) = oracle.query_and_ingest("whale", &mut core);
        assert!(added > 0);
        assert!(core.triple_count > triples_before);

        // "whale" should now be queryable in the core
        let whale = core.query_concept("whale");
        assert!(whale.is_some(), "whale should be in core after oracle ingest");
    }

    #[test]
    fn test_hit_rate() {
        let mut oracle = Oracle::new(100);
        oracle.register_backend(Box::new(build_test_backend()));

        oracle.query("whale");
        oracle.query("whale"); // hit
        oracle.query("dolphin");
        oracle.query("dolphin"); // hit
        oracle.query("whale"); // hit

        assert!((oracle.hit_rate() - 0.6).abs() < 0.01); // 3 hits / 5 lookups
    }

    #[test]
    fn test_full_pipeline_oracle_expands_then_generates() {
        use super::super::generate::Generator;

        let mut oracle = Oracle::new(100);
        oracle.register_backend(Box::new(build_test_backend()));

        let mut core = KnowledgeCore::new();
        core.ingest_batch(&ConceptNetParser::bootstrap_core());

        // Oracle expands the core with whale knowledge
        oracle.query_and_ingest("whale", &mut core);
        oracle.query_and_ingest("dolphin", &mut core);

        // Now the core knows about whales and dolphins
        let relatedness = core.relatedness("whale", "dolphin");
        assert!(
            relatedness > 0.0,
            "whale and dolphin should be related after oracle expansion: {relatedness}"
        );
    }
}
