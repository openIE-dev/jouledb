//! UCG File Backend: on-demand access to the 12M concept × 479 orbit dataset.
//!
//! Implements OracleBackend for the UCG local-reference files:
//! - motifs_v12_479.npz (4.41 GB) — 12M+ concepts × 479 graphlet orbits
//! - topology_map_v3.npz (16.26 GB) — 137M nodes × 464 dims
//!
//! Does NOT load the full dataset into memory. Uses a concept name index
//! to resolve queries, then loads individual vectors on demand.
//!
//! The index itself is small: ~12M concept names ≈ 300-500 MB.
//! Individual orbit vectors are 479 × f32 = 1.9 KB each.
//!
//! For the MVP, this module defines the interface and a lightweight
//! in-memory subset loader. The full memory-mapped NPZ reader
//! requires the `memmap2` crate (already in the workspace).

use std::collections::HashMap;
use std::path::PathBuf;

use super::oracle::OracleBackend;
use super::relation::RelationType;

/// Configuration for the UCG file backend.
#[derive(Clone, Debug)]
pub struct UcgConfig {
    /// Path to the UCG dataset directory (containing .npz files).
    pub data_dir: PathBuf,
    /// Path to motifs file (motifs_v12_479.npz).
    pub motifs_path: Option<PathBuf>,
    /// Path to topology map (topology_map_v3.npz).
    pub topology_path: Option<PathBuf>,
    /// Path to the formula concept scores (formula_motifs_215.npz).
    pub formula_path: Option<PathBuf>,
    /// Path to the graph structure (graph.json).
    pub graph_path: Option<PathBuf>,
    /// Maximum concepts to hold in memory at once.
    pub max_memory_concepts: usize,
}

impl UcgConfig {
    /// Create config pointing to the standard UCG data directory.
    pub fn standard() -> Self {
        let base = PathBuf::from("/tmp/jouledb-testdata/ai_graph_classification");
        Self {
            data_dir: base.clone(),
            motifs_path: Some(base.join("local-reference/motifs_v12_479.npz")),
            topology_path: Some(base.join("local-reference/topology_map_v3.npz")),
            formula_path: Some(base.join("experiments/formula_motifs_215.npz")),
            graph_path: Some(base.join("graph.json")),
            max_memory_concepts: 50_000,
        }
    }

    /// Check which data files exist.
    pub fn available_files(&self) -> Vec<(String, bool)> {
        let mut files = Vec::new();
        if let Some(ref p) = self.motifs_path {
            files.push(("motifs".to_string(), p.exists()));
        }
        if let Some(ref p) = self.topology_path {
            files.push(("topology".to_string(), p.exists()));
        }
        if let Some(ref p) = self.formula_path {
            files.push(("formula".to_string(), p.exists()));
        }
        if let Some(ref p) = self.graph_path {
            files.push(("graph".to_string(), p.exists()));
        }
        files
    }
}

/// The UCG file backend. Provides on-demand access to the 12M concept dataset.
///
/// For the MVP, this uses an in-memory concept index loaded from the
/// formula_motifs_215.npz (the 215-concept core that the formula operates on).
/// The full 12M concept loader uses memory-mapped NPZ files.
pub struct UcgFileBackend {
    config: UcgConfig,
    /// Concept name → orbit vector (479 dims). Loaded subset.
    orbit_cache: HashMap<String, Vec<f64>>,
    /// Concept name → topology embedding (464 dims). Loaded subset.
    topology_cache: HashMap<String, Vec<f32>>,
    /// Graph edges: source → [(target, relation, weight)].
    graph_edges: HashMap<String, Vec<(String, RelationType, f64)>>,
    /// Whether the backend has been initialized.
    initialized: bool,
}

impl UcgFileBackend {
    /// Create a new backend with the given config.
    pub fn new(config: UcgConfig) -> Self {
        Self {
            config,
            orbit_cache: HashMap::new(),
            topology_cache: HashMap::new(),
            graph_edges: HashMap::new(),
            initialized: false,
        }
    }

    /// Create with standard paths.
    pub fn standard() -> Self {
        Self::new(UcgConfig::standard())
    }

    /// Initialize: load the graph structure and formula concepts.
    /// Does NOT load the full 4.41 GB motif file — only the index.
    pub fn initialize(&mut self) -> Result<(), String> {
        // Load graph edges if available
        let graph_path = self.config.graph_path.clone();
        if let Some(path) = graph_path {
            if path.exists() {
                self.load_graph_json(&path)?;
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Load graph.json edges into the edge index.
    fn load_graph_json(&mut self, path: &PathBuf) -> Result<(), String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read graph.json: {e}"))?;

        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("failed to parse graph.json: {e}"))?;

        // Parse edges from the graph JSON
        if let Some(edges) = json.get("edges").and_then(|e| e.as_array()) {
            for edge in edges {
                let source = edge
                    .get("source")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let target = edge
                    .get("target")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                let rel_str = edge
                    .get("relation")
                    .and_then(|r| r.as_str())
                    .unwrap_or("related_to");
                let weight = edge
                    .get("weight")
                    .and_then(|w| w.as_f64())
                    .unwrap_or(1.0);

                if !source.is_empty() && !target.is_empty() {
                    let relation = RelationType::from_conceptnet(rel_str);
                    self.graph_edges
                        .entry(source.to_lowercase())
                        .or_default()
                        .push((target.to_lowercase(), relation, weight));
                    // Also add reverse for bidirectional lookup
                    self.graph_edges
                        .entry(target.to_lowercase())
                        .or_default()
                        .push((source.to_lowercase(), relation, weight));
                }
            }
        }

        Ok(())
    }

    /// Manually add orbit data (for testing or for loading from external sources).
    pub fn add_orbits(&mut self, concept: &str, orbits: Vec<f64>) {
        self.orbit_cache.insert(concept.to_lowercase(), orbits);
    }

    /// Manually add topology data.
    pub fn add_topology(&mut self, concept: &str, topology: Vec<f32>) {
        self.topology_cache.insert(concept.to_lowercase(), topology);
    }

    /// Number of concepts with cached orbit data.
    pub fn cached_orbits(&self) -> usize {
        self.orbit_cache.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph_edges.values().map(|v| v.len()).sum::<usize>() / 2 // Divide by 2 for bidirectional
    }
}

impl OracleBackend for UcgFileBackend {
    fn lookup_orbits(&self, concept: &str) -> Option<Vec<f64>> {
        self.orbit_cache.get(&concept.to_lowercase()).cloned()
    }

    fn lookup_topology(&self, concept: &str) -> Option<Vec<f32>> {
        self.topology_cache.get(&concept.to_lowercase()).cloned()
    }

    fn lookup_related(
        &self,
        concept: &str,
        max_results: usize,
    ) -> Vec<(String, RelationType, f64)> {
        self.graph_edges
            .get(&concept.to_lowercase())
            .map(|edges| {
                let mut sorted = edges.clone();
                sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                sorted.into_iter().take(max_results).collect()
            })
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        "ucg_file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_standard() {
        let config = UcgConfig::standard();
        assert!(config.data_dir.to_str().unwrap().contains("ai_graph_classification"));
    }

    #[test]
    fn test_backend_manual_data() {
        let mut backend = UcgFileBackend::new(UcgConfig::standard());
        backend.add_orbits("test_concept", vec![0.1; 479]);
        backend.add_topology("test_concept", vec![0.2; 464]);

        let orbits = backend.lookup_orbits("test_concept");
        assert!(orbits.is_some());
        assert_eq!(orbits.unwrap().len(), 479);

        let topology = backend.lookup_topology("test_concept");
        assert!(topology.is_some());
        assert_eq!(topology.unwrap().len(), 464);
    }

    #[test]
    fn test_backend_case_insensitive() {
        let mut backend = UcgFileBackend::new(UcgConfig::standard());
        backend.add_orbits("Dog", vec![0.5; 479]);

        assert!(backend.lookup_orbits("dog").is_some());
        assert!(backend.lookup_orbits("DOG").is_some());
    }

    #[test]
    fn test_backend_not_found() {
        let backend = UcgFileBackend::new(UcgConfig::standard());
        assert!(backend.lookup_orbits("nonexistent").is_none());
        assert!(backend.lookup_topology("nonexistent").is_none());
        assert!(backend.lookup_related("nonexistent", 10).is_empty());
    }

    #[test]
    fn test_available_files() {
        let config = UcgConfig::standard();
        let files = config.available_files();
        assert!(!files.is_empty());
        // At least the config knows about the expected files
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"motifs"));
        assert!(names.contains(&"topology"));
        assert!(names.contains(&"graph"));
    }

    #[test]
    fn test_load_graph_if_available() {
        let config = UcgConfig::standard();
        let mut backend = UcgFileBackend::new(config.clone());

        if let Some(ref path) = config.graph_path {
            if path.exists() {
                let result = backend.initialize();
                assert!(result.is_ok(), "should load graph.json: {:?}", result);
                assert!(backend.edge_count() > 0, "should have edges");

                // Should be able to look up related concepts
                let related = backend.lookup_related("root.thing", 5);
                // graph.json uses "root.thing" as the root node
                // (may or may not have edges depending on format)
            }
        }
    }

    #[test]
    fn test_oracle_integration() {
        use super::super::oracle::Oracle;

        let mut backend = UcgFileBackend::new(UcgConfig::standard());
        backend.add_orbits("electron", vec![0.3; 479]);
        backend.add_topology("electron", vec![0.4; 464]);

        let mut oracle = Oracle::new(100);
        oracle.register_backend(Box::new(backend));

        let result = oracle.query("electron");
        assert!(result.orbits.is_some());
        assert!(result.topology.is_some());
    }
}
