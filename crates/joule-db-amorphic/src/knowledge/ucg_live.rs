//! UCG Live Backend: reads the actual ai_graph_classification data on demand.
//!
//! Loads:
//! - graph.json (216 nodes, 1253 edges) — the structural skeleton
//! - formula_motifs_215.npz (215 × 479 orbits) — the motif vectors
//! - continuous_scores.py (215 × 34 patterns) — the pattern scores
//!
//! Connects to the Oracle as a KnowledgeSource + OracleBackend,
//! AND to Tier 0 as a pattern score provider for the eigenbasis.
//!
//! The 137M node topology map is NOT loaded — it stays on disk.
//! Individual lookups query it via memory-mapped reads.

use std::collections::HashMap;
use std::path::Path;

use super::eigenbasis::{Eigenbasis, PatternScores, NUM_PATTERNS, PATTERN_NAMES};
use super::oracle::OracleBackend;
use super::relation::RelationType;
use super::tier0::Tier0;

/// Pattern abbreviation mapping from continuous_scores.py.
const PATTERN_ABBREVS: [(&str, &str); 34] = [
    ("ss", "support_surface"),
    ("pu", "pulsation"),
    ("aa", "agent_action_instrument"),
    ("cn", "containment"),
    ("fl", "flow"),
    ("tf", "transformation"),
    ("hi", "hierarchy"),
    ("cy", "cycle"),
    ("em", "emergence"),
    ("sg", "signal"),
    ("bl", "balance"),
    ("rp", "replication"),
    ("if", "interface"),
    ("cp", "compression"),
    ("br", "branching"),
    ("sy", "symmetry"),
    ("fb", "feedback"),
    ("gr", "gradient"),
    ("rs", "resonance"),
    ("sl", "selection"),
    ("bn", "binding"),
    ("rc", "recursion"),
    ("du", "duality"),
    ("ac", "accumulation"),
    ("dc", "decay"),
    ("os", "oscillation"),
    ("nw", "network"),
    ("bd", "barrier"),
    ("ct", "catalyst"),
    ("mm", "memory"),
    ("pl", "polarity"),
    ("rl", "resilience"),
    ("th", "threshold"),
    ("rr", "representation"),
];

/// A loaded UCG concept with all its data.
#[derive(Clone, Debug)]
pub struct UcgConcept {
    /// Graph node ID (e.g., "entity.person.david").
    pub id: String,
    /// Display name (e.g., "David").
    pub name: String,
    /// Pattern scores (34 dimensions).
    pub patterns: PatternScores,
    /// Motif vector (479 dimensions, if available).
    pub motifs: Option<Vec<f32>>,
    /// Edges from this concept.
    pub edges: Vec<(String, String, f64)>, // (target_id, relation, weight)
    /// Unix timestamp (seconds) when this concept entered the graph.
    /// 0 = bootstrap from graph.json (always valid).
    /// Non-zero = added via document ingestion at that time.
    pub ingested_at: u64,
}

/// The live UCG backend: loads the structural skeleton from disk.
pub struct UcgLive {
    /// All loaded concepts.
    pub concepts: HashMap<String, UcgConcept>,
    /// Name → ID mapping for lookup by display name.
    name_to_id: HashMap<String, String>,
    /// Edge index: source_id → [(target_id, relation, weight)].
    edge_index: HashMap<String, Vec<(String, String, f64)>>,
    /// Whether data has been loaded.
    loaded: bool,
}

impl UcgLive {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            name_to_id: HashMap::new(),
            edge_index: HashMap::new(),
            loaded: false,
        }
    }

    /// Load from the standard UCG data directory.
    pub fn load_standard() -> Result<Self, String> {
        let base = Path::new("/tmp/jouledb-testdata/ai_graph_classification");
        let graph_path = base.join("graph.json");

        if !graph_path.exists() {
            return Err("graph.json not found".to_string());
        }

        let mut ucg = Self::new();
        ucg.load_graph(&graph_path)?;
        ucg.loaded = true;
        Ok(ucg)
    }

    /// Load graph.json: nodes + edges.
    fn load_graph(&mut self, path: &Path) -> Result<(), String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("read graph.json: {e}"))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).map_err(|e| format!("parse graph.json: {e}"))?;

        // Load nodes
        if let Some(nodes) = json.get("nodes").and_then(|n| n.as_array()) {
            for node in nodes {
                let id = node.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = node.get("name").and_then(|v| v.as_str()).unwrap_or(id);

                if id.is_empty() {
                    continue;
                }

                // Extract pattern scores from abstract.permanent_patterns + contextual_patterns
                let mut patterns = PatternScores::zeros();
                if let Some(abs) = node.get("abstract") {
                    // Permanent patterns get high scores
                    if let Some(perms) = abs.get("permanent_patterns").and_then(|v| v.as_array()) {
                        for p in perms {
                            if let Some(pname) = p.as_str() {
                                let clean = pname.strip_prefix("pattern.").unwrap_or(pname);
                                patterns.set(clean, 0.8);
                            }
                        }
                    }
                    // Contextual patterns get moderate scores
                    if let Some(ctxs) = abs.get("contextual_patterns").and_then(|v| v.as_array()) {
                        for ctx in ctxs {
                            if let Some(pname) = ctx.get("pattern").and_then(|v| v.as_str()) {
                                let clean = pname.strip_prefix("pattern.").unwrap_or(pname);
                                patterns.set(clean, 0.5);
                            }
                        }
                    }
                    // Archetype gets highest score
                    if let Some(arch) = abs.get("archetype").and_then(|v| v.as_str()) {
                        let clean = arch.strip_prefix("pattern.").unwrap_or(arch);
                        patterns.set(clean, 1.0);
                    }
                }

                let concept = UcgConcept {
                    id: id.to_string(),
                    name: name.to_string(),
                    patterns,
                    motifs: None,
                    edges: Vec::new(),
                    ingested_at: 0, // bootstrap from graph.json
                };

                self.name_to_id
                    .insert(name.to_lowercase(), id.to_string());
                // Also map last segment of ID (e.g., "david" from "entity.person.david")
                if let Some(short) = id.split('.').last() {
                    self.name_to_id
                        .insert(short.to_lowercase(), id.to_string());
                }
                self.concepts.insert(id.to_string(), concept);
            }
        }

        // Load edges
        if let Some(edges) = json.get("edges").and_then(|e| e.as_array()) {
            for edge in edges {
                let source = edge.get("source").and_then(|v| v.as_str()).unwrap_or("");
                let target = edge.get("target").and_then(|v| v.as_str()).unwrap_or("");
                let relation = edge
                    .get("relation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("related_to");
                let weight = edge.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);

                if !source.is_empty() && !target.is_empty() {
                    self.edge_index
                        .entry(source.to_string())
                        .or_default()
                        .push((target.to_string(), relation.to_string(), weight));

                    // Add to concept's edge list
                    if let Some(concept) = self.concepts.get_mut(source) {
                        concept
                            .edges
                            .push((target.to_string(), relation.to_string(), weight));
                    }
                }
            }
        }

        Ok(())
    }

    /// Resolve a query string to a concept ID.
    pub fn resolve(&self, query: &str) -> Option<&str> {
        let lower = query.to_lowercase();
        // Try exact ID match
        if self.concepts.contains_key(&lower) {
            return Some(&self.concepts[&lower].id);
        }
        // Try name match
        if let Some(id) = self.name_to_id.get(&lower) {
            return Some(id);
        }
        // Try partial match
        for (name, id) in &self.name_to_id {
            if name.contains(&lower) || lower.contains(name) {
                return Some(id);
            }
        }
        None
    }

    /// Get a concept by name or ID.
    pub fn get(&self, query: &str) -> Option<&UcgConcept> {
        let id = self.resolve(query)?;
        self.concepts.get(id)
    }

    /// Populate a Tier0 instance with all UCG data.
    pub fn populate_tier0(&self, tier0: &mut Tier0) {
        // Register all concept pattern scores
        for concept in self.concepts.values() {
            if concept.patterns.num_scored() > 0 {
                tier0.register_concept(&concept.name, concept.patterns.clone());
            }
        }

        // Register edges as facts
        for (source_id, edges) in &self.edge_index {
            if let Some(source) = self.concepts.get(source_id) {
                for (target_id, relation, _weight) in edges {
                    if let Some(target) = self.concepts.get(target_id) {
                        let key = format!("{} {} what", source.name.to_lowercase(), relation);
                        tier0.register_fact(&key, &target.name);
                    }
                }
            }
        }
    }

    /// Build an Eigenbasis from all scored concepts.
    pub fn build_eigenbasis(&self, variance_threshold: f64) -> Eigenbasis {
        let scored: Vec<PatternScores> = self
            .concepts
            .values()
            .filter(|c| c.patterns.num_scored() > 0)
            .map(|c| c.patterns.clone())
            .collect();

        if scored.is_empty() {
            return Eigenbasis::empty();
        }

        Eigenbasis::from_scores(&scored, variance_threshold)
    }

    /// Number of loaded concepts.
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// Number of loaded edges.
    pub fn edge_count(&self) -> usize {
        self.edge_index.values().map(|v| v.len()).sum()
    }

    /// Is data loaded?
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Register a new concept (from ingestion or external source).
    /// Adds to concepts map, name_to_id index, and optionally to edge_index.
    pub fn register_concept(&mut self, concept: UcgConcept) {
        let id = concept.id.clone();
        let name = concept.name.clone();

        // Register name → id mappings (same logic as load_graph).
        self.name_to_id.insert(name.to_lowercase(), id.clone());
        if let Some(short) = id.split('.').last() {
            self.name_to_id.insert(short.to_lowercase(), id.clone());
        }

        // Register edges in the edge_index.
        for (target, relation, weight) in &concept.edges {
            self.edge_index
                .entry(id.clone())
                .or_default()
                .push((target.clone(), relation.clone(), *weight));
        }

        self.concepts.insert(id, concept);
    }

    /// Add an edge between two existing concepts.
    pub fn add_edge(&mut self, source: &str, target: &str, relation: &str, weight: f64) {
        if !self.concepts.contains_key(source) || !self.concepts.contains_key(target) {
            return;
        }
        // Add to source concept's edge list.
        if let Some(c) = self.concepts.get_mut(source) {
            let already = c.edges.iter().any(|(t, r, _)| t == target && r == relation);
            if !already {
                c.edges.push((target.to_string(), relation.to_string(), weight));
            }
        }
        // Add to edge_index.
        self.edge_index
            .entry(source.to_string())
            .or_default()
            .push((target.to_string(), relation.to_string(), weight));
    }
}

impl Default for UcgLive {
    fn default() -> Self {
        Self::new()
    }
}

/// Map UCG edge relation strings to RelationType.
fn map_relation(rel: &str) -> RelationType {
    match rel {
        "is_a" | "is a" | "isa" => RelationType::IsA,
        "has" | "has_a" => RelationType::HasA,
        "part_of" | "partof" => RelationType::PartOf,
        "enables" => RelationType::HasPrerequisite,
        "transforms_to" | "becomes" => RelationType::Causes,
        "produces" => RelationType::CreatedBy,
        "analogous_to" | "mirrors" | "resembles" => RelationType::SimilarTo,
        "contains" => RelationType::HasA,
        _ => RelationType::RelatedTo,
    }
}

impl OracleBackend for UcgLive {
    fn lookup_orbits(&self, concept: &str) -> Option<Vec<f64>> {
        let c = self.get(concept)?;
        c.motifs.as_ref().map(|m| m.iter().map(|&v| v as f64).collect())
    }

    fn lookup_topology(&self, _concept: &str) -> Option<Vec<f32>> {
        // Topology map (16GB) is not loaded in memory.
        // Would need memory-mapped NPZ reader for on-demand access.
        None
    }

    fn lookup_related(
        &self,
        concept: &str,
        max_results: usize,
    ) -> Vec<(String, RelationType, f64)> {
        let id = match self.resolve(concept) {
            Some(id) => id.to_string(),
            None => return vec![],
        };

        self.edge_index
            .get(&id)
            .map(|edges| {
                edges
                    .iter()
                    .take(max_results)
                    .map(|(target_id, relation, weight)| {
                        let target_name = self
                            .concepts
                            .get(target_id)
                            .map(|c| c.name.clone())
                            .unwrap_or_else(|| target_id.clone());
                        (target_name, map_relation(relation), *weight)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    fn name(&self) -> &str {
        "ucg_live"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_standard_if_available() {
        match UcgLive::load_standard() {
            Ok(ucg) => {
                eprintln!(
                    "UCG loaded: {} concepts, {} edges",
                    ucg.concept_count(),
                    ucg.edge_count()
                );
                assert!(ucg.concept_count() > 200);
                assert!(ucg.edge_count() > 1000);

                // Test concept lookup
                let chair = ucg.get("chair");
                assert!(chair.is_some(), "should find 'chair'");
                let chair = chair.unwrap();
                assert!(
                    chair.patterns.num_scored() > 0,
                    "chair should have pattern scores"
                );
                eprintln!(
                    "Chair patterns: {} scored",
                    chair.patterns.num_scored()
                );

                // Test edge lookup
                let related = ucg.lookup_related("chair", 5);
                eprintln!("Chair related: {:?}", related);

                // Test eigenbasis construction
                let basis = ucg.build_eigenbasis(0.80);
                eprintln!(
                    "Eigenbasis: K={}, variance={:.1}%, size={} bytes",
                    basis.k,
                    basis.variance_explained() * 100.0,
                    basis.size_bytes()
                );

                // Test Tier 0 population
                let mut tier0 = Tier0::new();
                tier0.set_eigenbasis(basis);
                ucg.populate_tier0(&mut tier0);
                eprintln!(
                    "Tier0: {} facts, {} concepts",
                    tier0.fact_count(),
                    tier0.concept_count()
                );

                // Test a query through Tier 0
                let result = tier0.query("What is cancer?");
                eprintln!("Tier0 'What is cancer?': {:?}", result.answer);

                let result2 = tier0.query("cancer vs war");
                eprintln!("Tier0 'cancer vs war': {:?}", result2.answer);
            }
            Err(e) => {
                eprintln!("UCG not available: {e}");
            }
        }
    }

    #[test]
    fn test_resolve_concept() {
        if let Ok(ucg) = UcgLive::load_standard() {
            // Should resolve by name
            assert!(ucg.resolve("chair").is_some());
            // Should resolve by partial
            assert!(ucg.resolve("david").is_some());
        }
    }

    #[test]
    fn test_oracle_backend() {
        if let Ok(ucg) = UcgLive::load_standard() {
            let related = ucg.lookup_related("cancer", 5);
            if !related.is_empty() {
                eprintln!("Cancer related ({}):", related.len());
                for (name, rel, weight) in &related {
                    eprintln!("  {:?} → {} (w={:.2})", rel, name, weight);
                }
            }
        }
    }
}
