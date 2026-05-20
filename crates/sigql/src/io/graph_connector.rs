//! Graph Connector — bridges SigQL to knowledge graph backends.
//!
//! Subsumes SigSPARQL (WU Vienna, 2025) by connecting frequency-domain signal
//! queries to semantic knowledge graphs. Where SigSPARQL bolts signals onto
//! RDF/SPARQL, this integrates signals as first-class queryable entities
//! alongside graph topology.
//!
//! # Architecture
//!
//! ```text
//! SigQL Query
//!   FROM graph.traverse('Building-A', 'HAS_SENSOR', depth: 2) AS sensors
//!   LET vibration = sensors.signal('accel.z')
//!   TRANSFORM bandpass(1Hz, 50Hz)
//!   CORRELATE cross_correlation(vibration, lag: 500ms)
//!   AGGREGATE { coherence: mean }
//!       │
//!       ├──→ GraphConnector.traverse() → node IDs
//!       │        │
//!       │        └──→ for each node: load signal from SignalStorageConnector
//!       │
//!       └──→ SigQL runtime executes transforms + aggregates
//! ```
//!
//! # Usage
//!
//! The connector is backend-agnostic via the `GraphBackend` trait.
//! JouleDB's `WorkGraph` implements this, but any graph store can be plugged in.

use smol_str::SmolStr;
use std::collections::HashMap;

/// A node in the knowledge graph with associated signal sources.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node identifier
    pub id: SmolStr,
    /// Node label/type (e.g., "Sensor", "Building", "Patient")
    pub label: SmolStr,
    /// Properties as key-value pairs
    pub properties: HashMap<String, String>,
    /// Signal source names attached to this node
    pub signal_sources: Vec<SmolStr>,
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID
    pub from: SmolStr,
    /// Target node ID
    pub to: SmolStr,
    /// Edge type/label (e.g., "HAS_SENSOR", "MEASURED_BY", "CONNECTED_TO")
    pub edge_type: SmolStr,
    /// Edge properties
    pub properties: HashMap<String, String>,
}

/// Result of a graph traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    /// Nodes found during traversal
    pub nodes: Vec<GraphNode>,
    /// Edges traversed
    pub edges: Vec<GraphEdge>,
    /// Depth reached
    pub depth_reached: usize,
}

/// Backend-agnostic graph interface.
///
/// Implement this trait to connect SigQL to any graph database
/// (JouleDB WorkGraph, Neo4j, Neptune, etc.).
pub trait GraphBackend: Send + Sync {
    /// Traverse from a start node following edges of the given type.
    fn traverse(
        &self,
        start_node: &str,
        edge_type: Option<&str>,
        max_depth: usize,
    ) -> Result<TraversalResult, GraphError>;

    /// Find nodes matching a label pattern.
    fn find_nodes(&self, label: &str) -> Result<Vec<GraphNode>, GraphError>;

    /// Get a specific node by ID.
    fn get_node(&self, id: &str) -> Result<Option<GraphNode>, GraphError>;

    /// Get signal source names for a node.
    fn node_signals(&self, node_id: &str) -> Result<Vec<SmolStr>, GraphError>;
}

/// Graph connector errors.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    #[error("Traversal failed: {0}")]
    TraversalFailed(String),
    #[error("Backend error: {0}")]
    BackendError(String),
}

/// In-memory graph backend for testing and small datasets.
///
/// Stores nodes and edges in HashMaps. Sufficient for development
/// and small knowledge graphs; production use should connect to JouleDB WorkGraph.
#[derive(Debug, Default)]
pub struct InMemoryGraph {
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
}

impl InMemoryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.insert(node.id.to_string(), node);
    }

    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }
}

impl GraphBackend for InMemoryGraph {
    fn traverse(
        &self,
        start_node: &str,
        edge_type: Option<&str>,
        max_depth: usize,
    ) -> Result<TraversalResult, GraphError> {
        let mut visited = std::collections::HashSet::new();
        let mut result_nodes = Vec::new();
        let mut result_edges = Vec::new();
        let mut frontier = vec![start_node.to_string()];
        let mut depth = 0;

        while depth < max_depth && !frontier.is_empty() {
            let mut next_frontier = Vec::new();

            for node_id in &frontier {
                if visited.contains(node_id) {
                    continue;
                }
                visited.insert(node_id.clone());

                if let Some(node) = self.nodes.get(node_id) {
                    result_nodes.push(node.clone());
                }

                for edge in &self.edges {
                    if edge.from.as_str() == node_id {
                        let matches = edge_type
                            .map(|et| edge.edge_type.as_str() == et)
                            .unwrap_or(true);
                        if matches && !visited.contains(edge.to.as_str()) {
                            result_edges.push(edge.clone());
                            next_frontier.push(edge.to.to_string());
                        }
                    }
                }
            }

            frontier = next_frontier;
            depth += 1;
        }

        Ok(TraversalResult {
            nodes: result_nodes,
            edges: result_edges,
            depth_reached: depth,
        })
    }

    fn find_nodes(&self, label: &str) -> Result<Vec<GraphNode>, GraphError> {
        Ok(self
            .nodes
            .values()
            .filter(|n| n.label.as_str() == label)
            .cloned()
            .collect())
    }

    fn get_node(&self, id: &str) -> Result<Option<GraphNode>, GraphError> {
        Ok(self.nodes.get(id).cloned())
    }

    fn node_signals(&self, node_id: &str) -> Result<Vec<SmolStr>, GraphError> {
        Ok(self
            .nodes
            .get(node_id)
            .map(|n| n.signal_sources.clone())
            .unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_graph() -> InMemoryGraph {
        let mut g = InMemoryGraph::new();

        g.add_node(GraphNode {
            id: "building-a".into(),
            label: "Building".into(),
            properties: HashMap::new(),
            signal_sources: vec![],
        });
        g.add_node(GraphNode {
            id: "sensor-1".into(),
            label: "Sensor".into(),
            properties: [("type".into(), "accelerometer".into())].into(),
            signal_sources: vec!["accel.x".into(), "accel.y".into(), "accel.z".into()],
        });
        g.add_node(GraphNode {
            id: "sensor-2".into(),
            label: "Sensor".into(),
            properties: [("type".into(), "temperature".into())].into(),
            signal_sources: vec!["temp.c".into()],
        });

        g.add_edge(GraphEdge {
            from: "building-a".into(),
            to: "sensor-1".into(),
            edge_type: "HAS_SENSOR".into(),
            properties: HashMap::new(),
        });
        g.add_edge(GraphEdge {
            from: "building-a".into(),
            to: "sensor-2".into(),
            edge_type: "HAS_SENSOR".into(),
            properties: HashMap::new(),
        });

        g
    }

    #[test]
    fn test_traverse() {
        let g = build_test_graph();
        let result = g.traverse("building-a", Some("HAS_SENSOR"), 2).unwrap();

        assert_eq!(result.nodes.len(), 3); // building + 2 sensors
        assert_eq!(result.edges.len(), 2);
    }

    #[test]
    fn test_find_nodes() {
        let g = build_test_graph();
        let sensors = g.find_nodes("Sensor").unwrap();
        assert_eq!(sensors.len(), 2);
    }

    #[test]
    fn test_node_signals() {
        let g = build_test_graph();
        let signals = g.node_signals("sensor-1").unwrap();
        assert_eq!(signals.len(), 3);
        assert!(signals.contains(&SmolStr::new("accel.z")));
    }
}
