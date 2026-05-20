//! Graph Database Features
//!
//! Provides graph storage and traversal:
//! - Nodes with labels and properties
//! - Edges with types and properties
//! - Graph traversal operations
//! - Path finding algorithms
//! - Advanced graph algorithms (PageRank, community detection, centrality)

mod algorithms;
mod algorithms_ext;
pub use algorithms::{CentralityMeasures, CommunityResult, PageRankResult};
pub use algorithms_ext::{
    DijkstraResult, KCoreResult, KShortestResult, LabelPropResult, LinkScore,
    SccResult, TriangleResult,
};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

/// Node ID type
pub type NodeId = u64;

/// Edge ID type
pub type EdgeId = u64;

/// Graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Node ID
    pub id: NodeId,
    /// Labels
    pub labels: Vec<String>,
    /// Properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl Node {
    /// Create new node
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            labels: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Add label
    pub fn with_label(mut self, label: &str) -> Self {
        self.labels.push(label.to_string());
        self
    }

    /// Add property
    pub fn with_property(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.to_string(), value.into());
        self
    }

    /// Check if has label
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l == label)
    }

    /// Get property
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }
}

/// Graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Edge ID
    pub id: EdgeId,
    /// Source node ID
    pub from: NodeId,
    /// Target node ID
    pub to: NodeId,
    /// Edge type/label
    pub edge_type: String,
    /// Properties
    pub properties: HashMap<String, serde_json::Value>,
}

impl Edge {
    /// Create new edge
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, edge_type: &str) -> Self {
        Self {
            id,
            from,
            to,
            edge_type: edge_type.to_string(),
            properties: HashMap::new(),
        }
    }

    /// Add property
    pub fn with_property(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.to_string(), value.into());
        self
    }

    /// Get property
    pub fn get_property(&self, key: &str) -> Option<&serde_json::Value> {
        self.properties.get(key)
    }
}

/// Traversal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}

/// Traversal step
#[derive(Debug, Clone)]
pub enum TraversalStep {
    /// Follow edges of specific type
    Edge(String, Direction),
    /// Filter by label
    HasLabel(String),
    /// Filter by property
    HasProperty(String, serde_json::Value),
    /// Limit results
    Limit(usize),
    /// Skip results
    Skip(usize),
}

/// Graph traversal builder
#[derive(Debug, Clone)]
pub struct Traversal {
    /// Starting nodes
    pub start_nodes: Vec<NodeId>,
    /// Traversal steps
    pub steps: Vec<TraversalStep>,
    /// Max depth
    pub max_depth: Option<usize>,
}

impl Traversal {
    /// Start traversal from nodes
    pub fn from_nodes(nodes: Vec<NodeId>) -> Self {
        Self {
            start_nodes: nodes,
            steps: Vec::new(),
            max_depth: None,
        }
    }

    /// Start traversal from single node
    pub fn from_node(node: NodeId) -> Self {
        Self::from_nodes(vec![node])
    }

    /// Follow outgoing edges
    pub fn out(mut self, edge_type: &str) -> Self {
        self.steps.push(TraversalStep::Edge(
            edge_type.to_string(),
            Direction::Outgoing,
        ));
        self
    }

    /// Follow incoming edges
    pub fn in_(mut self, edge_type: &str) -> Self {
        self.steps.push(TraversalStep::Edge(
            edge_type.to_string(),
            Direction::Incoming,
        ));
        self
    }

    /// Follow edges in both directions
    pub fn both(mut self, edge_type: &str) -> Self {
        self.steps
            .push(TraversalStep::Edge(edge_type.to_string(), Direction::Both));
        self
    }

    /// Filter by label
    pub fn has_label(mut self, label: &str) -> Self {
        self.steps.push(TraversalStep::HasLabel(label.to_string()));
        self
    }

    /// Filter by property
    pub fn has_property(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.steps
            .push(TraversalStep::HasProperty(key.to_string(), value.into()));
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.steps.push(TraversalStep::Limit(n));
        self
    }

    /// Set max depth
    pub fn depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }
}

/// Path result
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Nodes in path
    pub nodes: Vec<Node>,
    /// Edges in path
    pub edges: Vec<Edge>,
    /// Total weight (if weighted)
    pub weight: f64,
}

impl PathResult {
    /// Get path length (number of edges)
    pub fn length(&self) -> usize {
        self.edges.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Graph configuration
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Enable indexing on labels
    pub index_labels: bool,
    /// Enable indexing on edge types
    pub index_edge_types: bool,
    /// Maximum path depth for queries
    pub max_path_depth: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            index_labels: true,
            index_edge_types: true,
            max_path_depth: 10,
        }
    }
}

/// Graph store
pub struct GraphStore {
    config: GraphConfig,
    /// Nodes: id -> node
    nodes: Arc<RwLock<HashMap<NodeId, Node>>>,
    /// Edges: id -> edge
    edges: Arc<RwLock<HashMap<EdgeId, Edge>>>,
    /// Outgoing edges index: node_id -> edge_ids
    outgoing: Arc<RwLock<HashMap<NodeId, Vec<EdgeId>>>>,
    /// Incoming edges index: node_id -> edge_ids
    incoming: Arc<RwLock<HashMap<NodeId, Vec<EdgeId>>>>,
    /// Label index: label -> node_ids
    label_index: Arc<RwLock<HashMap<String, HashSet<NodeId>>>>,
    /// Edge type index: type -> edge_ids
    edge_type_index: Arc<RwLock<HashMap<String, HashSet<EdgeId>>>>,
    /// Next node ID
    next_node_id: Arc<RwLock<NodeId>>,
    /// Next edge ID
    next_edge_id: Arc<RwLock<EdgeId>>,
}

impl GraphStore {
    /// Create new graph store
    pub fn new(config: GraphConfig) -> Self {
        Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            edges: Arc::new(RwLock::new(HashMap::new())),
            outgoing: Arc::new(RwLock::new(HashMap::new())),
            incoming: Arc::new(RwLock::new(HashMap::new())),
            label_index: Arc::new(RwLock::new(HashMap::new())),
            edge_type_index: Arc::new(RwLock::new(HashMap::new())),
            next_node_id: Arc::new(RwLock::new(1)),
            next_edge_id: Arc::new(RwLock::new(1)),
        }
    }

    /// Create with default config
    pub fn with_defaults() -> Self {
        Self::new(GraphConfig::default())
    }

    /// Create a node
    pub fn create_node(&self) -> NodeId {
        let mut id = self.next_node_id.write().unwrap();
        let node_id = *id;
        *id += 1;

        let node = Node::new(node_id);
        self.nodes.write().unwrap().insert(node_id, node);
        self.outgoing.write().unwrap().insert(node_id, Vec::new());
        self.incoming.write().unwrap().insert(node_id, Vec::new());

        node_id
    }

    /// Create a node with labels and properties
    pub fn create_node_with(
        &self,
        labels: Vec<&str>,
        properties: HashMap<&str, serde_json::Value>,
    ) -> NodeId {
        let node_id = self.create_node();

        // Add labels
        for label in &labels {
            self.add_label(node_id, label);
        }

        // Add properties
        for (key, value) in properties {
            self.set_property(node_id, key, value);
        }

        node_id
    }

    /// Get node by ID
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.nodes.read().unwrap().get(&id).cloned()
    }

    /// Add label to node
    pub fn add_label(&self, node_id: NodeId, label: &str) {
        if let Some(node) = self.nodes.write().unwrap().get_mut(&node_id) {
            if !node.labels.contains(&label.to_string()) {
                node.labels.push(label.to_string());

                // Update index
                if self.config.index_labels {
                    self.label_index
                        .write()
                        .unwrap()
                        .entry(label.to_string())
                        .or_insert_with(HashSet::new)
                        .insert(node_id);
                }
            }
        }
    }

    /// Set node property
    pub fn set_property(&self, node_id: NodeId, key: &str, value: serde_json::Value) {
        if let Some(node) = self.nodes.write().unwrap().get_mut(&node_id) {
            node.properties.insert(key.to_string(), value);
        }
    }

    /// Delete node
    pub fn delete_node(&self, node_id: NodeId) -> bool {
        // First remove all connected edges
        let edge_ids: Vec<EdgeId> = {
            let outgoing = self.outgoing.read().unwrap();
            let incoming = self.incoming.read().unwrap();
            let mut edges = Vec::new();
            if let Some(out) = outgoing.get(&node_id) {
                edges.extend(out.iter().copied());
            }
            if let Some(inc) = incoming.get(&node_id) {
                edges.extend(inc.iter().copied());
            }
            edges
        };

        for edge_id in edge_ids {
            self.delete_edge(edge_id);
        }

        // Remove from label index
        if let Some(node) = self.nodes.read().unwrap().get(&node_id) {
            for label in &node.labels {
                if let Some(set) = self.label_index.write().unwrap().get_mut(label) {
                    set.remove(&node_id);
                }
            }
        }

        // Remove node
        self.nodes.write().unwrap().remove(&node_id);
        self.outgoing.write().unwrap().remove(&node_id);
        self.incoming.write().unwrap().remove(&node_id);

        true
    }

    /// Create an edge
    pub fn create_edge(&self, from: NodeId, to: NodeId, edge_type: &str) -> Option<EdgeId> {
        // Verify nodes exist
        let nodes = self.nodes.read().unwrap();
        if !nodes.contains_key(&from) || !nodes.contains_key(&to) {
            return None;
        }
        drop(nodes);

        let mut id = self.next_edge_id.write().unwrap();
        let edge_id = *id;
        *id += 1;

        let edge = Edge::new(edge_id, from, to, edge_type);
        self.edges.write().unwrap().insert(edge_id, edge);

        // Update indices
        self.outgoing
            .write()
            .unwrap()
            .entry(from)
            .or_insert_with(Vec::new)
            .push(edge_id);

        self.incoming
            .write()
            .unwrap()
            .entry(to)
            .or_insert_with(Vec::new)
            .push(edge_id);

        if self.config.index_edge_types {
            self.edge_type_index
                .write()
                .unwrap()
                .entry(edge_type.to_string())
                .or_insert_with(HashSet::new)
                .insert(edge_id);
        }

        Some(edge_id)
    }

    /// Get edge by ID
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.edges.read().unwrap().get(&id).cloned()
    }

    /// Set edge property
    pub fn set_edge_property(&self, edge_id: EdgeId, key: &str, value: serde_json::Value) {
        if let Some(edge) = self.edges.write().unwrap().get_mut(&edge_id) {
            edge.properties.insert(key.to_string(), value);
        }
    }

    /// Delete edge
    pub fn delete_edge(&self, edge_id: EdgeId) -> bool {
        let edge = match self.edges.write().unwrap().remove(&edge_id) {
            Some(e) => e,
            None => return false,
        };

        // Update indices
        if let Some(out) = self.outgoing.write().unwrap().get_mut(&edge.from) {
            out.retain(|&id| id != edge_id);
        }
        if let Some(inc) = self.incoming.write().unwrap().get_mut(&edge.to) {
            inc.retain(|&id| id != edge_id);
        }
        if let Some(set) = self
            .edge_type_index
            .write()
            .unwrap()
            .get_mut(&edge.edge_type)
        {
            set.remove(&edge_id);
        }

        true
    }

    /// Get nodes by label
    pub fn get_nodes_by_label(&self, label: &str) -> Vec<Node> {
        let node_ids: Vec<NodeId> = if self.config.index_labels {
            self.label_index
                .read()
                .unwrap()
                .get(label)
                .map(|set| set.iter().copied().collect())
                .unwrap_or_default()
        } else {
            self.nodes
                .read()
                .unwrap()
                .iter()
                .filter(|(_, node)| node.has_label(label))
                .map(|(id, _)| *id)
                .collect()
        };

        let nodes = self.nodes.read().unwrap();
        node_ids
            .iter()
            .filter_map(|id| nodes.get(id).cloned())
            .collect()
    }

    /// Get outgoing edges from node
    pub fn get_outgoing_edges(&self, node_id: NodeId) -> Vec<Edge> {
        let edge_ids = self
            .outgoing
            .read()
            .unwrap()
            .get(&node_id)
            .cloned()
            .unwrap_or_default();

        let edges = self.edges.read().unwrap();
        edge_ids
            .iter()
            .filter_map(|id| edges.get(id).cloned())
            .collect()
    }

    /// Get incoming edges to node
    pub fn get_incoming_edges(&self, node_id: NodeId) -> Vec<Edge> {
        let edge_ids = self
            .incoming
            .read()
            .unwrap()
            .get(&node_id)
            .cloned()
            .unwrap_or_default();

        let edges = self.edges.read().unwrap();
        edge_ids
            .iter()
            .filter_map(|id| edges.get(id).cloned())
            .collect()
    }

    /// Get neighbors (outgoing)
    pub fn get_neighbors(&self, node_id: NodeId, edge_type: Option<&str>) -> Vec<Node> {
        let edges = self.get_outgoing_edges(node_id);
        let nodes = self.nodes.read().unwrap();

        edges
            .iter()
            .filter(|e| edge_type.map(|t| e.edge_type == t).unwrap_or(true))
            .filter_map(|e| nodes.get(&e.to).cloned())
            .collect()
    }

    /// Execute traversal
    pub fn traverse(&self, traversal: &Traversal) -> Vec<Node> {
        let mut current_nodes: HashSet<NodeId> = traversal.start_nodes.iter().copied().collect();
        let max_depth = traversal.max_depth.unwrap_or(self.config.max_path_depth);
        let mut depth = 0;

        for step in &traversal.steps {
            if depth >= max_depth {
                break;
            }

            match step {
                TraversalStep::Edge(edge_type, direction) => {
                    let mut next_nodes = HashSet::new();
                    let edges = self.edges.read().unwrap();
                    let outgoing = self.outgoing.read().unwrap();
                    let incoming = self.incoming.read().unwrap();

                    for node_id in &current_nodes {
                        let edge_ids = match direction {
                            Direction::Outgoing => {
                                outgoing.get(node_id).cloned().unwrap_or_default()
                            }
                            Direction::Incoming => {
                                incoming.get(node_id).cloned().unwrap_or_default()
                            }
                            Direction::Both => {
                                let mut all = outgoing.get(node_id).cloned().unwrap_or_default();
                                all.extend(incoming.get(node_id).cloned().unwrap_or_default());
                                all
                            }
                        };

                        for edge_id in edge_ids {
                            if let Some(edge) = edges.get(&edge_id) {
                                if edge.edge_type == *edge_type {
                                    match direction {
                                        Direction::Outgoing => next_nodes.insert(edge.to),
                                        Direction::Incoming => next_nodes.insert(edge.from),
                                        Direction::Both => {
                                            if edge.from == *node_id {
                                                next_nodes.insert(edge.to);
                                            } else {
                                                next_nodes.insert(edge.from);
                                            }
                                            true
                                        }
                                    };
                                }
                            }
                        }
                    }

                    current_nodes = next_nodes;
                    depth += 1;
                }
                TraversalStep::HasLabel(label) => {
                    let nodes = self.nodes.read().unwrap();
                    current_nodes
                        .retain(|id| nodes.get(id).map(|n| n.has_label(label)).unwrap_or(false));
                }
                TraversalStep::HasProperty(key, value) => {
                    let nodes = self.nodes.read().unwrap();
                    current_nodes.retain(|id| {
                        nodes
                            .get(id)
                            .and_then(|n| n.get_property(key))
                            .map(|v| v == value)
                            .unwrap_or(false)
                    });
                }
                TraversalStep::Limit(n) => {
                    current_nodes = current_nodes.into_iter().take(*n).collect();
                }
                TraversalStep::Skip(n) => {
                    current_nodes = current_nodes.into_iter().skip(*n).collect();
                }
            }
        }

        let nodes = self.nodes.read().unwrap();
        current_nodes
            .iter()
            .filter_map(|id| nodes.get(id).cloned())
            .collect()
    }

    /// Find shortest path (BFS)
    pub fn shortest_path(
        &self,
        from: NodeId,
        to: NodeId,
        edge_type: Option<&str>,
    ) -> Option<PathResult> {
        if from == to {
            return self.get_node(from).map(|node| PathResult {
                nodes: vec![node],
                edges: Vec::new(),
                weight: 0.0,
            });
        }

        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut queue: VecDeque<(NodeId, Vec<NodeId>, Vec<EdgeId>)> = VecDeque::new();

        queue.push_back((from, vec![from], Vec::new()));
        visited.insert(from);

        while let Some((current, path_nodes, path_edges)) = queue.pop_front() {
            if path_nodes.len() > self.config.max_path_depth {
                continue;
            }

            let edges = self.get_outgoing_edges(current);
            for edge in edges {
                if edge_type.map(|t| edge.edge_type != t).unwrap_or(false) {
                    continue;
                }

                if edge.to == to {
                    // Found path
                    let mut nodes_vec: Vec<Node> = Vec::new();
                    let nodes = self.nodes.read().unwrap();
                    for node_id in &path_nodes {
                        if let Some(node) = nodes.get(node_id) {
                            nodes_vec.push(node.clone());
                        }
                    }
                    if let Some(node) = nodes.get(&to) {
                        nodes_vec.push(node.clone());
                    }

                    let mut edges_vec: Vec<Edge> = Vec::new();
                    let edges_store = self.edges.read().unwrap();
                    for edge_id in &path_edges {
                        if let Some(e) = edges_store.get(edge_id) {
                            edges_vec.push(e.clone());
                        }
                    }
                    edges_vec.push(edge.clone());

                    let weight = edges_vec.len() as f64;
                    return Some(PathResult {
                        nodes: nodes_vec,
                        edges: edges_vec,
                        weight,
                    });
                }

                if !visited.contains(&edge.to) {
                    visited.insert(edge.to);
                    let mut new_path = path_nodes.clone();
                    new_path.push(edge.to);
                    let mut new_edges = path_edges.clone();
                    new_edges.push(edge.id);
                    queue.push_back((edge.to, new_path, new_edges));
                }
            }
        }

        None
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_node() {
        let store = GraphStore::with_defaults();
        let id = store.create_node();
        assert!(store.get_node(id).is_some());
    }

    #[test]
    fn test_node_labels() {
        let store = GraphStore::with_defaults();
        let id = store.create_node();
        store.add_label(id, "Person");

        let node = store.get_node(id).unwrap();
        assert!(node.has_label("Person"));
    }

    #[test]
    fn test_node_properties() {
        let store = GraphStore::with_defaults();
        let id = store.create_node();
        store.set_property(id, "name", serde_json::json!("Alice"));

        let node = store.get_node(id).unwrap();
        assert_eq!(node.get_property("name"), Some(&serde_json::json!("Alice")));
    }

    #[test]
    fn test_create_edge() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();

        let edge_id = store.create_edge(a, b, "KNOWS").unwrap();
        let edge = store.get_edge(edge_id).unwrap();

        assert_eq!(edge.from, a);
        assert_eq!(edge.to, b);
        assert_eq!(edge.edge_type, "KNOWS");
    }

    #[test]
    fn test_get_neighbors() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();

        store.add_label(a, "Person");
        store.add_label(b, "Person");
        store.add_label(c, "Person");

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");

        let neighbors = store.get_neighbors(a, Some("KNOWS"));
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_get_nodes_by_label() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();

        store.add_label(a, "Person");
        store.add_label(b, "Person");
        store.add_label(c, "Company");

        let people = store.get_nodes_by_label("Person");
        assert_eq!(people.len(), 2);
    }

    #[test]
    fn test_traversal() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();

        store.add_label(c, "Person");

        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let traversal = Traversal::from_node(a)
            .out("KNOWS")
            .out("KNOWS")
            .has_label("Person");

        let results = store.traverse(&traversal);
        assert_eq!(results.len(), 1);
        assert!(results[0].has_label("Person"));
    }

    #[test]
    fn test_shortest_path() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();

        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let path = store.shortest_path(a, c, None).unwrap();
        assert_eq!(path.length(), 2);
    }

    #[test]
    fn test_delete_node() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();

        store.create_edge(a, b, "KNOWS");
        store.delete_node(a);

        assert!(store.get_node(a).is_none());
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_delete_edge() {
        let store = GraphStore::with_defaults();
        let a = store.create_node();
        let b = store.create_node();

        let edge_id = store.create_edge(a, b, "KNOWS").unwrap();
        assert!(store.delete_edge(edge_id));
        assert!(store.get_edge(edge_id).is_none());
    }
}
