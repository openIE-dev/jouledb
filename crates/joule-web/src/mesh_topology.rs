//! Network mesh topology management — node and edge management, neighbor
//! discovery, connected component detection, mesh optimization, broadcast
//! via flooding, topology change events, and mesh health metrics.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── TopologyEvent ───────────────────────────────────────────────────────────

/// Events emitted when the topology changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyEvent {
    NodeAdded(String),
    NodeRemoved(String),
    EdgeAdded(String, String),
    EdgeRemoved(String, String),
}

impl fmt::Display for TopologyEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyEvent::NodeAdded(n) => write!(f, "+Node({})", n),
            TopologyEvent::NodeRemoved(n) => write!(f, "-Node({})", n),
            TopologyEvent::EdgeAdded(a, b) => write!(f, "+Edge({}<->{})", a, b),
            TopologyEvent::EdgeRemoved(a, b) => write!(f, "-Edge({}<->{})", a, b),
        }
    }
}

// ── MeshNode ────────────────────────────────────────────────────────────────

/// A node in the mesh network.
#[derive(Debug, Clone)]
pub struct MeshNode {
    pub id: String,
    pub neighbors: HashSet<String>,
    pub metadata: HashMap<String, String>,
}

impl MeshNode {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            neighbors: HashSet::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Degree (number of neighbors).
    pub fn degree(&self) -> usize {
        self.neighbors.len()
    }
}

impl fmt::Display for MeshNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeshNode({}, degree={})", self.id, self.degree())
    }
}

// ── MeshHealth ──────────────────────────────────────────────────────────────

/// Health metrics for the mesh network.
#[derive(Debug, Clone)]
pub struct MeshHealth {
    pub node_count: usize,
    pub edge_count: usize,
    pub connected_components: usize,
    pub is_connected: bool,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub min_degree: usize,
    pub redundancy_factor: f64,
}

impl fmt::Display for MeshHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MeshHealth(nodes={}, edges={}, components={}, connected={}, avg_deg={:.1})",
            self.node_count, self.edge_count, self.connected_components,
            self.is_connected, self.avg_degree,
        )
    }
}

// ── BroadcastResult ─────────────────────────────────────────────────────────

/// Result of a broadcast via flooding.
#[derive(Debug, Clone)]
pub struct BroadcastResult {
    pub reached: Vec<String>,
    pub unreached: Vec<String>,
    pub total_messages: usize,
}

// ── MeshNetwork ─────────────────────────────────────────────────────────────

/// Manages a mesh network topology.
pub struct MeshNetwork {
    nodes: HashMap<String, MeshNode>,
    events: Vec<TopologyEvent>,
}

impl MeshNetwork {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Add a node to the mesh.
    pub fn add_node(&mut self, id: impl Into<String>) -> bool {
        let nid = id.into();
        if self.nodes.contains_key(&nid) {
            return false;
        }
        self.events.push(TopologyEvent::NodeAdded(nid.clone()));
        self.nodes.insert(nid.clone(), MeshNode::new(nid));
        true
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: &str) -> bool {
        if let Some(node) = self.nodes.remove(id) {
            // Remove from all neighbors
            for neighbor_id in &node.neighbors {
                if let Some(neighbor) = self.nodes.get_mut(neighbor_id) {
                    neighbor.neighbors.remove(id);
                }
            }
            self.events.push(TopologyEvent::NodeRemoved(id.to_string()));
            true
        } else {
            false
        }
    }

    /// Add a bidirectional edge between two nodes.
    pub fn add_edge(&mut self, a: &str, b: &str) -> bool {
        if a == b {
            return false;
        }
        if !self.nodes.contains_key(a) || !self.nodes.contains_key(b) {
            return false;
        }
        let already = self.nodes[a].neighbors.contains(b);
        if already {
            return false;
        }
        self.nodes.get_mut(a).unwrap().neighbors.insert(b.to_string());
        self.nodes.get_mut(b).unwrap().neighbors.insert(a.to_string());
        self.events.push(TopologyEvent::EdgeAdded(a.to_string(), b.to_string()));
        true
    }

    /// Remove an edge.
    pub fn remove_edge(&mut self, a: &str, b: &str) -> bool {
        let removed_a = self.nodes.get_mut(a).map(|n| n.neighbors.remove(b)).unwrap_or(false);
        let removed_b = self.nodes.get_mut(b).map(|n| n.neighbors.remove(a)).unwrap_or(false);
        if removed_a || removed_b {
            self.events.push(TopologyEvent::EdgeRemoved(a.to_string(), b.to_string()));
            true
        } else {
            false
        }
    }

    /// Get a node by id.
    pub fn get_node(&self, id: &str) -> Option<&MeshNode> {
        self.nodes.get(id)
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges (each edge counted once).
    pub fn edge_count(&self) -> usize {
        let total_degree: usize = self.nodes.values().map(|n| n.degree()).sum();
        total_degree / 2
    }

    /// Neighbors of a node.
    pub fn neighbors(&self, id: &str) -> Vec<&str> {
        self.nodes
            .get(id)
            .map(|n| n.neighbors.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Find connected components using BFS.
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut components = Vec::new();

        for node_id in self.nodes.keys() {
            if visited.contains(node_id.as_str()) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node_id.as_str());
            visited.insert(node_id.as_str());

            while let Some(current) = queue.pop_front() {
                component.push(current.to_string());
                if let Some(node) = self.nodes.get(current) {
                    for neighbor in &node.neighbors {
                        if !visited.contains(neighbor.as_str()) {
                            visited.insert(neighbor.as_str());
                            queue.push_back(neighbor.as_str());
                        }
                    }
                }
            }
            component.sort();
            components.push(component);
        }
        components.sort_by_key(|c| c.len());
        components.reverse();
        components
    }

    /// Whether the mesh is fully connected (single component).
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }
        self.connected_components().len() == 1
    }

    /// Broadcast from a source node using flooding. Returns which nodes were reached.
    pub fn broadcast(&self, source: &str) -> BroadcastResult {
        let mut reached = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut total_messages = 0;

        if !self.nodes.contains_key(source) {
            return BroadcastResult {
                reached: vec![],
                unreached: self.nodes.keys().cloned().collect(),
                total_messages: 0,
            };
        }

        queue.push_back(source.to_string());
        visited.insert(source.to_string());
        reached.push(source.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&current) {
                for neighbor in &node.neighbors {
                    total_messages += 1;
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        reached.push(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        let unreached: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| !visited.contains(*k))
            .cloned()
            .collect();

        BroadcastResult { reached, unreached, total_messages }
    }

    /// Compute mesh diameter (longest shortest path) for the largest component.
    pub fn diameter(&self) -> usize {
        let components = self.connected_components();
        if components.is_empty() {
            return 0;
        }
        let largest = &components[0];
        let mut max_dist = 0;
        for start in largest {
            let dists = self.bfs_distances(start);
            for d in dists.values() {
                if *d > max_dist {
                    max_dist = *d;
                }
            }
        }
        max_dist
    }

    fn bfs_distances(&self, start: &str) -> HashMap<String, usize> {
        let mut dists = HashMap::new();
        let mut queue = VecDeque::new();
        dists.insert(start.to_string(), 0);
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            let current_dist = dists[&current];
            if let Some(node) = self.nodes.get(&current) {
                for neighbor in &node.neighbors {
                    if !dists.contains_key(neighbor) {
                        dists.insert(neighbor.clone(), current_dist + 1);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        dists
    }

    /// Health metrics snapshot.
    pub fn health(&self) -> MeshHealth {
        let node_count = self.node_count();
        let edge_count = self.edge_count();
        let components = self.connected_components();
        let degrees: Vec<usize> = self.nodes.values().map(|n| n.degree()).collect();
        let avg_degree = if node_count > 0 {
            degrees.iter().sum::<usize>() as f64 / node_count as f64
        } else {
            0.0
        };
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        let min_degree = degrees.iter().copied().min().unwrap_or(0);
        // Redundancy: ratio of edges to minimum spanning tree edges (n-1)
        let redundancy_factor = if node_count > 1 {
            edge_count as f64 / (node_count - 1) as f64
        } else {
            0.0
        };

        MeshHealth {
            node_count,
            edge_count,
            connected_components: components.len(),
            is_connected: components.len() <= 1,
            avg_degree,
            max_degree,
            min_degree,
            redundancy_factor,
        }
    }

    /// Drain pending topology events.
    pub fn drain_events(&mut self) -> Vec<TopologyEvent> {
        std::mem::take(&mut self.events)
    }
}

impl Default for MeshNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_mesh() -> MeshNetwork {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        m.add_node("c");
        m.add_edge("a", "b");
        m.add_edge("b", "c");
        m.add_edge("a", "c");
        m
    }

    #[test]
    fn test_add_node() {
        let mut m = MeshNetwork::new();
        assert!(m.add_node("a"));
        assert!(!m.add_node("a")); // duplicate
        assert_eq!(m.node_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut m = triangle_mesh();
        assert!(m.remove_node("b"));
        assert_eq!(m.node_count(), 2);
        // a and c should no longer list b as neighbor
        assert!(!m.neighbors("a").contains(&"b"));
        assert!(!m.neighbors("c").contains(&"b"));
    }

    #[test]
    fn test_add_edge() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        assert!(m.add_edge("a", "b"));
        assert!(!m.add_edge("a", "b")); // duplicate
        assert_eq!(m.edge_count(), 1);
    }

    #[test]
    fn test_add_edge_self_loop() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        assert!(!m.add_edge("a", "a"));
    }

    #[test]
    fn test_add_edge_missing_node() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        assert!(!m.add_edge("a", "missing"));
    }

    #[test]
    fn test_remove_edge() {
        let mut m = triangle_mesh();
        assert!(m.remove_edge("a", "b"));
        assert_eq!(m.edge_count(), 2);
        assert!(!m.neighbors("a").contains(&"b"));
    }

    #[test]
    fn test_neighbors() {
        let m = triangle_mesh();
        let mut n = m.neighbors("a");
        n.sort();
        assert_eq!(n, vec!["b", "c"]);
    }

    #[test]
    fn test_connected_components_single() {
        let m = triangle_mesh();
        let cc = m.connected_components();
        assert_eq!(cc.len(), 1);
        assert_eq!(cc[0].len(), 3);
    }

    #[test]
    fn test_connected_components_multiple() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        m.add_edge("a", "b");
        m.add_node("c");
        let cc = m.connected_components();
        assert_eq!(cc.len(), 2);
    }

    #[test]
    fn test_is_connected() {
        let m = triangle_mesh();
        assert!(m.is_connected());
    }

    #[test]
    fn test_is_not_connected() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        assert!(!m.is_connected());
    }

    #[test]
    fn test_broadcast_all_reached() {
        let m = triangle_mesh();
        let result = m.broadcast("a");
        assert_eq!(result.reached.len(), 3);
        assert!(result.unreached.is_empty());
    }

    #[test]
    fn test_broadcast_partial() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        m.add_edge("a", "b");
        m.add_node("c"); // isolated
        let result = m.broadcast("a");
        assert_eq!(result.reached.len(), 2);
        assert_eq!(result.unreached.len(), 1);
    }

    #[test]
    fn test_diameter() {
        // Line: a-b-c, diameter = 2
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        m.add_node("c");
        m.add_edge("a", "b");
        m.add_edge("b", "c");
        assert_eq!(m.diameter(), 2);
    }

    #[test]
    fn test_health() {
        let m = triangle_mesh();
        let h = m.health();
        assert_eq!(h.node_count, 3);
        assert_eq!(h.edge_count, 3);
        assert!(h.is_connected);
        assert_eq!(h.max_degree, 2);
        assert_eq!(h.min_degree, 2);
    }

    #[test]
    fn test_topology_events() {
        let mut m = MeshNetwork::new();
        m.add_node("a");
        m.add_node("b");
        m.add_edge("a", "b");
        let events = m.drain_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], TopologyEvent::NodeAdded("a".into()));
    }

    #[test]
    fn test_topology_event_display() {
        let e = TopologyEvent::EdgeAdded("a".into(), "b".into());
        assert_eq!(format!("{}", e), "+Edge(a<->b)");
    }

    #[test]
    fn test_mesh_node_display() {
        let n = MeshNode::new("test");
        assert!(format!("{}", n).contains("test"));
    }

    #[test]
    fn test_health_display() {
        let m = triangle_mesh();
        let h = m.health();
        let s = format!("{}", h);
        assert!(s.contains("MeshHealth"));
    }

    #[test]
    fn test_empty_mesh() {
        let m = MeshNetwork::new();
        assert!(m.is_connected());
        assert_eq!(m.node_count(), 0);
        assert_eq!(m.edge_count(), 0);
        assert_eq!(m.diameter(), 0);
    }
}
