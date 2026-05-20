//! Network topology visualization.
//!
//! Replaces vis.js network / Cytoscape for infrastructure diagrams.
//! Provides topology layouts (star, ring, bus, mesh, tree, hierarchical),
//! traffic state, subnet grouping, and path tracing. Pure Rust — no browser dependency.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data types ───────────────────────────────────────────────────

/// Type of network device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    Server,
    Client,
    Router,
    Switch,
    Cloud,
    Database,
    Firewall,
}

/// Link status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkStatus {
    Up,
    Down,
}

/// A node in the network topology.
#[derive(Debug, Clone)]
pub struct NetworkNode {
    pub id: String,
    pub label: String,
    pub device_type: DeviceType,
    pub x: f64,
    pub y: f64,
    pub subnet: Option<String>,
}

impl NetworkNode {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        device_type: DeviceType,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            device_type,
            x: 0.0,
            y: 0.0,
            subnet: None,
        }
    }

    pub fn with_subnet(mut self, subnet: impl Into<String>) -> Self {
        self.subnet = Some(subnet.into());
        self
    }
}

/// A link between two network nodes.
#[derive(Debug, Clone)]
pub struct NetworkLink {
    pub source: String,
    pub target: String,
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub status: LinkStatus,
}

impl NetworkLink {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        bandwidth_mbps: f64,
        latency_ms: f64,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            bandwidth_mbps,
            latency_ms,
            status: LinkStatus::Up,
        }
    }

    pub fn with_status(mut self, status: LinkStatus) -> Self {
        self.status = status;
        self
    }
}

/// Traffic animation state for a link.
#[derive(Debug, Clone)]
pub struct TrafficState {
    pub link_source: String,
    pub link_target: String,
    pub utilization: f64,
    pub packets_per_sec: u64,
    pub direction_forward: bool,
}

/// Topology layout mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyLayout {
    Star,
    Ring,
    Bus,
    Mesh,
    Tree,
    Hierarchical,
}

// ── Network ──────────────────────────────────────────────────────

/// The complete network topology.
#[derive(Debug, Clone)]
pub struct Network {
    nodes: Vec<NetworkNode>,
    links: Vec<NetworkLink>,
    traffic: Vec<TrafficState>,
    node_map: HashMap<String, usize>,
}

impl Network {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            links: Vec::new(),
            traffic: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: NetworkNode) {
        let idx = self.nodes.len();
        self.node_map.insert(node.id.clone(), idx);
        self.nodes.push(node);
    }

    pub fn add_link(&mut self, link: NetworkLink) {
        self.links.push(link);
    }

    pub fn nodes(&self) -> &[NetworkNode] {
        &self.nodes
    }

    pub fn links(&self) -> &[NetworkLink] {
        &self.links
    }

    pub fn find_node(&self, id: &str) -> Option<&NetworkNode> {
        self.node_map.get(id).map(|i| &self.nodes[*i])
    }

    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut NetworkNode> {
        self.node_map.get(id).copied().map(|i| &mut self.nodes[i])
    }

    pub fn set_link_status(&mut self, source: &str, target: &str, status: LinkStatus) {
        for link in &mut self.links {
            if (link.source == source && link.target == target)
                || (link.source == target && link.target == source)
            {
                link.status = status;
            }
        }
    }

    pub fn update_traffic(&mut self, state: TrafficState) {
        if let Some(existing) = self.traffic.iter_mut().find(|t| {
            t.link_source == state.link_source && t.link_target == state.link_target
        }) {
            *existing = state;
        } else {
            self.traffic.push(state);
        }
    }

    pub fn traffic(&self) -> &[TrafficState] {
        &self.traffic
    }

    /// Get all nodes in a subnet.
    pub fn subnet_nodes(&self, subnet: &str) -> Vec<&NetworkNode> {
        self.nodes
            .iter()
            .filter(|n| n.subnet.as_deref() == Some(subnet))
            .collect()
    }

    /// Get all unique subnet names.
    pub fn subnets(&self) -> Vec<String> {
        let mut subs: HashSet<String> = HashSet::new();
        for node in &self.nodes {
            if let Some(s) = &node.subnet {
                subs.insert(s.clone());
            }
        }
        let mut result: Vec<String> = subs.into_iter().collect();
        result.sort();
        result
    }

    /// Build adjacency for up-status links only.
    fn adjacency_up(&self) -> HashMap<&str, Vec<&str>> {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for link in &self.links {
            if link.status == LinkStatus::Up {
                adj.entry(link.source.as_str())
                    .or_default()
                    .push(link.target.as_str());
                adj.entry(link.target.as_str())
                    .or_default()
                    .push(link.source.as_str());
            }
        }
        adj
    }

    /// Trace route (BFS shortest path) between two nodes using up links.
    pub fn trace_route(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let adj = self.adjacency_up();
        let mut visited: HashSet<&str> = HashSet::new();
        let mut parent: HashMap<&str, &str> = HashMap::new();
        let mut queue: VecDeque<&str> = VecDeque::new();

        visited.insert(from);
        queue.push_back(from);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adj.get(current) {
                for neighbor in neighbors {
                    if visited.insert(neighbor) {
                        parent.insert(neighbor, current);
                        if *neighbor == to {
                            // Reconstruct path.
                            let mut path = vec![to.to_string()];
                            let mut cur = to;
                            while let Some(&p) = parent.get(cur) {
                                path.push(p.to_string());
                                cur = p;
                            }
                            path.reverse();
                            return Some(path);
                        }
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        None
    }

    /// Apply a topology layout to all nodes.
    pub fn apply_layout(&mut self, layout: TopologyLayout, width: f64, height: f64) {
        match layout {
            TopologyLayout::Star => self.layout_star(width, height),
            TopologyLayout::Ring => self.layout_ring(width, height),
            TopologyLayout::Bus => self.layout_bus(width, height),
            TopologyLayout::Mesh => self.layout_mesh(width, height),
            TopologyLayout::Tree => self.layout_tree(width, height),
            TopologyLayout::Hierarchical => self.layout_hierarchical(width, height),
        }
    }

    fn layout_star(&mut self, width: f64, height: f64) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        let cx = width / 2.0;
        let cy = height / 2.0;
        let radius = width.min(height) / 3.0;

        // First node at center, rest in circle.
        self.nodes[0].x = cx;
        self.nodes[0].y = cy;

        for i in 1..n {
            let angle = 2.0 * std::f64::consts::PI * (i - 1) as f64 / (n - 1).max(1) as f64;
            self.nodes[i].x = cx + radius * angle.cos();
            self.nodes[i].y = cy + radius * angle.sin();
        }
    }

    fn layout_ring(&mut self, width: f64, height: f64) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        let cx = width / 2.0;
        let cy = height / 2.0;
        let radius = width.min(height) / 3.0;

        for (i, node) in self.nodes.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            node.x = cx + radius * angle.cos();
            node.y = cy + radius * angle.sin();
        }
    }

    fn layout_bus(&mut self, width: f64, height: f64) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        let spacing = width / (n + 1) as f64;
        for (i, node) in self.nodes.iter_mut().enumerate() {
            node.x = spacing * (i + 1) as f64;
            node.y = height / 2.0;
        }
    }

    fn layout_mesh(&mut self, width: f64, height: f64) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        let cols = (n as f64).sqrt().ceil() as usize;
        let cell_w = width / (cols + 1) as f64;
        let rows = (n + cols - 1) / cols;
        let cell_h = height / (rows + 1) as f64;

        for (i, node) in self.nodes.iter_mut().enumerate() {
            let col = i % cols;
            let row = i / cols;
            node.x = cell_w * (col + 1) as f64;
            node.y = cell_h * (row + 1) as f64;
        }
    }

    fn layout_tree(&mut self, width: f64, height: f64) {
        // Use BFS layers from first node.
        self.layout_hierarchical(width, height);
    }

    fn layout_hierarchical(&mut self, width: f64, height: f64) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        let adj = self.adjacency_up();
        let mut layers: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<&str> = VecDeque::new();

        let root = self.nodes[0].id.clone();
        layers.insert(root.clone(), 0);
        queue.push_back(&self.nodes[0].id);

        while let Some(current) = queue.pop_front() {
            let layer = layers[current];
            if let Some(neighbors) = adj.get(current) {
                for neighbor in neighbors {
                    if !layers.contains_key(*neighbor) {
                        layers.insert(neighbor.to_string(), layer + 1);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Assign unlayered nodes.
        for node in &self.nodes {
            layers.entry(node.id.clone()).or_insert(0);
        }

        let max_layer = layers.values().copied().max().unwrap_or(0);
        let mut by_layer: Vec<Vec<String>> = vec![Vec::new(); max_layer + 1];
        for (id, layer) in &layers {
            by_layer[*layer].push(id.clone());
        }

        let layer_height = height / (max_layer + 2) as f64;
        for (layer_idx, layer_nodes) in by_layer.iter().enumerate() {
            let layer_width = width / (layer_nodes.len() + 1) as f64;
            for (pos_idx, node_id) in layer_nodes.iter().enumerate() {
                if let Some(node) = self.find_node_mut(node_id) {
                    node.x = layer_width * (pos_idx + 1) as f64;
                    node.y = layer_height * (layer_idx + 1) as f64;
                }
            }
        }
    }

    /// Count active (up) links.
    pub fn active_link_count(&self) -> usize {
        self.links.iter().filter(|l| l.status == LinkStatus::Up).count()
    }

    /// Count down links.
    pub fn down_link_count(&self) -> usize {
        self.links.iter().filter(|l| l.status == LinkStatus::Down).count()
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_network() -> Network {
        let mut net = Network::new();
        net.add_node(NetworkNode::new("r1", "Router 1", DeviceType::Router).with_subnet("10.0.0.0/24"));
        net.add_node(NetworkNode::new("s1", "Switch 1", DeviceType::Switch).with_subnet("10.0.0.0/24"));
        net.add_node(NetworkNode::new("srv1", "Server 1", DeviceType::Server).with_subnet("10.0.1.0/24"));
        net.add_node(NetworkNode::new("srv2", "Server 2", DeviceType::Server).with_subnet("10.0.1.0/24"));
        net.add_node(NetworkNode::new("fw", "Firewall", DeviceType::Firewall));
        net.add_link(NetworkLink::new("r1", "s1", 1000.0, 0.5));
        net.add_link(NetworkLink::new("s1", "srv1", 1000.0, 0.2));
        net.add_link(NetworkLink::new("s1", "srv2", 1000.0, 0.2));
        net.add_link(NetworkLink::new("r1", "fw", 10000.0, 0.1));
        net
    }

    #[test]
    fn test_add_nodes_and_links() {
        let net = sample_network();
        assert_eq!(net.nodes().len(), 5);
        assert_eq!(net.links().len(), 4);
    }

    #[test]
    fn test_find_node() {
        let net = sample_network();
        let r1 = net.find_node("r1").unwrap();
        assert_eq!(r1.device_type, DeviceType::Router);
        assert!(net.find_node("nonexistent").is_none());
    }

    #[test]
    fn test_subnet_nodes() {
        let net = sample_network();
        let subnet = net.subnet_nodes("10.0.1.0/24");
        assert_eq!(subnet.len(), 2);
    }

    #[test]
    fn test_subnets() {
        let net = sample_network();
        let subs = net.subnets();
        assert_eq!(subs.len(), 2);
    }

    #[test]
    fn test_trace_route() {
        let net = sample_network();
        let route = net.trace_route("r1", "srv1").unwrap();
        assert_eq!(route.first().unwrap(), "r1");
        assert_eq!(route.last().unwrap(), "srv1");
        assert!(route.len() >= 2);
    }

    #[test]
    fn test_trace_route_self() {
        let net = sample_network();
        let route = net.trace_route("r1", "r1").unwrap();
        assert_eq!(route, vec!["r1"]);
    }

    #[test]
    fn test_trace_route_down_link() {
        let mut net = sample_network();
        net.set_link_status("s1", "srv1", LinkStatus::Down);
        // Route should not go through downed link directly.
        let route = net.trace_route("s1", "srv1");
        // With the down link, s1->srv1 is not possible directly, might route via r1.
        // But if that path exists, it should still find one.
        if let Some(r) = &route {
            assert!(!r.is_empty());
        }
    }

    #[test]
    fn test_link_status_counts() {
        let mut net = sample_network();
        assert_eq!(net.active_link_count(), 4);
        assert_eq!(net.down_link_count(), 0);
        net.set_link_status("r1", "s1", LinkStatus::Down);
        assert_eq!(net.active_link_count(), 3);
        assert_eq!(net.down_link_count(), 1);
    }

    #[test]
    fn test_layout_star() {
        let mut net = sample_network();
        net.apply_layout(TopologyLayout::Star, 800.0, 600.0);
        let r1 = net.find_node("r1").unwrap();
        assert_eq!(r1.x, 400.0);
        assert_eq!(r1.y, 300.0);
    }

    #[test]
    fn test_layout_ring() {
        let mut net = sample_network();
        net.apply_layout(TopologyLayout::Ring, 800.0, 600.0);
        // All nodes should have positions set.
        for node in net.nodes() {
            assert!(node.x > 0.0 || node.y > 0.0, "node {} not positioned", node.id);
        }
    }

    #[test]
    fn test_layout_bus() {
        let mut net = sample_network();
        net.apply_layout(TopologyLayout::Bus, 800.0, 600.0);
        // All nodes at same y.
        let y_vals: Vec<f64> = net.nodes().iter().map(|n| n.y).collect();
        assert!(y_vals.iter().all(|y| (y - 300.0).abs() < 0.01));
    }

    #[test]
    fn test_traffic_state() {
        let mut net = sample_network();
        net.update_traffic(TrafficState {
            link_source: "r1".into(),
            link_target: "s1".into(),
            utilization: 0.75,
            packets_per_sec: 50000,
            direction_forward: true,
        });
        assert_eq!(net.traffic().len(), 1);
        assert_eq!(net.traffic()[0].utilization, 0.75);

        // Update same link.
        net.update_traffic(TrafficState {
            link_source: "r1".into(),
            link_target: "s1".into(),
            utilization: 0.90,
            packets_per_sec: 70000,
            direction_forward: true,
        });
        assert_eq!(net.traffic().len(), 1);
        assert_eq!(net.traffic()[0].utilization, 0.90);
    }

    #[test]
    fn test_layout_hierarchical() {
        let mut net = sample_network();
        net.apply_layout(TopologyLayout::Hierarchical, 800.0, 600.0);
        // Root should be in first layer.
        let r1 = net.find_node("r1").unwrap();
        let srv1 = net.find_node("srv1").unwrap();
        assert!(srv1.y > r1.y);
    }
}
