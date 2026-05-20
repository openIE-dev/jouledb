//! Graph data structure and layout algorithms.
//!
//! Replaces dagre, ELK, d3-hierarchy layout engines. Provides Sugiyama
//! layered layout, Fruchterman-Reingold spring layout, circular layout,
//! grid layout, and orthogonal edge routing. Pure Rust — no browser dependency.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data types ───────────────────────────────────────────────────

/// A node in the graph with position and dimensions.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Node {
    pub fn new(id: impl Into<String>, width: f64, height: f64) -> Self {
        Self {
            id: id.into(),
            x: 0.0,
            y: 0.0,
            width,
            height,
        }
    }

    pub fn center_x(&self) -> f64 {
        self.x + self.width / 2.0
    }

    pub fn center_y(&self) -> f64 {
        self.y + self.height / 2.0
    }
}

/// A directed edge between two nodes.
#[derive(Debug, Clone)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub label: Option<String>,
}

impl Edge {
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// A segment of an orthogonal edge route.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RouteSegment {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
}

/// A complete graph with nodes and edges.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    pub fn find_node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut Node> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    fn node_index(&self, id: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.id == id)
    }

    /// Build adjacency list (outgoing edges).
    fn adjacency(&self) -> HashMap<&str, Vec<&str>> {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.source.as_str())
                .or_default()
                .push(edge.target.as_str());
        }
        adj
    }

    /// Build reverse adjacency list (incoming edges).
    fn reverse_adjacency(&self) -> HashMap<&str, Vec<&str>> {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.target.as_str())
                .or_default()
                .push(edge.source.as_str());
        }
        adj
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Sugiyama / Layered Layout ────────────────────────────────────

/// Configuration for Sugiyama layered layout.
#[derive(Debug, Clone)]
pub struct SugiyamaConfig {
    pub layer_spacing: f64,
    pub node_spacing: f64,
    pub direction: LayoutDirection,
}

impl Default for SugiyamaConfig {
    fn default() -> Self {
        Self {
            layer_spacing: 100.0,
            node_spacing: 60.0,
            direction: LayoutDirection::TopToBottom,
        }
    }
}

/// Direction for layered layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutDirection {
    TopToBottom,
    LeftToRight,
}

/// Assign layers using longest-path from sources.
fn assign_layers(graph: &Graph) -> HashMap<String, usize> {
    let adj = graph.adjacency();
    let rev = graph.reverse_adjacency();
    let mut layers: HashMap<String, usize> = HashMap::new();

    // Find source nodes (no incoming edges).
    let sources: Vec<&str> = graph
        .nodes
        .iter()
        .filter(|n| rev.get(n.id.as_str()).map_or(true, |v| v.is_empty()))
        .map(|n| n.id.as_str())
        .collect();

    // BFS from sources.
    let mut queue: VecDeque<&str> = VecDeque::new();
    for s in &sources {
        layers.insert(s.to_string(), 0);
        queue.push_back(s);
    }

    // If no sources (cyclic), start from first node.
    if queue.is_empty() {
        if let Some(n) = graph.nodes.first() {
            layers.insert(n.id.clone(), 0);
            queue.push_back(n.id.as_str());
        }
    }

    while let Some(current) = queue.pop_front() {
        let current_layer = layers[current];
        if let Some(neighbors) = adj.get(current) {
            for neighbor in neighbors {
                let new_layer = current_layer + 1;
                let existing = layers.get(*neighbor).copied().unwrap_or(0);
                if new_layer > existing || !layers.contains_key(*neighbor) {
                    layers.insert(neighbor.to_string(), new_layer);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    // Ensure all nodes have a layer.
    for node in &graph.nodes {
        layers.entry(node.id.clone()).or_insert(0);
    }

    layers
}

/// Minimize crossings using barycenter heuristic.
fn barycenter_ordering(
    layers_map: &HashMap<String, usize>,
    graph: &Graph,
    max_layer: usize,
) -> Vec<Vec<String>> {
    // Group nodes by layer.
    let mut by_layer: Vec<Vec<String>> = vec![Vec::new(); max_layer + 1];
    for (id, layer) in layers_map {
        by_layer[*layer].push(id.clone());
    }

    // Sort each layer alphabetically first for determinism.
    for layer in &mut by_layer {
        layer.sort();
    }

    let adj = graph.adjacency();
    let rev = graph.reverse_adjacency();

    // Sweep down: order nodes by average position of predecessors.
    for layer_idx in 1..=max_layer {
        let prev_layer = &by_layer[layer_idx - 1];
        let prev_positions: HashMap<&str, f64> = prev_layer
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i as f64))
            .collect();

        let mut barycenters: Vec<(String, f64)> = Vec::new();
        for node_id in &by_layer[layer_idx] {
            if let Some(preds) = rev.get(node_id.as_str()) {
                let positions: Vec<f64> = preds
                    .iter()
                    .filter_map(|p| prev_positions.get(p).copied())
                    .collect();
                let bc = if positions.is_empty() {
                    0.0
                } else {
                    positions.iter().sum::<f64>() / positions.len() as f64
                };
                barycenters.push((node_id.clone(), bc));
            } else {
                barycenters.push((node_id.clone(), 0.0));
            }
        }

        barycenters.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        by_layer[layer_idx] = barycenters.into_iter().map(|(id, _)| id).collect();
    }

    // Sweep up for refinement.
    for layer_idx in (0..max_layer).rev() {
        let next_layer = &by_layer[layer_idx + 1];
        let next_positions: HashMap<&str, f64> = next_layer
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i as f64))
            .collect();

        let mut barycenters: Vec<(String, f64)> = Vec::new();
        for node_id in &by_layer[layer_idx] {
            if let Some(succs) = adj.get(node_id.as_str()) {
                let positions: Vec<f64> = succs
                    .iter()
                    .filter_map(|s| next_positions.get(s).copied())
                    .collect();
                let bc = if positions.is_empty() {
                    0.0
                } else {
                    positions.iter().sum::<f64>() / positions.len() as f64
                };
                barycenters.push((node_id.clone(), bc));
            } else {
                barycenters.push((node_id.clone(), 0.0));
            }
        }

        barycenters.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        by_layer[layer_idx] = barycenters.into_iter().map(|(id, _)| id).collect();
    }

    by_layer
}

/// Apply Sugiyama layered layout to the graph.
pub fn sugiyama_layout(graph: &mut Graph, config: &SugiyamaConfig) {
    if graph.nodes.is_empty() {
        return;
    }

    let layers = assign_layers(graph);
    let max_layer = layers.values().copied().max().unwrap_or(0);
    let ordered = barycenter_ordering(&layers, graph, max_layer);

    for (layer_idx, layer_nodes) in ordered.iter().enumerate() {
        for (pos_idx, node_id) in layer_nodes.iter().enumerate() {
            if let Some(node) = graph.find_node_mut(node_id) {
                match config.direction {
                    LayoutDirection::TopToBottom => {
                        node.x = pos_idx as f64 * config.node_spacing;
                        node.y = layer_idx as f64 * config.layer_spacing;
                    }
                    LayoutDirection::LeftToRight => {
                        node.x = layer_idx as f64 * config.layer_spacing;
                        node.y = pos_idx as f64 * config.node_spacing;
                    }
                }
            }
        }
    }
}

// ── Fruchterman-Reingold Spring Layout ───────────────────────────

/// Configuration for spring (force-directed) layout.
#[derive(Debug, Clone)]
pub struct SpringConfig {
    pub width: f64,
    pub height: f64,
    pub iterations: usize,
    pub repulsion: f64,
    pub attraction: f64,
    pub damping: f64,
}

impl Default for SpringConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 600.0,
            iterations: 50,
            repulsion: 10000.0,
            attraction: 0.01,
            damping: 0.9,
        }
    }
}

/// Apply Fruchterman-Reingold spring layout.
pub fn spring_layout(graph: &mut Graph, config: &SpringConfig) {
    let n = graph.nodes.len();
    if n == 0 {
        return;
    }

    // Initialize positions in a circle.
    let cx = config.width / 2.0;
    let cy = config.height / 2.0;
    let radius = config.width.min(config.height) / 3.0;

    for (i, node) in graph.nodes.iter_mut().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        node.x = cx + radius * angle.cos();
        node.y = cy + radius * angle.sin();
    }

    let area = config.width * config.height;
    let k = (area / n as f64).sqrt();

    for _iter in 0..config.iterations {
        // Compute displacements.
        let mut dx = vec![0.0_f64; n];
        let mut dy = vec![0.0_f64; n];

        // Repulsive forces between all node pairs.
        for i in 0..n {
            for j in (i + 1)..n {
                let ddx = graph.nodes[i].center_x() - graph.nodes[j].center_x();
                let ddy = graph.nodes[i].center_y() - graph.nodes[j].center_y();
                let dist = (ddx * ddx + ddy * ddy).sqrt().max(0.01);
                let force = config.repulsion / (dist * dist);
                let fx = ddx / dist * force;
                let fy = ddy / dist * force;
                dx[i] += fx;
                dy[i] += fy;
                dx[j] -= fx;
                dy[j] -= fy;
            }
        }

        // Attractive forces along edges.
        for edge in &graph.edges {
            let si = graph.node_index(&edge.source);
            let ti = graph.node_index(&edge.target);
            if let (Some(si), Some(ti)) = (si, ti) {
                let ddx = graph.nodes[si].center_x() - graph.nodes[ti].center_x();
                let ddy = graph.nodes[si].center_y() - graph.nodes[ti].center_y();
                let dist = (ddx * ddx + ddy * ddy).sqrt().max(0.01);
                let force = dist * config.attraction;
                let fx = ddx / dist * force;
                let fy = ddy / dist * force;
                dx[si] -= fx;
                dy[si] -= fy;
                dx[ti] += fx;
                dy[ti] += fy;
            }
        }

        // Apply displacements with damping and temperature.
        let temperature = k * (1.0 - _iter as f64 / config.iterations as f64);
        for i in 0..n {
            let disp = (dx[i] * dx[i] + dy[i] * dy[i]).sqrt().max(0.01);
            let scale = (temperature / disp).min(1.0) * config.damping;
            graph.nodes[i].x += dx[i] * scale;
            graph.nodes[i].y += dy[i] * scale;

            // Clamp to bounds.
            graph.nodes[i].x = graph.nodes[i].x.clamp(0.0, config.width - graph.nodes[i].width);
            graph.nodes[i].y = graph.nodes[i]
                .y
                .clamp(0.0, config.height - graph.nodes[i].height);
        }
    }
}

// ── Circular Layout ──────────────────────────────────────────────

/// Configuration for circular layout.
#[derive(Debug, Clone)]
pub struct CircularConfig {
    pub center_x: f64,
    pub center_y: f64,
    pub radius: f64,
}

impl Default for CircularConfig {
    fn default() -> Self {
        Self {
            center_x: 400.0,
            center_y: 300.0,
            radius: 200.0,
        }
    }
}

/// Place nodes evenly around a circle.
pub fn circular_layout(graph: &mut Graph, config: &CircularConfig) {
    let n = graph.nodes.len();
    if n == 0 {
        return;
    }

    for (i, node) in graph.nodes.iter_mut().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        node.x = config.center_x + config.radius * angle.cos() - node.width / 2.0;
        node.y = config.center_y + config.radius * angle.sin() - node.height / 2.0;
    }
}

// ── Grid Layout ──────────────────────────────────────────────────

/// Configuration for grid layout.
#[derive(Debug, Clone)]
pub struct GridConfig {
    pub columns: usize,
    pub cell_width: f64,
    pub cell_height: f64,
    pub padding: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            columns: 4,
            cell_width: 120.0,
            cell_height: 80.0,
            padding: 20.0,
        }
    }
}

/// Arrange nodes in a grid.
pub fn grid_layout(graph: &mut Graph, config: &GridConfig) {
    let cols = config.columns.max(1);
    for (i, node) in graph.nodes.iter_mut().enumerate() {
        let col = i % cols;
        let row = i / cols;
        node.x = col as f64 * (config.cell_width + config.padding);
        node.y = row as f64 * (config.cell_height + config.padding);
    }
}

// ── Orthogonal Edge Routing ──────────────────────────────────────

/// Compute orthogonal (right-angle) route segments for an edge.
pub fn orthogonal_route(graph: &Graph, edge: &Edge) -> Vec<RouteSegment> {
    let source = match graph.find_node(&edge.source) {
        Some(n) => n,
        None => return Vec::new(),
    };
    let target = match graph.find_node(&edge.target) {
        Some(n) => n,
        None => return Vec::new(),
    };

    let sx = source.center_x();
    let sy = source.center_y();
    let tx = target.center_x();
    let ty = target.center_y();

    // Route with a midpoint for the bend.
    let mid_y = (sy + ty) / 2.0;

    // Vertical from source to mid, horizontal to target x, vertical to target.
    let mut segments = Vec::new();

    if (sy - ty).abs() < 1.0 {
        // Same vertical level — straight horizontal.
        segments.push(RouteSegment {
            x1: sx,
            y1: sy,
            x2: tx,
            y2: ty,
        });
    } else if (sx - tx).abs() < 1.0 {
        // Same horizontal — straight vertical.
        segments.push(RouteSegment {
            x1: sx,
            y1: sy,
            x2: tx,
            y2: ty,
        });
    } else {
        // Three-segment orthogonal route.
        segments.push(RouteSegment {
            x1: sx,
            y1: sy,
            x2: sx,
            y2: mid_y,
        });
        segments.push(RouteSegment {
            x1: sx,
            y1: mid_y,
            x2: tx,
            y2: mid_y,
        });
        segments.push(RouteSegment {
            x1: tx,
            y1: mid_y,
            x2: tx,
            y2: ty,
        });
    }

    segments
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new("a", 40.0, 30.0));
        g.add_node(Node::new("b", 40.0, 30.0));
        g.add_node(Node::new("c", 40.0, 30.0));
        g.add_node(Node::new("d", 40.0, 30.0));
        g.add_edge(Edge::new("a", "b"));
        g.add_edge(Edge::new("a", "c"));
        g.add_edge(Edge::new("b", "d"));
        g.add_edge(Edge::new("c", "d"));
        g
    }

    #[test]
    fn test_node_creation() {
        let n = Node::new("n1", 100.0, 50.0);
        assert_eq!(n.id, "n1");
        assert_eq!(n.center_x(), 50.0);
        assert_eq!(n.center_y(), 25.0);
    }

    #[test]
    fn test_edge_with_label() {
        let e = Edge::new("a", "b").with_label("depends");
        assert_eq!(e.label.as_deref(), Some("depends"));
    }

    #[test]
    fn test_graph_find_node() {
        let g = sample_graph();
        assert!(g.find_node("a").is_some());
        assert!(g.find_node("z").is_none());
    }

    #[test]
    fn test_assign_layers() {
        let g = sample_graph();
        let layers = assign_layers(&g);
        assert_eq!(layers["a"], 0);
        assert!(layers["b"] > 0);
        assert!(layers["d"] > layers["b"]);
    }

    #[test]
    fn test_sugiyama_layout() {
        let mut g = sample_graph();
        sugiyama_layout(&mut g, &SugiyamaConfig::default());
        // Node "a" should be at layer 0.
        let a = g.find_node("a").unwrap();
        assert_eq!(a.y, 0.0);
        // Node "d" should be in a later layer.
        let d = g.find_node("d").unwrap();
        assert!(d.y > a.y);
    }

    #[test]
    fn test_sugiyama_left_to_right() {
        let mut g = sample_graph();
        let config = SugiyamaConfig {
            direction: LayoutDirection::LeftToRight,
            ..Default::default()
        };
        sugiyama_layout(&mut g, &config);
        let a = g.find_node("a").unwrap();
        let d = g.find_node("d").unwrap();
        assert!(d.x > a.x);
    }

    #[test]
    fn test_spring_layout_within_bounds() {
        let mut g = sample_graph();
        let config = SpringConfig {
            width: 400.0,
            height: 300.0,
            iterations: 20,
            ..Default::default()
        };
        spring_layout(&mut g, &config);
        for node in &g.nodes {
            assert!(node.x >= 0.0, "node {} x below 0", node.id);
            assert!(node.y >= 0.0, "node {} y below 0", node.id);
            assert!(node.x <= 400.0, "node {} x above width", node.id);
            assert!(node.y <= 300.0, "node {} y above height", node.id);
        }
    }

    #[test]
    fn test_circular_layout() {
        let mut g = Graph::new();
        for i in 0..6 {
            g.add_node(Node::new(format!("n{i}"), 20.0, 20.0));
        }
        let config = CircularConfig {
            center_x: 200.0,
            center_y: 200.0,
            radius: 100.0,
        };
        circular_layout(&mut g, &config);
        // First node should be at angle 0 → center_x + radius.
        let n0 = g.find_node("n0").unwrap();
        assert!((n0.x - (200.0 + 100.0 - 10.0)).abs() < 0.01);
    }

    #[test]
    fn test_grid_layout() {
        let mut g = Graph::new();
        for i in 0..7 {
            g.add_node(Node::new(format!("n{i}"), 40.0, 30.0));
        }
        let config = GridConfig {
            columns: 3,
            ..Default::default()
        };
        grid_layout(&mut g, &config);
        // Node 3 should be at row 1, col 0.
        let n3 = g.find_node("n3").unwrap();
        assert_eq!(n3.x, 0.0);
        assert!((n3.y - (config.cell_height + config.padding)).abs() < 0.01);
    }

    #[test]
    fn test_orthogonal_route_diagonal() {
        let mut g = Graph::new();
        let mut a = Node::new("a", 40.0, 30.0);
        a.x = 0.0;
        a.y = 0.0;
        let mut b = Node::new("b", 40.0, 30.0);
        b.x = 200.0;
        b.y = 200.0;
        g.add_node(a);
        g.add_node(b);
        let edge = Edge::new("a", "b");
        let route = orthogonal_route(&g, &edge);
        assert_eq!(route.len(), 3);
        // All segments should be axis-aligned.
        for seg in &route {
            assert!(
                (seg.x1 - seg.x2).abs() < 0.01 || (seg.y1 - seg.y2).abs() < 0.01,
                "segment not axis-aligned: {seg:?}"
            );
        }
    }

    #[test]
    fn test_orthogonal_route_same_row() {
        let mut g = Graph::new();
        let mut a = Node::new("a", 40.0, 30.0);
        a.x = 0.0;
        a.y = 100.0;
        let mut b = Node::new("b", 40.0, 30.0);
        b.x = 200.0;
        b.y = 100.0;
        g.add_node(a);
        g.add_node(b);
        let edge = Edge::new("a", "b");
        let route = orthogonal_route(&g, &edge);
        assert_eq!(route.len(), 1);
    }

    #[test]
    fn test_empty_graph_layouts() {
        let mut g = Graph::new();
        sugiyama_layout(&mut g, &SugiyamaConfig::default());
        spring_layout(&mut g, &SpringConfig::default());
        circular_layout(&mut g, &CircularConfig::default());
        grid_layout(&mut g, &GridConfig::default());
        assert!(g.nodes.is_empty());
    }

    #[test]
    fn test_adjacency_lists() {
        let g = sample_graph();
        let adj = g.adjacency();
        assert_eq!(adj["a"].len(), 2);
        let rev = g.reverse_adjacency();
        assert_eq!(rev["d"].len(), 2);
    }
}
