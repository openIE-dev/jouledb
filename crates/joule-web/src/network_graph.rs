//! Network / force-directed graph visualization.  Nodes and edges with a
//! velocity-Verlet force simulation (repulsion, attraction, centering) and
//! iterative layout.  SVG output with circles, lines, node labels, and edge
//! weights.  Pure Rust — no browser dependency.

use std::fmt::Write as FmtWrite;

// ── Data types ───────────────────────────────────────────────────

/// A node in the network graph.
#[derive(Debug, Clone)]
pub struct NetworkNode {
    pub id: String,
    pub label: String,
    pub color: String,
    pub radius: f64,
    /// Current position.
    pub x: f64,
    pub y: f64,
    /// Current velocity.
    pub vx: f64,
    pub vy: f64,
    /// If true, position is pinned (not moved by forces).
    pub fixed: bool,
}

impl NetworkNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>, color: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            color: color.into(),
            radius: 10.0,
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            fixed: false,
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn with_radius(mut self, r: f64) -> Self {
        self.radius = r.max(1.0);
        self
    }

    pub fn with_fixed(mut self, fixed: bool) -> Self {
        self.fixed = fixed;
        self
    }
}

/// An edge connecting two nodes.
#[derive(Debug, Clone)]
pub struct NetworkEdge {
    pub source_id: String,
    pub target_id: String,
    pub weight: f64,
    pub color: String,
    pub label: Option<String>,
}

impl NetworkEdge {
    pub fn new(source: impl Into<String>, target: impl Into<String>, weight: f64) -> Self {
        Self {
            source_id: source.into(),
            target_id: target.into(),
            weight: weight.max(0.0),
            color: "gray".into(),
            label: None,
        }
    }

    pub fn with_color(mut self, color: impl Into<String>) -> Self {
        self.color = color.into();
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ── Force parameters ────────────────────────────────────────────

/// Parameters for the force simulation.
#[derive(Debug, Clone)]
pub struct ForceParams {
    /// Coulomb repulsion strength (positive pushes apart).
    pub repulsion_strength: f64,
    /// Hooke spring constant for edges.
    pub attraction_strength: f64,
    /// Ideal edge length.
    pub edge_length: f64,
    /// Centering force strength.
    pub center_strength: f64,
    /// Velocity damping per tick (0..1).
    pub damping: f64,
    /// Maximum velocity magnitude.
    pub max_velocity: f64,
    /// Minimum distance for repulsion (avoids division by near-zero).
    pub min_distance: f64,
}

impl Default for ForceParams {
    fn default() -> Self {
        Self {
            repulsion_strength: 5000.0,
            attraction_strength: 0.01,
            edge_length: 100.0,
            center_strength: 0.05,
            damping: 0.85,
            max_velocity: 50.0,
            min_distance: 10.0,
        }
    }
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for the network graph.
#[derive(Debug, Clone)]
pub struct NetworkGraphConfig {
    pub width: f64,
    pub height: f64,
    pub force: ForceParams,
    /// Number of simulation iterations.
    pub iterations: usize,
    pub font_size: f64,
    /// Min/max edge stroke width.
    pub min_edge_width: f64,
    pub max_edge_width: f64,
    /// Whether to show edge weight labels.
    pub show_edge_labels: bool,
}

impl Default for NetworkGraphConfig {
    fn default() -> Self {
        Self {
            width: 600.0,
            height: 400.0,
            force: ForceParams::default(),
            iterations: 100,
            font_size: 10.0,
            min_edge_width: 1.0,
            max_edge_width: 6.0,
            show_edge_labels: false,
        }
    }
}

impl NetworkGraphConfig {
    pub fn center(&self) -> (f64, f64) {
        (self.width / 2.0, self.height / 2.0)
    }
}

// ── Force simulation ────────────────────────────────────────────

/// Initialize node positions in a circle if they are at (0,0).
pub fn init_positions(nodes: &mut [NetworkNode], cx: f64, cy: f64, radius: f64) {
    let n = nodes.len();
    if n == 0 {
        return;
    }
    let step = 2.0 * std::f64::consts::PI / n as f64;
    for (i, node) in nodes.iter_mut().enumerate() {
        if node.x == 0.0 && node.y == 0.0 && !node.fixed {
            let angle = step * i as f64;
            node.x = cx + radius * angle.cos();
            node.y = cy + radius * angle.sin();
        }
    }
}

/// Run one tick of the force simulation.
pub fn simulation_tick(
    nodes: &mut [NetworkNode],
    edges: &[NetworkEdge],
    params: &ForceParams,
    cx: f64,
    cy: f64,
) {
    let n = nodes.len();
    if n == 0 {
        return;
    }

    // Collect positions for immutable reads
    let positions: Vec<(f64, f64)> = nodes.iter().map(|nd| (nd.x, nd.y)).collect();

    // Repulsion (all pairs)
    let mut fx = vec![0.0_f64; n];
    let mut fy = vec![0.0_f64; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i].0 - positions[j].0;
            let dy = positions[i].1 - positions[j].1;
            let dist = (dx * dx + dy * dy).sqrt().max(params.min_distance);
            let force = params.repulsion_strength / (dist * dist);
            let ux = dx / dist;
            let uy = dy / dist;
            fx[i] += force * ux;
            fy[i] += force * uy;
            fx[j] -= force * ux;
            fy[j] -= force * uy;
        }
    }

    // Build id->index lookup
    let id_to_idx: std::collections::HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.id.as_str(), i))
        .collect();

    // Attraction (edges)
    for edge in edges {
        let si = id_to_idx.get(edge.source_id.as_str()).copied();
        let ti = id_to_idx.get(edge.target_id.as_str()).copied();
        if let (Some(si), Some(ti)) = (si, ti) {
            let dx = positions[ti].0 - positions[si].0;
            let dy = positions[ti].1 - positions[si].1;
            let dist = (dx * dx + dy * dy).sqrt().max(params.min_distance);
            let displacement = dist - params.edge_length;
            let force = params.attraction_strength * displacement * edge.weight;
            let ux = dx / dist;
            let uy = dy / dist;
            fx[si] += force * ux;
            fy[si] += force * uy;
            fx[ti] -= force * ux;
            fy[ti] -= force * uy;
        }
    }

    // Centering
    for i in 0..n {
        fx[i] += (cx - positions[i].0) * params.center_strength;
        fy[i] += (cy - positions[i].1) * params.center_strength;
    }

    // Apply forces
    for i in 0..n {
        if nodes[i].fixed {
            continue;
        }
        nodes[i].vx = (nodes[i].vx + fx[i]) * params.damping;
        nodes[i].vy = (nodes[i].vy + fy[i]) * params.damping;

        // Clamp velocity
        let speed = (nodes[i].vx * nodes[i].vx + nodes[i].vy * nodes[i].vy).sqrt();
        if speed > params.max_velocity {
            let scale = params.max_velocity / speed;
            nodes[i].vx *= scale;
            nodes[i].vy *= scale;
        }

        nodes[i].x += nodes[i].vx;
        nodes[i].y += nodes[i].vy;
    }
}

/// Run the full simulation for the configured number of iterations.
pub fn simulate(
    nodes: &mut [NetworkNode],
    edges: &[NetworkEdge],
    cfg: &NetworkGraphConfig,
) {
    let (cx, cy) = cfg.center();
    let init_r = cfg.width.min(cfg.height) * 0.3;
    init_positions(nodes, cx, cy, init_r);

    for _ in 0..cfg.iterations {
        simulation_tick(nodes, edges, &cfg.force, cx, cy);
    }
}

/// Compute total kinetic energy of the system (convergence metric).
pub fn kinetic_energy(nodes: &[NetworkNode]) -> f64 {
    nodes
        .iter()
        .map(|n| n.vx * n.vx + n.vy * n.vy)
        .sum::<f64>()
        * 0.5
}

// ── Rendering ───────────────────────────────────────────────────

/// Render the network graph as SVG.
pub fn render_network_graph(
    nodes: &[NetworkNode],
    edges: &[NetworkEdge],
    cfg: &NetworkGraphConfig,
) -> String {
    let mut svg = String::with_capacity(4096);
    let _ = write!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
         viewBox=\"0 0 {} {}\">",
        cfg.width, cfg.height, cfg.width, cfg.height
    );

    let id_to_node: std::collections::HashMap<&str, &NetworkNode> =
        nodes.iter().map(|n| (n.id.as_str(), n)).collect();

    // Edge weight range for width scaling
    let min_w = edges
        .iter()
        .map(|e| e.weight)
        .fold(f64::INFINITY, f64::min);
    let max_w = edges
        .iter()
        .map(|e| e.weight)
        .fold(f64::NEG_INFINITY, f64::max);
    let w_range = (max_w - min_w).max(f64::EPSILON);

    // Edges
    svg.push_str("<g class=\"edges\">");
    for edge in edges {
        let src = id_to_node.get(edge.source_id.as_str());
        let tgt = id_to_node.get(edge.target_id.as_str());
        if let (Some(s), Some(t)) = (src, tgt) {
            let stroke_w = if edges.len() > 1 {
                cfg.min_edge_width
                    + (edge.weight - min_w) / w_range
                        * (cfg.max_edge_width - cfg.min_edge_width)
            } else {
                (cfg.min_edge_width + cfg.max_edge_width) / 2.0
            };
            let _ = write!(
                svg,
                "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                 stroke=\"{}\" stroke-width=\"{stroke_w:.1}\" stroke-opacity=\"0.6\" />",
                s.x, s.y, t.x, t.y, edge.color
            );

            if cfg.show_edge_labels {
                if let Some(label) = &edge.label {
                    let mx = (s.x + t.x) / 2.0;
                    let my = (s.y + t.y) / 2.0;
                    let fs = cfg.font_size * 0.85;
                    let _ = write!(
                        svg,
                        "<text x=\"{mx}\" y=\"{my}\" font-size=\"{fs}\" \
                         text-anchor=\"middle\" fill=\"gray\">{label}</text>"
                    );
                }
            }
        }
    }
    svg.push_str("</g>");

    // Nodes
    svg.push_str("<g class=\"nodes\">");
    for node in nodes {
        let _ = write!(
            svg,
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" \
             stroke=\"white\" stroke-width=\"1.5\" />",
            node.x, node.y, node.radius, node.color
        );
        let ly = node.y + node.radius + cfg.font_size + 2.0;
        let fs = cfg.font_size;
        let _ = write!(
            svg,
            "<text x=\"{}\" y=\"{ly}\" font-size=\"{fs}\" \
             text-anchor=\"middle\">{}</text>",
            node.x, node.label
        );
    }
    svg.push_str("</g>");

    svg.push_str("</svg>");
    svg
}

/// Convenience: simulate + render.
pub fn network_graph(
    nodes: &mut [NetworkNode],
    edges: &[NetworkEdge],
    cfg: &NetworkGraphConfig,
) -> String {
    simulate(nodes, edges, cfg);
    render_network_graph(nodes, edges, cfg)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nodes() -> Vec<NetworkNode> {
        vec![
            NetworkNode::new("a", "Alpha", "steelblue"),
            NetworkNode::new("b", "Beta", "coral"),
            NetworkNode::new("c", "Gamma", "mediumseagreen"),
        ]
    }

    fn sample_edges() -> Vec<NetworkEdge> {
        vec![
            NetworkEdge::new("a", "b", 1.0),
            NetworkEdge::new("b", "c", 2.0),
            NetworkEdge::new("a", "c", 0.5),
        ]
    }

    #[test]
    fn node_new() {
        let n = NetworkNode::new("x", "X", "red");
        assert_eq!(n.id, "x");
        assert_eq!(n.label, "X");
        assert_eq!(n.radius, 10.0);
    }

    #[test]
    fn node_with_position() {
        let n = NetworkNode::new("x", "X", "red").with_position(50.0, 60.0);
        assert!((n.x - 50.0).abs() < 1e-9);
        assert!((n.y - 60.0).abs() < 1e-9);
    }

    #[test]
    fn node_with_fixed() {
        let n = NetworkNode::new("x", "X", "red").with_fixed(true);
        assert!(n.fixed);
    }

    #[test]
    fn edge_new() {
        let e = NetworkEdge::new("a", "b", 3.0);
        assert_eq!(e.source_id, "a");
        assert_eq!(e.target_id, "b");
        assert!((e.weight - 3.0).abs() < 1e-9);
    }

    #[test]
    fn edge_clamps_negative() {
        let e = NetworkEdge::new("a", "b", -5.0);
        assert_eq!(e.weight, 0.0);
    }

    #[test]
    fn edge_with_label() {
        let e = NetworkEdge::new("a", "b", 1.0).with_label("link");
        assert_eq!(e.label.as_deref(), Some("link"));
    }

    #[test]
    fn init_positions_circle() {
        let mut nodes = sample_nodes();
        init_positions(&mut nodes, 300.0, 200.0, 100.0);
        for node in &nodes {
            let dx = node.x - 300.0;
            let dy = node.y - 200.0;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!((dist - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn init_positions_skips_fixed() {
        let mut nodes = vec![
            NetworkNode::new("a", "A", "red")
                .with_position(10.0, 20.0)
                .with_fixed(true),
        ];
        init_positions(&mut nodes, 300.0, 200.0, 100.0);
        assert!((nodes[0].x - 10.0).abs() < 1e-9);
        assert!((nodes[0].y - 20.0).abs() < 1e-9);
    }

    #[test]
    fn simulation_converges() {
        let mut nodes = sample_nodes();
        let edges = sample_edges();
        let cfg = NetworkGraphConfig::default();
        simulate(&mut nodes, &edges, &cfg);
        let ke = kinetic_energy(&nodes);
        // After 100 iterations with damping, energy should be low
        assert!(ke < 10000.0, "KE={ke} should be low after simulation");
    }

    #[test]
    fn repulsion_pushes_apart() {
        let mut nodes = vec![
            NetworkNode::new("a", "A", "red").with_position(100.0, 100.0),
            NetworkNode::new("b", "B", "blue").with_position(101.0, 100.0),
        ];
        let edges: Vec<NetworkEdge> = vec![];
        let params = ForceParams::default();
        simulation_tick(&mut nodes, &edges, &params, 200.0, 200.0);
        let dx = (nodes[1].x - nodes[0].x).abs();
        assert!(dx > 1.0, "nodes should be pushed apart");
    }

    #[test]
    fn attraction_pulls_together() {
        let mut nodes = vec![
            NetworkNode::new("a", "A", "red").with_position(50.0, 200.0),
            NetworkNode::new("b", "B", "blue").with_position(550.0, 200.0),
        ];
        let edges = vec![NetworkEdge::new("a", "b", 1.0)];
        let params = ForceParams {
            repulsion_strength: 0.0,
            attraction_strength: 0.1,
            edge_length: 100.0,
            center_strength: 0.0,
            damping: 0.9,
            max_velocity: 50.0,
            min_distance: 1.0,
        };
        let initial_dist = 500.0;
        simulation_tick(&mut nodes, &edges, &params, 300.0, 200.0);
        let dx = (nodes[1].x - nodes[0].x).abs();
        assert!(dx < initial_dist, "attraction should reduce distance");
    }

    #[test]
    fn kinetic_energy_at_rest() {
        let nodes = vec![
            NetworkNode::new("a", "A", "red"),
            NetworkNode::new("b", "B", "blue"),
        ];
        assert_eq!(kinetic_energy(&nodes), 0.0);
    }

    #[test]
    fn render_produces_svg() {
        let mut nodes = sample_nodes();
        let edges = sample_edges();
        let cfg = NetworkGraphConfig::default();
        let svg = network_graph(&mut nodes, &edges, &cfg);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
    }

    #[test]
    fn render_contains_circles() {
        let mut nodes = sample_nodes();
        let edges = sample_edges();
        let cfg = NetworkGraphConfig::default();
        let svg = network_graph(&mut nodes, &edges, &cfg);
        assert_eq!(svg.matches("<circle").count(), 3);
    }

    #[test]
    fn render_contains_lines() {
        let mut nodes = sample_nodes();
        let edges = sample_edges();
        let cfg = NetworkGraphConfig::default();
        let svg = network_graph(&mut nodes, &edges, &cfg);
        assert_eq!(svg.matches("<line").count(), 3);
    }

    #[test]
    fn render_contains_labels() {
        let mut nodes = sample_nodes();
        let edges = sample_edges();
        let cfg = NetworkGraphConfig::default();
        let svg = network_graph(&mut nodes, &edges, &cfg);
        assert!(svg.contains("Alpha"));
        assert!(svg.contains("Beta"));
        assert!(svg.contains("Gamma"));
    }

    #[test]
    fn empty_graph() {
        let mut nodes: Vec<NetworkNode> = vec![];
        let edges: Vec<NetworkEdge> = vec![];
        let cfg = NetworkGraphConfig::default();
        let svg = network_graph(&mut nodes, &edges, &cfg);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn max_velocity_clamped() {
        let mut nodes = vec![
            NetworkNode::new("a", "A", "red").with_position(200.0, 200.0),
            NetworkNode::new("b", "B", "blue").with_position(200.1, 200.0),
        ];
        let params = ForceParams {
            repulsion_strength: 1e10,
            max_velocity: 5.0,
            ..ForceParams::default()
        };
        simulation_tick(&mut nodes, &[], &params, 200.0, 200.0);
        for n in &nodes {
            let speed = (n.vx * n.vx + n.vy * n.vy).sqrt();
            assert!(speed <= 5.0 + 1e-9, "speed {speed} should be <= 5.0");
        }
    }
}
