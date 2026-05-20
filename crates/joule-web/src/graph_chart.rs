//! Network/relationship chart — nodes, edges, circular layout, and SVG rendering.
//!
//! Models a graph for visualization with node sizing, edge width, category colors,
//! adjacency queries, degree computation, and SVG circle+line output.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Core Types ──────────────────────────────────────────────────

/// A node in the graph chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartNode {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub category: Option<String>,
    pub x: f64,
    pub y: f64,
}

impl ChartNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>, value: f64) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            value,
            category: None,
            x: 0.0,
            y: 0.0,
        }
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.x = x;
        self.y = y;
        self
    }
}

/// An edge in the graph chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartEdge {
    pub source_id: String,
    pub target_id: String,
    pub value: f64,
    pub label: Option<String>,
}

impl ChartEdge {
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            source_id: source.into(),
            target_id: target.into(),
            value,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// Tooltip data extracted for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTooltip {
    pub id: String,
    pub label: String,
    pub value: f64,
    pub category: Option<String>,
    pub degree: usize,
    pub neighbors: Vec<String>,
}

// ── Graph Chart ─────────────────────────────────────────────────

/// A network chart for visualization.
#[derive(Debug, Clone)]
pub struct GraphChart {
    pub nodes: Vec<ChartNode>,
    pub edges: Vec<ChartEdge>,
    /// Category name -> color mapping.
    pub category_colors: HashMap<String, String>,
    /// Min/max node radius for sizing.
    pub min_radius: f64,
    pub max_radius: f64,
    /// Min/max edge width for sizing.
    pub min_edge_width: f64,
    pub max_edge_width: f64,
}

impl GraphChart {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            category_colors: HashMap::new(),
            min_radius: 5.0,
            max_radius: 30.0,
            min_edge_width: 1.0,
            max_edge_width: 8.0,
        }
    }

    /// Add a node.
    pub fn add_node(&mut self, node: ChartNode) {
        self.nodes.push(node);
    }

    /// Add an edge.
    pub fn add_edge(&mut self, edge: ChartEdge) {
        self.edges.push(edge);
    }

    /// Assign a color to a category.
    pub fn set_category_color(&mut self, category: impl Into<String>, color: impl Into<String>) {
        self.category_colors.insert(category.into(), color.into());
    }

    /// Get color for a node's category.
    pub fn node_color(&self, node: &ChartNode) -> &str {
        node.category
            .as_ref()
            .and_then(|cat| self.category_colors.get(cat))
            .map(|s| s.as_str())
            .unwrap_or("#666666")
    }

    /// Find a node by id.
    pub fn get_node(&self, id: &str) -> Option<&ChartNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    // ── Layout ──────────────────────────────────────────────────

    /// Arrange nodes in a circular layout.
    pub fn circular_layout(&mut self, center_x: f64, center_y: f64, radius: f64) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
            node.x = center_x + radius * angle.cos();
            node.y = center_y + radius * angle.sin();
        }
    }

    // ── Sizing ──────────────────────────────────────────────────

    /// Compute node radius scaled by value.
    pub fn node_radius(&self, node: &ChartNode) -> f64 {
        let (vmin, vmax) = self.value_range();
        if (vmax - vmin).abs() < f64::EPSILON {
            return (self.min_radius + self.max_radius) / 2.0;
        }
        let t = (node.value - vmin) / (vmax - vmin);
        self.min_radius + t * (self.max_radius - self.min_radius)
    }

    /// Compute edge width scaled by value.
    pub fn edge_width(&self, edge: &ChartEdge) -> f64 {
        let (emin, emax) = self.edge_value_range();
        if (emax - emin).abs() < f64::EPSILON {
            return (self.min_edge_width + self.max_edge_width) / 2.0;
        }
        let t = (edge.value - emin) / (emax - emin);
        self.min_edge_width + t * (self.max_edge_width - self.min_edge_width)
    }

    fn value_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for node in &self.nodes {
            if node.value < min {
                min = node.value;
            }
            if node.value > max {
                max = node.value;
            }
        }
        if min.is_infinite() {
            (0.0, 0.0)
        } else {
            (min, max)
        }
    }

    fn edge_value_range(&self) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for edge in &self.edges {
            if edge.value < min {
                min = edge.value;
            }
            if edge.value > max {
                max = edge.value;
            }
        }
        if min.is_infinite() {
            (0.0, 0.0)
        } else {
            (min, max)
        }
    }

    // ── Graph Queries ───────────────────────────────────────────

    /// Get neighbors of a node.
    pub fn neighbors(&self, node_id: &str) -> Vec<&str> {
        let mut result = Vec::new();
        for edge in &self.edges {
            if edge.source_id == node_id {
                result.push(edge.target_id.as_str());
            } else if edge.target_id == node_id {
                result.push(edge.source_id.as_str());
            }
        }
        result
    }

    /// Adjacency check.
    pub fn are_adjacent(&self, a: &str, b: &str) -> bool {
        self.edges.iter().any(|e| {
            (e.source_id == a && e.target_id == b)
                || (e.source_id == b && e.target_id == a)
        })
    }

    /// Compute degree (number of edges) for a node.
    pub fn degree(&self, node_id: &str) -> usize {
        self.edges
            .iter()
            .filter(|e| e.source_id == node_id || e.target_id == node_id)
            .count()
    }

    /// Compute degrees for all nodes.
    pub fn all_degrees(&self) -> HashMap<&str, usize> {
        let mut degrees: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            degrees.insert(node.id.as_str(), 0);
        }
        for edge in &self.edges {
            *degrees.entry(edge.source_id.as_str()).or_insert(0) += 1;
            *degrees.entry(edge.target_id.as_str()).or_insert(0) += 1;
        }
        degrees
    }

    /// Extract tooltip data for a node.
    pub fn tooltip(&self, node_id: &str) -> Option<NodeTooltip> {
        let node = self.get_node(node_id)?;
        let neighbors: Vec<String> = self
            .neighbors(node_id)
            .iter()
            .map(|s| s.to_string())
            .collect();
        Some(NodeTooltip {
            id: node.id.clone(),
            label: node.label.clone(),
            value: node.value,
            category: node.category.clone(),
            degree: self.degree(node_id),
            neighbors,
        })
    }

    // ── SVG ─────────────────────────────────────────────────────

    /// Generate SVG representation.
    pub fn to_svg(&self, width: f64, height: f64) -> String {
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">"#
        );

        // Build node position lookup
        let positions: HashMap<&str, (f64, f64)> = self
            .nodes
            .iter()
            .map(|n| (n.id.as_str(), (n.x, n.y)))
            .collect();

        // Draw edges
        for edge in &self.edges {
            if let (Some(&(x1, y1)), Some(&(x2, y2))) = (
                positions.get(edge.source_id.as_str()),
                positions.get(edge.target_id.as_str()),
            ) {
                let w = self.edge_width(edge);
                svg.push_str(&format!(
                    r##"<line x1="{x1:.1}" y1="{y1:.1}" x2="{x2:.1}" y2="{y2:.1}" stroke="#999" stroke-width="{w:.1}" opacity="0.6"/>"##
                ));
            }
        }

        // Draw nodes
        for node in &self.nodes {
            let r = self.node_radius(node);
            let color = self.node_color(node);
            svg.push_str(&format!(
                r##"<circle cx="{:.1}" cy="{:.1}" r="{r:.1}" fill="{color}" stroke="#fff" stroke-width="2"/>"##,
                node.x, node.y
            ));
            svg.push_str(&format!(
                r#"<text x="{:.1}" y="{:.1}" text-anchor="middle" font-size="10" dy="4">{}</text>"#,
                node.x, node.y, node.label
            ));
        }

        svg.push_str("</svg>");
        svg
    }
}

impl Default for GraphChart {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chart() -> GraphChart {
        let mut chart = GraphChart::new();
        chart.add_node(ChartNode::new("a", "Alice", 10.0).with_category("team1"));
        chart.add_node(ChartNode::new("b", "Bob", 20.0).with_category("team1"));
        chart.add_node(ChartNode::new("c", "Carol", 30.0).with_category("team2"));
        chart.add_node(ChartNode::new("d", "Dave", 5.0).with_category("team2"));
        chart.add_edge(ChartEdge::new("a", "b", 1.0));
        chart.add_edge(ChartEdge::new("a", "c", 2.0));
        chart.add_edge(ChartEdge::new("b", "c", 3.0));
        chart.add_edge(ChartEdge::new("c", "d", 1.5));
        chart.set_category_color("team1", "#3498db");
        chart.set_category_color("team2", "#e74c3c");
        chart
    }

    #[test]
    fn circular_layout_positions() {
        let mut chart = sample_chart();
        chart.circular_layout(200.0, 200.0, 100.0);
        // First node should be at top (angle = -PI/2)
        assert!((chart.nodes[0].x - 200.0).abs() < 0.01);
        assert!((chart.nodes[0].y - 100.0).abs() < 0.01);
    }

    #[test]
    fn node_radius_scaling() {
        let chart = sample_chart();
        let r_min = chart.node_radius(&chart.nodes[3]); // value=5, min
        let r_max = chart.node_radius(&chart.nodes[2]); // value=30, max
        assert!((r_min - chart.min_radius).abs() < 0.01);
        assert!((r_max - chart.max_radius).abs() < 0.01);
    }

    #[test]
    fn edge_width_scaling() {
        let chart = sample_chart();
        let w_min = chart.edge_width(&chart.edges[0]); // value=1.0
        let w_max = chart.edge_width(&chart.edges[2]); // value=3.0
        assert!((w_min - chart.min_edge_width).abs() < 0.01);
        assert!((w_max - chart.max_edge_width).abs() < 0.01);
    }

    #[test]
    fn neighbors() {
        let chart = sample_chart();
        let mut n = chart.neighbors("a");
        n.sort();
        assert_eq!(n, vec!["b", "c"]);
    }

    #[test]
    fn adjacency() {
        let chart = sample_chart();
        assert!(chart.are_adjacent("a", "b"));
        assert!(chart.are_adjacent("b", "a")); // symmetric
        assert!(!chart.are_adjacent("a", "d"));
    }

    #[test]
    fn degree() {
        let chart = sample_chart();
        assert_eq!(chart.degree("a"), 2);
        assert_eq!(chart.degree("c"), 3);
        assert_eq!(chart.degree("d"), 1);
    }

    #[test]
    fn all_degrees() {
        let chart = sample_chart();
        let degrees = chart.all_degrees();
        assert_eq!(*degrees.get("a").unwrap(), 2);
        assert_eq!(*degrees.get("c").unwrap(), 3);
    }

    #[test]
    fn category_colors() {
        let chart = sample_chart();
        assert_eq!(chart.node_color(&chart.nodes[0]), "#3498db");
        assert_eq!(chart.node_color(&chart.nodes[2]), "#e74c3c");
    }

    #[test]
    fn default_color_no_category() {
        let chart = GraphChart::new();
        let node = ChartNode::new("x", "X", 1.0);
        assert_eq!(chart.node_color(&node), "#666666");
    }

    #[test]
    fn tooltip_data() {
        let chart = sample_chart();
        let tip = chart.tooltip("c").unwrap();
        assert_eq!(tip.label, "Carol");
        assert_eq!(tip.degree, 3);
        assert_eq!(tip.neighbors.len(), 3);
    }

    #[test]
    fn tooltip_not_found() {
        let chart = sample_chart();
        assert!(chart.tooltip("nonexistent").is_none());
    }

    #[test]
    fn svg_output() {
        let mut chart = sample_chart();
        chart.circular_layout(200.0, 200.0, 100.0);
        let svg = chart.to_svg(400.0, 400.0);
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("circle"));
        assert!(svg.contains("line"));
        assert!(svg.contains("Alice"));
        assert!(svg.contains("#3498db"));
    }

    #[test]
    fn edge_with_label() {
        let edge = ChartEdge::new("a", "b", 1.0).with_label("works with");
        assert_eq!(edge.label.as_deref(), Some("works with"));
    }
}
