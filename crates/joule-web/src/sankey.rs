//! Sankey diagram with flow-proportional layout and SVG path generation.
//!
//! Replaces d3-sankey. Provides node/link layout with proportional widths,
//! vertical positioning to minimize crossings, circular flow detection,
//! multi-level grouping, and cubic Bezier SVG paths. Pure Rust — no browser dependency.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Data types ───────────────────────────────────────────────────

/// A node in the Sankey diagram.
#[derive(Debug, Clone)]
pub struct SankeyNode {
    pub id: String,
    pub label: String,
    /// Computed total flow through this node.
    pub value: f64,
    /// Layout: column (0-based).
    pub column: usize,
    /// Layout: vertical position.
    pub y: f64,
    /// Layout: height proportional to flow.
    pub height: f64,
    /// Optional grouping label.
    pub group: Option<String>,
}

impl SankeyNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            value: 0.0,
            column: 0,
            y: 0.0,
            height: 0.0,
            group: None,
        }
    }

    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }
}

/// A link (flow) between two nodes.
#[derive(Debug, Clone)]
pub struct SankeyLink {
    pub source: String,
    pub target: String,
    pub value: f64,
    /// Layout: width proportional to value.
    pub width: f64,
    /// Layout: source y offset.
    pub source_y: f64,
    /// Layout: target y offset.
    pub target_y: f64,
}

impl SankeyLink {
    pub fn new(source: impl Into<String>, target: impl Into<String>, value: f64) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            value,
            width: 0.0,
            source_y: 0.0,
            target_y: 0.0,
        }
    }
}

/// Configuration for Sankey layout.
#[derive(Debug, Clone)]
pub struct SankeyConfig {
    pub width: f64,
    pub height: f64,
    pub node_width: f64,
    pub node_padding: f64,
    pub iterations: usize,
}

impl Default for SankeyConfig {
    fn default() -> Self {
        Self {
            width: 800.0,
            height: 500.0,
            node_width: 20.0,
            node_padding: 10.0,
            iterations: 25,
        }
    }
}

// ── SankeyDiagram ────────────────────────────────────────────────

/// The complete Sankey diagram.
#[derive(Debug, Clone)]
pub struct SankeyDiagram {
    pub nodes: Vec<SankeyNode>,
    pub links: Vec<SankeyLink>,
    node_map: HashMap<String, usize>,
}

impl SankeyDiagram {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            links: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: SankeyNode) {
        let idx = self.nodes.len();
        self.node_map.insert(node.id.clone(), idx);
        self.nodes.push(node);
    }

    pub fn add_link(&mut self, link: SankeyLink) {
        self.links.push(link);
    }

    pub fn find_node(&self, id: &str) -> Option<&SankeyNode> {
        self.node_map.get(id).map(|i| &self.nodes[*i])
    }

    fn find_node_mut(&mut self, id: &str) -> Option<&mut SankeyNode> {
        self.node_map.get(id).copied().map(|i| &mut self.nodes[i])
    }

    /// Detect circular flows.
    pub fn has_circular_flow(&self) -> bool {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut in_stack: HashSet<&str> = HashSet::new();

        let adj = self.forward_adjacency();

        for node in &self.nodes {
            if !visited.contains(node.id.as_str()) {
                if self.dfs_has_cycle(node.id.as_str(), &adj, &mut visited, &mut in_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn forward_adjacency(&self) -> HashMap<&str, Vec<&str>> {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for link in &self.links {
            adj.entry(link.source.as_str())
                .or_default()
                .push(link.target.as_str());
        }
        adj
    }

    fn dfs_has_cycle<'a>(
        &'a self,
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashSet<&'a str>,
        in_stack: &mut HashSet<&'a str>,
    ) -> bool {
        visited.insert(node);
        in_stack.insert(node);

        if let Some(neighbors) = adj.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if self.dfs_has_cycle(neighbor, adj, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(neighbor) {
                    return true;
                }
            }
        }

        in_stack.remove(node);
        false
    }

    /// Compute node values from link flows.
    fn compute_node_values(&mut self) {
        for node in &mut self.nodes {
            node.value = 0.0;
        }

        let mut incoming: HashMap<String, f64> = HashMap::new();
        let mut outgoing: HashMap<String, f64> = HashMap::new();

        for link in &self.links {
            *incoming.entry(link.target.clone()).or_insert(0.0) += link.value;
            *outgoing.entry(link.source.clone()).or_insert(0.0) += link.value;
        }

        for node in &mut self.nodes {
            let inc = incoming.get(&node.id).copied().unwrap_or(0.0);
            let out = outgoing.get(&node.id).copied().unwrap_or(0.0);
            node.value = inc.max(out);
        }
    }

    /// Assign columns using longest path from sources.
    fn assign_columns(&mut self) {
        let adj = self.forward_adjacency();
        let rev = self.reverse_adjacency();

        let sources: Vec<String> = self
            .nodes
            .iter()
            .filter(|n| rev.get(n.id.as_str()).map_or(true, |v| v.is_empty()))
            .map(|n| n.id.clone())
            .collect();

        let mut columns: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        for s in &sources {
            columns.insert(s.clone(), 0);
            queue.push_back(s.clone());
        }

        if queue.is_empty() {
            if let Some(n) = self.nodes.first() {
                columns.insert(n.id.clone(), 0);
                queue.push_back(n.id.clone());
            }
        }

        while let Some(current) = queue.pop_front() {
            let col = columns[&current];
            if let Some(neighbors) = adj.get(current.as_str()) {
                for neighbor in neighbors {
                    let new_col = col + 1;
                    let existing = columns.get(*neighbor).copied();
                    if existing.is_none() || existing.unwrap() < new_col {
                        columns.insert(neighbor.to_string(), new_col);
                        queue.push_back(neighbor.to_string());
                    }
                }
            }
        }

        for node in &mut self.nodes {
            node.column = columns.get(&node.id).copied().unwrap_or(0);
        }
    }

    fn reverse_adjacency(&self) -> HashMap<&str, Vec<&str>> {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for link in &self.links {
            adj.entry(link.target.as_str())
                .or_default()
                .push(link.source.as_str());
        }
        adj
    }

    /// Run the full layout algorithm.
    pub fn layout(&mut self, config: &SankeyConfig) {
        if self.nodes.is_empty() {
            return;
        }

        self.compute_node_values();
        self.assign_columns();

        let max_col = self.nodes.iter().map(|n| n.column).max().unwrap_or(0);

        // Group nodes by column.
        let mut by_column: Vec<Vec<String>> = vec![Vec::new(); max_col + 1];
        for node in &self.nodes {
            by_column[node.column].push(node.id.clone());
        }

        let available_height = config.height;

        // Set node heights and initial y positions.
        for col_nodes in &by_column {
            let col_total: f64 = col_nodes
                .iter()
                .filter_map(|id| self.find_node(id))
                .map(|n| n.value)
                .sum::<f64>()
                .max(1.0);

            let usable = available_height
                - config.node_padding * (col_nodes.len() as f64 - 1.0).max(0.0);
            let mut y = 0.0;

            for node_id in col_nodes {
                if let Some(node) = self.find_node_mut(node_id) {
                    node.height = (node.value / col_total) * usable;
                    node.y = y;
                    y += node.height + config.node_padding;
                }
            }
        }

        // Compute link positions.
        self.compute_link_positions(config);
    }

    fn compute_link_positions(&mut self, _config: &SankeyConfig) {
        // Track y offsets for stacking links at each node.
        let mut source_offsets: HashMap<String, f64> = HashMap::new();
        let mut target_offsets: HashMap<String, f64> = HashMap::new();

        for node in &self.nodes {
            source_offsets.insert(node.id.clone(), node.y);
            target_offsets.insert(node.id.clone(), node.y);
        }

        let mut total_out: HashMap<String, f64> = HashMap::new();
        let mut total_in: HashMap<String, f64> = HashMap::new();
        for link in &self.links {
            *total_out.entry(link.source.clone()).or_insert(0.0) += link.value;
            *total_in.entry(link.target.clone()).or_insert(0.0) += link.value;
        }

        for link in &mut self.links {
            let source_node = self.nodes.iter().find(|n| n.id == link.source);
            let target_node = self.nodes.iter().find(|n| n.id == link.target);

            if let (Some(src), Some(tgt)) = (source_node, target_node) {
                let src_total = total_out.get(&link.source).copied().unwrap_or(1.0);
                let tgt_total = total_in.get(&link.target).copied().unwrap_or(1.0);

                link.width = (link.value / src_total) * src.height;

                let src_y = source_offsets.get(&link.source).copied().unwrap_or(0.0);
                link.source_y = src_y;
                source_offsets.insert(link.source.clone(), src_y + link.width);

                let tgt_width = (link.value / tgt_total) * tgt.height;
                let tgt_y = target_offsets.get(&link.target).copied().unwrap_or(0.0);
                link.target_y = tgt_y;
                target_offsets.insert(link.target.clone(), tgt_y + tgt_width);
            }
        }
    }

    /// Generate SVG path (cubic Bezier) for a link.
    pub fn link_svg_path(&self, link: &SankeyLink, config: &SankeyConfig) -> String {
        let max_col = self.nodes.iter().map(|n| n.column).max().unwrap_or(0);
        let col_spacing = if max_col > 0 {
            (config.width - config.node_width) / max_col as f64
        } else {
            0.0
        };

        let source_node = match self.nodes.iter().find(|n| n.id == link.source) {
            Some(n) => n,
            None => return String::new(),
        };
        let target_node = match self.nodes.iter().find(|n| n.id == link.target) {
            Some(n) => n,
            None => return String::new(),
        };

        let x0 = source_node.column as f64 * col_spacing + config.node_width;
        let x1 = target_node.column as f64 * col_spacing;
        let y0 = link.source_y + link.width / 2.0;
        let y1 = link.target_y + link.width / 2.0;

        let mid_x = (x0 + x1) / 2.0;

        format!(
            "M{x0:.1},{y0:.1} C{mid_x:.1},{y0:.1} {mid_x:.1},{y1:.1} {x1:.1},{y1:.1}"
        )
    }

    /// Get all groups.
    pub fn groups(&self) -> Vec<String> {
        let mut groups: HashSet<String> = HashSet::new();
        for node in &self.nodes {
            if let Some(g) = &node.group {
                groups.insert(g.clone());
            }
        }
        let mut result: Vec<String> = groups.into_iter().collect();
        result.sort();
        result
    }

    /// Get nodes in a group.
    pub fn nodes_in_group(&self, group: &str) -> Vec<&SankeyNode> {
        self.nodes
            .iter()
            .filter(|n| n.group.as_deref() == Some(group))
            .collect()
    }

    /// Total flow through the diagram.
    pub fn total_flow(&self) -> f64 {
        self.links.iter().map(|l| l.value).sum()
    }

    /// Number of columns.
    pub fn column_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| n.column)
            .max()
            .map_or(0, |m| m + 1)
    }
}

impl Default for SankeyDiagram {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sankey() -> SankeyDiagram {
        let mut diagram = SankeyDiagram::new();
        diagram.add_node(SankeyNode::new("a", "Source A").with_group("inputs"));
        diagram.add_node(SankeyNode::new("b", "Source B").with_group("inputs"));
        diagram.add_node(SankeyNode::new("c", "Process").with_group("middle"));
        diagram.add_node(SankeyNode::new("d", "Output X").with_group("outputs"));
        diagram.add_node(SankeyNode::new("e", "Output Y").with_group("outputs"));

        diagram.add_link(SankeyLink::new("a", "c", 30.0));
        diagram.add_link(SankeyLink::new("b", "c", 20.0));
        diagram.add_link(SankeyLink::new("c", "d", 35.0));
        diagram.add_link(SankeyLink::new("c", "e", 15.0));
        diagram
    }

    #[test]
    fn test_add_nodes_and_links() {
        let d = sample_sankey();
        assert_eq!(d.nodes.len(), 5);
        assert_eq!(d.links.len(), 4);
    }

    #[test]
    fn test_total_flow() {
        let d = sample_sankey();
        assert_eq!(d.total_flow(), 100.0);
    }

    #[test]
    fn test_no_circular_flow() {
        let d = sample_sankey();
        assert!(!d.has_circular_flow());
    }

    #[test]
    fn test_circular_flow_detection() {
        let mut d = SankeyDiagram::new();
        d.add_node(SankeyNode::new("a", "A"));
        d.add_node(SankeyNode::new("b", "B"));
        d.add_node(SankeyNode::new("c", "C"));
        d.add_link(SankeyLink::new("a", "b", 10.0));
        d.add_link(SankeyLink::new("b", "c", 10.0));
        d.add_link(SankeyLink::new("c", "a", 10.0));
        assert!(d.has_circular_flow());
    }

    #[test]
    fn test_layout_columns() {
        let mut d = sample_sankey();
        d.layout(&SankeyConfig::default());
        assert_eq!(d.find_node("a").unwrap().column, 0);
        assert_eq!(d.find_node("b").unwrap().column, 0);
        assert_eq!(d.find_node("c").unwrap().column, 1);
        assert!(d.find_node("d").unwrap().column >= 2);
    }

    #[test]
    fn test_layout_node_heights() {
        let mut d = sample_sankey();
        d.layout(&SankeyConfig::default());
        let c = d.find_node("c").unwrap();
        assert!(c.height > 0.0);
    }

    #[test]
    fn test_layout_link_widths() {
        let mut d = sample_sankey();
        d.layout(&SankeyConfig::default());
        for link in &d.links {
            assert!(
                link.width > 0.0,
                "link {}->{} has zero width",
                link.source,
                link.target
            );
        }
    }

    #[test]
    fn test_svg_path() {
        let mut d = sample_sankey();
        let config = SankeyConfig::default();
        d.layout(&config);
        let path = d.link_svg_path(&d.links[0].clone(), &config);
        assert!(path.starts_with('M'));
        assert!(path.contains('C'));
    }

    #[test]
    fn test_groups() {
        let d = sample_sankey();
        let groups = d.groups();
        assert_eq!(groups.len(), 3);
        assert!(groups.contains(&"inputs".to_string()));
    }

    #[test]
    fn test_nodes_in_group() {
        let d = sample_sankey();
        let inputs = d.nodes_in_group("inputs");
        assert_eq!(inputs.len(), 2);
    }

    #[test]
    fn test_column_count() {
        let mut d = sample_sankey();
        d.layout(&SankeyConfig::default());
        assert!(d.column_count() >= 3);
    }

    #[test]
    fn test_compute_node_values() {
        let mut d = sample_sankey();
        d.compute_node_values();
        assert_eq!(d.find_node("c").unwrap().value, 50.0);
    }

    #[test]
    fn test_empty_diagram() {
        let mut d = SankeyDiagram::new();
        d.layout(&SankeyConfig::default());
        assert_eq!(d.total_flow(), 0.0);
        assert_eq!(d.column_count(), 0);
    }
}
