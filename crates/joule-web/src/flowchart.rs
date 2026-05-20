//! Flowchart builder with typed nodes, validation, and DSL parsing.
//!
//! Replaces mermaid / flowchart.js. Provides node types per ISO 5807,
//! typed connections (yes/no from decisions), validation, and auto-layout.
//! Pure Rust — no browser dependency.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Node types ───────────────────────────────────────────────────

/// Flowchart node types per ISO 5807.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Rounded rectangle — start or end.
    Start,
    /// Rounded rectangle — terminal.
    End,
    /// Rectangle — process step.
    Process,
    /// Diamond — decision (yes/no branches).
    Decision,
    /// Parallelogram — input/output.
    Io,
    /// Double-bordered rectangle — subroutine call.
    Subroutine,
    /// Small circle — on-page connector.
    Connector,
}

/// A node in the flowchart.
#[derive(Debug, Clone)]
pub struct FlowNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl FlowNode {
    pub fn new(id: impl Into<String>, kind: NodeKind, label: impl Into<String>) -> Self {
        let (w, h) = match kind {
            NodeKind::Connector => (40.0, 40.0),
            NodeKind::Decision => (120.0, 80.0),
            _ => (140.0, 60.0),
        };
        Self {
            id: id.into(),
            kind,
            label: label.into(),
            x: 0.0,
            y: 0.0,
            width: w,
            height: h,
        }
    }
}

/// Connection label for decision branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchLabel {
    Yes,
    No,
    /// Unlabeled connection.
    None,
}

/// A connection between flowchart nodes.
#[derive(Debug, Clone)]
pub struct FlowConnection {
    pub source: String,
    pub target: String,
    pub branch: BranchLabel,
}

// ── FlowchartBuilder ─────────────────────────────────────────────

/// Errors from flowchart validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowchartError {
    DuplicateNodeId(String),
    MissingNode(String),
    NoStartNode,
    NoEndNode,
    UnreachableEnd,
    DecisionMissingYes(String),
    DecisionMissingNo(String),
    CircularOnly,
}

impl std::fmt::Display for FlowchartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNodeId(id) => write!(f, "duplicate node id: {id}"),
            Self::MissingNode(id) => write!(f, "missing node: {id}"),
            Self::NoStartNode => write!(f, "no start node"),
            Self::NoEndNode => write!(f, "no end node"),
            Self::UnreachableEnd => write!(f, "end not reachable from start"),
            Self::DecisionMissingYes(id) => write!(f, "decision {id} missing yes branch"),
            Self::DecisionMissingNo(id) => write!(f, "decision {id} missing no branch"),
            Self::CircularOnly => write!(f, "graph has no terminal path"),
        }
    }
}

impl std::error::Error for FlowchartError {}

/// Builder for constructing flowcharts.
#[derive(Debug, Clone)]
pub struct Flowchart {
    nodes: Vec<FlowNode>,
    connections: Vec<FlowConnection>,
    node_map: HashMap<String, usize>,
}

impl Flowchart {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            connections: Vec::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: FlowNode) -> Result<(), FlowchartError> {
        if self.node_map.contains_key(&node.id) {
            return Err(FlowchartError::DuplicateNodeId(node.id.clone()));
        }
        let idx = self.nodes.len();
        self.node_map.insert(node.id.clone(), idx);
        self.nodes.push(node);
        Ok(())
    }

    pub fn connect(
        &mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        branch: BranchLabel,
    ) -> Result<(), FlowchartError> {
        let source = source.into();
        let target = target.into();
        if !self.node_map.contains_key(&source) {
            return Err(FlowchartError::MissingNode(source));
        }
        if !self.node_map.contains_key(&target) {
            return Err(FlowchartError::MissingNode(target));
        }
        self.connections.push(FlowConnection {
            source,
            target,
            branch,
        });
        Ok(())
    }

    pub fn nodes(&self) -> &[FlowNode] {
        &self.nodes
    }

    pub fn connections(&self) -> &[FlowConnection] {
        &self.connections
    }

    pub fn find_node(&self, id: &str) -> Option<&FlowNode> {
        self.node_map.get(id).map(|i| &self.nodes[*i])
    }

    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut FlowNode> {
        self.node_map.get(id).copied().map(|i| &mut self.nodes[i])
    }

    /// Validate the flowchart.
    pub fn validate(&self) -> Vec<FlowchartError> {
        let mut errors = Vec::new();

        let starts: Vec<&FlowNode> = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Start)
            .collect();
        let ends: Vec<&FlowNode> = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::End)
            .collect();

        if starts.is_empty() {
            errors.push(FlowchartError::NoStartNode);
        }
        if ends.is_empty() {
            errors.push(FlowchartError::NoEndNode);
        }

        // Check decisions have yes and no branches.
        for node in &self.nodes {
            if node.kind == NodeKind::Decision {
                let has_yes = self
                    .connections
                    .iter()
                    .any(|c| c.source == node.id && c.branch == BranchLabel::Yes);
                let has_no = self
                    .connections
                    .iter()
                    .any(|c| c.source == node.id && c.branch == BranchLabel::No);
                if !has_yes {
                    errors.push(FlowchartError::DecisionMissingYes(node.id.clone()));
                }
                if !has_no {
                    errors.push(FlowchartError::DecisionMissingNo(node.id.clone()));
                }
            }
        }

        // Check reachability: end reachable from start.
        if !starts.is_empty() && !ends.is_empty() {
            let adj = self.adjacency();
            let reachable = self.bfs_reachable(&starts[0].id, &adj);
            let end_reachable = ends.iter().any(|e| reachable.contains(e.id.as_str()));
            if !end_reachable {
                errors.push(FlowchartError::UnreachableEnd);
            }
        }

        errors
    }

    fn adjacency(&self) -> HashMap<&str, Vec<&str>> {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.nodes {
            adj.entry(node.id.as_str()).or_default();
        }
        for conn in &self.connections {
            adj.entry(conn.source.as_str())
                .or_default()
                .push(conn.target.as_str());
        }
        adj
    }

    fn bfs_reachable<'a>(
        &'a self,
        start: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
    ) -> HashSet<&'a str> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adj.get(current) {
                for neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        visited
    }

    /// Auto-layout: top-to-bottom, layered by BFS distance from start.
    pub fn auto_layout(&mut self, spacing_x: f64, spacing_y: f64) {
        let starts: Vec<String> = self
            .nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Start)
            .map(|n| n.id.clone())
            .collect();

        if starts.is_empty() {
            return;
        }

        let adj = self.adjacency();
        let mut layers: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<&str> = VecDeque::new();

        for start in &starts {
            layers.insert(start.clone(), 0);
            queue.push_back(start.as_str());
        }

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

        // Assign coordinates by layer.
        let max_layer = layers.values().copied().max().unwrap_or(0);
        let mut by_layer: Vec<Vec<String>> = vec![Vec::new(); max_layer + 1];
        for (id, layer) in &layers {
            by_layer[*layer].push(id.clone());
        }

        for (layer_idx, layer_nodes) in by_layer.iter().enumerate() {
            for (pos_idx, node_id) in layer_nodes.iter().enumerate() {
                if let Some(node) = self.find_node_mut(node_id) {
                    node.x = pos_idx as f64 * spacing_x;
                    node.y = layer_idx as f64 * spacing_y;
                }
            }
        }
    }
}

impl Default for Flowchart {
    fn default() -> Self {
        Self::new()
    }
}

// ── DSL Parser ───────────────────────────────────────────────────

/// Parse a simple flowchart DSL.
///
/// Format:
/// ```text
/// [start] Start
/// (process) Do something
/// {decision} Is it ok?
/// <io> Read input
/// [[subroutine]] Call sub
/// ((connector)) A
/// start -> process
/// decision -yes-> io
/// decision -no-> process
/// ```
pub fn parse_dsl(input: &str) -> Result<Flowchart, FlowchartError> {
    let mut fc = Flowchart::new();
    let mut pending_connections: Vec<(String, String, BranchLabel)> = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.contains("->") {
            // Connection line.
            let parts: Vec<&str> = if line.contains("-yes->") {
                let split: Vec<&str> = line.splitn(2, "-yes->").collect();
                pending_connections.push((
                    split[0].trim().to_string(),
                    split[1].trim().to_string(),
                    BranchLabel::Yes,
                ));
                continue;
            } else if line.contains("-no->") {
                let split: Vec<&str> = line.splitn(2, "-no->").collect();
                pending_connections.push((
                    split[0].trim().to_string(),
                    split[1].trim().to_string(),
                    BranchLabel::No,
                ));
                continue;
            } else {
                line.splitn(2, "->").collect()
            };
            pending_connections.push((
                parts[0].trim().to_string(),
                parts[1].trim().to_string(),
                BranchLabel::None,
            ));
        } else if line.starts_with('[') && !line.starts_with("[[") {
            // Start/end node: [id] label
            if let Some(close) = line.find(']') {
                let id = &line[1..close];
                let label = line[close + 1..].trim();
                let kind = if label.to_lowercase().contains("end") {
                    NodeKind::End
                } else {
                    NodeKind::Start
                };
                fc.add_node(FlowNode::new(id, kind, label))?;
            }
        } else if line.starts_with("[[") {
            // Subroutine: [[id]] label
            if let Some(close) = line.find("]]") {
                let id = &line[2..close];
                let label = line[close + 2..].trim();
                fc.add_node(FlowNode::new(id, NodeKind::Subroutine, label))?;
            }
        } else if line.starts_with("((") {
            // Connector: ((id)) label
            if let Some(close) = line.find("))") {
                let id = &line[2..close];
                let label = line[close + 2..].trim();
                fc.add_node(FlowNode::new(id, NodeKind::Connector, label))?;
            }
        } else if line.starts_with('(') {
            // Process: (id) label
            if let Some(close) = line.find(')') {
                let id = &line[1..close];
                let label = line[close + 1..].trim();
                fc.add_node(FlowNode::new(id, NodeKind::Process, label))?;
            }
        } else if line.starts_with('{') {
            // Decision: {id} label
            if let Some(close) = line.find('}') {
                let id = &line[1..close];
                let label = line[close + 1..].trim();
                fc.add_node(FlowNode::new(id, NodeKind::Decision, label))?;
            }
        } else if line.starts_with('<') {
            // IO: <id> label
            if let Some(close) = line.find('>') {
                let id = &line[1..close];
                let label = line[close + 1..].trim();
                fc.add_node(FlowNode::new(id, NodeKind::Io, label))?;
            }
        }
    }

    for (source, target, branch) in pending_connections {
        fc.connect(&source, &target, branch)?;
    }

    Ok(fc)
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_flowchart() -> Flowchart {
        let mut fc = Flowchart::new();
        fc.add_node(FlowNode::new("start", NodeKind::Start, "Begin")).unwrap();
        fc.add_node(FlowNode::new("proc", NodeKind::Process, "Do work")).unwrap();
        fc.add_node(FlowNode::new("dec", NodeKind::Decision, "OK?")).unwrap();
        fc.add_node(FlowNode::new("end", NodeKind::End, "Done")).unwrap();
        fc.connect("start", "proc", BranchLabel::None).unwrap();
        fc.connect("proc", "dec", BranchLabel::None).unwrap();
        fc.connect("dec", "end", BranchLabel::Yes).unwrap();
        fc.connect("dec", "proc", BranchLabel::No).unwrap();
        fc
    }

    #[test]
    fn test_build_flowchart() {
        let fc = simple_flowchart();
        assert_eq!(fc.nodes().len(), 4);
        assert_eq!(fc.connections().len(), 4);
    }

    #[test]
    fn test_duplicate_node_id() {
        let mut fc = Flowchart::new();
        fc.add_node(FlowNode::new("a", NodeKind::Process, "A")).unwrap();
        let result = fc.add_node(FlowNode::new("a", NodeKind::Process, "B"));
        assert!(matches!(result, Err(FlowchartError::DuplicateNodeId(_))));
    }

    #[test]
    fn test_missing_node_connect() {
        let mut fc = Flowchart::new();
        fc.add_node(FlowNode::new("a", NodeKind::Process, "A")).unwrap();
        let result = fc.connect("a", "b", BranchLabel::None);
        assert!(matches!(result, Err(FlowchartError::MissingNode(_))));
    }

    #[test]
    fn test_validate_valid() {
        let fc = simple_flowchart();
        let errors = fc.validate();
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_validate_no_start() {
        let mut fc = Flowchart::new();
        fc.add_node(FlowNode::new("end", NodeKind::End, "End")).unwrap();
        let errors = fc.validate();
        assert!(errors.iter().any(|e| matches!(e, FlowchartError::NoStartNode)));
    }

    #[test]
    fn test_validate_decision_branches() {
        let mut fc = Flowchart::new();
        fc.add_node(FlowNode::new("start", NodeKind::Start, "S")).unwrap();
        fc.add_node(FlowNode::new("dec", NodeKind::Decision, "?")).unwrap();
        fc.add_node(FlowNode::new("end", NodeKind::End, "E")).unwrap();
        fc.connect("start", "dec", BranchLabel::None).unwrap();
        fc.connect("dec", "end", BranchLabel::Yes).unwrap();
        // Missing No branch.
        let errors = fc.validate();
        assert!(errors
            .iter()
            .any(|e| matches!(e, FlowchartError::DecisionMissingNo(_))));
    }

    #[test]
    fn test_auto_layout() {
        let mut fc = simple_flowchart();
        fc.auto_layout(160.0, 100.0);
        let start = fc.find_node("start").unwrap();
        let proc = fc.find_node("proc").unwrap();
        assert!(proc.y > start.y);
    }

    #[test]
    fn test_parse_dsl_basic() {
        let dsl = r#"
[start] Begin
(proc) Process data
{dec} Valid?
[end] End
start -> proc
proc -> dec
dec -yes-> end
dec -no-> proc
"#;
        let fc = parse_dsl(dsl).unwrap();
        assert_eq!(fc.nodes().len(), 4);
        assert_eq!(fc.connections().len(), 4);
        assert!(fc.validate().is_empty());
    }

    #[test]
    fn test_parse_dsl_all_types() {
        let dsl = r#"
[s] Start
(p) Process
{d} Decision
<io> Input
[[sub]] Call sub
((conn)) A
[e] End
s -> p
p -> d
d -yes-> io
d -no-> sub
io -> conn
sub -> conn
conn -> e
"#;
        let fc = parse_dsl(dsl).unwrap();
        assert_eq!(fc.nodes().len(), 7);
        assert_eq!(fc.find_node("io").unwrap().kind, NodeKind::Io);
        assert_eq!(fc.find_node("sub").unwrap().kind, NodeKind::Subroutine);
        assert_eq!(fc.find_node("conn").unwrap().kind, NodeKind::Connector);
    }

    #[test]
    fn test_validate_unreachable_end() {
        let mut fc = Flowchart::new();
        fc.add_node(FlowNode::new("start", NodeKind::Start, "S")).unwrap();
        fc.add_node(FlowNode::new("proc", NodeKind::Process, "P")).unwrap();
        fc.add_node(FlowNode::new("end", NodeKind::End, "E")).unwrap();
        fc.connect("start", "proc", BranchLabel::None).unwrap();
        // No connection to end.
        let errors = fc.validate();
        assert!(errors
            .iter()
            .any(|e| matches!(e, FlowchartError::UnreachableEnd)));
    }

    #[test]
    fn test_node_dimensions_by_kind() {
        let decision = FlowNode::new("d", NodeKind::Decision, "?");
        let connector = FlowNode::new("c", NodeKind::Connector, "A");
        let process = FlowNode::new("p", NodeKind::Process, "Do");
        assert_eq!(connector.width, 40.0);
        assert_eq!(decision.width, 120.0);
        assert_eq!(process.width, 140.0);
    }
}
