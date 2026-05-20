//! Organizational chart with tree layout, headcount rollup, and reorg ops.
//!
//! Replaces OrgChart.js / Google org chart. Provides top-down tree layout,
//! span of control, reporting chain, department extraction, and reorg operations.
//! Pure Rust — no browser dependency.

use std::collections::{HashMap, VecDeque};

// ── Data types ───────────────────────────────────────────────────

/// A person/role in the organization.
#[derive(Debug, Clone)]
pub struct OrgNode {
    pub id: String,
    pub name: String,
    pub title: String,
    pub department: String,
    pub reports_to: Option<String>,
}

impl OrgNode {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        title: impl Into<String>,
        department: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            title: title.into(),
            department: department.into(),
            reports_to: None,
        }
    }

    pub fn with_reports_to(mut self, manager_id: impl Into<String>) -> Self {
        self.reports_to = Some(manager_id.into());
        self
    }
}

/// Layout position for a node.
#[derive(Debug, Clone, Copy)]
pub struct OrgPosition {
    pub x: f64,
    pub y: f64,
}

/// Errors for org chart operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrgError {
    DuplicateId(String),
    NodeNotFound(String),
    WouldCreateCycle,
    CannotMoveRoot,
}

impl std::fmt::Display for OrgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "duplicate id: {id}"),
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::WouldCreateCycle => write!(f, "operation would create a cycle"),
            Self::CannotMoveRoot => write!(f, "cannot move root node"),
        }
    }
}

impl std::error::Error for OrgError {}

// ── OrgChart ─────────────────────────────────────────────────────

/// The organizational chart.
#[derive(Debug, Clone)]
pub struct OrgChart {
    nodes: Vec<OrgNode>,
    node_map: HashMap<String, usize>,
    positions: HashMap<String, OrgPosition>,
}

impl OrgChart {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_map: HashMap::new(),
            positions: HashMap::new(),
        }
    }

    /// Build from a flat list of nodes.
    pub fn from_nodes(nodes: Vec<OrgNode>) -> Result<Self, OrgError> {
        let mut chart = Self::new();
        for node in nodes {
            chart.add_node(node)?;
        }
        Ok(chart)
    }

    pub fn add_node(&mut self, node: OrgNode) -> Result<(), OrgError> {
        if self.node_map.contains_key(&node.id) {
            return Err(OrgError::DuplicateId(node.id.clone()));
        }
        let idx = self.nodes.len();
        self.node_map.insert(node.id.clone(), idx);
        self.nodes.push(node);
        Ok(())
    }

    pub fn nodes(&self) -> &[OrgNode] {
        &self.nodes
    }

    pub fn find_node(&self, id: &str) -> Option<&OrgNode> {
        self.node_map.get(id).map(|i| &self.nodes[*i])
    }

    pub fn find_node_mut(&mut self, id: &str) -> Option<&mut OrgNode> {
        self.node_map.get(id).copied().map(|i| &mut self.nodes[i])
    }

    /// Find root nodes (no reports_to).
    pub fn roots(&self) -> Vec<&OrgNode> {
        self.nodes.iter().filter(|n| n.reports_to.is_none()).collect()
    }

    /// Get direct reports of a manager.
    pub fn direct_reports(&self, manager_id: &str) -> Vec<&OrgNode> {
        self.nodes
            .iter()
            .filter(|n| n.reports_to.as_deref() == Some(manager_id))
            .collect()
    }

    /// Span of control: number of direct reports.
    pub fn span_of_control(&self, manager_id: &str) -> usize {
        self.direct_reports(manager_id).len()
    }

    /// Reporting chain from a node up to the root.
    pub fn reporting_chain(&self, id: &str) -> Vec<String> {
        let mut chain = Vec::new();
        let mut current = id.to_string();
        let mut visited = std::collections::HashSet::new();

        loop {
            if !visited.insert(current.clone()) {
                break; // Cycle protection.
            }
            chain.push(current.clone());
            match self.find_node(&current) {
                Some(node) => match &node.reports_to {
                    Some(parent) => current = parent.clone(),
                    None => break,
                },
                None => break,
            }
        }
        chain
    }

    /// Extract all nodes in a department (including sub-departments).
    pub fn department_subtree(&self, department: &str) -> Vec<&OrgNode> {
        self.nodes
            .iter()
            .filter(|n| n.department == department)
            .collect()
    }

    /// Headcount rollup: total people under a manager (including self).
    pub fn headcount_rollup(&self, manager_id: &str) -> usize {
        let mut count = 0;
        let mut queue: VecDeque<&str> = VecDeque::new();
        queue.push_back(manager_id);

        while let Some(current) = queue.pop_front() {
            count += 1;
            for report in self.direct_reports(current) {
                queue.push_back(report.id.as_str());
            }
        }
        count
    }

    /// Move a node to report to a new parent.
    pub fn reorg(&mut self, node_id: &str, new_parent_id: &str) -> Result<(), OrgError> {
        if !self.node_map.contains_key(node_id) {
            return Err(OrgError::NodeNotFound(node_id.to_string()));
        }
        if !self.node_map.contains_key(new_parent_id) {
            return Err(OrgError::NodeNotFound(new_parent_id.to_string()));
        }

        // Check for cycle: new_parent cannot be a descendant of node.
        let chain = self.reporting_chain(new_parent_id);
        if chain.contains(&node_id.to_string()) {
            return Err(OrgError::WouldCreateCycle);
        }

        if let Some(node) = self.find_node_mut(node_id) {
            node.reports_to = Some(new_parent_id.to_string());
        }
        Ok(())
    }

    /// Layout: top-down tree with horizontal spacing per level.
    pub fn layout(&mut self, spacing_x: f64, spacing_y: f64) {
        self.positions.clear();
        let roots: Vec<String> = self.roots().iter().map(|n| n.id.clone()).collect();

        let mut x_offset = 0.0;
        for root in &roots {
            let width = self.layout_subtree(root, x_offset, 0.0, spacing_x, spacing_y);
            x_offset += width + spacing_x;
        }
    }

    fn layout_subtree(
        &mut self,
        node_id: &str,
        x_start: f64,
        y: f64,
        spacing_x: f64,
        spacing_y: f64,
    ) -> f64 {
        let children: Vec<String> = self
            .direct_reports(node_id)
            .iter()
            .map(|n| n.id.clone())
            .collect();

        if children.is_empty() {
            self.positions.insert(
                node_id.to_string(),
                OrgPosition { x: x_start, y },
            );
            return spacing_x;
        }

        let mut child_x = x_start;
        let mut total_width = 0.0;
        for child in &children {
            let w = self.layout_subtree(child, child_x, y + spacing_y, spacing_x, spacing_y);
            child_x += w;
            total_width += w;
        }

        // Center parent over children.
        let first_child_x = self.positions[&children[0]].x;
        let last_child_x = self.positions[children.last().unwrap()].x;
        let parent_x = (first_child_x + last_child_x) / 2.0;

        self.positions.insert(
            node_id.to_string(),
            OrgPosition { x: parent_x, y },
        );

        total_width
    }

    pub fn position(&self, id: &str) -> Option<&OrgPosition> {
        self.positions.get(id)
    }

    /// Get all unique departments.
    pub fn departments(&self) -> Vec<String> {
        let mut deps: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &self.nodes {
            deps.insert(node.department.clone());
        }
        let mut result: Vec<String> = deps.into_iter().collect();
        result.sort();
        result
    }

    /// Depth of a node (distance from root).
    pub fn depth(&self, id: &str) -> usize {
        self.reporting_chain(id).len() - 1
    }
}

impl Default for OrgChart {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_org() -> OrgChart {
        let nodes = vec![
            OrgNode::new("ceo", "Jane CEO", "CEO", "Executive"),
            OrgNode::new("cto", "Bob CTO", "CTO", "Engineering").with_reports_to("ceo"),
            OrgNode::new("cfo", "Carol CFO", "CFO", "Finance").with_reports_to("ceo"),
            OrgNode::new("eng1", "Dave", "Senior Engineer", "Engineering").with_reports_to("cto"),
            OrgNode::new("eng2", "Eve", "Engineer", "Engineering").with_reports_to("cto"),
            OrgNode::new("eng3", "Frank", "Engineer", "Engineering").with_reports_to("eng1"),
            OrgNode::new("fin1", "Grace", "Accountant", "Finance").with_reports_to("cfo"),
        ];
        OrgChart::from_nodes(nodes).unwrap()
    }

    #[test]
    fn test_build_org() {
        let org = sample_org();
        assert_eq!(org.nodes().len(), 7);
    }

    #[test]
    fn test_roots() {
        let org = sample_org();
        let roots = org.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "ceo");
    }

    #[test]
    fn test_direct_reports() {
        let org = sample_org();
        let reports = org.direct_reports("cto");
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn test_span_of_control() {
        let org = sample_org();
        assert_eq!(org.span_of_control("ceo"), 2);
        assert_eq!(org.span_of_control("cto"), 2);
        assert_eq!(org.span_of_control("eng2"), 0);
    }

    #[test]
    fn test_reporting_chain() {
        let org = sample_org();
        let chain = org.reporting_chain("eng3");
        assert_eq!(chain, vec!["eng3", "eng1", "cto", "ceo"]);
    }

    #[test]
    fn test_department_subtree() {
        let org = sample_org();
        let eng = org.department_subtree("Engineering");
        assert_eq!(eng.len(), 4); // cto + eng1 + eng2 + eng3
    }

    #[test]
    fn test_headcount_rollup() {
        let org = sample_org();
        assert_eq!(org.headcount_rollup("ceo"), 7);
        assert_eq!(org.headcount_rollup("cto"), 4);
        assert_eq!(org.headcount_rollup("eng1"), 2);
        assert_eq!(org.headcount_rollup("eng2"), 1);
    }

    #[test]
    fn test_reorg() {
        let mut org = sample_org();
        // Move eng2 to report to eng1.
        org.reorg("eng2", "eng1").unwrap();
        assert_eq!(org.find_node("eng2").unwrap().reports_to.as_deref(), Some("eng1"));
        assert_eq!(org.span_of_control("eng1"), 2);
    }

    #[test]
    fn test_reorg_cycle_prevention() {
        let mut org = sample_org();
        // Can't make cto report to eng1 (eng1 reports to cto).
        let result = org.reorg("cto", "eng1");
        assert!(matches!(result, Err(OrgError::WouldCreateCycle)));
    }

    #[test]
    fn test_layout() {
        let mut org = sample_org();
        org.layout(100.0, 80.0);
        let ceo_pos = org.position("ceo").unwrap();
        let cto_pos = org.position("cto").unwrap();
        assert!(cto_pos.y > ceo_pos.y);
    }

    #[test]
    fn test_departments() {
        let org = sample_org();
        let deps = org.departments();
        assert_eq!(deps.len(), 3);
    }

    #[test]
    fn test_depth() {
        let org = sample_org();
        assert_eq!(org.depth("ceo"), 0);
        assert_eq!(org.depth("cto"), 1);
        assert_eq!(org.depth("eng3"), 3);
    }
}
