//! General graph data structure — directed/undirected, adjacency list representation.
//!
//! Supports node/edge add/remove, edge weights, BFS, DFS, topological sort,
//! connected components, cycle detection, DOT export, and graph transpose.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Graph Kind ───────────────────────────────────────────────────────────────

/// Whether edges are directed or undirected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphKind {
    Directed,
    Undirected,
}

// ── Edge ─────────────────────────────────────────────────────────────────────

/// A weighted edge from `from` to `to`.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

// ── Graph ────────────────────────────────────────────────────────────────────

/// Adjacency-list graph with node labels and weighted edges.
#[derive(Debug, Clone)]
pub struct Graph {
    kind: GraphKind,
    /// Label for each node id.
    labels: HashMap<usize, String>,
    /// Adjacency list: node -> vec of (neighbor, weight).
    adj: HashMap<usize, Vec<(usize, f64)>>,
    next_id: usize,
    edge_count: usize,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new(kind: GraphKind) -> Self {
        Self {
            kind,
            labels: HashMap::new(),
            adj: HashMap::new(),
            next_id: 0,
            edge_count: 0,
        }
    }

    /// Graph kind (directed or undirected).
    pub fn kind(&self) -> GraphKind {
        self.kind
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.labels.len()
    }

    /// Number of edges (each undirected edge counted once).
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Add a node with the given label, returning its id.
    pub fn add_node(&mut self, label: &str) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.labels.insert(id, label.to_string());
        self.adj.entry(id).or_default();
        id
    }

    /// Check if a node exists.
    pub fn has_node(&self, id: usize) -> bool {
        self.labels.contains_key(&id)
    }

    /// Get a node's label.
    pub fn label(&self, id: usize) -> Option<&str> {
        self.labels.get(&id).map(|s| s.as_str())
    }

    /// Get all node ids.
    pub fn node_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.labels.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Remove a node and all its incident edges.
    pub fn remove_node(&mut self, id: usize) -> bool {
        if self.labels.remove(&id).is_none() {
            return false;
        }
        // Remove outgoing edges
        if let Some(neighbors) = self.adj.remove(&id) {
            self.edge_count -= neighbors.len();
            if self.kind == GraphKind::Undirected {
                // Also remove the reverse edge from each neighbor, but don't double-count
                for (nb, _) in &neighbors {
                    if let Some(nb_list) = self.adj.get_mut(nb) {
                        let before = nb_list.len();
                        nb_list.retain(|(n, _)| *n != id);
                        // Don't subtract again — we already counted these edges above
                        let _ = before;
                    }
                }
            }
        }
        // For directed graphs, also remove incoming edges from other nodes
        if self.kind == GraphKind::Directed {
            let keys: Vec<usize> = self.adj.keys().copied().collect();
            for k in keys {
                if let Some(list) = self.adj.get_mut(&k) {
                    let before = list.len();
                    list.retain(|(n, _)| *n != id);
                    self.edge_count -= before - list.len();
                }
            }
        }
        true
    }

    /// Add a weighted edge. For undirected graphs, the reverse edge is added automatically.
    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) -> bool {
        if !self.has_node(from) || !self.has_node(to) {
            return false;
        }
        self.adj.entry(from).or_default().push((to, weight));
        if self.kind == GraphKind::Undirected {
            self.adj.entry(to).or_default().push((from, weight));
        }
        self.edge_count += 1;
        true
    }

    /// Add an unweighted edge (weight = 1.0).
    pub fn add_edge_unweighted(&mut self, from: usize, to: usize) -> bool {
        self.add_edge(from, to, 1.0)
    }

    /// Remove an edge. For undirected, removes both directions.
    pub fn remove_edge(&mut self, from: usize, to: usize) -> bool {
        let removed = if let Some(list) = self.adj.get_mut(&from) {
            let before = list.len();
            if let Some(pos) = list.iter().position(|(n, _)| *n == to) {
                list.remove(pos);
                true
            } else {
                let _ = before;
                false
            }
        } else {
            false
        };
        if !removed {
            return false;
        }
        if self.kind == GraphKind::Undirected {
            if let Some(list) = self.adj.get_mut(&to) {
                if let Some(pos) = list.iter().position(|(n, _)| *n == from) {
                    list.remove(pos);
                }
            }
        }
        self.edge_count -= 1;
        true
    }

    /// Check if an edge exists from `from` to `to`.
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.adj
            .get(&from)
            .map(|list| list.iter().any(|(n, _)| *n == to))
            .unwrap_or(false)
    }

    /// Get the weight of an edge.
    pub fn edge_weight(&self, from: usize, to: usize) -> Option<f64> {
        self.adj.get(&from).and_then(|list| {
            list.iter().find(|(n, _)| *n == to).map(|(_, w)| *w)
        })
    }

    /// Neighbors of a node.
    pub fn neighbors(&self, id: usize) -> Vec<usize> {
        self.adj
            .get(&id)
            .map(|list| list.iter().map(|(n, _)| *n).collect())
            .unwrap_or_default()
    }

    /// Neighbors with weights.
    pub fn neighbors_weighted(&self, id: usize) -> Vec<(usize, f64)> {
        self.adj.get(&id).cloned().unwrap_or_default()
    }

    /// All edges as Edge structs.
    pub fn edges(&self) -> Vec<Edge> {
        let mut result = Vec::new();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut node_ids: Vec<usize> = self.adj.keys().copied().collect();
        node_ids.sort();
        for from in node_ids {
            if let Some(list) = self.adj.get(&from) {
                for (to, weight) in list {
                    if self.kind == GraphKind::Undirected {
                        let key = if from <= *to { (from, *to) } else { (*to, from) };
                        if seen.insert(key) {
                            result.push(Edge { from, to: *to, weight: *weight });
                        }
                    } else {
                        result.push(Edge { from, to: *to, weight: *weight });
                    }
                }
            }
        }
        result
    }

    /// In-degree of a node (for directed graphs).
    pub fn in_degree(&self, id: usize) -> usize {
        if self.kind == GraphKind::Undirected {
            return self.neighbors(id).len();
        }
        let mut count = 0;
        for list in self.adj.values() {
            count += list.iter().filter(|(n, _)| *n == id).count();
        }
        count
    }

    /// Out-degree of a node.
    pub fn out_degree(&self, id: usize) -> usize {
        self.adj.get(&id).map(|l| l.len()).unwrap_or(0)
    }

    // ── Traversal ────────────────────────────────────────────────────────────

    /// BFS from `start`, returning nodes in visit order.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        if !self.has_node(start) {
            return Vec::new();
        }
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut order = Vec::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut nbrs = self.neighbors(node);
            nbrs.sort();
            for nb in nbrs {
                if visited.insert(nb) {
                    queue.push_back(nb);
                }
            }
        }
        order
    }

    /// DFS from `start`, returning nodes in visit order.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        if !self.has_node(start) {
            return Vec::new();
        }
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.dfs_recursive(start, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        order: &mut Vec<usize>,
    ) {
        if !visited.insert(node) {
            return;
        }
        order.push(node);
        let mut nbrs = self.neighbors(node);
        nbrs.sort();
        for nb in nbrs {
            self.dfs_recursive(nb, visited, order);
        }
    }

    /// Topological sort (Kahn's algorithm). Returns None if the graph has a cycle.
    /// Only valid for directed graphs.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        if self.kind == GraphKind::Undirected {
            return None;
        }

        let mut in_deg: HashMap<usize, usize> = HashMap::new();
        for id in self.node_ids() {
            in_deg.insert(id, 0);
        }
        for list in self.adj.values() {
            for (to, _) in list {
                *in_deg.entry(*to).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut zero_deg: Vec<usize> = in_deg.iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        zero_deg.sort();
        for id in zero_deg {
            queue.push_back(id);
        }

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node);
            let mut nbrs = self.neighbors(node);
            nbrs.sort();
            for nb in nbrs {
                if let Some(d) = in_deg.get_mut(&nb) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(nb);
                    }
                }
            }
        }

        if result.len() == self.node_count() {
            Some(result)
        } else {
            None // cycle detected
        }
    }

    /// Find connected components (for undirected graphs) or weakly connected components.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        // For directed graphs, build an undirected view
        let undirected_adj = if self.kind == GraphKind::Directed {
            let mut ua: HashMap<usize, Vec<usize>> = HashMap::new();
            for (&from, list) in &self.adj {
                for (to, _) in list {
                    ua.entry(from).or_default().push(*to);
                    ua.entry(*to).or_default().push(from);
                }
            }
            Some(ua)
        } else {
            None
        };

        let mut ids = self.node_ids();
        ids.sort();
        for id in ids {
            if visited.contains(&id) {
                continue;
            }
            let mut component = Vec::new();
            let mut stack = vec![id];
            while let Some(node) = stack.pop() {
                if !visited.insert(node) {
                    continue;
                }
                component.push(node);
                let nbrs = if let Some(ua) = &undirected_adj {
                    ua.get(&node).cloned().unwrap_or_default()
                } else {
                    self.neighbors(node)
                };
                for nb in nbrs {
                    if !visited.contains(&nb) {
                        stack.push(nb);
                    }
                }
            }
            component.sort();
            components.push(component);
        }
        components
    }

    /// Detect if the graph has a cycle.
    pub fn has_cycle(&self) -> bool {
        if self.kind == GraphKind::Directed {
            self.has_cycle_directed()
        } else {
            self.has_cycle_undirected()
        }
    }

    fn has_cycle_directed(&self) -> bool {
        // White = 0, Gray = 1, Black = 2
        let mut color: HashMap<usize, u8> = HashMap::new();
        for id in self.node_ids() {
            color.insert(id, 0);
        }
        for id in self.node_ids() {
            if color[&id] == 0 && self.dfs_cycle_directed(id, &mut color) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle_directed(&self, node: usize, color: &mut HashMap<usize, u8>) -> bool {
        color.insert(node, 1); // gray
        for nb in self.neighbors(node) {
            match color.get(&nb) {
                Some(1) => return true, // back edge → cycle
                Some(0) => {
                    if self.dfs_cycle_directed(nb, color) {
                        return true;
                    }
                }
                _ => {}
            }
        }
        color.insert(node, 2); // black
        false
    }

    fn has_cycle_undirected(&self) -> bool {
        let mut visited = HashSet::new();
        for id in self.node_ids() {
            if !visited.contains(&id) {
                if self.dfs_cycle_undirected(id, None, &mut visited) {
                    return true;
                }
            }
        }
        false
    }

    fn dfs_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool {
        visited.insert(node);
        for nb in self.neighbors(node) {
            if !visited.contains(&nb) {
                if self.dfs_cycle_undirected(nb, Some(node), visited) {
                    return true;
                }
            } else if parent != Some(nb) {
                return true;
            }
        }
        false
    }

    // ── DOT Export ───────────────────────────────────────────────────────────

    /// Export the graph in DOT format for Graphviz.
    pub fn to_dot(&self) -> String {
        let mut lines = Vec::new();
        let (graph_type, arrow) = if self.kind == GraphKind::Directed {
            ("digraph", "->")
        } else {
            ("graph", "--")
        };
        lines.push(format!("{} {{", graph_type));

        let mut ids = self.node_ids();
        ids.sort();
        for id in &ids {
            let label = self.labels.get(id).cloned().unwrap_or_default();
            lines.push(format!("  {} [label=\"{}\"];", id, label));
        }

        let mut seen = HashSet::new();
        for from in &ids {
            let mut nbrs = self.neighbors_weighted(*from);
            nbrs.sort_by(|a, b| a.0.cmp(&b.0));
            for (to, weight) in nbrs {
                if self.kind == GraphKind::Undirected {
                    let key = if *from <= to { (*from, to) } else { (to, *from) };
                    if !seen.insert(key) {
                        continue;
                    }
                }
                if (weight - 1.0).abs() < f64::EPSILON {
                    lines.push(format!("  {} {} {};", from, arrow, to));
                } else {
                    lines.push(format!(
                        "  {} {} {} [label=\"{:.1}\"];",
                        from, arrow, to, weight
                    ));
                }
            }
        }

        lines.push("}".to_string());
        lines.join("\n")
    }

    // ── Transpose ────────────────────────────────────────────────────────────

    /// Return the transpose (reversed) graph. Only meaningful for directed graphs.
    pub fn transpose(&self) -> Graph {
        let mut g = Graph::new(self.kind);
        g.next_id = self.next_id;
        g.labels = self.labels.clone();
        for id in self.node_ids() {
            g.adj.entry(id).or_default();
        }
        if self.kind == GraphKind::Undirected {
            g.adj = self.adj.clone();
            g.edge_count = self.edge_count;
            return g;
        }
        for (&from, list) in &self.adj {
            for (to, weight) in list {
                g.adj.entry(*to).or_default().push((from, *weight));
                g.edge_count += 1;
            }
        }
        g
    }

    /// BFS shortest path (unweighted) from start to end.
    pub fn bfs_path(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if !self.has_node(start) || !self.has_node(end) {
            return None;
        }
        if start == end {
            return Some(vec![start]);
        }
        let mut visited = HashSet::new();
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            let mut nbrs = self.neighbors(node);
            nbrs.sort();
            for nb in nbrs {
                if visited.insert(nb) {
                    parent.insert(nb, node);
                    if nb == end {
                        // Reconstruct path
                        let mut path = vec![end];
                        let mut cur = end;
                        while let Some(&p) = parent.get(&cur) {
                            path.push(p);
                            cur = p;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(nb);
                }
            }
        }
        None
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_directed() -> Graph {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        g.add_edge_unweighted(a, b);
        g.add_edge_unweighted(a, c);
        g.add_edge_unweighted(b, d);
        g.add_edge_unweighted(c, d);
        g
    }

    fn sample_undirected() -> Graph {
        let mut g = Graph::new(GraphKind::Undirected);
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge_unweighted(a, b);
        g.add_edge_unweighted(b, c);
        g
    }

    #[test]
    fn test_add_node() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        assert_eq!(g.node_count(), 2);
        assert!(g.has_node(a));
        assert!(g.has_node(b));
        assert_eq!(g.label(a), Some("A"));
    }

    #[test]
    fn test_add_edge_directed() {
        let g = sample_directed();
        assert_eq!(g.edge_count(), 4);
        assert!(g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0));
    }

    #[test]
    fn test_add_edge_undirected() {
        let g = sample_undirected();
        assert_eq!(g.edge_count(), 2);
        assert!(g.has_edge(0, 1));
        assert!(g.has_edge(1, 0));
    }

    #[test]
    fn test_remove_node() {
        let mut g = sample_directed();
        assert!(g.remove_node(1)); // remove B
        assert!(!g.has_node(1));
        assert_eq!(g.node_count(), 3);
        assert!(!g.has_edge(0, 1));
    }

    #[test]
    fn test_remove_edge() {
        let mut g = sample_directed();
        assert!(g.remove_edge(0, 1));
        assert!(!g.has_edge(0, 1));
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn test_edge_weight() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        g.add_edge(a, b, 3.5);
        assert_eq!(g.edge_weight(a, b), Some(3.5));
        assert_eq!(g.edge_weight(b, a), None);
    }

    #[test]
    fn test_neighbors() {
        let g = sample_directed();
        let mut nbrs = g.neighbors(0);
        nbrs.sort();
        assert_eq!(nbrs, vec![1, 2]);
    }

    #[test]
    fn test_bfs() {
        let g = sample_directed();
        let order = g.bfs(0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_dfs() {
        let g = sample_directed();
        let order = g.dfs(0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_sort_dag() {
        let g = sample_directed();
        let topo = g.topological_sort().unwrap();
        assert_eq!(topo.len(), 4);
        // A must come before B and C; both before D
        let pos: HashMap<usize, usize> = topo.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        assert!(pos[&0] < pos[&1]);
        assert!(pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&3]);
    }

    #[test]
    fn test_topological_sort_cycle() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        g.add_edge_unweighted(a, b);
        g.add_edge_unweighted(b, a);
        assert!(g.topological_sort().is_none());
    }

    #[test]
    fn test_connected_components_undirected() {
        let mut g = Graph::new(GraphKind::Undirected);
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        let d = g.add_node("D");
        g.add_edge_unweighted(a, b);
        g.add_edge_unweighted(c, d);
        let cc = g.connected_components();
        assert_eq!(cc.len(), 2);
    }

    #[test]
    fn test_has_cycle_directed_no_cycle() {
        let g = sample_directed();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_has_cycle_directed_with_cycle() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge_unweighted(a, b);
        g.add_edge_unweighted(b, c);
        g.add_edge_unweighted(c, a);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_has_cycle_undirected() {
        let mut g = Graph::new(GraphKind::Undirected);
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge_unweighted(a, b);
        g.add_edge_unweighted(b, c);
        g.add_edge_unweighted(c, a);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_dot_export_directed() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("X");
        let b = g.add_node("Y");
        g.add_edge_unweighted(a, b);
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_dot_export_undirected() {
        let mut g = Graph::new(GraphKind::Undirected);
        let a = g.add_node("X");
        let b = g.add_node("Y");
        g.add_edge_unweighted(a, b);
        let dot = g.to_dot();
        assert!(dot.contains("graph {"));
        assert!(dot.contains("--"));
    }

    #[test]
    fn test_transpose() {
        let g = sample_directed();
        let gt = g.transpose();
        // Original: A->B, in transpose: B->A
        assert!(gt.has_edge(1, 0));
        assert!(!gt.has_edge(0, 1));
        assert_eq!(gt.node_count(), g.node_count());
        assert_eq!(gt.edge_count(), g.edge_count());
    }

    #[test]
    fn test_in_degree_out_degree() {
        let g = sample_directed();
        assert_eq!(g.out_degree(0), 2); // A -> B, C
        assert_eq!(g.in_degree(0), 0);
        assert_eq!(g.in_degree(3), 2); // B -> D, C -> D
    }

    #[test]
    fn test_bfs_path() {
        let g = sample_directed();
        let path = g.bfs_path(0, 3).unwrap();
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_bfs_path_no_path() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        let _ = (a, b);
        assert!(g.bfs_path(a, b).is_none());
    }

    #[test]
    fn test_edges_list() {
        let g = sample_directed();
        let edges = g.edges();
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn test_remove_edge_undirected() {
        let mut g = sample_undirected();
        assert!(g.remove_edge(0, 1));
        assert!(!g.has_edge(0, 1));
        assert!(!g.has_edge(1, 0));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_weighted_edge() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        let b = g.add_node("B");
        g.add_edge(a, b, 2.5);
        let dot = g.to_dot();
        assert!(dot.contains("2.5"));
    }

    #[test]
    fn test_self_loop_cycle() {
        let mut g = Graph::new(GraphKind::Directed);
        let a = g.add_node("A");
        g.add_edge_unweighted(a, a);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new(GraphKind::Directed);
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(!g.has_cycle());
        assert_eq!(g.connected_components().len(), 0);
    }

    #[test]
    fn test_node_ids_sorted() {
        let mut g = Graph::new(GraphKind::Directed);
        g.add_node("C");
        g.add_node("A");
        g.add_node("B");
        let ids = g.node_ids();
        assert_eq!(ids, vec![0, 1, 2]);
    }
}
