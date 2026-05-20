//! Call graph analysis — function nodes, call edges with counts,
//! hot path detection, call chain extraction, fan-in/fan-out analysis,
//! recursive call detection, call graph DOT export, and profiling data model.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Function Node ────────────────────────────────────────────────

/// A function in the call graph.
#[derive(Debug, Clone)]
pub struct FunctionNode {
    pub id: u32,
    pub name: String,
    pub module: String,
    pub self_time_us: u64,
    pub total_time_us: u64,
    pub call_count: u64,
}

impl FunctionNode {
    pub fn new(id: u32, name: &str, module: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            module: module.to_string(),
            self_time_us: 0,
            total_time_us: 0,
            call_count: 0,
        }
    }

    /// Qualified name: "module::name".
    pub fn qualified_name(&self) -> String {
        if self.module.is_empty() {
            self.name.clone()
        } else {
            format!("{}::{}", self.module, self.name)
        }
    }

    /// Average time per call in microseconds.
    pub fn avg_time_us(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.call_count as f64
        }
    }
}

// ── Call Edge ────────────────────────────────────────────────────

/// An edge in the call graph: caller -> callee.
#[derive(Debug, Clone)]
pub struct CallEdge {
    pub caller_id: u32,
    pub callee_id: u32,
    pub call_count: u64,
    pub total_time_us: u64,
}

impl CallEdge {
    pub fn new(caller_id: u32, callee_id: u32) -> Self {
        Self {
            caller_id,
            callee_id,
            call_count: 0,
            total_time_us: 0,
        }
    }

    pub fn with_stats(mut self, count: u64, time_us: u64) -> Self {
        self.call_count = count;
        self.total_time_us = time_us;
        self
    }
}

// ── Hot Path Entry ───────────────────────────────────────────────

/// A node on a hot path through the call graph.
#[derive(Debug, Clone)]
pub struct HotPathEntry {
    pub function_id: u32,
    pub function_name: String,
    pub cumulative_time_us: u64,
}

// ── Fan Analysis ─────────────────────────────────────────────────

/// Fan-in/fan-out analysis result for a single function.
#[derive(Debug, Clone)]
pub struct FanAnalysis {
    pub function_id: u32,
    pub function_name: String,
    pub fan_in: usize,
    pub fan_out: usize,
}

// ── Call Chain ───────────────────────────────────────────────────

/// A chain of function calls from a start to an end node.
#[derive(Debug, Clone)]
pub struct CallChain {
    pub nodes: Vec<u32>,
    pub total_time_us: u64,
}

impl CallChain {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ── Profiling Record ─────────────────────────────────────────────

/// A single profiling record: one invocation of a function.
#[derive(Debug, Clone)]
pub struct ProfilingRecord {
    pub caller_id: Option<u32>,
    pub callee_id: u32,
    pub duration_us: u64,
}

impl ProfilingRecord {
    pub fn new(caller_id: Option<u32>, callee_id: u32, duration_us: u64) -> Self {
        Self {
            caller_id,
            callee_id,
            duration_us,
        }
    }
}

// ── Call Graph ───────────────────────────────────────────────────

/// A directed call graph with profiling data.
pub struct CallGraph {
    nodes: HashMap<u32, FunctionNode>,
    /// (caller_id, callee_id) -> CallEdge
    edges: HashMap<(u32, u32), CallEdge>,
    next_id: u32,
}

impl CallGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a function node, returning its id.
    pub fn add_function(&mut self, name: &str, module: &str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, FunctionNode::new(id, name, module));
        id
    }

    /// Get a function node by id.
    pub fn get_function(&self, id: u32) -> Option<&FunctionNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable function node by id.
    pub fn get_function_mut(&mut self, id: u32) -> Option<&mut FunctionNode> {
        self.nodes.get_mut(&id)
    }

    /// Add or update a call edge.
    pub fn record_call(&mut self, caller_id: u32, callee_id: u32, duration_us: u64) {
        let edge = self
            .edges
            .entry((caller_id, callee_id))
            .or_insert_with(|| CallEdge::new(caller_id, callee_id));
        edge.call_count += 1;
        edge.total_time_us += duration_us;

        if let Some(callee) = self.nodes.get_mut(&callee_id) {
            callee.call_count += 1;
            callee.total_time_us += duration_us;
        }
    }

    /// Ingest profiling records in batch.
    pub fn ingest_records(&mut self, records: &[ProfilingRecord]) {
        for rec in records {
            if let Some(caller_id) = rec.caller_id {
                self.record_call(caller_id, rec.callee_id, rec.duration_us);
            } else if let Some(node) = self.nodes.get_mut(&rec.callee_id) {
                node.call_count += 1;
                node.total_time_us += rec.duration_us;
            }
        }
    }

    /// Number of function nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of call edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get all edges.
    pub fn edges(&self) -> Vec<&CallEdge> {
        self.edges.values().collect()
    }

    /// Get outgoing edges from a function (calls it makes).
    pub fn outgoing_edges(&self, function_id: u32) -> Vec<&CallEdge> {
        self.edges
            .values()
            .filter(|e| e.caller_id == function_id)
            .collect()
    }

    /// Get incoming edges to a function (calls it receives).
    pub fn incoming_edges(&self, function_id: u32) -> Vec<&CallEdge> {
        self.edges
            .values()
            .filter(|e| e.callee_id == function_id)
            .collect()
    }

    /// Compute fan-in and fan-out for a function.
    pub fn fan_analysis(&self, function_id: u32) -> Option<FanAnalysis> {
        let node = self.nodes.get(&function_id)?;
        let fan_in = self.incoming_edges(function_id).len();
        let fan_out = self.outgoing_edges(function_id).len();
        Some(FanAnalysis {
            function_id,
            function_name: node.name.clone(),
            fan_in,
            fan_out,
        })
    }

    /// Compute fan analysis for all nodes, sorted by (fan_out + fan_in) descending.
    pub fn all_fan_analysis(&self) -> Vec<FanAnalysis> {
        let mut results: Vec<FanAnalysis> = self
            .nodes
            .keys()
            .filter_map(|id| self.fan_analysis(*id))
            .collect();
        results.sort_by(|a, b| {
            let a_total = a.fan_in + a.fan_out;
            let b_total = b.fan_in + b.fan_out;
            b_total.cmp(&a_total).then_with(|| a.function_name.cmp(&b.function_name))
        });
        results
    }

    /// Detect the hot path: follow the highest-time edge from each node
    /// starting from root functions (fan-in == 0), or from the node
    /// with the highest total_time if no pure roots exist.
    pub fn hot_path(&self) -> Vec<HotPathEntry> {
        // Find root nodes (no incoming edges)
        let all_callees: HashSet<u32> = self.edges.values().map(|e| e.callee_id).collect();
        let mut roots: Vec<u32> = self
            .nodes
            .keys()
            .filter(|id| !all_callees.contains(id))
            .copied()
            .collect();

        if roots.is_empty() {
            // Pick the node with highest total_time
            if let Some(node) = self.nodes.values().max_by_key(|n| n.total_time_us) {
                roots.push(node.id);
            }
        }

        if roots.is_empty() {
            return Vec::new();
        }

        // Sort roots by total_time descending and pick the hottest
        roots.sort_by(|a, b| {
            let ta = self.nodes.get(b).map_or(0, |n| n.total_time_us);
            let tb = self.nodes.get(a).map_or(0, |n| n.total_time_us);
            ta.cmp(&tb)
        });

        let start = roots[0];
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        let mut current = start;

        loop {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            let node = match self.nodes.get(&current) {
                Some(n) => n,
                None => break,
            };

            path.push(HotPathEntry {
                function_id: current,
                function_name: node.name.clone(),
                cumulative_time_us: node.total_time_us,
            });

            // Follow the edge with the highest total_time
            let outgoing = self.outgoing_edges(current);
            if let Some(hottest) = outgoing.iter().max_by_key(|e| e.total_time_us) {
                current = hottest.callee_id;
            } else {
                break;
            }
        }

        path
    }

    /// Extract all call chains from `start_id` to `end_id` using BFS.
    /// Returns up to `max_chains` results to avoid combinatorial explosion.
    pub fn call_chains(&self, start_id: u32, end_id: u32, max_chains: usize) -> Vec<CallChain> {
        let mut results = Vec::new();
        let mut queue: VecDeque<(Vec<u32>, u64)> = VecDeque::new();
        queue.push_back((vec![start_id], 0));

        while let Some((path, time)) = queue.pop_front() {
            if results.len() >= max_chains {
                break;
            }

            let current = *path.last().unwrap();
            if current == end_id && path.len() > 1 {
                results.push(CallChain {
                    nodes: path,
                    total_time_us: time,
                });
                continue;
            }

            // Limit path length to avoid infinite loops
            if path.len() > self.nodes.len() {
                continue;
            }

            for edge in self.outgoing_edges(current) {
                if !path.contains(&edge.callee_id) || edge.callee_id == end_id {
                    let mut new_path = path.clone();
                    new_path.push(edge.callee_id);
                    queue.push_back((new_path, time + edge.total_time_us));
                }
            }
        }

        results
    }

    /// Detect recursive calls: edges where caller == callee (direct),
    /// or cycles in the graph (indirect).
    pub fn detect_recursion(&self) -> Vec<Vec<u32>> {
        let mut cycles = Vec::new();

        // Direct recursion
        for edge in self.edges.values() {
            if edge.caller_id == edge.callee_id {
                cycles.push(vec![edge.caller_id]);
            }
        }

        // Indirect recursion: DFS-based cycle detection
        for &start in self.nodes.keys() {
            let mut stack = vec![(start, vec![start])];
            let mut visited_from_start: HashSet<u32> = HashSet::new();

            while let Some((current, path)) = stack.pop() {
                if !visited_from_start.insert(current) && current != start {
                    continue;
                }

                for edge in self.outgoing_edges(current) {
                    if edge.callee_id == start && path.len() > 1 {
                        let mut cycle = path.clone();
                        cycle.push(start);
                        // Normalize: start with smallest id
                        let min_pos = cycle[..cycle.len() - 1]
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, v)| **v)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let mut normalized = cycle[min_pos..cycle.len() - 1].to_vec();
                        normalized.extend_from_slice(&cycle[..min_pos]);
                        normalized.push(normalized[0]);

                        if !cycles.iter().any(|c| *c == normalized) {
                            cycles.push(normalized);
                        }
                    } else if !path.contains(&edge.callee_id) {
                        let mut new_path = path.clone();
                        new_path.push(edge.callee_id);
                        stack.push((edge.callee_id, new_path));
                    }
                }
            }
        }

        cycles
    }

    /// Export the call graph in DOT (Graphviz) format.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph call_graph {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Sort nodes by id for deterministic output
        let mut node_ids: Vec<u32> = self.nodes.keys().copied().collect();
        node_ids.sort();

        for id in &node_ids {
            let node = &self.nodes[id];
            dot.push_str(&format!(
                "  n{} [label=\"{}\\ncalls={} time={}us\"];\n",
                id, node.name, node.call_count, node.total_time_us
            ));
        }

        dot.push('\n');

        // Sort edges for deterministic output
        let mut edge_keys: Vec<(u32, u32)> = self.edges.keys().copied().collect();
        edge_keys.sort();

        for key in &edge_keys {
            let edge = &self.edges[key];
            dot.push_str(&format!(
                "  n{} -> n{} [label=\"{}x {}us\"];\n",
                edge.caller_id, edge.callee_id, edge.call_count, edge.total_time_us
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Return functions sorted by total time descending.
    pub fn functions_by_time(&self) -> Vec<&FunctionNode> {
        let mut funcs: Vec<&FunctionNode> = self.nodes.values().collect();
        funcs.sort_by(|a, b| b.total_time_us.cmp(&a.total_time_us).then_with(|| a.name.cmp(&b.name)));
        funcs
    }

    /// Return functions sorted by call count descending.
    pub fn functions_by_calls(&self) -> Vec<&FunctionNode> {
        let mut funcs: Vec<&FunctionNode> = self.nodes.values().collect();
        funcs.sort_by(|a, b| b.call_count.cmp(&a.call_count).then_with(|| a.name.cmp(&b.name)));
        funcs
    }
}

impl Default for CallGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_graph() -> CallGraph {
        let mut g = CallGraph::new();
        let main_id = g.add_function("main", "app");
        let process_id = g.add_function("process", "app");
        let compute_id = g.add_function("compute", "math");
        let io_id = g.add_function("io_read", "io");

        g.record_call(main_id, process_id, 100);
        g.record_call(main_id, process_id, 120);
        g.record_call(process_id, compute_id, 80);
        g.record_call(process_id, io_id, 50);
        g.record_call(main_id, io_id, 30);
        g
    }

    #[test]
    fn test_add_function() {
        let mut g = CallGraph::new();
        let id = g.add_function("foo", "bar");
        assert_eq!(id, 0);
        let node = g.get_function(id).unwrap();
        assert_eq!(node.name, "foo");
        assert_eq!(node.module, "bar");
    }

    #[test]
    fn test_record_call() {
        let g = build_graph();
        assert_eq!(g.edge_count(), 4);
        // process was called 2x by main
        let process = g.get_function(1).unwrap();
        assert_eq!(process.call_count, 2);
        assert_eq!(process.total_time_us, 220);
    }

    #[test]
    fn test_node_edge_counts() {
        let g = build_graph();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 4);
    }

    #[test]
    fn test_outgoing_edges() {
        let g = build_graph();
        let out = g.outgoing_edges(0); // main
        assert_eq!(out.len(), 2); // calls process and io_read
    }

    #[test]
    fn test_incoming_edges() {
        let g = build_graph();
        let inc = g.incoming_edges(3); // io_read
        assert_eq!(inc.len(), 2); // called by main and process
    }

    #[test]
    fn test_fan_analysis() {
        let g = build_graph();
        let fa = g.fan_analysis(3).unwrap(); // io_read
        assert_eq!(fa.fan_in, 2);
        assert_eq!(fa.fan_out, 0);
    }

    #[test]
    fn test_all_fan_analysis() {
        let g = build_graph();
        let results = g.all_fan_analysis();
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_hot_path() {
        let g = build_graph();
        let path = g.hot_path();
        assert!(!path.is_empty());
        assert_eq!(path[0].function_name, "main");
    }

    #[test]
    fn test_call_chains() {
        let g = build_graph();
        let chains = g.call_chains(0, 3, 10);
        // main -> io_read directly, and main -> process -> io_read
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_call_chain_length() {
        let g = build_graph();
        let chains = g.call_chains(0, 3, 10);
        let direct = chains.iter().find(|c| c.len() == 2).unwrap();
        assert_eq!(direct.nodes[0], 0);
        assert_eq!(direct.nodes[1], 3);
    }

    #[test]
    fn test_detect_direct_recursion() {
        let mut g = CallGraph::new();
        let a = g.add_function("recurse", "");
        g.record_call(a, a, 10);
        let cycles = g.detect_recursion();
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0], vec![a]);
    }

    #[test]
    fn test_detect_indirect_recursion() {
        let mut g = CallGraph::new();
        let a = g.add_function("a", "");
        let b = g.add_function("b", "");
        let c = g.add_function("c", "");
        g.record_call(a, b, 10);
        g.record_call(b, c, 10);
        g.record_call(c, a, 10);
        let cycles = g.detect_recursion();
        assert!(!cycles.is_empty());
        // Should find cycle a -> b -> c -> a
        let cycle = cycles.iter().find(|c| c.len() > 1).unwrap();
        assert!(cycle.len() >= 3);
    }

    #[test]
    fn test_no_recursion() {
        let g = build_graph();
        let cycles = g.detect_recursion();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_dot_export() {
        let g = build_graph();
        let dot = g.to_dot();
        assert!(dot.starts_with("digraph call_graph"));
        assert!(dot.contains("main"));
        assert!(dot.contains("->"));
        assert!(dot.ends_with("}\n"));
    }

    #[test]
    fn test_functions_by_time() {
        let g = build_graph();
        let sorted = g.functions_by_time();
        // process has 220us total, should be first
        assert_eq!(sorted[0].name, "process");
    }

    #[test]
    fn test_functions_by_calls() {
        let g = build_graph();
        let sorted = g.functions_by_calls();
        // io_read called 2x, process called 2x
        let top_count = sorted[0].call_count;
        assert_eq!(top_count, 2);
    }

    #[test]
    fn test_qualified_name() {
        let node = FunctionNode::new(0, "foo", "bar");
        assert_eq!(node.qualified_name(), "bar::foo");
        let node2 = FunctionNode::new(0, "foo", "");
        assert_eq!(node2.qualified_name(), "foo");
    }

    #[test]
    fn test_avg_time() {
        let mut node = FunctionNode::new(0, "f", "");
        node.call_count = 4;
        node.total_time_us = 100;
        assert!((node.avg_time_us() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_avg_time_zero_calls() {
        let node = FunctionNode::new(0, "f", "");
        assert!((node.avg_time_us() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_ingest_records() {
        let mut g = CallGraph::new();
        let a = g.add_function("a", "");
        let b = g.add_function("b", "");
        g.ingest_records(&[
            ProfilingRecord::new(Some(a), b, 100),
            ProfilingRecord::new(Some(a), b, 200),
            ProfilingRecord::new(None, a, 50),
        ]);
        assert_eq!(g.get_function(b).unwrap().call_count, 2);
        assert_eq!(g.get_function(a).unwrap().call_count, 1);
    }

    #[test]
    fn test_empty_graph() {
        let g = CallGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(g.hot_path().is_empty());
    }

    #[test]
    fn test_call_chain_no_path() {
        let mut g = CallGraph::new();
        let a = g.add_function("a", "");
        let b = g.add_function("b", "");
        // No edge between a and b
        let chains = g.call_chains(a, b, 10);
        assert!(chains.is_empty());
    }

    #[test]
    fn test_call_chain_is_empty() {
        let chain = CallChain {
            nodes: Vec::new(),
            total_time_us: 0,
        };
        assert!(chain.is_empty());
    }

    #[test]
    fn test_edge_with_stats() {
        let edge = CallEdge::new(0, 1).with_stats(5, 200);
        assert_eq!(edge.call_count, 5);
        assert_eq!(edge.total_time_us, 200);
    }
}
