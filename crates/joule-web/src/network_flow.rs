//! Network flow algorithms — Ford-Fulkerson with BFS (Edmonds-Karp), max flow,
//! min cut, flow decomposition, capacity scaling, multi-source/multi-sink reduction.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Flow Network ─────────────────────────────────────────────────────────────

/// A directed flow network with capacities.
#[derive(Debug, Clone)]
pub struct FlowNetwork {
    /// Number of nodes.
    node_count: usize,
    /// Capacity matrix: node -> neighbor -> capacity.
    capacity: HashMap<usize, HashMap<usize, f64>>,
    /// Current flow matrix.
    flow: HashMap<usize, HashMap<usize, f64>>,
}

impl FlowNetwork {
    /// Create a new flow network with `n` nodes.
    pub fn new(n: usize) -> Self {
        Self {
            node_count: n,
            capacity: HashMap::new(),
            flow: HashMap::new(),
        }
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Add a directed edge with the given capacity.
    pub fn add_edge(&mut self, from: usize, to: usize, cap: f64) {
        *self
            .capacity
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0.0) += cap;
        // Ensure reverse edge entry exists for residual graph
        self.capacity.entry(to).or_default().entry(from).or_insert(0.0);
        self.flow.entry(from).or_default().entry(to).or_insert(0.0);
        self.flow.entry(to).or_default().entry(from).or_insert(0.0);
    }

    /// Get the capacity of edge (from, to).
    pub fn capacity(&self, from: usize, to: usize) -> f64 {
        self.capacity
            .get(&from)
            .and_then(|m| m.get(&to))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get the current flow on edge (from, to).
    pub fn flow_on(&self, from: usize, to: usize) -> f64 {
        self.flow
            .get(&from)
            .and_then(|m| m.get(&to))
            .copied()
            .unwrap_or(0.0)
    }

    /// Residual capacity of edge (from, to).
    pub fn residual_capacity(&self, from: usize, to: usize) -> f64 {
        self.capacity(from, to) - self.flow_on(from, to)
    }

    /// All edges with their capacity and flow as (from, to, capacity, flow).
    pub fn edges(&self) -> Vec<(usize, usize, f64, f64)> {
        let mut result = Vec::new();
        let mut sorted_froms: Vec<usize> = self.capacity.keys().copied().collect();
        sorted_froms.sort();
        for from in sorted_froms {
            if let Some(neighbors) = self.capacity.get(&from) {
                let mut sorted_tos: Vec<usize> = neighbors.keys().copied().collect();
                sorted_tos.sort();
                for to in sorted_tos {
                    let cap = self.capacity(from, to);
                    if cap > 0.0 {
                        let fl = self.flow_on(from, to);
                        result.push((from, to, cap, fl));
                    }
                }
            }
        }
        result
    }

    /// Get all neighbors with positive residual capacity.
    fn residual_neighbors(&self, node: usize) -> Vec<usize> {
        let mut result = Vec::new();
        // Forward edges
        if let Some(neighbors) = self.capacity.get(&node) {
            for (&to, _) in neighbors {
                if self.residual_capacity(node, to) > 0.0 {
                    result.push(to);
                }
            }
        }
        result
    }

    /// BFS to find augmenting path in residual graph.
    fn bfs_augmenting_path(&self, source: usize, sink: usize) -> Option<(Vec<usize>, f64)> {
        let mut visited = HashSet::new();
        let mut parent: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();

        visited.insert(source);
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            if node == sink {
                // Reconstruct path and find bottleneck
                let mut path = vec![sink];
                let mut cur = sink;
                while cur != source {
                    let p = parent[&cur];
                    path.push(p);
                    cur = p;
                }
                path.reverse();

                let mut bottleneck = f64::INFINITY;
                for i in 0..path.len() - 1 {
                    let rc = self.residual_capacity(path[i], path[i + 1]);
                    if rc < bottleneck {
                        bottleneck = rc;
                    }
                }
                return Some((path, bottleneck));
            }

            let mut neighbors = self.residual_neighbors(node);
            neighbors.sort();
            for nb in neighbors {
                if visited.insert(nb) {
                    parent.insert(nb, node);
                    queue.push_back(nb);
                }
            }
        }

        None
    }

    /// Reset all flows to zero.
    pub fn reset_flow(&mut self) {
        for inner in self.flow.values_mut() {
            for val in inner.values_mut() {
                *val = 0.0;
            }
        }
    }
}

// ── Max Flow Result ──────────────────────────────────────────────────────────

/// Result of a max flow computation.
#[derive(Debug, Clone)]
pub struct MaxFlowResult {
    /// Maximum flow value.
    pub max_flow: f64,
    /// Flow on each edge: (from, to, flow_value).
    pub edge_flows: Vec<(usize, usize, f64)>,
}

// ── Min Cut Result ───────────────────────────────────────────────────────────

/// Result of a min cut computation.
#[derive(Debug, Clone)]
pub struct MinCutResult {
    /// Cut value (equals max flow).
    pub cut_value: f64,
    /// Nodes on the source side of the cut.
    pub source_side: Vec<usize>,
    /// Nodes on the sink side of the cut.
    pub sink_side: Vec<usize>,
    /// Cut edges: (from, to, capacity).
    pub cut_edges: Vec<(usize, usize, f64)>,
}

// ── Flow Path ────────────────────────────────────────────────────────────────

/// A flow path for decomposition.
#[derive(Debug, Clone)]
pub struct FlowPath {
    /// Path from source to sink.
    pub path: Vec<usize>,
    /// Flow along this path.
    pub flow: f64,
}

// ── Edmonds-Karp (Ford-Fulkerson with BFS) ───────────────────────────────────

/// Compute maximum flow using Edmonds-Karp algorithm (Ford-Fulkerson with BFS).
pub fn edmonds_karp(network: &mut FlowNetwork, source: usize, sink: usize) -> MaxFlowResult {
    network.reset_flow();
    let mut total_flow = 0.0;

    while let Some((path, bottleneck)) = network.bfs_augmenting_path(source, sink) {
        // Augment flow along path
        for i in 0..path.len() - 1 {
            let from = path[i];
            let to = path[i + 1];
            *network
                .flow
                .entry(from)
                .or_default()
                .entry(to)
                .or_insert(0.0) += bottleneck;
            *network
                .flow
                .entry(to)
                .or_default()
                .entry(from)
                .or_insert(0.0) -= bottleneck;
        }
        total_flow += bottleneck;
    }

    let edge_flows = collect_edge_flows(network);

    MaxFlowResult {
        max_flow: total_flow,
        edge_flows,
    }
}

fn collect_edge_flows(network: &FlowNetwork) -> Vec<(usize, usize, f64)> {
    let mut result = Vec::new();
    let mut sorted_froms: Vec<usize> = network.capacity.keys().copied().collect();
    sorted_froms.sort();
    for from in sorted_froms {
        if let Some(neighbors) = network.capacity.get(&from) {
            let mut sorted_tos: Vec<usize> = neighbors.keys().copied().collect();
            sorted_tos.sort();
            for to in sorted_tos {
                let cap = network.capacity(from, to);
                if cap > 0.0 {
                    let fl = network.flow_on(from, to).max(0.0);
                    if fl > 0.0 {
                        result.push((from, to, fl));
                    }
                }
            }
        }
    }
    result
}

// ── Min Cut ──────────────────────────────────────────────────────────────────

/// Compute minimum cut. Runs max flow first, then finds reachable nodes from source
/// in the residual graph.
pub fn min_cut(network: &mut FlowNetwork, source: usize, sink: usize) -> MinCutResult {
    let result = edmonds_karp(network, source, sink);

    // BFS from source in residual graph
    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();
    reachable.insert(source);
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        for nb in network.residual_neighbors(node) {
            if reachable.insert(nb) {
                queue.push_back(nb);
            }
        }
    }

    let mut source_side: Vec<usize> = reachable.iter().copied().collect();
    source_side.sort();
    let mut sink_side: Vec<usize> = (0..network.node_count())
        .filter(|n| !reachable.contains(n))
        .collect();
    sink_side.sort();

    // Cut edges: edges from source side to sink side with positive capacity
    let mut cut_edges = Vec::new();
    for &s in &source_side {
        if let Some(neighbors) = network.capacity.get(&s) {
            let mut sorted_tos: Vec<usize> = neighbors.keys().copied().collect();
            sorted_tos.sort();
            for t in sorted_tos {
                if !reachable.contains(&t) {
                    let cap = network.capacity(s, t);
                    if cap > 0.0 {
                        cut_edges.push((s, t, cap));
                    }
                }
            }
        }
    }

    MinCutResult {
        cut_value: result.max_flow,
        source_side,
        sink_side,
        cut_edges,
    }
}

// ── Flow Decomposition ───────────────────────────────────────────────────────

/// Decompose flow into source-to-sink paths.
pub fn flow_decomposition(
    network: &FlowNetwork,
    source: usize,
    sink: usize,
) -> Vec<FlowPath> {
    // Work on a copy of the flow
    let mut residual_flow: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
    for (&from, neighbors) in &network.capacity {
        for (&to, _) in neighbors {
            let fl = network.flow_on(from, to);
            if fl > 0.0 {
                *residual_flow.entry(from).or_default().entry(to).or_insert(0.0) += fl;
            }
        }
    }

    let mut paths = Vec::new();

    loop {
        // DFS from source to sink in the flow graph
        let mut visited = HashSet::new();
        let path = dfs_find_path(source, sink, &residual_flow, &mut visited);

        match path {
            Some(p) => {
                // Find minimum flow along this path
                let mut min_flow = f64::INFINITY;
                for i in 0..p.len() - 1 {
                    let fl = residual_flow
                        .get(&p[i])
                        .and_then(|m| m.get(&p[i + 1]))
                        .copied()
                        .unwrap_or(0.0);
                    if fl < min_flow {
                        min_flow = fl;
                    }
                }

                if min_flow <= 0.0 {
                    break;
                }

                // Subtract flow along path
                for i in 0..p.len() - 1 {
                    if let Some(m) = residual_flow.get_mut(&p[i]) {
                        if let Some(f) = m.get_mut(&p[i + 1]) {
                            *f -= min_flow;
                        }
                    }
                }

                paths.push(FlowPath { path: p, flow: min_flow });
            }
            None => break,
        }
    }

    paths
}

fn dfs_find_path(
    current: usize,
    target: usize,
    flow: &HashMap<usize, HashMap<usize, f64>>,
    visited: &mut HashSet<usize>,
) -> Option<Vec<usize>> {
    if current == target {
        return Some(vec![target]);
    }
    visited.insert(current);

    if let Some(neighbors) = flow.get(&current) {
        let mut sorted_nbs: Vec<(&usize, &f64)> = neighbors.iter().collect();
        sorted_nbs.sort_by_key(|(k, _)| **k);
        for (&nb, &fl) in sorted_nbs {
            if fl > 0.0 && !visited.contains(&nb) {
                if let Some(mut path) = dfs_find_path(nb, target, flow, visited) {
                    path.insert(0, current);
                    return Some(path);
                }
            }
        }
    }

    None
}

// ── Capacity Scaling ─────────────────────────────────────────────────────────

/// Capacity scaling max flow algorithm.
pub fn capacity_scaling(network: &mut FlowNetwork, source: usize, sink: usize) -> MaxFlowResult {
    network.reset_flow();

    // Find maximum capacity
    let mut max_cap = 0.0_f64;
    for neighbors in network.capacity.values() {
        for &cap in neighbors.values() {
            if cap > max_cap {
                max_cap = cap;
            }
        }
    }

    if max_cap == 0.0 {
        return MaxFlowResult {
            max_flow: 0.0,
            edge_flows: Vec::new(),
        };
    }

    // Start with largest power of 2 <= max_cap
    let mut delta = 1.0_f64;
    while delta * 2.0 <= max_cap {
        delta *= 2.0;
    }

    let mut total_flow = 0.0;

    while delta >= 1.0 {
        // Find augmenting paths with bottleneck >= delta
        loop {
            let path = bfs_with_delta(network, source, sink, delta);
            match path {
                Some((p, bottleneck)) => {
                    for i in 0..p.len() - 1 {
                        let from = p[i];
                        let to = p[i + 1];
                        *network
                            .flow
                            .entry(from)
                            .or_default()
                            .entry(to)
                            .or_insert(0.0) += bottleneck;
                        *network
                            .flow
                            .entry(to)
                            .or_default()
                            .entry(from)
                            .or_insert(0.0) -= bottleneck;
                    }
                    total_flow += bottleneck;
                }
                None => break,
            }
        }
        delta /= 2.0;
    }

    let edge_flows = collect_edge_flows(network);

    MaxFlowResult {
        max_flow: total_flow,
        edge_flows,
    }
}

fn bfs_with_delta(
    network: &FlowNetwork,
    source: usize,
    sink: usize,
    delta: f64,
) -> Option<(Vec<usize>, f64)> {
    let mut visited = HashSet::new();
    let mut parent: HashMap<usize, usize> = HashMap::new();
    let mut queue = VecDeque::new();

    visited.insert(source);
    queue.push_back(source);

    while let Some(node) = queue.pop_front() {
        if node == sink {
            let mut path = vec![sink];
            let mut cur = sink;
            while cur != source {
                let p = parent[&cur];
                path.push(p);
                cur = p;
            }
            path.reverse();

            let mut bottleneck = f64::INFINITY;
            for i in 0..path.len() - 1 {
                let rc = network.residual_capacity(path[i], path[i + 1]);
                if rc < bottleneck {
                    bottleneck = rc;
                }
            }
            return Some((path, bottleneck));
        }

        let mut neighbors: Vec<usize> = Vec::new();
        if let Some(nbs) = network.capacity.get(&node) {
            for &to in nbs.keys() {
                if network.residual_capacity(node, to) >= delta {
                    neighbors.push(to);
                }
            }
        }
        neighbors.sort();
        for nb in neighbors {
            if visited.insert(nb) {
                parent.insert(nb, node);
                queue.push_back(nb);
            }
        }
    }

    None
}

// ── Multi-Source / Multi-Sink Reduction ───────────────────────────────────────

/// Create a network with a super-source and super-sink for multi-source/multi-sink problems.
/// Returns the new network and the (super_source, super_sink) node ids.
pub fn multi_source_multi_sink(
    original: &FlowNetwork,
    sources: &[usize],
    sinks: &[usize],
    source_caps: &[f64],
    sink_caps: &[f64],
) -> (FlowNetwork, usize, usize) {
    let n = original.node_count();
    let super_source = n;
    let super_sink = n + 1;
    let mut network = FlowNetwork::new(n + 2);

    // Copy original edges
    for (from, to, cap, _) in original.edges() {
        network.add_edge(from, to, cap);
    }

    // Connect super source to all sources
    for (i, &src) in sources.iter().enumerate() {
        let cap = if i < source_caps.len() { source_caps[i] } else { f64::INFINITY };
        network.add_edge(super_source, src, cap);
    }

    // Connect all sinks to super sink
    for (i, &snk) in sinks.iter().enumerate() {
        let cap = if i < sink_caps.len() { sink_caps[i] } else { f64::INFINITY };
        network.add_edge(snk, super_sink, cap);
    }

    (network, super_source, super_sink)
}

/// Get the residual graph as a list of (from, to, residual_capacity).
pub fn residual_graph(network: &FlowNetwork) -> Vec<(usize, usize, f64)> {
    let mut result = Vec::new();
    let mut sorted_froms: Vec<usize> = network.capacity.keys().copied().collect();
    sorted_froms.sort();
    for from in sorted_froms {
        if let Some(neighbors) = network.capacity.get(&from) {
            let mut sorted_tos: Vec<usize> = neighbors.keys().copied().collect();
            sorted_tos.sort();
            for to in sorted_tos {
                let rc = network.residual_capacity(from, to);
                if rc > 0.0 {
                    result.push((from, to, rc));
                }
            }
        }
    }
    result
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_network() -> FlowNetwork {
        // s(0) -> a(1): 10, s(0) -> b(2): 10
        // a(1) -> b(2): 2, a(1) -> t(3): 8
        // b(2) -> t(3): 10
        let mut n = FlowNetwork::new(4);
        n.add_edge(0, 1, 10.0);
        n.add_edge(0, 2, 10.0);
        n.add_edge(1, 2, 2.0);
        n.add_edge(1, 3, 8.0);
        n.add_edge(2, 3, 10.0);
        n
    }

    #[test]
    fn test_edmonds_karp_max_flow() {
        let mut n = simple_network();
        let result = edmonds_karp(&mut n, 0, 3);
        assert_eq!(result.max_flow, 18.0);
    }

    #[test]
    fn test_capacity_scaling_max_flow() {
        let mut n = simple_network();
        let result = capacity_scaling(&mut n, 0, 3);
        assert_eq!(result.max_flow, 18.0);
    }

    #[test]
    fn test_min_cut() {
        let mut n = simple_network();
        let result = min_cut(&mut n, 0, 3);
        assert_eq!(result.cut_value, 18.0);
        assert!(result.source_side.contains(&0));
        assert!(result.sink_side.contains(&3));
    }

    #[test]
    fn test_flow_decomposition() {
        let mut n = simple_network();
        let _ = edmonds_karp(&mut n, 0, 3);
        let paths = flow_decomposition(&n, 0, 3);
        let total: f64 = paths.iter().map(|p| p.flow).sum();
        assert_eq!(total, 18.0);
        for p in &paths {
            assert_eq!(p.path[0], 0);
            assert_eq!(*p.path.last().unwrap(), 3);
        }
    }

    #[test]
    fn test_zero_flow() {
        let mut n = FlowNetwork::new(2);
        // No edges
        let result = edmonds_karp(&mut n, 0, 1);
        assert_eq!(result.max_flow, 0.0);
    }

    #[test]
    fn test_single_edge() {
        let mut n = FlowNetwork::new(2);
        n.add_edge(0, 1, 5.0);
        let result = edmonds_karp(&mut n, 0, 1);
        assert_eq!(result.max_flow, 5.0);
    }

    #[test]
    fn test_parallel_edges() {
        let mut n = FlowNetwork::new(2);
        n.add_edge(0, 1, 3.0);
        n.add_edge(0, 1, 4.0);
        let result = edmonds_karp(&mut n, 0, 1);
        assert_eq!(result.max_flow, 7.0);
    }

    #[test]
    fn test_residual_capacity() {
        let n = simple_network();
        assert_eq!(n.residual_capacity(0, 1), 10.0);
        assert_eq!(n.residual_capacity(1, 0), 0.0);
    }

    #[test]
    fn test_network_edges() {
        let n = simple_network();
        let edges = n.edges();
        assert_eq!(edges.len(), 5);
    }

    #[test]
    fn test_multi_source_multi_sink() {
        let mut orig = FlowNetwork::new(4);
        orig.add_edge(0, 2, 5.0);
        orig.add_edge(1, 2, 3.0);
        orig.add_edge(2, 3, 8.0);

        let (mut network, ss, st) = multi_source_multi_sink(
            &orig,
            &[0, 1],
            &[3],
            &[10.0, 10.0],
            &[10.0],
        );
        let result = edmonds_karp(&mut network, ss, st);
        assert_eq!(result.max_flow, 8.0);
    }

    #[test]
    fn test_residual_graph_fn() {
        let mut n = FlowNetwork::new(2);
        n.add_edge(0, 1, 5.0);
        let rg = residual_graph(&n);
        assert!(rg.iter().any(|&(f, t, _)| f == 0 && t == 1));
    }

    #[test]
    fn test_min_cut_edges() {
        let mut n = simple_network();
        let result = min_cut(&mut n, 0, 3);
        let cut_cap: f64 = result.cut_edges.iter().map(|(_, _, c)| c).sum();
        assert_eq!(cut_cap, result.cut_value);
    }

    #[test]
    fn test_diamond_network() {
        // s -> a: 3, s -> b: 3, a -> t: 2, b -> t: 2, a -> b: 2
        let mut n = FlowNetwork::new(4);
        n.add_edge(0, 1, 3.0);
        n.add_edge(0, 2, 3.0);
        n.add_edge(1, 3, 2.0);
        n.add_edge(2, 3, 2.0);
        n.add_edge(1, 2, 2.0);
        let result = edmonds_karp(&mut n, 0, 3);
        assert_eq!(result.max_flow, 4.0);
    }

    #[test]
    fn test_flow_conservation() {
        let mut n = simple_network();
        let result = edmonds_karp(&mut n, 0, 3);
        // For each intermediate node, flow in == flow out
        for node in 1..3 {
            let flow_in: f64 = result
                .edge_flows
                .iter()
                .filter(|(_, to, _)| *to == node)
                .map(|(_, _, f)| f)
                .sum();
            let flow_out: f64 = result
                .edge_flows
                .iter()
                .filter(|(from, _, _)| *from == node)
                .map(|(_, _, f)| f)
                .sum();
            assert!(
                (flow_in - flow_out).abs() < 1e-9,
                "flow not conserved at node {}: in={}, out={}",
                node,
                flow_in,
                flow_out,
            );
        }
    }

    #[test]
    fn test_capacity_scaling_agrees() {
        let mut n1 = simple_network();
        let mut n2 = simple_network();
        let r1 = edmonds_karp(&mut n1, 0, 3);
        let r2 = capacity_scaling(&mut n2, 0, 3);
        assert_eq!(r1.max_flow, r2.max_flow);
    }
}
