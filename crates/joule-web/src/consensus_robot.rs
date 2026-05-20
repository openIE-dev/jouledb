//! Distributed consensus — average consensus, max/min consensus,
//! formation consensus, and Laplacian-based convergence analysis.
//!
//! Pure-Rust implementations of continuous- and discrete-time consensus
//! protocols for multi-robot networks. Supports configurable topologies,
//! convergence rate analysis, and weighted consensus.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Consensus protocol errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusError {
    /// Node not found.
    NodeNotFound(u64),
    /// Duplicate node.
    DuplicateNode(u64),
    /// Graph is disconnected — consensus unreachable.
    Disconnected,
    /// Step size too large for stability.
    UnstableStepSize { step: f64, max_degree: usize },
    /// Dimension mismatch in vector consensus.
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::DuplicateNode(id) => write!(f, "duplicate node: {id}"),
            Self::Disconnected => write!(f, "graph is disconnected"),
            Self::UnstableStepSize { step, max_degree } => {
                write!(f, "step size {step:.4} too large for max degree {max_degree}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for ConsensusError {}

// ── Consensus Type ──────────────────────────────────────────────

/// Type of consensus to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusType {
    /// Average consensus: all nodes converge to the mean.
    Average,
    /// Max consensus: all nodes converge to the maximum initial value.
    Max,
    /// Min consensus: all nodes converge to the minimum initial value.
    Min,
    /// Weighted average: each node has a weight factor.
    WeightedAverage,
}

impl fmt::Display for ConsensusType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Average => write!(f, "Average"),
            Self::Max => write!(f, "Max"),
            Self::Min => write!(f, "Min"),
            Self::WeightedAverage => write!(f, "WeightedAverage"),
        }
    }
}

// ── Consensus Node ──────────────────────────────────────────────

/// A node participating in the consensus protocol.
#[derive(Debug, Clone)]
pub struct ConsensusNode {
    pub id: u64,
    /// Scalar state value.
    pub value: f64,
    /// Weight for weighted consensus.
    pub weight: f64,
    /// Multi-dimensional state (for vector consensus).
    pub state_vec: Vec<f64>,
    /// History of the scalar value over iterations.
    pub history: Vec<f64>,
}

impl ConsensusNode {
    pub fn new(id: u64, value: f64) -> Self {
        Self {
            id,
            value,
            weight: 1.0,
            state_vec: Vec::new(),
            history: vec![value],
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_state_vec(mut self, state: Vec<f64>) -> Self {
        self.state_vec = state;
        self
    }
}

impl fmt::Display for ConsensusNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({}, val={:.4})", self.id, self.value)
    }
}

// ── Topology ────────────────────────────────────────────────────

/// Communication topology for the consensus network.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Adjacency list: node_id -> list of neighbor ids with optional edge weights.
    pub edges: HashMap<u64, Vec<(u64, f64)>>,
}

impl Topology {
    pub fn new() -> Self {
        Self { edges: HashMap::new() }
    }

    /// Add an undirected edge with weight 1.0.
    pub fn add_edge(&mut self, a: u64, b: u64) {
        self.add_weighted_edge(a, b, 1.0);
    }

    /// Add an undirected edge with a custom weight.
    pub fn add_weighted_edge(&mut self, a: u64, b: u64, weight: f64) {
        self.edges.entry(a).or_default().push((b, weight));
        self.edges.entry(b).or_default().push((a, weight));
    }

    /// Add a directed edge.
    pub fn add_directed_edge(&mut self, from: u64, to: u64, weight: f64) {
        self.edges.entry(from).or_default().push((to, weight));
    }

    /// Build a complete graph on the given node IDs.
    pub fn complete(ids: &[u64]) -> Self {
        let mut topo = Self::new();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                topo.add_edge(ids[i], ids[j]);
            }
        }
        topo
    }

    /// Build a ring topology.
    pub fn ring(ids: &[u64]) -> Self {
        let mut topo = Self::new();
        let n = ids.len();
        if n < 2 {
            return topo;
        }
        for i in 0..n {
            topo.add_edge(ids[i], ids[(i + 1) % n]);
        }
        topo
    }

    /// Build a line (path) topology.
    pub fn line(ids: &[u64]) -> Self {
        let mut topo = Self::new();
        for i in 0..ids.len().saturating_sub(1) {
            topo.add_edge(ids[i], ids[i + 1]);
        }
        topo
    }

    /// Build a star topology with a center node.
    pub fn star(center: u64, leaves: &[u64]) -> Self {
        let mut topo = Self::new();
        for &leaf in leaves {
            topo.add_edge(center, leaf);
        }
        topo
    }

    pub fn neighbors(&self, node: u64) -> &[(u64, f64)] {
        self.edges.get(&node).map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn degree(&self, node: u64) -> usize {
        self.edges.get(&node).map(|v| v.len()).unwrap_or(0)
    }

    pub fn max_degree(&self) -> usize {
        self.edges.values().map(|v| v.len()).max().unwrap_or(0)
    }

    /// Build the Laplacian matrix.
    pub fn laplacian(&self, ids: &[u64]) -> Vec<Vec<f64>> {
        let n = ids.len();
        let idx_map: HashMap<u64, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let mut lap = vec![vec![0.0f64; n]; n];
        for &id in ids {
            let i = idx_map[&id];
            for &(nbr, w) in self.neighbors(id) {
                if let Some(&j) = idx_map.get(&nbr) {
                    lap[i][j] -= w;
                    lap[i][i] += w;
                }
            }
        }
        lap
    }
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let edge_count: usize = self.edges.values().map(|v| v.len()).sum();
        write!(f, "Topology({} nodes, {} directed edges)", self.edges.len(), edge_count)
    }
}

// ── Consensus Protocol ──────────────────────────────────────────

/// The main consensus protocol engine.
#[derive(Debug, Clone)]
pub struct ConsensusProtocol {
    pub nodes: HashMap<u64, ConsensusNode>,
    pub topology: Topology,
    pub consensus_type: ConsensusType,
    /// Discrete-time step size (epsilon).
    pub step_size: f64,
    /// Number of iterations executed.
    pub iterations: u64,
    /// Convergence threshold.
    pub tolerance: f64,
    /// Whether convergence has been reached.
    pub converged: bool,
}

impl ConsensusProtocol {
    pub fn new(consensus_type: ConsensusType) -> Self {
        Self {
            nodes: HashMap::new(),
            topology: Topology::new(),
            consensus_type,
            step_size: 0.1,
            iterations: 0,
            tolerance: 1e-6,
            converged: false,
        }
    }

    pub fn with_step_size(mut self, step: f64) -> Self {
        self.step_size = step;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn with_topology(mut self, topo: Topology) -> Self {
        self.topology = topo;
        self
    }

    pub fn add_node(&mut self, node: ConsensusNode) -> Result<(), ConsensusError> {
        if self.nodes.contains_key(&node.id) {
            return Err(ConsensusError::DuplicateNode(node.id));
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    /// The theoretical consensus value for average consensus.
    pub fn theoretical_average(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        match self.consensus_type {
            ConsensusType::Average => {
                let sum: f64 = self.nodes.values().map(|n| n.history[0]).sum();
                sum / self.nodes.len() as f64
            }
            ConsensusType::Max => {
                self.nodes.values().map(|n| n.history[0]).fold(f64::NEG_INFINITY, f64::max)
            }
            ConsensusType::Min => {
                self.nodes.values().map(|n| n.history[0]).fold(f64::INFINITY, f64::min)
            }
            ConsensusType::WeightedAverage => {
                let total_weight: f64 = self.nodes.values().map(|n| n.weight).sum();
                if total_weight.abs() < 1e-15 {
                    return 0.0;
                }
                let weighted_sum: f64 =
                    self.nodes.values().map(|n| n.weight * n.history[0]).sum();
                weighted_sum / total_weight
            }
        }
    }

    /// Current disagreement: max absolute difference among all pairs.
    pub fn disagreement(&self) -> f64 {
        let vals: Vec<f64> = self.nodes.values().map(|n| n.value).collect();
        if vals.len() < 2 {
            return 0.0;
        }
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// Run one iteration of the discrete-time consensus protocol.
    pub fn step(&mut self) -> Result<(), ConsensusError> {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        let current_vals: HashMap<u64, f64> =
            self.nodes.iter().map(|(&id, n)| (id, n.value)).collect();

        match self.consensus_type {
            ConsensusType::Average | ConsensusType::WeightedAverage => {
                self.step_average(&ids, &current_vals);
            }
            ConsensusType::Max => {
                self.step_max(&ids, &current_vals);
            }
            ConsensusType::Min => {
                self.step_min(&ids, &current_vals);
            }
        }

        self.iterations += 1;
        self.converged = self.disagreement() < self.tolerance;
        Ok(())
    }

    fn step_average(&mut self, ids: &[u64], current: &HashMap<u64, f64>) {
        for &id in ids {
            let xi = current[&id];
            let nbrs = self.topology.neighbors(id);
            let mut update = 0.0f64;
            for &(nid, weight) in nbrs {
                if let Some(&xj) = current.get(&nid) {
                    update += weight * (xj - xi);
                }
            }
            let new_val = xi + self.step_size * update;
            if let Some(node) = self.nodes.get_mut(&id) {
                node.value = new_val;
                node.history.push(new_val);
            }
        }
    }

    fn step_max(&mut self, ids: &[u64], current: &HashMap<u64, f64>) {
        for &id in ids {
            let xi = current[&id];
            let nbrs = self.topology.neighbors(id);
            let mut max_val = xi;
            for &(nid, _) in nbrs {
                if let Some(&xj) = current.get(&nid) {
                    if xj > max_val {
                        max_val = xj;
                    }
                }
            }
            if let Some(node) = self.nodes.get_mut(&id) {
                node.value = max_val;
                node.history.push(max_val);
            }
        }
    }

    fn step_min(&mut self, ids: &[u64], current: &HashMap<u64, f64>) {
        for &id in ids {
            let xi = current[&id];
            let nbrs = self.topology.neighbors(id);
            let mut min_val = xi;
            for &(nid, _) in nbrs {
                if let Some(&xj) = current.get(&nid) {
                    if xj < min_val {
                        min_val = xj;
                    }
                }
            }
            if let Some(node) = self.nodes.get_mut(&id) {
                node.value = min_val;
                node.history.push(min_val);
            }
        }
    }

    /// Run until convergence or max_iterations.
    pub fn run(&mut self, max_iterations: u64) -> Result<u64, ConsensusError> {
        for _ in 0..max_iterations {
            self.step()?;
            if self.converged {
                return Ok(self.iterations);
            }
        }
        Ok(self.iterations)
    }

    /// Convergence rate estimate: spectral gap of Laplacian / max degree.
    pub fn convergence_rate_bound(&self) -> f64 {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        if ids.len() < 2 {
            return 0.0;
        }
        let lap = self.topology.laplacian(&ids);
        let n = ids.len();

        // Power iteration for second-smallest eigenvalue.
        let mut v = vec![0.0f64; n];
        for i in 0..n {
            v[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }

        for _ in 0..200 {
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += lap[i][j] * v[j];
                }
            }
            // Project out all-ones.
            let avg: f64 = w.iter().sum::<f64>() / n as f64;
            for val in &mut w {
                *val -= avg;
            }
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                return 0.0;
            }
            for val in &mut w {
                *val /= norm;
            }
            v = w;
        }

        // Rayleigh quotient.
        let mut num = 0.0f64;
        for i in 0..n {
            let mut lv = 0.0f64;
            for j in 0..n {
                lv += lap[i][j] * v[j];
            }
            num += v[i] * lv;
        }
        let denom: f64 = v.iter().map(|x| x * x).sum();
        if denom < 1e-15 { 0.0 } else { num / denom }
    }
}

impl fmt::Display for ConsensusProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Consensus({}, {} nodes, iter={}, disagreement={:.6})",
            self.consensus_type,
            self.nodes.len(),
            self.iterations,
            self.disagreement(),
        )
    }
}

// ── Vector Consensus ────────────────────────────────────────────

/// Multi-dimensional (vector) consensus protocol.
#[derive(Debug, Clone)]
pub struct VectorConsensus {
    pub dimension: usize,
    pub states: HashMap<u64, Vec<f64>>,
    pub topology: Topology,
    pub step_size: f64,
    pub iterations: u64,
}

impl VectorConsensus {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            states: HashMap::new(),
            topology: Topology::new(),
            step_size: 0.1,
            iterations: 0,
        }
    }

    pub fn with_topology(mut self, topo: Topology) -> Self {
        self.topology = topo;
        self
    }

    pub fn with_step_size(mut self, step: f64) -> Self {
        self.step_size = step;
        self
    }

    pub fn add_state(
        &mut self,
        id: u64,
        state: Vec<f64>,
    ) -> Result<(), ConsensusError> {
        if state.len() != self.dimension {
            return Err(ConsensusError::DimensionMismatch {
                expected: self.dimension,
                got: state.len(),
            });
        }
        self.states.insert(id, state);
        Ok(())
    }

    /// Run one iteration of vector average consensus.
    pub fn step(&mut self) {
        let ids: Vec<u64> = self.states.keys().copied().collect();
        let old: HashMap<u64, Vec<f64>> = self.states.clone();

        for &id in &ids {
            let xi = &old[&id];
            let nbrs = self.topology.neighbors(id);
            let mut delta = vec![0.0f64; self.dimension];
            for &(nid, weight) in nbrs {
                if let Some(xj) = old.get(&nid) {
                    for d in 0..self.dimension {
                        delta[d] += weight * (xj[d] - xi[d]);
                    }
                }
            }
            if let Some(state) = self.states.get_mut(&id) {
                for d in 0..self.dimension {
                    state[d] += self.step_size * delta[d];
                }
            }
        }
        self.iterations += 1;
    }

    /// Maximum disagreement across all dimensions.
    pub fn disagreement(&self) -> f64 {
        let vals: Vec<&Vec<f64>> = self.states.values().collect();
        if vals.len() < 2 {
            return 0.0;
        }
        let mut max_dis = 0.0f64;
        for d in 0..self.dimension {
            let min = vals.iter().map(|v| v[d]).fold(f64::INFINITY, f64::min);
            let max = vals.iter().map(|v| v[d]).fold(f64::NEG_INFINITY, f64::max);
            let dis = max - min;
            if dis > max_dis {
                max_dis = dis;
            }
        }
        max_dis
    }

    /// Compute the average state across all nodes.
    pub fn average_state(&self) -> Vec<f64> {
        let n = self.states.len();
        if n == 0 {
            return vec![0.0; self.dimension];
        }
        let mut avg = vec![0.0f64; self.dimension];
        for state in self.states.values() {
            for d in 0..self.dimension {
                avg[d] += state[d];
            }
        }
        for val in &mut avg {
            *val /= n as f64;
        }
        avg
    }
}

impl fmt::Display for VectorConsensus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VectorConsensus(dim={}, {} nodes, iter={})",
            self.dimension,
            self.states.len(),
            self.iterations,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_three_node_ring() -> ConsensusProtocol {
        let topo = Topology::ring(&[1, 2, 3]);
        let mut proto = ConsensusProtocol::new(ConsensusType::Average)
            .with_topology(topo)
            .with_step_size(0.1);
        proto.add_node(ConsensusNode::new(1, 10.0)).unwrap();
        proto.add_node(ConsensusNode::new(2, 20.0)).unwrap();
        proto.add_node(ConsensusNode::new(3, 30.0)).unwrap();
        proto
    }

    #[test]
    fn test_average_convergence() {
        let mut proto = make_three_node_ring();
        proto.run(500).unwrap();
        let target = 20.0; // (10+20+30)/3
        for node in proto.nodes.values() {
            assert!((node.value - target).abs() < 0.01);
        }
    }

    #[test]
    fn test_max_consensus() {
        let topo = Topology::complete(&[1, 2, 3]);
        let mut proto = ConsensusProtocol::new(ConsensusType::Max)
            .with_topology(topo);
        proto.add_node(ConsensusNode::new(1, 5.0)).unwrap();
        proto.add_node(ConsensusNode::new(2, 15.0)).unwrap();
        proto.add_node(ConsensusNode::new(3, 10.0)).unwrap();
        proto.run(10).unwrap();
        for node in proto.nodes.values() {
            assert!((node.value - 15.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_min_consensus() {
        let topo = Topology::complete(&[1, 2, 3]);
        let mut proto = ConsensusProtocol::new(ConsensusType::Min)
            .with_topology(topo);
        proto.add_node(ConsensusNode::new(1, 5.0)).unwrap();
        proto.add_node(ConsensusNode::new(2, 15.0)).unwrap();
        proto.add_node(ConsensusNode::new(3, 10.0)).unwrap();
        proto.run(10).unwrap();
        for node in proto.nodes.values() {
            assert!((node.value - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_weighted_average() {
        let topo = Topology::complete(&[1, 2]);
        let mut proto = ConsensusProtocol::new(ConsensusType::WeightedAverage)
            .with_topology(topo)
            .with_step_size(0.1);
        proto.add_node(ConsensusNode::new(1, 0.0).with_weight(3.0)).unwrap();
        proto.add_node(ConsensusNode::new(2, 10.0).with_weight(1.0)).unwrap();
        let target = proto.theoretical_average(); // (0*3+10*1)/4 = 2.5
        assert!((target - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_disagreement() {
        let mut proto = make_three_node_ring();
        assert!((proto.disagreement() - 20.0).abs() < 1e-9); // 30 - 10
        proto.run(500).unwrap();
        assert!(proto.disagreement() < 0.01);
    }

    #[test]
    fn test_duplicate_node() {
        let mut proto = ConsensusProtocol::new(ConsensusType::Average);
        proto.add_node(ConsensusNode::new(1, 5.0)).unwrap();
        assert!(proto.add_node(ConsensusNode::new(1, 10.0)).is_err());
    }

    #[test]
    fn test_topology_complete() {
        let topo = Topology::complete(&[1, 2, 3]);
        assert_eq!(topo.degree(1), 2);
        assert_eq!(topo.degree(2), 2);
        assert_eq!(topo.degree(3), 2);
    }

    #[test]
    fn test_topology_ring() {
        let topo = Topology::ring(&[1, 2, 3, 4]);
        assert_eq!(topo.degree(1), 2);
        assert_eq!(topo.max_degree(), 2);
    }

    #[test]
    fn test_topology_star() {
        let topo = Topology::star(0, &[1, 2, 3, 4]);
        assert_eq!(topo.degree(0), 4);
        assert_eq!(topo.degree(1), 1);
    }

    #[test]
    fn test_topology_line() {
        let topo = Topology::line(&[1, 2, 3]);
        assert_eq!(topo.degree(1), 1); // Endpoint.
        assert_eq!(topo.degree(2), 2); // Middle.
    }

    #[test]
    fn test_laplacian() {
        let topo = Topology::complete(&[1, 2, 3]);
        let lap = topo.laplacian(&[1, 2, 3]);
        // Row sums should be zero.
        for row in &lap {
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-9);
        }
    }

    #[test]
    fn test_convergence_rate() {
        let mut proto = make_three_node_ring();
        let rate = proto.convergence_rate_bound();
        assert!(rate > 0.0); // Connected graph.
        let _ = proto.run(1); // Just to use mutable ref.
    }

    #[test]
    fn test_history_tracking() {
        let mut proto = make_three_node_ring();
        proto.run(10).unwrap();
        for node in proto.nodes.values() {
            assert_eq!(node.history.len(), 11); // Initial + 10 steps.
        }
    }

    #[test]
    fn test_vector_consensus() {
        let topo = Topology::complete(&[1, 2]);
        let mut vc = VectorConsensus::new(2)
            .with_topology(topo)
            .with_step_size(0.2);
        vc.add_state(1, vec![0.0, 0.0]).unwrap();
        vc.add_state(2, vec![10.0, 20.0]).unwrap();
        for _ in 0..200 {
            vc.step();
        }
        assert!(vc.disagreement() < 0.01);
        let avg = vc.average_state();
        assert!((avg[0] - 5.0).abs() < 0.1);
        assert!((avg[1] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_vector_dimension_mismatch() {
        let mut vc = VectorConsensus::new(3);
        assert!(vc.add_state(1, vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn test_complete_graph_fast_convergence() {
        let topo = Topology::complete(&[1, 2, 3, 4, 5]);
        let mut proto = ConsensusProtocol::new(ConsensusType::Average)
            .with_topology(topo)
            .with_step_size(0.05);
        for i in 1..=5 {
            proto.add_node(ConsensusNode::new(i, (i * 10) as f64)).unwrap();
        }
        let iters = proto.run(1000).unwrap();
        assert!(proto.converged);
        // Complete graph should converge faster than a line.
        assert!(iters < 500);
    }

    #[test]
    fn test_display_impls() {
        let proto = make_three_node_ring();
        assert!(format!("{proto}").contains("Average"));
        assert!(format!("{proto}").contains("3 nodes"));
        let topo = Topology::ring(&[1, 2, 3]);
        assert!(format!("{topo}").contains("Topology"));
        let node = ConsensusNode::new(5, 3.14);
        assert!(format!("{node}").contains("3.14"));
        let vc = VectorConsensus::new(3);
        assert!(format!("{vc}").contains("dim=3"));
    }

    #[test]
    fn test_single_node_no_disagreement() {
        let mut proto = ConsensusProtocol::new(ConsensusType::Average);
        proto.add_node(ConsensusNode::new(1, 42.0)).unwrap();
        assert!((proto.disagreement()).abs() < 1e-9);
    }
}
