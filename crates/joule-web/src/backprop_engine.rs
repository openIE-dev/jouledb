//! Automatic differentiation and backpropagation engine.
//!
//! Builds a computational graph of operations, then performs reverse-mode
//! automatic differentiation (backpropagation) to compute gradients of a
//! scalar loss with respect to all upstream parameters:
//!
//! - [`ComputeGraph`] — directed acyclic graph of operations
//! - [`Node`] — a value in the graph with tracked lineage
//! - [`GradTape`] — records operations for backward pass
//! - [`GradAccumulator`] — collects gradients across micro-batches

use std::collections::HashMap;
use std::fmt;

// ── Node Identifiers ───────────────────────────────────────────────

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    pub fn index(self) -> u64 {
        self.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

// ── Operations ─────────────────────────────────────────────────────

/// Differentiable operations that can appear in the compute graph.
#[derive(Debug, Clone)]
pub enum Op {
    /// Leaf parameter — no parent operations.
    Parameter,
    /// Constant — no gradient flows through.
    Constant,
    /// Element-wise addition: c = a + b
    Add(NodeId, NodeId),
    /// Element-wise multiplication: c = a * b
    Mul(NodeId, NodeId),
    /// Matrix multiply (flattened): c = A @ B
    MatMul(NodeId, NodeId, usize, usize, usize),
    /// Element-wise ReLU: c = max(0, a)
    Relu(NodeId),
    /// Element-wise sigmoid: c = 1 / (1 + exp(-a))
    Sigmoid(NodeId),
    /// Element-wise tanh
    Tanh(NodeId),
    /// Scalar multiply: c = scalar * a
    ScalarMul(NodeId, f64),
    /// Sum reduction to scalar
    Sum(NodeId),
    /// Negation: c = -a
    Neg(NodeId),
    /// Power: c = a^n
    Pow(NodeId, f64),
}

// ── Node ───────────────────────────────────────────────────────────

/// A node in the computational graph holding a value and its creating op.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub value: Vec<f64>,
    pub op: Op,
    pub requires_grad: bool,
}

impl Node {
    pub fn scalar(&self) -> f64 {
        assert_eq!(self.value.len(), 1, "not a scalar node");
        self.value[0]
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node({}, dim={}, op={:?})",
            self.id.0,
            self.value.len(),
            std::mem::discriminant(&self.op)
        )
    }
}

// ── Compute Graph ──────────────────────────────────────────────────

/// Directed acyclic computation graph for forward and backward passes.
///
/// Nodes are added in topological order (parents before children).
/// The backward pass walks nodes in reverse to propagate gradients.
pub struct ComputeGraph {
    nodes: Vec<Node>,
    next_id: u64,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Create a trainable parameter node.
    pub fn parameter(&mut self, value: Vec<f64>) -> NodeId {
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Parameter,
            requires_grad: true,
        });
        id
    }

    /// Create a constant node (no gradient).
    pub fn constant(&mut self, value: Vec<f64>) -> NodeId {
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Constant,
            requires_grad: false,
        });
        id
    }

    /// Element-wise addition.
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let vb = &self.nodes[b.0 as usize].value;
        assert_eq!(va.len(), vb.len(), "add: dimension mismatch");
        let value: Vec<f64> = va.iter().zip(vb.iter()).map(|(x, y)| x + y).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Add(a, b),
            requires_grad: true,
        });
        id
    }

    /// Element-wise multiplication.
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let vb = &self.nodes[b.0 as usize].value;
        assert_eq!(va.len(), vb.len(), "mul: dimension mismatch");
        let value: Vec<f64> = va.iter().zip(vb.iter()).map(|(x, y)| x * y).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Mul(a, b),
            requires_grad: true,
        });
        id
    }

    /// Matrix multiplication: A(m×k) @ B(k×n) → C(m×n).
    /// Values are stored in row-major order.
    pub fn matmul(&mut self, a: NodeId, b: NodeId, m: usize, k: usize, n: usize) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let vb = &self.nodes[b.0 as usize].value;
        assert_eq!(va.len(), m * k, "matmul: A dimension mismatch");
        assert_eq!(vb.len(), k * n, "matmul: B dimension mismatch");

        let mut value = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += va[i * k + p] * vb[p * n + j];
                }
                value[i * n + j] = sum;
            }
        }

        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::MatMul(a, b, m, k, n),
            requires_grad: true,
        });
        id
    }

    /// Element-wise ReLU activation.
    pub fn relu(&mut self, a: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value: Vec<f64> = va.iter().map(|x| x.max(0.0)).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Relu(a),
            requires_grad: true,
        });
        id
    }

    /// Element-wise sigmoid activation.
    pub fn sigmoid(&mut self, a: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value: Vec<f64> = va.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Sigmoid(a),
            requires_grad: true,
        });
        id
    }

    /// Element-wise tanh activation.
    pub fn tanh(&mut self, a: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value: Vec<f64> = va.iter().map(|x| x.tanh()).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Tanh(a),
            requires_grad: true,
        });
        id
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&mut self, a: NodeId, scalar: f64) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value: Vec<f64> = va.iter().map(|x| x * scalar).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::ScalarMul(a, scalar),
            requires_grad: true,
        });
        id
    }

    /// Sum all elements to a scalar.
    pub fn sum(&mut self, a: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value = vec![va.iter().sum::<f64>()];
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Sum(a),
            requires_grad: true,
        });
        id
    }

    /// Negation: -a.
    pub fn neg(&mut self, a: NodeId) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value: Vec<f64> = va.iter().map(|x| -x).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Neg(a),
            requires_grad: true,
        });
        id
    }

    /// Element-wise power.
    pub fn pow(&mut self, a: NodeId, exponent: f64) -> NodeId {
        let va = &self.nodes[a.0 as usize].value;
        let value: Vec<f64> = va.iter().map(|x| x.powf(exponent)).collect();
        let id = self.alloc_id();
        self.nodes.push(Node {
            id,
            value,
            op: Op::Pow(a, exponent),
            requires_grad: true,
        });
        id
    }

    /// Read a node's forward-pass value.
    pub fn value(&self, id: NodeId) -> &[f64] {
        &self.nodes[id.0 as usize].value
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Backward pass: compute gradients for all nodes, starting from `loss_id`.
    /// The loss node must be scalar (single element).
    pub fn backward(&self, loss_id: NodeId) -> HashMap<NodeId, Vec<f64>> {
        let loss_node = &self.nodes[loss_id.0 as usize];
        assert_eq!(loss_node.value.len(), 1, "backward: loss must be scalar");

        let mut grads: HashMap<NodeId, Vec<f64>> = HashMap::new();
        // Seed: dL/dL = 1.0
        grads.insert(loss_id, vec![1.0]);

        // Walk nodes in reverse topological order
        for idx in (0..=loss_id.0 as usize).rev() {
            let node = &self.nodes[idx];
            let node_id = node.id;

            let upstream = match grads.get(&node_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            match &node.op {
                Op::Parameter | Op::Constant => {}
                Op::Add(a, b) => {
                    // d(a+b)/da = 1, d(a+b)/db = 1
                    accumulate_grad(&mut grads, *a, &upstream);
                    accumulate_grad(&mut grads, *b, &upstream);
                }
                Op::Mul(a, b) => {
                    // d(a*b)/da = b, d(a*b)/db = a
                    let va = &self.nodes[a.0 as usize].value;
                    let vb = &self.nodes[b.0 as usize].value;
                    let ga: Vec<f64> =
                        upstream.iter().zip(vb.iter()).map(|(u, v)| u * v).collect();
                    let gb: Vec<f64> =
                        upstream.iter().zip(va.iter()).map(|(u, v)| u * v).collect();
                    accumulate_grad(&mut grads, *a, &ga);
                    accumulate_grad(&mut grads, *b, &gb);
                }
                Op::MatMul(a, b, m, k, n) => {
                    // dL/dA = dL/dC @ B^T,  dL/dB = A^T @ dL/dC
                    let vb = &self.nodes[b.0 as usize].value;
                    let va = &self.nodes[a.0 as usize].value;
                    let mut ga = vec![0.0; m * k];
                    let mut gb = vec![0.0; k * n];
                    // ga = upstream(m×n) @ B^T(n×k)
                    for i in 0..*m {
                        for j in 0..*k {
                            let mut s = 0.0;
                            for p in 0..*n {
                                s += upstream[i * n + p] * vb[j * n + p];
                            }
                            ga[i * k + j] = s;
                        }
                    }
                    // gb = A^T(k×m) @ upstream(m×n)
                    for i in 0..*k {
                        for j in 0..*n {
                            let mut s = 0.0;
                            for p in 0..*m {
                                s += va[p * k + i] * upstream[p * n + j];
                            }
                            gb[i * n + j] = s;
                        }
                    }
                    accumulate_grad(&mut grads, *a, &ga);
                    accumulate_grad(&mut grads, *b, &gb);
                }
                Op::Relu(a) => {
                    let va = &self.nodes[a.0 as usize].value;
                    let ga: Vec<f64> = upstream
                        .iter()
                        .zip(va.iter())
                        .map(|(u, v)| if *v > 0.0 { *u } else { 0.0 })
                        .collect();
                    accumulate_grad(&mut grads, *a, &ga);
                }
                Op::Sigmoid(a) => {
                    let sig = &node.value;
                    let ga: Vec<f64> = upstream
                        .iter()
                        .zip(sig.iter())
                        .map(|(u, s)| u * s * (1.0 - s))
                        .collect();
                    accumulate_grad(&mut grads, *a, &ga);
                }
                Op::Tanh(a) => {
                    let th = &node.value;
                    let ga: Vec<f64> = upstream
                        .iter()
                        .zip(th.iter())
                        .map(|(u, t)| u * (1.0 - t * t))
                        .collect();
                    accumulate_grad(&mut grads, *a, &ga);
                }
                Op::ScalarMul(a, scalar) => {
                    let ga: Vec<f64> = upstream.iter().map(|u| u * scalar).collect();
                    accumulate_grad(&mut grads, *a, &ga);
                }
                Op::Sum(a) => {
                    let dim = self.nodes[a.0 as usize].value.len();
                    let ga = vec![upstream[0]; dim];
                    accumulate_grad(&mut grads, *a, &ga);
                }
                Op::Neg(a) => {
                    let ga: Vec<f64> = upstream.iter().map(|u| -u).collect();
                    accumulate_grad(&mut grads, *a, &ga);
                }
                Op::Pow(a, exponent) => {
                    let va = &self.nodes[a.0 as usize].value;
                    let ga: Vec<f64> = upstream
                        .iter()
                        .zip(va.iter())
                        .map(|(u, x)| u * exponent * x.powf(exponent - 1.0))
                        .collect();
                    accumulate_grad(&mut grads, *a, &ga);
                }
            }
        }

        grads
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ComputeGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComputeGraph(nodes={})", self.nodes.len())
    }
}

// ── Gradient Accumulator ───────────────────────────────────────────

/// Accumulates gradients from multiple backward passes (e.g., micro-batches)
/// and provides the averaged result for a parameter update.
pub struct GradAccumulator {
    accumulated: HashMap<NodeId, Vec<f64>>,
    count: u64,
}

impl GradAccumulator {
    pub fn new() -> Self {
        Self {
            accumulated: HashMap::new(),
            count: 0,
        }
    }

    /// Add gradients from one backward pass.
    pub fn add(&mut self, grads: &HashMap<NodeId, Vec<f64>>) {
        for (id, grad) in grads {
            let entry = self.accumulated.entry(*id).or_insert_with(|| vec![0.0; grad.len()]);
            for (a, g) in entry.iter_mut().zip(grad.iter()) {
                *a += g;
            }
        }
        self.count += 1;
    }

    /// Return averaged gradients and reset the accumulator.
    pub fn take_averaged(&mut self) -> HashMap<NodeId, Vec<f64>> {
        let scale = 1.0 / self.count.max(1) as f64;
        let mut result = HashMap::new();
        for (id, grad) in self.accumulated.drain() {
            result.insert(id, grad.iter().map(|g| g * scale).collect());
        }
        self.count = 0;
        result
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn reset(&mut self) {
        self.accumulated.clear();
        self.count = 0;
    }
}

impl Default for GradAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for GradAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradAccumulator(params={}, passes={})",
            self.accumulated.len(),
            self.count
        )
    }
}

// ── GradTape ───────────────────────────────────────────────────────

/// Lightweight gradient recording tape that stores (node_id, gradient) pairs
/// from a single backward pass, providing iteration and lookup.
pub struct GradTape {
    entries: Vec<(NodeId, Vec<f64>)>,
}

impl GradTape {
    pub fn from_backward(grads: HashMap<NodeId, Vec<f64>>) -> Self {
        let mut entries: Vec<_> = grads.into_iter().collect();
        entries.sort_by_key(|(id, _)| id.0);
        Self { entries }
    }

    pub fn get(&self, id: NodeId) -> Option<&[f64]> {
        self.entries
            .iter()
            .find(|(nid, _)| *nid == id)
            .map(|(_, g)| g.as_slice())
    }

    pub fn iter(&self) -> impl Iterator<Item = &(NodeId, Vec<f64>)> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Display for GradTape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GradTape(entries={})", self.entries.len())
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn accumulate_grad(grads: &mut HashMap<NodeId, Vec<f64>>, id: NodeId, incoming: &[f64]) {
    let entry = grads.entry(id).or_insert_with(|| vec![0.0; incoming.len()]);
    for (a, g) in entry.iter_mut().zip(incoming.iter()) {
        *a += g;
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_add_forward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0, 2.0]);
        let b = g.parameter(vec![3.0, 4.0]);
        let c = g.add(a, b);
        assert_eq!(g.value(c), &[4.0, 6.0]);
    }

    #[test]
    fn simple_add_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0]);
        let b = g.parameter(vec![2.0]);
        let c = g.add(a, b);
        let loss = g.sum(c);
        let grads = g.backward(loss);
        assert!((grads[&a][0] - 1.0).abs() < 1e-10);
        assert!((grads[&b][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mul_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![3.0]);
        let b = g.parameter(vec![4.0]);
        let c = g.mul(a, b);
        let loss = g.sum(c);
        let grads = g.backward(loss);
        // d(a*b)/da = b = 4, d(a*b)/db = a = 3
        assert!((grads[&a][0] - 4.0).abs() < 1e-10);
        assert!((grads[&b][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn relu_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![2.0, -1.0]);
        let b = g.relu(a);
        let loss = g.sum(b);
        let grads = g.backward(loss);
        // ReLU grad: 1 if x>0, 0 otherwise
        assert!((grads[&a][0] - 1.0).abs() < 1e-10);
        assert!((grads[&a][1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_forward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![0.0]);
        let b = g.sigmoid(a);
        assert!((g.value(b)[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn sigmoid_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![0.0]);
        let b = g.sigmoid(a);
        let loss = g.sum(b);
        let grads = g.backward(loss);
        // sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert!((grads[&a][0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn tanh_forward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![0.0]);
        let b = g.tanh(a);
        assert!((g.value(b)[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn scalar_mul_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![2.0]);
        let b = g.scalar_mul(a, 3.0);
        let loss = g.sum(b);
        let grads = g.backward(loss);
        assert!((grads[&a][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn neg_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![5.0]);
        let b = g.neg(a);
        let loss = g.sum(b);
        let grads = g.backward(loss);
        assert!((grads[&a][0] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn pow_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![3.0]);
        let b = g.pow(a, 2.0); // x^2
        let loss = g.sum(b);
        let grads = g.backward(loss);
        // d(x^2)/dx = 2x = 6
        assert!((grads[&a][0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_2x2() {
        let mut g = ComputeGraph::new();
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        let a = g.parameter(vec![1.0, 2.0, 3.0, 4.0]);
        let b = g.parameter(vec![5.0, 6.0, 7.0, 8.0]);
        let c = g.matmul(a, b, 2, 2, 2);
        // C = [[19,22],[43,50]]
        assert!((g.value(c)[0] - 19.0).abs() < 1e-10);
        assert!((g.value(c)[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_backward() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0, 2.0]); // 1x2
        let b = g.parameter(vec![3.0, 4.0]); // 2x1
        let c = g.matmul(a, b, 1, 2, 1);
        let loss = g.sum(c);
        let grads = g.backward(loss);
        // dL/dA = 1 @ B^T = [3, 4]
        assert!((grads[&a][0] - 3.0).abs() < 1e-10);
        assert!((grads[&a][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn chain_rule_composite() {
        // f(x) = (x * x) + x, df/dx = 2x + 1
        let mut g = ComputeGraph::new();
        let x = g.parameter(vec![3.0]);
        let x2 = g.mul(x, x);
        let y = g.add(x2, x);
        let loss = g.sum(y);
        let grads = g.backward(loss);
        // df/dx at x=3: 2*3 + 1 = 7
        assert!((grads[&x][0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn constant_no_grad() {
        let mut g = ComputeGraph::new();
        let c = g.constant(vec![5.0]);
        let x = g.parameter(vec![2.0]);
        let y = g.mul(c, x);
        let loss = g.sum(y);
        let grads = g.backward(loss);
        // dL/dx = c = 5
        assert!((grads[&x][0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn grad_tape_lookup() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![2.0]);
        let b = g.pow(a, 3.0);
        let loss = g.sum(b);
        let raw_grads = g.backward(loss);
        let tape = GradTape::from_backward(raw_grads);
        let grad_a = tape.get(a).unwrap();
        // d(x^3)/dx = 3x^2 = 12
        assert!((grad_a[0] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn grad_accumulator_averaging() {
        let mut acc = GradAccumulator::new();
        let nid = NodeId(0);
        let mut g1 = HashMap::new();
        g1.insert(nid, vec![2.0]);
        acc.add(&g1);
        let mut g2 = HashMap::new();
        g2.insert(nid, vec![4.0]);
        acc.add(&g2);
        let avg = acc.take_averaged();
        assert!((avg[&nid][0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn graph_display() {
        let g = ComputeGraph::new();
        assert!(format!("{g}").contains("ComputeGraph"));
    }

    #[test]
    fn node_id_display() {
        let id = NodeId(42);
        assert!(format!("{id}").contains("42"));
    }

    #[test]
    fn graph_node_count() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0]);
        let b = g.parameter(vec![2.0]);
        let _ = g.add(a, b);
        assert_eq!(g.node_count(), 3);
    }
}
