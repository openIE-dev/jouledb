//! Computational graph for tensor operations with topological sort execution
//! and node fusion.
//!
//! Represents neural network computation as a directed acyclic graph of
//! operators. Supports topological ordering, constant folding, dead-node
//! elimination, and operator fusion for inference optimisation.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Tensor Value ───────────────────────────────────────────────

/// Lightweight tensor value flowing through graph edges.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorVal {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl TensorVal {
    pub fn scalar(v: f64) -> Self {
        Self { shape: vec![1], data: vec![v] }
    }

    pub fn from_vec(shape: Vec<usize>, data: Vec<f64>) -> Self {
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { shape, data: vec![0.0; n] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

impl fmt::Display for TensorVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorVal(shape={:?})", self.shape)
    }
}

// ── Op Kind ────────────────────────────────────────────────────

/// Operator type for a graph node.
#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    /// Input placeholder.
    Input,
    /// Constant value baked into the graph.
    Constant(TensorVal),
    Add,
    Mul,
    MatMul,
    Relu,
    Sigmoid,
    Conv { kernel: Vec<usize>, stride: usize, padding: usize },
    ReduceSum { axis: usize },
    FusedAddRelu,
    Reshape { target: Vec<usize> },
    Neg,
}

impl fmt::Display for OpKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpKind::Input => write!(f, "Input"),
            OpKind::Constant(_) => write!(f, "Constant"),
            OpKind::Add => write!(f, "Add"),
            OpKind::Mul => write!(f, "Mul"),
            OpKind::MatMul => write!(f, "MatMul"),
            OpKind::Relu => write!(f, "Relu"),
            OpKind::Sigmoid => write!(f, "Sigmoid"),
            OpKind::Conv { .. } => write!(f, "Conv"),
            OpKind::ReduceSum { axis } => write!(f, "ReduceSum(axis={axis})"),
            OpKind::FusedAddRelu => write!(f, "FusedAddRelu"),
            OpKind::Reshape { target } => write!(f, "Reshape({target:?})"),
            OpKind::Neg => write!(f, "Neg"),
        }
    }
}

// ── Graph Node ─────────────────────────────────────────────────

/// Unique node identifier.
pub type NodeId = usize;

/// A single node in the computation graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub name: String,
    pub op: OpKind,
    /// Indices of input nodes.
    pub inputs: Vec<NodeId>,
    /// Cached output (populated during execution).
    pub output: Option<TensorVal>,
}

impl GraphNode {
    pub fn new(id: NodeId, name: impl Into<String>, op: OpKind) -> Self {
        Self { id, name: name.into(), op, inputs: Vec::new(), output: None }
    }
}

impl fmt::Display for GraphNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Node({}, '{}', op={})", self.id, self.name, self.op)
    }
}

// ── Tensor Graph ───────────────────────────────────────────────

/// Directed acyclic computation graph.
#[derive(Debug)]
pub struct TensorGraph {
    nodes: Vec<GraphNode>,
    output_ids: Vec<NodeId>,
}

impl TensorGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), output_ids: Vec::new() }
    }

    /// Add a node, returns its `NodeId`.
    pub fn add_node(&mut self, name: impl Into<String>, op: OpKind, inputs: &[NodeId]) -> NodeId {
        let id = self.nodes.len();
        let mut node = GraphNode::new(id, name, op);
        node.inputs = inputs.to_vec();
        self.nodes.push(node);
        id
    }

    /// Mark a node as a graph output.
    pub fn mark_output(&mut self, id: NodeId) {
        if !self.output_ids.contains(&id) {
            self.output_ids.push(id);
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Topological sort via Kahn's algorithm.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, String> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adjacency: Vec<Vec<NodeId>> = vec![Vec::new(); n];

        for node in &self.nodes {
            for &inp in &node.inputs {
                adjacency[inp].push(node.id);
                in_degree[node.id] += 1;
            }
        }

        let mut queue: VecDeque<NodeId> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(node_id) = queue.pop_front() {
            order.push(node_id);
            for &next in &adjacency[node_id] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            return Err("cycle detected in computation graph".into());
        }
        Ok(order)
    }

    /// Execute the graph with the given input bindings.
    pub fn execute(&mut self, inputs: &HashMap<NodeId, TensorVal>) -> Result<Vec<TensorVal>, String> {
        let order = self.topological_sort()?;

        for &id in &order {
            let op = self.nodes[id].op.clone();
            let input_ids = self.nodes[id].inputs.clone();

            let output = match &op {
                OpKind::Input => {
                    inputs.get(&id).cloned().ok_or_else(|| format!("missing input for node {id}"))?
                }
                OpKind::Constant(val) => val.clone(),
                OpKind::Add => {
                    let a = self.get_output(input_ids[0])?;
                    let b = self.get_output(input_ids[1])?;
                    elementwise_binary(a, b, |x, y| x + y)?
                }
                OpKind::Mul => {
                    let a = self.get_output(input_ids[0])?;
                    let b = self.get_output(input_ids[1])?;
                    elementwise_binary(a, b, |x, y| x * y)?
                }
                OpKind::Neg => {
                    let a = self.get_output(input_ids[0])?;
                    TensorVal {
                        shape: a.shape.clone(),
                        data: a.data.iter().map(|v| -v).collect(),
                    }
                }
                OpKind::MatMul => {
                    let a = self.get_output(input_ids[0])?;
                    let b = self.get_output(input_ids[1])?;
                    matmul_2d(a, b)?
                }
                OpKind::Relu => {
                    let a = self.get_output(input_ids[0])?;
                    TensorVal {
                        shape: a.shape.clone(),
                        data: a.data.iter().map(|v| v.max(0.0)).collect(),
                    }
                }
                OpKind::Sigmoid => {
                    let a = self.get_output(input_ids[0])?;
                    TensorVal {
                        shape: a.shape.clone(),
                        data: a.data.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
                    }
                }
                OpKind::FusedAddRelu => {
                    let a = self.get_output(input_ids[0])?;
                    let b = self.get_output(input_ids[1])?;
                    let added = elementwise_binary(a, b, |x, y| x + y)?;
                    TensorVal {
                        shape: added.shape,
                        data: added.data.into_iter().map(|v| v.max(0.0)).collect(),
                    }
                }
                OpKind::ReduceSum { axis } => {
                    let a = self.get_output(input_ids[0])?;
                    reduce_sum(a, *axis)?
                }
                OpKind::Reshape { target } => {
                    let a = self.get_output(input_ids[0])?;
                    TensorVal { shape: target.clone(), data: a.data.clone() }
                }
                OpKind::Conv { .. } => {
                    // Simplified: treat as pass-through for shape tracking
                    self.get_output(input_ids[0])?.clone()
                }
            };
            self.nodes[id].output = Some(output);
        }

        let mut results = Vec::new();
        for &oid in &self.output_ids {
            results.push(self.get_output(oid)?.clone());
        }
        Ok(results)
    }

    fn get_output(&self, id: NodeId) -> Result<&TensorVal, String> {
        self.nodes[id]
            .output
            .as_ref()
            .ok_or_else(|| format!("node {id} has no output yet"))
    }

    /// Constant-folding: evaluate nodes whose inputs are all constants.
    pub fn fold_constants(&mut self) {
        let order = match self.topological_sort() {
            Ok(o) => o,
            Err(_) => return,
        };
        for &id in &order {
            let input_ids = self.nodes[id].inputs.clone();
            if input_ids.is_empty() { continue; }
            let all_const = input_ids.iter().all(|i| matches!(&self.nodes[*i].op, OpKind::Constant(_)));
            if !all_const { continue; }

            let mut mini = TensorGraph::new();
            let mut id_map = HashMap::new();
            let mut inputs_map = HashMap::new();
            for &inp_id in &input_ids {
                if let OpKind::Constant(ref val) = self.nodes[inp_id].op {
                    let nid = mini.add_node("c", OpKind::Input, &[]);
                    id_map.insert(inp_id, nid);
                    inputs_map.insert(nid, val.clone());
                }
            }
            let mapped: Vec<NodeId> = input_ids.iter().map(|i| id_map[i]).collect();
            let tgt = mini.add_node("t", self.nodes[id].op.clone(), &mapped);
            mini.mark_output(tgt);
            if let Ok(results) = mini.execute(&inputs_map) {
                if let Some(val) = results.into_iter().next() {
                    self.nodes[id].op = OpKind::Constant(val);
                    self.nodes[id].inputs.clear();
                }
            }
        }
    }

    /// Dead-node elimination: remove nodes not reachable from outputs.
    pub fn dead_node_count(&self) -> usize {
        let reachable = self.reachable_from_outputs();
        self.nodes.len() - reachable.len()
    }

    fn reachable_from_outputs(&self) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack: Vec<NodeId> = self.output_ids.clone();
        while let Some(id) = stack.pop() {
            if visited.insert(id) {
                for &inp in &self.nodes[id].inputs {
                    stack.push(inp);
                }
            }
        }
        visited
    }

    /// Fuse Add→Relu sequences into FusedAddRelu nodes.
    pub fn fuse_add_relu(&mut self) -> usize {
        let mut fused = 0;
        let n = self.nodes.len();
        let mut relu_to_remove: HashSet<NodeId> = HashSet::new();

        for i in 0..n {
            if matches!(self.nodes[i].op, OpKind::Relu) && self.nodes[i].inputs.len() == 1 {
                let pred_id = self.nodes[i].inputs[0];
                if matches!(self.nodes[pred_id].op, OpKind::Add) {
                    // Check pred is only used by this relu
                    let pred_consumers: usize = self.nodes.iter()
                        .filter(|nd| nd.inputs.contains(&pred_id))
                        .count();
                    if pred_consumers == 1 {
                        relu_to_remove.insert(i);
                        fused += 1;
                    }
                }
            }
        }

        for &relu_id in &relu_to_remove {
            let add_id = self.nodes[relu_id].inputs[0];
            let add_inputs = self.nodes[add_id].inputs.clone();
            self.nodes[add_id].op = OpKind::FusedAddRelu;
            // Redirect consumers of relu to add
            for node in &mut self.nodes {
                for inp in &mut node.inputs {
                    if *inp == relu_id {
                        *inp = add_id;
                    }
                }
            }
            // Update output_ids
            for oid in &mut self.output_ids {
                if *oid == relu_id {
                    *oid = add_id;
                }
            }
            let _ = add_inputs; // inputs stay on the fused node
        }

        fused
    }
}

impl Default for TensorGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TensorGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorGraph(nodes={}, outputs={})", self.nodes.len(), self.output_ids.len())
    }
}

// ── Helpers ────────────────────────────────────────────────────

fn elementwise_binary(
    a: &TensorVal,
    b: &TensorVal,
    f: impl Fn(f64, f64) -> f64,
) -> Result<TensorVal, String> {
    if a.shape != b.shape {
        return Err(format!("shape mismatch: {:?} vs {:?}", a.shape, b.shape));
    }
    let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| f(*x, *y)).collect();
    Ok(TensorVal { shape: a.shape.clone(), data })
}

fn matmul_2d(a: &TensorVal, b: &TensorVal) -> Result<TensorVal, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err("matmul requires 2-D tensors".into());
    }
    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k1 != k2 {
        return Err(format!("inner dims mismatch: {k1} vs {k2}"));
    }
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k1 {
                s += a.data[i * k1 + p] * b.data[p * n + j];
            }
            data[i * n + j] = s;
        }
    }
    Ok(TensorVal { shape: vec![m, n], data })
}

fn reduce_sum(a: &TensorVal, axis: usize) -> Result<TensorVal, String> {
    if axis >= a.shape.len() {
        return Err(format!("axis {axis} out of range for shape {:?}", a.shape));
    }
    let mut new_shape = a.shape.clone();
    new_shape[axis] = 1;
    let out_size: usize = new_shape.iter().product();
    let mut data = vec![0.0; out_size];

    let outer: usize = a.shape[..axis].iter().product();
    let inner: usize = a.shape[axis + 1..].iter().product();
    let dim = a.shape[axis];

    for o in 0..outer {
        for d in 0..dim {
            for i in 0..inner {
                let src = o * dim * inner + d * inner + i;
                let dst = o * inner + i;
                data[dst] += a.data[src];
            }
        }
    }
    Ok(TensorVal { shape: new_shape, data })
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_val_scalar() {
        let t = TensorVal::scalar(3.14);
        assert_eq!(t.numel(), 1);
        assert_eq!(t.data[0], 3.14);
    }

    #[test]
    fn test_add_nodes() {
        let mut g = TensorGraph::new();
        let a = g.add_node("a", OpKind::Input, &[]);
        let b = g.add_node("b", OpKind::Input, &[]);
        let c = g.add_node("c", OpKind::Add, &[a, b]);
        assert_eq!(g.num_nodes(), 3);
        assert_eq!(g.node(c).unwrap().inputs, vec![a, b]);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let w = g.add_node("w", OpKind::Constant(TensorVal::scalar(2.0)), &[]);
        let m = g.add_node("mul", OpKind::Mul, &[x, w]);
        g.mark_output(m);
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 3);
        // x and w must come before m
        let pos_x = order.iter().position(|i| *i == x).unwrap();
        let pos_m = order.iter().position(|i| *i == m).unwrap();
        assert!(pos_x < pos_m);
    }

    #[test]
    fn test_execute_simple_add() {
        let mut g = TensorGraph::new();
        let a = g.add_node("a", OpKind::Input, &[]);
        let b = g.add_node("b", OpKind::Input, &[]);
        let c = g.add_node("c", OpKind::Add, &[a, b]);
        g.mark_output(c);

        let mut inputs = HashMap::new();
        inputs.insert(a, TensorVal::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
        inputs.insert(b, TensorVal::from_vec(vec![3], vec![4.0, 5.0, 6.0]));

        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_execute_relu() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let r = g.add_node("relu", OpKind::Relu, &[x]);
        g.mark_output(r);

        let mut inputs = HashMap::new();
        inputs.insert(x, TensorVal::from_vec(vec![4], vec![-1.0, 0.0, 1.0, -5.0]));
        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].data, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_execute_matmul() {
        let mut g = TensorGraph::new();
        let a = g.add_node("a", OpKind::Input, &[]);
        let b = g.add_node("b", OpKind::Input, &[]);
        let m = g.add_node("mm", OpKind::MatMul, &[a, b]);
        g.mark_output(m);

        let mut inputs = HashMap::new();
        inputs.insert(a, TensorVal::from_vec(vec![1, 2], vec![1.0, 2.0]));
        inputs.insert(b, TensorVal::from_vec(vec![2, 1], vec![3.0, 4.0]));
        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].shape, vec![1, 1]);
        assert_eq!(results[0].data[0], 11.0);
    }

    #[test]
    fn test_constant_folding() {
        let mut g = TensorGraph::new();
        let c1 = g.add_node("c1", OpKind::Constant(TensorVal::from_vec(vec![2], vec![2.0, 3.0])), &[]);
        let c2 = g.add_node("c2", OpKind::Constant(TensorVal::from_vec(vec![2], vec![4.0, 5.0])), &[]);
        let add = g.add_node("add", OpKind::Add, &[c1, c2]);
        g.mark_output(add);

        g.fold_constants();
        // add node should now be a Constant
        match &g.node(add).unwrap().op {
            OpKind::Constant(val) => {
                assert_eq!(val.data, vec![6.0, 8.0]);
            }
            other => panic!("expected Constant, got {other}"),
        }
    }

    #[test]
    fn test_dead_node_count() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let _dead = g.add_node("dead", OpKind::Input, &[]);
        let r = g.add_node("relu", OpKind::Relu, &[x]);
        g.mark_output(r);
        assert_eq!(g.dead_node_count(), 1);
    }

    #[test]
    fn test_fuse_add_relu() {
        let mut g = TensorGraph::new();
        let a = g.add_node("a", OpKind::Input, &[]);
        let b = g.add_node("b", OpKind::Input, &[]);
        let add = g.add_node("add", OpKind::Add, &[a, b]);
        let relu = g.add_node("relu", OpKind::Relu, &[add]);
        g.mark_output(relu);

        let count = g.fuse_add_relu();
        assert_eq!(count, 1);
        assert!(matches!(g.node(add).unwrap().op, OpKind::FusedAddRelu));
    }

    #[test]
    fn test_fused_add_relu_execution() {
        let mut g = TensorGraph::new();
        let a = g.add_node("a", OpKind::Input, &[]);
        let b = g.add_node("b", OpKind::Input, &[]);
        let f = g.add_node("fused", OpKind::FusedAddRelu, &[a, b]);
        g.mark_output(f);

        let mut inputs = HashMap::new();
        inputs.insert(a, TensorVal::from_vec(vec![3], vec![1.0, -5.0, 3.0]));
        inputs.insert(b, TensorVal::from_vec(vec![3], vec![-2.0, 2.0, -1.0]));

        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].data, vec![0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_reduce_sum() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let r = g.add_node("sum", OpKind::ReduceSum { axis: 1 }, &[x]);
        g.mark_output(r);

        let mut inputs = HashMap::new();
        inputs.insert(x, TensorVal::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].shape, vec![2, 1]);
        assert_eq!(results[0].data, vec![6.0, 15.0]);
    }

    #[test]
    fn test_neg_op() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let n = g.add_node("neg", OpKind::Neg, &[x]);
        g.mark_output(n);

        let mut inputs = HashMap::new();
        inputs.insert(x, TensorVal::from_vec(vec![3], vec![1.0, -2.0, 0.0]));
        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].data, vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_sigmoid_op() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let s = g.add_node("sig", OpKind::Sigmoid, &[x]);
        g.mark_output(s);

        let mut inputs = HashMap::new();
        inputs.insert(x, TensorVal::from_vec(vec![1], vec![0.0]));
        let results = g.execute(&inputs).unwrap();
        assert!((results[0].data[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_display_impls() {
        let g = TensorGraph::new();
        assert!(format!("{g}").contains("TensorGraph"));

        let node = GraphNode::new(0, "test", OpKind::Add);
        assert!(format!("{node}").contains("Add"));

        let tv = TensorVal::zeros(vec![2, 3]);
        assert!(format!("{tv}").contains("[2, 3]"));
    }

    #[test]
    fn test_multiple_outputs() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let r = g.add_node("relu", OpKind::Relu, &[x]);
        let s = g.add_node("sig", OpKind::Sigmoid, &[x]);
        g.mark_output(r);
        g.mark_output(s);

        let mut inputs = HashMap::new();
        inputs.insert(x, TensorVal::from_vec(vec![2], vec![1.0, -1.0]));
        let results = g.execute(&inputs).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].data, vec![1.0, 0.0]);
        assert!(results[1].data[0] > 0.7); // sigmoid(1)
    }

    #[test]
    fn test_chain_execution() {
        let mut g = TensorGraph::new();
        let x = g.add_node("x", OpKind::Input, &[]);
        let c = g.add_node("c", OpKind::Constant(TensorVal::from_vec(vec![2], vec![2.0, 2.0])), &[]);
        let m = g.add_node("mul", OpKind::Mul, &[x, c]);
        let r = g.add_node("relu", OpKind::Relu, &[m]);
        g.mark_output(r);

        let mut inputs = HashMap::new();
        inputs.insert(x, TensorVal::from_vec(vec![2], vec![-1.0, 3.0]));
        let results = g.execute(&inputs).unwrap();
        assert_eq!(results[0].data, vec![0.0, 6.0]);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let mut g = TensorGraph::new();
        let a = g.add_node("a", OpKind::Input, &[]);
        let b = g.add_node("b", OpKind::Input, &[]);
        let c = g.add_node("c", OpKind::Add, &[a, b]);
        g.mark_output(c);

        let mut inputs = HashMap::new();
        inputs.insert(a, TensorVal::from_vec(vec![2], vec![1.0, 2.0]));
        inputs.insert(b, TensorVal::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
        assert!(g.execute(&inputs).is_err());
    }

    #[test]
    fn test_default_trait() {
        let g = TensorGraph::default();
        assert_eq!(g.num_nodes(), 0);
    }
}
