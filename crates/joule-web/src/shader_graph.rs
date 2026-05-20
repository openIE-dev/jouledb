//! Node-based shader graph system.
//!
//! Node types: Input (UV, position, normal, time), Math (add, multiply, lerp,
//! dot, cross, pow, saturate), Texture (sample), Output (color, normal,
//! metallic, roughness, emission). Typed connections (float, vec2, vec3, vec4,
//! color). Graph validation, cycle detection, topological sort, evaluation.
//! Pure Rust — no external dependencies.

use std::collections::HashMap;
use std::fmt;

// ── Value types ─────────────────────────────────────────────────

/// Data types that can flow through graph edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Color,
}

impl DataType {
    /// Whether `src` can be implicitly cast to `dst`.
    pub fn compatible(src: DataType, dst: DataType) -> bool {
        if src == dst {
            return true;
        }
        // Color <-> Vec4, Vec3 -> Color (assume alpha 1).
        matches!(
            (src, dst),
            (DataType::Color, DataType::Vec4)
                | (DataType::Vec4, DataType::Color)
                | (DataType::Vec3, DataType::Color)
                | (DataType::Float, DataType::Vec2)
                | (DataType::Float, DataType::Vec3)
                | (DataType::Float, DataType::Vec4)
                | (DataType::Float, DataType::Color)
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Float => "float",
            DataType::Vec2 => "vec2",
            DataType::Vec3 => "vec3",
            DataType::Vec4 => "vec4",
            DataType::Color => "color",
        };
        write!(f, "{s}")
    }
}

// ── Runtime value ───────────────────────────────────────────────

/// Concrete value carried on an edge at evaluation time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Color([f32; 4]),
}

impl Value {
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Float(_) => DataType::Float,
            Value::Vec2(_) => DataType::Vec2,
            Value::Vec3(_) => DataType::Vec3,
            Value::Vec4(_) => DataType::Vec4,
            Value::Color(_) => DataType::Color,
        }
    }

    /// Coerce to float (take first component).
    pub fn as_float(&self) -> f32 {
        match self {
            Value::Float(v) => *v,
            Value::Vec2(v) => v[0],
            Value::Vec3(v) => v[0],
            Value::Vec4(v) => v[0],
            Value::Color(v) => v[0],
        }
    }

    /// Coerce to vec3.
    pub fn as_vec3(&self) -> [f32; 3] {
        match self {
            Value::Float(v) => [*v, *v, *v],
            Value::Vec2(v) => [v[0], v[1], 0.0],
            Value::Vec3(v) => *v,
            Value::Vec4(v) => [v[0], v[1], v[2]],
            Value::Color(v) => [v[0], v[1], v[2]],
        }
    }

    /// Coerce to vec4 / color.
    pub fn as_vec4(&self) -> [f32; 4] {
        match self {
            Value::Float(v) => [*v, *v, *v, *v],
            Value::Vec2(v) => [v[0], v[1], 0.0, 0.0],
            Value::Vec3(v) => [v[0], v[1], v[2], 1.0],
            Value::Vec4(v) => *v,
            Value::Color(v) => *v,
        }
    }
}

// ── Node kinds ──────────────────────────────────────────────────

/// Built-in input semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputSemantic {
    UV,
    Position,
    Normal,
    Time,
}

/// Math operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathOp {
    Add,
    Multiply,
    Lerp,
    Dot,
    Cross,
    Pow,
    Saturate,
}

/// Which output slot the output node represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputSlot {
    Color,
    Normal,
    Metallic,
    Roughness,
    Emission,
}

/// The kind of node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input(InputSemantic),
    Math(MathOp),
    Texture { width: u32, height: u32, data: Vec<[u8; 4]> },
    Output(OutputSlot),
    Constant(Value),
}

// ── Node ────────────────────────────────────────────────────────

/// Unique node identifier.
pub type NodeId = u32;

/// A node in the shader graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,
    pub label: String,
}

// ── Connection ──────────────────────────────────────────────────

/// Directed edge from one node's output to another node's input slot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Connection {
    pub from_node: NodeId,
    pub from_slot: u8,
    pub to_node: NodeId,
    pub to_slot: u8,
}

// ── Shader graph ────────────────────────────────────────────────

/// The complete shader graph.
#[derive(Debug, Clone)]
pub struct ShaderGraph {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
    next_id: NodeId,
}

impl ShaderGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), connections: Vec::new(), next_id: 0 }
    }

    /// Add a node and return its id.
    pub fn add_node(&mut self, kind: NodeKind, label: impl Into<String>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(Node { id, kind, label: label.into() });
        id
    }

    /// Connect `from_node:from_slot` → `to_node:to_slot`.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_slot: u8,
        to_node: NodeId,
        to_slot: u8,
    ) -> Result<(), String> {
        if self.find_node(from_node).is_none() {
            return Err(format!("source node {from_node} not found"));
        }
        if self.find_node(to_node).is_none() {
            return Err(format!("target node {to_node} not found"));
        }
        if from_node == to_node {
            return Err("self-loop not allowed".into());
        }
        self.connections.push(Connection { from_node, from_slot, to_node, to_slot });
        Ok(())
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    fn find_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    // ── Validation ──────────────────────────────────────────────

    /// Check for cycles using DFS.
    pub fn has_cycle(&self) -> bool {
        // Build adjacency list.
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for n in &self.nodes {
            adj.insert(n.id, Vec::new());
        }
        for c in &self.connections {
            adj.entry(c.from_node).or_default().push(c.to_node);
        }

        let mut visited: HashMap<NodeId, u8> = HashMap::new(); // 0=white, 1=grey, 2=black
        for n in &self.nodes {
            visited.insert(n.id, 0);
        }

        fn dfs(node: NodeId, adj: &HashMap<NodeId, Vec<NodeId>>, vis: &mut HashMap<NodeId, u8>) -> bool {
            vis.insert(node, 1);
            if let Some(neighbours) = adj.get(&node) {
                for &nb in neighbours {
                    match vis.get(&nb).copied().unwrap_or(0) {
                        1 => return true,
                        0 => {
                            if dfs(nb, adj, vis) {
                                return true;
                            }
                        }
                        _ => {}
                    }
                }
            }
            vis.insert(node, 2);
            false
        }

        for n in &self.nodes {
            if visited.get(&n.id).copied().unwrap_or(0) == 0 {
                if dfs(n.id, &adj, &mut visited) {
                    return true;
                }
            }
        }
        false
    }

    /// Check that all output nodes have their required inputs connected.
    pub fn outputs_connected(&self) -> Vec<(NodeId, OutputSlot)> {
        let mut missing = Vec::new();
        for n in &self.nodes {
            if let NodeKind::Output(slot) = &n.kind {
                let has_input = self.connections.iter().any(|c| c.to_node == n.id && c.to_slot == 0);
                if !has_input {
                    missing.push((n.id, *slot));
                }
            }
        }
        missing
    }

    /// Full validation: no cycles + all outputs connected.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.has_cycle() {
            errors.push("graph contains a cycle".into());
        }
        for (id, slot) in self.outputs_connected() {
            errors.push(format!("output node {id} ({slot:?}) has no input"));
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    // ── Topological sort ────────────────────────────────────────

    /// Kahn's algorithm — returns nodes in evaluation order or Err if cyclic.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, String> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for n in &self.nodes {
            in_degree.insert(n.id, 0);
        }
        for c in &self.connections {
            *in_degree.entry(c.to_node).or_insert(0) += 1;
        }

        let mut queue: Vec<NodeId> = Vec::new();
        // Collect nodes with in-degree 0 and sort for determinism.
        let mut zero_deg: Vec<NodeId> = in_degree.iter()
            .filter(|&(_, d)| *d == 0)
            .map(|(&id, _)| id)
            .collect();
        zero_deg.sort();
        queue.extend(zero_deg);

        let mut order = Vec::new();
        while let Some(node) = queue.pop() {
            order.push(node);
            // Collect neighbours and sort for deterministic output.
            let mut neighbours: Vec<NodeId> = self.connections
                .iter()
                .filter(|c| c.from_node == node)
                .map(|c| c.to_node)
                .collect();
            neighbours.sort();
            neighbours.dedup();
            for nb in neighbours {
                if let Some(deg) = in_degree.get_mut(&nb) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push(nb);
                    }
                }
            }
        }
        if order.len() == self.nodes.len() {
            Ok(order)
        } else {
            Err("cycle detected during topological sort".into())
        }
    }

    // ── Evaluation ──────────────────────────────────────────────

    /// Evaluate the graph with the given inputs and return output values.
    pub fn evaluate(&self, inputs: &HashMap<InputSemantic, Value>) -> Result<HashMap<OutputSlot, Value>, String> {
        let order = self.topological_sort()?;
        let mut values: HashMap<NodeId, Value> = HashMap::new();

        for &nid in &order {
            let node = self.find_node(nid).ok_or_else(|| format!("node {nid} missing"))?;
            let val = match &node.kind {
                NodeKind::Constant(v) => *v,
                NodeKind::Input(sem) => {
                    inputs.get(sem).copied().unwrap_or(Value::Float(0.0))
                }
                NodeKind::Math(op) => {
                    self.eval_math(*op, nid, &values)?
                }
                NodeKind::Texture { width, height, data } => {
                    self.eval_texture(*width, *height, data, nid, &values)?
                }
                NodeKind::Output(_) => {
                    // Pass through slot 0.
                    self.get_input_value(nid, 0, &values)?
                }
            };
            values.insert(nid, val);
        }

        let mut outputs = HashMap::new();
        for n in &self.nodes {
            if let NodeKind::Output(slot) = &n.kind {
                if let Some(v) = values.get(&n.id) {
                    outputs.insert(*slot, *v);
                }
            }
        }
        Ok(outputs)
    }

    fn get_input_value(&self, to_node: NodeId, to_slot: u8, values: &HashMap<NodeId, Value>) -> Result<Value, String> {
        for c in &self.connections {
            if c.to_node == to_node && c.to_slot == to_slot {
                return values.get(&c.from_node).copied()
                    .ok_or_else(|| format!("no value for node {}", c.from_node));
            }
        }
        Ok(Value::Float(0.0))
    }

    fn eval_math(&self, op: MathOp, nid: NodeId, values: &HashMap<NodeId, Value>) -> Result<Value, String> {
        let a = self.get_input_value(nid, 0, values)?;
        match op {
            MathOp::Add => {
                let b = self.get_input_value(nid, 1, values)?;
                let av = a.as_vec4();
                let bv = b.as_vec4();
                Ok(Value::Vec4([av[0]+bv[0], av[1]+bv[1], av[2]+bv[2], av[3]+bv[3]]))
            }
            MathOp::Multiply => {
                let b = self.get_input_value(nid, 1, values)?;
                let av = a.as_vec4();
                let bv = b.as_vec4();
                Ok(Value::Vec4([av[0]*bv[0], av[1]*bv[1], av[2]*bv[2], av[3]*bv[3]]))
            }
            MathOp::Lerp => {
                let b = self.get_input_value(nid, 1, values)?;
                let t = self.get_input_value(nid, 2, values)?.as_float();
                let av = a.as_vec4();
                let bv = b.as_vec4();
                Ok(Value::Vec4([
                    av[0] + (bv[0]-av[0]) * t,
                    av[1] + (bv[1]-av[1]) * t,
                    av[2] + (bv[2]-av[2]) * t,
                    av[3] + (bv[3]-av[3]) * t,
                ]))
            }
            MathOp::Dot => {
                let b = self.get_input_value(nid, 1, values)?;
                let av = a.as_vec3();
                let bv = b.as_vec3();
                Ok(Value::Float(av[0]*bv[0] + av[1]*bv[1] + av[2]*bv[2]))
            }
            MathOp::Cross => {
                let b = self.get_input_value(nid, 1, values)?;
                let av = a.as_vec3();
                let bv = b.as_vec3();
                Ok(Value::Vec3([
                    av[1]*bv[2] - av[2]*bv[1],
                    av[2]*bv[0] - av[0]*bv[2],
                    av[0]*bv[1] - av[1]*bv[0],
                ]))
            }
            MathOp::Pow => {
                let b = self.get_input_value(nid, 1, values)?;
                let base = a.as_float();
                let exp = b.as_float();
                Ok(Value::Float(base.powf(exp)))
            }
            MathOp::Saturate => {
                let v = a.as_vec4();
                Ok(Value::Vec4([
                    v[0].clamp(0.0, 1.0),
                    v[1].clamp(0.0, 1.0),
                    v[2].clamp(0.0, 1.0),
                    v[3].clamp(0.0, 1.0),
                ]))
            }
        }
    }

    fn eval_texture(&self, width: u32, height: u32, data: &[[u8; 4]], nid: NodeId, values: &HashMap<NodeId, Value>) -> Result<Value, String> {
        let uv_val = self.get_input_value(nid, 0, values)?;
        let uv = match uv_val {
            Value::Vec2(v) => v,
            Value::Float(f) => [f, f],
            _ => {
                let v4 = uv_val.as_vec4();
                [v4[0], v4[1]]
            }
        };
        let u = uv[0].fract().abs();
        let v = uv[1].fract().abs();
        let x = ((u * width as f32) as u32).min(width.saturating_sub(1));
        let y = ((v * height as f32) as u32).min(height.saturating_sub(1));
        let idx = (y * width + x) as usize;
        if idx < data.len() {
            let px = data[idx];
            Ok(Value::Color([
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
                px[3] as f32 / 255.0,
            ]))
        } else {
            Ok(Value::Color([0.0, 0.0, 0.0, 1.0]))
        }
    }
}

impl Default for ShaderGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_empty_graph_no_cycle() {
        let g = ShaderGraph::new();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_single_node() {
        let mut g = ShaderGraph::new();
        g.add_node(NodeKind::Constant(Value::Float(1.0)), "const");
        assert_eq!(g.node_count(), 1);
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_simple_connection() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(1.0)), "a");
        let b = g.add_node(NodeKind::Output(OutputSlot::Metallic), "out");
        g.connect(a, 0, b, 0).unwrap();
        assert_eq!(g.connection_count(), 1);
    }

    #[test]
    fn test_self_loop_rejected() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(1.0)), "a");
        assert!(g.connect(a, 0, a, 0).is_err());
    }

    #[test]
    fn test_invalid_node_connection() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(1.0)), "a");
        assert!(g.connect(a, 0, 999, 0).is_err());
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Math(MathOp::Add), "a");
        let b = g.add_node(NodeKind::Math(MathOp::Add), "b");
        let c = g.add_node(NodeKind::Math(MathOp::Add), "c");
        g.connect(a, 0, b, 0).unwrap();
        g.connect(b, 0, c, 0).unwrap();
        g.connect(c, 0, a, 0).unwrap();
        assert!(g.has_cycle());
    }

    #[test]
    fn test_no_cycle_dag() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(1.0)), "a");
        let b = g.add_node(NodeKind::Constant(Value::Float(2.0)), "b");
        let c = g.add_node(NodeKind::Math(MathOp::Add), "add");
        let d = g.add_node(NodeKind::Output(OutputSlot::Color), "out");
        g.connect(a, 0, c, 0).unwrap();
        g.connect(b, 0, c, 1).unwrap();
        g.connect(c, 0, d, 0).unwrap();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_topological_sort_simple() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(1.0)), "a");
        let b = g.add_node(NodeKind::Output(OutputSlot::Metallic), "out");
        g.connect(a, 0, b, 0).unwrap();
        let order = g.topological_sort().unwrap();
        let pos_a = order.iter().position(|x| *x == a).unwrap();
        let pos_b = order.iter().position(|x| *x == b).unwrap();
        assert!(pos_a < pos_b);
    }

    #[test]
    fn test_topological_sort_cycle_error() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Math(MathOp::Add), "a");
        let b = g.add_node(NodeKind::Math(MathOp::Add), "b");
        g.connect(a, 0, b, 0).unwrap();
        g.connect(b, 0, a, 0).unwrap();
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn test_outputs_connected() {
        let mut g = ShaderGraph::new();
        g.add_node(NodeKind::Output(OutputSlot::Color), "out");
        let missing = g.outputs_connected();
        assert_eq!(missing.len(), 1);
    }

    #[test]
    fn test_validate_success() {
        let mut g = ShaderGraph::new();
        let c = g.add_node(NodeKind::Constant(Value::Float(1.0)), "c");
        let o = g.add_node(NodeKind::Output(OutputSlot::Metallic), "out");
        g.connect(c, 0, o, 0).unwrap();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_validate_fail_cycle() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Math(MathOp::Add), "a");
        let b = g.add_node(NodeKind::Math(MathOp::Add), "b");
        g.connect(a, 0, b, 0).unwrap();
        g.connect(b, 0, a, 0).unwrap();
        assert!(g.validate().is_err());
    }

    #[test]
    fn test_eval_constant_to_output() {
        let mut g = ShaderGraph::new();
        let c = g.add_node(NodeKind::Constant(Value::Float(0.75)), "c");
        let o = g.add_node(NodeKind::Output(OutputSlot::Roughness), "out");
        g.connect(c, 0, o, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Roughness).unwrap();
        assert!(approx(v.as_float(), 0.75));
    }

    #[test]
    fn test_eval_add() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(1.0)), "a");
        let b = g.add_node(NodeKind::Constant(Value::Float(2.0)), "b");
        let add = g.add_node(NodeKind::Math(MathOp::Add), "add");
        let out = g.add_node(NodeKind::Output(OutputSlot::Metallic), "out");
        g.connect(a, 0, add, 0).unwrap();
        g.connect(b, 0, add, 1).unwrap();
        g.connect(add, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Metallic).unwrap();
        assert!(approx(v.as_float(), 3.0));
    }

    #[test]
    fn test_eval_multiply() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Vec3([2.0, 3.0, 4.0])), "a");
        let b = g.add_node(NodeKind::Constant(Value::Vec3([0.5, 0.5, 0.5])), "b");
        let mul = g.add_node(NodeKind::Math(MathOp::Multiply), "mul");
        let out = g.add_node(NodeKind::Output(OutputSlot::Color), "out");
        g.connect(a, 0, mul, 0).unwrap();
        g.connect(b, 0, mul, 1).unwrap();
        g.connect(mul, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Color).unwrap().as_vec3();
        assert!(approx(v[0], 1.0));
        assert!(approx(v[1], 1.5));
        assert!(approx(v[2], 2.0));
    }

    #[test]
    fn test_eval_dot() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Vec3([1.0, 0.0, 0.0])), "a");
        let b = g.add_node(NodeKind::Constant(Value::Vec3([0.0, 1.0, 0.0])), "b");
        let dot = g.add_node(NodeKind::Math(MathOp::Dot), "dot");
        let out = g.add_node(NodeKind::Output(OutputSlot::Metallic), "out");
        g.connect(a, 0, dot, 0).unwrap();
        g.connect(b, 0, dot, 1).unwrap();
        g.connect(dot, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Metallic).unwrap().as_float();
        assert!(approx(v, 0.0));
    }

    #[test]
    fn test_eval_cross() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Vec3([1.0, 0.0, 0.0])), "x");
        let b = g.add_node(NodeKind::Constant(Value::Vec3([0.0, 1.0, 0.0])), "y");
        let cross = g.add_node(NodeKind::Math(MathOp::Cross), "cross");
        let out = g.add_node(NodeKind::Output(OutputSlot::Normal), "out");
        g.connect(a, 0, cross, 0).unwrap();
        g.connect(b, 0, cross, 1).unwrap();
        g.connect(cross, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Normal).unwrap().as_vec3();
        assert!(approx(v[0], 0.0));
        assert!(approx(v[1], 0.0));
        assert!(approx(v[2], 1.0));
    }

    #[test]
    fn test_eval_lerp() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Float(0.0)), "a");
        let b = g.add_node(NodeKind::Constant(Value::Float(10.0)), "b");
        let t = g.add_node(NodeKind::Constant(Value::Float(0.3)), "t");
        let lerp = g.add_node(NodeKind::Math(MathOp::Lerp), "lerp");
        let out = g.add_node(NodeKind::Output(OutputSlot::Roughness), "out");
        g.connect(a, 0, lerp, 0).unwrap();
        g.connect(b, 0, lerp, 1).unwrap();
        g.connect(t, 0, lerp, 2).unwrap();
        g.connect(lerp, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Roughness).unwrap().as_float();
        assert!(approx(v, 3.0));
    }

    #[test]
    fn test_eval_saturate() {
        let mut g = ShaderGraph::new();
        let a = g.add_node(NodeKind::Constant(Value::Vec4([-0.5, 1.5, 0.5, 2.0])), "a");
        let sat = g.add_node(NodeKind::Math(MathOp::Saturate), "sat");
        let out = g.add_node(NodeKind::Output(OutputSlot::Color), "out");
        g.connect(a, 0, sat, 0).unwrap();
        g.connect(sat, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Color).unwrap().as_vec4();
        assert!(approx(v[0], 0.0));
        assert!(approx(v[1], 1.0));
        assert!(approx(v[2], 0.5));
        assert!(approx(v[3], 1.0));
    }

    #[test]
    fn test_eval_pow() {
        let mut g = ShaderGraph::new();
        let base = g.add_node(NodeKind::Constant(Value::Float(2.0)), "base");
        let exp = g.add_node(NodeKind::Constant(Value::Float(3.0)), "exp");
        let pw = g.add_node(NodeKind::Math(MathOp::Pow), "pow");
        let out = g.add_node(NodeKind::Output(OutputSlot::Metallic), "out");
        g.connect(base, 0, pw, 0).unwrap();
        g.connect(exp, 0, pw, 1).unwrap();
        g.connect(pw, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let v = result.get(&OutputSlot::Metallic).unwrap().as_float();
        assert!(approx(v, 8.0));
    }

    #[test]
    fn test_eval_input_node() {
        let mut g = ShaderGraph::new();
        let time = g.add_node(NodeKind::Input(InputSemantic::Time), "time");
        let out = g.add_node(NodeKind::Output(OutputSlot::Roughness), "out");
        g.connect(time, 0, out, 0).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert(InputSemantic::Time, Value::Float(0.42));
        let result = g.evaluate(&inputs).unwrap();
        let v = result.get(&OutputSlot::Roughness).unwrap().as_float();
        assert!(approx(v, 0.42));
    }

    #[test]
    fn test_eval_texture_sample() {
        let mut g = ShaderGraph::new();
        // 2x2 texture, top-left is red.
        let data = vec![
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
        ];
        let uv = g.add_node(NodeKind::Constant(Value::Vec2([0.0, 0.0])), "uv");
        let tex = g.add_node(NodeKind::Texture { width: 2, height: 2, data }, "tex");
        let out = g.add_node(NodeKind::Output(OutputSlot::Color), "out");
        g.connect(uv, 0, tex, 0).unwrap();
        g.connect(tex, 0, out, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        let c = result.get(&OutputSlot::Color).unwrap().as_vec4();
        assert!(approx(c[0], 1.0)); // red channel
        assert!(approx(c[1], 0.0));
    }

    #[test]
    fn test_data_type_compatibility() {
        assert!(DataType::compatible(DataType::Float, DataType::Vec3));
        assert!(DataType::compatible(DataType::Color, DataType::Vec4));
        assert!(DataType::compatible(DataType::Vec3, DataType::Color));
        assert!(!DataType::compatible(DataType::Vec2, DataType::Vec3));
    }

    #[test]
    fn test_value_coerce() {
        let f = Value::Float(3.0);
        assert!(approx(f.as_float(), 3.0));
        let v3 = f.as_vec3();
        assert!(approx(v3[0], 3.0));
        assert!(approx(v3[2], 3.0));
    }

    #[test]
    fn test_multiple_outputs() {
        let mut g = ShaderGraph::new();
        let c1 = g.add_node(NodeKind::Constant(Value::Float(0.5)), "c1");
        let c2 = g.add_node(NodeKind::Constant(Value::Float(0.8)), "c2");
        let o1 = g.add_node(NodeKind::Output(OutputSlot::Metallic), "o1");
        let o2 = g.add_node(NodeKind::Output(OutputSlot::Roughness), "o2");
        g.connect(c1, 0, o1, 0).unwrap();
        g.connect(c2, 0, o2, 0).unwrap();
        let result = g.evaluate(&HashMap::new()).unwrap();
        assert!(approx(result[&OutputSlot::Metallic].as_float(), 0.5));
        assert!(approx(result[&OutputSlot::Roughness].as_float(), 0.8));
    }
}
