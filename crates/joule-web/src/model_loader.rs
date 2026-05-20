//! ML model metadata, graph representation, and serialization.
//!
//! Provides an ONNX-inspired op set, computational graph, and sequential
//! model builder. Models serialize to/from JSON for storage and transfer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Op enum ─────────────────────────────────────────────────────

/// ONNX-like operator set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Conv {
        kernel_size: Vec<usize>,
        strides: Vec<usize>,
        pads: Vec<usize>,
    },
    MatMul,
    Relu,
    Sigmoid,
    Add,
    Reshape {
        target_shape: Vec<i64>,
    },
    Softmax {
        axis: i64,
    },
    MaxPool {
        kernel_size: Vec<usize>,
        strides: Vec<usize>,
    },
    BatchNorm {
        epsilon: f64,
        momentum: f64,
    },
    Concat {
        axis: i64,
    },
    Flatten {
        axis: i64,
    },
}

impl Op {
    /// Human-readable name for the op.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Conv { .. } => "Conv",
            Op::MatMul => "MatMul",
            Op::Relu => "Relu",
            Op::Sigmoid => "Sigmoid",
            Op::Add => "Add",
            Op::Reshape { .. } => "Reshape",
            Op::Softmax { .. } => "Softmax",
            Op::MaxPool { .. } => "MaxPool",
            Op::BatchNorm { .. } => "BatchNorm",
            Op::Concat { .. } => "Concat",
            Op::Flatten { .. } => "Flatten",
        }
    }
}

// ── Graph node / edge ───────────────────────────────────────────

/// A node in the computational graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier.
    pub id: String,
    /// The operation this node performs.
    pub op: Op,
    /// IDs of input edges (tensor names).
    pub inputs: Vec<String>,
    /// IDs of output edges (tensor names).
    pub outputs: Vec<String>,
    /// Optional attributes (key-value).
    pub attributes: HashMap<String, String>,
}

/// Describes a tensor flowing between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEdge {
    /// Edge name.
    pub name: String,
    /// Shape (may contain -1 for dynamic dims).
    pub shape: Vec<i64>,
    /// Element data type description.
    pub dtype: String,
}

// ── Model metadata ──────────────────────────────────────────────

/// Complete model metadata and graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    /// Model name.
    pub name: String,
    /// Semantic version.
    pub version: String,
    /// Opset version.
    pub opset: u32,
    /// Input tensor descriptors.
    pub inputs: Vec<TensorEdge>,
    /// Output tensor descriptors.
    pub outputs: Vec<TensorEdge>,
    /// Computational graph (ordered nodes).
    pub nodes: Vec<GraphNode>,
    /// All intermediate tensor edges.
    pub edges: Vec<TensorEdge>,
}

impl ModelMeta {
    /// Create a minimal model with no nodes.
    pub fn new(name: impl Into<String>, version: impl Into<String>, opset: u32) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            opset,
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return topological ordering of node IDs (currently insertion order).
    pub fn topological_order(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.id.clone()).collect()
    }

    /// Find a node by ID.
    pub fn find_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }
}

// ── Sequential model builder ────────────────────────────────────

/// Builder for sequential (linear chain) models.
pub struct SequentialBuilder {
    model: ModelMeta,
    next_edge_id: usize,
    last_output: String,
}

impl SequentialBuilder {
    /// Start a new sequential model.
    pub fn new(name: impl Into<String>, version: impl Into<String>, opset: u32) -> Self {
        let input_edge = TensorEdge {
            name: "input_0".to_string(),
            shape: vec![-1],
            dtype: "float32".to_string(),
        };
        let mut model = ModelMeta::new(name, version, opset);
        model.inputs.push(input_edge.clone());
        model.edges.push(input_edge);
        Self {
            model,
            next_edge_id: 1,
            last_output: "input_0".to_string(),
        }
    }

    /// Set the input shape.
    pub fn input_shape(mut self, shape: Vec<i64>) -> Self {
        if let Some(inp) = self.model.inputs.first_mut() {
            inp.shape = shape.clone();
        }
        if let Some(edge) = self.model.edges.first_mut() {
            edge.shape = shape;
        }
        self
    }

    /// Add an op to the sequential chain.
    pub fn add_op(mut self, op: Op) -> Self {
        let node_id = format!("node_{}", self.model.nodes.len());
        let output_name = format!("edge_{}", self.next_edge_id);
        self.next_edge_id += 1;

        let node = GraphNode {
            id: node_id,
            op,
            inputs: vec![self.last_output.clone()],
            outputs: vec![output_name.clone()],
            attributes: HashMap::new(),
        };
        self.model.nodes.push(node);

        let edge = TensorEdge {
            name: output_name.clone(),
            shape: vec![-1],
            dtype: "float32".to_string(),
        };
        self.model.edges.push(edge);
        self.last_output = output_name;
        self
    }

    /// Finalize and return the model.
    pub fn build(mut self) -> ModelMeta {
        let output_edge = TensorEdge {
            name: self.last_output.clone(),
            shape: vec![-1],
            dtype: "float32".to_string(),
        };
        self.model.outputs.push(output_edge);
        self.model
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_name() {
        assert_eq!(Op::Relu.name(), "Relu");
        assert_eq!(Op::MatMul.name(), "MatMul");
        assert_eq!(
            Op::Conv {
                kernel_size: vec![3, 3],
                strides: vec![1, 1],
                pads: vec![1, 1, 1, 1],
            }
            .name(),
            "Conv"
        );
    }

    #[test]
    fn test_model_meta_new() {
        let m = ModelMeta::new("test_model", "1.0.0", 13);
        assert_eq!(m.name, "test_model");
        assert_eq!(m.opset, 13);
        assert_eq!(m.node_count(), 0);
    }

    #[test]
    fn test_sequential_builder() {
        let model = SequentialBuilder::new("classifier", "1.0.0", 13)
            .input_shape(vec![1, 3, 224, 224])
            .add_op(Op::Conv {
                kernel_size: vec![3, 3],
                strides: vec![1, 1],
                pads: vec![1, 1, 1, 1],
            })
            .add_op(Op::Relu)
            .add_op(Op::MaxPool {
                kernel_size: vec![2, 2],
                strides: vec![2, 2],
            })
            .add_op(Op::Flatten { axis: 1 })
            .add_op(Op::MatMul)
            .add_op(Op::Softmax { axis: -1 })
            .build();

        assert_eq!(model.node_count(), 6);
        assert_eq!(model.inputs.len(), 1);
        assert_eq!(model.outputs.len(), 1);
        assert_eq!(model.inputs[0].shape, vec![1, 3, 224, 224]);
    }

    #[test]
    fn test_json_round_trip() {
        let model = SequentialBuilder::new("net", "0.1.0", 11)
            .add_op(Op::Relu)
            .add_op(Op::Sigmoid)
            .build();

        let json = model.to_json().unwrap();
        let loaded = ModelMeta::from_json(&json).unwrap();
        assert_eq!(loaded.name, "net");
        assert_eq!(loaded.node_count(), 2);
        assert_eq!(loaded.nodes[0].op, Op::Relu);
        assert_eq!(loaded.nodes[1].op, Op::Sigmoid);
    }

    #[test]
    fn test_find_node() {
        let model = SequentialBuilder::new("m", "1.0.0", 13)
            .add_op(Op::Relu)
            .add_op(Op::Add)
            .build();

        assert!(model.find_node("node_0").is_some());
        assert!(model.find_node("node_1").is_some());
        assert!(model.find_node("node_99").is_none());
    }

    #[test]
    fn test_topological_order() {
        let model = SequentialBuilder::new("m", "1.0.0", 13)
            .add_op(Op::Conv {
                kernel_size: vec![1],
                strides: vec![1],
                pads: vec![0, 0],
            })
            .add_op(Op::Relu)
            .add_op(Op::MatMul)
            .build();

        let order = model.topological_order();
        assert_eq!(order, vec!["node_0", "node_1", "node_2"]);
    }

    #[test]
    fn test_graph_node_connections() {
        let model = SequentialBuilder::new("m", "1.0.0", 13)
            .add_op(Op::Relu)
            .add_op(Op::Sigmoid)
            .build();

        // node_0 takes input_0, outputs edge_1
        let n0 = model.find_node("node_0").unwrap();
        assert_eq!(n0.inputs, vec!["input_0"]);
        assert_eq!(n0.outputs, vec!["edge_1"]);

        // node_1 takes edge_1, outputs edge_2
        let n1 = model.find_node("node_1").unwrap();
        assert_eq!(n1.inputs, vec!["edge_1"]);
        assert_eq!(n1.outputs, vec!["edge_2"]);
    }

    #[test]
    fn test_batch_norm_op() {
        let bn = Op::BatchNorm {
            epsilon: 1e-5,
            momentum: 0.1,
        };
        assert_eq!(bn.name(), "BatchNorm");

        let model = SequentialBuilder::new("bn_test", "1.0.0", 13)
            .add_op(bn.clone())
            .build();
        let json = model.to_json().unwrap();
        let loaded = ModelMeta::from_json(&json).unwrap();
        assert_eq!(loaded.nodes[0].op, bn);
    }

    #[test]
    fn test_model_with_attributes() {
        let mut model = ModelMeta::new("custom", "2.0.0", 15);
        let mut attrs = HashMap::new();
        attrs.insert("group".to_string(), "1".to_string());
        model.nodes.push(GraphNode {
            id: "conv_0".to_string(),
            op: Op::Conv {
                kernel_size: vec![3, 3],
                strides: vec![1, 1],
                pads: vec![1, 1, 1, 1],
            },
            inputs: vec!["input".to_string()],
            outputs: vec!["conv_out".to_string()],
            attributes: attrs,
        });

        let n = model.find_node("conv_0").unwrap();
        assert_eq!(n.attributes.get("group"), Some(&"1".to_string()));
    }

    #[test]
    fn test_concat_and_flatten_ops() {
        let model = SequentialBuilder::new("m", "1.0.0", 13)
            .add_op(Op::Concat { axis: 1 })
            .add_op(Op::Flatten { axis: 1 })
            .build();
        assert_eq!(model.nodes[0].op.name(), "Concat");
        assert_eq!(model.nodes[1].op.name(), "Flatten");
    }

    #[test]
    fn test_edges_tracked() {
        let model = SequentialBuilder::new("m", "1.0.0", 13)
            .add_op(Op::Relu)
            .add_op(Op::Relu)
            .add_op(Op::Relu)
            .build();
        // input_0 + 3 intermediate edges
        assert_eq!(model.edges.len(), 4);
    }
}
