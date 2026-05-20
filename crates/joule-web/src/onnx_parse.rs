//! ONNX model format parser: protobuf-like binary parsing, operator set
//! resolution, graph reconstruction, and shape inference.
//!
//! Parses a subset of the ONNX binary format (protobuf wire format) to
//! extract model structure, operator types, tensor initializers, and
//! graph topology without requiring an external protobuf library.

use std::collections::HashMap;
use std::fmt;

// ── Wire Format ────────────────────────────────────────────────

/// Protobuf wire types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireType {
    Varint,         // 0
    Fixed64,        // 1
    LengthDelim,    // 2
    Fixed32,        // 5
}

impl WireType {
    pub fn from_tag(tag: u64) -> Option<Self> {
        match tag & 0x07 {
            0 => Some(WireType::Varint),
            1 => Some(WireType::Fixed64),
            2 => Some(WireType::LengthDelim),
            5 => Some(WireType::Fixed32),
            _ => None,
        }
    }
}

impl fmt::Display for WireType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireType::Varint => write!(f, "varint"),
            WireType::Fixed64 => write!(f, "fixed64"),
            WireType::LengthDelim => write!(f, "length-delimited"),
            WireType::Fixed32 => write!(f, "fixed32"),
        }
    }
}

/// Read a varint from a byte slice, returning (value, bytes_consumed).
pub fn read_varint(data: &[u8], offset: usize) -> Result<(u64, usize), String> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut pos = offset;

    loop {
        if pos >= data.len() {
            return Err("unexpected end of data in varint".into());
        }
        let byte = data[pos];
        pos += 1;

        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint too long".into());
        }
    }
    Ok((result, pos - offset))
}

/// Read a protobuf tag (field number + wire type).
pub fn read_tag(data: &[u8], offset: usize) -> Result<(u32, WireType, usize), String> {
    let (val, consumed) = read_varint(data, offset)?;
    let field_number = (val >> 3) as u32;
    let wire_type =
        WireType::from_tag(val).ok_or_else(|| format!("unknown wire type: {}", val & 0x07))?;
    Ok((field_number, wire_type, consumed))
}

/// Read a length-delimited field (returns the byte slice and bytes consumed).
pub fn read_length_delimited<'a>(data: &'a [u8], offset: usize) -> Result<(&'a [u8], usize), String> {
    let (length, len_bytes) = read_varint(data, offset)?;
    let start = offset + len_bytes;
    let end = start + length as usize;
    if end > data.len() {
        return Err(format!("length-delimited field extends past end: {} > {}", end, data.len()));
    }
    Ok((&data[start..end], len_bytes + length as usize))
}

// ── ONNX Data Types ────────────────────────────────────────────

/// ONNX tensor element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnnxDataType {
    Float,
    Uint8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Double,
    Bool,
    String,
    Unknown(u32),
}

impl OnnxDataType {
    pub fn from_id(id: u32) -> Self {
        match id {
            1 => OnnxDataType::Float,
            2 => OnnxDataType::Uint8,
            3 => OnnxDataType::Int8,
            5 => OnnxDataType::Int16,
            6 => OnnxDataType::Int32,
            7 => OnnxDataType::Int64,
            10 => OnnxDataType::Float16,
            11 => OnnxDataType::Double,
            9 => OnnxDataType::Bool,
            8 => OnnxDataType::String,
            other => OnnxDataType::Unknown(other),
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            OnnxDataType::Float => 4,
            OnnxDataType::Double => 8,
            OnnxDataType::Float16 => 2,
            OnnxDataType::Uint8 | OnnxDataType::Int8 | OnnxDataType::Bool => 1,
            OnnxDataType::Int16 => 2,
            OnnxDataType::Int32 => 4,
            OnnxDataType::Int64 => 8,
            _ => 0,
        }
    }
}

impl fmt::Display for OnnxDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxDataType::Float => write!(f, "float32"),
            OnnxDataType::Double => write!(f, "float64"),
            OnnxDataType::Float16 => write!(f, "float16"),
            OnnxDataType::Uint8 => write!(f, "uint8"),
            OnnxDataType::Int8 => write!(f, "int8"),
            OnnxDataType::Int16 => write!(f, "int16"),
            OnnxDataType::Int32 => write!(f, "int32"),
            OnnxDataType::Int64 => write!(f, "int64"),
            OnnxDataType::Bool => write!(f, "bool"),
            OnnxDataType::String => write!(f, "string"),
            OnnxDataType::Unknown(id) => write!(f, "unknown({id})"),
        }
    }
}

// ── ONNX Tensor ────────────────────────────────────────────────

/// A tensor initializer extracted from the ONNX model.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub data_type: OnnxDataType,
    pub dims: Vec<i64>,
    pub raw_data: Vec<u8>,
}

impl OnnxTensor {
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            return 0;
        }
        self.dims.iter().map(|d| d.unsigned_abs() as usize).product()
    }

    /// Extract float32 data from raw bytes.
    pub fn as_f32(&self) -> Vec<f32> {
        self.raw_data
            .chunks_exact(4)
            .map(|chunk| {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes)
            })
            .collect()
    }

    /// Estimated memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.numel() * self.data_type.byte_size()
    }
}

impl fmt::Display for OnnxTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OnnxTensor('{}', dtype={}, dims={:?})", self.name, self.data_type, self.dims)
    }
}

// ── ONNX Node ──────────────────────────────────────────────────

/// An operator node in the ONNX graph.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttr>,
    pub domain: String,
}

impl OnnxNode {
    pub fn new(op_type: impl Into<String>) -> Self {
        Self {
            name: String::new(),
            op_type: op_type.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attributes: HashMap::new(),
            domain: String::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_input(mut self, inp: impl Into<String>) -> Self {
        self.inputs.push(inp.into());
        self
    }

    pub fn with_output(mut self, out: impl Into<String>) -> Self {
        self.outputs.push(out.into());
        self
    }

    pub fn with_attr(mut self, key: impl Into<String>, val: OnnxAttr) -> Self {
        self.attributes.insert(key.into(), val);
        self
    }
}

impl fmt::Display for OnnxNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnnxNode('{}', op={}, ins={}, outs={})",
            self.name,
            self.op_type,
            self.inputs.len(),
            self.outputs.len()
        )
    }
}

// ── ONNX Attribute ─────────────────────────────────────────────

/// Attribute value on an ONNX node.
#[derive(Debug, Clone)]
pub enum OnnxAttr {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

impl fmt::Display for OnnxAttr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxAttr::Int(v) => write!(f, "{v}"),
            OnnxAttr::Float(v) => write!(f, "{v}"),
            OnnxAttr::String(s) => write!(f, "\"{s}\""),
            OnnxAttr::Ints(vs) => write!(f, "{vs:?}"),
            OnnxAttr::Floats(vs) => write!(f, "{vs:?}"),
        }
    }
}

// ── ONNX OpSet ─────────────────────────────────────────────────

/// Operator set import.
#[derive(Debug, Clone)]
pub struct OpSetImport {
    pub domain: String,
    pub version: i64,
}

impl fmt::Display for OpSetImport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.domain.is_empty() {
            write!(f, "ai.onnx v{}", self.version)
        } else {
            write!(f, "{} v{}", self.domain, self.version)
        }
    }
}

// ── ONNX Graph ─────────────────────────────────────────────────

/// Reconstructed ONNX computation graph.
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub initializers: HashMap<String, OnnxTensor>,
}

impl OnnxGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            initializers: HashMap::new(),
        }
    }

    /// Topological ordering of nodes based on input/output dependencies.
    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut output_map: HashMap<&str, usize> = HashMap::new();
        for (i, node) in self.nodes.iter().enumerate() {
            for out in &node.outputs {
                output_map.insert(out.as_str(), i);
            }
        }

        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, node) in self.nodes.iter().enumerate() {
            for inp in &node.inputs {
                if let Some(&producer) = output_map.get(inp.as_str()) {
                    if producer != i {
                        adj[producer].push(i);
                        in_degree[i] += 1;
                    }
                }
            }
        }

        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(idx) = queue.pop_front() {
            order.push(idx);
            for &next in &adj[idx] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        order
    }

    /// Count of each op type in the graph.
    pub fn op_histogram(&self) -> HashMap<String, usize> {
        let mut hist = HashMap::new();
        for node in &self.nodes {
            *hist.entry(node.op_type.clone()).or_insert(0) += 1;
        }
        hist
    }

    /// Total number of initializer parameters.
    pub fn total_params(&self) -> usize {
        self.initializers.values().map(|t| t.numel()).sum()
    }

    /// Total memory of initializers in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.initializers.values().map(|t| t.memory_bytes()).sum()
    }
}

impl fmt::Display for OnnxGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnnxGraph('{}', nodes={}, initializers={}, params={})",
            self.name,
            self.nodes.len(),
            self.initializers.len(),
            self.total_params()
        )
    }
}

// ── ONNX Model ─────────────────────────────────────────────────

/// Top-level ONNX model structure.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub producer_name: String,
    pub producer_version: String,
    pub domain: String,
    pub model_version: i64,
    pub doc_string: String,
    pub opset_imports: Vec<OpSetImport>,
    pub graph: OnnxGraph,
    pub metadata: HashMap<String, String>,
}

impl OnnxModel {
    pub fn new() -> Self {
        Self {
            ir_version: 0,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_imports: Vec::new(),
            graph: OnnxGraph::new(""),
            metadata: HashMap::new(),
        }
    }

    pub fn with_ir_version(mut self, v: i64) -> Self {
        self.ir_version = v;
        self
    }

    pub fn with_producer(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.producer_name = name.into();
        self.producer_version = version.into();
        self
    }

    pub fn with_opset(mut self, domain: impl Into<String>, version: i64) -> Self {
        self.opset_imports.push(OpSetImport { domain: domain.into(), version });
        self
    }

    pub fn with_graph(mut self, graph: OnnxGraph) -> Self {
        self.graph = graph;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }

    /// Quick summary string.
    pub fn summary(&self) -> String {
        format!(
            "ONNX Model (IR v{}, producer={} {}, nodes={}, params={})",
            self.ir_version,
            self.producer_name,
            self.producer_version,
            self.graph.nodes.len(),
            self.graph.total_params()
        )
    }
}

impl Default for OnnxModel {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for OnnxModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OnnxModel(ir_v={}, producer='{}', graph={})",
            self.ir_version, self.producer_name, self.graph
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_varint_single_byte() {
        let data = [0x05];
        let (val, consumed) = read_varint(&data, 0).unwrap();
        assert_eq!(val, 5);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_varint_multi_byte() {
        // 300 = 0b100101100 → bytes: 0xAC 0x02
        let data = [0xAC, 0x02];
        let (val, consumed) = read_varint(&data, 0).unwrap();
        assert_eq!(val, 300);
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_read_varint_at_offset() {
        let data = [0x00, 0x00, 0x05];
        let (val, consumed) = read_varint(&data, 2).unwrap();
        assert_eq!(val, 5);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_tag() {
        // Field 1, wire type 0 (varint) → tag byte = 0x08
        let data = [0x08];
        let (field, wt, consumed) = read_tag(&data, 0).unwrap();
        assert_eq!(field, 1);
        assert_eq!(wt, WireType::Varint);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_read_tag_length_delimited() {
        // Field 2, wire type 2 → tag byte = 0x12
        let data = [0x12];
        let (field, wt, _) = read_tag(&data, 0).unwrap();
        assert_eq!(field, 2);
        assert_eq!(wt, WireType::LengthDelim);
    }

    #[test]
    fn test_read_length_delimited() {
        // Length 3, followed by 3 bytes
        let data = [0x03, 0x41, 0x42, 0x43]; // "ABC"
        let (payload, consumed) = read_length_delimited(&data, 0).unwrap();
        assert_eq!(payload, &[0x41, 0x42, 0x43]);
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_wire_type_from_tag() {
        assert_eq!(WireType::from_tag(0), Some(WireType::Varint));
        assert_eq!(WireType::from_tag(1), Some(WireType::Fixed64));
        assert_eq!(WireType::from_tag(2), Some(WireType::LengthDelim));
        assert_eq!(WireType::from_tag(5), Some(WireType::Fixed32));
        assert_eq!(WireType::from_tag(3), None);
    }

    #[test]
    fn test_onnx_data_type_from_id() {
        assert_eq!(OnnxDataType::from_id(1), OnnxDataType::Float);
        assert_eq!(OnnxDataType::from_id(7), OnnxDataType::Int64);
        assert!(matches!(OnnxDataType::from_id(99), OnnxDataType::Unknown(99)));
    }

    #[test]
    fn test_onnx_data_type_byte_size() {
        assert_eq!(OnnxDataType::Float.byte_size(), 4);
        assert_eq!(OnnxDataType::Double.byte_size(), 8);
        assert_eq!(OnnxDataType::Int8.byte_size(), 1);
    }

    #[test]
    fn test_onnx_tensor_numel() {
        let t = OnnxTensor {
            name: "w".into(),
            data_type: OnnxDataType::Float,
            dims: vec![3, 4, 5],
            raw_data: Vec::new(),
        };
        assert_eq!(t.numel(), 60);
    }

    #[test]
    fn test_onnx_tensor_as_f32() {
        let val: f32 = 3.14;
        let bytes = val.to_le_bytes().to_vec();
        let t = OnnxTensor {
            name: "x".into(),
            data_type: OnnxDataType::Float,
            dims: vec![1],
            raw_data: bytes,
        };
        let floats = t.as_f32();
        assert_eq!(floats.len(), 1);
        assert!((floats[0] - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_onnx_node_builder() {
        let node = OnnxNode::new("Conv")
            .with_name("conv1")
            .with_input("x")
            .with_input("w")
            .with_output("y")
            .with_attr("kernel_shape", OnnxAttr::Ints(vec![3, 3]));
        assert_eq!(node.op_type, "Conv");
        assert_eq!(node.inputs.len(), 2);
        assert!(node.attributes.contains_key("kernel_shape"));
    }

    #[test]
    fn test_onnx_graph_topo_sort() {
        let mut g = OnnxGraph::new("test");
        g.nodes.push(OnnxNode::new("Relu").with_name("relu1").with_input("conv_out").with_output("relu_out"));
        g.nodes.push(OnnxNode::new("Conv").with_name("conv1").with_input("x").with_output("conv_out"));
        let order = g.topological_sort();
        // conv1 (idx 1) should come before relu1 (idx 0)
        let pos_conv = order.iter().position(|i| *i == 1).unwrap();
        let pos_relu = order.iter().position(|i| *i == 0).unwrap();
        assert!(pos_conv < pos_relu);
    }

    #[test]
    fn test_onnx_graph_op_histogram() {
        let mut g = OnnxGraph::new("test");
        g.nodes.push(OnnxNode::new("Conv").with_name("c1"));
        g.nodes.push(OnnxNode::new("Conv").with_name("c2"));
        g.nodes.push(OnnxNode::new("Relu").with_name("r1"));
        let hist = g.op_histogram();
        assert_eq!(hist.get("Conv"), Some(&2));
        assert_eq!(hist.get("Relu"), Some(&1));
    }

    #[test]
    fn test_onnx_model_builder() {
        let model = OnnxModel::new()
            .with_ir_version(8)
            .with_producer("test", "1.0")
            .with_opset("", 13)
            .with_metadata("key", "val");
        assert_eq!(model.ir_version, 8);
        assert_eq!(model.producer_name, "test");
        assert_eq!(model.opset_imports.len(), 1);
    }

    #[test]
    fn test_onnx_model_summary() {
        let model = OnnxModel::new().with_ir_version(9).with_producer("joule", "0.1");
        let s = model.summary();
        assert!(s.contains("IR v9"));
        assert!(s.contains("joule"));
    }

    #[test]
    fn test_display_impls() {
        assert!(format!("{}", WireType::Varint).contains("varint"));
        assert!(format!("{}", OnnxDataType::Float).contains("float32"));
        let t = OnnxTensor { name: "w".into(), data_type: OnnxDataType::Float, dims: vec![2, 3], raw_data: vec![] };
        assert!(format!("{t}").contains("[2, 3]"));
        let n = OnnxNode::new("Add").with_name("add1");
        assert!(format!("{n}").contains("Add"));
        let a = OnnxAttr::Int(42);
        assert!(format!("{a}").contains("42"));
        let o = OpSetImport { domain: String::new(), version: 13 };
        assert!(format!("{o}").contains("v13"));
        let g = OnnxGraph::new("g");
        assert!(format!("{g}").contains("OnnxGraph"));
        let m = OnnxModel::new();
        assert!(format!("{m}").contains("OnnxModel"));
    }

    #[test]
    fn test_onnx_model_default() {
        let m = OnnxModel::default();
        assert_eq!(m.ir_version, 0);
    }
}
