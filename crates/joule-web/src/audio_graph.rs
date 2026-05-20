//! Audio processing graph — node-based routing with topological sort and cycle detection.
//!
//! Each node wraps an `AudioGraphProcessor` and can have multiple input/output
//! ports. The graph processes nodes in topological order, detects cycles,
//! supports node bypass, graph validation, and buffer management between nodes.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Processor Trait ─────────────────────────────────────────────

/// Audio processor for graph nodes.
pub trait AudioGraphProcessor: Send {
    /// Process input buffers to output buffers for `frames` samples.
    /// `inputs` is indexed by input port, `outputs` by output port.
    fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>], frames: usize);

    /// Number of input ports.
    fn input_count(&self) -> usize;

    /// Number of output ports.
    fn output_count(&self) -> usize;

    /// Processor name.
    fn name(&self) -> &str;

    /// Reset processor state (e.g., clear delay lines, filters).
    fn reset(&mut self) {}
}

// ── Connection ──────────────────────────────────────────────────

/// A connection between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Connection {
    pub source_id: u64,
    pub source_port: usize,
    pub dest_id: u64,
    pub dest_port: usize,
}

// ── Port Info ───────────────────────────────────────────────────

/// Describes an audio port.
#[derive(Debug, Clone)]
pub struct PortInfo {
    pub name: String,
    pub port_type: PortType,
}

/// Port direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortType {
    Input,
    Output,
}

// ── Graph Node ──────────────────────────────────────────────────

/// A node in the audio graph.
pub struct AudioGraphNode {
    pub id: u64,
    pub processor: Box<dyn AudioGraphProcessor>,
    pub group: Option<u64>,
    pub bypassed: bool,
    pub label: String,
}

impl AudioGraphNode {
    pub fn new(id: u64, processor: Box<dyn AudioGraphProcessor>) -> Self {
        let label = processor.name().to_string();
        Self {
            id,
            processor,
            group: None,
            bypassed: false,
            label,
        }
    }
}

// ── Audio Bus ───────────────────────────────────────────────────

/// An audio bus representing a multi-channel signal path.
#[derive(Debug, Clone)]
pub struct AudioBus {
    pub name: String,
    pub channels: usize,
    pub buffers: Vec<Vec<f32>>,
}

impl AudioBus {
    /// Create a new audio bus with `channels` channels of `frames` samples.
    pub fn new(name: &str, channels: usize, frames: usize) -> Self {
        Self {
            name: name.to_string(),
            channels,
            buffers: vec![vec![0.0; frames]; channels],
        }
    }

    /// Clear all buffers to silence.
    pub fn clear(&mut self) {
        for buf in &mut self.buffers {
            for s in buf.iter_mut() {
                *s = 0.0;
            }
        }
    }

    /// Get the number of frames.
    pub fn frames(&self) -> usize {
        self.buffers.first().map_or(0, |b| b.len())
    }

    /// Mix another bus into this one (additive).
    pub fn mix_from(&mut self, other: &AudioBus) {
        let channels = self.channels.min(other.channels);
        let frames = self.frames().min(other.frames());
        for ch in 0..channels {
            for i in 0..frames {
                self.buffers[ch][i] += other.buffers[ch][i];
            }
        }
    }

    /// Apply gain to all channels.
    pub fn apply_gain(&mut self, gain: f32) {
        for buf in &mut self.buffers {
            for s in buf.iter_mut() {
                *s *= gain;
            }
        }
    }
}

// ── Validation Error ────────────────────────────────────────────

/// Validation issue found in the graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationIssue {
    /// Node has an input port with no incoming connection.
    UnconnectedInput { node_id: u64, port: usize },
    /// Node has an output port with no outgoing connection.
    UnconnectedOutput { node_id: u64, port: usize },
    /// Connection references a port that exceeds the processor's port count.
    InvalidPort {
        node_id: u64,
        port: usize,
        direction: PortType,
    },
    /// The graph contains a cycle.
    CycleDetected,
}

// ── Audio Graph ─────────────────────────────────────────────────

/// Node-based audio routing graph.
pub struct AudioGraph {
    nodes: HashMap<u64, AudioGraphNode>,
    connections: Vec<Connection>,
    next_id: u64,
    sample_rate: f32,
    processing_order: Vec<u64>,
    order_dirty: bool,
}

impl AudioGraph {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            nodes: HashMap::new(),
            connections: Vec::new(),
            next_id: 1,
            sample_rate,
            processing_order: Vec::new(),
            order_dirty: true,
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, processor: Box<dyn AudioGraphProcessor>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, AudioGraphNode::new(id, processor));
        self.order_dirty = true;
        id
    }

    /// Add a node with a label.
    pub fn add_labeled_node(
        &mut self,
        processor: Box<dyn AudioGraphProcessor>,
        label: &str,
    ) -> u64 {
        let id = self.add_node(processor);
        if let Some(node) = self.nodes.get_mut(&id) {
            node.label = label.to_string();
        }
        id
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, id: u64) -> bool {
        if self.nodes.remove(&id).is_some() {
            self.connections
                .retain(|c| c.source_id != id && c.dest_id != id);
            self.order_dirty = true;
            true
        } else {
            false
        }
    }

    /// Connect an output port of one node to an input port of another.
    /// Returns `Err` if the connection would create a cycle or nodes don't exist.
    pub fn connect(
        &mut self,
        source_id: u64,
        source_port: usize,
        dest_id: u64,
        dest_port: usize,
    ) -> Result<(), GraphError> {
        if !self.nodes.contains_key(&source_id) {
            return Err(GraphError::NodeNotFound(source_id));
        }
        if !self.nodes.contains_key(&dest_id) {
            return Err(GraphError::NodeNotFound(dest_id));
        }

        // Validate port indices
        if let Some(node) = self.nodes.get(&source_id) {
            if source_port >= node.processor.output_count() {
                return Err(GraphError::PortOutOfRange);
            }
        }
        if let Some(node) = self.nodes.get(&dest_id) {
            if dest_port >= node.processor.input_count() {
                return Err(GraphError::PortOutOfRange);
            }
        }

        let conn = Connection {
            source_id,
            source_port,
            dest_id,
            dest_port,
        };

        if self.connections.contains(&conn) {
            return Ok(());
        }

        // Tentatively add and check for cycles
        self.connections.push(conn);
        if self.has_cycle() {
            self.connections.pop();
            return Err(GraphError::CycleDetected);
        }

        self.order_dirty = true;
        Ok(())
    }

    /// Disconnect a specific connection.
    pub fn disconnect(
        &mut self,
        source_id: u64,
        source_port: usize,
        dest_id: u64,
        dest_port: usize,
    ) -> bool {
        let conn = Connection {
            source_id,
            source_port,
            dest_id,
            dest_port,
        };
        let before = self.connections.len();
        self.connections.retain(|c| *c != conn);
        if self.connections.len() < before {
            self.order_dirty = true;
            true
        } else {
            false
        }
    }

    /// Disconnect all connections from a source node.
    pub fn disconnect_all_from(&mut self, source_id: u64) {
        self.connections.retain(|c| c.source_id != source_id);
        self.order_dirty = true;
    }

    /// Disconnect all connections to a destination node.
    pub fn disconnect_all_to(&mut self, dest_id: u64) {
        self.connections.retain(|c| c.dest_id != dest_id);
        self.order_dirty = true;
    }

    /// Bypass a node (passes input straight to output when processing).
    pub fn set_bypass(&mut self, node_id: u64, bypassed: bool) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.bypassed = bypassed;
        }
    }

    /// Check if a node is bypassed.
    pub fn is_bypassed(&self, node_id: u64) -> bool {
        self.nodes.get(&node_id).map_or(false, |n| n.bypassed)
    }

    /// Reset all node processors.
    pub fn reset_all(&mut self) {
        for node in self.nodes.values_mut() {
            node.processor.reset();
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Get a node label.
    pub fn node_label(&self, node_id: u64) -> Option<&str> {
        self.nodes.get(&node_id).map(|n| n.label.as_str())
    }

    /// Check if the graph contains a cycle using DFS.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();

        for &id in self.nodes.keys() {
            if !visited.contains(&id) && self.dfs_cycle(id, &mut visited, &mut on_stack) {
                return true;
            }
        }

        false
    }

    fn dfs_cycle(
        &self,
        node: u64,
        visited: &mut HashSet<u64>,
        on_stack: &mut HashSet<u64>,
    ) -> bool {
        visited.insert(node);
        on_stack.insert(node);

        for conn in &self.connections {
            if conn.source_id == node {
                let neighbor = conn.dest_id;
                if on_stack.contains(&neighbor) {
                    return true;
                }
                if !visited.contains(&neighbor)
                    && self.dfs_cycle(neighbor, visited, on_stack)
                {
                    return true;
                }
            }
        }

        on_stack.remove(&node);
        false
    }

    /// Compute topological sort using Kahn's algorithm.
    pub fn topological_sort(&self) -> Result<Vec<u64>, GraphError> {
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
        }
        for conn in &self.connections {
            *in_degree.entry(conn.dest_id).or_insert(0) += 1;
        }

        let mut queue: VecDeque<u64> = VecDeque::new();
        let mut zero_nodes: Vec<u64> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        zero_nodes.sort();
        for id in zero_nodes {
            queue.push_back(id);
        }

        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut next_nodes: Vec<u64> = Vec::new();
            for conn in &self.connections {
                if conn.source_id == node {
                    let deg = in_degree.get_mut(&conn.dest_id).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        next_nodes.push(conn.dest_id);
                    }
                }
            }
            next_nodes.sort();
            for n in next_nodes {
                queue.push_back(n);
            }
        }

        if order.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }

        Ok(order)
    }

    /// Process the entire graph for `frames` samples.
    pub fn process(&mut self, frames: usize) -> HashMap<u64, Vec<Vec<f32>>> {
        if self.order_dirty {
            self.processing_order = self.topological_sort().unwrap_or_default();
            self.order_dirty = false;
        }

        let order = self.processing_order.clone();

        // Allocate output buffers
        let mut outputs: HashMap<u64, Vec<Vec<f32>>> = HashMap::new();
        for &id in &order {
            if let Some(node) = self.nodes.get(&id) {
                let out_count = node.processor.output_count().max(1);
                let bufs: Vec<Vec<f32>> = (0..out_count).map(|_| vec![0.0f32; frames]).collect();
                outputs.insert(id, bufs);
            }
        }

        // Process in topological order
        for &id in &order {
            let input_count = self
                .nodes
                .get(&id)
                .map_or(1, |n| n.processor.input_count().max(1));

            let mut inputs: Vec<Vec<f32>> = (0..input_count)
                .map(|_| vec![0.0f32; frames])
                .collect();

            // Gather inputs from connected source nodes
            for conn in &self.connections {
                if conn.dest_id == id && conn.dest_port < input_count {
                    if let Some(src_outputs) = outputs.get(&conn.source_id) {
                        if conn.source_port < src_outputs.len() {
                            let src = &src_outputs[conn.source_port];
                            let dest = &mut inputs[conn.dest_port];
                            for i in 0..frames.min(src.len()).min(dest.len()) {
                                dest[i] += src[i];
                            }
                        }
                    }
                }
            }

            if let Some(node) = self.nodes.get_mut(&id) {
                if node.bypassed {
                    // Bypass: copy inputs to outputs directly
                    let out_count = node.processor.output_count().max(1);
                    let mut node_outputs: Vec<Vec<f32>> = (0..out_count)
                        .map(|_| vec![0.0f32; frames])
                        .collect();
                    let copy_count = inputs.len().min(node_outputs.len());
                    for i in 0..copy_count {
                        let len = frames.min(inputs[i].len()).min(node_outputs[i].len());
                        node_outputs[i][..len].copy_from_slice(&inputs[i][..len]);
                    }
                    outputs.insert(id, node_outputs);
                } else {
                    let out_count = node.processor.output_count().max(1);
                    let mut node_outputs: Vec<Vec<f32>> = (0..out_count)
                        .map(|_| vec![0.0f32; frames])
                        .collect();
                    node.processor.process(&inputs, &mut node_outputs, frames);
                    outputs.insert(id, node_outputs);
                }
            }
        }

        outputs
    }

    /// Assign a node to a subgraph group.
    pub fn set_node_group(&mut self, node_id: u64, group_id: u64) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.group = Some(group_id);
        }
    }

    /// Get all node IDs in a group.
    pub fn nodes_in_group(&self, group_id: u64) -> Vec<u64> {
        self.nodes
            .values()
            .filter(|n| n.group == Some(group_id))
            .map(|n| n.id)
            .collect()
    }

    /// Get the processing order.
    pub fn get_processing_order(&mut self) -> Result<Vec<u64>, GraphError> {
        if self.order_dirty {
            self.processing_order = self.topological_sort()?;
            self.order_dirty = false;
        }
        Ok(self.processing_order.clone())
    }

    /// Get connections for a specific node.
    pub fn connections_for(&self, node_id: u64) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.source_id == node_id || c.dest_id == node_id)
            .collect()
    }

    /// Get input connections for a node.
    pub fn inputs_for(&self, node_id: u64) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.dest_id == node_id)
            .collect()
    }

    /// Get output connections from a node.
    pub fn outputs_from(&self, node_id: u64) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.source_id == node_id)
            .collect()
    }

    /// Validate the graph for common issues.
    pub fn validate(&self) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if self.has_cycle() {
            issues.push(ValidationIssue::CycleDetected);
        }

        // Check port validity and unconnected ports
        for node in self.nodes.values() {
            let in_count = node.processor.input_count();
            let out_count = node.processor.output_count();

            // Check each input port has a connection
            for port in 0..in_count {
                let has_connection = self
                    .connections
                    .iter()
                    .any(|c| c.dest_id == node.id && c.dest_port == port);
                if !has_connection {
                    issues.push(ValidationIssue::UnconnectedInput {
                        node_id: node.id,
                        port,
                    });
                }
            }

            // Check each output port has a connection
            for port in 0..out_count {
                let has_connection = self
                    .connections
                    .iter()
                    .any(|c| c.source_id == node.id && c.source_port == port);
                if !has_connection {
                    issues.push(ValidationIssue::UnconnectedOutput {
                        node_id: node.id,
                        port,
                    });
                }
            }
        }

        // Check connections reference valid ports
        for conn in &self.connections {
            if let Some(src_node) = self.nodes.get(&conn.source_id) {
                if conn.source_port >= src_node.processor.output_count() {
                    issues.push(ValidationIssue::InvalidPort {
                        node_id: conn.source_id,
                        port: conn.source_port,
                        direction: PortType::Output,
                    });
                }
            }
            if let Some(dst_node) = self.nodes.get(&conn.dest_id) {
                if conn.dest_port >= dst_node.processor.input_count() {
                    issues.push(ValidationIssue::InvalidPort {
                        node_id: conn.dest_id,
                        port: conn.dest_port,
                        direction: PortType::Input,
                    });
                }
            }
        }

        issues
    }

    /// Get all source (root) nodes — nodes with no incoming connections.
    pub fn source_nodes(&self) -> Vec<u64> {
        let has_input: HashSet<u64> = self.connections.iter().map(|c| c.dest_id).collect();
        let mut sources: Vec<u64> = self
            .nodes
            .keys()
            .filter(|id| !has_input.contains(id))
            .copied()
            .collect();
        sources.sort();
        sources
    }

    /// Get all sink (leaf) nodes — nodes with no outgoing connections.
    pub fn sink_nodes(&self) -> Vec<u64> {
        let has_output: HashSet<u64> = self.connections.iter().map(|c| c.source_id).collect();
        let mut sinks: Vec<u64> = self
            .nodes
            .keys()
            .filter(|id| !has_output.contains(id))
            .copied()
            .collect();
        sinks.sort();
        sinks
    }
}

// ── Errors ──────────────────────────────────────────────────────

/// Graph operation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    NodeNotFound(u64),
    CycleDetected,
    PortOutOfRange,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::CycleDetected => write!(f, "cycle detected in audio graph"),
            Self::PortOutOfRange => write!(f, "port index out of range"),
        }
    }
}

impl std::error::Error for GraphError {}

// ── Built-in Processors ─────────────────────────────────────────

/// Pass-through processor (copies input to output).
pub struct PassthroughProcessor {
    inputs: usize,
    outputs: usize,
}

impl PassthroughProcessor {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self { inputs, outputs }
    }

    pub fn mono() -> Self {
        Self::new(1, 1)
    }

    pub fn stereo() -> Self {
        Self::new(2, 2)
    }
}

impl AudioGraphProcessor for PassthroughProcessor {
    fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>], frames: usize) {
        let n = inputs.len().min(outputs.len());
        for i in 0..n {
            let len = frames.min(inputs[i].len()).min(outputs[i].len());
            outputs[i][..len].copy_from_slice(&inputs[i][..len]);
        }
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn output_count(&self) -> usize {
        self.outputs
    }

    fn name(&self) -> &str {
        "Passthrough"
    }
}

/// Gain processor that scales all samples.
pub struct GainProcessor {
    pub gain: f32,
}

impl GainProcessor {
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl AudioGraphProcessor for GainProcessor {
    fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>], frames: usize) {
        if let (Some(inp), Some(out)) = (inputs.first(), outputs.first_mut()) {
            let len = frames.min(inp.len()).min(out.len());
            for i in 0..len {
                out[i] = inp[i] * self.gain;
            }
        }
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Gain"
    }
}

/// Mixer processor that sums multiple inputs to one output.
pub struct MixerProcessor {
    input_count: usize,
}

impl MixerProcessor {
    pub fn new(input_count: usize) -> Self {
        Self { input_count }
    }
}

impl AudioGraphProcessor for MixerProcessor {
    fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>], frames: usize) {
        if let Some(out) = outputs.first_mut() {
            for i in 0..frames.min(out.len()) {
                out[i] = 0.0;
                for inp in inputs {
                    if i < inp.len() {
                        out[i] += inp[i];
                    }
                }
            }
        }
    }

    fn input_count(&self) -> usize {
        self.input_count
    }

    fn output_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Mixer"
    }
}

/// Constant signal generator for testing.
pub struct ConstProcessor {
    pub value: f32,
}

impl ConstProcessor {
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}

impl AudioGraphProcessor for ConstProcessor {
    fn process(&mut self, _inputs: &[Vec<f32>], outputs: &mut [Vec<f32>], frames: usize) {
        if let Some(out) = outputs.first_mut() {
            for i in 0..frames.min(out.len()) {
                out[i] = self.value;
            }
        }
    }

    fn input_count(&self) -> usize {
        0
    }

    fn output_count(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "Const"
    }
}

/// Splitter: one input, multiple outputs (copies input to all outputs).
pub struct SplitterProcessor {
    output_count: usize,
}

impl SplitterProcessor {
    pub fn new(output_count: usize) -> Self {
        Self { output_count }
    }
}

impl AudioGraphProcessor for SplitterProcessor {
    fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>], frames: usize) {
        if let Some(inp) = inputs.first() {
            for out in outputs.iter_mut() {
                let len = frames.min(inp.len()).min(out.len());
                out[..len].copy_from_slice(&inp[..len]);
            }
        }
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        self.output_count
    }

    fn name(&self) -> &str {
        "Splitter"
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove_nodes() {
        let mut graph = AudioGraph::new(44100.0);
        let id = graph.add_node(Box::new(PassthroughProcessor::mono()));
        assert_eq!(graph.node_count(), 1);
        graph.remove_node(id);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn connect_nodes() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(GainProcessor::new(0.5)));
        assert!(graph.connect(a, 0, b, 0).is_ok());
        assert_eq!(graph.connection_count(), 1);
    }

    #[test]
    fn disconnect_nodes() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        assert!(graph.disconnect(a, 0, b, 0));
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn cycle_detection() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let c = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(b, 0, c, 0).unwrap();
        let result = graph.connect(c, 0, a, 0);
        assert_eq!(result, Err(GraphError::CycleDetected));
    }

    #[test]
    fn topological_sort_linear() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(GainProcessor::new(2.0)));
        let c = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(b, 0, c, 0).unwrap();

        let order = graph.topological_sort().unwrap();
        let pos_a = order.iter().position(|x| *x == a).unwrap();
        let pos_b = order.iter().position(|x| *x == b).unwrap();
        let pos_c = order.iter().position(|x| *x == c).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn process_linear_chain() {
        let mut graph = AudioGraph::new(44100.0);
        let src = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let gain_id = graph.add_node(Box::new(GainProcessor::new(0.5)));
        graph.connect(src, 0, gain_id, 0).unwrap();

        let outputs = graph.process(4);
        let gain_output = &outputs[&gain_id][0];
        assert_eq!(gain_output[0], 0.5);
        assert_eq!(gain_output[3], 0.5);
    }

    #[test]
    fn process_mixer() {
        let mut graph = AudioGraph::new(44100.0);
        let src1 = graph.add_node(Box::new(ConstProcessor::new(0.3)));
        let src2 = graph.add_node(Box::new(ConstProcessor::new(0.7)));
        let mixer = graph.add_node(Box::new(MixerProcessor::new(2)));
        graph.connect(src1, 0, mixer, 0).unwrap();
        graph.connect(src2, 0, mixer, 1).unwrap();

        let outputs = graph.process(4);
        let mixed = &outputs[&mixer][0];
        assert!((mixed[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn node_bypass() {
        let mut graph = AudioGraph::new(44100.0);
        let src = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let gain_id = graph.add_node(Box::new(GainProcessor::new(0.5)));
        graph.connect(src, 0, gain_id, 0).unwrap();

        // Without bypass
        let outputs = graph.process(4);
        assert!((outputs[&gain_id][0][0] - 0.5).abs() < 0.001);

        // With bypass
        graph.set_bypass(gain_id, true);
        assert!(graph.is_bypassed(gain_id));
        let outputs = graph.process(4);
        assert!((outputs[&gain_id][0][0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn node_not_found_error() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let result = graph.connect(a, 0, 999, 0);
        assert_eq!(result, Err(GraphError::NodeNotFound(999)));
    }

    #[test]
    fn port_out_of_range() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        // ConstProcessor has 1 output (port 0), trying port 1
        let result = graph.connect(a, 1, b, 0);
        assert_eq!(result, Err(GraphError::PortOutOfRange));
    }

    #[test]
    fn subgraph_grouping() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let c = graph.add_node(Box::new(PassthroughProcessor::mono()));

        graph.set_node_group(a, 1);
        graph.set_node_group(b, 1);
        graph.set_node_group(c, 2);

        let group1 = graph.nodes_in_group(1);
        assert_eq!(group1.len(), 2);
        assert!(group1.contains(&a));
        assert!(group1.contains(&b));
    }

    #[test]
    fn disconnect_all_from() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let c = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(a, 0, c, 0).unwrap();
        assert_eq!(graph.connection_count(), 2);
        graph.disconnect_all_from(a);
        assert_eq!(graph.connection_count(), 0);
    }

    #[test]
    fn connections_for_node() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let c = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(a, 0, c, 0).unwrap();
        let conns = graph.connections_for(a);
        assert_eq!(conns.len(), 2);
    }

    #[test]
    fn empty_graph_process() {
        let mut graph = AudioGraph::new(44100.0);
        let outputs = graph.process(256);
        assert!(outputs.is_empty());
    }

    #[test]
    fn remove_node_cleans_connections() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        graph.remove_node(a);
        assert_eq!(graph.connection_count(), 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn source_and_sink_nodes() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(GainProcessor::new(0.5)));
        let c = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(a, 0, b, 0).unwrap();
        graph.connect(b, 0, c, 0).unwrap();

        let sources = graph.source_nodes();
        assert!(sources.contains(&a));
        assert!(!sources.contains(&b));

        let sinks = graph.sink_nodes();
        assert!(sinks.contains(&c));
        assert!(!sinks.contains(&b));
    }

    #[test]
    fn validate_graph() {
        let mut graph = AudioGraph::new(44100.0);
        let a = graph.add_node(Box::new(ConstProcessor::new(1.0)));
        let b = graph.add_node(Box::new(PassthroughProcessor::mono()));
        // Don't connect them — b has an unconnected input
        let _unused = a;
        let _unused2 = b;
        let issues = graph.validate();
        // Should find at least unconnected ports
        assert!(!issues.is_empty());
    }

    #[test]
    fn labeled_node() {
        let mut graph = AudioGraph::new(44100.0);
        let id = graph.add_labeled_node(Box::new(ConstProcessor::new(1.0)), "Source A");
        assert_eq!(graph.node_label(id), Some("Source A"));
    }

    #[test]
    fn audio_bus_basic() {
        let mut bus = AudioBus::new("main", 2, 512);
        assert_eq!(bus.channels, 2);
        assert_eq!(bus.frames(), 512);
        bus.buffers[0][0] = 1.0;
        bus.clear();
        assert!(bus.buffers[0][0].abs() < 1e-6);
    }

    #[test]
    fn audio_bus_mix() {
        let mut bus1 = AudioBus::new("a", 1, 4);
        bus1.buffers[0] = vec![0.3, 0.3, 0.3, 0.3];
        let mut bus2 = AudioBus::new("b", 1, 4);
        bus2.buffers[0] = vec![0.7, 0.7, 0.7, 0.7];
        bus1.mix_from(&bus2);
        assert!((bus1.buffers[0][0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn splitter_processor() {
        let mut graph = AudioGraph::new(44100.0);
        let src = graph.add_node(Box::new(ConstProcessor::new(0.5)));
        let split = graph.add_node(Box::new(SplitterProcessor::new(2)));
        let out1 = graph.add_node(Box::new(PassthroughProcessor::mono()));
        let out2 = graph.add_node(Box::new(PassthroughProcessor::mono()));
        graph.connect(src, 0, split, 0).unwrap();
        graph.connect(split, 0, out1, 0).unwrap();
        graph.connect(split, 1, out2, 0).unwrap();

        let outputs = graph.process(4);
        assert!((outputs[&out1][0][0] - 0.5).abs() < 0.001);
        assert!((outputs[&out2][0][0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn graph_error_display() {
        assert_eq!(format!("{}", GraphError::CycleDetected), "cycle detected in audio graph");
        assert_eq!(
            format!("{}", GraphError::NodeNotFound(42)),
            "node 42 not found"
        );
    }
}
