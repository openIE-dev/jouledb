//! Audio processing graph engine with topological-order buffer processing.
//!
//! Nodes: source (oscillator, sample player), effect (gain, filter, delay),
//! output (speaker). Connections between node output→input ports. Process
//! graph per audio buffer in topological order. Per-node energy tracking.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Types ──────────────────────────────────────────────────────

/// Unique identifier for a node in the audio graph.
pub type NodeId = u64;

/// Unique identifier for a connection.
pub type ConnectionId = u64;

/// Sample rate in Hz.
pub type SampleRate = u32;

/// Audio buffer — interleaved f32 samples.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub channels: usize,
}

impl AudioBuffer {
    pub fn new(frames: usize, channels: usize) -> Self {
        Self {
            samples: vec![0.0; frames * channels],
            channels,
        }
    }

    pub fn frames(&self) -> usize {
        if self.channels == 0 { 0 } else { self.samples.len() / self.channels }
    }

    pub fn mix_from(&mut self, other: &AudioBuffer) {
        let len = self.samples.len().min(other.samples.len());
        for i in 0..len {
            self.samples[i] += other.samples[i];
        }
    }
}

// ── Node Types ─────────────────────────────────────────────────

/// Oscillator waveform shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Waveform {
    Sine,
    Square,
    Sawtooth,
    Triangle,
}

/// Source node configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceConfig {
    Oscillator { waveform: Waveform, frequency: f32, amplitude: f32, phase: f32 },
    SamplePlayer { sample_data: Vec<f32>, playback_rate: f32, looping: bool, position: usize },
}

/// Effect node configuration.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectConfig {
    Gain { level: f32 },
    Filter { cutoff: f32, resonance: f32, kind: FilterKind },
    Delay { delay_samples: usize, feedback: f32, buffer: Vec<f32>, write_pos: usize },
}

/// Filter type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterKind {
    LowPass,
    HighPass,
    BandPass,
}

/// Output node configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputConfig {
    Speaker { channel_count: usize },
}

/// The kind of node in the graph.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Source(SourceConfig),
    Effect(EffectConfig),
    Output(OutputConfig),
}

/// A node in the audio processing graph.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioNode {
    pub id: NodeId,
    pub name: String,
    pub kind: NodeKind,
    pub input_ports: usize,
    pub output_ports: usize,
    pub energy_uj: u64,
    pub enabled: bool,
}

/// A connection between two nodes (output port → input port).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Connection {
    pub id: ConnectionId,
    pub from_node: NodeId,
    pub from_port: usize,
    pub to_node: NodeId,
    pub to_port: usize,
}

/// Graph processing configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphConfig {
    pub sample_rate: SampleRate,
    pub buffer_size: usize,
    pub channels: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self { sample_rate: 44100, buffer_size: 256, channels: 2 }
    }
}

// ── Audio Graph Engine ─────────────────────────────────────────

/// The audio processing graph.
#[derive(Debug, Clone)]
pub struct AudioGraphEngine {
    nodes: HashMap<NodeId, AudioNode>,
    connections: HashMap<ConnectionId, Connection>,
    config: GraphConfig,
    next_node_id: NodeId,
    next_conn_id: ConnectionId,
    total_energy_uj: u64,
}

impl AudioGraphEngine {
    /// Create a new audio graph with the given config.
    pub fn new(config: GraphConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            connections: HashMap::new(),
            config,
            next_node_id: 1,
            next_conn_id: 1,
            total_energy_uj: 0,
        }
    }

    /// Return the graph config.
    pub fn config(&self) -> &GraphConfig {
        &self.config
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, name: &str, kind: NodeKind) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        let (input_ports, output_ports) = match &kind {
            NodeKind::Source(_) => (0, 1),
            NodeKind::Effect(_) => (1, 1),
            NodeKind::Output(_) => (1, 0),
        };
        let node = AudioNode {
            id,
            name: name.to_string(),
            kind,
            input_ports,
            output_ports,
            energy_uj: 0,
            enabled: true,
        };
        self.nodes.insert(id, node);
        id
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, id: NodeId) -> Option<AudioNode> {
        let node = self.nodes.remove(&id)?;
        let to_remove: Vec<ConnectionId> = self.connections.iter()
            .filter(|(_, c)| c.from_node == id || c.to_node == id)
            .map(|(cid, _)| *cid)
            .collect();
        for cid in to_remove {
            self.connections.remove(&cid);
        }
        Some(node)
    }

    /// Get a reference to a node.
    pub fn get_node(&self, id: NodeId) -> Option<&AudioNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable reference to a node.
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut AudioNode> {
        self.nodes.get_mut(&id)
    }

    /// Connect output port of one node to input port of another.
    pub fn connect(&mut self, from_node: NodeId, from_port: usize,
                   to_node: NodeId, to_port: usize) -> Result<ConnectionId, GraphError> {
        let from = self.nodes.get(&from_node).ok_or(GraphError::NodeNotFound(from_node))?;
        if from_port >= from.output_ports {
            return Err(GraphError::InvalidPort { node: from_node, port: from_port });
        }
        let to = self.nodes.get(&to_node).ok_or(GraphError::NodeNotFound(to_node))?;
        if to_port >= to.input_ports {
            return Err(GraphError::InvalidPort { node: to_node, port: to_port });
        }
        if from_node == to_node {
            return Err(GraphError::CycleDetected);
        }

        let id = self.next_conn_id;
        let conn = Connection { id, from_node, from_port, to_node, to_port };

        // Temporarily add and check for cycle
        self.connections.insert(id, conn);
        if self.has_cycle() {
            self.connections.remove(&id);
            return Err(GraphError::CycleDetected);
        }
        self.next_conn_id += 1;
        Ok(id)
    }

    /// Disconnect a connection by its ID.
    pub fn disconnect(&mut self, conn_id: ConnectionId) -> Option<Connection> {
        self.connections.remove(&conn_id)
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Check if the graph contains a cycle using DFS.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        for &id in self.nodes.keys() {
            if self.dfs_cycle(id, &mut visited, &mut stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(&self, node: NodeId, visited: &mut HashSet<NodeId>,
                 stack: &mut HashSet<NodeId>) -> bool {
        if stack.contains(&node) { return true; }
        if visited.contains(&node) { return false; }
        visited.insert(node);
        stack.insert(node);
        for conn in self.connections.values() {
            if conn.from_node == node {
                if self.dfs_cycle(conn.to_node, visited, stack) {
                    return true;
                }
            }
        }
        stack.remove(&node);
        false
    }

    /// Compute topological sort of the graph.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        if self.has_cycle() {
            return Err(GraphError::CycleDetected);
        }
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
        }
        for conn in self.connections.values() {
            *in_degree.entry(conn.to_node).or_insert(0) += 1;
        }
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut zero_nodes: Vec<NodeId> = in_degree.iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| *id)
            .collect();
        zero_nodes.sort();
        for n in zero_nodes {
            queue.push_back(n);
        }
        let mut result = Vec::new();
        while let Some(n) = queue.pop_front() {
            result.push(n);
            let mut next: Vec<NodeId> = Vec::new();
            for conn in self.connections.values() {
                if conn.from_node == n {
                    let deg = in_degree.get_mut(&conn.to_node).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        next.push(conn.to_node);
                    }
                }
            }
            next.sort();
            for m in next {
                queue.push_back(m);
            }
        }
        if result.len() != self.nodes.len() {
            return Err(GraphError::CycleDetected);
        }
        Ok(result)
    }

    /// Validate the graph: no cycles, all connections valid.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.has_cycle() {
            return Err(GraphError::CycleDetected);
        }
        for conn in self.connections.values() {
            if !self.nodes.contains_key(&conn.from_node) {
                return Err(GraphError::NodeNotFound(conn.from_node));
            }
            if !self.nodes.contains_key(&conn.to_node) {
                return Err(GraphError::NodeNotFound(conn.to_node));
            }
        }
        Ok(())
    }

    /// Process the entire graph, producing output at each output node.
    pub fn process(&mut self) -> Result<HashMap<NodeId, AudioBuffer>, GraphError> {
        let order = self.topological_sort()?;
        let buf_size = self.config.buffer_size;
        let channels = self.config.channels;
        let sr = self.config.sample_rate;

        let mut node_outputs: HashMap<NodeId, AudioBuffer> = HashMap::new();

        for &nid in &order {
            // Collect inputs from connected nodes
            let input_conns: Vec<Connection> = self.connections.values()
                .filter(|c| c.to_node == nid)
                .cloned()
                .collect();

            let mut input_buf = AudioBuffer::new(buf_size, channels);
            for conn in &input_conns {
                if let Some(src_buf) = node_outputs.get(&conn.from_node) {
                    input_buf.mix_from(src_buf);
                }
            }

            let node = self.nodes.get_mut(&nid).unwrap();
            let output = if node.enabled {
                let out = process_node(node, &input_buf, sr, buf_size, channels);
                // Energy: ~1 uJ per frame processed
                let energy = buf_size as u64;
                node.energy_uj += energy;
                out
            } else {
                AudioBuffer::new(buf_size, channels)
            };
            node_outputs.insert(nid, output);
        }

        // Accumulate total energy
        for node in self.nodes.values() {
            // already accumulated per-node
        }
        let _ = &self.nodes; // suppress warning

        // Return only output node buffers
        let mut result = HashMap::new();
        for (&nid, node) in &self.nodes {
            if matches!(node.kind, NodeKind::Output(_)) {
                if let Some(buf) = node_outputs.remove(&nid) {
                    result.insert(nid, buf);
                }
            }
        }
        Ok(result)
    }

    /// Total energy consumed by all nodes in microjoules.
    pub fn total_energy_uj(&self) -> u64 {
        self.nodes.values().map(|n| n.energy_uj).sum()
    }

    /// Reset all node energy counters.
    pub fn reset_energy(&mut self) {
        for node in self.nodes.values_mut() {
            node.energy_uj = 0;
        }
    }

    /// List all node IDs.
    pub fn node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// List all connection IDs.
    pub fn connection_ids(&self) -> Vec<ConnectionId> {
        let mut ids: Vec<ConnectionId> = self.connections.keys().cloned().collect();
        ids.sort();
        ids
    }
}

/// Process a single node given its input buffer.
fn process_node(node: &mut AudioNode, input: &AudioBuffer,
                sample_rate: SampleRate, buf_size: usize, channels: usize) -> AudioBuffer {
    match &mut node.kind {
        NodeKind::Source(src) => {
            let mut out = AudioBuffer::new(buf_size, channels);
            match src {
                SourceConfig::Oscillator { waveform, frequency, amplitude, phase } => {
                    let sr = sample_rate as f32;
                    for frame in 0..buf_size {
                        let t = *phase + (frame as f32 * *frequency / sr);
                        let sample = match waveform {
                            Waveform::Sine => (t * 2.0 * std::f32::consts::PI).sin() * *amplitude,
                            Waveform::Square => {
                                if (t * 2.0 * std::f32::consts::PI).sin() >= 0.0 {
                                    *amplitude
                                } else {
                                    -*amplitude
                                }
                            }
                            Waveform::Sawtooth => {
                                let frac = t - t.floor();
                                (2.0 * frac - 1.0) * *amplitude
                            }
                            Waveform::Triangle => {
                                let frac = t - t.floor();
                                (4.0 * (frac - 0.5).abs() - 1.0) * *amplitude
                            }
                        };
                        for ch in 0..channels {
                            out.samples[frame * channels + ch] = sample;
                        }
                    }
                    *phase += buf_size as f32 * *frequency / sample_rate as f32;
                }
                SourceConfig::SamplePlayer { sample_data, playback_rate, looping, position } => {
                    for frame in 0..buf_size {
                        let idx = (*position as f32 + frame as f32 * *playback_rate) as usize;
                        let sample = if idx < sample_data.len() {
                            sample_data[idx]
                        } else if *looping && !sample_data.is_empty() {
                            sample_data[idx % sample_data.len()]
                        } else {
                            0.0
                        };
                        for ch in 0..channels {
                            out.samples[frame * channels + ch] = sample;
                        }
                    }
                    *position += (buf_size as f32 * *playback_rate) as usize;
                }
            }
            out
        }
        NodeKind::Effect(eff) => {
            let mut out = input.clone();
            match eff {
                EffectConfig::Gain { level } => {
                    for s in &mut out.samples {
                        *s *= *level;
                    }
                }
                EffectConfig::Filter { cutoff, resonance, kind } => {
                    // Simple one-pole filter approximation
                    let rc = 1.0 / (2.0 * std::f32::consts::PI * *cutoff);
                    let dt = 1.0 / sample_rate as f32;
                    let alpha = dt / (rc + dt);
                    let _ = *resonance; // resonance affects Q, simplified here
                    let mut prev = vec![0.0f32; channels];
                    for frame in 0..buf_size.min(out.frames()) {
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let sample = out.samples[idx];
                            let filtered = match kind {
                                FilterKind::LowPass => {
                                    prev[ch] + alpha * (sample - prev[ch])
                                }
                                FilterKind::HighPass => {
                                    sample - (prev[ch] + alpha * (sample - prev[ch]))
                                }
                                FilterKind::BandPass => {
                                    let lp = prev[ch] + alpha * (sample - prev[ch]);
                                    sample - lp + 0.5 * lp
                                }
                            };
                            prev[ch] = filtered;
                            out.samples[idx] = filtered;
                        }
                    }
                }
                EffectConfig::Delay { delay_samples, feedback, buffer, write_pos } => {
                    let total = channels * *delay_samples;
                    if buffer.len() < total {
                        buffer.resize(total, 0.0);
                    }
                    for frame in 0..buf_size.min(out.frames()) {
                        for ch in 0..channels {
                            let idx = frame * channels + ch;
                            let buf_idx = (*write_pos * channels + ch) % buffer.len();
                            let delayed = buffer[buf_idx];
                            let mixed = out.samples[idx] + delayed * *feedback;
                            buffer[buf_idx] = out.samples[idx];
                            out.samples[idx] = mixed;
                        }
                        *write_pos = (*write_pos + 1) % *delay_samples;
                    }
                }
            }
            out
        }
        NodeKind::Output(_) => {
            input.clone()
        }
    }
}

/// Errors that can occur in graph operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    NodeNotFound(NodeId),
    InvalidPort { node: NodeId, port: usize },
    CycleDetected,
    ProcessingError(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "node not found: {}", id),
            GraphError::InvalidPort { node, port } => write!(f, "invalid port {} on node {}", port, node),
            GraphError::CycleDetected => write!(f, "cycle detected in audio graph"),
            GraphError::ProcessingError(msg) => write!(f, "processing error: {}", msg),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_engine() -> AudioGraphEngine {
        AudioGraphEngine::new(GraphConfig::default())
    }

    #[test]
    fn test_create_graph() {
        let g = default_engine();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.connection_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut g = default_engine();
        let osc = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        assert_eq!(g.node_count(), 1);
        let n = g.get_node(osc).unwrap();
        assert_eq!(n.name, "osc");
        assert_eq!(n.input_ports, 0);
        assert_eq!(n.output_ports, 1);
    }

    #[test]
    fn test_remove_node() {
        let mut g = default_engine();
        let osc = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let removed = g.remove_node(osc);
        assert!(removed.is_some());
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_remove_node_clears_connections() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let gain = g.add_node("gain", NodeKind::Effect(EffectConfig::Gain { level: 0.5 }));
        g.connect(src, 0, gain, 0).unwrap();
        assert_eq!(g.connection_count(), 1);
        g.remove_node(src);
        assert_eq!(g.connection_count(), 0);
    }

    #[test]
    fn test_connect_nodes() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 2 }));
        let cid = g.connect(src, 0, out, 0).unwrap();
        assert_eq!(g.connection_count(), 1);
        assert!(cid > 0);
    }

    #[test]
    fn test_invalid_port() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 2 }));
        assert!(matches!(g.connect(src, 5, out, 0), Err(GraphError::InvalidPort { .. })));
    }

    #[test]
    fn test_self_connection_rejected() {
        let mut g = default_engine();
        let eff = g.add_node("eff", NodeKind::Effect(EffectConfig::Gain { level: 1.0 }));
        assert!(matches!(g.connect(eff, 0, eff, 0), Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = default_engine();
        let a = g.add_node("a", NodeKind::Effect(EffectConfig::Gain { level: 1.0 }));
        let b = g.add_node("b", NodeKind::Effect(EffectConfig::Gain { level: 1.0 }));
        g.connect(a, 0, b, 0).unwrap();
        assert!(matches!(g.connect(b, 0, a, 0), Err(GraphError::CycleDetected)));
    }

    #[test]
    fn test_topological_sort() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let gain = g.add_node("gain", NodeKind::Effect(EffectConfig::Gain { level: 0.5 }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 2 }));
        g.connect(src, 0, gain, 0).unwrap();
        g.connect(gain, 0, out, 0).unwrap();
        let order = g.topological_sort().unwrap();
        let src_pos = order.iter().position(|x| *x == src).unwrap();
        let gain_pos = order.iter().position(|x| *x == gain).unwrap();
        let out_pos = order.iter().position(|x| *x == out).unwrap();
        assert!(src_pos < gain_pos);
        assert!(gain_pos < out_pos);
    }

    #[test]
    fn test_disconnect() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 2 }));
        let cid = g.connect(src, 0, out, 0).unwrap();
        g.disconnect(cid);
        assert_eq!(g.connection_count(), 0);
    }

    #[test]
    fn test_process_sine_osc() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        let result = g.process().unwrap();
        let buf = result.get(&out).unwrap();
        assert_eq!(buf.samples.len(), 64);
        // Sine at t=0 should be ~0
        assert!(buf.samples[0].abs() < 0.1);
    }

    #[test]
    fn test_gain_effect() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let gain = g.add_node("gain", NodeKind::Effect(EffectConfig::Gain { level: 0.5 }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, gain, 0).unwrap();
        g.connect(gain, 0, out, 0).unwrap();
        let result = g.process().unwrap();
        let buf = result.get(&out).unwrap();
        // All samples should be within [-0.5, 0.5] since gain is 0.5
        for &s in &buf.samples {
            assert!(s >= -0.5 - 1e-6 && s <= 0.5 + 1e-6);
        }
    }

    #[test]
    fn test_energy_tracking() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 128, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        g.process().unwrap();
        assert!(g.total_energy_uj() > 0);
    }

    #[test]
    fn test_reset_energy() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        g.process().unwrap();
        g.reset_energy();
        assert_eq!(g.total_energy_uj(), 0);
    }

    #[test]
    fn test_disabled_node() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        g.get_node_mut(src).unwrap().enabled = false;
        let result = g.process().unwrap();
        let buf = result.get(&out).unwrap();
        assert!(buf.samples.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn test_sample_player_source() {
        let samples: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("player", NodeKind::Source(SourceConfig::SamplePlayer {
            sample_data: samples, playback_rate: 1.0, looping: false, position: 0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        let result = g.process().unwrap();
        let buf = result.get(&out).unwrap();
        assert_eq!(buf.samples.len(), 64);
        assert!((buf.samples[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_validate_valid_graph() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 2 }));
        g.connect(src, 0, out, 0).unwrap();
        assert!(g.validate().is_ok());
    }

    #[test]
    fn test_audio_buffer_mix() {
        let mut a = AudioBuffer { samples: vec![1.0, 2.0, 3.0], channels: 1 };
        let b = AudioBuffer { samples: vec![0.5, 0.5, 0.5], channels: 1 };
        a.mix_from(&b);
        assert!((a.samples[0] - 1.5).abs() < 1e-6);
        assert!((a.samples[1] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_node_ids_sorted() {
        let mut g = default_engine();
        g.add_node("a", NodeKind::Effect(EffectConfig::Gain { level: 1.0 }));
        g.add_node("b", NodeKind::Effect(EffectConfig::Gain { level: 1.0 }));
        g.add_node("c", NodeKind::Effect(EffectConfig::Gain { level: 1.0 }));
        let ids = g.node_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids[0] < ids[1] && ids[1] < ids[2]);
    }

    #[test]
    fn test_square_wave_osc() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("sq", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Square, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        let result = g.process().unwrap();
        let buf = result.get(&out).unwrap();
        for &s in &buf.samples {
            assert!((s - 1.0).abs() < 1e-6 || (s - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_delay_effect() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let delay = g.add_node("delay", NodeKind::Effect(EffectConfig::Delay {
            delay_samples: 32, feedback: 0.3, buffer: Vec::new(), write_pos: 0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, delay, 0).unwrap();
        g.connect(delay, 0, out, 0).unwrap();
        let result = g.process().unwrap();
        let buf = result.get(&out).unwrap();
        assert_eq!(buf.samples.len(), 64);
    }

    #[test]
    fn test_nonexistent_node_connection() {
        let mut g = default_engine();
        let src = g.add_node("src", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        assert!(matches!(g.connect(src, 0, 999, 0), Err(GraphError::NodeNotFound(999))));
    }

    #[test]
    fn test_multiple_process_accumulates_energy() {
        let mut g = AudioGraphEngine::new(GraphConfig { sample_rate: 44100, buffer_size: 64, channels: 1 });
        let src = g.add_node("osc", NodeKind::Source(SourceConfig::Oscillator {
            waveform: Waveform::Sine, frequency: 440.0, amplitude: 1.0, phase: 0.0,
        }));
        let out = g.add_node("out", NodeKind::Output(OutputConfig::Speaker { channel_count: 1 }));
        g.connect(src, 0, out, 0).unwrap();
        g.process().unwrap();
        let e1 = g.total_energy_uj();
        g.process().unwrap();
        let e2 = g.total_energy_uj();
        assert!(e2 > e1);
    }

    #[test]
    fn test_graph_config_default() {
        let cfg = GraphConfig::default();
        assert_eq!(cfg.sample_rate, 44100);
        assert_eq!(cfg.buffer_size, 256);
        assert_eq!(cfg.channels, 2);
    }
}
