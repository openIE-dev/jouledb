//! Web Audio API–style audio engine — pure Rust, no browser dependencies.
//!
//! Provides an `AudioContext` with sample rate, channel count, and current time.
//! `AudioBuffer` stores interleaved channel data. Nodes implement the `AudioNode`
//! trait and can be wired into a processing graph that renders N frames at a time.

use std::collections::HashMap;

// ── AudioBuffer ─────────────────────────────────────────────────

/// Multi-channel audio buffer (one `Vec<f32>` per channel).
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    channels: Vec<Vec<f32>>,
    sample_rate: f32,
}

impl AudioBuffer {
    /// Create a silent buffer with `num_channels` channels of `length` samples.
    pub fn new(num_channels: usize, length: usize, sample_rate: f32) -> Self {
        Self {
            channels: vec![vec![0.0; length]; num_channels],
            sample_rate,
        }
    }

    /// Create a buffer from existing channel data.
    pub fn from_channels(channels: Vec<Vec<f32>>, sample_rate: f32) -> Self {
        Self { channels, sample_rate }
    }

    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }

    pub fn length(&self) -> usize {
        self.channels.first().map_or(0, |c| c.len())
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn duration(&self) -> f64 {
        self.length() as f64 / self.sample_rate as f64
    }

    /// Get immutable reference to channel data.
    pub fn get_channel_data(&self, channel: usize) -> Option<&[f32]> {
        self.channels.get(channel).map(|c| c.as_slice())
    }

    /// Get mutable reference to channel data.
    pub fn get_channel_data_mut(&mut self, channel: usize) -> Option<&mut [f32]> {
        self.channels.get_mut(channel).map(|c| c.as_mut_slice())
    }

    /// Copy data from one channel to another within this buffer.
    pub fn copy_channel(&mut self, src: usize, dst: usize) {
        if src == dst || src >= self.channels.len() || dst >= self.channels.len() {
            return;
        }
        let data = self.channels[src].clone();
        self.channels[dst] = data;
    }

    /// Mix (add) another buffer into this one.
    pub fn mix_in(&mut self, other: &AudioBuffer) {
        let ch = self.num_channels().min(other.num_channels());
        let len = self.length().min(other.length());
        for c in 0..ch {
            for i in 0..len {
                self.channels[c][i] += other.channels[c][i];
            }
        }
    }
}

// ── AudioNode trait ─────────────────────────────────────────────

/// Trait for audio processing nodes.
pub trait AudioNode: Send {
    /// Process `frames` samples from `input` into `output`.
    /// Both buffers have the same length >= `frames`.
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, frames: usize);

    /// Human-readable name for debugging.
    fn name(&self) -> &str;
}

// ── GainNode ────────────────────────────────────────────────────

/// Applies a linear gain to all channels.
#[derive(Debug, Clone)]
pub struct GainNode {
    pub gain: f32,
}

impl GainNode {
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl AudioNode for GainNode {
    fn process(&mut self, input: &AudioBuffer, output: &mut AudioBuffer, frames: usize) {
        let ch = input.num_channels().min(output.num_channels());
        for c in 0..ch {
            if let (Some(inp), Some(out)) = (
                input.get_channel_data(c),
                output.get_channel_data_mut(c),
            ) {
                for i in 0..frames.min(inp.len()).min(out.len()) {
                    out[i] = inp[i] * self.gain;
                }
            }
        }
    }

    fn name(&self) -> &str {
        "GainNode"
    }
}

// ── MixerNode ───────────────────────────────────────────────────

/// Sums multiple input buffers together.  Call `add_input` for each source
/// buffer, then `process` writes the mix to output and clears the accumulator.
#[derive(Debug)]
pub struct MixerNode {
    accum: Vec<Vec<f32>>,
    num_channels: usize,
    buffer_size: usize,
}

impl MixerNode {
    pub fn new(num_channels: usize, buffer_size: usize) -> Self {
        Self {
            accum: vec![vec![0.0; buffer_size]; num_channels],
            num_channels,
            buffer_size,
        }
    }

    /// Accumulate a source buffer into the mix.
    pub fn add_input(&mut self, buf: &AudioBuffer) {
        let ch = self.num_channels.min(buf.num_channels());
        let len = self.buffer_size.min(buf.length());
        for c in 0..ch {
            if let Some(src) = buf.get_channel_data(c) {
                for i in 0..len {
                    self.accum[c][i] += src[i];
                }
            }
        }
    }

    /// Reset the accumulator to silence.
    pub fn clear(&mut self) {
        for ch in &mut self.accum {
            for s in ch.iter_mut() {
                *s = 0.0;
            }
        }
    }
}

impl AudioNode for MixerNode {
    fn process(&mut self, _input: &AudioBuffer, output: &mut AudioBuffer, frames: usize) {
        let ch = self.num_channels.min(output.num_channels());
        for c in 0..ch {
            if let Some(out) = output.get_channel_data_mut(c) {
                let len = frames.min(out.len()).min(self.accum[c].len());
                out[..len].copy_from_slice(&self.accum[c][..len]);
            }
        }
        self.clear();
    }

    fn name(&self) -> &str {
        "MixerNode"
    }
}

// ── AudioContext ─────────────────────────────────────────────────

/// Manages an audio processing graph and renders frames.
pub struct AudioContext {
    sample_rate: f32,
    channel_count: usize,
    current_time: f64,
    nodes: Vec<(u64, Box<dyn AudioNode>)>,
    connections: Vec<(u64, u64)>, // (source, dest)
    next_id: u64,
}

impl AudioContext {
    pub fn new(sample_rate: f32, channel_count: usize) -> Self {
        Self {
            sample_rate,
            channel_count,
            current_time: 0.0,
            nodes: Vec::new(),
            connections: Vec::new(),
            next_id: 1,
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Add a node, returning its ID.
    pub fn add_node(&mut self, node: Box<dyn AudioNode>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push((id, node));
        id
    }

    /// Connect source node output to destination node input.
    pub fn connect(&mut self, source: u64, dest: u64) {
        self.connections.push((source, dest));
    }

    /// Disconnect a source from a destination.
    pub fn disconnect(&mut self, source: u64, dest: u64) {
        self.connections.retain(|&(s, d)| !(s == source && d == dest));
    }

    /// Remove a node and all its connections.
    pub fn remove_node(&mut self, id: u64) {
        self.nodes.retain(|(nid, _)| *nid != id);
        self.connections.retain(|&(s, d)| s != id && d != id);
    }

    /// Render `frames` samples by processing the graph in connection order.
    /// Returns the output of the last node in the chain.
    pub fn render(&mut self, frames: usize) -> AudioBuffer {
        let mut buffers: HashMap<u64, AudioBuffer> = HashMap::new();

        // Initialize silent buffers for all nodes
        for (id, _) in &self.nodes {
            buffers.insert(
                *id,
                AudioBuffer::new(self.channel_count, frames, self.sample_rate),
            );
        }

        // Build processing order: nodes in insertion order
        let node_ids: Vec<u64> = self.nodes.iter().map(|(id, _)| *id).collect();

        // Process each node
        for nid in &node_ids {
            // Gather inputs from connected sources
            let mut input = AudioBuffer::new(self.channel_count, frames, self.sample_rate);
            for &(src, dst) in &self.connections {
                if dst == *nid {
                    if let Some(src_buf) = buffers.get(&src) {
                        input.mix_in(src_buf);
                    }
                }
            }

            // Find and process the node
            let node_idx = self.nodes.iter().position(|(id, _)| *id == *nid);
            if let Some(idx) = node_idx {
                let mut output =
                    AudioBuffer::new(self.channel_count, frames, self.sample_rate);
                self.nodes[idx].1.process(&input, &mut output, frames);
                buffers.insert(*nid, output);
            }
        }

        // Return the last node's output
        if let Some(last_id) = node_ids.last() {
            buffers.remove(last_id).unwrap_or_else(|| {
                AudioBuffer::new(self.channel_count, frames, self.sample_rate)
            })
        } else {
            AudioBuffer::new(self.channel_count, frames, self.sample_rate)
        }
    }

    /// Advance the context clock by the given number of frames.
    pub fn advance_time(&mut self, frames: usize) {
        self.current_time += frames as f64 / self.sample_rate as f64;
    }

    /// Render and advance time.
    pub fn render_and_advance(&mut self, frames: usize) -> AudioBuffer {
        let buf = self.render(frames);
        self.advance_time(frames);
        buf
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_buffer_creation() {
        let buf = AudioBuffer::new(2, 1024, 44100.0);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.length(), 1024);
        assert_eq!(buf.sample_rate(), 44100.0);
        assert!((buf.duration() - 1024.0 / 44100.0).abs() < 1e-10);
    }

    #[test]
    fn audio_buffer_from_channels() {
        let ch0 = vec![1.0, 2.0, 3.0];
        let ch1 = vec![4.0, 5.0, 6.0];
        let buf = AudioBuffer::from_channels(vec![ch0, ch1], 48000.0);
        assert_eq!(buf.num_channels(), 2);
        assert_eq!(buf.length(), 3);
        assert_eq!(buf.get_channel_data(0).unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn audio_buffer_mix_in() {
        let mut a = AudioBuffer::from_channels(vec![vec![1.0, 2.0]], 44100.0);
        let b = AudioBuffer::from_channels(vec![vec![0.5, 0.5]], 44100.0);
        a.mix_in(&b);
        assert_eq!(a.get_channel_data(0).unwrap(), &[1.5, 2.5]);
    }

    #[test]
    fn audio_buffer_copy_channel() {
        let mut buf = AudioBuffer::from_channels(
            vec![vec![1.0, 2.0], vec![0.0, 0.0]],
            44100.0,
        );
        buf.copy_channel(0, 1);
        assert_eq!(buf.get_channel_data(1).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn gain_node_amplifies() {
        let input = AudioBuffer::from_channels(vec![vec![0.5, 1.0, -0.5]], 44100.0);
        let mut output = AudioBuffer::new(1, 3, 44100.0);
        let mut gain = GainNode::new(2.0);
        gain.process(&input, &mut output, 3);
        assert_eq!(output.get_channel_data(0).unwrap(), &[1.0, 2.0, -1.0]);
    }

    #[test]
    fn gain_node_silence() {
        let input = AudioBuffer::from_channels(vec![vec![0.5, 1.0]], 44100.0);
        let mut output = AudioBuffer::new(1, 2, 44100.0);
        let mut gain = GainNode::new(0.0);
        gain.process(&input, &mut output, 2);
        assert_eq!(output.get_channel_data(0).unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn mixer_node_sums() {
        let a = AudioBuffer::from_channels(vec![vec![1.0, 0.5]], 44100.0);
        let b = AudioBuffer::from_channels(vec![vec![0.5, 0.5]], 44100.0);
        let mut mixer = MixerNode::new(1, 2);
        mixer.add_input(&a);
        mixer.add_input(&b);
        let mut output = AudioBuffer::new(1, 2, 44100.0);
        let input = AudioBuffer::new(1, 2, 44100.0);
        mixer.process(&input, &mut output, 2);
        assert_eq!(output.get_channel_data(0).unwrap(), &[1.5, 1.0]);
    }

    #[test]
    fn context_creation() {
        let ctx = AudioContext::new(48000.0, 2);
        assert_eq!(ctx.sample_rate(), 48000.0);
        assert_eq!(ctx.channel_count(), 2);
        assert_eq!(ctx.current_time(), 0.0);
    }

    #[test]
    fn context_add_remove_nodes() {
        let mut ctx = AudioContext::new(44100.0, 1);
        let id1 = ctx.add_node(Box::new(GainNode::new(1.0)));
        let id2 = ctx.add_node(Box::new(GainNode::new(0.5)));
        assert_eq!(ctx.node_count(), 2);
        ctx.connect(id1, id2);
        assert_eq!(ctx.connection_count(), 1);
        ctx.remove_node(id1);
        assert_eq!(ctx.node_count(), 1);
        assert_eq!(ctx.connection_count(), 0);
    }

    #[test]
    fn context_render_gain_chain() {
        let mut ctx = AudioContext::new(44100.0, 1);
        // We need a source node that outputs a constant signal.
        struct ConstNode(f32);
        impl AudioNode for ConstNode {
            fn process(&mut self, _input: &AudioBuffer, output: &mut AudioBuffer, frames: usize) {
                if let Some(ch) = output.get_channel_data_mut(0) {
                    for i in 0..frames.min(ch.len()) {
                        ch[i] = self.0;
                    }
                }
            }
            fn name(&self) -> &str { "ConstNode" }
        }

        let src = ctx.add_node(Box::new(ConstNode(1.0)));
        let gain_id = ctx.add_node(Box::new(GainNode::new(0.5)));
        ctx.connect(src, gain_id);

        let output = ctx.render(4);
        let data = output.get_channel_data(0).unwrap();
        assert_eq!(data[0], 0.5);
        assert_eq!(data[3], 0.5);
    }

    #[test]
    fn context_advance_time() {
        let mut ctx = AudioContext::new(44100.0, 1);
        ctx.advance_time(44100);
        assert!((ctx.current_time() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn context_disconnect() {
        let mut ctx = AudioContext::new(44100.0, 1);
        let a = ctx.add_node(Box::new(GainNode::new(1.0)));
        let b = ctx.add_node(Box::new(GainNode::new(1.0)));
        ctx.connect(a, b);
        assert_eq!(ctx.connection_count(), 1);
        ctx.disconnect(a, b);
        assert_eq!(ctx.connection_count(), 0);
    }

    #[test]
    fn gain_node_multichannel() {
        let input = AudioBuffer::from_channels(
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            44100.0,
        );
        let mut output = AudioBuffer::new(2, 2, 44100.0);
        let mut gain = GainNode::new(0.5);
        gain.process(&input, &mut output, 2);
        assert_eq!(output.get_channel_data(0).unwrap(), &[0.5, 1.0]);
        assert_eq!(output.get_channel_data(1).unwrap(), &[1.5, 2.0]);
    }
}
