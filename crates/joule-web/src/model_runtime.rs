//! Model inference runtime with layer execution, memory management, and
//! input/output binding.
//!
//! Provides a lightweight runtime for executing neural network models
//! layer-by-layer without external framework dependencies. Manages
//! activation buffers, supports in-place operations, and tracks per-layer
//! latency for profiling.

use std::collections::HashMap;
use std::fmt;

// ── Tensor ─────────────────────────────────────────────────────

/// Multi-dimensional tensor backed by a flat `Vec<f64>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self { shape: shape.to_vec(), data: vec![0.0; len] }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f64>) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "shape/data mismatch");
        Self { shape: shape.to_vec(), data }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Element-wise add; shapes must match.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Tensor { shape: self.shape.clone(), data }
    }

    /// Element-wise multiply.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Tensor { shape: self.shape.clone(), data }
    }

    /// Matrix multiply for 2-D tensors [M,K] x [K,N] → [M,N].
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(other.shape[0], k);
        let n = other.shape[1];
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Tensor { shape: vec![m, n], data: out }
    }

    /// Apply ReLU in-place.
    pub fn relu_inplace(&mut self) {
        for v in &mut self.data {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
    }

    /// Softmax across last dimension.
    pub fn softmax(&self) -> Tensor {
        let last = *self.shape.last().unwrap_or(&1);
        let batches = self.data.len() / last;
        let mut out = self.data.clone();
        for b in 0..batches {
            let start = b * last;
            let end = start + last;
            let max_val = out[start..end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for v in &mut out[start..end] {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in &mut out[start..end] {
                    *v /= sum;
                }
            }
        }
        Tensor { shape: self.shape.clone(), data: out }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, numel={})", self.shape, self.numel())
    }
}

// ── Layer ──────────────────────────────────────────────────────

/// Supported layer operations.
#[derive(Debug, Clone)]
pub enum LayerOp {
    /// Dense (fully-connected): weight [out, in], bias [out].
    Dense { weight: Tensor, bias: Tensor },
    /// ReLU activation.
    Relu,
    /// Sigmoid activation.
    Sigmoid,
    /// Softmax across last axis.
    Softmax,
    /// Batch normalisation: gamma, beta, running_mean, running_var, epsilon.
    BatchNorm {
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        epsilon: f64,
    },
    /// Dropout (kept as pass-through at inference).
    Dropout { rate: f64 },
    /// Reshape to target shape (-1 allowed for one dim).
    Reshape { target: Vec<i64> },
}

/// A named layer in the model graph.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub op: LayerOp,
}

impl Layer {
    pub fn new(name: impl Into<String>, op: LayerOp) -> Self {
        Self { name: name.into(), op }
    }

    /// Execute this layer on the input tensor.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match &self.op {
            LayerOp::Dense { weight, bias } => {
                // input: [batch, in_features], weight: [out, in]
                let out = input.matmul(&transpose_2d(weight));
                broadcast_add(&out, bias)
            }
            LayerOp::Relu => {
                let mut out = input.clone();
                out.relu_inplace();
                out
            }
            LayerOp::Sigmoid => {
                let data: Vec<f64> = input.data.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
                Tensor { shape: input.shape.clone(), data }
            }
            LayerOp::Softmax => input.softmax(),
            LayerOp::BatchNorm { gamma, beta, running_mean, running_var, epsilon } => {
                let data: Vec<f64> = input
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let c = i % gamma.numel();
                        let normed = (x - running_mean.data[c])
                            / (running_var.data[c] + epsilon).sqrt();
                        gamma.data[c] * normed + beta.data[c]
                    })
                    .collect();
                Tensor { shape: input.shape.clone(), data }
            }
            LayerOp::Dropout { .. } => input.clone(), // pass-through at inference
            LayerOp::Reshape { target } => {
                let total = input.numel();
                let mut new_shape: Vec<usize> = Vec::new();
                let mut infer_idx = None;
                let mut known_product: usize = 1;
                for (i, &d) in target.iter().enumerate() {
                    if d < 0 {
                        infer_idx = Some(i);
                        new_shape.push(0);
                    } else {
                        new_shape.push(d as usize);
                        known_product *= d as usize;
                    }
                }
                if let Some(idx) = infer_idx {
                    new_shape[idx] = total / known_product;
                }
                Tensor { shape: new_shape, data: input.data.clone() }
            }
        }
    }
}

fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2);
    let (r, c) = (t.shape[0], t.shape[1]);
    let mut data = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            data[j * r + i] = t.data[i * c + j];
        }
    }
    Tensor { shape: vec![c, r], data }
}

fn broadcast_add(mat: &Tensor, bias: &Tensor) -> Tensor {
    let cols = *mat.shape.last().unwrap();
    let data: Vec<f64> = mat
        .data
        .iter()
        .enumerate()
        .map(|(i, v)| v + bias.data[i % cols])
        .collect();
    Tensor { shape: mat.shape.clone(), data }
}

// ── Binding ────────────────────────────────────────────────────

/// Describes an input or output binding by name and expected shape.
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub shape: Vec<usize>,
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({:?})", self.name, self.shape)
    }
}

// ── LayerProfile ───────────────────────────────────────────────

/// Per-layer profiling data collected during inference.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    pub layer_name: String,
    pub elapsed_us: u64,
    pub output_numel: usize,
}

// ── MemoryBudget ───────────────────────────────────────────────

/// Memory budget enforcement for activation buffers.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub max_bytes: usize,
    pub current_bytes: usize,
}

impl MemoryBudget {
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes, current_bytes: 0 }
    }

    pub fn can_allocate(&self, tensor: &Tensor) -> bool {
        let needed = tensor.numel() * 8; // f64 = 8 bytes
        self.current_bytes + needed <= self.max_bytes
    }

    pub fn allocate(&mut self, tensor: &Tensor) -> bool {
        let needed = tensor.numel() * 8;
        if self.current_bytes + needed > self.max_bytes {
            return false;
        }
        self.current_bytes += needed;
        true
    }

    pub fn release(&mut self, numel: usize) {
        let freed = numel * 8;
        self.current_bytes = self.current_bytes.saturating_sub(freed);
    }

    pub fn utilisation(&self) -> f64 {
        if self.max_bytes == 0 {
            return 0.0;
        }
        self.current_bytes as f64 / self.max_bytes as f64
    }
}

impl fmt::Display for MemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryBudget({}/{} bytes, {:.1}%)",
            self.current_bytes,
            self.max_bytes,
            self.utilisation() * 100.0
        )
    }
}

// ── ModelRuntime ───────────────────────────────────────────────

/// Runtime that executes a sequential model layer-by-layer.
#[derive(Debug)]
pub struct ModelRuntime {
    pub name: String,
    layers: Vec<Layer>,
    inputs: Vec<Binding>,
    outputs: Vec<Binding>,
    memory_budget: Option<MemoryBudget>,
    profiles: Vec<LayerProfile>,
    metadata: HashMap<String, String>,
}

impl ModelRuntime {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layers: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            memory_budget: None,
            profiles: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn with_input(mut self, binding: Binding) -> Self {
        self.inputs.push(binding);
        self
    }

    pub fn with_output(mut self, binding: Binding) -> Self {
        self.outputs.push(binding);
        self
    }

    pub fn with_memory_budget(mut self, max_bytes: usize) -> Self {
        self.memory_budget = Some(MemoryBudget::new(max_bytes));
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total parameter count across all Dense layers.
    pub fn param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| match &l.op {
                LayerOp::Dense { weight, bias } => weight.numel() + bias.numel(),
                LayerOp::BatchNorm { gamma, beta, .. } => gamma.numel() + beta.numel(),
                _ => 0,
            })
            .sum()
    }

    /// Run forward inference. Returns the final output tensor.
    pub fn infer(&mut self, input: &Tensor) -> Result<Tensor, String> {
        self.profiles.clear();
        let mut current = input.clone();

        if let Some(ref mut budget) = self.memory_budget {
            budget.current_bytes = 0;
            if !budget.allocate(&current) {
                return Err("input tensor exceeds memory budget".into());
            }
        }

        for layer in &self.layers {
            let start = std::time::Instant::now();
            let output = layer.forward(&current);

            if let Some(ref mut budget) = self.memory_budget {
                budget.release(current.numel());
                if !budget.allocate(&output) {
                    return Err(format!(
                        "layer '{}' output exceeds memory budget",
                        layer.name
                    ));
                }
            }

            self.profiles.push(LayerProfile {
                layer_name: layer.name.clone(),
                elapsed_us: start.elapsed().as_micros() as u64,
                output_numel: output.numel(),
            });
            current = output;
        }
        Ok(current)
    }

    /// Return profiling data from the last inference.
    pub fn last_profiles(&self) -> &[LayerProfile] {
        &self.profiles
    }

    /// Total inference time (microseconds) from last run.
    pub fn total_us(&self) -> u64 {
        self.profiles.iter().map(|p| p.elapsed_us).sum()
    }
}

impl fmt::Display for ModelRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelRuntime('{}', layers={}, params={})",
            self.name,
            self.num_layers(),
            self.param_count()
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_layer(name: &str, in_f: usize, out_f: usize) -> Layer {
        let weight = Tensor::from_vec(&[out_f, in_f], vec![0.1; out_f * in_f]);
        let bias = Tensor::from_vec(&[out_f], vec![0.01; out_f]);
        Layer::new(name, LayerOp::Dense { weight, bias })
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let b = Tensor::from_vec(&[3], vec![4.0, 5.0, 6.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_mul() {
        let a = Tensor::from_vec(&[2], vec![3.0, 4.0]);
        let b = Tensor::from_vec(&[2], vec![2.0, 0.5]);
        let c = a.mul(&b);
        assert_eq!(c.data, vec![6.0, 2.0]);
    }

    #[test]
    fn test_matmul_identity() {
        let a = Tensor::from_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let b = Tensor::from_vec(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_relu_inplace() {
        let mut t = Tensor::from_vec(&[4], vec![-1.0, 0.0, 1.0, -0.5]);
        t.relu_inplace();
        assert_eq!(t.data, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let t = Tensor::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        let s = t.softmax();
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_ordering() {
        let t = Tensor::from_vec(&[3], vec![1.0, 3.0, 2.0]);
        let s = t.softmax();
        assert!(s.data[1] > s.data[2]);
        assert!(s.data[2] > s.data[0]);
    }

    #[test]
    fn test_dense_layer_forward() {
        let layer = dense_layer("fc1", 3, 2);
        let input = Tensor::from_vec(&[1, 3], vec![1.0, 1.0, 1.0]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![1, 2]);
        // 0.1*1 + 0.1*1 + 0.1*1 + 0.01 = 0.31
        assert!((out.data[0] - 0.31).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_layer() {
        let layer = Layer::new("sig", LayerOp::Sigmoid);
        let input = Tensor::from_vec(&[3], vec![0.0, 100.0, -100.0]);
        let out = layer.forward(&input);
        assert!((out.data[0] - 0.5).abs() < 1e-10);
        assert!(out.data[1] > 0.99);
        assert!(out.data[2] < 0.01);
    }

    #[test]
    fn test_dropout_passthrough() {
        let layer = Layer::new("drop", LayerOp::Dropout { rate: 0.5 });
        let input = Tensor::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let out = layer.forward(&input);
        assert_eq!(out.data, input.data);
    }

    #[test]
    fn test_reshape() {
        let layer = Layer::new("reshape", LayerOp::Reshape { target: vec![2, -1] });
        let input = Tensor::from_vec(&[1, 6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    fn test_model_runtime_inference() {
        let mut rt = ModelRuntime::new("test_model")
            .with_layer(dense_layer("fc1", 4, 3))
            .with_layer(Layer::new("relu", LayerOp::Relu))
            .with_layer(dense_layer("fc2", 3, 2))
            .with_layer(Layer::new("sm", LayerOp::Softmax));
        let input = Tensor::from_vec(&[1, 4], vec![1.0, 0.5, -0.3, 0.8]);
        let out = rt.infer(&input).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        let sum: f64 = out.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_param_count() {
        let rt = ModelRuntime::new("m")
            .with_layer(dense_layer("fc1", 4, 3))  // 4*3 + 3 = 15
            .with_layer(dense_layer("fc2", 3, 2)); // 3*2 + 2 = 8
        assert_eq!(rt.param_count(), 23);
    }

    #[test]
    fn test_profiling() {
        let mut rt = ModelRuntime::new("m")
            .with_layer(dense_layer("fc1", 2, 2))
            .with_layer(Layer::new("relu", LayerOp::Relu));
        let input = Tensor::from_vec(&[1, 2], vec![1.0, 2.0]);
        rt.infer(&input).unwrap();
        assert_eq!(rt.last_profiles().len(), 2);
        assert_eq!(rt.last_profiles()[0].layer_name, "fc1");
    }

    #[test]
    fn test_memory_budget_ok() {
        let mut rt = ModelRuntime::new("m")
            .with_layer(Layer::new("relu", LayerOp::Relu))
            .with_memory_budget(1024);
        let input = Tensor::from_vec(&[2], vec![1.0, -1.0]);
        let out = rt.infer(&input).unwrap();
        assert_eq!(out.data, vec![1.0, 0.0]);
    }

    #[test]
    fn test_memory_budget_exceeded() {
        let mut rt = ModelRuntime::new("m")
            .with_layer(dense_layer("big", 10, 10))
            .with_memory_budget(16); // very small
        let input = Tensor::from_vec(&[1, 10], vec![1.0; 10]);
        assert!(rt.infer(&input).is_err());
    }

    #[test]
    fn test_batch_norm() {
        let layer = Layer::new("bn", LayerOp::BatchNorm {
            gamma: Tensor::from_vec(&[2], vec![1.0, 1.0]),
            beta: Tensor::from_vec(&[2], vec![0.0, 0.0]),
            running_mean: Tensor::from_vec(&[2], vec![0.0, 0.0]),
            running_var: Tensor::from_vec(&[2], vec![1.0, 1.0]),
            epsilon: 1e-5,
        });
        let input = Tensor::from_vec(&[1, 2], vec![2.0, -3.0]);
        let out = layer.forward(&input);
        assert!((out.data[0] - 2.0).abs() < 1e-3);
        assert!((out.data[1] - (-3.0)).abs() < 1e-3);
    }

    #[test]
    fn test_display_impls() {
        let t = Tensor::zeros(&[2, 3]);
        assert!(format!("{t}").contains("shape=[2, 3]"));

        let b = Binding { name: "input".into(), shape: vec![1, 3] };
        assert!(format!("{b}").contains("input"));

        let mb = MemoryBudget::new(1000);
        assert!(format!("{mb}").contains("1000"));

        let rt = ModelRuntime::new("m");
        assert!(format!("{rt}").contains("ModelRuntime"));
    }

    #[test]
    fn test_metadata() {
        let rt = ModelRuntime::new("m")
            .with_metadata("version", "1.0")
            .with_metadata("author", "test");
        assert_eq!(rt.metadata.get("version").unwrap(), "1.0");
    }

    #[test]
    fn test_memory_budget_utilisation() {
        let mut mb = MemoryBudget::new(100);
        assert_eq!(mb.utilisation(), 0.0);
        let t = Tensor::zeros(&[5]); // 5 * 8 = 40 bytes
        mb.allocate(&t);
        assert!((mb.utilisation() - 0.4).abs() < 1e-10);
    }
}
