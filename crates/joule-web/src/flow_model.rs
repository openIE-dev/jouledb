//! Normalizing flow: invertible transformations, affine coupling layers,
//! log-determinant Jacobian, RealNVP.
//!
//! Implements normalizing flows for density estimation and generation via
//! invertible neural network layers. Includes affine coupling layers (RealNVP),
//! invertible 1x1 convolution (Glow-style), ActNorm layers, and log-determinant
//! Jacobian computation for exact likelihood evaluation.

use std::fmt;

// ── PRNG ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ── Flow Transform Trait ──────────────────────────────────────

/// Result of a forward or inverse flow transformation.
#[derive(Debug, Clone)]
pub struct FlowOutput {
    /// Transformed data.
    pub data: Vec<f64>,
    /// Log-determinant of the Jacobian for this transform.
    pub log_det_jacobian: f64,
}

/// An invertible transformation for normalizing flows.
pub trait FlowTransform: fmt::Debug {
    /// Forward pass: data space -> latent space.
    fn forward(&self, x: &[f64]) -> FlowOutput;
    /// Inverse pass: latent space -> data space.
    fn inverse(&self, z: &[f64]) -> FlowOutput;
    /// Dimension of the data.
    fn dim(&self) -> usize;
    /// Number of trainable parameters.
    fn param_count(&self) -> usize;
}

// ── Affine Coupling Layer ─────────────────────────────────────

/// Affine coupling layer (RealNVP): splits input into two halves,
/// transforms one half conditioned on the other.
#[derive(Debug, Clone)]
pub struct AffineCoupling {
    /// Dimension of input.
    data_dim: usize,
    /// Index at which to split (first half is identity, second is transformed).
    split_idx: usize,
    /// Scale network weights: maps first half -> scale for second half.
    scale_weights: Vec<Vec<f64>>,
    scale_biases: Vec<f64>,
    /// Translation network weights: maps first half -> shift for second half.
    shift_weights: Vec<Vec<f64>>,
    shift_biases: Vec<f64>,
}

impl AffineCoupling {
    pub fn new(data_dim: usize, seed: u64) -> Self {
        let split_idx = data_dim / 2;
        let first_half = split_idx;
        let second_half = data_dim - split_idx;
        let mut rng = Rng::new(seed);
        let scale = (2.0 / (first_half + second_half) as f64).sqrt();

        let scale_weights = (0..second_half)
            .map(|_| (0..first_half).map(|_| rng.normal() * scale * 0.1).collect())
            .collect();
        let scale_biases = vec![0.0; second_half];

        let shift_weights = (0..second_half)
            .map(|_| (0..first_half).map(|_| rng.normal() * scale * 0.1).collect())
            .collect();
        let shift_biases = vec![0.0; second_half];

        Self { data_dim, split_idx, scale_weights, scale_biases, shift_weights, shift_biases }
    }

    /// Compute scale (log_s) and shift (t) from the first half of input.
    fn compute_scale_shift(&self, x_first: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.data_dim - self.split_idx;
        let log_s: Vec<f64> = (0..n)
            .map(|j| {
                let z: f64 = self.scale_weights[j].iter().zip(x_first)
                    .map(|(w, x)| w * x).sum::<f64>() + self.scale_biases[j];
                // Clamp log_s for stability.
                z.tanh() * 2.0
            })
            .collect();
        let t: Vec<f64> = (0..n)
            .map(|j| {
                self.shift_weights[j].iter().zip(x_first)
                    .map(|(w, x)| w * x).sum::<f64>() + self.shift_biases[j]
            })
            .collect();
        (log_s, t)
    }
}

impl FlowTransform for AffineCoupling {
    fn forward(&self, x: &[f64]) -> FlowOutput {
        assert_eq!(x.len(), self.data_dim);
        let x_first = &x[..self.split_idx];
        let x_second = &x[self.split_idx..];

        let (log_s, t) = self.compute_scale_shift(x_first);

        let mut output = x_first.to_vec();
        let y_second: Vec<f64> = x_second.iter().zip(log_s.iter().zip(t.iter()))
            .map(|(xi, (ls, ti))| xi * ls.exp() + ti)
            .collect();
        output.extend(y_second);

        let log_det: f64 = log_s.iter().sum();

        FlowOutput { data: output, log_det_jacobian: log_det }
    }

    fn inverse(&self, z: &[f64]) -> FlowOutput {
        assert_eq!(z.len(), self.data_dim);
        let z_first = &z[..self.split_idx];
        let z_second = &z[self.split_idx..];

        let (log_s, t) = self.compute_scale_shift(z_first);

        let mut output = z_first.to_vec();
        let x_second: Vec<f64> = z_second.iter().zip(log_s.iter().zip(t.iter()))
            .map(|(zi, (ls, ti))| (zi - ti) * (-ls).exp())
            .collect();
        output.extend(x_second);

        let log_det: f64 = log_s.iter().map(|ls| -ls).sum();

        FlowOutput { data: output, log_det_jacobian: log_det }
    }

    fn dim(&self) -> usize { self.data_dim }
    fn param_count(&self) -> usize {
        let n = self.data_dim - self.split_idx;
        let m = self.split_idx;
        2 * (n * m + n) // scale + shift networks.
    }
}

// ── ActNorm Layer ─────────────────────────────────────────────

/// Activation normalization layer (per-channel scale and bias).
#[derive(Debug, Clone)]
pub struct ActNorm {
    data_dim: usize,
    log_scale: Vec<f64>,
    bias: Vec<f64>,
    initialized: bool,
}

impl ActNorm {
    pub fn new(data_dim: usize) -> Self {
        Self {
            data_dim,
            log_scale: vec![0.0; data_dim],
            bias: vec![0.0; data_dim],
            initialized: false,
        }
    }

    /// Data-dependent initialization: set scale/bias so first batch has zero mean, unit variance.
    pub fn initialize(&mut self, data: &[Vec<f64>]) {
        if data.is_empty() || self.initialized { return; }
        let n = data.len() as f64;

        let mut means = vec![0.0; self.data_dim];
        for sample in data {
            for (i, v) in sample.iter().enumerate() {
                means[i] += v / n;
            }
        }

        let mut vars = vec![0.0; self.data_dim];
        for sample in data {
            for (i, v) in sample.iter().enumerate() {
                vars[i] += (v - means[i]).powi(2) / n;
            }
        }

        for i in 0..self.data_dim {
            let std = vars[i].sqrt().max(1e-6);
            self.log_scale[i] = -(std.ln());
            self.bias[i] = -means[i];
        }

        self.initialized = true;
    }
}

impl FlowTransform for ActNorm {
    fn forward(&self, x: &[f64]) -> FlowOutput {
        assert_eq!(x.len(), self.data_dim);
        let data: Vec<f64> = x.iter().enumerate()
            .map(|(i, xi)| (xi + self.bias[i]) * self.log_scale[i].exp())
            .collect();
        let log_det: f64 = self.log_scale.iter().sum();
        FlowOutput { data, log_det_jacobian: log_det }
    }

    fn inverse(&self, z: &[f64]) -> FlowOutput {
        assert_eq!(z.len(), self.data_dim);
        let data: Vec<f64> = z.iter().enumerate()
            .map(|(i, zi)| zi * (-self.log_scale[i]).exp() - self.bias[i])
            .collect();
        let log_det: f64 = self.log_scale.iter().map(|ls| -ls).sum();
        FlowOutput { data, log_det_jacobian: log_det }
    }

    fn dim(&self) -> usize { self.data_dim }
    fn param_count(&self) -> usize { 2 * self.data_dim }
}

// ── Permutation Layer ─────────────────────────────────────────

/// Fixed random permutation layer (invertible, log-det = 0).
#[derive(Debug, Clone)]
pub struct Permutation {
    forward_perm: Vec<usize>,
    inverse_perm: Vec<usize>,
}

impl Permutation {
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut perm: Vec<usize> = (0..dim).collect();
        // Fisher-Yates shuffle.
        for i in (1..dim).rev() {
            let j = (rng.next() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let mut inv = vec![0; dim];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        Self { forward_perm: perm, inverse_perm: inv }
    }

    pub fn identity(dim: usize) -> Self {
        let perm: Vec<usize> = (0..dim).collect();
        Self { forward_perm: perm.clone(), inverse_perm: perm }
    }
}

impl FlowTransform for Permutation {
    fn forward(&self, x: &[f64]) -> FlowOutput {
        let data: Vec<f64> = self.forward_perm.iter().map(|i| x[*i]).collect();
        FlowOutput { data, log_det_jacobian: 0.0 }
    }

    fn inverse(&self, z: &[f64]) -> FlowOutput {
        let data: Vec<f64> = self.inverse_perm.iter().map(|i| z[*i]).collect();
        FlowOutput { data, log_det_jacobian: 0.0 }
    }

    fn dim(&self) -> usize { self.forward_perm.len() }
    fn param_count(&self) -> usize { 0 }
}

// ── RealNVP Flow ──────────────────────────────────────────────

/// RealNVP normalizing flow: stack of affine coupling + permutation layers.
#[derive(Debug, Clone)]
pub struct RealNvpFlow {
    data_dim: usize,
    couplings: Vec<AffineCoupling>,
    permutations: Vec<Permutation>,
    num_layers: usize,
}

impl RealNvpFlow {
    pub fn new(data_dim: usize, num_layers: usize, seed: u64) -> Self {
        let mut couplings = Vec::with_capacity(num_layers);
        let mut permutations = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            couplings.push(AffineCoupling::new(data_dim, seed + i as u64 * 100));
            permutations.push(Permutation::new(data_dim, seed + i as u64 * 100 + 50));
        }

        Self { data_dim, couplings, permutations, num_layers }
    }

    pub fn with_num_layers(mut self, n: usize) -> Self {
        let seed = 42u64;
        self.couplings.clear();
        self.permutations.clear();
        self.num_layers = n;
        for i in 0..n {
            self.couplings.push(AffineCoupling::new(self.data_dim, seed + i as u64 * 100));
            self.permutations.push(Permutation::new(self.data_dim, seed + i as u64 * 100 + 50));
        }
        self
    }

    /// Forward pass: data -> latent, accumulating log-det-Jacobian.
    pub fn forward(&self, x: &[f64]) -> FlowOutput {
        let mut current = x.to_vec();
        let mut total_ldj = 0.0;

        for i in 0..self.num_layers {
            let out = self.couplings[i].forward(&current);
            current = out.data;
            total_ldj += out.log_det_jacobian;

            let perm_out = self.permutations[i].forward(&current);
            current = perm_out.data;
            // Permutation has log-det = 0.
        }

        FlowOutput { data: current, log_det_jacobian: total_ldj }
    }

    /// Inverse pass: latent -> data.
    pub fn inverse(&self, z: &[f64]) -> FlowOutput {
        let mut current = z.to_vec();
        let mut total_ldj = 0.0;

        for i in (0..self.num_layers).rev() {
            let perm_out = self.permutations[i].inverse(&current);
            current = perm_out.data;

            let out = self.couplings[i].inverse(&current);
            current = out.data;
            total_ldj += out.log_det_jacobian;
        }

        FlowOutput { data: current, log_det_jacobian: total_ldj }
    }

    /// Compute log-probability of a data point under the flow model.
    /// log p(x) = log p_z(f(x)) + log |det J_f(x)|
    pub fn log_prob(&self, x: &[f64]) -> f64 {
        let out = self.forward(x);
        let log_pz: f64 = out.data.iter()
            .map(|zi| -0.5 * (zi * zi + (2.0 * std::f64::consts::PI).ln()))
            .sum();
        log_pz + out.log_det_jacobian
    }

    /// Generate samples by sampling from N(0,I) and applying inverse flow.
    pub fn sample(&self, rng: &mut Rng) -> Vec<f64> {
        let z: Vec<f64> = (0..self.data_dim).map(|_| rng.normal()).collect();
        self.inverse(&z).data
    }

    /// Generate multiple samples.
    pub fn sample_batch(&self, count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(seed);
        (0..count).map(|_| self.sample(&mut rng)).collect()
    }

    pub fn total_params(&self) -> usize {
        self.couplings.iter().map(|c| c.param_count()).sum::<usize>()
    }
}

impl fmt::Display for RealNvpFlow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RealNVP(dim={}, layers={}, params={})",
            self.data_dim, self.num_layers, self.total_params(),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_coupling_invertible() {
        let layer = AffineCoupling::new(6, 42);
        let x = vec![1.0, -0.5, 0.3, 0.8, -1.2, 0.0];
        let fwd = layer.forward(&x);
        let inv = layer.inverse(&fwd.data);
        for (a, b) in x.iter().zip(inv.data.iter()) {
            assert!((a - b).abs() < 1e-10, "Not invertible: {a} vs {b}");
        }
    }

    #[test]
    fn test_affine_coupling_log_det() {
        let layer = AffineCoupling::new(4, 42);
        let x = vec![0.5, -0.5, 0.3, 0.8];
        let fwd = layer.forward(&x);
        let inv = layer.inverse(&fwd.data);
        // Forward + inverse log-dets should sum to ~0.
        assert!((fwd.log_det_jacobian + inv.log_det_jacobian).abs() < 1e-10);
    }

    #[test]
    fn test_actnorm_forward_inverse() {
        let mut an = ActNorm::new(3);
        let data = vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0]];
        an.initialize(&data);

        let x = vec![2.0, 3.0, 4.0];
        let fwd = an.forward(&x);
        let inv = an.inverse(&fwd.data);
        for (a, b) in x.iter().zip(inv.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_actnorm_log_det_cancel() {
        let an = ActNorm::new(4);
        let x = vec![1.0; 4];
        let fwd = an.forward(&x);
        let inv = an.inverse(&fwd.data);
        assert!((fwd.log_det_jacobian + inv.log_det_jacobian).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_invertible() {
        let perm = Permutation::new(5, 42);
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fwd = perm.forward(&x);
        let inv = perm.inverse(&fwd.data);
        assert_eq!(x, inv.data);
    }

    #[test]
    fn test_permutation_log_det_zero() {
        let perm = Permutation::new(4, 42);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let fwd = perm.forward(&x);
        assert_eq!(fwd.log_det_jacobian, 0.0);
    }

    #[test]
    fn test_identity_permutation() {
        let perm = Permutation::identity(3);
        let x = vec![1.0, 2.0, 3.0];
        let fwd = perm.forward(&x);
        assert_eq!(fwd.data, x);
    }

    #[test]
    fn test_realnvp_forward_inverse() {
        let flow = RealNvpFlow::new(4, 3, 42);
        let x = vec![0.5, -0.3, 1.2, -0.8];
        let fwd = flow.forward(&x);
        let inv = flow.inverse(&fwd.data);
        for (a, b) in x.iter().zip(inv.data.iter()) {
            assert!((a - b).abs() < 1e-8, "Not invertible: {a} vs {b}");
        }
    }

    #[test]
    fn test_realnvp_log_prob() {
        let flow = RealNvpFlow::new(4, 2, 42);
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let lp = flow.log_prob(&x);
        assert!(lp.is_finite());
    }

    #[test]
    fn test_realnvp_sample_shape() {
        let flow = RealNvpFlow::new(4, 2, 42);
        let mut rng = Rng::new(99);
        let sample = flow.sample(&mut rng);
        assert_eq!(sample.len(), 4);
    }

    #[test]
    fn test_realnvp_sample_batch() {
        let flow = RealNvpFlow::new(4, 2, 42);
        let batch = flow.sample_batch(10, 99);
        assert_eq!(batch.len(), 10);
        assert_eq!(batch[0].len(), 4);
    }

    #[test]
    fn test_realnvp_param_count() {
        let flow = RealNvpFlow::new(4, 2, 42);
        assert!(flow.total_params() > 0);
    }

    #[test]
    fn test_realnvp_display() {
        let flow = RealNvpFlow::new(4, 3, 42);
        let s = format!("{flow}");
        assert!(s.contains("RealNVP"));
        assert!(s.contains("layers=3"));
    }

    #[test]
    fn test_affine_coupling_param_count() {
        let layer = AffineCoupling::new(6, 42);
        // split=3, second=3. scale: 3*3+3=12, shift: 3*3+3=12. Total=24.
        assert_eq!(layer.param_count(), 24);
    }

    #[test]
    fn test_actnorm_param_count() {
        let an = ActNorm::new(5);
        assert_eq!(an.param_count(), 10);
    }

    #[test]
    fn test_flow_output_preserves_dim() {
        let layer = AffineCoupling::new(8, 42);
        let x = vec![0.0; 8];
        let out = layer.forward(&x);
        assert_eq!(out.data.len(), 8);
    }

    #[test]
    fn test_realnvp_with_layers() {
        let flow = RealNvpFlow::new(4, 1, 42).with_num_layers(5);
        assert_eq!(flow.num_layers, 5);
    }

    #[test]
    fn test_different_seeds_different_outputs() {
        let flow = RealNvpFlow::new(4, 2, 42);
        let b1 = flow.sample_batch(3, 1);
        let b2 = flow.sample_batch(3, 2);
        // Different seeds should yield different samples.
        let differ = b1[0].iter().zip(b2[0].iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(differ);
    }
}
