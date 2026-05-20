//! Activation functions and their derivatives for neural networks.
//!
//! Provides element-wise nonlinearities including ReLU, LeakyReLU,
//! GELU, Sigmoid, Tanh, Swish/SiLU, Mish, and vector-level Softmax.
//! Each function includes its analytical derivative for backprop.

use std::fmt;

// ── Activation Enum ───────────────────────────────────────────────

/// Enumeration of supported activation functions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Identity,
    Relu,
    LeakyRelu(f64),
    Elu(f64),
    Gelu,
    Sigmoid,
    Tanh,
    Swish,
    Mish,
    HardSigmoid,
    HardSwish,
    Softplus,
}

impl Activation {
    /// Apply the activation to a scalar.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Self::Identity => x,
            Self::Relu => relu(x),
            Self::LeakyRelu(alpha) => leaky_relu(x, *alpha),
            Self::Elu(alpha) => elu(x, *alpha),
            Self::Gelu => gelu(x),
            Self::Sigmoid => sigmoid(x),
            Self::Tanh => tanh_act(x),
            Self::Swish => swish(x),
            Self::Mish => mish(x),
            Self::HardSigmoid => hard_sigmoid(x),
            Self::HardSwish => hard_swish(x),
            Self::Softplus => softplus(x),
        }
    }

    /// Derivative of the activation given input `x`.
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Self::Identity => 1.0,
            Self::Relu => relu_deriv(x),
            Self::LeakyRelu(alpha) => leaky_relu_deriv(x, *alpha),
            Self::Elu(alpha) => elu_deriv(x, *alpha),
            Self::Gelu => gelu_deriv(x),
            Self::Sigmoid => sigmoid_deriv(x),
            Self::Tanh => tanh_deriv(x),
            Self::Swish => swish_deriv(x),
            Self::Mish => mish_deriv(x),
            Self::HardSigmoid => hard_sigmoid_deriv(x),
            Self::HardSwish => hard_swish_deriv(x),
            Self::Softplus => softplus_deriv(x),
        }
    }

    /// Apply element-wise to a vector.
    pub fn apply_vec(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|x| self.apply(*x)).collect()
    }

    /// Derivative element-wise on a vector.
    pub fn derivative_vec(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|x| self.derivative(*x)).collect()
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity => write!(f, "identity"),
            Self::Relu => write!(f, "relu"),
            Self::LeakyRelu(a) => write!(f, "leaky_relu({})", a),
            Self::Elu(a) => write!(f, "elu({})", a),
            Self::Gelu => write!(f, "gelu"),
            Self::Sigmoid => write!(f, "sigmoid"),
            Self::Tanh => write!(f, "tanh"),
            Self::Swish => write!(f, "swish"),
            Self::Mish => write!(f, "mish"),
            Self::HardSigmoid => write!(f, "hard_sigmoid"),
            Self::HardSwish => write!(f, "hard_swish"),
            Self::Softplus => write!(f, "softplus"),
        }
    }
}

// ── ReLU ──────────────────────────────────────────────────────────

/// Rectified Linear Unit: max(0, x).
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// ── Leaky ReLU ────────────────────────────────────────────────────

/// Leaky ReLU: x if x > 0, else alpha * x.
pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * x }
}

pub fn leaky_relu_deriv(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { 1.0 } else { alpha }
}

// ── ELU ───────────────────────────────────────────────────────────

/// Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1).
pub fn elu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

pub fn elu_deriv(x: f64, alpha: f64) -> f64 {
    if x > 0.0 { 1.0 } else { alpha * x.exp() }
}

// ── GELU ──────────────────────────────────────────────────────────

/// Gaussian Error Linear Unit (approximate form using tanh).
///
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}

pub fn gelu_deriv(x: f64) -> f64 {
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let cubic = x + 0.044715 * x.powi(3);
    let inner = sqrt_2_over_pi * cubic;
    let t = inner.tanh();
    let sech2 = 1.0 - t * t;
    let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + t) + 0.5 * x * sech2 * d_inner
}

// ── Sigmoid ───────────────────────────────────────────────────────

/// Logistic sigmoid: 1 / (1 + exp(-x)).
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

pub fn sigmoid_deriv(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

// ── Tanh ──────────────────────────────────────────────────────────

/// Hyperbolic tangent activation.
pub fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

pub fn tanh_deriv(x: f64) -> f64 {
    let t = x.tanh();
    1.0 - t * t
}

// ── Swish / SiLU ──────────────────────────────────────────────────

/// Swish (SiLU): x * sigmoid(x).
pub fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

pub fn swish_deriv(x: f64) -> f64 {
    let s = sigmoid(x);
    s + x * s * (1.0 - s)
}

// ── Mish ──────────────────────────────────────────────────────────

/// Mish: x * tanh(softplus(x)).
pub fn mish(x: f64) -> f64 {
    x * softplus(x).tanh()
}

pub fn mish_deriv(x: f64) -> f64 {
    let sp = softplus(x);
    let tsp = sp.tanh();
    let omega = 4.0 * (x + 1.0) + 4.0 * x.exp().powi(2) + x.exp().powi(3)
        + x.exp() * (4.0 * x + 6.0);
    let delta = 2.0 * x.exp() + x.exp().powi(2) + 2.0;
    let delta_sq = delta * delta;
    if delta_sq.abs() < 1e-30 {
        tsp
    } else {
        tsp + x * sigmoid(x) * (1.0 - tsp * tsp)
    }
}

// ── Hard Sigmoid ──────────────────────────────────────────────────

/// Piecewise linear approximation of sigmoid.
pub fn hard_sigmoid(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        x / 6.0 + 0.5
    }
}

pub fn hard_sigmoid_deriv(x: f64) -> f64 {
    if x > -3.0 && x < 3.0 { 1.0 / 6.0 } else { 0.0 }
}

// ── Hard Swish ────────────────────────────────────────────────────

/// Hard Swish: x * hard_sigmoid(x).
pub fn hard_swish(x: f64) -> f64 {
    x * hard_sigmoid(x)
}

pub fn hard_swish_deriv(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        x / 3.0 + 0.5
    }
}

// ── Softplus ──────────────────────────────────────────────────────

/// Softplus: ln(1 + exp(x)). Smooth approximation of ReLU.
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x // Avoid overflow
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn softplus_deriv(x: f64) -> f64 {
    sigmoid(x)
}

// ── Softmax (vector) ──────────────────────────────────────────────

/// Softmax: converts logits to probabilities.
///
/// Uses the numerically stable formulation: subtract max before exp.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f64; logits.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Jacobian of softmax: d softmax_i / d logit_j.
pub fn softmax_jacobian(probs: &[f64]) -> Vec<Vec<f64>> {
    let n = probs.len();
    let mut jac = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                jac[i][j] = probs[i] * (1.0 - probs[i]);
            } else {
                jac[i][j] = -probs[i] * probs[j];
            }
        }
    }
    jac
}

// ── Log-Softmax ───────────────────────────────────────────────────

/// Log-softmax for numerical stability with NLL loss.
pub fn log_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = logits.iter().map(|x| x - max_val).collect();
    let log_sum_exp = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
    shifted.iter().map(|x| x - log_sum_exp).collect()
}

// ── SELU ──────────────────────────────────────────────────────────

const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;

/// Scaled Exponential Linear Unit (for self-normalizing networks).
pub fn selu(x: f64) -> f64 {
    if x > 0.0 {
        SELU_LAMBDA * x
    } else {
        SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0)
    }
}

pub fn selu_deriv(x: f64) -> f64 {
    if x > 0.0 {
        SELU_LAMBDA
    } else {
        SELU_LAMBDA * SELU_ALPHA * x.exp()
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_relu() {
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_deriv() {
        assert_eq!(relu_deriv(1.0), 1.0);
        assert_eq!(relu_deriv(-1.0), 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        assert_eq!(leaky_relu(5.0, 0.01), 5.0);
        assert!((leaky_relu(-2.0, 0.01) - (-0.02)).abs() < EPS);
    }

    #[test]
    fn test_elu() {
        assert_eq!(elu(1.0, 1.0), 1.0);
        let neg = elu(-1.0, 1.0);
        assert!((neg - ((-1.0_f64).exp() - 1.0)).abs() < EPS);
    }

    #[test]
    fn test_gelu_zero() {
        assert!(gelu(0.0).abs() < EPS);
    }

    #[test]
    fn test_gelu_positive() {
        let val = gelu(1.0);
        assert!(val > 0.5 && val < 1.0);
    }

    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(100.0) < 1.0 + EPS);
        assert!(sigmoid(-100.0) > -EPS);
    }

    #[test]
    fn test_sigmoid_deriv_at_zero() {
        assert!((sigmoid_deriv(0.0) - 0.25).abs() < EPS);
    }

    #[test]
    fn test_tanh_zero() {
        assert!(tanh_act(0.0).abs() < EPS);
    }

    #[test]
    fn test_tanh_deriv_at_zero() {
        assert!((tanh_deriv(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_swish_zero() {
        assert!(swish(0.0).abs() < EPS);
    }

    #[test]
    fn test_swish_positive() {
        assert!(swish(2.0) > 0.0);
    }

    #[test]
    fn test_mish() {
        assert!(mish(0.0).abs() < EPS);
        assert!(mish(2.0) > 0.0);
    }

    #[test]
    fn test_hard_sigmoid() {
        assert_eq!(hard_sigmoid(-5.0), 0.0);
        assert_eq!(hard_sigmoid(5.0), 1.0);
        assert!((hard_sigmoid(0.0) - 0.5).abs() < EPS);
    }

    #[test]
    fn test_hard_swish() {
        assert_eq!(hard_swish(-5.0), 0.0);
        assert!((hard_swish(5.0) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - (2.0_f64).ln()).abs() < EPS);
        // For large x, softplus(x) ≈ x
        assert!((softplus(30.0) - 30.0).abs() < EPS);
    }

    #[test]
    fn test_softmax_basic() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_equal_logits() {
        let probs = softmax(&[0.0, 0.0, 0.0]);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < EPS);
        }
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_large_values() {
        // Should not overflow
        let probs = softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn test_softmax_jacobian_diagonal() {
        let probs = softmax(&[1.0, 2.0]);
        let jac = softmax_jacobian(&probs);
        // Diagonal: p_i * (1 - p_i)
        assert!((jac[0][0] - probs[0] * (1.0 - probs[0])).abs() < EPS);
    }

    #[test]
    fn test_log_softmax() {
        let ls = log_softmax(&[1.0, 2.0, 3.0]);
        let probs = softmax(&[1.0, 2.0, 3.0]);
        for (lp, p) in ls.iter().zip(probs.iter()) {
            assert!((lp - p.ln()).abs() < EPS);
        }
    }

    #[test]
    fn test_selu() {
        assert!((selu(0.0)).abs() < EPS);
        assert!(selu(1.0) > 1.0); // scaled up
        assert!(selu(-1.0) < 0.0);
    }

    #[test]
    fn test_activation_enum_apply_vec() {
        let act = Activation::Relu;
        let out = act.apply_vec(&[-1.0, 0.0, 1.0, 2.0]);
        assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_activation_display() {
        assert_eq!(format!("{}", Activation::Relu), "relu");
        assert_eq!(format!("{}", Activation::Gelu), "gelu");
        assert_eq!(format!("{}", Activation::Swish), "swish");
        assert_eq!(format!("{}", Activation::LeakyRelu(0.01)), "leaky_relu(0.01)");
    }

    #[test]
    fn test_numerical_gradient_sigmoid() {
        // Verify derivative with finite differences
        let x = 0.5;
        let h = 1e-7;
        let numerical = (sigmoid(x + h) - sigmoid(x - h)) / (2.0 * h);
        let analytical = sigmoid_deriv(x);
        assert!((numerical - analytical).abs() < 1e-5);
    }

    #[test]
    fn test_numerical_gradient_gelu() {
        let x = 0.5;
        let h = 1e-7;
        let numerical = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
        let analytical = gelu_deriv(x);
        assert!((numerical - analytical).abs() < 1e-4);
    }
}
