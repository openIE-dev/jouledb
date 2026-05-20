//! Loss functions for neural network training.
//!
//! Provides differentiable loss functions that measure the discrepancy
//! between model predictions and target values. Each function computes
//! both the scalar loss and its gradient with respect to predictions:
//!
//! - [`MseLoss`] — mean squared error (L2 loss)
//! - [`CrossEntropyLoss`] — categorical cross-entropy with softmax
//! - [`BinaryCrossEntropyLoss`] — binary classification loss
//! - [`HuberLoss`] — smooth L1 loss for robust regression
//! - [`FocalLoss`] — class-imbalance-aware cross-entropy
//! - [`ContrastiveLoss`] — metric learning for pair similarity
//! - [`TripletLoss`] — metric learning with anchor/positive/negative

use std::fmt;

// ── Loss Result ────────────────────────────────────────────────────

/// Holds both the scalar loss value and the gradient w.r.t. predictions.
#[derive(Debug, Clone)]
pub struct LossResult {
    pub loss: f64,
    pub grad: Vec<f64>,
}

impl LossResult {
    pub fn new(loss: f64, grad: Vec<f64>) -> Self {
        Self { loss, grad }
    }
}

impl fmt::Display for LossResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LossResult(loss={:.6}, grad_dim={})", self.loss, self.grad.len())
    }
}

// ── Mean Squared Error ─────────────────────────────────────────────

/// Mean Squared Error loss: `L = (1/n) Σ (pred_i - target_i)²`
///
/// Gradient: `dL/dpred_i = (2/n) * (pred_i - target_i)`
pub struct MseLoss {
    reduction: Reduction,
}

/// How to reduce per-element losses to a scalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reduction {
    Mean,
    Sum,
    None,
}

impl MseLoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    pub fn forward(&self, predictions: &[f64], targets: &[f64]) -> LossResult {
        assert_eq!(predictions.len(), targets.len());
        let n = predictions.len() as f64;
        let mut total = 0.0;
        let mut grad = Vec::with_capacity(predictions.len());

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let diff = p - t;
            total += diff * diff;
            grad.push(2.0 * diff);
        }

        match self.reduction {
            Reduction::Mean => {
                for g in grad.iter_mut() {
                    *g /= n;
                }
                LossResult::new(total / n, grad)
            }
            Reduction::Sum => LossResult::new(total, grad),
            Reduction::None => LossResult::new(total / n, grad),
        }
    }
}

impl Default for MseLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MseLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MSELoss(reduction={:?})", self.reduction)
    }
}

// ── Cross-Entropy Loss ─────────────────────────────────────────────

/// Categorical cross-entropy loss with built-in softmax.
///
/// Given raw logits and a target class index, computes:
/// ```text
/// softmax(x)_i = exp(x_i) / Σ exp(x_j)
/// L = -log(softmax(x)_target)
/// ```
pub struct CrossEntropyLoss {
    label_smoothing: f64,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            label_smoothing: 0.0,
        }
    }

    pub fn with_label_smoothing(mut self, alpha: f64) -> Self {
        self.label_smoothing = alpha.clamp(0.0, 1.0);
        self
    }

    /// Compute softmax cross-entropy for a single sample.
    /// `logits` are raw scores, `target` is the correct class index.
    pub fn forward(&self, logits: &[f64], target: usize) -> LossResult {
        assert!(target < logits.len(), "target index out of range");
        let probs = softmax(logits);
        let n_classes = logits.len() as f64;
        let alpha = self.label_smoothing;

        // Smoothed target distribution: (1-α)*one_hot + α/K
        let mut loss = 0.0;
        let mut grad = Vec::with_capacity(logits.len());

        for (i, &p) in probs.iter().enumerate() {
            let target_prob = if i == target {
                1.0 - alpha + alpha / n_classes
            } else {
                alpha / n_classes
            };
            loss -= target_prob * safe_ln(p);
            // Gradient of CE w.r.t. logits = softmax_output - target_distribution
            grad.push(p - target_prob);
        }

        LossResult::new(loss, grad)
    }

    /// Batch cross-entropy: averages loss over multiple samples.
    pub fn forward_batch(&self, logits_batch: &[Vec<f64>], targets: &[usize]) -> LossResult {
        assert_eq!(logits_batch.len(), targets.len());
        let n = logits_batch.len() as f64;
        let dim = logits_batch[0].len();
        let mut total_loss = 0.0;
        let mut avg_grad = vec![0.0; dim];

        for (logits, &target) in logits_batch.iter().zip(targets.iter()) {
            let result = self.forward(logits, target);
            total_loss += result.loss;
            for (ag, g) in avg_grad.iter_mut().zip(result.grad.iter()) {
                *ag += g / n;
            }
        }

        LossResult::new(total_loss / n, avg_grad)
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CrossEntropyLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrossEntropyLoss(smoothing={:.4})", self.label_smoothing)
    }
}

// ── Binary Cross-Entropy ───────────────────────────────────────────

/// Binary cross-entropy loss for sigmoid outputs.
///
/// ```text
/// L = -(1/n) Σ [t_i * log(p_i) + (1 - t_i) * log(1 - p_i)]
/// ```
pub struct BinaryCrossEntropyLoss {
    reduction: Reduction,
    pos_weight: f64,
}

impl BinaryCrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            pos_weight: 1.0,
        }
    }

    pub fn with_pos_weight(mut self, weight: f64) -> Self {
        self.pos_weight = weight;
        self
    }

    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    /// `predictions` should be in (0, 1) (sigmoid outputs).
    /// `targets` should be 0.0 or 1.0.
    pub fn forward(&self, predictions: &[f64], targets: &[f64]) -> LossResult {
        assert_eq!(predictions.len(), targets.len());
        let n = predictions.len() as f64;
        let mut total = 0.0;
        let mut grad = Vec::with_capacity(predictions.len());

        for (&p, &t) in predictions.iter().zip(targets.iter()) {
            let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
            let w = if t > 0.5 { self.pos_weight } else { 1.0 };
            let sample_loss =
                -w * (t * safe_ln(p_clamped) + (1.0 - t) * safe_ln(1.0 - p_clamped));
            total += sample_loss;
            // Gradient: -w * (t/p - (1-t)/(1-p))
            let g = -w * (t / p_clamped - (1.0 - t) / (1.0 - p_clamped));
            grad.push(g);
        }

        match self.reduction {
            Reduction::Mean => {
                for g in grad.iter_mut() {
                    *g /= n;
                }
                LossResult::new(total / n, grad)
            }
            Reduction::Sum => LossResult::new(total, grad),
            Reduction::None => LossResult::new(total / n, grad),
        }
    }
}

impl Default for BinaryCrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BinaryCrossEntropyLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCELoss(pos_weight={:.4})", self.pos_weight)
    }
}

// ── Huber Loss ─────────────────────────────────────────────────────

/// Huber (smooth L1) loss — quadratic for small errors, linear for large.
///
/// ```text
/// L_δ(a) = 0.5 * a²          if |a| ≤ δ
///          δ * (|a| - 0.5δ)   otherwise
/// ```
pub struct HuberLoss {
    delta: f64,
    reduction: Reduction,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self {
            delta: delta.abs().max(1e-10),
            reduction: Reduction::Mean,
        }
    }

    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }

    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta.abs().max(1e-10);
        self
    }

    pub fn forward(&self, predictions: &[f64], targets: &[f64]) -> LossResult {
        assert_eq!(predictions.len(), targets.len());
        let n = predictions.len() as f64;
        let mut total = 0.0;
        let mut grad = Vec::with_capacity(predictions.len());

        for (&p, &t) in predictions.iter().zip(targets.iter()) {
            let a = p - t;
            if a.abs() <= self.delta {
                total += 0.5 * a * a;
                grad.push(a);
            } else {
                total += self.delta * (a.abs() - 0.5 * self.delta);
                grad.push(self.delta * a.signum());
            }
        }

        match self.reduction {
            Reduction::Mean => {
                for g in grad.iter_mut() {
                    *g /= n;
                }
                LossResult::new(total / n, grad)
            }
            Reduction::Sum => LossResult::new(total, grad),
            Reduction::None => LossResult::new(total / n, grad),
        }
    }
}

impl fmt::Display for HuberLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HuberLoss(δ={:.4})", self.delta)
    }
}

// ── Focal Loss ─────────────────────────────────────────────────────

/// Focal loss (Lin et al., 2017) for addressing class imbalance.
///
/// Down-weights well-classified examples so the model focuses on hard ones:
/// ```text
/// FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
/// ```
pub struct FocalLoss {
    alpha: f64,
    gamma: f64,
}

impl FocalLoss {
    pub fn new(alpha: f64, focal_gamma: f64) -> Self {
        Self {
            alpha,
            gamma: focal_gamma,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_gamma(mut self, focal_gamma: f64) -> Self {
        self.gamma = focal_gamma;
        self
    }

    /// Compute focal loss for a single sample with softmax logits.
    pub fn forward(&self, logits: &[f64], target: usize) -> LossResult {
        assert!(target < logits.len());
        let probs = softmax(logits);
        let pt = probs[target].max(1e-15);

        let focal_weight = (1.0 - pt).powf(self.gamma);
        let loss = -self.alpha * focal_weight * safe_ln(pt);

        // Gradient: more complex due to focal modulation
        let mut grad = Vec::with_capacity(logits.len());
        for (i, &p) in probs.iter().enumerate() {
            let indicator = if i == target { 1.0 } else { 0.0 };
            // Simplified gradient combining softmax + focal weighting
            let base_grad = p - indicator;
            let focal_mod = self.alpha * focal_weight;
            grad.push(focal_mod * base_grad);
        }

        LossResult::new(loss, grad)
    }
}

impl fmt::Display for FocalLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FocalLoss(α={:.4}, γ={:.4})", self.alpha, self.gamma)
    }
}

// ── Contrastive Loss ───────────────────────────────────────────────

/// Contrastive loss for learning embeddings from pairs.
///
/// ```text
/// L = (1-y) * 0.5 * d² + y * 0.5 * max(0, margin - d)²
/// ```
/// where `y=0` for similar pairs, `y=1` for dissimilar pairs,
/// and `d` is the Euclidean distance between embeddings.
pub struct ContrastiveLoss {
    margin: f64,
}

impl ContrastiveLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    /// Compute loss for a pair of embeddings.
    /// `is_dissimilar`: `true` if the pair should be pushed apart.
    pub fn forward(
        &self,
        embedding_a: &[f64],
        embedding_b: &[f64],
        is_dissimilar: bool,
    ) -> LossResult {
        assert_eq!(embedding_a.len(), embedding_b.len());
        let dist = euclidean_distance(embedding_a, embedding_b);

        let (loss, scale_a) = if is_dissimilar {
            let hinge = (self.margin - dist).max(0.0);
            let l = 0.5 * hinge * hinge;
            // Gradient direction: push apart if within margin
            let s = if hinge > 0.0 { hinge / dist.max(1e-15) } else { 0.0 };
            (l, s)
        } else {
            let l = 0.5 * dist * dist;
            // Gradient direction: pull together
            let s = -1.0;
            (l, s)
        };

        // Gradient w.r.t. embedding_a
        let grad: Vec<f64> = embedding_a
            .iter()
            .zip(embedding_b.iter())
            .map(|(a, b)| scale_a * (b - a))
            .collect();

        LossResult::new(loss, grad)
    }
}

impl fmt::Display for ContrastiveLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContrastiveLoss(margin={:.4})", self.margin)
    }
}

// ── Triplet Loss ───────────────────────────────────────────────────

/// Triplet loss for metric learning with anchor/positive/negative.
///
/// ```text
/// L = max(0, d(a, p) - d(a, n) + margin)
/// ```
pub struct TripletLoss {
    margin: f64,
}

impl TripletLoss {
    pub fn new(margin: f64) -> Self {
        Self { margin }
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin = margin;
        self
    }

    pub fn forward(
        &self,
        anchor: &[f64],
        positive: &[f64],
        negative: &[f64],
    ) -> LossResult {
        assert_eq!(anchor.len(), positive.len());
        assert_eq!(anchor.len(), negative.len());

        let d_pos = euclidean_distance(anchor, positive);
        let d_neg = euclidean_distance(anchor, negative);
        let raw = d_pos - d_neg + self.margin;
        let loss = raw.max(0.0);

        // Gradient w.r.t. anchor: if loss > 0, push away from positive, toward negative
        let grad: Vec<f64> = if loss > 0.0 {
            let d_pos_safe = d_pos.max(1e-15);
            let d_neg_safe = d_neg.max(1e-15);
            anchor
                .iter()
                .zip(positive.iter().zip(negative.iter()))
                .map(|(a, (p, n))| {
                    (a - p) / d_pos_safe - (a - n) / d_neg_safe
                })
                .collect()
        } else {
            vec![0.0; anchor.len()]
        };

        LossResult::new(loss, grad)
    }
}

impl fmt::Display for TripletLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TripletLoss(margin={:.4})", self.margin)
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Safe natural log (clamps to avoid -inf).
fn safe_ln(x: f64) -> f64 {
    x.max(1e-15).ln()
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_zero_loss() {
        let loss = MseLoss::new();
        let result = loss.forward(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((result.loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn mse_known_value() {
        let loss = MseLoss::new();
        let result = loss.forward(&[1.0, 2.0], &[0.0, 0.0]);
        // (1+4)/2 = 2.5
        assert!((result.loss - 2.5).abs() < 1e-10);
    }

    #[test]
    fn mse_gradient_direction() {
        let loss = MseLoss::new();
        let result = loss.forward(&[3.0], &[1.0]);
        // Gradient should be positive (push prediction down toward target)
        assert!(result.grad[0] > 0.0);
    }

    #[test]
    fn mse_display() {
        let loss = MseLoss::new();
        assert!(format!("{loss}").contains("MSE"));
    }

    #[test]
    fn cross_entropy_correct_class() {
        let loss = CrossEntropyLoss::new();
        let logits = vec![10.0, 0.0, 0.0]; // strongly predicts class 0
        let result = loss.forward(&logits, 0);
        assert!(result.loss < 0.01); // very confident, low loss
    }

    #[test]
    fn cross_entropy_wrong_class() {
        let loss = CrossEntropyLoss::new();
        let logits = vec![10.0, 0.0, 0.0]; // predicts class 0
        let result = loss.forward(&logits, 2); // but target is class 2
        assert!(result.loss > 5.0); // high loss
    }

    #[test]
    fn cross_entropy_label_smoothing() {
        let no_smooth = CrossEntropyLoss::new();
        let smooth = CrossEntropyLoss::new().with_label_smoothing(0.1);
        let logits = vec![2.0, 1.0, 0.5];
        let r1 = no_smooth.forward(&logits, 0);
        let r2 = smooth.forward(&logits, 0);
        // With smoothing, loss for correct class should be slightly higher
        assert!(r2.loss > r1.loss);
    }

    #[test]
    fn cross_entropy_batch() {
        let loss = CrossEntropyLoss::new();
        let batch = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let targets = vec![0, 1];
        let result = loss.forward_batch(&batch, &targets);
        assert!(result.loss < 0.5);
    }

    #[test]
    fn bce_perfect_prediction() {
        let loss = BinaryCrossEntropyLoss::new();
        let result = loss.forward(&[0.999], &[1.0]);
        assert!(result.loss < 0.01);
    }

    #[test]
    fn bce_worst_prediction() {
        let loss = BinaryCrossEntropyLoss::new();
        let result = loss.forward(&[0.001], &[1.0]);
        assert!(result.loss > 5.0);
    }

    #[test]
    fn bce_pos_weight() {
        let unweighted = BinaryCrossEntropyLoss::new();
        let weighted = BinaryCrossEntropyLoss::new().with_pos_weight(2.0);
        let r1 = unweighted.forward(&[0.5], &[1.0]);
        let r2 = weighted.forward(&[0.5], &[1.0]);
        assert!(r2.loss > r1.loss);
    }

    #[test]
    fn huber_quadratic_regime() {
        let loss = HuberLoss::new(1.0);
        let result = loss.forward(&[0.5], &[0.0]);
        // |0.5| < 1.0, so quadratic: 0.5 * 0.25 = 0.125
        assert!((result.loss - 0.125).abs() < 1e-10);
    }

    #[test]
    fn huber_linear_regime() {
        let loss = HuberLoss::new(1.0);
        let result = loss.forward(&[5.0], &[0.0]);
        // |5| > 1.0, so linear: 1.0 * (5.0 - 0.5) = 4.5
        assert!((result.loss - 4.5).abs() < 1e-10);
    }

    #[test]
    fn focal_easy_example() {
        let loss = FocalLoss::new(1.0, 2.0);
        let logits = vec![10.0, 0.0]; // very confident class 0
        let result = loss.forward(&logits, 0);
        assert!(result.loss < 1e-6); // focal downweights easy examples
    }

    #[test]
    fn focal_hard_example() {
        let ce = CrossEntropyLoss::new();
        let fl = FocalLoss::new(1.0, 2.0);
        let logits = vec![0.1, 0.0];
        let ce_result = ce.forward(&logits, 0);
        let fl_result = fl.forward(&logits, 0);
        // Focal loss should be less than CE for uncertain predictions
        assert!(fl_result.loss < ce_result.loss);
    }

    #[test]
    fn contrastive_similar_pair() {
        let loss = ContrastiveLoss::new(1.0);
        let a = vec![1.0, 0.0];
        let b = vec![1.1, 0.0]; // close
        let result = loss.forward(&a, &b, false);
        assert!(result.loss < 0.01);
    }

    #[test]
    fn contrastive_dissimilar_within_margin() {
        let loss = ContrastiveLoss::new(5.0);
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0]; // distance=1 < margin=5
        let result = loss.forward(&a, &b, true);
        assert!(result.loss > 0.0); // penalized
    }

    #[test]
    fn triplet_satisfied() {
        let loss = TripletLoss::new(1.0);
        let anchor = vec![0.0, 0.0];
        let positive = vec![0.1, 0.0]; // close
        let negative = vec![10.0, 0.0]; // far
        let result = loss.forward(&anchor, &positive, &negative);
        assert!((result.loss - 0.0).abs() < 1e-10); // margin satisfied
    }

    #[test]
    fn triplet_violated() {
        let loss = TripletLoss::new(1.0);
        let anchor = vec![0.0, 0.0];
        let positive = vec![5.0, 0.0]; // far
        let negative = vec![1.0, 0.0]; // closer than positive
        let result = loss.forward(&anchor, &positive, &negative);
        assert!(result.loss > 0.0); // margin violated
    }

    #[test]
    fn softmax_sums_to_one() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
