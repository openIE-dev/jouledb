//! Knowledge distillation: teacher-student training, soft targets, temperature
//! scaling, feature matching, and response-based distillation.
//!
//! Transfers knowledge from a large teacher model to a smaller student model
//! by training the student to match the teacher's output distribution (soft
//! targets) or intermediate feature representations.

use std::collections::HashMap;
use std::fmt;

// ── Soft Targets ───────────────────────────────────────────────

/// Apply temperature scaling to logits and compute softmax.
pub fn softmax_with_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    let t = if temperature <= 0.0 { 1e-8 } else { temperature };
    let scaled: Vec<f64> = logits.iter().map(|l| l / t).collect();
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|s| (s - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// Compute KL divergence: KL(P || Q) = Σ P(x) * ln(P(x)/Q(x)).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            if *pi <= 0.0 {
                return 0.0;
            }
            let qi_safe = qi.max(1e-12);
            pi * (pi / qi_safe).ln()
        })
        .sum()
}

/// Cross-entropy loss: -Σ P(x) * ln(Q(x)).
pub fn cross_entropy(targets: &[f64], predictions: &[f64]) -> f64 {
    assert_eq!(targets.len(), predictions.len());
    -targets
        .iter()
        .zip(predictions.iter())
        .map(|(t, p)| {
            if *t <= 0.0 {
                return 0.0;
            }
            t * p.max(1e-12).ln()
        })
        .sum::<f64>()
}

/// Mean squared error between two vectors.
pub fn mse_loss(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    sum / a.len() as f64
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom == 0.0 {
        return 0.0;
    }
    dot / denom
}

// ── Distillation Loss ──────────────────────────────────────────

/// Type of distillation loss to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistillLossKind {
    /// KL divergence between soft teacher and student distributions.
    KLDivergence,
    /// MSE between teacher and student logits or features.
    MSE,
    /// Cosine similarity loss (1 - cos_sim).
    CosineSimilarity,
    /// Cross-entropy with soft teacher labels.
    SoftCrossEntropy,
}

impl fmt::Display for DistillLossKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistillLossKind::KLDivergence => write!(f, "kl-divergence"),
            DistillLossKind::MSE => write!(f, "mse"),
            DistillLossKind::CosineSimilarity => write!(f, "cosine"),
            DistillLossKind::SoftCrossEntropy => write!(f, "soft-cross-entropy"),
        }
    }
}

/// Compute the distillation loss between teacher and student outputs.
pub fn distillation_loss(
    teacher_logits: &[f64],
    student_logits: &[f64],
    temperature: f64,
    kind: DistillLossKind,
) -> f64 {
    match kind {
        DistillLossKind::KLDivergence => {
            let teacher_soft = softmax_with_temperature(teacher_logits, temperature);
            let student_soft = softmax_with_temperature(student_logits, temperature);
            let kl = kl_divergence(&teacher_soft, &student_soft);
            // Scale by T^2 as per Hinton et al.
            kl * temperature * temperature
        }
        DistillLossKind::MSE => mse_loss(teacher_logits, student_logits),
        DistillLossKind::CosineSimilarity => {
            1.0 - cosine_similarity(teacher_logits, student_logits)
        }
        DistillLossKind::SoftCrossEntropy => {
            let teacher_soft = softmax_with_temperature(teacher_logits, temperature);
            let student_soft = softmax_with_temperature(student_logits, temperature);
            cross_entropy(&teacher_soft, &student_soft) * temperature * temperature
        }
    }
}

// ── Feature Matching ───────────────────────────────────────────

/// Feature matching configuration for intermediate layer distillation.
#[derive(Debug, Clone)]
pub struct FeatureMatch {
    pub teacher_layer: String,
    pub student_layer: String,
    pub loss_kind: DistillLossKind,
    pub weight: f64,
}

impl FeatureMatch {
    pub fn new(
        teacher_layer: impl Into<String>,
        student_layer: impl Into<String>,
    ) -> Self {
        Self {
            teacher_layer: teacher_layer.into(),
            student_layer: student_layer.into(),
            loss_kind: DistillLossKind::MSE,
            weight: 1.0,
        }
    }

    pub fn with_loss(mut self, kind: DistillLossKind) -> Self {
        self.loss_kind = kind;
        self
    }

    pub fn with_weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    /// Compute the feature matching loss.
    pub fn compute_loss(&self, teacher_features: &[f64], student_features: &[f64]) -> f64 {
        let raw = match self.loss_kind {
            DistillLossKind::MSE => mse_loss(teacher_features, student_features),
            DistillLossKind::CosineSimilarity => {
                1.0 - cosine_similarity(teacher_features, student_features)
            }
            _ => mse_loss(teacher_features, student_features),
        };
        raw * self.weight
    }
}

impl fmt::Display for FeatureMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FeatureMatch({} -> {}, loss={}, w={:.2})",
            self.teacher_layer, self.student_layer, self.loss_kind, self.weight
        )
    }
}

// ── Distillation Config ────────────────────────────────────────

/// Configuration for the knowledge distillation pipeline.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Temperature for soft targets.
    pub temperature: f64,
    /// Weight for distillation loss (α).
    pub alpha: f64,
    /// Weight for hard-label cross-entropy loss (1-α).
    pub hard_label_weight: f64,
    /// Output loss kind.
    pub loss_kind: DistillLossKind,
    /// Feature matching pairs.
    pub feature_matches: Vec<FeatureMatch>,
    /// Learning rate for student.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: u32,
    /// Extra metadata.
    pub metadata: HashMap<String, String>,
}

impl DistillConfig {
    pub fn new() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            hard_label_weight: 0.3,
            loss_kind: DistillLossKind::KLDivergence,
            feature_matches: Vec::new(),
            learning_rate: 1e-3,
            epochs: 10,
            metadata: HashMap::new(),
        }
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_alpha(mut self, a: f64) -> Self {
        self.alpha = a;
        self.hard_label_weight = 1.0 - a;
        self
    }

    pub fn with_loss_kind(mut self, kind: DistillLossKind) -> Self {
        self.loss_kind = kind;
        self
    }

    pub fn with_feature_match(mut self, fm: FeatureMatch) -> Self {
        self.feature_matches.push(fm);
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_epochs(mut self, n: u32) -> Self {
        self.epochs = n;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DistillConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DistillConfig(T={}, α={:.2}, loss={}, features={}, epochs={})",
            self.temperature,
            self.alpha,
            self.loss_kind,
            self.feature_matches.len(),
            self.epochs
        )
    }
}

// ── Distillation Step ──────────────────────────────────────────

/// Result of one distillation training step.
#[derive(Debug, Clone)]
pub struct DistillStep {
    pub step: u64,
    pub distill_loss: f64,
    pub hard_loss: f64,
    pub feature_loss: f64,
    pub total_loss: f64,
}

impl fmt::Display for DistillStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Step {} — distill={:.6}, hard={:.6}, feature={:.6}, total={:.6}",
            self.step, self.distill_loss, self.hard_loss, self.feature_loss, self.total_loss
        )
    }
}

/// Compute a single distillation step.
pub fn compute_distill_step(
    config: &DistillConfig,
    teacher_logits: &[f64],
    student_logits: &[f64],
    hard_labels: &[f64],
    teacher_features: &HashMap<String, Vec<f64>>,
    student_features: &HashMap<String, Vec<f64>>,
    step: u64,
) -> DistillStep {
    // Distillation loss
    let distill_loss = distillation_loss(
        teacher_logits,
        student_logits,
        config.temperature,
        config.loss_kind,
    );

    // Hard label loss
    let student_soft = softmax_with_temperature(student_logits, 1.0);
    let hard_loss = cross_entropy(hard_labels, &student_soft);

    // Feature matching loss
    let mut feature_loss = 0.0;
    for fm in &config.feature_matches {
        if let (Some(tf), Some(sf)) = (
            teacher_features.get(&fm.teacher_layer),
            student_features.get(&fm.student_layer),
        ) {
            feature_loss += fm.compute_loss(tf, sf);
        }
    }

    let total_loss =
        config.alpha * distill_loss + config.hard_label_weight * hard_loss + feature_loss;

    DistillStep {
        step,
        distill_loss,
        hard_loss,
        feature_loss,
        total_loss,
    }
}

// ── Training History ───────────────────────────────────────────

/// Accumulates distillation training history.
#[derive(Debug, Clone)]
pub struct DistillHistory {
    pub steps: Vec<DistillStep>,
}

impl DistillHistory {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn record(&mut self, step: DistillStep) {
        self.steps.push(step);
    }

    /// Average total loss over all recorded steps.
    pub fn avg_total_loss(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.steps.iter().map(|s| s.total_loss).sum();
        sum / self.steps.len() as f64
    }

    /// Minimum total loss observed.
    pub fn min_total_loss(&self) -> f64 {
        self.steps
            .iter()
            .map(|s| s.total_loss)
            .fold(f64::INFINITY, f64::min)
    }

    /// Whether loss is trending down (comparing first and last quarter).
    pub fn is_improving(&self) -> bool {
        if self.steps.len() < 4 {
            return true;
        }
        let q = self.steps.len() / 4;
        let first_avg: f64 =
            self.steps[..q].iter().map(|s| s.total_loss).sum::<f64>() / q as f64;
        let last_avg: f64 =
            self.steps[self.steps.len() - q..].iter().map(|s| s.total_loss).sum::<f64>()
                / q as f64;
        last_avg < first_avg
    }
}

impl Default for DistillHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DistillHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DistillHistory(steps={}, avg_loss={:.6})",
            self.steps.len(),
            self.avg_total_loss()
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_temp_1() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_high_temperature() {
        let logits = vec![1.0, 2.0, 3.0];
        let soft = softmax_with_temperature(&logits, 100.0);
        // High temp → nearly uniform
        let diff = soft[0] - soft[2];
        assert!(diff.abs() < 0.05);
    }

    #[test]
    fn test_softmax_low_temperature() {
        let logits = vec![1.0, 2.0, 5.0];
        let sharp = softmax_with_temperature(&logits, 0.1);
        // Low temp → peaky
        assert!(sharp[2] > 0.99);
    }

    #[test]
    fn test_kl_divergence_same_dist() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_cross_entropy_perfect() {
        let targets = vec![0.0, 1.0, 0.0];
        let preds = vec![0.01, 0.98, 0.01];
        let ce = cross_entropy(&targets, &preds);
        assert!(ce > 0.0);
        assert!(ce < 0.1); // near-perfect prediction
    }

    #[test]
    fn test_mse_loss_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(mse_loss(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss_value() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        // ((1-3)^2 + (2-4)^2) / 2 = (4+4)/2 = 4
        assert!((mse_loss(&a, &b) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_distillation_loss_kl() {
        let teacher = vec![1.0, 5.0, 2.0];
        let student = vec![1.0, 5.0, 2.0];
        let loss = distillation_loss(&teacher, &student, 4.0, DistillLossKind::KLDivergence);
        assert!(loss.abs() < 1e-10); // identical logits → 0 KL
    }

    #[test]
    fn test_distillation_loss_mse() {
        let teacher = vec![1.0, 2.0];
        let student = vec![3.0, 4.0];
        let loss = distillation_loss(&teacher, &student, 1.0, DistillLossKind::MSE);
        assert!((loss - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_distillation_loss_cosine() {
        let teacher = vec![1.0, 0.0];
        let student = vec![0.0, 1.0];
        let loss = distillation_loss(&teacher, &student, 1.0, DistillLossKind::CosineSimilarity);
        assert!((loss - 1.0).abs() < 1e-10); // orthogonal → loss = 1
    }

    #[test]
    fn test_feature_match_mse() {
        let fm = FeatureMatch::new("layer3", "layer1").with_weight(2.0);
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![1.0, 2.0, 3.0];
        assert!(fm.compute_loss(&teacher, &student).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let cfg = DistillConfig::new()
            .with_temperature(6.0)
            .with_alpha(0.8)
            .with_loss_kind(DistillLossKind::MSE)
            .with_learning_rate(1e-4)
            .with_epochs(20)
            .with_metadata("note", "test");
        assert_eq!(cfg.temperature, 6.0);
        assert!((cfg.alpha - 0.8).abs() < 1e-10);
        assert!((cfg.hard_label_weight - 0.2).abs() < 1e-10);
        assert_eq!(cfg.epochs, 20);
    }

    #[test]
    fn test_config_default() {
        let cfg = DistillConfig::default();
        assert_eq!(cfg.temperature, 4.0);
    }

    #[test]
    fn test_compute_distill_step() {
        let cfg = DistillConfig::new().with_alpha(0.5);
        let teacher_logits = vec![1.0, 3.0, 0.5];
        let student_logits = vec![0.5, 2.5, 0.8];
        let hard_labels = vec![0.0, 1.0, 0.0];

        let step = compute_distill_step(
            &cfg,
            &teacher_logits,
            &student_logits,
            &hard_labels,
            &HashMap::new(),
            &HashMap::new(),
            0,
        );
        assert!(step.total_loss > 0.0);
        assert_eq!(step.feature_loss, 0.0);
    }

    #[test]
    fn test_history_tracking() {
        let mut history = DistillHistory::new();
        for i in 0..10 {
            history.record(DistillStep {
                step: i,
                distill_loss: 1.0 / (i as f64 + 1.0),
                hard_loss: 0.5 / (i as f64 + 1.0),
                feature_loss: 0.0,
                total_loss: 1.5 / (i as f64 + 1.0),
            });
        }
        assert!(history.avg_total_loss() > 0.0);
        assert!(history.min_total_loss() < history.avg_total_loss());
        assert!(history.is_improving());
    }

    #[test]
    fn test_display_impls() {
        assert!(format!("{}", DistillLossKind::KLDivergence).contains("kl"));
        let fm = FeatureMatch::new("t1", "s1");
        assert!(format!("{fm}").contains("t1"));
        let cfg = DistillConfig::new();
        assert!(format!("{cfg}").contains("DistillConfig"));
        let h = DistillHistory::new();
        assert!(format!("{h}").contains("steps=0"));
    }
}
