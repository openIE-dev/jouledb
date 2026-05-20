//! Confusion Matrix — TP/FP/TN/FN counts, accuracy, precision, recall,
//! F1 score, specificity, MCC, Cohen's kappa, and multi-class support.
//!
//! Pure Rust, std-only. Works with integer class labels.

use std::collections::HashMap;
use std::fmt;

// ── Binary Confusion Matrix ─────────────────────────────────────

/// Confusion matrix for binary classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryConfusion {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub fn_count: usize,
}

impl BinaryConfusion {
    pub fn new() -> Self {
        Self { tp: 0, fp: 0, tn: 0, fn_count: 0 }
    }

    /// Build from prediction and ground-truth slices (1 = positive, 0 = negative).
    pub fn from_predictions(predictions: &[usize], actuals: &[usize]) -> Self {
        let mut cm = Self::new();
        for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
            match (pred > 0, actual > 0) {
                (true, true) => cm.tp += 1,
                (true, false) => cm.fp += 1,
                (false, true) => cm.fn_count += 1,
                (false, false) => cm.tn += 1,
            }
        }
        cm
    }

    pub fn total(&self) -> usize {
        self.tp + self.fp + self.tn + self.fn_count
    }

    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 { return 0.0; }
        (self.tp + self.tn) as f64 / total as f64
    }

    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { return 0.0; }
        self.tp as f64 / denom as f64
    }

    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_count;
        if denom == 0 { return 0.0; }
        self.tp as f64 / denom as f64
    }

    pub fn specificity(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom == 0 { return 0.0; }
        self.tn as f64 / denom as f64
    }

    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 { return 0.0; }
        2.0 * p * r / (p + r)
    }

    /// F-beta score: generalization of F1.
    pub fn f_beta(&self, beta: f64) -> f64 {
        let p = self.precision();
        let r = self.recall();
        let b2 = beta * beta;
        let denom = b2 * p + r;
        if denom == 0.0 { return 0.0; }
        (1.0 + b2) * p * r / denom
    }

    /// Matthews Correlation Coefficient.
    pub fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let tn = self.tn as f64;
        let fp = self.fp as f64;
        let fn_ = self.fn_count as f64;
        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denom == 0.0 { return 0.0; }
        (tp * tn - fp * fn_) / denom
    }

    /// False positive rate (fall-out).
    pub fn fpr(&self) -> f64 {
        1.0 - self.specificity()
    }

    /// False negative rate (miss rate).
    pub fn fnr(&self) -> f64 {
        1.0 - self.recall()
    }

    /// Negative predictive value.
    pub fn npv(&self) -> f64 {
        let denom = self.tn + self.fn_count;
        if denom == 0 { return 0.0; }
        self.tn as f64 / denom as f64
    }
}

impl fmt::Display for BinaryConfusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryConfusion(tp={}, fp={}, tn={}, fn={}, acc={:.4})",
               self.tp, self.fp, self.tn, self.fn_count, self.accuracy())
    }
}

// ── Multi-Class Confusion Matrix ────────────────────────────────

/// Confusion matrix for multi-class classification.
#[derive(Debug, Clone)]
pub struct MultiClassConfusion {
    matrix: Vec<Vec<usize>>,
    labels: Vec<usize>,
    label_to_idx: HashMap<usize, usize>,
}

impl MultiClassConfusion {
    /// Create from known class labels.
    pub fn new(labels: Vec<usize>) -> Self {
        let n = labels.len();
        let label_to_idx: HashMap<usize, usize> =
            labels.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        Self {
            matrix: vec![vec![0; n]; n],
            labels,
            label_to_idx,
        }
    }

    /// Build from prediction and ground-truth slices.
    pub fn from_predictions(predictions: &[usize], actuals: &[usize]) -> Self {
        let mut all_labels: Vec<usize> = predictions
            .iter()
            .chain(actuals.iter())
            .copied()
            .collect();
        all_labels.sort();
        all_labels.dedup();
        let mut cm = Self::new(all_labels);
        for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
            cm.record(actual, pred);
        }
        cm
    }

    /// Record a single prediction.
    pub fn record(&mut self, actual: usize, predicted: usize) {
        if let (Some(&ai), Some(&pi)) = (self.label_to_idx.get(&actual), self.label_to_idx.get(&predicted)) {
            self.matrix[ai][pi] += 1;
        }
    }

    pub fn num_classes(&self) -> usize {
        self.labels.len()
    }

    /// Get count at (actual, predicted).
    pub fn get(&self, actual: usize, predicted: usize) -> usize {
        let ai = self.label_to_idx.get(&actual).copied().unwrap_or(0);
        let pi = self.label_to_idx.get(&predicted).copied().unwrap_or(0);
        self.matrix[ai][pi]
    }

    /// Total number of samples.
    pub fn total(&self) -> usize {
        self.matrix.iter().flat_map(|row| row.iter()).sum()
    }

    /// Overall accuracy.
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 { return 0.0; }
        let correct: usize = (0..self.labels.len()).map(|i| self.matrix[i][i]).sum();
        correct as f64 / total as f64
    }

    /// Per-class metrics: (precision, recall, f1, support) for a given class.
    pub fn class_metrics(&self, class: usize) -> ClassMetrics {
        let idx = match self.label_to_idx.get(&class) {
            Some(&i) => i,
            None => return ClassMetrics::zero(class),
        };
        let tp = self.matrix[idx][idx];
        let fp: usize = (0..self.labels.len()).filter(|i| *i != idx).map(|i| self.matrix[i][idx]).sum();
        let fn_: usize = (0..self.labels.len()).filter(|j| *j != idx).map(|j| self.matrix[idx][j]).sum();
        let support = tp + fn_;
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        ClassMetrics { class, precision, recall, f1, support }
    }

    /// Macro-averaged precision (unweighted average across classes).
    pub fn macro_precision(&self) -> f64 {
        let n = self.labels.len() as f64;
        if n == 0.0 { return 0.0; }
        self.labels.iter().map(|c| self.class_metrics(*c).precision).sum::<f64>() / n
    }

    /// Macro-averaged recall.
    pub fn macro_recall(&self) -> f64 {
        let n = self.labels.len() as f64;
        if n == 0.0 { return 0.0; }
        self.labels.iter().map(|c| self.class_metrics(*c).recall).sum::<f64>() / n
    }

    /// Macro-averaged F1.
    pub fn macro_f1(&self) -> f64 {
        let n = self.labels.len() as f64;
        if n == 0.0 { return 0.0; }
        self.labels.iter().map(|c| self.class_metrics(*c).f1).sum::<f64>() / n
    }

    /// Weighted-averaged F1 (weighted by class support).
    pub fn weighted_f1(&self) -> f64 {
        let total = self.total() as f64;
        if total == 0.0 { return 0.0; }
        self.labels.iter()
            .map(|c| {
                let m = self.class_metrics(*c);
                m.f1 * m.support as f64
            })
            .sum::<f64>() / total
    }

    /// Cohen's Kappa coefficient.
    pub fn cohen_kappa(&self) -> f64 {
        let n = self.total() as f64;
        if n == 0.0 { return 0.0; }
        let p0 = self.accuracy();
        let mut pe = 0.0;
        for i in 0..self.labels.len() {
            let row_sum: usize = self.matrix[i].iter().sum();
            let col_sum: usize = (0..self.labels.len()).map(|j| self.matrix[j][i]).sum();
            pe += (row_sum as f64 / n) * (col_sum as f64 / n);
        }
        if (1.0 - pe).abs() < 1e-15 { return 1.0; }
        (p0 - pe) / (1.0 - pe)
    }

    /// Return the raw matrix.
    pub fn raw_matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }
}

impl fmt::Display for MultiClassConfusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiClassConfusion(classes={}, acc={:.4})", self.num_classes(), self.accuracy())
    }
}

// ── Class Metrics ───────────────────────────────────────────────

/// Per-class evaluation metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassMetrics {
    pub class: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

impl ClassMetrics {
    fn zero(class: usize) -> Self {
        Self { class, precision: 0.0, recall: 0.0, f1: 0.0, support: 0 }
    }
}

impl fmt::Display for ClassMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Class {} (P={:.3}, R={:.3}, F1={:.3}, n={})",
               self.class, self.precision, self.recall, self.f1, self.support)
    }
}

// ── Classification Report ───────────────────────────────────────

/// Full classification report with per-class and aggregate metrics.
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub class_metrics: Vec<ClassMetrics>,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub weighted_f1: f64,
    pub total_samples: usize,
}

impl ClassificationReport {
    /// Build a report from a confusion matrix.
    pub fn from_confusion(cm: &MultiClassConfusion) -> Self {
        let class_metrics: Vec<ClassMetrics> = cm.labels.iter().map(|c| cm.class_metrics(*c)).collect();
        Self {
            class_metrics,
            accuracy: cm.accuracy(),
            macro_f1: cm.macro_f1(),
            weighted_f1: cm.weighted_f1(),
            total_samples: cm.total(),
        }
    }
}

impl fmt::Display for ClassificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Classification Report (n={}):", self.total_samples)?;
        writeln!(f, "{:<10} {:>9} {:>9} {:>9} {:>9}", "Class", "Precision", "Recall", "F1", "Support")?;
        for m in &self.class_metrics {
            writeln!(f, "{:<10} {:>9.4} {:>9.4} {:>9.4} {:>9}",
                     m.class, m.precision, m.recall, m.f1, m.support)?;
        }
        writeln!(f, "")?;
        writeln!(f, "Accuracy: {:.4}", self.accuracy)?;
        writeln!(f, "Macro F1: {:.4}", self.macro_f1)?;
        write!(f, "Weighted F1: {:.4}", self.weighted_f1)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_perfect() {
        let cm = BinaryConfusion { tp: 50, fp: 0, tn: 50, fn_count: 0 };
        assert!((cm.accuracy() - 1.0).abs() < 1e-9);
        assert!((cm.precision() - 1.0).abs() < 1e-9);
        assert!((cm.recall() - 1.0).abs() < 1e-9);
        assert!((cm.f1_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn binary_from_predictions() {
        let preds = vec![1, 1, 0, 0, 1];
        let actuals = vec![1, 0, 0, 1, 1];
        let cm = BinaryConfusion::from_predictions(&preds, &actuals);
        assert_eq!(cm.tp, 2);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.tn, 1);
        assert_eq!(cm.fn_count, 1);
    }

    #[test]
    fn binary_precision_recall() {
        let cm = BinaryConfusion { tp: 40, fp: 10, tn: 45, fn_count: 5 };
        assert!((cm.precision() - 0.8).abs() < 1e-9);
        assert!((cm.recall() - 40.0 / 45.0).abs() < 1e-9);
    }

    #[test]
    fn binary_f_beta() {
        let cm = BinaryConfusion { tp: 30, fp: 10, tn: 50, fn_count: 10 };
        let f1 = cm.f1_score();
        let f1_via_beta = cm.f_beta(1.0);
        assert!((f1 - f1_via_beta).abs() < 1e-9);
    }

    #[test]
    fn binary_mcc_perfect() {
        let cm = BinaryConfusion { tp: 50, fp: 0, tn: 50, fn_count: 0 };
        assert!((cm.mcc() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn binary_mcc_random() {
        let cm = BinaryConfusion { tp: 25, fp: 25, tn: 25, fn_count: 25 };
        assert!((cm.mcc() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn binary_specificity() {
        let cm = BinaryConfusion { tp: 40, fp: 5, tn: 45, fn_count: 10 };
        assert!((cm.specificity() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn binary_npv() {
        let cm = BinaryConfusion { tp: 40, fp: 5, tn: 45, fn_count: 10 };
        let expected = 45.0 / 55.0;
        assert!((cm.npv() - expected).abs() < 1e-9);
    }

    #[test]
    fn binary_display() {
        let cm = BinaryConfusion { tp: 10, fp: 2, tn: 8, fn_count: 0 };
        let txt = format!("{}", cm);
        assert!(txt.contains("tp=10"));
    }

    #[test]
    fn multi_class_basic() {
        let preds = vec![0, 0, 1, 1, 2, 2];
        let actuals = vec![0, 1, 1, 2, 2, 0];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        assert_eq!(cm.num_classes(), 3);
        assert_eq!(cm.total(), 6);
    }

    #[test]
    fn multi_class_perfect() {
        let preds = vec![0, 1, 2, 0, 1, 2];
        let actuals = vec![0, 1, 2, 0, 1, 2];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        assert!((cm.accuracy() - 1.0).abs() < 1e-9);
        assert!((cm.macro_f1() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn multi_class_per_class() {
        let preds = vec![0, 0, 0, 1, 1, 1];
        let actuals = vec![0, 0, 1, 1, 1, 0];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        let m0 = cm.class_metrics(0);
        assert_eq!(m0.support, 3);
    }

    #[test]
    fn multi_class_kappa_perfect() {
        let preds = vec![0, 1, 2, 0, 1, 2];
        let actuals = vec![0, 1, 2, 0, 1, 2];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        assert!((cm.cohen_kappa() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn multi_class_display() {
        let cm = MultiClassConfusion::from_predictions(&[0, 1], &[0, 1]);
        let txt = format!("{}", cm);
        assert!(txt.contains("classes=2"));
    }

    #[test]
    fn classification_report_structure() {
        let preds = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let actuals = vec![0, 1, 2, 0, 2, 1, 1, 1, 2];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        let report = ClassificationReport::from_confusion(&cm);
        assert_eq!(report.class_metrics.len(), 3);
        assert_eq!(report.total_samples, 9);
    }

    #[test]
    fn classification_report_display() {
        let preds = vec![0, 1, 0, 1];
        let actuals = vec![0, 1, 1, 1];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        let report = ClassificationReport::from_confusion(&cm);
        let txt = format!("{}", report);
        assert!(txt.contains("Classification Report"));
        assert!(txt.contains("Accuracy"));
    }

    #[test]
    fn class_metrics_display() {
        let m = ClassMetrics { class: 1, precision: 0.85, recall: 0.9, f1: 0.874, support: 20 };
        let txt = format!("{}", m);
        assert!(txt.contains("Class 1"));
    }

    #[test]
    fn weighted_f1_vs_macro_f1() {
        let preds = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 1];
        let actuals = vec![0, 0, 0, 1, 1, 1, 1, 1, 0, 0];
        let cm = MultiClassConfusion::from_predictions(&preds, &actuals);
        let macro_f1 = cm.macro_f1();
        let weighted_f1 = cm.weighted_f1();
        // They can differ with imbalanced classes
        assert!(macro_f1 >= 0.0 && macro_f1 <= 1.0);
        assert!(weighted_f1 >= 0.0 && weighted_f1 <= 1.0);
    }
}
