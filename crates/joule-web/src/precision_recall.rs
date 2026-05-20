//! Precision-Recall Curves — average precision (AP), interpolated
//! precision, F-beta score, micro/macro/weighted averaging, and
//! multi-class PR analysis.
//!
//! Pure Rust, std-only. All computation uses f64.

use std::collections::HashMap;
use std::fmt;

// ── PR Point ────────────────────────────────────────────────────

/// A single point on the precision-recall curve.
#[derive(Debug, Clone, PartialEq)]
pub struct PrPoint {
    pub precision: f64,
    pub recall: f64,
    pub threshold: f64,
}

impl PrPoint {
    pub fn new(precision: f64, recall: f64, threshold: f64) -> Self {
        Self { precision, recall, threshold }
    }

    /// F-beta score at this operating point.
    pub fn f_beta(&self, beta: f64) -> f64 {
        let b2 = beta * beta;
        let denom = b2 * self.precision + self.recall;
        if denom == 0.0 {
            return 0.0;
        }
        (1.0 + b2) * self.precision * self.recall / denom
    }

    /// F1 score at this point.
    pub fn f1(&self) -> f64 {
        self.f_beta(1.0)
    }
}

impl fmt::Display for PrPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PR(p={:.4}, r={:.4}, thr={:.4})", self.precision, self.recall, self.threshold)
    }
}

// ── Precision-Recall Curve ──────────────────────────────────────

/// Precision-recall curve with average precision.
#[derive(Debug, Clone)]
pub struct PrCurve {
    pub points: Vec<PrPoint>,
    pub ap: f64,
}

impl PrCurve {
    /// Compute PR curve from parallel score and label arrays.
    pub fn from_scores(scores: &[f64], labels: &[bool]) -> Self {
        if scores.is_empty() {
            return Self { points: Vec::new(), ap: 0.0 };
        }
        // Sort by descending score
        let mut pairs: Vec<(f64, bool)> = scores
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = pairs.iter().filter(|(_, l)| *l).count() as f64;
        if total_pos == 0.0 {
            return Self { points: Vec::new(), ap: 0.0 };
        }

        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut points = Vec::new();

        for &(score, label) in &pairs {
            if label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let precision = tp / (tp + fp);
            let recall = tp / total_pos;
            points.push(PrPoint::new(precision, recall, score));
        }

        let ap = average_precision(&points);
        Self { points, ap }
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Find the point maximizing F1.
    pub fn best_f1_point(&self) -> Option<&PrPoint> {
        self.points
            .iter()
            .max_by(|a, b| a.f1().partial_cmp(&b.f1()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the point maximizing F-beta.
    pub fn best_f_beta_point(&self, beta: f64) -> Option<&PrPoint> {
        self.points.iter().max_by(|a, b| {
            a.f_beta(beta)
                .partial_cmp(&b.f_beta(beta))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Threshold at a target precision.
    pub fn threshold_at_precision(&self, target: f64) -> Option<&PrPoint> {
        self.points
            .iter()
            .filter(|p| p.precision >= target)
            .max_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Threshold at a target recall.
    pub fn threshold_at_recall(&self, target: f64) -> Option<&PrPoint> {
        self.points
            .iter()
            .filter(|p| p.recall >= target)
            .max_by(|a, b| a.precision.partial_cmp(&b.precision).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Interpolated precision at given recall levels (11-point interpolation).
    pub fn interpolated_precision(&self, recall_levels: &[f64]) -> Vec<f64> {
        recall_levels
            .iter()
            .map(|r| {
                self.points
                    .iter()
                    .filter(|p| p.recall >= *r)
                    .map(|p| p.precision)
                    .fold(0.0_f64, f64::max)
            })
            .collect()
    }

    /// Standard 11-point interpolation (recall = 0.0, 0.1, ..., 1.0).
    pub fn eleven_point_interpolation(&self) -> Vec<f64> {
        let levels: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
        self.interpolated_precision(&levels)
    }

    /// Extract (recall, precision) pairs for plotting.
    pub fn plot_data(&self) -> (Vec<f64>, Vec<f64>) {
        let recalls = self.points.iter().map(|p| p.recall).collect();
        let precisions = self.points.iter().map(|p| p.precision).collect();
        (recalls, precisions)
    }
}

impl fmt::Display for PrCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrCurve(ap={:.4}, points={})", self.ap, self.points.len())
    }
}

// ── Average Precision ───────────────────────────────────────────

/// Compute average precision from a PR curve (area under the PR curve).
fn average_precision(points: &[PrPoint]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for p in points {
        let recall_change = p.recall - prev_recall;
        ap += recall_change * p.precision;
        prev_recall = p.recall;
    }
    ap
}

/// Compute AP using all-points interpolation (sklearn-style).
pub fn average_precision_interpolated(points: &[PrPoint]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    // Reverse order: from high recall to low
    let mut sorted: Vec<PrPoint> = points.to_vec();
    sorted.sort_by(|a, b| b.recall.partial_cmp(&a.recall).unwrap_or(std::cmp::Ordering::Equal));

    // Compute monotone-decreasing envelope of precision
    let mut max_prec = 0.0_f64;
    let mut interp: Vec<(f64, f64)> = Vec::new();
    for p in sorted.iter().rev() {
        max_prec = max_prec.max(p.precision);
        interp.push((p.recall, max_prec));
    }

    // Sum area under interpolated curve
    let mut ap = 0.0;
    let mut prev_recall = 0.0;
    for &(recall, prec) in &interp {
        let dr = recall - prev_recall;
        if dr > 0.0 {
            ap += dr * prec;
        }
        prev_recall = recall;
    }
    ap
}

// ── F-Beta Score ────────────────────────────────────────────────

/// Compute the F-beta score from precision and recall.
pub fn f_beta_score(precision: f64, recall: f64, beta: f64) -> f64 {
    let b2 = beta * beta;
    let denom = b2 * precision + recall;
    if denom == 0.0 {
        return 0.0;
    }
    (1.0 + b2) * precision * recall / denom
}

/// F1 convenience function.
pub fn f1_score(precision: f64, recall: f64) -> f64 {
    f_beta_score(precision, recall, 1.0)
}

/// F2 convenience (favors recall over precision).
pub fn f2_score(precision: f64, recall: f64) -> f64 {
    f_beta_score(precision, recall, 2.0)
}

/// F0.5 convenience (favors precision over recall).
pub fn f05_score(precision: f64, recall: f64) -> f64 {
    f_beta_score(precision, recall, 0.5)
}

// ── Multi-Class Averaging ───────────────────────────────────────

/// Per-class precision, recall, f1 from multi-class predictions.
#[derive(Debug, Clone)]
pub struct PerClassMetrics {
    pub class: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

impl fmt::Display for PerClassMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Class {} (P={:.4}, R={:.4}, F1={:.4}, n={})",
               self.class, self.precision, self.recall, self.f1, self.support)
    }
}

/// Compute per-class metrics from predictions and ground truth.
pub fn per_class_metrics(predictions: &[usize], actuals: &[usize]) -> Vec<PerClassMetrics> {
    let mut all_labels: Vec<usize> = predictions
        .iter()
        .chain(actuals.iter())
        .copied()
        .collect();
    all_labels.sort();
    all_labels.dedup();

    all_labels
        .iter()
        .map(|cls| {
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut fn_ = 0usize;
            for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
                if pred == *cls && actual == *cls {
                    tp += 1;
                } else if pred == *cls && actual != *cls {
                    fp += 1;
                } else if pred != *cls && actual == *cls {
                    fn_ += 1;
                }
            }
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
            let f1 = f1_score(precision, recall);
            PerClassMetrics { class: *cls, precision, recall, f1, support: tp + fn_ }
        })
        .collect()
}

/// Macro-averaged precision, recall, F1.
#[derive(Debug, Clone)]
pub struct MacroAverage {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl MacroAverage {
    pub fn compute(per_class: &[PerClassMetrics]) -> Self {
        let n = per_class.len() as f64;
        if n == 0.0 {
            return Self { precision: 0.0, recall: 0.0, f1: 0.0 };
        }
        Self {
            precision: per_class.iter().map(|m| m.precision).sum::<f64>() / n,
            recall: per_class.iter().map(|m| m.recall).sum::<f64>() / n,
            f1: per_class.iter().map(|m| m.f1).sum::<f64>() / n,
        }
    }
}

impl fmt::Display for MacroAverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Macro(P={:.4}, R={:.4}, F1={:.4})", self.precision, self.recall, self.f1)
    }
}

/// Micro-averaged precision, recall, F1 (aggregate TP/FP/FN across classes).
#[derive(Debug, Clone)]
pub struct MicroAverage {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl MicroAverage {
    pub fn compute(predictions: &[usize], actuals: &[usize]) -> Self {
        let mut classes: Vec<usize> = predictions
            .iter()
            .chain(actuals.iter())
            .copied()
            .collect();
        classes.sort();
        classes.dedup();

        let mut total_tp = 0usize;
        let mut total_fp = 0usize;
        let mut total_fn = 0usize;

        for &cls in &classes {
            for (&pred, &actual) in predictions.iter().zip(actuals.iter()) {
                if pred == cls && actual == cls {
                    total_tp += 1;
                } else if pred == cls && actual != cls {
                    total_fp += 1;
                } else if pred != cls && actual == cls {
                    total_fn += 1;
                }
            }
        }

        let precision = if total_tp + total_fp > 0 {
            total_tp as f64 / (total_tp + total_fp) as f64
        } else {
            0.0
        };
        let recall = if total_tp + total_fn > 0 {
            total_tp as f64 / (total_tp + total_fn) as f64
        } else {
            0.0
        };
        let f1 = f1_score(precision, recall);
        Self { precision, recall, f1 }
    }
}

impl fmt::Display for MicroAverage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Micro(P={:.4}, R={:.4}, F1={:.4})", self.precision, self.recall, self.f1)
    }
}

/// Weighted-averaged F1 (weighted by support).
pub fn weighted_f1(per_class: &[PerClassMetrics]) -> f64 {
    let total_support: usize = per_class.iter().map(|m| m.support).sum();
    if total_support == 0 {
        return 0.0;
    }
    per_class
        .iter()
        .map(|m| m.f1 * m.support as f64)
        .sum::<f64>()
        / total_support as f64
}

// ── Multi-Class AP (mAP) ───────────────────────────────────────

/// Mean Average Precision across classes (common in detection tasks).
#[derive(Debug, Clone)]
pub struct MeanAveragePrecision {
    pub per_class_ap: HashMap<usize, f64>,
    pub map_score: f64,
}

impl MeanAveragePrecision {
    /// Compute mAP from per-class score vectors and labels.
    /// `class_scores`: map from class -> (scores, is_positive) per sample.
    pub fn compute(class_scores: &HashMap<usize, (Vec<f64>, Vec<bool>)>) -> Self {
        let mut per_class_ap = HashMap::new();
        for (&cls, (scores, labels)) in class_scores {
            let pr = PrCurve::from_scores(scores, labels);
            per_class_ap.insert(cls, pr.ap);
        }
        let map_score = if per_class_ap.is_empty() {
            0.0
        } else {
            per_class_ap.values().sum::<f64>() / per_class_ap.len() as f64
        };
        Self { per_class_ap, map_score }
    }

    pub fn num_classes(&self) -> usize {
        self.per_class_ap.len()
    }
}

impl fmt::Display for MeanAveragePrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mAP={:.4} (classes={})", self.map_score, self.num_classes())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scores_labels() -> (Vec<f64>, Vec<bool>) {
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1];
        let labels = vec![true, true, false, true, false, false, true, false, false];
        (scores, labels)
    }

    #[test]
    fn pr_curve_basic() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        assert!(!pr.points.is_empty());
        assert!(pr.ap > 0.0 && pr.ap <= 1.0);
    }

    #[test]
    fn pr_curve_perfect() {
        let scores = vec![1.0, 0.9, 0.1, 0.0];
        let labels = vec![true, true, false, false];
        let pr = PrCurve::from_scores(&scores, &labels);
        assert!((pr.ap - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pr_curve_display() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        let txt = format!("{}", pr);
        assert!(txt.contains("ap="));
    }

    #[test]
    fn pr_point_f1() {
        let p = PrPoint::new(0.8, 0.6, 0.5);
        let f1 = p.f1();
        let expected = 2.0 * 0.8 * 0.6 / (0.8 + 0.6);
        assert!((f1 - expected).abs() < 1e-9);
    }

    #[test]
    fn pr_best_f1() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        let best = pr.best_f1_point();
        assert!(best.is_some());
        assert!(best.unwrap().f1() > 0.0);
    }

    #[test]
    fn pr_threshold_at_precision() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        let pt = pr.threshold_at_precision(0.8);
        assert!(pt.is_some());
        assert!(pt.unwrap().precision >= 0.8 - 1e-9);
    }

    #[test]
    fn pr_threshold_at_recall() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        let pt = pr.threshold_at_recall(0.5);
        assert!(pt.is_some());
        assert!(pt.unwrap().recall >= 0.5 - 1e-9);
    }

    #[test]
    fn eleven_point_interpolation() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        let interp = pr.eleven_point_interpolation();
        assert_eq!(interp.len(), 11);
        // Interpolated precision should be monotonically non-increasing
        for i in 1..interp.len() {
            assert!(interp[i] <= interp[i - 1] + 1e-9);
        }
    }

    #[test]
    fn f_beta_f1() {
        let f1 = f_beta_score(0.8, 0.6, 1.0);
        let expected = 2.0 * 0.8 * 0.6 / 1.4;
        assert!((f1 - expected).abs() < 1e-9);
    }

    #[test]
    fn f_beta_f2() {
        let f2 = f2_score(0.8, 0.6);
        // F2 favors recall, so F2 < F1 when precision > recall
        let f1 = f1_score(0.8, 0.6);
        assert!(f2 < f1);
    }

    #[test]
    fn f_beta_f05() {
        let f05 = f05_score(0.8, 0.6);
        let f1 = f1_score(0.8, 0.6);
        // F0.5 favors precision, so F0.5 > F1 when precision > recall
        assert!(f05 > f1);
    }

    #[test]
    fn per_class_metrics_basic() {
        let preds = vec![0, 0, 1, 1, 2, 2];
        let actuals = vec![0, 1, 1, 2, 2, 0];
        let metrics = per_class_metrics(&preds, &actuals);
        assert_eq!(metrics.len(), 3);
        for m in &metrics {
            assert!(m.precision >= 0.0 && m.precision <= 1.0);
            assert!(m.recall >= 0.0 && m.recall <= 1.0);
        }
    }

    #[test]
    fn macro_average_basic() {
        let preds = vec![0, 1, 2, 0, 1, 2];
        let actuals = vec![0, 1, 2, 0, 1, 2];
        let pc = per_class_metrics(&preds, &actuals);
        let macro_avg = MacroAverage::compute(&pc);
        assert!((macro_avg.precision - 1.0).abs() < 1e-9);
        assert!((macro_avg.recall - 1.0).abs() < 1e-9);
    }

    #[test]
    fn macro_display() {
        let m = MacroAverage { precision: 0.85, recall: 0.9, f1: 0.874 };
        assert!(format!("{}", m).contains("Macro"));
    }

    #[test]
    fn micro_average_basic() {
        let preds = vec![0, 1, 2, 0, 1, 2];
        let actuals = vec![0, 1, 2, 0, 1, 2];
        let micro = MicroAverage::compute(&preds, &actuals);
        assert!((micro.precision - 1.0).abs() < 1e-9);
    }

    #[test]
    fn micro_display() {
        let m = MicroAverage { precision: 0.8, recall: 0.75, f1: 0.774 };
        assert!(format!("{}", m).contains("Micro"));
    }

    #[test]
    fn weighted_f1_basic() {
        let preds = vec![0, 1, 0, 1, 0, 1];
        let actuals = vec![0, 1, 0, 1, 0, 1];
        let pc = per_class_metrics(&preds, &actuals);
        let wf1 = weighted_f1(&pc);
        assert!((wf1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn map_basic() {
        let mut class_scores = HashMap::new();
        class_scores.insert(0, (vec![0.9, 0.1], vec![true, false]));
        class_scores.insert(1, (vec![0.8, 0.2], vec![true, false]));
        let map = MeanAveragePrecision::compute(&class_scores);
        assert!((map.map_score - 1.0).abs() < 1e-9);
        assert_eq!(map.num_classes(), 2);
    }

    #[test]
    fn map_display() {
        let map = MeanAveragePrecision {
            per_class_ap: HashMap::new(),
            map_score: 0.85,
        };
        assert!(format!("{}", map).contains("mAP="));
    }

    #[test]
    fn pr_empty() {
        let pr = PrCurve::from_scores(&[], &[]);
        assert_eq!(pr.points.len(), 0);
        assert!((pr.ap - 0.0).abs() < 1e-9);
    }

    #[test]
    fn pr_plot_data() {
        let (scores, labels) = make_scores_labels();
        let pr = PrCurve::from_scores(&scores, &labels);
        let (recalls, precisions) = pr.plot_data();
        assert_eq!(recalls.len(), precisions.len());
    }
}
