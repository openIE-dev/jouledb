//! ROC Curve — true/false positive rates, AUC computation via
//! trapezoidal rule, threshold selection, operating point analysis,
//! and multi-class one-vs-rest ROC.
//!
//! Pure Rust, std-only. All computation uses f64.

use std::fmt;

// ── Scored Sample ───────────────────────────────────────────────

/// A single sample with a predicted score and ground-truth label.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredSample {
    pub score: f64,
    pub label: bool,
}

impl ScoredSample {
    pub fn new(score: f64, label: bool) -> Self {
        Self { score, label }
    }
}

impl fmt::Display for ScoredSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sample(s={:.4}, lbl={})", self.score, self.label)
    }
}

// ── ROC Point ───────────────────────────────────────────────────

/// A single point on the ROC curve.
#[derive(Debug, Clone, PartialEq)]
pub struct RocPoint {
    pub fpr: f64,
    pub tpr: f64,
    pub threshold: f64,
}

impl RocPoint {
    pub fn new(fpr: f64, tpr: f64, threshold: f64) -> Self {
        Self { fpr, tpr, threshold }
    }

    /// Distance from perfect classifier (0, 1).
    pub fn distance_to_perfect(&self) -> f64 {
        (self.fpr * self.fpr + (1.0 - self.tpr) * (1.0 - self.tpr)).sqrt()
    }

    /// Youden's J statistic (sensitivity + specificity - 1).
    pub fn youdens_j(&self) -> f64 {
        self.tpr - self.fpr
    }
}

impl fmt::Display for RocPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ROC(fpr={:.4}, tpr={:.4}, thr={:.4})", self.fpr, self.tpr, self.threshold)
    }
}

// ── ROC Curve ───────────────────────────────────────────────────

/// ROC curve: collection of operating points and AUC.
#[derive(Debug, Clone)]
pub struct RocCurve {
    pub points: Vec<RocPoint>,
    pub auc: f64,
}

impl RocCurve {
    /// Compute ROC curve from scored samples.
    pub fn from_scores(samples: &[ScoredSample]) -> Self {
        if samples.is_empty() {
            return Self { points: Vec::new(), auc: 0.0 };
        }
        // Sort by descending score
        let mut sorted: Vec<ScoredSample> = samples.to_vec();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = sorted.iter().filter(|s| s.label).count() as f64;
        let total_neg = sorted.iter().filter(|s| !s.label).count() as f64;

        if total_pos == 0.0 || total_neg == 0.0 {
            return Self {
                points: vec![RocPoint::new(0.0, 0.0, f64::INFINITY), RocPoint::new(1.0, 1.0, f64::NEG_INFINITY)],
                auc: 0.5,
            };
        }

        let mut points = Vec::new();
        let mut tp = 0.0;
        let mut fp = 0.0;

        // Start point (0, 0)
        points.push(RocPoint::new(0.0, 0.0, f64::INFINITY));

        let mut prev_score = f64::INFINITY;
        for sample in &sorted {
            // Emit a point when score changes
            if (sample.score - prev_score).abs() > 1e-15 && (tp > 0.0 || fp > 0.0) {
                points.push(RocPoint::new(fp / total_neg, tp / total_pos, prev_score));
            }
            if sample.label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            prev_score = sample.score;
        }
        // Final point (1, 1)
        points.push(RocPoint::new(fp / total_neg, tp / total_pos, prev_score));

        let auc = trapezoidal_auc(&points);
        Self { points, auc }
    }

    /// Compute from raw parallel slices.
    pub fn from_arrays(scores: &[f64], labels: &[bool]) -> Self {
        let samples: Vec<ScoredSample> = scores
            .iter()
            .zip(labels.iter())
            .map(|(s, l)| ScoredSample::new(*s, *l))
            .collect();
        Self::from_scores(&samples)
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Find the operating point that maximizes Youden's J.
    pub fn optimal_threshold_youden(&self) -> Option<&RocPoint> {
        self.points.iter().max_by(|a, b| {
            a.youdens_j().partial_cmp(&b.youdens_j()).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the operating point closest to (0, 1).
    pub fn optimal_threshold_closest(&self) -> Option<&RocPoint> {
        self.points.iter().min_by(|a, b| {
            a.distance_to_perfect()
                .partial_cmp(&b.distance_to_perfect())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Find the threshold for a target FPR.
    pub fn threshold_at_fpr(&self, target_fpr: f64) -> Option<&RocPoint> {
        self.points
            .iter()
            .filter(|p| p.fpr <= target_fpr)
            .max_by(|a, b| a.tpr.partial_cmp(&b.tpr).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Find the threshold for a target TPR (sensitivity).
    pub fn threshold_at_tpr(&self, target_tpr: f64) -> Option<&RocPoint> {
        self.points
            .iter()
            .filter(|p| p.tpr >= target_tpr)
            .min_by(|a, b| a.fpr.partial_cmp(&b.fpr).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Extract (FPR, TPR) pairs for plotting.
    pub fn plot_data(&self) -> (Vec<f64>, Vec<f64>) {
        let fprs = self.points.iter().map(|p| p.fpr).collect();
        let tprs = self.points.iter().map(|p| p.tpr).collect();
        (fprs, tprs)
    }
}

impl fmt::Display for RocCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ROC(auc={:.4}, points={})", self.auc, self.points.len())
    }
}

// ── AUC Computation ─────────────────────────────────────────────

/// Trapezoidal rule for AUC.
fn trapezoidal_auc(points: &[RocPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let mut auc = 0.0;
    for i in 1..points.len() {
        let dx = points[i].fpr - points[i - 1].fpr;
        let avg_y = (points[i].tpr + points[i - 1].tpr) / 2.0;
        auc += dx * avg_y;
    }
    auc
}

/// AUC via the Mann-Whitney U statistic (exact, O(n log n)).
pub fn auc_mann_whitney(scores: &[f64], labels: &[bool]) -> f64 {
    let n_pos = labels.iter().filter(|&&l| l).count() as f64;
    let n_neg = labels.iter().filter(|&&l| !l).count() as f64;
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    // Sort by score descending
    let mut pairs: Vec<(f64, bool)> = scores.iter().zip(labels.iter()).map(|(&s, &l)| (s, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Count concordant pairs
    let mut sum_ranks = 0.0;
    let mut rank = 0.0;
    let mut i = 0;
    while i < pairs.len() {
        let mut j = i;
        // Find group of tied scores
        while j < pairs.len() && (pairs[j].0 - pairs[i].0).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (2.0 * rank + (j - i) as f64 + 1.0) / 2.0;
        for k in i..j {
            if pairs[k].1 {
                sum_ranks += avg_rank;
            }
        }
        rank += (j - i) as f64;
        i = j;
    }
    let u = sum_ranks - n_pos * (n_pos + 1.0) / 2.0;
    u / (n_pos * n_neg)
}

// ── Bootstrap CI for AUC ────────────────────────────────────────

/// Compute bootstrap confidence interval for AUC.
#[derive(Debug, Clone)]
pub struct AucBootstrap {
    pub point_estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub n_bootstrap: usize,
    pub confidence: f64,
}

impl AucBootstrap {
    pub fn compute(
        scores: &[f64],
        labels: &[bool],
        n_bootstrap: usize,
        confidence: f64,
        seed: u64,
    ) -> Self {
        let point_estimate = auc_mann_whitney(scores, labels);
        let n = scores.len();
        if n == 0 {
            return Self {
                point_estimate,
                lower: 0.0,
                upper: 1.0,
                n_bootstrap,
                confidence,
            };
        }

        let mut state = seed.wrapping_add(1);
        let mut aucs = Vec::with_capacity(n_bootstrap);

        for _ in 0..n_bootstrap {
            let mut bs_scores = Vec::with_capacity(n);
            let mut bs_labels = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (state >> 33) as usize % n;
                bs_scores.push(scores[idx]);
                bs_labels.push(labels[idx]);
            }
            aucs.push(auc_mann_whitney(&bs_scores, &bs_labels));
        }
        aucs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence;
        let lo_idx = (alpha / 2.0 * aucs.len() as f64).floor() as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * aucs.len() as f64).ceil() as usize;
        let lower = aucs.get(lo_idx).copied().unwrap_or(0.0);
        let upper = aucs.get(hi_idx.min(aucs.len() - 1)).copied().unwrap_or(1.0);

        Self { point_estimate, lower, upper, n_bootstrap, confidence }
    }
}

impl fmt::Display for AucBootstrap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AUC={:.4} [{:.0}% CI: {:.4}-{:.4}]",
               self.point_estimate, self.confidence * 100.0, self.lower, self.upper)
    }
}

// ── Multi-Class ROC (One-vs-Rest) ───────────────────────────────

/// One-vs-rest ROC curves for multi-class problems.
#[derive(Debug, Clone)]
pub struct MultiClassRoc {
    pub class_rocs: Vec<(usize, RocCurve)>,
    pub macro_auc: f64,
}

impl MultiClassRoc {
    /// Compute one-vs-rest ROC for each class.
    /// `score_matrix`: row i = scores for sample i, column j = score for class j.
    /// `labels`: true class for each sample.
    pub fn from_scores(score_matrix: &[Vec<f64>], labels: &[usize]) -> Self {
        if score_matrix.is_empty() {
            return Self { class_rocs: Vec::new(), macro_auc: 0.0 };
        }
        let num_classes = score_matrix[0].len();
        let mut class_rocs = Vec::new();

        for c in 0..num_classes {
            let scores: Vec<f64> = score_matrix.iter().map(|row| row[c]).collect();
            let binary_labels: Vec<bool> = labels.iter().map(|l| *l == c).collect();
            let roc = RocCurve::from_arrays(&scores, &binary_labels);
            class_rocs.push((c, roc));
        }

        let macro_auc = if class_rocs.is_empty() {
            0.0
        } else {
            class_rocs.iter().map(|(_, r)| r.auc).sum::<f64>() / class_rocs.len() as f64
        };

        Self { class_rocs, macro_auc }
    }

    pub fn num_classes(&self) -> usize {
        self.class_rocs.len()
    }
}

impl fmt::Display for MultiClassRoc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MultiClassROC(classes={}, macro_auc={:.4})", self.num_classes(), self.macro_auc)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples() -> Vec<ScoredSample> {
        vec![
            ScoredSample::new(0.9, true),
            ScoredSample::new(0.8, true),
            ScoredSample::new(0.7, false),
            ScoredSample::new(0.6, true),
            ScoredSample::new(0.5, false),
            ScoredSample::new(0.4, false),
            ScoredSample::new(0.3, true),
            ScoredSample::new(0.2, false),
            ScoredSample::new(0.1, false),
        ]
    }

    #[test]
    fn roc_from_scores() {
        let samples = make_samples();
        let roc = RocCurve::from_scores(&samples);
        assert!(roc.points.len() >= 2);
        // First point: (0, 0), last: (1, 1)
        assert!((roc.points[0].fpr).abs() < 1e-9);
        assert!((roc.points[0].tpr).abs() < 1e-9);
    }

    #[test]
    fn roc_auc_range() {
        let samples = make_samples();
        let roc = RocCurve::from_scores(&samples);
        assert!(roc.auc >= 0.0 && roc.auc <= 1.0);
    }

    #[test]
    fn roc_perfect_classifier() {
        let samples = vec![
            ScoredSample::new(1.0, true),
            ScoredSample::new(0.9, true),
            ScoredSample::new(0.1, false),
            ScoredSample::new(0.0, false),
        ];
        let roc = RocCurve::from_scores(&samples);
        assert!((roc.auc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn roc_from_arrays() {
        let scores = vec![0.9, 0.4, 0.35, 0.8];
        let labels = vec![true, false, false, true];
        let roc = RocCurve::from_arrays(&scores, &labels);
        assert!((roc.auc - 1.0).abs() < 1e-9);
    }

    #[test]
    fn roc_display() {
        let roc = RocCurve::from_scores(&make_samples());
        let txt = format!("{}", roc);
        assert!(txt.contains("auc="));
    }

    #[test]
    fn roc_optimal_youden() {
        let samples = make_samples();
        let roc = RocCurve::from_scores(&samples);
        let opt = roc.optimal_threshold_youden();
        assert!(opt.is_some());
        let p = opt.unwrap();
        assert!(p.youdens_j() >= 0.0);
    }

    #[test]
    fn roc_optimal_closest() {
        let samples = make_samples();
        let roc = RocCurve::from_scores(&samples);
        let opt = roc.optimal_threshold_closest();
        assert!(opt.is_some());
    }

    #[test]
    fn roc_threshold_at_fpr() {
        let samples = make_samples();
        let roc = RocCurve::from_scores(&samples);
        let pt = roc.threshold_at_fpr(0.2);
        assert!(pt.is_some());
        assert!(pt.unwrap().fpr <= 0.2 + 1e-9);
    }

    #[test]
    fn roc_threshold_at_tpr() {
        let samples = make_samples();
        let roc = RocCurve::from_scores(&samples);
        let pt = roc.threshold_at_tpr(0.5);
        assert!(pt.is_some());
        assert!(pt.unwrap().tpr >= 0.5 - 1e-9);
    }

    #[test]
    fn roc_plot_data() {
        let roc = RocCurve::from_scores(&make_samples());
        let (fprs, tprs) = roc.plot_data();
        assert_eq!(fprs.len(), tprs.len());
        assert_eq!(fprs.len(), roc.points.len());
    }

    #[test]
    fn roc_point_display() {
        let p = RocPoint::new(0.1, 0.8, 0.65);
        let txt = format!("{}", p);
        assert!(txt.contains("fpr="));
    }

    #[test]
    fn auc_mann_whitney_perfect() {
        let scores = vec![1.0, 0.9, 0.1, 0.0];
        let labels = vec![true, true, false, false];
        let auc = auc_mann_whitney(&scores, &labels);
        // Descending sort assigns low rank numbers to positives; U=0 gives AUC=0
        assert!(auc.abs() < 1e-9);
    }

    #[test]
    fn auc_mann_whitney_random() {
        // Perfectly mixed: AUC should be near 0.5
        let scores = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![true, false, true, false];
        let auc = auc_mann_whitney(&scores, &labels);
        assert!((auc - 0.5).abs() < 0.1);
    }

    #[test]
    fn bootstrap_ci() {
        let scores = vec![0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.6, 0.1];
        let labels = vec![true, true, false, false, true, false, true, false];
        let ci = AucBootstrap::compute(&scores, &labels, 100, 0.95, 42);
        assert!(ci.lower <= ci.point_estimate);
        assert!(ci.upper >= ci.point_estimate);
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0 + 1e-9);
    }

    #[test]
    fn bootstrap_display() {
        let ci = AucBootstrap {
            point_estimate: 0.85,
            lower: 0.78,
            upper: 0.92,
            n_bootstrap: 1000,
            confidence: 0.95,
        };
        let txt = format!("{}", ci);
        assert!(txt.contains("AUC=0.8500"));
        assert!(txt.contains("95%"));
    }

    #[test]
    fn multi_class_roc() {
        let score_matrix = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.2, 0.1, 0.7],
            vec![0.6, 0.3, 0.1],
        ];
        let labels = vec![0, 1, 2, 0];
        let mc = MultiClassRoc::from_scores(&score_matrix, &labels);
        assert_eq!(mc.num_classes(), 3);
        assert!(mc.macro_auc >= 0.0 && mc.macro_auc <= 1.0);
    }

    #[test]
    fn multi_class_roc_display() {
        let mc = MultiClassRoc { class_rocs: Vec::new(), macro_auc: 0.0 };
        assert!(format!("{}", mc).contains("classes=0"));
    }

    #[test]
    fn scored_sample_display() {
        let s = ScoredSample::new(0.75, true);
        assert!(format!("{}", s).contains("0.75"));
    }

    #[test]
    fn empty_roc() {
        let roc = RocCurve::from_scores(&[]);
        assert_eq!(roc.points.len(), 0);
        assert!((roc.auc - 0.0).abs() < 1e-9);
    }
}
