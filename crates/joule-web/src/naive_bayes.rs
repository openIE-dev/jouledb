//! Naive Bayes Classifiers — Gaussian NB for continuous features and
//! Multinomial NB for count/text features, with Laplace smoothing and
//! probability predictions.
//!
//! Pure Rust — no external ML dependencies.

use std::fmt;

use serde::{Deserialize, Serialize};

// ── Gaussian Naive Bayes ────────────────────────────────────────

/// Per-class statistics for a single feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GaussianFeatureStats {
    mean: f64,
    variance: f64,
}

/// Gaussian Naive Bayes classifier for continuous features.
///
/// Assumes each feature follows a Gaussian distribution within each class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianNB {
    /// Per-class, per-feature statistics: class_stats[class][feature].
    class_stats: Vec<Vec<GaussianFeatureStats>>,
    /// Prior probabilities for each class.
    priors: Vec<f64>,
    /// Number of classes.
    n_classes: usize,
    /// Number of features.
    n_features: usize,
    /// Minimum variance to prevent division by zero.
    var_smoothing: f64,
}

impl GaussianNB {
    /// Create a new Gaussian NB classifier.
    pub fn new() -> Self {
        Self {
            class_stats: vec![],
            priors: vec![],
            n_classes: 0,
            n_features: 0,
            var_smoothing: 1e-9,
        }
    }

    /// Create with custom variance smoothing.
    pub fn with_var_smoothing(var_smoothing: f64) -> Self {
        Self { var_smoothing, ..Self::new() }
    }

    /// Fit the model on training data. Labels are 0-based class indices.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[usize]) {
        assert_eq!(features.len(), labels.len(), "feature/label mismatch");
        assert!(!features.is_empty(), "empty training set");

        self.n_features = features[0].len();
        self.n_classes = labels.iter().copied().max().unwrap_or(0) + 1;

        // Group samples by class
        let mut class_samples: Vec<Vec<usize>> = vec![vec![]; self.n_classes];
        for (i, &label) in labels.iter().enumerate() {
            class_samples[label].push(i);
        }

        // Compute priors
        let n_total = features.len() as f64;
        self.priors = class_samples.iter().map(|s| s.len() as f64 / n_total).collect();

        // Compute per-class, per-feature mean and variance
        self.class_stats = Vec::with_capacity(self.n_classes);
        for class_idx in 0..self.n_classes {
            let indices = &class_samples[class_idx];
            let mut feat_stats = Vec::with_capacity(self.n_features);

            for feat_idx in 0..self.n_features {
                if indices.is_empty() {
                    feat_stats.push(GaussianFeatureStats { mean: 0.0, variance: self.var_smoothing });
                    continue;
                }
                let vals: Vec<f64> = indices.iter().map(|i| features[*i][feat_idx]).collect();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / vals.len() as f64
                    + self.var_smoothing;
                feat_stats.push(GaussianFeatureStats { mean, variance });
            }

            self.class_stats.push(feat_stats);
        }
    }

    /// Predict the class label for a single sample.
    pub fn predict(&self, sample: &[f64]) -> usize {
        let log_probs = self.log_probabilities(sample);
        log_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict classes for multiple samples.
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Get class probabilities for a sample (normalized).
    pub fn predict_proba(&self, sample: &[f64]) -> Vec<f64> {
        let log_probs = self.log_probabilities(sample);
        // Numeric stability: subtract max
        let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = log_probs.iter().map(|lp| (lp - max_log).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Compute log-probabilities for each class.
    fn log_probabilities(&self, sample: &[f64]) -> Vec<f64> {
        assert_eq!(sample.len(), self.n_features, "dimension mismatch");
        let mut log_probs = Vec::with_capacity(self.n_classes);
        for class_idx in 0..self.n_classes {
            let mut log_prob = self.priors[class_idx].ln();
            for feat_idx in 0..self.n_features {
                let stats = &self.class_stats[class_idx][feat_idx];
                log_prob += gaussian_log_pdf(sample[feat_idx], stats.mean, stats.variance);
            }
            log_probs.push(log_prob);
        }
        log_probs
    }

    /// Compute accuracy on a test set.
    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        assert_eq!(features.len(), labels.len());
        if features.is_empty() {
            return 0.0;
        }
        let preds = self.predict_batch(features);
        let correct = preds.iter().zip(labels).filter(|(p, l)| p == l).count();
        correct as f64 / features.len() as f64
    }

    /// Return prior probabilities.
    pub fn priors(&self) -> &[f64] {
        &self.priors
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for GaussianNB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GaussianNB(classes={}, features={})", self.n_classes, self.n_features)
    }
}

/// Log of Gaussian PDF.
fn gaussian_log_pdf(x: f64, mean: f64, variance: f64) -> f64 {
    -0.5 * ((x - mean).powi(2) / variance + variance.ln() + (2.0 * std::f64::consts::PI).ln())
}

// ── Multinomial Naive Bayes ─────────────────────────────────────

/// Multinomial Naive Bayes classifier for count/text features.
///
/// Suitable for document classification with word counts or TF features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomialNB {
    /// Log conditional probabilities: feature_log_probs[class][feature].
    feature_log_probs: Vec<Vec<f64>>,
    /// Log prior probabilities for each class.
    log_priors: Vec<f64>,
    /// Number of classes.
    n_classes: usize,
    /// Number of features.
    n_features: usize,
    /// Laplace smoothing parameter.
    alpha: f64,
}

impl MultinomialNB {
    /// Create with default Laplace smoothing (alpha=1.0).
    pub fn new() -> Self {
        Self {
            feature_log_probs: vec![],
            log_priors: vec![],
            n_classes: 0,
            n_features: 0,
            alpha: 1.0,
        }
    }

    /// Create with custom smoothing parameter.
    pub fn with_alpha(alpha: f64) -> Self {
        assert!(alpha >= 0.0, "alpha must be non-negative");
        Self { alpha, ..Self::new() }
    }

    /// Fit the model on training data. Feature values should be non-negative counts.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[usize]) {
        assert_eq!(features.len(), labels.len(), "feature/label mismatch");
        assert!(!features.is_empty(), "empty training set");

        self.n_features = features[0].len();
        self.n_classes = labels.iter().copied().max().unwrap_or(0) + 1;

        // Group by class
        let mut class_samples: Vec<Vec<usize>> = vec![vec![]; self.n_classes];
        for (i, &label) in labels.iter().enumerate() {
            class_samples[label].push(i);
        }

        // Compute log priors
        let n_total = features.len() as f64;
        self.log_priors = class_samples.iter().map(|s| (s.len() as f64 / n_total).ln()).collect();

        // Compute feature log-probabilities with Laplace smoothing
        self.feature_log_probs = Vec::with_capacity(self.n_classes);
        for class_idx in 0..self.n_classes {
            let indices = &class_samples[class_idx];
            let mut feat_sums = vec![0.0_f64; self.n_features];
            for &i in indices {
                for (j, val) in features[i].iter().enumerate() {
                    feat_sums[j] += val;
                }
            }
            let total_count: f64 = feat_sums.iter().sum::<f64>() + self.alpha * self.n_features as f64;
            let log_probs: Vec<f64> = feat_sums
                .iter()
                .map(|s| ((s + self.alpha) / total_count).ln())
                .collect();
            self.feature_log_probs.push(log_probs);
        }
    }

    /// Predict the class label for a single sample.
    pub fn predict(&self, sample: &[f64]) -> usize {
        let log_probs = self.log_joint_likelihoods(sample);
        log_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict classes for multiple samples.
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Get normalized class probabilities.
    pub fn predict_proba(&self, sample: &[f64]) -> Vec<f64> {
        let log_probs = self.log_joint_likelihoods(sample);
        let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = log_probs.iter().map(|lp| (lp - max_log).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    fn log_joint_likelihoods(&self, sample: &[f64]) -> Vec<f64> {
        assert_eq!(sample.len(), self.n_features, "dimension mismatch");
        let mut results = Vec::with_capacity(self.n_classes);
        for class_idx in 0..self.n_classes {
            let mut log_prob = self.log_priors[class_idx];
            for feat_idx in 0..self.n_features {
                log_prob += sample[feat_idx] * self.feature_log_probs[class_idx][feat_idx];
            }
            results.push(log_prob);
        }
        results
    }

    /// Compute accuracy on a test set.
    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        assert_eq!(features.len(), labels.len());
        if features.is_empty() {
            return 0.0;
        }
        let preds = self.predict_batch(features);
        let correct = preds.iter().zip(labels).filter(|(p, l)| p == l).count();
        correct as f64 / features.len() as f64
    }

    /// Return the smoothing parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for MultinomialNB {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MultinomialNB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultinomialNB(classes={}, features={}, alpha={})",
            self.n_classes, self.n_features, self.alpha
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        let features = vec![
            // Class 0: cluster around (1, 1)
            vec![0.8, 0.9], vec![1.0, 1.0], vec![1.2, 1.1], vec![0.9, 1.2],
            vec![1.1, 0.8],
            // Class 1: cluster around (5, 5)
            vec![4.8, 4.9], vec![5.0, 5.0], vec![5.2, 5.1], vec![4.9, 5.2],
            vec![5.1, 4.8],
        ];
        let labels = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        (features, labels)
    }

    fn count_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        let features = vec![
            // Class 0: "sports" documents (high word1, word2)
            vec![5.0, 3.0, 0.0], vec![4.0, 4.0, 1.0], vec![6.0, 2.0, 0.0],
            // Class 1: "tech" documents (high word3)
            vec![0.0, 1.0, 5.0], vec![1.0, 0.0, 6.0], vec![0.0, 0.0, 7.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        (features, labels)
    }

    #[test]
    fn test_gaussian_log_pdf() {
        let lp = gaussian_log_pdf(0.0, 0.0, 1.0);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_fit_predict() {
        let (feats, labs) = gaussian_data();
        let mut gnb = GaussianNB::new();
        gnb.fit(&feats, &labs);

        assert_eq!(gnb.predict(&[1.0, 1.0]), 0);
        assert_eq!(gnb.predict(&[5.0, 5.0]), 1);
    }

    #[test]
    fn test_gaussian_accuracy() {
        let (feats, labs) = gaussian_data();
        let mut gnb = GaussianNB::new();
        gnb.fit(&feats, &labs);
        let acc = gnb.accuracy(&feats, &labs);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_predict_proba() {
        let (feats, labs) = gaussian_data();
        let mut gnb = GaussianNB::new();
        gnb.fit(&feats, &labs);

        let proba = gnb.predict_proba(&[1.0, 1.0]);
        assert_eq!(proba.len(), 2);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(proba[0] > 0.9); // Should be confident class 0
    }

    #[test]
    fn test_gaussian_priors() {
        let (feats, labs) = gaussian_data();
        let mut gnb = GaussianNB::new();
        gnb.fit(&feats, &labs);
        let priors = gnb.priors();
        assert_eq!(priors.len(), 2);
        assert!((priors[0] - 0.5).abs() < 1e-10);
        assert!((priors[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_batch() {
        let (feats, labs) = gaussian_data();
        let mut gnb = GaussianNB::new();
        gnb.fit(&feats, &labs);
        let preds = gnb.predict_batch(&feats);
        assert_eq!(preds, labs);
    }

    #[test]
    fn test_gaussian_var_smoothing() {
        let gnb = GaussianNB::with_var_smoothing(1e-6);
        assert!((gnb.var_smoothing - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_serialization() {
        let (feats, labs) = gaussian_data();
        let mut gnb = GaussianNB::new();
        gnb.fit(&feats, &labs);

        let json = gnb.to_json().unwrap();
        let gnb2 = GaussianNB::from_json(&json).unwrap();
        assert_eq!(gnb.predict(&[1.0, 1.0]), gnb2.predict(&[1.0, 1.0]));
        assert_eq!(gnb.predict(&[5.0, 5.0]), gnb2.predict(&[5.0, 5.0]));
    }

    #[test]
    fn test_gaussian_display() {
        let gnb = GaussianNB::new();
        let s = format!("{}", gnb);
        assert!(s.contains("GaussianNB"));
    }

    #[test]
    fn test_multinomial_fit_predict() {
        let (feats, labs) = count_data();
        let mut mnb = MultinomialNB::new();
        mnb.fit(&feats, &labs);

        assert_eq!(mnb.predict(&[5.0, 3.0, 0.0]), 0);
        assert_eq!(mnb.predict(&[0.0, 0.0, 7.0]), 1);
    }

    #[test]
    fn test_multinomial_accuracy() {
        let (feats, labs) = count_data();
        let mut mnb = MultinomialNB::new();
        mnb.fit(&feats, &labs);
        let acc = mnb.accuracy(&feats, &labs);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multinomial_predict_proba() {
        let (feats, labs) = count_data();
        let mut mnb = MultinomialNB::new();
        mnb.fit(&feats, &labs);

        let proba = mnb.predict_proba(&[5.0, 3.0, 0.0]);
        assert_eq!(proba.len(), 2);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(proba[0] > proba[1]);
    }

    #[test]
    fn test_multinomial_laplace_smoothing() {
        let (feats, labs) = count_data();
        let mut mnb = MultinomialNB::with_alpha(2.0);
        mnb.fit(&feats, &labs);
        assert!((mnb.alpha() - 2.0).abs() < 1e-10);
        // Should still classify correctly with heavier smoothing
        let acc = mnb.accuracy(&feats, &labs);
        assert!(acc > 0.8);
    }

    #[test]
    fn test_multinomial_batch() {
        let (feats, labs) = count_data();
        let mut mnb = MultinomialNB::new();
        mnb.fit(&feats, &labs);
        let preds = mnb.predict_batch(&feats);
        assert_eq!(preds, labs);
    }

    #[test]
    fn test_multinomial_serialization() {
        let (feats, labs) = count_data();
        let mut mnb = MultinomialNB::new();
        mnb.fit(&feats, &labs);

        let json = mnb.to_json().unwrap();
        let mnb2 = MultinomialNB::from_json(&json).unwrap();
        assert_eq!(mnb.predict(&[5.0, 3.0, 0.0]), mnb2.predict(&[5.0, 3.0, 0.0]));
    }

    #[test]
    fn test_multinomial_display() {
        let mnb = MultinomialNB::new();
        let s = format!("{}", mnb);
        assert!(s.contains("MultinomialNB"));
    }

    #[test]
    fn test_gaussian_default() {
        let gnb = GaussianNB::default();
        assert_eq!(gnb.n_classes, 0);
    }

    #[test]
    fn test_multinomial_default() {
        let mnb = MultinomialNB::default();
        assert_eq!(mnb.n_classes, 0);
        assert!((mnb.alpha - 1.0).abs() < 1e-10);
    }
}
