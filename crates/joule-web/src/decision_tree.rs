//! Decision Tree Classifier — ID3/C4.5-style with information gain and Gini
//! impurity, recursive splitting, pruning, feature importance, and text
//! visualization.
//!
//! Pure Rust — no external ML dependencies.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ── Split Criterion ─────────────────────────────────────────────

/// Criterion used to evaluate feature splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Criterion {
    /// Information gain (ID3/C4.5).
    InformationGain,
    /// Gini impurity (CART).
    Gini,
}

// ── Tree Node ───────────────────────────────────────────────────

/// Internal representation of a decision tree node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Leaf node with class prediction and sample count.
    Leaf {
        class: usize,
        n_samples: usize,
        class_counts: Vec<usize>,
    },
    /// Internal split node.
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
        n_samples: usize,
        impurity: f64,
    },
}

impl TreeNode {
    fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 0,
            TreeNode::Split { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    fn n_leaves(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Split { left, right, .. } => {
                left.n_leaves() + right.n_leaves()
            }
        }
    }
}

// ── Configuration ───────────────────────────────────────────────

/// Configuration for decision tree building.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    /// Maximum depth of the tree (None = unlimited).
    pub max_depth: Option<usize>,
    /// Minimum samples required to split.
    pub min_samples_split: usize,
    /// Minimum samples in a leaf.
    pub min_samples_leaf: usize,
    /// Split criterion.
    pub criterion: Criterion,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: Criterion::InformationGain,
        }
    }
}

// ── Decision Tree ───────────────────────────────────────────────

/// Decision tree classifier supporting ID3/C4.5 and CART criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    config: TreeConfig,
    root: Option<TreeNode>,
    n_features: usize,
    n_classes: usize,
    feature_importances: Vec<f64>,
}

impl DecisionTree {
    /// Create a new decision tree with the given configuration.
    pub fn new(config: TreeConfig) -> Self {
        Self {
            config,
            root: None,
            n_features: 0,
            n_classes: 0,
            feature_importances: vec![],
        }
    }

    /// Fit the tree to training data. Features is a slice of sample vectors,
    /// labels are 0-based class indices.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[usize]) {
        assert_eq!(features.len(), labels.len(), "feature/label mismatch");
        assert!(!features.is_empty(), "empty training set");
        self.n_features = features[0].len();
        self.n_classes = labels.iter().copied().max().unwrap_or(0) + 1;
        self.feature_importances = vec![0.0; self.n_features];

        let indices: Vec<usize> = (0..features.len()).collect();
        self.root = Some(self.build_node(features, labels, &indices, 0));

        // Normalize feature importances
        let total: f64 = self.feature_importances.iter().sum();
        if total > 0.0 {
            for v in &mut self.feature_importances {
                *v /= total;
            }
        }
    }

    fn build_node(
        &mut self,
        features: &[Vec<f64>],
        labels: &[usize],
        indices: &[usize],
        depth: usize,
    ) -> TreeNode {
        let n = indices.len();
        let class_counts = self.count_classes(labels, indices);
        let majority_class = class_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| **c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Stopping conditions
        if n < self.config.min_samples_split
            || self.config.max_depth.is_some_and(|d| depth >= d)
            || class_counts.iter().filter(|c| **c > 0).count() <= 1
        {
            return TreeNode::Leaf {
                class: majority_class,
                n_samples: n,
                class_counts,
            };
        }

        // Find best split
        let parent_impurity = self.compute_impurity(&class_counts, n);
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left = vec![];
        let mut best_right = vec![];

        for feat_idx in 0..self.n_features {
            let mut values: Vec<(f64, usize)> = indices
                .iter()
                .map(|i| (features[*i][feat_idx], labels[*i]))
                .collect();
            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            for window in values.windows(2) {
                if (window[0].0 - window[1].0).abs() < f64::EPSILON {
                    continue;
                }
                let threshold = (window[0].0 + window[1].0) / 2.0;
                let left_idx: Vec<usize> = indices
                    .iter()
                    .filter(|&&i| features[i][feat_idx] <= threshold)
                    .copied()
                    .collect();
                let right_idx: Vec<usize> = indices
                    .iter()
                    .filter(|&&i| features[i][feat_idx] > threshold)
                    .copied()
                    .collect();

                if left_idx.len() < self.config.min_samples_leaf
                    || right_idx.len() < self.config.min_samples_leaf
                {
                    continue;
                }

                let left_counts = self.count_classes(labels, &left_idx);
                let right_counts = self.count_classes(labels, &right_idx);
                let left_imp = self.compute_impurity(&left_counts, left_idx.len());
                let right_imp = self.compute_impurity(&right_counts, right_idx.len());

                let gain = parent_impurity
                    - (left_idx.len() as f64 / n as f64) * left_imp
                    - (right_idx.len() as f64 / n as f64) * right_imp;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                    best_left = left_idx;
                    best_right = right_idx;
                }
            }
        }

        if best_gain < 0.0 || best_left.is_empty() || best_right.is_empty() {
            return TreeNode::Leaf {
                class: majority_class,
                n_samples: n,
                class_counts,
            };
        }

        // Track feature importance
        self.feature_importances[best_feature] += best_gain * n as f64;

        let left_node = self.build_node(features, labels, &best_left, depth + 1);
        let right_node = self.build_node(features, labels, &best_right, depth + 1);

        TreeNode::Split {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_node),
            right: Box::new(right_node),
            n_samples: n,
            impurity: parent_impurity,
        }
    }

    fn count_classes(&self, labels: &[usize], indices: &[usize]) -> Vec<usize> {
        let mut counts = vec![0usize; self.n_classes];
        for &i in indices {
            counts[labels[i]] += 1;
        }
        counts
    }

    fn compute_impurity(&self, class_counts: &[usize], total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        match self.config.criterion {
            Criterion::InformationGain => {
                let mut entropy = 0.0;
                for &c in class_counts {
                    if c > 0 {
                        let p = c as f64 / total as f64;
                        entropy -= p * p.ln();
                    }
                }
                entropy
            }
            Criterion::Gini => {
                let mut gini = 1.0;
                for &c in class_counts {
                    let p = c as f64 / total as f64;
                    gini -= p * p;
                }
                gini
            }
        }
    }

    /// Predict the class label for a single sample.
    pub fn predict(&self, sample: &[f64]) -> usize {
        let root = self.root.as_ref().expect("tree not fitted");
        self.predict_node(root, sample)
    }

    fn predict_node(&self, node: &TreeNode, sample: &[f64]) -> usize {
        match node {
            TreeNode::Leaf { class, .. } => *class,
            TreeNode::Split { feature_index, threshold, left, right, .. } => {
                if sample[*feature_index] <= *threshold {
                    self.predict_node(left, sample)
                } else {
                    self.predict_node(right, sample)
                }
            }
        }
    }

    /// Predict classes for multiple samples.
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Predict with class probabilities.
    pub fn predict_proba(&self, sample: &[f64]) -> Vec<f64> {
        let root = self.root.as_ref().expect("tree not fitted");
        let counts = self.leaf_counts(root, sample);
        let total: usize = counts.iter().sum();
        counts.iter().map(|c| *c as f64 / total as f64).collect()
    }

    fn leaf_counts(&self, node: &TreeNode, sample: &[f64]) -> Vec<usize> {
        match node {
            TreeNode::Leaf { class_counts, .. } => class_counts.clone(),
            TreeNode::Split { feature_index, threshold, left, right, .. } => {
                if sample[*feature_index] <= *threshold {
                    self.leaf_counts(left, sample)
                } else {
                    self.leaf_counts(right, sample)
                }
            }
        }
    }

    /// Return feature importance scores (normalized, sums to 1.0).
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    /// Return the depth of the tree.
    pub fn depth(&self) -> usize {
        self.root.as_ref().map(|n| n.depth()).unwrap_or(0)
    }

    /// Return number of leaf nodes.
    pub fn n_leaves(&self) -> usize {
        self.root.as_ref().map(|n| n.n_leaves()).unwrap_or(0)
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

    /// Render the tree as a text visualization.
    pub fn to_text(&self) -> String {
        let mut buf = String::new();
        if let Some(root) = &self.root {
            self.render_node(root, &mut buf, "", true);
        }
        buf
    }

    fn render_node(&self, node: &TreeNode, buf: &mut String, prefix: &str, is_last: bool) {
        let connector = if prefix.is_empty() { "" } else if is_last { "`-- " } else { "|-- " };
        match node {
            TreeNode::Leaf { class, n_samples, .. } => {
                buf.push_str(&format!("{}{}class={} (n={})\n", prefix, connector, class, n_samples));
            }
            TreeNode::Split { feature_index, threshold, left, right, n_samples, .. } => {
                buf.push_str(&format!(
                    "{}{}feature[{}] <= {:.4} (n={})\n",
                    prefix, connector, feature_index, threshold, n_samples
                ));
                let new_prefix = if prefix.is_empty() {
                    String::new()
                } else if is_last {
                    format!("{}    ", prefix)
                } else {
                    format!("{}|   ", prefix)
                };
                self.render_node(left, buf, &new_prefix, false);
                self.render_node(right, buf, &new_prefix, true);
            }
        }
    }

    /// Serialize the tree to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize a tree from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Return the number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Return the number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
}

impl fmt::Display for DecisionTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DecisionTree(depth={}, leaves={}, features={}, classes={})",
            self.depth(),
            self.n_leaves(),
            self.n_features,
            self.n_classes
        )
    }
}

// ── Standalone impurity functions ───────────────────────────────

/// Compute entropy of a label distribution.
pub fn entropy(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut e = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / total as f64;
            e -= p * p.ln();
        }
    }
    e
}

/// Compute Gini impurity of a label distribution.
pub fn gini_impurity(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let mut gini = 1.0;
    for &c in counts {
        let p = c as f64 / total as f64;
        gini -= p * p;
    }
    gini
}

/// Compute information gain for a split.
pub fn information_gain(
    parent_counts: &[usize],
    left_counts: &[usize],
    right_counts: &[usize],
) -> f64 {
    let total: usize = parent_counts.iter().sum();
    let left_n: usize = left_counts.iter().sum();
    let right_n: usize = right_counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    entropy(parent_counts)
        - (left_n as f64 / total as f64) * entropy(left_counts)
        - (right_n as f64 / total as f64) * entropy(right_counts)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        let features = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let labels = vec![0, 1, 1, 0];
        (features, labels)
    }

    fn simple_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        let features = vec![
            vec![1.0, 2.0], vec![2.0, 3.0], vec![3.0, 4.0],
            vec![8.0, 7.0], vec![9.0, 8.0], vec![10.0, 9.0],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        (features, labels)
    }

    #[test]
    fn test_entropy_pure() {
        let e = entropy(&[10, 0]);
        assert!(e.abs() < 1e-10);
    }

    #[test]
    fn test_entropy_uniform() {
        let e = entropy(&[5, 5]);
        assert!((e - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_gini_pure() {
        let g = gini_impurity(&[10, 0]);
        assert!(g.abs() < 1e-10);
    }

    #[test]
    fn test_gini_uniform() {
        let g = gini_impurity(&[5, 5]);
        assert!((g - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_information_gain_calc() {
        let parent = vec![5, 5];
        let left = vec![5, 0];
        let right = vec![0, 5];
        let ig = information_gain(&parent, &left, &right);
        assert!((ig - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_fit_and_predict() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        assert_eq!(tree.predict(&[2.0, 3.0]), 0);
        assert_eq!(tree.predict(&[9.0, 8.0]), 1);
    }

    #[test]
    fn test_perfect_accuracy() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let acc = tree.accuracy(&feats, &labs);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_xor_dataset() {
        let (feats, labs) = xor_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let acc = tree.accuracy(&feats, &labs);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_depth_pruning() {
        let (feats, labs) = xor_data();
        let config = TreeConfig { max_depth: Some(1), ..Default::default() };
        let mut tree = DecisionTree::new(config);
        tree.fit(&feats, &labs);
        assert!(tree.depth() <= 1);
    }

    #[test]
    fn test_min_samples_split() {
        let (feats, labs) = simple_data();
        let config = TreeConfig { min_samples_split: 10, ..Default::default() };
        let mut tree = DecisionTree::new(config);
        tree.fit(&feats, &labs);
        // Should be just a leaf node (can't split with < 10 samples)
        assert_eq!(tree.depth(), 0);
    }

    #[test]
    fn test_gini_criterion() {
        let (feats, labs) = simple_data();
        let config = TreeConfig { criterion: Criterion::Gini, ..Default::default() };
        let mut tree = DecisionTree::new(config);
        tree.fit(&feats, &labs);
        let acc = tree.accuracy(&feats, &labs);
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_predict_proba() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let proba = tree.predict_proba(&[2.0, 3.0]);
        assert_eq!(proba.len(), 2);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_importances() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let imp = tree.feature_importances();
        assert_eq!(imp.len(), 2);
        let sum: f64 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tree_depth() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        assert!(tree.depth() >= 1);
    }

    #[test]
    fn test_n_leaves() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        assert!(tree.n_leaves() >= 2);
    }

    #[test]
    fn test_to_text() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let text = tree.to_text();
        assert!(!text.is_empty());
        assert!(text.contains("feature["));
    }

    #[test]
    fn test_serialization() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);

        let json = tree.to_json().unwrap();
        let tree2 = DecisionTree::from_json(&json).unwrap();
        assert_eq!(tree.predict(&[2.0, 3.0]), tree2.predict(&[2.0, 3.0]));
        assert_eq!(tree.predict(&[9.0, 8.0]), tree2.predict(&[9.0, 8.0]));
    }

    #[test]
    fn test_display() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let s = format!("{}", tree);
        assert!(s.contains("DecisionTree"));
    }

    #[test]
    fn test_predict_batch() {
        let (feats, labs) = simple_data();
        let mut tree = DecisionTree::new(TreeConfig::default());
        tree.fit(&feats, &labs);
        let preds = tree.predict_batch(&feats);
        assert_eq!(preds, labs);
    }

    #[test]
    fn test_min_samples_leaf() {
        let (feats, labs) = simple_data();
        let config = TreeConfig { min_samples_leaf: 3, ..Default::default() };
        let mut tree = DecisionTree::new(config);
        tree.fit(&feats, &labs);
        // With min_samples_leaf = 3, each leaf has >= 3 samples
        assert!(tree.n_leaves() >= 1);
    }
}
