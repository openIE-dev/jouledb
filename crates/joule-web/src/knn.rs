//! K-Nearest Neighbors — classification and regression with brute-force
//! and KD-tree indexing, multiple distance metrics, and cross-validation.
//!
//! Pure Rust — no external ML or numeric dependencies.

use std::fmt;

// ── Distance Metrics ────────────────────────────────────────────

/// Distance metric for KNN computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
}

impl DistanceMetric {
    /// Compute distance between two points of equal dimensionality.
    pub fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "dimension mismatch");
        match self {
            Self::Euclidean => {
                let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                sum.sqrt()
            }
            Self::Manhattan => {
                a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
            }
            Self::Cosine => {
                let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
                let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
                    1.0
                } else {
                    1.0 - dot / (norm_a * norm_b)
                }
            }
        }
    }
}

// ── KD-Tree ─────────────────────────────────────────────────────

/// A node in a KD-tree for spatial partitioning.
#[derive(Debug, Clone)]
enum KdNode {
    Leaf {
        indices: Vec<usize>,
    },
    Split {
        axis: usize,
        median: f64,
        left: Box<KdNode>,
        right: Box<KdNode>,
    },
}

/// KD-tree spatial index for fast nearest-neighbor queries.
#[derive(Debug, Clone)]
pub struct KdTree {
    root: Option<KdNode>,
    points: Vec<Vec<f64>>,
    dims: usize,
    leaf_size: usize,
}

impl KdTree {
    /// Build a KD-tree from the given points.
    pub fn build(points: &[Vec<f64>], leaf_size: usize) -> Self {
        if points.is_empty() {
            return Self { root: None, points: vec![], dims: 0, leaf_size };
        }
        let dims = points[0].len();
        let stored: Vec<Vec<f64>> = points.to_vec();
        let indices: Vec<usize> = (0..points.len()).collect();
        let root = Self::build_node(&stored, &indices, 0, dims, leaf_size);
        Self { root: Some(root), points: stored, dims, leaf_size }
    }

    fn build_node(
        points: &[Vec<f64>],
        indices: &[usize],
        depth: usize,
        dims: usize,
        leaf_size: usize,
    ) -> KdNode {
        if indices.len() <= leaf_size {
            return KdNode::Leaf { indices: indices.to_vec() };
        }
        let axis = depth % dims;
        let mut sorted = indices.to_vec();
        sorted.sort_by(|a, b| {
            points[*a][axis].partial_cmp(&points[*b][axis]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = sorted.len() / 2;
        let median = points[sorted[mid]][axis];
        let left_indices = &sorted[..mid];
        let right_indices = &sorted[mid..];
        if left_indices.is_empty() || right_indices.is_empty() {
            return KdNode::Leaf { indices: sorted };
        }
        KdNode::Split {
            axis,
            median,
            left: Box::new(Self::build_node(points, left_indices, depth + 1, dims, leaf_size)),
            right: Box::new(Self::build_node(points, right_indices, depth + 1, dims, leaf_size)),
        }
    }

    /// Find the k nearest neighbor indices for a query point using Euclidean distance.
    pub fn query(&self, point: &[f64], k: usize) -> Vec<(usize, f64)> {
        assert_eq!(point.len(), self.dims, "query dimension mismatch");
        let mut best: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
        if let Some(root) = &self.root {
            self.search_node(root, point, k, &mut best, 0);
        }
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        best.truncate(k);
        best
    }

    fn search_node(
        &self,
        node: &KdNode,
        point: &[f64],
        k: usize,
        best: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        match node {
            KdNode::Leaf { indices } => {
                for &idx in indices {
                    let dist = euclidean_dist(point, &self.points[idx]);
                    Self::insert_candidate(best, k, idx, dist);
                }
            }
            KdNode::Split { axis, median, left, right } => {
                let (first, second) = if point[*axis] <= *median {
                    (left, right)
                } else {
                    (right, left)
                };
                self.search_node(first, point, k, best, depth + 1);
                let plane_dist = (point[*axis] - *median).abs();
                let worst = if best.len() < k {
                    f64::INFINITY
                } else {
                    best.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max)
                };
                if plane_dist < worst {
                    self.search_node(second, point, k, best, depth + 1);
                }
            }
        }
    }

    fn insert_candidate(best: &mut Vec<(usize, f64)>, k: usize, idx: usize, dist: f64) {
        if best.len() < k {
            best.push((idx, dist));
        } else {
            let (worst_pos, worst_dist) = best
                .iter()
                .enumerate()
                .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, v.1))
                .unwrap();
            if dist < worst_dist {
                best[worst_pos] = (idx, dist);
            }
        }
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// ── Search Strategy ─────────────────────────────────────────────

/// Index strategy for neighbor search.
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Brute-force linear scan.
    BruteForce,
    /// KD-tree accelerated search.
    KdTreeIndex(KdTree),
}

// ── KNN Classifier ──────────────────────────────────────────────

/// K-Nearest Neighbors classifier.
#[derive(Debug, Clone)]
pub struct KnnClassifier {
    k: usize,
    metric: DistanceMetric,
    strategy: SearchStrategy,
    train_features: Vec<Vec<f64>>,
    train_labels: Vec<usize>,
    n_classes: usize,
}

impl KnnClassifier {
    /// Create a new KNN classifier.
    pub fn new(k: usize, metric: DistanceMetric) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            k,
            metric,
            strategy: SearchStrategy::BruteForce,
            train_features: vec![],
            train_labels: vec![],
            n_classes: 0,
        }
    }

    /// Fit on training data. Labels should be 0-based class indices.
    pub fn fit(&mut self, features: &[Vec<f64>], labels: &[usize]) {
        assert_eq!(features.len(), labels.len(), "feature/label count mismatch");
        assert!(!features.is_empty(), "empty training set");
        self.train_features = features.to_vec();
        self.train_labels = labels.to_vec();
        self.n_classes = labels.iter().copied().max().unwrap_or(0) + 1;
    }

    /// Enable KD-tree indexing (only meaningful for Euclidean distance).
    pub fn build_index(&mut self, leaf_size: usize) {
        let tree = KdTree::build(&self.train_features, leaf_size);
        self.strategy = SearchStrategy::KdTreeIndex(tree);
    }

    /// Predict class for a single sample using majority vote.
    pub fn predict(&self, sample: &[f64]) -> usize {
        let neighbors = self.find_neighbors(sample);
        self.majority_vote(&neighbors)
    }

    /// Predict classes for multiple samples.
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<usize> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    /// Predict with class probabilities (vote fractions).
    pub fn predict_proba(&self, sample: &[f64]) -> Vec<f64> {
        let neighbors = self.find_neighbors(sample);
        let mut counts = vec![0usize; self.n_classes];
        for (idx, _dist) in &neighbors {
            counts[self.train_labels[*idx]] += 1;
        }
        let total = neighbors.len() as f64;
        counts.iter().map(|c| *c as f64 / total).collect()
    }

    fn find_neighbors(&self, sample: &[f64]) -> Vec<(usize, f64)> {
        match &self.strategy {
            SearchStrategy::BruteForce => {
                let mut dists: Vec<(usize, f64)> = self.train_features
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (i, self.metric.distance(sample, f)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(self.k);
                dists
            }
            SearchStrategy::KdTreeIndex(tree) => {
                tree.query(sample, self.k)
            }
        }
    }

    fn majority_vote(&self, neighbors: &[(usize, f64)]) -> usize {
        let mut counts = vec![0usize; self.n_classes];
        for (idx, _dist) in neighbors {
            counts[self.train_labels[*idx]] += 1;
        }
        counts
            .iter()
            .enumerate()
            .max_by_key(|(_i, c)| **c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute accuracy on a test set.
    pub fn accuracy(&self, features: &[Vec<f64>], labels: &[usize]) -> f64 {
        assert_eq!(features.len(), labels.len());
        if features.is_empty() {
            return 0.0;
        }
        let predictions = self.predict_batch(features);
        let correct = predictions.iter().zip(labels).filter(|(p, l)| p == l).count();
        correct as f64 / features.len() as f64
    }

    /// Return current k.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Update k.
    pub fn set_k(&mut self, k: usize) {
        assert!(k > 0);
        self.k = k;
    }
}

impl fmt::Display for KnnClassifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KnnClassifier(k={}, metric={:?}, n_train={})", self.k, self.metric, self.train_features.len())
    }
}

// ── KNN Regressor ───────────────────────────────────────────────

/// K-Nearest Neighbors regressor using weighted average.
#[derive(Debug, Clone)]
pub struct KnnRegressor {
    k: usize,
    metric: DistanceMetric,
    strategy: SearchStrategy,
    train_features: Vec<Vec<f64>>,
    train_targets: Vec<f64>,
    weighted: bool,
}

impl KnnRegressor {
    /// Create a new KNN regressor.
    pub fn new(k: usize, metric: DistanceMetric, weighted: bool) -> Self {
        assert!(k > 0);
        Self {
            k,
            metric,
            strategy: SearchStrategy::BruteForce,
            train_features: vec![],
            train_targets: vec![],
            weighted,
        }
    }

    /// Fit on training data.
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) {
        assert_eq!(features.len(), targets.len());
        assert!(!features.is_empty());
        self.train_features = features.to_vec();
        self.train_targets = targets.to_vec();
    }

    /// Enable KD-tree indexing.
    pub fn build_index(&mut self, leaf_size: usize) {
        let tree = KdTree::build(&self.train_features, leaf_size);
        self.strategy = SearchStrategy::KdTreeIndex(tree);
    }

    /// Predict value for a single sample.
    pub fn predict(&self, sample: &[f64]) -> f64 {
        let neighbors = self.find_neighbors(sample);
        if self.weighted {
            self.weighted_average(&neighbors)
        } else {
            self.simple_average(&neighbors)
        }
    }

    /// Predict values for multiple samples.
    pub fn predict_batch(&self, samples: &[Vec<f64>]) -> Vec<f64> {
        samples.iter().map(|s| self.predict(s)).collect()
    }

    fn find_neighbors(&self, sample: &[f64]) -> Vec<(usize, f64)> {
        match &self.strategy {
            SearchStrategy::BruteForce => {
                let mut dists: Vec<(usize, f64)> = self.train_features
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (i, self.metric.distance(sample, f)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(self.k);
                dists
            }
            SearchStrategy::KdTreeIndex(tree) => {
                tree.query(sample, self.k)
            }
        }
    }

    fn simple_average(&self, neighbors: &[(usize, f64)]) -> f64 {
        let sum: f64 = neighbors.iter().map(|(idx, _)| self.train_targets[*idx]).sum();
        sum / neighbors.len() as f64
    }

    fn weighted_average(&self, neighbors: &[(usize, f64)]) -> f64 {
        let eps = 1e-10;
        let weights: Vec<f64> = neighbors.iter().map(|(_, d)| 1.0 / (d + eps)).collect();
        let total_weight: f64 = weights.iter().sum();
        let weighted_sum: f64 = neighbors
            .iter()
            .zip(weights.iter())
            .map(|((idx, _), w)| self.train_targets[*idx] * w)
            .sum();
        weighted_sum / total_weight
    }

    /// Compute mean squared error on a test set.
    pub fn mse(&self, features: &[Vec<f64>], targets: &[f64]) -> f64 {
        assert_eq!(features.len(), targets.len());
        if features.is_empty() {
            return 0.0;
        }
        let predictions = self.predict_batch(features);
        let sum_sq: f64 = predictions.iter().zip(targets).map(|(p, t)| (p - t).powi(2)).sum();
        sum_sq / features.len() as f64
    }
}

// ── Cross-Validation ────────────────────────────────────────────

/// Split data into k folds for cross-validation.
/// Returns a vec of (train_indices, test_indices) pairs.
pub fn cross_validation_split(n_samples: usize, n_folds: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    assert!(n_folds > 1, "need at least 2 folds");
    assert!(n_samples >= n_folds, "more folds than samples");

    let fold_size = n_samples / n_folds;
    let remainder = n_samples % n_folds;

    let mut folds = Vec::with_capacity(n_folds);
    let mut start = 0;
    for i in 0..n_folds {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        let test_indices: Vec<usize> = (start..end).collect();
        let train_indices: Vec<usize> = (0..start).chain(end..n_samples).collect();
        folds.push((train_indices, test_indices));
        start = end;
    }
    folds
}

/// Suggest a good k value by cross-validating multiple k values.
/// Returns (best_k, best_accuracy) pair.
pub fn select_k(
    features: &[Vec<f64>],
    labels: &[usize],
    k_candidates: &[usize],
    n_folds: usize,
    metric: DistanceMetric,
) -> (usize, f64) {
    assert!(!k_candidates.is_empty());
    let folds = cross_validation_split(features.len(), n_folds);
    let mut best_k = k_candidates[0];
    let mut best_acc = f64::NEG_INFINITY;

    for &k in k_candidates {
        let mut total_acc = 0.0;
        for (train_idx, test_idx) in &folds {
            let train_feats: Vec<Vec<f64>> = train_idx.iter().map(|i| features[*i].clone()).collect();
            let train_labs: Vec<usize> = train_idx.iter().map(|i| labels[*i]).collect();
            let test_feats: Vec<Vec<f64>> = test_idx.iter().map(|i| features[*i].clone()).collect();
            let test_labs: Vec<usize> = test_idx.iter().map(|i| labels[*i]).collect();

            let mut clf = KnnClassifier::new(k, metric);
            clf.fit(&train_feats, &train_labs);
            total_acc += clf.accuracy(&test_feats, &test_labs);
        }
        let avg_acc = total_acc / folds.len() as f64;
        if avg_acc > best_acc {
            best_acc = avg_acc;
            best_k = k;
        }
    }
    (best_k, best_acc)
}

// ── K Selection Heuristic ───────────────────────────────────────

/// Suggest k as sqrt(n_samples), clamped to odd values.
pub fn suggested_k(n_samples: usize) -> usize {
    let k = (n_samples as f64).sqrt().round() as usize;
    let k = k.max(1);
    if k % 2 == 0 { k + 1 } else { k }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn iris_like_data() -> (Vec<Vec<f64>>, Vec<usize>) {
        // Simplified 2D data with 3 clusters
        let features = vec![
            vec![1.0, 1.0], vec![1.1, 1.2], vec![0.9, 0.8], vec![1.2, 1.1],
            vec![5.0, 5.0], vec![5.1, 5.2], vec![4.9, 4.8], vec![5.2, 5.1],
            vec![9.0, 1.0], vec![9.1, 1.2], vec![8.9, 0.8], vec![9.2, 1.1],
        ];
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        (features, labels)
    }

    #[test]
    fn test_euclidean_distance() {
        let d = DistanceMetric::Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let d = DistanceMetric::Manhattan.distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let d = DistanceMetric::Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-10); // orthogonal => distance 1

        let d2 = DistanceMetric::Cosine.distance(&[1.0, 0.0], &[1.0, 0.0]);
        assert!(d2.abs() < 1e-10); // identical => distance 0
    }

    #[test]
    fn test_cosine_zero_vector() {
        let d = DistanceMetric::Cosine.distance(&[0.0, 0.0], &[1.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classifier_basic() {
        let (feats, labs) = iris_like_data();
        let mut clf = KnnClassifier::new(3, DistanceMetric::Euclidean);
        clf.fit(&feats, &labs);

        assert_eq!(clf.predict(&[1.0, 1.0]), 0);
        assert_eq!(clf.predict(&[5.0, 5.0]), 1);
        assert_eq!(clf.predict(&[9.0, 1.0]), 2);
    }

    #[test]
    fn test_classifier_accuracy() {
        let (feats, labs) = iris_like_data();
        let mut clf = KnnClassifier::new(3, DistanceMetric::Euclidean);
        clf.fit(&feats, &labs);

        let acc = clf.accuracy(&feats, &labs);
        assert!(acc > 0.9);
    }

    #[test]
    fn test_predict_proba() {
        let (feats, labs) = iris_like_data();
        let mut clf = KnnClassifier::new(3, DistanceMetric::Euclidean);
        clf.fit(&feats, &labs);

        let proba = clf.predict_proba(&[1.0, 1.0]);
        assert_eq!(proba.len(), 3);
        let sum: f64 = proba.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(proba[0] > 0.5);
    }

    #[test]
    fn test_classifier_kdtree() {
        let (feats, labs) = iris_like_data();
        let mut clf = KnnClassifier::new(3, DistanceMetric::Euclidean);
        clf.fit(&feats, &labs);
        clf.build_index(2);

        assert_eq!(clf.predict(&[1.0, 1.0]), 0);
        assert_eq!(clf.predict(&[5.0, 5.0]), 1);
        assert_eq!(clf.predict(&[9.0, 1.0]), 2);
    }

    #[test]
    fn test_predict_batch() {
        let (feats, labs) = iris_like_data();
        let mut clf = KnnClassifier::new(3, DistanceMetric::Euclidean);
        clf.fit(&feats, &labs);

        let preds = clf.predict_batch(&[vec![1.0, 1.0], vec![5.0, 5.0], vec![9.0, 1.0]]);
        assert_eq!(preds, vec![0, 1, 2]);
    }

    #[test]
    fn test_regressor_simple() {
        let features = vec![
            vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        ];
        let targets = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mut reg = KnnRegressor::new(2, DistanceMetric::Euclidean, false);
        reg.fit(&features, &targets);

        let pred = reg.predict(&[1.5]);
        assert!((pred - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_regressor_weighted() {
        let features = vec![vec![0.0], vec![2.0]];
        let targets = vec![0.0, 10.0];
        let mut reg = KnnRegressor::new(2, DistanceMetric::Euclidean, true);
        reg.fit(&features, &targets);

        // Point at 0.5 is closer to 0.0, so weighted should pull toward 0.0
        let pred = reg.predict(&[0.5]);
        assert!(pred < 5.0);
    }

    #[test]
    fn test_regressor_mse() {
        let features = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![0.0, 1.0, 2.0, 3.0];
        let mut reg = KnnRegressor::new(1, DistanceMetric::Euclidean, false);
        reg.fit(&features, &targets);

        let mse = reg.mse(&features, &targets);
        assert!(mse < 1e-10);
    }

    #[test]
    fn test_cross_validation_split() {
        let folds = cross_validation_split(10, 5);
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.len() + test.len(), 10);
        }
    }

    #[test]
    fn test_cross_validation_coverage() {
        let folds = cross_validation_split(10, 5);
        let mut all_test: Vec<usize> = folds.iter().flat_map(|(_, t)| t.clone()).collect();
        all_test.sort();
        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_test, expected);
    }

    #[test]
    fn test_select_k() {
        let (feats, labs) = iris_like_data();
        // Use 4 folds so each fold mixes classes (12 samples / 4 = 3 per fold),
        // avoiding degenerate splits where an entire class lands in one fold.
        let (best_k, best_acc) = select_k(&feats, &labs, &[1, 3], 4, DistanceMetric::Euclidean);
        assert!(best_k > 0);
        assert!(best_acc > 0.0);
    }

    #[test]
    fn test_suggested_k() {
        assert_eq!(suggested_k(25), 5);
        assert_eq!(suggested_k(100), 11); // sqrt(100)=10, make odd => 11
        assert_eq!(suggested_k(1), 1);
    }

    #[test]
    fn test_kdtree_build_empty() {
        let tree = KdTree::build(&[], 5);
        assert!(tree.root.is_none());
    }

    #[test]
    fn test_kdtree_single_point() {
        let tree = KdTree::build(&[vec![1.0, 2.0]], 5);
        let result = tree.query(&[1.0, 2.0], 1);
        assert_eq!(result.len(), 1);
        assert!(result[0].1 < 1e-10);
    }

    #[test]
    fn test_kdtree_query_k() {
        let points = vec![
            vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0],
            vec![10.0, 10.0], vec![11.0, 10.0],
        ];
        let tree = KdTree::build(&points, 2);
        let result = tree.query(&[0.0, 0.0], 3);
        assert_eq!(result.len(), 3);
        // Closest should be distance 0
        assert!(result[0].1 < 1e-10);
    }

    #[test]
    fn test_set_k() {
        let mut clf = KnnClassifier::new(3, DistanceMetric::Euclidean);
        assert_eq!(clf.k(), 3);
        clf.set_k(5);
        assert_eq!(clf.k(), 5);
    }

    #[test]
    fn test_classifier_display() {
        let clf = KnnClassifier::new(5, DistanceMetric::Manhattan);
        let s = format!("{}", clf);
        assert!(s.contains("k=5"));
        assert!(s.contains("Manhattan"));
    }

    #[test]
    fn test_manhattan_classifier() {
        let (feats, labs) = iris_like_data();
        let mut clf = KnnClassifier::new(3, DistanceMetric::Manhattan);
        clf.fit(&feats, &labs);

        assert_eq!(clf.predict(&[1.0, 1.0]), 0);
        assert_eq!(clf.predict(&[5.0, 5.0]), 1);
    }

    #[test]
    fn test_regressor_batch() {
        let features = vec![vec![0.0], vec![1.0], vec![2.0]];
        let targets = vec![0.0, 1.0, 2.0];
        let mut reg = KnnRegressor::new(1, DistanceMetric::Euclidean, false);
        reg.fit(&features, &targets);

        let preds = reg.predict_batch(&[vec![0.0], vec![1.0], vec![2.0]]);
        assert_eq!(preds.len(), 3);
        for (p, t) in preds.iter().zip(targets.iter()) {
            assert!((p - t).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cross_validation_uneven() {
        let folds = cross_validation_split(7, 3);
        assert_eq!(folds.len(), 3);
        let total_test: usize = folds.iter().map(|(_, t)| t.len()).sum();
        assert_eq!(total_test, 7);
    }
}
