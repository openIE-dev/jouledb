//! Feature Engineering — one-hot encoding, label encoding, binning,
//! polynomial features, interaction terms, and target encoding for
//! ML preprocessing.
//!
//! Pure Rust, std-only. All operations use f64.

use std::collections::HashMap;
use std::fmt;

// ── Label Encoder ───────────────────────────────────────────────

/// Maps categorical string values to integer labels and back.
#[derive(Debug, Clone)]
pub struct LabelEncoder {
    label_to_idx: HashMap<String, usize>,
    idx_to_label: Vec<String>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self {
            label_to_idx: HashMap::new(),
            idx_to_label: Vec::new(),
        }
    }

    /// Fit encoder on a slice of string labels.
    pub fn fit(&mut self, labels: &[&str]) {
        self.label_to_idx.clear();
        self.idx_to_label.clear();
        let mut unique: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        unique.sort();
        unique.dedup();
        for (i, label) in unique.into_iter().enumerate() {
            self.label_to_idx.insert(label.clone(), i);
            self.idx_to_label.push(label);
        }
    }

    pub fn num_classes(&self) -> usize {
        self.idx_to_label.len()
    }

    /// Encode a single label.
    pub fn encode(&self, label: &str) -> Option<usize> {
        self.label_to_idx.get(label).copied()
    }

    /// Decode an index back to a label.
    pub fn decode(&self, idx: usize) -> Option<&str> {
        self.idx_to_label.get(idx).map(|s| s.as_str())
    }

    /// Encode a batch of labels.
    pub fn encode_batch(&self, labels: &[&str]) -> Vec<Option<usize>> {
        labels.iter().map(|l| self.encode(l)).collect()
    }

    /// Decode a batch of indices.
    pub fn decode_batch(&self, indices: &[usize]) -> Vec<Option<&str>> {
        indices.iter().map(|i| self.decode(*i)).collect()
    }
}

impl fmt::Display for LabelEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LabelEncoder(classes={})", self.num_classes())
    }
}

// ── One-Hot Encoder ─────────────────────────────────────────────

/// One-hot encodes categorical integer values into binary vectors.
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    num_categories: usize,
    drop_first: bool,
}

impl OneHotEncoder {
    pub fn new(num_categories: usize) -> Self {
        Self { num_categories, drop_first: false }
    }

    pub fn with_drop_first(mut self, drop: bool) -> Self {
        self.drop_first = drop;
        self
    }

    /// Width of the encoded vector.
    pub fn encoded_width(&self) -> usize {
        if self.drop_first {
            self.num_categories.saturating_sub(1)
        } else {
            self.num_categories
        }
    }

    /// Encode a single category index into a binary vector.
    pub fn encode(&self, category: usize) -> Vec<f64> {
        let width = self.encoded_width();
        let mut vec = vec![0.0; width];
        let idx = if self.drop_first {
            if category == 0 {
                return vec; // reference category
            }
            category - 1
        } else {
            category
        };
        if idx < width {
            vec[idx] = 1.0;
        }
        vec
    }

    /// Encode a batch of category indices.
    pub fn encode_batch(&self, categories: &[usize]) -> Vec<Vec<f64>> {
        categories.iter().map(|c| self.encode(*c)).collect()
    }

    /// Decode a one-hot vector back to category index.
    pub fn decode(&self, encoded: &[f64]) -> usize {
        let offset = if self.drop_first { 1 } else { 0 };
        for (i, &v) in encoded.iter().enumerate() {
            if v > 0.5 {
                return i + offset;
            }
        }
        0
    }
}

impl fmt::Display for OneHotEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OneHotEncoder(cats={}, drop_first={})", self.num_categories, self.drop_first)
    }
}

// ── Binner ──────────────────────────────────────────────────────

/// Binning strategy for continuous features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinStrategy {
    Uniform,
    Quantile,
}

impl fmt::Display for BinStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform => write!(f, "uniform"),
            Self::Quantile => write!(f, "quantile"),
        }
    }
}

/// Discretizes continuous features into bins.
#[derive(Debug, Clone)]
pub struct Binner {
    num_bins: usize,
    strategy: BinStrategy,
    edges: Vec<f64>,
}

impl Binner {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins: num_bins.max(2),
            strategy: BinStrategy::Uniform,
            edges: Vec::new(),
        }
    }

    pub fn with_strategy(mut self, strategy: BinStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Fit bin edges on training data.
    pub fn fit(&mut self, values: &[f64]) {
        match self.strategy {
            BinStrategy::Uniform => {
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let step = (max - min) / self.num_bins as f64;
                self.edges = (0..=self.num_bins).map(|i| min + step * i as f64).collect();
            }
            BinStrategy::Quantile => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.edges = (0..=self.num_bins)
                    .map(|i| {
                        let pct = i as f64 / self.num_bins as f64;
                        let rank = pct * (sorted.len() - 1) as f64;
                        let lo = rank.floor() as usize;
                        let hi = (lo + 1).min(sorted.len() - 1);
                        let frac = rank - lo as f64;
                        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
                    })
                    .collect();
            }
        }
    }

    pub fn is_fitted(&self) -> bool {
        !self.edges.is_empty()
    }

    /// Assign a value to a bin index.
    pub fn transform(&self, value: f64) -> usize {
        if self.edges.len() < 2 {
            return 0;
        }
        for i in 1..self.edges.len() {
            if value <= self.edges[i] {
                return i - 1;
            }
        }
        self.num_bins - 1
    }

    /// Bin a batch of values.
    pub fn transform_batch(&self, values: &[f64]) -> Vec<usize> {
        values.iter().map(|v| self.transform(*v)).collect()
    }

    /// Return the edges.
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }
}

impl fmt::Display for Binner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Binner(bins={}, strategy={})", self.num_bins, self.strategy)
    }
}

// ── Polynomial Features ─────────────────────────────────────────

/// Generates polynomial feature combinations up to a given degree.
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self {
        Self {
            degree: degree.max(1),
            include_bias: true,
            interaction_only: false,
        }
    }

    pub fn with_bias(mut self, include: bool) -> Self {
        self.include_bias = include;
        self
    }

    pub fn with_interaction_only(mut self, interaction: bool) -> Self {
        self.interaction_only = interaction;
        self
    }

    /// Compute polynomial features for a single row.
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        let mut result = Vec::new();
        if self.include_bias {
            result.push(1.0);
        }

        if self.interaction_only {
            // Original features
            result.extend_from_slice(features);
            // All pairs, triples, etc. up to degree
            self.generate_interactions(features, &mut result);
        } else {
            // All combinations with repetition up to degree
            self.generate_poly(features, &mut result);
        }
        result
    }

    /// Transform a batch of rows.
    pub fn transform_batch(&self, rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
        rows.iter().map(|r| self.transform(r)).collect()
    }

    fn generate_poly(&self, features: &[f64], result: &mut Vec<f64>) {
        let n = features.len();
        // Generate all combinations with repetition of length 1..=degree
        for d in 1..=self.degree {
            let combos = combinations_with_rep(n, d);
            for combo in combos {
                let val: f64 = combo.iter().map(|i| features[*i]).product();
                result.push(val);
            }
        }
    }

    fn generate_interactions(&self, features: &[f64], result: &mut Vec<f64>) {
        let n = features.len();
        // Only distinct feature combinations (no repeated indices)
        for d in 2..=self.degree {
            let combos = combinations_no_rep(n, d);
            for combo in combos {
                let val: f64 = combo.iter().map(|i| features[*i]).product();
                result.push(val);
            }
        }
    }

    /// Number of output features for a given input dimension.
    pub fn output_dim(&self, input_dim: usize) -> usize {
        self.transform(&vec![1.0; input_dim]).len()
    }
}

impl fmt::Display for PolynomialFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PolynomialFeatures(degree={}, bias={}, interact_only={})",
               self.degree, self.include_bias, self.interaction_only)
    }
}

/// Generate all combinations with repetition of length k from 0..n.
fn combinations_with_rep(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = vec![0usize; k];
    loop {
        result.push(current.clone());
        // Increment rightmost, carry as needed
        let mut pos = k;
        while pos > 0 {
            pos -= 1;
            current[pos] += 1;
            if current[pos] < n {
                // Fill subsequent positions to maintain non-decreasing order
                for j in (pos + 1)..k {
                    current[j] = current[pos];
                }
                break;
            }
        }
        if pos == 0 && current[0] >= n {
            break;
        }
    }
    result
}

/// Generate all combinations without repetition of length k from 0..n.
fn combinations_no_rep(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k > n {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut current: Vec<usize> = (0..k).collect();
    loop {
        result.push(current.clone());
        let mut pos = k;
        while pos > 0 {
            pos -= 1;
            current[pos] += 1;
            if current[pos] <= n - k + pos {
                for j in (pos + 1)..k {
                    current[j] = current[j - 1] + 1;
                }
                break;
            }
        }
        if pos == 0 && current[0] > n - k {
            break;
        }
    }
    result
}

// ── Interaction Terms ───────────────────────────────────────────

/// Compute all pairwise interaction terms for a feature vector.
pub fn pairwise_interactions(features: &[f64]) -> Vec<f64> {
    let n = features.len();
    let mut interactions = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            interactions.push(features[i] * features[j]);
        }
    }
    interactions
}

/// Append pairwise interactions to the original feature vector.
pub fn append_interactions(features: &[f64]) -> Vec<f64> {
    let mut result = features.to_vec();
    result.extend(pairwise_interactions(features));
    result
}

// ── Target Encoder ──────────────────────────────────────────────

/// Target (mean) encoding: replace category with mean target value.
#[derive(Debug, Clone)]
pub struct TargetEncoder {
    mappings: HashMap<String, f64>,
    global_mean: f64,
    smoothing: f64,
}

impl TargetEncoder {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
            global_mean: 0.0,
            smoothing: 1.0,
        }
    }

    pub fn with_smoothing(mut self, s: f64) -> Self {
        self.smoothing = s.max(0.0);
        self
    }

    /// Fit on categorical values and corresponding targets.
    pub fn fit(&mut self, categories: &[&str], targets: &[f64]) {
        let n = categories.len().min(targets.len());
        self.global_mean = if n > 0 {
            targets[..n].iter().sum::<f64>() / n as f64
        } else {
            0.0
        };

        let mut sums: HashMap<String, f64> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for i in 0..n {
            let key = categories[i].to_string();
            *sums.entry(key.clone()).or_insert(0.0) += targets[i];
            *counts.entry(key).or_insert(0) += 1;
        }

        self.mappings.clear();
        for (cat, count) in &counts {
            let cat_mean = sums[cat] / *count as f64;
            // Smoothed: weighted average of category mean and global mean
            let smoothed = (*count as f64 * cat_mean + self.smoothing * self.global_mean)
                / (*count as f64 + self.smoothing);
            self.mappings.insert(cat.clone(), smoothed);
        }
    }

    /// Encode a single category.
    pub fn encode(&self, category: &str) -> f64 {
        self.mappings.get(category).copied().unwrap_or(self.global_mean)
    }

    /// Encode a batch.
    pub fn encode_batch(&self, categories: &[&str]) -> Vec<f64> {
        categories.iter().map(|c| self.encode(c)).collect()
    }
}

impl fmt::Display for TargetEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TargetEncoder(cats={}, smooth={:.1})", self.mappings.len(), self.smoothing)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_encoder_fit() {
        let mut enc = LabelEncoder::new();
        enc.fit(&["cat", "dog", "bird", "cat"]);
        assert_eq!(enc.num_classes(), 3);
    }

    #[test]
    fn label_encoder_roundtrip() {
        let mut enc = LabelEncoder::new();
        enc.fit(&["a", "b", "c"]);
        let idx = enc.encode("b").unwrap();
        assert_eq!(enc.decode(idx), Some("b"));
    }

    #[test]
    fn label_encoder_unknown() {
        let mut enc = LabelEncoder::new();
        enc.fit(&["x", "y"]);
        assert_eq!(enc.encode("z"), None);
    }

    #[test]
    fn label_encoder_display() {
        let enc = LabelEncoder::new();
        assert!(format!("{}", enc).contains("classes=0"));
    }

    #[test]
    fn onehot_encode() {
        let enc = OneHotEncoder::new(4);
        let v = enc.encode(2);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn onehot_drop_first() {
        let enc = OneHotEncoder::new(3).with_drop_first(true);
        assert_eq!(enc.encoded_width(), 2);
        let v = enc.encode(0);
        assert_eq!(v, vec![0.0, 0.0]);
        let v = enc.encode(1);
        assert_eq!(v, vec![1.0, 0.0]);
    }

    #[test]
    fn onehot_decode() {
        let enc = OneHotEncoder::new(4);
        let v = enc.encode(3);
        assert_eq!(enc.decode(&v), 3);
    }

    #[test]
    fn binner_uniform() {
        let mut b = Binner::new(4);
        b.fit(&[0.0, 25.0, 50.0, 75.0, 100.0]);
        assert_eq!(b.transform(12.0), 0);
        assert_eq!(b.transform(50.0), 1);
        assert_eq!(b.transform(99.0), 3);
    }

    #[test]
    fn binner_quantile() {
        let mut b = Binner::new(4).with_strategy(BinStrategy::Quantile);
        b.fit(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(b.is_fitted());
        assert_eq!(b.edges().len(), 5);
    }

    #[test]
    fn binner_display() {
        let b = Binner::new(5).with_strategy(BinStrategy::Quantile);
        assert!(format!("{}", b).contains("quantile"));
    }

    #[test]
    fn polynomial_degree2() {
        let pf = PolynomialFeatures::new(2).with_bias(false);
        let result = pf.transform(&[2.0, 3.0]);
        // degree 1: [2, 3], degree 2: [4, 6, 9]
        assert_eq!(result, vec![2.0, 3.0, 4.0, 6.0, 9.0]);
    }

    #[test]
    fn polynomial_with_bias() {
        let pf = PolynomialFeatures::new(1).with_bias(true);
        let result = pf.transform(&[5.0, 7.0]);
        assert_eq!(result[0], 1.0); // bias
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 7.0);
    }

    #[test]
    fn polynomial_interaction_only() {
        let pf = PolynomialFeatures::new(2).with_bias(false).with_interaction_only(true);
        let result = pf.transform(&[2.0, 3.0, 5.0]);
        // original: [2, 3, 5], interactions: [6, 10, 15]
        assert_eq!(result.len(), 6);
        assert!(result.contains(&6.0));
        assert!(result.contains(&15.0));
    }

    #[test]
    fn polynomial_output_dim() {
        let pf = PolynomialFeatures::new(2).with_bias(true);
        let dim = pf.output_dim(3);
        // 1 + 3 + 6 = 10 (bias + degree1 + degree2-with-rep)
        assert_eq!(dim, 10);
    }

    #[test]
    fn pairwise_interactions_basic() {
        let feat = vec![2.0, 3.0, 5.0];
        let inter = pairwise_interactions(&feat);
        assert_eq!(inter, vec![6.0, 10.0, 15.0]);
    }

    #[test]
    fn append_interactions_length() {
        let feat = vec![1.0, 2.0, 3.0, 4.0];
        let result = append_interactions(&feat);
        assert_eq!(result.len(), 4 + 6); // 4 original + C(4,2)=6
    }

    #[test]
    fn target_encoder_basic() {
        let mut enc = TargetEncoder::new().with_smoothing(0.0);
        enc.fit(&["a", "a", "b", "b"], &[1.0, 3.0, 10.0, 20.0]);
        assert!((enc.encode("a") - 2.0).abs() < 1e-9);
        assert!((enc.encode("b") - 15.0).abs() < 1e-9);
    }

    #[test]
    fn target_encoder_smoothing() {
        let mut enc = TargetEncoder::new().with_smoothing(2.0);
        enc.fit(&["a", "a", "b"], &[10.0, 10.0, 0.0]);
        // Global mean = 20/3 ≈ 6.667
        // "a" smoothed: (2*10 + 2*6.667)/(2+2) ≈ 8.333
        let a_val = enc.encode("a");
        assert!(a_val > 6.0 && a_val < 11.0);
    }

    #[test]
    fn target_encoder_unknown_category() {
        let mut enc = TargetEncoder::new();
        enc.fit(&["x", "y"], &[1.0, 2.0]);
        let val = enc.encode("z");
        assert!((val - 1.5).abs() < 1.0); // should be near global mean
    }

    #[test]
    fn onehot_batch() {
        let enc = OneHotEncoder::new(3);
        let batch = enc.encode_batch(&[0, 1, 2]);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[1], vec![0.0, 1.0, 0.0]);
    }
}
