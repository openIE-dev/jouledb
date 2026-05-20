//! ML Data Loader — batched iteration, shuffling, prefetch buffering,
//! train/val/test splits, and stratified sampling for machine-learning
//! data pipelines.
//!
//! Pure Rust, std-only. No external crates.

use std::collections::HashMap;
use std::fmt;

// ── Sample ──────────────────────────────────────────────────────

/// A single data sample with feature vector and optional label.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: Option<usize>,
    pub weight: f64,
}

impl Sample {
    pub fn new(features: Vec<f64>) -> Self {
        Self { features, label: None, weight: 1.0 }
    }

    pub fn with_label(mut self, label: usize) -> Self {
        self.label = Some(label);
        self
    }

    pub fn with_weight(mut self, w: f64) -> Self {
        self.weight = w;
        self
    }

    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sample(dim={}, label={:?}, w={:.3})", self.dim(), self.label, self.weight)
    }
}

// ── Split Ratio ─────────────────────────────────────────────────

/// Ratios for train / validation / test split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitRatio {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

impl SplitRatio {
    pub fn new(train: f64, val: f64, test: f64) -> Self {
        Self { train, val, test }
    }

    /// Standard 80/10/10 split.
    pub fn default_split() -> Self {
        Self { train: 0.8, val: 0.1, test: 0.1 }
    }

    pub fn is_valid(&self) -> bool {
        let total = self.train + self.val + self.test;
        (total - 1.0).abs() < 1e-9 && self.train >= 0.0 && self.val >= 0.0 && self.test >= 0.0
    }
}

impl fmt::Display for SplitRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Split(train={:.0}%, val={:.0}%, test={:.0}%)",
               self.train * 100.0, self.val * 100.0, self.test * 100.0)
    }
}

// ── Dataset ─────────────────────────────────────────────────────

/// A dataset is an ordered collection of samples.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

impl Dataset {
    pub fn new() -> Self {
        Self { samples: Vec::new() }
    }

    pub fn from_samples(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn push(&mut self, sample: Sample) {
        self.samples.push(sample);
    }

    /// Count samples per label class.
    pub fn class_distribution(&self) -> HashMap<usize, usize> {
        let mut dist = HashMap::new();
        for s in &self.samples {
            if let Some(lbl) = s.label {
                *dist.entry(lbl).or_insert(0) += 1;
            }
        }
        dist
    }

    /// Deterministic shuffle using a simple LCG seeded PRNG.
    pub fn shuffle(&mut self, seed: u64) {
        let n = self.samples.len();
        if n < 2 {
            return;
        }
        let mut state = seed.wrapping_add(1);
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            self.samples.swap(i, j);
        }
    }

    /// Split into train/val/test datasets.
    pub fn split(&self, ratio: &SplitRatio) -> (Dataset, Dataset, Dataset) {
        let n = self.samples.len();
        let train_end = (n as f64 * ratio.train).round() as usize;
        let val_end = train_end + (n as f64 * ratio.val).round() as usize;
        let train = Dataset::from_samples(self.samples[..train_end].to_vec());
        let val = Dataset::from_samples(self.samples[train_end..val_end.min(n)].to_vec());
        let test = Dataset::from_samples(self.samples[val_end.min(n)..].to_vec());
        (train, val, test)
    }

    /// Stratified split preserving class proportions.
    pub fn stratified_split(&self, ratio: &SplitRatio, seed: u64) -> (Dataset, Dataset, Dataset) {
        let mut by_class: HashMap<usize, Vec<&Sample>> = HashMap::new();
        let mut unlabelled: Vec<&Sample> = Vec::new();
        for s in &self.samples {
            match s.label {
                Some(lbl) => by_class.entry(lbl).or_default().push(s),
                None => unlabelled.push(s),
            }
        }
        let mut train_samples = Vec::new();
        let mut val_samples = Vec::new();
        let mut test_samples = Vec::new();

        let mut class_keys: Vec<usize> = by_class.keys().copied().collect();
        class_keys.sort();

        for cls in class_keys {
            let mut class_data: Vec<Sample> = by_class[&cls].iter().map(|s| (*s).clone()).collect();
            let mut mini = Dataset::from_samples(class_data.clone());
            mini.shuffle(seed.wrapping_add(cls as u64));
            class_data = mini.samples;
            let cn = class_data.len();
            let t_end = (cn as f64 * ratio.train).round() as usize;
            let v_end = t_end + (cn as f64 * ratio.val).round() as usize;
            train_samples.extend_from_slice(&class_data[..t_end]);
            val_samples.extend_from_slice(&class_data[t_end..v_end.min(cn)]);
            test_samples.extend_from_slice(&class_data[v_end.min(cn)..]);
        }
        // Unlabelled go to train
        for s in unlabelled {
            train_samples.push(s.clone());
        }
        (
            Dataset::from_samples(train_samples),
            Dataset::from_samples(val_samples),
            Dataset::from_samples(test_samples),
        )
    }
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dataset(n={})", self.samples.len())
    }
}

// ── Batch ───────────────────────────────────────────────────────

/// A batch of samples for iteration.
#[derive(Debug, Clone)]
pub struct Batch {
    pub samples: Vec<Sample>,
    pub index: usize,
}

impl Batch {
    pub fn new(samples: Vec<Sample>, index: usize) -> Self {
        Self { samples, index }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Extract feature matrix (row-major: samples x features).
    pub fn feature_matrix(&self) -> Vec<Vec<f64>> {
        self.samples.iter().map(|s| s.features.clone()).collect()
    }

    /// Extract label vector.
    pub fn labels(&self) -> Vec<Option<usize>> {
        self.samples.iter().map(|s| s.label).collect()
    }
}

impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Batch(idx={}, size={})", self.index, self.len())
    }
}

// ── DataLoader ──────────────────────────────────────────────────

/// Configuration and state for batched data iteration.
#[derive(Debug, Clone)]
pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle_each_epoch: bool,
    drop_last: bool,
    prefetch_count: usize,
    seed: u64,
    epoch: u64,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size: batch_size.max(1),
            shuffle_each_epoch: true,
            drop_last: false,
            prefetch_count: 2,
            seed: 42,
            epoch: 0,
        }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle_each_epoch = shuffle;
        self
    }

    pub fn with_drop_last(mut self, drop: bool) -> Self {
        self.drop_last = drop;
        self
    }

    pub fn with_prefetch(mut self, count: usize) -> Self {
        self.prefetch_count = count;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    pub fn dataset_len(&self) -> usize {
        self.dataset.len()
    }

    /// Generate all batches for one epoch, advancing the epoch counter.
    pub fn epoch_batches(&mut self) -> Vec<Batch> {
        if self.shuffle_each_epoch {
            let epoch_seed = self.seed.wrapping_add(self.epoch);
            self.dataset.shuffle(epoch_seed);
        }
        self.epoch += 1;

        let n = self.dataset.len();
        let mut batches = Vec::new();
        let mut idx = 0;
        let mut batch_idx = 0;

        while idx < n {
            let end = (idx + self.batch_size).min(n);
            let chunk = self.dataset.samples[idx..end].to_vec();
            if self.drop_last && chunk.len() < self.batch_size {
                break;
            }
            batches.push(Batch::new(chunk, batch_idx));
            batch_idx += 1;
            idx = end;
        }
        batches
    }

    /// Simulate prefetching by returning the next N batches from an epoch.
    pub fn prefetch_batches(&mut self) -> Vec<Batch> {
        let all = self.epoch_batches();
        all.into_iter().take(self.prefetch_count).collect()
    }

    pub fn current_epoch(&self) -> u64 {
        self.epoch
    }
}

impl fmt::Display for DataLoader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DataLoader(n={}, bs={}, shuffle={}, epochs={})",
               self.dataset.len(), self.batch_size, self.shuffle_each_epoch, self.epoch)
    }
}

// ── Sampler ─────────────────────────────────────────────────────

/// Weighted random sampler for imbalanced datasets.
#[derive(Debug, Clone)]
pub struct WeightedSampler {
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
}

impl WeightedSampler {
    pub fn new(weights: Vec<f64>, num_samples: usize) -> Self {
        Self { weights, num_samples, replacement: true }
    }

    pub fn with_replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    /// Compute class-balanced weights from label distribution.
    pub fn balanced_weights(labels: &[usize]) -> Vec<f64> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &l in labels {
            *counts.entry(l).or_insert(0) += 1;
        }
        let n = labels.len() as f64;
        let num_classes = counts.len() as f64;
        labels
            .iter()
            .map(|l| n / (num_classes * counts[l] as f64))
            .collect()
    }

    /// Generate sample indices using a deterministic PRNG.
    pub fn sample_indices(&self, seed: u64) -> Vec<usize> {
        if self.weights.is_empty() {
            return Vec::new();
        }
        let total: f64 = self.weights.iter().sum();
        if total <= 0.0 {
            return Vec::new();
        }
        // Build CDF
        let mut cdf = Vec::with_capacity(self.weights.len());
        let mut cumulative = 0.0;
        for &w in &self.weights {
            cumulative += w / total;
            cdf.push(cumulative);
        }
        let mut state = seed.wrapping_add(7);
        let mut indices = Vec::with_capacity(self.num_samples);
        let mut used = Vec::new();
        for _ in 0..self.num_samples {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state >> 11) as f64 / (1u64 << 53) as f64;
            let idx = cdf.partition_point(|c| *c < r).min(cdf.len() - 1);
            if !self.replacement && used.contains(&idx) {
                continue;
            }
            indices.push(idx);
            if !self.replacement {
                used.push(idx);
            }
        }
        indices
    }
}

impl fmt::Display for WeightedSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WeightedSampler(n_weights={}, n_samples={}, replace={})",
               self.weights.len(), self.num_samples, self.replacement)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n: usize, n_classes: usize) -> Dataset {
        let mut samples = Vec::new();
        for i in 0..n {
            let label = i % n_classes;
            samples.push(
                Sample::new(vec![i as f64, (i * 2) as f64])
                    .with_label(label),
            );
        }
        Dataset::from_samples(samples)
    }

    #[test]
    fn sample_creation() {
        let s = Sample::new(vec![1.0, 2.0]).with_label(3).with_weight(0.5);
        assert_eq!(s.dim(), 2);
        assert_eq!(s.label, Some(3));
        assert!((s.weight - 0.5).abs() < 1e-9);
    }

    #[test]
    fn sample_display() {
        let s = Sample::new(vec![1.0]).with_label(0);
        let txt = format!("{}", s);
        assert!(txt.contains("dim=1"));
    }

    #[test]
    fn split_ratio_valid() {
        assert!(SplitRatio::new(0.7, 0.15, 0.15).is_valid());
        assert!(!SplitRatio::new(0.5, 0.5, 0.5).is_valid());
    }

    #[test]
    fn split_ratio_display() {
        let r = SplitRatio::default_split();
        let txt = format!("{}", r);
        assert!(txt.contains("80%"));
    }

    #[test]
    fn dataset_len() {
        let ds = make_dataset(100, 5);
        assert_eq!(ds.len(), 100);
    }

    #[test]
    fn dataset_class_distribution() {
        let ds = make_dataset(100, 5);
        let dist = ds.class_distribution();
        assert_eq!(dist.len(), 5);
        for &count in dist.values() {
            assert_eq!(count, 20);
        }
    }

    #[test]
    fn dataset_shuffle_deterministic() {
        let mut ds1 = make_dataset(50, 3);
        let mut ds2 = make_dataset(50, 3);
        ds1.shuffle(42);
        ds2.shuffle(42);
        assert_eq!(ds1.samples, ds2.samples);
    }

    #[test]
    fn dataset_shuffle_changes_order() {
        let original = make_dataset(50, 3);
        let mut shuffled = original.clone();
        shuffled.shuffle(123);
        assert_ne!(original.samples, shuffled.samples);
    }

    #[test]
    fn dataset_split_sizes() {
        let ds = make_dataset(100, 2);
        let ratio = SplitRatio::new(0.7, 0.2, 0.1);
        let (train, val, test) = ds.split(&ratio);
        assert_eq!(train.len() + val.len() + test.len(), 100);
        assert_eq!(train.len(), 70);
        assert_eq!(val.len(), 20);
        assert_eq!(test.len(), 10);
    }

    #[test]
    fn stratified_split_preserves_proportions() {
        let ds = make_dataset(200, 4);
        let ratio = SplitRatio::new(0.8, 0.1, 0.1);
        let (train, val, test) = ds.stratified_split(&ratio, 99);
        let total = train.len() + val.len() + test.len();
        assert_eq!(total, 200);
        // Each class should be roughly proportional
        let train_dist = train.class_distribution();
        for &count in train_dist.values() {
            assert!(count >= 38 && count <= 42);
        }
    }

    #[test]
    fn batch_features_and_labels() {
        let samples = vec![
            Sample::new(vec![1.0, 2.0]).with_label(0),
            Sample::new(vec![3.0, 4.0]).with_label(1),
        ];
        let batch = Batch::new(samples, 0);
        assert_eq!(batch.feature_matrix().len(), 2);
        assert_eq!(batch.labels(), vec![Some(0), Some(1)]);
    }

    #[test]
    fn dataloader_num_batches() {
        let ds = make_dataset(100, 2);
        let loader = DataLoader::new(ds, 32);
        assert_eq!(loader.num_batches(), 4); // ceil(100/32)
    }

    #[test]
    fn dataloader_drop_last() {
        let ds = make_dataset(100, 2);
        let loader = DataLoader::new(ds, 32).with_drop_last(true);
        assert_eq!(loader.num_batches(), 3); // floor(100/32)
    }

    #[test]
    fn dataloader_epoch_batches() {
        let ds = make_dataset(50, 2);
        let mut loader = DataLoader::new(ds, 16).with_seed(7);
        let batches = loader.epoch_batches();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].index, 0);
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn dataloader_shuffle_between_epochs() {
        let ds = make_dataset(20, 2);
        let mut loader = DataLoader::new(ds, 20).with_seed(42);
        let epoch1 = loader.epoch_batches();
        let epoch2 = loader.epoch_batches();
        assert_ne!(epoch1[0].samples, epoch2[0].samples);
    }

    #[test]
    fn dataloader_no_shuffle() {
        let ds = make_dataset(10, 2);
        let mut loader = DataLoader::new(ds, 10).with_shuffle(false);
        let epoch1 = loader.epoch_batches();
        let epoch2 = loader.epoch_batches();
        assert_eq!(epoch1[0].samples, epoch2[0].samples);
    }

    #[test]
    fn dataloader_prefetch() {
        let ds = make_dataset(100, 2);
        let mut loader = DataLoader::new(ds, 10).with_prefetch(3);
        let prefetched = loader.prefetch_batches();
        assert_eq!(prefetched.len(), 3);
    }

    #[test]
    fn weighted_sampler_balanced() {
        let labels = vec![0, 0, 0, 0, 1, 1];
        let weights = WeightedSampler::balanced_weights(&labels);
        assert_eq!(weights.len(), 6);
        // Class 0 weight < class 1 weight (more samples in class 0)
        assert!(weights[0] < weights[4]);
    }

    #[test]
    fn weighted_sampler_indices() {
        let sampler = WeightedSampler::new(vec![1.0, 2.0, 3.0], 10);
        let indices = sampler.sample_indices(42);
        assert_eq!(indices.len(), 10);
        for &idx in &indices {
            assert!(idx < 3);
        }
    }

    #[test]
    fn dataloader_display() {
        let ds = make_dataset(10, 2);
        let loader = DataLoader::new(ds, 4);
        let txt = format!("{}", loader);
        assert!(txt.contains("n=10"));
        assert!(txt.contains("bs=4"));
    }
}
