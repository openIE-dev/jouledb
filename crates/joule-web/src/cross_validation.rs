//! Cross-Validation — k-fold, stratified k-fold, leave-one-out,
//! time series split, nested CV, and repeated CV for ML model evaluation.
//!
//! Pure Rust, std-only. All fold generators produce index pairs (train, test).

use std::collections::HashMap;
use std::fmt;

// ── Fold ────────────────────────────────────────────────────────

/// A single fold containing train and test indices.
#[derive(Debug, Clone, PartialEq)]
pub struct Fold {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub fold_index: usize,
}

impl Fold {
    pub fn new(train: Vec<usize>, test: Vec<usize>, index: usize) -> Self {
        Self {
            train_indices: train,
            test_indices: test,
            fold_index: index,
        }
    }

    pub fn train_size(&self) -> usize {
        self.train_indices.len()
    }

    pub fn test_size(&self) -> usize {
        self.test_indices.len()
    }
}

impl fmt::Display for Fold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fold(idx={}, train={}, test={})",
               self.fold_index, self.train_size(), self.test_size())
    }
}

// ── K-Fold ──────────────────────────────────────────────────────

/// Standard k-fold cross-validation.
#[derive(Debug, Clone)]
pub struct KFold {
    k: usize,
    shuffle: bool,
    seed: u64,
}

impl KFold {
    pub fn new(k: usize) -> Self {
        Self { k: k.max(2), shuffle: false, seed: 42 }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate all k folds for n samples.
    pub fn split(&self, n: usize) -> Vec<Fold> {
        let mut indices: Vec<usize> = (0..n).collect();
        if self.shuffle {
            deterministic_shuffle(&mut indices, self.seed);
        }
        let fold_size = n / self.k;
        let remainder = n % self.k;
        let mut folds = Vec::with_capacity(self.k);
        let mut start = 0;

        for i in 0..self.k {
            let extra = if i < remainder { 1 } else { 0 };
            let end = start + fold_size + extra;
            let test: Vec<usize> = indices[start..end].to_vec();
            let train: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();
            folds.push(Fold::new(train, test, i));
            start = end;
        }
        folds
    }

    pub fn num_folds(&self) -> usize {
        self.k
    }
}

impl fmt::Display for KFold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "KFold(k={}, shuffle={})", self.k, self.shuffle)
    }
}

// ── Stratified K-Fold ───────────────────────────────────────────

/// Stratified k-fold: preserves class proportions in each fold.
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    k: usize,
    shuffle: bool,
    seed: u64,
}

impl StratifiedKFold {
    pub fn new(k: usize) -> Self {
        Self { k: k.max(2), shuffle: false, seed: 42 }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate folds preserving label proportions.
    pub fn split(&self, labels: &[usize]) -> Vec<Fold> {
        let n = labels.len();
        // Group indices by class
        let mut by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            by_class.entry(label).or_default().push(i);
        }

        // Optionally shuffle within each class
        if self.shuffle {
            let mut class_keys: Vec<usize> = by_class.keys().copied().collect();
            class_keys.sort();
            for cls in class_keys {
                if let Some(indices) = by_class.get_mut(&cls) {
                    deterministic_shuffle(indices, self.seed.wrapping_add(cls as u64));
                }
            }
        }

        // Assign each class's samples to folds round-robin
        let mut fold_test: Vec<Vec<usize>> = vec![Vec::new(); self.k];
        let mut class_keys: Vec<usize> = by_class.keys().copied().collect();
        class_keys.sort();

        for cls in class_keys {
            let class_indices = &by_class[&cls];
            for (i, &idx) in class_indices.iter().enumerate() {
                fold_test[i % self.k].push(idx);
            }
        }

        // Build folds
        let all_indices: Vec<usize> = (0..n).collect();
        let mut folds = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let test_set: Vec<usize> = fold_test[i].clone();
            let mut test_set_sorted = test_set.clone();
            test_set_sorted.sort();
            let train: Vec<usize> = all_indices
                .iter()
                .filter(|idx| test_set_sorted.binary_search(idx).is_err())
                .copied()
                .collect();
            folds.push(Fold::new(train, test_set, i));
        }
        folds
    }

    pub fn num_folds(&self) -> usize {
        self.k
    }
}

impl fmt::Display for StratifiedKFold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StratifiedKFold(k={}, shuffle={})", self.k, self.shuffle)
    }
}

// ── Leave-One-Out ───────────────────────────────────────────────

/// Leave-one-out cross-validation (LOO-CV).
#[derive(Debug, Clone, Copy)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    pub fn new() -> Self {
        Self
    }

    /// Generate n folds, each with one test sample.
    pub fn split(&self, n: usize) -> Vec<Fold> {
        (0..n)
            .map(|i| {
                let train: Vec<usize> = (0..n).filter(|j| *j != i).collect();
                Fold::new(train, vec![i], i)
            })
            .collect()
    }

    pub fn num_folds(&self, n: usize) -> usize {
        n
    }
}

impl fmt::Display for LeaveOneOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LeaveOneOut")
    }
}

// ── Time Series Split ───────────────────────────────────────────

/// Expanding-window time series cross-validation.
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    max_train_size: Option<usize>,
    gap: usize,
}

impl TimeSeriesSplit {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits: n_splits.max(2),
            max_train_size: None,
            gap: 0,
        }
    }

    pub fn with_max_train_size(mut self, size: usize) -> Self {
        self.max_train_size = Some(size);
        self
    }

    pub fn with_gap(mut self, gap_size: usize) -> Self {
        self.gap = gap_size;
        self
    }

    /// Generate expanding-window folds respecting temporal ordering.
    pub fn split(&self, n: usize) -> Vec<Fold> {
        let test_size = n / (self.n_splits + 1);
        if test_size == 0 {
            return Vec::new();
        }

        let mut folds = Vec::with_capacity(self.n_splits);
        for i in 0..self.n_splits {
            let test_start = (i + 1) * test_size + self.gap;
            let test_end = ((i + 2) * test_size).min(n);
            if test_start >= n || test_start >= test_end {
                continue;
            }

            let train_end = test_start.saturating_sub(self.gap);
            let train_start = match self.max_train_size {
                Some(max) => train_end.saturating_sub(max),
                None => 0,
            };

            let train: Vec<usize> = (train_start..train_end).collect();
            let test: Vec<usize> = (test_start..test_end).collect();
            if !train.is_empty() && !test.is_empty() {
                folds.push(Fold::new(train, test, i));
            }
        }
        folds
    }

    pub fn num_splits(&self) -> usize {
        self.n_splits
    }
}

impl fmt::Display for TimeSeriesSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeSeriesSplit(splits={}, gap={})", self.n_splits, self.gap)
    }
}

// ── Nested CV ───────────────────────────────────────────────────

/// Nested cross-validation with outer and inner fold generators.
#[derive(Debug, Clone)]
pub struct NestedCV {
    outer_k: usize,
    inner_k: usize,
    shuffle: bool,
    seed: u64,
}

impl NestedCV {
    pub fn new(outer_k: usize, inner_k: usize) -> Self {
        Self {
            outer_k: outer_k.max(2),
            inner_k: inner_k.max(2),
            shuffle: false,
            seed: 42,
        }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate nested fold structure: outer folds, each containing inner folds.
    pub fn split(&self, n: usize) -> Vec<(Fold, Vec<Fold>)> {
        let outer = KFold::new(self.outer_k)
            .with_shuffle(self.shuffle)
            .with_seed(self.seed);
        let outer_folds = outer.split(n);

        outer_folds
            .into_iter()
            .map(|of| {
                let inner = KFold::new(self.inner_k)
                    .with_shuffle(self.shuffle)
                    .with_seed(self.seed.wrapping_add(of.fold_index as u64 + 100));
                // Inner folds operate on outer train indices
                let inner_n = of.train_indices.len();
                let raw_inner = inner.split(inner_n);
                // Remap inner indices to global indices
                let remapped: Vec<Fold> = raw_inner
                    .into_iter()
                    .map(|inf| {
                        let train: Vec<usize> = inf
                            .train_indices
                            .iter()
                            .map(|i| of.train_indices[*i])
                            .collect();
                        let test: Vec<usize> = inf
                            .test_indices
                            .iter()
                            .map(|i| of.train_indices[*i])
                            .collect();
                        Fold::new(train, test, inf.fold_index)
                    })
                    .collect();
                (of, remapped)
            })
            .collect()
    }
}

impl fmt::Display for NestedCV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NestedCV(outer={}, inner={})", self.outer_k, self.inner_k)
    }
}

// ── Repeated K-Fold ─────────────────────────────────────────────

/// Repeats k-fold CV multiple times with different shuffles.
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    k: usize,
    n_repeats: usize,
    seed: u64,
}

impl RepeatedKFold {
    pub fn new(k: usize, n_repeats: usize) -> Self {
        Self { k: k.max(2), n_repeats: n_repeats.max(1), seed: 42 }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generate all repeated fold sets.
    pub fn split(&self, n: usize) -> Vec<Fold> {
        let mut all_folds = Vec::new();
        for rep in 0..self.n_repeats {
            let kf = KFold::new(self.k)
                .with_shuffle(true)
                .with_seed(self.seed.wrapping_add(rep as u64 * 1000));
            let folds = kf.split(n);
            for (i, fold) in folds.into_iter().enumerate() {
                all_folds.push(Fold::new(
                    fold.train_indices,
                    fold.test_indices,
                    rep * self.k + i,
                ));
            }
        }
        all_folds
    }

    pub fn total_folds(&self) -> usize {
        self.k * self.n_repeats
    }
}

impl fmt::Display for RepeatedKFold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RepeatedKFold(k={}, repeats={})", self.k, self.n_repeats)
    }
}

// ── CV Results ──────────────────────────────────────────────────

/// Aggregated results from cross-validation runs.
#[derive(Debug, Clone)]
pub struct CvResults {
    pub scores: Vec<f64>,
}

impl CvResults {
    pub fn new(scores: Vec<f64>) -> Self {
        Self { scores }
    }

    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f64>() / self.scores.len() as f64
    }

    pub fn std_dev(&self) -> f64 {
        let m = self.mean();
        let n = self.scores.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let var = self.scores.iter().map(|s| (s - m).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    }

    pub fn min(&self) -> f64 {
        self.scores.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn num_folds(&self) -> usize {
        self.scores.len()
    }
}

impl fmt::Display for CvResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CV(mean={:.4} +/- {:.4}, n={})", self.mean(), self.std_dev(), self.num_folds())
    }
}

// ── Utility ─────────────────────────────────────────────────────

fn deterministic_shuffle(data: &mut [usize], seed: u64) {
    let n = data.len();
    if n < 2 {
        return;
    }
    let mut state = seed.wrapping_add(1);
    for i in (1..n).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        data.swap(i, j);
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kfold_basic() {
        let kf = KFold::new(5);
        let folds = kf.split(100);
        assert_eq!(folds.len(), 5);
        for fold in &folds {
            assert_eq!(fold.test_size(), 20);
            assert_eq!(fold.train_size(), 80);
        }
    }

    #[test]
    fn kfold_no_overlap() {
        let kf = KFold::new(3);
        let folds = kf.split(30);
        for fold in &folds {
            for &ti in &fold.test_indices {
                assert!(!fold.train_indices.contains(&ti));
            }
        }
    }

    #[test]
    fn kfold_complete_coverage() {
        let kf = KFold::new(4);
        let folds = kf.split(20);
        let mut all_test: Vec<usize> = folds.iter().flat_map(|f| f.test_indices.clone()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn kfold_shuffle() {
        let kf1 = KFold::new(3).with_shuffle(true).with_seed(42);
        let kf2 = KFold::new(3);
        let f1 = kf1.split(30);
        let f2 = kf2.split(30);
        assert_ne!(f1[0].test_indices, f2[0].test_indices);
    }

    #[test]
    fn kfold_display() {
        let kf = KFold::new(5);
        assert!(format!("{}", kf).contains("k=5"));
    }

    #[test]
    fn stratified_preserves_proportions() {
        let labels: Vec<usize> = (0..100).map(|i| if i < 70 { 0 } else { 1 }).collect();
        let skf = StratifiedKFold::new(5);
        let folds = skf.split(&labels);
        assert_eq!(folds.len(), 5);
        for fold in &folds {
            let test_labels: Vec<usize> = fold.test_indices.iter().map(|i| labels[*i]).collect();
            let class0 = test_labels.iter().filter(|&&l| l == 0).count();
            let class1 = test_labels.iter().filter(|&&l| l == 1).count();
            // Roughly 70/30 proportion
            assert!(class0 >= 10);
            assert!(class1 >= 3);
        }
    }

    #[test]
    fn stratified_display() {
        let skf = StratifiedKFold::new(10);
        assert!(format!("{}", skf).contains("k=10"));
    }

    #[test]
    fn loo_basic() {
        let loo = LeaveOneOut::new();
        let folds = loo.split(5);
        assert_eq!(folds.len(), 5);
        for fold in &folds {
            assert_eq!(fold.test_size(), 1);
            assert_eq!(fold.train_size(), 4);
        }
    }

    #[test]
    fn loo_display() {
        let loo = LeaveOneOut::new();
        assert!(format!("{}", loo).contains("LeaveOneOut"));
    }

    #[test]
    fn timeseries_expanding_window() {
        let ts = TimeSeriesSplit::new(3);
        let folds = ts.split(40);
        assert!(!folds.is_empty());
        // Each fold's train end < test start (temporal ordering)
        for fold in &folds {
            if !fold.train_indices.is_empty() && !fold.test_indices.is_empty() {
                let max_train = *fold.train_indices.iter().max().unwrap();
                let min_test = *fold.test_indices.iter().min().unwrap();
                assert!(max_train < min_test);
            }
        }
    }

    #[test]
    fn timeseries_with_gap() {
        let ts = TimeSeriesSplit::new(3).with_gap(2);
        let folds = ts.split(40);
        for fold in &folds {
            if !fold.train_indices.is_empty() && !fold.test_indices.is_empty() {
                let max_train = *fold.train_indices.iter().max().unwrap();
                let min_test = *fold.test_indices.iter().min().unwrap();
                assert!(min_test - max_train > 1);
            }
        }
    }

    #[test]
    fn timeseries_display() {
        let ts = TimeSeriesSplit::new(5).with_gap(3);
        let txt = format!("{}", ts);
        assert!(txt.contains("splits=5"));
        assert!(txt.contains("gap=3"));
    }

    #[test]
    fn nested_cv_structure() {
        let ncv = NestedCV::new(3, 2);
        let result = ncv.split(60);
        assert_eq!(result.len(), 3);
        for (outer, inner) in &result {
            assert_eq!(inner.len(), 2);
            // Inner train + inner test indices should be subset of outer train
            for inf in inner {
                for &idx in &inf.train_indices {
                    assert!(outer.train_indices.contains(&idx));
                }
                for &idx in &inf.test_indices {
                    assert!(outer.train_indices.contains(&idx));
                }
            }
        }
    }

    #[test]
    fn nested_cv_display() {
        let ncv = NestedCV::new(5, 3);
        assert!(format!("{}", ncv).contains("outer=5"));
    }

    #[test]
    fn repeated_kfold_total() {
        let rkf = RepeatedKFold::new(5, 3);
        let folds = rkf.split(50);
        assert_eq!(folds.len(), 15);
    }

    #[test]
    fn repeated_kfold_different_shuffles() {
        let rkf = RepeatedKFold::new(3, 2).with_seed(42);
        let folds = rkf.split(30);
        // First fold of repeat 0 vs first fold of repeat 1
        assert_ne!(folds[0].test_indices, folds[3].test_indices);
    }

    #[test]
    fn cv_results_basic() {
        let r = CvResults::new(vec![0.8, 0.85, 0.9, 0.82, 0.88]);
        assert!((r.mean() - 0.85).abs() < 1e-9);
        assert!(r.std_dev() > 0.0);
        assert!((r.min() - 0.8).abs() < 1e-9);
        assert!((r.max() - 0.9).abs() < 1e-9);
    }

    #[test]
    fn cv_results_display() {
        let r = CvResults::new(vec![0.9, 0.92, 0.88]);
        let txt = format!("{}", r);
        assert!(txt.contains("CV("));
        assert!(txt.contains("n=3"));
    }

    #[test]
    fn fold_display() {
        let fold = Fold::new(vec![0, 1, 2], vec![3, 4], 0);
        let txt = format!("{}", fold);
        assert!(txt.contains("train=3"));
        assert!(txt.contains("test=2"));
    }
}
