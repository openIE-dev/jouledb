//! Data sampling strategies.
//!
//! Replaces `scikit-learn`'s `train_test_split`, `reservoir sampling`, and similar
//! libraries with a pure-Rust sampler. Supports random, stratified, systematic
//! (every Nth), reservoir (fixed memory), weighted sampling, sample validation
//! (distribution comparison), and reproducible results with seed.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Errors from sampling operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerError {
    /// Sample size exceeds population.
    SampleTooLarge { requested: usize, available: usize },
    /// Empty dataset.
    EmptyDataset,
    /// Invalid configuration.
    InvalidConfig(String),
    /// Stratum not found for stratified sampling.
    StratumNotFound(String),
    /// Weights length mismatch.
    WeightsMismatch { expected: usize, got: usize },
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SampleTooLarge { requested, available } => {
                write!(f, "sample size {requested} exceeds population {available}")
            }
            Self::EmptyDataset => write!(f, "cannot sample from empty dataset"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::StratumNotFound(s) => write!(f, "stratum not found: {s}"),
            Self::WeightsMismatch { expected, got } => {
                write!(f, "weights length {got} does not match data length {expected}")
            }
        }
    }
}

impl std::error::Error for SamplerError {}

// ── PRNG ─────────────────────────────────────────────────────────

/// Xoshiro256** for deterministic sampling.
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Create a new RNG from a seed.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.next_u64() as usize) % max
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Fisher-Yates shuffle indices [0..n).
    fn shuffle_indices(&mut self, n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.next_usize(i + 1);
            indices.swap(i, j);
        }
        indices
    }
}

// ── Sample result ────────────────────────────────────────────────

/// Result of a sampling operation.
#[derive(Debug, Clone)]
pub struct SampleResult<T> {
    /// The sampled items.
    pub items: Vec<T>,
    /// Indices of selected items in the original data (when applicable).
    pub indices: Vec<usize>,
    /// Sample size.
    pub sample_size: usize,
    /// Original population size.
    pub population_size: usize,
    /// Sampling rate (sample_size / population_size).
    pub sampling_rate: f64,
}

impl<T> SampleResult<T> {
    fn new(items: Vec<T>, indices: Vec<usize>, population_size: usize) -> Self {
        let sample_size = items.len();
        let sampling_rate = if population_size == 0 {
            0.0
        } else {
            sample_size as f64 / population_size as f64
        };
        Self {
            items,
            indices,
            sample_size,
            population_size,
            sampling_rate,
        }
    }
}

// ── Distribution comparison ──────────────────────────────────────

/// Result of comparing sample distribution to population distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparison {
    /// Per-stratum comparison.
    pub strata: Vec<StratumComparison>,
    /// Maximum absolute deviation across all strata.
    pub max_deviation: f64,
    /// Mean absolute deviation.
    pub mean_deviation: f64,
    /// Whether the sample is considered representative (max_deviation below threshold).
    pub is_representative: bool,
}

/// Comparison for a single stratum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumComparison {
    /// Stratum label.
    pub label: String,
    /// Expected proportion in population.
    pub expected_proportion: f64,
    /// Actual proportion in sample.
    pub actual_proportion: f64,
    /// Absolute deviation.
    pub deviation: f64,
}

// ── Sampler ──────────────────────────────────────────────────────

/// Data sampler with multiple strategies.
pub struct DataSampler {
    rng: Rng,
}

impl DataSampler {
    /// Create a new sampler with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Rng::new(seed),
        }
    }

    /// Random sample: select `n` items uniformly at random without replacement.
    pub fn random_sample<T: Clone>(
        &mut self,
        data: &[T],
        n: usize,
    ) -> Result<SampleResult<T>, SamplerError> {
        if data.is_empty() {
            return Err(SamplerError::EmptyDataset);
        }
        if n > data.len() {
            return Err(SamplerError::SampleTooLarge {
                requested: n,
                available: data.len(),
            });
        }

        let shuffled = self.rng.shuffle_indices(data.len());
        let mut indices: Vec<usize> = shuffled[..n].to_vec();
        indices.sort();
        let items: Vec<T> = indices.iter().map(|i| data[*i].clone()).collect();

        Ok(SampleResult::new(items, indices, data.len()))
    }

    /// Systematic sample: select every `step`th item, starting from a random offset.
    pub fn systematic_sample<T: Clone>(
        &mut self,
        data: &[T],
        step: usize,
    ) -> Result<SampleResult<T>, SamplerError> {
        if data.is_empty() {
            return Err(SamplerError::EmptyDataset);
        }
        if step == 0 {
            return Err(SamplerError::InvalidConfig("step must be > 0".into()));
        }

        let offset = self.rng.next_usize(step.min(data.len()));
        let mut indices = Vec::new();
        let mut i = offset;
        while i < data.len() {
            indices.push(i);
            i += step;
        }

        let items: Vec<T> = indices.iter().map(|i| data[*i].clone()).collect();
        Ok(SampleResult::new(items, indices, data.len()))
    }

    /// Stratified sample: sample proportionally from each stratum.
    /// `stratum_fn` assigns each item to a stratum label.
    pub fn stratified_sample<T: Clone>(
        &mut self,
        data: &[T],
        n: usize,
        stratum_fn: impl Fn(&T) -> String,
    ) -> Result<SampleResult<T>, SamplerError> {
        if data.is_empty() {
            return Err(SamplerError::EmptyDataset);
        }
        if n > data.len() {
            return Err(SamplerError::SampleTooLarge {
                requested: n,
                available: data.len(),
            });
        }

        // Group by stratum, preserving order of first appearance.
        let mut stratum_order: Vec<String> = Vec::new();
        let mut strata: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, item) in data.iter().enumerate() {
            let label = stratum_fn(item);
            let entry = strata.entry(label.clone()).or_insert_with(|| {
                stratum_order.push(label);
                Vec::new()
            });
            entry.push(i);
        }

        // Proportional allocation.
        let mut all_indices = Vec::new();
        let mut remaining = n;

        for (si, label) in stratum_order.iter().enumerate() {
            let group = strata.get(label).unwrap();
            let proportion = group.len() as f64 / data.len() as f64;
            let stratum_n = if si == stratum_order.len() - 1 {
                remaining // give remainder to last stratum
            } else {
                let allocated = (n as f64 * proportion).round() as usize;
                allocated.min(remaining).min(group.len())
            };

            let shuffled = self.rng.shuffle_indices(group.len());
            let take = stratum_n.min(group.len());
            for &j in &shuffled[..take] {
                all_indices.push(group[j]);
            }
            remaining = remaining.saturating_sub(take);
        }

        all_indices.sort();
        let items: Vec<T> = all_indices.iter().map(|i| data[*i].clone()).collect();
        Ok(SampleResult::new(items, all_indices, data.len()))
    }

    /// Reservoir sampling: fixed-memory sampling from a stream.
    /// Processes items one at a time, keeping at most `k` items.
    pub fn reservoir_sample<T: Clone>(
        &mut self,
        data: &[T],
        k: usize,
    ) -> Result<SampleResult<T>, SamplerError> {
        if data.is_empty() {
            return Err(SamplerError::EmptyDataset);
        }

        let mut reservoir: Vec<(usize, T)> = Vec::with_capacity(k);

        for (i, item) in data.iter().enumerate() {
            if reservoir.len() < k {
                reservoir.push((i, item.clone()));
            } else {
                let j = self.rng.next_usize(i + 1);
                if j < k {
                    reservoir[j] = (i, item.clone());
                }
            }
        }

        reservoir.sort_by_key(|(idx, _)| *idx);
        let indices: Vec<usize> = reservoir.iter().map(|(idx, _)| *idx).collect();
        let items: Vec<T> = reservoir.into_iter().map(|(_, item)| item).collect();

        Ok(SampleResult::new(items, indices, data.len()))
    }

    /// Weighted sampling: sample `n` items with weights (without replacement).
    /// Higher weight = higher probability of selection.
    pub fn weighted_sample<T: Clone>(
        &mut self,
        data: &[T],
        weights: &[f64],
        n: usize,
    ) -> Result<SampleResult<T>, SamplerError> {
        if data.is_empty() {
            return Err(SamplerError::EmptyDataset);
        }
        if weights.len() != data.len() {
            return Err(SamplerError::WeightsMismatch {
                expected: data.len(),
                got: weights.len(),
            });
        }
        if n > data.len() {
            return Err(SamplerError::SampleTooLarge {
                requested: n,
                available: data.len(),
            });
        }

        // Efraimidis-Spirakis algorithm: assign key = rand^(1/w) to each item,
        // then pick the top-n keys.
        let mut keyed: Vec<(usize, f64)> = weights
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let r = self.rng.next_f64();
                // Avoid log(0) and division by zero.
                let w_safe = if w <= 0.0 { f64::EPSILON } else { w };
                let key = r.ln() / w_safe;
                (i, key)
            })
            .collect();

        // Higher keys = higher priority (less negative ln).
        keyed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut indices: Vec<usize> = keyed[..n].iter().map(|(i, _)| *i).collect();
        indices.sort();
        let items: Vec<T> = indices.iter().map(|i| data[*i].clone()).collect();

        Ok(SampleResult::new(items, indices, data.len()))
    }

    /// Validate a sample by comparing its distribution to the population.
    /// `label_fn` assigns a stratum label to each item.
    pub fn validate_sample<T>(
        &self,
        population: &[T],
        sample: &[T],
        label_fn: impl Fn(&T) -> String,
        threshold: f64,
    ) -> DistributionComparison {
        let pop_dist = compute_distribution(population, &label_fn);
        let sample_dist = compute_distribution(sample, &label_fn);

        let mut strata = Vec::new();
        let mut max_dev = 0.0f64;
        let mut total_dev = 0.0f64;

        for (label, &expected) in &pop_dist {
            let actual = sample_dist.get(label).copied().unwrap_or(0.0);
            let dev = (expected - actual).abs();
            max_dev = max_dev.max(dev);
            total_dev += dev;
            strata.push(StratumComparison {
                label: label.clone(),
                expected_proportion: expected,
                actual_proportion: actual,
                deviation: dev,
            });
        }

        // Check for strata only in the sample.
        for (label, &actual) in &sample_dist {
            if !pop_dist.contains_key(label) {
                let dev = actual;
                max_dev = max_dev.max(dev);
                total_dev += dev;
                strata.push(StratumComparison {
                    label: label.clone(),
                    expected_proportion: 0.0,
                    actual_proportion: actual,
                    deviation: dev,
                });
            }
        }

        let mean_dev = if strata.is_empty() {
            0.0
        } else {
            total_dev / strata.len() as f64
        };

        // Sort strata for deterministic output.
        strata.sort_by(|a, b| a.label.cmp(&b.label));

        DistributionComparison {
            strata,
            max_deviation: max_dev,
            mean_deviation: mean_dev,
            is_representative: max_dev <= threshold,
        }
    }
}

impl fmt::Debug for DataSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataSampler").finish()
    }
}

/// Compute the proportion distribution for each stratum.
fn compute_distribution<T>(
    data: &[T],
    label_fn: &impl Fn(&T) -> String,
) -> HashMap<String, f64> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for item in data {
        *counts.entry(label_fn(item)).or_insert(0) += 1;
    }
    let total = data.len() as f64;
    counts
        .into_iter()
        .map(|(k, v)| (k, v as f64 / total))
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_sample_size() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..100).collect();
        let result = sampler.random_sample(&data, 10).unwrap();
        assert_eq!(result.sample_size, 10);
        assert_eq!(result.population_size, 100);
        assert_eq!(result.items.len(), 10);
        assert_eq!(result.indices.len(), 10);
    }

    #[test]
    fn random_sample_no_duplicates() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..100).collect();
        let result = sampler.random_sample(&data, 50).unwrap();
        let mut sorted = result.indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 50);
    }

    #[test]
    fn random_sample_too_large() {
        let mut sampler = DataSampler::new(42);
        let data = vec![1, 2, 3];
        let err = sampler.random_sample(&data, 5).unwrap_err();
        assert!(matches!(err, SamplerError::SampleTooLarge { .. }));
    }

    #[test]
    fn random_sample_empty() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = vec![];
        let err = sampler.random_sample(&data, 1).unwrap_err();
        assert_eq!(err, SamplerError::EmptyDataset);
    }

    #[test]
    fn random_sample_reproducible() {
        let data: Vec<i32> = (0..100).collect();
        let mut s1 = DataSampler::new(42);
        let mut s2 = DataSampler::new(42);
        let r1 = s1.random_sample(&data, 10).unwrap();
        let r2 = s2.random_sample(&data, 10).unwrap();
        assert_eq!(r1.indices, r2.indices);
    }

    #[test]
    fn systematic_sample() {
        let mut sampler = DataSampler::new(0);
        let data: Vec<i32> = (0..20).collect();
        let result = sampler.systematic_sample(&data, 5).unwrap();
        // Should select roughly 20/5 = 4 items.
        assert!(result.items.len() >= 3 && result.items.len() <= 5);
        // All indices should be at most step apart.
        for w in result.indices.windows(2) {
            assert_eq!(w[1] - w[0], 5);
        }
    }

    #[test]
    fn systematic_sample_step_one() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..5).collect();
        let result = sampler.systematic_sample(&data, 1).unwrap();
        assert_eq!(result.items.len(), 5);
    }

    #[test]
    fn systematic_sample_zero_step() {
        let mut sampler = DataSampler::new(42);
        let data = vec![1, 2, 3];
        let err = sampler.systematic_sample(&data, 0).unwrap_err();
        assert!(matches!(err, SamplerError::InvalidConfig(_)));
    }

    #[test]
    fn stratified_sample() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<(&str, i32)> = vec![
            ("A", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5),
            ("B", 6), ("B", 7), ("B", 8), ("B", 9), ("B", 10),
        ];

        let result = sampler
            .stratified_sample(&data, 4, |item| item.0.to_string())
            .unwrap();
        assert_eq!(result.sample_size, 4);
        // Should have roughly equal proportions from A and B.
        let a_count = result.items.iter().filter(|item| item.0 == "A").count();
        let b_count = result.items.iter().filter(|item| item.0 == "B").count();
        assert!(a_count >= 1 && a_count <= 3);
        assert!(b_count >= 1 && b_count <= 3);
    }

    #[test]
    fn stratified_sample_too_large() {
        let mut sampler = DataSampler::new(42);
        let data = vec![("A", 1), ("B", 2)];
        let err = sampler
            .stratified_sample(&data, 5, |item| item.0.to_string())
            .unwrap_err();
        assert!(matches!(err, SamplerError::SampleTooLarge { .. }));
    }

    #[test]
    fn reservoir_sample() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..1000).collect();
        let result = sampler.reservoir_sample(&data, 10).unwrap();
        assert_eq!(result.sample_size, 10);
        assert_eq!(result.population_size, 1000);
    }

    #[test]
    fn reservoir_sample_larger_than_data() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..5).collect();
        let result = sampler.reservoir_sample(&data, 10).unwrap();
        // Should return all items since k > n.
        assert_eq!(result.sample_size, 5);
    }

    #[test]
    fn reservoir_sample_reproducible() {
        let data: Vec<i32> = (0..100).collect();
        let mut s1 = DataSampler::new(123);
        let mut s2 = DataSampler::new(123);
        let r1 = s1.reservoir_sample(&data, 10).unwrap();
        let r2 = s2.reservoir_sample(&data, 10).unwrap();
        assert_eq!(r1.items, r2.items);
    }

    #[test]
    fn weighted_sample() {
        let mut sampler = DataSampler::new(42);
        let data = vec!["rare", "common", "common", "common", "common"];
        let weights = vec![0.01, 10.0, 10.0, 10.0, 10.0];
        let result = sampler.weighted_sample(&data, &weights, 3).unwrap();
        assert_eq!(result.sample_size, 3);
        // "common" items should be more likely to be selected.
    }

    #[test]
    fn weighted_sample_mismatch() {
        let mut sampler = DataSampler::new(42);
        let data = vec![1, 2, 3];
        let weights = vec![1.0, 2.0];
        let err = sampler.weighted_sample(&data, &weights, 2).unwrap_err();
        assert!(matches!(err, SamplerError::WeightsMismatch { .. }));
    }

    #[test]
    fn weighted_sample_too_large() {
        let mut sampler = DataSampler::new(42);
        let data = vec![1, 2];
        let weights = vec![1.0, 1.0];
        let err = sampler.weighted_sample(&data, &weights, 5).unwrap_err();
        assert!(matches!(err, SamplerError::SampleTooLarge { .. }));
    }

    #[test]
    fn validate_sample_representative() {
        let sampler = DataSampler::new(42);
        let population: Vec<&str> = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];
        let sample: Vec<&str> = vec!["A", "A", "B", "B"];

        let comparison = sampler.validate_sample(
            &population,
            &sample,
            |item| item.to_string(),
            0.1,
        );
        assert!(comparison.is_representative);
        assert!(comparison.max_deviation <= 0.1);
    }

    #[test]
    fn validate_sample_not_representative() {
        let sampler = DataSampler::new(42);
        let population: Vec<&str> = vec!["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"];
        let sample: Vec<&str> = vec!["A", "A", "A", "A"]; // All A, no B

        let comparison = sampler.validate_sample(
            &population,
            &sample,
            |item| item.to_string(),
            0.1,
        );
        assert!(!comparison.is_representative);
        assert!(comparison.max_deviation > 0.1);
    }

    #[test]
    fn sampling_rate() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..100).collect();
        let result = sampler.random_sample(&data, 25).unwrap();
        assert!((result.sampling_rate - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn rng_different_seeds() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        // Very unlikely to produce same first value.
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn error_display() {
        let e = SamplerError::EmptyDataset;
        assert!(format!("{e}").contains("empty dataset"));
        let e2 = SamplerError::SampleTooLarge {
            requested: 100,
            available: 50,
        };
        assert!(format!("{e2}").contains("100"));
    }

    #[test]
    fn indices_are_sorted() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..50).collect();
        let result = sampler.random_sample(&data, 20).unwrap();
        let sorted = result.indices.windows(2).all(|w| w[0] <= w[1]);
        assert!(sorted);
    }

    #[test]
    fn full_sample() {
        let mut sampler = DataSampler::new(42);
        let data: Vec<i32> = (0..10).collect();
        let result = sampler.random_sample(&data, 10).unwrap();
        assert_eq!(result.sample_size, 10);
        let mut sorted_items = result.items.clone();
        sorted_items.sort();
        assert_eq!(sorted_items, data);
    }

    #[test]
    fn distribution_comparison_strata_sorted() {
        let sampler = DataSampler::new(42);
        let pop = vec!["C", "A", "B", "A", "C", "B"];
        let sample = vec!["A", "B", "C"];

        let comp = sampler.validate_sample(&pop, &sample, |x| x.to_string(), 1.0);
        let labels: Vec<&str> = comp.strata.iter().map(|s| s.label.as_str()).collect();
        assert_eq!(labels, vec!["A", "B", "C"]);
    }
}
