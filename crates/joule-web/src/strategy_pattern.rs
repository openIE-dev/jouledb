//! Strategy pattern — runtime-swappable algorithms with registry.
//!
//! Provides a `Strategy` trait, a `Context` that delegates to a chosen
//! strategy, built-in sorting strategies (bubble, insertion, merge, quick),
//! a comparison-based abstraction, and a `StrategyRegistry` for name-based
//! lookup and criteria-based selection.

use std::collections::HashMap;
use std::fmt;

// ── Strategy trait ─────────────────────────────────────────────────

/// A named strategy that operates on a mutable slice of `i64`.
pub trait Strategy: fmt::Debug {
    /// Human-readable name of this strategy.
    fn name(&self) -> &str;

    /// Execute the strategy on the given data.
    fn execute(&self, data: &mut [i64]);

    /// Worst-case time complexity as a descriptive string.
    fn complexity(&self) -> &str;

    /// Whether this strategy is stable (preserves relative order of equal elements).
    fn is_stable(&self) -> bool;
}

// ── Built-in sorting strategies ────────────────────────────────────

/// Bubble sort — O(n^2), stable.
#[derive(Debug, Clone)]
pub struct BubbleSort;

impl Strategy for BubbleSort {
    fn name(&self) -> &str {
        "bubble_sort"
    }

    fn execute(&self, data: &mut [i64]) {
        let n = data.len();
        for i in 0..n {
            let mut swapped = false;
            for j in 0..n.saturating_sub(i + 1) {
                if data[j] > data[j + 1] {
                    data.swap(j, j + 1);
                    swapped = true;
                }
            }
            if !swapped {
                break;
            }
        }
    }

    fn complexity(&self) -> &str {
        "O(n^2)"
    }

    fn is_stable(&self) -> bool {
        true
    }
}

/// Insertion sort — O(n^2), stable.
#[derive(Debug, Clone)]
pub struct InsertionSort;

impl Strategy for InsertionSort {
    fn name(&self) -> &str {
        "insertion_sort"
    }

    fn execute(&self, data: &mut [i64]) {
        for i in 1..data.len() {
            let mut j = i;
            while j > 0 && data[j - 1] > data[j] {
                data.swap(j - 1, j);
                j -= 1;
            }
        }
    }

    fn complexity(&self) -> &str {
        "O(n^2)"
    }

    fn is_stable(&self) -> bool {
        true
    }
}

/// Merge sort — O(n log n), stable.
#[derive(Debug, Clone)]
pub struct MergeSort;

impl MergeSort {
    fn merge_sort_impl(data: &mut [i64]) {
        let n = data.len();
        if n <= 1 {
            return;
        }
        let mid = n / 2;
        Self::merge_sort_impl(&mut data[..mid]);
        Self::merge_sort_impl(&mut data[mid..]);
        // merge
        let left: Vec<i64> = data[..mid].to_vec();
        let right: Vec<i64> = data[mid..].to_vec();
        let (mut i, mut j, mut k) = (0, 0, 0);
        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                data[k] = left[i];
                i += 1;
            } else {
                data[k] = right[j];
                j += 1;
            }
            k += 1;
        }
        while i < left.len() {
            data[k] = left[i];
            i += 1;
            k += 1;
        }
        while j < right.len() {
            data[k] = right[j];
            j += 1;
            k += 1;
        }
    }
}

impl Strategy for MergeSort {
    fn name(&self) -> &str {
        "merge_sort"
    }

    fn execute(&self, data: &mut [i64]) {
        Self::merge_sort_impl(data);
    }

    fn complexity(&self) -> &str {
        "O(n log n)"
    }

    fn is_stable(&self) -> bool {
        true
    }
}

/// Quick sort — O(n log n) average, unstable.
#[derive(Debug, Clone)]
pub struct QuickSort;

impl QuickSort {
    fn quick_sort_impl(data: &mut [i64]) {
        if data.len() <= 1 {
            return;
        }
        let pivot_idx = Self::partition(data);
        if pivot_idx > 0 {
            Self::quick_sort_impl(&mut data[..pivot_idx]);
        }
        Self::quick_sort_impl(&mut data[pivot_idx + 1..]);
    }

    fn partition(data: &mut [i64]) -> usize {
        let len = data.len();
        let pivot = data[len - 1];
        let mut i = 0usize;
        for j in 0..len - 1 {
            if data[j] <= pivot {
                data.swap(i, j);
                i += 1;
            }
        }
        data.swap(i, len - 1);
        i
    }
}

impl Strategy for QuickSort {
    fn name(&self) -> &str {
        "quick_sort"
    }

    fn execute(&self, data: &mut [i64]) {
        Self::quick_sort_impl(data);
    }

    fn complexity(&self) -> &str {
        "O(n log n) average"
    }

    fn is_stable(&self) -> bool {
        false
    }
}

// ── Comparison strategy ────────────────────────────────────────────

/// A strategy that sorts using a custom comparison function.
pub struct ComparisonSort {
    label: String,
    cmp: Box<dyn Fn(&i64, &i64) -> std::cmp::Ordering + Send + Sync>,
}

impl ComparisonSort {
    /// Create with a name and comparison closure.
    pub fn new(
        label: impl Into<String>,
        cmp: impl Fn(&i64, &i64) -> std::cmp::Ordering + Send + Sync + 'static,
    ) -> Self {
        Self {
            label: label.into(),
            cmp: Box::new(cmp),
        }
    }
}

impl fmt::Debug for ComparisonSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComparisonSort")
            .field("label", &self.label)
            .finish()
    }
}

impl Strategy for ComparisonSort {
    fn name(&self) -> &str {
        &self.label
    }

    fn execute(&self, data: &mut [i64]) {
        data.sort_by(&self.cmp);
    }

    fn complexity(&self) -> &str {
        "O(n log n)"
    }

    fn is_stable(&self) -> bool {
        true
    }
}

// ── Context ────────────────────────────────────────────────────────

/// Holds a strategy and delegates execution to it.
pub struct Context {
    strategy: Box<dyn Strategy>,
    execution_count: u64,
}

impl Context {
    /// Create a new context with the given strategy.
    pub fn new(strategy: Box<dyn Strategy>) -> Self {
        Self {
            strategy,
            execution_count: 0,
        }
    }

    /// Swap the strategy at runtime.
    pub fn set_strategy(&mut self, strategy: Box<dyn Strategy>) {
        self.strategy = strategy;
    }

    /// Return the name of the current strategy.
    pub fn strategy_name(&self) -> &str {
        self.strategy.name()
    }

    /// Execute the current strategy on the data.
    pub fn execute(&mut self, data: &mut [i64]) {
        self.strategy.execute(data);
        self.execution_count += 1;
    }

    /// How many times `execute` has been called.
    pub fn execution_count(&self) -> u64 {
        self.execution_count
    }

    /// Complexity of the current strategy.
    pub fn complexity(&self) -> &str {
        self.strategy.complexity()
    }

    /// Whether the current strategy is stable.
    pub fn is_stable(&self) -> bool {
        self.strategy.is_stable()
    }
}

// ── Selection criteria ─────────────────────────────────────────────

/// Criteria used to select a strategy from the registry.
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Prefer stable strategies.
    pub prefer_stable: bool,
    /// Maximum acceptable complexity class (e.g. "O(n log n)").
    pub max_complexity: Option<String>,
    /// Approximate input size — used as a heuristic.
    pub input_size: usize,
}

impl SelectionCriteria {
    pub fn new() -> Self {
        Self {
            prefer_stable: false,
            max_complexity: None,
            input_size: 0,
        }
    }

    pub fn with_prefer_stable(mut self, stable: bool) -> Self {
        self.prefer_stable = stable;
        self
    }

    pub fn with_max_complexity(mut self, complexity: impl Into<String>) -> Self {
        self.max_complexity = Some(complexity.into());
        self
    }

    pub fn with_input_size(mut self, size: usize) -> Self {
        self.input_size = size;
        self
    }
}

impl Default for SelectionCriteria {
    fn default() -> Self {
        Self::new()
    }
}

// ── Strategy metadata ──────────────────────────────────────────────

/// Metadata about a registered strategy.
#[derive(Debug, Clone)]
pub struct StrategyInfo {
    pub name: String,
    pub complexity: String,
    pub is_stable: bool,
}

// ── Registry ───────────────────────────────────────────────────────

/// Maps strategy names to factory closures.
pub struct StrategyRegistry {
    factories: HashMap<String, Box<dyn Fn() -> Box<dyn Strategy>>>,
    metadata: Vec<StrategyInfo>,
}

impl StrategyRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            metadata: Vec::new(),
        }
    }

    /// Create a registry pre-loaded with all built-in strategies.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register("bubble_sort", || Box::new(BubbleSort));
        reg.register("insertion_sort", || Box::new(InsertionSort));
        reg.register("merge_sort", || Box::new(MergeSort));
        reg.register("quick_sort", || Box::new(QuickSort));
        reg
    }

    /// Register a strategy factory under the given name.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        factory: impl Fn() -> Box<dyn Strategy> + 'static,
    ) {
        let key = name.into();
        // Probe the factory to capture metadata.
        let instance = factory();
        let info = StrategyInfo {
            name: key.clone(),
            complexity: instance.complexity().to_string(),
            is_stable: instance.is_stable(),
        };
        // Remove any prior entry from metadata.
        self.metadata.retain(|m| m.name != key);
        self.metadata.push(info);
        self.factories.insert(key, Box::new(factory));
    }

    /// Look up a strategy by exact name.
    pub fn get(&self, name: &str) -> Option<Box<dyn Strategy>> {
        self.factories.get(name).map(|f| f())
    }

    /// Return the number of registered strategies.
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// All registered strategy names (sorted for determinism).
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.factories.keys().cloned().collect();
        names.sort();
        names
    }

    /// Metadata for all registered strategies (sorted by name for determinism).
    pub fn metadata(&self) -> Vec<StrategyInfo> {
        let mut meta = self.metadata.clone();
        meta.sort_by(|a, b| a.name.cmp(&b.name));
        meta
    }

    /// Remove a strategy by name. Returns true if it was present.
    pub fn remove(&mut self, name: &str) -> bool {
        self.metadata.retain(|m| m.name != name);
        self.factories.remove(name).is_some()
    }

    /// Select the best strategy based on criteria.
    ///
    /// Scoring heuristic:
    /// - Stability match: +10
    /// - O(n log n) complexity: +5, O(n^2): +1
    /// - For small inputs (< 50), quadratic sorts get a bonus because they
    ///   have low overhead.
    pub fn select(&self, criteria: &SelectionCriteria) -> Option<Box<dyn Strategy>> {
        if self.metadata.is_empty() {
            return None;
        }

        let mut best_name: Option<&str> = None;
        let mut best_score = i64::MIN;

        for info in &self.metadata {
            let mut score: i64 = 0;

            // Stability preference.
            if criteria.prefer_stable && info.is_stable {
                score += 10;
            }

            // Complexity scoring.
            let is_loglinear = info.complexity.contains("log");
            let is_quadratic = info.complexity.contains("n^2");

            if is_loglinear {
                score += 5;
            } else if is_quadratic {
                score += 1;
            }

            // For small inputs, quadratic sorts with low overhead win.
            if criteria.input_size > 0 && criteria.input_size < 50 && is_quadratic {
                score += 8;
            }

            // Complexity ceiling check.
            if let Some(max) = &criteria.max_complexity {
                let max_is_loglinear = max.contains("log");
                if max_is_loglinear && is_quadratic {
                    // Quadratic exceeds the ceiling — penalize heavily.
                    score -= 100;
                }
            }

            if score > best_score {
                best_score = score;
                best_name = Some(&info.name);
            }
        }

        best_name.and_then(|n| self.get(n))
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_copy(data: &[i64]) -> Vec<i64> {
        let mut v = data.to_vec();
        v.sort();
        v
    }

    #[test]
    fn bubble_sort_basic() {
        let mut data = vec![5, 3, 8, 1, 2];
        BubbleSort.execute(&mut data);
        assert_eq!(data, sorted_copy(&[5, 3, 8, 1, 2]));
    }

    #[test]
    fn insertion_sort_basic() {
        let mut data = vec![9, 4, 7, 2, 6];
        InsertionSort.execute(&mut data);
        assert_eq!(data, sorted_copy(&[9, 4, 7, 2, 6]));
    }

    #[test]
    fn merge_sort_basic() {
        let mut data = vec![10, 1, 5, 3, 8, 7];
        MergeSort.execute(&mut data);
        assert_eq!(data, sorted_copy(&[10, 1, 5, 3, 8, 7]));
    }

    #[test]
    fn quick_sort_basic() {
        let mut data = vec![4, 2, 9, 1, 5, 3];
        QuickSort.execute(&mut data);
        assert_eq!(data, sorted_copy(&[4, 2, 9, 1, 5, 3]));
    }

    #[test]
    fn empty_slice() {
        for strategy in [
            Box::new(BubbleSort) as Box<dyn Strategy>,
            Box::new(InsertionSort),
            Box::new(MergeSort),
            Box::new(QuickSort),
        ] {
            let mut data: Vec<i64> = vec![];
            strategy.execute(&mut data);
            assert!(data.is_empty());
        }
    }

    #[test]
    fn single_element() {
        let mut data = vec![42];
        MergeSort.execute(&mut data);
        assert_eq!(data, vec![42]);
    }

    #[test]
    fn already_sorted() {
        let mut data = vec![1, 2, 3, 4, 5];
        BubbleSort.execute(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn reverse_sorted() {
        let mut data = vec![5, 4, 3, 2, 1];
        InsertionSort.execute(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn duplicate_values() {
        let mut data = vec![3, 1, 3, 1, 2, 2];
        MergeSort.execute(&mut data);
        assert_eq!(data, vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    fn comparison_sort_descending() {
        let cs = ComparisonSort::new("desc", |a, b| b.cmp(a));
        let mut data = vec![1, 5, 3, 2, 4];
        cs.execute(&mut data);
        assert_eq!(data, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn comparison_sort_by_absolute() {
        let cs = ComparisonSort::new("abs_sort", |a, b| a.abs().cmp(&b.abs()));
        let mut data = vec![-5, 3, -1, 4, -2];
        cs.execute(&mut data);
        assert_eq!(data, vec![-1, -2, 3, 4, -5]);
    }

    #[test]
    fn context_swap_strategy() {
        let mut ctx = Context::new(Box::new(BubbleSort));
        assert_eq!(ctx.strategy_name(), "bubble_sort");
        assert!(ctx.is_stable());

        ctx.set_strategy(Box::new(QuickSort));
        assert_eq!(ctx.strategy_name(), "quick_sort");
        assert!(!ctx.is_stable());
    }

    #[test]
    fn context_execution_count() {
        let mut ctx = Context::new(Box::new(MergeSort));
        assert_eq!(ctx.execution_count(), 0);

        ctx.execute(&mut [3, 1, 2]);
        ctx.execute(&mut [5, 4]);
        assert_eq!(ctx.execution_count(), 2);
    }

    #[test]
    fn context_complexity() {
        let ctx = Context::new(Box::new(BubbleSort));
        assert_eq!(ctx.complexity(), "O(n^2)");
    }

    #[test]
    fn registry_with_builtins() {
        let reg = StrategyRegistry::with_builtins();
        assert_eq!(reg.len(), 4);
        let names = reg.names();
        assert!(names.contains(&"bubble_sort".to_string()));
        assert!(names.contains(&"merge_sort".to_string()));
    }

    #[test]
    fn registry_get_and_execute() {
        let reg = StrategyRegistry::with_builtins();
        let s = reg.get("merge_sort").unwrap();
        let mut data = vec![5, 1, 3];
        s.execute(&mut data);
        assert_eq!(data, vec![1, 3, 5]);
    }

    #[test]
    fn registry_custom_strategy() {
        let mut reg = StrategyRegistry::new();
        reg.register("custom_desc", || {
            Box::new(ComparisonSort::new("custom_desc", |a, b| b.cmp(a)))
        });
        assert_eq!(reg.len(), 1);
        let s = reg.get("custom_desc").unwrap();
        let mut data = vec![1, 3, 2];
        s.execute(&mut data);
        assert_eq!(data, vec![3, 2, 1]);
    }

    #[test]
    fn registry_remove() {
        let mut reg = StrategyRegistry::with_builtins();
        assert!(reg.remove("bubble_sort"));
        assert_eq!(reg.len(), 3);
        assert!(reg.get("bubble_sort").is_none());
        assert!(!reg.remove("nonexistent"));
    }

    #[test]
    fn registry_metadata() {
        let reg = StrategyRegistry::with_builtins();
        let meta = reg.metadata();
        assert_eq!(meta.len(), 4);
        let merge_meta = meta.iter().find(|m| m.name == "merge_sort").unwrap();
        assert!(merge_meta.is_stable);
        assert_eq!(merge_meta.complexity, "O(n log n)");
    }

    #[test]
    fn select_stable_for_large_input() {
        let reg = StrategyRegistry::with_builtins();
        let criteria = SelectionCriteria::new()
            .with_prefer_stable(true)
            .with_input_size(1000)
            .with_max_complexity("O(n log n)");
        let s = reg.select(&criteria).unwrap();
        assert!(s.is_stable());
        assert!(s.complexity().contains("log"));
    }

    #[test]
    fn select_small_input_prefers_simple() {
        let reg = StrategyRegistry::with_builtins();
        let criteria = SelectionCriteria::new().with_input_size(10);
        let s = reg.select(&criteria).unwrap();
        // For small inputs, quadratic sorts with low overhead should be preferred.
        let c = s.complexity();
        assert!(c.contains("n^2"));
    }

    #[test]
    fn select_from_empty_registry() {
        let reg = StrategyRegistry::new();
        let criteria = SelectionCriteria::new();
        assert!(reg.select(&criteria).is_none());
    }

    #[test]
    fn strategy_names_sorted() {
        let reg = StrategyRegistry::with_builtins();
        let names = reg.names();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn negative_values() {
        let mut data = vec![-3, -1, -5, 0, 2];
        QuickSort.execute(&mut data);
        assert_eq!(data, vec![-5, -3, -1, 0, 2]);
    }
}
