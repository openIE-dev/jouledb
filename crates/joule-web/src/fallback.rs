//! Fallback strategies — primary/secondary/tertiary chains, cached fallback,
//! default fallback, timeout fallback, fallback metrics, and composition.
//!
//! Pure Rust fallback infrastructure for resilient operations.
//! Each fallback chain tries strategies in order and tracks which level succeeded.

use std::collections::HashMap;

// ── Fallback Level ──────────────────────────────────────────────

/// Which level in the fallback chain succeeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FallbackLevel {
    Primary,
    Secondary,
    Tertiary,
    Nth(usize),
    Default,
    Cached,
    None,
}

impl FallbackLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Secondary => "secondary",
            Self::Tertiary => "tertiary",
            Self::Nth(_) => "nth",
            Self::Default => "default",
            Self::Cached => "cached",
            Self::None => "none",
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Self::Primary => 0,
            Self::Secondary => 1,
            Self::Tertiary => 2,
            Self::Nth(n) => *n,
            Self::Default => usize::MAX - 1,
            Self::Cached => usize::MAX - 2,
            Self::None => usize::MAX,
        }
    }
}

// ── Fallback Result ─────────────────────────────────────────────

/// Result of a fallback chain execution.
#[derive(Debug, Clone)]
pub struct FallbackResult<T> {
    /// The value produced (if any).
    pub value: Option<T>,
    /// Which level succeeded.
    pub level: FallbackLevel,
    /// Errors from each attempted level.
    pub errors: Vec<(FallbackLevel, String)>,
    /// How many levels were attempted.
    pub attempts: usize,
}

impl<T> FallbackResult<T> {
    pub fn success(value: T, level: FallbackLevel, errors: Vec<(FallbackLevel, String)>, attempts: usize) -> Self {
        Self {
            value: Some(value),
            level,
            errors,
            attempts,
        }
    }

    pub fn failure(errors: Vec<(FallbackLevel, String)>, attempts: usize) -> Self {
        Self {
            value: None,
            level: FallbackLevel::None,
            errors,
            attempts,
        }
    }

    pub fn succeeded(&self) -> bool {
        self.value.is_some()
    }

    pub fn used_fallback(&self) -> bool {
        self.level != FallbackLevel::Primary && self.level != FallbackLevel::None
    }
}

// ── Fallback Metrics ────────────────────────────────────────────

/// Tracks how often each fallback level is used.
#[derive(Debug, Clone, Default)]
pub struct FallbackMetrics {
    /// Count of successes at each level.
    level_counts: Vec<(FallbackLevel, u64)>,
    /// Total invocations.
    total_invocations: u64,
    /// Total failures (all levels exhausted).
    total_failures: u64,
}

impl FallbackMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_success(&mut self, level: FallbackLevel) {
        self.total_invocations += 1;
        if let Some(entry) = self.level_counts.iter_mut().find(|(l, _)| *l == level) {
            entry.1 += 1;
        } else {
            self.level_counts.push((level, 1));
        }
    }

    pub fn record_failure(&mut self) {
        self.total_invocations += 1;
        self.total_failures += 1;
    }

    pub fn count_for_level(&self, level: FallbackLevel) -> u64 {
        self.level_counts
            .iter()
            .find(|(l, _)| *l == level)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }

    pub fn total_invocations(&self) -> u64 {
        self.total_invocations
    }

    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }

    pub fn primary_success_rate(&self) -> f64 {
        if self.total_invocations == 0 {
            return 0.0;
        }
        self.count_for_level(FallbackLevel::Primary) as f64 / self.total_invocations as f64
    }

    pub fn fallback_rate(&self) -> f64 {
        if self.total_invocations == 0 {
            return 0.0;
        }
        let primary = self.count_for_level(FallbackLevel::Primary);
        let non_primary = self.total_invocations - primary - self.total_failures;
        non_primary as f64 / self.total_invocations as f64
    }

    pub fn reset(&mut self) {
        self.level_counts.clear();
        self.total_invocations = 0;
        self.total_failures = 0;
    }
}

// ── Fallback Chain ──────────────────────────────────────────────

/// Outcome from a single strategy attempt.
#[derive(Debug)]
pub enum StrategyOutcome<T> {
    Ok(T),
    Err(String),
}

/// A chain of fallback strategies evaluated in order.
///
/// Usage: Register closures that produce `StrategyOutcome<T>`, then execute.
pub struct FallbackChain<T> {
    strategies: Vec<(String, Box<dyn Fn() -> StrategyOutcome<T>>)>,
    default_value: Option<T>,
    cached_value: Option<T>,
    metrics: FallbackMetrics,
}

impl<T: Clone> FallbackChain<T> {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            default_value: None,
            cached_value: None,
            metrics: FallbackMetrics::new(),
        }
    }

    /// Add a named strategy to the chain.
    pub fn add_strategy(
        &mut self,
        name: impl Into<String>,
        strategy: impl Fn() -> StrategyOutcome<T> + 'static,
    ) {
        self.strategies.push((name.into(), Box::new(strategy)));
    }

    /// Set the default value returned when all strategies fail.
    pub fn set_default(&mut self, value: T) {
        self.default_value = Some(value);
    }

    /// Set a cached value that can be used as fallback.
    pub fn set_cached(&mut self, value: T) {
        self.cached_value = Some(value);
    }

    /// Execute the chain, trying each strategy in order.
    pub fn execute(&mut self) -> FallbackResult<T> {
        let mut errors = Vec::new();
        let mut attempts = 0;

        for (i, (name, strategy)) in self.strategies.iter().enumerate() {
            attempts += 1;
            let level = match i {
                0 => FallbackLevel::Primary,
                1 => FallbackLevel::Secondary,
                2 => FallbackLevel::Tertiary,
                n => FallbackLevel::Nth(n),
            };

            match strategy() {
                StrategyOutcome::Ok(val) => {
                    self.metrics.record_success(level);
                    self.cached_value = Some(val.clone());
                    return FallbackResult::success(val, level, errors, attempts);
                }
                StrategyOutcome::Err(msg) => {
                    errors.push((level, format!("{name}: {msg}")));
                }
            }
        }

        // Try cached value.
        if let Some(cached) = &self.cached_value {
            attempts += 1;
            self.metrics.record_success(FallbackLevel::Cached);
            return FallbackResult::success(cached.clone(), FallbackLevel::Cached, errors, attempts);
        }

        // Try default.
        if let Some(default) = &self.default_value {
            attempts += 1;
            self.metrics.record_success(FallbackLevel::Default);
            return FallbackResult::success(default.clone(), FallbackLevel::Default, errors, attempts);
        }

        self.metrics.record_failure();
        FallbackResult::failure(errors, attempts)
    }

    pub fn metrics(&self) -> &FallbackMetrics {
        &self.metrics
    }

    pub fn strategy_count(&self) -> usize {
        self.strategies.len()
    }
}

// ── Timeout Fallback ────────────────────────────────────────────

/// Simulated timeout fallback: if the primary elapsed_ms exceeds timeout_ms,
/// use fallback value.
#[derive(Debug, Clone)]
pub struct TimeoutFallback<T> {
    timeout_ms: u64,
    fallback_value: T,
    timeout_count: u64,
    success_count: u64,
}

impl<T: Clone> TimeoutFallback<T> {
    pub fn new(timeout_ms: u64, fallback_value: T) -> Self {
        Self {
            timeout_ms,
            fallback_value,
            timeout_count: 0,
            success_count: 0,
        }
    }

    /// Evaluate: if elapsed_ms > timeout_ms, return fallback; else primary.
    pub fn evaluate(&mut self, primary: T, elapsed_ms: u64) -> (T, bool) {
        if elapsed_ms > self.timeout_ms {
            self.timeout_count += 1;
            (self.fallback_value.clone(), true)
        } else {
            self.success_count += 1;
            (primary, false)
        }
    }

    pub fn timeout_count(&self) -> u64 {
        self.timeout_count
    }

    pub fn success_count(&self) -> u64 {
        self.success_count
    }

    pub fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }
}

// ── Fallback Map ────────────────────────────────────────────────

/// Named fallback chains for different operations.
pub struct FallbackMap<T: Clone + 'static> {
    chains: HashMap<String, FallbackChain<T>>,
}

impl<T: Clone + 'static> FallbackMap<T> {
    pub fn new() -> Self {
        Self {
            chains: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, chain: FallbackChain<T>) {
        self.chains.insert(name.into(), chain);
    }

    pub fn execute(&mut self, name: &str) -> Option<FallbackResult<T>> {
        self.chains.get_mut(name).map(|chain| chain.execute())
    }

    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.chains.keys().map(|k| k.as_str()).collect();
        names.sort();
        names
    }

    pub fn len(&self) -> usize {
        self.chains.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }
}

// ── Composed Fallback ───────────────────────────────────────────

/// Compose multiple fallback values with priority.
#[derive(Debug, Clone)]
pub struct FallbackComposer<T> {
    values: Vec<(String, Option<T>)>,
    default: Option<T>,
}

impl<T: Clone> FallbackComposer<T> {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            default: None,
        }
    }

    /// Add a named value source (None = not available).
    pub fn add(&mut self, name: impl Into<String>, value: Option<T>) {
        self.values.push((name.into(), value));
    }

    /// Set the absolute-last-resort default.
    pub fn set_default(&mut self, value: T) {
        self.default = Some(value);
    }

    /// Resolve: return the first available value, or default.
    pub fn resolve(&self) -> Option<(T, &str)> {
        for (name, val) in &self.values {
            if let Some(v) = val {
                return Some((v.clone(), name.as_str()));
            }
        }
        self.default.as_ref().map(|v| (v.clone(), "default"))
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_succeeds() {
        let mut chain = FallbackChain::new();
        chain.add_strategy("primary", || StrategyOutcome::Ok(42));
        chain.add_strategy("secondary", || StrategyOutcome::Ok(99));
        let result = chain.execute();
        assert!(result.succeeded());
        assert_eq!(result.value, Some(42));
        assert_eq!(result.level, FallbackLevel::Primary);
        assert_eq!(result.attempts, 1);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_fallback_to_secondary() {
        let mut chain = FallbackChain::new();
        chain.add_strategy("primary", || StrategyOutcome::<i32>::Err("down".into()));
        chain.add_strategy("secondary", || StrategyOutcome::Ok(99));
        let result = chain.execute();
        assert!(result.succeeded());
        assert_eq!(result.value, Some(99));
        assert_eq!(result.level, FallbackLevel::Secondary);
        assert!(result.used_fallback());
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_fallback_to_tertiary() {
        let mut chain = FallbackChain::new();
        chain.add_strategy("p", || StrategyOutcome::<i32>::Err("err1".into()));
        chain.add_strategy("s", || StrategyOutcome::<i32>::Err("err2".into()));
        chain.add_strategy("t", || StrategyOutcome::Ok(77));
        let result = chain.execute();
        assert_eq!(result.level, FallbackLevel::Tertiary);
        assert_eq!(result.value, Some(77));
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_fallback_to_default() {
        let mut chain = FallbackChain::<i32>::new();
        chain.add_strategy("p", || StrategyOutcome::Err("fail".into()));
        chain.set_default(0);
        let result = chain.execute();
        assert_eq!(result.level, FallbackLevel::Default);
        assert_eq!(result.value, Some(0));
    }

    #[test]
    fn test_fallback_to_cached() {
        let mut chain = FallbackChain::<String>::new();
        chain.set_cached("cached_val".to_string());
        chain.add_strategy("p", || StrategyOutcome::Err("fail".into()));
        let result = chain.execute();
        assert_eq!(result.level, FallbackLevel::Cached);
        assert_eq!(result.value.as_deref(), Some("cached_val"));
    }

    #[test]
    fn test_all_fail() {
        let mut chain = FallbackChain::<i32>::new();
        chain.add_strategy("a", || StrategyOutcome::Err("e1".into()));
        chain.add_strategy("b", || StrategyOutcome::Err("e2".into()));
        let result = chain.execute();
        assert!(!result.succeeded());
        assert_eq!(result.level, FallbackLevel::None);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_metrics_primary() {
        let mut chain = FallbackChain::new();
        chain.add_strategy("p", || StrategyOutcome::Ok(1));
        chain.execute();
        chain.execute();
        assert_eq!(chain.metrics().count_for_level(FallbackLevel::Primary), 2);
        assert_eq!(chain.metrics().total_invocations(), 2);
    }

    #[test]
    fn test_metrics_fallback_rate() {
        let mut chain = FallbackChain::new();
        chain.add_strategy("p", || StrategyOutcome::Ok(1));
        chain.execute();
        assert!((chain.metrics().primary_success_rate() - 1.0).abs() < f64::EPSILON);
        assert!((chain.metrics().fallback_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_failure() {
        let mut chain = FallbackChain::<i32>::new();
        chain.add_strategy("p", || StrategyOutcome::Err("fail".into()));
        chain.execute();
        assert_eq!(chain.metrics().total_failures(), 1);
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = FallbackMetrics::new();
        metrics.record_success(FallbackLevel::Primary);
        metrics.record_failure();
        metrics.reset();
        assert_eq!(metrics.total_invocations(), 0);
        assert_eq!(metrics.total_failures(), 0);
    }

    #[test]
    fn test_timeout_fallback_succeeds() {
        let mut tf = TimeoutFallback::new(100, "fallback");
        let (val, timed_out) = tf.evaluate("ok", 50);
        assert_eq!(val, "ok");
        assert!(!timed_out);
        assert_eq!(tf.success_count(), 1);
    }

    #[test]
    fn test_timeout_fallback_exceeds() {
        let mut tf = TimeoutFallback::new(100, "fallback");
        let (val, timed_out) = tf.evaluate("ok", 200);
        assert_eq!(val, "fallback");
        assert!(timed_out);
        assert_eq!(tf.timeout_count(), 1);
    }

    #[test]
    fn test_fallback_level_as_str() {
        assert_eq!(FallbackLevel::Primary.as_str(), "primary");
        assert_eq!(FallbackLevel::Secondary.as_str(), "secondary");
        assert_eq!(FallbackLevel::Tertiary.as_str(), "tertiary");
        assert_eq!(FallbackLevel::Default.as_str(), "default");
        assert_eq!(FallbackLevel::Cached.as_str(), "cached");
        assert_eq!(FallbackLevel::None.as_str(), "none");
        assert_eq!(FallbackLevel::Nth(5).as_str(), "nth");
    }

    #[test]
    fn test_fallback_level_index() {
        assert_eq!(FallbackLevel::Primary.index(), 0);
        assert_eq!(FallbackLevel::Secondary.index(), 1);
        assert_eq!(FallbackLevel::Tertiary.index(), 2);
        assert_eq!(FallbackLevel::Nth(7).index(), 7);
    }

    #[test]
    fn test_fallback_map() {
        let mut map = FallbackMap::<i32>::new();
        let mut chain = FallbackChain::new();
        chain.add_strategy("p", || StrategyOutcome::Ok(10));
        map.register("op_a", chain);
        assert_eq!(map.len(), 1);
        let result = map.execute("op_a").unwrap();
        assert_eq!(result.value, Some(10));
        assert!(map.execute("nonexistent").is_none());
    }

    #[test]
    fn test_fallback_map_names() {
        let mut map = FallbackMap::<i32>::new();
        map.register("beta", FallbackChain::new());
        map.register("alpha", FallbackChain::new());
        let names = map.names();
        assert_eq!(names, vec!["alpha", "beta"]); // Sorted.
    }

    #[test]
    fn test_composer_resolve() {
        let mut comp = FallbackComposer::<i32>::new();
        comp.add("db", None);
        comp.add("cache", Some(42));
        comp.add("default", Some(0));
        let (val, source) = comp.resolve().unwrap();
        assert_eq!(val, 42);
        assert_eq!(source, "cache");
    }

    #[test]
    fn test_composer_default() {
        let mut comp = FallbackComposer::<i32>::new();
        comp.add("db", None);
        comp.set_default(0);
        let (val, source) = comp.resolve().unwrap();
        assert_eq!(val, 0);
        assert_eq!(source, "default");
    }

    #[test]
    fn test_composer_none() {
        let comp = FallbackComposer::<i32>::new();
        assert!(comp.resolve().is_none());
    }

    #[test]
    fn test_cached_updated_on_success() {
        let mut chain = FallbackChain::new();
        chain.add_strategy("p", || StrategyOutcome::Ok(42));
        chain.execute();
        // Now if primary fails, cached should be 42.
        chain.strategies.clear();
        chain.add_strategy("p", || StrategyOutcome::<i32>::Err("fail".into()));
        let result = chain.execute();
        assert_eq!(result.level, FallbackLevel::Cached);
        assert_eq!(result.value, Some(42));
    }

    #[test]
    fn test_strategy_count() {
        let mut chain = FallbackChain::<i32>::new();
        assert_eq!(chain.strategy_count(), 0);
        chain.add_strategy("a", || StrategyOutcome::Ok(1));
        chain.add_strategy("b", || StrategyOutcome::Ok(2));
        assert_eq!(chain.strategy_count(), 2);
    }
}
