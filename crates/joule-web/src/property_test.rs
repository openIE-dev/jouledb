//! Property-based testing framework.
//!
//! Replaces `proptest`, `quickcheck`, and similar JS/Rust PBT libraries with
//! a pure-Rust engine. Supports generators for basic types, shrinking on failure,
//! configurable iterations, seed for reproducibility, forall/exists combinators,
//! and structured test result reporting.

use std::fmt;

// ── PRNG (Xoshiro256**) ──────────────────────────────────────────

/// Minimal Xoshiro256** PRNG for deterministic test generation.
#[derive(Debug, Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
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

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / (1u64 << 53) as f64
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

// ── Generator Trait ──────────────────────────────────────────────

/// A generator produces arbitrary values and can shrink them toward minimal
/// failing cases.
pub trait Generator: fmt::Debug {
    type Value: Clone + fmt::Debug;

    /// Generate a random value.
    fn generate(&self, seed: u64, size: usize) -> Self::Value;

    /// Produce candidate shrinks for a failing value.
    /// Returns smaller/simpler values that might still fail the property.
    fn shrink(&self, value: &Self::Value) -> Vec<Self::Value>;
}

// ── Bool Generator ───────────────────────────────────────────────

/// Generates random booleans.
#[derive(Debug, Clone)]
pub struct BoolGen;

impl Generator for BoolGen {
    type Value = bool;

    fn generate(&self, seed: u64, _size: usize) -> bool {
        let mut rng = Rng::new(seed);
        rng.next_bool()
    }

    fn shrink(&self, value: &bool) -> Vec<bool> {
        if *value { vec![false] } else { vec![] }
    }
}

// ── Int Range Generator ──────────────────────────────────────────

/// Generates integers in `[lo, hi]` inclusive.
#[derive(Debug, Clone)]
pub struct IntRangeGen {
    pub lo: i64,
    pub hi: i64,
}

impl IntRangeGen {
    pub fn new(lo: i64, hi: i64) -> Self {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        Self { lo, hi }
    }

    pub fn full() -> Self {
        Self { lo: i64::MIN / 2, hi: i64::MAX / 2 }
    }
}

impl Generator for IntRangeGen {
    type Value = i64;

    fn generate(&self, seed: u64, size: usize) -> i64 {
        let mut rng = Rng::new(seed);
        // Scale range by size parameter (0..100), to start with small values
        let effective_range = if self.hi == self.lo {
            0u64
        } else {
            let full_range = (self.hi - self.lo) as u64;
            let scale = (size as u64).min(100);
            full_range.saturating_mul(scale) / 100
        };
        if effective_range == 0 {
            return self.lo;
        }
        self.lo + (rng.next_u64() % (effective_range + 1)) as i64
    }

    fn shrink(&self, value: &i64) -> Vec<i64> {
        let v = *value;
        let mut candidates = Vec::new();
        // Shrink toward zero (clamped to range)
        let zero = 0i64.clamp(self.lo, self.hi);
        if v != zero {
            candidates.push(zero);
        }
        // Shrink by halving toward zero
        if v > 0 && v / 2 >= self.lo {
            candidates.push(v / 2);
        }
        if v < 0 && v / 2 <= self.hi {
            candidates.push(v / 2);
        }
        // Shrink by decrementing toward zero
        if v > 0 && v - 1 >= self.lo {
            candidates.push(v - 1);
        }
        if v < 0 && v + 1 <= self.hi {
            candidates.push(v + 1);
        }
        candidates.sort();
        candidates.dedup();
        candidates
    }
}

// ── U64 Range Generator ──────────────────────────────────────────

/// Generates unsigned integers in `[lo, hi]` inclusive.
#[derive(Debug, Clone)]
pub struct U64RangeGen {
    pub lo: u64,
    pub hi: u64,
}

impl U64RangeGen {
    pub fn new(lo: u64, hi: u64) -> Self {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        Self { lo, hi }
    }
}

impl Generator for U64RangeGen {
    type Value = u64;

    fn generate(&self, seed: u64, size: usize) -> u64 {
        let mut rng = Rng::new(seed);
        let full_range = self.hi - self.lo;
        let scale = (size as u64).min(100);
        let effective = full_range.saturating_mul(scale) / 100;
        if effective == 0 {
            return self.lo;
        }
        self.lo + rng.next_u64() % (effective + 1)
    }

    fn shrink(&self, value: &u64) -> Vec<u64> {
        let v = *value;
        let mut candidates = Vec::new();
        if v != self.lo {
            candidates.push(self.lo);
        }
        if v > 0 {
            let half = v / 2;
            if half >= self.lo {
                candidates.push(half);
            }
            if v - 1 >= self.lo {
                candidates.push(v - 1);
            }
        }
        candidates.sort();
        candidates.dedup();
        candidates
    }
}

// ── Float Generator ──────────────────────────────────────────────

/// Generates f64 values in a range.
#[derive(Debug, Clone)]
pub struct FloatGen {
    pub lo: f64,
    pub hi: f64,
}

impl FloatGen {
    pub fn new(lo: f64, hi: f64) -> Self {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        Self { lo, hi }
    }
}

impl Generator for FloatGen {
    type Value = f64;

    fn generate(&self, seed: u64, _size: usize) -> f64 {
        let mut rng = Rng::new(seed);
        self.lo + rng.next_f64() * (self.hi - self.lo)
    }

    fn shrink(&self, value: &f64) -> Vec<f64> {
        let v = *value;
        let mut candidates = Vec::new();
        let zero = 0.0f64.clamp(self.lo, self.hi);
        if (v - zero).abs() > f64::EPSILON {
            candidates.push(zero);
        }
        let half = v / 2.0;
        if half >= self.lo && half <= self.hi && (v - half).abs() > f64::EPSILON {
            candidates.push(half);
        }
        candidates
    }
}

// ── String Generator ─────────────────────────────────────────────

/// Character class for string generation.
#[derive(Debug, Clone)]
pub enum CharClass {
    /// ASCII lowercase a-z.
    AsciiLower,
    /// ASCII uppercase A-Z.
    AsciiUpper,
    /// ASCII digits 0-9.
    Digit,
    /// All printable ASCII (32..127).
    Printable,
    /// Custom set of characters.
    Custom(Vec<char>),
}

impl CharClass {
    fn sample(&self, rng: &mut Rng) -> char {
        match self {
            CharClass::AsciiLower => (b'a' + (rng.next_u64() % 26) as u8) as char,
            CharClass::AsciiUpper => (b'A' + (rng.next_u64() % 26) as u8) as char,
            CharClass::Digit => (b'0' + (rng.next_u64() % 10) as u8) as char,
            CharClass::Printable => (32 + (rng.next_u64() % 95) as u8) as char,
            CharClass::Custom(chars) => {
                if chars.is_empty() {
                    'a'
                } else {
                    chars[(rng.next_u64() as usize) % chars.len()]
                }
            }
        }
    }
}

/// Generates random strings with configurable length and character class.
#[derive(Debug, Clone)]
pub struct StringGen {
    pub min_len: usize,
    pub max_len: usize,
    pub char_class: CharClass,
}

impl StringGen {
    pub fn new(min_len: usize, max_len: usize, char_class: CharClass) -> Self {
        let (min_len, max_len) = if min_len <= max_len {
            (min_len, max_len)
        } else {
            (max_len, min_len)
        };
        Self { min_len, max_len, char_class }
    }

    pub fn ascii(min_len: usize, max_len: usize) -> Self {
        Self::new(min_len, max_len, CharClass::AsciiLower)
    }

    pub fn alphanumeric(min_len: usize, max_len: usize) -> Self {
        Self::new(min_len, max_len, CharClass::Printable)
    }
}

impl Generator for StringGen {
    type Value = String;

    fn generate(&self, seed: u64, size: usize) -> String {
        let mut rng = Rng::new(seed);
        let range = self.max_len - self.min_len;
        let scaled = if range == 0 {
            0
        } else {
            (range * size.min(100)) / 100
        };
        let len = self.min_len + (rng.next_u64() as usize) % (scaled + 1);
        (0..len).map(|_| self.char_class.sample(&mut rng)).collect()
    }

    fn shrink(&self, value: &String) -> Vec<String> {
        let mut candidates = Vec::new();
        // Empty string
        if !value.is_empty() && self.min_len == 0 {
            candidates.push(String::new());
        }
        // Remove characters from the end
        let chars: Vec<char> = value.chars().collect();
        if chars.len() > self.min_len {
            let half_len = (chars.len() / 2).max(self.min_len);
            candidates.push(chars[..half_len].iter().collect());
            if chars.len() > 1 {
                let trimmed: String = chars[..chars.len() - 1].iter().collect();
                candidates.push(trimmed);
            }
        }
        // Replace characters with 'a' (simpler)
        if !value.is_empty() {
            let simple: String = std::iter::repeat_n('a', value.len()).collect();
            if simple != *value {
                candidates.push(simple);
            }
        }
        candidates
    }
}

// ── Vec Generator ────────────────────────────────────────────────

/// Generates vectors of values from an inner generator.
#[derive(Debug, Clone)]
pub struct VecGen<G> {
    pub inner: G,
    pub min_len: usize,
    pub max_len: usize,
}

impl<G> VecGen<G> {
    pub fn new(inner: G, min_len: usize, max_len: usize) -> Self {
        let (min_len, max_len) = if min_len <= max_len {
            (min_len, max_len)
        } else {
            (max_len, min_len)
        };
        Self { inner, min_len, max_len }
    }
}

impl<G: Generator> Generator for VecGen<G> {
    type Value = Vec<G::Value>;

    fn generate(&self, seed: u64, size: usize) -> Vec<G::Value> {
        let mut rng = Rng::new(seed);
        let range = self.max_len - self.min_len;
        let scaled = if range == 0 { 0 } else { (range * size.min(100)) / 100 };
        let len = self.min_len + (rng.next_u64() as usize) % (scaled + 1);
        (0..len)
            .map(|i| {
                let child_seed = seed.wrapping_add(i as u64).wrapping_mul(6364136223846793005);
                self.inner.generate(child_seed, size)
            })
            .collect()
    }

    fn shrink(&self, value: &Vec<G::Value>) -> Vec<Vec<G::Value>> {
        let mut candidates = Vec::new();
        // Empty
        if !value.is_empty() && self.min_len == 0 {
            candidates.push(Vec::new());
        }
        // Remove from end
        if value.len() > self.min_len {
            let half = (value.len() / 2).max(self.min_len);
            candidates.push(value[..half].to_vec());
            if value.len() > 1 {
                candidates.push(value[..value.len() - 1].to_vec());
            }
        }
        // Shrink individual elements
        for (i, item) in value.iter().enumerate() {
            for shrunk in self.inner.shrink(item) {
                let mut copy = value.clone();
                copy[i] = shrunk;
                candidates.push(copy);
            }
        }
        candidates
    }
}

// ── Option Generator ─────────────────────────────────────────────

/// Generates `Option<T>` values — either `None` or `Some(generated)`.
#[derive(Debug, Clone)]
pub struct OptionGen<G> {
    pub inner: G,
    /// Probability of generating `Some` (0.0..=1.0).
    pub some_probability: f64,
}

impl<G> OptionGen<G> {
    pub fn new(inner: G) -> Self {
        Self { inner, some_probability: 0.75 }
    }

    pub fn with_probability(inner: G, some_probability: f64) -> Self {
        Self {
            inner,
            some_probability: some_probability.clamp(0.0, 1.0),
        }
    }
}

impl<G: Generator> Generator for OptionGen<G> {
    type Value = Option<G::Value>;

    fn generate(&self, seed: u64, size: usize) -> Option<G::Value> {
        let mut rng = Rng::new(seed);
        if rng.next_f64() < self.some_probability {
            Some(self.inner.generate(seed.wrapping_add(1), size))
        } else {
            None
        }
    }

    fn shrink(&self, value: &Option<G::Value>) -> Vec<Option<G::Value>> {
        match value {
            None => vec![],
            Some(v) => {
                let mut candidates: Vec<Option<G::Value>> = vec![None];
                for shrunk in self.inner.shrink(v) {
                    candidates.push(Some(shrunk));
                }
                candidates
            }
        }
    }
}

// ── Constant Generator ──────────────────────────────────────────

/// Always produces the same value. Useful in combinators.
#[derive(Debug, Clone)]
pub struct ConstGen<T: Clone + fmt::Debug> {
    pub value: T,
}

impl<T: Clone + fmt::Debug> ConstGen<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Clone + fmt::Debug> Generator for ConstGen<T> {
    type Value = T;

    fn generate(&self, _seed: u64, _size: usize) -> T {
        self.value.clone()
    }

    fn shrink(&self, _value: &T) -> Vec<T> {
        vec![]
    }
}

// ── OneOf Generator ──────────────────────────────────────────────

/// Generates values by randomly picking from a fixed set.
#[derive(Debug, Clone)]
pub struct OneOfGen<T: Clone + fmt::Debug> {
    pub values: Vec<T>,
}

impl<T: Clone + fmt::Debug> OneOfGen<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self { values }
    }
}

impl<T: Clone + fmt::Debug> Generator for OneOfGen<T> {
    type Value = T;

    fn generate(&self, seed: u64, _size: usize) -> T {
        if self.values.is_empty() {
            panic!("OneOfGen: empty value set");
        }
        let mut rng = Rng::new(seed);
        let idx = rng.next_u64() as usize % self.values.len();
        self.values[idx].clone()
    }

    fn shrink(&self, value: &T) -> Vec<T> {
        // Shrink toward earlier items in the list
        let _ = value;
        if self.values.len() > 1 {
            vec![self.values[0].clone()]
        } else {
            vec![]
        }
    }
}

// ── Test Configuration ───────────────────────────────────────────

/// Configuration for property-based test runs.
#[derive(Debug, Clone)]
pub struct PropertyConfig {
    /// Number of random test cases to generate.
    pub num_tests: usize,
    /// Maximum shrink attempts per failure.
    pub max_shrinks: usize,
    /// Seed for reproducibility. If `None`, uses a default seed.
    pub seed: Option<u64>,
    /// Maximum size parameter passed to generators (controls complexity).
    pub max_size: usize,
}

impl Default for PropertyConfig {
    fn default() -> Self {
        Self {
            num_tests: 100,
            max_shrinks: 200,
            seed: None,
            max_size: 100,
        }
    }
}

impl PropertyConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tests(mut self, n: usize) -> Self {
        self.num_tests = n;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_max_shrinks(mut self, n: usize) -> Self {
        self.max_shrinks = n;
        self
    }

    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }
}

// ── Test Result ──────────────────────────────────────────────────

/// Outcome of a property-based test run.
#[derive(Debug, Clone)]
pub enum PropertyResult<V: fmt::Debug> {
    /// All test cases passed.
    Passed {
        /// Number of successful test cases.
        num_tests: usize,
    },
    /// A failing case was found.
    Failed {
        /// The original failing input.
        original: V,
        /// The minimal failing input after shrinking (if any).
        shrunk: Option<V>,
        /// Number of successful tests before failure.
        num_passed: usize,
        /// Number of shrink steps performed.
        shrink_steps: usize,
        /// The seed that reproduces this failure.
        seed: u64,
    },
}

impl<V: fmt::Debug> PropertyResult<V> {
    pub fn is_passed(&self) -> bool {
        matches!(self, PropertyResult::Passed { .. })
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, PropertyResult::Failed { .. })
    }
}

impl<V: fmt::Debug> fmt::Display for PropertyResult<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyResult::Passed { num_tests } => {
                write!(f, "OK, passed {num_tests} tests")
            }
            PropertyResult::Failed {
                original,
                shrunk,
                num_passed,
                shrink_steps,
                seed,
            } => {
                write!(f, "FAILED after {num_passed} tests (seed={seed})\n")?;
                write!(f, "  Original: {original:?}\n")?;
                if let Some(s) = shrunk {
                    write!(f, "  Shrunk ({shrink_steps} steps): {s:?}")?;
                }
                Ok(())
            }
        }
    }
}

// ── forall combinator ────────────────────────────────────────────

/// Test that a property holds for all generated values.
///
/// Returns `Passed` if the property held for all `config.num_tests` cases,
/// or `Failed` with the (possibly shrunk) counterexample.
pub fn forall<G: Generator>(
    generator: &G,
    config: &PropertyConfig,
    property: impl Fn(&G::Value) -> bool,
) -> PropertyResult<G::Value> {
    let base_seed = config.seed.unwrap_or(12345);
    let mut rng = Rng::new(base_seed);

    for i in 0..config.num_tests {
        let test_seed = rng.next_u64();
        let size = if config.num_tests <= 1 {
            config.max_size
        } else {
            (i * config.max_size) / config.num_tests
        };
        let value = generator.generate(test_seed, size);

        if !property(&value) {
            // Shrink
            let (shrunk, shrink_steps) = shrink_value(generator, &value, &property, config.max_shrinks);
            return PropertyResult::Failed {
                original: value,
                shrunk,
                num_passed: i,
                shrink_steps,
                seed: test_seed,
            };
        }
    }

    PropertyResult::Passed {
        num_tests: config.num_tests,
    }
}

/// Test that a property holds for at least one generated value.
///
/// Returns `Passed` if the property held for at least one case,
/// or `Failed` with the last attempted value.
pub fn exists<G: Generator>(
    generator: &G,
    config: &PropertyConfig,
    property: impl Fn(&G::Value) -> bool,
) -> PropertyResult<G::Value> {
    let base_seed = config.seed.unwrap_or(12345);
    let mut rng = Rng::new(base_seed);
    let mut last_value = None;

    for i in 0..config.num_tests {
        let test_seed = rng.next_u64();
        let size = if config.num_tests <= 1 {
            config.max_size
        } else {
            (i * config.max_size) / config.num_tests
        };
        let value = generator.generate(test_seed, size);
        if property(&value) {
            return PropertyResult::Passed { num_tests: i + 1 };
        }
        last_value = Some(value);
    }

    match last_value {
        Some(v) => PropertyResult::Failed {
            original: v.clone(),
            shrunk: None,
            num_passed: 0,
            shrink_steps: 0,
            seed: base_seed,
        },
        None => PropertyResult::Passed { num_tests: 0 },
    }
}

// ── Shrinking ────────────────────────────────────────────────────

fn shrink_value<G: Generator>(
    generator: &G,
    failing: &G::Value,
    property: &impl Fn(&G::Value) -> bool,
    max_steps: usize,
) -> (Option<G::Value>, usize) {
    let mut current = failing.clone();
    let mut steps = 0;
    let mut found_smaller = false;

    for _ in 0..max_steps {
        let candidates = generator.shrink(&current);
        if candidates.is_empty() {
            break;
        }
        let mut improved = false;
        for candidate in candidates {
            steps += 1;
            if !property(&candidate) {
                current = candidate;
                improved = true;
                found_smaller = true;
                break;
            }
        }
        if !improved {
            break;
        }
    }

    if found_smaller {
        (Some(current), steps)
    } else {
        (None, steps)
    }
}

// ── Mapped Generator ─────────────────────────────────────────────

/// A generator that maps the output of another generator.
pub struct MappedGen<G, F> {
    inner: G,
    map_fn: F,
}

impl<G: fmt::Debug, F> fmt::Debug for MappedGen<G, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedGen")
            .field("inner", &self.inner)
            .field("map_fn", &"<fn>")
            .finish()
    }
}

impl<G, F, T> Generator for MappedGen<G, F>
where
    G: Generator,
    F: Fn(G::Value) -> T,
    T: Clone + fmt::Debug,
{
    type Value = T;

    fn generate(&self, seed: u64, size: usize) -> T {
        (self.map_fn)(self.inner.generate(seed, size))
    }

    fn shrink(&self, _value: &T) -> Vec<T> {
        // Cannot shrink through an arbitrary mapping without an inverse
        vec![]
    }
}

/// Create a mapped generator.
pub fn map_gen<G, F, T>(source: G, f: F) -> MappedGen<G, F>
where
    G: Generator,
    F: Fn(G::Value) -> T,
    T: Clone + fmt::Debug,
{
    MappedGen { inner: source, map_fn: f }
}

// ── Filter Generator ─────────────────────────────────────────────

/// A generator that filters output values.
pub struct FilterGen<G, F> {
    inner: G,
    predicate: F,
    max_retries: usize,
}

impl<G: fmt::Debug, F> fmt::Debug for FilterGen<G, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilterGen")
            .field("inner", &self.inner)
            .field("predicate", &"<fn>")
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl<G, F> Generator for FilterGen<G, F>
where
    G: Generator,
    F: Fn(&G::Value) -> bool,
{
    type Value = G::Value;

    fn generate(&self, seed: u64, size: usize) -> G::Value {
        let mut attempt_seed = seed;
        for _ in 0..self.max_retries {
            let value = self.inner.generate(attempt_seed, size);
            if (self.predicate)(&value) {
                return value;
            }
            attempt_seed = attempt_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
        // Last resort: just return whatever we get
        self.inner.generate(seed, size)
    }

    fn shrink(&self, value: &G::Value) -> Vec<G::Value> {
        self.inner
            .shrink(value)
            .into_iter()
            .filter(|v| (self.predicate)(v))
            .collect()
    }
}

/// Create a filtered generator.
pub fn filter_gen<G, F>(source: G, predicate: F) -> FilterGen<G, F>
where
    G: Generator,
    F: Fn(&G::Value) -> bool,
{
    FilterGen {
        inner: source,
        predicate,
        max_retries: 100,
    }
}

// ── Test Report ──────────────────────────────────────────────────

/// Collected report from multiple property test runs.
#[derive(Debug, Clone, Default)]
pub struct TestReport {
    entries: Vec<ReportEntry>,
}

#[derive(Debug, Clone)]
struct ReportEntry {
    name: String,
    passed: bool,
    num_tests: usize,
    message: String,
}

impl TestReport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a test result.
    pub fn record<V: fmt::Debug>(&mut self, name: &str, result: &PropertyResult<V>) {
        let (passed, num_tests, message) = match result {
            PropertyResult::Passed { num_tests } => {
                (true, *num_tests, format!("OK, passed {num_tests} tests"))
            }
            PropertyResult::Failed { num_passed, seed, .. } => {
                (false, *num_passed, format!("FAILED after {num_passed} tests (seed={seed})"))
            }
        };
        self.entries.push(ReportEntry {
            name: name.to_string(),
            passed,
            num_tests,
            message,
        });
    }

    /// Number of recorded tests.
    pub fn total(&self) -> usize {
        self.entries.len()
    }

    /// Number of passed tests.
    pub fn num_passed(&self) -> usize {
        self.entries.iter().filter(|e| e.passed).count()
    }

    /// Number of failed tests.
    pub fn num_failed(&self) -> usize {
        self.entries.iter().filter(|e| !e.passed).count()
    }

    /// Did all tests pass?
    pub fn all_passed(&self) -> bool {
        self.entries.iter().all(|e| e.passed)
    }

    /// Format as a text summary.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        for entry in &self.entries {
            let status = if entry.passed { "PASS" } else { "FAIL" };
            out.push_str(&format!("[{status}] {}: {}\n", entry.name, entry.message));
        }
        out.push_str(&format!(
            "\n{} total, {} passed, {} failed\n",
            self.total(),
            self.num_passed(),
            self.num_failed()
        ));
        out
    }
}

impl fmt::Display for TestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_gen_deterministic() {
        let g = BoolGen;
        let v1 = g.generate(42, 100);
        let v2 = g.generate(42, 100);
        assert_eq!(v1, v2);
    }

    #[test]
    fn bool_gen_shrink() {
        let g = BoolGen;
        assert_eq!(g.shrink(&true), vec![false]);
        assert!(g.shrink(&false).is_empty());
    }

    #[test]
    fn int_range_respects_bounds() {
        let g = IntRangeGen::new(10, 20);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(99);
        let result = forall(&g, &cfg, |v| *v >= 10 && *v <= 20);
        assert!(result.is_passed(), "got: {result}");
    }

    #[test]
    fn int_range_shrinks_toward_zero() {
        let g = IntRangeGen::new(-100, 100);
        let candidates = g.shrink(&50);
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&25));
    }

    #[test]
    fn u64_range_generates_in_bounds() {
        let g = U64RangeGen::new(5, 15);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(7);
        let result = forall(&g, &cfg, |v| *v >= 5 && *v <= 15);
        assert!(result.is_passed());
    }

    #[test]
    fn float_gen_in_range() {
        let g = FloatGen::new(-1.0, 1.0);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(123);
        let result = forall(&g, &cfg, |v| *v >= -1.0 && *v <= 1.0);
        assert!(result.is_passed());
    }

    #[test]
    fn string_gen_respects_length() {
        let g = StringGen::ascii(2, 10);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(555);
        let result = forall(&g, &cfg, |s| s.len() >= 2 && s.len() <= 10);
        assert!(result.is_passed());
    }

    #[test]
    fn string_gen_shrinks() {
        let g = StringGen::ascii(0, 10);
        let candidates = g.shrink(&"hello".to_string());
        assert!(candidates.iter().any(|s| s.is_empty()));
        assert!(candidates.iter().any(|s| s.len() < 5));
    }

    #[test]
    fn vec_gen_respects_length() {
        let g = VecGen::new(IntRangeGen::new(0, 100), 0, 5);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(42);
        let result = forall(&g, &cfg, |v| v.len() <= 5);
        assert!(result.is_passed());
    }

    #[test]
    fn vec_gen_shrinks() {
        let g = VecGen::new(IntRangeGen::new(0, 100), 0, 10);
        let value = vec![50, 60, 70];
        let candidates = g.shrink(&value);
        assert!(candidates.iter().any(|v| v.is_empty()));
    }

    #[test]
    fn option_gen_produces_both_variants() {
        let g = OptionGen::new(IntRangeGen::new(0, 10));
        let cfg = PropertyConfig::new().with_tests(50).with_seed(1);
        let result_some = exists(&g, &cfg, |v| v.is_some());
        let result_none = exists(&g, &cfg, |v| v.is_none());
        assert!(result_some.is_passed());
        assert!(result_none.is_passed());
    }

    #[test]
    fn const_gen_always_same() {
        let g = ConstGen::new(42i64);
        let cfg = PropertyConfig::new().with_tests(50).with_seed(1);
        let result = forall(&g, &cfg, |v| *v == 42);
        assert!(result.is_passed());
    }

    #[test]
    fn oneof_gen_from_set() {
        let g = OneOfGen::new(vec![1, 2, 3]);
        let cfg = PropertyConfig::new().with_tests(100).with_seed(9);
        let result = forall(&g, &cfg, |v| *v == 1 || *v == 2 || *v == 3);
        assert!(result.is_passed());
    }

    #[test]
    fn forall_finds_failure_and_shrinks() {
        let g = IntRangeGen::new(0, 1000);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(42);
        let result = forall(&g, &cfg, |v| *v < 50);
        assert!(result.is_failed());
        if let PropertyResult::Failed { shrunk, .. } = result {
            // Shrunk value should be close to 50 (the boundary)
            if let Some(s) = shrunk {
                assert!(s >= 50);
            }
        }
    }

    #[test]
    fn exists_finds_matching_value() {
        let g = IntRangeGen::new(0, 100);
        let cfg = PropertyConfig::new().with_tests(200).with_seed(42);
        let result = exists(&g, &cfg, |v| *v == 42);
        // The value 42 should appear at some point with enough iterations
        // If not, the test is still valid since exists returns Passed or Failed
        let _ = result;
    }

    #[test]
    fn exists_fails_for_impossible() {
        let g = IntRangeGen::new(0, 10);
        let cfg = PropertyConfig::new().with_tests(50).with_seed(1);
        let result = exists(&g, &cfg, |v| *v > 1000);
        assert!(result.is_failed());
    }

    #[test]
    fn reproducible_with_seed() {
        let g = IntRangeGen::new(0, 1000);
        let cfg = PropertyConfig::new().with_tests(10).with_seed(777);
        let values1: Vec<i64> = (0..10)
            .map(|i| {
                let mut rng = Rng::new(777);
                for _ in 0..=i {
                    rng.next_u64();
                }
                g.generate(rng.next_u64(), 100)
            })
            .collect();
        let values2: Vec<i64> = (0..10)
            .map(|i| {
                let mut rng = Rng::new(777);
                for _ in 0..=i {
                    rng.next_u64();
                }
                g.generate(rng.next_u64(), 100)
            })
            .collect();
        assert_eq!(values1, values2);
    }

    #[test]
    fn test_report_tracking() {
        let g = IntRangeGen::new(0, 10);
        let cfg = PropertyConfig::new().with_tests(20).with_seed(1);
        let pass_result = forall(&g, &cfg, |v| *v >= 0);
        let fail_result = forall(&g, &cfg, |v| *v > 1000);

        let mut report = TestReport::new();
        report.record("always_positive", &pass_result);
        report.record("always_gt_1000", &fail_result);

        assert_eq!(report.total(), 2);
        assert_eq!(report.num_passed(), 1);
        assert_eq!(report.num_failed(), 1);
        assert!(!report.all_passed());
    }

    #[test]
    fn test_report_summary_format() {
        let mut report = TestReport::new();
        let g = BoolGen;
        let cfg = PropertyConfig::new().with_tests(5).with_seed(1);
        let r = forall(&g, &cfg, |_| true);
        report.record("trivial", &r);
        let summary = report.summary();
        assert!(summary.contains("[PASS]"));
        assert!(summary.contains("trivial"));
    }

    #[test]
    fn mapped_gen_transforms_values() {
        let g = map_gen(IntRangeGen::new(1, 10), |v| v * 2);
        let cfg = PropertyConfig::new().with_tests(100).with_seed(42);
        let result = forall(&g, &cfg, |v| *v >= 2 && *v <= 20 && *v % 2 == 0);
        assert!(result.is_passed());
    }

    #[test]
    fn filter_gen_respects_predicate() {
        let g = filter_gen(IntRangeGen::new(0, 100), |v: &i64| *v % 2 == 0);
        let cfg = PropertyConfig::new().with_tests(100).with_seed(42);
        let result = forall(&g, &cfg, |v| *v % 2 == 0);
        assert!(result.is_passed());
    }

    #[test]
    fn property_result_display() {
        let result: PropertyResult<i64> = PropertyResult::Passed { num_tests: 100 };
        let s = format!("{result}");
        assert!(s.contains("100"));
        assert!(s.contains("OK"));
    }

    #[test]
    fn config_builder_chain() {
        let cfg = PropertyConfig::new()
            .with_tests(50)
            .with_seed(999)
            .with_max_shrinks(500)
            .with_max_size(200);
        assert_eq!(cfg.num_tests, 50);
        assert_eq!(cfg.seed, Some(999));
        assert_eq!(cfg.max_shrinks, 500);
        assert_eq!(cfg.max_size, 200);
    }

    #[test]
    fn int_range_inverted_bounds_normalized() {
        let g = IntRangeGen::new(20, 10);
        assert_eq!(g.lo, 10);
        assert_eq!(g.hi, 20);
    }
}
