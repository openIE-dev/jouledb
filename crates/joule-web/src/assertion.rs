//! Rich assertion library — approximate equality, substring checks, JSON comparison,
//! sorted checks, predicate-based assertions, and a chainable `Expect<T>` wrapper.
//!
//! Provides descriptive failure messages for testing. All assertions panic on failure
//! with clear context about expected vs actual values.

use serde_json::Value as JsonValue;

// ── Free-Standing Assertions ────────────────────────────────────

/// Assert two f64 values are approximately equal within epsilon.
///
/// # Panics
/// Panics if `(left - right).abs() > epsilon`.
pub fn assert_approx_eq(left: f64, right: f64, epsilon: f64) {
    let diff = (left - right).abs();
    if diff > epsilon {
        panic!(
            "assertion failed: `assert_approx_eq`\n  left:    {left}\n  right:   {right}\n  diff:    {diff}\n  epsilon: {epsilon}"
        );
    }
}

/// Assert that `haystack` contains `needle` as a substring.
///
/// # Panics
/// Panics if `needle` is not found in `haystack`.
pub fn assert_contains(haystack: &str, needle: &str) {
    if !haystack.contains(needle) {
        panic!(
            "assertion failed: `assert_contains`\n  haystack: {:?}\n  needle:   {:?}",
            haystack, needle
        );
    }
}

/// Assert that `s` starts with `prefix`.
///
/// # Panics
/// Panics if `s` does not start with `prefix`.
pub fn assert_starts_with(s: &str, prefix: &str) {
    if !s.starts_with(prefix) {
        panic!(
            "assertion failed: `assert_starts_with`\n  string: {:?}\n  prefix: {:?}",
            s, prefix
        );
    }
}

/// Assert that `s` ends with `suffix`.
///
/// # Panics
/// Panics if `s` does not end with `suffix`.
pub fn assert_ends_with(s: &str, suffix: &str) {
    if !s.ends_with(suffix) {
        panic!(
            "assertion failed: `assert_ends_with`\n  string: {:?}\n  suffix: {:?}",
            s, suffix
        );
    }
}

/// Assert two JSON values are equal (object key order is irrelevant since
/// serde_json::Value already compares structurally).
///
/// # Panics
/// Panics if the two values differ.
pub fn assert_json_eq(left: &JsonValue, right: &JsonValue) {
    if left != right {
        panic!(
            "assertion failed: `assert_json_eq`\n  left:  {}\n  right: {}",
            serde_json::to_string_pretty(left).unwrap_or_else(|_| format!("{left:?}")),
            serde_json::to_string_pretty(right).unwrap_or_else(|_| format!("{right:?}")),
        );
    }
}

/// Assert that a slice is sorted in non-decreasing order.
///
/// # Panics
/// Panics if the slice is not sorted.
pub fn assert_sorted<T: PartialOrd + std::fmt::Debug>(slice: &[T]) {
    for i in 1..slice.len() {
        if slice[i - 1] > slice[i] {
            panic!(
                "assertion failed: `assert_sorted`\n  not sorted at index {}: {:?} > {:?}",
                i - 1,
                slice[i - 1],
                slice[i]
            );
        }
    }
}

/// Assert that all elements in a slice satisfy a predicate.
///
/// # Panics
/// Panics if any element fails the predicate.
pub fn assert_all<T: std::fmt::Debug, F: Fn(&T) -> bool>(slice: &[T], predicate: F) {
    for (i, item) in slice.iter().enumerate() {
        if !predicate(item) {
            panic!(
                "assertion failed: `assert_all`\n  element at index {i} failed predicate: {item:?}"
            );
        }
    }
}

/// Assert that at least one element in a slice satisfies a predicate.
///
/// # Panics
/// Panics if no element satisfies the predicate.
pub fn assert_any<T: std::fmt::Debug, F: Fn(&T) -> bool>(slice: &[T], predicate: F) {
    if !slice.iter().any(|item| predicate(item)) {
        panic!(
            "assertion failed: `assert_any`\n  no element satisfied the predicate in slice of length {}",
            slice.len()
        );
    }
}

// ── Expect<T> Wrapper ───────────────────────────────────────────

/// A chainable assertion wrapper for fluent test assertions.
///
/// # Example
/// ```ignore
/// expect(42).to_equal(42).to_be_gt(10);
/// expect(vec![1, 2, 3]).to_have_length(3).to_contain(&2);
/// ```
pub struct Expect<T> {
    value: T,
    label: Option<String>,
}

/// Create an `Expect` wrapper for chainable assertions.
pub fn expect<T>(value: T) -> Expect<T> {
    Expect { value, label: None }
}

impl<T> Expect<T> {
    /// Add a descriptive label for failure messages.
    pub fn labeled(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    fn context(&self) -> String {
        match &self.label {
            Some(l) => format!(" ({l})"),
            None => String::new(),
        }
    }

    /// Get a reference to the wrapped value.
    pub fn value(&self) -> &T {
        &self.value
    }
}

impl<T: PartialEq + std::fmt::Debug> Expect<T> {
    /// Assert the value equals `expected`.
    pub fn to_equal(self, expected: T) -> Self {
        if self.value != expected {
            panic!(
                "assertion failed: `to_equal`{}\n  expected: {expected:?}\n  actual:   {:?}",
                self.context(),
                self.value,
            );
        }
        self
    }

    /// Assert the value does not equal `other`.
    pub fn to_not_equal(self, other: T) -> Self {
        if self.value == other {
            panic!(
                "assertion failed: `to_not_equal`{}\n  value should not equal: {other:?}",
                self.context(),
            );
        }
        self
    }
}

impl<T: PartialOrd + std::fmt::Debug> Expect<T> {
    /// Assert the value is greater than `other`.
    pub fn to_be_gt(self, other: T) -> Self {
        if !(self.value > other) {
            panic!(
                "assertion failed: `to_be_gt`{}\n  expected > {other:?}\n  actual:    {:?}",
                self.context(),
                self.value,
            );
        }
        self
    }

    /// Assert the value is greater than or equal to `other`.
    pub fn to_be_gte(self, other: T) -> Self {
        if !(self.value >= other) {
            panic!(
                "assertion failed: `to_be_gte`{}\n  expected >= {other:?}\n  actual:     {:?}",
                self.context(),
                self.value,
            );
        }
        self
    }

    /// Assert the value is less than `other`.
    pub fn to_be_lt(self, other: T) -> Self {
        if !(self.value < other) {
            panic!(
                "assertion failed: `to_be_lt`{}\n  expected < {other:?}\n  actual:    {:?}",
                self.context(),
                self.value,
            );
        }
        self
    }

    /// Assert the value is less than or equal to `other`.
    pub fn to_be_lte(self, other: T) -> Self {
        if !(self.value <= other) {
            panic!(
                "assertion failed: `to_be_lte`{}\n  expected <= {other:?}\n  actual:     {:?}",
                self.context(),
                self.value,
            );
        }
        self
    }
}

impl<T: std::fmt::Debug> Expect<Vec<T>> {
    /// Assert the vec has the given length.
    pub fn to_have_length(self, len: usize) -> Self {
        if self.value.len() != len {
            panic!(
                "assertion failed: `to_have_length`{}\n  expected length: {len}\n  actual length:   {}",
                self.context(),
                self.value.len(),
            );
        }
        self
    }
}

impl<T: PartialEq + std::fmt::Debug> Expect<Vec<T>> {
    /// Assert the vec contains the given element.
    pub fn to_contain(self, item: &T) -> Self {
        if !self.value.contains(item) {
            panic!(
                "assertion failed: `to_contain`{}\n  expected to contain: {item:?}\n  actual: {:?}",
                self.context(),
                self.value,
            );
        }
        self
    }
}

impl Expect<String> {
    /// Assert the string has the given length.
    pub fn to_have_length(self, len: usize) -> Self {
        if self.value.len() != len {
            panic!(
                "assertion failed: `to_have_length`{}\n  expected length: {len}\n  actual length:   {}",
                self.context(),
                self.value.len(),
            );
        }
        self
    }

    /// Assert the string contains the given substring.
    pub fn to_contain(self, needle: &str) -> Self {
        if !self.value.contains(needle) {
            panic!(
                "assertion failed: `to_contain`{}\n  expected to contain: {:?}\n  actual: {:?}",
                self.context(),
                needle,
                self.value,
            );
        }
        self
    }
}

impl Expect<&str> {
    /// Assert the string has the given length.
    pub fn to_have_length(self, len: usize) -> Self {
        if self.value.len() != len {
            panic!(
                "assertion failed: `to_have_length`{}\n  expected length: {len}\n  actual length:   {}",
                self.context(),
                self.value.len(),
            );
        }
        self
    }

    /// Assert the string contains the given substring.
    pub fn to_contain(self, needle: &str) -> Self {
        if !self.value.contains(needle) {
            panic!(
                "assertion failed: `to_contain`{}\n  expected to contain: {:?}\n  actual: {:?}",
                self.context(),
                needle,
                self.value,
            );
        }
        self
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_eq_pass() {
        assert_approx_eq(1.0, 1.0000001, 0.001);
    }

    #[test]
    #[should_panic(expected = "assert_approx_eq")]
    fn approx_eq_fail() {
        assert_approx_eq(1.0, 2.0, 0.001);
    }

    #[test]
    fn contains_pass() {
        assert_contains("hello world", "world");
    }

    #[test]
    #[should_panic(expected = "assert_contains")]
    fn contains_fail() {
        assert_contains("hello world", "xyz");
    }

    #[test]
    fn starts_with_pass() {
        assert_starts_with("hello world", "hello");
    }

    #[test]
    #[should_panic(expected = "assert_starts_with")]
    fn starts_with_fail() {
        assert_starts_with("hello world", "world");
    }

    #[test]
    fn ends_with_pass() {
        assert_ends_with("hello world", "world");
    }

    #[test]
    #[should_panic(expected = "assert_ends_with")]
    fn ends_with_fail() {
        assert_ends_with("hello world", "hello");
    }

    #[test]
    fn json_eq_pass() {
        let a: JsonValue = serde_json::json!({"b": 2, "a": 1});
        let b: JsonValue = serde_json::json!({"a": 1, "b": 2});
        assert_json_eq(&a, &b);
    }

    #[test]
    #[should_panic(expected = "assert_json_eq")]
    fn json_eq_fail() {
        let a: JsonValue = serde_json::json!({"a": 1});
        let b: JsonValue = serde_json::json!({"a": 2});
        assert_json_eq(&a, &b);
    }

    #[test]
    fn sorted_pass() {
        assert_sorted(&[1, 2, 3, 4, 5]);
        assert_sorted(&[1, 1, 2, 3]); // non-decreasing ok
    }

    #[test]
    #[should_panic(expected = "assert_sorted")]
    fn sorted_fail() {
        assert_sorted(&[1, 3, 2, 4]);
    }

    #[test]
    fn all_pass() {
        assert_all(&[2, 4, 6, 8], |x| x % 2 == 0);
    }

    #[test]
    #[should_panic(expected = "assert_all")]
    fn all_fail() {
        assert_all(&[2, 4, 5, 8], |x| x % 2 == 0);
    }

    #[test]
    fn any_pass() {
        assert_any(&[1, 3, 4, 7], |x| x % 2 == 0);
    }

    #[test]
    #[should_panic(expected = "assert_any")]
    fn any_fail() {
        assert_any(&[1, 3, 5, 7], |x| x % 2 == 0);
    }

    #[test]
    fn expect_to_equal() {
        expect(42).to_equal(42);
    }

    #[test]
    #[should_panic(expected = "to_equal")]
    fn expect_to_equal_fail() {
        expect(42).to_equal(99);
    }

    #[test]
    fn expect_chain() {
        expect(42).to_equal(42).to_be_gt(10).to_be_lt(100);
    }

    #[test]
    fn expect_vec_length_and_contain() {
        expect(vec![1, 2, 3]).to_have_length(3).to_contain(&2);
    }

    #[test]
    fn expect_string_contain() {
        expect("hello world".to_string()).to_contain("world");
    }

    #[test]
    fn expect_str_contain() {
        expect("hello world").to_contain("hello");
    }

    #[test]
    fn expect_labeled() {
        expect(10).labeled("age").to_be_gte(0).to_be_lt(200);
    }

    #[test]
    #[should_panic(expected = "to_be_gt")]
    fn expect_gt_fail() {
        expect(5).to_be_gt(10);
    }

    #[test]
    fn expect_not_equal() {
        expect(1).to_not_equal(2);
    }

    #[test]
    fn expect_string_length() {
        expect("abc").to_have_length(3);
    }

    #[test]
    fn sorted_empty() {
        assert_sorted::<i32>(&[]);
    }

    #[test]
    fn all_empty() {
        assert_all::<i32, _>(&[], |_| false); // vacuously true
    }

    #[test]
    fn json_nested_eq() {
        let a: JsonValue = serde_json::json!({
            "users": [{"name": "Alice", "age": 30}],
            "count": 1
        });
        let b: JsonValue = serde_json::json!({
            "count": 1,
            "users": [{"age": 30, "name": "Alice"}]
        });
        assert_json_eq(&a, &b);
    }
}
