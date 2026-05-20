//! Mock object builder for testing.
//!
//! Replaces `mockall`, `mockito`, `sinon`, and similar JS/Rust mock libraries
//! with a pure-Rust mock system. Supports expected call recording, argument
//! matchers (any, eq, predicate), return value configuration, call count
//! verification, call order verification, and mock reset.

use std::collections::HashMap;
use std::fmt;

// ── Argument Matcher ─────────────────────────────────────────────

/// Strategy for matching arguments in mock expectations.
#[derive(Debug, Clone)]
pub enum ArgMatcher {
    /// Match any value.
    Any,
    /// Match by exact string equality.
    Eq(String),
    /// Match by string prefix.
    StartsWith(String),
    /// Match by string suffix.
    EndsWith(String),
    /// Match by contained substring.
    Contains(String),
    /// Match by regex-like pattern (simple glob: * matches any).
    Glob(String),
}

impl ArgMatcher {
    /// Check if a value matches this matcher.
    pub fn matches(&self, value: &str) -> bool {
        match self {
            ArgMatcher::Any => true,
            ArgMatcher::Eq(expected) => value == expected,
            ArgMatcher::StartsWith(prefix) => value.starts_with(prefix),
            ArgMatcher::EndsWith(suffix) => value.ends_with(suffix),
            ArgMatcher::Contains(sub) => value.contains(sub),
            ArgMatcher::Glob(pattern) => glob_match(pattern, value),
        }
    }
}

impl fmt::Display for ArgMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArgMatcher::Any => write!(f, "<any>"),
            ArgMatcher::Eq(v) => write!(f, "eq({v})"),
            ArgMatcher::StartsWith(v) => write!(f, "starts_with({v})"),
            ArgMatcher::EndsWith(v) => write!(f, "ends_with({v})"),
            ArgMatcher::Contains(v) => write!(f, "contains({v})"),
            ArgMatcher::Glob(v) => write!(f, "glob({v})"),
        }
    }
}

/// Simple glob matching where `*` matches zero or more characters.
fn glob_match(pattern: &str, value: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == value;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        match value[pos..].find(part) {
            Some(found) => {
                // First part must be at start
                if i == 0 && found != 0 {
                    return false;
                }
                pos += found + part.len();
            }
            None => return false,
        }
    }
    // If pattern doesn't end with *, remaining value must be consumed
    if !pattern.ends_with('*') {
        return pos == value.len();
    }
    true
}

// ── Call Count Expectation ───────────────────────────────────────

/// Expected call count for a mock method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallCount {
    /// Must be called exactly N times.
    Exactly(usize),
    /// Must be called at least N times.
    AtLeast(usize),
    /// Must be called at most N times.
    AtMost(usize),
    /// Must never be called.
    Never,
    /// Must be called between min and max times (inclusive).
    Between(usize, usize),
    /// No constraint on call count.
    Unconstrained,
}

impl CallCount {
    /// Check if actual count satisfies this constraint.
    pub fn satisfied_by(&self, actual: usize) -> bool {
        match self {
            CallCount::Exactly(n) => actual == *n,
            CallCount::AtLeast(n) => actual >= *n,
            CallCount::AtMost(n) => actual <= *n,
            CallCount::Never => actual == 0,
            CallCount::Between(lo, hi) => actual >= *lo && actual <= *hi,
            CallCount::Unconstrained => true,
        }
    }
}

impl fmt::Display for CallCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallCount::Exactly(n) => write!(f, "exactly {n}"),
            CallCount::AtLeast(n) => write!(f, "at least {n}"),
            CallCount::AtMost(n) => write!(f, "at most {n}"),
            CallCount::Never => write!(f, "never"),
            CallCount::Between(lo, hi) => write!(f, "between {lo} and {hi}"),
            CallCount::Unconstrained => write!(f, "any number of times"),
        }
    }
}

// ── Expectation ──────────────────────────────────────────────────

/// A single expectation for a mock method call.
#[derive(Debug, Clone)]
pub struct Expectation {
    /// Method name.
    pub method: String,
    /// Argument matchers (one per argument position).
    pub arg_matchers: Vec<ArgMatcher>,
    /// Expected call count.
    pub call_count: CallCount,
    /// Return value (as string).
    pub return_value: String,
    /// Ordering constraint: must be called after expectation at this index.
    pub after_index: Option<usize>,
}

impl Expectation {
    /// Create a new expectation for a method.
    pub fn new(method: &str) -> Self {
        Self {
            method: method.to_string(),
            arg_matchers: Vec::new(),
            call_count: CallCount::Unconstrained,
            return_value: String::new(),
            after_index: None,
        }
    }

    /// Add an argument matcher.
    pub fn with_arg(mut self, matcher: ArgMatcher) -> Self {
        self.arg_matchers.push(matcher);
        self
    }

    /// Set expected call count.
    pub fn times(mut self, count: CallCount) -> Self {
        self.call_count = count;
        self
    }

    /// Set return value.
    pub fn returns(mut self, value: &str) -> Self {
        self.return_value = value.to_string();
        self
    }

    /// Set ordering constraint.
    pub fn after(mut self, index: usize) -> Self {
        self.after_index = Some(index);
        self
    }

    /// Check if arguments match this expectation.
    pub fn matches_args(&self, args: &[&str]) -> bool {
        if self.arg_matchers.is_empty() {
            return true; // No matchers = match anything
        }
        if self.arg_matchers.len() != args.len() {
            return false;
        }
        self.arg_matchers
            .iter()
            .zip(args.iter())
            .all(|(matcher, arg)| matcher.matches(arg))
    }
}

// ── Call Record ──────────────────────────────────────────────────

/// A recorded call to a mock method.
#[derive(Debug, Clone)]
pub struct CallRecord {
    pub method: String,
    pub args: Vec<String>,
    pub sequence: usize,
}

impl fmt::Display for CallRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.method, self.args.join(", "))
    }
}

// ── Verification Error ──────────────────────────────────────────

/// Error from mock verification.
#[derive(Debug, Clone)]
pub struct VerificationError {
    pub errors: Vec<String>,
}

impl VerificationError {
    fn new() -> Self {
        Self { errors: Vec::new() }
    }

    fn push(&mut self, msg: String) {
        self.errors.push(msg);
    }

    fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
}

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, err) in self.errors.iter().enumerate() {
            if i > 0 {
                write!(f, "\n")?;
            }
            write!(f, "{err}")?;
        }
        Ok(())
    }
}

impl std::error::Error for VerificationError {}

// ── MockBuilder ──────────────────────────────────────────────────

/// Builder for mock objects with expectation recording and verification.
#[derive(Debug, Clone)]
pub struct MockBuilder {
    name: String,
    expectations: Vec<Expectation>,
    calls: Vec<CallRecord>,
    sequence_counter: usize,
    /// Track which expectation indices have been "used" (matched at least once).
    matched_expectations: HashMap<usize, usize>,
}

impl MockBuilder {
    /// Create a new mock with a descriptive name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            expectations: Vec::new(),
            calls: Vec::new(),
            sequence_counter: 0,
            matched_expectations: HashMap::new(),
        }
    }

    /// Mock name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add an expectation and return its index.
    pub fn expect(&mut self, expectation: Expectation) -> usize {
        let idx = self.expectations.len();
        self.expectations.push(expectation);
        idx
    }

    /// Record a call and return the matching return value, if any.
    pub fn call(&mut self, method: &str, args: &[&str]) -> Option<String> {
        self.sequence_counter += 1;
        self.calls.push(CallRecord {
            method: method.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            sequence: self.sequence_counter,
        });

        // Find matching expectation
        for (i, exp) in self.expectations.iter().enumerate() {
            if exp.method == method && exp.matches_args(args) {
                let count = self.matched_expectations.entry(i).or_insert(0);
                *count += 1;
                return Some(exp.return_value.clone());
            }
        }
        None
    }

    /// Number of calls recorded.
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Number of calls to a specific method.
    pub fn method_call_count(&self, method: &str) -> usize {
        self.calls.iter().filter(|c| c.method == method).count()
    }

    /// All recorded calls.
    pub fn calls(&self) -> &[CallRecord] {
        &self.calls
    }

    /// Calls to a specific method.
    pub fn calls_to(&self, method: &str) -> Vec<&CallRecord> {
        self.calls.iter().filter(|c| c.method == method).collect()
    }

    /// Verify all expectations are satisfied.
    pub fn verify(&self) -> Result<(), VerificationError> {
        let mut errors = VerificationError::new();

        for (i, exp) in self.expectations.iter().enumerate() {
            let actual_count = self.matched_expectations.get(&i).copied().unwrap_or(0);

            if !exp.call_count.satisfied_by(actual_count) {
                errors.push(format!(
                    "Mock '{}': method '{}' expected {} calls but got {actual_count}",
                    self.name, exp.method, exp.call_count
                ));
            }

            // Verify ordering
            if let Some(after_idx) = exp.after_index {
                let my_first_call = self.first_call_sequence_for_expectation(i);
                let their_last_call = self.last_call_sequence_for_expectation(after_idx);

                if let (Some(my_seq), Some(their_seq)) = (my_first_call, their_last_call) {
                    if my_seq <= their_seq {
                        errors.push(format!(
                            "Mock '{}': method '{}' called before required predecessor '{}'",
                            self.name, exp.method, self.expectations[after_idx].method
                        ));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Find the first call sequence number matching expectation at index.
    fn first_call_sequence_for_expectation(&self, exp_idx: usize) -> Option<usize> {
        let exp = &self.expectations[exp_idx];
        self.calls
            .iter()
            .filter(|c| c.method == exp.method)
            .map(|c| c.sequence)
            .min()
    }

    /// Find the last call sequence number matching expectation at index.
    fn last_call_sequence_for_expectation(&self, exp_idx: usize) -> Option<usize> {
        let exp = &self.expectations[exp_idx];
        self.calls
            .iter()
            .filter(|c| c.method == exp.method)
            .map(|c| c.sequence)
            .max()
    }

    /// Check if a method was called with specific arguments.
    pub fn was_called_with(&self, method: &str, args: &[&str]) -> bool {
        self.calls.iter().any(|c| {
            c.method == method
                && c.args.len() == args.len()
                && c.args.iter().zip(args.iter()).all(|(a, b)| a == b)
        })
    }

    /// Check if a method was ever called.
    pub fn was_called(&self, method: &str) -> bool {
        self.calls.iter().any(|c| c.method == method)
    }

    /// Get the arguments from the last call to a method.
    pub fn last_call_args(&self, method: &str) -> Option<Vec<String>> {
        self.calls
            .iter()
            .rev()
            .find(|c| c.method == method)
            .map(|c| c.args.clone())
    }

    /// Reset all recorded calls (keep expectations).
    pub fn reset_calls(&mut self) {
        self.calls.clear();
        self.matched_expectations.clear();
        self.sequence_counter = 0;
    }

    /// Reset everything (calls and expectations).
    pub fn reset(&mut self) {
        self.expectations.clear();
        self.reset_calls();
    }

    /// Number of expectations.
    pub fn num_expectations(&self) -> usize {
        self.expectations.len()
    }

    /// Generate a summary of all calls.
    pub fn call_summary(&self) -> String {
        if self.calls.is_empty() {
            return format!("Mock '{}': no calls recorded", self.name);
        }
        let mut out = format!("Mock '{}':\n", self.name);
        for c in &self.calls {
            out.push_str(&format!("  [{}] {}\n", c.sequence, c));
        }
        out
    }
}

// ── Expectation Builder ──────────────────────────────────────────

/// Fluent builder for setting up expectations on a mock.
pub struct ExpectationBuilder<'a> {
    mock: &'a mut MockBuilder,
    method: String,
    matchers: Vec<ArgMatcher>,
    count: CallCount,
    return_val: String,
    after: Option<usize>,
}

impl<'a> ExpectationBuilder<'a> {
    pub fn new(mock: &'a mut MockBuilder, method: &str) -> Self {
        Self {
            mock,
            method: method.to_string(),
            matchers: Vec::new(),
            count: CallCount::Unconstrained,
            return_val: String::new(),
            after: None,
        }
    }

    pub fn with_arg(mut self, matcher: ArgMatcher) -> Self {
        self.matchers.push(matcher);
        self
    }

    pub fn with_any_arg(mut self) -> Self {
        self.matchers.push(ArgMatcher::Any);
        self
    }

    pub fn with_eq(mut self, value: &str) -> Self {
        self.matchers.push(ArgMatcher::Eq(value.to_string()));
        self
    }

    pub fn times(mut self, count: CallCount) -> Self {
        self.count = count;
        self
    }

    pub fn once(mut self) -> Self {
        self.count = CallCount::Exactly(1);
        self
    }

    pub fn never(mut self) -> Self {
        self.count = CallCount::Never;
        self
    }

    pub fn returns(mut self, value: &str) -> Self {
        self.return_val = value.to_string();
        self
    }

    pub fn after(mut self, idx: usize) -> Self {
        self.after = Some(idx);
        self
    }

    /// Finalize and register the expectation. Returns the expectation index.
    pub fn build(self) -> usize {
        let exp = Expectation {
            method: self.method,
            arg_matchers: self.matchers,
            call_count: self.count,
            return_value: self.return_val,
            after_index: self.after,
        };
        self.mock.expect(exp)
    }
}

// ── Multi-mock Verifier ──────────────────────────────────────────

/// Verify multiple mocks at once.
pub fn verify_all(mocks: &[&MockBuilder]) -> Result<(), VerificationError> {
    let mut combined = VerificationError::new();
    for mock in mocks {
        if let Err(e) = mock.verify() {
            for err in e.errors {
                combined.push(err);
            }
        }
    }
    if combined.is_empty() {
        Ok(())
    } else {
        Err(combined)
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_mock_call() {
        let mut mock = MockBuilder::new("service");
        mock.expect(Expectation::new("get").returns("value"));
        let result = mock.call("get", &[]);
        assert_eq!(result, Some("value".to_string()));
    }

    #[test]
    fn unmatched_call_returns_none() {
        let mut mock = MockBuilder::new("service");
        let result = mock.call("unknown", &[]);
        assert_eq!(result, None);
    }

    #[test]
    fn eq_arg_matcher() {
        let matcher = ArgMatcher::Eq("hello".to_string());
        assert!(matcher.matches("hello"));
        assert!(!matcher.matches("world"));
    }

    #[test]
    fn any_arg_matcher() {
        let matcher = ArgMatcher::Any;
        assert!(matcher.matches("anything"));
        assert!(matcher.matches(""));
    }

    #[test]
    fn contains_arg_matcher() {
        let matcher = ArgMatcher::Contains("ello".to_string());
        assert!(matcher.matches("hello world"));
        assert!(!matcher.matches("hi world"));
    }

    #[test]
    fn starts_with_matcher() {
        let matcher = ArgMatcher::StartsWith("http".to_string());
        assert!(matcher.matches("https://example.com"));
        assert!(!matcher.matches("ftp://example.com"));
    }

    #[test]
    fn glob_matcher() {
        let matcher = ArgMatcher::Glob("*.json".to_string());
        assert!(matcher.matches("file.json"));
        assert!(!matcher.matches("file.xml"));
    }

    #[test]
    fn exact_call_count_verification() {
        let mut mock = MockBuilder::new("db");
        mock.expect(Expectation::new("save").times(CallCount::Exactly(2)));
        mock.call("save", &[]);
        mock.call("save", &[]);
        assert!(mock.verify().is_ok());
    }

    #[test]
    fn exact_count_fails_on_mismatch() {
        let mut mock = MockBuilder::new("db");
        mock.expect(Expectation::new("save").times(CallCount::Exactly(2)));
        mock.call("save", &[]);
        assert!(mock.verify().is_err());
    }

    #[test]
    fn at_least_count() {
        let mut mock = MockBuilder::new("log");
        mock.expect(Expectation::new("write").times(CallCount::AtLeast(1)));
        mock.call("write", &[]);
        mock.call("write", &[]);
        assert!(mock.verify().is_ok());
    }

    #[test]
    fn at_most_count() {
        let mut mock = MockBuilder::new("api");
        mock.expect(Expectation::new("fetch").times(CallCount::AtMost(3)));
        mock.call("fetch", &[]);
        assert!(mock.verify().is_ok());
    }

    #[test]
    fn never_called_verification() {
        let mut mock = MockBuilder::new("cache");
        mock.expect(Expectation::new("evict").times(CallCount::Never));
        assert!(mock.verify().is_ok());

        mock.call("evict", &[]);
        assert!(mock.verify().is_err());
    }

    #[test]
    fn between_count() {
        let count = CallCount::Between(2, 5);
        assert!(!count.satisfied_by(1));
        assert!(count.satisfied_by(2));
        assert!(count.satisfied_by(3));
        assert!(count.satisfied_by(5));
        assert!(!count.satisfied_by(6));
    }

    #[test]
    fn arg_matching_on_call() {
        let mut mock = MockBuilder::new("http");
        mock.expect(
            Expectation::new("get")
                .with_arg(ArgMatcher::Eq("/api/users".to_string()))
                .returns("200 OK"),
        );
        let result = mock.call("get", &["/api/users"]);
        assert_eq!(result, Some("200 OK".to_string()));

        let result2 = mock.call("get", &["/api/other"]);
        assert_eq!(result2, None);
    }

    #[test]
    fn was_called_with_check() {
        let mut mock = MockBuilder::new("service");
        mock.call("process", &["data1", "data2"]);
        assert!(mock.was_called_with("process", &["data1", "data2"]));
        assert!(!mock.was_called_with("process", &["data1", "other"]));
    }

    #[test]
    fn last_call_args() {
        let mut mock = MockBuilder::new("service");
        mock.call("log", &["first"]);
        mock.call("log", &["second"]);
        let args = mock.last_call_args("log").unwrap();
        assert_eq!(args, vec!["second"]);
    }

    #[test]
    fn reset_calls_keeps_expectations() {
        let mut mock = MockBuilder::new("svc");
        mock.expect(Expectation::new("run").times(CallCount::Exactly(1)));
        mock.call("run", &[]);
        mock.reset_calls();
        assert_eq!(mock.call_count(), 0);
        assert_eq!(mock.num_expectations(), 1);
    }

    #[test]
    fn full_reset() {
        let mut mock = MockBuilder::new("svc");
        mock.expect(Expectation::new("run"));
        mock.call("run", &[]);
        mock.reset();
        assert_eq!(mock.call_count(), 0);
        assert_eq!(mock.num_expectations(), 0);
    }

    #[test]
    fn call_summary_format() {
        let mut mock = MockBuilder::new("api");
        mock.call("get", &["/users"]);
        mock.call("post", &["/data"]);
        let summary = mock.call_summary();
        assert!(summary.contains("api"));
        assert!(summary.contains("get"));
        assert!(summary.contains("post"));
    }

    #[test]
    fn expectation_builder_fluent() {
        let mut mock = MockBuilder::new("svc");
        let idx = ExpectationBuilder::new(&mut mock, "fetch")
            .with_eq("/api")
            .once()
            .returns("ok")
            .build();
        assert_eq!(idx, 0);
        let result = mock.call("fetch", &["/api"]);
        assert_eq!(result, Some("ok".to_string()));
        assert!(mock.verify().is_ok());
    }

    #[test]
    fn verify_all_mocks() {
        let mut m1 = MockBuilder::new("a");
        m1.expect(Expectation::new("run").times(CallCount::Exactly(1)));
        m1.call("run", &[]);

        let mut m2 = MockBuilder::new("b");
        m2.expect(Expectation::new("go").times(CallCount::Exactly(1)));
        m2.call("go", &[]);

        assert!(verify_all(&[&m1, &m2]).is_ok());
    }

    #[test]
    fn verify_all_reports_combined_errors() {
        let m1 = MockBuilder::new("a");
        let m2 = MockBuilder::new("b");
        // Neither mock has calls, but both have expectations
        let mut m1 = m1;
        let mut m2 = m2;
        m1.expect(Expectation::new("run").times(CallCount::Exactly(1)));
        m2.expect(Expectation::new("go").times(CallCount::Exactly(1)));

        let err = verify_all(&[&m1, &m2]).unwrap_err();
        assert_eq!(err.errors.len(), 2);
    }

    #[test]
    fn call_count_display() {
        assert_eq!(format!("{}", CallCount::Exactly(3)), "exactly 3");
        assert_eq!(format!("{}", CallCount::Never), "never");
        assert_eq!(format!("{}", CallCount::AtLeast(1)), "at least 1");
    }

    #[test]
    fn method_call_count_tracking() {
        let mut mock = MockBuilder::new("svc");
        mock.call("a", &[]);
        mock.call("a", &[]);
        mock.call("b", &[]);
        assert_eq!(mock.method_call_count("a"), 2);
        assert_eq!(mock.method_call_count("b"), 1);
        assert_eq!(mock.method_call_count("c"), 0);
    }

    #[test]
    fn matcher_display() {
        assert_eq!(format!("{}", ArgMatcher::Any), "<any>");
        assert_eq!(format!("{}", ArgMatcher::Eq("x".into())), "eq(x)");
    }
}
