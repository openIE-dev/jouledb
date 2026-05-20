//! Extended Result/Option utilities — and_then chains, tap/inspect,
//! collect_results, partition_results, combine errors, Result<Vec> / Vec<Result>
//! transposition, timeout result, retry result, and fallback chains.
//!
//! Replaces lodash/fp-ts/neverthrow result utilities with pure-Rust
//! combinators that compose cleanly.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// A collection of errors gathered from multiple results.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiError<E> {
    errors: Vec<E>,
}

impl<E: fmt::Debug> fmt::Display for MultiError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} errors: {:?}", self.errors.len(), self.errors)
    }
}

impl<E: fmt::Debug> std::error::Error for MultiError<E> {}

impl<E> MultiError<E> {
    /// Create a new multi-error from a vec.
    pub fn new(errors: Vec<E>) -> Self {
        Self { errors }
    }

    /// Number of errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Borrow the error list.
    pub fn errors(&self) -> &[E] {
        &self.errors
    }

    /// Consume into a vec of errors.
    pub fn into_errors(self) -> Vec<E> {
        self.errors
    }

    /// Get the first error.
    pub fn first(&self) -> Option<&E> {
        self.errors.first()
    }
}

// ── ResultExt trait ─────────────────────────────────────────────

/// Extension methods on `Result<T, E>`.
pub trait ResultExt<T, E> {
    /// Inspect the Ok value without consuming.
    fn tap_ok(self, f: impl FnOnce(&T)) -> Result<T, E>;

    /// Inspect the Err value without consuming.
    fn tap_err(self, f: impl FnOnce(&E)) -> Result<T, E>;

    /// Provide a fallback value on error.
    fn or_default(self) -> T where T: Default;

    /// Convert the error using Display.
    fn map_err_to_string(self) -> Result<T, String> where E: fmt::Display;

    /// Combine with another result: if both Ok, return pair; if either Err, accumulate.
    fn and_combine<U>(self, other: Result<U, E>) -> Result<(T, U), Vec<E>>;

    /// Convert Ok to Some(Ok), Err to Some(Err), for chaining.
    fn into_option(self) -> Option<Result<T, E>>;

    /// Flat map the Ok value, like and_then but with a different name for clarity.
    fn flat_map<U>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<U, E>;
}

impl<T, E> ResultExt<T, E> for Result<T, E> {
    fn tap_ok(self, f: impl FnOnce(&T)) -> Result<T, E> {
        if let Ok(v) = &self {
            f(v);
        }
        self
    }

    fn tap_err(self, f: impl FnOnce(&E)) -> Result<T, E> {
        if let Err(e) = &self {
            f(e);
        }
        self
    }

    fn or_default(self) -> T where T: Default {
        self.unwrap_or_default()
    }

    fn map_err_to_string(self) -> Result<T, String> where E: fmt::Display {
        self.map_err(|e| e.to_string())
    }

    fn and_combine<U>(self, other: Result<U, E>) -> Result<(T, U), Vec<E>> {
        match (self, other) {
            (Ok(a), Ok(b)) => Ok((a, b)),
            (Err(e1), Err(e2)) => Err(vec![e1, e2]),
            (Err(e), _) => Err(vec![e]),
            (_, Err(e)) => Err(vec![e]),
        }
    }

    fn into_option(self) -> Option<Result<T, E>> {
        Some(self)
    }

    fn flat_map<U>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<U, E> {
        self.and_then(f)
    }
}

// ── OptionExt trait ─────────────────────────────────────────────

/// Extension methods on `Option<T>`.
pub trait OptionExt<T> {
    /// Inspect the Some value without consuming.
    fn tap_some(self, f: impl FnOnce(&T)) -> Option<T>;

    /// Convert to Result with a custom error.
    fn ok_or_else_with<E>(self, f: impl FnOnce() -> E) -> Result<T, E>;

    /// Combine with another option: if both Some, return pair.
    fn and_combine<U>(self, other: Option<U>) -> Option<(T, U)>;

    /// If None, try the fallback.
    fn or_try(self, f: impl FnOnce() -> Option<T>) -> Option<T>;

    /// Flat map alias.
    fn flat_map<U>(self, f: impl FnOnce(T) -> Option<U>) -> Option<U>;

    /// Convert Some to Ok(()), None to Err.
    fn to_result<E>(self, err: E) -> Result<T, E>;
}

impl<T> OptionExt<T> for Option<T> {
    fn tap_some(self, f: impl FnOnce(&T)) -> Option<T> {
        if let Some(v) = &self {
            f(v);
        }
        self
    }

    fn ok_or_else_with<E>(self, f: impl FnOnce() -> E) -> Result<T, E> {
        self.ok_or_else(f)
    }

    fn and_combine<U>(self, other: Option<U>) -> Option<(T, U)> {
        match (self, other) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    fn or_try(self, f: impl FnOnce() -> Option<T>) -> Option<T> {
        self.or_else(f)
    }

    fn flat_map<U>(self, f: impl FnOnce(T) -> Option<U>) -> Option<U> {
        self.and_then(f)
    }

    fn to_result<E>(self, err: E) -> Result<T, E> {
        self.ok_or(err)
    }
}

// ── Collection operations ───────────────────────────────────────

/// Collect an iterator of Results into Result<Vec<T>, Vec<E>>.
/// Unlike the standard collect which stops at the first error,
/// this collects ALL errors.
pub fn collect_results<T, E>(iter: impl IntoIterator<Item = Result<T, E>>) -> Result<Vec<T>, Vec<E>> {
    let mut oks = Vec::new();
    let mut errs = Vec::new();
    for item in iter {
        match item {
            Ok(v) => oks.push(v),
            Err(e) => errs.push(e),
        }
    }
    if errs.is_empty() {
        Ok(oks)
    } else {
        Err(errs)
    }
}

/// Partition an iterator of Results into (Vec<T>, Vec<E>).
pub fn partition_results<T, E>(iter: impl IntoIterator<Item = Result<T, E>>) -> (Vec<T>, Vec<E>) {
    let mut oks = Vec::new();
    let mut errs = Vec::new();
    for item in iter {
        match item {
            Ok(v) => oks.push(v),
            Err(e) => errs.push(e),
        }
    }
    (oks, errs)
}

/// Transpose a `Result<Vec<T>, E>` to `Vec<Result<T, E>>`.
/// Ok(vec) becomes vec of Ok, Err(e) becomes single-element vec of Err.
pub fn result_vec_to_vec_result<T, E: Clone>(result: Result<Vec<T>, E>) -> Vec<Result<T, E>> {
    match result {
        Ok(vec) => vec.into_iter().map(Ok).collect(),
        Err(e) => vec![Err(e)],
    }
}

/// Transpose a `Vec<Result<T, E>>` to `Result<Vec<T>, Vec<E>>`.
/// Collects all errors instead of failing fast.
pub fn vec_result_to_result_vec<T, E>(results: Vec<Result<T, E>>) -> Result<Vec<T>, Vec<E>> {
    collect_results(results)
}

/// Combine multiple errors into a single MultiError.
pub fn combine_errors<E>(errors: Vec<E>) -> Option<MultiError<E>> {
    if errors.is_empty() {
        None
    } else {
        Some(MultiError::new(errors))
    }
}

// ── Fallback chain ──────────────────────────────────────────────

/// Try a sequence of fallible computations, returning the first Ok.
/// If all fail, returns all errors.
pub fn fallback_chain<T, E>(attempts: Vec<Box<dyn FnOnce() -> Result<T, E>>>) -> Result<T, Vec<E>> {
    let mut errors = Vec::new();
    for attempt in attempts {
        match attempt() {
            Ok(v) => return Ok(v),
            Err(e) => errors.push(e),
        }
    }
    Err(errors)
}

/// Try a sequence of option-returning computations, returning the first Some.
pub fn first_some<T>(attempts: &[&dyn Fn() -> Option<T>]) -> Option<T> {
    for attempt in attempts {
        if let Some(v) = attempt() {
            return Some(v);
        }
    }
    None
}

// ── Retry ───────────────────────────────────────────────────────

/// Retry a fallible operation up to `max_retries` times.
/// Returns the first Ok, or the last Err.
pub fn retry<T, E>(max_retries: usize, mut f: impl FnMut(usize) -> Result<T, E>) -> Result<T, E> {
    let mut last_err = None;
    for attempt in 0..=max_retries {
        match f(attempt) {
            Ok(v) => return Ok(v),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap())
}

/// Retry, collecting all errors.
pub fn retry_collect<T, E>(
    max_retries: usize,
    mut f: impl FnMut(usize) -> Result<T, E>,
) -> Result<T, Vec<E>> {
    let mut errors = Vec::new();
    for attempt in 0..=max_retries {
        match f(attempt) {
            Ok(v) => return Ok(v),
            Err(e) => errors.push(e),
        }
    }
    Err(errors)
}

// ── Chain combinator ────────────────────────────────────────────

/// Chain a sequence of transformations on a Result, short-circuiting on error.
pub fn chain<T, E>(
    initial: Result<T, E>,
    steps: Vec<Box<dyn FnOnce(T) -> Result<T, E>>>,
) -> Result<T, E> {
    let mut current = initial;
    for step in steps {
        current = current.and_then(step);
    }
    current
}

/// Apply a sequence of validations to a value, collecting all errors.
pub fn validate_all<T, E>(value: &T, validators: &[fn(&T) -> Result<(), E>]) -> Result<(), Vec<E>> {
    let mut errors = Vec::new();
    for validator in validators {
        if let Err(e) = validator(value) {
            errors.push(e);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Map over a slice, collecting all Ok values or all Err values.
pub fn try_map<T, U, E>(items: &[T], f: impl Fn(&T) -> Result<U, E>) -> Result<Vec<U>, Vec<E>> {
    let results: Vec<Result<U, E>> = items.iter().map(f).collect();
    collect_results(results)
}

/// Filter a slice using a fallible predicate, collecting errors.
pub fn try_filter<T, E>(items: &[T], predicate: impl Fn(&T) -> Result<bool, E>) -> Result<Vec<&T>, Vec<E>> {
    let mut oks = Vec::new();
    let mut errs = Vec::new();
    for item in items {
        match predicate(item) {
            Ok(true) => oks.push(item),
            Ok(false) => {}
            Err(e) => errs.push(e),
        }
    }
    if errs.is_empty() {
        Ok(oks)
    } else {
        Err(errs)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tap_ok() {
        let mut seen = false;
        let r: Result<i32, &str> = Ok(42);
        let r = r.tap_ok(|v| {
            assert_eq!(*v, 42);
            seen = true;
        });
        assert!(seen);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn test_tap_err() {
        let mut seen = false;
        let r: Result<i32, &str> = Err("bad");
        let r = r.tap_err(|e| {
            assert_eq!(*e, "bad");
            seen = true;
        });
        assert!(seen);
        assert!(r.is_err());
    }

    #[test]
    fn test_or_default() {
        let r: Result<i32, &str> = Err("bad");
        assert_eq!(r.or_default(), 0);
        let r: Result<i32, &str> = Ok(42);
        assert_eq!(r.or_default(), 42);
    }

    #[test]
    fn test_map_err_to_string() {
        let r: Result<i32, i32> = Err(42);
        let r = r.map_err_to_string();
        assert_eq!(r.unwrap_err(), "42");
    }

    #[test]
    fn test_and_combine_both_ok() {
        let a: Result<i32, &str> = Ok(1);
        let b: Result<i32, &str> = Ok(2);
        assert_eq!(a.and_combine(b), Ok((1, 2)));
    }

    #[test]
    fn test_and_combine_both_err() {
        let a: Result<i32, &str> = Err("a");
        let b: Result<i32, &str> = Err("b");
        assert_eq!(a.and_combine(b), Err(vec!["a", "b"]));
    }

    #[test]
    fn test_and_combine_one_err() {
        let a: Result<i32, &str> = Ok(1);
        let b: Result<i32, &str> = Err("b");
        assert_eq!(a.and_combine(b), Err(vec!["b"]));
    }

    #[test]
    fn test_flat_map() {
        let r: Result<i32, &str> = Ok(5);
        let r2 = r.flat_map(|v| if v > 0 { Ok(v * 2) } else { Err("neg") });
        assert_eq!(r2, Ok(10));
    }

    #[test]
    fn test_option_tap_some() {
        let mut seen = false;
        let o = Some(42);
        let o = o.tap_some(|v| {
            assert_eq!(*v, 42);
            seen = true;
        });
        assert!(seen);
        assert_eq!(o, Some(42));
    }

    #[test]
    fn test_option_and_combine() {
        assert_eq!(Some(1).and_combine(Some(2)), Some((1, 2)));
        assert_eq!(Some(1).and_combine(None::<i32>), None);
    }

    #[test]
    fn test_option_or_try() {
        assert_eq!(None::<i32>.or_try(|| Some(42)), Some(42));
        assert_eq!(Some(1).or_try(|| Some(42)), Some(1));
    }

    #[test]
    fn test_option_to_result() {
        assert_eq!(Some(1).to_result("err"), Ok(1));
        assert_eq!(None::<i32>.to_result("err"), Err("err"));
    }

    #[test]
    fn test_collect_results_all_ok() {
        let items: Vec<Result<i32, &str>> = vec![Ok(1), Ok(2), Ok(3)];
        assert_eq!(collect_results(items), Ok(vec![1, 2, 3]));
    }

    #[test]
    fn test_collect_results_some_err() {
        let items: Vec<Result<i32, &str>> = vec![Ok(1), Err("a"), Ok(3), Err("b")];
        assert_eq!(collect_results(items), Err(vec!["a", "b"]));
    }

    #[test]
    fn test_partition_results() {
        let items: Vec<Result<i32, &str>> = vec![Ok(1), Err("a"), Ok(3)];
        let (oks, errs) = partition_results(items);
        assert_eq!(oks, vec![1, 3]);
        assert_eq!(errs, vec!["a"]);
    }

    #[test]
    fn test_result_vec_to_vec_result_ok() {
        let r: Result<Vec<i32>, &str> = Ok(vec![1, 2]);
        let v = result_vec_to_vec_result(r);
        assert_eq!(v, vec![Ok(1), Ok(2)]);
    }

    #[test]
    fn test_result_vec_to_vec_result_err() {
        let r: Result<Vec<i32>, &str> = Err("bad");
        let v = result_vec_to_vec_result(r);
        assert_eq!(v, vec![Err("bad")]);
    }

    #[test]
    fn test_vec_result_to_result_vec() {
        let v: Vec<Result<i32, &str>> = vec![Ok(1), Ok(2)];
        assert_eq!(vec_result_to_result_vec(v), Ok(vec![1, 2]));
    }

    #[test]
    fn test_combine_errors() {
        let errs: Vec<&str> = vec!["a", "b"];
        let multi = combine_errors(errs).unwrap();
        assert_eq!(multi.len(), 2);
        assert_eq!(multi.first(), Some(&"a"));
    }

    #[test]
    fn test_combine_errors_empty() {
        let errs: Vec<&str> = vec![];
        assert!(combine_errors(errs).is_none());
    }

    #[test]
    fn test_retry_success_first_try() {
        let result = retry(3, |_attempt| Ok::<_, &str>(42));
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_retry_success_after_failure() {
        let result = retry(3, |attempt| {
            if attempt < 2 { Err("not yet") } else { Ok(42) }
        });
        assert_eq!(result, Ok(42));
    }

    #[test]
    fn test_retry_all_fail() {
        let result = retry(2, |_attempt| Err::<i32, _>("fail"));
        assert_eq!(result, Err("fail"));
    }

    #[test]
    fn test_retry_collect_all_errors() {
        let result = retry_collect(2, |attempt| Err::<i32, _>(format!("fail-{attempt}")));
        assert_eq!(result, Err(vec!["fail-0".to_string(), "fail-1".to_string(), "fail-2".to_string()]));
    }

    #[test]
    fn test_fallback_chain_first_succeeds() {
        let attempts: Vec<Box<dyn FnOnce() -> Result<i32, String>>> =
            vec![Box::new(|| Ok(1)), Box::new(|| Ok(2))];
        assert_eq!(fallback_chain(attempts), Ok(1));
    }

    #[test]
    fn test_fallback_chain_second_succeeds() {
        let attempts: Vec<Box<dyn FnOnce() -> Result<i32, String>>> =
            vec![Box::new(|| Err("a".into())), Box::new(|| Ok(2))];
        assert_eq!(fallback_chain(attempts), Ok(2));
    }

    #[test]
    fn test_fallback_chain_all_fail() {
        let attempts: Vec<Box<dyn FnOnce() -> Result<i32, String>>> =
            vec![Box::new(|| Err("a".into())), Box::new(|| Err("b".into()))];
        let errs = fallback_chain(attempts).unwrap_err();
        assert_eq!(errs.len(), 2);
        assert_eq!(errs[0], "a");
        assert_eq!(errs[1], "b");
    }

    #[test]
    fn test_first_some() {
        let a: &dyn Fn() -> Option<i32> = &|| None;
        let b: &dyn Fn() -> Option<i32> = &|| Some(42);
        let c: &dyn Fn() -> Option<i32> = &|| Some(99);
        assert_eq!(first_some(&[a, b, c]), Some(42));
    }

    #[test]
    fn test_chain() {
        let steps: Vec<Box<dyn FnOnce(i32) -> Result<i32, String>>> = vec![
            Box::new(|x| Ok(x + 1)),
            Box::new(|x| Ok(x * 2)),
        ];
        assert_eq!(chain(Ok(5), steps), Ok(12));
    }

    #[test]
    fn test_chain_short_circuit() {
        let steps: Vec<Box<dyn FnOnce(i32) -> Result<i32, String>>> = vec![
            Box::new(|_| Err("stop".into())),
            Box::new(|x| Ok(x * 2)),
        ];
        assert_eq!(chain(Ok(5), steps), Err("stop".to_string()));
    }

    #[test]
    fn test_validate_all_pass() {
        let v = 5i32;
        fn pos(x: &i32) -> Result<(), String> {
            if *x > 0 { Ok(()) } else { Err("positive".into()) }
        }
        fn small(x: &i32) -> Result<(), String> {
            if *x < 10 { Ok(()) } else { Err("small".into()) }
        }
        let validators: Vec<fn(&i32) -> Result<(), String>> = vec![pos, small];
        assert!(validate_all(&v, &validators).is_ok());
    }

    #[test]
    fn test_validate_all_fail() {
        let v = -5i32;
        fn pos(x: &i32) -> Result<(), String> {
            if *x > 0 { Ok(()) } else { Err("positive".into()) }
        }
        fn small(x: &i32) -> Result<(), String> {
            if *x < 10 { Ok(()) } else { Err("small".into()) }
        }
        let validators: Vec<fn(&i32) -> Result<(), String>> = vec![pos, small];
        let errs = validate_all(&v, &validators).unwrap_err();
        assert_eq!(errs, vec!["positive".to_string()]);
    }

    #[test]
    fn test_try_map() {
        let items = vec![1, 2, 3];
        let result = try_map(&items, |x| Ok::<_, &str>(x * 2));
        assert_eq!(result, Ok(vec![2, 4, 6]));
    }

    #[test]
    fn test_try_map_with_errors() {
        let items = vec![1, -2, 3, -4];
        let result = try_map(&items, |x| {
            if *x > 0 { Ok(*x) } else { Err(format!("negative: {x}")) }
        });
        assert_eq!(result.unwrap_err().len(), 2);
    }

    #[test]
    fn test_try_filter() {
        let items = vec![1, 2, 3, 4, 5];
        let result = try_filter(&items, |x| Ok::<_, &str>(*x % 2 == 0));
        assert_eq!(result.unwrap(), vec![&2, &4]);
    }

    #[test]
    fn test_multi_error_display() {
        let me = MultiError::new(vec!["a", "b"]);
        let s = me.to_string();
        assert!(s.contains("2 errors"));
    }

    #[test]
    fn test_multi_error_into_errors() {
        let me = MultiError::new(vec![1, 2, 3]);
        assert_eq!(me.into_errors(), vec![1, 2, 3]);
    }

    #[test]
    fn test_into_option() {
        let r: Result<i32, &str> = Ok(42);
        assert_eq!(r.into_option(), Some(Ok(42)));
    }
}
