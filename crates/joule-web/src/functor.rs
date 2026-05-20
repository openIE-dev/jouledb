//! Functor and Applicative patterns for Rust.
//!
//! Provides `Functor` trait (`fmap`), `Applicative` trait (`pure`/`ap`),
//! generic `map_over` for containers, `lift`/`lift2`, `zip_apply`,
//! `traverse`, `sequence`, and `Validation<E,T>` (applicative validation
//! that collects all errors).

use std::fmt;

// ── Wrapper type ────────────────────────────────────────────────────────────

/// A simple wrapper that implements Functor and Applicative.
/// Used as the basis for generic functional composition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wrapper<T> {
    Value(T),
    Empty,
}

impl<T> Wrapper<T> {
    /// Wrap a value.
    pub fn value(v: T) -> Self {
        Wrapper::Value(v)
    }

    /// The empty wrapper.
    pub fn empty() -> Self {
        Wrapper::Empty
    }

    /// Is this `Value`?
    pub fn is_value(&self) -> bool {
        matches!(self, Wrapper::Value(_))
    }

    /// Extract the value or a default.
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Wrapper::Value(v) => v,
            Wrapper::Empty => default,
        }
    }

    /// Functor: map a function over the contained value.
    pub fn fmap<U>(self, f: impl FnOnce(T) -> U) -> Wrapper<U> {
        match self {
            Wrapper::Value(v) => Wrapper::Value(f(v)),
            Wrapper::Empty => Wrapper::Empty,
        }
    }

    /// Applicative: pure / return.
    pub fn pure(v: T) -> Self {
        Wrapper::Value(v)
    }

    /// Convert to Option.
    pub fn to_option(self) -> Option<T> {
        match self {
            Wrapper::Value(v) => Some(v),
            Wrapper::Empty => None,
        }
    }
}

/// Apply a wrapped function to a wrapped argument.
pub fn apply<A, B>(
    wf: Wrapper<Box<dyn FnOnce(A) -> B>>,
    wa: Wrapper<A>,
) -> Wrapper<B> {
    match (wf, wa) {
        (Wrapper::Value(f), Wrapper::Value(a)) => Wrapper::Value(f(a)),
        _ => Wrapper::Empty,
    }
}

// ── Lift ────────────────────────────────────────────────────────────────────

/// Lift a unary function into the Wrapper context.
pub fn lift<A, B>(f: impl Fn(A) -> B + 'static) -> Box<dyn Fn(Wrapper<A>) -> Wrapper<B>> {
    Box::new(move |wa| wa.fmap(|a| f(a)))
}

/// Lift a binary function into the Wrapper context.
pub fn lift2<A, B, C>(
    f: impl Fn(A, B) -> C + Clone + 'static,
) -> Box<dyn Fn(Wrapper<A>, Wrapper<B>) -> Wrapper<C>> {
    Box::new(move |wa, wb| match (wa, wb) {
        (Wrapper::Value(a), Wrapper::Value(b)) => Wrapper::Value(f(a, b)),
        _ => Wrapper::Empty,
    })
}

/// Lift a ternary function into the Wrapper context.
pub fn lift3<A, B, C, D>(
    f: impl Fn(A, B, C) -> D + Clone + 'static,
) -> Box<dyn Fn(Wrapper<A>, Wrapper<B>, Wrapper<C>) -> Wrapper<D>> {
    Box::new(move |wa, wb, wc| match (wa, wb, wc) {
        (Wrapper::Value(a), Wrapper::Value(b), Wrapper::Value(c)) => Wrapper::Value(f(a, b, c)),
        _ => Wrapper::Empty,
    })
}

// ── Generic map over containers ─────────────────────────────────────────────

/// Map a function over a `Vec`.
pub fn map_vec<A, B>(items: Vec<A>, f: impl Fn(A) -> B) -> Vec<B> {
    items.into_iter().map(f).collect()
}

/// Map a function over an `Option`.
pub fn map_option<A, B>(opt: Option<A>, f: impl FnOnce(A) -> B) -> Option<B> {
    opt.map(f)
}

/// Map a function over a `Result`.
pub fn map_result<A, B, E>(res: Result<A, E>, f: impl FnOnce(A) -> B) -> Result<B, E> {
    res.map(f)
}

// ── Zip-apply ───────────────────────────────────────────────────────────────

/// Zip-apply: apply each function in `fns` to the corresponding element in `args`.
pub fn zip_apply<A: Clone, B>(fns: Vec<Box<dyn Fn(&A) -> B>>, args: &[A]) -> Vec<B> {
    fns.iter()
        .zip(args.iter())
        .map(|(f, a)| f(a))
        .collect()
}

/// Zip-apply for Options: if both are Some, apply `f`.
pub fn zip_apply_option<A, B, C>(
    a: Option<A>,
    b: Option<B>,
    f: impl FnOnce(A, B) -> C,
) -> Option<C> {
    match (a, b) {
        (Some(x), Some(y)) => Some(f(x, y)),
        _ => None,
    }
}

/// Zip-apply for two Wrappers.
pub fn zip_apply_wrapper<A, B, C>(
    wa: Wrapper<A>,
    wb: Wrapper<B>,
    f: impl FnOnce(A, B) -> C,
) -> Wrapper<C> {
    match (wa, wb) {
        (Wrapper::Value(a), Wrapper::Value(b)) => Wrapper::Value(f(a, b)),
        _ => Wrapper::Empty,
    }
}

// ── Traverse & Sequence ─────────────────────────────────────────────────────

/// Traverse: map each element with `f` (which returns an Option), then sequence.
/// Returns `None` if any element maps to `None`.
pub fn traverse_option<A, B>(
    items: Vec<A>,
    f: impl Fn(A) -> Option<B>,
) -> Option<Vec<B>> {
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        match f(item) {
            Some(b) => result.push(b),
            None => return None,
        }
    }
    Some(result)
}

/// Sequence: convert a `Vec<Option<T>>` into `Option<Vec<T>>`.
pub fn sequence_option<T>(items: Vec<Option<T>>) -> Option<Vec<T>> {
    traverse_option(items, |x| x)
}

/// Traverse with Result: map each element, short-circuiting on first Err.
pub fn traverse_result<A, B, E>(
    items: Vec<A>,
    f: impl Fn(A) -> Result<B, E>,
) -> Result<Vec<B>, E> {
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        result.push(f(item)?);
    }
    Ok(result)
}

/// Sequence for Results.
pub fn sequence_result<T, E>(items: Vec<Result<T, E>>) -> Result<Vec<T>, E> {
    traverse_result(items, |x| x)
}

/// Traverse with Wrapper: map each element, returning Empty if any is Empty.
pub fn traverse_wrapper<A, B>(
    items: Vec<A>,
    f: impl Fn(A) -> Wrapper<B>,
) -> Wrapper<Vec<B>> {
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        match f(item) {
            Wrapper::Value(b) => result.push(b),
            Wrapper::Empty => return Wrapper::Empty,
        }
    }
    Wrapper::Value(result)
}

/// Sequence for Wrappers.
pub fn sequence_wrapper<T>(items: Vec<Wrapper<T>>) -> Wrapper<Vec<T>> {
    traverse_wrapper(items, |x| x)
}

// ── Validation ──────────────────────────────────────────────────────────────

/// An applicative validation type that collects ALL errors rather than
/// short-circuiting on the first one.
///
/// Unlike `Result`, combining two `Validation::Failure` values
/// concatenates their error lists.
#[derive(Clone, PartialEq, Eq)]
pub enum Validation<E, T> {
    Success(T),
    Failure(Vec<E>),
}

impl<E: fmt::Debug, T: fmt::Debug> fmt::Debug for Validation<E, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Validation::Success(v) => write!(f, "Success({v:?})"),
            Validation::Failure(errs) => write!(f, "Failure({errs:?})"),
        }
    }
}

impl<E, T> Validation<E, T> {
    /// Create a success.
    pub fn success(value: T) -> Self {
        Validation::Success(value)
    }

    /// Create a failure with a single error.
    pub fn failure(error: E) -> Self {
        Validation::Failure(vec![error])
    }

    /// Create a failure with multiple errors.
    pub fn failures(errors: Vec<E>) -> Self {
        Validation::Failure(errors)
    }

    /// Is this a success?
    pub fn is_success(&self) -> bool {
        matches!(self, Validation::Success(_))
    }

    /// Is this a failure?
    pub fn is_failure(&self) -> bool {
        matches!(self, Validation::Failure(_))
    }

    /// Map over the success value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Validation<E, U> {
        match self {
            Validation::Success(v) => Validation::Success(f(v)),
            Validation::Failure(errs) => Validation::Failure(errs),
        }
    }

    /// Map over the error values.
    pub fn map_errors<F>(self, f: impl Fn(E) -> F) -> Validation<F, T> {
        match self {
            Validation::Success(v) => Validation::Success(v),
            Validation::Failure(errs) => {
                Validation::Failure(errs.into_iter().map(f).collect())
            }
        }
    }

    /// Convert to a Result (errors are collected into a Vec).
    pub fn to_result(self) -> Result<T, Vec<E>> {
        match self {
            Validation::Success(v) => Ok(v),
            Validation::Failure(errs) => Err(errs),
        }
    }

    /// Create from a Result.
    pub fn from_result(res: Result<T, E>) -> Self {
        match res {
            Ok(v) => Validation::Success(v),
            Err(e) => Validation::Failure(vec![e]),
        }
    }

    /// Get the success value or a default.
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Validation::Success(v) => v,
            Validation::Failure(_) => default,
        }
    }

    /// Get errors (empty vec for Success).
    pub fn errors(&self) -> Vec<&E> {
        match self {
            Validation::Success(_) => Vec::new(),
            Validation::Failure(errs) => errs.iter().collect(),
        }
    }
}

/// Apply a validated function to a validated argument, collecting all errors.
pub fn validation_ap<E, A, B>(
    vf: Validation<E, Box<dyn FnOnce(A) -> B>>,
    va: Validation<E, A>,
) -> Validation<E, B> {
    match (vf, va) {
        (Validation::Success(f), Validation::Success(a)) => Validation::Success(f(a)),
        (Validation::Failure(mut e1), Validation::Failure(e2)) => {
            e1.extend(e2);
            Validation::Failure(e1)
        }
        (Validation::Failure(e), _) => Validation::Failure(e),
        (_, Validation::Failure(e)) => Validation::Failure(e),
    }
}

/// Combine two validations with a function, collecting all errors.
pub fn validation_combine<E, A, B, C>(
    va: Validation<E, A>,
    vb: Validation<E, B>,
    f: impl FnOnce(A, B) -> C,
) -> Validation<E, C> {
    match (va, vb) {
        (Validation::Success(a), Validation::Success(b)) => Validation::Success(f(a, b)),
        (Validation::Failure(mut e1), Validation::Failure(e2)) => {
            e1.extend(e2);
            Validation::Failure(e1)
        }
        (Validation::Failure(e), _) => Validation::Failure(e),
        (_, Validation::Failure(e)) => Validation::Failure(e),
    }
}

/// Combine three validations, collecting all errors.
pub fn validation_combine3<E, A, B, C, D>(
    va: Validation<E, A>,
    vb: Validation<E, B>,
    vc: Validation<E, C>,
    f: impl FnOnce(A, B, C) -> D,
) -> Validation<E, D> {
    let mut all_errors: Vec<E> = Vec::new();
    let a = match va {
        Validation::Success(v) => Some(v),
        Validation::Failure(e) => { all_errors.extend(e); None }
    };
    let b = match vb {
        Validation::Success(v) => Some(v),
        Validation::Failure(e) => { all_errors.extend(e); None }
    };
    let c = match vc {
        Validation::Success(v) => Some(v),
        Validation::Failure(e) => { all_errors.extend(e); None }
    };
    if all_errors.is_empty() {
        // All were successful — unwrap is safe.
        Validation::Success(f(a.unwrap(), b.unwrap(), c.unwrap()))
    } else {
        Validation::Failure(all_errors)
    }
}

/// Traverse with Validation, collecting all errors across elements.
pub fn traverse_validation<E, A, B>(
    items: Vec<A>,
    f: impl Fn(A) -> Validation<E, B>,
) -> Validation<E, Vec<B>> {
    let mut successes = Vec::new();
    let mut all_errors = Vec::new();
    for item in items {
        match f(item) {
            Validation::Success(v) => successes.push(v),
            Validation::Failure(errs) => all_errors.extend(errs),
        }
    }
    if all_errors.is_empty() {
        Validation::Success(successes)
    } else {
        Validation::Failure(all_errors)
    }
}

/// Sequence a Vec of Validations into a Validation of Vec, collecting all errors.
pub fn sequence_validation<E, T>(items: Vec<Validation<E, T>>) -> Validation<E, Vec<T>> {
    traverse_validation(items, |x| x)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_fmap() {
        let w = Wrapper::value(10);
        let result = w.fmap(|x| x * 3);
        assert_eq!(result, Wrapper::Value(30));
    }

    #[test]
    fn wrapper_fmap_empty() {
        let w: Wrapper<i32> = Wrapper::empty();
        assert_eq!(w.fmap(|x| x + 1), Wrapper::Empty);
    }

    #[test]
    fn wrapper_pure() {
        assert_eq!(Wrapper::pure(42), Wrapper::Value(42));
    }

    #[test]
    fn wrapper_apply() {
        let f: Wrapper<Box<dyn FnOnce(i32) -> i32>> =
            Wrapper::Value(Box::new(|x| x * 2));
        let result = apply(f, Wrapper::Value(5));
        assert_eq!(result, Wrapper::Value(10));
    }

    #[test]
    fn wrapper_apply_empty() {
        let f: Wrapper<Box<dyn FnOnce(i32) -> i32>> = Wrapper::Empty;
        let result = apply(f, Wrapper::Value(5));
        assert_eq!(result, Wrapper::Empty);
    }

    #[test]
    fn lift_fn() {
        let double = lift(|x: i32| x * 2);
        assert_eq!(double(Wrapper::Value(5)), Wrapper::Value(10));
        assert_eq!(double(Wrapper::Empty), Wrapper::Empty);
    }

    #[test]
    fn lift2_fn() {
        let add = lift2(|a: i32, b: i32| a + b);
        assert_eq!(add(Wrapper::Value(3), Wrapper::Value(4)), Wrapper::Value(7));
        assert_eq!(add(Wrapper::Value(3), Wrapper::Empty), Wrapper::Empty);
    }

    #[test]
    fn lift3_fn() {
        let sum3 = lift3(|a: i32, b: i32, c: i32| a + b + c);
        assert_eq!(
            sum3(Wrapper::Value(1), Wrapper::Value(2), Wrapper::Value(3)),
            Wrapper::Value(6)
        );
    }

    #[test]
    fn map_vec_fn() {
        assert_eq!(map_vec(vec![1, 2, 3], |x| x * 10), vec![10, 20, 30]);
    }

    #[test]
    fn map_option_fn() {
        assert_eq!(map_option(Some(5), |x| x + 1), Some(6));
        assert_eq!(map_option(None::<i32>, |x| x + 1), None);
    }

    #[test]
    fn map_result_fn() {
        let r: Result<i32, &str> = Ok(10);
        assert_eq!(map_result(r, |x| x * 2), Ok(20));
    }

    #[test]
    fn zip_apply_fn() {
        let fns: Vec<Box<dyn Fn(&i32) -> String>> = vec![
            Box::new(|x| format!("a={x}")),
            Box::new(|x| format!("b={x}")),
        ];
        let args = vec![1, 2];
        let result = zip_apply(fns, &args);
        assert_eq!(result, vec!["a=1", "b=2"]);
    }

    #[test]
    fn zip_apply_option_fn() {
        assert_eq!(zip_apply_option(Some(3), Some(4), |a, b| a + b), Some(7));
        assert_eq!(zip_apply_option(Some(3), None::<i32>, |a, b| a + b), None);
    }

    #[test]
    fn traverse_option_fn() {
        let result = traverse_option(vec![1, 2, 3], |x| {
            if x > 0 { Some(x * 10) } else { None }
        });
        assert_eq!(result, Some(vec![10, 20, 30]));
    }

    #[test]
    fn traverse_option_fail() {
        let result = traverse_option(vec![1, 0, 3], |x| {
            if x > 0 { Some(x * 10) } else { None }
        });
        assert_eq!(result, None);
    }

    #[test]
    fn sequence_option_fn() {
        assert_eq!(
            sequence_option(vec![Some(1), Some(2), Some(3)]),
            Some(vec![1, 2, 3])
        );
        assert_eq!(
            sequence_option(vec![Some(1), None, Some(3)]),
            None
        );
    }

    #[test]
    fn traverse_result_fn() {
        let result: Result<Vec<i32>, &str> =
            traverse_result(vec![1, 2, 3], |x| Ok(x * 10));
        assert_eq!(result, Ok(vec![10, 20, 30]));
    }

    #[test]
    fn sequence_result_fn() {
        let items: Vec<Result<i32, &str>> = vec![Ok(1), Ok(2), Ok(3)];
        assert_eq!(sequence_result(items), Ok(vec![1, 2, 3]));
    }

    #[test]
    fn traverse_wrapper_fn() {
        let result = traverse_wrapper(vec![1, 2, 3], |x| Wrapper::Value(x * 2));
        assert_eq!(result, Wrapper::Value(vec![2, 4, 6]));
    }

    #[test]
    fn sequence_wrapper_fn() {
        let items = vec![Wrapper::Value(1), Wrapper::Value(2)];
        assert_eq!(sequence_wrapper(items), Wrapper::Value(vec![1, 2]));
        let items2 = vec![Wrapper::Value(1), Wrapper::Empty];
        assert_eq!(sequence_wrapper(items2), Wrapper::Empty);
    }

    // --- Validation ---

    #[test]
    fn validation_success() {
        let v: Validation<String, i32> = Validation::success(42);
        assert!(v.is_success());
        assert!(!v.is_failure());
    }

    #[test]
    fn validation_failure() {
        let v: Validation<String, i32> = Validation::failure("bad".to_string());
        assert!(v.is_failure());
        assert_eq!(v.errors().len(), 1);
    }

    #[test]
    fn validation_map() {
        let v: Validation<String, i32> = Validation::success(10);
        let v2 = v.map(|x| x * 2);
        assert_eq!(v2, Validation::Success(20));
    }

    #[test]
    fn validation_combine_both_success() {
        let a: Validation<String, i32> = Validation::success(3);
        let b: Validation<String, i32> = Validation::success(4);
        let c = validation_combine(a, b, |x, y| x + y);
        assert_eq!(c, Validation::Success(7));
    }

    #[test]
    fn validation_combine_collects_errors() {
        let a: Validation<String, i32> = Validation::failure("err1".to_string());
        let b: Validation<String, i32> = Validation::failure("err2".to_string());
        let c = validation_combine(a, b, |x, y| x + y);
        match c {
            Validation::Failure(errs) => {
                assert_eq!(errs.len(), 2);
                assert_eq!(errs[0], "err1");
                assert_eq!(errs[1], "err2");
            }
            _ => panic!("expected failure"),
        }
    }

    #[test]
    fn validation_combine3_all_fail() {
        let a: Validation<&str, i32> = Validation::failure("e1");
        let b: Validation<&str, i32> = Validation::failure("e2");
        let c: Validation<&str, i32> = Validation::failure("e3");
        let result = validation_combine3(a, b, c, |x, y, z| x + y + z);
        match result {
            Validation::Failure(errs) => assert_eq!(errs, vec!["e1", "e2", "e3"]),
            _ => panic!("expected failure"),
        }
    }

    #[test]
    fn traverse_validation_collects_all() {
        let result = traverse_validation(vec![1, -2, 3, -4], |x| {
            if x > 0 {
                Validation::success(x)
            } else {
                Validation::failure(format!("{x} is negative"))
            }
        });
        match result {
            Validation::Failure(errs) => assert_eq!(errs.len(), 2),
            _ => panic!("expected failure"),
        }
    }

    #[test]
    fn sequence_validation_fn() {
        let items: Vec<Validation<&str, i32>> = vec![
            Validation::success(1),
            Validation::failure("e1"),
            Validation::success(3),
            Validation::failure("e2"),
        ];
        let result = sequence_validation(items);
        match result {
            Validation::Failure(errs) => assert_eq!(errs, vec!["e1", "e2"]),
            _ => panic!("expected failure"),
        }
    }

    #[test]
    fn validation_from_result() {
        let ok: Validation<&str, i32> = Validation::from_result(Ok(42));
        assert_eq!(ok, Validation::Success(42));
        let err: Validation<&str, i32> = Validation::from_result(Err("bad"));
        assert!(err.is_failure());
    }

    #[test]
    fn validation_to_result() {
        let v: Validation<&str, i32> = Validation::success(42);
        assert_eq!(v.to_result(), Ok(42));
    }

    #[test]
    fn validation_unwrap_or() {
        let v: Validation<&str, i32> = Validation::failure("oops");
        assert_eq!(v.unwrap_or(0), 0);
    }

    #[test]
    fn validation_map_errors() {
        let v: Validation<i32, &str> = Validation::failures(vec![1, 2, 3]);
        let v2 = v.map_errors(|e| e * 10);
        match v2 {
            Validation::Failure(errs) => assert_eq!(errs, vec![10, 20, 30]),
            _ => panic!("expected failure"),
        }
    }

    #[test]
    fn wrapper_to_option() {
        assert_eq!(Wrapper::Value(42).to_option(), Some(42));
        assert_eq!(Wrapper::<i32>::Empty.to_option(), None);
    }

    #[test]
    fn wrapper_unwrap_or() {
        assert_eq!(Wrapper::Value(5).unwrap_or(0), 5);
        assert_eq!(Wrapper::<i32>::Empty.unwrap_or(0), 0);
    }

    #[test]
    fn zip_apply_wrapper_fn() {
        let result = zip_apply_wrapper(
            Wrapper::Value(3),
            Wrapper::Value(4),
            |a, b| a + b,
        );
        assert_eq!(result, Wrapper::Value(7));
    }
}
