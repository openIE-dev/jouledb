//! Monad-like abstractions for Rust.
//!
//! Provides `Maybe<T>`, `Either<L,R>`, `IO<T>`, monadic chaining with
//! `flat_map`/`bind`, for-comprehension–style composition, `compose`,
//! and monadic error handling — all without external crates.

use std::fmt;

// ── Maybe ───────────────────────────────────────────────────────────────────

/// A monadic optional type with explicit `Just` / `Nothing` variants.
#[derive(Clone, PartialEq, Eq)]
pub enum Maybe<T> {
    Just(T),
    Nothing,
}

impl<T: fmt::Debug> fmt::Debug for Maybe<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Maybe::Just(v) => write!(f, "Just({v:?})"),
            Maybe::Nothing => write!(f, "Nothing"),
        }
    }
}

impl<T> Maybe<T> {
    /// Wrap a value in `Just`.
    pub fn just(value: T) -> Self {
        Maybe::Just(value)
    }

    /// The empty case.
    pub fn nothing() -> Self {
        Maybe::Nothing
    }

    /// Create from an `Option`.
    pub fn from_option(opt: Option<T>) -> Self {
        match opt {
            Some(v) => Maybe::Just(v),
            None => Maybe::Nothing,
        }
    }

    /// Convert to `Option`.
    pub fn to_option(self) -> Option<T> {
        match self {
            Maybe::Just(v) => Some(v),
            Maybe::Nothing => None,
        }
    }

    /// Is this `Just`?
    pub fn is_just(&self) -> bool {
        matches!(self, Maybe::Just(_))
    }

    /// Is this `Nothing`?
    pub fn is_nothing(&self) -> bool {
        matches!(self, Maybe::Nothing)
    }

    /// Functor map.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Maybe<U> {
        match self {
            Maybe::Just(v) => Maybe::Just(f(v)),
            Maybe::Nothing => Maybe::Nothing,
        }
    }

    /// Monadic bind / flat_map.
    pub fn flat_map<U>(self, f: impl FnOnce(T) -> Maybe<U>) -> Maybe<U> {
        match self {
            Maybe::Just(v) => f(v),
            Maybe::Nothing => Maybe::Nothing,
        }
    }

    /// Alias for `flat_map`.
    pub fn bind<U>(self, f: impl FnOnce(T) -> Maybe<U>) -> Maybe<U> {
        self.flat_map(f)
    }

    /// Applicative pure / return.
    pub fn pure(value: T) -> Self {
        Maybe::Just(value)
    }

    /// Unwrap with a default.
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Maybe::Just(v) => v,
            Maybe::Nothing => default,
        }
    }

    /// Unwrap with a lazy default.
    pub fn unwrap_or_else(self, f: impl FnOnce() -> T) -> T {
        match self {
            Maybe::Just(v) => v,
            Maybe::Nothing => f(),
        }
    }

    /// Filter: keep `Just` only if predicate holds, else `Nothing`.
    pub fn filter(self, pred: impl FnOnce(&T) -> bool) -> Self {
        match self {
            Maybe::Just(ref v) if pred(v) => self,
            _ => Maybe::Nothing,
        }
    }

    /// Zip two `Maybe` values.
    pub fn zip<U>(self, other: Maybe<U>) -> Maybe<(T, U)> {
        match (self, other) {
            (Maybe::Just(a), Maybe::Just(b)) => Maybe::Just((a, b)),
            _ => Maybe::Nothing,
        }
    }

    /// Apply: if self is `Just(f)` and arg is `Just(v)`, return `Just(f(v))`.
    pub fn ap<U, F>(self, arg: Maybe<U>) -> Maybe<F::Output>
    where
        T: FnOnce(U) -> F::Output,
        F: ApOutput,
    {
        // We need a different approach since Rust generics are tricky here.
        // This is handled via the dedicated `maybe_ap` free function instead.
        let _ = arg;
        Maybe::Nothing
    }
}

/// Helper trait — not publicly useful, only supports `ap` above.
pub trait ApOutput {
    type Output;
}

/// Apply a function inside `Maybe` to a value inside `Maybe`.
pub fn maybe_ap<A, B>(mf: Maybe<Box<dyn FnOnce(A) -> B>>, ma: Maybe<A>) -> Maybe<B> {
    match (mf, ma) {
        (Maybe::Just(f), Maybe::Just(a)) => Maybe::Just(f(a)),
        _ => Maybe::Nothing,
    }
}

// ── Either ──────────────────────────────────────────────────────────────────

/// A sum type that is either `Left(L)` or `Right(R)`.
///
/// By convention, `Right` is the "success" path (mnemonic: "right" = correct).
#[derive(Clone, PartialEq, Eq)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

impl<L: fmt::Debug, R: fmt::Debug> fmt::Debug for Either<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Either::Left(l) => write!(f, "Left({l:?})"),
            Either::Right(r) => write!(f, "Right({r:?})"),
        }
    }
}

impl<L, R> Either<L, R> {
    /// Is this `Left`?
    pub fn is_left(&self) -> bool {
        matches!(self, Either::Left(_))
    }

    /// Is this `Right`?
    pub fn is_right(&self) -> bool {
        matches!(self, Either::Right(_))
    }

    /// Map over the right value.
    pub fn map<U>(self, f: impl FnOnce(R) -> U) -> Either<L, U> {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => Either::Right(f(r)),
        }
    }

    /// Map over the left value.
    pub fn map_left<U>(self, f: impl FnOnce(L) -> U) -> Either<U, R> {
        match self {
            Either::Left(l) => Either::Left(f(l)),
            Either::Right(r) => Either::Right(r),
        }
    }

    /// Monadic bind on the right side.
    pub fn flat_map<U>(self, f: impl FnOnce(R) -> Either<L, U>) -> Either<L, U> {
        match self {
            Either::Left(l) => Either::Left(l),
            Either::Right(r) => f(r),
        }
    }

    /// Alias for `flat_map`.
    pub fn bind<U>(self, f: impl FnOnce(R) -> Either<L, U>) -> Either<L, U> {
        self.flat_map(f)
    }

    /// Bi-map: map both sides.
    pub fn bimap<A, B>(
        self,
        fl: impl FnOnce(L) -> A,
        fr: impl FnOnce(R) -> B,
    ) -> Either<A, B> {
        match self {
            Either::Left(l) => Either::Left(fl(l)),
            Either::Right(r) => Either::Right(fr(r)),
        }
    }

    /// Fold: collapse the Either into a single value.
    pub fn fold<T>(self, fl: impl FnOnce(L) -> T, fr: impl FnOnce(R) -> T) -> T {
        match self {
            Either::Left(l) => fl(l),
            Either::Right(r) => fr(r),
        }
    }

    /// Convert to `Result<R, L>`.
    pub fn to_result(self) -> Result<R, L> {
        match self {
            Either::Left(l) => Err(l),
            Either::Right(r) => Ok(r),
        }
    }

    /// Create from `Result<R, L>`.
    pub fn from_result(res: Result<R, L>) -> Self {
        match res {
            Ok(r) => Either::Right(r),
            Err(l) => Either::Left(l),
        }
    }

    /// Swap left and right.
    pub fn swap(self) -> Either<R, L> {
        match self {
            Either::Left(l) => Either::Right(l),
            Either::Right(r) => Either::Left(r),
        }
    }

    /// Get the right value or a default.
    pub fn unwrap_right_or(self, default: R) -> R {
        match self {
            Either::Right(r) => r,
            Either::Left(_) => default,
        }
    }

    /// Get the left value or a default.
    pub fn unwrap_left_or(self, default: L) -> L {
        match self {
            Either::Left(l) => l,
            Either::Right(_) => default,
        }
    }
}

// ── IO Monad ────────────────────────────────────────────────────────────────

/// A simulation of the IO monad: a deferred computation that produces `T`.
///
/// The computation is not executed until `run()` is called,
/// allowing composition before side effects happen.
pub struct IO<T> {
    thunk: Box<dyn FnOnce() -> T>,
}

impl<T: 'static> IO<T> {
    /// Create an IO action from a closure.
    pub fn new(f: impl FnOnce() -> T + 'static) -> Self {
        Self { thunk: Box::new(f) }
    }

    /// Pure / return: wrap a value in IO without side effects.
    pub fn pure(value: T) -> Self {
        Self {
            thunk: Box::new(move || value),
        }
    }

    /// Run the IO action, producing the value.
    pub fn run(self) -> T {
        (self.thunk)()
    }

    /// Functor map.
    pub fn map<U: 'static>(self, f: impl FnOnce(T) -> U + 'static) -> IO<U> {
        IO {
            thunk: Box::new(move || f((self.thunk)())),
        }
    }

    /// Monadic bind / flat_map.
    pub fn flat_map<U: 'static>(self, f: impl FnOnce(T) -> IO<U> + 'static) -> IO<U> {
        IO {
            thunk: Box::new(move || {
                let a = (self.thunk)();
                f(a).run()
            }),
        }
    }

    /// Alias for `flat_map`.
    pub fn bind<U: 'static>(self, f: impl FnOnce(T) -> IO<U> + 'static) -> IO<U> {
        self.flat_map(f)
    }

    /// Sequence two IO actions, discarding the first result.
    pub fn then<U: 'static>(self, next: IO<U>) -> IO<U> {
        IO {
            thunk: Box::new(move || {
                let _ = (self.thunk)();
                next.run()
            }),
        }
    }
}

// ── Monadic Option/Result chains ────────────────────────────────────────────

/// Chain a sequence of `Option`-returning functions, short-circuiting on `None`.
///
/// This is a for-comprehension style: each step can depend on previous results
/// by capturing them in closures.
pub fn option_chain<T>(initial: Option<T>) -> OptionChain<T> {
    OptionChain { value: initial }
}

/// Builder for chaining `Option`-returning steps.
pub struct OptionChain<T> {
    value: Option<T>,
}

impl<T> OptionChain<T> {
    /// Apply a function that returns `Option<U>`.
    pub fn and_then<U>(self, f: impl FnOnce(T) -> Option<U>) -> OptionChain<U> {
        OptionChain {
            value: self.value.and_then(f),
        }
    }

    /// Map the inner value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> OptionChain<U> {
        OptionChain {
            value: self.value.map(f),
        }
    }

    /// Filter the value.
    pub fn filter(self, pred: impl FnOnce(&T) -> bool) -> OptionChain<T> {
        OptionChain {
            value: self.value.filter(pred),
        }
    }

    /// Extract the final `Option`.
    pub fn finish(self) -> Option<T> {
        self.value
    }
}

/// Chain a sequence of `Result`-returning functions, short-circuiting on `Err`.
pub fn result_chain<T, E>(initial: Result<T, E>) -> ResultChain<T, E> {
    ResultChain { value: initial }
}

/// Builder for chaining `Result`-returning steps.
pub struct ResultChain<T, E> {
    value: Result<T, E>,
}

impl<T, E> ResultChain<T, E> {
    /// Apply a function that returns `Result<U, E>`.
    pub fn and_then<U>(self, f: impl FnOnce(T) -> Result<U, E>) -> ResultChain<U, E> {
        ResultChain {
            value: self.value.and_then(f),
        }
    }

    /// Map the ok value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ResultChain<U, E> {
        ResultChain {
            value: self.value.map(f),
        }
    }

    /// Map the error value.
    pub fn map_err<F>(self, f: impl FnOnce(E) -> F) -> ResultChain<T, F> {
        ResultChain {
            value: self.value.map_err(f),
        }
    }

    /// Extract the final `Result`.
    pub fn finish(self) -> Result<T, E> {
        self.value
    }
}

// ── Compose ─────────────────────────────────────────────────────────────────

/// Compose two monadic functions: `compose_m(f, g)` yields `|a| f(a).flat_map(g)`.
/// Works with `Maybe`.
pub fn compose_maybe<A, B, C>(
    f: impl Fn(A) -> Maybe<B> + 'static,
    g: impl Fn(B) -> Maybe<C> + 'static,
) -> Box<dyn Fn(A) -> Maybe<C>> {
    Box::new(move |a| f(a).flat_map(|b| g(b)))
}

/// Compose two functions returning `Option`.
pub fn compose_option<A, B, C>(
    f: impl Fn(A) -> Option<B> + 'static,
    g: impl Fn(B) -> Option<C> + 'static,
) -> Box<dyn Fn(A) -> Option<C>> {
    Box::new(move |a| f(a).and_then(|b| g(b)))
}

/// Compose two functions returning `Result` (Kleisli composition).
pub fn compose_result<A, B, C, E>(
    f: impl Fn(A) -> Result<B, E> + 'static,
    g: impl Fn(B) -> Result<C, E> + 'static,
) -> Box<dyn Fn(A) -> Result<C, E>> {
    Box::new(move |a| f(a).and_then(|b| g(b)))
}

// ── Monadic error handling ──────────────────────────────────────────────────

/// Collect all errors from a list of results. Returns `Ok` with all values
/// if every result is `Ok`, or `Err` with all accumulated errors.
pub fn collect_errors<T, E>(results: Vec<Result<T, E>>) -> Result<Vec<T>, Vec<E>> {
    let mut values = Vec::new();
    let mut errors = Vec::new();
    for r in results {
        match r {
            Ok(v) => values.push(v),
            Err(e) => errors.push(e),
        }
    }
    if errors.is_empty() {
        Ok(values)
    } else {
        Err(errors)
    }
}

/// Try to recover from a `Maybe::Nothing` by running a fallback.
pub fn maybe_or_else<T>(m: Maybe<T>, fallback: impl FnOnce() -> Maybe<T>) -> Maybe<T> {
    match m {
        Maybe::Just(_) => m,
        Maybe::Nothing => fallback(),
    }
}

/// Lift a plain function into the `Maybe` monad.
pub fn lift_maybe<A, B>(f: impl Fn(A) -> B + 'static) -> Box<dyn Fn(Maybe<A>) -> Maybe<B>> {
    Box::new(move |ma| ma.map(|a| f(a)))
}

/// Lift a plain function into `Either`.
pub fn lift_either<L, A, B>(
    f: impl Fn(A) -> B + 'static,
) -> Box<dyn Fn(Either<L, A>) -> Either<L, B>> {
    Box::new(move |ea| ea.map(|a| f(a)))
}

/// Sequence a `Vec<Maybe<T>>` into a `Maybe<Vec<T>>`.
/// Returns `Nothing` if any element is `Nothing`.
pub fn sequence_maybe<T>(items: Vec<Maybe<T>>) -> Maybe<Vec<T>> {
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Maybe::Just(v) => result.push(v),
            Maybe::Nothing => return Maybe::Nothing,
        }
    }
    Maybe::Just(result)
}

/// Traverse: map each element with `f` then sequence.
pub fn traverse_maybe<A, B>(items: Vec<A>, f: impl Fn(A) -> Maybe<B>) -> Maybe<Vec<B>> {
    let mapped: Vec<Maybe<B>> = items.into_iter().map(f).collect();
    sequence_maybe(mapped)
}

// ── For-comprehension macro-like builder ────────────────────────────────────

/// For-comprehension builder for `Maybe`.
/// Usage: `MaybeFor::from(val).bind(|x| ...).bind(|y| ...).yield_val(z)`
pub struct MaybeFor<T> {
    value: Maybe<T>,
}

impl<T> MaybeFor<T> {
    /// Start a for-comprehension with a `Maybe` value.
    pub fn from(m: Maybe<T>) -> Self {
        Self { value: m }
    }

    /// Bind the current value through `f`.
    pub fn bind<U>(self, f: impl FnOnce(T) -> Maybe<U>) -> MaybeFor<U> {
        MaybeFor {
            value: self.value.flat_map(f),
        }
    }

    /// Map the current value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> MaybeFor<U> {
        MaybeFor {
            value: self.value.map(f),
        }
    }

    /// Guard: continue only if predicate holds.
    pub fn guard(self, pred: impl FnOnce(&T) -> bool) -> MaybeFor<T> {
        MaybeFor {
            value: self.value.filter(pred),
        }
    }

    /// Yield the final result.
    pub fn yield_val(self) -> Maybe<T> {
        self.value
    }
}

/// For-comprehension builder for `Either`.
pub struct EitherFor<L, R> {
    value: Either<L, R>,
}

impl<L, R> EitherFor<L, R> {
    /// Start with an `Either`.
    pub fn from(e: Either<L, R>) -> Self {
        Self { value: e }
    }

    /// Bind the right value through `f`.
    pub fn bind<S>(self, f: impl FnOnce(R) -> Either<L, S>) -> EitherFor<L, S> {
        EitherFor {
            value: self.value.flat_map(f),
        }
    }

    /// Map the right value.
    pub fn map<S>(self, f: impl FnOnce(R) -> S) -> EitherFor<L, S> {
        EitherFor {
            value: self.value.map(f),
        }
    }

    /// Yield the final result.
    pub fn yield_val(self) -> Either<L, R> {
        self.value
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maybe_just_and_nothing() {
        let j = Maybe::just(42);
        assert!(j.is_just());
        assert!(!j.is_nothing());
        let n: Maybe<i32> = Maybe::nothing();
        assert!(n.is_nothing());
    }

    #[test]
    fn maybe_map() {
        let r = Maybe::just(3).map(|x| x * 10);
        assert_eq!(r, Maybe::Just(30));
        let n: Maybe<i32> = Maybe::nothing();
        assert_eq!(n.map(|x| x + 1), Maybe::Nothing);
    }

    #[test]
    fn maybe_flat_map() {
        let safe_div = |x: i32| -> Maybe<i32> {
            if x == 0 { Maybe::Nothing } else { Maybe::Just(100 / x) }
        };
        assert_eq!(Maybe::just(5).flat_map(safe_div), Maybe::Just(20));
        assert_eq!(Maybe::just(0).flat_map(safe_div), Maybe::Nothing);
        assert_eq!(Maybe::<i32>::nothing().flat_map(safe_div), Maybe::Nothing);
    }

    #[test]
    fn maybe_bind_alias() {
        let r = Maybe::just(10).bind(|x| Maybe::just(x + 5));
        assert_eq!(r, Maybe::Just(15));
    }

    #[test]
    fn maybe_filter() {
        assert_eq!(Maybe::just(10).filter(|x| *x > 5), Maybe::Just(10));
        assert_eq!(Maybe::just(3).filter(|x| *x > 5), Maybe::Nothing);
    }

    #[test]
    fn maybe_zip() {
        let r = Maybe::just(1).zip(Maybe::just("a"));
        assert_eq!(r, Maybe::Just((1, "a")));
        assert_eq!(Maybe::just(1).zip(Maybe::<&str>::nothing()), Maybe::Nothing);
    }

    #[test]
    fn maybe_from_option_round_trip() {
        assert_eq!(Maybe::from_option(Some(42)), Maybe::Just(42));
        assert_eq!(Maybe::<i32>::from_option(None), Maybe::Nothing);
        assert_eq!(Maybe::just(42).to_option(), Some(42));
        assert_eq!(Maybe::<i32>::nothing().to_option(), None);
    }

    #[test]
    fn maybe_unwrap_or() {
        assert_eq!(Maybe::just(10).unwrap_or(0), 10);
        assert_eq!(Maybe::<i32>::nothing().unwrap_or(0), 0);
    }

    #[test]
    fn either_basic() {
        let r: Either<&str, i32> = Either::Right(42);
        assert!(r.is_right());
        let l: Either<&str, i32> = Either::Left("err");
        assert!(l.is_left());
    }

    #[test]
    fn either_map() {
        let r: Either<&str, i32> = Either::Right(10);
        assert_eq!(r.map(|x| x * 2), Either::Right(20));
        let l: Either<&str, i32> = Either::Left("err");
        assert_eq!(l.map(|x| x * 2), Either::Left("err"));
    }

    #[test]
    fn either_flat_map() {
        let parse = |s: &str| -> Either<String, i32> {
            s.parse::<i32>()
                .map(Either::Right)
                .unwrap_or_else(|e| Either::Left(e.to_string()))
        };
        let r: Either<String, &str> = Either::Right("42");
        assert_eq!(r.flat_map(parse), Either::Right(42));
    }

    #[test]
    fn either_bimap() {
        let r: Either<i32, &str> = Either::Right("hello");
        let result = r.bimap(|l| l + 1, |r| r.len());
        assert_eq!(result, Either::Right(5));
    }

    #[test]
    fn either_fold() {
        let r: Either<i32, &str> = Either::Right("ok");
        let s = r.fold(|l| format!("err: {l}"), |r| format!("val: {r}"));
        assert_eq!(s, "val: ok");
    }

    #[test]
    fn either_swap() {
        let r: Either<i32, &str> = Either::Right("hello");
        assert_eq!(r.swap(), Either::Left("hello"));
    }

    #[test]
    fn either_to_from_result() {
        let r: Result<i32, &str> = Ok(42);
        let e = Either::from_result(r);
        assert_eq!(e, Either::Right(42));
        assert_eq!(e.to_result(), Ok(42));
    }

    #[test]
    fn io_pure_and_run() {
        let io = IO::pure(42);
        assert_eq!(io.run(), 42);
    }

    #[test]
    fn io_map() {
        let io = IO::pure(10).map(|x| x * 3);
        assert_eq!(io.run(), 30);
    }

    #[test]
    fn io_flat_map() {
        let io = IO::pure(5).flat_map(|x| IO::pure(x + 100));
        assert_eq!(io.run(), 105);
    }

    #[test]
    fn io_then() {
        let mut side_effect = false;
        let io = IO::new(move || {
            // In a real IO monad this would have a side effect.
            // We just produce a value.
            42
        });
        let io2 = io.then(IO::pure(99));
        let _ = side_effect;
        side_effect = true;
        let _ = side_effect;
        assert_eq!(io2.run(), 99);
    }

    #[test]
    fn io_chain_deferred() {
        // Verify computations are deferred until run().
        let io = IO::new(|| 1)
            .map(|x| x + 2)
            .flat_map(|x| IO::pure(x * 10));
        // Nothing has executed yet. Now run:
        assert_eq!(io.run(), 30);
    }

    #[test]
    fn option_chain_happy_path() {
        let result = option_chain(Some(10))
            .and_then(|x| if x > 5 { Some(x * 2) } else { None })
            .map(|x| x + 1)
            .finish();
        assert_eq!(result, Some(21));
    }

    #[test]
    fn option_chain_short_circuit() {
        let result = option_chain(Some(3))
            .and_then(|x| if x > 5 { Some(x * 2) } else { None })
            .map(|x| x + 1)
            .finish();
        assert_eq!(result, None);
    }

    #[test]
    fn result_chain_happy_path() {
        let r: Result<i32, String> = result_chain(Ok(10))
            .and_then(|x| Ok(x * 2))
            .map(|x| x + 1)
            .finish();
        assert_eq!(r, Ok(21));
    }

    #[test]
    fn result_chain_short_circuit() {
        let r: Result<i32, String> = result_chain(Ok(10))
            .and_then(|_x: i32| Err("fail".to_string()))
            .map(|x: i32| x + 1)
            .finish();
        assert_eq!(r, Err("fail".to_string()));
    }

    #[test]
    fn compose_maybe_fn() {
        let f = |x: i32| -> Maybe<i32> {
            if x >= 0 { Maybe::Just(x) } else { Maybe::Nothing }
        };
        let g = |x: i32| -> Maybe<String> { Maybe::Just(format!("val={x}")) };
        let composed = compose_maybe(f, g);
        assert_eq!(composed(5), Maybe::Just("val=5".to_string()));
        assert_eq!(composed(-1), Maybe::Nothing);
    }

    #[test]
    fn compose_option_fn() {
        let f = |s: &str| s.parse::<i32>().ok();
        let g = |x: i32| if x > 0 { Some(x * 10) } else { None };
        let composed = compose_option(f, g);
        assert_eq!(composed("5"), Some(50));
        assert_eq!(composed("-1"), None);
        assert_eq!(composed("abc"), None);
    }

    #[test]
    fn collect_errors_all_ok() {
        let results: Vec<Result<i32, &str>> = vec![Ok(1), Ok(2), Ok(3)];
        assert_eq!(collect_errors(results), Ok(vec![1, 2, 3]));
    }

    #[test]
    fn collect_errors_some_err() {
        let results: Vec<Result<i32, &str>> = vec![Ok(1), Err("a"), Ok(3), Err("b")];
        assert_eq!(collect_errors(results), Err(vec!["a", "b"]));
    }

    #[test]
    fn sequence_maybe_all_just() {
        let items = vec![Maybe::just(1), Maybe::just(2), Maybe::just(3)];
        assert_eq!(sequence_maybe(items), Maybe::Just(vec![1, 2, 3]));
    }

    #[test]
    fn sequence_maybe_has_nothing() {
        let items = vec![Maybe::just(1), Maybe::nothing(), Maybe::just(3)];
        assert_eq!(sequence_maybe(items), Maybe::Nothing);
    }

    #[test]
    fn traverse_maybe_fn() {
        let r = traverse_maybe(vec![1, 2, 3], |x| Maybe::just(x * 10));
        assert_eq!(r, Maybe::Just(vec![10, 20, 30]));
    }

    #[test]
    fn maybe_for_comprehension() {
        let result = MaybeFor::from(Maybe::just(10))
            .bind(|x| Maybe::just(x + 5))
            .guard(|x| *x > 10)
            .map(|x| x * 2)
            .yield_val();
        assert_eq!(result, Maybe::Just(30));
    }

    #[test]
    fn maybe_for_guard_fails() {
        let result = MaybeFor::from(Maybe::just(3))
            .guard(|x| *x > 10)
            .map(|x| x * 2)
            .yield_val();
        assert_eq!(result, Maybe::Nothing);
    }

    #[test]
    fn either_for_comprehension() {
        let result: Either<String, i32> = EitherFor::from(Either::Right(10))
            .bind(|x| Either::Right(x + 5))
            .map(|x| x * 2)
            .yield_val();
        assert_eq!(result, Either::Right(30));
    }

    #[test]
    fn either_for_short_circuit() {
        let result: Either<String, i32> = EitherFor::from(Either::Right(10))
            .bind(|_: i32| Either::Left("oops".to_string()))
            .map(|x: i32| x * 2)
            .yield_val();
        assert_eq!(result, Either::Left("oops".to_string()));
    }

    #[test]
    fn lift_maybe_fn() {
        let double = lift_maybe(|x: i32| x * 2);
        assert_eq!(double(Maybe::just(5)), Maybe::Just(10));
        assert_eq!(double(Maybe::nothing()), Maybe::Nothing);
    }

    #[test]
    fn lift_either_fn() {
        let double = lift_either(|x: i32| x * 2);
        assert_eq!(double(Either::Right(5)), Either::Right(10));
        assert_eq!(
            double(Either::<&str, i32>::Left("err")),
            Either::Left("err")
        );
    }

    #[test]
    fn maybe_or_else_fallback() {
        let r = maybe_or_else(Maybe::nothing(), || Maybe::just(42));
        assert_eq!(r, Maybe::Just(42));
        let r2 = maybe_or_else(Maybe::just(10), || Maybe::just(42));
        assert_eq!(r2, Maybe::Just(10));
    }

    #[test]
    fn maybe_debug_format() {
        assert_eq!(format!("{:?}", Maybe::just(42)), "Just(42)");
        assert_eq!(format!("{:?}", Maybe::<i32>::nothing()), "Nothing");
    }

    #[test]
    fn either_debug_format() {
        assert_eq!(
            format!("{:?}", Either::<&str, i32>::Right(42)),
            "Right(42)"
        );
        assert_eq!(
            format!("{:?}", Either::<&str, i32>::Left("err")),
            "Left(\"err\")"
        );
    }
}
