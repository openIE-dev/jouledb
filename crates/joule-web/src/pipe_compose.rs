//! Function pipeline and composition utilities.
//!
//! Provides pipe operator simulation (`Pipeline::of(val).pipe(f).pipe(g)`),
//! function composition, partial application, currying helpers,
//! `tap` (side effect without changing value), `identity`, and `constant`.

use std::fmt;

// ── Identity & Constant ─────────────────────────────────────────────────────

/// The identity function: returns its argument unchanged.
pub fn identity<T>(x: T) -> T {
    x
}

/// Create a function that always returns `value`, ignoring its argument.
pub fn constant<T: Clone + 'static>(value: T) -> Box<dyn Fn() -> T> {
    Box::new(move || value.clone())
}

/// Create a unary function that ignores its argument and returns `value`.
pub fn constant1<A, T: Clone + 'static>(value: T) -> Box<dyn Fn(A) -> T> {
    Box::new(move |_| value.clone())
}

// ── Tap ─────────────────────────────────────────────────────────────────────

/// Perform a side effect on a reference to `value`, then return `value` unchanged.
pub fn tap<T>(value: T, f: impl FnOnce(&T)) -> T {
    f(&value);
    value
}

/// Perform a mutable side effect on `value`, then return it.
pub fn tap_mut<T>(mut value: T, f: impl FnOnce(&mut T)) -> T {
    f(&mut value);
    value
}

// ── Pipeline ────────────────────────────────────────────────────────────────

/// A chainable pipeline that threads a value through a series of transforms.
///
/// ```text
/// Pipeline::of(10)
///     .pipe(|x| x * 2)
///     .pipe(|x| x + 1)
///     .tap(|x| println!("value: {x}"))
///     .value()    // → 21
/// ```
pub struct Pipeline<T> {
    val: T,
}

impl<T> Pipeline<T> {
    /// Start a pipeline with a value.
    pub fn of(value: T) -> Self {
        Self { val: value }
    }

    /// Apply a transform, producing a new pipeline.
    pub fn pipe<U>(self, f: impl FnOnce(T) -> U) -> Pipeline<U> {
        Pipeline { val: f(self.val) }
    }

    /// Conditionally apply a transform.
    pub fn pipe_if(self, condition: bool, f: impl FnOnce(T) -> T) -> Pipeline<T> {
        if condition {
            Pipeline { val: f(self.val) }
        } else {
            self
        }
    }

    /// Apply a fallible transform; returns `Result<Pipeline<U>, E>`.
    pub fn try_pipe<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Pipeline<U>, E> {
        f(self.val).map(|v| Pipeline { val: v })
    }

    /// Side effect: inspect the value without consuming it.
    pub fn tap(self, f: impl FnOnce(&T)) -> Self {
        f(&self.val);
        self
    }

    /// Mutable side effect.
    pub fn tap_mut(mut self, f: impl FnOnce(&mut T)) -> Self {
        f(&mut self.val);
        self
    }

    /// Extract the final value.
    pub fn value(self) -> T {
        self.val
    }

    /// Apply a function that takes a reference.
    pub fn inspect<U>(&self, f: impl FnOnce(&T) -> U) -> U {
        f(&self.val)
    }
}

impl<T: fmt::Debug> fmt::Debug for Pipeline<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipeline({:?})", self.val)
    }
}

impl<T: Clone> Clone for Pipeline<T> {
    fn clone(&self) -> Self {
        Pipeline {
            val: self.val.clone(),
        }
    }
}

impl<T: PartialEq> PartialEq for Pipeline<T> {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

// ── Compose ─────────────────────────────────────────────────────────────────

/// Compose two functions left-to-right: `compose(f, g)` yields `|x| g(f(x))`.
pub fn compose<A, B, C>(
    f: impl Fn(A) -> B + 'static,
    g: impl Fn(B) -> C + 'static,
) -> Box<dyn Fn(A) -> C> {
    Box::new(move |a| g(f(a)))
}

/// Compose three functions left-to-right.
pub fn compose3<A, B, C, D>(
    f: impl Fn(A) -> B + 'static,
    g: impl Fn(B) -> C + 'static,
    h: impl Fn(C) -> D + 'static,
) -> Box<dyn Fn(A) -> D> {
    Box::new(move |a| h(g(f(a))))
}

/// Compose a vector of functions (all same type) left-to-right.
pub fn compose_all<T: 'static>(fns: Vec<Box<dyn Fn(T) -> T>>) -> Box<dyn Fn(T) -> T> {
    Box::new(move |mut val| {
        for f in &fns {
            val = f(val);
        }
        val
    })
}

/// Compose right-to-left (mathematical convention): `compose_r(f, g)` yields `|x| f(g(x))`.
pub fn compose_r<A, B, C>(
    f: impl Fn(B) -> C + 'static,
    g: impl Fn(A) -> B + 'static,
) -> Box<dyn Fn(A) -> C> {
    Box::new(move |a| f(g(a)))
}

// ── Partial Application ─────────────────────────────────────────────────────

/// Partially apply the first argument of a binary function.
pub fn partial1<A: Clone + 'static, B, C>(
    f: impl Fn(A, B) -> C + 'static,
    a: A,
) -> Box<dyn Fn(B) -> C> {
    Box::new(move |b| f(a.clone(), b))
}

/// Partially apply the second argument of a binary function.
pub fn partial2<A, B: Clone + 'static, C>(
    f: impl Fn(A, B) -> C + 'static,
    b: B,
) -> Box<dyn Fn(A) -> C> {
    Box::new(move |a| f(a, b.clone()))
}

/// Partially apply the first two arguments of a ternary function.
pub fn partial12<A: Clone + 'static, B: Clone + 'static, C, D>(
    f: impl Fn(A, B, C) -> D + 'static,
    a: A,
    b: B,
) -> Box<dyn Fn(C) -> D> {
    Box::new(move |c| f(a.clone(), b.clone(), c))
}

// ── Currying ────────────────────────────────────────────────────────────────

/// Curry a binary function: `curry2(f)` yields `|a| |b| f(a, b)`.
///
/// Wraps the function in `Rc` internally so it can be shared across
/// the returned closures.
pub fn curry2<A: Clone + 'static, B: 'static, C: 'static>(
    f: impl Fn(A, B) -> C + 'static,
) -> Box<dyn Fn(A) -> Box<dyn Fn(B) -> C>> {
    let f = std::rc::Rc::new(f);
    Box::new(move |a: A| {
        let f = f.clone();
        Box::new(move |b: B| f(a.clone(), b))
    })
}

/// Curry a binary function using `Rc` for shared ownership.
pub fn curry2_rc<A: Clone + 'static, B: 'static, C: 'static>(
    f: std::rc::Rc<dyn Fn(A, B) -> C>,
) -> Box<dyn Fn(A) -> Box<dyn Fn(B) -> C>> {
    Box::new(move |a: A| {
        let f = f.clone();
        let a = a;
        Box::new(move |b: B| f(a.clone(), b))
    })
}

/// Curry a ternary function using `Rc`.
pub fn curry3_rc<A: Clone + 'static, B: Clone + 'static, C: 'static, D: 'static>(
    f: std::rc::Rc<dyn Fn(A, B, C) -> D>,
) -> Box<dyn Fn(A) -> Box<dyn Fn(B) -> Box<dyn Fn(C) -> D>>> {
    Box::new(move |a: A| {
        let f = f.clone();
        Box::new(move |b: B| {
            let f = f.clone();
            let a = a.clone();
            Box::new(move |c: C| f(a.clone(), b.clone(), c))
        })
    })
}

/// Uncurry a curried binary function.
pub fn uncurry2<A, B, C>(
    f: impl Fn(A) -> Box<dyn Fn(B) -> C> + 'static,
) -> Box<dyn Fn(A, B) -> C> {
    Box::new(move |a, b| f(a)(b))
}

// ── Flip ────────────────────────────────────────────────────────────────────

/// Flip the arguments of a binary function.
pub fn flip<A, B, C>(f: impl Fn(A, B) -> C + 'static) -> Box<dyn Fn(B, A) -> C> {
    Box::new(move |b, a| f(a, b))
}

// ── On ──────────────────────────────────────────────────────────────────────

/// The `on` combinator: `on(f, g)` yields `|a, b| f(g(a), g(b))`.
///
/// Useful for comparing/combining values after a projection.
pub fn on<A, B, C>(
    f: impl Fn(B, B) -> C + 'static,
    g: impl Fn(A) -> B + 'static,
) -> Box<dyn Fn(A, A) -> C> {
    Box::new(move |a1, a2| {
        let b1 = g(a1);
        let b2 = g(a2);
        f(b1, b2)
    })
}

// ── Converge ────────────────────────────────────────────────────────────────

/// Apply multiple functions to the same input, then combine the results.
///
/// `converge(combiner, [f, g])(x)` is `combiner(f(x), g(x))`.
pub fn converge2<A: Clone, B, C, D>(
    combiner: impl Fn(B, C) -> D + 'static,
    f: impl Fn(A) -> B + 'static,
    g: impl Fn(A) -> C + 'static,
) -> Box<dyn Fn(A) -> D> {
    Box::new(move |a: A| {
        let b = f(a.clone());
        let c = g(a);
        combiner(b, c)
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn identity_fn() {
        assert_eq!(identity(42), 42);
        assert_eq!(identity("hello"), "hello");
    }

    #[test]
    fn constant_fn() {
        let always_five = constant(5);
        assert_eq!(always_five(), 5);
        assert_eq!(always_five(), 5);
    }

    #[test]
    fn constant1_fn() {
        let always_ten = constant1::<i32, _>(10);
        assert_eq!(always_ten(99), 10);
        assert_eq!(always_ten(0), 10);
    }

    #[test]
    fn tap_fn() {
        let log = Rc::new(RefCell::new(String::new()));
        let log_ref = log.clone();
        let result = tap(42, |x| {
            *log_ref.borrow_mut() = format!("{x}");
        });
        assert_eq!(result, 42);
        assert_eq!(*log.borrow(), "42");
    }

    #[test]
    fn tap_mut_fn() {
        let result = tap_mut(vec![1, 2, 3], |v| v.push(4));
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn pipeline_basic() {
        let result = Pipeline::of(10)
            .pipe(|x| x * 2)
            .pipe(|x| x + 1)
            .value();
        assert_eq!(result, 21);
    }

    #[test]
    fn pipeline_pipe_if_true() {
        let result = Pipeline::of(10)
            .pipe_if(true, |x| x * 2)
            .value();
        assert_eq!(result, 20);
    }

    #[test]
    fn pipeline_pipe_if_false() {
        let result = Pipeline::of(10)
            .pipe_if(false, |x| x * 2)
            .value();
        assert_eq!(result, 10);
    }

    #[test]
    fn pipeline_try_pipe_ok() {
        let result = Pipeline::of(10)
            .try_pipe(|x| Ok::<_, &str>(x * 2))
            .unwrap()
            .value();
        assert_eq!(result, 20);
    }

    #[test]
    fn pipeline_try_pipe_err() {
        let result = Pipeline::of(10)
            .try_pipe(|_| Err::<i32, &str>("oops"));
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_tap() {
        let log = Rc::new(RefCell::new(0));
        let log_ref = log.clone();
        let result = Pipeline::of(42)
            .tap(move |x| { *log_ref.borrow_mut() = *x; })
            .value();
        assert_eq!(result, 42);
        assert_eq!(*log.borrow(), 42);
    }

    #[test]
    fn pipeline_tap_mut() {
        let result = Pipeline::of(vec![1, 2])
            .tap_mut(|v| v.push(3))
            .value();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn pipeline_type_change() {
        let result = Pipeline::of(42)
            .pipe(|x| format!("num={x}"))
            .pipe(|s| s.len())
            .value();
        assert_eq!(result, 6); // "num=42" is 6 chars
    }

    #[test]
    fn pipeline_inspect() {
        let p = Pipeline::of(42);
        let doubled = p.inspect(|x| x * 2);
        assert_eq!(doubled, 84);
        assert_eq!(p.value(), 42);
    }

    #[test]
    fn pipeline_debug() {
        let p = Pipeline::of(42);
        assert!(format!("{p:?}").contains("42"));
    }

    #[test]
    fn pipeline_clone() {
        let p = Pipeline::of(42);
        let p2 = p.clone();
        assert_eq!(p.value(), p2.value());
    }

    #[test]
    fn compose_fn() {
        let double_then_inc = compose(|x: i32| x * 2, |x| x + 1);
        assert_eq!(double_then_inc(5), 11);
    }

    #[test]
    fn compose3_fn() {
        let f = compose3(|x: i32| x + 1, |x| x * 2, |x| x - 3);
        assert_eq!(f(5), 9); // (5+1)*2 - 3 = 9
    }

    #[test]
    fn compose_all_fn() {
        let fns: Vec<Box<dyn Fn(i32) -> i32>> = vec![
            Box::new(|x| x + 1),
            Box::new(|x| x * 2),
            Box::new(|x| x - 3),
        ];
        let f = compose_all(fns);
        assert_eq!(f(5), 9); // (5+1)*2 - 3 = 9
    }

    #[test]
    fn compose_r_fn() {
        let f = compose_r(|x: i32| x + 1, |x: i32| x * 2);
        assert_eq!(f(5), 11); // (5*2) + 1 = 11
    }

    #[test]
    fn partial1_fn() {
        let add = |a: i32, b: i32| a + b;
        let add5 = partial1(add, 5);
        assert_eq!(add5(3), 8);
    }

    #[test]
    fn partial2_fn() {
        let sub = |a: i32, b: i32| a - b;
        let sub3 = partial2(sub, 3);
        assert_eq!(sub3(10), 7); // 10 - 3
    }

    #[test]
    fn partial12_fn() {
        let f = |a: i32, b: i32, c: i32| a + b + c;
        let f12 = partial12(f, 1, 2);
        assert_eq!(f12(3), 6);
    }

    #[test]
    fn curry2_rc_fn() {
        let add: Rc<dyn Fn(i32, i32) -> i32> = Rc::new(|a, b| a + b);
        let curried = curry2_rc(add);
        let add5 = curried(5);
        assert_eq!(add5(3), 8);
    }

    #[test]
    fn curry3_rc_fn() {
        let f: Rc<dyn Fn(i32, i32, i32) -> i32> = Rc::new(|a, b, c| a + b + c);
        let curried = curry3_rc(f);
        assert_eq!(curried(1)(2)(3), 6);
    }

    #[test]
    fn uncurry2_fn() {
        let curried = |a: i32| -> Box<dyn Fn(i32) -> i32> { Box::new(move |b| a + b) };
        let uncurried = uncurry2(curried);
        assert_eq!(uncurried(3, 4), 7);
    }

    #[test]
    fn flip_fn() {
        let sub = |a: i32, b: i32| a - b;
        let flipped = flip(sub);
        assert_eq!(flipped(3, 10), 7); // 10 - 3
    }

    #[test]
    fn on_fn() {
        let compare_abs = on(
            |a: i32, b: i32| a.cmp(&b),
            |x: i32| x.abs(),
        );
        assert_eq!(compare_abs(-5, 5), std::cmp::Ordering::Equal);
        assert_eq!(compare_abs(-3, 5), std::cmp::Ordering::Less);
    }

    #[test]
    fn converge2_fn() {
        // average = sum / count
        let avg = converge2(
            |sum: i32, count: usize| sum as f64 / count as f64,
            |v: Vec<i32>| v.iter().sum::<i32>(),
            |v: Vec<i32>| v.len(),
        );
        let result = avg(vec![2, 4, 6]);
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn pipeline_eq() {
        let a = Pipeline::of(42);
        let b = Pipeline::of(42);
        assert_eq!(a, b);
    }
}
