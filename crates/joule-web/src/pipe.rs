//! Functional pipe and compose utilities.
//!
//! Provides a fluent, chainable transformation API inspired by
//! functional programming pipe/compose patterns:
//!
//! - [`pipe`] — apply a single function to a value
//! - [`Pipe`] — chainable wrapper for multi-step transforms
//! - [`compose2`] / [`compose`] — function composition
//! - [`Pipeline`] — dynamic pipeline over `serde_json::Value`
//! - [`tap`], [`when`], [`try_pipe`] — utility combinators

use serde_json::Value;

// ── Free functions ──────────────────────────────────────────────────

/// Apply a single function to a value: `pipe(x, f) == f(x)`.
pub fn pipe<T, U>(value: T, f: impl FnOnce(T) -> U) -> U {
    f(value)
}

/// Side-effect on a reference without consuming.
pub fn tap<T>(value: &T, f: impl FnOnce(&T)) {
    f(value);
}

/// Conditionally apply a transform: if `condition` is true, returns
/// `f(value)`; otherwise returns `value` unchanged.
pub fn when<T>(condition: bool, value: T, f: impl FnOnce(T) -> T) -> T {
    if condition { f(value) } else { value }
}

/// Pipe through a fallible function.
pub fn try_pipe<T, E>(value: T, f: impl FnOnce(T) -> Result<T, E>) -> Result<T, E> {
    f(value)
}

/// Compose two functions: `compose2(f, g)` yields `|a| g(f(a))`.
pub fn compose2<A, B, C>(
    f: impl Fn(A) -> B + 'static,
    g: impl Fn(B) -> C + 'static,
) -> impl Fn(A) -> C {
    move |a| g(f(a))
}

/// Compose a vector of functions into a single function, applied
/// left-to-right.
pub fn compose<T: 'static>(fns: Vec<Box<dyn Fn(T) -> T>>) -> Box<dyn Fn(T) -> T> {
    Box::new(move |mut val| {
        for f in &fns {
            val = f(val);
        }
        val
    })
}

// ── Pipe<T> wrapper ─────────────────────────────────────────────────

/// Chainable wrapper that threads a value through a series of transforms.
pub struct Pipe<T> {
    value: T,
}

impl<T> Pipe<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }

    /// Apply `f` and wrap the result in a new `Pipe`.
    pub fn pipe<U>(self, f: impl FnOnce(T) -> U) -> Pipe<U> {
        Pipe { value: f(self.value) }
    }

    /// Conditionally apply `f`; no-op when `condition` is false.
    pub fn pipe_if(self, condition: bool, f: impl FnOnce(T) -> T) -> Pipe<T> {
        if condition {
            Pipe { value: f(self.value) }
        } else {
            self
        }
    }

    /// Pipe through a fallible function.
    pub fn pipe_result<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<Pipe<U>, E> {
        f(self.value).map(|v| Pipe { value: v })
    }

    /// Unwrap the inner value.
    pub fn unwrap(self) -> T {
        self.value
    }
}

// ── Pipeline (dynamic, JSON-based) ──────────────────────────────────

/// A dynamic pipeline that processes `serde_json::Value` through a
/// sequence of registered steps.
pub struct Pipeline {
    steps: Vec<Box<dyn Fn(Value) -> Value>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a processing step.
    pub fn step(mut self, f: impl Fn(Value) -> Value + 'static) -> Self {
        self.steps.push(Box::new(f));
        self
    }

    /// Execute all steps in order, threading `input` through each.
    pub fn execute(&self, input: Value) -> Value {
        let mut val = input;
        for step in &self.steps {
            val = step(val);
        }
        val
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pipe_transforms_value() {
        let result = pipe(5, |x| x * 2);
        assert_eq!(result, 10);
    }

    #[test]
    fn pipe_chain() {
        let result = Pipe::new(3)
            .pipe(|x| x + 1)
            .pipe(|x| x * 10)
            .pipe(|x: i32| x.to_string())
            .unwrap();
        assert_eq!(result, "40");
    }

    #[test]
    fn pipe_if_conditional() {
        let yes = Pipe::new(10).pipe_if(true, |x| x + 5).unwrap();
        let no = Pipe::new(10).pipe_if(false, |x| x + 5).unwrap();
        assert_eq!(yes, 15);
        assert_eq!(no, 10);
    }

    #[test]
    fn pipe_result_ok_and_err() {
        let ok = Pipe::new(42)
            .pipe_result(|x| Ok::<_, &str>(x + 1))
            .unwrap()
            .unwrap();
        assert_eq!(ok, 43);

        let err = Pipe::new(0).pipe_result(|_| Err::<i32, _>("fail"));
        assert!(err.is_err());
    }

    #[test]
    fn compose2_chains_two_functions() {
        let add1 = |x: i32| x + 1;
        let double = |x: i32| x * 2;
        let f = compose2(add1, double);
        assert_eq!(f(5), 12); // (5+1)*2
    }

    #[test]
    fn compose_many() {
        let fns: Vec<Box<dyn Fn(i32) -> i32>> = vec![
            Box::new(|x| x + 1),
            Box::new(|x| x * 3),
            Box::new(|x| x - 2),
        ];
        let f = compose(fns);
        assert_eq!(f(4), 13); // ((4+1)*3)-2
    }

    #[test]
    fn pipeline_steps_json() {
        let p = Pipeline::new()
            .step(|v| {
                if let Value::Number(n) = v {
                    json!(n.as_i64().unwrap_or(0) + 10)
                } else {
                    v
                }
            })
            .step(|v| {
                if let Value::Number(n) = v {
                    json!(n.as_i64().unwrap_or(0) * 2)
                } else {
                    v
                }
            });
        let result = p.execute(json!(5));
        assert_eq!(result, json!(30)); // (5+10)*2
    }

    #[test]
    fn tap_does_not_consume() {
        let val = 42;
        let mut seen = 0;
        tap(&val, |v| seen = *v);
        assert_eq!(val, 42);
        assert_eq!(seen, 42);
    }

    #[test]
    fn when_applies_conditionally() {
        assert_eq!(when(true, 10, |x| x + 5), 15);
        assert_eq!(when(false, 10, |x| x + 5), 10);
    }

    #[test]
    fn try_pipe_propagates_error() {
        let ok = try_pipe(10, |x| Ok::<_, &str>(x * 2));
        assert_eq!(ok.unwrap(), 20);

        let err = try_pipe(10, |_| Err::<i32, _>("boom"));
        assert!(err.is_err());
    }
}
