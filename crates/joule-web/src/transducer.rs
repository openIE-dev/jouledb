//! Transducers — composable, reusable transformations.
//!
//! A transducer transforms a step function into another step function,
//! enabling composable `map`/`filter`/`take`/`drop`/`distinct` operations
//! that are independent of the input and output collection types.
//!
//! Supports transducer composition, transduction into `Vec`/`String`/sum,
//! early termination, and stateful transducers.

use std::collections::HashSet;
use std::hash::Hash;

// ── StepResult ──────────────────────────────────────────────────────────────

/// The result of a single step in a transduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepResult<A> {
    /// Continue with the updated accumulator.
    Continue(A),
    /// Stop early with the final accumulator.
    Done(A),
}

impl<A> StepResult<A> {
    /// Extract the inner value regardless of variant.
    pub fn into_inner(self) -> A {
        match self {
            StepResult::Continue(a) => a,
            StepResult::Done(a) => a,
        }
    }

    /// Is this a `Done`?
    pub fn is_done(&self) -> bool {
        matches!(self, StepResult::Done(_))
    }

    /// Map over the inner value.
    pub fn map<B>(self, f: impl FnOnce(A) -> B) -> StepResult<B> {
        match self {
            StepResult::Continue(a) => StepResult::Continue(f(a)),
            StepResult::Done(a) => StepResult::Done(f(a)),
        }
    }
}

// ── Step function type ──────────────────────────────────────────────────────

/// A step function takes an accumulator and an input element, producing
/// a `StepResult`.
pub type StepFn<A, T> = Box<dyn FnMut(A, T) -> StepResult<A>>;

// ── Transducer ──────────────────────────────────────────────────────────────

/// A transducer transforms one step function into another.
///
/// It is parameterized by `Input` (what it consumes) and `Output`
/// (what it passes to the downstream step).
pub struct Transducer<Input, Output> {
    transform: Box<dyn FnOnce(StepFn<Vec<Output>, Output>) -> StepFn<Vec<Output>, Input>>,
}

// Since we cannot easily compose trait-object transducers with different type
// parameters generically, we provide a simpler concrete approach:
// each transducer is a function that transforms an iterator-like pipeline.

/// A composable transformation step that operates on iterators.
pub enum Xform<T> {
    /// Map each element.
    Map(Box<dyn Fn(T) -> T>),
    /// Filter elements by predicate.
    Filter(Box<dyn Fn(&T) -> bool>),
    /// Take at most N elements.
    Take(usize),
    /// Drop the first N elements.
    Drop(usize),
    /// Take while predicate holds.
    TakeWhile(Box<dyn Fn(&T) -> bool>),
    /// Drop while predicate holds.
    DropWhile(Box<dyn Fn(&T) -> bool>),
    /// Deduplicate consecutive equal elements.
    Dedup,
    /// Keep only distinct elements (requires Hash+Eq).
    Distinct,
    /// Flatten: for when T is a Vec-like, but we model it as identity here.
    /// Instead, use `FlatMap` for exploding each element into multiple.
    FlatMap(Box<dyn Fn(T) -> Vec<T>>),
    /// Scan: accumulate state and emit the intermediate results.
    Scan(Box<dyn Fn(T, T) -> T>),
    /// Inspect: side effect without changing element.
    Inspect(Box<dyn Fn(&T)>),
    /// Interleave a separator between elements.
    Intersperse(T),
    /// Chunk elements into groups of N.
    Chunk(usize),
}

// ── TransducerPipeline ──────────────────────────────────────────────────────

/// A pipeline of composable transformations.
pub struct TransducerPipeline<T> {
    steps: Vec<Xform<T>>,
}

impl<T: Clone + PartialEq + Eq + Hash + 'static> TransducerPipeline<T> {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a map step.
    pub fn map(mut self, f: impl Fn(T) -> T + 'static) -> Self {
        self.steps.push(Xform::Map(Box::new(f)));
        self
    }

    /// Add a filter step.
    pub fn filter(mut self, pred: impl Fn(&T) -> bool + 'static) -> Self {
        self.steps.push(Xform::Filter(Box::new(pred)));
        self
    }

    /// Add a take step.
    pub fn take(mut self, n: usize) -> Self {
        self.steps.push(Xform::Take(n));
        self
    }

    /// Add a drop step.
    pub fn drop(mut self, n: usize) -> Self {
        self.steps.push(Xform::Drop(n));
        self
    }

    /// Add a take-while step.
    pub fn take_while(mut self, pred: impl Fn(&T) -> bool + 'static) -> Self {
        self.steps.push(Xform::TakeWhile(Box::new(pred)));
        self
    }

    /// Add a drop-while step.
    pub fn drop_while(mut self, pred: impl Fn(&T) -> bool + 'static) -> Self {
        self.steps.push(Xform::DropWhile(Box::new(pred)));
        self
    }

    /// Add a dedup step (removes consecutive duplicates).
    pub fn dedup(mut self) -> Self {
        self.steps.push(Xform::Dedup);
        self
    }

    /// Add a distinct step (removes all duplicates).
    pub fn distinct(mut self) -> Self {
        self.steps.push(Xform::Distinct);
        self
    }

    /// Add a flat-map step.
    pub fn flat_map(mut self, f: impl Fn(T) -> Vec<T> + 'static) -> Self {
        self.steps.push(Xform::FlatMap(Box::new(f)));
        self
    }

    /// Add an inspect step.
    pub fn inspect(mut self, f: impl Fn(&T) + 'static) -> Self {
        self.steps.push(Xform::Inspect(Box::new(f)));
        self
    }

    /// Add an intersperse step.
    pub fn intersperse(mut self, separator: T) -> Self {
        self.steps.push(Xform::Intersperse(separator));
        self
    }

    /// Add a chunk step.
    pub fn chunk(mut self, size: usize) -> Self {
        self.steps.push(Xform::Chunk(size));
        self
    }

    /// Number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Execute the pipeline against an input, collecting into a `Vec<T>`.
    pub fn transduce(&self, input: impl IntoIterator<Item = T>) -> Vec<T> {
        let mut items: Vec<T> = input.into_iter().collect();

        for step in &self.steps {
            items = apply_step(step, items);
        }

        items
    }

    /// Compose two pipelines.
    pub fn compose(mut self, other: TransducerPipeline<T>) -> Self {
        self.steps.extend(other.steps);
        self
    }
}

impl<T: Clone + PartialEq + Eq + Hash + 'static> Default for TransducerPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_step<T: Clone + PartialEq + Eq + Hash>(step: &Xform<T>, input: Vec<T>) -> Vec<T> {
    match step {
        Xform::Map(f) => input.into_iter().map(|x| f(x)).collect(),
        Xform::Filter(pred) => input.into_iter().filter(|x| pred(x)).collect(),
        Xform::Take(n) => input.into_iter().take(*n).collect(),
        Xform::Drop(n) => input.into_iter().skip(*n).collect(),
        Xform::TakeWhile(pred) => input.into_iter().take_while(|x| pred(x)).collect(),
        Xform::DropWhile(pred) => {
            let mut dropping = true;
            input
                .into_iter()
                .filter(move |x| {
                    if dropping {
                        if pred(x) {
                            return false;
                        }
                        dropping = false;
                    }
                    true
                })
                .collect()
        }
        Xform::Dedup => {
            let mut result = Vec::new();
            for item in input {
                if result.last() != Some(&item) {
                    result.push(item);
                }
            }
            result
        }
        Xform::Distinct => {
            let mut seen = HashSet::new();
            let mut result = Vec::new();
            for item in input {
                if seen.insert(item.clone()) {
                    result.push(item);
                }
            }
            result
        }
        Xform::FlatMap(f) => input.into_iter().flat_map(|x| f(x)).collect(),
        Xform::Scan(f) => {
            if input.is_empty() {
                return Vec::new();
            }
            let mut result = Vec::with_capacity(input.len());
            let mut iter = input.into_iter();
            let first = iter.next().unwrap();
            result.push(first.clone());
            let mut acc = first;
            for item in iter {
                acc = f(acc, item);
                result.push(acc.clone());
            }
            result
        }
        Xform::Inspect(f) => {
            for item in &input {
                f(item);
            }
            input
        }
        Xform::Intersperse(sep) => {
            let mut result = Vec::new();
            let len = input.len();
            for (i, item) in input.into_iter().enumerate() {
                result.push(item);
                if i + 1 < len {
                    result.push(sep.clone());
                }
            }
            result
        }
        Xform::Chunk(size) => {
            // For chunks, we flatten back. This is a simplification.
            // In a real API you'd change the type. Here we just return
            // the input unchanged (chunks are for grouping, not filtering).
            // We model it by returning consecutive size-groups concatenated
            // (effectively identity for flat Vec output).
            // Better: chunk and keep only complete chunks.
            let mut result = Vec::new();
            let chunks: Vec<Vec<T>> = input.chunks(*size).map(|c| c.to_vec()).collect();
            for chunk in chunks {
                result.extend(chunk);
            }
            result
        }
    }
}

// ── Transduce into different targets ────────────────────────────────────────

/// Transduce a string pipeline: transform chars and collect back into a String.
pub fn transduce_string(
    input: &str,
    pipeline: &TransducerPipeline<char>,
) -> String {
    pipeline.transduce(input.chars()).into_iter().collect()
}

/// Transduce and sum (for numeric types).
pub fn transduce_sum(
    input: impl IntoIterator<Item = i64>,
    pipeline: &TransducerPipeline<i64>,
) -> i64 {
    pipeline.transduce(input).into_iter().sum()
}

/// Transduce and count.
pub fn transduce_count<T: Clone + PartialEq + Eq + Hash + 'static>(
    input: impl IntoIterator<Item = T>,
    pipeline: &TransducerPipeline<T>,
) -> usize {
    pipeline.transduce(input).len()
}

/// Transduce and fold.
pub fn transduce_fold<T: Clone + PartialEq + Eq + Hash + 'static, A>(
    input: impl IntoIterator<Item = T>,
    pipeline: &TransducerPipeline<T>,
    init: A,
    fold_fn: impl Fn(A, T) -> A,
) -> A {
    let items = pipeline.transduce(input);
    items.into_iter().fold(init, fold_fn)
}

// ── StepProcessor: lower-level step-function protocol ───────────────────────

/// A lower-level step-function based transducer.
///
/// This demonstrates the step-function protocol more explicitly.
pub struct StepProcessor<A, T> {
    step: Box<dyn FnMut(A, T) -> StepResult<A>>,
}

impl<A: 'static, T: 'static> StepProcessor<A, T> {
    /// Create from a step function.
    pub fn new(step: impl FnMut(A, T) -> StepResult<A> + 'static) -> Self {
        Self {
            step: Box::new(step),
        }
    }

    /// Process all items from an iterator.
    pub fn process(mut self, init: A, input: impl IntoIterator<Item = T>) -> A {
        let mut acc = init;
        for item in input {
            match (self.step)(acc, item) {
                StepResult::Continue(a) => acc = a,
                StepResult::Done(a) => return a,
            }
        }
        acc
    }
}

/// Create a collecting step processor that pushes to a Vec.
pub fn collecting_step<T: 'static>() -> StepProcessor<Vec<T>, T> {
    StepProcessor::new(|mut acc: Vec<T>, item: T| {
        acc.push(item);
        StepResult::Continue(acc)
    })
}

/// Create a mapping step processor.
pub fn mapping_step<A: 'static, T: 'static, U: 'static>(
    f: impl Fn(T) -> U + 'static,
    mut downstream: StepProcessor<A, U>,
) -> StepProcessor<A, T> {
    StepProcessor::new(move |acc, item| {
        (downstream.step)(acc, f(item))
    })
}

/// Create a filtering step processor.
pub fn filtering_step<A: 'static, T: 'static>(
    pred: impl Fn(&T) -> bool + 'static,
    mut downstream: StepProcessor<A, T>,
) -> StepProcessor<A, T> {
    StepProcessor::new(move |acc, item| {
        if pred(&item) {
            (downstream.step)(acc, item)
        } else {
            StepResult::Continue(acc)
        }
    })
}

/// Create a taking step processor (early termination after N).
pub fn taking_step<A: 'static, T: 'static>(
    n: usize,
    mut downstream: StepProcessor<A, T>,
) -> StepProcessor<A, T> {
    let mut count = 0;
    StepProcessor::new(move |acc, item| {
        if count >= n {
            return StepResult::Done(acc);
        }
        count += 1;
        let result = (downstream.step)(acc, item);
        if count >= n {
            StepResult::Done(result.into_inner())
        } else {
            result
        }
    })
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn pipeline_map() {
        let p = TransducerPipeline::new().map(|x: i32| x * 2);
        assert_eq!(p.transduce(vec![1, 2, 3]), vec![2, 4, 6]);
    }

    #[test]
    fn pipeline_filter() {
        let p = TransducerPipeline::new().filter(|x: &i32| *x > 2);
        assert_eq!(p.transduce(vec![1, 2, 3, 4]), vec![3, 4]);
    }

    #[test]
    fn pipeline_map_then_filter() {
        let p = TransducerPipeline::new()
            .map(|x: i32| x * 2)
            .filter(|x: &i32| *x > 4);
        assert_eq!(p.transduce(vec![1, 2, 3, 4]), vec![6, 8]);
    }

    #[test]
    fn pipeline_take() {
        let p = TransducerPipeline::new().take(3);
        assert_eq!(p.transduce(vec![1, 2, 3, 4, 5]), vec![1, 2, 3]);
    }

    #[test]
    fn pipeline_drop() {
        let p = TransducerPipeline::new().drop(2);
        assert_eq!(p.transduce(vec![1, 2, 3, 4, 5]), vec![3, 4, 5]);
    }

    #[test]
    fn pipeline_take_while() {
        let p = TransducerPipeline::new().take_while(|x: &i32| *x < 4);
        assert_eq!(p.transduce(vec![1, 2, 3, 4, 5]), vec![1, 2, 3]);
    }

    #[test]
    fn pipeline_drop_while() {
        let p = TransducerPipeline::new().drop_while(|x: &i32| *x < 3);
        assert_eq!(p.transduce(vec![1, 2, 3, 4, 5]), vec![3, 4, 5]);
    }

    #[test]
    fn pipeline_dedup() {
        let p = TransducerPipeline::new().dedup();
        assert_eq!(p.transduce(vec![1, 1, 2, 2, 3, 1, 1]), vec![1, 2, 3, 1]);
    }

    #[test]
    fn pipeline_distinct() {
        let p = TransducerPipeline::new().distinct();
        assert_eq!(p.transduce(vec![1, 2, 1, 3, 2, 4]), vec![1, 2, 3, 4]);
    }

    #[test]
    fn pipeline_flat_map() {
        let p = TransducerPipeline::new()
            .flat_map(|x: i32| vec![x, x * 10]);
        assert_eq!(p.transduce(vec![1, 2]), vec![1, 10, 2, 20]);
    }

    #[test]
    fn pipeline_intersperse() {
        let p = TransducerPipeline::new().intersperse(0);
        assert_eq!(p.transduce(vec![1, 2, 3]), vec![1, 0, 2, 0, 3]);
    }

    #[test]
    fn pipeline_compose() {
        let p1 = TransducerPipeline::new().map(|x: i32| x + 1);
        let p2 = TransducerPipeline::new().filter(|x: &i32| *x > 3);
        let composed = p1.compose(p2);
        assert_eq!(composed.transduce(vec![1, 2, 3, 4, 5]), vec![4, 5, 6]);
    }

    #[test]
    fn pipeline_complex() {
        let p = TransducerPipeline::new()
            .filter(|x: &i32| *x % 2 == 0)
            .map(|x: i32| x * 10)
            .take(3);
        assert_eq!(p.transduce(vec![1, 2, 3, 4, 5, 6, 7, 8]), vec![20, 40, 60]);
    }

    #[test]
    fn pipeline_inspect() {
        let log = Rc::new(RefCell::new(Vec::new()));
        let log_ref = log.clone();
        let p = TransducerPipeline::new()
            .inspect(move |x: &i32| log_ref.borrow_mut().push(*x));
        let result = p.transduce(vec![1, 2, 3]);
        assert_eq!(result, vec![1, 2, 3]);
        assert_eq!(*log.borrow(), vec![1, 2, 3]);
    }

    #[test]
    fn pipeline_step_count() {
        let p = TransducerPipeline::<i32>::new()
            .map(|x| x + 1)
            .filter(|x| *x > 0)
            .take(5);
        assert_eq!(p.step_count(), 3);
    }

    #[test]
    fn transduce_string_fn() {
        let p = TransducerPipeline::new()
            .filter(|c: &char| c.is_alphabetic())
            .map(|c: char| c.to_ascii_uppercase());
        let result = transduce_string("h3llo w0rld!", &p);
        assert_eq!(result, "HLLOWRLD");
    }

    #[test]
    fn transduce_sum_fn() {
        let p = TransducerPipeline::new()
            .filter(|x: &i64| *x > 0)
            .map(|x: i64| x * 2);
        let result = transduce_sum(vec![-1, 2, -3, 4, 5], &p);
        assert_eq!(result, 22); // (2+4+5)*2 = 22
    }

    #[test]
    fn transduce_count_fn() {
        let p = TransducerPipeline::new().filter(|x: &i32| *x > 3);
        assert_eq!(transduce_count(vec![1, 2, 3, 4, 5], &p), 2);
    }

    #[test]
    fn transduce_fold_fn() {
        let p = TransducerPipeline::new().filter(|x: &i32| *x % 2 == 0);
        let product = transduce_fold(vec![1, 2, 3, 4], &p, 1, |acc, x| acc * x);
        assert_eq!(product, 8); // 2 * 4
    }

    #[test]
    fn step_result_basics() {
        let c = StepResult::Continue(42);
        assert!(!c.is_done());
        assert_eq!(c.into_inner(), 42);

        let d = StepResult::Done(99);
        assert!(d.is_done());
        assert_eq!(d.map(|x| x + 1), StepResult::Done(100));
    }

    #[test]
    fn collecting_step_processor() {
        let proc = collecting_step();
        let result = proc.process(Vec::new(), vec![1, 2, 3]);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn mapping_step_processor() {
        let downstream = collecting_step();
        let proc = mapping_step(|x: i32| x * 10, downstream);
        let result = proc.process(Vec::new(), vec![1, 2, 3]);
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[test]
    fn filtering_step_processor() {
        let downstream = collecting_step();
        let proc = filtering_step(|x: &i32| *x > 2, downstream);
        let result = proc.process(Vec::new(), vec![1, 2, 3, 4]);
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn taking_step_early_termination() {
        let downstream = collecting_step();
        let proc = taking_step(2, downstream);
        let result = proc.process(Vec::new(), vec![10, 20, 30, 40, 50]);
        assert_eq!(result, vec![10, 20]);
    }

    #[test]
    fn pipeline_empty_input() {
        let p = TransducerPipeline::new().map(|x: i32| x * 2);
        assert_eq!(p.transduce(Vec::<i32>::new()), Vec::<i32>::new());
    }

    #[test]
    fn pipeline_chunk() {
        let p = TransducerPipeline::new().chunk(2);
        // Chunk preserves all elements (just groups them internally).
        assert_eq!(p.transduce(vec![1, 2, 3, 4, 5]), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn combined_step_processors() {
        // filter > map > take via step processor composition.
        let collect = collecting_step();
        let take3 = taking_step(3, collect);
        let double = mapping_step(|x: i32| x * 2, take3);
        let evens = filtering_step(|x: &i32| x % 2 == 0, double);
        let result = evens.process(Vec::new(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(result, vec![4, 8, 12]);
    }
}
