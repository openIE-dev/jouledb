//! Lazy evaluation primitives.
//!
//! Provides `Lazy<T>` (deferred computation with memoization),
//! `LazyStream<T>` (infinite lazy sequences with `take`/`filter`/`map`),
//! thunk chaining, forced evaluation, and lazy trees.

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

// ── Lazy<T> ─────────────────────────────────────────────────────────────────

/// A lazily evaluated, memoized value.
///
/// The computation runs at most once, on the first call to `force()`.
/// Subsequent calls return the cached result.
pub struct Lazy<T> {
    inner: RefCell<LazyState<T>>,
}

enum LazyState<T> {
    Deferred(Box<dyn FnOnce() -> T>),
    Evaluated(T),
    /// Temporary state during evaluation to avoid double-borrow.
    Evaluating,
}

impl<T> Lazy<T> {
    /// Create a new lazy value from a computation.
    pub fn new(f: impl FnOnce() -> T + 'static) -> Self {
        Self {
            inner: RefCell::new(LazyState::Deferred(Box::new(f))),
        }
    }

    /// Create an already-evaluated lazy value.
    pub fn from_value(value: T) -> Self {
        Self {
            inner: RefCell::new(LazyState::Evaluated(value)),
        }
    }

    /// Force evaluation, returning a reference to the value.
    ///
    /// The computation runs only on the first call; subsequent calls
    /// return the cached result.
    pub fn force(&self) -> std::cell::Ref<'_, T> {
        // If not yet evaluated, evaluate now.
        {
            let needs_eval = matches!(*self.inner.borrow(), LazyState::Deferred(_));
            if needs_eval {
                let thunk = {
                    let mut state = self.inner.borrow_mut();
                    match std::mem::replace(&mut *state, LazyState::Evaluating) {
                        LazyState::Deferred(f) => f,
                        _ => unreachable!(),
                    }
                };
                let value = thunk();
                *self.inner.borrow_mut() = LazyState::Evaluated(value);
            }
        }
        std::cell::Ref::map(self.inner.borrow(), |state| match state {
            LazyState::Evaluated(v) => v,
            _ => unreachable!(),
        })
    }

    /// Check if the value has been evaluated.
    pub fn is_evaluated(&self) -> bool {
        matches!(*self.inner.borrow(), LazyState::Evaluated(_))
    }
}

impl<T: Clone> Lazy<T> {
    /// Force and clone the value.
    pub fn get(&self) -> T {
        self.force().clone()
    }

    /// Map a function over the lazy value, producing a new lazy value.
    pub fn map<U: 'static>(self, f: impl FnOnce(T) -> U + 'static) -> Lazy<U>
    where
        T: 'static,
    {
        Lazy::new(move || f(self.get()))
    }

    /// Monadic flat_map.
    pub fn flat_map<U: Clone + 'static>(
        self,
        f: impl FnOnce(T) -> Lazy<U> + 'static,
    ) -> Lazy<U>
    where
        T: 'static,
    {
        Lazy::new(move || f(self.get()).get())
    }
}

impl<T: fmt::Debug> fmt::Debug for Lazy<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.inner.borrow() {
            LazyState::Evaluated(v) => write!(f, "Lazy({v:?})"),
            LazyState::Deferred(_) => write!(f, "Lazy(<deferred>)"),
            LazyState::Evaluating => write!(f, "Lazy(<evaluating>)"),
        }
    }
}

// ── Thunk ───────────────────────────────────────────────────────────────────

/// A chainable thunk: a deferred computation that can be chained
/// before being forced.
pub struct Thunk<T> {
    computation: Box<dyn FnOnce() -> T>,
}

impl<T: 'static> Thunk<T> {
    /// Create a thunk from a closure.
    pub fn new(f: impl FnOnce() -> T + 'static) -> Self {
        Self {
            computation: Box::new(f),
        }
    }

    /// Create a thunk that immediately returns a value.
    pub fn value(v: T) -> Self {
        Self {
            computation: Box::new(move || v),
        }
    }

    /// Chain: apply `f` to the result when forced.
    pub fn chain<U: 'static>(self, f: impl FnOnce(T) -> U + 'static) -> Thunk<U> {
        Thunk {
            computation: Box::new(move || f((self.computation)())),
        }
    }

    /// Force the thunk, running the computation.
    pub fn force(self) -> T {
        (self.computation)()
    }

    /// Flat-chain: the next thunk depends on the current value.
    pub fn flat_chain<U: 'static>(self, f: impl FnOnce(T) -> Thunk<U> + 'static) -> Thunk<U> {
        Thunk {
            computation: Box::new(move || f((self.computation)()).force()),
        }
    }
}

// ── LazyStream ──────────────────────────────────────────────────────────────

/// A lazily computed, potentially infinite stream of values.
///
/// Elements are generated on demand from a state + step function.
pub struct LazyStream<T> {
    gen_fn: Rc<dyn Fn(usize) -> Option<T>>,
}

impl<T: 'static> LazyStream<T> {
    /// Create a stream from a generator function.
    /// The function receives the zero-based index and returns `None` to end.
    pub fn from_fn(f: impl Fn(usize) -> Option<T> + 'static) -> Self {
        Self {
            gen_fn: Rc::new(f),
        }
    }

    /// Create an infinite stream from a generator that never returns `None`.
    pub fn infinite(f: impl Fn(usize) -> T + 'static) -> Self {
        Self {
            gen_fn: Rc::new(move |i| Some(f(i))),
        }
    }

    /// Create a stream from an iterator (finite). Requires `Clone`.
    pub fn from_iter(iter: impl IntoIterator<Item = T> + 'static) -> Self
    where
        T: Clone,
    {
        let items: Vec<T> = iter.into_iter().collect();
        let items = Rc::new(items);
        Self {
            gen_fn: Rc::new(move |i| items.get(i).cloned()),
        }
    }

    /// Create a stream that repeats a single value infinitely.
    pub fn repeat(value: T) -> Self
    where
        T: Clone,
    {
        Self {
            gen_fn: Rc::new(move |_| Some(value.clone())),
        }
    }

    /// Get the element at the given index.
    pub fn get(&self, index: usize) -> Option<T> {
        (self.gen_fn)(index)
    }

    /// Take the first `n` elements.
    pub fn take(&self, n: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            match (self.gen_fn)(i) {
                Some(v) => result.push(v),
                None => break,
            }
        }
        result
    }

    /// Map a function over the stream.
    pub fn map<U: 'static>(self, f: impl Fn(T) -> U + 'static) -> LazyStream<U> {
        let generator = self.gen_fn;
        LazyStream {
            gen_fn: Rc::new(move |i| generator(i).map(|v| f(v))),
        }
    }

    /// Take elements while predicate holds.
    pub fn take_while(&self, pred: impl Fn(&T) -> bool) -> Vec<T> {
        let mut result = Vec::new();
        let mut i = 0;
        loop {
            match (self.gen_fn)(i) {
                Some(v) => {
                    if pred(&v) {
                        result.push(v);
                        i += 1;
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        result
    }

    /// Drop the first `n` elements and return the rest (up to `limit`).
    pub fn drop_take(&self, drop_count: usize, take_count: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(take_count);
        for i in 0..take_count {
            match (self.gen_fn)(drop_count + i) {
                Some(v) => result.push(v),
                None => break,
            }
        }
        result
    }

    /// Fold up to `n` elements.
    pub fn fold<A>(&self, n: usize, init: A, f: impl Fn(A, T) -> A) -> A {
        let mut acc = init;
        for i in 0..n {
            match (self.gen_fn)(i) {
                Some(v) => acc = f(acc, v),
                None => break,
            }
        }
        acc
    }

    /// Zip two streams.
    pub fn zip<U: 'static>(self, other: LazyStream<U>) -> LazyStream<(T, U)> {
        let g1 = self.gen_fn;
        let g2 = other.gen_fn;
        LazyStream {
            gen_fn: Rc::new(move |i| {
                let a = g1(i)?;
                let b = g2(i)?;
                Some((a, b))
            }),
        }
    }

    /// Find the first element satisfying a predicate (up to `limit` elements).
    pub fn find(&self, limit: usize, pred: impl Fn(&T) -> bool) -> Option<T> {
        for i in 0..limit {
            match (self.gen_fn)(i) {
                Some(v) if pred(&v) => return Some(v),
                Some(_) => continue,
                None => break,
            }
        }
        None
    }
}

/// Create a stream of natural numbers starting from `start`.
pub fn naturals(start: usize) -> LazyStream<usize> {
    LazyStream::infinite(move |i| start + i)
}

// A separate filtered stream that adjusts indices properly.
/// Create a filtered lazy stream. Since filtering changes index mapping,
/// this eagerly scans up to `scan_limit` source elements.
pub fn lazy_filter<T: Clone + 'static>(
    source: &LazyStream<T>,
    pred: impl Fn(&T) -> bool + 'static,
    scan_limit: usize,
) -> LazyStream<T> {
    let mut filtered = Vec::new();
    for i in 0..scan_limit {
        match source.get(i) {
            Some(v) if pred(&v) => filtered.push(v),
            Some(_) => {}
            None => break,
        }
    }
    let filtered = Rc::new(filtered);
    LazyStream {
        gen_fn: Rc::new(move |i| filtered.get(i).cloned()),
    }
}

// ── LazyTree ────────────────────────────────────────────────────────────────

/// A lazily constructed tree. Children are computed on demand.
pub struct LazyTree<T> {
    pub value: T,
    children_fn: Box<dyn Fn(&T) -> Vec<LazyTree<T>>>,
    children_cache: RefCell<Option<Vec<LazyTree<T>>>>,
}

impl<T: Clone + 'static> LazyTree<T> {
    /// Create a lazy tree node.
    pub fn new(value: T, children_fn: impl Fn(&T) -> Vec<LazyTree<T>> + 'static) -> Self {
        Self {
            value,
            children_fn: Box::new(children_fn),
            children_cache: RefCell::new(None),
        }
    }

    /// Create a leaf node (no children).
    pub fn leaf(value: T) -> Self {
        Self {
            value,
            children_fn: Box::new(|_| Vec::new()),
            children_cache: RefCell::new(Some(Vec::new())),
        }
    }

    /// Force the children.
    fn ensure_children(&self) {
        let needs = self.children_cache.borrow().is_none();
        if needs {
            let children = (self.children_fn)(&self.value);
            *self.children_cache.borrow_mut() = Some(children);
        }
    }

    /// Number of direct children.
    pub fn child_count(&self) -> usize {
        self.ensure_children();
        self.children_cache.borrow().as_ref().unwrap().len()
    }

    /// Is this a leaf?
    pub fn is_leaf(&self) -> bool {
        self.child_count() == 0
    }

    /// Depth-first collect values up to a depth limit.
    pub fn collect_dfs(&self, max_depth: usize) -> Vec<T> {
        let mut result = Vec::new();
        self.dfs_helper(0, max_depth, &mut result);
        result
    }

    fn dfs_helper(&self, depth: usize, max_depth: usize, out: &mut Vec<T>) {
        out.push(self.value.clone());
        if depth < max_depth {
            self.ensure_children();
            let children = self.children_cache.borrow();
            for child in children.as_ref().unwrap() {
                child.dfs_helper(depth + 1, max_depth, out);
            }
        }
    }

    /// Breadth-first collect values up to a max count.
    ///
    /// Implemented via level-by-level DFS to avoid borrow lifetime issues
    /// with `RefCell`-cached children.
    pub fn collect_bfs(&self, max_count: usize) -> Vec<T> {
        let mut result = Vec::new();
        // Collect the root level first.
        result.push(self.value.clone());
        if result.len() >= max_count {
            return result;
        }
        // Then expand children level-by-level using DFS to gather each level.
        self.bfs_level(&mut result, max_count);
        result
    }

    fn bfs_level(&self, result: &mut Vec<T>, max_count: usize) {
        self.ensure_children();
        let children = self.children_cache.borrow();
        let children_ref = children.as_ref().unwrap();
        // First pass: add all child values.
        for child in children_ref {
            if result.len() >= max_count {
                return;
            }
            result.push(child.value.clone());
        }
        // Second pass: recurse into each child's children.
        for child in children_ref {
            if result.len() >= max_count {
                return;
            }
            child.bfs_level(result, max_count);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for LazyTree<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LazyTree({:?})", self.value)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_deferred_until_force() {
        let lazy = Lazy::new(|| 42);
        assert!(!lazy.is_evaluated());
        assert_eq!(*lazy.force(), 42);
        assert!(lazy.is_evaluated());
    }

    #[test]
    fn lazy_memoizes() {
        let counter = Rc::new(RefCell::new(0));
        let c = counter.clone();
        let lazy = Lazy::new(move || {
            *c.borrow_mut() += 1;
            100
        });
        assert_eq!(*lazy.force(), 100);
        assert_eq!(*lazy.force(), 100);
        assert_eq!(*counter.borrow(), 1); // computed only once
    }

    #[test]
    fn lazy_from_value() {
        let lazy = Lazy::from_value(99);
        assert!(lazy.is_evaluated());
        assert_eq!(*lazy.force(), 99);
    }

    #[test]
    fn lazy_get_clones() {
        let lazy = Lazy::new(|| "hello".to_string());
        assert_eq!(lazy.get(), "hello".to_string());
    }

    #[test]
    fn lazy_map() {
        let lazy = Lazy::new(|| 10);
        let mapped = lazy.map(|x| x * 3);
        assert_eq!(mapped.get(), 30);
    }

    #[test]
    fn lazy_flat_map() {
        let lazy = Lazy::new(|| 5);
        let chained = lazy.flat_map(|x| Lazy::new(move || x + 100));
        assert_eq!(chained.get(), 105);
    }

    #[test]
    fn lazy_debug() {
        let lazy = Lazy::new(|| 42);
        assert!(format!("{lazy:?}").contains("deferred"));
        let _ = lazy.force();
        assert!(format!("{lazy:?}").contains("42"));
    }

    #[test]
    fn thunk_value_and_force() {
        let t = Thunk::value(42);
        assert_eq!(t.force(), 42);
    }

    #[test]
    fn thunk_chain() {
        let result = Thunk::value(10)
            .chain(|x| x * 2)
            .chain(|x| x + 1)
            .force();
        assert_eq!(result, 21);
    }

    #[test]
    fn thunk_flat_chain() {
        let result = Thunk::value(5)
            .flat_chain(|x| Thunk::value(x * 10))
            .force();
        assert_eq!(result, 50);
    }

    #[test]
    fn stream_naturals() {
        let s = naturals(0);
        assert_eq!(s.take(5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn stream_get_single() {
        let s = naturals(1);
        assert_eq!(s.get(0), Some(1));
        assert_eq!(s.get(9), Some(10));
    }

    #[test]
    fn stream_infinite() {
        let s = LazyStream::infinite(|i| i * i);
        assert_eq!(s.take(4), vec![0, 1, 4, 9]);
    }

    #[test]
    fn stream_map() {
        let s = naturals(0).map(|x| x * 10);
        assert_eq!(s.take(3), vec![0, 10, 20]);
    }

    #[test]
    fn stream_take_while() {
        let s = naturals(0);
        assert_eq!(s.take_while(|x| *x < 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn stream_drop_take() {
        let s = naturals(0);
        assert_eq!(s.drop_take(3, 3), vec![3, 4, 5]);
    }

    #[test]
    fn stream_fold() {
        let s = naturals(1);
        let sum = s.fold(5, 0, |acc, x| acc + x);
        assert_eq!(sum, 15); // 1+2+3+4+5
    }

    #[test]
    fn stream_zip() {
        let a = naturals(0);
        let b = LazyStream::infinite(|i| (i as f64) * 0.5);
        let zipped = a.zip(b);
        assert_eq!(zipped.take(3), vec![(0, 0.0), (1, 0.5), (2, 1.0)]);
    }

    #[test]
    fn stream_find() {
        let s = naturals(0);
        assert_eq!(s.find(100, |x| *x > 10 && *x % 7 == 0), Some(14));
    }

    #[test]
    fn stream_repeat() {
        let s = LazyStream::repeat(42);
        assert_eq!(s.take(3), vec![42, 42, 42]);
    }

    #[test]
    fn lazy_filter_stream() {
        let s = naturals(0);
        let evens = lazy_filter(&s, |x| x % 2 == 0, 20);
        assert_eq!(evens.take(5), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn lazy_tree_leaf() {
        let t = LazyTree::leaf(42);
        assert!(t.is_leaf());
        assert_eq!(t.child_count(), 0);
    }

    #[test]
    fn lazy_tree_children() {
        let t = LazyTree::new(1, |v| {
            if *v >= 3 {
                vec![]
            } else {
                vec![LazyTree::leaf(v * 2), LazyTree::leaf(v * 2 + 1)]
            }
        });
        assert_eq!(t.child_count(), 2);
    }

    #[test]
    fn lazy_tree_dfs() {
        let t = LazyTree::new(1, |v| {
            if *v > 3 {
                vec![]
            } else {
                vec![LazyTree::leaf(v * 10)]
            }
        });
        let vals = t.collect_dfs(2);
        assert_eq!(vals, vec![1, 10]);
    }

    #[test]
    fn lazy_tree_bfs() {
        let t = LazyTree::new(1, |v| {
            if *v >= 4 {
                vec![]
            } else {
                vec![
                    LazyTree::leaf(v * 2),
                    LazyTree::leaf(v * 2 + 1),
                ]
            }
        });
        let vals = t.collect_bfs(10);
        // Root=1, children=2,3, grandchildren would be from 2 and 3
        // but 2 < 4 so leaf, 3 < 4 so leaf. Wait, we created them as leaf.
        // Actually the children_fn is only on the root.
        // The children are leaves, so: [1, 2, 3]
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn lazy_tree_debug() {
        let t = LazyTree::leaf(42);
        assert!(format!("{t:?}").contains("42"));
    }

    #[test]
    fn thunk_new() {
        let t = Thunk::new(|| 7 * 6);
        assert_eq!(t.force(), 42);
    }
}
