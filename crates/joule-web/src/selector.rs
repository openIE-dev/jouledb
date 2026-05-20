//! Memoized selectors (Reselect-style) for deriving state with caching,
//! composition, hit/miss statistics, and parameterized selector factories.

use std::collections::HashMap;

// ── Selector ──

/// A memoized selector that caches its output until the input changes.
pub struct Selector<Input, Output> {
    transform: Box<dyn Fn(&Input) -> Output>,
    cached_input: Option<Input>,
    cached_output: Option<Output>,
    hits: u64,
    misses: u64,
}

impl<Input, Output> Selector<Input, Output>
where
    Input: PartialEq + Clone,
    Output: Clone,
{
    /// Create a new selector with a transform function.
    pub fn new(transform: impl Fn(&Input) -> Output + 'static) -> Self {
        Self {
            transform: Box::new(transform),
            cached_input: None,
            cached_output: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Select: if input unchanged, return cached output; otherwise recompute.
    pub fn select(&mut self, input: &Input) -> Output {
        if let (Some(cached_in), Some(cached_out)) =
            (&self.cached_input, &self.cached_output)
        {
            if cached_in == input {
                self.hits += 1;
                return cached_out.clone();
            }
        }

        self.misses += 1;
        let output = (self.transform)(input);
        self.cached_input = Some(input.clone());
        self.cached_output = Some(output.clone());
        output
    }

    /// Reset the cache.
    pub fn reset(&mut self) {
        self.cached_input = None;
        self.cached_output = None;
    }

    /// Number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Total number of select calls.
    pub fn total_calls(&self) -> u64 {
        self.hits + self.misses
    }

    /// Hit rate as a fraction (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_calls();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ── Composed selectors ──

/// Create a selector that composes one input selector with a transform.
pub fn create_selector_1<S, A, Out>(
    sel_a: impl Fn(&S) -> A + 'static,
    transform: impl Fn(&A) -> Out + 'static,
) -> Selector<S, Out>
where
    S: PartialEq + Clone,
    A: 'static,
    Out: Clone,
{
    Selector::new(move |state: &S| {
        let a = sel_a(state);
        transform(&a)
    })
}

/// Create a selector that composes two input selectors with a transform.
pub fn create_selector_2<S, A, B, Out>(
    sel_a: impl Fn(&S) -> A + 'static,
    sel_b: impl Fn(&S) -> B + 'static,
    transform: impl Fn(&A, &B) -> Out + 'static,
) -> Selector<S, Out>
where
    S: PartialEq + Clone,
    A: 'static,
    B: 'static,
    Out: Clone,
{
    Selector::new(move |state: &S| {
        let a = sel_a(state);
        let b = sel_b(state);
        transform(&a, &b)
    })
}

/// Create a selector that composes three input selectors with a transform.
pub fn create_selector_3<S, A, B, C, Out>(
    sel_a: impl Fn(&S) -> A + 'static,
    sel_b: impl Fn(&S) -> B + 'static,
    sel_c: impl Fn(&S) -> C + 'static,
    transform: impl Fn(&A, &B, &C) -> Out + 'static,
) -> Selector<S, Out>
where
    S: PartialEq + Clone,
    A: 'static,
    B: 'static,
    C: 'static,
    Out: Clone,
{
    Selector::new(move |state: &S| {
        let a = sel_a(state);
        let b = sel_b(state);
        let c = sel_c(state);
        transform(&a, &b, &c)
    })
}

// ── Parameterized Selector (Factory) ──

/// A factory that creates memoized selectors per parameter value.
pub struct SelectorFactory<Param, Input, Output> {
    create_fn: Box<dyn Fn(&Param) -> Box<dyn Fn(&Input) -> Output>>,
    cache: HashMap<u64, Selector<Input, Output>>,
}

impl<Param, Input, Output> SelectorFactory<Param, Input, Output>
where
    Param: std::hash::Hash,
    Input: PartialEq + Clone,
    Output: Clone,
{
    /// Create a factory. The `create_fn` takes a param and returns a transform.
    pub fn new(
        create_fn: impl Fn(&Param) -> Box<dyn Fn(&Input) -> Output> + 'static,
    ) -> Self {
        Self {
            create_fn: Box::new(create_fn),
            cache: HashMap::new(),
        }
    }

    /// Get or create a selector for the given parameter.
    pub fn select(&mut self, param: &Param, input: &Input) -> Output {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        param.hash(&mut hasher);
        let key = std::hash::Hasher::finish(&hasher);

        if !self.cache.contains_key(&key) {
            let transform = (self.create_fn)(param);
            self.cache.insert(key, Selector {
                transform,
                cached_input: None,
                cached_output: None,
                hits: 0,
                misses: 0,
            });
        }
        self.cache.get_mut(&key).unwrap().select(input)
    }

    /// Number of cached selectors.
    pub fn cached_count(&self) -> usize {
        self.cache.len()
    }

    /// Clear all cached selectors.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

// ── Structural Sharing ──

/// Check if two values are structurally equal and return the original reference
/// to preserve identity (avoiding unnecessary re-renders/recomputation).
pub fn structural_share<T: PartialEq + Clone>(previous: &T, next: T) -> T {
    if *previous == next {
        previous.clone()
    } else {
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_select_and_cache() {
        let mut sel = Selector::new(|x: &i32| x * 2);
        assert_eq!(sel.select(&5), 10);
        assert_eq!(sel.misses(), 1);
        assert_eq!(sel.hits(), 0);

        // Same input — cache hit
        assert_eq!(sel.select(&5), 10);
        assert_eq!(sel.hits(), 1);
        assert_eq!(sel.misses(), 1);
    }

    #[test]
    fn cache_miss_on_change() {
        let mut sel = Selector::new(|x: &i32| x + 1);
        assert_eq!(sel.select(&10), 11);
        assert_eq!(sel.select(&20), 21);
        assert_eq!(sel.misses(), 2);
        assert_eq!(sel.hits(), 0);
    }

    #[test]
    fn reset_cache() {
        let mut sel = Selector::new(|x: &i32| *x);
        sel.select(&1);
        sel.select(&1);
        assert_eq!(sel.hits(), 1);

        sel.reset();
        sel.select(&1);
        assert_eq!(sel.misses(), 2); // initial + after reset
    }

    #[test]
    fn hit_rate() {
        let mut sel = Selector::new(|x: &i32| *x);
        assert_eq!(sel.hit_rate(), 0.0);

        sel.select(&1); // miss
        sel.select(&1); // hit
        sel.select(&1); // hit
        sel.select(&2); // miss
        assert_eq!(sel.total_calls(), 4);
        assert_eq!(sel.hit_rate(), 0.5);
    }

    #[test]
    fn composed_selector_1() {
        #[derive(Clone, PartialEq)]
        struct State {
            count: i32,
        }

        let mut sel = create_selector_1(
            |s: &State| s.count,
            |count| count * 10,
        );

        let s = State { count: 3 };
        assert_eq!(sel.select(&s), 30);
    }

    #[test]
    fn composed_selector_2() {
        #[derive(Clone, PartialEq)]
        struct State {
            a: i32,
            b: i32,
        }

        let mut sel = create_selector_2(
            |s: &State| s.a,
            |s: &State| s.b,
            |a, b| a + b,
        );

        let s = State { a: 3, b: 7 };
        assert_eq!(sel.select(&s), 10);
        // Same state — cache hit
        assert_eq!(sel.select(&s), 10);
        assert_eq!(sel.hits(), 1);
    }

    #[test]
    fn composed_selector_3() {
        #[derive(Clone, PartialEq)]
        struct State {
            x: i32,
            y: i32,
            z: i32,
        }

        let mut sel = create_selector_3(
            |s: &State| s.x,
            |s: &State| s.y,
            |s: &State| s.z,
            |x, y, z| x + y + z,
        );

        let s = State { x: 1, y: 2, z: 3 };
        assert_eq!(sel.select(&s), 6);
    }

    #[test]
    fn parameterized_selector_factory() {
        let mut factory = SelectorFactory::new(|multiplier: &i32| {
            let m = *multiplier;
            Box::new(move |x: &i32| x * m)
        });

        assert_eq!(factory.select(&2, &5), 10);
        assert_eq!(factory.select(&3, &5), 15);
        assert_eq!(factory.cached_count(), 2);

        // Same param, same input — cache hit
        assert_eq!(factory.select(&2, &5), 10);
    }

    #[test]
    fn factory_clear() {
        let mut factory = SelectorFactory::new(|_: &i32| {
            Box::new(|x: &i32| *x)
        });
        factory.select(&1, &10);
        factory.select(&2, &20);
        assert_eq!(factory.cached_count(), 2);
        factory.clear();
        assert_eq!(factory.cached_count(), 0);
    }

    #[test]
    fn structural_sharing() {
        let prev = vec![1, 2, 3];
        let next_same = vec![1, 2, 3];
        let result = structural_share(&prev, next_same);
        assert_eq!(result, vec![1, 2, 3]);

        let next_diff = vec![1, 2, 4];
        let result = structural_share(&prev, next_diff);
        assert_eq!(result, vec![1, 2, 4]);
    }

    #[test]
    fn selector_with_string_input() {
        let mut sel = Selector::new(|s: &String| s.len());
        assert_eq!(sel.select(&"hello".to_string()), 5);
        assert_eq!(sel.select(&"hello".to_string()), 5);
        assert_eq!(sel.hits(), 1);
        assert_eq!(sel.select(&"hi".to_string()), 2);
        assert_eq!(sel.misses(), 2);
    }
}
