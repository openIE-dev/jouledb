//! Generic collection abstractions — Collection trait (len/is_empty/iter/contains),
//! IndexedCollection, SortedCollection, UniqueCollection, collection adapters
//! (filtered/mapped/chained), and collection statistics.
//!
//! Replaces lodash, ramda, Immutable.js collection utilities with pure-Rust
//! generic collection abstractions.

use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::hash::Hash;

// ── Collection Trait ────────────────────────────────────────────

/// A generic collection that supports length, emptiness, and iteration.
pub trait Collection {
    /// The element type.
    type Item;

    /// Number of elements.
    fn len(&self) -> usize;

    /// Whether the collection is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over elements by reference.
    fn iter(&self) -> Box<dyn Iterator<Item = &Self::Item> + '_>;

    /// Whether the collection contains an element.
    fn contains(&self, item: &Self::Item) -> bool
    where
        Self::Item: PartialEq,
    {
        self.iter().any(|x| x == item)
    }

    /// Collect into a Vec.
    fn to_vec(&self) -> Vec<Self::Item>
    where
        Self::Item: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Count elements matching a predicate.
    fn count_where(&self, predicate: impl Fn(&Self::Item) -> bool) -> usize {
        self.iter().filter(|x| predicate(x)).count()
    }

    /// Whether all elements match a predicate.
    fn all(&self, predicate: impl Fn(&Self::Item) -> bool) -> bool {
        self.iter().all(|x| predicate(x))
    }

    /// Whether any element matches a predicate.
    fn any_match(&self, predicate: impl Fn(&Self::Item) -> bool) -> bool {
        self.iter().any(|x| predicate(x))
    }

    /// Find the first element matching a predicate.
    fn find_first(&self, predicate: impl Fn(&Self::Item) -> bool) -> Option<&Self::Item> {
        self.iter().find(|x| predicate(x))
    }
}

// ── IndexedCollection ───────────────────────────────────────────

/// A collection with index-based access.
pub trait IndexedCollection: Collection {
    /// Get element at index.
    fn get(&self, index: usize) -> Option<&Self::Item>;

    /// Get the first element.
    fn first(&self) -> Option<&Self::Item> {
        self.get(0)
    }

    /// Get the last element.
    fn last(&self) -> Option<&Self::Item> {
        if self.is_empty() {
            None
        } else {
            self.get(self.len() - 1)
        }
    }

    /// Find the index of the first element matching a predicate.
    fn position(&self, predicate: impl Fn(&Self::Item) -> bool) -> Option<usize> {
        for i in 0..self.len() {
            if let Some(item) = self.get(i) {
                if predicate(item) {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Slice from index `start` to `end` (exclusive).
    fn slice(&self, start: usize, end: usize) -> Vec<&Self::Item> {
        let actual_end = end.min(self.len());
        let mut result = Vec::new();
        for i in start..actual_end {
            if let Some(item) = self.get(i) {
                result.push(item);
            }
        }
        result
    }
}

// ── SortedCollection ────────────────────────────────────────────

/// A collection that maintains sorted order.
pub trait SortedCollection: Collection {
    /// Get the minimum element.
    fn min(&self) -> Option<&Self::Item>;

    /// Get the maximum element.
    fn max(&self) -> Option<&Self::Item>;

    /// Binary search for an element, returning its index or insertion point.
    fn binary_search(&self, item: &Self::Item) -> Result<usize, usize>
    where
        Self::Item: Ord;

    /// Get elements in the range [low, high].
    fn range(&self, low: &Self::Item, high: &Self::Item) -> Vec<&Self::Item>
    where
        Self::Item: Ord;
}

// ── UniqueCollection ────────────────────────────────────────────

/// A collection with no duplicate elements.
pub trait UniqueCollection: Collection {
    /// Insert an element. Returns true if it was newly inserted.
    fn insert(&mut self, item: Self::Item) -> bool;

    /// Remove an element. Returns true if it was present.
    fn remove(&mut self, item: &Self::Item) -> bool;

    /// Intersection with another unique collection.
    fn intersection_with(&self, other: &Self) -> Vec<Self::Item>
    where
        Self::Item: Clone + PartialEq;

    /// Union with another unique collection.
    fn union_with(&self, other: &Self) -> Vec<Self::Item>
    where
        Self::Item: Clone + PartialEq;

    /// Difference: elements in self not in other.
    fn difference_with(&self, other: &Self) -> Vec<Self::Item>
    where
        Self::Item: Clone + PartialEq;
}

// ── VecCollection ───────────────────────────────────────────────

/// A Vec-backed collection implementing Collection and IndexedCollection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VecCollection<T> {
    items: Vec<T>,
}

impl<T> VecCollection<T> {
    /// Create an empty VecCollection.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Create from a vec.
    pub fn from_vec(items: Vec<T>) -> Self {
        Self { items }
    }

    /// Push an item.
    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    /// Pop the last item.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    /// Remove at index.
    pub fn remove(&mut self, index: usize) -> T {
        self.items.remove(index)
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Consume into inner Vec.
    pub fn into_inner(self) -> Vec<T> {
        self.items
    }
}

impl<T> Default for VecCollection<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Collection for VecCollection<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        Box::new(self.items.iter())
    }
}

impl<T> IndexedCollection for VecCollection<T> {
    fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }
}

impl<T> From<Vec<T>> for VecCollection<T> {
    fn from(items: Vec<T>) -> Self {
        Self { items }
    }
}

// ── SortedVec ───────────────────────────────────────────────────

/// A Vec that maintains sorted order on insertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedVec<T: Ord> {
    items: Vec<T>,
}

impl<T: Ord> SortedVec<T> {
    /// Create an empty sorted vec.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Insert an item, maintaining sorted order.
    pub fn insert(&mut self, item: T) {
        let pos = self.items.binary_search(&item).unwrap_or_else(|p| p);
        self.items.insert(pos, item);
    }

    /// Remove an item.
    pub fn remove(&mut self, item: &T) -> bool {
        if let Ok(pos) = self.items.binary_search(item) {
            self.items.remove(pos);
            true
        } else {
            false
        }
    }

    /// Create from an unsorted iterator.
    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let mut items: Vec<T> = iter.into_iter().collect();
        items.sort();
        Self { items }
    }
}

impl<T: Ord> Default for SortedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord> Collection for SortedVec<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        Box::new(self.items.iter())
    }
}

impl<T: Ord> IndexedCollection for SortedVec<T> {
    fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }
}

impl<T: Ord> SortedCollection for SortedVec<T> {
    fn min(&self) -> Option<&T> {
        self.items.first()
    }

    fn max(&self) -> Option<&T> {
        self.items.last()
    }

    fn binary_search(&self, item: &T) -> Result<usize, usize> {
        self.items.binary_search(item)
    }

    fn range(&self, low: &T, high: &T) -> Vec<&T> {
        self.items.iter().filter(|x| *x >= low && *x <= high).collect()
    }
}

// ── UniqueSet ───────────────────────────────────────────────────

/// A BTreeSet-backed unique collection (sorted and unique).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniqueSet<T: Ord> {
    items: BTreeSet<T>,
}

impl<T: Ord> UniqueSet<T> {
    /// Create an empty unique set.
    pub fn new() -> Self {
        Self { items: BTreeSet::new() }
    }

    /// Create from an iterator, deduplicating.
    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        Self { items: iter.into_iter().collect() }
    }
}

impl<T: Ord> Default for UniqueSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord> Collection for UniqueSet<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &T> + '_> {
        Box::new(self.items.iter())
    }
}

impl<T: Ord + Clone> UniqueCollection for UniqueSet<T> {
    fn insert(&mut self, item: T) -> bool {
        self.items.insert(item)
    }

    fn remove(&mut self, item: &T) -> bool {
        self.items.remove(item)
    }

    fn intersection_with(&self, other: &Self) -> Vec<T> {
        self.items.intersection(&other.items).cloned().collect()
    }

    fn union_with(&self, other: &Self) -> Vec<T> {
        self.items.union(&other.items).cloned().collect()
    }

    fn difference_with(&self, other: &Self) -> Vec<T> {
        self.items.difference(&other.items).cloned().collect()
    }
}

// ── Collection Adapters ─────────────────────────────────────────

/// A filtered view over a collection.
pub struct FilteredCollection<'a, C: Collection> {
    source: &'a C,
    predicate: Box<dyn Fn(&C::Item) -> bool + 'a>,
}

impl<'a, C: Collection> FilteredCollection<'a, C> {
    /// Create a filtered view.
    pub fn new(source: &'a C, predicate: impl Fn(&C::Item) -> bool + 'a) -> Self {
        Self {
            source,
            predicate: Box::new(predicate),
        }
    }

    /// Count matching items.
    pub fn len(&self) -> usize {
        self.source.iter().filter(|x| (self.predicate)(x)).count()
    }

    /// Whether no items match.
    pub fn is_empty(&self) -> bool {
        !self.source.iter().any(|x| (self.predicate)(x))
    }

    /// Collect matching items.
    pub fn collect(&self) -> Vec<&C::Item> {
        self.source.iter().filter(|x| (self.predicate)(x)).collect()
    }
}

/// Map items from a collection, producing a new Vec.
pub fn map_collection<C: Collection, U>(
    source: &C,
    f: impl Fn(&C::Item) -> U,
) -> Vec<U> {
    source.iter().map(|x| f(x)).collect()
}

/// Chain two collections into a Vec of references.
pub fn chain_collections<'a, A, B>(
    a: &'a A,
    b: &'a B,
) -> Vec<&'a A::Item>
where
    A: Collection,
    B: Collection<Item = A::Item>,
{
    a.iter().chain(b.iter()).collect()
}

// ── Collection Statistics ───────────────────────────────────────

/// Statistics computed over a numeric collection.
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionStats {
    /// Number of elements.
    pub count: usize,
    /// Sum of elements.
    pub sum: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean (average).
    pub mean: f64,
    /// Variance.
    pub variance: f64,
    /// Standard deviation.
    pub std_dev: f64,
}

impl fmt::Display for CollectionStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "count={}, sum={:.2}, min={:.2}, max={:.2}, mean={:.2}, std_dev={:.2}",
            self.count, self.sum, self.min, self.max, self.mean, self.std_dev
        )
    }
}

/// Compute statistics over a collection of f64 values.
pub fn compute_stats(values: &[f64]) -> Option<CollectionStats> {
    if values.is_empty() {
        return None;
    }
    let count = values.len();
    let sum: f64 = values.iter().sum();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = sum / count as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    Some(CollectionStats { count, sum, min, max, mean, variance, std_dev })
}

/// Compute the median of a slice.
pub fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[mid - 1] + values[mid]) / 2.0)
    } else {
        Some(values[mid])
    }
}

/// Compute percentile (0.0 to 100.0) of a slice using nearest-rank method.
pub fn percentile(values: &mut [f64], p: f64) -> Option<f64> {
    if values.is_empty() || p < 0.0 || p > 100.0 {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p / 100.0 * values.len() as f64).ceil() as usize;
    let index = rank.min(values.len()).saturating_sub(1);
    Some(values[index])
}

/// Distinct elements in a collection (preserving first occurrence order).
pub fn distinct<T: Eq + Hash + Clone>(items: &[T]) -> Vec<T> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            result.push(item.clone());
        }
    }
    result
}

/// Group elements by a key function.
pub fn group_by<T, K: Eq + Hash>(
    items: &[T],
    key_fn: impl Fn(&T) -> K,
) -> Vec<(K, Vec<&T>)> {
    // Use a Vec of (K, Vec<&T>) to preserve insertion order.
    let mut groups: Vec<(K, Vec<&T>)> = Vec::new();
    for item in items {
        let k = key_fn(item);
        if let Some(group) = groups.iter_mut().find(|(gk, _)| *gk == k) {
            group.1.push(item);
        } else {
            groups.push((k, vec![item]));
        }
    }
    groups
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_collection_basic() {
        let mut c = VecCollection::new();
        c.push(1);
        c.push(2);
        c.push(3);
        assert_eq!(c.len(), 3);
        assert!(!c.is_empty());
        assert!(c.contains(&2));
        assert!(!c.contains(&4));
    }

    #[test]
    fn test_vec_collection_to_vec() {
        let c = VecCollection::from_vec(vec![1, 2, 3]);
        assert_eq!(c.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn test_vec_collection_indexed() {
        let c = VecCollection::from_vec(vec![10, 20, 30]);
        assert_eq!(c.get(0), Some(&10));
        assert_eq!(c.get(2), Some(&30));
        assert_eq!(c.get(3), None);
        assert_eq!(c.first(), Some(&10));
        assert_eq!(c.last(), Some(&30));
    }

    #[test]
    fn test_vec_collection_position() {
        let c = VecCollection::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(c.position(|x| *x == 3), Some(2));
        assert_eq!(c.position(|x| *x == 99), None);
    }

    #[test]
    fn test_vec_collection_slice() {
        let c = VecCollection::from_vec(vec![10, 20, 30, 40, 50]);
        let s = c.slice(1, 4);
        assert_eq!(s, vec![&20, &30, &40]);
    }

    #[test]
    fn test_vec_collection_count_where() {
        let c = VecCollection::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(c.count_where(|x| *x > 3), 2);
    }

    #[test]
    fn test_vec_collection_all_any() {
        let c = VecCollection::from_vec(vec![2, 4, 6]);
        assert!(c.all(|x| x % 2 == 0));
        assert!(c.any_match(|x| *x == 4));
        assert!(!c.any_match(|x| *x == 5));
    }

    #[test]
    fn test_vec_collection_find_first() {
        let c = VecCollection::from_vec(vec![1, 2, 3, 4]);
        assert_eq!(c.find_first(|x| *x > 2), Some(&3));
        assert_eq!(c.find_first(|x| *x > 10), None);
    }

    #[test]
    fn test_sorted_vec() {
        let mut sv = SortedVec::new();
        sv.insert(3);
        sv.insert(1);
        sv.insert(2);
        assert_eq!(sv.to_vec(), vec![1, 2, 3]);
        assert_eq!(sv.min(), Some(&1));
        assert_eq!(sv.max(), Some(&3));
    }

    #[test]
    fn test_sorted_vec_binary_search() {
        let sv = SortedVec::from_iter(vec![10, 20, 30, 40]);
        assert_eq!(sv.binary_search(&20), Ok(1));
        assert_eq!(sv.binary_search(&25), Err(2));
    }

    #[test]
    fn test_sorted_vec_range() {
        let sv = SortedVec::from_iter(vec![1, 3, 5, 7, 9]);
        let r = sv.range(&3, &7);
        assert_eq!(r, vec![&3, &5, &7]);
    }

    #[test]
    fn test_sorted_vec_remove() {
        let mut sv = SortedVec::from_iter(vec![1, 2, 3]);
        assert!(sv.remove(&2));
        assert!(!sv.remove(&2));
        assert_eq!(sv.to_vec(), vec![1, 3]);
    }

    #[test]
    fn test_unique_set() {
        let mut us = UniqueSet::new();
        assert!(us.insert(1));
        assert!(us.insert(2));
        assert!(!us.insert(1)); // duplicate
        assert_eq!(us.len(), 2);
    }

    #[test]
    fn test_unique_set_intersection() {
        let a = UniqueSet::from_iter(vec![1, 2, 3, 4]);
        let b = UniqueSet::from_iter(vec![3, 4, 5, 6]);
        assert_eq!(a.intersection_with(&b), vec![3, 4]);
    }

    #[test]
    fn test_unique_set_union() {
        let a = UniqueSet::from_iter(vec![1, 2, 3]);
        let b = UniqueSet::from_iter(vec![3, 4, 5]);
        assert_eq!(a.union_with(&b), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_unique_set_difference() {
        let a = UniqueSet::from_iter(vec![1, 2, 3, 4]);
        let b = UniqueSet::from_iter(vec![3, 4, 5]);
        assert_eq!(a.difference_with(&b), vec![1, 2]);
    }

    #[test]
    fn test_unique_set_remove() {
        let mut us = UniqueSet::from_iter(vec![1, 2, 3]);
        assert!(us.remove(&2));
        assert!(!us.remove(&2));
        assert_eq!(us.len(), 2);
    }

    #[test]
    fn test_filtered_collection() {
        let c = VecCollection::from_vec(vec![1, 2, 3, 4, 5]);
        let filtered = FilteredCollection::new(&c, |x| *x > 2);
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered.collect(), vec![&3, &4, &5]);
    }

    #[test]
    fn test_filtered_collection_empty() {
        let c = VecCollection::from_vec(vec![1, 2, 3]);
        let filtered = FilteredCollection::new(&c, |x| *x > 10);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_map_collection() {
        let c = VecCollection::from_vec(vec![1, 2, 3]);
        let mapped = map_collection(&c, |x| x * 2);
        assert_eq!(mapped, vec![2, 4, 6]);
    }

    #[test]
    fn test_chain_collections() {
        let a = VecCollection::from_vec(vec![1, 2]);
        let b = VecCollection::from_vec(vec![3, 4]);
        let chained = chain_collections(&a, &b);
        assert_eq!(chained, vec![&1, &2, &3, &4]);
    }

    #[test]
    fn test_compute_stats() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = compute_stats(&data).unwrap();
        assert_eq!(stats.count, 8);
        assert_eq!(stats.sum, 40.0);
        assert_eq!(stats.min, 2.0);
        assert_eq!(stats.max, 9.0);
        assert_eq!(stats.mean, 5.0);
        assert!((stats.variance - 4.0).abs() < 0.001);
        assert!((stats.std_dev - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_empty() {
        assert!(compute_stats(&[]).is_none());
    }

    #[test]
    fn test_median_odd() {
        let mut data = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut data), Some(2.0));
    }

    #[test]
    fn test_median_even() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(median(&mut data), Some(2.5));
    }

    #[test]
    fn test_median_empty() {
        assert_eq!(median(&mut []), None);
    }

    #[test]
    fn test_percentile() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&mut data, 50.0), Some(5.0));
        assert_eq!(percentile(&mut data, 100.0), Some(10.0));
        assert_eq!(percentile(&mut data, 0.0), Some(1.0));
    }

    #[test]
    fn test_distinct() {
        let items = vec![1, 2, 3, 2, 1, 4];
        let d = distinct(&items);
        assert_eq!(d, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_group_by() {
        let items = vec![1, 2, 3, 4, 5, 6];
        let groups = group_by(&items, |x| if *x % 2 == 0 { "even" } else { "odd" });
        // Since we use a Vec, order is insertion order.
        let odd_group = groups.iter().find(|(k, _)| *k == "odd").unwrap();
        let even_group = groups.iter().find(|(k, _)| *k == "even").unwrap();
        assert_eq!(odd_group.1.len(), 3);
        assert_eq!(even_group.1.len(), 3);
    }

    #[test]
    fn test_stats_display() {
        let stats = compute_stats(&[1.0, 2.0, 3.0]).unwrap();
        let s = stats.to_string();
        assert!(s.contains("count=3"));
    }

    #[test]
    fn test_vec_collection_pop_and_clear() {
        let mut c = VecCollection::from_vec(vec![1, 2, 3]);
        assert_eq!(c.pop(), Some(3));
        assert_eq!(c.len(), 2);
        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn test_empty_collections() {
        let v: VecCollection<i32> = VecCollection::new();
        assert!(v.is_empty());
        assert_eq!(v.first(), None);
        assert_eq!(v.last(), None);

        let sv: SortedVec<i32> = SortedVec::new();
        assert!(sv.is_empty());
        assert_eq!(sv.min(), None);
        assert_eq!(sv.max(), None);
    }

    #[test]
    fn test_vec_collection_into_inner() {
        let c = VecCollection::from_vec(vec![1, 2, 3]);
        let v = c.into_inner();
        assert_eq!(v, vec![1, 2, 3]);
    }
}
