//! Immutable data structures with structural sharing (Immer pattern).
//!
//! Provides persistent `ImmutableMap`, `ImmutableList`, an Immer-style `produce`
//! function, and an undo/redo `History` stack.

use std::fmt;

// ── ImmutableMap ────────────────────────────────────────────────

/// A persistent sorted map backed by a `Vec` of key-value pairs.
/// All mutation methods return a new map, leaving the original unchanged.
#[derive(Clone, PartialEq, Eq)]
pub struct ImmutableMap<K: Ord + Clone, V: Clone> {
    entries: Vec<(K, V)>,
}

impl<K: Ord + Clone, V: Clone> ImmutableMap<K, V> {
    /// Create an empty map.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return a new map with the key set to value.
    pub fn set(&self, key: K, value: V) -> Self {
        let mut entries = self.entries.clone();
        match entries.binary_search_by(|(k, _)| k.cmp(&key)) {
            Ok(idx) => entries[idx].1 = value,
            Err(idx) => entries.insert(idx, (key, value)),
        }
        Self { entries }
    }

    /// Look up a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .ok()
            .map(|idx| &self.entries[idx].1)
    }

    /// Return a new map with the key removed.
    pub fn remove(&self, key: &K) -> Self {
        let mut entries = self.entries.clone();
        if let Ok(idx) = entries.binary_search_by(|(k, _)| k.cmp(key)) {
            entries.remove(idx);
        }
        Self { entries }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the map contains the given key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries
            .binary_search_by(|(k, _)| k.cmp(key))
            .is_ok()
    }

    /// Iterate over keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.entries.iter().map(|(k, _)| k)
    }

    /// Iterate over values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    /// Iterate over key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

impl<K: Ord + Clone, V: Clone> Default for ImmutableMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone + fmt::Debug, V: Clone + fmt::Debug> fmt::Debug for ImmutableMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

// ── ImmutableList ───────────────────────────────────────────────

/// A persistent list backed by a `Vec`. All mutation methods return a new list.
#[derive(Clone, PartialEq, Eq)]
pub struct ImmutableList<T: Clone> {
    items: Vec<T>,
}

impl<T: Clone> ImmutableList<T> {
    /// Create an empty list.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Return a new list with value appended.
    pub fn push(&self, value: T) -> Self {
        let mut items = self.items.clone();
        items.push(value);
        Self { items }
    }

    /// Return a new list with the last element removed, and the removed element.
    pub fn pop(&self) -> (Self, Option<T>) {
        if self.items.is_empty() {
            return (self.clone(), None);
        }
        let mut items = self.items.clone();
        let val = items.pop();
        (Self { items }, val)
    }

    /// Get element at index.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    /// Return a new list with element at index replaced.
    pub fn set(&self, index: usize, value: T) -> Self {
        let mut items = self.items.clone();
        if index < items.len() {
            items[index] = value;
        }
        Self { items }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }
}

impl<T: Clone> Default for ImmutableList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for ImmutableList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.items.iter()).finish()
    }
}

// ── produce ─────────────────────────────────────────────────────

/// Immer-style produce: clone the base, apply mutations via recipe, return new value.
pub fn produce<T: Clone>(base: &T, recipe: impl FnOnce(&mut T)) -> T {
    let mut draft = base.clone();
    recipe(&mut draft);
    draft
}

// ── History ─────────────────────────────────────────────────────

/// Undo/redo history stack.
pub struct History<T: Clone> {
    states: Vec<T>,
    current: usize,
}

impl<T: Clone> History<T> {
    /// Create a new history with an initial state.
    pub fn new(initial: T) -> Self {
        Self {
            states: vec![initial],
            current: 0,
        }
    }

    /// Push a new state, discarding any redo history.
    pub fn push(&mut self, state: T) {
        self.states.truncate(self.current + 1);
        self.states.push(state);
        self.current += 1;
    }

    /// Undo to the previous state.
    pub fn undo(&mut self) -> Option<&T> {
        if self.current > 0 {
            self.current -= 1;
            Some(&self.states[self.current])
        } else {
            None
        }
    }

    /// Redo to the next state.
    pub fn redo(&mut self) -> Option<&T> {
        if self.current + 1 < self.states.len() {
            self.current += 1;
            Some(&self.states[self.current])
        } else {
            None
        }
    }

    /// The current state.
    pub fn current(&self) -> &T {
        &self.states[self.current]
    }

    /// Whether undo is possible.
    pub fn can_undo(&self) -> bool {
        self.current > 0
    }

    /// Whether redo is possible.
    pub fn can_redo(&self) -> bool {
        self.current + 1 < self.states.len()
    }

    /// Number of states in history.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether history is empty (should never be, but defensive).
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_set_returns_new() {
        let m1 = ImmutableMap::new();
        let m2 = m1.set("a", 1);
        assert_eq!(m2.get(&"a"), Some(&1));
        assert_eq!(m2.len(), 1);
    }

    #[test]
    fn original_unchanged() {
        let m1 = ImmutableMap::new().set("a", 1);
        let m2 = m1.set("b", 2);
        assert_eq!(m1.len(), 1);
        assert_eq!(m2.len(), 2);
        assert!(m1.get(&"b").is_none());
    }

    #[test]
    fn map_remove() {
        let m = ImmutableMap::new().set("a", 1).set("b", 2);
        let m2 = m.remove(&"a");
        assert_eq!(m2.len(), 1);
        assert!(m2.get(&"a").is_none());
        assert_eq!(m2.get(&"b"), Some(&2));
    }

    #[test]
    fn map_contains_key() {
        let m = ImmutableMap::new().set("x", 42);
        assert!(m.contains_key(&"x"));
        assert!(!m.contains_key(&"y"));
    }

    #[test]
    fn map_keys_iteration() {
        let m = ImmutableMap::new().set("b", 2).set("a", 1).set("c", 3);
        let keys: Vec<_> = m.keys().collect();
        assert_eq!(keys, vec![&"a", &"b", &"c"]);
    }

    #[test]
    fn map_overwrite() {
        let m = ImmutableMap::new().set("a", 1);
        let m2 = m.set("a", 99);
        assert_eq!(m.get(&"a"), Some(&1));
        assert_eq!(m2.get(&"a"), Some(&99));
    }

    #[test]
    fn list_push_pop() {
        let l = ImmutableList::new().push(1).push(2).push(3);
        assert_eq!(l.len(), 3);
        assert_eq!(l.get(0), Some(&1));
        assert_eq!(l.get(2), Some(&3));

        let (l2, val) = l.pop();
        assert_eq!(val, Some(3));
        assert_eq!(l2.len(), 2);
        assert_eq!(l.len(), 3); // original unchanged
    }

    #[test]
    fn list_set() {
        let l = ImmutableList::new().push(10).push(20).push(30);
        let l2 = l.set(1, 99);
        assert_eq!(l.get(1), Some(&20));
        assert_eq!(l2.get(1), Some(&99));
    }

    #[test]
    fn produce_applies_mutation() {
        let original = vec![1, 2, 3];
        let modified = produce(&original, |draft| {
            draft.push(4);
            draft[0] = 10;
        });
        assert_eq!(original, vec![1, 2, 3]);
        assert_eq!(modified, vec![10, 2, 3, 4]);
    }

    #[test]
    fn history_undo_redo() {
        let mut h = History::new(0);
        h.push(1);
        h.push(2);
        h.push(3);
        assert_eq!(*h.current(), 3);

        assert_eq!(h.undo(), Some(&2));
        assert_eq!(*h.current(), 2);

        assert_eq!(h.undo(), Some(&1));
        assert_eq!(*h.current(), 1);

        assert_eq!(h.redo(), Some(&2));
        assert_eq!(*h.current(), 2);

        // Push discards redo history
        h.push(99);
        assert_eq!(*h.current(), 99);
        assert!(!h.can_redo());
    }

    #[test]
    fn history_can_undo_redo() {
        let mut h = History::new("start");
        assert!(!h.can_undo());
        assert!(!h.can_redo());

        h.push("second");
        assert!(h.can_undo());
        assert!(!h.can_redo());

        h.undo();
        assert!(!h.can_undo());
        assert!(h.can_redo());
    }

    #[test]
    fn immutable_equality() {
        let m1 = ImmutableMap::new().set("a", 1).set("b", 2);
        let m2 = ImmutableMap::new().set("b", 2).set("a", 1);
        assert_eq!(m1, m2);
    }

    #[test]
    fn list_empty_pop() {
        let l: ImmutableList<i32> = ImmutableList::new();
        let (l2, val) = l.pop();
        assert!(val.is_none());
        assert!(l2.is_empty());
    }
}
