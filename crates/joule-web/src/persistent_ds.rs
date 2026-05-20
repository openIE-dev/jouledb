//! Persistent (immutable) data structures with structural sharing.
//!
//! Provides `PersistentVec<T>` (array mapped trie), `PersistentMap<K,V>`
//! (hash array mapped trie — HAMT), structural sharing via `Rc`,
//! efficient clone via path copying, and transient (mutable) mode for
//! bulk operations.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

// ── Constants ───────────────────────────────────────────────────────────────

/// Branching factor bits (5 → 32-way branching).
const BITS: u32 = 5;
/// Branching factor (2^BITS = 32).
const WIDTH: usize = 1 << BITS;
/// Mask for extracting a BITS-wide index.
const MASK: usize = WIDTH - 1;

// ── PersistentVec ───────────────────────────────────────────────────────────

/// A persistent vector backed by a 32-way trie.
///
/// All mutation methods return a new vector, sharing structure with
/// the old one via `Rc`.
#[derive(Clone)]
pub struct PersistentVec<T: Clone> {
    len: usize,
    shift: u32,
    root: Rc<VecNode<T>>,
    tail: Rc<Vec<T>>,
}

#[derive(Clone)]
enum VecNode<T: Clone> {
    Internal(Vec<Option<Rc<VecNode<T>>>>),
    Leaf(Vec<T>),
}

impl<T: Clone> PersistentVec<T> {
    /// Create an empty persistent vector.
    pub fn new() -> Self {
        Self {
            len: 0,
            shift: BITS,
            root: Rc::new(VecNode::Internal(Vec::new())),
            tail: Rc::new(Vec::new()),
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn tail_offset(&self) -> usize {
        if self.len < WIDTH {
            0
        } else {
            ((self.len - 1) >> BITS) << BITS
        }
    }

    /// Get the element at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        if index >= self.tail_offset() {
            return self.tail.get(index - self.tail_offset());
        }
        // Walk the trie.
        let mut node = &*self.root;
        let mut level = self.shift;
        loop {
            match node {
                VecNode::Internal(children) => {
                    let idx = (index >> level) & MASK;
                    match children.get(idx) {
                        Some(Some(child)) => {
                            node = child;
                            if level >= BITS {
                                level -= BITS;
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }
                VecNode::Leaf(items) => {
                    let idx = index & MASK;
                    return items.get(idx);
                }
            }
        }
    }

    /// Append an element, returning a new vector.
    pub fn push(&self, value: T) -> Self {
        // If tail has room, just extend it.
        if self.tail.len() < WIDTH {
            let mut new_tail = (*self.tail).clone();
            new_tail.push(value);
            return Self {
                len: self.len + 1,
                shift: self.shift,
                root: self.root.clone(),
                tail: Rc::new(new_tail),
            };
        }

        // Tail is full — push it into the trie and start a new tail.
        let tail_node = Rc::new(VecNode::Leaf((*self.tail).clone()));
        let new_root;
        let new_shift;

        // Check if root is full — need to grow.
        if (self.len >> BITS) > (1usize << self.shift) {
            let mut children: Vec<Option<Rc<VecNode<T>>>> = Vec::with_capacity(WIDTH);
            children.push(Some(self.root.clone()));
            children.push(Some(Self::new_path(self.shift, tail_node)));
            new_root = Rc::new(VecNode::Internal(children));
            new_shift = self.shift + BITS;
        } else {
            new_root = Self::push_tail(self.shift, &self.root, tail_node);
            new_shift = self.shift;
        }

        Self {
            len: self.len + 1,
            shift: new_shift,
            root: new_root,
            tail: Rc::new(vec![value]),
        }
    }

    fn new_path(level: u32, node: Rc<VecNode<T>>) -> Rc<VecNode<T>> {
        if level == 0 {
            return node;
        }
        let child = Self::new_path(level - BITS, node);
        let mut children: Vec<Option<Rc<VecNode<T>>>> = Vec::new();
        children.push(Some(child));
        Rc::new(VecNode::Internal(children))
    }

    fn push_tail(level: u32, parent: &VecNode<T>, tail_node: Rc<VecNode<T>>) -> Rc<VecNode<T>> {
        match parent {
            VecNode::Internal(children) => {
                let mut new_children = children.clone();
                if level == BITS {
                    // Insert at the next available slot.
                    new_children.push(Some(tail_node));
                } else {
                    let last_idx = new_children.len().saturating_sub(1);
                    if let Some(Some(child)) = new_children.get(last_idx) {
                        let new_child =
                            Self::push_tail(level - BITS, child, tail_node);
                        new_children[last_idx] = Some(new_child);
                    } else {
                        let new_child = Self::new_path(level - BITS, tail_node);
                        new_children.push(Some(new_child));
                    }
                }
                Rc::new(VecNode::Internal(new_children))
            }
            VecNode::Leaf(_) => {
                // Should not happen at level > 0.
                Rc::new(parent.clone())
            }
        }
    }

    /// Set the element at `index`, returning a new vector.
    /// Returns `None` if index is out of bounds.
    pub fn set(&self, index: usize, value: T) -> Option<Self> {
        if index >= self.len {
            return None;
        }
        if index >= self.tail_offset() {
            let mut new_tail = (*self.tail).clone();
            new_tail[index - self.tail_offset()] = value;
            return Some(Self {
                len: self.len,
                shift: self.shift,
                root: self.root.clone(),
                tail: Rc::new(new_tail),
            });
        }
        let new_root = Self::set_in_trie(self.shift, &self.root, index, value);
        Some(Self {
            len: self.len,
            shift: self.shift,
            root: new_root,
            tail: self.tail.clone(),
        })
    }

    fn set_in_trie(level: u32, node: &VecNode<T>, index: usize, value: T) -> Rc<VecNode<T>> {
        match node {
            VecNode::Internal(children) => {
                let idx = (index >> level) & MASK;
                let mut new_children = children.clone();
                if let Some(Some(child)) = new_children.get(idx) {
                    let new_child = Self::set_in_trie(level - BITS, child, index, value);
                    new_children[idx] = Some(new_child);
                }
                Rc::new(VecNode::Internal(new_children))
            }
            VecNode::Leaf(items) => {
                let idx = index & MASK;
                let mut new_items = items.clone();
                if idx < new_items.len() {
                    new_items[idx] = value;
                }
                Rc::new(VecNode::Leaf(new_items))
            }
        }
    }

    /// Collect all elements into a `Vec`.
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            if let Some(v) = self.get(i) {
                result.push(v.clone());
            }
        }
        result
    }

    /// Create from a slice.
    pub fn from_slice(items: &[T]) -> Self {
        let mut v = Self::new();
        for item in items {
            v = v.push(item.clone());
        }
        v
    }

    /// Map a function over all elements, returning a new persistent vector.
    pub fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> PersistentVec<U> {
        let items: Vec<U> = self.to_vec().iter().map(f).collect();
        PersistentVec::from_slice(&items)
    }

    /// Filter elements, returning a new persistent vector.
    pub fn filter(&self, pred: impl Fn(&T) -> bool) -> Self {
        let items: Vec<T> = self.to_vec().into_iter().filter(|v| pred(v)).collect();
        Self::from_slice(&items)
    }

    /// Iterator over elements.
    pub fn iter(&self) -> PersistentVecIter<'_, T> {
        PersistentVecIter { vec: self, index: 0 }
    }
}

impl<T: Clone> Default for PersistentVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + fmt::Debug> fmt::Debug for PersistentVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Clone + PartialEq> PartialEq for PersistentVec<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for i in 0..self.len {
            if self.get(i) != other.get(i) {
                return false;
            }
        }
        true
    }
}

impl<T: Clone + Eq> Eq for PersistentVec<T> {}

/// Iterator over `PersistentVec`.
pub struct PersistentVecIter<'a, T: Clone> {
    vec: &'a PersistentVec<T>,
    index: usize,
}

impl<'a, T: Clone> Iterator for PersistentVecIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.vec.len() {
            let val = self.vec.get(self.index);
            self.index += 1;
            val
        } else {
            None
        }
    }
}

// ── PersistentMap ───────────────────────────────────────────────────────────

/// A persistent hash map backed by a Hash Array Mapped Trie (HAMT).
///
/// All mutation methods return a new map; the old version is unchanged.
#[derive(Clone)]
pub struct PersistentMap<K: Hash + Eq + Clone, V: Clone> {
    root: Rc<HamtNode<K, V>>,
    len: usize,
}

#[derive(Clone)]
enum HamtNode<K: Hash + Eq + Clone, V: Clone> {
    Empty,
    Leaf(u64, K, V),
    Collision(u64, Vec<(K, V)>),
    Internal(u32, Vec<Rc<HamtNode<K, V>>>),
}

fn hash_key<K: Hash>(key: &K) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

impl<K: Hash + Eq + Clone, V: Clone> PersistentMap<K, V> {
    /// Create an empty persistent map.
    pub fn new() -> Self {
        Self {
            root: Rc::new(HamtNode::Empty),
            len: 0,
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up a value by key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let h = hash_key(key);
        Self::get_in(&self.root, h, key, 0)
    }

    fn get_in<'a>(node: &'a HamtNode<K, V>, hash: u64, key: &K, shift: u32) -> Option<&'a V> {
        match node {
            HamtNode::Empty => None,
            HamtNode::Leaf(h, k, v) => {
                if *h == hash && k == key { Some(v) } else { None }
            }
            HamtNode::Collision(h, entries) => {
                if *h != hash {
                    return None;
                }
                entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
            }
            HamtNode::Internal(bitmap, children) => {
                let idx = ((hash >> shift) & (MASK as u64)) as u32;
                let bit = 1u32 << idx;
                if bitmap & bit == 0 {
                    return None;
                }
                let child_idx = (bitmap & (bit - 1)).count_ones() as usize;
                Self::get_in(&children[child_idx], hash, key, shift + BITS)
            }
        }
    }

    /// Insert or update a key-value pair, returning a new map.
    pub fn insert(&self, key: K, value: V) -> Self {
        let h = hash_key(&key);
        let (new_root, added) = Self::insert_in(&self.root, h, key, value, 0);
        Self {
            root: Rc::new(new_root),
            len: if added { self.len + 1 } else { self.len },
        }
    }

    fn insert_in(
        node: &HamtNode<K, V>,
        hash: u64,
        key: K,
        value: V,
        shift: u32,
    ) -> (HamtNode<K, V>, bool) {
        match node {
            HamtNode::Empty => (HamtNode::Leaf(hash, key, value), true),
            HamtNode::Leaf(h, k, _v) => {
                if *h == hash && *k == key {
                    // Update existing.
                    (HamtNode::Leaf(hash, key, value), false)
                } else if *h == hash {
                    // Hash collision.
                    (
                        HamtNode::Collision(hash, vec![(k.clone(), _v.clone()), (key, value)]),
                        true,
                    )
                } else {
                    // Different hashes — expand into internal node.
                    let mut internal = HamtNode::Internal(0, Vec::new());
                    // Re-insert the existing leaf and the new entry.
                    let (int2, _) =
                        Self::insert_in(&internal, *h, k.clone(), _v.clone(), shift);
                    internal = int2;
                    let (int3, _) = Self::insert_in(&internal, hash, key, value, shift);
                    (int3, true)
                }
            }
            HamtNode::Collision(h, entries) => {
                if *h == hash {
                    let mut new_entries = entries.clone();
                    for entry in &mut new_entries {
                        if entry.0 == key {
                            entry.1 = value;
                            return (HamtNode::Collision(hash, new_entries), false);
                        }
                    }
                    new_entries.push((key, value));
                    (HamtNode::Collision(hash, new_entries), true)
                } else {
                    // Expand collision into internal node.
                    let mut internal: HamtNode<K, V> = HamtNode::Internal(0, Vec::new());
                    // Re-insert collision node first.
                    let idx_old = ((*h >> shift) & (MASK as u64)) as u32;
                    let bit_old = 1u32 << idx_old;
                    internal = HamtNode::Internal(bit_old, vec![Rc::new(node.clone())]);
                    let (result, _) = Self::insert_in(&internal, hash, key, value, shift);
                    (result, true)
                }
            }
            HamtNode::Internal(bitmap, children) => {
                let idx = ((hash >> shift) & (MASK as u64)) as u32;
                let bit = 1u32 << idx;
                let child_idx = (bitmap & (bit - 1)).count_ones() as usize;

                if bitmap & bit == 0 {
                    // Slot is empty — add new leaf.
                    let mut new_children = children.clone();
                    new_children.insert(child_idx, Rc::new(HamtNode::Leaf(hash, key, value)));
                    (HamtNode::Internal(bitmap | bit, new_children), true)
                } else {
                    // Slot exists — recurse.
                    let (new_child, added) =
                        Self::insert_in(&children[child_idx], hash, key, value, shift + BITS);
                    let mut new_children = children.clone();
                    new_children[child_idx] = Rc::new(new_child);
                    (HamtNode::Internal(*bitmap, new_children), added)
                }
            }
        }
    }

    /// Remove a key, returning a new map.
    pub fn remove(&self, key: &K) -> Self {
        let h = hash_key(key);
        let (new_root, removed) = Self::remove_in(&self.root, h, key, 0);
        Self {
            root: Rc::new(new_root),
            len: if removed { self.len - 1 } else { self.len },
        }
    }

    fn remove_in(
        node: &HamtNode<K, V>,
        hash: u64,
        key: &K,
        shift: u32,
    ) -> (HamtNode<K, V>, bool) {
        match node {
            HamtNode::Empty => (HamtNode::Empty, false),
            HamtNode::Leaf(h, k, _) => {
                if *h == hash && k == key {
                    (HamtNode::Empty, true)
                } else {
                    (node.clone(), false)
                }
            }
            HamtNode::Collision(h, entries) => {
                if *h != hash {
                    return (node.clone(), false);
                }
                let new_entries: Vec<_> =
                    entries.iter().filter(|(k, _)| k != key).cloned().collect();
                if new_entries.len() == entries.len() {
                    (node.clone(), false)
                } else if new_entries.len() == 1 {
                    let (k, v) = new_entries.into_iter().next().unwrap();
                    (HamtNode::Leaf(hash, k, v), true)
                } else {
                    (HamtNode::Collision(hash, new_entries), true)
                }
            }
            HamtNode::Internal(bitmap, children) => {
                let idx = ((hash >> shift) & (MASK as u64)) as u32;
                let bit = 1u32 << idx;
                if bitmap & bit == 0 {
                    return (node.clone(), false);
                }
                let child_idx = (bitmap & (bit - 1)).count_ones() as usize;
                let (new_child, removed) =
                    Self::remove_in(&children[child_idx], hash, key, shift + BITS);
                if !removed {
                    return (node.clone(), false);
                }
                match new_child {
                    HamtNode::Empty => {
                        let new_bitmap = bitmap & !bit;
                        if new_bitmap == 0 {
                            (HamtNode::Empty, true)
                        } else {
                            let mut new_children = children.clone();
                            new_children.remove(child_idx);
                            (HamtNode::Internal(new_bitmap, new_children), true)
                        }
                    }
                    _ => {
                        let mut new_children = children.clone();
                        new_children[child_idx] = Rc::new(new_child);
                        (HamtNode::Internal(*bitmap, new_children), true)
                    }
                }
            }
        }
    }

    /// Check if the map contains a key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Collect all key-value pairs. Order is NOT guaranteed.
    pub fn entries(&self) -> Vec<(K, V)> {
        let mut result = Vec::with_capacity(self.len);
        Self::collect_entries(&self.root, &mut result);
        result
    }

    fn collect_entries(node: &HamtNode<K, V>, out: &mut Vec<(K, V)>) {
        match node {
            HamtNode::Empty => {}
            HamtNode::Leaf(_, k, v) => out.push((k.clone(), v.clone())),
            HamtNode::Collision(_, entries) => {
                for (k, v) in entries {
                    out.push((k.clone(), v.clone()));
                }
            }
            HamtNode::Internal(_, children) => {
                for child in children {
                    Self::collect_entries(child, out);
                }
            }
        }
    }

    /// Collect all keys. Order is NOT guaranteed.
    pub fn keys(&self) -> Vec<K> {
        self.entries().into_iter().map(|(k, _)| k).collect()
    }

    /// Collect all values. Order is NOT guaranteed.
    pub fn values(&self) -> Vec<V> {
        self.entries().into_iter().map(|(_, v)| v).collect()
    }

    /// Create from an iterator of key-value pairs.
    pub fn from_iter(iter: impl IntoIterator<Item = (K, V)>) -> Self {
        let mut map = Self::new();
        for (k, v) in iter {
            map = map.insert(k, v);
        }
        map
    }

    /// Merge another map into this one, preferring values from `other`.
    pub fn merge(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (k, v) in other.entries() {
            result = result.insert(k, v);
        }
        result
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Default for PersistentMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone + fmt::Debug, V: Clone + fmt::Debug> fmt::Debug
    for PersistentMap<K, V>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self.entries();
        f.debug_map()
            .entries(entries.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

// ── TransientVec ────────────────────────────────────────────────────────────

/// A transient (mutable) mode for bulk-building a `PersistentVec`.
///
/// Allows efficient sequential pushes, then freezes into a persistent vector.
pub struct TransientVec<T: Clone> {
    items: Vec<T>,
}

impl<T: Clone> TransientVec<T> {
    /// Create a new transient vector.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Push an element mutably.
    pub fn push(&mut self, value: T) {
        self.items.push(value);
    }

    /// Current length.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Freeze into a persistent vector.
    pub fn freeze(self) -> PersistentVec<T> {
        PersistentVec::from_slice(&self.items)
    }
}

impl<T: Clone> Default for TransientVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A transient (mutable) mode for bulk-building a `PersistentMap`.
pub struct TransientMap<K: Hash + Eq + Clone, V: Clone> {
    entries: Vec<(K, V)>,
}

impl<K: Hash + Eq + Clone, V: Clone> TransientMap<K, V> {
    /// Create a new transient map.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: K, value: V) {
        // Remove existing if present.
        self.entries.retain(|(k, _)| k != &key);
        self.entries.push((key, value));
    }

    /// Current length.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Freeze into a persistent map.
    pub fn freeze(self) -> PersistentMap<K, V> {
        PersistentMap::from_iter(self.entries)
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Default for TransientMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- PersistentVec ---

    #[test]
    fn pvec_empty() {
        let v: PersistentVec<i32> = PersistentVec::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.get(0), None);
    }

    #[test]
    fn pvec_push_and_get() {
        let v = PersistentVec::new().push(10).push(20).push(30);
        assert_eq!(v.len(), 3);
        assert_eq!(v.get(0), Some(&10));
        assert_eq!(v.get(1), Some(&20));
        assert_eq!(v.get(2), Some(&30));
        assert_eq!(v.get(3), None);
    }

    #[test]
    fn pvec_structural_sharing() {
        let v1 = PersistentVec::new().push(1).push(2);
        let v2 = v1.push(3);
        // v1 is unchanged.
        assert_eq!(v1.len(), 2);
        assert_eq!(v2.len(), 3);
        assert_eq!(v1.get(0), Some(&1));
        assert_eq!(v2.get(2), Some(&3));
    }

    #[test]
    fn pvec_set() {
        let v = PersistentVec::new().push(1).push(2).push(3);
        let v2 = v.set(1, 20).unwrap();
        assert_eq!(v.get(1), Some(&2)); // original unchanged
        assert_eq!(v2.get(1), Some(&20));
    }

    #[test]
    fn pvec_set_out_of_bounds() {
        let v = PersistentVec::new().push(1);
        assert!(v.set(5, 99).is_none());
    }

    #[test]
    fn pvec_to_vec() {
        let v = PersistentVec::new().push(1).push(2).push(3);
        assert_eq!(v.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn pvec_from_slice() {
        let v = PersistentVec::from_slice(&[10, 20, 30]);
        assert_eq!(v.len(), 3);
        assert_eq!(v.to_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn pvec_many_elements() {
        let mut v = PersistentVec::new();
        for i in 0..100 {
            v = v.push(i);
        }
        assert_eq!(v.len(), 100);
        for i in 0..100 {
            assert_eq!(v.get(i), Some(&i));
        }
    }

    #[test]
    fn pvec_map() {
        let v = PersistentVec::from_slice(&[1, 2, 3]);
        let v2 = v.map(|x| x * 10);
        assert_eq!(v2.to_vec(), vec![10, 20, 30]);
    }

    #[test]
    fn pvec_filter() {
        let v = PersistentVec::from_slice(&[1, 2, 3, 4, 5]);
        let v2 = v.filter(|x| x % 2 == 0);
        assert_eq!(v2.to_vec(), vec![2, 4]);
    }

    #[test]
    fn pvec_iter() {
        let v = PersistentVec::from_slice(&[1, 2, 3]);
        let collected: Vec<_> = v.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn pvec_eq() {
        let a = PersistentVec::from_slice(&[1, 2, 3]);
        let b = PersistentVec::from_slice(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn pvec_debug() {
        let v = PersistentVec::from_slice(&[1, 2]);
        let s = format!("{v:?}");
        assert!(s.contains('1'));
        assert!(s.contains('2'));
    }

    // --- PersistentMap ---

    #[test]
    fn pmap_empty() {
        let m: PersistentMap<String, i32> = PersistentMap::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn pmap_insert_and_get() {
        let m = PersistentMap::new()
            .insert("a".to_string(), 1)
            .insert("b".to_string(), 2);
        assert_eq!(m.get(&"a".to_string()), Some(&1));
        assert_eq!(m.get(&"b".to_string()), Some(&2));
        assert_eq!(m.get(&"c".to_string()), None);
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn pmap_structural_sharing() {
        let m1 = PersistentMap::new().insert("x", 10);
        let m2 = m1.insert("y", 20);
        assert_eq!(m1.len(), 1);
        assert_eq!(m2.len(), 2);
        assert_eq!(m1.get(&"x"), Some(&10));
        assert_eq!(m2.get(&"y"), Some(&20));
    }

    #[test]
    fn pmap_update() {
        let m1 = PersistentMap::new().insert("a", 1);
        let m2 = m1.insert("a", 99);
        assert_eq!(m1.get(&"a"), Some(&1));
        assert_eq!(m2.get(&"a"), Some(&99));
        assert_eq!(m2.len(), 1);
    }

    #[test]
    fn pmap_remove() {
        let m = PersistentMap::new()
            .insert("a", 1)
            .insert("b", 2);
        let m2 = m.remove(&"a");
        assert_eq!(m.len(), 2);
        assert_eq!(m2.len(), 1);
        assert_eq!(m2.get(&"a"), None);
        assert_eq!(m2.get(&"b"), Some(&2));
    }

    #[test]
    fn pmap_remove_nonexistent() {
        let m = PersistentMap::new().insert("a", 1);
        let m2 = m.remove(&"z");
        assert_eq!(m2.len(), 1);
    }

    #[test]
    fn pmap_contains_key() {
        let m = PersistentMap::new().insert(42, "hello");
        assert!(m.contains_key(&42));
        assert!(!m.contains_key(&99));
    }

    #[test]
    fn pmap_many_entries() {
        let mut m = PersistentMap::new();
        for i in 0..50 {
            m = m.insert(i, i * 10);
        }
        assert_eq!(m.len(), 50);
        for i in 0..50 {
            assert_eq!(m.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn pmap_entries_keys_values() {
        let m = PersistentMap::new()
            .insert(1, "a")
            .insert(2, "b");
        let entries = m.entries();
        assert_eq!(entries.len(), 2);
        let mut keys = m.keys();
        keys.sort();
        assert_eq!(keys, vec![1, 2]);
    }

    #[test]
    fn pmap_merge() {
        let m1 = PersistentMap::new().insert("a", 1).insert("b", 2);
        let m2 = PersistentMap::new().insert("b", 20).insert("c", 30);
        let merged = m1.merge(&m2);
        assert_eq!(merged.get(&"a"), Some(&1));
        assert_eq!(merged.get(&"b"), Some(&20)); // m2 wins
        assert_eq!(merged.get(&"c"), Some(&30));
    }

    // --- Transient ---

    #[test]
    fn transient_vec_freeze() {
        let mut t = TransientVec::new();
        t.push(1);
        t.push(2);
        t.push(3);
        assert_eq!(t.len(), 3);
        let v = t.freeze();
        assert_eq!(v.to_vec(), vec![1, 2, 3]);
    }

    #[test]
    fn transient_map_freeze() {
        let mut t = TransientMap::new();
        t.insert("a", 1);
        t.insert("b", 2);
        assert_eq!(t.len(), 2);
        let m = t.freeze();
        assert_eq!(m.get(&"a"), Some(&1));
        assert_eq!(m.get(&"b"), Some(&2));
    }

    #[test]
    fn transient_map_overwrite() {
        let mut t = TransientMap::new();
        t.insert("a", 1);
        t.insert("a", 99);
        assert_eq!(t.len(), 1);
        let m = t.freeze();
        assert_eq!(m.get(&"a"), Some(&99));
    }
}
