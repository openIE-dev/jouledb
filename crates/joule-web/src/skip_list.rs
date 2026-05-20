//! Skip list — probabilistic balanced ordered data structure.
//!
//! O(log n) insert, remove, and search. Supports ordered iteration,
//! range queries, floor/ceiling, and rank operations.

use std::fmt;

const MAX_LEVEL: usize = 16;

// ── Node ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Node<K: Ord, V> {
    key: K,
    value: V,
    forward: Vec<Option<usize>>, // indices into the arena
}

// ── SkipList ────────────────────────────────────────────────────────────────

/// A skip list providing O(log n) search, insert, and removal on ordered keys.
pub struct SkipList<K: Ord, V> {
    arena: Vec<Option<Node<K, V>>>,
    head_forward: Vec<Option<usize>>,
    level: usize,
    len: usize,
    rng_state: u64,
}

impl<K: Ord + fmt::Debug, V: fmt::Debug> fmt::Debug for SkipList<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SkipList")
            .field("len", &self.len)
            .field("level", &self.level)
            .finish()
    }
}

impl<K: Ord, V> SkipList<K, V> {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            head_forward: vec![None; MAX_LEVEL],
            level: 0,
            len: 0,
            rng_state: 0x12345678_9ABCDEF0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn random_level(&mut self) -> usize {
        let mut lvl = 0;
        // xorshift64
        loop {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            if self.rng_state & 1 == 0 && lvl + 1 < MAX_LEVEL {
                lvl += 1;
            } else {
                break;
            }
        }
        lvl
    }

    fn alloc_node(&mut self, node: Node<K, V>) -> usize {
        // Reuse freed slots
        for (i, slot) in self.arena.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(node);
                return i;
            }
        }
        let idx = self.arena.len();
        self.arena.push(Some(node));
        idx
    }

    /// Insert a key-value pair. If the key already exists, update the value.
    pub fn insert(&mut self, key: K, value: V) {
        let mut update = vec![None::<usize>; MAX_LEVEL];
        let mut current_fwd = self.head_forward.clone();
        let mut last_node = None::<usize>;

        for i in (0..=self.level).rev() {
            loop {
                if let Some(idx) = current_fwd.get(i).and_then(|x| *x) {
                    let node = self.arena[idx].as_ref().unwrap();
                    if node.key < key {
                        current_fwd = node.forward.clone();
                        last_node = Some(idx);
                        continue;
                    } else if node.key == key {
                        // Update existing
                        self.arena[idx].as_mut().unwrap().value = value;
                        return;
                    }
                }
                break;
            }
            update[i] = last_node;
        }

        let new_level = self.random_level();
        let new_node_forward = vec![None; new_level + 1];
        let node = Node {
            key,
            value,
            forward: new_node_forward,
        };
        let new_idx = self.alloc_node(node);

        if new_level > self.level {
            self.level = new_level;
        }

        for i in 0..=new_level {
            let prev_next = match update[i] {
                Some(prev_idx) => {
                    let prev = self.arena[prev_idx].as_mut().unwrap();
                    while prev.forward.len() <= i {
                        prev.forward.push(None);
                    }
                    let old = prev.forward[i];
                    prev.forward[i] = Some(new_idx);
                    old
                }
                None => {
                    let old = self.head_forward[i];
                    self.head_forward[i] = Some(new_idx);
                    old
                }
            };
            self.arena[new_idx].as_mut().unwrap().forward[i] = prev_next;
        }

        self.len += 1;
    }

    /// Search for a key, returning a reference to its value.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut current_fwd = &self.head_forward;

        for i in (0..=self.level).rev() {
            loop {
                if let Some(idx) = current_fwd.get(i).and_then(|x| *x) {
                    let node = self.arena[idx].as_ref().unwrap();
                    if node.key < *key {
                        current_fwd = &node.forward;
                        continue;
                    } else if node.key == *key {
                        return Some(&node.value);
                    }
                }
                break;
            }
        }
        None
    }

    /// Check if a key exists in the skip list.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Remove a key, returning the value if it existed.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut update = vec![None::<usize>; MAX_LEVEL];
        let mut current_fwd = self.head_forward.clone();
        let mut target = None;

        for i in (0..=self.level).rev() {
            loop {
                if let Some(idx) = current_fwd[i] {
                    let node = self.arena[idx].as_ref().unwrap();
                    if node.key < *key {
                        update[i] = Some(idx);
                        current_fwd = node.forward.clone();
                        continue;
                    } else if node.key == *key {
                        target = Some(idx);
                    }
                }
                break;
            }
        }

        let target_idx = target?;

        let node_forward = self.arena[target_idx].as_ref().unwrap().forward.clone();
        for i in 0..node_forward.len() {
            match update[i] {
                Some(prev_idx) => {
                    let prev = self.arena[prev_idx].as_mut().unwrap();
                    if i < prev.forward.len() {
                        prev.forward[i] = node_forward[i];
                    }
                }
                None => {
                    if i < self.head_forward.len() {
                        self.head_forward[i] = node_forward[i];
                    }
                }
            }
        }

        let removed = self.arena[target_idx].take().unwrap();
        self.len -= 1;

        // Reduce level if needed
        while self.level > 0 && self.head_forward[self.level].is_none() {
            self.level -= 1;
        }

        Some(removed.value)
    }

    /// Iterate all key-value pairs in sorted order.
    pub fn iter(&self) -> SkipListIter<'_, K, V> {
        SkipListIter {
            list: self,
            current: self.head_forward[0],
        }
    }

    /// Range query: return all entries with keys in `[low, high]`.
    pub fn range(&self, low: &K, high: &K) -> Vec<(&K, &V)> {
        let mut result = Vec::new();
        // Find first node >= low
        let mut current_fwd = &self.head_forward;
        let mut start = None;

        for i in (0..=self.level).rev() {
            loop {
                if let Some(idx) = current_fwd.get(i).and_then(|x| *x) {
                    let node = self.arena[idx].as_ref().unwrap();
                    if node.key < *low {
                        current_fwd = &node.forward;
                        continue;
                    }
                }
                break;
            }
        }
        // current_fwd[0] should be first >= low, or a node whose forward[0] is first >= low
        if let Some(idx) = current_fwd.get(0).and_then(|x| *x) {
            let node = self.arena[idx].as_ref().unwrap();
            if node.key >= *low {
                start = Some(idx);
            }
        }

        let mut cur = start;
        while let Some(idx) = cur {
            let node = self.arena[idx].as_ref().unwrap();
            if node.key > *high {
                break;
            }
            result.push((&node.key, &node.value));
            cur = node.forward.first().and_then(|x| *x);
        }
        result
    }

    /// Floor: largest key <= given key.
    pub fn floor(&self, key: &K) -> Option<(&K, &V)> {
        let mut best = None;
        let mut current_fwd = &self.head_forward;

        for i in (0..=self.level).rev() {
            loop {
                if let Some(idx) = current_fwd.get(i).and_then(|x| *x) {
                    let node = self.arena[idx].as_ref().unwrap();
                    if node.key <= *key {
                        best = Some(idx);
                        current_fwd = &node.forward;
                        continue;
                    }
                }
                break;
            }
        }
        best.map(|idx| {
            let node = self.arena[idx].as_ref().unwrap();
            (&node.key, &node.value)
        })
    }

    /// Ceiling: smallest key >= given key.
    pub fn ceiling(&self, key: &K) -> Option<(&K, &V)> {
        let mut current_fwd = &self.head_forward;

        for i in (0..=self.level).rev() {
            loop {
                if let Some(idx) = current_fwd.get(i).and_then(|x| *x) {
                    let node = self.arena[idx].as_ref().unwrap();
                    if node.key < *key {
                        current_fwd = &node.forward;
                        continue;
                    }
                }
                break;
            }
        }
        if let Some(idx) = current_fwd.get(0).and_then(|x| *x) {
            let node = self.arena[idx].as_ref().unwrap();
            if node.key >= *key {
                return Some((&node.key, &node.value));
            }
        }
        None
    }

    /// Rank: number of keys strictly less than given key.
    pub fn rank(&self, key: &K) -> usize {
        let mut count = 0;
        let mut cur = self.head_forward[0];
        while let Some(idx) = cur {
            let node = self.arena[idx].as_ref().unwrap();
            if node.key < *key {
                count += 1;
                cur = node.forward.first().and_then(|x| *x);
            } else {
                break;
            }
        }
        count
    }
}

impl<K: Ord, V> Default for SkipList<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Iterator ────────────────────────────────────────────────────────────────

pub struct SkipListIter<'a, K: Ord, V> {
    list: &'a SkipList<K, V>,
    current: Option<usize>,
}

impl<'a, K: Ord, V> Iterator for SkipListIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current?;
        let node = self.list.arena[idx].as_ref().unwrap();
        self.current = node.forward.first().and_then(|x| *x);
        Some((&node.key, &node.value))
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut sl = SkipList::new();
        sl.insert(5, "five");
        sl.insert(3, "three");
        sl.insert(7, "seven");
        assert_eq!(sl.get(&5), Some(&"five"));
        assert_eq!(sl.get(&3), Some(&"three"));
        assert_eq!(sl.get(&7), Some(&"seven"));
        assert_eq!(sl.get(&1), None);
    }

    #[test]
    fn test_len_and_empty() {
        let mut sl = SkipList::new();
        assert!(sl.is_empty());
        sl.insert(1, ());
        assert_eq!(sl.len(), 1);
        assert!(!sl.is_empty());
    }

    #[test]
    fn test_update_existing() {
        let mut sl = SkipList::new();
        sl.insert(1, "old");
        sl.insert(1, "new");
        assert_eq!(sl.get(&1), Some(&"new"));
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut sl = SkipList::new();
        sl.insert(10, "ten");
        sl.insert(20, "twenty");
        assert_eq!(sl.remove(&10), Some("ten"));
        assert_eq!(sl.get(&10), None);
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut sl: SkipList<i32, i32> = SkipList::new();
        assert_eq!(sl.remove(&42), None);
    }

    #[test]
    fn test_ordered_iteration() {
        let mut sl = SkipList::new();
        sl.insert(30, "c");
        sl.insert(10, "a");
        sl.insert(20, "b");
        let keys: Vec<_> = sl.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![10, 20, 30]);
    }

    #[test]
    fn test_range_query() {
        let mut sl = SkipList::new();
        for i in 0..20 {
            sl.insert(i, i * 10);
        }
        let range: Vec<_> = sl.range(&5, &10).iter().map(|(k, _)| **k).collect();
        assert_eq!(range, vec![5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_floor() {
        let mut sl = SkipList::new();
        sl.insert(10, ());
        sl.insert(20, ());
        sl.insert(30, ());
        assert_eq!(sl.floor(&25).map(|(k, _)| *k), Some(20));
        assert_eq!(sl.floor(&20).map(|(k, _)| *k), Some(20));
        assert_eq!(sl.floor(&5), None);
    }

    #[test]
    fn test_ceiling() {
        let mut sl = SkipList::new();
        sl.insert(10, ());
        sl.insert(20, ());
        sl.insert(30, ());
        assert_eq!(sl.ceiling(&15).map(|(k, _)| *k), Some(20));
        assert_eq!(sl.ceiling(&20).map(|(k, _)| *k), Some(20));
        assert_eq!(sl.ceiling(&35), None);
    }

    #[test]
    fn test_rank() {
        let mut sl = SkipList::new();
        sl.insert(10, ());
        sl.insert(20, ());
        sl.insert(30, ());
        assert_eq!(sl.rank(&10), 0);
        assert_eq!(sl.rank(&20), 1);
        assert_eq!(sl.rank(&25), 2);
        assert_eq!(sl.rank(&30), 2);
        assert_eq!(sl.rank(&100), 3);
    }

    #[test]
    fn test_contains_key() {
        let mut sl = SkipList::new();
        sl.insert(42, "answer");
        assert!(sl.contains_key(&42));
        assert!(!sl.contains_key(&0));
    }

    #[test]
    fn test_many_inserts() {
        let mut sl = SkipList::new();
        for i in (0..100).rev() {
            sl.insert(i, i);
        }
        assert_eq!(sl.len(), 100);
        for i in 0..100 {
            assert_eq!(sl.get(&i), Some(&i));
        }
        let keys: Vec<_> = sl.iter().map(|(k, _)| *k).collect();
        let expected: Vec<_> = (0..100).collect();
        assert_eq!(keys, expected);
    }
}
