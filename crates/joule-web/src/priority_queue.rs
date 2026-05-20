//! Priority queue — binary heap with min/max modes, decrease-key, bounded capacity.
//!
//! Replaces JavaScript's lack of a native priority queue with a pure Rust
//! binary heap supporting both min-heap and max-heap, stable ordering for
//! equal priorities, peek, merge, drain, decrease/increase key, and bounded
//! capacity with eviction.

use std::collections::HashMap;

// ── Heap Mode ──────────────────────────────────────────────────

/// Whether the heap is min-oriented or max-oriented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapMode {
    Min,
    Max,
}

// ── Entry ──────────────────────────────────────────────────────

/// A heap entry with a key (priority) and value, plus insertion order for stability.
#[derive(Debug, Clone)]
struct Entry<V> {
    key: i64,
    value: V,
    seq: u64,
}

// ── PriorityQueue ──────────────────────────────────────────────

/// Binary heap priority queue.
#[derive(Debug, Clone)]
pub struct PriorityQueue<V: Clone + Eq + std::hash::Hash> {
    heap: Vec<Entry<V>>,
    mode: HeapMode,
    seq_counter: u64,
    capacity: Option<usize>,
    /// Map from value → index in heap for O(log n) decrease-key.
    index_map: HashMap<V, usize>,
}

impl<V: Clone + Eq + std::hash::Hash + std::fmt::Debug> PriorityQueue<V> {
    pub fn new(mode: HeapMode) -> Self {
        Self {
            heap: Vec::new(),
            mode,
            seq_counter: 0,
            capacity: None,
            index_map: HashMap::new(),
        }
    }

    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.capacity = Some(cap);
        self
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    pub fn mode(&self) -> HeapMode {
        self.mode
    }

    /// Returns true if `a` should be closer to the root than `b`.
    fn has_higher_priority(&self, a: &Entry<V>, b: &Entry<V>) -> bool {
        match self.mode {
            HeapMode::Min => {
                if a.key != b.key {
                    a.key < b.key
                } else {
                    a.seq < b.seq // stable: earlier insertion wins
                }
            }
            HeapMode::Max => {
                if a.key != b.key {
                    a.key > b.key
                } else {
                    a.seq < b.seq
                }
            }
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
        // Update index map.
        let vi = self.heap[i].value.clone();
        let vj = self.heap[j].value.clone();
        self.index_map.insert(vi, i);
        self.index_map.insert(vj, j);
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.has_higher_priority(&self.heap[idx], &self.heap[parent]) {
                self.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.heap.len();
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut best = idx;
            if left < len && self.has_higher_priority(&self.heap[left], &self.heap[best]) {
                best = left;
            }
            if right < len && self.has_higher_priority(&self.heap[right], &self.heap[best]) {
                best = right;
            }
            if best == idx {
                break;
            }
            self.swap(idx, best);
            idx = best;
        }
    }

    /// Push a value with the given priority key.
    /// If bounded and at capacity, evicts the lowest-priority element.
    /// Returns the evicted element, if any.
    pub fn push(&mut self, key: i64, value: V) -> Option<(i64, V)> {
        let mut evicted = None;

        if let Some(cap) = self.capacity {
            if cap == 0 {
                return Some((key, value));
            }
            if self.heap.len() >= cap {
                // Evict the element furthest from root (worst priority).
                // Find it by scanning leaves.
                let worst_idx = self.find_worst_idx();
                if let Some(wi) = worst_idx {
                    // Only evict if new item is better than worst.
                    let candidate = Entry {
                        key,
                        value: value.clone(),
                        seq: self.seq_counter,
                    };
                    if self.has_higher_priority(&candidate, &self.heap[wi]) {
                        let removed = self.remove_at(wi);
                        evicted = Some((removed.key, removed.value));
                    } else {
                        return Some((key, value));
                    }
                }
            }
        }

        let entry = Entry {
            key,
            value: value.clone(),
            seq: self.seq_counter,
        };
        self.seq_counter += 1;
        let idx = self.heap.len();
        self.heap.push(entry);
        self.index_map.insert(value, idx);
        self.sift_up(idx);
        evicted
    }

    /// Find the index of the worst-priority element.
    fn find_worst_idx(&self) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        let mut worst = 0;
        for i in 1..self.heap.len() {
            if self.has_higher_priority(&self.heap[worst], &self.heap[i]) {
                worst = i;
            }
        }
        Some(worst)
    }

    fn remove_at(&mut self, idx: usize) -> Entry<V> {
        let last = self.heap.len() - 1;
        self.swap(idx, last);
        let removed = self.heap.pop().unwrap();
        self.index_map.remove(&removed.value);
        if idx < self.heap.len() {
            self.sift_down(idx);
            self.sift_up(idx);
        }
        removed
    }

    /// Peek at the highest-priority element without removing it.
    pub fn peek(&self) -> Option<(i64, &V)> {
        self.heap.first().map(|e| (e.key, &e.value))
    }

    /// Pop the highest-priority element.
    pub fn pop(&mut self) -> Option<(i64, V)> {
        if self.heap.is_empty() {
            return None;
        }
        let entry = self.remove_at(0);
        Some((entry.key, entry.value))
    }

    /// Change the key of an existing value. Returns true if found.
    pub fn change_key(&mut self, value: &V, new_key: i64) -> bool {
        let idx = match self.index_map.get(value) {
            Some(i) => *i,
            None => return false,
        };
        self.heap[idx].key = new_key;
        self.sift_up(idx);
        // idx may have changed after sift_up; find current index.
        let idx = self.index_map[value];
        self.sift_down(idx);
        true
    }

    /// Decrease key (only changes if new_key is smaller).
    pub fn decrease_key(&mut self, value: &V, new_key: i64) -> bool {
        if let Some(idx) = self.index_map.get(value) {
            if new_key < self.heap[*idx].key {
                return self.change_key(value, new_key);
            }
        }
        false
    }

    /// Increase key (only changes if new_key is larger).
    pub fn increase_key(&mut self, value: &V, new_key: i64) -> bool {
        if let Some(idx) = self.index_map.get(value) {
            if new_key > self.heap[*idx].key {
                return self.change_key(value, new_key);
            }
        }
        false
    }

    /// Drain all elements in priority order.
    pub fn drain_sorted(&mut self) -> Vec<(i64, V)> {
        let mut out = Vec::with_capacity(self.heap.len());
        while let Some(item) = self.pop() {
            out.push(item);
        }
        out
    }

    /// Merge another heap into this one.
    pub fn merge(&mut self, other: &mut PriorityQueue<V>) {
        while let Some((key, value)) = other.pop() {
            self.push(key, value);
        }
    }

    /// Check if a value exists in the queue.
    pub fn contains(&self, value: &V) -> bool {
        self.index_map.contains_key(value)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_heap_basic() {
        let mut pq = PriorityQueue::new(HeapMode::Min);
        pq.push(5, "five");
        pq.push(1, "one");
        pq.push(3, "three");
        assert_eq!(pq.peek(), Some((1, &"one")));
        assert_eq!(pq.pop(), Some((1, "one")));
        assert_eq!(pq.pop(), Some((3, "three")));
        assert_eq!(pq.pop(), Some((5, "five")));
        assert!(pq.is_empty());
    }

    #[test]
    fn test_max_heap_basic() {
        let mut pq = PriorityQueue::new(HeapMode::Max);
        pq.push(5, "five");
        pq.push(1, "one");
        pq.push(3, "three");
        assert_eq!(pq.pop(), Some((5, "five")));
        assert_eq!(pq.pop(), Some((3, "three")));
        assert_eq!(pq.pop(), Some((1, "one")));
    }

    #[test]
    fn test_stable_ordering() {
        let mut pq = PriorityQueue::new(HeapMode::Min);
        pq.push(1, "first");
        pq.push(1, "second");
        pq.push(1, "third");
        assert_eq!(pq.pop().unwrap().1, "first");
        assert_eq!(pq.pop().unwrap().1, "second");
        assert_eq!(pq.pop().unwrap().1, "third");
    }

    #[test]
    fn test_decrease_key() {
        let mut pq = PriorityQueue::new(HeapMode::Min);
        pq.push(10, "a");
        pq.push(20, "b");
        pq.push(30, "c");
        assert!(pq.decrease_key(&"c", 5));
        assert_eq!(pq.peek(), Some((5, &"c")));
    }

    #[test]
    fn test_increase_key() {
        let mut pq = PriorityQueue::new(HeapMode::Max);
        pq.push(10, "a");
        pq.push(20, "b");
        assert!(pq.increase_key(&"a", 50));
        assert_eq!(pq.peek(), Some((50, &"a")));
    }

    #[test]
    fn test_bounded_capacity_eviction() {
        let mut pq = PriorityQueue::new(HeapMode::Min).with_capacity(3);
        pq.push(5, "e");
        pq.push(3, "c");
        pq.push(1, "a");
        // At capacity. Push a better element — worst (5) should be evicted.
        let evicted = pq.push(2, "b");
        assert_eq!(evicted, Some((5, "e")));
        assert_eq!(pq.len(), 3);
        // Push a worse element — should be rejected.
        let evicted = pq.push(10, "z");
        assert_eq!(evicted, Some((10, "z")));
    }

    #[test]
    fn test_drain_sorted() {
        let mut pq = PriorityQueue::new(HeapMode::Min);
        pq.push(3, "c");
        pq.push(1, "a");
        pq.push(2, "b");
        let drained = pq.drain_sorted();
        let keys: Vec<i64> = drained.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3]);
        assert!(pq.is_empty());
    }

    #[test]
    fn test_merge() {
        let mut pq1 = PriorityQueue::new(HeapMode::Min);
        pq1.push(1, "a");
        pq1.push(3, "c");
        let mut pq2 = PriorityQueue::new(HeapMode::Min);
        pq2.push(2, "b");
        pq2.push(4, "d");
        pq1.merge(&mut pq2);
        assert!(pq2.is_empty());
        assert_eq!(pq1.len(), 4);
        let drained = pq1.drain_sorted();
        let keys: Vec<i64> = drained.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_peek_without_pop() {
        let mut pq = PriorityQueue::new(HeapMode::Max);
        pq.push(42, "answer");
        assert_eq!(pq.peek(), Some((42, &"answer")));
        assert_eq!(pq.len(), 1); // Not removed.
        assert_eq!(pq.peek(), Some((42, &"answer")));
    }

    #[test]
    fn test_contains() {
        let mut pq = PriorityQueue::new(HeapMode::Min);
        pq.push(1, "x");
        assert!(pq.contains(&"x"));
        assert!(!pq.contains(&"y"));
    }

    #[test]
    fn test_empty_operations() {
        let mut pq: PriorityQueue<&str> = PriorityQueue::new(HeapMode::Min);
        assert!(pq.is_empty());
        assert_eq!(pq.peek(), None);
        assert_eq!(pq.pop(), None);
        assert!(!pq.decrease_key(&"x", 0));
    }
}
