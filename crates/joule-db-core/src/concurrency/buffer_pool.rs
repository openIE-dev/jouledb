//! Buffer pool with LRU eviction
//!
//! A thread-safe buffer pool that:
//! - Caches pages/nodes with Arc for zero-copy sharing
//! - Uses LRU eviction when capacity is exceeded
//! - Provides sharded locking for reduced contention
//!
//! # Design
//!
//! The buffer pool uses a simple but effective design:
//! - HashMap for O(1) lookups by page ID
//! - Doubly-linked list for O(1) LRU updates
//! - RwLock per shard (configurable shards, default 16)
//!
//! # Example
//!
//! ```rust,ignore
//! use joule_db_core::concurrency::{BufferPool, BufferPoolConfig};
//!
//! let pool = BufferPool::new(BufferPoolConfig::default());
//! pool.insert(1, Arc::new(node));
//! let node = pool.get(1);
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

/// Configuration for the buffer pool
#[derive(Clone, Debug)]
pub struct BufferPoolConfig {
    /// Maximum number of entries per shard
    pub capacity_per_shard: usize,
    /// Number of shards (must be power of 2)
    pub num_shards: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            capacity_per_shard: 256, // 256 * 16 = 4096 total entries
            num_shards: 16,
        }
    }
}

impl BufferPoolConfig {
    /// Create a small config for testing
    pub fn small() -> Self {
        Self {
            capacity_per_shard: 16,
            num_shards: 4,
        }
    }

    /// Create a large config for production
    pub fn large() -> Self {
        Self {
            capacity_per_shard: 1024,
            num_shards: 32,
        }
    }

    /// Total capacity across all shards
    pub fn total_capacity(&self) -> usize {
        self.capacity_per_shard * self.num_shards
    }
}

/// A node in the LRU linked list
struct LruNode<K, V> {
    /// The key this node represents (stored for consistency verification)
    key: K,
    value: Arc<V>,
    prev: Option<K>,
    next: Option<K>,
}

impl<K: Copy, V> LruNode<K, V> {
    /// Get the key stored in this node
    fn key(&self) -> K {
        self.key
    }
}

/// A single shard of the buffer pool
struct Shard<K, V>
where
    K: Eq + Hash + Copy,
{
    /// Key -> LruNode mapping
    map: HashMap<K, LruNode<K, V>>,
    /// Head of LRU list (most recently used)
    head: Option<K>,
    /// Tail of LRU list (least recently used)
    tail: Option<K>,
    /// Maximum capacity
    capacity: usize,
}

impl<K, V> Shard<K, V>
where
    K: Eq + Hash + Copy,
{
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            capacity,
        }
    }

    /// Get a value and move it to the front (most recently used)
    fn get(&mut self, key: &K) -> Option<Arc<V>> {
        if !self.map.contains_key(key) {
            return None;
        }

        // Move to front
        self.move_to_front(*key);

        self.map.get(key).map(|node| Arc::clone(&node.value))
    }

    /// Insert or update a value
    fn insert(&mut self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        if let Some(node) = self.map.get_mut(&key) {
            // Update existing
            let old = std::mem::replace(&mut node.value, value);
            self.move_to_front(key);
            return Some(old);
        }

        // Evict if necessary
        while self.map.len() >= self.capacity {
            self.evict_lru();
        }

        // Insert new node at front
        let node = LruNode {
            key,
            value,
            prev: None,
            next: self.head,
        };

        // Update old head's prev pointer
        if let Some(old_head) = self.head {
            if let Some(old_head_node) = self.map.get_mut(&old_head) {
                old_head_node.prev = Some(key);
            }
        }

        self.map.insert(key, node);
        self.head = Some(key);

        if self.tail.is_none() {
            self.tail = Some(key);
        }

        None
    }

    /// Remove a value
    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let node = self.map.remove(key)?;

        // Update links
        if let Some(prev_key) = node.prev {
            if let Some(prev_node) = self.map.get_mut(&prev_key) {
                prev_node.next = node.next;
            }
        } else {
            self.head = node.next;
        }

        if let Some(next_key) = node.next {
            if let Some(next_node) = self.map.get_mut(&next_key) {
                next_node.prev = node.prev;
            }
        } else {
            self.tail = node.prev;
        }

        Some(node.value)
    }

    /// Move a key to the front of the LRU list
    fn move_to_front(&mut self, key: K) {
        if self.head == Some(key) {
            return; // Already at front
        }

        // Get current links
        let (prev, next) = {
            let node = match self.map.get(&key) {
                Some(n) => n,
                None => return,
            };
            (node.prev, node.next)
        };

        // Update previous node's next pointer
        if let Some(prev_key) = prev {
            if let Some(prev_node) = self.map.get_mut(&prev_key) {
                prev_node.next = next;
            }
        }

        // Update next node's prev pointer
        if let Some(next_key) = next {
            if let Some(next_node) = self.map.get_mut(&next_key) {
                next_node.prev = prev;
            }
        } else {
            // This was the tail
            self.tail = prev;
        }

        // Update old head's prev pointer
        if let Some(old_head) = self.head {
            if let Some(old_head_node) = self.map.get_mut(&old_head) {
                old_head_node.prev = Some(key);
            }
        }

        // Move node to front
        if let Some(node) = self.map.get_mut(&key) {
            node.prev = None;
            node.next = self.head;
        }

        self.head = Some(key);
    }

    /// Evict the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(tail_key) = self.tail {
            self.remove(&tail_key);
        }
    }

    /// Get current size
    fn len(&self) -> usize {
        self.map.len()
    }

    /// Clear all entries
    fn clear(&mut self) {
        self.map.clear();
        self.head = None;
        self.tail = None;
    }
}

/// Thread-safe buffer pool with sharded locking
///
/// Uses sharding to reduce lock contention on multi-core systems.
/// Each shard has its own LRU list and is protected by a separate RwLock.
pub struct BufferPool<K, V>
where
    K: Eq + Hash + Copy,
{
    shards: Vec<RwLock<Shard<K, V>>>,
    shard_mask: usize,
}

impl<K, V> BufferPool<K, V>
where
    K: Eq + Hash + Copy,
{
    /// Create a new buffer pool with the given configuration
    pub fn new(config: BufferPoolConfig) -> Self {
        // Ensure num_shards is a power of 2
        let num_shards = config.num_shards.next_power_of_two();
        let shard_mask = num_shards - 1;

        let shards = (0..num_shards)
            .map(|_| RwLock::new(Shard::new(config.capacity_per_shard)))
            .collect();

        Self { shards, shard_mask }
    }

    /// Create a buffer pool with default configuration
    pub fn with_defaults() -> Self {
        Self::new(BufferPoolConfig::default())
    }

    /// Get the shard index for a key
    fn shard_index(&self, key: &K) -> usize
    where
        K: Hash,
    {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) & self.shard_mask
    }

    /// Get a value from the pool
    ///
    /// Returns `Some(Arc<V>)` if found, `None` otherwise.
    /// Updates the LRU position of the entry.
    pub fn get(&self, key: &K) -> Option<Arc<V>>
    where
        K: Hash,
    {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx]
            .write()
            .expect("lock poisoned: buffer pool shard write");
        shard.get(key)
    }

    /// Try to get a value without updating LRU position (read-only)
    ///
    /// This is faster but doesn't update access patterns.
    /// Useful for checking if a key exists.
    pub fn peek(&self, key: &K) -> Option<Arc<V>>
    where
        K: Hash,
    {
        let idx = self.shard_index(key);
        let shard = self.shards[idx]
            .read()
            .expect("lock poisoned: buffer pool shard read");
        shard.map.get(key).map(|node| Arc::clone(&node.value))
    }

    /// Insert a value into the pool
    ///
    /// If the key already exists, updates the value and returns the old one.
    /// May evict LRU entries if capacity is exceeded.
    pub fn insert(&self, key: K, value: Arc<V>) -> Option<Arc<V>>
    where
        K: Hash,
    {
        let idx = self.shard_index(&key);
        let mut shard = self.shards[idx]
            .write()
            .expect("lock poisoned: buffer pool shard write");
        shard.insert(key, value)
    }

    /// Insert a value only if it doesn't already exist
    ///
    /// Returns `true` if inserted, `false` if key already existed.
    pub fn insert_if_absent(&self, key: K, value: Arc<V>) -> bool
    where
        K: Hash,
    {
        let idx = self.shard_index(&key);
        let mut shard = self.shards[idx]
            .write()
            .expect("lock poisoned: buffer pool shard write");
        if shard.map.contains_key(&key) {
            false
        } else {
            shard.insert(key, value);
            true
        }
    }

    /// Remove a value from the pool
    pub fn remove(&self, key: &K) -> Option<Arc<V>>
    where
        K: Hash,
    {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx]
            .write()
            .expect("lock poisoned: buffer pool shard write");
        shard.remove(key)
    }

    /// Check if a key exists in the pool
    pub fn contains(&self, key: &K) -> bool
    where
        K: Hash,
    {
        let idx = self.shard_index(key);
        let shard = self.shards[idx]
            .read()
            .expect("lock poisoned: buffer pool shard read");
        shard.map.contains_key(key)
    }

    /// Get the total number of entries across all shards
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                s.read()
                    .expect("lock poisoned: buffer pool shard read")
                    .len()
            })
            .sum()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries from the pool
    pub fn clear(&self) {
        for shard in &self.shards {
            shard
                .write()
                .expect("lock poisoned: buffer pool shard write")
                .clear();
        }
    }

    /// Resize capacity per shard. Evicts excess LRU entries immediately.
    ///
    /// Used by the adaptive controller to free memory under pressure.
    pub fn resize(&self, new_capacity_per_shard: usize) {
        let capped = new_capacity_per_shard.max(1); // minimum 1
        for shard_lock in &self.shards {
            let mut shard = shard_lock
                .write()
                .expect("lock poisoned: buffer pool shard write");
            shard.capacity = capped;
            // Evict excess entries
            while shard.map.len() > capped {
                shard.evict_lru();
            }
        }
    }

    /// Get current capacity per shard.
    pub fn capacity_per_shard(&self) -> usize {
        self.shards
            .first()
            .and_then(|s| s.read().ok().map(|s| s.capacity))
            .unwrap_or(0)
    }

    /// Get or insert a value
    ///
    /// If the key exists, returns the existing value.
    /// Otherwise, calls the provided function to create a new value.
    pub fn get_or_insert_with<F>(&self, key: K, f: F) -> Arc<V>
    where
        K: Hash,
        F: FnOnce() -> V,
    {
        let idx = self.shard_index(&key);
        let mut shard = self.shards[idx]
            .write()
            .expect("lock poisoned: buffer pool shard write");

        if let Some(value) = shard.get(&key) {
            return value;
        }

        let value = Arc::new(f());
        shard.insert(key, Arc::clone(&value));
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let pool: BufferPool<u64, String> = BufferPool::new(BufferPoolConfig::small());

        // Insert
        assert!(pool.insert(1, Arc::new("one".to_string())).is_none());
        assert!(pool.insert(2, Arc::new("two".to_string())).is_none());

        // Get
        assert_eq!(*pool.get(&1).unwrap(), "one");
        assert_eq!(*pool.get(&2).unwrap(), "two");
        assert!(pool.get(&3).is_none());

        // Update
        let old = pool.insert(1, Arc::new("ONE".to_string()));
        assert_eq!(*old.unwrap(), "one");
        assert_eq!(*pool.get(&1).unwrap(), "ONE");

        // Remove
        let removed = pool.remove(&1);
        assert_eq!(*removed.unwrap(), "ONE");
        assert!(pool.get(&1).is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let config = BufferPoolConfig {
            capacity_per_shard: 3,
            num_shards: 1,
        };
        let pool: BufferPool<u64, i32> = BufferPool::new(config);

        // Fill to capacity
        pool.insert(1, Arc::new(1));
        pool.insert(2, Arc::new(2));
        pool.insert(3, Arc::new(3));
        assert_eq!(pool.len(), 3);

        // Access 1 to make it recently used
        pool.get(&1);

        // Insert 4, should evict 2 (least recently used)
        pool.insert(4, Arc::new(4));
        assert_eq!(pool.len(), 3);

        assert!(pool.get(&1).is_some()); // Still there
        assert!(pool.get(&2).is_none()); // Evicted
        assert!(pool.get(&3).is_some()); // Still there
        assert!(pool.get(&4).is_some()); // New entry
    }

    #[test]
    fn test_peek_no_lru_update() {
        let config = BufferPoolConfig {
            capacity_per_shard: 2,
            num_shards: 1,
        };
        let pool: BufferPool<u64, i32> = BufferPool::new(config);

        pool.insert(1, Arc::new(1));
        pool.insert(2, Arc::new(2));

        // Peek at 1 (doesn't update LRU)
        assert!(pool.peek(&1).is_some());

        // Insert 3, should evict 1 (still LRU because peek doesn't update)
        pool.insert(3, Arc::new(3));

        assert!(pool.peek(&1).is_none()); // Evicted
        assert!(pool.peek(&2).is_some());
        assert!(pool.peek(&3).is_some());
    }

    #[test]
    fn test_get_or_insert_with() {
        let pool: BufferPool<u64, String> = BufferPool::with_defaults();

        let val = pool.get_or_insert_with(1, || "created".to_string());
        assert_eq!(*val, "created");

        // Should return existing, not create new
        let val2 = pool.get_or_insert_with(1, || "should_not_be_used".to_string());
        assert_eq!(*val2, "created");
    }

    #[test]
    fn test_insert_if_absent() {
        let pool: BufferPool<u64, i32> = BufferPool::with_defaults();

        assert!(pool.insert_if_absent(1, Arc::new(100)));
        assert!(!pool.insert_if_absent(1, Arc::new(200)));
        assert_eq!(*pool.get(&1).unwrap(), 100); // Original value
    }

    #[test]
    fn test_clear() {
        let pool: BufferPool<u64, i32> = BufferPool::with_defaults();

        pool.insert(1, Arc::new(1));
        pool.insert(2, Arc::new(2));
        pool.insert(3, Arc::new(3));
        assert_eq!(pool.len(), 3);

        pool.clear();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let pool: Arc<BufferPool<u64, u64>> = Arc::new(BufferPool::with_defaults());
        let mut handles = vec![];

        // Spawn multiple threads doing concurrent operations
        for t in 0..4 {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let key = t * 1000 + i;
                    pool.insert(key, Arc::new(key));
                    assert!(pool.get(&key).is_some());
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all entries (may be less due to eviction)
        assert!(pool.len() > 0);
    }

    #[test]
    fn test_arc_sharing() {
        let pool: BufferPool<u64, String> = BufferPool::with_defaults();

        let original = Arc::new("shared".to_string());
        pool.insert(1, Arc::clone(&original));

        // Get returns a clone of the Arc, not a clone of the string
        let retrieved = pool.get(&1).unwrap();
        assert!(Arc::ptr_eq(&original, &retrieved));
    }
}
