//! Object pool for game entity recycling.
//!
//! Pre-allocates objects (bullets, particles, enemies) and recycles them to
//! avoid allocation churn. Provides acquire/release, auto-grow up to a max
//! capacity, warm-up, per-pool statistics, typed pools, pool shrinking,
//! and iteration over active objects.

use std::collections::HashMap;
use std::fmt;

// ── Pool object trait ──────────────────────────────────────────

/// Trait for objects managed by the pool. Must be cloneable and resettable.
pub trait Poolable: Clone + fmt::Debug {
    /// Reset the object to a clean default state (called on acquire).
    fn reset(&mut self);
}

// ── Pool entry ─────────────────────────────────────────────────

/// An entry in the object pool: wraps the object with active flag and ID.
#[derive(Debug, Clone)]
struct PoolEntry<T: Poolable> {
    id: u64,
    object: T,
    active: bool,
}

// ── Pool statistics ────────────────────────────────────────────

/// Runtime statistics for a single pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolStats {
    pub active: usize,
    pub inactive: usize,
    pub total: usize,
    pub peak_active: usize,
    pub acquires: u64,
    pub releases: u64,
    pub grows: u64,
}

impl PoolStats {
    /// Utilization ratio in [0, 1].
    pub fn utilization(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.active as f64 / self.total as f64
        }
    }
}

// ── Object pool ────────────────────────────────────────────────

/// A typed pool of recyclable objects.
pub struct ObjectPool<T: Poolable> {
    entries: Vec<PoolEntry<T>>,
    prototype: T,
    max_capacity: usize,
    next_id: u64,
    peak_active: usize,
    acquires: u64,
    releases: u64,
    grows: u64,
}

impl<T: Poolable> fmt::Debug for ObjectPool<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectPool")
            .field("total", &self.entries.len())
            .field("active", &self.active_count())
            .field("max", &self.max_capacity)
            .finish()
    }
}

impl<T: Poolable> ObjectPool<T> {
    /// Create a new pool with a prototype object and max capacity.
    pub fn new(prototype: T, max_capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            prototype,
            max_capacity: max_capacity.max(1),
            next_id: 1,
            peak_active: 0,
            acquires: 0,
            releases: 0,
            grows: 0,
        }
    }

    /// Pre-allocate `count` inactive objects (warm-up).
    pub fn warm_up(&mut self, count: usize) {
        let to_add = count.min(self.max_capacity.saturating_sub(self.entries.len()));
        for _ in 0..to_add {
            let id = self.next_id;
            self.next_id += 1;
            self.entries.push(PoolEntry {
                id,
                object: self.prototype.clone(),
                active: false,
            });
        }
    }

    /// Acquire an object from the pool: finds an inactive entry, resets it,
    /// marks it active, and returns its ID. If none available, auto-grows
    /// up to max_capacity.
    pub fn acquire(&mut self) -> Option<u64> {
        // Try to find an inactive entry.
        if let Some(entry) = self.entries.iter_mut().find(|e| !e.active) {
            entry.object.reset();
            entry.active = true;
            let id = entry.id;
            self.acquires += 1;
            self.update_peak();
            return Some(id);
        }

        // Auto-grow if under max capacity.
        if self.entries.len() < self.max_capacity {
            let id = self.next_id;
            self.next_id += 1;
            let mut obj = self.prototype.clone();
            obj.reset();
            self.entries.push(PoolEntry {
                id,
                object: obj,
                active: true,
            });
            self.acquires += 1;
            self.grows += 1;
            self.update_peak();
            return Some(id);
        }

        None // Pool exhausted.
    }

    /// Release an object back to the pool by ID.
    pub fn release(&mut self, id: u64) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            if entry.active {
                entry.active = false;
                self.releases += 1;
                return true;
            }
        }
        false
    }

    /// Release all active objects.
    pub fn release_all(&mut self) {
        for entry in &mut self.entries {
            if entry.active {
                entry.active = false;
                self.releases += 1;
            }
        }
    }

    /// Get a reference to an active object by ID.
    pub fn get(&self, id: u64) -> Option<&T> {
        self.entries
            .iter()
            .find(|e| e.id == id && e.active)
            .map(|e| &e.object)
    }

    /// Get a mutable reference to an active object by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut T> {
        self.entries
            .iter_mut()
            .find(|e| e.id == id && e.active)
            .map(|e| &mut e.object)
    }

    /// Iterate over all active objects (id, ref).
    pub fn active_iter(&self) -> impl Iterator<Item = (u64, &T)> {
        self.entries
            .iter()
            .filter(|e| e.active)
            .map(|e| (e.id, &e.object))
    }

    /// Iterate mutably over all active objects.
    pub fn active_iter_mut(&mut self) -> impl Iterator<Item = (u64, &mut T)> {
        self.entries
            .iter_mut()
            .filter(|e| e.active)
            .map(|e| (e.id, &mut e.object))
    }

    /// Collect IDs of all active objects.
    pub fn active_ids(&self) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|e| e.active)
            .map(|e| e.id)
            .collect()
    }

    /// Number of active objects.
    pub fn active_count(&self) -> usize {
        self.entries.iter().filter(|e| e.active).count()
    }

    /// Number of inactive (available) objects.
    pub fn inactive_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.active).count()
    }

    /// Total entries in the pool.
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// Max capacity.
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Whether the pool can still grow.
    pub fn can_grow(&self) -> bool {
        self.entries.len() < self.max_capacity
    }

    fn update_peak(&mut self) {
        let active = self.active_count();
        if active > self.peak_active {
            self.peak_active = active;
        }
    }

    /// Shrink the pool by removing excess inactive entries down to `target_total`.
    /// Never removes active entries. Returns the number of entries removed.
    pub fn shrink(&mut self, target_total: usize) -> usize {
        let mut removed = 0;
        while self.entries.len() > target_total {
            // Find last inactive entry.
            if let Some(idx) = self
                .entries
                .iter()
                .rposition(|e| !e.active)
            {
                self.entries.remove(idx);
                removed += 1;
            } else {
                break; // All remaining are active.
            }
        }
        removed
    }

    /// Statistics snapshot.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            active: self.active_count(),
            inactive: self.inactive_count(),
            total: self.total_count(),
            peak_active: self.peak_active,
            acquires: self.acquires,
            releases: self.releases,
            grows: self.grows,
        }
    }
}

// ── Typed pool registry ────────────────────────────────────────

/// A registry of named object pools, allowing one pool per logical type name.
pub struct PoolRegistry<T: Poolable> {
    pools: HashMap<String, ObjectPool<T>>,
}

impl<T: Poolable> fmt::Debug for PoolRegistry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolRegistry")
            .field("pools", &self.pools.len())
            .finish()
    }
}

impl<T: Poolable> PoolRegistry<T> {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Register a pool under a name.
    pub fn register(&mut self, name: &str, pool: ObjectPool<T>) {
        self.pools.insert(name.to_string(), pool);
    }

    /// Get a pool by name.
    pub fn get(&self, name: &str) -> Option<&ObjectPool<T>> {
        self.pools.get(name)
    }

    /// Get a mutable pool by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut ObjectPool<T>> {
        self.pools.get_mut(name)
    }

    /// Number of registered pools.
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }

    /// Aggregate stats across all pools.
    pub fn total_stats(&self) -> PoolStats {
        let mut stats = PoolStats {
            active: 0,
            inactive: 0,
            total: 0,
            peak_active: 0,
            acquires: 0,
            releases: 0,
            grows: 0,
        };
        for pool in self.pools.values() {
            let s = pool.stats();
            stats.active += s.active;
            stats.inactive += s.inactive;
            stats.total += s.total;
            stats.peak_active += s.peak_active;
            stats.acquires += s.acquires;
            stats.releases += s.releases;
            stats.grows += s.grows;
        }
        stats
    }

    /// Pool names.
    pub fn names(&self) -> Vec<String> {
        self.pools.keys().cloned().collect()
    }
}

impl<T: Poolable> Default for PoolRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test poolable object.
    #[derive(Debug, Clone, PartialEq)]
    struct Bullet {
        x: f64,
        y: f64,
        damage: i32,
    }

    impl Bullet {
        fn new() -> Self {
            Self {
                x: 0.0,
                y: 0.0,
                damage: 10,
            }
        }
    }

    impl Poolable for Bullet {
        fn reset(&mut self) {
            self.x = 0.0;
            self.y = 0.0;
            self.damage = 10;
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Particle {
        life: f64,
    }

    impl Particle {
        fn new() -> Self {
            Self { life: 1.0 }
        }
    }

    impl Poolable for Particle {
        fn reset(&mut self) {
            self.life = 1.0;
        }
    }

    #[test]
    fn new_pool_empty() {
        let pool = ObjectPool::new(Bullet::new(), 100);
        assert_eq!(pool.total_count(), 0);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn warm_up() {
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        pool.warm_up(10);
        assert_eq!(pool.total_count(), 10);
        assert_eq!(pool.inactive_count(), 10);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn warm_up_respects_max() {
        let mut pool = ObjectPool::new(Bullet::new(), 5);
        pool.warm_up(100);
        assert_eq!(pool.total_count(), 5);
    }

    #[test]
    fn acquire_from_warmed_pool() {
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        pool.warm_up(5);
        let id = pool.acquire().unwrap();
        assert_eq!(pool.active_count(), 1);
        assert!(pool.get(id).is_some());
    }

    #[test]
    fn acquire_auto_grows() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        let id = pool.acquire().unwrap();
        assert_eq!(pool.total_count(), 1);
        assert_eq!(pool.active_count(), 1);
        assert!(pool.get(id).is_some());
        assert_eq!(pool.stats().grows, 1);
    }

    #[test]
    fn acquire_fails_at_max_capacity() {
        let mut pool = ObjectPool::new(Bullet::new(), 2);
        pool.acquire().unwrap();
        pool.acquire().unwrap();
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn release_and_reuse() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        let id1 = pool.acquire().unwrap();
        // Modify the object.
        if let Some(b) = pool.get_mut(id1) {
            b.x = 50.0;
            b.damage = 999;
        }
        pool.release(id1);
        assert_eq!(pool.active_count(), 0);
        // Re-acquire — should be reset.
        let id2 = pool.acquire().unwrap();
        let b = pool.get(id2).unwrap();
        assert!((b.x - 0.0).abs() < 1e-9);
        assert_eq!(b.damage, 10);
    }

    #[test]
    fn release_returns_false_for_unknown_id() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        assert!(!pool.release(999));
    }

    #[test]
    fn release_all() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        pool.acquire();
        pool.acquire();
        pool.acquire();
        assert_eq!(pool.active_count(), 3);
        pool.release_all();
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn active_iter() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        let id1 = pool.acquire().unwrap();
        let _id2 = pool.acquire().unwrap();
        pool.release(id1);
        let active: Vec<_> = pool.active_iter().collect();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn active_iter_mut() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        pool.acquire();
        pool.acquire();
        for (_, bullet) in pool.active_iter_mut() {
            bullet.damage = 50;
        }
        for (_, bullet) in pool.active_iter() {
            assert_eq!(bullet.damage, 50);
        }
    }

    #[test]
    fn active_ids() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        let id1 = pool.acquire().unwrap();
        let id2 = pool.acquire().unwrap();
        let mut ids = pool.active_ids();
        ids.sort();
        let mut expected = vec![id1, id2];
        expected.sort();
        assert_eq!(ids, expected);
    }

    #[test]
    fn shrink_pool() {
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        pool.warm_up(20);
        let _id = pool.acquire().unwrap(); // 1 active
        let removed = pool.shrink(5);
        assert!(removed > 0);
        // Active entry should survive.
        assert_eq!(pool.active_count(), 1);
        assert!(pool.total_count() <= 5 || pool.active_count() == pool.total_count());
    }

    #[test]
    fn shrink_does_not_remove_active() {
        let mut pool = ObjectPool::new(Bullet::new(), 10);
        pool.acquire();
        pool.acquire();
        pool.acquire();
        // All 3 are active; shrink to 1 — nothing should be removed.
        let removed = pool.shrink(1);
        assert_eq!(removed, 0);
        assert_eq!(pool.total_count(), 3);
    }

    #[test]
    fn pool_stats() {
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        pool.warm_up(5);
        pool.acquire();
        pool.acquire();
        let s = pool.stats();
        assert_eq!(s.active, 2);
        assert_eq!(s.inactive, 3);
        assert_eq!(s.total, 5);
        assert_eq!(s.acquires, 2);
    }

    #[test]
    fn peak_active_tracks() {
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        let id1 = pool.acquire().unwrap();
        let _id2 = pool.acquire().unwrap();
        let _id3 = pool.acquire().unwrap();
        pool.release(id1);
        assert_eq!(pool.stats().peak_active, 3);
    }

    #[test]
    fn utilization_ratio() {
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        pool.warm_up(10);
        pool.acquire();
        pool.acquire();
        let u = pool.stats().utilization();
        assert!((u - 0.2).abs() < 1e-9);
    }

    #[test]
    fn utilization_empty_pool() {
        let pool = ObjectPool::new(Bullet::new(), 100);
        assert!((pool.stats().utilization() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn can_grow() {
        let mut pool = ObjectPool::new(Bullet::new(), 2);
        assert!(pool.can_grow());
        pool.acquire();
        pool.acquire();
        assert!(!pool.can_grow());
    }

    #[test]
    fn registry_basic() {
        let mut reg: PoolRegistry<Bullet> = PoolRegistry::new();
        reg.register("bullets", ObjectPool::new(Bullet::new(), 100));
        reg.register("enemy_bullets", ObjectPool::new(Bullet::new(), 50));
        assert_eq!(reg.pool_count(), 2);
        assert!(reg.get("bullets").is_some());
    }

    #[test]
    fn registry_acquire_release() {
        let mut reg: PoolRegistry<Bullet> = PoolRegistry::new();
        let mut pool = ObjectPool::new(Bullet::new(), 100);
        pool.warm_up(10);
        reg.register("bullets", pool);
        let id = reg.get_mut("bullets").unwrap().acquire().unwrap();
        assert_eq!(reg.get("bullets").unwrap().active_count(), 1);
        reg.get_mut("bullets").unwrap().release(id);
        assert_eq!(reg.get("bullets").unwrap().active_count(), 0);
    }

    #[test]
    fn registry_total_stats() {
        let mut reg: PoolRegistry<Bullet> = PoolRegistry::new();
        let mut p1 = ObjectPool::new(Bullet::new(), 100);
        p1.warm_up(5);
        p1.acquire();
        let mut p2 = ObjectPool::new(Bullet::new(), 100);
        p2.warm_up(3);
        p2.acquire();
        p2.acquire();
        reg.register("a", p1);
        reg.register("b", p2);
        let s = reg.total_stats();
        assert_eq!(s.active, 3);
        assert_eq!(s.total, 8);
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let pool = ObjectPool::new(Bullet::new(), 10);
        assert!(pool.get(999).is_none());
    }

    #[test]
    fn max_capacity_clamped() {
        let pool = ObjectPool::new(Bullet::new(), 0);
        assert_eq!(pool.max_capacity(), 1);
    }
}
