//! Lock-free data structures for JouleDB
//!
//! Provides lock-free alternatives to standard data structures for hot paths
//! where lock contention would be a bottleneck.
//!
//! ## Design Principles
//!
//! - Use atomic operations instead of locks
//! - CAS (Compare-And-Swap) for updates
//! - Memory ordering appropriate for use case
//! - Fallback to locked version when needed

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Lock-free counter
///
/// Thread-safe counter using atomic operations. Much faster than
/// RwLock<u64> for high-frequency increments.
#[derive(Debug, Default)]
pub struct LockFreeCounter {
    value: AtomicU64,
}

impl LockFreeCounter {
    /// Create a new counter starting at 0
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Create a new counter with initial value
    pub fn with_value(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
        }
    }

    /// Increment and return new value
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Add delta and return new value
    pub fn add(&self, delta: u64) -> u64 {
        self.value.fetch_add(delta, Ordering::Relaxed) + delta
    }

    /// Decrement and return new value
    pub fn decrement(&self) -> u64 {
        self.value.fetch_sub(1, Ordering::Relaxed).wrapping_sub(1)
    }

    /// Get current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Set value
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Compare and swap
    pub fn compare_and_swap(&self, expected: u64, new: u64) -> Result<u64, u64> {
        match self
            .value
            .compare_exchange(expected, new, Ordering::Acquire, Ordering::Relaxed)
        {
            Ok(old) => Ok(old),
            Err(current) => Err(current),
        }
    }
}

/// Lock-free size counter
///
/// Similar to LockFreeCounter but for usize (useful for collection sizes)
#[derive(Debug, Default)]
pub struct LockFreeSize {
    value: AtomicUsize,
}

impl LockFreeSize {
    /// Create a new size counter starting at 0
    pub fn new() -> Self {
        Self {
            value: AtomicUsize::new(0),
        }
    }

    /// Create with initial value
    pub fn with_value(initial: usize) -> Self {
        Self {
            value: AtomicUsize::new(initial),
        }
    }

    /// Increment and return new value
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement and return new value
    pub fn decrement(&self) -> usize {
        self.value.fetch_sub(1, Ordering::Relaxed).wrapping_sub(1)
    }

    /// Add delta and return new value
    pub fn add(&self, delta: usize) -> usize {
        self.value.fetch_add(delta, Ordering::Relaxed) + delta
    }

    /// Get current value
    pub fn get(&self) -> usize {
        self.value.load(Ordering::Relaxed)
    }

    /// Set value
    pub fn set(&self, value: usize) {
        self.value.store(value, Ordering::Relaxed);
    }
}

/// Lock-free statistics
///
/// Thread-safe statistics collection without locks
#[derive(Debug)]
pub struct LockFreeStats {
    /// Number of operations
    pub operations: LockFreeCounter,
    /// Number of successful operations
    pub successes: LockFreeCounter,
    /// Number of failures
    pub failures: LockFreeCounter,
    /// Total latency in nanoseconds
    pub total_latency_ns: LockFreeCounter,
}

impl LockFreeStats {
    /// Create new stats
    pub fn new() -> Self {
        Self {
            operations: LockFreeCounter::new(),
            successes: LockFreeCounter::new(),
            failures: LockFreeCounter::new(),
            total_latency_ns: LockFreeCounter::new(),
        }
    }

    /// Record a successful operation
    pub fn record_success(&self, latency_ns: u64) {
        self.operations.increment();
        self.successes.increment();
        self.total_latency_ns.add(latency_ns);
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        self.operations.increment();
        self.failures.increment();
    }

    /// Get success rate (0.0 to 1.0)
    pub fn success_rate(&self) -> f64 {
        let ops = self.operations.get();
        if ops == 0 {
            return 0.0;
        }
        self.successes.get() as f64 / ops as f64
    }

    /// Get average latency in nanoseconds
    pub fn avg_latency_ns(&self) -> u64 {
        let successes = self.successes.get();
        if successes == 0 {
            return 0;
        }
        self.total_latency_ns.get() / successes
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.operations.set(0);
        self.successes.set(0);
        self.failures.set(0);
        self.total_latency_ns.set(0);
    }
}

impl Default for LockFreeStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_counter_basic() {
        let counter = LockFreeCounter::new();
        assert_eq!(counter.get(), 0);

        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.get(), 1);

        assert_eq!(counter.add(5), 6);
        assert_eq!(counter.get(), 6);

        assert_eq!(counter.decrement(), 5);
        assert_eq!(counter.get(), 5);
    }

    #[test]
    fn test_counter_concurrent() {
        let counter = Arc::new(LockFreeCounter::new());
        let mut handles = vec![];

        // Spawn 10 threads, each incrementing 1000 times
        for _ in 0..10 {
            let counter = counter.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    counter.increment();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.get(), 10000);
    }

    #[test]
    fn test_stats() {
        let stats = LockFreeStats::new();

        stats.record_success(100);
        stats.record_success(200);
        stats.record_failure();

        assert_eq!(stats.operations.get(), 3);
        assert_eq!(stats.successes.get(), 2);
        assert_eq!(stats.failures.get(), 1);
        assert_eq!(stats.success_rate(), 2.0 / 3.0);
        assert_eq!(stats.avg_latency_ns(), 150); // (100 + 200) / 2
    }
}
