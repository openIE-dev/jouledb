//! Object pool pattern — reusable object checkout/return, lazy initialization,
//! pool size limits, health checking of returned objects, pool drain,
//! statistics (checkouts/returns/creates), and idle timeout cleanup.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by object pool operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectPoolError {
    /// Pool limit reached and no idle objects available.
    Exhausted,
    /// Object failed health check on return.
    HealthCheckFailed,
    /// Factory failed to create a new object.
    CreateFailed(String),
}

impl std::fmt::Display for ObjectPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exhausted => write!(f, "object pool exhausted"),
            Self::HealthCheckFailed => write!(f, "object failed health check"),
            Self::CreateFailed(msg) => write!(f, "factory create failed: {msg}"),
        }
    }
}

// ── PoolStats ────────────────────────────────────────────────────────────────

/// Statistics for the object pool.
#[derive(Debug, Clone, Default)]
pub struct ObjectPoolStats {
    pub total_created: u64,
    pub total_checkouts: u64,
    pub total_returns: u64,
    pub total_discards: u64,
    pub total_health_failures: u64,
    pub total_idle_timeouts: u64,
    pub current_idle: usize,
    pub current_checked_out: usize,
    pub peak_checked_out: usize,
}

impl ObjectPoolStats {
    /// Reuse ratio: returns / checkouts.
    pub fn reuse_ratio(&self) -> f64 {
        if self.total_checkouts == 0 {
            return 0.0;
        }
        // Reused = checkouts that came from idle pool, not newly created
        let reused = self.total_checkouts.saturating_sub(self.total_created);
        reused as f64 / self.total_checkouts as f64
    }
}

// ── Idle wrapper ─────────────────────────────────────────────────────────────

#[derive(Debug)]
struct IdleObject<T> {
    object: T,
    returned_at: Instant,
}

// ── ObjectPool ───────────────────────────────────────────────────────────────

/// Generic object pool with factory, health checking, idle timeout, and limits.
///
/// `F` is a factory closure that creates new objects.
/// `H` is an optional health-check closure; returns `true` if the object is healthy.
pub struct ObjectPool<T, F, H>
where
    F: FnMut() -> Result<T, String>,
    H: FnMut(&T) -> bool,
{
    idle: VecDeque<IdleObject<T>>,
    factory: F,
    health_check: H,
    max_size: usize,
    idle_timeout: Option<Duration>,
    checked_out: usize,
    stats: ObjectPoolStats,
}

impl<T, F, H> ObjectPool<T, F, H>
where
    F: FnMut() -> Result<T, String>,
    H: FnMut(&T) -> bool,
{
    /// Create a pool with a factory, health checker, max size, and idle timeout.
    pub fn new(
        factory: F,
        health_check: H,
        max_size: usize,
        idle_timeout: Option<Duration>,
    ) -> Self {
        assert!(max_size > 0, "pool max_size must be > 0");
        Self {
            idle: VecDeque::new(),
            factory,
            health_check,
            max_size,
            idle_timeout,
            checked_out: 0,
            stats: ObjectPoolStats::default(),
        }
    }

    /// Check out an object from the pool (reuse idle or create new).
    pub fn checkout(&mut self) -> Result<T, ObjectPoolError> {
        // Try to reuse an idle object.
        while let Some(idle_obj) = self.idle.pop_front() {
            // Check idle timeout.
            if let Some(timeout) = self.idle_timeout {
                if idle_obj.returned_at.elapsed() > timeout {
                    self.stats.total_idle_timeouts += 1;
                    self.stats.total_discards += 1;
                    continue; // discard timed-out object
                }
            }
            // Health check.
            if (self.health_check)(&idle_obj.object) {
                self.checked_out += 1;
                self.stats.total_checkouts += 1;
                self.stats.current_idle = self.idle.len();
                self.stats.current_checked_out = self.checked_out;
                if self.checked_out > self.stats.peak_checked_out {
                    self.stats.peak_checked_out = self.checked_out;
                }
                return Ok(idle_obj.object);
            }
            // Failed health check — discard.
            self.stats.total_health_failures += 1;
            self.stats.total_discards += 1;
        }

        // No reusable objects — create new if under limit.
        let total_live = self.checked_out + self.idle.len();
        if total_live >= self.max_size {
            return Err(ObjectPoolError::Exhausted);
        }

        let obj = (self.factory)().map_err(ObjectPoolError::CreateFailed)?;
        self.stats.total_created += 1;
        self.checked_out += 1;
        self.stats.total_checkouts += 1;
        self.stats.current_checked_out = self.checked_out;
        if self.checked_out > self.stats.peak_checked_out {
            self.stats.peak_checked_out = self.checked_out;
        }
        self.stats.current_idle = self.idle.len();
        Ok(obj)
    }

    /// Return an object to the pool.
    pub fn return_object(&mut self, obj: T) -> Result<(), ObjectPoolError> {
        // Health check on return.
        if !(self.health_check)(&obj) {
            self.checked_out = self.checked_out.saturating_sub(1);
            self.stats.total_health_failures += 1;
            self.stats.total_discards += 1;
            self.stats.current_checked_out = self.checked_out;
            return Err(ObjectPoolError::HealthCheckFailed);
        }

        self.checked_out = self.checked_out.saturating_sub(1);
        self.stats.total_returns += 1;
        self.idle.push_back(IdleObject {
            object: obj,
            returned_at: Instant::now(),
        });
        self.stats.current_idle = self.idle.len();
        self.stats.current_checked_out = self.checked_out;
        Ok(())
    }

    /// Pre-warm the pool by creating objects up to the given count.
    pub fn warm(&mut self, count: usize) -> Result<usize, ObjectPoolError> {
        let mut created = 0;
        let target = count.min(self.max_size);
        let current = self.idle.len() + self.checked_out;
        for _ in current..target {
            let obj = (self.factory)().map_err(ObjectPoolError::CreateFailed)?;
            self.stats.total_created += 1;
            self.idle.push_back(IdleObject {
                object: obj,
                returned_at: Instant::now(),
            });
            created += 1;
        }
        self.stats.current_idle = self.idle.len();
        Ok(created)
    }

    /// Drain all idle objects from the pool.
    pub fn drain(&mut self) -> Vec<T> {
        let objects: Vec<T> = self.idle.drain(..).map(|io| io.object).collect();
        self.stats.current_idle = 0;
        objects
    }

    /// Remove idle objects that have exceeded the idle timeout.
    pub fn cleanup_idle(&mut self) -> usize {
        let timeout = match self.idle_timeout {
            Some(t) => t,
            None => return 0,
        };
        let now = Instant::now();
        let before = self.idle.len();
        self.idle.retain(|io| {
            let age = now.duration_since(io.returned_at);
            age <= timeout
        });
        let removed = before - self.idle.len();
        self.stats.total_idle_timeouts += removed as u64;
        self.stats.total_discards += removed as u64;
        self.stats.current_idle = self.idle.len();
        removed
    }

    /// Number of idle objects.
    pub fn idle_count(&self) -> usize {
        self.idle.len()
    }

    /// Number of checked-out objects.
    pub fn checked_out_count(&self) -> usize {
        self.checked_out
    }

    /// Total live objects (idle + checked out).
    pub fn total_live(&self) -> usize {
        self.idle.len() + self.checked_out
    }

    /// Maximum pool size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get pool statistics.
    pub fn stats(&self) -> &ObjectPoolStats {
        &self.stats
    }

    /// Shrink the idle pool to at most `max_idle` objects.
    pub fn shrink_to(&mut self, max_idle: usize) -> usize {
        let mut removed = 0;
        while self.idle.len() > max_idle {
            self.idle.pop_front();
            removed += 1;
            self.stats.total_discards += 1;
        }
        self.stats.current_idle = self.idle.len();
        removed
    }
}

// ── Convenience constructors ─────────────────────────────────────────────────

/// Create a simple object pool with a factory and no health checking.
pub fn simple_pool<T>(
    factory: impl FnMut() -> Result<T, String>,
    max_size: usize,
) -> ObjectPool<T, impl FnMut() -> Result<T, String>, impl FnMut(&T) -> bool> {
    ObjectPool::new(factory, |_| true, max_size, None)
}

/// Create a pool with idle timeout.
pub fn pool_with_timeout<T>(
    factory: impl FnMut() -> Result<T, String>,
    max_size: usize,
    idle_timeout: Duration,
) -> ObjectPool<T, impl FnMut() -> Result<T, String>, impl FnMut(&T) -> bool> {
    ObjectPool::new(factory, |_| true, max_size, Some(idle_timeout))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn counter_factory() -> impl FnMut() -> Result<i32, String> {
        let mut n = 0;
        move || {
            n += 1;
            Ok(n)
        }
    }

    #[test]
    fn test_checkout_creates_new() {
        let mut pool = simple_pool(counter_factory(), 5);
        let obj = pool.checkout().unwrap();
        assert_eq!(obj, 1);
        assert_eq!(pool.checked_out_count(), 1);
        assert_eq!(pool.stats().total_created, 1);
    }

    #[test]
    fn test_return_and_reuse() {
        let mut pool = simple_pool(counter_factory(), 5);
        let obj = pool.checkout().unwrap();
        assert_eq!(obj, 1);
        pool.return_object(obj).unwrap();
        assert_eq!(pool.idle_count(), 1);
        // Checkout again — should reuse, not create new.
        let obj2 = pool.checkout().unwrap();
        assert_eq!(obj2, 1); // same object
        assert_eq!(pool.stats().total_created, 1);
        assert_eq!(pool.stats().total_checkouts, 2);
    }

    #[test]
    fn test_pool_exhausted() {
        let mut pool = simple_pool(counter_factory(), 2);
        pool.checkout().unwrap();
        pool.checkout().unwrap();
        assert_eq!(pool.checkout(), Err(ObjectPoolError::Exhausted));
    }

    #[test]
    fn test_health_check_on_return() {
        let mut pool = ObjectPool::new(
            counter_factory(),
            |val: &i32| *val > 0, // only positive values are "healthy"
            5,
            None,
        );
        let _obj = pool.checkout().unwrap();
        // Return an unhealthy object.
        let result = pool.return_object(-1);
        assert_eq!(result, Err(ObjectPoolError::HealthCheckFailed));
        assert_eq!(pool.stats().total_health_failures, 1);
    }

    #[test]
    fn test_health_check_on_checkout() {
        let call_count = std::cell::Cell::new(0u32);
        let mut pool = ObjectPool::new(
            counter_factory(),
            |val: &i32| {
                let n = call_count.get();
                call_count.set(n + 1);
                // Pass on first call (return_object), fail on second call
                // (checkout from idle), pass on all subsequent calls.
                if *val == 1 && n == 1 {
                    return false;
                }
                true
            },
            5,
            None,
        );
        let obj1 = pool.checkout().unwrap();
        pool.return_object(obj1).unwrap();
        // Now checkout will health-check the idle obj, fail, discard, and create new.
        let obj2 = pool.checkout().unwrap();
        assert_eq!(obj2, 2); // new object
        assert_eq!(pool.stats().total_health_failures, 1);
    }

    #[test]
    fn test_idle_timeout() {
        let mut pool = pool_with_timeout(counter_factory(), 5, Duration::from_millis(0));
        let obj = pool.checkout().unwrap();
        pool.return_object(obj).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        // Cleanup should remove the timed-out object.
        let removed = pool.cleanup_idle();
        assert_eq!(removed, 1);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_idle_timeout_on_checkout() {
        let mut pool = pool_with_timeout(counter_factory(), 5, Duration::from_millis(0));
        let obj = pool.checkout().unwrap();
        pool.return_object(obj).unwrap();
        std::thread::sleep(Duration::from_millis(2));
        // Checkout should skip the timed-out idle obj and create new.
        let obj2 = pool.checkout().unwrap();
        assert_eq!(obj2, 2);
        assert_eq!(pool.stats().total_idle_timeouts, 1);
    }

    #[test]
    fn test_warm_pool() {
        let mut pool = simple_pool(counter_factory(), 10);
        let warmed = pool.warm(5).unwrap();
        assert_eq!(warmed, 5);
        assert_eq!(pool.idle_count(), 5);
        assert_eq!(pool.stats().total_created, 5);
    }

    #[test]
    fn test_warm_respects_max_size() {
        let mut pool = simple_pool(counter_factory(), 3);
        let warmed = pool.warm(10).unwrap();
        assert_eq!(warmed, 3);
    }

    #[test]
    fn test_drain() {
        let mut pool = simple_pool(counter_factory(), 5);
        pool.warm(3).unwrap();
        let drained = pool.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_total_live() {
        let mut pool = simple_pool(counter_factory(), 10);
        pool.warm(3).unwrap();
        let _obj = pool.checkout().unwrap();
        assert_eq!(pool.total_live(), 3); // 2 idle + 1 checked out
    }

    #[test]
    fn test_peak_checked_out() {
        let mut pool = simple_pool(counter_factory(), 10);
        let o1 = pool.checkout().unwrap();
        let o2 = pool.checkout().unwrap();
        let o3 = pool.checkout().unwrap();
        assert_eq!(pool.stats().peak_checked_out, 3);
        pool.return_object(o1).unwrap();
        pool.return_object(o2).unwrap();
        pool.return_object(o3).unwrap();
        assert_eq!(pool.stats().peak_checked_out, 3); // peak retained
    }

    #[test]
    fn test_shrink_to() {
        let mut pool = simple_pool(counter_factory(), 10);
        pool.warm(6).unwrap();
        let removed = pool.shrink_to(2);
        assert_eq!(removed, 4);
        assert_eq!(pool.idle_count(), 2);
    }

    #[test]
    fn test_factory_error() {
        let mut pool = simple_pool(|| Err::<i32, _>("broken".to_string()), 5);
        let result = pool.checkout();
        assert!(matches!(result, Err(ObjectPoolError::CreateFailed(_))));
    }

    #[test]
    fn test_reuse_ratio() {
        let mut pool = simple_pool(counter_factory(), 5);
        let o = pool.checkout().unwrap(); // create
        pool.return_object(o).unwrap();
        let _o2 = pool.checkout().unwrap(); // reuse
        // 2 checkouts, 1 create => 1 reuse => ratio = 0.5
        assert!((pool.stats().reuse_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_multiple_checkout_return_cycles() {
        let mut pool = simple_pool(counter_factory(), 5);
        for _ in 0..10 {
            let obj = pool.checkout().unwrap();
            pool.return_object(obj).unwrap();
        }
        assert_eq!(pool.stats().total_checkouts, 10);
        assert_eq!(pool.stats().total_returns, 10);
        assert_eq!(pool.stats().total_created, 1); // only one object ever created
    }
}
