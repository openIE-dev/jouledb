//! Semaphore primitives — counting semaphore, binary semaphore, reader-writer lock.
//!
//! Pure Rust synchronization primitives modeled for deterministic simulation.
//! No OS-level blocking — uses permit tracking and FIFO fair queuing.

use std::collections::VecDeque;

// ── Acquire Result ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcquireResult {
    /// Permit acquired immediately.
    Acquired,
    /// No permits available, queued with this waiter id.
    Queued(u64),
    /// Semaphore is closed.
    Closed,
}

// ── Counting Semaphore ─────────────────────────────────────────

/// Counting semaphore with fair FIFO queuing.
#[derive(Debug)]
pub struct Semaphore {
    /// Maximum permits.
    max_permits: u32,
    /// Currently available permits.
    available: u32,
    /// FIFO queue of waiters (waiter_id, permits_requested).
    waiters: VecDeque<(u64, u32)>,
    /// Set of held guard ids for RAII release.
    held: Vec<(u64, u32)>,
    next_waiter: u64,
    closed: bool,
}

impl Semaphore {
    pub fn new(permits: u32) -> Self {
        Self {
            max_permits: permits,
            available: permits,
            waiters: VecDeque::new(),
            held: Vec::new(),
            next_waiter: 1,
            closed: false,
        }
    }

    pub fn max_permits(&self) -> u32 {
        self.max_permits
    }

    pub fn available(&self) -> u32 {
        self.available
    }

    pub fn waiters_count(&self) -> usize {
        self.waiters.len()
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn alloc_waiter(&mut self) -> u64 {
        let id = self.next_waiter;
        self.next_waiter += 1;
        id
    }

    /// Try to acquire `n` permits without queuing.
    pub fn try_acquire(&mut self, n: u32) -> Option<SemaphoreGuard> {
        if self.closed {
            return None;
        }
        // Fair queuing: don't skip waiters.
        if !self.waiters.is_empty() {
            return None;
        }
        if self.available >= n {
            self.available -= n;
            let id = self.alloc_waiter();
            self.held.push((id, n));
            Some(SemaphoreGuard { id, permits: n })
        } else {
            None
        }
    }

    /// Acquire `n` permits. If not immediately available, queues the request.
    pub fn acquire(&mut self, n: u32) -> AcquireResult {
        if self.closed {
            return AcquireResult::Closed;
        }
        if self.waiters.is_empty() && self.available >= n {
            self.available -= n;
            let id = self.alloc_waiter();
            self.held.push((id, n));
            AcquireResult::Acquired
        } else {
            let id = self.alloc_waiter();
            self.waiters.push_back((id, n));
            AcquireResult::Queued(id)
        }
    }

    /// Release permits by guard id.
    pub fn release(&mut self, guard: &SemaphoreGuard) {
        if let Some(idx) = self.held.iter().position(|(id, _)| *id == guard.id) {
            let (_, n) = self.held.remove(idx);
            self.available = (self.available + n).min(self.max_permits);
            self.drain_waiters();
        }
    }

    /// Release permits directly (without guard).
    pub fn release_permits(&mut self, n: u32) {
        self.available = (self.available + n).min(self.max_permits);
        self.drain_waiters();
    }

    /// Try to satisfy queued waiters.
    fn drain_waiters(&mut self) -> Vec<u64> {
        let mut granted = Vec::new();
        while let Some(&(id, n)) = self.waiters.front() {
            if self.available >= n {
                self.available -= n;
                self.held.push((id, n));
                self.waiters.pop_front();
                granted.push(id);
            } else {
                break; // FIFO — don't skip.
            }
        }
        granted
    }

    /// Close the semaphore. All future acquires fail.
    pub fn close(&mut self) {
        self.closed = true;
        self.waiters.clear();
    }
}

// ── Semaphore Guard ────────────────────────────────────────────

/// RAII guard for semaphore permits.
#[derive(Debug)]
pub struct SemaphoreGuard {
    pub id: u64,
    pub permits: u32,
}

// ── Binary Semaphore (Mutex) ───────────────────────────────────

/// Binary semaphore (mutex) — exactly 1 permit.
#[derive(Debug)]
pub struct BinarySemaphore {
    inner: Semaphore,
}

impl BinarySemaphore {
    pub fn new() -> Self {
        Self {
            inner: Semaphore::new(1),
        }
    }

    pub fn is_locked(&self) -> bool {
        self.inner.available() == 0
    }

    pub fn try_lock(&mut self) -> Option<SemaphoreGuard> {
        self.inner.try_acquire(1)
    }

    pub fn lock(&mut self) -> AcquireResult {
        self.inner.acquire(1)
    }

    pub fn unlock(&mut self, guard: &SemaphoreGuard) {
        self.inner.release(guard);
    }

    pub fn close(&mut self) {
        self.inner.close();
    }
}

impl Default for BinarySemaphore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Reader-Writer Lock ─────────────────────────────────────────

/// Reader-writer lock built on semaphores.
/// Multiple readers can hold the lock simultaneously, but writers are exclusive.
#[derive(Debug)]
pub struct RwLock {
    /// Semaphore for writer exclusion. Max permits = a large number for readers.
    max_readers: u32,
    sem: Semaphore,
}

impl RwLock {
    pub fn new(max_readers: u32) -> Self {
        Self {
            max_readers,
            sem: Semaphore::new(max_readers),
        }
    }

    /// Acquire a read lock (takes 1 permit).
    pub fn try_read(&mut self) -> Option<SemaphoreGuard> {
        self.sem.try_acquire(1)
    }

    /// Acquire a write lock (takes all permits).
    pub fn try_write(&mut self) -> Option<SemaphoreGuard> {
        self.sem.try_acquire(self.max_readers)
    }

    pub fn read_lock(&mut self) -> AcquireResult {
        self.sem.acquire(1)
    }

    pub fn write_lock(&mut self) -> AcquireResult {
        self.sem.acquire(self.max_readers)
    }

    pub fn release(&mut self, guard: &SemaphoreGuard) {
        self.sem.release(guard);
    }

    pub fn available_read_permits(&self) -> u32 {
        self.sem.available()
    }

    pub fn has_writer(&self) -> bool {
        self.sem.available() == 0 && self.sem.waiters_count() == 0
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_semaphore_basic() {
        let mut sem = Semaphore::new(3);
        assert_eq!(sem.available(), 3);
        let g1 = sem.try_acquire(1).unwrap();
        assert_eq!(sem.available(), 2);
        let g2 = sem.try_acquire(2).unwrap();
        assert_eq!(sem.available(), 0);
        assert!(sem.try_acquire(1).is_none());
        sem.release(&g1);
        assert_eq!(sem.available(), 1);
        sem.release(&g2);
        assert_eq!(sem.available(), 3);
    }

    #[test]
    fn test_semaphore_queuing() {
        let mut sem = Semaphore::new(1);
        let _g = sem.try_acquire(1).unwrap();
        let result = sem.acquire(1);
        assert!(matches!(result, AcquireResult::Queued(_)));
        assert_eq!(sem.waiters_count(), 1);
    }

    #[test]
    fn test_fair_queuing() {
        let mut sem = Semaphore::new(1);
        let g = sem.try_acquire(1).unwrap();
        // Two waiters queue up.
        let r1 = sem.acquire(1);
        let r2 = sem.acquire(1);
        assert!(matches!(r1, AcquireResult::Queued(_)));
        assert!(matches!(r2, AcquireResult::Queued(_)));
        // Release — first waiter should be granted.
        sem.release(&g);
        assert_eq!(sem.available(), 0); // Granted to first waiter.
        assert_eq!(sem.waiters_count(), 1); // Second still waiting.
    }

    #[test]
    fn test_try_acquire_respects_queue() {
        let mut sem = Semaphore::new(2);
        let _g = sem.try_acquire(1).unwrap();
        // Queue a waiter.
        sem.acquire(2);
        // try_acquire should fail even though 1 permit available — fairness.
        assert!(sem.try_acquire(1).is_none());
    }

    #[test]
    fn test_semaphore_close() {
        let mut sem = Semaphore::new(3);
        sem.close();
        assert!(sem.try_acquire(1).is_none());
        assert_eq!(sem.acquire(1), AcquireResult::Closed);
    }

    #[test]
    fn test_binary_semaphore() {
        let mut mtx = BinarySemaphore::new();
        assert!(!mtx.is_locked());
        let g = mtx.try_lock().unwrap();
        assert!(mtx.is_locked());
        assert!(mtx.try_lock().is_none());
        mtx.unlock(&g);
        assert!(!mtx.is_locked());
    }

    #[test]
    fn test_rwlock_multiple_readers() {
        let mut rw = RwLock::new(10);
        let r1 = rw.try_read().unwrap();
        let r2 = rw.try_read().unwrap();
        let r3 = rw.try_read().unwrap();
        assert_eq!(rw.available_read_permits(), 7);
        rw.release(&r1);
        rw.release(&r2);
        rw.release(&r3);
        assert_eq!(rw.available_read_permits(), 10);
    }

    #[test]
    fn test_rwlock_exclusive_writer() {
        let mut rw = RwLock::new(10);
        let w = rw.try_write().unwrap();
        assert_eq!(rw.available_read_permits(), 0);
        // No readers can acquire.
        assert!(rw.try_read().is_none());
        rw.release(&w);
        assert_eq!(rw.available_read_permits(), 10);
    }

    #[test]
    fn test_rwlock_writer_blocks_when_readers() {
        let mut rw = RwLock::new(10);
        let _r = rw.try_read().unwrap();
        // Writer needs all 10 permits, only 9 available.
        assert!(rw.try_write().is_none());
    }

    #[test]
    fn test_release_permits_directly() {
        let mut sem = Semaphore::new(5);
        let _g1 = sem.try_acquire(3).unwrap();
        assert_eq!(sem.available(), 2);
        sem.release_permits(3);
        assert_eq!(sem.available(), 5);
    }

    #[test]
    fn test_guard_raii_info() {
        let mut sem = Semaphore::new(5);
        let g = sem.try_acquire(3).unwrap();
        assert_eq!(g.permits, 3);
        sem.release(&g);
    }
}
