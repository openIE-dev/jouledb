//! Lock utility functions that recover from poisoned locks instead of panicking.
//!
//! In production, a poisoned lock (caused by a thread panicking while holding the lock)
//! should not crash the entire server. These helpers log the error and recover the
//! underlying data, allowing the server to continue operating.

use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Acquire a read lock, recovering from poisoning.
pub fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| {
        tracing::error!("RwLock read guard poisoned — recovering");
        poisoned.into_inner()
    })
}

/// Acquire a write lock, recovering from poisoning.
pub fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|poisoned| {
        tracing::error!("RwLock write guard poisoned — recovering");
        poisoned.into_inner()
    })
}

/// Acquire a mutex lock, recovering from poisoning.
pub fn mutex_lock<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| {
        tracing::error!("Mutex poisoned — recovering");
        poisoned.into_inner()
    })
}
