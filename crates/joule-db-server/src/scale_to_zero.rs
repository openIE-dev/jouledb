//! Scale-to-Zero for JouleDB
//!
//! Automatic suspend/resume of compute when no connections are active.
//! Resume on first incoming request. Tracks energy saved during suspension.

use serde::Serialize;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum ScaleToZeroError {
    #[error("invalid state transition: {from:?} -> {to:?}")]
    InvalidTransition {
        from: ServerLifecycleState,
        to: ServerLifecycleState,
    },

    #[error("server is suspended")]
    Suspended,

    #[error("internal error: {0}")]
    Internal(String),
}

// ============================================================================
// Lifecycle state
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerLifecycleState {
    Active,
    Idle,
    Suspending,
    Suspended,
    Resuming,
}

// ============================================================================
// Suspend report
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct SuspendReport {
    pub suspended_at: u64,
    pub idle_duration_secs: u64,
    pub state: ServerLifecycleState,
}

// ============================================================================
// Status report
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct StatusReport {
    pub state: ServerLifecycleState,
    pub uptime_secs: u64,
    pub last_activity_ms: u64,
    pub connection_count: u64,
    pub energy_saved_uj: u64,
    pub suspended_at: Option<u64>,
    pub total_suspensions: u64,
}

// ============================================================================
// ActivityTracker
// ============================================================================

pub struct ActivityTracker {
    last_activity: AtomicU64,
    connection_count: AtomicU64,
    state: RwLock<ServerLifecycleState>,
    started_at: u64,
    suspended_at: AtomicU64,
    total_suspended_ms: AtomicU64,
    total_suspensions: AtomicU64,
    /// Estimated idle power draw in microwatts (used to calculate energy saved)
    idle_power_uw: u64,
}

impl ActivityTracker {
    pub fn new() -> Self {
        let now = now_millis();
        Self {
            last_activity: AtomicU64::new(now),
            connection_count: AtomicU64::new(0),
            state: RwLock::new(ServerLifecycleState::Active),
            started_at: now,
            suspended_at: AtomicU64::new(0),
            total_suspended_ms: AtomicU64::new(0),
            total_suspensions: AtomicU64::new(0),
            // Default: ~5W idle power for a typical server process
            idle_power_uw: 5_000_000,
        }
    }

    /// Record activity (called on every request/command)
    pub fn touch(&self) {
        self.last_activity.store(now_millis(), Ordering::Relaxed);
    }

    /// Increment active connection count
    pub fn increment_connections(&self) -> u64 {
        self.connection_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrement active connection count (clamped to zero)
    pub fn decrement_connections(&self) -> u64 {
        loop {
            let current = self.connection_count.load(Ordering::Relaxed);
            if current == 0 {
                return 0;
            }
            match self.connection_count.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return current - 1,
                Err(_) => continue,
            }
        }
    }

    /// Get current connection count
    pub fn connection_count(&self) -> u64 {
        self.connection_count.load(Ordering::Relaxed)
    }

    /// Check if the server has been idle longer than the given timeout
    pub fn is_idle(&self, timeout: Duration) -> bool {
        let last = self.last_activity.load(Ordering::Relaxed);
        let now = now_millis();
        let elapsed_ms = now.saturating_sub(last);
        let conns = self.connection_count.load(Ordering::Relaxed);
        conns == 0 && elapsed_ms >= timeout.as_millis() as u64
    }

    /// Get current lifecycle state
    pub fn state(&self) -> ServerLifecycleState {
        *self.state.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Suspend the server (flush WAL, release resources)
    pub fn suspend(&self) -> Result<SuspendReport, ScaleToZeroError> {
        let mut state = self
            .state
            .write()
            .map_err(|e| ScaleToZeroError::Internal(e.to_string()))?;

        match *state {
            ServerLifecycleState::Active | ServerLifecycleState::Idle => {}
            other => {
                return Err(ScaleToZeroError::InvalidTransition {
                    from: other,
                    to: ServerLifecycleState::Suspending,
                });
            }
        }

        *state = ServerLifecycleState::Suspending;

        let now = now_millis();
        let last = self.last_activity.load(Ordering::Relaxed);
        let idle_duration_secs = now.saturating_sub(last) / 1000;

        // Mark as suspended
        self.suspended_at.store(now, Ordering::Relaxed);
        self.total_suspensions.fetch_add(1, Ordering::Relaxed);
        *state = ServerLifecycleState::Suspended;

        Ok(SuspendReport {
            suspended_at: now,
            idle_duration_secs,
            state: ServerLifecycleState::Suspended,
        })
    }

    /// Resume the server (rebuild state from checkpoint)
    pub fn resume(&self) -> Result<(), ScaleToZeroError> {
        let mut state = self
            .state
            .write()
            .map_err(|e| ScaleToZeroError::Internal(e.to_string()))?;

        if *state != ServerLifecycleState::Suspended {
            return Err(ScaleToZeroError::InvalidTransition {
                from: *state,
                to: ServerLifecycleState::Resuming,
            });
        }

        *state = ServerLifecycleState::Resuming;

        // Account for suspended time
        let suspended_at = self.suspended_at.load(Ordering::Relaxed);
        let now = now_millis();
        let suspended_ms = now.saturating_sub(suspended_at);
        self.total_suspended_ms
            .fetch_add(suspended_ms, Ordering::Relaxed);

        // Mark as active
        self.last_activity.store(now, Ordering::Relaxed);
        *state = ServerLifecycleState::Active;

        Ok(())
    }

    /// Estimate energy saved during all suspension periods (in microjoules)
    pub fn energy_saved_uj(&self) -> u64 {
        let total_ms = self.total_suspended_ms.load(Ordering::Relaxed);
        // If currently suspended, add ongoing suspension time
        let current_suspended = {
            let state = self.state();
            if state == ServerLifecycleState::Suspended {
                let suspended_at = self.suspended_at.load(Ordering::Relaxed);
                now_millis().saturating_sub(suspended_at)
            } else {
                0
            }
        };
        let total_ms = total_ms + current_suspended;
        // energy = power * time
        // idle_power_uw * total_ms / 1000 = microjoules
        (self.idle_power_uw * total_ms) / 1000
    }

    /// Get a full status report
    pub fn status(&self) -> StatusReport {
        let now = now_millis();
        let suspended_at_raw = self.suspended_at.load(Ordering::Relaxed);
        let state = self.state();
        StatusReport {
            state,
            uptime_secs: now.saturating_sub(self.started_at) / 1000,
            last_activity_ms: self.last_activity.load(Ordering::Relaxed),
            connection_count: self.connection_count.load(Ordering::Relaxed),
            energy_saved_uj: self.energy_saved_uj(),
            suspended_at: if state == ServerLifecycleState::Suspended {
                Some(suspended_at_raw)
            } else {
                None
            },
            total_suspensions: self.total_suspensions.load(Ordering::Relaxed),
        }
    }

    /// Check if server is suspended and needs to be resumed before processing
    pub fn ensure_active(&self) -> Result<(), ScaleToZeroError> {
        if self.state() == ServerLifecycleState::Suspended {
            self.resume()?;
        }
        self.touch();
        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker_is_active() {
        let tracker = ActivityTracker::new();
        assert_eq!(tracker.state(), ServerLifecycleState::Active);
        assert_eq!(tracker.connection_count(), 0);
    }

    #[test]
    fn test_touch_updates_activity() {
        let tracker = ActivityTracker::new();
        let before = tracker.last_activity.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(10));
        tracker.touch();
        let after = tracker.last_activity.load(Ordering::Relaxed);
        assert!(after >= before);
    }

    #[test]
    fn test_connection_counting() {
        let tracker = ActivityTracker::new();
        assert_eq!(tracker.increment_connections(), 1);
        assert_eq!(tracker.increment_connections(), 2);
        assert_eq!(tracker.connection_count(), 2);
        tracker.decrement_connections();
        assert_eq!(tracker.connection_count(), 1);
    }

    #[test]
    fn test_is_idle() {
        let tracker = ActivityTracker::new();
        // Just touched, should not be idle
        assert!(!tracker.is_idle(Duration::from_millis(100)));
        // With connections, never idle
        tracker.increment_connections();
        assert!(!tracker.is_idle(Duration::from_millis(0)));
        tracker.decrement_connections();
    }

    #[test]
    fn test_suspend_resume_cycle() {
        let tracker = ActivityTracker::new();

        let report = tracker.suspend().unwrap();
        assert_eq!(report.state, ServerLifecycleState::Suspended);
        assert_eq!(tracker.state(), ServerLifecycleState::Suspended);

        tracker.resume().unwrap();
        assert_eq!(tracker.state(), ServerLifecycleState::Active);
        assert_eq!(tracker.total_suspensions.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_cannot_suspend_when_already_suspended() {
        let tracker = ActivityTracker::new();
        tracker.suspend().unwrap();

        let result = tracker.suspend();
        assert!(result.is_err());
    }

    #[test]
    fn test_cannot_resume_when_active() {
        let tracker = ActivityTracker::new();
        let result = tracker.resume();
        assert!(result.is_err());
    }

    #[test]
    fn test_ensure_active_auto_resumes() {
        let tracker = ActivityTracker::new();
        tracker.suspend().unwrap();
        assert_eq!(tracker.state(), ServerLifecycleState::Suspended);

        tracker.ensure_active().unwrap();
        assert_eq!(tracker.state(), ServerLifecycleState::Active);
    }

    #[test]
    fn test_energy_saved_calculation() {
        let tracker = ActivityTracker::new();
        // Manually set some suspended time for deterministic test
        tracker.total_suspended_ms.store(1000, Ordering::Relaxed); // 1 second
        // 5W idle power = 5_000_000 µW, 1 second = 5_000_000 µJ = 5 J
        let saved = tracker.energy_saved_uj();
        assert_eq!(saved, 5_000_000);
    }

    #[test]
    fn test_status_report() {
        let tracker = ActivityTracker::new();
        let status = tracker.status();
        assert_eq!(status.state, ServerLifecycleState::Active);
        assert_eq!(status.connection_count, 0);
        assert!(status.suspended_at.is_none());
        assert_eq!(status.total_suspensions, 0);
    }
}
