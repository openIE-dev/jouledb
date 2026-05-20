//! Graceful Shutdown — pending operation tracking and orderly termination.
//!
//! When a sandbox instance is stopped, we need to:
//! 1. Stop accepting new JWP frames
//! 2. Wait for in-flight JWP frames to complete (or timeout)
//! 3. Finalize the energy receipt (sign and send to host)
//! 4. Clean up resources (network isolation, temp files)
//! 5. Send final SIGTERM, wait grace period, then SIGKILL
//!
//! This module tracks pending operations and coordinates orderly shutdown.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Shutdown phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShutdownPhase {
    /// Normal operation — accepting new work.
    Running,
    /// Draining — no new work accepted, waiting for in-flight to complete.
    Draining,
    /// Finalizing — generating final energy receipt, signing, cleanup.
    Finalizing,
    /// Terminated — process is dead, all resources released.
    Terminated,
}

/// Tracks pending operations for graceful shutdown.
pub struct PendingOperations {
    /// Number of in-flight JWP frames being processed.
    in_flight: AtomicU64,
    /// Total frames processed since start.
    total_processed: AtomicU64,
    /// Whether new operations are accepted.
    accepting: AtomicBool,
    /// Current shutdown phase.
    phase: std::sync::RwLock<ShutdownPhase>,
}

impl PendingOperations {
    pub fn new() -> Self {
        Self {
            in_flight: AtomicU64::new(0),
            total_processed: AtomicU64::new(0),
            accepting: AtomicBool::new(true),
            phase: std::sync::RwLock::new(ShutdownPhase::Running),
        }
    }

    /// Register a new operation starting. Returns false if draining.
    pub fn begin(&self) -> bool {
        if !self.accepting.load(Ordering::Acquire) {
            return false;
        }
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Mark an operation as complete.
    pub fn complete(&self) {
        let prev = self.in_flight.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(prev > 0, "complete() called with no in-flight operations");
        self.total_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Number of in-flight operations.
    pub fn in_flight(&self) -> u64 {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Total operations processed.
    pub fn total_processed(&self) -> u64 {
        self.total_processed.load(Ordering::Relaxed)
    }

    /// Whether accepting new operations.
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Relaxed)
    }

    /// Current shutdown phase.
    pub fn phase(&self) -> ShutdownPhase {
        *self.phase.read().unwrap()
    }

    /// Stop accepting new operations (begin drain).
    pub fn start_drain(&self) {
        self.accepting.store(false, Ordering::Release);
        *self.phase.write().unwrap() = ShutdownPhase::Draining;
    }

    /// Check if all in-flight operations are complete.
    pub fn is_drained(&self) -> bool {
        !self.is_accepting() && self.in_flight() == 0
    }
}

impl Default for PendingOperations {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for graceful shutdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownConfig {
    /// Maximum time to wait for in-flight operations to complete.
    pub drain_timeout: Duration,
    /// Grace period after SIGTERM before SIGKILL.
    pub kill_grace_period: Duration,
    /// Whether to generate a final energy receipt before termination.
    pub finalize_receipt: bool,
    /// Whether to clean up network isolation on shutdown.
    pub cleanup_network: bool,
    /// Whether to clean up temp files on shutdown.
    pub cleanup_temp_files: bool,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            drain_timeout: Duration::from_secs(30),
            kill_grace_period: Duration::from_secs(10),
            finalize_receipt: true,
            cleanup_network: true,
            cleanup_temp_files: true,
        }
    }
}

/// Orchestrates graceful shutdown of a sandbox instance.
pub struct ShutdownCoordinator {
    config: ShutdownConfig,
    pending: Arc<PendingOperations>,
    /// Callbacks to run during finalization (energy receipt, cleanup, etc.).
    finalize_callbacks: Vec<Box<dyn FnOnce() + Send>>,
}

impl ShutdownCoordinator {
    pub fn new(config: ShutdownConfig, pending: Arc<PendingOperations>) -> Self {
        Self {
            config,
            pending,
            finalize_callbacks: Vec::new(),
        }
    }

    /// Register a callback to run during the finalization phase.
    pub fn on_finalize(&mut self, callback: impl FnOnce() + Send + 'static) {
        self.finalize_callbacks.push(Box::new(callback));
    }

    /// Get the pending operations tracker (for sharing with JWP handlers).
    pub fn pending(&self) -> Arc<PendingOperations> {
        Arc::clone(&self.pending)
    }

    /// Execute graceful shutdown.
    ///
    /// Returns the shutdown result with timing and statistics.
    pub fn shutdown(self, target_pid: Option<u32>) -> ShutdownResult {
        let Self {
            config,
            pending,
            finalize_callbacks,
        } = self;
        let start = Instant::now();

        // Phase 1: Drain — stop accepting new work, wait for in-flight
        pending.start_drain();
        log::info!(
            "Graceful shutdown: draining ({} in-flight operations)",
            pending.in_flight()
        );

        let drain_start = Instant::now();
        let drained = wait_for_drain(&pending, &config);
        let drain_duration = drain_start.elapsed();

        if !drained {
            log::warn!(
                "Graceful shutdown: drain timeout after {:?} ({} operations still in-flight)",
                drain_duration,
                pending.in_flight()
            );
        }

        // Phase 2: Finalize — run callbacks (energy receipt, cleanup)
        *pending.phase.write().unwrap() = ShutdownPhase::Finalizing;
        log::info!("Graceful shutdown: finalizing");

        for callback in finalize_callbacks {
            callback();
        }

        // Phase 3: Terminate the process
        if let Some(pid) = target_pid {
            terminate_process(pid, &config);
        }

        *pending.phase.write().unwrap() = ShutdownPhase::Terminated;

        ShutdownResult {
            total_duration: start.elapsed(),
            drain_duration,
            drained_cleanly: drained,
            operations_completed: pending.total_processed(),
            operations_abandoned: pending.in_flight(),
        }
    }

}

/// Wait for in-flight operations to complete, up to drain_timeout.
fn wait_for_drain(pending: &PendingOperations, config: &ShutdownConfig) -> bool {
    let deadline = Instant::now() + config.drain_timeout;
    let poll_interval = Duration::from_millis(50);

    while Instant::now() < deadline {
        if pending.is_drained() {
            return true;
        }
        std::thread::sleep(poll_interval);
    }

    pending.is_drained()
}

/// Terminate a process: SIGTERM, wait grace period, then SIGKILL.
fn terminate_process(pid: u32, config: &ShutdownConfig) {
    #[cfg(unix)]
    {
        // Send SIGTERM
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
        log::info!("Graceful shutdown: SIGTERM sent to PID {}", pid);

        // Wait for grace period
        let deadline = Instant::now() + config.kill_grace_period;
        let poll_interval = Duration::from_millis(100);

        while Instant::now() < deadline {
            // Check if process is still alive
            if unsafe { libc::kill(pid as libc::pid_t, 0) } != 0 {
                log::info!("Graceful shutdown: PID {} exited cleanly", pid);
                return;
            }
            std::thread::sleep(poll_interval);
        }

        // Process still alive after grace period — SIGKILL
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
        log::warn!(
            "Graceful shutdown: SIGKILL sent to PID {} (grace period expired)",
            pid
        );
    }

    #[cfg(not(unix))]
    {
        let _ = pid;
        let _ = config;
        log::warn!("Graceful shutdown: process termination not supported on this platform");
    }
}

/// Result of a graceful shutdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownResult {
    /// Total wall-clock time for the shutdown sequence.
    pub total_duration: Duration,
    /// Time spent waiting for in-flight operations.
    pub drain_duration: Duration,
    /// Whether all in-flight operations completed before timeout.
    pub drained_cleanly: bool,
    /// Total operations completed during the instance's lifetime.
    pub operations_completed: u64,
    /// Operations that were abandoned due to drain timeout.
    pub operations_abandoned: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pending_operations_basic() {
        let pending = PendingOperations::new();
        assert!(pending.is_accepting());
        assert_eq!(pending.in_flight(), 0);
        assert_eq!(pending.total_processed(), 0);
        assert_eq!(pending.phase(), ShutdownPhase::Running);
    }

    #[test]
    fn test_pending_operations_begin_complete() {
        let pending = PendingOperations::new();

        assert!(pending.begin());
        assert_eq!(pending.in_flight(), 1);

        assert!(pending.begin());
        assert_eq!(pending.in_flight(), 2);

        pending.complete();
        assert_eq!(pending.in_flight(), 1);
        assert_eq!(pending.total_processed(), 1);

        pending.complete();
        assert_eq!(pending.in_flight(), 0);
        assert_eq!(pending.total_processed(), 2);
    }

    #[test]
    fn test_pending_operations_drain() {
        let pending = PendingOperations::new();
        assert!(pending.begin());

        pending.start_drain();
        assert!(!pending.is_accepting());
        assert!(!pending.begin()); // Rejected during drain
        assert_eq!(pending.phase(), ShutdownPhase::Draining);

        assert!(!pending.is_drained()); // Still 1 in-flight
        pending.complete();
        assert!(pending.is_drained());
    }

    #[test]
    fn test_shutdown_config_default() {
        let config = ShutdownConfig::default();
        assert_eq!(config.drain_timeout, Duration::from_secs(30));
        assert_eq!(config.kill_grace_period, Duration::from_secs(10));
        assert!(config.finalize_receipt);
        assert!(config.cleanup_network);
        assert!(config.cleanup_temp_files);
    }

    #[test]
    fn test_shutdown_config_serde() {
        let config = ShutdownConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ShutdownConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.drain_timeout, config.drain_timeout);
    }

    #[test]
    fn test_shutdown_coordinator_immediate() {
        let pending = Arc::new(PendingOperations::new());
        let config = ShutdownConfig {
            drain_timeout: Duration::from_millis(100),
            kill_grace_period: Duration::from_millis(100),
            finalize_receipt: true,
            cleanup_network: true,
            cleanup_temp_files: true,
        };

        let finalized = Arc::new(AtomicBool::new(false));
        let finalized_clone = Arc::clone(&finalized);

        let mut coordinator = ShutdownCoordinator::new(config, Arc::clone(&pending));
        coordinator.on_finalize(move || {
            finalized_clone.store(true, Ordering::SeqCst);
        });

        // No in-flight operations — should drain immediately
        let result = coordinator.shutdown(None);
        assert!(result.drained_cleanly);
        assert_eq!(result.operations_abandoned, 0);
        assert!(finalized.load(Ordering::SeqCst));
    }

    #[test]
    fn test_shutdown_coordinator_with_inflight() {
        let pending = Arc::new(PendingOperations::new());
        let config = ShutdownConfig {
            drain_timeout: Duration::from_millis(500),
            kill_grace_period: Duration::from_millis(100),
            finalize_receipt: false,
            cleanup_network: false,
            cleanup_temp_files: false,
        };

        // Start an in-flight operation
        assert!(pending.begin());

        let pending_clone = Arc::clone(&pending);
        // Complete the operation from another thread after 100ms
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            pending_clone.complete();
        });

        let coordinator = ShutdownCoordinator::new(config, Arc::clone(&pending));
        let result = coordinator.shutdown(None);

        assert!(result.drained_cleanly);
        assert_eq!(result.operations_completed, 1);
        assert_eq!(result.operations_abandoned, 0);
    }

    #[test]
    fn test_shutdown_coordinator_drain_timeout() {
        let pending = Arc::new(PendingOperations::new());
        let config = ShutdownConfig {
            drain_timeout: Duration::from_millis(50), // Very short timeout
            kill_grace_period: Duration::from_millis(50),
            finalize_receipt: false,
            cleanup_network: false,
            cleanup_temp_files: false,
        };

        // Start an in-flight operation that won't complete
        assert!(pending.begin());

        let coordinator = ShutdownCoordinator::new(config, Arc::clone(&pending));
        let result = coordinator.shutdown(None);

        assert!(!result.drained_cleanly);
        assert_eq!(result.operations_abandoned, 1);
    }

    #[test]
    fn test_shutdown_phase_serde() {
        let phases = vec![
            ShutdownPhase::Running,
            ShutdownPhase::Draining,
            ShutdownPhase::Finalizing,
            ShutdownPhase::Terminated,
        ];
        for phase in phases {
            let json = serde_json::to_string(&phase).unwrap();
            let parsed: ShutdownPhase = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, phase);
        }
    }

    #[test]
    fn test_shutdown_result_serde() {
        let result = ShutdownResult {
            total_duration: Duration::from_millis(150),
            drain_duration: Duration::from_millis(100),
            drained_cleanly: true,
            operations_completed: 42,
            operations_abandoned: 0,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: ShutdownResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.operations_completed, 42);
        assert!(parsed.drained_cleanly);
    }

    #[test]
    fn test_multiple_finalize_callbacks() {
        let pending = Arc::new(PendingOperations::new());
        let config = ShutdownConfig {
            drain_timeout: Duration::from_millis(50),
            kill_grace_period: Duration::from_millis(50),
            finalize_receipt: true,
            cleanup_network: true,
            cleanup_temp_files: true,
        };

        let counter = Arc::new(AtomicU64::new(0));

        let mut coordinator = ShutdownCoordinator::new(config, pending);

        for _ in 0..5 {
            let c = Arc::clone(&counter);
            coordinator.on_finalize(move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }

        coordinator.shutdown(None);
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }
}
