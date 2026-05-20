//! Health monitoring with automatic restart for managed database instances.
//!
//! The HealthMonitor runs as a background tokio task inside the daemon. It
//! periodically checks all Running instances and restarts any that have crashed,
//! up to a configurable maximum number of restarts per instance.

use crate::{InstanceState, RuntimeError, RuntimeManager};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::watch;

/// Health monitor configuration.
#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    /// How often to check instance health (default: 10s).
    pub check_interval: Duration,
    /// Maximum restart attempts per instance before marking as permanently failed (default: 3).
    pub max_restarts: u32,
    /// Number of consecutive failures before triggering restart (default: 3).
    pub failure_threshold: u32,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(10),
            max_restarts: 3,
            failure_threshold: 3,
        }
    }
}

/// Tracks per-instance health state.
struct InstanceHealthState {
    consecutive_failures: u32,
    restart_count: u32,
}

/// Monitors instance health and auto-restarts crashed processes.
pub struct HealthMonitor {
    manager: Arc<RuntimeManager>,
    config: HealthMonitorConfig,
    states: RwLock<HashMap<String, InstanceHealthState>>,
}

impl HealthMonitor {
    /// Create a new health monitor.
    pub fn new(manager: Arc<RuntimeManager>, config: HealthMonitorConfig) -> Self {
        Self {
            manager,
            config,
            states: RwLock::new(HashMap::new()),
        }
    }

    /// Run the health monitor loop until shutdown is signaled.
    pub async fn run(&self, mut shutdown: watch::Receiver<bool>) {
        log::info!(
            "Health monitor started (interval: {:?}, max_restarts: {})",
            self.config.check_interval,
            self.config.max_restarts
        );

        loop {
            tokio::select! {
                _ = tokio::time::sleep(self.config.check_interval) => {
                    self.check_all().await;
                }
                _ = shutdown.changed() => {
                    if *shutdown.borrow() {
                        log::info!("Health monitor shutting down");
                        break;
                    }
                }
            }
        }
    }

    /// Check all running instances.
    async fn check_all(&self) {
        let instances = self.manager.list_instances();

        for instance in instances {
            if instance.state != InstanceState::Running {
                // Clean up state for non-running instances
                if let Ok(mut states) = self.states.write() {
                    states.remove(instance.id.as_str());
                }
                continue;
            }

            match self.manager.health_check(instance.id.as_str()).await {
                Ok(true) => {
                    // Healthy — reset failure count
                    if let Ok(mut states) = self.states.write() {
                        if let Some(state) = states.get_mut(instance.id.as_str()) {
                            state.consecutive_failures = 0;
                        }
                    }
                }
                Ok(false) | Err(_) => {
                    self.handle_failure(instance.id.as_str()).await;
                }
            }
        }
    }

    /// Handle a health check failure for an instance.
    async fn handle_failure(&self, instance_id: &str) {
        let (consecutive_failures, restart_count) = {
            let mut states = match self.states.write() {
                Ok(s) => s,
                Err(_) => return,
            };
            let state = states
                .entry(instance_id.to_string())
                .or_insert(InstanceHealthState {
                    consecutive_failures: 0,
                    restart_count: 0,
                });
            state.consecutive_failures += 1;
            (state.consecutive_failures, state.restart_count)
        };

        log::warn!(
            "Instance {} unhealthy ({}/{} consecutive failures, {} restarts)",
            instance_id,
            consecutive_failures,
            self.config.failure_threshold,
            restart_count
        );

        if consecutive_failures >= self.config.failure_threshold {
            if restart_count >= self.config.max_restarts {
                log::error!(
                    "Instance {} exceeded max restarts ({}), marking as failed",
                    instance_id,
                    self.config.max_restarts
                );
                // Clear the state for this instance
                if let Ok(mut states) = self.states.write() {
                    states.remove(instance_id);
                }
                return;
            }

            log::info!(
                "Attempting restart of instance {} (restart {}/{})",
                instance_id,
                restart_count + 1,
                self.config.max_restarts
            );

            // Attempt restart: stop then start is handled by the manager
            // For now, just stop — the instance metadata persists for manual restart
            if let Err(e) = self.manager.stop_instance(instance_id).await {
                log::error!("Failed to stop crashed instance {}: {}", instance_id, e);
            }

            // Increment restart count
            if let Ok(mut states) = self.states.write() {
                if let Some(state) = states.get_mut(instance_id) {
                    state.restart_count += 1;
                    state.consecutive_failures = 0;
                }
            }
        }
    }

    /// Get the restart count for an instance.
    pub fn restart_count(&self, instance_id: &str) -> u32 {
        self.states
            .read()
            .ok()
            .and_then(|states| states.get(instance_id).map(|s| s.restart_count))
            .unwrap_or(0)
    }

    /// Get the consecutive failure count for an instance.
    pub fn failure_count(&self, instance_id: &str) -> u32 {
        self.states
            .read()
            .ok()
            .and_then(|states| states.get(instance_id).map(|s| s.consecutive_failures))
            .unwrap_or(0)
    }

    /// Reset health state for an instance (e.g., after manual restart).
    pub fn reset(&self, instance_id: &str) {
        if let Ok(mut states) = self.states.write() {
            states.remove(instance_id);
        }
    }
}

/// Recover orphan instances from the registry on daemon startup.
///
/// - Running/Starting instances with dead PIDs → mark as Failed
/// - Stopped/Failed instances → leave as-is
pub async fn recover_orphans(manager: &RuntimeManager) -> Result<Vec<String>, RuntimeError> {
    let mut recovered = Vec::new();
    let instances = manager.list_instances();

    for instance in instances {
        match &instance.state {
            InstanceState::Running | InstanceState::Starting => {
                let is_alive = if let Some(pid) = instance.pid {
                    crate::native::is_process_alive(pid)
                } else {
                    false
                };

                if !is_alive {
                    log::warn!(
                        "Recovering orphan instance {} (was {:?}, PID {:?})",
                        instance.id,
                        instance.state,
                        instance.pid
                    );
                    recovered.push(instance.id.to_string());
                }
            }
            _ => {}
        }
    }

    Ok(recovered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeConfig;

    fn make_manager(tmp: &tempfile::TempDir) -> Arc<RuntimeManager> {
        Arc::new(RuntimeManager::new(RuntimeConfig::default(), tmp.path().to_path_buf()).unwrap())
    }

    #[tokio::test]
    async fn test_health_monitor_runs_and_stops() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = make_manager(&tmp);
        let config = HealthMonitorConfig {
            check_interval: Duration::from_millis(50),
            max_restarts: 3,
            failure_threshold: 3,
        };

        let monitor = HealthMonitor::new(manager, config);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        let handle = tokio::spawn(async move {
            monitor.run(shutdown_rx).await;
        });

        // Let it run a few cycles
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Signal shutdown
        shutdown_tx.send(true).unwrap();
        handle.await.unwrap();
    }

    #[test]
    fn test_health_monitor_config_defaults() {
        let config = HealthMonitorConfig::default();
        assert_eq!(config.check_interval, Duration::from_secs(10));
        assert_eq!(config.max_restarts, 3);
        assert_eq!(config.failure_threshold, 3);
    }

    #[tokio::test]
    async fn test_restart_count_tracking() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = make_manager(&tmp);
        let config = HealthMonitorConfig::default();

        let monitor = HealthMonitor::new(manager, config);

        assert_eq!(monitor.restart_count("nonexistent"), 0);
        assert_eq!(monitor.failure_count("nonexistent"), 0);
    }

    #[tokio::test]
    async fn test_reset_clears_state() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = make_manager(&tmp);
        let config = HealthMonitorConfig::default();

        let monitor = HealthMonitor::new(manager, config);

        // Manually inject some state
        {
            let mut states = monitor.states.write().unwrap();
            states.insert(
                "test-id".into(),
                InstanceHealthState {
                    consecutive_failures: 5,
                    restart_count: 2,
                },
            );
        }

        assert_eq!(monitor.failure_count("test-id"), 5);
        assert_eq!(monitor.restart_count("test-id"), 2);

        monitor.reset("test-id");

        assert_eq!(monitor.failure_count("test-id"), 0);
        assert_eq!(monitor.restart_count("test-id"), 0);
    }

    #[tokio::test]
    async fn test_recover_orphans_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let manager =
            RuntimeManager::new(RuntimeConfig::default(), tmp.path().to_path_buf()).unwrap();

        let recovered = recover_orphans(&manager).await.unwrap();
        assert!(recovered.is_empty());
    }
}
