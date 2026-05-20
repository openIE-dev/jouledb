//! Energy Governor — adaptive container behavior modulation.
//!
//! Monitors hardware telemetry and modulates containerized workloads:
//! - CPU throttling via `docker update --cpus`
//! - Container pause/unpause under critical thermal pressure
//! - Integration with `DatabaseTuner` for engine-specific config commands
//!
//! This proves that a runtime can reduce energy consumption by adapting
//! container resource allocation to real hardware state.

use crate::{InstanceInfo, InstanceState, WorkloadKind};
use joule_db_energy::EnergyConfig;
use joule_db_energy::advisor::{ExecutionHint, HardwareAdvisor};
use joule_db_energy::monitor::{EnergySnapshot, ThermalState};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Configuration for the energy governor.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernorConfig {
    /// Whether governance is enabled.
    pub enabled: bool,
    /// Poll interval in milliseconds (how often to re-evaluate).
    pub poll_interval_ms: u64,
    /// CPU fraction to throttle to under Serious thermal (0.0-1.0).
    pub throttle_cpu_fraction: f64,
    /// CPU fraction under Critical thermal.
    pub critical_cpu_fraction: f64,
    /// Whether to pause containers under Critical thermal.
    pub pause_on_critical: bool,
    /// Whether to send database-specific tuning commands.
    pub auto_tune_databases: bool,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            poll_interval_ms: 5_000,
            throttle_cpu_fraction: 0.5,
            critical_cpu_fraction: 0.25,
            pause_on_critical: true,
            auto_tune_databases: true,
        }
    }
}

/// The action currently applied to an instance.
#[derive(Debug, Clone, PartialEq)]
pub enum GovernanceAction {
    /// No intervention — running at full capacity.
    Normal,
    /// CPU throttled via `docker update --cpus`.
    Throttled { cpus: f64 },
    /// Container paused via `docker pause`.
    Paused,
}

impl std::fmt::Display for GovernanceAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Normal => write!(f, "normal"),
            Self::Throttled { cpus } => write!(f, "throttled({:.2})", cpus),
            Self::Paused => write!(f, "paused"),
        }
    }
}

/// Per-instance governance state.
#[derive(Debug, Clone)]
pub struct GovernanceState {
    pub instance_id: String,
    pub current_action: GovernanceAction,
    pub paused: bool,
}

/// Result of a governance evaluation for one instance.
#[derive(Debug, Clone)]
pub struct GovernanceResult {
    pub instance_id: String,
    pub action: GovernanceAction,
    pub changed: bool,
}

/// Energy governor for container workloads.
pub struct EnergyGovernor {
    advisor: HardwareAdvisor,
    monitor: Arc<RwLock<EnergySnapshot>>,
    instances: RwLock<HashMap<String, GovernanceState>>,
    config: GovernorConfig,
}

impl EnergyGovernor {
    /// Create a new energy governor.
    pub fn new(
        config: GovernorConfig,
        energy_config: &EnergyConfig,
        monitor: Arc<RwLock<EnergySnapshot>>,
    ) -> Self {
        Self {
            advisor: HardwareAdvisor::new(energy_config),
            monitor,
            instances: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Get the governor configuration.
    pub fn config(&self) -> &GovernorConfig {
        &self.config
    }

    /// Evaluate all instances and apply governance actions.
    ///
    /// Called periodically (every `poll_interval_ms`).
    pub async fn tick(&self, instances: &[InstanceInfo]) -> Vec<GovernanceResult> {
        if !self.config.enabled {
            return Vec::new();
        }

        let snapshot = self.monitor.read().map(|s| s.clone()).unwrap_or_default();

        let hints = self.advisor.advise(&snapshot);
        let desired_action = self.decide_action(&hints, &snapshot);

        let mut results = Vec::new();

        for instance in instances {
            // Only govern running instances
            if instance.state != InstanceState::Running {
                continue;
            }

            let instance_id = &instance.id.0;
            let current_action = self.get_current_action(instance_id);
            let changed = current_action != desired_action;

            if changed {
                // Apply the new action
                if let Err(e) = self.apply_action(instance, &desired_action).await {
                    log::warn!(
                        "governance: failed to apply {:?} to {}: {}",
                        desired_action,
                        instance.name,
                        e
                    );
                    continue;
                }

                // Update tracked state
                let mut states = self.instances.write().unwrap();
                states.insert(
                    instance_id.clone(),
                    GovernanceState {
                        instance_id: instance_id.clone(),
                        current_action: desired_action.clone(),
                        paused: matches!(desired_action, GovernanceAction::Paused),
                    },
                );

                log::info!(
                    "governance: {} → {:?} for '{}'",
                    current_action,
                    desired_action,
                    instance.name
                );
            }

            results.push(GovernanceResult {
                instance_id: instance_id.clone(),
                action: desired_action.clone(),
                changed,
            });
        }

        results
    }

    /// Get the current governance action for an instance.
    pub fn get_current_action(&self, instance_id: &str) -> GovernanceAction {
        self.instances
            .read()
            .unwrap()
            .get(instance_id)
            .map(|s| s.current_action.clone())
            .unwrap_or(GovernanceAction::Normal)
    }

    /// Get governance state for all tracked instances.
    pub fn all_states(&self) -> Vec<GovernanceState> {
        self.instances.read().unwrap().values().cloned().collect()
    }

    /// Restore a single instance to normal (un-throttle, un-pause).
    pub async fn restore(&self, instance_id: &str) -> Result<(), crate::RuntimeError> {
        let state = { self.instances.read().unwrap().get(instance_id).cloned() };

        if let Some(state) = state {
            if state.paused {
                unpause_container(instance_id).await?;
            }
            // Remove CPU limit by setting a very high value
            if matches!(state.current_action, GovernanceAction::Throttled { .. }) {
                throttle_container(instance_id, 0.0).await?; // 0 = remove limit
            }
        }

        self.instances.write().unwrap().remove(instance_id);
        Ok(())
    }

    /// Restore all instances to normal (shutdown path).
    pub async fn restore_all(&self) -> Result<(), crate::RuntimeError> {
        let ids: Vec<String> = self.instances.read().unwrap().keys().cloned().collect();

        for id in ids {
            if let Err(e) = self.restore(&id).await {
                log::warn!("governance: failed to restore {}: {}", id, e);
            }
        }
        Ok(())
    }

    // --- Private helpers ---

    /// Decide the governance action based on advisor hints.
    fn decide_action(
        &self,
        hints: &[ExecutionHint],
        snapshot: &EnergySnapshot,
    ) -> GovernanceAction {
        let has_throttle = hints.iter().any(|h| matches!(h, ExecutionHint::Throttle));
        let has_reduce = hints
            .iter()
            .any(|h| matches!(h, ExecutionHint::ReduceBatchSize { .. }));

        if snapshot.thermal_state == ThermalState::Critical && self.config.pause_on_critical {
            GovernanceAction::Paused
        } else if has_throttle {
            GovernanceAction::Throttled {
                cpus: self.config.critical_cpu_fraction,
            }
        } else if has_reduce && snapshot.cpu_utilization > 0.8 {
            GovernanceAction::Throttled {
                cpus: self.config.throttle_cpu_fraction,
            }
        } else {
            GovernanceAction::Normal
        }
    }

    /// Apply a governance action to a specific instance.
    async fn apply_action(
        &self,
        instance: &InstanceInfo,
        action: &GovernanceAction,
    ) -> Result<(), crate::RuntimeError> {
        let instance_id = &instance.id.0;
        let current = self.get_current_action(instance_id);

        // If moving from Paused to anything else, unpause first
        if current == GovernanceAction::Paused && *action != GovernanceAction::Paused {
            unpause_container(instance_id).await?;
        }

        match action {
            GovernanceAction::Normal => {
                // Remove CPU throttle if it was applied
                if matches!(current, GovernanceAction::Throttled { .. }) {
                    throttle_container(instance_id, 0.0).await?;
                }
            }
            GovernanceAction::Throttled { cpus } => {
                throttle_container(instance_id, *cpus).await?;
            }
            GovernanceAction::Paused => {
                if current != GovernanceAction::Paused {
                    pause_container(instance_id).await?;
                }
            }
        }

        Ok(())
    }
}

/// Throttle a container's CPU allocation.
///
/// `cpus = 0.0` removes the limit (equivalent to `--cpus 0`).
async fn throttle_container(container_name: &str, cpus: f64) -> Result<(), crate::RuntimeError> {
    let tool = find_container_tool()?;
    let cpus_str = if cpus <= 0.0 {
        "0".to_string() // remove limit
    } else {
        format!("{:.2}", cpus)
    };

    let output = tokio::process::Command::new(&tool)
        .args(["update", "--cpus", &cpus_str, container_name])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| crate::RuntimeError::ProcessError(format!("docker update failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::warn!("docker update warning: {}", stderr.trim());
    }

    Ok(())
}

/// Pause a container (freeze all processes).
async fn pause_container(container_name: &str) -> Result<(), crate::RuntimeError> {
    let tool = find_container_tool()?;
    let output = tokio::process::Command::new(&tool)
        .args(["pause", container_name])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| crate::RuntimeError::ProcessError(format!("docker pause failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::RuntimeError::ProcessError(format!(
            "docker pause failed: {}",
            stderr.trim()
        )));
    }

    Ok(())
}

/// Unpause a container.
async fn unpause_container(container_name: &str) -> Result<(), crate::RuntimeError> {
    let tool = find_container_tool()?;
    let output = tokio::process::Command::new(&tool)
        .args(["unpause", container_name])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .map_err(|e| crate::RuntimeError::ProcessError(format!("docker unpause failed: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::RuntimeError::ProcessError(format!(
            "docker unpause failed: {}",
            stderr.trim()
        )));
    }

    Ok(())
}

/// Find the container tool (docker/podman/nerdctl).
pub(crate) fn find_container_tool() -> Result<String, crate::RuntimeError> {
    for tool in ["docker", "podman", "nerdctl"] {
        if crate::native::which_exists(tool) {
            return Ok(tool.to_string());
        }
    }
    Err(crate::RuntimeError::ProcessError(
        "no container runtime found".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InstanceId;
    use chrono::Utc;

    fn default_energy_config() -> EnergyConfig {
        EnergyConfig::default()
    }

    fn nominal_snapshot() -> EnergySnapshot {
        EnergySnapshot {
            power_watts: 15.0,
            cpu_utilization: 0.3,
            thermal_state: ThermalState::Nominal,
            memory_pressure: 0.2,
            gpu_available: true,
            gpu_utilization: 0.1,
            ..EnergySnapshot::default()
        }
    }

    fn make_governor(snapshot: EnergySnapshot) -> EnergyGovernor {
        let monitor = Arc::new(RwLock::new(snapshot));
        EnergyGovernor::new(GovernorConfig::default(), &default_energy_config(), monitor)
    }

    fn make_governor_with_monitor(
        snapshot: EnergySnapshot,
    ) -> (EnergyGovernor, Arc<RwLock<EnergySnapshot>>) {
        let monitor = Arc::new(RwLock::new(snapshot));
        let gov = EnergyGovernor::new(
            GovernorConfig::default(),
            &default_energy_config(),
            monitor.clone(),
        );
        (gov, monitor)
    }

    fn test_instance(name: &str) -> InstanceInfo {
        InstanceInfo {
            id: InstanceId(format!("test-{}", name)),
            name: name.to_string(),
            engine: crate::DatabaseEngine::default(),
            workload: WorkloadKind::Container {
                image: "postgres:16".into(),
            },
            mode: crate::RuntimeMode::Native,
            state: InstanceState::Running,
            created_at: Utc::now(),
            pid: Some(12345),
            ports: vec![],
            data_dir: "/tmp/test".into(),
            node_id: None,
            energy_port: None,
            accelerators: vec![],
            volumes: vec![],
            env_vars: HashMap::new(),
            labels: HashMap::new(),
        }
    }

    #[test]
    fn test_decide_action_nominal() {
        let gov = make_governor(nominal_snapshot());
        let hints = vec![ExecutionHint::Normal];
        let action = gov.decide_action(&hints, &nominal_snapshot());
        assert_eq!(action, GovernanceAction::Normal);
    }

    #[test]
    fn test_decide_action_serious_with_high_cpu() {
        let gov = make_governor(nominal_snapshot());
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Serious;
        snap.cpu_utilization = 0.9;
        let hints = vec![
            ExecutionHint::PreferCpu,
            ExecutionHint::ReduceBatchSize {
                suggested_factor: 2,
            },
        ];
        let action = gov.decide_action(&hints, &snap);
        assert!(matches!(action, GovernanceAction::Throttled { .. }));
    }

    #[test]
    fn test_decide_action_critical_pauses() {
        let gov = make_governor(nominal_snapshot());
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Critical;
        let hints = vec![
            ExecutionHint::Throttle,
            ExecutionHint::ReduceBatchSize {
                suggested_factor: 4,
            },
        ];
        let action = gov.decide_action(&hints, &snap);
        assert_eq!(action, GovernanceAction::Paused);
    }

    #[test]
    fn test_decide_action_throttle_without_critical() {
        let gov = make_governor(nominal_snapshot());
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Serious;
        snap.power_watts = 28.0; // > 80% of 30W TDP
        let hints = vec![ExecutionHint::Throttle];
        let action = gov.decide_action(&hints, &snap);
        assert!(matches!(action, GovernanceAction::Throttled { .. }));
    }

    #[test]
    fn test_governance_action_display() {
        assert_eq!(GovernanceAction::Normal.to_string(), "normal");
        assert_eq!(
            GovernanceAction::Throttled { cpus: 0.5 }.to_string(),
            "throttled(0.50)"
        );
        assert_eq!(GovernanceAction::Paused.to_string(), "paused");
    }

    #[tokio::test]
    async fn test_governor_nominal_no_action() {
        let gov = make_governor(nominal_snapshot());
        let instances = vec![test_instance("pg1")];
        let results = gov.tick(&instances).await;

        // First tick with nominal → Normal action (but changed from None → Normal)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].action, GovernanceAction::Normal);
    }

    #[tokio::test]
    async fn test_governor_skips_stopped_instances() {
        let gov = make_governor(nominal_snapshot());
        let mut inst = test_instance("pg1");
        inst.state = InstanceState::Stopped;
        let results = gov.tick(&[inst]).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_governor_disabled() {
        let monitor = Arc::new(RwLock::new(nominal_snapshot()));
        let mut config = GovernorConfig::default();
        config.enabled = false;
        let gov = EnergyGovernor::new(config, &default_energy_config(), monitor);
        let results = gov.tick(&[test_instance("pg1")]).await;
        assert!(results.is_empty());
    }

    #[test]
    fn test_governor_get_current_action_unknown() {
        let gov = make_governor(nominal_snapshot());
        let action = gov.get_current_action("nonexistent");
        assert_eq!(action, GovernanceAction::Normal);
    }

    #[test]
    fn test_governor_config_default() {
        let config = GovernorConfig::default();
        assert!(config.enabled);
        assert_eq!(config.poll_interval_ms, 5_000);
        assert_eq!(config.throttle_cpu_fraction, 0.5);
        assert_eq!(config.critical_cpu_fraction, 0.25);
        assert!(config.pause_on_critical);
        assert!(config.auto_tune_databases);
    }

    #[test]
    fn test_decide_action_serious_low_cpu_no_throttle() {
        let gov = make_governor(nominal_snapshot());
        let mut snap = nominal_snapshot();
        snap.thermal_state = ThermalState::Serious;
        snap.cpu_utilization = 0.3; // Low CPU — no need to throttle
        let hints = vec![
            ExecutionHint::PreferCpu,
            ExecutionHint::ReduceBatchSize {
                suggested_factor: 2,
            },
        ];
        let action = gov.decide_action(&hints, &snap);
        // ReduceBatchSize only triggers throttle when CPU > 80%
        assert_eq!(action, GovernanceAction::Normal);
    }
}
