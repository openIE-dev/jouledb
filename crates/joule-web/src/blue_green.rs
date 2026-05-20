//! Blue/green deployment: environment switching, traffic cutover,
//! rollback, pre-switch validation, connection draining, and
//! deployment state machine. Pure Rust — no I/O or real routing.

use std::collections::HashMap;
use std::fmt;

// ── Environment color ─────────────────────────────────────────────

/// The two environments in a blue/green deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvColor {
    Blue,
    Green,
}

impl EnvColor {
    /// Return the opposite color.
    pub fn opposite(self) -> Self {
        match self {
            Self::Blue => Self::Green,
            Self::Green => Self::Blue,
        }
    }
}

impl fmt::Display for EnvColor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blue => write!(f, "blue"),
            Self::Green => write!(f, "green"),
        }
    }
}

// ── Deployment state ──────────────────────────────────────────────

/// State machine for blue/green deployment lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployState {
    /// Idle — one environment serving, other available.
    Idle,
    /// Deploying new version to the inactive environment.
    Deploying,
    /// Running pre-switch validation (health checks, smoke tests).
    Validating,
    /// Draining connections from the active environment.
    Draining,
    /// Switching traffic to the new environment.
    Switching,
    /// New environment is live.
    Live,
    /// Rolled back to previous environment.
    RolledBack,
    /// Deployment failed.
    Failed,
}

impl fmt::Display for DeployState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::Deploying => write!(f, "deploying"),
            Self::Validating => write!(f, "validating"),
            Self::Draining => write!(f, "draining"),
            Self::Switching => write!(f, "switching"),
            Self::Live => write!(f, "live"),
            Self::RolledBack => write!(f, "rolled_back"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

// ── Validation check ──────────────────────────────────────────────

/// Result of a pre-switch validation check.
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Name of the check.
    pub name: String,
    /// Whether the check passed.
    pub passed: bool,
    /// Optional detail message.
    pub message: Option<String>,
    /// Duration of the check in microseconds.
    pub duration_us: u64,
}

impl ValidationCheck {
    /// Create a passing check.
    pub fn pass(name: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            message: None,
            duration_us: 0,
        }
    }

    /// Create a failing check.
    pub fn fail(name: &str, message: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            message: Some(message.to_string()),
            duration_us: 0,
        }
    }

    /// Set duration.
    pub fn with_duration(mut self, us: u64) -> Self {
        self.duration_us = us;
        self
    }
}

// ── Drain status ──────────────────────────────────────────────────

/// Models the connection draining process.
#[derive(Debug, Clone)]
pub struct DrainStatus {
    /// Total connections when draining started.
    pub initial_connections: u64,
    /// Current active connections.
    pub active_connections: u64,
    /// Maximum time to wait for drain (time units).
    pub max_drain_units: u64,
    /// Elapsed drain time.
    pub elapsed_units: u64,
}

impl DrainStatus {
    /// Create a new drain status.
    pub fn new(initial_connections: u64, max_drain_units: u64) -> Self {
        Self {
            initial_connections,
            active_connections: initial_connections,
            max_drain_units,
            elapsed_units: 0,
        }
    }

    /// Update the number of active connections and elapsed time.
    pub fn update(&mut self, active: u64, elapsed: u64) {
        self.active_connections = active;
        self.elapsed_units = elapsed;
    }

    /// Check if draining is complete (no connections or timed out).
    pub fn is_complete(&self) -> bool {
        self.active_connections == 0 || self.elapsed_units >= self.max_drain_units
    }

    /// Fraction of connections drained (0.0 to 1.0).
    pub fn drain_progress(&self) -> f64 {
        if self.initial_connections == 0 {
            return 1.0;
        }
        let drained = self
            .initial_connections
            .saturating_sub(self.active_connections);
        drained as f64 / self.initial_connections as f64
    }
}

// ── Environment info ──────────────────────────────────────────────

/// Information about a single environment.
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    /// Environment color.
    pub color: EnvColor,
    /// Deployed version tag.
    pub version: String,
    /// Whether this environment is currently serving traffic.
    pub is_active: bool,
    /// Health status (true = healthy).
    pub healthy: bool,
    /// Instance count.
    pub instances: u32,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl EnvironmentInfo {
    /// Create a new environment.
    pub fn new(color: EnvColor, version: &str) -> Self {
        Self {
            color,
            version: version.to_string(),
            is_active: false,
            healthy: true,
            instances: 1,
            metadata: HashMap::new(),
        }
    }
}

// ── Deployment event ──────────────────────────────────────────────

/// An event in the deployment lifecycle.
#[derive(Debug, Clone)]
pub struct DeploymentEvent {
    pub state: DeployState,
    pub message: String,
    pub timestamp_unit: u64,
}

// ── Blue/green deployment ─────────────────────────────────────────

/// Manages a blue/green deployment lifecycle.
#[derive(Debug, Clone)]
pub struct BlueGreenDeployment {
    /// Blue environment.
    pub blue: EnvironmentInfo,
    /// Green environment.
    pub green: EnvironmentInfo,
    /// Current state.
    pub state: DeployState,
    /// Which color is currently active (serving traffic).
    pub active_color: EnvColor,
    /// Validation checks run during pre-switch.
    validations: Vec<ValidationCheck>,
    /// Drain status (if draining).
    drain: Option<DrainStatus>,
    /// Event log.
    events: Vec<DeploymentEvent>,
}

impl BlueGreenDeployment {
    /// Create a new deployment with blue initially active.
    pub fn new(blue_version: &str, green_version: &str) -> Self {
        let mut blue = EnvironmentInfo::new(EnvColor::Blue, blue_version);
        blue.is_active = true;
        let green = EnvironmentInfo::new(EnvColor::Green, green_version);

        Self {
            blue,
            green,
            state: DeployState::Idle,
            active_color: EnvColor::Blue,
            validations: Vec::new(),
            drain: None,
            events: Vec::new(),
        }
    }

    /// Get the active environment.
    pub fn active_env(&self) -> &EnvironmentInfo {
        match self.active_color {
            EnvColor::Blue => &self.blue,
            EnvColor::Green => &self.green,
        }
    }

    /// Get the inactive (standby) environment.
    pub fn inactive_env(&self) -> &EnvironmentInfo {
        match self.active_color {
            EnvColor::Blue => &self.green,
            EnvColor::Green => &self.blue,
        }
    }

    /// Get mutable reference to inactive environment.
    fn inactive_env_mut(&mut self) -> &mut EnvironmentInfo {
        match self.active_color {
            EnvColor::Blue => &mut self.green,
            EnvColor::Green => &mut self.blue,
        }
    }

    /// Get mutable reference to active environment.
    fn active_env_mut(&mut self) -> &mut EnvironmentInfo {
        match self.active_color {
            EnvColor::Blue => &mut self.blue,
            EnvColor::Green => &mut self.green,
        }
    }

    /// Start deploying a new version to the inactive environment.
    pub fn begin_deploy(&mut self, new_version: &str, timestamp: u64) -> Result<(), String> {
        if self.state != DeployState::Idle && self.state != DeployState::Live {
            return Err(format!(
                "cannot begin deploy in state '{}'",
                self.state
            ));
        }

        self.inactive_env_mut().version = new_version.to_string();
        self.inactive_env_mut().healthy = false; // not yet validated
        self.state = DeployState::Deploying;
        self.validations.clear();
        self.log_event(
            timestamp,
            format!(
                "deploying {} to {} environment",
                new_version,
                self.active_color.opposite()
            ),
        );
        Ok(())
    }

    /// Mark the inactive environment as deployed and begin validation.
    pub fn begin_validation(&mut self, timestamp: u64) -> Result<(), String> {
        if self.state != DeployState::Deploying {
            return Err(format!(
                "cannot validate in state '{}'",
                self.state
            ));
        }
        self.state = DeployState::Validating;
        self.log_event(timestamp, "beginning pre-switch validation".into());
        Ok(())
    }

    /// Record a validation check result.
    pub fn record_validation(&mut self, check: ValidationCheck) {
        self.validations.push(check);
    }

    /// Check if all validations passed.
    pub fn all_validations_passed(&self) -> bool {
        !self.validations.is_empty() && self.validations.iter().all(|v| v.passed)
    }

    /// Get validation results.
    pub fn validations(&self) -> &[ValidationCheck] {
        &self.validations
    }

    /// Begin draining connections from the active environment.
    pub fn begin_drain(
        &mut self,
        initial_connections: u64,
        max_drain_units: u64,
        timestamp: u64,
    ) -> Result<(), String> {
        if self.state != DeployState::Validating {
            return Err(format!(
                "cannot drain in state '{}'",
                self.state
            ));
        }
        if !self.all_validations_passed() {
            return Err("validations have not all passed".into());
        }

        self.inactive_env_mut().healthy = true;
        self.drain = Some(DrainStatus::new(initial_connections, max_drain_units));
        self.state = DeployState::Draining;
        self.log_event(
            timestamp,
            format!("draining {initial_connections} connections from {}", self.active_color),
        );
        Ok(())
    }

    /// Update drain status. Returns true if drain is complete.
    pub fn update_drain(&mut self, active_connections: u64, elapsed: u64) -> bool {
        if let Some(ref mut drain) = self.drain {
            drain.update(active_connections, elapsed);
            drain.is_complete()
        } else {
            false
        }
    }

    /// Get drain status.
    pub fn drain_status(&self) -> Option<&DrainStatus> {
        self.drain.as_ref()
    }

    /// Switch traffic to the new environment.
    pub fn switch(&mut self, timestamp: u64) -> Result<(), String> {
        if self.state != DeployState::Draining {
            return Err(format!(
                "cannot switch in state '{}'",
                self.state
            ));
        }

        let new_active = self.active_color.opposite();
        self.state = DeployState::Switching;
        self.log_event(
            timestamp,
            format!("switching traffic from {} to {}", self.active_color, new_active),
        );

        self.active_env_mut().is_active = false;
        self.active_color = new_active;
        self.active_env_mut().is_active = true;

        self.state = DeployState::Live;
        self.drain = None;
        self.log_event(timestamp, format!("{} is now live", new_active));
        Ok(())
    }

    /// Roll back to the previous environment.
    pub fn rollback(&mut self, timestamp: u64) -> Result<(), String> {
        if self.state == DeployState::Idle {
            return Err("nothing to rollback".into());
        }

        let current = self.active_color;

        // If we're in the middle of switching, swap back.
        if self.state == DeployState::Live {
            let prev = current.opposite();
            self.active_env_mut().is_active = false;
            self.active_color = prev;
            self.active_env_mut().is_active = true;
        }

        self.state = DeployState::RolledBack;
        self.drain = None;
        self.log_event(
            timestamp,
            format!("rolled back to {} environment", self.active_color),
        );
        Ok(())
    }

    /// Mark deployment as failed.
    pub fn fail_deploy(&mut self, reason: &str, timestamp: u64) {
        self.state = DeployState::Failed;
        self.drain = None;
        self.log_event(timestamp, format!("deployment failed: {reason}"));
    }

    /// Get the event log.
    pub fn events(&self) -> &[DeploymentEvent] {
        &self.events
    }

    fn log_event(&mut self, timestamp_unit: u64, message: String) {
        self.events.push(DeploymentEvent {
            state: self.state,
            message,
            timestamp_unit,
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_color_opposite() {
        assert_eq!(EnvColor::Blue.opposite(), EnvColor::Green);
        assert_eq!(EnvColor::Green.opposite(), EnvColor::Blue);
    }

    #[test]
    fn env_color_display() {
        assert_eq!(EnvColor::Blue.to_string(), "blue");
        assert_eq!(EnvColor::Green.to_string(), "green");
    }

    #[test]
    fn deploy_state_display() {
        assert_eq!(DeployState::Idle.to_string(), "idle");
        assert_eq!(DeployState::Deploying.to_string(), "deploying");
        assert_eq!(DeployState::Validating.to_string(), "validating");
        assert_eq!(DeployState::Draining.to_string(), "draining");
        assert_eq!(DeployState::Switching.to_string(), "switching");
        assert_eq!(DeployState::Live.to_string(), "live");
        assert_eq!(DeployState::RolledBack.to_string(), "rolled_back");
        assert_eq!(DeployState::Failed.to_string(), "failed");
    }

    #[test]
    fn validation_check_pass() {
        let v = ValidationCheck::pass("smoke_test").with_duration(500);
        assert!(v.passed);
        assert_eq!(v.duration_us, 500);
        assert!(v.message.is_none());
    }

    #[test]
    fn validation_check_fail() {
        let v = ValidationCheck::fail("health", "503 from endpoint");
        assert!(!v.passed);
        assert_eq!(v.message.as_deref(), Some("503 from endpoint"));
    }

    #[test]
    fn drain_status_progress() {
        let mut drain = DrainStatus::new(100, 60);
        assert_eq!(drain.drain_progress(), 0.0);
        assert!(!drain.is_complete());

        drain.update(50, 30);
        assert!((drain.drain_progress() - 0.5).abs() < 1e-10);

        drain.update(0, 45);
        assert!(drain.is_complete());
        assert!((drain.drain_progress() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn drain_status_timeout() {
        let mut drain = DrainStatus::new(100, 60);
        drain.update(50, 60);
        assert!(drain.is_complete()); // timed out
    }

    #[test]
    fn drain_status_zero_initial() {
        let drain = DrainStatus::new(0, 60);
        assert_eq!(drain.drain_progress(), 1.0);
    }

    #[test]
    fn environment_info_creation() {
        let env = EnvironmentInfo::new(EnvColor::Blue, "v1.0.0");
        assert_eq!(env.color, EnvColor::Blue);
        assert_eq!(env.version, "v1.0.0");
        assert!(!env.is_active);
        assert!(env.healthy);
    }

    #[test]
    fn new_deployment_state() {
        let deploy = BlueGreenDeployment::new("v1.0", "v0.9");
        assert_eq!(deploy.state, DeployState::Idle);
        assert_eq!(deploy.active_color, EnvColor::Blue);
        assert!(deploy.active_env().is_active);
        assert!(!deploy.inactive_env().is_active);
        assert_eq!(deploy.active_env().version, "v1.0");
    }

    #[test]
    fn full_deployment_lifecycle() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");

        // Step 1: begin deploy to inactive (green).
        deploy.begin_deploy("v2.0", 0).unwrap();
        assert_eq!(deploy.state, DeployState::Deploying);
        assert_eq!(deploy.inactive_env().version, "v2.0");

        // Step 2: begin validation.
        deploy.begin_validation(1).unwrap();
        assert_eq!(deploy.state, DeployState::Validating);

        // Step 3: run validation checks.
        deploy.record_validation(ValidationCheck::pass("health"));
        deploy.record_validation(ValidationCheck::pass("smoke"));
        assert!(deploy.all_validations_passed());

        // Step 4: drain connections.
        deploy.begin_drain(50, 60, 2).unwrap();
        assert_eq!(deploy.state, DeployState::Draining);

        // Drain completes.
        let done = deploy.update_drain(0, 10);
        assert!(done);

        // Step 5: switch.
        deploy.switch(3).unwrap();
        assert_eq!(deploy.state, DeployState::Live);
        assert_eq!(deploy.active_color, EnvColor::Green);
        assert!(deploy.active_env().is_active);
        assert!(!deploy.inactive_env().is_active);
    }

    #[test]
    fn cannot_deploy_in_wrong_state() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        let result = deploy.begin_deploy("v3.0", 1);
        assert!(result.is_err());
    }

    #[test]
    fn cannot_validate_without_deploying() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        let result = deploy.begin_validation(0);
        assert!(result.is_err());
    }

    #[test]
    fn cannot_drain_without_validation() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        // No validations recorded → should fail.
        let result = deploy.begin_drain(10, 60, 2);
        assert!(result.is_err());
    }

    #[test]
    fn drain_requires_passing_validations() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        deploy.record_validation(ValidationCheck::fail("health", "bad"));
        let result = deploy.begin_drain(10, 60, 2);
        assert!(result.is_err());
    }

    #[test]
    fn cannot_switch_without_drain() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        let result = deploy.switch(0);
        assert!(result.is_err());
    }

    #[test]
    fn rollback_from_live() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        deploy.record_validation(ValidationCheck::pass("health"));
        deploy.begin_drain(0, 0, 2).unwrap();
        deploy.switch(3).unwrap();
        assert_eq!(deploy.active_color, EnvColor::Green);

        deploy.rollback(4).unwrap();
        assert_eq!(deploy.state, DeployState::RolledBack);
        assert_eq!(deploy.active_color, EnvColor::Blue);
    }

    #[test]
    fn rollback_from_deploying() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.rollback(1).unwrap();
        assert_eq!(deploy.state, DeployState::RolledBack);
        assert_eq!(deploy.active_color, EnvColor::Blue);
    }

    #[test]
    fn rollback_from_idle_fails() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        let result = deploy.rollback(0);
        assert!(result.is_err());
    }

    #[test]
    fn fail_deploy() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.fail_deploy("image pull failed", 1);
        assert_eq!(deploy.state, DeployState::Failed);
    }

    #[test]
    fn events_tracked() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        assert!(deploy.events().len() >= 2);
    }

    #[test]
    fn deploy_again_from_live() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        deploy.record_validation(ValidationCheck::pass("h"));
        deploy.begin_drain(0, 0, 2).unwrap();
        deploy.switch(3).unwrap();

        // Now deploy v3 to blue (which is now inactive).
        deploy.begin_deploy("v3.0", 4).unwrap();
        assert_eq!(deploy.state, DeployState::Deploying);
        assert_eq!(deploy.inactive_env().version, "v3.0");
    }

    #[test]
    fn all_validations_empty_is_false() {
        let deploy = BlueGreenDeployment::new("v1.0", "");
        assert!(!deploy.all_validations_passed());
    }

    #[test]
    fn mixed_validations() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        deploy.record_validation(ValidationCheck::pass("a"));
        deploy.record_validation(ValidationCheck::fail("b", "bad"));
        assert!(!deploy.all_validations_passed());
    }

    #[test]
    fn drain_status_none_initially() {
        let deploy = BlueGreenDeployment::new("v1.0", "");
        assert!(deploy.drain_status().is_none());
    }

    #[test]
    fn drain_status_cleared_after_switch() {
        let mut deploy = BlueGreenDeployment::new("v1.0", "");
        deploy.begin_deploy("v2.0", 0).unwrap();
        deploy.begin_validation(1).unwrap();
        deploy.record_validation(ValidationCheck::pass("h"));
        deploy.begin_drain(10, 60, 2).unwrap();
        assert!(deploy.drain_status().is_some());
        deploy.update_drain(0, 5);
        deploy.switch(3).unwrap();
        assert!(deploy.drain_status().is_none());
    }

    #[test]
    fn environment_metadata() {
        let mut env = EnvironmentInfo::new(EnvColor::Green, "v3");
        env.metadata.insert("region".into(), "us-east".into());
        assert_eq!(env.metadata["region"], "us-east");
    }
}
