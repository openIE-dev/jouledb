//! Supervision tree — supervisors managing child processes with restart strategies.
//!
//! Replaces Erlang/OTP-style supervision in pure Rust. Supports one_for_one,
//! one_for_all, and rest_for_one restart strategies, max restart intensity,
//! child specifications, tree visualization, and graceful shutdown.

use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Supervision tree domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisorError {
    /// Child not found.
    ChildNotFound(String),
    /// Supervisor not found.
    SupervisorNotFound(String),
    /// Duplicate child ID.
    DuplicateChild(String),
    /// Max restart intensity exceeded.
    MaxIntensityExceeded {
        supervisor: String,
        restarts: u32,
        period_ticks: u32,
    },
    /// Child already running.
    ChildAlreadyRunning(String),
    /// Supervisor is shutting down.
    ShuttingDown(String),
}

impl std::fmt::Display for SupervisorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChildNotFound(id) => write!(f, "child not found: {id}"),
            Self::SupervisorNotFound(id) => write!(f, "supervisor not found: {id}"),
            Self::DuplicateChild(id) => write!(f, "duplicate child: {id}"),
            Self::MaxIntensityExceeded {
                supervisor,
                restarts,
                period_ticks,
            } => write!(
                f,
                "supervisor {supervisor}: {restarts} restarts in {period_ticks} ticks"
            ),
            Self::ChildAlreadyRunning(id) => write!(f, "child {id} already running"),
            Self::ShuttingDown(id) => write!(f, "supervisor {id} is shutting down"),
        }
    }
}

impl std::error::Error for SupervisorError {}

// ── Restart Strategy ────────────────────────────────────────────

/// How a supervisor restarts children on failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Restart only the failed child.
    OneForOne,
    /// Restart all children when one fails.
    OneForAll,
    /// Restart the failed child and all children started after it.
    RestForOne,
}

impl Default for RestartStrategy {
    fn default() -> Self {
        Self::OneForOne
    }
}

// ── Child Type ──────────────────────────────────────────────────

/// Whether a child is a worker or a sub-supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildType {
    Worker,
    Supervisor,
}

// ── Restart Type ────────────────────────────────────────────────

/// Whether a child should be restarted on failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartType {
    /// Always restart.
    Permanent,
    /// Restart only on abnormal exit.
    Transient,
    /// Never restart.
    Temporary,
}

impl Default for RestartType {
    fn default() -> Self {
        Self::Permanent
    }
}

// ── Child State ─────────────────────────────────────────────────

/// Lifecycle state of a child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildState {
    Starting,
    Running,
    Failed,
    Stopped,
    Restarting,
}

// ── Child Spec ──────────────────────────────────────────────────

/// Specification for a child process.
#[derive(Debug, Clone)]
pub struct ChildSpec {
    pub id: String,
    pub child_type: ChildType,
    pub restart_type: RestartType,
    pub shutdown_timeout_ms: u64,
}

impl ChildSpec {
    pub fn worker(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            child_type: ChildType::Worker,
            restart_type: RestartType::default(),
            shutdown_timeout_ms: 5000,
        }
    }

    pub fn supervisor(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            child_type: ChildType::Supervisor,
            restart_type: RestartType::Permanent,
            shutdown_timeout_ms: 10000,
        }
    }

    /// Set restart type.
    pub fn with_restart(mut self, restart: RestartType) -> Self {
        self.restart_type = restart;
        self
    }

    /// Set shutdown timeout.
    pub fn with_shutdown_timeout(mut self, ms: u64) -> Self {
        self.shutdown_timeout_ms = ms;
        self
    }
}

// ── Child ───────────────────────────────────────────────────────

/// A managed child process.
#[derive(Debug, Clone)]
pub struct Child {
    pub spec: ChildSpec,
    pub state: ChildState,
    pub restart_count: u32,
    pub start_order: usize,
    pub uptime_ticks: u64,
    pub started_at_tick: u64,
}

impl Child {
    fn new(spec: ChildSpec, order: usize, tick: u64) -> Self {
        Self {
            spec,
            state: ChildState::Running,
            restart_count: 0,
            start_order: order,
            uptime_ticks: 0,
            started_at_tick: tick,
        }
    }
}

// ── Restart Intensity ───────────────────────────────────────────

/// Configuration for max restart frequency.
#[derive(Debug, Clone)]
pub struct RestartIntensity {
    /// Maximum number of restarts allowed in the period.
    pub max_restarts: u32,
    /// Period in ticks.
    pub period_ticks: u32,
}

impl Default for RestartIntensity {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            period_ticks: 60,
        }
    }
}

// ── Supervisor Event ────────────────────────────────────────────

/// Events emitted by the supervision tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SupervisorEvent {
    ChildStarted { supervisor: String, child: String },
    ChildStopped { supervisor: String, child: String },
    ChildRestarted { supervisor: String, child: String },
    ChildFailed { supervisor: String, child: String, reason: String },
    IntensityExceeded { supervisor: String },
    SupervisorShutDown { supervisor: String },
}

// ── Supervisor ──────────────────────────────────────────────────

/// A supervisor managing child processes.
#[derive(Debug)]
pub struct Supervisor {
    pub id: String,
    pub strategy: RestartStrategy,
    pub intensity: RestartIntensity,
    children: Vec<Child>,
    child_order: Vec<String>,
    restart_timestamps: Vec<u64>,
    pub events: Vec<SupervisorEvent>,
    current_tick: u64,
    shutting_down: bool,
}

impl Supervisor {
    pub fn new(id: impl Into<String>, strategy: RestartStrategy) -> Self {
        Self {
            id: id.into(),
            strategy,
            intensity: RestartIntensity::default(),
            children: Vec::new(),
            child_order: Vec::new(),
            restart_timestamps: Vec::new(),
            events: Vec::new(),
            current_tick: 0,
            shutting_down: false,
        }
    }

    /// Set restart intensity.
    pub fn with_intensity(mut self, intensity: RestartIntensity) -> Self {
        self.intensity = intensity;
        self
    }

    /// Advance the clock.
    pub fn tick(&mut self, ticks: u64) {
        self.current_tick += ticks;
        for child in &mut self.children {
            if child.state == ChildState::Running {
                child.uptime_ticks += ticks;
            }
        }
    }

    /// Start a child process.
    pub fn start_child(&mut self, spec: ChildSpec) -> Result<(), SupervisorError> {
        if self.shutting_down {
            return Err(SupervisorError::ShuttingDown(self.id.clone()));
        }
        if self.children.iter().any(|c| c.spec.id == spec.id) {
            return Err(SupervisorError::DuplicateChild(spec.id));
        }
        let order = self.children.len();
        let child_id = spec.id.clone();
        self.children.push(Child::new(spec, order, self.current_tick));
        self.child_order.push(child_id.clone());
        self.events.push(SupervisorEvent::ChildStarted {
            supervisor: self.id.clone(),
            child: child_id,
        });
        Ok(())
    }

    /// Stop a specific child.
    pub fn stop_child(&mut self, child_id: &str) -> Result<(), SupervisorError> {
        let child = self
            .children
            .iter_mut()
            .find(|c| c.spec.id == child_id)
            .ok_or_else(|| SupervisorError::ChildNotFound(child_id.to_string()))?;
        child.state = ChildState::Stopped;
        self.events.push(SupervisorEvent::ChildStopped {
            supervisor: self.id.clone(),
            child: child_id.to_string(),
        });
        Ok(())
    }

    /// Report a child failure and apply the restart strategy.
    pub fn child_failed(
        &mut self,
        child_id: &str,
        reason: impl Into<String>,
    ) -> Result<Vec<String>, SupervisorError> {
        let reason_str = reason.into();
        let child_id_str = child_id.to_string();

        // Find the child
        let child_idx = self
            .children
            .iter()
            .position(|c| c.spec.id == child_id)
            .ok_or_else(|| SupervisorError::ChildNotFound(child_id_str.clone()))?;

        self.children[child_idx].state = ChildState::Failed;
        self.events.push(SupervisorEvent::ChildFailed {
            supervisor: self.id.clone(),
            child: child_id_str.clone(),
            reason: reason_str,
        });

        // Check restart type
        let restart_type = self.children[child_idx].spec.restart_type;
        if restart_type == RestartType::Temporary {
            return Ok(Vec::new());
        }

        // Check restart intensity
        self.check_intensity()?;

        // Apply strategy
        let restarted = match self.strategy {
            RestartStrategy::OneForOne => {
                self.restart_child(child_idx)?;
                vec![child_id_str]
            }
            RestartStrategy::OneForAll => {
                let ids: Vec<String> = self
                    .children
                    .iter()
                    .map(|c| c.spec.id.clone())
                    .collect();
                for i in 0..self.children.len() {
                    self.restart_child(i)?;
                }
                ids
            }
            RestartStrategy::RestForOne => {
                let start_order = self.children[child_idx].start_order;
                let indices: Vec<usize> = self
                    .children
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c.start_order >= start_order)
                    .map(|(i, _)| i)
                    .collect();
                let ids: Vec<String> = indices
                    .iter()
                    .map(|i| self.children[*i].spec.id.clone())
                    .collect();
                for i in indices {
                    self.restart_child(i)?;
                }
                ids
            }
        };

        Ok(restarted)
    }

    fn restart_child(&mut self, idx: usize) -> Result<(), SupervisorError> {
        let child = &mut self.children[idx];
        child.state = ChildState::Running;
        child.restart_count += 1;
        child.uptime_ticks = 0;
        child.started_at_tick = self.current_tick;
        let child_id = child.spec.id.clone();
        self.events.push(SupervisorEvent::ChildRestarted {
            supervisor: self.id.clone(),
            child: child_id,
        });
        Ok(())
    }

    fn check_intensity(&mut self) -> Result<(), SupervisorError> {
        let period = self.intensity.period_ticks as u64;
        let tick = self.current_tick;
        self.restart_timestamps
            .retain(|t| tick.saturating_sub(*t) < period);
        if self.restart_timestamps.len() as u32 >= self.intensity.max_restarts {
            self.events.push(SupervisorEvent::IntensityExceeded {
                supervisor: self.id.clone(),
            });
            return Err(SupervisorError::MaxIntensityExceeded {
                supervisor: self.id.clone(),
                restarts: self.intensity.max_restarts,
                period_ticks: self.intensity.period_ticks,
            });
        }
        self.restart_timestamps.push(tick);
        Ok(())
    }

    /// Graceful shutdown — stop all children in reverse start order.
    pub fn shutdown(&mut self) -> Vec<String> {
        self.shutting_down = true;
        let mut stopped = Vec::new();
        // Stop in reverse order
        let indices: Vec<usize> = (0..self.children.len()).rev().collect();
        for i in indices {
            if self.children[i].state == ChildState::Running
                || self.children[i].state == ChildState::Starting
            {
                self.children[i].state = ChildState::Stopped;
                let child_id = self.children[i].spec.id.clone();
                stopped.push(child_id.clone());
                self.events.push(SupervisorEvent::ChildStopped {
                    supervisor: self.id.clone(),
                    child: child_id,
                });
            }
        }
        self.events.push(SupervisorEvent::SupervisorShutDown {
            supervisor: self.id.clone(),
        });
        stopped
    }

    /// Get a child by ID.
    pub fn get_child(&self, id: &str) -> Option<&Child> {
        self.children.iter().find(|c| c.spec.id == id)
    }

    /// Number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Number of running children.
    pub fn running_count(&self) -> usize {
        self.children
            .iter()
            .filter(|c| c.state == ChildState::Running)
            .count()
    }

    /// Get the tree visualization as an indented string.
    pub fn tree_string(&self) -> String {
        let mut out = format!("[supervisor: {}] strategy={:?}\n", self.id, self.strategy);
        for child in &self.children {
            let type_str = match child.spec.child_type {
                ChildType::Worker => "worker",
                ChildType::Supervisor => "supervisor",
            };
            let state_str = match child.state {
                ChildState::Starting => "starting",
                ChildState::Running => "running",
                ChildState::Failed => "failed",
                ChildState::Stopped => "stopped",
                ChildState::Restarting => "restarting",
            };
            out.push_str(&format!(
                "  [{type_str}: {}] state={state_str} restarts={}\n",
                child.spec.id, child.restart_count
            ));
        }
        out
    }

    /// Whether the supervisor is shutting down.
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Child IDs in start order.
    pub fn child_ids(&self) -> Vec<String> {
        self.child_order.clone()
    }

    /// Drain events.
    pub fn drain_events(&mut self) -> Vec<SupervisorEvent> {
        std::mem::take(&mut self.events)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_supervisor(strategy: RestartStrategy) -> Supervisor {
        let mut sup = Supervisor::new("root", strategy);
        sup.start_child(ChildSpec::worker("w1")).unwrap();
        sup.start_child(ChildSpec::worker("w2")).unwrap();
        sup.start_child(ChildSpec::worker("w3")).unwrap();
        sup
    }

    #[test]
    fn test_start_children() {
        let sup = make_supervisor(RestartStrategy::OneForOne);
        assert_eq!(sup.child_count(), 3);
        assert_eq!(sup.running_count(), 3);
    }

    #[test]
    fn test_duplicate_child() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        let err = sup.start_child(ChildSpec::worker("w1")).unwrap_err();
        assert_eq!(err, SupervisorError::DuplicateChild("w1".to_string()));
    }

    #[test]
    fn test_one_for_one_restart() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        let restarted = sup.child_failed("w2", "crash").unwrap();
        assert_eq!(restarted, vec!["w2"]);
        assert_eq!(sup.get_child("w2").unwrap().state, ChildState::Running);
        assert_eq!(sup.get_child("w2").unwrap().restart_count, 1);
        // w1 and w3 unaffected
        assert_eq!(sup.get_child("w1").unwrap().restart_count, 0);
        assert_eq!(sup.get_child("w3").unwrap().restart_count, 0);
    }

    #[test]
    fn test_one_for_all_restart() {
        let mut sup = make_supervisor(RestartStrategy::OneForAll);
        let restarted = sup.child_failed("w2", "crash").unwrap();
        assert_eq!(restarted.len(), 3);
        assert_eq!(sup.get_child("w1").unwrap().restart_count, 1);
        assert_eq!(sup.get_child("w2").unwrap().restart_count, 1);
        assert_eq!(sup.get_child("w3").unwrap().restart_count, 1);
    }

    #[test]
    fn test_rest_for_one_restart() {
        let mut sup = make_supervisor(RestartStrategy::RestForOne);
        let restarted = sup.child_failed("w2", "crash").unwrap();
        // w2 (order 1) and w3 (order 2) should restart, not w1 (order 0)
        assert_eq!(restarted.len(), 2);
        assert!(restarted.contains(&"w2".to_string()));
        assert!(restarted.contains(&"w3".to_string()));
        assert_eq!(sup.get_child("w1").unwrap().restart_count, 0);
    }

    #[test]
    fn test_temporary_child_not_restarted() {
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne);
        sup.start_child(ChildSpec::worker("temp").with_restart(RestartType::Temporary))
            .unwrap();
        let restarted = sup.child_failed("temp", "done").unwrap();
        assert!(restarted.is_empty());
        assert_eq!(sup.get_child("temp").unwrap().state, ChildState::Failed);
    }

    #[test]
    fn test_max_intensity_exceeded() {
        let intensity = RestartIntensity {
            max_restarts: 2,
            period_ticks: 100,
        };
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne).with_intensity(intensity);
        sup.start_child(ChildSpec::worker("w1")).unwrap();

        sup.child_failed("w1", "crash 1").unwrap();
        sup.child_failed("w1", "crash 2").unwrap();
        let err = sup.child_failed("w1", "crash 3").unwrap_err();
        assert!(matches!(err, SupervisorError::MaxIntensityExceeded { .. }));
    }

    #[test]
    fn test_intensity_window_expires() {
        let intensity = RestartIntensity {
            max_restarts: 2,
            period_ticks: 50,
        };
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne).with_intensity(intensity);
        sup.start_child(ChildSpec::worker("w1")).unwrap();

        sup.child_failed("w1", "crash 1").unwrap();
        sup.child_failed("w1", "crash 2").unwrap();
        // Advance past window
        sup.tick(60);
        // Should succeed again
        sup.child_failed("w1", "crash 3").unwrap();
        assert_eq!(sup.get_child("w1").unwrap().restart_count, 3);
    }

    #[test]
    fn test_stop_child() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        sup.stop_child("w2").unwrap();
        assert_eq!(sup.get_child("w2").unwrap().state, ChildState::Stopped);
        assert_eq!(sup.running_count(), 2);
    }

    #[test]
    fn test_stop_nonexistent_child() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        let err = sup.stop_child("nope").unwrap_err();
        assert_eq!(err, SupervisorError::ChildNotFound("nope".to_string()));
    }

    #[test]
    fn test_graceful_shutdown() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        let stopped = sup.shutdown();
        // Should stop in reverse order
        assert_eq!(stopped, vec!["w3", "w2", "w1"]);
        assert_eq!(sup.running_count(), 0);
        assert!(sup.is_shutting_down());
    }

    #[test]
    fn test_shutdown_rejects_new_children() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        sup.shutdown();
        let err = sup.start_child(ChildSpec::worker("w4")).unwrap_err();
        assert_eq!(err, SupervisorError::ShuttingDown("root".to_string()));
    }

    #[test]
    fn test_child_uptime_tracking() {
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne);
        sup.start_child(ChildSpec::worker("w1")).unwrap();
        sup.tick(100);
        assert_eq!(sup.get_child("w1").unwrap().uptime_ticks, 100);
    }

    #[test]
    fn test_tree_string() {
        let sup = make_supervisor(RestartStrategy::OneForOne);
        let tree = sup.tree_string();
        assert!(tree.contains("[supervisor: root]"));
        assert!(tree.contains("[worker: w1]"));
        assert!(tree.contains("state=running"));
    }

    #[test]
    fn test_child_ids_order() {
        let sup = make_supervisor(RestartStrategy::OneForOne);
        assert_eq!(sup.child_ids(), vec!["w1", "w2", "w3"]);
    }

    #[test]
    fn test_events_emitted() {
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne);
        sup.start_child(ChildSpec::worker("w1")).unwrap();
        sup.child_failed("w1", "err").unwrap();
        let events = sup.drain_events();
        assert!(events.iter().any(|e| matches!(e, SupervisorEvent::ChildStarted { .. })));
        assert!(events.iter().any(|e| matches!(e, SupervisorEvent::ChildFailed { .. })));
        assert!(events.iter().any(|e| matches!(e, SupervisorEvent::ChildRestarted { .. })));
    }

    #[test]
    fn test_child_spec_supervisor_type() {
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne);
        sup.start_child(ChildSpec::supervisor("sub")).unwrap();
        assert_eq!(
            sup.get_child("sub").unwrap().spec.child_type,
            ChildType::Supervisor
        );
    }

    #[test]
    fn test_child_not_found_on_failure() {
        let mut sup = make_supervisor(RestartStrategy::OneForOne);
        let err = sup.child_failed("ghost", "err").unwrap_err();
        assert_eq!(err, SupervisorError::ChildNotFound("ghost".to_string()));
    }

    #[test]
    fn test_restart_resets_uptime() {
        let mut sup = Supervisor::new("root", RestartStrategy::OneForOne);
        sup.start_child(ChildSpec::worker("w1")).unwrap();
        sup.tick(50);
        assert_eq!(sup.get_child("w1").unwrap().uptime_ticks, 50);
        sup.child_failed("w1", "crash").unwrap();
        assert_eq!(sup.get_child("w1").unwrap().uptime_ticks, 0);
    }
}
