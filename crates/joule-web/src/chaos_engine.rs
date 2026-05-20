//! Chaos engineering framework: experiment definition, blast radius targeting,
//! steady state hypothesis, fault injection actions, rollback, experiment
//! results, and safety abort conditions.

use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Types ──

/// State of a chaos experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentState {
    Draft,
    Ready,
    Running,
    Paused,
    Completed,
    Aborted,
    RolledBack,
}

impl ExperimentState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExperimentState::Draft => "draft",
            ExperimentState::Ready => "ready",
            ExperimentState::Running => "running",
            ExperimentState::Paused => "paused",
            ExperimentState::Completed => "completed",
            ExperimentState::Aborted => "aborted",
            ExperimentState::RolledBack => "rolled_back",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ExperimentState::Completed | ExperimentState::Aborted | ExperimentState::RolledBack
        )
    }
}

/// Outcome of a hypothesis check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisResult {
    Passed,
    Failed,
    Inconclusive,
}

impl HypothesisResult {
    pub fn as_str(&self) -> &'static str {
        match self {
            HypothesisResult::Passed => "passed",
            HypothesisResult::Failed => "failed",
            HypothesisResult::Inconclusive => "inconclusive",
        }
    }
}

/// Kind of fault to inject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultKind {
    Latency { added_ms: u64 },
    PacketLoss { percent: u32 },
    ServiceKill { service_name: String },
    CpuStress { percent: u32, duration_secs: u64 },
    MemoryStress { megabytes: u64, duration_secs: u64 },
    DiskFill { megabytes: u64 },
    NetworkPartition { from: String, to: String },
    DnsFailure { domain: String },
    ClockSkew { offset_secs: i64 },
    Custom { name: String, params: String },
}

impl FaultKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            FaultKind::Latency { .. } => "latency",
            FaultKind::PacketLoss { .. } => "packet_loss",
            FaultKind::ServiceKill { .. } => "service_kill",
            FaultKind::CpuStress { .. } => "cpu_stress",
            FaultKind::MemoryStress { .. } => "memory_stress",
            FaultKind::DiskFill { .. } => "disk_fill",
            FaultKind::NetworkPartition { .. } => "network_partition",
            FaultKind::DnsFailure { .. } => "dns_failure",
            FaultKind::ClockSkew { .. } => "clock_skew",
            FaultKind::Custom { .. } => "custom",
        }
    }

    pub fn description(&self) -> String {
        match self {
            FaultKind::Latency { added_ms } => {
                format!("Add {}ms latency", added_ms)
            }
            FaultKind::PacketLoss { percent } => {
                format!("{}% packet loss", percent)
            }
            FaultKind::ServiceKill { service_name } => {
                format!("Kill service: {}", service_name)
            }
            FaultKind::CpuStress {
                percent,
                duration_secs,
            } => format!("CPU stress at {}% for {}s", percent, duration_secs),
            FaultKind::MemoryStress {
                megabytes,
                duration_secs,
            } => format!("Allocate {}MB for {}s", megabytes, duration_secs),
            FaultKind::DiskFill { megabytes } => {
                format!("Fill {}MB disk space", megabytes)
            }
            FaultKind::NetworkPartition { from, to } => {
                format!("Partition {} from {}", from, to)
            }
            FaultKind::DnsFailure { domain } => {
                format!("DNS failure for {}", domain)
            }
            FaultKind::ClockSkew { offset_secs } => {
                format!("Clock skew of {}s", offset_secs)
            }
            FaultKind::Custom { name, params } => {
                format!("Custom: {} ({})", name, params)
            }
        }
    }
}

/// Blast radius — defines which resources are targeted.
#[derive(Debug, Clone)]
pub struct BlastRadius {
    pub target_services: Vec<String>,
    pub target_hosts: Vec<String>,
    pub target_percentage: f64,
    pub excluded_services: Vec<String>,
    pub max_affected_instances: Option<u32>,
    pub environment: String,
}

impl BlastRadius {
    pub fn new(environment: &str) -> Self {
        Self {
            target_services: Vec::new(),
            target_hosts: Vec::new(),
            target_percentage: 100.0,
            excluded_services: Vec::new(),
            max_affected_instances: None,
            environment: environment.to_string(),
        }
    }

    pub fn add_target_service(&mut self, service: &str) {
        self.target_services.push(service.to_string());
    }

    pub fn add_target_host(&mut self, host: &str) {
        self.target_hosts.push(host.to_string());
    }

    pub fn exclude_service(&mut self, service: &str) {
        self.excluded_services.push(service.to_string());
    }

    pub fn with_percentage(mut self, pct: f64) -> Self {
        self.target_percentage = pct.clamp(0.0, 100.0);
        self
    }

    pub fn with_max_instances(mut self, max: u32) -> Self {
        self.max_affected_instances = Some(max);
        self
    }

    pub fn is_service_targeted(&self, service: &str) -> bool {
        if self.excluded_services.iter().any(|s| s == service) {
            return false;
        }
        if self.target_services.is_empty() {
            return true;
        }
        self.target_services.iter().any(|s| s == service)
    }

    pub fn total_targets(&self) -> usize {
        self.target_services.len() + self.target_hosts.len()
    }
}

/// Steady state hypothesis — defines expected behavior before and after fault.
#[derive(Debug, Clone)]
pub struct SteadyStateHypothesis {
    pub title: String,
    pub description: String,
    pub probes: Vec<Probe>,
}

impl SteadyStateHypothesis {
    pub fn new(title: &str, description: &str) -> Self {
        Self {
            title: title.to_string(),
            description: description.to_string(),
            probes: Vec::new(),
        }
    }

    pub fn add_probe(&mut self, probe: Probe) {
        self.probes.push(probe);
    }

    /// Check all probes, returning the overall result.
    pub fn evaluate(&self) -> HypothesisResult {
        if self.probes.is_empty() {
            return HypothesisResult::Inconclusive;
        }
        let all_passed = self.probes.iter().all(|p| p.result == Some(HypothesisResult::Passed));
        let any_failed = self.probes.iter().any(|p| p.result == Some(HypothesisResult::Failed));
        if any_failed {
            HypothesisResult::Failed
        } else if all_passed {
            HypothesisResult::Passed
        } else {
            HypothesisResult::Inconclusive
        }
    }
}

/// A single probe (check) in a hypothesis.
#[derive(Debug, Clone)]
pub struct Probe {
    pub name: String,
    pub probe_type: ProbeType,
    pub tolerance: f64,
    pub result: Option<HypothesisResult>,
    pub measured_value: Option<f64>,
    pub message: Option<String>,
}

impl Probe {
    pub fn new(name: &str, probe_type: ProbeType, tolerance: f64) -> Self {
        Self {
            name: name.to_string(),
            probe_type,
            tolerance,
            result: None,
            measured_value: None,
            message: None,
        }
    }

    /// Evaluate the probe with a measured value.
    pub fn evaluate(&mut self, measured: f64) {
        self.measured_value = Some(measured);
        let passed = match &self.probe_type {
            ProbeType::ErrorRateBelow { max_rate } => measured <= *max_rate + self.tolerance,
            ProbeType::LatencyBelow { max_ms } => measured <= *max_ms + self.tolerance,
            ProbeType::AvailabilityAbove { min_percent } => measured >= *min_percent - self.tolerance,
            ProbeType::ThroughputAbove { min_rps } => measured >= *min_rps - self.tolerance,
            ProbeType::Custom { check_value } => (measured - check_value).abs() <= self.tolerance,
        };
        self.result = Some(if passed {
            HypothesisResult::Passed
        } else {
            HypothesisResult::Failed
        });
    }
}

/// Type of probe.
#[derive(Debug, Clone)]
pub enum ProbeType {
    ErrorRateBelow { max_rate: f64 },
    LatencyBelow { max_ms: f64 },
    AvailabilityAbove { min_percent: f64 },
    ThroughputAbove { min_rps: f64 },
    Custom { check_value: f64 },
}

/// Safety abort condition.
#[derive(Debug, Clone)]
pub struct SafetyAbort {
    pub name: String,
    pub condition: AbortCondition,
    pub triggered: bool,
    pub triggered_at: Option<DateTime<Utc>>,
}

impl SafetyAbort {
    pub fn new(name: &str, condition: AbortCondition) -> Self {
        Self {
            name: name.to_string(),
            condition,
            triggered: false,
            triggered_at: None,
        }
    }

    /// Check whether this abort should trigger.
    pub fn check(&mut self, current_value: f64) -> bool {
        let should_abort = match &self.condition {
            AbortCondition::MetricExceeds { threshold, .. } => current_value > *threshold,
            AbortCondition::MetricBelow { threshold, .. } => current_value < *threshold,
            AbortCondition::DurationExceeds { max_seconds } => current_value > *max_seconds as f64,
        };
        if should_abort && !self.triggered {
            self.triggered = true;
            self.triggered_at = Some(Utc::now());
        }
        should_abort
    }
}

/// Conditions for aborting an experiment.
#[derive(Debug, Clone)]
pub enum AbortCondition {
    MetricExceeds { metric: String, threshold: f64 },
    MetricBelow { metric: String, threshold: f64 },
    DurationExceeds { max_seconds: u64 },
}

/// Rollback action to undo a fault.
#[derive(Debug, Clone)]
pub struct RollbackAction {
    pub name: String,
    pub description: String,
    pub executed: bool,
    pub executed_at: Option<DateTime<Utc>>,
    pub success: Option<bool>,
}

impl RollbackAction {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            executed: false,
            executed_at: None,
            success: None,
        }
    }

    pub fn execute(&mut self, success: bool) {
        self.executed = true;
        self.executed_at = Some(Utc::now());
        self.success = Some(success);
    }
}

/// Result of a completed experiment.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub experiment_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub duration: Duration,
    pub hypothesis_before: HypothesisResult,
    pub hypothesis_after: HypothesisResult,
    pub steady_state_preserved: bool,
    pub aborted: bool,
    pub abort_reason: Option<String>,
    pub findings: Vec<String>,
}

impl ExperimentResult {
    pub fn is_success(&self) -> bool {
        self.steady_state_preserved && !self.aborted
    }
}

/// An experiment event for the log.
#[derive(Debug, Clone)]
pub struct ExperimentEvent {
    pub timestamp: DateTime<Utc>,
    pub kind: ExperimentEventKind,
    pub message: String,
}

impl ExperimentEvent {
    pub fn new(kind: ExperimentEventKind, message: &str) -> Self {
        Self {
            timestamp: Utc::now(),
            kind,
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentEventKind {
    StateChanged,
    FaultInjected,
    FaultRemoved,
    ProbeChecked,
    SafetyAbort,
    RollbackExecuted,
    Finding,
}

// ── Experiment ──

/// A chaos experiment.
#[derive(Debug)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub state: ExperimentState,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,
    pub blast_radius: BlastRadius,
    pub hypothesis: SteadyStateHypothesis,
    pub faults: Vec<FaultKind>,
    pub rollback_actions: Vec<RollbackAction>,
    pub safety_aborts: Vec<SafetyAbort>,
    pub events: Vec<ExperimentEvent>,
    pub findings: Vec<String>,
    pub tags: HashMap<String, String>,
    hypothesis_before: Option<HypothesisResult>,
}

impl Experiment {
    pub fn new(name: &str, description: &str, blast_radius: BlastRadius) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: description.to_string(),
            state: ExperimentState::Draft,
            created_at: Utc::now(),
            started_at: None,
            ended_at: None,
            blast_radius,
            hypothesis: SteadyStateHypothesis::new("Default", "System should remain stable"),
            faults: Vec::new(),
            rollback_actions: Vec::new(),
            safety_aborts: Vec::new(),
            events: Vec::new(),
            findings: Vec::new(),
            tags: HashMap::new(),
            hypothesis_before: None,
        }
    }

    pub fn set_hypothesis(&mut self, hypothesis: SteadyStateHypothesis) {
        self.hypothesis = hypothesis;
    }

    pub fn add_fault(&mut self, fault: FaultKind) {
        let desc = fault.description();
        self.faults.push(fault);
        self.rollback_actions.push(RollbackAction::new(
            &format!("rollback_{}", self.faults.len()),
            &format!("Undo: {}", desc),
        ));
    }

    pub fn add_safety_abort(&mut self, abort: SafetyAbort) {
        self.safety_aborts.push(abort);
    }

    pub fn set_tag(&mut self, key: &str, value: &str) {
        self.tags.insert(key.to_string(), value.to_string());
    }

    /// Mark the experiment as ready to run.
    pub fn mark_ready(&mut self) -> Result<(), String> {
        if self.state != ExperimentState::Draft {
            return Err("Experiment must be in Draft state".to_string());
        }
        if self.faults.is_empty() {
            return Err("Experiment must have at least one fault".to_string());
        }
        self.state = ExperimentState::Ready;
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::StateChanged,
            "Experiment marked as ready",
        ));
        Ok(())
    }

    /// Start the experiment — first checks hypothesis, then injects faults.
    pub fn start(&mut self) -> Result<(), String> {
        if self.state != ExperimentState::Ready {
            return Err("Experiment must be Ready".to_string());
        }
        self.state = ExperimentState::Running;
        self.started_at = Some(Utc::now());

        // Check pre-condition hypothesis
        let before = self.hypothesis.evaluate();
        self.hypothesis_before = Some(before);
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::ProbeChecked,
            &format!("Pre-experiment hypothesis: {}", before.as_str()),
        ));

        // Log fault injection
        for fault in &self.faults {
            self.events.push(ExperimentEvent::new(
                ExperimentEventKind::FaultInjected,
                &fault.description(),
            ));
        }

        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::StateChanged,
            "Experiment started",
        ));
        Ok(())
    }

    /// Check safety abort conditions with a set of current values.
    pub fn check_safety(&mut self, metric_values: &HashMap<String, f64>) -> bool {
        let mut should_abort = false;
        for abort in &mut self.safety_aborts {
            let value = match &abort.condition {
                AbortCondition::MetricExceeds { metric, .. }
                | AbortCondition::MetricBelow { metric, .. } => {
                    metric_values.get(metric).copied().unwrap_or(0.0)
                }
                AbortCondition::DurationExceeds { .. } => {
                    self.started_at
                        .map(|s| (Utc::now() - s).num_seconds() as f64)
                        .unwrap_or(0.0)
                }
            };
            if abort.check(value) {
                should_abort = true;
            }
        }
        if should_abort {
            self.abort("Safety condition triggered");
        }
        should_abort
    }

    /// Abort the experiment and roll back.
    pub fn abort(&mut self, reason: &str) {
        self.state = ExperimentState::Aborted;
        self.ended_at = Some(Utc::now());
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::SafetyAbort,
            reason,
        ));
        self.execute_rollbacks();
    }

    /// Execute all rollback actions.
    pub fn execute_rollbacks(&mut self) {
        for action in &mut self.rollback_actions {
            if !action.executed {
                action.execute(true);
            }
        }
        if self.state == ExperimentState::Running {
            self.state = ExperimentState::RolledBack;
            self.ended_at = Some(Utc::now());
        }
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::RollbackExecuted,
            "All rollback actions executed",
        ));
    }

    /// Complete the experiment, checking the post-condition hypothesis.
    pub fn complete(&mut self) -> ExperimentResult {
        let hypothesis_after = self.hypothesis.evaluate();
        self.state = ExperimentState::Completed;
        self.ended_at = Some(Utc::now());

        self.execute_rollbacks();

        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::ProbeChecked,
            &format!("Post-experiment hypothesis: {}", hypothesis_after.as_str()),
        ));
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::StateChanged,
            "Experiment completed",
        ));

        let started = self.started_at.unwrap_or_else(Utc::now);
        let ended = self.ended_at.unwrap_or_else(Utc::now);
        let hypothesis_before = self.hypothesis_before.unwrap_or(HypothesisResult::Inconclusive);
        let preserved = hypothesis_after == HypothesisResult::Passed;

        ExperimentResult {
            experiment_id: self.id.clone(),
            started_at: started,
            ended_at: ended,
            duration: ended - started,
            hypothesis_before,
            hypothesis_after,
            steady_state_preserved: preserved,
            aborted: false,
            abort_reason: None,
            findings: self.findings.clone(),
        }
    }

    /// Add a finding during the experiment.
    pub fn add_finding(&mut self, finding: &str) {
        self.findings.push(finding.to_string());
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::Finding,
            finding,
        ));
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn duration(&self) -> Duration {
        let started = self.started_at.unwrap_or(self.created_at);
        let ended = self.ended_at.unwrap_or_else(Utc::now);
        ended - started
    }

    /// Pause the experiment.
    pub fn pause(&mut self) -> Result<(), String> {
        if self.state != ExperimentState::Running {
            return Err("Not running".to_string());
        }
        self.state = ExperimentState::Paused;
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::StateChanged,
            "Experiment paused",
        ));
        Ok(())
    }

    /// Resume the experiment.
    pub fn resume(&mut self) -> Result<(), String> {
        if self.state != ExperimentState::Paused {
            return Err("Not paused".to_string());
        }
        self.state = ExperimentState::Running;
        self.events.push(ExperimentEvent::new(
            ExperimentEventKind::StateChanged,
            "Experiment resumed",
        ));
        Ok(())
    }
}

// ── Experiment Registry ──

/// Registry of chaos experiments.
#[derive(Debug, Default)]
pub struct ExperimentRegistry {
    experiments: Vec<Experiment>,
}

impl ExperimentRegistry {
    pub fn new() -> Self {
        Self {
            experiments: Vec::new(),
        }
    }

    pub fn register(&mut self, experiment: Experiment) -> String {
        let id = experiment.id.clone();
        self.experiments.push(experiment);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Experiment> {
        self.experiments.iter().find(|e| e.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Experiment> {
        self.experiments.iter_mut().find(|e| e.id == id)
    }

    pub fn count(&self) -> usize {
        self.experiments.len()
    }

    pub fn running(&self) -> Vec<&Experiment> {
        self.experiments
            .iter()
            .filter(|e| e.state == ExperimentState::Running)
            .collect()
    }

    pub fn completed(&self) -> Vec<&Experiment> {
        self.experiments
            .iter()
            .filter(|e| e.state == ExperimentState::Completed)
            .collect()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn make_experiment() -> Experiment {
        let mut br = BlastRadius::new("staging");
        br.add_target_service("api-gateway");
        let mut exp = Experiment::new("Latency test", "Test latency injection", br);
        exp.add_fault(FaultKind::Latency { added_ms: 200 });
        exp
    }

    #[test]
    fn test_experiment_creation() {
        let exp = make_experiment();
        assert_eq!(exp.state, ExperimentState::Draft);
        assert_eq!(exp.faults.len(), 1);
        assert_eq!(exp.rollback_actions.len(), 1);
    }

    #[test]
    fn test_experiment_lifecycle() {
        let mut exp = make_experiment();
        assert!(exp.mark_ready().is_ok());
        assert_eq!(exp.state, ExperimentState::Ready);
        assert!(exp.start().is_ok());
        assert_eq!(exp.state, ExperimentState::Running);
        let result = exp.complete();
        assert_eq!(exp.state, ExperimentState::Completed);
        assert!(!result.aborted);
    }

    #[test]
    fn test_experiment_ready_requires_faults() {
        let br = BlastRadius::new("staging");
        let mut exp = Experiment::new("Empty", "No faults", br);
        assert!(exp.mark_ready().is_err());
    }

    #[test]
    fn test_experiment_start_requires_ready() {
        let mut exp = make_experiment();
        assert!(exp.start().is_err()); // Still in Draft
    }

    #[test]
    fn test_experiment_abort() {
        let mut exp = make_experiment();
        exp.mark_ready().unwrap();
        exp.start().unwrap();
        exp.abort("Manual abort");
        assert_eq!(exp.state, ExperimentState::Aborted);
        assert!(exp.ended_at.is_some());
        // Rollbacks should have been executed
        assert!(exp.rollback_actions[0].executed);
    }

    #[test]
    fn test_experiment_pause_resume() {
        let mut exp = make_experiment();
        exp.mark_ready().unwrap();
        exp.start().unwrap();
        assert!(exp.pause().is_ok());
        assert_eq!(exp.state, ExperimentState::Paused);
        assert!(exp.resume().is_ok());
        assert_eq!(exp.state, ExperimentState::Running);
    }

    #[test]
    fn test_blast_radius() {
        let mut br = BlastRadius::new("production");
        br.add_target_service("api");
        br.add_target_service("worker");
        br.exclude_service("database");
        assert!(br.is_service_targeted("api"));
        assert!(!br.is_service_targeted("database"));
        assert!(!br.is_service_targeted("unknown"));
        assert_eq!(br.total_targets(), 2);
    }

    #[test]
    fn test_blast_radius_empty_targets() {
        let br = BlastRadius::new("staging");
        // No targets → everything is targeted (except exclusions)
        assert!(br.is_service_targeted("any-service"));
    }

    #[test]
    fn test_blast_radius_percentage() {
        let br = BlastRadius::new("staging").with_percentage(50.0);
        assert!((br.target_percentage - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hypothesis_all_pass() {
        let mut h = SteadyStateHypothesis::new("Stable", "System should be stable");
        let mut p1 = Probe::new("error_rate", ProbeType::ErrorRateBelow { max_rate: 0.01 }, 0.001);
        p1.evaluate(0.005);
        let mut p2 = Probe::new("latency", ProbeType::LatencyBelow { max_ms: 100.0 }, 5.0);
        p2.evaluate(80.0);
        h.add_probe(p1);
        h.add_probe(p2);
        assert_eq!(h.evaluate(), HypothesisResult::Passed);
    }

    #[test]
    fn test_hypothesis_one_fails() {
        let mut h = SteadyStateHypothesis::new("Test", "Test");
        let mut p1 = Probe::new("ok", ProbeType::ErrorRateBelow { max_rate: 0.01 }, 0.0);
        p1.evaluate(0.005);
        let mut p2 = Probe::new("bad", ProbeType::LatencyBelow { max_ms: 50.0 }, 0.0);
        p2.evaluate(200.0);
        h.add_probe(p1);
        h.add_probe(p2);
        assert_eq!(h.evaluate(), HypothesisResult::Failed);
    }

    #[test]
    fn test_hypothesis_empty_probes() {
        let h = SteadyStateHypothesis::new("Empty", "No probes");
        assert_eq!(h.evaluate(), HypothesisResult::Inconclusive);
    }

    #[test]
    fn test_probe_evaluate() {
        let mut p = Probe::new(
            "avail",
            ProbeType::AvailabilityAbove { min_percent: 99.9 },
            0.01,
        );
        p.evaluate(99.95);
        assert_eq!(p.result, Some(HypothesisResult::Passed));
    }

    #[test]
    fn test_safety_abort() {
        let mut abort = SafetyAbort::new(
            "error_spike",
            AbortCondition::MetricExceeds {
                metric: "error_rate".to_string(),
                threshold: 0.1,
            },
        );
        assert!(!abort.check(0.05));
        assert!(abort.check(0.15));
        assert!(abort.triggered);
    }

    #[test]
    fn test_safety_abort_below() {
        let mut abort = SafetyAbort::new(
            "availability_drop",
            AbortCondition::MetricBelow {
                metric: "availability".to_string(),
                threshold: 99.0,
            },
        );
        assert!(!abort.check(99.5));
        assert!(abort.check(98.0));
    }

    #[test]
    fn test_rollback_action() {
        let mut action = RollbackAction::new("undo_latency", "Remove added latency");
        assert!(!action.executed);
        action.execute(true);
        assert!(action.executed);
        assert_eq!(action.success, Some(true));
    }

    #[test]
    fn test_fault_kind_descriptions() {
        let f = FaultKind::Latency { added_ms: 100 };
        assert!(f.description().contains("100ms"));
        assert_eq!(f.as_str(), "latency");

        let f = FaultKind::ServiceKill {
            service_name: "web".to_string(),
        };
        assert!(f.description().contains("web"));
    }

    #[test]
    fn test_experiment_check_safety() {
        let mut exp = make_experiment();
        exp.add_safety_abort(SafetyAbort::new(
            "error_spike",
            AbortCondition::MetricExceeds {
                metric: "error_rate".to_string(),
                threshold: 0.1,
            },
        ));
        exp.mark_ready().unwrap();
        exp.start().unwrap();

        let mut metrics = HashMap::new();
        metrics.insert("error_rate".to_string(), 0.5);
        assert!(exp.check_safety(&metrics));
        assert_eq!(exp.state, ExperimentState::Aborted);
    }

    #[test]
    fn test_experiment_registry() {
        let mut registry = ExperimentRegistry::new();
        let exp = make_experiment();
        let id = registry.register(exp);
        assert_eq!(registry.count(), 1);
        assert!(registry.get(&id).is_some());
    }

    #[test]
    fn test_experiment_findings() {
        let mut exp = make_experiment();
        exp.add_finding("Service recovers within 5 seconds");
        assert_eq!(exp.findings.len(), 1);
        assert!(exp.event_count() > 0);
    }

    #[test]
    fn test_experiment_result_success() {
        let mut exp = make_experiment();
        let mut h = SteadyStateHypothesis::new("Stable", "System stays stable");
        let mut probe = Probe::new("err", ProbeType::ErrorRateBelow { max_rate: 0.05 }, 0.0);
        probe.evaluate(0.01);
        h.add_probe(probe);
        exp.set_hypothesis(h);
        exp.mark_ready().unwrap();
        exp.start().unwrap();
        let result = exp.complete();
        assert!(result.is_success());
        assert!(result.steady_state_preserved);
    }

    #[test]
    fn test_experiment_state_terminal() {
        assert!(!ExperimentState::Draft.is_terminal());
        assert!(!ExperimentState::Running.is_terminal());
        assert!(ExperimentState::Completed.is_terminal());
        assert!(ExperimentState::Aborted.is_terminal());
        assert!(ExperimentState::RolledBack.is_terminal());
    }

    #[test]
    fn test_experiment_tags() {
        let mut exp = make_experiment();
        exp.set_tag("team", "platform");
        assert_eq!(exp.tags.get("team").unwrap(), "platform");
    }
}
