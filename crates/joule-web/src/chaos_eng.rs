//! Chaos engineering: fault injection (latency, error, exception),
//! blast radius control, steady-state hypothesis, experiment definition,
//! abort conditions, and experiment execution. Pure Rust — no real
//! fault injection; everything is modeled in-memory.

use std::collections::HashMap;
use std::fmt;

// ── Fault type ────────────────────────────────────────────────────

/// Kind of fault to inject.
#[derive(Debug, Clone, PartialEq)]
pub enum FaultType {
    /// Add latency to requests (microseconds).
    Latency(u64),
    /// Return an error response with a given status code.
    HttpError(u16),
    /// Simulate an exception/panic with a message.
    Exception(String),
    /// Simulate a connection reset.
    ConnectionReset,
    /// Simulate resource exhaustion (e.g. OOM).
    ResourceExhaustion(String),
    /// Custom fault with a name.
    Custom(String),
}

impl fmt::Display for FaultType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Latency(us) => write!(f, "latency:{us}us"),
            Self::HttpError(code) => write!(f, "http_error:{code}"),
            Self::Exception(msg) => write!(f, "exception:{msg}"),
            Self::ConnectionReset => write!(f, "connection_reset"),
            Self::ResourceExhaustion(kind) => write!(f, "resource_exhaustion:{kind}"),
            Self::Custom(name) => write!(f, "custom:{name}"),
        }
    }
}

// ── Blast radius ──────────────────────────────────────────────────

/// Controls the scope of fault injection.
#[derive(Debug, Clone)]
pub struct BlastRadius {
    /// Percentage of requests to affect (0-100).
    pub percentage: u8,
    /// Target services (empty = all).
    pub target_services: Vec<String>,
    /// Target endpoints (empty = all).
    pub target_endpoints: Vec<String>,
    /// Maximum concurrent faults allowed.
    pub max_concurrent: u32,
    /// Current concurrent fault count.
    current_concurrent: u32,
}

impl BlastRadius {
    /// Create a new blast radius configuration.
    pub fn new(percentage: u8) -> Self {
        Self {
            percentage: percentage.min(100),
            target_services: Vec::new(),
            target_endpoints: Vec::new(),
            max_concurrent: u32::MAX,
            current_concurrent: 0,
        }
    }

    /// Limit to specific services.
    pub fn with_services(mut self, services: Vec<String>) -> Self {
        self.target_services = services;
        self
    }

    /// Limit to specific endpoints.
    pub fn with_endpoints(mut self, endpoints: Vec<String>) -> Self {
        self.target_endpoints = endpoints;
        self
    }

    /// Set maximum concurrent faults.
    pub fn with_max_concurrent(mut self, max: u32) -> Self {
        self.max_concurrent = max;
        self
    }

    /// Check if a request should be faulted given a hash bucket (0-99),
    /// service name, and endpoint.
    pub fn should_inject(&self, hash_bucket: u8, service: &str, endpoint: &str) -> bool {
        if hash_bucket >= self.percentage {
            return false;
        }
        if self.current_concurrent >= self.max_concurrent {
            return false;
        }
        if !self.target_services.is_empty()
            && !self.target_services.iter().any(|s| s == service)
        {
            return false;
        }
        if !self.target_endpoints.is_empty()
            && !self.target_endpoints.iter().any(|e| e == endpoint)
        {
            return false;
        }
        true
    }

    /// Increment concurrent fault count.
    pub fn begin_fault(&mut self) {
        self.current_concurrent = self.current_concurrent.saturating_add(1);
    }

    /// Decrement concurrent fault count.
    pub fn end_fault(&mut self) {
        self.current_concurrent = self.current_concurrent.saturating_sub(1);
    }

    /// Current concurrent fault count.
    pub fn concurrent_count(&self) -> u32 {
        self.current_concurrent
    }
}

// ── Steady-state hypothesis ───────────────────────────────────────

/// A measurable assertion about normal system behavior.
#[derive(Debug, Clone)]
pub struct SteadyStateHypothesis {
    /// Human-readable description.
    pub description: String,
    /// Probes to measure the steady state.
    pub probes: Vec<SteadyStateProbe>,
}

/// A single measurement for the hypothesis.
#[derive(Debug, Clone)]
pub struct SteadyStateProbe {
    /// Name of the probe.
    pub name: String,
    /// Expected value type and threshold.
    pub expected: ProbeExpectation,
    /// Actual measured value (filled during execution).
    pub actual: Option<f64>,
}

/// What we expect a probe to show.
#[derive(Debug, Clone)]
pub enum ProbeExpectation {
    /// Value should be less than threshold.
    LessThan(f64),
    /// Value should be greater than threshold.
    GreaterThan(f64),
    /// Value should be within range.
    InRange(f64, f64),
    /// Value should equal (within epsilon).
    Equals(f64),
}

impl ProbeExpectation {
    /// Check if an actual value meets the expectation.
    pub fn is_met(&self, actual: f64) -> bool {
        match self {
            Self::LessThan(threshold) => actual < *threshold,
            Self::GreaterThan(threshold) => actual > *threshold,
            Self::InRange(min, max) => actual >= *min && actual <= *max,
            Self::Equals(expected) => (actual - expected).abs() < 1e-9,
        }
    }
}

impl SteadyStateHypothesis {
    /// Create a new hypothesis.
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            probes: Vec::new(),
        }
    }

    /// Add a probe.
    pub fn add_probe(&mut self, name: &str, expected: ProbeExpectation) {
        self.probes.push(SteadyStateProbe {
            name: name.to_string(),
            expected,
            actual: None,
        });
    }

    /// Record an actual measurement for a probe.
    pub fn record(&mut self, name: &str, value: f64) {
        if let Some(probe) = self.probes.iter_mut().find(|p| p.name == name) {
            probe.actual = Some(value);
        }
    }

    /// Check if all probes with measurements meet their expectations.
    pub fn is_met(&self) -> bool {
        self.probes.iter().all(|p| {
            match p.actual {
                Some(actual) => p.expected.is_met(actual),
                None => false, // unrecorded probes fail
            }
        })
    }
}

// ── Abort condition ───────────────────────────────────────────────

/// Conditions that cause an experiment to abort immediately.
#[derive(Debug, Clone)]
pub enum AbortCondition {
    /// Abort if error rate exceeds this threshold.
    ErrorRateExceeds(f64),
    /// Abort if latency (us) exceeds this threshold.
    LatencyExceeds(u64),
    /// Abort if a custom metric exceeds a threshold.
    MetricExceeds { metric: String, threshold: f64 },
    /// Abort after a time limit (time units).
    TimeLimit(u64),
}

impl AbortCondition {
    /// Check if this condition is triggered.
    pub fn is_triggered(
        &self,
        error_rate: f64,
        latency_us: u64,
        custom_metrics: &HashMap<String, f64>,
        elapsed_units: u64,
    ) -> bool {
        match self {
            Self::ErrorRateExceeds(threshold) => error_rate > *threshold,
            Self::LatencyExceeds(threshold) => latency_us > *threshold,
            Self::MetricExceeds { metric, threshold } => {
                custom_metrics
                    .get(metric)
                    .map(|v| *v > *threshold)
                    .unwrap_or(false)
            }
            Self::TimeLimit(limit) => elapsed_units >= *limit,
        }
    }
}

// ── Experiment status ─────────────────────────────────────────────

/// Status of a chaos experiment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentStatus {
    /// Not yet started.
    Pending,
    /// Currently running.
    Running,
    /// Completed successfully (hypothesis held).
    Completed,
    /// Aborted due to safety condition.
    Aborted,
    /// Failed (hypothesis violated).
    Failed,
}

impl fmt::Display for ExperimentStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Aborted => write!(f, "aborted"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

// ── Experiment event ──────────────────────────────────────────────

/// An event during experiment execution.
#[derive(Debug, Clone)]
pub struct ExperimentEvent {
    pub status: ExperimentStatus,
    pub message: String,
    pub timestamp_unit: u64,
}

// ── Chaos experiment ──────────────────────────────────────────────

/// A chaos engineering experiment definition and execution state.
#[derive(Debug, Clone)]
pub struct ChaosExperiment {
    /// Experiment name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Fault to inject.
    pub fault: FaultType,
    /// Blast radius control.
    pub blast_radius: BlastRadius,
    /// Steady-state hypothesis.
    pub hypothesis: SteadyStateHypothesis,
    /// Abort conditions.
    pub abort_conditions: Vec<AbortCondition>,
    /// Current status.
    pub status: ExperimentStatus,
    /// Elapsed time units.
    pub elapsed_units: u64,
    /// Number of faults injected.
    pub faults_injected: u64,
    /// Event log.
    events: Vec<ExperimentEvent>,
}

impl ChaosExperiment {
    /// Create a new experiment.
    pub fn new(
        name: &str,
        description: &str,
        fault: FaultType,
        blast_radius: BlastRadius,
        hypothesis: SteadyStateHypothesis,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            fault,
            blast_radius,
            hypothesis,
            abort_conditions: Vec::new(),
            status: ExperimentStatus::Pending,
            elapsed_units: 0,
            faults_injected: 0,
            events: Vec::new(),
        }
    }

    /// Add an abort condition.
    pub fn with_abort(mut self, condition: AbortCondition) -> Self {
        self.abort_conditions.push(condition);
        self
    }

    /// Start the experiment.
    pub fn start(&mut self, timestamp: u64) {
        self.status = ExperimentStatus::Running;
        self.elapsed_units = 0;
        self.faults_injected = 0;
        self.log_event(timestamp, format!("experiment '{}' started", self.name));
    }

    /// Simulate one tick: inject a fault if conditions allow and check aborts.
    /// Returns whether a fault was injected in this tick.
    pub fn tick(
        &mut self,
        hash_bucket: u8,
        service: &str,
        endpoint: &str,
        error_rate: f64,
        latency_us: u64,
        custom_metrics: &HashMap<String, f64>,
        timestamp: u64,
    ) -> bool {
        if self.status != ExperimentStatus::Running {
            return false;
        }

        self.elapsed_units = self.elapsed_units.saturating_add(1);

        // Check abort conditions.
        for cond in &self.abort_conditions {
            if cond.is_triggered(error_rate, latency_us, custom_metrics, self.elapsed_units)
            {
                self.status = ExperimentStatus::Aborted;
                self.blast_radius.end_fault();
                self.log_event(timestamp, "experiment aborted: safety condition triggered".into());
                return false;
            }
        }

        // Try to inject a fault.
        if self.blast_radius.should_inject(hash_bucket, service, endpoint) {
            self.blast_radius.begin_fault();
            self.faults_injected = self.faults_injected.saturating_add(1);
            self.blast_radius.end_fault();
            return true;
        }

        false
    }

    /// Evaluate the hypothesis and finalize the experiment.
    pub fn evaluate(&mut self, timestamp: u64) {
        if self.status != ExperimentStatus::Running {
            return;
        }

        if self.hypothesis.is_met() {
            self.status = ExperimentStatus::Completed;
            self.log_event(timestamp, "hypothesis held; experiment completed".into());
        } else {
            self.status = ExperimentStatus::Failed;
            self.log_event(timestamp, "hypothesis violated; experiment failed".into());
        }
    }

    /// Get the event log.
    pub fn events(&self) -> &[ExperimentEvent] {
        &self.events
    }

    fn log_event(&mut self, timestamp_unit: u64, message: String) {
        self.events.push(ExperimentEvent {
            status: self.status,
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
    fn fault_type_display() {
        assert_eq!(FaultType::Latency(500).to_string(), "latency:500us");
        assert_eq!(FaultType::HttpError(503).to_string(), "http_error:503");
        assert_eq!(
            FaultType::Exception("boom".into()).to_string(),
            "exception:boom"
        );
        assert_eq!(
            FaultType::ConnectionReset.to_string(),
            "connection_reset"
        );
        assert_eq!(
            FaultType::ResourceExhaustion("memory".into()).to_string(),
            "resource_exhaustion:memory"
        );
        assert_eq!(
            FaultType::Custom("disk_full".into()).to_string(),
            "custom:disk_full"
        );
    }

    #[test]
    fn blast_radius_percentage() {
        let br = BlastRadius::new(10);
        assert!(br.should_inject(0, "svc", "/api"));
        assert!(br.should_inject(9, "svc", "/api"));
        assert!(!br.should_inject(10, "svc", "/api"));
        assert!(!br.should_inject(99, "svc", "/api"));
    }

    #[test]
    fn blast_radius_clamped() {
        let br = BlastRadius::new(200);
        assert_eq!(br.percentage, 100);
    }

    #[test]
    fn blast_radius_service_filter() {
        let br = BlastRadius::new(100)
            .with_services(vec!["api".into()]);
        assert!(br.should_inject(0, "api", "/health"));
        assert!(!br.should_inject(0, "frontend", "/"));
    }

    #[test]
    fn blast_radius_endpoint_filter() {
        let br = BlastRadius::new(100)
            .with_endpoints(vec!["/api/orders".into()]);
        assert!(br.should_inject(0, "svc", "/api/orders"));
        assert!(!br.should_inject(0, "svc", "/api/users"));
    }

    #[test]
    fn blast_radius_max_concurrent() {
        let mut br = BlastRadius::new(100).with_max_concurrent(2);
        assert!(br.should_inject(0, "s", "/e"));
        br.begin_fault();
        assert!(br.should_inject(0, "s", "/e"));
        br.begin_fault();
        assert!(!br.should_inject(0, "s", "/e")); // at max
        br.end_fault();
        assert!(br.should_inject(0, "s", "/e"));
    }

    #[test]
    fn blast_radius_concurrent_count() {
        let mut br = BlastRadius::new(50);
        assert_eq!(br.concurrent_count(), 0);
        br.begin_fault();
        assert_eq!(br.concurrent_count(), 1);
        br.end_fault();
        assert_eq!(br.concurrent_count(), 0);
    }

    #[test]
    fn probe_expectation_less_than() {
        let e = ProbeExpectation::LessThan(5.0);
        assert!(e.is_met(4.9));
        assert!(!e.is_met(5.0));
        assert!(!e.is_met(5.1));
    }

    #[test]
    fn probe_expectation_greater_than() {
        let e = ProbeExpectation::GreaterThan(10.0);
        assert!(e.is_met(10.1));
        assert!(!e.is_met(10.0));
    }

    #[test]
    fn probe_expectation_in_range() {
        let e = ProbeExpectation::InRange(1.0, 5.0);
        assert!(e.is_met(1.0));
        assert!(e.is_met(3.0));
        assert!(e.is_met(5.0));
        assert!(!e.is_met(0.9));
        assert!(!e.is_met(5.1));
    }

    #[test]
    fn probe_expectation_equals() {
        let e = ProbeExpectation::Equals(42.0);
        assert!(e.is_met(42.0));
        assert!(!e.is_met(42.1));
    }

    #[test]
    fn hypothesis_all_met() {
        let mut h = SteadyStateHypothesis::new("error rate stays low");
        h.add_probe("error_rate", ProbeExpectation::LessThan(0.05));
        h.add_probe("p99_latency", ProbeExpectation::LessThan(500.0));

        assert!(!h.is_met()); // no measurements yet

        h.record("error_rate", 0.02);
        h.record("p99_latency", 300.0);
        assert!(h.is_met());
    }

    #[test]
    fn hypothesis_one_violated() {
        let mut h = SteadyStateHypothesis::new("test");
        h.add_probe("a", ProbeExpectation::LessThan(1.0));
        h.add_probe("b", ProbeExpectation::LessThan(1.0));
        h.record("a", 0.5);
        h.record("b", 2.0);
        assert!(!h.is_met());
    }

    #[test]
    fn hypothesis_unrecorded_fails() {
        let mut h = SteadyStateHypothesis::new("test");
        h.add_probe("a", ProbeExpectation::LessThan(1.0));
        assert!(!h.is_met());
    }

    #[test]
    fn abort_error_rate() {
        let cond = AbortCondition::ErrorRateExceeds(0.1);
        let empty = HashMap::new();
        assert!(cond.is_triggered(0.2, 0, &empty, 0));
        assert!(!cond.is_triggered(0.05, 0, &empty, 0));
    }

    #[test]
    fn abort_latency() {
        let cond = AbortCondition::LatencyExceeds(5000);
        let empty = HashMap::new();
        assert!(cond.is_triggered(0.0, 6000, &empty, 0));
        assert!(!cond.is_triggered(0.0, 4000, &empty, 0));
    }

    #[test]
    fn abort_custom_metric() {
        let cond = AbortCondition::MetricExceeds {
            metric: "cpu".into(),
            threshold: 90.0,
        };
        let mut m = HashMap::new();
        m.insert("cpu".into(), 95.0);
        assert!(cond.is_triggered(0.0, 0, &m, 0));
    }

    #[test]
    fn abort_time_limit() {
        let cond = AbortCondition::TimeLimit(100);
        let empty = HashMap::new();
        assert!(cond.is_triggered(0.0, 0, &empty, 100));
        assert!(!cond.is_triggered(0.0, 0, &empty, 99));
    }

    #[test]
    fn abort_missing_metric() {
        let cond = AbortCondition::MetricExceeds {
            metric: "cpu".into(),
            threshold: 90.0,
        };
        let empty = HashMap::new();
        assert!(!cond.is_triggered(0.0, 0, &empty, 0));
    }

    #[test]
    fn experiment_status_display() {
        assert_eq!(ExperimentStatus::Pending.to_string(), "pending");
        assert_eq!(ExperimentStatus::Running.to_string(), "running");
        assert_eq!(ExperimentStatus::Completed.to_string(), "completed");
        assert_eq!(ExperimentStatus::Aborted.to_string(), "aborted");
        assert_eq!(ExperimentStatus::Failed.to_string(), "failed");
    }

    #[test]
    fn experiment_lifecycle_completed() {
        let mut h = SteadyStateHypothesis::new("system stays healthy");
        h.add_probe("error_rate", ProbeExpectation::LessThan(0.1));

        let mut exp = ChaosExperiment::new(
            "latency-test",
            "inject 500ms latency",
            FaultType::Latency(500_000),
            BlastRadius::new(10),
            h,
        )
        .with_abort(AbortCondition::ErrorRateExceeds(0.5));

        assert_eq!(exp.status, ExperimentStatus::Pending);

        exp.start(0);
        assert_eq!(exp.status, ExperimentStatus::Running);

        // Simulate ticks.
        let empty = HashMap::new();
        let injected = exp.tick(5, "api", "/orders", 0.01, 100, &empty, 1);
        assert!(injected);
        assert_eq!(exp.faults_injected, 1);

        let not_injected = exp.tick(50, "api", "/orders", 0.01, 100, &empty, 2);
        assert!(!not_injected);

        // Evaluate hypothesis.
        exp.hypothesis.record("error_rate", 0.02);
        exp.evaluate(3);
        assert_eq!(exp.status, ExperimentStatus::Completed);
    }

    #[test]
    fn experiment_aborted() {
        let h = SteadyStateHypothesis::new("test");
        let mut exp = ChaosExperiment::new(
            "error-test",
            "inject errors",
            FaultType::HttpError(500),
            BlastRadius::new(50),
            h,
        )
        .with_abort(AbortCondition::ErrorRateExceeds(0.3));

        exp.start(0);
        let empty = HashMap::new();
        exp.tick(0, "svc", "/api", 0.5, 0, &empty, 1);
        assert_eq!(exp.status, ExperimentStatus::Aborted);
    }

    #[test]
    fn experiment_failed() {
        let mut h = SteadyStateHypothesis::new("test");
        h.add_probe("p99", ProbeExpectation::LessThan(100.0));

        let mut exp = ChaosExperiment::new(
            "test",
            "desc",
            FaultType::Latency(1000),
            BlastRadius::new(10),
            h,
        );

        exp.start(0);
        exp.hypothesis.record("p99", 200.0); // violated
        exp.evaluate(1);
        assert_eq!(exp.status, ExperimentStatus::Failed);
    }

    #[test]
    fn experiment_tick_when_not_running() {
        let h = SteadyStateHypothesis::new("test");
        let mut exp = ChaosExperiment::new(
            "test",
            "desc",
            FaultType::Latency(100),
            BlastRadius::new(100),
            h,
        );
        let empty = HashMap::new();
        let injected = exp.tick(0, "s", "/e", 0.0, 0, &empty, 0);
        assert!(!injected);
    }

    #[test]
    fn experiment_events_logged() {
        let h = SteadyStateHypothesis::new("test");
        let mut exp = ChaosExperiment::new(
            "test",
            "desc",
            FaultType::ConnectionReset,
            BlastRadius::new(10),
            h,
        );
        exp.start(0);
        assert!(!exp.events().is_empty());
        assert!(exp.events()[0].message.contains("started"));
    }

    #[test]
    fn evaluate_when_not_running() {
        let h = SteadyStateHypothesis::new("test");
        let mut exp = ChaosExperiment::new(
            "test",
            "desc",
            FaultType::Latency(100),
            BlastRadius::new(10),
            h,
        );
        exp.evaluate(0); // should do nothing since still Pending
        assert_eq!(exp.status, ExperimentStatus::Pending);
    }

    #[test]
    fn blast_radius_no_filters_all_pass() {
        let br = BlastRadius::new(100);
        assert!(br.should_inject(0, "any_service", "/any/endpoint"));
    }

    #[test]
    fn hypothesis_empty_probes_met() {
        let h = SteadyStateHypothesis::new("trivial");
        assert!(h.is_met()); // no probes = trivially met
    }

    #[test]
    fn abort_time_limit_not_triggered_before() {
        let cond = AbortCondition::TimeLimit(10);
        let empty = HashMap::new();
        assert!(!cond.is_triggered(0.0, 0, &empty, 9));
        assert!(cond.is_triggered(0.0, 0, &empty, 10));
    }

    #[test]
    fn experiment_elapsed_increments() {
        let h = SteadyStateHypothesis::new("t");
        let mut exp = ChaosExperiment::new(
            "test",
            "",
            FaultType::Latency(0),
            BlastRadius::new(0),
            h,
        );
        exp.start(0);
        let empty = HashMap::new();
        exp.tick(50, "s", "/e", 0.0, 0, &empty, 1);
        exp.tick(50, "s", "/e", 0.0, 0, &empty, 2);
        exp.tick(50, "s", "/e", 0.0, 0, &empty, 3);
        assert_eq!(exp.elapsed_units, 3);
    }
}
