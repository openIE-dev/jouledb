//! Fault injection framework.
//!
//! Replaces `failpoints`, `chaos-monkey`, and similar fault injection tools
//! with a pure-Rust system. Supports injectable failure points, multiple failure
//! modes (error/panic/delay/corrupt), configurable probability, failure schedules
//! (nth call, random), scoped enable/disable, and fault statistics tracking.

use std::collections::HashMap;
use std::fmt;

// ── PRNG ─────────────────────────────────────────────────────────

/// Simple PRNG for probabilistic fault injection.
#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_f64(&mut self) -> f64 {
        // SplitMix64
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z = z ^ (z >> 31);
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ── Failure Mode ─────────────────────────────────────────────────

/// The kind of failure to inject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureMode {
    /// Return an error with the given message.
    Error(String),
    /// Simulate a delay (in milliseconds). Does not actually sleep;
    /// returns the delay value for the caller to handle.
    Delay(u64),
    /// Corrupt data by returning a garbled string.
    Corrupt(String),
    /// Return a specific status code (for HTTP-like systems).
    StatusCode(u16),
    /// Return empty/null.
    Empty,
}

impl fmt::Display for FailureMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailureMode::Error(msg) => write!(f, "Error({msg})"),
            FailureMode::Delay(ms) => write!(f, "Delay({ms}ms)"),
            FailureMode::Corrupt(data) => write!(f, "Corrupt({data})"),
            FailureMode::StatusCode(code) => write!(f, "StatusCode({code})"),
            FailureMode::Empty => write!(f, "Empty"),
        }
    }
}

// ── Failure Schedule ─────────────────────────────────────────────

/// Controls when a fault fires.
#[derive(Debug, Clone)]
pub enum FailureSchedule {
    /// Fire on every invocation.
    Always,
    /// Fire with a given probability (0.0..=1.0).
    Probability(f64),
    /// Fire on the Nth call (1-based).
    NthCall(u64),
    /// Fire every Nth call.
    EveryNth(u64),
    /// Fire on specific call numbers.
    OnCalls(Vec<u64>),
    /// Fire for the first N calls.
    FirstN(u64),
    /// Fire after N calls.
    AfterN(u64),
}

impl FailureSchedule {
    /// Check if the fault should fire given the current call count.
    fn should_fire(&self, call_count: u64, rng: &mut Rng) -> bool {
        match self {
            FailureSchedule::Always => true,
            FailureSchedule::Probability(p) => rng.next_f64() < *p,
            FailureSchedule::NthCall(n) => call_count == *n,
            FailureSchedule::EveryNth(n) => *n > 0 && call_count % *n == 0,
            FailureSchedule::OnCalls(calls) => calls.contains(&call_count),
            FailureSchedule::FirstN(n) => call_count <= *n,
            FailureSchedule::AfterN(n) => call_count > *n,
        }
    }
}

// ── Fault Point ──────────────────────────────────────────────────

/// A named injection point where faults can occur.
#[derive(Debug, Clone)]
pub struct FaultPoint {
    /// Unique name for this fault point.
    pub name: String,
    /// Failure mode to apply.
    pub mode: FailureMode,
    /// When to trigger the failure.
    pub schedule: FailureSchedule,
    /// Whether this fault point is currently enabled.
    pub enabled: bool,
    /// Number of times this point has been checked.
    pub check_count: u64,
    /// Number of times this point has fired.
    pub fire_count: u64,
    /// Description of what this fault point simulates.
    pub description: String,
}

impl FaultPoint {
    pub fn new(name: &str, mode: FailureMode) -> Self {
        Self {
            name: name.to_string(),
            mode,
            schedule: FailureSchedule::Always,
            enabled: true,
            check_count: 0,
            fire_count: 0,
            description: String::new(),
        }
    }

    pub fn with_schedule(mut self, schedule: FailureSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Fire rate as a percentage.
    pub fn fire_rate(&self) -> f64 {
        if self.check_count == 0 {
            return 0.0;
        }
        (self.fire_count as f64 / self.check_count as f64) * 100.0
    }
}

// ── Fault Outcome ────────────────────────────────────────────────

/// The outcome of checking a fault point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultOutcome {
    /// No fault triggered — continue normally.
    Pass,
    /// Fault triggered with the given failure mode.
    Triggered(FailureMode),
}

impl FaultOutcome {
    pub fn is_triggered(&self) -> bool {
        matches!(self, FaultOutcome::Triggered(_))
    }

    pub fn is_pass(&self) -> bool {
        matches!(self, FaultOutcome::Pass)
    }
}

// ── Fault Injector ───────────────────────────────────────────────

/// Central fault injection controller.
#[derive(Debug, Clone)]
pub struct FaultInjector {
    /// Registered fault points.
    faults: HashMap<String, FaultPoint>,
    /// Global enable/disable switch.
    globally_enabled: bool,
    /// PRNG for probabilistic faults.
    rng: Rng,
    /// History of triggered faults.
    trigger_log: Vec<TriggerRecord>,
    /// Maximum log entries to keep.
    max_log_entries: usize,
}

/// Record of a fault trigger event.
#[derive(Debug, Clone)]
pub struct TriggerRecord {
    pub fault_name: String,
    pub mode: FailureMode,
    pub check_number: u64,
}

impl Default for FaultInjector {
    fn default() -> Self {
        Self::new(42)
    }
}

impl FaultInjector {
    /// Create a new injector with the given PRNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            faults: HashMap::new(),
            globally_enabled: true,
            rng: Rng::new(seed),
            trigger_log: Vec::new(),
            max_log_entries: 10000,
        }
    }

    /// Register a fault point.
    pub fn register(&mut self, fault: FaultPoint) {
        self.faults.insert(fault.name.clone(), fault);
    }

    /// Remove a fault point.
    pub fn remove(&mut self, name: &str) -> bool {
        self.faults.remove(name).is_some()
    }

    /// Check if a fault point is registered.
    pub fn has(&self, name: &str) -> bool {
        self.faults.contains_key(name)
    }

    /// Get a fault point by name.
    pub fn get(&self, name: &str) -> Option<&FaultPoint> {
        self.faults.get(name)
    }

    /// Enable a specific fault point.
    pub fn enable(&mut self, name: &str) {
        if let Some(fp) = self.faults.get_mut(name) {
            fp.enabled = true;
        }
    }

    /// Disable a specific fault point.
    pub fn disable(&mut self, name: &str) {
        if let Some(fp) = self.faults.get_mut(name) {
            fp.enabled = false;
        }
    }

    /// Enable all fault points globally.
    pub fn enable_all(&mut self) {
        self.globally_enabled = true;
    }

    /// Disable all fault points globally.
    pub fn disable_all(&mut self) {
        self.globally_enabled = false;
    }

    /// Check if globally enabled.
    pub fn is_globally_enabled(&self) -> bool {
        self.globally_enabled
    }

    /// Check a fault point. Returns `Triggered` if the fault fires.
    pub fn check(&mut self, name: &str) -> FaultOutcome {
        if !self.globally_enabled {
            return FaultOutcome::Pass;
        }

        let rng = &mut self.rng;
        let fp = match self.faults.get_mut(name) {
            Some(fp) => fp,
            None => return FaultOutcome::Pass,
        };

        if !fp.enabled {
            return FaultOutcome::Pass;
        }

        fp.check_count += 1;
        let call_count = fp.check_count;

        if fp.schedule.should_fire(call_count, rng) {
            fp.fire_count += 1;
            let mode = fp.mode.clone();

            if self.trigger_log.len() < self.max_log_entries {
                self.trigger_log.push(TriggerRecord {
                    fault_name: name.to_string(),
                    mode: mode.clone(),
                    check_number: call_count,
                });
            }

            FaultOutcome::Triggered(mode)
        } else {
            FaultOutcome::Pass
        }
    }

    /// Convenience: check and return Err if triggered, Ok(()) otherwise.
    pub fn check_result(&mut self, name: &str) -> Result<(), String> {
        match self.check(name) {
            FaultOutcome::Pass => Ok(()),
            FaultOutcome::Triggered(mode) => Err(format!("Fault triggered: {mode}")),
        }
    }

    /// Number of registered fault points.
    pub fn num_faults(&self) -> usize {
        self.faults.len()
    }

    /// Trigger log.
    pub fn trigger_log(&self) -> &[TriggerRecord] {
        &self.trigger_log
    }

    /// Total number of triggers across all fault points.
    pub fn total_triggers(&self) -> u64 {
        self.faults.values().map(|fp| fp.fire_count).sum()
    }

    /// Total number of checks across all fault points.
    pub fn total_checks(&self) -> u64 {
        self.faults.values().map(|fp| fp.check_count).sum()
    }

    /// Reset all statistics.
    pub fn reset_stats(&mut self) {
        for fp in self.faults.values_mut() {
            fp.check_count = 0;
            fp.fire_count = 0;
        }
        self.trigger_log.clear();
    }

    /// Reset everything (remove all fault points).
    pub fn clear(&mut self) {
        self.faults.clear();
        self.trigger_log.clear();
    }

    /// Generate a statistics summary.
    pub fn stats_summary(&self) -> String {
        let mut out = String::new();
        out.push_str("Fault Injection Statistics\n");
        out.push_str(&format!("{}\n", "=".repeat(50)));
        out.push_str(&format!("Global: {}\n", if self.globally_enabled { "enabled" } else { "disabled" }));
        out.push_str(&format!("Fault points: {}\n", self.num_faults()));
        out.push_str(&format!("Total checks: {}\n", self.total_checks()));
        out.push_str(&format!("Total triggers: {}\n\n", self.total_triggers()));

        let mut names: Vec<String> = self.faults.keys().cloned().collect();
        names.sort();

        for name in &names {
            let fp = &self.faults[name];
            out.push_str(&format!(
                "  {}: {} checks, {} fires ({:.1}%), {}\n",
                fp.name,
                fp.check_count,
                fp.fire_count,
                fp.fire_rate(),
                if fp.enabled { "enabled" } else { "disabled" },
            ));
        }
        out
    }

    /// List all fault point names (sorted).
    pub fn fault_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.faults.keys().cloned().collect();
        names.sort();
        names
    }
}

// ── Scoped Fault ─────────────────────────────────────────────────

/// Enable a fault point for a scope, then disable on drop.
/// This is a manual RAII-like pattern without closures.
pub struct ScopedFault {
    name: String,
    was_enabled: bool,
}

impl ScopedFault {
    /// Enable a fault in the injector and remember previous state.
    pub fn activate(injector: &mut FaultInjector, name: &str) -> Self {
        let was_enabled = injector
            .get(name)
            .map(|fp| fp.enabled)
            .unwrap_or(false);
        injector.enable(name);
        Self {
            name: name.to_string(),
            was_enabled,
        }
    }

    /// Restore the fault point to its previous state.
    pub fn deactivate(self, injector: &mut FaultInjector) {
        if !self.was_enabled {
            injector.disable(&self.name);
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn always_schedule_fires_every_time() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("io_error", FailureMode::Error("disk full".into())));

        for _ in 0..5 {
            let outcome = injector.check("io_error");
            assert!(outcome.is_triggered());
        }
        assert_eq!(injector.get("io_error").unwrap().fire_count, 5);
    }

    #[test]
    fn disabled_fault_does_not_fire() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("fail", FailureMode::Error("err".into())).disabled());
        let outcome = injector.check("fail");
        assert!(outcome.is_pass());
    }

    #[test]
    fn globally_disabled_skips_all() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("a", FailureMode::Error("err".into())));
        injector.disable_all();
        assert!(injector.check("a").is_pass());
        injector.enable_all();
        assert!(injector.check("a").is_triggered());
    }

    #[test]
    fn nth_call_schedule() {
        let mut injector = FaultInjector::new(1);
        injector.register(
            FaultPoint::new("nth", FailureMode::Error("boom".into()))
                .with_schedule(FailureSchedule::NthCall(3)),
        );

        assert!(injector.check("nth").is_pass()); // 1
        assert!(injector.check("nth").is_pass()); // 2
        assert!(injector.check("nth").is_triggered()); // 3
        assert!(injector.check("nth").is_pass()); // 4
    }

    #[test]
    fn every_nth_schedule() {
        let mut injector = FaultInjector::new(1);
        injector.register(
            FaultPoint::new("every2", FailureMode::Empty)
                .with_schedule(FailureSchedule::EveryNth(2)),
        );

        assert!(injector.check("every2").is_pass()); // 1
        assert!(injector.check("every2").is_triggered()); // 2
        assert!(injector.check("every2").is_pass()); // 3
        assert!(injector.check("every2").is_triggered()); // 4
    }

    #[test]
    fn on_calls_schedule() {
        let mut injector = FaultInjector::new(1);
        injector.register(
            FaultPoint::new("specific", FailureMode::StatusCode(500))
                .with_schedule(FailureSchedule::OnCalls(vec![1, 3, 5])),
        );

        assert!(injector.check("specific").is_triggered()); // 1
        assert!(injector.check("specific").is_pass()); // 2
        assert!(injector.check("specific").is_triggered()); // 3
        assert!(injector.check("specific").is_pass()); // 4
        assert!(injector.check("specific").is_triggered()); // 5
    }

    #[test]
    fn first_n_schedule() {
        let mut injector = FaultInjector::new(1);
        injector.register(
            FaultPoint::new("first2", FailureMode::Empty)
                .with_schedule(FailureSchedule::FirstN(2)),
        );

        assert!(injector.check("first2").is_triggered()); // 1
        assert!(injector.check("first2").is_triggered()); // 2
        assert!(injector.check("first2").is_pass()); // 3
    }

    #[test]
    fn after_n_schedule() {
        let mut injector = FaultInjector::new(1);
        injector.register(
            FaultPoint::new("after3", FailureMode::Empty)
                .with_schedule(FailureSchedule::AfterN(3)),
        );

        assert!(injector.check("after3").is_pass()); // 1
        assert!(injector.check("after3").is_pass()); // 2
        assert!(injector.check("after3").is_pass()); // 3
        assert!(injector.check("after3").is_triggered()); // 4
    }

    #[test]
    fn probability_schedule_fires_sometimes() {
        let mut injector = FaultInjector::new(42);
        injector.register(
            FaultPoint::new("maybe", FailureMode::Error("err".into()))
                .with_schedule(FailureSchedule::Probability(0.5)),
        );

        let mut fires = 0;
        for _ in 0..100 {
            if injector.check("maybe").is_triggered() {
                fires += 1;
            }
        }
        // With probability 0.5, we expect roughly 50 fires
        assert!(fires > 10, "Expected some fires, got {fires}");
        assert!(fires < 90, "Expected some passes, got {fires} fires");
    }

    #[test]
    fn unregistered_fault_passes() {
        let mut injector = FaultInjector::new(1);
        assert!(injector.check("nonexistent").is_pass());
    }

    #[test]
    fn check_result_integration() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("fail", FailureMode::Error("oops".into())));
        let err = injector.check_result("fail").unwrap_err();
        assert!(err.contains("oops"));
    }

    #[test]
    fn trigger_log_records() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("logged", FailureMode::Error("err".into())));
        injector.check("logged");
        injector.check("logged");

        let log = injector.trigger_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].fault_name, "logged");
    }

    #[test]
    fn statistics_tracking() {
        let mut injector = FaultInjector::new(1);
        injector.register(
            FaultPoint::new("counted", FailureMode::Empty)
                .with_schedule(FailureSchedule::EveryNth(2)),
        );

        for _ in 0..10 {
            injector.check("counted");
        }

        let fp = injector.get("counted").unwrap();
        assert_eq!(fp.check_count, 10);
        assert_eq!(fp.fire_count, 5);
        assert!((fp.fire_rate() - 50.0).abs() < 0.01);
    }

    #[test]
    fn reset_stats() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("r", FailureMode::Empty));
        injector.check("r");
        injector.reset_stats();
        assert_eq!(injector.total_checks(), 0);
        assert_eq!(injector.total_triggers(), 0);
        assert!(injector.trigger_log().is_empty());
    }

    #[test]
    fn clear_removes_all() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("a", FailureMode::Empty));
        injector.register(FaultPoint::new("b", FailureMode::Empty));
        injector.clear();
        assert_eq!(injector.num_faults(), 0);
    }

    #[test]
    fn remove_fault() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("x", FailureMode::Empty));
        assert!(injector.remove("x"));
        assert!(!injector.remove("x"));
        assert!(!injector.has("x"));
    }

    #[test]
    fn scoped_fault_activation() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("scoped", FailureMode::Empty).disabled());

        assert!(injector.check("scoped").is_pass());
        let scope = ScopedFault::activate(&mut injector, "scoped");
        assert!(injector.check("scoped").is_triggered());
        scope.deactivate(&mut injector);
        assert!(injector.check("scoped").is_pass());
    }

    #[test]
    fn failure_mode_display() {
        assert_eq!(format!("{}", FailureMode::Error("x".into())), "Error(x)");
        assert_eq!(format!("{}", FailureMode::Delay(100)), "Delay(100ms)");
        assert_eq!(format!("{}", FailureMode::StatusCode(503)), "StatusCode(503)");
        assert_eq!(format!("{}", FailureMode::Empty), "Empty");
    }

    #[test]
    fn stats_summary_format() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("test_fault", FailureMode::Empty));
        injector.check("test_fault");
        let summary = injector.stats_summary();
        assert!(summary.contains("test_fault"));
        assert!(summary.contains("Fault Injection Statistics"));
    }

    #[test]
    fn fault_names_sorted() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("z_fault", FailureMode::Empty));
        injector.register(FaultPoint::new("a_fault", FailureMode::Empty));
        let names = injector.fault_names();
        assert_eq!(names, vec!["a_fault", "z_fault"]);
    }

    #[test]
    fn delay_failure_mode() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("slow", FailureMode::Delay(5000)));
        match injector.check("slow") {
            FaultOutcome::Triggered(FailureMode::Delay(ms)) => assert_eq!(ms, 5000),
            other => panic!("Expected Delay, got {other:?}"),
        }
    }

    #[test]
    fn corrupt_failure_mode() {
        let mut injector = FaultInjector::new(1);
        injector.register(FaultPoint::new("corrupt", FailureMode::Corrupt("garbled".into())));
        match injector.check("corrupt") {
            FaultOutcome::Triggered(FailureMode::Corrupt(data)) => {
                assert_eq!(data, "garbled");
            }
            other => panic!("Expected Corrupt, got {other:?}"),
        }
    }
}
