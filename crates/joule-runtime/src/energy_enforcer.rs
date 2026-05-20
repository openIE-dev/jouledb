//! Energy Budget Hard Kill — sidecar that samples RAPL and SIGKILL on exceeded budget.
//!
//! The `EnergyEnforcer` runs inside or alongside a sandbox instance, sampling
//! hardware energy counters (RAPL, NVML, estimation) at a configurable interval.
//! When cumulative energy exceeds the budget, the workload is immediately killed.
//!
//! This is the hard enforcement counterpart to `inv-energy::BudgetEnforcer`,
//! which only reports. This module acts.
//!
//! Design: the enforcer is the last line of defense. If the workload exceeds
//! its energy budget, it gets SIGKILL. No negotiation. Thermodynamics wins.

use inv_energy::meter::{EnergyMeter, EnergyMeterError};
use inv_energy::receipt::EnergyReceipt;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the energy enforcer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyEnforcerConfig {
    /// Maximum energy budget in microjoules.
    pub max_energy_uj: u64,
    /// Sampling interval for hardware energy counters.
    #[serde(with = "duration_millis")]
    pub sample_interval: Duration,
    /// Warning threshold (0.0-1.0) — emit warning when this fraction is consumed.
    pub warning_threshold: f64,
    /// Whether to actually kill the process on budget exceeded (vs. just log).
    pub hard_kill: bool,
    /// PID of the process to kill on budget exceeded.
    pub target_pid: Option<u32>,
}

impl Default for EnergyEnforcerConfig {
    fn default() -> Self {
        Self {
            max_energy_uj: 50_000_000, // 50 joules default
            sample_interval: Duration::from_millis(100),
            warning_threshold: 0.8,
            hard_kill: true,
            target_pid: None,
        }
    }
}

/// Shared state between the enforcer thread and callers.
#[derive(Debug)]
pub struct EnergyEnforcerState {
    /// Cumulative energy consumed in microjoules.
    consumed_uj: AtomicU64,
    /// Budget in microjoules.
    budget_uj: AtomicU64,
    /// Whether the budget has been exceeded.
    exceeded: AtomicBool,
    /// Whether the enforcer is running.
    running: AtomicBool,
    /// Whether a warning has been emitted.
    warned: AtomicBool,
}

impl EnergyEnforcerState {
    fn new(budget_uj: u64) -> Self {
        Self {
            consumed_uj: AtomicU64::new(0),
            budget_uj: AtomicU64::new(budget_uj),
            exceeded: AtomicBool::new(false),
            running: AtomicBool::new(false),
            warned: AtomicBool::new(false),
        }
    }

    /// Current energy consumed in microjoules.
    pub fn consumed_uj(&self) -> u64 {
        self.consumed_uj.load(Ordering::Relaxed)
    }

    /// Current energy consumed in joules.
    pub fn consumed_joules(&self) -> f64 {
        self.consumed_uj() as f64 / 1_000_000.0
    }

    /// Budget in microjoules.
    pub fn budget_uj(&self) -> u64 {
        self.budget_uj.load(Ordering::Relaxed)
    }

    /// Update the budget at runtime (e.g., after a contract extension).
    pub fn update_budget(&self, new_budget_uj: u64) {
        self.budget_uj.store(new_budget_uj, Ordering::Relaxed);
    }

    /// Budget in joules.
    pub fn budget_joules(&self) -> f64 {
        self.budget_uj() as f64 / 1_000_000.0
    }

    /// Utilization (0.0 to 1.0+).
    pub fn utilization(&self) -> f64 {
        let budget = self.budget_uj();
        if budget == 0 {
            return 0.0;
        }
        self.consumed_uj() as f64 / budget as f64
    }

    /// Remaining energy in microjoules.
    pub fn remaining_uj(&self) -> u64 {
        let budget = self.budget_uj();
        let consumed = self.consumed_uj();
        budget.saturating_sub(consumed)
    }

    /// Whether the budget has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.exceeded.load(Ordering::Relaxed)
    }

    /// Whether the enforcer is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

/// Energy enforcer — samples hardware counters and kills on budget exceeded.
pub struct EnergyEnforcer {
    config: EnergyEnforcerConfig,
    state: Arc<EnergyEnforcerState>,
    /// Handle to the sampling thread (if running).
    thread_handle: Option<std::thread::JoinHandle<EnforcerResult>>,
}

/// Result of an enforcer run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnforcerResult {
    /// Total energy consumed in microjoules.
    pub consumed_uj: u64,
    /// Budget in microjoules.
    pub budget_uj: u64,
    /// Whether the budget was exceeded.
    pub exceeded: bool,
    /// Number of samples taken.
    pub sample_count: u64,
    /// Wall-clock duration of enforcement.
    pub duration_ms: u64,
    /// Reason for stopping.
    pub stop_reason: StopReason,
}

/// Why the enforcer stopped.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Normal stop (workload completed or explicit stop).
    Normal,
    /// Budget exceeded — workload was killed.
    BudgetExceeded,
    /// Meter error — could not read hardware counters.
    MeterError(String),
    /// Process already dead.
    ProcessDead,
}

impl EnergyEnforcer {
    /// Create a new enforcer with the given config.
    pub fn new(config: EnergyEnforcerConfig) -> Self {
        let state = Arc::new(EnergyEnforcerState::new(config.max_energy_uj));
        Self {
            config,
            state,
            thread_handle: None,
        }
    }

    /// Get shared state for external monitoring.
    pub fn state(&self) -> Arc<EnergyEnforcerState> {
        Arc::clone(&self.state)
    }

    /// Start the enforcer sampling loop in a background thread.
    ///
    /// The meter is moved into the thread and sampled at the configured interval.
    /// If energy exceeds the budget and `hard_kill` is true, the target PID is killed.
    pub fn start(&mut self, meter: Box<dyn EnergyMeter + Send>) -> Result<(), EnergyMeterError> {
        if self.state.is_running() {
            return Ok(()); // Already running
        }

        // Take baseline reading
        let baseline = meter.read()?;
        let baseline_joules = baseline.joules.as_f64();

        let config = self.config.clone();
        let state = Arc::clone(&self.state);

        state.running.store(true, Ordering::SeqCst);

        let handle = std::thread::Builder::new()
            .name("energy-enforcer".to_string())
            .spawn(move || {
                let start = Instant::now();
                let mut sample_count: u64 = 0;

                loop {
                    if !state.running.load(Ordering::Relaxed) {
                        return EnforcerResult {
                            consumed_uj: state.consumed_uj(),
                            budget_uj: config.max_energy_uj,
                            exceeded: false,
                            sample_count,
                            duration_ms: start.elapsed().as_millis() as u64,
                            stop_reason: StopReason::Normal,
                        };
                    }

                    std::thread::sleep(config.sample_interval);
                    sample_count += 1;

                    // Check if target process is still alive
                    if let Some(pid) = config.target_pid {
                        if !is_process_alive(pid) {
                            state.running.store(false, Ordering::SeqCst);
                            return EnforcerResult {
                                consumed_uj: state.consumed_uj(),
                                budget_uj: config.max_energy_uj,
                                exceeded: false,
                                sample_count,
                                duration_ms: start.elapsed().as_millis() as u64,
                                stop_reason: StopReason::ProcessDead,
                            };
                        }
                    }

                    // Read current energy
                    let reading = match meter.read() {
                        Ok(r) => r,
                        Err(e) => {
                            log::error!("Energy enforcer: meter error: {}", e);
                            state.running.store(false, Ordering::SeqCst);
                            return EnforcerResult {
                                consumed_uj: state.consumed_uj(),
                                budget_uj: config.max_energy_uj,
                                exceeded: false,
                                sample_count,
                                duration_ms: start.elapsed().as_millis() as u64,
                                stop_reason: StopReason::MeterError(e.to_string()),
                            };
                        }
                    };

                    let consumed_joules = reading.joules.as_f64() - baseline_joules;
                    let consumed_uj = (consumed_joules * 1_000_000.0).max(0.0) as u64;
                    state.consumed_uj.store(consumed_uj, Ordering::Relaxed);

                    // Check warning threshold
                    let utilization = consumed_uj as f64 / config.max_energy_uj as f64;
                    if utilization >= config.warning_threshold
                        && !state.warned.load(Ordering::Relaxed)
                    {
                        log::warn!(
                            "Energy enforcer: {:.1}% of budget consumed ({} / {} µJ)",
                            utilization * 100.0,
                            consumed_uj,
                            config.max_energy_uj
                        );
                        state.warned.store(true, Ordering::Relaxed);
                    }

                    // Check budget exceeded
                    if consumed_uj >= config.max_energy_uj {
                        state.exceeded.store(true, Ordering::SeqCst);
                        state.running.store(false, Ordering::SeqCst);

                        log::error!(
                            "Energy enforcer: BUDGET EXCEEDED ({} µJ >= {} µJ)",
                            consumed_uj,
                            config.max_energy_uj
                        );

                        if config.hard_kill {
                            if let Some(pid) = config.target_pid {
                                kill_process(pid);
                                log::error!(
                                    "Energy enforcer: SIGKILL sent to PID {}",
                                    pid
                                );
                            }
                        }

                        return EnforcerResult {
                            consumed_uj,
                            budget_uj: config.max_energy_uj,
                            exceeded: true,
                            sample_count,
                            duration_ms: start.elapsed().as_millis() as u64,
                            stop_reason: StopReason::BudgetExceeded,
                        };
                    }
                }
            })
            .map_err(|e| EnergyMeterError::ReadFailed(format!("failed to spawn enforcer thread: {}", e)))?;

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Stop the enforcer and return the result.
    pub fn stop(self) -> Option<EnforcerResult> {
        self.state.running.store(false, Ordering::SeqCst);
        self.thread_handle.and_then(|h| h.join().ok())
    }

    /// Generate an energy receipt from the enforcer's accumulated data.
    pub fn receipt(&self, node_id: &str, region: &str) -> EnergyReceipt {
        let joules = self.state.consumed_joules();
        EnergyReceipt::new(
            joules,
            joules * 0.000_233, // global average CO2
            inv_energy::receipt::SiliconType::Cpu,
            inv_energy::receipt::MemoryTier::Dram,
            inv_energy::receipt::MeasurementSource::Kernel,
            node_id.to_string(),
            region.to_string(),
        )
    }

    /// Update the budget at runtime (e.g., after a top-up).
    pub fn update_budget(&self, new_budget_uj: u64) {
        self.state.budget_uj.store(new_budget_uj, Ordering::Relaxed);
    }
}

/// Check if a process is alive.
fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // kill(pid, 0) checks if process exists without sending a signal
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        true // Assume alive on non-Unix
    }
}

/// Kill a process with SIGKILL.
fn kill_process(pid: u32) {
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        log::error!("Energy enforcer: SIGKILL not supported on this platform");
    }
}

/// Serde helper for Duration as milliseconds.
mod duration_millis {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(duration: &Duration, ser: S) -> Result<S::Ok, S::Error> {
        (duration.as_millis() as u64).serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(de: D) -> Result<Duration, D::Error> {
        let ms = u64::deserialize(de)?;
        Ok(Duration::from_millis(ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enforcer_config_default() {
        let config = EnergyEnforcerConfig::default();
        assert_eq!(config.max_energy_uj, 50_000_000);
        assert_eq!(config.sample_interval, Duration::from_millis(100));
        assert!((config.warning_threshold - 0.8).abs() < 1e-10);
        assert!(config.hard_kill);
        assert!(config.target_pid.is_none());
    }

    #[test]
    fn test_enforcer_state_new() {
        let state = EnergyEnforcerState::new(100_000_000);
        assert_eq!(state.consumed_uj(), 0);
        assert_eq!(state.budget_uj(), 100_000_000);
        assert!(!state.is_exceeded());
        assert!(!state.is_running());
        assert!((state.utilization()).abs() < 1e-10);
        assert_eq!(state.remaining_uj(), 100_000_000);
    }

    #[test]
    fn test_enforcer_state_utilization() {
        let state = EnergyEnforcerState::new(100);
        state.consumed_uj.store(50, Ordering::Relaxed);
        assert!((state.utilization() - 0.5).abs() < 1e-10);
        assert_eq!(state.remaining_uj(), 50);
    }

    #[test]
    fn test_enforcer_state_zero_budget() {
        let state = EnergyEnforcerState::new(0);
        assert!((state.utilization()).abs() < 1e-10);
    }

    #[test]
    fn test_enforcer_config_serde() {
        let config = EnergyEnforcerConfig {
            max_energy_uj: 1_000_000,
            sample_interval: Duration::from_millis(50),
            warning_threshold: 0.9,
            hard_kill: false,
            target_pid: Some(12345),
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: EnergyEnforcerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_energy_uj, 1_000_000);
        assert_eq!(parsed.sample_interval, Duration::from_millis(50));
        assert!(!parsed.hard_kill);
        assert_eq!(parsed.target_pid, Some(12345));
    }

    #[test]
    fn test_enforcer_creation() {
        let config = EnergyEnforcerConfig::default();
        let enforcer = EnergyEnforcer::new(config);
        let state = enforcer.state();
        assert!(!state.is_running());
        assert!(!state.is_exceeded());
    }

    #[test]
    fn test_enforcer_receipt() {
        let config = EnergyEnforcerConfig::default();
        let enforcer = EnergyEnforcer::new(config);
        let receipt = enforcer.receipt("node-1", "us-east");
        assert!((receipt.energy_joules).abs() < 1e-10);
        assert!(receipt.verify());
    }

    #[test]
    fn test_enforcer_update_budget() {
        let config = EnergyEnforcerConfig::default();
        let enforcer = EnergyEnforcer::new(config);
        enforcer.update_budget(200_000_000);
        assert_eq!(enforcer.state().budget_uj(), 200_000_000);
    }

    #[test]
    fn test_enforcer_result_serde() {
        let result = EnforcerResult {
            consumed_uj: 42_000_000,
            budget_uj: 50_000_000,
            exceeded: false,
            sample_count: 420,
            duration_ms: 42000,
            stop_reason: StopReason::Normal,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: EnforcerResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.consumed_uj, 42_000_000);
        assert!(!parsed.exceeded);
    }

    #[test]
    fn test_stop_reason_variants_serde() {
        let reasons = vec![
            StopReason::Normal,
            StopReason::BudgetExceeded,
            StopReason::MeterError("test".into()),
            StopReason::ProcessDead,
        ];
        for reason in reasons {
            let json = serde_json::to_string(&reason).unwrap();
            let _parsed: StopReason = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_enforcer_start_and_stop() {
        let config = EnergyEnforcerConfig {
            max_energy_uj: 1_000_000_000, // 1000 joules — won't exceed
            sample_interval: Duration::from_millis(10),
            warning_threshold: 0.8,
            hard_kill: false,
            target_pid: None,
        };
        let mut enforcer = EnergyEnforcer::new(config);
        let meter = inv_energy::meter::EstimationMeter::new(15.0);

        enforcer.start(Box::new(meter)).unwrap();
        assert!(enforcer.state().is_running());

        std::thread::sleep(Duration::from_millis(50));

        let result = enforcer.stop().unwrap();
        assert!(!result.exceeded);
        assert!(result.sample_count > 0);
        assert!(matches!(result.stop_reason, StopReason::Normal));
    }
}
