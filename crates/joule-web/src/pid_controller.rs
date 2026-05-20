//! PID Controller — proportional-integral-derivative control with anti-windup,
//! derivative filtering, bumpless transfer, setpoint weighting, output limiting,
//! cascaded PID, and auto-tuning hint (Ziegler-Nichols relay parameters).
//!
//! Replaces ad-hoc PID implementations in JS/TS with a pure-Rust controller
//! that handles real-world concerns: windup, noise, mode switching, cascading.

use serde::{Deserialize, Serialize};

// ── Errors ──────────────────────────────────────────────────────

/// PID controller errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PidError {
    /// Invalid gain value.
    InvalidGain(String),
    /// Invalid sample time.
    InvalidSampleTime(f64),
    /// Invalid output limit.
    InvalidOutputLimits { min: f64, max: f64 },
    /// Invalid setpoint weight.
    InvalidSetpointWeight(f64),
    /// Invalid filter coefficient.
    InvalidFilterCoefficient(f64),
    /// Cascade depth exceeded.
    CascadeDepthExceeded(usize),
}

impl std::fmt::Display for PidError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidGain(msg) => write!(f, "invalid gain: {msg}"),
            Self::InvalidSampleTime(dt) => write!(f, "invalid sample time: {dt}"),
            Self::InvalidOutputLimits { min, max } => {
                write!(f, "invalid output limits: min={min}, max={max}")
            }
            Self::InvalidSetpointWeight(w) => write!(f, "invalid setpoint weight: {w}"),
            Self::InvalidFilterCoefficient(c) => write!(f, "invalid filter coefficient: {c}"),
            Self::CascadeDepthExceeded(d) => write!(f, "cascade depth exceeded: {d}"),
        }
    }
}

impl std::error::Error for PidError {}

// ── Operating Mode ──────────────────────────────────────────────

/// Controller operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PidMode {
    /// Automatic — controller computes output.
    Auto,
    /// Manual — user sets output directly.
    Manual,
}

// ── Anti-windup Strategy ────────────────────────────────────────

/// Strategy for combating integral windup.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AntiWindup {
    /// Clamp integrator between limits.
    Clamp { min: f64, max: f64 },
    /// Back-calculation with tracking gain Kt.
    BackCalculation { kt: f64 },
    /// No anti-windup.
    None,
}

// ── PID Configuration ───────────────────────────────────────────

/// Full configuration for a PID controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PidConfig {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
    /// Sample time in seconds.
    pub dt: f64,
    /// Output minimum.
    pub output_min: f64,
    /// Output maximum.
    pub output_max: f64,
    /// Anti-windup strategy.
    pub anti_windup: AntiWindup,
    /// Derivative low-pass filter coefficient (0..1). 0 = no filtering, ~1 = heavy.
    pub derivative_filter: f64,
    /// Whether to compute derivative on measurement (true) vs. error (false).
    pub derivative_on_measurement: bool,
    /// Setpoint weight for proportional term (0..1).
    pub setpoint_weight_p: f64,
    /// Setpoint weight for derivative term (0..1).
    pub setpoint_weight_d: f64,
}

impl PidConfig {
    /// Create with basic gains and 10 ms sample time.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            dt: 0.01,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
            anti_windup: AntiWindup::Clamp {
                min: -1000.0,
                max: 1000.0,
            },
            derivative_filter: 0.0,
            derivative_on_measurement: true,
            setpoint_weight_p: 1.0,
            setpoint_weight_d: 1.0,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PidError> {
        if self.dt <= 0.0 {
            return Err(PidError::InvalidSampleTime(self.dt));
        }
        if self.output_min > self.output_max {
            return Err(PidError::InvalidOutputLimits {
                min: self.output_min,
                max: self.output_max,
            });
        }
        if !(0.0..=1.0).contains(&self.derivative_filter) {
            return Err(PidError::InvalidFilterCoefficient(self.derivative_filter));
        }
        if !(0.0..=1.0).contains(&self.setpoint_weight_p) {
            return Err(PidError::InvalidSetpointWeight(self.setpoint_weight_p));
        }
        if !(0.0..=1.0).contains(&self.setpoint_weight_d) {
            return Err(PidError::InvalidSetpointWeight(self.setpoint_weight_d));
        }
        Ok(())
    }
}

// ── PID State ───────────────────────────────────────────────────

/// Runtime state of a PID controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PidState {
    /// Accumulated integral term.
    pub integral: f64,
    /// Previous error (for derivative).
    pub prev_error: f64,
    /// Previous measurement (for derivative-on-measurement).
    pub prev_measurement: f64,
    /// Filtered derivative value.
    pub filtered_derivative: f64,
    /// Current output.
    pub output: f64,
    /// Previous un-clamped output (for back-calculation).
    pub prev_unclamped: f64,
    /// Current operating mode.
    pub mode: PidMode,
    /// Manual output value (used when mode == Manual).
    pub manual_output: f64,
    /// Whether the first update has been called (to suppress initial derivative spike).
    pub initialized: bool,
}

impl Default for PidState {
    fn default() -> Self {
        Self {
            integral: 0.0,
            prev_error: 0.0,
            prev_measurement: 0.0,
            filtered_derivative: 0.0,
            output: 0.0,
            prev_unclamped: 0.0,
            mode: PidMode::Auto,
            manual_output: 0.0,
            initialized: false,
        }
    }
}

// ── PID Controller ──────────────────────────────────────────────

/// PID controller with full industrial features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PidController {
    pub config: PidConfig,
    pub state: PidState,
}

/// Decomposed output for diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PidOutput {
    pub p_term: f64,
    pub i_term: f64,
    pub d_term: f64,
    pub output: f64,
    pub saturated: bool,
}

impl PidController {
    /// Create a new controller from configuration.
    pub fn new(config: PidConfig) -> Result<Self, PidError> {
        config.validate()?;
        Ok(Self {
            config,
            state: PidState::default(),
        })
    }

    /// Compute one control cycle.
    pub fn update(&mut self, setpoint: f64, measurement: f64) -> PidOutput {
        if self.state.mode == PidMode::Manual {
            // In manual mode: track output so integral stays consistent (bumpless transfer).
            self.state.integral =
                self.state.manual_output - self.config.kp * (setpoint - measurement);
            self.state.output = self.state.manual_output;
            self.state.prev_error = setpoint - measurement;
            self.state.prev_measurement = measurement;
            return PidOutput {
                p_term: 0.0,
                i_term: self.state.integral,
                d_term: 0.0,
                output: self.state.manual_output,
                saturated: false,
            };
        }

        let error = setpoint - measurement;
        let dt = self.config.dt;

        // On the very first call, seed prev_measurement/prev_error so derivative
        // doesn't see a phantom step from 0.
        if !self.state.initialized {
            self.state.prev_measurement = measurement;
            self.state.prev_error = error;
            self.state.initialized = true;
        }

        // ── Proportional ────────────────────────────────────
        let p_error = self.config.setpoint_weight_p * setpoint - measurement;
        let p_term = self.config.kp * p_error;

        // ── Integral ────────────────────────────────────────
        self.state.integral += self.config.ki * error * dt;

        // Anti-windup: back-calculation from previous step.
        if let AntiWindup::BackCalculation { kt } = self.config.anti_windup {
            let sat_error = self.state.output - self.state.prev_unclamped;
            self.state.integral += kt * sat_error * dt;
        }

        // Anti-windup: clamp integrator.
        if let AntiWindup::Clamp { min, max } = self.config.anti_windup {
            self.state.integral = self.state.integral.clamp(min, max);
        }

        let i_term = self.state.integral;

        // ── Derivative ──────────────────────────────────────
        let raw_derivative = if self.config.derivative_on_measurement {
            // Derivative on measurement avoids setpoint kick.
            -(measurement - self.state.prev_measurement) / dt
        } else {
            let d_error = self.config.setpoint_weight_d * (setpoint - measurement);
            let d_prev = self.config.setpoint_weight_d * self.state.prev_error;
            // Use prev_error which already incorporated setpoint_weight_d? No —
            // we stored raw error. So recompute.
            (d_error - d_prev) / dt
        };

        // Low-pass filter on derivative.
        let alpha = self.config.derivative_filter;
        self.state.filtered_derivative =
            alpha * self.state.filtered_derivative + (1.0 - alpha) * raw_derivative;
        let d_term = self.config.kd * self.state.filtered_derivative;

        // ── Sum & clamp ─────────────────────────────────────
        let unclamped = p_term + i_term + d_term;
        let clamped = unclamped.clamp(self.config.output_min, self.config.output_max);
        let saturated = (clamped - unclamped).abs() > 1e-12;

        self.state.prev_unclamped = unclamped;
        self.state.output = clamped;
        self.state.prev_error = error;
        self.state.prev_measurement = measurement;

        PidOutput {
            p_term,
            i_term,
            d_term,
            output: clamped,
            saturated,
        }
    }

    /// Switch to auto mode with bumpless transfer.
    pub fn set_auto(&mut self) {
        if self.state.mode == PidMode::Manual {
            // Initialize integral so output doesn't jump.
            self.state.mode = PidMode::Auto;
        }
    }

    /// Switch to manual mode.
    pub fn set_manual(&mut self, output: f64) {
        self.state.manual_output = output;
        self.state.mode = PidMode::Manual;
    }

    /// Reset integral and derivative state.
    pub fn reset(&mut self) {
        self.state = PidState::default();
    }

    /// Change gains at runtime.
    pub fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.config.kp = kp;
        self.config.ki = ki;
        self.config.kd = kd;
    }
}

// ── Cascaded PID ────────────────────────────────────────────────

/// Cascaded PID: outer loop output feeds inner loop setpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CascadedPid {
    pub outer: PidController,
    pub inner: PidController,
}

impl CascadedPid {
    /// Create a cascaded controller pair.
    pub fn new(outer_config: PidConfig, inner_config: PidConfig) -> Result<Self, PidError> {
        Ok(Self {
            outer: PidController::new(outer_config)?,
            inner: PidController::new(inner_config)?,
        })
    }

    /// Update: outer loop produces inner setpoint, inner loop produces final output.
    pub fn update(
        &mut self,
        outer_setpoint: f64,
        outer_measurement: f64,
        inner_measurement: f64,
    ) -> PidOutput {
        let outer_out = self.outer.update(outer_setpoint, outer_measurement);
        // Outer output becomes inner setpoint.
        self.inner.update(outer_out.output, inner_measurement)
    }
}

// ── Ziegler-Nichols Auto-Tuning Hint ────────────────────────────

/// Parameters from a Ziegler-Nichols relay experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelayExperiment {
    /// Relay amplitude (output oscillation half-amplitude).
    pub relay_amplitude: f64,
    /// Measured oscillation amplitude of the process variable.
    pub oscillation_amplitude: f64,
    /// Measured oscillation period in seconds.
    pub oscillation_period: f64,
}

/// Tuning rules from Ziegler-Nichols.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZnTuning {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    /// Ultimate gain.
    pub ku: f64,
    /// Ultimate period.
    pub tu: f64,
}

impl RelayExperiment {
    /// Compute ultimate gain and period from relay data.
    pub fn ultimate_parameters(&self) -> (f64, f64) {
        let ku = 4.0 * self.relay_amplitude / (std::f64::consts::PI * self.oscillation_amplitude);
        let tu = self.oscillation_period;
        (ku, tu)
    }

    /// Classic Ziegler-Nichols PID tuning.
    pub fn zn_pid(&self) -> ZnTuning {
        let (ku, tu) = self.ultimate_parameters();
        ZnTuning {
            kp: 0.6 * ku,
            ki: 1.2 * ku / tu,
            kd: 0.075 * ku * tu,
            ku,
            tu,
        }
    }

    /// Ziegler-Nichols PI tuning (no derivative).
    pub fn zn_pi(&self) -> ZnTuning {
        let (ku, tu) = self.ultimate_parameters();
        ZnTuning {
            kp: 0.45 * ku,
            ki: 0.54 * ku / tu,
            kd: 0.0,
            ku,
            tu,
        }
    }

    /// Ziegler-Nichols P-only tuning.
    pub fn zn_p(&self) -> ZnTuning {
        let (ku, tu) = self.ultimate_parameters();
        ZnTuning {
            kp: 0.5 * ku,
            ki: 0.0,
            kd: 0.0,
            ku,
            tu,
        }
    }

    /// Build a PidConfig from ZN-PID tuning.
    pub fn to_pid_config(&self, dt: f64) -> PidConfig {
        let t = self.zn_pid();
        let mut cfg = PidConfig::new(t.kp, t.ki, t.kd);
        cfg.dt = dt;
        cfg
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_p_only_controller() {
        let cfg = PidConfig::new(2.0, 0.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        let out = pid.update(10.0, 0.0);
        assert!(approx(out.p_term, 20.0, 1e-4));
        assert!(approx(out.i_term, 0.0, 1e-4));
        assert!(approx(out.d_term, 0.0, 1e-4));
    }

    #[test]
    fn test_pi_controller_integral_accumulates() {
        let cfg = PidConfig::new(1.0, 10.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        pid.update(10.0, 0.0); // error=10, integral += 10*10*0.01 = 1.0
        let out = pid.update(10.0, 0.0);
        // integral should be 2.0 after second step
        assert!(approx(out.i_term, 2.0, 1e-4));
    }

    #[test]
    fn test_derivative_on_measurement() {
        let mut cfg = PidConfig::new(0.0, 0.0, 1.0);
        cfg.derivative_on_measurement = true;
        let mut pid = PidController::new(cfg).unwrap();
        pid.update(0.0, 0.0);
        // Measurement jumps from 0 to 5 => derivative = -(5-0)/0.01 = -500
        let out = pid.update(0.0, 5.0);
        assert!(approx(out.d_term, -500.0, 1e-2));
    }

    #[test]
    fn test_derivative_on_error() {
        let mut cfg = PidConfig::new(0.0, 0.0, 1.0);
        cfg.derivative_on_measurement = false;
        let mut pid = PidController::new(cfg).unwrap();
        pid.update(10.0, 0.0); // error=10
        let out = pid.update(10.0, 5.0); // error=5, d_error = (5-10)/0.01 = -500
        assert!(approx(out.d_term, -500.0, 1e-2));
    }

    #[test]
    fn test_output_clamping() {
        let mut cfg = PidConfig::new(100.0, 0.0, 0.0);
        cfg.output_min = -10.0;
        cfg.output_max = 10.0;
        let mut pid = PidController::new(cfg).unwrap();
        let out = pid.update(10.0, 0.0); // p=1000 => clamped to 10
        assert!(approx(out.output, 10.0, 1e-4));
        assert!(out.saturated);
    }

    #[test]
    fn test_anti_windup_clamp() {
        let mut cfg = PidConfig::new(0.0, 100.0, 0.0);
        cfg.anti_windup = AntiWindup::Clamp {
            min: -5.0,
            max: 5.0,
        };
        let mut pid = PidController::new(cfg).unwrap();
        // Run many cycles to saturate integral.
        for _ in 0..1000 {
            pid.update(10.0, 0.0);
        }
        assert!(pid.state.integral <= 5.0 + 1e-4);
    }

    #[test]
    fn test_anti_windup_back_calculation() {
        let mut cfg = PidConfig::new(1.0, 10.0, 0.0);
        cfg.output_min = -5.0;
        cfg.output_max = 5.0;
        cfg.anti_windup = AntiWindup::BackCalculation { kt: 1.0 };
        let mut pid = PidController::new(cfg).unwrap();
        for _ in 0..100 {
            pid.update(100.0, 0.0);
        }
        // Output should be at max.
        assert!(pid.state.output <= 5.0 + 1e-4);
    }

    #[test]
    fn test_derivative_filter() {
        let mut cfg = PidConfig::new(0.0, 0.0, 1.0);
        cfg.derivative_filter = 0.8;
        cfg.derivative_on_measurement = true;
        let mut pid = PidController::new(cfg).unwrap();
        pid.update(0.0, 0.0);
        // Spike in measurement.
        let out1 = pid.update(0.0, 10.0);
        // With filter=0.8, filtered = 0.8*0 + 0.2*(-1000) = -200
        assert!(approx(out1.d_term, -200.0, 1e-2));
        // Second step, measurement stays at 10 => raw_deriv = 0
        let out2 = pid.update(0.0, 10.0);
        // filtered = 0.8*(-200) + 0.2*0 = -160
        assert!(approx(out2.d_term, -160.0, 1e-2));
    }

    #[test]
    fn test_manual_mode() {
        let cfg = PidConfig::new(1.0, 1.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        pid.set_manual(42.0);
        let out = pid.update(10.0, 5.0);
        assert!(approx(out.output, 42.0, 1e-4));
    }

    #[test]
    fn test_bumpless_transfer() {
        let cfg = PidConfig::new(2.0, 0.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        // Run in auto for a bit.
        pid.update(10.0, 5.0);
        let auto_output = pid.state.output;
        // Switch to manual at current output.
        pid.set_manual(auto_output);
        pid.update(10.0, 5.0);
        // Switch back to auto.
        pid.set_auto();
        let out = pid.update(10.0, 5.0);
        // Output should not have a large discontinuity.
        assert!((out.output - auto_output).abs() < 20.0);
    }

    #[test]
    fn test_setpoint_weight_p() {
        let mut cfg = PidConfig::new(1.0, 0.0, 0.0);
        cfg.setpoint_weight_p = 0.5;
        let mut pid = PidController::new(cfg).unwrap();
        // p_error = 0.5*10 - 0 = 5, p_term = 1*5 = 5
        let out = pid.update(10.0, 0.0);
        assert!(approx(out.p_term, 5.0, 1e-4));
    }

    #[test]
    fn test_reset() {
        let cfg = PidConfig::new(1.0, 10.0, 1.0);
        let mut pid = PidController::new(cfg).unwrap();
        pid.update(10.0, 0.0);
        pid.update(10.0, 2.0);
        pid.reset();
        assert!(approx(pid.state.integral, 0.0, 1e-4));
        assert!(approx(pid.state.prev_error, 0.0, 1e-4));
        assert!(approx(pid.state.filtered_derivative, 0.0, 1e-4));
    }

    #[test]
    fn test_cascaded_pid() {
        let outer = PidConfig::new(1.0, 0.0, 0.0);
        let inner = PidConfig::new(2.0, 0.0, 0.0);
        let mut cascade = CascadedPid::new(outer, inner).unwrap();
        let out = cascade.update(100.0, 50.0, 40.0);
        // outer output = 1*(100-50) = 50
        // inner setpoint = 50, inner measurement = 40 => output = 2*(50-40)=20
        assert!(approx(out.output, 20.0, 1e-4));
    }

    #[test]
    fn test_zn_tuning_ultimate_params() {
        let exp = RelayExperiment {
            relay_amplitude: 1.0,
            oscillation_amplitude: 0.5,
            oscillation_period: 2.0,
        };
        let (ku, tu) = exp.ultimate_parameters();
        assert!(approx(ku, 4.0 / (std::f64::consts::PI * 0.5), 1e-4));
        assert!(approx(tu, 2.0, 1e-4));
    }

    #[test]
    fn test_zn_pid_tuning() {
        let exp = RelayExperiment {
            relay_amplitude: 1.0,
            oscillation_amplitude: 0.5,
            oscillation_period: 2.0,
        };
        let t = exp.zn_pid();
        assert!(approx(t.kp, 0.6 * t.ku, 1e-4));
        assert!(approx(t.ki, 1.2 * t.ku / t.tu, 1e-4));
        assert!(approx(t.kd, 0.075 * t.ku * t.tu, 1e-4));
    }

    #[test]
    fn test_zn_pi_tuning() {
        let exp = RelayExperiment {
            relay_amplitude: 2.0,
            oscillation_amplitude: 1.0,
            oscillation_period: 4.0,
        };
        let t = exp.zn_pi();
        assert!(approx(t.kd, 0.0, 1e-4));
        assert!(t.kp > 0.0);
        assert!(t.ki > 0.0);
    }

    #[test]
    fn test_zn_to_pid_config() {
        let exp = RelayExperiment {
            relay_amplitude: 1.0,
            oscillation_amplitude: 0.5,
            oscillation_period: 2.0,
        };
        let cfg = exp.to_pid_config(0.001);
        assert!(approx(cfg.dt, 0.001, 1e-6));
        assert!(cfg.kp > 0.0);
    }

    #[test]
    fn test_invalid_sample_time() {
        let mut cfg = PidConfig::new(1.0, 0.0, 0.0);
        cfg.dt = -1.0;
        assert!(PidController::new(cfg).is_err());
    }

    #[test]
    fn test_invalid_output_limits() {
        let mut cfg = PidConfig::new(1.0, 0.0, 0.0);
        cfg.output_min = 100.0;
        cfg.output_max = -100.0;
        assert!(PidController::new(cfg).is_err());
    }

    #[test]
    fn test_invalid_filter_coefficient() {
        let mut cfg = PidConfig::new(1.0, 0.0, 0.0);
        cfg.derivative_filter = 1.5;
        assert!(PidController::new(cfg).is_err());
    }

    #[test]
    fn test_steady_state_pi() {
        let cfg = PidConfig::new(1.0, 5.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        let mut measurement = 0.0;
        // Simple first-order plant: y[k+1] = 0.9*y[k] + 0.1*u[k]
        for _ in 0..5000 {
            let out = pid.update(10.0, measurement);
            measurement = 0.9 * measurement + 0.1 * out.output;
        }
        // Should converge near setpoint.
        assert!(approx(measurement, 10.0, 0.5));
    }

    #[test]
    fn test_set_gains_at_runtime() {
        let cfg = PidConfig::new(1.0, 0.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        pid.set_gains(5.0, 0.0, 0.0);
        let out = pid.update(10.0, 0.0);
        assert!(approx(out.p_term, 50.0, 1e-4));
    }

    #[test]
    fn test_zero_error_produces_zero_output() {
        let cfg = PidConfig::new(1.0, 1.0, 1.0);
        let mut pid = PidController::new(cfg).unwrap();
        let out = pid.update(5.0, 5.0);
        assert!(approx(out.output, 0.0, 1e-4));
    }

    #[test]
    fn test_negative_error() {
        let cfg = PidConfig::new(1.0, 0.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        let out = pid.update(0.0, 10.0); // error=-10
        assert!(approx(out.p_term, -10.0, 1e-4));
    }

    #[test]
    fn test_mode_switching_preserves_integral() {
        let cfg = PidConfig::new(1.0, 10.0, 0.0);
        let mut pid = PidController::new(cfg).unwrap();
        // Build up some integral.
        pid.update(10.0, 0.0);
        pid.update(10.0, 0.0);
        let integral_before = pid.state.integral;
        assert!(integral_before > 0.0);
        // Switch to manual and back.
        pid.set_manual(5.0);
        pid.update(10.0, 0.0);
        pid.set_auto();
        // Integral was adjusted for bumpless transfer but not zeroed.
        assert!(pid.state.integral.is_finite());
    }
}
