//! Motor Control — DC motor modeling, PWM duty-cycle mapping, closed-loop
//! speed and position control (PID), current limiting, and back-EMF
//! estimation for brushed DC motors.
//!
//! Pure-Rust motor control suitable for embedded and simulation workloads.
//! All math uses `f64`; no external crates.

use std::collections::VecDeque;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Motor control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum MotorError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Current limit exceeded.
    CurrentLimitExceeded { measured: f64, limit: f64 },
    /// Thermal shutdown.
    ThermalShutdown { temperature_c: f64 },
}

impl fmt::Display for MotorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::CurrentLimitExceeded { measured, limit } => {
                write!(f, "current {measured:.3} A exceeds limit {limit:.3} A")
            }
            Self::ThermalShutdown { temperature_c } => {
                write!(f, "thermal shutdown at {temperature_c:.1} °C")
            }
        }
    }
}

impl std::error::Error for MotorError {}

// ── DC Motor Model ─────────────────────────────────────────────

/// Simplified brushed DC motor electrical + mechanical model.
///
/// Electrical: V = R*i + L*(di/dt) + Ke*omega
/// Mechanical: J*(d_omega/dt) = Kt*i - B*omega - T_load
#[derive(Debug, Clone, PartialEq)]
pub struct DcMotorModel {
    /// Armature resistance (ohms).
    pub resistance: f64,
    /// Armature inductance (henrys).
    pub inductance: f64,
    /// Back-EMF constant (V·s/rad).
    pub ke: f64,
    /// Torque constant (N·m/A).
    pub kt: f64,
    /// Rotor inertia (kg·m²).
    pub inertia: f64,
    /// Viscous friction coefficient (N·m·s/rad).
    pub friction: f64,
    /// Current state: armature current (A).
    pub current: f64,
    /// Current state: angular velocity (rad/s).
    pub omega: f64,
    /// Current state: angular position (rad).
    pub theta: f64,
}

impl DcMotorModel {
    /// Create a new DC motor model with typical parameters.
    pub fn new(
        resistance: f64,
        inductance: f64,
        ke: f64,
        kt: f64,
        inertia: f64,
        friction: f64,
    ) -> Result<Self, MotorError> {
        if resistance <= 0.0 {
            return Err(MotorError::InvalidParameter("resistance must be > 0".into()));
        }
        if inductance < 0.0 {
            return Err(MotorError::InvalidParameter("inductance must be >= 0".into()));
        }
        if inertia <= 0.0 {
            return Err(MotorError::InvalidParameter("inertia must be > 0".into()));
        }
        Ok(Self {
            resistance,
            inductance,
            ke,
            kt,
            inertia,
            friction,
            current: 0.0,
            omega: 0.0,
            theta: 0.0,
        })
    }

    /// Typical small hobby motor (RS-775 class).
    pub fn hobby_motor() -> Self {
        Self {
            resistance: 0.5,
            inductance: 0.001,
            ke: 0.01,
            kt: 0.01,
            inertia: 5e-5,
            friction: 1e-4,
            current: 0.0,
            omega: 0.0,
            theta: 0.0,
        }
    }

    /// Step the motor model forward by `dt` seconds under applied voltage.
    ///
    /// Uses forward-Euler integration of the electrical and mechanical ODEs.
    pub fn step(&mut self, voltage: f64, load_torque: f64, dt: f64) {
        // Electrical: di/dt = (V - R*i - Ke*omega) / L
        let back_emf = self.ke * self.omega;
        if self.inductance > 1e-12 {
            let di_dt = (voltage - self.resistance * self.current - back_emf) / self.inductance;
            self.current += di_dt * dt;
        } else {
            // Quasi-static electrical (L ≈ 0).
            self.current = (voltage - back_emf) / self.resistance;
        }

        // Mechanical: d_omega/dt = (Kt*i - B*omega - T_load) / J
        let torque = self.kt * self.current - self.friction * self.omega - load_torque;
        let d_omega_dt = torque / self.inertia;
        self.omega += d_omega_dt * dt;
        self.theta += self.omega * dt;
    }

    /// Compute the back-EMF at current speed.
    pub fn back_emf(&self) -> f64 {
        self.ke * self.omega
    }

    /// Electrical power consumed (V * i approximation via R*i² + back-emf*i).
    pub fn electrical_power(&self) -> f64 {
        self.resistance * self.current * self.current + self.back_emf() * self.current
    }

    /// Mechanical power output (torque * omega).
    pub fn mechanical_power(&self) -> f64 {
        self.kt * self.current * self.omega
    }

    /// Efficiency (mechanical / electrical), clamped to [0, 1].
    pub fn efficiency(&self) -> f64 {
        let p_elec = self.electrical_power();
        if p_elec.abs() < 1e-12 {
            return 0.0;
        }
        (self.mechanical_power() / p_elec).clamp(0.0, 1.0)
    }

    /// Speed in RPM.
    pub fn speed_rpm(&self) -> f64 {
        self.omega * 60.0 / (2.0 * std::f64::consts::PI)
    }

    /// Reset state to rest.
    pub fn reset(&mut self) {
        self.current = 0.0;
        self.omega = 0.0;
        self.theta = 0.0;
    }
}

impl fmt::Display for DcMotorModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DcMotor(R={:.3}Ω, Ke={:.4}, ω={:.2} rad/s, i={:.3} A)",
            self.resistance, self.ke, self.omega, self.current
        )
    }
}

// ── PWM Duty Cycle ─────────────────────────────────────────────

/// PWM duty-cycle mapping with voltage clamping.
#[derive(Debug, Clone, PartialEq)]
pub struct PwmDutyCycle {
    /// Supply voltage (V).
    pub supply_voltage: f64,
    /// Current duty cycle [0.0, 1.0].
    pub duty: f64,
    /// Minimum duty (prevents motor stall issues).
    pub min_duty: f64,
    /// Maximum duty.
    pub max_duty: f64,
    /// PWM frequency (Hz).
    pub frequency: f64,
}

impl PwmDutyCycle {
    /// Create a new PWM driver.
    pub fn new(supply_voltage: f64, frequency: f64) -> Result<Self, MotorError> {
        if supply_voltage <= 0.0 {
            return Err(MotorError::InvalidParameter(
                "supply voltage must be > 0".into(),
            ));
        }
        if frequency <= 0.0 {
            return Err(MotorError::InvalidParameter("frequency must be > 0".into()));
        }
        Ok(Self {
            supply_voltage,
            duty: 0.0,
            min_duty: 0.0,
            max_duty: 1.0,
            frequency,
        })
    }

    /// Builder: set minimum duty.
    pub fn with_min_duty(mut self, min: f64) -> Self {
        self.min_duty = min.clamp(0.0, 1.0);
        self
    }

    /// Builder: set maximum duty.
    pub fn with_max_duty(mut self, max: f64) -> Self {
        self.max_duty = max.clamp(0.0, 1.0);
        self
    }

    /// Set duty cycle, clamped to [min_duty, max_duty].
    pub fn set_duty(&mut self, duty: f64) {
        self.duty = duty.clamp(self.min_duty, self.max_duty);
    }

    /// Effective voltage applied to motor.
    pub fn effective_voltage(&self) -> f64 {
        self.supply_voltage * self.duty
    }

    /// PWM period in seconds.
    pub fn period_s(&self) -> f64 {
        1.0 / self.frequency
    }

    /// On-time in seconds.
    pub fn on_time_s(&self) -> f64 {
        self.period_s() * self.duty
    }
}

impl fmt::Display for PwmDutyCycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PWM(duty={:.1}%, Veff={:.2}V, f={:.0}Hz)",
            self.duty * 100.0,
            self.effective_voltage(),
            self.frequency
        )
    }
}

// ── PID Controller ─────────────────────────────────────────────

/// Discrete PID controller with anti-windup, derivative filtering,
/// and output clamping.
#[derive(Debug, Clone, PartialEq)]
pub struct PidController {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
    /// Integral accumulator.
    integral: f64,
    /// Previous error (for derivative).
    prev_error: f64,
    /// Derivative low-pass filter coefficient [0, 1).
    pub derivative_filter: f64,
    /// Filtered derivative.
    filtered_derivative: f64,
    /// Output limits.
    pub output_min: f64,
    pub output_max: f64,
    /// Integral windup limit.
    pub integral_limit: f64,
}

impl PidController {
    /// Create a PID controller.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            derivative_filter: 0.0,
            filtered_derivative: 0.0,
            output_min: f64::NEG_INFINITY,
            output_max: f64::INFINITY,
            integral_limit: f64::INFINITY,
        }
    }

    /// Builder: set output limits.
    pub fn with_output_limits(mut self, min: f64, max: f64) -> Self {
        self.output_min = min;
        self.output_max = max;
        self
    }

    /// Builder: set integral windup limit.
    pub fn with_integral_limit(mut self, limit: f64) -> Self {
        self.integral_limit = limit.abs();
        self
    }

    /// Builder: set derivative low-pass filter coefficient.
    pub fn with_derivative_filter(mut self, alpha: f64) -> Self {
        self.derivative_filter = alpha.clamp(0.0, 0.999);
        self
    }

    /// Compute PID output for a given error and timestep.
    pub fn compute(&mut self, error: f64, dt: f64) -> f64 {
        if dt <= 0.0 {
            return 0.0;
        }

        // Proportional.
        let p = self.kp * error;

        // Integral with anti-windup clamping.
        self.integral += error * dt;
        self.integral = self.integral.clamp(-self.integral_limit, self.integral_limit);
        let i = self.ki * self.integral;

        // Derivative with optional low-pass filter.
        let raw_derivative = (error - self.prev_error) / dt;
        self.filtered_derivative = self.derivative_filter * self.filtered_derivative
            + (1.0 - self.derivative_filter) * raw_derivative;
        let d = self.kd * self.filtered_derivative;

        self.prev_error = error;

        (p + i + d).clamp(self.output_min, self.output_max)
    }

    /// Reset controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.filtered_derivative = 0.0;
    }
}

impl fmt::Display for PidController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PID(Kp={:.3}, Ki={:.3}, Kd={:.3})", self.kp, self.ki, self.kd)
    }
}

// ── Speed Controller ───────────────────────────────────────────

/// Closed-loop speed controller wrapping PID + PWM + motor model.
#[derive(Debug, Clone)]
pub struct SpeedController {
    /// PID controller for speed loop.
    pub pid: PidController,
    /// PWM driver.
    pub pwm: PwmDutyCycle,
    /// Target speed (rad/s).
    pub setpoint: f64,
    /// Speed measurement history for filtering.
    speed_history: VecDeque<f64>,
    /// Moving average window size.
    pub filter_window: usize,
}

impl SpeedController {
    /// Create a speed controller.
    pub fn new(pid: PidController, pwm: PwmDutyCycle) -> Self {
        Self {
            pid,
            pwm,
            setpoint: 0.0,
            speed_history: VecDeque::with_capacity(8),
            filter_window: 4,
        }
    }

    /// Builder: set filter window size.
    pub fn with_filter_window(mut self, window: usize) -> Self {
        self.filter_window = window.max(1);
        self
    }

    /// Set the desired speed in rad/s.
    pub fn set_speed(&mut self, omega_desired: f64) {
        self.setpoint = omega_desired;
    }

    /// Update the controller given measured speed, returning the duty cycle.
    pub fn update(&mut self, measured_omega: f64, dt: f64) -> f64 {
        // Moving-average filter on measured speed.
        self.speed_history.push_back(measured_omega);
        if self.speed_history.len() > self.filter_window {
            self.speed_history.pop_front();
        }
        let filtered: f64 =
            self.speed_history.iter().sum::<f64>() / self.speed_history.len() as f64;

        let error = self.setpoint - filtered;
        let output = self.pid.compute(error, dt);

        // Map PID output to duty cycle (normalize by supply voltage).
        let duty = (output / self.pwm.supply_voltage).clamp(0.0, 1.0);
        self.pwm.set_duty(duty);
        duty
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.pid.reset();
        self.pwm.duty = 0.0;
        self.speed_history.clear();
    }
}

impl fmt::Display for SpeedController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpeedCtrl(setpoint={:.2} rad/s, duty={:.1}%)",
            self.setpoint,
            self.pwm.duty * 100.0
        )
    }
}

// ── Position Controller ────────────────────────────────────────

/// Cascaded position controller: outer position loop drives inner speed loop.
#[derive(Debug, Clone)]
pub struct PositionController {
    /// Outer PID (position → speed setpoint).
    pub pos_pid: PidController,
    /// Inner speed controller.
    pub speed_ctrl: SpeedController,
    /// Target position (rad).
    pub target_position: f64,
    /// Maximum speed limit (rad/s).
    pub max_speed: f64,
}

impl PositionController {
    /// Create a cascaded position controller.
    pub fn new(
        pos_pid: PidController,
        speed_ctrl: SpeedController,
        max_speed: f64,
    ) -> Self {
        Self {
            pos_pid,
            speed_ctrl,
            target_position: 0.0,
            max_speed: max_speed.abs(),
        }
    }

    /// Builder: set maximum speed limit.
    pub fn with_max_speed(mut self, max: f64) -> Self {
        self.max_speed = max.abs();
        self
    }

    /// Set target position (rad).
    pub fn set_position(&mut self, theta: f64) {
        self.target_position = theta;
    }

    /// Update with measured position and speed, return duty cycle.
    pub fn update(&mut self, measured_theta: f64, measured_omega: f64, dt: f64) -> f64 {
        let pos_error = self.target_position - measured_theta;
        let speed_cmd = self.pos_pid.compute(pos_error, dt);
        let speed_cmd = speed_cmd.clamp(-self.max_speed, self.max_speed);

        self.speed_ctrl.set_speed(speed_cmd);
        self.speed_ctrl.update(measured_omega, dt)
    }

    /// Reset both loops.
    pub fn reset(&mut self) {
        self.pos_pid.reset();
        self.speed_ctrl.reset();
    }
}

impl fmt::Display for PositionController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PosCtrl(target={:.3} rad, max_spd={:.1} rad/s)",
            self.target_position, self.max_speed
        )
    }
}

// ── Current Limiter ────────────────────────────────────────────

/// Current limiter with soft and hard thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct CurrentLimiter {
    /// Soft limit — begins proportional reduction.
    pub soft_limit: f64,
    /// Hard limit — clamps to zero duty.
    pub hard_limit: f64,
    /// Thermal model: accumulated I²t.
    pub i2t_accumulator: f64,
    /// I²t trip threshold.
    pub i2t_limit: f64,
    /// Tripped flag.
    pub tripped: bool,
}

impl CurrentLimiter {
    /// Create a current limiter.
    pub fn new(soft_limit: f64, hard_limit: f64) -> Result<Self, MotorError> {
        if soft_limit <= 0.0 || hard_limit <= 0.0 {
            return Err(MotorError::InvalidParameter("limits must be > 0".into()));
        }
        if soft_limit > hard_limit {
            return Err(MotorError::InvalidParameter(
                "soft limit must be <= hard limit".into(),
            ));
        }
        Ok(Self {
            soft_limit,
            hard_limit,
            i2t_accumulator: 0.0,
            i2t_limit: hard_limit * hard_limit * 10.0,
            tripped: false,
        })
    }

    /// Builder: set I²t limit.
    pub fn with_i2t_limit(mut self, limit: f64) -> Self {
        self.i2t_limit = limit.abs();
        self
    }

    /// Apply current limiting, returning a scaling factor [0, 1] for duty.
    pub fn apply(&mut self, current: f64, dt: f64) -> Result<f64, MotorError> {
        let abs_i = current.abs();

        // I²t accumulation.
        self.i2t_accumulator += abs_i * abs_i * dt;
        // Slow decay.
        self.i2t_accumulator *= (-dt / 10.0_f64).exp();

        if self.i2t_accumulator > self.i2t_limit {
            self.tripped = true;
        }

        if self.tripped {
            return Err(MotorError::CurrentLimitExceeded {
                measured: abs_i,
                limit: self.hard_limit,
            });
        }

        if abs_i >= self.hard_limit {
            return Ok(0.0);
        }
        if abs_i <= self.soft_limit {
            return Ok(1.0);
        }

        // Linear ramp-down between soft and hard limits.
        let scale = 1.0 - (abs_i - self.soft_limit) / (self.hard_limit - self.soft_limit);
        Ok(scale.clamp(0.0, 1.0))
    }

    /// Reset the limiter (clear trip and accumulator).
    pub fn reset(&mut self) {
        self.i2t_accumulator = 0.0;
        self.tripped = false;
    }
}

impl fmt::Display for CurrentLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CurrentLimiter(soft={:.1}A, hard={:.1}A, tripped={})",
            self.soft_limit, self.hard_limit, self.tripped
        )
    }
}

// ── Back-EMF Estimator ─────────────────────────────────────────

/// Sensorless speed estimation via back-EMF measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct BackEmfEstimator {
    /// Motor resistance.
    pub resistance: f64,
    /// Motor Ke constant.
    pub ke: f64,
    /// Low-pass filter coefficient.
    pub alpha: f64,
    /// Filtered speed estimate (rad/s).
    pub estimated_omega: f64,
}

impl BackEmfEstimator {
    /// Create a back-EMF estimator.
    pub fn new(resistance: f64, ke: f64, alpha: f64) -> Result<Self, MotorError> {
        if ke.abs() < 1e-12 {
            return Err(MotorError::InvalidParameter("Ke must be nonzero".into()));
        }
        Ok(Self {
            resistance,
            ke,
            alpha: alpha.clamp(0.0, 0.999),
            estimated_omega: 0.0,
        })
    }

    /// Update estimate from measured voltage and current.
    pub fn update(&mut self, voltage: f64, current: f64) -> f64 {
        let emf = voltage - self.resistance * current;
        let omega_raw = emf / self.ke;
        self.estimated_omega =
            self.alpha * self.estimated_omega + (1.0 - self.alpha) * omega_raw;
        self.estimated_omega
    }

    /// Estimated speed in RPM.
    pub fn speed_rpm(&self) -> f64 {
        self.estimated_omega * 60.0 / (2.0 * std::f64::consts::PI)
    }
}

impl fmt::Display for BackEmfEstimator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BackEMF(ω_est={:.2} rad/s, {:.0} RPM)",
            self.estimated_omega,
            self.speed_rpm()
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f64 = 0.001; // 1 ms

    #[test]
    fn test_motor_model_creation() {
        let m = DcMotorModel::new(1.0, 0.001, 0.01, 0.01, 1e-4, 1e-4);
        assert!(m.is_ok());
    }

    #[test]
    fn test_motor_invalid_resistance() {
        let m = DcMotorModel::new(-1.0, 0.001, 0.01, 0.01, 1e-4, 1e-4);
        assert!(m.is_err());
    }

    #[test]
    fn test_motor_spins_up() {
        let mut m = DcMotorModel::hobby_motor();
        for _ in 0..5000 {
            m.step(12.0, 0.0, DT);
        }
        assert!(m.omega > 100.0, "motor should spin up: ω={}", m.omega);
    }

    #[test]
    fn test_motor_back_emf_rises() {
        let mut m = DcMotorModel::hobby_motor();
        for _ in 0..1000 {
            m.step(12.0, 0.0, DT);
        }
        assert!(m.back_emf() > 0.0);
    }

    #[test]
    fn test_motor_position_increases() {
        let mut m = DcMotorModel::hobby_motor();
        for _ in 0..1000 {
            m.step(6.0, 0.0, DT);
        }
        assert!(m.theta > 0.0);
    }

    #[test]
    fn test_motor_reset() {
        let mut m = DcMotorModel::hobby_motor();
        m.step(12.0, 0.0, DT);
        m.reset();
        assert_eq!(m.omega, 0.0);
        assert_eq!(m.current, 0.0);
        assert_eq!(m.theta, 0.0);
    }

    #[test]
    fn test_pwm_duty_clamping() {
        let mut pwm = PwmDutyCycle::new(24.0, 20_000.0).unwrap();
        pwm.set_duty(1.5);
        assert!((pwm.duty - 1.0).abs() < 1e-9);
        pwm.set_duty(-0.5);
        assert!((pwm.duty).abs() < 1e-9);
    }

    #[test]
    fn test_pwm_effective_voltage() {
        let mut pwm = PwmDutyCycle::new(12.0, 20_000.0).unwrap();
        pwm.set_duty(0.5);
        assert!((pwm.effective_voltage() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_pwm_on_time() {
        let mut pwm = PwmDutyCycle::new(12.0, 1000.0).unwrap();
        pwm.set_duty(0.25);
        assert!((pwm.on_time_s() - 0.00025).abs() < 1e-9);
    }

    #[test]
    fn test_pid_proportional() {
        let mut pid = PidController::new(2.0, 0.0, 0.0);
        let out = pid.compute(1.0, DT);
        assert!((out - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_pid_integral_accumulates() {
        let mut pid = PidController::new(0.0, 10.0, 0.0);
        pid.compute(1.0, 0.1);
        let out = pid.compute(1.0, 0.1);
        // integral = 1.0*0.1 + 1.0*0.1 = 0.2, Ki=10 → 2.0
        assert!((out - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_pid_derivative() {
        let mut pid = PidController::new(0.0, 0.0, 1.0);
        pid.compute(0.0, DT);
        let out = pid.compute(1.0, DT);
        // de/dt = (1 - 0) / 0.001 = 1000, Kd=1 → 1000
        assert!((out - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn test_pid_output_limits() {
        let mut pid = PidController::new(100.0, 0.0, 0.0)
            .with_output_limits(-5.0, 5.0);
        let out = pid.compute(1.0, DT);
        assert!((out - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_pid_integral_windup() {
        let mut pid = PidController::new(0.0, 100.0, 0.0)
            .with_integral_limit(1.0);
        for _ in 0..1000 {
            pid.compute(10.0, 0.1);
        }
        // Integral should be clamped to 1.0; output = 100 * 1.0 = 100
        assert!((pid.compute(10.0, 0.1) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_speed_controller_converges() {
        let pid = PidController::new(0.1, 0.5, 0.001)
            .with_output_limits(0.0, 24.0);
        let pwm = PwmDutyCycle::new(24.0, 20_000.0).unwrap();
        let mut ctrl = SpeedController::new(pid, pwm);
        let mut motor = DcMotorModel::hobby_motor();

        ctrl.set_speed(100.0);
        for _ in 0..10_000 {
            let duty = ctrl.update(motor.omega, DT);
            let voltage = duty * 24.0;
            motor.step(voltage, 0.0, DT);
        }
        let error = (motor.omega - 100.0).abs();
        assert!(error < 20.0, "speed should converge near 100: ω={}", motor.omega);
    }

    #[test]
    fn test_current_limiter_below_soft() {
        let mut lim = CurrentLimiter::new(5.0, 10.0).unwrap();
        let scale = lim.apply(3.0, DT).unwrap();
        assert!((scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_current_limiter_above_hard() {
        let mut lim = CurrentLimiter::new(5.0, 10.0).unwrap();
        let scale = lim.apply(15.0, DT).unwrap();
        assert!((scale).abs() < 1e-9);
    }

    #[test]
    fn test_current_limiter_between_limits() {
        let mut lim = CurrentLimiter::new(5.0, 10.0).unwrap();
        let scale = lim.apply(7.5, DT).unwrap();
        assert!(scale > 0.0 && scale < 1.0);
    }

    #[test]
    fn test_back_emf_estimator() {
        let mut est = BackEmfEstimator::new(0.5, 0.01, 0.5).unwrap();
        // Motor at 100 rad/s: V = R*i + Ke*ω ≈ 0.5*2 + 0.01*100 = 2.0
        let omega = est.update(2.0, 2.0);
        // emf = 2.0 - 0.5*2.0 = 1.0, omega_raw = 1.0/0.01 = 100
        assert!(omega > 0.0);
    }

    #[test]
    fn test_display_motor() {
        let m = DcMotorModel::hobby_motor();
        let s = format!("{m}");
        assert!(s.contains("DcMotor"));
    }

    #[test]
    fn test_display_pid() {
        let pid = PidController::new(1.0, 2.0, 3.0);
        let s = format!("{pid}");
        assert!(s.contains("PID"));
    }
}
