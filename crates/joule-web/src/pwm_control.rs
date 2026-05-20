//! PWM Generation — Frequency/duty-cycle configuration, edge-aligned and
//! center-aligned modes, complementary outputs with dead-time insertion,
//! multi-channel PWM, and output compare emulation.
//!
//! Pure-Rust PWM controller suitable for embedded simulation workloads.
//! All math uses `f64`; no external crates.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// PWM control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum PwmError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Channel index out of bounds.
    ChannelOutOfBounds { index: usize, count: usize },
    /// Dead time exceeds half-period.
    DeadTimeTooLarge { dead_time_s: f64, half_period_s: f64 },
}

impl fmt::Display for PwmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::ChannelOutOfBounds { index, count } => {
                write!(f, "channel {index} out of bounds (count={count})")
            }
            Self::DeadTimeTooLarge { dead_time_s, half_period_s } => {
                write!(
                    f,
                    "dead time {dead_time_s:.3e}s exceeds half-period {half_period_s:.3e}s"
                )
            }
        }
    }
}

impl std::error::Error for PwmError {}

// ── Alignment Mode ─────────────────────────────────────────────

/// PWM alignment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMode {
    /// Edge-aligned (sawtooth counter).
    EdgeAligned,
    /// Center-aligned (triangle counter).
    CenterAligned,
}

impl fmt::Display for AlignmentMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EdgeAligned => write!(f, "edge-aligned"),
            Self::CenterAligned => write!(f, "center-aligned"),
        }
    }
}

// ── PWM Channel ────────────────────────────────────────────────

/// State of a single PWM output at a given simulation time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PwmOutput {
    /// High (true) or low (false).
    pub level: bool,
    /// Time within the period (seconds).
    pub phase_time: f64,
}

/// A single PWM channel with duty, polarity, and phase offset.
#[derive(Debug, Clone, PartialEq)]
pub struct PwmChannel {
    /// Channel index.
    pub index: usize,
    /// Duty cycle [0.0, 1.0].
    pub duty: f64,
    /// Phase offset [0.0, 1.0) as fraction of period.
    pub phase_offset: f64,
    /// Inverted polarity.
    pub inverted: bool,
    /// Channel enabled.
    pub enabled: bool,
}

impl PwmChannel {
    /// Create a new PWM channel.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            duty: 0.0,
            phase_offset: 0.0,
            inverted: false,
            enabled: true,
        }
    }

    /// Builder: set duty cycle.
    pub fn with_duty(mut self, duty: f64) -> Self {
        self.duty = duty.clamp(0.0, 1.0);
        self
    }

    /// Builder: set phase offset.
    pub fn with_phase_offset(mut self, offset: f64) -> Self {
        self.phase_offset = offset.rem_euclid(1.0);
        self
    }

    /// Builder: set inverted polarity.
    pub fn with_inverted(mut self, inv: bool) -> Self {
        self.inverted = inv;
        self
    }

    /// Evaluate the output level at a given phase within the period [0, 1).
    pub fn evaluate_edge_aligned(&self, phase: f64) -> bool {
        if !self.enabled {
            return false;
        }
        let p = (phase + self.phase_offset).rem_euclid(1.0);
        let active = p < self.duty;
        if self.inverted { !active } else { active }
    }

    /// Evaluate center-aligned PWM at a given phase [0, 1).
    ///
    /// Triangle counter: ramps 0→1→0 within one period.
    pub fn evaluate_center_aligned(&self, phase: f64) -> bool {
        if !self.enabled {
            return false;
        }
        let p = (phase + self.phase_offset).rem_euclid(1.0);
        // Triangle wave: map phase to [0, 1] triangular.
        let triangle = if p < 0.5 { p * 2.0 } else { 2.0 - p * 2.0 };
        let threshold = self.duty;
        let active = triangle < threshold;
        if self.inverted { !active } else { active }
    }
}

impl fmt::Display for PwmChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ch{}(duty={:.1}%, phase={:.2}, inv={}, en={})",
            self.index,
            self.duty * 100.0,
            self.phase_offset,
            self.inverted,
            self.enabled
        )
    }
}

// ── Dead-Time Inserter ─────────────────────────────────────────

/// Dead-time insertion for complementary PWM pairs.
///
/// Prevents shoot-through in H-bridge / half-bridge drivers by ensuring
/// both high-side and low-side FETs are never on simultaneously.
#[derive(Debug, Clone, PartialEq)]
pub struct DeadTimeInserter {
    /// Rising-edge dead time (seconds).
    pub rising_dt: f64,
    /// Falling-edge dead time (seconds).
    pub falling_dt: f64,
    /// Previous high-side state.
    prev_high: bool,
    /// Previous low-side state.
    prev_low: bool,
    /// Time since last high-side transition.
    high_transition_timer: f64,
    /// Time since last low-side transition.
    low_transition_timer: f64,
}

impl DeadTimeInserter {
    /// Create a dead-time inserter with symmetric dead time.
    pub fn new(dead_time_s: f64) -> Result<Self, PwmError> {
        if dead_time_s < 0.0 {
            return Err(PwmError::InvalidParameter(
                "dead time must be >= 0".into(),
            ));
        }
        Ok(Self {
            rising_dt: dead_time_s,
            falling_dt: dead_time_s,
            prev_high: false,
            prev_low: false,
            high_transition_timer: f64::INFINITY,
            low_transition_timer: f64::INFINITY,
        })
    }

    /// Builder: set asymmetric dead times.
    pub fn with_asymmetric(mut self, rising: f64, falling: f64) -> Self {
        self.rising_dt = rising.max(0.0);
        self.falling_dt = falling.max(0.0);
        self
    }

    /// Process a complementary pair: given desired high-side output,
    /// produce (high_out, low_out) with dead-time insertion.
    pub fn process(&mut self, desired_high: bool, dt: f64) -> (bool, bool) {
        // Track transitions.
        if desired_high != self.prev_high {
            self.high_transition_timer = 0.0;
        } else {
            self.high_transition_timer += dt;
        }

        let desired_low = !desired_high;
        if desired_low != self.prev_low {
            self.low_transition_timer = 0.0;
        } else {
            self.low_transition_timer += dt;
        }

        // Apply dead time: delay rising edges.
        let high_out = if desired_high {
            self.high_transition_timer >= self.rising_dt
        } else {
            false
        };

        let low_out = if desired_low {
            self.low_transition_timer >= self.falling_dt
        } else {
            false
        };

        self.prev_high = desired_high;
        self.prev_low = desired_low;

        // Safety: never allow both on.
        if high_out && low_out {
            return (false, false);
        }

        (high_out, low_out)
    }

    /// Effective dead time (maximum of rising/falling).
    pub fn max_dead_time(&self) -> f64 {
        self.rising_dt.max(self.falling_dt)
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.prev_high = false;
        self.prev_low = false;
        self.high_transition_timer = f64::INFINITY;
        self.low_transition_timer = f64::INFINITY;
    }
}

impl fmt::Display for DeadTimeInserter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DeadTime(rise={:.1e}s, fall={:.1e}s)",
            self.rising_dt, self.falling_dt
        )
    }
}

// ── Multi-Channel PWM Generator ────────────────────────────────

/// Multi-channel PWM generator with configurable alignment and complementary
/// output support.
#[derive(Debug, Clone)]
pub struct PwmGenerator {
    /// PWM frequency (Hz).
    pub frequency: f64,
    /// Alignment mode.
    pub alignment: AlignmentMode,
    /// Channels.
    pub channels: Vec<PwmChannel>,
    /// Dead-time inserter (shared across complementary pairs).
    pub dead_time: Option<DeadTimeInserter>,
    /// Timer prescaler (integer divisor of system clock).
    pub prescaler: u32,
    /// Current phase [0.0, 1.0).
    phase: f64,
}

impl PwmGenerator {
    /// Create a new PWM generator.
    pub fn new(frequency: f64, num_channels: usize) -> Result<Self, PwmError> {
        if frequency <= 0.0 {
            return Err(PwmError::InvalidParameter("frequency must be > 0".into()));
        }
        let channels = (0..num_channels).map(PwmChannel::new).collect();
        Ok(Self {
            frequency,
            alignment: AlignmentMode::EdgeAligned,
            channels,
            dead_time: None,
            prescaler: 1,
            phase: 0.0,
        })
    }

    /// Builder: set alignment mode.
    pub fn with_alignment(mut self, mode: AlignmentMode) -> Self {
        self.alignment = mode;
        self
    }

    /// Builder: set dead time for complementary outputs.
    pub fn with_dead_time(mut self, dead_time_s: f64) -> Result<Self, PwmError> {
        let half_period = 0.5 / self.frequency;
        if dead_time_s > half_period {
            return Err(PwmError::DeadTimeTooLarge {
                dead_time_s,
                half_period_s: half_period,
            });
        }
        self.dead_time = Some(DeadTimeInserter::new(dead_time_s)?);
        Ok(self)
    }

    /// Builder: set prescaler.
    pub fn with_prescaler(mut self, prescaler: u32) -> Self {
        self.prescaler = prescaler.max(1);
        self
    }

    /// PWM period in seconds.
    pub fn period_s(&self) -> f64 {
        1.0 / self.frequency
    }

    /// Set duty cycle for a channel.
    pub fn set_duty(&mut self, channel: usize, duty: f64) -> Result<(), PwmError> {
        if channel >= self.channels.len() {
            return Err(PwmError::ChannelOutOfBounds {
                index: channel,
                count: self.channels.len(),
            });
        }
        self.channels[channel].duty = duty.clamp(0.0, 1.0);
        Ok(())
    }

    /// Set all channels to the same duty.
    pub fn set_duty_all(&mut self, duty: f64) {
        let d = duty.clamp(0.0, 1.0);
        for ch in &mut self.channels {
            ch.duty = d;
        }
    }

    /// Evaluate all channel outputs at the current phase.
    pub fn evaluate(&self) -> Vec<PwmOutput> {
        self.channels
            .iter()
            .map(|ch| {
                let level = match self.alignment {
                    AlignmentMode::EdgeAligned => ch.evaluate_edge_aligned(self.phase),
                    AlignmentMode::CenterAligned => ch.evaluate_center_aligned(self.phase),
                };
                PwmOutput {
                    level,
                    phase_time: self.phase * self.period_s(),
                }
            })
            .collect()
    }

    /// Evaluate complementary outputs for channel pairs (0,1), (2,3), ...
    pub fn evaluate_complementary(&mut self, dt: f64) -> Vec<(bool, bool)> {
        let outputs = self.evaluate();
        let mut pairs = Vec::new();
        let mut i = 0;
        while i + 1 < outputs.len() {
            let high_desired = outputs[i].level;
            if let Some(ref mut dti) = self.dead_time {
                pairs.push(dti.process(high_desired, dt));
            } else {
                pairs.push((high_desired, !high_desired));
            }
            i += 2;
        }
        pairs
    }

    /// Advance the phase by `dt` seconds.
    pub fn advance(&mut self, dt: f64) {
        let phase_inc = dt * self.frequency;
        self.phase = (self.phase + phase_inc).rem_euclid(1.0);
    }

    /// Step: advance and evaluate.
    pub fn step(&mut self, dt: f64) -> Vec<PwmOutput> {
        self.advance(dt);
        self.evaluate()
    }

    /// Reset phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        if let Some(ref mut dti) = self.dead_time {
            dti.reset();
        }
    }

    /// Compute the effective timer auto-reload value for a given system clock.
    pub fn auto_reload_value(&self, system_clock_hz: f64) -> u32 {
        let effective_clock = system_clock_hz / self.prescaler as f64;
        let arr = (effective_clock / self.frequency).round() as u32;
        arr.saturating_sub(1)
    }

    /// Compute compare value for a given duty cycle and auto-reload.
    pub fn compare_value(duty: f64, auto_reload: u32) -> u32 {
        let ccr = (duty * (auto_reload + 1) as f64).round() as u32;
        ccr.min(auto_reload + 1)
    }
}

impl fmt::Display for PwmGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PWMGen(f={:.0}Hz, {}ch, {}, phase={:.4})",
            self.frequency,
            self.channels.len(),
            self.alignment,
            self.phase
        )
    }
}

// ── Duty Ramp Generator ───────────────────────────────────────

/// Soft-start / soft-stop duty cycle ramp generator.
#[derive(Debug, Clone, PartialEq)]
pub struct DutyRamp {
    /// Current duty.
    pub current: f64,
    /// Target duty.
    pub target: f64,
    /// Ramp rate (duty per second).
    pub ramp_rate: f64,
}

impl DutyRamp {
    /// Create a ramp generator.
    pub fn new(ramp_rate: f64) -> Self {
        Self {
            current: 0.0,
            target: 0.0,
            ramp_rate: ramp_rate.abs(),
        }
    }

    /// Builder: set initial duty.
    pub fn with_initial(mut self, duty: f64) -> Self {
        self.current = duty.clamp(0.0, 1.0);
        self
    }

    /// Set new target duty.
    pub fn set_target(&mut self, duty: f64) {
        self.target = duty.clamp(0.0, 1.0);
    }

    /// Step the ramp; returns current duty.
    pub fn step(&mut self, dt: f64) -> f64 {
        let error = self.target - self.current;
        let max_change = self.ramp_rate * dt;

        if error.abs() <= max_change {
            self.current = self.target;
        } else {
            self.current += max_change * error.signum();
        }
        self.current
    }

    /// Whether the ramp has reached the target.
    pub fn is_settled(&self) -> bool {
        (self.current - self.target).abs() < 1e-9
    }
}

impl fmt::Display for DutyRamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DutyRamp(current={:.1}%, target={:.1}%, rate={:.1}/s)",
            self.current * 100.0,
            self.target * 100.0,
            self.ramp_rate
        )
    }
}

// ── Frequency Sweep ────────────────────────────────────────────

/// Linear frequency sweep generator for motor testing / system identification.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencySweep {
    /// Start frequency (Hz).
    pub start_hz: f64,
    /// End frequency (Hz).
    pub end_hz: f64,
    /// Sweep duration (s).
    pub duration: f64,
    /// Current time (s).
    pub time: f64,
    /// Fixed duty cycle during sweep.
    pub duty: f64,
}

impl FrequencySweep {
    /// Create a frequency sweep.
    pub fn new(start_hz: f64, end_hz: f64, duration: f64) -> Result<Self, PwmError> {
        if duration <= 0.0 {
            return Err(PwmError::InvalidParameter("duration must be > 0".into()));
        }
        if start_hz <= 0.0 || end_hz <= 0.0 {
            return Err(PwmError::InvalidParameter("frequencies must be > 0".into()));
        }
        Ok(Self {
            start_hz,
            end_hz,
            duration,
            time: 0.0,
            duty: 0.5,
        })
    }

    /// Builder: set duty cycle.
    pub fn with_duty(mut self, duty: f64) -> Self {
        self.duty = duty.clamp(0.0, 1.0);
        self
    }

    /// Current frequency (Hz).
    pub fn current_frequency(&self) -> f64 {
        let t = (self.time / self.duration).clamp(0.0, 1.0);
        self.start_hz + t * (self.end_hz - self.start_hz)
    }

    /// Advance time; returns current frequency.
    pub fn advance(&mut self, dt: f64) -> f64 {
        self.time += dt;
        self.current_frequency()
    }

    /// Whether the sweep is complete.
    pub fn is_complete(&self) -> bool {
        self.time >= self.duration
    }

    /// Reset sweep.
    pub fn reset(&mut self) {
        self.time = 0.0;
    }
}

impl fmt::Display for FrequencySweep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sweep({:.0}→{:.0}Hz, t={:.2}/{:.2}s)",
            self.start_hz, self.end_hz, self.time, self.duration
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_edge_aligned_50pct() {
        let ch = PwmChannel::new(0).with_duty(0.5);
        assert!(ch.evaluate_edge_aligned(0.0));
        assert!(ch.evaluate_edge_aligned(0.49));
        assert!(!ch.evaluate_edge_aligned(0.51));
    }

    #[test]
    fn test_channel_edge_aligned_inverted() {
        let ch = PwmChannel::new(0).with_duty(0.5).with_inverted(true);
        assert!(!ch.evaluate_edge_aligned(0.0));
        assert!(ch.evaluate_edge_aligned(0.75));
    }

    #[test]
    fn test_channel_center_aligned_50pct() {
        let ch = PwmChannel::new(0).with_duty(0.5);
        // At phase=0.25 triangle=0.5, duty=0.5, triangle < duty → false at boundary.
        // At phase=0.0 triangle=0.0, 0<0.5 → true.
        assert!(ch.evaluate_center_aligned(0.0));
    }

    #[test]
    fn test_channel_disabled() {
        let mut ch = PwmChannel::new(0).with_duty(0.5);
        ch.enabled = false;
        assert!(!ch.evaluate_edge_aligned(0.0));
        assert!(!ch.evaluate_center_aligned(0.0));
    }

    #[test]
    fn test_channel_phase_offset() {
        let ch = PwmChannel::new(0).with_duty(0.25).with_phase_offset(0.5);
        // With offset 0.5, phase 0.0 → effective 0.5, outside 0.25 duty → false.
        assert!(!ch.evaluate_edge_aligned(0.0));
        // phase 0.6 → effective 1.1 mod 1 = 0.1, inside 0.25 → true.
        assert!(ch.evaluate_edge_aligned(0.6));
    }

    #[test]
    fn test_generator_creation() {
        let pwm = PwmGenerator::new(20_000.0, 4);
        assert!(pwm.is_ok());
        assert_eq!(pwm.unwrap().channels.len(), 4);
    }

    #[test]
    fn test_generator_invalid_freq() {
        let pwm = PwmGenerator::new(0.0, 2);
        assert!(pwm.is_err());
    }

    #[test]
    fn test_generator_set_duty() {
        let mut pwm = PwmGenerator::new(1000.0, 2).unwrap();
        assert!(pwm.set_duty(0, 0.75).is_ok());
        assert!((pwm.channels[0].duty - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_generator_set_duty_oob() {
        let mut pwm = PwmGenerator::new(1000.0, 2).unwrap();
        assert!(pwm.set_duty(5, 0.5).is_err());
    }

    #[test]
    fn test_generator_period() {
        let pwm = PwmGenerator::new(20_000.0, 1).unwrap();
        assert!((pwm.period_s() - 5e-5).abs() < 1e-12);
    }

    #[test]
    fn test_generator_step_advances_phase() {
        let mut pwm = PwmGenerator::new(1000.0, 1).unwrap();
        pwm.step(0.0005); // half period
        assert!((pwm.phase - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dead_time_both_off_during_transition() {
        let mut dti = DeadTimeInserter::new(100e-6).unwrap();
        // Start with high off.
        dti.process(false, 1e-3);
        // Transition to high: during dead time both should be off.
        let (h, l) = dti.process(true, 10e-6);
        // High just started; timer < 100µs.
        assert!(!h, "high should be delayed");
        // Low just turned off → desired_low=false → low_out=false.
        assert!(!l, "low should be off after transition");
    }

    #[test]
    fn test_dead_time_high_after_delay() {
        let mut dti = DeadTimeInserter::new(50e-6).unwrap();
        dti.process(false, 1e-3);
        // Transition to high.
        dti.process(true, 10e-6);
        // After sufficient time, high should be on.
        let (h, _l) = dti.process(true, 100e-6);
        assert!(h, "high should be on after dead time");
    }

    #[test]
    fn test_auto_reload_value() {
        let pwm = PwmGenerator::new(20_000.0, 1).unwrap();
        // 72 MHz system clock, prescaler=1.
        let arr = pwm.auto_reload_value(72_000_000.0);
        // 72e6/20e3 = 3600, ARR = 3599.
        assert_eq!(arr, 3599);
    }

    #[test]
    fn test_compare_value() {
        let ccr = PwmGenerator::compare_value(0.5, 3599);
        assert_eq!(ccr, 1800);
    }

    #[test]
    fn test_duty_ramp() {
        let mut ramp = DutyRamp::new(1.0); // 1.0/s → takes 1s to go 0→1.
        ramp.set_target(1.0);
        for _ in 0..100 {
            ramp.step(0.01);
        }
        assert!(ramp.is_settled());
    }

    #[test]
    fn test_duty_ramp_partial() {
        let mut ramp = DutyRamp::new(2.0);
        ramp.set_target(0.5);
        let d = ramp.step(0.1);
        // max_change = 2.0 * 0.1 = 0.2; from 0 → 0.2.
        assert!((d - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_frequency_sweep() {
        let mut sweep = FrequencySweep::new(1000.0, 10_000.0, 1.0).unwrap();
        assert!((sweep.current_frequency() - 1000.0).abs() < 1e-6);
        sweep.advance(0.5);
        assert!((sweep.current_frequency() - 5500.0).abs() < 1e-6);
        sweep.advance(0.5);
        assert!((sweep.current_frequency() - 10_000.0).abs() < 1e-6);
        assert!(sweep.is_complete());
    }

    #[test]
    fn test_display_generator() {
        let pwm = PwmGenerator::new(20_000.0, 4).unwrap();
        let s = format!("{pwm}");
        assert!(s.contains("PWMGen"));
    }

    #[test]
    fn test_display_channel() {
        let ch = PwmChannel::new(2).with_duty(0.5);
        let s = format!("{ch}");
        assert!(s.contains("Ch2"));
    }
}
