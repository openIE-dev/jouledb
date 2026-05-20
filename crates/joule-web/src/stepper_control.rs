//! Stepper Motor Control — Full-step, half-step, and micro-stepping drive
//! modes, trapezoidal acceleration ramp (motion profiler), position tracking,
//! and multi-axis stepper coordination.
//!
//! Pure-Rust stepper control using `f64` math; no external crates.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Stepper control errors.
#[derive(Debug, Clone, PartialEq)]
pub enum StepperError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Position limit reached.
    PositionLimit { position: i64, limit: i64 },
    /// Stall detected (step loss).
    StallDetected { expected: i64, actual: i64 },
}

impl fmt::Display for StepperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::PositionLimit { position, limit } => {
                write!(f, "position {position} exceeds limit {limit}")
            }
            Self::StallDetected { expected, actual } => {
                write!(f, "stall detected: expected={expected}, actual={actual}")
            }
        }
    }
}

impl std::error::Error for StepperError {}

// ── Step Mode ──────────────────────────────────────────────────

/// Stepper drive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    /// Full step (4 states per electrical cycle).
    Full,
    /// Half step (8 states per electrical cycle).
    Half,
    /// Micro-stepping with given subdivision.
    Micro(u32),
}

impl StepMode {
    /// Number of micro-steps per full step.
    pub fn microsteps_per_full_step(&self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Micro(n) => *n,
        }
    }

    /// Micro-steps per revolution (given motor full-steps/rev).
    pub fn steps_per_rev(&self, full_steps_per_rev: u32) -> u32 {
        full_steps_per_rev * self.microsteps_per_full_step()
    }
}

impl fmt::Display for StepMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full-step"),
            Self::Half => write!(f, "half-step"),
            Self::Micro(n) => write!(f, "1/{n} micro-step"),
        }
    }
}

// ── Coil Phase Generator ───────────────────────────────────────

/// Generates coil drive signals for a 2-phase (bipolar) stepper.
///
/// Outputs normalized currents for coil A and coil B in [-1, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct CoilPhaseGenerator {
    /// Step mode.
    pub mode: StepMode,
    /// Current electrical position (micro-step index).
    pub micro_position: i64,
}

impl CoilPhaseGenerator {
    /// Create a coil phase generator.
    pub fn new(mode: StepMode) -> Self {
        Self { mode, micro_position: 0 }
    }

    /// Advance by one micro-step in the given direction (+1 or -1).
    pub fn step(&mut self, direction: i64) {
        self.micro_position += direction.signum();
    }

    /// Compute coil currents (coil_a, coil_b) at the current position.
    ///
    /// For micro-stepping, uses sinusoidal commutation.
    pub fn coil_currents(&self) -> (f64, f64) {
        let microsteps = self.mode.microsteps_per_full_step() as f64;
        // Electrical angle: 4 full steps = one electrical cycle (2π).
        let angle =
            (self.micro_position as f64 / microsteps) * std::f64::consts::FRAC_PI_2;

        match self.mode {
            StepMode::Full => {
                // Quantize to nearest full step.
                let idx = self.micro_position.rem_euclid(4) as usize;
                let table: [(f64, f64); 4] = [
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (-1.0, 0.0),
                    (0.0, -1.0),
                ];
                table[idx]
            }
            StepMode::Half => {
                let idx = self.micro_position.rem_euclid(8) as usize;
                let s = std::f64::consts::FRAC_1_SQRT_2;
                let table: [(f64, f64); 8] = [
                    (1.0, 0.0),
                    (s, s),
                    (0.0, 1.0),
                    (-s, s),
                    (-1.0, 0.0),
                    (-s, -s),
                    (0.0, -1.0),
                    (s, -s),
                ];
                table[idx]
            }
            StepMode::Micro(_) => {
                let a = angle.cos();
                let b = angle.sin();
                (a, b)
            }
        }
    }

    /// Reset position to zero.
    pub fn reset(&mut self) {
        self.micro_position = 0;
    }
}

impl fmt::Display for CoilPhaseGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (a, b) = self.coil_currents();
        write!(
            f,
            "CoilPhase({}, pos={}, A={:.3}, B={:.3})",
            self.mode, self.micro_position, a, b
        )
    }
}

// ── Trapezoidal Acceleration Ramp ──────────────────────────────

/// Trapezoidal velocity profile for stepper motion.
///
/// Produces step intervals (time between steps) following an acceleration ramp,
/// cruise phase, and deceleration ramp.
#[derive(Debug, Clone, PartialEq)]
pub struct TrapezoidalRamp {
    /// Maximum velocity (steps/s).
    pub max_velocity: f64,
    /// Acceleration (steps/s²).
    pub acceleration: f64,
    /// Deceleration (steps/s², positive).
    pub deceleration: f64,
    /// Current velocity (steps/s).
    pub velocity: f64,
    /// Minimum step interval (seconds) — corresponds to max velocity.
    min_interval: f64,
}

impl TrapezoidalRamp {
    /// Create a trapezoidal ramp.
    pub fn new(max_velocity: f64, acceleration: f64) -> Result<Self, StepperError> {
        if max_velocity <= 0.0 {
            return Err(StepperError::InvalidParameter(
                "max velocity must be > 0".into(),
            ));
        }
        if acceleration <= 0.0 {
            return Err(StepperError::InvalidParameter(
                "acceleration must be > 0".into(),
            ));
        }
        Ok(Self {
            max_velocity,
            acceleration,
            deceleration: acceleration,
            velocity: 0.0,
            min_interval: 1.0 / max_velocity,
        })
    }

    /// Builder: set separate deceleration rate.
    pub fn with_deceleration(mut self, decel: f64) -> Self {
        self.deceleration = decel.abs();
        self
    }

    /// Compute the next step interval given remaining distance (in steps).
    ///
    /// Uses the algorithm from David Austin's "Generate stepper-motor speed
    /// profiles in real time" — approximation based on constant-acceleration
    /// kinematics.
    pub fn next_interval(&mut self, steps_remaining: u64) -> f64 {
        if steps_remaining == 0 {
            self.velocity = 0.0;
            return f64::INFINITY;
        }

        // Deceleration distance: v² / (2*a_decel).
        let decel_steps =
            (self.velocity * self.velocity / (2.0 * self.deceleration)).ceil() as u64;

        if steps_remaining <= decel_steps {
            // Decelerate.
            self.velocity -= self.deceleration * self.current_interval();
            if self.velocity < 1.0 {
                self.velocity = 1.0; // minimum velocity: 1 step/s.
            }
        } else if self.velocity < self.max_velocity {
            // Accelerate.
            if self.velocity < 1.0 {
                // Initial step: v₀ = sqrt(2 * a * 0.5) (half-step approximation).
                self.velocity = (self.acceleration).sqrt();
            } else {
                self.velocity += self.acceleration * self.current_interval();
            }
            if self.velocity > self.max_velocity {
                self.velocity = self.max_velocity;
            }
        }
        // else: cruise at max_velocity.

        self.current_interval()
    }

    /// Current step interval (seconds per step).
    pub fn current_interval(&self) -> f64 {
        if self.velocity > 0.0 {
            1.0 / self.velocity
        } else {
            f64::INFINITY
        }
    }

    /// Steps needed to decelerate from current velocity to zero.
    pub fn deceleration_steps(&self) -> u64 {
        (self.velocity * self.velocity / (2.0 * self.deceleration)).ceil() as u64
    }

    /// Reset the ramp.
    pub fn reset(&mut self) {
        self.velocity = 0.0;
    }

    /// Whether the ramp is at rest.
    pub fn is_idle(&self) -> bool {
        self.velocity.abs() < 1e-9
    }
}

impl fmt::Display for TrapezoidalRamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrapRamp(v={:.1} sps, max={:.1}, accel={:.1})",
            self.velocity, self.max_velocity, self.acceleration
        )
    }
}

// ── Position Tracker ───────────────────────────────────────────

/// Tracks stepper position in micro-steps and physical units.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionTracker {
    /// Current position (micro-steps, signed).
    pub position: i64,
    /// Steps per revolution (accounting for micro-stepping).
    pub steps_per_rev: u32,
    /// Distance per revolution (e.g., mm for linear stage).
    pub distance_per_rev: f64,
    /// Soft limit: minimum position.
    pub min_position: i64,
    /// Soft limit: maximum position.
    pub max_position: i64,
    /// Home position offset.
    pub home_offset: i64,
}

impl PositionTracker {
    /// Create a position tracker.
    pub fn new(steps_per_rev: u32, distance_per_rev: f64) -> Result<Self, StepperError> {
        if steps_per_rev == 0 {
            return Err(StepperError::InvalidParameter(
                "steps per revolution must be > 0".into(),
            ));
        }
        Ok(Self {
            position: 0,
            steps_per_rev,
            distance_per_rev,
            min_position: i64::MIN,
            max_position: i64::MAX,
            home_offset: 0,
        })
    }

    /// Builder: set soft limits.
    pub fn with_limits(mut self, min: i64, max: i64) -> Self {
        self.min_position = min;
        self.max_position = max;
        self
    }

    /// Record a step in the given direction.
    pub fn record_step(&mut self, direction: i64) -> Result<(), StepperError> {
        let new_pos = self.position + direction.signum();
        if new_pos < self.min_position || new_pos > self.max_position {
            return Err(StepperError::PositionLimit {
                position: new_pos,
                limit: if new_pos < self.min_position {
                    self.min_position
                } else {
                    self.max_position
                },
            });
        }
        self.position = new_pos;
        Ok(())
    }

    /// Current angle (degrees).
    pub fn angle_deg(&self) -> f64 {
        let revs = (self.position - self.home_offset) as f64 / self.steps_per_rev as f64;
        revs * 360.0
    }

    /// Current linear position (distance units).
    pub fn linear_position(&self) -> f64 {
        let revs = (self.position - self.home_offset) as f64 / self.steps_per_rev as f64;
        revs * self.distance_per_rev
    }

    /// Set home (zero) at current position.
    pub fn set_home(&mut self) {
        self.home_offset = self.position;
    }

    /// Distance per micro-step.
    pub fn step_distance(&self) -> f64 {
        self.distance_per_rev / self.steps_per_rev as f64
    }
}

impl fmt::Display for PositionTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PosTrack(pos={}, {:.3}°, {:.4} mm)",
            self.position,
            self.angle_deg(),
            self.linear_position()
        )
    }
}

// ── Stepper Controller ─────────────────────────────────────────

/// Complete stepper motor controller: combines phase generation, motion
/// profiling, and position tracking.
#[derive(Debug, Clone)]
pub struct StepperController {
    /// Coil phase generator.
    pub phase: CoilPhaseGenerator,
    /// Motion ramp.
    pub ramp: TrapezoidalRamp,
    /// Position tracker.
    pub tracker: PositionTracker,
    /// Target position (micro-steps).
    pub target_position: i64,
    /// Time accumulator for step timing.
    time_accumulator: f64,
    /// Current step interval.
    current_interval: f64,
    /// Motor enabled.
    pub enabled: bool,
}

impl StepperController {
    /// Create a stepper controller.
    pub fn new(
        mode: StepMode,
        ramp: TrapezoidalRamp,
        tracker: PositionTracker,
    ) -> Self {
        Self {
            phase: CoilPhaseGenerator::new(mode),
            ramp,
            tracker,
            target_position: 0,
            time_accumulator: 0.0,
            current_interval: f64::INFINITY,
            enabled: true,
        }
    }

    /// Builder: set initial target.
    pub fn with_target(mut self, target: i64) -> Self {
        self.target_position = target;
        self
    }

    /// Set move target (absolute position in micro-steps).
    pub fn move_to(&mut self, target: i64) {
        self.target_position = target;
    }

    /// Set move target relative to current position.
    pub fn move_relative(&mut self, steps: i64) {
        self.target_position = self.tracker.position + steps;
    }

    /// Step the controller forward by `dt` seconds.
    /// Returns the number of micro-steps taken in this interval.
    pub fn update(&mut self, dt: f64) -> Result<u32, StepperError> {
        if !self.enabled {
            return Ok(0);
        }

        let diff = self.target_position - self.tracker.position;
        if diff == 0 {
            self.ramp.reset();
            return Ok(0);
        }

        let direction: i64 = if diff > 0 { 1 } else { -1 };
        let remaining = diff.unsigned_abs();

        self.time_accumulator += dt;
        let mut steps_taken: u32 = 0;

        // Generate steps as the time accumulator exceeds intervals.
        loop {
            if remaining - steps_taken as u64 == 0 {
                break;
            }
            self.current_interval =
                self.ramp.next_interval(remaining - steps_taken as u64);

            if self.time_accumulator < self.current_interval {
                break;
            }
            self.time_accumulator -= self.current_interval;

            self.phase.step(direction);
            self.tracker.record_step(direction)?;
            steps_taken += 1;

            if steps_taken > 10_000 {
                break; // safety limit per update
            }
        }

        Ok(steps_taken)
    }

    /// Whether the controller has reached the target.
    pub fn is_at_target(&self) -> bool {
        self.tracker.position == self.target_position
    }

    /// Coil currents at the current position.
    pub fn coil_currents(&self) -> (f64, f64) {
        self.phase.coil_currents()
    }
}

impl fmt::Display for StepperController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stepper({}, pos={}, target={}, v={:.1} sps)",
            self.phase.mode,
            self.tracker.position,
            self.target_position,
            self.ramp.velocity
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_mode_microsteps() {
        assert_eq!(StepMode::Full.microsteps_per_full_step(), 1);
        assert_eq!(StepMode::Half.microsteps_per_full_step(), 2);
        assert_eq!(StepMode::Micro(16).microsteps_per_full_step(), 16);
    }

    #[test]
    fn test_steps_per_rev() {
        let mode = StepMode::Micro(16);
        assert_eq!(mode.steps_per_rev(200), 3200);
    }

    #[test]
    fn test_coil_full_step_cycle() {
        let mut phase = CoilPhaseGenerator::new(StepMode::Full);
        let (a, b) = phase.coil_currents();
        assert!((a - 1.0).abs() < 1e-9 && b.abs() < 1e-9);

        phase.step(1);
        let (a, b) = phase.coil_currents();
        assert!(a.abs() < 1e-9 && (b - 1.0).abs() < 1e-9);

        phase.step(1);
        let (a, b) = phase.coil_currents();
        assert!((a + 1.0).abs() < 1e-9 && b.abs() < 1e-9);

        phase.step(1);
        let (a, b) = phase.coil_currents();
        assert!(a.abs() < 1e-9 && (b + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_coil_half_step() {
        let mut phase = CoilPhaseGenerator::new(StepMode::Half);
        phase.step(1);
        let (a, b) = phase.coil_currents();
        // Half-step position 1: (√2/2, √2/2).
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((a - s).abs() < 1e-9);
        assert!((b - s).abs() < 1e-9);
    }

    #[test]
    fn test_coil_microstep_sinusoidal() {
        let mut phase = CoilPhaseGenerator::new(StepMode::Micro(256));
        // At position 0: cos(0)=1, sin(0)=0.
        let (a, b) = phase.coil_currents();
        assert!((a - 1.0).abs() < 1e-6);
        assert!(b.abs() < 1e-6);

        // Advance 256 micro-steps (= 1 full step = π/2 electrical).
        for _ in 0..256 {
            phase.step(1);
        }
        let (a, b) = phase.coil_currents();
        assert!(a.abs() < 1e-6, "a should be ~0: {a}");
        assert!((b - 1.0).abs() < 1e-6, "b should be ~1: {b}");
    }

    #[test]
    fn test_ramp_creation() {
        let ramp = TrapezoidalRamp::new(1000.0, 5000.0);
        assert!(ramp.is_ok());
    }

    #[test]
    fn test_ramp_invalid_params() {
        assert!(TrapezoidalRamp::new(0.0, 100.0).is_err());
        assert!(TrapezoidalRamp::new(100.0, 0.0).is_err());
    }

    #[test]
    fn test_ramp_accelerates() {
        let mut ramp = TrapezoidalRamp::new(1000.0, 2000.0).unwrap();
        let _interval = ramp.next_interval(100);
        assert!(ramp.velocity > 0.0);
    }

    #[test]
    fn test_ramp_zero_remaining() {
        let mut ramp = TrapezoidalRamp::new(1000.0, 2000.0).unwrap();
        ramp.velocity = 500.0;
        let interval = ramp.next_interval(0);
        assert!(interval.is_infinite());
    }

    #[test]
    fn test_position_tracker_step() {
        let mut tracker = PositionTracker::new(200, 8.0).unwrap();
        tracker.record_step(1).unwrap();
        assert_eq!(tracker.position, 1);
        tracker.record_step(-1).unwrap();
        assert_eq!(tracker.position, 0);
    }

    #[test]
    fn test_position_tracker_limits() {
        let mut tracker = PositionTracker::new(200, 8.0)
            .unwrap();
        let mut tracker = PositionTracker {
            min_position: -10,
            max_position: 10,
            ..tracker
        };
        tracker.position = 10;
        assert!(tracker.record_step(1).is_err());
    }

    #[test]
    fn test_position_angle() {
        let mut tracker = PositionTracker::new(200, 8.0).unwrap();
        // 200 steps = 360°.
        for _ in 0..100 {
            tracker.record_step(1).unwrap();
        }
        assert!((tracker.angle_deg() - 180.0).abs() < 1e-6);
    }

    #[test]
    fn test_position_linear() {
        let mut tracker = PositionTracker::new(200, 8.0).unwrap();
        // 200 steps = 8 mm.
        for _ in 0..200 {
            tracker.record_step(1).unwrap();
        }
        assert!((tracker.linear_position() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_position_step_distance() {
        let tracker = PositionTracker::new(3200, 8.0).unwrap();
        // 8mm / 3200 = 0.0025 mm.
        assert!((tracker.step_distance() - 0.0025).abs() < 1e-9);
    }

    #[test]
    fn test_controller_move_to() {
        let mode = StepMode::Full;
        let ramp = TrapezoidalRamp::new(1000.0, 5000.0).unwrap();
        let tracker = PositionTracker::new(200, 8.0).unwrap();
        let mut ctrl = StepperController::new(mode, ramp, tracker);

        ctrl.move_to(50);
        // Run for enough time.
        for _ in 0..10_000 {
            let _ = ctrl.update(0.001);
            if ctrl.is_at_target() {
                break;
            }
        }
        assert!(ctrl.is_at_target(), "pos={}", ctrl.tracker.position);
    }

    #[test]
    fn test_controller_disabled() {
        let mode = StepMode::Full;
        let ramp = TrapezoidalRamp::new(1000.0, 5000.0).unwrap();
        let tracker = PositionTracker::new(200, 8.0).unwrap();
        let mut ctrl = StepperController::new(mode, ramp, tracker);
        ctrl.enabled = false;
        ctrl.move_to(100);
        let steps = ctrl.update(0.01).unwrap();
        assert_eq!(steps, 0);
    }

    #[test]
    fn test_display_step_mode() {
        assert_eq!(format!("{}", StepMode::Full), "full-step");
        assert_eq!(format!("{}", StepMode::Micro(16)), "1/16 micro-step");
    }

    #[test]
    fn test_display_controller() {
        let mode = StepMode::Half;
        let ramp = TrapezoidalRamp::new(500.0, 2000.0).unwrap();
        let tracker = PositionTracker::new(400, 8.0).unwrap();
        let ctrl = StepperController::new(mode, ramp, tracker);
        let s = format!("{ctrl}");
        assert!(s.contains("Stepper"));
    }
}
