//! Servo Driver — Angle command, pulse-width mapping, trajectory smoothing,
//! multi-servo coordination, and servo group sequencing for standard hobby
//! and industrial servos.
//!
//! Pure-Rust servo control using `f64` math; no external crates.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Servo driver errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ServoError {
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Angle out of range.
    AngleOutOfRange { angle_deg: f64, min: f64, max: f64 },
    /// Servo index out of bounds.
    IndexOutOfBounds { index: usize, count: usize },
}

impl fmt::Display for ServoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::AngleOutOfRange { angle_deg, min, max } => {
                write!(f, "angle {angle_deg:.1}° outside [{min:.1}°, {max:.1}°]")
            }
            Self::IndexOutOfBounds { index, count } => {
                write!(f, "servo index {index} out of bounds (count={count})")
            }
        }
    }
}

impl std::error::Error for ServoError {}

// ── Pulse Width Mapping ────────────────────────────────────────

/// Maps angle (degrees) to pulse width (microseconds) for a servo.
///
/// Standard hobby servos: 500–2500 µs for 0°–180°.
#[derive(Debug, Clone, PartialEq)]
pub struct PulseWidthMap {
    /// Minimum angle (degrees).
    pub min_angle: f64,
    /// Maximum angle (degrees).
    pub max_angle: f64,
    /// Pulse width at min angle (µs).
    pub min_pulse_us: f64,
    /// Pulse width at max angle (µs).
    pub max_pulse_us: f64,
}

impl PulseWidthMap {
    /// Standard hobby servo mapping: 0°–180° → 500–2500 µs.
    pub fn standard() -> Self {
        Self {
            min_angle: 0.0,
            max_angle: 180.0,
            min_pulse_us: 500.0,
            max_pulse_us: 2500.0,
        }
    }

    /// Create a custom mapping.
    pub fn new(
        min_angle: f64,
        max_angle: f64,
        min_pulse_us: f64,
        max_pulse_us: f64,
    ) -> Result<Self, ServoError> {
        if (max_angle - min_angle).abs() < 1e-6 {
            return Err(ServoError::InvalidParameter(
                "angle range must be nonzero".into(),
            ));
        }
        if (max_pulse_us - min_pulse_us).abs() < 1e-6 {
            return Err(ServoError::InvalidParameter(
                "pulse range must be nonzero".into(),
            ));
        }
        Ok(Self { min_angle, max_angle, min_pulse_us, max_pulse_us })
    }

    /// Convert angle (degrees) to pulse width (µs).
    pub fn angle_to_pulse(&self, angle_deg: f64) -> f64 {
        let t = (angle_deg - self.min_angle) / (self.max_angle - self.min_angle);
        self.min_pulse_us + t * (self.max_pulse_us - self.min_pulse_us)
    }

    /// Convert pulse width (µs) to angle (degrees).
    pub fn pulse_to_angle(&self, pulse_us: f64) -> f64 {
        let t = (pulse_us - self.min_pulse_us) / (self.max_pulse_us - self.min_pulse_us);
        self.min_angle + t * (self.max_angle - self.min_angle)
    }

    /// Pulse width at center position.
    pub fn center_pulse(&self) -> f64 {
        (self.min_pulse_us + self.max_pulse_us) / 2.0
    }
}

impl fmt::Display for PulseWidthMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PulseMap([{:.0}°,{:.0}°] → [{:.0},{:.0}] µs)",
            self.min_angle, self.max_angle, self.min_pulse_us, self.max_pulse_us
        )
    }
}

// ── Trajectory Smoother ────────────────────────────────────────

/// Trajectory smoother using trapezoidal velocity profile.
///
/// Smooths angle transitions to avoid abrupt servo jumps.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySmoother {
    /// Maximum angular velocity (deg/s).
    pub max_velocity: f64,
    /// Maximum angular acceleration (deg/s²).
    pub max_acceleration: f64,
    /// Current smoothed position (deg).
    pub position: f64,
    /// Current velocity (deg/s).
    pub velocity: f64,
    /// Target position (deg).
    pub target: f64,
}

impl TrajectorySmoother {
    /// Create a trajectory smoother.
    pub fn new(max_velocity: f64, max_acceleration: f64) -> Result<Self, ServoError> {
        if max_velocity <= 0.0 {
            return Err(ServoError::InvalidParameter(
                "max velocity must be > 0".into(),
            ));
        }
        if max_acceleration <= 0.0 {
            return Err(ServoError::InvalidParameter(
                "max acceleration must be > 0".into(),
            ));
        }
        Ok(Self {
            max_velocity,
            max_acceleration,
            position: 0.0,
            velocity: 0.0,
            target: 0.0,
        })
    }

    /// Set a new target position.
    pub fn set_target(&mut self, target_deg: f64) {
        self.target = target_deg;
    }

    /// Step the smoother forward by `dt` seconds; returns current smoothed position.
    pub fn step(&mut self, dt: f64) -> f64 {
        if dt <= 0.0 {
            return self.position;
        }

        let error = self.target - self.position;
        let dist = error.abs();

        if dist < 0.001 && self.velocity.abs() < 0.001 {
            self.position = self.target;
            self.velocity = 0.0;
            return self.position;
        }

        let direction = error.signum();

        // Braking distance: v² / (2*a).
        let braking_dist = self.velocity * self.velocity / (2.0 * self.max_acceleration);

        if braking_dist >= dist && self.velocity * direction > 0.0 {
            // Decelerate.
            let decel = self.max_acceleration * (-direction);
            self.velocity += decel * dt;
        } else {
            // Accelerate toward target, capped at max velocity.
            self.velocity += self.max_acceleration * direction * dt;
            self.velocity = self.velocity.clamp(-self.max_velocity, self.max_velocity);
        }

        self.position += self.velocity * dt;

        // Overshoot guard.
        if (self.target - self.position) * direction < 0.0 {
            self.position = self.target;
            self.velocity = 0.0;
        }

        self.position
    }

    /// Returns true if the smoother has reached the target.
    pub fn is_settled(&self) -> bool {
        (self.position - self.target).abs() < 0.01 && self.velocity.abs() < 0.01
    }

    /// Estimated time to reach target (simplified triangular profile).
    pub fn estimated_time(&self) -> f64 {
        let dist = (self.target - self.position).abs();
        if dist < 0.001 {
            return 0.0;
        }
        // Triangular profile: t = 2 * sqrt(dist / a)
        // Trapezoidal: min(triangular, dist/v_max + v_max/a)
        let t_tri = 2.0 * (dist / self.max_acceleration).sqrt();
        let t_trap = dist / self.max_velocity + self.max_velocity / self.max_acceleration;
        t_tri.min(t_trap)
    }
}

impl fmt::Display for TrajectorySmoother {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrajSmooth(pos={:.2}°, vel={:.2}°/s, target={:.2}°)",
            self.position, self.velocity, self.target
        )
    }
}

// ── Single Servo ───────────────────────────────────────────────

/// A single servo with angle limits, pulse mapping, and trajectory smoothing.
#[derive(Debug, Clone)]
pub struct Servo {
    /// Servo name / label.
    pub name: String,
    /// Pulse width mapping.
    pub pulse_map: PulseWidthMap,
    /// Trajectory smoother.
    pub smoother: TrajectorySmoother,
    /// Current commanded angle (deg).
    pub commanded_angle: f64,
    /// Trim offset (deg) for calibration.
    pub trim_offset: f64,
    /// Whether servo is enabled.
    pub enabled: bool,
}

impl Servo {
    /// Create a servo with standard hobby parameters.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            pulse_map: PulseWidthMap::standard(),
            smoother: TrajectorySmoother::new(300.0, 600.0).unwrap(),
            commanded_angle: 90.0,
            trim_offset: 0.0,
            enabled: true,
        }
    }

    /// Builder: set pulse mapping.
    pub fn with_pulse_map(mut self, map: PulseWidthMap) -> Self {
        self.pulse_map = map;
        self
    }

    /// Builder: set trajectory smoother.
    pub fn with_smoother(mut self, smoother: TrajectorySmoother) -> Self {
        self.smoother = smoother;
        self
    }

    /// Builder: set trim offset.
    pub fn with_trim(mut self, offset_deg: f64) -> Self {
        self.trim_offset = offset_deg;
        self
    }

    /// Command the servo to an angle (degrees).
    pub fn set_angle(&mut self, angle_deg: f64) -> Result<(), ServoError> {
        let min = self.pulse_map.min_angle;
        let max = self.pulse_map.max_angle;
        if angle_deg < min || angle_deg > max {
            return Err(ServoError::AngleOutOfRange {
                angle_deg,
                min,
                max,
            });
        }
        self.commanded_angle = angle_deg;
        self.smoother.set_target(angle_deg + self.trim_offset);
        Ok(())
    }

    /// Step the servo smoother; returns the pulse width in µs.
    pub fn update(&mut self, dt: f64) -> f64 {
        if !self.enabled {
            return 0.0;
        }
        let smoothed_angle = self.smoother.step(dt);
        self.pulse_map.angle_to_pulse(smoothed_angle)
    }

    /// Current smoothed position (deg).
    pub fn current_angle(&self) -> f64 {
        self.smoother.position - self.trim_offset
    }

    /// True if the servo has reached its target.
    pub fn is_settled(&self) -> bool {
        self.smoother.is_settled()
    }
}

impl fmt::Display for Servo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Servo('{}', cmd={:.1}°, pos={:.1}°, {})",
            self.name,
            self.commanded_angle,
            self.current_angle(),
            if self.enabled { "ON" } else { "OFF" }
        )
    }
}

// ── Multi-Servo Coordinator ────────────────────────────────────

/// Coordinates multiple servos with synchronized motion.
#[derive(Debug, Clone)]
pub struct ServoGroup {
    /// Named servos in this group.
    pub servos: Vec<Servo>,
    /// Whether to synchronize (all finish at same time).
    pub synchronized: bool,
}

impl ServoGroup {
    /// Create a new empty servo group.
    pub fn new() -> Self {
        Self {
            servos: Vec::new(),
            synchronized: false,
        }
    }

    /// Builder: enable synchronized motion.
    pub fn with_synchronization(mut self, sync: bool) -> Self {
        self.synchronized = sync;
        self
    }

    /// Add a servo to the group.
    pub fn add_servo(&mut self, servo: Servo) {
        self.servos.push(servo);
    }

    /// Number of servos.
    pub fn len(&self) -> usize {
        self.servos.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.servos.is_empty()
    }

    /// Set angle for a servo by index.
    pub fn set_angle(&mut self, index: usize, angle_deg: f64) -> Result<(), ServoError> {
        if index >= self.servos.len() {
            return Err(ServoError::IndexOutOfBounds {
                index,
                count: self.servos.len(),
            });
        }
        self.servos[index].set_angle(angle_deg)?;

        if self.synchronized {
            self.synchronize_velocities();
        }
        Ok(())
    }

    /// Set angles for all servos at once.
    pub fn set_angles(&mut self, angles: &[f64]) -> Result<(), ServoError> {
        let count = angles.len().min(self.servos.len());
        for i in 0..count {
            self.servos[i].set_angle(angles[i])?;
        }
        if self.synchronized {
            self.synchronize_velocities();
        }
        Ok(())
    }

    /// Adjust max velocities so all servos finish at approximately the same time.
    fn synchronize_velocities(&mut self) {
        if self.servos.is_empty() {
            return;
        }

        // Find the servo with the longest estimated travel time.
        let max_time = self
            .servos
            .iter()
            .map(|s| s.smoother.estimated_time())
            .fold(0.0_f64, f64::max);

        if max_time < 0.001 {
            return;
        }

        // Scale each servo's max velocity to match the longest travel time.
        for servo in &mut self.servos {
            let dist = (servo.smoother.target - servo.smoother.position).abs();
            if dist > 0.001 {
                // Desired avg velocity to finish in max_time.
                let desired_v = dist / max_time * 1.5; // 1.5× for trapezoidal profile
                servo.smoother.max_velocity = desired_v;
            }
        }
    }

    /// Step all servos; returns pulse widths.
    pub fn update(&mut self, dt: f64) -> Vec<f64> {
        self.servos.iter_mut().map(|s| s.update(dt)).collect()
    }

    /// True if all servos have settled.
    pub fn all_settled(&self) -> bool {
        self.servos.iter().all(|s| s.is_settled())
    }

    /// Get servo by name.
    pub fn find_by_name(&self, name: &str) -> Option<&Servo> {
        self.servos.iter().find(|s| s.name == name)
    }

    /// Get mutable servo by name.
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut Servo> {
        self.servos.iter_mut().find(|s| s.name == name)
    }
}

impl fmt::Display for ServoGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ServoGroup({} servos, sync={}, settled={})",
            self.servos.len(),
            self.synchronized,
            self.all_settled()
        )
    }
}

// ── Keyframe Sequencer ─────────────────────────────────────────

/// A keyframe for a servo group: target angles at a given timestamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Keyframe {
    /// Time offset from sequence start (seconds).
    pub time: f64,
    /// Target angles (one per servo).
    pub angles: Vec<f64>,
}

/// Sequencer that plays keyframe animations across a servo group.
#[derive(Debug, Clone)]
pub struct ServoSequencer {
    /// Ordered keyframes.
    pub keyframes: Vec<Keyframe>,
    /// Current playback time.
    pub current_time: f64,
    /// Current keyframe index.
    pub current_index: usize,
    /// Whether to loop.
    pub looping: bool,
    /// Total duration.
    pub duration: f64,
}

impl ServoSequencer {
    /// Create a new empty sequencer.
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            current_time: 0.0,
            current_index: 0,
            looping: false,
            duration: 0.0,
        }
    }

    /// Builder: enable looping.
    pub fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Add a keyframe (automatically sorted by time).
    pub fn add_keyframe(&mut self, kf: Keyframe) {
        self.keyframes.push(kf);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        self.duration = self
            .keyframes
            .last()
            .map(|k| k.time)
            .unwrap_or(0.0);
    }

    /// Interpolate angles at the current time.
    pub fn sample(&self) -> Vec<f64> {
        if self.keyframes.is_empty() {
            return Vec::new();
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].angles.clone();
        }

        let t = if self.looping && self.duration > 0.0 {
            self.current_time % self.duration
        } else {
            self.current_time
        };

        // Find surrounding keyframes.
        let mut lo = 0;
        for i in 0..self.keyframes.len() - 1 {
            if t >= self.keyframes[i].time && t <= self.keyframes[i + 1].time {
                lo = i;
                break;
            }
            if i == self.keyframes.len() - 2 {
                lo = i;
            }
        }
        let hi = (lo + 1).min(self.keyframes.len() - 1);

        let kf_lo = &self.keyframes[lo];
        let kf_hi = &self.keyframes[hi];
        let seg_dur = kf_hi.time - kf_lo.time;

        if seg_dur < 1e-9 {
            return kf_lo.angles.clone();
        }

        let alpha = ((t - kf_lo.time) / seg_dur).clamp(0.0, 1.0);
        let count = kf_lo.angles.len().min(kf_hi.angles.len());
        (0..count)
            .map(|i| kf_lo.angles[i] + alpha * (kf_hi.angles[i] - kf_lo.angles[i]))
            .collect()
    }

    /// Advance time and return interpolated angles.
    pub fn advance(&mut self, dt: f64) -> Vec<f64> {
        self.current_time += dt;
        self.sample()
    }

    /// Whether playback has finished (non-looping).
    pub fn is_finished(&self) -> bool {
        !self.looping && self.current_time >= self.duration
    }

    /// Reset to start.
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        self.current_index = 0;
    }
}

impl fmt::Display for ServoSequencer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Sequencer({} keyframes, t={:.2}/{:.2}s, loop={})",
            self.keyframes.len(),
            self.current_time,
            self.duration,
            self.looping
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_map_standard_center() {
        let map = PulseWidthMap::standard();
        assert!((map.center_pulse() - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_pulse_map_angle_to_pulse() {
        let map = PulseWidthMap::standard();
        // 0° → 500 µs, 180° → 2500 µs, 90° → 1500 µs.
        assert!((map.angle_to_pulse(0.0) - 500.0).abs() < 1e-6);
        assert!((map.angle_to_pulse(180.0) - 2500.0).abs() < 1e-6);
        assert!((map.angle_to_pulse(90.0) - 1500.0).abs() < 1e-6);
    }

    #[test]
    fn test_pulse_map_roundtrip() {
        let map = PulseWidthMap::standard();
        let angle = 45.0;
        let pulse = map.angle_to_pulse(angle);
        let back = map.pulse_to_angle(pulse);
        assert!((back - angle).abs() < 1e-9);
    }

    #[test]
    fn test_trajectory_smoother_settles() {
        let mut sm = TrajectorySmoother::new(300.0, 600.0).unwrap();
        sm.set_target(90.0);
        for _ in 0..5000 {
            sm.step(0.001);
        }
        assert!(sm.is_settled(), "pos={}, vel={}", sm.position, sm.velocity);
    }

    #[test]
    fn test_trajectory_smoother_zero_dt() {
        let mut sm = TrajectorySmoother::new(300.0, 600.0).unwrap();
        sm.set_target(45.0);
        let pos = sm.step(0.0);
        assert!((pos - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_trajectory_estimated_time() {
        let mut sm = TrajectorySmoother::new(100.0, 200.0).unwrap();
        sm.set_target(50.0);
        let t = sm.estimated_time();
        assert!(t > 0.0);
    }

    #[test]
    fn test_servo_creation() {
        let s = Servo::new("shoulder");
        assert_eq!(s.name, "shoulder");
        assert!(s.enabled);
    }

    #[test]
    fn test_servo_angle_range() {
        let mut s = Servo::new("test");
        assert!(s.set_angle(90.0).is_ok());
        assert!(s.set_angle(-10.0).is_err());
        assert!(s.set_angle(200.0).is_err());
    }

    #[test]
    fn test_servo_trim() {
        let mut s = Servo::new("test").with_trim(5.0);
        s.set_angle(90.0).unwrap();
        // Target should be 90+5=95
        assert!((s.smoother.target - 95.0).abs() < 1e-9);
    }

    #[test]
    fn test_servo_disabled_returns_zero_pulse() {
        let mut s = Servo::new("test");
        s.enabled = false;
        let pw = s.update(0.001);
        assert!((pw).abs() < 1e-9);
    }

    #[test]
    fn test_servo_group_add() {
        let mut group = ServoGroup::new();
        group.add_servo(Servo::new("s0"));
        group.add_servo(Servo::new("s1"));
        assert_eq!(group.len(), 2);
    }

    #[test]
    fn test_servo_group_set_angle_oob() {
        let mut group = ServoGroup::new();
        group.add_servo(Servo::new("s0"));
        assert!(group.set_angle(5, 90.0).is_err());
    }

    #[test]
    fn test_servo_group_find_by_name() {
        let mut group = ServoGroup::new();
        group.add_servo(Servo::new("elbow"));
        assert!(group.find_by_name("elbow").is_some());
        assert!(group.find_by_name("knee").is_none());
    }

    #[test]
    fn test_servo_group_update() {
        let mut group = ServoGroup::new();
        group.add_servo(Servo::new("s0"));
        group.add_servo(Servo::new("s1"));
        let pulses = group.update(0.001);
        assert_eq!(pulses.len(), 2);
    }

    #[test]
    fn test_sequencer_single_keyframe() {
        let mut seq = ServoSequencer::new();
        seq.add_keyframe(Keyframe {
            time: 0.0,
            angles: vec![45.0, 90.0],
        });
        let angles = seq.sample();
        assert_eq!(angles.len(), 2);
        assert!((angles[0] - 45.0).abs() < 1e-9);
    }

    #[test]
    fn test_sequencer_interpolation() {
        let mut seq = ServoSequencer::new();
        seq.add_keyframe(Keyframe { time: 0.0, angles: vec![0.0] });
        seq.add_keyframe(Keyframe { time: 1.0, angles: vec![100.0] });
        seq.current_time = 0.5;
        let angles = seq.sample();
        assert!((angles[0] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_sequencer_advance() {
        let mut seq = ServoSequencer::new();
        seq.add_keyframe(Keyframe { time: 0.0, angles: vec![0.0] });
        seq.add_keyframe(Keyframe { time: 2.0, angles: vec![180.0] });
        let angles = seq.advance(1.0);
        assert!((angles[0] - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_sequencer_finished() {
        let mut seq = ServoSequencer::new();
        seq.add_keyframe(Keyframe { time: 0.0, angles: vec![0.0] });
        seq.add_keyframe(Keyframe { time: 1.0, angles: vec![90.0] });
        assert!(!seq.is_finished());
        seq.current_time = 2.0;
        assert!(seq.is_finished());
    }

    #[test]
    fn test_display_servo() {
        let s = Servo::new("pan");
        let txt = format!("{s}");
        assert!(txt.contains("pan"));
    }

    #[test]
    fn test_display_group() {
        let group = ServoGroup::new();
        let txt = format!("{group}");
        assert!(txt.contains("ServoGroup"));
    }
}
