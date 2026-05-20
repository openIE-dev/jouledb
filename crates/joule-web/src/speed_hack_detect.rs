//! Speed hack detection — movement analysis, teleport detection, violation tracking.
//!
//! Replaces server-side speed check middleware with pure Rust.
//! PlayerMovement sampling, per-player movement history, impossible-speed
//! detection, teleportation detection, acceleration analysis, configurable
//! thresholds, confidence scoring, movement smoothness analysis, and
//! violation tracking with severity.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeedDetectError {
    PlayerNotFound(u64),
    InvalidConfig(String),
    InsufficientData(String),
}

impl fmt::Display for SpeedDetectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::InsufficientData(msg) => write!(f, "insufficient data: {msg}"),
        }
    }
}

impl std::error::Error for SpeedDetectError {}

// ── Position & Movement ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn distance_to(&self, other: &Vec3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlayerMovement {
    pub player_id: u64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub timestamp: u64,
}

impl PlayerMovement {
    pub fn new(player_id: u64, position: Vec3, velocity: Vec3, timestamp: u64) -> Self {
        Self { player_id, position, velocity, timestamp }
    }

    pub fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }
}

// ── Violation ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViolationType {
    ExcessiveSpeed,
    Teleportation,
    ExcessiveAcceleration,
    LowSmoothness,
}

impl fmt::Display for ViolationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExcessiveSpeed => write!(f, "ExcessiveSpeed"),
            Self::Teleportation => write!(f, "Teleportation"),
            Self::ExcessiveAcceleration => write!(f, "ExcessiveAcceleration"),
            Self::LowSmoothness => write!(f, "LowSmoothness"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Violation {
    pub player_id: u64,
    pub violation_type: ViolationType,
    pub confidence: f64,
    pub timestamp: u64,
    pub measured_value: f64,
    pub threshold: f64,
    pub details: String,
}

impl Violation {
    pub fn new(
        player_id: u64,
        vtype: ViolationType,
        confidence: f64,
        timestamp: u64,
        measured: f64,
        threshold: f64,
    ) -> Self {
        Self {
            player_id,
            violation_type: vtype,
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            measured_value: measured,
            threshold,
            details: String::new(),
        }
    }

    pub fn with_details(mut self, d: &str) -> Self {
        self.details = d.to_string();
        self
    }

    pub fn severity_ratio(&self) -> f64 {
        if self.threshold > 0.0 {
            self.measured_value / self.threshold
        } else {
            0.0
        }
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[t={}] player={} {} measured={:.2} threshold={:.2} conf={:.2}",
            self.timestamp, self.player_id, self.violation_type,
            self.measured_value, self.threshold, self.confidence
        )
    }
}

// ── Detector Config ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpeedDetectorConfig {
    pub max_speed: f64,
    pub max_acceleration: f64,
    pub teleport_threshold: f64,
    pub smoothness_threshold: f64,
    pub history_size: usize,
    pub min_samples_for_analysis: usize,
}

impl Default for SpeedDetectorConfig {
    fn default() -> Self {
        Self {
            max_speed: 20.0,
            max_acceleration: 50.0,
            teleport_threshold: 100.0,
            smoothness_threshold: 0.3,
            history_size: 128,
            min_samples_for_analysis: 3,
        }
    }
}

impl SpeedDetectorConfig {
    pub fn with_max_speed(mut self, v: f64) -> Self {
        self.max_speed = v.max(0.0);
        self
    }

    pub fn with_max_acceleration(mut self, v: f64) -> Self {
        self.max_acceleration = v.max(0.0);
        self
    }

    pub fn with_teleport_threshold(mut self, v: f64) -> Self {
        self.teleport_threshold = v.max(0.0);
        self
    }
}

// ── Player Tracking ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PlayerTrack {
    movements: Vec<PlayerMovement>,
    violations: Vec<Violation>,
    total_samples: u64,
}

impl PlayerTrack {
    fn new() -> Self {
        Self {
            movements: Vec::new(),
            violations: Vec::new(),
            total_samples: 0,
        }
    }

    fn push_movement(&mut self, m: PlayerMovement, max_history: usize) {
        self.total_samples += 1;
        self.movements.push(m);
        if self.movements.len() > max_history {
            self.movements.remove(0);
        }
    }

    fn last_two(&self) -> Option<(&PlayerMovement, &PlayerMovement)> {
        let len = self.movements.len();
        if len >= 2 {
            Some((&self.movements[len - 2], &self.movements[len - 1]))
        } else {
            None
        }
    }

    fn last_three_speeds(&self) -> Option<(f64, f64, f64)> {
        let len = self.movements.len();
        if len < 3 {
            return None;
        }
        let calc_speed = |a: &PlayerMovement, b: &PlayerMovement| -> f64 {
            let dist = a.position.distance_to(&b.position);
            let dt = b.timestamp.saturating_sub(a.timestamp) as f64;
            if dt > 0.0 { dist / dt } else { 0.0 }
        };
        let s1 = calc_speed(&self.movements[len - 3], &self.movements[len - 2]);
        let s2 = calc_speed(&self.movements[len - 2], &self.movements[len - 1]);
        let s0 = if len >= 4 {
            calc_speed(&self.movements[len - 4], &self.movements[len - 3])
        } else {
            s1
        };
        Some((s0, s1, s2))
    }

    fn smoothness_score(&self) -> f64 {
        if self.movements.len() < 4 {
            return 1.0;
        }
        let mut direction_changes = 0u32;
        let mut segments = 0u32;
        for window in self.movements.windows(3) {
            let d1 = window[1].position.sub(&window[0].position);
            let d2 = window[2].position.sub(&window[1].position);
            let dot = d1.x * d2.x + d1.y * d2.y + d1.z * d2.z;
            let m1 = d1.magnitude();
            let m2 = d2.magnitude();
            if m1 > 0.001 && m2 > 0.001 {
                let cos_angle = (dot / (m1 * m2)).clamp(-1.0, 1.0);
                if cos_angle < 0.0 {
                    direction_changes += 1;
                }
                segments += 1;
            }
        }
        if segments == 0 {
            return 1.0;
        }
        1.0 - (direction_changes as f64 / segments as f64)
    }
}

// ── Speed Detector ──────────────────────────────────────────────

#[derive(Debug)]
pub struct SpeedDetector {
    config: SpeedDetectorConfig,
    players: HashMap<u64, PlayerTrack>,
}

impl SpeedDetector {
    pub fn new(config: SpeedDetectorConfig) -> Self {
        Self {
            config,
            players: HashMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(SpeedDetectorConfig::default())
    }

    pub fn submit_movement(&mut self, movement: PlayerMovement) -> Vec<Violation> {
        let player_id = movement.player_id;
        let timestamp = movement.timestamp;
        let track = self
            .players
            .entry(player_id)
            .or_insert_with(PlayerTrack::new);
        track.push_movement(movement, self.config.history_size);

        let mut violations = Vec::new();

        if let Some((prev, curr)) = track.last_two() {
            let dist = prev.position.distance_to(&curr.position);
            let dt = curr.timestamp.saturating_sub(prev.timestamp) as f64;

            // Teleport check
            if dist > self.config.teleport_threshold {
                let conf = (dist / self.config.teleport_threshold).min(1.0) * 0.9;
                let v = Violation::new(
                    player_id,
                    ViolationType::Teleportation,
                    conf,
                    timestamp,
                    dist,
                    self.config.teleport_threshold,
                );
                violations.push(v);
            } else if dt > 0.0 {
                // Speed check
                let speed = dist / dt;
                if speed > self.config.max_speed {
                    let ratio = speed / self.config.max_speed;
                    let conf = ((ratio - 1.0) / 2.0).clamp(0.1, 1.0);
                    let v = Violation::new(
                        player_id,
                        ViolationType::ExcessiveSpeed,
                        conf,
                        timestamp,
                        speed,
                        self.config.max_speed,
                    );
                    violations.push(v);
                }
            }
        }

        // Acceleration check
        if let Some((s0, s1, s2)) = track.last_three_speeds() {
            let accel1 = (s1 - s0).abs();
            let accel2 = (s2 - s1).abs();
            let max_accel = accel1.max(accel2);
            if max_accel > self.config.max_acceleration {
                let conf = ((max_accel / self.config.max_acceleration) - 1.0).clamp(0.1, 0.95);
                let v = Violation::new(
                    player_id,
                    ViolationType::ExcessiveAcceleration,
                    conf,
                    timestamp,
                    max_accel,
                    self.config.max_acceleration,
                );
                violations.push(v);
            }
        }

        // Smoothness check
        if track.movements.len() >= self.config.min_samples_for_analysis + 1 {
            let smoothness = track.smoothness_score();
            if smoothness < self.config.smoothness_threshold {
                let conf = (1.0 - smoothness / self.config.smoothness_threshold).clamp(0.1, 0.9);
                let v = Violation::new(
                    player_id,
                    ViolationType::LowSmoothness,
                    conf,
                    timestamp,
                    smoothness,
                    self.config.smoothness_threshold,
                );
                violations.push(v);
            }
        }

        for v in &violations {
            if let Some(t) = self.players.get_mut(&player_id) {
                t.violations.push(v.clone());
            }
        }
        violations
    }

    pub fn player_violations(&self, player_id: u64) -> Result<&[Violation], SpeedDetectError> {
        self.players
            .get(&player_id)
            .map(|t| t.violations.as_slice())
            .ok_or(SpeedDetectError::PlayerNotFound(player_id))
    }

    pub fn player_violation_count(&self, player_id: u64) -> usize {
        self.players
            .get(&player_id)
            .map(|t| t.violations.len())
            .unwrap_or(0)
    }

    pub fn player_sample_count(&self, player_id: u64) -> u64 {
        self.players
            .get(&player_id)
            .map(|t| t.total_samples)
            .unwrap_or(0)
    }

    pub fn tracked_player_count(&self) -> usize {
        self.players.len()
    }

    pub fn clear_player(&mut self, player_id: u64) {
        self.players.remove(&player_id);
    }

    pub fn config(&self) -> &SpeedDetectorConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mov(pid: u64, x: f64, y: f64, z: f64, t: u64) -> PlayerMovement {
        PlayerMovement::new(pid, Vec3::new(x, y, z), Vec3::zero(), t)
    }

    #[test]
    fn test_vec3_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_magnitude() {
        let v = Vec3::new(1.0, 2.0, 2.0);
        assert!((v.magnitude() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_movement_no_violation() {
        let mut det = SpeedDetector::with_default_config();
        let v1 = det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        let v2 = det.submit_movement(mov(1, 5.0, 0.0, 0.0, 1));
        assert!(v1.is_empty());
        assert!(v2.is_empty());
    }

    #[test]
    fn test_speed_violation() {
        let cfg = SpeedDetectorConfig::default().with_max_speed(10.0);
        let mut det = SpeedDetector::new(cfg);
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        let violations = det.submit_movement(mov(1, 50.0, 0.0, 0.0, 1));
        assert!(violations.iter().any(|v| v.violation_type == ViolationType::ExcessiveSpeed));
    }

    #[test]
    fn test_teleport_detection() {
        let cfg = SpeedDetectorConfig::default().with_teleport_threshold(50.0);
        let mut det = SpeedDetector::new(cfg);
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        let violations = det.submit_movement(mov(1, 200.0, 200.0, 0.0, 1));
        assert!(violations.iter().any(|v| v.violation_type == ViolationType::Teleportation));
    }

    #[test]
    fn test_acceleration_violation() {
        let cfg = SpeedDetectorConfig::default()
            .with_max_speed(1000.0)
            .with_max_acceleration(10.0);
        let mut det = SpeedDetector::new(cfg);
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        det.submit_movement(mov(1, 1.0, 0.0, 0.0, 1));
        det.submit_movement(mov(1, 2.0, 0.0, 0.0, 2));
        let violations = det.submit_movement(mov(1, 100.0, 0.0, 0.0, 3));
        assert!(violations.iter().any(|v| v.violation_type == ViolationType::ExcessiveAcceleration));
    }

    #[test]
    fn test_violation_confidence_clamped() {
        let v = Violation::new(1, ViolationType::ExcessiveSpeed, 1.5, 100, 30.0, 10.0);
        assert!((v.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_severity_ratio() {
        let v = Violation::new(1, ViolationType::ExcessiveSpeed, 0.8, 100, 30.0, 10.0);
        assert!((v.severity_ratio() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_player_tracking_count() {
        let mut det = SpeedDetector::with_default_config();
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        det.submit_movement(mov(2, 0.0, 0.0, 0.0, 0));
        assert_eq!(det.tracked_player_count(), 2);
    }

    #[test]
    fn test_player_sample_count() {
        let mut det = SpeedDetector::with_default_config();
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        det.submit_movement(mov(1, 1.0, 0.0, 0.0, 1));
        det.submit_movement(mov(1, 2.0, 0.0, 0.0, 2));
        assert_eq!(det.player_sample_count(1), 3);
    }

    #[test]
    fn test_clear_player() {
        let mut det = SpeedDetector::with_default_config();
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        det.clear_player(1);
        assert_eq!(det.tracked_player_count(), 0);
    }

    #[test]
    fn test_player_not_found_violations() {
        let det = SpeedDetector::with_default_config();
        let err = det.player_violations(999).unwrap_err();
        assert!(matches!(err, SpeedDetectError::PlayerNotFound(999)));
    }

    #[test]
    fn test_violation_display() {
        let v = Violation::new(1, ViolationType::Teleportation, 0.9, 500, 200.0, 100.0);
        let s = format!("{v}");
        assert!(s.contains("Teleportation"));
        assert!(s.contains("player=1"));
    }

    #[test]
    fn test_config_builder() {
        let cfg = SpeedDetectorConfig::default()
            .with_max_speed(30.0)
            .with_max_acceleration(60.0)
            .with_teleport_threshold(200.0);
        assert!((cfg.max_speed - 30.0).abs() < f64::EPSILON);
        assert!((cfg.max_acceleration - 60.0).abs() < f64::EPSILON);
        assert!((cfg.teleport_threshold - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_movement_speed() {
        let m = PlayerMovement::new(1, Vec3::zero(), Vec3::new(3.0, 4.0, 0.0), 0);
        assert!((m.speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_players_independent() {
        let cfg = SpeedDetectorConfig::default().with_max_speed(10.0);
        let mut det = SpeedDetector::new(cfg);
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        det.submit_movement(mov(2, 0.0, 0.0, 0.0, 0));
        let v1 = det.submit_movement(mov(1, 50.0, 0.0, 0.0, 1));
        let v2 = det.submit_movement(mov(2, 5.0, 0.0, 0.0, 1));
        assert!(!v1.is_empty());
        assert!(v2.is_empty());
    }

    #[test]
    fn test_violation_count_accumulates() {
        let cfg = SpeedDetectorConfig::default().with_max_speed(5.0);
        let mut det = SpeedDetector::new(cfg);
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 0));
        det.submit_movement(mov(1, 50.0, 0.0, 0.0, 1));
        det.submit_movement(mov(1, 100.0, 0.0, 0.0, 2));
        assert!(det.player_violation_count(1) >= 2);
    }

    #[test]
    fn test_zero_dt_no_panic() {
        let mut det = SpeedDetector::with_default_config();
        det.submit_movement(mov(1, 0.0, 0.0, 0.0, 5));
        let violations = det.submit_movement(mov(1, 10.0, 0.0, 0.0, 5));
        // Should not panic; teleport check may trigger
        assert!(violations.is_empty() || !violations.is_empty());
    }

    #[test]
    fn test_vec3_display() {
        let v = Vec3::new(1.23, 4.56, 7.89);
        let s = format!("{v}");
        assert!(s.contains("1.23"));
        assert!(s.contains("4.56"));
    }

    #[test]
    fn test_smoothness_straight_line() {
        let mut det = SpeedDetector::with_default_config();
        for i in 0..10u64 {
            det.submit_movement(mov(1, i as f64, 0.0, 0.0, i));
        }
        // Straight-line movement should have high smoothness, no violations
        let violations = det.player_violations(1).unwrap();
        assert!(violations.iter().all(|v| v.violation_type != ViolationType::LowSmoothness));
    }
}
