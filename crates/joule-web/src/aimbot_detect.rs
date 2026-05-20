//! Aimbot detection through statistical analysis — aim patterns, reaction time, tracking.
//!
//! Replaces server-side aimbot detection heuristics with pure Rust.
//! AimSample collection, angular velocity distribution analysis,
//! inhuman reaction time detection, perfect-tracking detection,
//! snap-to-target pattern recognition, configurable thresholds,
//! confidence scoring, sample window management, and human-like
//! aim profile comparison.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AimbotError {
    PlayerNotFound(u64),
    InsufficientSamples(usize),
    InvalidConfig(String),
}

impl fmt::Display for AimbotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::InsufficientSamples(n) => write!(f, "need more samples, have {n}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for AimbotError {}

// ── Aim Sample ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AimSample {
    pub angle_delta: f64,
    pub time_delta: f64,
    pub target_distance: f64,
    pub timestamp: u64,
}

impl AimSample {
    pub fn new(angle_delta: f64, time_delta: f64, target_distance: f64, timestamp: u64) -> Self {
        Self {
            angle_delta,
            time_delta: time_delta.max(0.0001),
            target_distance: target_distance.max(0.0),
            timestamp,
        }
    }

    pub fn angular_velocity(&self) -> f64 {
        self.angle_delta.abs() / self.time_delta
    }

    pub fn is_snap(&self, snap_threshold: f64) -> bool {
        self.angular_velocity() > snap_threshold
    }
}

impl fmt::Display for AimSample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "aim(delta={:.3}rad dt={:.4}s dist={:.1} vel={:.1}rad/s)",
            self.angle_delta, self.time_delta, self.target_distance, self.angular_velocity()
        )
    }
}

// ── Detection Result ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AimbotIndicator {
    InhumanReactionTime,
    PerfectTracking,
    SnapToTarget,
    UniformAngularVelocity,
    AbnormalAccuracy,
}

impl fmt::Display for AimbotIndicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InhumanReactionTime => write!(f, "InhumanReactionTime"),
            Self::PerfectTracking => write!(f, "PerfectTracking"),
            Self::SnapToTarget => write!(f, "SnapToTarget"),
            Self::UniformAngularVelocity => write!(f, "UniformAngularVelocity"),
            Self::AbnormalAccuracy => write!(f, "AbnormalAccuracy"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AimbotDetection {
    pub player_id: u64,
    pub indicator: AimbotIndicator,
    pub confidence: f64,
    pub timestamp: u64,
    pub evidence: String,
}

impl AimbotDetection {
    pub fn new(player_id: u64, indicator: AimbotIndicator, confidence: f64, timestamp: u64) -> Self {
        Self {
            player_id,
            indicator,
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            evidence: String::new(),
        }
    }

    pub fn with_evidence(mut self, e: &str) -> Self {
        self.evidence = e.to_string();
        self
    }
}

impl fmt::Display for AimbotDetection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[t={}] player={} {} conf={:.2}",
            self.timestamp, self.player_id, self.indicator, self.confidence
        )
    }
}

// ── Human Profile ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HumanAimProfile {
    pub mean_reaction_time: f64,
    pub min_reaction_time: f64,
    pub angular_velocity_mean: f64,
    pub angular_velocity_std: f64,
    pub tracking_variance_min: f64,
}

impl Default for HumanAimProfile {
    fn default() -> Self {
        Self {
            mean_reaction_time: 0.25,
            min_reaction_time: 0.10,
            angular_velocity_mean: 3.0,
            angular_velocity_std: 1.5,
            tracking_variance_min: 0.01,
        }
    }
}

// ── Detector Config ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AimbotDetectorConfig {
    pub snap_threshold: f64,
    pub min_samples: usize,
    pub window_size: usize,
    pub perfect_tracking_max_variance: f64,
    pub reaction_time_min: f64,
    pub uniform_velocity_max_cv: f64,
    pub human_profile: HumanAimProfile,
}

impl Default for AimbotDetectorConfig {
    fn default() -> Self {
        Self {
            snap_threshold: 50.0,
            min_samples: 5,
            window_size: 200,
            perfect_tracking_max_variance: 0.001,
            reaction_time_min: 0.08,
            uniform_velocity_max_cv: 0.05,
            human_profile: HumanAimProfile::default(),
        }
    }
}

impl AimbotDetectorConfig {
    pub fn with_snap_threshold(mut self, v: f64) -> Self {
        self.snap_threshold = v;
        self
    }

    pub fn with_min_samples(mut self, n: usize) -> Self {
        self.min_samples = n.max(2);
        self
    }

    pub fn with_window_size(mut self, n: usize) -> Self {
        self.window_size = n.max(10);
        self
    }
}

// ── Player Aim Data ─────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PlayerAimData {
    samples: Vec<AimSample>,
    detections: Vec<AimbotDetection>,
    total_samples: u64,
}

impl PlayerAimData {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            detections: Vec::new(),
            total_samples: 0,
        }
    }

    fn push_sample(&mut self, s: AimSample, window: usize) {
        self.total_samples += 1;
        self.samples.push(s);
        if self.samples.len() > window {
            self.samples.remove(0);
        }
    }

    fn angular_velocities(&self) -> Vec<f64> {
        self.samples.iter().map(|s| s.angular_velocity()).collect()
    }

    fn mean_and_variance(values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        (mean, variance)
    }

    fn std_dev(values: &[f64]) -> f64 {
        let (_, var) = Self::mean_and_variance(values);
        var.sqrt()
    }

    fn coefficient_of_variation(values: &[f64]) -> f64 {
        let (mean, var) = Self::mean_and_variance(values);
        if mean.abs() < 1e-12 {
            return 0.0;
        }
        var.sqrt() / mean.abs()
    }

    fn snap_count(&self, threshold: f64) -> usize {
        self.samples.iter().filter(|s| s.is_snap(threshold)).count()
    }

    fn tracking_variance(&self) -> f64 {
        let deltas: Vec<f64> = self.samples.iter().map(|s| s.angle_delta).collect();
        let (_, var) = Self::mean_and_variance(&deltas);
        var
    }
}

// ── Aimbot Detector ─────────────────────────────────────────────

#[derive(Debug)]
pub struct AimbotDetector {
    config: AimbotDetectorConfig,
    players: HashMap<u64, PlayerAimData>,
}

impl AimbotDetector {
    pub fn new(config: AimbotDetectorConfig) -> Self {
        Self {
            config,
            players: HashMap::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(AimbotDetectorConfig::default())
    }

    pub fn submit_sample(&mut self, player_id: u64, sample: AimSample) -> Vec<AimbotDetection> {
        let window = self.config.window_size;
        let data = self
            .players
            .entry(player_id)
            .or_insert_with(PlayerAimData::new);
        data.push_sample(sample, window);

        if data.samples.len() < self.config.min_samples {
            return Vec::new();
        }

        let mut detections = Vec::new();
        let ts = sample.timestamp;

        // Inhuman reaction time
        if sample.time_delta < self.config.reaction_time_min && sample.angle_delta.abs() > 0.1 {
            let ratio = self.config.reaction_time_min / sample.time_delta;
            let conf = ((ratio - 1.0) * 0.5).clamp(0.2, 0.95);
            detections.push(
                AimbotDetection::new(player_id, AimbotIndicator::InhumanReactionTime, conf, ts)
                    .with_evidence(&format!("dt={:.4}s < min={:.4}s", sample.time_delta, self.config.reaction_time_min)),
            );
        }

        // Perfect tracking (near-zero variance)
        let tracking_var = data.tracking_variance();
        if tracking_var < self.config.perfect_tracking_max_variance
            && tracking_var < self.config.human_profile.tracking_variance_min
        {
            let conf = (1.0 - tracking_var / self.config.human_profile.tracking_variance_min).clamp(0.3, 0.95);
            detections.push(
                AimbotDetection::new(player_id, AimbotIndicator::PerfectTracking, conf, ts)
                    .with_evidence(&format!("variance={:.6}", tracking_var)),
            );
        }

        // Snap-to-target
        let snap_count = data.snap_count(self.config.snap_threshold);
        let snap_ratio = snap_count as f64 / data.samples.len() as f64;
        if snap_ratio > 0.3 {
            let conf = (snap_ratio * 0.9).clamp(0.2, 0.95);
            detections.push(
                AimbotDetection::new(player_id, AimbotIndicator::SnapToTarget, conf, ts)
                    .with_evidence(&format!("snap_ratio={:.2} ({}/{})", snap_ratio, snap_count, data.samples.len())),
            );
        }

        // Uniform angular velocity (bots have unnaturally consistent aim speed)
        let velocities = data.angular_velocities();
        let cv = PlayerAimData::coefficient_of_variation(&velocities);
        if cv < self.config.uniform_velocity_max_cv && cv > 0.0 {
            let conf = (1.0 - cv / self.config.uniform_velocity_max_cv).clamp(0.2, 0.90);
            detections.push(
                AimbotDetection::new(player_id, AimbotIndicator::UniformAngularVelocity, conf, ts)
                    .with_evidence(&format!("cv={:.4}", cv)),
            );
        }

        for d in &detections {
            if let Some(pd) = self.players.get_mut(&player_id) {
                pd.detections.push(d.clone());
            }
        }
        detections
    }

    pub fn player_detections(&self, player_id: u64) -> Result<&[AimbotDetection], AimbotError> {
        self.players
            .get(&player_id)
            .map(|d| d.detections.as_slice())
            .ok_or(AimbotError::PlayerNotFound(player_id))
    }

    pub fn player_detection_count(&self, player_id: u64) -> usize {
        self.players
            .get(&player_id)
            .map(|d| d.detections.len())
            .unwrap_or(0)
    }

    pub fn player_sample_count(&self, player_id: u64) -> u64 {
        self.players
            .get(&player_id)
            .map(|d| d.total_samples)
            .unwrap_or(0)
    }

    pub fn aggregate_confidence(&self, player_id: u64) -> Result<f64, AimbotError> {
        let data = self
            .players
            .get(&player_id)
            .ok_or(AimbotError::PlayerNotFound(player_id))?;
        if data.detections.is_empty() {
            return Ok(0.0);
        }
        let sum: f64 = data.detections.iter().map(|d| d.confidence).sum();
        Ok((sum / data.detections.len() as f64).clamp(0.0, 1.0))
    }

    pub fn tracked_player_count(&self) -> usize {
        self.players.len()
    }

    pub fn clear_player(&mut self, player_id: u64) {
        self.players.remove(&player_id);
    }

    pub fn config(&self) -> &AimbotDetectorConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(angle: f64, dt: f64, dist: f64, ts: u64) -> AimSample {
        AimSample::new(angle, dt, dist, ts)
    }

    fn fill_human_samples(det: &mut AimbotDetector, pid: u64, count: usize) {
        for i in 0..count {
            let angle = 0.05 + (i as f64 * 0.01).sin() * 0.03;
            let dt = 0.20 + (i as f64 * 0.017).cos() * 0.05;
            det.submit_sample(pid, sample(angle, dt, 50.0, i as u64));
        }
    }

    #[test]
    fn test_aim_sample_angular_velocity() {
        let s = sample(1.0, 0.5, 10.0, 0);
        assert!((s.angular_velocity() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_aim_sample_snap_detection() {
        let s = sample(5.0, 0.01, 10.0, 0);
        assert!(s.is_snap(100.0));
        let s2 = sample(0.1, 0.5, 10.0, 0);
        assert!(!s2.is_snap(100.0));
    }

    #[test]
    fn test_aim_sample_display() {
        let s = sample(0.5, 0.1, 20.0, 0);
        let txt = format!("{s}");
        assert!(txt.contains("aim("));
        assert!(txt.contains("rad/s"));
    }

    #[test]
    fn test_insufficient_samples_no_detection() {
        let cfg = AimbotDetectorConfig::default().with_min_samples(10);
        let mut det = AimbotDetector::new(cfg);
        for i in 0..5 {
            let results = det.submit_sample(1, sample(0.1, 0.2, 10.0, i));
            assert!(results.is_empty());
        }
    }

    #[test]
    fn test_inhuman_reaction_time() {
        let mut det = AimbotDetector::with_default_config();
        fill_human_samples(&mut det, 1, 10);
        let results = det.submit_sample(1, sample(1.0, 0.02, 50.0, 100));
        assert!(results.iter().any(|d| d.indicator == AimbotIndicator::InhumanReactionTime));
    }

    #[test]
    fn test_normal_reaction_time_ok() {
        let mut det = AimbotDetector::with_default_config();
        fill_human_samples(&mut det, 1, 10);
        let results = det.submit_sample(1, sample(0.3, 0.25, 50.0, 100));
        assert!(!results.iter().any(|d| d.indicator == AimbotIndicator::InhumanReactionTime));
    }

    #[test]
    fn test_snap_to_target() {
        let cfg = AimbotDetectorConfig::default().with_snap_threshold(10.0).with_min_samples(3);
        let mut det = AimbotDetector::new(cfg);
        for i in 0..10 {
            det.submit_sample(1, sample(5.0, 0.01, 50.0, i));
        }
        let total = det.player_detection_count(1);
        let dets = det.player_detections(1).unwrap();
        assert!(dets.iter().any(|d| d.indicator == AimbotIndicator::SnapToTarget));
        assert!(total > 0);
    }

    #[test]
    fn test_perfect_tracking_detection() {
        let cfg = AimbotDetectorConfig {
            perfect_tracking_max_variance: 0.001,
            min_samples: 3,
            ..AimbotDetectorConfig::default()
        };
        let mut det = AimbotDetector::new(cfg);
        // Constant angle delta = near-zero variance
        for i in 0..20 {
            det.submit_sample(1, sample(0.001, 0.2, 50.0, i));
        }
        let dets = det.player_detections(1).unwrap();
        assert!(dets.iter().any(|d| d.indicator == AimbotIndicator::PerfectTracking));
    }

    #[test]
    fn test_aggregate_confidence() {
        let mut det = AimbotDetector::with_default_config();
        fill_human_samples(&mut det, 1, 10);
        det.submit_sample(1, sample(5.0, 0.02, 50.0, 200));
        let conf = det.aggregate_confidence(1).unwrap();
        assert!(conf >= 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_aggregate_confidence_no_detections() {
        let mut det = AimbotDetector::with_default_config();
        fill_human_samples(&mut det, 1, 6);
        let conf = det.aggregate_confidence(1).unwrap();
        // Should be 0 or very low if no suspicious behavior
        assert!(conf >= 0.0);
    }

    #[test]
    fn test_player_not_found() {
        let det = AimbotDetector::with_default_config();
        assert!(matches!(det.player_detections(99), Err(AimbotError::PlayerNotFound(99))));
    }

    #[test]
    fn test_tracked_player_count() {
        let mut det = AimbotDetector::with_default_config();
        det.submit_sample(1, sample(0.1, 0.2, 10.0, 0));
        det.submit_sample(2, sample(0.1, 0.2, 10.0, 0));
        assert_eq!(det.tracked_player_count(), 2);
    }

    #[test]
    fn test_clear_player() {
        let mut det = AimbotDetector::with_default_config();
        det.submit_sample(1, sample(0.1, 0.2, 10.0, 0));
        det.clear_player(1);
        assert_eq!(det.tracked_player_count(), 0);
    }

    #[test]
    fn test_sample_window_management() {
        let cfg = AimbotDetectorConfig::default().with_window_size(10);
        let mut det = AimbotDetector::new(cfg);
        for i in 0..20 {
            det.submit_sample(1, sample(0.1, 0.2, 10.0, i));
        }
        assert_eq!(det.player_sample_count(1), 20);
    }

    #[test]
    fn test_detection_display() {
        let d = AimbotDetection::new(5, AimbotIndicator::SnapToTarget, 0.85, 1000);
        let s = format!("{d}");
        assert!(s.contains("player=5"));
        assert!(s.contains("SnapToTarget"));
    }

    #[test]
    fn test_indicator_display() {
        assert_eq!(format!("{}", AimbotIndicator::PerfectTracking), "PerfectTracking");
        assert_eq!(format!("{}", AimbotIndicator::AbnormalAccuracy), "AbnormalAccuracy");
    }

    #[test]
    fn test_config_builder() {
        let cfg = AimbotDetectorConfig::default()
            .with_snap_threshold(25.0)
            .with_min_samples(8)
            .with_window_size(50);
        assert!((cfg.snap_threshold - 25.0).abs() < f64::EPSILON);
        assert_eq!(cfg.min_samples, 8);
        assert_eq!(cfg.window_size, 50);
    }

    #[test]
    fn test_human_profile_defaults() {
        let p = HumanAimProfile::default();
        assert!(p.mean_reaction_time > 0.0);
        assert!(p.min_reaction_time > 0.0);
        assert!(p.angular_velocity_std > 0.0);
    }

    #[test]
    fn test_detection_with_evidence() {
        let d = AimbotDetection::new(1, AimbotIndicator::InhumanReactionTime, 0.9, 0)
            .with_evidence("dt=0.02s");
        assert_eq!(d.evidence, "dt=0.02s");
    }
}
