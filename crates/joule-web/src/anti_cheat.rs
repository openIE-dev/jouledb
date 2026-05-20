//! Anti-cheat detection engine — aggregates multiple detectors, scores players.
//!
//! Replaces EasyAntiCheat / Vanguard integration layers with pure Rust.
//! CheatType classification, Detection records with confidence scoring,
//! multi-detector aggregation, configurable thresholds, player risk scoring,
//! alert generation, detection history, false-positive tracking, and
//! detector enable/disable management.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntiCheatError {
    DetectorNotFound(String),
    DuplicateDetector(String),
    PlayerNotFound(u64),
    InvalidThreshold(String),
    DetectorDisabled(String),
}

impl fmt::Display for AntiCheatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DetectorNotFound(id) => write!(f, "detector not found: {id}"),
            Self::DuplicateDetector(id) => write!(f, "duplicate detector: {id}"),
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::InvalidThreshold(msg) => write!(f, "invalid threshold: {msg}"),
            Self::DetectorDisabled(id) => write!(f, "detector disabled: {id}"),
        }
    }
}

impl std::error::Error for AntiCheatError {}

// ── Cheat Types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheatType {
    SpeedHack,
    Aimbot,
    WallHack,
    Teleport,
    ResourceHack,
}

impl fmt::Display for CheatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpeedHack => write!(f, "SpeedHack"),
            Self::Aimbot => write!(f, "Aimbot"),
            Self::WallHack => write!(f, "WallHack"),
            Self::Teleport => write!(f, "Teleport"),
            Self::ResourceHack => write!(f, "ResourceHack"),
        }
    }
}

// ── Detection ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub cheat_type: CheatType,
    pub confidence: f64,
    pub timestamp: u64,
    pub player_id: u64,
    pub detector_id: String,
    pub details: String,
    pub false_positive: bool,
}

impl Detection {
    pub fn new(cheat_type: CheatType, confidence: f64, timestamp: u64, player_id: u64) -> Self {
        Self {
            cheat_type,
            confidence: confidence.clamp(0.0, 1.0),
            timestamp,
            player_id,
            detector_id: String::new(),
            details: String::new(),
            false_positive: false,
        }
    }

    pub fn with_detector(mut self, id: &str) -> Self {
        self.detector_id = id.to_string();
        self
    }

    pub fn with_details(mut self, details: &str) -> Self {
        self.details = details.to_string();
        self
    }

    pub fn mark_false_positive(&mut self) {
        self.false_positive = true;
    }
}

impl fmt::Display for Detection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[t={}] player={} type={} conf={:.2} detector={}",
            self.timestamp, self.player_id, self.cheat_type, self.confidence, self.detector_id
        )
    }
}

// ── Alert ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alert {
    pub player_id: u64,
    pub severity: AlertSeverity,
    pub risk_score: f64,
    pub detection_count: usize,
    pub timestamp: u64,
    pub primary_cheat_type: CheatType,
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ALERT [{}] player={} risk={:.2} detections={} type={}",
            self.severity, self.player_id, self.risk_score, self.detection_count,
            self.primary_cheat_type
        )
    }
}

// ── Detector Config ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DetectorConfig {
    pub id: String,
    pub enabled: bool,
    pub weight: f64,
    pub cheat_types: Vec<CheatType>,
}

impl DetectorConfig {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            enabled: true,
            weight: 1.0,
            cheat_types: Vec::new(),
        }
    }

    pub fn with_weight(mut self, w: f64) -> Self {
        self.weight = w.max(0.0);
        self
    }

    pub fn with_cheat_types(mut self, types: Vec<CheatType>) -> Self {
        self.cheat_types = types;
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ── Threshold Config ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    pub alert_low: f64,
    pub alert_medium: f64,
    pub alert_high: f64,
    pub alert_critical: f64,
    pub min_confidence: f64,
    pub history_window: u64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            alert_low: 0.3,
            alert_medium: 0.5,
            alert_high: 0.7,
            alert_critical: 0.9,
            min_confidence: 0.1,
            history_window: 3600,
        }
    }
}

impl ThresholdConfig {
    pub fn severity_for(&self, score: f64) -> Option<AlertSeverity> {
        if score >= self.alert_critical {
            Some(AlertSeverity::Critical)
        } else if score >= self.alert_high {
            Some(AlertSeverity::High)
        } else if score >= self.alert_medium {
            Some(AlertSeverity::Medium)
        } else if score >= self.alert_low {
            Some(AlertSeverity::Low)
        } else {
            None
        }
    }
}

// ── Player Record ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlayerRecord {
    pub player_id: u64,
    pub detections: Vec<Detection>,
    pub false_positive_count: usize,
    pub total_detection_count: usize,
}

impl PlayerRecord {
    pub fn new(player_id: u64) -> Self {
        Self {
            player_id,
            detections: Vec::new(),
            false_positive_count: 0,
            total_detection_count: 0,
        }
    }

    pub fn add_detection(&mut self, det: Detection) {
        self.total_detection_count += 1;
        self.detections.push(det);
    }

    pub fn mark_false_positive(&mut self, index: usize) -> bool {
        if let Some(det) = self.detections.get_mut(index) {
            if !det.false_positive {
                det.false_positive = true;
                self.false_positive_count += 1;
                return true;
            }
        }
        false
    }

    pub fn active_detections(&self) -> impl Iterator<Item = &Detection> {
        self.detections.iter().filter(|d| !d.false_positive)
    }

    pub fn active_count(&self) -> usize {
        self.detections.iter().filter(|d| !d.false_positive).count()
    }

    pub fn risk_score(&self, weights: &HashMap<String, f64>, min_confidence: f64) -> f64 {
        let active: Vec<_> = self
            .detections
            .iter()
            .filter(|d| !d.false_positive && d.confidence >= min_confidence)
            .collect();
        if active.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = active
            .iter()
            .map(|d| weights.get(&d.detector_id).copied().unwrap_or(1.0))
            .sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        let weighted_sum: f64 = active
            .iter()
            .map(|d| {
                let w = weights.get(&d.detector_id).copied().unwrap_or(1.0);
                d.confidence * w
            })
            .sum();
        (weighted_sum / total_weight).clamp(0.0, 1.0)
    }

    pub fn primary_cheat_type(&self) -> CheatType {
        let mut counts: HashMap<CheatType, usize> = HashMap::new();
        for d in self.detections.iter().filter(|d| !d.false_positive) {
            *counts.entry(d.cheat_type).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(t, _)| t)
            .unwrap_or(CheatType::SpeedHack)
    }

    pub fn detections_in_window(&self, now: u64, window: u64) -> Vec<&Detection> {
        let cutoff = now.saturating_sub(window);
        self.detections
            .iter()
            .filter(|d| d.timestamp >= cutoff && !d.false_positive)
            .collect()
    }
}

// ── Anti-Cheat Engine ───────────────────────────────────────────

#[derive(Debug)]
pub struct AntiCheatEngine {
    detectors: HashMap<String, DetectorConfig>,
    players: HashMap<u64, PlayerRecord>,
    alerts: Vec<Alert>,
    thresholds: ThresholdConfig,
    current_time: u64,
}

impl AntiCheatEngine {
    pub fn new() -> Self {
        Self {
            detectors: HashMap::new(),
            players: HashMap::new(),
            alerts: Vec::new(),
            thresholds: ThresholdConfig::default(),
            current_time: 0,
        }
    }

    pub fn with_thresholds(mut self, cfg: ThresholdConfig) -> Self {
        self.thresholds = cfg;
        self
    }

    pub fn set_time(&mut self, t: u64) {
        self.current_time = t;
    }

    pub fn register_detector(&mut self, cfg: DetectorConfig) -> Result<(), AntiCheatError> {
        if self.detectors.contains_key(&cfg.id) {
            return Err(AntiCheatError::DuplicateDetector(cfg.id));
        }
        self.detectors.insert(cfg.id.clone(), cfg);
        Ok(())
    }

    pub fn enable_detector(&mut self, id: &str) -> Result<(), AntiCheatError> {
        let det = self
            .detectors
            .get_mut(id)
            .ok_or_else(|| AntiCheatError::DetectorNotFound(id.to_string()))?;
        det.enabled = true;
        Ok(())
    }

    pub fn disable_detector(&mut self, id: &str) -> Result<(), AntiCheatError> {
        let det = self
            .detectors
            .get_mut(id)
            .ok_or_else(|| AntiCheatError::DetectorNotFound(id.to_string()))?;
        det.enabled = false;
        Ok(())
    }

    pub fn is_detector_enabled(&self, id: &str) -> Result<bool, AntiCheatError> {
        self.detectors
            .get(id)
            .map(|d| d.enabled)
            .ok_or_else(|| AntiCheatError::DetectorNotFound(id.to_string()))
    }

    pub fn detector_count(&self) -> usize {
        self.detectors.len()
    }

    pub fn submit_detection(&mut self, det: Detection) -> Result<Option<Alert>, AntiCheatError> {
        if !det.detector_id.is_empty() {
            if let Some(cfg) = self.detectors.get(&det.detector_id) {
                if !cfg.enabled {
                    return Err(AntiCheatError::DetectorDisabled(det.detector_id.clone()));
                }
            }
        }
        if det.confidence < self.thresholds.min_confidence {
            return Ok(None);
        }
        let player_id = det.player_id;
        let weights = self.detector_weights();
        let min_confidence = self.thresholds.min_confidence;
        let current_time = self.current_time;
        let history_window = self.thresholds.history_window;
        let record = self
            .players
            .entry(player_id)
            .or_insert_with(|| PlayerRecord::new(player_id));
        record.add_detection(det);

        let score = record.risk_score(&weights, min_confidence);
        let windowed = record.detections_in_window(current_time, history_window);

        if let Some(severity) = self.thresholds.severity_for(score) {
            let alert = Alert {
                player_id,
                severity,
                risk_score: score,
                detection_count: windowed.len(),
                timestamp: self.current_time,
                primary_cheat_type: record.primary_cheat_type(),
            };
            self.alerts.push(alert.clone());
            Ok(Some(alert))
        } else {
            Ok(None)
        }
    }

    pub fn player_risk_score(&self, player_id: u64) -> Result<f64, AntiCheatError> {
        let record = self
            .players
            .get(&player_id)
            .ok_or(AntiCheatError::PlayerNotFound(player_id))?;
        let weights = self.detector_weights();
        Ok(record.risk_score(&weights, self.thresholds.min_confidence))
    }

    pub fn player_detection_count(&self, player_id: u64) -> usize {
        self.players
            .get(&player_id)
            .map(|r| r.active_count())
            .unwrap_or(0)
    }

    pub fn player_false_positive_count(&self, player_id: u64) -> usize {
        self.players
            .get(&player_id)
            .map(|r| r.false_positive_count)
            .unwrap_or(0)
    }

    pub fn mark_false_positive(
        &mut self,
        player_id: u64,
        det_index: usize,
    ) -> Result<bool, AntiCheatError> {
        let record = self
            .players
            .get_mut(&player_id)
            .ok_or(AntiCheatError::PlayerNotFound(player_id))?;
        Ok(record.mark_false_positive(det_index))
    }

    pub fn detection_history(&self, player_id: u64) -> Option<&[Detection]> {
        self.players.get(&player_id).map(|r| r.detections.as_slice())
    }

    pub fn alerts(&self) -> &[Alert] {
        &self.alerts
    }

    pub fn alerts_for_player(&self, player_id: u64) -> Vec<&Alert> {
        self.alerts.iter().filter(|a| a.player_id == player_id).collect()
    }

    pub fn total_players_tracked(&self) -> usize {
        self.players.len()
    }

    fn detector_weights(&self) -> HashMap<String, f64> {
        self.detectors
            .iter()
            .filter(|(_, cfg)| cfg.enabled)
            .map(|(id, cfg)| (id.clone(), cfg.weight))
            .collect()
    }
}

impl Default for AntiCheatEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with_detector() -> AntiCheatEngine {
        let mut e = AntiCheatEngine::new();
        e.set_time(1000);
        e.register_detector(DetectorConfig::new("speed").with_weight(1.0)).unwrap();
        e
    }

    #[test]
    fn test_detection_creation() {
        let d = Detection::new(CheatType::SpeedHack, 0.85, 100, 42);
        assert_eq!(d.cheat_type, CheatType::SpeedHack);
        assert!((d.confidence - 0.85).abs() < f64::EPSILON);
        assert_eq!(d.player_id, 42);
    }

    #[test]
    fn test_detection_confidence_clamped() {
        let d = Detection::new(CheatType::Aimbot, 1.5, 0, 1);
        assert!((d.confidence - 1.0).abs() < f64::EPSILON);
        let d2 = Detection::new(CheatType::Aimbot, -0.3, 0, 1);
        assert!((d2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detection_builder() {
        let d = Detection::new(CheatType::WallHack, 0.6, 50, 7)
            .with_detector("wallhack-v2")
            .with_details("saw through wall at pos (10,20)");
        assert_eq!(d.detector_id, "wallhack-v2");
        assert!(d.details.contains("saw through wall"));
    }

    #[test]
    fn test_detection_display() {
        let d = Detection::new(CheatType::Teleport, 0.9, 200, 3).with_detector("tp-det");
        let s = format!("{d}");
        assert!(s.contains("player=3"));
        assert!(s.contains("Teleport"));
    }

    #[test]
    fn test_register_detector() {
        let mut e = AntiCheatEngine::new();
        e.register_detector(DetectorConfig::new("d1")).unwrap();
        assert_eq!(e.detector_count(), 1);
    }

    #[test]
    fn test_duplicate_detector_rejected() {
        let mut e = AntiCheatEngine::new();
        e.register_detector(DetectorConfig::new("d1")).unwrap();
        let err = e.register_detector(DetectorConfig::new("d1")).unwrap_err();
        assert!(matches!(err, AntiCheatError::DuplicateDetector(_)));
    }

    #[test]
    fn test_enable_disable_detector() {
        let mut e = AntiCheatEngine::new();
        e.register_detector(DetectorConfig::new("d1")).unwrap();
        assert!(e.is_detector_enabled("d1").unwrap());
        e.disable_detector("d1").unwrap();
        assert!(!e.is_detector_enabled("d1").unwrap());
        e.enable_detector("d1").unwrap();
        assert!(e.is_detector_enabled("d1").unwrap());
    }

    #[test]
    fn test_submit_detection_below_threshold() {
        let mut e = engine_with_detector();
        let d = Detection::new(CheatType::SpeedHack, 0.05, 1000, 1).with_detector("speed");
        let result = e.submit_detection(d).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_submit_detection_generates_alert() {
        let mut e = engine_with_detector();
        let d = Detection::new(CheatType::SpeedHack, 0.95, 1000, 1).with_detector("speed");
        let alert = e.submit_detection(d).unwrap();
        assert!(alert.is_some());
        let a = alert.unwrap();
        assert_eq!(a.player_id, 1);
        assert!(a.risk_score >= 0.9);
    }

    #[test]
    fn test_disabled_detector_rejected() {
        let mut e = AntiCheatEngine::new();
        e.register_detector(DetectorConfig::new("d1").disabled()).unwrap();
        let d = Detection::new(CheatType::Aimbot, 0.8, 100, 5).with_detector("d1");
        let err = e.submit_detection(d).unwrap_err();
        assert!(matches!(err, AntiCheatError::DetectorDisabled(_)));
    }

    #[test]
    fn test_player_risk_score() {
        let mut e = engine_with_detector();
        let d = Detection::new(CheatType::SpeedHack, 0.7, 1000, 42).with_detector("speed");
        e.submit_detection(d).unwrap();
        let score = e.player_risk_score(42).unwrap();
        assert!((score - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_player_not_found() {
        let e = AntiCheatEngine::new();
        let err = e.player_risk_score(999).unwrap_err();
        assert!(matches!(err, AntiCheatError::PlayerNotFound(999)));
    }

    #[test]
    fn test_detection_history() {
        let mut e = engine_with_detector();
        e.submit_detection(Detection::new(CheatType::SpeedHack, 0.5, 100, 1).with_detector("speed")).unwrap();
        e.submit_detection(Detection::new(CheatType::Aimbot, 0.6, 200, 1).with_detector("speed")).unwrap();
        let hist = e.detection_history(1).unwrap();
        assert_eq!(hist.len(), 2);
    }

    #[test]
    fn test_false_positive_tracking() {
        let mut e = engine_with_detector();
        e.submit_detection(Detection::new(CheatType::SpeedHack, 0.8, 100, 1).with_detector("speed")).unwrap();
        assert_eq!(e.player_false_positive_count(1), 0);
        e.mark_false_positive(1, 0).unwrap();
        assert_eq!(e.player_false_positive_count(1), 1);
        assert_eq!(e.player_detection_count(1), 0);
    }

    #[test]
    fn test_mark_false_positive_idempotent() {
        let mut e = engine_with_detector();
        e.submit_detection(Detection::new(CheatType::SpeedHack, 0.8, 100, 1).with_detector("speed")).unwrap();
        assert!(e.mark_false_positive(1, 0).unwrap());
        assert!(!e.mark_false_positive(1, 0).unwrap());
        assert_eq!(e.player_false_positive_count(1), 1);
    }

    #[test]
    fn test_alert_severity_levels() {
        let cfg = ThresholdConfig::default();
        assert_eq!(cfg.severity_for(0.1), None);
        assert_eq!(cfg.severity_for(0.3), Some(AlertSeverity::Low));
        assert_eq!(cfg.severity_for(0.5), Some(AlertSeverity::Medium));
        assert_eq!(cfg.severity_for(0.7), Some(AlertSeverity::High));
        assert_eq!(cfg.severity_for(0.95), Some(AlertSeverity::Critical));
    }

    #[test]
    fn test_alerts_for_player() {
        let mut e = engine_with_detector();
        e.submit_detection(Detection::new(CheatType::SpeedHack, 0.95, 1000, 1).with_detector("speed")).unwrap();
        e.submit_detection(Detection::new(CheatType::Aimbot, 0.95, 1000, 2).with_detector("speed")).unwrap();
        let p1_alerts = e.alerts_for_player(1);
        assert_eq!(p1_alerts.len(), 1);
        assert_eq!(p1_alerts[0].player_id, 1);
    }

    #[test]
    fn test_total_players_tracked() {
        let mut e = engine_with_detector();
        e.submit_detection(Detection::new(CheatType::SpeedHack, 0.5, 100, 1).with_detector("speed")).unwrap();
        e.submit_detection(Detection::new(CheatType::SpeedHack, 0.5, 100, 2).with_detector("speed")).unwrap();
        assert_eq!(e.total_players_tracked(), 2);
    }

    #[test]
    fn test_cheat_type_display() {
        assert_eq!(format!("{}", CheatType::SpeedHack), "SpeedHack");
        assert_eq!(format!("{}", CheatType::ResourceHack), "ResourceHack");
    }

    #[test]
    fn test_player_record_primary_cheat_type() {
        let mut rec = PlayerRecord::new(1);
        rec.add_detection(Detection::new(CheatType::SpeedHack, 0.5, 10, 1));
        rec.add_detection(Detection::new(CheatType::SpeedHack, 0.6, 20, 1));
        rec.add_detection(Detection::new(CheatType::Aimbot, 0.7, 30, 1));
        assert_eq!(rec.primary_cheat_type(), CheatType::SpeedHack);
    }
}
