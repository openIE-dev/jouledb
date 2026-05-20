//! Graceful degradation — feature flags for degraded mode, health-based
//! feature disabling, degradation levels, recovery detection, degradation
//! dashboard data, automatic recovery, and manual override.
//!
//! Pure Rust degradation management for resilient systems.
//! Features can be automatically disabled based on health signals and
//! re-enabled when the system recovers.

use std::collections::HashMap;

// ── Degradation Level ───────────────────────────────────────────

/// How degraded the system (or a feature) is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DegradationLevel {
    /// Fully operational.
    Normal,
    /// Minor degradation — non-critical features disabled.
    Mild,
    /// Significant degradation — many features disabled.
    Moderate,
    /// Severe — only essential features remain.
    Severe,
    /// Complete outage — system is down.
    Critical,
}

impl DegradationLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Mild => "mild",
            Self::Moderate => "moderate",
            Self::Severe => "severe",
            Self::Critical => "critical",
        }
    }

    pub fn ordinal(&self) -> u8 {
        match self {
            Self::Normal => 0,
            Self::Mild => 1,
            Self::Moderate => 2,
            Self::Severe => 3,
            Self::Critical => 4,
        }
    }
}

// ── Feature State ───────────────────────────────────────────────

/// State of a feature under degradation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureState {
    /// Feature is enabled and operational.
    Enabled,
    /// Feature is disabled due to degradation.
    Disabled,
    /// Feature is in a reduced-functionality mode.
    Reduced,
    /// Feature has been manually overridden (forced on or off).
    Overridden(bool),
}

impl FeatureState {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Enabled | Self::Reduced | Self::Overridden(true))
    }
}

// ── Feature Config ──────────────────────────────────────────────

/// Configuration for a degradable feature.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Feature name.
    pub name: String,
    /// At what degradation level this feature gets disabled.
    pub disable_at: DegradationLevel,
    /// At what level it goes into reduced mode (if any).
    pub reduce_at: Option<DegradationLevel>,
    /// Whether this feature is essential (never disabled).
    pub essential: bool,
    /// Description for dashboards.
    pub description: String,
}

impl FeatureConfig {
    pub fn new(name: impl Into<String>, disable_at: DegradationLevel) -> Self {
        Self {
            name: name.into(),
            disable_at,
            reduce_at: None,
            essential: false,
            description: String::new(),
        }
    }

    pub fn with_reduce_at(mut self, level: DegradationLevel) -> Self {
        self.reduce_at = Some(level);
        self
    }

    pub fn essential(mut self) -> Self {
        self.essential = true;
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

// ── Health Signal ───────────────────────────────────────────────

/// A health signal that can trigger degradation.
#[derive(Debug, Clone)]
pub struct HealthSignal {
    pub name: String,
    /// Health score 0.0 (dead) to 1.0 (perfect).
    pub score: f64,
    /// Threshold below which this signal triggers degradation.
    pub threshold: f64,
}

impl HealthSignal {
    pub fn new(name: impl Into<String>, score: f64, threshold: f64) -> Self {
        Self {
            name: name.into(),
            score: score.clamp(0.0, 1.0),
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.score >= self.threshold
    }

    pub fn update(&mut self, score: f64) {
        self.score = score.clamp(0.0, 1.0);
    }
}

// ── Recovery Detector ───────────────────────────────────────────

/// Detects when the system has recovered by requiring N consecutive
/// healthy checks before declaring recovery.
#[derive(Debug)]
pub struct RecoveryDetector {
    required_consecutive: u32,
    current_consecutive: u32,
    total_checks: u64,
    total_recoveries: u64,
}

impl RecoveryDetector {
    pub fn new(required_consecutive: u32) -> Self {
        Self {
            required_consecutive: required_consecutive.max(1),
            current_consecutive: 0,
            total_checks: 0,
            total_recoveries: 0,
        }
    }

    /// Report a health check result. Returns true if recovery is declared.
    pub fn check(&mut self, healthy: bool) -> bool {
        self.total_checks += 1;
        if healthy {
            self.current_consecutive += 1;
            if self.current_consecutive >= self.required_consecutive {
                self.current_consecutive = 0;
                self.total_recoveries += 1;
                return true;
            }
        } else {
            self.current_consecutive = 0;
        }
        false
    }

    pub fn consecutive_healthy(&self) -> u32 {
        self.current_consecutive
    }

    pub fn total_recoveries(&self) -> u64 {
        self.total_recoveries
    }

    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }

    pub fn progress_pct(&self) -> f64 {
        if self.required_consecutive == 0 {
            return 100.0;
        }
        (self.current_consecutive as f64 / self.required_consecutive as f64) * 100.0
    }
}

// ── Dashboard Data ──────────────────────────────────────────────

/// Data for a degradation dashboard.
#[derive(Debug, Clone)]
pub struct DashboardData {
    pub current_level: DegradationLevel,
    pub features: Vec<(String, FeatureState)>,
    pub health_signals: Vec<(String, f64, bool)>, // name, score, healthy
    pub overrides: Vec<(String, bool)>,
}

// ── Degradation Manager ─────────────────────────────────────────

/// Central manager for graceful degradation.
#[derive(Debug)]
pub struct DegradationManager {
    /// Current degradation level.
    level: DegradationLevel,
    /// Feature configurations.
    features: Vec<FeatureConfig>,
    /// Health signals.
    signals: Vec<HealthSignal>,
    /// Manual overrides (feature_name -> forced state).
    overrides: HashMap<String, bool>,
    /// Recovery detector.
    recovery: RecoveryDetector,
    /// Whether automatic level adjustment is enabled.
    auto_adjust: bool,
    /// Level transition history.
    level_history: Vec<(DegradationLevel, DegradationLevel)>,
    /// Maximum history entries.
    max_history: usize,
}

impl DegradationManager {
    pub fn new() -> Self {
        Self {
            level: DegradationLevel::Normal,
            features: Vec::new(),
            signals: Vec::new(),
            overrides: HashMap::new(),
            recovery: RecoveryDetector::new(3),
            auto_adjust: true,
            level_history: Vec::new(),
            max_history: 50,
        }
    }

    /// Set the recovery threshold (consecutive healthy checks).
    pub fn with_recovery_threshold(mut self, threshold: u32) -> Self {
        self.recovery = RecoveryDetector::new(threshold);
        self
    }

    /// Enable or disable automatic level adjustment.
    pub fn with_auto_adjust(mut self, enabled: bool) -> Self {
        self.auto_adjust = enabled;
        self
    }

    /// Register a feature.
    pub fn add_feature(&mut self, config: FeatureConfig) {
        self.features.push(config);
    }

    /// Register a health signal.
    pub fn add_signal(&mut self, signal: HealthSignal) {
        self.signals.push(signal);
    }

    /// Set a manual override for a feature.
    pub fn set_override(&mut self, feature: impl Into<String>, enabled: bool) {
        self.overrides.insert(feature.into(), enabled);
    }

    /// Remove a manual override.
    pub fn clear_override(&mut self, feature: &str) {
        self.overrides.remove(feature);
    }

    /// Clear all overrides.
    pub fn clear_all_overrides(&mut self) {
        self.overrides.clear();
    }

    /// Manually set the degradation level.
    pub fn set_level(&mut self, level: DegradationLevel) {
        if level != self.level {
            let old = self.level;
            self.level = level;
            self.level_history.push((old, level));
            if self.level_history.len() > self.max_history {
                self.level_history.remove(0);
            }
        }
    }

    /// Get the current degradation level.
    pub fn level(&self) -> DegradationLevel {
        self.level
    }

    /// Get the state of a feature given the current degradation level.
    pub fn feature_state(&self, feature_name: &str) -> FeatureState {
        // Check manual override first.
        if let Some(forced) = self.overrides.get(feature_name) {
            return FeatureState::Overridden(*forced);
        }

        let config = match self.features.iter().find(|f| f.name == feature_name) {
            Some(c) => c,
            None => return FeatureState::Enabled, // Unknown features are enabled.
        };

        if config.essential {
            return FeatureState::Enabled;
        }

        if self.level >= config.disable_at {
            return FeatureState::Disabled;
        }

        if let Some(reduce_at) = config.reduce_at {
            if self.level >= reduce_at {
                return FeatureState::Reduced;
            }
        }

        FeatureState::Enabled
    }

    /// Whether a feature is currently available (enabled, reduced, or forced on).
    pub fn is_feature_available(&self, feature_name: &str) -> bool {
        self.feature_state(feature_name).is_available()
    }

    /// Update a health signal.
    pub fn update_signal(&mut self, name: &str, score: f64) {
        if let Some(signal) = self.signals.iter_mut().find(|s| s.name == name) {
            signal.update(score);
        }
        if self.auto_adjust {
            self.auto_adjust_level();
        }
    }

    /// Automatically adjust the degradation level based on health signals.
    fn auto_adjust_level(&mut self) {
        let unhealthy_count = self.signals.iter().filter(|s| !s.is_healthy()).count();
        let total = self.signals.len();
        if total == 0 {
            return;
        }

        let unhealthy_ratio = unhealthy_count as f64 / total as f64;
        let new_level = if unhealthy_ratio >= 0.8 {
            DegradationLevel::Critical
        } else if unhealthy_ratio >= 0.6 {
            DegradationLevel::Severe
        } else if unhealthy_ratio >= 0.4 {
            DegradationLevel::Moderate
        } else if unhealthy_ratio > 0.0 {
            DegradationLevel::Mild
        } else {
            DegradationLevel::Normal
        };

        self.set_level(new_level);
    }

    /// Run a recovery check. Returns true if recovery was detected.
    pub fn check_recovery(&mut self) -> bool {
        let all_healthy = self.signals.iter().all(|s| s.is_healthy());
        let recovered = self.recovery.check(all_healthy);
        if recovered && self.level != DegradationLevel::Normal {
            self.set_level(DegradationLevel::Normal);
        }
        recovered
    }

    /// Generate dashboard data.
    pub fn dashboard(&self) -> DashboardData {
        let features = self
            .features
            .iter()
            .map(|f| (f.name.clone(), self.feature_state(&f.name)))
            .collect();

        let health_signals = self
            .signals
            .iter()
            .map(|s| (s.name.clone(), s.score, s.is_healthy()))
            .collect();

        let mut overrides: Vec<(String, bool)> = self
            .overrides
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        overrides.sort_by(|a, b| a.0.cmp(&b.0));

        DashboardData {
            current_level: self.level,
            features,
            health_signals,
            overrides,
        }
    }

    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }

    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    pub fn level_history(&self) -> &[(DegradationLevel, DegradationLevel)] {
        &self.level_history
    }

    pub fn recovery_detector(&self) -> &RecoveryDetector {
        &self.recovery
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degradation_level_ordering() {
        assert!(DegradationLevel::Normal < DegradationLevel::Mild);
        assert!(DegradationLevel::Mild < DegradationLevel::Moderate);
        assert!(DegradationLevel::Moderate < DegradationLevel::Severe);
        assert!(DegradationLevel::Severe < DegradationLevel::Critical);
    }

    #[test]
    fn test_degradation_level_as_str() {
        assert_eq!(DegradationLevel::Normal.as_str(), "normal");
        assert_eq!(DegradationLevel::Critical.as_str(), "critical");
    }

    #[test]
    fn test_feature_state_available() {
        assert!(FeatureState::Enabled.is_available());
        assert!(FeatureState::Reduced.is_available());
        assert!(FeatureState::Overridden(true).is_available());
        assert!(!FeatureState::Disabled.is_available());
        assert!(!FeatureState::Overridden(false).is_available());
    }

    #[test]
    fn test_feature_enabled_at_normal() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(FeatureConfig::new("search", DegradationLevel::Moderate));
        assert_eq!(mgr.feature_state("search"), FeatureState::Enabled);
        assert!(mgr.is_feature_available("search"));
    }

    #[test]
    fn test_feature_disabled_at_level() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(FeatureConfig::new("search", DegradationLevel::Moderate));
        mgr.set_level(DegradationLevel::Moderate);
        assert_eq!(mgr.feature_state("search"), FeatureState::Disabled);
        assert!(!mgr.is_feature_available("search"));
    }

    #[test]
    fn test_feature_reduced() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(
            FeatureConfig::new("analytics", DegradationLevel::Severe)
                .with_reduce_at(DegradationLevel::Mild),
        );
        mgr.set_level(DegradationLevel::Mild);
        assert_eq!(mgr.feature_state("analytics"), FeatureState::Reduced);
        assert!(mgr.is_feature_available("analytics"));
    }

    #[test]
    fn test_essential_feature() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(
            FeatureConfig::new("auth", DegradationLevel::Mild).essential(),
        );
        mgr.set_level(DegradationLevel::Critical);
        assert_eq!(mgr.feature_state("auth"), FeatureState::Enabled);
    }

    #[test]
    fn test_manual_override_on() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(FeatureConfig::new("search", DegradationLevel::Moderate));
        mgr.set_level(DegradationLevel::Critical);
        mgr.set_override("search", true);
        assert_eq!(mgr.feature_state("search"), FeatureState::Overridden(true));
        assert!(mgr.is_feature_available("search"));
    }

    #[test]
    fn test_manual_override_off() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(FeatureConfig::new("search", DegradationLevel::Severe));
        mgr.set_override("search", false);
        assert_eq!(mgr.feature_state("search"), FeatureState::Overridden(false));
        assert!(!mgr.is_feature_available("search"));
    }

    #[test]
    fn test_clear_override() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(FeatureConfig::new("search", DegradationLevel::Severe));
        mgr.set_override("search", false);
        mgr.clear_override("search");
        assert_eq!(mgr.feature_state("search"), FeatureState::Enabled);
        assert_eq!(mgr.override_count(), 0);
    }

    #[test]
    fn test_health_signal() {
        let mut signal = HealthSignal::new("cpu", 0.8, 0.5);
        assert!(signal.is_healthy());
        signal.update(0.3);
        assert!(!signal.is_healthy());
    }

    #[test]
    fn test_auto_adjust_from_signals() {
        let mut mgr = DegradationManager::new();
        mgr.add_signal(HealthSignal::new("cpu", 1.0, 0.5));
        mgr.add_signal(HealthSignal::new("mem", 1.0, 0.5));
        mgr.add_signal(HealthSignal::new("disk", 1.0, 0.5));
        assert_eq!(mgr.level(), DegradationLevel::Normal);

        // Make one unhealthy.
        mgr.update_signal("cpu", 0.2);
        // 1/3 unhealthy > 0 => Mild.
        assert_eq!(mgr.level(), DegradationLevel::Mild);
    }

    #[test]
    fn test_auto_adjust_severe() {
        let mut mgr = DegradationManager::new();
        mgr.add_signal(HealthSignal::new("a", 1.0, 0.5));
        mgr.add_signal(HealthSignal::new("b", 1.0, 0.5));
        mgr.add_signal(HealthSignal::new("c", 1.0, 0.5));
        mgr.add_signal(HealthSignal::new("d", 1.0, 0.5));
        mgr.add_signal(HealthSignal::new("e", 1.0, 0.5));

        // Make 4/5 unhealthy (80%) => Critical.
        mgr.update_signal("a", 0.1);
        mgr.update_signal("b", 0.1);
        mgr.update_signal("c", 0.1);
        mgr.update_signal("d", 0.1);
        assert_eq!(mgr.level(), DegradationLevel::Critical);
    }

    #[test]
    fn test_recovery_detector() {
        let mut rd = RecoveryDetector::new(3);
        assert!(!rd.check(true));
        assert!(!rd.check(true));
        assert!(rd.check(true)); // Third consecutive.
        assert_eq!(rd.total_recoveries(), 1);
    }

    #[test]
    fn test_recovery_detector_reset_on_failure() {
        let mut rd = RecoveryDetector::new(3);
        rd.check(true);
        rd.check(true);
        rd.check(false); // Resets.
        assert!(!rd.check(true));
        assert!(!rd.check(true));
        assert!(rd.check(true)); // Needs 3 more.
    }

    #[test]
    fn test_recovery_integration() {
        let mut mgr = DegradationManager::new()
            .with_recovery_threshold(2)
            .with_auto_adjust(false);
        mgr.add_signal(HealthSignal::new("cpu", 1.0, 0.5));
        mgr.set_level(DegradationLevel::Severe);

        // System healthy, check recovery.
        assert!(!mgr.check_recovery()); // 1st.
        assert!(mgr.check_recovery()); // 2nd => recovered.
        assert_eq!(mgr.level(), DegradationLevel::Normal);
    }

    #[test]
    fn test_dashboard() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.add_feature(FeatureConfig::new("search", DegradationLevel::Moderate));
        mgr.add_signal(HealthSignal::new("cpu", 0.9, 0.5));
        mgr.set_override("debug", true);
        let data = mgr.dashboard();
        assert_eq!(data.current_level, DegradationLevel::Normal);
        assert_eq!(data.features.len(), 1);
        assert_eq!(data.health_signals.len(), 1);
        assert_eq!(data.overrides.len(), 1);
    }

    #[test]
    fn test_level_history() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.set_level(DegradationLevel::Mild);
        mgr.set_level(DegradationLevel::Severe);
        let hist = mgr.level_history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0], (DegradationLevel::Normal, DegradationLevel::Mild));
        assert_eq!(hist[1], (DegradationLevel::Mild, DegradationLevel::Severe));
    }

    #[test]
    fn test_unknown_feature_enabled() {
        let mgr = DegradationManager::new();
        assert_eq!(mgr.feature_state("nonexistent"), FeatureState::Enabled);
    }

    #[test]
    fn test_feature_config_builder() {
        let cfg = FeatureConfig::new("search", DegradationLevel::Moderate)
            .with_reduce_at(DegradationLevel::Mild)
            .essential()
            .with_description("Full-text search");
        assert_eq!(cfg.name, "search");
        assert_eq!(cfg.disable_at, DegradationLevel::Moderate);
        assert_eq!(cfg.reduce_at, Some(DegradationLevel::Mild));
        assert!(cfg.essential);
        assert_eq!(cfg.description, "Full-text search");
    }

    #[test]
    fn test_recovery_progress() {
        let mut rd = RecoveryDetector::new(4);
        rd.check(true);
        rd.check(true);
        assert!((rd.progress_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_clear_all_overrides() {
        let mut mgr = DegradationManager::new().with_auto_adjust(false);
        mgr.set_override("a", true);
        mgr.set_override("b", false);
        assert_eq!(mgr.override_count(), 2);
        mgr.clear_all_overrides();
        assert_eq!(mgr.override_count(), 0);
    }

    #[test]
    fn test_degradation_ordinal() {
        assert_eq!(DegradationLevel::Normal.ordinal(), 0);
        assert_eq!(DegradationLevel::Mild.ordinal(), 1);
        assert_eq!(DegradationLevel::Moderate.ordinal(), 2);
        assert_eq!(DegradationLevel::Severe.ordinal(), 3);
        assert_eq!(DegradationLevel::Critical.ordinal(), 4);
    }
}
