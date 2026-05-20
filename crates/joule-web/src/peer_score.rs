//! Peer reputation and scoring system — composite scoring from uptime, latency,
//! reliability, and bandwidth metrics, score decay, penalties and rewards,
//! configurable metric weights, trust levels, and peer ranking.

use std::collections::HashMap;
use std::fmt;

// ── TrustLevel ──────────────────────────────────────────────────────────────

/// Trust level derived from a composite peer score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    Untrusted,
    Basic,
    Trusted,
    Preferred,
}

impl fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrustLevel::Untrusted => write!(f, "Untrusted"),
            TrustLevel::Basic => write!(f, "Basic"),
            TrustLevel::Trusted => write!(f, "Trusted"),
            TrustLevel::Preferred => write!(f, "Preferred"),
        }
    }
}

// ── ScoreThresholds ─────────────────────────────────────────────────────────

/// Thresholds for mapping composite score to trust levels.
#[derive(Debug, Clone)]
pub struct ScoreThresholds {
    pub basic: f64,
    pub trusted: f64,
    pub preferred: f64,
}

impl Default for ScoreThresholds {
    fn default() -> Self {
        Self { basic: 25.0, trusted: 50.0, preferred: 75.0 }
    }
}

impl ScoreThresholds {
    pub fn level_for(&self, score: f64) -> TrustLevel {
        if score >= self.preferred {
            TrustLevel::Preferred
        } else if score >= self.trusted {
            TrustLevel::Trusted
        } else if score >= self.basic {
            TrustLevel::Basic
        } else {
            TrustLevel::Untrusted
        }
    }
}

// ── MetricWeights ───────────────────────────────────────────────────────────

/// Configurable weights for each metric in the composite score.
#[derive(Debug, Clone)]
pub struct MetricWeights {
    pub uptime: f64,
    pub latency: f64,
    pub reliability: f64,
    pub bandwidth: f64,
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self { uptime: 0.25, latency: 0.25, reliability: 0.30, bandwidth: 0.20 }
    }
}

impl MetricWeights {
    /// Total of all weights.
    pub fn total(&self) -> f64 {
        self.uptime + self.latency + self.reliability + self.bandwidth
    }
}

// ── PeerMetrics ─────────────────────────────────────────────────────────────

/// Raw metrics for a peer.
#[derive(Debug, Clone)]
pub struct PeerMetrics {
    /// Uptime ratio (0.0 to 1.0).
    pub uptime: f64,
    /// Average latency in ms (lower is better). Normalized inversely.
    pub latency_ms: f64,
    /// Reliability ratio — fraction of successful operations (0.0 to 1.0).
    pub reliability: f64,
    /// Bandwidth score (0.0 to 100.0).
    pub bandwidth: f64,
}

impl PeerMetrics {
    pub fn new() -> Self {
        Self { uptime: 0.0, latency_ms: 1000.0, reliability: 0.0, bandwidth: 0.0 }
    }

    /// Normalize latency to a 0-100 score (lower latency = higher score).
    pub fn latency_score(&self) -> f64 {
        // 0ms -> 100, 1000ms -> 0, clamped
        (100.0 - (self.latency_ms / 10.0)).clamp(0.0, 100.0)
    }
}

impl Default for PeerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ── ScoreRecord ─────────────────────────────────────────────────────────────

/// A single point in the score history.
#[derive(Debug, Clone)]
pub struct ScoreRecord {
    pub score: f64,
    pub tick: u64,
}

// ── PeerScore ───────────────────────────────────────────────────────────────

/// Score state for a single peer.
#[derive(Debug, Clone)]
pub struct PeerScore {
    pub peer_id: String,
    pub metrics: PeerMetrics,
    pub composite: f64,
    pub trust_level: TrustLevel,
    pub penalties: f64,
    pub rewards: f64,
    pub history: Vec<ScoreRecord>,
    pub last_updated: u64,
}

impl PeerScore {
    pub fn new(peer_id: impl Into<String>) -> Self {
        Self {
            peer_id: peer_id.into(),
            metrics: PeerMetrics::new(),
            composite: 0.0,
            trust_level: TrustLevel::Untrusted,
            penalties: 0.0,
            rewards: 0.0,
            history: Vec::new(),
            last_updated: 0,
        }
    }
}

impl fmt::Display for PeerScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PeerScore({}: {:.1}, trust={})",
            self.peer_id, self.composite, self.trust_level,
        )
    }
}

// ── ReputationManager ───────────────────────────────────────────────────────

/// Manages peer reputation scores across the network.
pub struct ReputationManager {
    scores: HashMap<String, PeerScore>,
    weights: MetricWeights,
    thresholds: ScoreThresholds,
    decay_rate: f64,
    penalty_dropped_conn: f64,
    penalty_slow_response: f64,
    reward_good_relay: f64,
    reward_uptime_bonus: f64,
    max_history: usize,
    current_tick: u64,
}

impl ReputationManager {
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            weights: MetricWeights::default(),
            thresholds: ScoreThresholds::default(),
            decay_rate: 0.99,
            penalty_dropped_conn: 5.0,
            penalty_slow_response: 2.0,
            reward_good_relay: 3.0,
            reward_uptime_bonus: 1.0,
            max_history: 100,
            current_tick: 0,
        }
    }

    pub fn with_weights(mut self, weights: MetricWeights) -> Self {
        self.weights = weights;
        self
    }

    pub fn with_thresholds(mut self, thresholds: ScoreThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    pub fn with_decay_rate(mut self, rate: f64) -> Self {
        self.decay_rate = rate;
        self
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Register a peer (initializes with default score).
    pub fn register_peer(&mut self, peer_id: impl Into<String>) {
        let pid = peer_id.into();
        if !self.scores.contains_key(&pid) {
            self.scores.insert(pid.clone(), PeerScore::new(pid));
        }
    }

    /// Update raw metrics for a peer and recompute composite score.
    pub fn update_metrics(&mut self, peer_id: &str, metrics: PeerMetrics) {
        let entry = self.scores.entry(peer_id.to_string()).or_insert_with(|| PeerScore::new(peer_id));
        entry.metrics = metrics;
        self.recompute(peer_id);
    }

    /// Apply a penalty for bad behavior (e.g., dropped connection).
    pub fn penalize_dropped_connection(&mut self, peer_id: &str) {
        if let Some(score) = self.scores.get_mut(peer_id) {
            score.penalties += self.penalty_dropped_conn;
            self.recompute(peer_id);
        }
    }

    /// Apply a penalty for slow response.
    pub fn penalize_slow_response(&mut self, peer_id: &str) {
        if let Some(score) = self.scores.get_mut(peer_id) {
            score.penalties += self.penalty_slow_response;
            self.recompute(peer_id);
        }
    }

    /// Reward good relay behavior.
    pub fn reward_good_relay(&mut self, peer_id: &str) {
        if let Some(score) = self.scores.get_mut(peer_id) {
            score.rewards += self.reward_good_relay;
            self.recompute(peer_id);
        }
    }

    /// Reward uptime.
    pub fn reward_uptime(&mut self, peer_id: &str) {
        if let Some(score) = self.scores.get_mut(peer_id) {
            score.rewards += self.reward_uptime_bonus;
            self.recompute(peer_id);
        }
    }

    /// Apply score decay to all peers.
    pub fn apply_decay(&mut self) {
        let peers: Vec<String> = self.scores.keys().cloned().collect();
        for pid in &peers {
            if let Some(score) = self.scores.get_mut(pid) {
                score.penalties *= self.decay_rate;
                score.rewards *= self.decay_rate;
            }
            self.recompute(pid);
        }
    }

    fn recompute(&mut self, peer_id: &str) {
        // Need to clone out temporarily to avoid borrow issues
        let (composite, trust_level) = {
            let score = match self.scores.get(peer_id) {
                Some(s) => s,
                None => return,
            };
            let m = &score.metrics;
            let w = &self.weights;
            let total_w = w.total();
            if total_w == 0.0 {
                (0.0, TrustLevel::Untrusted)
            } else {
                let raw = (m.uptime * 100.0 * w.uptime
                    + m.latency_score() * w.latency
                    + m.reliability * 100.0 * w.reliability
                    + m.bandwidth * w.bandwidth)
                    / total_w;
                let adjusted = (raw + score.rewards - score.penalties).clamp(0.0, 100.0);
                let trust = self.thresholds.level_for(adjusted);
                (adjusted, trust)
            }
        };

        if let Some(score) = self.scores.get_mut(peer_id) {
            score.composite = composite;
            score.trust_level = trust_level;
            score.last_updated = self.current_tick;
            score.history.push(ScoreRecord {
                score: composite,
                tick: self.current_tick,
            });
            if score.history.len() > self.max_history {
                score.history.remove(0);
            }
        }
    }

    /// Get a peer's current score.
    pub fn get_score(&self, peer_id: &str) -> Option<&PeerScore> {
        self.scores.get(peer_id)
    }

    /// Get the trust level for a peer.
    pub fn trust_level(&self, peer_id: &str) -> Option<TrustLevel> {
        self.scores.get(peer_id).map(|s| s.trust_level)
    }

    /// Rank all peers by composite score (highest first).
    pub fn rank_peers(&self) -> Vec<(&str, f64)> {
        let mut ranked: Vec<_> = self
            .scores
            .iter()
            .map(|(id, s)| (id.as_str(), s.composite))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Peers at or above a given trust level.
    pub fn peers_at_level(&self, min_level: TrustLevel) -> Vec<&str> {
        self.scores
            .iter()
            .filter(|(_, s)| s.trust_level >= min_level)
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.scores.len()
    }
}

impl Default for ReputationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn good_metrics() -> PeerMetrics {
        PeerMetrics { uptime: 0.99, latency_ms: 50.0, reliability: 0.98, bandwidth: 80.0 }
    }

    fn poor_metrics() -> PeerMetrics {
        PeerMetrics { uptime: 0.3, latency_ms: 800.0, reliability: 0.4, bandwidth: 10.0 }
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Preferred > TrustLevel::Trusted);
        assert!(TrustLevel::Trusted > TrustLevel::Basic);
        assert!(TrustLevel::Basic > TrustLevel::Untrusted);
    }

    #[test]
    fn test_trust_level_display() {
        assert_eq!(format!("{}", TrustLevel::Preferred), "Preferred");
        assert_eq!(format!("{}", TrustLevel::Untrusted), "Untrusted");
    }

    #[test]
    fn test_score_thresholds() {
        let t = ScoreThresholds::default();
        assert_eq!(t.level_for(10.0), TrustLevel::Untrusted);
        assert_eq!(t.level_for(30.0), TrustLevel::Basic);
        assert_eq!(t.level_for(60.0), TrustLevel::Trusted);
        assert_eq!(t.level_for(80.0), TrustLevel::Preferred);
    }

    #[test]
    fn test_latency_score() {
        let m = PeerMetrics { latency_ms: 0.0, ..Default::default() };
        assert!((m.latency_score() - 100.0).abs() < 0.001);
        let m2 = PeerMetrics { latency_ms: 1000.0, ..Default::default() };
        assert!((m2.latency_score() - 0.0).abs() < 0.001);
        let m3 = PeerMetrics { latency_ms: 500.0, ..Default::default() };
        assert!((m3.latency_score() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_register_peer() {
        let mut rm = ReputationManager::new();
        rm.register_peer("alice");
        assert_eq!(rm.peer_count(), 1);
        assert!(rm.get_score("alice").is_some());
    }

    #[test]
    fn test_update_metrics_good() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("alice", good_metrics());
        let score = rm.get_score("alice").unwrap();
        assert!(score.composite > 50.0);
        assert!(score.trust_level >= TrustLevel::Trusted);
    }

    #[test]
    fn test_update_metrics_poor() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("bob", poor_metrics());
        let score = rm.get_score("bob").unwrap();
        assert!(score.composite < 50.0);
    }

    #[test]
    fn test_penalty_drops_score() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("alice", good_metrics());
        let before = rm.get_score("alice").unwrap().composite;
        rm.penalize_dropped_connection("alice");
        let after = rm.get_score("alice").unwrap().composite;
        assert!(after < before);
    }

    #[test]
    fn test_reward_raises_score() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("alice", PeerMetrics { uptime: 0.5, latency_ms: 200.0, reliability: 0.5, bandwidth: 50.0 });
        let before = rm.get_score("alice").unwrap().composite;
        rm.reward_good_relay("alice");
        let after = rm.get_score("alice").unwrap().composite;
        assert!(after > before);
    }

    #[test]
    fn test_decay_reduces_penalties() {
        let mut rm = ReputationManager::new().with_decay_rate(0.5);
        rm.update_metrics("a", good_metrics());
        rm.penalize_dropped_connection("a");
        let pen_before = rm.get_score("a").unwrap().penalties;
        rm.apply_decay();
        let pen_after = rm.get_score("a").unwrap().penalties;
        assert!(pen_after < pen_before);
    }

    #[test]
    fn test_rank_peers() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("good", good_metrics());
        rm.update_metrics("poor", poor_metrics());
        let ranked = rm.rank_peers();
        assert_eq!(ranked[0].0, "good");
        assert_eq!(ranked[1].0, "poor");
    }

    #[test]
    fn test_peers_at_level() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("good", good_metrics());
        rm.update_metrics("poor", poor_metrics());
        let trusted = rm.peers_at_level(TrustLevel::Trusted);
        assert!(trusted.contains(&"good"));
        assert!(!trusted.contains(&"poor"));
    }

    #[test]
    fn test_score_history() {
        let mut rm = ReputationManager::new();
        rm.tick(1);
        rm.update_metrics("a", good_metrics());
        rm.tick(2);
        rm.reward_good_relay("a");
        let score = rm.get_score("a").unwrap();
        assert!(score.history.len() >= 2);
    }

    #[test]
    fn test_score_clamped() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("a", good_metrics());
        // Add excessive rewards
        for _ in 0..100 {
            rm.reward_good_relay("a");
        }
        let score = rm.get_score("a").unwrap();
        assert!(score.composite <= 100.0);
    }

    #[test]
    fn test_score_floor_zero() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("a", poor_metrics());
        for _ in 0..100 {
            rm.penalize_dropped_connection("a");
        }
        let score = rm.get_score("a").unwrap();
        assert!(score.composite >= 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let w = MetricWeights { uptime: 1.0, latency: 0.0, reliability: 0.0, bandwidth: 0.0 };
        let mut rm = ReputationManager::new().with_weights(w);
        rm.update_metrics("a", PeerMetrics { uptime: 1.0, latency_ms: 999.0, reliability: 0.0, bandwidth: 0.0 });
        let score = rm.get_score("a").unwrap();
        assert!((score.composite - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_custom_thresholds() {
        let t = ScoreThresholds { basic: 10.0, trusted: 20.0, preferred: 90.0 };
        let mut rm = ReputationManager::new().with_thresholds(t);
        rm.update_metrics("a", good_metrics());
        // Score should be high but maybe not 90+
        let trust = rm.trust_level("a").unwrap();
        assert!(trust >= TrustLevel::Trusted);
    }

    #[test]
    fn test_peer_score_display() {
        let s = PeerScore::new("test");
        let display = format!("{}", s);
        assert!(display.contains("test"));
        assert!(display.contains("Untrusted"));
    }

    #[test]
    fn test_metric_weights_total() {
        let w = MetricWeights::default();
        assert!((w.total() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_penalize_slow_response() {
        let mut rm = ReputationManager::new();
        rm.update_metrics("a", good_metrics());
        let before = rm.get_score("a").unwrap().composite;
        rm.penalize_slow_response("a");
        let after = rm.get_score("a").unwrap().composite;
        assert!(after < before);
    }
}
