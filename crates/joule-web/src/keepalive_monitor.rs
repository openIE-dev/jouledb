//! Keepalive/heartbeat monitoring.
//!
//! Tracks heartbeat state per connection with configurable intervals
//! and timeouts. Detects dead connections, computes health scores,
//! supports adaptive interval based on connection activity, and provides
//! a grace period after timeout before disconnect.

use std::collections::BTreeMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Keepalive domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeepaliveError {
    /// Connection not found.
    ConnectionNotFound(u64),
    /// Duplicate connection.
    DuplicateConnection(u64),
    /// Connection already timed out.
    AlreadyTimedOut(u64),
    /// Connection already disconnected.
    AlreadyDisconnected(u64),
}

impl fmt::Display for KeepaliveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionNotFound(id) => write!(f, "connection not found: {id}"),
            Self::DuplicateConnection(id) => write!(f, "duplicate connection: {id}"),
            Self::AlreadyTimedOut(id) => write!(f, "connection {id} already timed out"),
            Self::AlreadyDisconnected(id) => write!(f, "connection {id} already disconnected"),
        }
    }
}

impl std::error::Error for KeepaliveError {}

// ── Connection Health ───────────────────────────────────────────

/// Health status of a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Connection is healthy.
    Healthy,
    /// Heartbeats are being missed.
    Degraded,
    /// Timeout reached — connection presumed dead.
    TimedOut,
    /// Grace period expired — ready for disconnect.
    Dead,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Degraded => write!(f, "degraded"),
            Self::TimedOut => write!(f, "timed_out"),
            Self::Dead => write!(f, "dead"),
        }
    }
}

// ── Connection State ────────────────────────────────────────────

/// Per-connection keepalive state.
#[derive(Debug, Clone)]
struct ConnectionState {
    last_heartbeat_tick: u64,
    last_activity_tick: u64,
    heartbeats_sent: u64,
    heartbeats_received: u64,
    missed_beats: u64,
    timed_out_at: Option<u64>,
    disconnected: bool,
    effective_interval: u64,
}

impl ConnectionState {
    fn new(tick: u64, interval: u64) -> Self {
        Self {
            last_heartbeat_tick: tick,
            last_activity_tick: tick,
            heartbeats_sent: 0,
            heartbeats_received: 0,
            missed_beats: 0,
            timed_out_at: None,
            disconnected: false,
            effective_interval: interval,
        }
    }
}

// ── Keepalive Config ────────────────────────────────────────────

/// Configuration for keepalive monitoring.
#[derive(Debug, Clone)]
pub struct KeepaliveConfig {
    pub interval_ms: u64,
    pub timeout_ms: u64,
    pub grace_period_ms: u64,
    pub adaptive: bool,
    pub min_interval_ms: u64,
    pub max_interval_ms: u64,
    pub activity_boost_factor: f64,
}

impl Default for KeepaliveConfig {
    fn default() -> Self {
        Self {
            interval_ms: 5000,
            timeout_ms: 15000,
            grace_period_ms: 5000,
            adaptive: false,
            min_interval_ms: 1000,
            max_interval_ms: 30000,
            activity_boost_factor: 0.5,
        }
    }
}

impl KeepaliveConfig {
    pub fn with_interval(mut self, ms: u64) -> Self {
        self.interval_ms = ms;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_grace_period(mut self, ms: u64) -> Self {
        self.grace_period_ms = ms;
        self
    }

    pub fn with_adaptive(mut self, enable: bool) -> Self {
        self.adaptive = enable;
        self
    }
}

// ── Keepalive Stats ─────────────────────────────────────────────

/// Aggregate keepalive statistics.
#[derive(Debug, Clone, Default)]
pub struct KeepaliveStats {
    pub total_heartbeats_sent: u64,
    pub total_heartbeats_received: u64,
    pub total_missed_beats: u64,
    pub total_timeouts: u64,
    pub total_disconnects: u64,
}

impl KeepaliveStats {
    pub fn response_rate(&self) -> f64 {
        if self.total_heartbeats_sent == 0 {
            return 1.0;
        }
        self.total_heartbeats_received as f64 / self.total_heartbeats_sent as f64
    }
}

impl fmt::Display for KeepaliveStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "sent={} recv={} missed={} timeouts={} disconnects={}",
            self.total_heartbeats_sent,
            self.total_heartbeats_received,
            self.total_missed_beats,
            self.total_timeouts,
            self.total_disconnects,
        )
    }
}

// ── Keepalive Monitor ───────────────────────────────────────────

/// Monitors connection liveness via heartbeats.
pub struct KeepaliveMonitor {
    config: KeepaliveConfig,
    connections: BTreeMap<u64, ConnectionState>,
    current_tick: u64,
    stats: KeepaliveStats,
}

impl KeepaliveMonitor {
    pub fn new(config: KeepaliveConfig) -> Self {
        Self {
            config,
            connections: BTreeMap::new(),
            current_tick: 0,
            stats: KeepaliveStats::default(),
        }
    }

    /// Advance the monitor clock.
    pub fn tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    /// Register a connection.
    pub fn register(&mut self, connection_id: u64) -> Result<(), KeepaliveError> {
        if self.connections.contains_key(&connection_id) {
            return Err(KeepaliveError::DuplicateConnection(connection_id));
        }
        self.connections.insert(
            connection_id,
            ConnectionState::new(self.current_tick, self.config.interval_ms),
        );
        Ok(())
    }

    /// Unregister a connection.
    pub fn unregister(&mut self, connection_id: u64) -> Result<(), KeepaliveError> {
        self.connections.remove(&connection_id)
            .ok_or(KeepaliveError::ConnectionNotFound(connection_id))?;
        Ok(())
    }

    /// Record that a heartbeat was sent to a connection.
    pub fn send_heartbeat(&mut self, connection_id: u64) -> Result<(), KeepaliveError> {
        let state = self.connections.get_mut(&connection_id)
            .ok_or(KeepaliveError::ConnectionNotFound(connection_id))?;
        if state.disconnected {
            return Err(KeepaliveError::AlreadyDisconnected(connection_id));
        }
        state.heartbeats_sent += 1;
        self.stats.total_heartbeats_sent += 1;
        Ok(())
    }

    /// Record that a heartbeat response was received from a connection.
    pub fn receive_heartbeat(&mut self, connection_id: u64) -> Result<(), KeepaliveError> {
        let state = self.connections.get_mut(&connection_id)
            .ok_or(KeepaliveError::ConnectionNotFound(connection_id))?;
        if state.disconnected {
            return Err(KeepaliveError::AlreadyDisconnected(connection_id));
        }
        state.last_heartbeat_tick = self.current_tick;
        state.heartbeats_received += 1;
        state.timed_out_at = None; // Clear timeout on successful heartbeat.
        self.stats.total_heartbeats_received += 1;
        Ok(())
    }

    /// Record general activity on a connection (data transfer, etc.).
    pub fn record_activity(&mut self, connection_id: u64) -> Result<(), KeepaliveError> {
        let config = &self.config;
        let tick = self.current_tick;
        let state = self.connections.get_mut(&connection_id)
            .ok_or(KeepaliveError::ConnectionNotFound(connection_id))?;
        state.last_activity_tick = tick;

        // Adaptive interval: more active connections get longer intervals.
        if config.adaptive {
            state.effective_interval = ((config.interval_ms as f64
                * (1.0 + config.activity_boost_factor)) as u64)
                .clamp(config.min_interval_ms, config.max_interval_ms);
        }

        Ok(())
    }

    /// Check all connections and return IDs that need heartbeats sent.
    pub fn connections_needing_heartbeat(&self) -> Vec<u64> {
        self.connections.iter()
            .filter(|(_, state)| {
                !state.disconnected
                    && state.timed_out_at.is_none()
                    && self.current_tick.saturating_sub(state.last_heartbeat_tick) >= state.effective_interval
            })
            .map(|(&id, _)| id)
            .collect()
    }

    /// Check for timeouts and update state. Returns newly timed-out connection IDs.
    pub fn check_timeouts(&mut self) -> Vec<u64> {
        let timeout = self.config.timeout_ms;
        let tick = self.current_tick;
        let mut newly_timed_out = Vec::new();

        for (&id, state) in &mut self.connections {
            if state.disconnected || state.timed_out_at.is_some() {
                continue;
            }
            let since_last = tick.saturating_sub(state.last_heartbeat_tick);
            if since_last >= timeout {
                state.timed_out_at = Some(tick);
                state.missed_beats += 1;
                newly_timed_out.push(id);
                self.stats.total_timeouts += 1;
                self.stats.total_missed_beats += 1;
            }
        }

        newly_timed_out
    }

    /// Check for connections past the grace period. Returns IDs ready for disconnect.
    pub fn check_grace_period(&mut self) -> Vec<u64> {
        let grace = self.config.grace_period_ms;
        let tick = self.current_tick;
        let mut to_disconnect = Vec::new();

        for (&id, state) in &mut self.connections {
            if state.disconnected {
                continue;
            }
            if let Some(timeout_tick) = state.timed_out_at {
                if tick.saturating_sub(timeout_tick) >= grace {
                    state.disconnected = true;
                    to_disconnect.push(id);
                    self.stats.total_disconnects += 1;
                }
            }
        }

        to_disconnect
    }

    /// Get the health status of a connection.
    pub fn health(&self, connection_id: u64) -> Result<HealthStatus, KeepaliveError> {
        let state = self.connections.get(&connection_id)
            .ok_or(KeepaliveError::ConnectionNotFound(connection_id))?;

        if state.disconnected {
            return Ok(HealthStatus::Dead);
        }
        if let Some(timeout_tick) = state.timed_out_at {
            let since_timeout = self.current_tick.saturating_sub(timeout_tick);
            if since_timeout >= self.config.grace_period_ms {
                return Ok(HealthStatus::Dead);
            }
            return Ok(HealthStatus::TimedOut);
        }

        let since_last = self.current_tick.saturating_sub(state.last_heartbeat_tick);
        if since_last > state.effective_interval * 2 {
            Ok(HealthStatus::Degraded)
        } else {
            Ok(HealthStatus::Healthy)
        }
    }

    /// Compute a health score for a connection (0.0 = dead, 1.0 = perfect).
    pub fn health_score(&self, connection_id: u64) -> Result<f64, KeepaliveError> {
        let state = self.connections.get(&connection_id)
            .ok_or(KeepaliveError::ConnectionNotFound(connection_id))?;

        if state.disconnected {
            return Ok(0.0);
        }

        let since_last = self.current_tick.saturating_sub(state.last_heartbeat_tick) as f64;
        let timeout = self.config.timeout_ms as f64;

        if timeout <= 0.0 {
            return Ok(1.0);
        }

        let score = 1.0 - (since_last / timeout).min(1.0);
        Ok(score)
    }

    /// Number of tracked connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Number of healthy connections (not timed out or disconnected).
    pub fn healthy_count(&self) -> usize {
        self.connections.values()
            .filter(|s| !s.disconnected && s.timed_out_at.is_none())
            .count()
    }

    /// Number of disconnected connections.
    pub fn disconnected_count(&self) -> usize {
        self.connections.values()
            .filter(|s| s.disconnected)
            .count()
    }

    /// Get statistics.
    pub fn stats(&self) -> &KeepaliveStats {
        &self.stats
    }
}

impl fmt::Display for KeepaliveMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KeepaliveMonitor(connections={}, healthy={}, disconnected={})",
            self.connection_count(),
            self.healthy_count(),
            self.disconnected_count(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_monitor() -> KeepaliveMonitor {
        KeepaliveMonitor::new(KeepaliveConfig::default())
    }

    #[test]
    fn register_connection() {
        let mut m = default_monitor();
        m.register(1).unwrap();
        assert_eq!(m.connection_count(), 1);
    }

    #[test]
    fn duplicate_registration_rejected() {
        let mut m = default_monitor();
        m.register(1).unwrap();
        let err = m.register(1).unwrap_err();
        assert!(matches!(err, KeepaliveError::DuplicateConnection(1)));
    }

    #[test]
    fn unregister_connection() {
        let mut m = default_monitor();
        m.register(1).unwrap();
        m.unregister(1).unwrap();
        assert_eq!(m.connection_count(), 0);
    }

    #[test]
    fn send_and_receive_heartbeat() {
        let mut m = default_monitor();
        m.register(1).unwrap();
        m.send_heartbeat(1).unwrap();
        m.receive_heartbeat(1).unwrap();
        assert_eq!(m.stats().total_heartbeats_sent, 1);
        assert_eq!(m.stats().total_heartbeats_received, 1);
    }

    #[test]
    fn healthy_connection() {
        let mut m = default_monitor();
        m.register(1).unwrap();
        assert_eq!(m.health(1).unwrap(), HealthStatus::Healthy);
    }

    #[test]
    fn degraded_after_missed_intervals() {
        let config = KeepaliveConfig::default().with_interval(100);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(250); // More than 2x interval without heartbeat.
        assert_eq!(m.health(1).unwrap(), HealthStatus::Degraded);
    }

    #[test]
    fn timeout_detection() {
        let config = KeepaliveConfig::default().with_timeout(200);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(100);
        let t1 = m.check_timeouts();
        assert!(t1.is_empty());
        m.tick(250);
        let t2 = m.check_timeouts();
        assert_eq!(t2, vec![1]);
    }

    #[test]
    fn timed_out_health_status() {
        let config = KeepaliveConfig::default().with_timeout(100);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(150);
        m.check_timeouts();
        assert_eq!(m.health(1).unwrap(), HealthStatus::TimedOut);
    }

    #[test]
    fn grace_period_then_dead() {
        let config = KeepaliveConfig::default()
            .with_timeout(100)
            .with_grace_period(100);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(150);
        m.check_timeouts(); // Timed out at tick 150.
        m.tick(200);
        let disconnected = m.check_grace_period();
        assert!(disconnected.is_empty()); // 50ms into 100ms grace — still alive.
        m.tick(260);
        let disconnected = m.check_grace_period();
        assert_eq!(disconnected, vec![1]); // 110ms past timeout > 100ms grace.
        assert_eq!(m.health(1).unwrap(), HealthStatus::Dead);
    }

    #[test]
    fn heartbeat_clears_timeout() {
        let config = KeepaliveConfig::default().with_timeout(100);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(150);
        m.check_timeouts();
        assert_eq!(m.health(1).unwrap(), HealthStatus::TimedOut);
        m.receive_heartbeat(1).unwrap();
        assert_eq!(m.health(1).unwrap(), HealthStatus::Healthy);
    }

    #[test]
    fn connections_needing_heartbeat() {
        let config = KeepaliveConfig::default().with_interval(100);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.register(2).unwrap();
        m.tick(50);
        assert!(m.connections_needing_heartbeat().is_empty());
        m.tick(110);
        let needing = m.connections_needing_heartbeat();
        assert_eq!(needing.len(), 2);
    }

    #[test]
    fn health_score_perfect() {
        let mut m = default_monitor();
        m.tick(0);
        m.register(1).unwrap();
        let score = m.health_score(1).unwrap();
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn health_score_degrades_over_time() {
        let config = KeepaliveConfig::default().with_timeout(1000);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(500);
        let score = m.health_score(1).unwrap();
        assert!((score - 0.5).abs() < 0.001);
    }

    #[test]
    fn health_score_zero_for_dead() {
        let config = KeepaliveConfig::default().with_timeout(100).with_grace_period(50);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(150);
        m.check_timeouts();
        m.tick(250);
        m.check_grace_period();
        assert!((m.health_score(1).unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn adaptive_interval() {
        let config = KeepaliveConfig::default()
            .with_interval(100)
            .with_adaptive(true);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.record_activity(1).unwrap();
        // With activity, effective interval should increase.
        m.tick(110);
        let needing = m.connections_needing_heartbeat();
        // Interval boosted, so may not need heartbeat yet at 110.
        assert!(needing.is_empty());
    }

    #[test]
    fn healthy_count() {
        let config = KeepaliveConfig::default().with_timeout(100);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.register(2).unwrap();
        assert_eq!(m.healthy_count(), 2);
        m.tick(150);
        m.check_timeouts();
        assert_eq!(m.healthy_count(), 0);
    }

    #[test]
    fn disconnected_count() {
        let config = KeepaliveConfig::default().with_timeout(100).with_grace_period(50);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(150);
        m.check_timeouts(); // Timed out at tick 150.
        m.tick(210); // 60ms past timeout > 50ms grace.
        m.check_grace_period();
        assert_eq!(m.disconnected_count(), 1);
    }

    #[test]
    fn stats_response_rate() {
        let mut stats = KeepaliveStats::default();
        assert_eq!(stats.response_rate(), 1.0);
        stats.total_heartbeats_sent = 10;
        stats.total_heartbeats_received = 8;
        assert!((stats.response_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn send_on_disconnected_fails() {
        let config = KeepaliveConfig::default().with_timeout(100).with_grace_period(0);
        let mut m = KeepaliveMonitor::new(config);
        m.tick(0);
        m.register(1).unwrap();
        m.tick(150);
        m.check_timeouts();
        m.check_grace_period();
        let err = m.send_heartbeat(1).unwrap_err();
        assert!(matches!(err, KeepaliveError::AlreadyDisconnected(1)));
    }

    #[test]
    fn monitor_display() {
        let m = default_monitor();
        let s = format!("{m}");
        assert!(s.contains("KeepaliveMonitor"));
    }

    #[test]
    fn config_builder() {
        let config = KeepaliveConfig::default()
            .with_interval(2000)
            .with_timeout(10000)
            .with_grace_period(3000)
            .with_adaptive(true);
        assert_eq!(config.interval_ms, 2000);
        assert_eq!(config.timeout_ms, 10000);
        assert_eq!(config.grace_period_ms, 3000);
        assert!(config.adaptive);
    }
}
