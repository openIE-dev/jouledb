//! Connection migration protocol.
//!
//! Manages persistent [`ConnectionId`]s across address changes. The
//! [`MigrationManager`] tracks active connections, initiates migration
//! proposals, performs path validation, handles seamless handover,
//! records migration history, supports concurrent migrations, and
//! rolls back on failure.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Migration domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationError {
    /// Connection not found.
    ConnectionNotFound(u64),
    /// Duplicate connection ID.
    DuplicateConnection(u64),
    /// No active migration to complete or rollback.
    NoActiveMigration(u64),
    /// Migration already in progress.
    MigrationInProgress(u64),
    /// Path validation failed.
    PathValidationFailed { connection_id: u64, reason: String },
    /// Rollback failed.
    RollbackFailed { connection_id: u64, reason: String },
    /// Connection is closed.
    ConnectionClosed(u64),
}

impl fmt::Display for MigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionNotFound(id) => write!(f, "connection not found: {id}"),
            Self::DuplicateConnection(id) => write!(f, "duplicate connection: {id}"),
            Self::NoActiveMigration(id) => write!(f, "no active migration for connection {id}"),
            Self::MigrationInProgress(id) => write!(f, "migration already in progress for {id}"),
            Self::PathValidationFailed { connection_id, reason } => {
                write!(f, "path validation failed for {connection_id}: {reason}")
            }
            Self::RollbackFailed { connection_id, reason } => {
                write!(f, "rollback failed for {connection_id}: {reason}")
            }
            Self::ConnectionClosed(id) => write!(f, "connection {id} is closed"),
        }
    }
}

impl std::error::Error for MigrationError {}

// ── Address ─────────────────────────────────────────────────────

/// A network address (IP + port).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NetAddress {
    pub ip: String,
    pub port: u16,
}

impl NetAddress {
    pub fn new(ip: impl Into<String>, port: u16) -> Self {
        Self { ip: ip.into(), port }
    }
}

impl fmt::Display for NetAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.ip, self.port)
    }
}

// ── Migration State ─────────────────────────────────────────────

/// State of an in-progress migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationState {
    /// Migration proposed but not yet validated.
    Proposed,
    /// Path is being probed.
    Probing,
    /// Path validated, ready for handover.
    Validated,
    /// Migration completed successfully.
    Completed,
    /// Migration failed and was rolled back.
    RolledBack,
}

impl fmt::Display for MigrationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proposed => write!(f, "Proposed"),
            Self::Probing => write!(f, "Probing"),
            Self::Validated => write!(f, "Validated"),
            Self::Completed => write!(f, "Completed"),
            Self::RolledBack => write!(f, "RolledBack"),
        }
    }
}

// ── Migration Record ────────────────────────────────────────────

/// A record of a single migration attempt.
#[derive(Debug, Clone)]
pub struct MigrationRecord {
    pub from_address: NetAddress,
    pub to_address: NetAddress,
    pub state: MigrationState,
    pub started_tick: u64,
    pub completed_tick: Option<u64>,
    pub probe_count: u32,
}

impl fmt::Display for MigrationRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} -> {} ({})",
            self.from_address, self.to_address, self.state,
        )
    }
}

// ── Connection Entry ────────────────────────────────────────────

/// A tracked connection.
#[derive(Debug, Clone)]
struct ConnectionEntry {
    id: u64,
    current_address: NetAddress,
    active_migration: Option<MigrationRecord>,
    history: VecDeque<MigrationRecord>,
    closed: bool,
}

// ── Migration Stats ─────────────────────────────────────────────

/// Aggregate migration statistics.
#[derive(Debug, Clone, Default)]
pub struct MigrationStats {
    pub total_migrations: u64,
    pub successful: u64,
    pub failed: u64,
    pub rollbacks: u64,
    pub probes_sent: u64,
}

impl MigrationStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_migrations == 0 {
            return 0.0;
        }
        self.successful as f64 / self.total_migrations as f64
    }
}

impl fmt::Display for MigrationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "total={} ok={} fail={} rollback={} probes={}",
            self.total_migrations,
            self.successful,
            self.failed,
            self.rollbacks,
            self.probes_sent,
        )
    }
}

// ── Manager Config ──────────────────────────────────────────────

/// Configuration for the migration manager.
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    pub max_connections: usize,
    pub max_history: usize,
    pub probes_required: u32,
    pub probe_timeout_ticks: u64,
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            max_connections: 10_000,
            max_history: 32,
            probes_required: 3,
            probe_timeout_ticks: 100,
        }
    }
}

impl MigrationConfig {
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    pub fn with_probes_required(mut self, n: u32) -> Self {
        self.probes_required = n;
        self
    }

    pub fn with_max_history(mut self, n: usize) -> Self {
        self.max_history = n;
        self
    }
}

// ── Migration Manager ───────────────────────────────────────────

/// Manages connection migrations across address changes.
pub struct MigrationManager {
    config: MigrationConfig,
    connections: BTreeMap<u64, ConnectionEntry>,
    stats: MigrationStats,
    current_tick: u64,
}

impl MigrationManager {
    pub fn new(config: MigrationConfig) -> Self {
        Self {
            config,
            connections: BTreeMap::new(),
            stats: MigrationStats::default(),
            current_tick: 0,
        }
    }

    /// Advance the tick clock.
    pub fn tick(&mut self, tick: u64) {
        self.current_tick = tick;
    }

    /// Register a connection.
    pub fn register(&mut self, connection_id: u64, address: NetAddress) -> Result<(), MigrationError> {
        if self.connections.contains_key(&connection_id) {
            return Err(MigrationError::DuplicateConnection(connection_id));
        }
        self.connections.insert(connection_id, ConnectionEntry {
            id: connection_id,
            current_address: address,
            active_migration: None,
            history: VecDeque::new(),
            closed: false,
        });
        Ok(())
    }

    /// Unregister (close) a connection.
    pub fn close(&mut self, connection_id: u64) -> Result<(), MigrationError> {
        let entry = self.connections.get_mut(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        entry.closed = true;
        Ok(())
    }

    /// Initiate a migration to a new address.
    pub fn initiate(
        &mut self,
        connection_id: u64,
        new_address: NetAddress,
    ) -> Result<(), MigrationError> {
        let entry = self.connections.get_mut(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        if entry.closed {
            return Err(MigrationError::ConnectionClosed(connection_id));
        }
        if entry.active_migration.is_some() {
            return Err(MigrationError::MigrationInProgress(connection_id));
        }

        entry.active_migration = Some(MigrationRecord {
            from_address: entry.current_address.clone(),
            to_address: new_address,
            state: MigrationState::Proposed,
            started_tick: self.current_tick,
            completed_tick: None,
            probe_count: 0,
        });

        self.stats.total_migrations += 1;
        Ok(())
    }

    /// Send a probe for path validation.
    pub fn probe(&mut self, connection_id: u64) -> Result<MigrationState, MigrationError> {
        let probes_required = self.config.probes_required;
        let entry = self.connections.get_mut(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        let migration = entry.active_migration.as_mut()
            .ok_or(MigrationError::NoActiveMigration(connection_id))?;

        migration.state = MigrationState::Probing;
        migration.probe_count += 1;
        self.stats.probes_sent += 1;

        if migration.probe_count >= probes_required {
            migration.state = MigrationState::Validated;
        }

        Ok(migration.state.clone())
    }

    /// Complete a validated migration (handover).
    pub fn complete(&mut self, connection_id: u64) -> Result<(), MigrationError> {
        let max_history = self.config.max_history;
        let tick = self.current_tick;
        let entry = self.connections.get_mut(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        let mut migration = entry.active_migration.take()
            .ok_or(MigrationError::NoActiveMigration(connection_id))?;

        if migration.state != MigrationState::Validated {
            // Not validated — put it back and return error.
            let state = migration.state.clone();
            entry.active_migration = Some(migration);
            return Err(MigrationError::PathValidationFailed {
                connection_id,
                reason: format!("state is {state}, expected Validated"),
            });
        }

        // Handover: update address.
        entry.current_address = migration.to_address.clone();
        migration.state = MigrationState::Completed;
        migration.completed_tick = Some(tick);

        entry.history.push_back(migration);
        while entry.history.len() > max_history {
            entry.history.pop_front();
        }

        self.stats.successful += 1;
        Ok(())
    }

    /// Rollback an in-progress migration.
    pub fn rollback(&mut self, connection_id: u64) -> Result<(), MigrationError> {
        let max_history = self.config.max_history;
        let tick = self.current_tick;
        let entry = self.connections.get_mut(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        let mut migration = entry.active_migration.take()
            .ok_or(MigrationError::NoActiveMigration(connection_id))?;

        migration.state = MigrationState::RolledBack;
        migration.completed_tick = Some(tick);

        entry.history.push_back(migration);
        while entry.history.len() > max_history {
            entry.history.pop_front();
        }

        self.stats.failed += 1;
        self.stats.rollbacks += 1;
        Ok(())
    }

    /// Get the current address for a connection.
    pub fn address(&self, connection_id: u64) -> Result<&NetAddress, MigrationError> {
        let entry = self.connections.get(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        Ok(&entry.current_address)
    }

    /// Whether a migration is in progress.
    pub fn has_active_migration(&self, connection_id: u64) -> Result<bool, MigrationError> {
        let entry = self.connections.get(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        Ok(entry.active_migration.is_some())
    }

    /// Get migration history for a connection.
    pub fn history(&self, connection_id: u64) -> Result<&VecDeque<MigrationRecord>, MigrationError> {
        let entry = self.connections.get(&connection_id)
            .ok_or(MigrationError::ConnectionNotFound(connection_id))?;
        Ok(&entry.history)
    }

    /// Number of tracked connections.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> &MigrationStats {
        &self.stats
    }
}

impl fmt::Display for MigrationManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MigrationManager(connections={}, {})",
            self.connections.len(),
            self.stats,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_mgr() -> MigrationManager {
        MigrationManager::new(MigrationConfig::default())
    }

    fn addr(ip: &str, port: u16) -> NetAddress {
        NetAddress::new(ip, port)
    }

    #[test]
    fn register_connection() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        assert_eq!(mgr.connection_count(), 1);
    }

    #[test]
    fn duplicate_registration_rejected() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        let err = mgr.register(1, addr("10.0.0.2", 4433)).unwrap_err();
        assert!(matches!(err, MigrationError::DuplicateConnection(1)));
    }

    #[test]
    fn initiate_migration() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("192.168.1.1", 5000)).unwrap();
        assert!(mgr.has_active_migration(1).unwrap());
    }

    #[test]
    fn initiate_on_closed_connection() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.close(1).unwrap();
        let err = mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap_err();
        assert!(matches!(err, MigrationError::ConnectionClosed(1)));
    }

    #[test]
    fn double_initiate_rejected() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        let err = mgr.initiate(1, addr("10.0.0.3", 4433)).unwrap_err();
        assert!(matches!(err, MigrationError::MigrationInProgress(1)));
    }

    #[test]
    fn probe_transitions_to_validated() {
        let config = MigrationConfig::default().with_probes_required(2);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        let s1 = mgr.probe(1).unwrap();
        assert_eq!(s1, MigrationState::Probing);
        let s2 = mgr.probe(1).unwrap();
        assert_eq!(s2, MigrationState::Validated);
    }

    #[test]
    fn complete_migration_updates_address() {
        let config = MigrationConfig::default().with_probes_required(1);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 5000)).unwrap();
        mgr.probe(1).unwrap();
        mgr.complete(1).unwrap();
        assert_eq!(mgr.address(1).unwrap().ip, "10.0.0.2");
        assert_eq!(mgr.address(1).unwrap().port, 5000);
        assert!(!mgr.has_active_migration(1).unwrap());
    }

    #[test]
    fn complete_without_validation_fails() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        let err = mgr.complete(1).unwrap_err();
        assert!(matches!(err, MigrationError::PathValidationFailed { .. }));
    }

    #[test]
    fn rollback_restores_original_address() {
        let config = MigrationConfig::default().with_probes_required(1);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        mgr.rollback(1).unwrap();
        // Address should remain unchanged.
        assert_eq!(mgr.address(1).unwrap().ip, "10.0.0.1");
        assert!(!mgr.has_active_migration(1).unwrap());
    }

    #[test]
    fn rollback_without_migration_fails() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        let err = mgr.rollback(1).unwrap_err();
        assert!(matches!(err, MigrationError::NoActiveMigration(1)));
    }

    #[test]
    fn migration_history_recorded() {
        let config = MigrationConfig::default().with_probes_required(1);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        mgr.probe(1).unwrap();
        mgr.complete(1).unwrap();
        let hist = mgr.history(1).unwrap();
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].state, MigrationState::Completed);
    }

    #[test]
    fn history_limit_enforced() {
        let config = MigrationConfig::default()
            .with_probes_required(1)
            .with_max_history(2);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        for i in 0..5 {
            let new_ip = format!("10.0.0.{}", i + 2);
            mgr.initiate(1, addr(&new_ip, 4433)).unwrap();
            mgr.probe(1).unwrap();
            mgr.complete(1).unwrap();
        }
        assert!(mgr.history(1).unwrap().len() <= 2);
    }

    #[test]
    fn stats_count_probes() {
        let config = MigrationConfig::default().with_probes_required(3);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        mgr.probe(1).unwrap();
        mgr.probe(1).unwrap();
        mgr.probe(1).unwrap();
        assert_eq!(mgr.stats().probes_sent, 3);
    }

    #[test]
    fn stats_success_rate() {
        let config = MigrationConfig::default().with_probes_required(1);
        let mut mgr = MigrationManager::new(config);
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        // Successful migration.
        mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap();
        mgr.probe(1).unwrap();
        mgr.complete(1).unwrap();
        // Failed migration.
        mgr.initiate(1, addr("10.0.0.3", 4433)).unwrap();
        mgr.rollback(1).unwrap();
        assert!((mgr.stats().success_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn close_connection() {
        let mut mgr = default_mgr();
        mgr.register(1, addr("10.0.0.1", 4433)).unwrap();
        mgr.close(1).unwrap();
        let err = mgr.initiate(1, addr("10.0.0.2", 4433)).unwrap_err();
        assert!(matches!(err, MigrationError::ConnectionClosed(1)));
    }

    #[test]
    fn address_display() {
        let a = addr("192.168.1.1", 8080);
        assert_eq!(format!("{a}"), "192.168.1.1:8080");
    }

    #[test]
    fn manager_display() {
        let mgr = default_mgr();
        let s = format!("{mgr}");
        assert!(s.contains("MigrationManager"));
    }

    #[test]
    fn config_builder() {
        let config = MigrationConfig::default()
            .with_max_connections(500)
            .with_probes_required(5)
            .with_max_history(10);
        assert_eq!(config.max_connections, 500);
        assert_eq!(config.probes_required, 5);
        assert_eq!(config.max_history, 10);
    }

    #[test]
    fn migration_record_display() {
        let r = MigrationRecord {
            from_address: addr("10.0.0.1", 4433),
            to_address: addr("10.0.0.2", 4433),
            state: MigrationState::Completed,
            started_tick: 0,
            completed_tick: Some(50),
            probe_count: 3,
        };
        let s = format!("{r}");
        assert!(s.contains("10.0.0.1:4433"));
        assert!(s.contains("Completed"));
    }
}
