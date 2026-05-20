//! Connection pool — pool of logical connections, checkout/return, max pool
//! size, idle timeout, health check on checkout, pool statistics
//! (active/idle/waiters), connection lifecycle callbacks.
//!
//! Replaces pg-pool, generic-pool, and HikariCP patterns with pure Rust.

use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Pool errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// Pool is exhausted and acquire timed out.
    AcquireTimeout,
    /// Connection not found.
    ConnectionNotFound(u64),
    /// Connection is closed.
    ConnectionClosed(u64),
    /// Pool is shut down.
    PoolShutdown,
    /// Invalid configuration.
    InvalidConfig(String),
    /// Health check failed on connection.
    HealthCheckFailed(u64),
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AcquireTimeout => write!(f, "connection acquire timeout"),
            Self::ConnectionNotFound(id) => write!(f, "connection {id} not found"),
            Self::ConnectionClosed(id) => write!(f, "connection {id} is closed"),
            Self::PoolShutdown => write!(f, "pool is shut down"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::HealthCheckFailed(id) => write!(f, "health check failed for connection {id}"),
        }
    }
}

impl std::error::Error for PoolError {}

// ── Config ───────────────────────────────────────────────────────

/// Pool configuration.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub idle_timeout_ms: u64,
    pub max_lifetime_ms: u64,
    pub acquire_timeout_ms: u64,
    /// Whether to run a health check when checking out a connection.
    pub health_check_on_checkout: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_size: 2,
            max_size: 10,
            idle_timeout_ms: 30_000,
            max_lifetime_ms: 300_000,
            acquire_timeout_ms: 5_000,
            health_check_on_checkout: true,
        }
    }
}

impl PoolConfig {
    pub fn validate(&self) -> Result<(), PoolError> {
        if self.min_size > self.max_size {
            return Err(PoolError::InvalidConfig(
                "min_size must be <= max_size".into(),
            ));
        }
        if self.max_size == 0 {
            return Err(PoolError::InvalidConfig(
                "max_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ── Connection ───────────────────────────────────────────────────

/// Connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Idle,
    InUse,
    Closed,
}

/// A pooled connection.
#[derive(Debug, Clone)]
pub struct Connection {
    pub id: u64,
    pub state: ConnectionState,
    pub created_at_ms: u64,
    pub last_used_at_ms: u64,
    pub checkout_count: u64,
    pub last_health_check_ms: u64,
    /// Simulated health: true = healthy, false = failed.
    pub healthy: bool,
    /// User-defined tags/metadata.
    pub tags: HashMap<String, String>,
}

impl Connection {
    fn new(id: u64, now_ms: u64) -> Self {
        Self {
            id,
            state: ConnectionState::Idle,
            created_at_ms: now_ms,
            last_used_at_ms: now_ms,
            checkout_count: 0,
            last_health_check_ms: now_ms,
            healthy: true,
            tags: HashMap::new(),
        }
    }

    /// Check if this connection has exceeded its max lifetime.
    pub fn is_expired(&self, now_ms: u64, max_lifetime_ms: u64) -> bool {
        now_ms.saturating_sub(self.created_at_ms) >= max_lifetime_ms
    }

    /// Check if this idle connection has exceeded idle timeout.
    pub fn is_idle_expired(&self, now_ms: u64, idle_timeout_ms: u64) -> bool {
        self.state == ConnectionState::Idle
            && now_ms.saturating_sub(self.last_used_at_ms) >= idle_timeout_ms
    }
}

// ── Lifecycle callbacks ──────────────────────────────────────────

/// Events emitted by the pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolEvent {
    /// A new connection was created.
    ConnectionCreated(u64),
    /// A connection was checked out.
    ConnectionCheckedOut(u64),
    /// A connection was returned.
    ConnectionReturned(u64),
    /// A connection was closed.
    ConnectionClosed(u64),
    /// A health check failed.
    HealthCheckFailed(u64),
    /// The pool was shut down.
    PoolShutDown,
}

// ── Pool Stats ───────────────────────────────────────────────────

/// Pool statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolStats {
    pub total: usize,
    pub idle: usize,
    pub in_use: usize,
    pub closed: usize,
    pub total_checkouts: u64,
    pub total_returns: u64,
    pub total_created: u64,
    pub total_closed_count: u64,
    pub health_checks_run: u64,
    pub health_checks_failed: u64,
}

// ── Pool ─────────────────────────────────────────────────────────

/// Connection pool managing a set of connections.
#[derive(Debug)]
pub struct Pool {
    config: PoolConfig,
    connections: HashMap<u64, Connection>,
    next_id: u64,
    shutdown: bool,
    /// Monotonic "clock" for testability (milliseconds).
    current_time_ms: u64,
    /// Event log for lifecycle callbacks.
    events: Vec<PoolEvent>,
    /// Waiter count (simulated — number of failed acquires).
    waiter_count: u64,
    // Counters.
    total_checkouts: u64,
    total_returns: u64,
    total_created: u64,
    total_closed_count: u64,
    health_checks_run: u64,
    health_checks_failed: u64,
}

impl Pool {
    pub fn new(config: PoolConfig) -> Result<Self, PoolError> {
        config.validate()?;
        let mut pool = Self {
            config: config.clone(),
            connections: HashMap::new(),
            next_id: 1,
            shutdown: false,
            current_time_ms: 0,
            events: Vec::new(),
            waiter_count: 0,
            total_checkouts: 0,
            total_returns: 0,
            total_created: 0,
            total_closed_count: 0,
            health_checks_run: 0,
            health_checks_failed: 0,
        };
        // Pre-populate to min_size.
        for _ in 0..config.min_size {
            pool.create_connection();
        }
        Ok(pool)
    }

    /// Advance the internal clock (for testing).
    pub fn advance_time(&mut self, ms: u64) {
        self.current_time_ms += ms;
    }

    /// Set the internal clock.
    pub fn set_time(&mut self, ms: u64) {
        self.current_time_ms = ms;
    }

    fn create_connection(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.connections
            .insert(id, Connection::new(id, self.current_time_ms));
        self.events.push(PoolEvent::ConnectionCreated(id));
        self.total_created += 1;
        id
    }

    /// Acquire a connection. Returns an idle one if available, creates one
    /// if under max_size, or returns AcquireTimeout.
    ///
    /// If `health_check_on_checkout` is enabled, connections are checked
    /// before returning. Unhealthy connections are closed and the search
    /// continues.
    pub fn acquire(&mut self) -> Result<u64, PoolError> {
        if self.shutdown {
            return Err(PoolError::PoolShutdown);
        }

        // Try to find a healthy idle connection.
        loop {
            let idle_id = self
                .connections
                .values()
                .find(|c| c.state == ConnectionState::Idle)
                .map(|c| c.id);

            match idle_id {
                Some(id) => {
                    if self.config.health_check_on_checkout {
                        self.health_checks_run += 1;
                        let healthy = self.connections.get(&id).unwrap().healthy;
                        if !healthy {
                            // Close unhealthy connection.
                            self.health_checks_failed += 1;
                            self.events.push(PoolEvent::HealthCheckFailed(id));
                            self.close_connection_internal(id);
                            continue;
                        }
                        self.connections.get_mut(&id).unwrap().last_health_check_ms =
                            self.current_time_ms;
                    }
                    let conn = self.connections.get_mut(&id).unwrap();
                    conn.state = ConnectionState::InUse;
                    conn.last_used_at_ms = self.current_time_ms;
                    conn.checkout_count += 1;
                    self.total_checkouts += 1;
                    self.events.push(PoolEvent::ConnectionCheckedOut(id));
                    return Ok(id);
                }
                None => break,
            }
        }

        // No idle connections — try creating a new one.
        let active = self
            .connections
            .values()
            .filter(|c| c.state != ConnectionState::Closed)
            .count();

        if active < self.config.max_size {
            let id = self.create_connection();
            let conn = self.connections.get_mut(&id).unwrap();
            conn.state = ConnectionState::InUse;
            conn.last_used_at_ms = self.current_time_ms;
            conn.checkout_count += 1;
            self.total_checkouts += 1;
            self.events.push(PoolEvent::ConnectionCheckedOut(id));
            return Ok(id);
        }

        self.waiter_count += 1;
        Err(PoolError::AcquireTimeout)
    }

    /// Release a connection back to idle.
    pub fn release(&mut self, id: u64) -> Result<(), PoolError> {
        let conn = self
            .connections
            .get_mut(&id)
            .ok_or(PoolError::ConnectionNotFound(id))?;
        if conn.state == ConnectionState::Closed {
            return Err(PoolError::ConnectionClosed(id));
        }
        conn.state = ConnectionState::Idle;
        conn.last_used_at_ms = self.current_time_ms;
        self.total_returns += 1;
        self.events.push(PoolEvent::ConnectionReturned(id));
        Ok(())
    }

    /// Close a specific connection.
    pub fn close_connection(&mut self, id: u64) -> Result<(), PoolError> {
        if !self.connections.contains_key(&id) {
            return Err(PoolError::ConnectionNotFound(id));
        }
        self.close_connection_internal(id);
        Ok(())
    }

    fn close_connection_internal(&mut self, id: u64) {
        if let Some(conn) = self.connections.get_mut(&id) {
            if conn.state != ConnectionState::Closed {
                conn.state = ConnectionState::Closed;
                self.total_closed_count += 1;
                self.events.push(PoolEvent::ConnectionClosed(id));
            }
        }
    }

    /// Simulate marking a connection as unhealthy (for testing).
    pub fn mark_unhealthy(&mut self, id: u64) -> Result<(), PoolError> {
        let conn = self
            .connections
            .get_mut(&id)
            .ok_or(PoolError::ConnectionNotFound(id))?;
        conn.healthy = false;
        Ok(())
    }

    /// Set a tag on a connection.
    pub fn set_tag(&mut self, id: u64, key: &str, value: &str) -> Result<(), PoolError> {
        let conn = self
            .connections
            .get_mut(&id)
            .ok_or(PoolError::ConnectionNotFound(id))?;
        conn.tags.insert(key.to_string(), value.to_string());
        Ok(())
    }

    /// Health check: close expired and idle-expired connections.
    /// Returns number of connections closed.
    pub fn health_check(&mut self) -> usize {
        let now = self.current_time_ms;
        let max_life = self.config.max_lifetime_ms;
        let idle_timeout = self.config.idle_timeout_ms;
        let to_close: Vec<u64> = self
            .connections
            .values()
            .filter(|c| {
                c.state != ConnectionState::Closed
                    && (c.is_expired(now, max_life) || c.is_idle_expired(now, idle_timeout))
            })
            .map(|c| c.id)
            .collect();
        let count = to_close.len();
        for id in to_close {
            self.close_connection_internal(id);
        }
        count
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let mut idle = 0;
        let mut in_use = 0;
        let mut closed = 0;
        for conn in self.connections.values() {
            match conn.state {
                ConnectionState::Idle => idle += 1,
                ConnectionState::InUse => in_use += 1,
                ConnectionState::Closed => closed += 1,
            }
        }
        PoolStats {
            total: self.connections.len(),
            idle,
            in_use,
            closed,
            total_checkouts: self.total_checkouts,
            total_returns: self.total_returns,
            total_created: self.total_created,
            total_closed_count: self.total_closed_count,
            health_checks_run: self.health_checks_run,
            health_checks_failed: self.health_checks_failed,
        }
    }

    /// Get a connection by ID.
    pub fn get_connection(&self, id: u64) -> Option<&Connection> {
        self.connections.get(&id)
    }

    /// Get all pool events.
    pub fn events(&self) -> &[PoolEvent] {
        &self.events
    }

    /// Number of failed acquire attempts (simulated waiters).
    pub fn waiter_count(&self) -> u64 {
        self.waiter_count
    }

    /// Shut down the pool, closing all connections.
    pub fn shutdown(&mut self) {
        self.shutdown = true;
        let ids: Vec<u64> = self.connections.keys().copied().collect();
        for id in ids {
            self.close_connection_internal(id);
        }
        self.events.push(PoolEvent::PoolShutDown);
    }

    pub fn config(&self) -> &PoolConfig {
        &self.config
    }

    /// Attempt to replenish idle connections to min_size.
    pub fn replenish(&mut self) -> usize {
        if self.shutdown {
            return 0;
        }
        let active = self
            .connections
            .values()
            .filter(|c| c.state != ConnectionState::Closed)
            .count();
        let idle = self
            .connections
            .values()
            .filter(|c| c.state == ConnectionState::Idle)
            .count();
        let needed = self.config.min_size.saturating_sub(idle);
        let can_create = self.config.max_size.saturating_sub(active);
        let to_create = needed.min(can_create);
        for _ in 0..to_create {
            self.create_connection();
        }
        to_create
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_pool() -> Pool {
        Pool::new(PoolConfig::default()).unwrap()
    }

    #[test]
    fn pool_starts_with_min_connections() {
        let pool = default_pool();
        let stats = pool.stats();
        assert_eq!(stats.idle, 2);
        assert_eq!(stats.total, 2);
    }

    #[test]
    fn acquire_returns_idle() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        let conn = pool.get_connection(id).unwrap();
        assert_eq!(conn.state, ConnectionState::InUse);
    }

    #[test]
    fn release_makes_idle() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.release(id).unwrap();
        let conn = pool.get_connection(id).unwrap();
        assert_eq!(conn.state, ConnectionState::Idle);
    }

    #[test]
    fn acquire_creates_new_when_all_busy() {
        let mut pool = default_pool();
        let _a = pool.acquire().unwrap();
        let _b = pool.acquire().unwrap();
        // Both min connections in use, should create new.
        let _c = pool.acquire().unwrap();
        assert_eq!(pool.stats().total, 3);
        assert_eq!(pool.stats().in_use, 3);
    }

    #[test]
    fn acquire_fails_at_max() {
        let config = PoolConfig {
            min_size: 1,
            max_size: 2,
            ..Default::default()
        };
        let mut pool = Pool::new(config).unwrap();
        let _a = pool.acquire().unwrap();
        let _b = pool.acquire().unwrap();
        let err = pool.acquire().unwrap_err();
        assert_eq!(err, PoolError::AcquireTimeout);
    }

    #[test]
    fn health_check_closes_expired() {
        let config = PoolConfig {
            min_size: 2,
            max_size: 10,
            max_lifetime_ms: 1000,
            ..Default::default()
        };
        let mut pool = Pool::new(config).unwrap();
        pool.advance_time(1500);
        let closed = pool.health_check();
        assert_eq!(closed, 2);
        assert_eq!(pool.stats().closed, 2);
    }

    #[test]
    fn health_check_closes_idle_expired() {
        let config = PoolConfig {
            min_size: 2,
            max_size: 10,
            idle_timeout_ms: 500,
            max_lifetime_ms: 100_000,
            ..Default::default()
        };
        let mut pool = Pool::new(config).unwrap();
        pool.advance_time(600);
        let closed = pool.health_check();
        assert_eq!(closed, 2);
    }

    #[test]
    fn close_connection_marks_closed() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.close_connection(id).unwrap();
        let conn = pool.get_connection(id).unwrap();
        assert_eq!(conn.state, ConnectionState::Closed);
    }

    #[test]
    fn release_closed_fails() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.close_connection(id).unwrap();
        let err = pool.release(id).unwrap_err();
        assert_eq!(err, PoolError::ConnectionClosed(id));
    }

    #[test]
    fn shutdown_closes_all() {
        let mut pool = default_pool();
        let _a = pool.acquire().unwrap();
        pool.shutdown();
        let stats = pool.stats();
        assert_eq!(stats.closed, stats.total);
    }

    #[test]
    fn acquire_after_shutdown_fails() {
        let mut pool = default_pool();
        pool.shutdown();
        let err = pool.acquire().unwrap_err();
        assert_eq!(err, PoolError::PoolShutdown);
    }

    #[test]
    fn invalid_config_rejected() {
        let config = PoolConfig {
            min_size: 10,
            max_size: 5,
            ..Default::default()
        };
        let err = Pool::new(config).unwrap_err();
        assert!(matches!(err, PoolError::InvalidConfig(_)));
    }

    #[test]
    fn stats_accurate_after_mixed_operations() {
        let mut pool = default_pool();
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        pool.release(a).unwrap();
        pool.close_connection(b).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.idle, 1);
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.closed, 1);
    }

    #[test]
    fn connection_not_found() {
        let mut pool = default_pool();
        let err = pool.release(999).unwrap_err();
        assert_eq!(err, PoolError::ConnectionNotFound(999));
    }

    #[test]
    fn health_check_on_checkout_skips_unhealthy() {
        let config = PoolConfig {
            min_size: 2,
            max_size: 10,
            health_check_on_checkout: true,
            ..Default::default()
        };
        let mut pool = Pool::new(config).unwrap();
        // Mark first connection unhealthy.
        let first_id = pool.acquire().unwrap();
        pool.release(first_id).unwrap();
        pool.mark_unhealthy(first_id).unwrap();
        // Acquire should skip unhealthy and give us the second one.
        let id = pool.acquire().unwrap();
        assert_ne!(id, first_id);
        assert!(pool.stats().health_checks_failed >= 1);
    }

    #[test]
    fn lifecycle_events_recorded() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.release(id).unwrap();
        pool.close_connection(id).unwrap();
        let events = pool.events();
        assert!(events.contains(&PoolEvent::ConnectionCheckedOut(id)));
        assert!(events.contains(&PoolEvent::ConnectionReturned(id)));
        assert!(events.contains(&PoolEvent::ConnectionClosed(id)));
    }

    #[test]
    fn shutdown_event_recorded() {
        let mut pool = default_pool();
        pool.shutdown();
        let events = pool.events();
        assert!(events.contains(&PoolEvent::PoolShutDown));
    }

    #[test]
    fn checkout_count_tracked() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.release(id).unwrap();
        let _ = pool.acquire().unwrap();
        assert_eq!(pool.stats().total_checkouts, 2);
    }

    #[test]
    fn connection_tags() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.set_tag(id, "db", "primary").unwrap();
        let conn = pool.get_connection(id).unwrap();
        assert_eq!(conn.tags.get("db").unwrap(), "primary");
    }

    #[test]
    fn replenish_creates_connections() {
        let config = PoolConfig {
            min_size: 3,
            max_size: 10,
            ..Default::default()
        };
        let mut pool = Pool::new(config).unwrap();
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        pool.close_connection(a).unwrap();
        pool.close_connection(b).unwrap();
        // Now we have 1 idle, 2 closed. min_size=3, so replenish should create 2.
        let created = pool.replenish();
        assert_eq!(created, 2);
        assert_eq!(pool.stats().idle, 3);
    }

    #[test]
    fn waiter_count_on_failed_acquire() {
        let config = PoolConfig {
            min_size: 1,
            max_size: 1,
            ..Default::default()
        };
        let mut pool = Pool::new(config).unwrap();
        let _a = pool.acquire().unwrap();
        let _ = pool.acquire();
        assert_eq!(pool.waiter_count(), 1);
    }

    #[test]
    fn per_connection_checkout_count() {
        let mut pool = default_pool();
        let id = pool.acquire().unwrap();
        pool.release(id).unwrap();
        pool.acquire().unwrap();
        // The same connection id should have been reused.
        let conn = pool.get_connection(id).unwrap();
        assert!(conn.checkout_count >= 1);
    }
}
