//! Simple connection pool for JouleDB.
//!
//! [`ConnectionPool`] maintains a set of idle [`Connection`]s and hands them
//! out on demand via [`PooledConnection`]. When a `PooledConnection` is
//! dropped it is automatically returned to the pool.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::sync::Mutex;

use crate::connection::{Connection, ConnectionConfig};
use crate::error::{ClientError, Result};

// ============================================================================
// PoolConfig
// ============================================================================

/// Configuration for a [`ConnectionPool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections the pool will hold.
    pub max_connections: usize,
    /// Minimum (initial) number of connections the pool will create.
    pub min_connections: usize,
    /// How long to wait when all connections are checked out.
    pub connection_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            connection_timeout: Duration::from_secs(5),
        }
    }
}

// ============================================================================
// PoolStats
// ============================================================================

/// Live statistics for a [`ConnectionPool`].
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total connections ever created.
    pub total_created: usize,
    /// Connections currently idle in the pool.
    pub idle: usize,
    /// Connections currently checked out.
    pub in_use: usize,
    /// Maximum configured pool size.
    pub max_connections: usize,
}

// ============================================================================
// ConnectionPool
// ============================================================================

/// A pool of reusable TCP connections to an JouleDB server.
///
/// # Example
///
/// ```no_run
/// # async fn example() -> joule_db_client::error::Result<()> {
/// use joule_db_client::connection::ConnectionConfig;
/// use joule_db_client::pool::{ConnectionPool, PoolConfig};
///
/// let pool = ConnectionPool::new(
///     ConnectionConfig::default(),
///     PoolConfig { max_connections: 5, min_connections: 2, ..Default::default() },
/// ).await?;
///
/// let conn = pool.get().await?;
/// conn.put("key", b"val", None).await?;
/// // `conn` is returned to the pool when dropped.
/// # Ok(())
/// # }
/// ```
pub struct ConnectionPool {
    conn_config: ConnectionConfig,
    pool_config: PoolConfig,
    idle: Mutex<VecDeque<Connection>>,
    total_created: AtomicUsize,
    in_use: AtomicUsize,
}

impl std::fmt::Debug for ConnectionPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionPool")
            .field("conn_config", &self.conn_config)
            .field("pool_config", &self.pool_config)
            .field("total_created", &self.total_created.load(Ordering::Relaxed))
            .field("in_use", &self.in_use.load(Ordering::Relaxed))
            .finish()
    }
}

impl ConnectionPool {
    /// Create a new pool, pre-populating it with `min_connections` connections.
    pub async fn new(conn_config: ConnectionConfig, pool_config: PoolConfig) -> Result<Self> {
        let mut idle = VecDeque::with_capacity(pool_config.max_connections);
        let mut created = 0usize;

        for _ in 0..pool_config.min_connections {
            let conn = Connection::connect(conn_config.clone()).await?;
            idle.push_back(conn);
            created += 1;
        }

        Ok(Self {
            conn_config,
            pool_config,
            idle: Mutex::new(idle),
            total_created: AtomicUsize::new(created),
            in_use: AtomicUsize::new(0),
        })
    }

    /// Acquire a connection from the pool. If none are idle and the pool has
    /// not reached `max_connections`, a new connection is created. If the pool
    /// is fully checked out, this waits up to `connection_timeout` before
    /// returning [`ClientError::PoolExhausted`].
    pub async fn get(&self) -> Result<PooledConnection<'_>> {
        // Fast path: try to pop an idle connection.
        {
            let mut idle = self.idle.lock().await;
            if let Some(conn) = idle.pop_front() {
                self.in_use.fetch_add(1, Ordering::Relaxed);
                return Ok(PooledConnection {
                    pool: self,
                    conn: Some(conn),
                });
            }
        }

        // Can we create a new one?
        let total = self.total_created.load(Ordering::Relaxed);
        if total < self.pool_config.max_connections {
            let conn = Connection::connect(self.conn_config.clone()).await?;
            self.total_created.fetch_add(1, Ordering::Relaxed);
            self.in_use.fetch_add(1, Ordering::Relaxed);
            return Ok(PooledConnection {
                pool: self,
                conn: Some(conn),
            });
        }

        // Pool is full -- wait with a timeout for a connection to be returned.
        let deadline = tokio::time::Instant::now() + self.pool_config.connection_timeout;
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            {
                let mut idle = self.idle.lock().await;
                if let Some(conn) = idle.pop_front() {
                    self.in_use.fetch_add(1, Ordering::Relaxed);
                    return Ok(PooledConnection {
                        pool: self,
                        conn: Some(conn),
                    });
                }
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(ClientError::PoolExhausted);
            }
        }
    }

    /// Return a connection to the pool (called automatically by
    /// `PooledConnection::drop`).
    async fn return_conn(&self, conn: Connection) {
        let mut idle = self.idle.lock().await;
        idle.push_back(conn);
        self.in_use.fetch_sub(1, Ordering::Relaxed);
    }

    /// Snapshot of pool statistics.
    pub fn stats(&self) -> PoolStats {
        let total_created = self.total_created.load(Ordering::Relaxed);
        let in_use = self.in_use.load(Ordering::Relaxed);
        PoolStats {
            total_created,
            idle: total_created.saturating_sub(in_use),
            in_use,
            max_connections: self.pool_config.max_connections,
        }
    }
}

// ============================================================================
// PooledConnection
// ============================================================================

/// A connection checked out from a [`ConnectionPool`].
///
/// When this value is dropped the underlying connection is returned to the
/// pool. All connection methods are available via `Deref`.
pub struct PooledConnection<'a> {
    pool: &'a ConnectionPool,
    conn: Option<Connection>,
}

impl std::fmt::Debug for PooledConnection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledConnection").finish_non_exhaustive()
    }
}

impl std::ops::Deref for PooledConnection<'_> {
    type Target = Connection;

    fn deref(&self) -> &Self::Target {
        self.conn
            .as_ref()
            .unwrap_or_else(|| unreachable!("PooledConnection used after drop"))
    }
}

impl Drop for PooledConnection<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            // Try to return the connection synchronously. The Mutex is a
            // tokio::sync::Mutex, so `try_lock` is available without being in
            // an async context.
            if let Ok(mut idle) = self.pool.idle.try_lock() {
                idle.push_back(conn);
                self.pool.in_use.fetch_sub(1, Ordering::Relaxed);
            } else {
                // Could not acquire the lock -- drop the connection and adjust
                // the counters so the pool can create a fresh one later.
                self.pool.in_use.fetch_sub(1, Ordering::Relaxed);
                self.pool.total_created.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_default() {
        let cfg = PoolConfig::default();
        assert_eq!(cfg.max_connections, 10);
        assert_eq!(cfg.min_connections, 1);
        assert_eq!(cfg.connection_timeout, Duration::from_secs(5));
    }

    #[test]
    fn test_pool_stats_display() {
        let stats = PoolStats {
            total_created: 5,
            idle: 3,
            in_use: 2,
            max_connections: 10,
        };
        assert_eq!(stats.total_created, 5);
        assert_eq!(stats.idle, 3);
        assert_eq!(stats.in_use, 2);
        assert_eq!(stats.max_connections, 10);
    }
}
