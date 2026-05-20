//! Connection Pool Implementation
//!
//! A generic, high-performance connection pool with:
//! - Configurable min/max connections
//! - Connection lifecycle management
//! - Idle timeout and connection validation
//! - Health checking background task
//! - Comprehensive statistics and metrics

use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::{interval, timeout};

/// Error types for connection pool operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// Timeout waiting to acquire a connection
    Timeout,
    /// Pool has been closed
    PoolClosed,
    /// Failed to create a new connection
    ConnectionFailed(String),
    /// Connection validation failed
    ValidationFailed(String),
    /// Pool is exhausted and cannot create more connections
    Exhausted,
    /// Configuration error
    ConfigError(String),
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolError::Timeout => write!(f, "Timeout waiting for connection"),
            PoolError::PoolClosed => write!(f, "Connection pool is closed"),
            PoolError::ConnectionFailed(msg) => write!(f, "Failed to create connection: {}", msg),
            PoolError::ValidationFailed(msg) => write!(f, "Connection validation failed: {}", msg),
            PoolError::Exhausted => write!(f, "Connection pool exhausted"),
            PoolError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for PoolError {}

/// Result type for pool operations
pub type PoolResult<T> = Result<T, PoolError>;

/// Configuration for the connection pool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of connections to maintain in the pool
    pub min_connections: usize,
    /// Maximum number of connections allowed in the pool
    pub max_connections: usize,
    /// Timeout for acquiring a connection from the pool
    pub acquire_timeout: Duration,
    /// Maximum time a connection can remain idle before being closed
    pub idle_timeout: Duration,
    /// Maximum lifetime of a connection before it's recycled
    pub max_lifetime: Duration,
    /// Interval between health checks
    pub health_check_interval: Duration,
    /// Whether to test connections on checkout
    pub test_on_checkout: bool,
    /// Whether to test connections when returned to pool
    pub test_on_return: bool,
    /// Number of connections to create during initialization
    pub initial_connections: usize,
    /// Timeout for connection creation
    pub connection_timeout: Duration,
    /// Timeout for validation checks
    pub validation_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 2,
            max_connections: 10,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(3600),
            health_check_interval: Duration::from_secs(30),
            test_on_checkout: true,
            test_on_return: false,
            initial_connections: 2,
            connection_timeout: Duration::from_secs(10),
            validation_timeout: Duration::from_secs(5),
        }
    }
}

impl PoolConfig {
    /// Create a new pool configuration builder
    pub fn builder() -> PoolConfigBuilder {
        PoolConfigBuilder::default()
    }

    /// Validate the configuration
    pub fn validate(&self) -> PoolResult<()> {
        if self.min_connections > self.max_connections {
            return Err(PoolError::ConfigError(
                "min_connections cannot exceed max_connections".to_string(),
            ));
        }
        if self.initial_connections > self.max_connections {
            return Err(PoolError::ConfigError(
                "initial_connections cannot exceed max_connections".to_string(),
            ));
        }
        if self.max_connections == 0 {
            return Err(PoolError::ConfigError(
                "max_connections must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Builder for PoolConfig
#[derive(Debug, Clone, Default)]
pub struct PoolConfigBuilder {
    config: PoolConfig,
}

impl PoolConfigBuilder {
    pub fn min_connections(mut self, n: usize) -> Self {
        self.config.min_connections = n;
        self
    }

    pub fn max_connections(mut self, n: usize) -> Self {
        self.config.max_connections = n;
        self
    }

    pub fn acquire_timeout(mut self, timeout: Duration) -> Self {
        self.config.acquire_timeout = timeout;
        self
    }

    pub fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.config.idle_timeout = timeout;
        self
    }

    pub fn max_lifetime(mut self, lifetime: Duration) -> Self {
        self.config.max_lifetime = lifetime;
        self
    }

    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.config.health_check_interval = interval;
        self
    }

    pub fn test_on_checkout(mut self, test: bool) -> Self {
        self.config.test_on_checkout = test;
        self
    }

    pub fn test_on_return(mut self, test: bool) -> Self {
        self.config.test_on_return = test;
        self
    }

    pub fn initial_connections(mut self, n: usize) -> Self {
        self.config.initial_connections = n;
        self
    }

    pub fn connection_timeout(mut self, timeout: Duration) -> Self {
        self.config.connection_timeout = timeout;
        self
    }

    pub fn validation_timeout(mut self, timeout: Duration) -> Self {
        self.config.validation_timeout = timeout;
        self
    }

    pub fn build(self) -> PoolResult<PoolConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Trait for connection factories
pub trait ConnectionFactory: Send + Sync + 'static {
    /// The connection type produced by this factory
    type Connection: Send + 'static;
    /// Error type for connection creation
    type Error: std::error::Error + Send + Sync + 'static;

    /// Create a new connection
    fn create(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Self::Connection, Self::Error>> + Send + '_>>;

    /// Validate an existing connection
    fn validate(&self, conn: &Self::Connection) -> Pin<Box<dyn Future<Output = bool> + Send + '_>>;

    /// Called before a connection is destroyed
    fn destroy(&self, _conn: Self::Connection) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async {})
    }
}

/// Internal wrapper for pooled connections with metadata
struct PooledConnectionInner<C> {
    connection: C,
    created_at: Instant,
    last_used: Instant,
    use_count: u64,
}

impl<C> PooledConnectionInner<C> {
    fn new(connection: C) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
            use_count: 0,
        }
    }

    fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }

    fn is_idle_timeout(&self, idle_timeout: Duration) -> bool {
        self.last_used.elapsed() > idle_timeout
    }

    fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
    }
}

/// Statistics for the connection pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of connections (active + idle)
    pub total_connections: usize,
    /// Number of connections currently in use
    pub active_connections: usize,
    /// Number of idle connections available
    pub idle_connections: usize,
    /// Total number of connections created
    pub connections_created: u64,
    /// Total number of connections destroyed
    pub connections_destroyed: u64,
    /// Total number of successful checkouts
    pub checkouts: u64,
    /// Total number of checkout timeouts
    pub timeouts: u64,
    /// Total number of validation failures
    pub validation_failures: u64,
    /// Average wait time for acquiring a connection (in microseconds)
    pub avg_wait_time_us: u64,
    /// Maximum wait time for acquiring a connection (in microseconds)
    pub max_wait_time_us: u64,
    /// Number of pending waiters
    pub pending_waiters: usize,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            total_connections: 0,
            active_connections: 0,
            idle_connections: 0,
            connections_created: 0,
            connections_destroyed: 0,
            checkouts: 0,
            timeouts: 0,
            validation_failures: 0,
            avg_wait_time_us: 0,
            max_wait_time_us: 0,
            pending_waiters: 0,
        }
    }
}

/// Internal statistics counters (atomic)
struct InternalStats {
    connections_created: AtomicU64,
    connections_destroyed: AtomicU64,
    checkouts: AtomicU64,
    timeouts: AtomicU64,
    validation_failures: AtomicU64,
    total_wait_time_us: AtomicU64,
    max_wait_time_us: AtomicU64,
    pending_waiters: AtomicUsize,
}

impl Default for InternalStats {
    fn default() -> Self {
        Self {
            connections_created: AtomicU64::new(0),
            connections_destroyed: AtomicU64::new(0),
            checkouts: AtomicU64::new(0),
            timeouts: AtomicU64::new(0),
            validation_failures: AtomicU64::new(0),
            total_wait_time_us: AtomicU64::new(0),
            max_wait_time_us: AtomicU64::new(0),
            pending_waiters: AtomicUsize::new(0),
        }
    }
}

/// Internal state of the connection pool
struct PoolInner<C> {
    /// Available connections
    idle_connections: VecDeque<PooledConnectionInner<C>>,
    /// Whether the pool is closed
    closed: bool,
    /// Current number of active (checked out) connections
    active_count: usize,
}

/// A generic connection pool
pub struct ConnectionPool<F: ConnectionFactory> {
    config: PoolConfig,
    factory: Arc<F>,
    inner: Arc<Mutex<PoolInner<F::Connection>>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<InternalStats>,
    notify: Arc<Notify>,
    shutdown: Arc<Notify>,
}

impl<F: ConnectionFactory> ConnectionPool<F> {
    /// Create a new connection pool
    pub async fn new(factory: F, config: PoolConfig) -> PoolResult<Arc<Self>> {
        config.validate()?;

        let pool = Arc::new(Self {
            semaphore: Arc::new(Semaphore::new(config.max_connections)),
            config,
            factory: Arc::new(factory),
            inner: Arc::new(Mutex::new(PoolInner {
                idle_connections: VecDeque::new(),
                closed: false,
                active_count: 0,
            })),
            stats: Arc::new(InternalStats::default()),
            notify: Arc::new(Notify::new()),
            shutdown: Arc::new(Notify::new()),
        });

        // Initialize pool with initial connections
        pool.initialize().await?;

        // Start health check background task
        pool.clone().start_health_check();

        Ok(pool)
    }

    /// Initialize the pool with initial connections
    async fn initialize(&self) -> PoolResult<()> {
        let initial = self
            .config
            .initial_connections
            .min(self.config.max_connections);

        for _ in 0..initial {
            match self.create_connection().await {
                Ok(conn) => {
                    let mut inner = self.inner.lock().await;
                    inner
                        .idle_connections
                        .push_back(PooledConnectionInner::new(conn));
                }
                Err(e) => {
                    // Log but don't fail initialization
                    eprintln!("Failed to create initial connection: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Create a new connection using the factory
    async fn create_connection(&self) -> PoolResult<F::Connection> {
        let result = timeout(self.config.connection_timeout, self.factory.create()).await;

        match result {
            Ok(Ok(conn)) => {
                self.stats
                    .connections_created
                    .fetch_add(1, Ordering::Relaxed);
                Ok(conn)
            }
            Ok(Err(e)) => Err(PoolError::ConnectionFailed(e.to_string())),
            Err(_) => Err(PoolError::Timeout),
        }
    }

    /// Validate a connection
    async fn validate_connection(&self, conn: &F::Connection) -> bool {
        let result = timeout(self.config.validation_timeout, self.factory.validate(conn)).await;

        match result {
            Ok(valid) => {
                if !valid {
                    self.stats
                        .validation_failures
                        .fetch_add(1, Ordering::Relaxed);
                }
                valid
            }
            Err(_) => {
                self.stats
                    .validation_failures
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Acquire a connection from the pool
    pub async fn get(&self) -> PoolResult<PooledConnection<F>> {
        let start = Instant::now();
        self.stats.pending_waiters.fetch_add(1, Ordering::Relaxed);

        // Try to acquire semaphore permit with timeout
        let permit = match timeout(
            self.config.acquire_timeout,
            self.semaphore.clone().acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                self.stats.pending_waiters.fetch_sub(1, Ordering::Relaxed);
                return Err(PoolError::PoolClosed);
            }
            Err(_) => {
                self.stats.pending_waiters.fetch_sub(1, Ordering::Relaxed);
                self.stats.timeouts.fetch_add(1, Ordering::Relaxed);
                return Err(PoolError::Timeout);
            }
        };

        self.stats.pending_waiters.fetch_sub(1, Ordering::Relaxed);

        // Try to get an existing connection
        let conn = self.get_idle_connection().await;

        let pooled_inner = match conn {
            Some(mut inner) => {
                // Validate if configured
                if self.config.test_on_checkout {
                    if !self.validate_connection(&inner.connection).await {
                        // Connection is bad, destroy it and try to create a new one
                        self.factory.destroy(inner.connection).await;
                        self.stats
                            .connections_destroyed
                            .fetch_add(1, Ordering::Relaxed);

                        let new_conn = self.create_connection().await?;
                        inner = PooledConnectionInner::new(new_conn);
                    }
                }
                inner.mark_used();
                inner
            }
            None => {
                // Create new connection
                let conn = self.create_connection().await?;
                let mut inner = PooledConnectionInner::new(conn);
                inner.mark_used();
                inner
            }
        };

        // Update active count
        {
            let mut inner = self.inner.lock().await;
            inner.active_count += 1;
        }

        // Record wait time
        let wait_time = start.elapsed().as_micros() as u64;
        self.stats
            .total_wait_time_us
            .fetch_add(wait_time, Ordering::Relaxed);
        self.stats
            .max_wait_time_us
            .fetch_max(wait_time, Ordering::Relaxed);
        self.stats.checkouts.fetch_add(1, Ordering::Relaxed);

        Ok(PooledConnection {
            inner: Some(pooled_inner),
            pool: self.inner.clone(),
            factory: self.factory.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            notify: self.notify.clone(),
            _permit: permit,
        })
    }

    /// Try to get an idle connection from the pool
    async fn get_idle_connection(&self) -> Option<PooledConnectionInner<F::Connection>> {
        let mut inner = self.inner.lock().await;

        if inner.closed {
            return None;
        }

        while let Some(conn) = inner.idle_connections.pop_front() {
            // Check if connection is still valid (not expired)
            if !conn.is_expired(self.config.max_lifetime)
                && !conn.is_idle_timeout(self.config.idle_timeout)
            {
                return Some(conn);
            }

            // Connection is expired, destroy it
            drop(inner);
            self.factory.destroy(conn.connection).await;
            self.stats
                .connections_destroyed
                .fetch_add(1, Ordering::Relaxed);
            inner = self.inner.lock().await;
        }

        None
    }

    /// Return a connection to the pool
    async fn return_connection(&self, mut conn: PooledConnectionInner<F::Connection>) {
        // Validate if configured
        if self.config.test_on_return {
            if !self.validate_connection(&conn.connection).await {
                self.factory.destroy(conn.connection).await;
                self.stats
                    .connections_destroyed
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
        }

        // Check if connection is still valid
        if conn.is_expired(self.config.max_lifetime) {
            self.factory.destroy(conn.connection).await;
            self.stats
                .connections_destroyed
                .fetch_add(1, Ordering::Relaxed);
            return;
        }

        conn.last_used = Instant::now();

        let mut inner = self.inner.lock().await;

        if inner.closed {
            drop(inner);
            self.factory.destroy(conn.connection).await;
            self.stats
                .connections_destroyed
                .fetch_add(1, Ordering::Relaxed);
            return;
        }

        inner.idle_connections.push_back(conn);
        self.notify.notify_one();
    }

    /// Start the health check background task
    fn start_health_check(self: Arc<Self>) {
        let pool = self;

        tokio::spawn(async move {
            let mut interval = interval(pool.config.health_check_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        pool.run_health_check().await;
                    }
                    _ = pool.shutdown.notified() => {
                        break;
                    }
                }
            }
        });
    }

    /// Run a health check on idle connections
    async fn run_health_check(&self) {
        let mut to_destroy = Vec::new();
        let mut to_return = Vec::new();

        {
            let mut inner = self.inner.lock().await;

            if inner.closed {
                return;
            }

            // Check all idle connections
            while let Some(conn) = inner.idle_connections.pop_front() {
                if conn.is_expired(self.config.max_lifetime)
                    || conn.is_idle_timeout(self.config.idle_timeout)
                {
                    to_destroy.push(conn);
                } else {
                    to_return.push(conn);
                }
            }

            // Return valid connections
            for conn in to_return {
                inner.idle_connections.push_back(conn);
            }
        }

        // Destroy expired connections outside the lock
        for conn in to_destroy {
            self.factory.destroy(conn.connection).await;
            self.stats
                .connections_destroyed
                .fetch_add(1, Ordering::Relaxed);
        }

        // Ensure minimum connections
        self.ensure_min_connections().await;
    }

    /// Ensure the pool has at least min_connections
    async fn ensure_min_connections(&self) {
        let (current_total, closed) = {
            let inner = self.inner.lock().await;
            let idle = inner.idle_connections.len();
            let active = inner.active_count;
            (idle + active, inner.closed)
        };

        if closed {
            return;
        }

        let needed = self.config.min_connections.saturating_sub(current_total);

        for _ in 0..needed {
            // Acquire semaphore first
            let permit = match self.semaphore.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => break,
            };

            match self.create_connection().await {
                Ok(conn) => {
                    let mut inner = self.inner.lock().await;
                    if !inner.closed {
                        inner
                            .idle_connections
                            .push_back(PooledConnectionInner::new(conn));
                    }
                }
                Err(e) => {
                    eprintln!("Failed to create connection for min pool: {}", e);
                }
            }

            // Release the permit since the connection goes to idle pool
            drop(permit);
        }
    }

    /// Get current pool statistics
    pub async fn stats(&self) -> PoolStats {
        let inner = self.inner.lock().await;
        let idle = inner.idle_connections.len();
        let active = inner.active_count;
        let total = idle + active;

        let checkouts = self.stats.checkouts.load(Ordering::Relaxed);
        let total_wait = self.stats.total_wait_time_us.load(Ordering::Relaxed);
        let avg_wait = if checkouts > 0 {
            total_wait / checkouts
        } else {
            0
        };

        PoolStats {
            total_connections: total,
            active_connections: active,
            idle_connections: idle,
            connections_created: self.stats.connections_created.load(Ordering::Relaxed),
            connections_destroyed: self.stats.connections_destroyed.load(Ordering::Relaxed),
            checkouts,
            timeouts: self.stats.timeouts.load(Ordering::Relaxed),
            validation_failures: self.stats.validation_failures.load(Ordering::Relaxed),
            avg_wait_time_us: avg_wait,
            max_wait_time_us: self.stats.max_wait_time_us.load(Ordering::Relaxed),
            pending_waiters: self.stats.pending_waiters.load(Ordering::Relaxed),
        }
    }

    /// Close the pool and destroy all connections
    pub async fn close(&self) {
        let connections_to_destroy: Vec<_> = {
            let mut inner = self.inner.lock().await;
            inner.closed = true;
            inner.idle_connections.drain(..).collect()
        };

        // Signal shutdown to health check task
        self.shutdown.notify_waiters();

        // Destroy all idle connections
        for conn in connections_to_destroy {
            self.factory.destroy(conn.connection).await;
            self.stats
                .connections_destroyed
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Check if the pool is closed
    pub async fn is_closed(&self) -> bool {
        let inner = self.inner.lock().await;
        inner.closed
    }

    /// Get the pool configuration
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
}

/// A connection borrowed from the pool
pub struct PooledConnection<F: ConnectionFactory> {
    inner: Option<PooledConnectionInner<F::Connection>>,
    pool: Arc<Mutex<PoolInner<F::Connection>>>,
    factory: Arc<F>,
    config: PoolConfig,
    stats: Arc<InternalStats>,
    notify: Arc<Notify>,
    _permit: OwnedSemaphorePermit,
}

impl<F: ConnectionFactory> PooledConnection<F> {
    /// Get the number of times this connection has been used
    pub fn use_count(&self) -> u64 {
        self.inner.as_ref().map(|i| i.use_count).unwrap_or(0)
    }

    /// Get the age of this connection
    pub fn age(&self) -> Duration {
        self.inner
            .as_ref()
            .map(|i| i.created_at.elapsed())
            .unwrap_or_default()
    }

    /// Get the time since this connection was last used
    pub fn idle_time(&self) -> Duration {
        self.inner
            .as_ref()
            .map(|i| i.last_used.elapsed())
            .unwrap_or_default()
    }

    /// Detach this connection from the pool (it won't be returned)
    pub fn detach(mut self) -> F::Connection {
        let inner = self.inner.take().expect("Connection already taken");
        inner.connection
    }
}

impl<F: ConnectionFactory> Deref for PooledConnection<F> {
    type Target = F::Connection;

    fn deref(&self) -> &Self::Target {
        &self
            .inner
            .as_ref()
            .expect("Connection already taken")
            .connection
    }
}

impl<F: ConnectionFactory> DerefMut for PooledConnection<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self
            .inner
            .as_mut()
            .expect("Connection already taken")
            .connection
    }
}

impl<F: ConnectionFactory> Drop for PooledConnection<F> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let pool = self.pool.clone();
            let factory = self.factory.clone();
            let config = self.config.clone();
            let stats = self.stats.clone();
            let notify = self.notify.clone();

            tokio::spawn(async move {
                // Decrement active count
                {
                    let mut pool_inner = pool.lock().await;
                    pool_inner.active_count = pool_inner.active_count.saturating_sub(1);
                }

                // Validate if configured
                if config.test_on_return {
                    let valid = {
                        let result = timeout(
                            config.validation_timeout,
                            factory.validate(&inner.connection),
                        )
                        .await;
                        match result {
                            Ok(valid) => valid,
                            Err(_) => false,
                        }
                    };

                    if !valid {
                        factory.destroy(inner.connection).await;
                        stats.connections_destroyed.fetch_add(1, Ordering::Relaxed);
                        stats.validation_failures.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                }

                // Check if connection is expired
                if inner.is_expired(config.max_lifetime) {
                    factory.destroy(inner.connection).await;
                    stats.connections_destroyed.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                // Return to pool
                let mut pool_inner = pool.lock().await;

                if pool_inner.closed {
                    drop(pool_inner);
                    factory.destroy(inner.connection).await;
                    stats.connections_destroyed.fetch_add(1, Ordering::Relaxed);
                    return;
                }

                let mut conn = inner;
                conn.last_used = Instant::now();
                pool_inner.idle_connections.push_back(conn);
                notify.notify_one();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    use tokio::time::sleep;

    /// Mock connection for testing
    #[derive(Debug)]
    struct MockConnection {
        id: u32,
        valid: Arc<std::sync::atomic::AtomicBool>,
    }

    /// Mock connection factory for testing
    struct MockFactory {
        next_id: AtomicU32,
        create_delay: Duration,
        should_fail: Arc<std::sync::atomic::AtomicBool>,
        created_count: AtomicU32,
        destroyed_count: AtomicU32,
        connections_valid: Arc<std::sync::atomic::AtomicBool>,
    }

    impl MockFactory {
        fn new() -> Self {
            Self {
                next_id: AtomicU32::new(0),
                create_delay: Duration::from_millis(10),
                should_fail: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                created_count: AtomicU32::new(0),
                destroyed_count: AtomicU32::new(0),
                connections_valid: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            }
        }

        fn with_create_delay(mut self, delay: Duration) -> Self {
            self.create_delay = delay;
            self
        }
    }

    impl ConnectionFactory for MockFactory {
        type Connection = MockConnection;
        type Error = std::io::Error;

        fn create(
            &self,
        ) -> Pin<Box<dyn Future<Output = Result<Self::Connection, Self::Error>> + Send + '_>>
        {
            Box::pin(async move {
                sleep(self.create_delay).await;

                if self.should_fail.load(Ordering::Relaxed) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        "Connection failed",
                    ));
                }

                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                self.created_count.fetch_add(1, Ordering::Relaxed);

                Ok(MockConnection {
                    id,
                    valid: self.connections_valid.clone(),
                })
            })
        }

        fn validate(
            &self,
            conn: &Self::Connection,
        ) -> Pin<Box<dyn Future<Output = bool> + Send + '_>> {
            let valid = conn.valid.load(Ordering::Relaxed);
            Box::pin(async move { valid })
        }

        fn destroy(
            &self,
            _conn: Self::Connection,
        ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            self.destroyed_count.fetch_add(1, Ordering::Relaxed);
            Box::pin(async {})
        }
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 5,
            initial_connections: 2,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();
        let stats = pool.stats().await;

        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.idle_connections, 2);
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_get_and_return_connection() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 1,
            max_connections: 5,
            initial_connections: 1,
            test_on_checkout: false,
            test_on_return: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Get a connection
        let conn = pool.get().await.unwrap();
        let conn_id = conn.id;

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.checkouts, 1);

        // Return the connection
        drop(conn);

        // Give time for async drop to complete
        sleep(Duration::from_millis(50)).await;

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 0);
        assert!(stats.idle_connections >= 1);

        // Get again - should get the same connection
        let conn2 = pool.get().await.unwrap();
        assert_eq!(conn2.id, conn_id);
    }

    #[tokio::test]
    async fn test_connection_limit() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 0,
            max_connections: 2,
            initial_connections: 0,
            acquire_timeout: Duration::from_millis(100),
            test_on_checkout: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Get two connections (max)
        let _conn1 = pool.get().await.unwrap();
        let _conn2 = pool.get().await.unwrap();

        // Third should timeout
        let result = pool.get().await;
        assert!(matches!(result, Err(PoolError::Timeout)));

        let stats = pool.stats().await;
        assert_eq!(stats.timeouts, 1);
    }

    #[tokio::test]
    async fn test_connection_validation_on_checkout() {
        let factory = MockFactory::new();
        let connections_valid = factory.connections_valid.clone();

        let config = PoolConfig {
            min_connections: 0,
            max_connections: 5,
            initial_connections: 1,
            test_on_checkout: true,
            test_on_return: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Get initial connection (creates one) - should succeed
        let first_conn_id = {
            let conn = pool.get().await.unwrap();
            conn.id
        };
        sleep(Duration::from_millis(50)).await;

        // Mark connections as invalid before trying to get
        connections_valid.store(false, Ordering::Relaxed);

        // Get again - validation will fail on the pooled connection
        // The pool will destroy the invalid connection and create a new one
        // But creating also uses the factory which validates with the same flag,
        // so we need to re-enable valid connections for new connections to work

        // Actually, the validate call is on the existing connection, and create
        // doesn't call validate. So let's keep it simple - mark invalid, get,
        // which will fail validation, trigger creation, then succeed
        let second_conn = pool.get().await.unwrap();

        // The second connection should have a different ID because the first
        // was destroyed due to validation failure
        assert_ne!(first_conn_id, second_conn.id);

        let stats = pool.stats().await;
        assert!(stats.validation_failures >= 1);
        assert!(stats.connections_destroyed >= 1);
    }

    #[tokio::test]
    async fn test_idle_timeout() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 0,
            max_connections: 5,
            initial_connections: 1,
            idle_timeout: Duration::from_millis(100),
            health_check_interval: Duration::from_millis(50),
            test_on_checkout: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Wait for idle timeout + health check
        sleep(Duration::from_millis(200)).await;

        let stats = pool.stats().await;
        // Connection should have been destroyed due to idle timeout
        assert!(stats.connections_destroyed >= 1);
    }

    #[tokio::test]
    async fn test_max_lifetime() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 0,
            max_connections: 5,
            initial_connections: 0,
            max_lifetime: Duration::from_millis(100),
            test_on_checkout: false,
            test_on_return: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Get and return a connection
        let conn = pool.get().await.unwrap();
        drop(conn);
        sleep(Duration::from_millis(50)).await;

        // Wait for max lifetime to expire
        sleep(Duration::from_millis(100)).await;

        // Get again - old connection should be expired
        let conn2 = pool.get().await.unwrap();

        // Should be a different connection ID since old one expired
        let stats = pool.stats().await;
        assert!(stats.connections_created >= 2);
        drop(conn2);
    }

    #[tokio::test]
    async fn test_pool_close() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 5,
            initial_connections: 2,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        assert!(!pool.is_closed().await);

        pool.close().await;

        assert!(pool.is_closed().await);

        let stats = pool.stats().await;
        assert_eq!(stats.idle_connections, 0);
    }

    #[tokio::test]
    async fn test_config_validation() {
        // min > max should fail
        let result = PoolConfig::builder()
            .min_connections(10)
            .max_connections(5)
            .build();
        assert!(result.is_err());

        // max = 0 should fail
        let result = PoolConfig::builder().max_connections(0).build();
        assert!(result.is_err());

        // initial > max should fail
        let result = PoolConfig::builder()
            .initial_connections(10)
            .max_connections(5)
            .build();
        assert!(result.is_err());

        // Valid config should succeed
        let result = PoolConfig::builder()
            .min_connections(2)
            .max_connections(10)
            .initial_connections(5)
            .build();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 0,
            max_connections: 5,
            initial_connections: 0,
            test_on_checkout: false,
            test_on_return: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Spawn multiple tasks that get and use connections
        let mut handles = Vec::new();

        for _ in 0..10 {
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                for _ in 0..5 {
                    let conn = pool.get().await.unwrap();
                    sleep(Duration::from_millis(10)).await;
                    drop(conn);
                }
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let stats = pool.stats().await;
        assert_eq!(stats.checkouts, 50);
        assert_eq!(stats.timeouts, 0);
    }

    #[tokio::test]
    async fn test_connection_metadata() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 0,
            max_connections: 5,
            initial_connections: 0,
            test_on_checkout: false,
            test_on_return: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        let conn = pool.get().await.unwrap();

        assert_eq!(conn.use_count(), 1);
        assert!(conn.age() < Duration::from_secs(1));
        assert!(conn.idle_time() < Duration::from_secs(1));

        drop(conn);
        sleep(Duration::from_millis(50)).await;

        // Get same connection again
        let conn = pool.get().await.unwrap();
        assert_eq!(conn.use_count(), 2);
    }

    #[tokio::test]
    async fn test_connection_detach() {
        let factory = MockFactory::new();
        let destroyed_count = Arc::new(AtomicU32::new(0));

        let config = PoolConfig {
            min_connections: 0,
            max_connections: 5,
            initial_connections: 0,
            test_on_checkout: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        let conn = pool.get().await.unwrap();
        let detached = conn.detach();

        // Connection should not be returned to pool
        sleep(Duration::from_millis(50)).await;

        let stats = pool.stats().await;
        // The detached connection won't be destroyed by the pool
        assert_eq!(detached.id, 0);
    }

    #[tokio::test]
    async fn test_stats_accuracy() {
        let factory = MockFactory::new();
        let config = PoolConfig {
            min_connections: 2,
            max_connections: 5,
            initial_connections: 2,
            acquire_timeout: Duration::from_millis(50),
            test_on_checkout: false,
            test_on_return: false,
            ..Default::default()
        };

        let pool = ConnectionPool::new(factory, config).await.unwrap();

        // Initial stats
        let stats = pool.stats().await;
        assert_eq!(stats.connections_created, 2);
        assert_eq!(stats.idle_connections, 2);
        assert_eq!(stats.active_connections, 0);

        // Get all connections
        let mut conns = Vec::new();
        for _ in 0..5 {
            conns.push(pool.get().await.unwrap());
        }

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 5);
        assert_eq!(stats.idle_connections, 0);
        assert_eq!(stats.checkouts, 5);

        // Try to get one more (should timeout)
        let _ = pool.get().await;

        let stats = pool.stats().await;
        assert_eq!(stats.timeouts, 1);

        // Return all connections
        drop(conns);
        sleep(Duration::from_millis(100)).await;

        let stats = pool.stats().await;
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.idle_connections, 5);
    }
}
