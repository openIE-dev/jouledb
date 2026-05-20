//! Production Deployment Support
//!
//! Provides production-ready deployment features:
//! - Graceful shutdown with signal handling
//! - Health check endpoints
//! - Startup and shutdown hooks
//! - Process management

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tokio::time::timeout;

/// Graceful shutdown manager with connection draining support
pub struct GracefulShutdown {
    /// Shutdown signal
    shutdown: Arc<Notify>,
    /// Shutdown requested flag
    requested: Arc<AtomicBool>,
    /// Shutdown timeout
    timeout: Duration,
    /// Shutdown start time
    start_time: Arc<std::sync::Mutex<Option<Instant>>>,
    /// Active connection count (for drain tracking)
    active_connections: Arc<AtomicUsize>,
    /// Notified when all connections drain to zero
    drained: Arc<Notify>,
}

impl GracefulShutdown {
    /// Create a new graceful shutdown manager
    pub fn new(timeout: Duration) -> Self {
        Self {
            shutdown: Arc::new(Notify::new()),
            requested: Arc::new(AtomicBool::new(false)),
            timeout,
            start_time: Arc::new(std::sync::Mutex::new(None)),
            active_connections: Arc::new(AtomicUsize::new(0)),
            drained: Arc::new(Notify::new()),
        }
    }

    /// Create with default timeout (30 seconds)
    pub fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }

    /// Request shutdown
    pub fn request_shutdown(&self) {
        if !self.requested.swap(true, Ordering::SeqCst) {
            *crate::lock_util::mutex_lock(&self.start_time) = Some(Instant::now());
            self.shutdown.notify_waiters();
        }
    }

    /// Check if shutdown has been requested
    pub fn is_shutdown_requested(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
    }

    /// Wait for shutdown signal
    pub async fn wait_for_shutdown(&self) {
        self.shutdown.notified().await;
    }

    /// Wait for shutdown with timeout
    pub async fn wait_for_shutdown_timeout(&self) -> bool {
        timeout(self.timeout, self.shutdown.notified())
            .await
            .is_ok()
    }

    /// Get shutdown notification handle
    pub fn shutdown_notify(&self) -> Arc<Notify> {
        self.shutdown.clone()
    }

    /// Get remaining shutdown time
    pub fn remaining_time(&self) -> Option<Duration> {
        let start_time = crate::lock_util::mutex_lock(&self.start_time);
        if let Some(start) = *start_time {
            let elapsed = start.elapsed();
            if elapsed < self.timeout {
                Some(self.timeout - elapsed)
            } else {
                Some(Duration::ZERO)
            }
        } else {
            None
        }
    }

    /// Register a new active connection. Returns a guard that decrements on drop.
    pub fn connection_guard(&self) -> ConnectionGuard {
        self.active_connections.fetch_add(1, Ordering::SeqCst);
        ConnectionGuard {
            counter: self.active_connections.clone(),
            drained: self.drained.clone(),
        }
    }

    /// Get the current number of active connections.
    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::SeqCst)
    }

    /// Wait for all active connections to drain, up to the shutdown timeout.
    /// Returns the number of connections remaining when the wait ended.
    pub async fn drain_connections(&self) -> usize {
        let remaining = self.active_connections.load(Ordering::SeqCst);
        if remaining == 0 {
            return 0;
        }
        let drain_timeout = self.remaining_time().unwrap_or(Duration::from_secs(30));
        tracing::info!(
            active = remaining,
            timeout_secs = drain_timeout.as_secs(),
            "Draining active connections..."
        );
        let _ = timeout(drain_timeout, self.drained.notified()).await;
        let final_count = self.active_connections.load(Ordering::SeqCst);
        if final_count > 0 {
            tracing::warn!(
                remaining = final_count,
                "Shutdown timeout: forcing close of remaining connections"
            );
        } else {
            tracing::info!("All connections drained successfully");
        }
        final_count
    }
}

/// RAII guard that decrements the active connection counter on drop.
pub struct ConnectionGuard {
    counter: Arc<AtomicUsize>,
    drained: Arc<Notify>,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        let prev = self.counter.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            // Was the last connection — notify drain waiters
            self.drained.notify_waiters();
        }
    }
}

/// Signal handler for graceful shutdown
pub struct SignalHandler {
    shutdown: Arc<GracefulShutdown>,
}

impl SignalHandler {
    /// Create a new signal handler
    pub fn new(shutdown: Arc<GracefulShutdown>) -> Self {
        Self { shutdown }
    }

    /// Start listening for shutdown signals
    pub async fn start(&self) {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};

            let mut sigterm =
                signal(SignalKind::terminate()).expect("failed to register SIGTERM handler");
            let mut sigint =
                signal(SignalKind::interrupt()).expect("failed to register SIGINT handler");
            let shutdown = self.shutdown.clone();

            tokio::spawn(async move {
                tokio::select! {
                    _ = sigterm.recv() => {
                        tracing::info!("Received SIGTERM, initiating graceful shutdown");
                        shutdown.request_shutdown();
                    }
                    _ = sigint.recv() => {
                        tracing::info!("Received SIGINT, initiating graceful shutdown");
                        shutdown.request_shutdown();
                    }
                }
            });
        }

        #[cfg(windows)]
        {
            use tokio::signal::windows::{ctrl_break, ctrl_c};

            let mut ctrl_c = ctrl_c().expect("failed to register Ctrl+C handler");
            let mut ctrl_break = ctrl_break().expect("failed to register Ctrl+Break handler");
            let shutdown = self.shutdown.clone();

            tokio::spawn(async move {
                tokio::select! {
                    _ = ctrl_c.recv() => {
                        tracing::info!("Received Ctrl+C, initiating graceful shutdown");
                        shutdown.request_shutdown();
                    }
                    _ = ctrl_break.recv() => {
                        tracing::info!("Received Ctrl+Break, initiating graceful shutdown");
                        shutdown.request_shutdown();
                    }
                }
            });
        }
    }
}

/// Server lifecycle hooks
pub struct LifecycleHooks {
    /// Pre-startup hook
    pub pre_startup: Option<
        Box<
            dyn Fn() -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<(), Box<dyn std::error::Error + Send + Sync>>,
                            > + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,
    /// Post-startup hook
    pub post_startup: Option<
        Box<
            dyn Fn() -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<(), Box<dyn std::error::Error + Send + Sync>>,
                            > + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,
    /// Pre-shutdown hook
    pub pre_shutdown: Option<
        Box<
            dyn Fn() -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<(), Box<dyn std::error::Error + Send + Sync>>,
                            > + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,
    /// Post-shutdown hook
    pub post_shutdown: Option<
        Box<
            dyn Fn() -> std::pin::Pin<
                    Box<
                        dyn std::future::Future<
                                Output = Result<(), Box<dyn std::error::Error + Send + Sync>>,
                            > + Send,
                    >,
                > + Send
                + Sync,
        >,
    >,
}

impl LifecycleHooks {
    /// Create empty lifecycle hooks
    pub fn new() -> Self {
        Self {
            pre_startup: None,
            post_startup: None,
            pre_shutdown: None,
            post_shutdown: None,
        }
    }

    /// Execute pre-startup hook
    pub async fn execute_pre_startup(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(hook) = &self.pre_startup {
            hook().await
        } else {
            Ok(())
        }
    }

    /// Execute post-startup hook
    pub async fn execute_post_startup(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(hook) = &self.post_startup {
            hook().await
        } else {
            Ok(())
        }
    }

    /// Execute pre-shutdown hook
    pub async fn execute_pre_shutdown(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(hook) = &self.pre_shutdown {
            hook().await
        } else {
            Ok(())
        }
    }

    /// Execute post-shutdown hook
    pub async fn execute_post_shutdown(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(hook) = &self.post_shutdown {
            hook().await
        } else {
            Ok(())
        }
    }
}

impl Default for LifecycleHooks {
    fn default() -> Self {
        Self::new()
    }
}

/// Production server manager
pub struct ProductionServer {
    shutdown: Arc<GracefulShutdown>,
    signal_handler: SignalHandler,
    hooks: LifecycleHooks,
}

impl ProductionServer {
    /// Create a new production server manager
    pub fn new(shutdown_timeout: Duration) -> Self {
        let shutdown = Arc::new(GracefulShutdown::new(shutdown_timeout));
        let signal_handler = SignalHandler::new(shutdown.clone());

        Self {
            shutdown,
            signal_handler,
            hooks: LifecycleHooks::default(),
        }
    }

    /// Create with default settings
    pub fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }

    /// Get shutdown manager
    pub fn shutdown(&self) -> Arc<GracefulShutdown> {
        self.shutdown.clone()
    }

    /// Get lifecycle hooks
    pub fn hooks_mut(&mut self) -> &mut LifecycleHooks {
        &mut self.hooks
    }

    /// Start signal handling
    pub async fn start_signal_handling(&self) {
        self.signal_handler.start().await;
    }

    /// Run server with graceful shutdown
    pub async fn run_with_shutdown<F, Fut>(
        &self,
        server_task: F,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce(Arc<GracefulShutdown>) -> Fut,
        Fut: std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    {
        // Execute pre-startup hooks
        self.hooks.execute_pre_startup().await?;

        // Start signal handling
        self.start_signal_handling().await;

        // Execute post-startup hooks
        self.hooks.execute_post_startup().await?;

        // Run server task
        let shutdown = self.shutdown.clone();
        let server_result = server_task(shutdown.clone()).await;

        // Execute pre-shutdown hooks
        self.hooks.execute_pre_shutdown().await?;

        // Wait for graceful shutdown
        if shutdown.is_shutdown_requested() {
            tracing::info!("Waiting for graceful shutdown...");
            let remaining = shutdown.remaining_time().unwrap_or(Duration::from_secs(30));
            if timeout(remaining, shutdown.wait_for_shutdown())
                .await
                .is_err()
            {
                tracing::warn!("Shutdown timeout exceeded, forcing shutdown");
            }
        }

        // Execute post-shutdown hooks
        self.hooks.execute_post_shutdown().await?;

        server_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_graceful_shutdown() {
        let shutdown = GracefulShutdown::default();

        assert!(!shutdown.is_shutdown_requested());

        shutdown.request_shutdown();
        assert!(shutdown.is_shutdown_requested());
    }

    #[tokio::test]
    async fn test_lifecycle_hooks() {
        let mut hooks = LifecycleHooks::new();

        hooks.pre_startup = Some(Box::new(|| {
            Box::pin(async {
                tracing::info!("Pre-startup hook");
                Ok(())
            })
        }));

        assert!(hooks.execute_pre_startup().await.is_ok());
    }

    #[test]
    fn test_connection_guard_tracks_count() {
        let shutdown = GracefulShutdown::default();
        assert_eq!(shutdown.active_connections(), 0);

        let guard1 = shutdown.connection_guard();
        assert_eq!(shutdown.active_connections(), 1);

        let guard2 = shutdown.connection_guard();
        assert_eq!(shutdown.active_connections(), 2);

        drop(guard1);
        assert_eq!(shutdown.active_connections(), 1);

        drop(guard2);
        assert_eq!(shutdown.active_connections(), 0);
    }

    #[tokio::test]
    async fn test_drain_with_no_connections() {
        let shutdown = GracefulShutdown::default();
        shutdown.request_shutdown();
        let remaining = shutdown.drain_connections().await;
        assert_eq!(remaining, 0);
    }

    #[tokio::test]
    async fn test_drain_notified_when_last_guard_drops() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(5));
        shutdown.request_shutdown();

        let guard = shutdown.connection_guard();
        assert_eq!(shutdown.active_connections(), 1);

        // Spawn a task that drops the guard after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            drop(guard);
        });

        let remaining = shutdown.drain_connections().await;
        assert_eq!(remaining, 0);
    }

    #[test]
    fn test_remaining_time() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(10));
        assert!(shutdown.remaining_time().is_none());

        shutdown.request_shutdown();
        let remaining = shutdown.remaining_time().unwrap();
        assert!(remaining <= Duration::from_secs(10));
        assert!(remaining > Duration::from_secs(9));
    }
}
