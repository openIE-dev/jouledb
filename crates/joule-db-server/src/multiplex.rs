//! Connection Multiplexing
//!
//! Allows multiple queries to be sent over a single connection with
//! request/response correlation for efficient resource usage.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

/// Request ID for correlation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    /// Generate a new request ID
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().as_u128() as u64)
    }

    /// Create from u64
    pub fn from_u64(id: u64) -> Self {
        Self(id)
    }

    /// Get as u64
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

/// Multiplexed request
#[derive(Debug)]
pub struct MultiplexedRequest<T> {
    /// Request ID
    pub id: RequestId,
    /// Request payload
    pub payload: T,
    /// Timestamp when request was created
    pub created_at: Instant,
}

impl<T> MultiplexedRequest<T> {
    /// Create a new multiplexed request
    pub fn new(payload: T) -> Self {
        Self {
            id: RequestId::new(),
            payload,
            created_at: Instant::now(),
        }
    }

    /// Get request age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Multiplexed response
#[derive(Debug)]
pub struct MultiplexedResponse<T> {
    /// Request ID this response corresponds to
    pub request_id: RequestId,
    /// Response payload
    pub payload: T,
    /// Timestamp when response was created
    pub created_at: Instant,
}

impl<T> MultiplexedResponse<T> {
    /// Create a new multiplexed response
    pub fn new(request_id: RequestId, payload: T) -> Self {
        Self {
            request_id,
            payload,
            created_at: Instant::now(),
        }
    }
}

/// Pending request tracker
struct PendingRequest<T> {
    /// Response channel
    response_tx: oneshot::Sender<T>,
    /// Request timestamp
    created_at: Instant,
    /// Request timeout
    timeout: Duration,
}

/// Connection multiplexer
///
/// Manages multiple in-flight requests on a single connection
pub struct ConnectionMultiplexer<T> {
    /// Pending requests
    pending: Arc<Mutex<HashMap<RequestId, PendingRequest<T>>>>,
    /// Default timeout for requests
    default_timeout: Duration,
    /// Maximum number of pending requests
    max_pending: usize,
}

impl<T> ConnectionMultiplexer<T> {
    /// Create a new multiplexer
    pub fn new(default_timeout: Duration, max_pending: usize) -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            default_timeout,
            max_pending,
        }
    }

    /// Create with default settings
    pub fn default() -> Self {
        Self::new(Duration::from_secs(30), 1000)
    }

    /// Register a new request and get its ID
    pub fn register_request(&self, timeout: Option<Duration>) -> (RequestId, oneshot::Receiver<T>) {
        let (tx, rx) = oneshot::channel();
        let request_id = RequestId::new();
        let timeout = timeout.unwrap_or(self.default_timeout);

        let mut pending = crate::lock_util::mutex_lock(&self.pending);

        // Check if we're at capacity
        if pending.len() >= self.max_pending {
            // Remove oldest request (simple FIFO eviction)
            if let Some(oldest_id) = pending
                .iter()
                .min_by_key(|(_, req)| req.created_at)
                .map(|(id, _)| *id)
            {
                pending.remove(&oldest_id);
            }
        }

        pending.insert(
            request_id,
            PendingRequest {
                response_tx: tx,
                created_at: Instant::now(),
                timeout,
            },
        );

        (request_id, rx)
    }

    /// Complete a request with a response
    pub fn complete_request(&self, request_id: RequestId, response: T) -> bool {
        let mut pending = crate::lock_util::mutex_lock(&self.pending);

        if let Some(pending_req) = pending.remove(&request_id) {
            // Send response (ignore if receiver dropped)
            let _ = pending_req.response_tx.send(response);
            true
        } else {
            false
        }
    }

    /// Cancel a request
    pub fn cancel_request(&self, request_id: RequestId) -> bool {
        let mut pending = crate::lock_util::mutex_lock(&self.pending);
        pending.remove(&request_id).is_some()
    }

    /// Clean up timed-out requests
    pub fn cleanup_timeouts(&self) -> usize {
        let mut pending = crate::lock_util::mutex_lock(&self.pending);
        let now = Instant::now();
        let mut removed = 0;

        pending.retain(|_, req| {
            if now.duration_since(req.created_at) > req.timeout {
                removed += 1;
                false
            } else {
                true
            }
        });

        removed
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        crate::lock_util::mutex_lock(&self.pending).len()
    }

    /// Cancel all pending requests
    pub fn cancel_all(&self) {
        crate::lock_util::mutex_lock(&self.pending).clear();
    }
}

/// Async request handler for multiplexed connections
pub struct MultiplexedHandler<TRequest, TResponse> {
    multiplexer: Arc<ConnectionMultiplexer<TResponse>>,
    handler: Arc<
        dyn Fn(TRequest) -> std::pin::Pin<Box<dyn std::future::Future<Output = TResponse> + Send>>
            + Send
            + Sync,
    >,
}

impl<TRequest, TResponse> MultiplexedHandler<TRequest, TResponse>
where
    TRequest: Send + 'static,
    TResponse: Send + 'static,
{
    /// Create a new multiplexed handler
    pub fn new<F, Fut>(handler: F) -> Self
    where
        F: Fn(TRequest) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = TResponse> + Send + 'static,
    {
        Self {
            multiplexer: Arc::new(ConnectionMultiplexer::default()),
            handler: Arc::new(move |req| Box::pin(handler(req))),
        }
    }

    /// Handle a request asynchronously
    pub async fn handle(&self, request: TRequest) -> Result<TResponse, MultiplexError> {
        let (request_id, response_rx) = self.multiplexer.register_request(None);

        // Spawn handler task
        let handler = self.handler.clone();
        let multiplexer = self.multiplexer.clone();

        tokio::spawn(async move {
            let response = handler(request).await;
            multiplexer.complete_request(request_id, response);
        });

        // Wait for response
        response_rx
            .await
            .map_err(|_| MultiplexError::RequestTimeout)
    }

    /// Get the multiplexer
    pub fn multiplexer(&self) -> Arc<ConnectionMultiplexer<TResponse>> {
        self.multiplexer.clone()
    }
}

/// Multiplexing errors
#[derive(Debug, Clone)]
pub enum MultiplexError {
    /// Request timed out
    RequestTimeout,
    /// Too many pending requests
    TooManyPending,
    /// Request was cancelled
    Cancelled,
}

impl std::fmt::Display for MultiplexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestTimeout => write!(f, "Request timeout"),
            Self::TooManyPending => write!(f, "Too many pending requests"),
            Self::Cancelled => write!(f, "Request cancelled"),
        }
    }
}

impl std::error::Error for MultiplexError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_multiplexer_basic() {
        let multiplexer = ConnectionMultiplexer::<String>::default();

        let (id1, mut rx1) = multiplexer.register_request(None);
        let (id2, mut rx2) = multiplexer.register_request(None);

        assert_eq!(multiplexer.pending_count(), 2);

        // Complete first request
        assert!(multiplexer.complete_request(id1, "response1".to_string()));
        assert_eq!(rx1.await.unwrap(), "response1");

        // Complete second request
        assert!(multiplexer.complete_request(id2, "response2".to_string()));
        assert_eq!(rx2.await.unwrap(), "response2");

        assert_eq!(multiplexer.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_multiplexer_timeout() {
        let multiplexer = ConnectionMultiplexer::<String>::new(Duration::from_millis(100), 100);

        let (_id, mut rx) = multiplexer.register_request(None);

        // Don't complete the request - should timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Cleanup timeouts
        let removed = multiplexer.cleanup_timeouts();
        assert_eq!(removed, 1);

        // Response should be cancelled
        assert!(rx.await.is_err());
    }

    #[tokio::test]
    async fn test_multiplexed_handler() {
        let handler =
            MultiplexedHandler::new(|req: String| async move { format!("Response to: {}", req) });

        let response = handler.handle("test".to_string()).await.unwrap();
        assert_eq!(response, "Response to: test");
    }
}
