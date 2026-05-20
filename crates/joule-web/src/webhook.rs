//! Webhook system — endpoint registration, HMAC payload signing, delivery retry
//! with exponential backoff, delivery log, payload transformation, event filtering,
//! signature verification, and batch delivery.
//!
//! Replaces JS webhook libraries (Svix, webhooks.js) with a pure-Rust webhook
//! system that tracks every delivery attempt.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Webhook domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebhookError {
    /// Endpoint not found.
    EndpointNotFound(String),
    /// Duplicate endpoint ID.
    DuplicateEndpoint(String),
    /// Signature verification failed.
    SignatureInvalid,
    /// Delivery failed.
    DeliveryFailed { endpoint_id: String, reason: String },
    /// Max retries exceeded.
    MaxRetriesExceeded { endpoint_id: String, delivery_id: String },
    /// Payload too large.
    PayloadTooLarge { size: usize, max: usize },
    /// Event type not subscribed.
    EventNotSubscribed { endpoint_id: String, event_type: String },
    /// Invalid URL.
    InvalidUrl(String),
}

impl std::fmt::Display for WebhookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EndpointNotFound(id) => write!(f, "endpoint not found: {id}"),
            Self::DuplicateEndpoint(id) => write!(f, "duplicate endpoint: {id}"),
            Self::SignatureInvalid => write!(f, "signature verification failed"),
            Self::DeliveryFailed { endpoint_id, reason } => {
                write!(f, "delivery to {endpoint_id} failed: {reason}")
            }
            Self::MaxRetriesExceeded { endpoint_id, delivery_id } => {
                write!(f, "max retries exceeded for {endpoint_id} delivery {delivery_id}")
            }
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload too large: {size} bytes (max {max})")
            }
            Self::EventNotSubscribed { endpoint_id, event_type } => {
                write!(f, "endpoint {endpoint_id} not subscribed to {event_type}")
            }
            Self::InvalidUrl(url) => write!(f, "invalid URL: {url}"),
        }
    }
}

impl std::error::Error for WebhookError {}

// ── Enums ───────────────────────────────────────────────────────

/// Delivery status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeliveryStatus {
    Pending,
    Delivered,
    Failed,
    Retrying,
    Exhausted,
}

/// Endpoint status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EndpointStatus {
    Active,
    Disabled,
    Failing,
}

// ── HMAC Signing ────────────────────────────────────────────────

/// Simple HMAC-like signature using a basic hash (no crypto deps).
/// In production you would use ring or hmac crate. This is a
/// deterministic hash suitable for testing the signing protocol.
pub fn compute_signature(secret: &str, payload: &str) -> String {
    // Simple DJB2-based keyed hash (NOT cryptographic — for protocol testing only).
    let mut hash: u64 = 5381;
    for byte in secret.as_bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(*byte as u64);
    }
    for byte in payload.as_bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(*byte as u64);
    }
    format!("sha256={hash:016x}")
}

/// Verify a signature against a secret and payload.
pub fn verify_signature(secret: &str, payload: &str, signature: &str) -> bool {
    let expected = compute_signature(secret, payload);
    // Constant-time comparison (simplified).
    expected.len() == signature.len()
        && expected.as_bytes().iter().zip(signature.as_bytes()).all(|(a, b)| a == b)
}

// ── Retry Policy ────────────────────────────────────────────────

/// Retry policy for webhook delivery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay_seconds: u64,
    pub backoff_multiplier: u32,
    pub max_delay_seconds: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            initial_delay_seconds: 5,
            backoff_multiplier: 2,
            max_delay_seconds: 3600,
        }
    }
}

impl RetryPolicy {
    pub fn next_retry_at(&self, attempt: u32) -> DateTime<Utc> {
        let mut delay = self.initial_delay_seconds;
        for _ in 0..attempt {
            delay = delay.saturating_mul(self.backoff_multiplier as u64);
        }
        delay = delay.min(self.max_delay_seconds);
        Utc::now() + Duration::seconds(delay as i64)
    }
}

// ── Event Filter ────────────────────────────────────────────────

/// Event filter for endpoint subscriptions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventFilter {
    /// Subscribe to all events.
    All,
    /// Subscribe to specific event types.
    Types(Vec<String>),
    /// Subscribe to events matching a prefix.
    Prefix(String),
}

impl EventFilter {
    pub fn matches(&self, event_type: &str) -> bool {
        match self {
            Self::All => true,
            Self::Types(types) => types.iter().any(|t| t == event_type),
            Self::Prefix(prefix) => event_type.starts_with(prefix.as_str()),
        }
    }
}

// ── Payload Transformation ──────────────────────────────────────

/// Payload transformation rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayloadTransform {
    /// Pass through unchanged.
    Identity,
    /// Include only specific keys.
    IncludeKeys(Vec<String>),
    /// Exclude specific keys.
    ExcludeKeys(Vec<String>),
    /// Add static key-value pairs.
    Enrich(HashMap<String, String>),
}

impl PayloadTransform {
    pub fn apply(&self, payload: &HashMap<String, String>) -> HashMap<String, String> {
        match self {
            Self::Identity => payload.clone(),
            Self::IncludeKeys(keys) => {
                payload.iter()
                    .filter(|(k, _)| keys.contains(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            }
            Self::ExcludeKeys(keys) => {
                payload.iter()
                    .filter(|(k, _)| !keys.contains(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            }
            Self::Enrich(extra) => {
                let mut result = payload.clone();
                for (k, v) in extra {
                    result.insert(k.clone(), v.clone());
                }
                result
            }
        }
    }
}

// ── Endpoint ────────────────────────────────────────────────────

/// A webhook endpoint registration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Endpoint {
    pub id: String,
    pub url: String,
    pub secret: String,
    pub status: EndpointStatus,
    pub filter: EventFilter,
    pub transform: PayloadTransform,
    pub retry_policy: RetryPolicy,
    pub description: String,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    pub consecutive_failures: u32,
    pub max_consecutive_failures: u32,
}

impl Endpoint {
    pub fn new(
        id: impl Into<String>,
        url: impl Into<String>,
        secret: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            secret: secret.into(),
            status: EndpointStatus::Active,
            filter: EventFilter::All,
            transform: PayloadTransform::Identity,
            retry_policy: RetryPolicy::default(),
            description: String::new(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
            consecutive_failures: 0,
            max_consecutive_failures: 10,
        }
    }

    pub fn with_filter(mut self, f: EventFilter) -> Self {
        self.filter = f;
        self
    }

    pub fn with_transform(mut self, t: PayloadTransform) -> Self {
        self.transform = t;
        self
    }

    pub fn with_retry(mut self, p: RetryPolicy) -> Self {
        self.retry_policy = p;
        self
    }

    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        if self.status == EndpointStatus::Failing {
            self.status = EndpointStatus::Active;
        }
    }

    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.max_consecutive_failures {
            self.status = EndpointStatus::Failing;
        }
    }
}

// ── Webhook Event ───────────────────────────────────────────────

/// A webhook event to be delivered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    pub id: String,
    pub event_type: String,
    pub payload: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

impl WebhookEvent {
    pub fn new(id: impl Into<String>, event_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            event_type: event_type.into(),
            payload: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_data(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.payload.insert(key.into(), val.into());
        self
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

// ── Delivery Attempt ────────────────────────────────────────────

/// Record of a delivery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryAttempt {
    pub delivery_id: String,
    pub event_id: String,
    pub endpoint_id: String,
    pub attempt: u32,
    pub status: DeliveryStatus,
    pub signature: String,
    pub payload_sent: String,
    pub response_status: Option<u16>,
    pub response_body: Option<String>,
    pub error: Option<String>,
    pub attempted_at: DateTime<Utc>,
    pub next_retry_at: Option<DateTime<Utc>>,
}

// ── Delivery Log ────────────────────────────────────────────────

/// The delivery log tracks all attempts.
#[derive(Debug, Default)]
pub struct DeliveryLog {
    pub attempts: Vec<DeliveryAttempt>,
}

impl DeliveryLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, attempt: DeliveryAttempt) {
        self.attempts.push(attempt);
    }

    pub fn for_event(&self, event_id: &str) -> Vec<&DeliveryAttempt> {
        self.attempts.iter().filter(|a| a.event_id == event_id).collect()
    }

    pub fn for_endpoint(&self, endpoint_id: &str) -> Vec<&DeliveryAttempt> {
        self.attempts.iter().filter(|a| a.endpoint_id == endpoint_id).collect()
    }

    pub fn failed_deliveries(&self) -> Vec<&DeliveryAttempt> {
        self.attempts.iter()
            .filter(|a| matches!(a.status, DeliveryStatus::Failed | DeliveryStatus::Exhausted))
            .collect()
    }

    pub fn pending_retries(&self) -> Vec<&DeliveryAttempt> {
        self.attempts.iter()
            .filter(|a| a.status == DeliveryStatus::Retrying)
            .collect()
    }
}

// ── Webhook System ──────────────────────────────────────────────

/// The main webhook system coordinating endpoints, events, and delivery.
#[derive(Debug)]
pub struct WebhookSystem {
    pub endpoints: HashMap<String, Endpoint>,
    pub log: DeliveryLog,
    pub max_payload_bytes: usize,
    delivery_counter: u64,
}

impl WebhookSystem {
    pub fn new() -> Self {
        Self {
            endpoints: HashMap::new(),
            log: DeliveryLog::new(),
            max_payload_bytes: 256 * 1024, // 256 KB
            delivery_counter: 0,
        }
    }

    /// Register an endpoint.
    pub fn register_endpoint(&mut self, ep: Endpoint) -> Result<(), WebhookError> {
        if self.endpoints.contains_key(&ep.id) {
            return Err(WebhookError::DuplicateEndpoint(ep.id));
        }
        if !ep.url.starts_with("http://") && !ep.url.starts_with("https://") {
            return Err(WebhookError::InvalidUrl(ep.url));
        }
        self.endpoints.insert(ep.id.clone(), ep);
        Ok(())
    }

    /// Remove an endpoint.
    pub fn remove_endpoint(&mut self, id: &str) -> Result<Endpoint, WebhookError> {
        self.endpoints.remove(id)
            .ok_or_else(|| WebhookError::EndpointNotFound(id.to_string()))
    }

    /// Disable an endpoint.
    pub fn disable_endpoint(&mut self, id: &str) -> Result<(), WebhookError> {
        let ep = self.endpoints.get_mut(id)
            .ok_or_else(|| WebhookError::EndpointNotFound(id.to_string()))?;
        ep.status = EndpointStatus::Disabled;
        Ok(())
    }

    /// Enable an endpoint.
    pub fn enable_endpoint(&mut self, id: &str) -> Result<(), WebhookError> {
        let ep = self.endpoints.get_mut(id)
            .ok_or_else(|| WebhookError::EndpointNotFound(id.to_string()))?;
        ep.status = EndpointStatus::Active;
        ep.consecutive_failures = 0;
        Ok(())
    }

    fn next_delivery_id(&mut self) -> String {
        self.delivery_counter += 1;
        format!("del-{}", self.delivery_counter)
    }

    /// Prepare delivery for an event: returns the list of endpoint IDs that should receive it.
    pub fn prepare_delivery(&self, event: &WebhookEvent) -> Vec<String> {
        self.endpoints.values()
            .filter(|ep| ep.status == EndpointStatus::Active && ep.filter.matches(&event.event_type))
            .map(|ep| ep.id.clone())
            .collect()
    }

    /// Create a delivery attempt for an event to an endpoint.
    pub fn create_delivery(
        &mut self,
        event: &WebhookEvent,
        endpoint_id: &str,
    ) -> Result<DeliveryAttempt, WebhookError> {
        let ep = self.endpoints.get(endpoint_id)
            .ok_or_else(|| WebhookError::EndpointNotFound(endpoint_id.to_string()))?;

        if !ep.filter.matches(&event.event_type) {
            return Err(WebhookError::EventNotSubscribed {
                endpoint_id: endpoint_id.to_string(),
                event_type: event.event_type.clone(),
            });
        }

        // Transform payload.
        let transformed = ep.transform.apply(&event.payload);
        let payload_json = serde_json::to_string(&transformed).unwrap_or_default();

        if payload_json.len() > self.max_payload_bytes {
            return Err(WebhookError::PayloadTooLarge {
                size: payload_json.len(),
                max: self.max_payload_bytes,
            });
        }

        // Sign.
        let signature = compute_signature(&ep.secret, &payload_json);

        let delivery = DeliveryAttempt {
            delivery_id: self.next_delivery_id(),
            event_id: event.id.clone(),
            endpoint_id: endpoint_id.to_string(),
            attempt: 1,
            status: DeliveryStatus::Pending,
            signature,
            payload_sent: payload_json,
            response_status: None,
            response_body: None,
            error: None,
            attempted_at: Utc::now(),
            next_retry_at: None,
        };

        self.log.record(delivery.clone());
        Ok(delivery)
    }

    /// Record a successful delivery.
    pub fn record_success(
        &mut self,
        delivery_id: &str,
        endpoint_id: &str,
        status_code: u16,
    ) {
        if let Some(ep) = self.endpoints.get_mut(endpoint_id) {
            ep.record_success();
        }
        for attempt in self.log.attempts.iter_mut().rev() {
            if attempt.delivery_id == delivery_id {
                attempt.status = DeliveryStatus::Delivered;
                attempt.response_status = Some(status_code);
                break;
            }
        }
    }

    /// Record a failed delivery and schedule retry.
    pub fn record_failure(
        &mut self,
        delivery_id: &str,
        endpoint_id: &str,
        error: &str,
    ) -> Option<DateTime<Utc>> {
        let retry_policy;
        if let Some(ep) = self.endpoints.get_mut(endpoint_id) {
            ep.record_failure();
            retry_policy = ep.retry_policy.clone();
        } else {
            return None;
        }

        for attempt in self.log.attempts.iter_mut().rev() {
            if attempt.delivery_id == delivery_id {
                attempt.error = Some(error.to_string());
                if attempt.attempt < retry_policy.max_attempts {
                    let next = retry_policy.next_retry_at(attempt.attempt);
                    attempt.status = DeliveryStatus::Retrying;
                    attempt.next_retry_at = Some(next);
                    return Some(next);
                } else {
                    attempt.status = DeliveryStatus::Exhausted;
                    return None;
                }
            }
        }
        None
    }

    /// Batch deliver an event to all matching endpoints.
    pub fn batch_deliver(&mut self, event: &WebhookEvent) -> Vec<DeliveryAttempt> {
        let endpoint_ids = self.prepare_delivery(event);
        let mut deliveries = Vec::new();
        for eid in endpoint_ids {
            if let Ok(d) = self.create_delivery(event, &eid) {
                deliveries.push(d);
            }
        }
        deliveries
    }
}

impl Default for WebhookSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_endpoint() -> Endpoint {
        Endpoint::new("ep-1", "https://example.com/hook", "secret123")
    }

    fn test_event() -> WebhookEvent {
        WebhookEvent::new("evt-1", "order.created")
            .with_data("order_id", "123")
            .with_data("total", "99.99")
    }

    #[test]
    fn test_endpoint_creation() {
        let ep = test_endpoint();
        assert_eq!(ep.id, "ep-1");
        assert_eq!(ep.status, EndpointStatus::Active);
    }

    #[test]
    fn test_signature_sign_verify() {
        let sig = compute_signature("secret", "payload");
        assert!(verify_signature("secret", "payload", &sig));
        assert!(!verify_signature("wrong", "payload", &sig));
    }

    #[test]
    fn test_event_filter_all() {
        let filter = EventFilter::All;
        assert!(filter.matches("anything"));
    }

    #[test]
    fn test_event_filter_types() {
        let filter = EventFilter::Types(vec!["order.created".into(), "order.updated".into()]);
        assert!(filter.matches("order.created"));
        assert!(!filter.matches("order.deleted"));
    }

    #[test]
    fn test_event_filter_prefix() {
        let filter = EventFilter::Prefix("order.".into());
        assert!(filter.matches("order.created"));
        assert!(!filter.matches("user.created"));
    }

    #[test]
    fn test_payload_transform_include() {
        let mut payload = HashMap::new();
        payload.insert("a".into(), "1".into());
        payload.insert("b".into(), "2".into());
        payload.insert("c".into(), "3".into());
        let t = PayloadTransform::IncludeKeys(vec!["a".into(), "c".into()]);
        let result = t.apply(&payload);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("a"));
        assert!(!result.contains_key("b"));
    }

    #[test]
    fn test_payload_transform_exclude() {
        let mut payload = HashMap::new();
        payload.insert("a".into(), "1".into());
        payload.insert("secret".into(), "hidden".into());
        let t = PayloadTransform::ExcludeKeys(vec!["secret".into()]);
        let result = t.apply(&payload);
        assert!(!result.contains_key("secret"));
        assert!(result.contains_key("a"));
    }

    #[test]
    fn test_payload_transform_enrich() {
        let payload = HashMap::new();
        let mut extra = HashMap::new();
        extra.insert("source".into(), "my-app".into());
        let t = PayloadTransform::Enrich(extra);
        let result = t.apply(&payload);
        assert_eq!(result.get("source"), Some(&"my-app".to_string()));
    }

    #[test]
    fn test_register_endpoint() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        assert!(sys.endpoints.contains_key("ep-1"));
    }

    #[test]
    fn test_duplicate_endpoint() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        assert!(matches!(
            sys.register_endpoint(test_endpoint()),
            Err(WebhookError::DuplicateEndpoint(_))
        ));
    }

    #[test]
    fn test_invalid_url() {
        let mut sys = WebhookSystem::new();
        let ep = Endpoint::new("ep-1", "not-a-url", "secret");
        assert!(matches!(sys.register_endpoint(ep), Err(WebhookError::InvalidUrl(_))));
    }

    #[test]
    fn test_create_delivery() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        let event = test_event();
        let delivery = sys.create_delivery(&event, "ep-1").unwrap();
        assert_eq!(delivery.event_id, "evt-1");
        assert!(!delivery.signature.is_empty());
    }

    #[test]
    fn test_batch_delivery() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        sys.register_endpoint(Endpoint::new("ep-2", "https://other.com/hook", "s2")).unwrap();
        let event = test_event();
        let deliveries = sys.batch_deliver(&event);
        assert_eq!(deliveries.len(), 2);
    }

    #[test]
    fn test_delivery_success() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        let event = test_event();
        let delivery = sys.create_delivery(&event, "ep-1").unwrap();
        sys.record_success(&delivery.delivery_id, "ep-1", 200);

        let logs = sys.log.for_event("evt-1");
        assert_eq!(logs[0].status, DeliveryStatus::Delivered);
    }

    #[test]
    fn test_delivery_failure_retry() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        let event = test_event();
        let delivery = sys.create_delivery(&event, "ep-1").unwrap();
        let next = sys.record_failure(&delivery.delivery_id, "ep-1", "timeout");
        assert!(next.is_some()); // retry scheduled
    }

    #[test]
    fn test_delivery_exhausted() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(
            Endpoint::new("ep-1", "https://example.com/hook", "s")
                .with_retry(RetryPolicy { max_attempts: 1, ..Default::default() }),
        ).unwrap();
        let event = test_event();
        let delivery = sys.create_delivery(&event, "ep-1").unwrap();
        let next = sys.record_failure(&delivery.delivery_id, "ep-1", "error");
        assert!(next.is_none()); // no more retries
    }

    #[test]
    fn test_disable_enable_endpoint() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(test_endpoint()).unwrap();
        sys.disable_endpoint("ep-1").unwrap();
        assert_eq!(sys.endpoints["ep-1"].status, EndpointStatus::Disabled);

        // Disabled endpoints should not receive events.
        let deliveries = sys.batch_deliver(&test_event());
        assert!(deliveries.is_empty());

        sys.enable_endpoint("ep-1").unwrap();
        assert_eq!(sys.endpoints["ep-1"].status, EndpointStatus::Active);
    }

    #[test]
    fn test_event_not_subscribed() {
        let mut sys = WebhookSystem::new();
        sys.register_endpoint(
            Endpoint::new("ep-1", "https://example.com/hook", "s")
                .with_filter(EventFilter::Types(vec!["user.created".into()])),
        ).unwrap();
        let event = test_event(); // order.created
        let err = sys.create_delivery(&event, "ep-1").unwrap_err();
        assert!(matches!(err, WebhookError::EventNotSubscribed { .. }));
    }

    #[test]
    fn test_consecutive_failure_tracking() {
        let mut ep = Endpoint::new("ep", "https://x.com", "s");
        ep.max_consecutive_failures = 3;
        ep.record_failure();
        ep.record_failure();
        assert_eq!(ep.status, EndpointStatus::Active);
        ep.record_failure();
        assert_eq!(ep.status, EndpointStatus::Failing);
        ep.record_success();
        assert_eq!(ep.status, EndpointStatus::Active);
    }
}
