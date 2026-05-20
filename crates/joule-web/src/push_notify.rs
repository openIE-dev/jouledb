//! Push notifications — VAPID key management, subscription modeling,
//! notification payload construction, urgency/TTL, topic deduplication,
//! and action buttons.
//!
//! Pure-Rust push notification types and logic. No actual HTTP requests
//! are made; callers use the constructed payloads with their own transport.

use std::collections::HashMap;
use std::fmt;

// ── VAPID keys ─────────────────────────────────────────────────────

/// VAPID (Voluntary Application Server Identification) key pair.
/// Stores raw key bytes — actual crypto is deferred to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VapidKeys {
    /// Base64url-encoded public key (P-256, uncompressed, 65 bytes).
    pub public_key: String,
    /// Base64url-encoded private key (P-256, 32 bytes).
    pub private_key: String,
    /// Contact email or URL for the application server.
    pub subject: String,
}

impl VapidKeys {
    pub fn new(
        public_key: impl Into<String>,
        private_key: impl Into<String>,
        subject: impl Into<String>,
    ) -> Self {
        Self {
            public_key: public_key.into(),
            private_key: private_key.into(),
            subject: subject.into(),
        }
    }

    /// Construct the VAPID `Authorization` header value (unsigned — the caller
    /// performs the actual JWT signing).
    pub fn authorization_header_claims(&self, audience: &str, expiry_secs: u64) -> VapidClaims {
        VapidClaims {
            aud: audience.to_string(),
            exp: expiry_secs,
            sub: self.subject.clone(),
        }
    }
}

/// Claims for a VAPID JWT token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VapidClaims {
    pub aud: String,
    pub exp: u64,
    pub sub: String,
}

impl VapidClaims {
    /// Render as a minimal JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"aud":"{}","exp":{},"sub":"{}"}}"#,
            self.aud, self.exp, self.sub
        )
    }
}

// ── Push subscription ──────────────────────────────────────────────

/// A push subscription from a client (matches the Push API PushSubscription).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PushSubscription {
    pub endpoint: String,
    /// P-256 public key for message encryption (base64url).
    pub p256dh: String,
    /// Authentication secret (base64url).
    pub auth: String,
    /// Optional expiration timestamp (seconds since epoch).
    pub expiration_time: Option<u64>,
}

impl PushSubscription {
    pub fn new(
        endpoint: impl Into<String>,
        p256dh: impl Into<String>,
        auth: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            p256dh: p256dh.into(),
            auth: auth.into(),
            expiration_time: None,
        }
    }

    pub fn with_expiration(mut self, ts: u64) -> Self {
        self.expiration_time = Some(ts);
        self
    }

    /// Whether the subscription has expired at the given timestamp.
    pub fn is_expired_at(&self, now_secs: u64) -> bool {
        self.expiration_time.map_or(false, |exp| now_secs >= exp)
    }

    /// Extract the origin from the endpoint URL.
    pub fn origin(&self) -> Option<&str> {
        // Find "://" then the next "/"
        let rest = self.endpoint.find("://").map(|i| &self.endpoint[i + 3..])?;
        let end = rest.find('/').unwrap_or(rest.len());
        let scheme_end = self.endpoint.find("://").unwrap() + 3 + end;
        Some(&self.endpoint[..scheme_end])
    }
}

// ── Notification urgency ───────────────────────────────────────────

/// Push message urgency (RFC 8030 §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Urgency {
    VeryLow,
    Low,
    Normal,
    High,
}

impl Urgency {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VeryLow => "very-low",
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "very-low" => Some(Self::VeryLow),
            "low" => Some(Self::Low),
            "normal" => Some(Self::Normal),
            "high" => Some(Self::High),
            _ => None,
        }
    }
}

impl fmt::Display for Urgency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Notification action ────────────────────────────────────────────

/// An action button on a notification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotificationAction {
    pub action: String,
    pub title: String,
    pub icon: Option<String>,
}

impl NotificationAction {
    pub fn new(action: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            title: title.into(),
            icon: None,
        }
    }

    pub fn with_icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = Some(icon.into());
        self
    }

    fn to_json(&self) -> String {
        match &self.icon {
            Some(icon) => format!(
                r#"{{"action":"{}","title":"{}","icon":"{}"}}"#,
                self.action, self.title, icon
            ),
            None => format!(
                r#"{{"action":"{}","title":"{}"}}"#,
                self.action, self.title
            ),
        }
    }
}

// ── Notification payload ───────────────────────────────────────────

/// Direction for notification text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationDir {
    Auto,
    Ltr,
    Rtl,
}

impl NotificationDir {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Ltr => "ltr",
            Self::Rtl => "rtl",
        }
    }
}

/// A push notification payload (matches the Notification API).
#[derive(Debug, Clone)]
pub struct NotificationPayload {
    pub title: String,
    pub body: Option<String>,
    pub icon: Option<String>,
    pub badge: Option<String>,
    pub image: Option<String>,
    pub tag: Option<String>,
    pub data: Option<String>,
    pub actions: Vec<NotificationAction>,
    pub dir: NotificationDir,
    pub lang: Option<String>,
    pub renotify: bool,
    pub require_interaction: bool,
    pub silent: bool,
    pub timestamp: Option<u64>,
}

impl NotificationPayload {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: None,
            icon: None,
            badge: None,
            image: None,
            tag: None,
            data: None,
            actions: Vec::new(),
            dir: NotificationDir::Auto,
            lang: None,
            renotify: false,
            require_interaction: false,
            silent: false,
            timestamp: None,
        }
    }

    pub fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(body.into());
        self
    }

    pub fn with_icon(mut self, icon: impl Into<String>) -> Self {
        self.icon = Some(icon.into());
        self
    }

    pub fn with_badge(mut self, badge: impl Into<String>) -> Self {
        self.badge = Some(badge.into());
        self
    }

    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = Some(image.into());
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    pub fn with_data(mut self, data: impl Into<String>) -> Self {
        self.data = Some(data.into());
        self
    }

    pub fn with_action(mut self, action: NotificationAction) -> Self {
        self.actions.push(action);
        self
    }

    pub fn with_dir(mut self, dir: NotificationDir) -> Self {
        self.dir = dir;
        self
    }

    pub fn with_renotify(mut self, renotify: bool) -> Self {
        self.renotify = renotify;
        self
    }

    pub fn with_require_interaction(mut self, require: bool) -> Self {
        self.require_interaction = require;
        self
    }

    pub fn with_silent(mut self, silent: bool) -> Self {
        self.silent = silent;
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Serialize the payload to a JSON string.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();
        parts.push(format!(r#""title":"{}""#, self.title));
        if let Some(ref b) = self.body {
            parts.push(format!(r#""body":"{}""#, b));
        }
        if let Some(ref i) = self.icon {
            parts.push(format!(r#""icon":"{}""#, i));
        }
        if let Some(ref b) = self.badge {
            parts.push(format!(r#""badge":"{}""#, b));
        }
        if let Some(ref img) = self.image {
            parts.push(format!(r#""image":"{}""#, img));
        }
        if let Some(ref t) = self.tag {
            parts.push(format!(r#""tag":"{}""#, t));
        }
        if let Some(ref d) = self.data {
            parts.push(format!(r#""data":{}"#, d));
        }
        if !self.actions.is_empty() {
            let actions_json: Vec<String> = self.actions.iter().map(|a| a.to_json()).collect();
            parts.push(format!(r#""actions":[{}]"#, actions_json.join(",")));
        }
        if self.dir != NotificationDir::Auto {
            parts.push(format!(r#""dir":"{}""#, self.dir.as_str()));
        }
        if let Some(ref l) = self.lang {
            parts.push(format!(r#""lang":"{}""#, l));
        }
        if self.renotify {
            parts.push(r#""renotify":true"#.to_string());
        }
        if self.require_interaction {
            parts.push(r#""requireInteraction":true"#.to_string());
        }
        if self.silent {
            parts.push(r#""silent":true"#.to_string());
        }
        if let Some(ts) = self.timestamp {
            parts.push(format!(r#""timestamp":{}"#, ts));
        }
        format!("{{{}}}", parts.join(","))
    }
}

// ── Push message (full request) ────────────────────────────────────

/// A complete push message ready for delivery.
#[derive(Debug, Clone)]
pub struct PushMessage {
    pub subscription: PushSubscription,
    pub payload: NotificationPayload,
    pub urgency: Urgency,
    /// Time-to-live in seconds (0 = immediate delivery only).
    pub ttl: u32,
    /// Topic for message replacement/deduplication.
    pub topic: Option<String>,
}

impl PushMessage {
    pub fn new(subscription: PushSubscription, payload: NotificationPayload) -> Self {
        Self {
            subscription,
            payload,
            urgency: Urgency::Normal,
            ttl: 86400, // 24 hours default
            topic: None,
        }
    }

    pub fn with_urgency(mut self, urgency: Urgency) -> Self {
        self.urgency = urgency;
        self
    }

    pub fn with_ttl(mut self, ttl: u32) -> Self {
        self.ttl = ttl;
        self
    }

    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = Some(topic.into());
        self
    }

    /// Build HTTP headers for the push request (without Authorization).
    pub fn headers(&self) -> HashMap<String, String> {
        let mut h = HashMap::new();
        h.insert("Content-Type".to_string(), "application/json".to_string());
        h.insert("TTL".to_string(), self.ttl.to_string());
        h.insert("Urgency".to_string(), self.urgency.as_str().to_string());
        if let Some(ref topic) = self.topic {
            h.insert("Topic".to_string(), topic.clone());
        }
        h
    }
}

// ── Topic deduplication ────────────────────────────────────────────

/// Manages topic-based deduplication of push messages.
#[derive(Debug, Default)]
pub struct TopicDeduplicator {
    /// Maps topic -> most recent message payload JSON.
    latest: HashMap<String, String>,
}

impl TopicDeduplicator {
    pub fn new() -> Self {
        Self { latest: HashMap::new() }
    }

    /// Record a message. Returns true if this is a new topic or updated content.
    pub fn record(&mut self, topic: &str, payload_json: &str) -> bool {
        let prev = self.latest.insert(topic.to_string(), payload_json.to_string());
        prev.as_deref() != Some(payload_json)
    }

    /// Check if a topic has been seen.
    pub fn has_topic(&self, topic: &str) -> bool {
        self.latest.contains_key(topic)
    }

    /// Clear all tracked topics.
    pub fn clear(&mut self) {
        self.latest.clear();
    }

    /// Number of tracked topics.
    pub fn topic_count(&self) -> usize {
        self.latest.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VAPID ──────────────────────────────────────────────────────

    #[test]
    fn vapid_keys_creation() {
        let keys = VapidKeys::new("pub123", "priv456", "mailto:admin@example.com");
        assert_eq!(keys.public_key, "pub123");
        assert_eq!(keys.private_key, "priv456");
        assert_eq!(keys.subject, "mailto:admin@example.com");
    }

    #[test]
    fn vapid_claims_json() {
        let keys = VapidKeys::new("pub", "priv", "mailto:test@example.com");
        let claims = keys.authorization_header_claims("https://push.example.com", 1700000000);
        let json = claims.to_json();
        assert!(json.contains(r#""aud":"https://push.example.com""#));
        assert!(json.contains(r#""exp":1700000000"#));
        assert!(json.contains(r#""sub":"mailto:test@example.com""#));
    }

    // ── Subscription ───────────────────────────────────────────────

    #[test]
    fn subscription_creation() {
        let sub = PushSubscription::new(
            "https://fcm.googleapis.com/fcm/send/abc",
            "key123",
            "auth456",
        );
        assert_eq!(sub.endpoint, "https://fcm.googleapis.com/fcm/send/abc");
        assert_eq!(sub.p256dh, "key123");
        assert_eq!(sub.auth, "auth456");
        assert!(sub.expiration_time.is_none());
    }

    #[test]
    fn subscription_expiration() {
        let sub = PushSubscription::new("https://example.com/push", "k", "a")
            .with_expiration(1000);
        assert!(!sub.is_expired_at(999));
        assert!(sub.is_expired_at(1000));
        assert!(sub.is_expired_at(1001));
    }

    #[test]
    fn subscription_no_expiration_never_expired() {
        let sub = PushSubscription::new("https://example.com/push", "k", "a");
        assert!(!sub.is_expired_at(u64::MAX));
    }

    #[test]
    fn subscription_origin() {
        let sub = PushSubscription::new(
            "https://fcm.googleapis.com/fcm/send/abc",
            "k", "a",
        );
        assert_eq!(sub.origin(), Some("https://fcm.googleapis.com"));
    }

    // ── Urgency ────────────────────────────────────────────────────

    #[test]
    fn urgency_roundtrip() {
        for u in [Urgency::VeryLow, Urgency::Low, Urgency::Normal, Urgency::High] {
            assert_eq!(Urgency::from_str(u.as_str()), Some(u));
        }
    }

    #[test]
    fn urgency_unknown() {
        assert_eq!(Urgency::from_str("invalid"), None);
    }

    #[test]
    fn urgency_display() {
        assert_eq!(Urgency::High.to_string(), "high");
        assert_eq!(Urgency::VeryLow.to_string(), "very-low");
    }

    // ── Notification action ────────────────────────────────────────

    #[test]
    fn action_json_without_icon() {
        let a = NotificationAction::new("reply", "Reply");
        let json = a.to_json();
        assert!(json.contains(r#""action":"reply""#));
        assert!(json.contains(r#""title":"Reply""#));
        assert!(!json.contains("icon"));
    }

    #[test]
    fn action_json_with_icon() {
        let a = NotificationAction::new("open", "Open")
            .with_icon("/icons/open.png");
        let json = a.to_json();
        assert!(json.contains(r#""icon":"/icons/open.png""#));
    }

    // ── Notification payload ───────────────────────────────────────

    #[test]
    fn payload_minimal() {
        let p = NotificationPayload::new("Test");
        let json = p.to_json();
        assert!(json.contains(r#""title":"Test""#));
        assert!(!json.contains("body"));
    }

    #[test]
    fn payload_with_body_and_icon() {
        let p = NotificationPayload::new("Alert")
            .with_body("Something happened")
            .with_icon("/icon.png");
        let json = p.to_json();
        assert!(json.contains(r#""body":"Something happened""#));
        assert!(json.contains(r#""icon":"/icon.png""#));
    }

    #[test]
    fn payload_with_actions() {
        let p = NotificationPayload::new("Message")
            .with_action(NotificationAction::new("reply", "Reply"))
            .with_action(NotificationAction::new("dismiss", "Dismiss"));
        let json = p.to_json();
        assert!(json.contains(r#""actions":[{"action":"reply","title":"Reply"},{"action":"dismiss","title":"Dismiss"}]"#));
    }

    #[test]
    fn payload_with_tag() {
        let p = NotificationPayload::new("Update")
            .with_tag("msg-123");
        let json = p.to_json();
        assert!(json.contains(r#""tag":"msg-123""#));
    }

    #[test]
    fn payload_with_badge_and_image() {
        let p = NotificationPayload::new("Photo")
            .with_badge("/badge.png")
            .with_image("/photo.jpg");
        let json = p.to_json();
        assert!(json.contains(r#""badge":"/badge.png""#));
        assert!(json.contains(r#""image":"/photo.jpg""#));
    }

    #[test]
    fn payload_with_dir_and_lang() {
        let p = NotificationPayload::new("Test")
            .with_dir(NotificationDir::Rtl);
        let json = p.to_json();
        assert!(json.contains(r#""dir":"rtl""#));
    }

    #[test]
    fn payload_flags() {
        let p = NotificationPayload::new("Urgent")
            .with_renotify(true)
            .with_require_interaction(true)
            .with_silent(true)
            .with_timestamp(1700000000);
        let json = p.to_json();
        assert!(json.contains(r#""renotify":true"#));
        assert!(json.contains(r#""requireInteraction":true"#));
        assert!(json.contains(r#""silent":true"#));
        assert!(json.contains(r#""timestamp":1700000000"#));
    }

    #[test]
    fn payload_data_passthrough() {
        let p = NotificationPayload::new("Test")
            .with_data(r#"{"key":"value"}"#);
        let json = p.to_json();
        assert!(json.contains(r#""data":{"key":"value"}"#));
    }

    // ── Push message ───────────────────────────────────────────────

    #[test]
    fn push_message_defaults() {
        let sub = PushSubscription::new("https://example.com/push", "k", "a");
        let payload = NotificationPayload::new("Test");
        let msg = PushMessage::new(sub, payload);
        assert_eq!(msg.urgency, Urgency::Normal);
        assert_eq!(msg.ttl, 86400);
        assert!(msg.topic.is_none());
    }

    #[test]
    fn push_message_headers() {
        let sub = PushSubscription::new("https://example.com/push", "k", "a");
        let payload = NotificationPayload::new("Test");
        let msg = PushMessage::new(sub, payload)
            .with_urgency(Urgency::High)
            .with_ttl(3600)
            .with_topic("chat-42");
        let h = msg.headers();
        assert_eq!(h.get("TTL").unwrap(), "3600");
        assert_eq!(h.get("Urgency").unwrap(), "high");
        assert_eq!(h.get("Topic").unwrap(), "chat-42");
        assert_eq!(h.get("Content-Type").unwrap(), "application/json");
    }

    #[test]
    fn push_message_no_topic_header() {
        let sub = PushSubscription::new("https://example.com/push", "k", "a");
        let payload = NotificationPayload::new("Test");
        let msg = PushMessage::new(sub, payload);
        let h = msg.headers();
        assert!(!h.contains_key("Topic"));
    }

    // ── Topic deduplication ────────────────────────────────────────

    #[test]
    fn dedup_new_topic() {
        let mut dedup = TopicDeduplicator::new();
        assert!(dedup.record("chat-1", r#"{"title":"Hi"}"#));
        assert!(dedup.has_topic("chat-1"));
        assert_eq!(dedup.topic_count(), 1);
    }

    #[test]
    fn dedup_same_content() {
        let mut dedup = TopicDeduplicator::new();
        dedup.record("chat-1", r#"{"title":"Hi"}"#);
        // Same content = not new
        assert!(!dedup.record("chat-1", r#"{"title":"Hi"}"#));
    }

    #[test]
    fn dedup_updated_content() {
        let mut dedup = TopicDeduplicator::new();
        dedup.record("chat-1", r#"{"title":"Hi"}"#);
        // Different content = new
        assert!(dedup.record("chat-1", r#"{"title":"Hello"}"#));
    }

    #[test]
    fn dedup_clear() {
        let mut dedup = TopicDeduplicator::new();
        dedup.record("t1", "a");
        dedup.record("t2", "b");
        dedup.clear();
        assert_eq!(dedup.topic_count(), 0);
        assert!(!dedup.has_topic("t1"));
    }

    // ── NotificationDir ────────────────────────────────────────────

    #[test]
    fn notification_dir_as_str() {
        assert_eq!(NotificationDir::Auto.as_str(), "auto");
        assert_eq!(NotificationDir::Ltr.as_str(), "ltr");
        assert_eq!(NotificationDir::Rtl.as_str(), "rtl");
    }
}
