//! Notification engine — channels (email/sms/push/webhook), template rendering,
//! recipient management, delivery tracking, retry on failure, notification
//! preferences, and batch notifications.
//!
//! Replaces Node.js notification libraries (Novu, node-notifier, Notifme) with
//! a pure-Rust notification engine that tracks every message from creation to delivery.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Notification engine domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotificationError {
    /// Template not found.
    TemplateNotFound(String),
    /// Recipient not found.
    RecipientNotFound(String),
    /// Notification not found.
    NotificationNotFound(String),
    /// Channel not configured for recipient.
    ChannelNotConfigured { recipient: String, channel: String },
    /// Duplicate template ID.
    DuplicateTemplate(String),
    /// Duplicate recipient ID.
    DuplicateRecipient(String),
    /// Template render error.
    RenderError(String),
    /// Max retries exceeded.
    MaxRetriesExceeded { notification_id: String, attempts: u32 },
    /// Batch is empty.
    EmptyBatch,
}

impl std::fmt::Display for NotificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TemplateNotFound(id) => write!(f, "template not found: {id}"),
            Self::RecipientNotFound(id) => write!(f, "recipient not found: {id}"),
            Self::NotificationNotFound(id) => write!(f, "notification not found: {id}"),
            Self::ChannelNotConfigured { recipient, channel } => {
                write!(f, "channel {channel} not configured for {recipient}")
            }
            Self::DuplicateTemplate(id) => write!(f, "duplicate template: {id}"),
            Self::DuplicateRecipient(id) => write!(f, "duplicate recipient: {id}"),
            Self::RenderError(msg) => write!(f, "render error: {msg}"),
            Self::MaxRetriesExceeded {
                notification_id,
                attempts,
            } => {
                write!(f, "notification {notification_id} max retries exceeded ({attempts})")
            }
            Self::EmptyBatch => write!(f, "batch is empty"),
        }
    }
}

impl std::error::Error for NotificationError {}

// ── Enums ───────────────────────────────────────────────────────

/// Notification delivery channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    Email,
    Sms,
    Push,
    Webhook,
    InApp,
}

/// Delivery status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeliveryStatus {
    Pending,
    Queued,
    Sent,
    Delivered,
    Failed,
    Bounced,
    Retrying,
}

/// Notification priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Normal,
    High,
    Urgent,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

// ── Data Structures ─────────────────────────────────────────────

/// A notification template with placeholders like `{{name}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTemplate {
    pub id: String,
    pub name: String,
    pub channel: Channel,
    pub subject_template: Option<String>,
    pub body_template: String,
    pub default_priority: Priority,
    pub created_at: DateTime<Utc>,
}

/// Recipient preferences per channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    pub channel: Channel,
    pub address: String,
    pub enabled: bool,
}

/// A notification recipient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipient {
    pub id: String,
    pub name: String,
    pub channels: Vec<ChannelConfig>,
    pub preferences: NotificationPreferences,
}

/// Recipient notification preferences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    /// Channels the recipient wants to receive.
    pub enabled_channels: Vec<Channel>,
    /// Quiet hours: if set, notifications are held during this window.
    pub quiet_start_hour: Option<u8>,
    pub quiet_end_hour: Option<u8>,
    /// Opt-out of specific template categories.
    pub opted_out_categories: Vec<String>,
}

impl Default for NotificationPreferences {
    fn default() -> Self {
        Self {
            enabled_channels: vec![Channel::Email, Channel::InApp],
            quiet_start_hour: None,
            quiet_end_hour: None,
            opted_out_categories: Vec::new(),
        }
    }
}

/// A delivery attempt record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryAttempt {
    pub attempt_number: u32,
    pub timestamp: DateTime<Utc>,
    pub status: DeliveryStatus,
    pub error_message: Option<String>,
}

/// A single notification instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub id: String,
    pub template_id: String,
    pub recipient_id: String,
    pub channel: Channel,
    pub subject: Option<String>,
    pub body: String,
    pub priority: Priority,
    pub status: DeliveryStatus,
    pub attempts: Vec<DeliveryAttempt>,
    pub created_at: DateTime<Utc>,
    pub delivered_at: Option<DateTime<Utc>>,
    pub category: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Batch notification request.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub template_id: String,
    pub recipient_ids: Vec<String>,
    pub variables: HashMap<String, String>,
    pub category: Option<String>,
    pub priority: Option<Priority>,
}

/// Retry policy for failed notifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_delay_seconds: u64,
    pub backoff_multiplier: u32,
    pub max_delay_seconds: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_seconds: 30,
            backoff_multiplier: 2,
            max_delay_seconds: 3600,
        }
    }
}

// ── Template Rendering ──────────────────────────────────────────

/// Render a template string by replacing `{{key}}` with values.
fn render_template(
    template: &str,
    variables: &HashMap<String, String>,
) -> Result<String, NotificationError> {
    let mut result = template.to_string();
    for (key, value) in variables {
        let placeholder = format!("{{{{{key}}}}}");
        result = result.replace(&placeholder, value);
    }
    // Check for unresolved placeholders.
    if result.contains("{{") && result.contains("}}") {
        let start = result.find("{{").unwrap();
        let end = result[start..].find("}}").map(|i| i + start + 2).unwrap_or(result.len());
        let unresolved = &result[start..end];
        return Err(NotificationError::RenderError(format!(
            "unresolved placeholder: {unresolved}"
        )));
    }
    Ok(result)
}

// ── Engine ──────────────────────────────────────────────────────

/// Notification engine manages templates, recipients, and delivery.
pub struct NotificationEngine {
    templates: HashMap<String, NotificationTemplate>,
    recipients: HashMap<String, Recipient>,
    notifications: Vec<Notification>,
    retry_policy: RetryPolicy,
    next_id: u64,
}

impl NotificationEngine {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            recipients: HashMap::new(),
            notifications: Vec::new(),
            retry_policy: RetryPolicy::default(),
            next_id: 1,
        }
    }

    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    fn generate_id(&mut self) -> String {
        let id = format!("notif-{}", self.next_id);
        self.next_id += 1;
        id
    }

    // ── Template Management ─────────────────────────────────────

    /// Register a notification template.
    pub fn register_template(&mut self, template: NotificationTemplate) -> Result<(), NotificationError> {
        if self.templates.contains_key(&template.id) {
            return Err(NotificationError::DuplicateTemplate(template.id.clone()));
        }
        self.templates.insert(template.id.clone(), template);
        Ok(())
    }

    /// Get a template by ID.
    pub fn get_template(&self, id: &str) -> Option<&NotificationTemplate> {
        self.templates.get(id)
    }

    /// Remove a template.
    pub fn remove_template(&mut self, id: &str) -> Result<NotificationTemplate, NotificationError> {
        self.templates
            .remove(id)
            .ok_or_else(|| NotificationError::TemplateNotFound(id.to_string()))
    }

    // ── Recipient Management ────────────────────────────────────

    /// Register a recipient.
    pub fn register_recipient(&mut self, recipient: Recipient) -> Result<(), NotificationError> {
        if self.recipients.contains_key(&recipient.id) {
            return Err(NotificationError::DuplicateRecipient(recipient.id.clone()));
        }
        self.recipients.insert(recipient.id.clone(), recipient);
        Ok(())
    }

    /// Get a recipient by ID.
    pub fn get_recipient(&self, id: &str) -> Option<&Recipient> {
        self.recipients.get(id)
    }

    /// Update recipient preferences.
    pub fn update_preferences(
        &mut self,
        recipient_id: &str,
        prefs: NotificationPreferences,
    ) -> Result<(), NotificationError> {
        let r = self
            .recipients
            .get_mut(recipient_id)
            .ok_or_else(|| NotificationError::RecipientNotFound(recipient_id.to_string()))?;
        r.preferences = prefs;
        Ok(())
    }

    // ── Send Notifications ──────────────────────────────────────

    /// Send a notification using a template.
    pub fn send(
        &mut self,
        template_id: &str,
        recipient_id: &str,
        variables: &HashMap<String, String>,
        category: Option<String>,
    ) -> Result<String, NotificationError> {
        let template = self
            .templates
            .get(template_id)
            .ok_or_else(|| NotificationError::TemplateNotFound(template_id.to_string()))?
            .clone();

        let recipient = self
            .recipients
            .get(recipient_id)
            .ok_or_else(|| NotificationError::RecipientNotFound(recipient_id.to_string()))?
            .clone();

        // Check preferences.
        if !recipient.preferences.enabled_channels.contains(&template.channel) {
            return Err(NotificationError::ChannelNotConfigured {
                recipient: recipient_id.to_string(),
                channel: format!("{:?}", template.channel),
            });
        }

        if let Some(cat) = &category {
            if recipient.preferences.opted_out_categories.contains(cat) {
                return Err(NotificationError::ChannelNotConfigured {
                    recipient: recipient_id.to_string(),
                    channel: format!("opted out of category: {cat}"),
                });
            }
        }

        // Render.
        let body = render_template(&template.body_template, variables)?;
        let subject = if let Some(subj_t) = &template.subject_template {
            Some(render_template(subj_t, variables)?)
        } else {
            None
        };

        let now = Utc::now();
        let id = self.generate_id();

        let notification = Notification {
            id: id.clone(),
            template_id: template_id.to_string(),
            recipient_id: recipient_id.to_string(),
            channel: template.channel,
            subject,
            body,
            priority: template.default_priority,
            status: DeliveryStatus::Queued,
            attempts: vec![DeliveryAttempt {
                attempt_number: 1,
                timestamp: now,
                status: DeliveryStatus::Queued,
                error_message: None,
            }],
            created_at: now,
            delivered_at: None,
            category,
            metadata: HashMap::new(),
        };

        self.notifications.push(notification);
        Ok(id)
    }

    /// Send batch notifications to multiple recipients.
    pub fn send_batch(
        &mut self,
        request: &BatchRequest,
    ) -> Result<Vec<String>, NotificationError> {
        if request.recipient_ids.is_empty() {
            return Err(NotificationError::EmptyBatch);
        }

        let mut ids = Vec::new();
        let recipient_ids = request.recipient_ids.clone();
        for rid in &recipient_ids {
            match self.send(
                &request.template_id,
                rid,
                &request.variables,
                request.category.clone(),
            ) {
                Ok(id) => ids.push(id),
                Err(_) => {
                    // Skip recipients that fail (e.g., opted out).
                    continue;
                }
            }
        }
        Ok(ids)
    }

    // ── Delivery Tracking ───────────────────────────────────────

    /// Mark a notification as delivered.
    pub fn mark_delivered(&mut self, notification_id: &str) -> Result<(), NotificationError> {
        let notif = self
            .notifications
            .iter_mut()
            .find(|n| n.id == notification_id)
            .ok_or_else(|| {
                NotificationError::NotificationNotFound(notification_id.to_string())
            })?;

        let now = Utc::now();
        notif.status = DeliveryStatus::Delivered;
        notif.delivered_at = Some(now);
        notif.attempts.push(DeliveryAttempt {
            attempt_number: notif.attempts.len() as u32 + 1,
            timestamp: now,
            status: DeliveryStatus::Delivered,
            error_message: None,
        });
        Ok(())
    }

    /// Mark a notification as failed and potentially retry.
    pub fn mark_failed(
        &mut self,
        notification_id: &str,
        error: &str,
    ) -> Result<DeliveryStatus, NotificationError> {
        let max_attempts = self.retry_policy.max_attempts;
        let notif = self
            .notifications
            .iter_mut()
            .find(|n| n.id == notification_id)
            .ok_or_else(|| {
                NotificationError::NotificationNotFound(notification_id.to_string())
            })?;

        let now = Utc::now();
        let attempt_count = notif.attempts.len() as u32;

        if attempt_count < max_attempts {
            notif.status = DeliveryStatus::Retrying;
            notif.attempts.push(DeliveryAttempt {
                attempt_number: attempt_count + 1,
                timestamp: now,
                status: DeliveryStatus::Retrying,
                error_message: Some(error.to_string()),
            });
            Ok(DeliveryStatus::Retrying)
        } else {
            notif.status = DeliveryStatus::Failed;
            notif.attempts.push(DeliveryAttempt {
                attempt_number: attempt_count + 1,
                timestamp: now,
                status: DeliveryStatus::Failed,
                error_message: Some(error.to_string()),
            });
            Ok(DeliveryStatus::Failed)
        }
    }

    /// Mark a notification as bounced.
    pub fn mark_bounced(&mut self, notification_id: &str) -> Result<(), NotificationError> {
        let notif = self
            .notifications
            .iter_mut()
            .find(|n| n.id == notification_id)
            .ok_or_else(|| {
                NotificationError::NotificationNotFound(notification_id.to_string())
            })?;
        notif.status = DeliveryStatus::Bounced;
        Ok(())
    }

    // ── Querying ────────────────────────────────────────────────

    /// Get notification by ID.
    pub fn get_notification(&self, id: &str) -> Option<&Notification> {
        self.notifications.iter().find(|n| n.id == id)
    }

    /// Get all notifications for a recipient.
    pub fn notifications_for_recipient(&self, recipient_id: &str) -> Vec<&Notification> {
        self.notifications
            .iter()
            .filter(|n| n.recipient_id == recipient_id)
            .collect()
    }

    /// Get notifications by status.
    pub fn notifications_by_status(&self, status: DeliveryStatus) -> Vec<&Notification> {
        self.notifications
            .iter()
            .filter(|n| n.status == status)
            .collect()
    }

    /// Get notifications needing retry.
    pub fn pending_retries(&self) -> Vec<&Notification> {
        self.notifications
            .iter()
            .filter(|n| n.status == DeliveryStatus::Retrying)
            .collect()
    }

    /// Compute the next retry time for a notification.
    pub fn next_retry_time(&self, notification_id: &str) -> Option<DateTime<Utc>> {
        let notif = self.notifications.iter().find(|n| n.id == notification_id)?;
        if notif.status != DeliveryStatus::Retrying {
            return None;
        }
        let attempt = notif.attempts.len() as u32;
        let delay = self.retry_policy.initial_delay_seconds as u64
            * self.retry_policy.backoff_multiplier.pow(attempt.saturating_sub(1)) as u64;
        let capped = delay.min(self.retry_policy.max_delay_seconds);
        let last_attempt_time = notif.attempts.last().map(|a| a.timestamp)?;
        Some(last_attempt_time + Duration::seconds(capped as i64))
    }

    /// Count notifications by status.
    pub fn delivery_stats(&self) -> HashMap<DeliveryStatus, usize> {
        let mut counts = HashMap::new();
        for n in &self.notifications {
            *counts.entry(n.status).or_insert(0) += 1;
        }
        counts
    }

    /// Total notifications sent.
    pub fn total_notifications(&self) -> usize {
        self.notifications.len()
    }

    /// Check if recipient is in quiet hours.
    pub fn is_quiet_hours(&self, recipient_id: &str, hour: u8) -> bool {
        if let Some(r) = self.recipients.get(recipient_id) {
            if let (Some(start), Some(end)) = (
                r.preferences.quiet_start_hour,
                r.preferences.quiet_end_hour,
            ) {
                if start <= end {
                    return hour >= start && hour < end;
                } else {
                    // Wraps midnight.
                    return hour >= start || hour < end;
                }
            }
        }
        false
    }
}

impl Default for NotificationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_template(id: &str, channel: Channel) -> NotificationTemplate {
        NotificationTemplate {
            id: id.to_string(),
            name: format!("Template {id}"),
            channel,
            subject_template: Some("Hello {{name}}".to_string()),
            body_template: "Dear {{name}}, your order {{order_id}} is ready.".to_string(),
            default_priority: Priority::Normal,
            created_at: Utc::now(),
        }
    }

    fn make_recipient(id: &str, channels: Vec<Channel>) -> Recipient {
        let configs: Vec<ChannelConfig> = channels
            .iter()
            .map(|c| ChannelConfig {
                channel: *c,
                address: format!("{id}@example.com"),
                enabled: true,
            })
            .collect();
        Recipient {
            id: id.to_string(),
            name: format!("User {id}"),
            channels: configs,
            preferences: NotificationPreferences {
                enabled_channels: channels,
                ..Default::default()
            },
        }
    }

    fn setup_engine() -> NotificationEngine {
        let mut engine = NotificationEngine::new();
        engine
            .register_template(make_template("t1", Channel::Email))
            .unwrap();
        engine
            .register_recipient(make_recipient("r1", vec![Channel::Email, Channel::Push]))
            .unwrap();
        engine
    }

    fn test_vars() -> HashMap<String, String> {
        let mut vars = HashMap::new();
        vars.insert("name".into(), "Alice".into());
        vars.insert("order_id".into(), "ORD-123".into());
        vars
    }

    #[test]
    fn test_register_template() {
        let engine = setup_engine();
        assert!(engine.get_template("t1").is_some());
    }

    #[test]
    fn test_duplicate_template() {
        let mut engine = setup_engine();
        let err = engine
            .register_template(make_template("t1", Channel::Email))
            .unwrap_err();
        assert_eq!(err, NotificationError::DuplicateTemplate("t1".into()));
    }

    #[test]
    fn test_register_recipient() {
        let engine = setup_engine();
        assert!(engine.get_recipient("r1").is_some());
    }

    #[test]
    fn test_duplicate_recipient() {
        let mut engine = setup_engine();
        let err = engine
            .register_recipient(make_recipient("r1", vec![Channel::Email]))
            .unwrap_err();
        assert_eq!(err, NotificationError::DuplicateRecipient("r1".into()));
    }

    #[test]
    fn test_send_notification() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        let notif = engine.get_notification(&id).unwrap();
        assert_eq!(notif.body, "Dear Alice, your order ORD-123 is ready.");
        assert_eq!(notif.subject.as_deref(), Some("Hello Alice"));
        assert_eq!(notif.status, DeliveryStatus::Queued);
    }

    #[test]
    fn test_template_not_found() {
        let mut engine = setup_engine();
        let err = engine.send("nope", "r1", &test_vars(), None).unwrap_err();
        assert_eq!(err, NotificationError::TemplateNotFound("nope".into()));
    }

    #[test]
    fn test_recipient_not_found() {
        let mut engine = setup_engine();
        let err = engine.send("t1", "nope", &test_vars(), None).unwrap_err();
        assert_eq!(err, NotificationError::RecipientNotFound("nope".into()));
    }

    #[test]
    fn test_channel_not_enabled() {
        let mut engine = setup_engine();
        engine
            .register_template(make_template("t2", Channel::Sms))
            .unwrap();
        let err = engine.send("t2", "r1", &test_vars(), None).unwrap_err();
        assert!(matches!(err, NotificationError::ChannelNotConfigured { .. }));
    }

    #[test]
    fn test_mark_delivered() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        engine.mark_delivered(&id).unwrap();
        let notif = engine.get_notification(&id).unwrap();
        assert_eq!(notif.status, DeliveryStatus::Delivered);
        assert!(notif.delivered_at.is_some());
    }

    #[test]
    fn test_mark_failed_with_retry() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        let status = engine.mark_failed(&id, "timeout").unwrap();
        assert_eq!(status, DeliveryStatus::Retrying);
    }

    #[test]
    fn test_max_retries_exhausted() {
        let mut engine = NotificationEngine::new()
            .with_retry_policy(RetryPolicy {
                max_attempts: 1,
                initial_delay_seconds: 1,
                backoff_multiplier: 1,
                max_delay_seconds: 10,
            });
        engine
            .register_template(make_template("t1", Channel::Email))
            .unwrap();
        engine
            .register_recipient(make_recipient("r1", vec![Channel::Email]))
            .unwrap();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        let status = engine.mark_failed(&id, "error").unwrap();
        assert_eq!(status, DeliveryStatus::Failed);
    }

    #[test]
    fn test_mark_bounced() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        engine.mark_bounced(&id).unwrap();
        let notif = engine.get_notification(&id).unwrap();
        assert_eq!(notif.status, DeliveryStatus::Bounced);
    }

    #[test]
    fn test_batch_notifications() {
        let mut engine = setup_engine();
        engine
            .register_recipient(make_recipient("r2", vec![Channel::Email]))
            .unwrap();
        let batch = BatchRequest {
            template_id: "t1".into(),
            recipient_ids: vec!["r1".into(), "r2".into()],
            variables: test_vars(),
            category: None,
            priority: None,
        };
        let ids = engine.send_batch(&batch).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_empty_batch() {
        let mut engine = setup_engine();
        let batch = BatchRequest {
            template_id: "t1".into(),
            recipient_ids: vec![],
            variables: HashMap::new(),
            category: None,
            priority: None,
        };
        let err = engine.send_batch(&batch).unwrap_err();
        assert_eq!(err, NotificationError::EmptyBatch);
    }

    #[test]
    fn test_notifications_for_recipient() {
        let mut engine = setup_engine();
        engine.send("t1", "r1", &test_vars(), None).unwrap();
        engine.send("t1", "r1", &test_vars(), None).unwrap();
        let notifs = engine.notifications_for_recipient("r1");
        assert_eq!(notifs.len(), 2);
    }

    #[test]
    fn test_notifications_by_status() {
        let mut engine = setup_engine();
        engine.send("t1", "r1", &test_vars(), None).unwrap();
        let queued = engine.notifications_by_status(DeliveryStatus::Queued);
        assert_eq!(queued.len(), 1);
    }

    #[test]
    fn test_delivery_stats() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        engine.mark_delivered(&id).unwrap();
        engine.send("t1", "r1", &test_vars(), None).unwrap();
        let stats = engine.delivery_stats();
        assert_eq!(stats.get(&DeliveryStatus::Delivered), Some(&1));
        assert_eq!(stats.get(&DeliveryStatus::Queued), Some(&1));
    }

    #[test]
    fn test_render_template_basic() {
        let mut vars = HashMap::new();
        vars.insert("name".into(), "Bob".into());
        let result = render_template("Hello {{name}}!", &vars).unwrap();
        assert_eq!(result, "Hello Bob!");
    }

    #[test]
    fn test_render_template_unresolved() {
        let vars = HashMap::new();
        let err = render_template("Hello {{name}}!", &vars).unwrap_err();
        assert!(matches!(err, NotificationError::RenderError(_)));
    }

    #[test]
    fn test_update_preferences() {
        let mut engine = setup_engine();
        let prefs = NotificationPreferences {
            enabled_channels: vec![Channel::Push],
            quiet_start_hour: Some(22),
            quiet_end_hour: Some(7),
            opted_out_categories: vec!["marketing".into()],
        };
        engine.update_preferences("r1", prefs).unwrap();
        let r = engine.get_recipient("r1").unwrap();
        assert_eq!(r.preferences.enabled_channels, vec![Channel::Push]);
    }

    #[test]
    fn test_opted_out_category() {
        let mut engine = setup_engine();
        let prefs = NotificationPreferences {
            enabled_channels: vec![Channel::Email],
            quiet_start_hour: None,
            quiet_end_hour: None,
            opted_out_categories: vec!["promo".into()],
        };
        engine.update_preferences("r1", prefs).unwrap();
        let err = engine
            .send("t1", "r1", &test_vars(), Some("promo".into()))
            .unwrap_err();
        assert!(matches!(err, NotificationError::ChannelNotConfigured { .. }));
    }

    #[test]
    fn test_quiet_hours_normal() {
        let mut engine = setup_engine();
        let prefs = NotificationPreferences {
            enabled_channels: vec![Channel::Email],
            quiet_start_hour: Some(22),
            quiet_end_hour: Some(7),
            opted_out_categories: vec![],
        };
        engine.update_preferences("r1", prefs).unwrap();
        assert!(engine.is_quiet_hours("r1", 23));
        assert!(engine.is_quiet_hours("r1", 3));
        assert!(!engine.is_quiet_hours("r1", 12));
    }

    #[test]
    fn test_quiet_hours_daytime() {
        let mut engine = setup_engine();
        let prefs = NotificationPreferences {
            enabled_channels: vec![Channel::Email],
            quiet_start_hour: Some(9),
            quiet_end_hour: Some(17),
            opted_out_categories: vec![],
        };
        engine.update_preferences("r1", prefs).unwrap();
        assert!(engine.is_quiet_hours("r1", 12));
        assert!(!engine.is_quiet_hours("r1", 20));
    }

    #[test]
    fn test_remove_template() {
        let mut engine = setup_engine();
        engine.remove_template("t1").unwrap();
        assert!(engine.get_template("t1").is_none());
    }

    #[test]
    fn test_total_notifications() {
        let mut engine = setup_engine();
        assert_eq!(engine.total_notifications(), 0);
        engine.send("t1", "r1", &test_vars(), None).unwrap();
        assert_eq!(engine.total_notifications(), 1);
    }

    #[test]
    fn test_next_retry_time() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        engine.mark_failed(&id, "err").unwrap();
        let next = engine.next_retry_time(&id);
        assert!(next.is_some());
    }

    #[test]
    fn test_pending_retries() {
        let mut engine = setup_engine();
        let id = engine.send("t1", "r1", &test_vars(), None).unwrap();
        engine.mark_failed(&id, "err").unwrap();
        let retries = engine.pending_retries();
        assert_eq!(retries.len(), 1);
    }
}
