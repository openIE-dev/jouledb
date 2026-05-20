//! Notification management — center, priorities, grouping, preferences.
//!
//! Replaces react-notifications, notistack notification center features with
//! a pure-Rust notification store and filtering engine.

use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

// ── Priority ────────────────────────────────────────────────────

/// Notification priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low,
    Normal,
    High,
    Urgent,
}

// ── Notification ────────────────────────────────────────────────

/// A notification entry.
#[derive(Debug, Clone)]
pub struct Notification {
    pub id: Uuid,
    pub notification_type: String,
    pub title: String,
    pub body: String,
    pub timestamp: DateTime<Utc>,
    pub read: bool,
    pub priority: Priority,
    pub action_url: Option<String>,
    pub group_id: Option<String>,
}

impl Notification {
    /// Create a new notification.
    pub fn new(notification_type: &str, title: &str, body: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            notification_type: notification_type.to_string(),
            title: title.to_string(),
            body: body.to_string(),
            timestamp: Utc::now(),
            read: false,
            priority: Priority::Normal,
            action_url: None,
            group_id: None,
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_action_url(mut self, url: &str) -> Self {
        self.action_url = Some(url.to_string());
        self
    }

    pub fn with_group(mut self, group_id: &str) -> Self {
        self.group_id = Some(group_id.to_string());
        self
    }

    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }
}

// ── NotificationPreferences ─────────────────────────────────────

/// Per-type notification preferences.
#[derive(Debug, Clone, Default)]
pub struct NotificationPreferences {
    /// Set of disabled notification types.
    disabled_types: HashSet<String>,
}

impl NotificationPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable notifications of a given type.
    pub fn disable_type(&mut self, notification_type: &str) {
        self.disabled_types.insert(notification_type.to_string());
    }

    /// Enable notifications of a given type.
    pub fn enable_type(&mut self, notification_type: &str) {
        self.disabled_types.remove(notification_type);
    }

    /// Check if a notification type is enabled.
    pub fn is_enabled(&self, notification_type: &str) -> bool {
        !self.disabled_types.contains(notification_type)
    }
}

// ── NotificationCenter ──────────────────────────────────────────

/// Central notification store.
#[derive(Debug, Clone, Default)]
pub struct NotificationCenter {
    notifications: Vec<Notification>,
    preferences: NotificationPreferences,
}

impl NotificationCenter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set preferences.
    pub fn set_preferences(&mut self, prefs: NotificationPreferences) {
        self.preferences = prefs;
    }

    /// Get preferences.
    pub fn preferences(&self) -> &NotificationPreferences {
        &self.preferences
    }

    /// Mutable preferences.
    pub fn preferences_mut(&mut self) -> &mut NotificationPreferences {
        &mut self.preferences
    }

    /// Add a notification (respects preferences).
    pub fn add(&mut self, notification: Notification) -> bool {
        if !self.preferences.is_enabled(&notification.notification_type) {
            return false;
        }
        self.notifications.push(notification);
        true
    }

    /// Mark a notification as read by id.
    pub fn mark_read(&mut self, id: Uuid) -> bool {
        if let Some(n) = self.notifications.iter_mut().find(|n| n.id == id) {
            n.read = true;
            true
        } else {
            false
        }
    }

    /// Mark all notifications as read.
    pub fn mark_all_read(&mut self) {
        for n in &mut self.notifications {
            n.read = true;
        }
    }

    /// Dismiss (remove) a notification by id.
    pub fn dismiss(&mut self, id: Uuid) -> bool {
        let len_before = self.notifications.len();
        self.notifications.retain(|n| n.id != id);
        self.notifications.len() < len_before
    }

    /// Dismiss all notifications.
    pub fn dismiss_all(&mut self) {
        self.notifications.clear();
    }

    /// Count of unread notifications.
    pub fn unread_count(&self) -> usize {
        self.notifications.iter().filter(|n| !n.read).count()
    }

    /// Badge count (same as unread count).
    pub fn badge_count(&self) -> usize {
        self.unread_count()
    }

    /// Total notification count.
    pub fn total_count(&self) -> usize {
        self.notifications.len()
    }

    /// Get all notifications sorted by timestamp (newest first).
    pub fn all_sorted(&self) -> Vec<&Notification> {
        let mut sorted: Vec<_> = self.notifications.iter().collect();
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        sorted
    }

    /// Filter notifications by type.
    pub fn filter_by_type(&self, notification_type: &str) -> Vec<&Notification> {
        self.notifications
            .iter()
            .filter(|n| n.notification_type == notification_type)
            .collect()
    }

    /// Group notifications by group_id.
    pub fn group_by_group_id(&self) -> HashMap<Option<String>, Vec<&Notification>> {
        let mut groups: HashMap<Option<String>, Vec<&Notification>> = HashMap::new();
        for n in &self.notifications {
            groups
                .entry(n.group_id.clone())
                .or_default()
                .push(n);
        }
        groups
    }

    /// Get a notification by id.
    pub fn get(&self, id: Uuid) -> Option<&Notification> {
        self.notifications.iter().find(|n| n.id == id)
    }

    /// Get unread notifications.
    pub fn unread(&self) -> Vec<&Notification> {
        self.notifications.iter().filter(|n| !n.read).collect()
    }

    /// Get notifications filtered by minimum priority.
    pub fn filter_by_min_priority(&self, min: Priority) -> Vec<&Notification> {
        self.notifications
            .iter()
            .filter(|n| n.priority >= min)
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    #[test]
    fn test_add_and_count() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "Title", "Body"));
        assert_eq!(center.total_count(), 1);
        assert_eq!(center.unread_count(), 1);
    }

    #[test]
    fn test_mark_read() {
        let mut center = NotificationCenter::new();
        let n = Notification::new("info", "Title", "Body");
        let id = n.id;
        center.add(n);
        assert_eq!(center.unread_count(), 1);
        center.mark_read(id);
        assert_eq!(center.unread_count(), 0);
    }

    #[test]
    fn test_mark_all_read() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "A", ""));
        center.add(Notification::new("info", "B", ""));
        center.add(Notification::new("info", "C", ""));
        assert_eq!(center.unread_count(), 3);
        center.mark_all_read();
        assert_eq!(center.unread_count(), 0);
    }

    #[test]
    fn test_dismiss() {
        let mut center = NotificationCenter::new();
        let n = Notification::new("info", "Title", "Body");
        let id = n.id;
        center.add(n);
        assert!(center.dismiss(id));
        assert_eq!(center.total_count(), 0);
    }

    #[test]
    fn test_dismiss_all() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "A", ""));
        center.add(Notification::new("alert", "B", ""));
        center.dismiss_all();
        assert_eq!(center.total_count(), 0);
    }

    #[test]
    fn test_filter_by_type() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "A", ""));
        center.add(Notification::new("alert", "B", ""));
        center.add(Notification::new("info", "C", ""));
        let infos = center.filter_by_type("info");
        assert_eq!(infos.len(), 2);
    }

    #[test]
    fn test_group_by_group_id() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "A", "").with_group("g1"));
        center.add(Notification::new("info", "B", "").with_group("g1"));
        center.add(Notification::new("info", "C", "").with_group("g2"));
        center.add(Notification::new("info", "D", ""));
        let groups = center.group_by_group_id();
        assert_eq!(groups[&Some("g1".to_string())].len(), 2);
        assert_eq!(groups[&Some("g2".to_string())].len(), 1);
        assert_eq!(groups[&None].len(), 1);
    }

    #[test]
    fn test_sort_by_timestamp() {
        let mut center = NotificationCenter::new();
        let now = Utc::now();
        center.add(Notification::new("info", "Old", "").with_timestamp(now - TimeDelta::hours(2)));
        center.add(Notification::new("info", "New", "").with_timestamp(now));
        center.add(Notification::new("info", "Mid", "").with_timestamp(now - TimeDelta::hours(1)));
        let sorted = center.all_sorted();
        assert_eq!(sorted[0].title, "New");
        assert_eq!(sorted[1].title, "Mid");
        assert_eq!(sorted[2].title, "Old");
    }

    #[test]
    fn test_badge_count() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "A", ""));
        center.add(Notification::new("info", "B", ""));
        let id = center.notifications[0].id;
        center.mark_read(id);
        assert_eq!(center.badge_count(), 1);
    }

    #[test]
    fn test_preferences_disable_type() {
        let mut center = NotificationCenter::new();
        center.preferences_mut().disable_type("promo");
        let added = center.add(Notification::new("promo", "Sale!", "50% off"));
        assert!(!added);
        assert_eq!(center.total_count(), 0);
        // Other types still work
        let added2 = center.add(Notification::new("info", "Update", "New version"));
        assert!(added2);
        assert_eq!(center.total_count(), 1);
    }

    #[test]
    fn test_priority_filter() {
        let mut center = NotificationCenter::new();
        center.add(Notification::new("info", "Low", "").with_priority(Priority::Low));
        center.add(Notification::new("info", "High", "").with_priority(Priority::High));
        center.add(Notification::new("info", "Urgent", "").with_priority(Priority::Urgent));
        let high_plus = center.filter_by_min_priority(Priority::High);
        assert_eq!(high_plus.len(), 2);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Urgent);
    }

    #[test]
    fn test_action_url() {
        let n = Notification::new("info", "Click", "")
            .with_action_url("https://example.com/action");
        assert_eq!(n.action_url.as_deref(), Some("https://example.com/action"));
    }
}
