//! Web Notifications API: permission management, notification lifecycle.

use chrono::{DateTime, Utc};
use uuid::Uuid;

// ── Types ───────────────────────────────────────────────────────

/// Notification permission state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationPermission {
    Default,
    Granted,
    Denied,
}

/// Action button within a notification.
#[derive(Debug, Clone, PartialEq)]
pub struct NotificationAction {
    pub action: String,
    pub title: String,
    pub icon: Option<String>,
}

/// Options for creating a notification.
#[derive(Debug, Clone, Default)]
pub struct NotificationOptions {
    pub body: Option<String>,
    pub icon: Option<String>,
    pub badge: Option<String>,
    pub tag: Option<String>,
    pub data: Option<serde_json::Value>,
    pub require_interaction: bool,
    pub silent: bool,
    pub vibrate: Vec<u32>,
    pub timestamp: Option<DateTime<Utc>>,
    pub actions: Vec<NotificationAction>,
}

/// A Web Notification instance.
#[derive(Debug, Clone)]
pub struct WebNotification {
    pub id: Uuid,
    pub title: String,
    pub options: NotificationOptions,
    pub created_at: DateTime<Utc>,
    pub clicked: bool,
    pub closed: bool,
}

// ── Manager ─────────────────────────────────────────────────────

/// Manages notification permission and active notifications.
#[derive(Debug)]
pub struct NotificationManager {
    permission: NotificationPermission,
    notifications: Vec<WebNotification>,
    max_visible: usize,
}

impl NotificationManager {
    pub fn new() -> Self {
        Self {
            permission: NotificationPermission::Default,
            notifications: Vec::new(),
            max_visible: 5,
        }
    }

    /// Simulate the permission prompt result.
    pub fn request_permission(&mut self, granted: bool) {
        self.permission = if granted {
            NotificationPermission::Granted
        } else {
            NotificationPermission::Denied
        };
    }

    /// Show a notification. Returns `None` if permission is not `Granted`.
    pub fn show(&mut self, title: impl Into<String>, options: NotificationOptions) -> Option<Uuid> {
        if self.permission != NotificationPermission::Granted {
            return None;
        }

        let id = Uuid::new_v4();
        let notif = WebNotification {
            id,
            title: title.into(),
            options,
            created_at: Utc::now(),
            clicked: false,
            closed: false,
        };

        self.notifications.push(notif);

        // Enforce max_visible by closing oldest active notifications.
        let active: Vec<usize> = self.notifications.iter().enumerate()
            .filter(|(_, n)| !n.closed)
            .map(|(i, _)| i)
            .collect();
        if active.len() > self.max_visible {
            let to_close = active.len() - self.max_visible;
            for &idx in active.iter().take(to_close) {
                self.notifications[idx].closed = true;
            }
        }

        Some(id)
    }

    /// Close a notification by id.
    pub fn close(&mut self, id: Uuid) {
        if let Some(n) = self.notifications.iter_mut().find(|n| n.id == id) {
            n.closed = true;
        }
    }

    /// Close all notifications.
    pub fn close_all(&mut self) {
        for n in &mut self.notifications {
            n.closed = true;
        }
    }

    /// Mark a notification as clicked.
    pub fn click(&mut self, id: Uuid) {
        if let Some(n) = self.notifications.iter_mut().find(|n| n.id == id) {
            n.clicked = true;
        }
    }

    /// Number of active (not closed) notifications.
    pub fn active_count(&self) -> usize {
        self.notifications.iter().filter(|n| !n.closed).count()
    }

    /// Find notifications by tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&WebNotification> {
        self.notifications
            .iter()
            .filter(|n| n.options.tag.as_deref() == Some(tag))
            .collect()
    }
}

impl Default for NotificationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> NotificationOptions {
        NotificationOptions::default()
    }

    fn tagged(tag: &str) -> NotificationOptions {
        NotificationOptions {
            tag: Some(tag.into()),
            ..Default::default()
        }
    }

    #[test]
    fn permission_check() {
        let mgr = NotificationManager::new();
        assert_eq!(mgr.permission, NotificationPermission::Default);
    }

    #[test]
    fn show_when_granted() {
        let mut mgr = NotificationManager::new();
        mgr.request_permission(true);
        let id = mgr.show("Hello", opts());
        assert!(id.is_some());
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn blocked_when_denied() {
        let mut mgr = NotificationManager::new();
        mgr.request_permission(false);
        assert!(mgr.show("Hello", opts()).is_none());
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn blocked_when_default() {
        let mut mgr = NotificationManager::new();
        assert!(mgr.show("Hello", opts()).is_none());
    }

    #[test]
    fn close_notification() {
        let mut mgr = NotificationManager::new();
        mgr.request_permission(true);
        let id = mgr.show("Hello", opts()).unwrap();
        assert_eq!(mgr.active_count(), 1);
        mgr.close(id);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn click_notification() {
        let mut mgr = NotificationManager::new();
        mgr.request_permission(true);
        let id = mgr.show("Hello", opts()).unwrap();
        mgr.click(id);
        let n = mgr.notifications.iter().find(|n| n.id == id).unwrap();
        assert!(n.clicked);
    }

    #[test]
    fn tag_filtering() {
        let mut mgr = NotificationManager::new();
        mgr.request_permission(true);
        mgr.show("A", tagged("chat"));
        mgr.show("B", tagged("email"));
        mgr.show("C", tagged("chat"));
        let chat = mgr.by_tag("chat");
        assert_eq!(chat.len(), 2);
        assert_eq!(mgr.by_tag("email").len(), 1);
        assert_eq!(mgr.by_tag("sms").len(), 0);
    }

    #[test]
    fn max_visible_enforced() {
        let mut mgr = NotificationManager::new();
        mgr.max_visible = 2;
        mgr.request_permission(true);
        mgr.show("1", opts());
        mgr.show("2", opts());
        mgr.show("3", opts()); // should auto-close the oldest
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn close_all() {
        let mut mgr = NotificationManager::new();
        mgr.request_permission(true);
        mgr.show("A", opts());
        mgr.show("B", opts());
        mgr.close_all();
        assert_eq!(mgr.active_count(), 0);
    }
}
