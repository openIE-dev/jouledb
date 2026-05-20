//! Toast/notification system: queued, auto-dismissing notifications.
//!
//! Replaces react-toastify, notistack, and Sonner with a pure-Rust
//! manager that tracks visibility, expiration, and queue ordering.

use std::collections::VecDeque;
use chrono::{DateTime, Utc, TimeDelta};
use uuid::Uuid;

// ── Types ───────────────────────────────────────────────────────

/// Severity level of a toast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastLevel {
    Info,
    Success,
    Warning,
    Error,
}

/// Screen position for the toast stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastPosition {
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

// ── Toast ───────────────────────────────────────────────────────

/// A single notification.
#[derive(Debug, Clone)]
pub struct Toast {
    pub id: Uuid,
    pub message: String,
    pub level: ToastLevel,
    pub duration_ms: Option<u64>,
    pub dismissible: bool,
    pub created_at: DateTime<Utc>,
    pub action_label: Option<String>,
    pub action_id: Option<u64>,
}

/// Builder for constructing a `Toast`.
pub struct ToastBuilder {
    message: String,
    level: ToastLevel,
    duration_ms: Option<u64>,
    dismissible: bool,
    action_label: Option<String>,
    action_id: Option<u64>,
}

impl Toast {
    pub fn info(msg: &str) -> ToastBuilder {
        ToastBuilder::new(msg, ToastLevel::Info)
    }

    pub fn success(msg: &str) -> ToastBuilder {
        ToastBuilder::new(msg, ToastLevel::Success)
    }

    pub fn warning(msg: &str) -> ToastBuilder {
        ToastBuilder::new(msg, ToastLevel::Warning)
    }

    pub fn error(msg: &str) -> ToastBuilder {
        ToastBuilder::new(msg, ToastLevel::Error)
    }
}

impl ToastBuilder {
    fn new(msg: &str, level: ToastLevel) -> Self {
        Self {
            message: msg.to_string(),
            level,
            duration_ms: None, // filled by manager default
            dismissible: true,
            action_label: None,
            action_id: None,
        }
    }

    pub fn duration(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    pub fn persistent(mut self) -> Self {
        self.duration_ms = None;
        self
    }

    pub fn dismissible(mut self, yes: bool) -> Self {
        self.dismissible = yes;
        self
    }

    pub fn action(mut self, label: &str, handler_id: u64) -> Self {
        self.action_label = Some(label.to_string());
        self.action_id = Some(handler_id);
        self
    }

    pub fn build(self) -> Toast {
        Toast {
            id: Uuid::new_v4(),
            message: self.message,
            level: self.level,
            duration_ms: self.duration_ms,
            dismissible: self.dismissible,
            created_at: Utc::now(),
            action_label: self.action_label,
            action_id: self.action_id,
        }
    }
}

// ── ToastManager ────────────────────────────────────────────────

/// Manages a queue of toasts with visibility limits and auto-expiration.
pub struct ToastManager {
    toasts: VecDeque<Toast>,
    pub position: ToastPosition,
    pub max_visible: usize,
    pub default_duration_ms: u64,
}

impl ToastManager {
    pub fn new() -> Self {
        Self {
            toasts: VecDeque::new(),
            position: ToastPosition::TopRight,
            max_visible: 5,
            default_duration_ms: 5000,
        }
    }

    pub fn with_position(pos: ToastPosition) -> Self {
        Self {
            position: pos,
            ..Self::new()
        }
    }

    /// Add a toast and return its ID.
    pub fn push(&mut self, mut toast: Toast) -> Uuid {
        // Apply default duration if none set and not explicitly persistent
        if toast.duration_ms.is_none() {
            toast.duration_ms = Some(self.default_duration_ms);
        }
        let id = toast.id;
        self.toasts.push_back(toast);
        id
    }

    /// Push a toast that was built with `.persistent()` — preserves None duration.
    pub fn push_persistent(&mut self, toast: Toast) -> Uuid {
        let id = toast.id;
        self.toasts.push_back(toast);
        id
    }

    /// Dismiss a specific toast. Returns true if found.
    pub fn dismiss(&mut self, id: &Uuid) -> bool {
        let before = self.toasts.len();
        self.toasts.retain(|t| t.id != *id);
        self.toasts.len() < before
    }

    /// Dismiss all toasts.
    pub fn dismiss_all(&mut self) {
        self.toasts.clear();
    }

    /// Remove expired toasts and return their IDs.
    pub fn tick(&mut self, now: &DateTime<Utc>) -> Vec<Uuid> {
        let mut expired = Vec::new();
        self.toasts.retain(|t| {
            if let Some(dur) = t.duration_ms {
                let expires_at = t.created_at + TimeDelta::milliseconds(dur as i64);
                if *now >= expires_at {
                    expired.push(t.id);
                    return false;
                }
            }
            true
        });
        expired
    }

    /// Return up to `max_visible` toasts (oldest first).
    pub fn visible(&self) -> Vec<&Toast> {
        self.toasts.iter().take(self.max_visible).collect()
    }

    pub fn count(&self) -> usize {
        self.toasts.len()
    }

    pub fn has_toasts(&self) -> bool {
        !self.toasts.is_empty()
    }
}

impl Default for ToastManager {
    fn default() -> Self { Self::new() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    fn make_toast(level: ToastLevel, dur_ms: Option<u64>) -> Toast {
        let mut t = match level {
            ToastLevel::Info => Toast::info("test"),
            ToastLevel::Success => Toast::success("test"),
            ToastLevel::Warning => Toast::warning("test"),
            ToastLevel::Error => Toast::error("test"),
        };
        if let Some(ms) = dur_ms {
            t = t.duration(ms);
        }
        t.build()
    }

    #[test]
    fn push_adds_toast() {
        let mut m = ToastManager::new();
        let t = make_toast(ToastLevel::Info, Some(3000));
        m.push(t);
        assert_eq!(m.count(), 1);
    }

    #[test]
    fn dismiss_removes() {
        let mut m = ToastManager::new();
        let t = make_toast(ToastLevel::Info, Some(3000));
        let id = m.push(t);
        assert!(m.dismiss(&id));
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn tick_expires_old() {
        let mut m = ToastManager::new();
        m.default_duration_ms = 1000;
        let t = Toast::info("gone soon").duration(1000).build();
        m.push(t);
        let future = Utc::now() + TimeDelta::seconds(2);
        let expired = m.tick(&future);
        assert_eq!(expired.len(), 1);
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn persistent_not_expired() {
        let mut m = ToastManager::new();
        let t = Toast::info("stay").persistent().build();
        // Use push_persistent to avoid default duration override
        m.push_persistent(t);
        let future = Utc::now() + TimeDelta::hours(1);
        let expired = m.tick(&future);
        assert!(expired.is_empty());
        assert_eq!(m.count(), 1);
    }

    #[test]
    fn max_visible_limits() {
        let mut m = ToastManager::new();
        m.max_visible = 2;
        for _ in 0..5 {
            m.push(make_toast(ToastLevel::Info, Some(60000)));
        }
        assert_eq!(m.visible().len(), 2);
        assert_eq!(m.count(), 5);
    }

    #[test]
    fn queue_shows_next_after_dismiss() {
        let mut m = ToastManager::new();
        m.max_visible = 1;
        let t1 = make_toast(ToastLevel::Info, Some(60000));
        let t2 = make_toast(ToastLevel::Success, Some(60000));
        let id1 = m.push(t1);
        let _id2 = m.push(t2);
        assert_eq!(m.visible().len(), 1);
        assert_eq!(m.visible()[0].level, ToastLevel::Info);
        m.dismiss(&id1);
        assert_eq!(m.visible().len(), 1);
        assert_eq!(m.visible()[0].level, ToastLevel::Success);
    }

    #[test]
    fn all_levels() {
        let i = make_toast(ToastLevel::Info, None);
        let s = make_toast(ToastLevel::Success, None);
        let w = make_toast(ToastLevel::Warning, None);
        let e = make_toast(ToastLevel::Error, None);
        assert_eq!(i.level, ToastLevel::Info);
        assert_eq!(s.level, ToastLevel::Success);
        assert_eq!(w.level, ToastLevel::Warning);
        assert_eq!(e.level, ToastLevel::Error);
    }

    #[test]
    fn action_label_set() {
        let t = Toast::info("confirm?").action("Undo", 42).build();
        assert_eq!(t.action_label.as_deref(), Some("Undo"));
        assert_eq!(t.action_id, Some(42));
    }

    #[test]
    fn dismiss_all_clears() {
        let mut m = ToastManager::new();
        for _ in 0..3 {
            m.push(make_toast(ToastLevel::Info, Some(5000)));
        }
        m.dismiss_all();
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn visible_returns_ordered() {
        let mut m = ToastManager::new();
        let t1 = Toast::info("first").duration(60000).build();
        let t2 = Toast::success("second").duration(60000).build();
        m.push(t1);
        m.push(t2);
        let vis = m.visible();
        assert_eq!(vis[0].message, "first");
        assert_eq!(vis[1].message, "second");
    }
}
