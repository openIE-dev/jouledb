//! Online status/presence system — presence tracking, auto-away, subscriptions.
//!
//! Replaces Discord.js presence / Firebase Realtime presence with pure Rust.
//! Status enums, per-user presence, heartbeat-driven auto-away, subscription
//! model (watch specific users), bulk queries, and statistics.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresenceError {
    UserNotFound(String),
    AlreadySubscribed(String),
    NotSubscribed(String),
}

impl fmt::Display for PresenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserNotFound(u) => write!(f, "user not found: {u}"),
            Self::AlreadySubscribed(u) => write!(f, "already subscribed: {u}"),
            Self::NotSubscribed(u) => write!(f, "not subscribed: {u}"),
        }
    }
}

impl std::error::Error for PresenceError {}

// ── PresenceStatus ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresenceStatus {
    Online,
    Away,
    Busy,
    Invisible,
    Offline,
}

impl PresenceStatus {
    pub fn is_visible(&self) -> bool {
        !matches!(self, Self::Invisible | Self::Offline)
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::Online)
    }
}

impl fmt::Display for PresenceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Online => write!(f, "online"),
            Self::Away => write!(f, "away"),
            Self::Busy => write!(f, "busy"),
            Self::Invisible => write!(f, "invisible"),
            Self::Offline => write!(f, "offline"),
        }
    }
}

// ── StatusChange ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusChange {
    pub user_id: String,
    pub old_status: PresenceStatus,
    pub new_status: PresenceStatus,
    pub timestamp: u64,
}

impl fmt::Display for StatusChange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} -> {} at {}", self.user_id, self.old_status, self.new_status, self.timestamp)
    }
}

// ── UserPresence ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct UserPresence {
    pub user_id: String,
    pub status: PresenceStatus,
    pub last_activity: u64,
    pub custom_message: Option<String>,
    pub connected_at: u64,
}

impl UserPresence {
    pub fn new(user_id: &str, connected_at: u64) -> Self {
        Self {
            user_id: user_id.to_string(),
            status: PresenceStatus::Online,
            last_activity: connected_at,
            custom_message: None,
            connected_at,
        }
    }

    pub fn with_message(mut self, msg: &str) -> Self {
        self.custom_message = Some(msg.to_string());
        self
    }

    pub fn idle_duration(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_activity)
    }

    pub fn session_duration(&self, now: u64) -> u64 {
        now.saturating_sub(self.connected_at)
    }

    /// Returns the apparent status (invisible shows as offline to others).
    pub fn apparent_status(&self) -> PresenceStatus {
        if self.status == PresenceStatus::Invisible {
            PresenceStatus::Offline
        } else {
            self.status
        }
    }
}

impl fmt::Display for UserPresence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.custom_message {
            Some(msg) => write!(f, "{} ({}) - {}", self.user_id, self.status, msg),
            None => write!(f, "{} ({})", self.user_id, self.status),
        }
    }
}

// ── PresenceStats ───────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct PresenceStats {
    pub online: usize,
    pub away: usize,
    pub busy: usize,
    pub invisible: usize,
    pub offline: usize,
}

impl PresenceStats {
    pub fn total(&self) -> usize {
        self.online + self.away + self.busy + self.invisible + self.offline
    }

    pub fn visible(&self) -> usize {
        self.online + self.away + self.busy
    }
}

// ── PresenceManager ─────────────────────────────────────────────

#[derive(Debug)]
pub struct PresenceManager {
    users: HashMap<String, UserPresence>,
    subscriptions: HashMap<String, HashSet<String>>,
    auto_away_timeout: u64,
    notifications: Vec<StatusChange>,
}

impl PresenceManager {
    pub fn new(auto_away_timeout: u64) -> Self {
        Self {
            users: HashMap::new(),
            subscriptions: HashMap::new(),
            auto_away_timeout,
            notifications: Vec::new(),
        }
    }

    pub fn connect(&mut self, user_id: &str, timestamp: u64) -> &UserPresence {
        self.users.entry(user_id.to_string()).or_insert_with(|| UserPresence::new(user_id, timestamp));
        self.users.get(user_id).unwrap()
    }

    pub fn disconnect(&mut self, user_id: &str, timestamp: u64) -> Option<StatusChange> {
        if let Some(presence) = self.users.get_mut(user_id) {
            let old = presence.status;
            if old != PresenceStatus::Offline {
                presence.status = PresenceStatus::Offline;
                let change = StatusChange {
                    user_id: user_id.to_string(),
                    old_status: old,
                    new_status: PresenceStatus::Offline,
                    timestamp,
                };
                self.notifications.push(change.clone());
                return Some(change);
            }
        }
        None
    }

    pub fn set_status(&mut self, user_id: &str, status: PresenceStatus, timestamp: u64) -> Result<StatusChange, PresenceError> {
        let presence = self.users.get_mut(user_id).ok_or_else(|| PresenceError::UserNotFound(user_id.to_string()))?;
        let old = presence.status;
        presence.status = status;
        presence.last_activity = timestamp;
        let change = StatusChange {
            user_id: user_id.to_string(),
            old_status: old,
            new_status: status,
            timestamp,
        };
        self.notifications.push(change.clone());
        Ok(change)
    }

    pub fn set_custom_message(&mut self, user_id: &str, msg: &str) -> Result<(), PresenceError> {
        let presence = self.users.get_mut(user_id).ok_or_else(|| PresenceError::UserNotFound(user_id.to_string()))?;
        presence.custom_message = Some(msg.to_string());
        Ok(())
    }

    pub fn heartbeat(&mut self, user_id: &str, timestamp: u64) -> Result<Option<StatusChange>, PresenceError> {
        let presence = self.users.get_mut(user_id).ok_or_else(|| PresenceError::UserNotFound(user_id.to_string()))?;
        presence.last_activity = timestamp;
        if presence.status == PresenceStatus::Away {
            let old = presence.status;
            presence.status = PresenceStatus::Online;
            let change = StatusChange {
                user_id: user_id.to_string(),
                old_status: old,
                new_status: PresenceStatus::Online,
                timestamp,
            };
            self.notifications.push(change.clone());
            return Ok(Some(change));
        }
        Ok(None)
    }

    /// Process auto-away for idle users. Returns list of status changes.
    pub fn tick_auto_away(&mut self, now: u64) -> Vec<StatusChange> {
        let mut changes = Vec::new();
        for presence in self.users.values_mut() {
            if presence.status == PresenceStatus::Online && presence.idle_duration(now) >= self.auto_away_timeout {
                let change = StatusChange {
                    user_id: presence.user_id.clone(),
                    old_status: PresenceStatus::Online,
                    new_status: PresenceStatus::Away,
                    timestamp: now,
                };
                presence.status = PresenceStatus::Away;
                changes.push(change);
            }
        }
        self.notifications.extend(changes.clone());
        changes
    }

    pub fn get_presence(&self, user_id: &str) -> Option<&UserPresence> {
        self.users.get(user_id)
    }

    pub fn get_apparent_status(&self, user_id: &str) -> Option<PresenceStatus> {
        self.users.get(user_id).map(|p| p.apparent_status())
    }

    pub fn subscribe(&mut self, watcher: &str, target: &str) -> Result<(), PresenceError> {
        let subs = self.subscriptions.entry(watcher.to_string()).or_default();
        if !subs.insert(target.to_string()) {
            return Err(PresenceError::AlreadySubscribed(target.to_string()));
        }
        Ok(())
    }

    pub fn unsubscribe(&mut self, watcher: &str, target: &str) -> Result<(), PresenceError> {
        let subs = self.subscriptions.get_mut(watcher).ok_or_else(|| PresenceError::NotSubscribed(target.to_string()))?;
        if !subs.remove(target) {
            return Err(PresenceError::NotSubscribed(target.to_string()));
        }
        Ok(())
    }

    pub fn watched_statuses(&self, watcher: &str) -> Vec<(&str, PresenceStatus)> {
        let Some(subs) = self.subscriptions.get(watcher) else {
            return Vec::new();
        };
        subs.iter()
            .filter_map(|uid| self.users.get(uid).map(|p| (uid.as_str(), p.apparent_status())))
            .collect()
    }

    pub fn bulk_presence<'a>(&self, user_ids: &[&'a str]) -> Vec<(&'a str, PresenceStatus)> {
        user_ids.iter()
            .filter_map(|uid| self.users.get(*uid).map(|p| (*uid, p.apparent_status())))
            .collect()
    }

    pub fn stats(&self) -> PresenceStats {
        let mut s = PresenceStats::default();
        for p in self.users.values() {
            match p.status {
                PresenceStatus::Online => s.online += 1,
                PresenceStatus::Away => s.away += 1,
                PresenceStatus::Busy => s.busy += 1,
                PresenceStatus::Invisible => s.invisible += 1,
                PresenceStatus::Offline => s.offline += 1,
            }
        }
        s
    }

    pub fn drain_notifications(&mut self) -> Vec<StatusChange> {
        std::mem::take(&mut self.notifications)
    }

    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    pub fn online_users(&self) -> Vec<&str> {
        self.users.values()
            .filter(|p| p.status.is_visible())
            .map(|p| p.user_id.as_str())
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> PresenceManager {
        PresenceManager::new(300) // 5 min auto-away
    }

    #[test]
    fn test_connect() {
        let mut m = mgr();
        let p = m.connect("alice", 1000);
        assert_eq!(p.status, PresenceStatus::Online);
        assert_eq!(m.user_count(), 1);
    }

    #[test]
    fn test_disconnect() {
        let mut m = mgr();
        m.connect("alice", 1000);
        let change = m.disconnect("alice", 2000).unwrap();
        assert_eq!(change.new_status, PresenceStatus::Offline);
    }

    #[test]
    fn test_set_status() {
        let mut m = mgr();
        m.connect("alice", 1000);
        let change = m.set_status("alice", PresenceStatus::Busy, 1001).unwrap();
        assert_eq!(change.old_status, PresenceStatus::Online);
        assert_eq!(change.new_status, PresenceStatus::Busy);
    }

    #[test]
    fn test_status_not_found() {
        let mut m = mgr();
        assert!(m.set_status("ghost", PresenceStatus::Online, 1).is_err());
    }

    #[test]
    fn test_invisible_appears_offline() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.set_status("alice", PresenceStatus::Invisible, 1001).unwrap();
        assert_eq!(m.get_apparent_status("alice"), Some(PresenceStatus::Offline));
    }

    #[test]
    fn test_custom_message() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.set_custom_message("alice", "Playing Rust").unwrap();
        assert_eq!(m.get_presence("alice").unwrap().custom_message.as_deref(), Some("Playing Rust"));
    }

    #[test]
    fn test_auto_away() {
        let mut m = mgr();
        m.connect("alice", 1000);
        let changes = m.tick_auto_away(1500); // 500s > 300s timeout
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].new_status, PresenceStatus::Away);
    }

    #[test]
    fn test_no_auto_away_if_active() {
        let mut m = mgr();
        m.connect("alice", 1000);
        let changes = m.tick_auto_away(1100); // 100s < 300s
        assert!(changes.is_empty());
    }

    #[test]
    fn test_heartbeat_revives_away() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.tick_auto_away(1500);
        let change = m.heartbeat("alice", 1600).unwrap();
        assert!(change.is_some());
        assert_eq!(m.get_presence("alice").unwrap().status, PresenceStatus::Online);
    }

    #[test]
    fn test_subscribe_and_watch() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.connect("bob", 1000);
        m.subscribe("alice", "bob").unwrap();
        let watched = m.watched_statuses("alice");
        assert_eq!(watched.len(), 1);
        assert_eq!(watched[0].1, PresenceStatus::Online);
    }

    #[test]
    fn test_subscribe_duplicate() {
        let mut m = mgr();
        m.connect("bob", 1000);
        m.subscribe("alice", "bob").unwrap();
        assert!(m.subscribe("alice", "bob").is_err());
    }

    #[test]
    fn test_unsubscribe() {
        let mut m = mgr();
        m.connect("bob", 1000);
        m.subscribe("alice", "bob").unwrap();
        m.unsubscribe("alice", "bob").unwrap();
        assert!(m.watched_statuses("alice").is_empty());
    }

    #[test]
    fn test_bulk_presence() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.connect("bob", 1000);
        m.set_status("bob", PresenceStatus::Away, 1001).unwrap();
        let bulk = m.bulk_presence(&["alice", "bob", "ghost"]);
        assert_eq!(bulk.len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.connect("bob", 1000);
        m.set_status("bob", PresenceStatus::Busy, 1001).unwrap();
        let s = m.stats();
        assert_eq!(s.online, 1);
        assert_eq!(s.busy, 1);
        assert_eq!(s.total(), 2);
    }

    #[test]
    fn test_online_users() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.connect("bob", 1000);
        m.set_status("bob", PresenceStatus::Invisible, 1001).unwrap();
        let online = m.online_users();
        assert_eq!(online.len(), 1);
        assert!(online.contains(&"alice"));
    }

    #[test]
    fn test_drain_notifications() {
        let mut m = mgr();
        m.connect("alice", 1000);
        m.set_status("alice", PresenceStatus::Away, 1001).unwrap();
        let n = m.drain_notifications();
        assert_eq!(n.len(), 1);
        assert!(m.drain_notifications().is_empty());
    }

    #[test]
    fn test_session_duration() {
        let mut m = mgr();
        m.connect("alice", 1000);
        assert_eq!(m.get_presence("alice").unwrap().session_duration(2000), 1000);
    }

    #[test]
    fn test_display_status_change() {
        let c = StatusChange {
            user_id: "alice".into(),
            old_status: PresenceStatus::Online,
            new_status: PresenceStatus::Away,
            timestamp: 100,
        };
        let s = format!("{c}");
        assert!(s.contains("online"));
        assert!(s.contains("away"));
    }

    #[test]
    fn test_display_user_presence() {
        let p = UserPresence::new("alice", 100).with_message("Playing");
        let s = format!("{p}");
        assert!(s.contains("Playing"));
    }

    #[test]
    fn test_presence_status_available() {
        assert!(PresenceStatus::Online.is_available());
        assert!(!PresenceStatus::Busy.is_available());
        assert!(!PresenceStatus::Away.is_available());
    }
}
