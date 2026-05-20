//! Presence system — online/offline/away/busy status, heartbeat tracking,
//! presence channels, user lists, last-seen timestamps, and presence diffs.
//!
//! Pure-Rust presence tracking that runs without real timers or network I/O.
//! Callers drive time forward by passing timestamps to heartbeat/tick methods.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

// ── Status ─────────────────────────────────────────────────────────

/// User presence status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresenceStatus {
    Online,
    Away,
    Busy,
    Offline,
}

impl fmt::Display for PresenceStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Online => write!(f, "online"),
            Self::Away => write!(f, "away"),
            Self::Busy => write!(f, "busy"),
            Self::Offline => write!(f, "offline"),
        }
    }
}

impl PresenceStatus {
    /// Whether this status is considered "active" (not offline).
    pub fn is_active(self) -> bool {
        !matches!(self, Self::Offline)
    }
}

// ── User presence info ─────────────────────────────────────────────

/// Presence information for a single user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserPresence {
    pub user_id: String,
    pub status: PresenceStatus,
    pub status_text: Option<String>,
    /// Millisecond timestamp of the last heartbeat.
    pub last_heartbeat_ms: u64,
    /// Millisecond timestamp when user first became present.
    pub joined_at_ms: u64,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, String>,
}

impl UserPresence {
    pub fn new(user_id: impl Into<String>, now_ms: u64) -> Self {
        Self {
            user_id: user_id.into(),
            status: PresenceStatus::Online,
            status_text: None,
            last_heartbeat_ms: now_ms,
            joined_at_ms: now_ms,
            metadata: HashMap::new(),
        }
    }
}

// ── Presence diff ──────────────────────────────────────────────────

/// A single presence change event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresenceChange {
    Joined {
        user_id: String,
        status: PresenceStatus,
    },
    Left {
        user_id: String,
    },
    StatusChanged {
        user_id: String,
        old: PresenceStatus,
        new: PresenceStatus,
    },
}

/// A batch of presence changes since the last snapshot.
#[derive(Debug, Clone, Default)]
pub struct PresenceDiff {
    pub changes: Vec<PresenceChange>,
}

impl PresenceDiff {
    pub fn new() -> Self {
        Self { changes: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub fn joins(&self) -> Vec<&str> {
        self.changes.iter().filter_map(|c| match c {
            PresenceChange::Joined { user_id, .. } => Some(user_id.as_str()),
            _ => None,
        }).collect()
    }

    pub fn leaves(&self) -> Vec<&str> {
        self.changes.iter().filter_map(|c| match c {
            PresenceChange::Left { user_id } => Some(user_id.as_str()),
            _ => None,
        }).collect()
    }
}

// ── Presence channel ───────────────────────────────────────────────

/// Configuration for a presence channel.
#[derive(Debug, Clone)]
pub struct PresenceConfig {
    /// Heartbeat timeout in milliseconds. Users not heard from within this
    /// interval are marked offline.
    pub heartbeat_timeout_ms: u64,
    /// How long to keep offline users in the roster before eviction (ms).
    pub eviction_delay_ms: u64,
    /// Maximum number of users in this channel (0 = unlimited).
    pub max_users: usize,
    /// Auto-transition to Away after this many ms of no heartbeat (0 = disabled).
    pub auto_away_ms: u64,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout_ms: 30_000,
            eviction_delay_ms: 60_000,
            auto_away_ms: 0,
            max_users: 0,
        }
    }
}

/// A presence channel that tracks multiple users.
#[derive(Debug)]
pub struct PresenceChannel {
    pub name: String,
    config: PresenceConfig,
    users: BTreeMap<String, UserPresence>,
    /// Tracks when offline users should be evicted (user_id -> evict_at_ms).
    eviction_queue: HashMap<String, u64>,
    pending_diff: PresenceDiff,
}

impl PresenceChannel {
    pub fn new(name: impl Into<String>, config: PresenceConfig) -> Self {
        Self {
            name: name.into(),
            config,
            users: BTreeMap::new(),
            eviction_queue: HashMap::new(),
            pending_diff: PresenceDiff::new(),
        }
    }

    /// Number of tracked users (including offline not yet evicted).
    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    /// Number of active (non-offline) users.
    pub fn active_count(&self) -> usize {
        self.users.values().filter(|u| u.status.is_active()).count()
    }

    /// Join a user to this channel. Returns false if channel is full.
    pub fn join(&mut self, user_id: impl Into<String>, now_ms: u64) -> bool {
        let uid = user_id.into();
        if self.config.max_users > 0 && self.active_count() >= self.config.max_users {
            if !self.users.contains_key(&uid) {
                return false;
            }
        }
        // Remove from eviction queue if re-joining
        self.eviction_queue.remove(&uid);

        let was_present = self.users.contains_key(&uid);
        let entry = self.users.entry(uid.clone()).or_insert_with(|| UserPresence::new(&uid, now_ms));
        let old_status = entry.status;
        entry.status = PresenceStatus::Online;
        entry.last_heartbeat_ms = now_ms;
        entry.joined_at_ms = now_ms;

        if !was_present {
            self.pending_diff.changes.push(PresenceChange::Joined {
                user_id: uid,
                status: PresenceStatus::Online,
            });
        } else if old_status != PresenceStatus::Online {
            self.pending_diff.changes.push(PresenceChange::StatusChanged {
                user_id: uid,
                old: old_status,
                new: PresenceStatus::Online,
            });
        }
        true
    }

    /// User explicitly leaves the channel.
    pub fn leave(&mut self, user_id: &str) {
        if self.users.remove(user_id).is_some() {
            self.eviction_queue.remove(user_id);
            self.pending_diff.changes.push(PresenceChange::Left {
                user_id: user_id.to_string(),
            });
        }
    }

    /// Record a heartbeat from a user. Returns false if user is not tracked.
    pub fn heartbeat(&mut self, user_id: &str, now_ms: u64) -> bool {
        if let Some(user) = self.users.get_mut(user_id) {
            user.last_heartbeat_ms = now_ms;
            if user.status == PresenceStatus::Away {
                let old = user.status;
                user.status = PresenceStatus::Online;
                self.pending_diff.changes.push(PresenceChange::StatusChanged {
                    user_id: user_id.to_string(),
                    old,
                    new: PresenceStatus::Online,
                });
            }
            true
        } else {
            false
        }
    }

    /// Set a user's status explicitly.
    pub fn set_status(&mut self, user_id: &str, status: PresenceStatus) -> bool {
        if let Some(user) = self.users.get_mut(user_id) {
            let old = user.status;
            if old != status {
                user.status = status;
                if status == PresenceStatus::Offline {
                    self.pending_diff.changes.push(PresenceChange::Left {
                        user_id: user_id.to_string(),
                    });
                } else {
                    self.pending_diff.changes.push(PresenceChange::StatusChanged {
                        user_id: user_id.to_string(),
                        old,
                        new: status,
                    });
                }
            }
            true
        } else {
            false
        }
    }

    /// Set status text for a user.
    pub fn set_status_text(&mut self, user_id: &str, text: Option<String>) -> bool {
        if let Some(user) = self.users.get_mut(user_id) {
            user.status_text = text;
            true
        } else {
            false
        }
    }

    /// Get presence for a specific user.
    pub fn get_user(&self, user_id: &str) -> Option<&UserPresence> {
        self.users.get(user_id)
    }

    /// Get all active users in sorted order.
    pub fn active_users(&self) -> Vec<&UserPresence> {
        self.users.values().filter(|u| u.status.is_active()).collect()
    }

    /// Get all users (including offline) in sorted order.
    pub fn all_users(&self) -> Vec<&UserPresence> {
        self.users.values().collect()
    }

    /// Tick the presence system: check for timeouts, auto-away, evictions.
    pub fn tick(&mut self, now_ms: u64) {
        let timeout = self.config.heartbeat_timeout_ms;
        let auto_away = self.config.auto_away_ms;
        let eviction_delay = self.config.eviction_delay_ms;

        let mut to_offline: Vec<String> = Vec::new();
        let mut to_away: Vec<String> = Vec::new();

        for (uid, user) in &self.users {
            if !user.status.is_active() {
                continue;
            }
            let elapsed = now_ms.saturating_sub(user.last_heartbeat_ms);
            if elapsed >= timeout {
                to_offline.push(uid.clone());
            } else if auto_away > 0 && elapsed >= auto_away && user.status == PresenceStatus::Online {
                to_away.push(uid.clone());
            }
        }

        for uid in to_away {
            if let Some(user) = self.users.get_mut(&uid) {
                let old = user.status;
                user.status = PresenceStatus::Away;
                self.pending_diff.changes.push(PresenceChange::StatusChanged {
                    user_id: uid,
                    old,
                    new: PresenceStatus::Away,
                });
            }
        }

        for uid in to_offline {
            if let Some(user) = self.users.get_mut(&uid) {
                let old = user.status;
                user.status = PresenceStatus::Offline;
                self.eviction_queue.insert(uid.clone(), now_ms + eviction_delay);
                self.pending_diff.changes.push(PresenceChange::StatusChanged {
                    user_id: uid,
                    old,
                    new: PresenceStatus::Offline,
                });
            }
        }

        // Evict users past eviction deadline
        let to_evict: Vec<String> = self.eviction_queue.iter()
            .filter(|&(_, evict_at)| now_ms >= *evict_at)
            .map(|(uid, _)| uid.clone())
            .collect();
        for uid in to_evict {
            self.users.remove(&uid);
            self.eviction_queue.remove(&uid);
        }
    }

    /// Take the accumulated diff, resetting it.
    pub fn take_diff(&mut self) -> PresenceDiff {
        std::mem::take(&mut self.pending_diff)
    }
}

// ── Presence tracker (multi-channel) ───────────────────────────────

/// Manages multiple presence channels.
#[derive(Debug, Default)]
pub struct PresenceTracker {
    channels: HashMap<String, PresenceChannel>,
    default_config: PresenceConfig,
}

impl PresenceTracker {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            default_config: PresenceConfig::default(),
        }
    }

    pub fn with_default_config(config: PresenceConfig) -> Self {
        Self {
            channels: HashMap::new(),
            default_config: config,
        }
    }

    /// Get or create a channel.
    pub fn channel(&mut self, name: &str) -> &mut PresenceChannel {
        let config = self.default_config.clone();
        self.channels.entry(name.to_string())
            .or_insert_with(|| PresenceChannel::new(name, config))
    }

    /// Get a channel if it exists (immutable).
    pub fn get_channel(&self, name: &str) -> Option<&PresenceChannel> {
        self.channels.get(name)
    }

    /// Remove a channel entirely.
    pub fn remove_channel(&mut self, name: &str) -> bool {
        self.channels.remove(name).is_some()
    }

    /// Tick all channels.
    pub fn tick_all(&mut self, now_ms: u64) {
        for ch in self.channels.values_mut() {
            ch.tick(now_ms);
        }
    }

    /// List all channel names.
    pub fn channel_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.channels.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Find which channels a user is active in.
    pub fn user_channels(&self, user_id: &str) -> Vec<&str> {
        let mut result: Vec<&str> = self.channels.iter()
            .filter(|(_, ch)| {
                ch.get_user(user_id).is_some_and(|u| u.status.is_active())
            })
            .map(|(name, _)| name.as_str())
            .collect();
        result.sort();
        result
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Status ─────────────────────────────────────────────────────

    #[test]
    fn status_display() {
        assert_eq!(PresenceStatus::Online.to_string(), "online");
        assert_eq!(PresenceStatus::Away.to_string(), "away");
        assert_eq!(PresenceStatus::Busy.to_string(), "busy");
        assert_eq!(PresenceStatus::Offline.to_string(), "offline");
    }

    #[test]
    fn status_is_active() {
        assert!(PresenceStatus::Online.is_active());
        assert!(PresenceStatus::Away.is_active());
        assert!(PresenceStatus::Busy.is_active());
        assert!(!PresenceStatus::Offline.is_active());
    }

    // ── User presence ──────────────────────────────────────────────

    #[test]
    fn user_presence_creation() {
        let up = UserPresence::new("alice", 1000);
        assert_eq!(up.user_id, "alice");
        assert_eq!(up.status, PresenceStatus::Online);
        assert_eq!(up.last_heartbeat_ms, 1000);
        assert_eq!(up.joined_at_ms, 1000);
        assert!(up.status_text.is_none());
    }

    // ── Presence diff ──────────────────────────────────────────────

    #[test]
    fn diff_empty() {
        let diff = PresenceDiff::new();
        assert!(diff.is_empty());
        assert!(diff.joins().is_empty());
        assert!(diff.leaves().is_empty());
    }

    #[test]
    fn diff_joins_and_leaves() {
        let mut diff = PresenceDiff::new();
        diff.changes.push(PresenceChange::Joined { user_id: "a".into(), status: PresenceStatus::Online });
        diff.changes.push(PresenceChange::Left { user_id: "b".into() });
        diff.changes.push(PresenceChange::Joined { user_id: "c".into(), status: PresenceStatus::Online });

        assert_eq!(diff.joins(), vec!["a", "c"]);
        assert_eq!(diff.leaves(), vec!["b"]);
        assert!(!diff.is_empty());
    }

    // ── Channel: join/leave ────────────────────────────────────────

    #[test]
    fn channel_join_and_leave() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        assert!(ch.join("alice", 1000));
        assert!(ch.join("bob", 1001));
        assert_eq!(ch.user_count(), 2);
        assert_eq!(ch.active_count(), 2);

        ch.leave("alice");
        assert_eq!(ch.user_count(), 1);
        assert_eq!(ch.active_count(), 1);
    }

    #[test]
    fn channel_join_diff() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 1000);
        let diff = ch.take_diff();
        assert_eq!(diff.joins(), vec!["alice"]);
    }

    #[test]
    fn channel_leave_diff() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 1000);
        ch.take_diff(); // clear
        ch.leave("alice");
        let diff = ch.take_diff();
        assert_eq!(diff.leaves(), vec!["alice"]);
    }

    #[test]
    fn channel_leave_nonexistent() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.leave("ghost");
        assert!(ch.take_diff().is_empty());
    }

    #[test]
    fn channel_max_users() {
        let config = PresenceConfig { max_users: 2, ..Default::default() };
        let mut ch = PresenceChannel::new("room", config);
        assert!(ch.join("a", 0));
        assert!(ch.join("b", 0));
        assert!(!ch.join("c", 0)); // full
        assert_eq!(ch.active_count(), 2);
    }

    #[test]
    fn channel_rejoin_existing_user_when_full() {
        let config = PresenceConfig { max_users: 1, ..Default::default() };
        let mut ch = PresenceChannel::new("room", config);
        assert!(ch.join("a", 0));
        // Re-joining same user should succeed even when "full"
        assert!(ch.join("a", 100));
    }

    // ── Heartbeat ──────────────────────────────────────────────────

    #[test]
    fn heartbeat_updates_timestamp() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 1000);
        assert!(ch.heartbeat("alice", 2000));
        assert_eq!(ch.get_user("alice").unwrap().last_heartbeat_ms, 2000);
    }

    #[test]
    fn heartbeat_unknown_user() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        assert!(!ch.heartbeat("ghost", 1000));
    }

    #[test]
    fn heartbeat_revives_away_user() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        ch.set_status("alice", PresenceStatus::Away);
        ch.take_diff(); // clear
        ch.heartbeat("alice", 100);
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Online);
        let diff = ch.take_diff();
        assert!(diff.changes.iter().any(|c| matches!(c,
            PresenceChange::StatusChanged { new: PresenceStatus::Online, .. })));
    }

    // ── Status management ──────────────────────────────────────────

    #[test]
    fn set_status() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        ch.take_diff();
        assert!(ch.set_status("alice", PresenceStatus::Busy));
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Busy);

        let diff = ch.take_diff();
        assert!(diff.changes.iter().any(|c| matches!(c,
            PresenceChange::StatusChanged { old: PresenceStatus::Online, new: PresenceStatus::Busy, .. })));
    }

    #[test]
    fn set_status_no_change() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        ch.take_diff();
        ch.set_status("alice", PresenceStatus::Online); // same
        assert!(ch.take_diff().is_empty());
    }

    #[test]
    fn set_status_unknown_user() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        assert!(!ch.set_status("ghost", PresenceStatus::Online));
    }

    #[test]
    fn set_status_text() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        assert!(ch.set_status_text("alice", Some("In a meeting".into())));
        assert_eq!(ch.get_user("alice").unwrap().status_text.as_deref(), Some("In a meeting"));
        assert!(ch.set_status_text("alice", None));
        assert!(ch.get_user("alice").unwrap().status_text.is_none());
    }

    // ── Tick: timeout and eviction ─────────────────────────────────

    #[test]
    fn tick_timeout() {
        let config = PresenceConfig { heartbeat_timeout_ms: 100, eviction_delay_ms: 200, ..Default::default() };
        let mut ch = PresenceChannel::new("room", config);
        ch.join("alice", 0);
        ch.take_diff();

        ch.tick(50);
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Online);

        ch.tick(101);
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Offline);
        let diff = ch.take_diff();
        assert!(diff.changes.iter().any(|c| matches!(c,
            PresenceChange::StatusChanged { new: PresenceStatus::Offline, .. })));
    }

    #[test]
    fn tick_eviction() {
        let config = PresenceConfig { heartbeat_timeout_ms: 10, eviction_delay_ms: 50, ..Default::default() };
        let mut ch = PresenceChannel::new("room", config);
        ch.join("alice", 0);

        ch.tick(15); // goes offline at t=15, evict at t=65
        assert_eq!(ch.user_count(), 1); // still tracked
        assert_eq!(ch.active_count(), 0);

        ch.tick(60); // not evicted yet
        assert_eq!(ch.user_count(), 1);

        ch.tick(66); // now evicted
        assert_eq!(ch.user_count(), 0);
    }

    #[test]
    fn tick_auto_away() {
        let config = PresenceConfig {
            heartbeat_timeout_ms: 200,
            auto_away_ms: 50,
            ..Default::default()
        };
        let mut ch = PresenceChannel::new("room", config);
        ch.join("alice", 0);
        ch.take_diff();

        ch.tick(51);
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Away);
        let diff = ch.take_diff();
        assert!(diff.changes.iter().any(|c| matches!(c,
            PresenceChange::StatusChanged { new: PresenceStatus::Away, .. })));
    }

    // ── User listing ───────────────────────────────────────────────

    #[test]
    fn active_users_list() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        ch.join("bob", 1);
        ch.set_status("bob", PresenceStatus::Offline);

        let active = ch.active_users();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].user_id, "alice");

        let all = ch.all_users();
        assert_eq!(all.len(), 2);
    }

    // ── Presence tracker (multi-channel) ───────────────────────────

    #[test]
    fn tracker_multiple_channels() {
        let mut tracker = PresenceTracker::new();
        tracker.channel("general").join("alice", 0);
        tracker.channel("random").join("alice", 0);
        tracker.channel("random").join("bob", 1);

        assert_eq!(tracker.channel("general").active_count(), 1);
        assert_eq!(tracker.channel("random").active_count(), 2);
    }

    #[test]
    fn tracker_channel_names() {
        let mut tracker = PresenceTracker::new();
        tracker.channel("beta");
        tracker.channel("alpha");
        assert_eq!(tracker.channel_names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn tracker_remove_channel() {
        let mut tracker = PresenceTracker::new();
        tracker.channel("room");
        assert!(tracker.remove_channel("room"));
        assert!(!tracker.remove_channel("room"));
        assert!(tracker.get_channel("room").is_none());
    }

    #[test]
    fn tracker_tick_all() {
        let config = PresenceConfig { heartbeat_timeout_ms: 10, eviction_delay_ms: 5, ..Default::default() };
        let mut tracker = PresenceTracker::with_default_config(config);
        tracker.channel("a").join("alice", 0);
        tracker.channel("b").join("bob", 0);

        tracker.tick_all(20);
        assert_eq!(tracker.get_channel("a").unwrap().active_count(), 0);
        assert_eq!(tracker.get_channel("b").unwrap().active_count(), 0);
    }

    #[test]
    fn tracker_user_channels() {
        let mut tracker = PresenceTracker::new();
        tracker.channel("a").join("alice", 0);
        tracker.channel("b").join("alice", 0);
        tracker.channel("c").join("bob", 0);

        assert_eq!(tracker.user_channels("alice"), vec!["a", "b"]);
        assert_eq!(tracker.user_channels("bob"), vec!["c"]);
        assert!(tracker.user_channels("ghost").is_empty());
    }

    #[test]
    fn rejoin_after_offline_clears_eviction() {
        let config = PresenceConfig { heartbeat_timeout_ms: 10, eviction_delay_ms: 100, ..Default::default() };
        let mut ch = PresenceChannel::new("room", config);
        ch.join("alice", 0);
        ch.tick(20); // goes offline, evict at 120
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Offline);

        ch.join("alice", 30); // rejoin
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Online);

        // Send a heartbeat within the timeout window before checking past
        // the original eviction time.
        ch.heartbeat("alice", 125);
        ch.tick(130); // past original eviction time
        // Should NOT be evicted because re-joined and heartbeat is fresh
        assert_eq!(ch.user_count(), 1);
        assert_eq!(ch.get_user("alice").unwrap().status, PresenceStatus::Online);
    }

    #[test]
    fn set_status_to_offline_generates_left() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        ch.take_diff();
        ch.set_status("alice", PresenceStatus::Offline);
        let diff = ch.take_diff();
        assert_eq!(diff.leaves(), vec!["alice"]);
    }

    #[test]
    fn metadata_on_user() {
        let mut ch = PresenceChannel::new("room", PresenceConfig::default());
        ch.join("alice", 0);
        if let Some(user) = ch.users.get_mut("alice") {
            user.metadata.insert("role".into(), "admin".into());
        }
        assert_eq!(ch.get_user("alice").unwrap().metadata.get("role").unwrap(), "admin");
    }
}
