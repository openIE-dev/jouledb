//! Voice chat lobby and channel management — channels, mute/deafen, volume, idle detection.
//!
//! Replaces Discord.js voice / LiveKit lobby management with pure Rust.
//! Voice channel lifecycle, join/leave, mute/unmute/deafen, speaking state
//! simulation, per-user volume, priority speaker, channel moves, stats,
//! and auto-disconnect on idle.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceError {
    ChannelNotFound(String),
    ChannelFull(String),
    UserNotInChannel(String),
    UserAlreadyInChannel(String),
    DuplicateChannel(String),
    InvalidVolume(u16),
}

impl fmt::Display for VoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelNotFound(id) => write!(f, "channel not found: {id}"),
            Self::ChannelFull(id) => write!(f, "channel full: {id}"),
            Self::UserNotInChannel(u) => write!(f, "user not in channel: {u}"),
            Self::UserAlreadyInChannel(u) => write!(f, "user already in channel: {u}"),
            Self::DuplicateChannel(id) => write!(f, "duplicate channel: {id}"),
            Self::InvalidVolume(v) => write!(f, "invalid volume {v}, max 200"),
        }
    }
}

impl std::error::Error for VoiceError {}

// ── VoiceUser ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VoiceUser {
    pub user_id: String,
    pub muted: bool,
    pub deafened: bool,
    pub speaking: bool,
    pub volume: u16,
    pub priority_speaker: bool,
    pub last_activity: u64,
    pub joined_at: u64,
}

impl VoiceUser {
    pub fn new(user_id: &str, joined_at: u64) -> Self {
        Self {
            user_id: user_id.to_string(),
            muted: false,
            deafened: false,
            speaking: false,
            volume: 100,
            priority_speaker: false,
            last_activity: joined_at,
            joined_at,
        }
    }

    pub fn idle_duration(&self, now: u64) -> u64 {
        now.saturating_sub(self.last_activity)
    }

    pub fn session_duration(&self, now: u64) -> u64 {
        now.saturating_sub(self.joined_at)
    }

    pub fn effective_volume(&self) -> u16 {
        if self.muted { 0 } else { self.volume }
    }
}

impl fmt::Display for VoiceUser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = if self.deafened {
            "deafened"
        } else if self.muted {
            "muted"
        } else if self.speaking {
            "speaking"
        } else {
            "idle"
        };
        write!(f, "{} ({})", self.user_id, state)
    }
}

// ── ChannelStats ────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ChannelStats {
    pub current_users: usize,
    pub peak_concurrent: usize,
    pub total_joins: u64,
}

// ── VoiceChannel ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VoiceChannel {
    pub id: String,
    pub name: String,
    pub max_users: usize,
    pub bitrate: u32,
    users: HashMap<String, VoiceUser>,
    stats: ChannelStats,
}

impl VoiceChannel {
    pub fn new(id: &str, name: &str, max_users: usize, bitrate: u32) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            max_users,
            bitrate,
            users: HashMap::new(),
            stats: ChannelStats::default(),
        }
    }

    pub fn join(&mut self, user_id: &str, timestamp: u64) -> Result<(), VoiceError> {
        if self.users.contains_key(user_id) {
            return Err(VoiceError::UserAlreadyInChannel(user_id.to_string()));
        }
        if self.users.len() >= self.max_users {
            return Err(VoiceError::ChannelFull(self.id.clone()));
        }
        self.users.insert(user_id.to_string(), VoiceUser::new(user_id, timestamp));
        self.stats.total_joins += 1;
        self.stats.current_users = self.users.len();
        if self.stats.current_users > self.stats.peak_concurrent {
            self.stats.peak_concurrent = self.stats.current_users;
        }
        Ok(())
    }

    pub fn leave(&mut self, user_id: &str) -> Result<VoiceUser, VoiceError> {
        let user = self.users.remove(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        self.stats.current_users = self.users.len();
        Ok(user)
    }

    pub fn mute(&mut self, user_id: &str) -> Result<(), VoiceError> {
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        u.muted = true;
        u.speaking = false;
        Ok(())
    }

    pub fn unmute(&mut self, user_id: &str) -> Result<(), VoiceError> {
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        u.muted = false;
        Ok(())
    }

    pub fn deafen(&mut self, user_id: &str) -> Result<(), VoiceError> {
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        u.deafened = true;
        u.muted = true;
        u.speaking = false;
        Ok(())
    }

    pub fn undeafen(&mut self, user_id: &str) -> Result<(), VoiceError> {
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        u.deafened = false;
        u.muted = false;
        Ok(())
    }

    pub fn set_speaking(&mut self, user_id: &str, speaking: bool, timestamp: u64) -> Result<(), VoiceError> {
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        if !u.muted {
            u.speaking = speaking;
            u.last_activity = timestamp;
        }
        Ok(())
    }

    pub fn set_volume(&mut self, user_id: &str, volume: u16) -> Result<(), VoiceError> {
        if volume > 200 {
            return Err(VoiceError::InvalidVolume(volume));
        }
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        u.volume = volume;
        Ok(())
    }

    pub fn set_priority_speaker(&mut self, user_id: &str, priority: bool) -> Result<(), VoiceError> {
        let u = self.users.get_mut(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        u.priority_speaker = priority;
        Ok(())
    }

    pub fn get_user(&self, user_id: &str) -> Option<&VoiceUser> {
        self.users.get(user_id)
    }

    pub fn user_count(&self) -> usize {
        self.users.len()
    }

    pub fn speaking_users(&self) -> Vec<&str> {
        self.users.values().filter(|u| u.speaking).map(|u| u.user_id.as_str()).collect()
    }

    pub fn stats(&self) -> &ChannelStats {
        &self.stats
    }

    /// Auto-disconnect users idle beyond threshold. Returns disconnected user IDs.
    pub fn auto_disconnect_idle(&mut self, now: u64, idle_threshold: u64) -> Vec<String> {
        let idle_ids: Vec<String> = self.users.values()
            .filter(|u| u.idle_duration(now) >= idle_threshold)
            .map(|u| u.user_id.clone())
            .collect();
        for id in &idle_ids {
            self.users.remove(id);
        }
        self.stats.current_users = self.users.len();
        idle_ids
    }
}

impl fmt::Display for VoiceChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}/{}, {}kbps)", self.name, self.user_count(), self.max_users, self.bitrate / 1000)
    }
}

// ── VoiceManager ────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct VoiceManager {
    channels: HashMap<String, VoiceChannel>,
    user_channel: HashMap<String, String>,
}

impl VoiceManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_channel(&mut self, id: &str, name: &str, max_users: usize, bitrate: u32) -> Result<(), VoiceError> {
        if self.channels.contains_key(id) {
            return Err(VoiceError::DuplicateChannel(id.to_string()));
        }
        self.channels.insert(id.to_string(), VoiceChannel::new(id, name, max_users, bitrate));
        Ok(())
    }

    pub fn remove_channel(&mut self, id: &str) -> Result<VoiceChannel, VoiceError> {
        let ch = self.channels.remove(id).ok_or_else(|| VoiceError::ChannelNotFound(id.to_string()))?;
        for user in ch.users.keys() {
            self.user_channel.remove(user);
        }
        Ok(ch)
    }

    pub fn join_channel(&mut self, user_id: &str, channel_id: &str, timestamp: u64) -> Result<(), VoiceError> {
        // Leave current channel first
        if let Some(current) = self.user_channel.get(user_id).cloned() {
            if current == channel_id {
                return Err(VoiceError::UserAlreadyInChannel(user_id.to_string()));
            }
            if let Some(ch) = self.channels.get_mut(&current) {
                let _ = ch.leave(user_id);
            }
        }
        let ch = self.channels.get_mut(channel_id).ok_or_else(|| VoiceError::ChannelNotFound(channel_id.to_string()))?;
        ch.join(user_id, timestamp)?;
        self.user_channel.insert(user_id.to_string(), channel_id.to_string());
        Ok(())
    }

    pub fn leave_channel(&mut self, user_id: &str) -> Result<VoiceUser, VoiceError> {
        let channel_id = self.user_channel.remove(user_id).ok_or_else(|| VoiceError::UserNotInChannel(user_id.to_string()))?;
        let ch = self.channels.get_mut(&channel_id).ok_or_else(|| VoiceError::ChannelNotFound(channel_id.clone()))?;
        ch.leave(user_id)
    }

    pub fn move_user(&mut self, user_id: &str, target_channel: &str, timestamp: u64) -> Result<(), VoiceError> {
        self.join_channel(user_id, target_channel, timestamp)
    }

    pub fn get_channel(&self, id: &str) -> Option<&VoiceChannel> {
        self.channels.get(id)
    }

    pub fn get_channel_mut(&mut self, id: &str) -> Option<&mut VoiceChannel> {
        self.channels.get_mut(id)
    }

    pub fn user_channel(&self, user_id: &str) -> Option<&str> {
        self.user_channel.get(user_id).map(|s| s.as_str())
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn channel() -> VoiceChannel {
        VoiceChannel::new("v1", "General Voice", 10, 128_000)
    }

    #[test]
    fn test_join_leave() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        assert_eq!(ch.user_count(), 1);
        ch.leave("alice").unwrap();
        assert_eq!(ch.user_count(), 0);
    }

    #[test]
    fn test_channel_full() {
        let mut ch = VoiceChannel::new("v1", "Small", 2, 64_000);
        ch.join("a", 1).unwrap();
        ch.join("b", 1).unwrap();
        assert!(ch.join("c", 1).is_err());
    }

    #[test]
    fn test_mute_unmute() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.mute("alice").unwrap();
        assert!(ch.get_user("alice").unwrap().muted);
        ch.unmute("alice").unwrap();
        assert!(!ch.get_user("alice").unwrap().muted);
    }

    #[test]
    fn test_deafen_undeafen() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.deafen("alice").unwrap();
        let u = ch.get_user("alice").unwrap();
        assert!(u.deafened);
        assert!(u.muted);
        ch.undeafen("alice").unwrap();
        let u = ch.get_user("alice").unwrap();
        assert!(!u.deafened);
        assert!(!u.muted);
    }

    #[test]
    fn test_speaking() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.set_speaking("alice", true, 1001).unwrap();
        assert!(ch.speaking_users().contains(&"alice"));
        ch.set_speaking("alice", false, 1002).unwrap();
        assert!(ch.speaking_users().is_empty());
    }

    #[test]
    fn test_muted_cannot_speak() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.mute("alice").unwrap();
        ch.set_speaking("alice", true, 1001).unwrap();
        assert!(!ch.get_user("alice").unwrap().speaking);
    }

    #[test]
    fn test_volume() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.set_volume("alice", 150).unwrap();
        assert_eq!(ch.get_user("alice").unwrap().volume, 150);
    }

    #[test]
    fn test_volume_invalid() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        assert!(ch.set_volume("alice", 201).is_err());
    }

    #[test]
    fn test_effective_volume_muted() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.set_volume("alice", 150).unwrap();
        ch.mute("alice").unwrap();
        assert_eq!(ch.get_user("alice").unwrap().effective_volume(), 0);
    }

    #[test]
    fn test_priority_speaker() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.set_priority_speaker("alice", true).unwrap();
        assert!(ch.get_user("alice").unwrap().priority_speaker);
    }

    #[test]
    fn test_auto_disconnect_idle() {
        let mut ch = channel();
        ch.join("alice", 1000).unwrap();
        ch.join("bob", 2000).unwrap();
        let disconnected = ch.auto_disconnect_idle(3000, 1500);
        assert_eq!(disconnected.len(), 1);
        assert!(disconnected.contains(&"alice".to_string()));
    }

    #[test]
    fn test_channel_stats() {
        let mut ch = channel();
        ch.join("a", 1).unwrap();
        ch.join("b", 1).unwrap();
        assert_eq!(ch.stats().peak_concurrent, 2);
        ch.leave("a").unwrap();
        assert_eq!(ch.stats().peak_concurrent, 2);
        assert_eq!(ch.stats().current_users, 1);
        assert_eq!(ch.stats().total_joins, 2);
    }

    #[test]
    fn test_voice_manager_create_join() {
        let mut mgr = VoiceManager::new();
        mgr.create_channel("v1", "General", 10, 128_000).unwrap();
        mgr.join_channel("alice", "v1", 1000).unwrap();
        assert_eq!(mgr.user_channel("alice"), Some("v1"));
    }

    #[test]
    fn test_voice_manager_move_user() {
        let mut mgr = VoiceManager::new();
        mgr.create_channel("v1", "A", 10, 128_000).unwrap();
        mgr.create_channel("v2", "B", 10, 128_000).unwrap();
        mgr.join_channel("alice", "v1", 1000).unwrap();
        mgr.move_user("alice", "v2", 1001).unwrap();
        assert_eq!(mgr.user_channel("alice"), Some("v2"));
        assert_eq!(mgr.get_channel("v1").unwrap().user_count(), 0);
    }

    #[test]
    fn test_voice_manager_leave() {
        let mut mgr = VoiceManager::new();
        mgr.create_channel("v1", "A", 10, 128_000).unwrap();
        mgr.join_channel("alice", "v1", 1000).unwrap();
        mgr.leave_channel("alice").unwrap();
        assert!(mgr.user_channel("alice").is_none());
    }

    #[test]
    fn test_display_voice_user() {
        let u = VoiceUser::new("alice", 1000);
        let s = format!("{u}");
        assert!(s.contains("alice"));
        assert!(s.contains("idle"));
    }

    #[test]
    fn test_display_channel() {
        let ch = channel();
        let s = format!("{ch}");
        assert!(s.contains("General Voice"));
        assert!(s.contains("128kbps"));
    }

    #[test]
    fn test_session_duration() {
        let u = VoiceUser::new("alice", 1000);
        assert_eq!(u.session_duration(2000), 1000);
    }

    #[test]
    fn test_duplicate_channel() {
        let mut mgr = VoiceManager::new();
        mgr.create_channel("v1", "A", 10, 128_000).unwrap();
        assert!(mgr.create_channel("v1", "B", 10, 128_000).is_err());
    }

    #[test]
    fn test_remove_channel() {
        let mut mgr = VoiceManager::new();
        mgr.create_channel("v1", "A", 10, 128_000).unwrap();
        mgr.remove_channel("v1").unwrap();
        assert_eq!(mgr.channel_count(), 0);
    }
}
