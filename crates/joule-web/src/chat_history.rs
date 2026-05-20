//! Chat history and persistence — logs, search, threading, reactions, read receipts.
//!
//! Replaces Firebase chat persistence / Slack history with pure Rust.
//! Paginated message logs, content/sender/time-range search, retention
//! policies, threading (reply-to), reactions, read cursors, compaction,
//! and edit/deletion tracking.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryError {
    MessageNotFound(u64),
    AlreadyDeleted(u64),
    DuplicateReaction(String),
    InvalidPage,
}

impl fmt::Display for HistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MessageNotFound(id) => write!(f, "message not found: {id}"),
            Self::AlreadyDeleted(id) => write!(f, "message already deleted: {id}"),
            Self::DuplicateReaction(r) => write!(f, "duplicate reaction: {r}"),
            Self::InvalidPage => write!(f, "invalid page parameters"),
        }
    }
}

impl std::error::Error for HistoryError {}

// ── MessageStatus ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageStatus {
    Active,
    Edited { original_content: String, edited_at: u64 },
    Deleted { deleted_at: u64 },
}

impl fmt::Display for MessageStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Edited { edited_at, .. } => write!(f, "edited at {edited_at}"),
            Self::Deleted { deleted_at } => write!(f, "deleted at {deleted_at}"),
        }
    }
}

// ── HistoryMessage ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HistoryMessage {
    pub id: u64,
    pub sender: String,
    pub content: String,
    pub timestamp: u64,
    pub reply_to: Option<u64>,
    pub reactions: HashMap<String, HashSet<String>>,
    pub status: MessageStatus,
}

impl HistoryMessage {
    pub fn new(id: u64, sender: &str, content: &str, timestamp: u64) -> Self {
        Self {
            id,
            sender: sender.to_string(),
            content: content.to_string(),
            timestamp,
            reply_to: None,
            reactions: HashMap::new(),
            status: MessageStatus::Active,
        }
    }

    pub fn with_reply_to(mut self, parent_id: u64) -> Self {
        self.reply_to = Some(parent_id);
        self
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, MessageStatus::Active | MessageStatus::Edited { .. })
    }

    pub fn is_deleted(&self) -> bool {
        matches!(self.status, MessageStatus::Deleted { .. })
    }

    pub fn reaction_count(&self, emoji: &str) -> usize {
        self.reactions.get(emoji).map_or(0, |s| s.len())
    }

    pub fn total_reactions(&self) -> usize {
        self.reactions.values().map(|s| s.len()).sum()
    }
}

impl fmt::Display for HistoryMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {} ({})", self.timestamp, self.sender, self.content, self.status)
    }
}

// ── RetentionPolicy ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub max_age_secs: Option<u64>,
    pub max_count: Option<usize>,
}

impl RetentionPolicy {
    pub fn new() -> Self {
        Self { max_age_secs: None, max_count: None }
    }

    pub fn with_max_age(mut self, secs: u64) -> Self {
        self.max_age_secs = Some(secs);
        self
    }

    pub fn with_max_count(mut self, count: usize) -> Self {
        self.max_count = Some(count);
        self
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ── Page ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Page<'a> {
    pub messages: Vec<&'a HistoryMessage>,
    pub page_index: usize,
    pub total_pages: usize,
}

// ── ChatLog ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChatLog {
    messages: Vec<HistoryMessage>,
    next_id: u64,
    read_cursors: HashMap<String, u64>,
    retention: RetentionPolicy,
}

impl ChatLog {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            next_id: 1,
            read_cursors: HashMap::new(),
            retention: RetentionPolicy::new(),
        }
    }

    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    pub fn set_retention(&mut self, policy: RetentionPolicy) {
        self.retention = policy;
    }

    pub fn append(&mut self, sender: &str, content: &str, timestamp: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.messages.push(HistoryMessage::new(id, sender, content, timestamp));
        id
    }

    pub fn append_reply(&mut self, sender: &str, content: &str, timestamp: u64, reply_to: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let msg = HistoryMessage::new(id, sender, content, timestamp).with_reply_to(reply_to);
        self.messages.push(msg);
        id
    }

    pub fn get(&self, id: u64) -> Option<&HistoryMessage> {
        self.messages.iter().find(|m| m.id == id)
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut HistoryMessage> {
        self.messages.iter_mut().find(|m| m.id == id)
    }

    pub fn edit_message(&mut self, id: u64, new_content: &str, edited_at: u64) -> Result<(), HistoryError> {
        let msg = self.get_mut(id).ok_or(HistoryError::MessageNotFound(id))?;
        if msg.is_deleted() {
            return Err(HistoryError::AlreadyDeleted(id));
        }
        let original = msg.content.clone();
        msg.content = new_content.to_string();
        msg.status = MessageStatus::Edited { original_content: original, edited_at };
        Ok(())
    }

    pub fn delete_message(&mut self, id: u64, deleted_at: u64) -> Result<(), HistoryError> {
        let msg = self.get_mut(id).ok_or(HistoryError::MessageNotFound(id))?;
        if msg.is_deleted() {
            return Err(HistoryError::AlreadyDeleted(id));
        }
        msg.status = MessageStatus::Deleted { deleted_at };
        Ok(())
    }

    pub fn add_reaction(&mut self, msg_id: u64, user: &str, emoji: &str) -> Result<(), HistoryError> {
        let msg = self.get_mut(msg_id).ok_or(HistoryError::MessageNotFound(msg_id))?;
        let users = msg.reactions.entry(emoji.to_string()).or_default();
        if !users.insert(user.to_string()) {
            return Err(HistoryError::DuplicateReaction(emoji.to_string()));
        }
        Ok(())
    }

    pub fn remove_reaction(&mut self, msg_id: u64, user: &str, emoji: &str) -> Result<(), HistoryError> {
        let msg = self.get_mut(msg_id).ok_or(HistoryError::MessageNotFound(msg_id))?;
        if let Some(users) = msg.reactions.get_mut(emoji) {
            users.remove(user);
            if users.is_empty() {
                msg.reactions.remove(emoji);
            }
        }
        Ok(())
    }

    pub fn set_read_cursor(&mut self, user: &str, msg_id: u64) {
        self.read_cursors.insert(user.to_string(), msg_id);
    }

    pub fn read_cursor(&self, user: &str) -> Option<u64> {
        self.read_cursors.get(user).copied()
    }

    pub fn unread_count(&self, user: &str) -> usize {
        let cursor = self.read_cursors.get(user).copied().unwrap_or(0);
        self.active_messages().filter(|m| m.id > cursor).count()
    }

    pub fn search_content(&self, query: &str) -> Vec<&HistoryMessage> {
        let lower = query.to_lowercase();
        self.active_messages().filter(|m| m.content.to_lowercase().contains(&lower)).collect()
    }

    pub fn search_sender(&self, sender: &str) -> Vec<&HistoryMessage> {
        self.active_messages().filter(|m| m.sender == sender).collect()
    }

    pub fn search_time_range(&self, start: u64, end: u64) -> Vec<&HistoryMessage> {
        self.active_messages().filter(|m| m.timestamp >= start && m.timestamp <= end).collect()
    }

    pub fn thread(&self, parent_id: u64) -> Vec<&HistoryMessage> {
        self.messages.iter().filter(|m| m.reply_to == Some(parent_id)).collect()
    }

    pub fn paginate(&self, page: usize, page_size: usize) -> Result<Page<'_>, HistoryError> {
        if page_size == 0 {
            return Err(HistoryError::InvalidPage);
        }
        let active: Vec<_> = self.active_messages().collect();
        let total_pages = if active.is_empty() { 1 } else { (active.len() + page_size - 1) / page_size };
        let start = page * page_size;
        let msgs = active.into_iter().skip(start).take(page_size).collect();
        Ok(Page { messages: msgs, page_index: page, total_pages })
    }

    pub fn compact(&mut self, now: u64) -> usize {
        let before = self.messages.len();
        if let Some(max_age) = self.retention.max_age_secs {
            let cutoff = now.saturating_sub(max_age);
            self.messages.retain(|m| m.timestamp >= cutoff);
        }
        if let Some(max_count) = self.retention.max_count {
            if self.messages.len() > max_count {
                let drain = self.messages.len() - max_count;
                self.messages.drain(..drain);
            }
        }
        before - self.messages.len()
    }

    pub fn export(&self) -> Vec<String> {
        self.messages.iter().map(|m| format!("{m}")).collect()
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn active_count(&self) -> usize {
        self.active_messages().count()
    }

    fn active_messages(&self) -> impl Iterator<Item = &HistoryMessage> {
        self.messages.iter().filter(|m| m.is_active())
    }
}

impl Default for ChatLog {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> ChatLog {
        let mut l = ChatLog::new();
        l.append("alice", "hello", 100);
        l.append("bob", "hi there", 101);
        l.append("alice", "how are you?", 102);
        l
    }

    #[test]
    fn test_append_and_len() {
        let l = log();
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn test_get_message() {
        let l = log();
        let m = l.get(1).unwrap();
        assert_eq!(m.sender, "alice");
        assert_eq!(m.content, "hello");
    }

    #[test]
    fn test_search_content() {
        let l = log();
        let found = l.search_content("hello");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].sender, "alice");
    }

    #[test]
    fn test_search_sender() {
        let l = log();
        let found = l.search_sender("alice");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_search_time_range() {
        let l = log();
        let found = l.search_time_range(100, 101);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_edit_message() {
        let mut l = log();
        l.edit_message(1, "hello world", 200).unwrap();
        let m = l.get(1).unwrap();
        assert_eq!(m.content, "hello world");
        assert!(matches!(m.status, MessageStatus::Edited { .. }));
    }

    #[test]
    fn test_delete_message() {
        let mut l = log();
        l.delete_message(2, 200).unwrap();
        assert!(l.get(2).unwrap().is_deleted());
        assert_eq!(l.active_count(), 2);
    }

    #[test]
    fn test_delete_already_deleted() {
        let mut l = log();
        l.delete_message(1, 200).unwrap();
        assert!(l.delete_message(1, 201).is_err());
    }

    #[test]
    fn test_reactions() {
        let mut l = log();
        l.add_reaction(1, "bob", "thumbs_up").unwrap();
        l.add_reaction(1, "charlie", "thumbs_up").unwrap();
        assert_eq!(l.get(1).unwrap().reaction_count("thumbs_up"), 2);
        assert_eq!(l.get(1).unwrap().total_reactions(), 2);
    }

    #[test]
    fn test_duplicate_reaction() {
        let mut l = log();
        l.add_reaction(1, "bob", "heart").unwrap();
        assert!(l.add_reaction(1, "bob", "heart").is_err());
    }

    #[test]
    fn test_remove_reaction() {
        let mut l = log();
        l.add_reaction(1, "bob", "heart").unwrap();
        l.remove_reaction(1, "bob", "heart").unwrap();
        assert_eq!(l.get(1).unwrap().total_reactions(), 0);
    }

    #[test]
    fn test_read_cursor_and_unread() {
        let mut l = log();
        l.set_read_cursor("alice", 1);
        assert_eq!(l.unread_count("alice"), 2);
        l.set_read_cursor("alice", 3);
        assert_eq!(l.unread_count("alice"), 0);
    }

    #[test]
    fn test_threading() {
        let mut l = log();
        l.append_reply("bob", "reply1", 103, 1);
        l.append_reply("charlie", "reply2", 104, 1);
        let thread = l.thread(1);
        assert_eq!(thread.len(), 2);
    }

    #[test]
    fn test_pagination() {
        let l = log();
        let p = l.paginate(0, 2).unwrap();
        assert_eq!(p.messages.len(), 2);
        assert_eq!(p.total_pages, 2);
        let p2 = l.paginate(1, 2).unwrap();
        assert_eq!(p2.messages.len(), 1);
    }

    #[test]
    fn test_pagination_invalid() {
        let l = log();
        assert!(l.paginate(0, 0).is_err());
    }

    #[test]
    fn test_compact_by_age() {
        let mut l = ChatLog::new().with_retention(RetentionPolicy::new().with_max_age(50));
        l.append("a", "old", 10);
        l.append("a", "new", 100);
        let removed = l.compact(110);
        assert_eq!(removed, 1);
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn test_compact_by_count() {
        let mut l = ChatLog::new().with_retention(RetentionPolicy::new().with_max_count(2));
        l.append("a", "m1", 1);
        l.append("a", "m2", 2);
        l.append("a", "m3", 3);
        let removed = l.compact(100);
        assert_eq!(removed, 1);
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn test_export() {
        let l = log();
        let exp = l.export();
        assert_eq!(exp.len(), 3);
        assert!(exp[0].contains("alice"));
    }

    #[test]
    fn test_display_history_message() {
        let m = HistoryMessage::new(1, "alice", "hello", 42);
        let s = format!("{m}");
        assert!(s.contains("alice"));
        assert!(s.contains("active"));
    }

    #[test]
    fn test_default_chatlog() {
        let l = ChatLog::default();
        assert!(l.is_empty());
        assert_eq!(l.len(), 0);
    }
}
