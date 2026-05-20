//! Chat room system — rooms, messages, moderation, ring-buffer history.
//!
//! Replaces Socket.IO / Pusher chat rooms with pure Rust.
//! Room lifecycle, membership, moderation (mute/kick), message filtering,
//! topic management, system announcements, and typing indicators.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRoomError {
    RoomNotFound(String),
    RoomFull(String),
    AlreadyInRoom(String),
    NotInRoom(String),
    UserMuted(String),
    DuplicateRoom(String),
    Filtered(String),
    NotAuthorized(String),
}

impl fmt::Display for ChatRoomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoomNotFound(id) => write!(f, "room not found: {id}"),
            Self::RoomFull(id) => write!(f, "room is full: {id}"),
            Self::AlreadyInRoom(u) => write!(f, "already in room: {u}"),
            Self::NotInRoom(u) => write!(f, "not in room: {u}"),
            Self::UserMuted(u) => write!(f, "user is muted: {u}"),
            Self::DuplicateRoom(id) => write!(f, "duplicate room: {id}"),
            Self::Filtered(r) => write!(f, "message filtered: {r}"),
            Self::NotAuthorized(u) => write!(f, "not authorized: {u}"),
        }
    }
}

impl std::error::Error for ChatRoomError {}

// ── MessageKind ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageKind {
    User,
    System,
}

impl fmt::Display for MessageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::System => write!(f, "system"),
        }
    }
}

// ── ChatMessage ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub id: u64,
    pub sender: String,
    pub content: String,
    pub timestamp: u64,
    pub kind: MessageKind,
}

impl ChatMessage {
    pub fn new(id: u64, sender: &str, content: &str, timestamp: u64) -> Self {
        Self {
            id,
            sender: sender.to_string(),
            content: content.to_string(),
            timestamp,
            kind: MessageKind::User,
        }
    }

    pub fn system(id: u64, content: &str, timestamp: u64) -> Self {
        Self {
            id,
            sender: String::from("system"),
            content: content.to_string(),
            timestamp,
            kind: MessageKind::System,
        }
    }
}

impl fmt::Display for ChatMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}: {}", self.timestamp, self.sender, self.content)
    }
}

// ── TypingIndicator ─────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypingIndicator {
    pub user_id: String,
    pub started_at: u64,
}

// ── ChatRoom ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChatRoom {
    pub id: String,
    pub name: String,
    pub topic: String,
    pub members: HashSet<String>,
    pub max_members: usize,
    history: VecDeque<ChatMessage>,
    history_capacity: usize,
    muted_users: HashSet<String>,
    next_msg_id: u64,
    typing: HashMap<String, u64>,
    filter_words: Vec<String>,
}

impl ChatRoom {
    pub fn new(id: &str, name: &str, max_members: usize, history_capacity: usize) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            topic: String::new(),
            members: HashSet::new(),
            max_members,
            history: VecDeque::with_capacity(history_capacity),
            history_capacity,
            muted_users: HashSet::new(),
            next_msg_id: 1,
            typing: HashMap::new(),
            filter_words: Vec::new(),
        }
    }

    pub fn set_topic(&mut self, topic: &str) {
        self.topic = topic.to_string();
    }

    pub fn add_filter_word(&mut self, word: &str) {
        self.filter_words.push(word.to_lowercase());
    }

    /// Join a user. Returns system announcement message.
    pub fn join(&mut self, user_id: &str) -> Result<ChatMessage, ChatRoomError> {
        if self.members.contains(user_id) {
            return Err(ChatRoomError::AlreadyInRoom(user_id.to_string()));
        }
        if self.members.len() >= self.max_members {
            return Err(ChatRoomError::RoomFull(self.id.clone()));
        }
        self.members.insert(user_id.to_string());
        let msg = self.push_system(&format!("{user_id} joined the room"));
        Ok(msg)
    }

    /// Leave a room. Returns system announcement message.
    pub fn leave(&mut self, user_id: &str) -> Result<ChatMessage, ChatRoomError> {
        if !self.members.remove(user_id) {
            return Err(ChatRoomError::NotInRoom(user_id.to_string()));
        }
        self.typing.remove(user_id);
        let msg = self.push_system(&format!("{user_id} left the room"));
        Ok(msg)
    }

    /// Send a message. Applies content filter and mute check.
    pub fn send_message(
        &mut self,
        sender: &str,
        content: &str,
        timestamp: u64,
    ) -> Result<&ChatMessage, ChatRoomError> {
        if !self.members.contains(sender) {
            return Err(ChatRoomError::NotInRoom(sender.to_string()));
        }
        if self.muted_users.contains(sender) {
            return Err(ChatRoomError::UserMuted(sender.to_string()));
        }
        let lower = content.to_lowercase();
        for word in &self.filter_words {
            if lower.contains(word.as_str()) {
                return Err(ChatRoomError::Filtered(word.clone()));
            }
        }
        let id = self.next_msg_id;
        self.next_msg_id += 1;
        let msg = ChatMessage::new(id, sender, content, timestamp);
        self.push_to_ring(msg);
        self.typing.remove(sender);
        Ok(self.history.back().unwrap())
    }

    pub fn mute_user(&mut self, user_id: &str) -> Result<(), ChatRoomError> {
        if !self.members.contains(user_id) {
            return Err(ChatRoomError::NotInRoom(user_id.to_string()));
        }
        self.muted_users.insert(user_id.to_string());
        Ok(())
    }

    pub fn unmute_user(&mut self, user_id: &str) {
        self.muted_users.remove(user_id);
    }

    pub fn kick_user(&mut self, user_id: &str) -> Result<ChatMessage, ChatRoomError> {
        if !self.members.remove(user_id) {
            return Err(ChatRoomError::NotInRoom(user_id.to_string()));
        }
        self.muted_users.remove(user_id);
        self.typing.remove(user_id);
        let msg = self.push_system(&format!("{user_id} was kicked from the room"));
        Ok(msg)
    }

    pub fn set_typing(&mut self, user_id: &str, timestamp: u64) {
        if self.members.contains(user_id) {
            self.typing.insert(user_id.to_string(), timestamp);
        }
    }

    pub fn clear_typing(&mut self, user_id: &str) {
        self.typing.remove(user_id);
    }

    pub fn typing_users(&self) -> Vec<&str> {
        self.typing.keys().map(|s| s.as_str()).collect()
    }

    pub fn message_count(&self) -> usize {
        self.history.len()
    }

    pub fn history(&self) -> &VecDeque<ChatMessage> {
        &self.history
    }

    pub fn recent_messages(&self, n: usize) -> Vec<&ChatMessage> {
        self.history.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect()
    }

    pub fn is_muted(&self, user_id: &str) -> bool {
        self.muted_users.contains(user_id)
    }

    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    // ── internal helpers ────────────────────────────────────────

    fn push_system(&mut self, content: &str) -> ChatMessage {
        let id = self.next_msg_id;
        self.next_msg_id += 1;
        let msg = ChatMessage::system(id, content, 0);
        self.push_to_ring(msg.clone());
        msg
    }

    fn push_to_ring(&mut self, msg: ChatMessage) {
        if self.history.len() == self.history_capacity {
            self.history.pop_front();
        }
        self.history.push_back(msg);
    }
}

impl fmt::Display for ChatRoom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChatRoom({}, members={}/{})",
            self.name,
            self.members.len(),
            self.max_members
        )
    }
}

// ── RoomManager ─────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct RoomManager {
    rooms: HashMap<String, ChatRoom>,
}

impl RoomManager {
    pub fn new() -> Self {
        Self { rooms: HashMap::new() }
    }

    pub fn create_room(
        &mut self,
        id: &str,
        name: &str,
        max_members: usize,
        history_capacity: usize,
    ) -> Result<(), ChatRoomError> {
        if self.rooms.contains_key(id) {
            return Err(ChatRoomError::DuplicateRoom(id.to_string()));
        }
        self.rooms.insert(id.to_string(), ChatRoom::new(id, name, max_members, history_capacity));
        Ok(())
    }

    pub fn destroy_room(&mut self, id: &str) -> Result<ChatRoom, ChatRoomError> {
        self.rooms.remove(id).ok_or_else(|| ChatRoomError::RoomNotFound(id.to_string()))
    }

    pub fn get_room(&self, id: &str) -> Option<&ChatRoom> {
        self.rooms.get(id)
    }

    pub fn get_room_mut(&mut self, id: &str) -> Option<&mut ChatRoom> {
        self.rooms.get_mut(id)
    }

    pub fn room_count(&self) -> usize {
        self.rooms.len()
    }

    pub fn list_rooms(&self) -> Vec<&str> {
        self.rooms.keys().map(|s| s.as_str()).collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn room() -> ChatRoom {
        ChatRoom::new("r1", "General", 10, 50)
    }

    #[test]
    fn test_join_and_leave() {
        let mut r = room();
        r.join("alice").unwrap();
        assert_eq!(r.member_count(), 1);
        r.leave("alice").unwrap();
        assert_eq!(r.member_count(), 0);
    }

    #[test]
    fn test_join_duplicate() {
        let mut r = room();
        r.join("alice").unwrap();
        assert!(r.join("alice").is_err());
    }

    #[test]
    fn test_room_full() {
        let mut r = ChatRoom::new("r1", "Small", 2, 10);
        r.join("a").unwrap();
        r.join("b").unwrap();
        assert_eq!(r.join("c"), Err(ChatRoomError::RoomFull("r1".into())));
    }

    #[test]
    fn test_send_message() {
        let mut r = room();
        r.join("alice").unwrap();
        r.send_message("alice", "hello", 100).unwrap();
        assert_eq!(r.message_count(), 2); // join announcement + message
    }

    #[test]
    fn test_send_not_member() {
        let mut r = room();
        assert!(r.send_message("ghost", "hi", 1).is_err());
    }

    #[test]
    fn test_mute_blocks_send() {
        let mut r = room();
        r.join("alice").unwrap();
        r.mute_user("alice").unwrap();
        assert!(r.send_message("alice", "hi", 1).is_err());
        assert!(r.is_muted("alice"));
    }

    #[test]
    fn test_unmute() {
        let mut r = room();
        r.join("alice").unwrap();
        r.mute_user("alice").unwrap();
        r.unmute_user("alice");
        assert!(!r.is_muted("alice"));
        r.send_message("alice", "hi", 1).unwrap();
    }

    #[test]
    fn test_kick_user() {
        let mut r = room();
        r.join("alice").unwrap();
        r.kick_user("alice").unwrap();
        assert_eq!(r.member_count(), 0);
    }

    #[test]
    fn test_message_filter() {
        let mut r = room();
        r.add_filter_word("badword");
        r.join("alice").unwrap();
        assert!(r.send_message("alice", "this is BADWORD here", 1).is_err());
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut r = ChatRoom::new("r1", "Tiny", 5, 3);
        r.join("a").unwrap();
        // join announcement is msg 1
        r.send_message("a", "msg1", 1).unwrap();
        r.send_message("a", "msg2", 2).unwrap();
        r.send_message("a", "msg3", 3).unwrap();
        // capacity=3, oldest dropped
        assert_eq!(r.message_count(), 3);
    }

    #[test]
    fn test_topic() {
        let mut r = room();
        r.set_topic("Rust chat");
        assert_eq!(r.topic, "Rust chat");
    }

    #[test]
    fn test_system_messages_on_join_leave() {
        let mut r = room();
        let join_msg = r.join("bob").unwrap();
        assert_eq!(join_msg.kind, MessageKind::System);
        assert!(join_msg.content.contains("joined"));
        let leave_msg = r.leave("bob").unwrap();
        assert!(leave_msg.content.contains("left"));
    }

    #[test]
    fn test_typing_indicator() {
        let mut r = room();
        r.join("alice").unwrap();
        r.set_typing("alice", 100);
        assert!(r.typing_users().contains(&"alice"));
        r.clear_typing("alice");
        assert!(r.typing_users().is_empty());
    }

    #[test]
    fn test_typing_cleared_on_send() {
        let mut r = room();
        r.join("alice").unwrap();
        r.set_typing("alice", 100);
        r.send_message("alice", "hello", 101).unwrap();
        assert!(!r.typing_users().contains(&"alice"));
    }

    #[test]
    fn test_recent_messages() {
        let mut r = room();
        r.join("a").unwrap();
        for i in 0..5 {
            r.send_message("a", &format!("m{i}"), i).unwrap();
        }
        let recent = r.recent_messages(2);
        assert_eq!(recent.len(), 2);
        assert!(recent[1].content.contains("m4"));
    }

    #[test]
    fn test_room_manager_create_destroy() {
        let mut mgr = RoomManager::new();
        mgr.create_room("r1", "General", 50, 100).unwrap();
        assert_eq!(mgr.room_count(), 1);
        mgr.destroy_room("r1").unwrap();
        assert_eq!(mgr.room_count(), 0);
    }

    #[test]
    fn test_room_manager_duplicate() {
        let mut mgr = RoomManager::new();
        mgr.create_room("r1", "A", 10, 10).unwrap();
        assert!(mgr.create_room("r1", "B", 10, 10).is_err());
    }

    #[test]
    fn test_display_chat_message() {
        let msg = ChatMessage::new(1, "alice", "hello", 42);
        let s = format!("{msg}");
        assert!(s.contains("alice"));
        assert!(s.contains("hello"));
    }

    #[test]
    fn test_display_chat_room() {
        let mut r = room();
        r.join("a").unwrap();
        let s = format!("{r}");
        assert!(s.contains("General"));
        assert!(s.contains("1/10"));
    }

    #[test]
    fn test_leave_not_member() {
        let mut r = room();
        assert!(r.leave("ghost").is_err());
    }
}
