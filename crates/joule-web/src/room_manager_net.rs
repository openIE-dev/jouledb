//! Network room management for multiplayer — create, destroy, search, events.
//!
//! Replaces room-manager.js / Mirror-NetworkRoom with pure Rust.
//! Room with id/capacity/properties, RoomManager with create/destroy/list,
//! auto-cleanup of empty rooms, room properties (key-value), locking,
//! visibility (public/private/hidden), events, player limit enforcement,
//! room search with filters.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomError {
    RoomNotFound(String),
    RoomFull(String),
    RoomLocked(String),
    PlayerAlreadyInRoom(String),
    PlayerNotInRoom(String),
    DuplicateRoom(String),
    RoomHidden(String),
    PropertyNotFound(String),
}

impl fmt::Display for RoomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoomNotFound(id) => write!(f, "room not found: {id}"),
            Self::RoomFull(id) => write!(f, "room full: {id}"),
            Self::RoomLocked(id) => write!(f, "room locked: {id}"),
            Self::PlayerAlreadyInRoom(p) => write!(f, "player already in room: {p}"),
            Self::PlayerNotInRoom(p) => write!(f, "player not in room: {p}"),
            Self::DuplicateRoom(id) => write!(f, "duplicate room: {id}"),
            Self::RoomHidden(id) => write!(f, "room is hidden: {id}"),
            Self::PropertyNotFound(k) => write!(f, "property not found: {k}"),
        }
    }
}

impl std::error::Error for RoomError {}

// ── Visibility ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoomVisibility {
    Public,
    Private,
    Hidden,
}

impl fmt::Display for RoomVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => write!(f, "Public"),
            Self::Private => write!(f, "Private"),
            Self::Hidden => write!(f, "Hidden"),
        }
    }
}

// ── Room Event ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoomEvent {
    PlayerJoined(String),
    PlayerLeft(String),
    RoomFull,
    RoomEmpty,
    PropertyChanged { key: String, value: String },
    RoomLocked,
    RoomUnlocked,
}

impl fmt::Display for RoomEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerJoined(p) => write!(f, "PlayerJoined({p})"),
            Self::PlayerLeft(p) => write!(f, "PlayerLeft({p})"),
            Self::RoomFull => write!(f, "RoomFull"),
            Self::RoomEmpty => write!(f, "RoomEmpty"),
            Self::PropertyChanged { key, value } => write!(f, "PropertyChanged({key}={value})"),
            Self::RoomLocked => write!(f, "RoomLocked"),
            Self::RoomUnlocked => write!(f, "RoomUnlocked"),
        }
    }
}

// ── Room ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Room {
    pub id: String,
    pub name: String,
    pub capacity: usize,
    pub visibility: RoomVisibility,
    pub locked: bool,
    players: Vec<String>,
    properties: HashMap<String, String>,
    events: Vec<RoomEvent>,
    event_limit: usize,
}

impl Room {
    pub fn new(id: &str, name: &str, capacity: usize) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            capacity,
            visibility: RoomVisibility::Public,
            locked: false,
            players: Vec::new(),
            properties: HashMap::new(),
            events: Vec::new(),
            event_limit: 500,
        }
    }

    pub fn with_visibility(mut self, vis: RoomVisibility) -> Self {
        self.visibility = vis;
        self
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn is_full(&self) -> bool {
        self.players.len() >= self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.players.is_empty()
    }

    pub fn players(&self) -> &[String] {
        &self.players
    }

    pub fn join(&mut self, player: &str) -> Result<(), RoomError> {
        if self.locked {
            return Err(RoomError::RoomLocked(self.id.clone()));
        }
        if self.players.contains(&player.to_string()) {
            return Err(RoomError::PlayerAlreadyInRoom(player.to_string()));
        }
        if self.is_full() {
            return Err(RoomError::RoomFull(self.id.clone()));
        }
        self.players.push(player.to_string());
        self.push_event(RoomEvent::PlayerJoined(player.to_string()));
        if self.is_full() {
            self.push_event(RoomEvent::RoomFull);
        }
        Ok(())
    }

    pub fn leave(&mut self, player: &str) -> Result<(), RoomError> {
        let idx = self
            .players
            .iter()
            .position(|p| p == player)
            .ok_or_else(|| RoomError::PlayerNotInRoom(player.to_string()))?;
        self.players.remove(idx);
        self.push_event(RoomEvent::PlayerLeft(player.to_string()));
        if self.is_empty() {
            self.push_event(RoomEvent::RoomEmpty);
        }
        Ok(())
    }

    pub fn lock(&mut self) {
        self.locked = true;
        self.push_event(RoomEvent::RoomLocked);
    }

    pub fn unlock(&mut self) {
        self.locked = false;
        self.push_event(RoomEvent::RoomUnlocked);
    }

    pub fn set_property(&mut self, key: &str, value: &str) {
        self.properties
            .insert(key.to_string(), value.to_string());
        self.push_event(RoomEvent::PropertyChanged {
            key: key.to_string(),
            value: value.to_string(),
        });
    }

    pub fn get_property(&self, key: &str) -> Result<&str, RoomError> {
        self.properties
            .get(key)
            .map(|s| s.as_str())
            .ok_or_else(|| RoomError::PropertyNotFound(key.to_string()))
    }

    pub fn properties(&self) -> &HashMap<String, String> {
        &self.properties
    }

    pub fn events(&self) -> &[RoomEvent] {
        &self.events
    }

    pub fn drain_events(&mut self) -> Vec<RoomEvent> {
        std::mem::take(&mut self.events)
    }

    fn push_event(&mut self, event: RoomEvent) {
        self.events.push(event);
        if self.events.len() > self.event_limit {
            self.events.remove(0);
        }
    }
}

impl fmt::Display for Room {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Room({}, {}/{}, {})",
            self.name,
            self.player_count(),
            self.capacity,
            self.visibility
        )
    }
}

// ── Search Filter ───────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct RoomFilter {
    pub name_contains: Option<String>,
    pub has_space: bool,
    pub visibility: Option<RoomVisibility>,
    pub property_match: HashMap<String, String>,
}

impl RoomFilter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name_contains = Some(name.to_lowercase());
        self
    }

    pub fn with_space(mut self) -> Self {
        self.has_space = true;
        self
    }

    pub fn with_visibility(mut self, vis: RoomVisibility) -> Self {
        self.visibility = Some(vis);
        self
    }

    pub fn with_property(mut self, key: &str, value: &str) -> Self {
        self.property_match.insert(key.to_string(), value.to_string());
        self
    }

    pub fn matches(&self, room: &Room) -> bool {
        if let Some(ref q) = self.name_contains {
            if !room.name.to_lowercase().contains(q) {
                return false;
            }
        }
        if self.has_space && room.is_full() {
            return false;
        }
        if let Some(vis) = self.visibility {
            if room.visibility != vis {
                return false;
            }
        }
        for (k, v) in &self.property_match {
            match room.properties.get(k) {
                Some(rv) if rv == v => {}
                _ => return false,
            }
        }
        true
    }
}

// ── Room Manager ────────────────────────────────────────────────

#[derive(Debug)]
pub struct RoomManager {
    rooms: HashMap<String, Room>,
    next_id: u64,
    auto_cleanup: bool,
}

impl RoomManager {
    pub fn new() -> Self {
        Self {
            rooms: HashMap::new(),
            next_id: 1,
            auto_cleanup: true,
        }
    }

    pub fn with_auto_cleanup(mut self, enabled: bool) -> Self {
        self.auto_cleanup = enabled;
        self
    }

    pub fn create(&mut self, name: &str, capacity: usize) -> String {
        let id = format!("room-{}", self.next_id);
        self.next_id += 1;
        self.rooms.insert(id.clone(), Room::new(&id, name, capacity));
        id
    }

    pub fn create_with_visibility(&mut self, name: &str, capacity: usize, vis: RoomVisibility) -> String {
        let id = self.create(name, capacity);
        self.rooms.get_mut(&id).unwrap().visibility = vis;
        id
    }

    pub fn get(&self, id: &str) -> Result<&Room, RoomError> {
        self.rooms
            .get(id)
            .ok_or_else(|| RoomError::RoomNotFound(id.to_string()))
    }

    pub fn get_mut(&mut self, id: &str) -> Result<&mut Room, RoomError> {
        self.rooms
            .get_mut(id)
            .ok_or_else(|| RoomError::RoomNotFound(id.to_string()))
    }

    pub fn destroy(&mut self, id: &str) -> Result<(), RoomError> {
        if self.rooms.remove(id).is_none() {
            return Err(RoomError::RoomNotFound(id.to_string()));
        }
        Ok(())
    }

    pub fn room_count(&self) -> usize {
        self.rooms.len()
    }

    pub fn cleanup_empty(&mut self) -> usize {
        let before = self.rooms.len();
        self.rooms.retain(|_, r| !r.is_empty());
        before - self.rooms.len()
    }

    pub fn list_public(&self) -> Vec<&Room> {
        self.rooms
            .values()
            .filter(|r| r.visibility == RoomVisibility::Public)
            .collect()
    }

    pub fn search(&self, filter: &RoomFilter) -> Vec<&Room> {
        self.rooms
            .values()
            .filter(|r| r.visibility != RoomVisibility::Hidden)
            .filter(|r| filter.matches(r))
            .collect()
    }

    /// Leave room and auto-cleanup if empty.
    pub fn leave_room(&mut self, room_id: &str, player: &str) -> Result<(), RoomError> {
        let room = self.get_mut(room_id)?;
        room.leave(player)?;
        if self.auto_cleanup && self.rooms.get(room_id).map_or(false, |r| r.is_empty()) {
            self.rooms.remove(room_id);
        }
        Ok(())
    }
}

impl Default for RoomManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_join() {
        let mut mgr = RoomManager::new();
        let id = mgr.create("Arena", 4);
        mgr.get_mut(&id).unwrap().join("alice").unwrap();
        assert_eq!(mgr.get(&id).unwrap().player_count(), 1);
    }

    #[test]
    fn room_full_error() {
        let mut room = Room::new("r1", "Small", 1);
        room.join("alice").unwrap();
        let err = room.join("bob").unwrap_err();
        assert!(matches!(err, RoomError::RoomFull(_)));
    }

    #[test]
    fn room_locked_error() {
        let mut room = Room::new("r1", "Locked", 4);
        room.lock();
        let err = room.join("alice").unwrap_err();
        assert!(matches!(err, RoomError::RoomLocked(_)));
    }

    #[test]
    fn join_leave_events() {
        let mut room = Room::new("r1", "Arena", 4);
        room.join("alice").unwrap();
        room.leave("alice").unwrap();
        let events = room.events();
        assert_eq!(events.len(), 3); // Joined + Left + Empty
        assert!(matches!(&events[0], RoomEvent::PlayerJoined(_)));
        assert!(matches!(&events[1], RoomEvent::PlayerLeft(_)));
        assert!(matches!(&events[2], RoomEvent::RoomEmpty));
    }

    #[test]
    fn room_full_event() {
        let mut room = Room::new("r1", "Tiny", 1);
        room.join("alice").unwrap();
        let last = room.events().last().unwrap();
        assert!(matches!(last, RoomEvent::RoomFull));
    }

    #[test]
    fn properties() {
        let mut room = Room::new("r1", "Arena", 4);
        room.set_property("mode", "ctf");
        assert_eq!(room.get_property("mode").unwrap(), "ctf");
        let err = room.get_property("missing").unwrap_err();
        assert!(matches!(err, RoomError::PropertyNotFound(_)));
    }

    #[test]
    fn property_change_event() {
        let mut room = Room::new("r1", "Arena", 4);
        room.set_property("map", "dust2");
        let ev = room.events().last().unwrap();
        assert!(matches!(ev, RoomEvent::PropertyChanged { .. }));
    }

    #[test]
    fn visibility() {
        let room = Room::new("r1", "Arena", 4).with_visibility(RoomVisibility::Private);
        assert_eq!(room.visibility, RoomVisibility::Private);
    }

    #[test]
    fn auto_cleanup_on_leave() {
        let mut mgr = RoomManager::new();
        let id = mgr.create("Arena", 4);
        mgr.get_mut(&id).unwrap().join("alice").unwrap();
        mgr.leave_room(&id, "alice").unwrap();
        assert_eq!(mgr.room_count(), 0);
    }

    #[test]
    fn no_auto_cleanup_when_disabled() {
        let mut mgr = RoomManager::new().with_auto_cleanup(false);
        let id = mgr.create("Arena", 4);
        mgr.get_mut(&id).unwrap().join("alice").unwrap();
        mgr.leave_room(&id, "alice").unwrap();
        assert_eq!(mgr.room_count(), 1);
    }

    #[test]
    fn manual_cleanup_empty() {
        let mut mgr = RoomManager::new().with_auto_cleanup(false);
        let id = mgr.create("Arena", 4);
        mgr.get_mut(&id).unwrap().join("alice").unwrap();
        mgr.get_mut(&id).unwrap().leave("alice").unwrap();
        let removed = mgr.cleanup_empty();
        assert_eq!(removed, 1);
    }

    #[test]
    fn search_by_name() {
        let mut mgr = RoomManager::new();
        let _id1 = mgr.create("Pro Arena", 4);
        let _id2 = mgr.create("Casual Zone", 4);
        let filter = RoomFilter::new().with_name("arena");
        let results = mgr.search(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Pro Arena");
    }

    #[test]
    fn search_with_space() {
        let mut mgr = RoomManager::new();
        let id1 = mgr.create("Open", 2);
        mgr.get_mut(&id1).unwrap().join("a").unwrap();
        let id2 = mgr.create("Full", 1);
        mgr.get_mut(&id2).unwrap().join("b").unwrap();
        let filter = RoomFilter::new().with_space();
        let results = mgr.search(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Open");
    }

    #[test]
    fn search_by_property() {
        let mut mgr = RoomManager::new();
        let id1 = mgr.create("A", 4);
        mgr.get_mut(&id1).unwrap().set_property("mode", "ctf");
        let _id2 = mgr.create("B", 4);
        let filter = RoomFilter::new().with_property("mode", "ctf");
        let results = mgr.search(&filter);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn hidden_rooms_excluded_from_search() {
        let mut mgr = RoomManager::new();
        let _id = mgr.create_with_visibility("Secret", 4, RoomVisibility::Hidden);
        let filter = RoomFilter::new();
        let results = mgr.search(&filter);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn list_public_only() {
        let mut mgr = RoomManager::new();
        let _pub = mgr.create("Pub", 4);
        let _priv = mgr.create_with_visibility("Priv", 4, RoomVisibility::Private);
        let public = mgr.list_public();
        assert_eq!(public.len(), 1);
    }

    #[test]
    fn drain_events() {
        let mut room = Room::new("r1", "Arena", 4);
        room.join("alice").unwrap();
        let events = room.drain_events();
        assert_eq!(events.len(), 1);
        assert!(room.events().is_empty());
    }

    #[test]
    fn display_room() {
        let room = Room::new("r1", "Arena", 4);
        assert!(room.to_string().contains("Arena"));
        assert!(room.to_string().contains("Public"));
    }

    #[test]
    fn destroy_room() {
        let mut mgr = RoomManager::new();
        let id = mgr.create("Arena", 4);
        mgr.destroy(&id).unwrap();
        assert_eq!(mgr.room_count(), 0);
    }

    #[test]
    fn lock_unlock() {
        let mut room = Room::new("r1", "Arena", 4);
        room.lock();
        assert!(room.locked);
        room.unlock();
        assert!(!room.locked);
    }
}
