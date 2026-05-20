//! Game lobby management — create, join, leave, ready-up, kick/ban, search.
//!
//! Replaces lobby.js / Photon-lobby with pure Rust.
//! Lobby lifecycle state machine (Waiting → Countdown → Starting → InGame),
//! password protection, player ready states, lobby chat, search by name
//! or game mode, host migration on leave, and ban list enforcement.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LobbyError {
    LobbyNotFound(String),
    LobbyFull(String),
    PlayerAlreadyInLobby(String),
    PlayerNotInLobby(String),
    NotHost(String),
    PlayerBanned(String),
    WrongPassword,
    InvalidState(String),
    DuplicateLobby(String),
    LobbyLocked(String),
}

impl fmt::Display for LobbyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LobbyNotFound(id) => write!(f, "lobby not found: {id}"),
            Self::LobbyFull(id) => write!(f, "lobby full: {id}"),
            Self::PlayerAlreadyInLobby(p) => write!(f, "player already in lobby: {p}"),
            Self::PlayerNotInLobby(p) => write!(f, "player not in lobby: {p}"),
            Self::NotHost(p) => write!(f, "not the host: {p}"),
            Self::PlayerBanned(p) => write!(f, "player banned: {p}"),
            Self::WrongPassword => write!(f, "incorrect lobby password"),
            Self::InvalidState(s) => write!(f, "invalid state transition: {s}"),
            Self::DuplicateLobby(id) => write!(f, "duplicate lobby: {id}"),
            Self::LobbyLocked(id) => write!(f, "lobby locked: {id}"),
        }
    }
}

impl std::error::Error for LobbyError {}

// ── Lobby State ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LobbyState {
    Waiting,
    Countdown,
    Starting,
    InGame,
    Closed,
}

impl fmt::Display for LobbyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Waiting => write!(f, "Waiting"),
            Self::Countdown => write!(f, "Countdown"),
            Self::Starting => write!(f, "Starting"),
            Self::InGame => write!(f, "InGame"),
            Self::Closed => write!(f, "Closed"),
        }
    }
}

impl LobbyState {
    pub fn can_transition_to(self, next: LobbyState) -> bool {
        matches!(
            (self, next),
            (Self::Waiting, Self::Countdown)
                | (Self::Countdown, Self::Starting)
                | (Self::Countdown, Self::Waiting)
                | (Self::Starting, Self::InGame)
                | (Self::InGame, Self::Closed)
                | (Self::Waiting, Self::Closed)
        )
    }
}

// ── Chat Message ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub sender: String,
    pub text: String,
    pub timestamp_ms: u64,
}

// ── Lobby Settings ──────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LobbySettings {
    pub game_mode: String,
    pub map: String,
    pub password: Option<String>,
    pub custom: HashMap<String, String>,
}

impl LobbySettings {
    pub fn new(game_mode: &str, map: &str) -> Self {
        Self {
            game_mode: game_mode.to_string(),
            map: map.to_string(),
            password: None,
            custom: HashMap::new(),
        }
    }

    pub fn with_password(mut self, pw: &str) -> Self {
        self.password = Some(pw.to_string());
        self
    }

    pub fn with_custom(mut self, key: &str, value: &str) -> Self {
        self.custom.insert(key.to_string(), value.to_string());
        self
    }
}

// ── Lobby ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Lobby {
    pub id: String,
    pub name: String,
    pub host: String,
    pub max_players: usize,
    pub settings: LobbySettings,
    pub state: LobbyState,
    players: Vec<String>,
    ready: HashSet<String>,
    banned: HashSet<String>,
    chat: Vec<ChatMessage>,
    chat_limit: usize,
    locked: bool,
}

impl fmt::Display for Lobby {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Lobby({}, {}/{}, {})",
            self.name,
            self.players.len(),
            self.max_players,
            self.state
        )
    }
}

impl Lobby {
    pub fn new(id: &str, name: &str, host: &str, max_players: usize, settings: LobbySettings) -> Self {
        let mut players = Vec::new();
        players.push(host.to_string());
        let mut ready = HashSet::new();
        ready.insert(host.to_string());
        Self {
            id: id.to_string(),
            name: name.to_string(),
            host: host.to_string(),
            max_players,
            settings,
            state: LobbyState::Waiting,
            players,
            ready,
            banned: HashSet::new(),
            chat: Vec::new(),
            chat_limit: 200,
            locked: false,
        }
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn players(&self) -> &[String] {
        &self.players
    }

    pub fn is_full(&self) -> bool {
        self.players.len() >= self.max_players
    }

    pub fn is_locked(&self) -> bool {
        self.locked
    }

    pub fn is_password_protected(&self) -> bool {
        self.settings.password.is_some()
    }

    pub fn set_locked(&mut self, locked: bool) {
        self.locked = locked;
    }

    pub fn join(&mut self, player: &str, password: Option<&str>) -> Result<(), LobbyError> {
        if self.state != LobbyState::Waiting {
            return Err(LobbyError::InvalidState("can only join in Waiting state".into()));
        }
        if self.locked {
            return Err(LobbyError::LobbyLocked(self.id.clone()));
        }
        if self.banned.contains(player) {
            return Err(LobbyError::PlayerBanned(player.to_string()));
        }
        if self.players.contains(&player.to_string()) {
            return Err(LobbyError::PlayerAlreadyInLobby(player.to_string()));
        }
        if self.is_full() {
            return Err(LobbyError::LobbyFull(self.id.clone()));
        }
        if let Some(ref pw) = self.settings.password {
            match password {
                Some(given) if given == pw => {}
                _ => return Err(LobbyError::WrongPassword),
            }
        }
        self.players.push(player.to_string());
        Ok(())
    }

    pub fn leave(&mut self, player: &str) -> Result<(), LobbyError> {
        let idx = self
            .players
            .iter()
            .position(|p| p == player)
            .ok_or_else(|| LobbyError::PlayerNotInLobby(player.to_string()))?;
        self.players.remove(idx);
        self.ready.remove(player);
        if player == self.host && !self.players.is_empty() {
            self.host = self.players[0].clone();
        }
        Ok(())
    }

    pub fn set_ready(&mut self, player: &str, is_ready: bool) -> Result<(), LobbyError> {
        if !self.players.contains(&player.to_string()) {
            return Err(LobbyError::PlayerNotInLobby(player.to_string()));
        }
        if is_ready {
            self.ready.insert(player.to_string());
        } else {
            self.ready.remove(player);
        }
        Ok(())
    }

    pub fn is_ready(&self, player: &str) -> bool {
        self.ready.contains(player)
    }

    pub fn all_ready(&self) -> bool {
        self.players.iter().all(|p| self.ready.contains(p))
    }

    pub fn kick(&mut self, requester: &str, target: &str) -> Result<(), LobbyError> {
        if requester != self.host {
            return Err(LobbyError::NotHost(requester.to_string()));
        }
        self.leave(target)
    }

    pub fn ban(&mut self, requester: &str, target: &str) -> Result<(), LobbyError> {
        if requester != self.host {
            return Err(LobbyError::NotHost(requester.to_string()));
        }
        if self.players.contains(&target.to_string()) {
            self.leave(target)?;
        }
        self.banned.insert(target.to_string());
        Ok(())
    }

    pub fn is_banned(&self, player: &str) -> bool {
        self.banned.contains(player)
    }

    pub fn transition(&mut self, next: LobbyState) -> Result<(), LobbyError> {
        if !self.state.can_transition_to(next) {
            return Err(LobbyError::InvalidState(format!(
                "{} -> {}",
                self.state, next
            )));
        }
        self.state = next;
        Ok(())
    }

    pub fn send_chat(&mut self, sender: &str, text: &str, timestamp_ms: u64) -> Result<(), LobbyError> {
        if !self.players.contains(&sender.to_string()) {
            return Err(LobbyError::PlayerNotInLobby(sender.to_string()));
        }
        self.chat.push(ChatMessage {
            sender: sender.to_string(),
            text: text.to_string(),
            timestamp_ms,
        });
        if self.chat.len() > self.chat_limit {
            self.chat.remove(0);
        }
        Ok(())
    }

    pub fn chat_history(&self) -> &[ChatMessage] {
        &self.chat
    }
}

// ── Lobby Manager ───────────────────────────────────────────────

#[derive(Debug)]
pub struct LobbyManager {
    lobbies: HashMap<String, Lobby>,
    next_id: u64,
}

impl LobbyManager {
    pub fn new() -> Self {
        Self {
            lobbies: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn create(
        &mut self,
        name: &str,
        host: &str,
        max_players: usize,
        settings: LobbySettings,
    ) -> Result<String, LobbyError> {
        let id = format!("lobby-{}", self.next_id);
        self.next_id += 1;
        let lobby = Lobby::new(&id, name, host, max_players, settings);
        self.lobbies.insert(id.clone(), lobby);
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Result<&Lobby, LobbyError> {
        self.lobbies
            .get(id)
            .ok_or_else(|| LobbyError::LobbyNotFound(id.to_string()))
    }

    pub fn get_mut(&mut self, id: &str) -> Result<&mut Lobby, LobbyError> {
        self.lobbies
            .get_mut(id)
            .ok_or_else(|| LobbyError::LobbyNotFound(id.to_string()))
    }

    pub fn close(&mut self, id: &str) -> Result<(), LobbyError> {
        if self.lobbies.remove(id).is_none() {
            return Err(LobbyError::LobbyNotFound(id.to_string()));
        }
        Ok(())
    }

    pub fn lobby_count(&self) -> usize {
        self.lobbies.len()
    }

    pub fn search_by_name(&self, query: &str) -> Vec<&Lobby> {
        let q = query.to_lowercase();
        self.lobbies
            .values()
            .filter(|l| l.name.to_lowercase().contains(&q))
            .collect()
    }

    pub fn search_by_game_mode(&self, mode: &str) -> Vec<&Lobby> {
        self.lobbies
            .values()
            .filter(|l| l.settings.game_mode == mode)
            .collect()
    }

    pub fn available_lobbies(&self) -> Vec<&Lobby> {
        self.lobbies
            .values()
            .filter(|l| l.state == LobbyState::Waiting && !l.is_full() && !l.is_locked())
            .collect()
    }
}

impl Default for LobbyManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> LobbySettings {
        LobbySettings::new("deathmatch", "dust2")
    }

    #[test]
    fn create_lobby() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("My Lobby", "alice", 4, settings()).unwrap();
        let lobby = mgr.get(&id).unwrap();
        assert_eq!(lobby.name, "My Lobby");
        assert_eq!(lobby.host, "alice");
        assert_eq!(lobby.player_count(), 1);
    }

    #[test]
    fn join_and_leave() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 4, settings()).unwrap();
        mgr.get_mut(&id).unwrap().join("bob", None).unwrap();
        assert_eq!(mgr.get(&id).unwrap().player_count(), 2);
        mgr.get_mut(&id).unwrap().leave("bob").unwrap();
        assert_eq!(mgr.get(&id).unwrap().player_count(), 1);
    }

    #[test]
    fn lobby_full_error() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 2, settings()).unwrap();
        mgr.get_mut(&id).unwrap().join("bob", None).unwrap();
        let err = mgr.get_mut(&id).unwrap().join("carol", None).unwrap_err();
        assert!(matches!(err, LobbyError::LobbyFull(_)));
    }

    #[test]
    fn password_protection() {
        let mut mgr = LobbyManager::new();
        let s = settings().with_password("secret");
        let id = mgr.create("L", "alice", 4, s).unwrap();
        let err = mgr.get_mut(&id).unwrap().join("bob", None).unwrap_err();
        assert!(matches!(err, LobbyError::WrongPassword));
        let err2 = mgr.get_mut(&id).unwrap().join("bob", Some("wrong")).unwrap_err();
        assert!(matches!(err2, LobbyError::WrongPassword));
        mgr.get_mut(&id).unwrap().join("bob", Some("secret")).unwrap();
        assert_eq!(mgr.get(&id).unwrap().player_count(), 2);
    }

    #[test]
    fn ready_state() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 4, settings()).unwrap();
        mgr.get_mut(&id).unwrap().join("bob", None).unwrap();
        assert!(!mgr.get(&id).unwrap().all_ready());
        mgr.get_mut(&id).unwrap().set_ready("bob", true).unwrap();
        assert!(mgr.get(&id).unwrap().all_ready());
        mgr.get_mut(&id).unwrap().set_ready("bob", false).unwrap();
        assert!(!mgr.get(&id).unwrap().all_ready());
    }

    #[test]
    fn kick_player() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 4, settings()).unwrap();
        mgr.get_mut(&id).unwrap().join("bob", None).unwrap();
        mgr.get_mut(&id).unwrap().kick("alice", "bob").unwrap();
        assert_eq!(mgr.get(&id).unwrap().player_count(), 1);
    }

    #[test]
    fn kick_requires_host() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 4, settings()).unwrap();
        mgr.get_mut(&id).unwrap().join("bob", None).unwrap();
        let err = mgr.get_mut(&id).unwrap().kick("bob", "alice").unwrap_err();
        assert!(matches!(err, LobbyError::NotHost(_)));
    }

    #[test]
    fn ban_player() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 4, settings()).unwrap();
        mgr.get_mut(&id).unwrap().join("bob", None).unwrap();
        mgr.get_mut(&id).unwrap().ban("alice", "bob").unwrap();
        assert!(mgr.get(&id).unwrap().is_banned("bob"));
        let err = mgr.get_mut(&id).unwrap().join("bob", None).unwrap_err();
        assert!(matches!(err, LobbyError::PlayerBanned(_)));
    }

    #[test]
    fn state_transitions() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        assert_eq!(lobby.state, LobbyState::Waiting);
        lobby.transition(LobbyState::Countdown).unwrap();
        lobby.transition(LobbyState::Starting).unwrap();
        lobby.transition(LobbyState::InGame).unwrap();
        lobby.transition(LobbyState::Closed).unwrap();
    }

    #[test]
    fn invalid_state_transition() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        let err = lobby.transition(LobbyState::InGame).unwrap_err();
        assert!(matches!(err, LobbyError::InvalidState(_)));
    }

    #[test]
    fn countdown_back_to_waiting() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        lobby.transition(LobbyState::Countdown).unwrap();
        lobby.transition(LobbyState::Waiting).unwrap();
        assert_eq!(lobby.state, LobbyState::Waiting);
    }

    #[test]
    fn host_migration_on_leave() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        lobby.join("bob", None).unwrap();
        lobby.leave("alice").unwrap();
        assert_eq!(lobby.host, "bob");
    }

    #[test]
    fn lobby_chat() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        lobby.send_chat("alice", "hello", 1000).unwrap();
        assert_eq!(lobby.chat_history().len(), 1);
        assert_eq!(lobby.chat_history()[0].text, "hello");
    }

    #[test]
    fn chat_requires_membership() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        let err = lobby.send_chat("bob", "hi", 1000).unwrap_err();
        assert!(matches!(err, LobbyError::PlayerNotInLobby(_)));
    }

    #[test]
    fn chat_buffer_limit() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        lobby.chat_limit = 3;
        for i in 0..5 {
            lobby.send_chat("alice", &format!("msg{i}"), i as u64).unwrap();
        }
        assert_eq!(lobby.chat_history().len(), 3);
        assert_eq!(lobby.chat_history()[0].text, "msg2");
    }

    #[test]
    fn search_by_name() {
        let mut mgr = LobbyManager::new();
        mgr.create("Pro Match", "a", 4, settings()).unwrap();
        mgr.create("Casual Fun", "b", 4, settings()).unwrap();
        let results = mgr.search_by_name("pro");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Pro Match");
    }

    #[test]
    fn search_by_game_mode() {
        let mut mgr = LobbyManager::new();
        mgr.create("L1", "a", 4, settings()).unwrap();
        mgr.create("L2", "b", 4, LobbySettings::new("ctf", "map1")).unwrap();
        let results = mgr.search_by_game_mode("ctf");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn available_lobbies_filter() {
        let mut mgr = LobbyManager::new();
        let id1 = mgr.create("Open", "a", 4, settings()).unwrap();
        let id2 = mgr.create("Full", "b", 1, settings()).unwrap();
        let _ = id2;
        mgr.get_mut(&id1).unwrap().set_locked(false);
        let avail = mgr.available_lobbies();
        assert_eq!(avail.len(), 1);
        assert_eq!(avail[0].name, "Open");
    }

    #[test]
    fn locked_lobby_prevents_join() {
        let mut lobby = Lobby::new("1", "L", "alice", 4, settings());
        lobby.set_locked(true);
        let err = lobby.join("bob", None).unwrap_err();
        assert!(matches!(err, LobbyError::LobbyLocked(_)));
    }

    #[test]
    fn display_impls() {
        let lobby = Lobby::new("1", "Test", "alice", 4, settings());
        assert!(lobby.to_string().contains("Test"));
        assert_eq!(LobbyState::Waiting.to_string(), "Waiting");
    }

    #[test]
    fn close_lobby() {
        let mut mgr = LobbyManager::new();
        let id = mgr.create("L", "alice", 4, settings()).unwrap();
        mgr.close(&id).unwrap();
        assert_eq!(mgr.lobby_count(), 0);
    }
}
