//! Game session relay and lifecycle management — create, pause, migrate, close.
//!
//! Replaces session.js / Photon-realtime with pure Rust.
//! GameSession with players and state machine (Active → Paused → Ending → Closed),
//! SessionRelay managing active sessions, disconnect/reconnect handling,
//! session timeout, host migration, and session statistics.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    SessionNotFound(String),
    PlayerNotInSession(String),
    PlayerAlreadyInSession(String),
    InvalidTransition(String),
    SessionFull(String),
    NotHost(String),
    SessionClosed(String),
    AlreadyConnected(String),
    NotDisconnected(String),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SessionNotFound(id) => write!(f, "session not found: {id}"),
            Self::PlayerNotInSession(p) => write!(f, "player not in session: {p}"),
            Self::PlayerAlreadyInSession(p) => write!(f, "player already in session: {p}"),
            Self::InvalidTransition(msg) => write!(f, "invalid transition: {msg}"),
            Self::SessionFull(id) => write!(f, "session full: {id}"),
            Self::NotHost(p) => write!(f, "not the host: {p}"),
            Self::SessionClosed(id) => write!(f, "session closed: {id}"),
            Self::AlreadyConnected(p) => write!(f, "already connected: {p}"),
            Self::NotDisconnected(p) => write!(f, "not disconnected: {p}"),
        }
    }
}

impl std::error::Error for SessionError {}

// ── Session State ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    Active,
    Paused,
    Ending,
    Closed,
}

impl SessionState {
    pub fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Active, Self::Paused)
                | (Self::Active, Self::Ending)
                | (Self::Paused, Self::Active)
                | (Self::Paused, Self::Ending)
                | (Self::Ending, Self::Closed)
        )
    }
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Paused => write!(f, "Paused"),
            Self::Ending => write!(f, "Ending"),
            Self::Closed => write!(f, "Closed"),
        }
    }
}

// ── Player Status ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerStatus {
    Connected,
    Disconnected { since_ms: u64 },
}

impl fmt::Display for PlayerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connected => write!(f, "Connected"),
            Self::Disconnected { since_ms } => write!(f, "Disconnected(since {since_ms}ms)"),
        }
    }
}

// ── Session Snapshot ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    pub timestamp_ms: u64,
    pub player_count: usize,
    pub connected_count: usize,
}

// ── Game Session ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GameSession {
    pub id: String,
    pub host: String,
    pub max_players: usize,
    pub state: SessionState,
    pub created_at_ms: u64,
    pub game_mode: String,
    players: HashMap<String, PlayerStatus>,
    reconnect_timeout_ms: u64,
    snapshots: Vec<SessionSnapshot>,
}

impl GameSession {
    pub fn new(id: &str, host: &str, max_players: usize, game_mode: &str, created_at_ms: u64) -> Self {
        let mut players = HashMap::new();
        players.insert(host.to_string(), PlayerStatus::Connected);
        Self {
            id: id.to_string(),
            host: host.to_string(),
            max_players,
            state: SessionState::Active,
            created_at_ms,
            game_mode: game_mode.to_string(),
            players,
            reconnect_timeout_ms: 30_000,
            snapshots: Vec::new(),
        }
    }

    pub fn with_reconnect_timeout(mut self, ms: u64) -> Self {
        self.reconnect_timeout_ms = ms;
        self
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    pub fn connected_count(&self) -> usize {
        self.players.values().filter(|s| matches!(s, PlayerStatus::Connected)).count()
    }

    pub fn player_ids(&self) -> Vec<&str> {
        self.players.keys().map(|s| s.as_str()).collect()
    }

    pub fn player_status(&self, player: &str) -> Option<&PlayerStatus> {
        self.players.get(player)
    }

    pub fn add_player(&mut self, player: &str) -> Result<(), SessionError> {
        if self.state == SessionState::Closed {
            return Err(SessionError::SessionClosed(self.id.clone()));
        }
        if self.players.contains_key(player) {
            return Err(SessionError::PlayerAlreadyInSession(player.to_string()));
        }
        if self.players.len() >= self.max_players {
            return Err(SessionError::SessionFull(self.id.clone()));
        }
        self.players.insert(player.to_string(), PlayerStatus::Connected);
        Ok(())
    }

    pub fn remove_player(&mut self, player: &str) -> Result<(), SessionError> {
        if self.players.remove(player).is_none() {
            return Err(SessionError::PlayerNotInSession(player.to_string()));
        }
        if player == self.host && !self.players.is_empty() {
            // Migrate host to first connected player, or first player.
            self.host = self
                .players
                .iter()
                .find(|(_, s)| matches!(s, PlayerStatus::Connected))
                .or_else(|| self.players.iter().next())
                .map(|(id, _)| id.clone())
                .unwrap();
        }
        Ok(())
    }

    pub fn disconnect(&mut self, player: &str, now_ms: u64) -> Result<(), SessionError> {
        let status = self
            .players
            .get_mut(player)
            .ok_or_else(|| SessionError::PlayerNotInSession(player.to_string()))?;
        if matches!(status, PlayerStatus::Disconnected { .. }) {
            return Err(SessionError::NotDisconnected(player.to_string()));
        }
        *status = PlayerStatus::Disconnected { since_ms: now_ms };
        Ok(())
    }

    pub fn reconnect(&mut self, player: &str) -> Result<(), SessionError> {
        let status = self
            .players
            .get_mut(player)
            .ok_or_else(|| SessionError::PlayerNotInSession(player.to_string()))?;
        if matches!(status, PlayerStatus::Connected) {
            return Err(SessionError::AlreadyConnected(player.to_string()));
        }
        *status = PlayerStatus::Connected;
        Ok(())
    }

    /// Remove players whose disconnect exceeded the timeout.
    pub fn expire_disconnected(&mut self, now_ms: u64) -> Vec<String> {
        let mut expired = Vec::new();
        let timeout = self.reconnect_timeout_ms;
        self.players.retain(|id, status| {
            if let PlayerStatus::Disconnected { since_ms } = status {
                if now_ms.saturating_sub(*since_ms) >= timeout {
                    expired.push(id.clone());
                    return false;
                }
            }
            true
        });
        // Fix host if expired.
        if expired.contains(&self.host) && !self.players.is_empty() {
            self.host = self.players.keys().next().unwrap().clone();
        }
        expired
    }

    pub fn transition(&mut self, next: SessionState) -> Result<(), SessionError> {
        if !self.state.can_transition_to(next) {
            return Err(SessionError::InvalidTransition(format!(
                "{} -> {}",
                self.state, next
            )));
        }
        self.state = next;
        Ok(())
    }

    pub fn migrate_host(&mut self, requester: &str, new_host: &str) -> Result<(), SessionError> {
        if requester != self.host {
            return Err(SessionError::NotHost(requester.to_string()));
        }
        if !self.players.contains_key(new_host) {
            return Err(SessionError::PlayerNotInSession(new_host.to_string()));
        }
        self.host = new_host.to_string();
        Ok(())
    }

    pub fn take_snapshot(&mut self, timestamp_ms: u64) {
        self.snapshots.push(SessionSnapshot {
            timestamp_ms,
            player_count: self.player_count(),
            connected_count: self.connected_count(),
        });
    }

    pub fn snapshots(&self) -> &[SessionSnapshot] {
        &self.snapshots
    }

    pub fn duration_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_at_ms)
    }
}

impl fmt::Display for GameSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Session({}, {}/{} players, {})",
            self.id,
            self.connected_count(),
            self.max_players,
            self.state
        )
    }
}

// ── Session Relay ───────────────────────────────────────────────

#[derive(Debug)]
pub struct SessionRelay {
    sessions: HashMap<String, GameSession>,
    next_id: u64,
}

impl SessionRelay {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn create(
        &mut self,
        host: &str,
        max_players: usize,
        game_mode: &str,
        created_at_ms: u64,
    ) -> String {
        let id = format!("session-{}", self.next_id);
        self.next_id += 1;
        let session = GameSession::new(&id, host, max_players, game_mode, created_at_ms);
        self.sessions.insert(id.clone(), session);
        id
    }

    pub fn get(&self, id: &str) -> Result<&GameSession, SessionError> {
        self.sessions
            .get(id)
            .ok_or_else(|| SessionError::SessionNotFound(id.to_string()))
    }

    pub fn get_mut(&mut self, id: &str) -> Result<&mut GameSession, SessionError> {
        self.sessions
            .get_mut(id)
            .ok_or_else(|| SessionError::SessionNotFound(id.to_string()))
    }

    pub fn active_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state == SessionState::Active)
            .count()
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    pub fn close(&mut self, id: &str) -> Result<(), SessionError> {
        let session = self.get_mut(id)?;
        if session.state != SessionState::Closed {
            // Force close.
            session.state = SessionState::Closed;
        }
        Ok(())
    }

    pub fn remove_closed(&mut self) -> usize {
        let before = self.sessions.len();
        self.sessions.retain(|_, s| s.state != SessionState::Closed);
        before - self.sessions.len()
    }

    pub fn find_by_player(&self, player: &str) -> Option<&GameSession> {
        self.sessions
            .values()
            .find(|s| s.players.contains_key(player))
    }
}

impl Default for SessionRelay {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_session() {
        let mut relay = SessionRelay::new();
        let id = relay.create("alice", 4, "deathmatch", 1000);
        let s = relay.get(&id).unwrap();
        assert_eq!(s.host, "alice");
        assert_eq!(s.player_count(), 1);
        assert_eq!(s.state, SessionState::Active);
    }

    #[test]
    fn add_remove_player() {
        let mut relay = SessionRelay::new();
        let id = relay.create("alice", 4, "dm", 0);
        relay.get_mut(&id).unwrap().add_player("bob").unwrap();
        assert_eq!(relay.get(&id).unwrap().player_count(), 2);
        relay.get_mut(&id).unwrap().remove_player("bob").unwrap();
        assert_eq!(relay.get(&id).unwrap().player_count(), 1);
    }

    #[test]
    fn session_full_error() {
        let mut relay = SessionRelay::new();
        let id = relay.create("alice", 2, "dm", 0);
        relay.get_mut(&id).unwrap().add_player("bob").unwrap();
        let err = relay.get_mut(&id).unwrap().add_player("carol").unwrap_err();
        assert!(matches!(err, SessionError::SessionFull(_)));
    }

    #[test]
    fn state_machine_transitions() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.transition(SessionState::Paused).unwrap();
        s.transition(SessionState::Active).unwrap();
        s.transition(SessionState::Ending).unwrap();
        s.transition(SessionState::Closed).unwrap();
    }

    #[test]
    fn invalid_transition() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        let err = s.transition(SessionState::Closed).unwrap_err();
        assert!(matches!(err, SessionError::InvalidTransition(_)));
    }

    #[test]
    fn disconnect_reconnect() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.add_player("bob").unwrap();
        s.disconnect("bob", 5000).unwrap();
        assert!(matches!(
            s.player_status("bob"),
            Some(PlayerStatus::Disconnected { .. })
        ));
        s.reconnect("bob").unwrap();
        assert!(matches!(s.player_status("bob"), Some(PlayerStatus::Connected)));
    }

    #[test]
    fn expire_disconnected() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0)
            .with_reconnect_timeout(10_000);
        s.add_player("bob").unwrap();
        s.disconnect("bob", 1000).unwrap();
        let expired = s.expire_disconnected(5000);
        assert!(expired.is_empty());
        let expired2 = s.expire_disconnected(12_000);
        assert_eq!(expired2, vec!["bob".to_string()]);
        assert_eq!(s.player_count(), 1);
    }

    #[test]
    fn host_migration_on_leave() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.add_player("bob").unwrap();
        s.remove_player("alice").unwrap();
        assert_eq!(s.host, "bob");
    }

    #[test]
    fn manual_host_migration() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.add_player("bob").unwrap();
        s.migrate_host("alice", "bob").unwrap();
        assert_eq!(s.host, "bob");
    }

    #[test]
    fn migrate_not_host_error() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.add_player("bob").unwrap();
        let err = s.migrate_host("bob", "bob").unwrap_err();
        assert!(matches!(err, SessionError::NotHost(_)));
    }

    #[test]
    fn session_snapshots() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.add_player("bob").unwrap();
        s.take_snapshot(1000);
        s.take_snapshot(2000);
        assert_eq!(s.snapshots().len(), 2);
        assert_eq!(s.snapshots()[0].player_count, 2);
    }

    #[test]
    fn duration() {
        let s = GameSession::new("1", "alice", 4, "dm", 5000);
        assert_eq!(s.duration_ms(8000), 3000);
    }

    #[test]
    fn connected_count() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.add_player("bob").unwrap();
        assert_eq!(s.connected_count(), 2);
        s.disconnect("bob", 100).unwrap();
        assert_eq!(s.connected_count(), 1);
    }

    #[test]
    fn relay_active_count() {
        let mut relay = SessionRelay::new();
        let id1 = relay.create("a", 4, "dm", 0);
        let _id2 = relay.create("b", 4, "dm", 0);
        relay.close(&id1).unwrap();
        assert_eq!(relay.active_count(), 1);
    }

    #[test]
    fn remove_closed_sessions() {
        let mut relay = SessionRelay::new();
        let id1 = relay.create("a", 4, "dm", 0);
        let _id2 = relay.create("b", 4, "dm", 0);
        relay.close(&id1).unwrap();
        let removed = relay.remove_closed();
        assert_eq!(removed, 1);
        assert_eq!(relay.session_count(), 1);
    }

    #[test]
    fn find_by_player() {
        let mut relay = SessionRelay::new();
        let id = relay.create("alice", 4, "dm", 0);
        let found = relay.find_by_player("alice").unwrap();
        assert_eq!(found.id, id);
        assert!(relay.find_by_player("ghost").is_none());
    }

    #[test]
    fn display_session() {
        let s = GameSession::new("s1", "alice", 4, "dm", 0);
        let d = s.to_string();
        assert!(d.contains("s1"));
        assert!(d.contains("Active"));
    }

    #[test]
    fn closed_session_rejects_join() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        s.transition(SessionState::Ending).unwrap();
        s.transition(SessionState::Closed).unwrap();
        let err = s.add_player("bob").unwrap_err();
        assert!(matches!(err, SessionError::SessionClosed(_)));
    }

    #[test]
    fn duplicate_player_error() {
        let mut s = GameSession::new("1", "alice", 4, "dm", 0);
        let err = s.add_player("alice").unwrap_err();
        assert!(matches!(err, SessionError::PlayerAlreadyInSession(_)));
    }
}
