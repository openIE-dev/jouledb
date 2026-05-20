//! Media Session API — metadata, playback state, action handlers, position state.

use std::collections::HashMap;

// ── Artwork ─────────────────────────────────────────────────────

/// A media artwork image reference.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaImage {
    pub src: String,
    pub sizes: Option<String>,
    pub image_type: Option<String>,
}

// ── Metadata ────────────────────────────────────────────────────

/// Media metadata for the current playing item.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaMetadata {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub artwork: Vec<MediaImage>,
}

impl MediaMetadata {
    pub fn new(title: impl Into<String>, artist: impl Into<String>, album: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            artist: artist.into(),
            album: album.into(),
            artwork: Vec::new(),
        }
    }

    /// Add an artwork image.
    pub fn with_artwork(mut self, src: impl Into<String>, sizes: Option<&str>, image_type: Option<&str>) -> Self {
        self.artwork.push(MediaImage {
            src: src.into(),
            sizes: sizes.map(|s| s.to_string()),
            image_type: image_type.map(|s| s.to_string()),
        });
        self
    }
}

// ── Playback State ──────────────────────────────────────────────

/// Playback state of the media session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    None,
    Paused,
    Playing,
}

// ── Actions ─────────────────────────────────────────────────────

/// Media session actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaSessionAction {
    Play,
    Pause,
    SeekBackward,
    SeekForward,
    PreviousTrack,
    NextTrack,
    Stop,
    SeekTo,
    SkipAd,
    HangUp,
    ToggleMicrophone,
    ToggleCamera,
}

/// Details provided with certain actions.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionDetails {
    Seek { offset: f64 },
    SeekTo { position: f64, fast_seek: bool },
    None,
}

/// A recorded action handler invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionRecord {
    pub action: MediaSessionAction,
    pub details: ActionDetails,
}

// ── Position State ──────────────────────────────────────────────

/// Current playback position information.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionState {
    pub duration: f64,
    pub playback_rate: f64,
    pub position: f64,
}

impl PositionState {
    pub fn new(duration: f64, playback_rate: f64, position: f64) -> Self {
        Self {
            duration,
            playback_rate,
            position,
        }
    }

    /// Remaining time in seconds.
    pub fn remaining(&self) -> f64 {
        (self.duration - self.position).max(0.0)
    }

    /// Progress as a fraction (0.0..=1.0).
    pub fn progress(&self) -> f64 {
        if self.duration <= 0.0 {
            return 0.0;
        }
        (self.position / self.duration).clamp(0.0, 1.0)
    }
}

// ── Media Session ───────────────────────────────────────────────

/// Media session — manages metadata, playback state, and action handlers.
#[derive(Debug, Clone)]
pub struct MediaSession {
    pub metadata: Option<MediaMetadata>,
    pub playback_state: PlaybackState,
    pub position_state: Option<PositionState>,
    active: bool,
    registered_actions: HashMap<MediaSessionAction, bool>,
    action_log: Vec<ActionRecord>,
}

impl MediaSession {
    pub fn new() -> Self {
        Self {
            metadata: None,
            playback_state: PlaybackState::None,
            position_state: None,
            active: false,
            registered_actions: HashMap::new(),
            action_log: Vec::new(),
        }
    }

    /// Activate the session.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the session.
    pub fn deactivate(&mut self) {
        self.active = false;
        self.playback_state = PlaybackState::None;
    }

    /// Whether the session is active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Set metadata for the current media.
    pub fn set_metadata(&mut self, metadata: MediaMetadata) {
        self.metadata = Some(metadata);
    }

    /// Clear metadata.
    pub fn clear_metadata(&mut self) {
        self.metadata = None;
    }

    /// Set the playback state.
    pub fn set_playback_state(&mut self, state: PlaybackState) {
        self.playback_state = state;
    }

    /// Set position state.
    pub fn set_position_state(&mut self, position: PositionState) {
        self.position_state = Some(position);
    }

    /// Register an action handler.
    pub fn set_action_handler(&mut self, action: MediaSessionAction) {
        self.registered_actions.insert(action, true);
    }

    /// Remove an action handler.
    pub fn remove_action_handler(&mut self, action: MediaSessionAction) {
        self.registered_actions.remove(&action);
    }

    /// Check if an action handler is registered.
    pub fn has_action_handler(&self, action: MediaSessionAction) -> bool {
        self.registered_actions.contains_key(&action)
    }

    /// Invoke an action. Returns true if the action was handled.
    pub fn invoke_action(&mut self, action: MediaSessionAction, details: ActionDetails) -> bool {
        if !self.active {
            return false;
        }
        if !self.registered_actions.contains_key(&action) {
            return false;
        }
        self.action_log.push(ActionRecord { action, details });
        true
    }

    /// Get all recorded action invocations.
    pub fn action_log(&self) -> &[ActionRecord] {
        &self.action_log
    }

    /// Clear the action log.
    pub fn clear_action_log(&mut self) {
        self.action_log.clear();
    }

    /// Get all registered actions.
    pub fn registered_actions(&self) -> Vec<MediaSessionAction> {
        self.registered_actions.keys().copied().collect()
    }
}

impl Default for MediaSession {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_metadata() {
        let m = MediaMetadata::new("Song", "Artist", "Album")
            .with_artwork("https://example.com/art.jpg", Some("512x512"), Some("image/jpeg"));
        assert_eq!(m.title, "Song");
        assert_eq!(m.artwork.len(), 1);
        assert_eq!(m.artwork[0].sizes.as_deref(), Some("512x512"));
    }

    #[test]
    fn session_activate_deactivate() {
        let mut s = MediaSession::new();
        assert!(!s.is_active());
        s.activate();
        assert!(s.is_active());
        s.deactivate();
        assert!(!s.is_active());
        assert_eq!(s.playback_state, PlaybackState::None);
    }

    #[test]
    fn playback_state_changes() {
        let mut s = MediaSession::new();
        s.set_playback_state(PlaybackState::Playing);
        assert_eq!(s.playback_state, PlaybackState::Playing);
        s.set_playback_state(PlaybackState::Paused);
        assert_eq!(s.playback_state, PlaybackState::Paused);
    }

    #[test]
    fn set_and_clear_metadata() {
        let mut s = MediaSession::new();
        s.set_metadata(MediaMetadata::new("Title", "Art", "Alb"));
        assert!(s.metadata.is_some());
        s.clear_metadata();
        assert!(s.metadata.is_none());
    }

    #[test]
    fn action_handler_registration() {
        let mut s = MediaSession::new();
        s.set_action_handler(MediaSessionAction::Play);
        s.set_action_handler(MediaSessionAction::Pause);
        assert!(s.has_action_handler(MediaSessionAction::Play));
        assert!(!s.has_action_handler(MediaSessionAction::Stop));
        s.remove_action_handler(MediaSessionAction::Play);
        assert!(!s.has_action_handler(MediaSessionAction::Play));
    }

    #[test]
    fn invoke_action() {
        let mut s = MediaSession::new();
        s.activate();
        s.set_action_handler(MediaSessionAction::Play);
        let ok = s.invoke_action(MediaSessionAction::Play, ActionDetails::None);
        assert!(ok);
        assert_eq!(s.action_log().len(), 1);
        assert_eq!(s.action_log()[0].action, MediaSessionAction::Play);
    }

    #[test]
    fn invoke_unregistered_action() {
        let mut s = MediaSession::new();
        s.activate();
        let ok = s.invoke_action(MediaSessionAction::Stop, ActionDetails::None);
        assert!(!ok);
        assert!(s.action_log().is_empty());
    }

    #[test]
    fn invoke_inactive_session() {
        let mut s = MediaSession::new();
        s.set_action_handler(MediaSessionAction::Play);
        let ok = s.invoke_action(MediaSessionAction::Play, ActionDetails::None);
        assert!(!ok);
    }

    #[test]
    fn seek_action_with_details() {
        let mut s = MediaSession::new();
        s.activate();
        s.set_action_handler(MediaSessionAction::SeekBackward);
        s.invoke_action(
            MediaSessionAction::SeekBackward,
            ActionDetails::Seek { offset: 10.0 },
        );
        let rec = &s.action_log()[0];
        assert_eq!(rec.details, ActionDetails::Seek { offset: 10.0 });
    }

    #[test]
    fn position_state() {
        let ps = PositionState::new(300.0, 1.0, 150.0);
        assert!((ps.remaining() - 150.0).abs() < 1e-9);
        assert!((ps.progress() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn position_state_edge_cases() {
        let ps = PositionState::new(0.0, 1.0, 0.0);
        assert!((ps.progress() - 0.0).abs() < 1e-9);
        assert!((ps.remaining() - 0.0).abs() < 1e-9);

        let ps2 = PositionState::new(100.0, 1.0, 200.0);
        assert!((ps2.remaining() - 0.0).abs() < 1e-9);
        assert!((ps2.progress() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn clear_action_log() {
        let mut s = MediaSession::new();
        s.activate();
        s.set_action_handler(MediaSessionAction::Play);
        s.invoke_action(MediaSessionAction::Play, ActionDetails::None);
        assert_eq!(s.action_log().len(), 1);
        s.clear_action_log();
        assert!(s.action_log().is_empty());
    }

    #[test]
    fn multiple_artwork() {
        let m = MediaMetadata::new("Song", "Artist", "Album")
            .with_artwork("small.jpg", Some("96x96"), Some("image/jpeg"))
            .with_artwork("large.jpg", Some("512x512"), Some("image/jpeg"));
        assert_eq!(m.artwork.len(), 2);
    }
}
