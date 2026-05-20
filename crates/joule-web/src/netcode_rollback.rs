//! Netcode rollback — rollback/replay networking (GGPO-style).
//!
//! Replaces GGPO/Rollcall with a pure-Rust rollback manager. Saves game state
//! per frame, rolls back to a past frame when corrected remote inputs arrive,
//! replays forward with the correct inputs, tracks frame advantage, validates
//! checksums, and supports sync-test mode for determinism verification.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Rollback engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackError {
    /// Frame is beyond the rollback window.
    FrameTooOld { frame: u64, oldest: u64 },
    /// No saved state for requested frame.
    NoSavedState { frame: u64 },
    /// Max rollback depth exceeded.
    MaxRollbackExceeded { requested: u32, max: u32 },
    /// Checksum mismatch between states.
    ChecksumMismatch { frame: u64, expected: u64, actual: u64 },
    /// Sync test failed: local and remote diverged.
    SyncTestFailed { frame: u64, local_hash: u64, remote_hash: u64 },
    /// Input not available for frame.
    MissingInput { frame: u64, player: u32 },
}

impl fmt::Display for RollbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameTooOld { frame, oldest } => {
                write!(f, "frame {frame} too old, oldest is {oldest}")
            }
            Self::NoSavedState { frame } => write!(f, "no saved state for frame {frame}"),
            Self::MaxRollbackExceeded { requested, max } => {
                write!(f, "rollback of {requested} frames exceeds max {max}")
            }
            Self::ChecksumMismatch { frame, expected, actual } => {
                write!(f, "checksum mismatch at frame {frame}: expected {expected:#x}, got {actual:#x}")
            }
            Self::SyncTestFailed { frame, local_hash, remote_hash } => {
                write!(f, "sync test failed at frame {frame}: local={local_hash:#x} remote={remote_hash:#x}")
            }
            Self::MissingInput { frame, player } => {
                write!(f, "missing input for player {player} at frame {frame}")
            }
        }
    }
}

impl std::error::Error for RollbackError {}

// ── Game State ──────────────────────────────────────────────────

/// A serialisable game state snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct GameStateSnapshot {
    pub frame: u64,
    pub data: Vec<u8>,
    pub checksum: u64,
}

impl GameStateSnapshot {
    pub fn new(frame: u64, data: Vec<u8>) -> Self {
        let checksum = Self::compute_checksum(&data);
        Self { frame, data, checksum }
    }

    /// Simple FNV-1a hash for checksum.
    fn compute_checksum(data: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    pub fn verify_checksum(&self) -> bool {
        Self::compute_checksum(&self.data) == self.checksum
    }
}

// ── Player Input ────────────────────────────────────────────────

/// Input for a single player at a specific frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerInput {
    pub player_id: u32,
    pub frame: u64,
    pub buttons: u32,
    pub stick_x: i16,
    pub stick_y: i16,
}

impl PlayerInput {
    pub fn new(player_id: u32, frame: u64, buttons: u32) -> Self {
        Self { player_id, frame, buttons, stick_x: 0, stick_y: 0 }
    }

    pub fn with_stick(mut self, x: i16, y: i16) -> Self {
        self.stick_x = x;
        self.stick_y = y;
        self
    }

    /// Predict input by repeating the previous input.
    pub fn predict_next(&self) -> Self {
        Self {
            player_id: self.player_id,
            frame: self.frame + 1,
            buttons: self.buttons,
            stick_x: self.stick_x,
            stick_y: self.stick_y,
        }
    }
}

impl fmt::Display for PlayerInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Input(p{}, frame={}, btn={:#06x})", self.player_id, self.frame, self.buttons)
    }
}

// ── Rollback Config ─────────────────────────────────────────────

/// Configuration for rollback networking.
#[derive(Debug, Clone)]
pub struct RollbackConfig {
    pub max_rollback_frames: u32,
    pub input_delay: u32,
    pub max_players: u32,
    pub sync_test_mode: bool,
    pub state_history_size: usize,
}

impl RollbackConfig {
    pub fn new() -> Self {
        Self {
            max_rollback_frames: 8,
            input_delay: 2,
            max_players: 2,
            sync_test_mode: false,
            state_history_size: 64,
        }
    }

    pub fn with_max_rollback(mut self, frames: u32) -> Self {
        self.max_rollback_frames = frames;
        self
    }

    pub fn with_input_delay(mut self, delay: u32) -> Self {
        self.input_delay = delay;
        self
    }

    pub fn with_players(mut self, n: u32) -> Self {
        self.max_players = n;
        self
    }

    pub fn with_sync_test(mut self, enable: bool) -> Self {
        self.sync_test_mode = enable;
        self
    }
}

impl Default for RollbackConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── Frame Advantage ─────────────────────────────────────────────

/// Tracks frame advantage between local and remote players.
#[derive(Debug, Clone)]
pub struct FrameAdvantage {
    pub local_frame: u64,
    pub remote_frame: u64,
}

impl FrameAdvantage {
    pub fn new() -> Self {
        Self { local_frame: 0, remote_frame: 0 }
    }

    pub fn update(&mut self, local: u64, remote: u64) {
        self.local_frame = local;
        self.remote_frame = remote;
    }

    /// Positive = local is ahead, negative = local is behind.
    pub fn advantage(&self) -> i64 {
        self.local_frame as i64 - self.remote_frame as i64
    }

    /// Whether the local side should wait (too far ahead).
    pub fn should_wait(&self, max_advantage: i64) -> bool {
        self.advantage() > max_advantage
    }
}

impl Default for FrameAdvantage {
    fn default() -> Self {
        Self::new()
    }
}

// ── Rollback Stats ──────────────────────────────────────────────

/// Runtime statistics for the rollback system.
#[derive(Debug, Clone, Default)]
pub struct RollbackStats {
    pub total_rollbacks: u64,
    pub total_frames_replayed: u64,
    pub max_rollback_depth: u32,
    pub checksum_failures: u64,
    pub predictions_correct: u64,
    pub predictions_wrong: u64,
}

impl RollbackStats {
    pub fn prediction_accuracy(&self) -> f64 {
        let total = self.predictions_correct + self.predictions_wrong;
        if total == 0 {
            1.0
        } else {
            self.predictions_correct as f64 / total as f64
        }
    }
}

impl fmt::Display for RollbackStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rollbacks: {}, Replayed: {}, MaxDepth: {}, Accuracy: {:.1}%",
            self.total_rollbacks,
            self.total_frames_replayed,
            self.max_rollback_depth,
            self.prediction_accuracy() * 100.0
        )
    }
}

// ── Rollback Manager ────────────────────────────────────────────

/// Core rollback/replay manager.
#[derive(Debug)]
pub struct RollbackManager {
    config: RollbackConfig,
    current_frame: u64,
    state_history: HashMap<u64, GameStateSnapshot>,
    /// Confirmed inputs: (frame, player_id) -> input.
    confirmed_inputs: HashMap<(u64, u32), PlayerInput>,
    /// Predicted inputs used in lieu of real ones.
    predicted_inputs: HashMap<(u64, u32), PlayerInput>,
    frame_advantage: FrameAdvantage,
    stats: RollbackStats,
    /// For sync test: remote checksums.
    remote_checksums: HashMap<u64, u64>,
}

impl RollbackManager {
    pub fn new(config: RollbackConfig) -> Self {
        Self {
            config,
            current_frame: 0,
            state_history: HashMap::new(),
            confirmed_inputs: HashMap::new(),
            predicted_inputs: HashMap::new(),
            frame_advantage: FrameAdvantage::new(),
            stats: RollbackStats::default(),
            remote_checksums: HashMap::new(),
        }
    }

    /// Save the current game state.
    pub fn save_state(&mut self, state: GameStateSnapshot) {
        // Evict old states beyond history size.
        if self.state_history.len() >= self.config.state_history_size {
            let oldest = self.current_frame.saturating_sub(self.config.state_history_size as u64);
            self.state_history.retain(|&frame, _| frame >= oldest);
        }
        self.state_history.insert(state.frame, state);
    }

    /// Load a saved state.
    pub fn load_state(&self, frame: u64) -> Result<&GameStateSnapshot, RollbackError> {
        self.state_history.get(&frame).ok_or(RollbackError::NoSavedState { frame })
    }

    /// Add a confirmed remote input.
    pub fn add_confirmed_input(&mut self, input: PlayerInput) {
        let key = (input.frame, input.player_id);
        // Check if a prediction existed and was correct.
        if let Some(predicted) = self.predicted_inputs.get(&key) {
            if predicted.buttons == input.buttons
                && predicted.stick_x == input.stick_x
                && predicted.stick_y == input.stick_y
            {
                self.stats.predictions_correct += 1;
            } else {
                self.stats.predictions_wrong += 1;
            }
        }
        self.confirmed_inputs.insert(key, input);
    }

    /// Add a predicted input for a remote player.
    pub fn add_predicted_input(&mut self, input: PlayerInput) {
        let key = (input.frame, input.player_id);
        self.predicted_inputs.insert(key, input);
    }

    /// Get input for a player at a frame (confirmed or predicted).
    pub fn get_input(&self, frame: u64, player_id: u32) -> Result<&PlayerInput, RollbackError> {
        let key = (frame, player_id);
        self.confirmed_inputs
            .get(&key)
            .or_else(|| self.predicted_inputs.get(&key))
            .ok_or(RollbackError::MissingInput { frame, player: player_id })
    }

    /// Determine if a rollback is needed: returns the earliest frame that has
    /// a confirmed input replacing a (potentially wrong) prediction.
    pub fn check_rollback_needed(&self) -> Option<u64> {
        let mut earliest_mismatch: Option<u64> = None;
        for (&(frame, player_id), confirmed) in &self.confirmed_inputs {
            if frame >= self.current_frame {
                continue;
            }
            if let Some(predicted) = self.predicted_inputs.get(&(frame, player_id)) {
                if predicted.buttons != confirmed.buttons
                    || predicted.stick_x != confirmed.stick_x
                    || predicted.stick_y != confirmed.stick_y
                {
                    match earliest_mismatch {
                        Some(f) if frame < f => earliest_mismatch = Some(frame),
                        None => earliest_mismatch = Some(frame),
                        _ => {}
                    }
                }
            }
        }
        earliest_mismatch
    }

    /// Perform a rollback to the given frame. Returns the state at that frame
    /// and the number of frames that need replaying.
    pub fn rollback_to(
        &mut self,
        target_frame: u64,
    ) -> Result<(&GameStateSnapshot, u32), RollbackError> {
        let depth = self.current_frame.saturating_sub(target_frame) as u32;
        if depth > self.config.max_rollback_frames {
            return Err(RollbackError::MaxRollbackExceeded {
                requested: depth,
                max: self.config.max_rollback_frames,
            });
        }
        let state = self.state_history.get(&target_frame).ok_or(RollbackError::NoSavedState {
            frame: target_frame,
        })?;

        // Update stats.
        self.stats.total_rollbacks += 1;
        self.stats.total_frames_replayed += depth as u64;
        if depth > self.stats.max_rollback_depth {
            self.stats.max_rollback_depth = depth;
        }

        Ok((state, depth))
    }

    /// Advance to the next frame.
    pub fn advance_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Validate checksum for a frame.
    pub fn validate_checksum(&self, frame: u64, expected: u64) -> Result<(), RollbackError> {
        if let Some(state) = self.state_history.get(&frame) {
            if state.checksum != expected {
                return Err(RollbackError::ChecksumMismatch {
                    frame,
                    expected,
                    actual: state.checksum,
                });
            }
        }
        Ok(())
    }

    /// Sync-test: compare local checksum against remote.
    pub fn sync_test(&mut self, frame: u64, remote_checksum: u64) -> Result<(), RollbackError> {
        self.remote_checksums.insert(frame, remote_checksum);
        if let Some(state) = self.state_history.get(&frame) {
            if state.checksum != remote_checksum {
                return Err(RollbackError::SyncTestFailed {
                    frame,
                    local_hash: state.checksum,
                    remote_hash: remote_checksum,
                });
            }
        }
        Ok(())
    }

    pub fn update_frame_advantage(&mut self, local: u64, remote: u64) {
        self.frame_advantage.update(local, remote);
    }

    pub fn current_frame(&self) -> u64 {
        self.current_frame
    }

    pub fn frame_advantage(&self) -> &FrameAdvantage {
        &self.frame_advantage
    }

    pub fn stats(&self) -> &RollbackStats {
        &self.stats
    }

    pub fn config(&self) -> &RollbackConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(frame: u64, data: &[u8]) -> GameStateSnapshot {
        GameStateSnapshot::new(frame, data.to_vec())
    }

    #[test]
    fn state_checksum_verify() {
        let state = make_state(0, b"hello world");
        assert!(state.verify_checksum());
    }

    #[test]
    fn state_checksum_differs() {
        let a = make_state(0, b"hello");
        let b = make_state(0, b"world");
        assert_ne!(a.checksum, b.checksum);
    }

    #[test]
    fn player_input_predict_next() {
        let input = PlayerInput::new(0, 5, 0x01).with_stick(100, -50);
        let next = input.predict_next();
        assert_eq!(next.frame, 6);
        assert_eq!(next.buttons, 0x01);
        assert_eq!(next.stick_x, 100);
    }

    #[test]
    fn frame_advantage_positive() {
        let mut fa = FrameAdvantage::new();
        fa.update(10, 7);
        assert_eq!(fa.advantage(), 3);
    }

    #[test]
    fn frame_advantage_negative() {
        let mut fa = FrameAdvantage::new();
        fa.update(5, 8);
        assert_eq!(fa.advantage(), -3);
    }

    #[test]
    fn frame_advantage_should_wait() {
        let mut fa = FrameAdvantage::new();
        fa.update(15, 10);
        assert!(fa.should_wait(3));
        assert!(!fa.should_wait(5));
    }

    #[test]
    fn manager_save_and_load() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        mgr.save_state(make_state(0, b"frame0"));
        let loaded = mgr.load_state(0).unwrap();
        assert_eq!(loaded.frame, 0);
    }

    #[test]
    fn manager_load_missing() {
        let mgr = RollbackManager::new(RollbackConfig::new());
        let err = mgr.load_state(99).unwrap_err();
        assert_eq!(err, RollbackError::NoSavedState { frame: 99 });
    }

    #[test]
    fn manager_add_confirmed_input() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        mgr.add_confirmed_input(PlayerInput::new(0, 1, 0x01));
        let input = mgr.get_input(1, 0).unwrap();
        assert_eq!(input.buttons, 0x01);
    }

    #[test]
    fn manager_predicted_input_fallback() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        mgr.add_predicted_input(PlayerInput::new(1, 5, 0x02));
        let input = mgr.get_input(5, 1).unwrap();
        assert_eq!(input.buttons, 0x02);
    }

    #[test]
    fn manager_confirmed_overrides_predicted() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        mgr.add_predicted_input(PlayerInput::new(0, 3, 0x01));
        mgr.add_confirmed_input(PlayerInput::new(0, 3, 0x02));
        let input = mgr.get_input(3, 0).unwrap();
        assert_eq!(input.buttons, 0x02);
    }

    #[test]
    fn manager_rollback_to() {
        let config = RollbackConfig::new().with_max_rollback(8);
        let mut mgr = RollbackManager::new(config);
        mgr.save_state(make_state(5, b"at5"));
        for _ in 0..10 {
            mgr.advance_frame();
        }
        let (state, depth) = mgr.rollback_to(5).unwrap();
        assert_eq!(state.frame, 5);
        assert_eq!(depth, 5);
    }

    #[test]
    fn manager_rollback_exceeds_max() {
        let config = RollbackConfig::new().with_max_rollback(3);
        let mut mgr = RollbackManager::new(config);
        mgr.save_state(make_state(0, b"at0"));
        for _ in 0..10 {
            mgr.advance_frame();
        }
        let err = mgr.rollback_to(0).unwrap_err();
        matches!(err, RollbackError::MaxRollbackExceeded { .. });
    }

    #[test]
    fn manager_validate_checksum_ok() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        let state = make_state(1, b"test");
        let checksum = state.checksum;
        mgr.save_state(state);
        assert!(mgr.validate_checksum(1, checksum).is_ok());
    }

    #[test]
    fn manager_validate_checksum_fail() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        mgr.save_state(make_state(1, b"test"));
        let err = mgr.validate_checksum(1, 0xDEAD).unwrap_err();
        matches!(err, RollbackError::ChecksumMismatch { .. });
    }

    #[test]
    fn manager_sync_test_pass() {
        let mut mgr = RollbackManager::new(RollbackConfig::new().with_sync_test(true));
        let state = make_state(1, b"sync");
        let checksum = state.checksum;
        mgr.save_state(state);
        assert!(mgr.sync_test(1, checksum).is_ok());
    }

    #[test]
    fn manager_sync_test_fail() {
        let mut mgr = RollbackManager::new(RollbackConfig::new().with_sync_test(true));
        mgr.save_state(make_state(1, b"sync"));
        let err = mgr.sync_test(1, 0xBAD).unwrap_err();
        matches!(err, RollbackError::SyncTestFailed { .. });
    }

    #[test]
    fn stats_prediction_accuracy() {
        let mut mgr = RollbackManager::new(RollbackConfig::new());
        // Add predicted then confirm matching.
        mgr.add_predicted_input(PlayerInput::new(0, 1, 0x01));
        mgr.add_confirmed_input(PlayerInput::new(0, 1, 0x01));
        // Add predicted then confirm mismatched.
        mgr.add_predicted_input(PlayerInput::new(0, 2, 0x01));
        mgr.add_confirmed_input(PlayerInput::new(0, 2, 0x02));
        assert!((mgr.stats().prediction_accuracy() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn stats_display() {
        let stats = RollbackStats {
            total_rollbacks: 5,
            total_frames_replayed: 20,
            max_rollback_depth: 4,
            predictions_correct: 90,
            predictions_wrong: 10,
            ..Default::default()
        };
        let s = format!("{stats}");
        assert!(s.contains("Rollbacks: 5"));
        assert!(s.contains("90.0%"));
    }
}
