//! Network input validation and sanitization — frame integrity, sequencing, rate limiting.
//!
//! Replaces netcode input-validation layers with pure Rust.
//! InputFrame processing, sequence gap detection, duplicate frame
//! detection, action rate limiting, input prediction bounds checking,
//! frame timing validation, configurable validation pipeline,
//! input history for pattern detection, and value sanitization.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputValidateError {
    EmptyFrame,
    InvalidChecksum { expected: u32, actual: u32 },
    PlayerNotRegistered(u64),
    ConfigError(String),
}

impl fmt::Display for InputValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFrame => write!(f, "empty frame"),
            Self::InvalidChecksum { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, got {actual}")
            }
            Self::PlayerNotRegistered(id) => write!(f, "player not registered: {id}"),
            Self::ConfigError(msg) => write!(f, "config error: {msg}"),
        }
    }
}

impl std::error::Error for InputValidateError {}

// ── Input Action ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct InputAction {
    pub action_id: u16,
    pub value: f64,
}

impl InputAction {
    pub fn new(action_id: u16, value: f64) -> Self {
        Self { action_id, value }
    }

    pub fn clamped(&self, min: f64, max: f64) -> Self {
        Self {
            action_id: self.action_id,
            value: self.value.clamp(min, max),
        }
    }
}

impl fmt::Display for InputAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "action(id={}, val={:.3})", self.action_id, self.value)
    }
}

// ── Input Frame ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct InputFrame {
    pub player_id: u64,
    pub sequence: u64,
    pub actions: Vec<InputAction>,
    pub checksum: u32,
    pub timestamp_ms: u64,
}

impl InputFrame {
    pub fn new(player_id: u64, sequence: u64, timestamp_ms: u64) -> Self {
        Self {
            player_id,
            sequence,
            actions: Vec::new(),
            checksum: 0,
            timestamp_ms,
        }
    }

    pub fn with_action(mut self, action: InputAction) -> Self {
        self.actions.push(action);
        self
    }

    pub fn with_checksum(mut self, cs: u32) -> Self {
        self.checksum = cs;
        self
    }

    pub fn compute_checksum(&self) -> u32 {
        let mut hash: u32 = self.sequence as u32;
        hash = hash.wrapping_mul(2654435761u32);
        hash ^= self.player_id as u32;
        for a in &self.actions {
            hash = hash.wrapping_add(a.action_id as u32);
            hash = hash.wrapping_mul(31);
            hash ^= a.value.to_bits() as u32;
        }
        hash
    }

    pub fn is_checksum_valid(&self) -> bool {
        self.checksum == 0 || self.checksum == self.compute_checksum()
    }

    pub fn action_count(&self) -> usize {
        self.actions.len()
    }
}

impl fmt::Display for InputFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frame(player={} seq={} actions={} t={}ms)",
            self.player_id, self.sequence, self.actions.len(), self.timestamp_ms
        )
    }
}

// ── Validation Issue ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IssueType {
    SequenceGap,
    DuplicateFrame,
    ChecksumMismatch,
    TooManyActions,
    FrameTooFast,
    FrameTooSlow,
    ValueOutOfBounds,
    PredictionExceeded,
}

impl fmt::Display for IssueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SequenceGap => write!(f, "SequenceGap"),
            Self::DuplicateFrame => write!(f, "DuplicateFrame"),
            Self::ChecksumMismatch => write!(f, "ChecksumMismatch"),
            Self::TooManyActions => write!(f, "TooManyActions"),
            Self::FrameTooFast => write!(f, "FrameTooFast"),
            Self::FrameTooSlow => write!(f, "FrameTooSlow"),
            Self::ValueOutOfBounds => write!(f, "ValueOutOfBounds"),
            Self::PredictionExceeded => write!(f, "PredictionExceeded"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationIssue {
    pub issue_type: IssueType,
    pub player_id: u64,
    pub sequence: u64,
    pub details: String,
}

impl ValidationIssue {
    pub fn new(issue_type: IssueType, player_id: u64, sequence: u64, details: &str) -> Self {
        Self {
            issue_type,
            player_id,
            sequence,
            details: details.to_string(),
        }
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] player={} seq={}: {}",
            self.issue_type, self.player_id, self.sequence, self.details
        )
    }
}

// ── Validation Rule ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ValueBounds {
    pub action_id: u16,
    pub min: f64,
    pub max: f64,
}

impl ValueBounds {
    pub fn new(action_id: u16, min: f64, max: f64) -> Self {
        Self { action_id, min, max }
    }
}

// ── Validator Config ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InputValidatorConfig {
    pub max_actions_per_frame: usize,
    pub min_frame_interval_ms: u64,
    pub max_frame_interval_ms: u64,
    pub max_prediction_ahead: u64,
    pub validate_checksum: bool,
    pub value_bounds: Vec<ValueBounds>,
    pub history_size: usize,
}

impl Default for InputValidatorConfig {
    fn default() -> Self {
        Self {
            max_actions_per_frame: 16,
            min_frame_interval_ms: 8,
            max_frame_interval_ms: 5000,
            max_prediction_ahead: 10,
            validate_checksum: true,
            value_bounds: Vec::new(),
            history_size: 256,
        }
    }
}

impl InputValidatorConfig {
    pub fn with_max_actions(mut self, n: usize) -> Self {
        self.max_actions_per_frame = n;
        self
    }

    pub fn with_bounds(mut self, b: ValueBounds) -> Self {
        self.value_bounds.push(b);
        self
    }

    pub fn with_min_interval(mut self, ms: u64) -> Self {
        self.min_frame_interval_ms = ms;
        self
    }

    pub fn with_max_interval(mut self, ms: u64) -> Self {
        self.max_frame_interval_ms = ms;
        self
    }
}

// ── Player History ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PlayerHistory {
    last_sequence: Option<u64>,
    last_timestamp_ms: Option<u64>,
    seen_sequences: Vec<u64>,
    issues: Vec<ValidationIssue>,
    total_frames: u64,
    history_size: usize,
}

impl PlayerHistory {
    fn new(history_size: usize) -> Self {
        Self {
            last_sequence: None,
            last_timestamp_ms: None,
            seen_sequences: Vec::new(),
            issues: Vec::new(),
            total_frames: 0,
            history_size,
        }
    }

    fn record_sequence(&mut self, seq: u64) {
        self.seen_sequences.push(seq);
        if self.seen_sequences.len() > self.history_size {
            self.seen_sequences.remove(0);
        }
        self.last_sequence = Some(seq);
        self.total_frames += 1;
    }

    fn is_duplicate(&self, seq: u64) -> bool {
        self.seen_sequences.contains(&seq)
    }

    fn has_gap(&self, seq: u64) -> Option<(u64, u64)> {
        if let Some(last) = self.last_sequence {
            if seq > last + 1 {
                return Some((last + 1, seq - 1));
            }
        }
        None
    }
}

// ── Input Validator ─────────────────────────────────────────────

#[derive(Debug)]
pub struct InputValidator {
    config: InputValidatorConfig,
    players: HashMap<u64, PlayerHistory>,
    server_sequence: u64,
}

impl InputValidator {
    pub fn new(config: InputValidatorConfig) -> Self {
        Self {
            config,
            players: HashMap::new(),
            server_sequence: 0,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(InputValidatorConfig::default())
    }

    pub fn register_player(&mut self, player_id: u64) {
        self.players
            .entry(player_id)
            .or_insert_with(|| PlayerHistory::new(self.config.history_size));
    }

    pub fn set_server_sequence(&mut self, seq: u64) {
        self.server_sequence = seq;
    }

    pub fn validate_frame(&mut self, frame: &InputFrame) -> Result<(InputFrame, Vec<ValidationIssue>), InputValidateError> {
        if frame.actions.is_empty() {
            return Err(InputValidateError::EmptyFrame);
        }

        let history = self
            .players
            .get_mut(&frame.player_id)
            .ok_or(InputValidateError::PlayerNotRegistered(frame.player_id))?;

        let mut issues = Vec::new();

        // Checksum validation
        if self.config.validate_checksum && frame.checksum != 0 {
            let computed = frame.compute_checksum();
            if frame.checksum != computed {
                issues.push(ValidationIssue::new(
                    IssueType::ChecksumMismatch,
                    frame.player_id,
                    frame.sequence,
                    &format!("expected {}, got {}", frame.checksum, computed),
                ));
            }
        }

        // Duplicate check
        if history.is_duplicate(frame.sequence) {
            issues.push(ValidationIssue::new(
                IssueType::DuplicateFrame,
                frame.player_id,
                frame.sequence,
                "duplicate sequence number",
            ));
        }

        // Sequence gap
        if let Some((from, to)) = history.has_gap(frame.sequence) {
            issues.push(ValidationIssue::new(
                IssueType::SequenceGap,
                frame.player_id,
                frame.sequence,
                &format!("missing sequences {}..{}", from, to),
            ));
        }

        // Actions per frame
        if frame.actions.len() > self.config.max_actions_per_frame {
            issues.push(ValidationIssue::new(
                IssueType::TooManyActions,
                frame.player_id,
                frame.sequence,
                &format!("{} actions > max {}", frame.actions.len(), self.config.max_actions_per_frame),
            ));
        }

        // Timing
        if let Some(last_ts) = history.last_timestamp_ms {
            let interval = frame.timestamp_ms.saturating_sub(last_ts);
            if interval < self.config.min_frame_interval_ms {
                issues.push(ValidationIssue::new(
                    IssueType::FrameTooFast,
                    frame.player_id,
                    frame.sequence,
                    &format!("{}ms < min {}ms", interval, self.config.min_frame_interval_ms),
                ));
            }
            if interval > self.config.max_frame_interval_ms {
                issues.push(ValidationIssue::new(
                    IssueType::FrameTooSlow,
                    frame.player_id,
                    frame.sequence,
                    &format!("{}ms > max {}ms", interval, self.config.max_frame_interval_ms),
                ));
            }
        }

        // Prediction bounds
        if frame.sequence > self.server_sequence + self.config.max_prediction_ahead {
            issues.push(ValidationIssue::new(
                IssueType::PredictionExceeded,
                frame.player_id,
                frame.sequence,
                &format!("seq {} too far ahead of server {}", frame.sequence, self.server_sequence),
            ));
        }

        // Sanitize values
        let bounds_map: HashMap<u16, &ValueBounds> = self
            .config
            .value_bounds
            .iter()
            .map(|b| (b.action_id, b))
            .collect();

        let mut sanitized_frame = frame.clone();
        for action in &mut sanitized_frame.actions {
            if let Some(bounds) = bounds_map.get(&action.action_id) {
                let original = action.value;
                action.value = action.value.clamp(bounds.min, bounds.max);
                if (action.value - original).abs() > f64::EPSILON {
                    issues.push(ValidationIssue::new(
                        IssueType::ValueOutOfBounds,
                        frame.player_id,
                        frame.sequence,
                        &format!("action {} clamped {:.3} -> {:.3}", action.action_id, original, action.value),
                    ));
                }
            }
        }

        history.record_sequence(frame.sequence);
        history.last_timestamp_ms = Some(frame.timestamp_ms);
        for issue in &issues {
            history.issues.push(issue.clone());
        }

        Ok((sanitized_frame, issues))
    }

    pub fn player_issue_count(&self, player_id: u64) -> usize {
        self.players
            .get(&player_id)
            .map(|h| h.issues.len())
            .unwrap_or(0)
    }

    pub fn player_frame_count(&self, player_id: u64) -> u64 {
        self.players
            .get(&player_id)
            .map(|h| h.total_frames)
            .unwrap_or(0)
    }

    pub fn tracked_player_count(&self) -> usize {
        self.players.len()
    }

    pub fn config(&self) -> &InputValidatorConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(pid: u64, seq: u64, ts: u64) -> InputFrame {
        InputFrame::new(pid, seq, ts)
            .with_action(InputAction::new(1, 0.5))
    }

    fn validator() -> InputValidator {
        let mut v = InputValidator::with_default_config();
        v.register_player(1);
        v
    }

    #[test]
    fn test_valid_frame() {
        let mut v = validator();
        let f = frame(1, 1, 100);
        let (_, issues) = v.validate_frame(&f).unwrap();
        assert!(issues.is_empty());
    }

    #[test]
    fn test_empty_frame_error() {
        let mut v = validator();
        let f = InputFrame::new(1, 1, 100);
        let err = v.validate_frame(&f).unwrap_err();
        assert!(matches!(err, InputValidateError::EmptyFrame));
    }

    #[test]
    fn test_unregistered_player() {
        let mut v = validator();
        let f = frame(99, 1, 100);
        let err = v.validate_frame(&f).unwrap_err();
        assert!(matches!(err, InputValidateError::PlayerNotRegistered(99)));
    }

    #[test]
    fn test_duplicate_frame() {
        let mut v = validator();
        v.validate_frame(&frame(1, 1, 100)).unwrap();
        let (_, issues) = v.validate_frame(&frame(1, 1, 200)).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::DuplicateFrame));
    }

    #[test]
    fn test_sequence_gap() {
        let mut v = validator();
        v.validate_frame(&frame(1, 1, 100)).unwrap();
        let (_, issues) = v.validate_frame(&frame(1, 5, 200)).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::SequenceGap));
    }

    #[test]
    fn test_too_many_actions() {
        let cfg = InputValidatorConfig::default().with_max_actions(2);
        let mut v = InputValidator::new(cfg);
        v.register_player(1);
        let f = InputFrame::new(1, 1, 100)
            .with_action(InputAction::new(1, 0.1))
            .with_action(InputAction::new(2, 0.2))
            .with_action(InputAction::new(3, 0.3));
        let (_, issues) = v.validate_frame(&f).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::TooManyActions));
    }

    #[test]
    fn test_frame_too_fast() {
        let cfg = InputValidatorConfig::default().with_min_interval(50);
        let mut v = InputValidator::new(cfg);
        v.register_player(1);
        v.validate_frame(&frame(1, 1, 100)).unwrap();
        let (_, issues) = v.validate_frame(&frame(1, 2, 110)).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::FrameTooFast));
    }

    #[test]
    fn test_frame_too_slow() {
        let cfg = InputValidatorConfig::default().with_max_interval(1000);
        let mut v = InputValidator::new(cfg);
        v.register_player(1);
        v.validate_frame(&frame(1, 1, 100)).unwrap();
        let (_, issues) = v.validate_frame(&frame(1, 2, 5000)).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::FrameTooSlow));
    }

    #[test]
    fn test_prediction_exceeded() {
        let mut v = validator();
        v.set_server_sequence(0);
        let f = frame(1, 50, 100);
        let (_, issues) = v.validate_frame(&f).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::PredictionExceeded));
    }

    #[test]
    fn test_value_sanitization() {
        let cfg = InputValidatorConfig::default().with_bounds(ValueBounds::new(1, -1.0, 1.0));
        let mut v = InputValidator::new(cfg);
        v.register_player(1);
        let f = InputFrame::new(1, 1, 100).with_action(InputAction::new(1, 5.0));
        let (sanitized, issues) = v.validate_frame(&f).unwrap();
        assert!((sanitized.actions[0].value - 1.0).abs() < f64::EPSILON);
        assert!(issues.iter().any(|i| i.issue_type == IssueType::ValueOutOfBounds));
    }

    #[test]
    fn test_value_in_bounds_no_issue() {
        let cfg = InputValidatorConfig::default().with_bounds(ValueBounds::new(1, -1.0, 1.0));
        let mut v = InputValidator::new(cfg);
        v.register_player(1);
        let f = frame(1, 1, 100);
        let (_, issues) = v.validate_frame(&f).unwrap();
        assert!(!issues.iter().any(|i| i.issue_type == IssueType::ValueOutOfBounds));
    }

    #[test]
    fn test_checksum_validation() {
        let mut v = validator();
        let f = InputFrame::new(1, 1, 100)
            .with_action(InputAction::new(1, 0.5))
            .with_checksum(12345);
        let (_, issues) = v.validate_frame(&f).unwrap();
        assert!(issues.iter().any(|i| i.issue_type == IssueType::ChecksumMismatch));
    }

    #[test]
    fn test_valid_checksum() {
        let mut v = validator();
        let mut f = InputFrame::new(1, 1, 100).with_action(InputAction::new(1, 0.5));
        let cs = f.compute_checksum();
        f = f.with_checksum(cs);
        let (_, issues) = v.validate_frame(&f).unwrap();
        assert!(!issues.iter().any(|i| i.issue_type == IssueType::ChecksumMismatch));
    }

    #[test]
    fn test_player_issue_count() {
        let mut v = validator();
        v.validate_frame(&frame(1, 1, 100)).unwrap();
        v.validate_frame(&frame(1, 1, 200)).unwrap();
        assert!(v.player_issue_count(1) > 0);
    }

    #[test]
    fn test_player_frame_count() {
        let mut v = validator();
        v.validate_frame(&frame(1, 1, 100)).unwrap();
        v.validate_frame(&frame(1, 2, 200)).unwrap();
        assert_eq!(v.player_frame_count(1), 2);
    }

    #[test]
    fn test_frame_display() {
        let f = frame(1, 42, 500);
        let s = format!("{f}");
        assert!(s.contains("seq=42"));
        assert!(s.contains("player=1"));
    }

    #[test]
    fn test_issue_display() {
        let issue = ValidationIssue::new(IssueType::SequenceGap, 1, 5, "gap 2..4");
        let s = format!("{issue}");
        assert!(s.contains("SequenceGap"));
        assert!(s.contains("gap 2..4"));
    }

    #[test]
    fn test_action_clamped() {
        let a = InputAction::new(1, 10.0);
        let c = a.clamped(-1.0, 1.0);
        assert!((c.value - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracked_player_count() {
        let mut v = InputValidator::with_default_config();
        v.register_player(1);
        v.register_player(2);
        assert_eq!(v.tracked_player_count(), 2);
    }
}
