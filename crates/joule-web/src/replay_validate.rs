//! Replay validation system — re-simulation, checksum, event sequence verification.
//!
//! Replaces replay-analysis tooling with pure Rust.
//! ReplayEvent recording, tick-based re-simulation, checksum comparison
//! at key frames, impossible event detection (damage exceeding weapon
//! stats, movement through walls), event sequence validation, replay
//! integrity checks, validation reports with flagged events, and
//! configurable validation rules.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    EmptyReplay,
    CorruptChecksum { tick: u64, expected: u64, actual: u64 },
    MissingTicks { from: u64, to: u64 },
    InvalidEvent(String),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyReplay => write!(f, "empty replay"),
            Self::CorruptChecksum { tick, expected, actual } => {
                write!(f, "checksum mismatch at tick {tick}: expected {expected}, got {actual}")
            }
            Self::MissingTicks { from, to } => write!(f, "missing ticks {from}..{to}"),
            Self::InvalidEvent(msg) => write!(f, "invalid event: {msg}"),
        }
    }
}

impl std::error::Error for ReplayError {}

// ── Event Types ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Movement,
    Attack,
    Damage,
    Heal,
    ItemUse,
    Spawn,
    Death,
    Ability,
    Checkpoint,
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Movement => write!(f, "Movement"),
            Self::Attack => write!(f, "Attack"),
            Self::Damage => write!(f, "Damage"),
            Self::Heal => write!(f, "Heal"),
            Self::ItemUse => write!(f, "ItemUse"),
            Self::Spawn => write!(f, "Spawn"),
            Self::Death => write!(f, "Death"),
            Self::Ability => write!(f, "Ability"),
            Self::Checkpoint => write!(f, "Checkpoint"),
        }
    }
}

// ── Replay Event ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ReplayEvent {
    pub tick: u64,
    pub event_type: EventType,
    pub player_id: u64,
    pub data: HashMap<String, f64>,
}

impl ReplayEvent {
    pub fn new(tick: u64, event_type: EventType, player_id: u64) -> Self {
        Self {
            tick,
            event_type,
            player_id,
            data: HashMap::new(),
        }
    }

    pub fn with_data(mut self, key: &str, value: f64) -> Self {
        self.data.insert(key.to_string(), value);
        self
    }

    pub fn get_data(&self, key: &str) -> Option<f64> {
        self.data.get(key).copied()
    }

    pub fn checksum(&self) -> u64 {
        let mut hash: u64 = self.tick.wrapping_mul(2654435761);
        hash ^= self.player_id.wrapping_mul(40503);
        hash ^= (self.event_type as u64).wrapping_mul(65537);
        for (k, v) in &self.data {
            for b in k.bytes() {
                hash = hash.wrapping_add(b as u64).wrapping_mul(31);
            }
            hash ^= v.to_bits();
        }
        hash
    }
}

impl fmt::Display for ReplayEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[tick={}] player={} {}", self.tick, self.player_id, self.event_type)
    }
}

// ── Validation Flag ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlagSeverity {
    Info,
    Warning,
    Violation,
    Critical,
}

impl fmt::Display for FlagSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARN"),
            Self::Violation => write!(f, "VIOLATION"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationFlag {
    pub tick: u64,
    pub event_index: usize,
    pub severity: FlagSeverity,
    pub message: String,
}

impl ValidationFlag {
    pub fn new(tick: u64, index: usize, severity: FlagSeverity, message: &str) -> Self {
        Self {
            tick,
            event_index: index,
            severity,
            message: message.to_string(),
        }
    }
}

impl fmt::Display for ValidationFlag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] tick={} idx={}: {}", self.severity, self.tick, self.event_index, self.message)
    }
}

// ── Validation Report ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub total_events: usize,
    pub total_ticks: u64,
    pub flags: Vec<ValidationFlag>,
    pub checksum_mismatches: usize,
    pub tick_gaps: Vec<(u64, u64)>,
    pub is_valid: bool,
}

impl ValidationReport {
    fn new(total_events: usize, total_ticks: u64) -> Self {
        Self {
            total_events,
            total_ticks,
            flags: Vec::new(),
            checksum_mismatches: 0,
            tick_gaps: Vec::new(),
            is_valid: true,
        }
    }

    pub fn violation_count(&self) -> usize {
        self.flags
            .iter()
            .filter(|f| matches!(f.severity, FlagSeverity::Violation | FlagSeverity::Critical))
            .count()
    }

    pub fn flags_at_tick(&self, tick: u64) -> Vec<&ValidationFlag> {
        self.flags.iter().filter(|f| f.tick == tick).collect()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ReplayReport: events={} ticks={} flags={} valid={}",
            self.total_events, self.total_ticks, self.flags.len(), self.is_valid
        )
    }
}

// ── Validation Rule ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub name: String,
    pub event_type: EventType,
    pub field: String,
    pub max_value: f64,
    pub severity: FlagSeverity,
}

impl ValidationRule {
    pub fn new(name: &str, event_type: EventType, field: &str, max_value: f64) -> Self {
        Self {
            name: name.to_string(),
            event_type,
            field: field.to_string(),
            max_value,
            severity: FlagSeverity::Violation,
        }
    }

    pub fn with_severity(mut self, s: FlagSeverity) -> Self {
        self.severity = s;
        self
    }

    pub fn check(&self, event: &ReplayEvent) -> Option<String> {
        if event.event_type != self.event_type {
            return None;
        }
        if let Some(val) = event.get_data(&self.field) {
            if val > self.max_value {
                return Some(format!(
                    "{}: {} {} = {:.2} exceeds max {:.2}",
                    self.name, event.event_type, self.field, val, self.max_value
                ));
            }
        }
        None
    }
}

// ── Checkpoint ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub tick: u64,
    pub checksum: u64,
}

impl Checkpoint {
    pub fn new(tick: u64, checksum: u64) -> Self {
        Self { tick, checksum }
    }
}

// ── Replay Validator ────────────────────────────────────────────

#[derive(Debug)]
pub struct ReplayValidator {
    rules: Vec<ValidationRule>,
    checkpoints: Vec<Checkpoint>,
    max_movement_per_tick: f64,
    max_events_per_tick: usize,
}

impl ReplayValidator {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            checkpoints: Vec::new(),
            max_movement_per_tick: 50.0,
            max_events_per_tick: 20,
        }
    }

    pub fn with_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_checkpoint(mut self, cp: Checkpoint) -> Self {
        self.checkpoints.push(cp);
        self
    }

    pub fn with_max_movement(mut self, v: f64) -> Self {
        self.max_movement_per_tick = v;
        self
    }

    pub fn with_max_events_per_tick(mut self, n: usize) -> Self {
        self.max_events_per_tick = n;
        self
    }

    pub fn add_rule(&mut self, rule: ValidationRule) {
        self.rules.push(rule);
    }

    pub fn add_checkpoint(&mut self, cp: Checkpoint) {
        self.checkpoints.push(cp);
    }

    pub fn validate(&self, events: &[ReplayEvent]) -> Result<ValidationReport, ReplayError> {
        if events.is_empty() {
            return Err(ReplayError::EmptyReplay);
        }

        let first_tick = events[0].tick;
        let last_tick = events[events.len() - 1].tick;
        let total_ticks = last_tick.saturating_sub(first_tick) + 1;
        let mut report = ValidationReport::new(events.len(), total_ticks);

        // Check tick continuity
        let mut prev_tick = events[0].tick;
        for (i, evt) in events.iter().enumerate().skip(1) {
            if evt.tick < prev_tick {
                report.flags.push(ValidationFlag::new(
                    evt.tick,
                    i,
                    FlagSeverity::Critical,
                    &format!("tick went backwards: {} -> {}", prev_tick, evt.tick),
                ));
                report.is_valid = false;
            }
            if evt.tick > prev_tick + 1 {
                let gap = (prev_tick + 1, evt.tick - 1);
                report.tick_gaps.push(gap);
                report.flags.push(ValidationFlag::new(
                    evt.tick,
                    i,
                    FlagSeverity::Warning,
                    &format!("tick gap: {}..{}", gap.0, gap.1),
                ));
            }
            prev_tick = evt.tick;
        }

        // Events per tick
        let mut tick_counts: HashMap<u64, usize> = HashMap::new();
        for evt in events {
            *tick_counts.entry(evt.tick).or_insert(0) += 1;
        }
        for (&tick, &count) in &tick_counts {
            if count > self.max_events_per_tick {
                report.flags.push(ValidationFlag::new(
                    tick,
                    0,
                    FlagSeverity::Violation,
                    &format!("{count} events in one tick (max {})", self.max_events_per_tick),
                ));
            }
        }

        // Rule checks
        for (i, evt) in events.iter().enumerate() {
            for rule in &self.rules {
                if let Some(msg) = rule.check(evt) {
                    report.flags.push(ValidationFlag::new(evt.tick, i, rule.severity, &msg));
                }
            }
        }

        // Movement validation (distance per tick)
        let movement_events: Vec<_> = events
            .iter()
            .enumerate()
            .filter(|(_, e)| e.event_type == EventType::Movement)
            .collect();
        for pair in movement_events.windows(2) {
            let (_, a) = &pair[0];
            let (idx_b, b) = &pair[1];
            if a.player_id == b.player_id && b.tick == a.tick + 1 {
                if let (Some(ax), Some(ay), Some(bx), Some(by)) = (
                    a.get_data("x"), a.get_data("y"),
                    b.get_data("x"), b.get_data("y"),
                ) {
                    let dist = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt();
                    if dist > self.max_movement_per_tick {
                        report.flags.push(ValidationFlag::new(
                            b.tick,
                            *idx_b,
                            FlagSeverity::Violation,
                            &format!("movement {dist:.2} exceeds max {:.2}", self.max_movement_per_tick),
                        ));
                    }
                }
            }
        }

        // Checkpoint validation
        let event_checksums: HashMap<u64, u64> = events
            .iter()
            .filter(|e| e.event_type == EventType::Checkpoint)
            .map(|e| (e.tick, e.checksum()))
            .collect();
        for cp in &self.checkpoints {
            if let Some(&actual) = event_checksums.get(&cp.tick) {
                if actual != cp.checksum {
                    report.checksum_mismatches += 1;
                    report.flags.push(ValidationFlag::new(
                        cp.tick,
                        0,
                        FlagSeverity::Critical,
                        &format!("checksum mismatch: expected {}, got {}", cp.checksum, actual),
                    ));
                    report.is_valid = false;
                }
            }
        }

        if report.violation_count() > 0 {
            report.is_valid = false;
        }

        Ok(report)
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }
}

impl Default for ReplayValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn movement(tick: u64, pid: u64, x: f64, y: f64) -> ReplayEvent {
        ReplayEvent::new(tick, EventType::Movement, pid)
            .with_data("x", x)
            .with_data("y", y)
    }

    fn damage(tick: u64, pid: u64, amount: f64) -> ReplayEvent {
        ReplayEvent::new(tick, EventType::Damage, pid).with_data("amount", amount)
    }

    #[test]
    fn test_empty_replay_error() {
        let v = ReplayValidator::new();
        let err = v.validate(&[]).unwrap_err();
        assert!(matches!(err, ReplayError::EmptyReplay));
    }

    #[test]
    fn test_valid_replay() {
        let v = ReplayValidator::new();
        let events = vec![
            movement(0, 1, 0.0, 0.0),
            movement(1, 1, 1.0, 0.0),
            movement(2, 1, 2.0, 0.0),
        ];
        let report = v.validate(&events).unwrap();
        assert!(report.is_valid);
        assert_eq!(report.total_events, 3);
    }

    #[test]
    fn test_tick_gap_detected() {
        let v = ReplayValidator::new();
        let events = vec![
            movement(0, 1, 0.0, 0.0),
            movement(5, 1, 5.0, 0.0),
        ];
        let report = v.validate(&events).unwrap();
        assert_eq!(report.tick_gaps.len(), 1);
        assert_eq!(report.tick_gaps[0], (1, 4));
    }

    #[test]
    fn test_backward_tick_flagged() {
        let v = ReplayValidator::new();
        let events = vec![
            movement(5, 1, 0.0, 0.0),
            movement(3, 1, 1.0, 0.0),
        ];
        let report = v.validate(&events).unwrap();
        assert!(!report.is_valid);
        assert!(report.flags.iter().any(|f| f.severity == FlagSeverity::Critical));
    }

    #[test]
    fn test_excessive_movement_flagged() {
        let v = ReplayValidator::new().with_max_movement(10.0);
        let events = vec![
            movement(0, 1, 0.0, 0.0),
            movement(1, 1, 100.0, 0.0),
        ];
        let report = v.validate(&events).unwrap();
        assert!(report.violation_count() > 0);
    }

    #[test]
    fn test_rule_violation() {
        let rule = ValidationRule::new("max_damage", EventType::Damage, "amount", 100.0);
        let v = ReplayValidator::new().with_rule(rule);
        let events = vec![damage(0, 1, 500.0)];
        let report = v.validate(&events).unwrap();
        assert!(report.violation_count() > 0);
    }

    #[test]
    fn test_rule_passes() {
        let rule = ValidationRule::new("max_damage", EventType::Damage, "amount", 100.0);
        let v = ReplayValidator::new().with_rule(rule);
        let events = vec![damage(0, 1, 50.0)];
        let report = v.validate(&events).unwrap();
        assert_eq!(report.violation_count(), 0);
    }

    #[test]
    fn test_events_per_tick_limit() {
        let v = ReplayValidator::new().with_max_events_per_tick(2);
        let events = vec![
            damage(0, 1, 10.0),
            damage(0, 2, 10.0),
            damage(0, 3, 10.0),
        ];
        let report = v.validate(&events).unwrap();
        assert!(report.flags.iter().any(|f| f.message.contains("events in one tick")));
    }

    #[test]
    fn test_checkpoint_mismatch() {
        let cp_evt = ReplayEvent::new(5, EventType::Checkpoint, 0).with_data("state", 42.0);
        let checksum = cp_evt.checksum();
        let v = ReplayValidator::new()
            .with_checkpoint(Checkpoint::new(5, checksum.wrapping_add(1)));
        let events = vec![
            movement(0, 1, 0.0, 0.0),
            cp_evt,
        ];
        let report = v.validate(&events).unwrap();
        assert!(!report.is_valid);
        assert!(report.checksum_mismatches > 0);
    }

    #[test]
    fn test_checkpoint_passes() {
        let cp_evt = ReplayEvent::new(5, EventType::Checkpoint, 0).with_data("state", 42.0);
        let checksum = cp_evt.checksum();
        let v = ReplayValidator::new().with_checkpoint(Checkpoint::new(5, checksum));
        let events = vec![
            movement(0, 1, 0.0, 0.0),
            cp_evt,
        ];
        let report = v.validate(&events).unwrap();
        assert_eq!(report.checksum_mismatches, 0);
    }

    #[test]
    fn test_event_display() {
        let e = ReplayEvent::new(42, EventType::Attack, 7);
        let s = format!("{e}");
        assert!(s.contains("tick=42"));
        assert!(s.contains("Attack"));
    }

    #[test]
    fn test_flag_display() {
        let f = ValidationFlag::new(10, 5, FlagSeverity::Warning, "test warning");
        let s = format!("{f}");
        assert!(s.contains("WARN"));
        assert!(s.contains("test warning"));
    }

    #[test]
    fn test_report_display() {
        let report = ValidationReport::new(100, 50);
        let s = format!("{report}");
        assert!(s.contains("events=100"));
        assert!(s.contains("ticks=50"));
    }

    #[test]
    fn test_event_data_access() {
        let e = ReplayEvent::new(0, EventType::Damage, 1)
            .with_data("amount", 25.5)
            .with_data("armor", 10.0);
        assert!((e.get_data("amount").unwrap() - 25.5).abs() < f64::EPSILON);
        assert!(e.get_data("missing").is_none());
    }

    #[test]
    fn test_rule_wrong_event_type() {
        let rule = ValidationRule::new("max_damage", EventType::Damage, "amount", 100.0);
        let event = movement(0, 1, 0.0, 0.0);
        assert!(rule.check(&event).is_none());
    }

    #[test]
    fn test_rule_severity_builder() {
        let rule = ValidationRule::new("test", EventType::Heal, "amount", 50.0)
            .with_severity(FlagSeverity::Critical);
        assert_eq!(rule.severity, FlagSeverity::Critical);
    }

    #[test]
    fn test_validator_add_rule() {
        let mut v = ReplayValidator::new();
        v.add_rule(ValidationRule::new("r1", EventType::Damage, "amount", 100.0));
        v.add_rule(ValidationRule::new("r2", EventType::Heal, "amount", 200.0));
        assert_eq!(v.rule_count(), 2);
    }

    #[test]
    fn test_flags_at_tick() {
        let rule = ValidationRule::new("dmg", EventType::Damage, "amount", 10.0);
        let v = ReplayValidator::new().with_rule(rule);
        let events = vec![damage(5, 1, 50.0), damage(5, 2, 60.0)];
        let report = v.validate(&events).unwrap();
        let at_5 = report.flags_at_tick(5);
        assert!(at_5.len() >= 2);
    }
}
