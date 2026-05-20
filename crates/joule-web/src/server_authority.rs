//! Server-authoritative validation framework — input checking, rate limiting, reconciliation.
//!
//! Replaces server-side authority middleware with pure Rust.
//! ClientInput processing, game-rule validation, InputValidationResult
//! classification, action rate limiting per tick, position reconciliation
//! with rubber-banding, resource validation, cooldown enforcement,
//! action permission checks, and validation statistics.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityError {
    PlayerNotFound(u64),
    InvalidConfig(String),
    RuleNotFound(String),
}

impl fmt::Display for AuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayerNotFound(id) => write!(f, "player not found: {id}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::RuleNotFound(name) => write!(f, "rule not found: {name}"),
        }
    }
}

impl std::error::Error for AuthorityError {}

// ── Action Types ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    Move,
    Attack,
    UseItem,
    Ability,
    Interact,
    Chat,
    Trade,
    Build,
}

impl fmt::Display for ActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Move => write!(f, "Move"),
            Self::Attack => write!(f, "Attack"),
            Self::UseItem => write!(f, "UseItem"),
            Self::Ability => write!(f, "Ability"),
            Self::Interact => write!(f, "Interact"),
            Self::Chat => write!(f, "Chat"),
            Self::Trade => write!(f, "Trade"),
            Self::Build => write!(f, "Build"),
        }
    }
}

// ── Client Input ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ClientInput {
    pub player_id: u64,
    pub action: ActionType,
    pub parameters: HashMap<String, f64>,
    pub tick: u64,
}

impl ClientInput {
    pub fn new(player_id: u64, action: ActionType, tick: u64) -> Self {
        Self {
            player_id,
            action,
            parameters: HashMap::new(),
            tick,
        }
    }

    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    pub fn get_param(&self, key: &str) -> Option<f64> {
        self.parameters.get(key).copied()
    }
}

impl fmt::Display for ClientInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[tick={}] player={} action={}", self.tick, self.player_id, self.action)
    }
}

// ── Validation Result ───────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum InputValidationResult {
    Valid,
    Modified { reason: String, original: HashMap<String, f64>, corrected: HashMap<String, f64> },
    Rejected { reason: String },
}

impl InputValidationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }

    pub fn is_modified(&self) -> bool {
        matches!(self, Self::Modified { .. })
    }
}

impl fmt::Display for InputValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => write!(f, "Valid"),
            Self::Modified { reason, .. } => write!(f, "Modified: {reason}"),
            Self::Rejected { reason } => write!(f, "Rejected: {reason}"),
        }
    }
}

// ── Player State ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub player_id: u64,
    pub x: f64,
    pub y: f64,
    pub resources: HashMap<String, f64>,
    pub cooldowns: HashMap<ActionType, u64>,
    pub permissions: Vec<ActionType>,
    pub last_action_tick: HashMap<ActionType, u64>,
    pub action_counts_this_tick: HashMap<(u64, ActionType), u32>,
}

impl PlayerState {
    pub fn new(player_id: u64) -> Self {
        Self {
            player_id,
            x: 0.0,
            y: 0.0,
            resources: HashMap::new(),
            cooldowns: HashMap::new(),
            permissions: vec![
                ActionType::Move, ActionType::Attack, ActionType::UseItem,
                ActionType::Ability, ActionType::Interact, ActionType::Chat,
                ActionType::Trade, ActionType::Build,
            ],
            last_action_tick: HashMap::new(),
            action_counts_this_tick: HashMap::new(),
        }
    }

    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.x = x;
        self.y = y;
        self
    }

    pub fn with_resource(mut self, name: &str, amount: f64) -> Self {
        self.resources.insert(name.to_string(), amount);
        self
    }

    pub fn set_cooldown(&mut self, action: ActionType, expires_tick: u64) {
        self.cooldowns.insert(action, expires_tick);
    }

    pub fn has_permission(&self, action: ActionType) -> bool {
        self.permissions.contains(&action)
    }

    pub fn revoke_permission(&mut self, action: ActionType) {
        self.permissions.retain(|a| *a != action);
    }

    pub fn is_on_cooldown(&self, action: ActionType, current_tick: u64) -> bool {
        self.cooldowns
            .get(&action)
            .map(|expires| current_tick < *expires)
            .unwrap_or(false)
    }
}

// ── Server Validator Config ─────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    pub max_actions_per_tick: HashMap<ActionType, u32>,
    pub max_move_distance: f64,
    pub rubber_band_threshold: f64,
    pub default_rate_limit: u32,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        let mut limits = HashMap::new();
        limits.insert(ActionType::Move, 1);
        limits.insert(ActionType::Attack, 2);
        limits.insert(ActionType::UseItem, 1);
        limits.insert(ActionType::Chat, 3);
        Self {
            max_actions_per_tick: limits,
            max_move_distance: 10.0,
            rubber_band_threshold: 15.0,
            default_rate_limit: 5,
        }
    }
}

// ── Validation Stats ────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    pub total_inputs: u64,
    pub valid_count: u64,
    pub modified_count: u64,
    pub rejected_count: u64,
    pub rate_limited_count: u64,
    pub cooldown_blocked_count: u64,
    pub permission_denied_count: u64,
}

impl ValidationStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_inputs == 0 {
            return 1.0;
        }
        self.valid_count as f64 / self.total_inputs as f64
    }
}

impl fmt::Display for ValidationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "inputs={} valid={} modified={} rejected={} accept={:.1}%",
            self.total_inputs, self.valid_count, self.modified_count,
            self.rejected_count, self.acceptance_rate() * 100.0
        )
    }
}

// ── Server Validator ────────────────────────────────────────────

#[derive(Debug)]
pub struct ServerValidator {
    config: ValidatorConfig,
    players: HashMap<u64, PlayerState>,
    stats: ValidationStats,
}

impl ServerValidator {
    pub fn new(config: ValidatorConfig) -> Self {
        Self {
            config,
            players: HashMap::new(),
            stats: ValidationStats::default(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(ValidatorConfig::default())
    }

    pub fn register_player(&mut self, state: PlayerState) {
        self.players.insert(state.player_id, state);
    }

    pub fn validate_input(&mut self, input: &ClientInput) -> Result<InputValidationResult, AuthorityError> {
        if !self.players.contains_key(&input.player_id) {
            return Err(AuthorityError::PlayerNotFound(input.player_id));
        }

        self.stats.total_inputs += 1;

        // Permission check
        {
            let player = self.players.get(&input.player_id).unwrap();
            if !player.has_permission(input.action) {
                self.stats.rejected_count += 1;
                self.stats.permission_denied_count += 1;
                return Ok(InputValidationResult::Rejected {
                    reason: format!("no permission for {}", input.action),
                });
            }
            if player.is_on_cooldown(input.action, input.tick) {
                self.stats.rejected_count += 1;
                self.stats.cooldown_blocked_count += 1;
                return Ok(InputValidationResult::Rejected {
                    reason: format!("{} on cooldown", input.action),
                });
            }
        }

        // Rate limit check
        {
            let player = self.players.get_mut(&input.player_id).unwrap();
            let tick_key = (input.tick, input.action);
            let count = player.action_counts_this_tick.entry(tick_key).or_insert(0);
            let limit = self
                .config
                .max_actions_per_tick
                .get(&input.action)
                .copied()
                .unwrap_or(self.config.default_rate_limit);
            if *count >= limit {
                self.stats.rejected_count += 1;
                self.stats.rate_limited_count += 1;
                return Ok(InputValidationResult::Rejected {
                    reason: format!("{} rate limited ({}/{})", input.action, count, limit),
                });
            }
            *count += 1;
        }

        // Action-specific validation
        let result = match input.action {
            ActionType::Move => {
                let player = self.players.get_mut(&input.player_id).unwrap();
                Self::validate_move_static(&self.config, input, player)
            }
            ActionType::UseItem | ActionType::Trade => {
                let player = self.players.get_mut(&input.player_id).unwrap();
                Self::validate_resource_action_static(input, player)
            }
            _ => InputValidationResult::Valid,
        };

        match &result {
            InputValidationResult::Valid => self.stats.valid_count += 1,
            InputValidationResult::Modified { .. } => self.stats.modified_count += 1,
            InputValidationResult::Rejected { .. } => self.stats.rejected_count += 1,
        }

        let player = self.players.get_mut(&input.player_id).unwrap();
        player.last_action_tick.insert(input.action, input.tick);
        Ok(result)
    }

    fn validate_move_static(config: &ValidatorConfig, input: &ClientInput, player: &mut PlayerState) -> InputValidationResult {
        let target_x = input.get_param("x").unwrap_or(player.x);
        let target_y = input.get_param("y").unwrap_or(player.y);
        let dx = target_x - player.x;
        let dy = target_y - player.y;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist > config.rubber_band_threshold {
            // Rubber-band snap back
            let mut original = HashMap::new();
            original.insert("x".to_string(), target_x);
            original.insert("y".to_string(), target_y);
            let mut corrected = HashMap::new();
            corrected.insert("x".to_string(), player.x);
            corrected.insert("y".to_string(), player.y);
            return InputValidationResult::Modified {
                reason: format!("rubber-band: dist {dist:.2} > {:.2}", config.rubber_band_threshold),
                original,
                corrected,
            };
        }

        if dist > config.max_move_distance {
            let scale = config.max_move_distance / dist;
            let clamped_x = player.x + dx * scale;
            let clamped_y = player.y + dy * scale;
            player.x = clamped_x;
            player.y = clamped_y;
            let mut original = HashMap::new();
            original.insert("x".to_string(), target_x);
            original.insert("y".to_string(), target_y);
            let mut corrected = HashMap::new();
            corrected.insert("x".to_string(), clamped_x);
            corrected.insert("y".to_string(), clamped_y);
            return InputValidationResult::Modified {
                reason: format!("clamped move: dist {dist:.2} > max {:.2}", config.max_move_distance),
                original,
                corrected,
            };
        }

        player.x = target_x;
        player.y = target_y;
        InputValidationResult::Valid
    }

    fn validate_resource_action_static(input: &ClientInput, player: &mut PlayerState) -> InputValidationResult {
        if let Some(cost) = input.get_param("cost") {
            let resource_name = input
                .parameters
                .keys()
                .find(|k| k.as_str() != "cost")
                .cloned()
                .unwrap_or_else(|| "gold".to_string());
            let available = player.resources.get(&resource_name).copied().unwrap_or(0.0);
            if cost > available {
                return InputValidationResult::Rejected {
                    reason: format!("insufficient {resource_name}: need {cost:.0}, have {available:.0}"),
                };
            }
            *player.resources.entry(resource_name).or_insert(0.0) -= cost;
        }
        InputValidationResult::Valid
    }

    pub fn stats(&self) -> &ValidationStats {
        &self.stats
    }

    pub fn player_state(&self, player_id: u64) -> Option<&PlayerState> {
        self.players.get(&player_id)
    }

    pub fn player_count(&self) -> usize {
        self.players.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> ServerValidator {
        let mut v = ServerValidator::with_default_config();
        v.register_player(
            PlayerState::new(1)
                .with_position(0.0, 0.0)
                .with_resource("gold", 100.0),
        );
        v
    }

    #[test]
    fn test_valid_move() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::Move, 1).with_param("x", 5.0).with_param("y", 0.0);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_move_clamped() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::Move, 1).with_param("x", 12.0).with_param("y", 0.0);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_modified());
    }

    #[test]
    fn test_rubber_band() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::Move, 1).with_param("x", 100.0).with_param("y", 0.0);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_modified());
        if let InputValidationResult::Modified { reason, .. } = result {
            assert!(reason.contains("rubber-band"));
        }
    }

    #[test]
    fn test_player_not_found() {
        let mut v = setup();
        let input = ClientInput::new(99, ActionType::Move, 1);
        assert!(matches!(v.validate_input(&input), Err(AuthorityError::PlayerNotFound(99))));
    }

    #[test]
    fn test_rate_limiting() {
        let mut v = setup();
        let i1 = ClientInput::new(1, ActionType::Move, 1).with_param("x", 1.0).with_param("y", 0.0);
        let i2 = ClientInput::new(1, ActionType::Move, 1).with_param("x", 2.0).with_param("y", 0.0);
        v.validate_input(&i1).unwrap();
        let result = v.validate_input(&i2).unwrap();
        assert!(result.is_rejected());
    }

    #[test]
    fn test_cooldown_enforcement() {
        let mut v = setup();
        {
            let p = v.players.get_mut(&1).unwrap();
            p.set_cooldown(ActionType::Attack, 10);
        }
        let input = ClientInput::new(1, ActionType::Attack, 5);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_rejected());
    }

    #[test]
    fn test_cooldown_expired() {
        let mut v = setup();
        {
            let p = v.players.get_mut(&1).unwrap();
            p.set_cooldown(ActionType::Attack, 5);
        }
        let input = ClientInput::new(1, ActionType::Attack, 10);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_permission_denied() {
        let mut v = setup();
        {
            let p = v.players.get_mut(&1).unwrap();
            p.revoke_permission(ActionType::Build);
        }
        let input = ClientInput::new(1, ActionType::Build, 1);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_rejected());
    }

    #[test]
    fn test_resource_validation_pass() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::UseItem, 1)
            .with_param("cost", 50.0)
            .with_param("gold", 0.0);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_valid());
    }

    #[test]
    fn test_resource_validation_fail() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::UseItem, 1)
            .with_param("cost", 200.0)
            .with_param("gold", 0.0);
        let result = v.validate_input(&input).unwrap();
        assert!(result.is_rejected());
    }

    #[test]
    fn test_stats_tracking() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::Move, 1).with_param("x", 1.0).with_param("y", 0.0);
        v.validate_input(&input).unwrap();
        assert_eq!(v.stats().total_inputs, 1);
        assert_eq!(v.stats().valid_count, 1);
    }

    #[test]
    fn test_acceptance_rate() {
        let mut v = setup();
        let good = ClientInput::new(1, ActionType::Move, 1).with_param("x", 1.0).with_param("y", 0.0);
        v.validate_input(&good).unwrap();
        let bad = ClientInput::new(1, ActionType::Move, 1).with_param("x", 2.0).with_param("y", 0.0);
        v.validate_input(&bad).unwrap();
        assert!(v.stats().acceptance_rate() < 1.0);
    }

    #[test]
    fn test_client_input_display() {
        let input = ClientInput::new(1, ActionType::Attack, 42);
        let s = format!("{input}");
        assert!(s.contains("Attack"));
        assert!(s.contains("tick=42"));
    }

    #[test]
    fn test_validation_result_display() {
        let r = InputValidationResult::Valid;
        assert_eq!(format!("{r}"), "Valid");
        let r2 = InputValidationResult::Rejected { reason: "test".to_string() };
        assert!(format!("{r2}").contains("Rejected"));
    }

    #[test]
    fn test_player_count() {
        let mut v = ServerValidator::with_default_config();
        v.register_player(PlayerState::new(1));
        v.register_player(PlayerState::new(2));
        assert_eq!(v.player_count(), 2);
    }

    #[test]
    fn test_stats_display() {
        let stats = ValidationStats { total_inputs: 100, valid_count: 90, ..Default::default() };
        let s = format!("{stats}");
        assert!(s.contains("inputs=100"));
        assert!(s.contains("valid=90"));
    }

    #[test]
    fn test_player_state_builder() {
        let p = PlayerState::new(1).with_position(5.0, 10.0).with_resource("mana", 200.0);
        assert!((p.x - 5.0).abs() < f64::EPSILON);
        assert!((p.resources["mana"] - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_action_type_display() {
        assert_eq!(format!("{}", ActionType::Move), "Move");
        assert_eq!(format!("{}", ActionType::Trade), "Trade");
    }

    #[test]
    fn test_position_updates_on_valid_move() {
        let mut v = setup();
        let input = ClientInput::new(1, ActionType::Move, 1).with_param("x", 5.0).with_param("y", 3.0);
        v.validate_input(&input).unwrap();
        let state = v.player_state(1).unwrap();
        assert!((state.x - 5.0).abs() < f64::EPSILON);
        assert!((state.y - 3.0).abs() < f64::EPSILON);
    }
}
