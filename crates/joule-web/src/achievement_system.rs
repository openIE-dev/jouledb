//! Achievement/trophy system — achievements, criteria, progress, rewards.
//!
//! Replaces achievement.js / Steamworks-achievement with pure Rust.
//! Achievement definitions, criteria types (counter/flag/multi-flag),
//! progress tracking, unlock timestamps, categories, rarity, rewards,
//! statistics backend, and serialization.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AchievementError {
    AchievementNotFound(String),
    AlreadyUnlocked(String),
    StatNotFound(String),
    DuplicateAchievement(String),
    InvalidCriteria(String),
}

impl fmt::Display for AchievementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AchievementNotFound(id) => write!(f, "achievement not found: {id}"),
            Self::AlreadyUnlocked(id) => write!(f, "already unlocked: {id}"),
            Self::StatNotFound(name) => write!(f, "stat not found: {name}"),
            Self::DuplicateAchievement(id) => write!(f, "duplicate achievement: {id}"),
            Self::InvalidCriteria(msg) => write!(f, "invalid criteria: {msg}"),
        }
    }
}

impl std::error::Error for AchievementError {}

// ── Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AchievementCategory {
    Exploration,
    Combat,
    Social,
    Crafting,
    Collection,
    Story,
    Misc,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Track a numeric counter reaching a threshold.
    Counter { key: String, threshold: u64 },
    /// A single boolean flag must be set.
    Flag { key: String },
    /// All listed flags must be set.
    MultiFlag { keys: Vec<String> },
}

impl Criteria {
    pub fn is_met(&self, stats: &StatsBackend) -> bool {
        match self {
            Self::Counter { key, threshold } => {
                stats.get_counter(key) >= *threshold
            }
            Self::Flag { key } => stats.get_flag(key),
            Self::MultiFlag { keys } => {
                keys.iter().all(|k| stats.get_flag(k))
            }
        }
    }

    pub fn progress(&self, stats: &StatsBackend) -> f64 {
        match self {
            Self::Counter { key, threshold } => {
                if *threshold == 0 { return 1.0; }
                let val = stats.get_counter(key) as f64;
                (val / *threshold as f64).min(1.0)
            }
            Self::Flag { key } => if stats.get_flag(key) { 1.0 } else { 0.0 },
            Self::MultiFlag { keys } => {
                if keys.is_empty() { return 1.0; }
                let set = keys.iter().filter(|k| stats.get_flag(k)).count();
                set as f64 / keys.len() as f64
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AchievementReward {
    Title(String),
    Badge(String),
    Item(u64, u32),
    Currency(u64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub hidden: bool,
    pub category: AchievementCategory,
    pub criteria: Criteria,
    pub rewards: Vec<AchievementReward>,
}

impl Achievement {
    pub fn new(id: &str, name: &str, description: &str, criteria: Criteria) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            icon: String::new(),
            hidden: false,
            category: AchievementCategory::Misc,
            criteria,
            rewards: Vec::new(),
        }
    }

    pub fn with_icon(mut self, icon: &str) -> Self { self.icon = icon.to_string(); self }
    pub fn with_hidden(mut self, h: bool) -> Self { self.hidden = h; self }
    pub fn with_category(mut self, c: AchievementCategory) -> Self { self.category = c; self }
    pub fn with_reward(mut self, r: AchievementReward) -> Self { self.rewards.push(r); self }
}

// ── Stats Backend ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct StatsBackend {
    counters: HashMap<String, u64>,
    flags: HashMap<String, bool>,
}

impl StatsBackend {
    pub fn new() -> Self {
        Self { counters: HashMap::new(), flags: HashMap::new() }
    }

    pub fn increment_counter(&mut self, key: &str, amount: u64) -> u64 {
        let entry = self.counters.entry(key.to_string()).or_insert(0);
        *entry += amount;
        *entry
    }

    pub fn set_counter(&mut self, key: &str, value: u64) {
        self.counters.insert(key.to_string(), value);
    }

    pub fn get_counter(&self, key: &str) -> u64 {
        self.counters.get(key).copied().unwrap_or(0)
    }

    pub fn set_flag(&mut self, key: &str, value: bool) {
        self.flags.insert(key.to_string(), value);
    }

    pub fn get_flag(&self, key: &str) -> bool {
        self.flags.get(key).copied().unwrap_or(false)
    }

    pub fn counter_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.counters.keys().map(|k| k.as_str()).collect();
        keys.sort();
        keys
    }

    pub fn flag_keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.flags.keys().map(|k| k.as_str()).collect();
        keys.sort();
        keys
    }

    pub fn serialize(&self) -> SerializedStats {
        let mut counters: Vec<(String, u64)> = self.counters.iter()
            .map(|(k, v)| (k.clone(), *v)).collect();
        counters.sort_by(|a, b| a.0.cmp(&b.0));
        let mut flags: Vec<(String, bool)> = self.flags.iter()
            .map(|(k, v)| (k.clone(), *v)).collect();
        flags.sort_by(|a, b| a.0.cmp(&b.0));
        SerializedStats { counters, flags }
    }

    pub fn deserialize(data: &SerializedStats) -> Self {
        let mut backend = Self::new();
        for (k, v) in &data.counters {
            backend.counters.insert(k.clone(), *v);
        }
        for (k, v) in &data.flags {
            backend.flags.insert(k.clone(), *v);
        }
        backend
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SerializedStats {
    pub counters: Vec<(String, u64)>,
    pub flags: Vec<(String, bool)>,
}

// ── Unlock Record ───────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnlockRecord {
    pub achievement_id: String,
    pub unlocked_at_ms: u64,
}

// ── Achievement System ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AchievementSystem {
    achievements: HashMap<String, Achievement>,
    unlocked: HashMap<String, UnlockRecord>,
    stats: StatsBackend,
    total_players: u64,
    unlock_counts: HashMap<String, u64>,
}

impl AchievementSystem {
    pub fn new() -> Self {
        Self {
            achievements: HashMap::new(),
            unlocked: HashMap::new(),
            stats: StatsBackend::new(),
            total_players: 1,
            unlock_counts: HashMap::new(),
        }
    }

    pub fn register(&mut self, achievement: Achievement) -> Result<(), AchievementError> {
        if self.achievements.contains_key(&achievement.id) {
            return Err(AchievementError::DuplicateAchievement(achievement.id.clone()));
        }
        self.achievements.insert(achievement.id.clone(), achievement);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&Achievement> {
        self.achievements.get(id)
    }

    pub fn is_unlocked(&self, id: &str) -> bool {
        self.unlocked.contains_key(id)
    }

    pub fn unlock_record(&self, id: &str) -> Option<&UnlockRecord> {
        self.unlocked.get(id)
    }

    pub fn stats(&self) -> &StatsBackend { &self.stats }
    pub fn stats_mut(&mut self) -> &mut StatsBackend { &mut self.stats }

    /// Increment a counter and check for newly unlocked achievements.
    pub fn increment_and_check(&mut self, key: &str, amount: u64, current_time_ms: u64) -> Vec<String> {
        self.stats.increment_counter(key, amount);
        self.check_all(current_time_ms)
    }

    /// Set a flag and check for newly unlocked achievements.
    pub fn set_flag_and_check(&mut self, key: &str, current_time_ms: u64) -> Vec<String> {
        self.stats.set_flag(key, true);
        self.check_all(current_time_ms)
    }

    /// Check all achievements and return newly unlocked IDs.
    pub fn check_all(&mut self, current_time_ms: u64) -> Vec<String> {
        let mut newly_unlocked = Vec::new();
        let ids: Vec<String> = self.achievements.keys().cloned().collect();
        for id in ids {
            if self.unlocked.contains_key(&id) { continue; }
            let criteria = self.achievements[&id].criteria.clone();
            if criteria.is_met(&self.stats) {
                self.unlocked.insert(id.clone(), UnlockRecord {
                    achievement_id: id.clone(),
                    unlocked_at_ms: current_time_ms,
                });
                *self.unlock_counts.entry(id.clone()).or_insert(0) += 1;
                newly_unlocked.push(id);
            }
        }
        newly_unlocked.sort();
        newly_unlocked
    }

    /// Get progress for an achievement (0.0 to 1.0).
    pub fn progress(&self, id: &str) -> Result<f64, AchievementError> {
        if self.unlocked.contains_key(id) { return Ok(1.0); }
        let ach = self.achievements.get(id)
            .ok_or_else(|| AchievementError::AchievementNotFound(id.to_string()))?;
        Ok(ach.criteria.progress(&self.stats))
    }

    /// Set rarity data (total players and per-achievement unlock counts).
    pub fn set_rarity_data(&mut self, total_players: u64, counts: HashMap<String, u64>) {
        self.total_players = total_players.max(1);
        self.unlock_counts = counts;
    }

    /// Get rarity as percentage of players who unlocked.
    pub fn rarity(&self, id: &str) -> Option<f64> {
        let count = self.unlock_counts.get(id).copied().unwrap_or(0);
        Some(count as f64 / self.total_players as f64 * 100.0)
    }

    pub fn by_category(&self, cat: AchievementCategory) -> Vec<&Achievement> {
        let mut results: Vec<&Achievement> = self.achievements.values()
            .filter(|a| a.category == cat)
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    pub fn visible_achievements(&self) -> Vec<&Achievement> {
        let mut results: Vec<&Achievement> = self.achievements.values()
            .filter(|a| !a.hidden || self.unlocked.contains_key(&a.id))
            .collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    pub fn all_achievements(&self) -> Vec<&Achievement> {
        let mut results: Vec<&Achievement> = self.achievements.values().collect();
        results.sort_by(|a, b| a.id.cmp(&b.id));
        results
    }

    pub fn unlocked_count(&self) -> usize { self.unlocked.len() }
    pub fn total_count(&self) -> usize { self.achievements.len() }

    pub fn completion_percentage(&self) -> f64 {
        if self.achievements.is_empty() { return 0.0; }
        self.unlocked.len() as f64 / self.achievements.len() as f64 * 100.0
    }

    pub fn rewards_for(&self, id: &str) -> Option<&[AchievementReward]> {
        self.achievements.get(id).map(|a| a.rewards.as_slice())
    }

    /// Serialize progress (unlocks + stats).
    pub fn serialize_progress(&self) -> SerializedProgress {
        let mut unlocks: Vec<(String, u64)> = self.unlocked.values()
            .map(|r| (r.achievement_id.clone(), r.unlocked_at_ms))
            .collect();
        unlocks.sort_by(|a, b| a.0.cmp(&b.0));
        SerializedProgress {
            unlocks,
            stats: self.stats.serialize(),
        }
    }

    pub fn deserialize_progress(&mut self, data: &SerializedProgress) {
        self.stats = StatsBackend::deserialize(&data.stats);
        self.unlocked.clear();
        for (id, ts) in &data.unlocks {
            self.unlocked.insert(id.clone(), UnlockRecord {
                achievement_id: id.clone(),
                unlocked_at_ms: *ts,
            });
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SerializedProgress {
    pub unlocks: Vec<(String, u64)>,
    pub stats: SerializedStats,
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> AchievementSystem {
        let mut sys = AchievementSystem::new();
        sys.register(
            Achievement::new("first_kill", "First Blood", "Kill your first enemy",
                Criteria::Counter { key: "kills".to_string(), threshold: 1 })
                .with_category(AchievementCategory::Combat)
                .with_reward(AchievementReward::Title("Warrior".to_string()))
        ).unwrap();
        sys.register(
            Achievement::new("explorer", "Explorer", "Visit all regions",
                Criteria::MultiFlag { keys: vec![
                    "visited_forest".to_string(),
                    "visited_desert".to_string(),
                    "visited_mountain".to_string(),
                ]})
                .with_category(AchievementCategory::Exploration)
        ).unwrap();
        sys.register(
            Achievement::new("crafter", "Master Crafter", "Craft 100 items",
                Criteria::Counter { key: "items_crafted".to_string(), threshold: 100 })
                .with_category(AchievementCategory::Crafting)
        ).unwrap();
        sys.register(
            Achievement::new("secret", "Hidden Secret", "Find the secret",
                Criteria::Flag { key: "found_secret".to_string() })
                .with_hidden(true)
                .with_category(AchievementCategory::Exploration)
                .with_reward(AchievementReward::Badge("secret_badge".to_string()))
        ).unwrap();
        sys.register(
            Achievement::new("social", "Friendly", "Make a friend",
                Criteria::Flag { key: "made_friend".to_string() })
                .with_category(AchievementCategory::Social)
                .with_reward(AchievementReward::Currency(500))
        ).unwrap();
        sys
    }

    #[test]
    fn register_achievement() {
        let sys = setup();
        assert_eq!(sys.total_count(), 5);
    }

    #[test]
    fn duplicate_achievement() {
        let mut sys = setup();
        let err = sys.register(
            Achievement::new("first_kill", "Dup", "Dup", Criteria::Flag { key: "x".to_string() })
        ).unwrap_err();
        assert!(matches!(err, AchievementError::DuplicateAchievement(_)));
    }

    #[test]
    fn counter_progress() {
        let mut sys = setup();
        sys.stats_mut().increment_counter("items_crafted", 50);
        let prog = sys.progress("crafter").unwrap();
        assert!((prog - 0.5).abs() < 1e-6);
    }

    #[test]
    fn counter_unlock() {
        let mut sys = setup();
        let unlocked = sys.increment_and_check("kills", 1, 1000);
        assert!(unlocked.contains(&"first_kill".to_string()));
        assert!(sys.is_unlocked("first_kill"));
    }

    #[test]
    fn flag_unlock() {
        let mut sys = setup();
        let unlocked = sys.set_flag_and_check("made_friend", 2000);
        assert!(unlocked.contains(&"social".to_string()));
    }

    #[test]
    fn multi_flag_partial() {
        let mut sys = setup();
        sys.set_flag_and_check("visited_forest", 100);
        sys.set_flag_and_check("visited_desert", 200);
        let prog = sys.progress("explorer").unwrap();
        assert!((prog - 2.0 / 3.0).abs() < 1e-6);
        assert!(!sys.is_unlocked("explorer"));
    }

    #[test]
    fn multi_flag_complete() {
        let mut sys = setup();
        sys.set_flag_and_check("visited_forest", 100);
        sys.set_flag_and_check("visited_desert", 200);
        let unlocked = sys.set_flag_and_check("visited_mountain", 300);
        assert!(unlocked.contains(&"explorer".to_string()));
    }

    #[test]
    fn already_unlocked_not_duplicated() {
        let mut sys = setup();
        sys.increment_and_check("kills", 1, 1000);
        let again = sys.increment_and_check("kills", 1, 2000);
        assert!(!again.contains(&"first_kill".to_string()));
    }

    #[test]
    fn unlocked_progress_is_one() {
        let mut sys = setup();
        sys.increment_and_check("kills", 1, 1000);
        assert!((sys.progress("first_kill").unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unlock_timestamp() {
        let mut sys = setup();
        sys.increment_and_check("kills", 1, 12345);
        let record = sys.unlock_record("first_kill").unwrap();
        assert_eq!(record.unlocked_at_ms, 12345);
    }

    #[test]
    fn hidden_achievement_visibility() {
        let sys = setup();
        let visible = sys.visible_achievements();
        // "secret" is hidden and not unlocked → not visible
        assert!(!visible.iter().any(|a| a.id == "secret"));
    }

    #[test]
    fn hidden_becomes_visible_on_unlock() {
        let mut sys = setup();
        sys.set_flag_and_check("found_secret", 500);
        let visible = sys.visible_achievements();
        assert!(visible.iter().any(|a| a.id == "secret"));
    }

    #[test]
    fn by_category() {
        let sys = setup();
        let combat = sys.by_category(AchievementCategory::Combat);
        assert_eq!(combat.len(), 1);
        assert_eq!(combat[0].id, "first_kill");
        let explore = sys.by_category(AchievementCategory::Exploration);
        assert_eq!(explore.len(), 2);
    }

    #[test]
    fn completion_percentage() {
        let mut sys = setup();
        assert!((sys.completion_percentage() - 0.0).abs() < 1e-6);
        sys.increment_and_check("kills", 1, 100);
        // 1 out of 5 = 20%
        assert!((sys.completion_percentage() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn rarity_calculation() {
        let mut sys = setup();
        let mut counts = HashMap::new();
        counts.insert("first_kill".to_string(), 900);
        counts.insert("explorer".to_string(), 50);
        sys.set_rarity_data(1000, counts);
        let r1 = sys.rarity("first_kill").unwrap();
        assert!((r1 - 90.0).abs() < 1e-6);
        let r2 = sys.rarity("explorer").unwrap();
        assert!((r2 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn rewards_lookup() {
        let sys = setup();
        let rewards = sys.rewards_for("first_kill").unwrap();
        assert_eq!(rewards.len(), 1);
        assert!(matches!(&rewards[0], AchievementReward::Title(t) if t == "Warrior"));
    }

    #[test]
    fn serialize_roundtrip() {
        let mut sys = setup();
        sys.increment_and_check("kills", 5, 100);
        sys.set_flag_and_check("visited_forest", 200);
        let data = sys.serialize_progress();
        // Reset and restore
        let mut sys2 = setup();
        sys2.deserialize_progress(&data);
        assert!(sys2.is_unlocked("first_kill"));
        assert_eq!(sys2.stats().get_counter("kills"), 5);
        assert!(sys2.stats().get_flag("visited_forest"));
    }

    #[test]
    fn stats_backend_operations() {
        let mut stats = StatsBackend::new();
        stats.increment_counter("a", 10);
        stats.increment_counter("a", 5);
        assert_eq!(stats.get_counter("a"), 15);
        stats.set_counter("a", 100);
        assert_eq!(stats.get_counter("a"), 100);
        assert_eq!(stats.get_counter("missing"), 0);
        stats.set_flag("done", true);
        assert!(stats.get_flag("done"));
        assert!(!stats.get_flag("other"));
    }

    #[test]
    fn stats_serialization() {
        let mut stats = StatsBackend::new();
        stats.set_counter("x", 42);
        stats.set_flag("y", true);
        let data = stats.serialize();
        let stats2 = StatsBackend::deserialize(&data);
        assert_eq!(stats2.get_counter("x"), 42);
        assert!(stats2.get_flag("y"));
    }

    #[test]
    fn achievement_not_found() {
        let sys = setup();
        let err = sys.progress("nonexistent").unwrap_err();
        assert!(matches!(err, AchievementError::AchievementNotFound(_)));
    }

    #[test]
    fn icon_and_builder() {
        let a = Achievement::new("test", "Test", "Testing",
            Criteria::Flag { key: "t".to_string() })
            .with_icon("trophy.png");
        assert_eq!(a.icon, "trophy.png");
    }

    #[test]
    fn empty_multi_flag_is_met() {
        let stats = StatsBackend::new();
        let criteria = Criteria::MultiFlag { keys: vec![] };
        assert!(criteria.is_met(&stats));
        assert!((criteria.progress(&stats) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_threshold_counter() {
        let stats = StatsBackend::new();
        let criteria = Criteria::Counter { key: "x".to_string(), threshold: 0 };
        assert!(criteria.is_met(&stats));
        assert!((criteria.progress(&stats) - 1.0).abs() < 1e-6);
    }
}
