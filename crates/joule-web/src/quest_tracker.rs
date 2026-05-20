//! Quest/objective tracking system — quests, objectives, rewards, chains.
//!
//! Replaces quest-log.js / RPGMaker-quest plugins with pure Rust.
//! Quest definitions, objective types, quest states, chains,
//! time-limited quests, rewards, and quest log management.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuestError {
    QuestNotFound(u64),
    QuestNotActive(u64),
    QuestAlreadyActive(u64),
    QuestNotAvailable(u64),
    QuestNotCompleted(u64),
    ObjectiveNotFound { quest_id: u64, obj_idx: usize },
    PrerequisiteNotMet { quest_id: u64, requires: u64 },
    QuestExpired(u64),
    DuplicateQuest(u64),
}

impl fmt::Display for QuestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QuestNotFound(id) => write!(f, "quest not found: {id}"),
            Self::QuestNotActive(id) => write!(f, "quest not active: {id}"),
            Self::QuestAlreadyActive(id) => write!(f, "quest already active: {id}"),
            Self::QuestNotAvailable(id) => write!(f, "quest not available: {id}"),
            Self::QuestNotCompleted(id) => write!(f, "quest not completed: {id}"),
            Self::ObjectiveNotFound { quest_id, obj_idx } => {
                write!(f, "objective {obj_idx} not found in quest {quest_id}")
            }
            Self::PrerequisiteNotMet { quest_id, requires } => {
                write!(f, "quest {quest_id} requires quest {requires} to be turned in")
            }
            Self::QuestExpired(id) => write!(f, "quest {id} has expired"),
            Self::DuplicateQuest(id) => write!(f, "duplicate quest: {id}"),
        }
    }
}

impl std::error::Error for QuestError {}

// ── Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuestState {
    Available,
    Active,
    Completed,
    Failed,
    TurnedIn,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Objective {
    KillCount { target: String, current: u32, required: u32 },
    CollectItem { item_id: u64, current: u32, required: u32 },
    TalkTo { npc: String, done: bool },
    ReachLocation { x: f64, y: f64, radius: f64, reached: bool },
    Custom { key: String, done: bool },
}

impl Objective {
    pub fn is_complete(&self) -> bool {
        match self {
            Self::KillCount { current, required, .. } => current >= required,
            Self::CollectItem { current, required, .. } => current >= required,
            Self::TalkTo { done, .. } => *done,
            Self::ReachLocation { reached, .. } => *reached,
            Self::Custom { done, .. } => *done,
        }
    }

    pub fn progress_fraction(&self) -> f64 {
        match self {
            Self::KillCount { current, required, .. } => {
                if *required == 0 { 1.0 } else { (*current as f64 / *required as f64).min(1.0) }
            }
            Self::CollectItem { current, required, .. } => {
                if *required == 0 { 1.0 } else { (*current as f64 / *required as f64).min(1.0) }
            }
            Self::TalkTo { done, .. } => if *done { 1.0 } else { 0.0 },
            Self::ReachLocation { reached, .. } => if *reached { 1.0 } else { 0.0 },
            Self::Custom { done, .. } => if *done { 1.0 } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reward {
    pub xp: u64,
    pub currency: u64,
    pub items: Vec<(u64, u32)>, // (item_id, quantity)
}

impl Reward {
    pub fn new() -> Self {
        Self { xp: 0, currency: 0, items: Vec::new() }
    }
    pub fn with_xp(mut self, xp: u64) -> Self { self.xp = xp; self }
    pub fn with_currency(mut self, c: u64) -> Self { self.currency = c; self }
    pub fn with_item(mut self, item_id: u64, qty: u32) -> Self {
        self.items.push((item_id, qty)); self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Quest {
    pub id: u64,
    pub name: String,
    pub description: String,
    pub objectives: Vec<Objective>,
    pub reward: Reward,
    pub state: QuestState,
    pub prerequisites: Vec<u64>,
    pub deadline_ms: Option<u64>,
    pub accepted_at_ms: Option<u64>,
}

impl Quest {
    pub fn new(id: u64, name: &str, description: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            description: description.to_string(),
            objectives: Vec::new(),
            reward: Reward::new(),
            state: QuestState::Available,
            prerequisites: Vec::new(),
            deadline_ms: None,
            accepted_at_ms: None,
        }
    }

    pub fn with_objective(mut self, obj: Objective) -> Self {
        self.objectives.push(obj); self
    }

    pub fn with_reward(mut self, reward: Reward) -> Self {
        self.reward = reward; self
    }

    pub fn with_prerequisite(mut self, quest_id: u64) -> Self {
        self.prerequisites.push(quest_id); self
    }

    pub fn with_deadline(mut self, duration_ms: u64) -> Self {
        self.deadline_ms = Some(duration_ms); self
    }

    pub fn all_objectives_complete(&self) -> bool {
        !self.objectives.is_empty() && self.objectives.iter().all(|o| o.is_complete())
    }

    pub fn overall_progress(&self) -> f64 {
        if self.objectives.is_empty() { return 0.0; }
        let sum: f64 = self.objectives.iter().map(|o| o.progress_fraction()).sum();
        sum / self.objectives.len() as f64
    }

    pub fn is_expired(&self, current_time_ms: u64) -> bool {
        if let (Some(deadline), Some(accepted)) = (self.deadline_ms, self.accepted_at_ms) {
            current_time_ms > accepted + deadline
        } else {
            false
        }
    }
}

// ── Quest Tracker ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct QuestTracker {
    quests: HashMap<u64, Quest>,
}

impl QuestTracker {
    pub fn new() -> Self {
        Self { quests: HashMap::new() }
    }

    pub fn register_quest(&mut self, quest: Quest) -> Result<(), QuestError> {
        if self.quests.contains_key(&quest.id) {
            return Err(QuestError::DuplicateQuest(quest.id));
        }
        self.quests.insert(quest.id, quest);
        Ok(())
    }

    pub fn get_quest(&self, id: u64) -> Result<&Quest, QuestError> {
        self.quests.get(&id).ok_or(QuestError::QuestNotFound(id))
    }

    pub fn accept_quest(&mut self, id: u64, current_time_ms: u64) -> Result<(), QuestError> {
        let quest = self.quests.get(&id).ok_or(QuestError::QuestNotFound(id))?;
        if quest.state != QuestState::Available {
            return Err(QuestError::QuestNotAvailable(id));
        }
        // Check prerequisites
        for pre_id in quest.prerequisites.clone() {
            let pre = self.quests.get(&pre_id).ok_or(QuestError::QuestNotFound(pre_id))?;
            if pre.state != QuestState::TurnedIn {
                return Err(QuestError::PrerequisiteNotMet { quest_id: id, requires: pre_id });
            }
        }
        let quest = self.quests.get_mut(&id).unwrap();
        quest.state = QuestState::Active;
        quest.accepted_at_ms = Some(current_time_ms);
        Ok(())
    }

    pub fn update_kill(&mut self, quest_id: u64, target: &str, count: u32) -> Result<bool, QuestError> {
        let quest = self.quests.get_mut(&quest_id).ok_or(QuestError::QuestNotFound(quest_id))?;
        if quest.state != QuestState::Active {
            return Err(QuestError::QuestNotActive(quest_id));
        }
        let mut any_updated = false;
        for obj in &mut quest.objectives {
            if let Objective::KillCount { target: t, current, required, .. } = obj {
                if t == target {
                    *current = (*current + count).min(*required);
                    any_updated = true;
                }
            }
        }
        // Auto-complete check
        if quest.all_objectives_complete() {
            quest.state = QuestState::Completed;
        }
        Ok(any_updated)
    }

    pub fn update_collect(&mut self, quest_id: u64, item_id: u64, count: u32) -> Result<bool, QuestError> {
        let quest = self.quests.get_mut(&quest_id).ok_or(QuestError::QuestNotFound(quest_id))?;
        if quest.state != QuestState::Active {
            return Err(QuestError::QuestNotActive(quest_id));
        }
        let mut any_updated = false;
        for obj in &mut quest.objectives {
            if let Objective::CollectItem { item_id: iid, current, required, .. } = obj {
                if *iid == item_id {
                    *current = (*current + count).min(*required);
                    any_updated = true;
                }
            }
        }
        if quest.all_objectives_complete() {
            quest.state = QuestState::Completed;
        }
        Ok(any_updated)
    }

    pub fn update_talk(&mut self, quest_id: u64, npc: &str) -> Result<bool, QuestError> {
        let quest = self.quests.get_mut(&quest_id).ok_or(QuestError::QuestNotFound(quest_id))?;
        if quest.state != QuestState::Active {
            return Err(QuestError::QuestNotActive(quest_id));
        }
        let mut any_updated = false;
        for obj in &mut quest.objectives {
            if let Objective::TalkTo { npc: n, done } = obj {
                if n == npc && !*done {
                    *done = true;
                    any_updated = true;
                }
            }
        }
        if quest.all_objectives_complete() {
            quest.state = QuestState::Completed;
        }
        Ok(any_updated)
    }

    pub fn update_location(&mut self, quest_id: u64, px: f64, py: f64) -> Result<bool, QuestError> {
        let quest = self.quests.get_mut(&quest_id).ok_or(QuestError::QuestNotFound(quest_id))?;
        if quest.state != QuestState::Active {
            return Err(QuestError::QuestNotActive(quest_id));
        }
        let mut any_updated = false;
        for obj in &mut quest.objectives {
            if let Objective::ReachLocation { x, y, radius, reached } = obj {
                if !*reached {
                    let dx = px - *x;
                    let dy = py - *y;
                    if dx * dx + dy * dy <= *radius * *radius {
                        *reached = true;
                        any_updated = true;
                    }
                }
            }
        }
        if quest.all_objectives_complete() {
            quest.state = QuestState::Completed;
        }
        Ok(any_updated)
    }

    pub fn update_custom(&mut self, quest_id: u64, key: &str) -> Result<bool, QuestError> {
        let quest = self.quests.get_mut(&quest_id).ok_or(QuestError::QuestNotFound(quest_id))?;
        if quest.state != QuestState::Active {
            return Err(QuestError::QuestNotActive(quest_id));
        }
        let mut any_updated = false;
        for obj in &mut quest.objectives {
            if let Objective::Custom { key: k, done } = obj {
                if k == key && !*done {
                    *done = true;
                    any_updated = true;
                }
            }
        }
        if quest.all_objectives_complete() {
            quest.state = QuestState::Completed;
        }
        Ok(any_updated)
    }

    pub fn turn_in(&mut self, id: u64) -> Result<&Reward, QuestError> {
        let quest = self.quests.get(&id).ok_or(QuestError::QuestNotFound(id))?;
        if quest.state != QuestState::Completed {
            return Err(QuestError::QuestNotCompleted(id));
        }
        let quest = self.quests.get_mut(&id).unwrap();
        quest.state = QuestState::TurnedIn;
        Ok(&self.quests[&id].reward)
    }

    pub fn fail_quest(&mut self, id: u64) -> Result<(), QuestError> {
        let quest = self.quests.get_mut(&id).ok_or(QuestError::QuestNotFound(id))?;
        if quest.state != QuestState::Active {
            return Err(QuestError::QuestNotActive(id));
        }
        quest.state = QuestState::Failed;
        Ok(())
    }

    /// Check and fail expired quests.
    pub fn check_expirations(&mut self, current_time_ms: u64) -> Vec<u64> {
        let active_ids: Vec<u64> = self.quests.values()
            .filter(|q| q.state == QuestState::Active)
            .map(|q| q.id)
            .collect();
        let mut expired = Vec::new();
        for id in active_ids {
            let quest = self.quests.get(&id).unwrap();
            if quest.is_expired(current_time_ms) {
                self.quests.get_mut(&id).unwrap().state = QuestState::Failed;
                expired.push(id);
            }
        }
        expired
    }

    pub fn active_quests(&self) -> Vec<&Quest> {
        let mut result: Vec<&Quest> = self.quests.values()
            .filter(|q| q.state == QuestState::Active)
            .collect();
        result.sort_by_key(|q| q.id);
        result
    }

    pub fn completed_quests(&self) -> Vec<&Quest> {
        let mut result: Vec<&Quest> = self.quests.values()
            .filter(|q| q.state == QuestState::Completed || q.state == QuestState::TurnedIn)
            .collect();
        result.sort_by_key(|q| q.id);
        result
    }

    pub fn available_quests(&self) -> Vec<&Quest> {
        let mut result: Vec<&Quest> = self.quests.values()
            .filter(|q| q.state == QuestState::Available)
            .collect();
        result.sort_by_key(|q| q.id);
        result
    }

    pub fn quest_count(&self) -> usize { self.quests.len() }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn slay_quest() -> Quest {
        Quest::new(1, "Dragon Slayer", "Kill 5 dragons")
            .with_objective(Objective::KillCount {
                target: "dragon".to_string(), current: 0, required: 5,
            })
            .with_reward(Reward::new().with_xp(1000).with_currency(500))
    }

    fn fetch_quest() -> Quest {
        Quest::new(2, "Herb Gathering", "Collect 10 herbs")
            .with_objective(Objective::CollectItem {
                item_id: 42, current: 0, required: 10,
            })
            .with_reward(Reward::new().with_xp(200).with_item(99, 3))
    }

    fn chain_quest() -> Quest {
        Quest::new(3, "Dragon's Lair", "Enter the lair")
            .with_prerequisite(1) // requires Dragon Slayer
            .with_objective(Objective::ReachLocation {
                x: 100.0, y: 200.0, radius: 10.0, reached: false,
            })
            .with_reward(Reward::new().with_xp(2000))
    }

    #[test]
    fn register_quest() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        assert_eq!(tracker.quest_count(), 1);
    }

    #[test]
    fn duplicate_quest() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        let err = tracker.register_quest(slay_quest()).unwrap_err();
        assert!(matches!(err, QuestError::DuplicateQuest(1)));
    }

    #[test]
    fn accept_quest() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        let q = tracker.get_quest(1).unwrap();
        assert_eq!(q.state, QuestState::Active);
    }

    #[test]
    fn accept_already_active() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        let err = tracker.accept_quest(1, 0).unwrap_err();
        assert!(matches!(err, QuestError::QuestNotAvailable(1)));
    }

    #[test]
    fn kill_progress() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        tracker.update_kill(1, "dragon", 3).unwrap();
        let q = tracker.get_quest(1).unwrap();
        let progress = q.overall_progress();
        assert!((progress - 0.6).abs() < 1e-6);
        assert_eq!(q.state, QuestState::Active);
    }

    #[test]
    fn kill_complete() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        tracker.update_kill(1, "dragon", 5).unwrap();
        let q = tracker.get_quest(1).unwrap();
        assert_eq!(q.state, QuestState::Completed);
    }

    #[test]
    fn kill_overcap() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        tracker.update_kill(1, "dragon", 100).unwrap();
        let q = tracker.get_quest(1).unwrap();
        if let Objective::KillCount { current, required, .. } = &q.objectives[0] {
            assert_eq!(*current, *required);
        }
    }

    #[test]
    fn collect_progress() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(fetch_quest()).unwrap();
        tracker.accept_quest(2, 0).unwrap();
        tracker.update_collect(2, 42, 7).unwrap();
        let q = tracker.get_quest(2).unwrap();
        assert!((q.overall_progress() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn turn_in() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        tracker.update_kill(1, "dragon", 5).unwrap();
        let reward = tracker.turn_in(1).unwrap();
        assert_eq!(reward.xp, 1000);
        assert_eq!(reward.currency, 500);
        let q = tracker.get_quest(1).unwrap();
        assert_eq!(q.state, QuestState::TurnedIn);
    }

    #[test]
    fn turn_in_not_completed() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        let err = tracker.turn_in(1).unwrap_err();
        assert!(matches!(err, QuestError::QuestNotCompleted(1)));
    }

    #[test]
    fn quest_chain() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.register_quest(chain_quest()).unwrap();
        // Can't accept chain quest without completing prerequisite
        let err = tracker.accept_quest(3, 0).unwrap_err();
        assert!(matches!(err, QuestError::PrerequisiteNotMet { .. }));
        // Complete the prerequisite
        tracker.accept_quest(1, 0).unwrap();
        tracker.update_kill(1, "dragon", 5).unwrap();
        tracker.turn_in(1).unwrap();
        // Now chain quest is available
        tracker.accept_quest(3, 0).unwrap();
        assert_eq!(tracker.get_quest(3).unwrap().state, QuestState::Active);
    }

    #[test]
    fn fail_quest() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        tracker.fail_quest(1).unwrap();
        assert_eq!(tracker.get_quest(1).unwrap().state, QuestState::Failed);
    }

    #[test]
    fn time_limited_quest() {
        let mut tracker = QuestTracker::new();
        let mut q = slay_quest().with_deadline(60_000); // 60 seconds
        tracker.register_quest(q).unwrap();
        tracker.accept_quest(1, 1000).unwrap();
        // Not expired at 50 seconds
        let expired = tracker.check_expirations(51_000);
        assert!(expired.is_empty());
        // Expired at 62 seconds
        let expired = tracker.check_expirations(62_000);
        assert_eq!(expired, vec![1]);
        assert_eq!(tracker.get_quest(1).unwrap().state, QuestState::Failed);
    }

    #[test]
    fn talk_to_objective() {
        let mut tracker = QuestTracker::new();
        let q = Quest::new(10, "Meet the Elder", "Talk to Elder Grim")
            .with_objective(Objective::TalkTo { npc: "Elder Grim".to_string(), done: false })
            .with_reward(Reward::new().with_xp(50));
        tracker.register_quest(q).unwrap();
        tracker.accept_quest(10, 0).unwrap();
        tracker.update_talk(10, "Elder Grim").unwrap();
        assert_eq!(tracker.get_quest(10).unwrap().state, QuestState::Completed);
    }

    #[test]
    fn reach_location() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(chain_quest()).unwrap();
        // Skip prerequisite for this test by directly manipulating
        // Register slay_quest as turned in
        let mut sq = slay_quest();
        sq.state = QuestState::TurnedIn;
        tracker.register_quest(sq).unwrap();
        tracker.accept_quest(3, 0).unwrap();
        // Too far
        tracker.update_location(3, 50.0, 50.0).unwrap();
        assert_eq!(tracker.get_quest(3).unwrap().state, QuestState::Active);
        // Within radius
        tracker.update_location(3, 105.0, 200.0).unwrap();
        assert_eq!(tracker.get_quest(3).unwrap().state, QuestState::Completed);
    }

    #[test]
    fn custom_objective() {
        let mut tracker = QuestTracker::new();
        let q = Quest::new(20, "Secret", "Do the secret thing")
            .with_objective(Objective::Custom { key: "secret_done".to_string(), done: false })
            .with_reward(Reward::new().with_xp(999));
        tracker.register_quest(q).unwrap();
        tracker.accept_quest(20, 0).unwrap();
        tracker.update_custom(20, "secret_done").unwrap();
        assert_eq!(tracker.get_quest(20).unwrap().state, QuestState::Completed);
    }

    #[test]
    fn active_quests_list() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.register_quest(fetch_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        tracker.accept_quest(2, 0).unwrap();
        assert_eq!(tracker.active_quests().len(), 2);
    }

    #[test]
    fn available_quests_list() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.register_quest(fetch_quest()).unwrap();
        assert_eq!(tracker.available_quests().len(), 2);
        tracker.accept_quest(1, 0).unwrap();
        assert_eq!(tracker.available_quests().len(), 1);
    }

    #[test]
    fn multi_objective_quest() {
        let mut tracker = QuestTracker::new();
        let q = Quest::new(30, "Multi", "Do multiple things")
            .with_objective(Objective::KillCount { target: "goblin".to_string(), current: 0, required: 3 })
            .with_objective(Objective::CollectItem { item_id: 5, current: 0, required: 2 })
            .with_reward(Reward::new().with_xp(500));
        tracker.register_quest(q).unwrap();
        tracker.accept_quest(30, 0).unwrap();
        tracker.update_kill(30, "goblin", 3).unwrap();
        assert_eq!(tracker.get_quest(30).unwrap().state, QuestState::Active);
        tracker.update_collect(30, 5, 2).unwrap();
        assert_eq!(tracker.get_quest(30).unwrap().state, QuestState::Completed);
    }

    #[test]
    fn update_wrong_target() {
        let mut tracker = QuestTracker::new();
        tracker.register_quest(slay_quest()).unwrap();
        tracker.accept_quest(1, 0).unwrap();
        let updated = tracker.update_kill(1, "goblin", 5).unwrap();
        assert!(!updated);
    }

    #[test]
    fn quest_not_found() {
        let tracker = QuestTracker::new();
        let err = tracker.get_quest(999).unwrap_err();
        assert!(matches!(err, QuestError::QuestNotFound(999)));
    }
}
