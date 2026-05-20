//! Emote/reaction system — emotes, sets, registry, usage tracking, combos.
//!
//! Replaces BetterTTV / 7TV / Twitch emote APIs with pure Rust.
//! Emote definitions, categorized sets, usage statistics, favorites,
//! combo detection, cooldowns, search, and custom emote metadata.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmoteError {
    EmoteNotFound(String),
    DuplicateEmote(String),
    OnCooldown { user: String, remaining_secs: u64 },
    SetNotFound(String),
    DuplicateSet(String),
}

impl fmt::Display for EmoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmoteNotFound(id) => write!(f, "emote not found: {id}"),
            Self::DuplicateEmote(id) => write!(f, "duplicate emote: {id}"),
            Self::OnCooldown { user, remaining_secs } => {
                write!(f, "{user} on cooldown: {remaining_secs}s remaining")
            }
            Self::SetNotFound(id) => write!(f, "emote set not found: {id}"),
            Self::DuplicateSet(id) => write!(f, "duplicate emote set: {id}"),
        }
    }
}

impl std::error::Error for EmoteError {}

// ── EmoteCategory ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmoteCategory {
    Happy,
    Sad,
    Angry,
    Surprised,
    Love,
    Action,
    Meme,
    Custom,
}

impl fmt::Display for EmoteCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Happy => write!(f, "happy"),
            Self::Sad => write!(f, "sad"),
            Self::Angry => write!(f, "angry"),
            Self::Surprised => write!(f, "surprised"),
            Self::Love => write!(f, "love"),
            Self::Action => write!(f, "action"),
            Self::Meme => write!(f, "meme"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

// ── Emote ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Emote {
    pub id: String,
    pub name: String,
    pub category: EmoteCategory,
    pub is_animated: bool,
    pub creator: Option<String>,
    pub created_at: u64,
}

impl Emote {
    pub fn new(id: &str, name: &str, category: EmoteCategory) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            category,
            is_animated: false,
            creator: None,
            created_at: 0,
        }
    }

    pub fn animated(mut self) -> Self {
        self.is_animated = true;
        self
    }

    pub fn with_creator(mut self, creator: &str) -> Self {
        self.creator = Some(creator.to_string());
        self
    }

    pub fn with_created_at(mut self, ts: u64) -> Self {
        self.created_at = ts;
        self
    }
}

impl fmt::Display for Emote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let anim = if self.is_animated { " (animated)" } else { "" };
        write!(f, ":{}: [{}{}]", self.name, self.category, anim)
    }
}

// ── EmoteSet ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EmoteSet {
    pub id: String,
    pub name: String,
    pub emote_ids: Vec<String>,
}

impl EmoteSet {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            emote_ids: Vec::new(),
        }
    }

    pub fn add_emote(&mut self, emote_id: &str) {
        if !self.emote_ids.contains(&emote_id.to_string()) {
            self.emote_ids.push(emote_id.to_string());
        }
    }

    pub fn remove_emote(&mut self, emote_id: &str) {
        self.emote_ids.retain(|e| e != emote_id);
    }

    pub fn count(&self) -> usize {
        self.emote_ids.len()
    }
}

impl fmt::Display for EmoteSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({} emotes)", self.name, self.count())
    }
}

// ── EmoteCombo ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EmoteCombo {
    pub emote_id: String,
    pub count: usize,
    pub participants: HashSet<String>,
}

// ── EmoteStats ──────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct EmoteStats {
    pub total_uses: u64,
    pub unique_users: usize,
    pub unique_emotes_used: usize,
}

// ── EmoteRegistry ───────────────────────────────────────────────

#[derive(Debug)]
pub struct EmoteRegistry {
    emotes: HashMap<String, Emote>,
    sets: HashMap<String, EmoteSet>,
    usage_counts: HashMap<String, u64>,
    user_favorites: HashMap<String, Vec<String>>,
    cooldowns: HashMap<String, u64>,
    cooldown_duration: u64,
    combo_window: VecDeque<(String, String, u64)>,
    combo_window_secs: u64,
    users_who_used: HashMap<String, HashSet<String>>,
}

impl EmoteRegistry {
    pub fn new(cooldown_duration: u64, combo_window_secs: u64) -> Self {
        Self {
            emotes: HashMap::new(),
            sets: HashMap::new(),
            usage_counts: HashMap::new(),
            user_favorites: HashMap::new(),
            cooldowns: HashMap::new(),
            cooldown_duration,
            combo_window: VecDeque::new(),
            combo_window_secs,
            users_who_used: HashMap::new(),
        }
    }

    pub fn register_emote(&mut self, emote: Emote) -> Result<(), EmoteError> {
        if self.emotes.contains_key(&emote.id) {
            return Err(EmoteError::DuplicateEmote(emote.id.clone()));
        }
        self.emotes.insert(emote.id.clone(), emote);
        Ok(())
    }

    pub fn remove_emote(&mut self, id: &str) -> Result<Emote, EmoteError> {
        self.emotes.remove(id).ok_or_else(|| EmoteError::EmoteNotFound(id.to_string()))
    }

    pub fn get_emote(&self, id: &str) -> Option<&Emote> {
        self.emotes.get(id)
    }

    pub fn register_set(&mut self, set: EmoteSet) -> Result<(), EmoteError> {
        if self.sets.contains_key(&set.id) {
            return Err(EmoteError::DuplicateSet(set.id.clone()));
        }
        self.sets.insert(set.id.clone(), set);
        Ok(())
    }

    pub fn get_set(&self, id: &str) -> Option<&EmoteSet> {
        self.sets.get(id)
    }

    /// Use an emote. Enforces cooldown and tracks usage/combo.
    pub fn use_emote(&mut self, user_id: &str, emote_id: &str, timestamp: u64) -> Result<(), EmoteError> {
        if !self.emotes.contains_key(emote_id) {
            return Err(EmoteError::EmoteNotFound(emote_id.to_string()));
        }
        // Cooldown check
        if let Some(&last_use) = self.cooldowns.get(user_id) {
            let elapsed = timestamp.saturating_sub(last_use);
            if elapsed < self.cooldown_duration {
                return Err(EmoteError::OnCooldown {
                    user: user_id.to_string(),
                    remaining_secs: self.cooldown_duration - elapsed,
                });
            }
        }
        self.cooldowns.insert(user_id.to_string(), timestamp);
        *self.usage_counts.entry(emote_id.to_string()).or_default() += 1;
        self.users_who_used.entry(emote_id.to_string()).or_default().insert(user_id.to_string());

        // Combo tracking
        self.combo_window.push_back((user_id.to_string(), emote_id.to_string(), timestamp));
        // Evict old entries
        while let Some(front) = self.combo_window.front() {
            if timestamp.saturating_sub(front.2) > self.combo_window_secs {
                self.combo_window.pop_front();
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Detect current combo (same emote used by multiple users in window).
    pub fn detect_combo(&self) -> Option<EmoteCombo> {
        let mut counts: HashMap<&str, HashSet<&str>> = HashMap::new();
        for (user, emote, _) in &self.combo_window {
            counts.entry(emote.as_str()).or_default().insert(user.as_str());
        }
        counts.into_iter()
            .filter(|(_, users)| users.len() >= 2)
            .max_by_key(|(_, users)| users.len())
            .map(|(emote_id, users)| EmoteCombo {
                emote_id: emote_id.to_string(),
                count: users.len(),
                participants: users.into_iter().map(|s| s.to_string()).collect(),
            })
    }

    pub fn search_by_name(&self, query: &str) -> Vec<&Emote> {
        let lower = query.to_lowercase();
        self.emotes.values().filter(|e| e.name.to_lowercase().contains(&lower)).collect()
    }

    pub fn search_by_category(&self, category: EmoteCategory) -> Vec<&Emote> {
        self.emotes.values().filter(|e| e.category == category).collect()
    }

    pub fn popular_emotes(&self, top_n: usize) -> Vec<(&str, u64)> {
        let mut sorted: Vec<_> = self.usage_counts.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(top_n);
        sorted
    }

    pub fn add_favorite(&mut self, user_id: &str, emote_id: &str) {
        let favs = self.user_favorites.entry(user_id.to_string()).or_default();
        if !favs.contains(&emote_id.to_string()) {
            favs.push(emote_id.to_string());
        }
    }

    pub fn remove_favorite(&mut self, user_id: &str, emote_id: &str) {
        if let Some(favs) = self.user_favorites.get_mut(user_id) {
            favs.retain(|e| e != emote_id);
        }
    }

    pub fn favorites(&self, user_id: &str) -> &[String] {
        self.user_favorites.get(user_id).map_or(&[], |v| v.as_slice())
    }

    pub fn usage_count(&self, emote_id: &str) -> u64 {
        self.usage_counts.get(emote_id).copied().unwrap_or(0)
    }

    pub fn stats(&self) -> EmoteStats {
        let unique_users: HashSet<&str> = self.users_who_used.values()
            .flat_map(|s| s.iter().map(|u| u.as_str()))
            .collect();
        EmoteStats {
            total_uses: self.usage_counts.values().sum(),
            unique_users: unique_users.len(),
            unique_emotes_used: self.usage_counts.len(),
        }
    }

    pub fn emote_count(&self) -> usize {
        self.emotes.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> EmoteRegistry {
        let mut r = EmoteRegistry::new(5, 10);
        r.register_emote(Emote::new("e1", "smile", EmoteCategory::Happy)).unwrap();
        r.register_emote(Emote::new("e2", "cry", EmoteCategory::Sad)).unwrap();
        r.register_emote(Emote::new("e3", "dance", EmoteCategory::Action).animated()).unwrap();
        r
    }

    #[test]
    fn test_register_emote() {
        let r = registry();
        assert_eq!(r.emote_count(), 3);
    }

    #[test]
    fn test_duplicate_emote() {
        let mut r = registry();
        assert!(r.register_emote(Emote::new("e1", "smile2", EmoteCategory::Happy)).is_err());
    }

    #[test]
    fn test_remove_emote() {
        let mut r = registry();
        r.remove_emote("e1").unwrap();
        assert_eq!(r.emote_count(), 2);
    }

    #[test]
    fn test_use_emote() {
        let mut r = registry();
        r.use_emote("alice", "e1", 100).unwrap();
        assert_eq!(r.usage_count("e1"), 1);
    }

    #[test]
    fn test_cooldown() {
        let mut r = registry();
        r.use_emote("alice", "e1", 100).unwrap();
        let err = r.use_emote("alice", "e2", 103);
        assert!(err.is_err());
        // After cooldown
        r.use_emote("alice", "e2", 106).unwrap();
    }

    #[test]
    fn test_search_by_name() {
        let r = registry();
        let found = r.search_by_name("smi");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "smile");
    }

    #[test]
    fn test_search_by_category() {
        let r = registry();
        let found = r.search_by_category(EmoteCategory::Sad);
        assert_eq!(found.len(), 1);
    }

    #[test]
    fn test_popular_emotes() {
        let mut r = registry();
        r.use_emote("alice", "e1", 100).unwrap();
        r.use_emote("bob", "e1", 100).unwrap();
        r.use_emote("charlie", "e2", 100).unwrap();
        let popular = r.popular_emotes(2);
        assert_eq!(popular[0].0, "e1");
        assert_eq!(popular[0].1, 2);
    }

    #[test]
    fn test_favorites() {
        let mut r = registry();
        r.add_favorite("alice", "e1");
        r.add_favorite("alice", "e3");
        assert_eq!(r.favorites("alice").len(), 2);
        r.remove_favorite("alice", "e1");
        assert_eq!(r.favorites("alice").len(), 1);
    }

    #[test]
    fn test_combo_detection() {
        let mut r = registry();
        r.use_emote("alice", "e1", 100).unwrap();
        r.use_emote("bob", "e1", 102).unwrap();
        r.use_emote("charlie", "e1", 104).unwrap();
        let combo = r.detect_combo().unwrap();
        assert_eq!(combo.emote_id, "e1");
        assert_eq!(combo.count, 3);
    }

    #[test]
    fn test_no_combo() {
        let mut r = registry();
        r.use_emote("alice", "e1", 100).unwrap();
        // Only one user, no combo
        assert!(r.detect_combo().is_none());
    }

    #[test]
    fn test_combo_window_expiry() {
        let mut r = EmoteRegistry::new(0, 5); // 0 cooldown, 5s window
        r.register_emote(Emote::new("e1", "smile", EmoteCategory::Happy)).unwrap();
        r.use_emote("alice", "e1", 100).unwrap();
        r.use_emote("bob", "e1", 200).unwrap(); // 100s later, alice expired
        // alice's entry should be evicted
        let combo = r.detect_combo();
        assert!(combo.is_none());
    }

    #[test]
    fn test_emote_set() {
        let mut set = EmoteSet::new("s1", "Default");
        set.add_emote("e1");
        set.add_emote("e2");
        assert_eq!(set.count(), 2);
        set.remove_emote("e1");
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn test_register_set() {
        let mut r = registry();
        r.register_set(EmoteSet::new("s1", "Default")).unwrap();
        assert!(r.get_set("s1").is_some());
    }

    #[test]
    fn test_stats() {
        let mut r = registry();
        r.use_emote("alice", "e1", 100).unwrap();
        r.use_emote("bob", "e2", 100).unwrap();
        let stats = r.stats();
        assert_eq!(stats.total_uses, 2);
        assert_eq!(stats.unique_users, 2);
        assert_eq!(stats.unique_emotes_used, 2);
    }

    #[test]
    fn test_animated_emote() {
        let e = Emote::new("e1", "dance", EmoteCategory::Action).animated();
        assert!(e.is_animated);
    }

    #[test]
    fn test_display_emote() {
        let e = Emote::new("e1", "smile", EmoteCategory::Happy);
        let s = format!("{e}");
        assert!(s.contains("smile"));
        assert!(s.contains("happy"));
    }

    #[test]
    fn test_display_emote_set() {
        let mut set = EmoteSet::new("s1", "Default");
        set.add_emote("e1");
        let s = format!("{set}");
        assert!(s.contains("Default"));
        assert!(s.contains("1 emote"));
    }

    #[test]
    fn test_emote_not_found() {
        let mut r = registry();
        assert!(r.use_emote("alice", "nonexistent", 100).is_err());
    }

    #[test]
    fn test_emote_with_creator() {
        let e = Emote::new("e1", "smile", EmoteCategory::Happy).with_creator("alice");
        assert_eq!(e.creator.as_deref(), Some("alice"));
    }
}
