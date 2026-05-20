//! Emoji reaction system — add/remove/toggle, summaries, exclusive mode, bursts.
//!
//! Replaces custom reaction UIs with a pure-Rust reaction model
//! supporting multi-emoji, exclusive mode, and burst detection.

use chrono::{DateTime, TimeDelta, Utc};
use std::collections::{HashMap, HashSet};

// ── Reaction ────────────────────────────────────────────────────

/// A single reaction from a user.
#[derive(Debug, Clone)]
pub struct Reaction {
    pub emoji: String,
    pub user_id: String,
    pub timestamp: DateTime<Utc>,
}

impl Reaction {
    pub fn new(emoji: &str, user_id: &str) -> Self {
        Self {
            emoji: emoji.to_string(),
            user_id: user_id.to_string(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }
}

// ── ReactionSummary ─────────────────────────────────────────────

/// Summary of reactions for a single emoji.
#[derive(Debug, Clone)]
pub struct ReactionSummary {
    pub emoji: String,
    pub count: usize,
    pub user_ids: Vec<String>,
}

// ── ReactionSet ─────────────────────────────────────────────────

/// All reactions on a single target entity.
#[derive(Debug, Clone, Default)]
pub struct ReactionSet {
    reactions: Vec<Reaction>,
    /// When true, each user can only have one reaction (adding a new one removes the old).
    exclusive: bool,
}

impl ReactionSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an exclusive reaction set (one reaction per user).
    pub fn new_exclusive() -> Self {
        Self {
            reactions: Vec::new(),
            exclusive: true,
        }
    }

    /// Set exclusive mode.
    pub fn set_exclusive(&mut self, exclusive: bool) {
        self.exclusive = exclusive;
    }

    /// Add a reaction. Returns true if added, false if it was a duplicate.
    /// In exclusive mode, adding a new emoji removes the user's previous reaction.
    pub fn add(&mut self, emoji: &str, user_id: &str) -> bool {
        // Check if user already reacted with this emoji.
        if self
            .reactions
            .iter()
            .any(|r| r.emoji == emoji && r.user_id == user_id)
        {
            return false;
        }

        if self.exclusive {
            // Remove any existing reaction from this user.
            self.reactions.retain(|r| r.user_id != user_id);
        }

        self.reactions.push(Reaction::new(emoji, user_id));
        true
    }

    /// Add a reaction with a specific timestamp.
    pub fn add_with_timestamp(
        &mut self,
        emoji: &str,
        user_id: &str,
        ts: DateTime<Utc>,
    ) -> bool {
        if self
            .reactions
            .iter()
            .any(|r| r.emoji == emoji && r.user_id == user_id)
        {
            return false;
        }

        if self.exclusive {
            self.reactions.retain(|r| r.user_id != user_id);
        }

        self.reactions
            .push(Reaction::new(emoji, user_id).with_timestamp(ts));
        true
    }

    /// Remove a specific reaction.
    pub fn remove(&mut self, emoji: &str, user_id: &str) -> bool {
        let len_before = self.reactions.len();
        self.reactions
            .retain(|r| !(r.emoji == emoji && r.user_id == user_id));
        self.reactions.len() < len_before
    }

    /// Toggle a reaction: add if absent, remove if present.
    pub fn toggle(&mut self, emoji: &str, user_id: &str) -> bool {
        if self.has_reacted(emoji, user_id) {
            self.remove(emoji, user_id);
            false // removed
        } else {
            self.add(emoji, user_id);
            true // added
        }
    }

    /// Check if a user has reacted with a specific emoji.
    pub fn has_reacted(&self, emoji: &str, user_id: &str) -> bool {
        self.reactions
            .iter()
            .any(|r| r.emoji == emoji && r.user_id == user_id)
    }

    /// Check if a user has any reaction.
    pub fn has_any_reaction(&self, user_id: &str) -> bool {
        self.reactions.iter().any(|r| r.user_id == user_id)
    }

    /// Total reaction count.
    pub fn total_count(&self) -> usize {
        self.reactions.len()
    }

    /// Count for a specific emoji.
    pub fn count_for(&self, emoji: &str) -> usize {
        self.reactions.iter().filter(|r| r.emoji == emoji).count()
    }

    /// Get a summary per emoji.
    pub fn summary(&self) -> Vec<ReactionSummary> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for r in &self.reactions {
            map.entry(r.emoji.clone())
                .or_default()
                .push(r.user_id.clone());
        }
        let mut summaries: Vec<ReactionSummary> = map
            .into_iter()
            .map(|(emoji, user_ids)| ReactionSummary {
                count: user_ids.len(),
                emoji,
                user_ids,
            })
            .collect();
        summaries.sort_by(|a, b| b.count.cmp(&a.count));
        summaries
    }

    /// Top N reactions by count.
    pub fn top_reactions(&self, n: usize) -> Vec<ReactionSummary> {
        let mut s = self.summary();
        s.truncate(n);
        s
    }

    /// Distinct emojis used.
    pub fn distinct_emojis(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for r in &self.reactions {
            if seen.insert(r.emoji.clone()) {
                result.push(r.emoji.clone());
            }
        }
        result
    }

    /// Count of reactions within a time window (burst detection).
    pub fn burst_count(&self, window: TimeDelta, now: DateTime<Utc>) -> usize {
        let cutoff = now - window;
        self.reactions
            .iter()
            .filter(|r| r.timestamp >= cutoff)
            .count()
    }

    /// Get all reactions.
    pub fn all(&self) -> &[Reaction] {
        &self.reactions
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_reaction() {
        let mut set = ReactionSet::new();
        assert!(set.add("👍", "alice"));
        assert_eq!(set.total_count(), 1);
    }

    #[test]
    fn test_add_duplicate_rejected() {
        let mut set = ReactionSet::new();
        assert!(set.add("👍", "alice"));
        assert!(!set.add("👍", "alice")); // duplicate
        assert_eq!(set.total_count(), 1);
    }

    #[test]
    fn test_remove_reaction() {
        let mut set = ReactionSet::new();
        set.add("👍", "alice");
        assert!(set.remove("👍", "alice"));
        assert_eq!(set.total_count(), 0);
    }

    #[test]
    fn test_toggle() {
        let mut set = ReactionSet::new();
        assert!(set.toggle("👍", "alice")); // added
        assert!(!set.toggle("👍", "alice")); // removed
        assert_eq!(set.total_count(), 0);
    }

    #[test]
    fn test_has_reacted() {
        let mut set = ReactionSet::new();
        set.add("❤️", "alice");
        assert!(set.has_reacted("❤️", "alice"));
        assert!(!set.has_reacted("❤️", "bob"));
        assert!(!set.has_reacted("👍", "alice"));
    }

    #[test]
    fn test_summary() {
        let mut set = ReactionSet::new();
        set.add("👍", "alice");
        set.add("👍", "bob");
        set.add("❤️", "alice");
        let summary = set.summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].emoji, "👍"); // most popular first
        assert_eq!(summary[0].count, 2);
        assert_eq!(summary[1].emoji, "❤️");
        assert_eq!(summary[1].count, 1);
    }

    #[test]
    fn test_top_reactions() {
        let mut set = ReactionSet::new();
        set.add("👍", "a");
        set.add("👍", "b");
        set.add("❤️", "a");
        set.add("😂", "c");
        let top = set.top_reactions(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].emoji, "👍");
    }

    #[test]
    fn test_exclusive_mode() {
        let mut set = ReactionSet::new_exclusive();
        set.add("👍", "alice");
        set.add("❤️", "alice"); // should replace 👍
        assert_eq!(set.total_count(), 1);
        assert!(!set.has_reacted("👍", "alice"));
        assert!(set.has_reacted("❤️", "alice"));
    }

    #[test]
    fn test_exclusive_different_users() {
        let mut set = ReactionSet::new_exclusive();
        set.add("👍", "alice");
        set.add("❤️", "bob");
        assert_eq!(set.total_count(), 2);
    }

    #[test]
    fn test_burst_count() {
        let mut set = ReactionSet::new();
        let now = Utc::now();
        set.add_with_timestamp("👍", "a", now - TimeDelta::seconds(2));
        set.add_with_timestamp("👍", "b", now - TimeDelta::seconds(1));
        set.add_with_timestamp("👍", "c", now);
        set.add_with_timestamp("👍", "d", now - TimeDelta::seconds(30));

        let burst = set.burst_count(TimeDelta::seconds(5), now);
        assert_eq!(burst, 3); // d is outside the 5s window
    }

    #[test]
    fn test_count_for_emoji() {
        let mut set = ReactionSet::new();
        set.add("👍", "a");
        set.add("👍", "b");
        set.add("❤️", "a");
        assert_eq!(set.count_for("👍"), 2);
        assert_eq!(set.count_for("❤️"), 1);
        assert_eq!(set.count_for("😂"), 0);
    }

    #[test]
    fn test_distinct_emojis() {
        let mut set = ReactionSet::new();
        set.add("👍", "a");
        set.add("👍", "b");
        set.add("❤️", "c");
        let distinct = set.distinct_emojis();
        assert_eq!(distinct.len(), 2);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut set = ReactionSet::new();
        assert!(!set.remove("👍", "nobody"));
    }
}
