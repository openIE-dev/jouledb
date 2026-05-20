//! Content Moderation Workflow — similarity-based trust & safety.
//!
//! $14B market (2026). Every content platform needs:
//! 1. Scan uploads against known-bad content (similarity search)
//! 2. Flag matches for human review (priority queue)
//! 3. Record actions with audit trail (compliance)
//!
//! This module uses the amorphic engine's HNSW similarity search
//! to match content against a policy of banned/flagged holograms.

use joule_db_hdc::BinaryHV;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::{AmorphicStore, QueryResult, RecordId, DIMENSION};

/// Action to take when content matches a moderation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModerationAction {
    /// Allow content (no violation detected)
    Allow,
    /// Flag for human review
    Flag,
    /// Quarantine (hide from public, pending review)
    Quarantine,
    /// Block immediately (high-confidence violation)
    Block,
}

/// A match against a moderation policy.
#[derive(Debug, Clone)]
pub struct ModerationMatch {
    /// The policy rule that matched
    pub rule_id: String,
    /// Reason for the match
    pub reason: String,
    /// Similarity score (0.0-1.0, higher = more similar to banned content)
    pub similarity: f32,
    /// Recommended action
    pub action: ModerationAction,
}

/// A moderation policy: a set of banned/flagged content holograms.
pub struct ModerationPolicy {
    /// Banned content holograms with reasons
    rules: Vec<ModerationRule>,
    /// Default similarity threshold for flagging
    pub flag_threshold: f32,
    /// Threshold for automatic blocking (higher = more certain)
    pub block_threshold: f32,
}

struct ModerationRule {
    id: String,
    reason: String,
    hologram: BinaryHV,
    action: ModerationAction,
    threshold: f32,
}

impl ModerationPolicy {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            flag_threshold: 0.7,
            block_threshold: 0.9,
        }
    }

    /// Add a banned content hologram to the policy.
    pub fn add_banned(
        &mut self,
        rule_id: &str,
        reason: &str,
        hologram: BinaryHV,
        action: ModerationAction,
    ) {
        self.rules.push(ModerationRule {
            id: rule_id.to_string(),
            reason: reason.to_string(),
            hologram,
            action,
            threshold: match action {
                ModerationAction::Block => self.block_threshold,
                ModerationAction::Quarantine => self.block_threshold,
                ModerationAction::Flag => self.flag_threshold,
                ModerationAction::Allow => 1.0, // Never matches
            },
        });
    }

    /// Add a banned content hologram from raw bytes (for convenience).
    pub fn add_banned_from_bytes(
        &mut self,
        rule_id: &str,
        reason: &str,
        content_bytes: &[u8],
        action: ModerationAction,
    ) {
        let hologram = BinaryHV::from_hash(content_bytes, DIMENSION);
        self.add_banned(rule_id, reason, hologram, action);
    }

    /// Scan content against this policy.
    /// Returns all matching rules sorted by severity.
    pub fn scan(&self, content_hologram: &BinaryHV) -> Vec<ModerationMatch> {
        let mut matches: Vec<ModerationMatch> = self
            .rules
            .iter()
            .filter_map(|rule| {
                let sim = content_hologram.similarity(&rule.hologram);
                if sim >= rule.threshold {
                    Some(ModerationMatch {
                        rule_id: rule.id.clone(),
                        reason: rule.reason.clone(),
                        similarity: sim,
                        action: rule.action,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by severity (Block > Quarantine > Flag) then by similarity
        matches.sort_by(|a, b| {
            let severity_a = match a.action {
                ModerationAction::Block => 3,
                ModerationAction::Quarantine => 2,
                ModerationAction::Flag => 1,
                ModerationAction::Allow => 0,
            };
            let severity_b = match b.action {
                ModerationAction::Block => 3,
                ModerationAction::Quarantine => 2,
                ModerationAction::Flag => 1,
                ModerationAction::Allow => 0,
            };
            severity_b
                .cmp(&severity_a)
                .then(b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal))
        });

        matches
    }

    /// Determine the strongest action from a set of matches.
    pub fn strongest_action(matches: &[ModerationMatch]) -> ModerationAction {
        matches
            .iter()
            .map(|m| m.action)
            .max_by_key(|a| match a {
                ModerationAction::Block => 3,
                ModerationAction::Quarantine => 2,
                ModerationAction::Flag => 1,
                ModerationAction::Allow => 0,
            })
            .unwrap_or(ModerationAction::Allow)
    }

    /// Number of rules in the policy.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

impl Default for ModerationPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Find all records in the store similar to a reference hologram.
/// Used for: "find everything similar to this banned content."
pub fn flag_similar_in_store(
    store: &AmorphicStore,
    reference: &BinaryHV,
    threshold: f32,
) -> QueryResult {
    store.query_media_similar(reference, 100, threshold)
}

/// A flagged item awaiting human review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlaggedItem {
    pub record_id: RecordId,
    pub matches: Vec<FlaggedMatch>,
    pub status: ReviewStatus,
    pub flagged_at_ms: u64,
}

/// Simplified match info for the review queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlaggedMatch {
    pub rule_id: String,
    pub reason: String,
    pub similarity: f32,
    pub action: ModerationAction,
}

/// Status of a flagged item in the review queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReviewStatus {
    Pending,
    InReview,
    Approved,
    Rejected,
    Escalated,
}

/// Priority queue for human review of flagged content.
pub struct ModerationQueue {
    /// Items indexed by priority (higher = more urgent). Uses BTreeMap for sorted access.
    items: BTreeMap<u64, Vec<FlaggedItem>>,
    /// Total items in the queue
    total: usize,
}

impl ModerationQueue {
    pub fn new() -> Self {
        Self {
            items: BTreeMap::new(),
            total: 0,
        }
    }

    /// Add a flagged item to the queue with a priority score.
    /// Higher priority = reviewed sooner.
    pub fn enqueue(&mut self, item: FlaggedItem, priority: u64) {
        self.items.entry(priority).or_default().push(item);
        self.total += 1;
    }

    /// Get the next item to review (highest priority).
    pub fn dequeue(&mut self) -> Option<FlaggedItem> {
        // BTreeMap is sorted ascending, so last = highest priority
        if let Some((&priority, items)) = self.items.last_key_value() {
            if let Some(item) = items.last().cloned() {
                let items_mut = self.items.get_mut(&priority).unwrap();
                items_mut.pop();
                if items_mut.is_empty() {
                    self.items.remove(&priority);
                }
                self.total -= 1;
                return Some(item);
            }
        }
        None
    }

    /// Number of items awaiting review.
    pub fn pending_count(&self) -> usize {
        self.total
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
}

impl Default for ModerationQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moderation_policy_scan() {
        let mut policy = ModerationPolicy::new();
        policy.flag_threshold = 0.7;
        policy.block_threshold = 0.9;

        // Add banned content
        let banned = BinaryHV::from_hash(b"banned_content_xyz", DIMENSION);
        policy.add_banned("rule_1", "NSFW content", banned.clone(), ModerationAction::Block);

        // Scan the exact same content — should match
        let matches = policy.scan(&banned);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].action, ModerationAction::Block);
        assert!(matches[0].similarity > 0.9);

        // Scan unrelated content — should not match
        let safe = BinaryHV::from_hash(b"cute_kitten_video", DIMENSION);
        let matches = policy.scan(&safe);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_strongest_action() {
        let matches = vec![
            ModerationMatch {
                rule_id: "r1".into(),
                reason: "test".into(),
                similarity: 0.8,
                action: ModerationAction::Flag,
            },
            ModerationMatch {
                rule_id: "r2".into(),
                reason: "test".into(),
                similarity: 0.95,
                action: ModerationAction::Block,
            },
        ];
        assert_eq!(
            ModerationPolicy::strongest_action(&matches),
            ModerationAction::Block
        );
    }

    #[test]
    fn test_moderation_queue() {
        let mut queue = ModerationQueue::new();

        queue.enqueue(
            FlaggedItem {
                record_id: 1,
                matches: vec![],
                status: ReviewStatus::Pending,
                flagged_at_ms: 1000,
            },
            50, // low priority
        );
        queue.enqueue(
            FlaggedItem {
                record_id: 2,
                matches: vec![],
                status: ReviewStatus::Pending,
                flagged_at_ms: 2000,
            },
            100, // high priority
        );

        assert_eq!(queue.pending_count(), 2);

        // Should dequeue highest priority first
        let item = queue.dequeue().unwrap();
        assert_eq!(item.record_id, 2);
        assert_eq!(queue.pending_count(), 1);
    }
}
