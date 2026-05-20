//! Activity/timeline feed — events, aggregation, cursor pagination, fan-out.
//!
//! Replaces Stream (getstream.io) / Feeds API with a pure-Rust activity feed
//! model supporting aggregation, pagination, and per-user fan-out.

use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

// ── FeedItem ────────────────────────────────────────────────────

/// A single activity in a feed.
#[derive(Debug, Clone)]
pub struct FeedItem {
    pub id: Uuid,
    pub actor_id: String,
    pub verb: String,
    pub object_type: String,
    pub object_id: String,
    pub target_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, Value>,
    pub seen: bool,
}

impl FeedItem {
    pub fn new(
        actor_id: &str,
        verb: &str,
        object_type: &str,
        object_id: &str,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            actor_id: actor_id.to_string(),
            verb: verb.to_string(),
            object_type: object_type.to_string(),
            object_id: object_id.to_string(),
            target_id: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            seen: false,
        }
    }

    pub fn with_target(mut self, target_id: &str) -> Self {
        self.target_id = Some(target_id.to_string());
        self
    }

    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Aggregation key: verb + object_type + object_id.
    pub fn aggregation_key(&self) -> String {
        format!("{}:{}:{}", self.verb, self.object_type, self.object_id)
    }
}

// ── AggregatedActivity ──────────────────────────────────────────

/// A group of similar activities (e.g. "Alice and 3 others liked your post").
#[derive(Debug, Clone)]
pub struct AggregatedActivity {
    pub verb: String,
    pub object_type: String,
    pub object_id: String,
    pub actor_ids: Vec<String>,
    pub latest_timestamp: DateTime<Utc>,
    pub count: usize,
}

impl AggregatedActivity {
    /// Human-readable summary.
    /// E.g., "Alice liked your post" or "Alice and 2 others liked your post".
    pub fn summary(&self, actor_name_fn: &dyn Fn(&str) -> String) -> String {
        let first_name = actor_name_fn(&self.actor_ids[0]);
        let others = self.count - 1;
        if others == 0 {
            format!("{first_name} {verb} your {obj}", verb = self.verb, obj = self.object_type)
        } else if others == 1 {
            format!(
                "{first_name} and 1 other {verb} your {obj}",
                verb = self.verb,
                obj = self.object_type
            )
        } else {
            format!(
                "{first_name} and {others} others {verb} your {obj}",
                verb = self.verb,
                obj = self.object_type
            )
        }
    }
}

// ── CursorPage ──────────────────────────────────────────────────

/// A page of feed items with cursor-based pagination.
#[derive(Debug, Clone)]
pub struct CursorPage<'a> {
    pub items: Vec<&'a FeedItem>,
    pub next_cursor: Option<Uuid>,
    pub has_more: bool,
}

// ── Feed ────────────────────────────────────────────────────────

/// A feed containing activity items.
#[derive(Debug, Clone, Default)]
pub struct Feed {
    items: Vec<FeedItem>,
}

impl Feed {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item to the feed.
    pub fn add(&mut self, item: FeedItem) {
        self.items.push(item);
        // Keep sorted by timestamp descending.
        self.items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    }

    /// Total item count.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the feed is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Count of unseen items.
    pub fn unseen_count(&self) -> usize {
        self.items.iter().filter(|i| !i.seen).count()
    }

    /// Mark an item as seen.
    pub fn mark_seen(&mut self, id: Uuid) -> bool {
        if let Some(item) = self.items.iter_mut().find(|i| i.id == id) {
            item.seen = true;
            true
        } else {
            false
        }
    }

    /// Mark all items as seen.
    pub fn mark_all_seen(&mut self) {
        for item in &mut self.items {
            item.seen = true;
        }
    }

    /// Filter by verb.
    pub fn filter_by_verb(&self, verb: &str) -> Vec<&FeedItem> {
        self.items.iter().filter(|i| i.verb == verb).collect()
    }

    /// Filter by object type.
    pub fn filter_by_object_type(&self, object_type: &str) -> Vec<&FeedItem> {
        self.items
            .iter()
            .filter(|i| i.object_type == object_type)
            .collect()
    }

    /// Cursor-based pagination. Returns `limit` items after the cursor.
    pub fn paginate(&self, cursor: Option<Uuid>, limit: usize) -> CursorPage<'_> {
        let start_idx = match cursor {
            Some(cursor_id) => {
                // Find the item after the cursor.
                self.items
                    .iter()
                    .position(|i| i.id == cursor_id)
                    .map(|pos| pos + 1)
                    .unwrap_or(0)
            }
            None => 0,
        };

        let end_idx = (start_idx + limit).min(self.items.len());
        let items: Vec<&FeedItem> = self.items[start_idx..end_idx].iter().collect();
        let has_more = end_idx < self.items.len();
        let next_cursor = if has_more {
            items.last().map(|i| i.id)
        } else {
            None
        };

        CursorPage {
            items,
            next_cursor,
            has_more,
        }
    }

    /// Aggregate similar activities by aggregation key.
    pub fn aggregate(&self) -> Vec<AggregatedActivity> {
        let mut groups: HashMap<String, Vec<&FeedItem>> = HashMap::new();
        for item in &self.items {
            groups
                .entry(item.aggregation_key())
                .or_default()
                .push(item);
        }

        let mut aggregated: Vec<AggregatedActivity> = groups
            .into_iter()
            .map(|(_key, items)| {
                let first = items[0];
                let mut actor_ids: Vec<String> = Vec::new();
                for item in &items {
                    if !actor_ids.contains(&item.actor_id) {
                        actor_ids.push(item.actor_id.clone());
                    }
                }
                let latest = items
                    .iter()
                    .map(|i| i.timestamp)
                    .max()
                    .unwrap_or(first.timestamp);
                AggregatedActivity {
                    verb: first.verb.clone(),
                    object_type: first.object_type.clone(),
                    object_id: first.object_id.clone(),
                    count: items.len(),
                    actor_ids,
                    latest_timestamp: latest,
                }
            })
            .collect();

        aggregated.sort_by(|a, b| b.latest_timestamp.cmp(&a.latest_timestamp));
        aggregated
    }

    /// All items (already sorted newest first).
    pub fn all(&self) -> &[FeedItem] {
        &self.items
    }
}

// ── FanOut ───────────────────────────────────────────────────────

/// Fan-out model: distributes activities to per-user feeds.
#[derive(Debug, Clone, Default)]
pub struct FanOutManager {
    /// Per-user feeds, keyed by user_id.
    feeds: HashMap<String, Feed>,
}

impl FanOutManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fan out an activity to a list of follower user_ids.
    pub fn fan_out(&mut self, item: &FeedItem, follower_ids: &[&str]) {
        for uid in follower_ids {
            let feed = self
                .feeds
                .entry(uid.to_string())
                .or_insert_with(Feed::new);
            feed.add(item.clone());
        }
    }

    /// Get a user's feed.
    pub fn get_feed(&self, user_id: &str) -> Option<&Feed> {
        self.feeds.get(user_id)
    }

    /// Get a mutable user feed.
    pub fn get_feed_mut(&mut self, user_id: &str) -> Option<&mut Feed> {
        self.feeds.get_mut(user_id)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    #[test]
    fn test_add_and_order() {
        let mut feed = Feed::new();
        let now = Utc::now();
        feed.add(FeedItem::new("alice", "liked", "post", "p1").with_timestamp(now - TimeDelta::hours(1)));
        feed.add(FeedItem::new("bob", "liked", "post", "p2").with_timestamp(now));
        assert_eq!(feed.len(), 2);
        assert_eq!(feed.all()[0].actor_id, "bob"); // newest first
    }

    #[test]
    fn test_filter_by_verb() {
        let mut feed = Feed::new();
        feed.add(FeedItem::new("alice", "liked", "post", "p1"));
        feed.add(FeedItem::new("bob", "commented", "post", "p1"));
        feed.add(FeedItem::new("carol", "liked", "post", "p2"));
        let likes = feed.filter_by_verb("liked");
        assert_eq!(likes.len(), 2);
    }

    #[test]
    fn test_filter_by_object_type() {
        let mut feed = Feed::new();
        feed.add(FeedItem::new("alice", "liked", "post", "p1"));
        feed.add(FeedItem::new("bob", "liked", "photo", "ph1"));
        let posts = feed.filter_by_object_type("post");
        assert_eq!(posts.len(), 1);
    }

    #[test]
    fn test_aggregation() {
        let mut feed = Feed::new();
        feed.add(FeedItem::new("alice", "liked", "post", "p1"));
        feed.add(FeedItem::new("bob", "liked", "post", "p1"));
        feed.add(FeedItem::new("carol", "liked", "post", "p1"));
        feed.add(FeedItem::new("dave", "commented", "post", "p1"));

        let agg = feed.aggregate();
        assert_eq!(agg.len(), 2); // likes + comments

        let likes_agg = agg.iter().find(|a| a.verb == "liked").unwrap();
        assert_eq!(likes_agg.count, 3);
        assert_eq!(likes_agg.actor_ids.len(), 3);
    }

    #[test]
    fn test_aggregation_summary() {
        let mut feed = Feed::new();
        feed.add(FeedItem::new("alice", "liked", "post", "p1"));
        feed.add(FeedItem::new("bob", "liked", "post", "p1"));
        feed.add(FeedItem::new("carol", "liked", "post", "p1"));

        let agg = feed.aggregate();
        let likes_agg = agg.iter().find(|a| a.verb == "liked").unwrap();
        let summary = likes_agg.summary(&|id| id.to_string());
        // First actor depends on HashMap order; just check structure
        assert!(summary.contains("liked"));
        assert!(summary.contains("2 others"));
    }

    #[test]
    fn test_cursor_pagination() {
        let mut feed = Feed::new();
        let now = Utc::now();
        for i in 0..10 {
            feed.add(
                FeedItem::new("user", "liked", "post", &format!("p{i}"))
                    .with_timestamp(now + TimeDelta::seconds(i as i64)),
            );
        }

        // First page
        let page1 = feed.paginate(None, 3);
        assert_eq!(page1.items.len(), 3);
        assert!(page1.has_more);

        // Second page using cursor
        let page2 = feed.paginate(page1.next_cursor, 3);
        assert_eq!(page2.items.len(), 3);
        assert!(page2.has_more);

        // Different items
        assert_ne!(page1.items[0].id, page2.items[0].id);
    }

    #[test]
    fn test_mark_seen() {
        let mut feed = Feed::new();
        let item = FeedItem::new("alice", "liked", "post", "p1");
        let id = item.id;
        feed.add(item);
        assert_eq!(feed.unseen_count(), 1);
        feed.mark_seen(id);
        assert_eq!(feed.unseen_count(), 0);
    }

    #[test]
    fn test_mark_all_seen() {
        let mut feed = Feed::new();
        feed.add(FeedItem::new("alice", "liked", "post", "p1"));
        feed.add(FeedItem::new("bob", "liked", "post", "p2"));
        assert_eq!(feed.unseen_count(), 2);
        feed.mark_all_seen();
        assert_eq!(feed.unseen_count(), 0);
    }

    #[test]
    fn test_fan_out() {
        let mut fan = FanOutManager::new();
        let item = FeedItem::new("alice", "posted", "photo", "ph1");
        fan.fan_out(&item, &["bob", "carol", "dave"]);

        assert_eq!(fan.get_feed("bob").unwrap().len(), 1);
        assert_eq!(fan.get_feed("carol").unwrap().len(), 1);
        assert_eq!(fan.get_feed("dave").unwrap().len(), 1);
        assert!(fan.get_feed("alice").is_none());
    }

    #[test]
    fn test_empty_feed() {
        let feed = Feed::new();
        assert!(feed.is_empty());
        assert_eq!(feed.len(), 0);
        let page = feed.paginate(None, 10);
        assert!(page.items.is_empty());
        assert!(!page.has_more);
    }

    #[test]
    fn test_metadata() {
        let item = FeedItem::new("alice", "liked", "post", "p1")
            .with_metadata("score", serde_json::json!(42));
        assert_eq!(item.metadata["score"], 42);
    }

    #[test]
    fn test_target_id() {
        let item = FeedItem::new("alice", "shared", "post", "p1")
            .with_target("group1");
        assert_eq!(item.target_id.as_deref(), Some("group1"));
    }

    #[test]
    fn test_single_actor_summary() {
        let mut feed = Feed::new();
        feed.add(FeedItem::new("alice", "liked", "post", "p1"));
        let agg = feed.aggregate();
        let summary = agg[0].summary(&|id| id.to_string());
        assert_eq!(summary, "alice liked your post");
    }
}
