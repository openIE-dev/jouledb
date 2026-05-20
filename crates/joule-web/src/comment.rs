//! Comment/discussion model — threaded comments, edit history, soft delete.
//!
//! Replaces Disqus / Commento / custom comment systems with a pure-Rust
//! comment tree model supporting threading, editing, and display ordering.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

// ── Comment ─────────────────────────────────────────────────────

/// A single comment.
#[derive(Debug, Clone)]
pub struct Comment {
    pub id: Uuid,
    pub author_id: String,
    pub body: String,
    pub timestamp: DateTime<Utc>,
    pub parent_id: Option<Uuid>,
    pub edited: bool,
    pub deleted: bool,
    pub edit_history: Vec<EditRecord>,
}

/// Record of a previous edit.
#[derive(Debug, Clone)]
pub struct EditRecord {
    pub previous_body: String,
    pub edited_at: DateTime<Utc>,
}

impl Comment {
    /// Create a new top-level comment.
    pub fn new(author_id: &str, body: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            author_id: author_id.to_string(),
            body: body.to_string(),
            timestamp: Utc::now(),
            parent_id: None,
            edited: false,
            deleted: false,
            edit_history: Vec::new(),
        }
    }

    /// Create a reply to a parent comment.
    pub fn reply(author_id: &str, body: &str, parent_id: Uuid) -> Self {
        Self {
            parent_id: Some(parent_id),
            ..Self::new(author_id, body)
        }
    }

    pub fn with_timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = ts;
        self
    }

    /// Edit this comment, saving the old body in history.
    pub fn edit(&mut self, new_body: &str) {
        if self.deleted {
            return;
        }
        self.edit_history.push(EditRecord {
            previous_body: self.body.clone(),
            edited_at: Utc::now(),
        });
        self.body = new_body.to_string();
        self.edited = true;
    }

    /// Soft delete: replace body with "[deleted]".
    pub fn soft_delete(&mut self) {
        if !self.deleted {
            self.edit_history.push(EditRecord {
                previous_body: self.body.clone(),
                edited_at: Utc::now(),
            });
            self.body = "[deleted]".to_string();
            self.deleted = true;
        }
    }
}

// ── Sort Order ──────────────────────────────────────────────────

/// Sort order for comments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentSort {
    Newest,
    Oldest,
    MostReplies,
}

// ── CommentThread ───────────────────────────────────────────────

/// A collection of comments forming a tree structure.
#[derive(Debug, Clone, Default)]
pub struct CommentThread {
    comments: Vec<Comment>,
}

impl CommentThread {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a comment to the thread.
    pub fn add(&mut self, comment: Comment) {
        self.comments.push(comment);
    }

    /// Add a reply to an existing comment.
    pub fn add_reply(&mut self, parent_id: Uuid, author_id: &str, body: &str) -> Option<Uuid> {
        // Verify parent exists.
        if !self.comments.iter().any(|c| c.id == parent_id) {
            return None;
        }
        let reply = Comment::reply(author_id, body, parent_id);
        let id = reply.id;
        self.comments.push(reply);
        Some(id)
    }

    /// Edit a comment by id.
    pub fn edit(&mut self, id: Uuid, new_body: &str) -> bool {
        if let Some(c) = self.comments.iter_mut().find(|c| c.id == id) {
            c.edit(new_body);
            true
        } else {
            false
        }
    }

    /// Soft-delete a comment by id.
    pub fn soft_delete(&mut self, id: Uuid) -> bool {
        if let Some(c) = self.comments.iter_mut().find(|c| c.id == id) {
            c.soft_delete();
            true
        } else {
            false
        }
    }

    /// Get a comment by id.
    pub fn get(&self, id: Uuid) -> Option<&Comment> {
        self.comments.iter().find(|c| c.id == id)
    }

    /// Total comment count (including replies and deleted).
    pub fn count(&self) -> usize {
        self.comments.len()
    }

    /// Count of non-deleted comments.
    pub fn active_count(&self) -> usize {
        self.comments.iter().filter(|c| !c.deleted).count()
    }

    /// Count of top-level comments.
    pub fn top_level_count(&self) -> usize {
        self.comments.iter().filter(|c| c.parent_id.is_none()).count()
    }

    /// Count direct replies to a comment.
    pub fn reply_count(&self, parent_id: Uuid) -> usize {
        self.comments
            .iter()
            .filter(|c| c.parent_id == Some(parent_id))
            .count()
    }

    /// Get direct children of a comment (or top-level if None).
    pub fn children_of(&self, parent_id: Option<Uuid>) -> Vec<&Comment> {
        self.comments
            .iter()
            .filter(|c| c.parent_id == parent_id)
            .collect()
    }

    /// Flatten the thread to display order (depth-first traversal).
    /// Returns (comment_ref, depth) pairs.
    pub fn flatten(&self) -> Vec<(&Comment, usize)> {
        let mut result = Vec::new();
        // Build children index.
        let mut children_map: HashMap<Option<Uuid>, Vec<&Comment>> = HashMap::new();
        for c in &self.comments {
            children_map.entry(c.parent_id).or_default().push(c);
        }
        // Sort children by timestamp within each group.
        for children in children_map.values_mut() {
            children.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        }
        // DFS from roots.
        fn dfs<'a>(
            parent: Option<Uuid>,
            depth: usize,
            map: &HashMap<Option<Uuid>, Vec<&'a Comment>>,
            out: &mut Vec<(&'a Comment, usize)>,
        ) {
            if let Some(children) = map.get(&parent) {
                for child in children {
                    out.push((child, depth));
                    dfs(Some(child.id), depth + 1, map, out);
                }
            }
        }
        dfs(None, 0, &children_map, &mut result);
        result
    }

    /// Sort top-level comments by the given order.
    pub fn sorted_top_level(&self, sort: CommentSort) -> Vec<&Comment> {
        let mut top: Vec<&Comment> = self.children_of(None);
        match sort {
            CommentSort::Newest => top.sort_by(|a, b| b.timestamp.cmp(&a.timestamp)),
            CommentSort::Oldest => top.sort_by(|a, b| a.timestamp.cmp(&b.timestamp)),
            CommentSort::MostReplies => {
                top.sort_by(|a, b| {
                    self.reply_count(b.id).cmp(&self.reply_count(a.id))
                });
            }
        }
        top
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeDelta;

    #[test]
    fn test_create_comment() {
        let c = Comment::new("alice", "Hello world");
        assert_eq!(c.author_id, "alice");
        assert_eq!(c.body, "Hello world");
        assert!(!c.edited);
        assert!(!c.deleted);
        assert!(c.parent_id.is_none());
    }

    #[test]
    fn test_reply() {
        let parent = Comment::new("alice", "Root comment");
        let reply = Comment::reply("bob", "Nice!", parent.id);
        assert_eq!(reply.parent_id, Some(parent.id));
    }

    #[test]
    fn test_edit_comment() {
        let mut c = Comment::new("alice", "Original");
        c.edit("Edited version");
        assert_eq!(c.body, "Edited version");
        assert!(c.edited);
        assert_eq!(c.edit_history.len(), 1);
        assert_eq!(c.edit_history[0].previous_body, "Original");
    }

    #[test]
    fn test_soft_delete() {
        let mut c = Comment::new("alice", "To be deleted");
        c.soft_delete();
        assert!(c.deleted);
        assert_eq!(c.body, "[deleted]");
        assert_eq!(c.edit_history.len(), 1);
    }

    #[test]
    fn test_edit_after_delete_no_op() {
        let mut c = Comment::new("alice", "Content");
        c.soft_delete();
        c.edit("Try to edit");
        assert_eq!(c.body, "[deleted]"); // Edit should be no-op
    }

    #[test]
    fn test_thread_add_and_count() {
        let mut thread = CommentThread::new();
        thread.add(Comment::new("alice", "First"));
        thread.add(Comment::new("bob", "Second"));
        assert_eq!(thread.count(), 2);
        assert_eq!(thread.top_level_count(), 2);
    }

    #[test]
    fn test_thread_add_reply() {
        let mut thread = CommentThread::new();
        let root = Comment::new("alice", "Root");
        let root_id = root.id;
        thread.add(root);
        let reply_id = thread.add_reply(root_id, "bob", "Reply").unwrap();
        assert_eq!(thread.count(), 2);
        assert_eq!(thread.reply_count(root_id), 1);
        assert_eq!(thread.get(reply_id).unwrap().parent_id, Some(root_id));
    }

    #[test]
    fn test_thread_reply_to_nonexistent() {
        let mut thread = CommentThread::new();
        let result = thread.add_reply(Uuid::new_v4(), "bob", "Reply");
        assert!(result.is_none());
    }

    #[test]
    fn test_flatten_depth_first() {
        let mut thread = CommentThread::new();
        let now = Utc::now();

        let root1 = Comment::new("alice", "Root 1").with_timestamp(now);
        let root1_id = root1.id;
        thread.add(root1);

        let root2 = Comment::new("bob", "Root 2").with_timestamp(now + TimeDelta::seconds(1));
        thread.add(root2);

        // Reply to root1
        thread.add_reply(root1_id, "carol", "Reply to R1");

        let flat = thread.flatten();
        assert_eq!(flat.len(), 3);
        assert_eq!(flat[0].0.body, "Root 1");
        assert_eq!(flat[0].1, 0); // depth 0
        assert_eq!(flat[1].0.body, "Reply to R1");
        assert_eq!(flat[1].1, 1); // depth 1
        assert_eq!(flat[2].0.body, "Root 2");
        assert_eq!(flat[2].1, 0); // depth 0
    }

    #[test]
    fn test_sort_newest() {
        let mut thread = CommentThread::new();
        let now = Utc::now();
        thread.add(Comment::new("alice", "Old").with_timestamp(now - TimeDelta::hours(2)));
        thread.add(Comment::new("bob", "New").with_timestamp(now));
        let sorted = thread.sorted_top_level(CommentSort::Newest);
        assert_eq!(sorted[0].body, "New");
    }

    #[test]
    fn test_sort_oldest() {
        let mut thread = CommentThread::new();
        let now = Utc::now();
        thread.add(Comment::new("alice", "Old").with_timestamp(now - TimeDelta::hours(2)));
        thread.add(Comment::new("bob", "New").with_timestamp(now));
        let sorted = thread.sorted_top_level(CommentSort::Oldest);
        assert_eq!(sorted[0].body, "Old");
    }

    #[test]
    fn test_sort_most_replies() {
        let mut thread = CommentThread::new();
        let c1 = Comment::new("alice", "Popular");
        let c1_id = c1.id;
        thread.add(c1);
        let c2 = Comment::new("bob", "Unpopular");
        thread.add(c2);
        thread.add_reply(c1_id, "x", "r1");
        thread.add_reply(c1_id, "y", "r2");
        let sorted = thread.sorted_top_level(CommentSort::MostReplies);
        assert_eq!(sorted[0].body, "Popular");
    }

    #[test]
    fn test_active_count_excludes_deleted() {
        let mut thread = CommentThread::new();
        let c = Comment::new("alice", "Will delete");
        let id = c.id;
        thread.add(c);
        thread.add(Comment::new("bob", "Keep"));
        thread.soft_delete(id);
        assert_eq!(thread.count(), 2);
        assert_eq!(thread.active_count(), 1);
    }

    #[test]
    fn test_multiple_edits() {
        let mut c = Comment::new("alice", "v1");
        c.edit("v2");
        c.edit("v3");
        assert_eq!(c.body, "v3");
        assert_eq!(c.edit_history.len(), 2);
        assert_eq!(c.edit_history[0].previous_body, "v1");
        assert_eq!(c.edit_history[1].previous_body, "v2");
    }
}
