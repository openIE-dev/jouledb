//! Screen reader announcement system: live region announcements, queuing,
//! deduplication, priority, history, and clearing.
//!
//! Pure data — no browser dependency. Maintains an announcement queue that
//! renderers can drain into live regions.

use std::collections::VecDeque;

// ── Priority & Politeness ─────────────────────────────────────

/// Announcement politeness level (maps to aria-live).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Politeness {
    /// Announced when convenient (aria-live="polite").
    Polite,
    /// Announced immediately, interrupting (aria-live="assertive").
    Assertive,
}

/// Fine-grained priority within a politeness level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Normal,
    High,
    Critical,
}

// ── Announcement ──────────────────────────────────────────────

/// A single screen reader announcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Announcement {
    pub message: String,
    pub politeness: Politeness,
    pub priority: Priority,
    /// Monotonic sequence number (assigned by the queue).
    pub seq: u64,
}

// ── Announcement Queue ────────────────────────────────────────

/// Manages queued announcements with deduplication and history.
#[derive(Debug)]
pub struct AnnouncementQueue {
    queue: VecDeque<Announcement>,
    history: Vec<Announcement>,
    next_seq: u64,
    /// Maximum history entries. 0 = unlimited.
    max_history: usize,
    /// If true, duplicate consecutive messages are dropped.
    deduplicate: bool,
}

impl AnnouncementQueue {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            history: Vec::new(),
            next_seq: 1,
            max_history: 1000,
            deduplicate: true,
        }
    }

    /// Set whether to deduplicate consecutive identical messages.
    pub fn set_deduplicate(&mut self, dedup: bool) {
        self.deduplicate = dedup;
    }

    /// Set maximum history size.
    pub fn set_max_history(&mut self, max: usize) {
        self.max_history = max;
    }

    /// Enqueue a polite announcement with normal priority.
    pub fn announce(&mut self, message: &str) {
        self.enqueue(message, Politeness::Polite, Priority::Normal);
    }

    /// Enqueue an assertive announcement with high priority.
    pub fn announce_assertive(&mut self, message: &str) {
        self.enqueue(message, Politeness::Assertive, Priority::High);
    }

    /// Enqueue an announcement with full control.
    pub fn enqueue(&mut self, message: &str, politeness: Politeness, priority: Priority) {
        if message.is_empty() {
            return;
        }
        // Deduplication: skip if the last queued message is identical.
        if self.deduplicate {
            if let Some(last) = self.queue.back() {
                if last.message == message && last.politeness == politeness {
                    return;
                }
            }
        }
        let ann = Announcement {
            message: message.into(),
            politeness,
            priority,
            seq: self.next_seq,
        };
        self.next_seq += 1;

        // Assertive announcements go to front (after other assertive ones).
        if politeness == Politeness::Assertive {
            // Find insertion point: after existing assertive, before polite.
            let pos = self
                .queue
                .iter()
                .position(|a| a.politeness == Politeness::Polite)
                .unwrap_or(self.queue.len());
            self.queue.insert(pos, ann);
        } else {
            self.queue.push_back(ann);
        }
    }

    /// Drain the next announcement. Returns `None` if the queue is empty.
    pub fn next(&mut self) -> Option<Announcement> {
        let ann = self.queue.pop_front()?;
        self.push_history(ann.clone());
        Some(ann)
    }

    /// Drain all pending announcements in order.
    pub fn drain_all(&mut self) -> Vec<Announcement> {
        let mut out = Vec::with_capacity(self.queue.len());
        while let Some(ann) = self.next() {
            out.push(ann);
        }
        out
    }

    /// Peek at the next announcement without removing it.
    pub fn peek(&self) -> Option<&Announcement> {
        self.queue.front()
    }

    /// Number of pending announcements.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Clear all pending announcements.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Get announcement history (most recent last).
    pub fn history(&self) -> &[Announcement] {
        &self.history
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    fn push_history(&mut self, ann: Announcement) {
        self.history.push(ann);
        if self.max_history > 0 && self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
}

impl Default for AnnouncementQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ── Live Region Renderer ──────────────────────────────────────

/// Tracks aria-live regions on a page for rendering announcements.
#[derive(Debug)]
pub struct LiveRegionManager {
    polite_region_id: String,
    assertive_region_id: String,
    polite_content: String,
    assertive_content: String,
}

impl LiveRegionManager {
    pub fn new(polite_id: &str, assertive_id: &str) -> Self {
        Self {
            polite_region_id: polite_id.into(),
            assertive_region_id: assertive_id.into(),
            polite_content: String::new(),
            assertive_content: String::new(),
        }
    }

    /// Apply an announcement to the appropriate region.
    pub fn apply(&mut self, ann: &Announcement) {
        match ann.politeness {
            Politeness::Polite => self.polite_content = ann.message.clone(),
            Politeness::Assertive => self.assertive_content = ann.message.clone(),
        }
    }

    /// Clear both regions.
    pub fn clear(&mut self) {
        self.polite_content.clear();
        self.assertive_content.clear();
    }

    /// Get the polite region's current content.
    pub fn polite_content(&self) -> &str {
        &self.polite_content
    }

    /// Get the assertive region's current content.
    pub fn assertive_content(&self) -> &str {
        &self.assertive_content
    }

    /// Get the polite region element ID.
    pub fn polite_region_id(&self) -> &str {
        &self.polite_region_id
    }

    /// Get the assertive region element ID.
    pub fn assertive_region_id(&self) -> &str {
        &self.assertive_region_id
    }

    /// Render the polite region as an HTML element.
    pub fn render_polite(&self) -> String {
        format!(
            "<div id=\"{}\" aria-live=\"polite\" aria-atomic=\"true\" class=\"sr-only\">{}</div>",
            self.polite_region_id, self.polite_content
        )
    }

    /// Render the assertive region as an HTML element.
    pub fn render_assertive(&self) -> String {
        format!(
            "<div id=\"{}\" aria-live=\"assertive\" aria-atomic=\"true\" class=\"sr-only\">{}</div>",
            self.assertive_region_id, self.assertive_content
        )
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_announce() {
        let mut q = AnnouncementQueue::new();
        q.announce("Item added to cart");
        assert_eq!(q.pending_count(), 1);
        let ann = q.next().unwrap();
        assert_eq!(ann.message, "Item added to cart");
        assert_eq!(ann.politeness, Politeness::Polite);
        assert_eq!(ann.seq, 1);
    }

    #[test]
    fn assertive_before_polite() {
        let mut q = AnnouncementQueue::new();
        q.announce("polite msg");
        q.announce_assertive("assertive msg");
        let first = q.next().unwrap();
        assert_eq!(first.politeness, Politeness::Assertive);
        assert_eq!(first.message, "assertive msg");
    }

    #[test]
    fn deduplication() {
        let mut q = AnnouncementQueue::new();
        q.announce("same");
        q.announce("same"); // duplicate, dropped
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn deduplication_disabled() {
        let mut q = AnnouncementQueue::new();
        q.set_deduplicate(false);
        q.announce("same");
        q.announce("same");
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn empty_message_ignored() {
        let mut q = AnnouncementQueue::new();
        q.announce("");
        assert!(q.is_empty());
    }

    #[test]
    fn drain_all() {
        let mut q = AnnouncementQueue::new();
        q.announce("a");
        q.announce("b");
        q.announce("c");
        let all = q.drain_all();
        assert_eq!(all.len(), 3);
        assert!(q.is_empty());
    }

    #[test]
    fn history_tracking() {
        let mut q = AnnouncementQueue::new();
        q.announce("first");
        q.announce("second");
        q.next();
        q.next();
        assert_eq!(q.history().len(), 2);
        assert_eq!(q.history()[0].message, "first");
        assert_eq!(q.history()[1].message, "second");
    }

    #[test]
    fn history_limit() {
        let mut q = AnnouncementQueue::new();
        q.set_max_history(2);
        for i in 0..5 {
            q.enqueue(&format!("msg{}", i), Politeness::Polite, Priority::Normal);
        }
        q.drain_all();
        assert_eq!(q.history().len(), 2);
        assert_eq!(q.history()[0].message, "msg3");
    }

    #[test]
    fn clear_queue() {
        let mut q = AnnouncementQueue::new();
        q.announce("a");
        q.announce("b");
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn peek_does_not_consume() {
        let mut q = AnnouncementQueue::new();
        q.announce("hello");
        assert_eq!(q.peek().unwrap().message, "hello");
        assert_eq!(q.pending_count(), 1);
    }

    #[test]
    fn live_region_manager_apply() {
        let mut mgr = LiveRegionManager::new("sr-polite", "sr-assertive");
        let ann = Announcement {
            message: "Loaded".into(),
            politeness: Politeness::Polite,
            priority: Priority::Normal,
            seq: 1,
        };
        mgr.apply(&ann);
        assert_eq!(mgr.polite_content(), "Loaded");
        assert!(mgr.assertive_content().is_empty());
    }

    #[test]
    fn live_region_render() {
        let mut mgr = LiveRegionManager::new("p", "a");
        mgr.apply(&Announcement {
            message: "Done".into(),
            politeness: Politeness::Assertive,
            priority: Priority::High,
            seq: 1,
        });
        let html = mgr.render_assertive();
        assert!(html.contains("aria-live=\"assertive\""));
        assert!(html.contains("Done"));
    }

    #[test]
    fn sequence_numbers_increment() {
        let mut q = AnnouncementQueue::new();
        q.announce("a");
        q.announce("b");
        let a = q.next().unwrap();
        let b = q.next().unwrap();
        assert_eq!(a.seq, 1);
        assert_eq!(b.seq, 2);
    }

    #[test]
    fn priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }
}
