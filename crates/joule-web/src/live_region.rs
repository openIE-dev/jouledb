//! ARIA live region engine: region types (polite, assertive, off), atomic
//! updates, relevant content types, busy state, nesting, and throttling.
//!
//! Pure data — no browser dependency. Models live region state and update
//! semantics so renderers can apply correct aria attributes and content.

use std::collections::HashMap;

// ── Region Politeness ─────────────────────────────────────────

/// The aria-live politeness setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionPoliteness {
    Off,
    Polite,
    Assertive,
}

impl RegionPoliteness {
    pub fn as_str(&self) -> &'static str {
        match self {
            RegionPoliteness::Off => "off",
            RegionPoliteness::Polite => "polite",
            RegionPoliteness::Assertive => "assertive",
        }
    }
}

// ── Relevant Content ──────────────────────────────────────────

/// Which types of content changes are relevant (aria-relevant).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelevantContent {
    pub additions: bool,
    pub removals: bool,
    pub text: bool,
}

impl RelevantContent {
    /// Default: additions + text.
    pub fn default_relevant() -> Self {
        Self { additions: true, removals: false, text: true }
    }

    /// All changes relevant.
    pub fn all() -> Self {
        Self { additions: true, removals: true, text: true }
    }

    /// Render to the `aria-relevant` attribute value.
    pub fn to_attr(&self) -> String {
        if self.additions && self.removals && self.text {
            return "all".into();
        }
        let mut parts = Vec::new();
        if self.additions {
            parts.push("additions");
        }
        if self.removals {
            parts.push("removals");
        }
        if self.text {
            parts.push("text");
        }
        parts.join(" ")
    }
}

impl Default for RelevantContent {
    fn default() -> Self {
        Self::default_relevant()
    }
}

// ── Update Record ─────────────────────────────────────────────

/// Type of content change within a live region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateKind {
    Addition(String),
    Removal(String),
    TextChange { old: String, new: String },
}

/// A recorded update within a live region.
#[derive(Debug, Clone)]
pub struct RegionUpdate {
    pub kind: UpdateKind,
    pub timestamp_ms: u64,
    pub seq: u64,
}

// ── Live Region ───────────────────────────────────────────────

/// A single ARIA live region.
#[derive(Debug)]
pub struct LiveRegion {
    pub id: String,
    pub politeness: RegionPoliteness,
    pub atomic: bool,
    pub relevant: RelevantContent,
    pub busy: bool,
    content: Vec<String>,
    updates: Vec<RegionUpdate>,
    next_seq: u64,
    /// Minimum milliseconds between updates (throttling).
    throttle_ms: u64,
    last_update_ms: u64,
    children: Vec<String>,
}

impl LiveRegion {
    pub fn new(id: &str, politeness: RegionPoliteness) -> Self {
        Self {
            id: id.into(),
            politeness,
            atomic: false,
            relevant: RelevantContent::default(),
            busy: false,
            content: Vec::new(),
            updates: Vec::new(),
            next_seq: 1,
            throttle_ms: 0,
            last_update_ms: 0,
            children: Vec::new(),
        }
    }

    /// Set atomic mode — the entire region is announced on change.
    pub fn set_atomic(&mut self, atomic: bool) {
        self.atomic = atomic;
    }

    /// Set relevant content types.
    pub fn set_relevant(&mut self, relevant: RelevantContent) {
        self.relevant = relevant;
    }

    /// Mark region as busy (suppresses announcements until cleared).
    pub fn set_busy(&mut self, busy: bool) {
        self.busy = busy;
    }

    /// Set throttle interval in milliseconds.
    pub fn set_throttle_ms(&mut self, ms: u64) {
        self.throttle_ms = ms;
    }

    /// Register a nested child region ID.
    pub fn add_child(&mut self, child_id: &str) {
        if !self.children.contains(&child_id.to_string()) {
            self.children.push(child_id.into());
        }
    }

    /// Remove a child region.
    pub fn remove_child(&mut self, child_id: &str) {
        self.children.retain(|c| c != child_id);
    }

    /// Get child region IDs.
    pub fn children(&self) -> &[String] {
        &self.children
    }

    /// Add content to the region. Returns true if the update was accepted
    /// (not throttled, not busy, and relevant).
    pub fn add_content(&mut self, text: &str, now_ms: u64) -> bool {
        if self.busy || self.politeness == RegionPoliteness::Off {
            return false;
        }
        if !self.relevant.additions {
            return false;
        }
        if self.throttle_ms > 0 && now_ms.saturating_sub(self.last_update_ms) < self.throttle_ms {
            return false;
        }
        self.content.push(text.into());
        self.record_update(UpdateKind::Addition(text.into()), now_ms);
        true
    }

    /// Remove content by index. Returns true if accepted.
    pub fn remove_content(&mut self, index: usize, now_ms: u64) -> bool {
        if self.busy || self.politeness == RegionPoliteness::Off {
            return false;
        }
        if !self.relevant.removals {
            return false;
        }
        if index >= self.content.len() {
            return false;
        }
        if self.throttle_ms > 0 && now_ms.saturating_sub(self.last_update_ms) < self.throttle_ms {
            return false;
        }
        let removed = self.content.remove(index);
        self.record_update(UpdateKind::Removal(removed), now_ms);
        true
    }

    /// Replace content at index. Returns true if accepted.
    pub fn update_content(&mut self, index: usize, new_text: &str, now_ms: u64) -> bool {
        if self.busy || self.politeness == RegionPoliteness::Off {
            return false;
        }
        if !self.relevant.text {
            return false;
        }
        if index >= self.content.len() {
            return false;
        }
        if self.throttle_ms > 0 && now_ms.saturating_sub(self.last_update_ms) < self.throttle_ms {
            return false;
        }
        let old = std::mem::replace(&mut self.content[index], new_text.into());
        self.record_update(UpdateKind::TextChange { old, new: new_text.into() }, now_ms);
        true
    }

    /// Set all content at once (bulk update).
    pub fn set_content(&mut self, items: Vec<String>, now_ms: u64) {
        self.content = items;
        self.last_update_ms = now_ms;
    }

    /// Get current content.
    pub fn content(&self) -> &[String] {
        &self.content
    }

    /// Get the text that a screen reader should announce.
    /// For atomic regions, returns the full concatenated content.
    /// For non-atomic, returns only the latest update's text.
    pub fn announcement_text(&self) -> Option<String> {
        if self.busy || self.politeness == RegionPoliteness::Off {
            return None;
        }
        if self.atomic {
            if self.content.is_empty() {
                return None;
            }
            Some(self.content.join(" "))
        } else {
            self.updates.last().map(|u| match &u.kind {
                UpdateKind::Addition(s) => s.clone(),
                UpdateKind::Removal(s) => format!("removed: {}", s),
                UpdateKind::TextChange { new, .. } => new.clone(),
            })
        }
    }

    /// Get update history.
    pub fn updates(&self) -> &[RegionUpdate] {
        &self.updates
    }

    /// Clear update history.
    pub fn clear_updates(&mut self) {
        self.updates.clear();
    }

    /// Render as an HTML element.
    pub fn render(&self) -> String {
        let mut attrs = vec![
            format!("id=\"{}\"", self.id),
            format!("aria-live=\"{}\"", self.politeness.as_str()),
        ];
        if self.atomic {
            attrs.push("aria-atomic=\"true\"".into());
        }
        let rel = self.relevant.to_attr();
        if rel != "additions text" {
            attrs.push(format!("aria-relevant=\"{}\"", rel));
        }
        if self.busy {
            attrs.push("aria-busy=\"true\"".into());
        }
        let inner = self.content.join(" ");
        format!("<div {}>{}</div>", attrs.join(" "), inner)
    }

    fn record_update(&mut self, kind: UpdateKind, now_ms: u64) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.last_update_ms = now_ms;
        self.updates.push(RegionUpdate { kind, timestamp_ms: now_ms, seq });
    }
}

// ── Region Registry ───────────────────────────────────────────

/// Manages multiple live regions on a page.
#[derive(Debug, Default)]
pub struct LiveRegionRegistry {
    regions: HashMap<String, LiveRegion>,
}

impl LiveRegionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, region: LiveRegion) {
        self.regions.insert(region.id.clone(), region);
    }

    pub fn unregister(&mut self, id: &str) {
        self.regions.remove(id);
    }

    pub fn get(&self, id: &str) -> Option<&LiveRegion> {
        self.regions.get(id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut LiveRegion> {
        self.regions.get_mut(id)
    }

    /// Collect all pending announcements across regions.
    pub fn pending_announcements(&self) -> Vec<(&str, String)> {
        let mut out = Vec::new();
        for region in self.regions.values() {
            if let Some(text) = region.announcement_text() {
                out.push((region.id.as_str(), text));
            }
        }
        out
    }

    pub fn count(&self) -> usize {
        self.regions.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polite_region_basic() {
        let mut r = LiveRegion::new("status", RegionPoliteness::Polite);
        assert!(r.add_content("Item saved", 100));
        assert_eq!(r.content(), &["Item saved"]);
    }

    #[test]
    fn off_region_rejects() {
        let mut r = LiveRegion::new("hidden", RegionPoliteness::Off);
        assert!(!r.add_content("ignored", 0));
    }

    #[test]
    fn busy_suppresses() {
        let mut r = LiveRegion::new("log", RegionPoliteness::Polite);
        r.set_busy(true);
        assert!(!r.add_content("loading...", 0));
        r.set_busy(false);
        assert!(r.add_content("done", 0));
    }

    #[test]
    fn throttling() {
        let mut r = LiveRegion::new("feed", RegionPoliteness::Polite);
        r.set_throttle_ms(500);
        assert!(r.add_content("msg1", 1000));
        assert!(!r.add_content("msg2", 1200)); // too soon
        assert!(r.add_content("msg3", 1500));  // ok
    }

    #[test]
    fn relevant_additions_only() {
        let mut r = LiveRegion::new("list", RegionPoliteness::Polite);
        r.set_relevant(RelevantContent { additions: true, removals: false, text: false });
        assert!(r.add_content("new item", 0));
        assert!(!r.remove_content(0, 0));
    }

    #[test]
    fn relevant_removals() {
        let mut r = LiveRegion::new("list", RegionPoliteness::Polite);
        r.set_relevant(RelevantContent::all());
        r.add_content("item", 0);
        assert!(r.remove_content(0, 0));
        assert!(r.content().is_empty());
    }

    #[test]
    fn text_change() {
        let mut r = LiveRegion::new("counter", RegionPoliteness::Assertive);
        r.add_content("Count: 0", 0);
        r.update_content(0, "Count: 1", 100);
        assert_eq!(r.content(), &["Count: 1"]);
        let last = r.updates().last().unwrap();
        if let UpdateKind::TextChange { old, new } = &last.kind {
            assert_eq!(old, "Count: 0");
            assert_eq!(new, "Count: 1");
        } else {
            panic!("expected TextChange");
        }
    }

    #[test]
    fn atomic_announcement() {
        let mut r = LiveRegion::new("timer", RegionPoliteness::Assertive);
        r.set_atomic(true);
        r.add_content("Time:", 0);
        r.add_content("12:00", 0);
        assert_eq!(r.announcement_text(), Some("Time: 12:00".into()));
    }

    #[test]
    fn non_atomic_latest_only() {
        let mut r = LiveRegion::new("log", RegionPoliteness::Polite);
        r.add_content("line 1", 0);
        r.add_content("line 2", 0);
        assert_eq!(r.announcement_text(), Some("line 2".into()));
    }

    #[test]
    fn relevant_attr_rendering() {
        assert_eq!(RelevantContent::all().to_attr(), "all");
        assert_eq!(RelevantContent::default_relevant().to_attr(), "additions text");
        let removals_only = RelevantContent { additions: false, removals: true, text: false };
        assert_eq!(removals_only.to_attr(), "removals");
    }

    #[test]
    fn render_html() {
        let mut r = LiveRegion::new("status", RegionPoliteness::Polite);
        r.set_atomic(true);
        r.add_content("Hello", 0);
        let html = r.render();
        assert!(html.contains("aria-live=\"polite\""));
        assert!(html.contains("aria-atomic=\"true\""));
        assert!(html.contains("Hello"));
    }

    #[test]
    fn nested_children() {
        let mut r = LiveRegion::new("parent", RegionPoliteness::Polite);
        r.add_child("child-1");
        r.add_child("child-2");
        r.add_child("child-1"); // duplicate
        assert_eq!(r.children().len(), 2);
        r.remove_child("child-1");
        assert_eq!(r.children(), &["child-2"]);
    }

    #[test]
    fn registry_basics() {
        let mut reg = LiveRegionRegistry::new();
        let mut r = LiveRegion::new("s1", RegionPoliteness::Polite);
        r.add_content("hello", 0);
        reg.register(r);
        assert_eq!(reg.count(), 1);
        let anns = reg.pending_announcements();
        assert_eq!(anns.len(), 1);
    }

    #[test]
    fn update_history() {
        let mut r = LiveRegion::new("log", RegionPoliteness::Polite);
        r.add_content("a", 0);
        r.add_content("b", 100);
        assert_eq!(r.updates().len(), 2);
        assert_eq!(r.updates()[0].seq, 1);
        assert_eq!(r.updates()[1].seq, 2);
        r.clear_updates();
        assert!(r.updates().is_empty());
    }
}
