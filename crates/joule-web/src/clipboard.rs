//! Clipboard state management with history and format detection.
//!
//! Replaces clipboard.js with a pure-Rust clipboard model. No browser APIs —
//! actual clipboard access is injected at the application boundary.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ── Clipboard Item ──

/// A single clipboard entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipboardItem {
    pub text: Option<String>,
    pub html: Option<String>,
    pub mime_type: String,
    pub timestamp: DateTime<Utc>,
}

impl ClipboardItem {
    /// Create a plain-text clipboard item.
    pub fn text(s: &str) -> Self {
        Self {
            text: Some(s.to_string()),
            html: None,
            mime_type: "text/plain".to_string(),
            timestamp: Utc::now(),
        }
    }

    /// Create an HTML clipboard item with a plain-text fallback.
    pub fn html(html: &str, fallback_text: &str) -> Self {
        Self {
            text: Some(fallback_text.to_string()),
            html: Some(html.to_string()),
            mime_type: "text/html".to_string(),
            timestamp: Utc::now(),
        }
    }

    /// Create a clipboard item with a custom MIME type.
    pub fn with_mime(mime: &str, text: &str) -> Self {
        Self {
            text: Some(text.to_string()),
            html: None,
            mime_type: mime.to_string(),
            timestamp: Utc::now(),
        }
    }
}

// ── Clipboard State ──

/// Tracks current clipboard content and history.
#[derive(Debug, Clone)]
pub struct ClipboardState {
    pub current: Option<ClipboardItem>,
    pub history: Vec<ClipboardItem>,
    pub max_history: usize,
}

impl ClipboardState {
    pub fn new() -> Self {
        Self {
            current: None,
            history: Vec::new(),
            max_history: 50,
        }
    }

    pub fn with_max_history(max: usize) -> Self {
        Self {
            max_history: max,
            ..Self::new()
        }
    }

    /// Copy an item to the clipboard.
    pub fn copy(&mut self, item: ClipboardItem) {
        self.current = Some(item.clone());
        self.history.push(item);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Shorthand: copy plain text.
    pub fn copy_text(&mut self, text: &str) {
        self.copy(ClipboardItem::text(text));
    }

    /// Get the current clipboard item.
    pub fn paste(&self) -> Option<&ClipboardItem> {
        self.current.as_ref()
    }

    /// Get the current clipboard text.
    pub fn paste_text(&self) -> Option<&str> {
        self.current.as_ref().and_then(|i| i.text.as_deref())
    }

    /// Clear the current item (history is preserved).
    pub fn clear(&mut self) {
        self.current = None;
    }

    /// Clear all history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    pub fn history(&self) -> &[ClipboardItem] {
        &self.history
    }

    pub fn history_item(&self, index: usize) -> Option<&ClipboardItem> {
        self.history.get(index)
    }
}

impl Default for ClipboardState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Clipboard Format Detection ──

/// Detected format of clipboard text content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipboardFormat {
    PlainText,
    Html,
    Url,
    Email,
    Json,
    Code,
    Markdown,
    Unknown,
}

/// Detect the likely format of a text string.
pub fn detect_format(text: &str) -> ClipboardFormat {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return ClipboardFormat::Unknown;
    }

    // URL
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return ClipboardFormat::Url;
    }

    // HTML
    if trimmed.starts_with('<') && trimmed.contains('>') {
        return ClipboardFormat::Html;
    }

    // JSON
    if (trimmed.starts_with('{') && trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && trimmed.ends_with(']'))
    {
        return ClipboardFormat::Json;
    }

    // Email
    if trimmed.contains('@') && trimmed.contains('.') && !trimmed.contains(' ') {
        return ClipboardFormat::Email;
    }

    // Markdown
    if trimmed.contains("# ")
        || trimmed.contains("**")
        || trimmed.contains("](")
        || trimmed.contains("```")
    {
        return ClipboardFormat::Markdown;
    }

    // Code heuristics
    if trimmed.contains("fn ")
        || trimmed.contains("function ")
        || trimmed.contains("class ")
        || trimmed.contains("def ")
    {
        return ClipboardFormat::Code;
    }
    // Indented lines (common in code)
    let indented_lines = trimmed
        .lines()
        .filter(|l| l.starts_with("    ") || l.starts_with('\t'))
        .count();
    if indented_lines > 0 && indented_lines as f64 / trimmed.lines().count().max(1) as f64 > 0.3 {
        return ClipboardFormat::Code;
    }

    ClipboardFormat::PlainText
}

// ── Clipboard Watcher ──

/// Polls for clipboard changes and notifies registered handlers.
#[derive(Debug, Clone)]
pub struct ClipboardWatcher {
    pub on_change_handlers: Vec<u64>,
    pub last_content: Option<String>,
}

impl ClipboardWatcher {
    pub fn new() -> Self {
        Self {
            on_change_handlers: Vec::new(),
            last_content: None,
        }
    }

    /// Register a handler ID to be notified on change.
    pub fn on_change(&mut self, handler_id: u64) {
        self.on_change_handlers.push(handler_id);
    }

    /// Check if content has changed. Returns handler IDs to notify if so.
    pub fn check(&mut self, current_content: &str) -> Option<Vec<u64>> {
        let changed = match &self.last_content {
            None => true,
            Some(prev) => prev != current_content,
        };
        self.last_content = Some(current_content.to_string());
        if changed {
            Some(self.on_change_handlers.clone())
        } else {
            None
        }
    }

    pub fn remove_handler(&mut self, handler_id: u64) {
        self.on_change_handlers.retain(|id| *id != handler_id);
    }
}

impl Default for ClipboardWatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_paste_roundtrip() {
        let mut state = ClipboardState::new();
        state.copy_text("hello");
        assert_eq!(state.paste_text(), Some("hello"));
    }

    #[test]
    fn history_tracks() {
        let mut state = ClipboardState::new();
        state.copy_text("a");
        state.copy_text("b");
        state.copy_text("c");
        assert_eq!(state.history().len(), 3);
        assert_eq!(state.history_item(0).unwrap().text.as_deref(), Some("a"));
    }

    #[test]
    fn max_history_enforced() {
        let mut state = ClipboardState::with_max_history(3);
        for i in 0..5 {
            state.copy_text(&format!("item{i}"));
        }
        assert_eq!(state.history().len(), 3);
        assert_eq!(
            state.history_item(0).unwrap().text.as_deref(),
            Some("item2")
        );
    }

    #[test]
    fn clear_current_only() {
        let mut state = ClipboardState::new();
        state.copy_text("hello");
        state.clear();
        assert!(state.paste().is_none());
        assert_eq!(state.history().len(), 1);
    }

    #[test]
    fn detect_format_url() {
        assert_eq!(
            detect_format("https://example.com"),
            ClipboardFormat::Url
        );
    }

    #[test]
    fn detect_format_html() {
        assert_eq!(
            detect_format("<div>hello</div>"),
            ClipboardFormat::Html
        );
    }

    #[test]
    fn detect_format_json() {
        assert_eq!(
            detect_format("{\"key\": \"value\"}"),
            ClipboardFormat::Json
        );
        assert_eq!(detect_format("[1, 2, 3]"), ClipboardFormat::Json);
    }

    #[test]
    fn detect_format_email() {
        assert_eq!(
            detect_format("user@example.com"),
            ClipboardFormat::Email
        );
    }

    #[test]
    fn copy_text_shorthand() {
        let mut state = ClipboardState::new();
        state.copy_text("quick");
        let item = state.paste().unwrap();
        assert_eq!(item.mime_type, "text/plain");
        assert_eq!(item.text.as_deref(), Some("quick"));
    }

    #[test]
    fn watcher_detects_change() {
        let mut w = ClipboardWatcher::new();
        w.on_change(1);
        w.on_change(2);
        let handlers = w.check("hello").unwrap();
        assert_eq!(handlers, vec![1, 2]);
    }

    #[test]
    fn watcher_ignores_same_content() {
        let mut w = ClipboardWatcher::new();
        w.on_change(1);
        w.check("hello");
        assert!(w.check("hello").is_none());
    }

    #[test]
    fn history_item_by_index() {
        let mut state = ClipboardState::new();
        state.copy_text("first");
        state.copy_text("second");
        assert_eq!(
            state.history_item(1).unwrap().text.as_deref(),
            Some("second")
        );
        assert!(state.history_item(5).is_none());
    }

    #[test]
    fn html_item_has_fallback_text() {
        let item = ClipboardItem::html("<b>hi</b>", "hi");
        assert_eq!(item.html.as_deref(), Some("<b>hi</b>"));
        assert_eq!(item.text.as_deref(), Some("hi"));
        assert_eq!(item.mime_type, "text/html");
    }
}
