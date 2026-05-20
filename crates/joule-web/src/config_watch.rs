//! Poll-based configuration file watcher with checksum comparison,
//! debounced reload, atomic config swap, version tracking, change history,
//! and rollback.
//!
//! Since we avoid external crates (inotify, kqueue, etc.), change detection
//! uses content hashing: if the hash changes between polls, we fire the
//! registered callbacks.

use std::collections::HashMap;

// ── Types ───────────────────────────────────────────────────────

/// A simple FNV-1a 64-bit hash for change detection.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A snapshot of a configuration at a point in time.
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    pub version: u64,
    pub content: String,
    pub hash: u64,
    pub timestamp_ms: u64,
}

/// Describes what changed between two snapshots.
#[derive(Debug, Clone)]
pub struct ChangeRecord {
    pub from_version: u64,
    pub to_version: u64,
    pub from_hash: u64,
    pub to_hash: u64,
    pub timestamp_ms: u64,
}

/// Status of a watched file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchStatus {
    /// No content has been loaded yet.
    Pending,
    /// Content is loaded and unchanged since last poll.
    Unchanged,
    /// Content changed on the most recent poll.
    Changed,
    /// An error occurred during the last poll.
    Error,
}

// ── WatchedFile ─────────────────────────────────────────────────

/// Tracks a single configuration source.
struct WatchedFile {
    /// Identifier for this watch.
    id: String,
    /// Current content hash (0 = not yet loaded).
    current_hash: u64,
    /// Current content.
    current_content: String,
    /// Monotonically increasing version counter.
    version: u64,
    /// Debounce: minimum milliseconds between reload actions.
    debounce_ms: u64,
    /// Timestamp of last reload (in caller-supplied milliseconds).
    last_reload_ms: u64,
    /// Status of the most recent poll.
    status: WatchStatus,
    /// History of change records.
    history: Vec<ChangeRecord>,
    /// Snapshots for rollback (most recent first).
    snapshots: Vec<ConfigSnapshot>,
    /// Maximum number of snapshots to retain.
    max_snapshots: usize,
}

// ── ConfigWatcher ───────────────────────────────────────────────

/// Manages one or more watched configuration sources, with poll-based
/// change detection and rollback support.
pub struct ConfigWatcher {
    files: HashMap<String, WatchedFile>,
}

impl ConfigWatcher {
    pub fn new() -> Self {
        Self { files: HashMap::new() }
    }

    /// Register a new watch.
    ///
    /// - `id`: unique identifier for this config source.
    /// - `debounce_ms`: minimum interval between reloads.
    /// - `max_snapshots`: how many old versions to retain for rollback.
    pub fn watch(
        &mut self,
        id: impl Into<String>,
        debounce_ms: u64,
        max_snapshots: usize,
    ) {
        let id = id.into();
        self.files.insert(id.clone(), WatchedFile {
            id,
            current_hash: 0,
            current_content: String::new(),
            version: 0,
            debounce_ms,
            last_reload_ms: 0,
            status: WatchStatus::Pending,
            history: Vec::new(),
            snapshots: Vec::new(),
            max_snapshots,
        });
    }

    /// Remove a watch.
    pub fn unwatch(&mut self, id: &str) -> bool {
        self.files.remove(id).is_some()
    }

    /// Poll a watched config source with new content and a timestamp.
    ///
    /// Returns `true` if the content was accepted (changed and debounce
    /// window passed), `false` otherwise.
    pub fn poll(
        &mut self,
        id: &str,
        content: &str,
        now_ms: u64,
    ) -> bool {
        let file = match self.files.get_mut(id) {
            Some(f) => f,
            None => return false,
        };

        let new_hash = fnv1a_hash(content.as_bytes());

        // No change.
        if new_hash == file.current_hash {
            if file.status != WatchStatus::Pending {
                file.status = WatchStatus::Unchanged;
            }
            return false;
        }

        // Debounce check.
        if file.version > 0 && now_ms.saturating_sub(file.last_reload_ms) < file.debounce_ms {
            return false;
        }

        // Save old snapshot before overwriting.
        if file.version > 0 {
            let snapshot = ConfigSnapshot {
                version: file.version,
                content: file.current_content.clone(),
                hash: file.current_hash,
                timestamp_ms: file.last_reload_ms,
            };
            file.snapshots.insert(0, snapshot);
            if file.snapshots.len() > file.max_snapshots {
                file.snapshots.pop();
            }

            file.history.push(ChangeRecord {
                from_version: file.version,
                to_version: file.version + 1,
                from_hash: file.current_hash,
                to_hash: new_hash,
                timestamp_ms: now_ms,
            });
        }

        file.current_hash = new_hash;
        file.current_content = content.to_string();
        file.version += 1;
        file.last_reload_ms = now_ms;
        file.status = WatchStatus::Changed;
        true
    }

    /// Mark a watch as having encountered an error (e.g., file unreadable).
    pub fn mark_error(&mut self, id: &str) {
        if let Some(f) = self.files.get_mut(id) {
            f.status = WatchStatus::Error;
        }
    }

    /// Get the current content of a watched config source.
    pub fn content(&self, id: &str) -> Option<&str> {
        self.files.get(id).map(|f| f.current_content.as_str())
    }

    /// Get the current version of a watched config source.
    pub fn version(&self, id: &str) -> Option<u64> {
        self.files.get(id).map(|f| f.version)
    }

    /// Get the current hash of a watched config source.
    pub fn hash(&self, id: &str) -> Option<u64> {
        self.files.get(id).map(|f| f.current_hash)
    }

    /// Get the current status.
    pub fn status(&self, id: &str) -> Option<WatchStatus> {
        self.files.get(id).map(|f| f.status)
    }

    /// Get the change history for a source.
    pub fn history(&self, id: &str) -> Option<&[ChangeRecord]> {
        self.files.get(id).map(|f| f.history.as_slice())
    }

    /// Get available rollback snapshots.
    pub fn snapshots(&self, id: &str) -> Option<&[ConfigSnapshot]> {
        self.files.get(id).map(|f| f.snapshots.as_slice())
    }

    /// Roll back to a specific version. Returns the restored content or None.
    pub fn rollback(&mut self, id: &str, target_version: u64, now_ms: u64) -> Option<String> {
        let file = self.files.get_mut(id)?;
        let idx = file.snapshots.iter().position(|s| s.version == target_version)?;
        let snapshot = file.snapshots.remove(idx);

        // Record the rollback as a change.
        file.history.push(ChangeRecord {
            from_version: file.version,
            to_version: file.version + 1,
            from_hash: file.current_hash,
            to_hash: snapshot.hash,
            timestamp_ms: now_ms,
        });

        // Save current as a snapshot.
        let current_snap = ConfigSnapshot {
            version: file.version,
            content: file.current_content.clone(),
            hash: file.current_hash,
            timestamp_ms: file.last_reload_ms,
        };
        file.snapshots.insert(0, current_snap);
        if file.snapshots.len() > file.max_snapshots {
            file.snapshots.pop();
        }

        file.current_content = snapshot.content.clone();
        file.current_hash = snapshot.hash;
        file.version += 1;
        file.last_reload_ms = now_ms;
        file.status = WatchStatus::Changed;

        Some(snapshot.content)
    }

    /// List all watched IDs.
    pub fn watched_ids(&self) -> Vec<&str> {
        let mut ids: Vec<_> = self.files.keys().map(|k| k.as_str()).collect();
        ids.sort();
        ids
    }

    /// Total number of watches.
    pub fn watch_count(&self) -> usize {
        self.files.len()
    }
}

impl Default for ConfigWatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_status() {
        let mut w = ConfigWatcher::new();
        w.watch("app.json", 100, 5);
        assert_eq!(w.status("app.json"), Some(WatchStatus::Pending));
        assert_eq!(w.version("app.json"), Some(0));
    }

    #[test]
    fn first_poll_accepts() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        assert!(w.poll("c", "v1", 0));
        assert_eq!(w.content("c"), Some("v1"));
        assert_eq!(w.version("c"), Some(1));
        assert_eq!(w.status("c"), Some(WatchStatus::Changed));
    }

    #[test]
    fn unchanged_content() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.poll("c", "same", 0);
        assert!(!w.poll("c", "same", 100));
        assert_eq!(w.status("c"), Some(WatchStatus::Unchanged));
    }

    #[test]
    fn changed_content() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.poll("c", "v1", 0);
        assert!(w.poll("c", "v2", 100));
        assert_eq!(w.content("c"), Some("v2"));
        assert_eq!(w.version("c"), Some(2));
    }

    #[test]
    fn debounce_blocks_rapid_change() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 500, 5);
        w.poll("c", "v1", 0);
        // Change within debounce window → rejected.
        assert!(!w.poll("c", "v2", 200));
        assert_eq!(w.content("c"), Some("v1"));
        // After debounce window → accepted.
        assert!(w.poll("c", "v2", 600));
        assert_eq!(w.content("c"), Some("v2"));
    }

    #[test]
    fn change_history_tracked() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.poll("c", "v1", 0);
        w.poll("c", "v2", 100);
        w.poll("c", "v3", 200);

        let hist = w.history("c").unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].from_version, 1);
        assert_eq!(hist[0].to_version, 2);
        assert_eq!(hist[1].from_version, 2);
        assert_eq!(hist[1].to_version, 3);
    }

    #[test]
    fn snapshots_retained() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 3);
        w.poll("c", "v1", 0);
        w.poll("c", "v2", 10);
        w.poll("c", "v3", 20);
        w.poll("c", "v4", 30);

        let snaps = w.snapshots("c").unwrap();
        // max_snapshots=3, versions 1,2,3 should be retained.
        assert_eq!(snaps.len(), 3);
        assert_eq!(snaps[0].version, 3);
        assert_eq!(snaps[1].version, 2);
        assert_eq!(snaps[2].version, 1);
    }

    #[test]
    fn snapshot_cap_enforced() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 2); // keep only 2 snapshots
        w.poll("c", "v1", 0);
        w.poll("c", "v2", 10);
        w.poll("c", "v3", 20);
        w.poll("c", "v4", 30);

        let snaps = w.snapshots("c").unwrap();
        assert_eq!(snaps.len(), 2);
        // Most recent first: v3, v2
        assert_eq!(snaps[0].version, 3);
        assert_eq!(snaps[1].version, 2);
    }

    #[test]
    fn rollback_restores_old_version() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.poll("c", "v1", 0);
        w.poll("c", "v2", 10);
        w.poll("c", "v3", 20);

        let restored = w.rollback("c", 1, 30);
        assert_eq!(restored, Some("v1".to_string()));
        assert_eq!(w.content("c"), Some("v1"));
        assert_eq!(w.version("c"), Some(4)); // version incremented
    }

    #[test]
    fn rollback_nonexistent_version() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.poll("c", "v1", 0);
        assert!(w.rollback("c", 999, 100).is_none());
    }

    #[test]
    fn unwatch() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        assert!(w.unwatch("c"));
        assert!(!w.unwatch("c")); // already removed
        assert_eq!(w.watch_count(), 0);
    }

    #[test]
    fn mark_error() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.mark_error("c");
        assert_eq!(w.status("c"), Some(WatchStatus::Error));
    }

    #[test]
    fn poll_unknown_id() {
        let mut w = ConfigWatcher::new();
        assert!(!w.poll("nope", "data", 0));
    }

    #[test]
    fn watched_ids() {
        let mut w = ConfigWatcher::new();
        w.watch("b", 0, 5);
        w.watch("a", 0, 5);
        assert_eq!(w.watched_ids(), vec!["a", "b"]);
    }

    #[test]
    fn hash_changes_with_content() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 5);
        w.poll("c", "hello", 0);
        let h1 = w.hash("c").unwrap();
        w.poll("c", "world", 10);
        let h2 = w.hash("c").unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn fnv1a_basic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        let h3 = fnv1a_hash(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn multiple_watches() {
        let mut w = ConfigWatcher::new();
        w.watch("a.json", 0, 3);
        w.watch("b.json", 0, 3);
        w.poll("a.json", "content_a", 0);
        w.poll("b.json", "content_b", 0);
        assert_eq!(w.content("a.json"), Some("content_a"));
        assert_eq!(w.content("b.json"), Some("content_b"));
        assert_eq!(w.watch_count(), 2);
    }

    #[test]
    fn rollback_saves_current_as_snapshot() {
        let mut w = ConfigWatcher::new();
        w.watch("c", 0, 10);
        w.poll("c", "v1", 0);
        w.poll("c", "v2", 10);

        // Before rollback: snapshot has v1.
        assert_eq!(w.snapshots("c").unwrap().len(), 1);

        w.rollback("c", 1, 20);
        // After rollback: snapshot should contain v2 (the version we rolled back from).
        let snaps = w.snapshots("c").unwrap();
        assert!(snaps.iter().any(|s| s.version == 2));
    }
}
