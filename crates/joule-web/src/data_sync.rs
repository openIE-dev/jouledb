//! Data synchronization: version-tracked records, last-write-wins merge,
//! conflict detection, push/pull, tombstone cleanup, and batch sync.

use serde_json::Value;
use std::collections::HashMap;

// ── Sync Record ──────────────────────────────────────────────────

/// A versioned, timestamped data record that supports soft deletion.
#[derive(Debug, Clone)]
pub struct SyncRecord {
    pub id: String,
    pub version: u64,
    pub data: Value,
    pub modified_at: i64,
    pub deleted: bool,
}

impl SyncRecord {
    pub fn new(id: &str, version: u64, data: Value, modified_at: i64) -> Self {
        Self {
            id: id.to_string(),
            version,
            data,
            modified_at,
            deleted: false,
        }
    }

    pub fn tombstone(id: &str, version: u64, modified_at: i64) -> Self {
        Self {
            id: id.to_string(),
            version,
            data: Value::Null,
            modified_at,
            deleted: true,
        }
    }
}

// ── Sync State ───────────────────────────────────────────────────

/// Tracks the synchronization state.
#[derive(Debug, Clone)]
pub struct SyncState {
    pub last_sync_version: u64,
}

impl SyncState {
    pub fn new() -> Self {
        Self { last_sync_version: 0 }
    }

    pub fn at_version(version: u64) -> Self {
        Self { last_sync_version: version }
    }
}

impl Default for SyncState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Conflict ─────────────────────────────────────────────────────

/// A merge conflict: the same record was modified on both sides.
#[derive(Debug, Clone)]
pub struct SyncConflict {
    pub id: String,
    pub local: SyncRecord,
    pub remote: SyncRecord,
}

// ── Merge Result ─────────────────────────────────────────────────

/// The outcome of a merge operation.
#[derive(Debug)]
pub struct MergeResult {
    pub merged: Vec<SyncRecord>,
    pub conflicts: Vec<SyncConflict>,
}

// ── SyncStore ────────────────────────────────────────────────────

/// An in-memory synchronized data store.
#[derive(Debug)]
pub struct SyncStore {
    records: HashMap<String, SyncRecord>,
    next_version: u64,
    pub state: SyncState,
}

impl SyncStore {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            next_version: 1,
            state: SyncState::new(),
        }
    }

    /// Insert or update a record, assigning a new version.
    pub fn put(&mut self, id: &str, data: Value, modified_at: i64) {
        let version = self.next_version;
        self.next_version += 1;
        self.records.insert(
            id.to_string(),
            SyncRecord::new(id, version, data, modified_at),
        );
    }

    /// Soft-delete a record.
    pub fn delete(&mut self, id: &str, modified_at: i64) {
        let version = self.next_version;
        self.next_version += 1;
        self.records.insert(
            id.to_string(),
            SyncRecord::tombstone(id, version, modified_at),
        );
    }

    /// Get a record by id.
    pub fn get(&self, id: &str) -> Option<&SyncRecord> {
        self.records.get(id).filter(|r| !r.deleted)
    }

    /// All records (including tombstones).
    pub fn all_records(&self) -> Vec<&SyncRecord> {
        self.records.values().collect()
    }

    /// Compute changes since a given version (for push).
    pub fn changes_since(&self, since_version: u64) -> Vec<&SyncRecord> {
        let mut changed: Vec<&SyncRecord> = self
            .records
            .values()
            .filter(|r| r.version > since_version)
            .collect();
        changed.sort_by_key(|r| r.version);
        changed
    }

    /// Push: return records changed since last sync for sending to remote.
    pub fn push_changes(&self) -> Vec<&SyncRecord> {
        self.changes_since(self.state.last_sync_version)
    }

    /// Pull: apply remote changes using last-write-wins merge.
    /// Returns conflicts where both sides modified the same record.
    pub fn pull_and_apply(&mut self, remote_changes: Vec<SyncRecord>) -> Vec<SyncConflict> {
        let mut conflicts = Vec::new();

        for remote in remote_changes {
            if let Some(local) = self.records.get(&remote.id) {
                // Both modified: detect conflict
                if local.version > self.state.last_sync_version {
                    // Local was also modified since last sync
                    if local.modified_at != remote.modified_at {
                        conflicts.push(SyncConflict {
                            id: remote.id.clone(),
                            local: local.clone(),
                            remote: remote.clone(),
                        });
                    }
                    // Last-write-wins
                    if remote.modified_at >= local.modified_at {
                        self.records.insert(remote.id.clone(), remote);
                    }
                } else {
                    // Only remote changed, apply directly
                    self.records.insert(remote.id.clone(), remote);
                }
            } else {
                // New from remote
                self.records.insert(remote.id.clone(), remote);
            }
        }

        conflicts
    }

    /// Merge two sets of changes (local + remote) with conflict detection.
    pub fn merge(
        local_changes: &[SyncRecord],
        remote_changes: &[SyncRecord],
    ) -> MergeResult {
        let mut merged = Vec::new();
        let mut conflicts = Vec::new();

        let local_map: HashMap<&str, &SyncRecord> =
            local_changes.iter().map(|r| (r.id.as_str(), r)).collect();
        let remote_map: HashMap<&str, &SyncRecord> =
            remote_changes.iter().map(|r| (r.id.as_str(), r)).collect();

        // Process all unique ids
        let mut all_ids: Vec<&str> = local_map.keys().copied().collect();
        for id in remote_map.keys() {
            if !local_map.contains_key(id) {
                all_ids.push(id);
            }
        }

        for id in all_ids {
            match (local_map.get(id), remote_map.get(id)) {
                (Some(local), Some(remote)) => {
                    conflicts.push(SyncConflict {
                        id: id.to_string(),
                        local: (*local).clone(),
                        remote: (*remote).clone(),
                    });
                    // LWW
                    if remote.modified_at >= local.modified_at {
                        merged.push((*remote).clone());
                    } else {
                        merged.push((*local).clone());
                    }
                }
                (Some(local), None) => merged.push((*local).clone()),
                (None, Some(remote)) => merged.push((*remote).clone()),
                (None, None) => {}
            }
        }

        MergeResult { merged, conflicts }
    }

    /// Mark sync complete at the current version.
    pub fn complete_sync(&mut self) {
        self.state.last_sync_version = self.next_version.saturating_sub(1);
    }

    /// Remove tombstones older than the given threshold timestamp.
    pub fn cleanup_tombstones(&mut self, older_than: i64) {
        self.records.retain(|_, r| !r.deleted || r.modified_at >= older_than);
    }

    /// Batch sync: apply a batch of records, auto-versioning each.
    pub fn batch_apply(&mut self, records: Vec<SyncRecord>) {
        for rec in records {
            let version = self.next_version;
            self.next_version += 1;
            let mut applied = rec;
            applied.version = version;
            self.records.insert(applied.id.clone(), applied);
        }
    }

    /// Number of live (non-deleted) records.
    pub fn live_count(&self) -> usize {
        self.records.values().filter(|r| !r.deleted).count()
    }

    /// Number of tombstones.
    pub fn tombstone_count(&self) -> usize {
        self.records.values().filter(|r| r.deleted).count()
    }
}

impl Default for SyncStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_put_and_get() {
        let mut store = SyncStore::new();
        store.put("a", json!({"name": "Alice"}), 1000);
        let rec = store.get("a").unwrap();
        assert_eq!(rec.data, json!({"name": "Alice"}));
        assert_eq!(rec.version, 1);
    }

    #[test]
    fn test_delete_tombstone() {
        let mut store = SyncStore::new();
        store.put("a", json!(1), 1000);
        store.delete("a", 2000);
        assert!(store.get("a").is_none());
        assert_eq!(store.tombstone_count(), 1);
    }

    #[test]
    fn test_changes_since() {
        let mut store = SyncStore::new();
        store.put("a", json!(1), 100);
        store.put("b", json!(2), 200);
        store.put("c", json!(3), 300);

        let since_1 = store.changes_since(1);
        assert_eq!(since_1.len(), 2);
        assert_eq!(since_1[0].id, "b");
        assert_eq!(since_1[1].id, "c");
    }

    #[test]
    fn test_push_changes() {
        let mut store = SyncStore::new();
        store.put("a", json!(1), 100);
        store.complete_sync();
        store.put("b", json!(2), 200);

        let changes = store.push_changes();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].id, "b");
    }

    #[test]
    fn test_pull_new_record() {
        let mut store = SyncStore::new();
        let remote = vec![SyncRecord::new("x", 10, json!("hello"), 5000)];
        let conflicts = store.pull_and_apply(remote);
        assert!(conflicts.is_empty());
        assert_eq!(store.get("x").unwrap().data, json!("hello"));
    }

    #[test]
    fn test_pull_conflict_lww() {
        let mut store = SyncStore::new();
        store.put("a", json!("local"), 1000);
        // Remote has later timestamp -> wins
        let remote = vec![SyncRecord::new("a", 5, json!("remote"), 2000)];
        let conflicts = store.pull_and_apply(remote);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(store.get("a").unwrap().data, json!("remote"));
    }

    #[test]
    fn test_pull_conflict_local_wins() {
        let mut store = SyncStore::new();
        store.put("a", json!("local"), 5000);
        let remote = vec![SyncRecord::new("a", 5, json!("remote"), 1000)];
        let conflicts = store.pull_and_apply(remote);
        assert_eq!(conflicts.len(), 1);
        assert_eq!(store.get("a").unwrap().data, json!("local"));
    }

    #[test]
    fn test_merge_no_overlap() {
        let local = vec![SyncRecord::new("a", 1, json!(1), 100)];
        let remote = vec![SyncRecord::new("b", 2, json!(2), 200)];
        let result = SyncStore::merge(&local, &remote);
        assert_eq!(result.merged.len(), 2);
        assert!(result.conflicts.is_empty());
    }

    #[test]
    fn test_merge_with_conflict() {
        let local = vec![SyncRecord::new("a", 1, json!("L"), 100)];
        let remote = vec![SyncRecord::new("a", 2, json!("R"), 200)];
        let result = SyncStore::merge(&local, &remote);
        assert_eq!(result.conflicts.len(), 1);
        // Remote wins (later modified_at)
        assert_eq!(result.merged[0].data, json!("R"));
    }

    #[test]
    fn test_cleanup_tombstones() {
        let mut store = SyncStore::new();
        store.put("a", json!(1), 100);
        store.delete("a", 200);
        store.put("b", json!(2), 300);
        store.delete("b", 5000);

        store.cleanup_tombstones(1000);
        // 'a' tombstone (modified_at=200) < 1000, removed
        // 'b' tombstone (modified_at=5000) >= 1000, kept
        assert_eq!(store.tombstone_count(), 1);
    }

    #[test]
    fn test_batch_apply() {
        let mut store = SyncStore::new();
        let batch = vec![
            SyncRecord::new("a", 0, json!(1), 100),
            SyncRecord::new("b", 0, json!(2), 200),
            SyncRecord::new("c", 0, json!(3), 300),
        ];
        store.batch_apply(batch);
        assert_eq!(store.live_count(), 3);
        // Versions were reassigned
        assert_eq!(store.get("a").unwrap().version, 1);
        assert_eq!(store.get("c").unwrap().version, 3);
    }

    #[test]
    fn test_live_count() {
        let mut store = SyncStore::new();
        store.put("a", json!(1), 100);
        store.put("b", json!(2), 200);
        store.delete("a", 300);
        assert_eq!(store.live_count(), 1);
        assert_eq!(store.tombstone_count(), 1);
    }

    #[test]
    fn test_complete_sync_resets_push() {
        let mut store = SyncStore::new();
        store.put("a", json!(1), 100);
        store.put("b", json!(2), 200);
        assert_eq!(store.push_changes().len(), 2);
        store.complete_sync();
        assert_eq!(store.push_changes().len(), 0);
    }
}
