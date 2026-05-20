//! State synchronization protocol — version-tracked state with full/incremental sync.
//!
//! Provides `SyncState` (version + checksum + data), `StateSyncManager` for
//! tracking local and remote states, full and incremental sync modes, conflict
//! detection on version divergence, request/response protocol messages,
//! bandwidth-aware batching, sync statistics, and configurable sync interval.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Sync domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncError {
    /// Remote peer not found.
    PeerNotFound(u64),
    /// Version conflict detected.
    VersionConflict { local: u64, remote: u64 },
    /// Checksum mismatch after sync.
    ChecksumMismatch { expected: u64, actual: u64 },
    /// Bandwidth budget exceeded for this sync cycle.
    BandwidthExceeded { budget: usize, required: usize },
    /// State key not found.
    KeyNotFound(String),
    /// Sync already in progress for this peer.
    SyncInProgress(u64),
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PeerNotFound(id) => write!(f, "peer not found: {id}"),
            Self::VersionConflict { local, remote } => {
                write!(f, "version conflict: local={local}, remote={remote}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected={expected}, actual={actual}")
            }
            Self::BandwidthExceeded { budget, required } => {
                write!(f, "bandwidth exceeded: budget={budget}, required={required}")
            }
            Self::KeyNotFound(k) => write!(f, "key not found: {k}"),
            Self::SyncInProgress(id) => write!(f, "sync already in progress for peer {id}"),
        }
    }
}

impl std::error::Error for SyncError {}

// ── Sync Mode ───────────────────────────────────────────────────

/// Full or incremental synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Transfer entire state.
    Full,
    /// Transfer only changed keys since a given version.
    Incremental,
}

impl fmt::Display for SyncMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Incremental => write!(f, "incremental"),
        }
    }
}

// ── Sync State ──────────────────────────────────────────────────

/// A versioned, checksummed state snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncState {
    pub version: u64,
    pub checksum: u64,
    pub data: HashMap<String, Vec<u8>>,
}

impl SyncState {
    pub fn new() -> Self {
        Self { version: 0, checksum: 0, data: HashMap::new() }
    }

    /// Set a key and bump version + recompute checksum.
    pub fn set(&mut self, key: impl Into<String>, value: Vec<u8>) {
        self.data.insert(key.into(), value);
        self.version += 1;
        self.checksum = self.compute_checksum();
    }

    /// Remove a key and bump version + recompute checksum.
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        let removed = self.data.remove(key);
        if removed.is_some() {
            self.version += 1;
            self.checksum = self.compute_checksum();
        }
        removed
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&Vec<u8>> {
        self.data.get(key)
    }

    /// Total byte size of all values.
    pub fn byte_size(&self) -> usize {
        self.data.values().map(|v| v.len()).sum::<usize>()
            + self.data.keys().map(|k| k.len()).sum::<usize>()
    }

    /// Simple FNV-like checksum over sorted keys + values.
    fn compute_checksum(&self) -> u64 {
        let mut keys: Vec<&String> = self.data.keys().collect();
        keys.sort();
        let mut hash: u64 = 0xcbf29ce484222325;
        for k in &keys {
            for b in k.bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            if let Some(v) = self.data.get(*k) {
                for b in v {
                    hash ^= *b as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
        }
        hash
    }
}

impl Default for SyncState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SyncState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SyncState(v{}, {} keys, cksum={:#x})", self.version, self.data.len(), self.checksum)
    }
}

// ── Protocol Messages ───────────────────────────────────────────

/// A sync request from one peer to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncRequest {
    pub from_peer: u64,
    pub to_peer: u64,
    pub mode: SyncMode,
    pub since_version: u64,
    pub max_bytes: Option<usize>,
}

/// A sync response carrying state updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncResponse {
    pub from_peer: u64,
    pub version: u64,
    pub checksum: u64,
    pub updates: Vec<(String, Vec<u8>)>,
    pub deletions: Vec<String>,
    pub truncated: bool,
}

// ── Sync Statistics ─────────────────────────────────────────────

/// Tracks sync performance metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncStats {
    pub full_syncs: u64,
    pub incremental_syncs: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub conflicts_detected: u64,
    pub conflicts_resolved: u64,
}

impl SyncStats {
    pub fn new() -> Self {
        Self {
            full_syncs: 0,
            incremental_syncs: 0,
            bytes_sent: 0,
            bytes_received: 0,
            conflicts_detected: 0,
            conflicts_resolved: 0,
        }
    }
}

impl Default for SyncStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SyncStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SyncStats(full={}, incr={}, sent={}B, recv={}B, conflicts={}/{})",
            self.full_syncs,
            self.incremental_syncs,
            self.bytes_sent,
            self.bytes_received,
            self.conflicts_resolved,
            self.conflicts_detected,
        )
    }
}

// ── Peer Record ─────────────────────────────────────────────────

/// Tracked state of a remote peer.
#[derive(Debug, Clone)]
struct PeerRecord {
    last_known_version: u64,
    last_known_checksum: u64,
    syncing: bool,
}

// ── Sync Config ─────────────────────────────────────────────────

/// Configuration for the sync manager.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    pub sync_interval_ms: u64,
    pub max_batch_bytes: usize,
    pub auto_resolve_conflicts: bool,
}

impl SyncConfig {
    pub fn new() -> Self {
        Self {
            sync_interval_ms: 1000,
            max_batch_bytes: 65536,
            auto_resolve_conflicts: true,
        }
    }

    pub fn with_interval(mut self, ms: u64) -> Self {
        self.sync_interval_ms = ms;
        self
    }

    pub fn with_max_batch_bytes(mut self, bytes: usize) -> Self {
        self.max_batch_bytes = bytes;
        self
    }

    pub fn with_auto_resolve(mut self, auto: bool) -> Self {
        self.auto_resolve_conflicts = auto;
        self
    }
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ── State Sync Manager ─────────────────────────────────────────

/// Manages state synchronization between local and remote peers.
pub struct StateSyncManager {
    pub local_id: u64,
    pub local_state: SyncState,
    peers: HashMap<u64, PeerRecord>,
    /// Change log: (version, key, was_delete).
    change_log: Vec<(u64, String, bool)>,
    pub config: SyncConfig,
    pub stats: SyncStats,
}

impl StateSyncManager {
    pub fn new(local_id: u64) -> Self {
        Self {
            local_id,
            local_state: SyncState::new(),
            peers: HashMap::new(),
            change_log: Vec::new(),
            config: SyncConfig::new(),
            stats: SyncStats::new(),
        }
    }

    pub fn with_config(mut self, config: SyncConfig) -> Self {
        self.config = config;
        self
    }

    /// Register a remote peer.
    pub fn add_peer(&mut self, peer_id: u64) {
        self.peers.insert(peer_id, PeerRecord {
            last_known_version: 0,
            last_known_checksum: 0,
            syncing: false,
        });
    }

    /// Remove a remote peer.
    pub fn remove_peer(&mut self, peer_id: u64) -> bool {
        self.peers.remove(&peer_id).is_some()
    }

    /// Get registered peer IDs.
    pub fn peer_ids(&self) -> Vec<u64> {
        self.peers.keys().copied().collect()
    }

    /// Set a local state key (records change in log).
    pub fn set(&mut self, key: impl Into<String>, value: Vec<u8>) {
        let k = key.into();
        self.local_state.set(k.clone(), value);
        self.change_log.push((self.local_state.version, k, false));
    }

    /// Remove a local state key (records deletion in log).
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        let removed = self.local_state.remove(key);
        if removed.is_some() {
            self.change_log.push((self.local_state.version, key.to_string(), true));
        }
        removed
    }

    /// Build a sync request for a given peer.
    pub fn build_request(&self, peer_id: u64) -> Result<SyncRequest, SyncError> {
        let peer = self.peers.get(&peer_id).ok_or(SyncError::PeerNotFound(peer_id))?;
        if peer.syncing {
            return Err(SyncError::SyncInProgress(peer_id));
        }

        let mode = if peer.last_known_version == 0 {
            SyncMode::Full
        } else {
            SyncMode::Incremental
        };

        Ok(SyncRequest {
            from_peer: self.local_id,
            to_peer: peer_id,
            mode,
            since_version: peer.last_known_version,
            max_bytes: Some(self.config.max_batch_bytes),
        })
    }

    /// Process an incoming sync request and produce a response.
    pub fn handle_request(&self, req: &SyncRequest) -> SyncResponse {
        match req.mode {
            SyncMode::Full => {
                let max = req.max_bytes.unwrap_or(usize::MAX);
                let mut updates = Vec::new();
                let mut total = 0usize;
                let mut truncated = false;
                let mut keys: Vec<&String> = self.local_state.data.keys().collect();
                keys.sort();
                for k in keys {
                    if let Some(v) = self.local_state.data.get(k) {
                        let entry_size = k.len() + v.len();
                        if total + entry_size > max {
                            truncated = true;
                            break;
                        }
                        updates.push((k.clone(), v.clone()));
                        total += entry_size;
                    }
                }
                SyncResponse {
                    from_peer: self.local_id,
                    version: self.local_state.version,
                    checksum: self.local_state.checksum,
                    updates,
                    deletions: Vec::new(),
                    truncated,
                }
            }
            SyncMode::Incremental => {
                let max = req.max_bytes.unwrap_or(usize::MAX);
                let mut updates = Vec::new();
                let mut deletions = Vec::new();
                let mut total = 0usize;
                let mut truncated = false;

                for (ver, key, is_delete) in &self.change_log {
                    if *ver <= req.since_version {
                        continue;
                    }
                    if *is_delete {
                        let entry_size = key.len();
                        if total + entry_size > max {
                            truncated = true;
                            break;
                        }
                        deletions.push(key.clone());
                        total += entry_size;
                    } else if let Some(v) = self.local_state.data.get(key) {
                        let entry_size = key.len() + v.len();
                        if total + entry_size > max {
                            truncated = true;
                            break;
                        }
                        updates.push((key.clone(), v.clone()));
                        total += entry_size;
                    }
                }

                SyncResponse {
                    from_peer: self.local_id,
                    version: self.local_state.version,
                    checksum: self.local_state.checksum,
                    updates,
                    deletions,
                    truncated,
                }
            }
        }
    }

    /// Apply a sync response to our local knowledge of a peer.
    pub fn apply_response(&mut self, resp: &SyncResponse) -> Result<(), SyncError> {
        let peer = self.peers.get_mut(&resp.from_peer)
            .ok_or(SyncError::PeerNotFound(resp.from_peer))?;

        // Detect conflict: peer version diverged from what we last knew.
        if peer.last_known_version > 0
            && resp.version < peer.last_known_version
        {
            self.stats.conflicts_detected += 1;
            if self.config.auto_resolve_conflicts {
                self.stats.conflicts_resolved += 1;
                // Accept the remote state (last-writer-wins).
            } else {
                return Err(SyncError::VersionConflict {
                    local: peer.last_known_version,
                    remote: resp.version,
                });
            }
        }

        let mut bytes_received: u64 = 0;
        for (k, v) in &resp.updates {
            bytes_received += (k.len() + v.len()) as u64;
            self.local_state.data.insert(k.clone(), v.clone());
        }
        for k in &resp.deletions {
            bytes_received += k.len() as u64;
            self.local_state.data.remove(k);
        }
        self.local_state.version = self.local_state.version.max(resp.version);
        self.local_state.checksum = self.local_state.compute_checksum();

        peer.last_known_version = resp.version;
        peer.last_known_checksum = resp.checksum;
        peer.syncing = false;

        self.stats.bytes_received += bytes_received;
        if resp.updates.len() + resp.deletions.len() > 0 {
            if resp.deletions.is_empty() && !resp.truncated {
                self.stats.full_syncs += 1;
            } else {
                self.stats.incremental_syncs += 1;
            }
        }

        Ok(())
    }

    /// Detect if a peer is out of sync.
    pub fn is_peer_stale(&self, peer_id: u64) -> Result<bool, SyncError> {
        let peer = self.peers.get(&peer_id).ok_or(SyncError::PeerNotFound(peer_id))?;
        Ok(peer.last_known_version < self.local_state.version)
    }

    /// Get changes since a given version.
    pub fn changes_since(&self, version: u64) -> Vec<(u64, String, bool)> {
        self.change_log.iter()
            .filter(|(v, _, _)| *v > version)
            .cloned()
            .collect()
    }

    /// Total byte size of local state.
    pub fn local_byte_size(&self) -> usize {
        self.local_state.byte_size()
    }
}

impl fmt::Display for StateSyncManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StateSyncManager(id={}, v={}, peers={}, keys={})",
            self.local_id,
            self.local_state.version,
            self.peers.len(),
            self.local_state.data.len(),
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_state_default() {
        let s = SyncState::new();
        assert_eq!(s.version, 0);
        assert_eq!(s.data.len(), 0);
    }

    #[test]
    fn sync_state_set_bumps_version() {
        let mut s = SyncState::new();
        s.set("a", vec![1, 2, 3]);
        assert_eq!(s.version, 1);
        s.set("b", vec![4]);
        assert_eq!(s.version, 2);
    }

    #[test]
    fn sync_state_remove() {
        let mut s = SyncState::new();
        s.set("a", vec![1]);
        let v = s.remove("a");
        assert_eq!(v, Some(vec![1]));
        assert_eq!(s.version, 2);
        assert!(s.remove("missing").is_none());
    }

    #[test]
    fn sync_state_checksum_changes() {
        let mut s = SyncState::new();
        s.set("k", vec![10]);
        let c1 = s.checksum;
        s.set("k", vec![20]);
        assert_ne!(s.checksum, c1);
    }

    #[test]
    fn sync_state_byte_size() {
        let mut s = SyncState::new();
        s.set("ab", vec![1, 2, 3]);
        // "ab" = 2 bytes key + 3 bytes value = 5
        assert_eq!(s.byte_size(), 5);
    }

    #[test]
    fn sync_state_display() {
        let s = SyncState::new();
        let d = format!("{s}");
        assert!(d.contains("SyncState"));
    }

    #[test]
    fn manager_add_remove_peer() {
        let mut mgr = StateSyncManager::new(1);
        mgr.add_peer(2);
        assert_eq!(mgr.peer_ids().len(), 1);
        assert!(mgr.remove_peer(2));
        assert!(!mgr.remove_peer(999));
    }

    #[test]
    fn manager_set_and_get() {
        let mut mgr = StateSyncManager::new(1);
        mgr.set("hello", vec![42]);
        assert_eq!(mgr.local_state.get("hello"), Some(&vec![42]));
    }

    #[test]
    fn manager_remove_records_change() {
        let mut mgr = StateSyncManager::new(1);
        mgr.set("x", vec![1]);
        mgr.remove("x");
        let changes = mgr.changes_since(0);
        assert_eq!(changes.len(), 2);
        assert!(!changes[0].2); // set
        assert!(changes[1].2);  // delete
    }

    #[test]
    fn build_request_full_for_new_peer() {
        let mut mgr = StateSyncManager::new(1);
        mgr.add_peer(2);
        let req = mgr.build_request(2).unwrap();
        assert_eq!(req.mode, SyncMode::Full);
    }

    #[test]
    fn build_request_unknown_peer_errors() {
        let mgr = StateSyncManager::new(1);
        assert!(matches!(mgr.build_request(99), Err(SyncError::PeerNotFound(99))));
    }

    #[test]
    fn handle_full_request() {
        let mut mgr = StateSyncManager::new(1);
        mgr.set("a", vec![1]);
        mgr.set("b", vec![2]);
        let req = SyncRequest {
            from_peer: 2, to_peer: 1, mode: SyncMode::Full,
            since_version: 0, max_bytes: None,
        };
        let resp = mgr.handle_request(&req);
        assert_eq!(resp.updates.len(), 2);
        assert!(!resp.truncated);
    }

    #[test]
    fn handle_request_truncation() {
        let mut mgr = StateSyncManager::new(1);
        mgr.set("long_key", vec![0; 100]);
        let req = SyncRequest {
            from_peer: 2, to_peer: 1, mode: SyncMode::Full,
            since_version: 0, max_bytes: Some(10),
        };
        let resp = mgr.handle_request(&req);
        assert!(resp.truncated);
    }

    #[test]
    fn handle_incremental_request() {
        let mut mgr = StateSyncManager::new(1);
        mgr.set("a", vec![1]);
        mgr.set("b", vec![2]);
        let req = SyncRequest {
            from_peer: 2, to_peer: 1, mode: SyncMode::Incremental,
            since_version: 1, max_bytes: None,
        };
        let resp = mgr.handle_request(&req);
        assert_eq!(resp.updates.len(), 1); // only "b"
    }

    #[test]
    fn apply_response_updates_state() {
        let mut mgr = StateSyncManager::new(1);
        mgr.add_peer(2);
        let resp = SyncResponse {
            from_peer: 2, version: 5, checksum: 0,
            updates: vec![("x".into(), vec![10])],
            deletions: vec![],
            truncated: false,
        };
        mgr.apply_response(&resp).unwrap();
        assert_eq!(mgr.local_state.get("x"), Some(&vec![10]));
    }

    #[test]
    fn apply_response_unknown_peer_errors() {
        let mut mgr = StateSyncManager::new(1);
        let resp = SyncResponse {
            from_peer: 99, version: 1, checksum: 0,
            updates: vec![], deletions: vec![], truncated: false,
        };
        assert!(matches!(mgr.apply_response(&resp), Err(SyncError::PeerNotFound(99))));
    }

    #[test]
    fn is_peer_stale_detection() {
        let mut mgr = StateSyncManager::new(1);
        mgr.add_peer(2);
        assert!(!mgr.is_peer_stale(2).unwrap());
        mgr.set("k", vec![1]);
        assert!(mgr.is_peer_stale(2).unwrap());
    }

    #[test]
    fn sync_stats_tracking() {
        let mut mgr = StateSyncManager::new(1);
        mgr.add_peer(2);
        mgr.set("data", vec![1, 2, 3]);
        let resp = SyncResponse {
            from_peer: 2, version: 1, checksum: 0,
            updates: vec![("k".into(), vec![9, 8])],
            deletions: vec![], truncated: false,
        };
        mgr.apply_response(&resp).unwrap();
        assert!(mgr.stats.bytes_received > 0);
    }

    #[test]
    fn config_builder() {
        let cfg = SyncConfig::new()
            .with_interval(500)
            .with_max_batch_bytes(1024)
            .with_auto_resolve(false);
        assert_eq!(cfg.sync_interval_ms, 500);
        assert_eq!(cfg.max_batch_bytes, 1024);
        assert!(!cfg.auto_resolve_conflicts);
    }

    #[test]
    fn sync_mode_display() {
        assert_eq!(format!("{}", SyncMode::Full), "full");
        assert_eq!(format!("{}", SyncMode::Incremental), "incremental");
    }
}
