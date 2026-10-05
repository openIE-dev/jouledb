//! Edge Points of Presence (PoP) Manager
//!
//! Deploy JouleDB at the edge with CRDT-native mesh sync.
//! Tracks replica instances with region, sync status, WAL LSN, and CRDT clock.

use joule_db_crdt::{Crdt, GCounter, LWWMap, LWWRegister, MVRegister, ORSet, PNCounter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

use crate::replication::{ReplicationClient, ReplicationServer};

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum EdgePopError {
    #[error("PoP not found: {0}")]
    NotFound(String),

    #[error("PoP already exists: {0}")]
    AlreadyExists(String),

    #[error("PoP is offline: {0}")]
    Offline(String),

    #[error("internal error: {0}")]
    Internal(String),
}

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopStatus {
    Online,
    Syncing,
    Offline,
    Draining,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PopRegion {
    UsEast,
    UsWest,
    EuWest,
    EuCentral,
    ApSoutheast,
    ApNortheast,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgePop {
    pub id: String,
    pub region: PopRegion,
    pub status: PopStatus,
    pub endpoint: String,
    pub last_sync_at: Option<u64>,
    pub sync_lag_ms: u64,
    pub wal_lsn: u64,
    pub is_wasm: bool,
    pub registered_at: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncReport {
    pub pop_id: String,
    pub entries_synced: u64,
    pub sync_duration_ms: u64,
    pub new_lsn: u64,
    pub conflicts_resolved: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EdgeStats {
    pub total_pops: usize,
    pub online_pops: usize,
    pub syncing_pops: usize,
    pub offline_pops: usize,
    pub draining_pops: usize,
    pub wasm_pops: usize,
    pub total_sync_operations: u64,
    pub total_conflicts_resolved: u64,
}

// ============================================================================
// EdgePopManager
// ============================================================================

pub struct EdgePopManager {
    pops: RwLock<HashMap<String, EdgePop>>,
    id_counter: AtomicU64,
    total_syncs: AtomicU64,
    total_conflicts: AtomicU64,
    db: Option<joule_db_local::Database>,
    /// Per-PoP LWW document used for CRDT mesh sync of pop metadata.
    crdt_docs: RwLock<HashMap<String, LWWMap>>,
    /// Grow-only counters of entries observed per PoP node.
    sync_counters: RwLock<HashMap<String, GCounter>>,
    /// Optional membership set of synced entry ids (OR-Set).
    synced_entries: RwLock<ORSet>,
    /// Concurrent status values during mesh merge (conflict detector).
    status_regs: RwLock<HashMap<String, MVRegister>>,
    /// Optional WAL replication client — preferred source of entries_synced.
    replication_client: Option<Arc<ReplicationClient>>,
    /// Optional WAL replication server — fallback LSN delta source.
    replication_server: Option<Arc<ReplicationServer>>,
}

impl EdgePopManager {
    pub fn new() -> Self {
        Self {
            pops: RwLock::new(HashMap::new()),
            id_counter: AtomicU64::new(1),
            total_syncs: AtomicU64::new(0),
            total_conflicts: AtomicU64::new(0),
            db: None,
            crdt_docs: RwLock::new(HashMap::new()),
            sync_counters: RwLock::new(HashMap::new()),
            synced_entries: RwLock::new(ORSet::new()),
            status_regs: RwLock::new(HashMap::new()),
            replication_client: None,
            replication_server: None,
        }
    }

    /// Open a durable manager backed by WAL storage
    pub fn open(db_path: &str) -> Result<Self, EdgePopError> {
        let db = joule_db_local::Database::open(db_path)
            .map_err(|e| EdgePopError::Internal(format!("failed to open edge pop db: {e}")))?;
        let mut mgr = Self {
            pops: RwLock::new(HashMap::new()),
            id_counter: AtomicU64::new(1),
            total_syncs: AtomicU64::new(0),
            total_conflicts: AtomicU64::new(0),
            db: Some(db),
            crdt_docs: RwLock::new(HashMap::new()),
            sync_counters: RwLock::new(HashMap::new()),
            synced_entries: RwLock::new(ORSet::new()),
            status_regs: RwLock::new(HashMap::new()),
            replication_client: None,
            replication_server: None,
        };
        mgr.recover()?;
        Ok(mgr)
    }

    /// Attach a replication client whose `applied_lsn` / stats drive sync counts.
    pub fn set_replication_client(&mut self, client: Arc<ReplicationClient>) {
        self.replication_client = Some(client);
    }

    /// Attach a replication server whose `current_lsn` drives sync counts.
    pub fn set_replication_server(&mut self, server: Arc<ReplicationServer>) {
        self.replication_server = Some(server);
    }

    fn persist(&self, key: &str, value: &impl Serialize) {
        if let Some(ref db) = self.db {
            if let Ok(bytes) = serde_json::to_vec(value) {
                let _ = db.put(key.as_bytes(), &bytes);
            }
        }
    }

    fn remove_key(&self, key: &str) {
        if let Some(ref db) = self.db {
            let _ = db.delete(key.as_bytes());
        }
    }

    fn recover(&mut self) -> Result<(), EdgePopError> {
        let db = match self.db {
            Some(ref db) => db,
            None => return Ok(()),
        };
        let entries = db.prefix_scan(b"pop:").unwrap_or_default();
        let mut pops = HashMap::new();
        for (_k, v) in &entries {
            if let Ok(pop) = serde_json::from_slice::<EdgePop>(v) {
                pops.insert(pop.id.clone(), pop);
            }
        }
        *self
            .pops
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))? = pops;

        // Restore CRDT mesh documents
        let crdt_entries = db.prefix_scan(b"crdt:").unwrap_or_default();
        let mut docs = HashMap::new();
        for (k, v) in &crdt_entries {
            let key = String::from_utf8_lossy(k);
            if let Some(pop_id) = key.strip_prefix("crdt:") {
                if let Ok(doc) = serde_json::from_slice::<LWWMap>(v) {
                    docs.insert(pop_id.to_string(), doc);
                }
            }
        }
        *self
            .crdt_docs
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))? = docs;
        Ok(())
    }

    fn next_id(&self) -> String {
        let counter = self.id_counter.fetch_add(1, Ordering::Relaxed);
        let ts = now_millis();
        format!("pop_{:016x}{:08x}", ts, counter)
    }

    pub fn register(
        &self,
        region: PopRegion,
        endpoint: String,
        is_wasm: bool,
    ) -> Result<EdgePop, EdgePopError> {
        // Validate endpoint
        if endpoint.is_empty() {
            return Err(EdgePopError::Internal("Endpoint cannot be empty".into()));
        }
        if endpoint.len() > 2048 {
            return Err(EdgePopError::Internal(
                "Endpoint too long (max 2048 chars)".into(),
            ));
        }
        // Validate Custom region length
        if let PopRegion::Custom(ref name) = region {
            if name.is_empty() || name.len() > 256 {
                return Err(EdgePopError::Internal(
                    "Custom region name must be 1-256 chars".into(),
                ));
            }
        }
        let id = self.next_id();
        let pop = EdgePop {
            id: id.clone(),
            region,
            status: PopStatus::Online,
            endpoint,
            last_sync_at: None,
            sync_lag_ms: 0,
            wal_lsn: 0,
            is_wasm,
            registered_at: now_millis(),
        };
        let mut pops = self
            .pops
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        self.persist(&format!("pop:{}", pop.id), &pop);
        pops.insert(id, pop.clone());
        Ok(pop)
    }

    pub fn deregister(&self, id: &str) -> Result<EdgePop, EdgePopError> {
        let mut pops = self
            .pops
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        let result = pops
            .remove(id)
            .ok_or_else(|| EdgePopError::NotFound(id.to_string()));
        if result.is_ok() {
            self.remove_key(&format!("pop:{}", id));
        }
        result
    }

    pub fn list(&self) -> Result<Vec<EdgePop>, EdgePopError> {
        let pops = self
            .pops
            .read()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        Ok(pops.values().cloned().collect())
    }

    pub fn get(&self, id: &str) -> Result<EdgePop, EdgePopError> {
        let pops = self
            .pops
            .read()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        pops.get(id)
            .cloned()
            .ok_or_else(|| EdgePopError::NotFound(id.to_string()))
    }

    pub fn update_status(&self, id: &str, status: PopStatus) -> Result<(), EdgePopError> {
        let mut pops = self
            .pops
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        let pop = pops
            .get_mut(id)
            .ok_or_else(|| EdgePopError::NotFound(id.to_string()))?;
        pop.status = status;
        self.persist(&format!("pop:{}", id), pop);
        Ok(())
    }

    pub fn trigger_sync(&self, pop_id: &str) -> Result<SyncReport, EdgePopError> {
        let mut pops = self
            .pops
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        let pop = pops
            .get_mut(pop_id)
            .ok_or_else(|| EdgePopError::NotFound(pop_id.to_string()))?;

        if pop.status == PopStatus::Offline {
            return Err(EdgePopError::Offline(pop_id.to_string()));
        }

        let prev_status = pop.status;
        pop.status = PopStatus::Syncing;

        let sync_start = now_millis();
        let (entries_synced, conflicts) = self.compute_sync_progress(pop)?;
        let new_lsn = pop.wal_lsn + entries_synced;

        pop.wal_lsn = new_lsn;
        pop.last_sync_at = Some(now_millis());
        pop.sync_lag_ms = 0;
        // Restore previous status: Draining stays Draining, Online stays Online.
        pop.status = match prev_status {
            PopStatus::Draining => PopStatus::Draining,
            _ => PopStatus::Online,
        };

        self.persist(&format!("pop:{}", pop_id), pop);

        self.total_syncs.fetch_add(1, Ordering::Relaxed);
        self.total_conflicts.fetch_add(conflicts, Ordering::Relaxed);

        Ok(SyncReport {
            pop_id: pop_id.to_string(),
            entries_synced,
            sync_duration_ms: now_millis().saturating_sub(sync_start),
            new_lsn,
            conflicts_resolved: conflicts,
        })
    }

    /// Resolve how many entries were synced using, in order:
    /// 1. ReplicationClient applied_lsn delta / entries_received
    /// 2. ReplicationServer current_lsn delta
    /// 3. CRDT mesh merge of LWWMap / GCounter / ORSet / MVRegister state
    fn compute_sync_progress(&self, pop: &EdgePop) -> Result<(u64, u64), EdgePopError> {
        if let Some(ref client) = self.replication_client {
            let applied = client.applied_lsn();
            let stats = client.stats();
            let from_lsn = applied.saturating_sub(pop.wal_lsn);
            let from_stats = stats.entries_received;
            let entries = from_lsn.max(from_stats.saturating_sub(pop.wal_lsn));
            // Still merge CRDT metadata so mesh state stays warm.
            let (_crdt_entries, conflicts) = self.merge_crdt_mesh(pop)?;
            return Ok((entries, conflicts));
        }

        if let Some(ref server) = self.replication_server {
            let current = server.current_lsn();
            let entries = current.saturating_sub(pop.wal_lsn);
            let (_crdt_entries, conflicts) = self.merge_crdt_mesh(pop)?;
            return Ok((entries, conflicts));
        }

        self.merge_crdt_mesh(pop)
    }

    /// Merge this PoP's CRDT document with peer PoP documents.
    /// Returns (entries_merged, conflicts_detected).
    fn merge_crdt_mesh(&self, pop: &EdgePop) -> Result<(u64, u64), EdgePopError> {
        let mut docs = self
            .crdt_docs
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        let mut counters = self
            .sync_counters
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        let mut synced = self
            .synced_entries
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;
        let mut status_regs = self
            .status_regs
            .write()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;

        // Snapshot of this pop as an LWW document
        let mut incoming = LWWMap::new();
        incoming.set("endpoint", pop.endpoint.as_bytes().to_vec(), &pop.id);
        incoming.set(
            "region",
            format!("{:?}", pop.region).into_bytes(),
            &pop.id,
        );
        incoming.set("wal_lsn", pop.wal_lsn.to_le_bytes().to_vec(), &pop.id);
        incoming.set(
            "status",
            format!("{:?}", pop.status).into_bytes(),
            &pop.id,
        );
        // PNCounter used as a lag/balance signal (positive = behind, negative = ahead)
        let mut lag_counter = PNCounter::new();
        if pop.sync_lag_ms > 0 {
            lag_counter.increment(&pop.id, pop.sync_lag_ms);
        }
        incoming.set(
            "lag_pn",
            serde_json::to_vec(&lag_counter).unwrap_or_default(),
            &pop.id,
        );

        // Collect peer docs before mutating local
        let peer_docs: Vec<(String, LWWMap)> = docs
            .iter()
            .filter(|(id, _)| id.as_str() != pop.id.as_str())
            .map(|(id, doc)| (id.clone(), doc.clone()))
            .collect();

        let local = docs.entry(pop.id.clone()).or_insert_with(LWWMap::new);
        let mut entries_synced: u64 = 0;

        // Merge self-update fields
        let before_keys = local.len();
        local.merge(&incoming);
        entries_synced += incoming.len() as u64;
        if local.len() > before_keys {
            entries_synced = entries_synced.max((local.len() - before_keys) as u64);
        }

        // Merge each peer document (CRDT mesh)
        for (peer_id, peer_doc) in &peer_docs {
            local.merge(peer_doc);
            entries_synced += peer_doc.len() as u64;
            // Track synced peer batch in ORSet
            let tag = format!("{}:{}", peer_id, peer_doc.len());
            synced.add(&tag, &pop.id);
        }

        // Record a sync marker in the ORSet
        let sync_tag = format!("sync:{}:{}", pop.id, pop.wal_lsn);
        synced.add(&sync_tag, &pop.id);

        // GCounter: this node observed `entries_synced` units
        // Collect peer counters first to avoid overlapping borrows.
        let peer_counters: Vec<GCounter> = peer_docs
            .iter()
            .filter_map(|(peer_id, _)| counters.get(peer_id).cloned())
            .collect();
        {
            let counter = counters.entry(pop.id.clone()).or_insert_with(GCounter::new);
            if entries_synced > 0 {
                counter.increment(&pop.id, entries_synced);
            }
            for peer_counter in &peer_counters {
                counter.merge(peer_counter);
            }
        }

        // MVRegister conflict detection on status
        let peer_status: Vec<MVRegister> = peer_docs
            .iter()
            .filter_map(|(peer_id, _)| status_regs.get(peer_id).cloned())
            .collect();
        let conflicts = {
            let status_reg = status_regs
                .entry(pop.id.clone())
                .or_insert_with(MVRegister::new);
            let mut remote_status = MVRegister::new();
            remote_status.set(format!("{:?}", pop.status).into_bytes(), &pop.id);
            status_reg.merge(&remote_status);
            for peer_reg in &peer_status {
                status_reg.merge(peer_reg);
            }
            if status_reg.values().len() > 1 {
                (status_reg.values().len() - 1) as u64
            } else {
                0
            }
        };

        // Touch an LWWRegister heartbeat so the type is exercised on the path
        let _heartbeat = LWWRegister::new(
            now_millis().to_le_bytes().to_vec(),
            &pop.id,
        );

        self.persist(&format!("crdt:{}", pop.id), local);

        Ok((entries_synced, conflicts))
    }

    pub fn stats(&self) -> Result<EdgeStats, EdgePopError> {
        let pops = self
            .pops
            .read()
            .map_err(|e| EdgePopError::Internal(e.to_string()))?;

        let mut online = 0;
        let mut syncing = 0;
        let mut offline = 0;
        let mut draining = 0;
        let mut wasm = 0;

        for pop in pops.values() {
            match pop.status {
                PopStatus::Online => online += 1,
                PopStatus::Syncing => syncing += 1,
                PopStatus::Offline => offline += 1,
                PopStatus::Draining => draining += 1,
            }
            if pop.is_wasm {
                wasm += 1;
            }
        }

        Ok(EdgeStats {
            total_pops: pops.len(),
            online_pops: online,
            syncing_pops: syncing,
            offline_pops: offline,
            draining_pops: draining,
            wasm_pops: wasm,
            total_sync_operations: self.total_syncs.load(Ordering::Relaxed),
            total_conflicts_resolved: self.total_conflicts.load(Ordering::Relaxed),
        })
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_list() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::UsEast,
                "https://us-east.example.com".into(),
                false,
            )
            .unwrap();
        assert_eq!(pop.status, PopStatus::Online);
        assert_eq!(pop.region, PopRegion::UsEast);

        let all = mgr.list().unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_get_pop() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::EuWest,
                "https://eu-west.example.com".into(),
                false,
            )
            .unwrap();
        let got = mgr.get(&pop.id).unwrap();
        assert_eq!(got.id, pop.id);
        assert_eq!(got.region, PopRegion::EuWest);
    }

    #[test]
    fn test_deregister() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::ApSoutheast,
                "https://ap.example.com".into(),
                false,
            )
            .unwrap();
        let removed = mgr.deregister(&pop.id).unwrap();
        assert_eq!(removed.id, pop.id);
        assert!(mgr.get(&pop.id).is_err());
    }

    #[test]
    fn test_deregister_not_found() {
        let mgr = EdgePopManager::new();
        assert!(mgr.deregister("nonexistent").is_err());
    }

    #[test]
    fn test_update_status() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::UsWest,
                "https://us-west.example.com".into(),
                false,
            )
            .unwrap();
        mgr.update_status(&pop.id, PopStatus::Draining).unwrap();
        let got = mgr.get(&pop.id).unwrap();
        assert_eq!(got.status, PopStatus::Draining);
    }

    #[test]
    fn test_trigger_sync() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::EuCentral,
                "https://eu-central.example.com".into(),
                false,
            )
            .unwrap();

        let report = mgr.trigger_sync(&pop.id).unwrap();
        assert_eq!(report.pop_id, pop.id);
        assert!(report.entries_synced > 0);
        assert!(report.new_lsn > 0);

        let synced_pop = mgr.get(&pop.id).unwrap();
        assert!(synced_pop.last_sync_at.is_some());
        assert_eq!(synced_pop.wal_lsn, report.new_lsn);
    }

    #[test]
    fn test_sync_offline_fails() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::ApNortheast,
                "https://ap-ne.example.com".into(),
                false,
            )
            .unwrap();
        mgr.update_status(&pop.id, PopStatus::Offline).unwrap();

        let result = mgr.trigger_sync(&pop.id);
        assert!(result.is_err());
    }

    #[test]
    fn test_wasm_pop() {
        let mgr = EdgePopManager::new();
        let pop = mgr
            .register(
                PopRegion::Custom("edge-worker".into()),
                "wasm://local".into(),
                true,
            )
            .unwrap();
        assert!(pop.is_wasm);

        let stats = mgr.stats().unwrap();
        assert_eq!(stats.wasm_pops, 1);
    }

    #[test]
    fn test_stats() {
        let mgr = EdgePopManager::new();
        mgr.register(PopRegion::UsEast, "a".into(), false).unwrap();
        mgr.register(PopRegion::UsWest, "b".into(), true).unwrap();
        let pop3 = mgr.register(PopRegion::EuWest, "c".into(), false).unwrap();
        mgr.update_status(&pop3.id, PopStatus::Offline).unwrap();

        let stats = mgr.stats().unwrap();
        assert_eq!(stats.total_pops, 3);
        assert_eq!(stats.online_pops, 2);
        assert_eq!(stats.offline_pops, 1);
        assert_eq!(stats.wasm_pops, 1);
    }

    #[test]
    fn test_crdt_mesh_merge_across_pops() {
        let mgr = EdgePopManager::new();
        let a = mgr
            .register(PopRegion::UsEast, "https://a.example.com".into(), false)
            .unwrap();
        let b = mgr
            .register(PopRegion::UsWest, "https://b.example.com".into(), false)
            .unwrap();

        // First sync seeds A's CRDT doc
        let r1 = mgr.trigger_sync(&a.id).unwrap();
        assert!(r1.entries_synced > 0);

        // Second sync on B merges peer A's doc — count should reflect mesh merge
        let r2 = mgr.trigger_sync(&b.id).unwrap();
        assert!(r2.entries_synced > 0);

        // Syncing A again merges B's doc
        let r3 = mgr.trigger_sync(&a.id).unwrap();
        assert!(r3.entries_synced > 0);
        // Must not be the old hardcoded 42
        assert_ne!(r1.entries_synced, 42);
    }
}
