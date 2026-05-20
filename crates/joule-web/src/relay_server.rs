//! Relay/TURN server for NAT traversal — allocation lifecycle, bandwidth
//! tracking, permission model, data relay between peers, and relay statistics.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── RelayAllocation ─────────────────────────────────────────────────────────

/// A single relay allocation binding a client to a relayed transport address.
#[derive(Debug, Clone)]
pub struct RelayAllocation {
    pub id: u64,
    pub client_id: String,
    pub relay_address: String,
    pub created_at: u64,
    pub expires_at: u64,
    pub bytes_relayed: u64,
    pub bandwidth_limit: u64,
    pub allowed_peers: HashSet<String>,
}

impl RelayAllocation {
    pub fn new(id: u64, client_id: impl Into<String>, lifetime: u64, now: u64) -> Self {
        Self {
            id,
            client_id: client_id.into(),
            relay_address: format!("relay://alloc-{}", id),
            created_at: now,
            expires_at: now + lifetime,
            bytes_relayed: 0,
            bandwidth_limit: u64::MAX,
            allowed_peers: HashSet::new(),
        }
    }

    pub fn with_bandwidth_limit(mut self, limit: u64) -> Self {
        self.bandwidth_limit = limit;
        self
    }

    /// Whether the allocation has expired.
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Whether a peer is permitted to use this allocation.
    pub fn is_peer_allowed(&self, peer_id: &str) -> bool {
        self.allowed_peers.is_empty() || self.allowed_peers.contains(peer_id)
    }

    /// Add a permitted peer.
    pub fn allow_peer(&mut self, peer_id: impl Into<String>) {
        self.allowed_peers.insert(peer_id.into());
    }

    /// Refresh the allocation lifetime from `now`.
    pub fn refresh(&mut self, lifetime: u64, now: u64) {
        self.expires_at = now + lifetime;
    }

    /// Remaining ticks before expiry.
    pub fn remaining(&self, now: u64) -> u64 {
        self.expires_at.saturating_sub(now)
    }

    /// Record bytes relayed. Returns false if bandwidth limit exceeded.
    pub fn record_bytes(&mut self, bytes: u64) -> bool {
        let new_total = self.bytes_relayed.saturating_add(bytes);
        if new_total > self.bandwidth_limit {
            return false;
        }
        self.bytes_relayed = new_total;
        true
    }
}

impl fmt::Display for RelayAllocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Allocation(id={}, client={}, relayed={}B)",
            self.id, self.client_id, self.bytes_relayed,
        )
    }
}

// ── RelayStats ──────────────────────────────────────────────────────────────

/// Aggregate statistics for the relay server.
#[derive(Debug, Clone, Default)]
pub struct RelayStats {
    pub active_allocations: usize,
    pub total_allocations_created: u64,
    pub total_allocations_expired: u64,
    pub total_bytes_relayed: u64,
    pub total_relay_operations: u64,
    pub denied_operations: u64,
}

impl fmt::Display for RelayStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RelayStats(active={}, created={}, expired={}, bytes={}, ops={}, denied={})",
            self.active_allocations, self.total_allocations_created,
            self.total_allocations_expired, self.total_bytes_relayed,
            self.total_relay_operations, self.denied_operations,
        )
    }
}

// ── RelayData ───────────────────────────────────────────────────────────────

/// A chunk of data being relayed through the server.
#[derive(Debug, Clone)]
pub struct RelayData {
    pub from_client: String,
    pub to_peer: String,
    pub payload: Vec<u8>,
}

// ── RelayServer ─────────────────────────────────────────────────────────────

/// Relay server managing allocations and data forwarding.
pub struct RelayServer {
    allocations: HashMap<u64, RelayAllocation>,
    client_allocs: HashMap<String, u64>,
    next_id: u64,
    default_lifetime: u64,
    max_allocations: usize,
    stats: RelayStats,
    current_tick: u64,
}

impl RelayServer {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            client_allocs: HashMap::new(),
            next_id: 1,
            default_lifetime: 600,
            max_allocations: 1000,
            stats: RelayStats::default(),
            current_tick: 0,
        }
    }

    pub fn with_default_lifetime(mut self, lifetime: u64) -> Self {
        self.default_lifetime = lifetime;
        self
    }

    pub fn with_max_allocations(mut self, max: usize) -> Self {
        self.max_allocations = max;
        self
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Create a new allocation for a client. Returns the allocation id.
    pub fn create_allocation(&mut self, client_id: impl Into<String>) -> Option<u64> {
        let cid = client_id.into();
        if self.client_allocs.contains_key(&cid) {
            return None; // client already has allocation
        }
        if self.allocations.len() >= self.max_allocations {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let alloc = RelayAllocation::new(id, cid.clone(), self.default_lifetime, self.current_tick);
        self.allocations.insert(id, alloc);
        self.client_allocs.insert(cid, id);
        self.stats.total_allocations_created += 1;
        self.stats.active_allocations = self.allocations.len();
        Some(id)
    }

    /// Create allocation with a specific bandwidth limit.
    pub fn create_allocation_with_limit(
        &mut self,
        client_id: impl Into<String>,
        bandwidth_limit: u64,
    ) -> Option<u64> {
        let cid = client_id.into();
        if self.client_allocs.contains_key(&cid) {
            return None;
        }
        if self.allocations.len() >= self.max_allocations {
            return None;
        }
        let id = self.next_id;
        self.next_id += 1;
        let alloc = RelayAllocation::new(id, cid.clone(), self.default_lifetime, self.current_tick)
            .with_bandwidth_limit(bandwidth_limit);
        self.allocations.insert(id, alloc);
        self.client_allocs.insert(cid, id);
        self.stats.total_allocations_created += 1;
        self.stats.active_allocations = self.allocations.len();
        Some(id)
    }

    /// Refresh an existing allocation.
    pub fn refresh_allocation(&mut self, alloc_id: u64) -> bool {
        if let Some(alloc) = self.allocations.get_mut(&alloc_id) {
            alloc.refresh(self.default_lifetime, self.current_tick);
            true
        } else {
            false
        }
    }

    /// Delete an allocation.
    pub fn delete_allocation(&mut self, alloc_id: u64) -> bool {
        if let Some(alloc) = self.allocations.remove(&alloc_id) {
            self.client_allocs.remove(&alloc.client_id);
            self.stats.active_allocations = self.allocations.len();
            true
        } else {
            false
        }
    }

    /// Add a permission for a peer on a client's allocation.
    pub fn add_permission(&mut self, alloc_id: u64, peer_id: impl Into<String>) -> bool {
        if let Some(alloc) = self.allocations.get_mut(&alloc_id) {
            alloc.allow_peer(peer_id);
            true
        } else {
            false
        }
    }

    /// Relay data from a client to a peer through the server.
    pub fn relay_data(&mut self, data: &RelayData) -> Result<(), String> {
        let alloc_id = self
            .client_allocs
            .get(&data.from_client)
            .copied()
            .ok_or_else(|| "no allocation for client".to_string())?;
        let alloc = self
            .allocations
            .get_mut(&alloc_id)
            .ok_or_else(|| "allocation not found".to_string())?;

        if alloc.is_expired(self.current_tick) {
            self.stats.denied_operations += 1;
            return Err("allocation expired".into());
        }
        if !alloc.is_peer_allowed(&data.to_peer) {
            self.stats.denied_operations += 1;
            return Err("peer not permitted".into());
        }
        if !alloc.record_bytes(data.payload.len() as u64) {
            self.stats.denied_operations += 1;
            return Err("bandwidth limit exceeded".into());
        }

        self.stats.total_bytes_relayed += data.payload.len() as u64;
        self.stats.total_relay_operations += 1;
        Ok(())
    }

    /// Expire all timed-out allocations.
    pub fn expire_allocations(&mut self) -> usize {
        let expired: Vec<u64> = self
            .allocations
            .iter()
            .filter(|(_, a)| a.is_expired(self.current_tick))
            .map(|(id, _)| *id)
            .collect();
        let count = expired.len();
        for id in expired {
            if let Some(alloc) = self.allocations.remove(&id) {
                self.client_allocs.remove(&alloc.client_id);
            }
            self.stats.total_allocations_expired += 1;
        }
        self.stats.active_allocations = self.allocations.len();
        count
    }

    /// Get allocation info.
    pub fn get_allocation(&self, alloc_id: u64) -> Option<&RelayAllocation> {
        self.allocations.get(&alloc_id)
    }

    /// Number of active allocations.
    pub fn active_count(&self) -> usize {
        self.allocations.len()
    }

    /// Current statistics.
    pub fn stats(&self) -> &RelayStats {
        &self.stats
    }
}

impl Default for RelayServer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_allocation() {
        let mut srv = RelayServer::new();
        let id = srv.create_allocation("client-1");
        assert!(id.is_some());
        assert_eq!(srv.active_count(), 1);
    }

    #[test]
    fn test_duplicate_client_allocation() {
        let mut srv = RelayServer::new();
        srv.create_allocation("client-1");
        assert!(srv.create_allocation("client-1").is_none());
    }

    #[test]
    fn test_max_allocations() {
        let mut srv = RelayServer::new().with_max_allocations(2);
        srv.create_allocation("a");
        srv.create_allocation("b");
        assert!(srv.create_allocation("c").is_none());
    }

    #[test]
    fn test_delete_allocation() {
        let mut srv = RelayServer::new();
        let id = srv.create_allocation("client-1").unwrap();
        assert!(srv.delete_allocation(id));
        assert_eq!(srv.active_count(), 0);
    }

    #[test]
    fn test_refresh_allocation() {
        let mut srv = RelayServer::new().with_default_lifetime(100);
        srv.tick(10);
        let id = srv.create_allocation("c").unwrap();
        srv.tick(50);
        srv.refresh_allocation(id);
        let alloc = srv.get_allocation(id).unwrap();
        assert_eq!(alloc.expires_at, 150);
    }

    #[test]
    fn test_relay_data_success() {
        let mut srv = RelayServer::new();
        srv.create_allocation("alice");
        let data = RelayData {
            from_client: "alice".into(),
            to_peer: "bob".into(),
            payload: vec![1, 2, 3],
        };
        assert!(srv.relay_data(&data).is_ok());
        assert_eq!(srv.stats().total_bytes_relayed, 3);
    }

    #[test]
    fn test_relay_data_no_allocation() {
        let mut srv = RelayServer::new();
        let data = RelayData {
            from_client: "nobody".into(),
            to_peer: "bob".into(),
            payload: vec![1],
        };
        assert!(srv.relay_data(&data).is_err());
    }

    #[test]
    fn test_relay_data_expired() {
        let mut srv = RelayServer::new().with_default_lifetime(10);
        srv.tick(0);
        srv.create_allocation("alice");
        srv.tick(20);
        let data = RelayData {
            from_client: "alice".into(),
            to_peer: "bob".into(),
            payload: vec![1],
        };
        assert!(srv.relay_data(&data).is_err());
    }

    #[test]
    fn test_relay_permission_denied() {
        let mut srv = RelayServer::new();
        let id = srv.create_allocation("alice").unwrap();
        srv.add_permission(id, "charlie");
        let data = RelayData {
            from_client: "alice".into(),
            to_peer: "bob".into(),
            payload: vec![1],
        };
        assert!(srv.relay_data(&data).is_err());
    }

    #[test]
    fn test_relay_permission_allowed() {
        let mut srv = RelayServer::new();
        let id = srv.create_allocation("alice").unwrap();
        srv.add_permission(id, "bob");
        let data = RelayData {
            from_client: "alice".into(),
            to_peer: "bob".into(),
            payload: vec![1, 2],
        };
        assert!(srv.relay_data(&data).is_ok());
    }

    #[test]
    fn test_bandwidth_limit() {
        let mut srv = RelayServer::new();
        srv.create_allocation_with_limit("alice", 5);
        let data = RelayData {
            from_client: "alice".into(),
            to_peer: "bob".into(),
            payload: vec![0; 6],
        };
        assert!(srv.relay_data(&data).is_err());
    }

    #[test]
    fn test_expire_allocations() {
        let mut srv = RelayServer::new().with_default_lifetime(10);
        srv.tick(0);
        srv.create_allocation("a");
        srv.create_allocation("b");
        srv.tick(15);
        let expired = srv.expire_allocations();
        assert_eq!(expired, 2);
        assert_eq!(srv.active_count(), 0);
    }

    #[test]
    fn test_allocation_remaining() {
        let alloc = RelayAllocation::new(1, "c", 100, 10);
        assert_eq!(alloc.remaining(50), 60);
        assert_eq!(alloc.remaining(110), 0);
    }

    #[test]
    fn test_allocation_display() {
        let alloc = RelayAllocation::new(42, "alice", 100, 0);
        let s = format!("{}", alloc);
        assert!(s.contains("42"));
        assert!(s.contains("alice"));
    }

    #[test]
    fn test_stats_after_relay() {
        let mut srv = RelayServer::new();
        srv.create_allocation("x");
        let data = RelayData { from_client: "x".into(), to_peer: "y".into(), payload: vec![0; 10] };
        srv.relay_data(&data).unwrap();
        let s = srv.stats();
        assert_eq!(s.total_relay_operations, 1);
        assert_eq!(s.total_bytes_relayed, 10);
    }

    #[test]
    fn test_stats_denied_count() {
        let mut srv = RelayServer::new();
        let data = RelayData { from_client: "x".into(), to_peer: "y".into(), payload: vec![1] };
        let _ = srv.relay_data(&data);
        assert_eq!(srv.stats().denied_operations, 0); // no alloc = no denial counter (error before check)
    }

    #[test]
    fn test_relay_stats_display() {
        let s = RelayStats::default();
        let display = format!("{}", s);
        assert!(display.contains("RelayStats"));
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut srv = RelayServer::new();
        assert!(!srv.delete_allocation(999));
    }

    #[test]
    fn test_after_delete_client_can_create_again() {
        let mut srv = RelayServer::new();
        let id = srv.create_allocation("c").unwrap();
        srv.delete_allocation(id);
        assert!(srv.create_allocation("c").is_some());
    }
}
