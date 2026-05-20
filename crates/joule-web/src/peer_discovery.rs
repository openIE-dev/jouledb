//! Peer discovery protocol — announce self, find peers by capability, expiry of
//! stale peers, seed node bootstrap, peer exchange, and discovery statistics.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Capability ──────────────────────────────────────────────────────────────

/// A capability a peer advertises (relay, storage, compute, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Capability {
    pub name: String,
    pub version: u32,
}

impl Capability {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self { name: name.into(), version }
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@v{}", self.name, self.version)
    }
}

// ── PeerInfo ────────────────────────────────────────────────────────────────

/// Information about a discovered peer.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: String,
    pub address: String,
    pub capabilities: HashSet<Capability>,
    pub last_seen: u64,
    pub announce_count: u64,
}

impl PeerInfo {
    pub fn new(id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            capabilities: HashSet::new(),
            last_seen: 0,
            announce_count: 0,
        }
    }

    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    pub fn with_last_seen(mut self, tick: u64) -> Self {
        self.last_seen = tick;
        self
    }

    /// Whether the peer has a specific capability by name (any version).
    pub fn has_capability(&self, name: &str) -> bool {
        self.capabilities.iter().any(|c| c.name == name)
    }

    /// Whether the peer has a capability with at least the given version.
    pub fn has_capability_version(&self, name: &str, min_version: u32) -> bool {
        self.capabilities.iter().any(|c| c.name == name && c.version >= min_version)
    }
}

impl fmt::Display for PeerInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Peer({} @ {}, caps={})", self.id, self.address, self.capabilities.len())
    }
}

// ── DiscoveryStats ──────────────────────────────────────────────────────────

/// Statistics for the discovery service.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryStats {
    pub total_announces: u64,
    pub total_queries: u64,
    pub total_expired: u64,
    pub total_exchanges: u64,
    pub peers_known: usize,
}

impl fmt::Display for DiscoveryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiscoveryStats(announces={}, queries={}, expired={}, exchanges={}, known={})",
            self.total_announces, self.total_queries, self.total_expired,
            self.total_exchanges, self.peers_known,
        )
    }
}

// ── DiscoveryConfig ─────────────────────────────────────────────────────────

/// Configuration for the discovery service.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Ticks before a peer is considered stale and removed.
    pub peer_ttl: u64,
    /// Maximum number of peers to store.
    pub max_peers: usize,
    /// Maximum peers returned in a single exchange.
    pub exchange_limit: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            peer_ttl: 300,
            max_peers: 1000,
            exchange_limit: 20,
        }
    }
}

// ── DiscoveryService ────────────────────────────────────────────────────────

/// Manages discovery of peers in the network.
pub struct DiscoveryService {
    self_id: String,
    self_address: String,
    self_capabilities: HashSet<Capability>,
    peers: HashMap<String, PeerInfo>,
    seed_nodes: Vec<(String, String)>,
    config: DiscoveryConfig,
    stats: DiscoveryStats,
    current_tick: u64,
}

impl DiscoveryService {
    pub fn new(id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            self_id: id.into(),
            self_address: address.into(),
            self_capabilities: HashSet::new(),
            peers: HashMap::new(),
            seed_nodes: Vec::new(),
            config: DiscoveryConfig::default(),
            stats: DiscoveryStats::default(),
            current_tick: 0,
        }
    }

    pub fn with_config(mut self, config: DiscoveryConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_capability(&mut self, cap: Capability) {
        self.self_capabilities.insert(cap);
    }

    pub fn add_seed_node(&mut self, id: impl Into<String>, address: impl Into<String>) {
        self.seed_nodes.push((id.into(), address.into()));
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Announce self — returns a PeerInfo suitable for broadcasting.
    pub fn announce_self(&mut self) -> PeerInfo {
        self.stats.total_announces += 1;
        PeerInfo {
            id: self.self_id.clone(),
            address: self.self_address.clone(),
            capabilities: self.self_capabilities.clone(),
            last_seen: self.current_tick,
            announce_count: self.stats.total_announces,
        }
    }

    /// Bootstrap from seed nodes. Each seed is added as a known peer.
    pub fn bootstrap(&mut self) -> usize {
        let seeds: Vec<_> = self.seed_nodes.clone();
        let mut added = 0;
        for (id, addr) in &seeds {
            if id != &self.self_id && !self.peers.contains_key(id.as_str()) {
                if self.peers.len() < self.config.max_peers {
                    let info = PeerInfo::new(id.clone(), addr.clone())
                        .with_last_seen(self.current_tick);
                    self.peers.insert(id.clone(), info);
                    added += 1;
                }
            }
        }
        added
    }

    /// Register or update a peer.
    pub fn register_peer(&mut self, info: PeerInfo) -> bool {
        if info.id == self.self_id {
            return false;
        }
        if self.peers.len() >= self.config.max_peers && !self.peers.contains_key(&info.id) {
            return false;
        }
        self.peers.insert(info.id.clone(), PeerInfo {
            last_seen: self.current_tick,
            ..info
        });
        self.stats.peers_known = self.peers.len();
        true
    }

    /// Get a peer by id.
    pub fn get_peer(&self, id: &str) -> Option<&PeerInfo> {
        self.peers.get(id)
    }

    /// Number of known peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Query for peers matching a capability name.
    pub fn query_by_capability(&mut self, name: &str) -> Vec<&PeerInfo> {
        self.stats.total_queries += 1;
        self.peers.values().filter(|p| p.has_capability(name)).collect()
    }

    /// Query for peers matching a capability with minimum version.
    pub fn query_by_capability_version(&mut self, name: &str, min_version: u32) -> Vec<&PeerInfo> {
        self.stats.total_queries += 1;
        self.peers
            .values()
            .filter(|p| p.has_capability_version(name, min_version))
            .collect()
    }

    /// Remove peers not seen since `current_tick - peer_ttl`.
    pub fn expire_stale_peers(&mut self) -> Vec<String> {
        let cutoff = self.current_tick.saturating_sub(self.config.peer_ttl);
        let stale: Vec<String> = self
            .peers
            .iter()
            .filter(|(_, p)| p.last_seen < cutoff)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &stale {
            self.peers.remove(id);
            self.stats.total_expired += 1;
        }
        self.stats.peers_known = self.peers.len();
        stale
    }

    /// Produce a set of peers to share with another node (peer exchange).
    pub fn peers_for_exchange(&mut self) -> Vec<PeerInfo> {
        self.stats.total_exchanges += 1;
        self.peers
            .values()
            .take(self.config.exchange_limit)
            .cloned()
            .collect()
    }

    /// Receive peers from an exchange — merge into local table.
    pub fn receive_exchange(&mut self, peers: Vec<PeerInfo>) -> usize {
        let mut added = 0;
        for p in peers {
            if p.id == self.self_id {
                continue;
            }
            if let Some(existing) = self.peers.get_mut(&p.id) {
                if p.last_seen > existing.last_seen {
                    existing.last_seen = p.last_seen;
                    existing.address = p.address.clone();
                    existing.capabilities = p.capabilities.clone();
                }
            } else if self.peers.len() < self.config.max_peers {
                self.peers.insert(p.id.clone(), p);
                added += 1;
            }
        }
        self.stats.peers_known = self.peers.len();
        added
    }

    /// All known peer ids.
    pub fn peer_ids(&self) -> Vec<String> {
        self.peers.keys().cloned().collect()
    }

    /// Current statistics snapshot.
    pub fn stats(&self) -> DiscoveryStats {
        DiscoveryStats {
            peers_known: self.peers.len(),
            ..self.stats.clone()
        }
    }

    /// Self id.
    pub fn self_id(&self) -> &str {
        &self.self_id
    }

    /// Self address.
    pub fn self_address(&self) -> &str {
        &self.self_address
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_service() -> DiscoveryService {
        DiscoveryService::new("node-0", "10.0.0.1:5000")
    }

    fn make_peer(id: &str) -> PeerInfo {
        PeerInfo::new(id, format!("10.0.0.{}:5000", id.len()))
    }

    #[test]
    fn test_announce_self() {
        let mut ds = make_service();
        ds.add_capability(Capability::new("relay", 1));
        let info = ds.announce_self();
        assert_eq!(info.id, "node-0");
        assert!(info.has_capability("relay"));
    }

    #[test]
    fn test_register_and_get_peer() {
        let mut ds = make_service();
        let p = make_peer("node-1").with_capability(Capability::new("storage", 2));
        assert!(ds.register_peer(p));
        assert!(ds.get_peer("node-1").is_some());
        assert_eq!(ds.peer_count(), 1);
    }

    #[test]
    fn test_refuse_self_registration() {
        let mut ds = make_service();
        let p = PeerInfo::new("node-0", "10.0.0.1:5000");
        assert!(!ds.register_peer(p));
        assert_eq!(ds.peer_count(), 0);
    }

    #[test]
    fn test_max_peers_limit() {
        let cfg = DiscoveryConfig { max_peers: 2, ..Default::default() };
        let mut ds = make_service().with_config(cfg);
        ds.register_peer(make_peer("a"));
        ds.register_peer(make_peer("b"));
        assert!(!ds.register_peer(make_peer("c")));
        assert_eq!(ds.peer_count(), 2);
    }

    #[test]
    fn test_query_by_capability() {
        let mut ds = make_service();
        ds.register_peer(make_peer("a").with_capability(Capability::new("relay", 1)));
        ds.register_peer(make_peer("b").with_capability(Capability::new("storage", 1)));
        ds.register_peer(make_peer("c").with_capability(Capability::new("relay", 2)));
        let relays = ds.query_by_capability("relay");
        assert_eq!(relays.len(), 2);
    }

    #[test]
    fn test_query_by_capability_version() {
        let mut ds = make_service();
        ds.register_peer(make_peer("a").with_capability(Capability::new("relay", 1)));
        ds.register_peer(make_peer("b").with_capability(Capability::new("relay", 3)));
        let v2plus = ds.query_by_capability_version("relay", 2);
        assert_eq!(v2plus.len(), 1);
        assert_eq!(v2plus[0].id, "b");
    }

    #[test]
    fn test_expire_stale_peers() {
        let cfg = DiscoveryConfig { peer_ttl: 10, ..Default::default() };
        let mut ds = make_service().with_config(cfg);
        ds.tick(5);
        ds.register_peer(make_peer("old"));
        ds.tick(20);
        ds.register_peer(make_peer("new"));
        let stale = ds.expire_stale_peers();
        assert_eq!(stale, vec!["old".to_string()]);
        assert_eq!(ds.peer_count(), 1);
    }

    #[test]
    fn test_bootstrap_seed_nodes() {
        let mut ds = make_service();
        ds.add_seed_node("seed-1", "10.0.1.1:5000");
        ds.add_seed_node("seed-2", "10.0.1.2:5000");
        let added = ds.bootstrap();
        assert_eq!(added, 2);
        assert_eq!(ds.peer_count(), 2);
    }

    #[test]
    fn test_bootstrap_skips_self() {
        let mut ds = make_service();
        ds.add_seed_node("node-0", "10.0.0.1:5000");
        ds.add_seed_node("seed-1", "10.0.1.1:5000");
        let added = ds.bootstrap();
        assert_eq!(added, 1);
    }

    #[test]
    fn test_peer_exchange_produce() {
        let mut ds = make_service();
        ds.register_peer(make_peer("a"));
        ds.register_peer(make_peer("b"));
        let exchange = ds.peers_for_exchange();
        assert_eq!(exchange.len(), 2);
    }

    #[test]
    fn test_peer_exchange_limit() {
        let cfg = DiscoveryConfig { exchange_limit: 1, ..Default::default() };
        let mut ds = make_service().with_config(cfg);
        ds.register_peer(make_peer("a"));
        ds.register_peer(make_peer("b"));
        let exchange = ds.peers_for_exchange();
        assert_eq!(exchange.len(), 1);
    }

    #[test]
    fn test_receive_exchange_new_peers() {
        let mut ds = make_service();
        let peers = vec![make_peer("x"), make_peer("y")];
        let added = ds.receive_exchange(peers);
        assert_eq!(added, 2);
        assert_eq!(ds.peer_count(), 2);
    }

    #[test]
    fn test_receive_exchange_updates_existing() {
        let mut ds = make_service();
        ds.tick(5);
        ds.register_peer(make_peer("x").with_last_seen(5));
        let updated = vec![make_peer("x").with_last_seen(10)];
        ds.receive_exchange(updated);
        assert_eq!(ds.get_peer("x").unwrap().last_seen, 10);
    }

    #[test]
    fn test_receive_exchange_skips_self() {
        let mut ds = make_service();
        let peers = vec![PeerInfo::new("node-0", "10.0.0.1:5000")];
        let added = ds.receive_exchange(peers);
        assert_eq!(added, 0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut ds = make_service();
        ds.register_peer(make_peer("a"));
        ds.announce_self();
        ds.announce_self();
        ds.query_by_capability("x");
        let s = ds.stats();
        assert_eq!(s.total_announces, 2);
        assert_eq!(s.total_queries, 1);
        assert_eq!(s.peers_known, 1);
    }

    #[test]
    fn test_peer_ids() {
        let mut ds = make_service();
        ds.register_peer(make_peer("a"));
        ds.register_peer(make_peer("b"));
        let mut ids = ds.peer_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn test_capability_display() {
        let c = Capability::new("relay", 3);
        assert_eq!(format!("{}", c), "relay@v3");
    }

    #[test]
    fn test_peer_info_display() {
        let p = make_peer("node-1").with_capability(Capability::new("relay", 1));
        let display = format!("{}", p);
        assert!(display.contains("node-1"));
        assert!(display.contains("caps=1"));
    }

    #[test]
    fn test_expire_no_stale() {
        let mut ds = make_service();
        ds.tick(5);
        ds.register_peer(make_peer("a"));
        ds.tick(6);
        let stale = ds.expire_stale_peers();
        assert!(stale.is_empty());
    }

    #[test]
    fn test_multiple_capabilities_peer() {
        let mut ds = make_service();
        let p = make_peer("multi")
            .with_capability(Capability::new("relay", 1))
            .with_capability(Capability::new("storage", 2))
            .with_capability(Capability::new("compute", 1));
        ds.register_peer(p);
        assert_eq!(ds.query_by_capability("relay").len(), 1);
        assert_eq!(ds.query_by_capability("storage").len(), 1);
        assert_eq!(ds.query_by_capability("compute").len(), 1);
        assert_eq!(ds.query_by_capability("missing").len(), 0);
    }
}
