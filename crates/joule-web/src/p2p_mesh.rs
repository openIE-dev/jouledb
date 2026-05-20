//! Peer-to-peer mesh network model — peer management, message routing,
//! heartbeat/keepalive, NAT traversal concepts, and reputation scoring.
//!
//! Replaces libp2p / simple-peer with a pure Rust mesh networking model.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Peer ───────────────────────────────────────────────────────

/// Unique identifier for a peer.
pub type PeerId = String;

/// Peer connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerState {
    Connecting,
    Connected,
    Disconnected,
}

impl fmt::Display for PeerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connecting => write!(f, "connecting"),
            Self::Connected => write!(f, "connected"),
            Self::Disconnected => write!(f, "disconnected"),
        }
    }
}

/// A peer in the mesh network.
#[derive(Debug, Clone)]
pub struct Peer {
    pub id: PeerId,
    pub address: String,
    pub state: PeerState,
    pub last_seen: u64,
    pub reputation: f64,
    pub metadata: HashMap<String, String>,
}

impl Peer {
    pub fn new(id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            address: address.into(),
            state: PeerState::Disconnected,
            last_seen: 0,
            reputation: 1.0,
            metadata: HashMap::new(),
        }
    }

    pub fn is_connected(&self) -> bool {
        self.state == PeerState::Connected
    }

    /// Check if this peer has timed out given the current time and threshold.
    pub fn is_timed_out(&self, now: u64, timeout_secs: u64) -> bool {
        if self.last_seen == 0 {
            return false;
        }
        now.saturating_sub(self.last_seen) > timeout_secs
    }
}

// ── Mesh Topology ──────────────────────────────────────────────

/// Mesh network topology type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshTopology {
    /// Every peer connects to every other peer.
    FullMesh,
    /// Peers connect to a subset of other peers.
    PartialMesh { max_connections: usize },
}

// ── Message ────────────────────────────────────────────────────

/// Message routing mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageTarget {
    /// Send to all peers.
    Broadcast,
    /// Send to one specific peer.
    Unicast(PeerId),
    /// Send to a named group of peers.
    Multicast(String),
}

/// A message in the mesh network.
#[derive(Debug, Clone)]
pub struct MeshMessage {
    pub id: u64,
    pub from: PeerId,
    pub target: MessageTarget,
    pub payload: Vec<u8>,
    pub ttl: u8,
    pub timestamp: u64,
}

impl MeshMessage {
    pub fn new(
        id: u64,
        from: impl Into<PeerId>,
        target: MessageTarget,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            id,
            from: from.into(),
            target,
            payload,
            ttl: 7,
            timestamp: 0,
        }
    }

    pub fn with_ttl(mut self, ttl: u8) -> Self {
        self.ttl = ttl;
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }
}

// ── Peer Discovery (Gossip) ────────────────────────────────────

/// Peer discovery via gossip protocol concept.
#[derive(Debug, Clone)]
pub struct PeerDiscovery {
    known_peers: HashMap<PeerId, String>,
    seen_announcements: HashSet<PeerId>,
}

impl PeerDiscovery {
    pub fn new() -> Self {
        Self {
            known_peers: HashMap::new(),
            seen_announcements: HashSet::new(),
        }
    }

    /// Announce a peer (returns true if this is a new peer).
    pub fn announce(&mut self, id: PeerId, address: String) -> bool {
        if self.seen_announcements.contains(&id) {
            return false;
        }
        self.seen_announcements.insert(id.clone());
        self.known_peers.insert(id, address);
        true
    }

    /// Get all known peer addresses.
    pub fn known_peers(&self) -> &HashMap<PeerId, String> {
        &self.known_peers
    }

    /// Remove a peer.
    pub fn remove(&mut self, id: &str) {
        self.known_peers.remove(id);
        self.seen_announcements.remove(id);
    }

    pub fn len(&self) -> usize {
        self.known_peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.known_peers.is_empty()
    }

    /// Generate gossip announcements to share with a peer.
    /// Returns peers the target hasn't seen (simplified concept).
    pub fn gossip_for(&self, exclude: &PeerId) -> Vec<(PeerId, String)> {
        self.known_peers
            .iter()
            .filter(|(id, _)| *id != exclude)
            .map(|(id, addr)| (id.clone(), addr.clone()))
            .collect()
    }
}

impl Default for PeerDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

// ── NAT Traversal (STUN concept) ──────────────────────────────

/// STUN message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunMessageType {
    BindingRequest,
    BindingResponse,
    BindingErrorResponse,
}

/// A simplified STUN binding request.
#[derive(Debug, Clone)]
pub struct StunBindingRequest {
    pub transaction_id: [u8; 12],
}

impl StunBindingRequest {
    pub fn new(transaction_id: [u8; 12]) -> Self {
        Self { transaction_id }
    }

    /// Serialize to bytes (simplified STUN header: type(2) + length(2) + magic(4) + txn(12)).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        // Binding Request = 0x0001
        buf.extend_from_slice(&0x0001u16.to_be_bytes());
        // Length = 0 (no attributes)
        buf.extend_from_slice(&0u16.to_be_bytes());
        // Magic cookie
        buf.extend_from_slice(&0x2112A442u32.to_be_bytes());
        // Transaction ID
        buf.extend_from_slice(&self.transaction_id);
        buf
    }
}

/// A simplified STUN binding response.
#[derive(Debug, Clone)]
pub struct StunBindingResponse {
    pub transaction_id: [u8; 12],
    /// The mapped address (reflexive transport address).
    pub mapped_address: String,
    pub mapped_port: u16,
}

impl StunBindingResponse {
    pub fn new(
        transaction_id: [u8; 12],
        address: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            transaction_id,
            mapped_address: address.into(),
            mapped_port: port,
        }
    }
}

// ── Mesh Network ───────────────────────────────────────────────

/// The mesh network coordinator.
#[derive(Debug, Clone)]
pub struct MeshNetwork {
    pub local_id: PeerId,
    pub topology: MeshTopology,
    peers: HashMap<PeerId, Peer>,
    groups: HashMap<String, HashSet<PeerId>>,
    message_log: Vec<MeshMessage>,
    next_message_id: u64,
    seen_message_ids: HashSet<u64>,
}

impl MeshNetwork {
    pub fn new(local_id: impl Into<PeerId>, topology: MeshTopology) -> Self {
        Self {
            local_id: local_id.into(),
            topology,
            peers: HashMap::new(),
            groups: HashMap::new(),
            message_log: Vec::new(),
            next_message_id: 1,
            seen_message_ids: HashSet::new(),
        }
    }

    /// Add a peer to the network.
    pub fn add_peer(&mut self, peer: Peer) {
        self.peers.insert(peer.id.clone(), peer);
    }

    /// Remove a peer.
    pub fn remove_peer(&mut self, id: &str) -> Option<Peer> {
        // Also remove from groups.
        for members in self.groups.values_mut() {
            members.remove(id);
        }
        self.peers.remove(id)
    }

    /// Get a peer by ID.
    pub fn peer(&self, id: &str) -> Option<&Peer> {
        self.peers.get(id)
    }

    /// Get a mutable peer by ID.
    pub fn peer_mut(&mut self, id: &str) -> Option<&mut Peer> {
        self.peers.get_mut(id)
    }

    /// Set peer state.
    pub fn set_peer_state(&mut self, id: &str, state: PeerState) -> bool {
        if let Some(peer) = self.peers.get_mut(id) {
            peer.state = state;
            true
        } else {
            false
        }
    }

    /// Get all connected peers.
    pub fn connected_peers(&self) -> Vec<&Peer> {
        self.peers.values().filter(|p| p.is_connected()).collect()
    }

    /// Total peer count.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Create a multicast group.
    pub fn create_group(&mut self, name: impl Into<String>) {
        self.groups.entry(name.into()).or_default();
    }

    /// Add a peer to a group.
    pub fn add_to_group(&mut self, group: &str, peer_id: PeerId) -> bool {
        if let Some(members) = self.groups.get_mut(group) {
            members.insert(peer_id);
            true
        } else {
            false
        }
    }

    /// Route a broadcast message. Returns list of recipient peer IDs.
    pub fn route_broadcast(&mut self, payload: Vec<u8>) -> Vec<PeerId> {
        let id = self.next_message_id;
        self.next_message_id += 1;
        let recipients: Vec<PeerId> = self
            .peers
            .values()
            .filter(|p| p.is_connected())
            .map(|p| p.id.clone())
            .collect();

        let msg = MeshMessage::new(
            id,
            self.local_id.clone(),
            MessageTarget::Broadcast,
            payload,
        );
        self.seen_message_ids.insert(id);
        self.message_log.push(msg);
        recipients
    }

    /// Route a unicast message. Returns true if the peer is connected.
    pub fn route_unicast(&mut self, target: &str, payload: Vec<u8>) -> bool {
        let connected = self
            .peers
            .get(target)
            .map(|p| p.is_connected())
            .unwrap_or(false);

        let id = self.next_message_id;
        self.next_message_id += 1;
        let msg = MeshMessage::new(
            id,
            self.local_id.clone(),
            MessageTarget::Unicast(target.to_string()),
            payload,
        );
        self.seen_message_ids.insert(id);
        self.message_log.push(msg);
        connected
    }

    /// Route a multicast message to a group. Returns recipient peer IDs.
    pub fn route_multicast(&mut self, group: &str, payload: Vec<u8>) -> Vec<PeerId> {
        let members = self
            .groups
            .get(group)
            .cloned()
            .unwrap_or_default();

        let recipients: Vec<PeerId> = members
            .iter()
            .filter(|id| {
                self.peers
                    .get(id.as_str())
                    .map(|p| p.is_connected())
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        let id = self.next_message_id;
        self.next_message_id += 1;
        let msg = MeshMessage::new(
            id,
            self.local_id.clone(),
            MessageTarget::Multicast(group.to_string()),
            payload,
        );
        self.seen_message_ids.insert(id);
        self.message_log.push(msg);
        recipients
    }

    /// Check if a message has already been seen (dedup for flooding).
    pub fn is_seen(&self, message_id: u64) -> bool {
        self.seen_message_ids.contains(&message_id)
    }

    /// Process a heartbeat from a peer.
    pub fn heartbeat(&mut self, peer_id: &str, now: u64) -> bool {
        if let Some(peer) = self.peers.get_mut(peer_id) {
            peer.last_seen = now;
            if peer.state != PeerState::Connected {
                peer.state = PeerState::Connected;
            }
            true
        } else {
            false
        }
    }

    /// Detect timed-out peers and mark them disconnected.
    pub fn detect_timeouts(&mut self, now: u64, timeout_secs: u64) -> Vec<PeerId> {
        let mut timed_out = Vec::new();
        for peer in self.peers.values_mut() {
            if peer.state == PeerState::Connected
                && peer.is_timed_out(now, timeout_secs)
            {
                peer.state = PeerState::Disconnected;
                timed_out.push(peer.id.clone());
            }
        }
        timed_out
    }

    /// Update a peer's reputation score.
    pub fn update_reputation(&mut self, peer_id: &str, delta: f64) -> Option<f64> {
        let peer = self.peers.get_mut(peer_id)?;
        peer.reputation = (peer.reputation + delta).clamp(0.0, 10.0);
        Some(peer.reputation)
    }

    /// Get peers sorted by reputation (highest first).
    pub fn peers_by_reputation(&self) -> Vec<&Peer> {
        let mut peers: Vec<&Peer> = self.peers.values().collect();
        peers.sort_by(|a, b| b.reputation.partial_cmp(&a.reputation).unwrap());
        peers
    }

    /// Message log length.
    pub fn message_count(&self) -> usize {
        self.message_log.len()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_network() -> MeshNetwork {
        let mut net = MeshNetwork::new("local", MeshTopology::FullMesh);
        let mut p1 = Peer::new("peer-1", "192.168.1.1:8000");
        p1.state = PeerState::Connected;
        p1.last_seen = 100;
        let mut p2 = Peer::new("peer-2", "192.168.1.2:8000");
        p2.state = PeerState::Connected;
        p2.last_seen = 100;
        let p3 = Peer::new("peer-3", "192.168.1.3:8000");
        net.add_peer(p1);
        net.add_peer(p2);
        net.add_peer(p3);
        net
    }

    #[test]
    fn peer_basics() {
        let p = Peer::new("node-1", "10.0.0.1:9000");
        assert_eq!(p.state, PeerState::Disconnected);
        assert!(!p.is_connected());
        assert_eq!(p.reputation, 1.0);
    }

    #[test]
    fn peer_timeout() {
        let mut p = Peer::new("node-1", "10.0.0.1:9000");
        p.last_seen = 100;
        assert!(!p.is_timed_out(110, 30));
        assert!(p.is_timed_out(200, 30));
    }

    #[test]
    fn add_remove_peers() {
        let mut net = make_network();
        assert_eq!(net.peer_count(), 3);
        net.remove_peer("peer-1");
        assert_eq!(net.peer_count(), 2);
        assert!(net.peer("peer-1").is_none());
    }

    #[test]
    fn connected_peers() {
        let net = make_network();
        let connected = net.connected_peers();
        assert_eq!(connected.len(), 2);
    }

    #[test]
    fn broadcast_routing() {
        let mut net = make_network();
        let recipients = net.route_broadcast(b"hello all".to_vec());
        assert_eq!(recipients.len(), 2);
        assert_eq!(net.message_count(), 1);
    }

    #[test]
    fn unicast_routing() {
        let mut net = make_network();
        assert!(net.route_unicast("peer-1", b"hello".to_vec()));
        assert!(!net.route_unicast("peer-3", b"hello".to_vec())); // disconnected
        assert!(!net.route_unicast("unknown", b"hello".to_vec()));
    }

    #[test]
    fn multicast_routing() {
        let mut net = make_network();
        net.create_group("team-a");
        net.add_to_group("team-a", "peer-1".into());
        net.add_to_group("team-a", "peer-3".into()); // disconnected

        let recipients = net.route_multicast("team-a", b"team msg".to_vec());
        assert_eq!(recipients.len(), 1);
        assert_eq!(recipients[0], "peer-1");
    }

    #[test]
    fn heartbeat_and_timeout() {
        let mut net = make_network();
        net.heartbeat("peer-1", 200);
        net.heartbeat("peer-2", 200);

        // At time 230, 30s timeout — nobody timed out yet.
        let timed_out = net.detect_timeouts(230, 30);
        assert!(timed_out.is_empty());

        // At 231, past the boundary.
        let timed_out = net.detect_timeouts(231, 30);
        assert_eq!(timed_out.len(), 2);

        // Both are now disconnected.
        assert_eq!(net.connected_peers().len(), 0);
    }

    #[test]
    fn reputation_scoring() {
        let mut net = make_network();
        net.update_reputation("peer-1", 2.0);
        assert_eq!(net.peer("peer-1").unwrap().reputation, 3.0);

        // Clamp to [0, 10].
        net.update_reputation("peer-1", 100.0);
        assert_eq!(net.peer("peer-1").unwrap().reputation, 10.0);

        net.update_reputation("peer-2", -5.0);
        assert_eq!(net.peer("peer-2").unwrap().reputation, 0.0);
    }

    #[test]
    fn peers_by_reputation() {
        let mut net = make_network();
        net.update_reputation("peer-1", 4.0); // 5.0
        net.update_reputation("peer-2", -0.5); // 0.5

        let ranked = net.peers_by_reputation();
        assert_eq!(ranked[0].id, "peer-1");
    }

    #[test]
    fn message_dedup() {
        let mut net = make_network();
        net.route_broadcast(b"first".to_vec());
        assert!(net.is_seen(1));
        assert!(!net.is_seen(999));
    }

    #[test]
    fn peer_discovery_gossip() {
        let mut disco = PeerDiscovery::new();
        assert!(disco.announce("p1".into(), "10.0.0.1:8000".into()));
        assert!(disco.announce("p2".into(), "10.0.0.2:8000".into()));
        // Duplicate announcement.
        assert!(!disco.announce("p1".into(), "10.0.0.1:8000".into()));

        assert_eq!(disco.len(), 2);
        let gossip = disco.gossip_for(&"p1".into());
        assert_eq!(gossip.len(), 1);
        assert_eq!(gossip[0].0, "p2");
    }

    #[test]
    fn stun_binding() {
        let txn = [1u8; 12];
        let req = StunBindingRequest::new(txn);
        let bytes = req.to_bytes();
        assert_eq!(bytes.len(), 20);
        // Type = 0x0001
        assert_eq!(bytes[0], 0x00);
        assert_eq!(bytes[1], 0x01);
        // Magic cookie
        assert_eq!(bytes[4], 0x21);
        assert_eq!(bytes[5], 0x12);

        let resp = StunBindingResponse::new(txn, "203.0.113.1", 54321);
        assert_eq!(resp.mapped_address, "203.0.113.1");
        assert_eq!(resp.mapped_port, 54321);
    }

    #[test]
    fn partial_mesh_topology() {
        let net = MeshNetwork::new(
            "local",
            MeshTopology::PartialMesh { max_connections: 3 },
        );
        assert!(matches!(
            net.topology,
            MeshTopology::PartialMesh { max_connections: 3 }
        ));
    }
}
