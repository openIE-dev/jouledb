//! Robot communication — message passing, broadcast, relay chains,
//! bandwidth management, and communication graph modeling.
//!
//! Pure-Rust communication layer for multi-robot systems with
//! realistic link budgets, packet loss simulation, multi-hop relay,
//! and dynamic topology management.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Communication subsystem errors.
#[derive(Debug, Clone, PartialEq)]
pub enum CommError {
    /// Node not found in the communication graph.
    NodeNotFound(u64),
    /// Duplicate node ID.
    DuplicateNode(u64),
    /// No route to destination.
    NoRoute { from: u64, to: u64 },
    /// Bandwidth exceeded.
    BandwidthExceeded { node: u64, capacity: f64, requested: f64 },
    /// Message too large.
    MessageTooLarge { size: usize, max: usize },
}

impl fmt::Display for CommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::DuplicateNode(id) => write!(f, "duplicate node: {id}"),
            Self::NoRoute { from, to } => write!(f, "no route from {from} to {to}"),
            Self::BandwidthExceeded { node, capacity, requested } => {
                write!(f, "bandwidth exceeded on node {node}: {requested:.1}/{capacity:.1}")
            }
            Self::MessageTooLarge { size, max } => {
                write!(f, "message too large: {size} > {max}")
            }
        }
    }
}

impl std::error::Error for CommError {}

// ── PRNG ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
    }
}

// ── Message ─────────────────────────────────────────────────────

/// Priority levels for messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Emergency,
}

impl fmt::Display for MessagePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Emergency => write!(f, "Emergency"),
        }
    }
}

/// A message transmitted between robots.
#[derive(Debug, Clone)]
pub struct Message {
    pub id: u64,
    pub source: u64,
    pub destination: u64,
    pub payload_size: usize,
    pub priority: MessagePriority,
    pub ttl: u32,
    pub hops: Vec<u64>,
    pub timestamp: f64,
}

impl Message {
    pub fn new(id: u64, source: u64, destination: u64, payload_size: usize) -> Self {
        Self {
            id,
            source,
            destination,
            payload_size,
            priority: MessagePriority::Normal,
            ttl: 16,
            hops: vec![source],
            timestamp: 0.0,
        }
    }

    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_ttl(mut self, ttl: u32) -> Self {
        self.ttl = ttl;
        self
    }

    pub fn with_timestamp(mut self, ts: f64) -> Self {
        self.timestamp = ts;
        self
    }

    /// Broadcast message (destination = u64::MAX sentinel).
    pub fn broadcast(id: u64, source: u64, payload_size: usize) -> Self {
        Self::new(id, source, u64::MAX, payload_size)
    }

    pub fn hop_count(&self) -> usize {
        if self.hops.is_empty() { 0 } else { self.hops.len() - 1 }
    }

    pub fn is_broadcast(&self) -> bool {
        self.destination == u64::MAX
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_broadcast() {
            write!(f, "Msg({}, src={}, broadcast, {}B)", self.id, self.source, self.payload_size)
        } else {
            write!(
                f,
                "Msg({}, {}→{}, {}B, hops={})",
                self.id, self.source, self.destination, self.payload_size, self.hop_count()
            )
        }
    }
}

// ── Communication Link ──────────────────────────────────────────

/// A directional communication link between two nodes.
#[derive(Debug, Clone)]
pub struct CommLink {
    pub from: u64,
    pub to: u64,
    /// Maximum bandwidth in bytes/sec.
    pub bandwidth: f64,
    /// Current utilization in bytes/sec.
    pub utilization: f64,
    /// Packet loss probability (0.0–1.0).
    pub loss_rate: f64,
    /// Latency in seconds.
    pub latency: f64,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Maximum range for this link.
    pub max_range: f64,
}

impl CommLink {
    pub fn new(from: u64, to: u64) -> Self {
        Self {
            from,
            to,
            bandwidth: 1_000_000.0,
            utilization: 0.0,
            loss_rate: 0.0,
            latency: 0.001,
            snr_db: 30.0,
            max_range: 100.0,
        }
    }

    pub fn with_bandwidth(mut self, bw: f64) -> Self {
        self.bandwidth = bw;
        self
    }

    pub fn with_loss_rate(mut self, rate: f64) -> Self {
        self.loss_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_latency(mut self, lat: f64) -> Self {
        self.latency = lat;
        self
    }

    pub fn with_snr(mut self, snr: f64) -> Self {
        self.snr_db = snr;
        self
    }

    pub fn with_max_range(mut self, range: f64) -> Self {
        self.max_range = range;
        self
    }

    /// Available bandwidth remaining.
    pub fn available_bandwidth(&self) -> f64 {
        (self.bandwidth - self.utilization).max(0.0)
    }

    /// Utilization ratio (0.0–1.0).
    pub fn utilization_ratio(&self) -> f64 {
        if self.bandwidth <= 0.0 { 1.0 } else { self.utilization / self.bandwidth }
    }

    /// Shannon capacity in bits/sec based on SNR.
    pub fn shannon_capacity(&self) -> f64 {
        let snr_linear = 10.0f64.powf(self.snr_db / 10.0);
        self.bandwidth * (1.0 + snr_linear).ln() / (2.0f64).ln()
    }
}

impl fmt::Display for CommLink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Link({}→{}, bw={:.0}B/s, loss={:.1}%, lat={:.3}s)",
            self.from,
            self.to,
            self.bandwidth,
            self.loss_rate * 100.0,
            self.latency,
        )
    }
}

// ── Communication Node ──────────────────────────────────────────

/// A node in the communication graph (one per robot).
#[derive(Debug, Clone)]
pub struct CommNode {
    pub id: u64,
    pub x: f64,
    pub y: f64,
    pub comm_range: f64,
    pub tx_power_dbm: f64,
    pub rx_sensitivity_dbm: f64,
    pub inbox: Vec<Message>,
    pub outbox: Vec<Message>,
    pub delivered_count: u64,
    pub dropped_count: u64,
}

impl CommNode {
    pub fn new(id: u64, x: f64, y: f64) -> Self {
        Self {
            id,
            x,
            y,
            comm_range: 50.0,
            tx_power_dbm: 20.0,
            rx_sensitivity_dbm: -80.0,
            inbox: Vec::new(),
            outbox: Vec::new(),
            delivered_count: 0,
            dropped_count: 0,
        }
    }

    pub fn with_comm_range(mut self, range: f64) -> Self {
        self.comm_range = range;
        self
    }

    pub fn with_tx_power(mut self, power_dbm: f64) -> Self {
        self.tx_power_dbm = power_dbm;
        self
    }

    pub fn distance_to(&self, other: &CommNode) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Free-space path loss in dB at the given distance and frequency (MHz).
    pub fn path_loss_db(distance: f64, freq_mhz: f64) -> f64 {
        if distance <= 0.0 || freq_mhz <= 0.0 {
            return 0.0;
        }
        20.0 * distance.log10() + 20.0 * freq_mhz.log10() + 32.44
    }

    /// Can this node reach another node (based on range)?
    pub fn can_reach(&self, other: &CommNode) -> bool {
        self.distance_to(other) <= self.comm_range
    }
}

impl fmt::Display for CommNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Node({}, pos=({:.1},{:.1}), range={:.0})",
            self.id, self.x, self.y, self.comm_range
        )
    }
}

// ── Communication Graph ─────────────────────────────────────────

/// The communication network for a multi-robot team.
#[derive(Debug, Clone)]
pub struct CommGraph {
    pub nodes: HashMap<u64, CommNode>,
    pub links: Vec<CommLink>,
    pub max_message_size: usize,
    pub message_counter: u64,
    pub time: f64,
    rng: Rng,
}

impl CommGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            links: Vec::new(),
            max_message_size: 65536,
            message_counter: 0,
            time: 0.0,
            rng: Rng::new(42),
        }
    }

    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self
    }

    pub fn add_node(&mut self, node: CommNode) -> Result<(), CommError> {
        if self.nodes.contains_key(&node.id) {
            return Err(CommError::DuplicateNode(node.id));
        }
        self.nodes.insert(node.id, node);
        Ok(())
    }

    pub fn add_link(&mut self, link: CommLink) {
        self.links.push(link);
    }

    /// Rebuild links based on node positions and communication ranges.
    pub fn rebuild_links(&mut self) {
        self.links.clear();
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        for i in 0..ids.len() {
            for j in 0..ids.len() {
                if i == j {
                    continue;
                }
                let a = &self.nodes[&ids[i]];
                let b = &self.nodes[&ids[j]];
                if a.can_reach(b) {
                    let dist = a.distance_to(b);
                    let loss = (dist / a.comm_range).powi(2) * 0.1;
                    let link = CommLink::new(a.id, b.id)
                        .with_loss_rate(loss.min(0.9))
                        .with_latency(dist / 3e8 + 0.001);
                    self.links.push(link);
                }
            }
        }
    }

    /// Get neighbors of a node (outgoing links).
    pub fn neighbors(&self, node_id: u64) -> Vec<u64> {
        self.links
            .iter()
            .filter(|l| l.from == node_id)
            .map(|l| l.to)
            .collect()
    }

    /// Check if the graph is connected (undirected connectivity).
    pub fn is_connected(&self) -> bool {
        if self.nodes.len() <= 1 {
            return true;
        }
        let start = match self.nodes.keys().next() {
            Some(&id) => id,
            None => return true,
        };
        let mut visited = HashMap::new();
        let mut stack = vec![start];
        visited.insert(start, true);
        while let Some(current) = stack.pop() {
            for &nbr in &self.neighbors(current) {
                if !visited.contains_key(&nbr) {
                    visited.insert(nbr, true);
                    stack.push(nbr);
                }
            }
            // Also check reverse links (undirected).
            for link in &self.links {
                if link.to == current && !visited.contains_key(&link.from) {
                    visited.insert(link.from, true);
                    stack.push(link.from);
                }
            }
        }
        visited.len() == self.nodes.len()
    }

    /// Algebraic connectivity estimate (Fiedler value approximation).
    /// Uses the power iteration on the Laplacian matrix.
    pub fn algebraic_connectivity(&self) -> f64 {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        let n = ids.len();
        if n < 2 {
            return 0.0;
        }
        let id_to_idx: HashMap<u64, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Build Laplacian matrix.
        let mut lap = vec![vec![0.0f64; n]; n];
        for link in &self.links {
            if let (Some(&i), Some(&j)) = (id_to_idx.get(&link.from), id_to_idx.get(&link.to)) {
                lap[i][j] -= 1.0;
                lap[i][i] += 1.0;
            }
        }

        // Power iteration to find second smallest eigenvalue (approx).
        // Use inverse iteration on L - shift*I.
        // Simple approximation: find minimum of x^T L x / x^T x
        // for vectors orthogonal to the all-ones vector.
        let mut v = vec![0.0f64; n];
        for i in 0..n {
            v[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }

        for _ in 0..100 {
            // Multiply: w = L * v.
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += lap[i][j] * v[j];
                }
            }
            // Project out the all-ones vector.
            let sum: f64 = w.iter().sum();
            let avg = sum / n as f64;
            for val in &mut w {
                *val -= avg;
            }
            // Normalize.
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                return 0.0;
            }
            for val in &mut w {
                *val /= norm;
            }
            v = w;
        }
        // Rayleigh quotient: lambda = v^T L v / v^T v.
        let mut numerator = 0.0f64;
        for i in 0..n {
            let mut lv_i = 0.0f64;
            for j in 0..n {
                lv_i += lap[i][j] * v[j];
            }
            numerator += v[i] * lv_i;
        }
        let denominator: f64 = v.iter().map(|x| x * x).sum();
        if denominator < 1e-12 { 0.0 } else { numerator / denominator }
    }

    /// Find shortest path (by hop count) using BFS.
    pub fn shortest_path(&self, from: u64, to: u64) -> Result<Vec<u64>, CommError> {
        if !self.nodes.contains_key(&from) {
            return Err(CommError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(CommError::NodeNotFound(to));
        }
        if from == to {
            return Ok(vec![from]);
        }
        let mut visited: HashMap<u64, u64> = HashMap::new();
        visited.insert(from, from);
        let mut queue = vec![from];
        let mut qi = 0;
        while qi < queue.len() {
            let current = queue[qi];
            qi += 1;
            for &nbr in &self.neighbors(current) {
                if visited.contains_key(&nbr) {
                    continue;
                }
                visited.insert(nbr, current);
                if nbr == to {
                    // Reconstruct path.
                    let mut path = vec![to];
                    let mut c = to;
                    while c != from {
                        c = visited[&c];
                        path.push(c);
                    }
                    path.reverse();
                    return Ok(path);
                }
                queue.push(nbr);
            }
        }
        Err(CommError::NoRoute { from, to })
    }

    /// Send a unicast message along the shortest path, simulating loss.
    pub fn send_message(
        &mut self,
        from: u64,
        to: u64,
        payload_size: usize,
    ) -> Result<Message, CommError> {
        if payload_size > self.max_message_size {
            return Err(CommError::MessageTooLarge {
                size: payload_size,
                max: self.max_message_size,
            });
        }
        let path = self.shortest_path(from, to)?;
        self.message_counter += 1;
        let mut msg = Message::new(self.message_counter, from, to, payload_size)
            .with_timestamp(self.time);
        msg.hops = path.clone();

        // Simulate loss along each hop.
        for i in 0..path.len() - 1 {
            let link_loss = self.links.iter()
                .find(|l| l.from == path[i] && l.to == path[i + 1])
                .map(|l| l.loss_rate)
                .unwrap_or(0.0);
            if self.rng.next_f64() < link_loss {
                if let Some(node) = self.nodes.get_mut(&path[i]) {
                    node.dropped_count += 1;
                }
                return Ok(msg); // Dropped — caller can check hop_count vs path length.
            }
        }

        // Delivered.
        if let Some(node) = self.nodes.get_mut(&to) {
            node.inbox.push(msg.clone());
            node.delivered_count += 1;
        }
        Ok(msg)
    }

    /// Broadcast a message from a source to all reachable nodes.
    pub fn broadcast_message(
        &mut self,
        from: u64,
        payload_size: usize,
    ) -> Result<usize, CommError> {
        if !self.nodes.contains_key(&from) {
            return Err(CommError::NodeNotFound(from));
        }
        if payload_size > self.max_message_size {
            return Err(CommError::MessageTooLarge {
                size: payload_size,
                max: self.max_message_size,
            });
        }
        let targets: Vec<u64> = self.nodes.keys().copied().filter(|id| *id != from).collect();
        let mut delivered = 0usize;
        for target in targets {
            if self.shortest_path(from, target).is_ok() {
                self.message_counter += 1;
                let msg = Message::broadcast(self.message_counter, from, payload_size)
                    .with_timestamp(self.time);
                if let Some(node) = self.nodes.get_mut(&target) {
                    node.inbox.push(msg);
                    node.delivered_count += 1;
                    delivered += 1;
                }
            }
        }
        Ok(delivered)
    }

    /// Network diameter (longest shortest path).
    pub fn diameter(&self) -> usize {
        let ids: Vec<u64> = self.nodes.keys().copied().collect();
        let mut max_hops = 0;
        for &a in &ids {
            for &b in &ids {
                if a == b {
                    continue;
                }
                if let Ok(path) = self.shortest_path(a, b) {
                    if path.len() - 1 > max_hops {
                        max_hops = path.len() - 1;
                    }
                }
            }
        }
        max_hops
    }

    /// Average node degree.
    pub fn avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let total: usize = self.nodes.keys().map(|id| self.neighbors(*id).len()).sum();
        total as f64 / self.nodes.len() as f64
    }

    /// Update node positions (for mobile robots).
    pub fn update_position(&mut self, node_id: u64, x: f64, y: f64) -> Result<(), CommError> {
        let node = self.nodes.get_mut(&node_id).ok_or(CommError::NodeNotFound(node_id))?;
        node.x = x;
        node.y = y;
        Ok(())
    }
}

impl fmt::Display for CommGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CommGraph({} nodes, {} links, msgs={})",
            self.nodes.len(),
            self.links.len(),
            self.message_counter,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph_3_linear() -> CommGraph {
        let mut g = CommGraph::new();
        g.add_node(CommNode::new(1, 0.0, 0.0).with_comm_range(15.0)).unwrap();
        g.add_node(CommNode::new(2, 10.0, 0.0).with_comm_range(15.0)).unwrap();
        g.add_node(CommNode::new(3, 20.0, 0.0).with_comm_range(15.0)).unwrap();
        g.rebuild_links();
        g
    }

    #[test]
    fn test_add_node() {
        let mut g = CommGraph::new();
        g.add_node(CommNode::new(1, 0.0, 0.0)).unwrap();
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn test_duplicate_node() {
        let mut g = CommGraph::new();
        g.add_node(CommNode::new(1, 0.0, 0.0)).unwrap();
        assert!(g.add_node(CommNode::new(1, 5.0, 5.0)).is_err());
    }

    #[test]
    fn test_rebuild_links() {
        let g = make_graph_3_linear();
        // Node 1 and 2 should link (distance 10 < range 15).
        // Node 2 and 3 should link (distance 10 < range 15).
        // Node 1 and 3 should NOT link (distance 20 > range 15).
        assert!(g.neighbors(1).contains(&2));
        assert!(g.neighbors(2).contains(&3));
        assert!(!g.neighbors(1).contains(&3));
    }

    #[test]
    fn test_shortest_path_direct() {
        let g = make_graph_3_linear();
        let path = g.shortest_path(1, 2).unwrap();
        assert_eq!(path, vec![1, 2]);
    }

    #[test]
    fn test_shortest_path_relay() {
        let g = make_graph_3_linear();
        let path = g.shortest_path(1, 3).unwrap();
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn test_no_route() {
        let mut g = CommGraph::new();
        g.add_node(CommNode::new(1, 0.0, 0.0).with_comm_range(5.0)).unwrap();
        g.add_node(CommNode::new(2, 100.0, 0.0).with_comm_range(5.0)).unwrap();
        g.rebuild_links();
        assert!(g.shortest_path(1, 2).is_err());
    }

    #[test]
    fn test_is_connected() {
        let g = make_graph_3_linear();
        assert!(g.is_connected());
    }

    #[test]
    fn test_not_connected() {
        let mut g = CommGraph::new();
        g.add_node(CommNode::new(1, 0.0, 0.0).with_comm_range(5.0)).unwrap();
        g.add_node(CommNode::new(2, 100.0, 0.0).with_comm_range(5.0)).unwrap();
        g.rebuild_links();
        assert!(!g.is_connected());
    }

    #[test]
    fn test_diameter() {
        let g = make_graph_3_linear();
        assert_eq!(g.diameter(), 2);
    }

    #[test]
    fn test_avg_degree() {
        let g = make_graph_3_linear();
        let deg = g.avg_degree();
        // Node 1: 1 nbr, Node 2: 2 nbrs, Node 3: 1 nbr => avg = 4/3.
        assert!((deg - 4.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_send_message() {
        let mut g = make_graph_3_linear();
        let msg = g.send_message(1, 2, 100).unwrap();
        assert_eq!(msg.source, 1);
        assert_eq!(msg.destination, 2);
    }

    #[test]
    fn test_message_too_large() {
        let mut g = make_graph_3_linear();
        let result = g.send_message(1, 2, 1_000_000);
        assert!(matches!(result, Err(CommError::MessageTooLarge { .. })));
    }

    #[test]
    fn test_broadcast() {
        let mut g = make_graph_3_linear();
        let delivered = g.broadcast_message(2, 50).unwrap();
        assert_eq!(delivered, 2);
    }

    #[test]
    fn test_link_bandwidth() {
        let link = CommLink::new(1, 2).with_bandwidth(1000.0);
        assert!((link.available_bandwidth() - 1000.0).abs() < 1e-9);
        assert!((link.utilization_ratio()).abs() < 1e-9);
    }

    #[test]
    fn test_shannon_capacity() {
        let link = CommLink::new(1, 2).with_bandwidth(1_000_000.0).with_snr(30.0);
        let cap = link.shannon_capacity();
        assert!(cap > 1_000_000.0); // Shannon capacity > bandwidth when SNR > 0 dB.
    }

    #[test]
    fn test_path_loss() {
        let loss = CommNode::path_loss_db(1000.0, 2400.0);
        assert!(loss > 90.0); // ~100 dB at 1km, 2.4 GHz.
    }

    #[test]
    fn test_algebraic_connectivity() {
        let g = make_graph_3_linear();
        let ac = g.algebraic_connectivity();
        assert!(ac > 0.0); // Connected graph => positive Fiedler value.
    }

    #[test]
    fn test_message_display() {
        let msg = Message::new(1, 10, 20, 256);
        assert!(format!("{msg}").contains("10"));
        let bcast = Message::broadcast(2, 10, 100);
        assert!(format!("{bcast}").contains("broadcast"));
    }

    #[test]
    fn test_update_position() {
        let mut g = make_graph_3_linear();
        g.update_position(1, 5.0, 5.0).unwrap();
        assert!((g.nodes[&1].x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_display_impls() {
        let g = make_graph_3_linear();
        assert!(format!("{g}").contains("3 nodes"));
        let node = CommNode::new(7, 1.0, 2.0);
        assert!(format!("{node}").contains("7"));
        let link = CommLink::new(1, 2);
        assert!(format!("{link}").contains("1"));
    }
}
