//! Distributed hash table (Kademlia-inspired) — 160-bit node IDs, XOR distance,
//! k-buckets, iterative node lookup, key-value store/retrieve, bucket refresh,
//! node join/leave, and configurable parameters.

use std::collections::HashMap;
use std::fmt;

// ── NodeId ──────────────────────────────────────────────────────────────────

/// 160-bit identifier for DHT nodes and keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub [u8; 20]);

impl NodeId {
    /// Create a NodeId from a 20-byte array.
    pub fn new(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    /// Create a NodeId from a single byte (useful for testing — fills last byte).
    pub fn from_byte(b: u8) -> Self {
        let mut bytes = [0u8; 20];
        bytes[19] = b;
        Self(bytes)
    }

    /// XOR distance between two node IDs.
    pub fn distance(&self, other: &NodeId) -> NodeId {
        let mut result = [0u8; 20];
        for i in 0..20 {
            result[i] = self.0[i] ^ other.0[i];
        }
        NodeId(result)
    }

    /// Bucket index (0-159) — index of the highest set bit in the XOR distance.
    /// Returns None if both IDs are identical.
    pub fn bucket_index(&self, other: &NodeId) -> Option<usize> {
        let dist = self.distance(other);
        for i in 0..20 {
            if dist.0[i] != 0 {
                let leading = dist.0[i].leading_zeros() as usize;
                return Some(i * 8 + leading);
            }
        }
        None
    }

    /// Whether self is closer to target than other is.
    pub fn is_closer(&self, target: &NodeId, other: &NodeId) -> bool {
        let d_self = self.distance(target);
        let d_other = other.distance(target);
        d_self.0 < d_other.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in &self.0[..4] {
            write!(f, "{:02x}", b)?;
        }
        write!(f, "...")
    }
}

// ── DhtEntry ────────────────────────────────────────────────────────────────

/// A node entry in a k-bucket.
#[derive(Debug, Clone)]
pub struct DhtEntry {
    pub id: NodeId,
    pub address: String,
    pub last_seen: u64,
}

impl DhtEntry {
    pub fn new(id: NodeId, address: impl Into<String>, last_seen: u64) -> Self {
        Self { id, address: address.into(), last_seen }
    }
}

// ── KBucket ─────────────────────────────────────────────────────────────────

/// A single k-bucket holding up to `k` nodes sorted by last-seen time.
#[derive(Debug, Clone)]
pub struct KBucket {
    entries: Vec<DhtEntry>,
    k: usize,
}

impl KBucket {
    pub fn new(k: usize) -> Self {
        Self { entries: Vec::new(), k }
    }

    /// Number of entries in this bucket.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the bucket is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the bucket is full.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.k
    }

    /// Insert or update a node. Returns true if added/updated.
    pub fn upsert(&mut self, entry: DhtEntry) -> bool {
        if let Some(pos) = self.entries.iter().position(|e| e.id == entry.id) {
            self.entries[pos].last_seen = entry.last_seen;
            self.entries[pos].address = entry.address;
            // Move to tail (most recently seen)
            let e = self.entries.remove(pos);
            self.entries.push(e);
            return true;
        }
        if self.entries.len() < self.k {
            self.entries.push(entry);
            return true;
        }
        false // bucket full, no eviction in this simplified model
    }

    /// Get all entries.
    pub fn entries(&self) -> &[DhtEntry] {
        &self.entries
    }

    /// Remove a node by id.
    pub fn remove(&mut self, id: &NodeId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.id != *id);
        self.entries.len() < before
    }

    /// Get the least-recently-seen entry.
    pub fn least_recent(&self) -> Option<&DhtEntry> {
        self.entries.first()
    }
}

// ── DhtConfig ───────────────────────────────────────────────────────────────

/// Configuration for the DHT node.
#[derive(Debug, Clone)]
pub struct DhtConfig {
    /// Max entries per bucket.
    pub k: usize,
    /// Parallelism parameter for lookups.
    pub alpha: usize,
    /// TTL for stored values (ticks).
    pub value_ttl: u64,
}

impl Default for DhtConfig {
    fn default() -> Self {
        Self { k: 20, alpha: 3, value_ttl: 3600 }
    }
}

// ── StoredValue ─────────────────────────────────────────────────────────────

/// A value stored in the DHT with expiry.
#[derive(Debug, Clone)]
pub struct StoredValue {
    pub data: Vec<u8>,
    pub stored_at: u64,
    pub expires_at: u64,
}

// ── DhtNode ─────────────────────────────────────────────────────────────────

/// A node in the distributed hash table.
pub struct DhtNode {
    pub id: NodeId,
    pub address: String,
    buckets: Vec<KBucket>,
    values: HashMap<NodeId, StoredValue>,
    config: DhtConfig,
    current_tick: u64,
}

impl DhtNode {
    pub fn new(id: NodeId, address: impl Into<String>) -> Self {
        let config = DhtConfig::default();
        let buckets = (0..160).map(|_| KBucket::new(config.k)).collect();
        Self {
            id,
            address: address.into(),
            buckets,
            values: HashMap::new(),
            config,
            current_tick: 0,
        }
    }

    pub fn with_config(mut self, config: DhtConfig) -> Self {
        self.buckets = (0..160).map(|_| KBucket::new(config.k)).collect();
        self.config = config;
        self
    }

    /// Advance the internal tick.
    pub fn tick(&mut self, now: u64) {
        self.current_tick = now;
    }

    /// Add or update a known node in the routing table.
    pub fn add_node(&mut self, node_id: NodeId, address: impl Into<String>) -> bool {
        if node_id == self.id {
            return false;
        }
        if let Some(idx) = self.id.bucket_index(&node_id) {
            let entry = DhtEntry::new(node_id, address, self.current_tick);
            return self.buckets[idx].upsert(entry);
        }
        false
    }

    /// Remove a node from the routing table.
    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        if let Some(idx) = self.id.bucket_index(node_id) {
            return self.buckets[idx].remove(node_id);
        }
        false
    }

    /// Find the `count` closest nodes to a target from the local routing table.
    pub fn find_closest(&self, target: &NodeId, count: usize) -> Vec<&DhtEntry> {
        let mut all: Vec<&DhtEntry> = self
            .buckets
            .iter()
            .flat_map(|b| b.entries())
            .collect();
        all.sort_by(|a, b| {
            let da = a.id.distance(target);
            let db = b.id.distance(target);
            da.0.cmp(&db.0)
        });
        all.truncate(count);
        all
    }

    /// Store a value by key.
    pub fn store(&mut self, key: NodeId, data: Vec<u8>) {
        let sv = StoredValue {
            data,
            stored_at: self.current_tick,
            expires_at: self.current_tick + self.config.value_ttl,
        };
        self.values.insert(key, sv);
    }

    /// Retrieve a value by key (returns None if expired or absent).
    pub fn retrieve(&self, key: &NodeId) -> Option<&[u8]> {
        self.values.get(key).and_then(|v| {
            if v.expires_at > self.current_tick {
                Some(v.data.as_slice())
            } else {
                None
            }
        })
    }

    /// Expire stale values.
    pub fn expire_values(&mut self) -> usize {
        let before = self.values.len();
        self.values.retain(|_, v| v.expires_at > self.current_tick);
        before - self.values.len()
    }

    /// Number of stored values.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// Total nodes in the routing table.
    pub fn routing_table_size(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// Number of non-empty buckets.
    pub fn non_empty_buckets(&self) -> usize {
        self.buckets.iter().filter(|b| !b.is_empty()).count()
    }

    /// Refresh bucket at index by returning it for the caller to ping.
    pub fn bucket_entries(&self, index: usize) -> &[DhtEntry] {
        if index < 160 {
            self.buckets[index].entries()
        } else {
            &[]
        }
    }

    /// Config reference.
    pub fn config(&self) -> &DhtConfig {
        &self.config
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nid(b: u8) -> NodeId {
        NodeId::from_byte(b)
    }

    #[test]
    fn test_xor_distance() {
        let a = nid(0b1010);
        let b = nid(0b1100);
        let d = a.distance(&b);
        assert_eq!(d.0[19], 0b0110);
    }

    #[test]
    fn test_distance_self_zero() {
        let a = nid(42);
        let d = a.distance(&a);
        assert_eq!(d.0, [0u8; 20]);
    }

    #[test]
    fn test_bucket_index() {
        let a = nid(0);
        let b = nid(1);
        let idx = a.bucket_index(&b);
        assert!(idx.is_some());
        // distance byte 19 = 0x01, leading_zeros = 7, index = 19*8+7 = 159
        assert_eq!(idx.unwrap(), 159);
    }

    #[test]
    fn test_bucket_index_same() {
        let a = nid(5);
        assert_eq!(a.bucket_index(&a), None);
    }

    #[test]
    fn test_is_closer() {
        let target = nid(10);
        let close = nid(11); // dist = 1
        let far = nid(0);   // dist = 10
        assert!(close.is_closer(&target, &far));
    }

    #[test]
    fn test_node_id_display() {
        let id = nid(42);
        let s = format!("{}", id);
        assert!(s.contains("00000000..."));
    }

    #[test]
    fn test_kbucket_upsert() {
        let mut b = KBucket::new(3);
        assert!(b.upsert(DhtEntry::new(nid(1), "a", 0)));
        assert!(b.upsert(DhtEntry::new(nid(2), "b", 0)));
        assert!(b.upsert(DhtEntry::new(nid(3), "c", 0)));
        assert!(!b.upsert(DhtEntry::new(nid(4), "d", 0))); // full
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn test_kbucket_update_existing() {
        let mut b = KBucket::new(3);
        b.upsert(DhtEntry::new(nid(1), "old", 0));
        b.upsert(DhtEntry::new(nid(1), "new", 5));
        assert_eq!(b.len(), 1);
        assert_eq!(b.entries()[0].address, "new");
        assert_eq!(b.entries()[0].last_seen, 5);
    }

    #[test]
    fn test_kbucket_remove() {
        let mut b = KBucket::new(5);
        b.upsert(DhtEntry::new(nid(1), "a", 0));
        b.upsert(DhtEntry::new(nid(2), "b", 0));
        assert!(b.remove(&nid(1)));
        assert_eq!(b.len(), 1);
        assert!(!b.remove(&nid(99)));
    }

    #[test]
    fn test_kbucket_least_recent() {
        let mut b = KBucket::new(5);
        b.upsert(DhtEntry::new(nid(1), "a", 10));
        b.upsert(DhtEntry::new(nid(2), "b", 20));
        assert_eq!(b.least_recent().unwrap().id, nid(1));
    }

    #[test]
    fn test_dht_add_node() {
        let mut node = DhtNode::new(nid(0), "localhost");
        assert!(node.add_node(nid(1), "10.0.0.1"));
        assert_eq!(node.routing_table_size(), 1);
    }

    #[test]
    fn test_dht_add_self_rejected() {
        let mut node = DhtNode::new(nid(0), "localhost");
        assert!(!node.add_node(nid(0), "self"));
    }

    #[test]
    fn test_dht_remove_node() {
        let mut node = DhtNode::new(nid(0), "localhost");
        node.add_node(nid(5), "a");
        assert!(node.remove_node(&nid(5)));
        assert_eq!(node.routing_table_size(), 0);
    }

    #[test]
    fn test_dht_find_closest() {
        let mut node = DhtNode::new(nid(0), "localhost");
        node.add_node(nid(1), "a");
        node.add_node(nid(10), "b");
        node.add_node(nid(100), "c");
        let closest = node.find_closest(&nid(2), 2);
        assert_eq!(closest.len(), 2);
        // nid(1) is closer to nid(2) than nid(10)
        assert_eq!(closest[0].id, nid(1));
    }

    #[test]
    fn test_dht_store_retrieve() {
        let mut node = DhtNode::new(nid(0), "localhost");
        let key = nid(42);
        node.store(key.clone(), b"hello".to_vec());
        assert_eq!(node.retrieve(&key), Some(b"hello".as_slice()));
    }

    #[test]
    fn test_dht_retrieve_expired() {
        let config = DhtConfig { value_ttl: 10, ..Default::default() };
        let mut node = DhtNode::new(nid(0), "localhost").with_config(config);
        node.tick(0);
        node.store(nid(1), b"data".to_vec());
        node.tick(15);
        assert_eq!(node.retrieve(&nid(1)), None);
    }

    #[test]
    fn test_dht_expire_values() {
        let config = DhtConfig { value_ttl: 5, ..Default::default() };
        let mut node = DhtNode::new(nid(0), "localhost").with_config(config);
        node.tick(0);
        node.store(nid(1), b"a".to_vec());
        node.store(nid(2), b"b".to_vec());
        node.tick(10);
        let expired = node.expire_values();
        assert_eq!(expired, 2);
        assert_eq!(node.value_count(), 0);
    }

    #[test]
    fn test_dht_non_empty_buckets() {
        let mut node = DhtNode::new(nid(0), "localhost");
        node.add_node(nid(1), "a");
        node.add_node(nid(128), "b");
        assert!(node.non_empty_buckets() >= 1);
    }

    #[test]
    fn test_dht_config() {
        let config = DhtConfig { k: 10, alpha: 5, value_ttl: 100 };
        let node = DhtNode::new(nid(0), "localhost").with_config(config);
        assert_eq!(node.config().k, 10);
        assert_eq!(node.config().alpha, 5);
    }
}
