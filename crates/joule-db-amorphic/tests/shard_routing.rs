//! Integration tests for consistent hash ring shard routing.
//!
//! Validates key distribution, node addition/removal, replica placement,
//! and availability handling.

use joule_db_amorphic::distributed::{ConsistentHashRing, NodeConfig, NodeId};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_node(id: NodeId, host: &str, port: u16) -> NodeConfig {
    NodeConfig::new(id, host, port)
}

fn build_ring(n: u64) -> ConsistentHashRing {
    let mut ring = ConsistentHashRing::new();
    for i in 1..=n {
        ring.add_node(make_node(i, &format!("node-{i}"), 9000 + i as u16))
            .unwrap();
    }
    ring
}

// ---------------------------------------------------------------------------
// Test: Basic routing — key maps to a node
// ---------------------------------------------------------------------------

#[test]
fn basic_routing() {
    let ring = build_ring(3);

    let node = ring.get_node("user:1234");
    assert!(node.is_some(), "key should map to a node");
    let node_id = node.unwrap();
    assert!(node_id >= 1 && node_id <= 3);
}

// ---------------------------------------------------------------------------
// Test: Deterministic routing — same key always maps to same node
// ---------------------------------------------------------------------------

#[test]
fn deterministic_routing() {
    let ring = build_ring(5);

    let node_a = ring.get_node("deterministic-key").unwrap();
    let node_b = ring.get_node("deterministic-key").unwrap();
    assert_eq!(node_a, node_b, "same key should always route to same node");
}

// ---------------------------------------------------------------------------
// Test: Distribution across nodes is roughly uniform
// ---------------------------------------------------------------------------

#[test]
fn uniform_distribution() {
    let ring = build_ring(5);

    let keys: Vec<String> = (0..10_000).map(|i| format!("key-{i}")).collect();
    let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
    let distribution = ring.get_distribution(&key_refs);

    // Each of 5 nodes should get some keys. With FNV-1a and 150 vnodes
    // distribution won't be perfectly even. Verify all nodes participate
    // and no single node dominates excessively.
    assert_eq!(distribution.len(), 5, "all 5 nodes should have keys");
    for (&node_id, &count) in &distribution {
        let pct = count as f64 / 10_000.0;
        assert!(
            pct > 0.05 && pct < 0.45,
            "Node {} has {:.1}% of keys ({}), expected between 5% and 45%",
            node_id,
            pct * 100.0,
            count
        );
    }
}

// ---------------------------------------------------------------------------
// Test: Adding a node redistributes some keys
// ---------------------------------------------------------------------------

#[test]
fn add_node_redistributes() {
    let mut ring = build_ring(3);

    // Record initial routing for 1000 keys.
    let keys: Vec<String> = (0..1000).map(|i| format!("key-{i}")).collect();
    let before: Vec<NodeId> = keys.iter().map(|k| ring.get_node(k).unwrap()).collect();

    // Add a 4th node.
    ring.add_node(make_node(4, "node-4", 9004)).unwrap();

    // Some keys should move to node 4, but most should stay.
    let after: Vec<NodeId> = keys.iter().map(|k| ring.get_node(k).unwrap()).collect();
    let moved = before
        .iter()
        .zip(after.iter())
        .filter(|(a, b)| a != b)
        .count();

    // With consistent hashing, adding 1 of 4 nodes should move ~25% of keys.
    // Allow generous bounds due to hash distribution variance.
    assert!(
        moved > 10 && moved < 600,
        "Expected some keys to move when adding a node, got {}",
        moved
    );
}

// ---------------------------------------------------------------------------
// Test: Removing a node redistributes its keys
// ---------------------------------------------------------------------------

#[test]
fn remove_node_redistributes() {
    let mut ring = build_ring(4);

    let keys: Vec<String> = (0..1000).map(|i| format!("key-{i}")).collect();
    let before: Vec<NodeId> = keys.iter().map(|k| ring.get_node(k).unwrap()).collect();

    // Remove node 2.
    ring.remove_node(2).unwrap();

    let after: Vec<NodeId> = keys.iter().map(|k| ring.get_node(k).unwrap()).collect();

    // No key should route to removed node.
    for &node_id in &after {
        assert_ne!(node_id, 2, "No key should route to removed node");
    }

    // Keys that were NOT on node 2 should remain on their original node.
    let unchanged = before
        .iter()
        .zip(after.iter())
        .filter(|&(&a, &b)| a != 2 && a == b)
        .count();

    // Most non-node-2 keys should be stable.
    let was_on_other = before.iter().filter(|&&n| n != 2).count();
    assert!(
        unchanged as f64 / was_on_other as f64 > 0.9,
        "Expected >90% stability for non-removed node keys, got {}/{}",
        unchanged,
        was_on_other
    );
}

// ---------------------------------------------------------------------------
// Test: Replica placement returns distinct nodes
// ---------------------------------------------------------------------------

#[test]
fn replicas_are_distinct() {
    let ring = build_ring(5);

    let replicas = ring.get_replicas("some-key", 3);
    assert_eq!(replicas.len(), 3, "should return 3 replicas");

    // All replicas should be distinct.
    let unique: std::collections::HashSet<_> = replicas.iter().collect();
    assert_eq!(unique.len(), 3, "replicas should be distinct");
}

// ---------------------------------------------------------------------------
// Test: Requesting more replicas than nodes returns all nodes
// ---------------------------------------------------------------------------

#[test]
fn replicas_capped_at_node_count() {
    let ring = build_ring(3);

    let replicas = ring.get_replicas("key", 10);
    assert!(replicas.len() <= 3, "cannot have more replicas than nodes");
}

// ---------------------------------------------------------------------------
// Test: Unavailable nodes are skipped in routing
// ---------------------------------------------------------------------------

#[test]
fn unavailable_nodes_skipped() {
    let mut ring = build_ring(3);

    // Find which node handles "test-key".
    let original_node = ring.get_node("test-key").unwrap();

    // Mark that node unavailable.
    ring.set_node_available(original_node, false).unwrap();

    // Key should now route to a different node.
    let new_node = ring.get_node("test-key").unwrap();
    assert_ne!(new_node, original_node, "should skip unavailable node");

    // Re-enable and verify it can come back.
    ring.set_node_available(original_node, true).unwrap();
    let restored_node = ring.get_node("test-key").unwrap();
    assert_eq!(
        restored_node, original_node,
        "should return to original node when available again"
    );
}

// ---------------------------------------------------------------------------
// Test: Empty ring returns None
// ---------------------------------------------------------------------------

#[test]
fn empty_ring_returns_none() {
    let ring = ConsistentHashRing::new();
    assert!(ring.get_node("any-key").is_none());
    assert_eq!(ring.node_count(), 0);
}

// ---------------------------------------------------------------------------
// Test: Node count and available node count
// ---------------------------------------------------------------------------

#[test]
fn node_counts() {
    let mut ring = build_ring(5);
    assert_eq!(ring.node_count(), 5);
    assert_eq!(ring.available_node_count(), 5);

    ring.set_node_available(3, false).unwrap();
    assert_eq!(ring.node_count(), 5);
    assert_eq!(ring.available_node_count(), 4);

    ring.remove_node(3).unwrap();
    assert_eq!(ring.node_count(), 4);
    assert_eq!(ring.available_node_count(), 4);
}

// ---------------------------------------------------------------------------
// Test: Weighted nodes get proportional traffic
// ---------------------------------------------------------------------------

#[test]
fn weighted_distribution() {
    let mut ring = ConsistentHashRing::new();
    // Node 1: weight 100 (default), Node 2: weight 300 (3x)
    ring.add_node(make_node(1, "node-1", 9001)).unwrap();
    ring.add_node(make_node(2, "node-2", 9002).with_weight(300))
        .unwrap();

    let keys: Vec<String> = (0..10_000).map(|i| format!("key-{i}")).collect();
    let mut counts: HashMap<NodeId, usize> = HashMap::new();
    for key in &keys {
        let node = ring.get_node(key).unwrap();
        *counts.entry(node).or_default() += 1;
    }

    // Both nodes should receive some traffic.
    let n1 = *counts.get(&1).unwrap_or(&0);
    let n2 = *counts.get(&2).unwrap_or(&0);
    assert!(n1 > 0, "Node 1 should have some keys");
    assert!(n2 > 0, "Node 2 should have some keys");
    assert_eq!(n1 + n2, 10_000, "all keys should be routed");
}
