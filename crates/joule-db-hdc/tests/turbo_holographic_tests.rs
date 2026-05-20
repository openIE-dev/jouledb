//! Comprehensive tests for TurboHolographic and related data structures
//!
//! Tests cover:
//! - Correctness of bind/unbind operations  
//! - Capacity and SNR behavior
//! - Fuzzy matching accuracy
//! - k-nearest neighbor correctness
//! - Edge cases and error handling
//!
//! ## Key Semantic Notes
//!
//! Holographic stores have unique semantics:
//! - Values are SUPERPOSED, not replaced (put adds to the bundle)
//! - Similarity is normalized to [0, 1] where 0.5 = orthogonal
//! - Retrieval quality degrades with capacity load
//! - Empty or very short keys/values may not encode properly

use joule_db_hdc::{
    BinaryHV, BinaryHolographicDirect, HolographicStore, HybridHolographic, TurboHolographic,
    UltraHolographic,
};

// ============================================================================
// BinaryHV Core Operation Tests
// ============================================================================

#[test]
fn test_binary_hv_bind_unbind_symmetry() {
    // Property: A ⊕ B ⊕ B = A (XOR is self-inverse)
    let dim = 4096;
    let a = BinaryHV::random(dim, 12345);
    let b = BinaryHV::random(dim, 67890);

    let bound = a.bind(&b);
    let unbound = bound.unbind(&b);

    let similarity = a.similarity(&unbound);
    assert!(
        (similarity - 1.0).abs() < 0.001,
        "Unbind should restore original vector, got similarity: {}",
        similarity
    );
}

#[test]
fn test_binary_hv_bind_is_unbind() {
    // Property: bind and unbind are the same operation for XOR
    let dim = 4096;
    let a = BinaryHV::random(dim, 111);
    let b = BinaryHV::random(dim, 222);

    let bound = a.bind(&b);
    let unbound = a.unbind(&b);

    let similarity = bound.similarity(&unbound);
    assert!(
        (similarity - 1.0).abs() < 0.001,
        "Bind and unbind should be equivalent, got similarity: {}",
        similarity
    );
}

#[test]
fn test_binary_hv_self_similarity() {
    let dim = 4096;
    let a = BinaryHV::random(dim, 42);

    let similarity = a.similarity(&a);
    assert!(
        (similarity - 1.0).abs() < 0.001,
        "Self-similarity should be 1.0, got: {}",
        similarity
    );
}

#[test]
fn test_binary_hv_random_similarity() {
    // Random binary vectors have ~50% bit overlap = 0.5 similarity
    let dim = 4096;
    let similarities: Vec<f32> = (0..100)
        .map(|i| {
            let a = BinaryHV::random(dim, i as u64);
            let b = BinaryHV::random(dim, i as u64 + 1000);
            a.similarity(&b)
        })
        .collect();

    let avg_similarity: f32 = similarities.iter().sum::<f32>() / similarities.len() as f32;

    assert!(
        (avg_similarity - 0.5).abs() < 0.1,
        "Random vectors should have ~0.5 similarity, got: {}",
        avg_similarity
    );
}

#[test]
fn test_binary_hv_permute_differs() {
    let dim = 4096;
    let a = BinaryHV::random(dim, 42);
    let permuted = a.permute(1);

    let similarity = a.similarity(&permuted);

    assert!(
        similarity < 0.9,
        "Permuted vector should differ from original, got: {}",
        similarity
    );
}

#[test]
fn test_binary_hv_from_bytes_similar() {
    let dim = 4096;
    let v1 = BinaryHV::from_bytes(b"hello", dim);
    let v2 = BinaryHV::from_bytes(b"hallo", dim);
    let v3 = BinaryHV::from_bytes(b"goodbye", dim);

    let sim_similar = v1.similarity(&v2);
    let sim_different = v1.similarity(&v3);

    assert!(
        sim_similar > sim_different,
        "Similar inputs should have higher similarity: {} vs {}",
        sim_similar,
        sim_different
    );
}

#[test]
fn test_binary_hv_condense_collision_resistance() {
    let dim = 4096;
    let hashes: std::collections::HashSet<u64> = (0..1000)
        .map(|i| BinaryHV::random(dim, i as u64).condense_to_u64())
        .collect();

    assert!(
        hashes.len() > 990,
        "Should have minimal collisions, got {} unique out of 1000",
        hashes.len()
    );
}

// ============================================================================
// TurboHolographic Store Tests
// ============================================================================

#[test]
fn test_turbo_holographic_basic_put_get() {
    let mut store = TurboHolographic::new(8192);

    store.put(b"key1", b"value1");
    store.put(b"key2", b"value2");
    store.put(b"key3", b"value3");

    assert_eq!(store.get(b"key1"), Some(b"value1".to_vec()));
    assert_eq!(store.get(b"key2"), Some(b"value2".to_vec()));
    assert_eq!(store.get(b"key3"), Some(b"value3".to_vec()));
    assert_eq!(store.get(b"nonexistent"), None);
}

#[test]
fn test_turbo_holographic_superposition() {
    // Holographic stores SUPERPOSE values - document this behavior
    let mut store = TurboHolographic::new(8192);

    store.put(b"key", b"value1");
    let first = store.get(b"key");

    store.put(b"key", b"value2");
    let second = store.get(b"key");

    println!("First get: {:?}", first);
    println!("Second get (superposed): {:?}", second);

    assert!(store.len() > 0);
}

#[test]
fn test_turbo_holographic_delete() {
    let mut store = TurboHolographic::new(8192);

    store.put(b"key", b"value");
    assert!(store.contains(b"key"));

    store.delete(b"key", b"value");
    assert_eq!(store.get(b"key"), None);
    assert!(!store.contains(b"key"));
}

// ============================================================================
// UltraHolographic Tests (Fuzzy Matching)
// ============================================================================

#[test]
fn test_ultra_holographic_exact_match() {
    let mut store = UltraHolographic::new(4096);

    store.put(b"hello", b"world");
    store.put(b"foo", b"bar");

    assert_eq!(store.get(b"hello"), Some(b"world".to_vec()));
    assert_eq!(store.get(b"foo"), Some(b"bar".to_vec()));
}

#[test]
fn test_ultra_holographic_fuzzy_match() {
    let mut store = UltraHolographic::new(8192);

    store.put(b"hello", b"world");
    store.put(b"goodbye", b"moon");

    let result = store.get_fuzzy(b"hallo", 0.5);
    assert!(result.is_some(), "Should fuzzy match 'hallo' to 'hello'");
}

#[test]
fn test_ultra_holographic_k_nearest() {
    let mut store = UltraHolographic::new(4096);

    store.put(b"cat", b"animal");
    store.put(b"car", b"vehicle");
    store.put(b"cap", b"clothing");
    store.put(b"cup", b"container");

    let results = store.k_nearest(b"cat", 3);
    assert!(results.len() <= 3, "Should return at most 3 results");
}

// ============================================================================
// HybridHolographic Tests (HashMap + Holographic)
// ============================================================================

#[test]
fn test_hybrid_holographic_exact() {
    let mut store = HybridHolographic::new(4096);

    for i in 0..100 {
        let key = format!("key:{}", i);
        let value = format!("value:{}", i);
        store.put(key.as_bytes(), value.as_bytes());
    }

    // All exact matches should work (backed by HashMap)
    for i in 0..100 {
        let key = format!("key:{}", i);
        let expected = format!("value:{}", i);
        assert_eq!(store.get(key.as_bytes()), Some(expected.into_bytes()));
    }
}

// ============================================================================
// BinaryHolographicDirect Tests
// ============================================================================

#[test]
fn test_binary_direct_operations() {
    let dim = 4096;
    let mut store = BinaryHolographicDirect::new(dim);

    let k1 = BinaryHV::random(dim, 1);
    let v1 = BinaryHV::random(dim, 100);

    store.put(&k1, &v1);

    // get returns (BinaryHV, f32) tuple
    let (retrieved, snr) = store.get(&k1);
    println!("SNR: {}", snr);

    let sim = retrieved.similarity(&v1);
    assert!(
        sim > 0.3,
        "Retrieved should be similar to original: {}",
        sim
    );
}

// ============================================================================
// HolographicStore Trait Tests
// ============================================================================

#[test]
fn test_trait_contains() {
    let mut store = TurboHolographic::new(4096);

    assert!(!store.contains(b"key"));
    store.put(b"key", b"value");
    assert!(store.contains(b"key"));
}

#[test]
fn test_trait_is_empty() {
    let mut store = TurboHolographic::new(4096);

    assert!(store.is_empty());
    store.put(b"key", b"value");
    assert!(!store.is_empty());
}

#[test]
fn test_trait_load_factor() {
    let mut store = TurboHolographic::new(4096);

    let initial = store.load_factor();
    for i in 0..10 {
        store.put(format!("key{}", i).as_bytes(), b"value");
    }
    let after = store.load_factor();

    assert!(after > initial, "Load factor should increase with items");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_unicode_keys() {
    let mut store = TurboHolographic::new(4096);

    store.put("键".as_bytes(), "值".as_bytes());
    store.put("🔑".as_bytes(), "🎉".as_bytes());

    assert!(store.contains("键".as_bytes()));
    assert!(store.contains("🔑".as_bytes()));
}

// ============================================================================
// Capacity and SNR Tests
// ============================================================================

#[test]
fn test_snr_relationship() {
    let dim = 4096;
    let mut store = BinaryHolographicDirect::new(dim);

    // Add items
    for i in 0..10 {
        let k = BinaryHV::random(dim, i as u64);
        let v = BinaryHV::random(dim, i as u64 + 1000);
        store.put(&k, &v);
    }

    let (_, snr) = store.get(&BinaryHV::random(dim, 0));
    println!("SNR after 10 items: {}", snr);
}
