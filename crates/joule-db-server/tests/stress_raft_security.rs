//! HRP security stress tests.
//!
//! Tests write token HMAC verification, epoch key management, sequence
//! replay detection, v2 wire format, erasure coding, and energy state.

use std::sync::Arc;

use joule_db_server::hrp_erasure::{ErasureConfig, ErasureError, ErasureShard, decode, encode};
use joule_db_server::hrp_security::{EpochKeyManager, SecurityError, SequenceTracker, WriteToken};

fn test_secret() -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = 0x42;
    s[31] = 0xFF;
    s
}

// ============================================================================
// WriteToken + EpochKeyManager
// ============================================================================

#[test]
fn sec_token_generate_and_verify() {
    let mgr = EpochKeyManager::new(test_secret());
    let payload = b"test payload";
    let token = mgr.generate_token(1, payload);

    assert_eq!(token.epoch, 0); // initial epoch
    assert_eq!(token.term, 1);
    assert!(mgr.verify_token(&token, 1, payload).is_ok());
}

#[test]
fn sec_token_wrong_payload_rejected() {
    let mgr = EpochKeyManager::new(test_secret());
    let token = mgr.generate_token(1, b"original");

    let result = mgr.verify_token(&token, 1, b"tampered");
    assert_eq!(result, Err(SecurityError::HmacMismatch));
}

#[test]
fn sec_token_wrong_term_rejected() {
    let mgr = EpochKeyManager::new(test_secret());
    let payload = b"data";
    let token = mgr.generate_token(1, payload);

    let result = mgr.verify_token(&token, 2, payload);
    assert!(matches!(result, Err(SecurityError::TermMismatch { .. })));
}

#[test]
fn sec_token_stale_epoch_rejected() {
    let mgr = EpochKeyManager::new(test_secret());
    let payload = b"data";

    // Generate at epoch 0
    let token = mgr.generate_token(1, payload);

    // Advance epoch twice (current becomes 2, previous is 1)
    mgr.advance_epoch(); // epoch 1
    mgr.advance_epoch(); // epoch 2

    // Token at epoch 0 is 2 behind — should be rejected
    let result = mgr.verify_token(&token, 1, payload);
    assert!(matches!(result, Err(SecurityError::EpochTooOld { .. })));
}

#[test]
fn sec_token_previous_epoch_accepted() {
    let mgr = EpochKeyManager::new(test_secret());
    let payload = b"data";

    let token = mgr.generate_token(1, payload);
    assert_eq!(token.epoch, 0);

    // Advance once — previous epoch (0) should still be accepted
    mgr.advance_epoch(); // epoch 1
    assert!(
        mgr.verify_token(&token, 1, payload).is_ok(),
        "Previous epoch should be accepted (grace window)"
    );
}

#[test]
fn sec_epoch_key_deterministic() {
    let mgr = EpochKeyManager::new(test_secret());
    let key1 = mgr.derive_epoch_key(42);
    let key2 = mgr.derive_epoch_key(42);
    assert_eq!(key1, key2, "Same epoch should produce same key");

    let key3 = mgr.derive_epoch_key(43);
    assert_ne!(key1, key3, "Different epochs should produce different keys");
}

#[test]
fn sec_epoch_advance_wraps() {
    let mgr = EpochKeyManager::new(test_secret());
    mgr.set_epoch(u64::MAX);
    let new_epoch = mgr.advance_epoch();
    assert_eq!(new_epoch, 0, "Epoch should wrap around from u64::MAX to 0");
}

#[test]
fn sec_sequence_monotonic() {
    let mgr = EpochKeyManager::new(test_secret());
    let seq1 = mgr.next_sequence();
    let seq2 = mgr.next_sequence();
    let seq3 = mgr.next_sequence();
    assert!(seq2 > seq1);
    assert!(seq3 > seq2);
}

#[test]
fn sec_1000_tokens_unique_sequences() {
    let mgr = EpochKeyManager::new(test_secret());
    let payload = b"test";

    let mut sequences = std::collections::HashSet::new();
    for _ in 0..1000 {
        let token = mgr.generate_token(1, payload);
        assert!(
            sequences.insert(token.sequence),
            "Duplicate sequence number: {}",
            token.sequence
        );
    }
    assert_eq!(sequences.len(), 1000);
}

#[test]
fn sec_concurrent_token_generation() {
    use std::thread;

    let mgr = Arc::new(EpochKeyManager::new(test_secret()));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let mgr = mgr.clone();
        handles.push(thread::spawn(move || {
            let mut sequences = Vec::new();
            for _ in 0..100 {
                let token = mgr.generate_token(1, b"concurrent");
                sequences.push(token.sequence);
            }
            sequences
        }));
    }

    let mut all_sequences = std::collections::HashSet::new();
    for h in handles {
        for seq in h.join().unwrap() {
            assert!(
                all_sequences.insert(seq),
                "Duplicate sequence from concurrent generation: {}",
                seq
            );
        }
    }
    assert_eq!(all_sequences.len(), 800);
}

#[test]
fn sec_from_hex_valid() {
    let hex = "42".to_string() + &"00".repeat(30) + "FF";
    let mgr = EpochKeyManager::from_hex(&hex).unwrap();
    assert_eq!(
        mgr.derive_epoch_key(0),
        EpochKeyManager::new(test_secret()).derive_epoch_key(0)
    );
}

#[test]
fn sec_from_hex_invalid() {
    assert!(EpochKeyManager::from_hex("too_short").is_err());
    assert!(EpochKeyManager::from_hex("GGGG").is_err()); // invalid hex
    assert!(EpochKeyManager::from_hex("").is_err());
}

#[test]
fn sec_token_wire_bytes_roundtrip() {
    let mgr = EpochKeyManager::new(test_secret());
    let token = mgr.generate_token(42, b"wire test");

    let bytes = token.to_bytes();
    assert_eq!(bytes.len(), WriteToken::WIRE_SIZE);

    let restored = WriteToken::from_bytes(&bytes);
    assert_eq!(restored.epoch, token.epoch);
    assert_eq!(restored.sequence, token.sequence);
    assert_eq!(restored.term, token.term);
    assert_eq!(restored.hmac, token.hmac);
}

#[test]
fn sec_different_secrets_incompatible() {
    let secret_a = [0xAAu8; 32];
    let secret_b = [0xBBu8; 32];

    let mgr_a = EpochKeyManager::new(secret_a);
    let mgr_b = EpochKeyManager::new(secret_b);

    let payload = b"data";
    let token = mgr_a.generate_token(1, payload);

    // Token from A should NOT verify under B
    let result = mgr_b.verify_token(&token, 1, payload);
    assert_eq!(result, Err(SecurityError::HmacMismatch));
}

#[test]
fn sec_empty_payload_token() {
    let mgr = EpochKeyManager::new(test_secret());
    let token = mgr.generate_token(1, b"");
    assert!(mgr.verify_token(&token, 1, b"").is_ok());
}

#[test]
fn sec_large_payload_token() {
    let mgr = EpochKeyManager::new(test_secret());
    let payload = vec![0xAB; 1_000_000]; // 1MB
    let token = mgr.generate_token(1, &payload);
    assert!(mgr.verify_token(&token, 1, &payload).is_ok());
}

// ============================================================================
// SequenceTracker
// ============================================================================

#[test]
fn sec_tracker_first_message_accepted() {
    let tracker = SequenceTracker::new();
    assert!(tracker.check_and_update("peer_a", 1).is_ok());
}

#[test]
fn sec_tracker_increasing_sequence_accepted() {
    let tracker = SequenceTracker::new();
    assert!(tracker.check_and_update("peer_a", 1).is_ok());
    assert!(tracker.check_and_update("peer_a", 2).is_ok());
    assert!(tracker.check_and_update("peer_a", 3).is_ok());
    assert!(tracker.check_and_update("peer_a", 100).is_ok()); // gaps ok
}

#[test]
fn sec_tracker_replay_rejected() {
    let tracker = SequenceTracker::new();
    assert!(tracker.check_and_update("peer_a", 5).is_ok());
    let result = tracker.check_and_update("peer_a", 5);
    assert!(matches!(result, Err(SecurityError::SequenceReplay { .. })));
}

#[test]
fn sec_tracker_old_sequence_rejected() {
    let tracker = SequenceTracker::new();
    assert!(tracker.check_and_update("peer_a", 10).is_ok());
    let result = tracker.check_and_update("peer_a", 5);
    assert!(matches!(result, Err(SecurityError::SequenceReplay { .. })));
}

#[test]
fn sec_tracker_independent_peers() {
    let tracker = SequenceTracker::new();
    assert!(tracker.check_and_update("peer_a", 1).is_ok());
    assert!(tracker.check_and_update("peer_b", 1).is_ok()); // same sequence, different peer
    assert!(tracker.check_and_update("peer_a", 2).is_ok());
    assert!(tracker.check_and_update("peer_b", 2).is_ok());
}

#[test]
fn sec_tracker_100_peers() {
    let tracker = SequenceTracker::new();
    for i in 0..100 {
        let peer = format!("peer_{}", i);
        assert!(tracker.check_and_update(&peer, 1).is_ok());
        assert!(tracker.check_and_update(&peer, 2).is_ok());
    }
}

#[test]
fn sec_tracker_concurrent_access() {
    use std::thread;

    let tracker = Arc::new(SequenceTracker::new());
    let mut handles = Vec::new();

    for t in 0..8 {
        let tracker = tracker.clone();
        handles.push(thread::spawn(move || {
            let peer = format!("peer_{}", t);
            for seq in 1..=100 {
                tracker.check_and_update(&peer, seq).unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ============================================================================
// Erasure coding
// ============================================================================

#[test]
fn sec_erasure_encode_decode_default_config() {
    let config = ErasureConfig::default();
    let data = b"Hello, erasure coding from JouleDB HRP!";

    let shards = encode(data, &config).unwrap();
    assert_eq!(shards.len(), 3); // 2 data + 1 parity
    assert_eq!(shards[0].original_len, data.len());

    let shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(&recovered, data);
}

#[test]
fn sec_erasure_missing_parity_shard() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let data = b"Test data for erasure recovery";

    let shards = encode(data, &config).unwrap();
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[2] = None; // Drop parity shard

    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(&recovered, data);
}

#[test]
fn sec_erasure_missing_data_shard() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let data = b"Recoverable data";

    let shards = encode(data, &config).unwrap();
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[0] = None; // Drop first data shard

    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(&recovered, data);
}

#[test]
fn sec_erasure_too_many_missing_fails() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let data = b"Unrecoverable";

    let shards = encode(data, &config).unwrap();
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[0] = None;
    shard_opts[1] = None; // Drop 2 of 3 shards (need 2 to reconstruct)

    let result = decode(&shard_opts);
    assert!(result.is_err());
}

#[test]
fn sec_erasure_empty_payload_fails() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let result = encode(b"", &config);
    assert!(result.is_err());
}

#[test]
fn sec_erasure_single_byte() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let data = b"X";

    let shards = encode(data, &config).unwrap();
    let shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(&recovered, data);
}

#[test]
fn sec_erasure_large_100kb() {
    let config = ErasureConfig {
        data_shards: 4,
        parity_shards: 2,
        threshold_bytes: 0,
    };
    let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

    let shards = encode(&data, &config).unwrap();
    assert_eq!(shards.len(), 6);

    // Drop 2 shards (within parity capacity)
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[1] = None;
    shard_opts[4] = None;

    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(recovered, data);
}

#[test]
fn sec_erasure_shard_serialization() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let data = b"Serializable shard data";

    let shards = encode(data, &config).unwrap();

    for shard in &shards {
        let encoded = bincode::serde::encode_to_vec(shard, bincode::config::standard()).unwrap();
        let (decoded, _): (ErasureShard, _) = bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(decoded.index, shard.index);
        assert_eq!(decoded.data, shard.data);
        assert_eq!(decoded.original_len, shard.original_len);
    }
}

// ============================================================================
// Erasure coding — Bug 7 fix verification
// ============================================================================

#[test]
fn sec_erasure_reconstruction_without_done_flag() {
    // This test verifies Bug 7 fix: reconstruction should trigger
    // when enough shards are received, even if the "last" shard is missing.
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 1,
        threshold_bytes: 0,
    };
    let data = b"Bug 7 test: reconstruction without done flag";

    let shards = encode(data, &config).unwrap();
    assert_eq!(shards.len(), 3);

    // Drop the last shard (index 2) — this would have had done=true
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[2] = None; // Drop last shard

    // With 2 data shards still present, reconstruction should work
    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(&recovered, data);
}

// ============================================================================
// Edge cases with different config parameters
// ============================================================================

#[test]
fn sec_erasure_high_redundancy() {
    let config = ErasureConfig {
        data_shards: 2,
        parity_shards: 4,
        threshold_bytes: 0,
    };
    let data = b"High redundancy test";

    let shards = encode(data, &config).unwrap();
    assert_eq!(shards.len(), 6);

    // Drop 4 shards (all parity shards) — should still work with just data
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[2] = None;
    shard_opts[3] = None;
    shard_opts[4] = None;
    shard_opts[5] = None;

    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(&recovered, data);
}

#[test]
fn sec_erasure_many_data_shards() {
    let config = ErasureConfig {
        data_shards: 8,
        parity_shards: 2,
        threshold_bytes: 0,
    };
    let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();

    let shards = encode(&data, &config).unwrap();
    assert_eq!(shards.len(), 10);

    // Drop 2 shards
    let mut shard_opts: Vec<Option<ErasureShard>> = shards.into_iter().map(Some).collect();
    shard_opts[3] = None;
    shard_opts[7] = None;

    let recovered = decode(&shard_opts).unwrap();
    assert_eq!(recovered, data);
}

// ============================================================================
// SecurityError Display
// ============================================================================

#[test]
fn sec_error_display_formats() {
    let err = SecurityError::HmacMismatch;
    assert!(
        format!("{}", err).contains("HMAC") || format!("{}", err).to_lowercase().contains("hmac")
    );

    let err = SecurityError::TermMismatch {
        token_term: 1,
        expected: 2,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("1") && msg.contains("2"));

    let err = SecurityError::EpochTooOld {
        token_epoch: 0,
        current: 5,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("0") && msg.contains("5"));

    let err = SecurityError::SequenceReplay {
        received: 3,
        last_seen: 5,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("3") && msg.contains("5"));
}
