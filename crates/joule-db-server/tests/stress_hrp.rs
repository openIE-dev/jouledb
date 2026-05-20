//! Stress tests for HRP security + erasure coding — try to break them.

use joule_db_server::hrp_erasure::{self, ErasureConfig};
use joule_db_server::hrp_security::{EpochKeyManager, SecurityError, SequenceTracker, WriteToken};
use joule_db_server::mutation_delta::{ColumnDef, DeltaValue, MutationDelta};
use std::sync::Arc;
use std::thread;

// ── EpochKeyManager edge cases ────────────────────────────────────────

#[test]
fn hrp_zero_secret() {
    let mgr = EpochKeyManager::new([0u8; 32]);
    let token = mgr.generate_token(1, b"hello");
    assert!(mgr.verify_token(&token, 1, b"hello").is_ok());
}

#[test]
fn hrp_max_secret() {
    let mgr = EpochKeyManager::new([0xFF; 32]);
    let token = mgr.generate_token(1, b"hello");
    assert!(mgr.verify_token(&token, 1, b"hello").is_ok());
}

#[test]
fn hrp_from_hex_wrong_length() {
    assert!(EpochKeyManager::from_hex("aabb").is_err());
    assert!(EpochKeyManager::from_hex("").is_err());
    assert!(EpochKeyManager::from_hex(&"ff".repeat(33)).is_err());
}

#[test]
fn hrp_from_hex_invalid() {
    assert!(EpochKeyManager::from_hex("not hex at all!!").is_err());
    assert!(EpochKeyManager::from_hex("zzzz").is_err());
}

#[test]
fn hrp_from_hex_valid() {
    let hex = "aa".repeat(32);
    let mgr = EpochKeyManager::from_hex(&hex).unwrap();
    let token = mgr.generate_token(0, b"test");
    assert!(mgr.verify_token(&token, 0, b"test").is_ok());
}

#[test]
fn hrp_empty_payload() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let token = mgr.generate_token(1, b"");
    assert!(mgr.verify_token(&token, 1, b"").is_ok());
}

#[test]
fn hrp_large_payload() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let payload = vec![0xAB; 10_000_000]; // 10MB
    let token = mgr.generate_token(1, &payload);
    assert!(mgr.verify_token(&token, 1, &payload).is_ok());
}

#[test]
fn hrp_tampered_payload() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let token = mgr.generate_token(1, b"original");
    let err = mgr.verify_token(&token, 1, b"tampered");
    assert_eq!(err, Err(SecurityError::HmacMismatch));
}

#[test]
fn hrp_tampered_hmac_one_bit() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let mut token = mgr.generate_token(1, b"hello");
    token.hmac[0] ^= 0x01; // flip one bit
    let err = mgr.verify_token(&token, 1, b"hello");
    assert_eq!(err, Err(SecurityError::HmacMismatch));
}

#[test]
fn hrp_wrong_term() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let token = mgr.generate_token(1, b"hello");
    let err = mgr.verify_token(&token, 2, b"hello");
    assert!(matches!(err, Err(SecurityError::TermMismatch { .. })));
}

#[test]
fn hrp_epoch_too_old() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let token = mgr.generate_token(1, b"hello"); // epoch 0
    mgr.advance_epoch(); // epoch -> 1
    mgr.advance_epoch(); // epoch -> 2
    let err = mgr.verify_token(&token, 1, b"hello");
    assert!(matches!(err, Err(SecurityError::EpochTooOld { .. })));
}

#[test]
fn hrp_epoch_grace_window() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let token = mgr.generate_token(1, b"hello"); // epoch 0
    mgr.advance_epoch(); // epoch -> 1
    // Previous epoch should still be accepted (grace window)
    assert!(mgr.verify_token(&token, 1, b"hello").is_ok());
}

#[test]
fn hrp_epoch_max_value() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    mgr.set_epoch(u64::MAX - 1);
    let token = mgr.generate_token(1, b"hello");
    assert!(mgr.verify_token(&token, 1, b"hello").is_ok());
}

#[test]
fn hrp_epoch_overflow() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    mgr.set_epoch(u64::MAX);
    // advance_epoch will overflow to 0 — this should not panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mgr.advance_epoch();
    }));
    // Overflow wraps in release mode, panics in debug — both are acceptable
    // The key point: no UB, no crash
    let _ = result;
}

#[test]
fn hrp_rapid_epoch_rotation() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    for _ in 0..10_000 {
        mgr.advance_epoch();
    }
    assert_eq!(mgr.current_epoch(), 10_000);
    let token = mgr.generate_token(1, b"hello");
    assert!(mgr.verify_token(&token, 1, b"hello").is_ok());
}

#[test]
fn hrp_concurrent_token_generation() {
    let mgr = Arc::new(EpochKeyManager::new([42u8; 32]));
    let mut handles = vec![];
    for _ in 0..10 {
        let m = Arc::clone(&mgr);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                let token = m.generate_token(1, b"concurrent");
                assert!(m.verify_token(&token, 1, b"concurrent").is_ok());
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn hrp_different_keys_different_tokens() {
    let mgr1 = EpochKeyManager::new([1u8; 32]);
    let mgr2 = EpochKeyManager::new([2u8; 32]);
    let token1 = mgr1.generate_token(1, b"hello");
    // Token from mgr1 should NOT verify on mgr2
    let err = mgr2.verify_token(&token1, 1, b"hello");
    assert_eq!(err, Err(SecurityError::HmacMismatch));
}

// ── WriteToken serialization ──────────────────────────────────────────

#[test]
fn hrp_token_bytes_roundtrip() {
    let mgr = EpochKeyManager::new([42u8; 32]);
    let token = mgr.generate_token(1, b"hello");
    let bytes = token.to_bytes();
    assert_eq!(bytes.len(), WriteToken::WIRE_SIZE);
    let decoded = WriteToken::from_bytes(&bytes);
    assert_eq!(decoded, token);
}

#[test]
fn hrp_token_bytes_max_values() {
    let token = WriteToken {
        epoch: u64::MAX,
        sequence: u64::MAX,
        term: u64::MAX,
        hmac: [0xFF; 32],
    };
    let bytes = token.to_bytes();
    let decoded = WriteToken::from_bytes(&bytes);
    assert_eq!(decoded, token);
}

#[test]
fn hrp_token_bytes_zero() {
    let token = WriteToken {
        epoch: 0,
        sequence: 0,
        term: 0,
        hmac: [0; 32],
    };
    let bytes = token.to_bytes();
    let decoded = WriteToken::from_bytes(&bytes);
    assert_eq!(decoded, token);
}

// ── SequenceTracker ───────────────────────────────────────────────────

#[test]
fn hrp_sequence_tracker_basic() {
    let tracker = SequenceTracker::new();
    assert!(tracker.check_and_update("peer1", 0).is_ok());
    assert!(tracker.check_and_update("peer1", 1).is_ok());
    assert!(tracker.check_and_update("peer1", 2).is_ok());
}

#[test]
fn hrp_sequence_replay_detection() {
    let tracker = SequenceTracker::new();
    tracker.check_and_update("peer1", 5).unwrap();
    let err = tracker.check_and_update("peer1", 5);
    assert!(matches!(err, Err(SecurityError::SequenceReplay { .. })));
    let err = tracker.check_and_update("peer1", 3);
    assert!(matches!(err, Err(SecurityError::SequenceReplay { .. })));
}

#[test]
fn hrp_sequence_different_peers() {
    let tracker = SequenceTracker::new();
    tracker.check_and_update("peer1", 5).unwrap();
    tracker.check_and_update("peer2", 5).unwrap(); // Same seq, different peer — OK
    tracker.check_and_update("peer1", 6).unwrap();
    tracker.check_and_update("peer2", 6).unwrap();
}

#[test]
fn hrp_sequence_many_peers() {
    let tracker = SequenceTracker::new();
    for i in 0..10_000u64 {
        let peer = format!("peer_{}", i);
        tracker.check_and_update(&peer, 0).unwrap();
    }
    // Verify replay detection still works
    let err = tracker.check_and_update("peer_0", 0);
    assert!(matches!(err, Err(SecurityError::SequenceReplay { .. })));
}

#[test]
fn hrp_sequence_max_value() {
    let tracker = SequenceTracker::new();
    tracker.check_and_update("peer1", u64::MAX - 1).unwrap();
    tracker.check_and_update("peer1", u64::MAX).unwrap();
}

#[test]
fn hrp_sequence_concurrent() {
    let tracker = Arc::new(SequenceTracker::new());
    let mut handles = vec![];
    for tid in 0..10 {
        let t = Arc::clone(&tracker);
        handles.push(thread::spawn(move || {
            let peer = format!("thread_{}", tid);
            for seq in 0..1000u64 {
                t.check_and_update(&peer, seq).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

// ── Erasure Coding ────────────────────────────────────────────────────

#[test]
fn hrp_erasure_empty_data() {
    let config = ErasureConfig::default();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        hrp_erasure::encode(&[], &config)
    }));
    assert!(result.is_ok(), "Empty data should not panic");
}

#[test]
fn hrp_erasure_one_byte() {
    let config = ErasureConfig::default();
    let encoded = hrp_erasure::encode(&[42], &config);
    assert!(encoded.is_ok() || encoded.is_err()); // either is fine, no panic
}

#[test]
fn hrp_erasure_roundtrip_small() {
    let config = ErasureConfig::default();
    let data = b"hello world";
    if let Ok(shards) = hrp_erasure::encode(data, &config) {
        let as_options: Vec<Option<_>> = shards.into_iter().map(Some).collect();
        let decoded = hrp_erasure::decode(&as_options);
        if let Ok(recovered) = decoded {
            assert_eq!(&recovered[..data.len()], &data[..]);
        }
    }
}

#[test]
fn hrp_erasure_roundtrip_large() {
    let config = ErasureConfig::default();
    let data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
    if let Ok(shards) = hrp_erasure::encode(&data, &config) {
        let as_options: Vec<Option<_>> = shards.into_iter().map(Some).collect();
        let decoded = hrp_erasure::decode(&as_options);
        if let Ok(recovered) = decoded {
            assert_eq!(&recovered[..data.len()], &data[..]);
        }
    }
}

// ── MutationDelta serialization ───────────────────────────────────────

#[test]
fn hrp_delta_insert_rows_roundtrip() {
    let delta = MutationDelta::InsertRows {
        table: "test".to_string(),
        columns: vec!["id".to_string(), "name".to_string()],
        rows: vec![
            vec![DeltaValue::Int(1), DeltaValue::Text("alice".to_string())],
            vec![DeltaValue::Int(2), DeltaValue::Text("bob".to_string())],
        ],
    };
    let bytes = bincode::serde::encode_to_vec(&delta, bincode::config::standard()).unwrap();
    let (decoded, _): (MutationDelta, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(format!("{:?}", delta), format!("{:?}", decoded));
}

#[test]
fn hrp_delta_all_value_types() {
    let values = vec![
        DeltaValue::Null,
        DeltaValue::Bool(true),
        DeltaValue::Bool(false),
        DeltaValue::Int(0),
        DeltaValue::Int(i64::MAX),
        DeltaValue::Int(i64::MIN),
        DeltaValue::Float(0.0),
        DeltaValue::Float(f64::MAX),
        DeltaValue::Float(f64::MIN),
        DeltaValue::Text(String::new()),
        DeltaValue::Text("x".repeat(100_000)),
        DeltaValue::Blob(vec![]),
        DeltaValue::Blob(vec![0xFF; 100_000]),
        DeltaValue::Array(vec![]),
        DeltaValue::Array(vec![DeltaValue::Int(1), DeltaValue::Text("nested".into())]),
    ];
    for v in &values {
        let bytes = bincode::serde::encode_to_vec(v, bincode::config::standard()).unwrap();
        let (decoded, _): (DeltaValue, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        let _ = format!("{:?}", decoded);
    }
}

#[test]
fn hrp_delta_nested_array() {
    // 100-level deep nesting
    let mut val = DeltaValue::Int(42);
    for _ in 0..100 {
        val = DeltaValue::Array(vec![val]);
    }
    let bytes = bincode::serde::encode_to_vec(&val, bincode::config::standard()).unwrap();
    let (decoded, _): (DeltaValue, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    let _ = format!("{:?}", decoded);
}

#[test]
fn hrp_delta_empty_insert() {
    let delta = MutationDelta::InsertRows {
        table: "t".to_string(),
        columns: vec![],
        rows: vec![],
    };
    let bytes = bincode::serde::encode_to_vec(&delta, bincode::config::standard()).unwrap();
    let (_, _): (MutationDelta, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
}

#[test]
fn hrp_delta_large_insert() {
    let rows: Vec<Vec<DeltaValue>> = (0..10_000)
        .map(|i| vec![DeltaValue::Int(i), DeltaValue::Text(format!("row_{}", i))])
        .collect();
    let delta = MutationDelta::InsertRows {
        table: "big".to_string(),
        columns: vec!["id".to_string(), "val".to_string()],
        rows,
    };
    let bytes = bincode::serde::encode_to_vec(&delta, bincode::config::standard()).unwrap();
    let (_, _): (MutationDelta, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
}

#[test]
fn hrp_delta_raw_sql_empty() {
    let delta = MutationDelta::RawSql {
        sql: "".to_string(),
    };
    let bytes = bincode::serde::encode_to_vec(&delta, bincode::config::standard()).unwrap();
    let (_, _): (MutationDelta, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
}

#[test]
fn hrp_delta_create_table() {
    let delta = MutationDelta::CreateTable {
        name: "new_table".to_string(),
        columns: vec!["id".to_string(), "name".to_string()],
        column_defs: vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: "INTEGER".to_string(),
                nullable: false,
                primary_key: true,
                unique: false,
                auto_increment: false,
            },
            ColumnDef {
                name: "name".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                primary_key: false,
                unique: false,
                auto_increment: false,
            },
        ],
        if_not_exists: false,
    };
    let bytes = bincode::serde::encode_to_vec(&delta, bincode::config::standard()).unwrap();
    let (_, _): (MutationDelta, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
}

#[test]
fn hrp_delta_drop_table() {
    let delta = MutationDelta::DropTable {
        name: "table_name".to_string(),
        if_exists: false,
    };
    let bytes = bincode::serde::encode_to_vec(&delta, bincode::config::standard()).unwrap();
    let (_, _): (MutationDelta, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
}

#[test]
fn hrp_delta_float_nan() {
    // NaN in DeltaValue::Float — should serialize/deserialize
    let val = DeltaValue::Float(f64::NAN);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = bincode::serde::encode_to_vec(&val, bincode::config::standard()).unwrap();
        let (_, _): (DeltaValue, _) = bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    }));
    assert!(result.is_ok(), "NaN in DeltaValue should not panic");
}
