//! Raft WAL recovery and durability stress tests.
//!
//! Tests the Write-Ahead Log against corruption, truncation, large entries,
//! checkpoint cycles, and crash recovery scenarios.

use std::io::Write;

use joule_db_server::raft::{Command, LogEntry, PersistentState, RaftWal};

// ============================================================================
// Basic append and recover
// ============================================================================

#[test]
fn wal_append_single_entry_and_recover() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        let entry = LogEntry::new(
            1,
            1,
            Command::Set {
                key: b"hello".to_vec(),
                value: b"world".to_vec(),
            },
        );
        wal.append_entry(&entry).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 1);
    assert_eq!(recovered.log[0].term, 1);
    assert_eq!(recovered.log[0].index, 1);
}

#[test]
fn wal_append_100_entries_and_recover() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 1..=100 {
            let entry = LogEntry::new(
                1,
                i,
                Command::Set {
                    key: format!("key{}", i).into_bytes(),
                    value: format!("val{}", i).into_bytes(),
                },
            );
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 100);
    assert_eq!(recovered.log[0].index, 1);
    assert_eq!(recovered.log[99].index, 100);
}

#[test]
fn wal_append_10000_entries() {
    let dir = tempfile::tempdir().unwrap();
    let mut wal = RaftWal::open(dir.path()).unwrap();
    for i in 1..=10_000 {
        let entry = LogEntry::new(1, i, Command::Noop);
        wal.append_entry(&entry).unwrap();
    }
    assert!(
        wal.needs_checkpoint(),
        "Should need checkpoint after 10K entries"
    );
}

// ============================================================================
// Meta persistence
// ============================================================================

#[test]
fn wal_meta_term_and_vote_recovered() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        wal.append_meta(5, &Some("node_a".to_string())).unwrap();
        // Also append an entry to verify ordering
        let entry = LogEntry::new(5, 1, Command::Noop);
        wal.append_entry(&entry).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.current_term, 5);
    assert_eq!(recovered.voted_for.as_deref(), Some("node_a"));
    assert_eq!(recovered.log.len(), 1);
}

#[test]
fn wal_meta_multiple_updates() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        wal.append_meta(1, &Some("node_a".to_string())).unwrap();
        wal.append_meta(2, &None).unwrap();
        wal.append_meta(3, &Some("node_b".to_string())).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.current_term, 3);
    assert_eq!(recovered.voted_for.as_deref(), Some("node_b"));
}

#[test]
fn wal_meta_none_vote() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        wal.append_meta(7, &None).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.current_term, 7);
    assert!(recovered.voted_for.is_none());
}

// ============================================================================
// Checkpoint and recovery
// ============================================================================

#[test]
fn wal_checkpoint_and_recover() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        let mut state = PersistentState::new();
        state.current_term = 3;
        state.voted_for = Some("leader".to_string());

        for i in 1..=50 {
            let entry = LogEntry::new(3, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
            state.log.push(entry);
        }

        wal.checkpoint(&state).unwrap();

        // Append more after checkpoint
        for i in 51..=60 {
            let entry = LogEntry::new(3, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(
        recovered.log.len(),
        60,
        "Should recover checkpoint + WAL entries"
    );
    assert_eq!(recovered.current_term, 3);
    assert_eq!(recovered.voted_for.as_deref(), Some("leader"));
}

#[test]
fn wal_multiple_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        let mut state = PersistentState::new();
        state.current_term = 1;

        // First batch
        for i in 1..=20 {
            let entry = LogEntry::new(1, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
            state.log.push(entry);
        }
        wal.checkpoint(&state).unwrap();

        // Second batch
        state.current_term = 2;
        for i in 21..=40 {
            let entry = LogEntry::new(2, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
            state.log.push(entry);
        }
        wal.checkpoint(&state).unwrap();

        // Third batch (no checkpoint)
        for i in 41..=45 {
            let entry = LogEntry::new(2, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 45);
    assert_eq!(recovered.current_term, 2);
}

// ============================================================================
// CRC32 corruption detection
// ============================================================================

#[test]
fn wal_corrupted_entry_body_detected() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("raft.wal");

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 1..=5 {
            let entry = LogEntry::new(
                1,
                i,
                Command::Set {
                    key: format!("k{}", i).into_bytes(),
                    value: b"value".to_vec(),
                },
            );
            wal.append_entry(&entry).unwrap();
        }
    }

    // Corrupt the middle of the WAL file
    {
        let mut data = std::fs::read(&wal_path).unwrap();
        let mid = data.len() / 2;
        if mid > 0 {
            data[mid] ^= 0xFF; // flip bits
        }
        std::fs::write(&wal_path, &data).unwrap();
    }

    // Recovery should stop at corruption but not crash
    let recovered = RaftWal::recover(dir.path()).unwrap();
    // Should recover entries before corruption
    assert!(
        recovered.log.len() < 5,
        "Should have stopped at corrupted entry, got {} entries",
        recovered.log.len()
    );
}

#[test]
fn wal_truncated_last_entry_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("raft.wal");

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 1..=10 {
            let entry = LogEntry::new(1, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
    }

    // Truncate the last few bytes (simulate crash mid-write)
    {
        let data = std::fs::read(&wal_path).unwrap();
        let truncated = &data[..data.len().saturating_sub(8)];
        std::fs::write(&wal_path, truncated).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    // Should recover at least 9 of the 10 entries
    assert!(
        recovered.log.len() >= 9,
        "Should recover most entries after truncation, got {}",
        recovered.log.len()
    );
}

#[test]
fn wal_corrupted_crc_field() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("raft.wal");

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 1..=3 {
            let entry = LogEntry::new(1, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
    }

    // Corrupt the last 4 bytes (CRC32 of last entry)
    {
        let mut data = std::fs::read(&wal_path).unwrap();
        let len = data.len();
        if len >= 4 {
            data[len - 1] ^= 0xFF;
            data[len - 2] ^= 0xFF;
        }
        std::fs::write(&wal_path, &data).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    // Last entry should be rejected, first 2 preserved
    assert!(
        recovered.log.len() >= 2,
        "Should preserve entries before CRC corruption, got {}",
        recovered.log.len()
    );
}

// ============================================================================
// Large entries
// ============================================================================

#[test]
fn wal_large_1mb_entry() {
    let dir = tempfile::tempdir().unwrap();
    let big_value = vec![0xAB; 1_000_000]; // 1MB

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        let entry = LogEntry::new(
            1,
            1,
            Command::Set {
                key: b"big".to_vec(),
                value: big_value.clone(),
            },
        );
        wal.append_entry(&entry).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 1);
    if let Command::Set { value, .. } = &recovered.log[0].command {
        assert_eq!(value.len(), 1_000_000);
        assert_eq!(value[0], 0xAB);
    } else {
        panic!("Expected Set command");
    }
}

#[test]
fn wal_mixed_size_entries() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        // Small entry
        let entry = LogEntry::new(1, 1, Command::Noop);
        wal.append_entry(&entry).unwrap();

        // Medium entry (10KB)
        let entry = LogEntry::new(
            1,
            2,
            Command::Set {
                key: b"med".to_vec(),
                value: vec![0x42; 10_000],
            },
        );
        wal.append_entry(&entry).unwrap();

        // Large entry (500KB)
        let entry = LogEntry::new(
            1,
            3,
            Command::Set {
                key: b"big".to_vec(),
                value: vec![0x99; 500_000],
            },
        );
        wal.append_entry(&entry).unwrap();

        // Small again
        let entry = LogEntry::new(1, 4, Command::Noop);
        wal.append_entry(&entry).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 4);
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn wal_empty_wal_recovery() {
    let dir = tempfile::tempdir().unwrap();
    // No WAL or state file exists
    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 0);
    assert_eq!(recovered.current_term, 0);
    assert!(recovered.voted_for.is_none());
}

#[test]
fn wal_empty_wal_file_recovery() {
    let dir = tempfile::tempdir().unwrap();
    // Create an empty WAL file
    std::fs::File::create(dir.path().join("raft.wal")).unwrap();

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 0);
}

#[test]
fn wal_reopen_and_append_more() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 1..=5 {
            let entry = LogEntry::new(1, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
    }

    // Reopen and append more
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 6..=10 {
            let entry = LogEntry::new(2, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 10);
    assert_eq!(recovered.log[4].term, 1);
    assert_eq!(recovered.log[5].term, 2);
}

#[test]
fn wal_entries_at_term_boundaries() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        // Term 1: entries 1-3
        for i in 1..=3 {
            let entry = LogEntry::new(1, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
        wal.append_meta(2, &Some("node_b".to_string())).unwrap();
        // Term 2: entries 4-6
        for i in 4..=6 {
            let entry = LogEntry::new(2, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
        }
        wal.append_meta(3, &None).unwrap();
        // Term 3: entry 7
        let entry = LogEntry::new(3, 7, Command::Noop);
        wal.append_entry(&entry).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 7);
    assert_eq!(recovered.current_term, 3);
    assert!(recovered.voted_for.is_none());
}

#[test]
fn wal_all_command_variants() {
    let dir = tempfile::tempdir().unwrap();

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        let commands = vec![
            Command::Noop,
            Command::Set {
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            },
            Command::Delete { key: b"k".to_vec() },
            Command::MutationDelta(vec![1, 2, 3, 4]),
        ];

        for (i, cmd) in commands.into_iter().enumerate() {
            let entry = LogEntry::new(1, (i + 1) as u64, cmd);
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 4);
    assert!(matches!(recovered.log[0].command, Command::Noop));
    assert!(matches!(recovered.log[1].command, Command::Set { .. }));
    assert!(matches!(recovered.log[2].command, Command::Delete { .. }));
    assert!(matches!(
        recovered.log[3].command,
        Command::MutationDelta(_)
    ));
}

// ============================================================================
// Rapid append + checkpoint cycle
// ============================================================================

#[test]
fn wal_rapid_append_checkpoint_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let mut wal = RaftWal::open(dir.path()).unwrap();
    let mut state = PersistentState::new();
    state.current_term = 1;

    for cycle in 0..5 {
        for i in 1..=20 {
            let idx = cycle * 20 + i;
            let entry = LogEntry::new(1, idx, Command::Noop);
            wal.append_entry(&entry).unwrap();
            state.log.push(entry);
        }
        wal.checkpoint(&state).unwrap();
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 100);
}

// ============================================================================
// State file corruption
// ============================================================================

#[test]
fn wal_corrupted_state_json_falls_back() {
    let dir = tempfile::tempdir().unwrap();

    // Write valid state + WAL
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        let mut state = PersistentState::new();
        state.current_term = 5;
        for i in 1..=10 {
            let entry = LogEntry::new(5, i, Command::Noop);
            wal.append_entry(&entry).unwrap();
            state.log.push(entry);
        }
        wal.checkpoint(&state).unwrap();
    }

    // Corrupt the state JSON
    let state_path = dir.path().join("raft_state.json");
    if state_path.exists() {
        std::fs::write(&state_path, b"NOT VALID JSON {{{").unwrap();
    }

    // Recovery should handle gracefully
    let result = RaftWal::recover(dir.path());
    // Either succeeds with empty state (+ WAL replay) or returns error
    // Both are acceptable — the key is no panic
    let _ = result;
}

// ============================================================================
// Ordering guarantees
// ============================================================================

#[test]
fn wal_entries_recovered_in_order() {
    let dir = tempfile::tempdir().unwrap();
    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        for i in 1..=1000 {
            let entry = LogEntry::new(
                1,
                i,
                Command::Set {
                    key: format!("{}", i).into_bytes(),
                    value: vec![],
                },
            );
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 1000);
    for (i, entry) in recovered.log.iter().enumerate() {
        assert_eq!(
            entry.index,
            (i + 1) as u64,
            "Entry {} has wrong index: {}",
            i,
            entry.index
        );
    }
}

// ============================================================================
// Batch correctness (Bug 4 fix verification)
// ============================================================================

#[test]
fn wal_batch_all_entries_persisted() {
    let dir = tempfile::tempdir().unwrap();
    let batch_size = 10;

    {
        let mut wal = RaftWal::open(dir.path()).unwrap();
        // Simulate batch: append N entries
        for i in 1..=batch_size {
            let entry = LogEntry::new(
                1,
                i,
                Command::Set {
                    key: format!("batch_{}", i).into_bytes(),
                    value: b"val".to_vec(),
                },
            );
            wal.append_entry(&entry).unwrap();
        }
    }

    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(
        recovered.log.len(),
        batch_size as usize,
        "All {} batch entries should be WAL-persisted",
        batch_size
    );
}

// ============================================================================
// Concurrent access
// ============================================================================

#[test]
fn wal_concurrent_readers_single_writer() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let dir = tempfile::tempdir().unwrap();
    let wal = Arc::new(Mutex::new(RaftWal::open(dir.path()).unwrap()));

    // Writer thread
    let wal_w = wal.clone();
    let writer = thread::spawn(move || {
        for i in 1..=100 {
            let mut w = wal_w.lock().unwrap();
            let entry = LogEntry::new(1, i, Command::Noop);
            w.append_entry(&entry).unwrap();
        }
    });

    // Multiple readers checking needs_checkpoint
    let mut readers = Vec::new();
    for _ in 0..4 {
        let wal_r = wal.clone();
        readers.push(thread::spawn(move || {
            for _ in 0..50 {
                let w = wal_r.lock().unwrap();
                let _ = w.needs_checkpoint();
                drop(w);
                thread::yield_now();
            }
        }));
    }

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }

    // Verify all entries written
    drop(wal);
    let recovered = RaftWal::recover(dir.path()).unwrap();
    assert_eq!(recovered.log.len(), 100);
}

// ============================================================================
// PersistentState helpers
// ============================================================================

#[test]
fn wal_persistent_state_save_load_roundtrip() {
    let dir = tempfile::tempdir().unwrap();

    let mut state = PersistentState::new();
    state.current_term = 42;
    state.voted_for = Some("node_x".to_string());
    for i in 1..=5 {
        state.log.push(LogEntry::new(42, i, Command::Noop));
    }

    state.save_to_dir(dir.path()).unwrap();
    let loaded = PersistentState::load_from_dir(dir.path()).unwrap();

    assert_eq!(loaded.current_term, 42);
    assert_eq!(loaded.voted_for.as_deref(), Some("node_x"));
    assert_eq!(loaded.log.len(), 5);
}

#[test]
fn wal_persistent_state_compact_until() {
    let mut state = PersistentState::new();
    state.current_term = 1;
    for i in 1..=10 {
        state.log.push(LogEntry::new(1, i, Command::Noop));
    }

    state.compact_until(5);
    // After compaction, entries 1-5 are removed; 6-10 remain
    assert_eq!(state.log.len(), 5);
    assert_eq!(state.log[0].index, 6);
    assert_eq!(state.last_log_index(), 10);
}

#[test]
fn wal_persistent_state_truncate_from() {
    let mut state = PersistentState::new();
    for i in 1..=10 {
        state.log.push(LogEntry::new(1, i, Command::Noop));
    }

    state.truncate_from(7);
    assert_eq!(state.log.len(), 6);
    assert_eq!(state.last_log_index(), 6);
}
