//! Stress tests for AmorphicTableStorage — try to break it.
#![cfg(feature = "amorphic")]

use joule_db_query::amorphic_executor::{AmorphicTableStorage, TableSchema};
use joule_db_query::ast::Value;
use joule_db_query::error::QueryResult;
use joule_db_query::executor::RowData;
use joule_db_query::executor::TableStorage;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

fn make_schema(name: &str, cols: &[&str]) -> TableSchema {
    TableSchema {
        name: name.to_string(),
        columns: cols.iter().map(|c| c.to_string()).collect(),
        primary_key: None,
    }
}

fn make_row(cols: &[&str], vals: &[Value]) -> RowData {
    RowData {
        columns: cols.iter().map(|c| c.to_string()).collect(),
        values: vals.to_vec(),
    }
}

// ── Table creation edge cases ─────────────────────────────────────────

#[test]
fn stress_create_many_tables() {
    let store = AmorphicTableStorage::new();
    for i in 0..10_000 {
        let name = format!("table_{}", i);
        store
            .create_table(make_schema(&name, &["id", "val"]))
            .unwrap();
    }
    // Verify we can query them all
    for i in 0..10_000 {
        let name = format!("table_{}", i);
        let rows = store.scan(&name).unwrap();
        assert!(rows.is_empty());
    }
}

#[test]
fn stress_create_table_many_columns() {
    let store = AmorphicTableStorage::new();
    let cols: Vec<String> = (0..1000).map(|i| format!("col_{}", i)).collect();
    let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
    store
        .create_table(make_schema("wide_table", &col_refs))
        .unwrap();
    let actual_cols = store.columns("wide_table").unwrap();
    assert_eq!(actual_cols.len(), 1000);
}

#[test]
fn stress_create_duplicate_table() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("dup", &["id"])).unwrap();
    let err = store.create_table(make_schema("dup", &["id"]));
    assert!(err.is_err(), "Should error on duplicate table");
}

#[test]
fn stress_empty_table_name() {
    let store = AmorphicTableStorage::new();
    // Empty string as table name — should either work or error gracefully
    let result = std::panic::catch_unwind(|| {
        let s = AmorphicTableStorage::new();
        let _ = s.create_table(make_schema("", &["id"]));
    });
    assert!(result.is_ok(), "Empty table name should not panic");
}

#[test]
fn stress_unicode_table_name() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("表テーブル🎯", &["列"]))
        .unwrap();
    let rows = store.scan("表テーブル🎯").unwrap();
    assert!(rows.is_empty());
}

#[test]
fn stress_very_long_table_name() {
    let store = AmorphicTableStorage::new();
    let long_name = "x".repeat(100_000);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = AmorphicTableStorage::new();
        let _ = s.create_table(make_schema(&long_name, &["id"]));
    }));
    assert!(result.is_ok(), "Long table name should not panic");
}

#[test]
fn stress_special_chars_table_name() {
    let store = AmorphicTableStorage::new();
    let names = [
        "table with spaces",
        "table\twith\ttabs",
        "table\nwith\nnewlines",
        "table/with/slashes",
        "table.with.dots",
        "table\\with\\backslashes",
        "table\"with\"quotes",
        "table'with'quotes",
        "table;with;semicolons",
        "table--with--dashes",
    ];
    for name in &names {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let s = AmorphicTableStorage::new();
            let _ = s.create_table(make_schema(name, &["id"]));
        }));
        assert!(result.is_ok(), "Table name '{}' should not panic", name);
    }
}

// ── Drop table edge cases ─────────────────────────────────────────────

#[test]
fn stress_drop_nonexistent() {
    let store = AmorphicTableStorage::new();
    let err = store.drop_table("nonexistent");
    assert!(err.is_err(), "Should error dropping nonexistent table");
}

#[test]
fn stress_drop_twice() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("t", &["id"])).unwrap();
    store.drop_table("t").unwrap();
    let err = store.drop_table("t");
    assert!(err.is_err(), "Should error dropping already-dropped table");
}

#[test]
fn stress_insert_into_dropped_table() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("t", &["id"])).unwrap();
    store.drop_table("t").unwrap();
    let err = store.insert("t", &make_row(&["id"], &[Value::Int(1)]));
    assert!(err.is_err(), "Should error inserting into dropped table");
}

#[test]
fn stress_scan_dropped_table() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("t", &["id"])).unwrap();
    store
        .insert("t", &make_row(&["id"], &[Value::Int(1)]))
        .unwrap();
    store.drop_table("t").unwrap();
    let err = store.scan("t");
    assert!(err.is_err(), "Should error scanning dropped table");
}

#[test]
fn stress_recreate_after_drop() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("t", &["id", "name"]))
        .unwrap();
    store
        .insert(
            "t",
            &make_row(&["id", "name"], &[Value::Int(1), Value::String("a".into())]),
        )
        .unwrap();
    store.drop_table("t").unwrap();
    // Recreate with different schema
    store
        .create_table(make_schema("t", &["x", "y", "z"]))
        .unwrap();
    let rows = store.scan("t").unwrap();
    assert!(rows.is_empty(), "Recreated table should be empty");
    let cols = store.columns("t").unwrap();
    assert_eq!(cols, vec!["x", "y", "z"]);
}

// ── Insert edge cases ─────────────────────────────────────────────────

#[test]
fn stress_insert_many_rows() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("big", &["id", "val"]))
        .unwrap();
    for i in 0..100_000 {
        store
            .insert(
                "big",
                &make_row(
                    &["id", "val"],
                    &[Value::Int(i), Value::String(format!("row_{}", i))],
                ),
            )
            .unwrap();
    }
    let rows = store.scan("big").unwrap();
    assert_eq!(rows.len(), 100_000);
}

#[test]
fn stress_insert_null_values() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("nulls", &["a", "b", "c"]))
        .unwrap();
    store
        .insert(
            "nulls",
            &make_row(&["a", "b", "c"], &[Value::Null, Value::Null, Value::Null]),
        )
        .unwrap();
    let rows = store.scan("nulls").unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].values.iter().all(|v| matches!(v, Value::Null)));
}

#[test]
fn stress_insert_type_variety() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("types", &["a"])).unwrap();
    let values = vec![
        Value::Null,
        Value::Bool(true),
        Value::Bool(false),
        Value::Int(0),
        Value::Int(i64::MAX),
        Value::Int(i64::MIN),
        Value::Float(0.0),
        Value::Float(f64::MAX),
        Value::Float(f64::MIN),
        Value::Float(f64::MIN_POSITIVE),
        Value::Float(f64::EPSILON),
        Value::String(String::new()),
        Value::String("x".repeat(1_000_000)),
        Value::String("null".into()),
        Value::String("\0\0\0".into()),
        Value::String("🎯🚀🌍".into()),
    ];
    for v in &values {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let s = AmorphicTableStorage::new();
            s.create_table(make_schema("t", &["a"])).unwrap();
            s.insert("t", &make_row(&["a"], &[v.clone()])).unwrap();
        }));
        assert!(result.is_ok(), "Inserting {:?} should not panic", v);
    }
}

#[test]
fn stress_insert_float_special() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("floats", &["val"])).unwrap();
    let specials = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        f64::MIN_POSITIVE,
    ];
    for f in specials {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let s = AmorphicTableStorage::new();
            s.create_table(make_schema("t", &["val"])).unwrap();
            s.insert("t", &make_row(&["val"], &[Value::Float(f)]))
                .unwrap();
            let _ = s.scan("t");
        }));
        assert!(result.is_ok(), "Float {} should not panic", f);
    }
}

#[test]
fn stress_insert_extra_columns() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("t", &["id"])).unwrap();
    // Insert row with more columns than schema — should not panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = AmorphicTableStorage::new();
        s.create_table(make_schema("t", &["id"])).unwrap();
        let _ = s.insert(
            "t",
            &make_row(
                &["id", "extra1", "extra2"],
                &[Value::Int(1), Value::Int(2), Value::Int(3)],
            ),
        );
    }));
    assert!(result.is_ok(), "Extra columns should not panic");
}

#[test]
fn stress_insert_missing_columns() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("t", &["id", "name", "val"]))
        .unwrap();
    // Insert row with fewer columns than schema
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = AmorphicTableStorage::new();
        s.create_table(make_schema("t", &["id", "name", "val"]))
            .unwrap();
        let _ = s.insert("t", &make_row(&["id"], &[Value::Int(1)]));
    }));
    assert!(result.is_ok(), "Missing columns should not panic");
}

#[test]
fn stress_insert_empty_row() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("t", &["id"])).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let s = AmorphicTableStorage::new();
        s.create_table(make_schema("t", &["id"])).unwrap();
        let _ = s.insert("t", &make_row(&[], &[]));
    }));
    assert!(result.is_ok(), "Empty row should not panic");
}

// ── Concurrent access ─────────────────────────────────────────────────

#[test]
fn stress_concurrent_inserts() {
    let store: Arc<AmorphicTableStorage> = Arc::new(AmorphicTableStorage::new());
    store
        .create_table(make_schema("concurrent", &["tid", "seq"]))
        .unwrap();

    let mut handles = vec![];
    for tid in 0..10 {
        let s: Arc<AmorphicTableStorage> = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for seq in 0..1000 {
                s.insert(
                    "concurrent",
                    &make_row(&["tid", "seq"], &[Value::Int(tid), Value::Int(seq)]),
                )
                .unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let rows = store.scan("concurrent").unwrap();
    assert_eq!(rows.len(), 10_000, "All 10,000 rows should be present");
}

#[test]
fn stress_concurrent_read_write() {
    let store: Arc<AmorphicTableStorage> = Arc::new(AmorphicTableStorage::new());
    store.create_table(make_schema("rw", &["id"])).unwrap();

    let mut handles = vec![];
    // 5 writers
    for _ in 0..5 {
        let s: Arc<AmorphicTableStorage> = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for i in 0..1000 {
                let _ = s.insert("rw", &make_row(&["id"], &[Value::Int(i)]));
            }
        }));
    }
    // 5 readers
    for _ in 0..5 {
        let s: Arc<AmorphicTableStorage> = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                let _ = s.scan("rw");
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
    // No deadlock, no panic
}

#[test]
fn stress_concurrent_create_same_table() {
    let store: Arc<AmorphicTableStorage> = Arc::new(AmorphicTableStorage::new());
    let mut handles = vec![];
    for _ in 0..10 {
        let s: Arc<AmorphicTableStorage> = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            s.create_table(make_schema("race", &["id"]))
        }));
    }
    let results: Vec<QueryResult<()>> = handles
        .into_iter()
        .map(|h: thread::JoinHandle<QueryResult<()>>| h.join().unwrap())
        .collect();
    let successes = results
        .iter()
        .filter(|r: &&QueryResult<()>| r.is_ok())
        .count();
    assert_eq!(successes, 1, "Exactly one CREATE should succeed");
}

#[test]
fn stress_concurrent_drop_while_querying() {
    let store: Arc<AmorphicTableStorage> = Arc::new(AmorphicTableStorage::new());
    store
        .create_table(make_schema("ephemeral", &["id"]))
        .unwrap();
    for i in 0..100 {
        store
            .insert("ephemeral", &make_row(&["id"], &[Value::Int(i)]))
            .unwrap();
    }

    let mut handles = vec![];
    // 1 thread drops the table
    let s: Arc<AmorphicTableStorage> = Arc::clone(&store);
    handles.push(thread::spawn(move || {
        thread::sleep(std::time::Duration::from_millis(1));
        let _ = s.drop_table("ephemeral");
    }));
    // 9 threads query it
    for _ in 0..9 {
        let s: Arc<AmorphicTableStorage> = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let _ = s.scan("ephemeral"); // May fail — that's OK
            }
        }));
    }

    for h in handles {
        assert!(h.join().is_ok(), "No thread should panic");
    }
}

// ── Wide rows ─────────────────────────────────────────────────────────

#[test]
fn stress_wide_rows() {
    let store = AmorphicTableStorage::new();
    let cols: Vec<String> = (0..100).map(|i| format!("col_{}", i)).collect();
    let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
    store.create_table(make_schema("wide", &col_refs)).unwrap();

    let vals: Vec<Value> = (0..100)
        .map(|i| Value::String("x".repeat(10_000)))
        .collect();
    store.insert("wide", &make_row(&col_refs, &vals)).unwrap();

    let rows = store.scan("wide").unwrap();
    assert_eq!(rows.len(), 1);
}

// ── String value edge cases ──────────────────────────────────────────

#[test]
fn stress_special_string_values() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("strings", &["val"]))
        .unwrap();

    let big_string = "a".repeat(1_000_000);
    let specials = vec![
        "\0",
        "\n\r\t",
        "\\",
        "\"",
        "'",
        "NULL",
        "null",
        "true",
        "false",
        "0",
        "",
        " ",
        "\u{200B}", // zero-width space
        "\u{FEFF}", // BOM
        big_string.as_str(),
    ];
    for s in specials {
        store
            .insert(
                "strings",
                &make_row(&["val"], &[Value::String(s.to_string())]),
            )
            .unwrap();
    }
    let rows = store.scan("strings").unwrap();
    assert_eq!(rows.len(), 15);
}

// ── Update edge cases ────────────────────────────────────────────────

#[test]
fn stress_update_nonexistent_table() {
    let store = AmorphicTableStorage::new();
    let mut assignments = HashMap::new();
    assignments.insert("id".to_string(), Value::Int(42));
    let err = store.update("nonexistent", &assignments, None);
    assert!(err.is_err());
}

#[test]
fn stress_update_all_rows() {
    let store = AmorphicTableStorage::new();
    store
        .create_table(make_schema("t", &["id", "val"]))
        .unwrap();
    for i in 0..100 {
        store
            .insert(
                "t",
                &make_row(
                    &["id", "val"],
                    &[Value::Int(i), Value::String("old".into())],
                ),
            )
            .unwrap();
    }
    let mut assignments = HashMap::new();
    assignments.insert("val".to_string(), Value::String("new".into()));
    let updated = store.update("t", &assignments, None).unwrap();
    assert_eq!(updated, 100);
}

// ── Delete edge cases ────────────────────────────────────────────────

#[test]
fn stress_delete_nonexistent_table() {
    let store = AmorphicTableStorage::new();
    let err = store.delete("nonexistent", None);
    assert!(err.is_err());
}

#[test]
fn stress_delete_all_rows() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("t", &["id"])).unwrap();
    for i in 0..100 {
        store
            .insert("t", &make_row(&["id"], &[Value::Int(i)]))
            .unwrap();
    }
    let deleted = store.delete("t", None).unwrap();
    assert_eq!(deleted, 100);
    let rows = store.scan("t").unwrap();
    assert!(rows.is_empty());
}

// ── Row counter overflow ─────────────────────────────────────────────

#[test]
fn stress_row_counter_many_inserts_deletes() {
    let store = AmorphicTableStorage::new();
    store.create_table(make_schema("churn", &["id"])).unwrap();
    for cycle in 0..100 {
        for i in 0..100 {
            store
                .insert("churn", &make_row(&["id"], &[Value::Int(i)]))
                .unwrap();
        }
        store.delete("churn", None).unwrap();
    }
    // 10,000 inserts and 100 deletes — row counter should keep incrementing without issue
    let rows = store.scan("churn").unwrap();
    assert!(rows.is_empty());
}
