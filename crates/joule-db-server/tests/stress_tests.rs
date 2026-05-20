//! Load/Stress Tests (Phase 6.3)
//!
//! Verifies the server handles concurrent load without degradation.
//! Tests connection churn, concurrent queries, and memory bounds.

use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};
use std::sync::Arc;

fn make_query(sql: &str) -> QueryRequest {
    QueryRequest {
        sql: sql.to_string(),
        params: Default::default(),
        args: Vec::new(),
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: Some(10000),
        branch_id: None,
        tenant_id: None,
    }
}

// ================================================================
// Concurrent Query Execution
// ================================================================

#[test]
fn test_concurrent_queries_100_threads() {
    let executor = Arc::new(SimpleQueryExecutor::new());

    // Create a table
    executor
        .execute(&make_query("CREATE TABLE stress (id INT, val TEXT)"))
        .unwrap();

    // Insert some seed data
    for i in 0..100 {
        executor
            .execute(&make_query(&format!(
                "INSERT INTO stress VALUES ({}, 'value{}')",
                i, i
            )))
            .unwrap();
    }

    // Spawn 100 threads doing concurrent reads
    let mut handles = Vec::new();
    for i in 0..100 {
        let exec = executor.clone();
        handles.push(std::thread::spawn(move || {
            let req = make_query(&format!("SELECT * FROM stress WHERE id = {}", i));
            let result = exec.execute(&req);
            assert!(result.is_ok(), "Query {} should succeed", i);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_writes_50_threads() {
    let executor = Arc::new(SimpleQueryExecutor::new());

    executor
        .execute(&make_query("CREATE TABLE write_stress (id INT, data TEXT)"))
        .unwrap();

    let mut handles = Vec::new();
    for i in 0..50 {
        let exec = executor.clone();
        handles.push(std::thread::spawn(move || {
            let req = make_query(&format!(
                "INSERT INTO write_stress VALUES ({}, 'data from thread {}')",
                i, i
            ));
            let result = exec.execute(&req);
            assert!(result.is_ok(), "Insert {} should succeed", i);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all rows were inserted
    let result = executor
        .execute(&make_query("SELECT * FROM write_stress"))
        .unwrap();
    assert_eq!(result.rows.len(), 50, "All 50 inserts should persist");
}

// ================================================================
// Mixed Read/Write Load
// ================================================================

#[test]
fn test_mixed_read_write_load() {
    let executor = Arc::new(SimpleQueryExecutor::new());

    executor
        .execute(&make_query("CREATE TABLE mixed (id INT, counter INT)"))
        .unwrap();

    for i in 0..20 {
        executor
            .execute(&make_query(&format!("INSERT INTO mixed VALUES ({}, 0)", i)))
            .unwrap();
    }

    let mut handles = Vec::new();

    // Writers: UPDATE operations
    for i in 0..25 {
        let exec = executor.clone();
        handles.push(std::thread::spawn(move || {
            let req = make_query(&format!(
                "UPDATE mixed SET counter = {} WHERE id = {}",
                i + 1,
                i % 20
            ));
            let _ = exec.execute(&req);
        }));
    }

    // Readers: SELECT operations (concurrent with writes)
    for _ in 0..25 {
        let exec = executor.clone();
        handles.push(std::thread::spawn(move || {
            let req = make_query("SELECT * FROM mixed");
            let result = exec.execute(&req);
            assert!(result.is_ok(), "Read should succeed during writes");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ================================================================
// Rapid Create/Drop Table Churn
// ================================================================

#[test]
fn test_table_create_drop_churn() {
    let executor = SimpleQueryExecutor::new();

    for i in 0..50 {
        let table_name = format!("churn_table_{}", i);
        executor
            .execute(&make_query(&format!(
                "CREATE TABLE {} (id INT, data TEXT)",
                table_name
            )))
            .unwrap();

        executor
            .execute(&make_query(&format!(
                "INSERT INTO {} VALUES (1, 'test')",
                table_name
            )))
            .unwrap();

        executor
            .execute(&make_query(&format!("DROP TABLE {}", table_name)))
            .unwrap();
    }
}

// ================================================================
// Large Result Set Handling
// ================================================================

#[test]
fn test_large_result_set_does_not_oom() {
    let executor = SimpleQueryExecutor::new();

    executor
        .execute(&make_query("CREATE TABLE large (id INT, padding TEXT)"))
        .unwrap();

    // Insert 1000 rows with moderately sized data
    for i in 0..1000 {
        let padding = "x".repeat(100);
        executor
            .execute(&make_query(&format!(
                "INSERT INTO large VALUES ({}, '{}')",
                i, padding
            )))
            .unwrap();
    }

    // SELECT all rows — should succeed without OOM
    let result = executor
        .execute(&make_query("SELECT * FROM large"))
        .unwrap();
    assert_eq!(result.rows.len(), 1000);
}

// ================================================================
// Parser Stress: Many Columns
// ================================================================

#[test]
fn test_many_columns_query() {
    let executor = SimpleQueryExecutor::new();

    // Create a table with many columns
    let cols: Vec<String> = (0..50).map(|i| format!("col{} TEXT", i)).collect();
    let create_sql = format!("CREATE TABLE wide ({})", cols.join(", "));
    executor.execute(&make_query(&create_sql)).unwrap();

    // Insert a row
    let vals: Vec<String> = (0..50).map(|i| format!("'value{}'", i)).collect();
    let insert_sql = format!("INSERT INTO wide VALUES ({})", vals.join(", "));
    executor.execute(&make_query(&insert_sql)).unwrap();

    // Select all columns
    let result = executor.execute(&make_query("SELECT * FROM wide")).unwrap();
    assert_eq!(result.columns.len(), 50);
    assert_eq!(result.rows.len(), 1);
}

// ================================================================
// Sustained Load Tests (Session 58 — Release Readiness)
// ================================================================

/// Sustained mixed read/write workload over configurable duration.
/// Default: 10 seconds. Override via STRESS_DURATION_SECS env var.
#[test]
fn test_sustained_mixed_workload() {
    let duration_secs: u64 = std::env::var("STRESS_DURATION_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let duration = std::time::Duration::from_secs(duration_secs);

    let executor = Arc::new(SimpleQueryExecutor::new());
    executor
        .execute(&make_query("CREATE TABLE sustained (id INT, val TEXT)"))
        .unwrap();
    for i in 0..50 {
        executor
            .execute(&make_query(&format!(
                "INSERT INTO sustained VALUES ({}, 'seed{}')",
                i, i
            )))
            .unwrap();
    }

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut handles = Vec::new();

    // 10 reader threads
    for t in 0..10 {
        let exec = executor.clone();
        let stop = stop.clone();
        handles.push(std::thread::spawn(move || {
            let mut reads = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let req = make_query(&format!("SELECT * FROM sustained WHERE id = {}", t * 5));
                let _ = exec.execute(&req);
                reads += 1;
            }
            reads
        }));
    }

    // 5 writer threads
    for t in 0..5 {
        let exec = executor.clone();
        let stop = stop.clone();
        handles.push(std::thread::spawn(move || {
            let mut writes = 0u64;
            let mut counter = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let req = make_query(&format!(
                    "INSERT INTO sustained VALUES ({}, 'thread{}_iter{}')",
                    1000 + t * 10000 + counter,
                    t,
                    counter
                ));
                let _ = exec.execute(&req);
                writes += 1;
                counter += 1;
            }
            writes
        }));
    }

    std::thread::sleep(duration);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);

    let mut total_reads = 0u64;
    let mut total_writes = 0u64;
    for (i, handle) in handles.into_iter().enumerate() {
        let ops = handle.join().unwrap();
        if i < 10 {
            total_reads += ops;
        } else {
            total_writes += ops;
        }
    }

    eprintln!(
        "Sustained mixed workload ({}s): {} reads, {} writes ({:.0} ops/s)",
        duration_secs,
        total_reads,
        total_writes,
        (total_reads + total_writes) as f64 / duration_secs as f64
    );
    assert!(total_reads > 0, "Should complete at least some reads");
    assert!(total_writes > 0, "Should complete at least some writes");
}

/// Connection scaling: 50, 100, 200, 500 concurrent threads.
#[test]
fn test_connection_scaling() {
    for thread_count in [50, 100, 200, 500] {
        let executor = Arc::new(SimpleQueryExecutor::new());
        executor
            .execute(&make_query(&format!(
                "CREATE TABLE scale_{} (id INT, data TEXT)",
                thread_count
            )))
            .unwrap();
        for i in 0..10 {
            executor
                .execute(&make_query(&format!(
                    "INSERT INTO scale_{} VALUES ({}, 'data')",
                    thread_count, i
                )))
                .unwrap();
        }

        let success = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let failure = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut handles = Vec::new();

        for _ in 0..thread_count {
            let exec = executor.clone();
            let ok = success.clone();
            let err = failure.clone();
            let tc = thread_count;
            handles.push(std::thread::spawn(move || {
                let req = make_query(&format!("SELECT * FROM scale_{}", tc));
                match exec.execute(&req) {
                    Ok(_) => ok.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    Err(_) => err.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                };
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let ok = success.load(std::sync::atomic::Ordering::Relaxed);
        let fail = failure.load(std::sync::atomic::Ordering::Relaxed);
        let pct = ok as f64 / (ok + fail) as f64 * 100.0;
        eprintln!(
            "Connection scaling ({}): {}/{} succeeded ({:.1}%)",
            thread_count,
            ok,
            ok + fail,
            pct
        );
        assert!(
            pct >= 95.0,
            "At least 95% of {} threads should succeed, got {:.1}%",
            thread_count,
            pct
        );
    }
}

/// DDL under concurrent DML load: CREATE/DROP tables while readers run.
#[test]
fn test_sustained_ddl_under_load() {
    let executor = Arc::new(SimpleQueryExecutor::new());

    // Stable table for background readers
    executor
        .execute(&make_query("CREATE TABLE stable_ddl (id INT, val TEXT)"))
        .unwrap();
    for i in 0..20 {
        executor
            .execute(&make_query(&format!(
                "INSERT INTO stable_ddl VALUES ({}, 'val{}')",
                i, i
            )))
            .unwrap();
    }

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let errors = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut handles = Vec::new();

    // 5 background reader threads on stable table
    for _ in 0..5 {
        let exec = executor.clone();
        let stop = stop.clone();
        let errs = errors.clone();
        handles.push(std::thread::spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let req = make_query("SELECT * FROM stable_ddl");
                if exec.execute(&req).is_err() {
                    errs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }));
    }

    // 20 DDL cycles: CREATE → INSERT → SELECT → DROP
    for i in 0..20 {
        let tbl = format!("ddl_churn_{}", i);
        executor
            .execute(&make_query(&format!(
                "CREATE TABLE {} (id INT, data TEXT)",
                tbl
            )))
            .unwrap();
        executor
            .execute(&make_query(&format!(
                "INSERT INTO {} VALUES (1, 'test')",
                tbl
            )))
            .unwrap();
        let result = executor
            .execute(&make_query(&format!("SELECT * FROM {}", tbl)))
            .unwrap();
        assert_eq!(result.rows.len(), 1);
        executor
            .execute(&make_query(&format!("DROP TABLE {}", tbl)))
            .unwrap();
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for handle in handles {
        handle.join().unwrap();
    }

    let errs = errors.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("DDL under load: 20 DDL cycles, {} reader errors", errs);
    assert_eq!(
        errs, 0,
        "Background readers should not see errors during DDL churn"
    );
}

/// Throughput measurement: single-threaded queries per second.
#[test]
fn test_throughput_measurement() {
    let executor = SimpleQueryExecutor::new();

    executor
        .execute(&make_query("CREATE TABLE throughput (id INT, val TEXT)"))
        .unwrap();
    for i in 0..100 {
        executor
            .execute(&make_query(&format!(
                "INSERT INTO throughput VALUES ({}, 'data{}')",
                i, i
            )))
            .unwrap();
    }

    let count = 1000;
    let start = std::time::Instant::now();
    for _ in 0..count {
        executor
            .execute(&make_query("SELECT * FROM throughput WHERE id = 42"))
            .unwrap();
    }
    let elapsed = start.elapsed();
    let qps = count as f64 / elapsed.as_secs_f64();

    eprintln!(
        "Throughput: {} queries in {:.2}s = {:.0} qps",
        count,
        elapsed.as_secs_f64(),
        qps
    );
    assert!(qps > 100.0, "Should sustain >100 qps, got {:.0}", qps);
}
