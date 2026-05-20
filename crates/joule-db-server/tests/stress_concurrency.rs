/*
 * Server Concurrency Stress Tests
 * Exercises thread-safety, lock contention, DDL/DML races,
 * high-fan-out reads, write skew, and resource exhaustion.
 */

use joule_db_server::query::{QueryExecutor, QueryRequest, SimpleQueryExecutor};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

fn q(sql: &str) -> QueryRequest {
    QueryRequest {
        sql: sql.to_string(),
        params: Default::default(),
        args: Vec::new(),
        explain: false,
        limit: None,
        session_id: None,
        query_timeout_ms: Some(30000),
        branch_id: None,
        tenant_id: None,
    }
}

// ── Parallel DDL ─────────────────────────────────────────────────────

#[test]
fn conc_parallel_create_tables_100() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    let handles: Vec<_> = (0..100)
        .map(|i| {
            let exec = exec.clone();
            thread::spawn(move || {
                exec.execute(&q(&format!("CREATE TABLE conc_t_{} (id INT, val TEXT)", i)))
            })
        })
        .collect();
    let mut ok = 0;
    for h in handles {
        if h.join().unwrap().is_ok() {
            ok += 1;
        }
    }
    // All 100 creates should succeed (unique names)
    assert_eq!(ok, 100, "All 100 CREATE TABLE should succeed");
}

#[test]
fn conc_parallel_drop_same_table() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE drop_race (id INT)")).unwrap();

    // 50 threads all try to DROP the same table simultaneously.
    // Under heavy contention, some or all may error — the key invariant
    // is that none should panic (thread-safety).
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let exec = exec.clone();
            thread::spawn(move || exec.execute(&q("DROP TABLE IF EXISTS drop_race")))
        })
        .collect();

    for h in handles {
        // join().unwrap() verifies no panic occurred in any thread
        let _ = h.join().unwrap();
    }
}

// ── Reader/Writer Contention ─────────────────────────────────────────

#[test]
fn conc_readers_during_writes() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE rw_contend (id INT, data TEXT)"))
        .unwrap();

    // Seed data
    for i in 0..20 {
        exec.execute(&q(&format!(
            "INSERT INTO rw_contend VALUES ({}, 'seed')",
            i
        )))
        .unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let read_count = Arc::new(AtomicU64::new(0));
    let write_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    // 20 reader threads
    for _ in 0..20 {
        let exec = exec.clone();
        let stop = stop.clone();
        let count = read_count.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let r = exec.execute(&q("SELECT * FROM rw_contend"));
                assert!(r.is_ok(), "Read should not fail during concurrent writes");
                count.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    // 10 writer threads
    for t in 0..10 {
        let exec = exec.clone();
        let stop = stop.clone();
        let count = write_count.clone();
        handles.push(thread::spawn(move || {
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                let _ = exec.execute(&q(&format!(
                    "INSERT INTO rw_contend VALUES ({}, 'writer_{}_row_{}')",
                    1000 + t * 10000 + i,
                    t,
                    i
                )));
                count.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
        }));
    }

    // Run for 3 seconds
    thread::sleep(Duration::from_secs(3));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }

    let reads = read_count.load(Ordering::Relaxed);
    let writes = write_count.load(Ordering::Relaxed);
    assert!(reads > 0, "Readers should have completed some reads");
    assert!(writes > 0, "Writers should have completed some writes");
}

// ── DDL + DML Race ───────────────────────────────────────────────────

#[test]
fn conc_create_insert_select_interleaved() {
    let exec = Arc::new(SimpleQueryExecutor::new());

    // Thread 1: create tables
    let exec1 = exec.clone();
    let h1 = thread::spawn(move || {
        for i in 0..30 {
            let _ = exec1.execute(&q(&format!(
                "CREATE TABLE IF NOT EXISTS race_{} (id INT, v TEXT)",
                i
            )));
        }
    });

    // Thread 2: try to insert into tables that may or may not exist yet
    let exec2 = exec.clone();
    let h2 = thread::spawn(move || {
        for i in 0..30 {
            let _ = exec2.execute(&q(&format!("INSERT INTO race_{} VALUES (1, 'test')", i)));
        }
    });

    // Thread 3: try to select from tables
    let exec3 = exec.clone();
    let h3 = thread::spawn(move || {
        for i in 0..30 {
            let _ = exec3.execute(&q(&format!("SELECT * FROM race_{}", i)));
        }
    });

    // None of these should panic
    h1.join().unwrap();
    h2.join().unwrap();
    h3.join().unwrap();
}

// ── High-Fan-Out Reads ──────────────────────────────────────────────

#[test]
fn conc_200_concurrent_selects() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE fan_out (id INT, data TEXT)"))
        .unwrap();
    for i in 0..50 {
        exec.execute(&q(&format!(
            "INSERT INTO fan_out VALUES ({}, '{}')",
            i,
            "x".repeat(100)
        )))
        .unwrap();
    }

    let handles: Vec<_> = (0..200)
        .map(|i| {
            let exec = exec.clone();
            thread::spawn(move || {
                let r = exec.execute(&q(&format!("SELECT * FROM fan_out WHERE id = {}", i % 50)));
                assert!(r.is_ok(), "Select {} should succeed", i);
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ── Write-Heavy Workload ─────────────────────────────────────────────

#[test]
fn conc_100_writers_same_table() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE write_heavy (id INT, src INT, msg TEXT)"))
        .unwrap();

    let handles: Vec<_> = (0..100)
        .map(|t| {
            let exec = exec.clone();
            thread::spawn(move || {
                for i in 0..10 {
                    let _ = exec.execute(&q(&format!(
                        "INSERT INTO write_heavy VALUES ({}, {}, 'thread {} row {}')",
                        t * 10 + i,
                        t,
                        t,
                        i
                    )));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let result = exec.execute(&q("SELECT * FROM write_heavy")).unwrap();
    assert_eq!(result.rows.len(), 1000, "All 1000 inserts should persist");
}

// ── Concurrent Aggregate Queries ─────────────────────────────────────

#[test]
fn conc_aggregates_during_inserts() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE agg_race (id INT, amount REAL)"))
        .unwrap();
    for i in 0..100 {
        exec.execute(&q(&format!(
            "INSERT INTO agg_race VALUES ({}, {})",
            i,
            i as f64 * 1.5
        )))
        .unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Readers doing aggregates
    for _ in 0..10 {
        let exec = exec.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = exec.execute(&q("SELECT COUNT(*), SUM(amount), AVG(amount), MIN(amount), MAX(amount) FROM agg_race"));
            }
        }));
    }

    // Writers inserting more rows
    for t in 0..5 {
        let exec = exec.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                let _ = exec.execute(&q(&format!(
                    "INSERT INTO agg_race VALUES ({}, {})",
                    10000 + t * 1000 + i,
                    i as f64 * 0.7
                )));
                i += 1;
            }
        }));
    }

    thread::sleep(Duration::from_secs(2));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

// ── Rapid Table Churn ────────────────────────────────────────────────

#[test]
fn conc_rapid_create_drop_cycle_parallel() {
    let exec = Arc::new(SimpleQueryExecutor::new());

    let handles: Vec<_> = (0..20)
        .map(|t| {
            let exec = exec.clone();
            thread::spawn(move || {
                for i in 0..25 {
                    let name = format!("churn_p_{}_{}", t, i);
                    let _ = exec.execute(&q(&format!("CREATE TABLE {} (id INT)", name)));
                    let _ = exec.execute(&q(&format!("INSERT INTO {} VALUES ({})", name, i)));
                    let _ = exec.execute(&q(&format!("SELECT * FROM {}", name)));
                    let _ = exec.execute(&q(&format!("DROP TABLE IF EXISTS {}", name)));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ── UPDATE Contention ────────────────────────────────────────────────

#[test]
fn conc_update_same_row_from_many_threads() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE counter (id INT, val INT)"))
        .unwrap();
    exec.execute(&q("INSERT INTO counter VALUES (1, 0)"))
        .unwrap();

    let handles: Vec<_> = (0..50)
        .map(|t| {
            let exec = exec.clone();
            thread::spawn(move || {
                for _ in 0..10 {
                    let _ =
                        exec.execute(&q(&format!("UPDATE counter SET val = {} WHERE id = 1", t)));
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // The row should still exist and have some value
    let result = exec
        .execute(&q("SELECT val FROM counter WHERE id = 1"))
        .unwrap();
    assert_eq!(
        result.rows.len(),
        1,
        "Row should survive concurrent updates"
    );
}

// ── DELETE During SELECT ─────────────────────────────────────────────

#[test]
fn conc_delete_during_select() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE del_race (id INT, data TEXT)"))
        .unwrap();
    for i in 0..100 {
        exec.execute(&q(&format!(
            "INSERT INTO del_race VALUES ({}, 'row_{}')",
            i, i
        )))
        .unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Readers
    for _ in 0..10 {
        let exec = exec.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = exec.execute(&q("SELECT * FROM del_race"));
            }
        }));
    }

    // Deleters
    for t in 0..5 {
        let exec = exec.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            for i in (t * 20)..((t + 1) * 20) {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let _ = exec.execute(&q(&format!("DELETE FROM del_race WHERE id = {}", i)));
            }
        }));
    }

    thread::sleep(Duration::from_secs(2));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

// ── Throughput Measurement ───────────────────────────────────────────

#[test]
fn conc_throughput_reads() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE tp (id INT, val TEXT)"))
        .unwrap();
    for i in 0..10 {
        exec.execute(&q(&format!("INSERT INTO tp VALUES ({}, 'v{}')", i, i)))
            .unwrap();
    }

    let count = Arc::new(AtomicU64::new(0));
    let start = Instant::now();
    let duration = Duration::from_secs(3);

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let exec = exec.clone();
            let count = count.clone();
            thread::spawn(move || {
                while start.elapsed() < duration {
                    let _ = exec.execute(&q("SELECT * FROM tp"));
                    count.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let total = count.load(Ordering::Relaxed);
    let qps = total as f64 / 3.0;
    // Just ensure it's above some baseline — the test shouldn't fail on slow CI
    assert!(
        total > 10,
        "Should complete some queries; got {} ({:.0} q/s)",
        total,
        qps
    );
}

// ── Nested Query Patterns ────────────────────────────────────────────

#[test]
fn conc_subquery_during_writes() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE sub_main (id INT, val INT)"))
        .unwrap();
    exec.execute(&q("CREATE TABLE sub_lookup (id INT, name TEXT)"))
        .unwrap();
    for i in 0..20 {
        exec.execute(&q(&format!(
            "INSERT INTO sub_main VALUES ({}, {})",
            i,
            i * 10
        )))
        .unwrap();
        exec.execute(&q(&format!(
            "INSERT INTO sub_lookup VALUES ({}, 'name_{}')",
            i, i
        )))
        .unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();

    // Readers doing subqueries
    for _ in 0..10 {
        let exec = exec.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = exec.execute(&q(
                    "SELECT * FROM sub_main WHERE val > (SELECT MIN(val) FROM sub_main)",
                ));
            }
        }));
    }

    // Writers
    for t in 0..5 {
        let exec = exec.clone();
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                let _ = exec.execute(&q(&format!(
                    "INSERT INTO sub_main VALUES ({}, {})",
                    1000 + t * 1000 + i,
                    i * 10
                )));
                i += 1;
            }
        }));
    }

    thread::sleep(Duration::from_secs(2));
    stop.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().unwrap();
    }
}

// ── Memory Pressure: Many Small Tables ───────────────────────────────

#[test]
fn conc_create_500_tables_then_query_all() {
    let exec = Arc::new(SimpleQueryExecutor::new());

    // Create 500 tables sequentially
    for i in 0..500 {
        exec.execute(&q(&format!("CREATE TABLE mem_{} (id INT, data TEXT)", i)))
            .unwrap();
        exec.execute(&q(&format!("INSERT INTO mem_{} VALUES (1, 'row')", i)))
            .unwrap();
    }

    // Query all 500 tables in parallel (50 threads, 10 tables each)
    let handles: Vec<_> = (0..50)
        .map(|t| {
            let exec = exec.clone();
            thread::spawn(move || {
                for i in (t * 10)..((t + 1) * 10) {
                    let r = exec.execute(&q(&format!("SELECT * FROM mem_{}", i)));
                    assert!(r.is_ok(), "Query mem_{} failed", i);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ── Complex Query Under Load ─────────────────────────────────────────

#[test]
fn conc_complex_queries_with_functions() {
    let exec = Arc::new(SimpleQueryExecutor::new());
    exec.execute(&q("CREATE TABLE func_load (id INT, name TEXT, score REAL)"))
        .unwrap();
    for i in 0..50 {
        exec.execute(&q(&format!(
            "INSERT INTO func_load VALUES ({}, 'user_{}', {})",
            i,
            i,
            i as f64 * 1.1
        )))
        .unwrap();
    }

    let handles: Vec<_> = (0..30)
        .map(|_| {
            let exec = exec.clone();
            thread::spawn(move || {
                // Mix of function calls in queries
                let _ = exec.execute(&q(
                    "SELECT UPPER(name), ABS(score), LENGTH(name) FROM func_load WHERE score > 10.0",
                ));
                let _ = exec.execute(&q(
                    "SELECT COALESCE(name, 'unknown'), ROUND(score, 2) FROM func_load",
                ));
                let _ = exec.execute(&q(
                    "SELECT COUNT(*), AVG(score), MAX(score) FROM func_load",
                ));
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}
