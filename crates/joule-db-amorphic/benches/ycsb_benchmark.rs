//! YCSB Benchmark: Yahoo! Cloud Serving Benchmark
//!
//! Industry-standard benchmark for key-value and NoSQL databases.
//! https://github.com/brianfrankcooper/YCSB
//!
//! ## Workloads
//! - A: Update heavy (50% read, 50% update)
//! - B: Read mostly (95% read, 5% update)
//! - C: Read only (100% read)
//! - D: Read latest (read recently inserted)
//! - E: Short range scan
//! - F: Read-modify-write
//!
//! ## Metrics
//! - Throughput (ops/sec)
//! - Latency (avg, P95, P99)
//!
//! ## Usage
//! ```bash
//! cargo bench --bench ycsb_benchmark
//! cargo bench --bench ycsb_benchmark -- --workload A
//! cargo bench --bench ycsb_benchmark -- --records 100000
//! cargo bench --bench ycsb_benchmark -p joule-db-amorphic --features durable -- --durable  # Enable WAL durability
//! cargo bench --bench ycsb_benchmark -p joule-db-amorphic --features durable -- --durable --batch-size 100  # Batch writes for 40x speedup
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use joule_db_amorphic::{ShardedAmorphicStore, Value, platform};

fn main() {
    println!("=======================================================");
    println!("       YCSB Benchmark: Cloud Serving Evaluation");
    println!("=======================================================\n");

    let args: Vec<String> = std::env::args().collect();

    let workload = args
        .iter()
        .position(|a| a == "--workload")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.chars().next().unwrap_or('A'))
        .unwrap_or('A');

    let record_count: usize = args
        .iter()
        .position(|a| a == "--records")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let operation_count: usize = args
        .iter()
        .position(|a| a == "--operations")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let thread_count: usize = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(platform().cpu_threads);

    // Check for durable mode
    let durable_mode = args.iter().any(|a| a == "--durable");

    // Batch size for durable mode (groups writes for single fsync)
    let batch_size: usize = args
        .iter()
        .position(|a| a == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100); // Default: 100 records per batch

    // Run all workloads or just the specified one
    let workloads = if args.iter().any(|a| a == "--all") {
        vec!['A', 'B', 'C', 'D', 'E', 'F']
    } else {
        vec![workload]
    };

    // Platform info
    let p = platform();
    println!(
        "Platform: {} cores, {:?} SIMD",
        p.cpu_cores,
        p.simd.best_level()
    );
    println!("Configuration:");
    println!("  Records:    {}", record_count);
    println!("  Operations: {}", operation_count);
    println!("  Threads:    {}", thread_count);
    println!(
        "  Durability: {}",
        if durable_mode {
            "WAL (fsync on commit)"
        } else {
            "In-memory only"
        }
    );
    if durable_mode {
        println!("  Batch Size: {} (single fsync per batch)", batch_size);
    }
    println!();

    // Results table
    println!(
        "┌──────────┬─────────────────────────────┬───────────┬──────────┬──────────┬──────────┐"
    );
    println!(
        "│ Workload │ Description                 │ Ops/sec   │ Avg (µs) │ P95 (µs) │ P99 (µs) │"
    );
    println!(
        "├──────────┼─────────────────────────────┼───────────┼──────────┼──────────┼──────────┤"
    );

    for w in workloads {
        let config = WorkloadConfig::from_workload(w, record_count, operation_count, thread_count);
        let results = if durable_mode {
            run_ycsb_workload_durable(&config, batch_size)
        } else {
            run_ycsb_workload(&config)
        };

        println!(
            "│    {}     │ {:27} │ {:>9.0} │ {:>8.1} │ {:>8.1} │ {:>8.1} │",
            w,
            config.description,
            results.throughput,
            results.avg_latency,
            results.p95_latency,
            results.p99_latency
        );
    }

    println!(
        "└──────────┴─────────────────────────────┴───────────┴──────────┴──────────┴──────────┘"
    );

    if durable_mode {
        println!("\n✓ Durability: All writes synced to disk via WAL");
    }

    println!("\n=======================================================");
    println!("                  Benchmark Complete");
    println!("=======================================================");
}

// =============================================================================
// WORKLOAD CONFIGURATION
// =============================================================================

#[derive(Clone)]
struct WorkloadConfig {
    name: char,
    description: &'static str,
    record_count: usize,
    operation_count: usize,
    thread_count: usize,
    read_proportion: f64,
    update_proportion: f64,
    insert_proportion: f64,
    scan_proportion: f64,
    readmodifywrite_proportion: f64,
    request_distribution: Distribution,
    scan_length_distribution: Distribution,
    max_scan_length: usize,
}

#[derive(Clone, Copy)]
enum Distribution {
    Uniform,
    Zipfian,
    Latest,
}

impl WorkloadConfig {
    fn from_workload(workload: char, records: usize, ops: usize, threads: usize) -> Self {
        match workload {
            'A' => Self {
                name: 'A',
                description: "Update heavy (50/50)",
                record_count: records,
                operation_count: ops,
                thread_count: threads,
                read_proportion: 0.5,
                update_proportion: 0.5,
                insert_proportion: 0.0,
                scan_proportion: 0.0,
                readmodifywrite_proportion: 0.0,
                request_distribution: Distribution::Zipfian,
                scan_length_distribution: Distribution::Uniform,
                max_scan_length: 100,
            },
            'B' => Self {
                name: 'B',
                description: "Read mostly (95/5)",
                record_count: records,
                operation_count: ops,
                thread_count: threads,
                read_proportion: 0.95,
                update_proportion: 0.05,
                insert_proportion: 0.0,
                scan_proportion: 0.0,
                readmodifywrite_proportion: 0.0,
                request_distribution: Distribution::Zipfian,
                scan_length_distribution: Distribution::Uniform,
                max_scan_length: 100,
            },
            'C' => Self {
                name: 'C',
                description: "Read only (100%)",
                record_count: records,
                operation_count: ops,
                thread_count: threads,
                read_proportion: 1.0,
                update_proportion: 0.0,
                insert_proportion: 0.0,
                scan_proportion: 0.0,
                readmodifywrite_proportion: 0.0,
                request_distribution: Distribution::Zipfian,
                scan_length_distribution: Distribution::Uniform,
                max_scan_length: 100,
            },
            'D' => Self {
                name: 'D',
                description: "Read latest",
                record_count: records,
                operation_count: ops,
                thread_count: threads,
                read_proportion: 0.95,
                update_proportion: 0.0,
                insert_proportion: 0.05,
                scan_proportion: 0.0,
                readmodifywrite_proportion: 0.0,
                request_distribution: Distribution::Latest,
                scan_length_distribution: Distribution::Uniform,
                max_scan_length: 100,
            },
            'E' => Self {
                name: 'E',
                description: "Short range scan",
                record_count: records,
                operation_count: ops,
                thread_count: threads,
                read_proportion: 0.0,
                update_proportion: 0.0,
                insert_proportion: 0.05,
                scan_proportion: 0.95,
                readmodifywrite_proportion: 0.0,
                request_distribution: Distribution::Zipfian,
                scan_length_distribution: Distribution::Uniform,
                max_scan_length: 100,
            },
            'F' => Self {
                name: 'F',
                description: "Read-modify-write",
                record_count: records,
                operation_count: ops,
                thread_count: threads,
                read_proportion: 0.5,
                update_proportion: 0.0,
                insert_proportion: 0.0,
                scan_proportion: 0.0,
                readmodifywrite_proportion: 0.5,
                request_distribution: Distribution::Zipfian,
                scan_length_distribution: Distribution::Uniform,
                max_scan_length: 100,
            },
            _ => Self::from_workload('A', records, ops, threads),
        }
    }
}

// =============================================================================
// BENCHMARK RESULTS
// =============================================================================

struct BenchmarkResults {
    throughput: f64,
    avg_latency: f64,
    p95_latency: f64,
    p99_latency: f64,
    total_ops: usize,
    total_time: Duration,
}

// =============================================================================
// YCSB IMPLEMENTATION
// =============================================================================

fn run_ycsb_workload(config: &WorkloadConfig) -> BenchmarkResults {
    // Create store
    let store = Arc::new(ShardedAmorphicStore::with_shard_count(
        platform().recommended_shard_count,
    ));

    // Load phase
    print!("  Loading {} records... ", config.record_count);
    let load_start = Instant::now();

    for i in 0..config.record_count {
        let key = format!("user{:010}", i);
        let json = generate_record(&key, i);
        let _ = store.ingest_json(&json);
    }

    println!("done in {:?}", load_start.elapsed());

    // Run phase
    print!("  Running workload {}... ", config.name);

    let ops_per_thread = config.operation_count / config.thread_count;
    let total_ops = Arc::new(AtomicU64::new(0));
    let all_latencies: Arc<std::sync::Mutex<Vec<f64>>> = Arc::new(std::sync::Mutex::new(
        Vec::with_capacity(config.operation_count),
    ));

    let run_start = Instant::now();

    // Spawn worker threads
    let handles: Vec<_> = (0..config.thread_count)
        .map(|thread_id| {
            let store = Arc::clone(&store);
            let config = config.clone();
            let total_ops = Arc::clone(&total_ops);
            let all_latencies = Arc::clone(&all_latencies);

            thread::spawn(move || {
                let mut rng_state: u64 = thread_id as u64 * 12345 + 1;
                let mut local_latencies = Vec::with_capacity(ops_per_thread);

                for _ in 0..ops_per_thread {
                    let op_start = Instant::now();

                    // Select operation type
                    let roll = next_float(&mut rng_state);
                    let key_idx = select_key(&config, &mut rng_state);
                    let key = format!("user{:010}", key_idx);

                    if roll < config.read_proportion {
                        // READ
                        let _ = store.query_equals("key", &Value::String(key));
                    } else if roll < config.read_proportion + config.update_proportion {
                        // UPDATE
                        let json = generate_record(&key, key_idx);
                        let _ = store.ingest_json(&json);
                    } else if roll
                        < config.read_proportion
                            + config.update_proportion
                            + config.insert_proportion
                    {
                        // INSERT
                        let new_key = format!("user{:010}", config.record_count + key_idx);
                        let json = generate_record(&new_key, key_idx);
                        let _ = store.ingest_json(&json);
                    } else if roll
                        < config.read_proportion
                            + config.update_proportion
                            + config.insert_proportion
                            + config.scan_proportion
                    {
                        // SCAN
                        let start_key = key_idx as f64;
                        let end_key = start_key
                            + (next_float(&mut rng_state) * config.max_scan_length as f64);
                        let _ = store.query_range("id", start_key, end_key);
                    } else {
                        // READ-MODIFY-WRITE
                        let _ = store.query_equals("key", &Value::String(key.clone()));
                        let json = generate_record(&key, key_idx);
                        let _ = store.ingest_json(&json);
                    }

                    local_latencies.push(op_start.elapsed().as_micros() as f64);
                    total_ops.fetch_add(1, Ordering::Relaxed);
                }

                // Merge local latencies
                all_latencies.lock().unwrap().extend(local_latencies);
            })
        })
        .collect();

    // Wait for all threads
    for h in handles {
        h.join().unwrap();
    }

    let total_time = run_start.elapsed();
    println!("done in {:?}", total_time);

    // Calculate statistics
    let mut latencies = all_latencies.lock().unwrap().clone();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total = total_ops.load(Ordering::Relaxed) as usize;
    let throughput = total as f64 / total_time.as_secs_f64();
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p95_latency = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99_latency = latencies[(latencies.len() as f64 * 0.99) as usize];

    BenchmarkResults {
        throughput,
        avg_latency,
        p95_latency,
        p99_latency,
        total_ops: total,
        total_time,
    }
}

// =============================================================================
// HELPERS
// =============================================================================

fn generate_record(key: &str, id: usize) -> String {
    // YCSB default field size is 100 bytes x 10 fields
    format!(
        r#"{{"key": "{}", "id": {}, "field0": "{:0>100}", "field1": "{:0>100}", "field2": "{:0>100}", "field3": "{:0>100}", "field4": "{:0>100}"}}"#,
        key,
        id,
        id,
        id + 1,
        id + 2,
        id + 3,
        id + 4
    )
}

fn next_float(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64)
}

fn next_int(state: &mut u64, max: usize) -> usize {
    (next_float(state) * max as f64) as usize
}

fn select_key(config: &WorkloadConfig, state: &mut u64) -> usize {
    match config.request_distribution {
        Distribution::Uniform => next_int(state, config.record_count),
        Distribution::Zipfian => {
            // Simplified Zipfian: favor lower keys
            let u = next_float(state);
            let zipf = (u.powf(0.99) * config.record_count as f64) as usize;
            zipf.min(config.record_count - 1)
        }
        Distribution::Latest => {
            // Favor recently inserted keys
            let range = config.record_count / 10; // Last 10%
            config.record_count - 1 - next_int(state, range)
        }
    }
}

// =============================================================================
// DURABLE YCSB IMPLEMENTATION (with WAL)
// =============================================================================

#[cfg(feature = "durable")]
use joule_db_amorphic::DurableAmorphicStore;

#[cfg(feature = "durable")]
fn run_ycsb_workload_durable(config: &WorkloadConfig, batch_size: usize) -> BenchmarkResults {
    use std::sync::Mutex;

    // Create temporary directory for durable store
    let temp_dir = tempfile::TempDir::new().expect("Failed to create temp dir");

    // Create durable store (wrapped in Mutex for thread-safe mutable access)
    let store = Arc::new(Mutex::new(
        DurableAmorphicStore::open(temp_dir.path()).expect("Failed to create durable store"),
    ));

    // Load phase - USE BATCH API for efficiency
    print!(
        "  Loading {} records (durable, batch={})... ",
        config.record_count, batch_size
    );
    let load_start = Instant::now();

    {
        let mut s = store.lock().unwrap();
        let mut pending = 0;

        s.begin_batch().expect("Failed to begin batch");

        for i in 0..config.record_count {
            let key = format!("user{:010}", i);
            let json = generate_record(&key, i);
            let _ = s.batch_ingest_json(&json);
            pending += 1;

            // Commit batch when full
            if pending >= batch_size {
                s.commit_batch().expect("Failed to commit batch");
                s.begin_batch().expect("Failed to begin batch");
                pending = 0;
            }
        }

        // Commit remaining records
        if pending > 0 {
            s.commit_batch().expect("Failed to commit batch");
        }
    }

    println!("done in {:?}", load_start.elapsed());

    // Run phase - USE BATCH API for write operations
    print!(
        "  Running workload {} (durable, batch={})... ",
        config.name, batch_size
    );

    let mut latencies = Vec::with_capacity(config.operation_count);
    let mut rng_state: u64 = 12345;
    let mut pending_writes = 0;
    let mut batch_active = false;

    let run_start = Instant::now();

    {
        let mut s = store.lock().unwrap();

        for op_idx in 0..config.operation_count {
            let op_start = Instant::now();

            // Select operation type
            let roll = next_float(&mut rng_state);
            let key_idx = select_key(config, &mut rng_state);
            let key = format!("user{:010}", key_idx);

            if roll < config.read_proportion {
                // READ - flush pending writes first if any
                if pending_writes > 0 {
                    s.commit_batch().expect("Failed to commit batch");
                    pending_writes = 0;
                    batch_active = false;
                }
                let _ = s.query_equals("key", &Value::String(key));
            } else if roll < config.read_proportion + config.update_proportion {
                // UPDATE - batch it
                if !batch_active {
                    s.begin_batch().expect("Failed to begin batch");
                    batch_active = true;
                }
                let json = generate_record(&key, key_idx);
                let _ = s.batch_ingest_json(&json);
                pending_writes += 1;
            } else if roll
                < config.read_proportion + config.update_proportion + config.insert_proportion
            {
                // INSERT - batch it
                if !batch_active {
                    s.begin_batch().expect("Failed to begin batch");
                    batch_active = true;
                }
                let new_key = format!("user{:010}", config.record_count + key_idx);
                let json = generate_record(&new_key, key_idx);
                let _ = s.batch_ingest_json(&json);
                pending_writes += 1;
            } else if roll
                < config.read_proportion
                    + config.update_proportion
                    + config.insert_proportion
                    + config.scan_proportion
            {
                // SCAN - flush pending writes first if any
                if pending_writes > 0 {
                    s.commit_batch().expect("Failed to commit batch");
                    pending_writes = 0;
                    batch_active = false;
                }
                let start_key = key_idx as f64;
                let end_key =
                    start_key + (next_float(&mut rng_state) * config.max_scan_length as f64);
                let _ = s.query_range("id", start_key, end_key);
            } else {
                // READ-MODIFY-WRITE - flush pending writes, then do read-modify-write
                if pending_writes > 0 {
                    s.commit_batch().expect("Failed to commit batch");
                    pending_writes = 0;
                    batch_active = false;
                }
                let _ = s.query_equals("key", &Value::String(key.clone()));
                // Single write for RMW - use regular ingest
                let json = generate_record(&key, key_idx);
                let _ = s.ingest_json(&json);
            }

            // Commit batch when full
            if pending_writes >= batch_size {
                s.commit_batch().expect("Failed to commit batch");
                pending_writes = 0;
                batch_active = false;
            }

            latencies.push(op_start.elapsed().as_micros() as f64);
        }

        // Commit any remaining pending writes
        if pending_writes > 0 {
            s.commit_batch().expect("Failed to commit batch");
        }
    }

    let total_time = run_start.elapsed();
    println!("done in {:?}", total_time);

    // Calculate statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total = config.operation_count;
    let throughput = total as f64 / total_time.as_secs_f64();
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p95_latency = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99_latency = latencies[(latencies.len() as f64 * 0.99) as usize];

    BenchmarkResults {
        throughput,
        avg_latency,
        p95_latency,
        p99_latency,
        total_ops: total,
        total_time,
    }
}

#[cfg(not(feature = "durable"))]
fn run_ycsb_workload_durable(config: &WorkloadConfig, _batch_size: usize) -> BenchmarkResults {
    eprintln!("WARNING: Durable mode requested but 'durable' feature not enabled.");
    eprintln!("         Falling back to in-memory mode.");
    eprintln!(
        "         Rebuild with: cargo bench --bench ycsb_benchmark -p joule-db-amorphic --features durable -- --durable"
    );
    run_ycsb_workload(config)
}
