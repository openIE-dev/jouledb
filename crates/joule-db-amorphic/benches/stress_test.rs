//! Stress test benchmarks for JouleDB V1 implementation
//!
//! Tests:
//! - ShardedAmorphicStore concurrent write throughput
//! - SIMD Hamming distance performance
//! - Semantic search performance
//! - Platform-optimized operations

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use joule_db_amorphic::{
    AmorphicStore, ShardedAmorphicStore, Value, hamming_distance_optimized,
    hamming_distances_batch_optimized, hamming_top_k_optimized, parallel_map, parallel_reduce,
    platform,
};
use joule_db_hdc::simd::{HammingEngine, hamming_distance, hamming_distances_batch};

fn main() {
    println!("=======================================================");
    println!("       JouleDB V1 Stress Test Benchmark");
    println!("=======================================================\n");

    // Show platform detection
    print_platform_info();

    bench_sharded_store_writes();
    bench_sharded_store_concurrent();
    bench_similarity_search();
    bench_simd_hamming();
    bench_parallel_operations();
    bench_query_performance();

    println!("\n=======================================================");
    println!("                  Benchmark Complete");
    println!("=======================================================");
}

/// Print detected platform capabilities
fn print_platform_info() {
    println!("--- Platform Detection ---");
    let p = platform();

    println!("  Architecture:    {:?}", p.arch);
    println!(
        "  CPU Cores:       {} physical, {} threads",
        p.cpu_cores, p.cpu_threads
    );
    println!(
        "  SIMD Level:      {:?} ({}-bit vectors)",
        p.simd.best_level(),
        p.simd.best_level().width_bits()
    );
    println!("  SIMD Features:");
    println!("    - AVX-512F:        {}", p.simd.avx512f);
    println!("    - AVX-512 POPCNT:  {}", p.simd.avx512_vpopcntdq);
    println!("    - AVX2:            {}", p.simd.avx2);
    println!("    - SSE4.2:          {}", p.simd.sse42);
    println!("    - NEON:            {}", p.simd.neon);
    println!("  Fast popcount:   {}", p.simd.has_fast_popcount());
    println!("  Recommended:");
    println!("    - Parallelism: {} threads", p.recommended_parallelism);
    println!("    - Shards:      {}", p.recommended_shard_count);
    println!("    - Batch size:  {}", p.recommended_batch_size);
    println!();
}

/// Benchmark single-threaded write throughput to ShardedAmorphicStore
fn bench_sharded_store_writes() {
    println!("--- ShardedAmorphicStore Single-Threaded Writes ---");

    let store = ShardedAmorphicStore::with_shard_count(8);
    let num_records = 10_000;

    let start = Instant::now();
    for i in 0..num_records {
        let json = format!(
            r#"{{"id": {}, "name": "User{}", "email": "user{}@example.com", "score": {}}}"#,
            i,
            i,
            i,
            i % 100
        );
        store.ingest_json(&json).unwrap();
    }
    let elapsed = start.elapsed();

    let ops_per_sec = num_records as f64 / elapsed.as_secs_f64();
    println!("  Records:     {}", num_records);
    println!("  Time:        {:?}", elapsed);
    println!("  Throughput:  {:.0} ops/sec", ops_per_sec);

    // Check distribution
    let sizes = store.shard_sizes().unwrap();
    let min = sizes.iter().min().unwrap();
    let max = sizes.iter().max().unwrap();
    println!(
        "  Shard dist:  min={}, max={}, ratio={:.2}x\n",
        min,
        max,
        *max as f64 / *min.max(&1) as f64
    );
}

/// Benchmark concurrent write throughput
fn bench_sharded_store_concurrent() {
    println!("--- ShardedAmorphicStore Concurrent Writes ---");

    let num_threads = 8;
    let records_per_thread = 5_000;
    let total_records = num_threads * records_per_thread;

    let store = Arc::new(ShardedAmorphicStore::with_shard_count(16));

    let start = Instant::now();
    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let store = Arc::clone(&store);
            thread::spawn(move || {
                for i in 0..records_per_thread {
                    let json = format!(
                        r#"{{"thread": {}, "id": {}, "data": "payload_{}_{}"}}"#,
                        t, i, t, i
                    );
                    store.ingest_json(&json).unwrap();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    let elapsed = start.elapsed();

    let ops_per_sec = total_records as f64 / elapsed.as_secs_f64();
    println!("  Threads:     {}", num_threads);
    println!("  Total recs:  {}", total_records);
    println!("  Time:        {:?}", elapsed);
    println!("  Throughput:  {:.0} ops/sec", ops_per_sec);
    println!(
        "  Per-thread:  {:.0} ops/sec\n",
        ops_per_sec / num_threads as f64
    );
}

/// Benchmark similarity search performance
fn bench_similarity_search() {
    println!("--- Similarity Search Performance ---");

    let mut store = AmorphicStore::new();
    let num_records = 1_000;

    // Ingest records
    let start = Instant::now();
    for i in 0..num_records {
        let json = format!(
            r#"{{"id": {}, "title": "Document about topic {} and subject {}", "content": "This is a longer piece of content discussing various aspects of item number {} including technical details and analysis."}}"#,
            i,
            i % 50,
            i % 30,
            i
        );
        store.ingest_json(&json).unwrap();
    }
    let ingest_time = start.elapsed();
    println!("  Ingested {} records in {:?}", num_records, ingest_time);

    // Benchmark similarity queries
    let queries = vec![
        "document topic technical",
        "content analysis details",
        "subject item number",
        "longer piece aspects",
    ];

    let num_iterations = 100;
    let k = 10;

    let start = Instant::now();
    for _ in 0..num_iterations {
        for query in &queries {
            let _ = store.query_similar_to(query, k);
        }
    }
    let elapsed = start.elapsed();

    let total_queries = num_iterations * queries.len();
    let qps = total_queries as f64 / elapsed.as_secs_f64();
    let avg_latency = elapsed / total_queries as u32;

    println!(
        "  Queries:     {} ({} iterations x {} queries)",
        total_queries,
        num_iterations,
        queries.len()
    );
    println!("  Time:        {:?}", elapsed);
    println!("  QPS:         {:.0}", qps);
    println!("  Avg latency: {:?}\n", avg_latency);
}

/// Benchmark SIMD Hamming distance
fn bench_simd_hamming() {
    println!("--- SIMD Hamming Distance Performance ---");

    let engine = HammingEngine::new();
    println!("  SIMD Level:  {:?}", engine.simd_level());

    // Create test vectors (10000 dimensions = 157 u64 words)
    let dim_words = 157; // ~10000 bits
    let num_vectors = 1000;

    let query: Vec<u64> = (0..dim_words)
        .map(|i| (i as u64).wrapping_mul(0x123456789ABCDEF0))
        .collect();

    let targets: Vec<Vec<u64>> = (0..num_vectors)
        .map(|v| {
            (0..dim_words)
                .map(|i| ((v * dim_words + i) as u64).wrapping_mul(0xFEDCBA9876543210))
                .collect()
        })
        .collect();

    // Single distance benchmark
    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hamming_distance(&query, &targets[0]);
    }
    let single_elapsed = start.elapsed();
    let single_ops = iterations as f64 / single_elapsed.as_secs_f64();

    println!("  Single distance:");
    println!("    Iterations: {}", iterations);
    println!("    Time:       {:?}", single_elapsed);
    println!("    Throughput: {:.0} ops/sec", single_ops);

    // Batch distance benchmark
    let refs: Vec<&[u64]> = targets.iter().map(|t| t.as_slice()).collect();
    let batch_iterations = 100;

    let start = Instant::now();
    for _ in 0..batch_iterations {
        let _ = hamming_distances_batch(&query, &refs);
    }
    let batch_elapsed = start.elapsed();
    let batch_comparisons = batch_iterations * num_vectors;
    let batch_ops = batch_comparisons as f64 / batch_elapsed.as_secs_f64();

    println!("  Batch distance ({} vectors):", num_vectors);
    println!("    Iterations: {}", batch_iterations);
    println!("    Time:       {:?}", batch_elapsed);
    println!("    Throughput: {:.0} comparisons/sec\n", batch_ops);
}

/// Benchmark parallel operations
fn bench_parallel_operations() {
    println!("--- Parallel Operations Performance ---");

    let p = platform();

    // Create test data
    let data: Vec<i64> = (0..100_000).collect();

    // Serial map
    let start = Instant::now();
    let _serial: Vec<i64> = data.iter().map(|x| x * x).collect();
    let serial_time = start.elapsed();

    // Parallel map
    let start = Instant::now();
    let _parallel = parallel_map(&data, |x| x * x);
    let parallel_time = start.elapsed();

    let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("  Map operation (100K items):");
    println!("    Serial:   {:?}", serial_time);
    println!(
        "    Parallel: {:?} ({:.1}x speedup)",
        parallel_time, speedup
    );

    // Serial reduce
    let start = Instant::now();
    let _serial_sum: i64 = data.iter().sum();
    let serial_time = start.elapsed();

    // Parallel reduce
    let start = Instant::now();
    let _parallel_sum = parallel_reduce(&data, |x| *x, |a, b| a + b, 0i64);
    let parallel_time = start.elapsed();

    let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();
    println!("  Reduce operation (100K items):");
    println!("    Serial:   {:?}", serial_time);
    println!(
        "    Parallel: {:?} ({:.1}x speedup)",
        parallel_time, speedup
    );

    // Hamming distance batch with optimization
    let dim_words = 157;
    let num_vectors = 1000;

    let query: Vec<u64> = (0..dim_words)
        .map(|i| (i as u64).wrapping_mul(0x123456789ABCDEF0))
        .collect();

    let targets: Vec<Vec<u64>> = (0..num_vectors)
        .map(|v| {
            (0..dim_words)
                .map(|i| ((v * dim_words + i) as u64).wrapping_mul(0xFEDCBA9876543210))
                .collect()
        })
        .collect();

    let refs: Vec<&[u64]> = targets.iter().map(|t| t.as_slice()).collect();

    // Optimized batch
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hamming_distances_batch_optimized(&query, &refs);
    }
    let batch_time = start.elapsed();

    let comparisons = iterations * num_vectors;
    let ops = comparisons as f64 / batch_time.as_secs_f64();
    println!(
        "  Hamming batch ({} vectors x {} iterations):",
        num_vectors, iterations
    );
    println!("    Time:       {:?}", batch_time);
    println!("    Throughput: {:.1}M comparisons/sec", ops / 1_000_000.0);

    // Top-k search
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = hamming_top_k_optimized(&query, &refs, 10);
    }
    let topk_time = start.elapsed();

    let ops = (iterations * num_vectors) as f64 / topk_time.as_secs_f64();
    println!(
        "  Top-10 search ({} vectors x {} iterations):",
        num_vectors, iterations
    );
    println!("    Time:       {:?}", topk_time);
    println!(
        "    Throughput: {:.1}M comparisons/sec\n",
        ops / 1_000_000.0
    );
}

/// Benchmark query performance
fn bench_query_performance() {
    println!("--- Query Performance ---");

    let store = ShardedAmorphicStore::with_shard_count(8);
    let num_records = 10_000;

    // Ingest data
    println!("  Ingesting {} records...", num_records);
    let start = Instant::now();
    for i in 0..num_records {
        let json = format!(
            r#"{{"id": {}, "category": "cat{}", "price": {}, "name": "Product {}"}}"#,
            i,
            i % 100,
            (i % 1000) as f64 * 0.99,
            i
        );
        store.ingest_json(&json).unwrap();
    }
    println!("  Ingest time: {:?}", start.elapsed());

    // Exact match queries
    let iterations = 1000;
    let start = Instant::now();
    for i in 0..iterations {
        let _ = store.query_equals("category", &Value::String(format!("cat{}", i % 100)));
    }
    let exact_elapsed = start.elapsed();
    let exact_qps = iterations as f64 / exact_elapsed.as_secs_f64();

    println!("  Exact match queries:");
    println!("    Queries:   {}", iterations);
    println!("    Time:      {:?}", exact_elapsed);
    println!("    QPS:       {:.0}", exact_qps);

    // Range queries
    let start = Instant::now();
    for i in 0..iterations {
        let min = (i % 100) as f64;
        let max = min + 100.0;
        let _ = store.query_range("price", min, max);
    }
    let range_elapsed = start.elapsed();
    let range_qps = iterations as f64 / range_elapsed.as_secs_f64();

    println!("  Range queries:");
    println!("    Queries:   {}", iterations);
    println!("    Time:      {:?}", range_elapsed);
    println!("    QPS:       {:.0}", range_qps);

    // Similarity queries
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = store.query_similar_to("Product electronics", 10);
    }
    let sim_elapsed = start.elapsed();
    let sim_qps = iterations as f64 / sim_elapsed.as_secs_f64();

    println!("  Similarity queries (k=10):");
    println!("    Queries:   {}", iterations);
    println!("    Time:      {:?}", sim_elapsed);
    println!("    QPS:       {:.0}\n", sim_qps);
}
