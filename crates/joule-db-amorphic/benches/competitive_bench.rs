//! Competitive Benchmarks
//!
//! Measures amorphic database performance against targets set by leading databases:
//! - Redis: ~100K-1M ops/sec for simple KV
//! - SQLite: ~50K reads/sec, ~5K writes/sec
//! - Pinecone: ~10K QPS for ANN search
//! - Neo4j: ~100K traversals/sec

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use joule_db_amorphic::{AmorphicStore, Value};
use std::collections::HashMap;

// ============================================================================
// INGESTION BENCHMARKS (Target: Redis SET ~100K/sec)
// ============================================================================

fn bench_json_ingestion_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingestion");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("json", size), size, |b, &size| {
            b.iter(|| {
                let mut store = AmorphicStore::new();
                for i in 0..size {
                    store
                        .ingest_json(&format!(
                            r#"{{"id": {}, "name": "item{}", "value": {}, "active": true}}"#,
                            i,
                            i,
                            i * 10
                        ))
                        .unwrap();
                }
                store
            });
        });
    }
    group.finish();
}

fn bench_row_ingestion_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingestion");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("row", size), size, |b, &size| {
            b.iter(|| {
                let mut store = AmorphicStore::new();
                for i in 0..size {
                    store
                        .ingest_row(
                            &["id", "name", "value"],
                            &[&i.to_string(), &format!("item{}", i), &(i * 10).to_string()],
                        )
                        .unwrap();
                }
                store
            });
        });
    }
    group.finish();
}

// ============================================================================
// POINT QUERY BENCHMARKS (Target: Redis GET ~100K/sec)
// ============================================================================

fn bench_exact_lookup(c: &mut Criterion) {
    // Pre-populate store
    let mut store = AmorphicStore::new();
    for i in 0..10000 {
        store
            .ingest_json(&format!(
                r#"{{"id": {}, "name": "item{}", "category": "cat{}"}}"#,
                i,
                i,
                i % 100
            ))
            .unwrap();
    }

    let mut group = c.benchmark_group("point_query");
    group.throughput(Throughput::Elements(1));

    group.bench_function("exact_match", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let target = format!("item{}", i % 10000);
            let result = store.query_equals("name", &Value::String(target));
            i += 1;
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// RANGE QUERY BENCHMARKS (Target: SQLite ~50K/sec)
// ============================================================================

fn bench_range_query(c: &mut Criterion) {
    let mut store = AmorphicStore::new();
    for i in 0..10000 {
        store
            .ingest_json(&format!(
                r#"{{"id": {}, "score": {}, "timestamp": {}}}"#,
                i,
                i % 1000,
                1700000000 + i
            ))
            .unwrap();
    }

    let mut group = c.benchmark_group("range_query");
    group.throughput(Throughput::Elements(1));

    // Narrow range (should return ~10 records)
    group.bench_function("narrow_range", |b| {
        b.iter(|| {
            let result = store.query_range("score", black_box(500.0), black_box(510.0));
            black_box(result)
        });
    });

    // Wide range (should return ~500 records)
    group.bench_function("wide_range", |b| {
        b.iter(|| {
            let result = store.query_range("score", black_box(0.0), black_box(500.0));
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// SIMILARITY SEARCH BENCHMARKS (Target: Pinecone ~10K QPS)
// ============================================================================

fn bench_similarity_search(c: &mut Criterion) {
    let mut store = AmorphicStore::new();

    // Create diverse dataset
    for i in 0..10000 {
        store
            .ingest_json(&format!(
                r#"{{"name": "entity{}", "type": "type{}", "region": "region{}", "score": {}}}"#,
                i,
                i % 50,
                i % 20,
                i % 1000
            ))
            .unwrap();
    }

    let mut group = c.benchmark_group("similarity");
    group.throughput(Throughput::Elements(1));

    // k-NN search
    group.bench_function("knn_top5", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 10000);
            let result = store.query_similar_to(&query, 5);
            i += 1;
            black_box(result)
        });
    });

    group.bench_function("knn_top20", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 10000);
            let result = store.query_similar_to(&query, 20);
            i += 1;
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// GPU-ACCELERATED SIMILARITY
// ============================================================================

fn bench_similarity_search_gpu(c: &mut Criterion) {
    use joule_db_amorphic::{GpuContext, GpuVectorStore};

    let mut store = AmorphicStore::new();

    // Create diverse dataset
    for i in 0..10000 {
        store
            .ingest_json(&format!(
                r#"{{"name": "entity{}", "type": "type{}", "region": "region{}", "score": {}}}"#,
                i,
                i % 50,
                i % 20,
                i % 1000
            ))
            .unwrap();
    }

    // Initialize GPU context
    let gpu = match GpuContext::new_sync() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU not available: {:?}", e);
            return;
        }
    };

    // Pre-upload vectors to GPU (one-time cost)
    let gpu_store = match store.create_gpu_store(&gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create GPU store: {:?}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("similarity_gpu");
    group.throughput(Throughput::Elements(1));

    // Slow path: re-upload vectors each query (for comparison)
    group.bench_function("knn_top5_gpu_slow", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 10000);
            let result = store.query_similar_to_gpu(&query, 5, &gpu);
            i += 1;
            black_box(result)
        });
    });

    // Fast path: vectors pre-uploaded to GPU
    group.bench_function("knn_top5_gpu_fast", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 10000);
            let result = store.query_similar_to_gpu_fast(&query, 5, &gpu, &gpu_store);
            i += 1;
            black_box(result)
        });
    });

    group.bench_function("knn_top20_gpu_fast", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 10000);
            let result = store.query_similar_to_gpu_fast(&query, 20, &gpu, &gpu_store);
            i += 1;
            black_box(result)
        });
    });

    group.finish();
}

// GPU benchmark at scale (100K vectors) - where GPU should shine
fn bench_similarity_search_gpu_100k(c: &mut Criterion) {
    use joule_db_amorphic::GpuContext;

    eprintln!("Building 100K vector dataset...");
    let mut store = AmorphicStore::new();

    // Create 100K vectors - GPU should win at this scale
    for i in 0..100_000 {
        store
            .ingest_json(&format!(
                r#"{{"name": "entity{}", "type": "type{}", "region": "region{}", "score": {}}}"#,
                i,
                i % 50,
                i % 20,
                i % 1000
            ))
            .unwrap();
    }
    eprintln!("Dataset created.");

    // Initialize GPU context
    let gpu = match GpuContext::new_sync() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU not available: {:?}", e);
            return;
        }
    };

    // Pre-upload vectors to GPU
    eprintln!("Uploading vectors to GPU...");
    let gpu_store = match store.create_gpu_store(&gpu) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create GPU store: {:?}", e);
            return;
        }
    };
    eprintln!("GPU store ready.");

    let mut group = c.benchmark_group("similarity_gpu_100k");
    group.throughput(Throughput::Elements(1));
    group.sample_size(50); // Fewer samples for slow benchmarks

    // CPU baseline at 100K scale
    group.bench_function("cpu_knn_top5", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 100_000);
            let result = store.query_similar_to(&query, 5);
            i += 1;
            black_box(result)
        });
    });

    // GPU fast path at 100K scale
    group.bench_function("gpu_knn_top5_fast", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let query = format!("entity{}", i % 100_000);
            let result = store.query_similar_to_gpu_fast(&query, 5, &gpu, &gpu_store);
            i += 1;
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// GRAPH TRAVERSAL BENCHMARKS (Target: Neo4j ~100K/sec)
// ============================================================================

fn bench_graph_traversal(c: &mut Criterion) {
    let mut store = AmorphicStore::new();

    // Create a social network-like graph
    // 1000 people, each knows ~10 others
    for i in 0..1000 {
        store
            .ingest_json(&format!(r#"{{"name": "person{}"}}"#, i))
            .unwrap();
    }

    for i in 0..1000 {
        for j in 0..10 {
            let target = (i * 7 + j * 13) % 1000; // Pseudo-random connections
            if target != i {
                store
                    .ingest_edge(
                        &format!("person{}", i),
                        "KNOWS",
                        &format!("person{}", target),
                    )
                    .unwrap();
            }
        }
    }

    let mut group = c.benchmark_group("graph");
    group.throughput(Throughput::Elements(1));

    // 1-hop traversal
    group.bench_function("traverse_depth1", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let start = format!("person{}", i % 1000);
            let result = store.query_graph(&start, "KNOWS", 1);
            i += 1;
            black_box(result)
        });
    });

    // 2-hop traversal
    group.bench_function("traverse_depth2", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let start = format!("person{}", i % 1000);
            let result = store.query_graph(&start, "KNOWS", 2);
            i += 1;
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// SQL-STYLE QUERY BENCHMARKS
// ============================================================================

fn bench_sql_queries(c: &mut Criterion) {
    let mut store = AmorphicStore::new();
    for i in 0..10000 {
        store
            .ingest_json(&format!(
                r#"{{"id": {}, "age": {}, "salary": {}, "department": "dept{}"}}"#,
                i,
                20 + (i % 50),
                30000 + (i % 100) * 1000,
                i % 10
            ))
            .unwrap();
    }

    let mut group = c.benchmark_group("sql");
    group.throughput(Throughput::Elements(1));

    group.bench_function("select_where_gt", |b| {
        b.iter(|| {
            let result = store.query_sql("SELECT * WHERE age > 40");
            black_box(result)
        });
    });

    group.bench_function("select_where_lt", |b| {
        b.iter(|| {
            let result = store.query_sql("SELECT * WHERE salary < 50000");
            black_box(result)
        });
    });

    group.finish();
}

// ============================================================================
// MIXED WORKLOAD BENCHMARK
// ============================================================================

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed");
    group.throughput(Throughput::Elements(100)); // 100 ops per iteration

    group.bench_function("read_heavy_90_10", |b| {
        let mut store = AmorphicStore::new();
        // Pre-populate
        for i in 0..1000 {
            store
                .ingest_json(&format!(r#"{{"id": {}, "val": {}}}"#, i, i))
                .unwrap();
        }

        let mut op_count = 0u64;
        b.iter(|| {
            for _ in 0..100 {
                if op_count % 10 == 0 {
                    // 10% writes
                    store
                        .ingest_json(&format!(r#"{{"id": {}, "val": {}}}"#, op_count, op_count))
                        .unwrap();
                } else {
                    // 90% reads
                    let _ = store.query_range(
                        "id",
                        (op_count % 1000) as f64,
                        (op_count % 1000 + 10) as f64,
                    );
                }
                op_count += 1;
            }
        });
    });

    group.bench_function("write_heavy_10_90", |b| {
        let mut store = AmorphicStore::new();

        let mut op_count = 0u64;
        b.iter(|| {
            for _ in 0..100 {
                if op_count % 10 != 0 {
                    // 90% writes
                    store
                        .ingest_json(&format!(r#"{{"id": {}, "val": {}}}"#, op_count, op_count))
                        .unwrap();
                } else {
                    // 10% reads
                    let _ = store.query_range("id", 0.0, 100.0);
                }
                op_count += 1;
            }
        });
    });

    group.finish();
}

// ============================================================================
// HOLOGRAM OPERATIONS (Core HDC)
// ============================================================================

fn bench_hologram_ops(c: &mut Criterion) {
    use joule_db_hdc::{BinaryHV, BundleAccumulator};

    let mut group = c.benchmark_group("hdc_core");

    let dim = 10000;
    let v1 = BinaryHV::random(dim, 42);
    let v2 = BinaryHV::random(dim, 43);

    group.bench_function("bind_xor", |b| {
        b.iter(|| black_box(v1.bind(&v2)));
    });

    group.bench_function("similarity", |b| {
        b.iter(|| black_box(v1.similarity(&v2)));
    });

    group.bench_function("permute", |b| {
        b.iter(|| black_box(v1.permute_words(100)));
    });

    group.bench_function("bundle_10", |b| {
        let vectors: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(dim, i)).collect();
        b.iter(|| {
            let mut acc = BundleAccumulator::new(dim);
            for v in &vectors {
                acc.add(v);
            }
            black_box(acc.threshold())
        });
    });

    group.finish();
}

// ============================================================================
// TIERED STORAGE BENCHMARKS (Hot/Warm/Cold)
// ============================================================================

fn bench_tiered_storage(c: &mut Criterion) {
    use joule_db_amorphic::tiered::TieredStore;
    use joule_db_hdc::BinaryHV;

    // Use in-memory store for clean hot tier benchmarks
    let mut store = TieredStore::in_memory();

    // Pre-populate with 1000 holograms (all in hot tier)
    for i in 0..1000u64 {
        let hv = BinaryHV::random(10000, i);
        store.put(i, hv).unwrap();
    }

    let mut group = c.benchmark_group("tiered_storage");
    group.throughput(Throughput::Elements(1));

    // Hot tier access (all items are hot in memory-only mode)
    group.bench_function("hot_tier_peek", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let result = store.peek(i % 1000);
            i += 1;
            black_box(result)
        });
    });

    // Put operation (no demotion in memory-only mode)
    group.bench_function("hot_put", |b| {
        let mut i = 1000u64;
        b.iter(|| {
            let hv = BinaryHV::random(10000, i);
            store.put(i, hv).unwrap();
            i += 1;
            black_box(())
        });
    });

    group.finish();
}

// Non-GPU benchmarks
criterion_group!(
    benches,
    bench_json_ingestion_throughput,
    bench_row_ingestion_throughput,
    bench_exact_lookup,
    bench_range_query,
    bench_similarity_search,
    bench_graph_traversal,
    bench_sql_queries,
    bench_mixed_workload,
    bench_hologram_ops,
    bench_tiered_storage,
);

// GPU benchmarks
criterion_group!(
    gpu_benches,
    bench_similarity_search_gpu,
    bench_similarity_search_gpu_100k,
);

criterion_main!(benches, gpu_benches);
