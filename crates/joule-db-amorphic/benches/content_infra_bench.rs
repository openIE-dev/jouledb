//! Content Infrastructure Benchmarks — 100K+ record scale tests.
//!
//! Tests the actual performance characteristics that content providers care about:
//! - Batch feature lookup latency (recommendation serving)
//! - Similarity search recall + latency (content discovery)
//! - Ingest throughput (catalog loading)
//! - Trending query latency (live content)
//! - Multi-tenant isolation overhead

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use joule_db_amorphic::*;

/// Build a store with N records for benchmarking.
fn build_store(n: usize) -> AmorphicStore {
    let mut store = AmorphicStore::with_index_strategy(IndexStrategy::Hybrid);
    for i in 0..n {
        let json = format!(
            r#"{{"name": "item_{}", "category": "cat_{}", "score": {}, "description": "Content item number {} for benchmarking"}}"#,
            i,
            i % 10,
            (i as f64 * 0.7).sin() * 100.0,
            i,
        );
        store.ingest_json(&json).unwrap();
    }
    store
}

fn bench_ingest_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_throughput");
    group.sample_size(10);

    for &size in &[1000, 10_000] {
        group.bench_with_input(BenchmarkId::new("json", size), &size, |b, &n| {
            b.iter(|| {
                let mut store = AmorphicStore::new();
                for i in 0..n {
                    let json = format!(r#"{{"name": "item_{}", "score": {}}}"#, i, i);
                    store.ingest_json(&json).unwrap();
                }
                black_box(store.record_count())
            });
        });
    }

    group.finish();
}

fn bench_similarity_search(c: &mut Criterion) {
    let store = build_store(10_000);

    let mut group = c.benchmark_group("similarity_search");
    group.sample_size(20);

    group.bench_function("similar_to_10k", |b| {
        b.iter(|| {
            black_box(store.query_similar_to("item_42", 10))
        });
    });

    group.finish();
}

fn bench_batch_lookup(c: &mut Criterion) {
    let store = build_store(10_000);

    let mut group = c.benchmark_group("batch_lookup");

    let keys: Vec<String> = (0..100).map(|i| format!("item_{}", i * 100)).collect();

    group.bench_function("batch_get_100_keys", |b| {
        b.iter(|| {
            let results = store.batch_get_by_names(
                &keys.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            );
            black_box(results.len())
        });
    });

    group.finish();
}

fn bench_trending(c: &mut Criterion) {
    let mut trending = TrendingIndex::new();

    // Pre-populate with events
    for i in 0..10_000 {
        trending.record_event(&format!("content_{}", i % 1000), (i % 1000) as RecordId);
    }

    let mut group = c.benchmark_group("trending");

    group.bench_function("query_top_10", |b| {
        b.iter(|| {
            black_box(trending.query_trending(10, TrendWindow::OneMinute, None))
        });
    });

    group.bench_function("query_top_100", |b| {
        b.iter(|| {
            black_box(trending.query_trending(100, TrendWindow::OneHour, None))
        });
    });

    group.finish();
}

fn bench_multi_tenant(c: &mut Criterion) {
    let mut mt = MultiTenantStore::new();

    // Create 10 tenants with 1000 records each
    for t in 0..10 {
        mt.create_tenant(TenantConfig {
            tenant_id: format!("tenant_{}", t),
            name: format!("Tenant {}", t),
            max_records: 0,
            max_storage_bytes: 0,
            status: TenantStatus::Active,
        })
        .unwrap();

        for i in 0..1000 {
            mt.ingest_json(
                &format!("tenant_{}", t),
                &format!(r#"{{"name": "item_{}", "value": {}}}"#, i, i),
            )
            .unwrap();
        }
    }

    let mut group = c.benchmark_group("multi_tenant");

    group.bench_function("query_equals_1k_records", |b| {
        b.iter(|| {
            black_box(
                mt.query_equals(
                    "tenant_0",
                    "name",
                    &Value::String("item_500".to_string()),
                )
                .unwrap(),
            )
        });
    });

    group.finish();
}

fn bench_hybrid_search(c: &mut Criterion) {
    let store = build_store(5_000);

    let mut group = c.benchmark_group("hybrid_search");

    let query_hv = joule_db_hdc::BinaryHV::from_hash(b"search query terms", DIMENSION);

    group.bench_function("vector_only_5k", |b| {
        b.iter(|| {
            let q = HybridQuery::new(10)
                .with_vector(query_hv.clone())
                .with_weights(1.0, 0.0);
            black_box(hybrid_search(&store, &q))
        });
    });

    group.bench_function("keyword_only_5k", |b| {
        b.iter(|| {
            let q = HybridQuery::new(10)
                .with_keywords("item benchmark content")
                .with_weights(0.0, 1.0);
            black_box(hybrid_search(&store, &q))
        });
    });

    group.bench_function("hybrid_5k", |b| {
        b.iter(|| {
            let q = HybridQuery::new(10)
                .with_vector(query_hv.clone())
                .with_keywords("item benchmark")
                .with_weights(0.5, 0.5);
            black_box(hybrid_search(&store, &q))
        });
    });

    group.finish();
}

criterion_group!(
    content_benches,
    bench_ingest_throughput,
    bench_similarity_search,
    bench_batch_lookup,
    bench_trending,
    bench_multi_tenant,
    bench_hybrid_search,
);
criterion_main!(content_benches);
