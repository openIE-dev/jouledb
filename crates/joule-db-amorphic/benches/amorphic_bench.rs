use criterion::{Criterion, black_box, criterion_group, criterion_main};
use joule_db_amorphic::AmorphicStore;

fn bench_json_ingestion(c: &mut Criterion) {
    c.bench_function("ingest_json", |b| {
        let mut store = AmorphicStore::new();
        b.iter(|| {
            store.ingest_json(black_box(
                r#"{"name": "Test", "value": 42, "active": true}"#,
            ))
        });
    });
}

fn bench_row_ingestion(c: &mut Criterion) {
    c.bench_function("ingest_row", |b| {
        let mut store = AmorphicStore::new();
        b.iter(|| {
            store.ingest_row(
                black_box(&["name", "age", "city"]),
                black_box(&["Alice", "30", "NYC"]),
            )
        });
    });
}

fn bench_query_range(c: &mut Criterion) {
    let mut store = AmorphicStore::new();
    for i in 0..1000 {
        store
            .ingest_json(&format!(r#"{{"id": {}, "value": {}}}"#, i, i * 10))
            .unwrap();
    }

    c.bench_function("query_range_1000", |b| {
        b.iter(|| store.query_range(black_box("value"), black_box(2500.0), black_box(7500.0)));
    });
}

fn bench_similarity_search(c: &mut Criterion) {
    let mut store = AmorphicStore::new();
    for i in 0..100 {
        store
            .ingest_json(&format!(
                r#"{{"name": "Item{}", "category": "cat{}", "value": {}}}"#,
                i,
                i % 10,
                i * 10
            ))
            .unwrap();
    }

    c.bench_function("similarity_search_100", |b| {
        b.iter(|| store.query_similar_to(black_box("Item50"), black_box(5)));
    });
}

criterion_group!(
    benches,
    bench_json_ingestion,
    bench_row_ingestion,
    bench_query_range,
    bench_similarity_search
);
criterion_main!(benches);
