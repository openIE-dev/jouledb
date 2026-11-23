//! Performance benchmarks for HTAP Systems

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn bench_format_conversion(c: &mut Criterion) {
    use joule_db_browser::htap::{ColumnDef, ColumnarSchema, ConversionStrategy, FormatConverter};

    let mut group = c.benchmark_group("htap_format_conversion");

    // Create sample B-tree data
    let mut btree_data = Vec::new();
    for i in 0..1000 {
        let key = format!("user_{}", i).into_bytes();
        let value = format!(r#"{{"name":"User{}","age":{}}}"#, i, 20 + (i % 50)).into_bytes();
        btree_data.extend_from_slice(&(key.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(&key);
        btree_data.extend_from_slice(&(value.len() as u32).to_le_bytes());
        btree_data.extend_from_slice(&value);
    }

    let schema = ColumnarSchema {
        columns: vec![
            ColumnDef {
                name: "name".to_string(),
                data_type: "string".to_string(),
                nullable: false,
            },
            ColumnDef {
                name: "age".to_string(),
                data_type: "int32".to_string(),
                nullable: false,
            },
        ],
    };

    // Benchmark CPU conversion
    group.bench_function("cpu_conversion_1000_rows", |b| {
        let converter = FormatConverter::new(ConversionStrategy::Cpu);
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| converter.btree_to_columnar(black_box(&btree_data), black_box(&schema)));
    });

    // Benchmark zero-copy conversion
    group.bench_function("zero_copy_conversion_1000_rows", |b| {
        let converter = FormatConverter::new(ConversionStrategy::ZeroCopy);
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| converter.btree_to_columnar(black_box(&btree_data), black_box(&schema)));
    });

    group.finish();
}

fn bench_columnar_queries(c: &mut Criterion) {
    use joule_db_browser::htap::ColumnarStore;

    let mut group = c.benchmark_group("htap_columnar_queries");

    // Create store with data
    let mut store = ColumnarStore::new(
        vec!["name".to_string(), "age".to_string()],
        vec!["string".to_string(), "float64".to_string()],
    );

    // Add rows
    for i in 0..10000 {
        store
            .add_row(vec![
                serde_json::json!(format!("user_{}", i)),
                serde_json::json!(i as f64 * 10.0),
            ])
            .unwrap();
    }

    // Benchmark COUNT query
    group.bench_function("count_query_10000_rows", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| store.query(black_box("SELECT COUNT(*) FROM table")));
    });

    // Benchmark SUM query
    group.bench_function("sum_query_10000_rows", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| store.query(black_box("SELECT SUM(age) FROM table")));
    });

    group.finish();
}

fn bench_columnar_add_row(c: &mut Criterion) {
    use joule_db_browser::htap::ColumnarStore;

    let mut group = c.benchmark_group("htap_columnar_add_row");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut store = ColumnarStore::new(
                vec!["name".to_string(), "age".to_string()],
                vec!["string".to_string(), "float64".to_string()],
            );
            b.iter(|| {
                for i in 0..size {
                    store
                        .add_row(vec![
                            serde_json::json!(format!("user_{}", i)),
                            serde_json::json!(i as f64 * 10.0),
                        ])
                        .unwrap();
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_format_conversion,
    bench_columnar_queries,
    bench_columnar_add_row
);
criterion_main!(benches);
