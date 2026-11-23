//! Performance benchmarks for Deterministic Execution

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use joule_db_browser::deterministic::MultiTabCoordinator;

fn bench_lock_acquisition(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic_lock_acquisition");

    // Benchmark Web Locks
    group.bench_function("weblocks_acquire", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let coord = MultiTabCoordinator::new("weblocks".to_string()).await;
                if let Ok(coord) = coord {
                    coord.acquire_lock(black_box("test_lock")).await.ok();
                }
            });
    });

    // Benchmark SharedArrayBuffer locks
    group.bench_function("sab_acquire", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let coord = MultiTabCoordinator::new("sab".to_string()).await;
                if let Ok(coord) = coord {
                    coord.acquire_lock(black_box("test_lock")).await.ok();
                }
            });
    });

    // Benchmark BroadcastChannel locks
    group.bench_function("broadcast_acquire", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let coord = MultiTabCoordinator::new("broadcast".to_string()).await;
                if let Ok(coord) = coord {
                    coord.acquire_lock(black_box("test_lock")).await.ok();
                }
            });
    });

    group.finish();
}

fn bench_transaction_coordination(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic_transaction_coordination");

    for size in [10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.to_async(tokio::runtime::Runtime::new().unwrap())
                .iter(|| async {
                    let coord = MultiTabCoordinator::new("weblocks".to_string()).await;
                    if let Ok(coord) = coord {
                        for i in 0..size {
                            let tx_id = format!("tx_{}", i);
                            coord.coordinate_transaction(black_box(&tx_id)).await.ok();
                        }
                    }
                });
        });
    }

    group.finish();
}

fn bench_coordinator_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("deterministic_coordinator_creation");

    // Benchmark Web Locks coordinator
    group.bench_function("weblocks_create", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| MultiTabCoordinator::new(black_box("weblocks".to_string())));
    });

    // Benchmark SharedArrayBuffer coordinator
    group.bench_function("sab_create", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| MultiTabCoordinator::new(black_box("sab".to_string())));
    });

    // Benchmark BroadcastChannel coordinator
    group.bench_function("broadcast_create", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| MultiTabCoordinator::new(black_box("broadcast".to_string())));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lock_acquisition,
    bench_transaction_coordination,
    bench_coordinator_creation
);
criterion_main!(benches);
