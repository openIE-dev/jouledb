//! Criterion benchmark suite for Cypher and CQL executors.
//!
//! Run with: `cargo bench -p joule-db-server --bench multimodel`
//! Quick compile check: `cargo bench -p joule-db-server --bench multimodel -- --test`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use joule_db_query::cql::CqlParser;
use joule_db_query::cypher::CypherParser;
use joule_db_server::amorphic_adapter::AmorphicTableStorage;
use joule_db_server::cql_executor::execute_cql;
use joule_db_server::cypher_executor::execute_cypher;
use std::sync::Arc;
use std::time::Instant;

/// Create a fresh AmorphicTableStorage backed by a temp directory.
fn bench_amorphic() -> Arc<AmorphicTableStorage> {
    let dir = std::env::temp_dir().join(format!("jouledb-bench-{}", uuid::Uuid::new_v4()));
    let store = joule_db_amorphic::DurableAmorphicStore::open(&dir).expect("bench store");
    Arc::new(AmorphicTableStorage::new(store))
}

// ---------------------------------------------------------------------------
// Cypher benchmarks
// ---------------------------------------------------------------------------

fn bench_cypher_create_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cypher_benches");

    for n in [100, 1000] {
        group.bench_with_input(BenchmarkId::new("create_nodes", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let amorphic = bench_amorphic();
                    let parser = CypherParser::new();
                    (amorphic, parser)
                },
                |(amorphic, mut parser)| {
                    for i in 0..n {
                        let query_str =
                            format!("CREATE (n:Person {{name: 'Person{}', age: {}}})", i, i);
                        let q = parser.parse(&query_str).expect("parse CREATE");
                        let _ = black_box(execute_cypher(&q, &amorphic, Instant::now()));
                    }
                },
            );
        });
    }

    group.finish();
}

fn bench_cypher_match_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("cypher_benches");

    group.bench_function("match_scan_1000", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CypherParser::new();
                // Pre-populate 1000 nodes
                for i in 0..1000 {
                    let query_str =
                        format!("CREATE (n:Person {{name: 'Person{}', age: {}}})", i, i);
                    let q = parser.parse(&query_str).expect("parse CREATE");
                    execute_cypher(&q, &amorphic, Instant::now()).expect("create node");
                }
                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                let q = parser
                    .parse("MATCH (n:Person) RETURN n.name")
                    .expect("parse MATCH");
                let _ = black_box(execute_cypher(&q, &amorphic, Instant::now()));
            },
        );
    });

    group.finish();
}

fn bench_cypher_match_with_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("cypher_benches");

    group.bench_function("match_with_filter_1000", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CypherParser::new();
                // Pre-populate 1000 nodes
                for i in 0..1000 {
                    let query_str =
                        format!("CREATE (n:Person {{name: 'Person{}', age: {}}})", i, i);
                    let q = parser.parse(&query_str).expect("parse CREATE");
                    execute_cypher(&q, &amorphic, Instant::now()).expect("create node");
                }
                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                let q = parser
                    .parse("MATCH (n:Person) WHERE n.age > 500 RETURN n.name")
                    .expect("parse MATCH WHERE");
                let _ = black_box(execute_cypher(&q, &amorphic, Instant::now()));
            },
        );
    });

    group.finish();
}

fn bench_cypher_relationship_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("cypher_benches");

    group.bench_function("relationship_traversal_star", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CypherParser::new();

                // Create hub node
                let q = parser
                    .parse("CREATE (hub:Person {name: 'Hub'})")
                    .expect("parse");
                execute_cypher(&q, &amorphic, Instant::now()).expect("create hub");

                // Create 100 spoke nodes and connect them to the hub
                for i in 0..100 {
                    let create_spoke = format!("CREATE (s:Person {{name: 'Spoke{}'}})", i);
                    let q = parser.parse(&create_spoke).expect("parse");
                    execute_cypher(&q, &amorphic, Instant::now()).expect("create spoke");

                    let connect = format!(
                        "MATCH (a:Person {{name: 'Hub'}}), (b:Person {{name: 'Spoke{}'}}) CREATE (a)-[:KNOWS]->(b)",
                        i
                    );
                    let q = parser.parse(&connect).expect("parse");
                    execute_cypher(&q, &amorphic, Instant::now()).expect("create edge");
                }

                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                let q = parser
                    .parse("MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
                    .expect("parse traversal");
                let _ = black_box(execute_cypher(&q, &amorphic, Instant::now()));
            },
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// CQL benchmarks
// ---------------------------------------------------------------------------

fn bench_cql_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("cql_benches");

    for n in [100, 1000] {
        group.bench_with_input(BenchmarkId::new("insert", n), &n, |b, &n| {
            b.iter_with_setup(
                || {
                    let amorphic = bench_amorphic();
                    let mut parser = CqlParser::new();
                    // Create the table first
                    let q = parser
                        .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
                        .expect("parse CREATE TABLE");
                    execute_cql(&q, &amorphic, Instant::now()).expect("create table");
                    (amorphic, parser)
                },
                |(amorphic, mut parser)| {
                    for i in 0..n {
                        let query_str = format!(
                            "INSERT INTO users (id, name, age) VALUES ({}, 'User{}', {})",
                            i, i, i
                        );
                        let q = parser.parse(&query_str).expect("parse INSERT");
                        let _ = black_box(execute_cql(&q, &amorphic, Instant::now()));
                    }
                },
            );
        });
    }

    group.finish();
}

fn bench_cql_select_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("cql_benches");

    group.bench_function("select_full_scan_1000", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CqlParser::new();
                let q = parser
                    .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
                    .expect("parse CREATE TABLE");
                execute_cql(&q, &amorphic, Instant::now()).expect("create table");

                // Pre-populate 1000 rows
                for i in 0..1000 {
                    let query_str = format!(
                        "INSERT INTO users (id, name, age) VALUES ({}, 'User{}', {})",
                        i, i, i
                    );
                    let q = parser.parse(&query_str).expect("parse INSERT");
                    execute_cql(&q, &amorphic, Instant::now()).expect("insert row");
                }
                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                let q = parser.parse("SELECT * FROM users").expect("parse SELECT");
                let _ = black_box(execute_cql(&q, &amorphic, Instant::now()));
            },
        );
    });

    group.finish();
}

fn bench_cql_select_with_where(c: &mut Criterion) {
    let mut group = c.benchmark_group("cql_benches");

    group.bench_function("select_with_where_1000", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CqlParser::new();
                let q = parser
                    .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
                    .expect("parse CREATE TABLE");
                execute_cql(&q, &amorphic, Instant::now()).expect("create table");

                // Pre-populate 1000 rows
                for i in 0..1000 {
                    let query_str = format!(
                        "INSERT INTO users (id, name, age) VALUES ({}, 'User{}', {})",
                        i, i, i
                    );
                    let q = parser.parse(&query_str).expect("parse INSERT");
                    execute_cql(&q, &amorphic, Instant::now()).expect("insert row");
                }
                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                let q = parser
                    .parse("SELECT * FROM users WHERE id = 500")
                    .expect("parse SELECT WHERE");
                let _ = black_box(execute_cql(&q, &amorphic, Instant::now()));
            },
        );
    });

    group.finish();
}

fn bench_cql_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("cql_benches");

    group.bench_function("batch_insert_10", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CqlParser::new();
                let q = parser
                    .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
                    .expect("parse CREATE TABLE");
                execute_cql(&q, &amorphic, Instant::now()).expect("create table");
                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                // Build a BATCH with 10 inserts
                let mut batch_str = String::from("BEGIN BATCH ");
                for i in 0..10 {
                    batch_str.push_str(&format!(
                        "INSERT INTO users (id, name, age) VALUES ({}, 'User{}', {}); ",
                        i, i, i
                    ));
                }
                batch_str.push_str("APPLY BATCH");
                let q = parser.parse(&batch_str).expect("parse BATCH");
                let _ = black_box(execute_cql(&q, &amorphic, Instant::now()));
            },
        );
    });

    group.bench_function("individual_insert_10", |b| {
        b.iter_with_setup(
            || {
                let amorphic = bench_amorphic();
                let mut parser = CqlParser::new();
                let q = parser
                    .parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT, age INT)")
                    .expect("parse CREATE TABLE");
                execute_cql(&q, &amorphic, Instant::now()).expect("create table");
                (amorphic, parser)
            },
            |(amorphic, mut parser)| {
                for i in 0..10 {
                    let query_str = format!(
                        "INSERT INTO users (id, name, age) VALUES ({}, 'User{}', {})",
                        i, i, i
                    );
                    let q = parser.parse(&query_str).expect("parse INSERT");
                    let _ = black_box(execute_cql(&q, &amorphic, Instant::now()));
                }
            },
        );
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

criterion_group!(
    cypher_benches,
    bench_cypher_create_nodes,
    bench_cypher_match_scan,
    bench_cypher_match_with_filter,
    bench_cypher_relationship_traversal
);

criterion_group!(
    cql_benches,
    bench_cql_insert,
    bench_cql_select_full_scan,
    bench_cql_select_with_where,
    bench_cql_batch_insert
);

criterion_main!(cypher_benches, cql_benches);
