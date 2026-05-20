//! Multimodel Benchmark
//!
//! Measures JouleDB's performance across all query paradigms and index types.
//! Each section exercises one component added in the SOTA unification plan.
//!
//! Run: cargo bench --bench multimodel_bench

use std::collections::{HashMap, HashSet};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          JouleDB Multimodel Benchmark Suite                 ║");
    println!("║  7 Query Languages · 5 Index Types · 18+ Graph Algorithms  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    bench_art_index();
    bench_minhash_lsh();
    bench_wcoj();
    bench_datalog();
    bench_sparql();
    bench_gremlin();
    bench_temporal();

    println!("══════════════════════════════════════════════════════════════");
    println!("  All benchmarks complete.");
    println!("══════════════════════════════════════════════════════════════");
}

// ---------------------------------------------------------------------------
// ART Index
// ---------------------------------------------------------------------------

fn bench_art_index() {
    use joule_db_core::index::{AdaptiveRadixTree, BTreeIndex, Index, OrderedIndex};

    println!("── ART Index vs BTree ──────────────────────────────────────");
    let n = 100_000u32;

    // ART insert
    let mut art = AdaptiveRadixTree::new();
    let start = Instant::now();
    for i in 0..n {
        art.insert(&i.to_be_bytes(), &i.to_le_bytes()).unwrap();
    }
    let art_insert = start.elapsed();

    // BTree insert
    let mut btree = BTreeIndex::new();
    let start = Instant::now();
    for i in 0..n {
        btree.insert(&i.to_be_bytes(), &i.to_le_bytes()).unwrap();
    }
    let btree_insert = start.elapsed();

    // ART lookup
    let start = Instant::now();
    for i in 0..n {
        art.get(&i.to_be_bytes()).unwrap();
    }
    let art_get = start.elapsed();

    // BTree lookup
    let start = Instant::now();
    for i in 0..n {
        btree.get(&i.to_be_bytes()).unwrap();
    }
    let btree_get = start.elapsed();

    println!("  {:>12} │ {:>12} │ {:>12}", "", "ART", "BTree");
    println!("  {:>12} │ {:>9.2}ms │ {:>9.2}ms",
        "Insert 100K",
        art_insert.as_secs_f64() * 1000.0,
        btree_insert.as_secs_f64() * 1000.0,
    );
    println!("  {:>12} │ {:>9.2}ms │ {:>9.2}ms",
        "Lookup 100K",
        art_get.as_secs_f64() * 1000.0,
        btree_get.as_secs_f64() * 1000.0,
    );
    let speedup = btree_get.as_secs_f64() / art_get.as_secs_f64().max(0.000001);
    println!("  ART speedup: {:.1}x", speedup);
    println!();
}

// ---------------------------------------------------------------------------
// MinHash-LSH
// ---------------------------------------------------------------------------

fn bench_minhash_lsh() {
    use joule_db_core::index::minhash_lsh::{MinHashLshConfig, MinHashLshIndex};
    use joule_db_core::index::Index;

    println!("── MinHash-LSH ────────────────────────────────────────────");
    let config = MinHashLshConfig { num_hashes: 64, bands: 8, rows: 8 };
    let mut idx = MinHashLshIndex::new(config);

    // Insert 1000 documents.
    let start = Instant::now();
    for i in 0..1000u32 {
        let doc = format!("document number {} with some content to generate trigrams for hashing", i);
        idx.insert(doc.as_bytes(), &i.to_le_bytes()).unwrap();
    }
    let insert_time = start.elapsed();

    // Query for near-duplicates.
    let query = b"document number 42 with some content to generate trigrams for hashing";
    let start = Instant::now();
    let results = idx.find_similar(query, 10).unwrap();
    let query_time = start.elapsed();

    println!("  Insert 1K docs: {:.2}ms", insert_time.as_secs_f64() * 1000.0);
    println!("  Query (top-10):  {:.3}ms ({} candidates)", query_time.as_secs_f64() * 1000.0, results.len());
    if let Some((_, sim)) = results.first() {
        println!("  Best match sim:  {:.3}", sim);
    }
    println!();
}

// ---------------------------------------------------------------------------
// WCOJ
// ---------------------------------------------------------------------------

fn bench_wcoj() {
    use joule_db_query::wcoj::{Atom, Relation, WcojQuery, execute_wcoj};

    println!("── WCOJ (Triangle Counting) ────────────────────────────────");

    // Build a random-ish graph with known structure.
    let n = 100;
    let mut edge_tuples = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            // Connect ~20% of pairs.
            if (i * 7 + j * 13) % 5 == 0 {
                let si = format!("{}", i);
                let sj = format!("{}", j);
                edge_tuples.push(vec![si.as_bytes().to_vec(), sj.as_bytes().to_vec()]);
                edge_tuples.push(vec![sj.as_bytes().to_vec(), si.as_bytes().to_vec()]);
            }
        }
    }

    let edges = Relation::new(
        vec!["src".into(), "dst".into()],
        edge_tuples,
    );

    let query = WcojQuery {
        atoms: vec![
            Atom { relation_name: "edges".into(), variables: vec!["X".into(), "Y".into()] },
            Atom { relation_name: "edges".into(), variables: vec!["Y".into(), "Z".into()] },
            Atom { relation_name: "edges".into(), variables: vec!["Z".into(), "X".into()] },
        ],
        output_variables: vec!["X".into(), "Y".into(), "Z".into()],
    };

    let relations = vec![("edges".into(), edges)];
    let start = Instant::now();
    let results = execute_wcoj(&query, &relations);
    let elapsed = start.elapsed();

    println!("  Graph: {} nodes, ~20% edge density", n);
    println!("  Triangles found: {} (directed)", results.len());
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
}

// ---------------------------------------------------------------------------
// Datalog
// ---------------------------------------------------------------------------

fn bench_datalog() {
    use joule_db_query::datalog::{DatalogParser, evaluate};

    println!("── Datalog (Transitive Closure) ──────────────────────────────");

    // Build a chain graph: 0→1→2→...→N.
    let n = 200;
    let mut program = String::new();
    for i in 0..n {
        program.push_str(&format!("edge(\"{}\", \"{}\").\n", i, i + 1));
    }
    program.push_str("reachable(X, Y) :- edge(X, Y).\n");
    program.push_str("reachable(X, Y) :- edge(X, Z), reachable(Z, Y).\n");
    program.push_str("?- reachable(\"0\", X).\n");

    let mut parser = DatalogParser::new();
    let prog = parser.parse(&program).unwrap();

    let start = Instant::now();
    let result = evaluate(&prog).unwrap();
    let elapsed = start.elapsed();

    println!("  Chain length: {} edges", n);
    println!("  Reachable from 0: {} nodes", result.rows.len());
    println!("  Iterations: {}", result.iterations);
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
}

// ---------------------------------------------------------------------------
// SPARQL
// ---------------------------------------------------------------------------

fn bench_sparql() {
    use joule_db_query::sparql::{SparqlParser, Triple, evaluate_sparql};

    println!("── SPARQL (Triple Pattern Matching) ─────────────────────────");

    // Build 10K triples.
    let n = 10_000;
    let mut triples = Vec::with_capacity(n);
    for i in 0..n {
        triples.push(Triple {
            subject: format!("person_{}", i % 1000),
            predicate: "name".into(),
            object: format!("Name_{}", i),
        });
    }

    let mut parser = SparqlParser::new();
    let query = parser.parse(r#"SELECT ?name WHERE { ?person <name> ?name } LIMIT 100"#).unwrap();

    let start = Instant::now();
    let result = evaluate_sparql(&query, &triples).unwrap();
    let elapsed = start.elapsed();

    println!("  Triples: {}", n);
    println!("  Results: {} (LIMIT 100)", result.rows.len());
    println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!();
}

// ---------------------------------------------------------------------------
// Gremlin
// ---------------------------------------------------------------------------

fn bench_gremlin() {
    use joule_db_query::gremlin::GremlinParser;

    println!("── Gremlin (Parse Throughput) ────────────────────────────────");

    let queries = vec![
        r#"g.V().has("name", "Alice").out("knows").values("name")"#,
        r#"g.V(1).outE("created").inV().path()"#,
        r#"g.V().hasLabel("person").count()"#,
        r#"g.addV("person").property("name", "Dave")"#,
        r#"g.V().out("knows").out("knows").dedup().limit(10)"#,
    ];

    let iterations = 10_000;
    let start = Instant::now();
    for _ in 0..iterations {
        for q in &queries {
            let mut parser = GremlinParser::new();
            parser.parse(q).unwrap();
        }
    }
    let elapsed = start.elapsed();

    let total = iterations * queries.len();
    let per_query_ns = elapsed.as_nanos() as f64 / total as f64;
    println!("  Parsed {} queries in {:.2}ms", total, elapsed.as_secs_f64() * 1000.0);
    println!("  Per query: {:.0}ns", per_query_ns);
    println!("  Throughput: {:.0}K queries/sec", total as f64 / elapsed.as_secs_f64() / 1000.0);
    println!();
}

// ---------------------------------------------------------------------------
// Temporal
// ---------------------------------------------------------------------------

fn bench_temporal() {
    use joule_db_core::temporal::{TemporalTable, TemporalClause};

    println!("── Temporal (Time-Travel Queries) ───────────────────────────");

    let table = TemporalTable::new("users");
    let n = 10_000;

    // Insert N rows at t=0.
    let start = Instant::now();
    for i in 0..n {
        let mut data = HashMap::new();
        data.insert("name".into(), serde_json::json!(format!("User_{}", i)));
        table.insert(&format!("user_{}", i), data, 0).unwrap();
    }
    let insert_time = start.elapsed();

    // Update half at t=1000.
    let start = Instant::now();
    for i in 0..n / 2 {
        let mut data = HashMap::new();
        data.insert("name".into(), serde_json::json!(format!("Updated_{}", i)));
        table.update(&format!("user_{}", i), data, 1000).unwrap();
    }
    let update_time = start.elapsed();

    // Point-in-time query at t=500 (before updates).
    let start = Instant::now();
    let historical = table.scan_as_of(500).unwrap();
    let asof_time = start.elapsed();

    // Current query.
    let start = Instant::now();
    let current = table.scan_current().unwrap();
    let current_time = start.elapsed();

    println!("  Insert {}:     {:.2}ms", n, insert_time.as_secs_f64() * 1000.0);
    println!("  Update {}:    {:.2}ms", n / 2, update_time.as_secs_f64() * 1000.0);
    println!("  AS OF (t=500): {:.2}ms ({} rows)", asof_time.as_secs_f64() * 1000.0, historical.len());
    println!("  Current scan:  {:.2}ms ({} rows)", current_time.as_secs_f64() * 1000.0, current.len());
    println!("  Total versions: {}", table.version_count());
    println!();
}
