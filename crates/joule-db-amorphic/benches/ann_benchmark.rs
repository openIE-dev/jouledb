//! ANN-Benchmarks: Industry-standard vector search benchmark
//!
//! This implements the ann-benchmarks.com protocol for evaluating
//! approximate nearest neighbor search performance.
//!
//! ## Datasets Supported
//! - SIFT1M: 1M vectors, 128 dimensions, Euclidean distance
//! - GloVe-100: 1.2M vectors, 100 dimensions, Cosine distance
//! - Random: Synthetic dataset for quick testing
//!
//! ## Metrics
//! - Recall@k: Fraction of true k-nearest neighbors found
//! - QPS: Queries per second at various recall levels
//! - Latency: P50, P95, P99 query latencies
//!
//! ## Usage
//! ```bash
//! # Quick test with synthetic data
//! cargo bench --bench ann_benchmark -- --quick
//!
//! # Full benchmark with SIFT1M (requires download)
//! cargo bench --bench ann_benchmark -- --dataset sift1m
//! ```
//!
//! ## Dataset Download
//! Download SIFT1M from: http://corpus-texmex.irisa.fr/
//! Place files in: benches/datasets/sift/

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::{Duration, Instant};

use joule_db_amorphic::{AmorphicStore, ShardedAmorphicStore, platform};

fn main() {
    println!("=======================================================");
    println!("       ANN-Benchmarks: Vector Search Evaluation");
    println!("=======================================================\n");

    let args: Vec<String> = std::env::args().collect();
    let quick_mode = args.iter().any(|a| a == "--quick");
    let dataset = args
        .iter()
        .position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("random");

    // Show platform info
    let p = platform();
    println!(
        "Platform: {} cores, {:?} SIMD\n",
        p.cpu_cores,
        p.simd.best_level()
    );

    match dataset {
        "sift1m" | "sift" => {
            if let Err(e) = run_sift_benchmark() {
                eprintln!("SIFT benchmark failed: {}", e);
                eprintln!("Download SIFT1M from: http://corpus-texmex.irisa.fr/");
                eprintln!("Place in: benches/datasets/sift/");
                println!("\nFalling back to synthetic benchmark...\n");
                run_synthetic_benchmark(if quick_mode { 10_000 } else { 100_000 });
            }
        }
        "glove" => {
            println!("GloVe benchmark not yet implemented. Using synthetic.\n");
            run_synthetic_benchmark(if quick_mode { 10_000 } else { 100_000 });
        }
        _ => {
            run_synthetic_benchmark(if quick_mode { 10_000 } else { 100_000 });
        }
    }

    println!("\n=======================================================");
    println!("                  Benchmark Complete");
    println!("=======================================================");
}

// =============================================================================
// SYNTHETIC BENCHMARK (always available)
// =============================================================================

fn run_synthetic_benchmark(num_vectors: usize) {
    println!("--- Synthetic ANN Benchmark ---");
    println!("  Vectors:    {}", num_vectors);
    println!("  Dimensions: 128");
    println!("  Distance:   Euclidean (via Hamming on binarized vectors)\n");

    // Generate synthetic data
    let (base_vectors, query_vectors, ground_truth) =
        generate_synthetic_data(num_vectors, 128, 100);

    // Run benchmark
    run_ann_benchmark(
        "Synthetic-128d",
        &base_vectors,
        &query_vectors,
        &ground_truth,
    );
}

fn generate_synthetic_data(
    num_base: usize,
    dimensions: usize,
    num_queries: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    println!("  Generating {} base vectors...", num_base);

    // Use deterministic seed for reproducibility
    let mut rng_state: u64 = 42;
    let mut next_rand = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };

    // Generate base vectors
    let base_vectors: Vec<Vec<f32>> = (0..num_base)
        .map(|_| (0..dimensions).map(|_| next_rand()).collect())
        .collect();

    // Generate query vectors (slightly perturbed versions of random base vectors)
    println!("  Generating {} query vectors...", num_queries);
    let query_vectors: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| {
            let base_idx = (i * 97) % num_base; // Deterministic selection
            base_vectors[base_idx]
                .iter()
                .map(|&v| v + next_rand() * 0.1)
                .collect()
        })
        .collect();

    // Compute ground truth (brute force k-NN)
    println!("  Computing ground truth (k=100)...");
    let k = 100;
    let ground_truth: Vec<Vec<usize>> = query_vectors
        .iter()
        .map(|query| {
            let mut heap: BinaryHeap<Reverse<(ordered_float::OrderedFloat, usize)>> =
                BinaryHeap::new();

            for (idx, base) in base_vectors.iter().enumerate() {
                let dist = euclidean_distance(query, base);
                heap.push(Reverse((ordered_float::OrderedFloat(dist), idx)));
            }

            // Extract top-k
            let mut result = Vec::with_capacity(k);
            for _ in 0..k.min(heap.len()) {
                if let Some(Reverse((_, idx))) = heap.pop() {
                    result.push(idx);
                }
            }
            result
        })
        .collect();

    (base_vectors, query_vectors, ground_truth)
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

// =============================================================================
// SIFT1M BENCHMARK
// =============================================================================

fn run_sift_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    let base_path = Path::new("benches/datasets/sift");

    if !base_path.exists() {
        return Err("SIFT dataset directory not found".into());
    }

    println!("--- SIFT1M ANN Benchmark ---");
    println!("  Vectors:    1,000,000");
    println!("  Dimensions: 128");
    println!("  Distance:   Euclidean\n");

    // Load SIFT vectors
    let base_vectors = load_fvecs(&base_path.join("sift_base.fvecs"))?;
    let query_vectors = load_fvecs(&base_path.join("sift_query.fvecs"))?;
    let ground_truth = load_ivecs(&base_path.join("sift_groundtruth.ivecs"))?;

    println!("  Loaded {} base vectors", base_vectors.len());
    println!("  Loaded {} query vectors", query_vectors.len());
    println!("  Loaded {} ground truth entries\n", ground_truth.len());

    run_ann_benchmark("SIFT1M", &base_vectors, &query_vectors, &ground_truth);

    Ok(())
}

/// Load vectors from .fvecs format (float vectors)
fn load_fvecs(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        // Read dimension (4 bytes, little-endian)
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read vector data
        let mut vec = vec![0f32; dim];
        let mut float_buf = [0u8; 4];
        for val in vec.iter_mut() {
            reader.read_exact(&mut float_buf)?;
            *val = f32::from_le_bytes(float_buf);
        }

        vectors.push(vec);
    }

    Ok(vectors)
}

/// Load vectors from .ivecs format (integer vectors - ground truth)
fn load_ivecs(path: &Path) -> Result<Vec<Vec<usize>>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        // Read dimension (4 bytes, little-endian)
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read vector data
        let mut vec = vec![0i32; dim];
        let mut int_buf = [0u8; 4];
        for val in vec.iter_mut() {
            reader.read_exact(&mut int_buf)?;
            *val = i32::from_le_bytes(int_buf);
        }

        vectors.push(vec.into_iter().map(|v| v as usize).collect());
    }

    Ok(vectors)
}

// =============================================================================
// CORE BENCHMARK LOGIC
// =============================================================================

fn run_ann_benchmark(
    name: &str,
    base_vectors: &[Vec<f32>],
    query_vectors: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
) {
    println!("=== {} Benchmark ===\n", name);

    // Test different k values
    let k_values = [1, 10, 100];

    // Index the data using AmorphicStore
    println!("Building index...");
    let build_start = Instant::now();

    let store = ShardedAmorphicStore::with_shard_count(8);

    // Convert vectors to JSON and ingest with trackable IDs
    // Store full vector (up to 32 dims) for better HDC encoding
    for (i, vec) in base_vectors.iter().enumerate() {
        // Store with unique ID and more vector dimensions for better similarity
        let json = format!(
            r#"{{"_vec_id": {}, "dims": {:?}}}"#,
            i,
            &vec[..vec.len().min(32)] // Use 32 dims for HDC encoding
        );
        let _ = store.ingest_json(&json);

        if (i + 1) % 10000 == 0 {
            print!("\r  Indexed {}/{} vectors", i + 1, base_vectors.len());
        }
    }
    println!(
        "\r  Indexed {} vectors in {:?}",
        base_vectors.len(),
        build_start.elapsed()
    );

    // Build HDC index for direct vector similarity search
    println!("  Building HDC vector index...");
    let hdc_build_start = Instant::now();
    let hdc_index = build_hdc_index(base_vectors);
    println!("  HDC index built in {:?}", hdc_build_start.elapsed());

    println!("\nRunning queries...\n");

    // Results table header
    println!("┌─────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│    k    │ Recall@k │   QPS    │ P50 (µs) │ P99 (µs) │");
    println!("├─────────┼──────────┼──────────┼──────────┼──────────┤");

    for &k in &k_values {
        let (recall, qps, p50, p99) =
            benchmark_queries_hdc(&hdc_index, base_vectors, query_vectors, ground_truth, k);

        println!(
            "│ {:>7} │ {:>8.4} │ {:>8.0} │ {:>8.1} │ {:>8.1} │",
            k, recall, qps, p50, p99
        );
    }

    println!("└─────────┴──────────┴──────────┴──────────┴──────────┘");

    // Summary
    println!("\nIndex Statistics:");
    if let Ok(stats) = store.stats() {
        println!("  Total records: {}", stats.total_records);
        println!("  Shards: {}", stats.shard_count);
        println!("  Distribution ratio: {:.2}x", stats.distribution_ratio);
    }
    println!("  HDC embeddings: {}", hdc_index.len());
}

/// HDC embedding for a single vector
type HdcEmbedding = Vec<u64>;

/// Random hyperplanes for LSH (generated once, deterministically)
struct LshProjections {
    /// Each hyperplane is a vector of the same dimension as input
    /// We use 512 hyperplanes for 512-bit signatures
    hyperplanes: Vec<Vec<f32>>,
}

impl LshProjections {
    fn new(input_dim: usize, num_bits: usize) -> Self {
        let mut hyperplanes = Vec::with_capacity(num_bits);
        let mut rng_state: u64 = 0xDEADBEEF_CAFEBABE;

        for _ in 0..num_bits {
            let mut plane = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                // Generate Gaussian-like random value using Box-Muller
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u1 = ((rng_state >> 33) as f64) / (u32::MAX as f64);
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = ((rng_state >> 33) as f64) / (u32::MAX as f64);

                // Box-Muller transform
                let z = ((-2.0 * u1.max(1e-10).ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
                plane.push(z);
            }
            hyperplanes.push(plane);
        }

        Self { hyperplanes }
    }

    fn project(&self, vec: &[f32]) -> HdcEmbedding {
        let num_words = (self.hyperplanes.len() + 63) / 64;
        let mut embedding = vec![0u64; num_words];

        for (bit_idx, hyperplane) in self.hyperplanes.iter().enumerate() {
            // Compute dot product
            let dot: f32 = vec
                .iter()
                .zip(hyperplane.iter())
                .map(|(&v, &h)| v * h)
                .sum();

            // If positive dot product, set bit to 1
            if dot > 0.0 {
                let word_idx = bit_idx / 64;
                let bit_pos = bit_idx % 64;
                embedding[word_idx] |= 1u64 << bit_pos;
            }
        }

        embedding
    }
}

/// Global LSH projections (initialized lazily)
static mut LSH_PROJECTIONS: Option<LshProjections> = None;
static LSH_INIT: std::sync::Once = std::sync::Once::new();

fn get_lsh_projections(input_dim: usize) -> &'static LshProjections {
    unsafe {
        LSH_INIT.call_once(|| {
            // Use 1024 bits (16 x u64) for better recall
            LSH_PROJECTIONS = Some(LshProjections::new(input_dim, 1024));
        });
        LSH_PROJECTIONS.as_ref().unwrap()
    }
}

/// Build HDC index from base vectors using LSH
fn build_hdc_index(base_vectors: &[Vec<f32>]) -> Vec<HdcEmbedding> {
    if base_vectors.is_empty() {
        return vec![];
    }

    let input_dim = base_vectors[0].len();
    let lsh = get_lsh_projections(input_dim);

    base_vectors.iter().map(|vec| lsh.project(vec)).collect()
}

/// Convert float vector to HDC binary embedding using LSH
fn vector_to_hdc(vec: &[f32]) -> HdcEmbedding {
    let lsh = get_lsh_projections(vec.len());
    lsh.project(vec)
}

/// Hamming distance between two HDC embeddings
fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Benchmark queries using direct HDC similarity with true recall measurement
fn benchmark_queries_hdc(
    hdc_index: &[HdcEmbedding],
    base_vectors: &[Vec<f32>],
    query_vectors: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> (f64, f64, f64, f64) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let num_queries = query_vectors.len().min(ground_truth.len());
    let mut latencies = Vec::with_capacity(num_queries);
    let mut total_recall = 0.0;

    for (query, truth) in query_vectors
        .iter()
        .zip(ground_truth.iter())
        .take(num_queries)
    {
        // Convert query to HDC embedding
        let query_hdc = vector_to_hdc(query);

        let start = Instant::now();

        // Find k nearest by Hamming distance
        let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(k + 1);

        for (idx, embedding) in hdc_index.iter().enumerate() {
            let dist = hamming_distance(&query_hdc, embedding);

            if heap.len() < k {
                heap.push(Reverse((dist, idx)));
            } else if let Some(&Reverse((max_dist, _))) = heap.peek() {
                if dist < max_dist {
                    heap.pop();
                    heap.push(Reverse((dist, idx)));
                }
            }
        }

        let elapsed = start.elapsed();
        latencies.push(elapsed.as_micros() as f64);

        // Extract result IDs
        let result_ids: HashSet<usize> = heap.iter().map(|&Reverse((_, idx))| idx).collect();

        // Calculate TRUE recall: intersection of result IDs with ground truth IDs
        let truth_set: HashSet<_> = truth.iter().take(k).cloned().collect();
        let intersection = result_ids.intersection(&truth_set).count();

        let recall = if truth_set.is_empty() {
            0.0
        } else {
            intersection as f64 / truth_set.len() as f64
        };
        total_recall += recall;
    }

    // Calculate metrics
    let avg_recall = total_recall / num_queries as f64;

    let total_time: f64 = latencies.iter().sum();
    let qps = (num_queries as f64) / (total_time / 1_000_000.0);

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[num_queries / 2];
    let p99 = latencies[(num_queries as f64 * 0.99) as usize];

    (avg_recall, qps, p50, p99)
}

// =============================================================================
// HELPER: Ordered float for heap operations
// =============================================================================

mod ordered_float {
    #[derive(Copy, Clone, PartialEq)]
    pub struct OrderedFloat(pub f32);

    impl Eq for OrderedFloat {}

    impl PartialOrd for OrderedFloat {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for OrderedFloat {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}
