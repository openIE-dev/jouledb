//! Capacity Curve Benchmark Suite - Holographic Property Validation
//!
//! Validates that data can be superposed into a single vector and retrieved accurately.
//! Generates data for "White Paper" graph: capacity vs accuracy.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// Core Types
// ============================================================================

/// High-dimensional vector for VSA operations
#[derive(Clone)]
struct HyperVector {
    components: Vec<f32>,
}

impl HyperVector {
    fn random(dimension: usize, seed: u64) -> Self {
        let mut components = Vec::with_capacity(dimension);
        let mut rng = seed;
        for _ in 0..dimension {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            components.push((rng as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32);
        }
        let mut hv = Self { components };
        hv.normalize();
        hv
    }

    fn zero(dimension: usize) -> Self {
        Self {
            components: vec![0.0; dimension],
        }
    }

    fn from_seed(dimension: usize, seed: &str) -> Self {
        Self::random(
            dimension,
            seed.bytes()
                .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
        )
    }

    fn normalize(&mut self) {
        let norm: f32 = self.components.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            self.components.iter_mut().for_each(|x| *x /= norm);
        }
    }

    fn similarity(&self, other: &HyperVector) -> f32 {
        let dot: f32 = self
            .components
            .iter()
            .zip(&other.components)
            .map(|(a, b)| a * b)
            .sum();
        let na: f32 = self.components.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = other.components.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 1e-10 && nb > 1e-10 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    fn dot(&self, other: &HyperVector) -> f32 {
        self.components
            .iter()
            .zip(&other.components)
            .map(|(a, b)| a * b)
            .sum()
    }

    fn bind(&self, other: &HyperVector) -> HyperVector {
        let mut result = HyperVector {
            components: self
                .components
                .iter()
                .zip(&other.components)
                .map(|(a, b)| a * b)
                .collect(),
        };
        result.normalize();
        result
    }

    fn bundle(&self, other: &HyperVector) -> HyperVector {
        HyperVector {
            components: self
                .components
                .iter()
                .zip(&other.components)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    fn add_inplace(&mut self, other: &HyperVector) {
        self.components
            .iter_mut()
            .zip(&other.components)
            .for_each(|(a, b)| *a += b);
    }

    fn permute(&self, shift: i32) -> HyperVector {
        let dim = self.components.len() as i32;
        let mut components = vec![0.0; self.components.len()];
        for i in 0..self.components.len() {
            components[i] =
                self.components[((i as i32 - shift) % dim + dim) as usize % self.components.len()];
        }
        HyperVector { components }
    }
}

/// Binary hypervector (memory efficient)
#[derive(Clone)]
struct BinaryHyperVector {
    bits: Vec<u64>,
    dimension: usize,
}

impl BinaryHyperVector {
    fn random(dimension: usize, seed: u64) -> Self {
        let mut bits = Vec::with_capacity((dimension + 63) / 64);
        let mut rng = seed;
        for _ in 0..((dimension + 63) / 64) {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            bits.push(rng);
        }
        Self { bits, dimension }
    }

    fn bind(&self, other: &BinaryHyperVector) -> BinaryHyperVector {
        BinaryHyperVector {
            bits: self
                .bits
                .iter()
                .zip(&other.bits)
                .map(|(a, b)| a ^ b)
                .collect(),
            dimension: self.dimension,
        }
    }

    fn similarity(&self, other: &BinaryHyperVector) -> f32 {
        let matching: u32 = self
            .bits
            .iter()
            .zip(&other.bits)
            .map(|(a, b)| (!(a ^ b)).count_ones())
            .sum();
        matching as f32 / self.dimension as f32
    }

    fn memory_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

// ============================================================================
// Holographic Memory
// ============================================================================

#[allow(dead_code)]
struct HolographicMemory {
    hologram: HyperVector,
    dimension: usize,
    item_count: usize,
}

impl HolographicMemory {
    fn new(dimension: usize) -> Self {
        Self {
            hologram: HyperVector::zero(dimension),
            dimension,
            item_count: 0,
        }
    }

    fn store(&mut self, key: &HyperVector, value: &HyperVector) {
        self.hologram.add_inplace(&key.bind(value));
        self.item_count += 1;
    }

    fn retrieve(&self, key: &HyperVector, expected: &HyperVector) -> f32 {
        self.hologram.bind(key).similarity(expected)
    }
}

// ============================================================================
// 1. Capacity Curve Test
// ============================================================================

#[derive(Clone, Debug)]
struct CapacityCurvePoint {
    items_stored: usize,
    mean_similarity: f32,
    min_similarity: f32,
    std_dev: f32,
}

#[allow(dead_code)]
struct CapacityCurve {
    dimension: usize,
    points: Vec<CapacityCurvePoint>,
    crash_point: Option<usize>,
}

impl CapacityCurve {
    fn to_csv(&self) -> String {
        let mut csv = String::from("items,mean_sim,min_sim,std_dev\n");
        for p in &self.points {
            csv.push_str(&format!(
                "{},{:.4},{:.4},{:.4}\n",
                p.items_stored, p.mean_similarity, p.min_similarity, p.std_dev
            ));
        }
        csv
    }
}

fn run_capacity_curve_test(dimension: usize, max_items: usize, step: usize) -> CapacityCurve {
    let mut memory = HolographicMemory::new(dimension);
    let mut stored: Vec<(HyperVector, HyperVector)> = Vec::new();
    let mut points = Vec::new();

    for item_idx in 0..max_items {
        let key = HyperVector::random(dimension, (item_idx * 2) as u64);
        let value = HyperVector::random(dimension, (item_idx * 2 + 1) as u64);
        memory.store(&key, &value);
        stored.push((key, value));

        if (item_idx + 1) % step == 0 || item_idx == max_items - 1 {
            let sims: Vec<f32> = stored.iter().map(|(k, v)| memory.retrieve(k, v)).collect();
            let mean: f32 = sims.iter().sum::<f32>() / sims.len() as f32;
            let min = sims.iter().cloned().fold(f32::INFINITY, f32::min);
            let variance: f32 =
                sims.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / sims.len() as f32;
            points.push(CapacityCurvePoint {
                items_stored: stored.len(),
                mean_similarity: mean,
                min_similarity: min,
                std_dev: variance.sqrt(),
            });
        }
    }

    let crash = points
        .iter()
        .find(|p| p.mean_similarity < 0.75)
        .map(|p| p.items_stored);
    CapacityCurve {
        dimension,
        points,
        crash_point: crash,
    }
}

// ============================================================================
// 2. Orthogonality Test
// ============================================================================

#[derive(Debug)]
#[allow(dead_code)]
struct OrthogonalityResult {
    dimension: usize,
    mean_dot: f32,
    std_dev: f32,
}

fn test_orthogonality(dimension: usize, num_vectors: usize) -> OrthogonalityResult {
    let vectors: Vec<HyperVector> = (0..num_vectors)
        .map(|i| HyperVector::random(dimension, i as u64))
        .collect();
    let mut dots = Vec::new();
    for i in 0..num_vectors {
        for j in (i + 1)..num_vectors {
            dots.push(vectors[i].dot(&vectors[j]));
        }
    }
    let mean: f32 = dots.iter().sum::<f32>() / dots.len() as f32;
    let variance: f32 = dots.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / dots.len() as f32;
    OrthogonalityResult {
        dimension,
        mean_dot: mean,
        std_dev: variance.sqrt(),
    }
}

// ============================================================================
// 3. Blind Operation Test
// ============================================================================

#[derive(Debug)]
struct BlindResult {
    operation: String,
    similarity: f32,
    success: bool,
}

fn test_blind_operations(dimension: usize) -> Vec<BlindResult> {
    let mut results = Vec::new();

    // Blind search
    let mut db = HyperVector::zero(dimension);
    let key_target = HyperVector::from_seed(dimension, "key_target");
    let target = HyperVector::from_seed(dimension, "target");
    db.add_inplace(&key_target.bind(&target));
    for i in 0..10 {
        db.add_inplace(
            &HyperVector::from_seed(dimension, &format!("k{}", i))
                .bind(&HyperVector::random(dimension, i as u64 + 1000)),
        );
    }
    let sim = db.bind(&key_target).similarity(&target);
    results.push(BlindResult {
        operation: "blind_search".into(),
        similarity: sim,
        success: sim > 0.5,
    });

    // Blind transform
    let a = HyperVector::from_seed(dimension, "a");
    let t = HyperVector::from_seed(dimension, "t");
    let sim = a.bind(&t).bind(&t).similarity(&a);
    results.push(BlindResult {
        operation: "blind_transform".into(),
        similarity: sim,
        success: sim > 0.7,
    });

    // Blind composition
    let role = HyperVector::from_seed(dimension, "role");
    let filler = HyperVector::from_seed(dimension, "filler");
    let sim = role.bind(&filler).bind(&role).similarity(&filler);
    results.push(BlindResult {
        operation: "blind_composition".into(),
        similarity: sim,
        success: sim > 0.8,
    });

    results
}

// ============================================================================
// 4. Hardware Comparison
// ============================================================================

mod scalar {
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

mod simd {
    pub fn dot_product_unrolled(a: &[f32], b: &[f32]) -> f32 {
        let chunks = a.len() / 4;
        let mut sum = 0.0f32;
        for i in 0..chunks {
            let s = i * 4;
            sum += a[s] * b[s] + a[s + 1] * b[s + 1] + a[s + 2] * b[s + 2] + a[s + 3] * b[s + 3];
        }
        for i in (chunks * 4)..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        let chunks = a.len() / 8;
        let mut sum = unsafe { _mm256_setzero_ps() };
        for i in 0..chunks {
            unsafe {
                let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
            }
        }
        let mut result = [0.0f32; 8];
        unsafe {
            _mm256_storeu_ps(result.as_mut_ptr(), sum);
        }
        let mut total: f32 = result.iter().sum();
        for i in (chunks * 8)..a.len() {
            total += a[i] * b[i];
        }
        total
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        dot_product_unrolled(a, b)
    }
}

// ============================================================================
// 5. Semantic Analogy Test (King - Man + Woman = Queen)
// ============================================================================

#[derive(Debug)]
struct AnalogyResult {
    query: String,
    expected: String,
    rank: usize,
    similarity: f32,
}

fn test_analogies(dimension: usize) -> Vec<AnalogyResult> {
    let concepts: HashMap<&str, HyperVector> = [
        "king", "queen", "man", "woman", "prince", "princess", "boy", "girl", "father", "mother",
        "brother", "sister", "husband", "wife",
    ]
    .iter()
    .map(|&s| (s, HyperVector::from_seed(dimension, s)))
    .collect();

    [
        ("king - man + woman", "queen", &["king", "man", "woman"]),
        (
            "prince - boy + girl",
            "princess",
            &["prince", "boy", "girl"],
        ),
        (
            "father - man + woman",
            "mother",
            &["father", "man", "woman"],
        ),
        (
            "husband - man + woman",
            "wife",
            &["husband", "man", "woman"],
        ),
    ]
    .iter()
    .map(|(q, exp, terms)| {
        let mut result = HyperVector::zero(dimension);
        for (i, c) in concepts[terms[0]].components.iter().enumerate() {
            result.components[i] =
                c - concepts[terms[1]].components[i] + concepts[terms[2]].components[i];
        }
        result.normalize();
        let mut sims: Vec<_> = concepts
            .iter()
            .filter(|(n, _)| !terms.contains(n))
            .map(|(n, v)| (*n, result.similarity(v)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rank = sims.iter().position(|(n, _)| *n == *exp).unwrap_or(999) + 1;
        AnalogyResult {
            query: q.to_string(),
            expected: exp.to_string(),
            rank,
            similarity: concepts[*exp].similarity(&result),
        }
    })
    .collect()
}

// ============================================================================
// Memory Efficiency Comparison
// ============================================================================

fn measure_memory_efficiency(
    dimension: usize,
    iterations: usize,
) -> Vec<(String, usize, f64, f64)> {
    let mut results = Vec::new();

    let v1 = HyperVector::random(dimension, 1);
    let v2 = HyperVector::random(dimension, 2);
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        black_box(v1.bind(&v2));
    }
    let bind_ops = iterations as f64 / start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        black_box(v1.similarity(&v2));
    }
    let sim_ops = iterations as f64 / start.elapsed().as_secs_f64();
    results.push(("dense_f32".into(), dimension * 4, bind_ops, sim_ops));

    let b1 = BinaryHyperVector::random(dimension, 1);
    let b2 = BinaryHyperVector::random(dimension, 2);
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        black_box(b1.bind(&b2));
    }
    let bind_ops = iterations as f64 / start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        black_box(b1.similarity(&b2));
    }
    let sim_ops = iterations as f64 / start.elapsed().as_secs_f64();
    results.push(("binary".into(), b1.memory_bytes(), bind_ops, sim_ops));

    results
}

// ============================================================================
// Criterion Benchmarks
// ============================================================================

fn bench_capacity_curve(c: &mut Criterion) {
    let mut group = c.benchmark_group("capacity_curve");
    group
        .measurement_time(Duration::from_secs(10))
        .sample_size(10);

    for dim in [1024, 4096, 10000] {
        group.bench_with_input(BenchmarkId::new("store", dim), &dim, |b, &d| {
            b.iter(|| {
                let mut m = HolographicMemory::new(d);
                m.store(&HyperVector::random(d, 1), &HyperVector::random(d, 2));
                black_box(m)
            })
        });
        group.bench_with_input(BenchmarkId::new("retrieve", dim), &dim, |b, &d| {
            let mut m = HolographicMemory::new(d);
            let pairs: Vec<_> = (0..100)
                .map(|i| {
                    let k = HyperVector::random(d, i * 2);
                    let v = HyperVector::random(d, i * 2 + 1);
                    m.store(&k, &v);
                    (k, v)
                })
                .collect();
            b.iter(|| black_box(m.retrieve(&pairs[50].0, &pairs[50].1)))
        });
    }
    group.finish();
}

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimension_scaling");
    group.measurement_time(Duration::from_secs(5));

    for dim in [1024, 4096, 10000, 100000] {
        group.throughput(Throughput::Elements(dim as u64));
        let v1 = HyperVector::random(dim, 1);
        let v2 = HyperVector::random(dim, 2);
        group.bench_with_input(BenchmarkId::new("bind", dim), &(&v1, &v2), |b, (a, c)| {
            b.iter(|| black_box(a.bind(c)))
        });
        group.bench_with_input(BenchmarkId::new("bundle", dim), &(&v1, &v2), |b, (a, c)| {
            b.iter(|| black_box(a.bundle(c)))
        });
        group.bench_with_input(BenchmarkId::new("permute", dim), &v1, |b, v| {
            b.iter(|| black_box(v.permute(5)))
        });
    }
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    for dim in [1024, 10000] {
        let dv1 = HyperVector::random(dim, 1);
        let dv2 = HyperVector::random(dim, 2);
        let bv1 = BinaryHyperVector::random(dim, 1);
        let bv2 = BinaryHyperVector::random(dim, 2);
        group.bench_with_input(
            BenchmarkId::new("dense_bind", dim),
            &(&dv1, &dv2),
            |b, (a, c)| b.iter(|| black_box(a.bind(c))),
        );
        group.bench_with_input(
            BenchmarkId::new("binary_bind", dim),
            &(&bv1, &bv2),
            |b, (a, c)| b.iter(|| black_box(a.bind(c))),
        );
        group.bench_with_input(
            BenchmarkId::new("dense_sim", dim),
            &(&dv1, &dv2),
            |b, (a, c)| b.iter(|| black_box(a.similarity(c))),
        );
        group.bench_with_input(
            BenchmarkId::new("binary_sim", dim),
            &(&bv1, &bv2),
            |b, (a, c)| b.iter(|| black_box(a.similarity(c))),
        );
    }
    group.finish();
}

fn bench_hardware_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("hardware");
    for dim in [1024, 4096, 10000] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).cos()).collect();
        group.bench_with_input(
            BenchmarkId::new("scalar", dim),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(scalar::dot_product(a, b))),
        );
        group.bench_with_input(
            BenchmarkId::new("simd_unrolled", dim),
            &(&a, &b),
            |bench, (a, b)| bench.iter(|| black_box(simd::dot_product_unrolled(a, b))),
        );
        group.bench_with_input(BenchmarkId::new("avx2", dim), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(simd::dot_product_avx2(a, b)))
        });
    }
    group.finish();
}

fn bench_analogies(c: &mut Criterion) {
    let mut group = c.benchmark_group("analogies");
    for dim in [1024, 4096, 10000] {
        let king = HyperVector::from_seed(dim, "king");
        let man = HyperVector::from_seed(dim, "man");
        let woman = HyperVector::from_seed(dim, "woman");
        let queen = HyperVector::from_seed(dim, "queen");
        group.bench_with_input(BenchmarkId::new("king-man+woman", dim), &dim, |b, &d| {
            b.iter(|| {
                let mut r = HyperVector::zero(d);
                for (i, c) in king.components.iter().enumerate() {
                    r.components[i] = c - man.components[i] + woman.components[i];
                }
                r.normalize();
                black_box(r.similarity(&queen))
            })
        });
    }
    group.finish();
}

fn bench_blind_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("blind_ops");
    for dim in [1024, 4096, 10000] {
        let mut db = HyperVector::zero(dim);
        let keys: Vec<_> = (0..100)
            .map(|i| HyperVector::from_seed(dim, &format!("k{}", i)))
            .collect();
        let vals: Vec<_> = (0..100)
            .map(|i| HyperVector::from_seed(dim, &format!("v{}", i)))
            .collect();
        for (k, v) in keys.iter().zip(&vals) {
            db.add_inplace(&k.bind(v));
        }
        group.bench_with_input(
            BenchmarkId::new("blind_search", dim),
            &(&db, &keys[50], &vals[50]),
            |b, (d, k, v)| b.iter(|| black_box(d.bind(k).similarity(v))),
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.05).sample_size(100).measurement_time(Duration::from_secs(5));
    targets = bench_capacity_curve, bench_dimension_scaling, bench_memory_efficiency, bench_hardware_comparison, bench_analogies, bench_blind_ops
);
criterion_main!(benches);

// ============================================================================
// Validation Tests & White Paper Data Generation
// ============================================================================

#[allow(dead_code)]
fn run_validation_tests() {
    println!("\n=== HOLOGRAPHIC PROPERTY VALIDATION ===\n");

    println!("1. CAPACITY CURVE (crash point where similarity < 0.75)");
    for dim in [1024, 4096, 10000] {
        let max = (dim as f64 * 0.3) as usize;
        let curve = run_capacity_curve_test(dim, max, (max / 20).max(1));
        println!(
            "   D={}: crash={:?}, final_sim={:.4}",
            dim,
            curve.crash_point,
            curve
                .points
                .last()
                .map(|p| p.mean_similarity)
                .unwrap_or(0.0)
        );
    }

    println!("\n2. ORTHOGONALITY (mean dot product should be ~0)");
    for dim in [1000, 10000] {
        let r = test_orthogonality(dim, 100);
        println!(
            "   D={}: mean={:.6}, std={:.6}, pass={}",
            dim,
            r.mean_dot,
            r.std_dev,
            r.mean_dot.abs() < 0.1
        );
    }

    println!("\n3. BLIND OPERATIONS");
    for r in test_blind_operations(10000) {
        println!(
            "   {}: sim={:.4}, pass={}",
            r.operation, r.similarity, r.success
        );
    }

    println!("\n4. SEMANTIC ANALOGIES");
    for r in test_analogies(10000) {
        println!(
            "   {}: expected={}, rank={}, sim={:.4}",
            r.query, r.expected, r.rank, r.similarity
        );
    }

    println!("\n5. MEMORY EFFICIENCY (D=10000)");
    for (enc, bytes, bind, sim) in measure_memory_efficiency(10000, 10000) {
        println!(
            "   {}: {} bytes, bind={:.0}/s, sim={:.0}/s",
            enc, bytes, bind, sim
        );
    }

    println!(
        "\n=== WHITE PAPER CSV ===\n{}",
        run_capacity_curve_test(10000, 3000, 100).to_csv()
    );
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_capacity_curve() {
        let c = run_capacity_curve_test(1024, 100, 10);
        assert!(!c.points.is_empty() && c.points[0].mean_similarity > 0.9);
    }

    #[test]
    fn test_orthogonality() {
        let r = test_orthogonality(10000, 50);
        assert!(r.mean_dot.abs() < 0.2);
    }

    #[test]
    fn test_blind_ops() {
        assert!(
            test_blind_operations(10000)
                .iter()
                .filter(|r| r.success)
                .count()
                >= 2
        );
    }

    #[test]
    fn test_binary_vectors() {
        let v1 = BinaryHyperVector::random(10000, 1);
        let v2 = BinaryHyperVector::random(10000, 2);
        assert!((v1.similarity(&v1) - 1.0).abs() < 0.001);
        let sim = v1.similarity(&v2);
        assert!(sim > 0.4 && sim < 0.6);
    }

    #[test]
    fn test_memory_ratio() {
        let dense = HyperVector::random(10000, 1);
        let binary = BinaryHyperVector::random(10000, 1);
        assert!(dense.components.len() * 4 / binary.memory_bytes() > 20);
    }

    #[test]
    #[ignore]
    fn run_full_validation() {
        run_validation_tests();
    }
}
