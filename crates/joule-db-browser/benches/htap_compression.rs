//! HTAP Compression Benchmark Suite
//!
//! Tests compression ratios across different data types and scales
//! Answers: Can we compress 5GB of data into smaller sizes using HTAP principles?

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Data generation strategies
#[derive(Debug, Clone, Copy)]
enum DataType {
    /// Random data (high entropy, low compression)
    Random,
    /// Structured data (tables, schemas, medium compression)
    Structured,
    /// Sparse data (mostly NULLs, high compression)
    Sparse,
    /// Repetitive data (few unique values, high compression)
    Repetitive,
    /// Time-series data (delta-encoded, high compression)
    TimeSeries,
    /// Low cardinality (dictionary-encoded, high compression)
    LowCardinality,
    /// Mixed data (combination of types)
    Mixed,
}

/// Data generator for different types
struct DataGenerator {
    rng: StdRng,
    data_type: DataType,
}

impl DataGenerator {
    fn new(data_type: DataType, seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            data_type,
        }
    }

    /// Generate data of specified size (in bytes)
    fn generate(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        match self.data_type {
            DataType::Random => self.generate_random(target_size),
            DataType::Structured => self.generate_structured(target_size),
            DataType::Sparse => self.generate_sparse(target_size),
            DataType::Repetitive => self.generate_repetitive(target_size),
            DataType::TimeSeries => self.generate_timeseries(target_size),
            DataType::LowCardinality => self.generate_low_cardinality(target_size),
            DataType::Mixed => self.generate_mixed(target_size),
        }
    }

    /// Random data: High entropy, low compression
    fn generate_random(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        while current_size < target_size {
            let row_size = 1024; // 1KB rows
            let mut row = vec![0u8; row_size];
            self.rng.fill(&mut row[..]);
            current_size += row_size;
            data.push(row);
        }

        data
    }

    /// Structured data: Tables with schemas, medium compression
    fn generate_structured(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        // Schema: id (8 bytes), name (32 bytes), age (4 bytes), salary (8 bytes), timestamp (8 bytes)
        let row_size = 60;
        let num_rows = target_size / row_size;

        for i in 0..num_rows {
            let mut row = Vec::with_capacity(row_size);

            // ID (8 bytes, sequential)
            row.extend_from_slice(&(i as u64).to_le_bytes());

            // Name (32 bytes, random but structured)
            let name = format!("User_{:08}", i);
            let mut name_bytes = name.into_bytes();
            name_bytes.resize(32, 0);
            row.extend_from_slice(&name_bytes);

            // Age (4 bytes, 18-80)
            let age = (18 + (i % 63)) as u32;
            row.extend_from_slice(&age.to_le_bytes());

            // Salary (8 bytes, 30k-200k)
            let salary = 30000 + (i % 170000) as u64;
            row.extend_from_slice(&salary.to_le_bytes());

            // Timestamp (8 bytes, sequential)
            let timestamp = 1609459200 + (i * 3600) as u64; // Unix timestamp
            row.extend_from_slice(&timestamp.to_le_bytes());

            current_size += row.len();
            data.push(row);
        }

        data
    }

    /// Sparse data: Mostly NULLs, high compression
    fn generate_sparse(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        let row_size = 1024; // 1KB rows
        let num_rows = target_size / row_size;

        for _ in 0..num_rows {
            let mut row = vec![0u8; row_size]; // All zeros (NULLs)

            // Only 1% of rows have data
            if self.rng.random::<f64>() < 0.01 {
                // Fill with random data
                self.rng.fill(&mut row[..]);
            }

            current_size += row_size;
            data.push(row);
        }

        data
    }

    /// Repetitive data: Few unique values, high compression
    fn generate_repetitive(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        // Only 10 unique values
        let unique_values: Vec<Vec<u8>> = (0..10)
            .map(|i| format!("Value_{}", i).into_bytes())
            .collect();

        let row_size = 64;
        let num_rows = target_size / row_size;

        for _ in 0..num_rows {
            let value_idx = self.rng.gen_range(0..10);
            let mut row = unique_values[value_idx].clone();
            row.resize(row_size, 0);
            current_size += row_size;
            data.push(row);
        }

        data
    }

    /// Time-series data: Delta-encoded, high compression
    fn generate_timeseries(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        // Schema: timestamp (8 bytes), value (8 bytes f64)
        let row_size = 16;
        let num_rows = target_size / row_size;

        let mut base_timestamp = 1609459200u64; // Unix timestamp
        let mut base_value = 100.0f64;

        for i in 0..num_rows {
            let mut row = Vec::with_capacity(row_size);

            // Timestamp (sequential, delta-encoded)
            base_timestamp += 1000; // 1 second increments
            row.extend_from_slice(&base_timestamp.to_le_bytes());

            // Value (small delta, predictable)
            base_value += self.rng.gen_range(-1.0..1.0);
            row.extend_from_slice(&base_value.to_le_bytes());

            current_size += row_size;
            data.push(row);
        }

        data
    }

    /// Low cardinality: Dictionary-encoded, high compression
    fn generate_low_cardinality(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        // Only 100 unique categories
        let categories: Vec<String> = (0..100).map(|i| format!("Category_{:03}", i)).collect();

        let row_size = 64;
        let num_rows = target_size / row_size;

        for _ in 0..num_rows {
            let cat_idx = self.rng.gen_range(0..100);
            let mut row = categories[cat_idx].clone().into_bytes();
            row.resize(row_size, 0);
            current_size += row_size;
            data.push(row);
        }

        data
    }

    /// Mixed data: Combination of types
    fn generate_mixed(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut data = Vec::new();
        let mut current_size = 0;

        // Mix: 30% structured, 20% sparse, 20% repetitive, 15% time-series, 15% low cardinality
        let mut structured_gen = DataGenerator::new(DataType::Structured, 1);
        let mut sparse_gen = DataGenerator::new(DataType::Sparse, 2);
        let mut repetitive_gen = DataGenerator::new(DataType::Repetitive, 3);
        let mut timeseries_gen = DataGenerator::new(DataType::TimeSeries, 4);
        let mut lowcard_gen = DataGenerator::new(DataType::LowCardinality, 5);

        while current_size < target_size {
            let remaining = target_size - current_size;
            let chunk_size = (remaining / 5).min(1024 * 1024); // 1MB chunks

            match self.rng.gen_range(0..100) {
                0..=29 => {
                    let chunk = structured_gen.generate(chunk_size);
                    data.extend(chunk);
                    current_size += chunk_size;
                }
                30..=49 => {
                    let chunk = sparse_gen.generate(chunk_size);
                    data.extend(chunk);
                    current_size += chunk_size;
                }
                50..=69 => {
                    let chunk = repetitive_gen.generate(chunk_size);
                    data.extend(chunk);
                    current_size += chunk_size;
                }
                70..=84 => {
                    let chunk = timeseries_gen.generate(chunk_size);
                    data.extend(chunk);
                    current_size += chunk_size;
                }
                _ => {
                    let chunk = lowcard_gen.generate(chunk_size);
                    data.extend(chunk);
                    current_size += chunk_size;
                }
            }
        }

        data
    }
}

/// Compression test results
#[derive(Debug)]
struct CompressionResults {
    data_type: DataType,
    original_size: usize,
    compressed_size: usize,
    compression_ratio: f64,
    compression_percent: f64,
}

/// Run compression test for a data type
fn run_compression_test(data_type: DataType, target_size: usize) -> CompressionResults {
    // Generate data
    let mut generator = DataGenerator::new(data_type, 42);
    let data = generator.generate(target_size);

    // Calculate original size
    let original_size: usize = data.iter().map(|row| row.len()).sum();

    // Apply actual columnar compression algorithms to the generated data
    let compressed_size = apply_columnar_compression(&data, data_type);

    let compression_ratio = original_size as f64 / compressed_size as f64;
    let compression_percent = (1.0 - (compressed_size as f64 / original_size as f64)) * 100.0;

    CompressionResults {
        data_type,
        original_size,
        compressed_size,
        compression_ratio,
        compression_percent,
    }
}

/// Apply actual columnar compression algorithms based on data type
fn apply_columnar_compression(data: &[Vec<u8>], data_type: DataType) -> usize {
    match data_type {
        DataType::Random => {
            // Random data: dictionary compression (won't help much with high entropy)
            apply_dictionary_compression(data, 64)
        }
        DataType::Structured => {
            // Structured data: dictionary encoding on fixed-width fields
            apply_dictionary_compression(data, 60)
        }
        DataType::Sparse => {
            // Sparse data: null bitmap + store only non-null rows
            let non_null: Vec<&Vec<u8>> = data
                .iter()
                .filter(|row| !row.iter().all(|&b| b == 0))
                .collect();
            let bitmap_size = (data.len() + 7) / 8; // 1 bit per row
            let non_null_size: usize = non_null.iter().map(|row| row.len()).sum();
            bitmap_size + non_null_size
        }
        DataType::Repetitive => {
            // Repetitive data: dictionary encoding (very few unique values)
            apply_dictionary_compression(data, 64)
        }
        DataType::TimeSeries => {
            // Time-series: delta encoding on the raw bytes
            let flat: Vec<u8> = data.iter().flat_map(|row| row.iter().copied()).collect();
            apply_delta_compression(&flat)
        }
        DataType::LowCardinality => {
            // Low cardinality: dictionary encoding
            apply_dictionary_compression(data, 64)
        }
        DataType::Mixed => {
            // Mixed: dictionary compression as a general-purpose approach
            apply_dictionary_compression(data, 128)
        }
    }
}

/// Dictionary compression: build unique chunk dictionary + indices
fn apply_dictionary_compression(data: &[Vec<u8>], chunk_size: usize) -> usize {
    use std::collections::HashMap;
    let mut dictionary: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut unique_values: Vec<Vec<u8>> = Vec::new();
    let mut index_count = 0usize;

    for row in data {
        for chunk in row.chunks(chunk_size.max(1)) {
            let dict_idx = unique_values.len();
            dictionary.entry(chunk.to_vec()).or_insert_with(|| {
                unique_values.push(chunk.to_vec());
                dict_idx
            });
            index_count += 1;
        }
    }

    let dict_size: usize = unique_values.iter().map(|v| v.len()).sum();
    let indices_size = index_count * 4; // 4 bytes per u32 index
    dict_size + indices_size
}

/// Delta compression: store first value + variable-length deltas
fn apply_delta_compression(data: &[u8]) -> usize {
    if data.len() < 8 {
        return data.len();
    }
    let mut total = 8; // first value stored in full
    let mut prev = i64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]);
    for chunk in data.chunks(8).skip(1) {
        if chunk.len() == 8 {
            let current = i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            let delta = current - prev;
            total += if (-128..=127).contains(&delta) { 1 } else { 8 };
            prev = current;
        } else {
            total += chunk.len();
        }
    }
    total
}

/// Benchmark compression across scales
fn benchmark_compression(c: &mut Criterion) {
    let sizes = vec![
        1 * 1024 * 1024,  // 1MB
        10 * 1024 * 1024, // 10MB
        50 * 1024 * 1024, // 50MB
    ];

    let data_types = vec![
        DataType::Random,
        DataType::Structured,
        DataType::Sparse,
        DataType::Repetitive,
        DataType::TimeSeries,
        DataType::LowCardinality,
        DataType::Mixed,
    ];

    let mut group = c.benchmark_group("htap_compression");
    group.sample_size(10);

    for size in &sizes {
        for data_type in &data_types {
            let size_mb = *size / (1024 * 1024);
            let type_name = format!("{:?}", data_type);
            let benchmark_id = BenchmarkId::from_parameter(format!("{}MB_{}", size_mb, type_name));

            group.bench_with_input(benchmark_id, size, |b, &size| {
                b.iter(|| black_box(run_compression_test(*data_type, size)));
            });
        }
    }

    group.finish();
}

/// Generate comprehensive report
fn generate_report(c: &mut Criterion) {
    let sizes = vec![
        1 * 1024 * 1024,  // 1MB
        10 * 1024 * 1024, // 10MB
        50 * 1024 * 1024, // 50MB
    ];

    let data_types = vec![
        DataType::Random,
        DataType::Structured,
        DataType::Sparse,
        DataType::Repetitive,
        DataType::TimeSeries,
        DataType::LowCardinality,
        DataType::Mixed,
    ];

    println!("\n=== HTAP Compression Test Results ===\n");
    println!(
        "{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}",
        "Data Type", "Original Size", "Compressed Size", "Ratio", "Compression %", "Target Size"
    );
    println!("{}", "-".repeat(95));

    for size in &sizes {
        for data_type in &data_types {
            let results = run_compression_test(*data_type, *size);
            let original_mb = results.original_size as f64 / (1024.0 * 1024.0);
            let compressed_mb = results.compressed_size as f64 / (1024.0 * 1024.0);
            let target_mb = *size as f64 / (1024.0 * 1024.0);

            println!(
                "{:<20} {:<15.2}MB {:<15.2}MB {:<15.2}x {:<15.2}% {:<15.2}MB",
                format!("{:?}", results.data_type),
                original_mb,
                compressed_mb,
                results.compression_ratio,
                results.compression_percent,
                target_mb
            );
        }
        println!();
    }
}

criterion_group!(benches, benchmark_compression);
criterion_main!(benches);
