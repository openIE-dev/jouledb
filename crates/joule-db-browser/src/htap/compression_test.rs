//! HTAP Compression Test Utilities
//!
//! Data generators and compression measurement tools

#[cfg(feature = "compression-test")]
use rand::Rng;
use std::collections::HashMap;

#[cfg(feature = "compression-test")]
/// Data generation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
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

#[cfg(feature = "compression-test")]
/// Compression test results
#[derive(Debug, Clone)]
pub struct CompressionResults {
    pub data_type: DataType,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_percent: f64,
    pub target_size: usize,
}

#[cfg(feature = "compression-test")]
/// Data generator trait
pub trait DataGenerator {
    fn generate(&mut self, target_size: usize) -> Vec<Vec<u8>>;
    fn data_type(&self) -> DataType;
}

#[cfg(feature = "compression-test")]
/// Compression test runner
pub struct CompressionTestRunner;

impl CompressionTestRunner {
    pub fn new() -> Self {
        Self
    }

    /// Run compression test for a data type and size (convenience method)
    pub async fn run_compression_test(
        &self,
        data_type: DataType,
        target_size: usize,
    ) -> CompressionResults {
        let generator = SpecificDataGenerator::new(data_type);
        self.run_test(generator, target_size).await
    }

    /// Run compression test for a data type and size using actual compression algorithms
    pub async fn run_test<G: DataGenerator>(
        &self,
        mut generator: G,
        target_size: usize,
    ) -> CompressionResults {
        // Generate data
        let data = generator.generate(target_size);

        // Calculate original size (row format)
        let original_size: usize = data.iter().map(|row| row.len()).sum();

        // Apply actual compression algorithms based on data type
        let compressed_size = match generator.data_type() {
            DataType::Random => {
                // Random data: dictionary compression (won't help much)
                self.apply_dictionary_compression_rows(&data, 64)
            }
            DataType::Structured => {
                // Structured: dictionary encoding on fixed-width fields
                self.apply_dictionary_compression_rows(&data, 60)
            }
            DataType::Sparse => {
                // Sparse: null bitmap + only non-null rows
                let non_null_count = data
                    .iter()
                    .filter(|row| !row.iter().all(|&b| b == 0))
                    .count();
                let unique_values: std::collections::HashSet<Vec<u8>> = data
                    .iter()
                    .filter(|row| !row.iter().all(|&b| b == 0))
                    .cloned()
                    .collect();
                let bitmap_size = (data.len() + 7) / 8;
                let dict_size: usize = unique_values.iter().map(|v| v.len()).sum();
                bitmap_size + dict_size + non_null_count * 4
            }
            DataType::Repetitive => {
                // Repetitive: dictionary encoding (very few unique values)
                self.apply_dictionary_compression_rows(&data, 64)
            }
            DataType::TimeSeries => {
                // Time-series: delta encoding
                let flat: Vec<u8> = data.iter().flat_map(|row| row.iter().copied()).collect();
                self.apply_delta_compression(&flat)
            }
            DataType::LowCardinality => {
                // Low cardinality: dictionary encoding
                self.apply_dictionary_compression_rows(&data, 64)
            }
            DataType::Mixed => {
                // Mixed: dictionary compression
                self.apply_dictionary_compression_rows(&data, 128)
            }
        };

        let compression_ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            1.0
        };

        let compression_percent = (1.0 - (compressed_size as f64 / original_size as f64)) * 100.0;

        CompressionResults {
            data_type: generator.data_type(),
            original_size,
            compressed_size,
            compression_ratio,
            compression_percent,
            target_size,
        }
    }

    /// Dictionary compression on row data: build unique chunk dictionary + indices
    fn apply_dictionary_compression_rows(&self, data: &[Vec<u8>], chunk_size: usize) -> usize {
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
    fn apply_delta_compression(&self, data: &[u8]) -> usize {
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
}

#[cfg(feature = "compression-test")]
/// Generate comprehensive compression report
pub async fn generate_compression_report<G: DataGenerator + Clone>(
    generator_factory: impl Fn(DataType) -> G,
) -> Vec<CompressionResults> {
    let sizes = vec![
        256 * 1024 * 1024,      // 256MB (0.25GB)
        512 * 1024 * 1024,      // 512MB (0.5GB)
        1024 * 1024 * 1024,     // 1GB
        2 * 1024 * 1024 * 1024, // 2GB
        5 * 1024 * 1024 * 1024, // 5GB
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

    let runner = CompressionTestRunner::new();
    let mut results = Vec::new();

    for size in sizes {
        for data_type in &data_types {
            let generator = generator_factory(*data_type);
            let result = runner.run_test(generator, size).await;
            results.push(result);
        }
    }

    results
}

#[cfg(feature = "compression-test")]
/// Format compression results as a table
pub fn format_results_table(results: &[CompressionResults]) -> String {
    let mut output = String::new();

    output.push_str("\n=== HTAP Compression Test Results ===\n\n");
    output.push_str(&format!(
        "{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}\n",
        "Data Type", "Original Size", "Compressed Size", "Ratio", "Compression %", "Target Size"
    ));
    output.push_str(&format!("{}\n", "-".repeat(95)));

    for result in results {
        let original_mb = result.original_size as f64 / (1024.0 * 1024.0);
        let compressed_mb = result.compressed_size as f64 / (1024.0 * 1024.0);
        let target_mb = result.target_size as f64 / (1024.0 * 1024.0);

        output.push_str(&format!(
            "{:<20} {:<15.2}MB {:<15.2}MB {:<15.2}x {:<15.2}% {:<15.2}MB\n",
            format!("{:?}", result.data_type),
            original_mb,
            compressed_mb,
            result.compression_ratio,
            result.compression_percent,
            target_mb
        ));
    }

    output
}

#[cfg(feature = "compression-test")]
/// Analyze 5GB compression results
pub fn analyze_5gb_compression(results: &[CompressionResults]) -> String {
    let mut output = String::new();

    output.push_str("\n=== 5GB Compression Analysis ===\n\n");

    let five_gb_results: Vec<_> = results
        .iter()
        .filter(|r| r.target_size == 5 * 1024 * 1024 * 1024)
        .collect();

    for result in five_gb_results {
        let original_gb = result.original_size as f64 / (1024.0 * 1024.0 * 1024.0);
        let compressed_gb = result.compressed_size as f64 / (1024.0 * 1024.0 * 1024.0);
        let compressed_mb = result.compressed_size as f64 / (1024.0 * 1024.0);

        output.push_str(&format!(
            "{:<20}: {:.2}GB → {:.2}GB ({:.2}MB) | Ratio: {:.2}x | Compression: {:.2}%\n",
            format!("{:?}", result.data_type),
            original_gb,
            compressed_gb,
            compressed_mb,
            result.compression_ratio,
            result.compression_percent
        ));

        // Check if we achieved 1000x compression (5GB → 5MB)
        if result.compression_ratio >= 1000.0 {
            output.push_str("  ✅ ACHIEVED 1000x COMPRESSION!\n");
        } else if result.compression_ratio >= 100.0 {
            output.push_str("  ⚠️  Achieved 100x+ compression (not 1000x)\n");
        } else {
            output.push_str("  ❌ Did not achieve 100x compression\n");
        }
    }

    output
}

#[cfg(feature = "compression-test")]
/// Specific data generator implementations
pub struct SpecificDataGenerator {
    data_type: DataType,
}

impl SpecificDataGenerator {
    pub fn new(data_type: DataType) -> Self {
        Self { data_type }
    }
}

impl DataGenerator for SpecificDataGenerator {
    fn generate(&mut self, target_size: usize) -> Vec<Vec<u8>> {
        let mut rng = rand::thread_rng();

        match self.data_type {
            DataType::Random => {
                // Random data: high entropy
                let num_rows = target_size / 1024; // 1KB rows
                (0..num_rows)
                    .map(|_| {
                        let mut row = vec![0u8; 1024];
                        rng.fill(&mut row[..]);
                        row
                    })
                    .collect()
            }
            DataType::Structured => {
                // Structured data: tables with schemas
                let num_rows = target_size / 256; // 256B rows
                (0..num_rows)
                    .map(|i| {
                        // Simulate structured data: ID, name, value
                        format!("{:08x}:name_{}:value_{}", i, i % 100, i % 1000).into_bytes()
                    })
                    .collect()
            }
            DataType::Sparse => {
                // Sparse data: mostly NULLs
                let num_rows = target_size / 1024;
                (0..num_rows)
                    .map(|i| {
                        if i % 100 == 0 {
                            // 1% non-null
                            format!("data_{}", i).into_bytes()
                        } else {
                            vec![0u8; 1024] // NULL padding
                        }
                    })
                    .collect()
            }
            DataType::Repetitive => {
                // Repetitive data: few unique values
                let num_rows = target_size / 64;
                let unique_values = vec!["value_a", "value_b", "value_c", "value_d", "value_e"];
                (0..num_rows)
                    .map(|i| {
                        let value = unique_values[i % unique_values.len()];
                        format!("row_{}:{}", i, value).into_bytes()
                    })
                    .collect()
            }
            DataType::TimeSeries => {
                // Time-series: delta-encoded
                let num_rows = target_size / 32;
                let mut base_value = 1000i64;
                (0..num_rows)
                    .map(|i| {
                        let delta = (i % 10) as i64 - 5; // Small deltas
                        base_value += delta;
                        format!("ts_{}:{}", i, base_value).into_bytes()
                    })
                    .collect()
            }
            DataType::LowCardinality => {
                // Low cardinality: dictionary-encoded
                let num_rows = target_size / 16;
                let categories = vec!["cat_a", "cat_b", "cat_c"];
                (0..num_rows)
                    .map(|i| {
                        let cat = categories[i % categories.len()];
                        format!("id_{}:{}", i, cat).into_bytes()
                    })
                    .collect()
            }
            DataType::Mixed => {
                // Mixed: combination of types
                let num_rows = target_size / 128;
                (0..num_rows)
                    .map(|i| {
                        match i % 4 {
                            0 => format!("random_{}", rng.random::<u64>()),
                            1 => format!("structured_{}:value_{}", i, i % 100),
                            2 => format!("repetitive_{}", i % 10),
                            _ => format!("timeseries_{}:{}", i, 1000 + i),
                        }
                        .into_bytes()
                    })
                    .collect()
            }
        }
    }

    fn data_type(&self) -> DataType {
        self.data_type
    }
}
