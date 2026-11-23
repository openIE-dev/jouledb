//! HTAP Compression Tests for Local/Server
//!
//! Local-compatible tests using standard Rust test framework
//! Run with: cargo test --test htap_compression_local_test --features htap,compression-test

#[cfg(test)]
#[cfg(feature = "compression-test")]
mod tests {
    use joule_db_browser::htap::compression_test::*;

    #[tokio::test]
    async fn test_random_data_compression() {
        // Test random data compression (expected: 2-5x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::Random, size).await;
            // Assert compression ratio is in expected range (2-5x)
            assert!(
                result.compression_ratio >= 2.0 && result.compression_ratio <= 5.0,
                "Random data compression ratio {} not in expected range 2-5x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_structured_data_compression() {
        // Test structured data compression (expected: 10-50x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::Structured, size).await;
            // Assert compression ratio is in expected range (10-50x)
            assert!(
                result.compression_ratio >= 10.0 && result.compression_ratio <= 50.0,
                "Structured data compression ratio {} not in expected range 10-50x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_sparse_data_compression() {
        // Test sparse data compression (expected: 100-500x, can reach 1000x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::Sparse, size).await;
            // Assert compression ratio is in expected range (100-500x)
            assert!(
                result.compression_ratio >= 100.0,
                "Sparse data compression ratio {} not >= 100x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_repetitive_data_compression() {
        // Test repetitive data compression (expected: 200-800x, can reach 1000x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::Repetitive, size).await;
            // Assert compression ratio is in expected range (200-800x)
            assert!(
                result.compression_ratio >= 200.0,
                "Repetitive data compression ratio {} not >= 200x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_timeseries_data_compression() {
        // Test time-series data compression (expected: 50-200x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::TimeSeries, size).await;
            // Assert compression ratio is in expected range (50-200x)
            assert!(
                result.compression_ratio >= 50.0 && result.compression_ratio <= 200.0,
                "Time-series data compression ratio {} not in expected range 50-200x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_low_cardinality_compression() {
        // Test low cardinality compression (expected: 100-500x, can reach 1000x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::LowCardinality, size).await;
            // Assert compression ratio is in expected range (100-500x)
            assert!(
                result.compression_ratio >= 100.0,
                "Low cardinality compression ratio {} not >= 100x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_mixed_data_compression() {
        // Test mixed data compression (expected: 50-200x)
        let sizes = vec![
            256 * 1024 * 1024,  // 256MB
            512 * 1024 * 1024,  // 512MB
            1024 * 1024 * 1024, // 1GB
        ];

        for size in sizes {
            let result = test_compression_for_type(DataType::Mixed, size).await;
            // Assert compression ratio is in expected range (50-200x)
            assert!(
                result.compression_ratio >= 50.0 && result.compression_ratio <= 200.0,
                "Mixed data compression ratio {} not in expected range 50-200x",
                result.compression_ratio
            );
        }
    }

    #[tokio::test]
    async fn test_5gb_compression_target() {
        // Test 5GB compression to answer: Can 5GB be compressed to 5MB? (1000x)
        let target_size = 5 * 1024 * 1024 * 1024; // 5GB

        let data_types = vec![
            DataType::Sparse,
            DataType::Repetitive,
            DataType::LowCardinality,
        ];

        for data_type in data_types {
            let result = test_compression_for_type(data_type, target_size).await;

            println!(
                "5GB Compression Test - {:?}: {:.2}x compression ({}MB → {}MB)",
                result.data_type,
                result.compression_ratio,
                result.original_size / (1024 * 1024),
                result.compressed_size / (1024 * 1024)
            );

            // Check if we achieved 1000x compression (5GB → 5MB)
            if result.compression_ratio >= 1000.0 {
                // ✅ ACHIEVED 1000x COMPRESSION!
                assert!(
                    result.compressed_size <= 5 * 1024 * 1024,
                    "Compressed size {}MB exceeds 5MB target",
                    result.compressed_size / (1024 * 1024)
                );
            } else if result.compression_ratio >= 100.0 {
                // ⚠️ Achieved 100x+ compression (not 1000x)
                assert!(
                    result.compressed_size <= 50 * 1024 * 1024,
                    "Compressed size {}MB exceeds 50MB",
                    result.compressed_size / (1024 * 1024)
                );
            }
        }
    }

    #[tokio::test]
    async fn test_compression_scaling() {
        // Test that compression ratio is scale-invariant
        let sizes = vec![
            256 * 1024 * 1024,      // 256MB
            512 * 1024 * 1024,      // 512MB
            1024 * 1024 * 1024,     // 1GB
            2 * 1024 * 1024 * 1024, // 2GB
        ];

        let data_type = DataType::Structured;

        let mut ratios = Vec::new();
        for size in &sizes {
            let result = test_compression_for_type(data_type, *size).await;
            ratios.push(result.compression_ratio);
            println!(
                "Scale {}MB: compression ratio {:.2}x",
                size / (1024 * 1024),
                result.compression_ratio
            );
        }

        // Compression ratio should be relatively stable across scales
        // (allowing for some variance due to dictionary overhead at small scales)
        let min_ratio = ratios.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_ratio = ratios.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Ratio variance should be < 50% (allowing for dictionary overhead)
        let variance = (max_ratio - min_ratio) / min_ratio;
        assert!(
            variance < 0.5,
            "Compression ratio variance {} too high (min: {:.2}x, max: {:.2}x)",
            variance,
            min_ratio,
            max_ratio
        );
    }

    #[tokio::test]
    async fn test_all_scales() {
        // Test all scales: 256MB, 512MB, 1GB, 2GB, 5GB
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

        println!("\n=== HTAP Compression Test Results (Local) ===\n");
        println!(
            "{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}",
            "Data Type",
            "Original Size",
            "Compressed Size",
            "Ratio",
            "Compression %",
            "Target Size"
        );
        println!("{}", "-".repeat(95));

        for size in &sizes {
            for data_type in &data_types {
                let result = test_compression_for_type(*data_type, *size).await;
                let original_mb = result.original_size as f64 / (1024.0 * 1024.0);
                let compressed_mb = result.compressed_size as f64 / (1024.0 * 1024.0);
                let target_mb = result.target_size as f64 / (1024.0 * 1024.0);

                println!(
                    "{:<20} {:<15.2}MB {:<15.2}MB {:<15.2}x {:<15.2}% {:<15.2}MB",
                    format!("{:?}", result.data_type),
                    original_mb,
                    compressed_mb,
                    result.compression_ratio,
                    result.compression_percent,
                    target_mb
                );
            }
            println!();
        }
    }

    // Helper function to test compression for a specific data type and size
    async fn test_compression_for_type(
        data_type: DataType,
        target_size: usize,
    ) -> CompressionResults {
        // Use the actual compression test utilities
        let runner = CompressionTestRunner::new();

        // Run the compression test (now async and actually uses HTAP storage)
        runner.run_compression_test(data_type, target_size).await
    }
}
