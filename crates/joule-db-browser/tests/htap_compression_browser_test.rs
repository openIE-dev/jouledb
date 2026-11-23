//! HTAP Compression Tests for Browser
//!
//! Browser-compatible tests using wasm-bindgen-test
//! Run with: wasm-pack test --headless --chrome --features htap,compression-test

use wasm_bindgen_test::*;

#[cfg(feature = "compression-test")]
use joule_db_browser::htap::compression_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_random_data_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::Random, size).await;
        assert!(
            result.compression_ratio >= 1.0,
            "Random data compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_structured_data_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::Structured, size).await;
        assert!(
            result.compression_ratio >= 5.0,
            "Structured data compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_sparse_data_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::Sparse, size).await;
        assert!(
            result.compression_ratio >= 10.0,
            "Sparse data compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_repetitive_data_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::Repetitive, size).await;
        assert!(
            result.compression_ratio >= 10.0,
            "Repetitive data compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_timeseries_data_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::TimeSeries, size).await;
        assert!(
            result.compression_ratio >= 5.0,
            "TimeSeries data compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_low_cardinality_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::LowCardinality, size).await;
        assert!(
            result.compression_ratio >= 10.0,
            "LowCardinality compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_mixed_data_compression() {
    let sizes = vec![
        256 * 1024 * 1024,  // 256MB
        512 * 1024 * 1024,  // 512MB
        1024 * 1024 * 1024, // 1GB
    ];

    for size in sizes {
        let result = test_compression_for_type(DataType::Mixed, size).await;
        assert!(
            result.compression_ratio >= 3.0,
            "Mixed data compression ratio too low: {}",
            result.compression_ratio
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_5gb_compression_target() {
    let target_size = 5 * 1024 * 1024 * 1024; // 5GB

    let data_types = vec![
        DataType::Sparse,
        DataType::Repetitive,
        DataType::LowCardinality,
    ];

    for data_type in data_types {
        let result = test_compression_for_type(data_type, target_size).await;
        // Verify meaningful compression occurred
        assert!(
            result.compression_ratio >= 10.0,
            "5GB {:?} compression ratio too low: {}",
            data_type,
            result.compression_ratio
        );
        assert!(
            result.compressed_size < result.original_size,
            "Compressed size should be smaller than original"
        );
    }
}

#[wasm_bindgen_test]
#[cfg(feature = "compression-test")]
async fn test_compression_scaling() {
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
        assert!(
            result.compression_ratio >= 1.0,
            "Compression ratio should be >= 1.0"
        );
        ratios.push(result.compression_ratio);
    }

    // Compression ratio should be relatively stable across scales
    let min_ratio = ratios.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_ratio = ratios.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let variance = (max_ratio - min_ratio) / min_ratio;
    assert!(
        variance < 0.5,
        "Compression ratio variance too high: {}",
        variance
    );
}

#[cfg(feature = "compression-test")]
async fn test_compression_for_type(data_type: DataType, target_size: usize) -> CompressionResults {
    let runner = CompressionTestRunner::new();
    runner.run_compression_test(data_type, target_size)
}
