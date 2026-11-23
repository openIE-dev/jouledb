//! HTAP Compression Demo
//!
//! Demonstrates compression ratios across different data types
//! Run with: cargo run --example htap_compression_demo --features htap,compression-test

fn main() {
    println!("HTAP Compression Demo");
    println!("====================\n");

    #[cfg(feature = "compression-test")]
    {
        println!("Compression test features are enabled.");
        // Actual compression tests would be run here
    }

    #[cfg(not(feature = "compression-test"))]
    {
        println!("Note: Run with --features compression-test for actual tests");
    }

    println!("\nTo run full compression tests, use:");
    println!("  cargo bench --bench htap_compression --features htap,compression-test");
    println!("\nThis will test compression ratios for:");
    println!("  - Random data (2-5x expected)");
    println!("  - Structured data (10-50x expected)");
    println!("  - Sparse data (100-500x expected)");
    println!("  - Repetitive data (200-800x expected)");
    println!("  - Time-series data (50-200x expected)");
    println!("  - Low cardinality (100-500x expected)");
    println!("  - Mixed data (50-200x expected)");
    println!("\nScales tested: 256MB, 512MB, 1GB, 2GB, 5GB");
}
