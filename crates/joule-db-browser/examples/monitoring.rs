//! Performance monitoring example
//!
//! Demonstrates metrics collection and DevTools integration

use joule_db_browser::monitoring::PerformanceMonitor;
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create performance monitor
    let monitor = Arc::new(PerformanceMonitor::new(1000));

    // Record query execution times
    for _ in 0..10 {
        let start = std::time::Instant::now();

        // Simulate query execution (blocking for simplicity in example)
        std::thread::sleep(Duration::from_millis(50));

        let duration = start.elapsed();
        monitor.record_query_time(duration);
    }

    // Update metrics
    monitor.update_gpu_utilization(0.85);
    monitor.update_memory_usage(1024 * 1024 * 256); // 256MB

    // Get metrics
    let metrics = monitor.get_metrics();
    println!("Total Queries: {}", metrics.total_queries);
    println!("GPU Utilization: {:.1}%", metrics.gpu_utilization * 100.0);
    println!("Memory Usage: {} MB", metrics.memory_usage / 1024 / 1024);

    // DevTools integration is available in wasm32 target
    #[cfg(target_arch = "wasm32")]
    {
        use joule_db_browser::DevToolsIntegration;
        let devtools = DevToolsIntegration::new(monitor);
        devtools.export_metrics();
    }

    Ok(())
}
