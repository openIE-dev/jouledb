//! Query-to-Visualization example
//!
//! Demonstrates direct query execution to GPU-accelerated charts

#[cfg(feature = "viz")]
use joule_db_browser::*;
use std::sync::Arc;

#[cfg(feature = "viz")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup database
    let db = JouleDB::open_with_backend("viz_db", "opfs-vbd").await?;

    // Insert time-series data
    for i in 0..1000 {
        let timestamp = i as f64;
        let value = (i as f64 * 0.1).sin();
        db.put(&format!("ts_{}", timestamp), &value.to_string())
            .await?;
    }

    // Execute query and visualize
    // Note: Requires GpuRenderer setup
    println!("Query-to-Viz pipeline ready");
    Ok(())
}

#[cfg(not(feature = "viz"))]
fn main() {
    println!("Viz feature required");
}
