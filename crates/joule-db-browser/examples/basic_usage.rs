//! Basic usage example for joule-db-browser
//!
//! Demonstrates core database operations

use joule_db_browser::{AsyncEngine, persistence::MemoryBackend};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create database with in-memory backend
    let backend = MemoryBackend::new();
    let engine = AsyncEngine::new(Arc::new(std::sync::RwLock::new(backend)));

    // Basic operations
    engine.put(b"user:1", b"Alice").await?;
    engine.put(b"user:2", b"Bob").await?;

    let value = engine.get(b"user:1").await?;
    println!(
        "Retrieved: {:?}",
        value.map(|v| String::from_utf8_lossy(&v).to_string())
    );

    // Delete
    engine.delete(b"user:2").await?;

    Ok(())
}
