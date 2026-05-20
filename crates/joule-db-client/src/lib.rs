//! # joule-db-client
//!
//! The core Rust client SDK for [JouleDB](https://joule-db.dev).
//!
//! This crate implements the binary wire protocol and provides both low-level
//! ([`connection::Connection`]) and high-level ([`client::Client`]) APIs for
//! communicating with an JouleDB TCP server.
//!
//! ## Quick Start
//!
//! ```no_run
//! # async fn example() -> joule_db_client::error::Result<()> {
//! use joule_db_client::client::Client;
//!
//! let client = Client::connect("127.0.0.1", 9000).await?;
//!
//! // Key-value operations
//! client.put("greeting", b"hello world", None).await?;
//! let value = client.get("greeting").await?;
//!
//! // SQL queries
//! let result = client.query("SELECT * FROM users", &[]).await?;
//! println!("{} rows returned", result.row_count);
//!
//! // Transactions
//! let tx = client.begin().await?;
//! tx.execute("INSERT INTO users (name) VALUES (?)", &[serde_json::json!("Alice")]).await?;
//! tx.commit().await?;
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod connection;
pub mod error;
pub mod pool;
pub mod protocol;

// Re-export the most commonly used types at the crate root for convenience.
pub use client::{Client, QueryResult};
pub use connection::{Connection, ConnectionConfig};
pub use error::ClientError;
pub use pool::{ConnectionPool, PoolConfig, PoolStats, PooledConnection};
pub use protocol::{Message, MessageType};
