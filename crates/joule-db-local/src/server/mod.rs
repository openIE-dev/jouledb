//! Server and client implementations for JouleDB
//!
//! This module provides network servers and clients that expose the JouleDB
//! database over the binary protocol:
//!
//! - `tcp` - High-performance binary protocol server over TCP
//! - `client` - TCP client for connecting to JouleDB servers
//!
//! ## Server Usage
//!
//! ```rust,ignore
//! use joule_db_local::{Database, server::TcpServer};
//!
//! #[tokio::main]
//! async fn main() {
//!     let db = Database::open("./mydb").unwrap();
//!     let server = TcpServer::new(db);
//!     server.run("127.0.0.1:6380").await.unwrap();
//! }
//! ```
//!
//! ## Client Usage
//!
//! ```rust,ignore
//! use joule_db_local::server::TcpClient;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut client = TcpClient::connect("127.0.0.1:6380").await.unwrap();
//!     client.put(b"key", b"value").await.unwrap();
//!     let value = client.get(b"key").await.unwrap();
//!     client.close().await.unwrap();
//! }
//! ```

#[cfg(feature = "server")]
pub mod tcp;

#[cfg(feature = "server")]
pub mod client;

#[cfg(feature = "server")]
pub use tcp::{ServerStats, ServerStatsSnapshot, TcpServer, TcpServerConfig};

#[cfg(feature = "server")]
pub use client::{TcpClient, TcpClientConfig};
