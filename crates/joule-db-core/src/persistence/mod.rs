//! Unified persistence infrastructure for JouleDB
//!
//! This module provides platform-agnostic abstractions for:
//!
//! - **Write-Ahead Logging (WAL)**: Durability and crash recovery
//! - **Snapshots**: Point-in-time backups
//! - **Network Protocol**: Binary protocol for client-server communication
//! - **Compute**: GPU/CPU/SIMD acceleration
//!
//! ## Cross-Platform Design
//!
//! JouleDB is designed to run on:
//! - **Native**: Desktop (Windows, macOS, Linux) and Server
//! - **Browser**: WebAssembly with IndexedDB/OPFS storage
//! - **Embedded**: MCUs with flash storage
//!
//! Each platform implements the traits defined here with platform-specific
//! backends while the core database logic remains identical.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                    JouleDB Application                          │
//! ├──────────────────────────────────────────────────────────────────┤
//! │                     joule-db-core                             │
//! │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
//! │  │  WAL Trait │  │  Snapshot  │  │  Network   │  │  Compute   │ │
//! │  │            │  │   Trait    │  │   Trait    │  │   Trait    │ │
//! │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
//! └────────┼───────────────┼───────────────┼───────────────┼────────┘
//!          │               │               │               │
//! ┌────────┼───────────────┼───────────────┼───────────────┼────────┐
//! │        │               │               │               │        │
//! │   ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐  │
//! │   │  File   │    │   File    │   │    TCP    │   │   wgpu    │  │
//! │   │  WAL    │    │ Snapshot  │   │  Server   │   │   GPU     │  │
//! │   └─────────┘    └───────────┘   └───────────┘   └───────────┘  │
//! │                    joule-db-local (Native)                     │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │   ┌─────────┐    ┌───────────┐   ┌───────────┐   ┌───────────┐  │
//! │   │IndexedDB│    │   OPFS    │   │ WebSocket │   │  WebGPU   │  │
//! │   │   WAL   │    │ Snapshot  │   │  Client   │   │    GPU    │  │
//! │   └─────────┘    └───────────┘   └───────────┘   └───────────┘  │
//! │                  joule-db-browser (WASM)                       │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │   ┌─────────┐    ┌───────────┐   ┌───────────┐   ┌───────────┐  │
//! │   │  Flash  │    │   EEPROM  │   │   UART    │   │   SIMD    │  │
//! │   │   WAL   │    │ Snapshot  │   │ Protocol  │   │    CPU    │  │
//! │   └─────────┘    └───────────┘   └───────────┘   └───────────┘  │
//! │                 joule-db-embedded (MCU)                       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_core::persistence::{
//!     WalBackend, PersistenceManager, PersistenceConfig,
//!     DurabilityPolicy, NetworkBackend, ComputeBackend,
//! };
//!
//! // Configure persistence
//! let config = PersistenceConfig {
//!     durability: DurabilityPolicy::SyncOnCommit,
//!     max_wal_size: 64 * 1024 * 1024,
//!     ..Default::default()
//! };
//!
//! // Platform-specific backends are created in joule-db-local/browser/embedded
//! ```

pub mod compute;
#[cfg(feature = "group-commit")]
pub mod group_commit;
pub mod network;
pub mod traits;

// Re-export main types
pub use traits::{
    // Configuration
    DurabilityPolicy,
    // WAL types
    LSN,
    PersistenceConfig,
    // Manager
    PersistenceManager,
    // Recovery types
    RecoveryResult,
    RecoveryStrategy,
    SnapshotBackend,
    // Snapshot types
    SnapshotMetadata,
    TxId,
    WalBackend,
    WalEntry,
    WalEntryType,
    // Utilities
    crc32,
};

pub use network::{
    // Connection types
    ConnectionState,
    ConnectionStats,
    ErrorCode,
    HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    Message,
    // Message types
    MessageFlags,
    // Backend traits
    NetworkBackend,
    NetworkListener,
    OpCode,
    // Protocol constants
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
    decode_key_value,
    // Helpers
    encode_key_value,
    encode_key_values,
    encode_keys,
};

#[cfg(feature = "async")]
pub use network::AsyncNetworkBackend;

pub use compute::{
    AggregationType,
    BindGroupHandle,
    BufferHandle,
    // Buffer types
    BufferUsage,
    // Backend trait
    ComputeBackend,
    // Context
    ComputeContext,
    // Operation types
    ComputeOp,
    ComputeResult,
    // CPU implementation
    CpuComputeBackend,
    DeviceCapabilities,
    // Device types
    DeviceType,
    HashAlgorithm,
    PipelineHandle,
};
