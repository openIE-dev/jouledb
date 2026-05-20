//! # JouleDB GPU
//!
//! GPU compute backend for JouleDB using wgpu (WebGPU).
//!
//! This crate provides high-performance GPU acceleration for database operations:
//! - Vector similarity search
//! - Batch key lookups with parallel hash computation
//! - Index building
//! - Aggregations (COUNT, SUM, AVG, MIN, MAX)
//! - Sorting and filtering
//!
//! ## Features
//!
//! - Cross-platform: Works on Windows, macOS, Linux, and Web (WebGPU)
//! - Uses wgpu for hardware abstraction
//! - Falls back to CPU compute when GPU is unavailable
//!
//! ## Usage
//!
//! ```rust,ignore
//! use joule_db_gpu::WgpuComputeBackend;
//! use joule_db_core::persistence::compute::{ComputeBackend, ComputeOp, BufferUsage};
//!
//! // Create GPU backend
//! let mut backend = pollster::block_on(WgpuComputeBackend::new()).unwrap();
//!
//! // Create buffers
//! let input = backend.create_buffer(1024, BufferUsage::STORAGE_READ, None).unwrap();
//! backend.write_buffer(input, 0, &data).unwrap();
//!
//! // Execute compute operation
//! let result = backend.execute(
//!     ComputeOp::Aggregate {
//!         agg_type: AggregationType::Sum,
//!         num_items: 256,
//!     },
//!     &[input],
//! ).unwrap();
//! ```

pub mod backend;
pub mod btree_serialization;
pub mod hdc_compute;
pub mod shaders;

pub use backend::{WgpuComputeBackend, WgpuComputeConfig};
pub use btree_serialization::{
    GPU_MAX_KEY_SIZE, GPU_MAX_KEYS, GPU_MAX_VALUE_SIZE, GpuBTreeNodeDeserializer,
    GpuBTreeNodeSerializer, GpuSerializedNode,
};
pub use hdc_compute::{HdcGpuConfig, HdcGpuEncoder};

// Re-export core types for convenience
pub use joule_db_core::persistence::compute::{
    AggregationType, BufferHandle, BufferUsage, ComputeBackend, ComputeOp, ComputeResult,
    DeviceCapabilities, DeviceType, HashAlgorithm,
};
