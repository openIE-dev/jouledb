//! Metal GPU Runtime for Joule
//!
//! This crate provides runtime support for executing Metal compute kernels
//! on Apple Silicon and macOS systems. It wraps the Metal API with a safe,
//! ergonomic Rust interface.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Joule Metal Runtime                       │
//! ├─────────────────────────────────────────────────────────────┤
//! │  MetalDevice      - GPU device enumeration and selection    │
//! │  MetalBuffer      - GPU buffer allocation and management    │
//! │  MetalCommandQueue - Command buffer submission              │
//! │  MetalComputePipeline - Compute kernel execution            │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Metal Framework (macOS/iOS)                     │
//! │  MTLDevice, MTLBuffer, MTLCommandQueue, MTLComputePipeline  │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! use joule_metal_rt::{MetalDevice, MetalBuffer, MetalComputePipeline};
//!
//! // Get the default Metal device
//! let device = MetalDevice::default()?;
//!
//! // Create buffers
//! let input = device.create_buffer::<f32>(1024)?;
//! let output = device.create_buffer::<f32>(1024)?;
//!
//! // Create compute pipeline from compiled shader
//! let pipeline = device.create_compute_pipeline(shader_bytecode)?;
//!
//! // Launch kernel
//! let queue = device.create_command_queue()?;
//! queue.launch_kernel(&pipeline, (1024, 1, 1), (32, 1, 1), &[&input, &output])?;
//! queue.wait_until_completed()?;
//! ```
//!
//! ## Platform Support
//!
//! This crate is designed for macOS and iOS. On other platforms, all types
//! and functions are available but return errors when used.

#![cfg_attr(not(target_os = "macos"), allow(unused_variables, dead_code))]

mod buffer;
mod command;
mod device;
mod error;
mod pipeline;
mod sync;

pub use buffer::{BufferUsage, MetalBuffer};
pub use command::{MetalCommandBuffer, MetalCommandQueue};
pub use device::{DeviceInfo, MetalDevice};
pub use error::{MetalError, MetalResult};
pub use pipeline::{MetalComputePipeline, MetalLibrary, ThreadgroupSize};
pub use sync::{MetalEvent, MetalFence, MetalSharedEvent};

/// Grid size for kernel dispatch (width, height, depth)
pub type GridSize = (u32, u32, u32);

/// Block/threadgroup size for kernel dispatch (width, height, depth)
pub type BlockSize = (u32, u32, u32);

/// Re-export common types for convenience
pub mod prelude {
    pub use super::{
        BlockSize, BufferUsage, GridSize, MetalBuffer, MetalCommandBuffer, MetalCommandQueue,
        MetalComputePipeline, MetalDevice, MetalError, MetalEvent, MetalFence, MetalLibrary,
        MetalResult, MetalSharedEvent, ThreadgroupSize,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_size_type() {
        let grid: GridSize = (1024, 1, 1);
        assert_eq!(grid.0, 1024);
        assert_eq!(grid.1, 1);
        assert_eq!(grid.2, 1);
    }

    #[test]
    fn test_block_size_type() {
        let block: BlockSize = (32, 8, 1);
        assert_eq!(block.0, 32);
        assert_eq!(block.1, 8);
        assert_eq!(block.2, 1);
    }
}
