//! JouleDB Edge - Database engine for edge devices with hardware acceleration
//!
//! This crate provides a specialized database implementation for edge devices
//! that have hardware accelerators like GPUs, NPUs, or TPUs. It is designed
//! for devices like:
//!
//! - **NVIDIA Jetson** (Nano, Xavier, Orin) - CUDA/TensorRT
//! - **Google Coral** (Dev Board, USB Accelerator) - Edge TPU
//! - **Raspberry Pi 5** - VideoCore VII
//! - **Rockchip RK3588** boards - RKNN NPU
//! - **Intel NCS2** - Movidius VPU
//! - **Hailo-8** accelerator boards
//!
//! ## Design Philosophy
//!
//! JouleDB Edge is NOT for bare-metal MCUs without accelerators.
//! We specifically target devices with:
//!
//! 1. **Hardware acceleration** (GPU/NPU/TPU)
//! 2. **Sufficient memory** (256MB+ RAM)
//! 3. **OS or RTOS** with memory management
//!
//! ## Features
//!
//! - Zero-copy operations where possible
//! - DMA-friendly memory layouts
//! - Accelerator-offloaded operations (hash, aggregation)
//! - Power-efficient batch processing
//! - Crash-safe persistence
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_edge::{EdgeDatabase, EdgeConfig, AcceleratorType};
//!
//! // Auto-detect available accelerator
//! let config = EdgeConfig::auto_detect();
//!
//! let db = EdgeDatabase::open("./edge.db", config)?;
//! db.put(b"sensor:001", b"temperature:23.5")?;
//! let value = db.get(b"sensor:001")?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

pub mod accelerator;
pub mod backends;
pub mod config;
pub mod database;
pub mod memory;
pub mod storage;
#[cfg(feature = "std")]
pub mod sync;

pub use accelerator::{Accelerator, AcceleratorCapabilities, AcceleratorType};
pub use config::EdgeConfig;
pub use database::EdgeDatabase;
pub use memory::{AlignedBuffer, DmaBuffer};
#[cfg(feature = "std")]
pub use sync::{SyncEngine, SyncRequest, SyncResponse, SyncState, VectorClock};
