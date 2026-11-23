//! Inter-Process Communication for Browser WASM
//!
//! Provides high-performance data transfer mechanisms between the main thread
//! and Web Workers using SharedArrayBuffer and Atomics.

mod shared_array_buffer;
mod zero_copy;

pub use shared_array_buffer::{SharedRingBuffer, SharedRingBufferWorker, get_worker_code};
pub use zero_copy::{ArrowIpcBridge, WorkerBridge};
