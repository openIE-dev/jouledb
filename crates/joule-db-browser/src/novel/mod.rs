//! Browser Integration for Novel Database Features
//!
//! This module provides WASM bindings and browser-specific optimizations
//! for joule-db-novel features, while keeping the core novel crate pure Rust.
//!
//! Architecture:
//! - Core algorithms remain in joule-db-novel (pure Rust, no WASM)
//! - Browser-specific wrappers here use wasm-bindgen
//! - GPU acceleration via joule-db-gpu
//! - OPFS persistence for holographic stores

pub mod holographic;
pub mod holographic_serialization;
pub mod hyperdimensional;
pub mod predictive;
pub mod query_history_storage;

// Re-export browser-specific types
pub use holographic::{BrowserHolographicStore, HolographicStoreConfig};
pub use hyperdimensional::{BrowserHyperdimensionalSearch, GpuHyperdimensionalOps};
pub use predictive::{BrowserQueryPredictor, PredictivePrefetch};
