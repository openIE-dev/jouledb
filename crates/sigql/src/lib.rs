//! # SigQL - Signal Query Language
//!
//! Query physical reality like a database.
//!
//! SigQL treats signals as first-class citizens with:
//! - **Frequency as a queryable dimension** (not just time)
//! - **Uncertainty propagation** through all operations
//! - **Causal semantics** for real-time safety
//! - **Multi-target compilation** (WebGPU, CUDA, SIMD)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use sigql::prelude::*;
//!
//! // Parse a SigQL query
//! let query = sigql::parse("
//!     FROM controller.imu.accel
//!     TRANSFORM bandpass(4Hz, 12Hz)
//!     WINDOW sliding(2s, 500ms)
//!     AGGREGATE { power: band_power(4Hz..12Hz) }
//!     RETURNING confidence(0.95)
//! ")?;
//!
//! // Compile for the current platform
//! let plan = sigql::compile(&query, Target::Simd)?;
//!
//! // Execute with a runtime
//! let mut runtime = Runtime::new(RuntimeConfig::default());
//! runtime.register_signal("controller.imu.accel", my_signal);
//! let result = runtime.execute(&plan)?;
//!
//! // Results include uncertainty bounds
//! println!("Power: {} ± {}", result.value, result.ci_width());
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//! │   Parser    │───▶│ Type Check  │───▶│  Optimizer  │───▶│   Codegen   │
//! │  (nom)      │    │  (signals)  │    │  (DSP)      │    │ (multi-tgt) │
//! └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
//!        │                                                        │
//!        ▼                                                        ▼
//!   SigQL Query                                            Execution Plan
//!   "FROM sensor                                           ┌─────────────┐
//!    TRANSFORM fft()"                                      │ LoadSignal  │
//!                                                          │ FFT         │
//!                                                          │ Store       │
//!                                                          └─────────────┘
//! ```
//!
//! ## Design Principles
//!
//! 1. **Signals are first-class** - Not columns of floats
//! 2. **Frequency is queryable** - Like time in TimescaleDB
//! 3. **Causality is enforced** - Windows can't peek into the future
//! 4. **Uncertainty is typed** - Every result carries confidence bounds
//! 5. **Compiles everywhere** - Same query → WebGPU / CUDA / SIMD

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod ast;
pub mod compile;
pub mod dsp;
pub mod parser;
pub mod runtime;
pub mod types;

#[cfg(feature = "std")]
pub mod io;

// Re-exports for convenience
pub use types::{
    ChannelSpec, Decibels, DynSignal, FrequencyBand, Hertz, Interval, Phase, PowerSpectralDensity,
    QualityFlags, SampleRate, Seconds, Signal, SignalMetadata, Spectrogram, Spectrum,
    UncertainResult, UncertainValue, WindowFunction,
};

pub use ast::{AggregateOp, Query, SignalExpr, TransformOp, WindowSpec};
pub use compile::{CompileError, Compiler, CompilerConfig, ExecutionPlan, Target};
pub use parser::{ParseError, parse_query, parse_signal_expr};
pub use runtime::{ExecutionResult, Runtime, RuntimeConfig, RuntimeError};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::ast::{Query, SignalExpr};
    pub use crate::compile::{Compiler, Target};
    pub use crate::parser::parse_query;
    pub use crate::runtime::{Runtime, RuntimeConfig};
    pub use crate::types::*;
}

/// Parse a SigQL query string into an AST
///
/// # Example
///
/// ```rust,ignore
/// let query = sigql::parse("FROM sensor.data TRANSFORM fft()")?;
/// ```
pub fn parse(query: &str) -> Result<Query, ParseError> {
    parse_query(query)
}

/// Compile a query for a specific target
///
/// # Example
///
/// ```rust,ignore
/// let plan = sigql::compile(&query, Target::WebGpu)?;
/// ```
pub fn compile(query: &Query, target: Target) -> Result<ExecutionPlan, CompileError> {
    let config = CompilerConfig {
        target,
        optimize: true,
        debug_info: false,
        default_sample_rate: 1000,
        default_fft_size: 1024,
    };
    let mut compiler = Compiler::new(config);
    compiler.compile(query)
}

/// SigQL version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple() {
        let result = parse("FROM sensor.data");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_with_transform() {
        let result = parse("FROM sensor.data TRANSFORM bandpass(4Hz, 12Hz)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_uncertain_value() {
        let uv = UncertainValue::from_ci(10.0, 1.0, 0.95, 100);
        assert!((uv.value - 10.0).abs() < 0.001);
        assert!(uv.is_significant());
    }

    #[test]
    fn test_frequency_band() {
        let band = FrequencyBand::parkinsonian_tremor();
        assert!(band.contains(Hertz::new(6.0)));
        assert!(!band.contains(Hertz::new(15.0)));
    }
}
