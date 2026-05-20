//! Analytical Functions for OLAP Queries
//!
//! Provides window functions, OLAP functions, and statistical functions
//! for analytical queries.

pub mod window;
pub use window::{
    FrameBound, WindowExecutor, WindowFrame, WindowFunction, WindowFunctionType, WindowSpec,
};
