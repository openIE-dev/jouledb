//! SigQL Type System
//!
//! Signals are first-class citizens with compile-time dimensional analysis.
//! Every computation propagates uncertainty through the type system.

pub mod signal;
pub mod spectrum;
pub mod uncertainty;
pub mod units;

pub use signal::*;
pub use spectrum::*;
pub use uncertainty::*;
pub use units::*;
