//! Aggregate operations re-exported from expr module


#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, format, string::{String, ToString}, vec, vec::Vec};
pub use super::expr::{AggregateOp, ClinicalScale, SmoothnessMethod};
