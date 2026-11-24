//! Aggregate operations re-exported from expr module

pub use super::expr::{AggregateOp, ClinicalScale, SmoothnessMethod};
#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
