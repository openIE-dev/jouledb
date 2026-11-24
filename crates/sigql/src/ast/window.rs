//! Window operations re-exported from expr module

pub use super::expr::{Causality, EventRef, FrequencyScale, WindowKind, WindowSpec};
#[cfg(not(feature = "std"))]
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
