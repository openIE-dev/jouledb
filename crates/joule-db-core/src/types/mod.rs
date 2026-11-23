//! Type system for JouleDB
//!
//! Defines the core value types and serialization.

pub mod spatial;
mod value;

pub use spatial::{Bbox3, Point3, Pose6, Quat, Spatial3dKind, Spatial3dValue};
pub use value::Value;
