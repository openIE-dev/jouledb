//! Type system for JouleDB
//!
//! Defines the core value types and serialization.

mod value;
pub mod spatial;

pub use value::Value;
pub use spatial::{Bbox3, Point3, Pose6, Quat, Spatial3dKind, Spatial3dValue};
