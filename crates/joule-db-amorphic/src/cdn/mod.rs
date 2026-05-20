//! JouleDB CDN — broadcast and streaming protocol adapters.
//!
//! Bridges the gap between broadcast industry standards and JouleDB's amorphic store:
//! - SCTE-35 ad break markers → temporal fields
//! - HLS/DASH manifest manipulation → edge ad insertion
//! - MXF metadata → amorphic record fields
//! - FAST channel scheduling → trending + ad targeting
//! - Platform API sync → distribution manifests
//! - SMPTE ST 2110 → essence metadata

pub mod scte35;
pub mod manifest;
pub mod mxf;
pub mod fast;
pub mod platform;
pub mod smpte2110;

pub use scte35::{Scte35Marker, Scte35Parser, SpliceCommand};
pub use manifest::{ManifestRewriter, AdBreak, ManifestFormat};
pub use mxf::{MxfMetadata, MxfExtractor};
pub use fast::{FastScheduler, ScheduleSlot, SlotType};
pub use platform::{PlatformConnector, PlatformType, SyncStatus};
pub use smpte2110::{EssenceMetadata, St2110Stream};
