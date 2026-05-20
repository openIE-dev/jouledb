//! MXF Metadata Extractor — broadcast interchange format → amorphic fields.
//!
//! MXF (Material eXchange Format) is the standard container for broadcast content.
//! This module extracts metadata from MXF headers into JouleDB records,
//! enabling unified queries across file-based and streaming content.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Value;

/// Extracted MXF metadata (the fields broadcast systems care about).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MxfMetadata {
    /// UMID (Unique Material Identifier)
    pub umid: Option<String>,
    /// Content title
    pub title: Option<String>,
    /// Duration in frames
    pub duration_frames: Option<u64>,
    /// Frame rate (e.g., 29.97, 25.0, 23.976)
    pub frame_rate: Option<f64>,
    /// Video codec (e.g., "AVC-Intra 100", "XDCAM HD422", "ProRes 422 HQ")
    pub video_codec: Option<String>,
    /// Video resolution
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Audio channels
    pub audio_channels: Option<u8>,
    /// Audio sample rate
    pub audio_sample_rate: Option<u32>,
    /// Timecode start (SMPTE timecode string "HH:MM:SS:FF")
    pub timecode_start: Option<String>,
    /// Creation date (ISO 8601)
    pub creation_date: Option<String>,
    /// Descriptive metadata (DMS-1 fields)
    pub descriptive: HashMap<String, String>,
    /// Technical metadata (picture/sound essence descriptors)
    pub technical: HashMap<String, String>,
}

/// MXF metadata extractor.
pub struct MxfExtractor;

impl MxfExtractor {
    /// Extract metadata from MXF file header bytes.
    ///
    /// In production, this would parse the KLV (Key-Length-Value) structure
    /// of the MXF header partition. This implementation handles the common
    /// metadata fields from the header metadata set.
    pub fn from_header(header_bytes: &[u8]) -> MxfMetadata {
        // MXF files start with a partition pack key (16 bytes)
        // followed by KLV-encoded metadata sets.
        // This is a structural placeholder — full KLV parsing would
        // use a dedicated MXF library.
        MxfMetadata {
            umid: Self::find_umid(header_bytes),
            title: None,
            duration_frames: None,
            frame_rate: None,
            video_codec: None,
            width: None,
            height: None,
            audio_channels: None,
            audio_sample_rate: None,
            timecode_start: None,
            creation_date: None,
            descriptive: HashMap::new(),
            technical: HashMap::new(),
        }
    }

    /// Create metadata from known fields (for testing or manual ingest).
    pub fn from_fields(
        title: &str,
        duration_frames: u64,
        frame_rate: f64,
        codec: &str,
        width: u32,
        height: u32,
    ) -> MxfMetadata {
        MxfMetadata {
            umid: None,
            title: Some(title.to_string()),
            duration_frames: Some(duration_frames),
            frame_rate: Some(frame_rate),
            video_codec: Some(codec.to_string()),
            width: Some(width),
            height: Some(height),
            audio_channels: Some(2),
            audio_sample_rate: Some(48000),
            timecode_start: Some("00:00:00:00".to_string()),
            creation_date: None,
            descriptive: HashMap::new(),
            technical: HashMap::new(),
        }
    }

    /// Convert MXF metadata to amorphic field map for ingestion.
    pub fn to_amorphic_fields(meta: &MxfMetadata) -> HashMap<String, Value> {
        let mut fields = HashMap::new();

        if let Some(ref umid) = meta.umid {
            fields.insert("umid".into(), Value::String(umid.clone()));
        }
        if let Some(ref title) = meta.title {
            fields.insert("name".into(), Value::String(title.clone()));
            fields.insert("title".into(), Value::String(title.clone()));
        }
        if let Some(frames) = meta.duration_frames {
            fields.insert("duration_frames".into(), Value::Int(frames as i64));
            if let Some(fps) = meta.frame_rate {
                let duration_secs = frames as f64 / fps;
                fields.insert("duration_secs".into(), Value::Float(duration_secs));
            }
        }
        if let Some(fps) = meta.frame_rate {
            fields.insert("frame_rate".into(), Value::Float(fps));
        }
        if let Some(ref codec) = meta.video_codec {
            fields.insert("video_codec".into(), Value::String(codec.clone()));
        }
        if let Some(w) = meta.width {
            fields.insert("width".into(), Value::Int(w as i64));
        }
        if let Some(h) = meta.height {
            fields.insert("height".into(), Value::Int(h as i64));
        }
        if let Some(ch) = meta.audio_channels {
            fields.insert("audio_channels".into(), Value::Int(ch as i64));
        }
        if let Some(ref tc) = meta.timecode_start {
            fields.insert("timecode_start".into(), Value::String(tc.clone()));
        }
        if let Some(ref date) = meta.creation_date {
            fields.insert("creation_date".into(), Value::String(date.clone()));
        }

        // Add descriptive metadata
        for (k, v) in &meta.descriptive {
            fields.insert(format!("dms_{}", k), Value::String(v.clone()));
        }

        fields.insert("_source_format".into(), Value::String("MXF".into()));
        fields
    }

    /// Duration in seconds.
    pub fn duration_secs(meta: &MxfMetadata) -> Option<f64> {
        match (meta.duration_frames, meta.frame_rate) {
            (Some(frames), Some(fps)) if fps > 0.0 => Some(frames as f64 / fps),
            _ => None,
        }
    }

    /// Try to find UMID in raw header bytes (simplified).
    fn find_umid(bytes: &[u8]) -> Option<String> {
        // UMID is 32 bytes, typically found after partition pack
        if bytes.len() >= 64 {
            // Simplified: hash the first 64 bytes as a pseudo-UMID
            let hash = bytes[..64]
                .iter()
                .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
            Some(format!("urn:smpte:umid:{:016x}", hash))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mxf_to_amorphic() {
        let meta = MxfExtractor::from_fields(
            "Breaking News Segment",
            45000, // 30min at 25fps
            25.0,
            "AVC-Intra 100",
            1920,
            1080,
        );

        let fields = MxfExtractor::to_amorphic_fields(&meta);

        assert_eq!(fields.get("name"), Some(&Value::String("Breaking News Segment".into())));
        assert_eq!(fields.get("width"), Some(&Value::Int(1920)));
        assert_eq!(fields.get("video_codec"), Some(&Value::String("AVC-Intra 100".into())));
        assert_eq!(fields.get("_source_format"), Some(&Value::String("MXF".into())));

        // Duration should be 1800 seconds (30 min)
        if let Some(Value::Float(d)) = fields.get("duration_secs") {
            assert!((d - 1800.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_duration_calculation() {
        let meta = MxfExtractor::from_fields("Test", 75000, 29.97, "ProRes", 1920, 1080);
        let dur = MxfExtractor::duration_secs(&meta).unwrap();
        assert!((dur - 2502.5).abs() < 1.0); // ~41.7 minutes
    }
}
