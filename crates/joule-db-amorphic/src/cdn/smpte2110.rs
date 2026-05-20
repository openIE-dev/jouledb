//! SMPTE ST 2110 Essence Metadata Bridge
//!
//! ST 2110 is the standard for uncompressed media over IP in broadcast.
//! This module captures essence descriptor metadata from ST 2110 streams
//! and stores it as amorphic records for unified querying.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Value;

/// SMPTE ST 2110 stream type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum St2110Type {
    /// ST 2110-20: Uncompressed video
    Video,
    /// ST 2110-30: PCM audio
    Audio,
    /// ST 2110-40: Ancillary data (captions, timecode)
    Ancillary,
}

/// Metadata from an ST 2110 essence stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EssenceMetadata {
    /// Stream identifier (multicast address:port)
    pub stream_id: String,
    /// Stream type
    pub stream_type: St2110Type,
    /// Multicast source address
    pub source_address: String,
    /// Multicast destination address
    pub dest_address: String,
    /// RTP payload type
    pub payload_type: u8,
    /// Sampling rate (video: frame rate × 1000, audio: sample rate)
    pub sampling_rate: u32,
    /// Video-specific fields
    pub video: Option<VideoEssence>,
    /// Audio-specific fields
    pub audio: Option<AudioEssence>,
    /// PTP clock domain
    pub ptp_domain: Option<u8>,
    /// SDP (Session Description Protocol) attributes
    pub sdp_attributes: HashMap<String, String>,
}

/// Video essence parameters (ST 2110-20).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEssence {
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub color_depth: u8,       // 8, 10, 12
    pub color_sampling: String, // "YCbCr-4:2:2", "RGB", etc.
    pub interlaced: bool,
    /// Bandwidth in Mbps (uncompressed video is huge)
    pub bandwidth_mbps: f64,
}

/// Audio essence parameters (ST 2110-30).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEssence {
    pub channels: u8,
    pub sample_rate: u32,
    pub bit_depth: u8,
    pub channel_order: String, // AES-10 / SMPTE ST 2110-30
}

/// An ST 2110 stream descriptor for the amorphic store.
#[derive(Debug, Clone)]
pub struct St2110Stream {
    pub metadata: EssenceMetadata,
}

impl St2110Stream {
    /// Create from SDP (Session Description Protocol) attributes.
    pub fn from_sdp(stream_id: &str, sdp: &HashMap<String, String>) -> Self {
        let stream_type = if sdp.get("media").map(|m| m.starts_with("video")).unwrap_or(false) {
            St2110Type::Video
        } else if sdp.get("media").map(|m| m.starts_with("audio")).unwrap_or(false) {
            St2110Type::Audio
        } else {
            St2110Type::Ancillary
        };

        Self {
            metadata: EssenceMetadata {
                stream_id: stream_id.to_string(),
                stream_type,
                source_address: sdp.get("source").cloned().unwrap_or_default(),
                dest_address: sdp.get("connection").cloned().unwrap_or_default(),
                payload_type: sdp.get("rtpmap").and_then(|r| r.split_whitespace().next()?.parse().ok()).unwrap_or(96),
                sampling_rate: 0,
                video: None,
                audio: None,
                ptp_domain: sdp.get("ptp-domain").and_then(|d| d.parse().ok()),
                sdp_attributes: sdp.clone(),
            },
        }
    }

    /// Create a video stream descriptor.
    pub fn video(
        stream_id: &str,
        width: u32,
        height: u32,
        frame_rate: f64,
        color_depth: u8,
    ) -> Self {
        let bandwidth = width as f64 * height as f64 * frame_rate * color_depth as f64 * 2.0 / 1_000_000.0; // rough Mbps
        Self {
            metadata: EssenceMetadata {
                stream_id: stream_id.to_string(),
                stream_type: St2110Type::Video,
                source_address: String::new(),
                dest_address: String::new(),
                payload_type: 96,
                sampling_rate: (frame_rate * 1000.0) as u32,
                video: Some(VideoEssence {
                    width,
                    height,
                    frame_rate,
                    color_depth,
                    color_sampling: "YCbCr-4:2:2".into(),
                    interlaced: false,
                    bandwidth_mbps: bandwidth,
                }),
                audio: None,
                ptp_domain: Some(0),
                sdp_attributes: HashMap::new(),
            },
        }
    }

    /// Create an audio stream descriptor.
    pub fn audio(stream_id: &str, channels: u8, sample_rate: u32, bit_depth: u8) -> Self {
        Self {
            metadata: EssenceMetadata {
                stream_id: stream_id.to_string(),
                stream_type: St2110Type::Audio,
                source_address: String::new(),
                dest_address: String::new(),
                payload_type: 97,
                sampling_rate: sample_rate,
                video: None,
                audio: Some(AudioEssence {
                    channels,
                    sample_rate,
                    bit_depth,
                    channel_order: "SMPTE".into(),
                }),
                ptp_domain: Some(0),
                sdp_attributes: HashMap::new(),
            },
        }
    }

    /// Convert to amorphic fields for storage.
    pub fn to_amorphic_fields(&self) -> HashMap<String, Value> {
        let m = &self.metadata;
        let mut fields = HashMap::new();

        fields.insert("name".into(), Value::String(m.stream_id.clone()));
        fields.insert("stream_type".into(), Value::String(format!("{:?}", m.stream_type)));
        fields.insert("source_address".into(), Value::String(m.source_address.clone()));
        fields.insert("dest_address".into(), Value::String(m.dest_address.clone()));
        fields.insert("payload_type".into(), Value::Int(m.payload_type as i64));
        fields.insert("_source_format".into(), Value::String("ST2110".into()));

        if let Some(ref v) = m.video {
            fields.insert("width".into(), Value::Int(v.width as i64));
            fields.insert("height".into(), Value::Int(v.height as i64));
            fields.insert("frame_rate".into(), Value::Float(v.frame_rate));
            fields.insert("color_depth".into(), Value::Int(v.color_depth as i64));
            fields.insert("bandwidth_mbps".into(), Value::Float(v.bandwidth_mbps));
            fields.insert("interlaced".into(), Value::Bool(v.interlaced));
        }

        if let Some(ref a) = m.audio {
            fields.insert("audio_channels".into(), Value::Int(a.channels as i64));
            fields.insert("audio_sample_rate".into(), Value::Int(a.sample_rate as i64));
            fields.insert("audio_bit_depth".into(), Value::Int(a.bit_depth as i64));
        }

        fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_stream() {
        let stream = St2110Stream::video("cam1-video", 1920, 1080, 59.94, 10);
        let fields = stream.to_amorphic_fields();

        assert_eq!(fields.get("width"), Some(&Value::Int(1920)));
        assert_eq!(fields.get("height"), Some(&Value::Int(1080)));
        assert_eq!(fields.get("_source_format"), Some(&Value::String("ST2110".into())));

        if let Some(Value::Float(bw)) = fields.get("bandwidth_mbps") {
            assert!(*bw > 1000.0); // Uncompressed 1080p60 is ~2.4 Gbps
        }
    }

    #[test]
    fn test_audio_stream() {
        let stream = St2110Stream::audio("cam1-audio", 8, 48000, 24);
        let fields = stream.to_amorphic_fields();

        assert_eq!(fields.get("audio_channels"), Some(&Value::Int(8)));
        assert_eq!(fields.get("audio_sample_rate"), Some(&Value::Int(48000)));
        assert_eq!(fields.get("stream_type"), Some(&Value::String("Audio".into())));
    }
}
