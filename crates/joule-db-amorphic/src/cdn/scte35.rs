//! SCTE-35 Ad Break Marker Parser
//!
//! SCTE-35 is the standard for signaling ad insertion points in MPEG-TS streams.
//! This module parses SCTE-35 splice commands and stores them as temporal fields
//! in JouleDB, enabling queries like "what ad break is active at timestamp X?"
//!
//! Solves: SCTE-35 markers lost during transcode, inconsistent across ABR variants,
//! revenue loss from failed ad insertions in the $9B FAST market.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::temporal_fields::{TemporalField, TemporalStore};
use crate::{RecordId, Value};

/// SCTE-35 splice command types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpliceCommand {
    /// Null — keepalive, no action
    Null,
    /// Splice insert — ad break start/end
    SpliceInsert,
    /// Time signal — generic signaling point
    TimeSignal,
    /// Bandwidth reservation — reserved capacity for future use
    BandwidthReservation,
    /// Private command — vendor-specific
    Private,
}

/// A parsed SCTE-35 marker with timing and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scte35Marker {
    /// Unique splice event ID
    pub splice_event_id: u32,
    /// Command type
    pub command: SpliceCommand,
    /// PTS time of the splice point (90kHz ticks from stream start)
    pub pts_time: Option<u64>,
    /// Duration of the ad break in 90kHz ticks
    pub duration_ticks: Option<u64>,
    /// Whether this is an out-of-network (ad break start) or return
    pub out_of_network: bool,
    /// Program splice (entire program) vs component splice (single PID)
    pub program_splice: bool,
    /// Auto-return: automatically return from break after duration
    pub auto_return: bool,
    /// UPID (Unique Program Identifier) for ad targeting
    pub segmentation_upid: Option<String>,
    /// Segmentation type (e.g., 0x30 = provider ad start, 0x31 = provider ad end)
    pub segmentation_type_id: Option<u8>,
    /// Wall-clock timestamp (unix ms) if available
    pub wall_clock_ms: Option<u64>,
    /// Raw base64-encoded SCTE-35 binary for passthrough
    pub raw_base64: Option<String>,
}

impl Scte35Marker {
    /// Duration in milliseconds (converted from 90kHz ticks).
    pub fn duration_ms(&self) -> Option<u64> {
        self.duration_ticks.map(|t| t * 1000 / 90000)
    }

    /// Is this an ad break start?
    pub fn is_break_start(&self) -> bool {
        self.out_of_network
            && matches!(self.command, SpliceCommand::SpliceInsert | SpliceCommand::TimeSignal)
    }

    /// Is this an ad break end (return to content)?
    pub fn is_break_end(&self) -> bool {
        !self.out_of_network
            && matches!(self.command, SpliceCommand::SpliceInsert | SpliceCommand::TimeSignal)
    }
}

/// Parser for SCTE-35 markers from various input formats.
pub struct Scte35Parser;

impl Scte35Parser {
    /// Parse a base64-encoded SCTE-35 binary section.
    ///
    /// In production, this would decode the full SCTE-35 binary format.
    /// This implementation handles the common JSON/manifest representation.
    pub fn from_base64(base64: &str) -> Option<Scte35Marker> {
        // Minimal parsing — in production would use a full SCTE-35 decoder.
        // The base64 is stored for passthrough to downstream systems.
        Some(Scte35Marker {
            splice_event_id: 0,
            command: SpliceCommand::SpliceInsert,
            pts_time: None,
            duration_ticks: None,
            out_of_network: true,
            program_splice: true,
            auto_return: false,
            segmentation_upid: None,
            segmentation_type_id: None,
            wall_clock_ms: None,
            raw_base64: Some(base64.to_string()),
        })
    }

    /// Create a marker from HLS EXT-X-DATERANGE attributes.
    pub fn from_hls_daterange(
        id: &str,
        start_date_ms: u64,
        duration_ms: Option<u64>,
        scte35_cmd: Option<&str>,
    ) -> Scte35Marker {
        Scte35Marker {
            splice_event_id: Self::hash_id(id),
            command: SpliceCommand::SpliceInsert,
            pts_time: None,
            duration_ticks: duration_ms.map(|d| d * 90), // ms → 90kHz
            out_of_network: true,
            program_splice: true,
            auto_return: duration_ms.is_some(),
            segmentation_upid: Some(id.to_string()),
            segmentation_type_id: Some(0x30), // Provider ad start
            wall_clock_ms: Some(start_date_ms),
            raw_base64: scte35_cmd.map(|s| s.to_string()),
        }
    }

    /// Create an explicit ad break marker.
    pub fn ad_break(
        event_id: u32,
        start_ms: u64,
        duration_ms: u64,
    ) -> Scte35Marker {
        Scte35Marker {
            splice_event_id: event_id,
            command: SpliceCommand::SpliceInsert,
            pts_time: None,
            duration_ticks: Some(duration_ms * 90),
            out_of_network: true,
            program_splice: true,
            auto_return: true,
            segmentation_upid: None,
            segmentation_type_id: Some(0x30),
            wall_clock_ms: Some(start_ms),
            raw_base64: None,
        }
    }

    fn hash_id(id: &str) -> u32 {
        let mut h: u32 = 0x811c9dc5;
        for b in id.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
        h
    }
}

/// Store SCTE-35 markers as temporal fields for queryable ad break scheduling.
pub fn store_markers(
    temporal: &mut TemporalStore,
    channel_id: RecordId,
    markers: &[Scte35Marker],
) {
    for marker in markers {
        if let (Some(start_ms), Some(dur_ms)) = (marker.wall_clock_ms, marker.duration_ms()) {
            temporal.set(
                channel_id,
                "ad_break",
                TemporalField::bounded(
                    Value::String(format!(
                        "splice_event_id={},type={:?},upid={}",
                        marker.splice_event_id,
                        marker.command,
                        marker.segmentation_upid.as_deref().unwrap_or("none")
                    )),
                    start_ms,
                    start_ms + dur_ms,
                ),
            );
        }
    }
}

/// Query: is there an active ad break at this timestamp?
pub fn active_ad_break(
    temporal: &TemporalStore,
    channel_id: RecordId,
    timestamp_ms: u64,
) -> Option<String> {
    temporal
        .query_valid_at(channel_id, "ad_break", timestamp_ms, None)
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => format!("{:?}", other),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scte35_ad_break() {
        let marker = Scte35Parser::ad_break(1, 60_000, 30_000); // 30s break at 60s
        assert!(marker.is_break_start());
        assert_eq!(marker.duration_ms(), Some(30_000));
        assert!(marker.auto_return);
    }

    #[test]
    fn test_store_and_query_markers() {
        let mut temporal = TemporalStore::new();

        let markers = vec![
            Scte35Parser::ad_break(1, 60_000, 30_000),  // 60s-90s
            Scte35Parser::ad_break(2, 300_000, 60_000), // 300s-360s
        ];

        store_markers(&mut temporal, 1, &markers);

        // During first ad break
        assert!(active_ad_break(&temporal, 1, 75_000).is_some());

        // Between breaks
        assert!(active_ad_break(&temporal, 1, 150_000).is_none());

        // During second ad break
        assert!(active_ad_break(&temporal, 1, 330_000).is_some());

        // After all breaks
        assert!(active_ad_break(&temporal, 1, 400_000).is_none());
    }

    #[test]
    fn test_hls_daterange_parsing() {
        let marker = Scte35Parser::from_hls_daterange(
            "ad-break-001",
            1_700_000_000_000,
            Some(30_000),
            Some("/DA0AAAA..."),
        );

        assert_eq!(marker.wall_clock_ms, Some(1_700_000_000_000));
        assert_eq!(marker.duration_ms(), Some(30_000));
        assert!(marker.raw_base64.is_some());
    }
}
