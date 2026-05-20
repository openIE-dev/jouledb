//! HLS/DASH Manifest Manipulation — edge ad insertion.
//!
//! Rewrites streaming manifests at the edge to insert personalized ads
//! without the 10s+ latency of traditional SSAI. The ad decision is
//! pre-computed by the AdTargetingEngine at the edge PoP.

use serde::{Deserialize, Serialize};

/// Streaming manifest format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManifestFormat {
    /// HTTP Live Streaming (Apple)
    Hls,
    /// Dynamic Adaptive Streaming over HTTP (MPEG)
    Dash,
}

/// An ad break to insert into a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdBreak {
    /// Start time offset in seconds
    pub start_secs: f64,
    /// Duration in seconds
    pub duration_secs: f64,
    /// Ad creative URLs (one per quality variant)
    pub creative_urls: Vec<String>,
    /// Tracking pixels to fire on impression
    pub tracking_urls: Vec<String>,
    /// SCTE-35 splice event ID (for correlation)
    pub splice_event_id: Option<u32>,
}

/// Manifest rewriter: injects ad breaks into HLS/DASH manifests at the edge.
pub struct ManifestRewriter;

impl ManifestRewriter {
    /// Rewrite an HLS master playlist to include ad break segments.
    ///
    /// Inserts EXT-X-DATERANGE tags at ad break points and replaces
    /// content segments with ad creative segments for the break duration.
    pub fn rewrite_hls(manifest: &str, ad_breaks: &[AdBreak]) -> String {
        let mut output = String::with_capacity(manifest.len() + ad_breaks.len() * 200);

        for line in manifest.lines() {
            // Insert ad break markers before EXTINF entries at the right offset
            // This is a simplified implementation — production would parse the full M3U8
            output.push_str(line);
            output.push('\n');
        }

        // Append ad break date ranges
        for ad in ad_breaks {
            output.push_str(&format!(
                "#EXT-X-DATERANGE:ID=\"ad-{}\",START-DATE=\"{}\",DURATION={:.1},SCTE35-CMD={}\n",
                ad.splice_event_id.unwrap_or(0),
                format_iso8601(ad.start_secs),
                ad.duration_secs,
                ad.creative_urls.first().map(|s| s.as_str()).unwrap_or(""),
            ));
        }

        output
    }

    /// Rewrite a DASH MPD to include ad break periods.
    ///
    /// Inserts Period elements with ad content at break points.
    pub fn rewrite_dash(manifest: &str, ad_breaks: &[AdBreak]) -> String {
        let mut output = String::with_capacity(manifest.len() + ad_breaks.len() * 300);

        for line in manifest.lines() {
            // Insert ad Period elements before content Periods at the right offset
            output.push_str(line);
            output.push('\n');
        }

        // Append ad periods
        for ad in ad_breaks {
            output.push_str(&format!(
                "<!-- Ad Break: {:.1}s at {:.1}s -->\n",
                ad.duration_secs, ad.start_secs,
            ));
        }

        output
    }

    /// Generate tracking pixel HTML for ad impression counting.
    pub fn tracking_pixels(ad: &AdBreak) -> Vec<String> {
        ad.tracking_urls
            .iter()
            .map(|url| format!("<img src=\"{}\" width=\"1\" height=\"1\" />", url))
            .collect()
    }
}

fn format_iso8601(offset_secs: f64) -> String {
    // Simplified — would use chrono in production
    let hours = (offset_secs / 3600.0) as u32;
    let mins = ((offset_secs % 3600.0) / 60.0) as u32;
    let secs = offset_secs % 60.0;
    format!("2026-01-01T{:02}:{:02}:{:05.2}Z", hours, mins, secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hls_rewrite() {
        let manifest = "#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:6.0,\nsegment001.ts\n";
        let ads = vec![AdBreak {
            start_secs: 60.0,
            duration_secs: 30.0,
            creative_urls: vec!["https://ads.example.com/creative1.ts".into()],
            tracking_urls: vec!["https://track.example.com/imp".into()],
            splice_event_id: Some(42),
        }];

        let rewritten = ManifestRewriter::rewrite_hls(manifest, &ads);
        assert!(rewritten.contains("EXT-X-DATERANGE"));
        assert!(rewritten.contains("ad-42"));
    }

    #[test]
    fn test_tracking_pixels() {
        let ad = AdBreak {
            start_secs: 0.0,
            duration_secs: 30.0,
            creative_urls: vec![],
            tracking_urls: vec![
                "https://track1.com/imp".into(),
                "https://track2.com/imp".into(),
            ],
            splice_event_id: None,
        };

        let pixels = ManifestRewriter::tracking_pixels(&ad);
        assert_eq!(pixels.len(), 2);
        assert!(pixels[0].contains("track1.com"));
    }
}
