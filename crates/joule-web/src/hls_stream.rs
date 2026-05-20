//! HLS streaming — M3U8 playlist parser, adaptive bitrate variant selection.

use std::collections::VecDeque;

// ── Segment ─────────────────────────────────────────────────────

/// A media segment in an HLS playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub uri: String,
    pub duration: f64,
    pub sequence: u64,
    pub discontinuity: bool,
    pub byte_range: Option<(u64, u64)>,
}

// ── Variant ─────────────────────────────────────────────────────

/// A variant stream in a master playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub uri: String,
    pub bandwidth: u64,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
    pub frame_rate: Option<f64>,
    pub name: Option<String>,
}

// ── Playlists ───────────────────────────────────────────────────

/// Parsed master playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct MasterPlaylist {
    pub variants: Vec<Variant>,
}

/// Parsed media playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaPlaylist {
    pub target_duration: f64,
    pub media_sequence: u64,
    pub segments: Vec<Segment>,
    pub ended: bool,
    pub playlist_type: Option<String>,
}

// ── Parser ──────────────────────────────────────────────────────

/// Parse result — either a master or media playlist.
#[derive(Debug, Clone, PartialEq)]
pub enum Playlist {
    Master(MasterPlaylist),
    Media(MediaPlaylist),
}

/// Parse errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    NotM3U8,
    InvalidTag(String),
    MissingUri,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotM3U8 => write!(f, "not a valid M3U8 file"),
            Self::InvalidTag(t) => write!(f, "invalid tag: {t}"),
            Self::MissingUri => write!(f, "missing URI for segment/variant"),
        }
    }
}

impl std::error::Error for ParseError {}

fn parse_attribute(attrs: &str, key: &str) -> Option<String> {
    for part in attrs.split(',') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix(key) {
            if let Some(val) = rest.strip_prefix('=') {
                return Some(val.trim_matches('"').to_string());
            }
        }
    }
    None
}

fn parse_attributes_aware(attrs: &str, key: &str) -> Option<String> {
    // Handle commas inside quoted strings
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in attrs.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
            current.push(ch);
        } else if ch == ',' && !in_quotes {
            result.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    for part in &result {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix(key) {
            if let Some(val) = rest.strip_prefix('=') {
                return Some(val.trim_matches('"').to_string());
            }
        }
    }
    None
}

/// Parse an M3U8 playlist string.
pub fn parse_m3u8(input: &str) -> Result<Playlist, ParseError> {
    let lines: Vec<&str> = input.lines().collect();
    if lines.is_empty() || !lines[0].trim().starts_with("#EXTM3U") {
        return Err(ParseError::NotM3U8);
    }

    // Detect master vs media
    let is_master = lines.iter().any(|l| l.starts_with("#EXT-X-STREAM-INF:"));

    if is_master {
        parse_master(&lines)
    } else {
        parse_media(&lines)
    }
}

fn parse_master(lines: &[&str]) -> Result<Playlist, ParseError> {
    let mut variants = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if let Some(attrs) = line.strip_prefix("#EXT-X-STREAM-INF:") {
            i += 1;
            if i >= lines.len() {
                return Err(ParseError::MissingUri);
            }
            let uri = lines[i].trim().to_string();
            if uri.is_empty() || uri.starts_with('#') {
                return Err(ParseError::MissingUri);
            }
            let bandwidth = parse_attributes_aware(attrs, "BANDWIDTH")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let resolution = parse_attributes_aware(attrs, "RESOLUTION").and_then(|v| {
                let parts: Vec<&str> = v.split('x').collect();
                if parts.len() == 2 {
                    Some((parts[0].parse().ok()?, parts[1].parse().ok()?))
                } else {
                    None
                }
            });
            let codecs = parse_attributes_aware(attrs, "CODECS");
            let frame_rate = parse_attributes_aware(attrs, "FRAME-RATE")
                .and_then(|v| v.parse::<f64>().ok());
            let name = parse_attributes_aware(attrs, "NAME");
            variants.push(Variant {
                uri,
                bandwidth,
                resolution,
                codecs,
                frame_rate,
                name,
            });
        }
        i += 1;
    }
    Ok(Playlist::Master(MasterPlaylist { variants }))
}

fn parse_media(lines: &[&str]) -> Result<Playlist, ParseError> {
    let mut target_duration = 0.0;
    let mut media_sequence: u64 = 0;
    let mut segments = Vec::new();
    let mut ended = false;
    let mut playlist_type = None;
    let mut next_discontinuity = false;
    let mut next_duration: Option<f64> = None;
    let mut seq = 0u64;
    let mut seq_set = false;

    for line in lines {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("#EXT-X-TARGETDURATION:") {
            target_duration = val.trim().parse().unwrap_or(0.0);
        } else if let Some(val) = line.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
            media_sequence = val.trim().parse().unwrap_or(0);
            seq = media_sequence;
            seq_set = true;
        } else if let Some(val) = line.strip_prefix("#EXT-X-PLAYLIST-TYPE:") {
            playlist_type = Some(val.trim().to_string());
        } else if line == "#EXT-X-ENDLIST" {
            ended = true;
        } else if line == "#EXT-X-DISCONTINUITY" {
            next_discontinuity = true;
        } else if let Some(val) = line.strip_prefix("#EXTINF:") {
            let dur_str = val.split(',').next().unwrap_or("0");
            next_duration = Some(dur_str.trim().parse().unwrap_or(0.0));
        } else if !line.is_empty() && !line.starts_with('#') {
            if !seq_set {
                seq_set = true;
            }
            let duration = next_duration.take().unwrap_or(0.0);
            segments.push(Segment {
                uri: line.to_string(),
                duration,
                sequence: seq,
                discontinuity: next_discontinuity,
                byte_range: None,
            });
            next_discontinuity = false;
            seq += 1;
        }
    }

    Ok(Playlist::Media(MediaPlaylist {
        target_duration,
        media_sequence,
        segments,
        ended,
        playlist_type,
    }))
}

// ── Adaptive Bitrate ────────────────────────────────────────────

/// Select the best variant for the estimated bandwidth (bits/sec).
pub fn select_variant(variants: &[Variant], estimated_bps: u64) -> Option<&Variant> {
    // Pick the highest bandwidth variant that fits within estimated bandwidth,
    // with a safety factor of 0.8
    let safe = (estimated_bps as f64 * 0.8) as u64;
    let mut best: Option<&Variant> = None;
    for v in variants {
        if v.bandwidth <= safe {
            match best {
                Some(current) if v.bandwidth > current.bandwidth => best = Some(v),
                None => best = Some(v),
                _ => {}
            }
        }
    }
    // If nothing fits, pick the lowest bandwidth variant
    if best.is_none() {
        best = variants.iter().min_by_key(|v| v.bandwidth);
    }
    best
}

// ── Sliding Window ──────────────────────────────────────────────

/// A sliding window playlist that maintains a limited number of segments.
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    pub segments: VecDeque<Segment>,
    pub max_segments: usize,
    next_sequence: u64,
}

impl SlidingWindow {
    pub fn new(max_segments: usize) -> Self {
        Self {
            segments: VecDeque::new(),
            max_segments,
            next_sequence: 0,
        }
    }

    /// Push a new segment, evicting the oldest if at capacity.
    pub fn push(&mut self, uri: String, duration: f64, discontinuity: bool) {
        if self.segments.len() >= self.max_segments {
            self.segments.pop_front();
        }
        self.segments.push_back(Segment {
            uri,
            duration,
            sequence: self.next_sequence,
            discontinuity,
            byte_range: None,
        });
        self.next_sequence += 1;
    }

    /// Get the media sequence number of the first segment.
    pub fn media_sequence(&self) -> u64 {
        self.segments.front().map(|s| s.sequence).unwrap_or(0)
    }

    /// Total duration of all segments in the window.
    pub fn total_duration(&self) -> f64 {
        self.segments.iter().map(|s| s.duration).sum()
    }
}

// ── Bandwidth Estimator ─────────────────────────────────────────

/// Simple exponentially weighted moving average bandwidth estimator.
#[derive(Debug, Clone)]
pub struct BandwidthEstimator {
    estimate_bps: f64,
    alpha: f64,
}

impl BandwidthEstimator {
    pub fn new(initial_bps: f64) -> Self {
        Self {
            estimate_bps: initial_bps,
            alpha: 0.3,
        }
    }

    /// Record a sample: bytes downloaded in the given duration (seconds).
    pub fn sample(&mut self, bytes: u64, duration_secs: f64) {
        if duration_secs <= 0.0 {
            return;
        }
        let measured = (bytes as f64 * 8.0) / duration_secs;
        self.estimate_bps = self.alpha * measured + (1.0 - self.alpha) * self.estimate_bps;
    }

    /// Current estimate in bits per second.
    pub fn estimate(&self) -> u64 {
        self.estimate_bps as u64
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER: &str = r#"#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1280000,RESOLUTION=640x360
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2560000,RESOLUTION=1280x720
mid.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=7680000,RESOLUTION=1920x1080
high.m3u8
"#;

    const MEDIA: &str = r#"#EXTM3U
#EXT-X-TARGETDURATION:10
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:9.009,
segment0.ts
#EXTINF:9.009,
segment1.ts
#EXTINF:3.003,
segment2.ts
#EXT-X-ENDLIST
"#;

    #[test]
    fn parse_master_playlist() {
        let pl = parse_m3u8(MASTER).unwrap();
        match pl {
            Playlist::Master(m) => {
                assert_eq!(m.variants.len(), 3);
                assert_eq!(m.variants[0].bandwidth, 1_280_000);
                assert_eq!(m.variants[2].resolution, Some((1920, 1080)));
            }
            _ => panic!("expected master playlist"),
        }
    }

    #[test]
    fn parse_media_playlist() {
        let pl = parse_m3u8(MEDIA).unwrap();
        match pl {
            Playlist::Media(m) => {
                assert_eq!(m.segments.len(), 3);
                assert!((m.segments[0].duration - 9.009).abs() < 1e-6);
                assert_eq!(m.segments[2].sequence, 2);
                assert!(m.ended);
                assert!((m.target_duration - 10.0).abs() < 1e-6);
            }
            _ => panic!("expected media playlist"),
        }
    }

    #[test]
    fn parse_not_m3u8() {
        let err = parse_m3u8("not a playlist").unwrap_err();
        assert_eq!(err, ParseError::NotM3U8);
    }

    #[test]
    fn select_variant_for_bandwidth() {
        let pl = parse_m3u8(MASTER).unwrap();
        let variants = match &pl {
            Playlist::Master(m) => &m.variants,
            _ => panic!(),
        };
        // With 3Mbps estimated, should pick mid (2.56M) since 3M*0.8=2.4M < 2.56M
        // Actually 2.4M < 2.56M so mid doesn't fit. Should pick low.
        let v = select_variant(variants, 3_000_000).unwrap();
        assert_eq!(v.uri, "low.m3u8");

        // With 5Mbps, safe = 4M, picks mid (2.56M fits)
        let v = select_variant(variants, 5_000_000).unwrap();
        assert_eq!(v.uri, "mid.m3u8");

        // With 10Mbps, safe = 8M, picks high (7.68M fits)
        let v = select_variant(variants, 10_000_000).unwrap();
        assert_eq!(v.uri, "high.m3u8");
    }

    #[test]
    fn select_variant_low_bandwidth() {
        let pl = parse_m3u8(MASTER).unwrap();
        let variants = match &pl {
            Playlist::Master(m) => &m.variants,
            _ => panic!(),
        };
        // Very low bandwidth — should still return lowest
        let v = select_variant(variants, 100).unwrap();
        assert_eq!(v.uri, "low.m3u8");
    }

    #[test]
    fn sliding_window() {
        let mut sw = SlidingWindow::new(3);
        sw.push("s0.ts".into(), 5.0, false);
        sw.push("s1.ts".into(), 5.0, false);
        sw.push("s2.ts".into(), 5.0, false);
        assert_eq!(sw.segments.len(), 3);
        assert_eq!(sw.media_sequence(), 0);

        sw.push("s3.ts".into(), 5.0, false);
        assert_eq!(sw.segments.len(), 3);
        assert_eq!(sw.media_sequence(), 1);
        assert_eq!(sw.segments.back().unwrap().uri, "s3.ts");
    }

    #[test]
    fn sliding_window_total_duration() {
        let mut sw = SlidingWindow::new(10);
        sw.push("a.ts".into(), 3.0, false);
        sw.push("b.ts".into(), 4.5, false);
        assert!((sw.total_duration() - 7.5).abs() < 1e-9);
    }

    #[test]
    fn bandwidth_estimator() {
        let mut est = BandwidthEstimator::new(1_000_000.0);
        // Download 125KB in 1 second = 1Mbps
        est.sample(125_000, 1.0);
        // EWMA: 0.3 * 1M + 0.7 * 1M = 1M
        assert_eq!(est.estimate(), 1_000_000);

        // Download 250KB in 1 second = 2Mbps
        est.sample(250_000, 1.0);
        // 0.3 * 2M + 0.7 * 1M = 1.3M
        assert_eq!(est.estimate(), 1_300_000);
    }

    #[test]
    fn discontinuity_in_media_playlist() {
        let input = r#"#EXTM3U
#EXT-X-TARGETDURATION:10
#EXTINF:10.0,
seg0.ts
#EXT-X-DISCONTINUITY
#EXTINF:10.0,
seg1.ts
"#;
        let pl = parse_m3u8(input).unwrap();
        match pl {
            Playlist::Media(m) => {
                assert!(!m.segments[0].discontinuity);
                assert!(m.segments[1].discontinuity);
            }
            _ => panic!("expected media"),
        }
    }

    #[test]
    fn variant_codecs() {
        let input = r#"#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=1000000,CODECS="avc1.42e00a,mp4a.40.2"
stream.m3u8
"#;
        let pl = parse_m3u8(input).unwrap();
        match pl {
            Playlist::Master(m) => {
                assert_eq!(m.variants[0].codecs.as_deref(), Some("avc1.42e00a,mp4a.40.2"));
            }
            _ => panic!("expected master"),
        }
    }

    #[test]
    fn empty_sliding_window() {
        let sw = SlidingWindow::new(5);
        assert_eq!(sw.media_sequence(), 0);
        assert!((sw.total_duration() - 0.0).abs() < 1e-9);
    }
}
