//! Media playlist formats — M3U/M3U8 parser and writer, PLS format,
//! extended M3U (duration, title), HLS playlist concepts (media/master
//! playlist, segments, target duration), playlist shuffling, and
//! duration totaling.
//!
//! Pure-Rust replacement for hls-parser, m3u8-parser, and similar
//! Node.js playlist parsing libraries.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from media playlist operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaylistError {
    ParseError(String),
    InvalidFormat(String),
    EmptyPlaylist,
}

impl fmt::Display for PlaylistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "playlist parse error: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "invalid playlist format: {msg}"),
            Self::EmptyPlaylist => write!(f, "playlist is empty"),
        }
    }
}

impl std::error::Error for PlaylistError {}

// ── M3U Entry ───────────────────────────────────────────────────

/// A single entry in an M3U playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct M3uEntry {
    /// Duration in seconds (-1 for unknown).
    pub duration: f64,
    /// Display title.
    pub title: String,
    /// URI/path to the media file.
    pub uri: String,
}

impl M3uEntry {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            duration: -1.0,
            title: String::new(),
            uri: uri.into(),
        }
    }

    pub fn with_duration(mut self, seconds: f64) -> Self {
        self.duration = seconds;
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }
}

// ── M3U Playlist ────────────────────────────────────────────────

/// An M3U/M3U8 playlist.
#[derive(Debug, Clone)]
pub struct M3uPlaylist {
    pub entries: Vec<M3uEntry>,
    pub extended: bool,
}

impl M3uPlaylist {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            extended: true,
        }
    }

    pub fn add_entry(&mut self, entry: M3uEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total duration of all entries (excluding unknown durations).
    pub fn total_duration(&self) -> f64 {
        self.entries
            .iter()
            .filter(|e| e.duration >= 0.0)
            .map(|e| e.duration)
            .sum()
    }

    /// Shuffle playlist entries using a deterministic seed.
    pub fn shuffle(&mut self, seed: u64) {
        let len = self.entries.len();
        if len <= 1 {
            return;
        }
        let mut rng = SimpleRng::new(seed);
        // Fisher-Yates shuffle
        for i in (1..len).rev() {
            let j = rng.next_usize(i + 1);
            self.entries.swap(i, j);
        }
    }

    /// Parse an M3U/M3U8 string.
    pub fn parse(input: &str) -> Result<Self, PlaylistError> {
        let mut playlist = Self::new();
        let lines: Vec<&str> = input.lines().collect();

        if lines.is_empty() {
            return Err(PlaylistError::EmptyPlaylist);
        }

        let first = lines[0].trim();
        playlist.extended = first == "#EXTM3U";

        let start = if playlist.extended { 1 } else { 0 };
        let mut current_duration = -1.0f64;
        let mut current_title = String::new();

        for line in &lines[start..] {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(extinf) = line.strip_prefix("#EXTINF:") {
                // Parse "#EXTINF:duration,title"
                if let Some(comma_pos) = extinf.find(',') {
                    let dur_str = &extinf[..comma_pos];
                    current_duration = dur_str.trim().parse::<f64>().unwrap_or(-1.0);
                    current_title = extinf[comma_pos + 1..].to_string();
                } else {
                    current_duration = extinf.trim().parse::<f64>().unwrap_or(-1.0);
                }
            } else if !line.starts_with('#') {
                // URI line
                playlist.add_entry(
                    M3uEntry::new(line)
                        .with_duration(current_duration)
                        .with_title(current_title.clone()),
                );
                current_duration = -1.0;
                current_title = String::new();
            }
        }

        Ok(playlist)
    }

    /// Render as M3U/M3U8 string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        if self.extended {
            out.push_str("#EXTM3U\n");
        }
        for entry in &self.entries {
            if self.extended {
                if entry.title.is_empty() {
                    let _ = write!(out, "#EXTINF:{:.0},\n", entry.duration);
                } else {
                    let _ = write!(out, "#EXTINF:{:.0},{}\n", entry.duration, entry.title);
                }
            }
            let _ = write!(out, "{}\n", entry.uri);
        }
        out
    }
}

impl Default for M3uPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

// ── PLS Format ──────────────────────────────────────────────────

/// An entry in a PLS playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct PlsEntry {
    pub file: String,
    pub title: String,
    pub length: i64,
}

impl PlsEntry {
    pub fn new(file: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            title: String::new(),
            length: -1,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_length(mut self, seconds: i64) -> Self {
        self.length = seconds;
        self
    }
}

/// A PLS format playlist.
#[derive(Debug, Clone)]
pub struct PlsPlaylist {
    pub entries: Vec<PlsEntry>,
}

impl PlsPlaylist {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, entry: PlsEntry) {
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total duration (excluding unknown lengths).
    pub fn total_duration(&self) -> i64 {
        self.entries
            .iter()
            .filter(|e| e.length > 0)
            .map(|e| e.length)
            .sum()
    }

    /// Parse a PLS string.
    pub fn parse(input: &str) -> Result<Self, PlaylistError> {
        let mut playlist = Self::new();
        let lines: Vec<&str> = input.lines().collect();

        if lines.is_empty() {
            return Err(PlaylistError::EmptyPlaylist);
        }

        // Find number of entries from NumberOfEntries= line
        let num_entries: usize = lines
            .iter()
            .find_map(|line| {
                let trimmed = line.trim();
                trimmed
                    .strip_prefix("NumberOfEntries=")
                    .and_then(|val| val.trim().parse().ok())
            })
            .unwrap_or(0);

        // Pre-fill entries
        for _ in 0..num_entries {
            playlist.entries.push(PlsEntry::new(""));
        }

        for line in &lines {
            let trimmed = line.trim();

            // FileN=...
            if let Some(rest) = trimmed.strip_prefix("File") {
                if let Some(eq_pos) = rest.find('=') {
                    if let Ok(idx) = rest[..eq_pos].parse::<usize>() {
                        let val = &rest[eq_pos + 1..];
                        if idx >= 1 && idx <= playlist.entries.len() {
                            playlist.entries[idx - 1].file = val.to_string();
                        }
                    }
                }
            }

            // TitleN=...
            if let Some(rest) = trimmed.strip_prefix("Title") {
                if let Some(eq_pos) = rest.find('=') {
                    if let Ok(idx) = rest[..eq_pos].parse::<usize>() {
                        let val = &rest[eq_pos + 1..];
                        if idx >= 1 && idx <= playlist.entries.len() {
                            playlist.entries[idx - 1].title = val.to_string();
                        }
                    }
                }
            }

            // LengthN=...
            if let Some(rest) = trimmed.strip_prefix("Length") {
                if let Some(eq_pos) = rest.find('=') {
                    if let Ok(idx) = rest[..eq_pos].parse::<usize>() {
                        let val = &rest[eq_pos + 1..];
                        if let Ok(length) = val.trim().parse::<i64>() {
                            if idx >= 1 && idx <= playlist.entries.len() {
                                playlist.entries[idx - 1].length = length;
                            }
                        }
                    }
                }
            }
        }

        // Remove entries with empty file paths (padding entries not filled)
        playlist.entries.retain(|e| !e.file.is_empty());

        Ok(playlist)
    }

    /// Render as PLS string.
    pub fn render(&self) -> String {
        let mut out = String::from("[playlist]\n");
        for (i, entry) in self.entries.iter().enumerate() {
            let n = i + 1;
            let _ = write!(out, "File{}={}\n", n, entry.file);
            if !entry.title.is_empty() {
                let _ = write!(out, "Title{}={}\n", n, entry.title);
            }
            let _ = write!(out, "Length{}={}\n", n, entry.length);
        }
        let _ = write!(out, "NumberOfEntries={}\n", self.entries.len());
        out.push_str("Version=2\n");
        out
    }
}

impl Default for PlsPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

// ── HLS Segment ─────────────────────────────────────────────────

/// A segment in an HLS media playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct HlsSegment {
    pub uri: String,
    pub duration: f64,
    pub title: Option<String>,
    pub byte_range: Option<(u64, u64)>,
    pub discontinuity: bool,
}

impl HlsSegment {
    pub fn new(uri: impl Into<String>, duration: f64) -> Self {
        Self {
            uri: uri.into(),
            duration,
            title: None,
            byte_range: None,
            discontinuity: false,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_byte_range(mut self, offset: u64, length: u64) -> Self {
        self.byte_range = Some((offset, length));
        self
    }

    pub fn with_discontinuity(mut self) -> Self {
        self.discontinuity = true;
        self
    }
}

// ── HLS Media Playlist ─────────────────────────────────────────

/// An HLS media playlist (contains segments).
#[derive(Debug, Clone)]
pub struct HlsMediaPlaylist {
    pub version: u8,
    pub target_duration: u32,
    pub media_sequence: u64,
    pub segments: Vec<HlsSegment>,
    pub end_list: bool,
    pub playlist_type: Option<HlsPlaylistType>,
}

/// HLS playlist type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlsPlaylistType {
    Vod,
    Event,
}

impl HlsMediaPlaylist {
    pub fn new(target_duration: u32) -> Self {
        Self {
            version: 3,
            target_duration,
            media_sequence: 0,
            segments: Vec::new(),
            end_list: false,
            playlist_type: None,
        }
    }

    pub fn with_version(mut self, version: u8) -> Self {
        self.version = version;
        self
    }

    pub fn with_media_sequence(mut self, seq: u64) -> Self {
        self.media_sequence = seq;
        self
    }

    pub fn with_type(mut self, playlist_type: HlsPlaylistType) -> Self {
        self.playlist_type = Some(playlist_type);
        self
    }

    pub fn with_end_list(mut self) -> Self {
        self.end_list = true;
        self
    }

    pub fn add_segment(&mut self, segment: HlsSegment) {
        self.segments.push(segment);
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Total duration of all segments.
    pub fn total_duration(&self) -> f64 {
        self.segments.iter().map(|s| s.duration).sum()
    }

    /// Parse an HLS media playlist.
    pub fn parse(input: &str) -> Result<Self, PlaylistError> {
        let lines: Vec<&str> = input.lines().collect();
        if lines.is_empty() || lines[0].trim() != "#EXTM3U" {
            return Err(PlaylistError::InvalidFormat(
                "missing #EXTM3U header".into(),
            ));
        }

        let mut playlist = Self::new(0);
        let mut current_duration = 0.0f64;
        let mut current_title: Option<String> = None;
        let mut is_discontinuity = false;

        for line in &lines[1..] {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(rest) = trimmed.strip_prefix("#EXT-X-VERSION:") {
                playlist.version = rest.trim().parse().unwrap_or(3);
            } else if let Some(rest) = trimmed.strip_prefix("#EXT-X-TARGETDURATION:") {
                playlist.target_duration = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("#EXT-X-MEDIA-SEQUENCE:") {
                playlist.media_sequence = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = trimmed.strip_prefix("#EXT-X-PLAYLIST-TYPE:") {
                playlist.playlist_type = match rest.trim() {
                    "VOD" => Some(HlsPlaylistType::Vod),
                    "EVENT" => Some(HlsPlaylistType::Event),
                    _ => None,
                };
            } else if trimmed == "#EXT-X-ENDLIST" {
                playlist.end_list = true;
            } else if trimmed == "#EXT-X-DISCONTINUITY" {
                is_discontinuity = true;
            } else if let Some(extinf) = trimmed.strip_prefix("#EXTINF:") {
                if let Some(comma) = extinf.find(',') {
                    current_duration = extinf[..comma].trim().parse().unwrap_or(0.0);
                    let t = extinf[comma + 1..].trim();
                    if !t.is_empty() {
                        current_title = Some(t.to_string());
                    }
                } else {
                    current_duration = extinf.trim_end_matches(',').trim().parse().unwrap_or(0.0);
                }
            } else if !trimmed.starts_with('#') {
                let mut seg = HlsSegment::new(trimmed, current_duration);
                if let Some(title) = current_title.take() {
                    seg = seg.with_title(title);
                }
                if is_discontinuity {
                    seg = seg.with_discontinuity();
                    is_discontinuity = false;
                }
                playlist.add_segment(seg);
                current_duration = 0.0;
            }
        }

        Ok(playlist)
    }

    /// Render as HLS media playlist string.
    pub fn render(&self) -> String {
        let mut out = String::from("#EXTM3U\n");
        let _ = write!(out, "#EXT-X-VERSION:{}\n", self.version);
        let _ = write!(out, "#EXT-X-TARGETDURATION:{}\n", self.target_duration);

        if self.media_sequence > 0 {
            let _ = write!(out, "#EXT-X-MEDIA-SEQUENCE:{}\n", self.media_sequence);
        }

        if let Some(pt) = &self.playlist_type {
            let type_str = match pt {
                HlsPlaylistType::Vod => "VOD",
                HlsPlaylistType::Event => "EVENT",
            };
            let _ = write!(out, "#EXT-X-PLAYLIST-TYPE:{}\n", type_str);
        }

        for seg in &self.segments {
            if seg.discontinuity {
                out.push_str("#EXT-X-DISCONTINUITY\n");
            }
            if let Some(title) = &seg.title {
                let _ = write!(out, "#EXTINF:{:.3},{}\n", seg.duration, title);
            } else {
                let _ = write!(out, "#EXTINF:{:.3},\n", seg.duration);
            }
            if let Some((offset, length)) = seg.byte_range {
                let _ = write!(out, "#EXT-X-BYTERANGE:{}@{}\n", length, offset);
            }
            let _ = write!(out, "{}\n", seg.uri);
        }

        if self.end_list {
            out.push_str("#EXT-X-ENDLIST\n");
        }

        out
    }
}

// ── HLS Master Playlist ────────────────────────────────────────

/// A variant stream in an HLS master playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct HlsVariant {
    pub uri: String,
    pub bandwidth: u64,
    pub resolution: Option<(u32, u32)>,
    pub codecs: Option<String>,
    pub name: Option<String>,
}

impl HlsVariant {
    pub fn new(uri: impl Into<String>, bandwidth: u64) -> Self {
        Self {
            uri: uri.into(),
            bandwidth,
            resolution: None,
            codecs: None,
            name: None,
        }
    }

    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }

    pub fn with_codecs(mut self, codecs: impl Into<String>) -> Self {
        self.codecs = Some(codecs.into());
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// An HLS master playlist (contains variant streams).
#[derive(Debug, Clone)]
pub struct HlsMasterPlaylist {
    pub variants: Vec<HlsVariant>,
}

impl HlsMasterPlaylist {
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
        }
    }

    pub fn add_variant(&mut self, variant: HlsVariant) {
        self.variants.push(variant);
    }

    pub fn variant_count(&self) -> usize {
        self.variants.len()
    }

    /// Render as master playlist string.
    pub fn render(&self) -> String {
        let mut out = String::from("#EXTM3U\n");
        for var in &self.variants {
            let _ = write!(out, "#EXT-X-STREAM-INF:BANDWIDTH={}", var.bandwidth);
            if let Some((w, h)) = var.resolution {
                let _ = write!(out, ",RESOLUTION={}x{}", w, h);
            }
            if let Some(codecs) = &var.codecs {
                let _ = write!(out, ",CODECS=\"{}\"", codecs);
            }
            if let Some(name) = &var.name {
                let _ = write!(out, ",NAME=\"{}\"", name);
            }
            out.push('\n');
            let _ = write!(out, "{}\n", var.uri);
        }
        out
    }
}

impl Default for HlsMasterPlaylist {
    fn default() -> Self {
        Self::new()
    }
}

// ── Conversion Utilities ────────────────────────────────────────

/// Convert an M3U playlist to PLS format.
pub fn m3u_to_pls(m3u: &M3uPlaylist) -> PlsPlaylist {
    let mut pls = PlsPlaylist::new();
    for entry in &m3u.entries {
        let mut pls_entry = PlsEntry::new(&entry.uri);
        if !entry.title.is_empty() {
            pls_entry = pls_entry.with_title(&entry.title);
        }
        if entry.duration >= 0.0 {
            pls_entry = pls_entry.with_length(entry.duration as i64);
        }
        pls.add_entry(pls_entry);
    }
    pls
}

/// Convert a PLS playlist to M3U format.
pub fn pls_to_m3u(pls: &PlsPlaylist) -> M3uPlaylist {
    let mut m3u = M3uPlaylist::new();
    for entry in &pls.entries {
        let mut m3u_entry = M3uEntry::new(&entry.file);
        if !entry.title.is_empty() {
            m3u_entry = m3u_entry.with_title(&entry.title);
        }
        if entry.length > 0 {
            m3u_entry = m3u_entry.with_duration(entry.length as f64);
        }
        m3u.add_entry(m3u_entry);
    }
    m3u
}

/// Format a duration in seconds as HH:MM:SS.
pub fn format_duration(seconds: f64) -> String {
    let total = seconds as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{:02}:{:02}:{:02}", h, m, s)
    } else {
        format!("{:02}:{:02}", m, s)
    }
}

// ── Simple RNG ──────────────────────────────────────────────────

/// Minimal deterministic PRNG (xorshift64) for shuffle.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x12345678_9ABCDEF0 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m3u_parse_extended() {
        let input = "#EXTM3U\n#EXTINF:180,Song One\nhttp://example.com/song1.mp3\n#EXTINF:240,Song Two\nhttp://example.com/song2.mp3\n";
        let playlist = M3uPlaylist::parse(input).unwrap();
        assert_eq!(playlist.len(), 2);
        assert!((playlist.entries[0].duration - 180.0).abs() < 0.01);
        assert_eq!(playlist.entries[0].title, "Song One");
    }

    #[test]
    fn test_m3u_parse_simple() {
        let input = "http://example.com/song1.mp3\nhttp://example.com/song2.mp3\n";
        let playlist = M3uPlaylist::parse(input).unwrap();
        assert_eq!(playlist.len(), 2);
        assert!(!playlist.extended);
    }

    #[test]
    fn test_m3u_render() {
        let mut playlist = M3uPlaylist::new();
        playlist.add_entry(
            M3uEntry::new("http://example.com/song.mp3")
                .with_duration(200.0)
                .with_title("My Song"),
        );
        let output = playlist.render();
        assert!(output.contains("#EXTM3U"));
        assert!(output.contains("#EXTINF:200,My Song"));
        assert!(output.contains("http://example.com/song.mp3"));
    }

    #[test]
    fn test_m3u_roundtrip() {
        let mut original = M3uPlaylist::new();
        original.add_entry(
            M3uEntry::new("file1.mp3")
                .with_duration(120.0)
                .with_title("Track 1"),
        );
        original.add_entry(
            M3uEntry::new("file2.mp3")
                .with_duration(180.0)
                .with_title("Track 2"),
        );
        let rendered = original.render();
        let parsed = M3uPlaylist::parse(&rendered).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.entries[0].title, "Track 1");
    }

    #[test]
    fn test_m3u_total_duration() {
        let mut playlist = M3uPlaylist::new();
        playlist.add_entry(M3uEntry::new("a.mp3").with_duration(100.0));
        playlist.add_entry(M3uEntry::new("b.mp3").with_duration(200.0));
        playlist.add_entry(M3uEntry::new("c.mp3")); // -1 duration
        assert!((playlist.total_duration() - 300.0).abs() < 0.01);
    }

    #[test]
    fn test_m3u_shuffle() {
        let mut playlist = M3uPlaylist::new();
        for i in 0..10 {
            playlist.add_entry(M3uEntry::new(format!("track{}.mp3", i)));
        }
        let original_uris: Vec<_> = playlist.entries.iter().map(|e| e.uri.clone()).collect();
        playlist.shuffle(42);
        let shuffled_uris: Vec<_> = playlist.entries.iter().map(|e| e.uri.clone()).collect();
        // Should be a permutation
        assert_eq!(shuffled_uris.len(), original_uris.len());
        // Highly unlikely to be the same order with 10 items
        assert_ne!(shuffled_uris, original_uris);
    }

    #[test]
    fn test_pls_parse() {
        let input = "[playlist]\nFile1=http://example.com/song1.mp3\nTitle1=Song One\nLength1=180\nFile2=http://example.com/song2.mp3\nTitle2=Song Two\nLength2=240\nNumberOfEntries=2\nVersion=2\n";
        let playlist = PlsPlaylist::parse(input).unwrap();
        assert_eq!(playlist.len(), 2);
        assert_eq!(playlist.entries[0].file, "http://example.com/song1.mp3");
        assert_eq!(playlist.entries[0].title, "Song One");
        assert_eq!(playlist.entries[0].length, 180);
    }

    #[test]
    fn test_pls_render() {
        let mut playlist = PlsPlaylist::new();
        playlist.add_entry(
            PlsEntry::new("http://example.com/song.mp3")
                .with_title("My Song")
                .with_length(300),
        );
        let output = playlist.render();
        assert!(output.contains("[playlist]"));
        assert!(output.contains("File1=http://example.com/song.mp3"));
        assert!(output.contains("Title1=My Song"));
        assert!(output.contains("Length1=300"));
        assert!(output.contains("NumberOfEntries=1"));
    }

    #[test]
    fn test_pls_total_duration() {
        let mut playlist = PlsPlaylist::new();
        playlist.add_entry(PlsEntry::new("a.mp3").with_length(100));
        playlist.add_entry(PlsEntry::new("b.mp3").with_length(200));
        assert_eq!(playlist.total_duration(), 300);
    }

    #[test]
    fn test_hls_media_playlist_parse() {
        let input = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:9.009,\nsegment0.ts\n#EXTINF:9.009,\nsegment1.ts\n#EXTINF:3.003,\nsegment2.ts\n#EXT-X-ENDLIST\n";
        let playlist = HlsMediaPlaylist::parse(input).unwrap();
        assert_eq!(playlist.version, 3);
        assert_eq!(playlist.target_duration, 10);
        assert_eq!(playlist.segment_count(), 3);
        assert!(playlist.end_list);
    }

    #[test]
    fn test_hls_media_playlist_render() {
        let mut playlist = HlsMediaPlaylist::new(10)
            .with_version(3)
            .with_type(HlsPlaylistType::Vod)
            .with_end_list();
        playlist.add_segment(HlsSegment::new("seg0.ts", 9.009));
        playlist.add_segment(HlsSegment::new("seg1.ts", 9.009));
        let output = playlist.render();
        assert!(output.contains("#EXTM3U"));
        assert!(output.contains("#EXT-X-TARGETDURATION:10"));
        assert!(output.contains("#EXT-X-PLAYLIST-TYPE:VOD"));
        assert!(output.contains("#EXT-X-ENDLIST"));
        assert!(output.contains("seg0.ts"));
    }

    #[test]
    fn test_hls_total_duration() {
        let mut playlist = HlsMediaPlaylist::new(10);
        playlist.add_segment(HlsSegment::new("a.ts", 9.0));
        playlist.add_segment(HlsSegment::new("b.ts", 9.0));
        playlist.add_segment(HlsSegment::new("c.ts", 3.0));
        assert!((playlist.total_duration() - 21.0).abs() < 0.01);
    }

    #[test]
    fn test_hls_discontinuity() {
        let mut playlist = HlsMediaPlaylist::new(10);
        playlist.add_segment(HlsSegment::new("a.ts", 9.0));
        playlist.add_segment(HlsSegment::new("b.ts", 9.0).with_discontinuity());
        let output = playlist.render();
        assert!(output.contains("#EXT-X-DISCONTINUITY"));
    }

    #[test]
    fn test_hls_master_playlist() {
        let mut master = HlsMasterPlaylist::new();
        master.add_variant(
            HlsVariant::new("low.m3u8", 800_000)
                .with_resolution(640, 360)
                .with_codecs("avc1.42e00a"),
        );
        master.add_variant(
            HlsVariant::new("high.m3u8", 3_000_000)
                .with_resolution(1920, 1080),
        );
        let output = master.render();
        assert!(output.contains("#EXT-X-STREAM-INF:"));
        assert!(output.contains("BANDWIDTH=800000"));
        assert!(output.contains("RESOLUTION=640x360"));
        assert!(output.contains("low.m3u8"));
    }

    #[test]
    fn test_m3u_to_pls_conversion() {
        let mut m3u = M3uPlaylist::new();
        m3u.add_entry(
            M3uEntry::new("song.mp3")
                .with_duration(200.0)
                .with_title("My Song"),
        );
        let pls = m3u_to_pls(&m3u);
        assert_eq!(pls.len(), 1);
        assert_eq!(pls.entries[0].file, "song.mp3");
        assert_eq!(pls.entries[0].title, "My Song");
        assert_eq!(pls.entries[0].length, 200);
    }

    #[test]
    fn test_pls_to_m3u_conversion() {
        let mut pls = PlsPlaylist::new();
        pls.add_entry(
            PlsEntry::new("song.mp3")
                .with_title("My Song")
                .with_length(200),
        );
        let m3u = pls_to_m3u(&pls);
        assert_eq!(m3u.len(), 1);
        assert_eq!(m3u.entries[0].uri, "song.mp3");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(65.0), "01:05");
        assert_eq!(format_duration(3661.0), "01:01:01");
        assert_eq!(format_duration(0.0), "00:00");
    }

    #[test]
    fn test_empty_m3u_parse() {
        let result = M3uPlaylist::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_hls_parse_missing_header() {
        let result = HlsMediaPlaylist::parse("not a playlist");
        assert!(result.is_err());
    }

    #[test]
    fn test_m3u_entry_defaults() {
        let entry = M3uEntry::new("test.mp3");
        assert!((entry.duration - (-1.0)).abs() < 0.01);
        assert!(entry.title.is_empty());
    }

    #[test]
    fn test_hls_segment_byte_range() {
        let seg = HlsSegment::new("file.ts", 10.0).with_byte_range(0, 1024);
        assert_eq!(seg.byte_range, Some((0, 1024)));
    }

    #[test]
    fn test_master_playlist_variant_count() {
        let mut master = HlsMasterPlaylist::new();
        assert_eq!(master.variant_count(), 0);
        master.add_variant(HlsVariant::new("a.m3u8", 1_000_000));
        assert_eq!(master.variant_count(), 1);
    }
}
