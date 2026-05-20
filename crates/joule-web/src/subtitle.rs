//! Subtitle parsing — SRT, WebVTT cue timing, cue positioning, active cue queries.

// ── Cue Positioning ─────────────────────────────────────────────

/// Horizontal alignment of a cue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueAlign {
    Start,
    Center,
    End,
    Left,
    Right,
}

/// Positioning for a subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub struct CuePosition {
    /// Vertical position as a percentage (0..100), or None for auto.
    pub line: Option<f64>,
    /// Horizontal position as a percentage (0..100), or None for auto.
    pub position: Option<f64>,
    /// Text alignment.
    pub align: CueAlign,
}

impl Default for CuePosition {
    fn default() -> Self {
        Self {
            line: None,
            position: None,
            align: CueAlign::Center,
        }
    }
}

// ── Cue ─────────────────────────────────────────────────────────

/// A single subtitle cue.
#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    /// Optional cue identifier.
    pub id: Option<String>,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Raw text content (may contain basic formatting tags).
    pub text: String,
    /// Positioning info.
    pub position: CuePosition,
}

impl Cue {
    /// Duration of this cue in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Whether this cue is active at the given timestamp (ms).
    pub fn is_active_at(&self, time_ms: u64) -> bool {
        time_ms >= self.start_ms && time_ms < self.end_ms
    }

    /// Strip formatting tags and return plain text.
    pub fn plain_text(&self) -> String {
        let mut result = String::with_capacity(self.text.len());
        let mut in_tag = false;
        for ch in self.text.chars() {
            if ch == '<' {
                in_tag = true;
            } else if ch == '>' {
                in_tag = false;
            } else if !in_tag {
                result.push(ch);
            }
        }
        result
    }
}

// ── Subtitle Track ──────────────────────────────────────────────

/// A subtitle track containing sorted cues.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleTrack {
    pub label: String,
    pub language: String,
    cues: Vec<Cue>,
}

impl SubtitleTrack {
    pub fn new(label: impl Into<String>, language: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            language: language.into(),
            cues: Vec::new(),
        }
    }

    /// Add a cue, maintaining sort order by start time.
    pub fn add_cue(&mut self, cue: Cue) {
        let pos = self.cues.partition_point(|c| c.start_ms <= cue.start_ms);
        self.cues.insert(pos, cue);
    }

    /// Get all cues.
    pub fn cues(&self) -> &[Cue] {
        &self.cues
    }

    /// Get all cues active at the given timestamp (ms).
    pub fn active_cues_at(&self, time_ms: u64) -> Vec<&Cue> {
        // Binary search for the first cue that could be active,
        // then scan forward.
        let start_idx = self.cues.partition_point(|c| c.end_ms <= time_ms);
        self.cues[start_idx..]
            .iter()
            .take_while(|c| c.start_ms <= time_ms)
            .filter(|c| c.is_active_at(time_ms))
            .collect()
    }

    /// Total number of cues.
    pub fn len(&self) -> usize {
        self.cues.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cues.is_empty()
    }
}

// ── SRT Parser ──────────────────────────────────────────────────

/// Parse error.
#[derive(Debug, Clone, PartialEq)]
pub enum SubtitleError {
    InvalidFormat(String),
    InvalidTimecode(String),
}

impl std::fmt::Display for SubtitleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(m) => write!(f, "invalid format: {m}"),
            Self::InvalidTimecode(m) => write!(f, "invalid timecode: {m}"),
        }
    }
}

impl std::error::Error for SubtitleError {}

/// Parse an SRT timecode like "01:02:03,456" into milliseconds.
fn parse_srt_time(s: &str) -> Result<u64, SubtitleError> {
    let s = s.trim();
    // Format: HH:MM:SS,mmm
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(SubtitleError::InvalidTimecode(s.into()));
    }
    let h: u64 = parts[0].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
    let m: u64 = parts[1].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
    let sec_parts: Vec<&str> = parts[2].split(',').collect();
    if sec_parts.len() != 2 {
        return Err(SubtitleError::InvalidTimecode(s.into()));
    }
    let sec: u64 = sec_parts[0].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
    let ms: u64 = sec_parts[1].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
    Ok(h * 3_600_000 + m * 60_000 + sec * 1_000 + ms)
}

/// Parse an SRT subtitle string into a `SubtitleTrack`.
pub fn parse_srt(input: &str) -> Result<SubtitleTrack, SubtitleError> {
    let mut track = SubtitleTrack::new("SRT", "und");
    let blocks: Vec<&str> = input.trim().split("\n\n").collect();

    for block in blocks {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            continue;
        }
        // First line: sequence number (used as id)
        let id = lines[0].trim().to_string();
        // Second line: timecodes
        let timing = lines[1].trim();
        let arrow_parts: Vec<&str> = timing.split("-->").collect();
        if arrow_parts.len() != 2 {
            return Err(SubtitleError::InvalidFormat(format!("bad timing line: {timing}")));
        }
        let start_ms = parse_srt_time(arrow_parts[0])?;
        let end_ms = parse_srt_time(arrow_parts[1])?;
        // Remaining lines: text
        let text = lines[2..].join("\n");
        track.add_cue(Cue {
            id: Some(id),
            start_ms,
            end_ms,
            text,
            position: CuePosition::default(),
        });
    }
    Ok(track)
}

// ── VTT Parser ──────────────────────────────────────────────────

/// Parse a VTT timecode like "01:02:03.456" or "02:03.456" into milliseconds.
fn parse_vtt_time(s: &str) -> Result<u64, SubtitleError> {
    let s = s.trim();
    let parts: Vec<&str> = s.split(':').collect();
    match parts.len() {
        3 => {
            let h: u64 = parts[0].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            let m: u64 = parts[1].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            let sec_parts: Vec<&str> = parts[2].split('.').collect();
            if sec_parts.len() != 2 {
                return Err(SubtitleError::InvalidTimecode(s.into()));
            }
            let sec: u64 = sec_parts[0].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            let ms: u64 = sec_parts[1].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            Ok(h * 3_600_000 + m * 60_000 + sec * 1_000 + ms)
        }
        2 => {
            let m: u64 = parts[0].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            let sec_parts: Vec<&str> = parts[1].split('.').collect();
            if sec_parts.len() != 2 {
                return Err(SubtitleError::InvalidTimecode(s.into()));
            }
            let sec: u64 = sec_parts[0].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            let ms: u64 = sec_parts[1].parse().map_err(|_| SubtitleError::InvalidTimecode(s.into()))?;
            Ok(m * 60_000 + sec * 1_000 + ms)
        }
        _ => Err(SubtitleError::InvalidTimecode(s.into())),
    }
}

/// Parse cue settings from a VTT timing line (after the timestamps).
fn parse_vtt_settings(settings: &str) -> CuePosition {
    let mut pos = CuePosition::default();
    for part in settings.split_whitespace() {
        if let Some(val) = part.strip_prefix("line:") {
            if let Some(pct) = val.strip_suffix('%') {
                pos.line = pct.parse().ok();
            }
        } else if let Some(val) = part.strip_prefix("position:") {
            if let Some(pct) = val.strip_suffix('%') {
                pos.position = pct.parse().ok();
            }
        } else if let Some(val) = part.strip_prefix("align:") {
            pos.align = match val {
                "start" => CueAlign::Start,
                "center" | "middle" => CueAlign::Center,
                "end" => CueAlign::End,
                "left" => CueAlign::Left,
                "right" => CueAlign::Right,
                _ => CueAlign::Center,
            };
        }
    }
    pos
}

/// Parse a WebVTT subtitle string into a `SubtitleTrack`.
pub fn parse_vtt(input: &str) -> Result<SubtitleTrack, SubtitleError> {
    let mut track = SubtitleTrack::new("VTT", "und");
    let lines: Vec<&str> = input.lines().collect();

    if lines.is_empty() || !lines[0].trim().starts_with("WEBVTT") {
        return Err(SubtitleError::InvalidFormat("missing WEBVTT header".into()));
    }

    let mut i = 1;
    while i < lines.len() {
        // Skip blank lines and NOTE blocks
        let line = lines[i].trim();
        if line.is_empty() || line.starts_with("NOTE") {
            i += 1;
            continue;
        }

        // Check if this line is a timing line (contains "-->")
        let (cue_id, timing_idx) = if line.contains("-->") {
            (None, i)
        } else {
            // This might be a cue id
            if i + 1 < lines.len() && lines[i + 1].contains("-->") {
                (Some(line.to_string()), i + 1)
            } else {
                i += 1;
                continue;
            }
        };

        let timing_line = lines[timing_idx].trim();
        let arrow_pos = timing_line.find("-->").unwrap();
        let start_str = &timing_line[..arrow_pos];
        let after_arrow = &timing_line[arrow_pos + 3..];
        // After the end timecode there may be settings
        let after_parts: Vec<&str> = after_arrow.trim().splitn(2, ' ').collect();
        let end_str = after_parts[0];
        let settings = if after_parts.len() > 1 { after_parts[1] } else { "" };

        let start_ms = parse_vtt_time(start_str)?;
        let end_ms = parse_vtt_time(end_str)?;
        let position = parse_vtt_settings(settings);

        // Collect text lines until blank line
        let mut text_lines = Vec::new();
        let mut j = timing_idx + 1;
        while j < lines.len() && !lines[j].trim().is_empty() {
            text_lines.push(lines[j].trim());
            j += 1;
        }

        track.add_cue(Cue {
            id: cue_id,
            start_ms,
            end_ms,
            text: text_lines.join("\n"),
            position,
        });

        i = j + 1;
    }

    Ok(track)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SRT_SAMPLE: &str = "\
1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,500
This is a subtitle.

3
00:01:00,000 --> 00:01:05,000
Third cue.
";

    const VTT_SAMPLE: &str = "\
WEBVTT

1
00:00:01.000 --> 00:00:04.000
Hello from VTT!

2
00:00:05.000 --> 00:00:08.500 align:start position:10%
<b>Bold</b> and <i>italic</i>.
";

    #[test]
    fn parse_srt_basic() {
        let track = parse_srt(SRT_SAMPLE).unwrap();
        assert_eq!(track.len(), 3);
        assert_eq!(track.cues()[0].text, "Hello, world!");
        assert_eq!(track.cues()[0].start_ms, 1_000);
        assert_eq!(track.cues()[0].end_ms, 4_000);
    }

    #[test]
    fn parse_srt_timecodes() {
        let track = parse_srt(SRT_SAMPLE).unwrap();
        assert_eq!(track.cues()[2].start_ms, 60_000);
        assert_eq!(track.cues()[2].end_ms, 65_000);
    }

    #[test]
    fn srt_cue_duration() {
        let track = parse_srt(SRT_SAMPLE).unwrap();
        assert_eq!(track.cues()[1].duration_ms(), 3_500);
    }

    #[test]
    fn active_cues_at_timestamp() {
        let track = parse_srt(SRT_SAMPLE).unwrap();
        let active = track.active_cues_at(2_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "Hello, world!");

        let active = track.active_cues_at(4_500);
        assert_eq!(active.len(), 0);

        let active = track.active_cues_at(6_000);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].text, "This is a subtitle.");
    }

    #[test]
    fn parse_vtt_basic() {
        let track = parse_vtt(VTT_SAMPLE).unwrap();
        assert_eq!(track.len(), 2);
        assert_eq!(track.cues()[0].text, "Hello from VTT!");
    }

    #[test]
    fn vtt_positioning() {
        let track = parse_vtt(VTT_SAMPLE).unwrap();
        let cue = &track.cues()[1];
        assert_eq!(cue.position.align, CueAlign::Start);
        assert!((cue.position.position.unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn vtt_formatting_tags() {
        let track = parse_vtt(VTT_SAMPLE).unwrap();
        let cue = &track.cues()[1];
        assert!(cue.text.contains("<b>"));
        assert_eq!(cue.plain_text(), "Bold and italic.");
    }

    #[test]
    fn vtt_missing_header() {
        let err = parse_vtt("not a vtt file").unwrap_err();
        assert!(matches!(err, SubtitleError::InvalidFormat(_)));
    }

    #[test]
    fn srt_invalid_timecode() {
        let bad = "1\nbad timecode\nHello\n";
        let err = parse_srt(bad).unwrap_err();
        assert!(matches!(err, SubtitleError::InvalidFormat(_)));
    }

    #[test]
    fn cue_is_active_at_boundaries() {
        let cue = Cue {
            id: None,
            start_ms: 1000,
            end_ms: 2000,
            text: "test".into(),
            position: CuePosition::default(),
        };
        assert!(cue.is_active_at(1000)); // inclusive start
        assert!(!cue.is_active_at(2000)); // exclusive end
        assert!(cue.is_active_at(1500));
        assert!(!cue.is_active_at(999));
    }

    #[test]
    fn vtt_short_timecodes() {
        let input = "WEBVTT\n\n02:03.456 --> 02:10.000\nShort format\n";
        let track = parse_vtt(input).unwrap();
        assert_eq!(track.cues()[0].start_ms, 123_456);
    }

    #[test]
    fn subtitle_track_sorted_insert() {
        let mut track = SubtitleTrack::new("test", "en");
        track.add_cue(Cue {
            id: None, start_ms: 5000, end_ms: 6000,
            text: "second".into(), position: CuePosition::default(),
        });
        track.add_cue(Cue {
            id: None, start_ms: 1000, end_ms: 2000,
            text: "first".into(), position: CuePosition::default(),
        });
        assert_eq!(track.cues()[0].text, "first");
        assert_eq!(track.cues()[1].text, "second");
    }
}
