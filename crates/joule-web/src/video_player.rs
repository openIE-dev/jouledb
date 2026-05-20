//! Video player state machine — playback controls, tracks, subtitles, events.

use std::collections::VecDeque;

// ── Playback State ──────────────────────────────────────────────

/// Video playback state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoState {
    Idle,
    Playing,
    Paused,
    Buffering,
    Ended,
    Error,
}

/// A time range representing a buffered region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}

impl TimeRange {
    pub fn new(start: f64, end: f64) -> Self {
        assert!(end >= start, "end must be >= start");
        Self { start, end }
    }

    /// Duration of this range in seconds.
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    /// Whether a timestamp falls within this range.
    pub fn contains(&self, t: f64) -> bool {
        t >= self.start && t <= self.end
    }
}

// ── Video Track ─────────────────────────────────────────────────

/// Video track metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoTrack {
    pub id: u32,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub codec: String,
    pub bitrate: Option<u64>,
    pub active: bool,
}

impl VideoTrack {
    pub fn resolution_label(&self) -> String {
        match self.height {
            0..=360 => "360p".into(),
            361..=480 => "480p".into(),
            481..=720 => "720p".into(),
            721..=1080 => "1080p".into(),
            1081..=1440 => "1440p".into(),
            _ => "4K+".into(),
        }
    }
}

// ── Subtitle Track ──────────────────────────────────────────────

/// A subtitle track reference.
#[derive(Debug, Clone, PartialEq)]
pub struct SubtitleTrackRef {
    pub id: u32,
    pub label: String,
    pub language: String,
    pub active: bool,
}

// ── Events ──────────────────────────────────────────────────────

/// Events emitted by the video player.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoEvent {
    Play,
    Pause,
    Seek { from: f64, to: f64 },
    Ended,
    RateChange(f64),
    VolumeChange(f64),
    Mute(bool),
    BufferUpdate(Vec<TimeRange>),
    TrackChange(u32),
    SubtitleChange(Option<u32>),
    Error(String),
    TimeUpdate(f64),
    DurationChange(f64),
}

// ── Player ──────────────────────────────────────────────────────

/// Video player state and controls.
#[derive(Debug, Clone)]
pub struct VideoPlayer {
    pub state: VideoState,
    pub current_time: f64,
    pub duration: f64,
    pub playback_rate: f64,
    pub volume: f64,
    pub muted: bool,
    pub buffered: Vec<TimeRange>,
    pub tracks: Vec<VideoTrack>,
    pub subtitle_tracks: Vec<SubtitleTrackRef>,
    pub active_subtitle: Option<u32>,
    pub loop_playback: bool,
    events: VecDeque<VideoEvent>,
}

impl VideoPlayer {
    /// Create a new video player with the given duration.
    pub fn new(duration: f64) -> Self {
        Self {
            state: VideoState::Idle,
            current_time: 0.0,
            duration,
            playback_rate: 1.0,
            volume: 1.0,
            muted: false,
            buffered: Vec::new(),
            tracks: Vec::new(),
            subtitle_tracks: Vec::new(),
            active_subtitle: None,
            loop_playback: false,
            events: VecDeque::new(),
        }
    }

    /// Drain all pending events.
    pub fn drain_events(&mut self) -> Vec<VideoEvent> {
        self.events.drain(..).collect()
    }

    /// Start or resume playback.
    pub fn play(&mut self) {
        match self.state {
            VideoState::Idle | VideoState::Paused | VideoState::Buffering => {
                self.state = VideoState::Playing;
                self.events.push_back(VideoEvent::Play);
            }
            VideoState::Ended if self.loop_playback => {
                self.current_time = 0.0;
                self.state = VideoState::Playing;
                self.events.push_back(VideoEvent::Seek { from: self.duration, to: 0.0 });
                self.events.push_back(VideoEvent::Play);
            }
            _ => {}
        }
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        if self.state == VideoState::Playing || self.state == VideoState::Buffering {
            self.state = VideoState::Paused;
            self.events.push_back(VideoEvent::Pause);
        }
    }

    /// Seek to a specific time in seconds.
    pub fn seek(&mut self, time: f64) {
        let clamped = time.clamp(0.0, self.duration);
        let from = self.current_time;
        self.current_time = clamped;
        self.events.push_back(VideoEvent::Seek { from, to: clamped });
        if self.state == VideoState::Ended && clamped < self.duration {
            self.state = VideoState::Paused;
        }
    }

    /// Set playback rate (e.g. 0.5, 1.0, 2.0).
    pub fn set_rate(&mut self, rate: f64) {
        let clamped = rate.clamp(0.25, 4.0);
        self.playback_rate = clamped;
        self.events.push_back(VideoEvent::RateChange(clamped));
    }

    /// Set volume (0.0..=1.0).
    pub fn set_volume(&mut self, vol: f64) {
        let clamped = vol.clamp(0.0, 1.0);
        self.volume = clamped;
        self.events.push_back(VideoEvent::VolumeChange(clamped));
    }

    /// Toggle mute.
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        self.events.push_back(VideoEvent::Mute(muted));
    }

    /// Simulate a time update (called by the runtime loop).
    pub fn tick(&mut self, delta_seconds: f64) {
        if self.state != VideoState::Playing {
            return;
        }
        let advance = delta_seconds * self.playback_rate;
        self.current_time += advance;
        if self.current_time >= self.duration {
            self.current_time = self.duration;
            if self.loop_playback {
                self.current_time = 0.0;
                self.events.push_back(VideoEvent::Seek { from: self.duration, to: 0.0 });
            } else {
                self.state = VideoState::Ended;
                self.events.push_back(VideoEvent::Ended);
            }
        }
        self.events.push_back(VideoEvent::TimeUpdate(self.current_time));
    }

    /// Update buffered ranges.
    pub fn set_buffered(&mut self, ranges: Vec<TimeRange>) {
        self.buffered = ranges.clone();
        self.events.push_back(VideoEvent::BufferUpdate(ranges));
    }

    /// Add a video track.
    pub fn add_track(&mut self, track: VideoTrack) {
        self.tracks.push(track);
    }

    /// Select a video track by id.
    pub fn select_track(&mut self, id: u32) -> bool {
        let mut found = false;
        for t in &mut self.tracks {
            if t.id == id {
                t.active = true;
                found = true;
            } else {
                t.active = false;
            }
        }
        if found {
            self.events.push_back(VideoEvent::TrackChange(id));
        }
        found
    }

    /// Add a subtitle track.
    pub fn add_subtitle_track(&mut self, track: SubtitleTrackRef) {
        self.subtitle_tracks.push(track);
    }

    /// Select a subtitle track (or None to disable).
    pub fn select_subtitle(&mut self, id: Option<u32>) {
        for st in &mut self.subtitle_tracks {
            st.active = id == Some(st.id);
        }
        self.active_subtitle = id;
        self.events.push_back(VideoEvent::SubtitleChange(id));
    }

    /// Report an error.
    pub fn report_error(&mut self, msg: impl Into<String>) {
        self.state = VideoState::Error;
        self.events.push_back(VideoEvent::Error(msg.into()));
    }

    /// Signal buffering.
    pub fn set_buffering(&mut self) {
        if self.state == VideoState::Playing {
            self.state = VideoState::Buffering;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_pause_cycle() {
        let mut p = VideoPlayer::new(60.0);
        p.play();
        assert_eq!(p.state, VideoState::Playing);
        p.pause();
        assert_eq!(p.state, VideoState::Paused);
        let evts = p.drain_events();
        assert_eq!(evts.len(), 2);
        assert_eq!(evts[0], VideoEvent::Play);
        assert_eq!(evts[1], VideoEvent::Pause);
    }

    #[test]
    fn seek_clamps() {
        let mut p = VideoPlayer::new(120.0);
        p.seek(200.0);
        assert_eq!(p.current_time, 120.0);
        p.seek(-5.0);
        assert_eq!(p.current_time, 0.0);
    }

    #[test]
    fn tick_advances_time() {
        let mut p = VideoPlayer::new(10.0);
        p.play();
        p.drain_events();
        p.tick(1.0);
        assert!((p.current_time - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tick_at_rate() {
        let mut p = VideoPlayer::new(100.0);
        p.play();
        p.set_rate(2.0);
        p.drain_events();
        p.tick(1.0);
        assert!((p.current_time - 2.0).abs() < 1e-9);
    }

    #[test]
    fn tick_ends_playback() {
        let mut p = VideoPlayer::new(2.0);
        p.play();
        p.drain_events();
        p.tick(3.0);
        assert_eq!(p.state, VideoState::Ended);
        assert_eq!(p.current_time, 2.0);
    }

    #[test]
    fn loop_restarts() {
        let mut p = VideoPlayer::new(2.0);
        p.loop_playback = true;
        p.play();
        p.drain_events();
        p.tick(3.0);
        assert_eq!(p.state, VideoState::Playing);
        assert!(p.current_time < 2.0);
    }

    #[test]
    fn volume_and_mute() {
        let mut p = VideoPlayer::new(10.0);
        p.set_volume(0.5);
        assert!((p.volume - 0.5).abs() < 1e-9);
        p.set_volume(2.0);
        assert!((p.volume - 1.0).abs() < 1e-9);
        p.set_muted(true);
        assert!(p.muted);
    }

    #[test]
    fn track_selection() {
        let mut p = VideoPlayer::new(10.0);
        p.add_track(VideoTrack {
            id: 1, width: 1920, height: 1080, fps: 30.0,
            codec: "avc1".into(), bitrate: Some(5_000_000), active: false,
        });
        p.add_track(VideoTrack {
            id: 2, width: 1280, height: 720, fps: 30.0,
            codec: "avc1".into(), bitrate: Some(2_500_000), active: false,
        });
        assert!(p.select_track(2));
        assert!(p.tracks[1].active);
        assert!(!p.tracks[0].active);
    }

    #[test]
    fn subtitle_selection() {
        let mut p = VideoPlayer::new(10.0);
        p.add_subtitle_track(SubtitleTrackRef {
            id: 1, label: "English".into(), language: "en".into(), active: false,
        });
        p.select_subtitle(Some(1));
        assert_eq!(p.active_subtitle, Some(1));
        assert!(p.subtitle_tracks[0].active);
        p.select_subtitle(None);
        assert_eq!(p.active_subtitle, None);
        assert!(!p.subtitle_tracks[0].active);
    }

    #[test]
    fn time_range() {
        let r = TimeRange::new(5.0, 15.0);
        assert!((r.duration() - 10.0).abs() < 1e-9);
        assert!(r.contains(10.0));
        assert!(!r.contains(3.0));
    }

    #[test]
    fn resolution_label() {
        let t = VideoTrack {
            id: 1, width: 1920, height: 1080, fps: 30.0,
            codec: "avc1".into(), bitrate: None, active: true,
        };
        assert_eq!(t.resolution_label(), "1080p");
    }

    #[test]
    fn error_state() {
        let mut p = VideoPlayer::new(10.0);
        p.play();
        p.report_error("decode failure");
        assert_eq!(p.state, VideoState::Error);
        let evts = p.drain_events();
        assert!(evts.iter().any(|e| matches!(e, VideoEvent::Error(msg) if msg == "decode failure")));
    }

    #[test]
    fn buffering_state() {
        let mut p = VideoPlayer::new(10.0);
        p.play();
        p.set_buffering();
        assert_eq!(p.state, VideoState::Buffering);
        p.play();
        assert_eq!(p.state, VideoState::Playing);
    }

    #[test]
    fn seek_from_ended() {
        let mut p = VideoPlayer::new(5.0);
        p.play();
        p.tick(10.0);
        assert_eq!(p.state, VideoState::Ended);
        p.seek(2.0);
        assert_eq!(p.state, VideoState::Paused);
        assert!((p.current_time - 2.0).abs() < 1e-9);
    }
}
