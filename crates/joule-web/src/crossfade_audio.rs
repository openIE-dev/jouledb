//! Audio crossfade and transition system.
//!
//! Crossfade between two audio sources with multiple fade curves,
//! configurable duration, auto-crossfade, gapless playback, and
//! playlist management with pre-buffering.

use std::collections::VecDeque;

// ── Fade Curves ────────────────────────────────────────────────

/// Available crossfade curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadeCurve {
    /// Linear: gain = t (out: 1-t)
    Linear,
    /// Equal-power: uses sin/cos so total power is constant.
    EqualPower,
    /// S-curve: smooth sigmoid transition.
    SCurve,
    /// Logarithmic: fast start, slow end.
    Logarithmic,
}

impl CrossfadeCurve {
    /// Compute incoming gain at position t (0.0 = start, 1.0 = end).
    pub fn incoming_gain(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            CrossfadeCurve::Linear => t,
            CrossfadeCurve::EqualPower => {
                (t * std::f64::consts::FRAC_PI_2).sin()
            }
            CrossfadeCurve::SCurve => {
                // Hermite-style S-curve: 3t^2 - 2t^3
                3.0 * t * t - 2.0 * t * t * t
            }
            CrossfadeCurve::Logarithmic => {
                if t < 1e-12 { return 0.0; }
                let log_min = 0.001f64.ln();
                let log_val = (t.max(0.001)).ln();
                1.0 - (log_val / log_min).clamp(0.0, 1.0)
            }
        }
    }

    /// Compute outgoing gain at position t (0.0 = start, 1.0 = end).
    pub fn outgoing_gain(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            CrossfadeCurve::Linear => 1.0 - t,
            CrossfadeCurve::EqualPower => {
                ((1.0 - t) * std::f64::consts::FRAC_PI_2).sin()
            }
            CrossfadeCurve::SCurve => {
                let s = 1.0 - t;
                3.0 * s * s - 2.0 * s * s * s
            }
            CrossfadeCurve::Logarithmic => {
                1.0 - self.incoming_gain(t)
            }
        }
    }
}

// ── Crossfade State ────────────────────────────────────────────

/// State of a crossfade operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadeState {
    /// No crossfade active.
    Idle,
    /// Crossfade in progress.
    Active,
    /// Crossfade completed.
    Done,
}

/// Crossfade position mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossfadePosition {
    /// Overlap: both tracks play simultaneously during transition.
    Overlap,
    /// Cut: outgoing stops at midpoint, incoming starts at midpoint.
    Cut,
}

/// A crossfade between two audio sources.
#[derive(Debug, Clone, PartialEq)]
pub struct Crossfade {
    pub curve: CrossfadeCurve,
    pub duration_samples: usize,
    pub elapsed_samples: usize,
    pub state: CrossfadeState,
    pub position_mode: CrossfadePosition,
}

impl Crossfade {
    /// Create a new crossfade.
    pub fn new(curve: CrossfadeCurve, duration_samples: usize) -> Self {
        Self {
            curve,
            duration_samples,
            elapsed_samples: 0,
            state: CrossfadeState::Idle,
            position_mode: CrossfadePosition::Overlap,
        }
    }

    /// Create with a specific position mode.
    pub fn with_position(curve: CrossfadeCurve, duration_samples: usize,
                         position: CrossfadePosition) -> Self {
        Self {
            curve,
            duration_samples,
            elapsed_samples: 0,
            state: CrossfadeState::Idle,
            position_mode: position,
        }
    }

    /// Start the crossfade.
    pub fn start(&mut self) {
        self.elapsed_samples = 0;
        self.state = CrossfadeState::Active;
    }

    /// Reset the crossfade.
    pub fn reset(&mut self) {
        self.elapsed_samples = 0;
        self.state = CrossfadeState::Idle;
    }

    /// Current progress (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.duration_samples == 0 { return 1.0; }
        (self.elapsed_samples as f64 / self.duration_samples as f64).clamp(0.0, 1.0)
    }

    /// Whether the crossfade is complete.
    pub fn is_done(&self) -> bool {
        self.state == CrossfadeState::Done || self.elapsed_samples >= self.duration_samples
    }

    /// Process one frame: returns (outgoing_gain, incoming_gain).
    pub fn process_frame(&mut self) -> (f64, f64) {
        match self.state {
            CrossfadeState::Idle => (1.0, 0.0),
            CrossfadeState::Done => (0.0, 1.0),
            CrossfadeState::Active => {
                let t = self.progress();
                let out_gain = self.curve.outgoing_gain(t);
                let in_gain = self.curve.incoming_gain(t);
                self.elapsed_samples += 1;
                if self.elapsed_samples >= self.duration_samples {
                    self.state = CrossfadeState::Done;
                }

                match self.position_mode {
                    CrossfadePosition::Overlap => (out_gain, in_gain),
                    CrossfadePosition::Cut => {
                        if t < 0.5 {
                            (out_gain, 0.0)
                        } else {
                            (0.0, in_gain)
                        }
                    }
                }
            }
        }
    }

    /// Process a buffer pair: apply crossfade gains to outgoing and incoming.
    pub fn process_buffers(&mut self, outgoing: &[f64], incoming: &[f64]) -> Vec<f64> {
        let len = outgoing.len().max(incoming.len());
        let mut result = vec![0.0; len];
        for i in 0..len {
            let (out_g, in_g) = self.process_frame();
            let out_sample = if i < outgoing.len() { outgoing[i] } else { 0.0 };
            let in_sample = if i < incoming.len() { incoming[i] } else { 0.0 };
            result[i] = out_sample * out_g + in_sample * in_g;
        }
        result
    }
}

// ── Track / Playlist ───────────────────────────────────────────

/// A track in the playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTrack {
    pub name: String,
    pub samples: Vec<f64>,
    pub looping: bool,
    pub pre_buffered: bool,
}

impl AudioTrack {
    pub fn new(name: &str, samples: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            samples,
            looping: false,
            pre_buffered: false,
        }
    }

    /// Get the total duration in samples.
    pub fn duration_samples(&self) -> usize {
        self.samples.len()
    }
}

/// Playlist with crossfade support.
#[derive(Debug, Clone)]
pub struct CrossfadePlaylist {
    tracks: VecDeque<AudioTrack>,
    current_position: usize,
    current_index: Option<usize>,
    crossfade_duration: usize,
    crossfade_curve: CrossfadeCurve,
    auto_crossfade: bool,
    active_crossfade: Option<Crossfade>,
    crossfade_start_pos: usize,
    playing: bool,
    looping_playlist: bool,
}

impl CrossfadePlaylist {
    /// Create a new playlist.
    pub fn new(crossfade_duration: usize, curve: CrossfadeCurve) -> Self {
        Self {
            tracks: VecDeque::new(),
            current_position: 0,
            current_index: None,
            crossfade_duration,
            crossfade_curve: curve,
            auto_crossfade: true,
            active_crossfade: None,
            crossfade_start_pos: 0,
            playing: false,
            looping_playlist: false,
        }
    }

    /// Add a track to the end of the playlist.
    pub fn add_track(&mut self, track: AudioTrack) {
        self.tracks.push_back(track);
    }

    /// Insert a track at a position.
    pub fn insert_track(&mut self, index: usize, track: AudioTrack) {
        let idx = index.min(self.tracks.len());
        self.tracks.insert(idx, track);
    }

    /// Remove a track by index.
    pub fn remove_track(&mut self, index: usize) -> Option<AudioTrack> {
        if index < self.tracks.len() {
            Some(self.tracks.remove(index).unwrap())
        } else {
            None
        }
    }

    /// Number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Get current track index.
    pub fn current_index(&self) -> Option<usize> {
        self.current_index
    }

    /// Set auto-crossfade.
    pub fn set_auto_crossfade(&mut self, auto_cf: bool) {
        self.auto_crossfade = auto_cf;
    }

    /// Set crossfade duration in samples.
    pub fn set_crossfade_duration(&mut self, duration: usize) {
        self.crossfade_duration = duration;
    }

    /// Set crossfade curve.
    pub fn set_crossfade_curve(&mut self, curve: CrossfadeCurve) {
        self.crossfade_curve = curve;
    }

    /// Set playlist looping.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping_playlist = looping;
    }

    /// Start playback.
    pub fn play(&mut self) {
        if self.current_index.is_none() && !self.tracks.is_empty() {
            self.current_index = Some(0);
        }
        self.playing = true;
    }

    /// Pause playback.
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Stop playback and reset.
    pub fn stop(&mut self) {
        self.playing = false;
        self.current_position = 0;
        self.current_index = None;
        self.active_crossfade = None;
    }

    /// Skip to next track with crossfade.
    pub fn next(&mut self) {
        if let Some(idx) = self.current_index {
            let next_idx = if idx + 1 < self.tracks.len() {
                idx + 1
            } else if self.looping_playlist {
                0
            } else {
                return;
            };
            self.begin_crossfade_to(next_idx);
        }
    }

    /// Skip to previous track with crossfade.
    pub fn previous(&mut self) {
        if let Some(idx) = self.current_index {
            let prev_idx = if idx > 0 {
                idx - 1
            } else if self.looping_playlist && !self.tracks.is_empty() {
                self.tracks.len() - 1
            } else {
                return;
            };
            self.begin_crossfade_to(prev_idx);
        }
    }

    /// Begin crossfade to a specific track index.
    fn begin_crossfade_to(&mut self, next_idx: usize) {
        let mut cf = Crossfade::new(self.crossfade_curve, self.crossfade_duration);
        cf.start();
        self.active_crossfade = Some(cf);
        self.crossfade_start_pos = self.current_position;
        self.current_index = Some(next_idx);
    }

    /// Pre-buffer the next track (mark it ready).
    pub fn pre_buffer_next(&mut self) {
        if let Some(idx) = self.current_index {
            let next = if idx + 1 < self.tracks.len() {
                idx + 1
            } else if self.looping_playlist {
                0
            } else {
                return;
            };
            if let Some(track) = self.tracks.get_mut(next) {
                track.pre_buffered = true;
            }
        }
    }

    /// Is playback active?
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Process a block of samples from the playlist.
    pub fn process(&mut self, frames: usize) -> Vec<f64> {
        if !self.playing || self.tracks.is_empty() {
            return vec![0.0; frames];
        }

        let mut output = vec![0.0; frames];
        let current_idx = match self.current_index {
            Some(i) if i < self.tracks.len() => i,
            _ => return output,
        };

        // Check if we need auto-crossfade
        if self.auto_crossfade && self.active_crossfade.is_none() {
            let track = &self.tracks[current_idx];
            if !track.looping {
                let remaining = track.duration_samples().saturating_sub(self.current_position);
                if remaining <= self.crossfade_duration && remaining > 0 {
                    let next_idx = if current_idx + 1 < self.tracks.len() {
                        Some(current_idx + 1)
                    } else if self.looping_playlist {
                        Some(0)
                    } else {
                        None
                    };
                    if let Some(next) = next_idx {
                        self.begin_crossfade_to(next);
                    }
                }
            }
        }

        if let Some(ref mut cf) = self.active_crossfade {
            // During crossfade: mix outgoing (previous) and incoming (current)
            let outgoing_idx = if current_idx > 0 {
                current_idx - 1
            } else {
                self.tracks.len() - 1
            };

            for i in 0..frames {
                let (out_g, in_g) = cf.process_frame();

                let out_sample = if outgoing_idx < self.tracks.len() {
                    let track = &self.tracks[outgoing_idx];
                    let pos = self.crossfade_start_pos + i;
                    if pos < track.samples.len() {
                        track.samples[pos]
                    } else if track.looping && !track.samples.is_empty() {
                        track.samples[pos % track.samples.len()]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let in_sample = if current_idx < self.tracks.len() {
                    let track = &self.tracks[current_idx];
                    if i < track.samples.len() {
                        track.samples[i]
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                output[i] = out_sample * out_g + in_sample * in_g;
            }

            if cf.is_done() {
                self.active_crossfade = None;
                self.current_position = frames;
            }
        } else {
            // Normal playback
            let track = &self.tracks[current_idx];
            for i in 0..frames {
                let pos = self.current_position + i;
                let sample = if pos < track.samples.len() {
                    track.samples[pos]
                } else if track.looping && !track.samples.is_empty() {
                    track.samples[pos % track.samples.len()]
                } else {
                    0.0
                };
                output[i] = sample;
            }
            self.current_position += frames;

            // Check if track ended
            if !self.tracks[current_idx].looping
                && self.current_position >= self.tracks[current_idx].duration_samples()
            {
                let next = if current_idx + 1 < self.tracks.len() {
                    Some(current_idx + 1)
                } else if self.looping_playlist {
                    Some(0)
                } else {
                    None
                };
                if let Some(next_idx) = next {
                    self.current_index = Some(next_idx);
                    self.current_position = 0;
                } else {
                    self.playing = false;
                }
            }
        }

        output
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_curve_start() {
        let c = CrossfadeCurve::Linear;
        assert!((c.incoming_gain(0.0) - 0.0).abs() < 1e-10);
        assert!((c.outgoing_gain(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_curve_end() {
        let c = CrossfadeCurve::Linear;
        assert!((c.incoming_gain(1.0) - 1.0).abs() < 1e-10);
        assert!((c.outgoing_gain(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_curve_mid() {
        let c = CrossfadeCurve::Linear;
        assert!((c.incoming_gain(0.5) - 0.5).abs() < 1e-10);
        assert!((c.outgoing_gain(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equal_power_constant_power() {
        let c = CrossfadeCurve::EqualPower;
        // At midpoint, combined power should be ~1.0
        let in_g = c.incoming_gain(0.5);
        let out_g = c.outgoing_gain(0.5);
        let power = in_g * in_g + out_g * out_g;
        assert!((power - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_s_curve_endpoints() {
        let c = CrossfadeCurve::SCurve;
        assert!((c.incoming_gain(0.0) - 0.0).abs() < 1e-10);
        assert!((c.incoming_gain(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_s_curve_midpoint() {
        let c = CrossfadeCurve::SCurve;
        let mid = c.incoming_gain(0.5);
        assert!((mid - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_logarithmic_endpoints() {
        let c = CrossfadeCurve::Logarithmic;
        assert!((c.incoming_gain(0.0) - 0.0).abs() < 1e-6);
        assert!((c.incoming_gain(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_crossfade_idle() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 100);
        let (out, inc) = cf.process_frame();
        assert!((out - 1.0).abs() < 1e-10);
        assert!((inc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_active() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 100);
        cf.start();
        let (out, inc) = cf.process_frame();
        assert!(out > 0.9);
        assert!(inc < 0.1);
    }

    #[test]
    fn test_crossfade_completion() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 10);
        cf.start();
        for _ in 0..10 {
            cf.process_frame();
        }
        assert!(cf.is_done());
    }

    #[test]
    fn test_crossfade_done_gains() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 10);
        cf.start();
        for _ in 0..10 {
            cf.process_frame();
        }
        let (out, inc) = cf.process_frame();
        assert!((out - 0.0).abs() < 1e-10);
        assert!((inc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_cut_mode() {
        let mut cf = Crossfade::with_position(
            CrossfadeCurve::Linear, 100, CrossfadePosition::Cut,
        );
        cf.start();
        // First frame: should hear outgoing only
        let (out, inc) = cf.process_frame();
        assert!(out > 0.0);
        assert!((inc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_crossfade_process_buffers() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 100);
        cf.start();
        let outgoing = vec![1.0; 50];
        let incoming = vec![0.5; 50];
        let result = cf.process_buffers(&outgoing, &incoming);
        assert_eq!(result.len(), 50);
        // First sample: mostly outgoing
        assert!(result[0] > 0.5);
    }

    #[test]
    fn test_crossfade_reset() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 100);
        cf.start();
        for _ in 0..50 {
            cf.process_frame();
        }
        cf.reset();
        assert_eq!(cf.state, CrossfadeState::Idle);
        assert_eq!(cf.elapsed_samples, 0);
    }

    #[test]
    fn test_playlist_create() {
        let pl = CrossfadePlaylist::new(1000, CrossfadeCurve::Linear);
        assert_eq!(pl.track_count(), 0);
        assert!(!pl.is_playing());
    }

    #[test]
    fn test_playlist_add_track() {
        let mut pl = CrossfadePlaylist::new(1000, CrossfadeCurve::Linear);
        pl.add_track(AudioTrack::new("track1", vec![0.5; 44100]));
        assert_eq!(pl.track_count(), 1);
    }

    #[test]
    fn test_playlist_play_stop() {
        let mut pl = CrossfadePlaylist::new(100, CrossfadeCurve::Linear);
        pl.add_track(AudioTrack::new("t1", vec![0.5; 1000]));
        pl.play();
        assert!(pl.is_playing());
        pl.stop();
        assert!(!pl.is_playing());
    }

    #[test]
    fn test_playlist_process() {
        let mut pl = CrossfadePlaylist::new(100, CrossfadeCurve::Linear);
        pl.add_track(AudioTrack::new("t1", vec![0.7; 1000]));
        pl.play();
        let out = pl.process(64);
        assert_eq!(out.len(), 64);
        assert!((out[0] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_playlist_empty_process() {
        let mut pl = CrossfadePlaylist::new(100, CrossfadeCurve::Linear);
        pl.play();
        let out = pl.process(64);
        assert!(out.iter().all(|s| s.abs() < 1e-10));
    }

    #[test]
    fn test_playlist_remove_track() {
        let mut pl = CrossfadePlaylist::new(100, CrossfadeCurve::Linear);
        pl.add_track(AudioTrack::new("t1", vec![0.5; 100]));
        pl.add_track(AudioTrack::new("t2", vec![0.3; 100]));
        pl.remove_track(0);
        assert_eq!(pl.track_count(), 1);
    }

    #[test]
    fn test_playlist_pre_buffer() {
        let mut pl = CrossfadePlaylist::new(100, CrossfadeCurve::Linear);
        pl.add_track(AudioTrack::new("t1", vec![0.5; 1000]));
        pl.add_track(AudioTrack::new("t2", vec![0.3; 1000]));
        pl.play();
        pl.pre_buffer_next();
        // t2 should be marked pre-buffered
        assert!(pl.tracks[1].pre_buffered);
    }

    #[test]
    fn test_playlist_looping() {
        let mut pl = CrossfadePlaylist::new(10, CrossfadeCurve::Linear);
        pl.set_looping(true);
        pl.set_auto_crossfade(false);
        let samples: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        pl.add_track(AudioTrack::new("t1", samples));
        pl.play();
        // Process past end of track
        pl.process(100);
        // Should loop back to start
        assert!(pl.is_playing());
    }

    #[test]
    fn test_track_duration() {
        let track = AudioTrack::new("test", vec![0.0; 44100]);
        assert_eq!(track.duration_samples(), 44100);
    }

    #[test]
    fn test_crossfade_progress() {
        let mut cf = Crossfade::new(CrossfadeCurve::Linear, 100);
        cf.start();
        assert!((cf.progress() - 0.0).abs() < 1e-10);
        for _ in 0..50 {
            cf.process_frame();
        }
        assert!((cf.progress() - 0.5).abs() < 0.02);
    }
}
