//! Step sequencer and timeline engine.
//!
//! Provides a grid-based step sequencer with per-step note, velocity, gate
//! length, and probability. Supports pattern chaining, song arrangement,
//! swing timing, per-track mute/solo, and tick-based transport with
//! configurable PPQ (pulses per quarter note). Pure Rust.

use std::collections::HashMap;

// ── Step & Pattern ───────────────────────────────────────────────

/// A single step in the sequencer grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub note: u8,
    pub velocity: u8,
    pub gate_length: f64, // 0.0..=1.0 fraction of step duration
    pub probability: f64, // 0.0..=1.0 chance of playing
    pub active: bool,
}

impl Default for Step {
    fn default() -> Self {
        Self {
            note: 60,
            velocity: 100,
            gate_length: 0.5,
            probability: 1.0,
            active: false,
        }
    }
}

/// A track is a sequence of steps.
#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub name: String,
    pub steps: Vec<Step>,
    pub muted: bool,
    pub solo: bool,
    pub channel: u8,
}

impl Track {
    pub fn new(name: &str, num_steps: usize, channel: u8) -> Self {
        Self {
            name: name.to_string(),
            steps: (0..num_steps).map(|_| Step::default()).collect(),
            muted: false,
            solo: false,
            channel,
        }
    }

    /// Set a step's note and activate it.
    pub fn set_step(&mut self, idx: usize, note: u8, velocity: u8) {
        if let Some(step) = self.steps.get_mut(idx) {
            step.note = note;
            step.velocity = velocity;
            step.active = true;
        }
    }

    /// Toggle a step on/off.
    pub fn toggle_step(&mut self, idx: usize) {
        if let Some(step) = self.steps.get_mut(idx) {
            step.active = !step.active;
        }
    }

    /// Set probability for a step.
    pub fn set_probability(&mut self, idx: usize, prob: f64) {
        if let Some(step) = self.steps.get_mut(idx) {
            step.probability = prob.clamp(0.0, 1.0);
        }
    }

    /// Set gate length for a step.
    pub fn set_gate(&mut self, idx: usize, gate: f64) {
        if let Some(step) = self.steps.get_mut(idx) {
            step.gate_length = gate.clamp(0.0, 1.0);
        }
    }
}

/// A pattern contains multiple tracks with the same step count.
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub name: String,
    pub tracks: Vec<Track>,
    pub num_steps: usize,
    pub swing: f64, // 0.0 = no swing, 0.5 = 50% swing (even steps delayed by half step)
}

impl Pattern {
    pub fn new(name: &str, num_steps: usize) -> Self {
        Self {
            name: name.to_string(),
            tracks: Vec::new(),
            num_steps,
            swing: 0.0,
        }
    }

    /// Add a track to the pattern.
    pub fn add_track(&mut self, name: &str, channel: u8) -> usize {
        let idx = self.tracks.len();
        self.tracks.push(Track::new(name, self.num_steps, channel));
        idx
    }

    /// Remove a track by index.
    pub fn remove_track(&mut self, idx: usize) -> Option<Track> {
        if idx < self.tracks.len() {
            Some(self.tracks.remove(idx))
        } else {
            None
        }
    }

    /// Set swing amount (0.0 to 1.0).
    pub fn set_swing(&mut self, amount: f64) {
        self.swing = amount.clamp(0.0, 1.0);
    }

    /// Calculate the tick offset for a given step index, including swing.
    pub fn step_tick(&self, step: usize, ticks_per_step: u32) -> u32 {
        let base = step as u32 * ticks_per_step;
        if step % 2 == 1 && self.swing > 0.0 {
            let swing_offset = (ticks_per_step as f64 * self.swing * 0.5) as u32;
            base + swing_offset
        } else {
            base
        }
    }

    /// Get all active steps with timing for a given track.
    /// Returns: (step_index, tick_offset, note, velocity, gate_ticks).
    pub fn active_steps(&self, track_idx: usize, ticks_per_step: u32) -> Vec<(usize, u32, u8, u8, u32)> {
        let track = match self.tracks.get(track_idx) {
            Some(t) => t,
            None => return Vec::new(),
        };
        let mut result = Vec::new();
        for (i, step) in track.steps.iter().enumerate() {
            if step.active {
                let tick = self.step_tick(i, ticks_per_step);
                let gate_ticks = (ticks_per_step as f64 * step.gate_length) as u32;
                result.push((i, tick, step.note, step.velocity, gate_ticks));
            }
        }
        result
    }

    /// Check which tracks should sound considering mute/solo.
    pub fn audible_tracks(&self) -> Vec<usize> {
        let any_solo = self.tracks.iter().any(|t| t.solo);
        self.tracks
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                if any_solo {
                    t.solo && !t.muted
                } else {
                    !t.muted
                }
            })
            .map(|(i, _)| i)
            .collect()
    }
}

// ── Song Arrangement ─────────────────────────────────────────────

/// An entry in the song arrangement (pattern index + repeat count).
#[derive(Debug, Clone, PartialEq)]
pub struct ArrangementEntry {
    pub pattern_index: usize,
    pub repeat_count: u32,
}

/// Song arrangement: a sequence of patterns with repeat counts.
#[derive(Debug, Clone, PartialEq)]
pub struct Song {
    pub name: String,
    pub patterns: Vec<Pattern>,
    pub arrangement: Vec<ArrangementEntry>,
}

impl Song {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            patterns: Vec::new(),
            arrangement: Vec::new(),
        }
    }

    /// Add a pattern to the song's library.
    pub fn add_pattern(&mut self, pattern: Pattern) -> usize {
        let idx = self.patterns.len();
        self.patterns.push(pattern);
        idx
    }

    /// Append a pattern to the arrangement.
    pub fn arrange(&mut self, pattern_index: usize, repeat_count: u32) {
        if pattern_index < self.patterns.len() {
            self.arrangement.push(ArrangementEntry { pattern_index, repeat_count });
        }
    }

    /// Total number of pattern plays in the arrangement.
    pub fn total_plays(&self) -> u32 {
        self.arrangement.iter().map(|e| e.repeat_count).sum()
    }

    /// Flatten arrangement into ordered list of pattern indices.
    pub fn flatten_arrangement(&self) -> Vec<usize> {
        let mut flat = Vec::new();
        for entry in &self.arrangement {
            for _ in 0..entry.repeat_count {
                flat.push(entry.pattern_index);
            }
        }
        flat
    }

    /// Total ticks for the entire song.
    pub fn total_ticks(&self, ticks_per_step: u32) -> u64 {
        let mut total: u64 = 0;
        for entry in &self.arrangement {
            if let Some(pat) = self.patterns.get(entry.pattern_index) {
                let pat_ticks = pat.num_steps as u64 * ticks_per_step as u64;
                total += pat_ticks * entry.repeat_count as u64;
            }
        }
        total
    }
}

// ── Transport ────────────────────────────────────────────────────

/// Transport state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransportState {
    Stopped,
    Playing,
    Paused,
}

/// Tick-based transport with PPQ timing.
#[derive(Debug, Clone, PartialEq)]
pub struct Transport {
    pub state: TransportState,
    pub tick: u64,
    pub ppq: u32, // pulses per quarter note
    pub bpm: f64,
    ticks_per_step: u32,
    steps_per_beat: u32,
}

impl Transport {
    pub fn new(bpm: f64, ppq: u32, steps_per_beat: u32) -> Self {
        let ticks_per_step = ppq / steps_per_beat;
        Self {
            state: TransportState::Stopped,
            tick: 0,
            ppq,
            bpm,
            ticks_per_step,
            steps_per_beat,
        }
    }

    pub fn play(&mut self) {
        self.state = TransportState::Playing;
    }

    pub fn stop(&mut self) {
        self.state = TransportState::Stopped;
        self.tick = 0;
    }

    pub fn pause(&mut self) {
        if self.state == TransportState::Playing {
            self.state = TransportState::Paused;
        }
    }

    pub fn resume(&mut self) {
        if self.state == TransportState::Paused {
            self.state = TransportState::Playing;
        }
    }

    pub fn seek(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Advance by N ticks. Returns true if still playing.
    pub fn advance(&mut self, ticks: u64) -> bool {
        if self.state == TransportState::Playing {
            self.tick += ticks;
            true
        } else {
            false
        }
    }

    /// Current beat number (0-indexed).
    pub fn current_beat(&self) -> u64 {
        self.tick / self.ppq as u64
    }

    /// Current step within the current beat.
    pub fn current_step_in_beat(&self) -> u32 {
        let beat_tick = (self.tick % self.ppq as u64) as u32;
        beat_tick / self.ticks_per_step
    }

    /// Ticks per step.
    pub fn ticks_per_step(&self) -> u32 {
        self.ticks_per_step
    }

    /// Microseconds per tick at current BPM.
    pub fn us_per_tick(&self) -> f64 {
        let us_per_beat = 60_000_000.0 / self.bpm;
        us_per_beat / self.ppq as f64
    }

    /// Seconds elapsed at current tick.
    pub fn elapsed_seconds(&self) -> f64 {
        self.tick as f64 * self.us_per_tick() / 1_000_000.0
    }
}

// ── Sequencer Engine ─────────────────────────────────────────────

/// An output event generated by the sequencer.
#[derive(Debug, Clone, PartialEq)]
pub struct SequencerEvent {
    pub tick: u64,
    pub track_index: usize,
    pub channel: u8,
    pub note: u8,
    pub velocity: u8,
    pub duration_ticks: u32,
}

/// Render a pattern into a list of sequencer events.
pub fn render_pattern(pattern: &Pattern, ticks_per_step: u32, start_tick: u64) -> Vec<SequencerEvent> {
    let audible = pattern.audible_tracks();
    let mut events = Vec::new();

    for &track_idx in &audible {
        let track = &pattern.tracks[track_idx];
        for (step_idx, step) in track.steps.iter().enumerate() {
            if !step.active {
                continue;
            }
            // Probability check: include if probability >= 1.0 (deterministic for rendering)
            if step.probability < 1.0 {
                continue; // Skip probabilistic steps in deterministic render
            }
            let tick_offset = pattern.step_tick(step_idx, ticks_per_step) as u64;
            let gate_ticks = (ticks_per_step as f64 * step.gate_length) as u32;
            events.push(SequencerEvent {
                tick: start_tick + tick_offset,
                track_index: track_idx,
                channel: track.channel,
                note: step.note,
                velocity: step.velocity,
                duration_ticks: gate_ticks,
            });
        }
    }
    events.sort_by_key(|e| e.tick);
    events
}

/// Render an entire song arrangement into events.
pub fn render_song(song: &Song, ticks_per_step: u32) -> Vec<SequencerEvent> {
    let mut events = Vec::new();
    let mut tick_offset: u64 = 0;

    for entry in &song.arrangement {
        if let Some(pattern) = song.patterns.get(entry.pattern_index) {
            let pat_duration = pattern.num_steps as u64 * ticks_per_step as u64;
            for repeat in 0..entry.repeat_count {
                let start = tick_offset + repeat as u64 * pat_duration;
                events.extend(render_pattern(pattern, ticks_per_step, start));
            }
            tick_offset += entry.repeat_count as u64 * pat_duration;
        }
    }
    events.sort_by_key(|e| e.tick);
    events
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kick_pattern() -> Pattern {
        let mut pat = Pattern::new("Four on floor", 16);
        let kick = pat.add_track("Kick", 9);
        for i in (0..16).step_by(4) {
            pat.tracks[kick].set_step(i, 36, 127);
        }
        pat
    }

    #[test]
    fn test_step_default() {
        let s = Step::default();
        assert!(!s.active);
        assert_eq!(s.note, 60);
        assert!((s.probability - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_track_set_step() {
        let mut t = Track::new("Test", 8, 0);
        t.set_step(3, 48, 100);
        assert!(t.steps[3].active);
        assert_eq!(t.steps[3].note, 48);
        assert_eq!(t.steps[3].velocity, 100);
    }

    #[test]
    fn test_track_toggle() {
        let mut t = Track::new("Test", 4, 0);
        assert!(!t.steps[0].active);
        t.toggle_step(0);
        assert!(t.steps[0].active);
        t.toggle_step(0);
        assert!(!t.steps[0].active);
    }

    #[test]
    fn test_pattern_add_track() {
        let mut pat = Pattern::new("P1", 16);
        let idx = pat.add_track("Kick", 9);
        assert_eq!(idx, 0);
        assert_eq!(pat.tracks.len(), 1);
        assert_eq!(pat.tracks[0].steps.len(), 16);
    }

    #[test]
    fn test_pattern_remove_track() {
        let mut pat = Pattern::new("P1", 8);
        pat.add_track("A", 0);
        pat.add_track("B", 1);
        let removed = pat.remove_track(0).unwrap();
        assert_eq!(removed.name, "A");
        assert_eq!(pat.tracks.len(), 1);
    }

    #[test]
    fn test_swing_timing() {
        let mut pat = Pattern::new("Swing", 4);
        pat.set_swing(0.5);
        let tps = 120; // ticks per step
        // Even steps: no offset
        assert_eq!(pat.step_tick(0, tps), 0);
        // Odd steps: delayed by swing * 0.5 * tps = 0.5 * 0.5 * 120 = 30
        assert_eq!(pat.step_tick(1, tps), 120 + 30);
        assert_eq!(pat.step_tick(2, tps), 240);
        assert_eq!(pat.step_tick(3, tps), 360 + 30);
    }

    #[test]
    fn test_active_steps() {
        let pat = make_kick_pattern();
        let active = pat.active_steps(0, 120);
        assert_eq!(active.len(), 4); // steps 0, 4, 8, 12
        assert_eq!(active[0].0, 0);
        assert_eq!(active[1].0, 4);
        assert_eq!(active[2].0, 8);
        assert_eq!(active[3].0, 12);
    }

    #[test]
    fn test_mute_solo() {
        let mut pat = Pattern::new("P", 4);
        pat.add_track("A", 0);
        pat.add_track("B", 1);
        pat.add_track("C", 2);

        // All audible by default
        assert_eq!(pat.audible_tracks(), vec![0, 1, 2]);

        // Mute track B
        pat.tracks[1].muted = true;
        assert_eq!(pat.audible_tracks(), vec![0, 2]);

        // Solo track C (overrides mute)
        pat.tracks[2].solo = true;
        assert_eq!(pat.audible_tracks(), vec![2]);
    }

    #[test]
    fn test_transport_states() {
        let mut t = Transport::new(120.0, 480, 4);
        assert_eq!(t.state, TransportState::Stopped);

        t.play();
        assert_eq!(t.state, TransportState::Playing);

        t.pause();
        assert_eq!(t.state, TransportState::Paused);

        t.resume();
        assert_eq!(t.state, TransportState::Playing);

        t.stop();
        assert_eq!(t.state, TransportState::Stopped);
        assert_eq!(t.tick, 0);
    }

    #[test]
    fn test_transport_advance() {
        let mut t = Transport::new(120.0, 480, 4);
        t.play();
        assert!(t.advance(480));
        assert_eq!(t.tick, 480);
        assert_eq!(t.current_beat(), 1);
    }

    #[test]
    fn test_transport_no_advance_when_stopped() {
        let mut t = Transport::new(120.0, 480, 4);
        assert!(!t.advance(100));
        assert_eq!(t.tick, 0);
    }

    #[test]
    fn test_transport_seek() {
        let mut t = Transport::new(120.0, 480, 4);
        t.seek(960);
        assert_eq!(t.tick, 960);
        assert_eq!(t.current_beat(), 2);
    }

    #[test]
    fn test_transport_elapsed_seconds() {
        let t = Transport::new(120.0, 480, 4);
        // At 120 BPM, 1 beat = 0.5 sec, 480 ticks = 0.5 sec
        let us_per_tick = t.us_per_tick();
        let expected = 480.0 * us_per_tick / 1_000_000.0;
        assert!((expected - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_render_pattern() {
        let pat = make_kick_pattern();
        let events = render_pattern(&pat, 120, 0);
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].tick, 0);
        assert_eq!(events[0].note, 36);
        assert_eq!(events[1].tick, 480); // step 4 * 120 tps
    }

    #[test]
    fn test_song_arrangement() {
        let mut song = Song::new("Test Song");
        let pat = make_kick_pattern();
        let idx = song.add_pattern(pat);
        song.arrange(idx, 2);
        assert_eq!(song.total_plays(), 2);

        let flat = song.flatten_arrangement();
        assert_eq!(flat, vec![0, 0]);
    }

    #[test]
    fn test_song_total_ticks() {
        let mut song = Song::new("Test");
        let pat = Pattern::new("P", 16);
        let idx = song.add_pattern(pat);
        song.arrange(idx, 4);
        // 16 steps * 120 tps * 4 repeats = 7680
        assert_eq!(song.total_ticks(120), 7680);
    }

    #[test]
    fn test_render_song() {
        let mut song = Song::new("Test");
        let mut pat = Pattern::new("P", 4);
        let tk = pat.add_track("K", 0);
        pat.tracks[tk].set_step(0, 36, 100);
        let idx = song.add_pattern(pat);
        song.arrange(idx, 2);

        let events = render_song(&song, 120);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].tick, 0);
        assert_eq!(events[1].tick, 480); // second pattern starts at 4*120=480
    }

    #[test]
    fn test_gate_and_probability() {
        let mut track = Track::new("Test", 4, 0);
        track.set_step(0, 60, 100);
        track.set_gate(0, 0.75);
        track.set_probability(0, 0.5);
        assert!((track.steps[0].gate_length - 0.75).abs() < 1e-9);
        assert!((track.steps[0].probability - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pattern_swing_range_clamp() {
        let mut pat = Pattern::new("P", 4);
        pat.set_swing(1.5);
        assert!((pat.swing - 1.0).abs() < 1e-9);
        pat.set_swing(-0.3);
        assert!((pat.swing - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_transport_step_in_beat() {
        let mut t = Transport::new(120.0, 480, 4);
        t.play();
        t.advance(240); // half a beat = 2 steps into beat
        assert_eq!(t.current_step_in_beat(), 2);
    }
}
