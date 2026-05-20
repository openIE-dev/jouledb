//! Tempo and timing synchronization.
//!
//! BPM with sample-accurate timing, beat/bar tracking, tempo changes
//! (instant + ramp), time signatures, beat subdivision, sync callbacks,
//! tap tempo detection, MIDI clock generation (24 PPQ), and metronome
//! click generation. Pure Rust.

// ── Time Signature ───────────────────────────────────────────────

/// Time signature (e.g. 4/4, 3/4, 6/8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl TimeSignature {
    pub fn new(num: u8, denom: u8) -> Self {
        Self {
            numerator: num.max(1),
            denominator: if denom.is_power_of_two() && denom >= 1 { denom } else { 4 },
        }
    }

    /// Beats per bar for this time signature.
    pub fn beats_per_bar(&self) -> u8 {
        self.numerator
    }

    /// Quarter notes per bar.
    pub fn quarter_notes_per_bar(&self) -> f64 {
        self.numerator as f64 * 4.0 / self.denominator as f64
    }
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self { numerator: 4, denominator: 4 }
    }
}

// ── Beat Subdivision ─────────────────────────────────────────────

/// Musical subdivision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Subdivision {
    Whole,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    ThirtySecond,
    TripletQuarter,
    TripletEighth,
    TripletSixteenth,
    DottedQuarter,
    DottedEighth,
}

impl Subdivision {
    /// Duration in quarter notes.
    pub fn quarter_note_value(&self) -> f64 {
        match self {
            Subdivision::Whole => 4.0,
            Subdivision::Half => 2.0,
            Subdivision::Quarter => 1.0,
            Subdivision::Eighth => 0.5,
            Subdivision::Sixteenth => 0.25,
            Subdivision::ThirtySecond => 0.125,
            Subdivision::TripletQuarter => 2.0 / 3.0,
            Subdivision::TripletEighth => 1.0 / 3.0,
            Subdivision::TripletSixteenth => 1.0 / 6.0,
            Subdivision::DottedQuarter => 1.5,
            Subdivision::DottedEighth => 0.75,
        }
    }

    /// Duration in seconds at given BPM.
    pub fn seconds(&self, bpm: f64) -> f64 {
        self.quarter_note_value() * 60.0 / bpm
    }

    /// Duration in samples at given BPM and sample rate.
    pub fn samples(&self, bpm: f64, sample_rate: f64) -> f64 {
        self.seconds(bpm) * sample_rate
    }
}

// ── Tempo Change ─────────────────────────────────────────────────

/// Type of tempo change.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TempoChangeKind {
    Instant,
    Ramp { duration_beats: f64 },
}

/// A scheduled tempo change.
#[derive(Debug, Clone, PartialEq)]
pub struct TempoChange {
    pub beat: f64,
    pub target_bpm: f64,
    pub kind: TempoChangeKind,
}

// ── Tempo Clock ──────────────────────────────────────────────────

/// Sample-accurate tempo clock with beat/bar tracking.
#[derive(Debug, Clone)]
pub struct TempoClock {
    pub bpm: f64,
    pub time_signature: TimeSignature,
    pub sample_rate: f64,
    sample_position: u64,
    beat_position: f64,
    tempo_changes: Vec<TempoChange>,
    current_change_idx: usize,
}

impl TempoClock {
    pub fn new(bpm: f64, time_sig: TimeSignature, sample_rate: f64) -> Self {
        Self {
            bpm: bpm.max(1.0),
            time_signature: time_sig,
            sample_rate,
            sample_position: 0,
            beat_position: 0.0,
            tempo_changes: Vec::new(),
            current_change_idx: 0,
        }
    }

    /// Add a tempo change event.
    pub fn add_tempo_change(&mut self, change: TempoChange) {
        self.tempo_changes.push(change);
        self.tempo_changes.sort_by(|a, b| a.beat.partial_cmp(&b.beat).unwrap());
    }

    /// Current beat position (fractional).
    pub fn beat(&self) -> f64 {
        self.beat_position
    }

    /// Current bar number (0-indexed).
    pub fn bar(&self) -> u64 {
        let qn_per_bar = self.time_signature.quarter_notes_per_bar();
        if qn_per_bar < 1e-9 { return 0; }
        (self.beat_position / qn_per_bar).floor() as u64
    }

    /// Beat within current bar (0-indexed, fractional).
    pub fn beat_in_bar(&self) -> f64 {
        let qn_per_bar = self.time_signature.quarter_notes_per_bar();
        if qn_per_bar < 1e-9 { return 0.0; }
        self.beat_position % qn_per_bar
    }

    /// Current sample position.
    pub fn sample(&self) -> u64 {
        self.sample_position
    }

    /// Samples per beat at current BPM.
    pub fn samples_per_beat(&self) -> f64 {
        60.0 / self.bpm * self.sample_rate
    }

    /// Advance by one sample, returning sync events.
    pub fn advance_sample(&mut self) -> Vec<SyncEvent> {
        let prev_beat = self.beat_position;
        let beats_per_sample = self.bpm / 60.0 / self.sample_rate;

        // Check for tempo changes
        self.apply_tempo_changes();

        self.beat_position += beats_per_sample;
        self.sample_position += 1;

        let mut events = Vec::new();

        // Beat boundary crossed?
        if prev_beat.floor() < self.beat_position.floor() {
            events.push(SyncEvent::Beat(self.beat_position.floor() as u64));
        }

        // Bar boundary crossed?
        let qn_per_bar = self.time_signature.quarter_notes_per_bar();
        if qn_per_bar > 0.0 {
            let prev_bar = (prev_beat / qn_per_bar).floor() as u64;
            let curr_bar = (self.beat_position / qn_per_bar).floor() as u64;
            if prev_bar < curr_bar {
                events.push(SyncEvent::Bar(curr_bar));
            }
        }

        events
    }

    /// Advance by N samples, collecting sync events.
    pub fn advance_samples(&mut self, count: u64) -> Vec<SyncEvent> {
        let mut events = Vec::new();
        for _ in 0..count {
            events.extend(self.advance_sample());
        }
        events
    }

    /// Reset clock to beginning.
    pub fn reset(&mut self) {
        self.sample_position = 0;
        self.beat_position = 0.0;
        self.current_change_idx = 0;
    }

    /// Set BPM instantly.
    pub fn set_bpm(&mut self, bpm: f64) {
        self.bpm = bpm.max(1.0);
    }

    fn apply_tempo_changes(&mut self) {
        while self.current_change_idx < self.tempo_changes.len() {
            let change = &self.tempo_changes[self.current_change_idx];
            if self.beat_position >= change.beat {
                match change.kind {
                    TempoChangeKind::Instant => {
                        self.bpm = change.target_bpm;
                    }
                    TempoChangeKind::Ramp { duration_beats } => {
                        let elapsed = self.beat_position - change.beat;
                        if elapsed < duration_beats {
                            let prev_bpm = if self.current_change_idx > 0 {
                                self.tempo_changes[self.current_change_idx - 1].target_bpm
                            } else {
                                self.bpm
                            };
                            let t = elapsed / duration_beats;
                            self.bpm = prev_bpm + (change.target_bpm - prev_bpm) * t;
                            break;
                        } else {
                            self.bpm = change.target_bpm;
                        }
                    }
                }
                self.current_change_idx += 1;
            } else {
                break;
            }
        }
    }

    /// Elapsed wall-clock seconds.
    pub fn elapsed_seconds(&self) -> f64 {
        self.sample_position as f64 / self.sample_rate
    }
}

/// Sync events generated by the clock.
#[derive(Debug, Clone, PartialEq)]
pub enum SyncEvent {
    Beat(u64),
    Bar(u64),
}

// ── Tap Tempo ────────────────────────────────────────────────────

/// Tap tempo detector: averages intervals between taps to determine BPM.
#[derive(Debug, Clone)]
pub struct TapTempo {
    tap_times: Vec<f64>, // seconds
    max_taps: usize,
    timeout: f64, // seconds; if gap > timeout, reset
}

impl TapTempo {
    pub fn new(max_taps: usize, timeout_seconds: f64) -> Self {
        Self {
            tap_times: Vec::new(),
            max_taps: max_taps.max(2),
            timeout: timeout_seconds,
        }
    }

    /// Record a tap at the given time (seconds).
    pub fn tap(&mut self, time: f64) {
        // Reset if timeout exceeded
        if let Some(&last) = self.tap_times.last() {
            if time - last > self.timeout {
                self.tap_times.clear();
            }
        }

        self.tap_times.push(time);
        if self.tap_times.len() > self.max_taps {
            self.tap_times.remove(0);
        }
    }

    /// Get the detected BPM, or None if insufficient taps.
    pub fn bpm(&self) -> Option<f64> {
        if self.tap_times.len() < 2 {
            return None;
        }
        let intervals: Vec<f64> = self.tap_times
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let avg_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        if avg_interval < 1e-6 {
            return None;
        }
        Some(60.0 / avg_interval)
    }

    /// Reset the tap buffer.
    pub fn reset(&mut self) {
        self.tap_times.clear();
    }

    /// Number of taps recorded.
    pub fn tap_count(&self) -> usize {
        self.tap_times.len()
    }
}

// ── MIDI Clock Generation ────────────────────────────────────────

/// MIDI clock generator (24 PPQ standard).
#[derive(Debug, Clone)]
pub struct MidiClockGenerator {
    ppq: u32,
    sample_rate: f64,
    bpm: f64,
    sample_accumulator: f64,
    clock_count: u64,
}

impl MidiClockGenerator {
    pub fn new(bpm: f64, sample_rate: f64) -> Self {
        Self {
            ppq: 24,
            sample_rate,
            bpm,
            sample_accumulator: 0.0,
            clock_count: 0,
        }
    }

    /// Set BPM.
    pub fn set_bpm(&mut self, bpm: f64) {
        self.bpm = bpm.max(1.0);
    }

    /// Samples between MIDI clock ticks.
    pub fn samples_per_clock(&self) -> f64 {
        self.sample_rate * 60.0 / (self.bpm * self.ppq as f64)
    }

    /// Process one sample. Returns true if a MIDI clock should be sent.
    pub fn process_sample(&mut self) -> bool {
        self.sample_accumulator += 1.0;
        let spc = self.samples_per_clock();
        if self.sample_accumulator >= spc {
            self.sample_accumulator -= spc;
            self.clock_count += 1;
            true
        } else {
            false
        }
    }

    /// Process N samples. Returns the number of MIDI clocks generated.
    pub fn process_samples(&mut self, count: u64) -> u64 {
        let mut clocks = 0;
        for _ in 0..count {
            if self.process_sample() {
                clocks += 1;
            }
        }
        clocks
    }

    /// Total clock ticks generated.
    pub fn total_clocks(&self) -> u64 {
        self.clock_count
    }

    /// Reset.
    pub fn reset(&mut self) {
        self.sample_accumulator = 0.0;
        self.clock_count = 0;
    }
}

// ── Metronome Click ──────────────────────────────────────────────

/// Metronome click type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClickType {
    Downbeat,
    Beat,
    Subdivision,
}

/// A metronome click event.
#[derive(Debug, Clone, PartialEq)]
pub struct ClickEvent {
    pub sample: u64,
    pub click_type: ClickType,
    pub amplitude: f64,
}

/// Generate metronome click events for a given number of bars.
pub fn generate_clicks(
    bpm: f64,
    time_sig: TimeSignature,
    sample_rate: f64,
    num_bars: u32,
    subdivisions: u8,
) -> Vec<ClickEvent> {
    let mut events = Vec::new();
    let beats_per_bar = time_sig.beats_per_bar() as u32;
    let seconds_per_beat = 60.0 / bpm * (4.0 / time_sig.denominator as f64);
    let subs = subdivisions.max(1) as u32;

    for bar in 0..num_bars {
        for beat in 0..beats_per_bar {
            for sub in 0..subs {
                let beat_offset = bar * beats_per_bar + beat;
                let time = beat_offset as f64 * seconds_per_beat
                    + sub as f64 * seconds_per_beat / subs as f64;
                let sample = (time * sample_rate).round() as u64;

                let (click_type, amplitude) = if beat == 0 && sub == 0 {
                    (ClickType::Downbeat, 1.0)
                } else if sub == 0 {
                    (ClickType::Beat, 0.7)
                } else {
                    (ClickType::Subdivision, 0.4)
                };

                events.push(ClickEvent { sample, click_type, amplitude });
            }
        }
    }
    events
}

/// Generate a short click waveform (sine burst) as f64 samples.
pub fn click_waveform(frequency: f64, sample_rate: f64, duration_ms: f64) -> Vec<f64> {
    let num_samples = (duration_ms / 1000.0 * sample_rate).ceil() as usize;
    let mut buf = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate;
        let env = 1.0 - (i as f64 / num_samples as f64); // linear decay
        let sample = (2.0 * std::f64::consts::PI * frequency * t).sin() * env;
        buf.push(sample);
    }
    buf
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_sig_default() {
        let ts = TimeSignature::default();
        assert_eq!(ts.numerator, 4);
        assert_eq!(ts.denominator, 4);
        assert_eq!(ts.beats_per_bar(), 4);
    }

    #[test]
    fn test_time_sig_3_4() {
        let ts = TimeSignature::new(3, 4);
        assert_eq!(ts.beats_per_bar(), 3);
        assert!((ts.quarter_notes_per_bar() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_time_sig_6_8() {
        let ts = TimeSignature::new(6, 8);
        assert!((ts.quarter_notes_per_bar() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_subdivision_seconds() {
        let s = Subdivision::Quarter.seconds(120.0);
        assert!((s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_subdivision_eighth() {
        let s = Subdivision::Eighth.seconds(120.0);
        assert!((s - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_subdivision_samples() {
        let s = Subdivision::Quarter.samples(120.0, 44100.0);
        assert!((s - 22050.0).abs() < 1.0);
    }

    #[test]
    fn test_tempo_clock_beat_tracking() {
        let ts = TimeSignature::new(4, 4);
        let mut clock = TempoClock::new(120.0, ts, 44100.0);
        // Advance one beat worth of samples (22050 at 120 BPM / 44100 Hz)
        let samples_per_beat = 22050u64;
        let events = clock.advance_samples(samples_per_beat);
        let beat_events: Vec<_> = events.iter().filter(|e| matches!(e, SyncEvent::Beat(_))).collect();
        assert_eq!(beat_events.len(), 1);
    }

    #[test]
    fn test_tempo_clock_bar_tracking() {
        let ts = TimeSignature::new(4, 4);
        let mut clock = TempoClock::new(120.0, ts, 44100.0);
        // Advance 4 beats = 1 bar = 88200 samples
        let events = clock.advance_samples(88200);
        let bar_events: Vec<_> = events.iter().filter(|e| matches!(e, SyncEvent::Bar(_))).collect();
        assert_eq!(bar_events.len(), 1);
    }

    #[test]
    fn test_tempo_clock_reset() {
        let ts = TimeSignature::new(4, 4);
        let mut clock = TempoClock::new(120.0, ts, 44100.0);
        clock.advance_samples(10000);
        assert!(clock.beat() > 0.0);
        clock.reset();
        assert!((clock.beat() - 0.0).abs() < 1e-9);
        assert_eq!(clock.sample(), 0);
    }

    #[test]
    fn test_tempo_change_instant() {
        let ts = TimeSignature::new(4, 4);
        let mut clock = TempoClock::new(120.0, ts, 44100.0);
        clock.add_tempo_change(TempoChange {
            beat: 1.0,
            target_bpm: 140.0,
            kind: TempoChangeKind::Instant,
        });
        // Advance past beat 1
        clock.advance_samples(44100); // ~2 beats at 120 BPM
        assert!((clock.bpm - 140.0).abs() < 1e-6);
    }

    #[test]
    fn test_tap_tempo_basic() {
        let mut tap = TapTempo::new(8, 3.0);
        // 120 BPM = 0.5 seconds per beat
        tap.tap(0.0);
        tap.tap(0.5);
        tap.tap(1.0);
        tap.tap(1.5);
        let bpm = tap.bpm().unwrap();
        assert!((bpm - 120.0).abs() < 1.0);
    }

    #[test]
    fn test_tap_tempo_insufficient() {
        let mut tap = TapTempo::new(8, 3.0);
        tap.tap(0.0);
        assert!(tap.bpm().is_none());
    }

    #[test]
    fn test_tap_tempo_timeout() {
        let mut tap = TapTempo::new(8, 2.0);
        tap.tap(0.0);
        tap.tap(0.5);
        tap.tap(5.0); // 5 sec gap > 2 sec timeout -> reset
        assert_eq!(tap.tap_count(), 1);
    }

    #[test]
    fn test_midi_clock_24ppq() {
        let mut clk = MidiClockGenerator::new(120.0, 44100.0);
        // 1 beat = 22050 samples. 24 clocks per beat.
        let clocks = clk.process_samples(22050);
        assert_eq!(clocks, 24);
    }

    #[test]
    fn test_midi_clock_reset() {
        let mut clk = MidiClockGenerator::new(120.0, 44100.0);
        clk.process_samples(1000);
        clk.reset();
        assert_eq!(clk.total_clocks(), 0);
    }

    #[test]
    fn test_metronome_clicks_4_4() {
        let ts = TimeSignature::new(4, 4);
        let clicks = generate_clicks(120.0, ts, 44100.0, 2, 1);
        // 4 beats * 2 bars = 8 clicks
        assert_eq!(clicks.len(), 8);
        // First click is downbeat
        assert_eq!(clicks[0].click_type, ClickType::Downbeat);
        assert!((clicks[0].amplitude - 1.0).abs() < 1e-9);
        // Second click is beat
        assert_eq!(clicks[1].click_type, ClickType::Beat);
    }

    #[test]
    fn test_metronome_subdivisions() {
        let ts = TimeSignature::new(4, 4);
        let clicks = generate_clicks(120.0, ts, 44100.0, 1, 2);
        // 4 beats * 2 subs = 8 clicks per bar
        assert_eq!(clicks.len(), 8);
    }

    #[test]
    fn test_click_waveform() {
        let wave = click_waveform(1000.0, 44100.0, 10.0);
        assert!(!wave.is_empty());
        // First sample should be near 0 (sin(0))
        assert!(wave[0].abs() < 0.01);
        // Last sample should be near 0 (decayed)
        assert!(wave.last().unwrap().abs() < 0.1);
    }

    #[test]
    fn test_subdivision_triplet() {
        let s = Subdivision::TripletEighth.seconds(120.0);
        // 1/3 of a beat at 120 BPM = 1/3 * 0.5 = 0.1667 sec
        assert!((s - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_tempo_clock_elapsed() {
        let ts = TimeSignature::new(4, 4);
        let mut clock = TempoClock::new(120.0, ts, 44100.0);
        clock.advance_samples(44100);
        assert!((clock.elapsed_seconds() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_beat_in_bar() {
        let ts = TimeSignature::new(4, 4);
        let mut clock = TempoClock::new(120.0, ts, 44100.0);
        // Advance 5 beats (2.5 seconds at 120 BPM = 110250 samples)
        clock.advance_samples(110250);
        let bib = clock.beat_in_bar();
        assert!((bib - 1.0).abs() < 0.01); // 5th quarter note = beat 1 of bar 2
    }

    #[test]
    fn test_time_sig_invalid_denom() {
        let ts = TimeSignature::new(4, 3); // 3 is not power of 2, falls back to 4
        assert_eq!(ts.denominator, 4);
    }
}
