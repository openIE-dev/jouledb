//! Reverb effect processor using the Schroeder model.
//!
//! Algorithmic reverb with parallel comb filters, series allpass filters,
//! early reflections (tapped delay line), late reverb, stereo decorrelation,
//! and freeze mode.

// ── Comb Filter ────────────────────────────────────────────────

/// Feedback comb filter.
#[derive(Debug, Clone, PartialEq)]
pub struct CombFilter {
    buffer: Vec<f64>,
    write_pos: usize,
    feedback: f64,
    damping: f64,
    prev_output: f64,
}

impl CombFilter {
    /// Create a new comb filter with given delay in samples.
    pub fn new(delay_samples: usize, feedback: f64, damping: f64) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
            damping: damping.clamp(0.0, 1.0),
            prev_output: 0.0,
        }
    }

    /// Process one sample through the comb filter.
    pub fn process(&mut self, input: f64) -> f64 {
        let delayed = self.buffer[self.write_pos];
        // Low-pass filtered feedback (damping)
        let filtered = delayed * (1.0 - self.damping) + self.prev_output * self.damping;
        self.prev_output = filtered;
        let output = filtered;
        self.buffer[self.write_pos] = input + filtered * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
        self.prev_output = 0.0;
    }

    /// Set feedback amount.
    pub fn set_feedback(&mut self, feedback: f64) {
        self.feedback = feedback;
    }

    /// Set damping amount.
    pub fn set_damping(&mut self, damping: f64) {
        self.damping = damping.clamp(0.0, 1.0);
    }
}

// ── Allpass Filter ─────────────────────────────────────────────

/// Allpass filter for diffusion.
#[derive(Debug, Clone, PartialEq)]
pub struct AllpassFilter {
    buffer: Vec<f64>,
    write_pos: usize,
    feedback: f64,
}

impl AllpassFilter {
    /// Create a new allpass filter with given delay.
    pub fn new(delay_samples: usize, feedback: f64) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            feedback,
        }
    }

    /// Process one sample.
    pub fn process(&mut self, input: f64) -> f64 {
        let delayed = self.buffer[self.write_pos];
        let output = -input + delayed;
        self.buffer[self.write_pos] = input + delayed * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

// ── Tapped Delay Line ──────────────────────────────────────────

/// Tapped delay line for early reflections.
#[derive(Debug, Clone, PartialEq)]
pub struct TappedDelayLine {
    buffer: Vec<f64>,
    write_pos: usize,
    taps: Vec<(usize, f64)>, // (delay_in_samples, gain)
}

impl TappedDelayLine {
    /// Create with a maximum delay size.
    pub fn new(max_delay: usize) -> Self {
        Self {
            buffer: vec![0.0; max_delay.max(1)],
            write_pos: 0,
            taps: Vec::new(),
        }
    }

    /// Add a tap at a given delay (in samples) with a gain.
    pub fn add_tap(&mut self, delay: usize, gain: f64) {
        if delay < self.buffer.len() {
            self.taps.push((delay, gain));
        }
    }

    /// Process one sample, returning the sum of all taps.
    pub fn process(&mut self, input: f64) -> f64 {
        self.buffer[self.write_pos] = input;
        let buf_len = self.buffer.len();
        let mut output = 0.0;
        for &(delay, gain) in &self.taps {
            let read_pos = if self.write_pos >= delay {
                self.write_pos - delay
            } else {
                buf_len - (delay - self.write_pos)
            };
            output += self.buffer[read_pos] * gain;
        }
        self.write_pos = (self.write_pos + 1) % buf_len;
        output
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }

    /// Number of taps.
    pub fn tap_count(&self) -> usize {
        self.taps.len()
    }
}

// ── Reverb Parameters ──────────────────────────────────────────

/// Reverb engine parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ReverbParams {
    /// Room size (0.0-1.0), affects comb filter delays.
    pub room_size: f64,
    /// Damping (0.0-1.0), higher = less high-frequency content.
    pub damping: f64,
    /// Wet/dry mix (0.0 = all dry, 1.0 = all wet).
    pub wet_dry: f64,
    /// Pre-delay in milliseconds.
    pub pre_delay_ms: f64,
    /// Decay time in seconds.
    pub decay_time: f64,
    /// Number of comb filters.
    pub comb_count: usize,
    /// Number of allpass filters.
    pub allpass_count: usize,
    /// Freeze mode (infinite decay).
    pub freeze: bool,
    /// Stereo spread (0.0-1.0).
    pub stereo_spread: f64,
}

impl Default for ReverbParams {
    fn default() -> Self {
        Self {
            room_size: 0.5,
            damping: 0.5,
            wet_dry: 0.3,
            pre_delay_ms: 10.0,
            decay_time: 2.0,
            comb_count: 8,
            allpass_count: 4,
            freeze: false,
            stereo_spread: 0.5,
        }
    }
}

// ── Reverb Engine ──────────────────────────────────────────────

/// Default Schroeder comb filter delay times (in samples at 44100 Hz).
const COMB_DELAYS: [usize; 8] = [1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116];

/// Default allpass delay times.
const ALLPASS_DELAYS: [usize; 4] = [225, 556, 441, 341];

/// Stereo offset for right channel comb filters.
const STEREO_OFFSET: usize = 23;

/// Algorithmic reverb engine (Schroeder model).
#[derive(Debug, Clone)]
pub struct ReverbEngine {
    params: ReverbParams,
    sample_rate: u32,
    // Left channel
    combs_left: Vec<CombFilter>,
    allpasses_left: Vec<AllpassFilter>,
    // Right channel
    combs_right: Vec<CombFilter>,
    allpasses_right: Vec<AllpassFilter>,
    // Early reflections
    early_reflections: TappedDelayLine,
    // Pre-delay
    pre_delay_buffer: Vec<f64>,
    pre_delay_write_pos: usize,
    pre_delay_samples: usize,
}

impl ReverbEngine {
    /// Create a new reverb engine.
    pub fn new(sample_rate: u32, params: ReverbParams) -> Self {
        let feedback = compute_feedback(params.room_size, params.decay_time);
        let freeze_feedback = if params.freeze { 1.0 } else { feedback };

        let comb_count = params.comb_count.min(COMB_DELAYS.len());
        let allpass_count = params.allpass_count.min(ALLPASS_DELAYS.len());

        let scale = sample_rate as f64 / 44100.0;

        let combs_left: Vec<CombFilter> = (0..comb_count)
            .map(|i| {
                let delay = ((COMB_DELAYS[i] as f64 * scale * (0.8 + params.room_size * 0.4)) as usize).max(1);
                CombFilter::new(delay, freeze_feedback, params.damping)
            })
            .collect();

        let combs_right: Vec<CombFilter> = (0..comb_count)
            .map(|i| {
                let base = COMB_DELAYS[i] + (STEREO_OFFSET as f64 * params.stereo_spread) as usize;
                let delay = ((base as f64 * scale * (0.8 + params.room_size * 0.4)) as usize).max(1);
                CombFilter::new(delay, freeze_feedback, params.damping)
            })
            .collect();

        let allpasses_left: Vec<AllpassFilter> = (0..allpass_count)
            .map(|i| {
                let delay = ((ALLPASS_DELAYS[i] as f64 * scale) as usize).max(1);
                AllpassFilter::new(delay, 0.5)
            })
            .collect();

        let allpasses_right: Vec<AllpassFilter> = (0..allpass_count)
            .map(|i| {
                let delay = ((ALLPASS_DELAYS[i] as f64 * scale) as usize + STEREO_OFFSET / 2).max(1);
                AllpassFilter::new(delay, 0.5)
            })
            .collect();

        // Early reflections
        let max_er_delay = (0.05 * sample_rate as f64) as usize; // 50ms max
        let mut early_reflections = TappedDelayLine::new(max_er_delay.max(1));
        let er_taps = [
            (0.005, 0.8), (0.012, 0.6), (0.020, 0.5),
            (0.028, 0.4), (0.035, 0.3), (0.042, 0.25),
        ];
        for &(time, gain) in &er_taps {
            let samp = (time * sample_rate as f64) as usize;
            if samp < max_er_delay {
                early_reflections.add_tap(samp, gain);
            }
        }

        let pre_delay_samples = ((params.pre_delay_ms / 1000.0) * sample_rate as f64) as usize;
        let pre_delay_buffer = vec![0.0; pre_delay_samples.max(1)];

        Self {
            params,
            sample_rate,
            combs_left,
            allpasses_left,
            combs_right,
            allpasses_right,
            early_reflections,
            pre_delay_buffer,
            pre_delay_write_pos: 0,
            pre_delay_samples,
        }
    }

    /// Get current parameters.
    pub fn params(&self) -> &ReverbParams {
        &self.params
    }

    /// Set room size (rebuilds feedback).
    pub fn set_room_size(&mut self, room_size: f64) {
        self.params.room_size = room_size.clamp(0.0, 1.0);
        let fb = if self.params.freeze { 1.0 } else {
            compute_feedback(self.params.room_size, self.params.decay_time)
        };
        for c in &mut self.combs_left { c.set_feedback(fb); }
        for c in &mut self.combs_right { c.set_feedback(fb); }
    }

    /// Set damping.
    pub fn set_damping(&mut self, damping: f64) {
        self.params.damping = damping.clamp(0.0, 1.0);
        for c in &mut self.combs_left { c.set_damping(self.params.damping); }
        for c in &mut self.combs_right { c.set_damping(self.params.damping); }
    }

    /// Set wet/dry mix.
    pub fn set_wet_dry(&mut self, mix: f64) {
        self.params.wet_dry = mix.clamp(0.0, 1.0);
    }

    /// Enable/disable freeze mode.
    pub fn set_freeze(&mut self, freeze: bool) {
        self.params.freeze = freeze;
        let fb = if freeze { 1.0 } else {
            compute_feedback(self.params.room_size, self.params.decay_time)
        };
        for c in &mut self.combs_left { c.set_feedback(fb); }
        for c in &mut self.combs_right { c.set_feedback(fb); }
    }

    /// Process a mono input buffer, returning stereo output (left, right).
    pub fn process_mono(&mut self, input: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let len = input.len();
        let mut left_out = vec![0.0f64; len];
        let mut right_out = vec![0.0f64; len];

        let wet = self.params.wet_dry;
        let dry = 1.0 - wet;

        for i in 0..len {
            let sample = input[i];

            // Pre-delay
            let pre_delayed = if self.pre_delay_samples > 0 {
                let out = self.pre_delay_buffer[self.pre_delay_write_pos];
                self.pre_delay_buffer[self.pre_delay_write_pos] = sample;
                self.pre_delay_write_pos = (self.pre_delay_write_pos + 1) % self.pre_delay_samples;
                out
            } else {
                sample
            };

            // Early reflections
            let early = self.early_reflections.process(pre_delayed);

            // Parallel comb filters (sum for left channel)
            let mut comb_sum_l = 0.0;
            for comb in &mut self.combs_left {
                comb_sum_l += comb.process(pre_delayed);
            }

            // Parallel comb filters (sum for right channel)
            let mut comb_sum_r = 0.0;
            for comb in &mut self.combs_right {
                comb_sum_r += comb.process(pre_delayed);
            }

            // Normalize by comb count
            let comb_count = self.combs_left.len().max(1) as f64;
            comb_sum_l /= comb_count;
            comb_sum_r /= comb_count;

            // Series allpass filters
            let mut ap_l = comb_sum_l;
            for ap in &mut self.allpasses_left {
                ap_l = ap.process(ap_l);
            }

            let mut ap_r = comb_sum_r;
            for ap in &mut self.allpasses_right {
                ap_r = ap.process(ap_r);
            }

            // Mix early + late reverb
            let reverb_l = early * 0.5 + ap_l;
            let reverb_r = early * 0.5 + ap_r;

            // Wet/dry mix
            left_out[i] = sample * dry + reverb_l * wet;
            right_out[i] = sample * dry + reverb_r * wet;
        }

        (left_out, right_out)
    }

    /// Process stereo input, returning stereo output.
    pub fn process_stereo(&mut self, left: &[f64], right: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let len = left.len().min(right.len());
        let mono: Vec<f64> = (0..len).map(|i| (left[i] + right[i]) * 0.5).collect();
        self.process_mono(&mono)
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        for c in &mut self.combs_left { c.reset(); }
        for c in &mut self.combs_right { c.reset(); }
        for a in &mut self.allpasses_left { a.reset(); }
        for a in &mut self.allpasses_right { a.reset(); }
        self.early_reflections.reset();
        self.pre_delay_buffer.fill(0.0);
        self.pre_delay_write_pos = 0;
    }
}

/// Compute feedback coefficient from room size and decay time.
fn compute_feedback(room_size: f64, decay_time: f64) -> f64 {
    let base = 0.7 + 0.28 * room_size.clamp(0.0, 1.0);
    let decay_factor = decay_time.clamp(0.1, 20.0) / 2.0;
    (base * decay_factor).clamp(0.0, 0.99)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comb_filter_basic() {
        let mut comb = CombFilter::new(4, 0.5, 0.0);
        let out = comb.process(1.0);
        // First output is from empty buffer
        assert!(out.abs() < 1e-10);
    }

    #[test]
    fn test_comb_filter_delay() {
        let mut comb = CombFilter::new(4, 0.0, 0.0);
        comb.process(1.0);
        comb.process(0.0);
        comb.process(0.0);
        comb.process(0.0);
        let out = comb.process(0.0);
        // After 4 samples delay, the original 1.0 comes back (no feedback)
        assert!((out - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_comb_filter_feedback() {
        let mut comb = CombFilter::new(2, 0.5, 0.0);
        comb.process(1.0);
        comb.process(0.0);
        let out = comb.process(0.0);
        // 1.0 appears after 2 samples, feedback makes next echo at 4 samples
        assert!((out - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_comb_filter_reset() {
        let mut comb = CombFilter::new(4, 0.5, 0.0);
        comb.process(1.0);
        comb.reset();
        let out = comb.process(0.0);
        assert!(out.abs() < 1e-10);
    }

    #[test]
    fn test_allpass_filter() {
        let mut ap = AllpassFilter::new(4, 0.5);
        // Allpass should produce output: feed an impulse and collect output
        let mut outputs = Vec::new();
        for i in 0..20 {
            let input = if i == 0 { 1.0 } else { 0.0 };
            outputs.push(ap.process(input));
        }
        // First sample should have the negative pass-through component
        assert!((outputs[0] - (-1.0)).abs() < 1e-10);
        // Later samples should have decaying energy from feedback
        let tail_energy: f64 = outputs[4..].iter().map(|s| s * s).sum();
        assert!(tail_energy > 0.01);
    }

    #[test]
    fn test_allpass_reset() {
        let mut ap = AllpassFilter::new(4, 0.5);
        ap.process(1.0);
        ap.reset();
        // Should be silent after reset
        let out = ap.process(0.0);
        assert!(out.abs() < 1e-10);
    }

    #[test]
    fn test_tapped_delay_line() {
        let mut tdl = TappedDelayLine::new(100);
        tdl.add_tap(5, 1.0);
        for _ in 0..5 {
            tdl.process(0.0);
        }
        // Now write a 1.0
        tdl.process(1.0);
        // It should appear at tap 5 samples later
        let mut found = false;
        for _ in 0..10 {
            let out = tdl.process(0.0);
            if (out - 1.0).abs() < 1e-10 {
                found = true;
                break;
            }
        }
        assert!(found);
    }

    #[test]
    fn test_tapped_delay_line_multi_tap() {
        let mut tdl = TappedDelayLine::new(100);
        tdl.add_tap(0, 0.5);
        tdl.add_tap(2, 0.3);
        assert_eq!(tdl.tap_count(), 2);
        let out = tdl.process(1.0);
        // Tap at 0 should return 0.5 * 1.0 immediately
        assert!((out - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_reverb_params_default() {
        let p = ReverbParams::default();
        assert!((p.room_size - 0.5).abs() < 1e-10);
        assert_eq!(p.comb_count, 8);
        assert_eq!(p.allpass_count, 4);
    }

    #[test]
    fn test_reverb_engine_create() {
        let engine = ReverbEngine::new(44100, ReverbParams::default());
        assert_eq!(engine.combs_left.len(), 8);
        assert_eq!(engine.allpasses_left.len(), 4);
    }

    #[test]
    fn test_reverb_process_silence() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        let input = vec![0.0; 256];
        let (left, right) = engine.process_mono(&input);
        assert_eq!(left.len(), 256);
        assert_eq!(right.len(), 256);
        // All silence in → all silence out
        assert!(left.iter().all(|s| s.abs() < 1e-10));
        assert!(right.iter().all(|s| s.abs() < 1e-10));
    }

    #[test]
    fn test_reverb_process_impulse() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        let mut input = vec![0.0; 4096];
        input[0] = 1.0;
        let (left, right) = engine.process_mono(&input);
        // Should have reverb tail
        let tail_energy: f64 = left[256..].iter().chain(right[256..].iter())
            .map(|s| s * s).sum();
        assert!(tail_energy > 1e-6);
    }

    #[test]
    fn test_reverb_wet_dry() {
        let mut params = ReverbParams::default();
        params.wet_dry = 0.0; // All dry
        let mut engine = ReverbEngine::new(44100, params);
        let input = vec![0.5; 64];
        let (left, _right) = engine.process_mono(&input);
        // Dry-only: output ≈ input after pre-delay settles
        let last = left[left.len() - 1];
        assert!((last - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_reverb_freeze() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        engine.set_freeze(true);
        assert!(engine.params().freeze);
        // Freeze = feedback 1.0
        for c in &engine.combs_left {
            assert!((c.feedback - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_reverb_set_room_size() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        engine.set_room_size(0.8);
        assert!((engine.params().room_size - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_reverb_set_damping() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        engine.set_damping(0.7);
        assert!((engine.params().damping - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_reverb_set_wet_dry() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        engine.set_wet_dry(0.6);
        assert!((engine.params().wet_dry - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_reverb_process_stereo() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        let left = vec![0.5; 64];
        let right = vec![0.3; 64];
        let (out_l, out_r) = engine.process_stereo(&left, &right);
        assert_eq!(out_l.len(), 64);
        assert_eq!(out_r.len(), 64);
    }

    #[test]
    fn test_reverb_reset() {
        let mut engine = ReverbEngine::new(44100, ReverbParams::default());
        let mut input = vec![0.0; 256];
        input[0] = 1.0;
        engine.process_mono(&input);
        engine.reset();
        let silence = vec![0.0; 256];
        let (left, right) = engine.process_mono(&silence);
        assert!(left.iter().all(|s| s.abs() < 1e-10));
        assert!(right.iter().all(|s| s.abs() < 1e-10));
    }

    #[test]
    fn test_compute_feedback() {
        let fb = compute_feedback(0.5, 2.0);
        assert!(fb > 0.0 && fb < 1.0);
    }

    #[test]
    fn test_compute_feedback_large_room() {
        let fb_small = compute_feedback(0.1, 2.0);
        let fb_large = compute_feedback(0.9, 2.0);
        assert!(fb_large > fb_small);
    }

    #[test]
    fn test_stereo_decorrelation() {
        let mut params = ReverbParams::default();
        params.stereo_spread = 1.0; // Maximum stereo spread
        let mut engine = ReverbEngine::new(44100, params);
        let mut input = vec![0.0; 8192];
        input[0] = 1.0;
        let (left, right) = engine.process_mono(&input);
        // Left and right should differ (stereo decorrelation from offset comb delays)
        let mut any_diff = false;
        for i in 0..left.len().min(right.len()) {
            if (left[i] - right[i]).abs() > 1e-12 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff);
    }

    #[test]
    fn test_comb_damping() {
        let mut comb_no_damp = CombFilter::new(4, 0.5, 0.0);
        let mut comb_damp = CombFilter::new(4, 0.5, 0.9);
        comb_no_damp.process(1.0);
        comb_damp.process(1.0);
        for _ in 0..3 {
            comb_no_damp.process(0.0);
            comb_damp.process(0.0);
        }
        let out_no = comb_no_damp.process(0.0);
        let out_d = comb_damp.process(0.0);
        // Damped should attenuate more
        assert!(out_d.abs() <= out_no.abs() + 1e-6);
    }
}
