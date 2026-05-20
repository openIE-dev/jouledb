//! Audio delay effects — simple delay, feedback delay, ping-pong, chorus, flanger.
//!
//! All delay effects use circular buffer delay lines with configurable delay time,
//! feedback, wet/dry mix, and modulation. Supports tap delay for multi-tap effects.

// ── Delay Line (core) ──────────────────────────────────────────

/// A circular buffer delay line.
#[derive(Debug, Clone)]
pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    max_delay_samples: usize,
}

impl DelayLine {
    /// Create a delay line with the given maximum delay in samples.
    pub fn new(max_delay_samples: usize) -> Self {
        let size = max_delay_samples.max(1);
        Self {
            buffer: vec![0.0; size],
            write_pos: 0,
            max_delay_samples: size,
        }
    }

    /// Create a delay line from maximum delay time in seconds.
    pub fn from_time(max_delay_secs: f64, sample_rate: f64) -> Self {
        let samples = (max_delay_secs * sample_rate) as usize;
        Self::new(samples)
    }

    pub fn max_delay(&self) -> usize {
        self.max_delay_samples
    }

    /// Write a sample into the delay line.
    pub fn write(&mut self, sample: f32) {
        self.buffer[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.max_delay_samples;
    }

    /// Read a sample from `delay` samples ago.
    pub fn read(&self, delay: usize) -> f32 {
        let delay_clamped = delay.min(self.max_delay_samples - 1);
        let read_pos =
            (self.write_pos + self.max_delay_samples - delay_clamped - 1) % self.max_delay_samples;
        self.buffer[read_pos]
    }

    /// Read with fractional delay using linear interpolation.
    pub fn read_interpolated(&self, delay: f64) -> f32 {
        let max = (self.max_delay_samples - 1) as f64;
        let delay_clamped = delay.clamp(0.0, max);
        let delay_floor = delay_clamped.floor() as usize;
        let frac = delay_clamped - delay_clamped.floor();

        let s0 = self.read(delay_floor);
        let s1 = self.read(delay_floor + 1);
        s0 + (s1 - s0) * frac as f32
    }

    /// Clear the delay buffer.
    pub fn clear(&mut self) {
        for s in &mut self.buffer {
            *s = 0.0;
        }
    }
}

// ── Simple Delay ────────────────────────────────────────────────

/// Simple delay effect with wet/dry mix.
#[derive(Debug, Clone)]
pub struct SimpleDelay {
    delay_line: DelayLine,
    delay_samples: usize,
    wet: f32,
    dry: f32,
}

impl SimpleDelay {
    /// Create a simple delay.
    pub fn new(delay_samples: usize, max_delay: usize, wet: f32, dry: f32) -> Self {
        Self {
            delay_line: DelayLine::new(max_delay),
            delay_samples: delay_samples.min(max_delay),
            wet,
            dry,
        }
    }

    /// Create from time values.
    pub fn from_time(delay_secs: f64, max_delay_secs: f64, sample_rate: f64) -> Self {
        let delay_samples = (delay_secs * sample_rate) as usize;
        let max_samples = (max_delay_secs * sample_rate) as usize;
        Self::new(delay_samples, max_samples, 0.5, 0.5)
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    pub fn set_delay_samples(&mut self, samples: usize) {
        self.delay_samples = samples.min(self.delay_line.max_delay() - 1);
    }

    /// Process a single sample.
    pub fn tick(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(self.delay_samples);
        self.delay_line.write(input);
        input * self.dry + delayed * self.wet
    }

    /// Process a buffer in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

// ── Feedback Delay ──────────────────────────────────────────────

/// Delay with feedback (echo effect).
#[derive(Debug, Clone)]
pub struct FeedbackDelay {
    delay_line: DelayLine,
    delay_samples: usize,
    feedback: f32,
    wet: f32,
    dry: f32,
}

impl FeedbackDelay {
    pub fn new(delay_samples: usize, max_delay: usize, feedback: f32) -> Self {
        Self {
            delay_line: DelayLine::new(max_delay),
            delay_samples: delay_samples.min(max_delay.saturating_sub(1)),
            feedback: feedback.clamp(-0.99, 0.99),
            wet: 0.5,
            dry: 0.5,
        }
    }

    /// Create from time values.
    pub fn from_time(delay_secs: f64, max_delay_secs: f64, feedback: f32, sample_rate: f64) -> Self {
        let delay_samples = (delay_secs * sample_rate) as usize;
        let max_samples = (max_delay_secs * sample_rate) as usize;
        Self::new(delay_samples, max_samples, feedback)
    }

    pub fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(-0.99, 0.99);
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    pub fn set_delay_samples(&mut self, samples: usize) {
        self.delay_samples = samples.min(self.delay_line.max_delay() - 1);
    }

    /// Process a single sample.
    pub fn tick(&mut self, input: f32) -> f32 {
        let delayed = self.delay_line.read(self.delay_samples);
        let write_val = input + delayed * self.feedback;
        self.delay_line.write(write_val);
        input * self.dry + delayed * self.wet
    }

    /// Process a buffer in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

// ── Ping-Pong Delay ────────────────────────────────────────────

/// Stereo ping-pong delay that bounces between left and right channels.
#[derive(Debug, Clone)]
pub struct PingPongDelay {
    left_delay: DelayLine,
    right_delay: DelayLine,
    delay_samples: usize,
    feedback: f32,
    wet: f32,
    dry: f32,
}

impl PingPongDelay {
    pub fn new(delay_samples: usize, max_delay: usize, feedback: f32) -> Self {
        Self {
            left_delay: DelayLine::new(max_delay),
            right_delay: DelayLine::new(max_delay),
            delay_samples: delay_samples.min(max_delay.saturating_sub(1)),
            feedback: feedback.clamp(-0.99, 0.99),
            wet: 0.5,
            dry: 0.5,
        }
    }

    pub fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(-0.99, 0.99);
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    /// Process a stereo sample pair (left_in, right_in) -> (left_out, right_out).
    pub fn tick(&mut self, left_in: f32, right_in: f32) -> (f32, f32) {
        let left_delayed = self.left_delay.read(self.delay_samples);
        let right_delayed = self.right_delay.read(self.delay_samples);

        // Cross-feed: left delay feeds right, right feeds left
        self.left_delay
            .write(left_in + right_delayed * self.feedback);
        self.right_delay
            .write(right_in + left_delayed * self.feedback);

        let left_out = left_in * self.dry + left_delayed * self.wet;
        let right_out = right_in * self.dry + right_delayed * self.wet;

        (left_out, right_out)
    }

    /// Process interleaved stereo buffer in place (L0 R0 L1 R1 ...).
    pub fn process_interleaved(&mut self, samples: &mut [f32]) {
        let frames = samples.len() / 2;
        for f in 0..frames {
            let l = samples[f * 2];
            let r = samples[f * 2 + 1];
            let (lo, ro) = self.tick(l, r);
            samples[f * 2] = lo;
            samples[f * 2 + 1] = ro;
        }
    }

    pub fn clear(&mut self) {
        self.left_delay.clear();
        self.right_delay.clear();
    }
}

// ── Chorus Effect ───────────────────────────────────────────────

/// Chorus effect using modulated delay.
#[derive(Debug, Clone)]
pub struct ChorusEffect {
    delay_line: DelayLine,
    base_delay_samples: f64,
    depth_samples: f64,
    rate_hz: f64,
    phase: f64,
    sample_rate: f64,
    wet: f32,
    dry: f32,
}

impl ChorusEffect {
    /// Create a chorus effect.
    /// `base_delay_ms`: center delay (typically 7-20 ms).
    /// `depth_ms`: modulation depth (typically 1-5 ms).
    /// `rate_hz`: modulation rate (typically 0.5-3 Hz).
    pub fn new(base_delay_ms: f64, depth_ms: f64, rate_hz: f64, sample_rate: f64) -> Self {
        let base_samples = base_delay_ms * 0.001 * sample_rate;
        let depth_samples = depth_ms * 0.001 * sample_rate;
        let max_delay = (base_samples + depth_samples + 10.0) as usize;
        Self {
            delay_line: DelayLine::new(max_delay),
            base_delay_samples: base_samples,
            depth_samples,
            rate_hz,
            phase: 0.0,
            sample_rate,
            wet: 0.5,
            dry: 0.5,
        }
    }

    pub fn set_rate(&mut self, rate_hz: f64) {
        self.rate_hz = rate_hz;
    }

    pub fn set_depth_ms(&mut self, depth_ms: f64) {
        self.depth_samples = depth_ms * 0.001 * self.sample_rate;
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    /// Process a single sample.
    pub fn tick(&mut self, input: f32) -> f32 {
        let lfo = (self.phase * 2.0 * std::f64::consts::PI).sin();
        let delay = self.base_delay_samples + lfo * self.depth_samples;
        let delayed = self.delay_line.read_interpolated(delay);
        self.delay_line.write(input);

        self.phase += self.rate_hz / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        input * self.dry + delayed * self.wet
    }

    /// Process a buffer in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
        self.phase = 0.0;
    }
}

// ── Flanger Effect ──────────────────────────────────────────────

/// Flanger effect — very short modulated delay with feedback.
#[derive(Debug, Clone)]
pub struct FlangerEffect {
    delay_line: DelayLine,
    base_delay_samples: f64,
    depth_samples: f64,
    rate_hz: f64,
    feedback: f32,
    phase: f64,
    sample_rate: f64,
    wet: f32,
    dry: f32,
}

impl FlangerEffect {
    /// Create a flanger effect.
    /// `base_delay_ms`: center delay (typically 1-5 ms).
    /// `depth_ms`: modulation depth (typically 0.5-2 ms).
    /// `rate_hz`: modulation rate (typically 0.1-1 Hz).
    pub fn new(
        base_delay_ms: f64,
        depth_ms: f64,
        rate_hz: f64,
        feedback: f32,
        sample_rate: f64,
    ) -> Self {
        let base_samples = base_delay_ms * 0.001 * sample_rate;
        let depth_samples = depth_ms * 0.001 * sample_rate;
        let max_delay = (base_samples + depth_samples + 10.0) as usize;
        Self {
            delay_line: DelayLine::new(max_delay),
            base_delay_samples: base_samples,
            depth_samples,
            rate_hz,
            feedback: feedback.clamp(-0.99, 0.99),
            phase: 0.0,
            sample_rate,
            wet: 0.5,
            dry: 0.5,
        }
    }

    pub fn set_rate(&mut self, rate_hz: f64) {
        self.rate_hz = rate_hz;
    }

    pub fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(-0.99, 0.99);
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    /// Process a single sample.
    pub fn tick(&mut self, input: f32) -> f32 {
        let lfo = (self.phase * 2.0 * std::f64::consts::PI).sin();
        let delay = self.base_delay_samples + lfo * self.depth_samples;
        let delayed = self.delay_line.read_interpolated(delay);

        self.delay_line.write(input + delayed * self.feedback);

        self.phase += self.rate_hz / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }

        input * self.dry + delayed * self.wet
    }

    /// Process a buffer in place.
    pub fn process(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick(*s);
        }
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
        self.phase = 0.0;
    }
}

// ── Tap Delay ───────────────────────────────────────────────────

/// A single tap in a multi-tap delay.
#[derive(Debug, Clone)]
pub struct DelayTap {
    pub delay_samples: usize,
    pub gain: f32,
    pub pan: f32, // -1 = left, 0 = center, 1 = right
}

/// Multi-tap delay with configurable taps.
#[derive(Debug, Clone)]
pub struct TapDelay {
    delay_line: DelayLine,
    taps: Vec<DelayTap>,
    feedback: f32,
    feedback_tap: usize,
    wet: f32,
    dry: f32,
}

impl TapDelay {
    pub fn new(max_delay_samples: usize) -> Self {
        Self {
            delay_line: DelayLine::new(max_delay_samples),
            taps: Vec::new(),
            feedback: 0.0,
            feedback_tap: 0,
            wet: 0.5,
            dry: 0.5,
        }
    }

    /// Add a tap.
    pub fn add_tap(&mut self, delay_samples: usize, gain: f32, pan: f32) -> usize {
        self.taps.push(DelayTap {
            delay_samples: delay_samples.min(self.delay_line.max_delay() - 1),
            gain,
            pan: pan.clamp(-1.0, 1.0),
        });
        self.taps.len() - 1
    }

    pub fn tap_count(&self) -> usize {
        self.taps.len()
    }

    pub fn set_feedback(&mut self, feedback: f32, tap_index: usize) {
        self.feedback = feedback.clamp(-0.99, 0.99);
        self.feedback_tap = tap_index;
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    /// Process a mono input sample, returning a mono output (sum of all taps).
    pub fn tick_mono(&mut self, input: f32) -> f32 {
        let fb_sample = if self.feedback_tap < self.taps.len() {
            let tap = &self.taps[self.feedback_tap];
            self.delay_line.read(tap.delay_samples) * tap.gain
        } else {
            0.0
        };

        self.delay_line.write(input + fb_sample * self.feedback);

        let mut wet_sum = 0.0f32;
        for tap in &self.taps {
            wet_sum += self.delay_line.read(tap.delay_samples) * tap.gain;
        }

        input * self.dry + wet_sum * self.wet
    }

    /// Process a mono input, returning stereo (L, R) using tap pan positions.
    pub fn tick_stereo(&mut self, input: f32) -> (f32, f32) {
        let fb_sample = if self.feedback_tap < self.taps.len() {
            let tap = &self.taps[self.feedback_tap];
            self.delay_line.read(tap.delay_samples) * tap.gain
        } else {
            0.0
        };

        self.delay_line.write(input + fb_sample * self.feedback);

        let mut left = 0.0f32;
        let mut right = 0.0f32;
        for tap in &self.taps {
            let sample = self.delay_line.read(tap.delay_samples) * tap.gain;
            let pan_l = ((1.0 - tap.pan) * 0.5).sqrt();
            let pan_r = ((1.0 + tap.pan) * 0.5).sqrt();
            left += sample * pan_l;
            right += sample * pan_r;
        }

        let dry_half = input * self.dry;
        (dry_half + left * self.wet, dry_half + right * self.wet)
    }

    /// Process a buffer in place (mono).
    pub fn process_mono(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.tick_mono(*s);
        }
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

// ── Modulated Delay ─────────────────────────────────────────────

/// Delay with external modulation input for delay time.
#[derive(Debug, Clone)]
pub struct ModulatedDelay {
    delay_line: DelayLine,
    base_delay_samples: f64,
    mod_depth_samples: f64,
    feedback: f32,
    wet: f32,
    dry: f32,
}

impl ModulatedDelay {
    pub fn new(
        base_delay_ms: f64,
        mod_depth_ms: f64,
        max_delay_ms: f64,
        feedback: f32,
        sample_rate: f64,
    ) -> Self {
        let base = base_delay_ms * 0.001 * sample_rate;
        let depth = mod_depth_ms * 0.001 * sample_rate;
        let max = (max_delay_ms * 0.001 * sample_rate) as usize;
        Self {
            delay_line: DelayLine::new(max),
            base_delay_samples: base,
            mod_depth_samples: depth,
            feedback: feedback.clamp(-0.99, 0.99),
            wet: 0.5,
            dry: 0.5,
        }
    }

    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet;
    }

    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry;
    }

    pub fn set_feedback(&mut self, fb: f32) {
        self.feedback = fb.clamp(-0.99, 0.99);
    }

    /// Process a sample with external modulation signal (expected range -1..1).
    pub fn tick(&mut self, input: f32, modulation: f32) -> f32 {
        let delay = self.base_delay_samples + modulation as f64 * self.mod_depth_samples;
        let delayed = self.delay_line.read_interpolated(delay.max(0.0));
        self.delay_line.write(input + delayed * self.feedback);
        input * self.dry + delayed * self.wet
    }

    /// Process a buffer with a parallel modulation buffer.
    pub fn process(&mut self, samples: &mut [f32], modulation: &[f32]) {
        let len = samples.len().min(modulation.len());
        for i in 0..len {
            samples[i] = self.tick(samples[i], modulation[i]);
        }
    }

    pub fn clear(&mut self) {
        self.delay_line.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_line_read_write() {
        let mut dl = DelayLine::new(10);
        dl.write(1.0);
        dl.write(2.0);
        dl.write(3.0);
        assert!((dl.read(0) - 3.0).abs() < 1e-6);
        assert!((dl.read(1) - 2.0).abs() < 1e-6);
        assert!((dl.read(2) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn delay_line_circular() {
        let mut dl = DelayLine::new(4);
        for i in 0..8 {
            dl.write(i as f32);
        }
        // Last 4 values: 4, 5, 6, 7
        assert!((dl.read(0) - 7.0).abs() < 1e-6);
        assert!((dl.read(1) - 6.0).abs() < 1e-6);
        assert!((dl.read(3) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn delay_line_interpolated() {
        let mut dl = DelayLine::new(10);
        dl.write(0.0);
        dl.write(1.0);
        let val = dl.read_interpolated(0.5);
        assert!((val - 0.5).abs() < 0.01, "interpolated = {val}");
    }

    #[test]
    fn delay_line_clear() {
        let mut dl = DelayLine::new(10);
        dl.write(5.0);
        dl.clear();
        assert!((dl.read(0)).abs() < 1e-6);
    }

    #[test]
    fn simple_delay_passthrough() {
        let mut delay = SimpleDelay::new(0, 100, 0.0, 1.0);
        assert!((delay.tick(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn simple_delay_produces_echo() {
        let mut delay = SimpleDelay::new(4, 100, 1.0, 0.0);
        // Write 4 samples
        for _ in 0..4 {
            delay.tick(1.0);
        }
        // Next read should have the delayed signal
        // Actually sample at delay=4 means we need to write 5 samples before echo
        let mut found_echo = false;
        for _ in 0..10 {
            let out = delay.tick(0.0);
            if out > 0.5 {
                found_echo = true;
                break;
            }
        }
        assert!(found_echo, "simple delay should produce echo");
    }

    #[test]
    fn feedback_delay_decays() {
        let mut delay = FeedbackDelay::new(10, 1000, 0.5);
        delay.set_wet(1.0);
        delay.set_dry(0.0);
        // Send impulse
        delay.tick(1.0);
        for _ in 0..9 {
            delay.tick(0.0);
        }
        let first_echo = delay.tick(0.0);
        for _ in 0..9 {
            delay.tick(0.0);
        }
        let second_echo = delay.tick(0.0);
        // Second echo should be quieter (feedback = 0.5)
        assert!(
            second_echo.abs() < first_echo.abs() + 0.01,
            "echoes should decay: first={first_echo}, second={second_echo}"
        );
    }

    #[test]
    fn feedback_delay_feedback_clamp() {
        let mut delay = FeedbackDelay::new(10, 100, 2.0);
        // Feedback should be clamped to 0.99
        delay.set_feedback(2.0);
        // Should not explode
        delay.tick(1.0);
        for _ in 0..100 {
            let val = delay.tick(0.0);
            assert!(val.abs() < 100.0, "feedback should not explode");
        }
    }

    #[test]
    fn ping_pong_stereo() {
        let mut pp = PingPongDelay::new(10, 1000, 0.5);
        pp.set_wet(1.0);
        pp.set_dry(0.0);
        // Send impulse to left only
        pp.tick(1.0, 0.0);
        for _ in 0..10 {
            pp.tick(0.0, 0.0);
        }
        let (l, r) = pp.tick(0.0, 0.0);
        // Should have signal in at least one channel
        assert!(
            l.abs() > 0.01 || r.abs() > 0.01,
            "ping-pong should produce output: l={l}, r={r}"
        );
    }

    #[test]
    fn ping_pong_clear() {
        let mut pp = PingPongDelay::new(10, 100, 0.5);
        pp.tick(1.0, 1.0);
        pp.clear();
        let (l, r) = pp.tick(0.0, 0.0);
        assert!(l.abs() < 1e-6 && r.abs() < 1e-6);
    }

    #[test]
    fn chorus_produces_output() {
        let mut chorus = ChorusEffect::new(10.0, 3.0, 1.0, 44100.0);
        chorus.set_wet(0.5);
        chorus.set_dry(0.5);
        let mut buf: Vec<f32> = (0..4410)
            .map(|i| ((i as f64 / 44100.0 * 440.0 * 2.0 * std::f64::consts::PI).sin()) as f32)
            .collect();
        chorus.process(&mut buf);
        assert!(buf.iter().any(|s| s.abs() > 0.1));
    }

    #[test]
    fn chorus_clear() {
        let mut chorus = ChorusEffect::new(10.0, 3.0, 1.0, 44100.0);
        chorus.tick(1.0);
        chorus.clear();
        // After clear, should produce minimal output for zero input
        let val = chorus.tick(0.0);
        assert!(val.abs() < 0.01);
    }

    #[test]
    fn flanger_produces_output() {
        let mut flanger = FlangerEffect::new(2.0, 1.0, 0.5, 0.7, 44100.0);
        let mut buf: Vec<f32> = (0..4410)
            .map(|i| ((i as f64 / 44100.0 * 440.0 * 2.0 * std::f64::consts::PI).sin()) as f32)
            .collect();
        flanger.process(&mut buf);
        assert!(buf.iter().any(|s| s.abs() > 0.1));
    }

    #[test]
    fn flanger_feedback_clamp() {
        let mut flanger = FlangerEffect::new(2.0, 1.0, 0.5, 2.0, 44100.0);
        // Should be clamped to 0.99
        flanger.tick(1.0);
        for _ in 0..1000 {
            let val = flanger.tick(0.0);
            assert!(val.abs() < 100.0, "flanger should not explode");
        }
    }

    #[test]
    fn tap_delay_basic() {
        let mut td = TapDelay::new(1000);
        td.add_tap(100, 1.0, 0.0);
        td.add_tap(200, 0.5, -1.0);
        assert_eq!(td.tap_count(), 2);
    }

    #[test]
    fn tap_delay_mono() {
        let mut td = TapDelay::new(100);
        td.add_tap(10, 1.0, 0.0);
        td.set_wet(1.0);
        td.set_dry(0.0);
        // Impulse
        td.tick_mono(1.0);
        for _ in 0..9 {
            td.tick_mono(0.0);
        }
        let echo = td.tick_mono(0.0);
        assert!(echo.abs() > 0.5, "tap delay should produce echo, got {echo}");
    }

    #[test]
    fn tap_delay_stereo() {
        let mut td = TapDelay::new(100);
        td.add_tap(5, 1.0, -1.0); // hard left
        td.add_tap(10, 1.0, 1.0); // hard right
        td.set_wet(1.0);
        td.set_dry(0.0);
        td.tick_stereo(1.0);
        for _ in 0..4 {
            td.tick_stereo(0.0);
        }
        let (l, r) = td.tick_stereo(0.0);
        // Left tap at 5 samples should have output in left channel
        assert!(l.abs() > 0.5 || r.abs() > 0.01);
    }

    #[test]
    fn modulated_delay_basic() {
        let mut md = ModulatedDelay::new(10.0, 5.0, 50.0, 0.3, 44100.0);
        md.set_wet(0.5);
        md.set_dry(0.5);
        let val = md.tick(0.5, 0.0);
        // Should produce output (at least the dry signal)
        assert!(val.abs() > 0.2);
    }

    #[test]
    fn modulated_delay_with_modulation() {
        let mut md = ModulatedDelay::new(5.0, 3.0, 20.0, 0.0, 44100.0);
        md.set_wet(1.0);
        md.set_dry(0.0);
        // Write some samples
        for i in 0..1000 {
            let input = (i as f64 * 0.01).sin() as f32;
            let modulation = (i as f64 * 0.001).sin() as f32;
            md.tick(input, modulation);
        }
        // Just verify it runs without panics
    }

    #[test]
    fn delay_from_time() {
        let dl = DelayLine::from_time(0.1, 44100.0);
        assert_eq!(dl.max_delay(), 4410);
    }

    #[test]
    fn simple_delay_process_buffer() {
        let mut delay = SimpleDelay::new(5, 100, 0.5, 0.5);
        let mut buf = vec![1.0f32; 20];
        delay.process(&mut buf);
        // Output should be non-zero
        assert!(buf.iter().any(|s| s.abs() > 0.4));
    }
}
