//! ADSR envelope generator — attack/decay/sustain/release with linear and
//! exponential curves, gate control, envelope follower, and multi-stage envelopes.
//!
//! Used for amplitude shaping, filter modulation, and any parameter automation
//! in audio synthesis.

use std::fmt;

// ── Envelope Stage ──────────────────────────────────────────────

/// Current stage of an ADSR envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeStage {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

impl fmt::Display for EnvelopeStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Attack => write!(f, "Attack"),
            Self::Decay => write!(f, "Decay"),
            Self::Sustain => write!(f, "Sustain"),
            Self::Release => write!(f, "Release"),
        }
    }
}

// ── Curve Type ──────────────────────────────────────────────────

/// Interpolation curve type for envelope segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveType {
    /// Linear interpolation.
    Linear,
    /// Exponential curve. The parameter controls steepness (positive = fast start,
    /// negative = slow start). A value of 0 is equivalent to linear.
    Exponential(f64),
}

// ── ADSR Envelope ───────────────────────────────────────────────

/// ADSR (Attack, Decay, Sustain, Release) envelope generator.
#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    /// Attack time in seconds.
    attack_time: f64,
    /// Decay time in seconds.
    decay_time: f64,
    /// Sustain level (0.0 to 1.0).
    sustain_level: f64,
    /// Release time in seconds.
    release_time: f64,
    /// Attack curve type.
    attack_curve: CurveType,
    /// Decay curve type.
    decay_curve: CurveType,
    /// Release curve type.
    release_curve: CurveType,
    /// Sample rate.
    sample_rate: f64,
    /// Current stage.
    stage: EnvelopeStage,
    /// Current output value.
    value: f64,
    /// Progress within current stage (0.0 to 1.0).
    stage_progress: f64,
    /// Samples elapsed in current stage.
    stage_samples: u64,
    /// Total samples for current stage.
    stage_total_samples: u64,
    /// Value at the start of the release stage (for smooth release from any level).
    release_start_value: f64,
    /// Whether the gate is currently on.
    gate_on: bool,
}

impl AdsrEnvelope {
    /// Create a new ADSR envelope with all-linear curves.
    pub fn new(
        attack_time: f64,
        decay_time: f64,
        sustain_level: f64,
        release_time: f64,
        sample_rate: f64,
    ) -> Self {
        Self {
            attack_time: attack_time.max(0.0),
            decay_time: decay_time.max(0.0),
            sustain_level: sustain_level.clamp(0.0, 1.0),
            release_time: release_time.max(0.0),
            attack_curve: CurveType::Linear,
            decay_curve: CurveType::Linear,
            release_curve: CurveType::Linear,
            sample_rate,
            stage: EnvelopeStage::Idle,
            value: 0.0,
            stage_progress: 0.0,
            stage_samples: 0,
            stage_total_samples: 0,
            release_start_value: 0.0,
            gate_on: false,
        }
    }

    pub fn stage(&self) -> EnvelopeStage {
        self.stage
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn is_active(&self) -> bool {
        self.stage != EnvelopeStage::Idle
    }

    pub fn set_attack_time(&mut self, time: f64) {
        self.attack_time = time.max(0.0);
    }

    pub fn set_decay_time(&mut self, time: f64) {
        self.decay_time = time.max(0.0);
    }

    pub fn set_sustain_level(&mut self, level: f64) {
        self.sustain_level = level.clamp(0.0, 1.0);
    }

    pub fn set_release_time(&mut self, time: f64) {
        self.release_time = time.max(0.0);
    }

    pub fn set_attack_curve(&mut self, curve: CurveType) {
        self.attack_curve = curve;
    }

    pub fn set_decay_curve(&mut self, curve: CurveType) {
        self.decay_curve = curve;
    }

    pub fn set_release_curve(&mut self, curve: CurveType) {
        self.release_curve = curve;
    }

    /// Trigger the gate (note on). Starts from Attack stage.
    pub fn gate_on(&mut self) {
        self.gate_on = true;
        self.stage = EnvelopeStage::Attack;
        self.stage_samples = 0;
        self.stage_total_samples = self.time_to_samples(self.attack_time);
        if self.stage_total_samples == 0 {
            self.value = 1.0;
            self.transition_to_decay();
        }
    }

    /// Release the gate (note off). Starts Release stage from current value.
    pub fn gate_off(&mut self) {
        self.gate_on = false;
        if self.stage == EnvelopeStage::Idle {
            return;
        }
        self.release_start_value = self.value;
        self.stage = EnvelopeStage::Release;
        self.stage_samples = 0;
        self.stage_total_samples = self.time_to_samples(self.release_time);
        if self.stage_total_samples == 0 {
            self.value = 0.0;
            self.stage = EnvelopeStage::Idle;
        }
    }

    /// Generate a single envelope sample.
    pub fn tick(&mut self) -> f32 {
        match self.stage {
            EnvelopeStage::Idle => {
                self.value = 0.0;
            }
            EnvelopeStage::Attack => {
                if self.stage_total_samples == 0 {
                    self.value = 1.0;
                    self.transition_to_decay();
                } else {
                    self.stage_progress =
                        self.stage_samples as f64 / self.stage_total_samples as f64;
                    self.value = apply_curve(self.stage_progress, self.attack_curve);
                    self.stage_samples += 1;
                    if self.stage_samples >= self.stage_total_samples {
                        self.value = 1.0;
                        self.transition_to_decay();
                    }
                }
            }
            EnvelopeStage::Decay => {
                if self.stage_total_samples == 0 {
                    self.value = self.sustain_level;
                    self.stage = EnvelopeStage::Sustain;
                } else {
                    self.stage_progress =
                        self.stage_samples as f64 / self.stage_total_samples as f64;
                    let curve_val = apply_curve(self.stage_progress, self.decay_curve);
                    self.value = 1.0 - curve_val * (1.0 - self.sustain_level);
                    self.stage_samples += 1;
                    if self.stage_samples >= self.stage_total_samples {
                        self.value = self.sustain_level;
                        self.stage = EnvelopeStage::Sustain;
                    }
                }
            }
            EnvelopeStage::Sustain => {
                self.value = self.sustain_level;
            }
            EnvelopeStage::Release => {
                if self.stage_total_samples == 0 {
                    self.value = 0.0;
                    self.stage = EnvelopeStage::Idle;
                } else {
                    self.stage_progress =
                        self.stage_samples as f64 / self.stage_total_samples as f64;
                    let curve_val = apply_curve(self.stage_progress, self.release_curve);
                    self.value = self.release_start_value * (1.0 - curve_val);
                    self.stage_samples += 1;
                    if self.stage_samples >= self.stage_total_samples {
                        self.value = 0.0;
                        self.stage = EnvelopeStage::Idle;
                    }
                }
            }
        }
        self.value as f32
    }

    /// Generate `count` envelope samples into the output buffer.
    pub fn generate(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s = self.tick();
        }
    }

    /// Reset the envelope to idle.
    pub fn reset(&mut self) {
        self.stage = EnvelopeStage::Idle;
        self.value = 0.0;
        self.stage_progress = 0.0;
        self.stage_samples = 0;
        self.gate_on = false;
    }

    /// Get sample points for visualization (returns normalized time/value pairs).
    pub fn visualization_points(&self, points_per_stage: usize) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        let total_time = self.attack_time + self.decay_time + 0.5 + self.release_time;
        if total_time <= 0.0 {
            return points;
        }

        // Attack phase
        for i in 0..=points_per_stage {
            let t = i as f64 / points_per_stage as f64;
            let val = apply_curve(t, self.attack_curve);
            let time = t * self.attack_time / total_time;
            points.push((time, val));
        }

        // Decay phase
        let attack_end = self.attack_time / total_time;
        for i in 1..=points_per_stage {
            let t = i as f64 / points_per_stage as f64;
            let val = 1.0 - apply_curve(t, self.decay_curve) * (1.0 - self.sustain_level);
            let time = attack_end + t * self.decay_time / total_time;
            points.push((time, val));
        }

        // Sustain phase (flat)
        let decay_end = (self.attack_time + self.decay_time) / total_time;
        let sustain_end = (self.attack_time + self.decay_time + 0.5) / total_time;
        points.push((decay_end, self.sustain_level));
        points.push((sustain_end, self.sustain_level));

        // Release phase
        for i in 1..=points_per_stage {
            let t = i as f64 / points_per_stage as f64;
            let val = self.sustain_level * (1.0 - apply_curve(t, self.release_curve));
            let time = sustain_end + t * self.release_time / total_time;
            points.push((time, val));
        }

        points
    }

    fn transition_to_decay(&mut self) {
        self.stage = EnvelopeStage::Decay;
        self.stage_samples = 0;
        self.stage_total_samples = self.time_to_samples(self.decay_time);
        if self.stage_total_samples == 0 {
            self.value = self.sustain_level;
            self.stage = EnvelopeStage::Sustain;
        }
    }

    fn time_to_samples(&self, time: f64) -> u64 {
        (time * self.sample_rate) as u64
    }
}

// ── Curve Application ───────────────────────────────────────────

/// Apply a curve to a linear progress value (0.0 to 1.0).
fn apply_curve(t: f64, curve: CurveType) -> f64 {
    let t_clamped = t.clamp(0.0, 1.0);
    match curve {
        CurveType::Linear => t_clamped,
        CurveType::Exponential(steepness) => {
            if steepness.abs() < 1e-6 {
                return t_clamped;
            }
            // Attempt exponential curve: (e^(-st) - 1) / (e^(-s) - 1)
            // Positive steepness = convex (fast start), negative = concave (slow start).
            let neg_s = -steepness;
            let es = neg_s.exp();
            let denom = es - 1.0;
            if denom.abs() < 1e-10 {
                return t_clamped;
            }
            ((neg_s * t_clamped).exp() - 1.0) / denom
        }
    }
}

// ── Envelope Follower ───────────────────────────────────────────

/// Detects the amplitude envelope of an audio signal.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    /// Attack coefficient (lower = slower).
    attack_coeff: f64,
    /// Release coefficient (lower = slower).
    release_coeff: f64,
    /// Current envelope value.
    value: f64,
}

impl EnvelopeFollower {
    /// Create an envelope follower.
    /// `attack_ms` and `release_ms` control response time.
    pub fn new(attack_ms: f64, release_ms: f64, sample_rate: f64) -> Self {
        Self {
            attack_coeff: Self::ms_to_coeff(attack_ms, sample_rate),
            release_coeff: Self::ms_to_coeff(release_ms, sample_rate),
            value: 0.0,
        }
    }

    fn ms_to_coeff(ms: f64, sample_rate: f64) -> f64 {
        if ms <= 0.0 {
            return 1.0;
        }
        let samples = ms * 0.001 * sample_rate;
        (-1.0 / samples).exp()
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    /// Process a single input sample, returning the envelope value.
    pub fn tick(&mut self, input: f32) -> f32 {
        let abs_input = (input as f64).abs();
        let coeff = if abs_input > self.value {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.value = coeff * self.value + (1.0 - coeff) * abs_input;
        self.value as f32
    }

    /// Process a buffer, writing envelope values to output.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = self.tick(input[i]);
        }
    }
}

// ── Multi-Stage Envelope ────────────────────────────────────────

/// A segment in a multi-stage envelope.
#[derive(Debug, Clone)]
pub struct EnvelopeSegment {
    /// Target value at end of segment.
    pub target: f64,
    /// Duration of segment in seconds.
    pub duration: f64,
    /// Curve type for interpolation.
    pub curve: CurveType,
}

/// A free-form multi-stage envelope generator.
#[derive(Debug, Clone)]
pub struct MultiStageEnvelope {
    segments: Vec<EnvelopeSegment>,
    sample_rate: f64,
    /// Current segment index.
    current_segment: usize,
    /// Samples elapsed in current segment.
    segment_samples: u64,
    /// Total samples for current segment.
    segment_total_samples: u64,
    /// Value at start of current segment.
    start_value: f64,
    /// Current output value.
    value: f64,
    /// Whether the envelope is running.
    active: bool,
    /// Whether to loop the envelope.
    looping: bool,
    /// Sustain segment index (envelope holds here until released), or None.
    sustain_point: Option<usize>,
    /// Whether the gate is on.
    gate: bool,
}

impl MultiStageEnvelope {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            segments: Vec::new(),
            sample_rate,
            current_segment: 0,
            segment_samples: 0,
            segment_total_samples: 0,
            start_value: 0.0,
            value: 0.0,
            active: false,
            looping: false,
            sustain_point: None,
            gate: false,
        }
    }

    /// Add a segment to the envelope.
    pub fn add_segment(&mut self, segment: EnvelopeSegment) {
        self.segments.push(segment);
    }

    /// Set the sustain point (segment index where envelope holds until released).
    pub fn set_sustain_point(&mut self, index: Option<usize>) {
        self.sustain_point = index;
    }

    /// Set whether the envelope loops.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Start the envelope (gate on).
    pub fn trigger(&mut self) {
        if self.segments.is_empty() {
            return;
        }
        self.active = true;
        self.gate = true;
        self.current_segment = 0;
        self.start_value = 0.0;
        self.segment_samples = 0;
        let dur = self.segments[0].duration;
        self.segment_total_samples = (dur * self.sample_rate) as u64;
    }

    /// Release the envelope (gate off). Continues past sustain point.
    pub fn release(&mut self) {
        self.gate = false;
    }

    /// Generate one sample.
    pub fn tick(&mut self) -> f32 {
        if !self.active || self.segments.is_empty() {
            return self.value as f32;
        }

        // Check for sustain hold
        if let Some(sp) = self.sustain_point {
            if self.gate && self.current_segment > sp {
                // Hold at sustain level
                return self.value as f32;
            }
        }

        if self.current_segment >= self.segments.len() {
            if self.looping {
                self.current_segment = 0;
                self.start_value = self.value;
                self.segment_samples = 0;
                let dur = self.segments[0].duration;
                self.segment_total_samples = (dur * self.sample_rate) as u64;
            } else {
                self.active = false;
                return self.value as f32;
            }
        }

        let seg = &self.segments[self.current_segment];
        if self.segment_total_samples == 0 {
            self.value = seg.target;
            self.advance_segment();
            return self.value as f32;
        }

        let progress = self.segment_samples as f64 / self.segment_total_samples as f64;
        let curved = apply_curve(progress, seg.curve);
        self.value = self.start_value + (seg.target - self.start_value) * curved;

        self.segment_samples += 1;
        if self.segment_samples >= self.segment_total_samples {
            self.value = seg.target;
            self.advance_segment();
        }

        self.value as f32
    }

    /// Generate samples into output.
    pub fn generate(&mut self, output: &mut [f32]) {
        for s in output.iter_mut() {
            *s = self.tick();
        }
    }

    pub fn reset(&mut self) {
        self.current_segment = 0;
        self.segment_samples = 0;
        self.value = 0.0;
        self.start_value = 0.0;
        self.active = false;
        self.gate = false;
    }

    fn advance_segment(&mut self) {
        self.start_value = self.value;
        self.current_segment += 1;

        // Check sustain hold
        if let Some(sp) = self.sustain_point {
            if self.gate && self.current_segment > sp {
                return;
            }
        }

        if self.current_segment < self.segments.len() {
            self.segment_samples = 0;
            let dur = self.segments[self.current_segment].duration;
            self.segment_total_samples = (dur * self.sample_rate) as u64;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;

    fn gen_samples(env: &mut AdsrEnvelope, count: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; count];
        env.generate(&mut buf);
        buf
    }

    #[test]
    fn idle_produces_zero() {
        let mut env = AdsrEnvelope::new(0.01, 0.01, 0.5, 0.01, SR);
        let samples = gen_samples(&mut env, 100);
        assert!(samples.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn attack_reaches_peak() {
        let mut env = AdsrEnvelope::new(0.01, 0.01, 0.5, 0.01, SR);
        env.gate_on();
        // 0.01s = 441 samples at 44100
        let samples = gen_samples(&mut env, 500);
        // Should reach 1.0 by end of attack
        assert!(
            samples.iter().any(|s| *s > 0.99),
            "max = {}",
            samples.iter().cloned().fold(0.0f32, f32::max)
        );
    }

    #[test]
    fn decay_to_sustain() {
        let mut env = AdsrEnvelope::new(0.001, 0.01, 0.5, 0.01, SR);
        env.gate_on();
        let samples = gen_samples(&mut env, 1000);
        // Near end, should be at sustain level (~0.5)
        let last = samples[999];
        assert!(
            (last - 0.5).abs() < 0.05,
            "expected ~0.5, got {last}"
        );
    }

    #[test]
    fn sustain_holds() {
        let mut env = AdsrEnvelope::new(0.001, 0.001, 0.7, 0.01, SR);
        env.gate_on();
        let _ = gen_samples(&mut env, 500);
        assert_eq!(env.stage(), EnvelopeStage::Sustain);
        let more = gen_samples(&mut env, 100);
        assert!(more.iter().all(|s| (*s - 0.7).abs() < 0.01));
    }

    #[test]
    fn release_to_zero() {
        let mut env = AdsrEnvelope::new(0.001, 0.001, 0.5, 0.01, SR);
        env.gate_on();
        let _ = gen_samples(&mut env, 500);
        env.gate_off();
        let samples = gen_samples(&mut env, 1000);
        let last = samples[999];
        assert!(
            last < 0.01,
            "should release to ~0, got {last}"
        );
    }

    #[test]
    fn envelope_goes_idle() {
        let mut env = AdsrEnvelope::new(0.001, 0.001, 0.5, 0.005, SR);
        env.gate_on();
        let _ = gen_samples(&mut env, 500);
        env.gate_off();
        let _ = gen_samples(&mut env, 500);
        assert_eq!(env.stage(), EnvelopeStage::Idle);
        assert!(!env.is_active());
    }

    #[test]
    fn exponential_attack_curve() {
        let mut env = AdsrEnvelope::new(0.01, 0.01, 0.5, 0.01, SR);
        env.set_attack_curve(CurveType::Exponential(3.0));
        env.gate_on();
        let samples = gen_samples(&mut env, 500);
        // Exponential attack with positive steepness = fast start
        // Midpoint should be above 0.5 (convex curve)
        let mid = samples[200];
        assert!(mid > 0.3, "exponential attack midpoint = {mid}");
    }

    #[test]
    fn zero_attack_time() {
        let mut env = AdsrEnvelope::new(0.0, 0.01, 0.5, 0.01, SR);
        env.gate_on();
        let s = env.tick();
        // With zero attack, should immediately be at peak or decay
        assert!(s > 0.9 || env.stage() == EnvelopeStage::Decay);
    }

    #[test]
    fn zero_release_time() {
        let mut env = AdsrEnvelope::new(0.001, 0.001, 0.5, 0.0, SR);
        env.gate_on();
        let _ = gen_samples(&mut env, 500);
        env.gate_off();
        // Should immediately go to idle
        let s = env.tick();
        assert!(s < 0.01);
        assert_eq!(env.stage(), EnvelopeStage::Idle);
    }

    #[test]
    fn reset_envelope() {
        let mut env = AdsrEnvelope::new(0.01, 0.01, 0.5, 0.01, SR);
        env.gate_on();
        let _ = gen_samples(&mut env, 100);
        env.reset();
        assert_eq!(env.stage(), EnvelopeStage::Idle);
        assert!(env.value() < 1e-10);
    }

    #[test]
    fn visualization_points() {
        let env = AdsrEnvelope::new(0.1, 0.1, 0.5, 0.2, SR);
        let points = env.visualization_points(10);
        assert!(points.len() >= 20);
        // First point should be (0, 0)
        assert!(points[0].0 < 0.01);
        assert!(points[0].1 < 0.01);
        // Last point should be near (1, 0)
        let last = points.last().unwrap();
        assert!(last.1 < 0.01);
    }

    #[test]
    fn envelope_follower_tracks_amplitude() {
        let mut follower = EnvelopeFollower::new(1.0, 50.0, SR);
        // Feed a loud signal
        for _ in 0..4410 {
            follower.tick(0.8);
        }
        assert!(
            follower.value() > 0.5,
            "follower should track loud signal, got {}",
            follower.value()
        );
        // Feed silence
        for _ in 0..44100 {
            follower.tick(0.0);
        }
        assert!(
            follower.value() < 0.05,
            "follower should decay to near zero, got {}",
            follower.value()
        );
    }

    #[test]
    fn envelope_follower_reset() {
        let mut follower = EnvelopeFollower::new(1.0, 50.0, SR);
        follower.tick(0.5);
        follower.tick(0.5);
        follower.reset();
        assert!(follower.value() < 1e-10);
    }

    #[test]
    fn multi_stage_envelope_basic() {
        let mut env = MultiStageEnvelope::new(SR);
        env.add_segment(EnvelopeSegment {
            target: 1.0,
            duration: 0.001,
            curve: CurveType::Linear,
        });
        env.add_segment(EnvelopeSegment {
            target: 0.0,
            duration: 0.001,
            curve: CurveType::Linear,
        });
        assert_eq!(env.segment_count(), 2);
        env.trigger();
        assert!(env.is_active());
        let mut buf = vec![0.0f32; 200];
        env.generate(&mut buf);
        // Should reach 1.0 then come back to 0.0
        assert!(buf.iter().any(|s| *s > 0.9));
    }

    #[test]
    fn multi_stage_looping() {
        let mut env = MultiStageEnvelope::new(1000.0); // low SR for fast test
        env.add_segment(EnvelopeSegment {
            target: 1.0,
            duration: 0.01,
            curve: CurveType::Linear,
        });
        env.add_segment(EnvelopeSegment {
            target: 0.0,
            duration: 0.01,
            curve: CurveType::Linear,
        });
        env.set_looping(true);
        env.trigger();
        let mut buf = vec![0.0f32; 100];
        env.generate(&mut buf);
        // Should still be active after completing all segments (looping)
        assert!(env.is_active());
    }

    #[test]
    fn multi_stage_sustain_point() {
        let mut env = MultiStageEnvelope::new(1000.0);
        env.add_segment(EnvelopeSegment {
            target: 1.0,
            duration: 0.01,
            curve: CurveType::Linear,
        });
        env.add_segment(EnvelopeSegment {
            target: 0.5,
            duration: 0.01,
            curve: CurveType::Linear,
        });
        env.add_segment(EnvelopeSegment {
            target: 0.0,
            duration: 0.01,
            curve: CurveType::Linear,
        });
        env.set_sustain_point(Some(1));
        env.trigger();
        let mut buf = vec![0.0f32; 100];
        env.generate(&mut buf);
        // Should hold at sustain level
        let last = buf[99];
        assert!(
            (last - 0.5).abs() < 0.1,
            "should hold near sustain, got {last}"
        );
    }

    #[test]
    fn stage_display() {
        assert_eq!(format!("{}", EnvelopeStage::Attack), "Attack");
        assert_eq!(format!("{}", EnvelopeStage::Idle), "Idle");
    }

    #[test]
    fn apply_curve_linear() {
        assert!((apply_curve(0.0, CurveType::Linear)).abs() < 1e-10);
        assert!((apply_curve(0.5, CurveType::Linear) - 0.5).abs() < 1e-10);
        assert!((apply_curve(1.0, CurveType::Linear) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn apply_curve_exponential() {
        let val = apply_curve(0.5, CurveType::Exponential(3.0));
        // With positive steepness, midpoint should be above 0.5
        assert!(val > 0.5, "exponential(3.0) at 0.5 = {val}");
        // Endpoints should still be 0 and 1
        assert!((apply_curve(0.0, CurveType::Exponential(3.0))).abs() < 1e-6);
        assert!((apply_curve(1.0, CurveType::Exponential(3.0)) - 1.0).abs() < 1e-6);
    }
}
