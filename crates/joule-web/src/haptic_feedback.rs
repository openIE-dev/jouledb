//! Haptic/vibration feedback system.
//!
//! Haptic effects: constant force, ramp, sine wave, custom waveform. Duration,
//! intensity (0.0-1.0), frequency. Effect layering (multiple simultaneous).
//! Haptic channels (left/right motor). Predefined patterns: click, explosion,
//! heartbeat, engine_rev. Energy cost tracking per haptic event.

// ── Haptic Channel ──────────────────────────────────────────────

/// Haptic motor channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HapticChannel {
    Left,
    Right,
    Both,
}

// ── Waveform Type ───────────────────────────────────────────────

/// Type of haptic waveform.
#[derive(Debug, Clone, PartialEq)]
pub enum WaveformType {
    /// Constant intensity for the duration.
    Constant,
    /// Linear ramp from start_intensity to end_intensity.
    Ramp { start_intensity: f64, end_intensity: f64 },
    /// Sine wave oscillation.
    Sine { frequency_hz: f64 },
    /// Custom waveform: list of (time_fraction 0..1, intensity 0..1) points.
    Custom { points: Vec<(f64, f64)> },
}

// ── Haptic Effect ───────────────────────────────────────────────

/// A single haptic effect.
#[derive(Debug, Clone, PartialEq)]
pub struct HapticEffect {
    pub id: u64,
    pub channel: HapticChannel,
    pub waveform: WaveformType,
    pub intensity: f64,
    pub duration_ms: f64,
    pub elapsed_ms: f64,
    pub active: bool,
    /// Energy cost in microjoules consumed so far.
    pub energy_uj: f64,
}

impl HapticEffect {
    pub fn new(id: u64, channel: HapticChannel, waveform: WaveformType, intensity: f64, duration_ms: f64) -> Self {
        Self {
            id,
            channel,
            waveform,
            intensity: intensity.clamp(0.0, 1.0),
            duration_ms: duration_ms.max(0.0),
            elapsed_ms: 0.0,
            active: true,
            energy_uj: 0.0,
        }
    }

    /// Sample the effect intensity at the current time.
    pub fn sample(&self) -> f64 {
        if !self.active || self.duration_ms <= 0.0 { return 0.0; }
        let t = (self.elapsed_ms / self.duration_ms).clamp(0.0, 1.0);
        let wave = match &self.waveform {
            WaveformType::Constant => 1.0,
            WaveformType::Ramp { start_intensity, end_intensity } => {
                start_intensity + (end_intensity - start_intensity) * t
            }
            WaveformType::Sine { frequency_hz } => {
                let cycles = frequency_hz * (self.elapsed_ms / 1000.0);
                (cycles * 2.0 * std::f64::consts::PI).sin() * 0.5 + 0.5
            }
            WaveformType::Custom { points } => {
                interpolate_custom(points, t)
            }
        };
        (wave * self.intensity).clamp(0.0, 1.0)
    }

    /// Advance the effect by dt milliseconds. Returns the energy consumed in this step.
    pub fn advance(&mut self, dt_ms: f64) -> f64 {
        if !self.active { return 0.0; }
        let sample_before = self.sample();
        self.elapsed_ms += dt_ms;
        if self.elapsed_ms >= self.duration_ms {
            self.active = false;
        }
        // Energy = intensity * time (simplified model: µJ = intensity * dt_ms * scale)
        let avg_intensity = (sample_before + self.sample()) / 2.0;
        let energy = avg_intensity * dt_ms * 10.0; // 10 µJ per ms at full intensity
        self.energy_uj += energy;
        energy
    }

    /// Progress fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.duration_ms <= 0.0 { return 1.0; }
        (self.elapsed_ms / self.duration_ms).clamp(0.0, 1.0)
    }

    /// Remaining time in ms.
    pub fn remaining_ms(&self) -> f64 {
        (self.duration_ms - self.elapsed_ms).max(0.0)
    }
}

fn interpolate_custom(points: &[(f64, f64)], t: f64) -> f64 {
    if points.is_empty() { return 0.0; }
    if points.len() == 1 { return points[0].1.clamp(0.0, 1.0); }
    if t <= points[0].0 { return points[0].1.clamp(0.0, 1.0); }
    let last = points.len() - 1;
    if t >= points[last].0 { return points[last].1.clamp(0.0, 1.0); }

    for i in 0..last {
        let (t0, v0) = points[i];
        let (t1, v1) = points[i + 1];
        if t >= t0 && t <= t1 {
            let frac = if (t1 - t0).abs() < 1e-12 { 0.0 } else { (t - t0) / (t1 - t0) };
            return (v0 + (v1 - v0) * frac).clamp(0.0, 1.0);
        }
    }
    points[last].1.clamp(0.0, 1.0)
}

// ── Predefined Patterns ─────────────────────────────────────────

/// Predefined haptic patterns.
pub struct HapticPatterns;

impl HapticPatterns {
    /// Short click feedback.
    pub fn click() -> (WaveformType, f64, f64) {
        (WaveformType::Constant, 0.6, 30.0)
    }

    /// Explosion feedback.
    pub fn explosion() -> (WaveformType, f64, f64) {
        (WaveformType::Ramp { start_intensity: 1.0, end_intensity: 0.0 }, 1.0, 500.0)
    }

    /// Heartbeat pattern.
    pub fn heartbeat() -> (WaveformType, f64, f64) {
        let points = vec![
            (0.0, 0.0), (0.1, 1.0), (0.2, 0.0),
            (0.35, 0.7), (0.45, 0.0), (1.0, 0.0),
        ];
        (WaveformType::Custom { points }, 0.8, 600.0)
    }

    /// Engine rev vibration.
    pub fn engine_rev() -> (WaveformType, f64, f64) {
        (WaveformType::Sine { frequency_hz: 30.0 }, 0.4, 1000.0)
    }

    /// Soft landing.
    pub fn soft_landing() -> (WaveformType, f64, f64) {
        (WaveformType::Ramp { start_intensity: 0.5, end_intensity: 0.0 }, 0.5, 150.0)
    }

    /// Damage taken.
    pub fn damage() -> (WaveformType, f64, f64) {
        let points = vec![
            (0.0, 1.0), (0.15, 0.2), (0.3, 0.8), (0.5, 0.1), (1.0, 0.0),
        ];
        (WaveformType::Custom { points }, 0.9, 400.0)
    }
}

// ── Haptic System ───────────────────────────────────────────────

/// Manages multiple simultaneous haptic effects with energy tracking.
pub struct HapticSystem {
    effects: Vec<HapticEffect>,
    next_id: u64,
    total_energy_uj: f64,
    max_simultaneous: usize,
    /// Output: combined intensity per channel after tick.
    left_intensity: f64,
    right_intensity: f64,
}

impl HapticSystem {
    pub fn new(max_simultaneous: usize) -> Self {
        Self {
            effects: Vec::new(),
            next_id: 1,
            total_energy_uj: 0.0,
            max_simultaneous,
            left_intensity: 0.0,
            right_intensity: 0.0,
        }
    }

    /// Play a haptic effect. Returns the effect ID.
    pub fn play(&mut self, channel: HapticChannel, waveform: WaveformType, intensity: f64, duration_ms: f64) -> u64 {
        // Remove inactive effects first
        self.effects.retain(|e| e.active);

        if self.effects.len() >= self.max_simultaneous {
            // Remove the oldest effect
            if !self.effects.is_empty() {
                self.effects.remove(0);
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.effects.push(HapticEffect::new(id, channel, waveform, intensity, duration_ms));
        id
    }

    /// Play a predefined pattern.
    pub fn play_pattern(&mut self, channel: HapticChannel, pattern: (WaveformType, f64, f64)) -> u64 {
        self.play(channel, pattern.0, pattern.1, pattern.2)
    }

    /// Stop a specific effect by ID.
    pub fn stop(&mut self, id: u64) -> bool {
        if let Some(effect) = self.effects.iter_mut().find(|e| e.id == id) {
            effect.active = false;
            true
        } else {
            false
        }
    }

    /// Stop all active effects.
    pub fn stop_all(&mut self) {
        for effect in &mut self.effects {
            effect.active = false;
        }
    }

    /// Advance all effects by dt_ms. Computes combined channel intensities.
    pub fn tick(&mut self, dt_ms: f64) {
        let mut left = 0.0_f64;
        let mut right = 0.0_f64;

        for effect in &mut self.effects {
            let energy = effect.advance(dt_ms);
            self.total_energy_uj += energy;
            let sample = effect.sample();
            match effect.channel {
                HapticChannel::Left => left += sample,
                HapticChannel::Right => right += sample,
                HapticChannel::Both => { left += sample; right += sample; }
            }
        }

        self.left_intensity = left.clamp(0.0, 1.0);
        self.right_intensity = right.clamp(0.0, 1.0);

        // Prune completed effects
        self.effects.retain(|e| e.active);
    }

    /// Combined left motor intensity (0.0 to 1.0).
    pub fn left_intensity(&self) -> f64 { self.left_intensity }

    /// Combined right motor intensity (0.0 to 1.0).
    pub fn right_intensity(&self) -> f64 { self.right_intensity }

    /// Number of currently active effects.
    pub fn active_count(&self) -> usize {
        self.effects.iter().filter(|e| e.active).count()
    }

    /// Total energy consumed in microjoules.
    pub fn total_energy_uj(&self) -> f64 { self.total_energy_uj }

    /// Reset energy counter.
    pub fn reset_energy(&mut self) {
        self.total_energy_uj = 0.0;
    }

    /// Get an active effect by ID.
    pub fn effect(&self, id: u64) -> Option<&HapticEffect> {
        self.effects.iter().find(|e| e.id == id && e.active)
    }
}

impl Default for HapticSystem {
    fn default() -> Self { Self::new(8) }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_effect() {
        let effect = HapticEffect::new(1, HapticChannel::Both, WaveformType::Constant, 0.5, 100.0);
        assert!((effect.sample() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ramp_effect() {
        let mut effect = HapticEffect::new(1, HapticChannel::Both,
            WaveformType::Ramp { start_intensity: 1.0, end_intensity: 0.0 }, 1.0, 100.0);
        assert!((effect.sample() - 1.0).abs() < 1e-6);
        effect.advance(50.0);
        assert!((effect.sample() - 0.5).abs() < 1e-6);
        effect.advance(50.0);
        // At exactly 100ms, should be at 0.0
        assert!((effect.sample() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sine_effect() {
        let effect = HapticEffect::new(1, HapticChannel::Left,
            WaveformType::Sine { frequency_hz: 10.0 }, 1.0, 1000.0);
        let s = effect.sample();
        // At t=0, sin(0) = 0 => 0*0.5 + 0.5 = 0.5
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_custom_waveform() {
        let points = vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)];
        let mut effect = HapticEffect::new(1, HapticChannel::Right,
            WaveformType::Custom { points }, 1.0, 100.0);
        assert!((effect.sample() - 0.0).abs() < 1e-6);
        effect.advance(50.0);
        assert!((effect.sample() - 1.0).abs() < 1e-6);
        effect.advance(50.0);
        assert!((effect.sample() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_effect_completes() {
        let mut effect = HapticEffect::new(1, HapticChannel::Both,
            WaveformType::Constant, 1.0, 50.0);
        assert!(effect.active);
        effect.advance(60.0);
        assert!(!effect.active);
    }

    #[test]
    fn test_effect_progress() {
        let mut effect = HapticEffect::new(1, HapticChannel::Both,
            WaveformType::Constant, 1.0, 100.0);
        assert!((effect.progress() - 0.0).abs() < 1e-9);
        effect.advance(50.0);
        assert!((effect.progress() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_effect_remaining() {
        let mut effect = HapticEffect::new(1, HapticChannel::Both,
            WaveformType::Constant, 1.0, 100.0);
        assert!((effect.remaining_ms() - 100.0).abs() < 1e-9);
        effect.advance(40.0);
        assert!((effect.remaining_ms() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_energy_tracking() {
        let mut effect = HapticEffect::new(1, HapticChannel::Both,
            WaveformType::Constant, 1.0, 100.0);
        effect.advance(100.0);
        assert!(effect.energy_uj > 0.0);
    }

    #[test]
    fn test_intensity_clamped() {
        let effect = HapticEffect::new(1, HapticChannel::Both,
            WaveformType::Constant, 1.5, 100.0);
        assert!((effect.intensity - 1.0).abs() < 1e-9);
        let effect2 = HapticEffect::new(2, HapticChannel::Both,
            WaveformType::Constant, -0.5, 100.0);
        assert!((effect2.intensity - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_system_play() {
        let mut sys = HapticSystem::new(8);
        let id = sys.play(HapticChannel::Both, WaveformType::Constant, 0.5, 100.0);
        assert_eq!(id, 1);
        assert_eq!(sys.active_count(), 1);
    }

    #[test]
    fn test_system_tick() {
        let mut sys = HapticSystem::new(8);
        sys.play(HapticChannel::Left, WaveformType::Constant, 0.5, 100.0);
        sys.tick(16.0);
        assert!(sys.left_intensity() > 0.0);
        assert!((sys.right_intensity() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_system_both_channels() {
        let mut sys = HapticSystem::new(8);
        sys.play(HapticChannel::Both, WaveformType::Constant, 0.5, 100.0);
        sys.tick(16.0);
        assert!(sys.left_intensity() > 0.0);
        assert!(sys.right_intensity() > 0.0);
    }

    #[test]
    fn test_system_stop() {
        let mut sys = HapticSystem::new(8);
        let id = sys.play(HapticChannel::Both, WaveformType::Constant, 0.5, 100.0);
        assert!(sys.stop(id));
        sys.tick(1.0);
        assert_eq!(sys.active_count(), 0);
    }

    #[test]
    fn test_system_stop_all() {
        let mut sys = HapticSystem::new(8);
        sys.play(HapticChannel::Left, WaveformType::Constant, 0.5, 100.0);
        sys.play(HapticChannel::Right, WaveformType::Constant, 0.5, 100.0);
        sys.stop_all();
        sys.tick(1.0);
        assert_eq!(sys.active_count(), 0);
    }

    #[test]
    fn test_system_max_simultaneous() {
        let mut sys = HapticSystem::new(2);
        sys.play(HapticChannel::Both, WaveformType::Constant, 0.5, 1000.0);
        sys.play(HapticChannel::Both, WaveformType::Constant, 0.5, 1000.0);
        sys.play(HapticChannel::Both, WaveformType::Constant, 0.5, 1000.0);
        // Should have removed oldest, keeping 2
        assert!(sys.active_count() <= 2);
    }

    #[test]
    fn test_system_energy() {
        let mut sys = HapticSystem::new(8);
        sys.play(HapticChannel::Both, WaveformType::Constant, 1.0, 100.0);
        sys.tick(100.0);
        assert!(sys.total_energy_uj() > 0.0);
    }

    #[test]
    fn test_system_reset_energy() {
        let mut sys = HapticSystem::new(8);
        sys.play(HapticChannel::Both, WaveformType::Constant, 1.0, 100.0);
        sys.tick(100.0);
        sys.reset_energy();
        assert!((sys.total_energy_uj() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_predefined_click() {
        let (wf, intensity, duration) = HapticPatterns::click();
        assert!(matches!(wf, WaveformType::Constant));
        assert!((intensity - 0.6).abs() < 1e-9);
        assert!((duration - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_predefined_explosion() {
        let (wf, intensity, duration) = HapticPatterns::explosion();
        assert!(matches!(wf, WaveformType::Ramp { .. }));
        assert!((intensity - 1.0).abs() < 1e-9);
        assert!((duration - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_predefined_heartbeat() {
        let (wf, _intensity, _duration) = HapticPatterns::heartbeat();
        assert!(matches!(wf, WaveformType::Custom { .. }));
    }

    #[test]
    fn test_predefined_engine_rev() {
        let (wf, _intensity, duration) = HapticPatterns::engine_rev();
        assert!(matches!(wf, WaveformType::Sine { .. }));
        assert!((duration - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_play_pattern() {
        let mut sys = HapticSystem::new(8);
        let id = sys.play_pattern(HapticChannel::Both, HapticPatterns::click());
        assert!(sys.effect(id).is_some());
    }

    #[test]
    fn test_effect_auto_cleanup() {
        let mut sys = HapticSystem::new(8);
        sys.play(HapticChannel::Both, WaveformType::Constant, 1.0, 10.0);
        sys.tick(20.0);
        assert_eq!(sys.active_count(), 0);
    }

    #[test]
    fn test_interpolate_custom_empty() {
        let val = interpolate_custom(&[], 0.5);
        assert!((val - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_custom_single() {
        let val = interpolate_custom(&[(0.0, 0.7)], 0.5);
        assert!((val - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_default_system() {
        let sys = HapticSystem::default();
        assert_eq!(sys.active_count(), 0);
    }

    #[test]
    fn test_stop_nonexistent() {
        let mut sys = HapticSystem::new(8);
        assert!(!sys.stop(999));
    }
}
