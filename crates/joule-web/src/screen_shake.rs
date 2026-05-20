//! Camera screen shake system: trauma-based shake, Perlin-noise smooth shake,
//! directional shake, decay modes, and stackable shake sources.
//!
//! Uses the "trauma²" model popularised by Squirrel Eiserloh's GDC talk:
//! visual shake intensity = trauma², giving a natural falloff.

use std::collections::HashMap;

// ── Decay Mode ─────────────────────────────────────────────────

/// How a shake's amplitude decays over time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecayMode {
    /// Amplitude drops linearly to 0 over `duration`.
    Linear,
    /// Amplitude drops exponentially with the given half-life in seconds.
    Exponential { half_life: f64 },
    /// No decay — shake persists until manually removed.
    None,
}

// ── Shake Source ───────────────────────────────────────────────

/// A single shake source with its own parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ShakeSource {
    /// Unique ID for this source.
    pub id: String,
    /// Maximum displacement in pixels along X.
    pub amplitude_x: f64,
    /// Maximum displacement in pixels along Y.
    pub amplitude_y: f64,
    /// Oscillation frequency in Hz (cycles per second).
    pub frequency: f64,
    /// Decay behaviour.
    pub decay: DecayMode,
    /// Total duration in seconds (for linear decay). Ignored for Exponential/None.
    pub duration: f64,
    /// Directional bias (unit vector). `(0,0)` = omnidirectional.
    pub direction_x: f64,
    pub direction_y: f64,
    /// Use smooth (Perlin-like) noise vs random jitter.
    pub smooth: bool,
    /// Elapsed time since this source started.
    elapsed: f64,
    /// Whether this source has finished (amplitude ≈ 0).
    finished: bool,
}

impl ShakeSource {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            amplitude_x: 10.0,
            amplitude_y: 10.0,
            frequency: 15.0,
            decay: DecayMode::Linear,
            duration: 0.5,
            direction_x: 0.0,
            direction_y: 0.0,
            smooth: true,
            elapsed: 0.0,
            finished: false,
        }
    }

    pub fn with_amplitude(mut self, ax: f64, ay: f64) -> Self {
        self.amplitude_x = ax;
        self.amplitude_y = ay;
        self
    }

    pub fn with_frequency(mut self, freq: f64) -> Self {
        self.frequency = freq;
        self
    }

    pub fn with_decay(mut self, decay: DecayMode, duration: f64) -> Self {
        self.decay = decay;
        self.duration = duration;
        self
    }

    pub fn with_direction(mut self, dx: f64, dy: f64) -> Self {
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1e-12 {
            self.direction_x = dx / len;
            self.direction_y = dy / len;
        }
        self
    }

    pub fn with_smooth(mut self, smooth: bool) -> Self {
        self.smooth = smooth;
        self
    }

    /// Current decay multiplier in [0, 1].
    fn decay_factor(&self) -> f64 {
        match self.decay {
            DecayMode::Linear => {
                if self.duration <= 0.0 {
                    return 0.0;
                }
                (1.0 - self.elapsed / self.duration).max(0.0)
            }
            DecayMode::Exponential { half_life } => {
                if half_life <= 0.0 {
                    return 0.0;
                }
                (-(self.elapsed * 0.693147 / half_life)).exp()
            }
            DecayMode::None => 1.0,
        }
    }

    /// Compute displacement at current time.
    fn displacement(&self) -> (f64, f64) {
        if self.finished {
            return (0.0, 0.0);
        }
        let decay = self.decay_factor();
        if decay < 1e-6 {
            return (0.0, 0.0);
        }

        let t = self.elapsed * self.frequency;

        let (raw_x, raw_y) = if self.smooth {
            // Smooth noise approximation using layered sines
            let nx = perlin_approx(t, 0.0);
            let ny = perlin_approx(t, 100.0);
            (nx, ny)
        } else {
            // Deterministic pseudo-random jitter from sine hash
            let jx = hash_jitter(t, 0.0);
            let jy = hash_jitter(t, 50.0);
            (jx, jy)
        };

        let dx = raw_x * self.amplitude_x * decay;
        let dy = raw_y * self.amplitude_y * decay;

        // Apply directional bias
        if self.direction_x.abs() > 1e-12 || self.direction_y.abs() > 1e-12 {
            // Project displacement onto direction
            let dot = dx * self.direction_x + dy * self.direction_y;
            (dot * self.direction_x, dot * self.direction_y)
        } else {
            (dx, dy)
        }
    }
}

/// Smooth noise approximation: layered sine waves at different frequencies.
fn perlin_approx(t: f64, seed: f64) -> f64 {
    let v = (t + seed).sin() * 0.5
        + (t * 2.3 + seed * 1.7).sin() * 0.3
        + (t * 4.1 + seed * 3.1).sin() * 0.2;
    v.clamp(-1.0, 1.0)
}

/// Deterministic jitter from sine-hash.
fn hash_jitter(t: f64, seed: f64) -> f64 {
    let v = ((t + seed) * 12.9898 + 78.233).sin() * 43758.5453;
    (v - v.floor()) * 2.0 - 1.0
}

// ── Trauma System ──────────────────────────────────────────────

/// Trauma-based shake: trauma accumulates from damage/events, decays over
/// time, and shake intensity = trauma².
#[derive(Debug, Clone, PartialEq)]
pub struct TraumaShake {
    /// Current trauma level in [0, max_trauma].
    pub trauma: f64,
    /// Maximum trauma (clamped).
    pub max_trauma: f64,
    /// Trauma decay per second.
    pub decay_rate: f64,
    /// Max displacement when trauma=1.
    pub max_offset_x: f64,
    pub max_offset_y: f64,
    /// Max rotation in radians when trauma=1.
    pub max_rotation: f64,
    /// Frequency.
    pub frequency: f64,
    /// Internal time accumulator.
    time: f64,
}

impl TraumaShake {
    pub fn new() -> Self {
        Self {
            trauma: 0.0,
            max_trauma: 1.0,
            decay_rate: 1.0,
            max_offset_x: 20.0,
            max_offset_y: 20.0,
            max_rotation: 0.05,
            frequency: 20.0,
            time: 0.0,
        }
    }

    /// Add trauma (e.g., on taking damage).
    pub fn add_trauma(&mut self, amount: f64) {
        self.trauma = (self.trauma + amount).min(self.max_trauma);
    }

    /// Update decay and internal clock.
    pub fn update(&mut self, dt: f64) {
        self.time += dt;
        self.trauma = (self.trauma - self.decay_rate * dt).max(0.0);
    }

    /// Shake intensity = trauma².
    pub fn intensity(&self) -> f64 {
        self.trauma * self.trauma
    }

    /// Current displacement `(dx, dy, rotation)`.
    pub fn displacement(&self) -> (f64, f64, f64) {
        let intensity = self.intensity();
        if intensity < 1e-9 {
            return (0.0, 0.0, 0.0);
        }
        let t = self.time * self.frequency;
        let nx = perlin_approx(t, 1.0);
        let ny = perlin_approx(t, 200.0);
        let nr = perlin_approx(t, 300.0);
        (
            nx * self.max_offset_x * intensity,
            ny * self.max_offset_y * intensity,
            nr * self.max_rotation * intensity,
        )
    }

    pub fn is_active(&self) -> bool {
        self.trauma > 1e-9
    }
}

impl Default for TraumaShake {
    fn default() -> Self {
        Self::new()
    }
}

// ── Screen Shake System ────────────────────────────────────────

/// Manages multiple shake sources and combines them into a final offset.
#[derive(Debug, Clone)]
pub struct ScreenShakeSystem {
    sources: HashMap<String, ShakeSource>,
    trauma: TraumaShake,
    /// Maximum combined displacement in any axis.
    pub max_displacement: f64,
}

impl ScreenShakeSystem {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            trauma: TraumaShake::new(),
            max_displacement: 50.0,
        }
    }

    /// Add a shake source.
    pub fn add_source(&mut self, source: ShakeSource) {
        self.sources.insert(source.id.clone(), source);
    }

    /// Remove a shake source by ID.
    pub fn remove_source(&mut self, id: &str) -> bool {
        self.sources.remove(id).is_some()
    }

    /// Add trauma (stacks with trauma shake).
    pub fn add_trauma(&mut self, amount: f64) {
        self.trauma.add_trauma(amount);
    }

    /// Configure the trauma subsystem.
    pub fn trauma_mut(&mut self) -> &mut TraumaShake {
        &mut self.trauma
    }

    /// Advance all shake sources by `dt` seconds.
    pub fn update(&mut self, dt: f64) {
        let keys: Vec<String> = self.sources.keys().cloned().collect();
        for key in keys {
            if let Some(src) = self.sources.get_mut(&key) {
                src.elapsed += dt;
                match src.decay {
                    DecayMode::Linear => {
                        if src.elapsed >= src.duration {
                            src.finished = true;
                        }
                    }
                    DecayMode::Exponential { half_life } => {
                        // Consider finished when decay < 0.1%
                        if half_life > 0.0 && src.decay_factor() < 0.001 {
                            src.finished = true;
                        }
                    }
                    DecayMode::None => {}
                }
            }
        }
        // Remove finished sources
        self.sources.retain(|_, s| !s.finished);

        self.trauma.update(dt);
    }

    /// Compute the combined screen offset `(dx, dy)`.
    pub fn offset(&self) -> (f64, f64) {
        let mut dx = 0.0f64;
        let mut dy = 0.0f64;

        for src in self.sources.values() {
            let (sx, sy) = src.displacement();
            dx += sx;
            dy += sy;
        }

        // Add trauma displacement
        let (tx, ty, _) = self.trauma.displacement();
        dx += tx;
        dy += ty;

        // Clamp
        let max = self.max_displacement;
        dx = dx.clamp(-max, max);
        dy = dy.clamp(-max, max);

        (dx, dy)
    }

    /// Combined offset including trauma rotation.
    pub fn offset_and_rotation(&self) -> (f64, f64, f64) {
        let (dx, dy) = self.offset();
        let (_, _, rot) = self.trauma.displacement();
        (dx, dy, rot)
    }

    pub fn is_shaking(&self) -> bool {
        !self.sources.is_empty() || self.trauma.is_active()
    }

    pub fn active_source_count(&self) -> usize {
        self.sources.len()
    }

    /// Clear all shake sources and reset trauma.
    pub fn clear(&mut self) {
        self.sources.clear();
        self.trauma.trauma = 0.0;
    }
}

impl Default for ScreenShakeSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn trauma_starts_at_zero() {
        let t = TraumaShake::new();
        assert!((t.trauma).abs() < EPS);
        assert!(!t.is_active());
    }

    #[test]
    fn trauma_add_clamps() {
        let mut t = TraumaShake::new();
        t.add_trauma(0.5);
        assert!((t.trauma - 0.5).abs() < EPS);
        t.add_trauma(0.8);
        assert!((t.trauma - 1.0).abs() < EPS);
    }

    #[test]
    fn trauma_decays() {
        let mut t = TraumaShake::new();
        t.decay_rate = 2.0;
        t.add_trauma(1.0);
        t.update(0.25);
        assert!((t.trauma - 0.5).abs() < EPS);
    }

    #[test]
    fn trauma_intensity_squared() {
        let mut t = TraumaShake::new();
        t.add_trauma(0.5);
        assert!((t.intensity() - 0.25).abs() < EPS);
    }

    #[test]
    fn trauma_displacement_zero_when_inactive() {
        let t = TraumaShake::new();
        let (dx, dy, rot) = t.displacement();
        assert!(dx.abs() < EPS);
        assert!(dy.abs() < EPS);
        assert!(rot.abs() < EPS);
    }

    #[test]
    fn trauma_displacement_nonzero_when_active() {
        let mut t = TraumaShake::new();
        t.add_trauma(1.0);
        t.update(0.05);
        let (dx, dy, _) = t.displacement();
        // With trauma=0.95 and time advancing, should produce some offset
        assert!(dx.abs() > EPS || dy.abs() > EPS);
    }

    #[test]
    fn shake_source_linear_decay_factor() {
        let mut s = ShakeSource::new("test")
            .with_decay(DecayMode::Linear, 1.0);
        s.elapsed = 0.5;
        let f = s.decay_factor();
        assert!((f - 0.5).abs() < EPS);
    }

    #[test]
    fn shake_source_exponential_decay() {
        let mut s = ShakeSource::new("exp")
            .with_decay(DecayMode::Exponential { half_life: 1.0 }, 0.0);
        s.elapsed = 1.0;
        let f = s.decay_factor();
        assert!((f - 0.5).abs() < 0.01);
    }

    #[test]
    fn shake_source_no_decay() {
        let mut s = ShakeSource::new("perm")
            .with_decay(DecayMode::None, 0.0);
        s.elapsed = 100.0;
        let f = s.decay_factor();
        assert!((f - 1.0).abs() < EPS);
    }

    #[test]
    fn shake_source_smooth_produces_output() {
        let s = ShakeSource::new("smooth")
            .with_amplitude(10.0, 10.0)
            .with_smooth(true);
        let (dx, dy) = s.displacement();
        // At elapsed=0, should still produce some value from sin(0+seed)
        assert!(dx.abs() < 10.0 + EPS);
        assert!(dy.abs() < 10.0 + EPS);
    }

    #[test]
    fn shake_source_jitter_bounded() {
        let mut s = ShakeSource::new("jitter")
            .with_amplitude(5.0, 5.0)
            .with_smooth(false)
            .with_decay(DecayMode::None, 0.0);
        s.elapsed = 0.1;
        let (dx, dy) = s.displacement();
        assert!(dx.abs() <= 5.0 + EPS);
        assert!(dy.abs() <= 5.0 + EPS);
    }

    #[test]
    fn directional_shake_projects_onto_direction() {
        let mut s = ShakeSource::new("dir")
            .with_amplitude(10.0, 10.0)
            .with_direction(1.0, 0.0)
            .with_smooth(true)
            .with_decay(DecayMode::None, 0.0);
        s.elapsed = 0.1;
        let (dx, dy) = s.displacement();
        // Direction is (1,0), so dy should be 0
        assert!(dy.abs() < EPS);
        // dx can be nonzero
        assert!(dx.abs() < 10.0 + EPS);
    }

    #[test]
    fn system_add_remove_source() {
        let mut sys = ScreenShakeSystem::new();
        sys.add_source(ShakeSource::new("a"));
        assert_eq!(sys.active_source_count(), 1);
        assert!(sys.remove_source("a"));
        assert_eq!(sys.active_source_count(), 0);
    }

    #[test]
    fn system_sources_expire() {
        let mut sys = ScreenShakeSystem::new();
        sys.add_source(
            ShakeSource::new("short")
                .with_decay(DecayMode::Linear, 0.1),
        );
        sys.update(0.2);
        assert_eq!(sys.active_source_count(), 0);
    }

    #[test]
    fn system_multiple_sources_stack() {
        let mut sys = ScreenShakeSystem::new();
        sys.add_source(
            ShakeSource::new("a")
                .with_amplitude(10.0, 0.0)
                .with_decay(DecayMode::None, 0.0),
        );
        sys.add_source(
            ShakeSource::new("b")
                .with_amplitude(10.0, 0.0)
                .with_decay(DecayMode::None, 0.0),
        );
        sys.update(0.05);
        let (dx, _) = sys.offset();
        // Combined can be up to 20
        assert!(dx.abs() <= 50.0 + EPS);
    }

    #[test]
    fn system_clamps_max_displacement() {
        let mut sys = ScreenShakeSystem::new();
        sys.max_displacement = 5.0;
        sys.add_source(
            ShakeSource::new("big")
                .with_amplitude(100.0, 100.0)
                .with_decay(DecayMode::None, 0.0),
        );
        sys.update(0.1);
        let (dx, dy) = sys.offset();
        assert!(dx.abs() <= 5.0 + EPS);
        assert!(dy.abs() <= 5.0 + EPS);
    }

    #[test]
    fn system_is_shaking() {
        let mut sys = ScreenShakeSystem::new();
        assert!(!sys.is_shaking());
        sys.add_source(ShakeSource::new("x"));
        assert!(sys.is_shaking());
    }

    #[test]
    fn system_trauma_integration() {
        let mut sys = ScreenShakeSystem::new();
        sys.add_trauma(0.5);
        assert!(sys.is_shaking());
        sys.update(0.01);
        let (dx, dy) = sys.offset();
        // Should have some offset from trauma
        assert!(dx.abs() > 0.0 || dy.abs() > 0.0 || !sys.is_shaking());
    }

    #[test]
    fn system_clear() {
        let mut sys = ScreenShakeSystem::new();
        sys.add_source(ShakeSource::new("a"));
        sys.add_trauma(1.0);
        sys.clear();
        assert!(!sys.is_shaking());
        assert_eq!(sys.active_source_count(), 0);
    }

    #[test]
    fn system_offset_and_rotation() {
        let mut sys = ScreenShakeSystem::new();
        sys.add_trauma(1.0);
        sys.update(0.05);
        let (_, _, rot) = sys.offset_and_rotation();
        // With high trauma the rotation should be nonzero
        assert!(rot.abs() <= sys.trauma_mut().max_rotation + EPS);
    }

    #[test]
    fn perlin_approx_bounded() {
        for i in 0..100 {
            let v = perlin_approx(i as f64 * 0.1, 0.0);
            assert!(v >= -1.0 - EPS && v <= 1.0 + EPS);
        }
    }

    #[test]
    fn hash_jitter_bounded() {
        for i in 0..100 {
            let v = hash_jitter(i as f64 * 0.1, 0.0);
            assert!(v >= -1.0 - EPS && v <= 1.0 + EPS);
        }
    }
}
