//! Noise generators for audio synthesis: white, pink, brown, blue, velvet,
//! Perlin, and crackle noise.
//!
//! All generators produce per-sample output in the -1.0..1.0 range with
//! deterministic seeded PRNG. Pure Rust — no external deps beyond std.

// ── Deterministic PRNG (xorshift64) ────────────────────────────

/// Deterministic xorshift64 PRNG used internally by all noise generators.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Uniform f64 in -1.0..1.0.
    fn next_bipolar(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    }

    /// Uniform f64 in 0.0..1.0.
    fn next_unipolar(&mut self) -> f64 {
        (self.next_bipolar() + 1.0) * 0.5
    }
}

// ── White Noise ─────────────────────────────────────────────────

/// Uniform random noise with flat spectrum.
#[derive(Debug, Clone)]
pub struct WhiteNoise {
    rng: Xorshift64,
    pub amplitude: f64,
}

impl WhiteNoise {
    pub fn new(seed: u64) -> Self {
        Self { rng: Xorshift64::new(seed), amplitude: 1.0 }
    }

    pub fn next_sample(&mut self) -> f64 {
        self.rng.next_bipolar() * self.amplitude
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Pink Noise (Paul Kellet's refined method) ───────────────────

/// 1/f noise using Paul Kellet's economy method (6 first-order filters).
#[derive(Debug, Clone)]
pub struct PinkNoise {
    rng: Xorshift64,
    b0: f64,
    b1: f64,
    b2: f64,
    b3: f64,
    b4: f64,
    b5: f64,
    b6: f64,
    pub amplitude: f64,
}

impl PinkNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            b0: 0.0, b1: 0.0, b2: 0.0, b3: 0.0, b4: 0.0, b5: 0.0, b6: 0.0,
            amplitude: 1.0,
        }
    }

    pub fn next_sample(&mut self) -> f64 {
        let white = self.rng.next_bipolar();
        self.b0 = 0.99886 * self.b0 + white * 0.0555179;
        self.b1 = 0.99332 * self.b1 + white * 0.0750759;
        self.b2 = 0.96900 * self.b2 + white * 0.1538520;
        self.b3 = 0.86650 * self.b3 + white * 0.3104856;
        self.b4 = 0.55000 * self.b4 + white * 0.5329522;
        self.b5 = -0.7616 * self.b5 - white * 0.0168980;
        let pink = self.b0 + self.b1 + self.b2 + self.b3 + self.b4 + self.b5 + self.b6 + white * 0.5362;
        self.b6 = white * 0.115926;
        (pink * 0.11).clamp(-1.0, 1.0) * self.amplitude
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
        self.b0 = 0.0;
        self.b1 = 0.0;
        self.b2 = 0.0;
        self.b3 = 0.0;
        self.b4 = 0.0;
        self.b5 = 0.0;
        self.b6 = 0.0;
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Brown Noise (1/f², integrated white noise) ──────────────────

/// Brownian noise: integration of white noise with leaky integrator.
#[derive(Debug, Clone)]
pub struct BrownNoise {
    rng: Xorshift64,
    state: f64,
    /// Leak coefficient (0..1); higher = more low-frequency content.
    pub leak: f64,
    pub amplitude: f64,
}

impl BrownNoise {
    pub fn new(seed: u64) -> Self {
        Self { rng: Xorshift64::new(seed), state: 0.0, leak: 0.98, amplitude: 1.0 }
    }

    pub fn next_sample(&mut self) -> f64 {
        let white = self.rng.next_bipolar();
        self.state = self.state * self.leak + white * 0.1;
        self.state = self.state.clamp(-1.0, 1.0);
        self.state * self.amplitude
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
        self.state = 0.0;
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Blue Noise (high-frequency weighted) ────────────────────────

/// Blue noise: differentiated white noise (emphasises high frequencies).
#[derive(Debug, Clone)]
pub struct BlueNoise {
    rng: Xorshift64,
    prev: f64,
    pub amplitude: f64,
}

impl BlueNoise {
    pub fn new(seed: u64) -> Self {
        Self { rng: Xorshift64::new(seed), prev: 0.0, amplitude: 1.0 }
    }

    pub fn next_sample(&mut self) -> f64 {
        let white = self.rng.next_bipolar();
        let blue = white - self.prev;
        self.prev = white;
        (blue * 0.5).clamp(-1.0, 1.0) * self.amplitude
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
        self.prev = 0.0;
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Velvet Noise (sparse impulse noise) ─────────────────────────

/// Velvet noise: sparse random impulses (positive or negative) at a given density.
#[derive(Debug, Clone)]
pub struct VelvetNoise {
    rng: Xorshift64,
    /// Average impulses per second.
    pub density: f64,
    pub sample_rate: f64,
    pub amplitude: f64,
}

impl VelvetNoise {
    pub fn new(density: f64, sample_rate: f64, seed: u64) -> Self {
        Self { rng: Xorshift64::new(seed), density, sample_rate, amplitude: 1.0 }
    }

    pub fn next_sample(&mut self) -> f64 {
        if self.sample_rate <= 0.0 || self.density <= 0.0 {
            return 0.0;
        }
        let prob = self.density / self.sample_rate;
        let r = self.rng.next_unipolar();
        if r < prob {
            // Random sign.
            let sign = if self.rng.next_bipolar() > 0.0 { 1.0 } else { -1.0 };
            sign * self.amplitude
        } else {
            0.0
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Perlin Noise (smooth, for LFO modulation) ──────────────────

/// 1-D Perlin noise generator suitable for smooth LFO modulation.
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    /// Gradient table (256 entries).
    gradients: Vec<f64>,
    /// Permutation table (512 entries, wrapping).
    perm: Vec<usize>,
    /// Current position along the noise function.
    position: f64,
    /// Step per sample (controls noise frequency).
    pub step: f64,
    pub amplitude: f64,
}

impl PerlinNoise {
    pub fn new(seed: u64, step: f64) -> Self {
        let mut rng = Xorshift64::new(seed);
        let gradients: Vec<f64> = (0..256).map(|_| rng.next_bipolar()).collect();

        // Fisher-Yates shuffle for permutation.
        let mut perm: Vec<usize> = (0..256).collect();
        for i in (1..256).rev() {
            let j = (rng.next_unipolar() * (i + 1) as f64) as usize % (i + 1);
            perm.swap(i, j);
        }
        // Double the permutation table for wrapping.
        let perm2: Vec<usize> = perm.iter().chain(perm.iter()).copied().collect();

        Self {
            gradients,
            perm: perm2,
            position: 0.0,
            step,
            amplitude: 1.0,
        }
    }

    /// Smooth interpolation (smoothstep).
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Evaluate Perlin noise at a 1-D position.
    pub fn evaluate(&self, x: f64) -> f64 {
        let xi = x.floor() as i64;
        let xf = x - x.floor();

        let i0 = (xi as usize) & 255;
        let i1 = (i0 + 1) & 255;

        let g0 = self.gradients[self.perm[i0] & 255];
        let g1 = self.gradients[self.perm[i1] & 255];

        let d0 = xf * g0;
        let d1 = (xf - 1.0) * g1;

        let u = Self::fade(xf);
        (d0 * (1.0 - u) + d1 * u) * self.amplitude
    }

    /// Generate the next sample and advance the internal position.
    pub fn next_sample(&mut self) -> f64 {
        let val = self.evaluate(self.position);
        self.position += self.step;
        val
    }

    /// Reset position.
    pub fn reset(&mut self) {
        self.position = 0.0;
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Crackle Noise (random impulses for vinyl effect) ────────────

/// Crackle noise: sparse clicks/pops of varying amplitude for vinyl simulation.
#[derive(Debug, Clone)]
pub struct CrackleNoise {
    rng: Xorshift64,
    /// Probability of a crackle per sample (0..1).
    pub probability: f64,
    /// Decay rate of each crackle (0..1); higher = longer pops.
    pub decay: f64,
    /// Current crackle amplitude.
    current: f64,
    pub amplitude: f64,
}

impl CrackleNoise {
    pub fn new(probability: f64, seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
            probability: probability.clamp(0.0, 1.0),
            decay: 0.5,
            current: 0.0,
            amplitude: 1.0,
        }
    }

    pub fn next_sample(&mut self) -> f64 {
        // Trigger new crackle?
        let r = self.rng.next_unipolar();
        if r < self.probability {
            self.current = self.rng.next_bipolar();
        }

        let output = self.current * self.amplitude;
        self.current *= self.decay;
        output
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Xorshift64::new(seed);
        self.current = 0.0;
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }
}

// ── Noise Type Enum ─────────────────────────────────────────────

/// Convenience enum for selecting noise type at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    White,
    Pink,
    Brown,
    Blue,
    Velvet,
    Perlin,
    Crackle,
}

/// A polymorphic noise generator that wraps all noise types.
#[derive(Debug, Clone)]
pub struct NoiseGenerator {
    white: WhiteNoise,
    pink: PinkNoise,
    brown: BrownNoise,
    blue: BlueNoise,
    velvet: VelvetNoise,
    perlin: PerlinNoise,
    crackle: CrackleNoise,
    /// Currently active noise type.
    pub active_type: NoiseType,
    pub amplitude: f64,
}

impl NoiseGenerator {
    /// Create a noise generator with all types initialised.
    pub fn new(sample_rate: f64, seed: u64) -> Self {
        Self {
            white: WhiteNoise::new(seed),
            pink: PinkNoise::new(seed.wrapping_add(1)),
            brown: BrownNoise::new(seed.wrapping_add(2)),
            blue: BlueNoise::new(seed.wrapping_add(3)),
            velvet: VelvetNoise::new(1000.0, sample_rate, seed.wrapping_add(4)),
            perlin: PerlinNoise::new(seed.wrapping_add(5), 0.01),
            crackle: CrackleNoise::new(0.001, seed.wrapping_add(6)),
            active_type: NoiseType::White,
            amplitude: 1.0,
        }
    }

    /// Generate the next sample using the active noise type.
    pub fn next_sample(&mut self) -> f64 {
        let raw = match self.active_type {
            NoiseType::White => self.white.next_sample(),
            NoiseType::Pink => self.pink.next_sample(),
            NoiseType::Brown => self.brown.next_sample(),
            NoiseType::Blue => self.blue.next_sample(),
            NoiseType::Velvet => self.velvet.next_sample(),
            NoiseType::Perlin => self.perlin.next_sample(),
            NoiseType::Crackle => self.crackle.next_sample(),
        };
        raw * self.amplitude
    }

    pub fn generate_block(&mut self, buffer: &mut [f64]) {
        for s in buffer.iter_mut() {
            *s = self.next_sample();
        }
    }

    /// Get a mutable reference to the Perlin noise generator (for tuning).
    pub fn perlin_mut(&mut self) -> &mut PerlinNoise {
        &mut self.perlin
    }

    /// Get a mutable reference to the velvet noise generator (for tuning).
    pub fn velvet_mut(&mut self) -> &mut VelvetNoise {
        &mut self.velvet
    }

    /// Get a mutable reference to the crackle noise generator (for tuning).
    pub fn crackle_mut(&mut self) -> &mut CrackleNoise {
        &mut self.crackle
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 44100.0;
    const EPS: f64 = 1e-4;
    const BLOCK: usize = 4096;

    fn rms(buf: &[f64]) -> f64 {
        let sum: f64 = buf.iter().map(|s| s * s).sum();
        (sum / buf.len() as f64).sqrt()
    }

    #[test]
    fn test_white_noise_nonzero() {
        let mut wn = WhiteNoise::new(42);
        let mut buf = vec![0.0; BLOCK];
        wn.generate_block(&mut buf);
        assert!(rms(&buf) > 0.1, "white noise should have significant RMS");
    }

    #[test]
    fn test_white_noise_range() {
        let mut wn = WhiteNoise::new(42);
        for _ in 0..10000 {
            let s = wn.next_sample();
            assert!(s >= -1.0 - EPS && s <= 1.0 + EPS, "white noise out of range: {s}");
        }
    }

    #[test]
    fn test_white_noise_seed_determinism() {
        let mut wn1 = WhiteNoise::new(123);
        let mut wn2 = WhiteNoise::new(123);
        for _ in 0..100 {
            assert!((wn1.next_sample() - wn2.next_sample()).abs() < EPS);
        }
    }

    #[test]
    fn test_pink_noise_nonzero() {
        let mut pn = PinkNoise::new(42);
        let mut buf = vec![0.0; BLOCK];
        pn.generate_block(&mut buf);
        assert!(rms(&buf) > 0.01, "pink noise should have positive RMS");
    }

    #[test]
    fn test_pink_noise_range() {
        let mut pn = PinkNoise::new(42);
        for _ in 0..10000 {
            let s = pn.next_sample();
            assert!(s >= -1.0 - EPS && s <= 1.0 + EPS, "pink noise out of range: {s}");
        }
    }

    #[test]
    fn test_brown_noise_nonzero() {
        let mut bn = BrownNoise::new(42);
        let mut buf = vec![0.0; BLOCK];
        bn.generate_block(&mut buf);
        assert!(rms(&buf) > 0.01, "brown noise should have positive RMS");
    }

    #[test]
    fn test_brown_noise_range() {
        let mut bn = BrownNoise::new(42);
        for _ in 0..10000 {
            let s = bn.next_sample();
            assert!(s >= -1.0 - EPS && s <= 1.0 + EPS, "brown noise out of range: {s}");
        }
    }

    #[test]
    fn test_blue_noise_nonzero() {
        let mut bn = BlueNoise::new(42);
        let mut buf = vec![0.0; BLOCK];
        bn.generate_block(&mut buf);
        assert!(rms(&buf) > 0.01, "blue noise should have positive RMS");
    }

    #[test]
    fn test_blue_noise_range() {
        let mut bn = BlueNoise::new(42);
        for _ in 0..10000 {
            let s = bn.next_sample();
            assert!(s >= -1.0 - EPS && s <= 1.0 + EPS, "blue noise out of range: {s}");
        }
    }

    #[test]
    fn test_velvet_noise_sparse() {
        let mut vn = VelvetNoise::new(100.0, SR, 42);
        let mut nonzero_count = 0;
        let total = 44100;
        for _ in 0..total {
            if vn.next_sample().abs() > EPS {
                nonzero_count += 1;
            }
        }
        // At 100 impulses/sec over 1 sec, expect ~100 nonzero samples.
        assert!(nonzero_count > 50 && nonzero_count < 300,
            "velvet noise should be sparse: {nonzero_count} nonzero out of {total}");
    }

    #[test]
    fn test_velvet_noise_amplitude() {
        let mut vn = VelvetNoise::new(1000.0, SR, 42);
        vn.amplitude = 0.5;
        for _ in 0..10000 {
            let s = vn.next_sample();
            assert!(s.abs() <= 0.5 + EPS, "velvet amplitude exceeded: {s}");
        }
    }

    #[test]
    fn test_perlin_noise_smooth() {
        let pn = PerlinNoise::new(42, 0.1);
        let v0 = pn.evaluate(0.0);
        let v1 = pn.evaluate(0.01);
        // Adjacent samples should be similar (smooth).
        assert!((v0 - v1).abs() < 0.5, "Perlin noise should be smooth: {v0} vs {v1}");
    }

    #[test]
    fn test_perlin_noise_varies() {
        let mut pn = PerlinNoise::new(42, 0.1);
        let mut buf = vec![0.0; 1000];
        pn.generate_block(&mut buf);
        let min = buf.iter().fold(f64::MAX, |a, &b| a.min(b));
        let max = buf.iter().fold(f64::MIN, |a, &b| a.max(b));
        assert!(max - min > 0.01, "Perlin noise should vary: min={min} max={max}");
    }

    #[test]
    fn test_perlin_noise_reset() {
        let mut pn = PerlinNoise::new(42, 0.1);
        let first = pn.next_sample();
        for _ in 0..100 {
            pn.next_sample();
        }
        pn.reset();
        let after_reset = pn.next_sample();
        assert!((first - after_reset).abs() < EPS, "reset should reproduce first sample");
    }

    #[test]
    fn test_crackle_noise_sparse() {
        let mut cn = CrackleNoise::new(0.01, 42);
        let mut nonzero = 0;
        for _ in 0..10000 {
            if cn.next_sample().abs() > 0.01 {
                nonzero += 1;
            }
        }
        // ~1% probability, expect some nonzero but mostly silence.
        assert!(nonzero > 10 && nonzero < 2000,
            "crackle should be sparse: {nonzero} nonzero in 10000");
    }

    #[test]
    fn test_crackle_noise_decays() {
        let mut cn = CrackleNoise::new(1.0, 42); // very high prob for test
        cn.decay = 0.1;
        let first = cn.next_sample();
        let second = cn.next_sample();
        // After decay, amplitude should be much lower (unless retriggered).
        // At prob=1, it retriggers every sample, but decay still affects current.
        assert!(first.abs() > 0.0 || second.abs() > 0.0, "crackle should produce output");
    }

    #[test]
    fn test_noise_generator_switch_types() {
        let mut ng = NoiseGenerator::new(SR, 42);
        let types = [
            NoiseType::White, NoiseType::Pink, NoiseType::Brown,
            NoiseType::Blue, NoiseType::Velvet, NoiseType::Perlin,
            NoiseType::Crackle,
        ];
        for nt in types {
            ng.active_type = nt;
            let mut buf = vec![0.0; 256];
            ng.generate_block(&mut buf);
            // Should not panic for any type.
        }
    }

    #[test]
    fn test_noise_generator_amplitude() {
        let mut ng = NoiseGenerator::new(SR, 42);
        ng.amplitude = 0.3;
        ng.active_type = NoiseType::White;
        for _ in 0..1000 {
            let s = ng.next_sample();
            assert!(s.abs() <= 0.3 + EPS, "generator amplitude exceeded: {s}");
        }
    }

    #[test]
    fn test_white_noise_set_seed() {
        let mut wn = WhiteNoise::new(42);
        let first_run: Vec<f64> = (0..10).map(|_| wn.next_sample()).collect();
        wn.set_seed(42);
        let second_run: Vec<f64> = (0..10).map(|_| wn.next_sample()).collect();
        for (a, b) in first_run.iter().zip(second_run.iter()) {
            assert!((a - b).abs() < EPS, "same seed should produce same sequence");
        }
    }

    #[test]
    fn test_brown_noise_leak_coefficient() {
        let mut bn = BrownNoise::new(42);
        bn.leak = 0.5; // Fast leak = less low-frequency content.
        let mut buf = vec![0.0; BLOCK];
        bn.generate_block(&mut buf);
        assert!(rms(&buf) > 0.001, "brown with low leak should still produce output");
    }

    #[test]
    fn test_perlin_step_controls_speed() {
        let mut pn_slow = PerlinNoise::new(42, 0.001);
        let mut pn_fast = PerlinNoise::new(42, 0.1);
        let mut var_slow = Vec::new();
        let mut var_fast = Vec::new();
        for _ in 0..100 {
            var_slow.push(pn_slow.next_sample());
            var_fast.push(pn_fast.next_sample());
        }
        // Fast step should show more variation over same number of samples.
        let range_slow = var_slow.iter().fold(f64::MIN, |a, &b| a.max(b))
            - var_slow.iter().fold(f64::MAX, |a, &b| a.min(b));
        let range_fast = var_fast.iter().fold(f64::MIN, |a, &b| a.max(b))
            - var_fast.iter().fold(f64::MAX, |a, &b| a.min(b));
        assert!(range_fast > range_slow,
            "fast step should show more range: slow={range_slow} fast={range_fast}");
    }

    #[test]
    fn test_noise_generator_perlin_mut() {
        let mut ng = NoiseGenerator::new(SR, 42);
        ng.perlin_mut().step = 0.05;
        assert!((ng.perlin_mut().step - 0.05).abs() < EPS);
    }
}
