// Procedural Terrain Generation — Fractal Brownian Motion (fBm) and related multi-octave noise
// fBm, turbulence, ridged multi-fractal, domain warping

use std::fmt;

/// A pluggable noise source trait for fBm and related functions.
pub trait NoiseSource {
    fn sample2d(&self, x: f64, y: f64) -> f64;
    fn sample3d(&self, x: f64, y: f64, z: f64) -> f64;
}

/// Built-in simple value noise (hash-based) for standalone use.
#[derive(Clone)]
pub struct ValueNoise {
    seed: u64,
}

impl fmt::Debug for ValueNoise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValueNoise").field("seed", &self.seed).finish()
    }
}

impl ValueNoise {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn hash2(x: i64, y: i64, seed: u64) -> f64 {
        let mut h = seed;
        h = h.wrapping_add(x as u64).wrapping_mul(6364136223846793005);
        h = h.wrapping_add(y as u64).wrapping_mul(6364136223846793005);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        // Map to [-1, 1]
        (h as i64 as f64) / (i64::MAX as f64)
    }

    fn hash3(x: i64, y: i64, z: i64, seed: u64) -> f64 {
        let mut h = seed;
        h = h.wrapping_add(x as u64).wrapping_mul(6364136223846793005);
        h = h.wrapping_add(y as u64).wrapping_mul(6364136223846793005);
        h = h.wrapping_add(z as u64).wrapping_mul(6364136223846793005);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        (h as i64 as f64) / (i64::MAX as f64)
    }

    fn smoothstep(t: f64) -> f64 {
        t * t * (3.0 - 2.0 * t)
    }

    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }
}

impl NoiseSource for ValueNoise {
    fn sample2d(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = Self::smoothstep(x - x.floor());
        let fy = Self::smoothstep(y - y.floor());

        let v00 = Self::hash2(ix, iy, self.seed);
        let v10 = Self::hash2(ix + 1, iy, self.seed);
        let v01 = Self::hash2(ix, iy + 1, self.seed);
        let v11 = Self::hash2(ix + 1, iy + 1, self.seed);

        let top = Self::lerp(v00, v10, fx);
        let bot = Self::lerp(v01, v11, fx);
        Self::lerp(top, bot, fy)
    }

    fn sample3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let iz = z.floor() as i64;
        let fx = Self::smoothstep(x - x.floor());
        let fy = Self::smoothstep(y - y.floor());
        let fz = Self::smoothstep(z - z.floor());

        let v000 = Self::hash3(ix, iy, iz, self.seed);
        let v100 = Self::hash3(ix + 1, iy, iz, self.seed);
        let v010 = Self::hash3(ix, iy + 1, iz, self.seed);
        let v110 = Self::hash3(ix + 1, iy + 1, iz, self.seed);
        let v001 = Self::hash3(ix, iy, iz + 1, self.seed);
        let v101 = Self::hash3(ix + 1, iy, iz + 1, self.seed);
        let v011 = Self::hash3(ix, iy + 1, iz + 1, self.seed);
        let v111 = Self::hash3(ix + 1, iy + 1, iz + 1, self.seed);

        let x00 = Self::lerp(v000, v100, fx);
        let x10 = Self::lerp(v010, v110, fx);
        let x01 = Self::lerp(v001, v101, fx);
        let x11 = Self::lerp(v011, v111, fx);

        let y0 = Self::lerp(x00, x10, fy);
        let y1 = Self::lerp(x01, x11, fy);

        Self::lerp(y0, y1, fz)
    }
}

/// Configuration for fractal noise generation.
#[derive(Debug, Clone, PartialEq)]
pub struct FractalConfig {
    pub octaves: u32,
    pub lacunarity: f64,
    pub persistence: f64,
}

impl Default for FractalConfig {
    fn default() -> Self {
        Self {
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.5,
        }
    }
}

impl FractalConfig {
    pub fn new(octaves: u32, lacunarity: f64, persistence: f64) -> Self {
        Self {
            octaves: octaves.clamp(1, 16),
            lacunarity,
            persistence,
        }
    }
}

/// Fractal Brownian Motion generator.
#[derive(Debug, Clone)]
pub struct FractalBrownian {
    noise: ValueNoise,
    config: FractalConfig,
}

impl FractalBrownian {
    pub fn new(seed: u64) -> Self {
        Self {
            noise: ValueNoise::new(seed),
            config: FractalConfig::default(),
        }
    }

    pub fn with_config(mut self, config: FractalConfig) -> Self {
        self.config = FractalConfig::new(config.octaves, config.lacunarity, config.persistence);
        self
    }

    pub fn config(&self) -> &FractalConfig {
        &self.config
    }

    /// Standard fBm: sum of octaves with increasing frequency and decreasing amplitude.
    pub fn fbm2d(&self, x: f64, y: f64) -> f64 {
        self.fbm2d_with(&self.noise, x, y)
    }

    /// fBm using a custom noise source.
    pub fn fbm2d_with<N: NoiseSource>(&self, source: &N, x: f64, y: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut max_amp = 0.0;
        for _ in 0..self.config.octaves {
            total += source.sample2d(x * freq, y * freq) * amp;
            max_amp += amp;
            freq *= self.config.lacunarity;
            amp *= self.config.persistence;
        }
        if max_amp > 0.0 { total / max_amp } else { 0.0 }
    }

    /// 3D fBm.
    pub fn fbm3d(&self, x: f64, y: f64, z: f64) -> f64 {
        self.fbm3d_with(&self.noise, x, y, z)
    }

    pub fn fbm3d_with<N: NoiseSource>(&self, source: &N, x: f64, y: f64, z: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut max_amp = 0.0;
        for _ in 0..self.config.octaves {
            total += source.sample3d(x * freq, y * freq, z * freq) * amp;
            max_amp += amp;
            freq *= self.config.lacunarity;
            amp *= self.config.persistence;
        }
        if max_amp > 0.0 { total / max_amp } else { 0.0 }
    }

    /// Turbulence: sum of absolute value of noise per octave (always positive).
    pub fn turbulence2d(&self, x: f64, y: f64) -> f64 {
        self.turbulence2d_with(&self.noise, x, y)
    }

    pub fn turbulence2d_with<N: NoiseSource>(&self, source: &N, x: f64, y: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut max_amp = 0.0;
        for _ in 0..self.config.octaves {
            total += source.sample2d(x * freq, y * freq).abs() * amp;
            max_amp += amp;
            freq *= self.config.lacunarity;
            amp *= self.config.persistence;
        }
        if max_amp > 0.0 { total / max_amp } else { 0.0 }
    }

    /// 3D turbulence.
    pub fn turbulence3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut max_amp = 0.0;
        for _ in 0..self.config.octaves {
            total += self.noise.sample3d(x * freq, y * freq, z * freq).abs() * amp;
            max_amp += amp;
            freq *= self.config.lacunarity;
            amp *= self.config.persistence;
        }
        if max_amp > 0.0 { total / max_amp } else { 0.0 }
    }

    /// Ridged multi-fractal: 1 - |noise|, squared, weighted by previous octave.
    pub fn ridged2d(&self, x: f64, y: f64) -> f64 {
        self.ridged2d_with(&self.noise, x, y)
    }

    pub fn ridged2d_with<N: NoiseSource>(&self, source: &N, x: f64, y: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut weight = 1.0;
        let mut max_val = 0.0;
        for _ in 0..self.config.octaves {
            let signal = source.sample2d(x * freq, y * freq);
            let ridge = 1.0 - signal.abs();
            let ridge_sq = ridge * ridge;
            let weighted = ridge_sq * weight;
            total += weighted * amp;
            max_val += amp;
            weight = (ridge_sq).clamp(0.0, 1.0);
            freq *= self.config.lacunarity;
            amp *= self.config.persistence;
        }
        if max_val > 0.0 { total / max_val } else { 0.0 }
    }

    /// 3D ridged multi-fractal.
    pub fn ridged3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut weight = 1.0;
        let mut max_val = 0.0;
        for _ in 0..self.config.octaves {
            let signal = self.noise.sample3d(x * freq, y * freq, z * freq);
            let ridge = 1.0 - signal.abs();
            let ridge_sq = ridge * ridge;
            let weighted = ridge_sq * weight;
            total += weighted * amp;
            max_val += amp;
            weight = ridge_sq.clamp(0.0, 1.0);
            freq *= self.config.lacunarity;
            amp *= self.config.persistence;
        }
        if max_val > 0.0 { total / max_val } else { 0.0 }
    }

    /// Domain warping: use noise to offset input coordinates, then sample again.
    pub fn domain_warp2d(&self, x: f64, y: f64, warp_strength: f64) -> f64 {
        self.domain_warp2d_with(&self.noise, x, y, warp_strength)
    }

    pub fn domain_warp2d_with<N: NoiseSource>(
        &self,
        source: &N,
        x: f64,
        y: f64,
        warp_strength: f64,
    ) -> f64 {
        let ox = self.fbm2d_with(source, x, y) * warp_strength;
        let oy = self.fbm2d_with(source, x + 5.2, y + 1.3) * warp_strength;
        self.fbm2d_with(source, x + ox, y + oy)
    }

    /// Multi-level domain warping (warp the warp).
    pub fn domain_warp2d_multi(&self, x: f64, y: f64, warp_strength: f64, levels: u32) -> f64 {
        let lvl = levels.clamp(1, 4);
        let mut cx = x;
        let mut cy = y;
        for _ in 0..lvl {
            let ox = self.fbm2d(cx, cy) * warp_strength;
            let oy = self.fbm2d(cx + 5.2, cy + 1.3) * warp_strength;
            cx = x + ox;
            cy = y + oy;
        }
        self.fbm2d(cx, cy)
    }

    /// Generate a 2D fBm heightmap.
    pub fn generate_heightmap(&self, width: usize, height: usize, scale: f64) -> Vec<Vec<f64>> {
        let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| self.fbm2d(x as f64 / s, y as f64 / s))
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_noise_deterministic() {
        let a = ValueNoise::new(42);
        let b = ValueNoise::new(42);
        assert!((a.sample2d(1.5, 2.5) - b.sample2d(1.5, 2.5)).abs() < 1e-12);
    }

    #[test]
    fn test_value_noise_range() {
        let n = ValueNoise::new(42);
        for i in 0..200 {
            let x = i as f64 * 0.17 - 15.0;
            let y = i as f64 * 0.31 - 20.0;
            let v = n.sample2d(x, y);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "value out of range: {v}");
        }
    }

    #[test]
    fn test_value_noise_3d_range() {
        let n = ValueNoise::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.23 - 10.0;
            let v = n.sample3d(t, t * 0.7, t * 1.3);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_fbm2d_range() {
        let fbm = FractalBrownian::new(42);
        for i in 0..200 {
            let x = i as f64 * 0.13;
            let y = i as f64 * 0.29;
            let v = fbm.fbm2d(x, y);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "fbm out of range: {v}");
        }
    }

    #[test]
    fn test_fbm3d_range() {
        let fbm = FractalBrownian::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.19;
            let v = fbm.fbm3d(t, t * 0.7, t * 1.3);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_turbulence_non_negative() {
        let fbm = FractalBrownian::new(42);
        for i in 0..100 {
            let x = i as f64 * 0.23;
            let y = i as f64 * 0.37;
            let v = fbm.turbulence2d(x, y);
            assert!(v >= -1e-12, "turbulence should be non-negative: {v}");
        }
    }

    #[test]
    fn test_turbulence3d_non_negative() {
        let fbm = FractalBrownian::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.19;
            let v = fbm.turbulence3d(t, t * 0.7, t * 1.3);
            assert!(v >= -1e-12);
        }
    }

    #[test]
    fn test_ridged_non_negative() {
        let fbm = FractalBrownian::new(42);
        for i in 0..100 {
            let x = i as f64 * 0.23;
            let y = i as f64 * 0.37;
            let v = fbm.ridged2d(x, y);
            assert!(v >= -1e-6, "ridged should be non-negative: {v}");
        }
    }

    #[test]
    fn test_ridged3d_non_negative() {
        let fbm = FractalBrownian::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.19;
            let v = fbm.ridged3d(t, t * 0.7, t * 1.3);
            assert!(v >= -1e-6);
        }
    }

    #[test]
    fn test_domain_warp_finite() {
        let fbm = FractalBrownian::new(42);
        for i in 0..50 {
            let x = i as f64 * 0.23;
            let y = i as f64 * 0.37;
            let v = fbm.domain_warp2d(x, y, 4.0);
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_domain_warp_multi_levels() {
        let fbm = FractalBrownian::new(42);
        let v1 = fbm.domain_warp2d_multi(1.5, 2.5, 2.0, 1);
        let v2 = fbm.domain_warp2d_multi(1.5, 2.5, 2.0, 3);
        // Different warping levels should generally produce different values
        assert!(v1.is_finite());
        assert!(v2.is_finite());
    }

    #[test]
    fn test_config_defaults() {
        let cfg = FractalConfig::default();
        assert_eq!(cfg.octaves, 6);
        assert!((cfg.lacunarity - 2.0).abs() < 1e-12);
        assert!((cfg.persistence - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_config_clamp_octaves() {
        let cfg = FractalConfig::new(0, 2.0, 0.5);
        assert_eq!(cfg.octaves, 1);
        let cfg2 = FractalConfig::new(100, 2.0, 0.5);
        assert_eq!(cfg2.octaves, 16);
    }

    #[test]
    fn test_with_config() {
        let fbm = FractalBrownian::new(42)
            .with_config(FractalConfig::new(4, 2.5, 0.3));
        assert_eq!(fbm.config().octaves, 4);
        assert!((fbm.config().lacunarity - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_single_octave_equals_noise() {
        let fbm = FractalBrownian::new(42)
            .with_config(FractalConfig::new(1, 2.0, 0.5));
        let vn = ValueNoise::new(42);
        let fbm_val = fbm.fbm2d(1.5, 2.5);
        let noise_val = vn.sample2d(1.5, 2.5);
        assert!((fbm_val - noise_val).abs() < 1e-10);
    }

    #[test]
    fn test_generate_heightmap_dimensions() {
        let fbm = FractalBrownian::new(42);
        let hm = fbm.generate_heightmap(16, 8, 4.0);
        assert_eq!(hm.len(), 8);
        for row in &hm {
            assert_eq!(row.len(), 16);
        }
    }

    #[test]
    fn test_generate_heightmap_values_bounded() {
        let fbm = FractalBrownian::new(42);
        let hm = fbm.generate_heightmap(32, 32, 8.0);
        for row in &hm {
            for &v in row {
                assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
            }
        }
    }

    #[test]
    fn test_deterministic_fbm() {
        let a = FractalBrownian::new(42);
        let b = FractalBrownian::new(42);
        assert!((a.fbm2d(3.7, 4.2) - b.fbm2d(3.7, 4.2)).abs() < 1e-12);
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = FractalBrownian::new(1);
        let b = FractalBrownian::new(999);
        assert!((a.fbm2d(3.7, 4.2) - b.fbm2d(3.7, 4.2)).abs() > 1e-9);
    }

    #[test]
    fn test_spatial_variation() {
        let fbm = FractalBrownian::new(42);
        let mut distinct = 0;
        let base = fbm.fbm2d(0.5, 0.5);
        for i in 1..20 {
            let v = fbm.fbm2d(0.5 + i as f64 * 0.7, 0.5 + i as f64 * 0.3);
            if (v - base).abs() > 1e-6 { distinct += 1; }
        }
        assert!(distinct > 10);
    }

    #[test]
    fn test_value_noise_debug() {
        let n = ValueNoise::new(42);
        let s = format!("{:?}", n);
        assert!(s.contains("ValueNoise"));
    }

    #[test]
    fn test_fractal_config_partial_eq() {
        let a = FractalConfig::new(6, 2.0, 0.5);
        let b = FractalConfig::new(6, 2.0, 0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn test_negative_coordinates() {
        let fbm = FractalBrownian::new(42);
        let v = fbm.fbm2d(-5.3, -8.7);
        assert!(v.is_finite());
        let vt = fbm.turbulence2d(-3.0, -4.0);
        assert!(vt.is_finite() && vt >= -1e-12);
        let vr = fbm.ridged2d(-3.0, -4.0);
        assert!(vr.is_finite());
    }
}
