// Procedural Terrain Generation — Classic Perlin noise
// Permutation table, gradient vectors, fade/interpolation, 2D/3D noise, tiling, octave layering

use std::fmt;

// ── 12 gradient vectors for 3D Perlin noise ──────────────────────────
const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

/// Perlin noise generator with seeded permutation table.
#[derive(Clone)]
pub struct PerlinNoise {
    perm: [u8; 512],
    seed: u64,
}

impl fmt::Debug for PerlinNoise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerlinNoise")
            .field("seed", &self.seed)
            .finish()
    }
}

impl PerlinNoise {
    /// Create a new Perlin noise generator from a seed.
    pub fn new(seed: u64) -> Self {
        let mut table: [u8; 256] = [0; 256];
        for i in 0..256 {
            table[i] = i as u8;
        }
        // Fisher-Yates shuffle seeded by a simple LCG
        let mut rng = seed;
        for i in (1..256).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            table.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = table[i & 255];
        }
        Self { perm, seed }
    }

    /// Return the seed used for this generator.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Fade curve: 6t^5 - 15t^4 + 10t^3
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    /// Linear interpolation.
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }

    /// Dot product of gradient and distance vector (2D).
    fn grad2(&self, hash: usize, x: f64, y: f64) -> f64 {
        let g = &GRAD3[hash % 12];
        g[0] * x + g[1] * y
    }

    /// Dot product of gradient and distance vector (3D).
    fn grad3(&self, hash: usize, x: f64, y: f64, z: f64) -> f64 {
        let g = &GRAD3[hash % 12];
        g[0] * x + g[1] * y + g[2] * z
    }

    fn hash_index(&self, i: usize) -> usize {
        self.perm[i & 511] as usize
    }

    /// 2D Perlin noise in [-1, 1].
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        let xi = x.floor() as i64;
        let yi = y.floor() as i64;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = Self::fade(xf);
        let v = Self::fade(yf);

        let ix = (xi & 255) as usize;
        let iy = (yi & 255) as usize;

        let aa = self.hash_index(self.hash_index(ix) + iy);
        let ab = self.hash_index(self.hash_index(ix) + iy + 1);
        let ba = self.hash_index(self.hash_index(ix + 1) + iy);
        let bb = self.hash_index(self.hash_index(ix + 1) + iy + 1);

        let x1 = Self::lerp(self.grad2(aa, xf, yf), self.grad2(ba, xf - 1.0, yf), u);
        let x2 = Self::lerp(
            self.grad2(ab, xf, yf - 1.0),
            self.grad2(bb, xf - 1.0, yf - 1.0),
            u,
        );
        Self::lerp(x1, x2, v)
    }

    /// 3D Perlin noise in [-1, 1].
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let xi = x.floor() as i64;
        let yi = y.floor() as i64;
        let zi = z.floor() as i64;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = Self::fade(xf);
        let v = Self::fade(yf);
        let w = Self::fade(zf);

        let ix = (xi & 255) as usize;
        let iy = (yi & 255) as usize;
        let iz = (zi & 255) as usize;

        let aaa = self.hash_index(self.hash_index(self.hash_index(ix) + iy) + iz);
        let aab = self.hash_index(self.hash_index(self.hash_index(ix) + iy) + iz + 1);
        let aba = self.hash_index(self.hash_index(self.hash_index(ix) + iy + 1) + iz);
        let abb = self.hash_index(self.hash_index(self.hash_index(ix) + iy + 1) + iz + 1);
        let baa = self.hash_index(self.hash_index(self.hash_index(ix + 1) + iy) + iz);
        let bab = self.hash_index(self.hash_index(self.hash_index(ix + 1) + iy) + iz + 1);
        let bba = self.hash_index(self.hash_index(self.hash_index(ix + 1) + iy + 1) + iz);
        let bbb = self.hash_index(self.hash_index(self.hash_index(ix + 1) + iy + 1) + iz + 1);

        let x1 = Self::lerp(
            self.grad3(aaa, xf, yf, zf),
            self.grad3(baa, xf - 1.0, yf, zf),
            u,
        );
        let x2 = Self::lerp(
            self.grad3(aba, xf, yf - 1.0, zf),
            self.grad3(bba, xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = Self::lerp(x1, x2, v);

        let x3 = Self::lerp(
            self.grad3(aab, xf, yf, zf - 1.0),
            self.grad3(bab, xf - 1.0, yf, zf - 1.0),
            u,
        );
        let x4 = Self::lerp(
            self.grad3(abb, xf, yf - 1.0, zf - 1.0),
            self.grad3(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = Self::lerp(x3, x4, v);

        Self::lerp(y1, y2, w)
    }

    /// 2D tiling / periodic noise: wraps coordinates at `period_x` x `period_y`.
    pub fn noise2d_tiling(&self, x: f64, y: f64, period_x: u32, period_y: u32) -> f64 {
        let px = period_x.max(1) as usize;
        let py = period_y.max(1) as usize;

        let xi = x.floor() as i64;
        let yi = y.floor() as i64;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = Self::fade(xf);
        let v = Self::fade(yf);

        let wrap_x = |val: i64| (val.rem_euclid(px as i64)) as usize;
        let wrap_y = |val: i64| (val.rem_euclid(py as i64)) as usize;

        let ix0 = wrap_x(xi);
        let ix1 = wrap_x(xi + 1);
        let iy0 = wrap_y(yi);
        let iy1 = wrap_y(yi + 1);

        let aa = self.hash_index(self.hash_index(ix0) + iy0);
        let ab = self.hash_index(self.hash_index(ix0) + iy1);
        let ba = self.hash_index(self.hash_index(ix1) + iy0);
        let bb = self.hash_index(self.hash_index(ix1) + iy1);

        let x1 = Self::lerp(self.grad2(aa, xf, yf), self.grad2(ba, xf - 1.0, yf), u);
        let x2 = Self::lerp(
            self.grad2(ab, xf, yf - 1.0),
            self.grad2(bb, xf - 1.0, yf - 1.0),
            u,
        );
        Self::lerp(x1, x2, v)
    }

    /// 3D tiling / periodic noise.
    pub fn noise3d_tiling(
        &self,
        x: f64,
        y: f64,
        z: f64,
        px: u32,
        py: u32,
        pz: u32,
    ) -> f64 {
        let ppx = px.max(1) as usize;
        let ppy = py.max(1) as usize;
        let ppz = pz.max(1) as usize;

        let xi = x.floor() as i64;
        let yi = y.floor() as i64;
        let zi = z.floor() as i64;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = Self::fade(xf);
        let v = Self::fade(yf);
        let w = Self::fade(zf);

        let wx = |val: i64| (val.rem_euclid(ppx as i64)) as usize;
        let wy = |val: i64| (val.rem_euclid(ppy as i64)) as usize;
        let wz = |val: i64| (val.rem_euclid(ppz as i64)) as usize;

        let ix0 = wx(xi);
        let ix1 = wx(xi + 1);
        let iy0 = wy(yi);
        let iy1 = wy(yi + 1);
        let iz0 = wz(zi);
        let iz1 = wz(zi + 1);

        let aaa = self.hash_index(self.hash_index(self.hash_index(ix0) + iy0) + iz0);
        let aab = self.hash_index(self.hash_index(self.hash_index(ix0) + iy0) + iz1);
        let aba = self.hash_index(self.hash_index(self.hash_index(ix0) + iy1) + iz0);
        let abb = self.hash_index(self.hash_index(self.hash_index(ix0) + iy1) + iz1);
        let baa = self.hash_index(self.hash_index(self.hash_index(ix1) + iy0) + iz0);
        let bab = self.hash_index(self.hash_index(self.hash_index(ix1) + iy0) + iz1);
        let bba = self.hash_index(self.hash_index(self.hash_index(ix1) + iy1) + iz0);
        let bbb = self.hash_index(self.hash_index(self.hash_index(ix1) + iy1) + iz1);

        let l1 = Self::lerp(
            self.grad3(aaa, xf, yf, zf),
            self.grad3(baa, xf - 1.0, yf, zf),
            u,
        );
        let l2 = Self::lerp(
            self.grad3(aba, xf, yf - 1.0, zf),
            self.grad3(bba, xf - 1.0, yf - 1.0, zf),
            u,
        );
        let y1 = Self::lerp(l1, l2, v);

        let l3 = Self::lerp(
            self.grad3(aab, xf, yf, zf - 1.0),
            self.grad3(bab, xf - 1.0, yf, zf - 1.0),
            u,
        );
        let l4 = Self::lerp(
            self.grad3(abb, xf, yf - 1.0, zf - 1.0),
            self.grad3(bbb, xf - 1.0, yf - 1.0, zf - 1.0),
            u,
        );
        let y2 = Self::lerp(l3, l4, v);

        Self::lerp(y1, y2, w)
    }

    /// Multi-octave (fBm-style) layering for 2D noise.
    pub fn octave_noise2d(
        &self,
        x: f64,
        y: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let oct = octaves.clamp(1, 16);
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut max_amp = 0.0;
        for _ in 0..oct {
            total += self.noise2d(x * freq, y * freq) * amp;
            max_amp += amp;
            freq *= lacunarity;
            amp *= persistence;
        }
        total / max_amp
    }

    /// Multi-octave (fBm-style) layering for 3D noise.
    pub fn octave_noise3d(
        &self,
        x: f64,
        y: f64,
        z: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let oct = octaves.clamp(1, 16);
        let mut total = 0.0;
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut max_amp = 0.0;
        for _ in 0..oct {
            total += self.noise3d(x * freq, y * freq, z * freq) * amp;
            max_amp += amp;
            freq *= lacunarity;
            amp *= persistence;
        }
        total / max_amp
    }

    /// Generate a 2D noise grid of the given dimensions.
    pub fn generate_grid(&self, width: usize, height: usize, scale: f64) -> Vec<Vec<f64>> {
        let s = if scale.abs() < 1e-12 { 1.0 } else { scale };
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| self.noise2d(x as f64 / s, y as f64 / s))
                    .collect()
            })
            .collect()
    }
}

// ── Tests ────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_same_seed() {
        let a = PerlinNoise::new(42);
        let b = PerlinNoise::new(42);
        assert!((a.noise2d(1.5, 2.5) - b.noise2d(1.5, 2.5)).abs() < 1e-12);
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = PerlinNoise::new(1);
        let b = PerlinNoise::new(2);
        // Extremely unlikely to match at an arbitrary point
        assert!((a.noise2d(3.7, 4.2) - b.noise2d(3.7, 4.2)).abs() > 1e-9);
    }

    #[test]
    fn test_2d_range() {
        let p = PerlinNoise::new(0);
        for i in 0..500 {
            let x = (i as f64) * 0.17;
            let y = (i as f64) * 0.31;
            let v = p.noise2d(x, y);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "out of range: {v}");
        }
    }

    #[test]
    fn test_3d_range() {
        let p = PerlinNoise::new(99);
        for i in 0..300 {
            let t = i as f64 * 0.23;
            let v = p.noise3d(t, t * 0.7, t * 1.3);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "out of range: {v}");
        }
    }

    #[test]
    fn test_fade_endpoints() {
        assert!((PerlinNoise::fade(0.0)).abs() < 1e-12);
        assert!((PerlinNoise::fade(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fade_midpoint() {
        assert!((PerlinNoise::fade(0.5) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_lerp() {
        assert!((PerlinNoise::lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-12);
        assert!((PerlinNoise::lerp(2.0, 8.0, 0.0) - 2.0).abs() < 1e-12);
        assert!((PerlinNoise::lerp(2.0, 8.0, 1.0) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_noise_at_integer_coordinates() {
        // At integer coords, the fractional parts are 0 so noise = 0
        let p = PerlinNoise::new(7);
        assert!(p.noise2d(0.0, 0.0).abs() < 1e-12);
        assert!(p.noise2d(5.0, 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_noise3d_at_integer_coordinates() {
        let p = PerlinNoise::new(7);
        assert!(p.noise3d(0.0, 0.0, 0.0).abs() < 1e-12);
        assert!(p.noise3d(2.0, 3.0, 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_continuity_2d() {
        let p = PerlinNoise::new(42);
        let eps = 1e-5;
        let v = p.noise2d(3.5, 7.2);
        let vx = p.noise2d(3.5 + eps, 7.2);
        let vy = p.noise2d(3.5, 7.2 + eps);
        assert!((v - vx).abs() < 0.01);
        assert!((v - vy).abs() < 0.01);
    }

    #[test]
    fn test_continuity_3d() {
        let p = PerlinNoise::new(42);
        let eps = 1e-5;
        let v = p.noise3d(1.1, 2.2, 3.3);
        let vx = p.noise3d(1.1 + eps, 2.2, 3.3);
        assert!((v - vx).abs() < 0.01);
    }

    #[test]
    fn test_tiling_2d_wraps() {
        let p = PerlinNoise::new(55);
        let period = 8;
        // Check many sample points along the wrapping boundary
        for i in 0..20 {
            let y = i as f64 * 0.37;
            let a = p.noise2d_tiling(0.5, y, period, period);
            let b = p.noise2d_tiling(0.5 + period as f64, y, period, period);
            assert!((a - b).abs() < 1e-10, "tiling mismatch at y={y}: {a} vs {b}");
        }
    }

    #[test]
    fn test_tiling_3d_wraps() {
        let p = PerlinNoise::new(55);
        let per = 4;
        let a = p.noise3d_tiling(0.5, 0.5, 0.5, per, per, per);
        let b = p.noise3d_tiling(0.5 + per as f64, 0.5, 0.5, per, per, per);
        assert!((a - b).abs() < 1e-10);
    }

    #[test]
    fn test_octave_noise2d_range() {
        let p = PerlinNoise::new(77);
        for i in 0..200 {
            let x = i as f64 * 0.13;
            let v = p.octave_noise2d(x, x * 0.9, 6, 2.0, 0.5);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6, "octave out of range: {v}");
        }
    }

    #[test]
    fn test_octave_noise3d_range() {
        let p = PerlinNoise::new(88);
        for i in 0..100 {
            let t = i as f64 * 0.19;
            let v = p.octave_noise3d(t, t * 0.8, t * 1.1, 4, 2.0, 0.5);
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_octave_clamps_to_valid_range() {
        let p = PerlinNoise::new(1);
        // 0 octaves should be clamped to 1
        let v = p.octave_noise2d(1.5, 2.5, 0, 2.0, 0.5);
        assert!(v.is_finite());
        // 100 octaves clamped to 16
        let v2 = p.octave_noise2d(1.5, 2.5, 100, 2.0, 0.5);
        assert!(v2.is_finite());
    }

    #[test]
    fn test_generate_grid_dimensions() {
        let p = PerlinNoise::new(10);
        let grid = p.generate_grid(16, 8, 4.0);
        assert_eq!(grid.len(), 8);
        for row in &grid {
            assert_eq!(row.len(), 16);
        }
    }

    #[test]
    fn test_generate_grid_values_in_range() {
        let p = PerlinNoise::new(10);
        let grid = p.generate_grid(32, 32, 8.0);
        for row in &grid {
            for &v in row {
                assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
            }
        }
    }

    #[test]
    fn test_generate_grid_zero_scale() {
        let p = PerlinNoise::new(10);
        let grid = p.generate_grid(4, 4, 0.0);
        assert_eq!(grid.len(), 4);
    }

    #[test]
    fn test_negative_coordinates() {
        let p = PerlinNoise::new(33);
        let v = p.noise2d(-5.3, -8.7);
        assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
        let v3 = p.noise3d(-2.1, -3.4, -7.8);
        assert!(v3 >= -1.0 - 1e-6 && v3 <= 1.0 + 1e-6);
    }

    #[test]
    fn test_seed_getter() {
        let p = PerlinNoise::new(12345);
        assert_eq!(p.seed(), 12345);
    }

    #[test]
    fn test_clone() {
        let p = PerlinNoise::new(42);
        let p2 = p.clone();
        assert!((p.noise2d(1.0, 1.0) - p2.noise2d(1.0, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_debug_format() {
        let p = PerlinNoise::new(42);
        let s = format!("{:?}", p);
        assert!(s.contains("PerlinNoise"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_spatial_variation() {
        let p = PerlinNoise::new(42);
        let mut distinct = 0;
        let base = p.noise2d(0.5, 0.5);
        for i in 1..20 {
            let v = p.noise2d(0.5 + i as f64 * 0.7, 0.5 + i as f64 * 0.3);
            if (v - base).abs() > 1e-6 {
                distinct += 1;
            }
        }
        assert!(distinct > 10, "noise should vary across space");
    }
}
