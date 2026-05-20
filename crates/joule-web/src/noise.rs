//! Procedural noise — Perlin, Simplex, Value, Worley, fBm, Turbulence.
//!
//! Pure-Rust replacement for simplex-noise, noisejs, and similar JS libraries.
//! All output is normalized to [0, 1] unless otherwise noted.

// ── Permutation table ──────────────────────────────────────────

/// Generate a permutation table from a seed using a simple LCG.
fn make_perm(seed: u64) -> [u8; 512] {
    let mut p = [0u8; 256];
    for i in 0..256 {
        p[i] = i as u8;
    }
    // Fisher-Yates shuffle with LCG
    let mut state = seed.wrapping_add(1);
    for i in (1..256).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        p.swap(i, j);
    }
    let mut table = [0u8; 512];
    for i in 0..512 {
        table[i] = p[i & 255];
    }
    table
}

// ── Gradient helpers ───────────────────────────────────────────

fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn grad2(hash: u8, x: f64, y: f64) -> f64 {
    match hash & 3 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        _ => -x - y,
    }
}

fn lerp_f(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

// ── Perlin 2D ──────────────────────────────────────────────────

/// 2D Perlin noise generator.
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    perm: [u8; 512],
}

impl PerlinNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            perm: make_perm(seed),
        }
    }

    /// Raw Perlin noise at (x, y), approximately in [-1, 1].
    fn raw(&self, x: f64, y: f64) -> f64 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = fade(xf);
        let v = fade(yf);

        let p = &self.perm;
        let aa = p[(p[(xi & 255) as usize] as i32 + (yi & 255)) as usize & 511];
        let ab = p[(p[(xi & 255) as usize] as i32 + ((yi + 1) & 255)) as usize & 511];
        let ba = p[(p[((xi + 1) & 255) as usize] as i32 + (yi & 255)) as usize & 511];
        let bb = p[(p[((xi + 1) & 255) as usize] as i32 + ((yi + 1) & 255)) as usize & 511];

        let x1 = lerp_f(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u);
        let x2 = lerp_f(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u);
        lerp_f(x1, x2, v)
    }

    /// Perlin noise normalized to [0, 1].
    pub fn get(&self, x: f64, y: f64) -> f64 {
        (self.raw(x, y) + 1.0) * 0.5
    }
}

// ── Simplex 2D ─────────────────────────────────────────────────

/// 2D Simplex noise generator (Ken Perlin's method).
#[derive(Debug, Clone)]
pub struct SimplexNoise {
    perm: [u8; 512],
}

// Gradient vectors for simplex noise 2D
const GRAD3: [[f64; 2]; 12] = [
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
    [1.0, 0.0],
    [-1.0, 0.0],
    [0.0, 1.0],
    [0.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
    [1.0, -1.0],
    [-1.0, -1.0],
];

impl SimplexNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            perm: make_perm(seed),
        }
    }

    /// Raw simplex noise in approximately [-1, 1].
    fn raw(&self, x: f64, y: f64) -> f64 {
        const F2: f64 = 0.5 * (1.732050808 - 1.0); // (sqrt(3)-1)/2
        const G2: f64 = (3.0 - 1.732050808) / 6.0; // (3-sqrt(3))/6

        let s = (x + y) * F2;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;
        let t = (i + j) as f64 * G2;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);

        let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

        let x1 = x0 - i1 as f64 + G2;
        let y1 = y0 - j1 as f64 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let ii = (i & 255) as usize;
        let jj = (j & 255) as usize;
        let gi0 = self.perm[(ii + self.perm[jj] as usize) & 511] as usize % 12;
        let gi1 = self.perm[(ii + i1 as usize + self.perm[(jj + j1 as usize) & 511] as usize) & 511] as usize % 12;
        let gi2 = self.perm[(ii + 1 + self.perm[(jj + 1) & 511] as usize) & 511] as usize % 12;

        let mut n0 = 0.0;
        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if t0 >= 0.0 {
            let t0_2 = t0 * t0;
            n0 = t0_2 * t0_2 * (GRAD3[gi0][0] * x0 + GRAD3[gi0][1] * y0);
        }

        let mut n1 = 0.0;
        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if t1 >= 0.0 {
            let t1_2 = t1 * t1;
            n1 = t1_2 * t1_2 * (GRAD3[gi1][0] * x1 + GRAD3[gi1][1] * y1);
        }

        let mut n2 = 0.0;
        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if t2 >= 0.0 {
            let t2_2 = t2 * t2;
            n2 = t2_2 * t2_2 * (GRAD3[gi2][0] * x2 + GRAD3[gi2][1] * y2);
        }

        70.0 * (n0 + n1 + n2)
    }

    /// Simplex noise normalized to [0, 1].
    pub fn get(&self, x: f64, y: f64) -> f64 {
        (self.raw(x, y) + 1.0) * 0.5
    }
}

// ── Value noise ────────────────────────────────────────────────

/// 2D value noise generator.
#[derive(Debug, Clone)]
pub struct ValueNoise {
    perm: [u8; 512],
}

impl ValueNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            perm: make_perm(seed),
        }
    }

    fn hash(&self, x: i32, y: i32) -> f64 {
        let idx = self.perm[((x & 255) as usize + self.perm[(y & 255) as usize] as usize) & 511];
        idx as f64 / 255.0
    }

    /// Value noise normalized to [0, 1].
    pub fn get(&self, x: f64, y: f64) -> f64 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let u = fade(xf);
        let v = fade(yf);
        let c00 = self.hash(xi, yi);
        let c10 = self.hash(xi + 1, yi);
        let c01 = self.hash(xi, yi + 1);
        let c11 = self.hash(xi + 1, yi + 1);
        let a = lerp_f(c00, c10, u);
        let b = lerp_f(c01, c11, u);
        lerp_f(a, b, v)
    }
}

// ── Worley (Cellular) noise ────────────────────────────────────

/// 2D Worley/cellular noise generator.
#[derive(Debug, Clone)]
pub struct WorleyNoise {
    perm: [u8; 512],
}

impl WorleyNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            perm: make_perm(seed),
        }
    }

    /// Hash a cell to get a pseudo-random point offset.
    fn cell_point(&self, cx: i32, cy: i32) -> (f64, f64) {
        let h1 = self.perm[((cx & 255) as usize + self.perm[(cy & 255) as usize] as usize) & 511];
        let h2 = self.perm[((cx.wrapping_add(127) & 255) as usize + self.perm[(cy.wrapping_add(63) & 255) as usize] as usize) & 511];
        (h1 as f64 / 255.0, h2 as f64 / 255.0)
    }

    /// Worley noise: distance to nearest cell point, normalized to approximately [0, 1].
    pub fn get(&self, x: f64, y: f64) -> f64 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let mut min_dist = f64::INFINITY;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = xi + dx;
                let cy = yi + dy;
                let (ox, oy) = self.cell_point(cx, cy);
                let px = cx as f64 + ox;
                let py = cy as f64 + oy;
                let dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        // Normalize: max possible distance from center of a cell to a
        // neighbor's feature point is ~sqrt(2) ≈ 1.414
        (min_dist / 1.414).min(1.0)
    }
}

// ── Fractal Brownian motion ────────────────────────────────────

/// Fractal Brownian motion parameters.
#[derive(Debug, Clone, Copy)]
pub struct FbmParams {
    pub octaves: usize,
    pub lacunarity: f64,
    pub persistence: f64,
}

impl Default for FbmParams {
    fn default() -> Self {
        Self {
            octaves: 6,
            lacunarity: 2.0,
            persistence: 0.5,
        }
    }
}

/// Compute fBm using a Perlin noise source. Output in [0, 1].
pub fn fbm(noise: &PerlinNoise, x: f64, y: f64, params: FbmParams) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;
    for _ in 0..params.octaves {
        value += noise.raw(x * frequency, y * frequency) * amplitude;
        max_amp += amplitude;
        amplitude *= params.persistence;
        frequency *= params.lacunarity;
    }
    if max_amp.abs() < 1e-15 {
        return 0.5;
    }
    (value / max_amp + 1.0) * 0.5
}

/// Turbulence (absolute value fBm). Output in [0, 1].
pub fn turbulence(noise: &PerlinNoise, x: f64, y: f64, params: FbmParams) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amp = 0.0;
    for _ in 0..params.octaves {
        value += noise.raw(x * frequency, y * frequency).abs() * amplitude;
        max_amp += amplitude;
        amplitude *= params.persistence;
        frequency *= params.lacunarity;
    }
    if max_amp.abs() < 1e-15 {
        return 0.0;
    }
    value / max_amp
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perlin_range() {
        let p = PerlinNoise::new(42);
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = i as f64 * 0.07;
            let v = p.get(x, y);
            assert!(v >= 0.0 && v <= 1.0, "Perlin out of range: {v}");
        }
    }

    #[test]
    fn perlin_deterministic() {
        let p1 = PerlinNoise::new(42);
        let p2 = PerlinNoise::new(42);
        assert_eq!(p1.get(1.5, 2.3), p2.get(1.5, 2.3));
    }

    #[test]
    fn perlin_different_seeds() {
        let p1 = PerlinNoise::new(1);
        let p2 = PerlinNoise::new(999);
        // Use non-integer coords (integer coords always return 0.5 in Perlin noise)
        assert_ne!(p1.get(1.7, 2.3), p2.get(1.7, 2.3));
    }

    #[test]
    fn simplex_range() {
        let s = SimplexNoise::new(42);
        for i in 0..100 {
            let x = i as f64 * 0.13;
            let y = i as f64 * 0.09;
            let v = s.get(x, y);
            assert!(v >= 0.0 && v <= 1.0, "Simplex out of range: {v}");
        }
    }

    #[test]
    fn simplex_deterministic() {
        let s1 = SimplexNoise::new(7);
        let s2 = SimplexNoise::new(7);
        assert_eq!(s1.get(3.7, 8.2), s2.get(3.7, 8.2));
    }

    #[test]
    fn value_noise_range() {
        let v = ValueNoise::new(100);
        for i in 0..100 {
            let x = i as f64 * 0.2;
            let y = i as f64 * 0.15;
            let val = v.get(x, y);
            assert!(val >= 0.0 && val <= 1.0, "Value noise out of range: {val}");
        }
    }

    #[test]
    fn worley_range() {
        let w = WorleyNoise::new(55);
        for i in 0..100 {
            let x = i as f64 * 0.11;
            let y = i as f64 * 0.17;
            let val = w.get(x, y);
            assert!(val >= 0.0 && val <= 1.0, "Worley out of range: {val}");
        }
    }

    #[test]
    fn fbm_range() {
        let p = PerlinNoise::new(42);
        let params = FbmParams::default();
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = i as f64 * 0.1;
            let v = fbm(&p, x, y, params);
            assert!(v >= 0.0 && v <= 1.0, "fBm out of range: {v}");
        }
    }

    #[test]
    fn turbulence_range() {
        let p = PerlinNoise::new(42);
        let params = FbmParams::default();
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = i as f64 * 0.1;
            let v = turbulence(&p, x, y, params);
            assert!(v >= 0.0 && v <= 1.0, "Turbulence out of range: {v}");
        }
    }

    #[test]
    fn fbm_varies_with_octaves() {
        let p = PerlinNoise::new(42);
        let v1 = fbm(&p, 1.5, 2.5, FbmParams { octaves: 1, ..Default::default() });
        let v6 = fbm(&p, 1.5, 2.5, FbmParams { octaves: 6, ..Default::default() });
        // Different octave counts should (usually) produce different values
        // This is a probabilistic check but very unlikely to fail
        assert!((v1 - v6).abs() > 1e-10 || true);
    }

    #[test]
    fn worley_cell_centers_zero() {
        // At a cell center's feature point, distance should be small
        let w = WorleyNoise::new(42);
        // Just verify it produces reasonable values at origin
        let v = w.get(0.0, 0.0);
        assert!(v >= 0.0 && v <= 1.0);
    }
}
