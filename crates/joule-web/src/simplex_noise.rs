// Procedural Terrain Generation — Simplex noise (Perlin's improved algorithm)
// 2D/3D/4D, analytical derivatives, domain rotation, seed support

use std::fmt;

// Skew / unskew constants for 2D simplex
const F2: f64 = 0.3660254037844386; // (sqrt(3) - 1) / 2
const G2: f64 = 0.21132486540518713; // (3 - sqrt(3)) / 6

// Skew / unskew constants for 3D simplex
const F3: f64 = 1.0 / 3.0;
const G3: f64 = 1.0 / 6.0;

// Skew / unskew constants for 4D simplex
const F4: f64 = 0.30901699437494742; // (sqrt(5) - 1) / 4
const G4: f64 = 0.13819660112501051; // (5 - sqrt(5)) / 20

const GRAD3: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

const GRAD4: [[f64; 4]; 32] = [
    [0.0,1.0,1.0,1.0],[0.0,1.0,1.0,-1.0],[0.0,1.0,-1.0,1.0],[0.0,1.0,-1.0,-1.0],
    [0.0,-1.0,1.0,1.0],[0.0,-1.0,1.0,-1.0],[0.0,-1.0,-1.0,1.0],[0.0,-1.0,-1.0,-1.0],
    [1.0,0.0,1.0,1.0],[1.0,0.0,1.0,-1.0],[1.0,0.0,-1.0,1.0],[1.0,0.0,-1.0,-1.0],
    [-1.0,0.0,1.0,1.0],[-1.0,0.0,1.0,-1.0],[-1.0,0.0,-1.0,1.0],[-1.0,0.0,-1.0,-1.0],
    [1.0,1.0,0.0,1.0],[1.0,1.0,0.0,-1.0],[1.0,-1.0,0.0,1.0],[1.0,-1.0,0.0,-1.0],
    [-1.0,1.0,0.0,1.0],[-1.0,1.0,0.0,-1.0],[-1.0,-1.0,0.0,1.0],[-1.0,-1.0,0.0,-1.0],
    [1.0,1.0,1.0,0.0],[1.0,1.0,-1.0,0.0],[1.0,-1.0,1.0,0.0],[1.0,-1.0,-1.0,0.0],
    [-1.0,1.0,1.0,0.0],[-1.0,1.0,-1.0,0.0],[-1.0,-1.0,1.0,0.0],[-1.0,-1.0,-1.0,0.0],
];

/// Analytical derivative result for 2D simplex noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseDerivative2D {
    pub value: f64,
    pub dx: f64,
    pub dy: f64,
}

/// Analytical derivative result for 3D simplex noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseDerivative3D {
    pub value: f64,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

/// Simplex noise generator.
#[derive(Clone)]
pub struct SimplexNoise {
    perm: [u16; 512],
    seed: u64,
}

impl fmt::Debug for SimplexNoise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimplexNoise").field("seed", &self.seed).finish()
    }
}

impl SimplexNoise {
    /// Create a new simplex noise generator from a seed.
    pub fn new(seed: u64) -> Self {
        let mut table: [u16; 256] = [0; 256];
        for i in 0..256 {
            table[i] = i as u16;
        }
        let mut rng = seed;
        for i in (1..256).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            table.swap(i, j);
        }
        let mut perm = [0u16; 512];
        for i in 0..512 {
            perm[i] = table[i & 255];
        }
        Self { perm, seed }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn hash(&self, i: i64) -> usize {
        self.perm[(i & 255) as usize] as usize
    }

    fn dot2(g: &[f64; 3], x: f64, y: f64) -> f64 {
        g[0] * x + g[1] * y
    }

    fn dot3(g: &[f64; 3], x: f64, y: f64, z: f64) -> f64 {
        g[0] * x + g[1] * y + g[2] * z
    }

    fn dot4(g: &[f64; 4], x: f64, y: f64, z: f64, w: f64) -> f64 {
        g[0] * x + g[1] * y + g[2] * z + g[3] * w
    }

    /// 2D simplex noise in approximately [-1, 1].
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        let s = (x + y) * F2;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;
        let t = (i + j) as f64 * G2;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);

        let (i1, j1) = if x0 > y0 { (1i64, 0i64) } else { (0, 1) };

        let x1 = x0 - i1 as f64 + G2;
        let y1 = y0 - j1 as f64 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let gi0 = self.hash(i + self.hash(j) as i64) % 12;
        let gi1 = self.hash(i + i1 + self.hash(j + j1) as i64) % 12;
        let gi2 = self.hash(i + 1 + self.hash(j + 1) as i64) % 12;

        let mut n = 0.0;
        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if t0 > 0.0 {
            let t0sq = t0 * t0;
            n += t0sq * t0sq * Self::dot2(&GRAD3[gi0], x0, y0);
        }
        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if t1 > 0.0 {
            let t1sq = t1 * t1;
            n += t1sq * t1sq * Self::dot2(&GRAD3[gi1], x1, y1);
        }
        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if t2 > 0.0 {
            let t2sq = t2 * t2;
            n += t2sq * t2sq * Self::dot2(&GRAD3[gi2], x2, y2);
        }
        70.0 * n
    }

    /// 2D simplex noise with analytical derivatives.
    pub fn noise2d_deriv(&self, x: f64, y: f64) -> NoiseDerivative2D {
        let s = (x + y) * F2;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;
        let t = (i + j) as f64 * G2;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);

        let (i1, j1) = if x0 > y0 { (1i64, 0i64) } else { (0, 1) };

        let x1 = x0 - i1 as f64 + G2;
        let y1 = y0 - j1 as f64 + G2;
        let x2 = x0 - 1.0 + 2.0 * G2;
        let y2 = y0 - 1.0 + 2.0 * G2;

        let gi0 = self.hash(i + self.hash(j) as i64) % 12;
        let gi1 = self.hash(i + i1 + self.hash(j + j1) as i64) % 12;
        let gi2 = self.hash(i + 1 + self.hash(j + 1) as i64) % 12;

        let mut val = 0.0;
        let mut dx = 0.0;
        let mut dy = 0.0;

        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if t0 > 0.0 {
            let g = &GRAD3[gi0];
            let t0sq = t0 * t0;
            let t0_4 = t0sq * t0sq;
            let gdot = g[0] * x0 + g[1] * y0;
            val += t0_4 * gdot;
            let deriv_coeff = -8.0 * t0sq * t0 * gdot;
            dx += deriv_coeff * x0 + t0_4 * g[0];
            dy += deriv_coeff * y0 + t0_4 * g[1];
        }

        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if t1 > 0.0 {
            let g = &GRAD3[gi1];
            let t1sq = t1 * t1;
            let t1_4 = t1sq * t1sq;
            let gdot = g[0] * x1 + g[1] * y1;
            val += t1_4 * gdot;
            let deriv_coeff = -8.0 * t1sq * t1 * gdot;
            dx += deriv_coeff * x1 + t1_4 * g[0];
            dy += deriv_coeff * y1 + t1_4 * g[1];
        }

        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if t2 > 0.0 {
            let g = &GRAD3[gi2];
            let t2sq = t2 * t2;
            let t2_4 = t2sq * t2sq;
            let gdot = g[0] * x2 + g[1] * y2;
            val += t2_4 * gdot;
            let deriv_coeff = -8.0 * t2sq * t2 * gdot;
            dx += deriv_coeff * x2 + t2_4 * g[0];
            dy += deriv_coeff * y2 + t2_4 * g[1];
        }

        NoiseDerivative2D {
            value: 70.0 * val,
            dx: 70.0 * dx,
            dy: 70.0 * dy,
        }
    }

    /// 3D simplex noise in approximately [-1, 1].
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let s = (x + y + z) * F3;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;
        let k = (z + s).floor() as i64;
        let t = (i + j + k) as f64 * G3;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);
        let z0 = z - (k as f64 - t);

        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 { (1,0,0,1,1,0) }
            else if x0 >= z0 { (1,0,0,1,0,1) }
            else { (0,0,1,1,0,1) }
        } else {
            if y0 < z0 { (0,0,1,0,1,1) }
            else if x0 < z0 { (0,1,0,0,1,1) }
            else { (0,1,0,1,1,0) }
        };

        let x1 = x0 - i1 as f64 + G3;
        let y1 = y0 - j1 as f64 + G3;
        let z1 = z0 - k1 as f64 + G3;
        let x2 = x0 - i2 as f64 + 2.0 * G3;
        let y2 = y0 - j2 as f64 + 2.0 * G3;
        let z2 = z0 - k2 as f64 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let gi0 = self.hash(i + self.hash(j + self.hash(k) as i64) as i64) % 12;
        let gi1 = self.hash(i + i1 as i64 + self.hash(j + j1 as i64 + self.hash(k + k1 as i64) as i64) as i64) % 12;
        let gi2 = self.hash(i + i2 as i64 + self.hash(j + j2 as i64 + self.hash(k + k2 as i64) as i64) as i64) % 12;
        let gi3 = self.hash(i + 1 + self.hash(j + 1 + self.hash(k + 1) as i64) as i64) % 12;

        let mut n = 0.0;
        for &(gi, xv, yv, zv) in &[
            (gi0, x0, y0, z0), (gi1, x1, y1, z1),
            (gi2, x2, y2, z2), (gi3, x3, y3, z3),
        ] {
            let tv = 0.6 - xv * xv - yv * yv - zv * zv;
            if tv > 0.0 {
                let tvsq = tv * tv;
                n += tvsq * tvsq * Self::dot3(&GRAD3[gi], xv, yv, zv);
            }
        }
        32.0 * n
    }

    /// 3D simplex noise with analytical derivatives.
    pub fn noise3d_deriv(&self, x: f64, y: f64, z: f64) -> NoiseDerivative3D {
        let s = (x + y + z) * F3;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;
        let k = (z + s).floor() as i64;
        let t = (i + j + k) as f64 * G3;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);
        let z0 = z - (k as f64 - t);

        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 { (1,0,0,1,1,0) }
            else if x0 >= z0 { (1,0,0,1,0,1) }
            else { (0,0,1,1,0,1) }
        } else {
            if y0 < z0 { (0,0,1,0,1,1) }
            else if x0 < z0 { (0,1,0,0,1,1) }
            else { (0,1,0,1,1,0) }
        };

        let offsets: [(f64,f64,f64); 4] = [
            (x0, y0, z0),
            (x0 - i1 as f64 + G3, y0 - j1 as f64 + G3, z0 - k1 as f64 + G3),
            (x0 - i2 as f64 + 2.0*G3, y0 - j2 as f64 + 2.0*G3, z0 - k2 as f64 + 2.0*G3),
            (x0 - 1.0 + 3.0*G3, y0 - 1.0 + 3.0*G3, z0 - 1.0 + 3.0*G3),
        ];

        let gis = [
            self.hash(i + self.hash(j + self.hash(k) as i64) as i64) % 12,
            self.hash(i + i1 as i64 + self.hash(j + j1 as i64 + self.hash(k + k1 as i64) as i64) as i64) % 12,
            self.hash(i + i2 as i64 + self.hash(j + j2 as i64 + self.hash(k + k2 as i64) as i64) as i64) % 12,
            self.hash(i + 1 + self.hash(j + 1 + self.hash(k + 1) as i64) as i64) % 12,
        ];

        let mut val = 0.0;
        let mut ddx = 0.0;
        let mut ddy = 0.0;
        let mut ddz = 0.0;

        for idx in 0..4 {
            let (xv, yv, zv) = offsets[idx];
            let tv = 0.6 - xv * xv - yv * yv - zv * zv;
            if tv > 0.0 {
                let g = &GRAD3[gis[idx]];
                let tvsq = tv * tv;
                let tv4 = tvsq * tvsq;
                let gdot = g[0] * xv + g[1] * yv + g[2] * zv;
                val += tv4 * gdot;
                let dc = -8.0 * tvsq * tv * gdot;
                ddx += dc * xv + tv4 * g[0];
                ddy += dc * yv + tv4 * g[1];
                ddz += dc * zv + tv4 * g[2];
            }
        }

        NoiseDerivative3D {
            value: 32.0 * val,
            dx: 32.0 * ddx,
            dy: 32.0 * ddy,
            dz: 32.0 * ddz,
        }
    }

    /// 4D simplex noise for animated noise fields.
    pub fn noise4d(&self, x: f64, y: f64, z: f64, w: f64) -> f64 {
        let s = (x + y + z + w) * F4;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;
        let k = (z + s).floor() as i64;
        let l = (w + s).floor() as i64;
        let t = (i + j + k + l) as f64 * G4;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);
        let z0 = z - (k as f64 - t);
        let w0 = w - (l as f64 - t);

        // Determine simplex traversal order via ranking
        let mut rank = [0u8; 4];
        let vals = [x0, y0, z0, w0];
        for a in 0..4 {
            for b in (a + 1)..4 {
                if vals[a] > vals[b] { rank[a] += 1; } else { rank[b] += 1; }
            }
        }
        let coord = |r: u8, threshold: u8| -> i64 { if r >= threshold { 1 } else { 0 } };

        let mut n = 0.0;
        for step in 0..5u8 {
            let threshold = 4 - step;
            let (oi, oj, ok, ol) = if step < 4 {
                (coord(rank[0], threshold), coord(rank[1], threshold),
                 coord(rank[2], threshold), coord(rank[3], threshold))
            } else {
                (1, 1, 1, 1)
            };
            let xv = x0 - oi as f64 + step as f64 * G4;
            let yv = y0 - oj as f64 + step as f64 * G4;
            let zv = z0 - ok as f64 + step as f64 * G4;
            let wv = w0 - ol as f64 + step as f64 * G4;

            let tv = 0.6 - xv * xv - yv * yv - zv * zv - wv * wv;
            if tv > 0.0 {
                let gi = self.hash(
                    i + oi + self.hash(
                        j + oj + self.hash(
                            k + ok + self.hash(l + ol) as i64
                        ) as i64
                    ) as i64
                ) % 32;
                let tvsq = tv * tv;
                n += tvsq * tvsq * Self::dot4(&GRAD4[gi], xv, yv, zv, wv);
            }
        }
        27.0 * n
    }

    /// Domain rotation to reduce axis-aligned artifacts (2D).
    /// Applies a rotation by `angle` radians before sampling.
    pub fn noise2d_rotated(&self, x: f64, y: f64, angle: f64) -> f64 {
        let (sin_a, cos_a) = angle.sin_cos();
        let rx = x * cos_a - y * sin_a;
        let ry = x * sin_a + y * cos_a;
        self.noise2d(rx, ry)
    }

    /// Domain rotation for 3D: rotates input around the (1,1,1) axis.
    pub fn noise3d_rotated(&self, x: f64, y: f64, z: f64) -> f64 {
        // Rotation around (1,1,1) by ~39.23 degrees (magic angle for simplex)
        let r = 2.0 / 3.0;
        let rx = r * x + (r - 1.0) * y + (r - 1.0) * z;
        let ry = (r - 1.0) * x + r * y + (r - 1.0) * z;
        let rz = (r - 1.0) * x + (r - 1.0) * y + r * z;
        self.noise3d(rx, ry, rz)
    }

    /// Multi-octave simplex noise for 2D.
    pub fn octave2d(&self, x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64 {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_same_seed() {
        let a = SimplexNoise::new(42);
        let b = SimplexNoise::new(42);
        assert!((a.noise2d(1.5, 2.5) - b.noise2d(1.5, 2.5)).abs() < 1e-12);
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = SimplexNoise::new(1);
        let b = SimplexNoise::new(999);
        assert!((a.noise2d(3.7, 4.2) - b.noise2d(3.7, 4.2)).abs() > 1e-9);
    }

    #[test]
    fn test_2d_range() {
        let s = SimplexNoise::new(77);
        for i in 0..500 {
            let x = i as f64 * 0.17 - 40.0;
            let y = i as f64 * 0.31 - 70.0;
            let v = s.noise2d(x, y);
            assert!(v >= -1.5 && v <= 1.5, "2d out of range: {v} at ({x},{y})");
        }
    }

    #[test]
    fn test_3d_range() {
        let s = SimplexNoise::new(88);
        for i in 0..300 {
            let t = i as f64 * 0.23 - 30.0;
            let v = s.noise3d(t, t * 0.7, t * 1.3);
            assert!(v >= -1.5 && v <= 1.5, "3d out of range: {v}");
        }
    }

    #[test]
    fn test_4d_range() {
        let s = SimplexNoise::new(55);
        for i in 0..200 {
            let t = i as f64 * 0.19 - 20.0;
            let v = s.noise4d(t, t * 0.6, t * 1.1, t * 0.3);
            assert!(v >= -2.0 && v <= 2.0, "4d out of range: {v}");
        }
    }

    #[test]
    fn test_continuity_2d() {
        let s = SimplexNoise::new(42);
        let eps = 1e-6;
        let v = s.noise2d(3.5, 7.2);
        let vx = s.noise2d(3.5 + eps, 7.2);
        assert!((v - vx).abs() < 0.01);
    }

    #[test]
    fn test_continuity_3d() {
        let s = SimplexNoise::new(42);
        let eps = 1e-6;
        let v = s.noise3d(1.1, 2.2, 3.3);
        let vx = s.noise3d(1.1 + eps, 2.2, 3.3);
        assert!((v - vx).abs() < 0.01);
    }

    #[test]
    fn test_derivative_2d_finite_diff() {
        let s = SimplexNoise::new(42);
        let eps = 1e-5;
        let d = s.noise2d_deriv(3.5, 7.2);
        let vx = s.noise2d(3.5 + eps, 7.2);
        let vy = s.noise2d(3.5, 7.2 + eps);
        let fd_dx = (vx - d.value) / eps;
        let fd_dy = (vy - d.value) / eps;
        assert!((d.dx - fd_dx).abs() < 0.5, "dx: {} vs fd {}", d.dx, fd_dx);
        assert!((d.dy - fd_dy).abs() < 0.5, "dy: {} vs fd {}", d.dy, fd_dy);
    }

    #[test]
    fn test_derivative_3d_finite_diff() {
        let s = SimplexNoise::new(42);
        let eps = 1e-5;
        let d = s.noise3d_deriv(1.1, 2.2, 3.3);
        let vx = s.noise3d(1.1 + eps, 2.2, 3.3);
        let fd_dx = (vx - d.value) / eps;
        assert!((d.dx - fd_dx).abs() < 0.5, "dx: {} vs fd {}", d.dx, fd_dx);
    }

    #[test]
    fn test_derivative_value_matches_noise() {
        let s = SimplexNoise::new(42);
        let d = s.noise2d_deriv(5.0, 3.0);
        let v = s.noise2d(5.0, 3.0);
        assert!((d.value - v).abs() < 1e-12);
    }

    #[test]
    fn test_derivative_3d_value_matches_noise() {
        let s = SimplexNoise::new(42);
        let d = s.noise3d_deriv(1.0, 2.0, 3.0);
        let v = s.noise3d(1.0, 2.0, 3.0);
        assert!((d.value - v).abs() < 1e-12);
    }

    #[test]
    fn test_rotated_2d_range() {
        let s = SimplexNoise::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.3;
            let v = s.noise2d_rotated(t, t * 0.7, 0.7854);
            assert!(v >= -1.5 && v <= 1.5);
        }
    }

    #[test]
    fn test_rotated_3d_range() {
        let s = SimplexNoise::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.3;
            let v = s.noise3d_rotated(t, t * 0.7, t * 1.2);
            assert!(v >= -1.5 && v <= 1.5);
        }
    }

    #[test]
    fn test_octave2d_range() {
        let s = SimplexNoise::new(42);
        for i in 0..200 {
            let x = i as f64 * 0.11;
            let v = s.octave2d(x, x * 0.8, 6, 2.0, 0.5);
            assert!(v >= -2.0 && v <= 2.0, "octave out of range: {v}");
        }
    }

    #[test]
    fn test_spatial_variation_2d() {
        let s = SimplexNoise::new(42);
        let mut distinct = 0;
        let base = s.noise2d(0.5, 0.5);
        for i in 1..20 {
            let v = s.noise2d(0.5 + i as f64 * 0.7, 0.5 + i as f64 * 0.3);
            if (v - base).abs() > 1e-6 { distinct += 1; }
        }
        assert!(distinct > 10);
    }

    #[test]
    fn test_spatial_variation_3d() {
        let s = SimplexNoise::new(42);
        let mut distinct = 0;
        let base = s.noise3d(0.5, 0.5, 0.5);
        for i in 1..20 {
            let v = s.noise3d(0.5 + i as f64 * 0.5, 0.5 + i as f64 * 0.3, 0.5 + i as f64 * 0.7);
            if (v - base).abs() > 1e-6 { distinct += 1; }
        }
        assert!(distinct > 10);
    }

    #[test]
    fn test_negative_coordinates() {
        let s = SimplexNoise::new(33);
        let v2 = s.noise2d(-5.3, -8.7);
        assert!(v2.is_finite());
        let v3 = s.noise3d(-2.1, -3.4, -7.8);
        assert!(v3.is_finite());
        let v4 = s.noise4d(-1.0, -2.0, -3.0, -4.0);
        assert!(v4.is_finite());
    }

    #[test]
    fn test_seed_getter() {
        let s = SimplexNoise::new(12345);
        assert_eq!(s.seed(), 12345);
    }

    #[test]
    fn test_clone() {
        let a = SimplexNoise::new(42);
        let b = a.clone();
        assert!((a.noise2d(1.0, 1.0) - b.noise2d(1.0, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_debug_format() {
        let s = SimplexNoise::new(42);
        let d = format!("{:?}", s);
        assert!(d.contains("SimplexNoise"));
        assert!(d.contains("42"));
    }

    #[test]
    fn test_4d_deterministic() {
        let a = SimplexNoise::new(42);
        let b = SimplexNoise::new(42);
        assert!((a.noise4d(1.0, 2.0, 3.0, 4.0) - b.noise4d(1.0, 2.0, 3.0, 4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_derivative_struct_partial_eq() {
        let d1 = NoiseDerivative2D { value: 1.0, dx: 2.0, dy: 3.0 };
        let d2 = NoiseDerivative2D { value: 1.0, dx: 2.0, dy: 3.0 };
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_noise3d_derivative_struct() {
        let d = NoiseDerivative3D { value: 1.0, dx: 2.0, dy: 3.0, dz: 4.0 };
        let d2 = d;
        assert_eq!(d, d2);
    }
}
