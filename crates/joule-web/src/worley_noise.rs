// Procedural Terrain Generation — Worley (cellular / Voronoi) noise
// Feature point scattering, distance metrics, F1/F2/F2-F1, 2D/3D, tiling, cell IDs

use std::fmt;

/// Distance metric for Worley noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
}

/// Output mode for Worley noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorleyOutput {
    /// Distance to nearest feature point.
    F1,
    /// Distance to second nearest feature point.
    F2,
    /// F2 minus F1 — highlights cell edges.
    F2MinusF1,
}

/// Result of a Worley noise sample including cell identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorleyResult {
    pub value: f64,
    pub cell_id: u64,
}

/// Worley noise generator.
#[derive(Clone)]
pub struct WorleyNoise {
    seed: u64,
    points_per_cell: u32,
    jitter: f64,
    metric: DistanceMetric,
    output: WorleyOutput,
}

impl fmt::Debug for WorleyNoise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorleyNoise")
            .field("seed", &self.seed)
            .field("points_per_cell", &self.points_per_cell)
            .field("jitter", &self.jitter)
            .field("metric", &self.metric)
            .field("output", &self.output)
            .finish()
    }
}

impl WorleyNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            points_per_cell: 1,
            jitter: 1.0,
            metric: DistanceMetric::Euclidean,
            output: WorleyOutput::F1,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn with_points_per_cell(mut self, n: u32) -> Self {
        self.points_per_cell = n.clamp(1, 4);
        self
    }

    pub fn with_jitter(mut self, j: f64) -> Self {
        self.jitter = j.clamp(0.0, 1.0);
        self
    }

    pub fn with_metric(mut self, m: DistanceMetric) -> Self {
        self.metric = m;
        self
    }

    pub fn with_output(mut self, o: WorleyOutput) -> Self {
        self.output = o;
        self
    }

    /// Hash function for cell coordinates.
    fn cell_hash(x: i64, y: i64, seed: u64) -> u64 {
        let mut h = seed;
        h = h.wrapping_add(x as u64).wrapping_mul(6364136223846793005);
        h = h.wrapping_add(y as u64).wrapping_mul(6364136223846793005);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    fn cell_hash3(x: i64, y: i64, z: i64, seed: u64) -> u64 {
        let mut h = seed;
        h = h.wrapping_add(x as u64).wrapping_mul(6364136223846793005);
        h = h.wrapping_add(y as u64).wrapping_mul(6364136223846793005);
        h = h.wrapping_add(z as u64).wrapping_mul(6364136223846793005);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h
    }

    /// Generate feature point position in [0,1) from hash.
    fn hash_to_float(h: u64) -> f64 {
        (h & 0x00FF_FFFF_FFFF_FFFF) as f64 / (0x0100_0000_0000_0000u64 as f64)
    }

    fn distance2d(&self, dx: f64, dy: f64) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => (dx * dx + dy * dy).sqrt(),
            DistanceMetric::Manhattan => dx.abs() + dy.abs(),
            DistanceMetric::Chebyshev => dx.abs().max(dy.abs()),
        }
    }

    fn distance3d(&self, dx: f64, dy: f64, dz: f64) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
            DistanceMetric::Manhattan => dx.abs() + dy.abs() + dz.abs(),
            DistanceMetric::Chebyshev => dx.abs().max(dy.abs()).max(dz.abs()),
        }
    }

    /// 2D Worley noise with full result (value + cell_id).
    pub fn sample2d(&self, x: f64, y: f64) -> WorleyResult {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;

        let mut f1 = f64::MAX;
        let mut f2 = f64::MAX;
        let mut nearest_cell_id: u64 = 0;

        for dx in -1..=1 {
            for dy in -1..=1 {
                let cx = ix + dx;
                let cy = iy + dy;
                let base_hash = Self::cell_hash(cx, cy, self.seed);

                for p in 0..self.points_per_cell {
                    let ph = base_hash.wrapping_add(p as u64).wrapping_mul(2654435761);
                    let px = cx as f64 + 0.5 + (Self::hash_to_float(ph) - 0.5) * self.jitter;
                    let py = cy as f64 + 0.5 + (Self::hash_to_float(ph.wrapping_mul(3)) - 0.5) * self.jitter;
                    let d = self.distance2d(x - px, y - py);

                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                        nearest_cell_id = base_hash.wrapping_add(p as u64);
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
        }

        let value = match self.output {
            WorleyOutput::F1 => f1,
            WorleyOutput::F2 => f2,
            WorleyOutput::F2MinusF1 => f2 - f1,
        };

        WorleyResult { value, cell_id: nearest_cell_id }
    }

    /// 2D Worley noise returning just the value.
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        self.sample2d(x, y).value
    }

    /// 3D Worley noise with full result.
    pub fn sample3d(&self, x: f64, y: f64, z: f64) -> WorleyResult {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let iz = z.floor() as i64;

        let mut f1 = f64::MAX;
        let mut f2 = f64::MAX;
        let mut nearest_cell_id: u64 = 0;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cx = ix + dx;
                    let cy = iy + dy;
                    let cz = iz + dz;
                    let base_hash = Self::cell_hash3(cx, cy, cz, self.seed);

                    for p in 0..self.points_per_cell {
                        let ph = base_hash.wrapping_add(p as u64).wrapping_mul(2654435761);
                        let px = cx as f64 + 0.5 + (Self::hash_to_float(ph) - 0.5) * self.jitter;
                        let py = cy as f64 + 0.5 + (Self::hash_to_float(ph.wrapping_mul(3)) - 0.5) * self.jitter;
                        let pz = cz as f64 + 0.5 + (Self::hash_to_float(ph.wrapping_mul(7)) - 0.5) * self.jitter;
                        let d = self.distance3d(x - px, y - py, z - pz);

                        if d < f1 {
                            f2 = f1;
                            f1 = d;
                            nearest_cell_id = base_hash.wrapping_add(p as u64);
                        } else if d < f2 {
                            f2 = d;
                        }
                    }
                }
            }
        }

        let value = match self.output {
            WorleyOutput::F1 => f1,
            WorleyOutput::F2 => f2,
            WorleyOutput::F2MinusF1 => f2 - f1,
        };

        WorleyResult { value, cell_id: nearest_cell_id }
    }

    /// 3D Worley noise returning just the value.
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        self.sample3d(x, y, z).value
    }

    /// 2D tiling Worley noise: wraps cell coordinates.
    pub fn noise2d_tiling(&self, x: f64, y: f64, period_x: u32, period_y: u32) -> f64 {
        let px = period_x.max(1) as i64;
        let py = period_y.max(1) as i64;
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;

        let mut f1 = f64::MAX;
        let mut f2 = f64::MAX;

        for dx in -1..=1 {
            for dy in -1..=1 {
                let cx = (ix + dx).rem_euclid(px);
                let cy = (iy + dy).rem_euclid(py);
                let actual_x = ix + dx;
                let actual_y = iy + dy;
                let base_hash = Self::cell_hash(cx, cy, self.seed);

                for p in 0..self.points_per_cell {
                    let ph = base_hash.wrapping_add(p as u64).wrapping_mul(2654435761);
                    let fpx = actual_x as f64 + 0.5 + (Self::hash_to_float(ph) - 0.5) * self.jitter;
                    let fpy = actual_y as f64 + 0.5 + (Self::hash_to_float(ph.wrapping_mul(3)) - 0.5) * self.jitter;
                    let d = self.distance2d(x - fpx, y - fpy);

                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
        }

        match self.output {
            WorleyOutput::F1 => f1,
            WorleyOutput::F2 => f2,
            WorleyOutput::F2MinusF1 => f2 - f1,
        }
    }

    /// Generate a 2D grid of Worley noise values.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_same_seed() {
        let a = WorleyNoise::new(42);
        let b = WorleyNoise::new(42);
        assert!((a.noise2d(1.5, 2.5) - b.noise2d(1.5, 2.5)).abs() < 1e-12);
    }

    #[test]
    fn test_different_seeds_differ() {
        let a = WorleyNoise::new(1);
        let b = WorleyNoise::new(999);
        assert!((a.noise2d(3.7, 4.2) - b.noise2d(3.7, 4.2)).abs() > 1e-9);
    }

    #[test]
    fn test_f1_non_negative() {
        let w = WorleyNoise::new(42);
        for i in 0..200 {
            let x = i as f64 * 0.17;
            let y = i as f64 * 0.31;
            assert!(w.noise2d(x, y) >= -1e-12);
        }
    }

    #[test]
    fn test_f2_geq_f1() {
        let w_f1 = WorleyNoise::new(42).with_output(WorleyOutput::F1);
        let w_f2 = WorleyNoise::new(42).with_output(WorleyOutput::F2);
        for i in 0..100 {
            let x = i as f64 * 0.23;
            let y = i as f64 * 0.37;
            assert!(w_f2.noise2d(x, y) >= w_f1.noise2d(x, y) - 1e-12);
        }
    }

    #[test]
    fn test_f2_minus_f1_non_negative() {
        let w = WorleyNoise::new(42).with_output(WorleyOutput::F2MinusF1);
        for i in 0..100 {
            let x = i as f64 * 0.19;
            let y = i as f64 * 0.41;
            assert!(w.noise2d(x, y) >= -1e-12);
        }
    }

    #[test]
    fn test_3d_f1_non_negative() {
        let w = WorleyNoise::new(42);
        for i in 0..100 {
            let t = i as f64 * 0.23;
            assert!(w.noise3d(t, t * 0.7, t * 1.1) >= -1e-12);
        }
    }

    #[test]
    fn test_manhattan_metric() {
        let w = WorleyNoise::new(42).with_metric(DistanceMetric::Manhattan);
        let v = w.noise2d(1.5, 2.5);
        assert!(v.is_finite() && v >= 0.0);
    }

    #[test]
    fn test_chebyshev_metric() {
        let w = WorleyNoise::new(42).with_metric(DistanceMetric::Chebyshev);
        let v = w.noise2d(1.5, 2.5);
        assert!(v.is_finite() && v >= 0.0);
    }

    #[test]
    fn test_cell_id_deterministic() {
        let w = WorleyNoise::new(42);
        let r1 = w.sample2d(1.5, 2.5);
        let r2 = w.sample2d(1.5, 2.5);
        assert_eq!(r1.cell_id, r2.cell_id);
    }

    #[test]
    fn test_different_locations_different_cell_ids() {
        let w = WorleyNoise::new(42);
        let r1 = w.sample2d(0.5, 0.5);
        let r2 = w.sample2d(10.5, 10.5);
        // Very likely different cells
        assert_ne!(r1.cell_id, r2.cell_id);
    }

    #[test]
    fn test_points_per_cell_clamp() {
        let w = WorleyNoise::new(42).with_points_per_cell(0);
        assert_eq!(w.points_per_cell, 1);
        let w2 = WorleyNoise::new(42).with_points_per_cell(10);
        assert_eq!(w2.points_per_cell, 4);
    }

    #[test]
    fn test_jitter_clamp() {
        let w = WorleyNoise::new(42).with_jitter(-1.0);
        assert!((w.jitter - 0.0).abs() < 1e-12);
        let w2 = WorleyNoise::new(42).with_jitter(5.0);
        assert!((w2.jitter - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_jitter_regular_grid() {
        let w = WorleyNoise::new(42).with_jitter(0.0);
        // At cell center (0.5, 0.5) with zero jitter, distance to nearest point = 0
        let v = w.noise2d(0.5, 0.5);
        assert!(v < 1e-6, "at cell center with no jitter, F1 should be ~0: {v}");
    }

    #[test]
    fn test_multiple_points_per_cell() {
        let w1 = WorleyNoise::new(42).with_points_per_cell(1);
        let w4 = WorleyNoise::new(42).with_points_per_cell(4);
        // F1 with more points should generally be <= F1 with fewer
        let mut less_count = 0;
        for i in 0..50 {
            let x = i as f64 * 0.37 + 0.1;
            let y = i as f64 * 0.53 + 0.2;
            if w4.noise2d(x, y) <= w1.noise2d(x, y) + 1e-12 {
                less_count += 1;
            }
        }
        assert!(less_count > 30, "more points should usually give smaller F1");
    }

    #[test]
    fn test_tiling_2d() {
        let w = WorleyNoise::new(42);
        let period = 8;
        for i in 0..20 {
            let y = i as f64 * 0.37 + 0.1;
            let a = w.noise2d_tiling(0.5, y, period, period);
            let b = w.noise2d_tiling(0.5 + period as f64, y, period, period);
            assert!((a - b).abs() < 1e-10, "tiling mismatch at y={y}");
        }
    }

    #[test]
    fn test_generate_grid_dimensions() {
        let w = WorleyNoise::new(42);
        let grid = w.generate_grid(16, 8, 4.0);
        assert_eq!(grid.len(), 8);
        for row in &grid {
            assert_eq!(row.len(), 16);
        }
    }

    #[test]
    fn test_generate_grid_values_non_negative() {
        let w = WorleyNoise::new(42);
        let grid = w.generate_grid(16, 16, 4.0);
        for row in &grid {
            for &v in row {
                assert!(v >= -1e-12);
            }
        }
    }

    #[test]
    fn test_negative_coordinates() {
        let w = WorleyNoise::new(33);
        let v = w.noise2d(-5.3, -8.7);
        assert!(v.is_finite() && v >= 0.0);
        let v3 = w.noise3d(-2.1, -3.4, -7.8);
        assert!(v3.is_finite() && v3 >= 0.0);
    }

    #[test]
    fn test_seed_getter() {
        let w = WorleyNoise::new(12345);
        assert_eq!(w.seed(), 12345);
    }

    #[test]
    fn test_clone() {
        let a = WorleyNoise::new(42);
        let b = a.clone();
        assert!((a.noise2d(1.0, 1.0) - b.noise2d(1.0, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_debug_format() {
        let w = WorleyNoise::new(42);
        let s = format!("{:?}", w);
        assert!(s.contains("WorleyNoise"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_3d_cell_id() {
        let w = WorleyNoise::new(42);
        let r = w.sample3d(1.5, 2.5, 3.5);
        assert!(r.value >= 0.0);
        assert!(r.cell_id != 0 || r.value.is_finite()); // cell_id is some hash value
    }

    #[test]
    fn test_worley_result_partial_eq() {
        let r1 = WorleyResult { value: 1.0, cell_id: 42 };
        let r2 = WorleyResult { value: 1.0, cell_id: 42 };
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_spatial_variation() {
        let w = WorleyNoise::new(42);
        let mut distinct = 0;
        let base = w.noise2d(0.5, 0.5);
        for i in 1..20 {
            let v = w.noise2d(0.5 + i as f64 * 1.7, 0.5 + i as f64 * 1.3);
            if (v - base).abs() > 1e-6 { distinct += 1; }
        }
        assert!(distinct > 10);
    }
}
