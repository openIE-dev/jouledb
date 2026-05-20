//! Terrain — heightmap grid, bilinear interpolation, normal computation,
//! diamond-square fractal generation, LOD levels, and thermal erosion.

use crate::webgl::Vec3;

// ── Heightmap ─────────────────────────────────────────────────

/// A 2D grid of f64 heights representing terrain.
#[derive(Debug, Clone)]
pub struct Heightmap {
    /// Width (number of columns).
    pub width: usize,
    /// Depth (number of rows).
    pub depth: usize,
    /// Row-major height data: heights[z * width + x].
    pub heights: Vec<f64>,
    /// World-space spacing between grid points.
    pub cell_size: f64,
}

impl Heightmap {
    /// Create a flat heightmap at the given height.
    pub fn new(width: usize, depth: usize, cell_size: f64) -> Self {
        Self {
            width,
            depth,
            heights: vec![0.0; width * depth],
            cell_size,
        }
    }

    /// Get height at integer grid coordinates.
    pub fn get(&self, x: usize, z: usize) -> f64 {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x]
        } else {
            0.0
        }
    }

    /// Set height at integer grid coordinates.
    pub fn set(&mut self, x: usize, z: usize, h: f64) {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x] = h;
        }
    }

    /// Query height at world-space (wx, wz) with bilinear interpolation.
    pub fn height_at(&self, wx: f64, wz: f64) -> f64 {
        let fx = wx / self.cell_size;
        let fz = wz / self.cell_size;

        let x0 = fx.floor() as i64;
        let z0 = fz.floor() as i64;
        let x1 = x0 + 1;
        let z1 = z0 + 1;

        let tx = fx - x0 as f64;
        let tz = fz - z0 as f64;

        let h00 = self.get_clamped(x0, z0);
        let h10 = self.get_clamped(x1, z0);
        let h01 = self.get_clamped(x0, z1);
        let h11 = self.get_clamped(x1, z1);

        let h0 = h00 + (h10 - h00) * tx;
        let h1 = h01 + (h11 - h01) * tx;
        h0 + (h1 - h0) * tz
    }

    /// Get height with clamped coordinates.
    fn get_clamped(&self, x: i64, z: i64) -> f64 {
        let cx = x.clamp(0, self.width as i64 - 1) as usize;
        let cz = z.clamp(0, self.depth as i64 - 1) as usize;
        self.heights[cz * self.width + cx]
    }

    /// Compute the surface normal at world-space (wx, wz) using central differences.
    pub fn normal_at(&self, wx: f64, wz: f64) -> Vec3 {
        let eps = self.cell_size * 0.5;
        let hx0 = self.height_at(wx - eps, wz);
        let hx1 = self.height_at(wx + eps, wz);
        let hz0 = self.height_at(wx, wz - eps);
        let hz1 = self.height_at(wx, wz + eps);

        let dx = (hx1 - hx0) / (2.0 * eps);
        let dz = (hz1 - hz0) / (2.0 * eps);

        Vec3::new(-dx, 1.0, -dz).normalize()
    }

    /// Get the min and max height values.
    pub fn height_range(&self) -> (f64, f64) {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for &h in &self.heights {
            if h < min { min = h; }
            if h > max { max = h; }
        }
        if self.heights.is_empty() {
            (0.0, 0.0)
        } else {
            (min, max)
        }
    }

    /// Generate a LOD level by subsampling the grid. `factor` must be >= 1.
    /// Returns a new heightmap with approximately `width/factor` x `depth/factor` points.
    pub fn lod(&self, factor: usize) -> Heightmap {
        let factor = factor.max(1);
        let new_w = ((self.width - 1) / factor + 1).max(1);
        let new_d = ((self.depth - 1) / factor + 1).max(1);
        let new_cell = self.cell_size * factor as f64;

        let mut hm = Heightmap::new(new_w, new_d, new_cell);
        for nz in 0..new_d {
            for nx in 0..new_w {
                let ox = (nx * factor).min(self.width - 1);
                let oz = (nz * factor).min(self.depth - 1);
                hm.set(nx, nz, self.get(ox, oz));
            }
        }
        hm
    }

    // ── Diamond-Square Generation ─────────────────────────────

    /// Generate fractal terrain using the diamond-square algorithm.
    /// `size` must be (2^n + 1) for some n. `roughness` controls amplitude decay.
    /// `seed` seeds a simple PRNG.
    pub fn diamond_square(size: usize, roughness: f64, seed: u64) -> Self {
        // Ensure size is 2^n + 1.
        let n = ((size - 1) as f64).log2().ceil() as u32;
        let actual_size = (1 << n) + 1;

        let mut hm = Heightmap::new(actual_size, actual_size, 1.0);
        let mut rng = DsRng::new(seed);

        // Seed corners.
        hm.set(0, 0, rng.range(-1.0, 1.0));
        hm.set(actual_size - 1, 0, rng.range(-1.0, 1.0));
        hm.set(0, actual_size - 1, rng.range(-1.0, 1.0));
        hm.set(actual_size - 1, actual_size - 1, rng.range(-1.0, 1.0));

        let mut step = actual_size - 1;
        let mut scale = roughness;

        while step > 1 {
            let half = step / 2;

            // Diamond step.
            let mut z = 0;
            while z < actual_size - 1 {
                let mut x = 0;
                while x < actual_size - 1 {
                    let avg = (hm.get(x, z)
                        + hm.get(x + step, z)
                        + hm.get(x, z + step)
                        + hm.get(x + step, z + step))
                        / 4.0;
                    hm.set(x + half, z + half, avg + rng.range(-scale, scale));
                    x += step;
                }
                z += step;
            }

            // Square step.
            let mut z = 0;
            while z < actual_size {
                let start_x = if (z / half) % 2 == 0 { half } else { 0 };
                let mut x = start_x;
                while x < actual_size {
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    if x >= half {
                        sum += hm.get(x - half, z);
                        count += 1.0;
                    }
                    if x + half < actual_size {
                        sum += hm.get(x + half, z);
                        count += 1.0;
                    }
                    if z >= half {
                        sum += hm.get(x, z - half);
                        count += 1.0;
                    }
                    if z + half < actual_size {
                        sum += hm.get(x, z + half);
                        count += 1.0;
                    }
                    hm.set(x, z, sum / count + rng.range(-scale, scale));
                    x += step;
                }
                z += half;
            }

            step = half;
            scale *= 0.5;
        }

        hm
    }

    // ── Thermal Erosion ───────────────────────────────────────

    /// Simulate thermal erosion: material moves downhill if slope exceeds `talus_angle`.
    /// `iterations` controls how many passes to run. `amount` is the fraction of
    /// excess height moved per step.
    pub fn thermal_erosion(&mut self, iterations: usize, talus_angle: f64, amount: f64) {
        let talus = talus_angle.tan() * self.cell_size;

        for _ in 0..iterations {
            let prev = self.heights.clone();

            for z in 0..self.depth {
                for x in 0..self.width {
                    let h = prev[z * self.width + x];
                    let mut max_diff = 0.0f64;
                    let mut total_diff = 0.0;
                    let mut neighbor_count = 0;

                    let neighbors = self.neighbor_indices(x, z);
                    let mut diffs = Vec::new();

                    for (nx, nz) in &neighbors {
                        let nh = prev[nz * self.width + nx];
                        let diff = h - nh;
                        if diff > talus {
                            diffs.push((*nx, *nz, diff));
                            total_diff += diff;
                            if diff > max_diff {
                                max_diff = diff;
                            }
                            neighbor_count += 1;
                        } else {
                            diffs.push((*nx, *nz, 0.0));
                        }
                    }

                    if neighbor_count == 0 || total_diff < 1e-12 {
                        continue;
                    }

                    // Distribute material proportionally.
                    let move_total = max_diff * amount * 0.5;
                    self.heights[z * self.width + x] -= move_total;
                    for (nx, nz, diff) in &diffs {
                        if *diff > talus {
                            let share = diff / total_diff;
                            self.heights[nz * self.width + nx] += move_total * share;
                        }
                    }
                }
            }
        }
    }

    fn neighbor_indices(&self, x: usize, z: usize) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for dz in [-1i64, 0, 1] {
            for dx in [-1i64, 0, 1] {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let nx = x as i64 + dx;
                let nz = z as i64 + dz;
                if nx >= 0 && nx < self.width as i64 && nz >= 0 && nz < self.depth as i64 {
                    result.push((nx as usize, nz as usize));
                }
            }
        }
        result
    }
}

/// Simple xorshift PRNG for diamond-square.
struct DsRng {
    state: u64,
}

impl DsRng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn range(&mut self, min: f64, max: f64) -> f64 {
        let t = (self.next_u64() % 1_000_000) as f64 / 1_000_000.0;
        min + t * (max - min)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn flat_heightmap() {
        let hm = Heightmap::new(10, 10, 1.0);
        assert!((hm.get(5, 5)).abs() < EPS);
        assert!((hm.height_at(5.5, 5.5)).abs() < EPS);
    }

    #[test]
    fn set_and_get() {
        let mut hm = Heightmap::new(10, 10, 1.0);
        hm.set(3, 4, 7.5);
        assert!((hm.get(3, 4) - 7.5).abs() < EPS);
    }

    #[test]
    fn bilinear_interpolation() {
        let mut hm = Heightmap::new(3, 3, 1.0);
        hm.set(0, 0, 0.0);
        hm.set(1, 0, 10.0);
        hm.set(0, 1, 0.0);
        hm.set(1, 1, 10.0);
        // Midpoint should be 5.0.
        let h = hm.height_at(0.5, 0.5);
        assert!((h - 5.0).abs() < EPS, "h = {h}");
    }

    #[test]
    fn normal_on_flat_is_up() {
        let hm = Heightmap::new(10, 10, 1.0);
        let n = hm.normal_at(5.0, 5.0);
        assert!((n.x).abs() < EPS);
        assert!((n.y - 1.0).abs() < EPS);
        assert!((n.z).abs() < EPS);
    }

    #[test]
    fn normal_on_slope() {
        let mut hm = Heightmap::new(10, 10, 1.0);
        // Create a slope in the X direction.
        for z in 0..10 {
            for x in 0..10 {
                hm.set(x, z, x as f64);
            }
        }
        let n = hm.normal_at(5.0, 5.0);
        // Normal should tilt in the -X direction.
        assert!(n.x < 0.0, "n.x = {}", n.x);
        assert!(n.y > 0.0);
    }

    #[test]
    fn diamond_square_produces_variation() {
        let hm = Heightmap::diamond_square(17, 1.0, 42);
        assert_eq!(hm.width, 17);
        assert_eq!(hm.depth, 17);
        let (min, max) = hm.height_range();
        assert!(max > min, "terrain should have variation: min={min}, max={max}");
    }

    #[test]
    fn diamond_square_deterministic() {
        let a = Heightmap::diamond_square(9, 0.8, 123);
        let b = Heightmap::diamond_square(9, 0.8, 123);
        assert_eq!(a.heights, b.heights);
    }

    #[test]
    fn lod_reduces_size() {
        let hm = Heightmap::diamond_square(33, 1.0, 7);
        let lod2 = hm.lod(2);
        assert!(lod2.width < hm.width);
        assert!(lod2.depth < hm.depth);
        // LOD cell size is doubled.
        assert!((lod2.cell_size - 2.0).abs() < EPS);
    }

    #[test]
    fn lod_1_preserves_size() {
        let hm = Heightmap::new(10, 10, 1.0);
        let lod1 = hm.lod(1);
        assert_eq!(lod1.width, hm.width);
        assert_eq!(lod1.depth, hm.depth);
    }

    #[test]
    fn thermal_erosion_reduces_peak() {
        let mut hm = Heightmap::new(5, 5, 1.0);
        // Create a sharp peak.
        hm.set(2, 2, 100.0);
        let peak_before = hm.get(2, 2);
        hm.thermal_erosion(10, 0.1, 0.5);
        let peak_after = hm.get(2, 2);
        assert!(
            peak_after < peak_before,
            "erosion should reduce peak: before={peak_before}, after={peak_after}"
        );
    }

    #[test]
    fn thermal_erosion_conserves_mass_roughly() {
        let mut hm = Heightmap::new(5, 5, 1.0);
        hm.set(2, 2, 50.0);
        let total_before: f64 = hm.heights.iter().sum();
        hm.thermal_erosion(5, 0.1, 0.5);
        let total_after: f64 = hm.heights.iter().sum();
        // Should be approximately conserved (within floating-point tolerance).
        assert!(
            (total_before - total_after).abs() < 1.0,
            "mass not conserved: before={total_before}, after={total_after}"
        );
    }

    #[test]
    fn height_range_on_varied_terrain() {
        let mut hm = Heightmap::new(5, 5, 1.0);
        hm.set(0, 0, -3.0);
        hm.set(4, 4, 7.0);
        let (min, max) = hm.height_range();
        assert!((min - (-3.0)).abs() < EPS);
        assert!((max - 7.0).abs() < EPS);
    }
}
