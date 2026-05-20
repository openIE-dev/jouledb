// Procedural Terrain Generation — Hydraulic erosion simulation on heightmap
// Droplet-based (particle) erosion, gradient via bilinear interpolation, thermal erosion

use std::fmt;

/// Configuration for hydraulic erosion simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct ErosionConfig {
    pub iterations: u32,
    pub seed: u64,
    pub max_lifetime: u32,
    pub inertia: f64,
    pub sediment_capacity_factor: f64,
    pub min_sediment_capacity: f64,
    pub erosion_rate: f64,
    pub deposition_rate: f64,
    pub evaporation_rate: f64,
    pub gravity: f64,
    pub initial_water: f64,
    pub initial_speed: f64,
    pub erosion_radius: u32,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            iterations: 10000,
            seed: 42,
            max_lifetime: 64,
            inertia: 0.05,
            sediment_capacity_factor: 4.0,
            min_sediment_capacity: 0.01,
            erosion_rate: 0.3,
            deposition_rate: 0.3,
            evaporation_rate: 0.01,
            gravity: 4.0,
            initial_water: 1.0,
            initial_speed: 1.0,
            erosion_radius: 3,
        }
    }
}

/// Configuration for thermal erosion.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalConfig {
    pub iterations: u32,
    pub talus_angle: f64,
    pub transfer_rate: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            talus_angle: 0.05,
            transfer_rate: 0.5,
        }
    }
}

/// A heightmap for erosion simulation.
#[derive(Clone)]
pub struct Heightmap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl fmt::Debug for Heightmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Heightmap")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl Heightmap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn from_data(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, x: usize, y: usize, val: f64) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val;
        }
    }

    /// Bilinear interpolation at fractional coordinates.
    pub fn sample(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();

        let get_safe = |px: i64, py: i64| -> f64 {
            if px >= 0 && px < self.width as i64 && py >= 0 && py < self.height as i64 {
                self.data[py as usize * self.width + px as usize]
            } else {
                0.0
            }
        };

        let v00 = get_safe(ix, iy);
        let v10 = get_safe(ix + 1, iy);
        let v01 = get_safe(ix, iy + 1);
        let v11 = get_safe(ix + 1, iy + 1);

        let top = v00 * (1.0 - fx) + v10 * fx;
        let bot = v01 * (1.0 - fx) + v11 * fx;
        top * (1.0 - fy) + bot * fy
    }

    /// Compute gradient at a position using bilinear interpolation of neighbors.
    pub fn gradient(&self, x: f64, y: f64) -> (f64, f64) {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();

        let get_safe = |px: i64, py: i64| -> f64 {
            if px >= 0 && px < self.width as i64 && py >= 0 && py < self.height as i64 {
                self.data[py as usize * self.width + px as usize]
            } else {
                0.0
            }
        };

        let v00 = get_safe(ix, iy);
        let v10 = get_safe(ix + 1, iy);
        let v01 = get_safe(ix, iy + 1);
        let v11 = get_safe(ix + 1, iy + 1);

        let gx = (v10 - v00) * (1.0 - fy) + (v11 - v01) * fy;
        let gy = (v01 - v00) * (1.0 - fx) + (v11 - v10) * fx;
        (gx, gy)
    }

    /// Find the minimum and maximum height values.
    pub fn min_max(&self) -> (f64, f64) {
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for &v in &self.data {
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        (lo, hi)
    }

    /// Normalize heights to [0, 1].
    pub fn normalize(&mut self) {
        let (lo, hi) = self.min_max();
        let range = hi - lo;
        if range.abs() < 1e-12 { return; }
        for v in &mut self.data {
            *v = (*v - lo) / range;
        }
    }
}

/// Hydraulic erosion simulator.
pub struct ErosionSim {
    config: ErosionConfig,
}

impl fmt::Debug for ErosionSim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErosionSim").field("config", &self.config).finish()
    }
}

impl ErosionSim {
    pub fn new(config: ErosionConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ErosionConfig {
        &self.config
    }

    /// Simple seeded RNG.
    fn next_rng(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f64 / (u32::MAX as f64)
    }

    /// Run hydraulic erosion on the heightmap.
    pub fn erode(&self, heightmap: &mut Heightmap) {
        let w = heightmap.width;
        let h = heightmap.height;
        if w < 3 || h < 3 { return; }

        let mut rng_state = self.config.seed;

        for _ in 0..self.config.iterations {
            let start_x = Self::next_rng(&mut rng_state) * (w as f64 - 2.0) + 1.0;
            let start_y = Self::next_rng(&mut rng_state) * (h as f64 - 2.0) + 1.0;

            let mut px = start_x;
            let mut py = start_y;
            let mut dir_x = 0.0f64;
            let mut dir_y = 0.0f64;
            let mut speed = self.config.initial_speed;
            let mut water = self.config.initial_water;
            let mut sediment = 0.0f64;

            for _ in 0..self.config.max_lifetime {
                let ix = px.floor() as usize;
                let iy = py.floor() as usize;
                if ix >= w - 1 || iy >= h - 1 || ix == 0 || iy == 0 {
                    break;
                }

                let (gx, gy) = heightmap.gradient(px, py);
                let old_height = heightmap.sample(px, py);

                // Update direction with inertia
                dir_x = dir_x * self.config.inertia - gx * (1.0 - self.config.inertia);
                dir_y = dir_y * self.config.inertia - gy * (1.0 - self.config.inertia);

                let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
                if dir_len < 1e-10 {
                    // Random direction
                    dir_x = Self::next_rng(&mut rng_state) * 2.0 - 1.0;
                    dir_y = Self::next_rng(&mut rng_state) * 2.0 - 1.0;
                    let dl = (dir_x * dir_x + dir_y * dir_y).sqrt();
                    if dl < 1e-10 { break; }
                    dir_x /= dl;
                    dir_y /= dl;
                } else {
                    dir_x /= dir_len;
                    dir_y /= dir_len;
                }

                let new_x = px + dir_x;
                let new_y = py + dir_y;

                if new_x < 1.0 || new_x >= (w - 2) as f64
                    || new_y < 1.0 || new_y >= (h - 2) as f64
                {
                    break;
                }

                let new_height = heightmap.sample(new_x, new_y);
                let height_diff = new_height - old_height;

                let sed_capacity = ((-height_diff).max(0.0) * speed * water
                    * self.config.sediment_capacity_factor)
                    .max(self.config.min_sediment_capacity);

                if sediment > sed_capacity || height_diff > 0.0 {
                    // Deposit sediment
                    let deposit = if height_diff > 0.0 {
                        sediment.min(height_diff)
                    } else {
                        (sediment - sed_capacity) * self.config.deposition_rate
                    };
                    sediment -= deposit;
                    // Distribute deposit to the four surrounding cells
                    let fx = px - px.floor();
                    let fy = py - py.floor();
                    heightmap.data[iy * w + ix] += deposit * (1.0 - fx) * (1.0 - fy);
                    heightmap.data[iy * w + ix + 1] += deposit * fx * (1.0 - fy);
                    heightmap.data[(iy + 1) * w + ix] += deposit * (1.0 - fx) * fy;
                    heightmap.data[(iy + 1) * w + ix + 1] += deposit * fx * fy;
                } else {
                    // Erode
                    let erode_amount = ((sed_capacity - sediment) * self.config.erosion_rate)
                        .min(-height_diff);
                    let radius = self.config.erosion_radius as i64;
                    let mut weight_sum = 0.0;
                    let mut affected: Vec<(usize, f64)> = Vec::new();

                    for ey in -radius..=radius {
                        for ex in -radius..=radius {
                            let ax = ix as i64 + ex;
                            let ay = iy as i64 + ey;
                            if ax >= 0 && ax < w as i64 && ay >= 0 && ay < h as i64 {
                                let dist = ((ex * ex + ey * ey) as f64).sqrt();
                                if dist <= radius as f64 {
                                    let wt = (1.0 - dist / (radius as f64 + 1.0)).max(0.0);
                                    weight_sum += wt;
                                    affected.push((ay as usize * w + ax as usize, wt));
                                }
                            }
                        }
                    }

                    if weight_sum > 0.0 {
                        for (idx, wt) in &affected {
                            heightmap.data[*idx] -= erode_amount * wt / weight_sum;
                        }
                    }
                    sediment += erode_amount;
                }

                // Update speed and water
                speed = (speed * speed + height_diff * self.config.gravity)
                    .max(0.0)
                    .sqrt();
                water *= 1.0 - self.config.evaporation_rate;

                px = new_x;
                py = new_y;

                if water < 0.001 { break; }
            }
        }
    }

    /// Thermal erosion: move material downhill where slopes exceed talus angle.
    pub fn thermal_erode(heightmap: &mut Heightmap, config: &ThermalConfig) {
        let w = heightmap.width;
        let h = heightmap.height;
        if w < 3 || h < 3 { return; }

        let neighbors: [(i64, i64); 8] = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0),           (1, 0),
            (-1, 1),  (0, 1),  (1, 1),
        ];

        for _ in 0..config.iterations {
            // Collect changes first to avoid aliasing
            let mut changes: Vec<(usize, f64)> = Vec::new();

            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let center = heightmap.data[y * w + x];
                    let mut max_diff = 0.0f64;
                    let mut total_diff = 0.0f64;
                    let mut lower_neighbors: Vec<(usize, f64)> = Vec::new();

                    for &(dx, dy) in &neighbors {
                        let nx = (x as i64 + dx) as usize;
                        let ny = (y as i64 + dy) as usize;
                        let nh = heightmap.data[ny * w + nx];
                        let diff = center - nh;
                        if diff > config.talus_angle {
                            if diff > max_diff { max_diff = diff; }
                            total_diff += diff;
                            lower_neighbors.push((ny * w + nx, diff));
                        }
                    }

                    if !lower_neighbors.is_empty() && total_diff > 0.0 {
                        let move_amount = max_diff * config.transfer_rate * 0.5;
                        changes.push((y * w + x, -move_amount));
                        for (idx, diff) in &lower_neighbors {
                            changes.push((*idx, move_amount * diff / total_diff));
                        }
                    }
                }
            }

            for (idx, delta) in &changes {
                heightmap.data[*idx] += delta;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sloped_heightmap(w: usize, h: usize) -> Heightmap {
        let mut data = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = 1.0 - (x as f64 + y as f64) / ((w + h) as f64);
            }
        }
        Heightmap::from_data(w, h, data)
    }

    fn make_peak_heightmap(w: usize, h: usize) -> Heightmap {
        let cx = w as f64 / 2.0;
        let cy = h as f64 / 2.0;
        let mut data = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                data[y * w + x] = (1.0 - (dx * dx + dy * dy).sqrt() / cx).max(0.0);
            }
        }
        Heightmap::from_data(w, h, data)
    }

    #[test]
    fn test_heightmap_new() {
        let hm = Heightmap::new(16, 16);
        assert_eq!(hm.width, 16);
        assert_eq!(hm.height, 16);
        assert_eq!(hm.data.len(), 256);
    }

    #[test]
    fn test_heightmap_get_set() {
        let mut hm = Heightmap::new(8, 8);
        hm.set(3, 5, 0.75);
        assert!((hm.get(3, 5) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_heightmap_out_of_bounds() {
        let hm = Heightmap::new(8, 8);
        assert!((hm.get(100, 100) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_heightmap_sample_at_integer() {
        let mut hm = Heightmap::new(8, 8);
        hm.set(3, 4, 0.5);
        assert!((hm.sample(3.0, 4.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_heightmap_sample_interpolation() {
        let mut hm = Heightmap::new(8, 8);
        hm.set(2, 2, 0.0);
        hm.set(3, 2, 1.0);
        hm.set(2, 3, 0.0);
        hm.set(3, 3, 1.0);
        let v = hm.sample(2.5, 2.5);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_gradient_flat() {
        let hm = Heightmap::from_data(4, 4, vec![0.5; 16]);
        let (gx, gy) = hm.gradient(1.5, 1.5);
        assert!(gx.abs() < 1e-12);
        assert!(gy.abs() < 1e-12);
    }

    #[test]
    fn test_gradient_slope_x() {
        let mut hm = Heightmap::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                hm.set(x, y, x as f64);
            }
        }
        let (gx, _gy) = hm.gradient(1.5, 1.5);
        assert!(gx > 0.5, "gradient should be positive in x: {gx}");
    }

    #[test]
    fn test_min_max() {
        let hm = Heightmap::from_data(2, 2, vec![0.1, 0.5, 0.3, 0.9]);
        let (lo, hi) = hm.min_max();
        assert!((lo - 0.1).abs() < 1e-12);
        assert!((hi - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_normalize() {
        let mut hm = Heightmap::from_data(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        hm.normalize();
        let (lo, hi) = hm.min_max();
        assert!((lo - 0.0).abs() < 1e-12);
        assert!((hi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalize_flat() {
        let mut hm = Heightmap::from_data(2, 2, vec![5.0; 4]);
        hm.normalize();
        // Should not crash; all values remain 5.0 (no range to normalize)
        for &v in &hm.data {
            assert!((v - 5.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_erosion_default_config() {
        let cfg = ErosionConfig::default();
        assert_eq!(cfg.iterations, 10000);
        assert!((cfg.inertia - 0.05).abs() < 1e-12);
    }

    #[test]
    fn test_erosion_modifies_terrain() {
        let mut hm = make_peak_heightmap(32, 32);
        let before: Vec<f64> = hm.data.clone();
        let sim = ErosionSim::new(ErosionConfig {
            iterations: 500,
            ..ErosionConfig::default()
        });
        sim.erode(&mut hm);
        let changed = hm.data.iter().zip(before.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "erosion should modify the heightmap");
    }

    #[test]
    fn test_erosion_lowers_peak() {
        let mut hm = make_peak_heightmap(32, 32);
        let peak_before = hm.get(16, 16);
        let sim = ErosionSim::new(ErosionConfig {
            iterations: 2000,
            ..ErosionConfig::default()
        });
        sim.erode(&mut hm);
        let peak_after = hm.get(16, 16);
        assert!(
            peak_after < peak_before + 1e-6,
            "erosion should lower the peak: before={peak_before}, after={peak_after}"
        );
    }

    #[test]
    fn test_erosion_deterministic() {
        let mut hm1 = make_sloped_heightmap(16, 16);
        let mut hm2 = make_sloped_heightmap(16, 16);
        let cfg = ErosionConfig {
            iterations: 200,
            seed: 42,
            ..ErosionConfig::default()
        };
        ErosionSim::new(cfg.clone()).erode(&mut hm1);
        ErosionSim::new(cfg).erode(&mut hm2);
        for i in 0..hm1.data.len() {
            assert!((hm1.data[i] - hm2.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_erosion_small_map_no_crash() {
        let mut hm = Heightmap::new(2, 2);
        let sim = ErosionSim::new(ErosionConfig::default());
        sim.erode(&mut hm); // Should not crash on tiny map
    }

    #[test]
    fn test_thermal_default_config() {
        let cfg = ThermalConfig::default();
        assert_eq!(cfg.iterations, 50);
        assert!(cfg.talus_angle > 0.0);
    }

    #[test]
    fn test_thermal_erosion_smooths() {
        let mut hm = make_peak_heightmap(32, 32);
        let peak_before = hm.get(16, 16);
        ErosionSim::thermal_erode(&mut hm, &ThermalConfig::default());
        let peak_after = hm.get(16, 16);
        assert!(
            peak_after <= peak_before + 1e-6,
            "thermal erosion should smooth peak"
        );
    }

    #[test]
    fn test_thermal_erosion_flat_unchanged() {
        let mut hm = Heightmap::from_data(8, 8, vec![0.5; 64]);
        let before: Vec<f64> = hm.data.clone();
        ErosionSim::thermal_erode(&mut hm, &ThermalConfig::default());
        for i in 0..hm.data.len() {
            assert!((hm.data[i] - before[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_thermal_small_map_no_crash() {
        let mut hm = Heightmap::new(2, 2);
        ErosionSim::thermal_erode(&mut hm, &ThermalConfig::default());
    }

    #[test]
    fn test_heightmap_clone() {
        let hm = make_peak_heightmap(8, 8);
        let hm2 = hm.clone();
        assert_eq!(hm.data, hm2.data);
    }

    #[test]
    fn test_heightmap_debug() {
        let hm = Heightmap::new(8, 8);
        let s = format!("{:?}", hm);
        assert!(s.contains("Heightmap"));
        assert!(s.contains("8"));
    }

    #[test]
    fn test_erosion_sim_debug() {
        let sim = ErosionSim::new(ErosionConfig::default());
        let s = format!("{:?}", sim);
        assert!(s.contains("ErosionSim"));
    }

    #[test]
    fn test_erosion_config_partial_eq() {
        let a = ErosionConfig::default();
        let b = ErosionConfig::default();
        assert_eq!(a, b);
    }
}
