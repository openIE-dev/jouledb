//! # Sensor Model
//!
//! Probabilistic sensor models for robotics applications. Implements beam
//! models, likelihood field models, ray-casting simulation, and measurement
//! noise models used in localization and mapping.

use std::fmt;

// ── Measurement Noise ──

/// Gaussian noise model for sensor measurements.
#[derive(Clone, Debug)]
pub struct GaussianNoise {
    pub mean: f64,
    pub std_dev: f64,
    seed: u64,
}

impl GaussianNoise {
    pub fn new(std_dev: f64) -> Self {
        Self { mean: 0.0, std_dev, seed: 12345 }
    }

    pub fn with_mean(mut self, mean: f64) -> Self {
        self.mean = mean;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn next_uniform(&mut self) -> f64 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.seed >> 33) as f64 / (1u64 << 31) as f64
    }

    /// Generate a sample using Box-Muller transform.
    pub fn sample(&mut self) -> f64 {
        let u1 = self.next_uniform().max(1e-15);
        let u2 = self.next_uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        self.mean + self.std_dev * z
    }

    /// Evaluate the Gaussian PDF at a given value.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        let norm = 1.0 / (self.std_dev * (2.0 * std::f64::consts::PI).sqrt());
        norm * (-0.5 * z * z).exp()
    }

    /// Log-likelihood of a measurement.
    pub fn log_likelihood(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std_dev;
        -0.5 * z * z - self.std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

impl fmt::Display for GaussianNoise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GaussianNoise(mean={:.3}, std={:.3})", self.mean, self.std_dev)
    }
}

// ── Beam Model ──

/// Beam model for range sensors (laser/sonar), combining hit, short, max, and random components.
#[derive(Clone, Debug)]
pub struct BeamModel {
    pub z_hit: f64,
    pub z_short: f64,
    pub z_max: f64,
    pub z_rand: f64,
    pub sigma_hit: f64,
    pub lambda_short: f64,
    pub max_range: f64,
}

impl BeamModel {
    pub fn new(max_range: f64) -> Self {
        Self {
            z_hit: 0.7,
            z_short: 0.1,
            z_max: 0.1,
            z_rand: 0.1,
            sigma_hit: 0.2,
            lambda_short: 1.0,
            max_range,
        }
    }

    pub fn with_weights(mut self, hit: f64, short: f64, max: f64, rand: f64) -> Self {
        let total = hit + short + max + rand;
        if total > 0.0 {
            self.z_hit = hit / total;
            self.z_short = short / total;
            self.z_max = max / total;
            self.z_rand = rand / total;
        }
        self
    }

    pub fn with_sigma_hit(mut self, sigma: f64) -> Self {
        self.sigma_hit = sigma.max(0.001);
        self
    }

    pub fn with_lambda_short(mut self, lambda: f64) -> Self {
        self.lambda_short = lambda.max(0.001);
        self
    }

    /// Probability of measurement z given expected distance z_expected.
    pub fn probability(&self, z: f64, z_expected: f64) -> f64 {
        if z < 0.0 || z > self.max_range {
            return 0.0;
        }

        // Hit: Gaussian centered at expected range
        let p_hit = if z <= self.max_range {
            let diff = z - z_expected;
            let p = (-0.5 * diff * diff / (self.sigma_hit * self.sigma_hit)).exp();
            let norm = self.sigma_hit * (2.0 * std::f64::consts::PI).sqrt();
            p / norm
        } else {
            0.0
        };

        // Short: exponential decay for unexpectedly short readings
        let p_short = if z <= z_expected && z_expected > 0.0 {
            let eta = 1.0 / (1.0 - (-self.lambda_short * z_expected).exp());
            eta * self.lambda_short * (-self.lambda_short * z).exp()
        } else {
            0.0
        };

        // Max: point mass at max_range
        let p_max = if (z - self.max_range).abs() < 0.01 { 1.0 } else { 0.0 };

        // Random: uniform distribution
        let p_rand = 1.0 / self.max_range;

        self.z_hit * p_hit + self.z_short * p_short + self.z_max * p_max + self.z_rand * p_rand
    }

    /// Log-probability of a measurement.
    pub fn log_probability(&self, z: f64, z_expected: f64) -> f64 {
        let p = self.probability(z, z_expected);
        if p > 0.0 { p.ln() } else { -100.0 }
    }

    /// Compute total log-likelihood for a scan of measurements.
    pub fn scan_likelihood(&self, measurements: &[f64], expected: &[f64]) -> f64 {
        measurements.iter().zip(expected.iter())
            .map(|(z, ze)| self.log_probability(*z, *ze))
            .sum()
    }
}

impl fmt::Display for BeamModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BeamModel(hit={:.2}, short={:.2}, max={:.2}, rand={:.2}, range={:.1})",
            self.z_hit, self.z_short, self.z_max, self.z_rand, self.max_range)
    }
}

// ── Likelihood Field Model ──

/// Likelihood field model for range sensors — faster than beam model by
/// precomputing distance fields on the map.
#[derive(Clone, Debug)]
pub struct LikelihoodField {
    distance_field: Vec<f64>,
    width: usize,
    height: usize,
    resolution: f64,
    sigma_hit: f64,
    z_hit: f64,
    z_rand: f64,
    max_range: f64,
}

impl LikelihoodField {
    pub fn new(occupancy: &[bool], width: usize, height: usize, resolution: f64) -> Self {
        let distance_field = Self::compute_distance_field(occupancy, width, height, resolution);
        Self {
            distance_field,
            width,
            height,
            resolution,
            sigma_hit: 0.2,
            z_hit: 0.9,
            z_rand: 0.1,
            max_range: 30.0,
        }
    }

    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma_hit = sigma.max(0.001);
        self
    }

    pub fn with_max_range(mut self, range: f64) -> Self {
        self.max_range = range.max(0.1);
        self
    }

    /// Compute distance transform using brute-force (for simplicity).
    fn compute_distance_field(occupancy: &[bool], width: usize, height: usize, resolution: f64) -> Vec<f64> {
        let n = width * height;
        let mut field = vec![f64::MAX; n];

        // Collect occupied cells
        let mut occupied = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if occupancy[y * width + x] {
                    occupied.push((x, y));
                    field[y * width + x] = 0.0;
                }
            }
        }

        // BFS-like distance propagation (approximate)
        for y in 0..height {
            for x in 0..width {
                let mut min_dist = field[y * width + x];
                for &(ox, oy) in &occupied {
                    let dx = (x as f64 - ox as f64) * resolution;
                    let dy = (y as f64 - oy as f64) * resolution;
                    let d = (dx * dx + dy * dy).sqrt();
                    if d < min_dist {
                        min_dist = d;
                    }
                }
                field[y * width + x] = min_dist;
            }
        }
        field
    }

    /// Get distance to nearest obstacle at a world position.
    pub fn distance_at(&self, x: f64, y: f64) -> f64 {
        let gx = (x / self.resolution).floor() as isize;
        let gy = (y / self.resolution).floor() as isize;
        if gx < 0 || gy < 0 || gx >= self.width as isize || gy >= self.height as isize {
            return self.max_range;
        }
        self.distance_field[gy as usize * self.width + gx as usize]
    }

    /// Probability of a single endpoint given the distance field.
    pub fn endpoint_probability(&self, x: f64, y: f64) -> f64 {
        let dist = self.distance_at(x, y);
        let p_hit = (-0.5 * dist * dist / (self.sigma_hit * self.sigma_hit)).exp()
            / (self.sigma_hit * (2.0 * std::f64::consts::PI).sqrt());
        let p_rand = 1.0 / self.max_range;
        self.z_hit * p_hit + self.z_rand * p_rand
    }

    pub fn grid_size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

impl fmt::Display for LikelihoodField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LikelihoodField({}x{}, sigma={:.3})", self.width, self.height, self.sigma_hit)
    }
}

// ── Ray Casting ──

/// Simple ray caster for 2D occupancy grids using DDA algorithm.
#[derive(Clone, Debug)]
pub struct RayCaster {
    occupancy: Vec<bool>,
    width: usize,
    height: usize,
    resolution: f64,
    max_range: f64,
}

impl RayCaster {
    pub fn new(occupancy: Vec<bool>, width: usize, height: usize, resolution: f64) -> Self {
        Self { occupancy, width, height, resolution, max_range: 30.0 }
    }

    pub fn with_max_range(mut self, range: f64) -> Self {
        self.max_range = range;
        self
    }

    fn is_occupied(&self, gx: isize, gy: isize) -> bool {
        if gx < 0 || gy < 0 || gx >= self.width as isize || gy >= self.height as isize {
            return true; // out of bounds treated as occupied
        }
        self.occupancy[gy as usize * self.width + gx as usize]
    }

    /// Cast a single ray and return the range measurement.
    pub fn cast_ray(&self, ox: f64, oy: f64, angle: f64) -> f64 {
        let dx = angle.cos();
        let dy = angle.sin();
        let step = self.resolution * 0.5;
        let max_steps = (self.max_range / step) as usize;

        for i in 1..=max_steps {
            let dist = step * i as f64;
            let px = ox + dx * dist;
            let py = oy + dy * dist;
            let gx = (px / self.resolution).floor() as isize;
            let gy = (py / self.resolution).floor() as isize;
            if self.is_occupied(gx, gy) {
                return dist;
            }
        }
        self.max_range
    }

    /// Cast multiple rays in a fan pattern.
    pub fn cast_scan(&self, ox: f64, oy: f64, start_angle: f64, end_angle: f64, num_beams: usize) -> Vec<f64> {
        let mut ranges = Vec::with_capacity(num_beams);
        let angle_step = if num_beams > 1 {
            (end_angle - start_angle) / (num_beams - 1) as f64
        } else {
            0.0
        };
        for i in 0..num_beams {
            let angle = start_angle + angle_step * i as f64;
            ranges.push(self.cast_ray(ox, oy, angle));
        }
        ranges
    }
}

impl fmt::Display for RayCaster {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RayCaster({}x{}, res={:.3}, max={:.1})",
            self.width, self.height, self.resolution, self.max_range)
    }
}

// ── Sensor Configuration ──

/// Configuration for a generic range sensor.
#[derive(Clone, Debug)]
pub struct SensorConfig {
    pub max_range: f64,
    pub min_range: f64,
    pub fov: f64,
    pub num_beams: usize,
    pub noise_std: f64,
    pub update_rate_hz: f64,
}

impl SensorConfig {
    pub fn lidar() -> Self {
        Self {
            max_range: 30.0,
            min_range: 0.1,
            fov: 2.0 * std::f64::consts::PI,
            num_beams: 360,
            noise_std: 0.02,
            update_rate_hz: 10.0,
        }
    }

    pub fn sonar() -> Self {
        Self {
            max_range: 5.0,
            min_range: 0.2,
            fov: 0.5,
            num_beams: 1,
            noise_std: 0.05,
            update_rate_hz: 20.0,
        }
    }

    pub fn with_max_range(mut self, r: f64) -> Self {
        self.max_range = r;
        self
    }

    pub fn with_noise(mut self, std: f64) -> Self {
        self.noise_std = std;
        self
    }

    pub fn with_num_beams(mut self, n: usize) -> Self {
        self.num_beams = n.max(1);
        self
    }

    pub fn angular_resolution(&self) -> f64 {
        if self.num_beams > 1 {
            self.fov / (self.num_beams - 1) as f64
        } else {
            self.fov
        }
    }
}

impl fmt::Display for SensorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SensorConfig(range=[{:.1},{:.1}], fov={:.2}rad, beams={}, noise={:.3})",
            self.min_range, self.max_range, self.fov, self.num_beams, self.noise_std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gaussian_noise_sample() {
        let mut noise = GaussianNoise::new(1.0).with_seed(42);
        let s = noise.sample();
        assert!(s.is_finite());
    }

    #[test]
    fn test_gaussian_noise_pdf() {
        let noise = GaussianNoise::new(1.0);
        let p = noise.pdf(0.0);
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((p - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_noise_pdf_peak() {
        let noise = GaussianNoise::new(0.5).with_mean(2.0);
        let p_at_mean = noise.pdf(2.0);
        let p_away = noise.pdf(5.0);
        assert!(p_at_mean > p_away);
    }

    #[test]
    fn test_gaussian_log_likelihood() {
        let noise = GaussianNoise::new(1.0);
        let ll = noise.log_likelihood(0.0);
        assert!(ll < 0.0);
        assert!(ll > -5.0);
    }

    #[test]
    fn test_beam_model_creation() {
        let bm = BeamModel::new(30.0);
        let total = bm.z_hit + bm.z_short + bm.z_max + bm.z_rand;
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_beam_model_hit() {
        let bm = BeamModel::new(10.0).with_sigma_hit(0.1);
        let p_close = bm.probability(5.0, 5.0);
        let p_far = bm.probability(5.0, 8.0);
        assert!(p_close > p_far);
    }

    #[test]
    fn test_beam_model_out_of_range() {
        let bm = BeamModel::new(10.0);
        assert_eq!(bm.probability(-1.0, 5.0), 0.0);
        assert_eq!(bm.probability(11.0, 5.0), 0.0);
    }

    #[test]
    fn test_beam_model_scan_likelihood() {
        let bm = BeamModel::new(10.0);
        let measurements = vec![5.0, 5.1, 4.9];
        let expected = vec![5.0, 5.0, 5.0];
        let ll = bm.scan_likelihood(&measurements, &expected);
        assert!(ll.is_finite());
    }

    #[test]
    fn test_beam_model_display() {
        let bm = BeamModel::new(10.0);
        let s = format!("{bm}");
        assert!(s.contains("BeamModel"));
    }

    #[test]
    fn test_likelihood_field_creation() {
        let mut occ = vec![false; 100];
        occ[55] = true; // one obstacle
        let lf = LikelihoodField::new(&occ, 10, 10, 0.1);
        assert_eq!(lf.grid_size(), (10, 10));
    }

    #[test]
    fn test_likelihood_field_distance() {
        let mut occ = vec![false; 25];
        occ[12] = true; // center of 5x5 grid
        let lf = LikelihoodField::new(&occ, 5, 5, 1.0);
        let d = lf.distance_at(2.5, 2.5);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_likelihood_field_probability() {
        let mut occ = vec![false; 100];
        occ[55] = true;
        let lf = LikelihoodField::new(&occ, 10, 10, 0.1).with_sigma(0.2);
        let p = lf.endpoint_probability(0.5, 0.5);
        assert!(p > 0.0);
    }

    #[test]
    fn test_ray_caster_empty() {
        let occ = vec![false; 100];
        let rc = RayCaster::new(occ, 10, 10, 1.0).with_max_range(5.0);
        let range = rc.cast_ray(5.0, 5.0, 0.0);
        assert!((range - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_ray_caster_hit() {
        let mut occ = vec![false; 100];
        occ[5 * 10 + 8] = true; // obstacle at (8,5)
        let rc = RayCaster::new(occ, 10, 10, 1.0).with_max_range(20.0);
        let range = rc.cast_ray(2.0, 5.5, 0.0); // shooting east
        assert!(range < 7.0);
    }

    #[test]
    fn test_ray_caster_scan() {
        let occ = vec![false; 400];
        let rc = RayCaster::new(occ, 20, 20, 0.5).with_max_range(5.0);
        let scan = rc.cast_scan(5.0, 5.0, -PI / 4.0, PI / 4.0, 5);
        assert_eq!(scan.len(), 5);
    }

    #[test]
    fn test_sensor_config_lidar() {
        let cfg = SensorConfig::lidar();
        assert_eq!(cfg.num_beams, 360);
        assert!((cfg.fov - 2.0 * PI).abs() < 1e-6);
    }

    #[test]
    fn test_sensor_config_sonar() {
        let cfg = SensorConfig::sonar();
        assert_eq!(cfg.num_beams, 1);
    }

    #[test]
    fn test_sensor_config_angular_resolution() {
        let cfg = SensorConfig::lidar();
        let res = cfg.angular_resolution();
        assert!(res > 0.0);
        assert!(res < 0.02);
    }

    #[test]
    fn test_sensor_config_builder() {
        let cfg = SensorConfig::lidar()
            .with_max_range(50.0)
            .with_noise(0.01)
            .with_num_beams(720);
        assert!((cfg.max_range - 50.0).abs() < 1e-10);
        assert_eq!(cfg.num_beams, 720);
    }
}
