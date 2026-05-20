//! Procedural galaxy generation — spiral, elliptical, irregular galaxies.
//!
//! Replaces three.js galaxy generators / GalSim with pure Rust.
//! Spiral arms (logarithmic), elliptical (smooth), star density falloff,
//! bulge, dust lanes, Salpeter IMF, globular clusters, rotation curves.

use std::f64::consts::PI;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Domain errors for galaxy generation.
#[derive(Debug, Clone, PartialEq)]
pub enum GalaxyError {
    /// Star count must be positive.
    InvalidStarCount(usize),
    /// Arm count must be at least 1 for spiral.
    InvalidArmCount(usize),
    /// Galaxy radius must be positive.
    NonPositiveRadius(f64),
    /// Seed must be non-zero.
    ZeroSeed,
    /// Mass must be positive.
    NonPositiveMass(f64),
}

impl fmt::Display for GalaxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStarCount(n) => write!(f, "star count must be positive, got {n}"),
            Self::InvalidArmCount(n) => write!(f, "arm count must be >= 1, got {n}"),
            Self::NonPositiveRadius(r) => write!(f, "radius must be positive, got {r}"),
            Self::ZeroSeed => write!(f, "seed must be non-zero"),
            Self::NonPositiveMass(m) => write!(f, "mass must be positive, got {m}"),
        }
    }
}

impl std::error::Error for GalaxyError {}

// ── PRNG ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform float in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.uniform() * (hi - lo)
    }

    /// Approximate Gaussian (Box-Muller).
    fn gaussian(&mut self, mean: f64, stddev: f64) -> f64 {
        let u1 = self.uniform().max(1e-30);
        let u2 = self.uniform();
        mean + stddev * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Power-law distributed value in [x_min, x_max] with exponent alpha.
    /// Salpeter IMF: alpha = -2.35 (for dN/dM ~ M^alpha).
    fn power_law(&mut self, x_min: f64, x_max: f64, alpha: f64) -> f64 {
        let u = self.uniform();
        let exp = alpha + 1.0;
        if exp.abs() < 1e-12 {
            // Special case: alpha = -1 => log distribution.
            return x_min * (x_max / x_min).powf(u);
        }
        let a = x_min.powf(exp);
        let b = x_max.powf(exp);
        (a + u * (b - a)).powf(1.0 / exp)
    }
}

// ── Galaxy Star ─────────────────────────────────────────────────

/// A generated star in the galaxy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalaxyStar {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Mass in solar masses.
    pub mass: f64,
    /// RGB color (r, g, b) in [0, 1].
    pub color_r: f64,
    pub color_g: f64,
    pub color_b: f64,
    /// Whether this star belongs to a globular cluster.
    pub in_cluster: bool,
}

// ── Galaxy Type ─────────────────────────────────────────────────

/// Type of galaxy morphology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GalaxyType {
    Spiral,
    Elliptical,
    Irregular,
}

// ── Galaxy Config ───────────────────────────────────────────────

/// Configuration for galaxy generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GalaxyConfig {
    pub galaxy_type: GalaxyType,
    pub star_count: usize,
    pub radius: f64,
    pub seed: u64,
    /// Number of spiral arms (spiral only).
    pub arm_count: usize,
    /// Winding tightness (radians per unit radius). Higher = tighter wind.
    pub winding_factor: f64,
    /// Arm spread (dispersion perpendicular to arm).
    pub arm_spread: f64,
    /// Bulge fraction (0.0 to 1.0) of stars in central bulge.
    pub bulge_fraction: f64,
    /// Bulge radius as fraction of galaxy radius.
    pub bulge_radius_frac: f64,
    /// Disk thickness as fraction of galaxy radius.
    pub disk_thickness: f64,
    /// Number of globular clusters.
    pub globular_clusters: usize,
    /// Stars per globular cluster.
    pub cluster_star_count: usize,
    /// Salpeter IMF: minimum mass (solar masses).
    pub min_mass: f64,
    /// Salpeter IMF: maximum mass (solar masses).
    pub max_mass: f64,
}

impl Default for GalaxyConfig {
    fn default() -> Self {
        Self {
            galaxy_type: GalaxyType::Spiral,
            star_count: 1000,
            radius: 50.0,
            seed: 42,
            arm_count: 2,
            winding_factor: 0.5,
            arm_spread: 0.15,
            bulge_fraction: 0.2,
            bulge_radius_frac: 0.15,
            disk_thickness: 0.02,
            globular_clusters: 5,
            cluster_star_count: 50,
            min_mass: 0.1,
            max_mass: 100.0,
        }
    }
}

// ── Galaxy ──────────────────────────────────────────────────────

/// A generated galaxy.
#[derive(Debug, Clone)]
pub struct Galaxy {
    pub stars: Vec<GalaxyStar>,
    pub config: GalaxyConfig,
}

impl Galaxy {
    /// Generate a galaxy from a configuration.
    pub fn generate(config: GalaxyConfig) -> Result<Self, GalaxyError> {
        if config.star_count == 0 {
            return Err(GalaxyError::InvalidStarCount(0));
        }
        if config.radius <= 0.0 {
            return Err(GalaxyError::NonPositiveRadius(config.radius));
        }
        if config.seed == 0 {
            return Err(GalaxyError::ZeroSeed);
        }
        if config.galaxy_type == GalaxyType::Spiral && config.arm_count == 0 {
            return Err(GalaxyError::InvalidArmCount(0));
        }

        let mut rng = Rng::new(config.seed);
        let mut stars = Vec::with_capacity(config.star_count + config.globular_clusters * config.cluster_star_count);

        let bulge_count = (config.star_count as f64 * config.bulge_fraction) as usize;
        let disk_count = config.star_count - bulge_count;

        // Generate bulge stars.
        let bulge_r = config.radius * config.bulge_radius_frac;
        for _ in 0..bulge_count {
            let r = rng.gaussian(0.0, bulge_r * 0.4).abs();
            let theta = rng.uniform_range(0.0, 2.0 * PI);
            let phi = rng.uniform_range(-PI, PI);
            let x = r * theta.cos() * phi.cos();
            let y = r * theta.sin() * phi.cos();
            let z = r * phi.sin() * 0.6; // slightly oblate
            let mass = rng.power_law(config.min_mass, config.max_mass, -2.35);
            let (cr, cg, cb) = mass_to_color(mass);
            stars.push(GalaxyStar { x, y, z, mass, color_r: cr, color_g: cg, color_b: cb, in_cluster: false });
        }

        // Generate disk stars.
        match config.galaxy_type {
            GalaxyType::Spiral => {
                for _ in 0..disk_count {
                    let arm = (rng.uniform() * config.arm_count as f64) as usize;
                    let arm_angle_offset = 2.0 * PI * arm as f64 / config.arm_count as f64;
                    // Exponential radial distribution.
                    let t = rng.uniform();
                    let r = -config.radius * 0.4 * (1.0 - t).max(1e-30).ln();
                    let r = r.min(config.radius);
                    let spiral_angle = arm_angle_offset + config.winding_factor * r.ln().max(0.0);
                    let spread = rng.gaussian(0.0, config.arm_spread * r.max(0.1));
                    let theta = spiral_angle + spread / r.max(0.1);
                    let x = r * theta.cos();
                    let y = r * theta.sin();
                    let z = rng.gaussian(0.0, config.radius * config.disk_thickness);
                    let mass = rng.power_law(config.min_mass, config.max_mass, -2.35);
                    let (cr, cg, cb) = mass_to_color(mass);
                    stars.push(GalaxyStar { x, y, z, mass, color_r: cr, color_g: cg, color_b: cb, in_cluster: false });
                }
            }
            GalaxyType::Elliptical => {
                for _ in 0..disk_count {
                    let r = rng.gaussian(0.0, config.radius * 0.3).abs();
                    let theta = rng.uniform_range(0.0, 2.0 * PI);
                    let phi = rng.uniform_range(-PI / 2.0, PI / 2.0);
                    let x = r * theta.cos() * phi.cos();
                    let y = r * theta.sin() * phi.cos() * 0.7;
                    let z = r * phi.sin() * 0.5;
                    let mass = rng.power_law(config.min_mass, config.max_mass, -2.35);
                    let (cr, cg, cb) = mass_to_color(mass);
                    stars.push(GalaxyStar { x, y, z, mass, color_r: cr, color_g: cg, color_b: cb, in_cluster: false });
                }
            }
            GalaxyType::Irregular => {
                for _ in 0..disk_count {
                    // Multiple clumps at random positions.
                    let clump_x = rng.gaussian(0.0, config.radius * 0.3);
                    let clump_y = rng.gaussian(0.0, config.radius * 0.3);
                    let x = clump_x + rng.gaussian(0.0, config.radius * 0.1);
                    let y = clump_y + rng.gaussian(0.0, config.radius * 0.1);
                    let z = rng.gaussian(0.0, config.radius * 0.05);
                    let mass = rng.power_law(config.min_mass, config.max_mass, -2.35);
                    let (cr, cg, cb) = mass_to_color(mass);
                    stars.push(GalaxyStar { x, y, z, mass, color_r: cr, color_g: cg, color_b: cb, in_cluster: false });
                }
            }
        }

        // Globular clusters.
        for _ in 0..config.globular_clusters {
            let cr = rng.uniform_range(config.radius * 0.3, config.radius * 1.2);
            let ctheta = rng.uniform_range(0.0, 2.0 * PI);
            let cphi = rng.gaussian(0.0, 0.5);
            let cx = cr * ctheta.cos() * cphi.cos();
            let cy = cr * ctheta.sin() * cphi.cos();
            let cz = cr * cphi.sin();
            let cluster_radius = config.radius * 0.02;
            for _ in 0..config.cluster_star_count {
                let dx = rng.gaussian(0.0, cluster_radius);
                let dy = rng.gaussian(0.0, cluster_radius);
                let dz = rng.gaussian(0.0, cluster_radius);
                let mass = rng.power_law(config.min_mass, 2.0, -2.35); // clusters are old, low mass
                let (cr2, cg2, cb2) = mass_to_color(mass);
                stars.push(GalaxyStar {
                    x: cx + dx, y: cy + dy, z: cz + dz,
                    mass, color_r: cr2, color_g: cg2, color_b: cb2,
                    in_cluster: true,
                });
            }
        }

        Ok(Galaxy { stars, config })
    }

    /// Total number of stars.
    pub fn star_count(&self) -> usize {
        self.stars.len()
    }

    /// Total mass (in solar masses).
    pub fn total_mass(&self) -> f64 {
        self.stars.iter().map(|s| s.mass).sum()
    }

    /// Number of stars in globular clusters.
    pub fn cluster_star_count(&self) -> usize {
        self.stars.iter().filter(|s| s.in_cluster).count()
    }

    /// Average color of the galaxy.
    pub fn average_color(&self) -> (f64, f64, f64) {
        if self.stars.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let n = self.stars.len() as f64;
        let r: f64 = self.stars.iter().map(|s| s.color_r).sum::<f64>() / n;
        let g: f64 = self.stars.iter().map(|s| s.color_g).sum::<f64>() / n;
        let b: f64 = self.stars.iter().map(|s| s.color_b).sum::<f64>() / n;
        (r, g, b)
    }

    /// Check if star is in a dust lane (close to spiral arm plane, high density).
    pub fn is_in_dust_lane(&self, star_idx: usize) -> bool {
        if star_idx >= self.stars.len() {
            return false;
        }
        let s = &self.stars[star_idx];
        if self.config.galaxy_type != GalaxyType::Spiral {
            return false;
        }
        // Dust lanes are close to the disk plane and within the arm region.
        let r = (s.x * s.x + s.y * s.y).sqrt();
        let z_threshold = self.config.radius * self.config.disk_thickness * 0.5;
        s.z.abs() < z_threshold && r < self.config.radius * 0.8 && r > self.config.radius * 0.1
    }

    /// Rotation curve: circular velocity at radius r.
    /// Uses a simplified model: v(r) = v_max * r / sqrt(r^2 + r_c^2).
    pub fn rotation_velocity(&self, r: f64) -> f64 {
        let v_max = 220.0; // km/s (Milky Way-like)
        let r_c = self.config.radius * 0.1; // core radius
        v_max * r / (r * r + r_c * r_c).sqrt()
    }

    /// Mass distribution function: stars per radial bin.
    pub fn radial_density(&self, n_bins: usize) -> Vec<(f64, usize)> {
        if n_bins == 0 {
            return Vec::new();
        }
        let max_r = self.config.radius * 1.5;
        let dr = max_r / n_bins as f64;
        let mut bins = vec![0usize; n_bins];
        for s in &self.stars {
            let r = (s.x * s.x + s.y * s.y).sqrt();
            let idx = ((r / dr) as usize).min(n_bins - 1);
            bins[idx] += 1;
        }
        bins.into_iter()
            .enumerate()
            .map(|(i, count)| (dr * (i as f64 + 0.5), count))
            .collect()
    }
}

// ── Color from Mass ─────────────────────────────────────────────

/// Approximate star color from mass (solar masses).
fn mass_to_color(mass: f64) -> (f64, f64, f64) {
    // Approximate temperature from mass: T ~ 5778 * M^0.57
    let temp = 5778.0 * mass.powf(0.57);
    let t = temp / 100.0;
    let r = if t <= 66.0 { 1.0 } else {
        (329.698727446 * (t - 60.0).powf(-0.1332047592) / 255.0).clamp(0.0, 1.0)
    };
    let g = if t <= 66.0 {
        ((99.4708025861 * t.ln() - 161.1195681661) / 255.0).clamp(0.0, 1.0)
    } else {
        (288.1221695283 * (t - 60.0).powf(-0.0755148492) / 255.0).clamp(0.0, 1.0)
    };
    let b = if t >= 66.0 { 1.0 } else if t <= 19.0 { 0.0 } else {
        ((138.5177312231 * (t - 10.0).ln() - 305.0447927307) / 255.0).clamp(0.0, 1.0)
    };
    (r, g, b)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn spiral_generation() {
        let config = GalaxyConfig::default();
        let galaxy = Galaxy::generate(config).unwrap();
        assert!(galaxy.star_count() >= 1000); // disk + bulge + clusters
    }

    #[test]
    fn elliptical_generation() {
        let config = GalaxyConfig {
            galaxy_type: GalaxyType::Elliptical,
            star_count: 500,
            seed: 123,
            ..Default::default()
        };
        let galaxy = Galaxy::generate(config).unwrap();
        assert!(galaxy.star_count() >= 500);
    }

    #[test]
    fn irregular_generation() {
        let config = GalaxyConfig {
            galaxy_type: GalaxyType::Irregular,
            star_count: 300,
            seed: 99,
            ..Default::default()
        };
        let galaxy = Galaxy::generate(config).unwrap();
        assert!(galaxy.star_count() >= 300);
    }

    #[test]
    fn zero_star_count_error() {
        let config = GalaxyConfig { star_count: 0, ..Default::default() };
        assert!(Galaxy::generate(config).is_err());
    }

    #[test]
    fn zero_seed_error() {
        let config = GalaxyConfig { seed: 0, ..Default::default() };
        assert!(Galaxy::generate(config).is_err());
    }

    #[test]
    fn negative_radius_error() {
        let config = GalaxyConfig { radius: -1.0, ..Default::default() };
        assert!(Galaxy::generate(config).is_err());
    }

    #[test]
    fn zero_arms_spiral_error() {
        let config = GalaxyConfig {
            galaxy_type: GalaxyType::Spiral,
            arm_count: 0,
            ..Default::default()
        };
        assert!(Galaxy::generate(config).is_err());
    }

    #[test]
    fn deterministic_generation() {
        let config = GalaxyConfig { star_count: 100, seed: 42, ..Default::default() };
        let g1 = Galaxy::generate(config.clone()).unwrap();
        let g2 = Galaxy::generate(config).unwrap();
        assert_eq!(g1.star_count(), g2.star_count());
        assert!(approx_eq(g1.total_mass(), g2.total_mass(), 1e-10));
    }

    #[test]
    fn total_mass_positive() {
        let config = GalaxyConfig::default();
        let galaxy = Galaxy::generate(config).unwrap();
        assert!(galaxy.total_mass() > 0.0);
    }

    #[test]
    fn globular_clusters_present() {
        let config = GalaxyConfig {
            globular_clusters: 3,
            cluster_star_count: 20,
            ..Default::default()
        };
        let galaxy = Galaxy::generate(config).unwrap();
        assert_eq!(galaxy.cluster_star_count(), 60);
    }

    #[test]
    fn no_globular_clusters() {
        let config = GalaxyConfig { globular_clusters: 0, ..Default::default() };
        let galaxy = Galaxy::generate(config).unwrap();
        assert_eq!(galaxy.cluster_star_count(), 0);
    }

    #[test]
    fn average_color_valid() {
        let config = GalaxyConfig { star_count: 500, ..Default::default() };
        let galaxy = Galaxy::generate(config).unwrap();
        let (r, g, b) = galaxy.average_color();
        assert!(r >= 0.0 && r <= 1.0);
        assert!(g >= 0.0 && g <= 1.0);
        assert!(b >= 0.0 && b <= 1.0);
    }

    #[test]
    fn rotation_curve_rises_then_flattens() {
        let config = GalaxyConfig::default();
        let galaxy = Galaxy::generate(config).unwrap();
        let v1 = galaxy.rotation_velocity(1.0);
        let v10 = galaxy.rotation_velocity(10.0);
        let v50 = galaxy.rotation_velocity(50.0);
        let v100 = galaxy.rotation_velocity(100.0);
        assert!(v10 > v1);
        // Should flatten at large r.
        let ratio = (v100 - v50).abs() / v50;
        assert!(ratio < 0.5);
    }

    #[test]
    fn radial_density_profile() {
        let config = GalaxyConfig { star_count: 1000, ..Default::default() };
        let galaxy = Galaxy::generate(config).unwrap();
        let density = galaxy.radial_density(10);
        assert_eq!(density.len(), 10);
        let total: usize = density.iter().map(|(_, c)| *c).sum();
        assert_eq!(total, galaxy.star_count());
    }

    #[test]
    fn dust_lane_detection() {
        let config = GalaxyConfig { star_count: 500, ..Default::default() };
        let galaxy = Galaxy::generate(config).unwrap();
        let mut dust_count = 0;
        for i in 0..galaxy.star_count() {
            if galaxy.is_in_dust_lane(i) {
                dust_count += 1;
            }
        }
        // Some stars should be in dust lanes for a spiral.
        assert!(dust_count > 0);
    }

    #[test]
    fn mass_distribution_salpeter() {
        let config = GalaxyConfig { star_count: 5000, seed: 77, ..Default::default() };
        let galaxy = Galaxy::generate(config).unwrap();
        let low_mass: usize = galaxy.stars.iter().filter(|s| s.mass < 1.0).count();
        let high_mass: usize = galaxy.stars.iter().filter(|s| s.mass > 10.0).count();
        // Salpeter IMF: many more low-mass than high-mass stars.
        assert!(low_mass > high_mass * 5);
    }

    #[test]
    fn star_positions_finite() {
        let config = GalaxyConfig { star_count: 500, ..Default::default() };
        let galaxy = Galaxy::generate(config).unwrap();
        for s in &galaxy.stars {
            assert!(s.x.is_finite());
            assert!(s.y.is_finite());
            assert!(s.z.is_finite());
            assert!(s.mass > 0.0);
        }
    }

    #[test]
    fn mass_to_color_hot_star() {
        let (r, g, b) = mass_to_color(50.0); // Very massive => very hot => blue-white.
        assert!(b > 0.5);
        assert!(r > 0.5);
    }

    #[test]
    fn mass_to_color_cool_star() {
        let (r, _g, b) = mass_to_color(0.2); // Low mass => cool => reddish.
        assert!(r > b);
    }

    #[test]
    fn bulge_stars_near_center() {
        let config = GalaxyConfig {
            star_count: 1000,
            bulge_fraction: 1.0, // All bulge
            ..Default::default()
        };
        let galaxy = Galaxy::generate(config).unwrap();
        let near_center = galaxy.stars.iter().filter(|s| {
            let r = (s.x * s.x + s.y * s.y + s.z * s.z).sqrt();
            r < galaxy.config.radius * 0.5
        }).count();
        // Most bulge stars should be near center.
        assert!(near_center > galaxy.stars.len() / 2);
    }
}
