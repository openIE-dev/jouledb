// Procedural Terrain Generation — Biome assignment from environmental parameters
// Whittaker diagram, temperature/moisture models, biome transitions, biome properties

use std::fmt;

/// Biome types based on the Whittaker diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BiomeType {
    Tundra,
    Taiga,
    TemperateForest,
    TemperateRainforest,
    TropicalForest,
    TropicalRainforest,
    Desert,
    Grassland,
    Savanna,
    Shrubland,
    BorealForest,
    Wetland,
    Alpine,
    Ocean,
    Beach,
}

impl fmt::Display for BiomeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tundra => write!(f, "Tundra"),
            Self::Taiga => write!(f, "Taiga"),
            Self::TemperateForest => write!(f, "Temperate Forest"),
            Self::TemperateRainforest => write!(f, "Temperate Rainforest"),
            Self::TropicalForest => write!(f, "Tropical Forest"),
            Self::TropicalRainforest => write!(f, "Tropical Rainforest"),
            Self::Desert => write!(f, "Desert"),
            Self::Grassland => write!(f, "Grassland"),
            Self::Savanna => write!(f, "Savanna"),
            Self::Shrubland => write!(f, "Shrubland"),
            Self::BorealForest => write!(f, "Boreal Forest"),
            Self::Wetland => write!(f, "Wetland"),
            Self::Alpine => write!(f, "Alpine"),
            Self::Ocean => write!(f, "Ocean"),
            Self::Beach => write!(f, "Beach"),
        }
    }
}

/// Properties associated with a biome.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeProperties {
    pub biome: BiomeType,
    pub tree_density: f64,
    pub ground_color: [f64; 3],
    pub height_variation: f64,
    pub base_temperature: f64,
    pub base_moisture: f64,
}

impl BiomeProperties {
    pub fn for_biome(biome: BiomeType) -> Self {
        match biome {
            BiomeType::Tundra => Self {
                biome, tree_density: 0.02, ground_color: [0.85, 0.88, 0.90],
                height_variation: 0.1, base_temperature: -10.0, base_moisture: 0.3,
            },
            BiomeType::Taiga => Self {
                biome, tree_density: 0.6, ground_color: [0.2, 0.35, 0.2],
                height_variation: 0.2, base_temperature: -5.0, base_moisture: 0.5,
            },
            BiomeType::TemperateForest => Self {
                biome, tree_density: 0.7, ground_color: [0.25, 0.45, 0.15],
                height_variation: 0.3, base_temperature: 10.0, base_moisture: 0.6,
            },
            BiomeType::TemperateRainforest => Self {
                biome, tree_density: 0.85, ground_color: [0.15, 0.5, 0.1],
                height_variation: 0.35, base_temperature: 12.0, base_moisture: 0.9,
            },
            BiomeType::TropicalForest => Self {
                biome, tree_density: 0.8, ground_color: [0.1, 0.55, 0.1],
                height_variation: 0.25, base_temperature: 25.0, base_moisture: 0.7,
            },
            BiomeType::TropicalRainforest => Self {
                biome, tree_density: 0.95, ground_color: [0.05, 0.45, 0.05],
                height_variation: 0.2, base_temperature: 27.0, base_moisture: 0.95,
            },
            BiomeType::Desert => Self {
                biome, tree_density: 0.01, ground_color: [0.82, 0.75, 0.55],
                height_variation: 0.15, base_temperature: 30.0, base_moisture: 0.05,
            },
            BiomeType::Grassland => Self {
                biome, tree_density: 0.05, ground_color: [0.55, 0.65, 0.25],
                height_variation: 0.1, base_temperature: 15.0, base_moisture: 0.35,
            },
            BiomeType::Savanna => Self {
                biome, tree_density: 0.15, ground_color: [0.65, 0.6, 0.3],
                height_variation: 0.1, base_temperature: 25.0, base_moisture: 0.25,
            },
            BiomeType::Shrubland => Self {
                biome, tree_density: 0.1, ground_color: [0.5, 0.5, 0.3],
                height_variation: 0.15, base_temperature: 18.0, base_moisture: 0.2,
            },
            BiomeType::BorealForest => Self {
                biome, tree_density: 0.55, ground_color: [0.2, 0.3, 0.18],
                height_variation: 0.25, base_temperature: 0.0, base_moisture: 0.55,
            },
            BiomeType::Wetland => Self {
                biome, tree_density: 0.2, ground_color: [0.3, 0.4, 0.25],
                height_variation: 0.05, base_temperature: 15.0, base_moisture: 0.9,
            },
            BiomeType::Alpine => Self {
                biome, tree_density: 0.0, ground_color: [0.7, 0.7, 0.75],
                height_variation: 0.4, base_temperature: -15.0, base_moisture: 0.4,
            },
            BiomeType::Ocean => Self {
                biome, tree_density: 0.0, ground_color: [0.1, 0.2, 0.6],
                height_variation: 0.0, base_temperature: 10.0, base_moisture: 1.0,
            },
            BiomeType::Beach => Self {
                biome, tree_density: 0.03, ground_color: [0.9, 0.85, 0.65],
                height_variation: 0.02, base_temperature: 20.0, base_moisture: 0.5,
            },
        }
    }
}

/// Result of biome sampling at a point.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeSample {
    pub primary: BiomeType,
    pub secondary: Option<BiomeType>,
    pub blend_factor: f64,
    pub temperature: f64,
    pub moisture: f64,
    pub altitude: f64,
}

/// Configuration for biome generation.
#[derive(Debug, Clone, PartialEq)]
pub struct BiomeConfig {
    pub sea_level: f64,
    pub beach_threshold: f64,
    pub alpine_threshold: f64,
    pub temperature_latitude_factor: f64,
    pub temperature_altitude_factor: f64,
    pub moisture_noise_scale: f64,
    pub rain_shadow_strength: f64,
    pub transition_width: f64,
}

impl Default for BiomeConfig {
    fn default() -> Self {
        Self {
            sea_level: 0.3,
            beach_threshold: 0.02,
            alpine_threshold: 0.85,
            temperature_latitude_factor: 40.0,
            temperature_altitude_factor: 30.0,
            moisture_noise_scale: 1.0,
            rain_shadow_strength: 0.3,
            transition_width: 0.1,
        }
    }
}

/// Biome generator using Whittaker diagram approach.
pub struct BiomeGenerator {
    config: BiomeConfig,
    seed: u64,
}

impl fmt::Debug for BiomeGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BiomeGenerator")
            .field("seed", &self.seed)
            .field("config", &self.config)
            .finish()
    }
}

impl BiomeGenerator {
    pub fn new(seed: u64, config: BiomeConfig) -> Self {
        Self { config, seed }
    }

    pub fn config(&self) -> &BiomeConfig {
        &self.config
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Simple hash-based noise for moisture.
    fn noise_at(&self, x: f64, y: f64) -> f64 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = x - x.floor();
        let fy = y - y.floor();
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sy = fy * fy * (3.0 - 2.0 * fy);

        let hash = |px: i64, py: i64| -> f64 {
            let mut h = self.seed;
            h = h.wrapping_add(px as u64).wrapping_mul(6364136223846793005);
            h = h.wrapping_add(py as u64).wrapping_mul(6364136223846793005);
            h ^= h >> 33;
            (h & 0xFFFF) as f64 / 65535.0
        };

        let v00 = hash(ix, iy);
        let v10 = hash(ix + 1, iy);
        let v01 = hash(ix, iy + 1);
        let v11 = hash(ix + 1, iy + 1);

        let a = v00 + (v10 - v00) * sx;
        let b = v01 + (v11 - v01) * sx;
        a + (b - a) * sy
    }

    /// Compute temperature at a point given latitude (0..1) and altitude (0..1).
    pub fn temperature(&self, latitude: f64, altitude: f64) -> f64 {
        let base = 30.0 - latitude.abs() * self.config.temperature_latitude_factor;
        let alt_penalty = altitude.max(0.0) * self.config.temperature_altitude_factor;
        base - alt_penalty
    }

    /// Compute moisture at a point using noise and rain shadow effect.
    pub fn moisture(&self, x: f64, y: f64, altitude: f64) -> f64 {
        let base_moisture = self.noise_at(
            x * self.config.moisture_noise_scale,
            y * self.config.moisture_noise_scale,
        );
        // Rain shadow: higher altitude reduces moisture on lee side
        let shadow = (altitude - 0.5).max(0.0) * self.config.rain_shadow_strength;
        (base_moisture - shadow).clamp(0.0, 1.0)
    }

    /// Classify biome from temperature and moisture using Whittaker diagram.
    pub fn classify(&self, temperature: f64, moisture: f64) -> BiomeType {
        // Normalized temperature ranges
        if temperature < -10.0 {
            return BiomeType::Alpine;
        }
        if temperature < -2.0 {
            if moisture > 0.5 { return BiomeType::Tundra; }
            return BiomeType::Alpine;
        }
        if temperature < 5.0 {
            if moisture > 0.6 { return BiomeType::Taiga; }
            if moisture > 0.4 { return BiomeType::BorealForest; }
            return BiomeType::Tundra;
        }
        if temperature < 15.0 {
            if moisture > 0.8 { return BiomeType::TemperateRainforest; }
            if moisture > 0.5 { return BiomeType::TemperateForest; }
            if moisture > 0.3 { return BiomeType::Grassland; }
            return BiomeType::Shrubland;
        }
        if temperature < 22.0 {
            if moisture > 0.7 { return BiomeType::TemperateRainforest; }
            if moisture > 0.4 { return BiomeType::TemperateForest; }
            if moisture > 0.2 { return BiomeType::Grassland; }
            return BiomeType::Desert;
        }
        // Hot temperatures
        if moisture > 0.8 { return BiomeType::TropicalRainforest; }
        if moisture > 0.5 { return BiomeType::TropicalForest; }
        if moisture > 0.25 { return BiomeType::Savanna; }
        BiomeType::Desert
    }

    /// Sample biome at a world position.
    /// `latitude`: 0.0 (equator) to 1.0 (pole).
    /// `altitude`: 0.0 (sea level) to 1.0 (max height).
    pub fn sample(&self, x: f64, y: f64, latitude: f64, altitude: f64) -> BiomeSample {
        // Handle ocean / beach
        if altitude < self.config.sea_level {
            return BiomeSample {
                primary: BiomeType::Ocean,
                secondary: None,
                blend_factor: 0.0,
                temperature: self.temperature(latitude, 0.0),
                moisture: 1.0,
                altitude,
            };
        }

        if altitude < self.config.sea_level + self.config.beach_threshold {
            let temp = self.temperature(latitude, altitude);
            let moist = self.moisture(x, y, altitude);
            return BiomeSample {
                primary: BiomeType::Beach,
                secondary: Some(self.classify(temp, moist)),
                blend_factor: (altitude - self.config.sea_level) / self.config.beach_threshold,
                temperature: temp,
                moisture: moist,
                altitude,
            };
        }

        // Alpine at very high altitude
        if altitude > self.config.alpine_threshold {
            let temp = self.temperature(latitude, altitude);
            let moist = self.moisture(x, y, altitude);
            return BiomeSample {
                primary: BiomeType::Alpine,
                secondary: Some(self.classify(temp, moist)),
                blend_factor: ((altitude - self.config.alpine_threshold)
                    / (1.0 - self.config.alpine_threshold))
                    .min(1.0),
                temperature: temp,
                moisture: moist,
                altitude,
            };
        }

        let temp = self.temperature(latitude, altitude);
        let moist = self.moisture(x, y, altitude);
        let primary = self.classify(temp, moist);

        // Check for transition zone by perturbing temperature and moisture
        let tw = self.config.transition_width;
        let neighbor_biome = self.classify(temp + tw * 5.0, moist);
        let (secondary, blend) = if neighbor_biome != primary {
            // Use noise for blend factor in transition zone
            let blend_noise = self.noise_at(x * 3.7, y * 3.7);
            (Some(neighbor_biome), blend_noise * 0.5)
        } else {
            (None, 0.0)
        };

        BiomeSample {
            primary,
            secondary,
            blend_factor: blend,
            temperature: temp,
            moisture: moist,
            altitude,
        }
    }

    /// Generate a biome map for a grid.
    /// `altitudes`: 2D heightmap (row-major, values 0..1).
    /// `latitude_fn`: maps grid y to latitude (0..1).
    pub fn generate_map(
        &self,
        width: usize,
        height: usize,
        altitudes: &[f64],
        latitude_fn: impl Fn(usize) -> f64,
    ) -> Vec<BiomeSample> {
        assert!(altitudes.len() >= width * height);
        let mut result = Vec::with_capacity(width * height);
        for y in 0..height {
            let lat = latitude_fn(y);
            for x in 0..width {
                let alt = altitudes[y * width + x];
                result.push(self.sample(x as f64, y as f64, lat, alt));
            }
        }
        result
    }

    /// Blend properties of two biomes.
    pub fn blend_properties(a: &BiomeProperties, b: &BiomeProperties, t: f64) -> BiomeProperties {
        let t_clamped = t.clamp(0.0, 1.0);
        let inv = 1.0 - t_clamped;
        BiomeProperties {
            biome: if t_clamped < 0.5 { a.biome } else { b.biome },
            tree_density: a.tree_density * inv + b.tree_density * t_clamped,
            ground_color: [
                a.ground_color[0] * inv + b.ground_color[0] * t_clamped,
                a.ground_color[1] * inv + b.ground_color[1] * t_clamped,
                a.ground_color[2] * inv + b.ground_color[2] * t_clamped,
            ],
            height_variation: a.height_variation * inv + b.height_variation * t_clamped,
            base_temperature: a.base_temperature * inv + b.base_temperature * t_clamped,
            base_moisture: a.base_moisture * inv + b.base_moisture * t_clamped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_desert() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(35.0, 0.1), BiomeType::Desert);
    }

    #[test]
    fn test_classify_tropical_rainforest() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(28.0, 0.9), BiomeType::TropicalRainforest);
    }

    #[test]
    fn test_classify_tundra() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(-5.0, 0.6), BiomeType::Tundra);
    }

    #[test]
    fn test_classify_taiga() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(2.0, 0.7), BiomeType::Taiga);
    }

    #[test]
    fn test_classify_temperate_forest() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(12.0, 0.6), BiomeType::TemperateForest);
    }

    #[test]
    fn test_classify_grassland() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(12.0, 0.35), BiomeType::Grassland);
    }

    #[test]
    fn test_classify_alpine() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(-12.0, 0.3), BiomeType::Alpine);
    }

    #[test]
    fn test_classify_savanna() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        assert_eq!(bg.classify(25.0, 0.3), BiomeType::Savanna);
    }

    #[test]
    fn test_temperature_equator_sea_level() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let temp = bg.temperature(0.0, 0.0);
        assert!((temp - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_decreases_with_latitude() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let t0 = bg.temperature(0.0, 0.0);
        let t1 = bg.temperature(0.5, 0.0);
        assert!(t0 > t1);
    }

    #[test]
    fn test_temperature_decreases_with_altitude() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let t0 = bg.temperature(0.3, 0.0);
        let t1 = bg.temperature(0.3, 0.5);
        assert!(t0 > t1);
    }

    #[test]
    fn test_moisture_range() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        for i in 0..100 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            let m = bg.moisture(x, y, 0.5);
            assert!(m >= 0.0 && m <= 1.0, "moisture out of range: {m}");
        }
    }

    #[test]
    fn test_sample_ocean() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let s = bg.sample(5.0, 5.0, 0.3, 0.1);
        assert_eq!(s.primary, BiomeType::Ocean);
    }

    #[test]
    fn test_sample_beach() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let s = bg.sample(5.0, 5.0, 0.3, 0.31);
        assert_eq!(s.primary, BiomeType::Beach);
    }

    #[test]
    fn test_sample_alpine() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let s = bg.sample(5.0, 5.0, 0.3, 0.9);
        assert_eq!(s.primary, BiomeType::Alpine);
    }

    #[test]
    fn test_generate_map() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let w = 8;
        let h = 8;
        let altitudes: Vec<f64> = (0..w * h).map(|i| i as f64 / (w * h) as f64).collect();
        let map = bg.generate_map(w, h, &altitudes, |y| y as f64 / h as f64);
        assert_eq!(map.len(), w * h);
    }

    #[test]
    fn test_biome_properties() {
        let p = BiomeProperties::for_biome(BiomeType::Desert);
        assert!((p.tree_density - 0.01).abs() < 1e-6);
        assert_eq!(p.biome, BiomeType::Desert);
    }

    #[test]
    fn test_blend_properties() {
        let a = BiomeProperties::for_biome(BiomeType::Desert);
        let b = BiomeProperties::for_biome(BiomeType::TropicalRainforest);
        let blended = BiomeGenerator::blend_properties(&a, &b, 0.5);
        let expected_density = (a.tree_density + b.tree_density) / 2.0;
        assert!((blended.tree_density - expected_density).abs() < 1e-6);
    }

    #[test]
    fn test_blend_at_zero() {
        let a = BiomeProperties::for_biome(BiomeType::Desert);
        let b = BiomeProperties::for_biome(BiomeType::Taiga);
        let blended = BiomeGenerator::blend_properties(&a, &b, 0.0);
        assert!((blended.tree_density - a.tree_density).abs() < 1e-6);
        assert_eq!(blended.biome, BiomeType::Desert);
    }

    #[test]
    fn test_blend_at_one() {
        let a = BiomeProperties::for_biome(BiomeType::Desert);
        let b = BiomeProperties::for_biome(BiomeType::Taiga);
        let blended = BiomeGenerator::blend_properties(&a, &b, 1.0);
        assert!((blended.tree_density - b.tree_density).abs() < 1e-6);
        assert_eq!(blended.biome, BiomeType::Taiga);
    }

    #[test]
    fn test_biome_type_display() {
        assert_eq!(format!("{}", BiomeType::Tundra), "Tundra");
        assert_eq!(format!("{}", BiomeType::TropicalRainforest), "Tropical Rainforest");
    }

    #[test]
    fn test_config_default() {
        let cfg = BiomeConfig::default();
        assert!((cfg.sea_level - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_seed_getter() {
        let bg = BiomeGenerator::new(12345, BiomeConfig::default());
        assert_eq!(bg.seed(), 12345);
    }

    #[test]
    fn test_debug_format() {
        let bg = BiomeGenerator::new(42, BiomeConfig::default());
        let s = format!("{:?}", bg);
        assert!(s.contains("BiomeGenerator"));
    }
}
