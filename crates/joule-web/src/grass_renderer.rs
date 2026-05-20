//! Grass and vegetation rendering: per-blade generation from density maps,
//! wind animation, distance-based LOD and density fade, seasonal color
//! variation, and chunk-based culling.
//!
//! Pure Rust — all blade geometry and animation is computed on CPU.

use std::collections::HashMap;

// ── Vec3 / Vec2 / Color ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn distance_to(&self, other: &Vec3) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
    pub fn lerp(&self, other: &Color, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

// ── Season ───────────────────────────────────────────────────────

/// Seasonal variation for grass coloring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

impl Season {
    /// Base grass color for the season.
    pub fn grass_color(&self) -> Color {
        match self {
            Season::Spring => Color::new(0.3, 0.75, 0.15, 1.0),
            Season::Summer => Color::new(0.25, 0.65, 0.1, 1.0),
            Season::Autumn => Color::new(0.6, 0.55, 0.15, 1.0),
            Season::Winter => Color::new(0.45, 0.5, 0.3, 1.0),
        }
    }

    /// Tip color (slightly different from base for gradient).
    pub fn tip_color(&self) -> Color {
        match self {
            Season::Spring => Color::new(0.5, 0.9, 0.3, 1.0),
            Season::Summer => Color::new(0.55, 0.8, 0.2, 1.0),
            Season::Autumn => Color::new(0.75, 0.6, 0.1, 1.0),
            Season::Winter => Color::new(0.6, 0.6, 0.35, 1.0),
        }
    }

    /// Height multiplier (grass grows less in winter).
    pub fn height_factor(&self) -> f32 {
        match self {
            Season::Spring => 1.0,
            Season::Summer => 1.1,
            Season::Autumn => 0.85,
            Season::Winter => 0.5,
        }
    }
}

// ── Grass LOD ────────────────────────────────────────────────────

/// LOD mode for grass blades at various distances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrassLod {
    /// Full 3D blade geometry.
    FullBlade,
    /// Billboard quad facing camera.
    Billboard,
    /// Faded to terrain color (not rendered).
    TerrainFade,
}

/// LOD thresholds for grass rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassLodConfig {
    pub billboard_distance: f32,
    pub fade_distance: f32,
}

impl Default for GrassLodConfig {
    fn default() -> Self {
        Self {
            billboard_distance: 30.0,
            fade_distance: 60.0,
        }
    }
}

impl GrassLodConfig {
    pub fn select_lod(&self, distance: f32) -> GrassLod {
        if distance < self.billboard_distance {
            GrassLod::FullBlade
        } else if distance < self.fade_distance {
            GrassLod::Billboard
        } else {
            GrassLod::TerrainFade
        }
    }

    /// Alpha fade factor for smooth transition (1 = fully visible, 0 = gone).
    pub fn fade_factor(&self, distance: f32) -> f32 {
        if distance < self.billboard_distance {
            1.0
        } else if distance < self.fade_distance {
            let t = (distance - self.billboard_distance)
                / (self.fade_distance - self.billboard_distance);
            (1.0 - t).max(0.0)
        } else {
            0.0
        }
    }
}

// ── Density map ──────────────────────────────────────────────────

/// 2D grid controlling grass density. Values in [0, 1].
#[derive(Debug, Clone, PartialEq)]
pub struct DensityMap {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl DensityMap {
    pub fn new(width: usize, height: usize, default_density: f32) -> Self {
        Self {
            width,
            height,
            data: vec![default_density.clamp(0.0, 1.0); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            0.0
        }
    }

    pub fn set(&mut self, x: usize, y: usize, val: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = val.clamp(0.0, 1.0);
        }
    }

    /// Bilinear sample at fractional coordinates.
    pub fn sample(&self, fx: f32, fy: f32) -> f32 {
        let x0 = fx.floor().max(0.0) as usize;
        let y0 = fy.floor().max(0.0) as usize;
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));
        let s = fx - fx.floor();
        let t = fy - fy.floor();
        let v00 = self.get(x0, y0);
        let v10 = self.get(x1, y0);
        let v01 = self.get(x0, y1);
        let v11 = self.get(x1, y1);
        let top = v00 * (1.0 - s) + v10 * s;
        let bot = v01 * (1.0 - s) + v11 * s;
        top * (1.0 - t) + bot * t
    }
}

// ── Grass blade ──────────────────────────────────────────────────

/// A single grass blade instance.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassBlade {
    pub position: Vec3,
    pub height: f32,
    pub width: f32,
    pub bend: f32,
    pub rotation: f32,
    pub color_variation: f32,
    pub lod: GrassLod,
}

impl GrassBlade {
    pub fn new(position: Vec3, height: f32, width: f32) -> Self {
        Self {
            position,
            height,
            width,
            bend: 0.0,
            rotation: 0.0,
            color_variation: 0.0,
            lod: GrassLod::FullBlade,
        }
    }

    /// Tip position after applying bend.
    pub fn tip_position(&self) -> Vec3 {
        let bend_offset = self.bend * self.height;
        Vec3::new(
            self.position.x + bend_offset * self.rotation.cos(),
            self.position.y + self.height * (1.0 - self.bend * 0.5),
            self.position.z + bend_offset * self.rotation.sin(),
        )
    }

    /// Compute blade color from season + per-blade variation.
    pub fn compute_color(&self, season: &Season) -> Color {
        let base = season.grass_color();
        let tip = season.tip_color();
        let vary = self.color_variation * 0.15;
        Color::new(
            (base.r + vary).clamp(0.0, 1.0),
            (base.g - vary * 0.5).clamp(0.0, 1.0),
            (base.b + vary * 0.3).clamp(0.0, 1.0),
            1.0,
        ).lerp(&tip, 0.5)
    }
}

// ── Wind ─────────────────────────────────────────────────────────

/// Wind parameters for grass animation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindConfig {
    pub direction: Vec2,
    pub strength: f32,
    pub frequency: f32,
    pub gust_frequency: f32,
    pub gust_strength: f32,
}

impl Default for WindConfig {
    fn default() -> Self {
        Self {
            direction: Vec2::new(1.0, 0.0),
            strength: 0.3,
            frequency: 1.5,
            gust_frequency: 0.3,
            gust_strength: 0.5,
        }
    }
}

impl WindConfig {
    /// Compute wind displacement for a blade at given position and time.
    pub fn compute_displacement(&self, pos: &Vec3, time: f32) -> Vec2 {
        let phase = pos.x * 0.7 + pos.z * 1.3;
        let base_sway = (time * self.frequency + phase).sin() * self.strength;
        let gust = ((time * self.gust_frequency + phase * 0.5).sin() * 0.5 + 0.5)
            * self.gust_strength;
        let total = base_sway + gust;
        Vec2::new(self.direction.x * total, self.direction.y * total)
    }

    /// Compute bend amount for wind animation (0..1 range).
    pub fn compute_bend(&self, pos: &Vec3, time: f32) -> f32 {
        let disp = self.compute_displacement(pos, time);
        let mag = (disp.x * disp.x + disp.y * disp.y).sqrt();
        mag.clamp(0.0, 1.0)
    }
}

// ── Grass chunk ──────────────────────────────────────────────────

/// Coordinate of a grass chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GrassChunkCoord {
    pub cx: i32,
    pub cz: i32,
}

/// A chunk of grass blades for spatial culling.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassChunk {
    pub coord: GrassChunkCoord,
    pub blades: Vec<GrassBlade>,
    pub visible: bool,
}

impl GrassChunk {
    pub fn new(coord: GrassChunkCoord) -> Self {
        Self {
            coord,
            blades: Vec::new(),
            visible: true,
        }
    }

    pub fn blade_count(&self) -> usize {
        self.blades.len()
    }

    pub fn visible_blade_count(&self) -> usize {
        if self.visible {
            self.blades.len()
        } else {
            0
        }
    }
}

// ── Grass renderer ───────────────────────────────────────────────

/// Configuration for grass generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassConfig {
    pub blades_per_unit: f32,
    pub base_height: f32,
    pub height_variation: f32,
    pub base_width: f32,
    pub width_variation: f32,
    pub chunk_size: f32,
    pub season: Season,
    pub lod_config: GrassLodConfig,
    pub wind: WindConfig,
}

impl Default for GrassConfig {
    fn default() -> Self {
        Self {
            blades_per_unit: 4.0,
            base_height: 0.5,
            height_variation: 0.2,
            base_width: 0.03,
            width_variation: 0.01,
            chunk_size: 16.0,
            season: Season::Summer,
            lod_config: GrassLodConfig::default(),
            wind: WindConfig::default(),
        }
    }
}

/// Main grass renderer.
#[derive(Debug, Clone)]
pub struct GrassRenderer {
    pub config: GrassConfig,
    chunks: HashMap<(i32, i32), GrassChunk>,
}

impl GrassRenderer {
    pub fn new(config: GrassConfig) -> Self {
        Self {
            config,
            chunks: HashMap::new(),
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn total_blade_count(&self) -> usize {
        self.chunks.values().map(|c| c.blade_count()).sum()
    }

    pub fn visible_blade_count(&self) -> usize {
        self.chunks.values().map(|c| c.visible_blade_count()).sum()
    }

    /// Generate grass blades for a rectangular area using a density map.
    pub fn generate_area(
        &mut self,
        min_x: f32,
        min_z: f32,
        max_x: f32,
        max_z: f32,
        density_map: &DensityMap,
        terrain_height: &dyn Fn(f32, f32) -> f32,
    ) {
        let spacing = 1.0 / self.config.blades_per_unit.max(0.1);
        let mut x = min_x;
        while x < max_x {
            let mut z = min_z;
            while z < max_z {
                let dm_u = (x - min_x) / (max_x - min_x).max(1e-6)
                    * (density_map.width as f32 - 1.0);
                let dm_v = (z - min_z) / (max_z - min_z).max(1e-6)
                    * (density_map.height as f32 - 1.0);
                let density = density_map.sample(dm_u, dm_v);

                // Pseudo-random hash for determinism
                let hash = pseudo_hash(x, z);
                if hash < density {
                    let jitter_x = fract(hash * 127.1) * spacing * 0.5;
                    let jitter_z = fract(hash * 311.7) * spacing * 0.5;
                    let bx = x + jitter_x;
                    let bz = z + jitter_z;
                    let by = terrain_height(bx, bz);

                    let h_var = (fract(hash * 43.7) - 0.5) * 2.0
                        * self.config.height_variation;
                    let w_var = (fract(hash * 97.3) - 0.5) * 2.0
                        * self.config.width_variation;

                    let mut blade = GrassBlade::new(
                        Vec3::new(bx, by, bz),
                        (self.config.base_height + h_var)
                            * self.config.season.height_factor(),
                        (self.config.base_width + w_var).max(0.005),
                    );
                    blade.rotation = fract(hash * 73.9) * std::f32::consts::TAU;
                    blade.color_variation = fract(hash * 157.3);

                    let cx = (bx / self.config.chunk_size).floor() as i32;
                    let cz = (bz / self.config.chunk_size).floor() as i32;
                    let chunk = self.chunks
                        .entry((cx, cz))
                        .or_insert_with(|| GrassChunk::new(GrassChunkCoord { cx, cz }));
                    chunk.blades.push(blade);
                }
                z += spacing;
            }
            x += spacing;
        }
    }

    /// Update LOD and visibility for all blades based on camera position.
    pub fn update_lod(&mut self, camera: &Vec3) {
        let cfg = self.config.lod_config;
        let keys: Vec<(i32, i32)> = self.chunks.keys().copied().collect();
        for key in keys {
            if let Some(chunk) = self.chunks.get_mut(&key) {
                // Quick chunk-level visibility test
                let chunk_center = Vec3::new(
                    (chunk.coord.cx as f32 + 0.5) * self.config.chunk_size,
                    camera.y,
                    (chunk.coord.cz as f32 + 0.5) * self.config.chunk_size,
                );
                let chunk_dist = camera.distance_to(&chunk_center);
                if chunk_dist > cfg.fade_distance + self.config.chunk_size {
                    chunk.visible = false;
                    continue;
                }
                chunk.visible = true;
                for blade in &mut chunk.blades {
                    let dist = camera.distance_to(&blade.position);
                    blade.lod = cfg.select_lod(dist);
                }
            }
        }
    }

    /// Apply wind animation to all visible blades.
    pub fn animate_wind(&mut self, time: f32) {
        let wind = self.config.wind;
        let keys: Vec<(i32, i32)> = self.chunks.keys().copied().collect();
        for key in keys {
            if let Some(chunk) = self.chunks.get_mut(&key) {
                if !chunk.visible {
                    continue;
                }
                for blade in &mut chunk.blades {
                    blade.bend = wind.compute_bend(&blade.position, time);
                }
            }
        }
    }

    /// Get sorted chunk coords for deterministic iteration.
    pub fn chunk_coords_sorted(&self) -> Vec<GrassChunkCoord> {
        let mut coords: Vec<GrassChunkCoord> = self
            .chunks
            .values()
            .map(|c| c.coord)
            .collect();
        coords.sort_by(|a, b| a.cx.cmp(&b.cx).then(a.cz.cmp(&b.cz)));
        coords
    }

    /// Count blades at each LOD level.
    pub fn lod_statistics(&self) -> (usize, usize, usize) {
        let mut full = 0usize;
        let mut billboard = 0usize;
        let mut faded = 0usize;
        for chunk in self.chunks.values() {
            if !chunk.visible {
                faded += chunk.blades.len();
                continue;
            }
            for blade in &chunk.blades {
                match blade.lod {
                    GrassLod::FullBlade => full += 1,
                    GrassLod::Billboard => billboard += 1,
                    GrassLod::TerrainFade => faded += 1,
                }
            }
        }
        (full, billboard, faded)
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn pseudo_hash(x: f32, z: f32) -> f32 {
    let v = (x * 12.9898 + z * 78.233).sin() * 43758.5453;
    fract(v)
}

fn fract(v: f32) -> f32 {
    v - v.floor()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn vec3_distance() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!(approx(a.distance_to(&b), 5.0, 1e-5));
    }

    #[test]
    fn color_lerp_midpoint() {
        let a = Color::new(0.0, 0.0, 0.0, 1.0);
        let b = Color::new(1.0, 1.0, 1.0, 1.0);
        let mid = a.lerp(&b, 0.5);
        assert!(approx(mid.r, 0.5, 1e-6));
    }

    #[test]
    fn season_colors_differ() {
        let sp = Season::Spring.grass_color();
        let au = Season::Autumn.grass_color();
        assert!((sp.r - au.r).abs() > 0.01 || (sp.g - au.g).abs() > 0.01);
    }

    #[test]
    fn season_height_factors() {
        assert!(Season::Summer.height_factor() > Season::Winter.height_factor());
    }

    #[test]
    fn grass_lod_selection() {
        let cfg = GrassLodConfig::default();
        assert_eq!(cfg.select_lod(10.0), GrassLod::FullBlade);
        assert_eq!(cfg.select_lod(45.0), GrassLod::Billboard);
        assert_eq!(cfg.select_lod(100.0), GrassLod::TerrainFade);
    }

    #[test]
    fn grass_lod_fade_factor() {
        let cfg = GrassLodConfig::default();
        assert!(approx(cfg.fade_factor(0.0), 1.0, 1e-6));
        assert!(approx(cfg.fade_factor(100.0), 0.0, 1e-6));
        let mid = cfg.fade_factor(45.0);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn density_map_basic() {
        let dm = DensityMap::new(4, 4, 0.8);
        assert!(approx(dm.get(0, 0), 0.8, 1e-6));
        assert!(approx(dm.get(3, 3), 0.8, 1e-6));
    }

    #[test]
    fn density_map_set_get() {
        let mut dm = DensityMap::new(4, 4, 0.0);
        dm.set(2, 1, 0.75);
        assert!(approx(dm.get(2, 1), 0.75, 1e-6));
    }

    #[test]
    fn density_map_clamp() {
        let mut dm = DensityMap::new(2, 2, 0.0);
        dm.set(0, 0, 1.5);
        assert!(approx(dm.get(0, 0), 1.0, 1e-6));
        dm.set(0, 0, -0.5);
        assert!(approx(dm.get(0, 0), 0.0, 1e-6));
    }

    #[test]
    fn density_map_bilinear() {
        let mut dm = DensityMap::new(2, 2, 0.0);
        dm.set(0, 0, 0.0);
        dm.set(1, 0, 1.0);
        dm.set(0, 1, 1.0);
        dm.set(1, 1, 0.0);
        let mid = dm.sample(0.5, 0.5);
        assert!(approx(mid, 0.5, 1e-4));
    }

    #[test]
    fn grass_blade_tip_position() {
        let blade = GrassBlade::new(Vec3::new(0.0, 0.0, 0.0), 1.0, 0.03);
        let tip = blade.tip_position();
        assert!(approx(tip.y, 1.0, 1e-4));
    }

    #[test]
    fn grass_blade_bent_tip() {
        let mut blade = GrassBlade::new(Vec3::new(0.0, 0.0, 0.0), 1.0, 0.03);
        blade.bend = 0.5;
        blade.rotation = 0.0;
        let tip = blade.tip_position();
        assert!(tip.y < 1.0, "bent blade should be shorter vertically");
        assert!(tip.x > 0.0, "bent blade tip should offset in X");
    }

    #[test]
    fn grass_blade_color_season() {
        let blade = GrassBlade::new(Vec3::new(0.0, 0.0, 0.0), 1.0, 0.03);
        let spring_color = blade.compute_color(&Season::Spring);
        let autumn_color = blade.compute_color(&Season::Autumn);
        // Different seasons should produce different colors
        assert!(
            (spring_color.r - autumn_color.r).abs() > 0.01
                || (spring_color.g - autumn_color.g).abs() > 0.01
        );
    }

    #[test]
    fn wind_displacement_varies_with_time() {
        let wind = WindConfig::default();
        let pos = Vec3::new(5.0, 0.0, 5.0);
        let d1 = wind.compute_displacement(&pos, 0.0);
        let d2 = wind.compute_displacement(&pos, 1.0);
        assert!(
            (d1.x - d2.x).abs() > 1e-6 || (d1.y - d2.y).abs() > 1e-6,
            "wind should change over time"
        );
    }

    #[test]
    fn wind_bend_bounded() {
        let wind = WindConfig::default();
        for t in 0..100 {
            let b = wind.compute_bend(&Vec3::new(0.0, 0.0, 0.0), t as f32 * 0.1);
            assert!(b >= 0.0 && b <= 1.0);
        }
    }

    #[test]
    fn grass_chunk_new() {
        let chunk = GrassChunk::new(GrassChunkCoord { cx: 0, cz: 0 });
        assert_eq!(chunk.blade_count(), 0);
        assert!(chunk.visible);
    }

    #[test]
    fn grass_renderer_generate() {
        let cfg = GrassConfig {
            blades_per_unit: 2.0,
            chunk_size: 8.0,
            ..GrassConfig::default()
        };
        let mut renderer = GrassRenderer::new(cfg);
        let dm = DensityMap::new(4, 4, 1.0);
        renderer.generate_area(0.0, 0.0, 4.0, 4.0, &dm, &|_x, _z| 0.0);
        assert!(renderer.total_blade_count() > 0);
        assert!(renderer.chunk_count() > 0);
    }

    #[test]
    fn grass_renderer_lod_update() {
        let cfg = GrassConfig {
            blades_per_unit: 2.0,
            chunk_size: 8.0,
            ..GrassConfig::default()
        };
        let mut renderer = GrassRenderer::new(cfg);
        let dm = DensityMap::new(4, 4, 1.0);
        renderer.generate_area(0.0, 0.0, 8.0, 8.0, &dm, &|_x, _z| 0.0);
        let camera = Vec3::new(0.0, 5.0, 0.0);
        renderer.update_lod(&camera);
        let (full, _bb, _fade) = renderer.lod_statistics();
        assert!(full > 0, "close blades should be FullBlade");
    }

    #[test]
    fn grass_renderer_wind_animation() {
        let cfg = GrassConfig {
            blades_per_unit: 2.0,
            chunk_size: 8.0,
            ..GrassConfig::default()
        };
        let mut renderer = GrassRenderer::new(cfg);
        let dm = DensityMap::new(2, 2, 1.0);
        renderer.generate_area(0.0, 0.0, 2.0, 2.0, &dm, &|_x, _z| 0.0);
        renderer.animate_wind(1.5);
        // At least some blades should have non-zero bend
        let has_bent = renderer.chunks.values().any(|c| c.blades.iter().any(|b| b.bend > 0.0));
        assert!(has_bent);
    }

    #[test]
    fn grass_renderer_zero_density() {
        let cfg = GrassConfig::default();
        let mut renderer = GrassRenderer::new(cfg);
        let dm = DensityMap::new(4, 4, 0.0);
        renderer.generate_area(0.0, 0.0, 4.0, 4.0, &dm, &|_x, _z| 0.0);
        assert_eq!(renderer.total_blade_count(), 0);
    }

    #[test]
    fn grass_renderer_visible_count() {
        let cfg = GrassConfig {
            blades_per_unit: 2.0,
            chunk_size: 8.0,
            ..GrassConfig::default()
        };
        let mut renderer = GrassRenderer::new(cfg);
        let dm = DensityMap::new(4, 4, 1.0);
        renderer.generate_area(0.0, 0.0, 4.0, 4.0, &dm, &|_x, _z| 0.0);
        let total = renderer.total_blade_count();
        assert_eq!(renderer.visible_blade_count(), total);
    }

    #[test]
    fn grass_renderer_sorted_coords() {
        let cfg = GrassConfig {
            blades_per_unit: 2.0,
            chunk_size: 4.0,
            ..GrassConfig::default()
        };
        let mut renderer = GrassRenderer::new(cfg);
        let dm = DensityMap::new(4, 4, 1.0);
        renderer.generate_area(0.0, 0.0, 12.0, 12.0, &dm, &|_x, _z| 0.0);
        let coords = renderer.chunk_coords_sorted();
        for w in coords.windows(2) {
            let a = w[0];
            let b = w[1];
            assert!(
                (a.cx, a.cz) <= (b.cx, b.cz),
                "coords should be sorted"
            );
        }
    }

    #[test]
    fn pseudo_hash_deterministic() {
        let a = pseudo_hash(1.5, 2.7);
        let b = pseudo_hash(1.5, 2.7);
        assert!(approx(a, b, 1e-10));
    }

    #[test]
    fn fract_function() {
        assert!(approx(fract(3.7), 0.7, 1e-5));
        assert!(approx(fract(-0.3), 0.7, 1e-5));
    }
}
