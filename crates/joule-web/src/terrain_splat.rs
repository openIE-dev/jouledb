//! Terrain texture splatting: multi-layer blending via splat maps, triplanar
//! projection, height-based and slope-based auto-splatting, macro-variation,
//! and per-layer scale/offset.
//!
//! Pure Rust — all blending math runs on CPU. Actual texture sampling is
//! abstracted through `SplatSampler`; the rendering layer plugs in the real
//! GPU texture reads.

// ── Color helper ─────────────────────────────────────────────────

/// RGBA color in [0, 1] linear space.
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

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
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

    pub fn scale(&self, s: f32) -> Color {
        Color {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
            a: self.a,
        }
    }

    pub fn add(&self, other: &Color) -> Color {
        Color {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: (self.a + other.a).min(1.0),
        }
    }
}

// ── Vec3 (minimal) ───────────────────────────────────────────────

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
}

// ── Vec2 ─────────────────────────────────────────────────────────

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

// ── Splat map ────────────────────────────────────────────────────

/// A 2D grid of 4-channel blend weights (one per texture layer).
/// For more than 4 layers, use multiple SplatMaps.
#[derive(Debug, Clone, PartialEq)]
pub struct SplatMap {
    pub width: usize,
    pub height: usize,
    /// Row-major RGBA weights. Each pixel = 4 f32 in [0,1].
    pub data: Vec<f32>,
}

impl SplatMap {
    pub fn new(width: usize, height: usize) -> Self {
        let count = width * height * 4;
        Self {
            width,
            height,
            data: vec![0.0; count],
        }
    }

    /// Uniform first-layer map (channel 0 = 1.0 everywhere).
    pub fn uniform_first(width: usize, height: usize) -> Self {
        let count = width * height * 4;
        let mut data = vec![0.0; count];
        for i in (0..count).step_by(4) {
            data[i] = 1.0;
        }
        Self { width, height, data }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        (y * self.width + x) * 4
    }

    pub fn get_weights(&self, x: usize, y: usize) -> [f32; 4] {
        if x >= self.width || y >= self.height {
            return [0.0; 4];
        }
        let i = self.idx(x, y);
        [self.data[i], self.data[i + 1], self.data[i + 2], self.data[i + 3]]
    }

    pub fn set_weights(&mut self, x: usize, y: usize, w: [f32; 4]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let i = self.idx(x, y);
        self.data[i] = w[0];
        self.data[i + 1] = w[1];
        self.data[i + 2] = w[2];
        self.data[i + 3] = w[3];
    }

    /// Normalize weights so they sum to 1.0 at every pixel.
    pub fn normalize(&mut self) {
        for i in (0..self.data.len()).step_by(4) {
            let sum = self.data[i] + self.data[i + 1] + self.data[i + 2] + self.data[i + 3];
            if sum > 1e-10 {
                self.data[i] /= sum;
                self.data[i + 1] /= sum;
                self.data[i + 2] /= sum;
                self.data[i + 3] /= sum;
            }
        }
    }

    /// Bilinear sample of weights at fractional (fx, fy) in [0, width-1] x [0, height-1].
    pub fn sample_bilinear(&self, fx: f32, fy: f32) -> [f32; 4] {
        let x0 = fx.floor().max(0.0) as usize;
        let y0 = fy.floor().max(0.0) as usize;
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));
        let s = fx - fx.floor();
        let t = fy - fy.floor();
        let w00 = self.get_weights(x0, y0);
        let w10 = self.get_weights(x1, y0);
        let w01 = self.get_weights(x0, y1);
        let w11 = self.get_weights(x1, y1);
        let mut out = [0.0f32; 4];
        for c in 0..4 {
            let top = w00[c] * (1.0 - s) + w10[c] * s;
            let bot = w01[c] * (1.0 - s) + w11[c] * s;
            out[c] = top * (1.0 - t) + bot * t;
        }
        out
    }
}

// ── Texture layer ────────────────────────────────────────────────

/// Per-layer texture configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureLayer {
    pub name: String,
    pub base_color: Color,
    pub scale: f32,
    pub offset: Vec2,
    /// Height value for height-based blending (0..1). Higher = on top.
    pub height_blend_value: f32,
    /// Min/max slope (in radians) for auto-splatting.
    pub slope_range: (f32, f32),
}

impl TextureLayer {
    pub fn new(name: &str, base_color: Color) -> Self {
        Self {
            name: name.to_string(),
            base_color,
            scale: 1.0,
            offset: Vec2::new(0.0, 0.0),
            height_blend_value: 0.0,
            slope_range: (0.0, std::f32::consts::FRAC_PI_2),
        }
    }

    /// Compute the UV for a world position given this layer's scale and offset.
    pub fn compute_uv(&self, world_x: f32, world_z: f32) -> Vec2 {
        Vec2::new(
            world_x * self.scale + self.offset.x,
            world_z * self.scale + self.offset.y,
        )
    }

    /// Sample layer color. In a real system this reads the texture;
    /// here we modulate base_color with a procedural pattern.
    pub fn sample_color(&self, u: f32, v: f32) -> Color {
        let pattern = ((u * 10.0).sin() * (v * 10.0).sin() * 0.1 + 0.9).clamp(0.5, 1.0);
        self.base_color.scale(pattern)
    }
}

// ── Triplanar projection ─────────────────────────────────────────

/// Compute triplanar blend weights from a surface normal.
/// Steep surfaces get more X/Z projection; flat surfaces get Y projection.
pub fn triplanar_weights(normal: &Vec3, sharpness: f32) -> Vec3 {
    let ax = normal.x.abs().powf(sharpness);
    let ay = normal.y.abs().powf(sharpness);
    let az = normal.z.abs().powf(sharpness);
    let sum = ax + ay + az;
    if sum < 1e-10 {
        return Vec3::new(0.333, 0.334, 0.333);
    }
    Vec3::new(ax / sum, ay / sum, az / sum)
}

/// Sample a layer using triplanar projection.
pub fn triplanar_sample(
    layer: &TextureLayer,
    world_pos: &Vec3,
    normal: &Vec3,
    sharpness: f32,
) -> Color {
    let w = triplanar_weights(normal, sharpness);
    // X-axis projection: use (z, y) as UV
    let cx = layer.sample_color(
        world_pos.z * layer.scale + layer.offset.x,
        world_pos.y * layer.scale + layer.offset.y,
    );
    // Y-axis projection: use (x, z) as UV
    let cy = layer.sample_color(
        world_pos.x * layer.scale + layer.offset.x,
        world_pos.z * layer.scale + layer.offset.y,
    );
    // Z-axis projection: use (x, y) as UV
    let cz = layer.sample_color(
        world_pos.x * layer.scale + layer.offset.x,
        world_pos.y * layer.scale + layer.offset.y,
    );
    cx.scale(w.x).add(&cy.scale(w.y)).add(&cz.scale(w.z))
}

// ── Height-based blending ────────────────────────────────────────

/// Sharp height-based blend between two layers.
/// `depth` controls the transition width (smaller = sharper).
pub fn height_blend(weight_a: f32, height_a: f32, weight_b: f32, height_b: f32, depth: f32) -> (f32, f32) {
    let ma = weight_a + height_a;
    let mb = weight_b + height_b;
    let max_val = ma.max(mb) - depth;
    let ba = (ma - max_val).max(0.0);
    let bb = (mb - max_val).max(0.0);
    let sum = ba + bb;
    if sum < 1e-10 {
        return (0.5, 0.5);
    }
    (ba / sum, bb / sum)
}

// ── Slope-based auto-splatting ───────────────────────────────────

/// Compute the slope angle in radians from a surface normal (assumes Y-up).
pub fn slope_angle(normal: &Vec3) -> f32 {
    let ny = normal.y.abs().clamp(0.0, 1.0);
    ny.acos()
}

/// Auto-splat: given a normal, return 4-channel weights for up to 4 layers
/// based on each layer's slope range.
pub fn auto_splat_weights(normal: &Vec3, layers: &[TextureLayer]) -> [f32; 4] {
    let angle = slope_angle(normal);
    let mut weights = [0.0f32; 4];
    let count = layers.len().min(4);
    for i in 0..count {
        let (lo, hi) = layers[i].slope_range;
        if angle >= lo && angle <= hi {
            let mid = (lo + hi) * 0.5;
            let half = (hi - lo) * 0.5;
            if half > 1e-10 {
                weights[i] = 1.0 - ((angle - mid).abs() / half).min(1.0);
            } else {
                weights[i] = 1.0;
            }
        }
    }
    // Normalize
    let sum: f32 = weights.iter().sum();
    if sum > 1e-10 {
        for w in &mut weights {
            *w /= sum;
        }
    }
    weights
}

// ── Macro-variation ──────────────────────────────────────────────

/// Procedural macro-variation (breaks texture tiling at large scale).
/// Returns a factor in [1 - strength, 1 + strength].
pub fn macro_variation(world_x: f32, world_z: f32, scale: f32, strength: f32) -> f32 {
    let v = (world_x * scale * 0.3).sin() * (world_z * scale * 0.7).cos();
    1.0 + v * strength.clamp(0.0, 1.0)
}

// ── Terrain splat compositor ─────────────────────────────────────

/// Composites multiple texture layers using a splat map.
#[derive(Debug, Clone)]
pub struct TerrainSplat {
    pub layers: Vec<TextureLayer>,
    pub splat_maps: Vec<SplatMap>,
    pub triplanar_sharpness: f32,
    pub macro_scale: f32,
    pub macro_strength: f32,
}

impl TerrainSplat {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            splat_maps: Vec::new(),
            triplanar_sharpness: 4.0,
            macro_scale: 0.01,
            macro_strength: 0.15,
        }
    }

    pub fn add_layer(&mut self, layer: TextureLayer) -> usize {
        let idx = self.layers.len();
        self.layers.push(layer);
        idx
    }

    pub fn add_splat_map(&mut self, map: SplatMap) -> usize {
        let idx = self.splat_maps.len();
        self.splat_maps.push(map);
        idx
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Sample the composited terrain color at a world position with given splat UV.
    pub fn sample(
        &self,
        world_pos: &Vec3,
        normal: &Vec3,
        splat_u: f32,
        splat_v: f32,
    ) -> Color {
        if self.layers.is_empty() {
            return Color::black();
        }
        // Gather weights from first splat map
        let weights = if let Some(map) = self.splat_maps.first() {
            map.sample_bilinear(splat_u, splat_v)
        } else {
            // Fallback: uniform first layer
            [1.0, 0.0, 0.0, 0.0]
        };
        let angle = slope_angle(normal);
        let use_triplanar = angle > 0.6; // ~35 degrees
        let mut result = Color::new(0.0, 0.0, 0.0, 1.0);
        let count = self.layers.len().min(4);
        for i in 0..count {
            if weights[i] < 1e-6 {
                continue;
            }
            let layer = &self.layers[i];
            let color = if use_triplanar {
                triplanar_sample(layer, world_pos, normal, self.triplanar_sharpness)
            } else {
                let uv = layer.compute_uv(world_pos.x, world_pos.z);
                layer.sample_color(uv.x, uv.y)
            };
            let macro_mod = macro_variation(
                world_pos.x,
                world_pos.z,
                self.macro_scale,
                self.macro_strength,
            );
            let modulated = color.scale(macro_mod);
            result = result.add(&modulated.scale(weights[i]));
        }
        result
    }

    /// Blend two layers using height-based blending at given weights.
    pub fn sample_height_blended(
        &self,
        layer_a: usize,
        layer_b: usize,
        weight_a: f32,
        weight_b: f32,
        world_pos: &Vec3,
        depth: f32,
    ) -> Color {
        if layer_a >= self.layers.len() || layer_b >= self.layers.len() {
            return Color::black();
        }
        let la = &self.layers[layer_a];
        let lb = &self.layers[layer_b];
        let (ba, bb) = height_blend(
            weight_a,
            la.height_blend_value,
            weight_b,
            lb.height_blend_value,
            depth,
        );
        let uv_a = la.compute_uv(world_pos.x, world_pos.z);
        let uv_b = lb.compute_uv(world_pos.x, world_pos.z);
        let ca = la.sample_color(uv_a.x, uv_a.y);
        let cb = lb.sample_color(uv_b.x, uv_b.y);
        ca.scale(ba).add(&cb.scale(bb))
    }
}

impl Default for TerrainSplat {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn color_lerp() {
        let a = Color::black();
        let b = Color::white();
        let mid = a.lerp(&b, 0.5);
        assert!(approx(mid.r, 0.5, 1e-6));
        assert!(approx(mid.g, 0.5, 1e-6));
    }

    #[test]
    fn color_scale_and_add() {
        let c = Color::new(0.5, 0.5, 0.5, 1.0);
        let s = c.scale(2.0);
        assert!(approx(s.r, 1.0, 1e-6));
        let sum = Color::black().add(&Color::white());
        assert!(approx(sum.r, 1.0, 1e-6));
    }

    #[test]
    fn splat_map_new() {
        let m = SplatMap::new(4, 4);
        assert_eq!(m.data.len(), 64);
        let w = m.get_weights(0, 0);
        assert!(approx(w[0], 0.0, 1e-6));
    }

    #[test]
    fn splat_map_uniform_first() {
        let m = SplatMap::uniform_first(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let w = m.get_weights(x, y);
                assert!(approx(w[0], 1.0, 1e-6));
                assert!(approx(w[1], 0.0, 1e-6));
            }
        }
    }

    #[test]
    fn splat_map_set_get() {
        let mut m = SplatMap::new(4, 4);
        m.set_weights(2, 3, [0.1, 0.2, 0.3, 0.4]);
        let w = m.get_weights(2, 3);
        assert!(approx(w[0], 0.1, 1e-6));
        assert!(approx(w[3], 0.4, 1e-6));
    }

    #[test]
    fn splat_map_normalize() {
        let mut m = SplatMap::new(2, 2);
        m.set_weights(0, 0, [2.0, 2.0, 0.0, 0.0]);
        m.normalize();
        let w = m.get_weights(0, 0);
        assert!(approx(w[0], 0.5, 1e-6));
        assert!(approx(w[1], 0.5, 1e-6));
    }

    #[test]
    fn splat_map_bilinear() {
        let mut m = SplatMap::new(2, 2);
        m.set_weights(0, 0, [1.0, 0.0, 0.0, 0.0]);
        m.set_weights(1, 0, [0.0, 1.0, 0.0, 0.0]);
        m.set_weights(0, 1, [0.0, 0.0, 1.0, 0.0]);
        m.set_weights(1, 1, [0.0, 0.0, 0.0, 1.0]);
        let w = m.sample_bilinear(0.5, 0.5);
        assert!(approx(w[0], 0.25, 1e-4));
        assert!(approx(w[1], 0.25, 1e-4));
        assert!(approx(w[2], 0.25, 1e-4));
        assert!(approx(w[3], 0.25, 1e-4));
    }

    #[test]
    fn splat_map_out_of_bounds() {
        let m = SplatMap::new(2, 2);
        let w = m.get_weights(10, 10);
        assert!(approx(w[0], 0.0, 1e-6));
    }

    #[test]
    fn texture_layer_uv() {
        let layer = TextureLayer::new("grass", Color::new(0.2, 0.8, 0.1, 1.0));
        let uv = layer.compute_uv(5.0, 10.0);
        assert!(approx(uv.x, 5.0, 1e-6));
        assert!(approx(uv.y, 10.0, 1e-6));
    }

    #[test]
    fn texture_layer_uv_with_scale() {
        let mut layer = TextureLayer::new("rock", Color::white());
        layer.scale = 0.5;
        layer.offset = Vec2::new(1.0, 2.0);
        let uv = layer.compute_uv(10.0, 20.0);
        assert!(approx(uv.x, 6.0, 1e-6)); // 10*0.5+1
        assert!(approx(uv.y, 12.0, 1e-6)); // 20*0.5+2
    }

    #[test]
    fn texture_layer_sample_bounded() {
        let layer = TextureLayer::new("dirt", Color::new(0.5, 0.3, 0.1, 1.0));
        let c = layer.sample_color(1.0, 2.0);
        assert!(c.r >= 0.0 && c.r <= 1.0);
        assert!(c.g >= 0.0 && c.g <= 1.0);
    }

    #[test]
    fn triplanar_weights_flat() {
        let up = Vec3::new(0.0, 1.0, 0.0);
        let w = triplanar_weights(&up, 4.0);
        assert!(w.y > w.x && w.y > w.z);
    }

    #[test]
    fn triplanar_weights_steep() {
        let side = Vec3::new(1.0, 0.0, 0.0);
        let w = triplanar_weights(&side, 4.0);
        assert!(w.x > w.y && w.x > w.z);
    }

    #[test]
    fn triplanar_weights_sum() {
        let n = Vec3::new(0.5, 0.7, 0.3);
        let w = triplanar_weights(&n, 2.0);
        let sum = w.x + w.y + w.z;
        assert!(approx(sum, 1.0, 1e-4));
    }

    #[test]
    fn slope_angle_flat() {
        let up = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx(slope_angle(&up), 0.0, 1e-6));
    }

    #[test]
    fn slope_angle_vertical() {
        let side = Vec3::new(1.0, 0.0, 0.0);
        assert!(approx(slope_angle(&side), std::f32::consts::FRAC_PI_2, 1e-4));
    }

    #[test]
    fn auto_splat_flat_terrain() {
        let grass = {
            let mut l = TextureLayer::new("grass", Color::new(0.2, 0.8, 0.1, 1.0));
            l.slope_range = (0.0, 0.8);
            l
        };
        let rock = {
            let mut l = TextureLayer::new("rock", Color::new(0.5, 0.5, 0.5, 1.0));
            l.slope_range = (0.6, 1.571);
            l
        };
        // Nearly flat: tiny slope so angle is near 0 but inside grass center
        let normal = Vec3::new(0.05, 0.999, 0.0);
        let w = auto_splat_weights(&normal, &[grass, rock]);
        assert!(w[0] > w[1], "flat terrain should be mostly grass: w={:?}", w);
    }

    #[test]
    fn auto_splat_steep_terrain() {
        let grass = {
            let mut l = TextureLayer::new("grass", Color::new(0.2, 0.8, 0.1, 1.0));
            l.slope_range = (0.0, 0.3);
            l
        };
        let rock = {
            let mut l = TextureLayer::new("rock", Color::new(0.5, 0.5, 0.5, 1.0));
            l.slope_range = (0.3, 1.571);
            l
        };
        let normal = Vec3::new(0.8, 0.2, 0.0);
        let w = auto_splat_weights(&normal, &[grass, rock]);
        assert!(w[1] > w[0], "steep terrain should be mostly rock");
    }

    #[test]
    fn height_blend_equal() {
        let (a, b) = height_blend(0.5, 0.5, 0.5, 0.5, 0.1);
        assert!(approx(a, 0.5, 1e-4));
        assert!(approx(b, 0.5, 1e-4));
    }

    #[test]
    fn height_blend_one_dominant() {
        let (a, b) = height_blend(0.5, 1.0, 0.5, 0.0, 0.01);
        assert!(a > b, "higher height value should dominate");
    }

    #[test]
    fn macro_variation_range() {
        for x in 0..10 {
            for z in 0..10 {
                let v = macro_variation(x as f32, z as f32, 0.1, 0.3);
                assert!(v >= 0.5 && v <= 1.5, "variation {v} out of range");
            }
        }
    }

    #[test]
    fn terrain_splat_sample_single_layer() {
        let mut ts = TerrainSplat::new();
        ts.add_layer(TextureLayer::new("grass", Color::new(0.2, 0.8, 0.1, 1.0)));
        ts.add_splat_map(SplatMap::uniform_first(4, 4));
        let pos = Vec3::new(1.0, 0.0, 1.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let c = ts.sample(&pos, &normal, 0.5, 0.5);
        assert!(c.r > 0.0 || c.g > 0.0);
    }

    #[test]
    fn terrain_splat_empty() {
        let ts = TerrainSplat::new();
        let c = ts.sample(
            &Vec3::new(0.0, 0.0, 0.0),
            &Vec3::new(0.0, 1.0, 0.0),
            0.0,
            0.0,
        );
        assert!(approx(c.r, 0.0, 1e-6));
    }

    #[test]
    fn terrain_splat_height_blended() {
        let mut ts = TerrainSplat::new();
        let mut grass = TextureLayer::new("grass", Color::new(0.2, 0.8, 0.1, 1.0));
        grass.height_blend_value = 0.3;
        let mut rock = TextureLayer::new("rock", Color::new(0.5, 0.5, 0.5, 1.0));
        rock.height_blend_value = 0.7;
        ts.add_layer(grass);
        ts.add_layer(rock);
        let pos = Vec3::new(1.0, 0.0, 1.0);
        let c = ts.sample_height_blended(0, 1, 0.5, 0.5, &pos, 0.05);
        // Rock has higher height_blend_value → should contribute more
        assert!(c.r > 0.0 || c.g > 0.0);
    }

    #[test]
    fn terrain_splat_default() {
        let ts = TerrainSplat::default();
        assert_eq!(ts.layer_count(), 0);
    }

    #[test]
    fn triplanar_sample_color() {
        let layer = TextureLayer::new("test", Color::new(0.5, 0.5, 0.5, 1.0));
        let pos = Vec3::new(1.0, 2.0, 3.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let c = triplanar_sample(&layer, &pos, &normal, 4.0);
        assert!(c.r > 0.0 && c.g > 0.0 && c.b > 0.0);
    }
}
